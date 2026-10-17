#include "v200/execution/owner_execution.hpp"

#if CUDA_VERSION < 12000
#error "V200 execution requires authentic CUDA 12.x Driver API headers"
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace v200::execution {
namespace {

struct EventRef {
  std::uint64_t slot{};
  std::uint64_t generation{};

  friend bool operator<(const EventRef& a, const EventRef& b) noexcept {
    return a.slot < b.slot || (a.slot == b.slot && a.generation < b.generation);
  }
  friend bool operator==(const EventRef& a, const EventRef& b) noexcept {
    return a.slot == b.slot && a.generation == b.generation;
  }
};

struct OwnerKey {
  LogicalDeviceId device{};
  OwnerId owner{};
  friend bool operator<(const OwnerKey& a, const OwnerKey& b) noexcept {
    return a.device < b.device || (a.device == b.device && a.owner < b.owner);
  }
};

struct StreamKey {
  LogicalDeviceId device{};
  LogicalStreamId stream{};
  friend bool operator<(const StreamKey& a, const StreamKey& b) noexcept {
    return a.device < b.device || (a.device == b.device && a.stream < b.stream);
  }
};

Status driver_error(CUresult result) noexcept {
  return {ErrorCode::driver_failure, result};
}

}  // namespace

class OwnerExecution::Impl {
 public:
  explicit Impl(DriverDispatch& driver) : driver_(driver) {}

  struct OwnerRecord {
    CUcontext context{};
  };

  struct EventRecord {
    EventRef ref{};
    LogicalDeviceId device{};
    LogicalStreamId source_stream{};
    OwnerId owner{};
    CUcontext context{};
    CUevent handle{};
    std::uint64_t sequence{};
    std::size_t token_refs{};
    std::size_t head_refs{};
    bool completed{};
    bool captured{};
    CUresult record_result{CUDA_SUCCESS};
  };

  struct TokenRecord {
    std::uint64_t generation{};
    std::size_t references{1};
    LogicalDeviceId device{};
    LogicalStreamId source_stream{};  // zero means a device-wide snapshot
    std::vector<EventRef> events;
  };

  struct StreamOwner {
    CUstream native_stream{};
    bool dirty{};
  };

  struct StreamRecord {
    LogicalDeviceId device{};
    std::uint64_t sequence{};
    bool poisoned{};
    Status poison{};
    bool native_only{true};
    std::map<OwnerId, StreamOwner> owners;
    std::map<OwnerId, EventRef> heads;
  };

  struct DeviceRecord {
    bool poisoned{};
    Status poison{};
  };

  Status register_owner(LogicalDeviceId device, OwnerId owner,
                       CUcontext context) noexcept {
    if (context == nullptr || device == 0) return {ErrorCode::invalid_argument};
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    const OwnerKey key{device, owner};
    if (owners_.find(key) != owners_.end()) return {ErrorCode::duplicate_binding};
    owners_.emplace(key, OwnerRecord{context});
    devices_.try_emplace(device, DeviceRecord{});
    return Status::success();
  }

  Status register_stream(LogicalDeviceId device, LogicalStreamId stream,
                         OwnerId owner, CUstream native_stream) noexcept {
    if (device == 0 || stream == 0) return {ErrorCode::invalid_argument};
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    auto owner_it = owners_.find({device, owner});
    if (owner_it == owners_.end()) return {ErrorCode::unknown_owner};
    auto& record = streams_[{device, stream}];
    if (record.device != 0 && record.device != device) {
      return {ErrorCode::duplicate_binding};
    }
    record.device = device;
    if (record.owners.find(owner) != record.owners.end()) {
      return {ErrorCode::duplicate_binding};
    }
    record.owners.emplace(owner, StreamOwner{native_stream, false});
    devices_.try_emplace(device, DeviceRecord{});
    return Status::success();
  }

  Status unregister_stream(LogicalDeviceId device, LogicalStreamId stream,
                           OwnerId owner) noexcept {
    auto current = current_token(device, stream);
    if (!current) return current.status;
    const Status waited = wait(current.value);
    const Status released = release(current.value);
    if (!waited) return waited;
    if (!released) return released;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = streams_.find({device, stream});
    if (it == streams_.end()) return {ErrorCode::unknown_stream};
    auto owner_it = it->second.owners.find(owner);
    if (owner_it == it->second.owners.end()) return {ErrorCode::unknown_owner};
    auto head_it = it->second.heads.find(owner);
    if (head_it != it->second.heads.end()) {
      EventRecord* event = find_event_locked(head_it->second);
      if (event != nullptr && event->head_refs != 0) --event->head_refs;
      it->second.heads.erase(head_it);
    }
    it->second.owners.erase(owner_it);
    if (it->second.owners.empty()) streams_.erase(it);
    return collect_locked();
  }

  Status observe_native_submission(LogicalDeviceId device,
                                   LogicalStreamId stream, OwnerId owner,
                                   CUstream native_stream,
                                   CUresult forwarded_result) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    auto it = streams_.find({device, stream});
    if (it == streams_.end()) return {ErrorCode::unknown_stream};
    auto stream_owner = it->second.owners.find(owner);
    if (stream_owner == it->second.owners.end()) return {ErrorCode::unknown_owner};
    if (stream_owner->second.native_stream != native_stream) {
      return {ErrorCode::invalid_argument};
    }
    StreamRecord& record = it->second;
    if (!record.native_only) return {ErrorCode::unavailable};
    // A passthrough operation cannot insert the peer waits needed after work
    // has already been queued. Refuse to pretend it completed a joined prefix.
    for (const auto& head : record.heads) {
      if (head.first != owner) return {ErrorCode::unavailable};
    }
    for (const auto& dirty : record.owners) {
      if (dirty.second.dirty && dirty.first != owner) {
        return {ErrorCode::unavailable};
      }
    }
    stream_owner->second.dirty = true;
    if (forwarded_result != CUDA_SUCCESS) {
      const Status failure = driver_error(forwarded_result);
      poison_locked(record, failure);
      poison_device_locked(device, failure);
      return failure;
    }
    return Status::success();
  }

  Status forward_native_submission(LogicalDeviceId device,
                                   LogicalStreamId stream, OwnerId owner,
                                   CUstream native_stream,
                                   const EnqueueCallback& forward) noexcept {
    if (!forward) return {ErrorCode::invalid_argument};
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    auto it = streams_.find({device, stream});
    if (it == streams_.end()) return {ErrorCode::unknown_stream};
    auto stream_owner = it->second.owners.find(owner);
    if (stream_owner == it->second.owners.end()) return {ErrorCode::unknown_owner};
    if (stream_owner->second.native_stream != native_stream) {
      return {ErrorCode::invalid_argument};
    }
    StreamRecord& record = it->second;
    if (!record.native_only) return {ErrorCode::unavailable};
    for (const auto& head : record.heads) {
      if (head.first != owner) return {ErrorCode::unavailable};
    }
    for (const auto& other : record.owners) {
      if (other.second.dirty && other.first != owner) {
        return {ErrorCode::unavailable};
      }
    }
    const Status device_poison = device_poison_locked(device);
    if (!device_poison.ok()) return device_poison;
    if (record.poisoned) return record.poison;
    const auto owner_record = owners_.find({device, owner});
    if (owner_record == owners_.end()) return {ErrorCode::unknown_owner};

    CUresult result = driver_.ctx_push_current(owner_record->second.context);
    if (result != CUDA_SUCCESS) return driver_error(result);
    try {
      result = forward(native_stream);
    } catch (...) {
      result = CUDA_ERROR_UNKNOWN;
    }
    // Publish dirty state before releasing the serialization lock so no peer
    // submit can pass this operation without sealing its full prefix.
    stream_owner->second.dirty = true;
    Status status = result == CUDA_SUCCESS ? Status::success() : driver_error(result);
    // CU_STREAM_PER_THREAD is host-thread-local. A later token snapshot may
    // run on another host thread, where recording to the same sentinel would
    // capture that thread's PTDS instead. Seal it here, on the dispatching
    // thread, before allowing the caller to hand the stream/token to another
    // thread. The context is already current and the adapter lock is held.
    if (native_stream == CU_STREAM_PER_THREAD) {
      const Status captured = record_event_current_locked(
          device, stream, owner, record, owner_record->second.context,
          native_stream);
      if (captured.ok()) {
        stream_owner->second.dirty = false;
      } else if (status.ok()) {
        status = captured;
      }
    }
    const CUresult popped = pop_context(owner_record->second.context);
    if (popped != CUDA_SUCCESS) {
      status = {ErrorCode::context_restore_failed, popped};
    }
    if (!status.ok()) {
      poison_locked(record, status);
      poison_device_locked(device, status);
    }
    return status;
  }

  SubmissionResult submit(LogicalDeviceId device, LogicalStreamId stream,
                         OwnerId owner, const CompletionToken* dependencies,
                         std::size_t dependency_count,
                         const EnqueueCallback& enqueue) noexcept {
    if (dependency_count != 0 && dependencies == nullptr) {
      return {{ErrorCode::invalid_argument}, {}};
    }
    if (!enqueue) return {{ErrorCode::invalid_argument}, {}};

    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {{ErrorCode::closed}, {}};
    auto stream_it = streams_.find({device, stream});
    if (stream_it == streams_.end()) return {{ErrorCode::unknown_stream}, {}};
    auto owner_it = owners_.find({device, owner});
    if (owner_it == owners_.end()) return {{ErrorCode::unknown_owner}, {}};
    StreamRecord& record = stream_it->second;
    auto stream_owner_it = record.owners.find(owner);
    if (stream_owner_it == record.owners.end()) return {{ErrorCode::unknown_owner}, {}};
    Status poison = device_poison_locked(device);
    if (!poison.ok()) return {poison, {}};
    if (record.poisoned) return {record.poison, {}};

    // Native anchor operations may have been forwarded before this owner was
    // asked to participate. Seal every such prefix before deriving the peer
    // waits below, otherwise the first owner-to-owner handoff could overtake
    // work that has not yet acquired a private event.
    Status sealed = seal_dirty_owners_locked(device, stream, record);
    if (!sealed.ok()) {
      poison_locked(record, sealed);
      poison_device_locked(device, sealed);
      return {sealed, {}};
    }

    std::vector<EventRef> waits;
    for (const auto& head : record.heads) {
      if (head.first != owner) waits.push_back(head.second);
    }
    for (std::size_t i = 0; i < dependency_count; ++i) {
      TokenRecord* dependency = find_token_locked(dependencies[i]);
      if (dependency == nullptr) return {{ErrorCode::stale_token}, {}};
      if (contains_stream_prefix_locked(*dependency, device, stream)) {
        return {{ErrorCode::dependency_cycle}, {}};
      }
      waits.insert(waits.end(), dependency->events.begin(), dependency->events.end());
    }
    deduplicate(waits);

    CUresult result = driver_.ctx_push_current(owner_it->second.context);
    if (result != CUDA_SUCCESS) return {driver_error(result), {}};
    Status work_status = Status::success();
    for (const EventRef wait_ref : waits) {
      EventRecord* event = find_event_locked(wait_ref);
      if (event == nullptr) {
        work_status = {ErrorCode::stale_token};
        break;
      }
      result = driver_.stream_wait_event(stream_owner_it->second.native_stream,
                                         event->handle, 0);
      if (result != CUDA_SUCCESS) {
        work_status = driver_error(result);
        break;
      }
    }
    if (work_status.ok()) {
      try {
        result = enqueue(stream_owner_it->second.native_stream);
      } catch (...) {
        result = CUDA_ERROR_UNKNOWN;
      }
      if (result != CUDA_SUCCESS) work_status = driver_error(result);
    }
    // Even after a synchronous enqueue error, record a tail event when possible
    // so already-issued work remains observable and drainable.
    stream_owner_it->second.dirty = true;
    record.native_only = false;
    CompletionToken completion{};
    const Status record_status = seal_owner_locked(device, stream, owner, record,
                                                   owner_it->second.context,
                                                   stream_owner_it->second);
    if (record_status.ok()) {
      auto token = snapshot_stream_locked(device, stream, record);
      if (token) completion = token.value;
      else work_status = token.status;
    } else {
      work_status = record_status;
    }
    const CUresult popped_context = pop_context(owner_it->second.context);
    if (popped_context != CUDA_SUCCESS && work_status.ok()) {
      work_status = {ErrorCode::context_restore_failed, popped_context};
    }
    if (!work_status.ok()) {
      poison_locked(record, work_status);
      poison_device_locked(device, work_status);
    }
    return {work_status, completion};
  }

  SubmissionResult wait_on_stream(LogicalDeviceId device,
                                  LogicalStreamId stream, OwnerId owner,
                                  CompletionToken dependency_token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {{ErrorCode::closed}, {}};
    auto stream_it = streams_.find({device, stream});
    if (stream_it == streams_.end()) return {{ErrorCode::unknown_stream}, {}};
    auto owner_it = owners_.find({device, owner});
    if (owner_it == owners_.end()) return {{ErrorCode::unknown_owner}, {}};
    StreamRecord& record = stream_it->second;
    auto stream_owner_it = record.owners.find(owner);
    if (stream_owner_it == record.owners.end()) return {{ErrorCode::unknown_owner}, {}};
    Status poison = device_poison_locked(device);
    if (!poison.ok()) return {poison, {}};
    if (record.poisoned) return {record.poison, {}};
    TokenRecord* dependency = find_token_locked(dependency_token);
    if (dependency == nullptr) return {{ErrorCode::stale_token}, {}};
    if (contains_stream_prefix_locked(*dependency, device, stream)) {
      return {{ErrorCode::dependency_cycle}, {}};
    }

    Status sealed = seal_dirty_owners_locked(device, stream, record);
    if (!sealed.ok()) {
      poison_locked(record, sealed);
      poison_device_locked(device, sealed);
      return {sealed, {}};
    }

    std::vector<EventRef> waits = dependency->events;
    for (const auto& head : record.heads) {
      if (head.first != owner) waits.push_back(head.second);
    }
    deduplicate(waits);
    CUresult result = driver_.ctx_push_current(owner_it->second.context);
    if (result != CUDA_SUCCESS) return {driver_error(result), {}};
    Status status = Status::success();
    for (const EventRef wait_ref : waits) {
      EventRecord* event = find_event_locked(wait_ref);
      if (event == nullptr) {
        status = {ErrorCode::stale_token};
        break;
      }
      result = driver_.stream_wait_event(stream_owner_it->second.native_stream,
                                         event->handle, 0);
      if (result != CUDA_SUCCESS) {
        status = driver_error(result);
        break;
      }
    }
    stream_owner_it->second.dirty = true;
    record.native_only = false;
    CompletionToken completion{};
    if (status.ok()) {
      status = seal_owner_locked(device, stream, owner, record,
                                 owner_it->second.context,
                                 stream_owner_it->second);
      if (status.ok()) {
        auto token = snapshot_stream_locked(device, stream, record);
        if (token) completion = token.value;
        else status = token.status;
      }
    }
    const CUresult popped_context = pop_context(owner_it->second.context);
    if (popped_context != CUDA_SUCCESS && status.ok()) {
      status = {ErrorCode::context_restore_failed, popped_context};
    }
    if (!status.ok()) {
      poison_locked(record, status);
      poison_device_locked(device, status);
    }
    return {status, completion};
  }

  Result<CompletionToken> current_token(LogicalDeviceId device,
                                       LogicalStreamId stream) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {{ErrorCode::closed}, {}};
    auto it = streams_.find({device, stream});
    if (it == streams_.end()) return {{ErrorCode::unknown_stream}, {}};
    for (auto& owner_stream : it->second.owners) {
      if (!owner_stream.second.dirty) continue;
      auto owner_it = owners_.find({device, owner_stream.first});
      if (owner_it == owners_.end()) return {{ErrorCode::unknown_owner}, {}};
      const Status status = seal_owner_locked(device, stream,
                                             owner_stream.first, it->second,
                                             owner_it->second.context,
                                             owner_stream.second);
      if (!status.ok()) {
        poison_locked(it->second, status);
        poison_device_locked(device, status);
        return {status, {}};
      }
    }
    return snapshot_stream_locked(device, stream, it->second);
  }

  Result<CompletionToken> current_device_token(LogicalDeviceId device) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {{ErrorCode::closed}, {}};
    if (devices_.find(device) == devices_.end()) return {{ErrorCode::unknown_owner}, {}};
    for (auto& stream_entry : streams_) {
      if (stream_entry.first.device != device) continue;
      StreamRecord& stream = stream_entry.second;
      for (auto& owner_stream : stream.owners) {
        if (!owner_stream.second.dirty) continue;
        auto owner_it = owners_.find({device, owner_stream.first});
        if (owner_it == owners_.end()) return {{ErrorCode::unknown_owner}, {}};
        const Status status = seal_owner_locked(device, stream_entry.first.stream,
                                               owner_stream.first, stream,
                                               owner_it->second.context,
                                               owner_stream.second);
        if (!status.ok()) {
          poison_locked(stream, status);
          poison_device_locked(device, status);
          return {status, {}};
        }
      }
    }
    std::vector<EventRef> events;
    for (const auto& stream_entry : streams_) {
      if (stream_entry.first.device != device) continue;
      for (const auto& head : stream_entry.second.heads) events.push_back(head.second);
    }
    deduplicate(events);
    return create_token_locked(device, 0, std::move(events));
  }

  Status collect() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    return collect_locked();
  }

  Status shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return Status::success();
    if (!tokens_.empty()) return {ErrorCode::resource_busy};
    // A successful shutdown must account for work that was forwarded but has
    // not yet been represented by a private event. If capture and its safe
    // synchronous fallback both fail, preserve the stream records and fail.
    for (auto& stream_entry : streams_) {
      StreamRecord& stream = stream_entry.second;
      const Status sealed = seal_dirty_owners_locked(
          stream_entry.first.device, stream_entry.first.stream, stream);
      if (!sealed.ok()) {
        poison_locked(stream, sealed);
        poison_device_locked(stream_entry.first.device, sealed);
        return sealed;
      }
    }
    for (auto& stream_entry : streams_) {
      for (const auto& head : stream_entry.second.heads) {
        EventRecord* event = find_event_locked(head.second);
        if (event != nullptr && event->head_refs != 0) --event->head_refs;
      }
      stream_entry.second.heads.clear();
    }
    Status first_error = Status::success();
    std::vector<EventRef> refs;
    refs.reserve(events_.size());
    for (const auto& event_entry : events_) refs.push_back(event_entry.second.ref);
    for (const EventRef ref : refs) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) continue;
      Status status = synchronize_event_locked(*event);
      if (status.ok()) status = destroy_event_locked(*event);
      if (!status.ok() && first_error.ok()) first_error = status;
    }
    if (!first_error.ok()) return first_error;
    streams_.clear();
    owners_.clear();
    devices_.clear();
    closed_ = true;
    return Status::success();
  }

  void best_effort_shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    // No destructor can report a failed drain. Keep every event allocated if
    // a dirty native stream cannot be synchronously proven idle.
    for (auto& stream_entry : streams_) {
      for (auto& owner_stream : stream_entry.second.owners) {
        if (!owner_stream.second.dirty) continue;
        const auto owner = owners_.find({stream_entry.first.device,
                                         owner_stream.first});
        if (owner == owners_.end()) return;
        CUresult result = driver_.ctx_push_current(owner->second.context);
        if (result != CUDA_SUCCESS) return;
        result = driver_.stream_synchronize(owner_stream.second.native_stream);
        const CUresult popped = pop_context(owner->second.context);
        if (result != CUDA_SUCCESS || popped != CUDA_SUCCESS) return;
        mark_stream_events_complete_locked(stream_entry.first.device,
                                           stream_entry.first.stream,
                                           owner_stream.first);
        owner_stream.second.dirty = false;
      }
    }
    for (auto& entry : tokens_) {
      for (const EventRef ref : entry.second.events) {
        EventRecord* event = find_event_locked(ref);
        if (event != nullptr && event->token_refs != 0) --event->token_refs;
      }
    }
    tokens_.clear();
    for (auto& stream_entry : streams_) {
      for (const auto& head : stream_entry.second.heads) {
        EventRecord* event = find_event_locked(head.second);
        if (event != nullptr && event->head_refs != 0) --event->head_refs;
      }
      stream_entry.second.heads.clear();
    }
    std::vector<EventRef> refs;
    refs.reserve(events_.size());
    for (const auto& event : events_) refs.push_back(event.second.ref);
    for (const EventRef ref : refs) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) continue;
      if (synchronize_event_locked(*event).ok()) (void)destroy_event_locked(*event);
    }
  }

  Status retain(CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    TokenRecord* record = find_token_locked(token);
    if (record == nullptr) return {ErrorCode::stale_token};
    if (record->references == std::numeric_limits<std::size_t>::max()) {
      return {ErrorCode::resource_busy};
    }
    ++record->references;
    return Status::success();
  }

  QueryResult is_complete(CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {{ErrorCode::closed}, false};
    TokenRecord* record = find_token_locked(token);
    if (record == nullptr) return {{ErrorCode::stale_token}, false};
    bool complete = true;
    Status first_error = Status::success();
    for (const EventRef ref : record->events) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) {
        if (first_error.ok()) first_error = {ErrorCode::stale_token};
        complete = false;
        continue;
      }
      if (event->completed) continue;
      auto status_and_context = query_event_locked(*event);
      const Status status = status_and_context.first;
      const CUresult result = status_and_context.second;
      if (!status.ok()) {
        if (result == CUDA_ERROR_NOT_READY) {
          complete = false;
          continue;
        }
        poison_event_locked(*event, status);
        if (first_error.ok()) first_error = status;
        complete = false;
      } else {
        event->completed = true;
      }
    }
    const Status collected = collect_locked();
    if (!collected.ok() && first_error.ok()) first_error = collected;
    if (!first_error.ok()) return {first_error, false};
    return {Status::success(), complete};
  }

  Status wait(CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    TokenRecord* record = find_token_locked(token);
    if (record == nullptr) return {ErrorCode::stale_token};
    const std::vector<EventRef> refs = record->events;
    Status first_error = Status::success();
    for (const EventRef ref : refs) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) {
        if (first_error.ok()) first_error = {ErrorCode::stale_token};
        continue;
      }
      const Status status = synchronize_event_locked(*event);
      if (!status.ok() && first_error.ok()) first_error = status;
    }
    const Status collected = collect_locked();
    if (!collected.ok() && first_error.ok()) first_error = collected;
    return first_error;
  }

  Status release(CompletionToken token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return {ErrorCode::closed};
    auto it = tokens_.find(token.slot);
    if (it == tokens_.end() || it->second.generation != token.generation) {
      return {ErrorCode::stale_token};
    }
    if (it->second.references == 0) return {ErrorCode::stale_token};
    if (--it->second.references != 0) return Status::success();
    for (const EventRef ref : it->second.events) {
      EventRecord* event = find_event_locked(ref);
      if (event != nullptr && event->token_refs != 0) --event->token_refs;
    }
    free_token_slots_.push_back(token.slot);
    tokens_.erase(it);
    return collect_locked();
  }

 private:
  using TokenMap = std::map<std::uint64_t, TokenRecord>;
  using EventMap = std::map<std::uint64_t, EventRecord>;

  static void deduplicate(std::vector<EventRef>& refs) {
    std::sort(refs.begin(), refs.end());
    refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
  }

  EventRecord* find_event_locked(EventRef ref) noexcept {
    auto it = events_.find(ref.slot);
    if (it == events_.end() || it->second.ref.generation != ref.generation) return nullptr;
    return &it->second;
  }

  TokenRecord* find_token_locked(CompletionToken token) noexcept {
    auto it = tokens_.find(token.slot);
    if (it == tokens_.end() || it->second.generation != token.generation) return nullptr;
    return &it->second;
  }

  bool contains_stream_prefix_locked(const TokenRecord& token,
                                     LogicalDeviceId device,
                                     LogicalStreamId stream) noexcept {
    if (token.device != device) return false;
    for (const EventRef ref : token.events) {
      const EventRecord* event = find_event_locked(ref);
      if (event != nullptr && event->device == device &&
          event->source_stream == stream) {
        return true;
      }
    }
    return false;
  }

  Status device_poison_locked(LogicalDeviceId device) noexcept {
    auto it = devices_.find(device);
    if (it == devices_.end()) return {ErrorCode::unknown_owner};
    return it->second.poisoned ? it->second.poison : Status::success();
  }

  void poison_locked(StreamRecord& stream, Status status) noexcept {
    if (!stream.poisoned) {
      stream.poisoned = true;
      stream.poison = status;
    }
  }

  void poison_device_locked(LogicalDeviceId device, Status status) noexcept {
    DeviceRecord& record = devices_[device];
    if (!record.poisoned) {
      record.poisoned = true;
      record.poison = status;
    }
  }

  void poison_event_locked(EventRecord& event, Status status) noexcept {
    poison_device_locked(event.device, status);
    auto stream = streams_.find({event.device, event.source_stream});
    if (stream != streams_.end()) poison_locked(stream->second, status);
  }

  void mark_stream_events_complete_locked(LogicalDeviceId device,
                                           LogicalStreamId stream,
                                           OwnerId owner) noexcept {
    // A successful cuStreamSynchronize proves every operation queued on this
    // owner stream is complete, including event handles whose record call
    // returned an ambiguous error before the drain.
    for (auto& entry : events_) {
      EventRecord& event = entry.second;
      if (event.device != device || event.source_stream != stream ||
          event.owner != owner) {
        continue;
      }
      event.record_result = CUDA_SUCCESS;
      event.captured = true;
      event.completed = true;
    }
  }

  CUresult pop_context(CUcontext expected) noexcept {
    CUcontext popped = nullptr;
    const CUresult result = driver_.ctx_pop_current(&popped);
    if (result != CUDA_SUCCESS) return result;
    if (popped != expected) return CUDA_ERROR_INVALID_CONTEXT;
    return CUDA_SUCCESS;
  }

  Status seal_owner_locked(LogicalDeviceId device, LogicalStreamId stream,
                           OwnerId owner, StreamRecord& record,
                           CUcontext context, StreamOwner& stream_owner) noexcept {
    if (!stream_owner.dirty) return Status::success();
    const CUresult pushed = driver_.ctx_push_current(context);
    if (pushed != CUDA_SUCCESS) return driver_error(pushed);
    Status status = record_event_current_locked(device, stream, owner, record,
                                                context,
                                                stream_owner.native_stream);
    const CUresult popped = pop_context(context);
    if (popped != CUDA_SUCCESS && status.ok()) {
      status = {ErrorCode::context_restore_failed, popped};
    }
    if (status.ok()) stream_owner.dirty = false;
    return status;
  }

  Status seal_dirty_owners_locked(LogicalDeviceId device,
                                  LogicalStreamId stream,
                                  StreamRecord& record) noexcept {
    for (auto& owner_entry : record.owners) {
      if (!owner_entry.second.dirty) continue;
      const auto owner = owners_.find({device, owner_entry.first});
      if (owner == owners_.end()) return {ErrorCode::unknown_owner};
      const Status status = seal_owner_locked(
          device, stream, owner_entry.first, record, owner->second.context,
          owner_entry.second);
      if (!status.ok()) return status;
    }
    return Status::success();
  }

  Status record_event_current_locked(LogicalDeviceId device,
                                     LogicalStreamId stream, OwnerId owner,
                                     StreamRecord& stream_record,
                                     CUcontext context,
                                     CUstream native_stream) noexcept {
    CUevent handle = nullptr;
    CUresult result = driver_.event_create(&handle, CU_EVENT_DISABLE_TIMING);
    if (result != CUDA_SUCCESS) {
      // Event allocation can fail after work has already been queued. A
      // synchronous stream drain is a conservative correctness fallback; if
      // that also fails, dirty state remains for fail-stop/shutdown handling.
      const CUresult synchronized = driver_.stream_synchronize(native_stream);
      if (synchronized != CUDA_SUCCESS) return driver_error(synchronized);
      mark_stream_events_complete_locked(device, stream, owner);
      return Status::success();
    }
    const EventRef ref = allocate_event_ref_locked();
    EventRecord event{ref, device, stream, owner, context, handle,
                      ++stream_record.sequence, 0, 1, false, false,
                      CUDA_SUCCESS};
    auto previous = stream_record.heads.find(owner);
    if (previous != stream_record.heads.end()) {
      EventRecord* old = find_event_locked(previous->second);
      if (old != nullptr && old->head_refs != 0) --old->head_refs;
    }
    events_.emplace(ref.slot, event);
    stream_record.heads[owner] = ref;
    // A driver error may report an earlier asynchronous failure even when the
    // event command reached the stream. Preserve the event and fail-stop state;
    // never destroy it or claim the prefix complete on this ambiguity.
    result = driver_.event_record(handle, native_stream);
    EventRecord* stored = find_event_locked(ref);
    if (stored == nullptr) return {ErrorCode::stale_token};
    if (result != CUDA_SUCCESS) {
      const CUresult synchronized = driver_.stream_synchronize(native_stream);
      if (synchronized == CUDA_SUCCESS) {
        // The event may not have been recorded, but the stream drain proves
        // that the represented prefix is complete and the handle is safe to
        // retire when references disappear.
        mark_stream_events_complete_locked(device, stream, owner);
        return Status::success();
      }
      stored->record_result = synchronized;
      return driver_error(synchronized);
    }
    stored->captured = true;
    return Status::success();
  }

  std::pair<Status, CUresult> query_event_locked(EventRecord& event) noexcept {
    if (!event.captured) return {driver_error(event.record_result), event.record_result};
    const auto owner = owners_.find({event.device, event.owner});
    const CUcontext context = owner == owners_.end() ? event.context : owner->second.context;
    CUresult result = driver_.ctx_push_current(context);
    if (result != CUDA_SUCCESS) return {driver_error(result), result};
    result = driver_.event_query(event.handle);
    const CUresult popped = pop_context(context);
    if (popped != CUDA_SUCCESS) {
      return {{ErrorCode::context_restore_failed, popped}, popped};
    }
    if (result == CUDA_SUCCESS) return {Status::success(), result};
    if (result == CUDA_ERROR_NOT_READY) return {{ErrorCode::driver_failure, result}, result};
    return {driver_error(result), result};
  }

  Status synchronize_event_locked(EventRecord& event) noexcept {
    if (event.completed) return Status::success();
    if (!event.captured) return driver_error(event.record_result);
    const auto owner = owners_.find({event.device, event.owner});
    const CUcontext context = owner == owners_.end() ? event.context : owner->second.context;
    CUresult result = driver_.ctx_push_current(context);
    if (result != CUDA_SUCCESS) {
      const Status status = driver_error(result);
      poison_event_locked(event, status);
      return status;
    }
    result = driver_.event_synchronize(event.handle);
    const CUresult popped = pop_context(context);
    Status status = result == CUDA_SUCCESS ? Status::success() : driver_error(result);
    if (popped != CUDA_SUCCESS) status = {ErrorCode::context_restore_failed, popped};
    if (status.ok()) event.completed = true;
    else poison_event_locked(event, status);
    return status;
  }

  Status destroy_event_locked(EventRecord& event) noexcept {
    if (event.token_refs != 0 || event.head_refs != 0 || !event.completed ||
        !event.captured) {
      return {ErrorCode::resource_busy};
    }
    const auto owner = owners_.find({event.device, event.owner});
    const CUcontext context = owner == owners_.end() ? event.context : owner->second.context;
    CUresult result = driver_.ctx_push_current(context);
    if (result != CUDA_SUCCESS) return driver_error(result);
    result = driver_.event_destroy(event.handle);
    const CUresult popped = pop_context(context);
    // Once destroy succeeds the CUDA handle is gone, even if restoring the
    // caller's context fails. Retire the record before returning that error so
    // later cleanup cannot issue a second destroy against a dead handle.
    if (result == CUDA_SUCCESS) {
      free_event_slots_.push_back(event.ref.slot);
      events_.erase(event.ref.slot);
    }
    if (popped != CUDA_SUCCESS) return {ErrorCode::context_restore_failed, popped};
    return result == CUDA_SUCCESS ? Status::success() : driver_error(result);
  }

  Status collect_locked() noexcept {
    Status first_error = Status::success();
    std::vector<EventRef> candidates;
    for (const auto& entry : events_) {
      if (entry.second.token_refs == 0 && entry.second.head_refs == 0) {
        candidates.push_back(entry.second.ref);
      }
    }
    for (const EventRef ref : candidates) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) continue;
      if (!event->completed) {
        const auto status_result = query_event_locked(*event);
        if (!status_result.first.ok()) {
          if (status_result.second == CUDA_ERROR_NOT_READY) continue;
          poison_event_locked(*event, status_result.first);
          if (first_error.ok()) first_error = status_result.first;
          continue;
        }
        event->completed = true;
      }
      const Status destroyed = destroy_event_locked(*event);
      if (!destroyed.ok() && first_error.ok()) first_error = destroyed;
    }
    return first_error;
  }

  EventRef allocate_event_ref_locked() {
    std::uint64_t slot = 0;
    if (!free_event_slots_.empty()) {
      slot = free_event_slots_.back();
      free_event_slots_.pop_back();
      ++event_generations_[slot];
    } else {
      slot = next_event_slot_++;
      event_generations_[slot] = 1;
    }
    return {slot, event_generations_[slot]};
  }

  CompletionToken allocate_token_locked() {
    std::uint64_t slot = 0;
    if (!free_token_slots_.empty()) {
      slot = free_token_slots_.back();
      free_token_slots_.pop_back();
      ++token_generations_[slot];
    } else {
      slot = next_token_slot_++;
      token_generations_[slot] = 1;
    }
    return {slot, token_generations_[slot]};
  }

  Result<CompletionToken> create_token_locked(LogicalDeviceId device,
                                             LogicalStreamId source_stream,
                                             std::vector<EventRef> events) noexcept {
    deduplicate(events);
    const CompletionToken token = allocate_token_locked();
    TokenRecord record;
    record.generation = token.generation;
    record.references = 1;
    record.device = device;
    record.source_stream = source_stream;
    record.events = std::move(events);
    for (const EventRef ref : record.events) {
      EventRecord* event = find_event_locked(ref);
      if (event == nullptr) {
        free_token_slots_.push_back(token.slot);
        return {{ErrorCode::stale_token}, {}};
      }
      ++event->token_refs;
    }
    tokens_.emplace(token.slot, std::move(record));
    return {Status::success(), token};
  }

  Result<CompletionToken> snapshot_stream_locked(LogicalDeviceId device,
                                                LogicalStreamId stream,
                                                const StreamRecord& record) noexcept {
    std::vector<EventRef> events;
    for (const auto& head : record.heads) events.push_back(head.second);
    return create_token_locked(device, stream, std::move(events));
  }

  DriverDispatch& driver_;
  std::mutex mutex_;
  bool closed_{false};
  std::map<OwnerKey, OwnerRecord> owners_;
  std::map<StreamKey, StreamRecord> streams_;
  std::map<LogicalDeviceId, DeviceRecord> devices_;
  EventMap events_;
  TokenMap tokens_;
  std::map<std::uint64_t, std::uint64_t> event_generations_;
  std::map<std::uint64_t, std::uint64_t> token_generations_;
  std::vector<std::uint64_t> free_event_slots_;
  std::vector<std::uint64_t> free_token_slots_;
  std::uint64_t next_event_slot_{1};
  std::uint64_t next_token_slot_{1};
};

OwnerExecution::OwnerExecution(DriverDispatch& driver)
    : impl_(std::make_unique<Impl>(driver)) {}

OwnerExecution::~OwnerExecution() {
  if (impl_) impl_->best_effort_shutdown();
}

Status OwnerExecution::register_owner(LogicalDeviceId device, OwnerId owner,
                                      CUcontext context) noexcept {
  return impl_->register_owner(device, owner, context);
}

Status OwnerExecution::register_stream(LogicalDeviceId device,
                                       LogicalStreamId stream, OwnerId owner,
                                       CUstream native_stream) noexcept {
  return impl_->register_stream(device, stream, owner, native_stream);
}

Status OwnerExecution::unregister_stream(LogicalDeviceId device,
                                         LogicalStreamId stream,
                                         OwnerId owner) noexcept {
  return impl_->unregister_stream(device, stream, owner);
}

Status OwnerExecution::observe_native_submission(LogicalDeviceId device,
                                                 LogicalStreamId stream,
                                                 OwnerId owner,
                                                 CUstream native_stream,
                                                 CUresult forwarded_result) noexcept {
  return impl_->observe_native_submission(device, stream, owner, native_stream,
                                          forwarded_result);
}

Status OwnerExecution::forward_native_submission(
    LogicalDeviceId device, LogicalStreamId stream, OwnerId owner,
    CUstream native_stream, const EnqueueCallback& forward) noexcept {
  return impl_->forward_native_submission(device, stream, owner, native_stream,
                                          forward);
}

SubmissionResult OwnerExecution::submit(LogicalDeviceId device,
                                        LogicalStreamId stream, OwnerId owner,
                                        const CompletionToken* dependencies,
                                        std::size_t dependency_count,
                                        const EnqueueCallback& enqueue) noexcept {
  return impl_->submit(device, stream, owner, dependencies, dependency_count,
                       enqueue);
}

SubmissionResult OwnerExecution::wait_on_stream(LogicalDeviceId device,
                                                LogicalStreamId stream,
                                                OwnerId owner,
                                                CompletionToken dependency) noexcept {
  return impl_->wait_on_stream(device, stream, owner, dependency);
}

Result<CompletionToken> OwnerExecution::current_token(LogicalDeviceId device,
                                                       LogicalStreamId stream) noexcept {
  return impl_->current_token(device, stream);
}

Result<CompletionToken> OwnerExecution::current_device_token(
    LogicalDeviceId device) noexcept {
  return impl_->current_device_token(device);
}

Status OwnerExecution::collect() noexcept { return impl_->collect(); }
Status OwnerExecution::shutdown() noexcept { return impl_->shutdown(); }
Status OwnerExecution::retain(CompletionToken token) noexcept {
  return impl_->retain(token);
}
QueryResult OwnerExecution::is_complete(CompletionToken token) noexcept {
  return impl_->is_complete(token);
}
Status OwnerExecution::wait(CompletionToken token) noexcept {
  return impl_->wait(token);
}
Status OwnerExecution::release(CompletionToken token) noexcept {
  return impl_->release(token);
}

}  // namespace v200::execution
