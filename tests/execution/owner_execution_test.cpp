#include "v200/execution/owner_execution.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace v200::execution;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

template <class T>
T* fake_handle(std::uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

class FakeDriver final : public DriverDispatch {
 public:
  CUresult ctx_push_current(CUcontext context) noexcept override {
    context_stack.push_back(context);
    return CUDA_SUCCESS;
  }

  CUresult ctx_pop_current(CUcontext* context) noexcept override {
    if (fail_next_pop) {
      fail_next_pop = false;
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (context_stack.size() <= 1) return CUDA_ERROR_INVALID_CONTEXT;
    *context = context_stack.back();
    context_stack.pop_back();
    return CUDA_SUCCESS;
  }

  CUresult event_create(CUevent* event, unsigned flags) noexcept override {
    if (flags != CU_EVENT_DISABLE_TIMING) return CUDA_ERROR_INVALID_VALUE;
    if (event_create_result != CUDA_SUCCESS) return event_create_result;
    const auto id = next_event++;
    *event = fake_handle<CUevent_st>(id);
    events.emplace(*event, EventState{});
    return CUDA_SUCCESS;
  }

  CUresult event_destroy(CUevent event) noexcept override {
    ++destroy_calls;
    auto it = events.find(event);
    if (it == events.end()) return CUDA_ERROR_INVALID_HANDLE;
    if (!it->second.complete) return CUDA_ERROR_NOT_READY;
    events.erase(it);
    destroyed.push_back(event);
    if (fail_pop_after_destroy) {
      fail_pop_after_destroy = false;
      fail_next_pop = true;
    }
    return CUDA_SUCCESS;
  }

  CUresult event_record(CUevent event, CUstream stream) noexcept override {
    auto it = events.find(event);
    if (it == events.end()) return CUDA_ERROR_INVALID_HANDLE;
    it->second.recorded_stream = stream;
    it->second.recorded_context = context_stack.back();
    it->second.recorded_thread = std::this_thread::get_id();
    recorded.push_back(event);
    return event_record_result;
  }

  CUresult event_query(CUevent event) noexcept override {
    ++query_calls;
    auto it = events.find(event);
    if (it == events.end()) return CUDA_ERROR_INVALID_HANDLE;
    const auto failed = query_failures.find(event);
    if (failed != query_failures.end()) return failed->second;
    return it->second.complete ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
  }

  CUresult event_synchronize(CUevent event) noexcept override {
    ++synchronize_calls;
    synchronized.push_back(event);
    auto it = events.find(event);
    if (it == events.end()) return CUDA_ERROR_INVALID_HANDLE;
    const auto failed = synchronize_failures.find(event);
    if (failed != synchronize_failures.end()) return failed->second;
    it->second.complete = true;
    return CUDA_SUCCESS;
  }

  CUresult stream_synchronize(CUstream stream) noexcept override {
    ++stream_synchronize_calls;
    synchronized_streams.push_back(stream);
    if (stream_synchronize_result != CUDA_SUCCESS) return stream_synchronize_result;
    for (auto& entry : events) {
      if (entry.second.recorded_stream == stream) entry.second.complete = true;
    }
    return CUDA_SUCCESS;
  }

  CUresult stream_wait_event(CUstream stream, CUevent event,
                             unsigned flags) noexcept override {
    if (flags != 0) return CUDA_ERROR_INVALID_VALUE;
    if (events.find(event) == events.end()) return CUDA_ERROR_INVALID_HANDLE;
    waits.emplace_back(stream, event);
    return wait_result;
  }

  void complete(CUevent event) { events.at(event).complete = true; }
  void complete_all() {
    for (auto& entry : events) entry.second.complete = true;
  }

  struct EventState {
    CUstream recorded_stream{};
    CUcontext recorded_context{};
    std::thread::id recorded_thread{};
    bool complete{false};
  };

  std::vector<CUcontext> context_stack{fake_handle<CUctx_st>(0x999)};
  std::map<CUevent, EventState> events;
  std::map<CUevent, CUresult> query_failures;
  std::map<CUevent, CUresult> synchronize_failures;
  std::vector<CUevent> recorded;
  std::vector<CUevent> destroyed;
  std::vector<CUevent> synchronized;
  std::vector<CUstream> synchronized_streams;
  std::vector<std::pair<CUstream, CUevent>> waits;
  std::uint64_t next_event{0x1000};
  std::size_t destroy_calls{0};
  std::size_t query_calls{0};
  std::size_t synchronize_calls{0};
  std::size_t stream_synchronize_calls{0};
  CUresult wait_result{CUDA_SUCCESS};
  CUresult event_create_result{CUDA_SUCCESS};
  CUresult event_record_result{CUDA_SUCCESS};
  CUresult stream_synchronize_result{CUDA_SUCCESS};
  bool fail_pop_after_destroy{false};
  bool fail_next_pop{false};
};

constexpr LogicalDeviceId kDevice = 1;
constexpr LogicalStreamId kStream = 7;
constexpr OwnerId kOwnerA = 10;
constexpr OwnerId kOwnerB = 20;
const auto kContextA = fake_handle<CUctx_st>(0x101);
const auto kContextB = fake_handle<CUctx_st>(0x202);
const auto kAmbient = fake_handle<CUctx_st>(0x999);
const auto kNativeA = fake_handle<CUstream_st>(0x11);
const auto kNativeB = fake_handle<CUstream_st>(0x22);

void register_two_owner_stream(OwnerExecution& execution) {
  require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
          "register owner A");
  require(execution.register_owner(kDevice, kOwnerB, kContextB).ok(),
          "register owner B");
  require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
          "register owner A stream");
  require(execution.register_stream(kDevice, kStream, kOwnerB, kNativeB).ok(),
          "register owner B stream");
}

void test_native_prefix_peer_fanin_and_context() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  register_two_owner_stream(execution);
  const auto ambient_before = driver.context_stack;
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe native anchor operation");

  bool callback_ran = false;
  const auto submitted = execution.submit(
      kDevice, kStream, kOwnerB, nullptr, 0,
      [&](CUstream stream) {
        callback_ran = true;
        require(stream == kNativeB, "callback receives configured native stream");
        require(driver.context_stack.back() == kContextB,
                "callback runs with configured owner context current");
        require(driver.waits.size() == 1,
                "peer prefix wait is queued before callback work");
        require(driver.waits.front().first == kNativeB,
                "peer wait targets destination owner stream");
        return CUDA_SUCCESS;
      });
  require(submitted.status.ok() && submitted.completion.valid(),
          "submit returns full-prefix completion token");
  require(callback_ran, "peer callback executed");
  require(driver.recorded.size() == 2,
          "both dirty owner prefixes receive completion events");
  require(driver.events.at(driver.recorded[0]).recorded_stream == kNativeA,
          "anchor event follows its native stream prefix");
  require(driver.events.at(driver.recorded[0]).recorded_context == kContextA,
          "anchor event recorded under owner A context");
  require(driver.events.at(driver.recorded[1]).recorded_context == kContextB,
          "destination event recorded under owner B context");
  require(driver.context_stack == ambient_before && driver.context_stack.back() == kAmbient,
          "ambient context is restored after fan-in submission");

  driver.complete(driver.recorded[0]);
  const auto partial = execution.is_complete(submitted.completion);
  require(partial.status.ok() && !partial.complete,
          "full-prefix token remains pending while a peer event is pending");
  driver.complete(driver.recorded[1]);
  const auto complete = execution.is_complete(submitted.completion);
  require(complete.status.ok() && complete.complete,
          "full-prefix token completes only after every owner event");
  require(execution.release(submitted.completion).ok(), "release fan-in token");
  require(execution.shutdown().ok(), "shutdown drains and destroys events safely");
  require(driver.events.empty() && driver.destroyed.size() == 2,
          "completed event resources are destroyed at shutdown");
}

void test_dependency_cycle_and_stale_token_generation() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  register_two_owner_stream(execution);
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe native work before token capture");
  const auto old = execution.current_token(kDevice, kStream);
  require(old.ok() && old.value.valid(), "capture native full-prefix token");

  const auto self_wait = execution.wait_on_stream(kDevice, kStream, kOwnerB,
                                                  old.value);
  require(self_wait.status.code == ErrorCode::dependency_cycle,
          "logical stream cannot wait on its own prefix token");
  const auto device_prefix = execution.current_device_token(kDevice);
  require(device_prefix.ok(), "capture a full device prefix token");
  const auto device_self_wait = execution.wait_on_stream(
      kDevice, kStream, kOwnerB, device_prefix.value);
  require(device_self_wait.status.code == ErrorCode::dependency_cycle,
          "device token containing this stream cannot create a wait cycle");
  require(execution.release(device_prefix.value).ok(),
          "release full-device token after cycle check");

  const auto cycle = execution.submit(
      kDevice, kStream, kOwnerB, &old.value, 1,
      [](CUstream) { return CUDA_SUCCESS; });
  require(cycle.status.code == ErrorCode::dependency_cycle,
          "submission rejects an explicit same-stream dependency cycle");

  require(execution.release(old.value).ok(), "release old token ref");
  const auto replacement = execution.current_token(kDevice, kStream);
  require(replacement.ok() && replacement.value.slot == old.value.slot,
          "released token slot is reused");
  require(replacement.value.generation > old.value.generation,
          "reused token slot advances its generation");
  const auto stale = execution.is_complete(old.value);
  require(stale.status.code == ErrorCode::stale_token,
          "old token generation cannot resolve after slot reuse");
  require(execution.release(replacement.value).ok(), "release replacement token");
  driver.complete_all();
  require(execution.shutdown().ok(), "shutdown after stale-generation test");
}

void test_completion_lifetime_no_premature_free() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
          "register lifetime owner");
  require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
          "register lifetime stream");
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe first native prefix");
  const auto first = execution.current_token(kDevice, kStream);
  require(first.ok(), "capture first event prefix");
  const CUevent first_event = driver.recorded.back();
  require(execution.retain(first.value).ok(),
          "memory-use owner retains the completion token");
  require(execution.release(first.value).ok(), "release first prefix token");
  require(execution.is_complete(first.value).status.ok(),
          "retained token remains valid after caller drops its reference");
  require(execution.collect().ok(), "collect cannot retire current stream head");
  require(driver.events.find(first_event) != driver.events.end(),
          "stream head keeps unfinished event alive");
  require(driver.destroyed.empty(), "unfinished current head was not destroyed");

  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe successor native prefix");
  const auto second = execution.current_token(kDevice, kStream);
  require(second.ok(), "capture successor prefix");
  const CUevent second_event = driver.recorded.back();
  require(first_event != second_event, "each prefix gets a distinct event handle");
  require(driver.events.find(first_event) != driver.events.end(),
          "superseding an event does not free it before completion");
  driver.complete(first_event);
  require(execution.collect().ok(), "retire completed event after head advances");
  require(driver.events.find(first_event) != driver.events.end(),
          "retained allocation-use token prevents event destruction");
  require(execution.is_complete(first.value).complete,
          "retained allocation-use token remains queryable after head advances");
  require(execution.release(first.value).ok(),
          "release retained allocation-use token after completion");
  require(driver.events.find(first_event) == driver.events.end(),
          "completed superseded event is reclaimed after token release");
  require(driver.events.find(second_event) != driver.events.end(),
          "current unfinished event remains alive");
  require(driver.destroyed.size() == 1 && driver.destroyed.front() == first_event,
          "only the completed unreferenced event is destroyed");
  require(execution.release(second.value).ok(), "release second token");
  driver.complete(second_event);
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe third native prefix to exercise event slot reuse");
  const auto third = execution.current_token(kDevice, kStream);
  require(third.ok(), "capture third prefix after event-slot reuse");
  const CUevent third_event = driver.recorded.back();
  require(execution.is_complete(first.value).status.code == ErrorCode::stale_token,
          "retired token generation stays stale after event slot reuse");
  require(execution.collect().ok(), "reclaim completed second event after replacement");
  require(driver.events.find(second_event) == driver.events.end(),
          "superseded completed second event is reclaimed");
  driver.complete(third_event);
  require(execution.release(third.value).ok(), "release third token");
  require(execution.shutdown().ok(), "shutdown lifetime test");
}

void test_async_peer_failure_fail_stops_and_drains_all_events() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  register_two_owner_stream(execution);
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "observe anchor before async failure");
  const auto submitted = execution.submit(kDevice, kStream, kOwnerB, nullptr, 0,
                                         [](CUstream) { return CUDA_SUCCESS; });
  require(submitted.status.ok() && submitted.completion.valid(),
          "create peer fan-in before async failure");
  const auto failed_event = driver.recorded.front();
  driver.query_failures[failed_event] = CUDA_ERROR_LAUNCH_FAILED;
  driver.synchronize_failures[failed_event] = CUDA_ERROR_LAUNCH_FAILED;

  const auto query = execution.is_complete(submitted.completion);
  require(query.status.code == ErrorCode::driver_failure && !query.complete,
          "async peer error is surfaced instead of reporting completion");
  bool later_callback_ran = false;
  const auto rejected = execution.submit(
      kDevice, kStream, kOwnerA, nullptr, 0,
      [&](CUstream) {
        later_callback_ran = true;
        return CUDA_SUCCESS;
      });
  require(rejected.status.code == ErrorCode::driver_failure && !later_callback_ran,
          "async peer failure prevents later owner submissions");

  const auto before_sync = driver.synchronize_calls;
  const Status waited = execution.wait(submitted.completion);
  require(waited.code == ErrorCode::driver_failure,
          "failed prefix remains failed when synchronizing");
  require(driver.synchronize_calls - before_sync == 2,
          "wait still attempts to drain every event in the full prefix");
  require(std::find(driver.synchronized.begin(), driver.synchronized.end(),
                    driver.recorded.back()) != driver.synchronized.end(),
          "peer event is attempted even after another owner fails");
  require(driver.destroyed.empty(),
          "events are retained after unproven async failure");
  require(execution.release(submitted.completion).ok(),
          "caller can drop a failed token after handling its error");
  require(driver.destroyed.empty(),
          "failed event stays pinned by stream ownership after token release");
}

void test_synchronous_enqueue_error_is_drainable_and_context_restored() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
          "register enqueue error owner");
  require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
          "register enqueue error stream");
  const auto ambient = driver.context_stack;
  const auto result = execution.submit(
      kDevice, kStream, kOwnerA, nullptr, 0,
      [&](CUstream stream) {
        require(stream == kNativeA && driver.context_stack.back() == kContextA,
                "failed enqueue executes with owner binding");
        return CUDA_ERROR_INVALID_VALUE;
      });
  require(result.status.code == ErrorCode::driver_failure,
          "synchronous enqueue error is propagated");
  require(result.completion.valid(),
          "tail event is returned so already-issued work remains drainable");
  require(driver.context_stack == ambient && driver.context_stack.back() == kAmbient,
          "ambient context is restored on callback error");
  require(driver.recorded.size() == 1,
          "tail event is recorded after a synchronous callback error");
  require(execution.wait(result.completion).ok(),
          "captured tail event remains waitable after synchronous error");
  require(execution.release(result.completion).ok(), "release errored submission token");
  const auto rejected = execution.submit(kDevice, kStream, kOwnerA, nullptr, 0,
                                        [](CUstream) { return CUDA_SUCCESS; });
  require(rejected.status.code == ErrorCode::driver_failure,
          "synchronous callback error fail-stops subsequent work");
  require(execution.shutdown().ok(), "shutdown after drainable callback error");
}

void test_native_forward_is_atomic_with_peer_submission() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  register_two_owner_stream(execution);
  std::mutex gate_mutex;
  std::condition_variable gate_changed;
  bool native_entered = false;
  bool release_native = false;
  bool peer_started = false;
  Status native_status{};
  SubmissionResult peer_result{};

  std::thread native([&] {
    native_status = execution.forward_native_submission(
        kDevice, kStream, kOwnerA, kNativeA, [&](CUstream stream) {
          require(stream == kNativeA, "native forward receives registered stream");
          std::unique_lock<std::mutex> lock(gate_mutex);
          native_entered = true;
          gate_changed.notify_all();
          gate_changed.wait(lock, [&] { return release_native; });
          return CUDA_SUCCESS;
        });
  });

  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_changed.wait(lock, [&] { return native_entered; });
  }
  std::thread peer([&] {
    {
      std::lock_guard<std::mutex> lock(gate_mutex);
      peer_started = true;
      gate_changed.notify_all();
    }
    peer_result = execution.submit(
        kDevice, kStream, kOwnerB, nullptr, 0, [&](CUstream stream) {
          require(stream == kNativeB, "peer submit receives its native stream");
          require(driver.waits.size() == 1,
                  "peer cannot overtake a native call before its prefix is sealed");
          return CUDA_SUCCESS;
        });
  });
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate_changed.wait(lock, [&] { return peer_started; });
    release_native = true;
  }
  gate_changed.notify_all();
  native.join();
  peer.join();
  require(native_status.ok(), "native operation and prefix publish atomically");
  require(peer_result.status.ok() && peer_result.completion.valid(),
          "peer submission follows the atomic native operation");
  require(driver.waits.size() == 1 && driver.waits.front().first == kNativeB,
          "cross-owner wait reaches the destination stream");
  require(execution.wait(peer_result.completion).ok(),
          "wait for atomic fan-in result");
  require(execution.release(peer_result.completion).ok(),
          "release atomic fan-in token");
  driver.complete_all();
  require(execution.shutdown().ok(), "shutdown after atomic-forward test");
}

void test_event_capture_failure_uses_safe_stream_drain() {
  {
    FakeDriver driver;
    OwnerExecution execution(driver);
    require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
            "register event-create fallback owner");
    require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
            "register event-create fallback stream");
    require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                                kNativeA, CUDA_SUCCESS)
                .ok(),
            "mark work dirty before event-create fallback");
    driver.event_create_result = CUDA_ERROR_OUT_OF_MEMORY;
    const auto token = execution.current_token(kDevice, kStream);
    require(token.ok() && token.value.valid(),
            "successful fallback drain provides a safe completed prefix");
    require(driver.stream_synchronize_calls == 1 &&
                driver.synchronized_streams.front() == kNativeA,
            "event-create failure drains the actual native stream");
    const auto complete = execution.is_complete(token.value);
    require(complete.status.ok() && complete.complete,
            "fallback-completed prefix is safely complete");
    require(execution.release(token.value).ok(), "release fallback token");
    require(execution.shutdown().ok(), "shutdown after safe fallback");
  }

  {
    FakeDriver driver;
    OwnerExecution execution(driver);
    require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
            "register failed-fallback owner");
    require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
            "register failed-fallback stream");
    require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                                kNativeA, CUDA_SUCCESS)
                .ok(),
            "mark native prefix dirty for failed fallback");
    driver.event_create_result = CUDA_ERROR_OUT_OF_MEMORY;
    driver.stream_synchronize_result = CUDA_ERROR_LAUNCH_FAILED;
    const auto failed = execution.current_token(kDevice, kStream);
    require(failed.status.code == ErrorCode::driver_failure,
            "event and stream-drain failures are not mistaken for completion");
    require(execution.shutdown().code == ErrorCode::driver_failure,
            "shutdown fails while dirty work remains unproven");
    driver.stream_synchronize_result = CUDA_SUCCESS;
    require(execution.shutdown().ok(),
            "shutdown retries and drains dirty work before releasing stream state");
  }

  {
    FakeDriver driver;
    OwnerExecution execution(driver);
    require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
            "register ambiguous event-record owner");
    require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
            "register ambiguous event-record stream");
    require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                                kNativeA, CUDA_SUCCESS)
                .ok(),
            "mark work dirty before ambiguous event record");
    driver.event_record_result = CUDA_ERROR_LAUNCH_FAILED;
    driver.stream_synchronize_result = CUDA_ERROR_LAUNCH_FAILED;
    const auto failed = execution.current_token(kDevice, kStream);
    require(failed.status.code == ErrorCode::driver_failure,
            "ambiguous event record plus failed drain is not completion");
    require(execution.shutdown().code == ErrorCode::driver_failure,
            "shutdown preserves unproven event-record state");

    // A later successful stream drain proves both the failed event attempt and
    // any newer event handle safe, allowing shutdown to retire the state.
    driver.stream_synchronize_result = CUDA_SUCCESS;
    require(execution.shutdown().ok(),
            "successful retry drain recovers ambiguous event-record state");
    require(driver.events.empty(),
            "recovered failed event handles are destroyed after full drain");
  }

  {
    FakeDriver driver;
    OwnerExecution execution(driver);
    require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
            "register event-record fallback owner");
    require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
            "register event-record fallback stream");
    require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                                kNativeA, CUDA_SUCCESS)
                .ok(),
            "mark native prefix dirty before event-record fallback");
    driver.event_record_result = CUDA_ERROR_LAUNCH_FAILED;
    const auto token = execution.current_token(kDevice, kStream);
    require(token.ok() && execution.is_complete(token.value).complete,
            "stream drain proves completion when event record reports failure");
    require(driver.stream_synchronize_calls == 1,
            "event-record failure also uses the safe stream-drain fallback");
    require(execution.release(token.value).ok(), "release event-record fallback token");
    require(execution.shutdown().ok(), "shutdown after event-record fallback");
    require(driver.events.empty(), "synchronized event handle is safely destroyed");
  }
}

void test_per_thread_default_stream_is_sealed_on_forwarding_thread() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
          "register PTDS owner");
  require(execution.register_stream(kDevice, kStream, kOwnerA,
                                    CU_STREAM_PER_THREAD)
              .ok(),
          "register per-thread default stream sentinel");

  const auto dispatch_thread = std::this_thread::get_id();
  bool forwarded = false;
  const auto status = execution.forward_native_submission(
      kDevice, kStream, kOwnerA, CU_STREAM_PER_THREAD,
      [&](CUstream stream) {
        forwarded = true;
        require(stream == CU_STREAM_PER_THREAD,
                "native callback receives per-thread stream sentinel");
        return CUDA_SUCCESS;
      });
  require(status.ok() && forwarded, "forward native PTDS operation");
  require(driver.recorded.size() == 1 &&
              driver.events.at(driver.recorded.front()).recorded_stream ==
                  CU_STREAM_PER_THREAD,
          "PTDS prefix is captured before forwarding returns");
  require(driver.events.at(driver.recorded.front()).recorded_thread ==
              dispatch_thread,
          "PTDS completion event is recorded on the dispatch host thread");

  Result<CompletionToken> snapshot;
  std::thread query_thread([&] {
    snapshot = execution.current_token(kDevice, kStream);
  });
  query_thread.join();
  require(snapshot.ok() && snapshot.value.valid(),
          "another host thread can retrieve the eagerly captured PTDS token");
  require(driver.recorded.size() == 1 &&
              driver.events.at(driver.recorded.front()).recorded_thread ==
                  dispatch_thread,
          "cross-thread token retrieval does not record against the wrong PTDS");
  require(execution.release(snapshot.value).ok(), "release PTDS token");
  driver.complete_all();
  require(execution.shutdown().ok(), "shutdown after PTDS prefix test");
}

void test_destroy_success_context_restore_failure_does_not_retry_handle() {
  FakeDriver driver;
  OwnerExecution execution(driver);
  require(execution.register_owner(kDevice, kOwnerA, kContextA).ok(),
          "register cleanup owner");
  require(execution.register_stream(kDevice, kStream, kOwnerA, kNativeA).ok(),
          "register cleanup stream");
  require(execution.observe_native_submission(kDevice, kStream, kOwnerA,
                                              kNativeA, CUDA_SUCCESS)
              .ok(),
          "mark cleanup stream dirty");
  const auto token = execution.current_token(kDevice, kStream);
  require(token.ok(), "capture cleanup prefix");
  require(execution.release(token.value).ok(), "release cleanup prefix token");

  driver.fail_pop_after_destroy = true;
  const Status first = execution.shutdown();
  require(first.code == ErrorCode::context_restore_failed,
          "shutdown reports context restoration failure after destroy");
  require(driver.events.empty() && driver.destroy_calls == 1,
          "successfully destroyed CUDA event is retired despite failed pop");
  require(execution.shutdown().ok(),
          "remaining bookkeeping can close without retrying a dead handle");
  require(driver.destroy_calls == 1,
          "shutdown never destroys the same CUDA event handle twice");
}

}  // namespace

int main() {
  test_native_prefix_peer_fanin_and_context();
  test_dependency_cycle_and_stale_token_generation();
  test_completion_lifetime_no_premature_free();
  test_async_peer_failure_fail_stops_and_drains_all_events();
  test_synchronous_enqueue_error_is_drainable_and_context_restored();
  test_native_forward_is_atomic_with_peer_submission();
  test_event_capture_failure_uses_safe_stream_drain();
  test_per_thread_default_stream_is_sealed_on_forwarding_thread();
  test_destroy_success_context_restore_failure_does_not_retry_handle();
  std::cout << "owner execution host tests passed\n";
  return 0;
}
