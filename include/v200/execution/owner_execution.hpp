#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace v200::execution {

using LogicalDeviceId = std::uint64_t;
using LogicalStreamId = std::uint64_t;
using OwnerId = std::uint32_t;

enum class ErrorCode : std::uint8_t {
  success = 0,
  invalid_argument,
  unavailable,
  duplicate_binding,
  unknown_owner,
  unknown_stream,
  stale_token,
  dependency_cycle,
  driver_failure,
  stream_poisoned,
  context_restore_failed,
  resource_busy,
  closed,
};

struct Status {
  ErrorCode code{ErrorCode::success};
  CUresult driver_result{CUDA_SUCCESS};

  [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::success; }
  explicit operator bool() const noexcept { return ok(); }

  static Status success() noexcept { return {}; }
};

template <class T>
struct Result {
  Status status{};
  T value{};

  [[nodiscard]] bool ok() const noexcept { return status.ok(); }
  explicit operator bool() const noexcept { return ok(); }
};

struct CompletionToken {
  std::uint64_t slot{0};
  std::uint64_t generation{0};

  [[nodiscard]] bool valid() const noexcept { return slot != 0 && generation != 0; }
  friend bool operator==(CompletionToken a, CompletionToken b) noexcept {
    return a.slot == b.slot && a.generation == b.generation;
  }
  friend bool operator!=(CompletionToken a, CompletionToken b) noexcept {
    return !(a == b);
  }
};

struct QueryResult {
  Status status{};
  bool complete{false};
};

// The caller owns one reference returned by current_token/current_device_token
// and one reference returned by submit/wait_on_stream. Tokens remain immutable
// prefix snapshots. Retain before storing a token beyond the current call.
class CompletionProvider {
 public:
  virtual ~CompletionProvider() = default;
  virtual Status retain(CompletionToken token) noexcept = 0;
  virtual QueryResult is_complete(CompletionToken token) noexcept = 0;
  virtual Status wait(CompletionToken token) noexcept = 0;
  virtual Status release(CompletionToken token) noexcept = 0;
};

// All calls are made with the associated owner's CUDA context current. The
// injected dispatch makes host tests independent of libcuda and hardware.
class DriverDispatch {
 public:
  virtual ~DriverDispatch() = default;
  virtual CUresult ctx_push_current(CUcontext context) noexcept = 0;
  virtual CUresult ctx_pop_current(CUcontext* context) noexcept = 0;
  virtual CUresult event_create(CUevent* event, unsigned flags) noexcept = 0;
  virtual CUresult event_destroy(CUevent event) noexcept = 0;
  virtual CUresult event_record(CUevent event, CUstream stream) noexcept = 0;
  virtual CUresult event_query(CUevent event) noexcept = 0;
  virtual CUresult event_synchronize(CUevent event) noexcept = 0;
  virtual CUresult stream_synchronize(CUstream stream) noexcept = 0;
  virtual CUresult stream_wait_event(CUstream stream, CUevent event,
                                    unsigned flags) noexcept = 0;
};

using EnqueueCallback = std::function<CUresult(CUstream)>;

struct SubmissionResult {
  Status status{};
  CompletionToken completion{};
};

class OwnerExecution final : public CompletionProvider {
 public:
  explicit OwnerExecution(DriverDispatch& driver);
  ~OwnerExecution() override;

  OwnerExecution(const OwnerExecution&) = delete;
  OwnerExecution& operator=(const OwnerExecution&) = delete;

  Status register_owner(LogicalDeviceId device, OwnerId owner,
                       CUcontext context) noexcept;
  Status register_stream(LogicalDeviceId device, LogicalStreamId stream,
                         OwnerId owner, CUstream native_stream) noexcept;
  Status unregister_stream(LogicalDeviceId device, LogicalStreamId stream,
                           OwnerId owner) noexcept;

  // For a single-anchor native path that is serialized by an external gate,
  // call this after a forwarded asynchronous CUDA operation. It marks that
  // owner stream dirty; the next token snapshot records one private event
  // after the whole native prefix. Concurrent wrappers should instead use
  // forward_native_submission so dispatch and prefix publication are atomic.
  // If forwarded_result is an error, the stream enters fail-stop state while
  // still remaining drainable.
  Status observe_native_submission(LogicalDeviceId device,
                                   LogicalStreamId stream, OwnerId owner,
                                   CUstream native_stream,
                                   CUresult forwarded_result) noexcept;

  // Atomically forward one native operation and publish its prefix while
  // holding the same per-adapter lock used by owner submissions. The callback
  // must invoke the real underlying CUDA entry point and must not re-enter this
  // OwnerExecution object.
  Status forward_native_submission(LogicalDeviceId device,
                                   LogicalStreamId stream, OwnerId owner,
                                   CUstream native_stream,
                                   const EnqueueCallback& forward) noexcept;

  // Enqueue an explicit owner operation. Dependencies are immutable earlier
  // prefixes; work is ordered after every owner already touched by this logical
  // stream, the callback runs under the configured context, and its returned
  // token covers the resulting full prefix.
  SubmissionResult submit(LogicalDeviceId device, LogicalStreamId stream,
                         OwnerId owner,
                         const CompletionToken* dependencies,
                         std::size_t dependency_count,
                         const EnqueueCallback& enqueue) noexcept;

  // Enqueue a logical stream wait and return a token for the new full prefix.
  SubmissionResult wait_on_stream(LogicalDeviceId device,
                                  LogicalStreamId stream, OwnerId owner,
                                  CompletionToken dependency) noexcept;

  Result<CompletionToken> current_token(LogicalDeviceId device,
                                       LogicalStreamId stream) noexcept;
  Result<CompletionToken> current_device_token(
      LogicalDeviceId device) noexcept;

  Status collect() noexcept;
  Status shutdown() noexcept;

  Status retain(CompletionToken token) noexcept override;
  QueryResult is_complete(CompletionToken token) noexcept override;
  Status wait(CompletionToken token) noexcept override;
  Status release(CompletionToken token) noexcept override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace v200::execution
