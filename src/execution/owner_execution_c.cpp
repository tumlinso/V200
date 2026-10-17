#include "v200/execution/owner_execution_c.h"

#include "v200/execution/owner_execution.hpp"

#include <memory>
#include <new>
#include <vector>

namespace {

class CDriverDispatch final : public v200::execution::DriverDispatch {
 public:
  explicit CDriverDispatch(const v200_execution_driver_api_v1& api) : api_(api) {}

  CUresult ctx_push_current(CUcontext context) noexcept override {
    return static_cast<CUresult>(api_.ctx_push_current(api_.user_data, context));
  }
  CUresult ctx_pop_current(CUcontext* context) noexcept override {
    void* popped = nullptr;
    const CUresult result = static_cast<CUresult>(
        api_.ctx_pop_current(api_.user_data, &popped));
    *context = static_cast<CUcontext>(popped);
    return result;
  }
  CUresult event_create(CUevent* event, unsigned flags) noexcept override {
    void* created = nullptr;
    const CUresult result = static_cast<CUresult>(
        api_.event_create(api_.user_data, &created, flags));
    *event = static_cast<CUevent>(created);
    return result;
  }
  CUresult event_destroy(CUevent event) noexcept override {
    return static_cast<CUresult>(api_.event_destroy(api_.user_data, event));
  }
  CUresult event_record(CUevent event, CUstream stream) noexcept override {
    return static_cast<CUresult>(
        api_.event_record(api_.user_data, event, stream));
  }
  CUresult event_query(CUevent event) noexcept override {
    return static_cast<CUresult>(api_.event_query(api_.user_data, event));
  }
  CUresult event_synchronize(CUevent event) noexcept override {
    return static_cast<CUresult>(
        api_.event_synchronize(api_.user_data, event));
  }
  CUresult stream_synchronize(CUstream stream) noexcept override {
    return static_cast<CUresult>(
        api_.stream_synchronize(api_.user_data, stream));
  }
  CUresult stream_wait_event(CUstream stream, CUevent event,
                             unsigned flags) noexcept override {
    return static_cast<CUresult>(
        api_.stream_wait_event(api_.user_data, stream, event, flags));
  }

 private:
  v200_execution_driver_api_v1 api_;
};

}  // namespace

struct v200_execution_handle {
  explicit v200_execution_handle(const v200_execution_driver_api_v1& api)
      : driver(api), execution(driver) {}

  CDriverDispatch driver;
  v200::execution::OwnerExecution execution;
};

namespace {

using v200::execution::CompletionToken;
using v200::execution::ErrorCode;
using v200::execution::LogicalDeviceId;
using v200::execution::LogicalStreamId;
using v200::execution::OwnerId;
using v200::execution::Status;

v200_execution_status as_status(Status status) noexcept {
  return static_cast<v200_execution_status>(status.code);
}

CompletionToken as_cpp(v200_completion_token token) noexcept {
  return {token.slot, token.generation};
}

v200_completion_token as_c(CompletionToken token) noexcept {
  return {token.slot, token.generation};
}

bool valid(v200_execution_handle* handle) noexcept { return handle != nullptr; }

}  // namespace

extern "C" {

v200_execution_handle* v200_execution_create(
    const v200_execution_driver_api_v1* driver_api) {
  if (driver_api == nullptr ||
      driver_api->abi_version != V200_EXECUTION_ABI_VERSION ||
      driver_api->struct_size < sizeof(v200_execution_driver_api_v1) ||
      driver_api->ctx_push_current == nullptr ||
      driver_api->ctx_pop_current == nullptr ||
      driver_api->event_create == nullptr ||
      driver_api->event_destroy == nullptr ||
      driver_api->event_record == nullptr ||
      driver_api->event_query == nullptr ||
      driver_api->event_synchronize == nullptr ||
      driver_api->stream_synchronize == nullptr ||
      driver_api->stream_wait_event == nullptr) {
    return nullptr;
  }
  try {
    return new v200_execution_handle(*driver_api);
  } catch (...) {
    return nullptr;
  }
}

void v200_execution_destroy(v200_execution_handle* handle) { delete handle; }

int v200_execution_shutdown(v200_execution_handle* handle) {
  return valid(handle) ? as_status(handle->execution.shutdown())
                       : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_register_owner(v200_execution_handle* handle,
                                  v200_logical_device_id device,
                                  v200_owner_id owner, void* context) {
  return valid(handle)
             ? as_status(handle->execution.register_owner(
                   static_cast<LogicalDeviceId>(device),
                   static_cast<OwnerId>(owner), static_cast<CUcontext>(context)))
             : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_register_stream(v200_execution_handle* handle,
                                   v200_logical_device_id device,
                                   v200_logical_stream_id stream,
                                   v200_owner_id owner, void* native_stream) {
  return valid(handle)
             ? as_status(handle->execution.register_stream(
                   static_cast<LogicalDeviceId>(device),
                   static_cast<LogicalStreamId>(stream),
                   static_cast<OwnerId>(owner),
                   static_cast<CUstream>(native_stream)))
             : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_unregister_stream(v200_execution_handle* handle,
                                     v200_logical_device_id device,
                                     v200_logical_stream_id stream,
                                     v200_owner_id owner) {
  return valid(handle)
             ? as_status(handle->execution.unregister_stream(
                   static_cast<LogicalDeviceId>(device),
                   static_cast<LogicalStreamId>(stream),
                   static_cast<OwnerId>(owner)))
             : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_observe_native_submission(
    v200_execution_handle* handle, v200_logical_device_id device,
    v200_logical_stream_id stream, v200_owner_id owner, void* native_stream,
    int forwarded_driver_result) {
  return valid(handle)
             ? as_status(handle->execution.observe_native_submission(
                   static_cast<LogicalDeviceId>(device),
                   static_cast<LogicalStreamId>(stream),
                   static_cast<OwnerId>(owner),
                   static_cast<CUstream>(native_stream),
                   static_cast<CUresult>(forwarded_driver_result)))
             : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_forward_native(
    v200_execution_handle* handle, v200_logical_device_id device,
    v200_logical_stream_id stream, v200_owner_id owner, void* native_stream,
    v200_execution_enqueue_fn forward, void* user_data,
    int* forwarded_result_out) {
  if (!valid(handle) || forward == nullptr) return V200_EXECUTION_INVALID_ARGUMENT;
  int forwarded = CUDA_ERROR_INVALID_VALUE;
  const Status status = handle->execution.forward_native_submission(
      static_cast<LogicalDeviceId>(device),
      static_cast<LogicalStreamId>(stream), static_cast<OwnerId>(owner),
      static_cast<CUstream>(native_stream),
      [forward, user_data, &forwarded](CUstream stream_handle) {
        forwarded = forward(user_data, stream_handle);
        return static_cast<CUresult>(forwarded);
      });
  if (forwarded_result_out != nullptr) *forwarded_result_out = forwarded;
  return as_status(status);
}

int v200_execution_current_token(v200_execution_handle* handle,
                                 v200_logical_device_id device,
                                 v200_logical_stream_id stream,
                                 v200_completion_token* token_out) {
  if (!valid(handle) || token_out == nullptr) return V200_EXECUTION_INVALID_ARGUMENT;
  const auto result = handle->execution.current_token(
      static_cast<LogicalDeviceId>(device),
      static_cast<LogicalStreamId>(stream));
  if (result) *token_out = as_c(result.value);
  return as_status(result.status);
}

int v200_execution_current_device_token(v200_execution_handle* handle,
                                        v200_logical_device_id device,
                                        v200_completion_token* token_out) {
  if (!valid(handle) || token_out == nullptr) return V200_EXECUTION_INVALID_ARGUMENT;
  const auto result = handle->execution.current_device_token(
      static_cast<LogicalDeviceId>(device));
  if (result) *token_out = as_c(result.value);
  return as_status(result.status);
}

int v200_execution_is_complete(v200_execution_handle* handle,
                               v200_completion_token token,
                               int* complete_out, int* driver_result_out) {
  if (!valid(handle) || complete_out == nullptr) return V200_EXECUTION_INVALID_ARGUMENT;
  const auto result = handle->execution.is_complete(as_cpp(token));
  *complete_out = result.complete ? 1 : 0;
  if (driver_result_out != nullptr) *driver_result_out = result.status.driver_result;
  return as_status(result.status);
}

int v200_execution_wait(v200_execution_handle* handle,
                        v200_completion_token token,
                        int* driver_result_out) {
  if (!valid(handle)) return V200_EXECUTION_INVALID_ARGUMENT;
  const Status status = handle->execution.wait(as_cpp(token));
  if (driver_result_out != nullptr) *driver_result_out = status.driver_result;
  return as_status(status);
}

int v200_execution_retain(v200_execution_handle* handle,
                          v200_completion_token token) {
  return valid(handle) ? as_status(handle->execution.retain(as_cpp(token)))
                       : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_release(v200_execution_handle* handle,
                           v200_completion_token token) {
  return valid(handle) ? as_status(handle->execution.release(as_cpp(token)))
                       : V200_EXECUTION_INVALID_ARGUMENT;
}

int v200_execution_wait_on_stream(v200_execution_handle* handle,
                                  v200_logical_device_id device,
                                  v200_logical_stream_id stream,
                                  v200_owner_id owner,
                                  v200_completion_token dependency,
                                  v200_completion_token* result_out,
                                  int* driver_result_out) {
  if (!valid(handle)) return V200_EXECUTION_INVALID_ARGUMENT;
  const auto result = handle->execution.wait_on_stream(
      static_cast<LogicalDeviceId>(device),
      static_cast<LogicalStreamId>(stream), static_cast<OwnerId>(owner),
      as_cpp(dependency));
  if (result_out != nullptr && result.completion.valid()) {
    *result_out = as_c(result.completion);
  }
  if (driver_result_out != nullptr) *driver_result_out = result.status.driver_result;
  return as_status(result.status);
}

int v200_execution_submit(v200_execution_handle* handle,
                          v200_logical_device_id device,
                          v200_logical_stream_id stream, v200_owner_id owner,
                          const v200_completion_token* dependencies,
                          size_t dependency_count,
                          v200_execution_enqueue_fn enqueue,
                          void* user_data,
                          v200_completion_token* result_out,
                          int* driver_result_out) {
  if (!valid(handle) || enqueue == nullptr ||
      (dependency_count != 0 && dependencies == nullptr)) {
    return V200_EXECUTION_INVALID_ARGUMENT;
  }
  std::vector<CompletionToken> cpp_dependencies;
  try {
    cpp_dependencies.reserve(dependency_count);
    for (size_t i = 0; i < dependency_count; ++i) {
      cpp_dependencies.push_back(as_cpp(dependencies[i]));
    }
  } catch (...) {
    return V200_EXECUTION_UNAVAILABLE;
  }
  const auto result = handle->execution.submit(
      static_cast<LogicalDeviceId>(device),
      static_cast<LogicalStreamId>(stream), static_cast<OwnerId>(owner),
      cpp_dependencies.data(), cpp_dependencies.size(),
      [enqueue, user_data](CUstream native_stream) {
        return static_cast<CUresult>(enqueue(user_data, native_stream));
      });
  if (result_out != nullptr && result.completion.valid()) {
    *result_out = as_c(result.completion);
  }
  if (driver_result_out != nullptr) *driver_result_out = result.status.driver_result;
  return as_status(result.status);
}

}  // extern "C"
