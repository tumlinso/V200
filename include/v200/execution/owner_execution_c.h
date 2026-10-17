#ifndef V200_EXECUTION_OWNER_EXECUTION_C_H
#define V200_EXECUTION_OWNER_EXECUTION_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V200_EXECUTION_ABI_VERSION 1u

typedef struct v200_execution_handle v200_execution_handle;
typedef uint64_t v200_logical_device_id;
typedef uint64_t v200_logical_stream_id;
typedef uint32_t v200_owner_id;

typedef struct v200_completion_token {
  uint64_t slot;
  uint64_t generation;
} v200_completion_token;

typedef enum v200_execution_status {
  V200_EXECUTION_OK = 0,
  V200_EXECUTION_INVALID_ARGUMENT,
  V200_EXECUTION_UNAVAILABLE,
  V200_EXECUTION_DUPLICATE_BINDING,
  V200_EXECUTION_UNKNOWN_OWNER,
  V200_EXECUTION_UNKNOWN_STREAM,
  V200_EXECUTION_STALE_TOKEN,
  V200_EXECUTION_DEPENDENCY_CYCLE,
  V200_EXECUTION_DRIVER_FAILURE,
  V200_EXECUTION_STREAM_POISONED,
  V200_EXECUTION_CONTEXT_RESTORE_FAILED,
  V200_EXECUTION_RESOURCE_BUSY,
  V200_EXECUTION_CLOSED
} v200_execution_status;

// Function pointers use opaque CUDA handles so this header is usable from C
// without exposing CUDA typedefs. Each function returns the CUDA Driver API
// CUresult value (0 is success; 600 is CUDA_ERROR_NOT_READY).
typedef struct v200_execution_driver_api_v1 {
  uint32_t abi_version;
  size_t struct_size;
  void* user_data;
  int (*ctx_push_current)(void* user_data, void* context);
  int (*ctx_pop_current)(void* user_data, void** context_out);
  int (*event_create)(void* user_data, void** event_out, unsigned flags);
  int (*event_destroy)(void* user_data, void* event);
  int (*event_record)(void* user_data, void* event, void* stream);
  int (*event_query)(void* user_data, void* event);
  int (*event_synchronize)(void* user_data, void* event);
  int (*stream_synchronize)(void* user_data, void* stream);
  int (*stream_wait_event)(void* user_data, void* stream, void* event,
                           unsigned flags);
} v200_execution_driver_api_v1;

typedef int (*v200_execution_enqueue_fn)(void* user_data, void* native_stream);

v200_execution_handle* v200_execution_create(
    const v200_execution_driver_api_v1* driver_api);
void v200_execution_destroy(v200_execution_handle* handle);
int v200_execution_shutdown(v200_execution_handle* handle);

int v200_execution_register_owner(v200_execution_handle* handle,
                                  v200_logical_device_id device,
                                  v200_owner_id owner, void* context);
int v200_execution_register_stream(v200_execution_handle* handle,
                                   v200_logical_device_id device,
                                   v200_logical_stream_id stream,
                                   v200_owner_id owner, void* native_stream);
int v200_execution_unregister_stream(v200_execution_handle* handle,
                                     v200_logical_device_id device,
                                     v200_logical_stream_id stream,
                                     v200_owner_id owner);

// Call after a successfully forwarded async operation. Native observation is
// deliberately limited to one owner per logical stream; use wait_on_stream or
// submit when work must join another physical owner.
int v200_execution_observe_native_submission(
    v200_execution_handle* handle, v200_logical_device_id device,
    v200_logical_stream_id stream, v200_owner_id owner, void* native_stream,
    int forwarded_driver_result);
// Prefer this call for intercepted native work: forward runs under the same
// adapter lock used by submit/wait, preventing a peer from overtaking the
// operation before its completion prefix is published. The callback must call
// the real underlying CUDA entry point and must not re-enter this handle.
int v200_execution_forward_native(
    v200_execution_handle* handle, v200_logical_device_id device,
    v200_logical_stream_id stream, v200_owner_id owner, void* native_stream,
    v200_execution_enqueue_fn forward, void* user_data,
    int* forwarded_result_out);

int v200_execution_current_token(v200_execution_handle* handle,
                                 v200_logical_device_id device,
                                 v200_logical_stream_id stream,
                                 v200_completion_token* token_out);
int v200_execution_current_device_token(v200_execution_handle* handle,
                                        v200_logical_device_id device,
                                        v200_completion_token* token_out);
int v200_execution_is_complete(v200_execution_handle* handle,
                               v200_completion_token token,
                               int* complete_out, int* driver_result_out);
int v200_execution_wait(v200_execution_handle* handle,
                        v200_completion_token token,
                        int* driver_result_out);
int v200_execution_retain(v200_execution_handle* handle,
                          v200_completion_token token);
int v200_execution_release(v200_execution_handle* handle,
                           v200_completion_token token);

int v200_execution_wait_on_stream(v200_execution_handle* handle,
                                  v200_logical_device_id device,
                                  v200_logical_stream_id stream,
                                  v200_owner_id owner,
                                  v200_completion_token dependency,
                                  v200_completion_token* result_out,
                                  int* driver_result_out);
int v200_execution_submit(v200_execution_handle* handle,
                          v200_logical_device_id device,
                          v200_logical_stream_id stream, v200_owner_id owner,
                          const v200_completion_token* dependencies,
                          size_t dependency_count,
                          v200_execution_enqueue_fn enqueue,
                          void* user_data,
                          v200_completion_token* result_out,
                          int* driver_result_out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // V200_EXECUTION_OWNER_EXECUTION_C_H
