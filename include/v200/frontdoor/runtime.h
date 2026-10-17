#ifndef V200_FRONTDOOR_RUNTIME_H
#define V200_FRONTDOOR_RUNTIME_H

#include <stdint.h>

#if defined(__GNUC__)
#define V200_FRONTDOOR_API __attribute__((visibility("default")))
#else
#define V200_FRONTDOOR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct v200_execution_handle v200_execution_handle;
typedef uint64_t v200_logical_device_id;
typedef uint32_t v200_owner_id;

/* ABI shared with the optional V200 execution module. */
#define V200_FRONTDOOR_ABI_VERSION 1u

/* Return a stable process-local stream identity. CUDA stream handles remain native. */
V200_FRONTDOOR_API uint64_t v200_frontdoor_stream_id(const void *stream);

/* Release a non-default stream identity after its native stream is destroyed. */
V200_FRONTDOOR_API void v200_frontdoor_forget_stream(const void *stream);

/* 0 means transparent native mode; 1 means anchor presentation; 2 means pair memory mode. */
V200_FRONTDOOR_API uint32_t v200_frontdoor_mode(void);

/* Resolve symbols from the exact already-loaded CUDA DSO, including one loaded
 * RTLD_LOCAL by a backend plugin. Handles are kept alive until process exit. */
V200_FRONTDOOR_API void *v200_frontdoor_resolve_driver_symbol(const char *name);
V200_FRONTDOOR_API void *v200_frontdoor_resolve_runtime_symbol(const char *name);
/* Append one same-process owner ledger receipt when V200_PROFILE_LOG is set. */
V200_FRONTDOOR_API void v200_frontdoor_snapshot(const char *event_name);

/* Attach the launcher-owned execution ledger. The launcher registers physical
 * owners first; the frontdoor registers native CUDA streams lazily. Ownership
 * of `handle` remains with the caller, which must detach before destruction. */
V200_FRONTDOOR_API int v200_frontdoor_attach_execution(
    v200_execution_handle *handle, v200_logical_device_id logical_device,
    v200_owner_id owner_id);
V200_FRONTDOOR_API void v200_frontdoor_detach_execution(
    v200_execution_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
