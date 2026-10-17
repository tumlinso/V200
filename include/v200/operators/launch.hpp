#ifndef V200_OPERATORS_LAUNCH_HPP
#define V200_OPERATORS_LAUNCH_HPP

#include <cuda_runtime_api.h>

#if defined(__GNUC__)
#define V200_OPERATOR_API __attribute__((visibility("default")))
#else
#define V200_OPERATOR_API
#endif

/* The callback is the real cudaLaunchKernel entry resolved by the V200
 * frontdoor. The operator hook is called from the frontdoor's asynchronous
 * dispatch callback, so its returned work must join back to anchor_stream. */
typedef cudaError_t (CUDARTAPI *v200_operator_native_launch_fn)(
    const void *, dim3, dim3, void **, size_t, cudaStream_t);

#ifdef __cplusplus
extern "C" {
#endif

/* Return 0 only when this hook has made no CUDA calls and the caller should
 * forward the original launch once. Return 1 when handled; *status then holds
 * the CUDA result, including errors after a partial submission. */
V200_OPERATOR_API int v200_operator_try_launch(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    size_t shared_bytes, cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch, cudaError_t *status);

#ifdef __cplusplus
}
#endif

#endif
