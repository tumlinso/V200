#ifndef V200_OPERATORS_LAUNCH_HPP
#define V200_OPERATORS_LAUNCH_HPP

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
#define V200_OPERATOR_API __attribute__((visibility("default")))
#else
#define V200_OPERATOR_API
#endif

/* The frontdoor supplies the real cudaLaunchKernel entry resolved from the
 * already-loaded CUDA runtime. The callback must return with all peer work
 * joined to the supplied anchor stream. */
typedef cudaError_t (CUDARTAPI *v200_operator_native_launch_fn)(
    const void *, dim3, dim3, void **, size_t, cudaStream_t);

enum {
    V200_OPERATOR_MMVQ_Q4K_PLAIN = 1,
    V200_OPERATOR_MMVQ_Q4K_GATE_FUSED = 2,
};

/* Return the total number of exact registered Q4_K MMVQ host stubs. If `out`
 * is non-null, fill up to `capacity` entries. This is metadata-only. */
extern "C" {

/* Return 0 only when the hook made no CUDA calls and the frontdoor should
 * forward the original launch. Return 1 when handled, including errors after
 * any partial submission. */
V200_OPERATOR_API int v200_operator_try_launch(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    size_t shared_bytes, cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch, cudaError_t *status);

V200_OPERATOR_API std::size_t v200_operator_list_mmvq_host_functions(
    const void **out, std::size_t capacity);

/* Classify one exact kernel launch without submitting work. Returns one only
 * when the supported Q4_K expert bank is wholly resident on one proven owner,
 * its pointers satisfy the current managed-memory contract, and its active
 * weight payload clears the conservative routing threshold. `owner_index`
 * uses profile.owner_device[0/1]. */
V200_OPERATOR_API int v200_operator_inspect_mmvq_bank(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    std::uint32_t *owner_index, std::uint32_t *family,
    std::size_t *bank_bytes, std::size_t *active_weight_bytes);

}

#endif
