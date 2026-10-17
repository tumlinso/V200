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
    V200_OPERATOR_MMVQ_Q6K_PLAIN = 3,
    V200_OPERATOR_MMVQ_Q6K_GATE_FUSED = 4,
};

typedef struct v200_operator_mmvq_kernel_descriptor {
    std::uint32_t family;
    std::uint32_t quant_block_elements;
    std::size_t quant_block_bytes;
    std::uint32_t minimum_expert_columns;
    std::uint32_t output_rows_per_block;
} v200_operator_mmvq_kernel_descriptor;

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

/* Exact source-name classification and quantized row geometry. Returns zero
 * for any unrecognized kernel name. This metadata path performs no CUDA calls. */
V200_OPERATOR_API int v200_operator_describe_mmvq_kernel(
    const char *registered_name,
    v200_operator_mmvq_kernel_descriptor *descriptor);

/* Classify one exact kernel launch without submitting work. Returns one only
 * when a supported Q4_K or Q6_K expert bank is wholly resident on one owner,
 * its pointers satisfy the current managed-memory contract, and its active
 * weight payload clears the conservative routing threshold. `owner_index`
 * uses profile.owner_device[0/1]. */
V200_OPERATOR_API int v200_operator_inspect_mmvq_bank(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    std::uint32_t *owner_index, std::uint32_t *family,
    std::size_t *bank_bytes, std::size_t *active_weight_bytes);

}

#endif
