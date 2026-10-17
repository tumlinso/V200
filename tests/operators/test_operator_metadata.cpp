#include "v200/operators/launch.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr const char *q4_plain =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q4_gate =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q6_plain =
    "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q6_gate =
    "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q4_plain_small_k =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q4_gate_small_k =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q6_plain_small_k =
    "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q6_gate_small_k =
    "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";

void check(const char *name, uint32_t expected_family,
           size_t expected_block_bytes, uint32_t expected_rows_per_block = 1) {
    v200_operator_mmvq_kernel_descriptor descriptor{};
    assert(v200_operator_describe_mmvq_kernel(name, &descriptor) == 1);
    assert(descriptor.family == expected_family);
    assert(descriptor.quant_block_elements == 256);
    assert(descriptor.quant_block_bytes == expected_block_bytes);
    assert(descriptor.minimum_expert_columns == 768);
    assert(descriptor.output_rows_per_block == expected_rows_per_block);
}

}  // namespace

int main() {
    check(q4_plain, V200_OPERATOR_MMVQ_Q4K_PLAIN, 144);
    check(q4_gate, V200_OPERATOR_MMVQ_Q4K_GATE_FUSED, 144);
    check(q6_plain, V200_OPERATOR_MMVQ_Q6K_PLAIN, 210);
    check(q6_gate, V200_OPERATOR_MMVQ_Q6K_GATE_FUSED, 210);
    check(q4_plain_small_k, V200_OPERATOR_MMVQ_Q4K_PLAIN, 144, 4);
    check(q4_gate_small_k, V200_OPERATOR_MMVQ_Q4K_GATE_FUSED, 144, 4);
    check(q6_plain_small_k, V200_OPERATOR_MMVQ_Q6K_PLAIN, 210, 4);
    check(q6_gate_small_k, V200_OPERATOR_MMVQ_Q6K_GATE_FUSED, 210, 4);

    v200_operator_mmvq_kernel_descriptor descriptor{};
    assert(v200_operator_describe_mmvq_kernel("mul_mat_vec_q", &descriptor) == 0);
    assert(v200_operator_describe_mmvq_kernel(nullptr, &descriptor) == 0);
    assert(v200_operator_describe_mmvq_kernel(q6_plain, nullptr) == 0);
    return 0;
}
