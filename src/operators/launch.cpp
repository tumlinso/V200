#include "v200/operators/launch.hpp"

#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"
#include "v200/operators/mmvq_plan.hpp"
#include "v200/operators/pair_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unistd.h>

namespace {

constexpr const char *q4k_mmvq_name =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr const char *q4k_mmvq_gate_name =
    "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
constexpr size_t q4k_block_bytes = 144;
constexpr uint32_t q4k_block_elements = 256;
constexpr size_t q8_1_block_bytes = 36;
constexpr uint32_t q8_1_block_elements = 32;
constexpr size_t minimum_peer_active_weight_bytes = 4 * 1024 * 1024;
constexpr size_t max_verify_output_bytes = 1024 * 1024;

struct FusionArgs {
    const void *x_bias;
    const void *gate;
    const void *gate_bias;
    const void *x_scale;
    const void *gate_scale;
    int32_t glu_op;
};
static_assert(sizeof(FusionArgs) == 48, "pinned llama fusion-argument ABI changed");

/* ABI of the one supported compiled kernel specialization. The device-name
 * registration string pins the template arguments; this struct only copies
 * the corresponding runtime argument bytes. */
struct Q4KMmvqArgs {
    const void *vx;
    const void *vy;
    const int32_t *ids;
    FusionArgs fusion;
    float *dst;
    uint32_t ncols_x;
    uint3 nchannels_y;
    uint32_t stride_row_x;
    uint32_t stride_col_y;
    uint32_t stride_col_dst;
    uint3 channel_ratio;
    uint32_t stride_channel_x;
    uint32_t stride_channel_y;
    uint32_t stride_channel_dst;
    uint3 sample_ratio;
    uint32_t stride_sample_x;
    uint32_t stride_sample_y;
    uint32_t stride_sample_dst;
    uint32_t ids_stride;
};
static_assert(offsetof(Q4KMmvqArgs, ids_stride) + sizeof(uint32_t) == sizeof(Q4KMmvqArgs),
              "unexpected Q4_K MMVQ host argument packing");

struct Registry {
    std::mutex mutex;
    std::unordered_map<const void *, std::string> names;
};

struct OperatorCounters {
    std::atomic<uint64_t> matched_calls{0};
    std::atomic<uint64_t> anchor_local_calls{0};
    std::atomic<uint64_t> peer_owner_calls{0};
    std::atomic<uint64_t> split_calls{0};
    std::atomic<uint64_t> anchor_rows{0};
    std::atomic<uint64_t> peer_rows{0};
    std::atomic<uint64_t> verified_rows{0};
    std::atomic<uint64_t> verification_failures{0};
    std::atomic<uint64_t> expert_matched_calls{0};
    std::atomic<uint64_t> expert_anchor_local_calls{0};
    std::atomic<uint64_t> expert_peer_owner_calls{0};
    std::atomic<uint64_t> expert_small_peer_fallback_calls{0};
    std::atomic<uint64_t> expert_rows{0};
    std::atomic<uint64_t> expert_anchor_rows{0};
    std::atomic<uint64_t> expert_peer_rows{0};
    std::atomic<uint64_t> expert_verified_calls{0};
    std::atomic<uint64_t> expert_verified_outputs{0};
    std::atomic<uint64_t> expert_plain_verified_calls{0};
    std::atomic<uint64_t> expert_plain_verified_outputs{0};
    std::atomic<uint64_t> expert_fused_verified_calls{0};
    std::atomic<uint64_t> expert_fused_verified_outputs{0};
    std::atomic<uint64_t> expert_verification_failures{0};
};

void write_operator_event(const char *event, const char *route,
                          uint64_t matched_calls, uint64_t anchor_local_calls,
                          uint64_t peer_owner_calls, uint64_t split_calls,
                          uint64_t anchor_rows, uint64_t peer_rows,
                          uint64_t verified_rows, float max_abs_error,
                          int status, uint64_t expert_matched_calls = 0,
                          uint64_t expert_anchor_local_calls = 0,
                          uint64_t expert_peer_owner_calls = 0,
                          uint64_t expert_small_peer_fallback_calls = 0,
                          uint64_t expert_rows = 0,
                          uint64_t expert_anchor_rows = 0,
                          uint64_t expert_peer_rows = 0,
                          uint64_t expert_verified_calls = 0,
                          uint64_t expert_verified_outputs = 0,
                          uint64_t expert_plain_verified_calls = 0,
                          uint64_t expert_plain_verified_outputs = 0,
                          uint64_t expert_fused_verified_calls = 0,
                          uint64_t expert_fused_verified_outputs = 0,
                          uint64_t expert_verification_failures = 0) noexcept {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    char line[1536];
    const int count = std::snprintf(
        line, sizeof(line),
        "{\"schema\":\"v200-operator-ledger/v1\",\"event\":\"%s\","
        "\"operation\":\"q4k_mmvq\",\"route\":\"%s\","
        "\"matched_calls\":%llu,\"anchor_local_calls\":%llu,"
        "\"peer_owner_calls\":%llu,\"split_calls\":%llu,"
        "\"anchor_rows\":%llu,\"peer_rows\":%llu,"
        "\"verified_rows\":%llu,\"max_abs_error\":%.9g,\"status\":%d,"
        "\"expert_matched_calls\":%llu,\"expert_anchor_local_calls\":%llu,"
        "\"expert_peer_owner_calls\":%llu,\"expert_small_peer_fallback_calls\":%llu,"
        "\"expert_rows\":%llu,\"expert_anchor_rows\":%llu,"
        "\"expert_peer_rows\":%llu,\"expert_verified_calls\":%llu,"
        "\"expert_verified_outputs\":%llu,"
        "\"expert_plain_verified_calls\":%llu,"
        "\"expert_plain_verified_outputs\":%llu,"
        "\"expert_fused_verified_calls\":%llu,"
        "\"expert_fused_verified_outputs\":%llu,"
        "\"expert_verification_failures\":%llu}\n",
        event, route,
        static_cast<unsigned long long>(matched_calls),
        static_cast<unsigned long long>(anchor_local_calls),
        static_cast<unsigned long long>(peer_owner_calls),
        static_cast<unsigned long long>(split_calls),
        static_cast<unsigned long long>(anchor_rows),
        static_cast<unsigned long long>(peer_rows),
        static_cast<unsigned long long>(verified_rows), max_abs_error, status,
        static_cast<unsigned long long>(expert_matched_calls),
        static_cast<unsigned long long>(expert_anchor_local_calls),
        static_cast<unsigned long long>(expert_peer_owner_calls),
        static_cast<unsigned long long>(expert_small_peer_fallback_calls),
        static_cast<unsigned long long>(expert_rows),
        static_cast<unsigned long long>(expert_anchor_rows),
        static_cast<unsigned long long>(expert_peer_rows),
        static_cast<unsigned long long>(expert_verified_calls),
        static_cast<unsigned long long>(expert_verified_outputs),
        static_cast<unsigned long long>(expert_plain_verified_calls),
        static_cast<unsigned long long>(expert_plain_verified_outputs),
        static_cast<unsigned long long>(expert_fused_verified_calls),
        static_cast<unsigned long long>(expert_fused_verified_outputs),
        static_cast<unsigned long long>(expert_verification_failures));
    if (count <= 0 || static_cast<size_t>(count) >= sizeof(line)) return;

    static std::mutex *write_mutex = new std::mutex();
    std::lock_guard<std::mutex> lock(*write_mutex);
    const int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return;
    size_t written = 0;
    while (written < static_cast<size_t>(count)) {
        const ssize_t result = write(fd, line + written, static_cast<size_t>(count) - written);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) break;
        written += static_cast<size_t>(result);
    }
    close(fd);
}

OperatorCounters &operator_counters() {
    static OperatorCounters counters;
    static const bool exit_report_registered = [] {
        return std::atexit([] {
            OperatorCounters &final = operator_counters();
            write_operator_event(
                "q4k_mmvq_process_exit", "summary",
                final.matched_calls.load(), final.anchor_local_calls.load(),
                final.peer_owner_calls.load(), final.split_calls.load(),
                final.anchor_rows.load(), final.peer_rows.load(),
                final.verified_rows.load(), -1.0f,
                final.verification_failures.load() == 0 ? 0 : 1,
                final.expert_matched_calls.load(),
                final.expert_anchor_local_calls.load(),
                final.expert_peer_owner_calls.load(),
                final.expert_small_peer_fallback_calls.load(),
                final.expert_rows.load(), final.expert_anchor_rows.load(),
                final.expert_peer_rows.load(), final.expert_verified_calls.load(),
                final.expert_verified_outputs.load(),
                final.expert_plain_verified_calls.load(),
                final.expert_plain_verified_outputs.load(),
                final.expert_fused_verified_calls.load(),
                final.expert_fused_verified_outputs.load(),
                final.expert_verification_failures.load());
        }) == 0;
    }();
    (void)exit_report_registered;
    return counters;
}

Registry &registry() {
    // CUDA can register dependency kernels before other DSO dynamic
    // initializers. A function-local registry is initialized on first use.
    static Registry *value = new Registry();
    return *value;
}

std::string registered_kernel_name(const void *host_function) {
    Registry &registered = registry();
    std::lock_guard<std::mutex> lock(registered.mutex);
    const auto found = registered.names.find(host_function);
    return found == registered.names.end() ? std::string{} : found->second;
}

bool all_null(const FusionArgs &fusion) noexcept {
    return fusion.x_bias == nullptr && fusion.gate == nullptr &&
           fusion.gate_bias == nullptr && fusion.x_scale == nullptr &&
           fusion.gate_scale == nullptr;
}

bool query_pair_managed_range(const void *pointer, size_t bytes,
                              v200_allocation_info *out) noexcept {
    if (pointer == nullptr || bytes == 0) return false;
    v200_allocation_info allocation{};
    allocation.abi_version = V200_MEMORY_ABI_VERSION;
    allocation.struct_size = sizeof(allocation);
    if (v200_memory_query(pointer, &allocation) != CUDA_SUCCESS ||
        allocation.base == 0 || allocation.requested_bytes == 0) {
        return false;
    }
    const uintptr_t base = static_cast<uintptr_t>(allocation.base);
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address < base) return false;
    const uintptr_t offset = address - base;
    if (offset > allocation.requested_bytes ||
        bytes > allocation.requested_bytes - offset) {
        return false;
    }
    if (out != nullptr) *out = allocation;
    return true;
}

bool is_pair_managed_range(const void *pointer, size_t bytes) noexcept {
    return query_pair_managed_range(pointer, bytes, nullptr);
}

bool ranges_overlap(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) noexcept {
    const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
    if (left_bytes > UINTPTR_MAX - left_begin || right_bytes > UINTPTR_MAX - right_begin)
        return true;
    const uintptr_t left_end = left_begin + left_bytes;
    const uintptr_t right_end = right_begin + right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

bool checked_multiply(size_t left, size_t right, size_t *out) noexcept {
    if (out == nullptr || (left != 0 && right > SIZE_MAX / left)) return false;
    *out = left * right;
    return true;
}

bool checked_add(size_t left, size_t right, size_t *out) noexcept {
    if (out == nullptr || right > SIZE_MAX - left) return false;
    *out = left + right;
    return true;
}

bool allocation_contains_range(const v200_allocation_info &allocation,
                               uintptr_t address, size_t bytes) noexcept {
    if (allocation.base == 0 || allocation.requested_bytes == 0 || bytes == 0)
        return false;
    const uintptr_t base = static_cast<uintptr_t>(allocation.base);
    if (address < base) return false;
    const uintptr_t offset = address - base;
    return offset <= allocation.requested_bytes &&
           bytes <= allocation.requested_bytes - offset;
}

bool owner_for_contiguous_range(uintptr_t address, size_t bytes,
                                const v200_allocation_info &allocation,
                                const v200_memory_pair_profile &profile,
                                uint32_t *owner_index) noexcept {
    if (owner_index == nullptr || bytes == 0 ||
        allocation.abi_version != V200_MEMORY_ABI_VERSION ||
        allocation.struct_size < sizeof(v200_allocation_info) ||
        allocation.mapped_bytes == 0 ||
        allocation.requested_bytes > allocation.mapped_bytes ||
        allocation.owner_bytes[0] > allocation.mapped_bytes ||
        allocation.owner_bytes[1] > allocation.mapped_bytes ||
        allocation.owner_bytes[0] > allocation.mapped_bytes - allocation.owner_bytes[1] ||
        allocation.owner_bytes[0] + allocation.owner_bytes[1] != allocation.mapped_bytes ||
        !allocation_contains_range(allocation, address, bytes)) {
        return false;
    }

    if (allocation.extent_count == 1) {
        const bool owner0 = allocation.owner_bytes[0] == allocation.mapped_bytes &&
                            allocation.owner_bytes[1] == 0;
        const bool owner1 = allocation.owner_bytes[1] == allocation.mapped_bytes &&
                            allocation.owner_bytes[0] == 0;
        if (!owner0 && !owner1) return false;
        *owner_index = owner0 ? 0u : 1u;
        return true;
    }
    if (allocation.extent_count != 2 || allocation.owner_bytes[0] == 0 ||
        allocation.owner_bytes[1] == 0 ||
        (allocation.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_KNOWN) == 0 ||
        profile.abi_version != V200_MEMORY_ABI_VERSION ||
        profile.struct_size < sizeof(v200_memory_pair_profile) ||
        profile.owner_device[0] == profile.owner_device[1] ||
        profile.default_owner != profile.owner_device[0] ||
        profile.pair_access_qualified == 0 || profile.qualification_id[0] == '\0') {
        return false;
    }

    const uint32_t first_owner =
        (allocation.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_1) != 0
            ? 1u : 0u;
    const uint32_t second_owner = 1u - first_owner;
    const uintptr_t base = static_cast<uintptr_t>(allocation.base);
    if (allocation.owner_bytes[first_owner] > UINTPTR_MAX - base ||
        bytes > UINTPTR_MAX - address) {
        return false;
    }
    const uintptr_t boundary = base + allocation.owner_bytes[first_owner];
    const uintptr_t end = address + bytes;
    if (end <= boundary) {
        *owner_index = first_owner;
        return true;
    }
    if (address >= boundary) {
        *owner_index = second_owner;
        return true;
    }
    // The current expert path routes a complete bank to one owner; boundary
    // banks remain on the original application path until the ID planner is
    // integrated and qualified.
    return false;
}

template <typename Function>
Function runtime_symbol(const char *name, const char *fallback = nullptr) noexcept {
    void *address = v200_frontdoor_resolve_runtime_symbol(name);
    if (address == nullptr && fallback != nullptr)
        address = v200_frontdoor_resolve_runtime_symbol(fallback);
    return reinterpret_cast<Function>(address);
}

struct VerificationResult {
    bool performed = false;
    bool mismatch = false;
    bool compared = false;
    bool partition_launch_succeeded = false;
    cudaError_t status = cudaSuccess;
    float max_abs_error = -1.0f;
};

VerificationResult verify_owner_launch(
    const void *host_function, dim3 grid, dim3 block, void **original_arguments,
    size_t shared_bytes, cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch,
    const v200::operators::OwnerPartition *partitions, unsigned int partition_count,
    const void *destination, size_t output_bytes) noexcept {
    VerificationResult result{};
    if (output_bytes == 0 || output_bytes > max_verify_output_bytes ||
        output_bytes % sizeof(float) != 0) {
        return result;
    }

    using StreamSynchronize = cudaError_t (CUDARTAPI *)(cudaStream_t);
    using Memcpy = cudaError_t (CUDARTAPI *)(void *, const void *, size_t, cudaMemcpyKind);
    auto stream_synchronize = runtime_symbol<StreamSynchronize>("cudaStreamSynchronize");
    auto memcpy = runtime_symbol<Memcpy>("cudaMemcpy", "cudaMemcpy_v2");
    if (stream_synchronize == nullptr || memcpy == nullptr) return result;

    const size_t count = output_bytes / sizeof(float);
    std::unique_ptr<float[]> reference(new (std::nothrow) float[count]);
    std::unique_ptr<float[]> actual(new (std::nothrow) float[count]);
    if (!reference || !actual) return result;

    result.performed = true;
    result.status = native_launch(host_function, grid, block, original_arguments,
                                  shared_bytes, anchor_stream);
    if (result.status != cudaSuccess) return result;
    result.status = stream_synchronize(anchor_stream);
    if (result.status != cudaSuccess) return result;
    result.status = memcpy(reference.get(), destination, output_bytes,
                           cudaMemcpyDeviceToHost);
    if (result.status != cudaSuccess) return result;

    result.status = v200::operators::launch_owner_partitions(
        partitions, partition_count, host_function, block, shared_bytes,
        anchor_stream, native_launch);
    if (result.status != cudaSuccess) return result;
    result.partition_launch_succeeded = true;
    result.status = stream_synchronize(anchor_stream);
    if (result.status != cudaSuccess) return result;
    result.status = memcpy(actual.get(), destination, output_bytes,
                           cudaMemcpyDeviceToHost);
    if (result.status != cudaSuccess) return result;

    result.max_abs_error = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float difference = std::fabs(reference[i] - actual[i]);
        if (!std::isfinite(difference)) {
            if (std::memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
                result.mismatch = true;
                result.max_abs_error = INFINITY;
                break;
            }
        } else {
            result.max_abs_error = std::max(result.max_abs_error, difference);
            if (std::memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
                result.mismatch = true;
            }
        }
    }
    result.compared = true;
    if (result.mismatch) result.status = cudaErrorUnknown;
    return result;
}

void **argument_addresses(Q4KMmvqArgs &args, std::array<void *, 19> &addresses) noexcept {
    addresses = {{&args.vx, &args.vy, &args.ids, &args.fusion, &args.dst,
                  &args.ncols_x, &args.nchannels_y, &args.stride_row_x,
                  &args.stride_col_y, &args.stride_col_dst, &args.channel_ratio,
                  &args.stride_channel_x, &args.stride_channel_y,
                  &args.stride_channel_dst, &args.sample_ratio,
                  &args.stride_sample_x, &args.stride_sample_y,
                  &args.stride_sample_dst, &args.ids_stride}};
    return addresses.data();
}

enum class MmvqFamily : uint32_t {
    q4k_plain = V200_OPERATOR_MMVQ_Q4K_PLAIN,
    q4k_gate_fused = V200_OPERATOR_MMVQ_Q4K_GATE_FUSED,
};

struct ExpertBankCandidate {
    Q4KMmvqArgs args{};
    v200_memory_pair_profile profile{};
    uint32_t owner_index = 0;
    MmvqFamily family = MmvqFamily::q4k_plain;
    size_t bank_bytes = 0;
    size_t active_weight_bytes = 0;
    uint64_t rows = 0;
    size_t remote_transfer_bytes = 0;
    size_t input_bytes = 0;
    size_t ids_bytes = 0;
    size_t output_bytes = 0;
};

bool capture_q4k_arguments(void **arguments, Q4KMmvqArgs *captured) noexcept {
    if (arguments == nullptr || captured == nullptr) return false;
    const size_t sizes[19] = {
        sizeof(captured->vx), sizeof(captured->vy), sizeof(captured->ids),
        sizeof(captured->fusion), sizeof(captured->dst), sizeof(captured->ncols_x),
        sizeof(captured->nchannels_y), sizeof(captured->stride_row_x),
        sizeof(captured->stride_col_y), sizeof(captured->stride_col_dst),
        sizeof(captured->channel_ratio), sizeof(captured->stride_channel_x),
        sizeof(captured->stride_channel_y), sizeof(captured->stride_channel_dst),
        sizeof(captured->sample_ratio), sizeof(captured->stride_sample_x),
        sizeof(captured->stride_sample_y), sizeof(captured->stride_sample_dst),
        sizeof(captured->ids_stride)};
    const size_t offsets[19] = {
        offsetof(Q4KMmvqArgs, vx), offsetof(Q4KMmvqArgs, vy),
        offsetof(Q4KMmvqArgs, ids), offsetof(Q4KMmvqArgs, fusion),
        offsetof(Q4KMmvqArgs, dst), offsetof(Q4KMmvqArgs, ncols_x),
        offsetof(Q4KMmvqArgs, nchannels_y), offsetof(Q4KMmvqArgs, stride_row_x),
        offsetof(Q4KMmvqArgs, stride_col_y), offsetof(Q4KMmvqArgs, stride_col_dst),
        offsetof(Q4KMmvqArgs, channel_ratio), offsetof(Q4KMmvqArgs, stride_channel_x),
        offsetof(Q4KMmvqArgs, stride_channel_y), offsetof(Q4KMmvqArgs, stride_channel_dst),
        offsetof(Q4KMmvqArgs, sample_ratio), offsetof(Q4KMmvqArgs, stride_sample_x),
        offsetof(Q4KMmvqArgs, stride_sample_y), offsetof(Q4KMmvqArgs, stride_sample_dst),
        offsetof(Q4KMmvqArgs, ids_stride)};
    auto *destination = reinterpret_cast<unsigned char *>(captured);
    for (size_t i = 0; i < 19; ++i) {
        if (arguments[i] == nullptr) return false;
        std::memcpy(destination + offsets[i], arguments[i], sizes[i]);
    }
    return true;
}

bool prepare_expert_bank_candidate(const void *host_function, dim3 grid,
                                   dim3 block, void **arguments,
                                   ExpertBankCandidate *out) noexcept {
    if (out == nullptr || host_function == nullptr || grid.x == 0 || grid.y <= 1 ||
        grid.z != 1 || block.x != 32 || block.y != 4 || block.z != 1) {
        return false;
    }

    const std::string name = registered_kernel_name(host_function);
    MmvqFamily family{};
    if (name == q4k_mmvq_name) {
        family = MmvqFamily::q4k_plain;
    } else if (name == q4k_mmvq_gate_name) {
        family = MmvqFamily::q4k_gate_fused;
    } else {
        return false;
    }

    ExpertBankCandidate candidate{};
    candidate.family = family;
    if (!capture_q4k_arguments(arguments, &candidate.args)) return false;
    const Q4KMmvqArgs &args = candidate.args;
    if (args.ids == nullptr || args.vx == nullptr || args.vy == nullptr ||
        args.dst == nullptr || args.ncols_x < 2048 ||
        args.ncols_x % q4k_block_elements != 0 ||
        args.ncols_x % q8_1_block_elements != 0 || args.stride_row_x == 0 ||
        args.stride_row_x > SIZE_MAX / q4k_block_bytes ||
        static_cast<size_t>(args.stride_row_x) * q4k_block_elements != args.ncols_x ||
        args.stride_sample_x == 0 || args.stride_channel_y == 0 ||
        args.nchannels_y.z == 0 || args.stride_channel_dst < grid.x) {
        return false;
    }
    if (family == MmvqFamily::q4k_plain) {
        if (!all_null(args.fusion)) return false;
    } else if (args.fusion.gate == nullptr || args.fusion.x_bias != nullptr ||
               args.fusion.gate_bias != nullptr || args.fusion.x_scale != nullptr ||
               args.fusion.gate_scale != nullptr) {
        return false;
    }

    size_t row_bytes = 0;
    size_t active_rows = 0;
    size_t id_bytes = 0;
    size_t blocks_per_input = 0;
    size_t last_input_channel = 0;
    size_t last_input_offset_bytes = 0;
    size_t one_expert_bytes = 0;
    size_t output_elements = 0;
    size_t output_offset = 0;
    if (!checked_multiply(args.stride_row_x, q4k_block_bytes, &row_bytes) ||
        !checked_multiply(grid.x, grid.y, &active_rows) ||
        !checked_multiply(active_rows, row_bytes, &candidate.active_weight_bytes) ||
        !checked_multiply(grid.x, row_bytes, &one_expert_bytes) ||
        !checked_multiply(grid.y, sizeof(int32_t), &id_bytes) ||
        !checked_multiply(args.ncols_x / q8_1_block_elements,
                          q8_1_block_bytes, &blocks_per_input) ||
        !checked_multiply(std::min(grid.y, args.nchannels_y.z) - 1,
                          args.stride_channel_y, &last_input_channel) ||
        !checked_multiply(last_input_channel, q8_1_block_bytes,
                          &last_input_offset_bytes) ||
        !checked_add(last_input_offset_bytes, blocks_per_input,
                     &candidate.input_bytes) ||
        !checked_multiply(grid.y - 1, args.stride_channel_dst, &output_offset) ||
        !checked_add(output_offset, grid.x, &output_elements) ||
        !checked_multiply(output_elements, sizeof(float), &candidate.output_bytes) ||
        !checked_add(candidate.input_bytes, candidate.output_bytes,
                     &candidate.remote_transfer_bytes) ||
        !checked_multiply(args.stride_sample_x, q4k_block_bytes,
                          &candidate.bank_bytes)) {
        return false;
    }
    candidate.ids_bytes = id_bytes;
    candidate.rows = static_cast<uint64_t>(active_rows);
    if (one_expert_bytes > candidate.bank_bytes ||
        args.stride_channel_x == 0 || args.stride_channel_x > args.stride_sample_x ||
        !query_pair_managed_range(args.vx, candidate.bank_bytes, nullptr) ||
        !query_pair_managed_range(args.ids, candidate.ids_bytes, nullptr) ||
        !query_pair_managed_range(args.vy, candidate.input_bytes, nullptr) ||
        !query_pair_managed_range(args.dst, candidate.output_bytes, nullptr)) {
        return false;
    }
    if (ranges_overlap(args.vx, candidate.bank_bytes, args.ids, candidate.ids_bytes) ||
        ranges_overlap(args.vx, candidate.bank_bytes, args.vy, candidate.input_bytes) ||
        ranges_overlap(args.vx, candidate.bank_bytes, args.dst, candidate.output_bytes) ||
        ranges_overlap(args.ids, candidate.ids_bytes, args.vy, candidate.input_bytes) ||
        ranges_overlap(args.ids, candidate.ids_bytes, args.dst, candidate.output_bytes) ||
        ranges_overlap(args.vy, candidate.input_bytes, args.dst, candidate.output_bytes)) {
        return false;
    }

    v200_allocation_info weight_allocation{};
    if (!query_pair_managed_range(args.vx, candidate.bank_bytes, &weight_allocation))
        return false;
    candidate.profile.abi_version = V200_MEMORY_ABI_VERSION;
    candidate.profile.struct_size = sizeof(candidate.profile);
    if (v200_memory_get_profile(&candidate.profile) != CUDA_SUCCESS ||
        candidate.profile.owner_device[0] == candidate.profile.owner_device[1] ||
        candidate.profile.abi_version != V200_MEMORY_ABI_VERSION ||
        candidate.profile.struct_size < sizeof(v200_memory_pair_profile) ||
        candidate.profile.default_owner != candidate.profile.owner_device[0] ||
        candidate.profile.pair_access_qualified == 0 ||
        candidate.profile.qualification_id[0] == '\0' ||
        !owner_for_contiguous_range(reinterpret_cast<uintptr_t>(args.vx),
                                    candidate.bank_bytes, weight_allocation,
                                    candidate.profile, &candidate.owner_index)) {
        return false;
    }
    if (family == MmvqFamily::q4k_gate_fused) {
        v200_allocation_info gate_allocation{};
        uint32_t gate_owner = 0;
        if (!query_pair_managed_range(args.fusion.gate, candidate.bank_bytes,
                                      &gate_allocation) ||
            !owner_for_contiguous_range(
                reinterpret_cast<uintptr_t>(args.fusion.gate), candidate.bank_bytes,
                gate_allocation, candidate.profile, &gate_owner) ||
            gate_owner != candidate.owner_index ||
            ranges_overlap(args.fusion.gate, candidate.bank_bytes, args.ids,
                           candidate.ids_bytes) ||
            ranges_overlap(args.fusion.gate, candidate.bank_bytes, args.vy,
                           candidate.input_bytes) ||
            ranges_overlap(args.fusion.gate, candidate.bank_bytes, args.dst,
                           candidate.output_bytes)) {
            return false;
        }
    }

    *out = candidate;
    return true;
}

bool above_peer_work_threshold(const ExpertBankCandidate &candidate) noexcept {
    size_t threshold = 0;
    return checked_add(minimum_peer_active_weight_bytes,
                       candidate.remote_transfer_bytes, &threshold) &&
           candidate.active_weight_bytes >= threshold;
}

int try_launch_expert_bank(const void *host_function, dim3 grid, dim3 block,
                           void **arguments, size_t shared_bytes,
                           cudaStream_t anchor_stream,
                           v200_operator_native_launch_fn native_launch,
                           cudaError_t *status) noexcept {
    if (status == nullptr || v200_frontdoor_mode() != 2) return 0;
    ExpertBankCandidate candidate{};
    if (!prepare_expert_bank_candidate(host_function, grid, block, arguments,
                                       &candidate)) {
        return 0;
    }

    OperatorCounters &counters = operator_counters();
    counters.expert_matched_calls.fetch_add(1);
    if (candidate.owner_index == 0 || !above_peer_work_threshold(candidate)) {
        const uint64_t local = counters.expert_anchor_local_calls.fetch_add(1) + 1;
        counters.expert_anchor_rows.fetch_add(candidate.rows);
        if (candidate.owner_index == 1)
            counters.expert_small_peer_fallback_calls.fetch_add(1);
        if (local == 1 || (local % 64) == 0) {
            write_operator_event(
                candidate.owner_index == 0 ? "q4k_mmvq_expert_anchor_bank"
                                           : "q4k_mmvq_expert_small_peer_fallback",
                candidate.owner_index == 0 ? "anchor-owner" : "small-peer-fallback",
                counters.matched_calls.load(), counters.anchor_local_calls.load(),
                counters.peer_owner_calls.load(), counters.split_calls.load(),
                counters.anchor_rows.load(), counters.peer_rows.load(),
                counters.verified_rows.load(), -1.0f, 0,
                counters.expert_matched_calls.load(),
                counters.expert_anchor_local_calls.load(),
                counters.expert_peer_owner_calls.load(),
                counters.expert_small_peer_fallback_calls.load(),
                counters.expert_rows.load(), counters.expert_anchor_rows.load(),
                counters.expert_peer_rows.load(), counters.expert_verified_calls.load(),
                counters.expert_verified_outputs.load(),
                counters.expert_plain_verified_calls.load(),
                counters.expert_plain_verified_outputs.load(),
                counters.expert_fused_verified_calls.load(),
                counters.expert_fused_verified_outputs.load(),
                counters.expert_verification_failures.load());
            if (local == 1)
                v200_frontdoor_snapshot("operator_q4k_mmvq_expert_anchor");
        }
        return 0;
    }

    v200::operators::OwnerPartition partition{};
    partition.device = candidate.profile.owner_device[candidate.owner_index];
    partition.grid = grid;
    partition.arguments = arguments;
    static std::atomic<bool> plain_verification_attempted{false};
    static std::atomic<bool> fused_verification_attempted{false};
    const char *verify_enabled = std::getenv("V200_OPERATOR_VERIFY");
    const bool verify_requested = verify_enabled != nullptr &&
        std::strcmp(verify_enabled, "1") == 0 &&
        candidate.output_bytes <= max_verify_output_bytes;
    std::atomic<bool> &verify_attempted =
        candidate.family == MmvqFamily::q4k_plain
            ? plain_verification_attempted : fused_verification_attempted;
    const bool verify_this_route = verify_requested &&
                                   !verify_attempted.exchange(true);
    VerificationResult verification{};
    if (verify_this_route) {
        verification = verify_owner_launch(
            host_function, grid, block, arguments, shared_bytes, anchor_stream,
            native_launch, &partition, 1, candidate.args.dst,
            candidate.output_bytes);
    }
    if (!verification.performed) {
        *status = v200::operators::launch_owner_partitions(
            &partition, 1, host_function, block, shared_bytes, anchor_stream,
            native_launch);
    } else {
        *status = verification.status;
    }

    const bool launch_succeeded = verification.performed
        ? verification.partition_launch_succeeded : *status == cudaSuccess;
    const bool verification_failed = verification.mismatch ||
        (verification.performed && verification.status != cudaSuccess);
    if (verification_failed)
        counters.expert_verification_failures.fetch_add(1);
    if (verification.compared) {
        const uint64_t output_count = candidate.output_bytes / sizeof(float);
        counters.expert_verified_calls.fetch_add(1);
        counters.expert_verified_outputs.fetch_add(output_count);
        if (candidate.family == MmvqFamily::q4k_plain) {
            counters.expert_plain_verified_calls.fetch_add(1);
            counters.expert_plain_verified_outputs.fetch_add(output_count);
        } else {
            counters.expert_fused_verified_calls.fetch_add(1);
            counters.expert_fused_verified_outputs.fetch_add(output_count);
        }
    }
    uint64_t peer_calls = counters.expert_peer_owner_calls.load();
    if (launch_succeeded) {
        peer_calls = counters.expert_peer_owner_calls.fetch_add(1) + 1;
        counters.expert_rows.fetch_add(candidate.rows);
        counters.expert_peer_rows.fetch_add(candidate.rows);
    }
    if (launch_succeeded || verification.performed) {
        if (peer_calls == 1 || (peer_calls % 64) == 0 || verification.performed ||
            verification_failed) {
            write_operator_event(
                verification_failed ? "q4k_mmvq_expert_verification_failed"
                    : verification.performed ? "q4k_mmvq_expert_verification"
                                             : "q4k_mmvq_expert_peer_bank",
                candidate.family == MmvqFamily::q4k_plain
                    ? "peer-bank-plain" : "peer-bank-gate-fused",
                counters.matched_calls.load(), counters.anchor_local_calls.load(),
                counters.peer_owner_calls.load(), counters.split_calls.load(),
                counters.anchor_rows.load(), counters.peer_rows.load(),
                counters.verified_rows.load(), verification.max_abs_error, *status,
                counters.expert_matched_calls.load(),
                counters.expert_anchor_local_calls.load(),
                counters.expert_peer_owner_calls.load(),
                counters.expert_small_peer_fallback_calls.load(),
                counters.expert_rows.load(), counters.expert_anchor_rows.load(),
                counters.expert_peer_rows.load(), counters.expert_verified_calls.load(),
                counters.expert_verified_outputs.load(),
                counters.expert_plain_verified_calls.load(),
                counters.expert_plain_verified_outputs.load(),
                counters.expert_fused_verified_calls.load(),
                counters.expert_fused_verified_outputs.load(),
                counters.expert_verification_failures.load());
            if (peer_calls == 1 || verification.performed || verification_failed)
                v200_frontdoor_snapshot("operator_q4k_mmvq_expert_peer_bank");
        }
    }
    return 1;
}

} // namespace

extern "C" std::size_t v200_operator_list_mmvq_host_functions(
    const void **out, std::size_t capacity) {
    Registry &registered = registry();
    std::lock_guard<std::mutex> lock(registered.mutex);
    std::size_t count = 0;
    for (const auto &entry : registered.names) {
        if (entry.second != q4k_mmvq_name && entry.second != q4k_mmvq_gate_name)
            continue;
        if (out != nullptr && count < capacity) out[count] = entry.first;
        ++count;
    }
    return count;
}

extern "C" int v200_operator_inspect_mmvq_bank(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    std::uint32_t *owner_index, std::uint32_t *family,
    std::size_t *bank_bytes, std::size_t *active_weight_bytes) {
    if (owner_index == nullptr || family == nullptr || bank_bytes == nullptr ||
        active_weight_bytes == nullptr) {
        return 0;
    }
    ExpertBankCandidate candidate{};
    if (!prepare_expert_bank_candidate(host_function, grid, block, arguments,
                                       &candidate) ||
        !above_peer_work_threshold(candidate)) {
        return 0;
    }
    *owner_index = candidate.owner_index;
    *family = static_cast<std::uint32_t>(candidate.family);
    *bank_bytes = candidate.bank_bytes;
    *active_weight_bytes = candidate.active_weight_bytes;
    return 1;
}

/* Registration is metadata-only. CUDA module registration remains owned by
 * the original runtime DSO, resolved through the frontdoor's loaded-library
 * resolver so RTLD_LOCAL backend plugins still register normally. */
extern "C" void CUDARTAPI __cudaRegisterFunction(
    void **fat_cubin_handle, const char *host_function, char *device_function,
    const char *device_name, int thread_limit, uint3 *thread_id, uint3 *block_id,
    dim3 *block_dim, dim3 *grid_dim, int *warp_size) {
    using Register = void (CUDARTAPI *)(void **, const char *, char *, const char *,
                                       int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
    auto real = reinterpret_cast<Register>(
        v200_frontdoor_resolve_runtime_symbol("__cudaRegisterFunction"));
    if (real == nullptr) {
        real = reinterpret_cast<Register>(dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
    }

    if (host_function != nullptr && device_name != nullptr) {
        Registry &registered = registry();
        std::lock_guard<std::mutex> lock(registered.mutex);
        registered.names[reinterpret_cast<const void *>(host_function)] = device_name;
    }
    // Silently dropping CUDA module registration makes every downstream kernel
    // launch fail much later with an unrelated invalid-function error.
    if (real == nullptr) std::abort();
    real(fat_cubin_handle, host_function, device_function, device_name, thread_limit,
         thread_id, block_id, block_dim, grid_dim, warp_size);
}

extern "C" int v200_operator_try_launch(
    const void *host_function, dim3 grid, dim3 block, void **arguments,
    size_t shared_bytes, cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch, cudaError_t *status) {
    if (status == nullptr || host_function == nullptr || arguments == nullptr ||
        native_launch == nullptr || v200_frontdoor_mode() != 2) {
        return 0;
    }

    const std::string name = registered_kernel_name(host_function);
    if (name == q4k_mmvq_gate_name) {
        return try_launch_expert_bank(host_function, grid, block, arguments,
                                     shared_bytes, anchor_stream, native_launch,
                                     status);
    }
    if (name != q4k_mmvq_name) return 0;
    if (grid.y > 1) {
        return try_launch_expert_bank(host_function, grid, block, arguments,
                                     shared_bytes, anchor_stream, native_launch,
                                     status);
    }

    // On Volta this exact Q4_K, single-column specialization uses one output
    // row per block. Small grids stay on the original launch to avoid paying
    // event and peer-stream startup for negligible work.
    if (grid.x < 64 || grid.y != 1 || grid.z != 1 || block.x != 32 ||
        block.y != 4 || block.z != 1) return 0;

    Q4KMmvqArgs captured{};
    for (size_t i = 0; i < 19; ++i) {
        if (arguments[i] == nullptr) return 0;
    }
    const size_t argument_sizes[19] = {
        sizeof(captured.vx), sizeof(captured.vy), sizeof(captured.ids), sizeof(captured.fusion),
        sizeof(captured.dst), sizeof(captured.ncols_x), sizeof(captured.nchannels_y),
        sizeof(captured.stride_row_x), sizeof(captured.stride_col_y), sizeof(captured.stride_col_dst),
        sizeof(captured.channel_ratio), sizeof(captured.stride_channel_x),
        sizeof(captured.stride_channel_y), sizeof(captured.stride_channel_dst),
        sizeof(captured.sample_ratio), sizeof(captured.stride_sample_x),
        sizeof(captured.stride_sample_y), sizeof(captured.stride_sample_dst), sizeof(captured.ids_stride)};
    const size_t argument_offsets[19] = {
        offsetof(Q4KMmvqArgs, vx), offsetof(Q4KMmvqArgs, vy), offsetof(Q4KMmvqArgs, ids),
        offsetof(Q4KMmvqArgs, fusion), offsetof(Q4KMmvqArgs, dst), offsetof(Q4KMmvqArgs, ncols_x),
        offsetof(Q4KMmvqArgs, nchannels_y), offsetof(Q4KMmvqArgs, stride_row_x),
        offsetof(Q4KMmvqArgs, stride_col_y), offsetof(Q4KMmvqArgs, stride_col_dst),
        offsetof(Q4KMmvqArgs, channel_ratio), offsetof(Q4KMmvqArgs, stride_channel_x),
        offsetof(Q4KMmvqArgs, stride_channel_y), offsetof(Q4KMmvqArgs, stride_channel_dst),
        offsetof(Q4KMmvqArgs, sample_ratio), offsetof(Q4KMmvqArgs, stride_sample_x),
        offsetof(Q4KMmvqArgs, stride_sample_y), offsetof(Q4KMmvqArgs, stride_sample_dst),
        offsetof(Q4KMmvqArgs, ids_stride)};
    auto *captured_bytes = reinterpret_cast<unsigned char *>(&captured);
    for (size_t i = 0; i < 19; ++i) {
        std::memcpy(captured_bytes + argument_offsets[i], arguments[i], argument_sizes[i]);
    }
    if (captured.ids != nullptr || !all_null(captured.fusion) ||
        captured.vx == nullptr || captured.vy == nullptr || captured.dst == nullptr ||
        captured.ncols_x == 0 || captured.ncols_x % q4k_block_elements != 0 ||
        captured.stride_row_x == 0 ||
        captured.stride_row_x > SIZE_MAX / q4k_block_bytes ||
        static_cast<size_t>(captured.stride_row_x) * q4k_block_elements != captured.ncols_x) {
        return 0;
    }

    const size_t bytes_per_row = static_cast<size_t>(captured.stride_row_x) * q4k_block_bytes;
    if (grid.x > SIZE_MAX / bytes_per_row ||
        captured.ncols_x / 32u > SIZE_MAX / q8_1_block_bytes ||
        grid.x > SIZE_MAX / sizeof(float) ||
        !is_pair_managed_range(captured.vx, static_cast<size_t>(grid.x) * bytes_per_row) ||
        !is_pair_managed_range(captured.vy, static_cast<size_t>(captured.ncols_x / 32u) * q8_1_block_bytes) ||
        !is_pair_managed_range(captured.dst, static_cast<size_t>(grid.x) * sizeof(float))) {
        return 0;
    }
    const size_t weight_bytes = static_cast<size_t>(grid.x) * bytes_per_row;
    const size_t input_bytes = static_cast<size_t>(captured.ncols_x / 32u) * q8_1_block_bytes;
    const size_t output_bytes = static_cast<size_t>(grid.x) * sizeof(float);
    if (ranges_overlap(captured.vx, weight_bytes, captured.vy, input_bytes) ||
        ranges_overlap(captured.vx, weight_bytes, captured.dst, output_bytes) ||
        ranges_overlap(captured.vy, input_bytes, captured.dst, output_bytes)) {
        return 0;
    }

    v200_allocation_info allocation{};
    allocation.abi_version = V200_MEMORY_ABI_VERSION;
    allocation.struct_size = sizeof(allocation);
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    if (v200_memory_query(captured.vx, &allocation) != CUDA_SUCCESS ||
        v200_memory_get_profile(&profile) != CUDA_SUCCESS) {
        return 0;
    }

    v200::operators::MmvqRows rows{};
    if (!v200::operators::plan_q4k_mmvq_rows(
            reinterpret_cast<uintptr_t>(captured.vx), bytes_per_row, grid.x,
            allocation, profile, &rows)) {
        return 0;
    }
    OperatorCounters &counters = operator_counters();
    const uint64_t matched = counters.matched_calls.fetch_add(1) + 1;
    if (!rows.split && rows.first_owner == 0) {
        const uint64_t local = counters.anchor_local_calls.fetch_add(1) + 1;
        counters.anchor_rows.fetch_add(grid.x);
        write_operator_event("q4k_mmvq_anchor_local", "anchor-local", matched,
                             local, counters.peer_owner_calls.load(),
                             counters.split_calls.load(),
                             counters.anchor_rows.load(), counters.peer_rows.load(),
                             counters.verified_rows.load(), -1.0f, 0);
        if (local == 1 || (local % 64) == 0) {
            v200_frontdoor_snapshot("operator_q4k_mmvq_anchor_local");
        }
        return 0;
    }

    // Build per-owner argument packs. Each row is computed wholly by one
    // unchanged llama Q4_K kernel; no floating-point reduction is repartitioned.
    std::array<Q4KMmvqArgs, 2> packs{{captured, captured}};
    std::array<std::array<void *, 19>, 2> argument_addresses_by_owner{};
    std::array<uint32_t, 2> launch_rows{{rows.first_rows, rows.second_rows}};
    const unsigned int partition_count = rows.split ? 2u : 1u;
    if (rows.split) {
        const uintptr_t offset = static_cast<uintptr_t>(rows.first_rows) * bytes_per_row;
        packs[1].vx = reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(captured.vx) + offset);
        packs[1].dst = reinterpret_cast<float *>(reinterpret_cast<uintptr_t>(captured.dst) +
                                                 static_cast<uintptr_t>(rows.first_rows) * sizeof(float));
    }

    for (unsigned int i = 0; i < partition_count; ++i) {
        argument_addresses(packs[i], argument_addresses_by_owner[i]);
    }

    // A managed tensor can be physically placed on the peer while every other
    // pointer remains addressable there. The bridge event-joins these launches
    // to anchor_stream before this callback returns.
    std::array<v200::operators::OwnerPartition, 2> partitions{};
    for (unsigned int i = 0; i < partition_count; ++i) {
        const uint32_t owner = i == 0 ? rows.first_owner : rows.second_owner;
        partitions[i].device = profile.owner_device[owner];
        partitions[i].grid = dim3(launch_rows[i], grid.y, grid.z);
        partitions[i].arguments = argument_addresses_by_owner[i].data();
    }
    static std::atomic<bool> peer_verification_attempted{false};
    static std::atomic<bool> split_verification_attempted{false};
    const char *verify_enabled = std::getenv("V200_OPERATOR_VERIFY");
    const bool verify_route = verify_enabled != nullptr &&
                              std::strcmp(verify_enabled, "1") == 0 &&
                              output_bytes <= max_verify_output_bytes;
    const bool verify_this_route = verify_route &&
        (rows.split ? !split_verification_attempted.exchange(true)
                    : !peer_verification_attempted.exchange(true));
    VerificationResult verification{};
    if (verify_this_route) {
        verification = verify_owner_launch(
            host_function, grid, block, arguments, shared_bytes, anchor_stream,
            native_launch, partitions.data(), partition_count, captured.dst,
            output_bytes);
    }
    if (!verification.performed) {
        *status = v200::operators::launch_owner_partitions(
            partitions.data(), partition_count, host_function, block, shared_bytes,
            anchor_stream, native_launch);
    } else {
        *status = verification.status;
    }

    const bool partition_launch_succeeded = verification.performed
        ? verification.partition_launch_succeeded
        : *status == cudaSuccess;
    if (partition_launch_succeeded) {
        const uint64_t anchor_rows = rows.first_owner == 0
            ? rows.first_rows : (rows.split ? rows.second_rows : 0u);
        const uint64_t peer_rows = rows.first_owner == 1
            ? rows.first_rows : (rows.split ? rows.second_rows : 0u);
        if (rows.split) {
            counters.split_calls.fetch_add(1);
        } else {
            counters.peer_owner_calls.fetch_add(1);
        }
        counters.anchor_rows.fetch_add(anchor_rows);
        counters.peer_rows.fetch_add(peer_rows);
        if (verification.performed) {
            if (verification.compared) counters.verified_rows.fetch_add(grid.x);
            if (verification.mismatch || verification.status != cudaSuccess)
                counters.verification_failures.fetch_add(1);
        }

        static std::atomic<bool> peer_event_emitted{false};
        static std::atomic<bool> split_event_emitted{false};
        const bool first_route_event = rows.split
            ? !split_event_emitted.exchange(true)
            : !peer_event_emitted.exchange(true);
        write_operator_event(
            (verification.mismatch || (verification.performed && verification.status != cudaSuccess))
                ? "q4k_mmvq_verification_failed" :
                (verification.performed ? "q4k_mmvq_verification" : "q4k_mmvq_route"),
            rows.split ? "split" : "peer-owner", matched,
            counters.anchor_local_calls.load(), counters.peer_owner_calls.load(),
            counters.split_calls.load(), counters.anchor_rows.load(),
            counters.peer_rows.load(),
            verification.compared ? grid.x : 0,
            verification.max_abs_error, *status);
        if (first_route_event || verification.performed || verification.mismatch) {
            v200_frontdoor_snapshot(rows.split ? "operator_q4k_mmvq_split_owners"
                                              : "operator_q4k_mmvq_peer_owner");
        }
    }
    return 1;
}
