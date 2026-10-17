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
constexpr size_t q4k_block_bytes = 144;
constexpr uint32_t q4k_block_elements = 256;
constexpr size_t q8_1_block_bytes = 36;
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
};

void write_operator_event(const char *event, const char *route,
                          uint64_t matched_calls, uint64_t anchor_local_calls,
                          uint64_t peer_owner_calls, uint64_t split_calls,
                          uint64_t anchor_rows, uint64_t peer_rows,
                          uint64_t verified_rows, float max_abs_error,
                          int status) noexcept {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    char line[1024];
    const int count = std::snprintf(
        line, sizeof(line),
        "{\"schema\":\"v200-operator-ledger/v1\",\"event\":\"%s\","
        "\"operation\":\"q4k_mmvq\",\"route\":\"%s\","
        "\"matched_calls\":%llu,\"anchor_local_calls\":%llu,"
        "\"peer_owner_calls\":%llu,\"split_calls\":%llu,"
        "\"anchor_rows\":%llu,\"peer_rows\":%llu,"
        "\"verified_rows\":%llu,\"max_abs_error\":%.9g,\"status\":%d}\n",
        event, route,
        static_cast<unsigned long long>(matched_calls),
        static_cast<unsigned long long>(anchor_local_calls),
        static_cast<unsigned long long>(peer_owner_calls),
        static_cast<unsigned long long>(split_calls),
        static_cast<unsigned long long>(anchor_rows),
        static_cast<unsigned long long>(peer_rows),
        static_cast<unsigned long long>(verified_rows), max_abs_error, status);
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
                final.verification_failures.load() == 0 ? 0 : 1);
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

bool is_pair_managed_range(const void *pointer, size_t bytes) noexcept {
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
    return offset <= allocation.requested_bytes &&
           bytes <= allocation.requested_bytes - offset;
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

} // namespace

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
    if (name != q4k_mmvq_name) return 0;

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
        const uint64_t anchor_rows = rows.split ? rows.first_rows : 0;
        const uint64_t peer_rows = rows.split ? rows.second_rows : grid.x;
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
