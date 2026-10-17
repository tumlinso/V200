#include "runtime_state.hpp"

#include "v200/frontdoor/abi_profile.hpp"
#include "v200/execution/owner_execution_c.h"
#include "v200/memory/api.h"
#include "v200/operators/graph_rewrite.hpp"
#include "v200/operators/launch.hpp"
#include "vmm_location.hpp"

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <mutex>
#include <set>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <vector>

using namespace v200::frontdoor::detail;

// The CUDA header optionally maps these spellings to the per-thread default
// stream ABI. The frontdoor exports both ABI forms explicitly instead.
#undef cuMemsetD8Async
#undef cuMemsetD16Async
#undef cuMemsetD32Async
#undef cuMemsetD2D8Async
#undef cuMemsetD2D16Async
#undef cuMemsetD2D32Async

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(
    CUdeviceptr, const void *, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(
    void *, CUdeviceptr, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(
    CUdeviceptr, CUdeviceptr, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr, unsigned char, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD16Async_ptsz(CUdeviceptr, unsigned short, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr, unsigned int, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD2D8Async_ptsz(
    CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD2D16Async_ptsz(
    CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemsetD2D32Async_ptsz(
    CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream);
extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream, CUevent, unsigned int);

namespace {

void write_multi_pair_copy_event(uint32_t dst_pair, uint32_t src_pair,
                                 size_t bytes, cudaError_t status,
                                 double elapsed_ms, const char *route);
void register_process_exit_snapshot(uint32_t pair);

struct DeviceSelection {
    bool valid = false;
    bool attempted = false;
    int count = 0;
    int ordinal[4] = {-1, -1, -1, -1};
    CUdevice device[4] = {-1, -1, -1, -1};
    CUuuid uuid[4]{};
    CUresult driver_error = CUDA_ERROR_NOT_INITIALIZED;
};

std::mutex selection_mutex;
DeviceSelection selection;
std::mutex profile_mutex;
int profile_state[2] = {0, 0}; // 0 unseen, 1 initializing, 2 ready, 3 failed
CUresult profile_error[2] = {CUDA_ERROR_NOT_INITIALIZED,
                              CUDA_ERROR_NOT_INITIALIZED};
std::shared_mutex tracked_use_gate;
std::mutex stream_gate_mutex;
std::map<uint64_t, std::shared_ptr<std::mutex>> stream_gates;
std::mutex event_owner_mutex;
std::map<uintptr_t, uint32_t> event_owner_pairs;
thread_local bool profile_initializing[2] = {false, false};
thread_local bool capture_native_forward = false;
struct ContextPairRestoreEntry {
    CUcontext pushed_context = nullptr;
    uint32_t previous_pair = 0;
};
thread_local std::vector<ContextPairRestoreEntry> context_pair_restore_stack;

bool save_context_pair_for_restore(uint32_t previous_pair) noexcept {
    try {
        context_pair_restore_stack.push_back({nullptr, previous_pair});
        return true;
    } catch (...) {
        return false;
    }
}

void bind_context_pair_restore(CUcontext context) noexcept {
    if (!context_pair_restore_stack.empty())
        context_pair_restore_stack.back().pushed_context = context;
}

void discard_context_pair_restore() noexcept {
    if (!context_pair_restore_stack.empty()) context_pair_restore_stack.pop_back();
}

bool take_context_pair_restore(CUcontext context, uint32_t *pair) noexcept {
    if (pair == nullptr || context_pair_restore_stack.empty() || context == nullptr ||
        context_pair_restore_stack.back().pushed_context != context) return false;
    *pair = context_pair_restore_stack.back().previous_pair;
    context_pair_restore_stack.pop_back();
    return true;
}

struct GraphRewriteInfo {
    size_t matched_nodes = 0;
    size_t rewritten_nodes = 0;
    cudaError_t status = cudaSuccess;
};

std::mutex graph_info_mutex;
std::map<cudaGraph_t, GraphRewriteInfo> captured_graph_info;
std::map<cudaGraphExec_t, GraphRewriteInfo> graph_exec_info;
std::atomic<uint64_t> graph_replay_log_count{0};

bool graph_capture_enabled() noexcept {
    const char *enabled = std::getenv("V200_GRAPH_CAPTURE");
    return pair_mode() && enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

class CaptureNativeForwardScope {
public:
    CaptureNativeForwardScope() noexcept : previous_(capture_native_forward) {
        capture_native_forward = true;
    }
    ~CaptureNativeForwardScope() { capture_native_forward = previous_; }
    CaptureNativeForwardScope(const CaptureNativeForwardScope &) = delete;
    CaptureNativeForwardScope &operator=(const CaptureNativeForwardScope &) = delete;
private:
    bool previous_;
};

void write_graph_profile_snapshot(const char *event_name, uintptr_t handle,
                                  const GraphRewriteInfo &info);

std::shared_ptr<std::mutex> gate_for_stream(void *stream) {
    const uint64_t id = stream_id(stream);
    std::lock_guard<std::mutex> lock(stream_gate_mutex);
    auto &gate = stream_gates[id];
    if (!gate) gate = std::make_shared<std::mutex>();
    return gate;
}

void forget_stream_gate(void *stream) {
    const uint64_t id = stream_id(stream);
    std::lock_guard<std::mutex> lock(stream_gate_mutex);
    stream_gates.erase(id);
}

bool parse_uuid_list(const char *text, std::array<std::array<unsigned char, 16>, 4> &out,
                     int &count) {
    count = 0;
    if (text == nullptr || *text == '\0') return false;
    const char *cursor = text;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        if (count == 4 || *cursor == '\0') return false;
        if (std::strncmp(cursor, "GPU-", 4) == 0 || std::strncmp(cursor, "gpu-", 4) == 0) cursor += 4;
        std::array<unsigned char, 16> bytes{};
        int digits = 0;
        int high = -1;
        while (*cursor != '\0' && *cursor != ',') {
            const unsigned char c = static_cast<unsigned char>(*cursor++);
            if (c == '-' || c == ':' || c == ' ') continue;
            int value = -1;
            if (c >= '0' && c <= '9') value = c - '0';
            else if (c >= 'a' && c <= 'f') value = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') value = c - 'A' + 10;
            else return false;
            if (high < 0) high = value;
            else {
                if (digits >= 16) return false;
                bytes[static_cast<std::size_t>(digits++)] = static_cast<unsigned char>((high << 4) | value);
                high = -1;
            }
        }
        if (digits != 16 || high >= 0) return false;
        out[static_cast<std::size_t>(count++)] = bytes;
        if (*cursor == ',') {
            ++cursor;
            if (*cursor == '\0') return false;
        } else if (*cursor != '\0') {
            return false;
        }
    }
    return count == 1 || count == 2 || count == 4;
}

template <class Function>
Function next_fn(const char *symbol) {
    return resolve_next<Function>(symbol);
}

bool ensure_device_selection() {
    std::lock_guard<std::mutex> lock(selection_mutex);
    if (selection.attempted) return selection.valid;
    selection.attempted = true;
    const uint32_t mode = current_mode();
    if (mode != 1 && mode != 2) {
        selection.driver_error = CUDA_ERROR_INVALID_VALUE;
        return false;
    }

    std::array<std::array<unsigned char, 16>, 4> wanted{};
    int wanted_count = 0;
    if (!parse_uuid_list(std::getenv("V200_DEVICE_UUIDS"), wanted, wanted_count) ||
        (mode == 1 && wanted_count != 1) ||
        (mode == 2 && wanted_count != 2 && wanted_count != 4)) {
        selection.driver_error = CUDA_ERROR_INVALID_VALUE;
        return false;
    }

    using Init = CUresult (CUDAAPI *)(unsigned int);
    using GetCount = CUresult (CUDAAPI *)(int *);
    using GetDevice = CUresult (CUDAAPI *)(CUdevice *, int);
    using GetUuid = CUresult (CUDAAPI *)(CUuuid *, CUdevice);
    using GetAttribute = CUresult (CUDAAPI *)(int *, CUdevice_attribute, CUdevice);
    auto init = next_fn<Init>("cuInit");
    auto get_count = next_fn<GetCount>("cuDeviceGetCount");
    auto get_device = next_fn<GetDevice>("cuDeviceGet");
    auto get_uuid = next_fn<GetUuid>("cuDeviceGetUuid_v2");
    if (get_uuid == nullptr) get_uuid = next_fn<GetUuid>("cuDeviceGetUuid");
    auto get_attribute = next_fn<GetAttribute>("cuDeviceGetAttribute");
    if (!init || !get_count || !get_device || !get_uuid || !get_attribute) {
        selection.driver_error = CUDA_ERROR_NOT_SUPPORTED;
        return false;
    }

    CUresult result;
    {
        InternalCallScope bypass;
        result = init(0);
    }
    if (result != CUDA_SUCCESS) {
        selection.driver_error = result;
        return false;
    }
    int physical_count = 0;
    {
        InternalCallScope bypass;
        result = get_count(&physical_count);
    }
    if (result != CUDA_SUCCESS || physical_count <= 0) {
        selection.driver_error = result == CUDA_SUCCESS ? CUDA_ERROR_NO_DEVICE : result;
        return false;
    }

    bool found[4] = {false, false, false, false};
    for (int ordinal = 0; ordinal < physical_count; ++ordinal) {
        CUdevice device = -1;
        CUuuid uuid{};
        int major = 0;
        int minor = 0;
        {
            InternalCallScope bypass;
            result = get_device(&device, ordinal);
            if (result == CUDA_SUCCESS) result = get_uuid(&uuid, device);
            if (result == CUDA_SUCCESS) result = get_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            if (result == CUDA_SUCCESS) result = get_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        }
        if (result != CUDA_SUCCESS) {
            selection.driver_error = result;
            return false;
        }
        if (major != 7 || minor != 0) continue;
        for (int wanted_index = 0; wanted_index < wanted_count; ++wanted_index) {
            if (std::memcmp(uuid.bytes, wanted[static_cast<std::size_t>(wanted_index)].data(), 16) == 0) {
                if (found[wanted_index]) {
                    selection.driver_error = CUDA_ERROR_INVALID_DEVICE;
                    return false;
                }
                selection.ordinal[wanted_index] = ordinal;
                selection.device[wanted_index] = device;
                selection.uuid[wanted_index] = uuid;
                found[wanted_index] = true;
            }
        }
    }
    for (int i = 0; i < wanted_count; ++i) {
        if (!found[i]) {
            selection.driver_error = CUDA_ERROR_INVALID_DEVICE;
            return false;
        }
    }
    selection.count = wanted_count;
    selection.driver_error = CUDA_SUCCESS;
    selection.valid = true;
    return true;
}

CUresult device_map_error() {
    if (ensure_device_selection()) return CUDA_SUCCESS;
    return selection.driver_error;
}

bool logical_device(int logical, int &ordinal, CUdevice &device) {
    if (!ensure_device_selection()) return false;
    const uint32_t mode = current_mode();
    const int logical_count = mode == 2 ? selection.count / 2 : 1;
    if (logical < 0 || logical >= logical_count) return false;
    const int selection_index = mode == 2 ? logical * 2 : 0;
    ordinal = selection.ordinal[selection_index];
    device = selection.device[selection_index];
    return true;
}

uint32_t visible_logical_device_count() {
    if (!ensure_device_selection()) return 0;
    return current_mode() == 2
        ? static_cast<uint32_t>(selection.count / 2) : 1u;
}

bool map_device_id(CUdevice logical, CUdevice &physical) {
    int ordinal = -1;
    return logical_device(static_cast<int>(logical), ordinal, physical);
}

int logical_ordinal_for_physical(int physical) {
    if (!ensure_device_selection()) return -1;
    const int logical_count = current_mode() == 2 ? selection.count / 2 : 1;
    for (int i = 0; i < logical_count; ++i) {
        const int selection_index = current_mode() == 2 ? i * 2 : 0;
        if (selection.ordinal[selection_index] == physical) return i;
    }
    return -1;
}

int pair_index_for_native_device(CUdevice device) {
    if (!ensure_device_selection()) return -1;
    const int pair_count = pair_mode() ? selection.count / 2 : 1;
    for (int pair = 0; pair < pair_count; ++pair) {
        const int first_owner = pair_mode() ? pair * 2 : 0;
        const int owner_count = pair_mode() ? 2 : 1;
        for (int owner = 0; owner < owner_count; ++owner) {
            if (selection.device[first_owner + owner] == device) return pair;
        }
    }
    return -1;
}

bool map_external_vmm_device(CUdevice requested,
                             v200::frontdoor::detail::NativePairLocation *mapped) {
    if (mapped == nullptr || !ensure_device_selection()) return false;
    std::array<int, 4> selected{};
    for (int i = 0; i < selection.count; ++i)
        selected[static_cast<size_t>(i)] = static_cast<int>(selection.device[i]);
    return v200::frontdoor::detail::map_external_vmm_device(
        static_cast<int>(requested), selected.data(), selection.count, mapped);
}

CUresult pair_index_for_context(CUcontext context, uint32_t *pair_index) {
    if (context == nullptr || pair_index == nullptr) return CUDA_ERROR_INVALID_VALUE;
    using GetCurrent = CUresult (CUDAAPI *)(CUcontext *);
    using Push = CUresult (CUDAAPI *)(CUcontext);
    using Pop = CUresult (CUDAAPI *)(CUcontext *);
    using GetDevice = CUresult (CUDAAPI *)(CUdevice *);
    auto get_current = next_fn<GetCurrent>("cuCtxGetCurrent");
    auto push = next_fn<Push>("cuCtxPushCurrent_v2");
    if (push == nullptr) push = next_fn<Push>("cuCtxPushCurrent");
    auto pop = next_fn<Pop>("cuCtxPopCurrent_v2");
    if (pop == nullptr) pop = next_fn<Pop>("cuCtxPopCurrent");
    auto get_device = next_fn<GetDevice>("cuCtxGetDevice");
    if (get_current == nullptr || push == nullptr || pop == nullptr || get_device == nullptr)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUcontext previous = nullptr;
    CUresult result;
    {
        InternalCallScope bypass;
        result = get_current(&previous);
    }
    if (result != CUDA_SUCCESS) return result;
    {
        InternalCallScope bypass;
        result = push(context);
    }
    if (result != CUDA_SUCCESS) return result;
    CUdevice device = -1;
    {
        InternalCallScope bypass;
        result = get_device(&device);
    }
    CUcontext popped = nullptr;
    CUresult pop_status;
    {
        InternalCallScope bypass;
        pop_status = pop(&popped);
    }
    if (result == CUDA_SUCCESS && pop_status != CUDA_SUCCESS) result = pop_status;
    if (result != CUDA_SUCCESS) return result;
    if (popped != context) return CUDA_ERROR_INVALID_CONTEXT;
    const int mapped = pair_index_for_native_device(device);
    if (mapped < 0) return CUDA_ERROR_INVALID_DEVICE;
    *pair_index = static_cast<uint32_t>(mapped);
    (void)previous;
    return CUDA_SUCCESS;
}

void remember_event_owner(cudaEvent_t event, uint32_t pair) {
    if (event == nullptr) return;
    std::lock_guard<std::mutex> lock(event_owner_mutex);
    event_owner_pairs[reinterpret_cast<uintptr_t>(event)] = pair;
}

int known_event_owner(cudaEvent_t event) {
    if (event == nullptr) return -1;
    std::lock_guard<std::mutex> lock(event_owner_mutex);
    const auto found = event_owner_pairs.find(reinterpret_cast<uintptr_t>(event));
    return found == event_owner_pairs.end() ? -1 : static_cast<int>(found->second);
}

void forget_event_owner(cudaEvent_t event) {
    if (event == nullptr) return;
    std::lock_guard<std::mutex> lock(event_owner_mutex);
    event_owner_pairs.erase(reinterpret_cast<uintptr_t>(event));
}

cudaError_t cuda_error_for(CUresult result) {
    switch (result) {
        case CUDA_SUCCESS: return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
        case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
        case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
        case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
        case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
        default: return cudaErrorUnknown;
    }
}

CUresult driver_error_for(cudaError_t result) {
    switch (result) {
        case cudaSuccess: return CUDA_SUCCESS;
        case cudaErrorInvalidValue: return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation: return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorInitializationError: return CUDA_ERROR_NOT_INITIALIZED;
        case cudaErrorInvalidDevice: return CUDA_ERROR_INVALID_DEVICE;
        case cudaErrorNotReady: return CUDA_ERROR_NOT_READY;
        case cudaErrorNotSupported: return CUDA_ERROR_NOT_SUPPORTED;
        default: return CUDA_ERROR_UNKNOWN;
    }
}

using MemConfigured = int (*)();
using MemOwnsPointer = int (*)(const void *);
using MemAllocateOn = CUresult (*)(CUdevice, size_t, size_t, void **);
using MemReleasePointer = CUresult (*)(void *);
using MemVmmAddressReserve = CUresult (*)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
using MemVmmAddressFree = CUresult (*)(CUdeviceptr, size_t);
using MemVmmCreate = CUresult (*)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
using MemVmmGranularity = CUresult (*)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
using MemVmmMap = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
using MemVmmSetAccess = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
using MemVmmUnmap = CUresult (*)(CUdeviceptr, size_t);
using MemVmmRelease = CUresult (*)(CUmemGenericAllocationHandle);
using MemOwnsHandle = int (*)(CUmemGenericAllocationHandle);
using MemRetainAllUse = CUresult (*)(uint64_t, uint64_t, uint64_t);
using MemBeginSubmission = CUresult (*)(uint64_t *);
using MemEndSubmission = CUresult (*)(uint64_t, int);

template <class Function>
Function memory_fn(const char *name) {
    return reinterpret_cast<Function>(lookup_v200(name));
}

int memory_pair_for_range(const void *ptr, size_t size) {
    if (!pair_mode() || !ensure_device_selection()) return -1;
    using OwnsRange = int (*)(uint32_t, const void *, size_t);
    auto owns = memory_fn<OwnsRange>("v200_memory_owns_range_for_pair");
    if (owns == nullptr) return -1;
    const uint32_t pair_count = visible_logical_device_count();
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
        if (owns(pair, ptr, size) != 0) return static_cast<int>(pair);
    }
    return -1;
}

CUresult ensure_pair_profile();
cudaError_t prepare_runtime_call();

bool pair_memory_ready() {
    if (!pair_mode() || ensure_pair_profile() != CUDA_SUCCESS) return false;
    auto configured = memory_fn<MemConfigured>("v200_memory_is_configured");
    return configured != nullptr && configured() != 0;
}

bool execution_attached() noexcept { return execution_handle() != nullptr; }

class StreamPairScope {
public:
    explicit StreamPairScope(void *stream) noexcept : previous_(current_pair()) {
        const int owner = stream_owner_pair(stream);
        if (owner < 0 || static_cast<uint32_t>(owner) == previous_) return;
        if (select_pair(static_cast<uint32_t>(owner)) != 0) {
            status_ = cudaErrorInvalidResourceHandle;
            return;
        }
        switched_ = true;
        status_ = prepare_runtime_call();
        if (status_ != cudaSuccess) {
            (void)select_pair(previous_);
            switched_ = false;
        }
    }

    ~StreamPairScope() {
        if (switched_) {
            (void)select_pair(previous_);
            (void)prepare_runtime_call();
        }
    }

    cudaError_t status() const noexcept { return status_; }
    StreamPairScope(const StreamPairScope &) = delete;
    StreamPairScope &operator=(const StreamPairScope &) = delete;

private:
    uint32_t previous_ = 0;
    bool switched_ = false;
    cudaError_t status_ = cudaSuccess;
};

class ExplicitPairScope {
public:
    explicit ExplicitPairScope(uint32_t pair) noexcept : previous_(current_pair()) {
        if (!pair_mode() || pair >= visible_logical_device_count()) {
            status_ = cudaErrorInvalidDevice;
            return;
        }
        if (select_pair(pair) != 0) {
            status_ = cudaErrorInvalidDevice;
            return;
        }
        switched_ = pair != previous_;
        status_ = prepare_runtime_call();
        if (status_ != cudaSuccess && switched_) {
            (void)select_pair(previous_);
            switched_ = false;
        }
    }

    ~ExplicitPairScope() {
        if (!switched_) return;
        (void)select_pair(previous_);
        (void)prepare_runtime_call();
    }

    cudaError_t status() const noexcept { return status_; }
    ExplicitPairScope(const ExplicitPairScope &) = delete;
    ExplicitPairScope &operator=(const ExplicitPairScope &) = delete;

private:
    uint32_t previous_ = 0;
    bool switched_ = false;
    cudaError_t status_ = cudaSuccess;
};

class EventPairScope {
public:
    explicit EventPairScope(cudaEvent_t event) noexcept : previous_(current_pair()) {
        const int owner = known_event_owner(event);
        if (owner < 0 || static_cast<uint32_t>(owner) == previous_) return;
        if (select_pair(static_cast<uint32_t>(owner)) != 0) {
            status_ = cudaErrorInvalidResourceHandle;
            return;
        }
        switched_ = true;
        status_ = prepare_runtime_call();
        if (status_ != cudaSuccess) {
            (void)select_pair(previous_);
            switched_ = false;
        }
    }

    ~EventPairScope() {
        if (!switched_) return;
        (void)select_pair(previous_);
        (void)prepare_runtime_call();
    }

    cudaError_t status() const noexcept { return status_; }
    EventPairScope(const EventPairScope &) = delete;
    EventPairScope &operator=(const EventPairScope &) = delete;

private:
    uint32_t previous_ = 0;
    bool switched_ = false;
    cudaError_t status_ = cudaSuccess;
};

class DriverPairTlsScope {
public:
    explicit DriverPairTlsScope(uint32_t pair) noexcept : previous_(current_pair()) {
        if (pair >= visible_logical_device_count() || select_pair(pair) != 0) {
            status_ = CUDA_ERROR_INVALID_DEVICE;
            return;
        }
        switched_ = pair != previous_;
    }

    ~DriverPairTlsScope() {
        if (switched_) (void)select_pair(previous_);
    }

    CUresult status() const noexcept { return status_; }
    DriverPairTlsScope(const DriverPairTlsScope &) = delete;
    DriverPairTlsScope &operator=(const DriverPairTlsScope &) = delete;

private:
    uint32_t previous_ = 0;
    bool switched_ = false;
    CUresult status_ = CUDA_SUCCESS;
};

CUresult require_pair_profile() {
    if (!pair_mode()) return CUDA_SUCCESS;
    return ensure_pair_profile();
}

CUresult ensure_pair_profile(uint32_t requested_pair) {
    if (!pair_mode()) return CUDA_ERROR_INVALID_VALUE;
    if (!ensure_device_selection()) return selection.driver_error;
    if (requested_pair >= static_cast<uint32_t>(selection.count / 2))
        return CUDA_ERROR_INVALID_DEVICE;
    const uint32_t previous_pair = current_pair();
    const int selected = select_pair(requested_pair);
    if (selected != 0) return CUDA_ERROR_INVALID_DEVICE;
    if (profile_initializing[requested_pair]) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lock(profile_mutex);
    if (profile_state[requested_pair] == 2) {
        (void)select_pair(previous_pair);
        return CUDA_SUCCESS;
    }
    if (profile_state[requested_pair] == 3 || profile_state[requested_pair] == 1) {
        const CUresult saved = profile_error[requested_pair];
        (void)select_pair(previous_pair);
        return saved;
    }
    profile_state[requested_pair] = 1;
    profile_initializing[requested_pair] = true;
    using InitializePair = CUresult (*)(uint32_t);
    using InitializeSingle = CUresult (*)();
    auto initialize_pair = memory_fn<InitializePair>("v200_initialize_profile_pair");
    auto initialize_single = memory_fn<InitializeSingle>("v200_initialize_profile");
    if (initialize_pair != nullptr) {
        InternalCallScope bypass;
        profile_error[requested_pair] = initialize_pair(requested_pair);
    } else if (requested_pair == 0 && initialize_single != nullptr) {
        InternalCallScope bypass;
        profile_error[requested_pair] = initialize_single();
    } else {
        profile_error[requested_pair] = CUDA_ERROR_NOT_INITIALIZED;
    }
    profile_initializing[requested_pair] = false;
    if (profile_error[requested_pair] == CUDA_SUCCESS) {
        auto configured = memory_fn<MemConfigured>("v200_memory_is_configured");
        auto retain_all = memory_fn<MemRetainAllUse>("v200_memory_retain_all_use");
        auto begin = memory_fn<MemBeginSubmission>("v200_memory_begin_submission");
        auto end = memory_fn<MemEndSubmission>("v200_memory_end_submission");
        using Forward = int (*)(v200_execution_handle *, v200_logical_device_id,
                                v200_logical_stream_id, v200_owner_id, void *,
                                v200_execution_enqueue_fn, void *, int *);
        auto forward = memory_fn<Forward>("v200_execution_forward_native");
        if (configured == nullptr || configured() == 0 || retain_all == nullptr ||
            begin == nullptr || end == nullptr || forward == nullptr || !execution_attached()) {
            profile_error[requested_pair] = CUDA_ERROR_NOT_INITIALIZED;
        }
    }
    profile_state[requested_pair] = profile_error[requested_pair] == CUDA_SUCCESS ? 2 : 3;
    const CUresult result = profile_error[requested_pair];
    if (result == CUDA_SUCCESS) {
        v200_frontdoor_snapshot("profile_initialized");
        // Register after the pair manager and symbol-resolution caches have
        // been constructed. Its callback then runs before those destructors.
        register_process_exit_snapshot(requested_pair);
    }
    (void)select_pair(previous_pair);
    return result;
}

CUresult ensure_pair_profile() {
    return ensure_pair_profile(current_pair());
}

CUresult native_owner_free_bytes(CUdevice device, size_t *free_bytes) {
    using Retain = CUresult (CUDAAPI *)(CUcontext *, CUdevice);
    using Release = CUresult (CUDAAPI *)(CUdevice);
    using Push = CUresult (CUDAAPI *)(CUcontext);
    using Pop = CUresult (CUDAAPI *)(CUcontext *);
    using GetInfo = CUresult (CUDAAPI *)(size_t *, size_t *);
    auto retain = next_fn<Retain>("cuDevicePrimaryCtxRetain");
    auto release = next_fn<Release>("cuDevicePrimaryCtxRelease_v2");
    if (release == nullptr) release = next_fn<Release>("cuDevicePrimaryCtxRelease");
    auto push = next_fn<Push>("cuCtxPushCurrent_v2");
    if (push == nullptr) push = next_fn<Push>("cuCtxPushCurrent");
    auto pop = next_fn<Pop>("cuCtxPopCurrent_v2");
    if (pop == nullptr) pop = next_fn<Pop>("cuCtxPopCurrent");
    auto get_info = next_fn<GetInfo>("cuMemGetInfo_v2");
    if (get_info == nullptr) get_info = next_fn<GetInfo>("cuMemGetInfo");
    if (retain == nullptr || release == nullptr || push == nullptr || pop == nullptr ||
        get_info == nullptr) return CUDA_ERROR_NOT_SUPPORTED;

    CUcontext context = nullptr;
    {
        InternalCallScope bypass;
        const CUresult retained = retain(&context, device);
        if (retained != CUDA_SUCCESS) return retained;
    }
    CUresult status = CUDA_SUCCESS;
    bool pushed = false;
    {
        InternalCallScope bypass;
        status = push(context);
        pushed = status == CUDA_SUCCESS;
    }
    size_t total = 0;
    if (status == CUDA_SUCCESS) {
        InternalCallScope bypass;
        status = get_info(free_bytes, &total);
    }
    CUresult pop_status = CUDA_SUCCESS;
    if (pushed) {
        CUcontext restored = nullptr;
        InternalCallScope bypass;
        pop_status = pop(&restored);
        if (pop_status == CUDA_SUCCESS && restored != context)
            pop_status = CUDA_ERROR_INVALID_CONTEXT;
    }
    // A failed pop leaves the caller's context stack uncertain. Preserve the
    // extra primary-context retain so the context remains alive for recovery.
    if (pushed && pop_status != CUDA_SUCCESS) return pop_status;
    CUresult release_status;
    {
        InternalCallScope bypass;
        release_status = release(device);
    }
    if (release_status != CUDA_SUCCESS) return release_status;
    return status;
}

CUresult pair_memory_capacity(uint32_t pair_index, size_t *free_bytes,
                              size_t *total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
    const CUresult ready = ensure_pair_profile(pair_index);
    if (ready != CUDA_SUCCESS) return ready;
    auto get_usage = memory_fn<CUresult (*)(uint32_t, v200_memory_usage_info *)>(
        "v200_memory_get_usage_for_pair");
    auto get_profile = memory_fn<CUresult (*)(uint32_t, v200_memory_pair_profile *)>(
        "v200_memory_get_profile_for_pair");
    if (get_usage == nullptr || get_profile == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    const CUresult queried = get_usage(pair_index, &usage);
    if (queried != CUDA_SUCCESS) return queried;
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    const CUresult profile_status = get_profile(pair_index, &profile);
    if (profile_status != CUDA_SUCCESS) return profile_status;

    uint64_t free_total = 0;
    uint64_t capacity_total = 0;
    for (size_t owner = 0; owner != 2; ++owner) {
        const uint64_t budget = usage.owner_budget_bytes[owner];
        const uint64_t used = usage.owner_used_bytes[owner];
        if (used > budget || capacity_total > std::numeric_limits<uint64_t>::max() - budget)
            return CUDA_ERROR_INVALID_VALUE;
        capacity_total += budget;
        size_t physical_free = 0;
        const CUresult physical_status = native_owner_free_bytes(profile.owner_device[owner],
                                                                 &physical_free);
        if (physical_status != CUDA_SUCCESS) return physical_status;
        constexpr uint64_t reserve = 1ULL << 30;
        const uint64_t physical_available = physical_free > reserve ? physical_free - reserve : 0;
        const uint64_t ledger_available = budget - used;
        const uint64_t owner_available = std::min<uint64_t>(ledger_available, physical_available);
        if (free_total > std::numeric_limits<uint64_t>::max() - owner_available)
            return CUDA_ERROR_INVALID_VALUE;
        free_total += owner_available;
    }
    if (capacity_total > std::numeric_limits<size_t>::max() ||
        free_total > std::numeric_limits<size_t>::max()) return CUDA_ERROR_INVALID_VALUE;
    *free_bytes = static_cast<size_t>(free_total);
    *total_bytes = static_cast<size_t>(capacity_total);
    return CUDA_SUCCESS;
}

CUresult synchronize_native_anchor() {
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return driver_error_for(prepared);
    using Fn = CUresult (CUDAAPI *)(void);
    auto synchronize = next_fn<Fn>("cuCtxSynchronize");
    if (synchronize == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    InternalCallScope bypass;
    return synchronize();
}

CUresult wait_stream_prefix(void *stream) {
    if (!execution_attached()) return CUDA_SUCCESS;
    v200_completion_token token{};
    const int current = execution_token_for_stream(stream, &token);
    if (current != V200_EXECUTION_OK) return CUDA_ERROR_NOT_SUPPORTED;
    int driver_result = CUDA_SUCCESS;
    const int waited = execution_wait_token(token, &driver_result);
    (void)execution_release_token(token);
    return waited == V200_EXECUTION_OK ? static_cast<CUresult>(driver_result) : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult wait_device_prefix() {
    if (!execution_attached()) return CUDA_SUCCESS;
    v200_completion_token token{};
    const int current = execution_token_for_device(&token);
    if (current != V200_EXECUTION_OK) return CUDA_ERROR_NOT_SUPPORTED;
    int driver_result = CUDA_SUCCESS;
    const int waited = execution_wait_token(token, &driver_result);
    (void)execution_release_token(token);
    return waited == V200_EXECUTION_OK ? static_cast<CUresult>(driver_result) : CUDA_ERROR_NOT_SUPPORTED;
}

template <class Result>
Result frontdoor_error(CUresult result) {
    if constexpr (std::is_same_v<Result, cudaError_t>) return cuda_error_for(result);
    else return static_cast<Result>(result);
}

template <class Call>
int execution_forward_thunk(void *user_data, void *native_stream) {
    auto *call = static_cast<Call *>(user_data);
    InternalCallScope bypass;
    return static_cast<int>((*call)(native_stream));
}

template <class Call>
auto dispatch_async(void *stream, Call &&call) -> decltype(call(stream)) {
    using Result = decltype(call(stream));
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess)
        return frontdoor_error<Result>(driver_error_for(pair_scope.status()));
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return frontdoor_error<Result>(profile);

    // CUDA capture rejects the completion-event insertion performed by the
    // ordinary submission path. With graph support explicitly enabled, query
    // capture state and preserve the caller's native anchor operation exactly;
    // one tracked graph launch accounts for all captured memory uses on replay.
    if (graph_capture_enabled()) {
        using IsCapturing = cudaError_t (CUDARTAPI *)(cudaStream_t, cudaStreamCaptureStatus *);
        auto is_capturing = next_fn<IsCapturing>("cudaStreamIsCapturing");
        if (is_capturing == nullptr) return frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        cudaError_t query_status;
        {
            InternalCallScope bypass;
            query_status = is_capturing(reinterpret_cast<cudaStream_t>(stream), &status);
        }
        if (query_status != cudaSuccess) return static_cast<Result>(query_status);
        if (status == cudaStreamCaptureStatusInvalidated)
            return static_cast<Result>(cudaErrorStreamCaptureInvalidated);
        if (status == cudaStreamCaptureStatusActive) {
            CaptureNativeForwardScope capture_scope;
            InternalCallScope bypass;
            return call(stream);
        }
    }

    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    auto stream_gate = gate_for_stream(stream);
    std::lock_guard<std::mutex> stream_lock(*stream_gate);

    if (!execution_attached()) {
        if (pair_mode()) return frontdoor_error<Result>(CUDA_ERROR_NOT_INITIALIZED);
        InternalCallScope bypass;
        return call(stream);
    }
    const int registered = register_execution_stream(stream);
    if (registered != V200_EXECUTION_OK && registered != V200_EXECUTION_DUPLICATE_BINDING) {
        return frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
    }

    using Forward = int (*)(v200_execution_handle *, v200_logical_device_id,
                            v200_logical_stream_id, v200_owner_id, void *,
                            v200_execution_enqueue_fn, void *, int *);
    auto forward = memory_fn<Forward>("v200_execution_forward_native");
    auto end = memory_fn<MemEndSubmission>("v200_memory_end_submission");
    uint64_t lease = 0;
    if (pair_mode()) {
        auto begin = memory_fn<MemBeginSubmission>("v200_memory_begin_submission");
        if (begin == nullptr || end == nullptr || forward == nullptr)
            return frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
        const CUresult started = begin(&lease);
        if (started != CUDA_SUCCESS) return frontdoor_error<Result>(started);
        if (lease == 0) return frontdoor_error<Result>(CUDA_ERROR_NOT_INITIALIZED);
    }

    int forwarded_result = CUDA_ERROR_NOT_SUPPORTED;
    int adapter_status = V200_EXECUTION_UNAVAILABLE;
    if (forward != nullptr) {
        adapter_status = forward(execution_handle(), execution_device(), stream_id(stream),
                                 execution_owner(), stream,
                                 &execution_forward_thunk<std::remove_reference_t<Call>>,
                                 &call, &forwarded_result);
    } else {
        if (pair_mode()) return frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
        Result result;
        {
            InternalCallScope bypass;
            result = call(stream);
        }
        forwarded_result = static_cast<int>(result);
        adapter_status = observe_execution_submission(stream, forwarded_result);
    }

    Result result = static_cast<Result>(forwarded_result);
    bool completion_recorded = adapter_status == V200_EXECUTION_OK &&
                               forwarded_result == static_cast<int>(Result{});
    if (completion_recorded && pair_mode()) {
        v200_completion_token token{};
        const int token_status = execution_token_for_stream(stream, &token);
        auto retain = memory_fn<MemRetainAllUse>("v200_memory_retain_all_use");
        const CUresult retained = token_status == V200_EXECUTION_OK && retain != nullptr
            ? retain(stream_id(stream), token.slot, token.generation) : CUDA_ERROR_NOT_SUPPORTED;
        if (token_status == V200_EXECUTION_OK) (void)execution_release_token(token);
        completion_recorded = retained == CUDA_SUCCESS;
    }
    if (pair_mode() && lease != 0) {
        const CUresult ended = end(lease, completion_recorded ? 1 : 0);
        if (ended != CUDA_SUCCESS && forwarded_result == static_cast<int>(Result{})) {
            result = frontdoor_error<Result>(ended);
        }
    }
    if (adapter_status != V200_EXECUTION_OK && forwarded_result == static_cast<int>(Result{})) {
        result = frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
    } else if (!completion_recorded && pair_mode() && forwarded_result == static_cast<int>(Result{})) {
        result = frontdoor_error<Result>(CUDA_ERROR_NOT_SUPPORTED);
    }
    return result;
}

int unregister_execution_stream(void *stream) {
    auto *handle = execution_handle();
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    using Unregister = int (*)(v200_execution_handle *, v200_logical_device_id,
                               v200_logical_stream_id, v200_owner_id);
    auto fn = memory_fn<Unregister>("v200_execution_unregister_stream");
    return fn ? fn(handle, execution_device(), stream_id(stream), execution_owner())
              : V200_EXECUTION_UNAVAILABLE;
}

} // namespace

namespace {

bool has_suffix(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

void *driver_wrapper_for(std::string_view requested, bool per_thread) {
    if (has_suffix(requested, "_ptsz")) requested.remove_suffix(5);
    if (requested == "cuGetProcAddress") requested = "cuGetProcAddress_v2";
    if (requested == "cuDeviceTotalMem") requested = "cuDeviceTotalMem_v2";
    if (requested == "cuMemGetInfo") requested = "cuMemGetInfo_v2";
    if (requested == "cuMemcpyHtoDAsync") requested = "cuMemcpyHtoDAsync_v2";
    if (requested == "cuMemcpyDtoHAsync") requested = "cuMemcpyDtoHAsync_v2";
    if (requested == "cuMemcpyDtoDAsync") requested = "cuMemcpyDtoDAsync_v2";

    if (requested == "cuDeviceGetCount") return reinterpret_cast<void *>(&cuDeviceGetCount);
    if (requested == "cuDeviceGet") return reinterpret_cast<void *>(&cuDeviceGet);
    if (requested == "cuDeviceTotalMem_v2") return reinterpret_cast<void *>(&cuDeviceTotalMem_v2);
    if (requested == "cuGetProcAddress_v2") return reinterpret_cast<void *>(&cuGetProcAddress_v2);
    if (requested == "cuMemAddressReserve") return reinterpret_cast<void *>(&cuMemAddressReserve);
    if (requested == "cuMemAddressFree") return reinterpret_cast<void *>(&cuMemAddressFree);
    if (requested == "cuMemCreate") return reinterpret_cast<void *>(&cuMemCreate);
    if (requested == "cuMemGetInfo_v2") return reinterpret_cast<void *>(&cuMemGetInfo_v2);
    if (requested == "cuMemGetAllocationGranularity") return reinterpret_cast<void *>(&cuMemGetAllocationGranularity);
    if (requested == "cuMemMap") return reinterpret_cast<void *>(&cuMemMap);
    if (requested == "cuMemSetAccess") return reinterpret_cast<void *>(&cuMemSetAccess);
    if (requested == "cuMemUnmap") return reinterpret_cast<void *>(&cuMemUnmap);
    if (requested == "cuMemRelease") return reinterpret_cast<void *>(&cuMemRelease);
    if (requested == "cuMemcpyHtoDAsync_v2")
        return per_thread ? reinterpret_cast<void *>(&cuMemcpyHtoDAsync_v2_ptsz)
                          : reinterpret_cast<void *>(&cuMemcpyHtoDAsync_v2);
    if (requested == "cuMemcpyDtoHAsync_v2")
        return per_thread ? reinterpret_cast<void *>(&cuMemcpyDtoHAsync_v2_ptsz)
                          : reinterpret_cast<void *>(&cuMemcpyDtoHAsync_v2);
    if (requested == "cuMemcpyDtoDAsync_v2")
        return per_thread ? reinterpret_cast<void *>(&cuMemcpyDtoDAsync_v2_ptsz)
                          : reinterpret_cast<void *>(&cuMemcpyDtoDAsync_v2);
    if (requested == "cuMemsetD8Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD8Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD8Async);
    if (requested == "cuMemsetD16Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD16Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD16Async);
    if (requested == "cuMemsetD32Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD32Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD32Async);
    if (requested == "cuMemsetD2D8Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD2D8Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD2D8Async);
    if (requested == "cuMemsetD2D16Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD2D16Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD2D16Async);
    if (requested == "cuMemsetD2D32Async")
        return per_thread ? reinterpret_cast<void *>(&cuMemsetD2D32Async_ptsz)
                          : reinterpret_cast<void *>(&cuMemsetD2D32Async);
    if (requested == "cuLaunchKernel")
        return per_thread ? reinterpret_cast<void *>(&cuLaunchKernel_ptsz)
                          : reinterpret_cast<void *>(&cuLaunchKernel);
    if (requested == "cuStreamSynchronize")
        return per_thread ? reinterpret_cast<void *>(&cuStreamSynchronize_ptsz)
                          : reinterpret_cast<void *>(&cuStreamSynchronize);
    if (requested == "cuStreamWaitEvent")
        return per_thread ? reinterpret_cast<void *>(&cuStreamWaitEvent_ptsz)
                          : reinterpret_cast<void *>(&cuStreamWaitEvent);
    return nullptr;
}

const v200::frontdoor::ApiEntry *driver_api_profile(std::string_view requested) {
    if (has_suffix(requested, "_ptsz")) requested.remove_suffix(5);
    if (requested == "cuGetProcAddress") requested = "cuGetProcAddress_v2";
    if (requested == "cuDeviceTotalMem") requested = "cuDeviceTotalMem_v2";
    if (requested == "cuMemGetInfo") requested = "cuMemGetInfo_v2";
    if (requested == "cuMemcpyHtoDAsync") requested = "cuMemcpyHtoDAsync_v2";
    if (requested == "cuMemcpyDtoHAsync") requested = "cuMemcpyDtoHAsync_v2";
    if (requested == "cuMemcpyDtoDAsync") requested = "cuMemcpyDtoDAsync_v2";
    return v200::frontdoor::find_api(requested);
}

bool native_proc_is_per_thread(const char *requested, void *native_pointer) {
    if (requested == nullptr || native_pointer == nullptr) return false;
    std::string_view name(requested);
    if (has_suffix(name, "_ptsz")) return true;
    std::string_view base = name;
    if (base == "cuMemcpyHtoDAsync") base = "cuMemcpyHtoDAsync_v2";
    else if (base == "cuMemcpyDtoHAsync") base = "cuMemcpyDtoHAsync_v2";
    else if (base == "cuMemcpyDtoDAsync") base = "cuMemcpyDtoDAsync_v2";
    if (base == "cuMemcpyHtoDAsync_v2" || base == "cuMemcpyDtoHAsync_v2" ||
        base == "cuMemcpyDtoDAsync_v2" || base == "cuLaunchKernel" ||
        base == "cuMemsetD8Async" || base == "cuMemsetD16Async" ||
        base == "cuMemsetD32Async" || base == "cuMemsetD2D8Async" ||
        base == "cuMemsetD2D16Async" || base == "cuMemsetD2D32Async" ||
        base == "cuStreamSynchronize" || base == "cuStreamWaitEvent") {
        std::string symbol(base);
        symbol.append("_ptsz");
        return lookup_driver(symbol.c_str()) == native_pointer;
    }
    return false;
}

bool is_pair_safe_native_passthrough(std::string_view requested) {
    const auto *profile = driver_api_profile(requested);
    return profile != nullptr &&
           profile->disposition == v200::frontdoor::ApiDisposition::native_anchor_passthrough;
}

} // namespace

extern "C" {

CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **function,
                                     int cuda_version, cuuint64_t flags,
                                     CUdriverProcAddressQueryResult *symbol_status) {
    using Fn = CUresult (CUDAAPI *)(const char *, void **, int, cuuint64_t,
                                    CUdriverProcAddressQueryResult *);
    auto real = reinterpret_cast<Fn>(lookup_driver("cuGetProcAddress_v2"));
    if (internal_call() || !presentation_enabled())
        return real ? real(symbol, function, cuda_version, flags, symbol_status)
                    : CUDA_ERROR_NOT_SUPPORTED;
    if (real == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult result;
    {
        InternalCallScope bypass;
        result = real(symbol, function, cuda_version, flags, symbol_status);
    }
    if (result != CUDA_SUCCESS || symbol == nullptr || function == nullptr || *function == nullptr)
        return result;

    const auto *entry = driver_api_profile(symbol);
    const bool per_thread = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0 ||
                            ((flags & (CU_GET_PROC_ADDRESS_LEGACY_STREAM |
                                       CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)) == 0 &&
                             native_proc_is_per_thread(symbol, *function));
    void *wrapper = driver_wrapper_for(symbol, per_thread);
    if (entry != nullptr && entry->disposition == v200::frontdoor::ApiDisposition::wrapped) {
        if (cuda_version < entry->minimum_cuda_version) {
            if (pair_mode()) {
                *function = nullptr;
                if (symbol_status != nullptr)
                    *symbol_status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            return result;
        }
        if (wrapper != nullptr) *function = wrapper;
        else if (pair_mode()) {
            *function = nullptr;
            if (symbol_status != nullptr) *symbol_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    } else if (pair_mode() && !is_pair_safe_native_passthrough(symbol)) {
        *function = nullptr;
        if (symbol_status != nullptr) *symbol_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return result;
}

cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **function,
    unsigned long long flags, enum cudaDriverEntryPointQueryResult *driver_status) {
    using Fn = cudaError_t (CUDARTAPI *)(const char *, void **, unsigned long long,
                                         enum cudaDriverEntryPointQueryResult *);
    auto real = next_fn<Fn>("cudaGetDriverEntryPoint");
    if (internal_call() || !presentation_enabled())
        return real ? real(symbol, function, flags, driver_status) : cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    const cudaError_t result = real(symbol, function, flags, driver_status);
    if (result != cudaSuccess || symbol == nullptr || function == nullptr || *function == nullptr)
        return result;
    const auto *entry = driver_api_profile(symbol);
    const bool per_thread = (flags & cudaEnablePerThreadDefaultStream) != 0 ||
                            ((flags & (cudaEnableLegacyStream |
                                       cudaEnablePerThreadDefaultStream)) == 0 &&
                             native_proc_is_per_thread(symbol, *function));
    void *wrapper = driver_wrapper_for(symbol, per_thread);
    if (entry != nullptr && entry->disposition == v200::frontdoor::ApiDisposition::wrapped) {
        if (wrapper != nullptr) *function = wrapper;
        else if (pair_mode()) {
            *function = nullptr;
            if (driver_status != nullptr)
                *driver_status = cudaDriverEntryPointSymbolNotFound;
            return cudaErrorNotSupported;
        }
    } else if (pair_mode() && !is_pair_safe_native_passthrough(symbol)) {
        *function = nullptr;
        if (driver_status != nullptr) *driver_status = cudaDriverEntryPointSymbolNotFound;
        return cudaErrorNotSupported;
    }
    return result;
}

cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(const char *symbol, void **function,
    unsigned int cuda_version, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driver_status) {
    using Fn = cudaError_t (CUDARTAPI *)(const char *, void **, unsigned int,
        unsigned long long, enum cudaDriverEntryPointQueryResult *);
    auto real = next_fn<Fn>("cudaGetDriverEntryPointByVersion");
    if (internal_call() || !presentation_enabled())
        return real ? real(symbol, function, cuda_version, flags, driver_status)
                    : cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    const cudaError_t result = real(symbol, function, cuda_version, flags, driver_status);
    if (result != cudaSuccess || symbol == nullptr || function == nullptr || *function == nullptr)
        return result;
    const auto *entry = driver_api_profile(symbol);
    const bool per_thread = (flags & cudaEnablePerThreadDefaultStream) != 0 ||
                            ((flags & (cudaEnableLegacyStream |
                                       cudaEnablePerThreadDefaultStream)) == 0 &&
                             native_proc_is_per_thread(symbol, *function));
    void *wrapper = driver_wrapper_for(symbol, per_thread);
    if (entry != nullptr && entry->disposition == v200::frontdoor::ApiDisposition::wrapped) {
        if (cuda_version < static_cast<unsigned int>(entry->minimum_cuda_version)) {
            *function = nullptr;
            if (driver_status != nullptr)
                *driver_status = cudaDriverEntryPointVersionNotSufficent;
            return cudaErrorNotSupported;
        }
        if (wrapper != nullptr) *function = wrapper;
        else if (pair_mode()) {
            *function = nullptr;
            if (driver_status != nullptr)
                *driver_status = cudaDriverEntryPointSymbolNotFound;
            return cudaErrorNotSupported;
        }
    } else if (pair_mode() && !is_pair_safe_native_passthrough(symbol)) {
        *function = nullptr;
        if (driver_status != nullptr) *driver_status = cudaDriverEntryPointSymbolNotFound;
        return cudaErrorNotSupported;
    }
    return result;
}

} // extern "C"

namespace {

int map_runtime_device(int logical, int &physical) {
    CUdevice device = -1;
    if (!logical_device(logical, physical, device)) return -1;
    return 0;
}

cudaError_t prepare_runtime_call() {
    if (!presentation_enabled()) return cudaSuccess;
    if (!ensure_device_selection()) return cuda_error_for(selection.driver_error);
    int physical = -1;
    CUdevice selected = -1;
    if (!logical_device(static_cast<int>(current_pair()), physical, selected))
        return cudaErrorInvalidDevice;
    using Get = cudaError_t (CUDARTAPI *)(int *);
    using Set = cudaError_t (CUDARTAPI *)(int);
    auto get = next_fn<Get>("cudaGetDevice");
    auto set = next_fn<Set>("cudaSetDevice");
    if (!get || !set) return cudaErrorNotSupported;
    int current = -1;
    {
        InternalCallScope bypass;
        const cudaError_t result = get(&current);
        if (result != cudaSuccess) return result;
        if (current == physical) return cudaSuccess;
        return set(physical);
    }
}

cudaError_t runtime_device_ordinal(int logical, int &physical) {
    if (!presentation_enabled()) {
        physical = logical;
        return cudaSuccess;
    }
    if (!ensure_device_selection()) return cuda_error_for(selection.driver_error);
    CUdevice device = -1;
    if (!logical_device(logical, physical, device)) return cudaErrorInvalidDevice;
    return cudaSuccess;
}

cudaError_t register_created_stream(cudaStream_t *stream) {
    if (stream == nullptr) return cudaSuccess;
    if (!execution_attached()) return cudaSuccess;
    const int status = register_execution_stream(*stream);
    if (status == V200_EXECUTION_OK || status == V200_EXECUTION_DUPLICATE_BINDING) return cudaSuccess;
    using Destroy = cudaError_t (CUDARTAPI *)(cudaStream_t);
    auto destroy = next_fn<Destroy>("cudaStreamDestroy");
    if (destroy == nullptr) return cudaErrorNotSupported;
    cudaError_t cleanup;
    {
        InternalCallScope bypass;
        cleanup = destroy(*stream);
    }
    if (cleanup == cudaSuccess) {
        forget_stream_gate(*stream);
        forget_stream(*stream);
        *stream = nullptr;
        return cudaErrorNotSupported;
    }
    return cleanup;
}

cudaError_t prepare_async_call() {
    const CUresult profile = require_pair_profile();
    return profile == CUDA_SUCCESS ? prepare_runtime_call() : cuda_error_for(profile);
}

CUstream per_thread_tracking_stream(CUstream stream) noexcept {
    return stream == nullptr ? CU_STREAM_PER_THREAD : stream;
}

} // namespace

extern "C" {

int v200_frontdoor_select_pair(uint32_t pair_index) {
    if (!pair_mode() || pair_index >= visible_logical_device_count()) return -1;
    return select_pair(pair_index);
}

cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    using Fn = cudaError_t (CUDARTAPI *)(int *);
    auto real = next_fn<Fn>("cudaGetDeviceCount");
    if (internal_call() || !presentation_enabled()) return real ? real(count) : cudaErrorNotSupported;
    if (count == nullptr) return cudaErrorInvalidValue;
    if (real == nullptr) return cudaErrorNotSupported;
    bool selection_ready = false;
    {
        std::lock_guard<std::mutex> lock(selection_mutex);
        selection_ready = selection.valid;
    }
    if (!selection_ready) {
        // A directly linked CUDA Runtime may defer loading libcuda until its
        // first runtime query. Prime that native path before asking the Driver
        // API resolver to enumerate UUIDs; otherwise it sees no loaded
        // libcuda and reports NOT_SUPPORTED on a cold process.
        int native_count = 0;
        cudaError_t native_status;
        {
            InternalCallScope bypass;
            native_status = real(&native_count);
        }
        if (native_status != cudaSuccess) return native_status;
    }
    if (!ensure_device_selection()) return cuda_error_for(selection.driver_error);
    *count = static_cast<int>(visible_logical_device_count());
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    using Fn = cudaError_t (CUDARTAPI *)(int *);
    auto real = next_fn<Fn>("cudaGetDevice");
    if (internal_call() || !presentation_enabled()) return real ? real(device) : cudaErrorNotSupported;
    if (device == nullptr) return cudaErrorInvalidValue;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    *device = pair_mode() ? static_cast<int>(current_pair()) : 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
    using Fn = cudaError_t (CUDARTAPI *)(int);
    auto real = next_fn<Fn>("cudaSetDevice");
    if (internal_call() || !presentation_enabled()) return real ? real(device) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t map_result = runtime_device_ordinal(device, physical);
    if (map_result != cudaSuccess) return map_result;
    if (!real) return cudaErrorNotSupported;
    const uint32_t previous_pair = current_pair();
    if (pair_mode() && select_pair(static_cast<uint32_t>(device)) != 0)
        return cudaErrorInvalidDevice;
    InternalCallScope bypass;
    const cudaError_t result = real(physical);
    if (result != cudaSuccess && pair_mode()) (void)select_pair(previous_pair);
    return result;
}

cudaError_t CUDARTAPI cudaGetDeviceProperties_v2(cudaDeviceProp *properties, int device) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaDeviceProp *, int);
    auto real = next_fn<Fn>("cudaGetDeviceProperties_v2");
    if (!real) real = next_fn<Fn>("cudaGetDeviceProperties");
    if (internal_call() || !presentation_enabled()) return real ? real(properties, device) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t mapped = runtime_device_ordinal(device, physical);
    if (mapped != cudaSuccess) return mapped;
    if (!real) return cudaErrorNotSupported;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    cudaError_t result;
    {
        InternalCallScope bypass;
        result = real(properties, physical);
    }
    if (result != cudaSuccess || properties == nullptr || !pair_mode()) return result;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const CUresult capacity = pair_memory_capacity(static_cast<uint32_t>(device),
                                                   &free_bytes, &total_bytes);
    (void)free_bytes;
    if (capacity != CUDA_SUCCESS) return cuda_error_for(capacity);
    properties->totalGlobalMem = total_bytes;
    return result;
}

#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif
cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp *properties, int device)
    __attribute__((alias("cudaGetDeviceProperties_v2")));

cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, cudaDeviceAttr attribute, int device) {
    using Fn = cudaError_t (CUDARTAPI *)(int *, cudaDeviceAttr, int);
    auto real = next_fn<Fn>("cudaDeviceGetAttribute");
    if (internal_call() || !presentation_enabled()) return real ? real(value, attribute, device) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t mapped = runtime_device_ordinal(device, physical);
    if (mapped != cudaSuccess) return mapped;
    InternalCallScope bypass;
    return real ? real(value, attribute, physical) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pci_bus_id, int length, int device) {
    using Fn = cudaError_t (CUDARTAPI *)(char *, int, int);
    auto real = next_fn<Fn>("cudaDeviceGetPCIBusId");
    if (internal_call() || !presentation_enabled())
        return real ? real(pci_bus_id, length, device) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t mapped = runtime_device_ordinal(device, physical);
    if (mapped != cudaSuccess) return mapped;
    InternalCallScope bypass;
    return real ? real(pci_bus_id, length, physical) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *can_access, int device, int peer_device) {
    using Fn = cudaError_t (CUDARTAPI *)(int *, int, int);
    auto real = next_fn<Fn>("cudaDeviceCanAccessPeer");
    if (internal_call() || !presentation_enabled()) return real ? real(can_access, device, peer_device) : cudaErrorNotSupported;
    int physical = -1;
    int physical_peer = -1;
    cudaError_t mapped = runtime_device_ordinal(device, physical);
    if (mapped != cudaSuccess) return mapped;
    mapped = runtime_device_ordinal(peer_device, physical_peer);
    if (mapped != cudaSuccess) return mapped;
    if (pair_mode() && device != peer_device && selection.count == 4) {
        if (can_access == nullptr) return cudaErrorInvalidValue;
        *can_access = 0;
        return cudaSuccess;
    }
    InternalCallScope bypass;
    return real ? real(can_access, physical, physical_peer) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(int, unsigned int);
    auto real = next_fn<Fn>("cudaDeviceEnablePeerAccess");
    if (internal_call() || !presentation_enabled()) return real ? real(peer_device, flags) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t mapped = runtime_device_ordinal(peer_device, physical);
    if (mapped != cudaSuccess) return mapped;
    if (pair_mode() && selection.count == 4 &&
        static_cast<int>(current_pair()) != peer_device)
        return cudaErrorPeerAccessUnsupported;
    InternalCallScope bypass;
    return real ? real(physical, flags) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaMalloc(void **device_ptr, size_t bytes) {
    using Fn = cudaError_t (CUDARTAPI *)(void **, size_t);
    auto real = next_fn<Fn>("cudaMalloc");
    if (internal_call() || !pair_mode()) {
        const cudaError_t prepared = prepare_runtime_call();
        if (prepared != cudaSuccess) return prepared;
        InternalCallScope bypass;
        return real ? real(device_ptr, bytes) : cudaErrorNotSupported;
    }
    if (device_ptr == nullptr) return cudaErrorInvalidValue;
    const CUresult profile = ensure_pair_profile();
    if (profile != CUDA_SUCCESS) return cuda_error_for(profile);
    auto allocate = memory_fn<MemAllocateOn>("v200_memory_allocate_on");
    void *result = nullptr;
    CUresult allocated = CUDA_ERROR_NOT_SUPPORTED;
    const size_t anchor_index = pair_mode() ? static_cast<size_t>(current_pair()) * 2 : 0;
    if (allocate != nullptr && anchor_index < static_cast<size_t>(selection.count))
        allocated = allocate(selection.device[anchor_index], bytes, 0, &result);
    if (allocated == CUDA_SUCCESS) {
        *device_ptr = result;
        v200_frontdoor_snapshot("cuda_malloc");
    }
    return cuda_error_for(allocated);
}

cudaError_t CUDARTAPI cudaFree(void *device_ptr) {
    using Fn = cudaError_t (CUDARTAPI *)(void *);
    auto real = next_fn<Fn>("cudaFree");
    if (internal_call()) return real ? real(device_ptr) : cudaErrorNotSupported;
    if (pair_mode() && ensure_pair_profile() == CUDA_SUCCESS) {
        using OwnsForPair = int (*)(uint32_t, const void *);
        auto owns = memory_fn<OwnsForPair>("v200_memory_owns_pointer_for_pair");
        const uint32_t previous_pair = current_pair();
        const uint32_t pair_count = visible_logical_device_count();
        int owner_pair = -1;
        if (owns != nullptr) {
            for (uint32_t pair = 0; pair < pair_count; ++pair) {
                if (owns(pair, device_ptr) != 0) {
                    owner_pair = static_cast<int>(pair);
                    break;
                }
            }
        }
        if (owner_pair >= 0) {
            if (select_pair(static_cast<uint32_t>(owner_pair)) != 0)
                return cudaErrorInvalidDevice;
            std::unique_lock<std::shared_mutex> gate(tracked_use_gate);
            const CUresult prefix = wait_device_prefix();
            if (prefix != CUDA_SUCCESS) {
                (void)select_pair(previous_pair);
                (void)prepare_runtime_call();
                return cuda_error_for(prefix);
            }
            const CUresult drained = synchronize_native_anchor();
            if (drained != CUDA_SUCCESS) {
                (void)select_pair(previous_pair);
                (void)prepare_runtime_call();
                return cuda_error_for(drained);
            }
            auto release = memory_fn<MemReleasePointer>("v200_memory_release");
            const CUresult result = release ? release(device_ptr) : CUDA_ERROR_NOT_SUPPORTED;
            v200_frontdoor_snapshot("cuda_free");
            (void)select_pair(previous_pair);
            (void)prepare_runtime_call();
            return cuda_error_for(result);
        }
    }
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    return real ? real(device_ptr) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaMallocManaged(void **device_ptr, size_t bytes, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(void **, size_t, unsigned int);
    auto real = next_fn<Fn>("cudaMallocManaged");
    if (internal_call()) return real ? real(device_ptr, bytes, flags) : cudaErrorNotSupported;
    if (pair_mode()) return cudaErrorNotSupported;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    return real ? real(device_ptr, bytes, flags) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    using Fn = cudaError_t (CUDARTAPI *)(size_t *, size_t *);
    auto real = next_fn<Fn>("cudaMemGetInfo");
    if (internal_call()) return real ? real(free_bytes, total_bytes) : cudaErrorNotSupported;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    if (pair_mode()) {
        if (free_bytes == nullptr || total_bytes == nullptr) return cudaErrorInvalidValue;
        size_t pair_free = 0;
        size_t pair_total = 0;
        const CUresult capacity = pair_memory_capacity(current_pair(), &pair_free,
                                                       &pair_total);
        if (capacity != CUDA_SUCCESS) return cuda_error_for(capacity);
        *free_bytes = pair_free;
        *total_bytes = pair_total;
        return cudaSuccess;
    }
    InternalCallScope bypass;
    return real ? real(free_bytes, total_bytes) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaMemset(void *device_ptr, int value, size_t bytes) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, int, size_t);
    auto real = next_fn<Fn>("cudaMemset");
    if (internal_call()) return real ? real(device_ptr, value, bytes) : cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    if (!real) return cudaErrorNotSupported;
    return dispatch_async(nullptr, [=](void *) {
        return real(device_ptr, value, bytes);
    });
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void *function, dim3 grid, dim3 block,
                                      void **arguments, size_t shared_bytes,
                                      cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    using OperatorTryLaunch = decltype(&v200_operator_try_launch);
    auto real = next_fn<Fn>("cudaLaunchKernel");
    if (internal_call() || !real)
        return real ? real(function, grid, block, arguments, shared_bytes, stream)
                    : cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        const char *enabled = std::getenv("V200_PAIR_COMPUTE");
        if (!capture_native_forward && pair_mode() && enabled != nullptr &&
            std::strcmp(enabled, "1") == 0) {
            const auto operator_launch = reinterpret_cast<OperatorTryLaunch>(
                lookup_v200("v200_operator_try_launch"));
            if (operator_launch != nullptr) {
                cudaError_t operator_status = cudaErrorNotSupported;
                const int handled = operator_launch(
                    function, grid, block, arguments, shared_bytes,
                    reinterpret_cast<cudaStream_t>(native_stream), real, &operator_status);
                if (handled == 1) return operator_status;
                if (handled != 0) return cudaErrorNotSupported;
            }
        }
        return real(function, grid, block, arguments, shared_bytes,
                    reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *function, dim3 grid,
                                                 dim3 block, void **arguments,
                                                 size_t shared_bytes, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaLaunchCooperativeKernel");
    if (internal_call() || !real)
        return real ? real(function, grid, block, arguments, shared_bytes, stream)
                    : cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(function, grid, block, arguments, shared_bytes,
                    reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                         const void *function, void **arguments) {
    using Fn = cudaError_t (CUDARTAPI *)(const cudaLaunchConfig_t *, const void *, void **);
    auto real = next_fn<Fn>("cudaLaunchKernelExC");
    if (internal_call() || !real) return real ? real(config, function, arguments) : cudaErrorNotSupported;
    const cudaStream_t stream = config == nullptr ? nullptr : config->stream;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        if (config == nullptr) return real(config, function, arguments);
        cudaLaunchConfig_t redirected = *config;
        redirected.stream = reinterpret_cast<cudaStream_t>(native_stream);
        return real(&redirected, function, arguments);
    });
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t bytes,
                                     enum cudaMemcpyKind kind, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    auto real = next_fn<Fn>("cudaMemcpyAsync");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, src, bytes, kind, reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                                       size_t spitch, size_t width, size_t height,
                                       enum cudaMemcpyKind kind, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, size_t, const void *, size_t, size_t, size_t,
                                         enum cudaMemcpyKind, cudaStream_t);
    auto real = next_fn<Fn>("cudaMemcpy2DAsync");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, dpitch, src, spitch, width, height, kind,
                    reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                                         int src_device, size_t bytes, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, int, const void *, int, size_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaMemcpyPeerAsync");
    if (!real) return cudaErrorNotSupported;
    if (pair_mode() && ensure_device_selection() && selection.count == 4 &&
        dst_device != src_device) {
        if (dst_device < 0 || src_device < 0 ||
            dst_device >= static_cast<int>(visible_logical_device_count()) ||
            src_device >= static_cast<int>(visible_logical_device_count()))
            return cudaErrorInvalidDevice;
        int stream_owner = stream_owner_pair(stream);
        const uint32_t stream_pair = stream_owner < 0
            ? current_pair() : static_cast<uint32_t>(stream_owner);
        if (stream_pair >= visible_logical_device_count()) return cudaErrorInvalidResourceHandle;
        ExplicitPairScope destination_scope(static_cast<uint32_t>(dst_device));
        if (destination_scope.status() != cudaSuccess) return destination_scope.status();
        int physical_dst = -1;
        int physical_src = -1;
        cudaError_t mapped = runtime_device_ordinal(dst_device, physical_dst);
        if (mapped != cudaSuccess) return mapped;
        mapped = runtime_device_ordinal(src_device, physical_src);
        if (mapped != cudaSuccess) return mapped;
        using TryCopy = int (*)(uint32_t, uint32_t, uint32_t, void *, const void *,
                                size_t, cudaStream_t, cudaError_t *);
        auto try_copy = reinterpret_cast<TryCopy>(
            lookup_v200("v200_multi_pair_try_copy_peer_async"));
        if (try_copy == nullptr) return cudaErrorNotSupported;
        const cudaError_t prepared = prepare_async_call();
        if (prepared != cudaSuccess) return prepared;
        return dispatch_async(nullptr, [=](void *) {
            const auto started = std::chrono::steady_clock::now();
            cudaError_t status = cudaErrorNotSupported;
            const int handled = try_copy(static_cast<uint32_t>(dst_device),
                                         static_cast<uint32_t>(src_device), stream_pair,
                                         dst, src, bytes,
                                         stream, &status);
            const char *route = "host_stage";
            if (handled == 0) {
                // The hook promises that unhandled ranges have no V200
                // ownership and that it made no changes, so native fallback
                // is safe for this narrow case. CUDA associates the supplied
                // stream with its creating device, which can be the source
                // pair in llama.cpp's peer-copy path. Temporarily select that
                // pair while forwarding the original stream, then restore the
                // destination pair before dispatch_async records its prefix.
                ExplicitPairScope source_scope(stream_pair);
                status = source_scope.status();
                if (status == cudaSuccess) {
                    status = real(dst, physical_dst, src, physical_src, bytes,
                                  stream);
                }
                route = "native_untracked";
            } else if (handled != 1) {
                status = cudaErrorNotSupported;
                route = "unhandled";
            }
            const auto stopped = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                stopped - started).count();
            write_multi_pair_copy_event(static_cast<uint32_t>(dst_device),
                                        static_cast<uint32_t>(src_device), bytes,
                                        status, elapsed_ms, route);
            return status;
        });
    }
    int physical_dst = dst_device;
    int physical_src = src_device;
    if (presentation_enabled()) {
        cudaError_t mapped = runtime_device_ordinal(dst_device, physical_dst);
        if (mapped != cudaSuccess) return mapped;
        mapped = runtime_device_ordinal(src_device, physical_src);
        if (mapped != cudaSuccess) return mapped;
    }
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, physical_dst, src, physical_src, bytes,
                    reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaMemsetAsync(void *device_ptr, int value, size_t bytes,
                                     cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, int, size_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaMemsetAsync");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(device_ptr, value, bytes, reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t *, unsigned int);
    auto real = next_fn<Fn>("cudaStreamCreateWithFlags");
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(stream, flags) : cudaErrorNotSupported;
    if (result == cudaSuccess) return register_created_stream(stream);
    return result;
}

cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t *, unsigned int, int);
    auto real = next_fn<Fn>("cudaStreamCreateWithPriority");
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(stream, flags, priority) : cudaErrorNotSupported;
    if (result == cudaSuccess) return register_created_stream(stream);
    return result;
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t);
    auto real = next_fn<Fn>("cudaStreamDestroy");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    std::unique_lock<std::shared_mutex> outer(tracked_use_gate);
    auto stream_gate = gate_for_stream(stream);
    std::lock_guard<std::mutex> stream_lock(*stream_gate);
    const CUresult waited = wait_stream_prefix(stream);
    if (waited != CUDA_SUCCESS) return cuda_error_for(waited);
    cudaError_t result;
    {
        InternalCallScope bypass;
        result = real(stream);
    }
    if (result == cudaSuccess && execution_attached()) {
        const int unregistered = unregister_execution_stream(stream);
        if (unregistered != V200_EXECUTION_OK) return cudaErrorNotSupported;
    }
    if (result == cudaSuccess) {
        forget_stream_gate(stream);
        forget_stream(stream);
    }
    return result;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t);
    auto real = next_fn<Fn>("cudaStreamSynchronize");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    auto stream_gate = gate_for_stream(stream);
    std::lock_guard<std::mutex> stream_lock(*stream_gate);
    const CUresult waited = wait_stream_prefix(stream);
    if (waited != CUDA_SUCCESS) return cuda_error_for(waited);
    InternalCallScope bypass;
    return real(stream);
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
    using Fn = cudaError_t (CUDARTAPI *)(void);
    auto real = next_fn<Fn>("cudaDeviceSynchronize");
    if (!real) return cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    std::unique_lock<std::shared_mutex> outer(tracked_use_gate);
    const CUresult waited = wait_device_prefix();
    if (waited != CUDA_SUCCESS) return cuda_error_for(waited);
    InternalCallScope bypass;
    return real();
}

cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t *);
    auto real = next_fn<Fn>("cudaEventCreate");
    if (internal_call() || !pair_mode()) return real ? real(event) : cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(event) : cudaErrorNotSupported;
    if (result == cudaSuccess && event != nullptr) remember_event_owner(*event, current_pair());
    return result;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t *, unsigned int);
    auto real = next_fn<Fn>("cudaEventCreateWithFlags");
    if (internal_call() || !pair_mode()) return real ? real(event, flags) : cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(event, flags) : cudaErrorNotSupported;
    if (result == cudaSuccess && event != nullptr) remember_event_owner(*event, current_pair());
    return result;
}

cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t);
    auto real = next_fn<Fn>("cudaEventDestroy");
    if (internal_call() || !pair_mode()) return real ? real(event) : cudaErrorNotSupported;
    EventPairScope pair_scope(event);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(event) : cudaErrorNotSupported;
    if (result == cudaSuccess) forget_event_owner(event);
    return result;
}

cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t);
    auto real = next_fn<Fn>("cudaEventQuery");
    if (internal_call() || !pair_mode()) return real ? real(event) : cudaErrorNotSupported;
    EventPairScope pair_scope(event);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    return real ? real(event) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaEventRecord");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    const cudaError_t result = dispatch_async(stream, [=](void *native_stream) {
        return real(event, reinterpret_cast<cudaStream_t>(native_stream));
    });
    if (result == cudaSuccess && pair_mode()) remember_event_owner(event, current_pair());
    return result;
}

cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, cudaEvent_t, unsigned int);
    auto real = next_fn<Fn>("cudaStreamWaitEvent");
    if (!real) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(reinterpret_cast<cudaStream_t>(native_stream), event, flags);
    });
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t);
    auto real = next_fn<Fn>("cudaEventSynchronize");
    if (internal_call()) return real ? real(event) : cudaErrorNotSupported;
    EventPairScope pair_scope(event);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    InternalCallScope bypass;
    return real ? real(event) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaGraphLaunch");
    if (internal_call()) return real ? real(graph, stream) : cudaErrorNotSupported;
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    if (!real) return cudaErrorNotSupported;
    const cudaError_t result = dispatch_async(stream, [=](void *native_stream) {
        return real(graph, reinterpret_cast<cudaStream_t>(native_stream));
    });
    if (result == cudaSuccess && pair_mode() && graph_capture_enabled()) {
        // A bounded sample proves that the executable was replayed without
        // turning every inference launch into synchronous profile-file I/O.
        if (graph_replay_log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
            GraphRewriteInfo info{};
            {
                std::lock_guard<std::mutex> lock(graph_info_mutex);
                const auto found = graph_exec_info.find(graph);
                if (found != graph_exec_info.end()) info = found->second;
            }
            write_graph_profile_snapshot("graph_replay", reinterpret_cast<uintptr_t>(graph), info);
        }
    }
    return result;
}

cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, enum cudaStreamCaptureMode);
    auto real = next_fn<Fn>("cudaStreamBeginCapture");
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    if (pair_mode()) {
        const cudaError_t prepared = prepare_async_call();
        if (prepared != cudaSuccess) return prepared;
    }
    InternalCallScope bypass;
    return real ? real(stream, mode) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *graph) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, cudaGraph_t *);
    auto real = next_fn<Fn>("cudaStreamEndCapture");
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    if (graph != nullptr) *graph = nullptr;
    cudaError_t result;
    {
        InternalCallScope bypass;
        result = real(stream, graph);
    }
    if (result != cudaSuccess || !pair_mode() || !graph_capture_enabled() ||
        graph == nullptr || *graph == nullptr) return result;

    GraphRewriteInfo info{};
    const char *pair_compute = std::getenv("V200_PAIR_COMPUTE");
    if (pair_compute != nullptr && std::strcmp(pair_compute, "1") == 0) {
        using RewriteGraph = decltype(&v200_operator_rewrite_captured_graph);
        auto rewrite = reinterpret_cast<RewriteGraph>(
            lookup_v200("v200_operator_rewrite_captured_graph"));
        if (rewrite == nullptr) {
            result = cudaErrorNotSupported;
        } else {
            InternalCallScope bypass;
            result = rewrite(*graph, &info.matched_nodes, &info.rewritten_nodes);
        }
        info.status = result;
        if (result != cudaSuccess) {
            // A rewrite may have updated earlier nodes before a later node
            // fails. Never expose such a partially rewritten graph to llama.
            using Destroy = cudaError_t (CUDARTAPI *)(cudaGraph_t);
            auto destroy = next_fn<Destroy>("cudaGraphDestroy");
            if (destroy != nullptr) {
                InternalCallScope bypass;
                (void)destroy(*graph);
            }
            *graph = nullptr;
            write_graph_profile_snapshot("graph_capture_rewrite", 0, info);
            return result;
        }
    }
    {
        std::lock_guard<std::mutex> lock(graph_info_mutex);
        captured_graph_info[*graph] = info;
    }
    write_graph_profile_snapshot("graph_capture_rewrite",
                                 reinterpret_cast<uintptr_t>(*graph), info);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                            enum cudaStreamCaptureStatus *status) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, enum cudaStreamCaptureStatus *);
    auto real = next_fn<Fn>("cudaStreamIsCapturing");
    if (internal_call()) return real ? real(stream, status) : cudaErrorNotSupported;
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return pair_scope.status();
    InternalCallScope bypass;
    return real ? real(stream, status) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *graph_exec, cudaGraph_t graph,
                                           unsigned long long flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
    auto real = next_fn<Fn>("cudaGraphInstantiate");
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    cudaError_t result;
    {
        InternalCallScope bypass;
        result = real(graph_exec, graph, flags);
    }
    if (result == cudaSuccess && pair_mode() && graph_capture_enabled() &&
        graph_exec != nullptr && *graph_exec != nullptr) {
        GraphRewriteInfo info{};
        {
            std::lock_guard<std::mutex> lock(graph_info_mutex);
            const auto found = captured_graph_info.find(graph);
            if (found != captured_graph_info.end()) info = found->second;
            graph_exec_info[*graph_exec] = info;
        }
        write_graph_profile_snapshot("graph_instantiate",
                                     reinterpret_cast<uintptr_t>(*graph_exec), info);
    }
    return result;
}

cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t graph_exec, cudaGraph_t graph,
                                          cudaGraphExecUpdateResultInfo *result_info) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo *);
    auto real = next_fn<Fn>("cudaGraphExecUpdate");
    if (pair_mode() && !graph_capture_enabled()) return cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    cudaError_t result;
    {
        InternalCallScope bypass;
        result = real(graph_exec, graph, result_info);
    }
    const bool update_accepted = result == cudaSuccess &&
        (result_info == nullptr || result_info->result == cudaGraphExecUpdateSuccess);
    if (update_accepted && pair_mode() && graph_capture_enabled()) {
        GraphRewriteInfo info{};
        {
            std::lock_guard<std::mutex> lock(graph_info_mutex);
            const auto found = captured_graph_info.find(graph);
            if (found != captured_graph_info.end()) info = found->second;
            graph_exec_info[graph_exec] = info;
        }
        write_graph_profile_snapshot("graph_exec_update",
                                     reinterpret_cast<uintptr_t>(graph_exec), info);
    }
    return result;
}

cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraph_t);
    auto real = next_fn<Fn>("cudaGraphDestroy");
    if (internal_call()) return real ? real(graph) : cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    InternalCallScope bypass;
    const cudaError_t result = real(graph);
    if (result == cudaSuccess && pair_mode() && graph_capture_enabled()) {
        std::lock_guard<std::mutex> lock(graph_info_mutex);
        captured_graph_info.erase(graph);
    }
    return result;
}

cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graph_exec) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t);
    auto real = next_fn<Fn>("cudaGraphExecDestroy");
    if (internal_call()) return real ? real(graph_exec) : cudaErrorNotSupported;
    if (real == nullptr) return cudaErrorNotSupported;
    InternalCallScope bypass;
    const cudaError_t result = real(graph_exec);
    if (result == cudaSuccess && pair_mode() && graph_capture_enabled()) {
        std::lock_guard<std::mutex> lock(graph_info_mutex);
        graph_exec_info.erase(graph_exec);
    }
    return result;
}

} // extern "C"

extern "C" {

/* Driver device presentation uses native CUdevice/CUcontext handles.  The
 * logical capacity is pair-backed while compute properties stay anchor-local. */
CUresult CUDAAPI cuDeviceGetCount(int *count) {
    using Fn = CUresult (CUDAAPI *)(int *);
    auto real = next_fn<Fn>("cuDeviceGetCount");
    if (internal_call() || !presentation_enabled()) return real ? real(count) : CUDA_ERROR_NOT_SUPPORTED;
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!ensure_device_selection()) return selection.driver_error;
    *count = static_cast<int>(visible_logical_device_count());
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    using Fn = CUresult (CUDAAPI *)(CUdevice *, int);
    auto real = next_fn<Fn>("cuDeviceGet");
    if (internal_call() || !presentation_enabled()) return real ? real(device, ordinal) : CUDA_ERROR_NOT_SUPPORTED;
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!ensure_device_selection()) return selection.driver_error;
    int native_ordinal = -1;
    CUdevice physical = -1;
    if (!logical_device(ordinal, native_ordinal, physical))
        return CUDA_ERROR_INVALID_DEVICE;
    *device = physical;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate_v2(CUcontext *context, unsigned int flags, CUdevice device) {
    using Fn = CUresult (CUDAAPI *)(CUcontext *, unsigned int, CUdevice);
    auto real = next_fn<Fn>("cuCtxCreate_v2");
    if (internal_call() || !presentation_enabled())
        return real ? real(context, flags, device) : CUDA_ERROR_NOT_SUPPORTED;
    if (real == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    const int pair = pair_index_for_native_device(device);
    if (pair < 0) return CUDA_ERROR_INVALID_DEVICE;
    const uint32_t previous_pair = current_pair();
    if (!save_context_pair_for_restore(previous_pair)) return CUDA_ERROR_OUT_OF_MEMORY;
    if (select_pair(static_cast<uint32_t>(pair)) != 0) {
        discard_context_pair_restore();
        return CUDA_ERROR_INVALID_DEVICE;
    }
    InternalCallScope bypass;
    const CUresult result = real(context, flags, device);
    if (result != CUDA_SUCCESS) {
        discard_context_pair_restore();
        (void)select_pair(previous_pair);
    } else {
        bind_context_pair_restore(context == nullptr ? nullptr : *context);
    }
    return result;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
    using Fn = CUresult (CUDAAPI *)(CUcontext);
    using GetCurrent = CUresult (CUDAAPI *)(CUcontext *);
    auto real = next_fn<Fn>("cuCtxSetCurrent");
    auto get_current = next_fn<GetCurrent>("cuCtxGetCurrent");
    if (internal_call() || !presentation_enabled())
        return real ? real(context) : CUDA_ERROR_NOT_SUPPORTED;
    if (!real || !get_current) return CUDA_ERROR_NOT_SUPPORTED;
    CUcontext previous = nullptr;
    CUresult result;
    {
        InternalCallScope bypass;
        result = get_current(&previous);
    }
    if (result != CUDA_SUCCESS) return result;
    {
        InternalCallScope bypass;
        result = real(context);
    }
    if (result != CUDA_SUCCESS || context == nullptr) return result;
    uint32_t pair = 0;
    result = pair_index_for_context(context, &pair);
    if (result == CUDA_SUCCESS && select_pair(pair) == 0) return CUDA_SUCCESS;
    {
        InternalCallScope bypass;
        (void)real(previous);
    }
    return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_DEVICE : result;
}

CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext context) {
    using Fn = CUresult (CUDAAPI *)(CUcontext);
    using Pop = CUresult (CUDAAPI *)(CUcontext *);
    auto real = next_fn<Fn>("cuCtxPushCurrent_v2");
    if (real == nullptr) real = next_fn<Fn>("cuCtxPushCurrent");
    auto pop = next_fn<Pop>("cuCtxPopCurrent_v2");
    if (pop == nullptr) pop = next_fn<Pop>("cuCtxPopCurrent");
    if (internal_call() || !presentation_enabled())
        return real ? real(context) : CUDA_ERROR_NOT_SUPPORTED;
    if (context == nullptr || real == nullptr || pop == nullptr)
        return context == nullptr ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_NOT_SUPPORTED;
    const uint32_t previous_pair = current_pair();
    if (!save_context_pair_for_restore(previous_pair)) return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult result;
    {
        InternalCallScope bypass;
        result = real(context);
    }
    if (result != CUDA_SUCCESS) {
        discard_context_pair_restore();
        return result;
    }
    bind_context_pair_restore(context);
    uint32_t pair = 0;
    result = pair_index_for_context(context, &pair);
    if (result == CUDA_SUCCESS && select_pair(pair) == 0) return CUDA_SUCCESS;
    CUcontext popped = nullptr;
    {
        InternalCallScope bypass;
        (void)pop(&popped);
    }
    discard_context_pair_restore();
    (void)select_pair(previous_pair);
    return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_DEVICE : result;
}

CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext *context) {
    using Fn = CUresult (CUDAAPI *)(CUcontext *);
    using GetCurrent = CUresult (CUDAAPI *)(CUcontext *);
    auto real = next_fn<Fn>("cuCtxPopCurrent_v2");
    if (real == nullptr) real = next_fn<Fn>("cuCtxPopCurrent");
    auto get_current = next_fn<GetCurrent>("cuCtxGetCurrent");
    if (internal_call() || !presentation_enabled())
        return real ? real(context) : CUDA_ERROR_NOT_SUPPORTED;
    if (real == nullptr || get_current == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult result;
    {
        InternalCallScope bypass;
        result = real(context);
    }
    if (result != CUDA_SUCCESS) return result;
    const CUcontext popped_context = context == nullptr ? nullptr : *context;
    uint32_t saved_pair = current_pair();
    const bool has_saved_pair = take_context_pair_restore(popped_context, &saved_pair);
    CUcontext active = nullptr;
    {
        InternalCallScope bypass;
        result = get_current(&active);
    }
    if (result != CUDA_SUCCESS) {
        if (has_saved_pair) (void)select_pair(saved_pair);
        return result;
    }
    if (active == nullptr) {
        // A matching Create/Push wrapper saved the previously selected pair.
        // If no wrapper observed this stack entry, preserve cudaSetDevice's
        // thread-local default rather than guessing pair 0.
        return !has_saved_pair || select_pair(saved_pair) == 0
            ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
    }
    uint32_t pair = 0;
    result = pair_index_for_context(active, &pair);
    if (result != CUDA_SUCCESS) {
        if (has_saved_pair) (void)select_pair(saved_pair);
        return result;
    }
    return select_pair(pair) == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *can_access, CUdevice device,
                                       CUdevice peer_device) {
    using Fn = CUresult (CUDAAPI *)(int *, CUdevice, CUdevice);
    auto real = next_fn<Fn>("cuDeviceCanAccessPeer");
    if (internal_call() || !presentation_enabled())
        return real ? real(can_access, device, peer_device) : CUDA_ERROR_NOT_SUPPORTED;
    if (can_access == nullptr) return CUDA_ERROR_INVALID_VALUE;
    const int device_pair = pair_index_for_native_device(device);
    const int peer_pair = pair_index_for_native_device(peer_device);
    if (device_pair < 0 || peer_pair < 0) return CUDA_ERROR_INVALID_DEVICE;
    if (pair_mode() && selection.count == 4 && device_pair != peer_pair) {
        *can_access = 0;
        return CUDA_SUCCESS;
    }
    InternalCallScope bypass;
    return real ? real(can_access, device, peer_device) : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peer_context, unsigned int flags) {
    using Fn = CUresult (CUDAAPI *)(CUcontext, unsigned int);
    auto real = next_fn<Fn>("cuCtxEnablePeerAccess");
    if (internal_call() || !presentation_enabled())
        return real ? real(peer_context, flags) : CUDA_ERROR_NOT_SUPPORTED;
    if (pair_mode() && selection.count == 4) return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    InternalCallScope bypass;
    return real ? real(peer_context, flags) : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice device) {
    using Fn = CUresult (CUDAAPI *)(size_t *, CUdevice);
    auto real = next_fn<Fn>("cuDeviceTotalMem_v2");
    if (internal_call() || !presentation_enabled())
        return real ? real(bytes, device) : CUDA_ERROR_NOT_SUPPORTED;
    if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!pair_mode()) {
        InternalCallScope bypass;
        return real ? real(bytes, device) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!ensure_device_selection()) return selection.driver_error;
    int pair_index = -1;
    const int visible_pairs = selection.count / 2;
    for (int pair = 0; pair < visible_pairs; ++pair) {
        if (device == selection.device[pair * 2]) {
            pair_index = pair;
            break;
        }
    }
    if (pair_index < 0) return CUDA_ERROR_INVALID_DEVICE;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const CUresult capacity = pair_memory_capacity(static_cast<uint32_t>(pair_index),
                                                   &free_bytes, &total_bytes);
    (void)free_bytes;
    if (capacity != CUDA_SUCCESS) return capacity;
    *bytes = total_bytes;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) {
    using Fn = CUresult (CUDAAPI *)(size_t *, size_t *);
    auto real = next_fn<Fn>("cuMemGetInfo_v2");
    if (internal_call() || !presentation_enabled())
        return real ? real(free_bytes, total_bytes) : CUDA_ERROR_NOT_SUPPORTED;
    if (!pair_mode()) {
        InternalCallScope bypass;
        return real ? real(free_bytes, total_bytes) : CUDA_ERROR_NOT_SUPPORTED;
    }
    return pair_memory_capacity(current_pair(), free_bytes, total_bytes);
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyHtoDAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, src, bytes, reinterpret_cast<CUstream>(native_stream));
    });
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dst, const void *src, size_t bytes,
                                            CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyHtoDAsync_v2_ptsz");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(tracked, [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(void *, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoHAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, src, bytes, reinterpret_cast<CUstream>(native_stream));
    });
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void *dst, CUdeviceptr src, size_t bytes,
                                            CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(void *, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoHAsync_v2_ptsz");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(tracked, [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoDAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(dst, src, bytes, reinterpret_cast<CUstream>(native_stream));
    });
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t bytes,
                                            CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoDAsync_v2_ptsz");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(tracked, [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

#define V200_DEFINE_MEMSET_ASYNC_1D(name, value_type)                                      \
    CUresult CUDAAPI name(CUdeviceptr dst, value_type value, size_t count, CUstream stream) { \
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, value_type, size_t, CUstream);         \
        auto real = next_fn<Fn>(#name);                                                     \
        if (internal_call() || !real)                                                       \
            return real ? real(dst, value, count, stream) : CUDA_ERROR_NOT_SUPPORTED;       \
        const CUresult profile = require_pair_profile();                                    \
        if (profile != CUDA_SUCCESS) return profile;                                        \
        return dispatch_async(stream, [=](void *native_stream) {                            \
            return real(dst, value, count, reinterpret_cast<CUstream>(native_stream));       \
        });                                                                                 \
    }                                                                                       \
    CUresult CUDAAPI name##_ptsz(CUdeviceptr dst, value_type value, size_t count,            \
                                 CUstream stream) {                                         \
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, value_type, size_t, CUstream);         \
        auto real = next_fn<Fn>(#name "_ptsz");                                             \
        if (internal_call() || !real)                                                       \
            return real ? real(dst, value, count, stream) : CUDA_ERROR_NOT_SUPPORTED;       \
        const CUresult profile = require_pair_profile();                                    \
        if (profile != CUDA_SUCCESS) return profile;                                        \
        return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {             \
            return real(dst, value, count, stream);                                          \
        });                                                                                 \
    }

#define V200_DEFINE_MEMSET_ASYNC_2D(name, value_type)                                        \
    CUresult CUDAAPI name(CUdeviceptr dst, size_t pitch, value_type value, size_t width,      \
                          size_t height, CUstream stream) {                                  \
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t, value_type, size_t, size_t,      \
                                        CUstream);                                            \
        auto real = next_fn<Fn>(#name);                                                       \
        if (internal_call() || !real)                                                        \
            return real ? real(dst, pitch, value, width, height, stream)                      \
                        : CUDA_ERROR_NOT_SUPPORTED;                                          \
        const CUresult profile = require_pair_profile();                                     \
        if (profile != CUDA_SUCCESS) return profile;                                         \
        return dispatch_async(stream, [=](void *native_stream) {                             \
            return real(dst, pitch, value, width, height,                                     \
                        reinterpret_cast<CUstream>(native_stream));                            \
        });                                                                                  \
    }                                                                                        \
    CUresult CUDAAPI name##_ptsz(CUdeviceptr dst, size_t pitch, value_type value,             \
                                 size_t width, size_t height, CUstream stream) {              \
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t, value_type, size_t, size_t,      \
                                        CUstream);                                            \
        auto real = next_fn<Fn>(#name "_ptsz");                                              \
        if (internal_call() || !real)                                                        \
            return real ? real(dst, pitch, value, width, height, stream)                     \
                        : CUDA_ERROR_NOT_SUPPORTED;                                          \
        const CUresult profile = require_pair_profile();                                     \
        if (profile != CUDA_SUCCESS) return profile;                                         \
        return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {              \
            return real(dst, pitch, value, width, height, stream);                            \
        });                                                                                  \
    }

V200_DEFINE_MEMSET_ASYNC_1D(cuMemsetD8Async, unsigned char)
V200_DEFINE_MEMSET_ASYNC_1D(cuMemsetD16Async, unsigned short)
V200_DEFINE_MEMSET_ASYNC_1D(cuMemsetD32Async, unsigned int)
V200_DEFINE_MEMSET_ASYNC_2D(cuMemsetD2D8Async, unsigned char)
V200_DEFINE_MEMSET_ASYNC_2D(cuMemsetD2D16Async, unsigned short)
V200_DEFINE_MEMSET_ASYNC_2D(cuMemsetD2D32Async, unsigned int)

#undef V200_DEFINE_MEMSET_ASYNC_1D
#undef V200_DEFINE_MEMSET_ASYNC_2D

CUresult CUDAAPI cuLaunchKernel(CUfunction function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_bytes, CUstream stream, void **kernel_params, void **extra) {
    using Fn = CUresult (CUDAAPI *)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
    auto real = next_fn<Fn>("cuLaunchKernel");
    if (internal_call() || !real) return real ? real(function, grid_x, grid_y, grid_z,
        block_x, block_y, block_z, shared_bytes, stream, kernel_params, extra)
        : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                    shared_bytes, reinterpret_cast<CUstream>(native_stream), kernel_params, extra);
    });
}

CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_bytes, CUstream stream, void **kernel_params, void **extra) {
    using Fn = CUresult (CUDAAPI *)(CUfunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
    auto real = next_fn<Fn>("cuLaunchKernel_ptsz");
    if (internal_call() || !real) return real ? real(function, grid_x, grid_y, grid_z,
        block_x, block_y, block_z, shared_bytes, stream, kernel_params, extra)
        : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(tracked, [=](void *) {
        return real(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                    shared_bytes, stream, kernel_params, extra);
    });
}

CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUstream);
    auto real = next_fn<Fn>("cuStreamSynchronize");
    if (internal_call() || !real) return real ? real(stream) : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    auto gate = gate_for_stream(stream);
    std::lock_guard<std::mutex> stream_lock(*gate);
    const CUresult waited = wait_stream_prefix(stream);
    if (waited != CUDA_SUCCESS) return waited;
    InternalCallScope bypass;
    return real(stream);
}

CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUstream);
    auto real = next_fn<Fn>("cuStreamSynchronize_ptsz");
    if (internal_call() || !real) return real ? real(stream) : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    const CUstream tracked_stream = tracked;
    auto gate = gate_for_stream(tracked_stream);
    std::lock_guard<std::mutex> stream_lock(*gate);
    const CUresult waited = wait_stream_prefix(tracked_stream);
    if (waited != CUDA_SUCCESS) return waited;
    InternalCallScope bypass;
    return real(stream);
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    using Fn = CUresult (CUDAAPI *)(CUstream, CUevent, unsigned int);
    auto real = next_fn<Fn>("cuStreamWaitEvent");
    if (internal_call() || !real) return real ? real(stream, event, flags) : CUDA_ERROR_NOT_SUPPORTED;
    StreamPairScope pair_scope(stream);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(reinterpret_cast<CUstream>(native_stream), event, flags);
    });
}

CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream stream, CUevent event, unsigned int flags) {
    using Fn = CUresult (CUDAAPI *)(CUstream, CUevent, unsigned int);
    auto real = next_fn<Fn>("cuStreamWaitEvent_ptsz");
    if (internal_call() || !real) return real ? real(stream, event, flags) : CUDA_ERROR_NOT_SUPPORTED;
    const CUstream tracked = per_thread_tracking_stream(stream);
    StreamPairScope pair_scope(tracked);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(tracked, [=](void *) {
        return real(stream, event, flags);
    });
}

} // extern "C"

namespace {

extern "C" CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                                 CUdeviceptr requested, unsigned long long flags) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
        auto real = next_fn<Fn>("cuMemAddressReserve");
        return real ? real(ptr, size, alignment, requested, flags) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmAddressReserve>("v200_memory_vmm_address_reserve");
    InternalCallScope bypass;
    return fn ? fn(ptr, size, alignment, requested, flags) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t);
        auto real = next_fn<Fn>("cuMemAddressFree");
        return real ? real(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
    }
    const int owner = memory_pair_for_range(reinterpret_cast<const void *>(ptr), size);
    const uint32_t target_pair = owner < 0 ? current_pair() : static_cast<uint32_t>(owner);
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmAddressFree>("v200_memory_vmm_address_free");
    std::unique_lock<std::shared_mutex> use_lock(tracked_use_gate);
    if (owner >= 0) {
        const CUresult waited = wait_device_prefix();
        if (waited != CUDA_SUCCESS) return waited;
        const CUresult drained = synchronize_native_anchor();
        if (drained != CUDA_SUCCESS) return drained;
    }
    InternalCallScope bypass;
    return fn ? fn(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                         const CUmemAllocationProp *prop, unsigned long long flags) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
        auto real = next_fn<Fn>("cuMemCreate");
        if (!real) return CUDA_ERROR_NOT_SUPPORTED;
        if (presentation_enabled() && prop != nullptr && prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
            if (!ensure_device_selection()) return selection.driver_error;
            if (static_cast<CUdevice>(prop->location.id) != selection.device[0]) return CUDA_ERROR_INVALID_DEVICE;
        }
        InternalCallScope bypass;
        return real(handle, size, prop, flags);
    }
    CUmemAllocationProp native_prop{};
    const CUmemAllocationProp *effective_prop = prop;
    uint32_t target_pair = current_pair();
    if (prop != nullptr && prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        v200::frontdoor::detail::NativePairLocation mapped;
        if (!map_external_vmm_device(static_cast<CUdevice>(prop->location.id), &mapped))
            return CUDA_ERROR_INVALID_DEVICE;
        target_pair = static_cast<uint32_t>(mapped.pair);
        native_prop = *prop;
        native_prop.location.id = mapped.device;
        effective_prop = &native_prop;
    }
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmCreate>("v200_memory_vmm_create");
    InternalCallScope bypass;
    return fn ? fn(handle, size, effective_prop, flags) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *granularity,
                                                           const CUmemAllocationProp *prop,
                                                           CUmemAllocationGranularity_flags option) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
        auto real = next_fn<Fn>("cuMemGetAllocationGranularity");
        if (!real) return CUDA_ERROR_NOT_SUPPORTED;
        if (!presentation_enabled() || prop == nullptr || prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
            return real(granularity, prop, option);
        }
        InternalCallScope bypass;
        return real(granularity, prop, option);
    }
    CUmemAllocationProp native_prop{};
    const CUmemAllocationProp *effective_prop = prop;
    uint32_t target_pair = current_pair();
    if (prop != nullptr && prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        v200::frontdoor::detail::NativePairLocation mapped;
        if (!map_external_vmm_device(static_cast<CUdevice>(prop->location.id), &mapped))
            return CUDA_ERROR_INVALID_DEVICE;
        target_pair = static_cast<uint32_t>(mapped.pair);
        native_prop = *prop;
        native_prop.location.id = mapped.device;
        effective_prop = &native_prop;
    }
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmGranularity>("v200_memory_vmm_get_granularity");
    InternalCallScope bypass;
    return fn ? fn(granularity, effective_prop, option) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                                     CUmemGenericAllocationHandle handle, unsigned long long flags) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
        auto real = next_fn<Fn>("cuMemMap");
        return real ? real(ptr, size, offset, handle, flags) : CUDA_ERROR_NOT_SUPPORTED;
    }
    const int owner = memory_pair_for_range(reinterpret_cast<const void *>(ptr), size);
    const uint32_t target_pair = owner < 0 ? current_pair() : static_cast<uint32_t>(owner);
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmMap>("v200_memory_vmm_map");
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;
    {
        InternalCallScope bypass;
        result = fn ? fn(ptr, size, offset, handle, flags) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (result == CUDA_SUCCESS) v200_frontdoor_snapshot("cu_mem_map");
    return result;
}

extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                           const CUmemAccessDesc *desc, size_t count) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
        auto real = next_fn<Fn>("cuMemSetAccess");
        if (!real) return CUDA_ERROR_NOT_SUPPORTED;
        if (!presentation_enabled() || desc == nullptr) return real(ptr, size, desc, count);
        if (!ensure_device_selection()) return selection.driver_error;
        for (size_t i = 0; i < count; ++i) {
            if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE) return CUDA_ERROR_NOT_SUPPORTED;
            if (static_cast<CUdevice>(desc[i].location.id) != selection.device[0]) return CUDA_ERROR_INVALID_DEVICE;
        }
        InternalCallScope bypass;
        return real(ptr, size, desc, count);
    }
    const int owner = memory_pair_for_range(reinterpret_cast<const void *>(ptr), size);
    const uint32_t target_pair = owner < 0 ? current_pair() : static_cast<uint32_t>(owner);
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    if (desc == nullptr || count == 0 || count > 2) return CUDA_ERROR_INVALID_VALUE;
    std::array<CUmemAccessDesc, 2> native_desc{};
    for (size_t i = 0; i < count; ++i) {
        if (desc[i].location.type != CU_MEM_LOCATION_TYPE_DEVICE)
            return CUDA_ERROR_NOT_SUPPORTED;
        v200::frontdoor::detail::NativePairLocation mapped;
        if (!map_external_vmm_device(
                static_cast<CUdevice>(desc[i].location.id), &mapped) ||
            mapped.pair != static_cast<int>(target_pair))
            return CUDA_ERROR_INVALID_DEVICE;
        native_desc[i] = desc[i];
        native_desc[i].location.id = mapped.device;
    }
    auto fn = memory_fn<MemVmmSetAccess>("v200_memory_vmm_set_access");
    InternalCallScope bypass;
    return fn ? fn(ptr, size, native_desc.data(), count) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t);
        auto real = next_fn<Fn>("cuMemUnmap");
        return real ? real(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
    }
    const int owner = memory_pair_for_range(reinterpret_cast<const void *>(ptr), size);
    const uint32_t target_pair = owner < 0 ? current_pair() : static_cast<uint32_t>(owner);
    ExplicitPairScope pair_scope(target_pair);
    if (pair_scope.status() != cudaSuccess) return driver_error_for(pair_scope.status());
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmUnmap>("v200_memory_vmm_unmap");
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;
    {
        std::unique_lock<std::shared_mutex> use_lock(tracked_use_gate);
        if (owner >= 0) {
            result = wait_device_prefix();
            if (result == CUDA_SUCCESS) result = synchronize_native_anchor();
        }
        if (result == CUDA_SUCCESS || owner < 0) {
            InternalCallScope bypass;
            result = fn ? fn(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    if (result == CUDA_SUCCESS) v200_frontdoor_snapshot("cu_mem_unmap");
    return result;
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUmemGenericAllocationHandle);
        auto real = next_fn<Fn>("cuMemRelease");
        return real ? real(handle) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto owns = memory_fn<MemOwnsHandle>("v200_memory_owns_handle");
    if (owns == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    if (owns(handle) == 0) {
        using Fn = CUresult (CUDAAPI *)(CUmemGenericAllocationHandle);
        auto real = next_fn<Fn>("cuMemRelease");
        if (!real) return CUDA_ERROR_NOT_SUPPORTED;
        InternalCallScope bypass;
        return real(handle);
    }
    auto fn = memory_fn<MemVmmRelease>("v200_memory_vmm_release");
    std::unique_lock<std::shared_mutex> use_lock(tracked_use_gate);
    InternalCallScope bypass;
    return fn ? fn(handle) : CUDA_ERROR_NOT_SUPPORTED;
}

std::string json_string(const char *text, size_t limit) {
    std::string escaped;
    escaped.push_back('"');
    if (text != nullptr) {
        for (size_t i = 0; i < limit && text[i] != '\0'; ++i) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == '"' || c == '\\') {
                escaped.push_back('\\');
                escaped.push_back(static_cast<char>(c));
            } else if (c < 0x20) {
                char encoded[7];
                std::snprintf(encoded, sizeof(encoded), "\\u%04x", c);
                escaped.append(encoded);
            } else {
                escaped.push_back(static_cast<char>(c));
            }
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::mutex profile_log_mutex;
std::once_flag exit_snapshot_registration[2];

void write_multi_pair_copy_event(uint32_t dst_pair, uint32_t src_pair,
                                 size_t bytes, cudaError_t status,
                                 double elapsed_ms, const char *route) {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    char line[512];
    const std::string route_json = json_string(route == nullptr ? "unknown" : route, 64);
    const int count = std::snprintf(line, sizeof(line),
        "{\"schema\":\"v200-multi-pair-copy/v1\",\"event\":\"cross_pair_copy\","
        "\"dst_pair\":%u,\"src_pair\":%u,\"bytes\":%llu,\"status\":%d,"
        "\"route\":%s,\"blocking\":true,\"elapsed_ms\":%.6f}\n",
        dst_pair, src_pair, static_cast<unsigned long long>(bytes),
        static_cast<int>(status), route_json.c_str(), elapsed_ms);
    if (count <= 0 || static_cast<size_t>(count) >= sizeof(line)) return;
    std::lock_guard<std::mutex> lock(profile_log_mutex);
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

void write_exit_pair_snapshot(uint32_t pair) {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    using GetProfile = CUresult (*)(uint32_t, v200_memory_pair_profile *);
    using GetUsage = CUresult (*)(uint32_t, v200_memory_usage_info *);
    auto get_profile = memory_fn<GetProfile>("v200_memory_get_profile_for_pair");
    auto get_usage = memory_fn<GetUsage>("v200_memory_get_usage_for_pair");
    if (get_profile == nullptr || get_usage == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(profile_mutex);
        if (pair >= 2 || profile_state[pair] != 2) return;
    }
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    if (get_profile(pair, &profile) != CUDA_SUCCESS ||
        get_usage(pair, &usage) != CUDA_SUCCESS) return;
    const std::string anchor = json_string(profile.owner_uuid[0], sizeof(profile.owner_uuid[0]));
    const std::string peer = json_string(profile.owner_uuid[1], sizeof(profile.owner_uuid[1]));
    const std::string qualification = json_string(profile.qualification_id,
                                                   sizeof(profile.qualification_id));
    char line[2048];
    const int count = std::snprintf(line, sizeof(line),
        "{\"schema\":\"v200-frontdoor-ledger/v1\",\"event\":\"process_exit_final\","
        "\"mode\":\"%s\",\"pair_index\":%u,\"anchor_uuid\":%s,\"peer_uuid\":%s,"
        "\"qualification_id\":%s,\"owner_budget_bytes\":[%llu,%llu],"
        "\"owner_used_bytes\":[%llu,%llu],\"live_allocation_count\":%llu,"
        "\"live_reservation_count\":%llu,\"cleanup_pending_count\":%llu}\n",
        mode_name(current_mode()), pair, anchor.c_str(), peer.c_str(), qualification.c_str(),
        static_cast<unsigned long long>(usage.owner_budget_bytes[0]),
        static_cast<unsigned long long>(usage.owner_budget_bytes[1]),
        static_cast<unsigned long long>(usage.owner_used_bytes[0]),
        static_cast<unsigned long long>(usage.owner_used_bytes[1]),
        static_cast<unsigned long long>(usage.live_allocation_count),
        static_cast<unsigned long long>(usage.live_reservation_count),
        static_cast<unsigned long long>(usage.cleanup_pending_count));
    if (count <= 0 || static_cast<size_t>(count) >= sizeof(line)) return;
    std::lock_guard<std::mutex> lock(profile_log_mutex);
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

void write_exit_pair0_snapshot() { write_exit_pair_snapshot(0); }
void write_exit_pair1_snapshot() { write_exit_pair_snapshot(1); }

void register_process_exit_snapshot(uint32_t pair) {
    if (pair >= 2) return;
    // Prime both symbol lookup caches and the manager's pair query path before
    // registering the callback; function-local static destructors therefore
    // remain ordered after this final query.
    using GetProfile = CUresult (*)(uint32_t, v200_memory_pair_profile *);
    using GetUsage = CUresult (*)(uint32_t, v200_memory_usage_info *);
    auto get_profile = memory_fn<GetProfile>("v200_memory_get_profile_for_pair");
    auto get_usage = memory_fn<GetUsage>("v200_memory_get_usage_for_pair");
    if (get_profile == nullptr || get_usage == nullptr) return;
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    if (get_profile(pair, &profile) != CUDA_SUCCESS ||
        get_usage(pair, &usage) != CUDA_SUCCESS) return;
    std::call_once(exit_snapshot_registration[pair], [pair] {
        (void)std::atexit(pair == 0 ? write_exit_pair0_snapshot
                                    : write_exit_pair1_snapshot);
    });
}

void write_profile_snapshot(const char *event_name) {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    auto get_profile = memory_fn<CUresult (*)(v200_memory_pair_profile *)>("v200_memory_get_profile");
    auto get_usage = memory_fn<CUresult (*)(v200_memory_usage_info *)>("v200_memory_get_usage");
    if (get_profile == nullptr || get_usage == nullptr) return;
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    if (get_profile(&profile) != CUDA_SUCCESS || get_usage(&usage) != CUDA_SUCCESS) return;

    const std::string event = json_string(event_name == nullptr ? "snapshot" : event_name, 128);
    const std::string anchor = json_string(profile.owner_uuid[0], sizeof(profile.owner_uuid[0]));
    const std::string peer = json_string(profile.owner_uuid[1], sizeof(profile.owner_uuid[1]));
    const std::string qualification = json_string(profile.qualification_id, sizeof(profile.qualification_id));
    char line[2048];
    const int count = std::snprintf(line, sizeof(line),
        "{\"schema\":\"v200-frontdoor-ledger/v1\",\"event\":%s,\"mode\":\"%s\","
        "\"anchor_uuid\":%s,\"peer_uuid\":%s,\"qualification_id\":%s,"
        "\"owner_budget_bytes\":[%llu,%llu],\"owner_used_bytes\":[%llu,%llu],"
        "\"live_allocation_count\":%llu,\"live_reservation_count\":%llu,"
        "\"cleanup_pending_count\":%llu}\n",
        event.c_str(), mode_name(current_mode()), anchor.c_str(), peer.c_str(), qualification.c_str(),
        static_cast<unsigned long long>(usage.owner_budget_bytes[0]),
        static_cast<unsigned long long>(usage.owner_budget_bytes[1]),
        static_cast<unsigned long long>(usage.owner_used_bytes[0]),
        static_cast<unsigned long long>(usage.owner_used_bytes[1]),
        static_cast<unsigned long long>(usage.live_allocation_count),
        static_cast<unsigned long long>(usage.live_reservation_count),
        static_cast<unsigned long long>(usage.cleanup_pending_count));
    if (count <= 0 || static_cast<size_t>(count) >= sizeof(line)) return;
    std::lock_guard<std::mutex> lock(profile_log_mutex);
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

void write_graph_profile_snapshot(const char *event_name, uintptr_t handle,
                                  const GraphRewriteInfo &info) {
    const char *path = std::getenv("V200_PROFILE_LOG");
    if (path == nullptr || path[0] == '\0') return;
    auto get_profile = memory_fn<CUresult (*)(v200_memory_pair_profile *)>("v200_memory_get_profile");
    auto get_usage = memory_fn<CUresult (*)(v200_memory_usage_info *)>("v200_memory_get_usage");
    if (get_profile == nullptr || get_usage == nullptr) return;
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    if (get_profile(&profile) != CUDA_SUCCESS || get_usage(&usage) != CUDA_SUCCESS) return;

    const std::string event = json_string(event_name == nullptr ? "graph" : event_name, 128);
    const std::string anchor = json_string(profile.owner_uuid[0], sizeof(profile.owner_uuid[0]));
    const std::string peer = json_string(profile.owner_uuid[1], sizeof(profile.owner_uuid[1]));
    const std::string qualification = json_string(profile.qualification_id, sizeof(profile.qualification_id));
    char line[2304];
    const int count = std::snprintf(line, sizeof(line),
        "{\"schema\":\"v200-frontdoor-ledger/v1\",\"event\":%s,\"mode\":\"%s\","
        "\"anchor_uuid\":%s,\"peer_uuid\":%s,\"qualification_id\":%s,"
        "\"owner_budget_bytes\":[%llu,%llu],\"owner_used_bytes\":[%llu,%llu],"
        "\"live_allocation_count\":%llu,\"live_reservation_count\":%llu,"
        "\"cleanup_pending_count\":%llu,\"graph_handle\":\"0x%llx\","
        "\"graph_matched_nodes\":%llu,\"graph_rewritten_nodes\":%llu,"
        "\"graph_status\":%d}\n",
        event.c_str(), mode_name(current_mode()), anchor.c_str(), peer.c_str(), qualification.c_str(),
        static_cast<unsigned long long>(usage.owner_budget_bytes[0]),
        static_cast<unsigned long long>(usage.owner_budget_bytes[1]),
        static_cast<unsigned long long>(usage.owner_used_bytes[0]),
        static_cast<unsigned long long>(usage.owner_used_bytes[1]),
        static_cast<unsigned long long>(usage.live_allocation_count),
        static_cast<unsigned long long>(usage.live_reservation_count),
        static_cast<unsigned long long>(usage.cleanup_pending_count),
        static_cast<unsigned long long>(handle),
        static_cast<unsigned long long>(info.matched_nodes),
        static_cast<unsigned long long>(info.rewritten_nodes),
        static_cast<int>(info.status));
    if (count <= 0 || static_cast<size_t>(count) >= sizeof(line)) return;
    std::lock_guard<std::mutex> lock(profile_log_mutex);
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

} // namespace

extern "C" void v200_frontdoor_snapshot(const char *event_name) {
    write_profile_snapshot(event_name);
}
