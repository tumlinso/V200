#include "runtime_state.hpp"

#include "v200/frontdoor/abi_profile.hpp"
#include "v200/execution/owner_execution_c.h"
#include "v200/memory/api.h"

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(
    CUdeviceptr, const void *, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(
    void *, CUdeviceptr, size_t, CUstream);
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(
    CUdeviceptr, CUdeviceptr, size_t, CUstream);
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream);
extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream, CUevent, unsigned int);

namespace {

struct DeviceSelection {
    bool valid = false;
    bool attempted = false;
    int count = 0;
    int ordinal[2] = {-1, -1};
    CUdevice device[2] = {-1, -1};
    CUuuid uuid[2]{};
    CUresult driver_error = CUDA_ERROR_NOT_INITIALIZED;
};

std::mutex selection_mutex;
DeviceSelection selection;
std::mutex profile_mutex;
int profile_state = 0; // 0 unseen, 1 initializing, 2 ready, 3 failed
CUresult profile_error = CUDA_ERROR_NOT_INITIALIZED;
std::shared_mutex tracked_use_gate;
std::mutex stream_gate_mutex;
std::map<uint64_t, std::shared_ptr<std::mutex>> stream_gates;
thread_local bool profile_initializing = false;

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

bool parse_uuid_list(const char *text, std::array<std::array<unsigned char, 16>, 2> &out,
                     int &count) {
    count = 0;
    if (text == nullptr || *text == '\0') return false;
    const char *cursor = text;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        if (count == 2 || *cursor == '\0') return false;
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
    return count == 1 || count == 2;
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

    std::array<std::array<unsigned char, 16>, 2> wanted{};
    int wanted_count = 0;
    const int expected_count = mode == 2 ? 2 : 1;
    if (!parse_uuid_list(std::getenv("V200_DEVICE_UUIDS"), wanted, wanted_count) ||
        wanted_count != expected_count) {
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

    bool found[2] = {false, false};
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
    if (logical != 0 || !ensure_device_selection()) return false;
    ordinal = selection.ordinal[0];
    device = selection.device[0];
    return true;
}

bool map_device_id(CUdevice logical, CUdevice &physical) {
    int ordinal = -1;
    return logical_device(static_cast<int>(logical), ordinal, physical);
}

int logical_ordinal_for_physical(int physical) {
    if (!ensure_device_selection()) return -1;
    for (int i = 0; i < selection.count; ++i) {
        if (selection.ordinal[i] == physical) return 0;
    }
    return -1;
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

CUresult ensure_pair_profile();

bool pair_memory_ready() {
    if (!pair_mode() || ensure_pair_profile() != CUDA_SUCCESS) return false;
    auto configured = memory_fn<MemConfigured>("v200_memory_is_configured");
    return configured != nullptr && configured() != 0;
}

bool execution_attached() noexcept { return execution_handle() != nullptr; }

CUresult require_pair_profile() {
    if (!pair_mode()) return CUDA_SUCCESS;
    return ensure_pair_profile();
}

CUresult ensure_pair_profile() {
    if (!pair_mode()) return CUDA_ERROR_INVALID_VALUE;
    if (!ensure_device_selection()) return selection.driver_error;
    if (profile_initializing) return CUDA_ERROR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lock(profile_mutex);
    if (profile_state == 2) return CUDA_SUCCESS;
    if (profile_state == 3 || profile_state == 1) return profile_error;
    profile_state = 1;
    profile_initializing = true;
    using Initialize = CUresult (*)();
    auto initialize = memory_fn<Initialize>("v200_initialize_profile");
    if (initialize == nullptr) {
        profile_error = CUDA_ERROR_NOT_INITIALIZED;
    } else {
        InternalCallScope bypass;
        profile_error = initialize();
    }
    profile_initializing = false;
    if (profile_error == CUDA_SUCCESS) {
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
            profile_error = CUDA_ERROR_NOT_INITIALIZED;
        }
    }
    profile_state = profile_error == CUDA_SUCCESS ? 2 : 3;
    if (profile_error == CUDA_SUCCESS) v200_frontdoor_snapshot("profile_initialized");
    return profile_error;
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return frontdoor_error<Result>(profile);
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
    if (requested == "cuMemcpyHtoDAsync") requested = "cuMemcpyHtoDAsync_v2";
    if (requested == "cuMemcpyDtoHAsync") requested = "cuMemcpyDtoHAsync_v2";
    if (requested == "cuMemcpyDtoDAsync") requested = "cuMemcpyDtoDAsync_v2";

    if (requested == "cuDeviceGetCount") return reinterpret_cast<void *>(&cuDeviceGetCount);
    if (requested == "cuDeviceGet") return reinterpret_cast<void *>(&cuDeviceGet);
    if (requested == "cuGetProcAddress_v2") return reinterpret_cast<void *>(&cuGetProcAddress_v2);
    if (requested == "cuMemAddressReserve") return reinterpret_cast<void *>(&cuMemAddressReserve);
    if (requested == "cuMemAddressFree") return reinterpret_cast<void *>(&cuMemAddressFree);
    if (requested == "cuMemCreate") return reinterpret_cast<void *>(&cuMemCreate);
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
        if (current == selection.ordinal[0]) return cudaSuccess;
        return set(selection.ordinal[0]);
    }
}

cudaError_t runtime_device_ordinal(int logical, int &physical) {
    if (!presentation_enabled()) {
        physical = logical;
        return cudaSuccess;
    }
    if (!ensure_device_selection()) return cuda_error_for(selection.driver_error);
    if (logical != 0) return cudaErrorInvalidDevice;
    physical = selection.ordinal[0];
    return cudaSuccess;
}

cudaError_t register_created_stream(cudaStream_t stream) {
    if (!execution_attached()) return cudaSuccess;
    const int status = register_execution_stream(stream);
    if (status == V200_EXECUTION_OK || status == V200_EXECUTION_DUPLICATE_BINDING) return cudaSuccess;
    return cudaErrorNotSupported;
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

cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    using Fn = cudaError_t (CUDARTAPI *)(int *);
    auto real = next_fn<Fn>("cudaGetDeviceCount");
    if (internal_call() || !presentation_enabled()) return real ? real(count) : cudaErrorNotSupported;
    if (count == nullptr) return cudaErrorInvalidValue;
    if (!ensure_device_selection()) return cuda_error_for(selection.driver_error);
    *count = 1;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    using Fn = cudaError_t (CUDARTAPI *)(int *);
    auto real = next_fn<Fn>("cudaGetDevice");
    if (internal_call() || !presentation_enabled()) return real ? real(device) : cudaErrorNotSupported;
    if (device == nullptr) return cudaErrorInvalidValue;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    *device = 0;
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
    InternalCallScope bypass;
    return real(physical);
}

cudaError_t CUDARTAPI cudaGetDeviceProperties_v2(cudaDeviceProp *properties, int device) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaDeviceProp *, int);
    auto real = next_fn<Fn>("cudaGetDeviceProperties_v2");
    if (!real) real = next_fn<Fn>("cudaGetDeviceProperties");
    if (internal_call() || !presentation_enabled()) return real ? real(properties, device) : cudaErrorNotSupported;
    int physical = -1;
    const cudaError_t mapped = runtime_device_ordinal(device, physical);
    if (mapped != cudaSuccess) return mapped;
    InternalCallScope bypass;
    return real ? real(properties, physical) : cudaErrorNotSupported;
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
    if (allocate != nullptr) allocated = allocate(selection.device[0], bytes, 0, &result);
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
        auto owns = memory_fn<MemOwnsPointer>("v200_memory_owns_pointer");
        if (owns != nullptr && owns(device_ptr) != 0) {
            std::unique_lock<std::shared_mutex> gate(tracked_use_gate);
            auto release = memory_fn<MemReleasePointer>("v200_memory_release");
            const CUresult result = release ? release(device_ptr) : CUDA_ERROR_NOT_SUPPORTED;
            v200_frontdoor_snapshot("cuda_free");
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
    InternalCallScope bypass;
    return real ? real(free_bytes, total_bytes) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaMemset(void *device_ptr, int value, size_t bytes) {
    using Fn = cudaError_t (CUDARTAPI *)(void *, int, size_t);
    auto real = next_fn<Fn>("cudaMemset");
    if (internal_call()) return real ? real(device_ptr, value, bytes) : cudaErrorNotSupported;
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return cuda_error_for(profile);
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    std::shared_lock<std::shared_mutex> use_lock(tracked_use_gate);
    InternalCallScope bypass;
    return real ? real(device_ptr, value, bytes) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void *function, dim3 grid, dim3 block,
                                      void **arguments, size_t shared_bytes,
                                      cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaLaunchKernel");
    if (internal_call() || !real)
        return real ? real(function, grid, block, arguments, shared_bytes, stream)
                    : cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
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
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    const cudaStream_t stream = config == nullptr ? nullptr : config->stream;
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
    if (result == cudaSuccess && stream != nullptr) return register_created_stream(*stream);
    return result;
}

cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t *, unsigned int, int);
    auto real = next_fn<Fn>("cudaStreamCreateWithPriority");
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    InternalCallScope bypass;
    const cudaError_t result = real ? real(stream, flags, priority) : cudaErrorNotSupported;
    if (result == cudaSuccess && stream != nullptr) return register_created_stream(*stream);
    return result;
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t);
    auto real = next_fn<Fn>("cudaStreamDestroy");
    if (!real) return cudaErrorNotSupported;
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
        forget_stream(stream);
        forget_stream_gate(stream);
    }
    return result;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t);
    auto real = next_fn<Fn>("cudaStreamSynchronize");
    if (!real) return cudaErrorNotSupported;
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

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaEvent_t, cudaStream_t);
    auto real = next_fn<Fn>("cudaEventRecord");
    if (!real) return cudaErrorNotSupported;
    const cudaError_t prepared = prepare_async_call();
    if (prepared != cudaSuccess) return prepared;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(event, reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, cudaEvent_t, unsigned int);
    auto real = next_fn<Fn>("cudaStreamWaitEvent");
    if (!real) return cudaErrorNotSupported;
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
    if (pair_mode()) return cudaErrorNotSupported;
    const cudaError_t prepared = prepare_runtime_call();
    if (prepared != cudaSuccess) return prepared;
    if (!real) return cudaErrorNotSupported;
    return dispatch_async(stream, [=](void *native_stream) {
        return real(graph, reinterpret_cast<cudaStream_t>(native_stream));
    });
}

cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, enum cudaStreamCaptureMode);
    auto real = next_fn<Fn>("cudaStreamBeginCapture");
    if (pair_mode()) return cudaErrorNotSupported;
    InternalCallScope bypass;
    return real ? real(stream, mode) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *graph) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, cudaGraph_t *);
    auto real = next_fn<Fn>("cudaStreamEndCapture");
    if (pair_mode()) return cudaErrorNotSupported;
    InternalCallScope bypass;
    return real ? real(stream, graph) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream,
                                            enum cudaStreamCaptureStatus *status) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaStream_t, enum cudaStreamCaptureStatus *);
    auto real = next_fn<Fn>("cudaStreamIsCapturing");
    if (internal_call()) return real ? real(stream, status) : cudaErrorNotSupported;
    if (pair_mode()) return cudaErrorNotSupported;
    InternalCallScope bypass;
    return real ? real(stream, status) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *graph_exec, cudaGraph_t graph,
                                           unsigned long long flags) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
    auto real = next_fn<Fn>("cudaGraphInstantiate");
    if (pair_mode()) return cudaErrorNotSupported;
    InternalCallScope bypass;
    return real ? real(graph_exec, graph, flags) : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t graph_exec, cudaGraph_t graph,
                                          cudaGraphExecUpdateResultInfo *result_info) {
    using Fn = cudaError_t (CUDARTAPI *)(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo *);
    auto real = next_fn<Fn>("cudaGraphExecUpdate");
    if (pair_mode()) return cudaErrorNotSupported;
    InternalCallScope bypass;
    return real ? real(graph_exec, graph, result_info) : cudaErrorNotSupported;
}

} // extern "C"

extern "C" {

/* Driver device presentation uses native CUdevice/CUcontext handles.  Only
 * the ordinal namespace is collapsed; properties remain those of the anchor. */
CUresult CUDAAPI cuDeviceGetCount(int *count) {
    using Fn = CUresult (CUDAAPI *)(int *);
    auto real = next_fn<Fn>("cuDeviceGetCount");
    if (internal_call() || !presentation_enabled()) return real ? real(count) : CUDA_ERROR_NOT_SUPPORTED;
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!ensure_device_selection()) return selection.driver_error;
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    using Fn = CUresult (CUDAAPI *)(CUdevice *, int);
    auto real = next_fn<Fn>("cuDeviceGet");
    if (internal_call() || !presentation_enabled()) return real ? real(device, ordinal) : CUDA_ERROR_NOT_SUPPORTED;
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    if (!ensure_device_selection()) return selection.driver_error;
    *device = selection.device[0];
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, const void *, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyHtoDAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(void *, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoHAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes,
                                      CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    auto real = next_fn<Fn>("cuMemcpyDtoDAsync_v2");
    if (internal_call() || !real) return real ? real(dst, src, bytes, stream) : CUDA_ERROR_NOT_SUPPORTED;
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {
        return real(dst, src, bytes, stream);
    });
}

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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {
        return real(function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                    shared_bytes, stream, kernel_params, extra);
    });
}

CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    using Fn = CUresult (CUDAAPI *)(CUstream);
    auto real = next_fn<Fn>("cuStreamSynchronize");
    if (internal_call() || !real) return real ? real(stream) : CUDA_ERROR_NOT_SUPPORTED;
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    std::shared_lock<std::shared_mutex> outer(tracked_use_gate);
    const CUstream tracked_stream = per_thread_tracking_stream(stream);
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
    const CUresult profile = require_pair_profile();
    if (profile != CUDA_SUCCESS) return profile;
    return dispatch_async(per_thread_tracking_stream(stream), [=](void *) {
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
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmAddressFree>("v200_memory_vmm_address_free");
    std::unique_lock<std::shared_mutex> use_lock(tracked_use_gate);
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
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmCreate>("v200_memory_vmm_create");
    InternalCallScope bypass;
    return fn ? fn(handle, size, prop, flags) : CUDA_ERROR_NOT_SUPPORTED;
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
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmGranularity>("v200_memory_vmm_get_granularity");
    InternalCallScope bypass;
    return fn ? fn(granularity, prop, option) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                                     CUmemGenericAllocationHandle handle, unsigned long long flags) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
        auto real = next_fn<Fn>("cuMemMap");
        return real ? real(ptr, size, offset, handle, flags) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmMap>("v200_memory_vmm_map");
    InternalCallScope bypass;
    return fn ? fn(ptr, size, offset, handle, flags) : CUDA_ERROR_NOT_SUPPORTED;
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
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmSetAccess>("v200_memory_vmm_set_access");
    InternalCallScope bypass;
    return fn ? fn(ptr, size, desc, count) : CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (internal_call() || !pair_mode()) {
        using Fn = CUresult (CUDAAPI *)(CUdeviceptr, size_t);
        auto real = next_fn<Fn>("cuMemUnmap");
        return real ? real(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!pair_memory_ready()) return ensure_pair_profile();
    auto fn = memory_fn<MemVmmUnmap>("v200_memory_vmm_unmap");
    std::unique_lock<std::shared_mutex> use_lock(tracked_use_gate);
    InternalCallScope bypass;
    return fn ? fn(ptr, size) : CUDA_ERROR_NOT_SUPPORTED;
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

} // namespace

extern "C" void v200_frontdoor_snapshot(const char *event_name) {
    write_profile_snapshot(event_name);
}

