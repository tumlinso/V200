#include "runtime_state.hpp"

#include <dlfcn.h>
#include <link.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <cstring>

namespace v200::frontdoor::detail {
namespace {
thread_local unsigned internal_depth = 0;
thread_local bool resolving = false;
std::atomic<std::uint64_t> next_stream_id{3};
thread_local std::uint64_t per_thread_default_id = 0;
std::mutex stream_mutex;
std::map<const void *, std::uint64_t> stream_ids;
std::mutex execution_mutex;
v200_execution_handle *attached_execution = nullptr;
v200_logical_device_id attached_device = 0;
v200_owner_id attached_owner = 0;
std::set<std::uint64_t> execution_streams;
std::mutex native_library_mutex;
void *native_driver_handle = nullptr;
void *native_runtime_handle = nullptr;
std::string native_driver_path;
std::string native_runtime_path;

struct CachedSymbol {
    const char *name;
    std::atomic<void *> value{nullptr};
};

} // namespace

InternalCallScope::InternalCallScope() noexcept { ++internal_depth; }
InternalCallScope::~InternalCallScope() { --internal_depth; }
bool internal_call() noexcept { return internal_depth != 0 || resolving; }

void *lookup_next(const char *symbol) noexcept {
    if (symbol == nullptr || resolving) return nullptr;
    resolving = true;
    void *result = dlsym(RTLD_NEXT, symbol);
    resolving = false;
    return result;
}

namespace {
struct NativeLibrarySearch {
    const char *prefix;
    std::string path;
};

int find_loaded_library(dl_phdr_info *info, size_t, void *opaque) {
    auto *search = static_cast<NativeLibrarySearch *>(opaque);
    if (info == nullptr || info->dlpi_name == nullptr || info->dlpi_name[0] == '\0') return 0;
    const char *base = std::strrchr(info->dlpi_name, '/');
    base = base == nullptr ? info->dlpi_name : base + 1;
    if (std::strncmp(base, search->prefix, std::strlen(search->prefix)) != 0) return 0;
    search->path = info->dlpi_name;
    return 1;
}

void *symbol_in_loaded_library(const char *symbol, bool driver) noexcept {
    if (symbol == nullptr) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(native_library_mutex);
        void *&handle = driver ? native_driver_handle : native_runtime_handle;
        std::string &path = driver ? native_driver_path : native_runtime_path;
        const char *prefix = driver ? "libcuda.so" : "libcudart.so";
        if (handle == nullptr) {
            NativeLibrarySearch search{prefix, {}};
            (void)dl_iterate_phdr(&find_loaded_library, &search);
            if (!search.path.empty()) {
                handle = dlopen(search.path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
                if (handle != nullptr) path = std::move(search.path);
            }
        }
        if (handle == nullptr) return nullptr;
        return dlsym(handle, symbol);
    } catch (...) {
        return nullptr;
    }
}
} // namespace

void *lookup_driver(const char *symbol) noexcept {
    if (symbol == nullptr || resolving) return nullptr;
    resolving = true;
    void *result = symbol_in_loaded_library(symbol, true);
    resolving = false;
    return result;
}

void *lookup_runtime(const char *symbol) noexcept {
    if (symbol == nullptr || resolving) return nullptr;
    resolving = true;
    void *result = symbol_in_loaded_library(symbol, false);
    resolving = false;
    return result;
}

void *lookup_native(const char *symbol) noexcept {
    if (symbol == nullptr || resolving) return nullptr;
    resolving = true;
    void *result = dlsym(RTLD_NEXT, symbol);
    if (result == nullptr) {
        const bool runtime = std::strncmp(symbol, "cuda", 4) == 0;
        const bool driver = !runtime && std::strncmp(symbol, "cu", 2) == 0;
        result = symbol_in_loaded_library(symbol, driver);
    }
    resolving = false;
    return result;
}

void *lookup_v200(const char *symbol) noexcept {
    if (symbol == nullptr || resolving) return nullptr;
    resolving = true;
    void *result = dlsym(RTLD_DEFAULT, symbol);
    resolving = false;
    return result;
}

uint32_t current_mode() noexcept {
    const char *value = std::getenv("V200_MODE");
    if (value == nullptr || value[0] == '\0' || value[0] == 'o' || value[0] == '0') return 0;
    if (value[0] == 'a' || value[0] == 'A' || value[0] == '1') return 1;
    if (value[0] == 'p' || value[0] == 'P' || value[0] == '2') return 2;
    return UINT32_MAX;
}

bool pair_mode() noexcept { return current_mode() == 2; }
bool presentation_enabled() noexcept {
    const auto mode = current_mode();
    return mode == 1 || mode == 2;
}

const char *mode_name(uint32_t mode) noexcept {
    switch (mode) {
        case 0: return "native";
        case 1: return "anchor";
        case 2: return "pair";
        default: return "invalid";
    }
}

uint64_t stream_id(const void *stream) noexcept {
    // CUDA's legacy default is null; per-thread default is the fixed sentinel value 2.
    const auto numeric = reinterpret_cast<std::uintptr_t>(stream);
    std::lock_guard<std::mutex> lock(stream_mutex);
    if (stream == nullptr) return 1;
    if (numeric == 2) {
        if (per_thread_default_id == 0)
            per_thread_default_id = next_stream_id.fetch_add(1, std::memory_order_relaxed);
        return per_thread_default_id;
    }
    auto [it, inserted] = stream_ids.emplace(stream, 0);
    if (inserted) it->second = next_stream_id.fetch_add(1, std::memory_order_relaxed);
    return it->second;
}

void forget_stream(const void *stream) noexcept {
    if (stream == nullptr || reinterpret_cast<std::uintptr_t>(stream) == 2) return;
    std::lock_guard<std::mutex> lock(stream_mutex);
    stream_ids.erase(stream);
}

v200_execution_handle *execution_handle() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return attached_execution;
}

v200_logical_device_id execution_device() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return attached_device;
}

v200_owner_id execution_owner() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return attached_owner;
}

int register_execution_stream(void *stream) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    const auto id = stream_id(stream);
    const auto device = execution_device();
    const auto owner = execution_owner();
    {
        std::lock_guard<std::mutex> lock(execution_mutex);
        if (attached_execution != handle) return V200_EXECUTION_UNAVAILABLE;
        if (!execution_streams.insert(id).second) return V200_EXECUTION_OK;
    }
    using Register = int (*)(v200_execution_handle *, v200_logical_device_id,
                             v200_logical_stream_id, v200_owner_id, void *);
    auto fn = reinterpret_cast<Register>(lookup_v200("v200_execution_register_stream"));
    if (fn == nullptr) {
        std::lock_guard<std::mutex> lock(execution_mutex);
        execution_streams.erase(id);
        return V200_EXECUTION_UNAVAILABLE;
    }
    const int result = fn(handle, device, id, owner, stream);
    if (result != V200_EXECUTION_OK && result != V200_EXECUTION_DUPLICATE_BINDING) {
        std::lock_guard<std::mutex> lock(execution_mutex);
        execution_streams.erase(id);
    }
    return result;
}

int observe_execution_submission(void *stream, int driver_result) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    const int registered = register_execution_stream(stream);
    if (registered != V200_EXECUTION_OK && registered != V200_EXECUTION_DUPLICATE_BINDING) return registered;
    using Observe = int (*)(v200_execution_handle *, v200_logical_device_id,
                            v200_logical_stream_id, v200_owner_id, void *, int);
    auto fn = reinterpret_cast<Observe>(lookup_v200("v200_execution_observe_native_submission"));
    if (fn == nullptr) return V200_EXECUTION_UNAVAILABLE;
    return fn(handle, execution_device(), stream_id(stream), execution_owner(), stream, driver_result);
}

int execution_token_for_stream(void *stream, v200_completion_token *token) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr || token == nullptr) return V200_EXECUTION_UNAVAILABLE;
    const int registered = register_execution_stream(stream);
    if (registered != V200_EXECUTION_OK && registered != V200_EXECUTION_DUPLICATE_BINDING) return registered;
    using Current = int (*)(v200_execution_handle *, v200_logical_device_id,
                            v200_logical_stream_id, v200_completion_token *);
    auto fn = reinterpret_cast<Current>(lookup_v200("v200_execution_current_token"));
    return fn ? fn(handle, execution_device(), stream_id(stream), token) : V200_EXECUTION_UNAVAILABLE;
}

int execution_token_for_device(v200_completion_token *token) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr || token == nullptr) return V200_EXECUTION_UNAVAILABLE;
    using Current = int (*)(v200_execution_handle *, v200_logical_device_id, v200_completion_token *);
    auto fn = reinterpret_cast<Current>(lookup_v200("v200_execution_current_device_token"));
    return fn ? fn(handle, execution_device(), token) : V200_EXECUTION_UNAVAILABLE;
}

int execution_wait_token(v200_completion_token token, int *driver_result) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    using Wait = int (*)(v200_execution_handle *, v200_completion_token, int *);
    auto fn = reinterpret_cast<Wait>(lookup_v200("v200_execution_wait"));
    return fn ? fn(handle, token, driver_result) : V200_EXECUTION_UNAVAILABLE;
}

int execution_release_token(v200_completion_token token) noexcept {
    auto *handle = execution_handle();
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    using Release = int (*)(v200_execution_handle *, v200_completion_token);
    auto fn = reinterpret_cast<Release>(lookup_v200("v200_execution_release"));
    return fn ? fn(handle, token) : V200_EXECUTION_UNAVAILABLE;
}

int attach_execution(v200_execution_handle *handle, v200_logical_device_id device,
                     v200_owner_id owner) noexcept {
    if (handle == nullptr || device == 0) return V200_EXECUTION_INVALID_ARGUMENT;
    using Register = int (*)(v200_execution_handle *, v200_logical_device_id,
                             v200_logical_stream_id, v200_owner_id, void *);
    auto register_stream = reinterpret_cast<Register>(lookup_v200("v200_execution_register_stream"));
    if (register_stream == nullptr) return V200_EXECUTION_UNAVAILABLE;
    {
        std::lock_guard<std::mutex> lock(execution_mutex);
        if (attached_execution != nullptr && attached_execution != handle) return V200_EXECUTION_RESOURCE_BUSY;
        attached_execution = handle;
        attached_device = device;
        attached_owner = owner;
        execution_streams.clear();
    }
    const int legacy = register_execution_stream(nullptr);
    const int per_thread = register_execution_stream(reinterpret_cast<void *>(2));
    if ((legacy != V200_EXECUTION_OK && legacy != V200_EXECUTION_DUPLICATE_BINDING) ||
        (per_thread != V200_EXECUTION_OK && per_thread != V200_EXECUTION_DUPLICATE_BINDING)) {
        detach_execution(handle);
        return legacy != V200_EXECUTION_OK ? legacy : per_thread;
    }
    return V200_EXECUTION_OK;
}

void detach_execution(v200_execution_handle *handle) noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    if (attached_execution != handle) return;
    attached_execution = nullptr;
    attached_device = 0;
    attached_owner = 0;
    execution_streams.clear();
}

} // namespace v200::frontdoor::detail

extern "C" uint64_t v200_frontdoor_stream_id(const void *stream) {
    return v200::frontdoor::detail::stream_id(stream);
}

extern "C" void v200_frontdoor_forget_stream(const void *stream) {
    v200::frontdoor::detail::forget_stream(stream);
}

extern "C" uint32_t v200_frontdoor_mode(void) {
    return v200::frontdoor::detail::current_mode();
}

extern "C" void *v200_frontdoor_resolve_driver_symbol(const char *name) {
    return v200::frontdoor::detail::lookup_driver(name);
}

extern "C" void *v200_frontdoor_resolve_runtime_symbol(const char *name) {
    return v200::frontdoor::detail::lookup_runtime(name);
}

extern "C" int v200_frontdoor_attach_execution(v200_execution_handle *handle,
                                                  v200_logical_device_id device,
                                                  v200_owner_id owner) {
    return v200::frontdoor::detail::attach_execution(handle, device, owner);
}

extern "C" void v200_frontdoor_detach_execution(v200_execution_handle *handle) {
    v200::frontdoor::detail::detach_execution(handle);
}
