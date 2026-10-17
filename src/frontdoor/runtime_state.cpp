#include "runtime_state.hpp"
#include "positive_symbol_cache.hpp"

#include <dlfcn.h>
#include <link.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <cstring>

namespace v200::frontdoor::detail {
namespace {
struct CachedExecutionCurrentToken {};
struct CachedExecutionReleaseToken {};

thread_local unsigned internal_depth = 0;
thread_local bool resolving = false;
std::atomic<std::uint64_t> next_stream_id{3};
thread_local std::uint32_t selected_pair = 0;
thread_local std::array<std::uint64_t, 2> per_thread_default_ids{};
std::mutex stream_mutex;
std::map<std::pair<std::uint32_t, const void *>, std::uint64_t> stream_ids;
std::mutex execution_mutex;
struct ExecutionBinding {
    v200_execution_handle *handle = nullptr;
    v200_logical_device_id device = 0;
    v200_owner_id owner = 0;
    std::set<std::uint64_t> streams;
};
std::array<ExecutionBinding, 2> execution_bindings;
std::mutex native_library_mutex;
void *native_driver_handle = nullptr;
void *native_runtime_handle = nullptr;
std::string native_driver_path;
std::string native_runtime_path;

struct CachedSymbol {
    const char *name;
    std::atomic<void *> value{nullptr};
};

std::uint64_t stream_id_for_pair(const void *stream, std::uint32_t pair) noexcept {
    const auto numeric = reinterpret_cast<std::uintptr_t>(stream);
    std::lock_guard<std::mutex> lock(stream_mutex);
    // Give each pair a distinct legacy default stream while keeping pair 0's
    // historical ID. Explicit stream IDs remain process-unique.
    if (stream == nullptr) return static_cast<std::uint64_t>(pair) + 1;
    if (numeric == 2) {
        auto &id = per_thread_default_ids[pair];
        if (id == 0) id = next_stream_id.fetch_add(1, std::memory_order_relaxed);
        return id;
    }
    const auto key = std::make_pair(pair, stream);
    auto [it, inserted] = stream_ids.emplace(key, 0);
    if (inserted) it->second = next_stream_id.fetch_add(1, std::memory_order_relaxed);
    return it->second;
}

int find_stream_pair_locked(const void *stream) noexcept {
    for (std::uint32_t pair = 0; pair != 2; ++pair) {
        const auto found = stream_ids.find(std::make_pair(pair, stream));
        if (found != stream_ids.end()) return static_cast<int>(pair);
    }
    return -1;
}

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

uint32_t current_pair() noexcept { return selected_pair; }

int select_pair(uint32_t pair_index) noexcept {
    if (pair_index >= execution_bindings.size()) return -1;
    const std::uint32_t previous = selected_pair;
    selected_pair = pair_index;
    using Select = int (*)(uint32_t);
    auto select_memory = reinterpret_cast<Select>(lookup_v200("v200_memory_select_pair"));
    if (select_memory != nullptr) {
        const int status = select_memory(pair_index);
        if (status != 0) {
            selected_pair = previous;
            (void)select_memory(previous);
            return status;
        }
    }
    return 0;
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
    const int owner = stream_owner_pair(stream);
    return stream_id_for_pair(stream, owner < 0 ? current_pair()
                                                : static_cast<uint32_t>(owner));
}

int stream_owner_pair(const void *stream) noexcept {
    if (stream == nullptr || reinterpret_cast<std::uintptr_t>(stream) == 2)
        return static_cast<int>(current_pair());
    std::lock_guard<std::mutex> lock(stream_mutex);
    const auto active = stream_ids.find(std::make_pair(current_pair(), stream));
    if (active != stream_ids.end()) return static_cast<int>(current_pair());
    return find_stream_pair_locked(stream);
}

void forget_stream(const void *stream) noexcept {
    if (stream == nullptr || reinterpret_cast<std::uintptr_t>(stream) == 2) return;
    std::lock_guard<std::mutex> lock(stream_mutex);
    const int owner = find_stream_pair_locked(stream);
    if (owner >= 0) stream_ids.erase(std::make_pair(static_cast<uint32_t>(owner), stream));
}

v200_execution_handle *execution_handle() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return execution_bindings[current_pair()].handle;
}

v200_logical_device_id execution_device() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return execution_bindings[current_pair()].device;
}

v200_owner_id execution_owner() noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return execution_bindings[current_pair()].owner;
}

int register_execution_stream(void *stream) noexcept {
    uint32_t pair = current_pair();
    const int stream_pair = stream_owner_pair(stream);
    if (stream_pair >= 0) pair = static_cast<uint32_t>(stream_pair);
    return register_execution_stream_for_pair(stream, pair);
}

int register_execution_stream_for_pair(void *stream, uint32_t pair) noexcept {
    if (pair >= execution_bindings.size()) return V200_EXECUTION_INVALID_ARGUMENT;
    const auto id = stream_id_for_pair(stream, pair);
    v200_execution_handle *handle = nullptr;
    v200_logical_device_id device = 0;
    v200_owner_id owner = 0;
    {
        std::lock_guard<std::mutex> lock(execution_mutex);
        auto &binding = execution_bindings[pair];
        handle = binding.handle;
        device = binding.device;
        owner = binding.owner;
        if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
        if (!binding.streams.insert(id).second) return V200_EXECUTION_OK;
    }
    if (handle == nullptr) return V200_EXECUTION_UNAVAILABLE;
    using Register = int (*)(v200_execution_handle *, v200_logical_device_id,
                             v200_logical_stream_id, v200_owner_id, void *);
    auto fn = reinterpret_cast<Register>(lookup_v200("v200_execution_register_stream"));
    if (fn == nullptr) {
        std::lock_guard<std::mutex> lock(execution_mutex);
        if (execution_bindings[pair].handle == handle)
            execution_bindings[pair].streams.erase(id);
        return V200_EXECUTION_UNAVAILABLE;
    }
    const int result = fn(handle, device, id, owner, stream);
    if (result != V200_EXECUTION_OK && result != V200_EXECUTION_DUPLICATE_BINDING) {
        std::lock_guard<std::mutex> lock(execution_mutex);
        if (execution_bindings[pair].handle == handle)
            execution_bindings[pair].streams.erase(id);
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
    auto fn = positive_symbol<CachedExecutionCurrentToken, Current>([]() noexcept {
        return reinterpret_cast<Current>(lookup_v200("v200_execution_current_token"));
    });
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
    auto fn = positive_symbol<CachedExecutionReleaseToken, Release>([]() noexcept {
        return reinterpret_cast<Release>(lookup_v200("v200_execution_release"));
    });
    return fn ? fn(handle, token) : V200_EXECUTION_UNAVAILABLE;
}

int attach_execution(v200_execution_handle *handle, v200_logical_device_id device,
                     v200_owner_id owner) noexcept {
    if (handle == nullptr || device == 0 || device > execution_bindings.size())
        return V200_EXECUTION_INVALID_ARGUMENT;
    const uint32_t pair = static_cast<uint32_t>(device - 1);
    using Register = int (*)(v200_execution_handle *, v200_logical_device_id,
                             v200_logical_stream_id, v200_owner_id, void *);
    auto register_stream = reinterpret_cast<Register>(lookup_v200("v200_execution_register_stream"));
    if (register_stream == nullptr) return V200_EXECUTION_UNAVAILABLE;
    {
        std::lock_guard<std::mutex> lock(execution_mutex);
        auto &binding = execution_bindings[pair];
        if (binding.handle != nullptr) {
            if (binding.handle == handle && binding.device == device && binding.owner == owner)
                return V200_EXECUTION_OK;
            return V200_EXECUTION_RESOURCE_BUSY;
        }
        for (std::size_t other = 0; other < execution_bindings.size(); ++other) {
            if (other != pair && execution_bindings[other].handle == handle)
                return V200_EXECUTION_RESOURCE_BUSY;
        }
        binding.handle = handle;
        binding.device = device;
        binding.owner = owner;
        binding.streams.clear();
    }
    const int legacy = register_execution_stream_for_pair(nullptr, pair);
    const int per_thread = register_execution_stream_for_pair(reinterpret_cast<void *>(2), pair);
    if ((legacy != V200_EXECUTION_OK && legacy != V200_EXECUTION_DUPLICATE_BINDING) ||
        (per_thread != V200_EXECUTION_OK && per_thread != V200_EXECUTION_DUPLICATE_BINDING)) {
        detach_execution(handle);
        return legacy != V200_EXECUTION_OK ? legacy : per_thread;
    }
    return V200_EXECUTION_OK;
}

void detach_execution(v200_execution_handle *handle) noexcept {
    std::lock_guard<std::mutex> lock(execution_mutex);
    for (auto &binding : execution_bindings) {
        if (binding.handle != handle) continue;
        binding = ExecutionBinding{};
    }
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

extern "C" uint32_t v200_frontdoor_current_pair(void) {
    return v200::frontdoor::detail::current_pair();
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
