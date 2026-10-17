#pragma once

#include <cstdint>
#include <string_view>

namespace v200::frontdoor {

enum class ApiDisposition : std::uint8_t {
    wrapped,
    native_anchor_passthrough,
    unsupported,
    unknown,
};

enum class StreamVariant : std::uint8_t {
    legacy,
    per_thread,
};

struct ApiEntry {
    std::string_view name;
    int minimum_cuda_version;
    ApiDisposition disposition;
    bool has_per_thread_variant;
};

const ApiEntry *find_api(std::string_view name) noexcept;
ApiDisposition classify_api(std::string_view name) noexcept;
bool version_available(const ApiEntry &entry, int cuda_version) noexcept;

} // namespace v200::frontdoor
