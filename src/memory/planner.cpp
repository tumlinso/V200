#include "v200/memory/manager.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace v200::memory {
namespace {

size_t gcd_size(size_t a, size_t b) noexcept {
    while (b != 0) {
        const size_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

bool checked_mul(size_t a, size_t b, size_t *out) noexcept {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<size_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool checked_add(size_t a, size_t b, size_t *out) noexcept {
    if (out == nullptr || b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    *out = a + b;
    return true;
}

uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t modulus) noexcept {
    uint64_t result = 0;
    a %= modulus;
    while (b != 0) {
        if ((b & 1u) != 0) {
            result = result >= modulus - a ? result - (modulus - a) : result + a;
        }
        b >>= 1u;
        if (b != 0) {
            a = a >= modulus - a ? a - (modulus - a) : a + a;
        }
    }
    return result;
}

bool modular_inverse(uint64_t value, uint64_t modulus, uint64_t *inverse) noexcept {
    if (inverse == nullptr || modulus == 0) {
        return false;
    }
    if (modulus == 1) {
        *inverse = 0;
        return true;
    }
    uint64_t t = 0;
    uint64_t next_t = 1;
    uint64_t r = modulus;
    uint64_t next_r = value % modulus;
    while (next_r != 0) {
        const uint64_t quotient = r / next_r;
        const uint64_t product = mul_mod(quotient, next_t, modulus);
        const uint64_t coefficient = t >= product ? t - product : modulus - (product - t);
        const uint64_t old_r = r;
        r = next_r;
        next_r = old_r - quotient * next_r;
        t = next_t;
        next_t = coefficient;
    }
    if (r != 1) {
        return false;
    }
    *inverse = t;
    return true;
}

} // namespace

PlanResult plan_extents(size_t logical_bytes, CUdevice preferred,
                        const v200_memory_pair_profile &profile,
                        const size_t used_bytes[2]) {
    PlanResult result;
    if (logical_bytes == 0 || used_bytes == nullptr ||
        profile.owner_device[0] == profile.owner_device[1] ||
        profile.owner_granularity[0] == 0 || profile.owner_granularity[1] == 0) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }

    const size_t preferred_index = profile.owner_device[0] == preferred ? 0u :
                                   profile.owner_device[1] == preferred ? 1u : 2u;
    if (preferred_index > 1) {
        result.status = CUDA_ERROR_INVALID_DEVICE;
        return result;
    }
    const size_t other_index = 1u - preferred_index;
    const size_t preferred_granularity = profile.owner_granularity[preferred_index];
    const size_t other_granularity = profile.owner_granularity[other_index];
    const uint64_t preferred_budget = profile.owner_budget_bytes[preferred_index];
    const uint64_t other_budget = profile.owner_budget_bytes[other_index];
    if (used_bytes[preferred_index] > preferred_budget ||
        used_bytes[other_index] > other_budget) {
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    const uint64_t preferred_free_u64 = preferred_budget - used_bytes[preferred_index];
    const uint64_t other_free_u64 = other_budget - used_bytes[other_index];
    const size_t preferred_free = preferred_free_u64 > std::numeric_limits<size_t>::max()
                                      ? std::numeric_limits<size_t>::max()
                                      : static_cast<size_t>(preferred_free_u64);
    const size_t other_free = other_free_u64 > std::numeric_limits<size_t>::max()
                                  ? std::numeric_limits<size_t>::max()
                                  : static_cast<size_t>(other_free_u64);
    const size_t preferred_capacity = preferred_free - preferred_free % preferred_granularity;
    const size_t other_capacity = other_free - other_free % other_granularity;

    if (logical_bytes % preferred_granularity != 0) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    if (logical_bytes <= preferred_capacity) {
        result.extents.push_back({preferred, 0, logical_bytes});
        return result;
    }
    if (profile.pair_access_qualified == 0) {
        result.status = CUDA_ERROR_NOT_SUPPORTED;
        return result;
    }
    if (preferred_capacity <= std::numeric_limits<size_t>::max() - other_capacity &&
        logical_bytes > preferred_capacity + other_capacity) {
        result.status = CUDA_ERROR_OUT_OF_MEMORY;
        return result;
    }

    const size_t gcd = gcd_size(preferred_granularity, other_granularity);
    if (logical_bytes % gcd != 0) {
        result.status = CUDA_ERROR_NOT_SUPPORTED;
        return result;
    }
    const size_t reduced_modulus = other_granularity / gcd;
    size_t lcm = 0;
    if (!checked_mul(preferred_granularity, reduced_modulus, &lcm)) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    uint64_t k_residue = 0;
    if (reduced_modulus > 1) {
        uint64_t inverse = 0;
        const uint64_t reduced_preferred = preferred_granularity / gcd;
        const uint64_t rhs = logical_bytes / gcd;
        if (!modular_inverse(reduced_preferred, reduced_modulus, &inverse)) {
            result.status = CUDA_ERROR_NOT_SUPPORTED;
            return result;
        }
        k_residue = mul_mod(rhs, inverse, reduced_modulus);
    }
    size_t first_residue = 0;
    if (!checked_mul(preferred_granularity, static_cast<size_t>(k_residue), &first_residue)) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }

    const size_t minimum_first = logical_bytes > other_capacity ? logical_bytes - other_capacity : 0;
    const size_t maximum_first = std::min(logical_bytes, preferred_capacity);
    size_t first = first_residue;
    if (first < minimum_first) {
        const size_t gap = minimum_first - first;
        size_t multiples = gap / lcm;
        if (gap % lcm != 0 && !checked_add(multiples, 1, &multiples)) {
            result.status = CUDA_ERROR_INVALID_VALUE;
            return result;
        }
        size_t advance = 0;
        if (!checked_mul(multiples, lcm, &advance) || !checked_add(first, advance, &first)) {
            result.status = CUDA_ERROR_INVALID_VALUE;
            return result;
        }
    }
    if (first > maximum_first || logical_bytes - first > other_capacity) {
        result.status = CUDA_ERROR_OUT_OF_MEMORY;
        return result;
    }

    if (first != 0) {
        result.extents.push_back({preferred, 0, first});
    }
    const size_t second = logical_bytes - first;
    if (second != 0) {
        result.extents.push_back({profile.owner_device[other_index], first, second});
    }
    if (result.extents.empty() ||
        result.extents.front().logical_offset != 0 ||
        result.extents.back().logical_offset + result.extents.back().size != logical_bytes) {
        result.extents.clear();
        result.status = CUDA_ERROR_INVALID_VALUE;
    }
    return result;
}

} // namespace v200::memory
