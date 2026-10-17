#include "v200/operators/mmvq_plan.hpp"

#include <limits>

namespace v200::operators {
namespace {

bool contains(uintptr_t base, size_t bytes, uintptr_t address,
              size_t length) noexcept {
    if (address < base || length > bytes) return false;
    const uintptr_t offset = address - base;
    return offset <= bytes - length;
}

} // namespace

bool plan_q4k_mmvq_rows(uintptr_t weight, size_t bytes_per_row,
                        uint32_t rows, const v200_allocation_info &allocation,
                        const v200_memory_pair_profile &profile,
                        MmvqRows *out) noexcept {
    if (out == nullptr || weight == 0 || bytes_per_row == 0 || rows < 2 ||
        allocation.abi_version != V200_MEMORY_ABI_VERSION ||
        allocation.struct_size < sizeof(v200_allocation_info) ||
        allocation.base == 0 || allocation.mapped_bytes == 0 ||
        allocation.requested_bytes == 0 ||
        profile.abi_version != V200_MEMORY_ABI_VERSION ||
        profile.struct_size < sizeof(v200_memory_pair_profile) ||
        profile.pair_access_qualified == 0 ||
        profile.owner_device[0] == profile.owner_device[1] ||
        profile.default_owner != profile.owner_device[0]) {
        return false;
    }

    if (rows > std::numeric_limits<size_t>::max() / bytes_per_row) return false;
    const size_t matrix_bytes = static_cast<size_t>(rows) * bytes_per_row;
    if (!contains(static_cast<uintptr_t>(allocation.base),
                  allocation.requested_bytes, weight, matrix_bytes)) {
        return false;
    }
    if (allocation.owner_bytes[0] > allocation.mapped_bytes ||
        allocation.owner_bytes[1] > allocation.mapped_bytes ||
        allocation.owner_bytes[0] > allocation.mapped_bytes - allocation.owner_bytes[1] ||
        allocation.owner_bytes[0] + allocation.owner_bytes[1] != allocation.mapped_bytes) {
        return false;
    }

    MmvqRows planned{};
    if (allocation.extent_count == 1) {
        const bool on_first = allocation.owner_bytes[0] == allocation.mapped_bytes &&
                              allocation.owner_bytes[1] == 0;
        const bool on_second = allocation.owner_bytes[1] == allocation.mapped_bytes &&
                               allocation.owner_bytes[0] == 0;
        if (!on_first && !on_second) return false;
        planned.first_owner = on_first ? 0u : 1u;
        planned.first_rows = rows;
        *out = planned;
        return true;
    }

    // The allocator emits at most two contiguous extents in preferred-owner
    // order. cudaMalloc in pair mode always prefers profile.owner_device[0].
    if (allocation.extent_count != 2 || allocation.owner_bytes[0] == 0 ||
        allocation.owner_bytes[1] == 0) {
        return false;
    }
    const uintptr_t base = static_cast<uintptr_t>(allocation.base);
    if (allocation.owner_bytes[0] > std::numeric_limits<uintptr_t>::max() - base) {
        return false;
    }
    const uintptr_t boundary = base + allocation.owner_bytes[0];
    const uintptr_t end = weight + matrix_bytes;
    if (end < weight) return false;

    if (end <= boundary) {
        planned.first_owner = 0;
        planned.first_rows = rows;
    } else if (weight >= boundary) {
        planned.first_owner = 1;
        planned.first_rows = rows;
    } else {
        const uintptr_t bytes_before_boundary = boundary - weight;
        const size_t rows_before = static_cast<size_t>(bytes_before_boundary / bytes_per_row);
        const size_t split_rows = rows_before +
            ((bytes_before_boundary % bytes_per_row) != 0 ? 1u : 0u);
        if (split_rows > rows) return false;
        if (split_rows < 32 || rows - split_rows < 32) {
            planned.first_owner = split_rows * 2 >= rows ? 0u : 1u;
            planned.first_rows = rows;
        } else {
            planned.first_owner = 0;
            planned.first_rows = static_cast<uint32_t>(split_rows);
            planned.second_owner = 1;
            planned.second_rows = rows - planned.first_rows;
            planned.split = true;
        }
    }
    *out = planned;
    return true;
}

} // namespace v200::operators
