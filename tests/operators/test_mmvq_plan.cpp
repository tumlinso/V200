#include "v200/operators/mmvq_plan.hpp"

#include <cassert>
#include <cstdint>

namespace {

v200_memory_pair_profile profile() {
    v200_memory_pair_profile value{};
    value.abi_version = V200_MEMORY_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.owner_device[0] = 0;
    value.owner_device[1] = 1;
    value.default_owner = 0;
    value.pair_access_qualified = 1;
    return value;
}

v200_allocation_info allocation(uintptr_t base, size_t mapped,
                                size_t first, size_t second, uint32_t extents) {
    v200_allocation_info value{};
    value.abi_version = V200_MEMORY_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.base = base;
    value.requested_bytes = mapped;
    value.mapped_bytes = mapped;
    value.owner_bytes[0] = first;
    value.owner_bytes[1] = second;
    value.extent_count = extents;
    return value;
}

} // namespace

int main() {
    constexpr uintptr_t base = 0x10000000;
    constexpr size_t row_bytes = 2048;
    constexpr uint32_t rows = 128;
    constexpr size_t total = row_bytes * rows;
    auto pair = profile();
    v200::operators::MmvqRows result{};

    auto first_only = allocation(base, total, total, 0, 1);
    assert(v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                               first_only, pair, &result));
    assert(!result.split && result.first_owner == 0 && result.first_rows == rows);

    auto peer_only = allocation(base, total, 0, total, 1);
    assert(v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                               peer_only, pair, &result));
    assert(!result.split && result.first_owner == 1 && result.first_rows == rows);

    auto evenly_split = allocation(base, total, total / 2, total / 2, 2);
    assert(v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                               evenly_split, pair, &result));
    assert(result.split && result.first_owner == 0 && result.first_rows == 64 &&
           result.second_owner == 1 && result.second_rows == 64);

    auto unaligned = allocation(base, total, total / 2 + 1, total / 2 - 1, 2);
    assert(v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                               unaligned, pair, &result));
    assert(result.split && result.first_rows == 65 && result.second_rows == 63);

    auto bad_owner_sum = allocation(base, total, total / 2, total / 2 - 1, 2);
    assert(!v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                                bad_owner_sum, pair, &result));

    auto inside_split = evenly_split;
    inside_split.base = base;
    assert(v200::operators::plan_q4k_mmvq_rows(base + row_bytes * 4, row_bytes,
                                               rows - 4, inside_split, pair, &result));
    assert(result.split && result.first_rows == 60 && result.second_rows == 64);

    pair.default_owner = 1;
    assert(!v200::operators::plan_q4k_mmvq_rows(base, row_bytes, rows,
                                                evenly_split, pair, &result));
}
