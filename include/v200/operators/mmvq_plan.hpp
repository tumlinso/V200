#ifndef V200_OPERATORS_MMVQ_PLAN_HPP
#define V200_OPERATORS_MMVQ_PLAN_HPP

#include <cuda.h>
#include <cstddef>
#include <cstdint>

#include "v200/memory/api.h"

namespace v200::operators {

/* Row ownership for the pinned Q4_K decode kernel. The kernel's Volta
 * specialization has one output row per grid.x block. */
struct MmvqRows {
    uint32_t first_owner = 0;
    uint32_t first_rows = 0;
    uint32_t second_owner = 0;
    uint32_t second_rows = 0;
    bool split = false;
};

/* Returns false when the VMM record cannot prove an owner range. `weight`
 * may point into an allocation, as it can for a tensor view. */
bool plan_q4k_mmvq_rows(uintptr_t weight, size_t bytes_per_row,
                        uint32_t rows, const v200_allocation_info &allocation,
                        const v200_memory_pair_profile &profile,
                        MmvqRows *out) noexcept;

} // namespace v200::operators

#endif
