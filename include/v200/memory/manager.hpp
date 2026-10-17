#ifndef V200_MEMORY_MANAGER_HPP
#define V200_MEMORY_MANAGER_HPP

#include "v200/memory/api.h"
#include "v200/execution/owner_execution.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace v200::memory {

struct ExtentPlan {
    CUdevice owner = 0;
    size_t logical_offset = 0;
    size_t size = 0;
};

struct PlanResult {
    CUresult status = CUDA_SUCCESS;
    std::vector<ExtentPlan> extents;
};

/* Pure planning helper: no CUDA calls and no budget mutation. */
PlanResult plan_extents(size_t logical_bytes, CUdevice preferred,
                        const v200_memory_pair_profile &profile,
                        const size_t used_bytes[2]);

class MemoryManager {
public:
    explicit MemoryManager(uint32_t pair_index = 0);
    ~MemoryManager();
    MemoryManager(const MemoryManager &) = delete;
    MemoryManager &operator=(const MemoryManager &) = delete;

    CUresult configure(const v200_memory_pair_profile &profile,
                       const v200_memory_driver_api &driver);
    CUresult allocate(CUdevice preferred, size_t bytes, size_t alignment,
                      void **device_ptr);
    CUresult release(void *device_ptr);
    CUresult query(const void *ptr, v200_allocation_info *out) const;
    bool is_configured() const noexcept;
    bool owns_pointer(const void *ptr) const;
    bool owns_pointer_range(const void *ptr, size_t size) const;
    bool owns_reservation_range(CUdeviceptr ptr, size_t size) const;
    CUresult get_profile(v200_memory_pair_profile *out) const;
    CUresult get_usage(v200_memory_usage_info *out) const;

    CUresult vmm_get_granularity(size_t *granularity,
                                 const CUmemAllocationProp *prop,
                                 CUmemAllocationGranularity_flags option);
    CUresult vmm_address_reserve(CUdeviceptr *ptr, size_t size,
                                 size_t alignment, CUdeviceptr requested,
                                 unsigned long long flags);
    CUresult vmm_address_free(CUdeviceptr ptr, size_t size);
    CUresult vmm_create(CUmemGenericAllocationHandle *handle, size_t size,
                        const CUmemAllocationProp *prop,
                        unsigned long long flags);
    CUresult vmm_map(CUdeviceptr ptr, size_t size, size_t offset,
                     CUmemGenericAllocationHandle handle,
                     unsigned long long flags);
    CUresult vmm_set_access(CUdeviceptr ptr, size_t size,
                            const CUmemAccessDesc *desc, size_t count);
    CUresult vmm_unmap(CUdeviceptr ptr, size_t size);
    CUresult vmm_release(CUmemGenericAllocationHandle handle);
    bool owns_handle(CUmemGenericAllocationHandle handle) const;
    void set_completion_provider(execution::CompletionProvider *provider) noexcept;
    CUresult retain_use(CUdeviceptr ptr, uint64_t logical_stream_id,
                        execution::CompletionToken token);
    CUresult retain_all_use(uint64_t logical_stream_id,
                            execution::CompletionToken token);
    /* C++ callers without a logical stream ID keep distinct prefix refs. */
    CUresult retain_use(CUdeviceptr ptr, execution::CompletionToken token) {
        return retain_use(ptr, 0, token);
    }
    CUresult retain_all_use(execution::CompletionToken token) {
        return retain_all_use(0, token);
    }
    CUresult begin_submission(uint64_t *lease);
    CUresult end_submission(uint64_t lease, bool completion_recorded);
    CUresult retry_cleanup();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

MemoryManager *manager_for_pair(uint32_t pair_index) noexcept;
MemoryManager &default_manager();

} // namespace v200::memory

#endif
