#ifndef V200_MEMORY_API_H
#define V200_MEMORY_API_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V200_MEMORY_ABI_VERSION 1u

/* Function table supplied by the process-scoped CUDA frontdoor.  The test
 * backend supplies the same calls with deterministic failure injection. */
typedef struct v200_memory_driver_api {
    uint32_t abi_version;
    size_t struct_size;
    void *user_data;
    CUresult (CUDAAPI *mem_address_reserve)(void *, CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (CUDAAPI *mem_address_free)(void *, CUdeviceptr, size_t);
    CUresult (CUDAAPI *mem_create)(void *, CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
    CUresult (CUDAAPI *mem_release)(void *, CUmemGenericAllocationHandle);
    CUresult (CUDAAPI *mem_map)(void *, CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (CUDAAPI *mem_unmap)(void *, CUdeviceptr, size_t);
    CUresult (CUDAAPI *mem_set_access)(void *, CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    CUresult (CUDAAPI *mem_get_granularity)(void *, size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
} v200_memory_driver_api;

/* Values are invocation-scoped facts supplied after pair qualification.  A
 * nonempty qualification_id is required before extents may use both owners. */
typedef struct v200_memory_pair_profile {
    uint32_t abi_version;
    size_t struct_size;
    CUdevice owner_device[2];
    size_t owner_granularity[2];
    uint64_t owner_budget_bytes[2];
    CUdevice default_owner;
    uint32_t pair_access_qualified;
    char owner_uuid[2][80];
    char qualification_id[160];
} v200_memory_pair_profile;

typedef struct v200_allocation_info {
    uint32_t abi_version;
    size_t struct_size;
    uint64_t allocation_id;
    CUdeviceptr base;
    size_t requested_bytes;
    size_t mapped_bytes;
    size_t owner_bytes[2];
    uint32_t extent_count;
    uint32_t flags;
} v200_allocation_info;

typedef struct v200_memory_usage_info {
    uint32_t abi_version;
    size_t struct_size;
    uint64_t owner_used_bytes[2];
    uint64_t owner_budget_bytes[2];
    uint64_t live_allocation_count;
    uint64_t live_reservation_count;
    uint64_t cleanup_pending_count;
} v200_memory_usage_info;

/* Configure once before allocations. Reconfiguration is rejected while the
 * ledger contains live or cleanup-pending objects. */
CUresult v200_memory_configure(const v200_memory_pair_profile *profile,
                               const v200_memory_driver_api *driver);
int v200_memory_is_configured(void);
CUresult v200_memory_get_profile(v200_memory_pair_profile *out);

/* Allocate one canonical VA range. `preferred_device` must identify one of
 * the two configured physical owners. The ordinary form uses default_owner. */
CUresult v200_memory_allocate_on(CUdevice preferred_device, size_t bytes,
                                 size_t alignment, void **device_ptr);
CUresult v200_memory_allocate(size_t bytes, size_t alignment, void **device_ptr);
CUresult v200_memory_release(void *device_ptr);
CUresult v200_memory_query(const void *ptr, v200_allocation_info *out);
int v200_memory_owns_pointer(const void *ptr);
/* The frontdoor attaches a sealed logical-stream prefix to managed memory so
 * teardown can wait before unmapping or returning its backing to CUDA. A
 * nonzero stream ID coalesces a newer dominating prefix over the older prefix
 * for that logical stream. ID zero preserves distinct prefixes when a caller
 * cannot prove stream identity and dominance. */
CUresult v200_memory_retain_use(const void *ptr, uint64_t logical_stream_id,
                               uint64_t token_slot, uint64_t token_generation);
/* For opaque kernel launches whose pointer arguments cannot be inspected,
 * conservatively retain every live managed allocation to the sealed prefix.
 * A newer prefix replaces the older retained prefix for the same logical
 * stream; stream ID zero disables this coalescing. */
CUresult v200_memory_retain_all_use(uint64_t logical_stream_id,
                                    uint64_t token_slot,
                                   uint64_t token_generation);
/* Bracket every asynchronous dispatch that could touch managed memory. Begin
 * before forwarding; end after the prefix has been retained. Free and unmap
 * wait for outstanding leases. A zero lease means the manager was unconfigured. */
CUresult v200_memory_begin_submission(uint64_t *lease);
CUresult v200_memory_end_submission(uint64_t lease, int completion_recorded);
CUresult v200_memory_get_usage(v200_memory_usage_info *out);
CUresult v200_memory_retry_cleanup(void);

/* Driver VMM gateway used by the frontdoor. Compound handles are synthetic
 * tokens understood only by this gateway and are never passed to libcuda. */
CUresult v200_memory_vmm_get_granularity(size_t *granularity,
                                         const CUmemAllocationProp *prop,
                                         CUmemAllocationGranularity_flags option);
CUresult v200_memory_vmm_address_reserve(CUdeviceptr *ptr, size_t size,
                                         size_t alignment, CUdeviceptr requested,
                                         unsigned long long flags);
CUresult v200_memory_vmm_address_free(CUdeviceptr ptr, size_t size);
CUresult v200_memory_vmm_create(CUmemGenericAllocationHandle *handle,
                                size_t size, const CUmemAllocationProp *prop,
                                unsigned long long flags);
CUresult v200_memory_vmm_map(CUdeviceptr ptr, size_t size, size_t offset,
                             CUmemGenericAllocationHandle handle,
                             unsigned long long flags);
CUresult v200_memory_vmm_set_access(CUdeviceptr ptr, size_t size,
                                    const CUmemAccessDesc *desc, size_t count);
CUresult v200_memory_vmm_unmap(CUdeviceptr ptr, size_t size);
CUresult v200_memory_vmm_release(CUmemGenericAllocationHandle handle);
int v200_memory_owns_handle(CUmemGenericAllocationHandle handle);

#ifdef __cplusplus
}
#endif

#endif
