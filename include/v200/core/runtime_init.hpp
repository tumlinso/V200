#pragma once

#include "v200/execution/owner_execution_c.h"
#include "v200/memory/manager.hpp"

#include <cuda.h>

#include <cstdint>
#include <memory>

namespace v200::core {

struct RuntimeInitInputs {
  const char *ordered_uuid_pair{};
  const char *b01_evidence_id{};
  const char *owner0_budget_mib{};
  const char *owner1_budget_mib{};
};

struct RuntimeInitApi {
  using Init = CUresult(CUDAAPI *)(unsigned int);
  using DeviceCount = CUresult(CUDAAPI *)(int *);
  using DeviceGet = CUresult(CUDAAPI *)(CUdevice *, int);
  using DeviceGetUuid = CUresult(CUDAAPI *)(CUuuid *, CUdevice);
  using DeviceGetAttribute = CUresult(CUDAAPI *)(int *, CUdevice_attribute,
                                                  CUdevice);
  using PrimaryContextRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using PrimaryContextRelease = CUresult(CUDAAPI *)(CUdevice);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);
  using AllocationGranularity = CUresult(CUDAAPI *)(
      size_t *, const CUmemAllocationProp *,
      CUmemAllocationGranularity_flags);

  Init init{};
  DeviceCount device_count{};
  DeviceGet device_get{};
  DeviceGetUuid device_get_uuid{};
  DeviceGetAttribute device_get_attribute{};
  PrimaryContextRetain primary_context_retain{};
  PrimaryContextRelease primary_context_release{};
  ContextPush context_push{};
  ContextPop context_pop{};
  AllocationGranularity allocation_granularity{};
  v200_execution_driver_api_v1 execution{};
  v200_memory_driver_api memory{};
  int (*register_owner)(v200_execution_handle *, uint64_t, uint32_t, void *){};
  int (*attach_frontdoor)(v200_execution_handle *, uint64_t, uint32_t){};
  void (*detach_frontdoor)(v200_execution_handle *){};
};

// Initializes the execution and VMM ledgers from an exact ordered UUID pair
// and pinned B01 evidence ID. The supplied API and table user_data must remain
// valid until this object is destroyed. The process initializer intentionally
// has process lifetime and no CUDA teardown constructor.
class RuntimeInitializer final {
 public:
  explicit RuntimeInitializer(memory::MemoryManager &manager);
  ~RuntimeInitializer();

  RuntimeInitializer(const RuntimeInitializer &) = delete;
  RuntimeInitializer &operator=(const RuntimeInitializer &) = delete;

  CUresult initialize(const RuntimeInitInputs &inputs,
                      const RuntimeInitApi &api) noexcept;
  [[nodiscard]] bool initialized() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace v200::core
