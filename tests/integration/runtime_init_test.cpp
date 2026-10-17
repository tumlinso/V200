#include "v200/core/runtime_init.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

constexpr char kPair[] =
    "GPU-21131915-1488-23af-38dd-1743ae1f5cc8,GPU-d745da9c-7649-8334-41f1-483afa6f3206";
constexpr char kB01[] = "8426eab5-278d-4687-ab7d-f5ec937e46ec";
constexpr std::uint64_t kMiB = 1024ull * 1024ull;

struct Fake {
  std::array<CUdevice, 2> devices{{20, 10}};  // Enumeration order differs from owner order.
  int device_count{2};
  CUdevice fail_retain_device{-1};
  int attach_error{V200_EXECUTION_OK};
  CUresult context_pop_result{CUDA_SUCCESS};
  CUresult granularity_result{CUDA_SUCCESS};
  bool return_wrong_popped_context{false};
  int init_calls{};
  int retained_count{};
  int release_count{};
  int attach_count{};
  int detach_count{};
  int event_sync_count{};
  v200_execution_handle *attached_handle{};
  CUcontext current{reinterpret_cast<CUcontext>(0x777)};
  std::vector<CUcontext> context_stack;
  std::unordered_map<CUevent, bool> events;
  std::uintptr_t next_event{0x1000};
  std::uintptr_t next_memory_handle{0x2000};
  std::size_t mem_granularity{2 * 1024 * 1024};
  CUdeviceptr next_address{0x10000000};
};

Fake *g_fake = nullptr;
Fake &fake(void *data = nullptr) { return data == nullptr ? *g_fake : *static_cast<Fake *>(data); }

CUresult CUDAAPI fake_init(unsigned int) {
  ++g_fake->init_calls;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_device_count(int *count) {
  *count = g_fake->device_count;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_device_get(CUdevice *device, int ordinal) {
  if (ordinal < 0 || ordinal >= g_fake->device_count) return CUDA_ERROR_INVALID_DEVICE;
  *device = g_fake->devices[static_cast<std::size_t>(ordinal)];
  return CUDA_SUCCESS;
}

void write_uuid(CUuuid *uuid, CUdevice device) {
  static constexpr unsigned char owner0[16] = {
      0x21, 0x13, 0x19, 0x15, 0x14, 0x88, 0x23, 0xaf,
      0x38, 0xdd, 0x17, 0x43, 0xae, 0x1f, 0x5c, 0xc8};
  static constexpr unsigned char owner1[16] = {
      0xd7, 0x45, 0xda, 0x9c, 0x76, 0x49, 0x83, 0x34,
      0x41, 0xf1, 0x48, 0x3a, 0xfa, 0x6f, 0x32, 0x06};
  std::memcpy(uuid->bytes, device == 10 ? owner0 : owner1, sizeof(uuid->bytes));
}

CUresult CUDAAPI fake_device_uuid(CUuuid *uuid, CUdevice device) {
  write_uuid(uuid, device);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_attribute(int *value, CUdevice_attribute attribute, CUdevice) {
  if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) *value = 7;
  else if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) *value = 0;
  else if (attribute == CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) *value = 1;
  else return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_retain(CUcontext *context, CUdevice device) {
  if (device == g_fake->fail_retain_device) return CUDA_ERROR_UNKNOWN;
  *context = reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0xa00 + device));
  ++g_fake->retained_count;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_release(CUdevice) {
  ++g_fake->release_count;
  --g_fake->retained_count;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_push(CUcontext context) {
  g_fake->context_stack.push_back(g_fake->current);
  g_fake->current = context;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_pop(CUcontext *context) {
  if (g_fake->context_pop_result != CUDA_SUCCESS) return g_fake->context_pop_result;
  if (g_fake->context_stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
  *context = g_fake->current;
  g_fake->current = g_fake->context_stack.back();
  g_fake->context_stack.pop_back();
  if (g_fake->return_wrong_popped_context) {
    *context = reinterpret_cast<CUcontext>(0x999);
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_granularity(size_t *value, const CUmemAllocationProp *,
                                  CUmemAllocationGranularity_flags) {
  if (g_fake->granularity_result != CUDA_SUCCESS) return g_fake->granularity_result;
  *value = g_fake->mem_granularity;
  return CUDA_SUCCESS;
}

int fake_exec_push(void *data, void *context) {
  auto &f = fake(data);
  f.context_stack.push_back(f.current);
  f.current = static_cast<CUcontext>(context);
  return CUDA_SUCCESS;
}
int fake_exec_pop(void *data, void **context) {
  (void)data;
  CUcontext popped = nullptr;
  const auto status = fake_pop(&popped);
  if (context != nullptr) *context = popped;
  return status;
}
int fake_event_create(void *data, void **event, unsigned) {
  auto &f = fake(data);
  auto handle = reinterpret_cast<CUevent>(f.next_event++);
  f.events.emplace(handle, false);
  *event = handle;
  return CUDA_SUCCESS;
}
int fake_event_destroy(void *data, void *event) {
  fake(data).events.erase(static_cast<CUevent>(event));
  return CUDA_SUCCESS;
}
int fake_event_record(void *, void *, void *) { return CUDA_SUCCESS; }
int fake_event_query(void *data, void *event) {
  auto &f = fake(data);
  const auto found = f.events.find(static_cast<CUevent>(event));
  if (found == f.events.end()) return CUDA_ERROR_INVALID_HANDLE;
  return found->second ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
int fake_event_synchronize(void *data, void *event) {
  auto &f = fake(data);
  const auto found = f.events.find(static_cast<CUevent>(event));
  if (found == f.events.end()) return CUDA_ERROR_INVALID_HANDLE;
  found->second = true;
  ++f.event_sync_count;
  return CUDA_SUCCESS;
}
int fake_stream_synchronize(void *, void *) { return CUDA_SUCCESS; }
int fake_stream_wait_event(void *, void *, void *, unsigned) { return CUDA_SUCCESS; }
int fake_register_owner(v200_execution_handle *handle, uint64_t device,
                        uint32_t owner, void *context) {
  return v200_execution_register_owner(handle, device, owner, context);
}
int fake_attach(v200_execution_handle *handle, uint64_t device, uint32_t owner) {
  ++g_fake->attach_count;
  g_fake->attached_handle = handle;
  if (g_fake->attach_error != V200_EXECUTION_OK) return g_fake->attach_error;
  int status = v200_execution_register_stream(handle, device, 1, owner,
                                              reinterpret_cast<void *>(1));
  if (status != V200_EXECUTION_OK) return status;
  return v200_execution_register_stream(handle, device, 2, owner,
                                       reinterpret_cast<void *>(2));
}
void fake_detach(v200_execution_handle *) { ++g_fake->detach_count; }

CUresult CUDAAPI fake_address_reserve(void *data, CUdeviceptr *ptr, size_t size,
                                      size_t, CUdeviceptr requested,
                                      unsigned long long) {
  auto &f = fake(data);
  *ptr = requested != 0 ? requested : f.next_address;
  f.next_address += size + 0x1000;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_address_free(void *, CUdeviceptr, size_t) { return CUDA_SUCCESS; }
CUresult CUDAAPI fake_mem_create(void *data, CUmemGenericAllocationHandle *handle,
                                 size_t, const CUmemAllocationProp *, unsigned long long) {
  *handle = static_cast<CUmemGenericAllocationHandle>(fake(data).next_memory_handle++);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_mem_release(void *, CUmemGenericAllocationHandle) { return CUDA_SUCCESS; }
CUresult CUDAAPI fake_mem_map(void *, CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle, unsigned long long) {
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_mem_unmap(void *, CUdeviceptr, size_t) { return CUDA_SUCCESS; }
CUresult CUDAAPI fake_mem_set_access(void *, CUdeviceptr, size_t,
                                     const CUmemAccessDesc *, size_t) {
  return CUDA_SUCCESS;
}
CUresult CUDAAPI fake_mem_get_granularity(void *data, size_t *value,
                                          const CUmemAllocationProp *,
                                          CUmemAllocationGranularity_flags) {
  *value = fake(data).mem_granularity;
  return CUDA_SUCCESS;
}

v200::core::RuntimeInitApi make_api(Fake *state) {
  v200::core::RuntimeInitApi api{};
  api.init = fake_init;
  api.device_count = fake_device_count;
  api.device_get = fake_device_get;
  api.device_get_uuid = fake_device_uuid;
  api.device_get_attribute = fake_attribute;
  api.primary_context_retain = fake_retain;
  api.primary_context_release = fake_release;
  api.context_push = fake_push;
  api.context_pop = fake_pop;
  api.allocation_granularity = fake_granularity;
  api.execution.abi_version = V200_EXECUTION_ABI_VERSION;
  api.execution.struct_size = sizeof(api.execution);
  api.execution.user_data = state;
  api.execution.ctx_push_current = fake_exec_push;
  api.execution.ctx_pop_current = fake_exec_pop;
  api.execution.event_create = fake_event_create;
  api.execution.event_destroy = fake_event_destroy;
  api.execution.event_record = fake_event_record;
  api.execution.event_query = fake_event_query;
  api.execution.event_synchronize = fake_event_synchronize;
  api.execution.stream_synchronize = fake_stream_synchronize;
  api.execution.stream_wait_event = fake_stream_wait_event;
  api.memory.abi_version = V200_MEMORY_ABI_VERSION;
  api.memory.struct_size = sizeof(api.memory);
  api.memory.user_data = state;
  api.memory.mem_address_reserve = fake_address_reserve;
  api.memory.mem_address_free = fake_address_free;
  api.memory.mem_create = fake_mem_create;
  api.memory.mem_release = fake_mem_release;
  api.memory.mem_map = fake_mem_map;
  api.memory.mem_unmap = fake_mem_unmap;
  api.memory.mem_set_access = fake_mem_set_access;
  api.memory.mem_get_granularity = fake_mem_get_granularity;
  api.register_owner = fake_register_owner;
  api.attach_frontdoor = fake_attach;
  api.detach_frontdoor = fake_detach;
  return api;
}

v200::core::RuntimeInitInputs inputs(const char *budget0 = "64",
                                     const char *budget1 = "64") {
  return {kPair, kB01, budget0, budget1};
}

int enqueue_ok(void *, void *) { return CUDA_SUCCESS; }

void test_rejects_wrong_proof_before_cuda() {
  Fake state{};
  g_fake = &state;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    auto invalid = inputs();
    invalid.b01_evidence_id = "unqualified";
    assert(initializer.initialize(invalid, api) == CUDA_ERROR_INVALID_VALUE);
    assert(initializer.initialize(inputs(), api) == CUDA_ERROR_INVALID_VALUE);
    assert(state.init_calls == 0);
    assert(state.retained_count == 0);
    assert(!manager.is_configured());
  }
}

void test_rejects_bad_budget_before_cuda() {
  Fake state{};
  g_fake = &state;
  v200::memory::MemoryManager manager;
  v200::core::RuntimeInitializer initializer(manager);
  auto api = make_api(&state);
  assert(initializer.initialize(inputs("0", "64"), api) == CUDA_ERROR_INVALID_VALUE);
  assert(state.init_calls == 0);
}

void test_releases_first_context_if_second_retain_fails() {
  Fake state{};
  state.fail_retain_device = 20;
  g_fake = &state;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    assert(initializer.initialize(inputs(), api) == CUDA_ERROR_UNKNOWN);
    assert(state.retained_count == 0);
    assert(state.release_count == 1);
    assert(state.attach_count == 0);
    assert(!manager.is_configured());
  }
}

void test_attach_failure_rolls_back_contexts_and_execution() {
  Fake state{};
  state.attach_error = V200_EXECUTION_UNAVAILABLE;
  g_fake = &state;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    assert(initializer.initialize(inputs(), api) == CUDA_ERROR_NOT_SUPPORTED);
    assert(state.retained_count == 0);
    assert(state.release_count == 2);
    assert(state.attach_count == 1);
    assert(state.detach_count == 0);
    assert(!manager.is_configured());
  }
}

void test_uncertain_context_restore_preserves_primary_contexts() {
  static Fake state{};
  state.return_wrong_popped_context = true;
  g_fake = &state;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    assert(initializer.initialize(inputs(), api) == CUDA_ERROR_INVALID_CONTEXT);
    assert(state.current == reinterpret_cast<CUcontext>(0x777));
    assert(state.context_stack.empty());
    assert(state.retained_count == 2);
    assert(state.release_count == 0);
    assert(!manager.is_configured());
  }
  // The initializer intentionally retains/leaks its context graph after a
  // restore mismatch; make this test its last use of the persistent fake.
  state.return_wrong_popped_context = false;
}

void test_pop_error_dominates_granularity_error_and_preserves_contexts() {
  static Fake state{};
  state.context_pop_result = CUDA_ERROR_UNKNOWN;
  state.granularity_result = CUDA_ERROR_INVALID_VALUE;
  g_fake = &state;
  const int retained_before = state.retained_count;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    assert(initializer.initialize(inputs(), api) == CUDA_ERROR_UNKNOWN);
    assert(state.retained_count == retained_before + 2);
    assert(state.release_count == 0);
    assert(!manager.is_configured());
  }
  // Model the test thread's reset after recording that the adapter deliberately
  // kept the context reference because its stack could not be restored.
  state.current = reinterpret_cast<CUcontext>(0x777);
  state.context_stack.clear();
  state.context_pop_result = CUDA_SUCCESS;
  state.granularity_result = CUDA_SUCCESS;
}

void test_success_maps_order_budgets_contexts_and_completion_bridge() {
  Fake state{};
  g_fake = &state;
  v200::memory::MemoryManager manager;
  {
    v200::core::RuntimeInitializer initializer(manager);
    auto api = make_api(&state);
    assert(initializer.initialize(inputs(), api) == CUDA_SUCCESS);
    assert(initializer.initialized());
    assert(initializer.initialize(inputs("1", "1"), api) == CUDA_SUCCESS);
    assert(state.init_calls == 1);
    assert(state.retained_count == 2);
    assert(state.attach_count == 1);
    assert(state.current == reinterpret_cast<CUcontext>(0x777));

    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    assert(manager.get_profile(&profile) == CUDA_SUCCESS);
    assert(profile.owner_device[0] == 10 && profile.owner_device[1] == 20);
    assert(profile.default_owner == 10);
    assert(profile.owner_budget_bytes[0] == 64 * kMiB);
    assert(profile.owner_budget_bytes[1] == 64 * kMiB);
    assert(profile.owner_granularity[0] == 2 * 1024 * 1024);
    assert(profile.owner_granularity[1] == 2 * 1024 * 1024);
    assert(profile.pair_access_qualified == 1);
    assert(std::strcmp(profile.qualification_id, kB01) == 0);
    if (std::strcmp(profile.owner_uuid[0], "GPU-21131915-1488-23af-38dd-1743ae1f5cc8") != 0 ||
        std::strcmp(profile.owner_uuid[1], "GPU-d745da9c-7649-8334-41f1-483afa6f3206") != 0) {
      std::fprintf(stderr, "profile UUIDs: <%s> <%s>\n", profile.owner_uuid[0],
                   profile.owner_uuid[1]);
      assert(false);
    }

    void *allocation = nullptr;
    assert(manager.allocate(10, 4096, 0, &allocation) == CUDA_SUCCESS);
    assert(allocation != nullptr);
    int forwarded = CUDA_ERROR_UNKNOWN;
    assert(v200_execution_forward_native(
               state.attached_handle, 1, 1, 0, reinterpret_cast<void *>(1),
               enqueue_ok, nullptr, &forwarded) == V200_EXECUTION_OK);
    assert(forwarded == CUDA_SUCCESS);
    v200_completion_token token{};
    assert(v200_execution_current_token(state.attached_handle, 1, 1, &token) ==
           V200_EXECUTION_OK);
    assert(manager.retain_use(reinterpret_cast<CUdeviceptr>(allocation), 1,
                              {token.slot, token.generation}) == CUDA_SUCCESS);
    assert(v200_execution_release(state.attached_handle, token) == V200_EXECUTION_OK);
    assert(manager.release(allocation) == CUDA_SUCCESS);
    assert(state.event_sync_count >= 1);
    v200_memory_usage_info usage{};
    usage.abi_version = V200_MEMORY_ABI_VERSION;
    usage.struct_size = sizeof(usage);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 0 && usage.owner_used_bytes[1] == 0);
    assert(usage.live_allocation_count == 0);
    assert(usage.live_reservation_count == 0);
    assert(usage.cleanup_pending_count == 0);
  }
  assert(state.retained_count == 0);
  assert(state.release_count == 2);
  assert(state.detach_count == 1);
  assert(state.current == reinterpret_cast<CUcontext>(0x777));
}

}  // namespace

int main() {
  test_rejects_wrong_proof_before_cuda();
  test_rejects_bad_budget_before_cuda();
  test_releases_first_context_if_second_retain_fails();
  test_attach_failure_rolls_back_contexts_and_execution();
  test_success_maps_order_budgets_contexts_and_completion_bridge();
  test_uncertain_context_restore_preserves_primary_contexts();
  test_pop_error_dominates_granularity_error_and_preserves_contexts();
  std::puts("runtime initializer host tests passed");
  return 0;
}
