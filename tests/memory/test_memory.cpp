#include "v200/memory/manager.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct FakeDriver {
    struct AccessCall {
        CUdeviceptr ptr = 0;
        size_t size = 0;
        std::vector<CUmemAccessDesc> descriptors;
    };
    CUdeviceptr next_address = 0x10000000;
    CUmemGenericAllocationHandle next_handle = 1;
    std::map<CUdeviceptr, size_t> reservations;
    std::map<CUdeviceptr, size_t> mappings;
    std::set<CUmemGenericAllocationHandle> handles;
    std::vector<size_t> owner_created_bytes{0, 0};
    std::vector<AccessCall> access_calls;
    size_t map_calls = 0;
    size_t unmap_calls = 0;
    size_t release_calls = 0;
    size_t create_calls = 0;
    size_t fail_map_call = 0;
    size_t fail_unmap_call = 0;
    size_t fail_release_call = 0;
    CUresult map_failure = CUDA_ERROR_OUT_OF_MEMORY;
    CUresult unmap_failure = CUDA_ERROR_UNKNOWN;
    CUresult release_failure = CUDA_ERROR_UNKNOWN;
};

size_t index_of(CUdevice device) {
    assert(device == 0 || device == 1);
    return static_cast<size_t>(device);
}

CUresult CUDAAPI fake_reserve(void *data, CUdeviceptr *ptr, size_t size,
                              size_t alignment, CUdeviceptr requested,
                              unsigned long long flags) {
    auto &driver = *static_cast<FakeDriver *>(data);
    if (ptr == nullptr || size == 0 || requested != 0 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    size_t align = alignment == 0 ? 1 : alignment;
    CUdeviceptr value = driver.next_address;
    const CUdeviceptr rem = value % align;
    if (rem != 0) value += align - rem;
    *ptr = value;
    driver.reservations.emplace(value, size);
    driver.next_address = value + size + 0x1000;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_address_free(void *data, CUdeviceptr ptr, size_t size) {
    auto &driver = *static_cast<FakeDriver *>(data);
    auto it = driver.reservations.find(ptr);
    if (it == driver.reservations.end() || it->second != size) return CUDA_ERROR_INVALID_VALUE;
    driver.reservations.erase(it);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_create(void *data, CUmemGenericAllocationHandle *handle,
                             size_t size, const CUmemAllocationProp *prop,
                             unsigned long long flags) {
    auto &driver = *static_cast<FakeDriver *>(data);
    if (handle == nullptr || prop == nullptr || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    *handle = driver.next_handle++;
    driver.handles.insert(*handle);
    driver.owner_created_bytes[index_of(prop->location.id)] += size;
    ++driver.create_calls;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_release(void *data, CUmemGenericAllocationHandle handle) {
    auto &driver = *static_cast<FakeDriver *>(data);
    ++driver.release_calls;
    if (driver.fail_release_call != 0 && driver.release_calls == driver.fail_release_call) {
        return driver.release_failure;
    }
    if (driver.handles.erase(handle) != 1) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_map(void *data, CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
    auto &driver = *static_cast<FakeDriver *>(data);
    ++driver.map_calls;
    if (driver.fail_map_call != 0 && driver.map_calls == driver.fail_map_call) return driver.map_failure;
    if (offset != 0 || flags != 0 || driver.handles.count(handle) == 0) return CUDA_ERROR_INVALID_VALUE;
    for (const auto &mapping : driver.mappings) {
        const CUdeviceptr a_end = ptr + size;
        const CUdeviceptr b_end = mapping.first + mapping.second;
        if (ptr < b_end && mapping.first < a_end) return CUDA_ERROR_INVALID_VALUE;
    }
    driver.mappings.emplace(ptr, size);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_unmap(void *data, CUdeviceptr ptr, size_t size) {
    auto &driver = *static_cast<FakeDriver *>(data);
    ++driver.unmap_calls;
    if (driver.fail_unmap_call != 0 && driver.unmap_calls == driver.fail_unmap_call) return driver.unmap_failure;
    CUdeviceptr cursor = ptr;
    const CUdeviceptr end = ptr + size;
    while (cursor < end) {
        auto it = driver.mappings.find(cursor);
        if (it == driver.mappings.end() || it->second > end - cursor) return CUDA_ERROR_INVALID_VALUE;
        cursor += it->second;
        driver.mappings.erase(it);
    }
    return cursor == end ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI fake_set_access(void *data, CUdeviceptr ptr, size_t size,
                                 const CUmemAccessDesc *desc, size_t count) {
    if (desc == nullptr || count == 0) return CUDA_ERROR_INVALID_VALUE;
    auto &driver = *static_cast<FakeDriver *>(data);
    FakeDriver::AccessCall call;
    call.ptr = ptr;
    call.size = size;
    call.descriptors.assign(desc, desc + count);
    driver.access_calls.push_back(std::move(call));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI fake_granularity(void *data, size_t *granularity,
                                  const CUmemAllocationProp *prop,
                                  CUmemAllocationGranularity_flags option) {
    (void)data;
    if (granularity == nullptr || prop == nullptr ||
        (option != CU_MEM_ALLOC_GRANULARITY_MINIMUM && option != CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = prop->location.id == 0 ? 4 : 6;
    return CUDA_SUCCESS;
}

struct FakeCompletion final : v200::execution::CompletionProvider {
    bool complete = false;
    bool wait_fails = false;
    int retains = 0;
    int releases = 0;
    int waits = 0;
    int fail_retain_call = 0;
    int fail_release_call = 0;
    v200::execution::Status retain(v200::execution::CompletionToken) noexcept override {
        ++retains;
        if (fail_retain_call != 0 && retains == fail_retain_call) {
            return {v200::execution::ErrorCode::driver_failure, CUDA_ERROR_UNKNOWN};
        }
        return {};
    }
    v200::execution::QueryResult is_complete(v200::execution::CompletionToken) noexcept override {
        return {{}, complete};
    }
    v200::execution::Status wait(v200::execution::CompletionToken) noexcept override {
        ++waits;
        if (wait_fails) return {v200::execution::ErrorCode::driver_failure, CUDA_ERROR_UNKNOWN};
        complete = true;
        return {};
    }
    v200::execution::Status release(v200::execution::CompletionToken) noexcept override {
        ++releases;
        if (fail_release_call != 0 && releases == fail_release_call) {
            return {v200::execution::ErrorCode::driver_failure, CUDA_ERROR_UNKNOWN};
        }
        return {};
    }
};

v200_memory_pair_profile make_profile(bool qualified = true) {
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    profile.owner_device[0] = 0;
    profile.owner_device[1] = 1;
    profile.owner_granularity[0] = 4;
    profile.owner_granularity[1] = 6;
    profile.owner_budget_bytes[0] = 20;
    profile.owner_budget_bytes[1] = 20;
    profile.default_owner = 0;
    profile.pair_access_qualified = qualified ? 1u : 0u;
    std::snprintf(profile.owner_uuid[0], sizeof(profile.owner_uuid[0]), "GPU-test-a");
    std::snprintf(profile.owner_uuid[1], sizeof(profile.owner_uuid[1]), "GPU-test-b");
    std::snprintf(profile.qualification_id, sizeof(profile.qualification_id), "B01-test-receipt");
    return profile;
}

v200_memory_driver_api make_driver(FakeDriver &fake) {
    v200_memory_driver_api driver{};
    driver.abi_version = V200_MEMORY_ABI_VERSION;
    driver.struct_size = sizeof(driver);
    driver.user_data = &fake;
    driver.mem_address_reserve = fake_reserve;
    driver.mem_address_free = fake_address_free;
    driver.mem_create = fake_create;
    driver.mem_release = fake_release;
    driver.mem_map = fake_map;
    driver.mem_unmap = fake_unmap;
    driver.mem_set_access = fake_set_access;
    driver.mem_get_granularity = fake_granularity;
    return driver;
}

void test_planner() {
    auto profile = make_profile();
    size_t used[2] = {0, 0};
    auto plan = v200::memory::plan_extents(24, 0, profile, used);
    assert(plan.status == CUDA_SUCCESS);
    assert(plan.extents.size() == 2);
    assert(plan.extents[0].owner == 0 && plan.extents[0].size == 12);
    assert(plan.extents[1].owner == 1 && plan.extents[1].logical_offset == 12 && plan.extents[1].size == 12);

    profile.pair_access_qualified = 0;
    plan = v200::memory::plan_extents(24, 0, profile, used);
    assert(plan.status == CUDA_ERROR_NOT_SUPPORTED && plan.extents.empty());
    plan = v200::memory::plan_extents(5, 0, profile, used);
    assert(plan.status == CUDA_ERROR_INVALID_VALUE);

    profile.pair_access_qualified = 1;
    profile.owner_budget_bytes[0] = 4;
    profile.owner_budget_bytes[1] = 6;
    plan = v200::memory::plan_extents(8, 0, profile, used);
    assert(plan.status == CUDA_ERROR_OUT_OF_MEMORY && plan.extents.empty());

    profile.owner_granularity[0] = 5000000003ULL;
    profile.owner_granularity[1] = 5000000009ULL;
    profile.owner_budget_bytes[0] = 10000000006ULL;
    profile.owner_budget_bytes[1] = 10000000018ULL;
    used[0] = 5000000003ULL;
    used[1] = 0;
    plan = v200::memory::plan_extents(10000000006ULL, 0, profile, used);
    assert(plan.status == CUDA_ERROR_INVALID_VALUE);
}

void test_compound_vmm_and_lifetime() {
    FakeDriver fake;
    auto profile = make_profile();
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);

    CUdeviceptr reservation = 0;
    assert(manager.vmm_address_reserve(&reservation, 32, 0, 0, 0) == CUDA_SUCCESS);
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    CUmemGenericAllocationHandle handle = 0;
    assert(manager.vmm_create(&handle, 24, &prop, 0) == CUDA_SUCCESS);
    assert(handle != 0 && manager.owns_handle(handle));
    assert(fake.owner_created_bytes[0] == 12 && fake.owner_created_bytes[1] == 12);
    assert(manager.vmm_map(reservation, 24, 0, handle, 0) == CUDA_SUCCESS);
    CUdeviceptr alias_reservation = 0;
    assert(manager.vmm_address_reserve(&alias_reservation, 24, 0, 0, 0) == CUDA_SUCCESS);
    assert(manager.vmm_map(alias_reservation, 24, 0, handle, 0) == CUDA_ERROR_NOT_SUPPORTED);
    assert(manager.vmm_address_free(alias_reservation, 24) == CUDA_SUCCESS);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    assert(manager.vmm_set_access(reservation, 24, &access, 1) == CUDA_SUCCESS);
    assert(fake.access_calls.size() == 1);
    assert(fake.access_calls.back().descriptors.size() == 2);
    assert(fake.access_calls.back().descriptors[0].location.id == 0 &&
           fake.access_calls.back().descriptors[1].location.id == 1);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    assert(manager.vmm_set_access(reservation, 24, &access, 1) == CUDA_SUCCESS);
    assert(fake.access_calls.back().descriptors.size() == 2 &&
           fake.access_calls.back().descriptors[0].flags == CU_MEM_ACCESS_FLAGS_PROT_NONE &&
           fake.access_calls.back().descriptors[1].flags == CU_MEM_ACCESS_FLAGS_PROT_NONE);

    FakeCompletion completion;
    manager.set_completion_provider(&completion);
    const v200::execution::CompletionToken token{1, 1};
    assert(manager.retain_use(reservation + 8, token) == CUDA_SUCCESS);
    assert(manager.vmm_release(handle) == CUDA_SUCCESS);
    assert(!manager.owns_handle(handle));
    v200_memory_usage_info usage{};
    usage.struct_size = sizeof(usage);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 12 && usage.owner_used_bytes[1] == 12);
    assert(manager.vmm_unmap(reservation, 24) == CUDA_SUCCESS);
    assert(completion.waits == 1 && completion.releases == 1);
    v200_allocation_info info{};
    info.struct_size = sizeof(info);
    const void *old_address = reinterpret_cast<const void *>(static_cast<uintptr_t>(reservation));
    assert(manager.query(old_address, &info) == CUDA_ERROR_INVALID_VALUE);
    assert(!manager.owns_pointer(old_address));
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 0 && usage.owner_used_bytes[1] == 0);
    assert(manager.vmm_address_free(reservation, 32) == CUDA_SUCCESS);
}

void test_single_extent_allocation_has_pair_access() {
    FakeDriver fake;
    auto profile = make_profile();
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);

    void *pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &pointer) == CUDA_SUCCESS);
    assert(pointer != nullptr && fake.access_calls.size() == 1);
    const auto &access = fake.access_calls.front().descriptors;
    assert(access.size() == 2);
    assert(access[0].location.id == 0 && access[1].location.id == 1);
    assert(access[0].flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE &&
           access[1].flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    v200_allocation_info info{};
    info.struct_size = sizeof(info);
    assert(manager.query(pointer, &info) == CUDA_SUCCESS);
    assert(info.extent_count == 1 && info.owner_bytes[0] == 12 && info.owner_bytes[1] == 0);
    assert((info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_KNOWN) != 0);
    assert((info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_1) == 0);
    assert(manager.release(pointer) == CUDA_SUCCESS);
}

void test_first_extent_owner_flags() {
    FakeDriver fake;
    auto profile = make_profile();
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);

    struct Case {
        CUdevice owner;
        size_t requested_bytes;
        uint32_t expected_extents;
    };
    const Case cases[]{{0, 8, 1}, {1, 8, 1}, {0, 24, 2}, {1, 24, 2}};
    for (const Case &test : cases) {
        void *pointer = nullptr;
        assert(manager.allocate(test.owner, test.requested_bytes, 0, &pointer) == CUDA_SUCCESS);
        v200_allocation_info info{};
        info.struct_size = sizeof(info);
        assert(manager.query(pointer, &info) == CUDA_SUCCESS);
        assert(info.extent_count == test.expected_extents);
        assert((info.flags & V200_ALLOCATION_INFO_FLAG_ALLOCATION_API) != 0);
        assert((info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_KNOWN) != 0);
        const bool first_owner_is_one =
            (info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_1) != 0;
        assert(first_owner_is_one == (test.owner == 1));
        assert(info.owner_bytes[test.owner] != 0);
        assert(manager.release(pointer) == CUDA_SUCCESS);
    }
}

void test_single_extent_vmm_anchor_access_expands_to_pair() {
    FakeDriver fake;
    auto profile = make_profile();
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);

    CUdeviceptr reservation = 0;
    assert(manager.vmm_address_reserve(&reservation, 12, 0, 0, 0) == CUDA_SUCCESS);
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 1;
    CUmemGenericAllocationHandle handle = 0;
    assert(manager.vmm_create(&handle, 12, &prop, 0) == CUDA_SUCCESS);
    assert(manager.vmm_map(reservation, 12, 0, handle, 0) == CUDA_SUCCESS);

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    v200_allocation_info info{};
    info.struct_size = sizeof(info);
    assert(manager.query(reinterpret_cast<void *>(static_cast<uintptr_t>(reservation)), &info) == CUDA_SUCCESS);
    assert((info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_KNOWN) != 0);
    assert((info.flags & V200_ALLOCATION_INFO_FLAG_FIRST_EXTENT_OWNER_1) != 0);
    assert((info.flags & V200_ALLOCATION_INFO_FLAG_ALLOCATION_API) == 0);
    assert(manager.vmm_set_access(reservation, 12, &access, 1) == CUDA_SUCCESS);
    assert(fake.access_calls.size() == 1);
    assert(fake.access_calls.back().descriptors.size() == 2);
    assert(fake.access_calls.back().descriptors[0].location.id == 0 &&
           fake.access_calls.back().descriptors[1].location.id == 1);

    assert(manager.vmm_release(handle) == CUDA_SUCCESS);
    assert(manager.vmm_unmap(reservation, 12) == CUDA_SUCCESS);
    assert(manager.vmm_address_free(reservation, 12) == CUDA_SUCCESS);
}

void test_allocate_rollback_and_retry() {
    FakeDriver fake;
    auto profile = make_profile();
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    fake.fail_map_call = 2;
    void *pointer = reinterpret_cast<void *>(1);
    assert(manager.allocate(0, 24, 0, &pointer) == CUDA_ERROR_OUT_OF_MEMORY);
    assert(pointer == nullptr);
    assert(fake.mappings.empty() && fake.reservations.empty() && fake.handles.empty());
    v200_memory_usage_info usage{};
    usage.struct_size = sizeof(usage);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 0 && usage.owner_used_bytes[1] == 0);

    fake.fail_map_call = 0;
    assert(manager.allocate(0, 24, 0, &pointer) == CUDA_SUCCESS);
    assert(pointer != nullptr);
    v200_allocation_info info{};
    info.struct_size = sizeof(info);
    auto interior = static_cast<void *>(static_cast<unsigned char *>(pointer) + 20);
    assert(manager.query(interior, &info) == CUDA_SUCCESS);
    assert(info.requested_bytes == 24 && info.mapped_bytes == 24 && info.extent_count == 2);

    fake.fail_unmap_call = fake.unmap_calls + 1;
    assert(manager.release(pointer) == CUDA_ERROR_UNKNOWN);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.cleanup_pending_count == 1);
    fake.fail_unmap_call = 0;
    assert(manager.retry_cleanup() == CUDA_SUCCESS);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 0 && usage.owner_used_bytes[1] == 0);
    assert(fake.mappings.empty() && fake.reservations.empty() && fake.handles.empty());
}

void test_completion_failure_keeps_accounting() {
    FakeDriver fake;
    auto profile = make_profile(false);
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    FakeCompletion completion;
    completion.wait_fails = true;
    manager.set_completion_provider(&completion);
    void *pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &pointer) == CUDA_SUCCESS);
    assert(manager.retain_use(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer)), {7, 2}) == CUDA_SUCCESS);
    assert(manager.release(pointer) == CUDA_ERROR_UNKNOWN);
    v200_memory_usage_info usage{};
    usage.struct_size = sizeof(usage);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 8);
    completion.wait_fails = false;
    assert(manager.release(pointer) == CUDA_SUCCESS);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS && usage.owner_used_bytes[0] == 0);
}

void test_retain_all_and_fail_stop() {
    {
        FakeDriver fake;
        auto profile = make_profile(false);
        auto driver = make_driver(fake);
        v200::memory::MemoryManager manager;
        assert(manager.configure(profile, driver) == CUDA_SUCCESS);
        FakeCompletion completion;
        manager.set_completion_provider(&completion);
        void *first = nullptr;
        void *second = nullptr;
        assert(manager.allocate(0, 8, 0, &first) == CUDA_SUCCESS);
        assert(manager.allocate(0, 8, 0, &second) == CUDA_SUCCESS);
        assert(manager.retain_all_use({9, 3}) == CUDA_SUCCESS);
        assert(completion.retains == 2);
        assert(manager.release(first) == CUDA_SUCCESS);
        assert(completion.waits == 1 && completion.releases == 1);
        assert(manager.release(second) == CUDA_SUCCESS);
        assert(completion.waits == 1 && completion.releases == 2);
    }
    {
        FakeDriver fake;
        auto profile = make_profile(false);
        auto driver = make_driver(fake);
        v200::memory::MemoryManager manager;
        assert(manager.configure(profile, driver) == CUDA_SUCCESS);
        FakeCompletion completion;
        completion.fail_retain_call = 2;
        manager.set_completion_provider(&completion);
        void *first = nullptr;
        void *second = nullptr;
        assert(manager.allocate(0, 8, 0, &first) == CUDA_SUCCESS);
        assert(manager.allocate(0, 8, 0, &second) == CUDA_SUCCESS);
        assert(manager.retain_all_use({10, 4}) == CUDA_ERROR_UNKNOWN);
        assert(manager.release(first) == CUDA_ERROR_NOT_READY);
        assert(manager.release(second) == CUDA_ERROR_NOT_READY);
        v200_memory_usage_info usage{};
        usage.struct_size = sizeof(usage);
        assert(manager.get_usage(&usage) == CUDA_SUCCESS);
        assert(usage.owner_used_bytes[0] == 16 && usage.cleanup_pending_count == 2);
        assert(fake.mappings.size() == 2 && fake.handles.size() == 2);
    }
}

void test_stream_prefix_retention_is_coalesced() {
    FakeDriver fake;
    auto profile = make_profile(false);
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    FakeCompletion completion;
    manager.set_completion_provider(&completion);
    void *pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &pointer) == CUDA_SUCCESS);
    const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer));

    assert(manager.retain_use(address, 41, {31, 1}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 41, {31, 2}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 42, {32, 1}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 0, {33, 1}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 0, {33, 1}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 0, {33, 2}) == CUDA_SUCCESS);

    /* Stream 41 advanced in place, stream 42 is independent, and stream 0
     * deliberately keeps distinct prefixes without unbounded duplicate refs. */
    assert(completion.retains == 5);
    assert(completion.releases == 1);
    assert(manager.release(pointer) == CUDA_SUCCESS);
    assert(completion.releases == 5);
}

void test_stream_coalesce_release_failure_fail_stops() {
    FakeDriver fake;
    auto profile = make_profile(false);
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    FakeCompletion completion;
    completion.fail_release_call = 1;
    manager.set_completion_provider(&completion);
    void *pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &pointer) == CUDA_SUCCESS);
    const auto address = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer));

    assert(manager.retain_use(address, 77, {51, 1}) == CUDA_SUCCESS);
    assert(manager.retain_use(address, 77, {51, 2}) == CUDA_ERROR_UNKNOWN);
    assert(completion.retains == 2 && completion.releases == 1);
    assert(manager.release(pointer) == CUDA_ERROR_NOT_READY);
    v200_memory_usage_info usage{};
    usage.struct_size = sizeof(usage);
    assert(manager.get_usage(&usage) == CUDA_SUCCESS);
    assert(usage.owner_used_bytes[0] == 8 && usage.cleanup_pending_count == 1);
    assert(fake.mappings.size() == 1 && fake.handles.size() == 1);
}

void test_unqualified_owner_local_access() {
    FakeDriver fake;
    auto profile = make_profile(false);
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    CUdeviceptr reservation = 0;
    assert(manager.vmm_address_reserve(&reservation, 12, 0, 0, 0) == CUDA_SUCCESS);
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 1;
    CUmemGenericAllocationHandle handle = 0;
    assert(manager.vmm_create(&handle, 12, &prop, 0) == CUDA_SUCCESS);
    assert(manager.vmm_map(reservation, 12, 0, handle, 0) == CUDA_SUCCESS);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 1;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    assert(manager.vmm_set_access(reservation, 12, &access, 1) == CUDA_SUCCESS);
    access.location.id = 0;
    assert(manager.vmm_set_access(reservation, 12, &access, 1) == CUDA_ERROR_NOT_SUPPORTED);
    assert(manager.vmm_release(handle) == CUDA_SUCCESS);
    assert(manager.vmm_unmap(reservation, 12) == CUDA_SUCCESS);
    assert(manager.vmm_address_free(reservation, 12) == CUDA_SUCCESS);
}

void test_submission_lease_closes_retain_race() {
    FakeDriver fake;
    auto profile = make_profile(false);
    auto driver = make_driver(fake);
    v200::memory::MemoryManager manager;
    assert(manager.configure(profile, driver) == CUDA_SUCCESS);
    FakeCompletion completion;
    manager.set_completion_provider(&completion);
    void *pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &pointer) == CUDA_SUCCESS);

    uint64_t lease = 0;
    assert(manager.begin_submission(&lease) == CUDA_SUCCESS && lease != 0);
    assert(manager.retain_all_use({22, 5}) == CUDA_SUCCESS);

    std::atomic<bool> released{false};
    std::thread freer([&] {
        assert(manager.release(pointer) == CUDA_SUCCESS);
        released.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    assert(!released.load(std::memory_order_acquire));
    assert(manager.end_submission(lease, true) == CUDA_SUCCESS);
    freer.join();
    assert(released.load(std::memory_order_acquire));
    assert(completion.waits == 1 && completion.releases == 1);

    void *unsafe_pointer = nullptr;
    assert(manager.allocate(0, 8, 0, &unsafe_pointer) == CUDA_SUCCESS);
    assert(manager.begin_submission(&lease) == CUDA_SUCCESS && lease != 0);
    assert(manager.end_submission(lease, false) == CUDA_ERROR_NOT_READY);
    assert(manager.release(unsafe_pointer) == CUDA_ERROR_NOT_READY);
    assert(fake.mappings.size() == 1 && fake.handles.size() == 1);
}

} // namespace

int main() {
    test_planner();
    test_single_extent_allocation_has_pair_access();
    test_first_extent_owner_flags();
    test_single_extent_vmm_anchor_access_expands_to_pair();
    test_compound_vmm_and_lifetime();
    test_allocate_rollback_and_retry();
    test_completion_failure_keeps_accounting();
    test_retain_all_and_fail_stop();
    test_stream_prefix_retention_is_coalesced();
    test_stream_coalesce_release_failure_fail_stops();
    test_unqualified_owner_local_access();
    test_submission_lease_closes_retain_race();
    std::cout << "v200 memory tests passed\n";
}
