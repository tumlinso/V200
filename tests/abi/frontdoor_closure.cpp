#include "v200/memory/api.h"
#include "v200/frontdoor/runtime.h"

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
int fail(const char *message, int code = 1) {
    std::fprintf(stderr, "FAIL: %s (code=%d)\n", message, code);
    return 1;
}

bool frontdoor_address(void *address) {
    Dl_info info{};
    return address != nullptr && dladdr(address, &info) != 0 && info.dli_fname != nullptr &&
           std::strstr(info.dli_fname, "libv200_frontdoor") != nullptr;
}
}

int main(int argc, char **argv) {
    if (v200_frontdoor_mode() != 0 && argc == 1)
        return fail("default closure test must remain in native mode");
    using GetCount = cudaError_t (CUDARTAPI *)(int *);
    auto get_count = reinterpret_cast<GetCount>(dlsym(RTLD_DEFAULT, "cudaGetDeviceCount"));
    auto driver_proc = reinterpret_cast<CUresult (CUDAAPI *)(const char *, void **, int,
        cuuint64_t, CUdriverProcAddressQueryResult *)>(dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2"));
    if (!frontdoor_address(reinterpret_cast<void *>(get_count)) ||
        !frontdoor_address(reinterpret_cast<void *>(driver_proc)))
        return fail("preload closure did not bind runtime and versioned Driver wrappers");

    if (argc == 1) {
        std::puts("frontdoor export closure passed; no CUDA API was called");
        return 0;
    }
    if (argc != 2 || std::strcmp(argv[1], "--execute") != 0)
        return fail("supported option is --execute (run only inside the CUDA controller session)");

    int logical_count = 0;
    cudaError_t runtime_result = get_count(&logical_count);
    if (runtime_result != cudaSuccess || logical_count != 1)
        return fail("runtime presentation must report exactly one logical device", runtime_result);
    using GetProperties = cudaError_t (CUDARTAPI *)(cudaDeviceProp *, int);
    auto get_properties = reinterpret_cast<GetProperties>(
        dlsym(RTLD_DEFAULT, "cudaGetDeviceProperties_v2"));
    if (!frontdoor_address(reinterpret_cast<void *>(get_properties)))
        return fail("device properties wrapper is missing");
    auto set_device = reinterpret_cast<cudaError_t (CUDARTAPI *)(int)>(
        dlsym(RTLD_DEFAULT, "cudaSetDevice"));
    if (set_device == nullptr) return fail("device selection wrapper is missing");
    cudaDeviceProp properties{};
    runtime_result = get_properties(&properties, 0);
    if (runtime_result != cudaSuccess) return fail("anchor properties query failed", runtime_result);
    runtime_result = set_device(0);
    if (runtime_result != cudaSuccess) return fail("logical device selection failed", runtime_result);

    using GetMemoryQuery = CUresult (*)(const void *, v200_allocation_info *);
    auto memory_query = reinterpret_cast<GetMemoryQuery>(dlsym(RTLD_DEFAULT, "v200_memory_query"));
    auto cuda_alloc = reinterpret_cast<cudaError_t (CUDARTAPI *)(void **, size_t)>(
        dlsym(RTLD_DEFAULT, "cudaMalloc"));
    auto cuda_free = reinterpret_cast<cudaError_t (CUDARTAPI *)(void *)>(
        dlsym(RTLD_DEFAULT, "cudaFree"));
    auto memcpy_async = reinterpret_cast<cudaError_t (CUDARTAPI *)(void *, const void *, size_t,
        enum cudaMemcpyKind, cudaStream_t)>(dlsym(RTLD_DEFAULT, "cudaMemcpyAsync"));
    auto synchronize = reinterpret_cast<cudaError_t (CUDARTAPI *)(void)>(
        dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize"));
    if (cuda_alloc == nullptr || cuda_free == nullptr || memcpy_async == nullptr ||
        synchronize == nullptr || memory_query == nullptr)
        return fail("memory/execution integration DSOs are not loaded in the process closure");

    std::array<unsigned, 8> input{{3, 5, 8, 13, 21, 34, 55, 89}};
    std::array<unsigned, 8> output{};
    void *device = nullptr;
    runtime_result = cuda_alloc(&device, sizeof(input));
    if (runtime_result != cudaSuccess) return fail("canonical allocation failed", runtime_result);
    v200_allocation_info allocation{};
    allocation.abi_version = V200_MEMORY_ABI_VERSION;
    allocation.struct_size = sizeof(allocation);
    const CUresult queried = memory_query(device, &allocation);
    if (queried != CUDA_SUCCESS || allocation.requested_bytes < sizeof(input)) {
        (void)cuda_free(device);
        return fail("canonical allocation query failed", queried);
    }
    runtime_result = memcpy_async(device, input.data(), sizeof(input), cudaMemcpyHostToDevice, nullptr);
    if (runtime_result == cudaSuccess) runtime_result = synchronize();
    if (runtime_result == cudaSuccess)
        runtime_result = memcpy_async(output.data(), device, sizeof(output), cudaMemcpyDeviceToHost, nullptr);
    if (runtime_result == cudaSuccess) runtime_result = synchronize();
    if (runtime_result != cudaSuccess) {
        (void)cuda_free(device);
        return fail("anchor async copy and completion failed", runtime_result);
    }
    if (output != input) {
        (void)cuda_free(device);
        return fail("round-trip copy mismatch");
    }
    runtime_result = cuda_free(device);
    if (runtime_result != cudaSuccess) return fail("canonical free failed", runtime_result);

    using DriverCount = CUresult (CUDAAPI *)(int *);
    using DriverProc = CUresult (CUDAAPI *)(const char *, void **, int, cuuint64_t,
                                            CUdriverProcAddressQueryResult *);
    auto driver_count = reinterpret_cast<DriverCount>(dlsym(RTLD_DEFAULT, "cuDeviceGetCount"));
    if (driver_count == nullptr) return fail("Driver device count wrapper is missing");
    CUresult driver_result = driver_count(&logical_count);
    if (driver_result != CUDA_SUCCESS || logical_count != 1)
        return fail("Driver presentation must report exactly one logical device", driver_result);
    auto get_proc = reinterpret_cast<DriverProc>(dlsym(RTLD_DEFAULT, "cuGetProcAddress_v2"));
    if (get_proc == nullptr) return fail("versioned cuGetProcAddress_v2 wrapper is missing");
    void *granularity_pointer = nullptr;
    CUdriverProcAddressQueryResult query_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    driver_result = get_proc("cuMemGetAllocationGranularity", &granularity_pointer, 12000,
        CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query_status);
    if (driver_result != CUDA_SUCCESS || query_status != CU_GET_PROC_ADDRESS_SUCCESS ||
        !frontdoor_address(granularity_pointer))
        return fail("versioned proc-address query did not return the VMM wrapper", driver_result);

    using ProfileGetter = CUresult (*)(v200_memory_pair_profile *);
    auto get_profile = reinterpret_cast<ProfileGetter>(dlsym(RTLD_DEFAULT, "v200_memory_get_profile"));
    if (get_profile == nullptr) return fail("pair memory profile API is missing");
    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    driver_result = get_profile(&profile);
    if (driver_result != CUDA_SUCCESS) return fail("pair profile query failed", driver_result);

    using GetGranularity = CUresult (CUDAAPI *)(size_t *, const CUmemAllocationProp *,
                                                CUmemAllocationGranularity_flags);
    auto get_granularity = reinterpret_cast<GetGranularity>(granularity_pointer);
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = profile.owner_device[0];
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t granularity = 0;
    driver_result = get_granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (driver_result != CUDA_SUCCESS || granularity == 0)
        return fail("VMM granularity query failed", driver_result);

    using AddressReserve = CUresult (CUDAAPI *)(CUdeviceptr *, size_t, size_t, CUdeviceptr,
                                                unsigned long long);
    using AddressFree = CUresult (CUDAAPI *)(CUdeviceptr, size_t);
    using Create = CUresult (CUDAAPI *)(CUmemGenericAllocationHandle *, size_t,
                                        const CUmemAllocationProp *, unsigned long long);
    using Map = CUresult (CUDAAPI *)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                                     unsigned long long);
    using SetAccess = CUresult (CUDAAPI *)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    using Unmap = CUresult (CUDAAPI *)(CUdeviceptr, size_t);
    using Release = CUresult (CUDAAPI *)(CUmemGenericAllocationHandle);
    auto reserve = reinterpret_cast<AddressReserve>(dlsym(RTLD_DEFAULT, "cuMemAddressReserve"));
    auto address_free = reinterpret_cast<AddressFree>(dlsym(RTLD_DEFAULT, "cuMemAddressFree"));
    auto create = reinterpret_cast<Create>(dlsym(RTLD_DEFAULT, "cuMemCreate"));
    auto map = reinterpret_cast<Map>(dlsym(RTLD_DEFAULT, "cuMemMap"));
    auto set_access = reinterpret_cast<SetAccess>(dlsym(RTLD_DEFAULT, "cuMemSetAccess"));
    auto unmap = reinterpret_cast<Unmap>(dlsym(RTLD_DEFAULT, "cuMemUnmap"));
    auto release = reinterpret_cast<Release>(dlsym(RTLD_DEFAULT, "cuMemRelease"));
    if (reserve == nullptr || address_free == nullptr || create == nullptr || map == nullptr ||
        set_access == nullptr || unmap == nullptr || release == nullptr)
        return fail("direct VMM wrapper closure is incomplete");

    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handle{};
    bool reserved = false, created = false, mapped = false;
    driver_result = reserve(&address, granularity, granularity, 0, 0);
    if (driver_result == CUDA_SUCCESS) {
        reserved = true;
        driver_result = create(&handle, granularity, &prop, 0);
    }
    if (driver_result == CUDA_SUCCESS) {
        created = true;
        driver_result = map(address, granularity, 0, handle, 0);
    }
    if (driver_result == CUDA_SUCCESS) {
        mapped = true;
        std::array<CUmemAccessDesc, 2> access{};
        for (size_t i = 0; i < access.size(); ++i) {
            access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access[i].location.id = profile.owner_device[i];
            access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        driver_result = set_access(address, granularity, access.data(), access.size());
    }
    if (mapped) {
        const CUresult cleanup = unmap(address, granularity);
        if (driver_result == CUDA_SUCCESS) driver_result = cleanup;
    }
    if (reserved) {
        const CUresult cleanup = address_free(address, granularity);
        if (driver_result == CUDA_SUCCESS) driver_result = cleanup;
    }
    if (created) {
        const CUresult cleanup = release(handle);
        if (driver_result == CUDA_SUCCESS) driver_result = cleanup;
    }
    if (driver_result != CUDA_SUCCESS) return fail("direct pair VMM lifecycle failed", driver_result);

    using Snapshot = void (*)(const char *);
    auto snapshot = reinterpret_cast<Snapshot>(dlsym(RTLD_DEFAULT, "v200_frontdoor_snapshot"));
    if (snapshot != nullptr) snapshot("abi_closure_complete");
    std::puts("controller-backed frontdoor CUDA closure checks passed");
    return 0;
}
