#include <cuda.h>
#include <cuda_runtime_api.h>
#include "v200/frontdoor/runtime.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {
int fail(const char *message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}
}

int main(int argc, char **argv) {
    if (argc != 3) return fail("expected exact fake libcuda and libcudart paths");
    if (v200_frontdoor_mode() != 0) return fail("test must start in transparent native mode");
    void *driver = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void *runtime = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (driver == nullptr || runtime == nullptr) return fail("fake CUDA DSO failed to load locally");

    using Resolver = void *(*)(const char *);
    auto resolve_driver = reinterpret_cast<Resolver>(dlsym(RTLD_DEFAULT,
        "v200_frontdoor_resolve_driver_symbol"));
    auto resolve_runtime = reinterpret_cast<Resolver>(dlsym(RTLD_DEFAULT,
        "v200_frontdoor_resolve_runtime_symbol"));
    if (resolve_driver == nullptr || resolve_runtime == nullptr)
        return fail("frontdoor native DSO resolver is not exported");

    void *fake_driver_count = dlsym(driver, "cuDeviceGetCount");
    void *fake_runtime_count = dlsym(runtime, "cudaGetDeviceCount");
    if (fake_driver_count == nullptr || fake_runtime_count == nullptr)
        return fail("fake CUDA symbols are missing");
    if (resolve_driver("cuDeviceGetCount") != fake_driver_count)
        return fail("driver resolver did not preserve the RTLD_LOCAL libcuda identity");
    if (resolve_runtime("cudaGetDeviceCount") != fake_runtime_count)
        return fail("runtime resolver did not preserve the RTLD_LOCAL libcudart identity");

    auto frontdoor_driver_count = reinterpret_cast<CUresult (CUDAAPI *)(int *)>(
        dlsym(RTLD_DEFAULT, "cuDeviceGetCount"));
    auto frontdoor_runtime_count = reinterpret_cast<cudaError_t (CUDARTAPI *)(int *)>(
        dlsym(RTLD_DEFAULT, "cudaGetDeviceCount"));
    if (frontdoor_driver_count == nullptr || frontdoor_runtime_count == nullptr)
        return fail("frontdoor wrappers are missing from the global scope");
    int count = 0;
    if (frontdoor_driver_count(&count) != CUDA_SUCCESS || count != 37)
        return fail("Driver wrapper failed to dispatch to RTLD_LOCAL libcuda");
    const cudaError_t runtime_result = frontdoor_runtime_count(&count);
    if (runtime_result != cudaSuccess || count != 53) {
        std::fprintf(stderr, "runtime result=%d count=%d\n", static_cast<int>(runtime_result), count);
        return fail("Runtime wrapper failed to dispatch to RTLD_LOCAL libcudart");
    }

    std::puts("RTLD_LOCAL CUDA closure checks passed without CUDA initialization");
    return 0;
}
