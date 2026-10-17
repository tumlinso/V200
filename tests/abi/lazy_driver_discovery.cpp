#include "v200/frontdoor/runtime.h"

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int fail(const char *message, int code = 1) {
    std::fprintf(stderr, "FAIL: %s (code=%d)\n", message, code);
    return 1;
}

bool frontdoor_address(void *address) {
    Dl_info info{};
    return address != nullptr && dladdr(address, &info) != 0 &&
           info.dli_fname != nullptr &&
           std::strstr(info.dli_fname, "libv200_frontdoor") != nullptr;
}
}

int main(int argc, char **argv) {
    if (argc != 3) return fail("expected fake libcudart and libcuda paths");
    if (setenv("V200_MODE", "pair", 1) != 0 ||
        setenv("V200_DEVICE_UUIDS",
               "000000000000000000000000000000a1,"
               "000000000000000000000000000000b2", 1) != 0 ||
        setenv("V200_FAKE_DRIVER_PATH", argv[2], 1) != 0)
        return fail("failed to set fake lazy-driver environment");
    if (v200_frontdoor_mode() != 2)
        return fail("frontdoor did not enter pair mode");

    void *driver_before = dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD);
    if (driver_before != nullptr) {
        (void)dlclose(driver_before);
        return fail("fake libcuda was loaded before the first Runtime query");
    }
    void *runtime = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (runtime == nullptr) return fail("fake lazy libcudart failed to load");

    using GetCount = cudaError_t (CUDARTAPI *)(int *);
    auto get_count = reinterpret_cast<GetCount>(
        dlsym(RTLD_DEFAULT, "cudaGetDeviceCount"));
    if (!frontdoor_address(reinterpret_cast<void *>(get_count)))
        return fail("Runtime query did not resolve to the V200 frontdoor");

    int logical_count = 0;
    const cudaError_t status = get_count(&logical_count);
    if (status != cudaSuccess)
        return fail("cold Runtime query failed to load/discover native libcuda", status);
    if (logical_count != 1)
        return fail("two fake physical devices must produce one logical pair",
                    logical_count);

    void *driver_after = dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD);
    if (driver_after == nullptr)
        return fail("native Runtime query did not lazily load fake libcuda");
    (void)dlclose(driver_after);
    (void)dlclose(runtime);
    std::puts("cold Runtime query initialized libcuda before V200 UUID discovery");
    return 0;
}
