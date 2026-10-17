#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#ifdef V200_FAKE_DRIVER
#ifdef V200_FAKE_LAZY_DRIVER
CUresult CUDAAPI cuInit(unsigned int flags) {
    (void)flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
    if (count == 0) return CUDA_ERROR_INVALID_VALUE;
    *count = 2;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    if (device == 0 || ordinal < 0 || ordinal >= 2) return CUDA_ERROR_INVALID_VALUE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice device) {
    if (uuid == 0 || device < 0 || device >= 2) return CUDA_ERROR_INVALID_VALUE;
    memset(uuid, 0, sizeof(*uuid));
    uuid->bytes[15] = (char)(device == 0 ? 0xa1 : 0xb2);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice device) {
    return cuDeviceGetUuid_v2(uuid, device);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *value, CUdevice_attribute attribute,
                                      CUdevice device) {
    if (value == 0 || device < 0 || device >= 2) return CUDA_ERROR_INVALID_VALUE;
    if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) *value = 7;
    else if (attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) *value = 0;
    else *value = 0;
    return CUDA_SUCCESS;
}
#else
CUresult CUDAAPI cuDeviceGetCount(int *count) {
    if (count == 0) return CUDA_ERROR_INVALID_VALUE;
    *count = 37;
    return CUDA_SUCCESS;
}
#endif
#else
cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    if (count == 0) return cudaErrorInvalidValue;
#ifdef V200_FAKE_LAZY_RUNTIME
    static void *driver_handle = 0;
    if (driver_handle == 0) {
        const char *path = getenv("V200_FAKE_DRIVER_PATH");
        if (path == 0 || path[0] == '\0') return cudaErrorInitializationError;
        driver_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (driver_handle == 0) return cudaErrorInitializationError;
    }
    *count = 2;
#else
    *count = 53;
#endif
    return cudaSuccess;
}
#endif
