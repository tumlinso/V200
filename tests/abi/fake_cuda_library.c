#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef V200_FAKE_DRIVER
CUresult CUDAAPI cuDeviceGetCount(int *count) {
    if (count == 0) return CUDA_ERROR_INVALID_VALUE;
    *count = 37;
    return CUDA_SUCCESS;
}
#else
cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    if (count == 0) return cudaErrorInvalidValue;
    *count = 53;
    return cudaSuccess;
}
#endif
