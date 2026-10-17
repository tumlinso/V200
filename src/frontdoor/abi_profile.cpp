#include "v200/frontdoor/abi_profile.hpp"

namespace v200::frontdoor {
namespace {

constexpr ApiEntry entries[]{
    {"cuDeviceGetCount", 2000, ApiDisposition::wrapped, false},
    {"cuDeviceGet", 2000, ApiDisposition::wrapped, false},
    {"cuDeviceTotalMem_v2", 2000, ApiDisposition::wrapped, false},
    {"cuDeviceGetAttribute", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cuDeviceGetUuid_v2", 9020, ApiDisposition::native_anchor_passthrough, false},
    {"cuGetErrorString", 4000, ApiDisposition::native_anchor_passthrough, false},
    {"cuGetProcAddress_v2", 11030, ApiDisposition::wrapped, false},
    {"cuMemAddressReserve", 10020, ApiDisposition::wrapped, false},
    {"cuMemAddressFree", 10020, ApiDisposition::wrapped, false},
    {"cuMemCreate", 10020, ApiDisposition::wrapped, false},
    {"cuMemGetInfo_v2", 2000, ApiDisposition::wrapped, false},
    {"cuMemGetAllocationGranularity", 10020, ApiDisposition::wrapped, false},
    {"cuMemMap", 10020, ApiDisposition::wrapped, false},
    {"cuMemRelease", 10020, ApiDisposition::wrapped, false},
    {"cuMemSetAccess", 10020, ApiDisposition::wrapped, false},
    {"cuMemUnmap", 10020, ApiDisposition::wrapped, false},
    {"cuMemcpyDtoDAsync_v2", 2000, ApiDisposition::wrapped, true},
    {"cuMemcpyDtoHAsync_v2", 2000, ApiDisposition::wrapped, true},
    {"cuMemcpyHtoDAsync_v2", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD8Async", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD16Async", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD32Async", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD2D8Async", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD2D16Async", 2000, ApiDisposition::wrapped, true},
    {"cuMemsetD2D32Async", 2000, ApiDisposition::wrapped, true},
    {"cuLaunchKernel", 4000, ApiDisposition::wrapped, true},
    {"cuStreamSynchronize", 2000, ApiDisposition::wrapped, true},
    {"cuStreamWaitEvent", 2000, ApiDisposition::wrapped, true},
    {"cudaGetDeviceCount", 2000, ApiDisposition::wrapped, false},
    {"cudaGetDevice", 2000, ApiDisposition::wrapped, false},
    {"cudaSetDevice", 2000, ApiDisposition::wrapped, false},
    {"cudaGetDeviceProperties_v2", 2000, ApiDisposition::wrapped, false},
    {"cudaDeviceGetAttribute", 2000, ApiDisposition::wrapped, false},
    {"cudaDeviceGetPCIBusId", 2000, ApiDisposition::wrapped, false},
    {"cudaDeviceCanAccessPeer", 2000, ApiDisposition::wrapped, false},
    {"cudaDeviceEnablePeerAccess", 2000, ApiDisposition::wrapped, false},
    {"cudaDeviceSynchronize", 2000, ApiDisposition::wrapped, false},
    {"cudaMalloc", 2000, ApiDisposition::wrapped, false},
    {"cudaFree", 2000, ApiDisposition::wrapped, false},
    {"cudaMallocManaged", 6000, ApiDisposition::unsupported, false},
    {"cudaMemGetInfo", 2000, ApiDisposition::wrapped, false},
    {"cudaMemset", 2000, ApiDisposition::wrapped, false},
    {"cudaMemcpyAsync", 2000, ApiDisposition::wrapped, true},
    {"cudaMemcpy2DAsync", 2000, ApiDisposition::wrapped, true},
    {"cudaMemcpyPeerAsync", 4000, ApiDisposition::wrapped, true},
    {"cudaMemsetAsync", 2000, ApiDisposition::wrapped, true},
    {"cudaStreamCreateWithFlags", 4000, ApiDisposition::wrapped, false},
    {"cudaStreamCreateWithPriority", 5000, ApiDisposition::wrapped, false},
    {"cudaStreamDestroy", 2000, ApiDisposition::wrapped, false},
    {"cudaStreamSynchronize", 2000, ApiDisposition::wrapped, true},
    {"cudaStreamWaitEvent", 2000, ApiDisposition::wrapped, true},
    {"cudaEventRecord", 2000, ApiDisposition::wrapped, true},
    {"cudaEventSynchronize", 2000, ApiDisposition::wrapped, false},
    {"cudaEventCreateWithFlags", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaEventDestroy", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaLaunchKernel", 7000, ApiDisposition::wrapped, true},
    {"cudaLaunchKernelExC", 12000, ApiDisposition::wrapped, true},
    {"cudaLaunchCooperativeKernel", 9000, ApiDisposition::wrapped, true},
    {"cudaGraphInstantiate", 10000, ApiDisposition::wrapped, false},
    {"cudaGraphLaunch", 10000, ApiDisposition::wrapped, true},
    {"cudaGraphExecUpdate", 10000, ApiDisposition::wrapped, false},
    {"cudaGraphDestroy", 10000, ApiDisposition::wrapped, false},
    {"cudaGraphExecDestroy", 10000, ApiDisposition::wrapped, false},
    {"cudaStreamBeginCapture", 10000, ApiDisposition::wrapped, false},
    {"cudaStreamEndCapture", 10000, ApiDisposition::wrapped, false},
    {"cudaStreamIsCapturing", 10000, ApiDisposition::wrapped, false},
    {"cudaHostAlloc", 4000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaMallocHost", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaFreeHost", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaHostRegister", 4000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaHostUnregister", 4000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaHostGetDevicePointer", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaFuncGetAttributes", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaFuncSetAttribute", 7000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", 9000,
     ApiDisposition::native_anchor_passthrough, false},
    {"cudaSetDeviceFlags", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaGetErrorString", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaGetLastError", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaPeekAtLastError", 2000, ApiDisposition::native_anchor_passthrough, false},
    {"cudaGetDriverEntryPoint", 11030, ApiDisposition::wrapped, true},
    {"cudaGetDriverEntryPointByVersion", 12000, ApiDisposition::wrapped, true},
};

} // namespace

const ApiEntry *find_api(std::string_view name) noexcept {
    for (const auto &entry : entries) {
        if (entry.name == name) return &entry;
    }
    return nullptr;
}

ApiDisposition classify_api(std::string_view name) noexcept {
    const auto *entry = find_api(name);
    return entry ? entry->disposition : ApiDisposition::unknown;
}

bool version_available(const ApiEntry &entry, int cuda_version) noexcept {
    return cuda_version >= entry.minimum_cuda_version;
}

} // namespace v200::frontdoor
