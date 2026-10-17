#include <cuda_runtime_api.h>
#include <cuda.h>

#include <dlfcn.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

constexpr std::size_t kBytes = 64 * 1024;

bool check(cudaError_t status, const char *operation);

struct DriverVmmApi {
  using GetGranularity = CUresult(CUDAAPI *)(
      std::size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
  using AddressReserve = CUresult(CUDAAPI *)(
      CUdeviceptr *, std::size_t, std::size_t, CUdeviceptr, unsigned long long);
  using Create = CUresult(CUDAAPI *)(
      CUmemGenericAllocationHandle *, std::size_t, const CUmemAllocationProp *,
      unsigned long long);
  using Map = CUresult(CUDAAPI *)(
      CUdeviceptr, std::size_t, std::size_t, CUmemGenericAllocationHandle,
      unsigned long long);
  using SetAccess = CUresult(CUDAAPI *)(
      CUdeviceptr, std::size_t, const CUmemAccessDesc *, std::size_t);
  using Unmap = CUresult(CUDAAPI *)(CUdeviceptr, std::size_t);
  using AddressFree = CUresult(CUDAAPI *)(CUdeviceptr, std::size_t);
  using Release = CUresult(CUDAAPI *)(CUmemGenericAllocationHandle);

  GetGranularity get_granularity{};
  AddressReserve address_reserve{};
  Create create{};
  Map map{};
  SetAccess set_access{};
  Unmap unmap{};
  AddressFree address_free{};
  Release release{};

  bool ready() const {
    return get_granularity != nullptr && address_reserve != nullptr &&
           create != nullptr && map != nullptr && set_access != nullptr &&
           unmap != nullptr && address_free != nullptr && release != nullptr;
  }
};

template <typename Function>
Function driver_symbol(const char *name) {
  return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

DriverVmmApi resolve_vmm_api() {
  DriverVmmApi api{};
  // This is deliberately called only after cudaGetDeviceCount initializes the
  // CUDA runtime and loads the interposed Driver API surface.
  api.get_granularity = driver_symbol<DriverVmmApi::GetGranularity>(
      "cuMemGetAllocationGranularity");
  api.address_reserve = driver_symbol<DriverVmmApi::AddressReserve>(
      "cuMemAddressReserve");
  api.create = driver_symbol<DriverVmmApi::Create>("cuMemCreate");
  api.map = driver_symbol<DriverVmmApi::Map>("cuMemMap");
  api.set_access = driver_symbol<DriverVmmApi::SetAccess>("cuMemSetAccess");
  api.unmap = driver_symbol<DriverVmmApi::Unmap>("cuMemUnmap");
  api.address_free = driver_symbol<DriverVmmApi::AddressFree>("cuMemAddressFree");
  api.release = driver_symbol<DriverVmmApi::Release>("cuMemRelease");
  return api;
}

bool check_driver(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) return true;
  std::fprintf(stderr, "%s failed: CUDA Driver error %d\n", operation,
               static_cast<int>(status));
  return false;
}

bool vmm_round_trip(int logical_device, const DriverVmmApi &driver) {
  if (!check(cudaSetDevice(logical_device), "cudaSetDevice(VMM)")) return false;

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = logical_device;
  std::size_t granularity = 0;
  if (!check_driver(driver.get_granularity(
                       &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                   "cuMemGetAllocationGranularity")) return false;
  if (granularity == 0) {
    std::fprintf(stderr, "logical CUDA%d returned zero VMM granularity\n", logical_device);
    return false;
  }

  CUdeviceptr address = 0;
  CUmemGenericAllocationHandle handle{};
  bool reserved = false;
  bool created = false;
  bool mapped = false;
  bool ok = check_driver(driver.address_reserve(&address, granularity, 0, 0, 0),
                         "cuMemAddressReserve(VMM)");
  reserved = ok;
  if (ok) {
    ok = check_driver(driver.create(&handle, granularity, &prop, 0), "cuMemCreate(VMM)");
    created = ok;
  }
  if (ok) {
    ok = check_driver(driver.map(address, granularity, 0, handle, 0), "cuMemMap(VMM)");
    mapped = ok;
  }
  if (ok) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = logical_device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ok = check_driver(driver.set_access(address, granularity, &access, 1),
                      "cuMemSetAccess(VMM)");
  }
  if (ok) {
    ok = check_driver(driver.release(handle), "cuMemRelease(VMM)");
    if (ok) created = false;
  }
  if (ok) {
    auto *pointer = reinterpret_cast<void *>(static_cast<std::uintptr_t>(address));
    ok = check(cudaMemset(pointer, 0x5a, granularity), "cudaMemset(VMM)") &&
         check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(VMM)");
    if (ok) {
      std::vector<unsigned char> observed(granularity);
      ok = check(cudaMemcpy(observed.data(), pointer, observed.size(),
                            cudaMemcpyDeviceToHost), "cudaMemcpy(VMM readback)");
      if (ok) {
        for (unsigned char value : observed) {
          if (value != 0x5a) {
            std::fprintf(stderr, "logical CUDA%d VMM readback mismatch\n", logical_device);
            ok = false;
            break;
          }
        }
      }
    }
  }

  bool cleanup_ok = true;
  if (mapped) cleanup_ok = check_driver(driver.unmap(address, granularity),
                                       "cuMemUnmap(VMM)") && cleanup_ok;
  if (created) cleanup_ok = check_driver(driver.release(handle),
                                         "cuMemRelease(VMM cleanup)") && cleanup_ok;
  if (reserved) cleanup_ok = check_driver(driver.address_free(address, granularity),
                                          "cuMemAddressFree(VMM)") && cleanup_ok;
  if (!ok || !cleanup_ok) return false;
  std::printf("VMM_OK device=%d bytes=%zu\n", logical_device, granularity);
  return true;
}

enum class StreamMode { source_event_wait, destination_stream, default_stream };

const char *mode_name(StreamMode mode) {
  switch (mode) {
    case StreamMode::source_event_wait: return "source_stream_event_wait";
    case StreamMode::destination_stream: return "destination_stream";
    case StreamMode::default_stream: return "default_stream";
  }
  return "unknown";
}

bool check(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s (%d)\n", operation,
               cudaGetErrorString(status), static_cast<int>(status));
  return false;
}

bool copy_and_check(int src_device, int dst_device, void *src, void *dst,
                    const std::vector<unsigned char> &expected,
                    StreamMode mode,
                    const std::array<cudaStream_t, 2> &streams,
                    const std::array<cudaEvent_t, 2> &events) {
  if (!check(cudaSetDevice(src_device), "cudaSetDevice(source)")) return false;
  if (!check(cudaMemcpy(src, expected.data(), expected.size(),
                        cudaMemcpyHostToDevice), "cudaMemcpy(source upload)")) return false;
  cudaStream_t selected_stream = nullptr;
  if (mode == StreamMode::source_event_wait) {
    selected_stream = streams[static_cast<std::size_t>(src_device)];
  } else if (mode == StreamMode::destination_stream) {
    selected_stream = streams[static_cast<std::size_t>(dst_device)];
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(destination)")) return false;
  }
  if (!check(cudaMemcpyPeerAsync(dst, dst_device, src, src_device,
                                 expected.size(), selected_stream),
             "cudaMemcpyPeerAsync(cross-pair)")) return false;
  if (mode == StreamMode::source_event_wait) {
    const cudaEvent_t event = events[static_cast<std::size_t>(src_device)];
    if (!check(cudaEventRecord(event, selected_stream), "cudaEventRecord(source)")) return false;
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(destination wait)")) return false;
    const cudaStream_t destination = streams[static_cast<std::size_t>(dst_device)];
    if (!check(cudaStreamWaitEvent(destination, event, 0), "cudaStreamWaitEvent(destination)")) return false;
    if (!check(cudaStreamSynchronize(destination), "cudaStreamSynchronize(destination)")) return false;
  } else if (mode == StreamMode::destination_stream) {
    if (!check(cudaStreamSynchronize(selected_stream), "cudaStreamSynchronize(destination)")) return false;
  } else {
    if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(source default stream)")) return false;
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(destination default stream)")) return false;
    if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(destination default stream)")) return false;
  }
  std::vector<unsigned char> observed(expected.size());
  if (!check(cudaMemcpy(observed.data(), dst, observed.size(),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(destination download)")) return false;
  if (observed != expected) {
    std::fprintf(stderr, "cross-pair payload mismatch for CUDA%d -> CUDA%d\n",
                 src_device, dst_device);
    return false;
  }
  std::printf("COPY_OK src_device=%d dst_device=%d bytes=%zu mode=%s\n",
              src_device, dst_device, expected.size(), mode_name(mode));
  return true;
}

}  // namespace

int main() {
  int device_count = 0;
  if (!check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) return 1;
  if (device_count != 2) {
    std::fprintf(stderr, "expected 2 logical V200 devices, got %d\n", device_count);
    return 1;
  }
  const DriverVmmApi vmm_api = resolve_vmm_api();
  if (!vmm_api.ready()) {
    std::fprintf(stderr, "interposed CUDA VMM Driver API symbols were not resolved after CUDA initialization\n");
    return 1;
  }
  if (!vmm_round_trip(0, vmm_api) || !vmm_round_trip(1, vmm_api)) return 1;

  std::array<void *, 2> buffers{};
  std::array<cudaStream_t, 2> streams{};
  std::array<cudaEvent_t, 2> events{};
  for (int device = 0; device < 2; ++device) {
    if (!check(cudaSetDevice(device), "cudaSetDevice(allocate)")) return 1;
    if (!check(cudaMalloc(&buffers[static_cast<std::size_t>(device)], kBytes),
               "cudaMalloc")) return 1;
    if (!check(cudaStreamCreateWithFlags(&streams[static_cast<std::size_t>(device)],
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags")) return 1;
    if (!check(cudaEventCreateWithFlags(&events[static_cast<std::size_t>(device)],
                                        cudaEventDisableTiming),
               "cudaEventCreateWithFlags")) return 1;
  }

  bool ok = true;
  std::vector<unsigned char> forward(kBytes);
  std::vector<unsigned char> reverse(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    forward[i] = static_cast<unsigned char>((i * 13u + 7u) & 0xffu);
    reverse[i] = static_cast<unsigned char>((i * 29u + 3u) & 0xffu);
  }
  for (StreamMode mode : {StreamMode::source_event_wait,
                          StreamMode::destination_stream,
                          StreamMode::default_stream}) {
    ok = copy_and_check(0, 1, buffers[0], buffers[1], forward, mode,
                        streams, events) && ok;
    ok = copy_and_check(1, 0, buffers[1], buffers[0], reverse, mode,
                        streams, events) && ok;
  }

  for (int device = 1; device >= 0; --device) {
    if (!check(cudaSetDevice(device), "cudaSetDevice(free)")) ok = false;
    if (events[static_cast<std::size_t>(device)] != nullptr) {
      if (!check(cudaEventDestroy(events[static_cast<std::size_t>(device)]),
                 "cudaEventDestroy")) ok = false;
    }
    if (streams[static_cast<std::size_t>(device)] != nullptr) {
      if (!check(cudaStreamDestroy(streams[static_cast<std::size_t>(device)]),
                 "cudaStreamDestroy")) ok = false;
    }
    if (buffers[static_cast<std::size_t>(device)] != nullptr) {
      if (!check(cudaFree(buffers[static_cast<std::size_t>(device)]), "cudaFree")) ok = false;
      buffers[static_cast<std::size_t>(device)] = nullptr;
    }
  }
  return ok ? 0 : 1;
}
