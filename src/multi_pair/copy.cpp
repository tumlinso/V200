#include "v200/multi_pair/copy.hpp"

#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kPairCount = 2;
constexpr size_t kChunkBytes = 8u * 1024u * 1024u;
constexpr size_t kMaximumCopyBytes = 1024u * 1024u * 1024u;

struct DriverApi {
  using ContextGetCurrent = CUresult(CUDAAPI *)(CUcontext *);
  using ContextGetDevice = CUresult(CUDAAPI *)(CUdevice *);
  using PrimaryContextRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using PrimaryContextRelease = CUresult(CUDAAPI *)(CUdevice);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);
  using ContextSetCurrent = CUresult(CUDAAPI *)(CUcontext);
  using ContextSynchronize = CUresult(CUDAAPI *)();
  using StreamSynchronize = CUresult(CUDAAPI *)(CUstream);
  using HostAlloc = CUresult(CUDAAPI *)(void **, size_t, unsigned int);
  using HostFree = CUresult(CUDAAPI *)(void *);
  using CopyDtoH = CUresult(CUDAAPI *)(void *, CUdeviceptr, size_t);
  using CopyHtoD = CUresult(CUDAAPI *)(CUdeviceptr, const void *, size_t);

  ContextGetCurrent get_current{};
  ContextGetDevice get_device{};
  PrimaryContextRetain primary_retain{};
  PrimaryContextRelease primary_release{};
  ContextPush push{};
  ContextPop pop{};
  ContextSetCurrent set_current{};
  ContextSynchronize synchronize{};
  StreamSynchronize stream_synchronize{};
  HostAlloc host_alloc{};
  HostFree host_free{};
  CopyDtoH copy_dtoh{};
  CopyHtoD copy_htod{};

  bool ready() const noexcept {
    return get_current != nullptr && get_device != nullptr &&
           primary_retain != nullptr && primary_release != nullptr &&
           push != nullptr && pop != nullptr && set_current != nullptr &&
           synchronize != nullptr && stream_synchronize != nullptr &&
           host_alloc != nullptr && host_free != nullptr &&
           copy_dtoh != nullptr && copy_htod != nullptr;
  }
};

struct RuntimeApi {
  using StreamIsCapturing = cudaError_t(CUDARTAPI *)(
      cudaStream_t, cudaStreamCaptureStatus *);

  StreamIsCapturing stream_is_capturing{};
};

template <typename Function>
Function driver_symbol(const char *name, const char *fallback = nullptr) noexcept {
  void *address = v200_frontdoor_resolve_driver_symbol(name);
  if (address == nullptr && fallback != nullptr)
    address = v200_frontdoor_resolve_driver_symbol(fallback);
  return reinterpret_cast<Function>(address);
}

void resolve_driver(DriverApi *api) noexcept {
  if (api == nullptr || api->ready()) return;
  api->get_current = driver_symbol<DriverApi::ContextGetCurrent>("cuCtxGetCurrent");
  api->get_device = driver_symbol<DriverApi::ContextGetDevice>("cuCtxGetDevice");
  api->primary_retain =
      driver_symbol<DriverApi::PrimaryContextRetain>("cuDevicePrimaryCtxRetain");
  api->primary_release = driver_symbol<DriverApi::PrimaryContextRelease>(
      "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
  api->push = driver_symbol<DriverApi::ContextPush>("cuCtxPushCurrent_v2",
                                                    "cuCtxPushCurrent");
  api->pop = driver_symbol<DriverApi::ContextPop>("cuCtxPopCurrent_v2",
                                                  "cuCtxPopCurrent");
  api->set_current = driver_symbol<DriverApi::ContextSetCurrent>("cuCtxSetCurrent");
  api->synchronize = driver_symbol<DriverApi::ContextSynchronize>("cuCtxSynchronize");
  api->stream_synchronize =
      driver_symbol<DriverApi::StreamSynchronize>("cuStreamSynchronize");
  api->host_alloc = driver_symbol<DriverApi::HostAlloc>("cuMemHostAlloc");
  api->host_free = driver_symbol<DriverApi::HostFree>("cuMemFreeHost");
  api->copy_dtoh = driver_symbol<DriverApi::CopyDtoH>("cuMemcpyDtoH_v2",
                                                     "cuMemcpyDtoH");
  api->copy_htod = driver_symbol<DriverApi::CopyHtoD>("cuMemcpyHtoD_v2",
                                                     "cuMemcpyHtoD");
}

cudaError_t reject_capture(cudaStream_t stream) noexcept {
  RuntimeApi runtime{};
  runtime.stream_is_capturing = reinterpret_cast<RuntimeApi::StreamIsCapturing>(
      v200_frontdoor_resolve_runtime_symbol("cudaStreamIsCapturing"));
  if (runtime.stream_is_capturing == nullptr) return cudaErrorNotSupported;
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t result = runtime.stream_is_capturing(stream, &capture_status);
  if (result != cudaSuccess) return result;
  if (capture_status == cudaStreamCaptureStatusActive)
    return cudaErrorStreamCaptureUnsupported;
  if (capture_status == cudaStreamCaptureStatusInvalidated)
    return cudaErrorStreamCaptureInvalidated;
  return cudaSuccess;
}

cudaError_t map_driver_error(CUresult result) noexcept {
  switch (result) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
      return cudaErrorInvalidValue;
    case CUDA_ERROR_INVALID_DEVICE:
      return cudaErrorInvalidDevice;
    case CUDA_ERROR_OUT_OF_MEMORY:
      return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED:
      return cudaErrorInitializationError;
    case CUDA_ERROR_NOT_SUPPORTED:
      return cudaErrorNotSupported;
    case CUDA_ERROR_INVALID_CONTEXT:
      return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_INVALID_HANDLE:
      return cudaErrorInvalidResourceHandle;
    default:
      return cudaErrorUnknown;
  }
}

class ContextGuard {
 public:
  ContextGuard(DriverApi &driver, CUcontext context,
               CUcontext restore_context) noexcept
      : driver_(driver), restore_context_(restore_context), context_(context) {
    status_ = driver_.push(context_);
    pushed_ = status_ == CUDA_SUCCESS;
  }

  ~ContextGuard() {
    if (pushed_) (void)restore();
  }

  CUresult status() const noexcept { return status_; }

  CUresult restore() noexcept {
    if (!pushed_) return status_;
    pushed_ = false;
    CUcontext popped = nullptr;
    const CUresult popped_status = driver_.pop(&popped);
    if (popped_status == CUDA_SUCCESS && popped == context_) {
      CUcontext current = nullptr;
      const CUresult current_status = driver_.get_current(&current);
      if (current_status == CUDA_SUCCESS && current == restore_context_)
        return CUDA_SUCCESS;
    }
    const CUresult restored = driver_.set_current(restore_context_);
    return restored == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : restored;
  }

 private:
  DriverApi &driver_;
  CUcontext restore_context_{};
  CUcontext context_{};
  CUresult status_ = CUDA_SUCCESS;
  bool pushed_ = false;
};

class PairLeases {
 public:
  CUresult begin(uint32_t dst_pair, uint32_t src_pair) noexcept {
    CUresult result = v200_memory_begin_submission_for_pair(dst_pair, &dst_lease_);
    if (result != CUDA_SUCCESS) return result;
    dst_active_ = dst_lease_ != 0;
    if (!dst_active_) return CUDA_ERROR_NOT_INITIALIZED;

    result = v200_memory_begin_submission_for_pair(src_pair, &src_lease_);
    if (result != CUDA_SUCCESS) {
      finish_one(dst_pair, dst_lease_, true, &dst_active_);
      return result;
    }
    src_active_ = src_lease_ != 0;
    if (!src_active_) {
      finish_one(dst_pair, dst_lease_, true, &dst_active_);
      return CUDA_ERROR_NOT_INITIALIZED;
    }
    dst_pair_ = dst_pair;
    src_pair_ = src_pair;
    return CUDA_SUCCESS;
  }

  ~PairLeases() {
    if (src_active_)
      (void)v200_memory_end_submission_for_pair(src_pair_, src_lease_, 0);
    if (dst_active_)
      (void)v200_memory_end_submission_for_pair(dst_pair_, dst_lease_, 0);
  }

  CUresult finish(bool completion_known) noexcept {
    CUresult src_result = CUDA_SUCCESS;
    CUresult dst_result = CUDA_SUCCESS;
    if (src_active_) {
      src_active_ = false;
      src_result = v200_memory_end_submission_for_pair(
          src_pair_, src_lease_, completion_known ? 1 : 0);
    }
    if (dst_active_) {
      dst_active_ = false;
      dst_result = v200_memory_end_submission_for_pair(
          dst_pair_, dst_lease_, completion_known ? 1 : 0);
    }
    if (src_result != CUDA_SUCCESS) return src_result;
    return dst_result;
  }

 private:
  static void finish_one(uint32_t pair, uint64_t lease, bool known,
                         bool *active) noexcept {
    if (active == nullptr || !*active) return;
    *active = false;
    (void)v200_memory_end_submission_for_pair(pair, lease, known ? 1 : 0);
  }

  uint32_t dst_pair_ = 0;
  uint32_t src_pair_ = 0;
  uint64_t dst_lease_ = 0;
  uint64_t src_lease_ = 0;
  bool dst_active_ = false;
  bool src_active_ = false;
};

class HostBuffer {
 public:
  explicit HostBuffer(DriverApi &driver) noexcept : driver_(driver) {}
  ~HostBuffer() {
    if (buffer_ != nullptr) (void)driver_.host_free(buffer_);
  }

  CUresult allocate(size_t bytes) noexcept {
    return driver_.host_alloc(&buffer_, bytes, CU_MEMHOSTALLOC_PORTABLE);
  }
  void *get() const noexcept { return buffer_; }

 private:
  DriverApi &driver_;
  void *buffer_ = nullptr;
};

class RetainedPrimaryContext {
 public:
  RetainedPrimaryContext(DriverApi &driver, CUdevice device) noexcept
      : driver_(driver), device_(device) {}
  ~RetainedPrimaryContext() {
    if (retained_) (void)driver_.primary_release(device_);
  }

  CUresult acquire() noexcept {
    const CUresult result = driver_.primary_retain(&context_, device_);
    retained_ = result == CUDA_SUCCESS;
    return result;
  }
  CUcontext get() const noexcept { return context_; }

 private:
  DriverApi &driver_;
  CUdevice device_;
  CUcontext context_{};
  bool retained_ = false;
};

CUresult synchronize_context(DriverApi &driver, CUcontext context,
                             CUcontext restore_context) noexcept {
  ContextGuard guard(driver, context, restore_context);
  if (guard.status() != CUDA_SUCCESS) return guard.status();
  const CUresult synchronized = driver.synchronize();
  const CUresult restored = guard.restore();
  return synchronized == CUDA_SUCCESS ? restored : synchronized;
}

bool pointer_known_to_v200(const void *ptr) noexcept {
  return v200_memory_owns_pointer_for_pair(0, ptr) != 0 ||
         v200_memory_owns_pointer_for_pair(1, ptr) != 0;
}

int handled(cudaError_t value, cudaError_t *status) noexcept {
  if (status != nullptr) *status = value;
  return 1;
}

cudaError_t staged_copy(uint32_t dst_pair, uint32_t src_pair,
                        uint32_t stream_pair, void *dst, const void *src,
                        size_t bytes, cudaStream_t stream) noexcept {
  if (dst_pair > 1 || src_pair > 1) return cudaErrorInvalidDevice;
  if (bytes == 0) return cudaSuccess;
  if (dst == nullptr || src == nullptr) return cudaErrorInvalidValue;
  if (bytes > kMaximumCopyBytes) return cudaErrorNotSupported;
  if (stream_pair != dst_pair && stream_pair != src_pair)
    return cudaErrorInvalidDevice;
  if (dst_pair != v200_frontdoor_current_pair() ||
      dst_pair != v200_memory_current_pair())
    return cudaErrorInvalidDevice;

  v200_memory_pair_profile dst_profile{};
  dst_profile.abi_version = V200_MEMORY_ABI_VERSION;
  dst_profile.struct_size = sizeof(dst_profile);
  CUresult memory_result =
      v200_memory_get_profile_for_pair(dst_pair, &dst_profile);
  if (memory_result != CUDA_SUCCESS) return map_driver_error(memory_result);
  v200_memory_pair_profile src_profile{};
  src_profile.abi_version = V200_MEMORY_ABI_VERSION;
  src_profile.struct_size = sizeof(src_profile);
  memory_result = v200_memory_get_profile_for_pair(src_pair, &src_profile);
  if (memory_result != CUDA_SUCCESS) return map_driver_error(memory_result);
  if (dst_profile.pair_access_qualified == 0 ||
      src_profile.pair_access_qualified == 0 ||
      dst_profile.qualification_id[0] == '\0' ||
      src_profile.qualification_id[0] == '\0')
    return cudaErrorNotSupported;

  DriverApi driver{};
  resolve_driver(&driver);
  if (!driver.ready()) return cudaErrorNotSupported;

  CUcontext destination_context = nullptr;
  CUresult result = driver.get_current(&destination_context);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (destination_context == nullptr) return cudaErrorInvalidResourceHandle;
  CUdevice current_device = -1;
  result = driver.get_device(&current_device);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (current_device != dst_profile.owner_device[0])
    return cudaErrorInvalidDevice;

  RetainedPrimaryContext source_primary(driver, src_profile.owner_device[0]);
  result = source_primary.acquire();
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  const CUcontext source_context = source_primary.get();
  if (source_context == nullptr)
    return cudaErrorInvalidResourceHandle;

  const CUcontext stream_context =
      stream_pair == src_pair ? source_context : destination_context;
  cudaError_t capture_status = cudaSuccess;
  {
    ContextGuard stream_guard(driver, stream_context, destination_context);
    if (stream_guard.status() != CUDA_SUCCESS)
      return map_driver_error(stream_guard.status());
    capture_status = reject_capture(stream);
    const CUresult restored = stream_guard.restore();
    if (capture_status == cudaSuccess && restored != CUDA_SUCCESS)
      return map_driver_error(restored);
  }
  if (capture_status != cudaSuccess) return capture_status;

  PairLeases leases;
  result = leases.begin(dst_pair, src_pair);
  if (result != CUDA_SUCCESS) return map_driver_error(result);

  const bool dst_range =
      v200_memory_owns_range_for_pair(dst_pair, dst, bytes) != 0;
  const bool src_range =
      v200_memory_owns_range_for_pair(src_pair, src, bytes) != 0;
  if (!dst_range || !src_range) {
    (void)leases.finish(true);
    return cudaErrorInvalidValue;
  }

  HostBuffer buffer(driver);
  result = buffer.allocate(std::min(bytes, kChunkBytes));
  if (result != CUDA_SUCCESS) {
    (void)leases.finish(true);
    return map_driver_error(result);
  }

  {
    ContextGuard stream_guard(driver, stream_context, destination_context);
    if (stream_guard.status() != CUDA_SUCCESS) {
      (void)leases.finish(true);
      return map_driver_error(stream_guard.status());
    }
    result = driver.stream_synchronize(reinterpret_cast<CUstream>(stream));
    const CUresult restored = stream_guard.restore();
    if (result == CUDA_SUCCESS && restored != CUDA_SUCCESS) result = restored;
  }
  if (result != CUDA_SUCCESS) {
    (void)leases.finish(false);
    return map_driver_error(result);
  }

  result = synchronize_context(driver, source_context, destination_context);
  if (result == CUDA_SUCCESS)
    result = synchronize_context(driver, destination_context, destination_context);
  if (result != CUDA_SUCCESS) {
    (void)leases.finish(false);
    return map_driver_error(result);
  }

  bool copy_attempted = false;
  auto *host = static_cast<unsigned char *>(buffer.get());
  auto *destination = static_cast<unsigned char *>(dst);
  const auto *source = static_cast<const unsigned char *>(src);
  size_t offset = 0;
  while (result == CUDA_SUCCESS && offset < bytes) {
    const size_t chunk = std::min(kChunkBytes, bytes - offset);
    {
      ContextGuard source_guard(driver, source_context, destination_context);
      if (source_guard.status() != CUDA_SUCCESS) {
        result = source_guard.status();
      } else {
        copy_attempted = true;
        result = driver.copy_dtoh(
            host, reinterpret_cast<CUdeviceptr>(source + offset), chunk);
        const CUresult restored = source_guard.restore();
        if (result == CUDA_SUCCESS && restored != CUDA_SUCCESS)
          result = restored;
      }
    }
    if (result != CUDA_SUCCESS) break;
    {
      ContextGuard destination_guard(driver, destination_context,
                                     destination_context);
      if (destination_guard.status() != CUDA_SUCCESS) {
        result = destination_guard.status();
      } else {
        copy_attempted = true;
        result = driver.copy_htod(
            reinterpret_cast<CUdeviceptr>(destination + offset), host, chunk);
        const CUresult restored = destination_guard.restore();
        if (result == CUDA_SUCCESS && restored != CUDA_SUCCESS)
          result = restored;
      }
    }
    if (result == CUDA_SUCCESS) offset += chunk;
  }

  const bool copies_complete = result == CUDA_SUCCESS || !copy_attempted;
  const CUresult lease_status = leases.finish(copies_complete);
  if (result == CUDA_SUCCESS && lease_status != CUDA_SUCCESS)
    result = lease_status;
  return map_driver_error(result);
}

}  // namespace

extern "C" int v200_multi_pair_try_copy_peer_async(
    uint32_t dst_pair, uint32_t src_pair, uint32_t stream_pair, void *dst,
    const void *src, size_t bytes, cudaStream_t stream, cudaError_t *status) {
  if (status != nullptr) *status = cudaSuccess;
  if (dst_pair == src_pair) return 0;
  if (dst_pair >= kPairCount || src_pair >= kPairCount)
    return handled(cudaErrorInvalidDevice, status);
  if (bytes == 0) return handled(cudaSuccess, status);

  // Only intercept copies whose both complete ranges are V200 allocations.
  // If neither endpoint belongs to V200, the frontdoor can use its normal
  // native route. Any partial, wrong-pair, or out-of-bounds managed range must
  // fail closed to avoid forwarding a logical managed pointer to CUDA.
  const bool dst_managed = pointer_known_to_v200(dst);
  const bool src_managed = pointer_known_to_v200(src);
  if (!dst_managed && !src_managed) return 0;
  const bool expected_dst =
      v200_memory_owns_range_for_pair(dst_pair, dst, bytes) != 0;
  const bool expected_src =
      v200_memory_owns_range_for_pair(src_pair, src, bytes) != 0;
  if (!expected_dst || !expected_src)
    return handled(cudaErrorInvalidValue, status);

  const cudaError_t result = staged_copy(dst_pair, src_pair, stream_pair, dst,
                                         src, bytes, stream);
  return handled(result, status);
}
