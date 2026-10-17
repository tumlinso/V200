#include "v200/multi_pair/copy.hpp"
#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"

#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expression)                                                       \
  do {                                                                          \
    if (!(expression)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #expression,         \
                   __FILE__, __LINE__);                                         \
      std::abort();                                                             \
    }                                                                           \
  } while (false)

struct CUctx_st { CUdevice device; };
static CUctx_st contexts[2]{{0}, {2}};
static thread_local CUcontext current = &contexts[1];
static thread_local std::vector<CUcontext> stack;
static std::vector<unsigned char> *source_memory;
static std::vector<unsigned char> *destination_memory;
static int begin_count[2]{};
static int end_count[2]{};
static int end_ok[2]{};
static cudaStreamCaptureStatus fake_capture_status = cudaStreamCaptureStatusNone;
static CUdevice capture_context_device = -1;
static CUdevice stream_sync_context_device = -1;
static int context_sync_count[2]{};

static CUresult CUDAAPI fake_get_current(CUcontext *out) { *out = current; return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_get_device(CUdevice *out) { *out = current->device; return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_primary_retain(CUcontext *out, CUdevice device) {
  if (device == 0) *out = &contexts[0];
  else if (device == 2) *out = &contexts[1];
  else return CUDA_ERROR_INVALID_DEVICE;
  return CUDA_SUCCESS;
}
static CUresult CUDAAPI fake_primary_release(CUdevice) { return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_push(CUcontext context) { stack.push_back(current); current = context; return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_pop(CUcontext *out) { *out = current; current = stack.back(); stack.pop_back(); return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_set_current(CUcontext context) { current = context; return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_sync() {
  const int index = current->device == 0 ? 0 : 1;
  ++context_sync_count[index];
  return CUDA_SUCCESS;
}
static CUresult CUDAAPI fake_stream_sync(CUstream) {
  stream_sync_context_device = current->device;
  return CUDA_SUCCESS;
}
static CUresult CUDAAPI fake_host_alloc(void **out, size_t bytes, unsigned int) { *out = std::malloc(bytes); return *out ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
static CUresult CUDAAPI fake_host_free(void *ptr) { std::free(ptr); return CUDA_SUCCESS; }
static CUresult CUDAAPI fake_dtoh(void *dst, CUdeviceptr src, size_t bytes) {
  CHECK(current->device == 0);
  std::memcpy(dst, reinterpret_cast<const void *>(src), bytes); return CUDA_SUCCESS;
}
static CUresult CUDAAPI fake_htod(CUdeviceptr dst, const void *src, size_t bytes) {
  CHECK(current->device == 2);
  std::memcpy(reinterpret_cast<void *>(dst), src, bytes); return CUDA_SUCCESS;
}
static cudaError_t CUDARTAPI fake_capture(cudaStream_t, cudaStreamCaptureStatus *status) {
  capture_context_device = current->device;
  *status = fake_capture_status; return cudaSuccess;
}

extern "C" void *v200_frontdoor_resolve_driver_symbol(const char *name) {
#define MATCH(n, f) if (std::strcmp(name, n) == 0) return reinterpret_cast<void *>(f)
  MATCH("cuCtxGetCurrent", fake_get_current);
  MATCH("cuCtxGetDevice", fake_get_device);
  MATCH("cuDevicePrimaryCtxRetain", fake_primary_retain);
  MATCH("cuDevicePrimaryCtxRelease_v2", fake_primary_release);
  MATCH("cuCtxPushCurrent_v2", fake_push);
  MATCH("cuCtxPopCurrent_v2", fake_pop);
  MATCH("cuCtxSetCurrent", fake_set_current);
  MATCH("cuCtxSynchronize", fake_sync);
  MATCH("cuStreamSynchronize", fake_stream_sync);
  MATCH("cuMemHostAlloc", fake_host_alloc);
  MATCH("cuMemFreeHost", fake_host_free);
  MATCH("cuMemcpyDtoH_v2", fake_dtoh);
  MATCH("cuMemcpyHtoD_v2", fake_htod);
#undef MATCH
  return nullptr;
}
extern "C" void *v200_frontdoor_resolve_runtime_symbol(const char *name) {
  return std::strcmp(name, "cudaStreamIsCapturing") == 0
      ? reinterpret_cast<void *>(fake_capture) : nullptr;
}
extern "C" uint32_t v200_frontdoor_current_pair(void) { return 1; }
extern "C" CUresult v200_memory_get_profile_for_pair(uint32_t pair, v200_memory_pair_profile *out) {
  if (pair > 1) return CUDA_ERROR_INVALID_DEVICE;
  out->abi_version = V200_MEMORY_ABI_VERSION;
  out->struct_size = sizeof(*out);
  out->owner_device[0] = static_cast<CUdevice>(pair * 2);
  out->owner_device[1] = static_cast<CUdevice>(pair * 2 + 1);
  out->pair_access_qualified = 1;
  std::strcpy(out->qualification_id, "cpu-contract");
  return CUDA_SUCCESS;
}
extern "C" uint32_t v200_memory_current_pair(void) { return 1; }
extern "C" int v200_memory_owns_pointer_for_pair(uint32_t pair, const void *ptr) {
  auto contains = [ptr](const std::vector<unsigned char> *v) {
    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto b = reinterpret_cast<uintptr_t>(v->data());
    return p >= b && p < b + v->size();
  };
  return (pair == 0 && source_memory && contains(source_memory)) ||
         (pair == 1 && destination_memory && contains(destination_memory));
}
extern "C" int v200_memory_owns_range_for_pair(uint32_t pair, const void *ptr, size_t bytes) {
  if (bytes == 0) return 0;
  auto contains = [ptr, bytes](const std::vector<unsigned char> *v) {
    auto p = reinterpret_cast<uintptr_t>(ptr);
    auto b = reinterpret_cast<uintptr_t>(v->data());
    return p >= b && bytes <= v->size() && p - b <= v->size() - bytes;
  };
  return (pair == 0 && source_memory && contains(source_memory)) ||
         (pair == 1 && destination_memory && contains(destination_memory));
}
extern "C" CUresult v200_memory_begin_submission_for_pair(uint32_t pair, uint64_t *lease) {
  if (pair > 1) return CUDA_ERROR_INVALID_DEVICE;
  *lease = ++begin_count[pair]; return CUDA_SUCCESS;
}
extern "C" CUresult v200_memory_end_submission_for_pair(uint32_t pair, uint64_t, int ok) {
  ++end_count[pair]; end_ok[pair] += ok; return CUDA_SUCCESS;
}

int main() {
  cudaError_t status = cudaErrorUnknown;
  CHECK(v200_multi_pair_try_copy_peer_async(1, 1, 1, nullptr, nullptr, 1,
                                              nullptr, &status) == 0);
  CHECK(status == cudaSuccess);
  CHECK(v200_multi_pair_try_copy_peer_async(1, 0, 0,
                                              reinterpret_cast<void *>(0x1000),
                                              reinterpret_cast<void *>(0x2000),
                                              64, nullptr, &status) == 0);
  std::vector<unsigned char> src(10u * 1024u * 1024u + 17u);
  std::vector<unsigned char> dst(src.size(), 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<unsigned char>((i * 17u) & 255u);
  source_memory = &src;
  destination_memory = &dst;
  CHECK(v200_multi_pair_try_copy_peer_async(1, 0, 0, dst.data() + 1,
                                              src.data(), src.size(), nullptr,
                                              &status) == 1);
  CHECK(status == cudaErrorInvalidValue);
  CHECK(begin_count[0] == 0 && begin_count[1] == 0);
  fake_capture_status = cudaStreamCaptureStatusActive;
  CHECK(v200_multi_pair_try_copy_peer_async(1, 0, 0, dst.data(), src.data(),
                                              src.size(), nullptr, &status) == 1);
  CHECK(status == cudaErrorStreamCaptureUnsupported);
  CHECK(begin_count[0] == 0 && begin_count[1] == 0);
  fake_capture_status = cudaStreamCaptureStatusNone;
  CHECK(v200_multi_pair_try_copy_peer_async(1, 0, 0, dst.data(), src.data(),
                                              src.size(), nullptr, &status) == 1);
  CHECK(status == cudaSuccess);
  CHECK(src == dst);
  CHECK(current == &contexts[1]);
  CHECK(capture_context_device == 0 && stream_sync_context_device == 0);
  CHECK(context_sync_count[0] > 0 && context_sync_count[1] > 0);
  CHECK(begin_count[0] == 1 && begin_count[1] == 1);
  CHECK(end_count[0] == 1 && end_count[1] == 1);
  CHECK(end_ok[0] == 1 && end_ok[1] == 1);
  return 0;
}
