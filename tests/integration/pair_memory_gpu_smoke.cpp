// Opt-in real pair initializer/memory closure. Build only by default; execution
// requires the outer controller to bind B02, UUIDs, and a finite memory cap.
#include "v200/core/runtime_init.h"
#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kPair[] =
    "GPU-21131915-1488-23af-38dd-1743ae1f5cc8,GPU-d745da9c-7649-8334-41f1-483afa6f3206";
constexpr char kB01[] = "8426eab5-278d-4687-ab7d-f5ec937e46ec";
constexpr size_t kMiB = 1024 * 1024;
constexpr size_t kAllocationMiB = 96;
constexpr unsigned char kFill = 0x6d;
unsigned cleanup_errors = 0;

void check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) return;
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (name ? name : "CUDA error") + " " +
                           (message ? message : ""));
}

void cleanup_check(CUresult result, const char *operation) noexcept {
  if (result == CUDA_SUCCESS) return;
  ++cleanup_errors;
  std::fprintf(stderr, "cleanup failure: %s (%d)\n", operation,
               static_cast<int>(result));
}

bool env_equals(const char *name, const char *expected) {
  const char *value = std::getenv(name);
  return value != nullptr && std::string(value) == expected;
}

void validate_runner(int argc, char **argv) {
  if (argc != 2 || std::string(argv[1]) != "--execute") {
    throw std::runtime_error("require --execute under the B02 controller");
  }
  const char *receipt = std::getenv("TODO_GPU_LEASE_RECEIPT");
  if (receipt == nullptr || *receipt == '\0' || !std::ifstream(receipt).good()) {
    throw std::runtime_error("missing controller receipt; use the B02 bounded runner");
  }
  if (!env_equals("V200_BUNDLE_ID", "B02")) {
    throw std::runtime_error("pair memory smoke must be bound to B02");
  }
  if (!env_equals("V200_MODE", "pair") || !env_equals("V200_DEVICE_UUIDS", kPair) ||
      !env_equals("V200_B01_EVIDENCE_ID", kB01)) {
    throw std::runtime_error("exact pair mode, ordered UUIDs, and B01 evidence ID are required");
  }
  if (!env_equals("V200_OWNER0_BUDGET_MIB", "64") ||
      !env_equals("V200_OWNER1_BUDGET_MIB", "64")) {
    throw std::runtime_error("set both test owner budgets to 64 MiB");
  }
  const char *cap_text = std::getenv("V200_BUNDLE_VRAM_CAP_MIB_PER_GPU");
  if (cap_text == nullptr || *cap_text == '\0') {
    throw std::runtime_error("missing controller per-GPU allocation cap");
  }
  char *end = nullptr;
  const unsigned long cap = std::strtoul(cap_text, &end, 10);
  if (end == cap_text || *end != '\0' || cap < kAllocationMiB || cap > 12288) {
    throw std::runtime_error("controller cap must cover 96 MiB and be at most 12288 MiB");
  }
}

class OwnedClosure {
 public:
  using ContextRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using ContextRelease = CUresult(CUDAAPI *)(CUdevice);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);

  OwnedClosure(CUdevice device, ContextRetain retain, ContextRelease release,
               ContextPush push, ContextPop pop)
      : device_(device), retain_(retain), release_(release), push_(push), pop_(pop) {}
  ~OwnedClosure() { cleanup(); }
  OwnedClosure(const OwnedClosure &) = delete;
  OwnedClosure &operator=(const OwnedClosure &) = delete;

  void retain_and_push() {
    check(retain_(&context_, device_), "native cuDevicePrimaryCtxRetain anchor");
    retained_ = true;
    check(push_(context_), "native cuCtxPushCurrent anchor");
    pushed_ = true;
  }
  void set_allocation(void *ptr) { allocation_ = ptr; }

  bool cleanup() noexcept {
    if (closed_) return cleanup_errors == 0;
    bool safe = true;
    if (allocation_ != nullptr) {
      const CUresult result = v200_memory_release(allocation_);
      cleanup_check(result, "v200_memory_release");
      if (result == CUDA_SUCCESS) allocation_ = nullptr;
      else safe = false;
    }
    if (pushed_) {
      CUcontext popped = nullptr;
      const CUresult result = pop_(&popped);
      cleanup_check(result, "native cuCtxPopCurrent anchor");
      if (result == CUDA_SUCCESS) {
        pushed_ = false;
        if (popped != context_) {
          ++cleanup_errors;
          safe = false;
        }
      } else {
        safe = false;
      }
    }
    if (safe && retained_) {
      const CUresult result = release_(device_);
      cleanup_check(result, "native cuDevicePrimaryCtxRelease anchor");
      if (result == CUDA_SUCCESS) retained_ = false;
      else safe = false;
    }
    closed_ = true;
    return cleanup_errors == 0 && safe;
  }

 private:
  CUdevice device_{};
  ContextRetain retain_{};
  ContextRelease release_{};
  ContextPush push_{};
  ContextPop pop_{};
  CUcontext context_{};
  void *allocation_{};
  bool retained_{false};
  bool pushed_{false};
  bool closed_{false};
};

}  // namespace

namespace {

struct NativeDriver {
  using Init = CUresult(CUDAAPI *)(unsigned int);
  using DeviceCount = CUresult(CUDAAPI *)(int *);
  using DeviceGet = CUresult(CUDAAPI *)(CUdevice *, int);
  using DeviceUuid = CUresult(CUDAAPI *)(CUuuid *, CUdevice);
  using ContextRetain = OwnedClosure::ContextRetain;
  using ContextRelease = OwnedClosure::ContextRelease;
  using ContextPush = OwnedClosure::ContextPush;
  using ContextPop = OwnedClosure::ContextPop;
  using MemsetD8 = CUresult(CUDAAPI *)(CUdeviceptr, unsigned char, size_t);
  using MemcpyDtoH = CUresult(CUDAAPI *)(void *, CUdeviceptr, size_t);

  Init init{};
  DeviceCount device_count{};
  DeviceGet device_get{};
  DeviceUuid device_uuid{};
  ContextRetain context_retain{};
  ContextRelease context_release{};
  ContextPush context_push{};
  ContextPop context_pop{};
  MemsetD8 memset_d8{};
  MemcpyDtoH memcpy_dtoh{};

  template <class Fn>
  static Fn resolve(const char *name, const char *fallback = nullptr) {
    auto *symbol = v200_frontdoor_resolve_driver_symbol(name);
    if (symbol == nullptr && fallback != nullptr) {
      symbol = v200_frontdoor_resolve_driver_symbol(fallback);
    }
    return reinterpret_cast<Fn>(symbol);
  }

  void load() {
    init = resolve<Init>("cuInit");
    device_count = resolve<DeviceCount>("cuDeviceGetCount");
    device_get = resolve<DeviceGet>("cuDeviceGet");
    device_uuid = resolve<DeviceUuid>("cuDeviceGetUuid_v2", "cuDeviceGetUuid");
    context_retain = resolve<ContextRetain>("cuDevicePrimaryCtxRetain");
    context_release = resolve<ContextRelease>("cuDevicePrimaryCtxRelease_v2",
                                              "cuDevicePrimaryCtxRelease");
    context_push = resolve<ContextPush>("cuCtxPushCurrent_v2", "cuCtxPushCurrent");
    context_pop = resolve<ContextPop>("cuCtxPopCurrent_v2", "cuCtxPopCurrent");
    memset_d8 = resolve<MemsetD8>("cuMemsetD8_v2", "cuMemsetD8");
    memcpy_dtoh = resolve<MemcpyDtoH>("cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    if (!init || !device_count || !device_get || !device_uuid || !context_retain ||
        !context_release || !context_push || !context_pop || !memset_d8 || !memcpy_dtoh) {
      throw std::runtime_error("frontdoor could not resolve the complete native Driver API surface");
    }
  }
};

std::string native_uuid(const NativeDriver &driver, CUdevice device) {
  CUuuid raw{};
  check(driver.device_uuid(&raw, device), "native cuDeviceGetUuid");
  char text[41]{};
  std::snprintf(text, sizeof(text),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[0])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[1])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[2])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[3])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[4])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[5])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[6])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[7])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[8])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[9])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[10])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[11])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[12])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[13])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[14])),
                static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[15])));
  return text;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    validate_runner(argc, argv);
    check(v200_initialize_profile(), "v200_initialize_profile");
    NativeDriver native;
    native.load();
    check(native.init(0), "native cuInit");
    int logical_count = 0;
    check(cuDeviceGetCount(&logical_count), "frontdoor logical cuDeviceGetCount");
    if (logical_count != 1) {
      throw std::runtime_error("frontdoor must present exactly one logical anchor device");
    }
    int count = 0;
    check(native.device_count(&count), "native cuDeviceGetCount");
    if (count != 2) throw std::runtime_error("native CUDA view must expose exactly the ordered pair");
    CUdevice physical[2] = {-1, -1};
    check(native.device_get(&physical[0], 0), "native cuDeviceGet owner 0");
    check(native.device_get(&physical[1], 1), "native cuDeviceGet owner 1");
    if (native_uuid(native, physical[0]) != "GPU-21131915-1488-23af-38dd-1743ae1f5cc8" ||
        native_uuid(native, physical[1]) != "GPU-d745da9c-7649-8334-41f1-483afa6f3206") {
      throw std::runtime_error("controller-visible device order differs from exact UUID binding");
    }

    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    check(v200_memory_get_profile(&profile), "v200_memory_get_profile");
    if (profile.owner_device[0] != physical[0] || profile.owner_device[1] != physical[1] ||
        profile.owner_budget_bytes[0] != 64 * kMiB ||
        profile.owner_budget_bytes[1] != 64 * kMiB ||
        std::string(profile.qualification_id) != kB01) {
      throw std::runtime_error("initializer profile does not match ordered 64 MiB B01 binding");
    }

    OwnedClosure owned(profile.owner_device[0], native.context_retain,
                       native.context_release, native.context_push, native.context_pop);
    owned.retain_and_push();
    void *allocation = nullptr;
    check(v200_memory_allocate_on(profile.owner_device[0], kAllocationMiB * kMiB, 0, &allocation),
          "v200_memory_allocate_on 96 MiB");
    owned.set_allocation(allocation);

    v200_allocation_info allocation_info{};
    allocation_info.abi_version = V200_MEMORY_ABI_VERSION;
    allocation_info.struct_size = sizeof(allocation_info);
    check(v200_memory_query(allocation, &allocation_info), "v200_memory_query");
    if (allocation_info.owner_bytes[0] == 0 || allocation_info.owner_bytes[1] == 0 ||
        allocation_info.owner_bytes[0] + allocation_info.owner_bytes[1] <
            kAllocationMiB * kMiB) {
      throw std::runtime_error("96 MiB allocation did not use both physical owners");
    }

    check(native.memset_d8(reinterpret_cast<CUdeviceptr>(allocation), kFill,
                           kAllocationMiB * kMiB), "native cuMemsetD8 full split range");
    std::vector<unsigned char> host(kAllocationMiB * kMiB);
    check(native.memcpy_dtoh(host.data(), reinterpret_cast<CUdeviceptr>(allocation), host.size()),
          "native cuMemcpyDtoH full split range");
    if (!std::all_of(host.begin(), host.end(),
                     [](unsigned char byte) { return byte == kFill; })) {
      throw std::runtime_error("full-range anchor write/read verification failed");
    }

    v200_memory_usage_info before_free{};
    before_free.abi_version = V200_MEMORY_ABI_VERSION;
    before_free.struct_size = sizeof(before_free);
    check(v200_memory_get_usage(&before_free), "v200_memory_get_usage before free");
    if (before_free.owner_used_bytes[0] == 0 || before_free.owner_used_bytes[1] == 0 ||
        before_free.live_allocation_count != 1) {
      throw std::runtime_error("owner ledger did not retain both backing extents");
    }

    if (!owned.cleanup()) throw std::runtime_error("owned CUDA cleanup failed");
    v200_memory_usage_info after_free{};
    after_free.abi_version = V200_MEMORY_ABI_VERSION;
    after_free.struct_size = sizeof(after_free);
    check(v200_memory_get_usage(&after_free), "v200_memory_get_usage after free");
    if (after_free.live_allocation_count != 0 || after_free.live_reservation_count != 0 ||
        after_free.cleanup_pending_count != 0 || after_free.owner_used_bytes[0] != 0 ||
        after_free.owner_used_bytes[1] != 0 || cleanup_errors != 0) {
      throw std::runtime_error("pair allocation did not retire to a zero live ledger");
    }

    std::cout << "{\"status\":\"GPU_PASS\",\"qualification_only\":true"
              << ",\"bundle\":\"B02\",\"ordered_uuids\":[\""
              << profile.owner_uuid[0] << "\",\"" << profile.owner_uuid[1]
              << "\"],\"requested_bytes\":" << allocation_info.requested_bytes
              << ",\"mapped_bytes\":" << allocation_info.mapped_bytes
              << ",\"owner_bytes\":[" << allocation_info.owner_bytes[0] << ','
              << allocation_info.owner_bytes[1]
              << "],\"full_range_anchor_write_read\":true"
              << ",\"ledger_zero_after_free\":true"
              << ",\"arbitrary_kernel_transparency\":false}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pair memory GPU smoke: " << error.what() << '\n';
    return 1;
  }
}
