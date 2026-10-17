#include "v200/core/runtime_init.h"

#include "v200/core/runtime_init.hpp"
#include "v200/frontdoor/runtime.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>

namespace v200::core {
namespace {

constexpr std::uint32_t kAnchorOwner = 0;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr std::uint64_t kMaxOwnerBudgetMiB = 12288;
constexpr std::uint64_t kPhysicalReserveBytes = kGiB;
constexpr char kExpectedUuid0[] = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8";
constexpr char kExpectedUuid1[] = "GPU-d745da9c-7649-8334-41f1-483afa6f3206";
constexpr char kExpectedUuid2[] = "GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505";
constexpr char kExpectedUuid3[] = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a";
constexpr char kExpectedB01[] = "8426eab5-278d-4687-ab7d-f5ec937e46ec";
constexpr char kExpectedB01Pair1[] = "1c97d504-468e-434c-81ea-112a4bb83e74";
constexpr char kExpectedPair[] =
    "GPU-21131915-1488-23af-38dd-1743ae1f5cc8,GPU-d745da9c-7649-8334-41f1-483afa6f3206";
constexpr char kExpectedPair1[] =
    "GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505,GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a";

bool parse_uuid(std::string_view text, CUuuid *out) noexcept {
  if (out == nullptr || text.size() != 40 || text.substr(0, 4) != "GPU-") return false;
  CUuuid parsed{};
  std::size_t byte = 0;
  int high = -1;
  for (std::size_t i = 4; i < text.size(); ++i) {
    if (i == 12 || i == 17 || i == 22 || i == 27) {
      if (text[i] != '-') return false;
      continue;
    }
    const char c = text[i];
    int value = -1;
    if (c >= '0' && c <= '9') value = c - '0';
    else if (c >= 'a' && c <= 'f') value = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') value = c - 'A' + 10;
    if (value < 0) return false;
    if (high < 0) high = value;
    else {
      if (byte >= sizeof(parsed.bytes)) return false;
      parsed.bytes[byte++] = static_cast<unsigned char>((high << 4) | value);
      high = -1;
    }
  }
  if (byte != sizeof(parsed.bytes) || high >= 0) return false;
  *out = parsed;
  return true;
}

std::string format_uuid(const CUuuid &uuid) {
  char text[41]{};
  std::snprintf(text, sizeof(text),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[0])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[1])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[2])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[3])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[4])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[5])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[6])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[7])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[8])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[9])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[10])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[11])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[12])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[13])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[14])),
                static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[15])));
  return text;
}

bool expected_input(const RuntimeInitInputs &inputs,
                    std::array<CUuuid, 2> *uuids) noexcept {
  if (inputs.ordered_uuid_pair == nullptr || inputs.b01_evidence_id == nullptr ||
      inputs.ordered_uuid_pair[0] == '\0' || inputs.b01_evidence_id[0] == '\0' ||
      inputs.logical_device_id == 0) {
    return false;
  }
  const std::string_view ordered(inputs.ordered_uuid_pair);
  const std::string_view evidence(inputs.b01_evidence_id);
  if (ordered == std::string_view(kExpectedPair)) {
    return inputs.logical_device_id == 1 &&
           evidence == std::string_view(kExpectedB01) &&
           parse_uuid(kExpectedUuid0, &(*uuids)[0]) &&
           parse_uuid(kExpectedUuid1, &(*uuids)[1]);
  }
  if (ordered == std::string_view(kExpectedPair1)) {
    return inputs.logical_device_id == 2 &&
           kExpectedB01Pair1[0] != '\0' &&
           evidence == std::string_view(kExpectedB01Pair1) &&
           parse_uuid(kExpectedUuid2, &(*uuids)[0]) &&
           parse_uuid(kExpectedUuid3, &(*uuids)[1]);
  }
  return false;
}

bool split_exact_list(const char *text, std::array<std::string_view, 4> *items,
                      std::size_t *count) noexcept {
  if (text == nullptr || items == nullptr || count == nullptr) return false;
  const std::string_view value(text);
  if (value.empty()) return false;
  std::size_t begin = 0;
  std::size_t used = 0;
  while (begin <= value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? value.size() : comma;
    if (end == begin || used == items->size()) return false;
    (*items)[used++] = value.substr(begin, end - begin);
    if (comma == std::string_view::npos) break;
    begin = comma + 1;
  }
  if (used != 2 && used != 4) return false;
  *count = used;
  return true;
}

bool parse_budget(const char *text, std::uint64_t *bytes) noexcept {
  if (bytes == nullptr) return false;
  if (text == nullptr || *text == '\0') {
    *bytes = kMaxOwnerBudgetMiB * kMiB;
    return true;
  }
  const char *end = text + std::strlen(text);
  std::uint64_t mib = 0;
  const auto parsed = std::from_chars(text, end, mib, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != end || mib == 0 ||
      mib > kMaxOwnerBudgetMiB || mib > std::numeric_limits<std::uint64_t>::max() / kMiB) {
    return false;
  }
  *bytes = mib * kMiB;
  return true;
}

bool api_complete(const RuntimeInitApi &api) noexcept {
  const auto &exec = api.execution;
  const auto &mem = api.memory;
  return api.init != nullptr && api.device_count != nullptr &&
         api.device_get != nullptr && api.device_get_uuid != nullptr &&
         api.device_get_attribute != nullptr &&
         api.primary_context_retain != nullptr &&
         api.primary_context_release != nullptr && api.context_push != nullptr &&
         api.context_pop != nullptr && api.allocation_granularity != nullptr &&
         api.memory_info != nullptr &&
         exec.abi_version == V200_EXECUTION_ABI_VERSION &&
         exec.struct_size >= sizeof(v200_execution_driver_api_v1) &&
         exec.ctx_push_current != nullptr && exec.ctx_pop_current != nullptr &&
         exec.event_create != nullptr && exec.event_destroy != nullptr &&
         exec.event_record != nullptr && exec.event_query != nullptr &&
         exec.event_synchronize != nullptr && exec.stream_synchronize != nullptr &&
         exec.stream_wait_event != nullptr &&
         mem.abi_version == V200_MEMORY_ABI_VERSION &&
         mem.struct_size >= sizeof(v200_memory_driver_api) &&
         mem.mem_address_reserve != nullptr && mem.mem_address_free != nullptr &&
         mem.mem_create != nullptr && mem.mem_release != nullptr &&
         mem.mem_map != nullptr && mem.mem_unmap != nullptr &&
         mem.mem_set_access != nullptr && mem.mem_get_granularity != nullptr &&
         api.register_owner != nullptr && api.attach_frontdoor != nullptr &&
         api.detach_frontdoor != nullptr;
}

execution::Status execution_status(int status, int driver_result = CUDA_SUCCESS) noexcept {
  if (status == V200_EXECUTION_OK) return execution::Status::success();
  execution::Status converted{};
  if (status < V200_EXECUTION_OK || status > V200_EXECUTION_CLOSED) {
    converted.code = execution::ErrorCode::driver_failure;
  } else {
    converted.code = static_cast<execution::ErrorCode>(status);
  }
  converted.driver_result = static_cast<CUresult>(driver_result);
  return converted;
}

class CCompletionProvider final : public execution::CompletionProvider {
 public:
  explicit CCompletionProvider(v200_execution_handle *handle) : handle_(handle) {}

  execution::Status retain(execution::CompletionToken token) noexcept override {
    return execution_status(v200_execution_retain(handle_, {token.slot, token.generation}));
  }
  execution::QueryResult is_complete(execution::CompletionToken token) noexcept override {
    int complete = 0;
    int driver_result = CUDA_SUCCESS;
    const int status = v200_execution_is_complete(
        handle_, {token.slot, token.generation}, &complete, &driver_result);
    return {execution_status(status, driver_result), complete != 0};
  }
  execution::Status wait(execution::CompletionToken token) noexcept override {
    int driver_result = CUDA_SUCCESS;
    const int status = v200_execution_wait(
        handle_, {token.slot, token.generation}, &driver_result);
    return execution_status(status, driver_result);
  }
  execution::Status release(execution::CompletionToken token) noexcept override {
    return execution_status(v200_execution_release(
        handle_, {token.slot, token.generation}));
  }

 private:
  v200_execution_handle *handle_{};
};

class NativeBindings final {
 public:
  using Init = CUresult(CUDAAPI *)(unsigned int);
  using DeviceCount = CUresult(CUDAAPI *)(int *);
  using DeviceGet = CUresult(CUDAAPI *)(CUdevice *, int);
  using DeviceGetUuid = CUresult(CUDAAPI *)(CUuuid *, CUdevice);
  using DeviceGetAttribute = CUresult(CUDAAPI *)(int *, CUdevice_attribute, CUdevice);
  using ContextRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using ContextRelease = CUresult(CUDAAPI *)(CUdevice);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);
  using Granularity = CUresult(CUDAAPI *)(size_t *, const CUmemAllocationProp *,
                                          CUmemAllocationGranularity_flags);
  using MemoryInfo = CUresult(CUDAAPI *)(size_t *, size_t *);
  using EventCreate = CUresult(CUDAAPI *)(CUevent *, unsigned int);
  using EventDestroy = CUresult(CUDAAPI *)(CUevent);
  using EventRecord = CUresult(CUDAAPI *)(CUevent, CUstream);
  using EventQuery = CUresult(CUDAAPI *)(CUevent);
  using EventSynchronize = CUresult(CUDAAPI *)(CUevent);
  using StreamSynchronize = CUresult(CUDAAPI *)(CUstream);
  using StreamWaitEvent = CUresult(CUDAAPI *)(CUstream, CUevent, unsigned int);
  using AddressReserve = CUresult(CUDAAPI *)(CUdeviceptr *, size_t, size_t,
                                             CUdeviceptr, unsigned long long);
  using AddressFree = CUresult(CUDAAPI *)(CUdeviceptr, size_t);
  using MemCreate = CUresult(CUDAAPI *)(CUmemGenericAllocationHandle *, size_t,
                                        const CUmemAllocationProp *, unsigned long long);
  using MemRelease = CUresult(CUDAAPI *)(CUmemGenericAllocationHandle);
  using MemMap = CUresult(CUDAAPI *)(CUdeviceptr, size_t, size_t,
                                     CUmemGenericAllocationHandle, unsigned long long);
  using MemUnmap = CUresult(CUDAAPI *)(CUdeviceptr, size_t);
  using MemSetAccess = CUresult(CUDAAPI *)(CUdeviceptr, size_t,
                                           const CUmemAccessDesc *, size_t);

  Init init{};
  DeviceCount device_count{};
  DeviceGet device_get{};
  DeviceGetUuid device_get_uuid{};
  DeviceGetAttribute device_get_attribute{};
  ContextRetain primary_retain{};
  ContextRelease primary_release{};
  ContextPush context_push{};
  ContextPop context_pop{};
  Granularity allocation_granularity{};
  MemoryInfo memory_info{};
  EventCreate event_create{};
  EventDestroy event_destroy{};
  EventRecord event_record{};
  EventRecord event_record_ptsz{};
  EventQuery event_query{};
  EventSynchronize event_synchronize{};
  StreamSynchronize stream_synchronize{};
  StreamSynchronize stream_synchronize_ptsz{};
  StreamWaitEvent stream_wait_event{};
  StreamWaitEvent stream_wait_event_ptsz{};
  AddressReserve mem_address_reserve{};
  AddressFree mem_address_free{};
  MemCreate mem_create{};
  MemRelease mem_release{};
  MemMap mem_map{};
  MemUnmap mem_unmap{};
  MemSetAccess mem_set_access{};

  bool load() noexcept {
    auto resolve = [](const char *name) {
      return v200_frontdoor_resolve_driver_symbol(name);
    };
    auto lookup = [&](const char *primary, const char *fallback = nullptr) {
      void *symbol = resolve(primary);
      return symbol == nullptr && fallback != nullptr ? resolve(fallback) : symbol;
    };
    init = reinterpret_cast<Init>(lookup("cuInit"));
    device_count = reinterpret_cast<DeviceCount>(lookup("cuDeviceGetCount"));
    device_get = reinterpret_cast<DeviceGet>(lookup("cuDeviceGet"));
    device_get_uuid = reinterpret_cast<DeviceGetUuid>(
        lookup("cuDeviceGetUuid_v2", "cuDeviceGetUuid"));
    device_get_attribute = reinterpret_cast<DeviceGetAttribute>(lookup("cuDeviceGetAttribute"));
    primary_retain = reinterpret_cast<ContextRetain>(lookup("cuDevicePrimaryCtxRetain"));
    primary_release = reinterpret_cast<ContextRelease>(
        lookup("cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"));
    context_push = reinterpret_cast<ContextPush>(
        lookup("cuCtxPushCurrent_v2", "cuCtxPushCurrent"));
    context_pop = reinterpret_cast<ContextPop>(
        lookup("cuCtxPopCurrent_v2", "cuCtxPopCurrent"));
    allocation_granularity = reinterpret_cast<Granularity>(
        lookup("cuMemGetAllocationGranularity"));
    memory_info = reinterpret_cast<MemoryInfo>(
        lookup("cuMemGetInfo_v2", "cuMemGetInfo"));
    event_create = reinterpret_cast<EventCreate>(lookup("cuEventCreate"));
    event_destroy = reinterpret_cast<EventDestroy>(
        lookup("cuEventDestroy_v2", "cuEventDestroy"));
    event_record = reinterpret_cast<EventRecord>(lookup("cuEventRecord"));
    event_record_ptsz = reinterpret_cast<EventRecord>(lookup("cuEventRecord_ptsz"));
    event_query = reinterpret_cast<EventQuery>(lookup("cuEventQuery"));
    event_synchronize = reinterpret_cast<EventSynchronize>(lookup("cuEventSynchronize"));
    stream_synchronize = reinterpret_cast<StreamSynchronize>(lookup("cuStreamSynchronize"));
    stream_synchronize_ptsz = reinterpret_cast<StreamSynchronize>(
        lookup("cuStreamSynchronize_ptsz"));
    stream_wait_event = reinterpret_cast<StreamWaitEvent>(lookup("cuStreamWaitEvent"));
    stream_wait_event_ptsz = reinterpret_cast<StreamWaitEvent>(
        lookup("cuStreamWaitEvent_ptsz"));
    mem_address_reserve = reinterpret_cast<AddressReserve>(lookup("cuMemAddressReserve"));
    mem_address_free = reinterpret_cast<AddressFree>(lookup("cuMemAddressFree"));
    mem_create = reinterpret_cast<MemCreate>(lookup("cuMemCreate"));
    mem_release = reinterpret_cast<MemRelease>(lookup("cuMemRelease"));
    mem_map = reinterpret_cast<MemMap>(lookup("cuMemMap"));
    mem_unmap = reinterpret_cast<MemUnmap>(lookup("cuMemUnmap"));
    mem_set_access = reinterpret_cast<MemSetAccess>(lookup("cuMemSetAccess"));
    return init && device_count && device_get && device_get_uuid &&
           device_get_attribute && primary_retain && primary_release &&
           context_push && context_pop && allocation_granularity &&
           memory_info &&
           event_create && event_destroy && event_record && event_record_ptsz && event_query &&
           event_synchronize && stream_synchronize && stream_wait_event &&
           stream_synchronize_ptsz && stream_wait_event_ptsz &&
           mem_address_reserve && mem_address_free && mem_create && mem_release &&
           mem_map && mem_unmap && mem_set_access;
  }

  static NativeBindings &self(void *data) noexcept {
    return *static_cast<NativeBindings *>(data);
  }
  static bool per_thread_stream(CUstream stream) noexcept {
    return reinterpret_cast<std::uintptr_t>(stream) == 2;
  }

  static int exec_push(void *data, void *context) {
    auto &d = self(data);
    return static_cast<int>(d.context_push(static_cast<CUcontext>(context)));
  }
  static int exec_pop(void *data, void **out) {
    auto &d = self(data);
    CUcontext context = nullptr;
    const CUresult status = d.context_pop(&context);
    if (out != nullptr) *out = context;
    return static_cast<int>(status);
  }
  static int exec_event_create(void *data, void **out, unsigned flags) {
    auto &d = self(data);
    CUevent event = nullptr;
    const CUresult status = d.event_create(&event, flags);
    if (out != nullptr) *out = event;
    return static_cast<int>(status);
  }
  static int exec_event_destroy(void *data, void *event) {
    return static_cast<int>(self(data).event_destroy(static_cast<CUevent>(event)));
  }
  static int exec_event_record(void *data, void *event, void *stream) {
    auto &d = self(data);
    const auto native_stream = static_cast<CUstream>(stream);
    auto fn = per_thread_stream(native_stream) ? d.event_record_ptsz : d.event_record;
    return static_cast<int>(fn == nullptr ? CUDA_ERROR_NOT_SUPPORTED
                                          : fn(static_cast<CUevent>(event), native_stream));
  }
  static int exec_event_query(void *data, void *event) {
    return static_cast<int>(self(data).event_query(static_cast<CUevent>(event)));
  }
  static int exec_event_synchronize(void *data, void *event) {
    return static_cast<int>(self(data).event_synchronize(static_cast<CUevent>(event)));
  }
  static int exec_stream_synchronize(void *data, void *stream) {
    auto &d = self(data);
    const auto native_stream = static_cast<CUstream>(stream);
    auto fn = per_thread_stream(native_stream) ? d.stream_synchronize_ptsz
                                               : d.stream_synchronize;
    return static_cast<int>(fn == nullptr ? CUDA_ERROR_NOT_SUPPORTED
                                          : fn(native_stream));
  }
  static int exec_stream_wait_event(void *data, void *stream, void *event,
                                    unsigned flags) {
    auto &d = self(data);
    const auto native_stream = static_cast<CUstream>(stream);
    auto fn = per_thread_stream(native_stream) ? d.stream_wait_event_ptsz
                                               : d.stream_wait_event;
    return static_cast<int>(fn == nullptr ? CUDA_ERROR_NOT_SUPPORTED
                                          : fn(native_stream, static_cast<CUevent>(event), flags));
  }

  static CUresult CUDAAPI mem_address_reserve_call(
      void *data, CUdeviceptr *ptr, size_t size, size_t alignment,
      CUdeviceptr requested, unsigned long long flags) {
    return self(data).mem_address_reserve(ptr, size, alignment, requested, flags);
  }
  static CUresult CUDAAPI mem_address_free_call(void *data, CUdeviceptr ptr, size_t size) {
    return self(data).mem_address_free(ptr, size);
  }
  static CUresult CUDAAPI mem_create_call(
      void *data, CUmemGenericAllocationHandle *handle, size_t size,
      const CUmemAllocationProp *prop, unsigned long long flags) {
    return self(data).mem_create(handle, size, prop, flags);
  }
  static CUresult CUDAAPI mem_release_call(void *data, CUmemGenericAllocationHandle handle) {
    return self(data).mem_release(handle);
  }
  static CUresult CUDAAPI mem_map_call(
      void *data, CUdeviceptr ptr, size_t size, size_t offset,
      CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return self(data).mem_map(ptr, size, offset, handle, flags);
  }
  static CUresult CUDAAPI mem_unmap_call(void *data, CUdeviceptr ptr, size_t size) {
    return self(data).mem_unmap(ptr, size);
  }
  static CUresult CUDAAPI mem_set_access_call(
      void *data, CUdeviceptr ptr, size_t size,
      const CUmemAccessDesc *descs, size_t count) {
    return self(data).mem_set_access(ptr, size, descs, count);
  }
  static CUresult CUDAAPI mem_get_granularity_call(
      void *data, size_t *granularity, const CUmemAllocationProp *prop,
      CUmemAllocationGranularity_flags flags) {
    return self(data).allocation_granularity(granularity, prop, flags);
  }

  RuntimeInitApi make_api() noexcept {
    RuntimeInitApi api{};
    api.init = init;
    api.device_count = device_count;
    api.device_get = device_get;
    api.device_get_uuid = device_get_uuid;
    api.device_get_attribute = device_get_attribute;
    api.primary_context_retain = primary_retain;
    api.primary_context_release = primary_release;
    api.context_push = context_push;
    api.context_pop = context_pop;
    api.allocation_granularity = allocation_granularity;
    api.memory_info = memory_info;
    api.execution.abi_version = V200_EXECUTION_ABI_VERSION;
    api.execution.struct_size = sizeof(api.execution);
    api.execution.user_data = this;
    api.execution.ctx_push_current = exec_push;
    api.execution.ctx_pop_current = exec_pop;
    api.execution.event_create = exec_event_create;
    api.execution.event_destroy = exec_event_destroy;
    api.execution.event_record = exec_event_record;
    api.execution.event_query = exec_event_query;
    api.execution.event_synchronize = exec_event_synchronize;
    api.execution.stream_synchronize = exec_stream_synchronize;
    api.execution.stream_wait_event = exec_stream_wait_event;
    api.memory.abi_version = V200_MEMORY_ABI_VERSION;
    api.memory.struct_size = sizeof(api.memory);
    api.memory.user_data = this;
    api.memory.mem_address_reserve = mem_address_reserve_call;
    api.memory.mem_address_free = mem_address_free_call;
    api.memory.mem_create = mem_create_call;
    api.memory.mem_release = mem_release_call;
    api.memory.mem_map = mem_map_call;
    api.memory.mem_unmap = mem_unmap_call;
    api.memory.mem_set_access = mem_set_access_call;
    api.memory.mem_get_granularity = mem_get_granularity_call;
    api.register_owner = v200_execution_register_owner;
    api.attach_frontdoor = v200_frontdoor_attach_execution;
    api.detach_frontdoor = v200_frontdoor_detach_execution;
    return api;
  }
};

bool pair_mode_requested() noexcept {
  const char *mode = std::getenv("V200_MODE");
  return mode != nullptr && (std::strcmp(mode, "pair") == 0 || std::strcmp(mode, "2") == 0);
}

}  // namespace

class RuntimeInitializer::Impl final {
 public:
  explicit Impl(memory::MemoryManager &manager) : manager_(manager) {}

  ~Impl() { (void)cleanup_for_destruction(); }

  CUresult initialize(const RuntimeInitInputs &inputs,
                      const RuntimeInitApi &api) noexcept {
    try {
      return initialize_impl(inputs, api);
    } catch (...) {
      return fail(CUDA_ERROR_OUT_OF_MEMORY);
    }
  }

  CUresult initialize_impl(const RuntimeInitInputs &inputs,
                           const RuntimeInitApi &api) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (attempted_) return result_;
    attempted_ = true;
    api_ = api;

    std::array<CUuuid, 2> wanted{};
    if (!expected_input(inputs, &wanted)) return fail(CUDA_ERROR_INVALID_VALUE);
    if (!parse_budget(inputs.owner0_budget_mib, &budgets_[0]) ||
        !parse_budget(inputs.owner1_budget_mib, &budgets_[1])) {
      return fail(CUDA_ERROR_INVALID_VALUE);
    }
    if (!api_complete(api_)) return fail(CUDA_ERROR_NOT_SUPPORTED);

    CUresult status = api_.init(0);
    if (status != CUDA_SUCCESS) return fail(status);
    int count = 0;
    status = api_.device_count(&count);
    if (status != CUDA_SUCCESS) return fail(status);
    if (count <= 0) return fail(CUDA_ERROR_NO_DEVICE);

    bool found[2]{false, false};
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      CUdevice device = -1;
      CUuuid uuid{};
      status = api_.device_get(&device, ordinal);
      if (status == CUDA_SUCCESS) status = api_.device_get_uuid(&uuid, device);
      if (status != CUDA_SUCCESS) return fail(status);
      for (std::size_t owner = 0; owner < 2; ++owner) {
        if (std::memcmp(uuid.bytes, wanted[owner].bytes, sizeof(uuid.bytes)) != 0) continue;
        if (found[owner]) return fail(CUDA_ERROR_INVALID_DEVICE);
        int major = 0;
        int minor = 0;
        int vmm = 0;
        status = api_.device_get_attribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        if (status == CUDA_SUCCESS) {
          status = api_.device_get_attribute(
              &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        }
        if (status == CUDA_SUCCESS) {
          status = api_.device_get_attribute(
              &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
        }
        if (status != CUDA_SUCCESS) return fail(status);
        if (major != 7 || minor != 0 || vmm == 0) return fail(CUDA_ERROR_NOT_SUPPORTED);
        devices_[owner] = device;
        uuids_[owner] = uuid;
        found[owner] = true;
      }
    }
    if (!found[0] || !found[1] || devices_[0] == devices_[1]) {
      return fail(CUDA_ERROR_INVALID_DEVICE);
    }

    for (std::size_t owner = 0; owner < 2; ++owner) {
      status = api_.primary_context_retain(&contexts_[owner], devices_[owner]);
      if (status != CUDA_SUCCESS) return fail(status);
      retained_[owner] = true;
    }

    for (std::size_t owner = 0; owner < 2; ++owner) {
      status = api_.context_push(contexts_[owner]);
      if (status != CUDA_SUCCESS) return fail(status);
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = devices_[owner];
      status = api_.allocation_granularity(
          &granularities_[owner], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      CUcontext popped = nullptr;
      const CUresult pop_status = api_.context_pop(&popped);
      if (pop_status != CUDA_SUCCESS) {
        unsafe_context_state_ = true;
        return fail(pop_status);
      }
      if (popped != contexts_[owner]) {
        unsafe_context_state_ = true;
        return fail(CUDA_ERROR_INVALID_CONTEXT);
      }
      if (status != CUDA_SUCCESS) return fail(status);
      if (granularities_[owner] == 0) {
        return fail(CUDA_ERROR_INVALID_CONTEXT);
      }

      size_t physical_free = 0;
      size_t physical_total = 0;
      status = api_.context_push(contexts_[owner]);
      if (status != CUDA_SUCCESS) return fail(status);
      status = api_.memory_info(&physical_free, &physical_total);
      CUcontext memory_popped = nullptr;
      const CUresult memory_pop_status = api_.context_pop(&memory_popped);
      if (memory_pop_status != CUDA_SUCCESS) {
        unsafe_context_state_ = true;
        return fail(memory_pop_status);
      }
      if (memory_popped != contexts_[owner]) {
        unsafe_context_state_ = true;
        return fail(CUDA_ERROR_INVALID_CONTEXT);
      }
      if (status != CUDA_SUCCESS) return fail(status);
      if (physical_total == 0 || physical_free > physical_total) {
        return fail(CUDA_ERROR_INVALID_VALUE);
      }
      const std::uint64_t usable_free = physical_free > kPhysicalReserveBytes
                                            ? physical_free - kPhysicalReserveBytes
                                            : 0;
      budgets_[owner] = std::min<std::uint64_t>(budgets_[owner], usable_free);
      if (budgets_[owner] == 0) return fail(CUDA_ERROR_OUT_OF_MEMORY);
    }

    handle_ = v200_execution_create(&api_.execution);
    if (handle_ == nullptr) return fail(CUDA_ERROR_OUT_OF_MEMORY);
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const int registered = api_.register_owner(
          handle_, inputs.logical_device_id,
          static_cast<std::uint32_t>(owner), contexts_[owner]);
      if (registered != V200_EXECUTION_OK) return fail(CUDA_ERROR_NOT_SUPPORTED);
    }
    const int attached = api_.attach_frontdoor(
        handle_, inputs.logical_device_id, kAnchorOwner);
    if (attached != V200_EXECUTION_OK) return fail(CUDA_ERROR_NOT_SUPPORTED);
    attached_ = true;

    v200_memory_pair_profile profile{};
    profile.abi_version = V200_MEMORY_ABI_VERSION;
    profile.struct_size = sizeof(profile);
    profile.owner_device[0] = devices_[0];
    profile.owner_device[1] = devices_[1];
    profile.owner_granularity[0] = granularities_[0];
    profile.owner_granularity[1] = granularities_[1];
    profile.owner_budget_bytes[0] = budgets_[0];
    profile.owner_budget_bytes[1] = budgets_[1];
    profile.default_owner = devices_[0];
    profile.pair_access_qualified = 1;
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const std::string text = format_uuid(uuids_[owner]);
      std::snprintf(profile.owner_uuid[owner], sizeof(profile.owner_uuid[owner]), "%s", text.c_str());
    }
    std::snprintf(profile.qualification_id, sizeof(profile.qualification_id), "%s",
                  inputs.b01_evidence_id);

    provider_ = std::make_unique<CCompletionProvider>(handle_);
    status = manager_.configure(profile, api_.memory);
    if (status != CUDA_SUCCESS) return fail(status);
    memory_configured_ = true;
    manager_.set_completion_provider(provider_.get());
    result_ = CUDA_SUCCESS;
    return result_;
  }

  bool initialized() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return attempted_ && result_ == CUDA_SUCCESS;
  }

  bool cleanup_for_destruction() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (memory_configured_) {
      v200_memory_usage_info usage{};
      usage.abi_version = V200_MEMORY_ABI_VERSION;
      usage.struct_size = sizeof(usage);
      if (manager_.get_usage(&usage) != CUDA_SUCCESS || usage.live_allocation_count != 0 ||
          usage.live_reservation_count != 0 || usage.cleanup_pending_count != 0 ||
          usage.owner_used_bytes[0] != 0 || usage.owner_used_bytes[1] != 0) {
        return false;
      }
      manager_.set_completion_provider(nullptr);
    }
    if (attached_) {
      api_.detach_frontdoor(handle_);
      attached_ = false;
    }
    bool safe_to_release = !unsafe_context_state_;
    if (handle_ != nullptr) {
      if (v200_execution_shutdown(handle_) != V200_EXECUTION_OK) {
        safe_to_release = false;
      } else {
        v200_execution_destroy(handle_);
        handle_ = nullptr;
      }
    }
    if (safe_to_release) {
      for (std::size_t reverse = 2; reverse > 0; --reverse) {
        const std::size_t owner = reverse - 1;
        if (!retained_[owner]) continue;
        if (api_.primary_context_release(devices_[owner]) == CUDA_SUCCESS) {
          retained_[owner] = false;
        } else {
          safe_to_release = false;
        }
      }
    }
    if (!safe_to_release) return false;
    provider_.reset();
    return true;
  }

 private:
  CUresult fail(CUresult status) noexcept {
    result_ = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
    const bool clean = cleanup_partial();
    (void)clean;
    return result_;
  }

  bool cleanup_partial() noexcept {
    if (attached_) {
      api_.detach_frontdoor(handle_);
      attached_ = false;
    }
    bool safe_to_release = !unsafe_context_state_;
    if (handle_ != nullptr) {
      if (v200_execution_shutdown(handle_) != V200_EXECUTION_OK) {
        safe_to_release = false;
      } else {
        v200_execution_destroy(handle_);
        handle_ = nullptr;
      }
    }
    if (safe_to_release) {
      for (std::size_t reverse = 2; reverse > 0; --reverse) {
        const std::size_t owner = reverse - 1;
        if (!retained_[owner]) continue;
        if (api_.primary_context_release(devices_[owner]) == CUDA_SUCCESS) {
          retained_[owner] = false;
        } else {
          safe_to_release = false;
        }
      }
    }
    provider_.reset();
    return safe_to_release;
  }

  memory::MemoryManager &manager_;
  mutable std::mutex mutex_;
  RuntimeInitApi api_{};
  bool attempted_{false};
  bool memory_configured_{false};
  bool attached_{false};
  bool unsafe_context_state_{false};
  CUresult result_{CUDA_ERROR_NOT_INITIALIZED};
  std::array<CUdevice, 2> devices_{{-1, -1}};
  std::array<CUuuid, 2> uuids_{};
  std::array<CUcontext, 2> contexts_{};
  std::array<bool, 2> retained_{{false, false}};
  std::array<size_t, 2> granularities_{{0, 0}};
  std::array<std::uint64_t, 2> budgets_{{0, 0}};
  v200_execution_handle *handle_{};
  std::unique_ptr<CCompletionProvider> provider_;
};

RuntimeInitializer::RuntimeInitializer(memory::MemoryManager &manager)
    : impl_(std::make_unique<Impl>(manager)) {}

RuntimeInitializer::~RuntimeInitializer() {
  if (impl_ && !impl_->cleanup_for_destruction()) {
    // Active allocations, failed event retirement, or an unreleasable primary
    // context make teardown unsafe. Keep the entire provider/handle graph alive.
    (void)impl_.release();
  }
}

CUresult RuntimeInitializer::initialize(const RuntimeInitInputs &inputs,
                                        const RuntimeInitApi &api) noexcept {
  return impl_->initialize(inputs, api);
}

bool RuntimeInitializer::initialized() const noexcept {
  return impl_->initialized();
}

}  // namespace v200::core

namespace {

struct ProcessRuntime final {
  explicit ProcessRuntime(v200::memory::MemoryManager &pair_manager)
      : initializer(pair_manager) {}

  CUresult run(std::uint32_t pair_index) {
    if (!v200::core::pair_mode_requested()) return CUDA_ERROR_INVALID_VALUE;
    if (pair_index > 1) return CUDA_ERROR_INVALID_VALUE;

    std::array<std::string_view, 4> ordered_uuids{};
    std::size_t uuid_count = 0;
    if (!v200::core::split_exact_list(
            std::getenv("V200_DEVICE_UUIDS"), &ordered_uuids, &uuid_count) ||
        (uuid_count == 2 && pair_index != 0)) {
      return CUDA_ERROR_INVALID_VALUE;
    }

    const std::size_t offset = static_cast<std::size_t>(pair_index) * 2;
    std::string ordered_pair;
    ordered_pair.reserve(82);
    ordered_pair.append(ordered_uuids[offset]);
    ordered_pair.push_back(',');
    ordered_pair.append(ordered_uuids[offset + 1]);

    const char *single_evidence = std::getenv("V200_B01_EVIDENCE_ID");
    const char *pair_evidence = std::getenv("V200_B01_EVIDENCE_IDS");
    std::string evidence_id;
    if (uuid_count == 2) {
      if (pair_evidence != nullptr && pair_evidence[0] != '\0') {
        return CUDA_ERROR_INVALID_VALUE;
      }
      if (single_evidence == nullptr || single_evidence[0] == '\0') {
        return CUDA_ERROR_INVALID_VALUE;
      }
      evidence_id = single_evidence;
    } else {
      if (single_evidence != nullptr && single_evidence[0] != '\0') {
        return CUDA_ERROR_INVALID_VALUE;
      }
      std::array<std::string_view, 4> evidence_ids{};
      std::size_t evidence_count = 0;
      if (!v200::core::split_exact_list(pair_evidence, &evidence_ids,
                                        &evidence_count) || evidence_count != 2) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      evidence_id = evidence_ids[pair_index];
    }

    if (!bindings.load()) return CUDA_ERROR_NOT_SUPPORTED;
    api = bindings.make_api();
    const v200::core::RuntimeInitInputs inputs{
        ordered_pair.c_str(),
        evidence_id.c_str(),
        std::getenv("V200_OWNER0_BUDGET_MIB"),
        std::getenv("V200_OWNER1_BUDGET_MIB"),
        static_cast<std::uint64_t>(pair_index) + 1};
    return initializer.initialize(inputs, api);
  }

  v200::core::NativeBindings bindings;
  v200::core::RuntimeInitApi api{};
  v200::core::RuntimeInitializer initializer;
};

}  // namespace

extern "C" V200_CORE_API CUresult CUDAAPI v200_initialize_profile(void) {
  return v200_initialize_profile_pair(0);
}

extern "C" V200_CORE_API CUresult CUDAAPI
v200_initialize_profile_pair(std::uint32_t pair_index) {
  if (pair_index > 1) return CUDA_ERROR_INVALID_VALUE;
  static std::mutex runtime_mutex;
  static std::array<ProcessRuntime *, 2> runtimes{};
  std::lock_guard<std::mutex> lock(runtime_mutex);
  try {
    if (runtimes[pair_index] == nullptr) {
      auto *manager = v200::memory::manager_for_pair(pair_index);
      if (manager == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
      runtimes[pair_index] = new (std::nothrow) ProcessRuntime(*manager);
      if (runtimes[pair_index] == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return runtimes[pair_index]->run(pair_index);
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
