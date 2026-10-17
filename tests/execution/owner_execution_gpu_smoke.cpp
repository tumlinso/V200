// Finite optional hardware check for the real Driver API completion adapter.
// This executable is never registered with CTest and refuses to run without
// the B02 bounded-runner environment, expected ordered UUIDs, and --execute.
#include "v200/execution/owner_execution.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace v200::execution;

namespace {

constexpr LogicalDeviceId kLogicalDevice = 1;
constexpr LogicalStreamId kLogicalStream = 1;
constexpr OwnerId kOwnerA = 0;
constexpr OwnerId kOwnerB = 1;
constexpr std::size_t kBytesPerOwner = 4096;
constexpr std::size_t kMinimumFreeHeadroom = 1024 * 1024;
unsigned cleanup_failures = 0;

void check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (name == nullptr ? "CUDA error" : name) + " " +
                           (message == nullptr ? "" : message));
}

void check(Status status, const char* operation) {
  if (status.ok()) return;
  throw std::runtime_error(std::string(operation) + " failed with adapter status " +
                           std::to_string(static_cast<unsigned>(status.code)) +
                           " (CUDA " + std::to_string(status.driver_result) + ")");
}

void cleanup(CUresult result, const char* operation) noexcept {
  if (result == CUDA_SUCCESS) return;
  ++cleanup_failures;
  std::fprintf(stderr, "cleanup failure: %s (%d)\n", operation,
               static_cast<int>(result));
}

class RealDriver final : public DriverDispatch {
 public:
  CUresult ctx_push_current(CUcontext context) noexcept override {
    return cuCtxPushCurrent(context);
  }
  CUresult ctx_pop_current(CUcontext* context) noexcept override {
    return cuCtxPopCurrent(context);
  }
  CUresult event_create(CUevent* event, unsigned flags) noexcept override {
    return cuEventCreate(event, flags);
  }
  CUresult event_destroy(CUevent event) noexcept override {
    return cuEventDestroy(event);
  }
  CUresult event_record(CUevent event, CUstream stream) noexcept override {
    return cuEventRecord(event, stream);
  }
  CUresult event_query(CUevent event) noexcept override {
    return cuEventQuery(event);
  }
  CUresult event_synchronize(CUevent event) noexcept override {
    return cuEventSynchronize(event);
  }
  CUresult stream_synchronize(CUstream stream) noexcept override {
    return cuStreamSynchronize(stream);
  }
  CUresult stream_wait_event(CUstream stream, CUevent event,
                             unsigned flags) noexcept override {
    return cuStreamWaitEvent(stream, event, flags);
  }
};

class ContextScope {
 public:
  explicit ContextScope(CUcontext context) : context_(context) {
    check(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
    active_ = true;
  }
  ~ContextScope() {
    if (active_) {
      CUcontext popped = nullptr;
      cleanup(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
      if (popped != context_) ++cleanup_failures;
    }
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;

  void finish() {
    if (!active_) return;
    CUcontext popped = nullptr;
    check(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
    active_ = false;
    if (popped != context_) throw std::runtime_error("unexpected popped CUDA context");
  }

 private:
  CUcontext context_{};
  bool active_{false};
};

class PrimaryContexts {
 public:
  explicit PrimaryContexts(const std::array<CUdevice, 2>& devices)
      : devices_(devices) {
    try {
      for (; retained_ < contexts_.size(); ++retained_) {
        check(cuDevicePrimaryCtxRetain(&contexts_[retained_], devices_[retained_]),
              "cuDevicePrimaryCtxRetain");
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~PrimaryContexts() { release(); }
  PrimaryContexts(const PrimaryContexts&) = delete;
  PrimaryContexts& operator=(const PrimaryContexts&) = delete;

  CUcontext operator[](std::size_t i) const { return contexts_[i]; }
  CUdevice device(std::size_t i) const { return devices_[i]; }

  void release() noexcept {
    if (cleanup_failures != 0) {
      std::fprintf(stderr,
                   "preserving retained primary contexts after unsafe cleanup\n");
      retained_ = 0;
      return;
    }
    while (retained_ != 0) {
      --retained_;
      cleanup(cuDevicePrimaryCtxRelease(devices_[retained_]),
              "cuDevicePrimaryCtxRelease");
    }
  }

 private:
  std::array<CUdevice, 2> devices_{};
  std::array<CUcontext, 2> contexts_{};
  std::size_t retained_{0};
};

class OwnerResources {
 public:
  explicit OwnerResources(const PrimaryContexts& contexts) : contexts_(contexts) {
    try {
      for (std::size_t i = 0; i < 2; ++i) {
        ContextScope scope(contexts_[i]);
        std::size_t free_before = 0;
        std::size_t total_bytes = 0;
        check(cuMemGetInfo(&free_before, &total_bytes), "cuMemGetInfo before smoke allocation");
        if (free_before < kBytesPerOwner + kMinimumFreeHeadroom) {
          throw std::runtime_error("less than 1 MiB of bounded smoke headroom remains");
        }
        free_after_context_[i] = free_before;
        check(cuStreamCreate(&streams_[i], CU_STREAM_NON_BLOCKING),
              "cuStreamCreate");
        check(cuMemAlloc(&buffers_[i], kBytesPerOwner), "cuMemAlloc");
        allocated_[i] = true;
        check(cuMemGetInfo(&free_after_allocation_[i], &total_bytes),
              "cuMemGetInfo after smoke allocation");
        scope.finish();
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~OwnerResources() { release(); }
  OwnerResources(const OwnerResources&) = delete;
  OwnerResources& operator=(const OwnerResources&) = delete;

  CUstream stream(std::size_t i) const { return streams_[i]; }
  CUdeviceptr buffer(std::size_t i) const { return buffers_[i]; }
  std::size_t free_after_context(std::size_t i) const { return free_after_context_[i]; }
  std::size_t free_after_allocation(std::size_t i) const {
    return free_after_allocation_[i];
  }

  void release() noexcept {
    for (std::size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i] == nullptr && !allocated_[i]) continue;
      if (cuCtxPushCurrent(contexts_[i]) != CUDA_SUCCESS) {
        ++cleanup_failures;
        // Preserve the allocation and stream if the owning context cannot be
        // made current; freeing them here could race unknown work.
        continue;
      }
      bool drained = streams_[i] == nullptr ||
                     cuStreamSynchronize(streams_[i]) == CUDA_SUCCESS;
      if (!drained) {
        ++cleanup_failures;
        std::fprintf(stderr, "cleanup failure: owner stream drain\n");
      }
      if (drained && allocated_[i]) {
        cleanup(cuMemFree(buffers_[i]), "cuMemFree");
        allocated_[i] = false;
      }
      if (drained && streams_[i] != nullptr) {
        cleanup(cuStreamDestroy(streams_[i]), "cuStreamDestroy");
        streams_[i] = nullptr;
      }
      CUcontext popped = nullptr;
      const CUresult pop_result = cuCtxPopCurrent(&popped);
      cleanup(pop_result, "cleanup context pop");
      if (pop_result == CUDA_SUCCESS && popped != contexts_[i]) ++cleanup_failures;
    }
  }

 private:
  const PrimaryContexts& contexts_;
  std::array<CUstream, 2> streams_{};
  std::array<CUdeviceptr, 2> buffers_{};
  std::array<bool, 2> allocated_{};
  std::array<std::size_t, 2> free_after_context_{};
  std::array<std::size_t, 2> free_after_allocation_{};
};

class PinnedBuffer {
 public:
  PinnedBuffer(CUcontext context, std::size_t bytes) : size_(bytes) {
    ContextScope scope(context);
    check(cuMemHostAlloc(reinterpret_cast<void**>(&data_), size_,
                         CU_MEMHOSTALLOC_PORTABLE),
          "cuMemHostAlloc portable staging");
    scope.finish();
  }
  ~PinnedBuffer() {
    if (data_ != nullptr) cleanup(cuMemFreeHost(data_), "cuMemFreeHost");
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  unsigned char* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  unsigned char* data_{nullptr};
  std::size_t size_{};
};

std::string uuid(CUdevice device) {
  CUuuid raw{};
  check(cuDeviceGetUuid(&raw, device), "cuDeviceGetUuid");
  std::ostringstream out;
  out << "GPU-" << std::hex << std::setfill('0');
  for (unsigned i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
    out << std::setw(2)
        << static_cast<unsigned>(static_cast<unsigned char>(raw.bytes[i]));
  }
  return out.str();
}

struct Args {
  bool execute{false};
  std::string expected_a;
  std::string expected_b;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--execute") {
      args.execute = true;
    } else if ((arg == "--expect-a" || arg == "--expect-b") && i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--expect-a") args.expected_a = value;
      else args.expected_b = value;
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if (!args.execute || args.expected_a.empty() || args.expected_b.empty() ||
      args.expected_a == args.expected_b) {
    throw std::runtime_error(
        "require --execute --expect-a GPU-UUID --expect-b GPU-UUID");
  }
  return args;
}

void validate_bounded_runner(const Args& args) {
  const char* receipt_path = std::getenv("TODO_GPU_LEASE_RECEIPT");
  const char* bundle_id = std::getenv("V200_BUNDLE_ID");
  const char* cap_text = std::getenv("V200_BUNDLE_VRAM_CAP_MIB_PER_GPU");
  if (receipt_path == nullptr || *receipt_path == '\0') {
    throw std::runtime_error("missing controller receipt; use the B02 bounded runner");
  }
  std::ifstream receipt(receipt_path);
  if (!receipt.good()) throw std::runtime_error("controller receipt path is unreadable");
  if (bundle_id == nullptr || std::string(bundle_id) != "B02") {
    throw std::runtime_error("smoke must be launched under the B02 controller binding");
  }
  if (cap_text == nullptr || *cap_text == '\0') {
    throw std::runtime_error("missing B02 per-GPU allocation cap");
  }
  std::size_t consumed = 0;
  const unsigned long cap_mib = std::stoul(cap_text, &consumed);
  if (consumed != std::string(cap_text).size() || cap_mib == 0 || cap_mib > 1024) {
    throw std::runtime_error("B02 allocation cap must be in range 1..1024 MiB");
  }
  if (kBytesPerOwner > cap_mib * 1024UL * 1024UL) {
    throw std::runtime_error("smoke allocation exceeds the admitted B02 cap");
  }
  // The parent runner validates receipt contents, process ancestry, project
  // root, visible UUID set and the finite wall cap. This child check is only a
  // tripwire; receipt/environment presence alone never authorizes a run.
  if (args.expected_a.empty() || args.expected_b.empty()) {
    throw std::runtime_error("exact owner UUIDs are required");
  }
}

void require_ambient_context(CUcontext expected, const char* operation) {
  CUcontext actual = nullptr;
  check(cuCtxGetCurrent(&actual), "cuCtxGetCurrent");
  if (actual != expected) {
    throw std::runtime_error(std::string(operation) + " did not restore ambient context");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    validate_bounded_runner(args);
    check(cuInit(0), "cuInit");
    CUcontext ambient = nullptr;
    check(cuCtxGetCurrent(&ambient), "cuCtxGetCurrent");
    int count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count != 2) throw std::runtime_error("exactly two controller-visible GPUs are required");
    std::array<CUdevice, 2> devices{};
    check(cuDeviceGet(&devices[0], 0), "cuDeviceGet owner A");
    check(cuDeviceGet(&devices[1], 1), "cuDeviceGet owner B");
    const std::array<std::string, 2> actual_uuids{uuid(devices[0]), uuid(devices[1])};
    if (actual_uuids[0] != args.expected_a || actual_uuids[1] != args.expected_b) {
      throw std::runtime_error("visible ordered UUIDs differ from B02 root binding");
    }
    int peer_ab = 0;
    int peer_ba = 0;
    for (const CUdevice device : devices) {
      int major = 0;
      int minor = 0;
      check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 device), "cuDeviceGetAttribute major");
      check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                 device), "cuDeviceGetAttribute minor");
      if (major != 7 || minor != 0) {
        throw std::runtime_error("the bound V200 pair must expose sm_70 devices");
      }
    }
    check(cuDeviceCanAccessPeer(&peer_ab, devices[0], devices[1]),
          "cuDeviceCanAccessPeer A-to-B");
    check(cuDeviceCanAccessPeer(&peer_ba, devices[1], devices[0]),
          "cuDeviceCanAccessPeer B-to-A");
    if (!peer_ab || !peer_ba) throw std::runtime_error("bidirectional peer capability is required");

    PrimaryContexts contexts(devices);
    PinnedBuffer staging(contexts[0], kBytesPerOwner);
    std::fill(staging.data(), staging.data() + staging.size(), 0x00);
    OwnerResources resources(contexts);
    RealDriver driver;
    OwnerExecution execution(driver);
    check(execution.register_owner(kLogicalDevice, kOwnerA, contexts[0]),
          "register owner A");
    check(execution.register_owner(kLogicalDevice, kOwnerB, contexts[1]),
          "register owner B");
    check(execution.register_stream(kLogicalDevice, kLogicalStream, kOwnerA,
                                    resources.stream(0)), "register stream A");
    check(execution.register_stream(kLogicalDevice, kLogicalStream, kOwnerB,
                                    resources.stream(1)), "register stream B");

    // Native single-anchor work is observed and sealed into an immutable
    // prefix. The second owner then joins through the adapter's explicit
    // stream-wait path, never through a post-hoc native peer enqueue.
    {
      ContextScope scope(contexts[0]);
      check(cuMemsetD8Async(resources.buffer(0), 0x2a, kBytesPerOwner,
                           resources.stream(0)), "cuMemsetD8Async owner A");
      check(cuMemcpyDtoHAsync(staging.data(), resources.buffer(0),
                              kBytesPerOwner, resources.stream(0)),
            "cuMemcpyDtoHAsync owner A staging");
      scope.finish();
    }
    require_ambient_context(ambient, "owner A native submission");
    const Status observed = execution.observe_native_submission(
        kLogicalDevice, kLogicalStream, kOwnerA, resources.stream(0), CUDA_SUCCESS);
    if (!observed.ok()) throw std::runtime_error("observe owner A prefix failed");
    const auto anchor = execution.current_token(kLogicalDevice, kLogicalStream);
    if (!anchor.ok()) throw std::runtime_error("seal owner A prefix failed");
    require_ambient_context(ambient, "owner A prefix capture");

    const auto joined = execution.submit(
        kLogicalDevice, kLogicalStream, kOwnerB, nullptr, 0,
        [&](CUstream stream) {
          CUcontext current = nullptr;
          CUresult result = cuCtxGetCurrent(&current);
          if (result != CUDA_SUCCESS || current != contexts[1] ||
              stream != resources.stream(1)) {
            return CUDA_ERROR_INVALID_CONTEXT;
          }
          result = cuMemsetD8Async(resources.buffer(1), 0x5b,
                                   kBytesPerOwner, stream);
          if (result != CUDA_SUCCESS) return result;
          return cuMemcpyHtoDAsync(resources.buffer(1), staging.data(),
                                   kBytesPerOwner, stream);
        });
    if (!joined.status.ok() || !joined.completion.valid()) {
      if (anchor.value.valid()) (void)execution.release(anchor.value);
      throw std::runtime_error("owner B peer fan-in submission failed");
    }
    require_ambient_context(ambient, "owner B peer fan-in");
    if (!execution.wait(joined.completion).ok()) {
      (void)execution.release(joined.completion);
      if (anchor.value.valid()) (void)execution.release(anchor.value);
      throw std::runtime_error("full-prefix event wait failed");
    }
    const QueryResult complete = execution.is_complete(joined.completion);
    if (!complete.status.ok() || !complete.complete) {
      (void)execution.release(joined.completion);
      if (anchor.value.valid()) (void)execution.release(anchor.value);
      throw std::runtime_error("full-prefix token did not complete after wait");
    }

    std::array<unsigned char, kBytesPerOwner> bytes_a{};
    std::array<unsigned char, kBytesPerOwner> bytes_b{};
    if (!std::all_of(staging.data(), staging.data() + staging.size(),
                     [](unsigned char value) { return value == 0x2a; })) {
      (void)execution.release(joined.completion);
      if (anchor.value.valid()) (void)execution.release(anchor.value);
      throw std::runtime_error("peer event did not order staged source data");
    }
    {
      ContextScope scope(contexts[0]);
      check(cuMemcpyDtoH(bytes_a.data(), resources.buffer(0), bytes_a.size()),
            "cuMemcpyDtoH owner A");
      scope.finish();
    }
    {
      ContextScope scope(contexts[1]);
      check(cuMemcpyDtoH(bytes_b.data(), resources.buffer(1), bytes_b.size()),
            "cuMemcpyDtoH owner B");
      scope.finish();
    }
    if (!std::all_of(bytes_a.begin(), bytes_a.end(),
                     [](unsigned char value) { return value == 0x2a; }) ||
        !std::all_of(bytes_b.begin(), bytes_b.end(),
                     [](unsigned char value) { return value == 0x5b; })) {
      (void)execution.release(joined.completion);
      if (anchor.value.valid()) (void)execution.release(anchor.value);
      throw std::runtime_error("per-owner memset/copy verification failed");
    }
    require_ambient_context(ambient, "owner data verification");

    if (anchor.value.valid() && !execution.release(anchor.value).ok()) {
      (void)execution.release(joined.completion);
      throw std::runtime_error("release anchor prefix failed");
    }
    if (!execution.release(joined.completion).ok()) {
      throw std::runtime_error("release full-prefix token failed");
    }
    if (!execution.shutdown().ok()) throw std::runtime_error("execution shutdown failed");
    require_ambient_context(ambient, "execution shutdown");
    resources.release();
    contexts.release();
    if (cleanup_failures != 0) throw std::runtime_error("owned CUDA cleanup failed");

    std::cout << "{\"status\":\"GPU_PASS\",\"qualification_only\":true"
              << ",\"bundle\":\"B02\",\"ordered_uuids\":[\""
              << actual_uuids[0] << "\",\"" << actual_uuids[1]
              << "\"],\"owner_bytes\":" << kBytesPerOwner
              << ",\"free_after_context_bytes\":["
              << resources.free_after_context(0) << ','
              << resources.free_after_context(1) << ']'
              << ",\"free_after_owner_alloc_bytes\":["
              << resources.free_after_allocation(0) << ','
              << resources.free_after_allocation(1) << ']'
              << ",\"owner_a_copy_ok\":true,\"owner_b_copy_ok\":true"
              << ",\"native_prefix\":true,\"peer_fanin_wait\":true"
              << ",\"peer_dependency_dataflow_ok\":true"
              << ",\"context_restored\":true,\"arbitrary_kernel_transparency\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "owner execution GPU smoke: " << error.what() << '\n';
    return 1;
  }
}
