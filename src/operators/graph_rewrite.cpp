#include "v200/operators/graph_rewrite.hpp"

#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"
#include "v200/operators/launch.hpp"

#include <cuda.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <vector>

namespace {

struct DriverApi {
  using ContextGetCurrent = CUresult(CUDAAPI *)(CUcontext *);
  using ContextGetDevice = CUresult(CUDAAPI *)(CUdevice *);
  using PrimaryContextRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);
  using ContextSetCurrent = CUresult(CUDAAPI *)(CUcontext);
  using GraphGetNodes = CUresult(CUDAAPI *)(CUgraph, CUgraphNode *, size_t *);
  using GraphNodeGetType = CUresult(CUDAAPI *)(CUgraphNode, CUgraphNodeType *);
  using GraphKernelGetParams = CUresult(CUDAAPI *)(
      CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
  using GraphKernelSetParams = CUresult(CUDAAPI *)(
      CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);

  ContextGetCurrent get_current{};
  ContextGetDevice get_device{};
  PrimaryContextRetain primary_retain{};
  ContextPush push{};
  ContextPop pop{};
  ContextSetCurrent set_current{};
  GraphGetNodes graph_get_nodes{};
  GraphNodeGetType graph_node_get_type{};
  GraphKernelGetParams graph_kernel_get_params{};
  GraphKernelSetParams graph_kernel_set_params{};

  bool context_ready() const noexcept {
    return get_current != nullptr && get_device != nullptr &&
           primary_retain != nullptr && push != nullptr && pop != nullptr &&
           set_current != nullptr;
  }

  bool graph_ready() const noexcept {
    return graph_get_nodes != nullptr && graph_node_get_type != nullptr &&
           graph_kernel_get_params != nullptr &&
           graph_kernel_set_params != nullptr;
  }
};

struct RuntimeApi {
  using GetFunction = cudaError_t(CUDARTAPI *)(cudaFunction_t *, const void *);

  GetFunction get_function{};

  bool ready() const noexcept { return get_function != nullptr; }
};

struct PersistentPeerContext {
  std::mutex mutex;
  DriverApi driver{};
  CUdevice peer_device = -1;
  CUcontext context{};
  bool retained = false;
};

PersistentPeerContext &peer_context_state() {
  // The context reference is intentionally process-lifetime. Graph execs can
  // retain peer CUfunction handles until their owning application destroys
  // them, and a CUDA teardown destructor would race Runtime shutdown.
  static PersistentPeerContext *state = new PersistentPeerContext();
  return *state;
}

template <typename Function>
Function driver_symbol(const char *name, const char *fallback = nullptr) noexcept {
  void *address = v200_frontdoor_resolve_driver_symbol(name);
  if (address == nullptr && fallback != nullptr)
    address = v200_frontdoor_resolve_driver_symbol(fallback);
  return reinterpret_cast<Function>(address);
}

template <typename Function>
Function runtime_symbol(const char *name) noexcept {
  return reinterpret_cast<Function>(v200_frontdoor_resolve_runtime_symbol(name));
}

void resolve_driver(DriverApi *api) noexcept {
  if (api == nullptr ||
      (api->context_ready() && api->graph_ready())) return;
  api->get_current = driver_symbol<DriverApi::ContextGetCurrent>("cuCtxGetCurrent");
  api->get_device = driver_symbol<DriverApi::ContextGetDevice>("cuCtxGetDevice");
  api->primary_retain =
      driver_symbol<DriverApi::PrimaryContextRetain>("cuDevicePrimaryCtxRetain");
  api->push = driver_symbol<DriverApi::ContextPush>(
      "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
  api->pop = driver_symbol<DriverApi::ContextPop>(
      "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
  api->set_current = driver_symbol<DriverApi::ContextSetCurrent>("cuCtxSetCurrent");
  api->graph_get_nodes =
      driver_symbol<DriverApi::GraphGetNodes>("cuGraphGetNodes");
  api->graph_node_get_type =
      driver_symbol<DriverApi::GraphNodeGetType>("cuGraphNodeGetType");
  api->graph_kernel_get_params = driver_symbol<DriverApi::GraphKernelGetParams>(
      "cuGraphKernelNodeGetParams_v2");
  api->graph_kernel_set_params = driver_symbol<DriverApi::GraphKernelSetParams>(
      "cuGraphKernelNodeSetParams_v2");
}

void resolve_runtime(RuntimeApi *api) noexcept {
  if (api == nullptr || api->ready()) return;
  api->get_function = runtime_symbol<RuntimeApi::GetFunction>("cudaGetFuncBySymbol");
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

cudaError_t load_pair_profile(v200_memory_pair_profile *profile) noexcept {
  if (profile == nullptr) return cudaErrorInvalidValue;
  std::memset(profile, 0, sizeof(*profile));
  profile->abi_version = V200_MEMORY_ABI_VERSION;
  profile->struct_size = sizeof(*profile);
  const CUresult result = v200_memory_get_profile(profile);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (profile->abi_version != V200_MEMORY_ABI_VERSION ||
      profile->struct_size < sizeof(*profile) ||
      profile->owner_device[0] == profile->owner_device[1])
    return cudaErrorInvalidValue;
  if (profile->pair_access_qualified == 0 || profile->qualification_id[0] == '\0')
    return cudaErrorNotSupported;
  return cudaSuccess;
}

cudaError_t ensure_anchor_context(DriverApi &driver,
                                  const v200_memory_pair_profile &profile,
                                  CUcontext *anchor_out) noexcept {
  CUcontext anchor = nullptr;
  CUresult result = driver.get_current(&anchor);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (anchor == nullptr) return cudaErrorInvalidResourceHandle;
  CUdevice device = -1;
  result = driver.get_device(&device);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (device != profile.owner_device[0]) return cudaErrorInvalidDevice;
  *anchor_out = anchor;
  return cudaSuccess;
}

cudaError_t ensure_peer_context(PersistentPeerContext &state,
                                const v200_memory_pair_profile &profile) noexcept {
  resolve_driver(&state.driver);
  if (!state.driver.context_ready()) return cudaErrorNotSupported;
  if (state.retained) {
    return state.peer_device == profile.owner_device[1]
               ? cudaSuccess
               : cudaErrorInvalidDevice;
  }
  CUcontext context = nullptr;
  const CUresult result = state.driver.primary_retain(
      &context, profile.owner_device[1]);
  if (result != CUDA_SUCCESS) return map_driver_error(result);
  if (context == nullptr) return cudaErrorInvalidResourceHandle;
  state.peer_device = profile.owner_device[1];
  state.context = context;
  state.retained = true;
  return cudaSuccess;
}

class PeerContextGuard {
 public:
  PeerContextGuard(DriverApi &driver, CUcontext peer,
                   CUcontext anchor) noexcept
      : driver_(driver), anchor_(anchor), peer_context_(peer) {
    status_ = driver_.push(peer);
    pushed_ = status_ == CUDA_SUCCESS;
  }

  ~PeerContextGuard() {
    if (pushed_) (void)restore();
  }

  CUresult status() const noexcept { return status_; }

  CUresult restore() noexcept {
    if (!pushed_) return status_;
    pushed_ = false;
    CUcontext popped = nullptr;
    const CUresult pop_result = driver_.pop(&popped);
    if (pop_result == CUDA_SUCCESS && popped == peer_context_) {
      CUcontext current = nullptr;
      const CUresult get_result = driver_.get_current(&current);
      if (get_result == CUDA_SUCCESS && current == anchor_) return CUDA_SUCCESS;
    }
    const CUresult set_result = driver_.set_current(anchor_);
    return set_result == CUDA_SUCCESS
               ? CUDA_ERROR_INVALID_CONTEXT
               : set_result;
  }

 private:
  DriverApi &driver_;
  CUcontext anchor_{};
  CUcontext peer_context_{};
  CUresult status_ = CUDA_SUCCESS;
  bool pushed_ = false;
};

struct RewritePlan {
  CUgraphNode node{};
  CUDA_KERNEL_NODE_PARAMS params{};
  const void *host_function{};
  std::uint32_t owner_index = 0;
};

cudaError_t rewrite_graph(cudaGraph_t graph, std::size_t *matched_nodes,
                          std::size_t *rewritten_nodes) {
  if (matched_nodes != nullptr) *matched_nodes = 0;
  if (rewritten_nodes != nullptr) *rewritten_nodes = 0;
  if (graph == nullptr) return cudaErrorInvalidValue;

  // The capture wrapper gates this path, and keeping the same opt-in check
  // here prevents an accidental direct call from changing ordinary graphs.
  const char *graph_opt_in = std::getenv("V200_GRAPH_CAPTURE");
  const char *compute_opt_in = std::getenv("V200_PAIR_COMPUTE");
  if (graph_opt_in == nullptr || std::strcmp(graph_opt_in, "1") != 0 ||
      compute_opt_in == nullptr || std::strcmp(compute_opt_in, "1") != 0)
    return cudaSuccess;

  v200_memory_pair_profile profile{};
  cudaError_t status = load_pair_profile(&profile);
  if (status != cudaSuccess) return status;

  PersistentPeerContext &peer_state = peer_context_state();
  std::lock_guard<std::mutex> state_lock(peer_state.mutex);
  resolve_driver(&peer_state.driver);
  if (!peer_state.driver.context_ready() || !peer_state.driver.graph_ready())
    return cudaErrorNotSupported;

  CUcontext anchor_context = nullptr;
  status = ensure_anchor_context(peer_state.driver, profile, &anchor_context);
  if (status != cudaSuccess) return status;

  RuntimeApi runtime{};
  resolve_runtime(&runtime);
  if (!runtime.ready()) return cudaErrorNotSupported;

  const std::size_t registered_count =
      v200_operator_list_mmvq_host_functions(nullptr, 0);
  if (registered_count == 0) return cudaSuccess;
  std::vector<const void *> host_functions(registered_count);
  const std::size_t filled_count = v200_operator_list_mmvq_host_functions(
      host_functions.data(), host_functions.size());
  if (filled_count > host_functions.size()) return cudaErrorUnknown;
  host_functions.resize(filled_count);

  std::map<const void *, cudaFunction_t> anchor_functions;
  for (const void *host_function : host_functions) {
    cudaFunction_t function = nullptr;
    const cudaError_t get_result =
        runtime.get_function(&function, host_function);
    if (get_result == cudaErrorInvalidDeviceFunction) continue;
    if (get_result != cudaSuccess) return get_result;
    if (function != nullptr) anchor_functions.emplace(host_function, function);
  }
  if (anchor_functions.empty()) return cudaSuccess;

  std::size_t node_count = 0;
  CUgraph driver_graph = reinterpret_cast<CUgraph>(graph);
  CUresult driver_status =
      peer_state.driver.graph_get_nodes(driver_graph, nullptr, &node_count);
  if (driver_status != CUDA_SUCCESS) return map_driver_error(driver_status);
  if (node_count == 0) return cudaSuccess;
  std::vector<CUgraphNode> nodes(node_count);
  driver_status = peer_state.driver.graph_get_nodes(driver_graph, nodes.data(),
                                                     &node_count);
  if (driver_status != CUDA_SUCCESS) return map_driver_error(driver_status);
  if (node_count > nodes.size()) return cudaErrorUnknown;
  nodes.resize(node_count);

  std::vector<RewritePlan> plans;
  for (CUgraphNode node : nodes) {
    CUgraphNodeType node_type = CU_GRAPH_NODE_TYPE_EMPTY;
    driver_status = peer_state.driver.graph_node_get_type(node, &node_type);
    if (driver_status != CUDA_SUCCESS) return map_driver_error(driver_status);
    if (node_type != CU_GRAPH_NODE_TYPE_KERNEL) continue;

    CUDA_KERNEL_NODE_PARAMS params{};
    driver_status =
        peer_state.driver.graph_kernel_get_params(node, &params);
    if (driver_status != CUDA_SUCCESS) return map_driver_error(driver_status);
    // Captured cudaLaunchKernel nodes expose kernelParams. The packed `extra`
    // ABI has no validated argument decoder yet, so it stays anchor-owned.
    if (params.kernelParams == nullptr || params.extra != nullptr) continue;

    const void *host_function = nullptr;
    for (const auto &entry : anchor_functions) {
      if (entry.second == params.func) {
        host_function = entry.first;
        break;
      }
    }
    if (host_function == nullptr) continue;

    std::uint32_t owner_index = 0;
    std::uint32_t family = 0;
    std::size_t bank_bytes = 0;
    std::size_t active_weight_bytes = 0;
    const dim3 grid(params.gridDimX, params.gridDimY, params.gridDimZ);
    const dim3 block(params.blockDimX, params.blockDimY, params.blockDimZ);
    if (!v200_operator_inspect_mmvq_bank(
            host_function, grid, block,
            params.kernelParams, &owner_index, &family, &bank_bytes,
            &active_weight_bytes))
      continue;
    (void)family;
    (void)bank_bytes;
    (void)active_weight_bytes;
    if (owner_index != 1) continue;
    plans.push_back({node, params, host_function, owner_index});
  }
  if (matched_nodes != nullptr) *matched_nodes = plans.size();
  if (plans.empty()) return cudaSuccess;

  status = ensure_peer_context(peer_state, profile);
  if (status != cudaSuccess) return status;

  std::vector<std::pair<const void *, cudaFunction_t>> peer_functions;
  peer_functions.reserve(plans.size());
  PeerContextGuard peer_guard(peer_state.driver, peer_state.context,
                              anchor_context);
  if (peer_guard.status() != CUDA_SUCCESS)
    return map_driver_error(peer_guard.status());
  for (const RewritePlan &plan : plans) {
    bool known = false;
    for (const auto &entry : peer_functions) {
      if (entry.first == plan.host_function) {
        known = true;
        break;
      }
    }
    if (known) continue;
    cudaFunction_t function = nullptr;
    status = runtime.get_function(&function, plan.host_function);
    if (status != cudaSuccess) {
      const CUresult restored = peer_guard.restore();
      return restored == CUDA_SUCCESS ? status : map_driver_error(restored);
    }
    if (function == nullptr) {
      const CUresult restored = peer_guard.restore();
      return restored == CUDA_SUCCESS ? cudaErrorInvalidDeviceFunction
                                      : map_driver_error(restored);
    }
    const auto anchor_function = anchor_functions.find(plan.host_function);
    if (anchor_function == anchor_functions.end()) {
      const CUresult restored = peer_guard.restore();
      return restored == CUDA_SUCCESS ? cudaErrorInvalidDeviceFunction
                                      : map_driver_error(restored);
    }
    if (function == anchor_function->second) {
      const CUresult restored = peer_guard.restore();
      return restored == CUDA_SUCCESS ? cudaErrorNotSupported
                                      : map_driver_error(restored);
    }
    peer_functions.emplace_back(plan.host_function, function);
  }
  const CUresult restored = peer_guard.restore();
  if (restored != CUDA_SUCCESS) return map_driver_error(restored);

  // Only the function handle changes. CUDA copies the complete parameter
  // record into the node, retaining its original arguments, launch shape,
  // shared-memory size, and graph edges.
  std::size_t rewritten = 0;
  for (RewritePlan &plan : plans) {
    cudaFunction_t peer_function = nullptr;
    for (const auto &entry : peer_functions) {
      if (entry.first == plan.host_function) {
        peer_function = entry.second;
        break;
      }
    }
    if (peer_function == nullptr) return cudaErrorUnknown;
    plan.params.func = peer_function;
    driver_status = peer_state.driver.graph_kernel_set_params(plan.node,
                                                              &plan.params);
    if (driver_status != CUDA_SUCCESS) {
      if (rewritten_nodes != nullptr) *rewritten_nodes = rewritten;
      return map_driver_error(driver_status);
    }
    ++rewritten;
  }
  if (rewritten_nodes != nullptr) *rewritten_nodes = rewritten;
  return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t CUDARTAPI v200_operator_rewrite_captured_graph(
    cudaGraph_t graph, std::size_t *matched_nodes,
    std::size_t *rewritten_nodes) noexcept {
  try {
    return rewrite_graph(graph, matched_nodes, rewritten_nodes);
  } catch (const std::bad_alloc &) {
    if (matched_nodes != nullptr) *matched_nodes = 0;
    if (rewritten_nodes != nullptr) *rewritten_nodes = 0;
    return cudaErrorMemoryAllocation;
  } catch (...) {
    if (matched_nodes != nullptr) *matched_nodes = 0;
    if (rewritten_nodes != nullptr) *rewritten_nodes = 0;
    return cudaErrorUnknown;
  }
}
