#include "v200/operators/pair_runtime.hpp"

#include "v200/frontdoor/runtime.h"
#include "v200/memory/api.h"

#include <cstring>
#include <mutex>

namespace v200::operators {
namespace {

struct DriverApi {
  using PrimaryRetain = CUresult(CUDAAPI *)(CUcontext *, CUdevice);
  using ContextGetCurrent = CUresult(CUDAAPI *)(CUcontext *);
  using ContextGetDevice = CUresult(CUDAAPI *)(CUdevice *);
  using ContextPush = CUresult(CUDAAPI *)(CUcontext);
  using ContextPop = CUresult(CUDAAPI *)(CUcontext *);
  using ContextSetCurrent = CUresult(CUDAAPI *)(CUcontext);
  using StreamCreate = CUresult(CUDAAPI *)(CUstream *, unsigned int);
  using StreamSynchronize = CUresult(CUDAAPI *)(CUstream);
  using EventCreate = CUresult(CUDAAPI *)(CUevent *, unsigned int);
  using EventRecord = CUresult(CUDAAPI *)(CUevent, CUstream);
  using StreamWaitEvent = CUresult(CUDAAPI *)(CUstream, CUevent, unsigned int);

  PrimaryRetain primary_retain{};
  ContextGetCurrent context_get_current{};
  ContextGetDevice context_get_device{};
  ContextPush context_push{};
  ContextPop context_pop{};
  ContextSetCurrent context_set_current{};
  StreamCreate stream_create{};
  StreamSynchronize stream_synchronize{};
  EventCreate event_create{};
  EventRecord event_record{};
  StreamWaitEvent stream_wait_event{};

  bool ready() const noexcept {
    return primary_retain != nullptr && context_get_current != nullptr &&
           context_get_device != nullptr && context_push != nullptr &&
           context_pop != nullptr && context_set_current != nullptr &&
           stream_create != nullptr && stream_synchronize != nullptr &&
           event_create != nullptr && event_record != nullptr &&
           stream_wait_event != nullptr;
  }
};

struct PairState {
  DriverApi api{};
  CUdevice owner_device[2]{};
  CUcontext anchor_context{};
  CUcontext peer_context{};
  CUstream peer_stream{};
  CUevent anchor_ready{};
  CUevent peer_done{};
  bool peer_context_retained = false;
  bool initialized = false;
};

std::mutex pair_mutex;
PairState pair_state;

template <typename Function>
Function driver_symbol(const char *name, const char *fallback = nullptr) noexcept {
  void *address = v200_frontdoor_resolve_driver_symbol(name);
  if (address == nullptr && fallback != nullptr)
    address = v200_frontdoor_resolve_driver_symbol(fallback);
  return reinterpret_cast<Function>(address);
}

void resolve_driver_api(DriverApi *api) noexcept {
  if (api == nullptr || api->ready()) return;
  api->primary_retain = driver_symbol<DriverApi::PrimaryRetain>(
      "cuDevicePrimaryCtxRetain");
  api->context_get_current =
      driver_symbol<DriverApi::ContextGetCurrent>("cuCtxGetCurrent");
  api->context_get_device =
      driver_symbol<DriverApi::ContextGetDevice>("cuCtxGetDevice");
  api->context_push = driver_symbol<DriverApi::ContextPush>(
      "cuCtxPushCurrent_v2", "cuCtxPushCurrent");
  api->context_pop = driver_symbol<DriverApi::ContextPop>(
      "cuCtxPopCurrent_v2", "cuCtxPopCurrent");
  api->context_set_current =
      driver_symbol<DriverApi::ContextSetCurrent>("cuCtxSetCurrent");
  api->stream_create =
      driver_symbol<DriverApi::StreamCreate>("cuStreamCreate");
  api->stream_synchronize =
      driver_symbol<DriverApi::StreamSynchronize>("cuStreamSynchronize");
  api->event_create = driver_symbol<DriverApi::EventCreate>("cuEventCreate");
  api->event_record = driver_symbol<DriverApi::EventRecord>("cuEventRecord");
  api->stream_wait_event =
      driver_symbol<DriverApi::StreamWaitEvent>("cuStreamWaitEvent");
}

cudaError_t to_cuda_error(CUresult result) noexcept {
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
    default:
      return cudaErrorUnknown;
  }
}

CUstream driver_stream(cudaStream_t stream) noexcept {
  // CUDA 12.9 uses matching values for Runtime and Driver legacy/PTDS stream
  // sentinels. Preserve both exactly; in particular, value 1 is legacy and
  // value 2 is per-thread.
  return reinterpret_cast<CUstream>(stream);
}

CUresult restore_anchor_context(PairState &state) noexcept {
  CUcontext current = nullptr;
  CUresult result = state.api.context_get_current(&current);
  if (result == CUDA_SUCCESS && current == state.anchor_context)
    return CUDA_SUCCESS;
  return state.api.context_set_current(state.anchor_context);
}

CUresult pop_peer_context(PairState &state) noexcept {
  CUcontext popped = nullptr;
  const CUresult result = state.api.context_pop(&popped);
  if (result == CUDA_SUCCESS && popped == state.peer_context) {
    CUcontext current = nullptr;
    const CUresult current_result = state.api.context_get_current(&current);
    if (current_result == CUDA_SUCCESS && current == state.anchor_context)
      return CUDA_SUCCESS;
  }
  const CUresult restored = restore_anchor_context(state);
  return restored == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : restored;
}

template <typename Operation>
CUresult with_peer_context(PairState &state, Operation &&operation) noexcept {
  CUresult result = state.api.context_push(state.peer_context);
  if (result != CUDA_SUCCESS) return result;
  result = operation(state);
  const CUresult popped = pop_peer_context(state);
  return popped == CUDA_SUCCESS ? result : popped;
}

CUresult create_peer_resources(PairState &state) noexcept {
  if (state.peer_stream == nullptr) {
    CUresult result = state.api.stream_create(&state.peer_stream,
                                               CU_STREAM_NON_BLOCKING);
    if (result != CUDA_SUCCESS) return result;
  }
  if (state.peer_done == nullptr) {
    CUresult result = state.api.event_create(&state.peer_done,
                                              CU_EVENT_DISABLE_TIMING);
    if (result != CUDA_SUCCESS) return result;
  }
  return CUDA_SUCCESS;
}

CUresult sync_peer_operation(PairState &state) noexcept {
  return state.api.stream_synchronize(state.peer_stream);
}

CUresult synchronize_peer(PairState &state) noexcept {
  return with_peer_context(state, sync_peer_operation);
}

CUresult create_peer_resources_operation(PairState &state) noexcept {
  return create_peer_resources(state);
}

CUresult initialize_pair(PairState &state,
                         const v200_memory_pair_profile &profile) noexcept {
  resolve_driver_api(&state.api);
  if (!state.api.ready()) return CUDA_ERROR_NOT_SUPPORTED;

  if (state.owner_device[0] == 0 && state.owner_device[1] == 0) {
    state.owner_device[0] = profile.owner_device[0];
    state.owner_device[1] = profile.owner_device[1];
  } else if (state.owner_device[0] != profile.owner_device[0] ||
             state.owner_device[1] != profile.owner_device[1]) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  CUcontext current = nullptr;
  CUresult result = state.api.context_get_current(&current);
  if (result != CUDA_SUCCESS) return result;
  if (current == nullptr) return CUDA_ERROR_INVALID_CONTEXT;

  CUdevice current_device = -1;
  result = state.api.context_get_device(&current_device);
  if (result != CUDA_SUCCESS) return result;
  if (current_device != profile.owner_device[0]) return CUDA_ERROR_INVALID_DEVICE;
  if (state.anchor_context == nullptr) state.anchor_context = current;
  if (current != state.anchor_context) return CUDA_ERROR_INVALID_CONTEXT;
  if (state.initialized) return CUDA_SUCCESS;

  if (!state.peer_context_retained) {
    result = state.api.primary_retain(&state.peer_context,
                                     profile.owner_device[1]);
    if (result != CUDA_SUCCESS) return result;
    if (state.peer_context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
    state.peer_context_retained = true;
  }

  if (state.anchor_ready == nullptr) {
    result = state.api.event_create(&state.anchor_ready,
                                    CU_EVENT_DISABLE_TIMING);
    if (result != CUDA_SUCCESS) return result;
  }

  result = with_peer_context(state, create_peer_resources_operation);
  if (result != CUDA_SUCCESS) return result;
  state.initialized = true;
  return CUDA_SUCCESS;
}

cudaError_t validate_profile(v200_memory_pair_profile *profile) noexcept {
  if (profile == nullptr) return cudaErrorInvalidValue;
  std::memset(profile, 0, sizeof(*profile));
  profile->abi_version = V200_MEMORY_ABI_VERSION;
  profile->struct_size = sizeof(*profile);
  const CUresult result = v200_memory_get_profile(profile);
  if (result != CUDA_SUCCESS) return to_cuda_error(result);
  if (profile->abi_version != V200_MEMORY_ABI_VERSION ||
      profile->struct_size < sizeof(*profile) ||
      profile->owner_device[0] == profile->owner_device[1])
    return cudaErrorInvalidValue;
  if (profile->pair_access_qualified == 0 || profile->qualification_id[0] == '\0')
    return cudaErrorNotSupported;
  return cudaSuccess;
}

bool valid_grid(dim3 grid) noexcept {
  return grid.x != 0 && grid.y != 0 && grid.z != 0;
}

CUresult record_anchor_ready(PairState &state, CUstream stream) noexcept {
  return state.api.event_record(state.anchor_ready, stream);
}

struct PeerLaunch {
  const OwnerPartition *partition{};
  const void *host_function{};
  dim3 block{};
  std::size_t shared_bytes{};
  v200_operator_native_launch_fn native_launch{};
  cudaError_t launch_result = cudaSuccess;
};

CUresult enqueue_peer(PairState &state, PeerLaunch &launch) noexcept {
  CUresult result = state.api.stream_wait_event(state.peer_stream,
                                                 state.anchor_ready, 0);
  if (result != CUDA_SUCCESS) return result;
  launch.launch_result = launch.native_launch(
      launch.host_function, launch.partition->grid, launch.block,
      launch.partition->arguments, launch.shared_bytes,
      reinterpret_cast<cudaStream_t>(state.peer_stream));
  if (launch.launch_result != cudaSuccess) {
    const CUresult drained = state.api.stream_synchronize(state.peer_stream);
    return drained == CUDA_SUCCESS ? CUDA_ERROR_LAUNCH_FAILED : drained;
  }
  return state.api.event_record(state.peer_done, state.peer_stream);
}

CUresult enqueue_peer_operation(PairState &state, PeerLaunch &launch) noexcept {
  return enqueue_peer(state, launch);
}

CUresult join_peer(PairState &state, CUstream anchor_stream) noexcept {
  return state.api.stream_wait_event(anchor_stream, state.peer_done, 0);
}

CUresult join_peer_operation(PairState &state) noexcept {
  CUcontext current = nullptr;
  CUresult result = state.api.context_get_current(&current);
  if (result != CUDA_SUCCESS) return result;
  if (current != state.anchor_context) return CUDA_ERROR_INVALID_CONTEXT;
  return CUDA_SUCCESS;
}

}  // namespace

cudaError_t launch_owner_partitions(
    const OwnerPartition *partitions, std::size_t count,
    const void *host_function, dim3 block, std::size_t shared_bytes,
    cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch) noexcept {
  if (partitions == nullptr || (count != 1 && count != 2) ||
      host_function == nullptr || native_launch == nullptr || !valid_grid(block))
    return cudaErrorInvalidValue;

  std::lock_guard<std::mutex> lock(pair_mutex);
  v200_memory_pair_profile profile{};
  const cudaError_t profile_status = validate_profile(&profile);
  if (profile_status != cudaSuccess) return profile_status;

  const OwnerPartition *anchor_part = nullptr;
  const OwnerPartition *peer_part = nullptr;
  for (std::size_t index = 0; index < count; ++index) {
    if (!valid_grid(partitions[index].grid)) return cudaErrorInvalidConfiguration;
    if (partitions[index].device == profile.owner_device[0]) {
      if (anchor_part != nullptr) return cudaErrorInvalidValue;
      anchor_part = &partitions[index];
    } else if (partitions[index].device == profile.owner_device[1]) {
      if (peer_part != nullptr) return cudaErrorInvalidValue;
      peer_part = &partitions[index];
    } else {
      return cudaErrorInvalidDevice;
    }
  }
  if (anchor_part == nullptr && peer_part == nullptr)
    return cudaErrorInvalidValue;

  if (peer_part == nullptr) {
    if (anchor_part == nullptr) return cudaErrorInvalidValue;
    return native_launch(host_function, anchor_part->grid, block,
                         anchor_part->arguments, shared_bytes, anchor_stream);
  }

  PairState &state = pair_state;
  const CUresult initialized = initialize_pair(state, profile);
  if (initialized != CUDA_SUCCESS) return to_cuda_error(initialized);

  const CUstream anchor_native_stream = driver_stream(anchor_stream);
  CUresult result = record_anchor_ready(state, anchor_native_stream);
  if (result != CUDA_SUCCESS) return to_cuda_error(result);

  PeerLaunch peer_launch{peer_part, host_function, block, shared_bytes,
                         native_launch, cudaSuccess};
  result = with_peer_context(
      state, [&peer_launch](PairState &inner) noexcept {
        return enqueue_peer_operation(inner, peer_launch);
      });
  if (result != CUDA_SUCCESS) {
    // The peer callback can fail after the stream wait or launch was accepted.
    // Synchronize the owned stream before surfacing the error when possible;
    // the outer memory lease remains fail-stopped if completion is uncertain.
    const CUresult drained = synchronize_peer(state);
    if (drained != CUDA_SUCCESS) return to_cuda_error(drained);
    if (peer_launch.launch_result != cudaSuccess)
      return peer_launch.launch_result;
    return to_cuda_error(result);
  }

  if (anchor_part != nullptr) {
    const cudaError_t anchor_result = native_launch(
        host_function, anchor_part->grid, block, anchor_part->arguments,
        shared_bytes, anchor_stream);
    if (anchor_result != cudaSuccess) {
      const CUresult peer_drained = synchronize_peer(state);
      const CUresult anchor_drained = state.api.stream_synchronize(anchor_native_stream);
      if (peer_drained != CUDA_SUCCESS) return to_cuda_error(peer_drained);
      if (anchor_drained != CUDA_SUCCESS) return to_cuda_error(anchor_drained);
      return anchor_result;
    }
  }

  result = join_peer(state, anchor_native_stream);
  if (result != CUDA_SUCCESS) {
    const CUresult peer_drained = synchronize_peer(state);
    const CUresult anchor_drained = state.api.stream_synchronize(anchor_native_stream);
    if (peer_drained != CUDA_SUCCESS) return to_cuda_error(peer_drained);
    if (anchor_drained != CUDA_SUCCESS) return to_cuda_error(anchor_drained);
    return to_cuda_error(result);
  }

  result = join_peer_operation(state);
  return result == CUDA_SUCCESS ? cudaSuccess : to_cuda_error(result);
}

}  // namespace v200::operators
