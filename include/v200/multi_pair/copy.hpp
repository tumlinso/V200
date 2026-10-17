#ifndef V200_MULTI_PAIR_COPY_HPP
#define V200_MULTI_PAIR_COPY_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* Cross-logical-pair copy hook for the frontdoor's cudaMemcpyPeerAsync
 * wrapper. A return of 0 means the hook did not handle the operation and had
 * no CUDA side effects. A return of 1 means it handled the operation; status
 * contains the result and callers must not fall back to native peer copy on
 * error. `stream_pair` identifies the logical pair that owns `stream`; it can
 * be the source pair (as in ggml-cuda's peer-copy path). Cross-pair V200
 * ranges use bounded synchronous pinned-host staging until a qualified direct
 * physical route is implemented. */
int v200_multi_pair_try_copy_peer_async(
    uint32_t dst_pair, uint32_t src_pair, uint32_t stream_pair, void *dst,
    const void *src, size_t bytes, cudaStream_t stream, cudaError_t *status);

#ifdef __cplusplus
}
#endif

#endif  // V200_MULTI_PAIR_COPY_HPP
