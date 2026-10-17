#ifndef V200_OPERATORS_GRAPH_REWRITE_HPP
#define V200_OPERATORS_GRAPH_REWRITE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>

#if defined(__GNUC__)
#define V200_GRAPH_REWRITE_API __attribute__((visibility("default")))
#else
#define V200_GRAPH_REWRITE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites only top-level, exact Q4_K MMVQ kernel nodes whose complete
// supported bank and pointer contract qualify for the physical peer owner.
// Graph arguments, dimensions, dependencies and nonmatching nodes are
// preserved. On success with zero rewrites, graph is unchanged. If a CUDA
// operation fails after an earlier node was rewritten, the caller must
// discard the graph and return the reported error rather than launch it.
V200_GRAPH_REWRITE_API cudaError_t CUDARTAPI
v200_operator_rewrite_captured_graph(cudaGraph_t graph,
                                     std::size_t *matched_nodes,
                                     std::size_t *rewritten_nodes) noexcept;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // V200_OPERATORS_GRAPH_REWRITE_HPP
