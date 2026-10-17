#ifndef V200_OPERATORS_PAIR_RUNTIME_HPP
#define V200_OPERATORS_PAIR_RUNTIME_HPP

#include "v200/operators/launch.hpp"

#include <cuda.h>
#include <cstddef>

namespace v200::operators {

// One native invocation of the selected operation on a physical owner.  The
// argument table and its pointed-to values only need to remain valid until
// launch_owner_partitions returns; the frontdoor's submission lease retains
// the referenced device allocations through the joined anchor completion.
struct OwnerPartition {
  CUdevice device{};
  dim3 grid{};
  void **arguments{};
};

// Enqueue one or two physical-owner partitions as one anchor-stream operation.
// Owner zero is the application-visible anchor. Owner one runs on a
// process-lifetime nonblocking stream in its retained primary context. Returns
// a CUDA runtime error on any partial enqueue; callers must not fall back once
// this function has been entered.
cudaError_t launch_owner_partitions(
    const OwnerPartition *partitions, std::size_t count,
    const void *host_function, dim3 block, std::size_t shared_bytes,
    cudaStream_t anchor_stream,
    v200_operator_native_launch_fn native_launch) noexcept;

}  // namespace v200::operators

#endif  // V200_OPERATORS_PAIR_RUNTIME_HPP
