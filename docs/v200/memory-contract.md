# V200 memory contract

The memory target is a CUDA 12 Driver VMM ownership ledger for the process
frontdoor. It is the single allocation and VMM path for pointers or generic
handles returned by the V200 facade. Unknown native pointers and handles stay
outside this ledger and may be forwarded by the frontdoor according to the
CUDA API being wrapped. The library has no CUDA-context constructor and performs
no device work during static initialization.

## Profile and allocation policy

The initializer supplies `v200_memory_pair_profile` after CUDA initialization.
It names two distinct physical CUDA owners, their UUIDs, the default owner,
each owner's allocation budget, and an optional pair qualification ID. A
qualified profile must include both UUIDs and a nonempty qualification ID.
The profile producer is responsible for binding that ID to the exact pair and
foundation evidence; the memory library does not independently validate an
evidence receipt.

For `v200_memory_allocate[_on]`, the manager queries the active Driver VMM
minimum granularity, checks the requested byte count and alignment, rounds to a
legal logical size, and plans physical owner extents before creating them.
It prefers the requested owner. A request that fits its remaining budget stays
owner-local. If it does not fit, the first-cut representation is one contiguous
physical extent per owner in a single canonical virtual address range. The
split must satisfy both owners' granularity constraints, the remaining budgets,
and checked arithmetic. Pair-backed extents are rejected unless the profile is
explicitly qualified. If no legal two-extent split fits, allocation fails
without publication. The manager does not suballocate, compact, or search
arbitrary fragmented extents.

The budgets are enforced before publication and are also available through
`v200_memory_get_usage`. The report contains bytes used and configured for each
owner, live mapped allocation count, live reservation count, and cleanup-pending
count. These are manager-accounted VMM bytes; process-wide driver usage must be
measured separately by the consumer harness.

The qualification flag permits the memory layer to represent and map compound
pair backing and to grant access to either configured owner. It is not a claim
that an arbitrary kernel is safe to run against peer memory. The current
foundation evidence covers the explicitly tested access pattern only; the
consumer or execution profile must separately authorize any workload that
relies on cross-owner execution. Without qualification, access descriptors must
remain local to each physical extent's owner.

## Direct Driver VMM gateway

Frontdoor wrappers must route managed `cuMemAddressReserve`,
`cuMemAddressFree`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`,
and `cuMemRelease` operations through the `v200_memory_vmm_*` functions. This
keeps `cudaMalloc`, pointer ownership queries, and llama.cpp's direct VMM pool
in the same ledger. The current llama.cpp path reserves a 32 GiB virtual range,
maps device-local extents at the tail, releases each generic handle after the
map, and unmaps the coalesced range before freeing the reservation.

The first cut supports only pinned device allocations for one of the two
configured owners, with no export handle types or allocation flags. Reservations
must request an unconstrained address; map offsets and flags must be zero. A
synthetic opaque handle represents the one or two tracked native physical
handles and is recognized only by the gateway's live handle table. A handle can
be mapped once at a time; aliases and repeated mappings are rejected. Unmap
must cover complete contiguous mapped extents. Access lists may name one or
both configured owners; remote access requires the qualified profile. Address
free requires an exact tracked base and size and no remaining mapped extents.

Imports and exports, fixed-address reservations, nonzero map offsets, aliasing,
partial remapping, host or NUMA allocations, compression, RDMA, tile-pool and
other allocation flags, and unsupported properties fail closed. These calls
must not fall through to native CUDA when the input is a synthetic handle or a
managed address.

## Pointer identity and lifetime

`v200_memory_owns_pointer` answers whether an address belongs to a tracked
canonical allocation, including a cleanup-pending allocation whose teardown
must be retried. `v200_memory_query` and `v200_memory_retain_use` additionally
require the queried byte to remain mapped. An unknown pointer returns
`CUDA_ERROR_INVALID_VALUE`; an unconfigured manager returns
`CUDA_ERROR_NOT_INITIALIZED` from status queries. An allocation API release
accepts only its exact base pointer. Direct VMM mappings are released through
the VMM gateway, not `v200_memory_release`.

The execution bridge installs a `CompletionProvider` whose lifetime must
outlast every retained memory token. For each native asynchronous dispatch, the
frontdoor calls `v200_memory_begin_submission` before forwarding
an asynchronous operation that may touch managed memory, then records the
native submission and snapshots the sealed owner-execution prefix. It calls
`v200_memory_retain_use` for known pointer arguments or
`v200_memory_retain_all_use` when generic kernel arguments are opaque, and only
then calls `v200_memory_end_submission(lease, 1)`. End with `0` if observation,
snapshotting, or retention fails; this enters fail-stop mode while backing is
still protected by the lease. The latter retention form conservatively attaches
the prefix to every live mapped managed allocation. Free and unmap wait until
all submission leases retire, then wait for retained prefixes and release their
references. A failed wait leaves the allocation and accounting intact for
retry. If a prefix cannot be retained after dispatch, the manager enters
fail-stop mode and refuses cleanup so the backing cannot be freed while work
may still be using it.

Retention calls include a logical stream ID. For a nonzero ID, a newer sealed
prefix replaces the retained prefix for that allocation and stream after the
new reference is retained and the old reference is released. This bounds
long-lived server bookkeeping to roughly one retained prefix per live
allocation and logical stream. The execution bridge guarantees that a newer
snapshot dominates the earlier prefix for the same stream; callers unable to
prove that identity use ID zero, which preserves each distinct prefix. A failed
old-reference release keeps both entries and fail-stops cleanup. The caller
must not reuse a stream ID for a different native stream while retained
references remain.

Driver calls are transactional where the API permits it. Partial create/map
failure rolls back completed native operations. If rollback itself fails, the
object remains in the ledger and `v200_memory_retry_cleanup` retries teardown;
usage remains charged until each physical handle and mapping is actually gone.

## Validation boundary

The CPU-only memory tests use an injected Driver dispatch and completion
provider. They cover granularity planning, compound extents, budget accounting,
overflow rejection, map rollback, retry after unmap failure, completion waits,
fail-stop retention, owner-local access, and direct VMM lifetime. They do not
qualify GPU behavior. Pair VMM hardware qualification and capacity remain owned
by the project foundation and consumer tasks.
