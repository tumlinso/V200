# V200 owner execution and completion contract

`v200::execution::OwnerExecution` tracks asynchronous work already forwarded to
native CUDA streams and work explicitly enqueued under an owner context. It
provides immutable prefix tokens so the memory layer can delay release until
all streams that used an allocation have completed. The same implementation
has a C bridge in `v200/execution/owner_execution_c.h` for ABI wrappers that
cannot link the C++ target directly.

## Prefixes and dependencies

Each logical stream may bind one native CUDA stream per owner. A native call
that was successfully forwarded must be serialized with prefix publication.
Concurrent wrappers should use `forward_native_submission`, which executes
the real native call and marks the prefix dirty under the same adapter lock
used by peer submissions. `observe_native_submission` is only safe when an
external per-stream gate covers both the native dispatch and observation. It
marks that owner prefix dirty. The next
`current_token`, `current_device_token`, `submit`, or `wait_on_stream` records
an internal timing-disabled event after each dirty native prefix. Tokens hold
all current owner-head events for the selected logical stream, or all owner
heads on a logical device. They therefore represent full prefixes rather than
the most recent event from only one owner.

For `CU_STREAM_PER_THREAD`, `forward_native_submission` eagerly records the
completion event on the dispatching host thread before it returns. A later
snapshot can safely run on another host thread because it reuses that captured
event instead of recording against that later thread's distinct PTDS.

`submit` first seals dirty native prefixes, then inserts event waits for other
owner heads and any explicit dependency tokens before invoking its enqueue
callback. Same-stream ordering is implicit. Passing a token from the same
logical stream as an explicit dependency is rejected as a cycle. An ordinary
native stream call cannot retroactively insert a peer wait, so
`observe_native_submission` refuses multi-owner streams after the first owner
has joined. Use `submit` or `wait_on_stream` to perform that handoff.

The callback runs while the configured owner context is current and receives
that owner's native stream. It must enqueue work through the underlying CUDA
driver/runtime dispatch and must not re-enter the same `OwnerExecution` object.
The adapter serializes its bookkeeping and callbacks under one mutex for this
first implementation; it is a correctness path, not a throughput scheduler.

## Token ownership and safe event lifetime

The caller owns one token reference returned by `current_token`,
`current_device_token`, `submit`, or `wait_on_stream`; release it exactly once,
or retain it before handing it to another subsystem. The memory manager should
retain one reference for each allocation use and release that reference only
after successful retirement. Tokens include slot generations. An event can be
destroyed only after it is complete and no token or stream head references it.
Reused token and event slots advance their generations, so stale handles fail
closed. Failed queries, waits, event recording, or context restoration poison
the logical device/stream and do not make event storage eligible for reuse.

`is_complete` performs event queries and never waits. `wait` synchronizes every
event in its token even when an earlier peer event reports an error, preserving
the first failure while attempting to drain the full prefix. On a failure,
owners must stop new submissions and keep the associated allocation/backing
alive. An error is not evidence that queued work has stopped.

If event creation or recording fails while a stream is dirty, the adapter
attempts `stream_synchronize` under the owner context. A successful synchronous
drain safely represents that prefix as complete; if the fallback also fails,
the stream stays dirty and the device fails closed.

`shutdown` requires all caller token references to be released. It first seals
or synchronously drains every dirty stream, then drains known events before
destroying them. If any work cannot be proven complete, it returns an error
without dropping the dirty stream record. The destructor makes a best-effort
synchronous drain and leaves CUDA event resources alone if it cannot prove the
streams idle. Callers needing error reporting must call `shutdown` explicitly
and keep the injected `DriverDispatch` alive until afterward. The provider
object must outlive every allocation ledger or wrapper that may still retain
or release its tokens.

## Initial support boundary

The first route is deliberately conservative: native single-anchor stream
observation plus explicit owner callbacks and event-based peer fan-in. CUDA
event/stream behavior must still be checked against the target V200 runtime;
host tests use an injected fake dispatch and are not hardware qualification.
No latency or throughput claim follows from this adapter.

The following families are explicitly unsupported in this version and must
not be reported as successfully coordinated: CUDA graph capture, graph launch,
graph update or graph replay over pair-owned allocations; arbitrary split
kernels or automatic command rewriting; unqualified module/device-global
state and atomics across owners; and concurrent submissions across streams as
a scheduling policy. Such work must remain on a verified native single-owner
path or fail closed at the frontdoor. Cross-device event waits, event context
compatibility, driver ABI interposition, and pending-work behavior require the
separate finite controller-backed GPU smoke for the first actual workload.

## Optional real-driver smoke

`V200_EXECUTION_BUILD_GPU_SMOKE=ON` builds
`v200_execution_gpu_smoke`; it is never registered with CTest and is not run by
the normal CPU gate. The finite harness checks two ordered expected V100 UUIDs,
uses primary contexts and one native stream per owner, allocates 4 KiB per
device plus a small pinned host staging buffer, captures a native owner-A
prefix, then submits owner-B work that must consume owner-A's staged copy only
after the adapter's cross-context event wait. It waits on the full-prefix token,
copies both owner buffers back, verifies their bytes, and checks that caller
context state is restored. It makes no PTX-kernel, NVLink-route, transparency,
or performance claim.

Build it without execution using the CUDA 12 toolkit and the opt-in CMake
option. A future B02 invocation must go through
`planning/v200-bootstrap/tools/run_bounded.py` as a child of the admitted
foreground controller, with the exact ordered UUIDs and B02 cap bound by the
root-owned outer receipt. The child additionally requires `--execute`, a
readable receipt path, `V200_BUNDLE_ID=B02`, the injected allocation cap, and
matching ordered UUIDs. Environment variables alone do not prove admission.
