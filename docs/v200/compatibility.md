# V200 compatibility boundary

V200 keeps its native llama-server path as the default. The pair facade starts
CUDA graph capture only when the launcher receives `--cuda-graphs`; this option
also enables llama's graph path for the matched native control. Ordinary runs
continue to set `GGML_CUDA_DISABLE_GRAPHS=1` and do not query stream-capture
state on each launch.

In pair mode, capture records the application's ordinary anchor-side CUDA
operations. While a stream is actively capturing, V200 bypasses its normal
per-operation completion-event insertion and skips the pair operator hook,
which uses peer streams and cross-context event joins that cannot be captured
safely. After `cudaStreamEndCapture`, the opt-in pair-compute path asks the
operator library to rewrite only recognized Q4_K MMVQ kernel nodes whose
arguments, weight-bank ownership, and peer function are qualified. All other
nodes remain as captured. The launcher keeps the captured graph's dependencies
and parameters intact; a replay passes through the ordinary completion ledger
as one asynchronous submission and retains pair allocations through that
replay's completion prefix.

Capture bypasses the launch-time `V200_OPERATOR_VERIFY` comparison because
that check synchronizes and copies results while a graph is being built. The
consumer's exact-response oracle still checks inference output; graph-specific
numeric comparison remains part of hardware qualification rather than a claim
made by this capture hook.

The current wrapper surface covers the pinned llama runtime sequence:
`cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`,
`cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch`,
`cudaGraphDestroy`, and `cudaGraphExecDestroy`. CUDA graph APIs outside this
sequence, including graph construction APIs and the flags-based instantiate
variant, are not qualified. As in ordinary CUDA graph use, the application
must keep device allocations referenced by a graph alive for the graph's
lifetime and until its replays complete; V200 tracks each replay but does not
own the entire graph-exec lifetime.

This graph path is an implementation candidate until the bounded controller
qualification completes. A zero `graph_rewritten_nodes` count means capture
and replay used the anchor fallback for that graph; it does not demonstrate
peer compute. Graph rewrite counts are reported in the existing
`v200-frontdoor-ledger/v1` profile records so they can be read alongside
allocation ownership and cleanup accounting.

Use `scripts/v200/serve.py --mode preview --workload resident_capacity
--pair-compute --cuda-graphs` for the matched one-shot graph qualification
plan. The native control receives graph enablement without V200 interception;
the pair facade receives `V200_GRAPH_CAPTURE=1` and the same llama graph
configuration. The default launch remains graph-off.
