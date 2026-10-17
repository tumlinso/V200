# V200 implementation charter

## Product goal and current authorization

The product goal is one ordinary application-facing CUDA-compatible device
for each NVLink-connected pair of 16 GB V100 SXM2 GPUs, with useful aggregate
memory and dual compute. The first target is the existing, unchanged
`llama-server` executable using one logical device and no application-selected
physical multi-GPU split. This is the T2 compatibility target from the
bootstrap contract. A narrower or rebuilt consumer can be useful evidence,
but it is not silently relabeled as that target.

The user has authorized implementation toward a first usable llama-server
deliverable. The package under `planning/v200-bootstrap/` is design guidance
and a set of suggested outcomes; actual source, measured evidence, this active
charter, and later user direction govern implementation. The model fixture has
not yet been selected. Do not download or assume a model on the user's behalf.

The first delivery should arrive as soon as a bounded, correct path can be
shown. Keep the launcher, diagnostics, failure behavior and rollback usable
while compatibility coverage and performance improve. Do not claim a target
tier, operation, capacity or speedup until its required evidence exists.

## Implementation ownership

The root owns cross-stream architecture, integration, task lifecycle and
acceptance. First-class Luna workers own the bounded modules assigned to them,
including investigation, implementation and focused validation. Resolve
routine interface and code organization choices within those boundaries.
Create durable work boundaries only when ownership, dependency, integration
or risk calls for them. Keep the Todo Orchestrator as task authority; never
edit its SQLite database or generated projections directly.

The semantic seams in [interfaces.md](interfaces.md) freeze only the
information and ordering that independent modules must agree on. Implementers
choose local C++ names and layouts. Unknown source, active work and user data
remain preserved.

## GPU and service budget

Build and run host checks before acquiring a GPU. GPU execution uses the
installed CUDA controller, with admitted UUIDs, topology and cleanup evidence.
No background watch is armed. GPU tests remain out of default CPU builds and
test runs.

The user expressly authorized evicting running llama-server and Project
Control GPU workloads to free a GPU for V200 work. Use supported quiesce or
service-stop controls for those workloads when needed, record the action, and
verify release before admission. This authorization does not extend to
unrelated GPU processes, device reset, driver changes, clocks, firmware,
system-wide device presentation, or arbitrary process killing.

Avoid holding a GPU lease while compiling, inspecting source, waiting on a
decision or preparing evidence. Use small foundational checks first, then
expand only when their result justifies the next admitted bundle. Never spend
GPU time to turn an unknown into a presumed pass.

## Evidence and claims

Report `CPU_PASS`, `COMPILED_ONLY`, `GPU_PASS`, `NOT_RUN`, `BLOCKED`,
`INCONCLUSIVE` or `FAILED` accurately. CPU examples and compile-only CUDA
targets do not establish runtime or hardware behavior. Keep binary, library,
source and fixture identities with results. Unknown is not zero.

For each milestone, report what now works, exact consumer and stack identity,
the tested boundary, matched baseline when relevant, limits, and the next
bounded result. Ask the user when evidence requires changing the external
product contract, model/configuration or numerical semantics, or when an
alternative materially changes compatibility, latency or throughput goals.
