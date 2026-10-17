# V200 design dossier

**Design date:** 17 October 2026. **Decision status:** recommended initial architecture, with guarded alternatives. **Hardware status:** no new hardware qualification in this authoring session.

References such as [A-R07], [A-E20], [W-VMM] and [P-ADAPT] resolve in [the source register](../research/SOURCES.md). Statements labeled “design” or “inference” are proposals, not measured facts. Exact source identities and coverage limits are retained in `research/source-lock.json`.

## 1. Product contract and success

A V200 is a logical CUDA-compatible accelerator built from one explicitly selected NVLink V100 pair. Within a participating application process, ordinary device discovery should show one device per pair. An application using one V200 should not need two CUDA device IDs, explicit physical-device tensor splitting, or a V200-specific model definition. The application may use its ordinary multi-device configuration to distribute work across **two logical V200s**; it should not have to know their four physical constituents.

The pair provides nominally 32 GB aggregate HBM and 160 physical SMs. Neither number is a promise of uniformly local memory, 32 GB free for application payload, or twice the speed of every kernel. Report actual bytes, per-domain headroom, and feature limits. Keep `sm_70`; never advertise Hopper, Blackwell, BF16/TF32 tensor support, or another architecture to persuade software to select incompatible code. [A-R07, W-CUDA12]

Three milestones are distinct:

**Capacity prototype:** a real existing consumer sees one device and correctly uses resident data exceeding one V100's available HBM. It may initially execute substantial computation on one owner. This is useful, but it is not dual-compute completion.

**Accelerated pair:** the selected application's significant work uses both GPUs, with measured end-to-end benefit or a precisely explained trade-off. Aggregate counters alone do not establish useful parallelism.

**Two-device system:** the four-GPU machine exposes two logical devices, preserves pair-local optimization, and works with the consumer's ordinary two-device distribution. Cross-pair execution must be separately qualified.

Broader application and CUDA API support grows after these milestones. Success on one binary is a compatibility profile, not universal CUDA compatibility.

## 2. Evidence that changes the architecture

The installed atlas already reports strong topology separation. Its October 6 campaign measured 4 MiB one-way payload rates of 111.9–114.7 GB/s within the recorded NVLink pairs, versus 5.16–5.17 GB/s on two slower accessible pairings. Four other directions lacked direct peer eligibility. These are particular measured transfer paths, not proof of link saturation, fixed runtime constants, or forwarding through intermediate GPUs. [A-E20]

VMM is a credible substrate, but qualification is narrower than our target. The existing E23 used two fixed aliases of one physical allocation. It does not establish one allocation spanning both GPUs, concurrent writable alias semantics, or compatibility with an application's own VMM allocator. NVIDIA documents reservation, physical backing, mapping and access as separate operations. [A-E23, W-VMM]

The low-level route is not automatically the fastest. In the selected atlas cases, event-based handshakes and batched/graph execution compared favorably with the tested peer-atomic and resident-executor alternatives. Those results do not establish universal rankings; they do refute treating a permanent GPU polling scheduler as a necessary starting point. [A-E16, A-E19, A-E22]

Current llama.cpp source is especially important: it contains custom CUDA kernels as well as library calls, direct Driver API VMM operations, conditional static versus dynamic CUDA linkage, and graph-related paths. A `cudaMalloc` shim plus cuBLAS wrappers is therefore not a sufficient compatibility plan. Driver entry-point lookup can also bypass ordinary ELF symbol interception unless handled correctly. [W-LLAMA-CMAKE, W-LLAMA-CUDA, W-ENTRY]

Finally, the NVIDIA open kernel module supports Turing and later, not Volta. Its source is useful evidence about machinery, not a drop-in V100 driver-modification route. Native Volta build work should use a qualified CUDA 12.x stack; CUDA 13 removes offline compilation support for this generation. Do not replace the user's working driver to obtain a preferred development setup. [W-OPEN, W-CUDA13]

## 3. Recommended architecture and why

### 3.1 Initial selection

Build a **process-scoped hybrid CUDA presentation layer** over the installed NVIDIA Driver API. Separate its compatibility front end from a pair-execution core. Begin with ordinary CUDA events, physical contexts and explicit VMM access permissions. Add qualified operation decomposition and selected kernel transformations only where the measured workload benefits.

```text
existing executable / libraries
             │ ordinary CUDA-facing calls and entry-point requests
             ▼
V200 presentation + ABI registry + device/property contract
             │ logical handles, canonical addresses, operation descriptors
             ▼
pair core: memory ownership + completion + execution policy
          ┌──┴───────────────────┐
          ▼                      ▼
    real context A         real context B
    V100 / HBM A ══ NVLink ══ V100 / HBM B
          │                      │
     host root A             host root B
```

This is a **design choice**, not a claim that a complete compatible shim already exists. Its highest risk is closing the interface used by real CUDA runtimes and libraries while keeping handles and asynchronous semantics consistent. V200-02 must expose that risk before a large allocator or kernel-rewriter effort accumulates.

Process scope is intentional: application-facing presentation is the stated product requirement. Physical `nvidia-smi`, PCI enumeration, kernel ownership and administrator diagnostics may continue to show four V100s. Do not globally spoof hardware or hide physical failure information. A launcher or injected compatibility library is acceptable delivery plumbing for the target; requiring source modifications or a special application backend is a consequential compatibility change.

### 3.2 Alternatives evaluated

| Family | Strength | Central risk | Position in this design |
|---|---|---|---|
| Library-only decomposition | Fast path to selected GEMMs; retains optimized libraries | Misses custom kernels, allocation and device semantics | Internal optimization, not the product boundary |
| Framework/backend integration | High semantic visibility; easier layout/sharding | Consumer must integrate a special backend; narrower transparency | Explicit user-approved fallback or reference baseline |
| Driver/runtime hybrid facade | Ordinary consumer surface; public physical substrate; progressive coverage | Runtime private paths, ABI closure, handle identity, graphs | Recommended first route |
| Full CUDA implementation/translation runtime | Complete control over virtual device semantics | Very large compatibility/compiler undertaking | Escalation only after bounded evidence |
| Kernel-driver/channel virtualization | Lower-level control and potential hardware submission gains | Volta proprietary stack, version fragility, system-wide failure domain | Isolated, approved research branch, not first critical path |
| General PTX/SASS distributed rewriting | Potential binary transparency for broad kernels | Semantics, alias analysis, grid/atomic/global state, code metadata | Qualified families first; never assume universal splitting |
| Remote process/service proxy | Isolation; clear ownership | Launch latency, pointer/IPC complexity, duplicated runtime state | Conditional alternative when isolation value justifies cost |

These families are composable. A different mechanism can replace one subsystem without rewriting the consumer contract. Do not materialize every alternative as a blocking Todo task. Record the selector, try the smallest discriminator, and implement the selected branch. [P-ADAPT]

### 3.3 Anchor-handle versus fully virtual-handle strategy

The initial candidate is **anchor-backed handles**: where safe, an outward stream or context corresponds to a real object on the pair's anchor GPU; companion state records the peer object and pending work. This can reduce incompatibility when NVIDIA libraries expect genuine handles. It is not permission to let unclassified calls escape the facade. For example, an exported function pointer that creates memory on the anchor behind the ownership ledger invalidates aggregation.

A fully synthetic-handle front end gives stricter control but requires every handle-consuming path to be translated before entering NVIDIA code. Compare these strategies in the ABI-closure spike. Select one per handle family; maintain explicit tags and ownership. Never mix real and synthetic values by guessing pointer shape. Retain the option to use real streams but virtual allocation handles if the reachable VMM API requires it.

The implementing root can change this internal strategy without asking the user if the same external contract and safety boundary hold. A need to patch consumer source, intercept undocumented private tables extensively, or move to a daemon is a checkpoint.

## 4. Compatibility front end

### 4.1 Pin and audit a concrete executable

Use the inspected llama.cpp commit `83f9389419ef93117d1444f87adfcd9f03ec0509` as a **source baseline**, not proof that it is the binary currently installed on the host. Record the actual executable hash, build flags, CUDA runtime/library versions, loaded DSOs, model and tokenizer hashes, quantization, graph policy and entry points. Prefer the user's current working executable for the first binary-transparency test; compare it with the pinned build when needed.

Do CPU-only ELF/source inspection first. `tools/inspect_elf.py` invokes `readelf`, not `ldd` or the target executable. It cannot discover every runtime `dlopen`, `dlsym` or private function table; its output is a lower bound. Complete that audit with a short, controller-admitted trace once a harness exists.

### 4.2 Surface closure

The observed surface, not a wish list of every CUDA API, determines the first profile. Cover these classes when used:

- Device enumeration, selection, properties, memory accounting, UUID/BDF association and peer attributes.
- Primary/explicit contexts and per-host-thread current-context stacks.
- Allocation/free, pointer/range queries, copies/memsets, host registration and direct Driver VMM reservation/backing/mapping/access/release.
- Module/library loading, function lookup, CUDA runtime registration paths, kernel arguments and launches.
- Streams, default-stream variants, events, synchronization and asynchronous error observation.
- Library handle creation, streams, pointer modes, workspace and dispatch.
- Graph capture, instantiation, replay and update, when the selected application invokes them.
- Versioned driver entry-point access, including the ABI selected by `cuGetProcAddress` and runtime driver-entry lookup.

A request for a driver function name is not enough to select its wrapper: CUDA version and legacy/per-thread default-stream flags matter. Use exact header typedefs, separate old and new function signatures, and keep a versioned table. Do not reinterpret a four-argument entry-point routine as the newer five-argument one. [W-ENTRY]

Bootstrap interception must avoid recursively intercepting its own physical calls. Use a verified physical dispatch table and a narrow thread-local internal-call guard; preserve the application's current context on return. Initialization is once-only and reentrancy-aware. Logging must not invoke CUDA while a loader lock or resolver guard is held.

Unknown APIs fall into three explicit states: proven transparent pass-through, not yet qualified, or unsupported. Return a clear documented error for unsupported behavior; do not return success, discard a synchronization request, or forward virtual handles to real driver functions. Audit mode may record unknown entry points and stop before incorrect work.

### 4.3 Static linkage and runtime-private paths

The current upstream build conditionally selects static CUDA libraries when `GGML_STATIC` is enabled; it is incorrect to claim all llama builds use static CUDART. Establish the actual binary's linkage. A Driver API boundary can still observe work issued by a static runtime only if its driver access paths are controlled. Ordinary `LD_PRELOAD` function wrappers alone do not prove this. [W-LLAMA-CMAKE, W-ENTRY]

A source-unmodified rebuild with dynamic CUDA linkage or graphs disabled is an informative **intermediate compatibility tier**, not equivalent to the unchanged-binary target. If that is the only feasible delivery route, show the user the difference before accepting it as V200's final presentation.

## 5. Device identity, properties and capacity

Use a stable logical identity derived from the sorted physical UUID pair and a configuration generation, not ordinals. Preserve physical UUIDs in a diagnostic mapping. Revalidate both members and the allowed topology before each deployment profile is activated. Report nominal pair memory in documentation but actual capacity in bytes through queries.

Do not simply sum every property:

| Property | Policy |
|---|---|
| Name | Distinguishable `V200 (2 x V100)` label, not a claim of a vendor-issued model |
| Compute capability | Native `7.0`, with only supported feature bits |
| Memory total/free | Actual logical allocation policy, reserves and committed state; no impossible uniform free-pool promise |
| Multiprocessor count | 160 only under a qualified virtual scheduling/property contract; never double cooperative residency by implication |
| Registers/shared memory/max threads per block | Physical per-SM/per-block constraints, not doubled |
| Cooperative/multi-device launch | Advertise only the implemented semantics; otherwise unsupported in the profile |
| UUID/PCI information | Logical identity plus explicit physical mapping; no fabricated physical PCI function |
| Peer access to another V200 | Logical capability that the runtime actually implements, not one anchor's capability copied blindly |

An initial anchor-only capacity prototype may need conservative compute properties. Record that as a stage limitation rather than exposing 160 while silently satisfying only an 80-SM cooperative-launch contract. Ordinary grids can be distributed without requiring every launch to use both GPUs; correctness still governs capability reporting.

Memory accounting must reserve scratch, event/graph/module state and any replicas on each owner. “Free A + free B” is at best an upper bound when an allocation requires an indivisible owner region. Track total logical payload, per-owner physical occupancy, fragmentation and temporary workspace separately. Failed allocation must unwind only its own partially created resources.

## 6. Memory architecture

### 6.1 Canonical VMM view

The recommended core memory representation is a stable **canonical virtual-address range** with a ledger of physical extents. Each extent has an owner, a backing handle, size, permissions, lifecycle state and last-writer/completion information. Each member is granted only the permissions necessary to execute qualified operations on that range. Local versus peer cost remains explicit internally. [W-VMM, A-R08]

For the first pair arena:

1. Query both devices' VMM capability, directed peer access and allocation granularities.
2. Calculate a checked common granularity and owner budgets; reserve one address range.
3. Create owner-A and owner-B physical allocations; map whole extents into the reserved range.
4. Explicitly grant both devices access using `cuMemSetAccess`.
5. Test accesses from both devices under explicit completion ordering.
6. Drain owned work before unmapping, releasing backing handles and freeing the reservation.

The supplied probe implements this small pattern. It does not make CUDA allocations virtual automatically. `cuDeviceEnablePeerAccess` alone is not a substitute for VMM access grants. The inspected API requires zero map offset and whole mapped-range unmapping; the allocator must track the exact extents it created. Do not infer a suballocation mapping ability from a nonzero `offset` parameter that is not supported in the pinned API. [W-VMM]

### 6.2 Intercept applications' own VMM

This is central for llama, not a later exotic feature. A consumer may reserve VA, create a physical allocation for its selected device, map it, set access, query properties, retain/export a handle, and release it separately. Wrapping only `cudaMalloc` leaves that path outside the pair allocator. [W-LLAMA-CUDA]

Choose one of two qualified implementations: represent the consumer's logical allocation handle as compound backing and translate the observed VMM lifecycle, or retain a real handle where the entire allocation remains on one owner and compose allocations at a higher supported boundary. The latter cannot satisfy a single physical-allocation request larger than one GPU by pretending the real handle is larger. V200-03 must include this exact case in the compatibility decision.

`cuMemGetAllocationPropertiesFromHandle`, retain/release, access queries and `cuMemGetAddressRange` must agree with the chosen logical model where exposed. IPC/export is not implied by an in-process compound handle. Either implement its serialization and lifetime contract later or report unsupported before handing it to another process.

### 6.3 Placement and layout

Start with coarse owner extents and canonical application layout. Do not choose fine page striping by default: an opaque matrix kernel may then fetch a large fraction of its input across NVLink. Preserve local hot data when an operation contract identifies ownership. Allow replicated immutable inputs or compact activation buffers when their reuse repays the cost.

A different internal representation is permissible only with a known producer/consumer contract. Merely mapping a column shard does not make an arbitrary existing kernel interpret a different stride correctly. Materializing a canonical result has a cost; keep it in the performance model. Unknown consumers force either a valid canonical view, a legal owner execution, or an explicit unsupported result.

Initially, do not remap live buffers. Treat changes of backing or layout as synchronized phase changes, and invalidate derived copies after writes. Prefer a single canonical mapping to writable aliases. CUDA's documented alias rules and the atlas's fixed-alias experiment do not justify coherent concurrent writes through unrelated virtual aliases. [W-CUDA12, A-E23]

### 6.4 Correct fallback is conditional

“Run the kernel on the anchor” is a fallback only if all referenced state is accessible **and** its memory-scope assumptions remain legal. A kernel using device-scope atomics on an allocation physically owned by a peer, unmanaged device globals, a device heap, texture objects or cooperative grid synchronization needs specific qualification. A pointer's numerical validity is not a semantic proof.

For an opaque operation without a safe remote contract, retaining owner-local storage or staging a known range can be valid. If the ranges are unknown, do not guess from pointer-like argument values. Refuse or escalate the missing operation contract. No generic migration of arbitrary pointer graphs is assumed.

## 7. Execution and scheduling

### 7.1 Dispatch classes

Every launch is assigned a qualified route, with the capability decision cached outside the hot path:

**Owner execution:** full native operation on one physical GPU, preserving its native kernel semantics. Remote memory eligibility is part of the operation contract.

**Qualified split grid:** a kernel whose logical work units are proven partitionable, with explicit indexing transformation or an equivalent replacement kernel.

**Semantic distributed operation:** known math/communication operation decomposed across owners, with declared inputs, outputs, aliasing, numerical contract and completion.

**Unsupported:** an observable diagnostic and no false success.

Do not treat “does not contain a barrier instruction” as a proof of split safety. Device globals, atomic scope, cooperative assumptions, persistent work queues, cross-CTA communication, texture handles and implicit library workspace are also relevant.

### 7.2 Safe grid splitting

For a 1D logical grid of G blocks and B threads per block, a partition beginning at block O computes:

`logical_block = physical_block + O`

`logical_thread = logical_block * B + threadIdx.x`

A grid-stride loop still advances by **G * B**, not by the smaller physical grid. Both `blockIdx` and `gridDim` behavior must be preserved; higher-dimensional grids need a documented flatten/unflatten mapping without integer overflow. Launch each logical block exactly once, including tails and empty partitions.

The CPU sample verifies partition coverage and the difference between logical and physical grid stride. The PTX probe uses an explicit element offset. Neither sample transforms an arbitrary binary, and neither authorizes splitting a kernel with inter-block synchronization.

For PTX rewriting, use structured parsing, exact target/entry identities, preserved parameter layouts, function/call effects and launch metadata. For SASS, also preserve relocations, resource metadata, branch targets, constants and scheduling dependencies. Qualify exact cubin hashes and invalidate the qualification when binaries change. An unknown kernel does not inherit a contract from a matching symbol substring. [A-M48]

### 7.3 Libraries and custom kernels

Use vendor libraries where their operation contracts are clean. For column-major `C = alpha op(A) op(B) + beta C`, an M- or N-split can produce disjoint output regions if offsets, transposition, leading dimensions, batched strides and aliasing are handled correctly. Each region applies beta exactly once. A K-split requires reduction of partial sums and applies beta once to the combined output, not once per partial. Changing reduction order may change numerical results.

Host/device pointer modes, math mode, workspace ownership, handle streams, algorithm caches and graph capture must remain consistent. Do not accidentally synchronize every device scalar back to the CPU. A wrapper must preserve asynchronous parameter lifetimes and the native library's error contract.

For quantized inference, profiling may show custom MMQ/MMV, dequantization or attention kernels dominate instead of cuBLAS. Their packed block formats, row strides, accumulation precision, tail handling and KV layout determine the partition. Implement the **highest-impact qualified family actually observed**, not a predetermined catalogue of transformer operators. [W-LLAMA-CUDA]

An operation contract may permit retaining local shards between compatible stages, which can avoid repeated all-gathers. Without that contract, materialize the canonical result. The first implementation should not attempt to recover an entire model graph from arbitrary instruction streams.

### 7.4 Scheduling policy

Begin with deterministic choices based on known operation size, current placement and qualified costs. There is no need for a learning scheduler, continual online tuning, global dynamic work queue or CPU model service. A small table keyed by binary/operation/shape/layout/pair profile is sufficient initially.

Choose whole-owner execution for small or asymmetric work when coordination would dominate. Split significant local work when predicted saved execution time exceeds launch, transfer and output-canonicalization costs. Profile a representative prefill and decode separately; they are different regimes. Policy overrides should be explicit and produce an evidence record, not silently retrain at runtime.

## 8. Completion, contexts and failure

### 8.1 Anchored fan-out/fan-in

A candidate low-overhead implementation uses one real outward stream on anchor A and a companion on B. For an admitted split operation:

```text
A stream: predecessors → start event → A work ───────────→ wait(done B) → next consumer
B stream:                wait(start) → B work → done B ──┘
```

NVIDIA documents cross-device event waits. This can let later anchor-stream operations observe completion of both halves without a CPU-wide device synchronize. The event record remains on its owning device; compute elapsed intervals on one device's clock or with host wall time, not by subtracting event timestamps from different GPUs. [W-CUDA12]

This is a proof skeleton requiring implementation qualification. Event reuse must not race an outstanding wait; event pooling needs generations or conservative retirement. Each edge must be submitted in a schedulable order. A release/acquire flag does not replace a missing scheduler-visible dependency. Graph capture needs its own translation rather than blindly recording dynamic companion work during capture. [A-R06]

Initial host synchronization is acceptable inside the prototype as explicitly tracked performance debt; it is not the optimized end state. Prefer a small correct implementation over a sophisticated unqualified resident protocol.

### 8.2 Logical semantics

A logical stream's synchronization joins all work issued into that stream before the call. Logical event record describes the joined prefix, not just whichever physical kernel finishes first. Default stream semantics must follow the caller's selected legacy or per-thread ABI. Nonblocking streams and cross-stream dependencies remain distinguishable. Do not add cross-thread serialization merely to simplify the implementation unless documented as a temporary profile limitation.

A logical context includes the two physical contexts and their per-thread attachment bookkeeping. Internal calls save and restore the caller's current context. Module state is owner-specific: identical code loading is not automatically coherent device globals. Unknown mutable globals remain single-owner or unsupported; read-only globals can be replicated with explicit update invalidation.

CUDA graph support follows observed nodes and launch patterns. First qualify a limited captured sequence without changing the application. If disabling graphs is necessary for first progress, label that intermediate tier and revisit; do not silently accept a permanently slower/less-compatible build. Graph replay must not allocate new hidden work unboundedly or omit peer edges.

### 8.3 Errors and cancellation

Validate arguments before partial side effects where possible. If one physical launch fails after another has been submitted, mark the logical operation failed, drain owned work as safely as the driver permits, and surface the asynchronous error at documented observation points. Never report success because the anchor succeeded while its peer failed.

No transparent replay of an arbitrary partially executed kernel: writes and atomics may not be idempotent. A peer failure is initially **fail-stop for that logical device/process**. Clean up only owned resources. Do not reset a GPU, kill another user's process, change clocks or unload the driver to make a test green.

Runtime timeouts diagnose lack of progress; a host timeout does not prove a hung GPU context was cleaned up. If process termination cannot release owned work, stop the campaign and request operator intervention. Never continue performance tests on a possibly contaminated device.

## 9. Making best use of NVLink

NVLink utilization is a means, not a success metric. The strongest optimization may eliminate traffic entirely by computing where data already lives. The runtime should compare direct peer access, explicit staging, owner computation and compact result transfer, and justified read-only replication. [A-R07]

A useful cost model is:

`T_split ≈ max(T_A, T_B) + T_comm + T_dispatch + T_materialize + T_interference`

`T_stage ≈ T_copy(bytes) + repetitions * T_local`

`T_remote ≈ T_dependent_latency + remote_bytes / effective_remote_bandwidth`

The terms are workload- and path-dependent. Do not add peak engine throughputs as if resources were independent. Concurrent remote reads consume owner memory-system capacity; copy and compute may interfere. A measured resident-kernel speedup is not an application speedup if conversion, gathering, launch or setup dominates.

As an illustrative bound, 8 GiB of peer traffic at the atlas's roughly 112 GB/s measured payload rate takes about 77 ms before additional effects. This is an arithmetic illustration using one historical rate, not a V200 prediction. A naive remote half of a memory-heavy operation can erase most compute scaling. That is why the design keeps canonical addressing separate from owner-aware execution and why “one heap” is not the performance solution by itself.

Prefer large, coalesced payloads and useful overlap when the baseline supports them. For small dependencies, compare event paths with properly scoped peer signaling; the existing atlas does not justify making atomic polling mandatory. Use all directly connected links as the hardware/driver route provides, but verify actual direction and counters when claiming link use. No inferred custom packet routing, port control or GPU transit switching.

The pair's two host roots are a possible benefit. Verify page location, allowed NUMA nodes and root association before claiming local ingress. E21 did not successfully measure both nodes. Data already produced on one node may require movement; pinning alone does not create locality. Split ingestion only when it reduces total critical-path cost and does not add a larger reassembly transfer. [A-E21]

Between V200s, route by the actual directed physical connectivity. The runtime can select eligible endpoints for logical copies; a consumer can perform ordinary layer/pipeline distribution across the two logical devices. A generic CUDA shim does not automatically understand model layers, sequence parallelism or expert routing. More semantic inter-pair scheduling is a separately qualified extension, not a hidden assumption of the device abstraction.

### 9.1 Cross-pair API truthfulness

On the recorded topology, an endpoint on one V200 may reach only one member of the other V200 directly. **Bulk-copy routability is not the same capability as arbitrary peer-pointer dereference.** `cudaDeviceCanAccessPeer` must not return a broad logical guarantee merely because one anchor-to-anchor link works. A runtime can implement a logical copy through eligible owner endpoints without making every SM able to address every remote extent. If general logical peer access is unsupported, report that fact and qualify how the consumer chooses copy/staging paths.

NCCL and NVML discovery may consult physical identities outside ordinary CUDA enumeration. They need their own observed compatibility strategy for two-V200 use; the first pair proof does not automatically virtualize them. Do not route a logical ordinal directly into a native collective that has discovered four unrelated physical ranks. An internal pair collective and an external logical collective have different contracts. This is part of V200-11, not a claim that an arbitrary four-GPU consumer works unchanged on day one.

## 10. Performance acceptance without benchmark inflation

Benchmark exact useful work. For inference, lock model/tokenizer/quantization/KV types, context size, prompts, batch/ubatch, offload, graph and fit policies, sampling settings, source and library versions. Record cold load, prompt processing, token decode, p50/p95 request latency where enough samples exist, memory occupancy and full wall time. Do not compare a reduced context or changed quantization as a virtualization win.

Baselines include native single V100 where the workload fits; native two-V100 NVLink configurations appropriate for the pinned consumer; and native four-GPU configurations for the two-V200 stage. For capacity exceeding one V100, compare with a native pair, not only a slow CPU-offload baseline. A same-GPU/no-splitting facade run isolates interception overhead. A capacity-only run isolates remote-memory cost from distributed execution benefit.

Initial practical decision bands are **proposed engineering defaults**, not findings: aim for low-single-digit facade overhead on fitting controls; flag a repeatable greater-than-10% application regression against a matched native pair; compare prefill and decode separately rather than averaging away a loss. A 5% apparent gain that disappears in run-to-run variation is not a promotion. Use a small matched sample set first; add trials only when a consequential decision remains ambiguous. Numeric thresholds can be changed by the root with a recorded rationale, but a material trade-off in the user's objective requires consultation.

Correctness for byte copies/index kernels is exact. Floating-point kernels use an explicit operation-appropriate tolerance and an independent reference; distinguish FP16 input rounding, accumulator precision and changed reduction order. For model tests, inspect logits against the matched native path and use deterministic generation as an additional check, not sole proof. No silent lower precision, dropped tokens, reduced context or changed sampling.

## 11. Execution sequencing

The outcome graph keeps GPU use away from the majority of coding. V200-01 binds sources/contracts and adopts the package. V200-02 establishes a coherent consumer-facing boundary. V200-03 and V200-04 build the memory and completion foundations; V200-05 prepares the actual consumer fixture and matched baseline. Their CPU work can proceed independently after interface agreement.

A small early qualification bundle checks **interface closure plus pair VMM and completion**. It prevents weeks of work built on an untested foundational premise. Then V200-06 delivers the first real single-device application. V200-07 qualifies capacity beyond one V100. V200-08 and V200-09 add the chosen high-impact distributed path and NVLink-aware placement. V200-10 extends observed compatibility only where it affects the application. V200-11 integrates the second V200; V200-12 conducts the bounded performance decision. V200-13 packages deployment and V200-14 integrates and hands off the release.

These are durable outcomes, not separate records for reading, coding, compiling and testing. The implementing root chooses local sequencing and subagents. If a gate fails, keep useful independent CPU work available and record the blocker. Do not mark a foundational result as passed merely because unrelated work can continue. [P-ADAPT]

## 12. Alternatives and decision triggers

**Boundary failure:** dynamic entry points/private calls cannot be closed safely within a bounded spike. Show unchanged-binary versus source-unmodified rebuild versus backend integration, measured or unmeasured status, and engineering consequences. Ask the user before changing the transparency target.

**Memory failure:** pair VMM or compound-handle semantics fail on the actual stack. Distinguish capability failure from implementation defect. Compare legal per-allocation ownership, managed-memory capacity, and application-aware sharding; the latter two change the performance/contract assumptions and need explicit acceptance if promoted.

**Scheduling failure:** transparent grid splitting does not cover the dominant kernels or performs poorly. Prefer targeted semantic replacements within the same outward ABI when correct; ask before replacing the product with a special consumer backend.

**Performance conflict:** a more transparent route is materially slower than a less transparent route, or prefill wins while decode loses. Present matched evidence and ask which trade-off to optimize. No arbitrary blended score decides for the user.

**Deep mechanism escalation:** SASS rewriting, private channel/QMD control, driver/firmware modifications or changes affecting other workloads need a source-backed reachability argument, explicit rollback and operator permission. Physical capability is not public authorization. [A-R09, A-M48]

The decision catalogue defines compact question formats and who can decide. Report at material milestones; do not ask for every tile size or small refactor, and do not monopolize GPUs while waiting for the user.

## 13. What this design intentionally does not build first

No full CUDA API emulator, kernel-driver fork, automatic symbolic proof engine, online ML scheduler, global distributed memory coherence protocol, permanent GPU worker service, comprehensive ISA characterization suite, or all-app compatibility matrix is a prerequisite to the first useful consumer run.

Also no fake shortcut: returning two-device capacity from a query while allocations fail, launching the same unsplit kernel twice, rewriting only `blockIdx`, losing graph edges, lowering precision invisibly, claiming NVLink from topology labels alone, or accepting CPU/offload fallback as aggregate-HBM proof.

The first architecture is deliberately reversible. Its reusable assets are the consumer contract, physical identity map, allocation/operation/completion records, compatibility tests and evidence. If the preferred facade cannot carry the target workload well, the next agent should have enough evidence to make that a specific choice with the user—not an excuse to abandon the project or a reason to conceal the limitation.
