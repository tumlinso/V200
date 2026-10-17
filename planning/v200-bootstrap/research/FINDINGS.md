# Research synthesis: what is established, what remains decisive

The existing SXM2 atlas is the starting evidence, not a research plan to rerun. Its 16 reference chapters, 52 mechanisms and dated experimental records already cover the unusual hardware ideas. This phase narrowed the unresolved issues to **consumer-boundary closure, legal pair memory, qualified execution, and end-to-end benefit**. References resolve in [SOURCES.md](SOURCES.md).

## Findings with architectural consequences

| Finding | Evidence | Consequence | Not established |
|---|---|---|---|
| Real topology has strongly different pair costs | A-E20, October 6 directed copy cases | Pair-local work is a valuable structural objective | Fixed route, saturation, every size or simultaneous compute |
| Canonical VA and physical ownership are separable | W-VMM, A-R08 | VMM is the primary pair-arena candidate | Universal allocation compatibility or coherent writable aliases |
| The existing alias test used one allocation | A-E23 | Add one small two-owner spanning qualification | Existing atlas already proved a pooled pair allocator |
| Peer access and staged access have distinct costs | A-E18 | Compare pull, push, owner compute and stage by reuse | Cache-level explanation or universal latency constant |
| Selected events/batching beat selected resident protocols | A-E16, A-E19, A-E22 | Start from conventional completion; qualify fancier options | All peer atomics/persistent kernels are slow |
| Second NUMA-node transfer attempt failed | A-E21 | Treat dual ingress as an open machine-specific benefit | A demonstrated two-root performance gain |
| llama has direct Driver VMM and custom kernels | W-LLAMA-CUDA, W-LLAMA-CMAKE | Need more than malloc/cuBLAS interception | Source inspection alone proves unchanged-binary virtualization |
| Driver entry points select ABI and stream variant | W-ENTRY | Version-aware wrapper table and escape-path audit | Interposing ELF symbols catches every runtime path |
| V100 is excluded from NVIDIA's open kernel module target | W-OPEN | Do not start from a supposedly supported Volta open-driver fork | Low-level Volta experimentation is categorically impossible |
| Native Volta compilation needs an older supported toolkit | W-CUDA13, W-LLAMA-CMAKE | Pin qualified CUDA12.x and exact compiler/library stack | A host driver change is required or authorized |

## Measured numbers worth retaining, with boundaries

A-E20 reported 111.9–114.7 GB/s one-way 4 MiB payload transfers within recorded physical pairs 0↔2 and 1↔3, versus 5.16–5.17 GB/s on 0↔1 and 2↔3. Eight directed paths were eligible and four were not. These are historical payload measurements; use actual UUID/stack/configuration receipts and fresh qualification before assuming a runtime cost. Never compare one-way payload with a quoted bidirectional aggregate.

A-E18's 1 MiB dependent-chain medians were 1.312 ms local and 2.450 ms remote. Staging preparation was 0.015 ms followed by a 1.238 ms local chain. The entire chain and preparation are distinct intervals, not isolated memory-operation latency. This motivates comparing staging with reuse rather than treating direct peer reads as universally preferable.

A-E19 reported complete selected handshake times of 0.335 ms for its peer-atomic protocol, 0.0124 ms for its event path and 0.0142 ms for its host-staged event path. The measurements do not isolate primitive costs or prove globally synchronized device clocks. A-E16 likewise observed a slower resident executor for the selected tasks. These negative results are valuable counterevidence to a mandatory permanent GPU rescheduler.

A-E21 measured node0 pinned pages only: node1 binding failed. Do not write “NUMA gives twice the host bandwidth” in a V200 performance claim. Page placement, permitted memory nodes, roots and data-production locality must be established first.

## The novel engineering problem is semantic, not just bandwidth

A single large address range and a single device name solve naming problems. They do not automatically solve kernel scheduling, module globals, atomic scope, library handles, memory pools, graph capture or asynchronous errors. The chosen initial architecture therefore treats identity, allocation and completion as separately qualified interfaces, then optimizes operations with known semantics.

A kernel grid's nominal independence does not mean its compiled physical-device assumptions can be ignored. Even simple grid-stride loops change behavior if physical gridDim replaces logical gridDim. A conservative owner fallback can also be invalid when device-scope operations target remote backing. These cases are explicit contract gates, not merely performance caveats. [A-R06, A-M48, W-CUDA12]

The strongest initial opportunity is **mixed execution**: preserve native owner execution where legal; split a small number of expensive operations correctly; preserve canonical layouts at unknown boundaries; and let observed reuse justify replicas or retained shards. This is a design inference. It has not been benchmarked as V200.

## Precedent, used conservatively

The SnuRHAC developer record describes work on a CUDA runtime, driver-level cluster unified memory and access-pattern analysis to offer one virtual GPU over a multi-GPU system. That is evidence that transparent aggregation has been a concrete systems research problem, not proof that a small drop-in shim for today's llama binary is available. The full paper and reproduction were not analyzed here. Its value is to identify memory-access analysis and runtime/driver cooperation as relevant alternatives. [W-SNURHAC, W-SNURHAC-DOI]

## Remaining uncertainty with bounded discriminators

**U01 — interface closure.** Can the actual executable's runtime, direct driver calls, function-pointer lookup and libraries be consistently mediated? CPU audit followed by one small admitted trace; stop expanding wrappers blindly if the boundary is fundamentally bypassed.

**U02 — pair-backed VMM.** Does the actual pair/stack support the intended multi-owner canonical range from both contexts, including the consumer's direct VMM lifecycle? Small explicit write/read/event test first; larger capacity later. Fixed-alias evidence does not answer this.

**U03 — launch semantics.** Which dominant workload kernels have an exact partitionable contract or a correct semantic replacement? Read their code/layouts and qualify selected shapes with a native oracle. No universal kernel proof effort first.

**U04 — performance.** Does reduced owner work outweigh dispatch, synchronization, communication, conversion and canonicalization? Compare native owner, facade owner, native pair and candidate pair on matched application work.

**U05 — device presentation outside one pair.** Can the consumer use two logical devices without misreading cross-pair peer capability or NCCL/NVML identity? Explicitly distinguish bulk copy routing from arbitrary remote pointer access; qualify after one pair works.

**U06 — host ingress.** Is both-node placement available and beneficial on this actual host? Only a small late placement test if loader time is a material limiter.

**U07 — deeper controls.** Does a specific command/SASS mechanism save a measured cost enough to justify fragility? Keep private driver/firmware branches unmaterialized until the root has a scoped case and user authorization.

## Research completion boundary

This design phase inspected the relevant existing evidence, current CUDA compatibility constraints, source-level consumer boundaries and the actual Todo ingestion contract. It does not claim to have remeasured the machine, read every upstream private ABI, proved arbitrary CUDA transparency or completed all research protocols in the atlas. Those would require the explicitly defined implementation and hardware qualifications, not more confident prose.
