# Application-visible compatibility contract

## Transparency levels

| Tier | Observable promise | Acceptance label |
|---|---|---|
| T0 | Test program calls a V200-specific API | Development evidence only; not the requested presentation |
| T1 | Real consumer source unchanged, but rebuilt/configured for a restricted boundary | Useful intermediate; exact flags disclosed |
| T2 | Existing qualified executable unchanged; launcher/injection and ordinary single-device selection only | Initial product target |
| T3 | A broader, versioned set of ordinary CUDA binaries, graphs/libraries and two logical devices | Expansion goal, never implied by T2 |
| T4 | System-wide synthetic hardware/device identity | Not required for initial consumer contract; operator approval before such changes |

A result must state its actual tier. T1 is not secretly relabeled T2; T2 is not universal CUDA. App uses one logical ID, no explicit two-physical-GPU split. Applications outside the launcher remain on normal CUDA.

## First qualified profile

Linux x86-64; one process; two V100 SXM2 `sm_70` devices with verified direct NVLink peer path; pinned CUDA 12.x toolkit/runtime/libraries; one concrete consumer binary and model fixture. Exact kernel/module hashes and graph mode are recorded. The two-V200/four-physical-GPU profile is separately accepted.

The first profile must cover the operations actually exercised, including direct VMM if used. Unknown features are surfaced as unsupported with operation and qualification reason. IPC, CUDA dynamic parallelism, cooperative multi-grid execution, textures/surfaces, MPS, runtime-private APIs, memory pools and arbitrary graph updates are **not presumed supported**. Support them when observed and justified, or refuse before incorrect work. A consumer requiring one of them makes it part of that consumer's critical path.

## Resource claims

Memory queries report bytes under the logical allocator contract and physical reserves. Capacity proof requires resident live payload beyond either physical device's available capacity, with both owners' allocations recorded and no hidden host oversubscription/offload. Nominal 32 GB is not free capacity.

Keep native architecture capability 7.0 and physical per-block limits. The desired 160 aggregate SM report is conditioned on a coherent launch/occupancy feature contract; unsupported cooperative behavior is not inferred from this number. Diagnostics retain both physical devices. Global `nvidia-smi` output is not the acceptance surface.

## Semantics that cannot be silently weakened

Correct device/context ownership; stream prefix completion; event order; default-stream mode; argument lifetime; all advertised allocation and pointer query behavior; explicit numerical contract; error propagation; no unqualified alias coherence; no duplicate logical blocks; no use-after-free while peer work remains in flight.

Opaque owner execution is valid only when its referenced memory and synchronization scopes are legal. An anchor-only run using remote backing is not automatically a safe fallback for every CUDA kernel. The profile must identify the fallback's supported memory/operation class.

## Pass evidence

Keep executable/source/library/model hashes, invocation without physical multi-GPU flags, logical enumeration, physical allocations, output checks, a representative trace showing useful work on both devices for accelerated acceptance, matched baselines, and unsupported-feature records. Sanitizer runs supplement correctness; they do not prove memory-model legality.
