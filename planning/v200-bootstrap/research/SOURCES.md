# Source register

References are evidence locators, not execution permission. Connected files were read directly through Project Control. The package ships original synthesis, not a copied hardware atlas or a fork of Todo. Hashes identify inspected content even when a repository is dirty. Re-fetch before relying on changed contracts.

## A-SKILL

Project/repository: `skills`. Path: `cuda/SKILL.md`.

SHA-256: `b821b34a890d6081ebf9ad1c83cb4f5345ed6cf51c62342749a4d3a1004905b8`.

Full entry instructions; advisory, no GPU execution authority.

## A-HOST

Project/repository: `skills`. Path: `cuda/references/execution/host-execution.md`.

SHA-256: `5e8ba8349c57fe513c084a23b444ed44df72682c91947ca192320b30a5a9714d`.

Host controller contract; build before reservation and quiescence.

## A-BACKGROUND

Project/repository: `skills`. Path: `cuda/references/execution/controller-background-contract.md`.

SHA-256: `39b355e5eb57099f71c7247c82099d3641b381b3f96c3b7fa8e4a76045dc836c`.

Foreground/background spec and explicit arming, no watch armed.

## A-CONTROLLER

Project/repository: `skills`. Path: `cuda/scripts/cuda_controller.py`.

SHA-256: `07fd3704bdb441479c8accd75075e83fbccec824fd8d0b6e73f5e1df0c344959`.

Selected ranges, including explicit resources.gpu_uuids and directed admission; not full source validation.

## A-LEASE

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/scripts/atlas_host.py`.

SHA-256: `c588d8a1028a280996c6cd286d82845fb38c70ceefd1675297eec22b75b26f49`.

Selected lease/visible UUID check implementation; use canonical mechanism, not a new reservation manager.

## A-STATUS

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/CURRENT_STATUS.md`.

SHA-256: `f3f9285ab7d0c85f79c1c83141aa1f6d7a40abbd722b6122ee6b73285223e46f`.

Dated 2026-10-06 evidence status; not every protocol executed.

## A-NEEDS

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/NEED_INDEX.md`.

SHA-256: `ddfc3bd88ce00eaf7b625b7291feadf0528f7c110427a8f2915a8546bf0ebc82`.

Navigation only, not mechanism proof.

## A-R06

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/reference/R06-participation-visibility-and-progress.md`.

SHA-256: `4b02909469914911d50a73f3128880cec97a2d7f02060a4ae280d98c85718a1c`.

Participation, visibility and progress contracts.

## A-R07

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/reference/R07-nvlink-pcie-and-numa-as-information-boundaries.md`.

SHA-256: `8a2c22bf34971da6c9571cbe6fdacf97293298278b79ba7ac8ecf253d4164b4b`.

Directed topology and pull/push/staging trade-offs; no generic coherence or transit routing.

## A-R08

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/reference/R08-virtual-memory-ats-and-faults.md`.

SHA-256: `89d993bf0825ef15c198bcbaa6921f39cee5b4553f7f06c2ce040b7d622c7a4a`.

VMM/UVA/managed memory distinct; platform-specific ATS and fault machinery.

## A-R09

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/reference/R09-the-non-sm-command-machine.md`.

SHA-256: `aae12126bafb2f528ecd144780ba9c87793adbc82c1ceb94139291c480641dd8`.

Restricted CE/QMD machinery; not arbitrary DMA vector arithmetic or public command permission.

## A-M48

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/mechanisms/M48-cubin-and-sass-transformation.md`.

SHA-256: `af7b5233f9c337c247e82fe5229944c06198bae4cf63b8575cf27e51d41c8615`.

Binary metadata and qualification requirements, not an arbitrary kernel rewriter.

## A-E00

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E00-inventory-and-capability-gates.md`.

SHA-256: `bdb151d8c67f43e0775aacd9c8fb4608a2fd052fbe251fa9b241a9840613ed36`.

One read-only inventory; no performance proof.

## A-E16

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E16-persistent-roles-and-residency.md`.

SHA-256: `89775e284b94e0e74dc806f4a9ab3511e7bc9300d6de6da6e52bba02ece30086`.

Selected resident/batched/graph cases only.

## A-E18

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E18-peer-loads-cache-behavior-and-latency.md`.

SHA-256: `70847696a231f80fb049041ea5705b8dbf58e8bdaec5f7a16b3011ef39f4b6da`.

One pair, dependent/independent/staged accesses; not cache coherence.

## A-E19

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E19-peer-signaling-and-clock-relationships.md`.

SHA-256: `ab12bc9e78dee31cb81ab619cb249179d4710be99fe6ba5fba4c53bc2f43dd88`.

Complete 64-phase protocols, not primitive latency.

## A-E20

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E20-actual-nvlink-routing-and-saturation.md`.

SHA-256: `9b7925b29101332b9815d5c6ae4941bc971db4a3090249759bf3a21bf70bf2e0`.

4 MiB directed rates; no independent route/saturation counters.

## A-E21

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E21-host-numa-ingress.md`.

SHA-256: `d8f91b29c268f8167a45696b24f3cc3c0d6b279e8ed637bd9a558241ac527f43`.

Only one NUMA node measured successfully; no local/remote comparison.

## A-E22

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E22-stream-memory-operations.md`.

SHA-256: `af414f8e137c605b23c4da1e0b7b159ca6375b4f915a113dc2a3271cb503ac03`.

Reachability with event ordering; no established speedup.

## A-E23

Project/repository: `skills`. Path: `cuda/references/architectures/volta/v100_atlas/experiments/E23-vmm-alias-and-view-behavior.md`.

SHA-256: `82ca70049a240cbc8a73b44dd3e3290e9c8457b603c914a1a8293efd941f9410`.

Two fixed aliases of one allocation; not multi-device spanning backing.

## P-README

Project/repository: `project-control`. Path: `README.md`.

SHA-256: `c632d2df9d01cf22a0f96064e9c471ff0facd8638de0763d0fc6c69209f5609c`.

Current preledger CLI and source lifecycle.

## P-ADAPT

Project/repository: `project-control`. Path: `docs/ADAPTIVE_EPICS.md`.

SHA-256: `881eea6774f4995cdc6d4899eb2b7a3c48be13df0298aad71e45b227050c5f9e`.

Durable outcomes, selected alternatives, root autonomy.

## P-PRELEDGER

Project/repository: `project-control`. Path: `src/project_control/preledger.py`.

SHA-256: `f47180104ca5b8b219d057a86339fc2c0d018e2e8e0eae12b88c94e6a4f2f339`.

Full compiler read; exact byte-verified temporary copy used locally, not shipped.

## P-PLAN

Project/repository: `project-control`. Path: `src/todo_orchestrator/plan.py`.

SHA-256: `fde2ca5a4b74ec47dc02d46d1ca9d286c902074f3589d1886988e348243e170d`.

Validator source lines 1–220 read; full live Todo validation not claimed.

## P-INGEST

Project/repository: `project-control`. Path: `docs/mutation/V1_CONTRACT.md`.

SHA-256: `30029c9fed0da3dd984441349432b980995cdadc289b463dfe201b0c9746de39`.

Supported preledger fields; historical public tool counts not used.

## W-CUDA12

CUDA C Programming Guide 12.9.1.

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html

Multi-device streams/events, VMM alias rules and generation-specific semantics.

## W-VMM

CUDA Driver API 12.9.1: Virtual Memory Management.

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__VA.html

Reservation/backing/access separation, zero map offset and whole mapped extent unmap.

## W-ENTRY

CUDA Driver API 12.9.1: Driver Entry Point Access.

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html

Version-selected function pointers and default-stream variants.

## W-CUDA13

NVIDIA: What is new and important in CUDA Toolkit 13.0.

https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/

Pre-7.5 offline compilation support removed; do not use CUDA13 for native Volta compilation.

## W-OPEN

NVIDIA open GPU kernel modules.

https://github.com/NVIDIA/open-gpu-kernel-modules

Supported GPU compatibility is Turing and newer, not Volta.

## W-LLAMA-CMAKE

llama.cpp CUDA build rules.

https://github.com/ggml-org/llama.cpp/blob/83f9389419ef93117d1444f87adfcd9f03ec0509/ggml/src/ggml-cuda/CMakeLists.txt

Pinned source; conditional static linkage, driver VMM linkage, graphs/custom CUDA compilation.

## W-LLAMA-CUDA

llama.cpp CUDA backend.

https://github.com/ggml-org/llama.cpp/blob/83f9389419ef93117d1444f87adfcd9f03ec0509/ggml/src/ggml-cuda/ggml-cuda.cu

Source baseline for direct VMM and custom-kernel dispatch; actual host binary separately audited.

## W-LLAMA-SERVER

llama.cpp server options.

https://github.com/ggml-org/llama.cpp/blob/83f9389419ef93117d1444f87adfcd9f03ec0509/tools/server/README.md

Ordinary device selection, split/fit options; benchmark exact chosen build flags.

## W-SNURHAC

SnuRHAC original developer record.

https://gangwon.dev/

Author reports runtime, driver-level cluster unified memory and memory-access-pattern analysis for single-virtual-GPU CUDA; no reproduction attempted.

## W-SNURHAC-DOI

SnuRHAC, HPDC 2021 bibliographic record.

https://doi.org/10.1145/3431379.3460647

Precedent only; full paper not analyzed in this session.

## Evidence weighting

API guarantees, specific measurements and design hypotheses are separate. The atlas retains its original authoring statements and later campaign updates; use dated coverage instead of assuming the original NOT_RUN text remains the latest state. Papers/patents/reverse engineering are not equivalent to vendor API guarantees. No source establishes a completed V200, arbitrary split-safe kernels or a measured application speedup here.
