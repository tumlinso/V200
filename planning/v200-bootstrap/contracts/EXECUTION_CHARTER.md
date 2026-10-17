# V200 execution charter

## Authority and fixed objective

The user authorized research, design and assembly of this package. **Implementation, Todo application, GPU admission, continuous watches, driver changes and system configuration are not performed by package assembly.** The next execution owner must receive the appropriate workflow and hardware authorization.

The invariant product goal is one ordinary application-facing CUDA-compatible device per NVLink V100 pair, with useful aggregate memory and dual compute. Performance selects internal mechanisms; correctness constrains them. No silent conversion into a conventional application-specific multi-GPU backend.

## Agent autonomy

The root owns architecture, final acceptance and user questions. Workers own bounded files/outcomes delegated under the root's active task. Use Project Control/Todo's existing lifecycle rather than a separate queue or direct SQLite edits. Tactical choices—layouts, event pooling, batching, algorithm thresholds, owner placement, code organization, focused tests—are autonomous within the accepted contract.

Create durable tasks only for meaningful outcome, dependency, ownership, interface, integration or risk boundaries. The 14 tasks in this bootstrap do not require 14 agents or parallel worktrees. One root can execute them with a few bounded helpers. Read relevant task sheets, not the entire atlas every time.

## Hardware economy

Build and CPU-check before acquiring a GPU. Reuse the atlas's dated evidence with its limits. Run a short foundational qualifier early; defer bulk capacity, performance, four-GPU and advanced-mechanism tests into the specified bundles. Do not attach full GPU suites to every commit or task completion.

Default bundle caps in `bundles/catalog.json` are engineering ceilings, not estimates of work promised to the user. They do not authorize execution. Use actual controller admission, runtime UUIDs, topology checks, idle checks and release receipts. A busy user's GPU means **defer**, not kill or reset. Do not hold a lease while compiling, waiting for the user, searching documentation or deciding what to test.

An optional background watch must be explicitly armed by the operator through the existing CUDA controller, scoped to a ready revision and bounded bundle. This release arms nothing. The small package renderer is only a spec generator, not another scheduler.

## Proof and reporting

Separate `CPU_PASS`, `COMPILED_ONLY`, `GPU_PASS`, `NOT_RUN`, `BLOCKED`, `INCONCLUSIVE` and `FAILED` in evidence. A CPU simulation cannot satisfy a GPU acceptance condition. A negative experiment can finish an investigation but cannot count as delivering the product it disproved. A setup-excluded kernel win cannot be promoted as an application win.

At a meaningful milestone report: what now works; exact application/stack profile; matched measurement and uncertainty; current limitation; next bounded outcome. At a material product trade-off use the decision note and ask the user. In the absence of a user decision, keep the original objective; do not silently select the cheaper alternative.

## Non-negotiable prohibitions

No unrelated repository edits, resets, force pushes, service changes, driver upgrades/downgrades, clock changes, firmware/PTE/MMIO experiments, or killing foreign GPU processes. No broad network/model download without a chosen fixture and authorization. No fabricated successful GPU evidence. No changing tests or numerical tolerance solely to hide a defect.
