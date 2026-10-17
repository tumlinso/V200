# Canonical Todo import and handoff

This package was prepared against the inspected **Project Control preledger compiler v2**, which emits native Todo **schema v2**. The installed Todo source also understands newer first-class run declarations, but this compiler does not generate them. The package therefore does not pretend that advisory lane names are authoritative run/lane assignments.

## Deployment prerequisites

An authorized operator selects or creates the V200 repository, registers it through the supported Project Control workspace interface, and initializes the corresponding Todo authority using the installed supported workflow. The catalog inspected during assembly did not contain a V200 project. Registration alone does not initialize Todo. No registration, initialization or application was performed here.

Place the complete extracted package at `planning/v200-bootstrap/` in that repository. The task scope and task-sheet references assume this layout. Preserve any existing edits/state. The immutable authoring records stay under this directory; implementation source belongs to the proposed `src/`, `include/`, `tests/`, `benchmarks/`, `docs/v200/` and packaging paths.

The target repository label in the preledger is **V200**. It is not a physical path or assumed existing project identifier. Use the registered project ID when validating/applying. If a different repository label or package location is selected, update the authoritative task records and derived index/manifest deliberately, then recompile. Do not substitute labels at runtime while dropping all selected tasks.

## Compile without mutation

In the canonical installed/source Project Control environment:

```sh
project-control plan compile \
  --package /path/to/V200/planning/v200-bootstrap \
  --repository-label V200 \
  --output /tmp/v200.todo-plan.json
```

The optional adapter uses the same installed compiler, not a bundled reimplementation:

```sh
python planning/v200-bootstrap/tools/compile_plan.py \
  --package planning/v200-bootstrap --repository-label V200 \
  --output /tmp/v200.todo-plan.json \
  --compare planning/v200-bootstrap/v200.todo-plan.json
```

Use `scripts/pc-dev run ...` from the actual Project Control checkout when working in its source environment, as documented there. This package does not install or patch Project Control. A changed compiler hash or changed output requires inspection; it is not fixed by weakening the comparison.

## Validate and apply only in the authorized project

After the real project ID and Todo authority exist, validate with the canonical workflow kernel:

```sh
project-control plan validate --project v200 --file /tmp/v200.todo-plan.json
```

Here `v200` is the **proposed** project ID; use the actual registered ID if different. Review validation/diff and the selected repository authority before an explicit application:

```sh
project-control plan apply --project v200 --file /tmp/v200.todo-plan.json
```

The apply command is a future authorized workflow mutation. It was not run during assembly. Never hand-edit SQLite, claims, task projections or generated current-state artifacts to simulate import.

## Configure actual execution, not a fictional run tree

The root uses the installed native task/run/lane surface to establish execution and take V200-01. This release intentionally has 14 coarse outcomes and ordinary dependencies, not a predetermined swarm of agents. Suggested parallelism lives in task guidance. Declare real lanes/workspaces only when concurrent work and stable interfaces justify them.

The compiler preserves task mechanism/validation/completion information in provenance notes. It does **not** translate arbitrary custom fields into executable command gates, frozen interfaces or resource claims. This package does not invent those claims. The assigned root binds the current task's CPU gate and, where actually needed, a ready GPU bundle using the existing Todo/CUDA-controller integration. Avoid making every finish action rerun all hardware bundles.

The supporting interface/decision documents are design contracts, not falsely published implementation interfaces. Freeze a real interface only when its owner and content are known. Conditional alternatives in D01–D08 are selected inside outcomes; materialize only the selected branch if a new ownership boundary is necessary.

## Evidence reuse and final acceptance

Package validation proves file integrity, canonical compilation and the included CPU logic. It does not satisfy implementation tasks. GPU bundles remain NOT_RUN until actual admitted runs produce case-level evidence. Existing atlas evidence is prior evidence with explicit limits, not new V200 qualification.

Finish implementation through the normal native outcome/checkpoint/integration workflow, with current producer commits and evidence. Preserve negative experiments and unresolved user decisions. Do not mark the release complete until single-device presentation, capacity, useful dual compute and the agreed two-device scope are actually accepted, or the user explicitly revises the charter.
