# V200 — research, design and Todo bootstrap

**Release:** 1.0 · **Prepared:** 17 October 2026 · **State:** design and bootstrap, not an operational virtual GPU.

## The product

Present each NVLink-connected pair of 16 GB V100 SXM2 GPUs as **one ordinary application-visible CUDA-compatible accelerator**, nominally 32 GB aggregate HBM and 160 physical SMs. Existing applications should select that device without configuring its two physical constituents. Four physical V100s become two logical V200s. Preserve locality internally and optimize useful end-to-end performance, not the appearance of uniform hardware.

This package recommends a **process-scoped, hybrid CUDA compatibility layer**, a canonical virtual-memory view, explicit completion tracking, and selectively qualified distributed execution. It does **not** assume that intercepting device enumeration, concatenating VRAM, or adjusting block indices is sufficient. The first implementation must test those hard compatibility boundaries early, then reach a real application quickly. The implementing root can change internal mechanisms in response to evidence. Reducing the external product contract requires a user decision.

## Read in this order

1. [Design](docs/DESIGN.md): selected architecture, alternatives, interfaces, correctness, performance model and expansion route.
2. [Research findings](research/FINDINGS.md): what the installed atlas and current upstream sources actually establish.
3. [Execution charter](contracts/EXECUTION_CHARTER.md) and [decision checkpoints](contracts/DECISIONS.md): autonomy, escalation and bounded GPU use.
4. [Outcome plan](tasks/INDEX.md): 14 durable implementation outcomes. Their authoritative authoring records are `proposed_todos.json`.
5. [Import and handoff](docs/IMPORT.md): canonical compilation and safe adoption into the existing Todo workflow.

The [implementation handoff](IMPLEMENTER.md) is the entry point for the next agent. The [compatibility contract](contracts/COMPATIBILITY.md) prevents a restricted prototype from being mislabeled as universal CUDA compatibility. [GPU bundles](bundles/README.md) are dormant test definitions; nothing schedules or arms them automatically.

## Included examples

`examples/` contains a CPU-buildable allocation/grid planning library and a Driver API pair-VMM/event qualification program with an embedded, explicit-index PTX kernel. The latter is a **small hardware probe and example**, not a CUDA virtualizer. It requires a qualified CUDA 12.x build environment and an admitted pair. No CUDA execution was performed while assembling this package.

`tools/` contains an ELF dependency inspector that never runs the inspected application, a strict bundle-to-controller spec renderer, a bounded child runner, a canonical-compiler adapter, and package consistency checks. None is a replacement for Todo or the CUDA controller.

## Validation and boundaries

See [validation report](validation/REPORT.md) and `validation/results.json` for checks actually performed. The authoring package is compiled using a byte-verified copy of the inspected Project Control compiler; that copy is **not** shipped or installed. Native plan application and validation against a live V200 Todo authority remain deployment steps. No V200 project was registered, no connected repository was modified, and no GPU was reserved by this work.

The plan's target repository label is `V200`. It is a proposed new authority, not an assertion that a registered project already exists. Default installation layout places this folder at `planning/v200-bootstrap/` within the eventual V200 repository.

## Outputs

- `proposed_todos.json` + `preledger.json`: supported authoring format.
- `v200.todo-plan.json`: deterministic schema-v2 compiler output, not an applied plan.
- `dependency_edges.csv`: checked derived edge index.
- `docs/`, `contracts/`, `research/`: design and executable acceptance guidance.
- `examples/`, `tools/`, `tests/`: starter code and CPU checks.
- `bundles/catalog.json`: seven bounded GPU bundles, all initially `NOT_RUN`.
- `MANIFEST.sha256`: release integrity, independent of task execution evidence.

The fastest useful path is **boundary closure → legal pair memory/completion → real single-device application → capacity beyond one V100 → measured dual-compute acceleration**. Do not spend the first iteration implementing a universal SASS rewriter or a new benchmark platform.
