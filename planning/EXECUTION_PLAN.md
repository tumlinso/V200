# V200 execution plan

Recorded 17 October 2026. This is the local adoption plan for the supplied bootstrap, with the user's delivery preferences applied. It supplements the preserved package; it is not Todo authority.

## Objective and first workload

Deliver useful software early and improve it while it is used. The target remains one application-visible CUDA-compatible device per NVLink V100 pair, with useful aggregate memory and eventually useful compute on both members. The user selected the existing llama.cpp / llama-server workload for the first deliverable.

Host PATH currently resolves `llama-server` and `llama-cli` under `/home/tumlinson/Software/llama.cpp/build/bin/`. They were not executed. Their binary hashes, actual linkage, model fixture, configuration, and runtime compatibility still need to be pinned. The package's source pin is a proposed comparison baseline, not proof of the installed executable's identity.

## What arrived

- A design and source register, compatibility tiers, six semantic interface contracts, and eight decision checkpoints.
- Fourteen proposed Todo outcomes with 24 dependency edges, supported authoring records and canonical compiled plan.
- Seven dormant, bounded GPU bundles. Their definitions are test inputs, not successful execution evidence or an active schedule.
- A CPU allocation/grid-planning example, a separate unqualified CUDA Driver VMM/event probe, and package/ELF/bundle/runner tools with tests.

This is a substantial design bootstrap, not an operational CUDA compatibility layer. Its hardest unproven assumptions are coherent interception of the actual consumer's CUDA surface and legal, useful access to memory backed by both GPUs. Allocation and enumeration demos alone do not establish either whole-application correctness or acceleration.

## Delivery order and amendments

Keep the existing 14 outcomes as broad tracking boundaries. Do not launch 14 agents, create an alternative scheduler, or build speculative implementations of every design branch.

1. **Resolve the actual consumer boundary first.** Pin the existing executable and an existing model; inspect its ELF/source dependencies and driver-entry lookup. Define a small correctness fixture and matched native control. Establish the supported Project Control/Todo authority before implementation claims. Agree only the identity, allocation, completion and dispatch interfaces needed by the first owners.
2. **Run the smallest discriminators early.** Build the foundational pair-VMM probe before GPU admission; qualify two-owner mapping, access and completion in one short admitted session. Trace the real consumer's reachable CUDA entry points as soon as a safe harness exists. These checks precede major allocator, graph or kernel-transform investments. A failed check stops only the dependent implementation; other CPU work continues.
3. **Deliver the first integrated preview at V200-06.** One real llama workload selects one logical device, returns checked output and has a reproducible launch command, limited support statement, diagnostics and a reversible disable path. Initially conservative compute behavior is acceptable when correct and disclosed. This proves integration; it does not yet establish added capacity or speed.
4. **Prioritize V200-07 for the first capacity benefit.** Demonstrate useful resident application payload exceeding either physical V100's available memory, with both owners accounted for and no silent host offload or workload reduction. This is the first milestone that establishes added capacity. Do not promise all nominal 32 GB as available payload.
5. **Optimize the measured bottleneck.** Select the dominant qualified operation for V200-08, then only the locality and compatibility work needed by that profile (V200-09/10). Report application throughput/latency and capacity benefit separately. Retain a faster legal owner route when splitting loses.
6. **Expand after feedback.** Two logical V200s, broader application/API coverage, full packaging and release acceptance follow once the one-pair preview is useful. Keep V200-11 through V200-14 as full-release outcomes; they do not block distributing a clearly labeled earlier preview.

Amendment: bring the minimum launcher, usage instructions, disable path and support limits forward from final packaging into the V200-06 preview. Do not wait for four-GPU integration or broad optimization to make the first profile usable. This changes delivery scheduling, not the full product acceptance contract.

Aim at T2 first: the actual qualified executable remains unchanged apart from launcher/injection plumbing. If a source-unmodified rebuild or restricted configuration (T1) is the quickest useful intermediate, report its exact restrictions and delivery trade-off; do not spend a long cycle silently chasing T2 or label T1 as T2. A need to patch consumer source, adopt a special backend, or abandon aggregate-memory/transparency goals is a product decision for the user.

No credible completion date or performance claim is available until the actual ABI boundary and pair-memory discriminators are resolved. Report those findings promptly rather than forecast from the task count.

## Parallel Luna ownership

Use four substantial implementation streams after the minimal shared interface is agreed. Each implementer owns investigation, design, implementation, debugging and focused tests for its objective. Configured Luna profiles do the engineering; no Project Control coding workers. The root owns the common build, shared public headers, integration, workflow lifecycle and final acceptance.

| Stream | Objective | Proposed exclusive ownership | Return condition |
| --- | --- | --- | --- |
| A: CUDA boundary | V200-02; versioned entry-point interception, coherent device/context/handle presentation, explicit unsupported cases | `src/frontend/`, `tests/frontend/` | Observed consumer surface accounted for; focused closure harness and exact gaps |
| B: pair memory | V200-03; canonical VA, compound/physical handles, extent ownership, rollback and deferred release | `src/memory/`, `tests/memory/` | Ledger and cleanup checked on CPU; ready pair test; GPU claims require actual receipts |
| C: completion and owner execution | V200-04; streams/events, complete-prefix tokens, context restoration, errors and lifetimes | `src/execution/`, `tests/execution/` | Dependency/error/lifetime cases checked; explicit integration seam to A and B |
| D: real consumer | V200-05; exact binary/model profile, oracle, native controls and launch fixture | `tests/consumer/`, `benchmarks/fixtures/`, `docs/v200/consumer-profile.json` | Reproducible fitting fixture, candidate capacity fixture and stated baseline boundaries |

These paths are proposed ownership, not files already implemented. Root integrates the seams continuously using a small vertical test: resolve an entry point, allocate, execute, observe full completion, free. Avoid four isolated frameworks that first meet at the end. Shared-header changes have one owner and a short explicit contract update.

Keep A and D in immediate contact: D's observed surface determines A's scope. Keep B and C aligned on pending-use tokens and free/unmap semantics. Run CPU work concurrently; coordinate GPU sessions as one bounded queue rather than four competing reservations. Root may assign one worker to execute a selected bundle under the parent scope.

Use a scoped Luna verifier/reviewer at meaningful joins or for consequential memory/ABI changes; recycle completed workers. Review independently against the contract rather than merely repeating implementation tests. Do not create a permanent reviewer lane, duplicate source surveys, or add extra agents unless an independent objective justifies them. The session cap is 16 spawned agents total, including nested workers; four implementation streams and a recycled scoped reviewer are the normal shape. Sol is exceptional; no Astra delegation.

## Keep the user's attention cost low

Handle routine mechanisms, layouts, build repairs, test fixes and supported task operations autonomously. Reuse existing assets and accepted evidence. A focused change receives focused tests and review. Re-run broader checks only when changed behavior or unresolved evidence warrants it.

Provide updates when real behavior changes or a blocker changes the route: what works, the exact limitation, and the next useful result. Ask the user only for a consequential product choice or genuinely missing input that cannot be inferred or found safely. Present a concrete recommendation and trade-offs in that case. Never hold a GPU lease while waiting for an answer.

The user explicitly authorizes eviction of running llama servers and Project Control GPU services when GPU work needs the devices. Use the supported controller/service lifecycle, verify the selected processes and released resources, and record what was stopped so required services can be restored deliberately. Prefer graceful eviction; that permission does not justify killing unrelated GPU jobs or bypassing controller admission. No eviction is needed for this planning/CPU-validation work. No continuous watch is requested.

## Current evidence and execution prerequisites

Completed in this preparation:

- All 66 archive files extracted to `planning/v200-bootstrap/`; all 65 manifest-listed file hashes passed. Package files remain unchanged.
- Package validator: 385 checks, 14 tasks, 24 edges, 7 bundles, no errors.
- Python tools: 14 tests passed.
- CPU-only CMake configure/build and CTest passed; planning executable reported 1,461 checks and `gpu_evidence:false`.
- Current canonical Project Control CLI compiled all 14 tasks with 24 edges and no warnings. Output at `/tmp/v200-compiled-plan.json` matched the shipped `v200.todo-plan.json` byte for byte. The helper's ordinary Python environment lacks `project_control`; use the canonical CLI/runtime.
- V200 is registered; Git and Todo authority are initialized, and the 14-outcome plan is applied at Todo revision 2. No implementation task has been claimed or completed. See [bootstrap status](BOOTSTRAP_STATUS.md).

Validation used Python 3.12.3, g++ 13.3.0 and CMake 3.28.3. Temporary CPU build outputs are under `/tmp/v200-bootstrap-validation/`; they are disposable, not delivery artifacts. Checks prove only the tested host code/package behavior. No CUDA build, driver probe, GPU qualification, consumer execution or speedup measurement ran.

The initial setup prerequisite is resolved: Git is initialized on `main`, the requested GitHub remote is configured, and Project Control's supported workflow front door is installed. The native plan was validated against the actual authority identity before application. Todo reports only V200-01 ready, 14 planned outcomes, no active claims, and a clean doctor result. Start implementation with the supported `next_task` workflow; do not claim all four streams before their foundation is accepted. Earlier restricted connector reads were inconclusive, so use current authoritative identity and source-backed workflow checks rather than those stale results. No Git metadata, database, or generated Todo projection was hand-edited.

The original archive stays intact. Scheduling amendments live in this overlay so package provenance remains verifiable. If authoritative tasks need changes at adoption, derive, compile, validate and review an amended payload through Project Control rather than editing SQLite or claiming this Markdown is current task state.
