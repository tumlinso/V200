# V200 outcome plan

`proposed_todos.json` is the semantic authoring source. This index and the CSV are derived. Each task is a durable outcome; inspect/code/build/test steps remain local to the implementing agent. Core GPU proof is explicit in integration outcomes, not implied by CPU-ready foundation tasks.

| Outcome | Depends on | Hardware boundary |
|---|---|---|
| [V200-01 — Adopt the charter, source pins and minimal execution foundation](V200-01.md) | start | None; any later read-only runtime inventory belongs to an admitted bundle. |
| [V200-02 — Implement a coherent and auditable CUDA presentation boundary](V200-02.md) | V200-01 | B02 owned by V200-06; no claim of live closure here. |
| [V200-03 — Implement canonical pair memory and ownership lifetimes](V200-03.md) | V200-01 | B01 joint foundation proof in V200-06; B04 capacity in V200-07. |
| [V200-04 — Implement logical completion and safe owner execution](V200-04.md) | V200-01 | B01/B02 accepted jointly by V200-06; no permanent polling benchmark. |
| [V200-05 — Prepare the real consumer and matched native baseline fixture](V200-05.md) | V200-01 | B03 baseline/application session in V200-06; larger fixtures deferred. |
| [V200-06 — Deliver the first real single-device consumer](V200-06.md) | V200-02, V200-03, V200-04, V200-05 | B01, B02, B03 |
| [V200-07 — Qualify useful capacity beyond one physical V100](V200-07.md) | V200-06 | B04 |
| [V200-08 — Accelerate the dominant qualified operation across the pair](V200-08.md) | V200-06 | B03 selected operator/consumer subset |
| [V200-09 — Optimize pair locality and NVLink use with a small cost policy](V200-09.md) | V200-03, V200-08 | B05 in V200-12; a narrow B03 regression only after changed code. |
| [V200-10 — Close the observed compatibility profile and targeted regressions](V200-10.md) | V200-06 | B03 selected regression subset |
| [V200-11 — Expose and integrate two logical V200 devices](V200-11.md) | V200-04, V200-07, V200-08, V200-09 | B06 |
| [V200-12 — Make the measured performance and product decision](V200-12.md) | V200-07, V200-08, V200-09, V200-10 | B05 |
| [V200-13 — Package a reversible deployment and diagnostic surface](V200-13.md) | V200-11, V200-12 | Reuse accepted B03/B06 evidence; one focused deployment smoke only if binaries or loader paths changed. |
| [V200-14 — Integrate and hand off the accepted V200 release](V200-14.md) | V200-13 | Reuse compatible evidence; do not rerun all bundles by default. |

## Critical path and practical parallelism

V200-01 → foundations V200-02/03/04 and consumer fixture V200-05 → V200-06 real consumer → V200-07 capacity and V200-08 acceleration → V200-09 locality / V200-10 observed compatibility → V200-12 pair performance and V200-11 two-device integration → V200-13 deployment → V200-14 release.

The imported schema-v2 plan does not invent a run/lane tree. The authorized root configures actual first-class lanes with the installed workflow surface. Suggested parallelism is advisory until assigned with nonoverlapping paths. Shared interface changes go through the root.

D01–D08 are checkpoints within outcomes, not eight additional bureaucratic tasks. B07 is an optional low-level discriminator, not a prerequisite for completion. Negative evidence can terminate an experiment, but a missing product outcome requires a user-approved change or remains blocked.
