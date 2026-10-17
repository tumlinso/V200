# Evidence-driven decisions and user checkpoints

These choices are declarative guidance, not extra tasks that all have to run. The root records the selected branch in Todo's supported decision/context surface and in the project decision log. A join waits for the chosen outcome, not every hypothetical branch.

| ID | Trigger and minimum evidence | Root may decide | Ask the user when |
|---|---|---|---|
| D01 — front door | Actual binary/DSO/entry-point audit and bounded closure test | Anchor vs synthetic handles; wrapper organization | Only a rebuild, source patch, backend change or extensive private ABI route can deliver the goal |
| D02 — pair memory | Both-direction VMM access, failure cleanup and consumer VMM lifecycle | Extent sizes, reserves, ownership within canonical contract | Host oversubscription, loss of single large allocation, or explicit app sharding replaces aggregate HBM |
| D03 — first acceleration | Measured dominant operation and matched alternatives | Library split vs qualified kernel replacement; owner vs split | A different application interface or substantially narrower binary contract is required |
| D04 — performance frontier | Matched native baseline, repeated practical difference, uncertainty | Tile/layout/communication tuning | Transparency costs materially more throughput/latency, or prefill and decode objectives conflict |
| D05 — GPU budget | Bundle admission/cap and meaningful missing evidence | Reduce sweep, defer bundle, continue independent CPU work | More GPU/time/memory/disruption than agreed ceilings is necessary |
| D06 — low-level escalation | Exact mechanism, available interface, expected saved cost, rollback | PTX work within explicit task and safe process boundary | Private driver/channel, MMIO/PTE, firmware, reset or system-wide effects are proposed |
| D07 — representation/numerics | Equivalent useful work with tolerance and conversion costs | Exact layout or numerically permitted decomposition | Lower precision, altered model/configuration or relaxed semantic contract would be required |
| D08 — delivery scope | Real consumer acceptance and remaining support debt | Package layout and narrow regression set | T1 is proposed as final instead of T2, or general-device goal is to be reduced |

## Decision note template

**Question:** one concrete user preference needed.

**Observed:** executable/commit/model/pair; tested operation; matched baseline and candidate; sample count; metric difference and uncertainty. State missing measurement as missing.

**Interpretation:** mechanism supported by the evidence, separately from hypotheses.

**Choices:** original target and the strongest alternative; performance, compatibility, implementation and GPU-time consequences. Do not create a bogus binary choice when an incremental hybrid exists.

**Recommendation:** root's technical recommendation and why, with confidence. The user decides the product trade-off.

**Until answered:** affected outcome blocked; safe independent CPU work allowed; no GPU lease retained.

## Initial decision bands

Provisional engineering defaults: investigate repeatable greater-than-10% matched application regressions; do not promote low-single-digit differences without enough repeatability; aim for low-single-digit facade overhead on fitting controls. These are neither hardware facts nor automatic product rejection. Report prefill, decode and memory benefit separately. The user has not selected an arbitrary throughput/latency weighted score.
