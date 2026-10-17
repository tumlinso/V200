# Paused checkpoint — 17 October 2026

Paused at the user's request. No performance run or four-GPU run should start until resumed. The active V200-12 claim was released through Project Control; V200-11 workers stopped before editing source. No GPU compute processes remained at the pause check. Project Control stays active; its optional inference service remains inactive.

Accepted working milestones:

- `d0a81e8` / V200-07: unchanged llama-server, one visible CUDA device, 18,550,716,425 bytes of model tensors fully GPU-resident across two V100s; peak managed backing 19,436,404,736 bytes; correct answer and zero final accounting.
- `a22190c` / V200-08: real Q4_K matvec execution on both owners; 4,096 peer outputs bitwise equal to the original kernel.
- `8ffe592` / V200-09: owner-local fused expert execution, including 71 peer calls and a 6,144-output bitwise comparison; model identity caching reduces repeated startup scans.
- `2bd3020` / V200-10: graph capture and replay with up to 34 expert nodes redirected to peer-context functions, correct response, and clean release. Graphs remain an explicit `--cuda-graphs` option.

The final CPU repair removes a leftover smoke-only guard from the full-resident finite service path; 27 fixture/launcher checks pass. This repair received no new GPU run because work is paused. The retained graph evidence verifier still passes.

Single-matrix splitting and the plain expert family remain CPU-checked/unobserved on hardware. There is no steady-state performance or two-logical-device claim.

Resume with Project Control `next_task` for run `V200-PAIR-PERF-MULTIPAIR-1`, task `V200-12`. The next work is a small matched comparison using the qualified libraries, then V200-11. Its plan is `planning/performance-multipair.todo-plan.json`. No benchmark implementation or multi-pair source edits have landed yet. Use a separate multi-pair build directory to preserve the working binaries.

Finite full-resident server command (leave off `--execute` for a dry run):

```sh
python3 scripts/v200/serve.py --mode serve --workload resident_capacity \
  --pair-compute --cuda-graphs --duration-seconds 300 --execute
```

Do not execute it while this pause remains in effect. Omit `--cuda-graphs` for the qualified graph-off route; omit `--pair-compute` for the capacity-only route. Running the installed llama-server without V200 preload restores native behavior.
