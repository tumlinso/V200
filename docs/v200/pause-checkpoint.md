# Requested pause checkpoint — 17 October 2026

Paused at the user's request after the locality fix, qualification, and publication. V200-14 remains unfinished; no broader product acceptance is implied.

## New working result

The opt-in `V200_PREFER_LOCAL_HBM=1` policy selects the largest legal preferred-owner extent instead of the smallest. Large allocations leave 1 GiB of soft anchor workspace headroom, with full-budget fallback. The same candidate binaries measured **78.053 tokens/s with locality enabled versus 63.146 with the original placement** (23.6% improvement); matched native measured 123.357. All warmup/measured outputs matched native exactly. Full model residency, actual peer Q4/Q6 computation, CUDA graphs, normal exit and zero accounting passed. Both admitted owners were released. CPU 15-test candidate suite, 45 consumer tests within it, 16 comparison tests and 4 deployment tests passed; saved-evidence verifiers passed.

The candidate is in `build-perf`. `build/` and `build-multipair/` remain the frozen working previews; their recorded library hashes are unchanged. Build and use the qualified single-pair candidate:

```bash
./packaging/build.sh --candidate
python3 scripts/v200/serve.py --mode serve --workload resident_capacity \
  --pair-compute --extended-expert-compute --prefer-local-hbm --cuda-graphs \
  --duration-seconds 300 --execute
```

The public flag selects the already qualified candidate library paths; its environment/path wiring passed CPU tests and a dry plan. The qualification ran the same DSOs/consumer through the bounded comparison harness; the public CLI itself was not given an additional GPU smoke. The existing two-logical-device server retains its previous profile and graphs-off boundary.

## Evidence and code

- Runtime implementation: `aaf01a7`; finite trace/comparison support: `a3d4ca9`. These commits follow the earlier pushed preview checkpoint `b78b5e2`.
- Locality/native comparison: `evidence/B08/locality-r1/comparison/comparison.json`, controller `827f2d85-d588-4403-84ff-56d185933cdb`.
- Cache-only same-binary ablation: `evidence/B08/cache-only-r1/comparison/comparison.json`, controller `d2b1df8f-1b21-4a5d-862f-2ca225abb94f`.
- `evidence/B08/locality-qualification.json` pins current candidate artifacts and physical UUIDs. `tests/release/verify_locality.py` verifies retained results and current candidate/frozen library identities without GPU execution.
- The failed Nsight attempt is retained separately as `trace-baseline-r1`, controller `da67c8cc-2ac9-49e4-b4c0-d050948188b9`. It produced no CUDA kernel data or measured NVTX range, stalled during setup/warmup and required SIGKILL cleanup. Canonical host resources were released and physical GPU allocations reclaimed. Its partial summary is not steady-state evidence.

## Resume

The biggest confirmed tested loss was placement; the remaining 78-versus-123 gap is unresolved. The changed policy also reduced rewritten peer graph nodes from 68 to 35, so the data do not distinguish remote-memory cost from cross-device graph dependency cost. The small positive-symbol cache did not demonstrate a speedup in these samples.

Before another GPU run, choose one bounded question from the existing evidence. A useful next step is a reliable native/virtualized trace path that actually sees CUDA activity, or a short controlled check separating remote input access from graph-context transitions. Peer-local Q8 activation/ID staging is only a source-derived candidate: one-time capture staging would be incorrect because inputs change each replay. Do not implement a broad staging/graph policy without evidence. Plain Q6 attention routing covers little remaining data and was not selected as the dominant fix.

Keep the transparent pair interface and exact numerics. Preserve the working previews, use the HPC CUDA 12.9 toolkit and supported controller admission, and keep checks finite. No GPU check or background campaign is pending. The optional Project Control inference service remains stopped; its observer remains active. Root retains final acceptance on resume; all Luna streams have returned their checkpoint work.
