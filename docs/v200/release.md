# V200 usable preview checkpoint — 17 October 2026

Latest checkpoint: the opt-in `--prefer-local-hbm` candidate now measures 78.053 versus 63.146 tokens/s with the same binaries and original placement, against 123.357 native. It preserves full model residency and exact output. Use `./packaging/build.sh --candidate`, then add `--prefer-local-hbm` to the single-pair command below. The frozen single-/two-device preview remains available. See [B08 performance](performance.md) and the [requested pause checkpoint](pause-checkpoint.md).

The first usable restricted llama-server deliverable is available. **The full V200 product is not accepted as complete.** Capacity, selected peer execution and two logical devices now work; native-competitive performance and broad transparent CUDA execution remain open. V200-14 stays open for that final acceptance. The architecture remains pair virtualization; the faster native layer split is a fallback, not a replacement V200 backend.

## Working application capabilities

| Capability | Retained evidence |
| --- | --- |
| Unchanged llama-server selects one logical device and holds an 18,550,716,425-byte model tensor payload, exceeding either 16 GiB V100; all 49 layers plus embedding on GPU, no CPU offload | B04 capacity-r1; repeated by B03 extended-r2 and B07 |
| Actual Q4_K matvec/fused experts and small-K Q4_K/Q6_K expert down projections execute on the peer owning their weights | B03 extended-r2, `429d012d-d9d9-44a5-b6bd-88c105011a66`; 16,384 outputs bitwise verified for each observed new family |
| Single-pair graph capture/replay with up to 68 peer-context nodes | Same B03 run; replayed again in B07 |
| One process enumerates two logical CUDA devices backed by four physical V100s, loads the full model with ordinary logical-device layer splitting, produces the correct completion, and returns both ledgers to zero | B06, `bdf88682-63b9-4788-8943-b411c03ee1f1`; 19,518,193,664 bytes peak backing, all four owners populated |
| Finite two-device loopback HTTP service with exact startup self-check and normal cleanup | 10-second deployment smoke, `d34961d6-ac1d-4998-81e8-24195639c9f6` |

The model, binary, native-library closure, exact UUID pairs and toolkit are pinned in the consumer/multipair profiles. V200 uses the HPC SDK CUDA 12.9 toolkit. Source checkpoints are `7a04b35` (extended expert operations), `56dad35` (two-device functionality/finite service), and `16f756b` (deployment package and performance records). Evidence retains source differences for runs performed before the integration commit; later CPU package builds reproduced the accepted core DSO hashes.

## Use it

For a single logical V200 with the qualified extended expert and graph paths:

```bash
python3 scripts/v200/serve.py --mode serve --workload resident_capacity \
  --pair-compute --extended-expert-compute --cuda-graphs --duration-seconds 300 --execute
```

For two logical V200s (graphs off):

```bash
python3 scripts/v200/serve_multi.py --mode serve --duration-seconds 300 --execute
```

Both stay in the foreground, use controller admission, and bind to `http://127.0.0.1:8080`. Omit `--execute` for the plan; select `--port` if occupied. No server was left running for this handoff. [Deployment](deployment.md) covers the CPU build/diagnostic commands, native fallback and removal. The original `build/` baseline remains intact.

## Performance and supported boundary

B07 (`d5c28e95-2772-474c-86c8-318bc533fc9a`) compared one logical V200 with the native two-V100 layer split: **64.095 versus 124.951 median decode tokens/s**, with identical generated outputs. Both used full residency and graphs. One warmup plus three 64-token requests is a descriptive sample; client wall times include first-measured-request outliers. Added Q4/Q6 coverage has not established a material speedup over the earlier Q4-only run. [Raw results and limitations](performance.md) are retained. No four-GPU throughput claim is made.

The runtime is process-scoped; it does not unify NVML/PCI devices or distribute arbitrary CUDA kernels. Qualified operation dispatch is tied to exact native-kernel families and source/library identities. Unrecognized kernels stay on the anchor. Single-pair application backing is capped at 12 GiB per owner and live headroom/reserve; the two-device fixture uses 4+12 GiB per pair to exercise all owners. These are supported envelopes, not an unconditional 32 GiB promise. Cross-pair copies block and use pinned-host staging (8 MiB chunks, 1 GiB/call limit); arbitrary cross-pair peer pointers are not advertised. Two-device graphs, general VMM import/export/aliases and broad native-library/consumer compatibility remain unqualified.

## Checks and continuation

`./packaging/build.sh` passed all 13 CPU CTests; the 20 multipair tests, 11 comparison tests and 4 deployment checks passed. The independent finite-service lifecycle review found no blocker. `python3 tests/release/verify_preview.py` checks the retained capacity/compute/graph/two-device/service evidence, current core DSO identity and canonical release records without initializing CUDA. These checks reuse hardware evidence; they are not another hardware qualification.

Every admitted run completed and its canonical owner reports `released` with no resources. The authorized Project Control inference service was stopped through its supported lifecycle; the observer service was preserved. Final GPU-process inspection was empty. No background watch, model service or new resource manager was installed.

Continue within the existing outcome structure: isolate the concrete decode/prefill bottleneck with a short attributable check, implement the dominant operation/locality improvement, and compare against the same native baseline. Expand prefill/kernel coverage and two-device graph/copy compatibility when real consumer paths need them. Keep exact numerics, the transparent per-pair interface, the working preview and native fallback. Final product acceptance must not be inferred from completion of the earlier implementation outcomes or from this preview checkpoint.
