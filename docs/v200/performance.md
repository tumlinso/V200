# Performance records

## Anchor-local placement (B08, 17 October 2026)

The new opt-in allocation policy fixes a confirmed locality bottleneck. The old spill planner selected the **smallest** legal preferred-owner extent, filling peer HBM first. The pinned model therefore put about 12 GiB on the peer and 5.28 GiB on the anchor while unrecognized operations continued to execute on the anchor. The candidate selects the largest legal preferred extent, leaving 1 GiB of soft anchor headroom for subsequent KV/workspace allocations. If that preference cannot satisfy a large allocation, it uses the full original pair budgets; aggregate capacity is preserved.

| Case | Median decode tokens/s | Median server prefill |
| --- | ---: | ---: |
| Native two-V100 layer split | 123.357 | 58.621 ms |
| Candidate dispatch cache, original placement | 63.146 | See raw ablation |
| Same candidate binaries, locality enabled | **78.053** | 129.012 ms |

The placement change increased decode throughput by **23.6%** against the same-binary ablation. It still reaches only 63.3% of the matched native rate. The cache-only result does not establish a host-cache speedup over B07. These are descriptive three-request medians for the existing prompt/model, one preceding warmup and 64 generated tokens per request. Native/locality cases ran together for 44.464 seconds; the separate cache-only ablation used the same DSOs and flags except placement. All warmup and measured outputs matched the native reference exactly.

The locality case retained the full 18,550,716,425-byte GPU tensor payload. Peak owner backing was [12,696,158,208,6,740,246,528] bytes, with zero final accounting, normal process exit and successful graph capture/replay. Peer graph rewrites fell from 68 in B07 to 35; qualified Q4/Q6 peer computation remained present. This associates the gain with placement, without distinguishing remote-memory cost from the effect of fewer cross-device graph dependencies. The remaining performance gap is not explained conclusively.

Evidence: [matched native/locality run](../../evidence/B08/locality-r1/comparison/comparison.json), [cache-only ablation](../../evidence/B08/cache-only-r1/comparison/comparison.json), and [qualification/identity summary](../../evidence/B08/locality-qualification.json). Controller IDs are `827f2d85-d588-4403-84ff-56d185933cdb` and `d2b1df8f-1b21-4a5d-862f-2ca225abb94f`; both canonical owners are released. The 15 CPU candidate CTests and 16 comparison checks passed. The immutable `build-multipair` preview remains available; the candidate uses `build-perf`.

An earlier node-level Nsight diagnostic (`da67c8cc-2ac9-49e4-b4c0-d050948188b9`) hit the finite service limit under instrumentation and required forced child cleanup. Its final application accounting did not close before process termination; the canonical host reservation was released and GPU memory was reclaimed. It is retained as a failed diagnostic, not a successful qualification or throughput result. No additional profiling run was started before the user's requested pause.

## Extended expert path (B07, 17 October 2026)

The newer Q4_K small-K and Q6_K expert paths execute useful peer work correctly, but do not close the native-pair performance gap. This remains an opt-in capacity/compute preview, not a native-speed replacement.

| Matched case | Median decode tokens/s | Median server prefill | Median client wall |
| --- | ---: | ---: | ---: |
| Native two-V100 layer split | 124.951 | 58.522 ms | 564.902 ms |
| One logical V200, extended Q4/Q6 peer experts | 64.095 | 131.696 ms | 1,117.452 ms |

Both cases used the same unchanged pinned model, full GPU residency, graphs enabled, fixed prompt and 64-token request, one warmup and three measured requests. All warmup/measured outputs matched exactly. The V200 profile observed 90 low-column Q4 peer calls and 86 Q6 peer calls; graph capture/replay passed. These are ordinary-dispatch call counts, not a count of every graph replay. Kernel verification was disabled during timing; prior B03 extended-r2 separately established bitwise equality for 16,384 outputs per observed new family. The V200 ledger returned to zero and both servers exited normally.

The raw server decode samples were 124.935, 125.143, 124.951 tokens/s for native and 63.889, 64.095, 64.220 for V200. Client wall samples were 767.390, 564.902, 564.829 ms and 2,629.852, 1,117.452, 1,117.324 ms respectively. The first measured client request remains an outlier despite warmup. Three samples for one prompt are descriptive; they do not establish sustained throughput, latency distributions, or a causal bottleneck. The older B05 Q4-only case is a separate run, so the small difference between its 63.503 tokens/s and this 64.095 is not evidence of an extended-path speedup.

Source was committed and clean at `56dad356323eda4f0c29ee94411f9c49c92c4d4f`; current `build-multipair` DSOs, native closure, fixture, helper and harness hashes were captured and rechecked. The controller admitted only the first NVLink pair for 44.213 seconds of comparison, with 15 seconds of service per case and a 90-second comparison ceiling. Evidence: `d5c28e95-2772-474c-86c8-318bc533fc9a`; [raw comparison](../../evidence/B07/extended-r1/comparison/comparison.json), [source/artifact preflight](../../evidence/B07/extended-r1.preflight.json), and [canonical release](../../evidence/B07/extended-r1.host-release.json). Resources were released. This is a single-pair comparison; B06 establishes two-device correctness separately and makes no four-GPU performance claim.

The next optimization should target the measured decode/prefill gap while retaining the transparent pair interface. Native layer splitting stays available as the faster fallback; it does not replace the V200 architecture. No placement threshold or speculative mechanism is promoted on the basis of these samples.

## Earlier qualified Q4 comparison (B05)

The bounded B05 comparison passed on 17 October 2026. It confirms that the current V200 preview can run the unchanged llama-server with a model payload spread across both V100s and can route useful Q4_K matrix-vector work to the peer GPU. It does **not** show native-pair throughput parity or establish a broad performance result. Keep pair-compute opt-in while implementation and measurement continue.

## What was compared

All cases used the same pinned llama-server, model, fixture, 32-token prompt, greedy 64-token decode request, and one warmup followed by three measured requests. CUDA graphs were enabled in all cases. Each case first passed the exact arithmetic startup oracle (`2`); the three measured generation responses were byte-for-byte identical across all cases. The workload disabled prompt caching and used seed 42, temperature 0, and top-k 1.

| Case | Device/execution mode | Median server decode | Measured server decode (tokens/s) | Median server prefill |
| --- | --- | ---: | ---: | ---: |
| Native pair | Two physical V100s; native layer split | 124.021 tokens/s (507.978 ms) | 124.021, 124.345, 124.008 | 58.901 ms (543.284 tokens/s) |
| V200 anchor | One logical V200; pair capacity, anchor compute | 54.213 tokens/s (1,162.079 ms) | 53.830, 54.213, 54.355 | 153.661 ms (208.251 tokens/s) |
| V200 pair compute | One logical V200; pair capacity and qualified peer Q4 execution | 63.503 tokens/s (992.080 ms) | 63.756, 63.436, 63.503 | 130.587 ms (245.047 tokens/s) |

The reported decode and prefill figures come from llama-server's per-request timing fields. Each row has only three measured requests for one prompt and one model, so these are descriptive results for this run, not stable application-wide throughput estimates. The V200 pair-compute median was above the V200 anchor median in this run, while both were below the native two-GPU layer-split median. Do not interpret these three-sample ratios as a general speedup or product-level performance boundary.

Client wall time is separately retained because it varied much more than the server-reported decode time:

| Case | Raw measured client wall times (ms), in request order | Median (ms) |
| --- | --- | ---: |
| Native pair | 772.954, 569.829, 569.893 | 569.893 |
| V200 anchor | 3,217.660, 1,311.935, 1,316.039 | 1,316.039 |
| V200 pair compute | 2,757.094, 1,126.822, 1,124.776 | 1,126.822 |

The first *measured* wall request was an outlier in every case despite one preceding warmup. Server decode timings remained comparatively tight within each case (population standard deviations 0.637 ms, 4.776 ms, and 2.142 ms respectively). The records do not identify the source of the extra client wall time; it must not be attributed to a particular CUDA, launch, or host cause without targeted evidence. The outliers also make these three-request wall medians unsuitable for latency claims.

## Demonstrated functionality and limits

The unchanged one-device V200 consumer loaded all 49 transformer layers and the input embedding to CUDA with an 18,550,716,425-byte model tensor payload (about 18.55 GB / 17.28 GiB), larger than either individual 16 GiB V100. The V200 owner ledger peaked at 6,551,502,848 bytes on the anchor and 12,884,901,888 bytes on the peer, 19,436,404,736 bytes total; each owner remained within its 12 GiB allocation budget. Both V200 cases ended with zero live allocation/reservation and cleanup-pending counts, and the host controller reservation was released with no resources held.

In the pair-compute case, the operator profile recorded 435 peer-owner Q4_K matvec calls and 176 peer-owner fused expert-bank calls, covering 839,680 ordinary peer rows and 1,081,344 expert peer rows. The graph path captured and replayed successfully; up to 34 matched graph nodes were rewritten in the peer-compute case. The response remained exactly equal to the native-pair output. This demonstrates real peer execution in the selected Q4 path; it does not establish that every model operation executes on both GPUs, that the peer path is faster than the native layer split, or that all quantization families are covered. In particular, Q6 expert-bank support remains outside this qualified path.

Pair capacity and useful peer computation are functional in this restricted preview. The preview remains opt-in and has compatibility boundaries documented in [first-consumer.md](first-consumer.md). The B05 records do not justify presenting it as a native-speed replacement. Continue implementation and tune the measured paths before making a broader performance claim.

## Reproduction and provenance

The controller admitted one finite B05 bundle with a 260-second comparison limit and 65 seconds per server case; total recorded comparison elapsed time was 216.563 seconds. No model download was needed. The exact workload, all per-request raw timing fields and outputs, artifact hashes, and case evidence are retained in [the B05 comparison result](../../evidence/B05/pair-r1/comparison/comparison.json). The controller run passed as evidence `24630d18-8d52-4a53-912b-79c86c088ba5`; its canonical host release record reports `released` with an empty resource list.

The comparison used the qualified V200 source commit `2bd3020146cd4378a6bb60a614b361d030a3bc34`, the frozen consumer harness commit `df12371c23801291a08c631527e36f22dd21d66c`, and llama.cpp commit `f280b26983ad0fdb705a0d9ebf0503e76f2899b0`. The existing Qwen3-Coder-30B-A3B-Instruct Q4_K_M model hash is `fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad`. All 15 pinned binary and fixture artifacts passed hash checks against [the qualified manifest](../../benchmarks/comparison/qualified-binaries.json). The graph path's prior bounded hardware qualification is evidence `797dc352-ff64-40cf-ae18-3dcfc29874f2`. The two physical V100 UUIDs, driver/toolchain identities, per-case logs, memory telemetry, and exact artifact hashes are recorded in the linked evidence directory.

This result is tied to the pinned pair, consumer, model, binaries, and workload above. It is not evidence for the second V200 pair, arbitrary CUDA applications, sustained/concurrent serving, latency distributions beyond these samples, or the full product target.
