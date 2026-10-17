# V200-07: resident application capacity

V200-07 demonstrates an unchanged llama-server binary loading a model tensor payload that is larger than either V100 in the selected NVLink pair. The application sees one CUDA device, `CUDA0`; the V200 memory allocator backs its model allocations across both physical devices.

## Run the bounded workload

The launcher runs a matched native two-GPU control followed by the one-device V200 facade case. It uses the CUDA controller, checks current per-device headroom, and stops after the finite completion request and cleanup:

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --mode preview --workload resident_capacity --execute
```

Remove `--execute` to inspect the planned run without entering the GPU controller. The launcher selects the recorded V100 pair and HPC SDK CUDA 12.9 toolchain. It uses the existing pinned llama-server and model; it does not download model data or patch llama.cpp.

The workload uses Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `--ctx-size 8192`, `--n-gpu-layers 999`, `--fit off`, and `--split-mode none`. Since llama.cpp normally keeps the input embedding on CPU, this workload uses its standard `--override-tensor '^token_embd\.weight$=CUDA0'` option. The verbose startup log records that override and the model buffers; CUDA graphs remain disabled. Both controls use the same consumer command and exact completion oracle (`2`).

## B04 result

The bounded B04 run passed with controller evidence `8b4250ac-792d-4c75-8359-292150ef800e`.

The model's GGUF tensor payload is **18,550,716,425 bytes** (17.28 GiB). Each selected Tesla V100-SXM2 has 16,384 MiB (17,179,869,184 bytes), so that payload cannot fit on either GPU individually. The llama log records 49/49 layers on CUDA and a `CUDA0` model buffer of 17,691.35 MiB. It also confirms `token_embd.weight` was assigned to CUDA0.

For the V200 facade case, the owner ledger reached a peak of **6,551,502,848 bytes** on the anchor and **12,884,901,888 bytes** on the peer, for **19,436,404,736 bytes** total. Both owners stayed within their 12 GiB application budgets. Process-scoped NVML telemetry independently observed the server using both selected UUIDs. The completion response matched `2`; server shutdown was clean; the final ledger had zero used bytes, live allocations, reservations, and cleanup-pending entries. The host resource release record shows the controller reservation returned with no resources held.

The matched native-pair control also returned the expected output and showed the same full model residency. This is a correctness and memory-capacity result, not a performance comparison.

Evidence is retained under [the B04 run](../../evidence/B04/capacity-r1/), including the native and facade case JSON/logs, owner-ledger JSONL, execution record, [controller evidence](../../evidence/B04/capacity-r1.controller-evidence.json), [controller result](../../evidence/B04/capacity-r1.controller-result.json), and [host release](../../evidence/B04/capacity-r1.host-release.json). Run `python3 tests/capacity/verify_capacity.py` to re-check the retained artifacts without initializing CUDA or re-hashing the model file.

## Limits

This result establishes aggregate model residency through one logical CUDA device. It does not establish useful computation on the second GPU, improved throughput or latency, graph compatibility, an adaptive placement policy, or the nominal 32 GB capacity under arbitrary workloads. The current preview keeps CUDA computation on the anchor and disables CUDA graphs. Each owner's startup budget is capped at 12 GiB and at the physical free memory observed during initialization minus a 1 GiB reserve; `cudaMemGetInfo` reports those usable pair budgets. V200-08 is the next step for useful pair-native compute.
