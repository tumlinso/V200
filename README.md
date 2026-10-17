# V200

V200's first restricted preview runs the existing llama-server through a process-scoped CUDA facade presenting the qualified NVLink V100 pair as one logical device.

Start the foreground server on this host:

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --execute
```

It serves `http://127.0.0.1:8080` for 30 minutes after its startup check. Use another terminal for clients; `GET /health` reports readiness. Omit `--execute` to inspect the plan. `--duration-seconds` and `--port` configure the finite session; Ctrl-C requests cleanup. The launcher uses GPU-controller admission and the HPC SDK CUDA **12.9** toolkit.

The default foreground service remains a restricted correctness preview: **28 GPU layers, a 2,048-token context, CUDA graphs disabled, and one compute anchor**. Native and V200 completion checks returned the expected answer; a client also passed the live server's health and chat endpoints.

V200-07 now has a bounded full-resident capacity workload. It loads the pinned Qwen3-Coder-30B Q4_K_M model with all 49 llama layers and its input embedding on one logical `CUDA0`. The 18.55 GB model tensor payload was backed across both 16 GiB V100s; the deterministic completion returned `2`, and allocation accounting returned to zero. To run the matched native-pair and V200 capacity cases through the CUDA controller:

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --mode preview --workload resident_capacity --execute
```

This establishes pair-backed application memory capacity. It does **not** demonstrate dual-GPU computation or speedup; the preview remains graph-off and computes on the anchor. See [the capacity evidence and current limits](docs/v200/capacity.md), [consumer usage](docs/v200/first-consumer.md), [planning](planning/README.md), and the [execution plan](planning/EXECUTION_PLAN.md). The original bootstrap remains intact under `planning/v200-bootstrap/`; Todo authority tracks progress. No model download or llama.cpp source patch is required.

Git remote: <https://github.com/tumlinso/V200.git>.
