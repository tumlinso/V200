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

V200-08 also executes eligible Q4_K matrix-vector operations on the physical owner of their weights: the full-resident consumer completed with 68 anchor-local and 174 peer-owner calls. The first peer operation matched the original kernel bit-for-bit across 4,096 outputs; managed allocation accounting returned to zero. Graphs remain disabled; single-matrix splitting is CPU-checked only, and speedup is not established. See [operator evidence](docs/v200/operator-profile.json), [capacity](docs/v200/capacity.md), and [consumer usage](docs/v200/first-consumer.md). No model download or llama.cpp source patch is required.

Run a finite full-resident server with pair compute (omit `--pair-compute` for the capacity-only fallback):

```sh
python3 scripts/v200/serve.py --mode serve --workload resident_capacity --pair-compute --duration-seconds 300 --execute
```


Git remote: <https://github.com/tumlinso/V200.git>.
