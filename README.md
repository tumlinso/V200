# V200

V200 is an experimental, application-scoped CUDA facade for NVLink-connected V100 pairs. The unchanged pinned llama-server can now load an **18.55 GB GPU tensor payload through one logical device**, execute selected Q4_K/Q6_K expert operations on the peer V100, and capture/replay qualified CUDA graphs. Four physical V100s also enumerate as **two logical devices in one process**, with real model residency on all four owners and correct inference.

Run the full-resident single-V200 server on this host:

```sh
python3 scripts/v200/serve.py --mode serve --workload resident_capacity --pair-compute --extended-expert-compute --cuda-graphs --duration-seconds 300 --execute
```

Run the two-V200 server (CUDA graphs off for this qualified profile):

```sh
python3 scripts/v200/serve_multi.py --mode serve --duration-seconds 300 --execute
```

Both bind to `http://127.0.0.1:8080`; use `/health` to check readiness. They stay in the foreground, use CUDA-controller admission, and release their owned processes and resources when the finite session ends. Omit `--execute` to inspect a plan; `--port` changes the port. Builds use the HPC SDK CUDA **12.9** toolkit. No model download, llama.cpp source patch, system driver change or background daemon is required.

The first implementation is deliberately restricted. Peer compute matches exact kernel families in the pinned llama CUDA backend; it is not arbitrary-kernel distribution or universal CUDA compatibility. Inter-pair copies currently use blocking pinned-host staging. A pair uses qualified HBM budgets rather than advertising all nominal memory as available. The original restricted preview and native CUDA remain reversible fallbacks.

The latest matched comparison measured **64.1 tokens/s for extended V200 pair compute versus 125.0 for native two-GPU layer splitting**. The earlier Q4-only comparison measured 63.5 versus 54.2 for V200 anchor compute. This is real capacity and peer execution, with substantial performance work remaining. See [performance](docs/v200/performance.md), [two-device behavior](docs/v200/two-device.md), [capacity](docs/v200/capacity.md), and [consumer details](docs/v200/first-consumer.md).

The original restricted preview remains available with `python3 scripts/v200/serve.py --execute` (28 GPU layers, 2,048-token context, graphs off). [Deployment and diagnostics](docs/v200/deployment.md) describe the supported profiles and rollback.

Git remote: <https://github.com/tumlinso/V200.git>.
