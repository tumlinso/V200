# V200

V200's first restricted preview runs the existing llama-server through a process-scoped CUDA facade presenting the qualified NVLink V100 pair as one logical device.

Start the foreground server on this host:

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --execute
```

It serves `http://127.0.0.1:8080` for 30 minutes after its startup check. Use another terminal for clients; `GET /health` reports readiness. Omit `--execute` to inspect the plan. `--duration-seconds` and `--port` configure the finite session; Ctrl-C requests cleanup. The launcher uses GPU-controller admission and the HPC SDK CUDA **12.9** toolkit.

The existing Qwen3-Coder-30B Q4_K_M model, executable and libraries are pinned. This preview uses **28 GPU layers, a 2,048-token context, partial CPU offload, CUDA graphs disabled, and one compute anchor**. Native and V200 completion checks both returned the expected answer. An independent client also passed the live server's health and OpenAI-compatible chat endpoints; shutdown released the pair.

Both-owner memory backing passed a separate 96 MiB full-range check. **Application capacity beyond one V100, dual-GPU compute and speedup remain unqualified.** No model download, consumer-source patch or global CUDA injection is needed.

See [usage, evidence and limits](docs/v200/first-consumer.md), [planning](planning/README.md), and the [execution plan](planning/EXECUTION_PLAN.md). The original bootstrap remains intact under `planning/v200-bootstrap/`; Todo authority tracks progress.

Git remote: <https://github.com/tumlinso/V200.git>.
