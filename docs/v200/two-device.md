# Two-logical-device llama service

`serve_multi.py` can keep the accepted two-V200 llama workload available as a
finite foreground HTTP service. The default mode remains the B06 correctness
bundle. To serve for 30 minutes on port 8080:

```bash
python3 scripts/v200/serve_multi.py --mode serve --execute
```

Choose another finite lifetime and loopback port with:

```bash
python3 scripts/v200/serve_multi.py --mode serve \
  --duration-seconds 600 --port 8090 --execute
```

The launcher builds and runs its CPU gates before requesting admission from the
CUDA controller. During service it holds the four physical GPUs in the
qualified order and presents two logical devices to the unchanged
`llama-server`, using its ordinary `CUDA0,CUDA1` layer split of `1,1`. Startup
loads all model layers and the input embedding onto GPU memory, performs the
fixture's exact completion self-check, then prints the endpoint
(`http://127.0.0.1:<port>`). The service binds to loopback, keeps CUDA graphs
disabled, monitors the four physical owners, and stops after the requested
duration. The launcher remains in the foreground and cleans up its owned
server process and allocations on normal completion or Ctrl-C.
The run record and server log are kept under
`evidence/B06/serving/runs/<run-id>/consumer/`, separate from the accepted B06
correctness-bundle artifacts.

Use `/health` to check readiness. The service uses the pinned Qwen3-Coder
30B Q4_K_M model and the currently qualified pair budgets of 4 GiB plus 12 GiB
per logical V200. This mode is a bounded usability path; it carries no
throughput claim and does not enable graph capture.
