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

The accepted B06 correctness bundle is [recorded here](../../evidence/B06/multipair-runs/multipair-20261017T150959Z-75b105fcd3/consumer/bundle.json), controller evidence `bdf88682-63b9-4788-8943-b411c03ee1f1`. Model backing peaked at 19,518,193,664 bytes across all four owners; both pair ledgers ended at zero. The finite service subsequently passed a 10-second deployment smoke, exact completion and normal exit, evidence `d34961d6-ac1d-4998-81e8-24195639c9f6`, with [release confirmed](../../evidence/B06/serving-smoke-r1.host-release.json).

Within each logical device, qualified kernels may access peer-backed HBM through the pair. Between logical devices, arbitrary peer pointers are not advertised. Copies currently block and stage through pinned host memory in chunks of up to 8 MiB, with a 1 GiB per-call bound. The six qualified small-copy cases cover both directions with source-stream event/wait, destination-stream and default-stream ordering. They are correctness evidence, not a transfer-performance claim. Physical UUIDs and each pair's B01 permission evidence are pinned in [the profile](multipair-profile.json); PCIe/NUMA paths between pairs remain distinct from within-pair NVLink.
