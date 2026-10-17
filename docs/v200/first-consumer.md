# First llama-server preview

Validated on 17 October 2026 against this host's qualified V100 pair and existing llama.cpp installation. This is a **T1 restricted preview** under the package's compatibility definitions: the installed executable is unchanged, but the graph-off/partial-offload configuration restricts the boundary. T2 remains the product target. This is not unrestricted CUDA compatibility or the final aggregate-memory/compute qualification.

## Start and use it

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --execute
```

The foreground session defaults to 1,800 seconds after the startup correctness check, on loopback port 8080. The launcher prints its endpoint and evidence location before entering controller admission. Startup verifies the full existing model and executable hashes before loading; allow roughly a minute on this host. Readiness is available from another terminal:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"v200-qwen3-coder-30b","messages":[{"role":"user","content":"What is 1 + 1? Answer with exactly one digit."}],"temperature":0,"max_tokens":8}'
```

Use `--duration-seconds 60 --port 8123` for a short session. Maximum serving duration is 3,000 seconds, with a separate bounded startup/cleanup allowance. Ctrl-C requests scoped cleanup; normal expiry also stops the owned server. A dry run omits `--execute`. To reproduce the matched one-shot native/V200 check:

```sh
python3 /home/tumlinson/V200/scripts/v200/serve.py --mode preview --execute
```

The launcher changes only its child's environment. Running the installed llama-server normally, without this launcher or V200 preload variables, restores native behavior. GPU admission can refuse a busy pair; the optional `project-control-inference.service` was stopped for this development work and remains stopped so it does not occupy the deliverable's devices. The Project Control workflow service remains active. No persistent server was left running by validation.

## Tested profile

- Existing `llama-server`, llama.cpp commit `f280b26983ad0fdb705a0d9ebf0503e76f2899b0`; artifact identities are in [consumer-profile.json](consumer-profile.json) and [source-profile.json](source-profile.json).
- Existing Qwen3-Coder-30B-A3B Q4_K_M GGUF; no download or copy. Exact model hash is in the fixture.
- 28 GPU layers, context 2,048, batch/ubatch 64, four CPU threads, one server slot, f16 KV, fit off, flash attention off.
- `GGML_CUDA_DISABLE_GRAPHS=1` in both native and V200 controls. Compute stays on the anchor; the pair memory allocator is available and enforces 12 GiB of application backing per owner.
- Ordered physical UUIDs `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`, `GPU-d745da9c-7649-8334-41f1-483afa6f3206`. Qualification is specific to these devices and the pinned stack.
- All new V200 builds and controller specifications use `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9`. The existing consumer is retained as installed, with its library identity recorded.

The runtime reports conservative anchor properties rather than invented aggregate SM/capacity values. Unsupported graph and VMM routes return errors. Import/export, aliases and arbitrary remote-kernel compatibility are outside the tested profile. The arithmetic smoke and one chat request establish a usable integration path, not broad model quality, sustained-load reliability or performance.

## Evidence

| Check | Result and retained record |
| --- | --- |
| Integrated host suite | Nine CTest groups passed, including 23 consumer tests and initializer fault injection; I01 final gate `9b617037-45f4-411a-b988-7bdeabac6912` |
| Foundational pair access | B01 [pair profile](../../evidence/foundation/pair-profile.json), controller `8426eab5-278d-4687-ab7d-f5ec937e46ec` |
| Owner completion/dataflow | B02 [execution retry](../../evidence/B02/execution-r2.controller-result.json), `f9e1d6eb-599e-41f0-ab7e-54ce164b9c05` |
| Logical-device ABI and VMM closure | B02 [frontdoor](../../evidence/B02/frontdoor.controller-result.json), `2c6cad1b-d526-4c1f-837b-a1a66b8d13ba` |
| Production memory across both owners | B02 [pair-memory retry](../../evidence/B02/pair-memory-r2.controller-result.json), `63a510fd-21c4-4960-84ca-572232fea914`; all 96 MiB checked, zero ledger after free |
| Matched native completion | B03 [native](../../evidence/B03/native.controller-result.json), `f2b89fd2-0a43-4dc4-82e8-56fade091835`; exact normalized answer `2` |
| V200 completion | B03 [facade](../../evidence/B03/facade.controller-result.json), `aa82e8d5-f6a4-4f62-b5d6-4e56acc53c96`; same answer, managed allocations and zero ledger on exit |
| Usable foreground server | [Service](../../evidence/V200-SERVE/first-service.controller-result.json), `547cfd3a-4eba-426a-9e1f-9b3448f7e1b8`; 20-second serving window, [independent health/chat client](../../evidence/V200-SERVE/first-service-client.json), clean shutdown |

Each successful hardware record retains controller artifacts, exact commands, source/binary identities and host release state. Rejected attempts remain alongside the retries: the initial execution smoke reported a late pinned-memory cleanup failure, and the first pair-memory fixture mistakenly checked physical enumeration through the logical facade. Neither is counted as a pass. Completion-r2 did not establish a fully quiescent post-run sampling window; its reservation release and absence of orphaned processes were verified.

Recheck the retained evidence without launching CUDA or inference:

```sh
python3 /home/tumlinson/V200/tests/integration/first_consumer.py --service
```

## Build

```sh
cmake -S /home/tumlinson/V200 -B /home/tumlinson/V200/build \
  -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
  -DV200_CUDA_TOOLKIT_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
  -DBUILD_TESTING=ON
cmake --build /home/tumlinson/V200/build --parallel 4
ctest --test-dir /home/tumlinson/V200/build --output-on-failure
```

CTest does not execute GPU work. Live checks use the supported CUDA controller and finite bundle specifications. Build and run as the project owner; the desktop sandbox may separately restrict loopback tests, GPU inspection or authority writes.

The next product milestone is useful fully resident llama payload beyond one V100's capacity, followed by measured dispatch/locality optimization. The dormant capacity fixture is a candidate, not successful evidence. Two logical pairs, broader CUDA coverage and packaging remain later work.
