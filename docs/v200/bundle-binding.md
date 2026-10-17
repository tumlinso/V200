# GPU qualification binding

The user authorizes implementation and foreground GPU qualification for the first existing llama-server profile, including eviction of running llama servers and Project Control GPU services when needed. No persistent background watch is armed. Unrelated GPU users are preserved.

Build before admission. Use the canonical Project Control Python runtime to invoke the configured CUDA controller. The installed controller still imports a removed historical identity module; `build-support/run_controller.py` delegates that one import to the current fail-closed `project_control.runtime_identity.bind_runtime` and runs the unchanged controller:

```sh
/home/tumlinson/project-control/scripts/pc-dev python \
  /home/tumlinson/V200/build-support/run_controller.py \
  /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py \
  run --spec /absolute/path/to/spec.json --json
```

The launcher verifies the configured Skills root and runtime identity. It grants no resources itself. Controller reservations, foreground interference exclusion, idle proof, and release remain authoritative; environment visibility alone is insufficient. Controller `--help` passed through this launcher without GPU work.

For B01/B02, render the preserved package bundle definition using actual selected UUIDs and an authentic CUDA 12.x toolkit root. `planning/v200-bootstrap/tools/render_bundle.py` creates a spec but does not execute it. `tools/run_bounded.py` verifies an active `CUDA-FOREGROUND-LEASE/1` receipt, exact project/UUIDs, a live controller ancestor, and visible device identity before starting its finite child. The original `binding.example.json` is inert.

The first pair-memory fixture is `v200_pair_probe`: explicit expected UUIDs, a small bounded allocation per owner, joined event completion, verification from both GPUs, and owned cleanup. A successful run qualifies only those tested mechanisms. It does not prove the llama ABI boundary or a capacity/throughput gain.

Hardware identity, direct NVLink topology, peer eligibility, VMM granularity and selected physical ordinals are pending runtime qualification. Recorded atlas pair ordinals are guidance, not placement authority. Do not hold a lease during builds or discussion. Store the exact spec, source/binary/fixture hashes, stdout/stderr, controller evidence identifier and cleanup result for each live test. Keep CPU_PASS, COMPILED_ONLY, GPU_PASS, NOT_RUN and unresolved failures distinct.

Llama application sessions use an explicitly pinned existing model and finite request harness, with matched native and facade controls. No silent model/precision/offload changes, unconditional server restarts, or claims of pair acceleration from enumeration alone.
