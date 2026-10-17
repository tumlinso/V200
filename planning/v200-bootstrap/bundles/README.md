# Bounded GPU bundles

These are **seven dormant, task-owned execution contracts**. B01 includes source code; B02–B06 depend on the corresponding implementation fixtures; B07 is deliberately unmaterialized. A defined bundle is not a completed test. All are initially NOT_RUN and none is armed automatically.

| Bundle | Purpose | Devices | Wall ceiling | Application allocation cap per GPU |
|---|---|---:|---:|---:|
| [B01](B01.md) | Small pair VMM and completion qualification | 2 | 90 s | 512 MiB |
| [B02](B02.md) | Observed consumer ABI and device boundary closure | 2 | 120 s | 1024 MiB |
| [B03](B03.md) | First consumer and selected regression/operation check | 2 | 300 s | 12288 MiB |
| [B04](B04.md) | Resident capacity exceeding a single V100 | 2 | 300 s | 14336 MiB |
| [B05](B05.md) | Matched pair performance decision | 2 | 600 s | 14336 MiB |
| [B06](B06.md) | Two logical devices on all four physical GPUs | 4 | 300 s | 14336 MiB |
| [B07](B07.md) | One justified advanced-mechanism discriminator | 2 | 120 s | 1024 MiB |

The caps are proposed engineering limits, not promises of test duration or hardware isolation guarantees. CUDA context/driver overhead is recorded separately; the allocation cap must be enforced by each fixture before allocating. The OS does not impose a hard VRAM quota through this renderer. A larger requirement opens D05, not an automatic increase.

## Running a ready bundle later

First build/configure on CPU. Create a local binding JSON using `binding.example.json`, replace placeholders with the actual authorized paths/UUIDs, and select the exact command for the implemented fixture. The binder records source paths, a toolkit and binaries. Rendering does **not** execute anything:

```sh
python planning/v200-bootstrap/tools/render_bundle.py \
  --catalog planning/v200-bootstrap/bundles/catalog.json \
  --bundle B01 --binding /path/to/local-binding.json \
  --output /path/to/B01.controller.json
```

Review the generated spec. A later authorized execution uses the existing controller:

```sh
python /path/to/skills/cuda/scripts/cuda_controller.py run \
  --spec /path/to/B01.controller.json --json
```

The renderer uses documented/inspected foreground keys: `resources.gpu_uuids`, `gpus`, `cpu_threads`, `ram_bytes`, NVLink isolation; `benchmark.build_argv`; `build_timeout`; `timeout`; `toolchain`; `binary_paths`; `paths`; `command_cwd`; and `argv`. It does not invent a GPU reservation API. Explicit UUIDs avoid the controller's fallback to the first physical ordinals. The child independently verifies the lease/visible set before running its command.

The controlled child timeout is shorter than the outer capture timeout, allowing owned-process termination and evidence recording before the controller releases its lease. A hung/unreaped process or unexpected fault stops further work; an environment variable is not proof of cleanup. Do not reset or kill foreign work.

## Early versus deferred

Run B01/B02 once the foundation binaries are ready, because postponing these feasibility risks until the end would waste implementation. B03 reaches a real workload quickly. B04–B06 are later consolidated checks. During a wait, continue independent CPU work and hold no GPU lease. Do not attach all bundles as automatic completion reruns on every Todo task.

For optional unattended operation, use the existing CUDA controller's explicit background arm/enqueue contract after authorization, with a fixed ready revision, selected bundle and explicit correctness/repetition bounds. Do not simply copy its default long correctness loop. This package intentionally contains **no automatic arming script**. Background state and host reservation authority remain in the existing system.

## Reusing evidence

A result is reusable only when its relevant source/binary/stack/pair/fixture identity and semantic contract match. Documentation-only edits do not require a new GPU run. Changes to the frontend ABI, allocation lifecycle, stream graph or target kernel invalidate the corresponding narrow qualification. Do not relabel old evidence as a new run.
