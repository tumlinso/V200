# V200 deployment preview

This checkout has a host-side diagnosis and a repeatable CUDA 12.9 build entry.
Neither installs anything system-wide, initializes a GPU, downloads the model,
nor leaves a resident service. Run these from the checkout:

```bash
python3 tools/v200/diagnose.py
./packaging/build.sh
```

`diagnose.py --json` includes the source revision and dirty state, profile and
fixture hashes, the two logical-to-physical UUID mappings, CMake cache/toolchain
checks, V200 DSO hashes, and the pinned llama-server binary/runtime-library
hashes. It checks the model path and recorded size without rehashing the 18.6 GB
file. Its `ldd`/hash observations describe the local files; they are not a new
CUDA execution or performance qualification. B06 evidence records the pinned
full-resident two-device correctness result and a finite 10-second HTTP service
smoke. It does not establish four-GPU throughput.

The build script always configures `build-multipair/`; it refuses a symlinked
directory or a CMake cache belonging to another checkout. It does not modify
the single-pair `build/` tree. It builds with the HPC CUDA 12.9 toolkit and
runs CTest with GPU tests disabled. The generated probe executable is only
compiled; it is not run by this script.

## Run the preview

For two process-local logical devices, start the bounded foreground service:

```bash
python3 scripts/v200/serve_multi.py --mode serve --execute
```

It binds to loopback port 8080 by default, performs its pinned startup
correctness request, reports the endpoint, and exits after 30 minutes. Set a
shorter `--duration-seconds` and a different loopback `--port` if needed. Use
Ctrl-C for early shutdown; the launcher cleans up its owned server process.
The validated profile maps CUDA0 to physical UUIDs
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8` and
`GPU-d745da9c-7649-8334-41f1-483afa6f3206`, and CUDA1 to
`GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505` and
`GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`.

The accepted finite profile is pinned to llama.cpp commit
`f280b26983ad0fdb705a0d9ebf0503e76f2899b0` and the local Qwen3-Coder 30B
Q4_K_M fixture. The service uses 4 GiB plus 12 GiB per logical pair for this
workload, keeps CUDA graphs disabled, and uses blocking host staging for
cross-pair copies. It is a correctness/usability preview, not a general CUDA
virtualization layer or a performance claim. The dedicated single-pair
extended expert path is separately opt-in:

```bash
python3 scripts/v200/serve.py --mode serve --workload resident_capacity \
  --pair-compute --extended-expert-compute --execute
```

That command requires a qualified B01 pair receipt and launches one V200. Its
extended expert dispatch is limited to the observed Q4_K/Q6_K paths; ordinary
unrecognized kernels stay on the native anchor route. Add `--cuda-graphs` to
this single-pair command to opt into the separately qualified graph path. The
two-device service above keeps graphs disabled.

The B07 comparison measured median decode at 64.095 tokens/s for the
single-pair extended V200 path and 124.951 tokens/s for the matched native
two-GPU control. These are three-request samples for one prompt and model;
V200 was slower in this case and is not currently a speed replacement. See
`docs/v200/performance.md` and `evidence/B07/extended-r1/` for sample details.

## Native fallback and removal

The accepted native fallback evidence is B06's `native_four` control. It runs
inside the existing CUDA-controller admission in
`python3 scripts/v200/serve_multi.py --mode correctness --execute`; that
bounded comparison runs the native control and then the V200 correctness
control. This checkout has no native-only HTTP service wrapper. For another
native workload, use its normal command through the host CUDA controller. Do
not launch a hand-copied GPU command outside admission, and do not set V200
preloads or environment variables. The matched native control's child inputs
were:

```json
{
  "CUDA_VISIBLE_DEVICES": "GPU-21131915-1488-23af-38dd-1743ae1f5cc8,GPU-d745da9c-7649-8334-41f1-483afa6f3206,GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505,GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a",
  "executable": "/home/tumlinson/Software/llama.cpp/build/bin/llama-server",
  "fixture_arguments": [
    "--model", "/home/tumlinson/.local/share/core4/models/qwen3-coder-30b-a3b-instruct-q4-k-m/fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad/model.gguf",
    "--host", "127.0.0.1", "--ctx-size", "8192", "--batch-size", "64",
    "--ubatch-size", "64", "--parallel", "1", "--threads", "4",
    "--threads-batch", "4", "--n-gpu-layers", "999", "--fit", "off",
    "--flash-attn", "off", "--cache-type-k", "f16", "--cache-type-v", "f16",
    "--alias", "v200-qwen3-coder-30b", "--device", "CUDA0,CUDA1,CUDA2,CUDA3",
    "--split-mode", "layer", "--tensor-split", "1,1,1,1",
    "--override-tensor", "^token_embd\\.weight$=CUDA0", "--verbosity", "5"
  ]
}
```

The controller-backed V200 services stop on normal duration expiry or Ctrl-C;
no launcher daemon or driver configuration is installed. To remove only this
checkout's generated build outputs, stop any V200 service and run from the
repository root:

```bash
rm -rf -- build-multipair
```

This leaves source, evidence, the native llama.cpp checkout, other CUDA
software, and `build/` untouched. The repository itself is not removed.
