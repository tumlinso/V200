# Local build and validation

The repository build is CPU-first. It uses only CMake and a C++17 compiler by
default, builds the package's small CPU planning example, and registers the
host source-profile check when Python is available. It does not enable the
CUDA language, contact a model service, reserve a GPU, or run a GPU test.

```sh
cmake -S . -B build -DV200_BUILD_CUDA_PROBE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Worker source trees are added when their own `CMakeLists.txt` exists. Their
CPU checks may build and run by default; any GPU test must be disabled by
default and enabled only after the CUDA controller admits that specific run.
`V200_ENABLE_GPU_TESTS=ON` is a build switch, not an admission token.

The included Driver API pair probe can be compiled without executing it:

```sh
cmake -S . -B build-cuda-probe \
  -DV200_BUILD_CUDA_PROBE=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build-cuda-probe --target v200_pair_probe --parallel
```

Compilation is not hardware qualification. Do not invoke `v200_pair_probe`
directly; its execution belongs to the admitted B01 bundle. No watch is armed.

Use `python3 build-support/run_controller.py --help` to invoke the installed
CUDA controller in this checkout. The local shim binds the active
Project Control runtime identity and then calls the unchanged controller; it
does not bypass controller admission or evidence checks.

The installed consumer inventory is in `docs/v200/source-profile.json`.
It records immutable source and binary identifiers plus the compiler's
observed build settings. It does not select a model fixture or claim runtime
compatibility; loader resolution and the required call surface still need
consumer qualification.
