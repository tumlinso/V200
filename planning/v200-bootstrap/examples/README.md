# Starter code: what it proves and what it does not

## CPU reference

From this package directory:

```sh
cmake -S examples -B /tmp/v200-examples -DV200_BUILD_CUDA_PROBE=OFF
cmake --build /tmp/v200-examples --parallel 2
ctest --test-dir /tmp/v200-examples --output-on-failure
```

`planning.hpp` implements checked granularity arithmetic, a small baseline two-owner extent plan, interval slicing and an explicit logical-grid stride reference. Balancing bytes is not an optimized production placement strategy. Tests establish these CPU functions on the tested inputs, not GPU memory semantics or arbitrary kernel split safety.

## Driver API pair qualification example

After selecting the actual qualified CUDA12.x toolkit, build without reserving a GPU:

```sh
cmake -S examples -B /path/to/V200/build/bootstrap-examples \
  -DV200_BUILD_CUDA_PROBE=ON -DCUDAToolkit_ROOT=/path/to/cuda-12.x
cmake --build /path/to/V200/build/bootstrap-examples --target v200_pair_probe --parallel 2
```

No `nvcc` compilation is required for this particular host C++ program: it links the Driver API and loads a small embedded PTX module at runtime. It still needs authentic CUDA headers, a suitable driver and actual GPUs for qualification. The CUDA target was **not compiled or executed during package assembly**.

Run only through B01 in `bundles/`, with the actual UUIDs bound and admitted. The sample refuses absent lease environment and wrong visible UUID order. The bounded wrapper validates the controller receipt/ancestry; an environment variable by itself is not permission. Default payload is 64 KiB per owner, rounded to queried VMM granularity, capped at 64 MiB mapped per owner. It creates no long-lived process and changes no clocks, driver state or NUMA policy.

The example maps two physical allocations into one canonical range, grants both devices access, writes disjoint explicit-index ranges, joins with cross-device events and verifies the complete range from **both GPUs' SMs**, using owner-local error counters. Its event graph is a small concrete completion example. It drains owned streams before destruction. Parent deadline and post-run controller checks remain necessary if the driver stops making progress.

This does not prove a physical NVLink route, cache coherence, application binary transparency, arbitrary PTX rewriting, a production allocator, a 24 GiB workload, or any speedup. Default range alignment may differ across drivers. `GPU_PASS` output would describe only this probe and only after it actually runs.

## Production adaptation

The example logs cleanup errors and refuses a success result if cleanup fails. The outer runner/controller must still classify timeout or process-release uncertainty. Extend allocation lifetimes, VMM handle queries and asynchronous errors under the design contracts. Do not copy the toy `fill` kernel as an alleged generic rescheduler. Example inputs and stage labels must not become runtime hardware constants.
