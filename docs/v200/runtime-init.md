# Lazy pair runtime initializer

`libv200_pair_init.so` exports `CUresult v200_initialize_profile(void)` for
the V200 frontdoor. The frontdoor calls it on the first pair-mode CUDA path
after it has resolved the process's already-loaded Driver API library. The
initializer performs no CUDA calls from a DSO constructor and opens no second
CUDA library handle.

The initializer validates the runtime inputs before changing CUDA state. The
first profile requires all of the following exact values:

| Variable | Required value |
| --- | --- |
| `V200_MODE` | `pair` or `2` |
| `V200_DEVICE_UUIDS` | `GPU-21131915-1488-23af-38dd-1743ae1f5cc8,GPU-d745da9c-7649-8334-41f1-483afa6f3206` |
| `V200_B01_EVIDENCE_ID` | `8426eab5-278d-4687-ab7d-f5ec937e46ec` |
| `V200_OWNER0_BUDGET_MIB` | optional, default `12288`, integer `1..12288` |
| `V200_OWNER1_BUDGET_MIB` | optional, default `12288`, integer `1..12288` |

The UUIDs name the ordered V100 pair for this deployment. Current invocation
ordinals are discovered by UUID; they are never stored as persistent device
identity. Both matched devices must report compute capability 7.0 and Driver
VMM support. Each primary context is retained, pushed only while querying that
owner's minimum VMM granularity, and popped before continuing. The original
thread context therefore remains current. The two contexts and the process
execution ledger remain alive for process lifetime after successful setup.

The memory ledger receives the ordered devices, UUIDs, granularity, per-owner
budgets, anchor owner 0, and the validated B01 evidence ID as its pair-access
qualification. Its completion provider forwards token retain, query, wait,
and release to the same `v200_execution_handle` attached to the frontdoor. The
frontdoor presents one logical device anchored on owner 0; the execution
adapter retains both physical contexts for explicit owner work and full-prefix
completion. Static memory and execution libraries are linked into
`libv200_pair_init.so` once, so the process has one owner ledger and one
completion ledger. The pair initializer does not link a second copy into the
frontdoor.

Initialization fails closed on missing or mismatched UUID/proof inputs,
invalid budgets, absent CUDA entry points, duplicate or missing devices,
unsupported architecture/VMM, primary-context failures, invalid granularity,
frontdoor attachment errors, or memory-profile configuration errors. The
result is sticky for that initializer instance. A failure after partial setup
detaches and drains what it can; if native cleanup fails, it preserves the
handle/context graph rather than releasing resources that may still be in use.

## Current boundary

The B01 qualification attached to this profile covers explicit disjoint VMM
writes, joined completion, and full-view verification by both devices. It does
not qualify arbitrary kernel transparency, NVLink route or performance,
consumer compatibility, or capacity beyond one GPU. The initializer records
that evidence ID; it does not expand the evidence.

For the first llama preview, CUDA graphs remain disabled. Graph capture/replay,
application-selected multi-device split modes, device-global behavior, and
atomic or other arbitrary cross-owner kernel access are unsupported by this
profile. The current facade path uses one anchor device and only the operation
families described by the frontdoor ABI profile. A missing or unqualified
route must return an error; pair-backed memory alone does not make an unknown
kernel legal. This setup is not yet a llama-server acceptance result.

## Host checks and opt-in hardware closure

The default build contains no GPU test execution. The injectable initializer
host test covers exact proof/budget validation, UUID order, context-stack
restoration, profile publication, completion-provider forwarding, successful
retirement, and partial-failure cleanup:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target v200_runtime_init_tests
ctest --test-dir build -R v200_runtime_init_host --output-on-failure
```

To compile (without running) the finite real-Driver closure:

```sh
cmake -S . -B build -DBUILD_TESTING=ON \
  -DV200_INTEGRATION_BUILD_GPU_SMOKE=ON
cmake --build build --target v200_pair_memory_gpu_smoke
```

`v200_pair_memory_gpu_smoke --execute` requires the outer CUDA controller to
admit the exact pair with B02, a bounded per-GPU cap, the exact ordered UUID
and B01 variables above, and both owner budgets set to `64` MiB. It requests a
96 MiB logical VMM allocation, checks nonzero backing on both owners, writes
and reads every byte from the anchor Driver context, releases the allocation,
and verifies zero live/used/pending ledger state before printing `GPU_PASS`.
The controller receipt and process ancestry are validated by the outer
runner; receipt-file presence alone does not authorize execution. The harness
is build-only unless a controller-backed run is explicitly arranged by the
project owner.
