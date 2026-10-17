# V200 bootstrap — validation report

**Release:** 1.0 · **Checked:** 17 October 2026 · **Result:** CPU checks passed; canonical Todo plan compiled deterministically. This is validation of the delivered bootstrap, not of an operational V200.

## Executed checks

| Check | Actual result | What it establishes |
|---|---|---|
| Python unit suite | **14 passed; 0 failed; 0 skipped** | Bundle binding, inert rendering, receipt checks, bounded CPU child cleanup/timeout, ELF inspection, package structure and compiler-adapter refusal behavior |
| C++17 build | **Passed** with `-Wall -Wextra -Wpedantic -Werror` | The CPU reference example builds in the recorded authoring environment |
| CTest | **1/1 passed** | CPU planning reference test executes successfully |
| CPU reference assertions | **1,461 passed** | Checked arithmetic, owner extents, boundary slicing and logical grid-stride coverage on the included inputs |
| Evidence schema checks | **5 passed** | Schema validity and claim-boundary examples; synthetic records are not hardware evidence |
| Package consistency | **Passed** | JSON, task identity, acyclic dependencies, derived edges, runbook references, relative links and dormant bundle state |
| Canonical preledger compilation | **14 tasks; 24 dependencies; 0 warnings; 0 excluded tasks** | Actual inspected Project Control compiler produced native Todo schema-v2 JSON |
| Deterministic recompilation | **Identical output** | Same authoring inputs and inspected compiler produce the same plan bytes |

The canonical compiler file was read from the connected `project-control` source and transferred into an isolated temporary authoring directory. Its **complete byte hash matched** the inspected source before execution. The temporary compiler is not shipped or installed. No replacement Todo compiler or scheduler is included.

Compiler SHA-256:

```text
f47180104ca5b8b219d057a86339fc2c0d018e2e8e0eae12b88c94e6a4f2f339
```

Native plan SHA-256:

```text
8007770c2905311f3e19a5a406133acc677656e4ba8f35cb002eeb401eb32b89
```

The plan imports no invented frozen interfaces, run tree, GPU reservations or fake completed checkpoints. The implementation outcomes remain unexecuted.

## Validation environment

Python `3.13.5`; `g++ (Debian 14.2.0-19) 14.2.0`; `cmake version 3.31.6`. CPU build and tests ran in the authoring container, not on the user's GPU host. No CUDA toolkit or `nvidia-smi` was available in that environment.

Logs and machine-readable results are included in this directory: [results](results.json), [canonical compilation](canonical-compilation.json), [Python tests](python-tests.log), [CMake build](cmake-build.log), [CTest](ctest.log), [CPU checks](cpp-checks.log), [schema checks](schema-checks.json), and [package consistency](package-structure.log).

## Deliberately not claimed

**The CUDA pair-VMM example was not compiled or run.** It includes an embedded PTX module and needs a qualified CUDA 12.x environment plus the real admitted GPU pair. Its actual headers/Driver API compatibility, mapping behavior and completion semantics remain B01 hardware qualification.

No existing consumer was run through a V200 facade, no application compatibility or speedup was measured, and no GPU experiment was executed or reserved. All seven GPU bundles remain `NOT_RUN`, with automatic execution disabled. Existing atlas measurements are dated prior evidence, not new V200 results.

**Live Todo authority validation and application were not performed.** The canonical compile is complete; deployment still needs a real registered/initialized V200 authority, the installed kernel's validation/diff, and an explicitly authorized apply. See [import instructions](../docs/IMPORT.md). The provided consistency checker does not impersonate the native Todo validator.

No connected repository, active project state, claims, GPU clocks, driver configuration, host NUMA policy or background watch was modified.

## Release integrity

`MANIFEST.sha256` covers the delivered files except itself. Final release verification checks those hashes, recompiles with the inspected canonical compiler, compares the plan bytes, and tests the ZIP archive after this report is written. Integrity demonstrates an intact package, not hardware correctness. The downloadable release-check receipt records that final pass separately to avoid a self-referential manifest.
