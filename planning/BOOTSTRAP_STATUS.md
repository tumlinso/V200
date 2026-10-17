# V200 bootstrap and implementation status

Updated 17 October 2026. Bootstrap setup is complete; implementation is now
underway. This status summarizes the preserved package, current Todo state and
evidence. It does not replace Todo authority.

## Preserved package and project identity

- The archive contains 66 files and 343,567 expanded bytes. Its SHA-256 is
  `8683ad0fca152a7f833f0215ccae1f7564da65256db957b37c275b75bfb1b0a8`; all 65
  manifest-listed file hashes passed at adoption. Package files remain
  unchanged under [v200-bootstrap/](v200-bootstrap/README.md).
- Repository: `/home/tumlinson/V200`, branch `main`; runtime foundation commit
  `8b6b9a8`. Origin is `https://github.com/tumlinso/V200.git`; no push is
  recorded.
- Project Control workspace/repository: `v200` / `v200`. Todo project UUID:
  `6c6463a7-0f66-4e12-813f-c17e46185775`. The original 14-outcome plan was
  applied through the supported workflow at revision 2; its SHA-256 is
  `8007770c2905311f3e19a5a406133acc677656e4ba8f35cb002eeb401eb32b89`.

## Current work and profiles

V200-01 through V200-05, Q01 and I01 are complete. Run 1 / V200-06 is active.
The existing llama-server source, binary, native fixture and profiles are
pinned in [source profile](../docs/v200/source-profile.json) and
[consumer profile](../docs/v200/consumer-profile.json). The selected toolkit
root is HPC SDK CUDA 12.9 at
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9`.

Host validation is green: all 9 CTest groups and 23 consumer tests passed;
initialization was reviewed. These checks do not establish GPU or application
facade behavior.

## Retained GPU and consumer evidence

- **B01:** controller receipt `8426eab5-278d-4687-ab7d-f5ec937e46ec` succeeded.
  See [controller result](../evidence/foundation/B01.controller-result.json).
- **B02 completion retry:** receipt `f9e1d6eb-599e-41f0-ab7e-54ce164b9c05`
  succeeded. The 4 KiB case checked owner copies, peer fan-in completion,
  dependent data flow and context restoration; it explicitly does not prove
  arbitrary-kernel transparency. See [result](../evidence/B02/execution-r2.controller-result.json)
  and [case output](../evidence/B02/execution-r2/stdout.txt).
- **B02 front door:** receipt `2c6cad1b-d526-4c1f-837b-a1a66b8d13ba`
  succeeded. The bounded closure covered malloc/free and VMM map/unmap paths.
  See [result](../evidence/B02/frontdoor.controller-result.json) and
  [profile trace](../evidence/B02/frontdoor-profile.jsonl).
- **B02 pair memory:** receipt `63a510fd-21c4-4960-84ca-572232fea914`
  succeeded. A 96 MiB range used 32 MiB and 64 MiB on the two owners, passed
  full-range anchor write/read, and left the ledger empty after free. See
  [result](../evidence/B02/pair-memory-r2.controller-result.json) and
  [case output](../evidence/B02/pair-memory-r2/stdout.txt). Rejected attempts
  remain preserved: completion cleanup failed in the initial execution case,
  and the first memory fixture confused logical and physical enumeration.
- **B03 native control:** receipt `f2b89fd2-0a43-4dc4-82e8-56fade091835`
  succeeded. The pinned llama-server answered `2` exactly for the Qwen3-Coder
  30B Q4_K_M smoke fixture, with 28 GPU layers, 2048 context, graphs disabled
  and partial CPU offload. Peak observed GPU use was 10,562 MiB on one physical
  GPU. This establishes the native fitting control only. See
  [controller result](../evidence/B03/native.controller-result.json) and
  [case record](../evidence/B03/native/consumer/smoke-native_single-1792239954218618376.json).

- **B03 facade:** receipt `aa82e8d5-f6a4-4f62-b5d6-4e56acc53c96` succeeded.
  It returned the matched answer `2`, tracked its managed allocations, and
  retired to zero ledger usage. See [result](../evidence/B03/facade.controller-result.json).
- **Foreground service:** receipt `547cfd3a-4eba-426a-9e1f-9b3448f7e1b8`
  succeeded for a 20-second service window. An independent client passed
  health and OpenAI-compatible chat checks; shutdown released the pair.
  See [result](../evidence/V200-SERVE/first-service.controller-result.json)
  and [client](../evidence/V200-SERVE/first-service-client.json).

The restricted first preview is usable; [quickstart and limits](../docs/v200/first-consumer.md)
are available. V200-06 is closing its artifact/host acceptance gates. Application
capacity beyond one V100, useful dual-GPU compute and speedup remain unqualified.
