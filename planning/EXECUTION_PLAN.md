# V200 execution plan

Updated 17 October 2026. This is a concise planning overlay; Todo Orchestrator
remains task authority. The source package in [v200-bootstrap/](v200-bootstrap/README.md)
is preserved and advisory.

## Goal and first delivery

The product goal remains one application-visible CUDA-compatible device per
NVLink V100 pair. The first workload is the existing `llama-server`, launched
as one logical device without application-selected physical multi-GPU
splitting. The intended compatibility target is T2: the installed executable
remains unchanged behind the V200 launcher.

Deliver the smallest honest, useful preview first: one reproducible request,
correct output, clear support limits, diagnostics, and a reversible disable
path. This first delivery is a T1 restricted preview; T2 remains the target.
Native and V200 completion checks now pass, as does an independent
health/chat client against the finite foreground server. The first preview's
graph-off and partial-offload restrictions are explicit in the
[usage notes](../docs/v200/first-consumer.md).

## Current implementation

- V200-01 through V200-06, Q01 and I01 are complete. The first-delivery run is closed.
- Source, consumer and native-control identities are pinned in
  [source profile](../docs/v200/source-profile.json) and
  [consumer profile](../docs/v200/consumer-profile.json). The selected HPC SDK
  CUDA 12.9 root is `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9`.
- The host suite is green: 9 CTest groups and 23 consumer tests passed.
- B01 pair qualification, B02 completion/front-door and the 96 MiB two-owner
  memory check have retained controller evidence. These bounded checks do not
  establish application capacity or general CUDA transparency.
- A native llama-server control returns the exact expected answer `2` with 28
  GPU layers, 2048 context, graphs disabled and partial CPU offload. It is a
  fitting correctness control, not a V200 facade or performance result.
- B03 V200 returns the same exact answer, tracks real managed allocations,
  and retires to an empty ledger. The foreground launcher passed a 20-second
  service window and an independent health/chat request, then released the pair.

See [bootstrap and evidence status](BOOTSTRAP_STATUS.md) for receipts and exact
fixture details.

## Next work

1. Use the restricted preview in projects and keep its tested profile and limits
   visible as it evolves. V200-06's host and retained-evidence gates passed.
2. Prove useful llama-server payload above the available memory of either one
   V100, with both physical owners' bytes recorded and no CPU offload or hidden
   host oversubscription. The 96 MiB allocator check does not satisfy this.
3. Separately demonstrate useful work on both GPUs against a matched native
   control before claiming dual-compute acceleration. Then optimize the
   measured bottleneck and expand compatibility from observed use.

Capacity beyond one V100, useful dual-GPU compute, and broad CUDA compatibility
remain unproven. No performance or delivery-date claim is made yet.

The user authorized evicting llama-server and Project Control GPU workloads
when V200 needs the devices. Use the supported controller and service lifecycle,
verify release, and leave unrelated GPU workloads untouched.
