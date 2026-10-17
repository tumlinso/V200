# Owner-local computation

The qualified path keeps weights in their existing physical HBM and moves computation to them. Activations and outputs remain in canonical pair-accessible allocations; the peer reads and writes those small spans directly over NVLink. There is no model copy, CPU routing-ID readback, or replicated writable alias.

For expert-indexed Q4_K decode, the unchanged kernel can run on the peer when the entire expert bank is peer-owned. Gate-fused operations also require the gate bank on that same owner. Banks crossing an ownership boundary, unknown layouts, and unsupported families retain the original anchor launch. Actual extent-order metadata determines placement, including allocations made with the peer as the preferred owner.

The initial expert policy requires at least 4 MiB of active weight bytes plus the activation/output span. This is a conservative estimate informed by the existing V100 atlas E19 event protocol and E20 NVLink measurements, not a measured crossover for this bridge. Small expert operations stay on the anchor. The earlier non-expert row path retains its existing policy. No adaptive policy subsystem is introduced.

The real full-resident llama run in `evidence/B03/locality-r1` passed: 71 fused expert calls executed on the peer and 26 on the anchor. The first routed expert operation matched the original kernel bit-for-bit across 6,144 outputs; all model tensors remained GPU-resident and cleanup returned the owner ledger to zero. Controller evidence: `7d72d5a0-cb5b-4b2b-b64b-17fd45ba5f06`. The plain expert family and graph rewrite remain unqualified by that run.

For the later B05 comparison, hold the pinned model, context, prompt, graph policy and complete request work constant across native pair layer splitting, V200 anchor execution, and V200 owner routing. Compare repeated request latency and useful tokens per second, with startup and first-operation verification reported separately. The short correctness request here establishes no inference speedup. Staging or replication should be added only if that comparison identifies a concrete reuse benefit.
