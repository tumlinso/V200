# Core interface contracts — design, not frozen implementation APIs

The root freezes only interfaces necessary for parallel work. These contracts are deliberately semantic; C++ naming and layouts can change. They are not imported as falsely frozen Todo interfaces in this release.

## I01 Physical identity and pair profile

Input: admitted physical UUIDs and runtime facts. Output: stable pair identity, real device/context handles, directed capabilities, VMM granularities, per-owner budgets, topology evidence identity and stack fingerprint. Reject duplicate members, wrong architecture or unqualified peer path. Host ordinals are derived per invocation. Discovery is not performance qualification.

## I02 Allocation ledger

Input: logical byte/alignment/lifetime request plus optional known operation-placement hints. Output: canonical VA, exact extent map, backing/refcounts, owner bytes, permissions, coherence/version state and pending-use tokens. Release waits or defers until owned uses complete. Partial creation rollback is transactional for its own allocations. Handle queries return the chosen logical semantics consistently. Direct application VMM participates in the same ledger.

## I03 Logical completion

Input: logical stream and operation dependencies. Output: a token representing the complete logical prefix, including both owners. A later dependent operation cannot see only one half. Error state accompanies the token. Legacy/PTDS, nonblocking streams, graph replay and context thread stacks remain explicit. No cross-GPU elapsed-timestamp subtraction.

## I04 Operation qualification

Key: source/binary/module hash, exact entry ABI, architecture, operation/dtype/layout/shape family, numerical and aliasing constraints, stack/pair profile. Value: owner/split/semantic/unsupported route, index mapping, memory ranges, synchronization assumptions, resource bounds and qualification receipt. Unknown changes invalidate the entry. String names and sampled outputs alone do not confer safety.

## I05 Cost and placement policy

Input: qualified operation, extents, useful-work dimensions and dated cost observations. Output: chosen owners, legal moves/replicas, split and completion plan, plus a reason code. Compare full cost including staging/canonicalization. A policy is replaceable; correctness contracts are not. No online model/scheduler service is required initially.

## I06 Evidence

Every result has status, task/bundle/decision association, source/binary/fixture hashes, stack, physical/logical identity, exact timed boundary, correctness oracle, samples or null, contamination/limits and cleanup status. Unknown is not zero. GPU evidence cannot be synthesized from CPU mocks or this design report.
