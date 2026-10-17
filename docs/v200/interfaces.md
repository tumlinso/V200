# I01–I04 semantic seams for parallel implementation

These are the minimum cross-module contracts for the first implementation.
They preserve the bootstrap package's meaning while leaving public C++ names,
object layout and local algorithms to the owning streams. They do not claim
that any CUDA operation is already supported.

## I01 — physical pair profile

**Input:** the admitted physical UUIDs and runtime-reported device, driver,
toolkit and topology facts for one invocation.

**Output:** a stable pair identity; the two real device/context handles;
directed access and VMM granularities; per-owner memory budgets; and the
identity of the topology evidence and stack fingerprint.

**Required behavior:** reject duplicate or wrong-architecture members and an
unqualified peer path. Derive host ordinals from the current invocation;
ordinals are not persistent identity. Discovery alone never asserts NVLink
performance or application compatibility.

## I02 — allocation ledger

**Request:** logical size, alignment, lifetime, access intent, plus optional
placement hints already justified by qualified operations.

**Result:** one canonical virtual address and exact extent map; physical
backing ownership and per-owner bytes; permissions; alias/coherence version;
reference counts; and outstanding-use tokens. All VMM-backed and directly
observed application VMM allocations participate in the same ledger.

**Required behavior:** either publish a complete allocation or roll back that
request's partial work. Release waits or defers until every recorded use is
complete. Pointer/handle queries return stable logical semantics. Unknown
aliasing or lifetime semantics fail explicitly instead of fabricating
coherence.

## I03 — logical completion

**Submission:** a logical stream operation and its dependency tokens.

**Result:** a token for the complete logical stream prefix, including all
physical owners involved, with asynchronous error state attached.

**Required behavior:** a wait or dependent operation cannot observe one
owner's completion while work from the same prefix remains on another. Keep
legacy versus per-thread default stream behavior, nonblocking streams,
context-thread stacks, graph replay and error propagation explicit. Do not
subtract timestamps from unrelated device clocks.

## I04 — operation qualification

**Key:** source, executable, module and entry ABI identity; architecture and
pair profile; operation, dtype, layout and shape family; aliasing and
numerical constraints; and the relevant stack fingerprint.

**Value:** an explicit `qualified_owner`, `qualified_split`, `supported_fallback`
or `unsupported` route, with index mapping, memory ranges, synchronization
assumptions, resource bounds and qualification receipt.

**Required behavior:** any unknown or changed key field invalidates the entry.
Names or a few matching outputs do not establish safety. A fallback names the
memory and operation class it supports; remote backing alone does not make an
arbitrary kernel legal.

## Integration rule

Each stream may choose its own API and data structures. Before wiring modules
together, expose equivalents of these inputs, outputs, failure states and
identity fields. Keep I01 identity and I04 qualification available to I02
placement and I03 ordering; do not turn the seam into a new scheduler or
central runtime service. The llama consumer stream records what this binary
actually exercises and maps those calls onto these seams; profile presence
alone does not grant a route.
