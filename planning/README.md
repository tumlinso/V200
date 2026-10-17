# V200 planning

Updated 17 October 2026. Start with the [execution plan](EXECUTION_PLAN.md) and
[current status](BOOTSTRAP_STATUS.md).

The original bootstrap remains extracted under [v200-bootstrap/](v200-bootstrap/README.md)
without edits. Its architecture and proposed outcomes are suggestions; Todo
authority records actual task state, and implementation is governed by the
current charter and evidence.

The archive contains 66 files and 343,567 expanded bytes. Archive SHA-256:
`8683ad0fca152a7f833f0215ccae1f7564da65256db957b37c275b75bfb1b0a8`.

The first restricted llama-server preview passed native/facade completion,
live health/chat access, and clean foreground shutdown. See the
[usage and evidence](../docs/v200/first-consumer.md). V200-01 through V200-05,
Q01 and I01 are complete; run 1 / V200-06 is closing acceptance. Application
capacity beyond one V100 and useful dual-GPU compute remain unqualified.
