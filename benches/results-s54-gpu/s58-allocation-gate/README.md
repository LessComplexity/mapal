# Allocation gate v2 — reviewed, not GPU-executed

The corrected packet passes independent real-process cancellation and output-path
checks. All 432 archive members match the reviewed manifest. The kernel, runtime,
fixtures, numerical oracles and counter predictions are unchanged from the first packet.

- [Exact prepared commands, limits and scope](prepared-v2-README.md)
- [Independent review and reproductions](review/REVIEW.md)
- [Durable recovery receipt](recovery.json)
- [Rejected v1](rejected-v1/README.md): preserved before any upload; detached children
  survived cancellation and nested output paths corrupted the input inventory.

Planned checks are 640 ordinary processes and 28 separate traced counter processes.
Local stubs and process tests are not GPU correctness, allocation counts or performance.
The compiler candidate remains isolated. A hardware run requires fresh budget/host/deadline
binding; no consumed launcher may be reused.
