# RTX 5090 competitive CUTLASS comparison

2026-10-22. Eight held-out processes / 24 arm rows admitted. Six of twelve configurations compiled; all six passed the required numerical gates. The selected 16×64×16 configuration wins over the old default but remains behind cuBLAS. [Readable comparison and limits](../../../../../../docs/performance/gpu/kernels/competitive-cutlass.md).

| Artifact | Purpose |
| --- | --- |
| [Ordinary raw archive](raw/ordinary.tar.gz) | All 337 original ordinary files, including stdout/stderr, process/build receipts, binaries, gates, tuning and confirmation |
| [Runner summary](summary.json) | Original sealed ordinary result; all failures and rejected builds retained |
| [Independent admission](checks/independent-admission.json) | Payload/artifact hashes, raw numerical words, request identity and selection gates |
| [Independent event review](checks/raw-event-review.json) | Raw medians, nearest-rank percentiles, winner selection and comparison ratios |
| [Independent value review](checks/raw-value-review.json) | Separate Fraction/IEEE-word parser; no harness checker imported |
| [Protocol reviewer](checks/audit_protocol.py), [event reviewer](checks/review_events.py), [value reviewer](checks/review_values.py) | Reproducers with captured local input locations; use recovery paths when restoring inputs |
| [Recovery receipt](recovery.json) | Public archive SHA256 and private source/full-trace recovery locations |
| [Cleanup](checks/cleanup.json) | Exact task-owned instance independently confirmed absent |

The archive was re-opened and every member compared byte-for-byte by SHA256 with the original file. `summary.json` seals all 336 other ordinary artifacts. Captured identifiers and paths are immutable provenance; new reports use descriptive names. Profiling data is stored under the separate [permission-denial experiment](../../../disassembly/library-kernels/rtx-5090-permission-denial/README.md).
