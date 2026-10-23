# Eliminating unused GPU output allocations


RTX 5090, machine146747/host552627, driver590.48.01, CUDA toolkit12.8.93. The gate forces
streams; the shipped RTX 5090 profile disables streams. Before/after each complete 320
ordinary fresh-process checks (eight cases × Exact/Contract ×20), followed by28 separate
traced counter processes. Root checks packet/result hashes, exact observed stdout,
empty ordinary stderr, process exits, binaries/PTX and raw counter records after retrieval.
All640 ordinary and28 counter processes pass.

| Dataflow case | Allocation requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| Independent chains | 7 → 6 | 114,688 → 98,304 |
| Diamond | 8 → 7 | 131,072 → 114,688 |
| Two MLP chains | 12 → 10 | 196,608 → 163,840 |
| Omitted raw outputs | 8 → 6 | 131,072 → 98,304 |
| Single-chain arena control | 1 → 1 | 81,920 → 81,920 |
| Materialized-raw control | 8 → 8 | 131,072 → 131,072 |
| Forked-tile control | 8 → 8 | 131,072 → 131,072 |

Exact and Contract agree on these counts; successful frees match actual allocations.
Requested bytes are neither physical driver peaks nor a latency measurement. Counter
tracing synchronizes launches and cannot establish ordinary timing or stream overlap.
The ordinary checks observe the fixture outputs, not every possible output or schedule.

`resident.rs:FnEmit::chain_prologue` now consumes the existing selected `Epi.store_out`
fact. It omits only raw storage that the grouped kernel does not write. No source hint,
benchmark name, runtime cache, arena-guard removal or new dependency representation is
introduced. Twenty-eight device LLVM units remain unchanged; preserved S58 static proof
checks surviving host-body/slot/order equivalence. Existing physical-writer completion
edges and zero-slot free behavior support the separate GPU validation.

The reviewed patch is `2c8cdc1964abbd0516a46c5da1e02a76ba59e214e205275c8e478dc23e85ffd5`.
The allocation packet remains `a93247bef4e50b96ed148dbad6c66a77caf1f10462ab6af8b2210c7a4de545cd`.
After integration, required-clang workspace testing passes1,187 tests with three native
macOS hardware-detection failures and one ignored. A complete backend-library rerun with
host access passes59/59, including those three:1,190 distinct tests pass across the runs.
No full unrestricted-workspace rerun is claimed.


Latency, before/after on four parts: [allocation latency](allocation-latency.md).

[Raw gate and integration evidence](../../../../benches/results/gpu/memory/allocation-elimination/validation/) · [All memory results](README.md)
