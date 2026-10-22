# Benchmark evidence

[Readable performance reports](../../docs/performance/README.md) explain conclusions and limitations. These folders preserve the underlying artifacts.

| Evidence | Contents |
| --- | --- |
| [GPU](gpu/README.md) | Kernels, workloads, fusion, allocation, profiling and validation |
| [CPU](cpu/README.md) | Matrix kernels, memory and scheduling |
| [Compiler](compiler/README.md) | Compilation cost and emission equivalence |
| [Historical path map](path-map.json) | Old location → new location, with unchanged raw-file SHA256 hashes |
| [Migration verification](migration-check.json) | Relocation counts and evidence-preservation checks |

Use descriptive topic/experiment directories for new runs. Keep ordinary timing, correctness checks, profiling, and rental receipts separate. Captured files are immutable evidence: their original session labels and paths may remain inside their contents. The path map resolves those old references without rewriting the evidence.
