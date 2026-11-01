# Performance results

Start with [current GPU results](gpu/current-results.md). Reports are organized by the question they answer; raw evidence lives under [benchmark results](../../benches/results/README.md).

| Topic | Read here | What it covers |
| --- | --- | --- |
| GPU kernel comparisons | [Kernels](gpu/kernels/README.md) | Mapal, cuBLAS and CUTLASS; small-M alternatives; arithmetic guarantees |
| Composed and large workloads | [Workloads](gpu/workloads/README.md) | Residual MLP, PyTorch, 16K matrices and model-shaped projections |
| GPU memory and lifetime | [Memory](gpu/memory/README.md) | Allocation elimination, arenas, retention and ordering |
| GPU disassembly | [Disassembly](gpu/disassembly/README.md) | PTX, actual executed SASS, resource use and unresolved causes |
| GPU validation | [Validation](gpu/validation/README.md) | Architecture selection, independent confirmations and limitations |
| CPU performance | [CPU](cpu/README.md) | Matrix kernels, packing, residency and scheduling |
| Compiler cost | [Compiler](compiler/README.md) | Compilation time and guard-gating equivalence |
| Early GEMM measurements | [Matmul history](matmul/README.md) | Earlier CPU/GPU comparisons, preserved with descriptive filenames |
| Measurement conventions | [Methodology](methodology/README.md) | Algorithm families, precision and measurement boundaries |

Use descriptive topic and experiment names for new reports and evidence. Session numbers belong only in historical provenance and immutable handoff logs, never in new result titles or storage folders. Separate ordinary timings, correctness, profiling and operational receipts. Every comparison must state the machine, shape, precision, measured region and admission limits.

The [path migration map](../../benches/results/path-map.json) resolves historical filenames to their current locations. Raw artifact contents and hashes were preserved; historical session logs and captured paths inside receipts remain unchanged.
