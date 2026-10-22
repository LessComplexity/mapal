# Matrix performance history

[Performance index](../README.md) · [Current GPU results](../gpu/current-results.md) · [CPU results](../cpu/README.md)

These are historical experiments. Read each report’s machine, arithmetic and measurement limits before comparing numbers. The [original measurement chronology](history/measurement-chronology.md) preserves the detailed sequence and corrections.

| Experiment | Focus |
| --- | --- |
| [Literal-input baseline](history/literal-input-baseline.md) | Early compiler and GPU baseline |
| [Procedural-input baseline](history/procedural-input-baseline.md) | Input generation and larger matrix coverage |
| [Naive kernel comparison](history/naive-kernel-comparison.md) | CUDA and other language baselines |
| [Parallel execution and FMA](history/parallel-execution-and-fma.md) | CPU concurrency and contracted GPU arithmetic |
| [Bit-exact SIMD tiles](history/bit-exact-simd-tiles.md) | Ordered per-cell reduction with independent SIMD lanes |
| [Register blocking](history/register-blocking.md) | Multiple live output accumulators and SIMD width |
| [FMA, packing and loop lifting](history/fma-packing-and-loop-lifting.md) | Input layout and recognized reduction structure |
| [Reduction splitting and shape kernels](history/reduction-split-and-shape-kernels.md) | FIR and convolution alongside GEMM |
| [K-panel measurements](history/k-panel-measurements.md) | Panel nesting costs and vector accumulator follow-up |
| [Large-K panel nesting](history/large-k-panel-nesting.md) | Profile-derived panels, row reuse and scheduling |
| [Timing boundary and OpenBLAS](history/timing-boundary-and-openblas-comparison.md) | Page residency, timing corrections and CPU library comparison |
| [SME kernel](../cpu/matmul/sme-kernel.md) | Matrix instruction lowering on Apple hardware |
| [SME throughput ceiling](../cpu/matmul/sme-throughput-ceiling.md) | Operand residency, unit throughput and panel sweeps |
