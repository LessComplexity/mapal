# Residual MLP: Mapal, CUDA, cuBLAS and PyTorch

A newer [same-device comparison with tuned fused/unfused cuBLASLt and CUTLASS](fused-library-comparison.md) is available. This report preserves the earlier five-arm run.

All ten requested rows pass independent admission under the repaired v4 packet. Both case summaries pass; 85 frozen payload hashes, 75 outer receipt hashes and 235 raw artifact hashes per shape match. Full tiny intermediates/final values, six representative checkpoint inventories and all 35 timed final observations pass the unchanged numerical gates. The reader explicitly uses packet-v4 with manifest SHA256 `201fd7d81e77cdbf866ed6411b11d84fc6452c7be32c3fe68c4bd2e623e880dc`. Attempt02 remains a separate failed attempt with 0/10 timing admissions; its data are not merged into these samples.

Actual hardware is an RTX 5090 with driver 610.43.02; native nvcc is 12.8.93, cuBLAS 120804 uses queried PEDANTIC math, and Torch 2.13.0+cu130 reports CUDA 13.0/IEEE global+matmul precision. Inductor is fullgraph/static with CUDA graphs off; the frozen source sets suppress_errors false and the actual stderr contains distinct checker and final-only AOT/Inductor generated-code records, supporting successful compilation without eager fallback. This benchmark is the frozen `3a0d435` LLVM/compiler baseline, Contract face, with no allocation-elimination candidate.

Each cell below is synchronized work-region wall time: upper median [p10,p90], in milliseconds, from thirty observations after five discarded warmups.

| Implementation | M128/d256/h512 | M128/d1024/h2048 |
|---|---:|---:|
|Mapal Contract|0.082370 [0.074550,0.095230]|0.138100 [0.136001,0.145200]|
|Naive ordered-FMA CUDA|0.053141 [0.052800,0.054581]|0.339042 [0.335702,0.348171]|
|cuBLAS pedantic|0.056230 [0.053630,0.060941]|0.102531 [0.100081,0.106440]|
|Torch eager|0.209481 [0.204521,0.230271]|0.223571 [0.215731,0.250681]|
|Inductor fullgraph, CUDA graphs off|0.310192 [0.298292,0.326291]|0.366742 [0.338292,0.390122]|

At the smaller size Mapal takes 1.550× naive CUDA's time and 1.465× cuBLAS's time. At the larger size Mapal is 2.455× faster than naive CUDA while taking 1.347× cuBLAS's time. Eager/Inductor take 2.543×/3.766× Mapal time at the smaller size and 1.619×/2.656× at the larger size. These ratios describe this fixed resident-work protocol; no universal ordering or statistical confidence interval is implied.

The synchronized-region event medians are respectively 0.076128/0.047744/0.050688/0.183712/0.285408 ms for the smaller shape and 0.131392/0.333728/0.097024/0.197344/0.332320 ms for the larger shape, in the table's implementation order. Their rankings agree with the wall medians. These events include host submission/completion gaps and are not pure kernel durations. The full event/wall percentile table and all raw admitted values are in `attempt05-independent-reader/REPORT.md` and `analysis.json`.

The first eager checker/final-only calls take 210.970/0.539 ms and 173.626/1.011 ms for the small/large shapes. The corresponding Inductor calls take 2769.094/671.517 ms and 2975.443/738.778 ms. These cold measurements include initialization or compilation and remain separate from the steady table; they are not full application startup benchmarks.

Torch's before/after allocated bytes are unchanged within each run. Its allocated baseline is the declared resident input bytes plus exactly 33,554,432 B at both shapes (inputs 1,444,864 B and 21,508,096 B). The residual allocation has no identified owner in these snapshots and is not attributed to a library workspace. Eager peak-allocated minus baseline is 1,179,648 B small and 4,718,592 B large; compiled is 524,288 B and 2,097,152 B. Reserved peaks differ substantially and include process caching behavior. These snapshots are not driver peaks or total library workspace measurements. Mapal's source arena requests remain 1,838,080 B/22,032,384 B versus explicit CUDA buffer requests 2,493,440 B/25,702,400 B; no allocation-elimination latency or measured memory-saving attribution follows from those different scopes.

The numerical comparison is deliberate: Mapal/naive require ordered true-fmaf bits and both scalar-reference budgets; cuBLAS/Torch reductions must pass both predeclared componentwise budgets. Full tiny tensors are checked. Representative stages publish 256 selected cells each; the large references cover complete rows 0,42,84,127 and their selected checkpoint indices, not every large output element. Values are observed outside clocks, and checker/final-only graphs are separate.

The next evidence-backed work is to test supported fused library epilogues against the same MLP graph and budget, plus a graph-enabled Torch arm, rather than infer a fusion gain from kernel count or add primitive speedups together. Mapal loses to this cuBLAS arm at both sizes despite beating the restricted Torch arms; the reason requires executed kernel/resource evidence. To test allocation or retention causally, equalize the input-retention/reset protocol and isolate storage policy while preserving device work. Independent inspection of both emitted final-only LLVM programs locates six generation calls before start at line 1111, three work launch sites between start and end at line 1201, and no generation inside. Current Mapal therefore regenerates immutable inputs before each clock while the external arms retain them, and Torch allocates intermediates inside work. Independent branches, representative recurrence, retained-intermediate/full-recompute comparisons and a wider shape inventory remain separate missing generic evidence.

No new measurements are invented here. This report uses the retrieved raw receipts under `/private/tmp/mapal-s61-gpu/attempt-05/retrieved/trace/composed-results` and the unmodified v4 independent reader. Metadata/cold/memory extracts are retained in `attempt05-metadata.json`.

Evidence: [complete experiment directory](../../../../benches/results/gpu/workloads/residual-mlp/comparison).
