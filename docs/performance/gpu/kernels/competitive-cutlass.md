# Competitive CUTLASS on small-M projections

The shape-appropriate CUTLASS configuration is **2.32–2.34× faster than the old default** in held-out measurements. It beats the frozen Mapal baseline in these four cells. cuBLAS remains faster than both. This corrects the interpretation of the earlier default-only CUTLASS comparison; it does not establish a general library or compiler ranking.

Measured 2026-10-22 on RTX 5090, driver 595.91.07, nvcc 12.8.93, cuBLAS 120804. All buffers are preallocated. Ordinary CUDA-event samples use the existing three-leg round-robin protocol, three warmups and a 256 MiB memset before each start event. Reported values are upper medians of 30 held-out samples, in milliseconds; lower is better. Compilation, tuning and profiling are outside those samples.

| M×N×K | Mapal face | Mapal baseline | Default CUTLASS | Tuned CUTLASS | cuBLAS | Mapal / tuned | Mapal / cuBLAS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16×2048×7168 | Exact | 0.2580 | 0.5140 | 0.2212 | 0.0512 | 1.166× | 5.039× |
| 16×2048×7168 | FMA | 0.2621 | 0.5140 | 0.2209 | 0.0512 | 1.187× | 5.119× |
| 16×1024×4096 | Exact | 0.1516 | 0.3011 | 0.1290 | 0.0184 | 1.175× | 8.239× |
| 16×1024×4096 | FMA | 0.1454 | 0.3011 | 0.1287 | 0.0184 | 1.130× | 7.902× |

Mapal/cuBLAS values above come from the selected-CUTLASS confirmation process. The default control runs in a separate fresh process. Its unchanged Mapal median differs by 1.56% in the larger FMA cell and is identical in the other three; unchanged cuBLAS medians are identical in all four. Those are observed controls, not confidence intervals or independent-host confirmation. Raw samples are printed to four decimal places in milliseconds.

## What was actually searched

Twelve explicit FP32 SIMT configurations were attempted against pinned CUTLASS commit `91843b71d90737e0a9b86f7d9df213cff5a3526f`. Six compiled and passed the required gates. The winner was selected from nine Contract tuning samples, with a fixed lexicographic tie break, before fresh 30-sample confirmations on both Mapal arithmetic faces. No confirmation sample chose the winner.

| CUTLASS block M×N×K | Warp M×N×K | Threads | Tuning ms, N2048/K7168 | Tuning ms, N1024/K4096 |
| --- | --- | ---: | ---: | ---: |
| 128×128×8, original default | 32×64×8 | 256 | 0.5120 | 0.3011 |
| 16×32×8 | 16×32×8 | 32 | 0.6980 | 0.4055 |
| 16×64×8 | 16×32×8 | 64 | 0.3994 | 0.2294 |
| 16×128×8 | 16×32×8 | 128 | 0.4301 | 0.2536 |
| **16×64×16, selected** | **8×32×16** | **128** | **0.2212** | **0.1287** |
| 8×32×8 | 8×16×8 | 64 | 0.3625 | 0.2109 |

The 8×8×8, 8×16×8, 16×16×8, 4×16×8, 8×16×16 and 4×16×32 configurations failed compilation at the pinned iterator's `ThreadMap::Iterations::kColumn > 0` assertion. Their diagnostics are preserved; they are not performance losses. This finite search does not establish the best possible CUTLASS configuration. More blocks alone did not identify the winner. Block shape, warp shape, panel depth and thread count vary together, so no single-factor causal claim follows.

## Numerical and measurement limits

- Mapal uses the frozen independent-output implementation, labelled `old` in captured artifacts. The cooperative/deeper alternatives are not included, and measurements from another host cannot be substituted for a same-run comparison.
- Mapal preserves the selected ordered scalar face. Libraries pass a fixed componentwise bound against both scalar references; this does not grant an ordered-bit guarantee to cuBLAS or CUTLASS. CUTLASS is FP32 SIMT; cuBLAS reports `DEFAULT_MATH` 0 and the process forces `NVIDIA_TF32_OVERRIDE=0`. Actual library instruction selection has not yet been captured.
- Bounded cases check every output at M16/N256/K7168 and M16/N128/K4096. Representative N2048/N1024 cases check 256 outputs per arm. Exported ordinary values are from the pre-timing gate, not every timed output.
- Input hashes name the frozen deterministic generator and oracle inputs. A separate full GPU A/B byte dump was not captured. This is a primitive kernel test, not a retention, allocation-latency, fused-program or production-model measurement.

Independent admission verifies 983 frozen payload files, 336 ordinary artifacts and 63 outer artifacts, including every build/PTX/oracle/argv binding. A separate root implementation, importing no harness checker, rechecks all **254,976 raw value words** with exact rational differences and the audited outward bounds. Another reader reconstructs all **1,044 raw timing events**, selection and reported medians/nearest-rank percentiles. Eight held-out processes / 24 arm rows pass.

The later Nsight Compute phase failed with `ERR_NVGPUCTRPERM` before producing a report. Its instrumentation is separate from ordinary timing. No library SASS, occupancy, traffic or stall conclusion is admitted from that failure. The [diagnostic evidence](../../../../benches/results/gpu/disassembly/library-kernels/rtx-5090-permission-denial/README.md) records the exact limit.

Next work: capture the actual library kernel images through the already-proved callback approach, compare supported cuBLASLt/CUTLASS fused epilogues on the same residual MLP, and only then choose a compiler realization to change. The [IR fact map](../disassembly/ir-facts-and-realization-gaps.md) distinguishes existing facts from unsupported lowering; this timing table alone cannot identify a missing fact.

[Raw results and independent checks](../../../../benches/results/gpu/kernels/competitive-cutlass/rtx-5090/README.md) · [Current results](../current-results.md)
