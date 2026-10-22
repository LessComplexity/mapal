# Current GPU results and next experiments

Updated 2026-10-22. Correctness, steady timing, allocation counts and profiling are separate evidence classes. No generic win over cuBLAS has been established.

| Question | Verified result | Detailed report |
| --- | --- | --- |
| Does Mapal beat cuBLAS on a composed residual MLP? | No: at M128/d256/h512, Mapal 0.082370 ms vs cuBLAS 0.056230 ms; at M128/d1024/h2048, 0.138100 vs 0.102531 ms. Mapal beats the restricted PyTorch eager and graph-disabled Inductor arms. | [Residual MLP comparison](workloads/residual-mlp.md) |
| Can known dataflow eliminate allocations? | Yes: unused raw-output allocations drop from 7→6, 8→7, 12→10 or 8→6 in the tested cases; controls stay unchanged. 640 ordinary processes and 28 separate counter processes pass. Latency gain remains unmeasured. | [Unused output allocations](memory/dead-output-allocation.md) |
| Does the 8×64/K16 tile alternative help? | Against the original, FMA improves 1.253× at M16/N2048/K7168 and 1.159× at M16/N1024/K4096; Exact takes 1.227× and 1.013× as long, respectively. Tile aspect and panel depth change together. cuBLAS is faster in all four cells; no selector promotion. | [Tile geometry and panel comparison](kernels/panel-depth-comparison.md) |
| Is the current CUTLASS comparison competitive? | Its default 128×128 tile generates only 8/16 CTAs for the M16 cases. Twelve shape-appropriate candidates have passed preparation review; GPU compilation and held-out timing await a working rental. | [CUTLASS configuration audit](kernels/cutlass-baseline-audit.md) |
| Do we have disassembly from the actual launch? | Four Mapal Contract images pass exact PTX/context/module/function/launch joins. Library-kernel capture is the next diagnostic. | [Executed Mapal kernels](disassembly/executed-mapal-images.md) |
| What about 16K and model-shaped matrices? | Earlier large-shape measurements and all failures remain available; they are distinct from the M128 composed test. | [Large matrices](workloads/large-matrices-and-mlp-validation.md) |

Next: finish the competitive CUTLASS confirmation, capture the selected cuBLAS/CUTLASS/Mapal kernels with Nsight Compute, and test supported library epilogue fusion on the same composed graph. Only then attribute a gap to missing dataflow facts, unused facts or a backend implementation choice.

The allocation correction is integrated; the kernel alternatives remain isolated. Current test status, active experiments and rental ownership are recorded in [the live handoff](../../next-session.md).

[Browse GPU reports](README.md) · [Browse raw GPU evidence](../../../benches/results/gpu/README.md)
