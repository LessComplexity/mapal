# Current GPU results and next experiments

Updated 2026-10-22. Correctness, steady timing, allocation counts and profiling are separate evidence classes. No generic win over cuBLAS has been established.

| Question | Verified result | Detailed report |
| --- | --- | --- |
| Does Mapal beat competitive fused libraries on a residual MLP? | No: at M128/d256/h512 and M128/d1024/h2048, Mapal takes0.073090/0.128935 ms versus tuned fused cuBLASLt0.032659/0.059620 ms (2.24×/2.16× faster). Tuned fused CUTLASS has lower observed medians too; Mapal beats tested eager/graph-disabled Inductor. All30 same-device confirmation/control rows admitted. | [Fused library comparison](workloads/fused-library-comparison.md) |
| Can known dataflow eliminate allocations? | Yes: unused raw-output allocations drop from 7→6, 8→7, 12→10 or 8→6 in the tested cases; controls stay unchanged. 640 ordinary processes and 28 separate counter processes pass. Latency gain remains unmeasured. | [Unused output allocations](memory/dead-output-allocation.md) |
| Does the 8×64/K16 tile alternative help? | Against the original, FMA improves 1.253× at M16/N2048/K7168 and 1.159× at M16/N1024/K4096; Exact takes 1.227× and 1.013× as long, respectively. Tile aspect and panel depth change together. cuBLAS is faster in all four cells; no selector promotion. | [Tile geometry and panel comparison](kernels/panel-depth-comparison.md) |
| What changes when CUTLASS is tuned? | Six of twelve configurations compiled and passed gates. Selected 16×64×16 is 2.32–2.34× faster than the old default and beats the frozen Mapal baseline in all four cells; cuBLAS remains 4.31–7.01× faster than tuned CUTLASS. | [Competitive CUTLASS comparison](kernels/competitive-cutlass.md) |
| Do we have disassembly from the actual launch? | Four Mapal Contract images pass exact PTX/context/module/function/launch joins. The later NCU library diagnostic was denied counter permission and produced no report; callback library capture is being prepared. | [Executed Mapal kernels](disassembly/executed-mapal-images.md) |
| What about 16K and model-shaped matrices? | Earlier large-shape measurements and all failures remain available; they are distinct from the M128 composed test. | [Large matrices](workloads/large-matrices-and-mlp-validation.md) |

Next: capture the selected cuBLAS/CUTLASS/Mapal images without requesting hardware counters, compare hardware realization against known IR facts, then test any proposed mechanism with a controlled intervention. The composed fused/unfused library comparison is complete at the two measured widths; it does not establish the cause of Mapal’s gap.

The allocation correction is integrated; the kernel alternatives remain isolated. Current test status, active experiments and rental ownership are recorded in [the live handoff](../../next-session.md).

[Browse GPU reports](README.md) · [Browse raw GPU evidence](../../../benches/results/gpu/README.md)
