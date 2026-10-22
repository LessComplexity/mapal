# Current GPU results and next experiments

Updated 2026-10-22. Correctness, steady timing, allocation counts and profiling are separate evidence classes. No generic win over cuBLAS has been established.

| Question | Verified result | Detailed report |
| --- | --- | --- |
| Does scheduling consume IR and hardware facts consistently? | A proved request/actual-arithmetic mismatch is corrected. 65 library and176 integration tests pass; eight successful LLVM/PTX controls are unchanged. No new GPU speed result. | [Fact-driven scheduling](disassembly/fact-driven-scheduling.md) |
| Are the composed clocks valid? | Completion is included; no proved unit or missing-sync defect. They measure warmed resident computation, with differing input preparation and allocation behavior across arms. | [Measurement boundaries](workloads/measurement-boundaries.md) |
| Does Mapal beat competitive fused libraries on a residual MLP? | No: at M128/d256/h512 and M128/d1024/h2048, Mapal takes0.073090/0.128935 ms versus tuned fused cuBLASLt0.032659/0.059620 ms (2.24×/2.16× faster). Tuned fused CUTLASS has lower observed medians too; Mapal beats tested eager/graph-disabled Inductor. All30 same-device confirmation/control rows admitted. | [Fused library comparison](workloads/fused-library-comparison.md) |
| Can known dataflow eliminate allocations? | Yes: unused raw-output allocations drop from 7→6, 8→7, 12→10 or 8→6 in the tested cases; controls stay unchanged. 640 ordinary processes and 28 separate counter processes pass. Latency gain remains unmeasured. | [Unused output allocations](memory/dead-output-allocation.md) |
| Does the 8×64/K16 tile alternative help? | Against the original, FMA improves 1.253× at M16/N2048/K7168 and 1.159× at M16/N1024/K4096; Exact takes 1.227× and 1.013× as long, respectively. Tile aspect and panel depth change together. cuBLAS is faster in all four cells; no selector promotion. | [Tile geometry and panel comparison](kernels/panel-depth-comparison.md) |
| What changes when CUTLASS is tuned? | Six of twelve configurations compiled and passed gates. Selected 16×64×16 is 2.32–2.34× faster than the old default and beats the frozen Mapal baseline in all four cells; cuBLAS remains 4.31–7.01× faster than tuned CUTLASS. | [Competitive CUTLASS comparison](kernels/competitive-cutlass.md) |
| Do we have disassembly from the actual launch? | Four earlier Mapal Contract images are admitted. The new counter-free pilot ran Mapal and passed numerical checks, then rejected callback attribution before disassembly; library legs were skipped. Exact failed evidence is retained. | [Executed Mapal kernels](disassembly/executed-mapal-images.md) |
| What about 16K and model-shaped matrices? | Earlier large-shape measurements and all failures remain available; they are distinct from the M128 composed test. | [Large matrices](workloads/large-matrices-and-mlp-validation.md) |

Next: prove and repair the observer's live callback/schema mismatches locally, then capture the selected library images. The [experiment sequence](disassembly/optimization-experiments.md) separates geometry, submission, retention, allocation and fusion. The composed fused/unfused library comparison is complete at the two measured widths; it does not establish the cause of Mapal’s gap.

The allocation correction is integrated; the kernel alternatives remain isolated. Current test status, active experiments and rental ownership are recorded in [the live handoff](../../next-session.md).

[Browse GPU reports](README.md) · [Browse raw GPU evidence](../../../benches/results/gpu/README.md)
