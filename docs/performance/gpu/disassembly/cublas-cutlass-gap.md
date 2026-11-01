# Mapal, cuBLAS and CUTLASS: disassembly gaps

Historical provenance: S57.

Frozen source: `b4152c5`. This comparison uses accepted A100, RTX 4090 and RTX 5090
primary results plus the separate RTX 5090 large-shape experiment. No compiler change,
new compilation or GPU action was performed for this analysis.

The [full report](../../../../benches/results/gpu/disassembly/gap-analysis/REPORT.md),
[30-row table](../../../../benches/results/gpu/disassembly/gap-analysis/comparison.csv),
[independent review](../../../../benches/results/gpu/disassembly/gap-analysis/independent-review.md)
and [correction receipt](../../../../benches/results/gpu/disassembly/gap-analysis/correction-receipt.json)
preserve instructions, exact profiler/function joins, values, resources and limitations.
The corrected report binds 108 inputs and 11 outputs. Its full recovery copy contains
287 hash-checked files / 186,357,087 bytes under
`.claude/worktrees/_archive-s54-20261021/s57-ptx-sass-gap/`; see the
[archive receipt](../../../../benches/results/gpu/disassembly/gap-analysis/archive-verification.json).
Absolute temporary paths in the original report are provenance paths; this archive
preserves the extracted profiling evidence after those temporary paths expire.

## Results and interpretation

Thirty-sample CUDA-event medians. Each row compares implementations on the same GPU
run; the 16K supplement uses a different 5090 host from the primary. These are the
pinned library configurations, not an exhaustive library tuning search.

| GPU / shape / requested face | Mapal ms | cuBLAS ms | CUTLASS ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| RTX 5090 / 4096 square / FMA | 2.6327 | 2.3869 | 2.6171 | cuBLAS wins; no Mapal lead over CUTLASS |
| RTX 5090 / 16K square / FMA | 191.6580 | 154.1048 | 194.0900 | cuBLAS wins; small CUTLASS difference is not an established gain |
| RTX 5090 / 16K square / TF32 | 115.7243 | 80.9953 | 101.6248 | Both libraries win |
| RTX 5090 / M16, N18432, K7168 / FMA | 2.0644 | 0.3400 | 0.5530 | Mapal is 6.07× slower than cuBLAS |

RTX 4090 FMA is a positive counterexample: library/Mapal ratios at 4096 are
1.055× cuBLAS and 1.053× CUTLASS; at 8192 they are 1.023× and 1.035× on the primary.
The later [independent confirmation](../validation/integrated-placement.md#independent-rtx-4090-confirmation)
repeats the favorable 4096 median ratios (1.053× / 1.067×), but **does not repeat
the 8192 win** (1.000× / 0.932×). Its 8192 distributions are broad and overlap.
Higher Mapal registers/thread do not prevent the observed primary wins, so lower
register count alone is not a justified optimization objective. The original disassembly
artifact remains a primary-run analysis; the later timing evidence is preserved separately.

Only Mapal has numeric acceptance gates in the extracted-kernel harness. Exact/FMA
uses its ordered bitwise oracle; TF32 uses four times cuBLAS's sampled error. Library
errors are diagnostics. The diagnostic denominator is `max(abs(f64_ref), 0.001)`;
there is no fixed absolute accuracy threshold. The accepted M16 expert TF32 gate has
diagnostics 3.302 / 3.302 / 3.248 for Mapal/cuBLAS/CUTLASS. The full report and
[large-shape tables](../workloads/large-matrices-and-mlp-validation.md) preserve this limitation and the
two rejected M1 TF32 invocations. A harness pass does not certify application accuracy.

## Proven differences and required follow-up

1. **Small-M route:** the dense M16 FMA kernel computes one output per thread with a
   scalar K-loop, no shared memory and no cooperative tile. The route and large timing
   gap are proven. Test shape-derived cooperative loads/skinny tiles across all three
   projection families, preserving the declared accumulation order. Repeated load
   instructions alone do not prove DRAM traffic or its share of elapsed time.
2. **Blackwell TF32 conversion:** saved Mapal PTX uses 96 static `.rna` conversions,
   lowered by offline ptxas into integer adjustment/check sequences. The pinned CUTLASS
   executable uses 96 hardware `F2FP` conversion instructions. Their tie rules differ;
   the halfway example is independently verified. `Reduced::Tf32` explicitly specifies
   `cvt.rna` in `crates/backends/llvm/src/lib.rs:408`. The ties-even proposal in ADR-0039
   is undecided. Changing conversion is a semantic change, and its timing benefit is
   unproven. Keep the current contract unless the language decision changes.
3. **Tiled pipeline:** inspected TF32 kernels already use MMA and asynchronous copies.
   FMA/TF32 driver records report no local bytes, and corresponding offline ptxas reports
   no spills. Neither missing Tensor Cores nor spills is an established explanation.
   Isolated tile/pipeline experiments need actual executed-image evidence and dynamic
   memory/issue counters before assigning a cause.

Mapal SASS here is separately assembled; the actual driver-JIT image is not saved.
CUTLASS fatbin functions match actual profiler names. cuBLAS SASS and useful hardware
counters are absent. This analysis identifies experiments, not proven faster fixes.
CPU/GPU joint placement, allocation, retention and composed-program timing are separate
work items; the [host LLVM evidence](../../../../benches/results/gpu/host-dispatch/optimization-audit/README.md)
proves existing CPU vectorization around GPU calls, not joint scheduling gains.
