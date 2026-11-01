# Independent static review of PTX/SASS comparison

Reviewed REPORT.md SHA
e6b28b8a95fa852e28db9a6f12976154f8bfaa798b86a1ca1776eceb763a8e10,
comparison.csv, analysis/verification/artifact manifests and completion receipt.
No source edit, build, GPU execution or remote action was performed. Only this
review file was written; author outputs remain unchanged.

The instruction/provenance evidence supports the report's bounded observations.
Two presentation corrections are needed before publishing:

1. REPORT.md:5 says each supplemental row passed a three-leg sampled value gate;
   line63 refers to external-library numeric gates. The primitive tile harness
   only gates Mapal's values. cuBLAS/CUTLASS error values are diagnostics, without
   an independent absolute or relative accuracy acceptance threshold. In frozen
   tile_bench.cu:419 the metric is specifically
   `max_sample(abs(got-f64_ref) / max(abs(f64_ref), 1e-3))`, a clamped/scaled
   relative-error diagnostic. It must not be interpreted as ordinary relative
   error or a percentage when the clamp applies. For the included
   deepseek_expert_m16_n2048_k7168__tf32/base/harness.stdout, raw lines4–6 report
   CUTLASS3.248e+0, cuBLAS3.302e+0 and Mapal3.302e+0; Mapal's relative-to-cuBLAS
   four-times gate still says yes. These large **diagnostic values**, not a
   fixed-accuracy guarantee, must remain visible. The primitive suite remains
   overall pass=false, and all timing rows are individually gated Mapal cases.
   For the16K TF32 row the diagnostics are Mapal.1043 and both libraries.1059.
   No absolute error or330% interpretation follows from those numbers alone.

   The large supplemental repository report already distinguishes mandatory
   Mapal gates from library diagnostics at
   docs/performance/s57-large-composed-evidence.md:31–34 and exposes requested
   face/actual route in tables. Its phrase “validates256sampled outputs per leg”
   at31 is too broad; “computes error diagnostics” would remove that ambiguity.
   It does not otherwise independently assert accepted library accuracy.
   Explicitly showing the clamped metric and the large admitted TF32 diagnostics
   would make the numerical limit clear in both reports.

2. REPORT.md:30 small-M table needs a requested-face column. Its two
   16×18432×7168/fp32_fma rows are Contract and requestedTF32 respectively:
   2.0644/0.3400/0.5530ms versus2.0716/0.3338/0.3359ms. The same emitted arithmetic
   route does not make their library math requests/value gates identical.
   comparison.csv already retains the face; expose it in the human table too.

Independent checks completed:

- All104 artifact-manifest input hashes/sizes and all9 completion-output
  hashes/sizes match. Additionally, every inspected prof member was hashed
  directly inside its accepted profiling tar:10 members per A100/4090/5090,
  all30 equal the inspected SASS/ptxas/SQLite files. This closes the provenance
  join between accepted archive digests and extracted inspection inputs.
- All30 comparison rows' medians and library/Mapal divisions match the bound
  final/supplemental source records. The final timing rows were already checked
  independently against saved raw samples in the separate export audit. No
  timing/resource ratio establishes causation, speedup from conversion changes,
  occupancy, useful overlap, bandwidth pressure or dynamic instruction count.
- All9 CUTLASS resource-to-fatbin joins were independently re-read from SQLite
  mangled-name StringIds and matched exactly one function marker in the proper
  profile's SASS. These are9 joins, not necessarily9 distinct function names.
  SASS architecture headers are sm80/sm89/sm120 as appropriate. The resource
  table's registers, block threads and total shared bytes match the queried
  rows; reported launch counts are8. No cuBLAS disassembly or Mapal driver-JIT
  machine image is present. The report correctly labels Mapal SASS as offline
  ptxas output and CUTLASS as executable-fatbin SASS matched by function name,
  rather than claiming a captured driver-loaded machine image.
- For RTX5090 TF32, independently counted96 `cvt.rna.tf32.f32` sites in the
  exact launched PTX entry. Its matching offline SASS function has96
  IMAD.IADD.U32 sites and0 F2FP.TF32.F32.PACK_B; the exact CUTLASS fatbin function
  has96 F2FP.TF32.F32.PACK_B sites. Other entry/generator functions were excluded.
  These are static counts; no one-for-one dynamic-cost or performance inference
  is justified. The pinned CUTLASS converter at numeric_conversion.h:647 and
  PreferredRoundingMode at8358–8367 support the stated RN versus RNA distinction.
- The0x3f801000 example is correct by independent bit arithmetic. It equals
  1+2^-11, exactly halfway between TF32 values1 and1+2^-10; retained low bit is
  even. RN-even gives0x3f800000=1, RNA-away gives0x3f802000=1.0009765625.
  This is a semantic distinction, not proof that a rounding substitution is
  legal or faster, and no GPU rounding test was performed. NVIDIA's
  [PTX rounding/cvt specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
  confirms these rules and the newer architecture requirement for RN TF32.

The causal and legality limits in the current report should remain intact.
After the wording/table correction, refreshing the author outputs and completion
hashes is sufficient; no new compilation or performance conclusion follows.
