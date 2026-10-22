# Requested-TF32 M1 fallback: preserved failure

The M1/N2048/K7168 DeepSeek expert case fails its existing requested-TF32 value
gate in both base and null runs. Its Contract counterpart passes the existing
bitwise in-order FMA gate. This establishes a gate/realization distinction; no
compiler wrong-arithmetic defect or numerical cause of a specific cell is proved.
The supplemental experiment remains overall pass=false:98/100 invocations pass.
No gate, source, binary, benchmark input or GPU state was changed here.

`fallback-value-investigation-receipt.json` binds the primitive integrity proof,
raw outputs, process statuses/argv, harness source/binary and compiled artifacts.
The proof's integrity_pass=true, pass=false, result/archive/manifest SHA bindings
were independently rechecked. Actual raw files are under
`integrated/results/primitive-rtx-5090/retrieved/trace/primitive-results/`.

|Observed item|Contract|Requested TF32|
|---|---|---|
|Entry|mapal_tile_538afa580cc851c9|mapal_tile_5343b47e4b7c8bad|
|Grid/block/shared bytes|8×1 /256 /0|8×1 /256 /0|
|Runtime registers/local bytes|40 /0|40 /0|
|Launched entry arithmetic|8 static fma.rn.f32 sites in K loop|same|
|Floating narrowing/TF32 casts or MMA sites|0|0|
|Mapal printed maxrel vs f64|1.787e-3|1.787e-3|
|cuBLAS printed maxrel vs f64|1.248e-4|1.248e-4|
|CUTLASS printed maxrel vs f64|1.787e-3|2.652e-1|
|Mapal gate|bit_exact_vs_in_order_fold=yes|within_4x_cublas_tf32=NO|
|Base/null process exit|0 /0|1 /1|
|Timing rows|present|absent: gate failed before timing|

Both complete PTX modules are identical after replacing only their primary entry
symbol with one common name; normalized SHA is
`5a61117f0372e2c2602175777785c7959ba0f3387d4d4665e1fc76c7eee59482`.
The original module SHAs remain distinct because the symbols differ. Within each
case its base/null PTX files are literally byte-identical. The proof is stronger
than finding an FMA instruction: the full launched entry, load/store widths,
loop instructions/constants and geometry match. Operands are loaded with
`ld.global.nc.b32`; eight `fma.rn.f32` operations advance the scalar accumulation;
the result is stored with `st.global.b32`. Its only conversions are integer
address/index conversions, with no floating narrowing or TF32 operation. The
other module entries are input generators; the harness launches the recorded
primary entry with its own seeded f32 A/B buffers.

In unchanged `tile_bench.cu:414`, Contract's sampled oracle uses fmaf. At417 it
compares Mapal bits against that oracle when tf32=false. At423–426 requested
TF32 instead checks maxrel_Mapal <=4*maxrel_cuBLAS. From printed diagnostics the
observed ratio is14.31891, versus allowed4 (threshold approximately4.992e-4).
This directly explains the recorded gate failure while the corresponding
Contract arithmetic passes its own contract. Printed errors are rounded, but
the gap is much larger than that rounding. Actual gate evaluation uses the
unrounded errors inside the unchanged harness.

The stderr text `1 of256cells differ from the in-order fold` is misleading in
this mode: line426 sets `mismatches=1` as a Boolean when the global TF32 error
gate fails. It does not enumerate one failing sampled cell. Saved output gives
no cell index, per-cell f64 reference, absolute error or mismatch count; no
cancellation/reduction-order cause should be asserted from these summaries.
The cuBLAS math-mode request also does not identify its actual kernel arithmetic.
No gate was relaxed, no failed-case timing is synthesized, and the large CUTLASS
TF32 error remains an explicit diagnostic rather than an independently accepted
accuracy guarantee. Additional numeric instrumentation would be a separate
reviewed experiment.
