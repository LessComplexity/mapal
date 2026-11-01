# Actually executed Mapal, cuBLAS and CUTLASS kernels

The repaired observer now admits all three implementations and all four selected kernel
entries for **M16 × N2048 × K7168, Contract**. The captured cuBLAS call launches a
CUTLASS-named GEMM followed by a split-K reduction. The frozen Mapal implementation and
the selected standalone CUTLASS configuration each launch one kernel. This establishes
an actual dispatch difference, without establishing how much of the timing gap it causes.

The capture uses the exact previously measured binary, original paths, synthetic inputs
and selected CUTLASS candidate `m16-n64-k16`. It does not execute the later compiler
correction at `50a57e5`. Instrumented durations are excluded from performance admission.

## Launches and numerical results

| Implementation / selected entry | Grid | Threads per block | Dynamic shared bytes per block |
| --- | --- | ---: | ---: |
| Mapal independent-output kernel | 128 × 1 × 1 | 256 | 0 |
| cuBLAS GEMM, `cutlass_80_simt_sgemm_128x32_8x5_nn_align1` | 8 × 2 × 32 | 128 | 25,600 |
| cuBLAS `splitKreduce_kernel` | 64 × 1 × 1 | 512 (32 × 16) | 0 |
| Tuned standalone CUTLASS | 1 × 32 × 1 | 128 | 10,496 |

These are captured launch arguments. Named library tile dimensions are not substituted
for Mapal's logical axes; exact partition extents require argument and address analysis.
Dynamic shared bytes are not total shared allocation or measured traffic. More blocks
do not by themselves establish achieved occupancy or speed.

| Implementation | Ordered-FMA bit matches among the 256 checked outputs | Existing componentwise error gate |
| --- | ---: | --- |
| Mapal | 256 / 256 | Pass |
| Tuned CUTLASS | 256 / 256 | Pass |
| cuBLAS | 5 / 256 | Pass |

The same counts hold before and after each selected-call process. Independent arithmetic
checks cover all **4,608 exported IEEE words** across three processes, three implementations
and two phases. Each phase samples256 of32,768 output cells. CUTLASS's observed matches
are not a universal ordered-bit guarantee. cuBLAS's251 differing sampled outputs prove
that this invocation does not reproduce the ordered-FMA result; they remain within the
frozen library comparison budget.

## Instructions visible in the selected entries

| Entry | Decoded structure | Recorded resources |
| --- | --- | --- |
| Mapal | Global loads feed one ordered FMA chain per output; no shared-load/store or barrier mnemonics in this entry | 40 registers/thread, local0 |
| Tuned CUTLASS | Shared-memory staging, barriers and several independent output accumulators | 56 registers/thread, local0, static shared0 |
| cuBLAS GEMM | `LDGSTS` global-to-shared copies, dependency barriers, shared loads and multiple accumulators | Internal register/static-shared attributes not captured |
| cuBLAS reduction | Loads, FP32 additions and output stores in a second selected entry | Internal register/static-shared attributes not captured |

NVIDIA documents `LDGSTS` as an asynchronous global-to-shared copy. Its presence proves
the instruction is in the captured entry, not how much transfer overlaps computation.
[Instruction reference](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html#instruction-set-reference)

All four decoded entries contain scalar `FFMA`; no MMA-family mnemonic appears in these
selected listings. That is a static instruction observation, with the decoder limitations
below, rather than a measurement of tensor-core utilization. Static instruction totals
include unrolled loops, branches, optional paths and padding; the cuBLAS GEMM even has22
constant-false LDS rows. They cannot be compared as runtime operation counts.

[Exact instruction offsets, line anchors and static counts](../../../../benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/analysis/REPORT.md)
make these observations reproducible. They do not identify a new speedup or prove that
the missing staging in this frozen Mapal selection is the cause of its earlier timing.

## What the compiler can use

Existing IR already carries M/N/K, affine addresses, independent output cells, reduction
operand order and lifetimes. The backend supplies device resource limits. The
[fact map](ir-facts-and-realization-gaps.md) names their real consumers. The missing
ingredient is not another copy of matrix dimensions.

The following experiments follow from the evidence:

1. **Preserve ordered arithmetic:** test cooperative loads, output coarsening and K-panel
   staging as independent factors, retaining ordered output bits and unchanged controls.
   Earlier cooperative alternatives have both wins and regressions; this capture does
   not promote them.
2. **Test asynchronous staging separately:** use existing affine/alignment and target
   facts to admit a copy/pipeline alternative, keeping arithmetic and output mapping
   fixed where possible. Capture its exact code and measure fresh ordinary distributions.
3. **Permit a different reduction order explicitly:** model the numerical permission,
   then let reduction shape and hardware facts select legal partial reductions, scratch
   lifetimes and completion dependencies. Current Contract allows ordered FMA, not a
   split-K/tree reassociation. A cuBLAS-like realization cannot silently replace it.

For either route, compare a frozen selection in fresh ordinary runs with matched inputs
and clocks. An actual split-K dispatch is stronger evidence than an algorithm attribute,
but its causal speed contribution still needs a controlled split/no-split comparison.
The [experiment sequence](optimization-experiments.md) keeps dispatch, retention,
allocation and fusion measurements separate from primitive-kernel timing.

## Provenance and limits

The diagnostic RTX5090 is `GPU-6f5a16b4-bef2-c3b0-6c58-c80edb2eed53`, driver610.43.02.
The earlier competitive timing host used driver595.91.07. This new capture cannot
retroactively establish the earlier host's selected kernels or explain its timings.
The [published timing table](../kernels/competitive-cutlass.md) remains unchanged.

Admission checks exact binary/request/source/argv and all module, context, function,
invocation, activity, launch and decoder associations. Mapal joins directly through the
observed loader/function and exact PTX. Library entries require a unique active ELF
entry plus authoritative FUNCTION metadata. All selected internal launches are retained.
The independent reader recomputes numerical/provenance checks and replays the sealed,
independently reviewed attribution parser; it is not a second join implementation.

All four decoder outputs contain unknown `?trans` and `?WAIT` annotations. Qualified decoded
instruction facts remain reportable; annotation meanings, dynamic instruction counts,
memory traffic, stalls and achieved occupancy remain unmeasured. No counters were
requested and no host security settings changed.

All504 recovered files and the10,938,433-byte archive are independently rehashed. The
owned instance57271085 is independently absent at20:48:03.025Z. The preceding unavailable
offer produced no instance; all pre-existing user machines were protected.

[Raw disassembly and admission evidence](../../../../benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/README.md)
· [Current results](../current-results.md)
