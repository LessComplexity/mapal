# Independent final PTX/SASS comparison — 2026-10-21

Read-only analysis of frozen `b4152c5`, accepted A100/4090/5090 final primaries and the separately retrieved 5090 primitive extension. All 328 frozen source files and HEAD remain unchanged. No source edits, new compilation, remote actions or GPU execution were performed. `verification.json` binds raw medians, accepted profiling archive hashes, 18 shipped PTX hashes, 9 exact CUTLASS profiler/fatbin names and 12 supplemental stdout hashes. `artifact-manifest.json` records every inspected input. Semble MCP search did not return before termination; this report makes artifact-level claims rather than inventing source-level selection rationale.

## Measured gap, before assigning causes

30-sample ordinary CUDA-event medians; ratio = library time / Mapal time, so >1 favors Mapal. Each listed supplemental row passed the harness’s Mapal acceptance gate. cuBLAS and CUTLASS error outputs are diagnostics; the harness imposes no independent accuracy acceptance threshold on either library. The supplemental run as a whole failed 2 of 100 invocations (separate M=1 TF32 failures), and is not accepted as a fully passing run. Different machines/drivers/power limits mean 16K cannot be treated as a same-host scaling experiment. These are the pinned library configurations, not an exhaustive tuned-library search.

| GPU | N (square) | Arithmetic | Mapal ms | cuBLAS / Mapal | CUTLASS / Mapal |
|---|---:|---|---:|---:|---:|
| a100 | 4096 | FP32 FMA | 7.7763 | 0.929 | 0.972 |
| a100 | 4096 | TF32 | 1.6466 | 0.662 | 0.712 |
| a100 | 8192 | FP32 FMA | 58.5492 | 0.980 | 0.979 |
| a100 | 8192 | TF32 | 13.5311 | 0.656 | 0.719 |
| rtx-4090 | 4096 | FP32 FMA | 2.3480 | 1.055 | 1.053 |
| rtx-4090 | 4096 | TF32 | 1.7019 | 0.911 | 0.926 |
| rtx-4090 | 8192 | FP32 FMA | 20.1687 | 1.023 | 1.035 |
| rtx-4090 | 8192 | TF32 | 13.7912 | 0.937 | 0.946 |
| rtx-5090 | 4096 | FP32 FMA | 2.6327 | 0.907 | 0.994 |
| rtx-5090 | 4096 | TF32 | 1.5460 | 0.899 | 0.886 |
| rtx-5090 | 8192 | FP32 FMA | 21.0535 | 0.932 | 1.002 |
| rtx-5090 | 8192 | TF32 | 11.5948 | 0.867 | 0.892 |
| 5090 supplement | 16384 | contract | 191.6580 | 0.804 | 1.013 |
| 5090 supplement | 16384 | exact | 285.6049 | 0.550 | 0.692 |
| 5090 supplement | 16384 | tf32 | 115.7243 | 0.700 | 0.878 |

The complete 18 final timing rows, including Exact, are retained in `analysis.json`. Exact emits separate rounded multiply/add and has an ordered-fold value contract; replacing those operations with FMA is a semantic change. Its measured loss against FMA libraries is not by itself a missing optimization.

The error diagnostic printed as `maxrel_vs_f64` is exactly `max_sample(abs(got - f64_ref) / max(abs(f64_ref), 0.001))` (`tile_bench.cu:419`). It is a clamped/scaled relative-error measure, not ordinary relative error when the denominator clamp applies. In the included 16×2048×7168 requested-TF32 base case, the raw diagnostics are Mapal **3.302**, cuBLAS **3.302**, CUTLASS **3.248**; Mapal passes because its value is within four times cuBLAS’s. In the 16K TF32 base case they are Mapal **0.1043**, cuBLAS **0.1059**, CUTLASS **0.1059**. These values supply no fixed absolute-error bound and must not be read as percentages. This report makes no independent library-accuracy claim.

## Three actionable distinctions, with limits

### 1. Small-M algorithm selection is a larger measured gap than the square FMA gaps

| Shape M×N×K | Requested face | Actual Mapal route | Mapal ms | cuBLAS ms | CUTLASS ms |
|---|---|---|---:|---:|---:|
| 16×18432×7168 | contract | fp32_fma | 2.0644 | 0.3400 | 0.5530 |
| 16×18432×7168 | tf32 | fp32_fma | 2.0716 | 0.3338 | 0.3359 |
| 16×2048×7168 | tf32 | mma_sync_tf32 | 0.1516 | 0.0451 | 0.3256 |
| 16×1024×4096 | contract | fp32_fma | 0.1475 | 0.0184 | 0.3011 |
| 16×1024×4096 | tf32 | mma_sync_tf32 | 0.0860 | 0.0184 | 0.1884 |

The dense M=16 FMA entry maps one thread to one output at `/private/tmp/mapal-s56-gpu/large-composed/primitive/run-bundle-rtx-5090/cases/deepseek_dense_m16_n18432_k7168__contract/base.ptx:30`; its K loop at line 43 loads two scalar global operands per FMA and carries the same accumulation chain through all 7168 K positions (four steps per loop). No shared memory, warp shuffle, async copy or MMA appears in that entry; driver resources report 40 registers, 0 shared, 0 local bytes. This proves the current route is a per-output dot-product fallback, not a tiled cooperative implementation. It is 6.07× slower than cuBLAS in this row; the Qwen-shaped FMA fallback is 8.02× slower. The narrower TF32 expert shapes do use MMA, but still lose to cuBLAS while beating this pinned CUTLASS configuration. Thus neither 'always use Tensor Cores' nor 'copy the current CUTLASS configuration' follows from the data.
Minimum next experiment: compare shape-derived small-M cooperative reduction/skinny-tile candidates with the existing route on the same inputs and precision contract. Preserve ordered accumulation where required; prove legality before changing its reduction tree. Measure all three shape families, not a benchmark-name switch. Dynamic memory traffic/cache hit rates are absent, so repeated scalar loads are established instructions, not proven DRAM traffic or the sole timing cause. Prior fallback equivalence/value-gate investigation is reused, not reinterpreted here.

### 2. Blackwell TF32 conversion differs in both instructions and rounding contract

Mapal's entry has 96 static `cvt.rna.tf32.f32` sites, beginning at `/private/tmp/mapal-s56-gpu/integrated/bundles/rtx-5090/rtx-5090/mm_4096_tf.ptx:205`. Saved offline ptxas SASS contains 96 `IMAD.IADD.U32` sites and finite/Inf checks, with zero `F2FP` instructions; representative conversion sequence at `/private/tmp/mapal-s56-gpu/ptx-sass-gap/rtx-5090-primary-retry4/prof/mm_4096_tf.sass:553`. The pinned CUTLASS TF32 function has 96 `F2FP.TF32.F32.PACK_B` sites, first at `/private/tmp/mapal-s56-gpu/ptx-sass-gap/rtx-5090-primary-retry4/prof/tile_bench.sass:847`. The function's mangled name exactly matches the actual Nsight kernel record. Counts cover the static function, including loop/drain copies; they are not dynamic instruction counts.
This is not a drop-in equivalent instruction replacement. Pinned CUTLASS `include/cutlass/gemm/warp/mma_tensor_op.h:363` chooses `PreferredRoundingMode`; `numeric_conversion.h:8358` defaults to nearest-even and selects half-ulp/truncate only for `__CUDA_ARCH__ < 900`; its nearest-even conversion uses `cvt.rn.tf32.f32` at line 647 on >=900. Mapal emits nearest-ties-away (`.rna`). NVIDIA documents the distinction and >=sm90 support for `.rn` TF32 conversion in the [PTX conversion specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt). For the exactly representable FP32 halfway input `0x3f801000` = 1.00048828125, nearest-even produces 1.0 and nearest-away produces 1.0009765625; `verification.json` checks the exact scalar arithmetic, not a GPU execution.
The current interim contract is specified in source: `crates/backends/llvm/src/lib.rs:408` documents `Reduced::Tf32` as operands rounded with `cvt.rna`. Accepted ADR-0032 requires bit-changing choices to belong to the language and forbids rounding changes via ambient backend config. ADR-0039 proposes explicit nearest-even `to_tf32` conversion, but is marked candidate / NOT decided. Consequently replacing `.rna` with `.rn` changes the current documented numeric face; it is not a value-invariant tuning change. A separately declared nearest-even experimental face could measure the opportunity, with midpoint, signed, subnormal, Inf/NaN and end-to-end checks, but results must remain labeled as a different numeric contract. Capture the actual driver-loaded image or load the tested cubin directly, and measure instruction/issue counts before attributing a performance gap to the conversion. The A100/4090 CUTLASS fatbins also use integer conversion sequences, so this is specifically a proven Blackwell distinction, not an explanation for the large A100 TF32 gap.

### 3. Pipeline/tiling differs, but causal counters are missing

All three final 4096/8192 TF32 Mapal entries contain `mma.sync...m16n8k8...tf32` and `cp.async` commit/wait operations. Their offline 4096 SASS contains HMMA. FMA tiled entries contain 128-bit global loads, vector shared loads, FFMA and shared-memory barriers. Thus absent Tensor Cores, absent async copies, or entirely scalar loads cannot explain these tiled paths. A pending copy group is issued before arithmetic in the loop, but static order does not measure useful overlap.

| 4096 actual Nsight resources (registers/thread; threads/block; total shared bytes) | Mapal | cuBLAS | CUTLASS |
|---|---|---|---|
| a100 FMA | 128; 256; 16640 | 122; 128; 12544 | 126; 256; 16640 |
| a100 TF32 | 230; 128; 71680 | 230; 256; 147456 | 224; 128; 65536 |
| rtx-4090 FMA | 218; 128; 16640 | 202; 256; 49152 | 122; 256; 16640 |
| rtx-4090 TF32 | 198; 128; 37888 | 220; 256; 73728 | 224; 128; 65536 |
| rtx-5090 FMA | 128; 256; 16640 | 210; 256; 49152 | 124; 256; 16640 |
| rtx-5090 TF32 | 122; 256; 37888 | 234; 128; 98304 | 230; 128; 65536 |
These resource rows each have eight profiler launches, and all report local bytes/thread = 0. The 4090 FMA kernel wins despite having 218 registers/thread versus CUTLASS 122. This refutes using a lower register count alone as an optimization target; it does not establish actual occupancy. A100 TF32 loses to CUTLASS despite equal 128-thread blocks and similar register counts (230 vs 224). 5090 FMA and CUTLASS have the same 32×32 grid, 256-thread blocks and 16640 shared bytes, yet differing timing; those matching resources do not establish matching execution.
The actual 5090 cuBLAS FMA name selects `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4...>` and its grid is 256×2; this is the cuBLAS leg, not the separate pinned CUTLASS 128×128 kernel. The actual A100 cuBLAS FMA uses `ampere_sgemm_128x64_nn` at 4096 and `ampere_sgemm_128x128_nn` at 8192. Different library-selected tiles are proven. TF32 CUTLASS uses a four-stage 128×128×16 template (`tile_bench.cu:71–81`); Mapal PTX uses waits for zero pending groups and ping-pong storage. Whether deeper pipelining or another tile is better requires an isolated same-contract experiment and memory/issue stall measurements. Static function sizes, barrier totals and address arithmetic cannot be compared as dynamic cost without normalizing loop trips and epilogues.

## Evidence boundaries and counterevidence

- No exact driver-executed Mapal JIT machine image or cuBLAS library disassembly is saved. `box_profile.sh:19–24` separately invokes ptxas and cuobjdump; Mapal timing loads PTX through the driver. CUTLASS SASS is from the executable fatbin and its exact function names are bound to Nsight. 16K has PTX and runtime attributes, but no saved library Nsight identities or SASS. Do not silently extend the 4096 disassembly to its 16K library implementation.
- All scoped FMA/TF32 driver records have zero local bytes and their offline ptxas logs have zero spill stores/loads. The broad hypothesis that their gaps are caused by spilling has no supporting evidence. 5090 Exact is different: driver local bytes are nonzero, and offline SASS has local loads/stores; that is preserved, but neither dynamic spill traffic nor its cost is measured.
- No achieved occupancy, DRAM/cache/shared throughput, issue/stall or dynamic instruction counter data is present. The NCU CSVs are empty. No resource/ratio observation is promoted to a bottleneck diagnosis.
- The Mapal acceptance gate checks 256 sampled outputs: Exact/Contract require ordered-fold bit equality; requested TF32 requires Mapal’s clamped error diagnostic to be no greater than four times cuBLAS’s diagnostic. There is no independent fixed absolute or relative accuracy bound for TF32 and no library-accuracy acceptance gate. Sampling does not prove every output cell. There is no composed-program, allocation, retention, PyTorch or paged-attention performance claim in this single-kernel audit.

## Reproducibility

Run `python3 /private/tmp/mapal-s56-gpu/ptx-sass-gap/analyze.py`, then `verify.py`, then `write_report.py`. `analysis.json` includes full opcode counts with line lists, profiler SQL and kernel names, raw resource lines and timing row provenance. `artifact-manifest.json` has complete SHA256 hashes; excerpts are not detached from originals. No tuning change is recommended as proven faster; the prioritized experiments above follow measured gaps and concrete instruction differences.
