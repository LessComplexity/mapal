# Task 3.6: competitive FP32 primitive references — review draft

Stage 1 prepares a bounded candidate space for the existing M16 GEMM comparison. It does not change Mapal, its emitted PTX/launches, the cuBLAS SGEMM call, the seed or input formula. The current pinned CUTLASS default remains a named control. Stage 2 adds explicit supported library epilogues for the composed residual MLP after the composed v4 baseline has actual admission.

## Workload and precision

Use the frozen B packet's eight cases: full-value bounded gates at M16/N256/K7168 and M16/N128/K4096, representative timing requests at M16/N2048/K7168 and M16/N1024/K4096, each under Mapal Exact and Contract labels. Input storage is FP32, row major, A stride K, B stride N and C/D stride N. Inputs retain the LCG seed `0x5eed` and `frand` construction. Alpha is 1 and beta is 0; output buffers are preallocated and separately owned per implementation. The new process requires NVIDIA_TF32_OVERRIDE=0 and records it. There is no layout conversion, TF32, autocast, TensorOp substitution, quantization, split-K or hidden fallback in this initial candidate space.

The existing cuBLAS arm remains SGEMM with `CUBLAS_DEFAULT_MATH`, opN/opN, dimensions N/M/K and operands B/A. The actual mode and library version are queried. The mode does not prove scalar execution; any executed Tensor Core or other internal library decomposition must be reported from actual kernel evidence, not inferred. The CUTLASS candidates all explicitly use `OpClassSimt`, Sm80 template tag and compilation for sm_120. The same pinned CUTLASS commit `91843b71d90737e0a9b86f7d9df213cff5a3526f` is used.

Evidence: `harness/tile_bench_checked.cu:frand` (114–117), `main` input setup (309–343), type definitions (65–76), handle/arguments (365–391), and `run` SGEMM call (395–406) in `/private/tmp/mapal-s60-kernel/bcomparison/`.

## Preregistered 12-candidate space

Every shape below is an experimental explicit instantiation, not a proposed universal selector. All have instruction tile 1×1×1, two pipeline stages, identity swizzle 1, A/B template alignment 1 element, FP32 accumulation and `LinearCombination<float,1,float,float>` epilogue. Serial split-K is false and split_k_slices is 1.

| ID | Threadblock M×N×K | Warp M×N×K | Source-derived threads | Total CTAs N1024 / N2048 |
|---|---|---|---:|---:|
|default-control|128×128×8|32×64×8|256|8 / 16|
|m16-n32-k8|16×32×8|16×32×8|32|32 / 64|
|m16-n64-k8|16×64×8|16×32×8|64|16 / 32|
|m16-n128-k8|16×128×8|16×32×8|128|8 / 16|
|m16-n64-k16|16×64×16|8×32×16|128|16 / 32|
|m8-n8-k8|8×8×8|8×8×8|32|256 / 512|
|m8-n16-k8|8×16×8|8×16×8|32|128 / 256|
|m8-n32-k8|8×32×8|8×16×8|64|64 / 128|
|m16-n16-k8|16×16×8|8×16×8|64|64 / 128|
|m4-n16-k8|4×16×8|4×16×8|32|256 / 512|
|m8-n16-k16|8×16×16|8×16×16|32|128 / 256|
|m4-n16-k32|4×16×32|4×16×32|32|256 / 512|

Source audit checks the pinned row-major/row-major `DefaultMmaCore` specialization, its tile/warp divisibility, warp-thread factorization and shared-memory padding divisibility. In particular, K16 uses warp M8 so its padding 2 is divisible by LaneM2; the superficially tempting K16/warp-M16 combination fails that source constraint and is excluded. All experimental candidate CTA M extents fit M16; none silently increases parallelism along K. Candidate legality is provisional until actual nvcc instantiation, `can_implement`, launch/resource checks and unchanged numerical gates pass on hardware. A rejection stays an explicit unsupported candidate.

Evidence: `cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:DefaultMmaCore<RowMajor,RowMajor>` (432–569), including `WarpCount`, `kThreads`, `simt_get_warp_threads_m`, `kPaddingM` and lane checks; `cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:GemmIdentityThreadblockSwizzle::get_tiled_shape/get_grid_shape` (65–127); `cutlass/include/cutlass/gemm/device/gemm.h:Gemm` default arguments (188–224) and `Arguments` (322–344).

## Numerical admission for every arm

Keep Mapal's existing ordered-bit requirements as additional independent labels: Exact requires the operationwise FP32 fold; Contract requires the true C `fmaf` fold. Library arms do not receive an ordered-bit guarantee merely by passing a tolerance gate.

For every checked output cell `(i,j)`, generate the two scalar FP32 oracle values. Instantiate the existing mathematical exact-input dot-product bound with exact integer/rational arithmetic:

`qA=A*2^23; qB=B*2^23; S_num=sum(abs(qA*qB))`

`E_exact=2*K*S_num/((2^23-K)*2^46)+4*K/2^149`.

The frozen input generator guarantees integer q values, verified during preparation. Store the smallest binary64 value E_ij greater than or equal to E_exact, proving it by exact rational comparison of the value and its predecessor. This replaces the unexecuted draft's inadequate fsum-plus-final-nextafter implementation; it preserves the mathematical bound and introduces no fitted tolerance.

An arm passes only if each actual value is finite and within E_ij of BOTH scalar FP32 oracle faces. The independent Python gate compares actual-reference differences exactly as rationals. The C++ gate conservatively encloses subtraction before its pre/post comparison. This is a mathematical componentwise absolute bound, not a measured maximum relative error or fitted tolerance. Alpha1/beta0 do not add a scaling or incoming-C error term. This stage allows reordered FP32 reductions under that budget while reporting ordered-bit matches separately. The existing full bounded output inventories and the 256 representative cells remain unchanged; there is no new all-tensor correctness claim for representative cases. Failure of any library's numeric gate excludes its tuning/timing observations, preserves its raw bits and reports a failure.

Evidence: `/private/tmp/mapal-s60-composed/packet-v3/pytorch/numerics.py:gamma` (17–19), `budgets.mm` (34–43; exact operands make propagated error zero), and `compare` (100–113). The B harness currently only enforces finite library values and prints error (`tile_bench_checked.cu:main`, 428–460); the new benchmark-only checker closes that gap without changing device computation.

## Storage, measurement and selection

Retain preallocated resident A/B and separate C/D outputs. CUTLASS initialization, any context/library setup, host oracle construction, allocations and cold calls are outside steady timing. Beta0 makes the prior output value irrelevant; no per-request output reset is needed or omitted from an algorithm that requires one. If later split-K/workspace algorithms need a reset, that reset belongs inside their measured request and must be separately recorded.

Preserve the primitive cache protocol: 256 MiB memset before each start event, then time the actual work with events. Record wall intervals separately if added; do not replace event data or claim events equal summed kernel cycles. Keep the original default-control and cuBLAS arms in the same confirmation protocol. Each candidate is built separately and runs in its own original three-leg process, preserving Mapal→cuBLAS→CUTLASS order. The default control receives an equivalent separate process. This is a new experiment and does not replace historical rows.

The bounded offline tuning budget is three warmups plus nine samples per numerical-admitted CUTLASS candidate/shape. Select by upper-median event time, break exact ties by candidate ID, freeze the choice in a receipt, and record tuning process wall time and all rejected candidates. Report confirmation from a fresh process with three warmups plus thirty held-out samples. The same chosen tile serves both Exact and Contract labels because the library implementation and inputs are identical; report Mapal's separate guarantees. Publish all candidate confirmation rows when budget permits; label a best-of-candidates result as a tuned result. Compile and tuning time are excluded from steady latency and published separately.

Record actual dtype/layout/strides, declared math mode, pinned library/compiler versions, candidate ID/tile/warp/instruction/stages/swizzle/split-K, algorithm configuration, requested and queried launch resources, preallocated bytes, reset/workspace behavior, gate inventory/digests, raw value bits and both oracle/budget digests. Library executed-kernel name/module capture is a separate instrumented observation; it must not be mixed into ordinary latency.

## Stage 2 boundary

After composed baseline admission, prepare the same residual MLP workload with supported cuBLASLt/CUTLASS bias and bias+ReLU epilogues, retaining exact outputs, shapes, input residency and reference input bits. Keep unfused controls. Audit supported row-major/layout restrictions and epilogue arithmetic order before selecting an implementation; keep residual addition and final power-of-two scaling explicit unless their fusion has a separately reviewed bound. Reuse the existing full tiny and representative checkpoint gates, supplement the budget if the supported arithmetic DAG requires it, and separate compile/autotune/setup from steady requests. A primitive tile win alone cannot establish compiler-fusion advantage.

The local benchmark derivative is implemented in tile_bench_competitive.cu, candidate_config.hpp and componentwise_gate.hpp. The frozen harness, emitted PTX, repository and original compute/timing statements remain unchanged. CPU oracle/checker/parser tests pass; CUDA compilation, can_implement, resources, gate admission, tuning and held-out timing remain pending hardware. Each candidate compiles in isolation through COMPETITIVE_CANDIDATE_INDEX so a failed instantiation is retained explicitly.

This finite initial study covers two M16 families and their bounded correctness shapes. It does not establish a generic Mapal selector. Adding wide M16/N18432/K7168, M128 neighbors, large controls and other algorithms requires an explicit expanded manifest and the same admission rules. No primitive speedup is summed into a composed fusion gain.
