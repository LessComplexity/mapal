# CUTLASS baseline configuration audit

The measured 10–15× gap is real for this one pinned CUTLASS instantiation. The source chooses an untuned SIMT kernel with a 128-row threadblock tile for an M=16 problem. This establishes a concrete mapping mismatch. It does not establish which cuBLAS kernel ran or attribute the whole measured gap to that mismatch.

The independently checked packet has 963 sealed files, pinned CUTLASS commit `91843b71d90737e0a9b86f7d9df213cff5a3526f`, and 154 retrieved runtime files with matching hashes. The actual harness binary is `attempt-03/retrieved/trace/kernel-B/checked`, SHA256 `fc4f1e684dd3cd046f8d49eda3f66f0ed84d628c62ef7c9ea94ace61de602422`. The build receipt records `nvcc -O3 -std=c++17 -arch=sm_120` and no `CUTLASS_ARCH` override; the template architecture tag remains Sm80. The compiled architecture and template tuning tag are distinct facts.

| Configuration | Executed CUTLASS arm | cuBLAS arm |
|---|---|---|
| Storage/accumulation | FP32 A/B/C/D and FP32 accumulator | `cublasSgemm`, float inputs/output/scalars; selected internal implementation unobserved |
| Logical layout | A/B/C/D row major, leading dimensions K/N/N | Same row-major product represented as column-major Cᵀ=BᵀAᵀ; opN/opN, API dimensions N/M/K, leading dimensions N/K/N |
| Operator | `OpClassSimt`, `OpMultiplyAdd`; scalar instruction shape 1×1×1 | `CUBLAS_DEFAULT_MATH`, requested/reported 0; no algorithm selected or recorded |
| Tile | Threadblock 128×128×8; warp 32×64×8 | Unknown |
| Pipeline/access | 2 stages; template A/B alignment 1 element | Unknown |
| Epilogue | `LinearCombination<float,1,float,float>`, alpha 1/beta 0 | alpha 1/beta 0, no bias/activation |
| Grid/splitK | Identity swizzle 1; serial split-K false, slices 1 | Unknown |
| Dispatch | One statically chosen type; `can_implement`, initialize once, then call | Library dispatch through SGEMM; no cuBLASLt selection/autotuning/workspace configuration in harness |
| Resources | Source-derived 256 threads; no runtime library register/shared-memory/local-byte record | No runtime launch/resource record |

The separate `CutlassTf32` type is defined as TensorOp/Sm80, tile 128×128×16, warp 64×64×16, instruction 16×8×8, stages 4, vector-4 epilogue. No attempt 03 B invocation passes `--tf32`; both Exact and Contract cases use the SIMT type. `--contract` changes Mapal/reference arithmetic selection without changing either library type or requested math mode.

Source anchors: [CUTLASS types and default architecture](/private/tmp/mapal-s60-kernel/bcomparison/harness/tile_bench_checked.cu:42), [default SIMT configuration](/private/tmp/mapal-s60-kernel/bcomparison/cutlass/include/cutlass/gemm/device/default_gemm_configuration.h:73), [Gemm default template arguments](/private/tmp/mapal-s60-kernel/bcomparison/cutlass/include/cutlass/gemm/device/gemm.h:188), [splitK argument default](/private/tmp/mapal-s60-kernel/bcomparison/cutlass/include/cutlass/gemm/device/gemm.h:322), [math mode, arguments and dispatch](/private/tmp/mapal-s60-kernel/bcomparison/harness/tile_bench_checked.cu:365).

From those sealed defaults and [identity grid calculation](/private/tmp/mapal-s60-kernel/bcomparison/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:65), M16/N1024 uses a 1×8×1 grid and M16/N2048 a 1×16×1 grid. Only 16/128 rows of each threadblock tile lie in the problem. The retrieved device reports 170 SMs. Thus this source mapping supplies only 8 or 16 CTAs in total, fewer than the physical SM count; this is a source-derived concurrency bound, not measured occupancy or proof that masked rows execute the full arithmetic cost. CUTLASS's [launch site](/private/tmp/mapal-s60-kernel/bcomparison/cutlass/include/cutlass/gemm/device/gemm.h:473) uses that grid and the kernel thread count. The actual retrieved binary's mangled SIMT kernel symbol confirms `MmaPipelined<128,128,8>`, `MmaSimt<32,64,8>` and 256-thread iterator types. It also contains the unused TensorOp kernel, so merely finding a TensorOp symbol in the binary would not prove execution.

The following ranges span the six independently launched timing processes per cell (old/s58/B and their identical-code nulls); each process contributes its upper median of 30 event samples. These are descriptive process ranges, not confidence intervals or a pooled median.

| M,N,K / Mapal validation face | cuBLAS median range ms | CUTLASS median range ms | CUTLASS/cuBLAS range |
|---|---:|---:|---:|
|16,2048,7168 / Contract|0.0532|0.5202–0.5243|9.778–9.855×|
|16,2048,7168 / Exact|0.0532|0.5222–0.5243|9.816–9.855×|
|16,1024,4096 / Contract|0.0205|0.3052–0.3072|14.888–14.985×|
|16,1024,4096 / Exact|0.0205|0.3052|14.888×|

Runtime library records consistently report cuBLAS 120804 and CUDA runtime 12080 with requested default math/reported 0. Timing has three warmups per leg, fixed round-robin Mapal→cuBLAS→CUTLASS, a 256 MiB memset before each start event, and preallocated resident inputs/output. Initialization is outside events; each event brackets the actual library invocation. See [validation/timing implementation](/private/tmp/mapal-s60-kernel/bcomparison/harness/tile_bench_checked.cu:409). The raw timestamps are printed to 0.0001 ms, so ratios above inherit that precision.

Numerical admission differs from the composed benchmark: B explicitly enforces the ordered-fold bit check for Mapal; library outputs must be finite and have their maximum relative error versus f64 reported, but the harness imposes no independent error threshold on CUTLASS/cuBLAS. The passing all-three-variant B gate therefore does not mean those libraries received the composed packet's componentwise budget gate. See [actual checks](/private/tmp/mapal-s60-kernel/bcomparison/harness/tile_bench_checked.cu:428).

`DEFAULT_MATH` is not a scalar-instruction or pedantic guarantee. NVIDIA's matching 12.8 documentation describes precision requirements separately from use of Tensor Cores. Accordingly, the actual cuBLAS arithmetic instruction family remains unknown despite the recorded mode 0. [NVIDIA cuBLAS12.8 math modes](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasmath-t).

What still requires actual kernel evidence: exact cuBLAS executed kernel name and module image; cuBLAS launch count/grid/block/shared memory/registers; selected library decomposition/splitK/epilogue and actual arithmetic; CUTLASS disassembly/resources and confirmation of its executed launch. CUTLASS's compiled image is already embedded in the retrieved checked binary, so static disassembly can be produced without a new timing run. Merely disassembling an arbitrary cuBLAS library symbol cannot identify the path selected for these calls.

This audit does not identify a missing Mapal IR fact. The comparison demonstrates different explicit mapping choices, while the harness presents all implementations the same known dimensions and row-major buffers. To call a performance difference an IR-fact gap requires an identified useful fact, evidence it is unavailable at Mapal's decision point, and evidence the observed library kernel exploits it. To call it a mapping gap requires that fact already be available but the selected launch/arithmetic/storage route differ. Executed-image comparison plus the compiler decision witness is still needed to make that classification. No benchmark, production code, gates or frozen sources were changed.

Evidence: [complete experiment directory](../../../../benches/results/gpu/kernels/cutlass-baseline).
