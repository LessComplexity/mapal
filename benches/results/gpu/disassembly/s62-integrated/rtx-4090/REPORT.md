# S62 integrated vs cuBLAS / cuBLASLt / CUTLASS — RTX 4090 (sm_89): executed code and hardware usage

Box: one RTX 4090 (vast 57310933, driver 580.173.02, CUDA 12.8 toolkit, 128 SMs, sm clock fact 2625 MHz, Ryzen 7 3700X), 2026-10-23T02:50:50Z–02:58:59Z.
Mapal = `s62-integrated` @ 6a0e3a0 (`integrated_bundle.sh`, PTX byte-identical to the integrator's final sm_89 PTX; `raw/ptx-sha256.txt`), base = 4e1102c.
Libraries: cuBLAS 12.8 default FP32 (`CUBLAS_DEFAULT_MATH`, `NVIDIA_TF32_OVERRIDE=0`), stock CUTLASS SIMT 128×128×8 (tile_bench), tuned CUTLASS 16×64×16 / warp 8×32×16 / 2 stages (competitive-cutlass.md `m16-n64-k16`), MLP: cuBLASLt `RELU_BIAS` + `beta·x`, CUTLASS `LinearCombinationRelu` (bias as C, ldc 0) + `beta·x`. Pinned CUTLASS 91843b71.

Value gates before every timing: Mapal arms bit-exact on 256 sampled cells to the CPU fold in their order (reassoc also within the ADR-0040 bound); libraries within γ_K·Σ|ab|; MLP: Mapal stdout == CPU oracle on every arm, libraries' whole y within 1e-3·max(|ref|, rms) of an f64 reference. All passed.
Timing: kernel cells = `tile_bench --splitk` (every arm + three libraries in one process, 256 MiB flush per sample, ABBA), 30 samples × 2 processes, second process reversed. MLP = 20 fresh processes per arm, ABBA order; libraries `mlp_bench` 30 samples × 2 processes.
Executed code: counter-free observer (`capture.cpp` from `library-image-capture-v2.tar.gz`, sha256 48f288f4…): one call per leg inside a `cudaProfilerStart/Stop` region, every launched function joined by name to the module image the driver loaded, `cuobjdump -res-usage` + `-sass` of that image. 43 of 43 region captures finish clean (failed=0, 1 start, 1 stop); the 8 Mapal MLP-program captures have no region (all launches kept, flagged `incomplete-final-flush-or-region` at exit by design). No counters (ncu blocked on vast).

Null variation on this box: byte-identical PTX copy **≤ 2.6 %** per process (cells), **≤ 0.7 %** (MLP `iter ms`); library process-to-process: cuBLAS ≤ 0.1 %, tuned CUTLASS ≤ 2.6 %, stock CUTLASS ≤ 5.9 % (cells); MLP libraries 0.3–5.9 %. A verdict is on-par when the ratio is within max(box null, that library's process-to-process variation).

## Per-cell comparison (kernel ms, mean of two process medians; ratio = Mapal / other, < 1 = Mapal faster)

| cell | face | Mapal | cuBLAS | stock CUTLASS | tuned CUTLASS | vs cuBLAS | vs stock | vs tuned | Mapal base (4e1102c) |
|---|---|---:|---:|---:|---:|---|---|---|---:|
| 16x2048x7168 | Exact | 0.1028 | 0.0952 | 0.4987 | 0.2012 | worse (1.08) | better (0.21) | better (0.51) | 0.5030 |
| 16x2048x7168 | Contract | 0.0988 | 0.0952 | 0.4987 | 0.2012 | worse (1.04) | better (0.20) | better (0.49) | 0.5023 |
| 16x2048x7168 | reassoc | 0.1024 | 0.0952 | 0.4987 | 0.2012 | worse (1.08) | better (0.21) | better (0.51) | — |
| 16x2048x7168 | reassoc+C | 0.0986 | 0.0952 | 0.4987 | 0.2012 | worse (1.04) | better (0.20) | better (0.49) | — |
| 16x1024x4096 | Exact | 0.0689 | 0.0358 | 0.2959 | 0.1167 | worse (1.93) | better (0.23) | better (0.59) | 0.2838 |
| 16x1024x4096 | Contract | 0.0381 | 0.0358 | 0.2959 | 0.1167 | worse (1.06) | better (0.13) | better (0.33) | 0.1321 |
| 16x1024x4096 | reassoc | 0.0391 | 0.0358 | 0.2959 | 0.1167 | worse (1.09) | better (0.13) | better (0.33) | — |
| 16x1024x4096 | reassoc+C | 0.0375 | 0.0358 | 0.2959 | 0.1167 | worse (1.05) | better (0.13) | better (0.32) | — |
| 16x18432x7168 | Exact | 0.6180 | 0.6359 | 0.9758 | 0.6152 | better (0.97) | better (0.63) | on-par (1.00) | 2.7965 |
| 16x18432x7168 | Contract | 0.6185 | 0.6359 | 0.9758 | 0.6152 | better (0.97) | better (0.63) | on-par (1.01) | 2.9872 |
| 16x18432x7168 | reassoc | 0.6237 | 0.6359 | 0.9758 | 0.6152 | on-par (0.98) | better (0.64) | on-par (1.01) | — |
| 16x18432x7168 | reassoc+C | 0.6244 | 0.6359 | 0.9758 | 0.6152 | on-par (0.98) | better (0.64) | on-par (1.01) | — |
| 128x2048x7168 | Exact | 0.2038 | 0.1486 | 0.5175 | 0.2262 | worse (1.37) | better (0.39) | better (0.90) | 0.2038 |
| 128x2048x7168 | Contract | 0.1623 | 0.1486 | 0.5175 | 0.2262 | worse (1.09) | better (0.31) | better (0.72) | 0.1884 |
| 128x2048x7168 | reassoc | 0.1839 | 0.1486 | 0.5175 | 0.2262 | worse (1.24) | better (0.36) | better (0.81) | — |
| 128x2048x7168 | reassoc+C | 0.1618 | 0.1486 | 0.5175 | 0.2262 | worse (1.09) | better (0.31) | better (0.72) | — |
| 4096x4096x4096 | Exact | 3.9806 | 2.4439 | 2.4426 | 6.6422 | worse (1.63) | worse (1.63) | better (0.60) | 3.9527 |
| 4096x4096x4096 | Contract | 2.3055 | 2.4439 | 2.4426 | 6.6422 | better (0.94) | better (0.94) | better (0.35) | 2.3014 |
| 4096x4096x4096 | reassoc | 3.9550 | 2.4439 | 2.4426 | 6.6422 | worse (1.62) | worse (1.62) | better (0.60) | — |
| 4096x4096x4096 | reassoc+C | 2.3007 | 2.4439 | 2.4426 | 6.6422 | better (0.94) | better (0.94) | better (0.35) | — |

## Residual MLP whole programs (M128) vs libraries

Kernels only: Mapal nsys GPU durations (one process, operands L2-resident from the device generators) vs library warm-up calls (median of 3, nsys). Warm iteration: Mapal `iter ms` under `CUDA_MODULE_LOADING=EAGER` (host clock, median of 20 processes) vs library steady cudaEvent time (30 × 2). Cold: shipped Mapal `iter ms` (driver default LAZY) vs each library's first call in a fresh process (host clock, 2 processes; includes the library's own lazy module load). Threshold: 5 % for single-sample classes.

| MLP | face | class | Mapal ms | vs cuBLAS eager | vs cuBLASLt fused | vs stock CUTLASS fused | vs tuned CUTLASS fused |
|---|---|---|---:|---|---|---|---|
| d256 | Exact | kernels only (nsys GPU, L2-warm) | 0.0228 | worse (1.75) | worse (2.09) | better (0.37) | worse (1.40) |
| d256 | Exact | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0296 | worse (1.28) | worse (1.49) | better (0.46) | worse (1.03) |
| d256 | Exact | cold single iteration (Mapal LAZY = shipped / library first call) | 0.0669 | better (0.004) | worse (1.19) | better (0.35) | better (0.68) |
| d256 | Contract | kernels only (nsys GPU, L2-warm) | 0.0159 | worse (1.22) | worse (1.46) | better (0.26) | on-par (0.98) |
| d256 | Contract | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0394 | worse (1.71) | worse (1.98) | better (0.61) | worse (1.37) |
| d256 | Contract | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1899 | better (<0.02) | worse (3.38) | on-par (1.00) | worse (1.94) |
| d1024 | Exact | kernels only (nsys GPU, L2-warm) | 0.0785 | worse (1.96) | worse (2.01) | better (0.35) | worse (1.19) |
| d1024 | Exact | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0822 | worse (1.16) | worse (1.21) | better (0.35) | better (0.85) |
| d1024 | Exact | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1124 | better (<0.02) | better (<0.02) | better (0.37) | better (0.91) |
| d1024 | Contract | kernels only (nsys GPU, L2-warm) | 0.0564 | worse (1.41) | worse (1.44) | better (0.25) | better (0.86) |
| d1024 | Contract | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0768 | worse (1.08) | worse (1.13) | better (0.33) | better (0.79) |
| d1024 | Contract | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1750 | better (<0.02) | better (<0.02) | better (0.58) | worse (1.41) |

| program | face | Mapal on LAZY | Mapal on EAGER | base LAZY | base EAGER | null LAZY | Mapal traced kernel sum (MAPAL_GPU_TRACE) | eager cuBLAS steady / first | cuBLASLt fused steady / first | stock CUTLASS steady / first | tuned CUTLASS steady / first |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| pt_rm_m128_d1024_h2048 | _exact | 0.1124 | 0.0822 | 0.1127 | 0.0823 | 0.1132 | 0.0850 | 0.0708 / 17.3732 | 0.0680 / 11.9779 | 0.2318 / 0.3035 | 0.0968 / 0.1237 |
| pt_rm_m128_d1024_h2048 | _c | 0.1750 | 0.0768 | 0.1031 | 0.0637 | 0.1742 | 0.0645 | 0.0708 / 17.3732 | 0.0680 / 11.9779 | 0.2318 / 0.3035 | 0.0968 / 0.1237 |
| pt_rm_m128_d256_h512 | _exact | 0.0669 | 0.0296 | 0.0665 | 0.0297 | 0.0671 | 0.0340 | 0.0231 / 17.5164 | 0.0198 / 0.0562 | 0.0645 / 0.1908 | 0.0287 / 0.0979 |
| pt_rm_m128_d256_h512 | _c | 0.1899 | 0.0394 | 0.0829 | 0.0249 | 0.1886 | 0.0285 | 0.0231 / 17.5164 | 0.0198 / 0.0562 | 0.0645 / 0.1908 | 0.0287 / 0.0979 |

## Hardware usage

RTX 4090 peak 1008 GB/s DRAM, 82.6 TFLOP/s FP32 (128 SM × 128 lanes × 2 × 2.52 GHz boost), ridge 82 FLOP/B. Bytes = A+B+C (+ 2·S·M·N·4 split workspace when a reduce kernel runs): the lower bound of DRAM traffic after the 256 MiB flush. MLP bytes = x, W1, bias, W2, hidden write+read, residual read, y; MLP operands are L2-resident, so their GB/s is not DRAM. Roof = max(bytes/BW, FLOPs/peak).

| cell | implementation | ms | bytes moved (MB) | GB/s | % DRAM peak | GFLOP/s | % FP32 peak | FLOP/B | roof ms | ms / roof | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16x2048x7168 | cuBLAS | 0.0952 | 63.5 (S=16) | 667 | 66% | 4937 | 6.0% | 7.4 | 0.0630 | 1.51 | memory |
| 16x2048x7168 | stock CUTLASS | 0.4987 | 59.3 | 119 | 12% | 942 | 1.1% | 7.9 | 0.0588 | 8.48 | memory roof, latency/issue-limited |
| 16x2048x7168 | tuned CUTLASS | 0.2012 | 59.3 | 295 | 29% | 2334 | 2.8% | 7.9 | 0.0588 | 3.42 | memory roof, latency/issue-limited |
| 16x2048x7168 | Mapal E | 0.1028 | 59.3 | 577 | 57% | 4570 | 5.5% | 7.9 | 0.0588 | 1.75 | memory |
| 16x2048x7168 | Mapal C | 0.0988 | 59.3 | 600 | 60% | 4755 | 5.8% | 7.9 | 0.0588 | 1.68 | memory |
| 16x2048x7168 | Mapal reassoc | 0.1024 | 59.3 | 579 | 57% | 4585 | 5.6% | 7.9 | 0.0588 | 1.74 | memory |
| 16x2048x7168 | Mapal reassoc+C | 0.0986 | 59.3 | 602 | 60% | 4764 | 5.8% | 7.9 | 0.0588 | 1.68 | memory |
| 16x1024x4096 | cuBLAS | 0.0358 | 18.0 (S=7) | 503 | 50% | 3749 | 4.5% | 7.4 | 0.0179 | 2.00 | memory roof, latency/issue-limited |
| 16x1024x4096 | stock CUTLASS | 0.2959 | 17.1 | 58 | 6% | 454 | 0.5% | 7.8 | 0.0170 | 17.44 | memory roof, latency/issue-limited |
| 16x1024x4096 | tuned CUTLASS | 0.1167 | 17.1 | 147 | 15% | 1150 | 1.4% | 7.8 | 0.0170 | 6.88 | memory roof, latency/issue-limited |
| 16x1024x4096 | Mapal E | 0.0689 | 17.1 | 248 | 25% | 1947 | 2.4% | 7.8 | 0.0170 | 4.06 | memory roof, latency/issue-limited |
| 16x1024x4096 | Mapal C | 0.0381 | 17.1 | 450 | 45% | 3527 | 4.3% | 7.8 | 0.0170 | 2.24 | memory roof, latency/issue-limited |
| 16x1024x4096 | Mapal reassoc | 0.0391 | 19.2 (S=16) | 492 | 49% | 3437 | 4.2% | 7.0 | 0.0190 | 2.05 | memory roof, latency/issue-limited |
| 16x1024x4096 | Mapal reassoc+C | 0.0375 | 17.1 | 456 | 45% | 3579 | 4.3% | 7.8 | 0.0170 | 2.21 | memory roof, latency/issue-limited |
| 16x18432x7168 | cuBLAS | 0.6359 | 530.1 | 834 | 83% | 6649 | 8.1% | 8.0 | 0.5259 | 1.21 | memory |
| 16x18432x7168 | stock CUTLASS | 0.9758 | 530.1 | 543 | 54% | 4332 | 5.2% | 8.0 | 0.5259 | 1.86 | memory |
| 16x18432x7168 | tuned CUTLASS | 0.6152 | 530.1 | 862 | 85% | 6872 | 8.3% | 8.0 | 0.5259 | 1.17 | memory |
| 16x18432x7168 | Mapal E | 0.6180 | 530.1 | 858 | 85% | 6841 | 8.3% | 8.0 | 0.5259 | 1.18 | memory |
| 16x18432x7168 | Mapal C | 0.6185 | 530.1 | 857 | 85% | 6836 | 8.3% | 8.0 | 0.5259 | 1.18 | memory |
| 16x18432x7168 | Mapal reassoc | 0.6237 | 534.8 (S=2) | 858 | 85% | 6779 | 8.2% | 7.9 | 0.5306 | 1.18 | memory |
| 16x18432x7168 | Mapal reassoc+C | 0.6244 | 534.8 (S=2) | 856 | 85% | 6771 | 8.2% | 7.9 | 0.5306 | 1.18 | memory |
| 128x2048x7168 | cuBLAS | 0.1486 | 63.4 | 427 | 42% | 25299 | 30.6% | 59.2 | 0.0629 | 2.36 | memory roof, latency/issue-limited |
| 128x2048x7168 | stock CUTLASS | 0.5175 | 63.4 | 123 | 12% | 7261 | 8.8% | 59.2 | 0.0629 | 8.22 | memory roof, latency/issue-limited |
| 128x2048x7168 | tuned CUTLASS | 0.2262 | 63.4 | 280 | 28% | 16614 | 20.1% | 59.2 | 0.0629 | 3.59 | memory roof, latency/issue-limited |
| 128x2048x7168 | Mapal E | 0.2038 | 63.4 | 311 | 31% | 18440 | 22.3% | 59.2 | 0.0629 | 3.24 | memory roof, latency/issue-limited |
| 128x2048x7168 | Mapal C | 0.1623 | 63.4 | 391 | 39% | 23162 | 28.0% | 59.2 | 0.0629 | 2.58 | memory roof, latency/issue-limited |
| 128x2048x7168 | Mapal reassoc | 0.1839 | 71.8 (S=4) | 391 | 39% | 20441 | 24.8% | 52.3 | 0.0713 | 2.58 | memory roof, latency/issue-limited |
| 128x2048x7168 | Mapal reassoc+C | 0.1618 | 63.4 | 392 | 39% | 23227 | 28.1% | 59.2 | 0.0629 | 2.57 | memory roof, latency/issue-limited |
| 4096x4096x4096 | cuBLAS | 2.4439 | 201.3 | 82 | 8% | 56238 | 68.1% | 682.7 | 1.6644 | 1.47 | compute |
| 4096x4096x4096 | stock CUTLASS | 2.4426 | 201.3 | 82 | 8% | 56267 | 68.1% | 682.7 | 1.6644 | 1.47 | compute |
| 4096x4096x4096 | tuned CUTLASS | 6.6422 | 201.3 | 30 | 3% | 20692 | 25.1% | 682.7 | 1.6644 | 3.99 | compute roof, latency/issue-limited |
| 4096x4096x4096 | Mapal E | 3.9806 | 201.3 | 51 | 5% | 34528 | 41.8% | 682.7 | 1.6644 | 2.39 | compute roof, latency/issue-limited |
| 4096x4096x4096 | Mapal C | 2.3055 | 201.3 | 87 | 9% | 59614 | 72.2% | 682.7 | 1.6644 | 1.39 | compute |
| 4096x4096x4096 | Mapal reassoc | 3.9550 | 201.3 | 51 | 5% | 34751 | 42.1% | 682.7 | 1.6644 | 2.38 | compute roof, latency/issue-limited |
| 4096x4096x4096 | Mapal reassoc+C | 2.3007 | 201.3 | 88 | 9% | 59737 | 72.3% | 682.7 | 1.6644 | 1.38 | compute |
| MLP d256 | Mapal E (nsys, 2 kernels) | 0.0228 | 1.97 | 86 | 9% | 2943 | 3.6% | 34 | 0.0020 | 11.7 | memory roof, latency/launch-limited |
| MLP d256 | Mapal C (nsys, 2 kernels) | 0.0159 | 1.97 | 124 | 12% | 4221 | 5.1% | 34 | 0.0020 | 8.1 | memory roof, latency/launch-limited |
| MLP d256 | cuBLAS eager (6 kernels, warm) | 0.0130 | 1.97 | 151 | 15% | 5162 | 6.3% | 34 | 0.0020 | 6.7 | memory roof, latency/launch-limited |
| MLP d256 | cuBLASLt fused (4 kernels, warm) | 0.0109 | 1.97 | 181 | 18% | 6157 | 7.5% | 34 | 0.0020 | 5.6 | memory roof, latency/launch-limited |
| MLP d256 | stock CUTLASS fused (2, warm) | 0.0618 | 1.97 | 32 | 3% | 1086 | 1.3% | 34 | 0.0020 | 31.7 | memory roof, latency/launch-limited |
| MLP d256 | tuned CUTLASS fused (2, warm) | 0.0163 | 1.97 | 121 | 12% | 4117 | 5.0% | 34 | 0.0020 | 8.3 | memory roof, latency/launch-limited |
| MLP d1024 | Mapal E (nsys, 2 kernels) | 0.0785 | 20.46 | 261 | 26% | 13678 | 16.6% | 52 | 0.0203 | 3.9 | memory roof, latency/launch-limited |
| MLP d1024 | Mapal C (nsys, 2 kernels) | 0.0564 | 20.46 | 363 | 36% | 19038 | 23.1% | 52 | 0.0203 | 2.8 | memory roof, latency/launch-limited |
| MLP d1024 | cuBLAS eager (5 kernels, warm) | 0.0401 | 20.46 | 510 | 51% | 26777 | 32.4% | 52 | 0.0203 | 2.0 | memory roof, latency/launch-limited |
| MLP d1024 | cuBLASLt fused (4 kernels, warm) | 0.0391 | 20.46 | 523 | 52% | 27461 | 33.3% | 52 | 0.0203 | 1.9 | memory roof, latency/launch-limited |
| MLP d1024 | stock CUTLASS fused (2, warm) | 0.2252 | 20.46 | 91 | 9% | 4768 | 5.8% | 52 | 0.0203 | 11.1 | memory roof, latency/launch-limited |
| MLP d1024 | tuned CUTLASS fused (2, warm) | 0.0658 | 20.46 | 311 | 31% | 16318 | 19.8% | 52 | 0.0203 | 3.2 | memory roof, latency/launch-limited |

## Executed kernels and resources

Observer: the functions launched in the profiler region, joined to the loaded image; registers/local/static smem from `cuobjdump -res-usage` of that image, dynamic smem from the launch. Occupancy = theoretical blocks/SM from regs (256-register warp granularity), threads, smem (+1 KB reserved), 24-block cap; waves = blocks / (blocks/SM × 128).

| cell | implementation | kernel | grid (blocks) | block | regs | local | static smem | dyn smem | blocks/SM (limit) | warps/SM of 48 | waves (128 SMs) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 16x2048x7168 | cuBLAS | `ampere_sgemm_128x32_nn` | 16,1,16 (256) | 256 | 57 | 0 | 16384 | 0 | 4 (regs) | 32 | 0.50 |
| 16x2048x7168 | cuBLAS | `_ZN8cublasLt19splitKreduce_kernelILi32ELi16EiffffLb0EfffL...` | 64,1,1 (64) | 512 | 44 | 0 | 0 | 0 | 2 (regs) | 32 | 0.25 |
| 16x2048x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,16,1 (16) | 256 | 122 | 0 | 0 | 16640 | 2 (regs) | 16 | 0.06 |
| 16x2048x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,32,1 (32) | 128 | 56 | 0 | 0 | 10496 | 8 (smem) | 32 | 0.03 |
| 16x2048x7168 | Mapal E | `mapal_tile_6444915cf30a6889` | 64,2,1 (128) | 64 | 254 | 0 | 0 | 45056 | 2 (smem) | 4 | 0.50 |
| 16x2048x7168 | Mapal C | `mapal_tile_6e4b9bcdd0843cf4` | 64,2,1 (128) | 64 | 251 | 0 | 0 | 45056 | 2 (smem) | 4 | 0.50 |
| 16x2048x7168 | Mapal reassoc | `mapal_tile_4142009e76f88ee2` | 64,2,1 (128) | 64 | 254 | 0 | 0 | 45056 | 2 (smem) | 4 | 0.50 |
| 16x2048x7168 | Mapal reassoc+C | `mapal_tile_6575df28aa8a2321` | 64,2,1 (128) | 64 | 251 | 0 | 0 | 45056 | 2 (smem) | 4 | 0.50 |
| 16x1024x4096 | cuBLAS | `ampere_sgemm_32x32_sliced1x4_nn` | 32,1,7 (224) | 128 | 86 | 0 | 32768 | 0 | 3 (smem) | 12 | 0.58 |
| 16x1024x4096 | cuBLAS | `_ZN8cublasLt19splitKreduce_kernelILi32ELi16EiffffLb0EfffL...` | 32,1,1 (32) | 512 | 44 | 0 | 0 | 0 | 2 (regs) | 32 | 0.12 |
| 16x1024x4096 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,8,1 (8) | 256 | 122 | 0 | 0 | 16640 | 2 (regs) | 16 | 0.03 |
| 16x1024x4096 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,16,1 (16) | 128 | 56 | 0 | 0 | 10496 | 8 (smem) | 32 | 0.02 |
| 16x1024x4096 | Mapal E | `mapal_tile_d4baff99a3f45267` | 64,2,1 (128) | 32 | 52 | 0 | 0 | 4608 | 18 (smem) | 18 | 0.06 |
| 16x1024x4096 | Mapal C | `mapal_tile_163c7f5ccf8b5e2a` | 64,2,1 (128) | 32 | 128 | 0 | 0 | 98304 | 1 (smem) | 1 | 1.00 |
| 16x1024x4096 | Mapal reassoc | `mapal_tile_0a11ebba9da4cbd1` | 64,32,1 (2048) | 32 | 54 | 0 | 0 | 4608 | 18 (smem) | 18 | 0.89 |
| 16x1024x4096 | Mapal reassoc | `mapal_tile_c991024d1c2da42e` | 64,1,1 (64) | 256 | 38 | 0 | 0 | 0 | 6 (regs) | 48 | 0.08 |
| 16x1024x4096 | Mapal reassoc+C | `mapal_tile_fb0b34a207952c9b` | 64,2,1 (128) | 32 | 128 | 0 | 0 | 98304 | 1 (smem) | 1 | 1.00 |
| 16x18432x7168 | cuBLAS | `ampere_sgemm_128x32_sliced1x4_nn` | 144,1,5 (720) | 256 | 134 | 0 | 41984 | 0 | 1 (regs) | 8 | 5.62 |
| 16x18432x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,144,1 (144) | 256 | 122 | 0 | 0 | 16640 | 2 (regs) | 16 | 0.56 |
| 16x18432x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,288,1 (288) | 128 | 56 | 0 | 0 | 10496 | 8 (smem) | 32 | 0.28 |
| 16x18432x7168 | Mapal E | `mapal_tile_8a786a6bb40482bb` | 576,1,1 (576) | 64 | 66 | 0 | 0 | 3328 | 14 (regs) | 28 | 0.32 |
| 16x18432x7168 | Mapal C | `mapal_tile_0ca772afc3771626` | 576,1,1 (576) | 64 | 77 | 0 | 0 | 3328 | 12 (regs) | 24 | 0.38 |
| 16x18432x7168 | Mapal reassoc | `mapal_tile_bf4b396fc6689bb5` | 576,2,1 (1152) | 64 | 66 | 0 | 0 | 3328 | 14 (regs) | 28 | 0.64 |
| 16x18432x7168 | Mapal reassoc | `mapal_tile_9cfa6fdf8bc25e32` | 1152,1,1 (1152) | 256 | 12 | 0 | 0 | 0 | 6 (threads) | 48 | 1.50 |
| 16x18432x7168 | Mapal reassoc+C | `mapal_tile_1361775047c6af14` | 576,2,1 (1152) | 64 | 76 | 0 | 0 | 3328 | 12 (regs) | 24 | 0.75 |
| 16x18432x7168 | Mapal reassoc+C | `mapal_tile_d022ad5feed23e63` | 1152,1,1 (1152) | 256 | 12 | 0 | 0 | 0 | 6 (threads) | 48 | 1.50 |
| 128x2048x7168 | cuBLAS | `_ZN7cutlass7Kernel2I42cutlass_80_simt_sgemm_128x64_8x5_nn...` | 16,2,23 (736) | 128 | 126 | 0 | 0 | 30720 | 3 (smem) | 12 | 1.92 |
| 128x2048x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 1,16,1 (16) | 256 | 122 | 0 | 0 | 16640 | 2 (regs) | 16 | 0.06 |
| 128x2048x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 8,32,1 (256) | 128 | 56 | 0 | 0 | 10496 | 8 (smem) | 32 | 0.25 |
| 128x2048x7168 | Mapal E | `mapal_tile_a964c4a6897c62eb` | 64,2,1 (128) | 128 | 62 | 0 | 0 | 19968 | 4 (smem) | 16 | 0.25 |
| 128x2048x7168 | Mapal C | `mapal_tile_d83f12ee616a9cb6` | 64,2,1 (128) | 128 | 80 | 0 | 0 | 79872 | 1 (smem) | 4 | 1.00 |
| 128x2048x7168 | Mapal reassoc | `mapal_tile_9efc30f1debe4ae5` | 64,8,1 (512) | 128 | 64 | 0 | 0 | 19968 | 4 (smem) | 16 | 1.00 |
| 128x2048x7168 | Mapal reassoc | `mapal_tile_807a9b57e2664102` | 1024,1,1 (1024) | 256 | 12 | 0 | 0 | 0 | 6 (threads) | 48 | 1.33 |
| 128x2048x7168 | Mapal reassoc+C | `mapal_tile_d2dee8607c2f26c7` | 64,2,1 (128) | 128 | 80 | 0 | 0 | 79872 | 1 (smem) | 4 | 1.00 |
| 4096x4096x4096 | cuBLAS | `_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_n...` | 256,2,3 (1536) | 256 | 202 | 0 | 0 | 49152 | 1 (regs) | 8 | 12.00 |
| 4096x4096x4096 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 32,32,1 (1024) | 256 | 122 | 0 | 0 | 16640 | 2 (regs) | 16 | 4.00 |
| 4096x4096x4096 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock...` | 256,64,1 (16384) | 128 | 56 | 0 | 0 | 10496 | 8 (smem) | 32 | 16.00 |
| 4096x4096x4096 | Mapal E | `mapal_tile_2dd01bc8b8404cf3` | 32,32,1 (1024) | 128 | 255 | 0 | 0 | 16640 | 2 (regs) | 8 | 4.00 |
| 4096x4096x4096 | Mapal C | `mapal_tile_ef9b0c1ddb04347e` | 32,32,1 (1024) | 128 | 218 | 0 | 0 | 16640 | 2 (regs) | 8 | 4.00 |
| 4096x4096x4096 | Mapal reassoc | `mapal_tile_6f5769e03e0cb858` | 32,32,1 (1024) | 128 | 255 | 0 | 0 | 16640 | 2 (regs) | 8 | 4.00 |
| 4096x4096x4096 | Mapal reassoc+C | `mapal_tile_617d16a5df3b6ebf` | 32,32,1 (1024) | 128 | 218 | 0 | 0 | 16640 | 2 (regs) | 8 | 4.00 |

## SASS instruction mix (static counts of the executed images; `(…)` = the hot loop, the backward branch with the most FP ops)

Outputs/thread = M·N·S / (blocks × threads). Accum regs = distinct `d` of `FFMA d,a,b,d` in the loop (library tiles; checks against the tile names). k-steps/trip = loop MACs / accumulators. Static counts: no dynamic or stall claim.

| cell | implementation | kernel | instr | FFMA (loop) | FMUL+FADD (loop) | LDG 32/64/128 (loop) | LDGSTS 32/64/128 (loop) | LDS 32/64/128 (loop) | STS (loop) | BAR (loop) | branch+pred (loop) | int+mov (loop) | FP ops/global load (loop) | FP ops/global byte (loop) | outputs/thread (useful) | accum regs | k-steps/trip |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| 16x2048x7168 | cuBLAS | `ampere_sgemm_128x32_nn` | 624 | 132 (128) | 16 (0) | 26/4/0 (5/0/0) | 0/0/0 (0/0/0) | 4/0/22 (0/0/16) | 12 (5) | 7 (1) | 105 (5) | 269 (9) | 4.93 (25.6) | 1.088 (6.4) | 8.0 | 16 | 8 |
| 16x2048x7168 | cuBLAS | `_ZN8cublasLt19splitKreduce_kernelILi3...` | 352 | 8 (—) | 49 (—) | 38/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 67 (—) | 169 (—) | 1.5 (—) | 0.375 (—) | 1.0 | — | — |
| 16x2048x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 2352 | 576 (512) | 128 (0) | 84/2/0 (8/0/0) | 0/0/0 (0/0/0) | 128/0/36 (0/0/32) | 144 (8) | 34 (1) | 299 (10) | 782 (58) | 8.19 (64.0) | 2.0 (16.0) | 8.0 | 64 | 8 |
| 16x2048x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 776 | 136 (128) | 16 (0) | 32/2/0 (10/0/0) | 0/0/0 (0/0/0) | 16/17/17 (0/16/16) | 36 (10) | 10 (1) | 92 (12) | 377 (67) | 4.47 (12.8) | 1.056 (3.2) | 8.0 | 8 | 16 |
| 16x2048x7168 | Mapal E | `mapal_tile_6444915cf30a6889` | 2832 | 0 (0) | 2048 (2048) | 0/0/80 (0/0/40) | 0/0/0 (0/0/0) | 256/0/256 (256/0/256) | 64 (64) | 2 (2) | 6 (4) | 107 (39) | 25.6 (51.2) | 1.6 (3.2) | 4.0 | = outputs | 256 |
| 16x2048x7168 | Mapal C | `mapal_tile_6e4b9bcdd0843cf4` | 1808 | 1024 (1024) | 0 (0) | 0/0/80 (0/0/40) | 0/0/0 (0/0/0) | 256/0/256 (256/0/256) | 64 (64) | 2 (2) | 6 (4) | 107 (39) | 12.8 (25.6) | 0.8 (1.6) | 4.0 | = outputs | 256 |
| 16x2048x7168 | Mapal reassoc | `mapal_tile_4142009e76f88ee2` | 2832 | 0 (0) | 2048 (2048) | 0/0/80 (0/0/40) | 0/0/0 (0/0/0) | 256/0/256 (256/0/256) | 64 (64) | 2 (2) | 6 (4) | 107 (39) | 25.6 (51.2) | 1.6 (3.2) | 4.0 | = outputs | 256 |
| 16x2048x7168 | Mapal reassoc+C | `mapal_tile_6575df28aa8a2321` | 1808 | 1024 (1024) | 0 (0) | 0/0/80 (0/0/40) | 0/0/0 (0/0/0) | 256/0/256 (256/0/256) | 64 (64) | 2 (2) | 6 (4) | 107 (39) | 12.8 (25.6) | 0.8 (1.6) | 4.0 | = outputs | 256 |
| 16x1024x4096 | cuBLAS | `ampere_sgemm_32x32_sliced1x4_nn` | 904 | 257 (256) | 35 (0) | 37/4/0 (16/0/0) | 0/0/0 (0/0/0) | 4/0/31 (0/0/24) | 34 (16) | 7 (1) | 130 (6) | 333 (37) | 7.12 (16.0) | 1.622 (4.0) | 4.0 | 32 | 8 |
| 16x1024x4096 | cuBLAS | `_ZN8cublasLt19splitKreduce_kernelILi3...` | 352 | 8 (—) | 49 (—) | 38/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 67 (—) | 169 (—) | 1.5 (—) | 0.375 (—) | 1.0 | — | — |
| 16x1024x4096 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 2352 | 576 (512) | 128 (0) | 84/2/0 (8/0/0) | 0/0/0 (0/0/0) | 128/0/36 (0/0/32) | 144 (8) | 34 (1) | 299 (10) | 782 (58) | 8.19 (64.0) | 2.0 (16.0) | 8.0 | 64 | 8 |
| 16x1024x4096 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 776 | 136 (128) | 16 (0) | 32/2/0 (10/0/0) | 0/0/0 (0/0/0) | 16/17/17 (0/16/16) | 36 (10) | 10 (1) | 92 (12) | 377 (67) | 4.47 (12.8) | 1.056 (3.2) | 8.0 | 8 | 16 |
| 16x1024x4096 | Mapal E | `mapal_tile_d4baff99a3f45267` | 288 | 0 (0) | 128 (128) | 0/0/0 (0/0/0) | 12/0/6 (4/0/2) | 19/0/16 (19/0/16) | 0 (0) | 5 (3) | 6 (4) | 81 (34) | 7.11 (21.33) | 0.889 (2.667) | 4.0 | = outputs | 16 |
| 16x1024x4096 | Mapal C | `mapal_tile_163c7f5ccf8b5e2a` | 4048 | 2048 (2048) | 0 (0) | 0/0/0 (0/0/0) | 256/0/128 (128/0/64) | 515/0/512 (515/0/512) | 0 (0) | 4 (3) | 8 (6) | 568 (224) | 5.33 (10.67) | 0.667 (1.333) | 4.0 | = outputs | 512 |
| 16x1024x4096 | Mapal reassoc | `mapal_tile_0a11ebba9da4cbd1` | 288 | 0 (0) | 128 (128) | 0/0/0 (0/0/0) | 12/0/6 (4/0/2) | 19/0/16 (19/0/16) | 0 (0) | 5 (3) | 6 (4) | 87 (34) | 7.11 (21.33) | 0.889 (2.667) | 4.0 | = outputs | 16 |
| 16x1024x4096 | Mapal reassoc | `mapal_tile_c991024d1c2da42e` | 56 | 0 (—) | 15 (—) | 16/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 4 (—) | 9 (—) | 0.94 (—) | 0.234 (—) | 16.0 | — | — |
| 16x1024x4096 | Mapal reassoc+C | `mapal_tile_fb0b34a207952c9b` | 4048 | 2048 (2048) | 0 (0) | 0/0/0 (0/0/0) | 256/0/128 (128/0/64) | 515/0/512 (515/0/512) | 0 (0) | 4 (3) | 8 (6) | 568 (224) | 5.33 (10.67) | 0.667 (1.333) | 4.0 | = outputs | 512 |
| 16x18432x7168 | cuBLAS | `ampere_sgemm_128x32_sliced1x4_nn` | 1232 | 514 (512) | 70 (0) | 47/4/0 (20/0/0) | 0/0/0 (0/0/0) | 8/0/44 (0/0/32) | 43 (20) | 7 (1) | 156 (7) | 311 (22) | 11.45 (25.6) | 2.655 (6.4) | 8.0 | 64 | 8 |
| 16x18432x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 2352 | 576 (512) | 128 (0) | 84/2/0 (8/0/0) | 0/0/0 (0/0/0) | 128/0/36 (0/0/32) | 144 (8) | 34 (1) | 299 (10) | 782 (58) | 8.19 (64.0) | 2.0 (16.0) | 8.0 | 64 | 8 |
| 16x18432x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 776 | 136 (128) | 16 (0) | 32/2/0 (10/0/0) | 0/0/0 (0/0/0) | 16/17/17 (0/16/16) | 36 (10) | 10 (1) | 92 (12) | 377 (67) | 4.47 (12.8) | 1.056 (3.2) | 8.0 | 8 | 16 |
| 16x18432x7168 | Mapal E | `mapal_tile_8a786a6bb40482bb` | 376 | 0 (0) | 256 (256) | 0/0/6 (0/0/3) | 0/0/0 (0/0/0) | 0/16/16 (0/16/16) | 6 (6) | 2 (2) | 6 (4) | 55 (8) | 42.67 (85.33) | 2.667 (5.333) | 8.0 | = outputs | 16 |
| 16x18432x7168 | Mapal C | `mapal_tile_0ca772afc3771626` | 248 | 128 (128) | 0 (0) | 0/0/6 (0/0/3) | 0/0/0 (0/0/0) | 0/16/16 (0/16/16) | 6 (6) | 2 (2) | 5 (3) | 55 (8) | 21.33 (42.67) | 1.333 (2.667) | 8.0 | = outputs | 16 |
| 16x18432x7168 | Mapal reassoc | `mapal_tile_bf4b396fc6689bb5` | 376 | 0 (0) | 256 (256) | 0/0/6 (0/0/3) | 0/0/0 (0/0/0) | 0/16/16 (0/16/16) | 6 (6) | 2 (2) | 6 (4) | 57 (8) | 42.67 (85.33) | 2.667 (5.333) | 8.0 | = outputs | 16 |
| 16x18432x7168 | Mapal reassoc | `mapal_tile_9cfa6fdf8bc25e32` | 32 | 0 (—) | 1 (—) | 2/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 4 (—) | 9 (—) | 0.5 (—) | 0.125 (—) | 2.0 | — | — |
| 16x18432x7168 | Mapal reassoc+C | `mapal_tile_1361775047c6af14` | 248 | 128 (128) | 0 (0) | 0/0/6 (0/0/3) | 0/0/0 (0/0/0) | 0/16/16 (0/16/16) | 6 (6) | 2 (2) | 5 (3) | 57 (8) | 21.33 (42.67) | 1.333 (2.667) | 8.0 | = outputs | 16 |
| 16x18432x7168 | Mapal reassoc+C | `mapal_tile_d022ad5feed23e63` | 32 | 0 (—) | 1 (—) | 2/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 4 (—) | 9 (—) | 0.5 (—) | 0.125 (—) | 2.0 | — | — |
| 128x2048x7168 | cuBLAS | `_ZN7cutlass7Kernel2I42cutlass_80_simt...` | 2528 | 576 (512) | 128 (0) | 68/4/0 (0/0/0) | 60/0/0 (12/0/0) | 132/0/36 (0/0/32) | 128 (0) | 50 (3) | 391 (17) | 680 (47) | 5.33 (42.67) | 1.294 (10.667) | 64.0 | 64 | 8 |
| 128x2048x7168 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 2352 | 576 (512) | 128 (0) | 84/2/0 (8/0/0) | 0/0/0 (0/0/0) | 128/0/36 (0/0/32) | 144 (8) | 34 (1) | 299 (10) | 782 (58) | 8.19 (64.0) | 2.0 (16.0) | 64.0 | 64 | 8 |
| 128x2048x7168 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 776 | 136 (128) | 16 (0) | 32/2/0 (10/0/0) | 0/0/0 (0/0/0) | 16/17/17 (0/16/16) | 36 (10) | 10 (1) | 92 (12) | 377 (67) | 4.47 (12.8) | 1.056 (3.2) | 8.0 | 8 | 16 |
| 128x2048x7168 | Mapal E | `mapal_tile_a964c4a6897c62eb` | 688 | 0 (0) | 512 (512) | 0/0/0 (0/0/0) | 24/0/3 (8/0/1) | 3/0/32 (3/0/32) | 0 (0) | 5 (3) | 7 (5) | 85 (29) | 18.96 (56.89) | 3.556 (10.667) | 16.0 | = outputs | 16 |
| 128x2048x7168 | Mapal C | `mapal_tile_d83f12ee616a9cb6` | 1440 | 1024 (1024) | 0 (0) | 0/0/0 (0/0/0) | 96/0/12 (32/0/4) | 3/0/128 (3/0/128) | 0 (0) | 5 (3) | 9 (7) | 144 (50) | 9.48 (28.44) | 1.778 (5.333) | 16.0 | = outputs | 64 |
| 128x2048x7168 | Mapal reassoc | `mapal_tile_9efc30f1debe4ae5` | 704 | 0 (0) | 512 (512) | 0/0/0 (0/0/0) | 24/0/3 (8/0/1) | 3/0/32 (3/0/32) | 0 (0) | 5 (3) | 7 (5) | 100 (32) | 18.96 (56.89) | 3.556 (10.667) | 16.0 | = outputs | 16 |
| 128x2048x7168 | Mapal reassoc | `mapal_tile_807a9b57e2664102` | 32 | 0 (—) | 3 (—) | 4/0/0 (—) | 0/0/0 (—) | 0/0/0 (—) | 0 (—) | 0 (—) | 4 (—) | 9 (—) | 0.75 (—) | 0.188 (—) | 4.0 | — | — |
| 128x2048x7168 | Mapal reassoc+C | `mapal_tile_d2dee8607c2f26c7` | 1440 | 1024 (1024) | 0 (0) | 0/0/0 (0/0/0) | 96/0/12 (32/0/4) | 3/0/128 (3/0/128) | 0 (0) | 5 (3) | 9 (7) | 144 (50) | 9.48 (28.44) | 1.778 (5.333) | 16.0 | = outputs | 64 |
| 4096x4096x4096 | cuBLAS | `_ZN7cutlass7Kernel2I43cutlass_80_simt...` | 4496 | 1152 (1024) | 256 (0) | 132/4/0 (0/0/0) | 48/0/0 (12/0/0) | 260/0/54 (0/0/48) | 256 (0) | 82 (3) | 755 (17) | 960 (37) | 7.65 (85.33) | 1.872 (21.333) | 128.0 | 128 | 8 |
| 4096x4096x4096 | stock CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 2352 | 576 (512) | 128 (0) | 84/2/0 (8/0/0) | 0/0/0 (0/0/0) | 128/0/36 (0/0/32) | 144 (8) | 34 (1) | 299 (10) | 782 (58) | 8.19 (64.0) | 2.0 (16.0) | 64.0 | 64 | 8 |
| 4096x4096x4096 | tuned CUTLASS | `_ZN7cutlass6KernelINS_4gemm6kernel4Ge...` | 776 | 136 (128) | 16 (0) | 32/2/0 (10/0/0) | 0/0/0 (0/0/0) | 16/17/17 (0/16/16) | 36 (10) | 10 (1) | 92 (12) | 377 (67) | 4.47 (12.8) | 1.056 (3.2) | 8.0 | 8 | 16 |
| 4096x4096x4096 | Mapal E | `mapal_tile_2dd01bc8b8404cf3` | 4392 | 0 (0) | 4096 (4096) | 0/0/16 (0/0/8) | 0/0/0 (0/0/0) | 0/0/96 (0/0/96) | 20 (20) | 2 (2) | 6 (4) | 116 (7) | 256.0 (512.0) | 16.0 (32.0) | 128.0 | = outputs | 16 |
| 4096x4096x4096 | Mapal C | `mapal_tile_ef9b0c1ddb04347e` | 2336 | 2048 (2048) | 0 (0) | 0/0/16 (0/0/8) | 0/0/0 (0/0/0) | 0/0/96 (0/0/96) | 20 (20) | 2 (2) | 6 (4) | 108 (7) | 128.0 (256.0) | 8.0 (16.0) | 128.0 | = outputs | 16 |
| 4096x4096x4096 | Mapal reassoc | `mapal_tile_6f5769e03e0cb858` | 4392 | 0 (0) | 4096 (4096) | 0/0/16 (0/0/8) | 0/0/0 (0/0/0) | 0/0/96 (0/0/96) | 20 (20) | 2 (2) | 6 (4) | 116 (7) | 256.0 (512.0) | 16.0 (32.0) | 128.0 | = outputs | 16 |
| 4096x4096x4096 | Mapal reassoc+C | `mapal_tile_617d16a5df3b6ebf` | 2336 | 2048 (2048) | 0 (0) | 0/0/16 (0/0/8) | 0/0/0 (0/0/0) | 0/0/96 (0/0/96) | 20 (20) | 2 (2) | 6 (4) | 108 (7) | 128.0 (256.0) | 8.0 (16.0) | 128.0 | = outputs | 16 |

## Residual MLP hotspots (nsys, one process per row; span = end of the operand-generation sync to the end of the result sync, what `iter ms` clocks)

| process | iter ms (this nsys run) | span µs | host gap before 1st tile launch µs | launch API µs (n) | gaps between tile launches µs | final sync µs | tile kernels GPU µs (n) | kernel share of span | outside span: module load / alloc / free ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pt_rm_m128_d256_h512.LAZY | 0.0781 | 92.5 | 41.8 | 11.9 (2) | 21.8 | 16.6 | 22.8 (2) | 25% | 0.65 / 0.26 / 0.12 |
| pt_rm_m128_d256_h512.EAGER | 0.0336 | 48.2 | 16.9 | 12.0 (2) | 1.6 | 17.4 | 22.8 (2) | 47% | 0.76 / 0.27 / 0.11 |
| pt_rm_m128_d256_h512.base.LAZY | 0.0758 | 90.9 | 40.6 | 11.6 (2) | 21.7 | 16.5 | 22.8 (2) | 25% | 0.65 / 0.29 / 0.11 |
| pt_rm_m128_d256_h512_c.LAZY | 0.1942 | 210.5 | 110.0 | 12.7 (2) | 76.4 | 10.8 | 15.9 (2) | 8% | 3.02 / 0.29 / 0.12 |
| pt_rm_m128_d256_h512_c.EAGER | 0.0458 | 55.8 | 29.7 | 11.9 (2) | 2.4 | 11.3 | 16.4 (2) | 29% | 3.20 / 0.29 / 0.11 |
| pt_rm_m128_d256_h512_c.base.LAZY | 0.1041 | 127.2 | 63.8 | 20.9 (2) | 29.2 | 12.8 | 17.6 (2) | 14% | 0.64 / 0.28 / 0.11 |
| pt_rm_m128_d1024_h2048.LAZY | 0.1281 | 138.6 | 43.3 | 12.2 (2) | 32.7 | 49.8 | 78.5 (2) | 57% | 0.86 / 0.27 / 0.13 |
| pt_rm_m128_d1024_h2048.EAGER | 0.0895 | 106.0 | 19.2 | 11.8 (2) | 1.6 | 72.9 | 77.9 (2) | 73% | 0.95 / 0.27 / 0.13 |
| pt_rm_m128_d1024_h2048.base.LAZY | 0.1239 | 138.8 | 47.1 | 12.3 (2) | 29.0 | 49.8 | 78.9 (2) | 57% | 0.84 / 0.27 / 0.14 |
| pt_rm_m128_d1024_h2048_c.LAZY | 0.1848 | 207.6 | 85.7 | 12.7 (2) | 75.8 | 32.7 | 56.4 (2) | 27% | 2.32 / 0.26 / 0.14 |
| pt_rm_m128_d1024_h2048_c.EAGER | 0.0848 | 100.5 | 34.8 | 11.9 (2) | 2.2 | 51.1 | 57.1 (2) | 57% | 2.44 / 0.29 / 0.13 |
| pt_rm_m128_d1024_h2048_c.base.LAZY | 0.1142 | 128.8 | 54.1 | 12.5 (2) | 27.1 | 34.6 | 59.6 (2) | 46% | 0.76 / 0.27 / 0.13 |

Driver calls inside the span (observer, enter order): integrated Contract `cuModuleGetFunction, cuFuncSetAttribute, cuLaunchKernel` per launch; base Contract `cuModuleGetFunction, cuLaunchKernel` (`raw/obs.tar.xz`, `mlp.*/events.tsv`).

| MLP kernel (Contract) | grid × block | regs | dyn smem | SASS lines | GPU µs (nsys) |
|---|---|---:|---:|---:|---:|
| d256 GEMM1+bias+ReLU, integrated | 32×4 × 32 | 111 | 57 344 | 5 877 | 6.8 |
| d256 GEMM2+residual, integrated | 16×4 × 32 | 113 | 86 016 | 6 258 | 9.2 |
| d256 GEMM1+bias+ReLU, base | 32×4 × 32 | 72 | 5 376 | 901 | 6.3 |
| d256 GEMM2+residual, base | 16×4 × 32 | 72 | 5 376 | 802 | 11.3 |
| d1024 GEMM1+bias+ReLU, integrated | 64×2 × 128 | 80 | 79 872 | 3 138 | 25.4 |
| d1024 GEMM2+residual, integrated | 32×4 × 64 | 127 | 73 728 | 6 037 | 31.1 |
| d1024 GEMM1+bias+ReLU, base | 64×2 × 128 | 65 | 19 968 | 1 234 | 26.6 |
| d1024 GEMM2+residual, base | 32×4 × 64 | 75 | 9 216 | 1 109 | 33.0 |

| library leg (nsys, mlp_bench) | d256 launches | d256 kernels µs warm / flushed | d1024 launches | d1024 kernels µs warm / flushed |
|---|---:|---|---:|---|
| cuBLAS eager (sgemm, bias+ReLU, sgemm, add) | 6 | 13.0 / 19.2 | 5 | 40.1 / 66.1 |
| cuBLASLt fused | 4 | 10.9 / 16.8 | 4 | 39.1 / 63.6 |
| stock CUTLASS fused | 2 | 61.8 / 63.9 | 2 | 225.2 / 229.4 |
| tuned CUTLASS fused | 2 | 16.3 / 27.7 | 2 | 65.8 / 93.1 |

cuBLASLt's "fused" epilogue on sm_89 FP32 is not in the GEMM main loop: d256 runs `ampere_sgemm_64x32_sliced1x4` (8×4×4, split-K 4) + `splitKreduce` with bias+ReLU, then `ampere_sgemm_32x32_sliced1x4` (8×4×4) + `splitKreduce` with β·x; d1024 runs `ampere_sgemm_32x32_sliced1x4` (64×4) + `cublasLt::epilogue::impl::globalKernel` (bias+ReLU pass), then `ampere_sgemm_128x32` (8×4×8, split-K 8) + `splitKreduce` (`raw/obs-kernels.tsv`).

**Geometry regression (1.35–2.3× at equal kernel ms): reproduces, mechanism measured.** Contract d256 integrated/base `iter ms` 0.1899/0.0829 = 2.29× (LAZY, the driver default), d1024 0.1750/0.1031 = 1.70×; the integrated kernels are faster on the GPU (15.9 vs 17.6 µs, 56.4 vs 59.6 µs). With `CUDA_MODULE_LOADING=EAGER` the same binaries take 0.0394/0.0249 (1.58×) and 0.0768/0.0637 (1.21×): 0.150 ms of the d256 integrated span is the driver loading the two functions on their first `cuModuleGetFunction` inside the timed span (host gaps 110 + 76 µs with no other CUDA call), and the integrated kernels' images are 6.5–7.8× larger (deep unrolled panels). The remaining EAGER gap is host time before/between launches (29.7 vs 16.9 µs before the first launch) with a `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)` per launch (57–86 KB > 48 KB) that the base kernels do not need. Exact d256/d1024: integrated text = base text; on/base 1.00.

## Gaps (every implementation slower than Mapal is listed under the per-cell table; these are the cells where Mapal is slower)

| # | cell | vs | ratio (Mapal/other) | SASS / usage difference that explains it | fact that would remove it |
|---|---|---|---|---|---|
| 1 | 16×1024×4096 Exact | cuBLAS | 1.93 | Mapal E runs the K16-panel tile: 128 blocks × 1 warp, 4.6 KB smem (3 stages × 1.5 KB), per 16-k trip 6 LDGSTS (4×4 B, 2×16 B), 35 LDS, 3 BAR for 64 MACs → ≤ 4.6 KB in flight per SM, 248 GB/s (25 %). cuBLAS: `ampere_sgemm_32x32_sliced1x4` 224 blocks × 4 warps (split-K 7) + reduce, 32 KB smem, 503 GB/s. Same-box Mapal Contract on the same grid uses 2 × 48 KB K512 panels (192 LDGSTS, 2048 FFMA per trip): 450 GB/s, 1.06× cuBLAS. Static: `emit --gpu-on=grid-share` gives Exact the same 98 304-byte launch; the row `rtx-4090 grid-share Under/Exact` (profile.rs:1143, cite from the pre-integration s62 v2 text) keeps it off. Exact-with-grid-share was not timed on this box. | The cell is memory-bound (7.8 FLOP/B, ridge 82): the face changes FP ops per MAC, not bytes. Panel depth/stages from `tile_plan` panel bytes × resident blocks vs the part's bandwidth × latency per SM (TargetProfile), not a fill × face row. |
| 2 | 16×1024×4096 Exact | Mapal Contract (same box) | 1.81 | as #1 | as #1 |
| 3 | 128×2048×7168 Exact | cuBLAS | 1.37 | Exact = FMUL+FADD (2 issue slots per MAC, ordered rounding); at 59 FLOP/B the cell sits at the ridge, so doubling FP issue moves it onto the compute roof. Mapal E: 128 blocks × 4 warps, 62 regs, 19.5 KB, 1 block/SM by grid, 512 FP ops / 9 LDGSTS / 35 LDS / 3 BAR per 16-k trip, 16 outputs/thread. cuBLAS `cutlass_80_simt_sgemm_128x64_8x5`: 736 blocks (split-K 23, serial in-kernel), 3 blocks/SM = 12 warps/SM, 5-stage cp.async, 64 outputs/thread. | Face semantics (Exact forbids FFMA); grid fill from N×M only because ordered faces cannot split K. |
| 4 | 128×2048×7168 Contract (and reassoc+C) | cuBLAS | 1.09 | Mapal C: 128 blocks × 4 warps, 80 regs, 78 KB smem → 1 block/SM (smem-bound), 1 warp per scheduler; per 64-k trip 36 LDGSTS (32 × 4 B for A, 4 × 16 B for B), 131 LDS, 3 BAR, 1024 FFMA, 16 outputs/thread. 23.2 TFLOP/s (28 %), 391 GB/s. cuBLAS: 12 warps/SM, 64 outputs/thread, 25.3 TFLOP/s (31 %), 427 GB/s. | Warps per scheduler / outputs per thread from TargetProfile (4 schedulers, regs/SM) against the `tile_plan` panel smem that limits residency to 1 block. |
| 5 | 128×2048×7168 reassoc | cuBLAS | 1.24 | split S=4 (512 blocks × 4 warps + 1024-block reduce, +8.4 MB workspace); process medians 0.2039 / 0.1638 ms (24 % apart, bimodal; min 0.1628 both). | — (not stable on this box) |
| 6 | 16×2048×7168 Exact / Contract | cuBLAS | 1.08 / 1.04 | Mapal: 128 blocks × 2 warps, 254/251 regs; per K256 trip 40 LDG.128 into registers, then 64 STS, 44 KB smem, 64 STS + 512 LDS + 2 BAR per 1024 FFMA; 1 block per SM → 2 warps/SM. M=16 is split into two 8-row blocks, so every B element is fetched by 2 blocks (117 MB B traffic from L2 vs 58.7 MB). cuBLAS `ampere_sgemm_128x32_nn`: 256 blocks × 8 warps (split-K 16) + reduce, 16 warps/SM, 8 of 16 accumulators useful (32-row tile on M=16), B fetched once: 667 vs 600 GB/s. | B's reuse count is M (`tile_plan` rows): a block tile with all 16 rows halves B's L2→SM traffic; fill from N (2048/BN blocks) instead of M. |
| 7 | 16×1024×4096 Contract / reassoc+C / reassoc | cuBLAS | 1.06 / 1.05 / 1.09 | C and reassoc+C: 1 warp per SM (128 × 32-thread blocks, 96 KB smem → 1 block/SM) vs cuBLAS 224 × 4-warp blocks, 450–456 vs 503 GB/s. reassoc: S=16 split of the K16 tile (2048 × 1-warp blocks, 18 blocks/SM) + 64-block reduce, +2.1 MB workspace, 492 GB/s. | Same as #1 (bytes in flight are there; warps per SM are not). |
| 8 | 4096³ Exact / reassoc | cuBLAS, stock CUTLASS | 1.63 / 1.62 | Exact FMUL+FADD: 4096 FP instructions per 16-k trip vs 2048 FFMA; Mapal E 3.98 ms = 84 % of the Exact roof (2 × 1.66 ms), 255 regs, 8 warps/SM; Contract on the same tile is 0.94× cuBLAS. | Face semantics, not a realization gap. |
| 9 | MLP d256 / d1024 Contract, kernels only | cuBLASLt | 1.46 / 1.44 | d256 GEMM2: 64 blocks × 1 warp (half the SMs idle), GEMM1 128 × 1 warp, 16 outputs/thread, 57–86 KB smem; d1024 GEMM2 128 × 2 warps (127 regs). cuBLASLt splits K 4–8 (128–256 blocks × 4–8 warps) and pays an extra reduce/epilogue kernel. Mapal fuses bias+ReLU / residual into the tile kernels (2 launches vs 4). | Ordered faces cannot split K; the reassoc face (ADR-0040) is not applied to the MLP sites. Fill for a K=512/2048, M·N=32 768 site needs either split-K under reassoc or smaller output tiles (`tile_plan`, Fill vs SM count). |
| 10 | MLP d256 / d1024 Exact, kernels only | cuBLASLt | 2.09 / 2.01 | as #9 plus FMUL+FADD; integrated Exact = base text (32-thread blocks, 64 regs, 5.4 KB / 20 KB smem). | as #9 |
| 11 | MLP d256 Contract, cold single iteration (shipped) | cuBLASLt first call | 3.38 | Lazy function load of ≈ 2 900–3 100-instruction images on the first `cuModuleGetFunction` inside the timed span (186 µs of a 210 µs span), plus `cuFuncSetAttribute` per launch. EAGER: 0.0394 ms. | Launch sites, kernel names and their dynamic smem are known at emit time (host IR launch list): resolve every function, set its smem attribute and force-load it once in init (the `eager-init` thread already exists), and cache the handle instead of `cuModuleGetFunction` per launch (mapal_gpu.c:226). |
| 12 | MLP d256 / d1024 Contract, warm iteration | cuBLASLt steady | 1.98 / 1.13 | EAGER host time around the kernels (17–35 µs before the first launch, ~12 µs launch API for 2 launches) vs library steady events (host-side API costs overlap the GPU queue there). | as #11 for the per-launch calls. |
| 13 | MLP d256 Exact, cold | cuBLASLt first call | 1.19 | 41.8 + 21.8 µs of lazy load gaps in a 92.5 µs span (base = integrated). | as #11 |
| 14 | MLP Exact kernels only d256 / d1024; Exact warm d256 | tuned CUTLASS | 1.40 / 1.19; 1.03 | FMUL+FADD on the base 1-warp tiles (Contract kernels are 0.98 / 0.86). | Face semantics + #9 fill. |
| 15 | MLP Contract warm d256; cold d256 / d1024 | tuned CUTLASS | 1.37; 1.94 / 1.41 | #11 / #12 host costs (tuned CUTLASS first call 0.098 / 0.124 ms: statically linked sm_89 SASS, no lazy JIT image). | as #11 |
| 16 | MLP kernels only E d256/d1024, C d256/d1024; warm E 1.28/1.16, C 1.71/1.08 | cuBLAS eager (4 calls) | 1.75 / 1.96, 1.22 / 1.41 | same kernel geometry as #9; cuBLAS eager splits K 4–8 and adds 2 elementwise kernels (5–6 launches). | as #9 / #11 |

Mapal is faster than or on-par with: cuBLAS at 16×18432×7168 (0.97–0.98) and 4096³ Contract (0.94); stock CUTLASS everywhere except 4096³ Exact; tuned CUTLASS in every kernel cell (on-par at 16×18432×7168; 0.32–0.90 elsewhere); the MLP rows are #14–#15. Stock CUTLASS (128-row tile) uses 8 of 64 accumulators at M=16 and launches 8–144 blocks; tuned CUTLASS (16×64) launches 16 384 blocks at 4096³ (8 outputs/thread, 6.64 ms).

## Evidence

- `raw/results-rtx-4090.txt` — every number above (box log, written as produced); `raw/tables.md` regenerates with `tar xJf raw/obs.tar.xz -C /tmp/t && python3 tools/analyze.py raw/results-rtx-4090.txt /tmp/t` (needs `tools/mix.py`).
- `raw/obs-kernels.tsv`, `raw/obs.tar.xz` — observer events, launched-function SASS, res-usage per capture; `raw/static-sass.tar.xz` — ptxas (12.8) SASS/res-usage of every Mapal PTX and the harness binaries; `raw/ptxas/`; `raw/ptx-mix.json`.
- `raw/nsys.tar.xz` — `nsys stats` CSV (cells: grids/regs/smem; MLP: API + GPU traces); `raw/hotspots.md` (`tools/hotspots.py`), `raw/libkern.txt` (`tools/libkern.py`), `raw/mlp-trace.tar.xz` (MAPAL_GPU_TRACE).
- Full trace with module cubins and `.nsys-rep`: `.benchmark-artifacts/gpu/library-images/counter-free/rtx-4090/2026-10-23-s62-integrated/trace-rtx-4090.tgz`, 37 547 866 bytes, sha256 cc036eccf53ae2b540a6d08cf3c20ad9d306153b9476f7638d65405c6125f1ce (not in git).
- `tools/` — the box script (`box_disasm.sh`, run as the bundle's `box_run.sh` via `vast_part.sh`), `tile_bench_s62d.cu` (tile_bench + tuned CUTLASS leg, γ_K library gate, `PROFILE_LEG`), `mlp_bench.cu`, `capture.cpp` (v2, unchanged), `obs_extract.py`. On this image CUPTI lives in `targets/x86_64-linux/lib`; `capture.so` was built by hand on the box during the MLP section (the committed script carries that path).
- `raw/instances.txt` — 57309646 (upload failed) and 57310933 created and destroyed with `-y`; both absent at 02:59:13Z; pre-existing 57065574 57100404 57126023 57133432 57285249 untouched.
