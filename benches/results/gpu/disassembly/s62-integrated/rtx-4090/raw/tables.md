Box null variation (byte-identical PTX copy, per process, max over cells and E/C): **2.6%**; per cell: 16x2048x7168 0.7%, 16x1024x4096 2.6%, 16x18432x7168 0.2%, 128x2048x7168 0.6%, 4096x4096x4096 0.2%.
Library process-to-process variation: max 5.9%.

### Per-cell comparison (kernel ms, mean of two process medians; ratio = Mapal / other, < 1 = Mapal faster; on-par = within max(box null, that library leg's process-to-process variation))

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

### Executed kernels and resources (observer: launched in the profiler region, joined to the saved image; res-usage of that image)

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

### SASS instruction mix (static; executed images; hot loop = backward branch with the most FFMA)

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

### Hardware usage (peak: 1008 GB/s DRAM, 82.6 TFLOP/s FP32 at 2.52 GHz boost; ridge 82 FLOP/B)

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

### Residual MLP whole programs (M128; Mapal `iter ms` = host clock around the computation in a fresh process, median of the reps; library steady = cudaEvent GPU ms, mean of two process medians; library first = host clock around the first call + sync in a fresh process)

| program | face | Mapal on LAZY | Mapal on EAGER | base LAZY | base EAGER | null LAZY | Mapal traced kernel sum | eager cuBLAS steady / first | cuBLASLt fused steady / first | stock CUTLASS steady / first | tuned CUTLASS steady / first |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| pt_rm_m128_d1024_h2048 | _exact | 0.1124 | 0.0822 | 0.1127 | 0.0823 | 0.1132 | 0.0850 | 0.0708 / 17.3732 | 0.0680 / 11.9779 | 0.2318 / 0.3035 | 0.0968 / 0.1237 |
| pt_rm_m128_d1024_h2048 | _c | 0.1750 | 0.0768 | 0.1031 | 0.0637 | 0.1742 | 0.0645 | 0.0708 / 17.3732 | 0.0680 / 11.9779 | 0.2318 / 0.3035 | 0.0968 / 0.1237 |
| pt_rm_m128_d256_h512 | _exact | 0.0669 | 0.0296 | 0.0665 | 0.0297 | 0.0671 | 0.0340 | 0.0231 / 17.5164 | 0.0198 / 0.0562 | 0.0645 / 0.1908 | 0.0287 / 0.0979 |
| pt_rm_m128_d256_h512 | _c | 0.1899 | 0.0394 | 0.0829 | 0.0249 | 0.1886 | 0.0285 | 0.0231 / 17.5164 | 0.0198 / 0.0562 | 0.0645 / 0.1908 | 0.0287 / 0.0979 |

