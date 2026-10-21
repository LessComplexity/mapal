# S53 GPU optimizations — integrated results

Branch `s53-gpu-opt` · RTX 4090 (sm_89), H100 PCIe (sm_90a), RTX 5090 (sm_120), rented on vast.ai ·
results `benches/results-s53-gpu/integrated-*.txt` (tabulate: `python3 benches/gpu/table.py <file>`) ·
per-item results `benches/results-s53-gpu/<item>-*.txt` · log `docs/sessions/2026-10-21-s53-gpu-optimizations.md`

Kernel rows: 30 reps interleaved with cuBLAS and CUTLASS in one `tile_bench` process, medians, L2 flushed. Exact and
contract rows are bit-exact against the in-order fold; TF32 rows are within 4× cuBLAS-TF32's error. CUTLASS is the
sm80 kernel family (the harness builds for plain sm_90 on H100).

## 1. Headline: shipped Mapal vs cuBLAS / CUTLASS, n = 2048 / 4096 / 8192 (medians; ~ = min–max ranges overlap)

| part | face | vs cuBLAS | vs CUTLASS | S52 kernel vs cuBLAS (same run) |
|---|---|---|---|---|
| rtx-4090 | exact | 0.66 / 0.61 / 0.60 | 0.64 / 0.61 / 0.60 | 0.64 / 0.59 / 0.56 |
| rtx-4090 | contract | 1.04~ / 1.05~ / 1.05~ | 1.02~ / 1.05~ / 1.02~ | 0.93 / 0.87 / 0.84 |
| rtx-4090 | TF32 | 0.89 / 0.91 / 0.94~ | 0.90 / 0.93 / 0.95~ | 0.89 / 0.91 / 0.93 |
| h100-pcie | exact | 0.48 / 0.59 / 0.53 | 0.63 / 0.66 / 0.66 | 0.44 / 0.44 / 0.41 |
| h100-pcie | contract | 0.85 / 1.03~ / 0.95~ | 1.11 / 1.14 / 1.12~ | 0.73 / 0.88 / 0.80 |
| h100-pcie | TF32 | 0.62 / 0.65 / 0.56 | 0.96 / 1.08~ / 1.19 | 0.50 / 0.42 / 0.34 |
| rtx-5090 | exact | 0.54 / 0.58 / 0.58 | 0.62 / 0.64 / 0.67 | 0.54 / 0.58 / 0.57 |
| rtx-5090 | contract | 0.87 / 0.91~ / 0.92~ | 0.99~ / 1.00~ / 1.01~ | 0.80 / 0.84 / 0.81 |
| rtx-5090 | TF32 | 0.85 / 0.90 / 0.84 | 0.88 / 0.90 / 0.91~ | 0.85 / 0.90 / 0.84 |

## 2. Every size and face

### rtx-4090 — `integrated-rtx-4090.txt`; contract 2048–8192 is the `on-unroll` arm (byte-identical to the shipped kernel)

| n | face | Mapal ms | cuBLAS ms | CUTLASS ms | vs cuBLAS | vs CUTLASS | vs S52 | values | geometry |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1024 | contract | 0.070 | 0.053 | 0.089 | 0.76 | 1.28 | 1.00~ | yes | 16x8×256 r91 |
| 1024 | exact | 0.082 | 0.055 | 0.083 | 0.68 | 1.01~ | 1.15 | yes | 16x8×256 r84 |
| 1024 | TF32 | 0.050 | 0.039 | 0.061 | 0.77 | 1.22 | 1.65 | yes | 16x16×128 r102 |
| 1536 | contract | 0.252 | 0.199 | 0.236 | 0.79 | 0.94 |  | yes | 12x12×256 r113 |
| 1536 | exact | 0.414 | 0.199 | 0.236 | 0.48 | 0.57 |  | yes | 12x12×256 r128 spill16 |
| 1536 | TF32 | 0.159 | 0.137 | 0.173 | 0.86 | 1.09~ |  | yes | 12x24×128 r128 |
| 2048 | contract | 0.320 | 0.332 | 0.325 | 1.04~ | 1.02~ | 1.12 | yes | 16x16×128 r218 |
| 2048 | exact | 0.505 | 0.331 | 0.325 | 0.66 | 0.64 | 1.03~ | yes | 16x16×128 r255 |
| 2048 | TF32 | 0.254 | 0.226 | 0.229 | 0.89 | 0.90 | 1.00~ | yes | 16x16×128 r198 |
| 3072 | contract | 1.300 | 1.101 | 1.182 | 0.85 | 0.91 |  | yes | 24x24×128 r206 |
| 3072 | exact | 2.164 | 1.095 | 1.177 | 0.51 | 0.54 |  | yes | 24x24×128 r255 |
| 3072 | TF32 | 0.815 | 0.746 | 0.751 | 0.91 | 0.92 |  | yes | 24x24×128 r198 |
| 4096 | contract | 2.521 | 2.658 | 2.657 | 1.05~ | 1.05~ | 1.18 | yes | 32x32×128 r218 |
| 4096 | exact | 4.254 | 2.608 | 2.608 | 0.61 | 0.61 | 1.05~ | yes | 32x32×128 r255 |
| 4096 | TF32 | 1.717 | 1.571 | 1.595 | 0.91 | 0.93 | 1.00~ | yes | 32x32×128 r198 |
| 8192 | contract | 23.572 | 24.665 | 24.055 | 1.05~ | 1.02~ | 1.19~ | yes | 64x64×128 r218 |
| 8192 | exact | 36.833 | 21.990 | 22.160 | 0.60 | 0.60 | 1.05~ | yes | 64x64×128 r255 |
| 8192 | TF32 | 14.248 | 13.363 | 13.537 | 0.94~ | 0.95~ | 1.00~ | yes | 64x64×128 r198 |

### h100-pcie — `integrated-h100-pcie-confirm.txt`; contract 2048–8192 is the `on-affine` arm (byte-identical to the shipped kernel)

| n | face | Mapal ms | cuBLAS ms | CUTLASS ms | vs cuBLAS | vs CUTLASS | vs S52 | values | geometry |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1024 | contract | 0.137 | 0.089 | 0.124 | 0.65 | 0.90 | 1.07 | yes | 16x8×256 r73 |
| 1024 | exact | 0.195 | 0.090 | 0.124 | 0.46 | 0.63 | 1.23 | yes | 16x8×256 r106 |
| 1024 | TF32 | 0.043 | 0.025 | 0.034 | 0.57 | 0.80 | 1.11 | yes | 16x8×128 r147 |
| 1536 | contract | 0.324 | 0.213 | 0.355 | 0.66 | 1.10 |  | yes | 12x12×256 r128 |
| 1536 | exact | 0.569 | 0.213 | 0.355 | 0.37 | 0.63 |  | yes | 12x12×256 r128 |
| 1536 | TF32 | 0.099 | 0.054 | 0.085 | 0.54 | 0.85 |  | yes | 12x24×128 r164 |
| 2048 | contract | 0.622 | 0.527 | 0.690 | 0.85 | 1.11 | 1.16 | yes | 16x16×256 r127 |
| 2048 | exact | 1.089 | 0.524 | 0.687 | 0.48 | 0.63 | 1.12 | yes | 16x16×256 r128 |
| 2048 | TF32 | 0.171 | 0.106 | 0.164 | 0.62 | 0.96 | 1.24 | yes | 16x8×256 r212 |
| 3072 | contract | 1.859 | 1.798 | 2.118 | 0.97~ | 1.14 |  | yes | 24x24×256 r128 |
| 3072 | exact | 3.214 | 1.780 | 2.102 | 0.55 | 0.65 |  | yes | 24x24×256 r128 |
| 3072 | TF32 | 0.369 | 0.284 | 0.447 | 0.77 | 1.21 |  | yes | 24x12×256 r213 |
| 4096 | contract | 3.734 | 3.855 | 4.253 | 1.03~ | 1.14 | 1.16 | yes | 32x32×256 r127 |
| 4096 | exact | 6.281 | 3.694 | 4.162 | 0.59 | 0.66 | 1.33 | yes | 32x32×256 r128 |
| 4096 | TF32 | 0.904 | 0.591 | 0.972 | 0.65 | 1.08~ | 1.57 | yes | 32x16×256 r212 |
| 8192 | contract | 36.665 | 34.739 | 41.056 | 0.95~ | 1.12~ | 1.19~ | yes | 64x64×256 r127 |
| 8192 | exact | 56.248 | 29.567 | 37.033 | 0.53 | 0.66 | 1.37 | yes | 64x64×256 r128 |
| 8192 | TF32 | 7.850 | 4.385 | 9.374 | 0.56 | 1.19 | 1.64 | yes | 64x32×256 r212 |

### rtx-5090 — `integrated-rtx-5090-confirm.txt`

| n | face | Mapal ms | cuBLAS ms | CUTLASS ms | vs cuBLAS | vs CUTLASS | vs S52 | values | geometry |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1024 | contract | 0.057 | 0.045 | 0.092 | 0.79 | 1.61 | 1.29 | yes | 16x16×256 r79 |
| 1024 | exact | 0.068 | 0.045 | 0.092 | 0.67 | 1.36 | 1.12 | yes | 16x16×256 r77 |
| 1024 | TF32 | 0.039 | 0.031 | 0.051 | 0.79 | 1.32 | 1.89 | yes | 16x16×256 r66 |
| 1536 | contract | 0.145 | 0.119 | 0.133 | 0.82 | 0.92 |  | yes | 24x12×256 r95 |
| 1536 | exact | 0.209 | 0.119 | 0.133 | 0.57 | 0.64 |  | yes | 24x12×256 r101 |
| 1536 | TF32 | 0.092 | 0.078 | 0.078 | 0.84 | 0.84 |  | yes | 12x24×256 r80 |
| 2048 | contract | 0.305 | 0.264 | 0.301 | 0.87 | 0.99~ | 1.08 | yes | 16x16×256 r128 |
| 2048 | exact | 0.487 | 0.264 | 0.301 | 0.54 | 0.62 | 1.00~ | yes | 16x16×256 r128 spill8 |
| 2048 | TF32 | 0.223 | 0.190 | 0.197 | 0.85 | 0.88 | 1.00~ | yes | 16x16×128 r192 |
| 3072 | contract | 1.167 | 0.950 | 1.073 | 0.81 | 0.92~ |  | yes | 24x24×256 r128 spill40 |
| 3072 | exact | 1.655 | 0.942 | 1.061 | 0.57 | 0.64 |  | yes | 24x24×256 r128 spill8 |
| 3072 | TF32 | 0.647 | 0.578 | 0.580 | 0.89 | 0.90 |  | yes | 24x24×128 r192 |
| 4096 | contract | 2.487 | 2.258 | 2.485 | 0.91~ | 1.00~ | 1.08~ | yes | 32x32×256 r128 |
| 4096 | exact | 3.849 | 2.241 | 2.473 | 0.58 | 0.64 | 1.00~ | yes | 32x32×256 r128 spill8 |
| 4096 | TF32 | 1.489 | 1.345 | 1.341 | 0.90 | 0.90 | 1.00~ | yes | 32x32×128 r192 |
| 8192 | contract | 21.943 | 20.284 | 22.177 | 0.92~ | 1.01~ | 1.11~ | yes | 64x64×256 r128 |
| 8192 | exact | 33.111 | 19.133 | 22.143 | 0.58 | 0.67 | 1.03~ | yes | 64x64×256 r128 spill8 |
| 8192 | TF32 | 12.387 | 10.413 | 11.227 | 0.84 | 0.91~ | 1.00~ | yes | 64x64×128 r192 |

## 3. Rungs

### 3a. Each rung alone (its item branch, ON vs OFF in one box session; > 1 = ON faster; ~ = ranges overlap)

| rung | part | face | effect alone | file |
|---|---|---|---|---|
| clock-fence | all | e2e | not a speedup: fenced `iter ms` = the site's trace sum (+0.05…+0.25 ms); S52's read 1.2–28 ms, the generation | clock-fence-*.txt |
| arch-rungs (table) | 4090 | contract | 4096: 1.07 (disjoint, 3 sessions); 2048/8192 1.02–1.11~ | arch-rungs-rtx-4090*.txt |
| arch-rungs (table) | H100 PCIe | exact | 1.24 / 1.09 / 1.29 / 1.21 at 1024…8192 (disjoint, 2 sessions) | arch-rungs-h100-pcie*.txt |
| arch-rungs (table) | H100 PCIe | contract | 1.20 / 1.07 / 1.07 at 1024/2048/4096 (disjoint) | arch-rungs-h100-pcie*.txt |
| arch-rungs (table) | 5090 | contract | 1.02 / 1.02 / 1.02~ at 2048/4096/8192 (6/6 positive, all overlap) | arch-rungs-rtx-5090*.txt |
| warp-fill | 4090 | TF32 / exact | 1024: 1.64 / 1.11; 1536 TF32 1.27 (disjoint) | warp-fill-rtx-4090.txt |
| warp-fill | H100 PCIe | exact / TF32 | 1024 exact 1.13; 1536 TF32 1.12 (disjoint) | warp-fill-h100-pcie.txt |
| warp-fill | 5090 | TF32 / contract / exact | 1024: 1.50 / 1.16 / 1.03~; 1536: 1.29 / 1.10 / 1.19; **2048 TF32 0.973 (disjoint)** | warp-fill-rtx-5090.txt |
| fma-regs (4 rungs) | 4090 | contract, register staging | 1.118 / 1.111 / 1.088 at 2048/4096/8192 (disjoint) = 1.04–1.07× cuBLAS | fma-regs-rtx-4090.txt |
| fma-regs (4 rungs) | 4090 | exact, register staging | 1.065 / 1.058 / 1.048; register staging beats cp.async by 1.03–1.05 with it | fma-regs-rtx-4090.txt |
| fma-regs: idx32+affine+pad | 5090 | contract | 1.053 / 1.040 / 1.037; second session 1.066 / 1.050 / 1.080 | fma-regs-rtx-5090*.txt |
| fma-regs: idx32+affine+pad | H100 PCIe | contract | 1.032 / 1.036 / 0.993 (one session) | fma-regs-h100-pcie.txt |
| wgmma-hopper | H100 PCIe | TF32 | 1.10 / 1.24 / 1.53 / 1.64 at 1024…8192 (disjoint) | wgmma-hopper-h100-pcie.txt |
| resident | 4090 / H100 / 5090 | e2e_4096 exact, wall | 1.11 / 1.12 / 1.14 (5090 host B 1.06); GPU phase sums 3.3× / 2.7× / 7.5× | resident-*.txt |
| resident | 4090 / H100 / 5090 | e2e_chain_4096 exact, wall | 1.18 / 1.16 / 1.19 (disjoint) | resident-*.txt |
| devgen | 4090 / H100 | e2e_4096 exact, wall | 1.10 / 1.11 (disjoint) | resident-*.txt |
| retain-delta | 4090 / H100 / 5090 | decode 1024×256, wall | exact 1.57 / 1.53 / 1.63; contract 1.65 / 1.50 / 1.63 (disjoint) | retain-delta-*.txt |
| eager-init | 4090 / H100 / 5090 | e2e_4096 exact, wall | 1.48 & 1.31 / 1.37 & 1.40 / 1.38 & 1.15~ (two sessions each) | host-runtime*-*.txt |

### 3b. Each rung in the integrated build: its arm against the session's shipped kernel (ON/OFF, > 1 = ON faster; ~ = ranges overlap)

Sessions: 4090 `integrated-rtx-4090.txt`, H100 `integrated-h100-pcie-confirm.txt`, 5090 `integrated-rtx-5090-confirm.txt`. `shipped` is that session's; the 4090 contract `unroll` row and the H100 contract `affine` row are the two flips made after it (§3c).

| part | face | rung | shipped | n=1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---:|---:|---:|---:|
| rtx-4090 | contract | affine | OFF | 1.000~ |  |  |  |
| rtx-4090 | contract | affine | ON |  | 0.991~ | 0.987~ | 1.004~ |
| rtx-4090 | contract | arch-rungs | ON | 1.000~ | 0.973 | 0.938~ | 0.966~ |
| rtx-4090 | contract | async | OFF |  | 0.982 | 0.978~ | 0.963~ |
| rtx-4090 | contract | async | ON | 1.132 |  |  |  |
| rtx-4090 | contract | fma-regs | OFF | 1.000~ |  |  |  |
| rtx-4090 | contract | fma-regs | ON |  | 0.994~ | 0.944~ | 0.943~ |
| rtx-4090 | contract | idx32 | OFF | 1.000~ |  |  |  |
| rtx-4090 | contract | idx32 | ON |  | 0.967 | 0.977~ | 0.994~ |
| rtx-4090 | contract | interleave | ON | 1.000~ | 1.548 | 1.429 | 1.492 |
| rtx-4090 | contract | min-blocks | ON | 1.089 | 0.991~ | 0.990~ | 0.996~ |
| rtx-4090 | contract | pad | ON |  | 0.985 | 0.981~ | 1.014~ |
| rtx-4090 | contract | prefetch | ON |  | 1.047 | 1.015~ | 1.001~ |
| rtx-4090 | contract | unroll | OFF |  | 1.083 | 1.166 | 1.086~ |
| rtx-4090 | contract | unroll | ON | 1.059 |  |  |  |
| rtx-4090 | contract | warp | ON | 1.000~ |  |  |  |
| rtx-4090 | contract | warp-fill | ON | 1.000~ |  |  |  |
| rtx-4090 | contract | wide-fma | ON |  | 0.991~ | 0.998~ | 0.996~ |
| rtx-4090 | exact | affine | ON | 1.012~ | 1.057 | 1.018~ | 1.012~ |
| rtx-4090 | exact | arch-rungs | ON |  | 1.032~ | 1.044~ | 1.050~ |
| rtx-4090 | exact | async | OFF |  | 0.969~ | 0.957~ | 0.934~ |
| rtx-4090 | exact | async | ON | 1.138 |  |  |  |
| rtx-4090 | exact | fma-regs | ON | 1.050 | 1.075 | 1.042~ | 1.041~ |
| rtx-4090 | exact | idx32 | ON | 1.050 | 1.037~ | 1.022~ | 1.003~ |
| rtx-4090 | exact | interleave | ON | 1.039~ | 1.122 | 1.100~ | 1.104~ |
| rtx-4090 | exact | min-blocks | ON | 1.062 | 1.061 | 1.035~ | 1.023~ |
| rtx-4090 | exact | pad | ON |  | 1.037 | 1.042~ | 1.057~ |
| rtx-4090 | exact | prefetch | ON |  | 1.726 | 1.544 | 1.577 |
| rtx-4090 | exact | unroll | ON | 1.112 | 1.055 | 1.078~ | 1.077~ |
| rtx-4090 | exact | warp | ON | 1.050~ |  |  |  |
| rtx-4090 | exact | warp-fill | ON | 1.089 |  |  |  |
| rtx-4090 | exact | wide-fma | ON |  | 1.077 | 1.074~ | 1.066~ |
| rtx-4090 | TF32 | arch-rungs | ON | 1.000~ | 1.000~ | 1.001~ | 0.995~ |
| rtx-4090 | TF32 | ldmatrix | OFF | 1.000~ | 1.000~ | 0.999~ | 1.007~ |
| rtx-4090 | TF32 | warp-fill | ON | 1.651 |  |  |  |
| rtx-4090 | TF32 | wide | ON | 1.020~ | 1.044 | 1.042 | 1.044~ |
| h100-pcie | contract | affine | OFF | 0.996~ | 1.035 | 1.027~ | 1.020~ |
| h100-pcie | contract | arch-rungs | ON | 1.101 | 1.394 | 1.211 | 1.161~ |
| h100-pcie | contract | async | OFF | 0.875 | 0.892 | 0.864 | 0.860~ |
| h100-pcie | contract | fma-regs | OFF | 1.034 | 0.733 | 0.853 | 0.910~ |
| h100-pcie | contract | fma-regs | ON | 1.036 | 1.235 | 1.249 | 1.252 |
| h100-pcie | contract | idx32 | ON | 1.036 | 1.105 | 1.115 | 1.107~ |
| h100-pcie | contract | interleave | ON | 1.000~ | 1.177 | 1.137 | 1.166~ |
| h100-pcie | contract | min-blocks | OFF | 0.997~ | 1.012 | 1.011~ | 0.973~ |
| h100-pcie | contract | pad | OFF | 1.033 |  |  |  |
| h100-pcie | contract | pad | ON |  | 1.115 | 1.132 | 1.134~ |
| h100-pcie | contract | prefetch | ON | 1.935 | 1.076 | 1.131 | 1.130~ |
| h100-pcie | contract | unroll | ON | 1.074 | 1.100 | 1.092 | 1.114~ |
| h100-pcie | contract | warp | OFF | 0.976 |  |  |  |
| h100-pcie | contract | warp | ON |  | 1.236 | 1.261 | 1.248~ |
| h100-pcie | contract | warp-fill | ON | 1.058 |  |  |  |
| h100-pcie | contract | wide-fma | OFF |  | 0.767 | 0.878 | 0.934~ |
| h100-pcie | exact | affine | OFF |  | 0.867 | 0.831 | 0.814 |
| h100-pcie | exact | arch-rungs | ON | 1.099 | 2.242 | 1.955 | 1.587 |
| h100-pcie | exact | async | OFF | 0.910 | 0.904 | 0.756 | 0.735 |
| h100-pcie | exact | fma-regs | OFF | 1.018 | 0.736 | 0.879 | 0.871~ |
| h100-pcie | exact | idx32 | OFF | 0.958 | 0.963 | 0.956 | 0.885~ |
| h100-pcie | exact | interleave | ON | 0.986 | 1.037 | 1.040 | 1.081~ |
| h100-pcie | exact | min-blocks | OFF |  | 0.967 | 0.960 | 0.872~ |
| h100-pcie | exact | min-blocks | ON | 1.097 |  |  |  |
| h100-pcie | exact | pad | OFF | 1.042 | 0.876 | 0.844 | 0.789 |
| h100-pcie | exact | prefetch | ON | 1.908 | 1.925 | 1.996 | 1.867 |
| h100-pcie | exact | unroll | ON | 1.033 | 1.003~ | 1.032 | 1.126~ |
| h100-pcie | exact | warp | ON | 0.989 | 1.032 | 1.031 | 1.107~ |
| h100-pcie | exact | warp-fill | ON | 0.985~ |  |  |  |
| h100-pcie | exact | wide-fma | OFF |  | 0.722 | 0.858 | 0.837~ |
| h100-pcie | TF32 | arch-rungs | ON | 0.995~ | 0.994~ | 1.010~ | 0.999~ |
| h100-pcie | TF32 | min-blocks | OFF | 1.005~ | 1.004~ | 1.001~ | 1.002~ |
| h100-pcie | TF32 | wgmma-hopper | ON | 1.188 | 1.239 | 1.528 | 1.652 |
| rtx-5090 | contract | affine | ON |  | 1.027~ | 1.012~ | 1.014~ |
| rtx-5090 | contract | arch-rungs | ON | 1.108 | 1.067 | 1.125~ | 1.017~ |
| rtx-5090 | contract | async | OFF | 0.902 | 0.937 | 0.935~ | 0.905~ |
| rtx-5090 | contract | fma-regs | OFF | 1.000~ | 1.007~ | 0.909~ | 0.989~ |
| rtx-5090 | contract | fma-regs | ON | 1.000~ | 1.060 | 1.058~ | 1.061~ |
| rtx-5090 | contract | idx32 | OFF | 1.000~ |  |  |  |
| rtx-5090 | contract | idx32 | ON |  | 1.053 | 1.040~ | 1.047~ |
| rtx-5090 | contract | interleave | ON |  | 1.094 | 1.078~ | 1.105~ |
| rtx-5090 | contract | min-blocks | ON | 1.037~ | 1.007~ | 1.010~ | 1.017~ |
| rtx-5090 | contract | pad | ON | 1.000~ | 1.007~ | 0.994~ | 1.010~ |
| rtx-5090 | contract | prefetch | ON | 1.143 | 1.060 | 1.042~ | 1.057~ |
| rtx-5090 | contract | unroll | ON | 1.000~ | 1.127 | 1.093 | 1.101~ |
| rtx-5090 | contract | warp | ON | 1.000~ | 1.013~ | 1.010~ | 1.028~ |
| rtx-5090 | contract | warp-fill | ON | 1.609 |  |  |  |
| rtx-5090 | contract | wide-fma | OFF |  | 1.007~ | 0.913~ | 0.990~ |
| rtx-5090 | exact | affine | OFF |  | 0.992~ | 1.006~ | 0.988~ |
| rtx-5090 | exact | arch-rungs | ON | 1.092 | 1.009~ | 1.085 | 1.019~ |
| rtx-5090 | exact | async | OFF | 0.917 |  |  |  |
| rtx-5090 | exact | async | ON |  | 1.034 | 0.980~ | 0.973~ |
| rtx-5090 | exact | fma-regs | OFF | 1.000~ | 0.987~ | 0.907~ | 0.977~ |
| rtx-5090 | exact | idx32 | OFF | 1.000~ | 1.000~ | 1.005~ | 0.983~ |
| rtx-5090 | exact | interleave | ON |  | 1.013~ | 1.018~ | 1.024~ |
| rtx-5090 | exact | min-blocks | ON | 1.213 | 0.987~ | 0.986~ | 1.000~ |
| rtx-5090 | exact | pad | OFF | 1.000~ |  |  |  |
| rtx-5090 | exact | prefetch | ON | 1.364 |  |  |  |
| rtx-5090 | exact | unroll | ON | 1.061 | 1.038 | 1.007~ | 1.006~ |
| rtx-5090 | exact | warp | ON | 1.000~ | 0.987~ | 1.001~ | 1.014~ |
| rtx-5090 | exact | warp-fill | ON | 1.848 |  |  |  |
| rtx-5090 | exact | wide-fma | OFF |  |  | 0.922 | 0.971~ |
| rtx-5090 | TF32 | arch-rungs | ON | 1.265 | 1.028 |  |  |
| rtx-5090 | TF32 | ldmatrix | ON | 1.000~ | 1.009~ | 1.015 | 1.018~ |
| rtx-5090 | TF32 | min-blocks | ON | 1.000~ | 1.000~ | 1.001~ | 1.000~ |
| rtx-5090 | TF32 | warp-fill | OFF |  | 0.973 |  |  |
| rtx-5090 | TF32 | warp-fill | ON | 1.946 |  |  |  |
| rtx-5090 | TF32 | wide | OFF | 0.791 |  |  |  |
| rtx-5090 | TF32 | wide | ON |  | 1.000~ | 1.011 | 1.031~ |

### 3c. Table rows the integrated run changed

| part | rung | where | item-branch decision | integrated measurement | shipped |
|---|---|---|---|---|---|
| 4090 | `unroll` | filled, contract | OFF (S52 tile: no stable sign) | on-arm 1.083 / 1.166 / 1.086 at 2048 / 4096 / 8192 (with `wide-fma`) | ON |
| 4090 | `async` | filled, exact | ON | with `fma-regs`, register staging wins: on-arm 0.969 / 0.957 / 0.934 | OFF |
| all three | `retain-delta` | decode | ON (vs S52 round trips) | `resident` takes the loop and is faster (§4c) | OFF |
| H100 PCIe | `min-blocks` | filled, both faces | ON | the 2-block cap spills under `idx32`+`pad`: off-arm 1.06 / 1.06 / 0.98 contract; confirm on-arm exact 0.967 / 0.960 / 0.872 | OFF |
| H100 PCIe | `affine` | exact; under-filled contract | ON (provisional) | confirm on-arm exact 0.867 / 0.831 / 0.814; session 1 contract 1024 0.921 | OFF |
| H100 PCIe | `affine` | filled, contract | ON | session 1 (min-blocks on) 0.963 / 0.960 / 1.002; confirm (min-blocks off) 1.035 / 1.027 / 1.020 | ON |
| H100 PCIe | `warp` | under-filled, contract | ON | 0.139 vs 0.136 ms; confirm on-arm 0.976 | OFF |
| H100 PCIe | `ldmatrix` | filled | OFF | row dropped (8% between byte-identical arms in one session; moot under `wgmma`) | ON |
| 5090 | `async` | under-filled | ON (S52 tile) | on `warp-fill`'s 4×4 tile: 0.0635 vs 0.0573 ms contract, 0.0737 vs 0.0676 ms exact; confirm on-arm 0.902 / 0.917 | OFF |
| 5090 | `wide` (TF32) | under-filled | ON | 0.0492 vs 0.0389 ms; confirm on-arm 0.791 | OFF |
| 5090 | `warp-fill` | filled, contract | ON | TF32 2048 on-arm 0.973 (both sessions) | OFF |

Every shipped kernel after these flips was re-emitted and is byte-identical to an arm measured in the session it is
quoted from: 4090 contract 2048–8192 = `on-unroll`; H100 contract 2048–8192 = confirm `on-affine` = session-1
`off-min-blocks`; 5090 1024 = `off-async` / `off-wide`; decode = `off-retain-delta` (4090, H100). The 5090 decode
also takes register staging at its 1024×256×256 sites under the `async` flip (confirm session).

Unmeasured profiles take the measured part of their architecture: `cuda-ada` ← 4090, `h100` (SXM) ← H100 PCIe.

## 4. Whole programs: host side and transfers

`integrated-*.txt` (session 1). 7 processes per arm, arms interleaved; rep 1 of each new PTX pays the driver's JIT
(the max of most ranges, hence the ~). `iter ms` is the program's own clock: S53's is fenced around the site; S52's
read the input generation. The S52 host object runs on the S53 runtime's lazy path. Every timed program prints one
element, so `resident` copies 4 bytes out; a program that reads its whole result keeps that d2h.

### 4a. Before / after: the S52 host object vs the integrated build (same process interleaving)

| part | program | face | build | wall ms med (range) | own `iter ms` | GPU phases ms (excl. context) | context wait ms | bytes in / out | cuBLAS whole-program floor ms |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| rtx-4090 | e2e_1024 | _exact | S52 | 288.2 (273.8–364.2) | 1.31 | 1.55 | 173.4 | 8 MB / 4 MB |  |
| rtx-4090 | e2e_1024 | _exact | S53 | 286.2 (265.5–338.3) | 0.49 | 1.22 | 179.3 | 0 B / 4 B | 230.4 |
| rtx-4090 | e2e_1024 | _c | S52 | 280.8 (260.9–318.3) | 1.31 | 1.52 | 186.3 | 8 MB / 4 MB |  |
| rtx-4090 | e2e_1024 | _c | S53 | 283.5 (244.1–339.1) | 0.46 | 1.03 | 175.3 | 0 B / 4 B | 230.4 |
| rtx-4090 | e2e_4096 | _exact | S52 | 431.6 (417.8–561.8) | 21.89 | 16.54 | 176.6 | 128 MB / 64 MB |  |
| rtx-4090 | e2e_4096 | _exact | S53 | 298.0 (295.1–618.2) | 5.34 | 6.99 | 75.4 | 0 B / 4 B | 231.9 |
| rtx-4090 | e2e_4096 | _c | S52 | 426.2 (413.8–503.1) | 21.93 | 15.17 | 174.0 | 128 MB / 64 MB |  |
| rtx-4090 | e2e_4096 | _c | S53 | 306.0 (282.0–372.0) | 4.11 | 6.01 | 83.4 | 0 B / 4 B | 231.9 |
| h100-pcie | e2e_1024 | _exact | S52 | 495.7 (487.3–813.1) | 1.29 | 1.46 | 363.6 | 8 MB / 4 MB |  |
| h100-pcie | e2e_1024 | _exact | S53 | 479.7 (461.8–574.3) | 0.51 | 1.10 | 356.9 | 0 B / 4 B | 464.2 |
| h100-pcie | e2e_1024 | _c | S52 | 493.0 (486.3–842.6) | 1.28 | 1.43 | 375.2 | 8 MB / 4 MB |  |
| h100-pcie | e2e_1024 | _c | S53 | 509.8 (453.7–1015.5) | 0.47 | 1.06 | 357.0 | 0 B / 4 B | 464.2 |
| h100-pcie | e2e_4096 | _exact | S52 | 687.7 (670.4–1470.2) | 21.14 | 18.21 | 366.1 | 128 MB / 64 MB |  |
| h100-pcie | e2e_4096 | _exact | S53 | 523.5 (499.7–696.3) | 6.87 | 7.69 | 245.1 | 0 B / 4 B | 447.9 |
| h100-pcie | e2e_4096 | _c | S52 | 689.2 (662.6–934.2) | 21.06 | 13.99 | 367.9 | 128 MB / 64 MB |  |
| h100-pcie | e2e_4096 | _c | S53 | 523.9 (491.0–1067.9) | 4.18 | 4.97 | 235.5 | 0 B / 4 B | 447.9 |
| rtx-5090 | e2e_1024 | _exact | S52 | 191.9 (181.0–237.0) | 1.10 | 0.95 | 115.2 | 8 MB / 4 MB |  |
| rtx-5090 | e2e_1024 | _exact | S53 | 180.5 (175.8–223.7) | 0.22 | 0.46 | 110.9 | 0 B / 4 B | 159.9 |
| rtx-5090 | e2e_1024 | _c | S52 | 196.8 (179.8–227.7) | 1.08 | 0.97 | 121.1 | 8 MB / 4 MB |  |
| rtx-5090 | e2e_1024 | _c | S53 | 187.0 (176.6–210.6) | 0.20 | 0.58 | 120.5 | 0 B / 4 B | 159.9 |
| rtx-5090 | e2e_4096 | _exact | S52 | 258.4 (248.6–274.9) | 17.96 | 11.79 | 115.8 | 128 MB / 64 MB |  |
| rtx-5090 | e2e_4096 | _exact | S53 | 198.6 (187.2–318.9) | 3.62 | 4.15 | 75.6 | 0 B / 4 B | 174.2 |
| rtx-5090 | e2e_4096 | _c | S52 | 258.4 (247.4–330.8) | 18.40 | 10.72 | 116.4 | 128 MB / 64 MB |  |
| rtx-5090 | e2e_4096 | _c | S53 | 194.3 (184.7–245.9) | 2.28 | 2.80 | 79.1 | 0 B / 4 B | 174.2 |

### 4b. Host rungs in the integrated build: process wall, arm / shipped (> 1 = the rung pays; ~ = ranges overlap)

| part | program | face | resident off | devgen off | eager-init off | S52 host |
|---|---|---|---:|---:|---:|---:|
| rtx-4090 | e2e_1024 | _exact | 0.961~ | 1.033~ | 0.989~ | 1.007~ |
| rtx-4090 | e2e_1024 | _c | 0.989~ | 0.990~ | 0.983~ | 0.990~ |
| rtx-4090 | e2e_4096 | _exact | 1.062~ | 1.047~ | 1.369~ | 1.448~ |
| rtx-4090 | e2e_4096 | _c | 1.091~ | 1.030~ | 1.342 | 1.393 |
| rtx-4090 | mm_share_1024 | _exact | 0.973~ | 1.015~ | 0.992~ |  |
| rtx-4090 | mm_share_1024 | _c | 1.009~ | 0.997~ | 1.023~ |  |
| rtx-4090 | mm_chain_2048 | _exact | 1.039~ | 1.021~ | 1.094~ |  |
| rtx-4090 | mm_chain_2048 | _c | 1.068~ | 1.072~ | 1.131~ |  |
| rtx-4090 | e2e_chain_4096 | _exact | 1.093~ | 1.104~ | 1.434~ |  |
| rtx-4090 | e2e_chain_4096 | _c | 1.071~ | 1.072~ | 1.391 |  |
| rtx-4090 | loop_mm_1024_8 | _exact | 1.074~ | 0.973~ | 1.046~ |  |
| rtx-4090 | loop_mm_1024_8 | _c | 1.053~ | 0.996~ | 1.078~ |  |
| h100-pcie | e2e_1024 | _exact | 1.029~ | 1.019~ | 1.013~ | 1.033~ |
| h100-pcie | e2e_1024 | _c | 0.961~ | 0.955~ | 0.964~ | 0.967~ |
| h100-pcie | e2e_4096 | _exact | 1.020~ | 1.030~ | 1.250~ | 1.314~ |
| h100-pcie | e2e_4096 | _c | 1.031~ | 1.034~ | 1.263~ | 1.316~ |
| h100-pcie | mm_share_1024 | _exact | 1.018~ | 0.994~ | 1.012~ |  |
| h100-pcie | mm_share_1024 | _c | 1.003~ | 1.014~ | 1.028~ |  |
| h100-pcie | mm_chain_2048 | _exact | 0.994~ | 0.994~ | 1.082~ |  |
| h100-pcie | mm_chain_2048 | _c | 1.008~ | 1.019~ | 1.089~ |  |
| h100-pcie | e2e_chain_4096 | _exact | 1.028~ | 1.083~ | 1.326 |  |
| h100-pcie | e2e_chain_4096 | _c | 1.051~ | 1.073~ | 1.367 |  |
| h100-pcie | loop_mm_1024_8 | _exact | 1.028~ | 1.030~ | 1.061~ |  |
| h100-pcie | loop_mm_1024_8 | _c | 0.991~ | 0.958~ | 0.986~ |  |
| rtx-5090 | e2e_1024 | _exact | 0.986~ | 1.038~ | 1.067~ | 1.063~ |
| rtx-5090 | e2e_1024 | _c | 0.986~ | 0.965~ | 1.004~ | 1.052~ |
| rtx-5090 | e2e_4096 | _exact | 1.050~ | 1.070~ | 1.169~ | 1.301~ |
| rtx-5090 | e2e_4096 | _c | 1.017~ | 1.078~ | 1.182~ | 1.330 |
| rtx-5090 | mm_share_1024 | _exact | 0.980~ | 0.957~ | 0.991~ |  |
| rtx-5090 | mm_share_1024 | _c | 0.990~ | 0.946~ | 0.957~ |  |
| rtx-5090 | mm_chain_2048 | _exact | 1.024~ | 1.039~ | 1.076~ |  |
| rtx-5090 | mm_chain_2048 | _c | 1.064~ | 1.075~ | 1.060~ |  |
| rtx-5090 | e2e_chain_4096 | _exact | 1.090~ | 1.135~ | 1.265~ |  |
| rtx-5090 | e2e_chain_4096 | _c | 1.123~ | 1.142~ | 1.228~ |  |
| rtx-5090 | loop_mm_1024_8 | _exact | 1.004~ | 0.978~ | 1.059~ |  |
| rtx-5090 | loop_mm_1024_8 | _c | 1.027~ | 1.018~ | 1.043~ |  |

### 4c. Decode (T = 1024, d = 256): process wall s, 3 processes

| part | face | shipped (resident) | retain-delta on (resident yields) | every site a round trip |
|---|---|---:|---:|---:|
| rtx-4090 | _exact | 0.905 | 1.589 | 2.409 |
| rtx-4090 | _c | 0.886 | 1.568 | 2.465 |
| h100-pcie | _exact | 1.109 | 1.756 | 2.467 |
| h100-pcie | _c | 1.144 | 1.906 | 2.410 |
| rtx-5090 | _exact | 0.630 | 0.825 | 1.257 |
| rtx-5090 | _c | 0.616 | 0.847 | 1.253 |

## 5. Rejected

| what | where | why (measured) | file |
|---|---|---|---|
| `pool` (stream-ordered allocator) | all parts | 23 of 24 e2e medians 3–17 ms slower; first pool allocation 4–19 ms vs 0.1–3.0 ms of `cuMemFree` saved. **Deleted** (field, arm, runtime path) | host-runtime-*.txt |
| `wgmma-reg-stage` | H100 PCIe | slower at 2048–8192 (0.201 / 0.956 / 8.35 vs 0.172 / 0.912 / 7.54 ms); +17% at 1024 only. Removed | wgmma-hopper-h100-pcie-run2-regstage.txt |
| `wgmma-overlap` (2 MMA groups in flight) | H100 PCIe | nsys 4096 0.977 ms vs 0.804 ms; failed the TF32 value gate with reg-stage at 1024 (a race). Removed | wgmma-hopper-h100-pcie-run3-overlap.txt |
| `wide-fma` (8×16 FMA tile) | H100 PCIe | contract register staging 0.822 / 0.950 / 1.015; exact 0.750 / 0.897 / 0.971 | fma-regs-h100-pcie.txt |
| `wide-fma` | 5090 | contract register staging 1.012 / **0.923** / 1.058 and 0.994 / **0.911** / 0.984 (4096 disjoint) | fma-regs-rtx-5090*.txt |
| idx32 / affine / pad, exact face | H100 PCIe, 5090 | no gain: H100 1.005 / 1.000 / 0.999; 5090 cp.async 0.996–1.000 (idx32), 0.986–0.997 (affine) | fma-regs-*.txt |
| `cp.async` FMA staging | 4090 filled (both faces), 5090 contract and under-filled, H100 everywhere | 4090 contract 2.823 vs 3.027 ms at 4096 without it; H100 exact 6.428 vs 8.292 ms | arch-rungs-*.txt, fma-regs-rtx-4090.txt |
| `ldmatrix` | 4090 | no effect (0.084/0.255/1.733/14.402 vs 0.084/0.254/1.731/14.387 ms) | arch-rungs-rtx-4090.txt |
| `warp` filled contract | 4090 | no stable sign across sessions (inert on the `wide-fma` tile) | arch-rungs-rtx-4090*.txt |
| `min-blocks` under-filled contract | H100 PCIe | 0.139 vs 0.122 ms at 1024 | arch-rungs-h100-pcie.txt |
| `warp-fill` filled contract | 5090 | TF32 2048: 0.2314 vs 0.2252 ms (disjoint); FMA kernels identical | warp-fill-rtx-5090.txt |
| H100 `ldmatrix` OFF row (arch-rungs) | H100 PCIe | dropped: 8% gap between byte-identical arms in the same session; ON faster at 2048 in both sessions. Moot under `wgmma` | arch-rungs-h100-pcie*.txt |
| `devgen` on a float `%` body | all | not bit-exact: host `fmodf` vs NVPTX `frem` expansion, 2051 / 4096 elements differ. Now refused (`fmod_64` in the differential) | review |
| `retain-delta` (shipping) | all three | slower than `resident` on the one program it fires on (§4c); kept as a rung, `--gpu-on=retain-delta` | integrated-*.txt |

## 6. Correctness

| session | differential (program × face × arm vs oracle) | trap before the first site (rc 101) | kernel benches: bit-exact · TF32-bounded |
|---|---|---|---|
| 4090 `integrated-rtx-4090.txt` | 126 / 126 | 10 / 10 | 105 / 105 · 23 / 23 |
| H100 `integrated-h100-pcie.txt` | 126 / 126 | 10 / 10 | 119 / 119 · 19 / 19 |
| H100 `integrated-h100-pcie-confirm.txt` | 134 / 134 | 10 / 10 | 119 / 119 · 22 / 22 |
| 5090 `integrated-rtx-5090.txt` | 126 / 126 | 10 / 10 | 107 / 107 · 21 / 21 |
| 5090 `integrated-rtx-5090-confirm.txt` | 134 / 134 | 10 / 10 | 109 / 109 · 26 / 26 |

Differential programs: the six S52 tile programs, `wide_2048x2048x64` (a filled grid: the 4090's `wide-fma` tile),
`fmod_64` (a float `%` operand `devgen` must leave on the host), `chain_256`, `share_256`, `loop_128_8`, `decode_16_8`,
`decode_1024_256`, `loop_rewrite_16_8`; both faces, every arm that changes the emission.
