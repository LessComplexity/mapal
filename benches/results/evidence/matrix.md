# Mapal vs the fastest competitor at the same precision

Built 2026-11-22 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations) vs the torch bf16 legs (bf16 throughout). The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), SDPA (eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash). **Not included:** torch with CUDA graphs, hand-written Triton kernels, the FlashAttention-2/3 package, FlashInfer, cuDNN attention, TensorRT-LLM / vLLM.

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | 5b6c0a29, 7a2b67bf, c03a52a3, c8b83028, d1927ac1 | long-prefill/suite@c03a52a3: 37; large-gemm/resM@7a2b67bf: 24; long-prefill/resL@c8b83028: 9; half-precision/run1@d1927ac1: 7; half-precision/run2@d1927ac1: 7; device-geometry/v3/model@5b6c0a29: 2 |
| rtx-3090 | 60f0f8ce | 60f0f8ce/rtx-3090@60f0f8ce: 83 |
| a100 | 60f0f8ce | 60f0f8ce/a100@60f0f8ce: 83 |
| rtx-4090 | 60f0f8ce | 60f0f8ce/rtx-4090@60f0f8ce: 83 |
| h100 | 60f0f8ce | 60f0f8ce/h100@60f0f8ce: 83 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 1.167 (31) | 1.178 (31) | 1.213 (31) | 1.195 (31) | 1.497 (31) |
| MLP chains | 1.049 (6) | 1.160 (6) | 1.236 (6) | 1.030 (6) | 1.615 (6) |
| Attention / softmax / long rows | 0.780 (14) | 0.924 (11) | 1.201 (11) | 0.922 (11) | 0.908 (11) |
| LLM decode / prefill / block (f32, tf32, bf16w) | 1.029 (19) | 1.166 (19) | 1.198 (19) | 1.223 (19) | 0.950 (19) |
| Conv / FIR / elementwise | 0.531 (7) | 0.877 (7) | 0.949 (7) | 0.735 (7) | 0.985 (7) |
| Fan-in/out, loops, CPU+GPU | 1.046 (9) | 1.100 (9) | 1.190 (9) | 1.005 (9) | 1.187 (9) |
| all f32 rows | 0.882 (66) | 1.041 (63) | 1.073 (63) | 0.996 (63) | 1.096 (63) |
| all tf32 rows | 1.283 (13) | 1.262 (13) | 1.628 (13) | 1.300 (13) | 2.034 (13) |
| all bf16w rows | 1.573 (7) | 1.409 (7) | 1.569 (7) | 1.613 (7) | 0.968 (7) |
| **all rows** | 0.978 (86) | 1.101 (83) | 1.182 (83) | 1.082 (83) | 1.195 (83) |
| rows timed on every part (83) | 0.975 (83) | 1.101 (83) | 1.182 (83) | 1.082 (83) | 1.195 (83) |
| win / even / loss | 19/24/43 | 17/13/53 | 16/13/54 | 17/26/40 | 18/1/64 |

Geomean cells: value (rows). Parts time different row sets (the RTX 5080 adds 5080-only rows, the rented parts the full model-scale set); compare columns on the "every part" row.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 1.02≈ 3.669/3.585 (chain-lt) | 0.96≈ 5.596/5.828 (chain-lt) | 1.08 7.88/7.296 (chain-lt) | 0.99≈ 2.407/2.434 (chain-lt) | 1.13 2.999/2.663 (chain-lt) |
| mm_16x2048x7168 | f32 | 1.09 0.08705/0.0799 (cublas) | 1.02≈ 0.09005/0.08805 (cublas) | 1.02≈ 0.09215/0.0906 (cublas) ±6% | 1.13 0.1084/0.0963 (cublas) | 1.84 0.09575/0.05195 (cublas) |
| mm_4096x4096x4096 | f32 | 1.05≈ 3.823/3.65 (cublas) | 1.11 7.115/6.409 (cublas) | 1.04≈ 7.681/7.402 (cublas) | 0.97≈ 2.393/2.477 (cublas) | 1.18 3.136/2.658 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.08 0.8082/0.7496 (lt) | 0.98≈ 1.235/1.265 (torch-compile) | 0.97≈ 1.727/1.781 (cublas) | 0.98≈ 0.6337/0.6492 (lt-fresh) | 1.07 0.7124/0.6633 (lt-fresh) |
| pj_m128_n4096_k14336 | f32 | 1.14 0.457/0.4014 (cublas-fresh) | 1.08 0.7261/0.6748 (torch-eager) | 1.08 0.9178/0.8509 (cublas-fresh) | 1.09 0.3881/0.3574 (lt-fresh) | 1.21 0.378/0.3131 (lt) |
| pj_m128_n4096_k4096 | f32 | 1.07 0.1445/0.1352 (cublas-fresh) | 1.23 0.2524/0.2048 (lt) | 1.18 0.3061/0.2601 (cublas) | 1.33 0.1357/0.1024 (lt-fresh) | 1.51 0.1476/0.098 (lt) |
| pj_m128_n6144_k4096 | f32 | 1.16 0.2086/0.1802 (cublas-fresh) | 1.20 0.3484/0.2908 (lt) | 1.16 0.4317/0.3727 (lt-fresh) | 1.22 0.1911/0.1567 (lt-fresh) | 1.36 0.2016/0.1486 (lt-fresh) |
| pj_m16_n18432_k7168 | f32 | 1.02≈ 0.6014/0.5878 (cublas-fresh) | 1.07 0.653/0.6113 (torch-compile) | 0.98≈ 0.4347/0.4454 (torch-compile) | 1.04≈ 0.6352/0.6113 (lt-fresh) | 1.12 0.2317/0.2072 (torch-compile) |
| pj_m16_n2048_k7168 | f32 | **0.95** 0.04653/0.0492 (cublas-fresh) | 1.43 0.1199/0.084 (torch-compile) | 1.19 0.1035/0.087 (cublas-fresh) | 1.74 0.05337/0.0307 (cublas-fresh) ±16% | 1.58 0.07647/0.0484 (lt-fresh) |
| pj_m16_n28672_k4096 | f32 | 1.05 0.5483/0.5222 (lt-fresh) | 1.09 0.5942/0.5458 (torch-compile) | 0.99≈ 0.4203/0.426 (torch-compile) | 1.04≈ 0.5731/0.5509 (cublas-fresh) | 1.26 0.2216/0.1761 (torch-compile) |
| pj_m16_n4096_k14336 | f32 | 0.96≈ 0.2668/0.2765 (torch-compile) | 1.13 0.3119/0.2755 (torch-compile) | **0.90** 0.2269/0.2509 (lt) | 1.05 0.318/0.3021 (cublas-fresh) | 1.20 0.1393/0.1159 (lt) |
| pj_m16_n4096_k4096 | f32 | 0.96≈ 0.05285/0.0553 (cublas-fresh) | 1.60 0.1372/0.086 (torch-compile) | 1.13 0.1004/0.0891 (cublas-fresh) | 1.54 0.0535/0.0348 (cublas-fresh) | 1.48 0.07107/0.0479 (lt-fresh) |
| pj_m16_n6144_k4096 | f32 | 1.09 0.0956/0.0881 (lt-fresh) | 1.34 0.1664/0.1239 (torch-compile) | 1.11 0.1291/0.1167 (torch-compile) | 1.22 0.1221/0.1004 (cublas-fresh) | 1.55 0.08844/0.0572 (torch-compile) ±6% |
| pj_m1_n18432_k7168 | f32 | 0.99≈ 0.6047/0.6103 (torch-compile) | 1.04≈ 0.6312/0.6062 (lt-fresh) | 1.10 0.4308/0.3922 (lt-fresh) | 1.17 0.6274/0.5366 (lt-fresh) | 1.21 0.226/0.1868 (lt-fresh) |
| pj_m1_n2048_k7168 | f32 | 1.24 0.03288/0.0266 (cublas-fresh) | 1.39 0.1038/0.0748 (lt-fresh) | 1.99 0.08766/0.044 (cublas-fresh) | 1.20 0.0332/0.0276 (lt-fresh) ±6% | 2.17 0.05269/0.0243 (cublas-fresh) ±7% |
| pj_m1_n28672_k4096 | f32 | 1.11 0.5582/0.5038 (lt-fresh) | 1.06 0.5718/0.5376 (lt) | 1.10 0.3827/0.3492 (torch-compile) | 1.18 0.5685/0.4823 (lt-fresh) | 1.24 0.2046/0.1649 (lt-fresh) |
| pj_m1_n4096_k14336 | f32 | 1.10 0.2545/0.2314 (cublas-fresh) | 1.09 0.3025/0.2765 (lt) | 1.13 0.2021/0.1782 (cublas-fresh) | 1.47 0.2995/0.2038 (lt-fresh) | 1.56 0.1285/0.0824 (cublas-fresh) ±8% |
| pj_m1_n4096_k4096 | f32 | 2.21 0.04971/0.0225 (lt-fresh) | 1.31 0.113/0.086 (lt) | 1.75 0.0932/0.0532 (cublas-fresh) | 1.66 0.03228/0.0195 (lt-fresh) | 2.89 0.06915/0.0239 (lt-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.95 0.1/0.0512 (cublas-fresh) | 1.23 0.151/0.1229 (lt-fresh) | 1.49 0.116/0.0778 (lt-fresh) | 1.43 0.08477/0.0592 (lt-fresh) | 1.90 0.06935/0.0365 (cublas-fresh) |
| pj_m2048_n28672_k4096 | f32 | 0.97≈ 11.81/12.23 (lt-fresh) | 1.11 21.02/18.86 (torch-compile) | 1.03≈ 26.86/26.02 (torch-compile) | 1.05≈ 8.673/8.29 (lt) | 1.10 10.16/9.204 (torch-compile) |
| pj_m2048_n4096_k14336 | f32 | 1.00≈ 6.437/6.42 (cublas-fresh) | 1.06 11.09/10.47 (torch-compile) | 1.03≈ 13.34/12.88 (torch-eager) | **0.95** 4.017/4.236 (torch-compile) | 1.16 5.24/4.533 (torch-eager) |
| pj_m2048_n4096_k4096 | f32 | 1.05 1.843/1.747 (torch-eager) | 1.05≈ 3.01/2.869 (torch-compile) | 1.04≈ 3.867/3.709 (cublas-fresh) | 0.99≈ 1.167/1.173 (cublas-fresh) | 1.15 1.515/1.318 (torch-compile) |
| pj_m2048_n6144_k4096 | f32 | 1.03≈ 2.653/2.578 (torch-compile) | 1.10 4.584/4.177 (cublas) | **0.94** 5.804/6.147 (lt) | 0.98≈ 1.751/1.791 (cublas) | 1.13 2.231/1.977 (torch-eager) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.18 0.6966/0.5898 (cublas-tf32-fresh) | 1.15 1.071/0.9298 (cublas-tf32) | 1.19 0.5376/0.4526 (lt-tf32) | 1.01≈ 0.5749/0.5704 (cublas-tf32-fresh) | 1.36 0.2515/0.1856 (cublas-tf32) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.47 0.4558/0.3092 (cublas-tf32-fresh) | 1.16 0.7229/0.6216 (torch-compile-tf32) | 2.02 0.4672/0.2314 (lt-tf32-fresh) | 1.24 0.38/0.3072 (lt-tf32-fresh) | 2.31 0.2466/0.1067 (lt-tf32) |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.61 0.1354/0.084 (cublas-tf32-fresh) | 1.31 0.219/0.1669 (lt-tf32-fresh) | 1.79 0.1482/0.0829 (cublas-tf32-fresh) | 1.43 0.0955/0.0666 (cublas-tf32-fresh) | 1.94 0.08557/0.0442 (cublas-tf32-fresh) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.92 0.2392/0.1249 (cublas-tf32-fresh) | 1.89 0.4239/0.2243 (lt-tf32-fresh) | 1.57 0.1612/0.1024 (cublas-tf32-fresh) | 1.58 0.1808/0.1147 (cublas-tf32-fresh) | 2.94 0.1566/0.0532 (lt-tf32-fresh) |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.13 9.042/8 (cublas-tf32-fresh) | 1.16 14.85/12.83 (torch-eager-tf32) | 1.43 5.928/4.153 (cublas-tf32-fresh) | 1.49 8.235/5.542 (torch-compile-tf32) | 1.57 2.037/1.294 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.15 4.901/4.251 (cublas-tf32-fresh) | 1.19 8.366/7.007 (torch-eager-tf32) | 1.26 2.709/2.147 (cublas-tf32) | 1.19 3.31/2.788 (cublas-tf32-fresh) | 1.68 0.9885/0.5878 (torch-eager-tf32) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.22 1.418/1.163 (cublas-tf32-fresh) | 1.18 2.408/2.038 (torch-compile-tf32) | 1.37 0.8018/0.5857 (lt-tf32) | 1.11 0.952/0.8581 (lt-tf32) | 1.77 0.3111/0.1759 (cublas-tf32-fresh) |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.15 2.014/1.75 (lt-tf32-fresh) | 1.19 3.413/2.871 (torch-compile-tf32) | 1.53 1.246/0.8161 (torch-eager-tf32) | 1.19 1.423/1.196 (torch-eager-tf32) | 1.77 0.4608/0.2603 (lt-tf32-fresh) |
| geomean f32 | | 1.115 (23) | 1.150 (23) | 1.127 (23) | 1.171 (23) | 1.386 (23) |
| geomean tf32 | | 1.331 (8) | 1.262 (8) | 1.498 (8) | 1.265 (8) | 1.868 (8) |
| **geomean** | | 1.167 (31) | 1.178 (31) | 1.213 (31) | 1.195 (31) | 1.497 (31) |
| win / even / loss | | 1/9/21 | 0/5/26 | 2/8/21 | 1/9/21 | 0/0/31 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.10 1.944/1.773 (chain-lt2) | 1.09 3.253/2.98 (chain-lt2) | 1.17 4.399/3.76 (chain-streams) | 0.96≈ 1.184/1.232 (chain-cutlass) | 1.11 1.511/1.36 (chain-streams) |
| mlp_2048 | f32 | 1.04≈ 0.9462/0.9134 (chain-lt) | 1.06 1.621/1.532 (torch-compile) | 1.13 2.222/1.964 (chain-lt) | 0.96≈ 0.5925/0.6195 (chain-cutlass) | 1.11 0.761/0.6858 (chain-lt) |
| mlp_512 | f32 | **0.91** 0.03898/0.043 (chain-eager) | 1.50 0.07958/0.0532 (chain-lt) | 1.24 0.07773/0.0625 (chain-lt) | 1.03≈ 0.03892/0.0379 (chain-lt) | 2.21 0.07836/0.0354 (chain-lt) ±12% |
| smlp_4096_256_1024 | f32 | 1.10 0.1651/0.1495 (torch-compile) | 1.23 0.2825/0.2294 (torch-compile) | 1.18 0.3455/0.2918 (chain-eager) | 0.99≈ 0.1083/0.1096 (torch-compile) | 1.21 0.1353/0.1116 (chain-lt) ±6% |
| mlp2_2048_tf32 | tf32 | 1.08 1.669/1.545 (torch-eager-tf32) | 1.08 2.834/2.626 (torch-eager-tf32) | 1.28 0.932/0.7301 (torch-compile-tf32) | 1.14 0.9645/0.8489 (torch-compile-tf32) | 2.19 0.5347/0.2437 (torch-eager-tf32) |
| mlp_2048_tf32 | tf32 | 1.08 0.8308/0.766 (torch-eager-tf32) | 1.06 1.395/1.317 (torch-eager-tf32) | 1.43 0.5081/0.3553 (torch-eager-tf32) | 1.13 0.4891/0.4342 (torch-eager-tf32) | 2.45 0.2734/0.1117 (torch-eager-tf32) |
| geomean f32 | | 1.033 (4) | 1.208 (4) | 1.182 (4) | 0.983 (4) | 1.349 (4) |
| geomean tf32 | | 1.082 (2) | 1.069 (2) | 1.351 (2) | 1.131 (2) | 2.317 (2) |
| **geomean** | | 1.049 (6) | 1.160 (6) | 1.236 (6) | 1.030 (6) | 1.615 (6) |
| win / even / loss | | 1/1/4 | 0/0/6 | 0/0/6 | 0/4/2 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.32 0.03788/0.0287 (torch-compile) | 1.83 0.1067/0.0584 (chain-fused) | 1.32 0.06356/0.0481 (torch-compile) | 1.47 0.04645/0.0317 (torch-compile) | 2.40 0.09197/0.0384 (chain-fused) ±9% |
| attn_2048_128 | f32 | 1.12 0.1173/0.1044 (torch-sdpa-compile) | - | - | - | - |
| attn_2048_64 | f32 | 1.07 0.07973/0.0748 (torch-compile) | - | - | - | - |
| causal_1024_256 | f32 | **0.32** 15.21/48.24 (torch-compile) | **0.13** 25.04/199.7 (torch-compile) | **0.51** 36.8/72.58 (torch-compile) | **0.18** 18.38/101.2 (torch-compile) | **0.19** 38.36/201.1 (torch-compile) ±13% |
| decode_1024_256 | f32 | **0.54** 8.707/16.19 (chain-cache) | **0.79** 26.1/33.22 (chain-cache) | 0.96≈ 27.39/28.66 (chain-cache) | **0.75** 14.73/19.68 (chain-cache) | **0.73** 28.07/38.44 (chain-cache) ±6% |
| gen_1024_512_256 | f32 | **0.32** 7.757/24.39 (torch-compile) | **0.13** 12.81/100.7 (torch-compile) | **0.48** 18.53/38.55 (torch-compile) | **0.18** 9.32/50.58 (torch-compile) | **0.13** 12.2/96.46 (torch-compile) ±10% |
| lsm_1_131072 | f32 | **0.65** 0.01525/0.0236 (torch-compile) | 11.18 0.3208/0.0287 (torch-compile) | 10.80 0.3543/0.0328 (torch-compile) | 8.39 0.2232/0.0266 (torch-sdpa-compile) | 5.81 0.3387/0.0583 (torch-sdpa) |
| lsm_32_8192 | f32 | 1.80 0.01841/0.0102 (torch-compile) | 2.69 0.04146/0.0154 (torch-compile) | 1.67 0.0307/0.0184 (torch-compile) | 2.44 0.02688/0.011 (torch-compile) ±6% | 2.52 0.04112/0.0163 (torch-sdpa) ±8% |
| lsm_4_32768 | f32 | **0.87** 0.01602/0.0184 (torch-sdpa) ±6% | 3.56 0.08762/0.0246 (torch-sdpa) | 2.38 0.0732/0.0307 (torch-sdpa) | 2.97 0.05782/0.0195 (torch-sdpa) | 2.83 0.06518/0.023 (torch-sdpa) ±15% |
| prefill_1024_512_256 | f32 | **0.32** 7.739/24.43 (torch-compile) | **0.13** 13.16/98.09 (torch-compile) | **0.49** 18.87/38.52 (torch-compile) | **0.19** 9.51/50.58 (torch-compile) | **0.13** 13.2/99.58 (torch-compile) |
| rowsum_2048 | f32 | 1.00≈ 0.4636/0.4649 (chain-eager) | 1.10 0.8528/0.7731 (torch-compile) | 1.04≈ 1.035/0.9964 (chain-eager) | 0.98≈ 0.3122/0.3187 (torch-compile) | 1.19 0.4147/0.3478 (torch-compile) |
| rowsum_4096 | f32 | 1.07 3.747/3.5 (torch-compile) | 1.07 6.225/5.807 (chain-eager) | 1.06 7.749/7.339 (torch-compile) | 1.00≈ 2.379/2.387 (torch-eager) | 1.12 2.996/2.683 (torch-eager) |
| softmax_rows_2048 | f32 | 1.03≈ 0.4872/0.471 (torch-compile) | - | - | - | - |
| softmax_rows_4096 | f32 | 1.01≈ 3.805/3.765 (chain-fused) | 1.08 6.444/5.964 (torch-compile) | 1.06 7.817/7.378 (torch-compile) | 1.00≈ 2.442/2.451 (torch-compile) | 1.12 3.024/2.712 (torch-compile) |
| **geomean** | | 0.780 (14) | 0.924 (11) | 1.201 (11) | 0.922 (11) | 0.908 (11) |
| win / even / loss | | 6/3/5 | 4/0/7 | 3/2/6 | 4/3/4 | 4/0/7 |

## LLM decode / prefill / block (f32, tf32, bf16w)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 1.00≈ 1.071/1.074 (torch-compile) | **0.94** 1.133/1.203 (torch-compile) | 1.14 0.9588/0.8421 (torch-compile) | 1.42 1.494/1.053 (torch-compile) | 1.25 0.5689/0.4535 (torch-compile) |
| llm_blkpre_t2048 | f32 | 0.99≈ 27.45/27.74 (torch-compile) | **0.93** 45.99/49.42 (torch-compile) | 1.01≈ 55.87/55.18 (torch-compile) | 0.98≈ 18.55/18.98 (torch-compile) | 1.02≈ 21.58/21.13 (torch-eager) |
| llm_dec_b1_l2048 | f32 | **0.93** 0.2391/0.2575 (torch-compile) | **0.76** 0.2774/0.3642 (torch-compile) | **0.87** 0.2153/0.2488 (torch-compile) | **0.88** 0.242/0.2737 (torch-compile) | **0.47** 0.1158/0.2469 (torch-compile) |
| llm_dec_b1_l8192 | f32 | 0.96≈ 0.3314/0.3444 (torch-compile) | **0.83** 0.3777/0.4557 (torch-compile) | **0.70** 0.2756/0.3942 (torch-compile) | 1.02≈ 0.3439/0.3384 (torch-compile) | **0.49** 0.1659/0.3386 (torch-compile) |
| llm_dec_b32_l2048 | f32 | **0.45** 1.033/2.277 (torch-compile) | **0.45** 1.224/2.696 (torch-compile) | **0.47** 0.8594/1.842 (torch-compile) | **0.49** 1.129/2.309 (torch-compile) | **0.40** 0.4715/1.182 (torch-compile) |
| llm_dec_b32_l8192 | f32 | **0.37** 3.52/9.523 (torch-compile) | **0.40** 4.282/10.83 (torch-compile) | **0.39** 2.535/6.525 (torch-compile) | **0.40** 3.765/9.405 (torch-compile) | **0.26** 1.038/3.991 (torch-compile) |
| llm_dec_b8_l2048 | f32 | **0.64** 0.4337/0.6782 (torch-compile) | 1.15 1.023/0.8885 (torch-compile) | **0.64** 0.3983/0.6228 (torch-compile) | **0.70** 0.4967/0.7128 (torch-compile) | **0.44** 0.207/0.469 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.43** 1.064/2.466 (torch-compile) | **0.61** 1.801/2.939 (torch-compile) | **0.44** 0.8091/1.823 (torch-compile) | **0.45** 1.132/2.532 (torch-compile) | **0.41** 0.4684/1.147 (torch-compile) |
| llm_pre_t2048 | f32 | 1.04≈ 8.636/8.311 (torch-compile) ±19% | 1.16 14.46/12.44 (torch-compile) | 1.01≈ 15.13/15.01 (torch-compile) | 1.04≈ 6.318/6.063 (torch-compile) | 1.12 6.153/5.517 (torch-compile) |
| llm_pre_t8192 | f32 | 0.97≈ 82.06/84.25 (torch-compile) | 13.74 1731/126 (torch-compile) | 12.30 1431/116.3 (torch-compile) | 13.50 753.2/55.8 (torch-compile) | 19.06 827.9/43.43 (torch-compile) |
| llm_blkpre_t2048_tf32 | tf32 | 1.38 25.66/18.53 (torch-compile-tf32) ±13% | 1.05≈ 33.89/32.39 (torch-compile-tf32) | 1.47 16.37/11.13 (torch-compile-tf32) | 1.25 16.84/13.46 (torch-compile-tf32) | 1.91 6.719/3.51 (torch-compile-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.27 7.71/6.068 (torch-compile-tf32) | 1.43 12.75/8.922 (torch-compile-tf32) | 2.03 7.412/3.655 (torch-compile-tf32) | 1.21 5.821/4.822 (torch-compile-tf32) | 2.12 3.242/1.529 (torch-compile-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.14 0.6369/0.5566 (torch-bf16-compile) | 1.05≈ 0.7123/0.6788 (torch-bf16-compile) | 1.75 0.7899/0.4515 (torch-bf16-compile) | 1.04≈ 0.5946/0.572 (torch-bf16-compile) | **0.82** 0.3523/0.4288 (torch-bf16-compile) |
| llm_dec_b1_l2048_bf16w | bf16w | 1.05≈ 0.1494/0.1428 (torch-bf16-compile) | **0.74** 0.1854/0.2502 (torch-bf16-compile) | 1.09 0.1698/0.1551 (torch-bf16-compile) | **0.89** 0.1542/0.1724 (torch-bf16-compile) | **0.33** 0.09614/0.2931 (torch-bf16-compile) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.31 0.2423/0.1851 (torch-bf16-compile) | 0.95≈ 0.2869/0.3018 (torch-bf16-compile) | 1.16 0.2305/0.1981 (torch-bf16-compile) | 1.24 0.2575/0.2076 (torch-bf16-compile) | **0.84** 0.1412/0.1675 (torch-bf16-compile) |
| llm_dec_b32_l2048_bf16w | bf16w | 2.07 0.9569/0.4631 (torch-bf16-compile) | 1.65 1.196/0.7245 (torch-bf16-compile) | 1.83 0.8665/0.4747 (torch-bf16-compile) | 2.25 1.023/0.4542 (torch-bf16-compile) | 1.62 0.4911/0.3037 (torch-bf16-compile) |
| llm_dec_b32_l8192_bf16w | bf16w | 2.31 3.437/1.486 (torch-bf16-compile) | 1.76 4.244/2.412 (torch-bf16-compile) | 2.11 2.541/1.206 (torch-bf16) | 2.70 3.661/1.354 (torch-bf16-compile) | 1.88 1.054/0.5618 (torch-bf16-compile) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.57 0.3413/0.2169 (torch-bf16-compile) | 2.35 0.8375/0.3557 (torch-bf16-compile) | 1.52 0.3471/0.2285 (torch-bf16-compile) | 1.77 0.4174/0.2364 (torch-bf16-compile) | **0.74** 0.1897/0.2566 (torch-bf16-compile) |
| llm_dec_b8_l8192_bf16w | bf16w | 2.02 0.9745/0.4821 (torch-bf16) | 2.18 1.615/0.7404 (torch-bf16-compile) | 1.80 0.7625/0.4232 (torch-bf16) | 2.29 1.051/0.4597 (torch-bf16-compile) | 1.56 0.4542/0.2903 (torch-bf16-compile) |
| geomean f32 | | 0.727 (10) | 1.011 (10) | 0.922 (10) | 1.007 (10) | 0.807 (10) |
| geomean tf32 | | 1.326 (2) | 1.222 (2) | 1.727 (2) | 1.229 (2) | 2.015 (2) |
| geomean bf16w | | 1.573 (7) | 1.409 (7) | 1.569 (7) | 1.613 (7) | 0.968 (7) |
| **geomean** | | 1.029 (19) | 1.166 (19) | 1.198 (19) | 1.223 (19) | 0.950 (19) |
| win / even / loss | | 5/6/8 | 8/3/8 | 6/2/11 | 6/4/9 | 10/1/8 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.44** 0.0665/0.1516 (torch-compile) | **0.68** 0.1124/0.1649 (torch-compile) | **0.53** 0.07259/0.1372 (torch-eager) | **0.51** 0.04794/0.0932 (torch-eager) | **0.50** 0.04777/0.0946 (torch-compile) |
| convmc_8_64_56_3 | f32 | **0.66** 0.08194/0.1249 (torch-eager) | 1.87 0.1516/0.0809 (torch-eager) | 2.25 0.1958/0.087 (torch-eager) | 1.67 0.09405/0.0563 (torch-eager) | 1.76 0.1272/0.0724 (torch-eager) |
| ew_4096 | f32 | **0.37** 0.09188/0.2458 (torch-compile) | **0.65** 0.1559/0.2406 (chain-fused) | **0.76** 0.1178/0.1556 (torch-compile) | **0.34** 0.07694/0.2243 (chain-fused) | 1.38 0.1144/0.0828 (chain-fused) ±16% |
| ewconvew_2048_5 | f32 | **0.42** 0.07839/0.1864 (torch-compile) | **0.52** 0.1385/0.2673 (torch-compile) | **0.49** 0.08772/0.1772 (torch-compile) | **0.49** 0.0616/0.1249 (torch-compile) | **0.52** 0.06018/0.1167 (torch-compile) |
| fir_4194304_16 | f32 | **0.44** 0.04552/0.1044 (torch-eager) | **0.67** 0.07665/0.1137 (torch-compile) | **0.59** 0.05038/0.085 (torch-compile) | **0.55** 0.03517/0.0645 (torch-compile) | **0.65** 0.03974/0.0611 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.48** 0.05232/0.1085 (torch-compile) | **0.74** 0.1039/0.1403 (torch-compile) | **0.64** 0.06194/0.0963 (torch-compile) | **0.61** 0.04107/0.0676 (torch-compile) ±5% | **0.69** 0.04602/0.0664 (torch-compile) |
| convmc_8_64_56_3_tf32 | tf32 | 1.26 0.08121/0.0645 (torch-eager-tf32) | 1.87 0.1513/0.0809 (torch-eager-tf32) | 4.07 0.1915/0.0471 (torch-eager-tf32) | 2.40 0.1006/0.042 (torch-eager-tf32) | 3.16 0.1232/0.039 (torch-eager-tf32) |
| geomean f32 | | 0.460 (6) | 0.773 (6) | 0.744 (6) | 0.603 (6) | 0.811 (6) |
| geomean tf32 | | 1.259 (1) | 1.870 (1) | 4.067 (1) | 2.396 (1) | 3.160 (1) |
| **geomean** | | 0.531 (7) | 0.877 (7) | 0.949 (7) | 0.735 (7) | 0.985 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 5/0/2 | 5/0/2 | 4/0/3 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 1.05≈ 1.44/1.378 (torch-compile) | 1.04≈ 2.45/2.357 (chain-streams) | 1.10 3.246/2.949 (chain-lt) | 0.97≈ 0.8958/0.9247 (chain-cutlass) | 1.14 1.181/1.034 (chain-lt) |
| fanout_2048 | f32 | 1.08 0.9857/0.9133 (chain-streams) | 1.05 1.656/1.573 (chain-streams) | 1.15 2.253/1.967 (chain-streams) | 0.95≈ 0.6075/0.638 (chain-streams) | 1.15 0.7925/0.6865 (chain-streams) |
| indep_2048 | f32 | 1.09 0.9981/0.9175 (torch-compile) | 1.05≈ 1.65/1.575 (torch-compile) | 1.14 2.246/1.968 (torch-compile) | 0.95≈ 0.5937/0.6226 (chain-cutlass) | 1.16 0.8044/0.693 (chain-lt) |
| indep_512 | f32 | 1.08 0.03943/0.0366 (chain-streams) | 1.65 0.08637/0.0522 (chain-streams) | 1.32 0.07592/0.0573 (chain-streams) | 1.09≈ 0.04115/0.0379 (chain-streams) ±35% | 1.85 0.06053/0.0328 (chain-streams) ±7% |
| loop_4096 | f32 | 0.99≈ 28.43/28.75 (chain-lt) | 0.99≈ 46.87/47.55 (chain-lt) | 1.07 62.97/59.03 (torch-eager) | 0.96≈ 18.75/19.46 (chain-cutlass) | 1.12 23.81/21.26 (chain-lt) |
| loopmm_1024_8 | f32 | 1.10 0.6475/0.5898 (torch-compile) | 1.11 1.109/0.9984 (torch-compile) | 1.37 1.435/1.044 (torch-eager) | 1.18 0.4281/0.3625 (torch-compile) | 1.15 0.5319/0.4614 (torch-eager) |
| rt_mlp2_1024 | f32 | 1.01≈ 2.561/2.545 (torch-compile) | 1.03≈ 4.303/4.195 (torch-compile) | 1.43 6.834/4.774 (torch-compile) | 1.13 1.687/1.492 (torch-compile) | 1.05 2.146/2.043 (torch-compile) |
| rt_mlp2_2048 | f32 | 1.01≈ 16.27/16.04 (torch-compile) | 0.99≈ 26.37/26.7 (torch-compile) | 1.16 37.17/31.95 (torch-compile) | **0.89** 9.677/10.92 (torch-compile) | 1.05 12.05/11.44 (torch-compile) |
| twin_2048 | f32 | 1.02≈ 0.9371/0.9175 (torch-compile) | 1.12 1.636/1.462 (torch-compile) | 1.03≈ 2.021/1.964 (torch-eager) | 0.96≈ 0.5949/0.6175 (torch-compile) | 1.16 0.792/0.6845 (torch-compile) |
| **geomean** | | 1.046 (9) | 1.100 (9) | 1.190 (9) | 1.005 (9) | 1.187 (9) |
| win / even / loss | | 0/5/4 | 0/5/4 | 0/1/8 | 1/6/2 | 0/0/9 |

## bf16 real-world reference

Torch bf16 / bf16-compile (the layer in bfloat16, SDPA flash; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3195 (bf16-compile) · f32 2.53 | 0.4803 (bf16) · f32 2.57 | 0.2908 (bf16-compile) · f32 5.94 | 0.3154 (bf16-compile) · f32 2.01 | 0.095 (bf16-compile) · f32 7.50 |
| pj_m128_n4096_k14336 | 0.172 (bf16) · f32 2.66 | 0.2785 (bf16) · f32 2.61 | 0.126 (bf16-compile) · f32 7.28 | 0.1894 (bf16-compile) · f32 2.05 | 0.0596 (bf16-compile) · f32 6.34 |
| pj_m128_n4096_k4096 | 0.0532 (bf16-compile) · f32 2.72 | 0.085 (bf16) · f32 2.97 | 0.0573 (bf16-compile) · f32 5.34 | 0.0604 (bf16-compile) · f32 2.25 | 0.025 (bf16) · f32 5.90 |
| pj_m128_n6144_k4096 | 0.086 (bf16-compile) · f32 2.43 | 0.128 (bf16) · f32 2.72 | 0.0676 (bf16-compile) · f32 6.39 | 0.0819 (bf16-compile) · f32 2.33 | 0.0339 (bf16) · f32 5.95 |
| pj_m16_n18432_k7168 | 0.3092 (bf16-compile) · f32 1.94 | 0.3082 (bf16-compile) · f32 2.12 | 0.213 (bf16-compile) · f32 2.04 | 0.3399 (bf16-compile) · f32 1.87 | 0.0988 (bf16) · f32 2.35 |
| pj_m16_n2048_k7168 | 0.0389 (bf16-compile) · f32 1.20 | 0.044 (bf16-compile) · f32 2.73 | 0.044 (bf16-compile) · f32 2.35 | 0.0492 (bf16-compile) · f32 1.08 | 0.0231 (bf16) · f32 3.31 |
| pj_m16_n28672_k4096 | 0.2744 (bf16-compile) · f32 2.00 | 0.2734 (bf16-compile) · f32 2.17 | 0.1874 (bf16-compile) · f32 2.24 | 0.3082 (bf16-compile) · f32 1.86 | 0.0911 (bf16-compile) · f32 2.43 |
| pj_m16_n4096_k14336 | 0.1393 (bf16-compile) · f32 1.91 | 0.1423 (bf16-compile) · f32 2.19 | 0.1106 (bf16-compile) · f32 2.05 | 0.177 (bf16-compile) · f32 1.80 | 0.0562 (bf16) · f32 2.48 |
| pj_m16_n4096_k4096 | 0.043 (bf16-compile) · f32 1.23 | 0.0512 (bf16) · f32 2.68 | 0.044 (bf16-compile) · f32 2.28 | 0.0543 (bf16-compile) · f32 0.99 | 0.0243 (bf16) · f32 2.92 |
| pj_m16_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.51 | 0.0666 (bf16-compile) · f32 2.50 | 0.0604 (bf16-compile) · f32 2.14 | 0.0788 (bf16-compile) · f32 1.55 | 0.0302 (bf16) · f32 2.93 |
| pj_m1_n18432_k7168 | 0.3092 (bf16) · f32 1.96 | 0.3287 (bf16-compile) · f32 1.92 | 0.212 (bf16-compile) · f32 2.03 | 0.3031 (bf16-compile) · f32 2.07 | 0.0986 (bf16) · f32 2.29 |
| pj_m1_n2048_k7168 | 0.043 (bf16-compile) · f32 0.76 | 0.0471 (bf16-compile) · f32 2.20 | 0.043 (bf16) · f32 2.04 | 0.0503 (bf16-compile) · f32 0.66 | 0.0222 (bf16) · f32 2.37 |
| pj_m1_n28672_k4096 | 0.2744 (bf16-compile) · f32 2.03 | 0.2806 (bf16-compile) · f32 2.04 | 0.1833 (bf16-compile) · f32 2.09 | 0.2733 (bf16-compile) · f32 2.08 | 0.0922 (bf16-compile) · f32 2.22 |
| pj_m1_n4096_k14336 | 0.1413 (bf16) · f32 1.80 | 0.1485 (bf16-compile) · f32 2.04 | 0.1014 (bf16-compile) · f32 1.99 | 0.1516 (bf16-compile) · f32 1.98 | 0.0555 (bf16) · f32 2.31 |
| pj_m1_n4096_k4096 | 0.0471 (bf16-compile) · f32 1.06 | 0.0492 (bf16-compile) · f32 2.30 | 0.0399 (bf16-compile) · f32 2.34 | 0.0564 (bf16-compile) · f32 0.57 | 0.0228 (bf16) · f32 3.03 |
| pj_m1_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.58 | 0.0646 (bf16-compile) · f32 2.34 | 0.0553 (bf16-compile) · f32 2.10 | 0.0799 (bf16) · f32 1.06 | 0.0323 (bf16) · f32 2.15 |
| pj_m2048_n28672_k4096 | 3.947 (bf16-compile) · f32 2.99 | 6.386 (bf16-compile) · f32 3.29 | 1.979 (bf16) · f32 13.57 | 2.822 (bf16) · f32 3.07 | 0.6197 (bf16-compile) · f32 16.39 |
| pj_m2048_n4096_k14336 | 2.085 (bf16) · f32 3.09 | 3.777 (bf16) · f32 2.94 | 0.8991 (bf16) · f32 14.83 | 1.499 (bf16-compile) · f32 2.68 | 0.3059 (bf16) · f32 17.13 |
| pj_m2048_n4096_k4096 | 0.596 (bf16-compile) · f32 3.09 | 1.093 (bf16) · f32 2.75 | 0.3092 (bf16) · f32 12.51 | 0.4332 (bf16-compile) · f32 2.69 | 0.0941 (bf16) · f32 16.10 |
| pj_m2048_n6144_k4096 | 0.8765 (bf16-compile) · f32 3.03 | 1.395 (bf16-compile) · f32 3.29 | 0.4147 (bf16-compile) · f32 14.00 | 0.6011 (bf16-compile) · f32 2.91 | 0.1392 (bf16) · f32 16.03 |
| llm_blkdec_b1_l4096 | 0.5549 (bf16-compile) · f32 1.93 · bf16w 1.15 | 0.6837 (bf16-compile) · f32 1.66 · bf16w 1.04 | 0.4522 (bf16-compile) · f32 2.12 · bf16w 1.75 | 0.5718 (bf16-compile) · f32 2.61 · bf16w 1.04 | 0.4299 (bf16-compile) · f32 1.32 · bf16w 0.82 |
| llm_blkpre_t2048 | 7.984 (bf16-compile) · f32 3.44 | 13.61 (bf16-compile) · f32 3.38 | 4.481 (bf16-compile) · f32 12.47 | 5.614 (bf16-compile) · f32 3.31 | - |
| llm_dec_b1_l2048 | 0.1448 (bf16-compile) · f32 1.65 · bf16w 1.03 | 0.2483 (bf16-compile) · f32 1.12 · bf16w 0.75 | 0.1739 (bf16-compile) · f32 1.24 · bf16w 0.98 | 0.1716 (bf16-compile) · f32 1.41 · bf16w 0.90 | 0.2791 (bf16-compile) · f32 0.42 · bf16w 0.34 |
| llm_dec_b1_l8192 | 0.1863 (bf16-compile) · f32 1.78 · bf16w 1.30 | 0.316 (bf16-compile) · f32 1.20 · bf16w 0.91 | 0.1977 (bf16-compile) · f32 1.39 · bf16w 1.17 | 0.2076 (bf16-compile) · f32 1.66 · bf16w 1.24 | 0.2733 (bf16-compile) · f32 0.61 · bf16w 0.52 |
| llm_dec_b32_l2048 | 0.4589 (bf16-compile) · f32 2.25 · bf16w 2.09 | 0.7248 (bf16-compile) · f32 1.69 · bf16w 1.65 | 0.4934 (bf16) · f32 1.74 · bf16w 1.76 | 0.4565 (bf16-compile) · f32 2.47 · bf16w 2.24 | 0.2181 (bf16-compile) · f32 2.16 · bf16w 2.25 |
| llm_dec_b32_l8192 | 1.487 (bf16-compile) · f32 2.37 · bf16w 2.31 | 2.407 (bf16-compile) · f32 1.78 · bf16w 1.76 | 1.21 (bf16) · f32 2.10 · bf16w 2.10 | 1.356 (bf16-compile) · f32 2.78 · bf16w 2.70 | 0.5537 (bf16-compile) · f32 1.87 · bf16w 1.90 |
| llm_dec_b8_l2048 | 0.2174 (bf16-compile) · f32 2.00 · bf16w 1.57 | 0.3464 (bf16-compile) · f32 2.95 · bf16w 2.42 | 0.2289 (bf16-compile) · f32 1.74 · bf16w 1.52 | 0.2382 (bf16-compile) · f32 2.08 · bf16w 1.75 | 0.2476 (bf16-compile) · f32 0.84 · bf16w 0.77 |
| llm_dec_b8_l8192 | 0.4824 (bf16) · f32 2.21 · bf16w 2.02 | 0.7373 (bf16-compile) · f32 2.44 · bf16w 2.19 | 0.4227 (bf16) · f32 1.91 · bf16w 1.80 | 0.4631 (bf16-compile) · f32 2.44 · bf16w 2.27 | 0.2852 (bf16-compile) · f32 1.64 · bf16w 1.59 |
| llm_pre_t2048 | 1.921 (bf16-compile) · f32 4.50 | 3.22 (bf16-compile) · f32 4.49 | 1.046 (bf16-compile) · f32 14.46 | 1.329 (bf16-compile) · f32 4.75 | 0.4071 (bf16-compile) · f32 15.12 |
| llm_pre_t8192 | 11.33 (bf16-compile) · f32 7.24 | 18.12 (bf16-compile) · f32 95.53 | 6.524 (bf16-compile) · f32 219.28 | 7.848 (bf16-compile) · f32 95.98 | 2.007 (bf16-compile) · f32 412.44 |

## Verdict flips vs the mixed-precision gap tables

gap.py's rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables (gap-<part>.md, gap.md) row for row. Rows a part's gap table did not cover (the RTX 5080's model-scale rows, reported in model-scale.md) are judged against that rule applied to the same data.

| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |
|---|---|---|---|---|
| rtx-5080 | llm_blkdec_b1_l4096 | 0.5549 (torch-bf16-compile) 1.93 loss | 1.074 (torch-compile) 1.00 even | old best a bf16 leg |
| rtx-5080 | llm_blkpre_t2048 | 7.984 (torch-bf16-compile) 3.44 loss | 27.74 (torch-compile) 0.99 even | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l2048 | 0.1448 (torch-bf16-compile) 1.65 loss | 0.2575 (torch-compile) 0.93 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l8192 | 0.1863 (torch-bf16-compile) 1.78 loss | 0.3444 (torch-compile) 0.96 even | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l2048 | 0.4589 (torch-bf16-compile) 2.25 loss | 2.277 (torch-compile) 0.45 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l8192 | 1.487 (torch-bf16-compile) 2.37 loss | 9.523 (torch-compile) 0.37 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l2048 | 0.2174 (torch-bf16-compile) 2.00 loss | 0.6782 (torch-compile) 0.64 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l8192 | 0.4824 (torch-bf16) 2.21 loss | 2.466 (torch-compile) 0.43 win | old best a bf16 leg |
| rtx-5080 | llm_pre_t2048 | 1.921 (torch-bf16-compile) 4.50 loss | 8.311 (torch-compile) 1.04 even | old best a bf16 leg |
| rtx-5080 | llm_pre_t8192 | 11.33 (torch-bf16-compile) 7.24 loss | 84.25 (torch-compile) 0.97 even | old best a bf16 leg |
| rtx-5080 | pj_m16_n18432_k7168 | 0.3092 (torch-bf16-compile) 1.94 loss | 0.5878 (cublas-fresh) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n2048_k7168 | 0.0389 (torch-bf16-compile) 1.20 loss | 0.0492 (cublas-fresh) 0.95 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n4096_k14336 | 0.1393 (torch-bf16-compile) 1.91 loss | 0.2765 (torch-compile) 0.96 even | old best a bf16 leg |
| rtx-5080 | pj_m16_n4096_k4096 | 0.043 (torch-bf16-compile) 1.23 loss | 0.0553 (cublas-fresh) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n18432_k7168 | 0.3092 (torch-bf16) 1.96 loss | 0.6103 (torch-compile) 0.99 even | old best a bf16 leg |
| rtx-5080 | pj_m1_n2048_k7168 | 0.043 (torch-bf16-compile) 0.76 win | 0.0266 (cublas-fresh) 1.24 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096 | 3.947 (torch-bf16-compile) 2.99 loss | 12.23 (lt-fresh) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k14336 | 2.085 (torch-bf16) 3.09 loss | 6.42 (cublas-fresh) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n6144_k4096 | 0.8765 (torch-bf16-compile) 3.03 loss | 2.578 (torch-compile) 1.03 even | old best a bf16 leg |
| rtx-3090 | llm_blkdec_b1_l4096 | 0.6837 (torch-bf16-compile) 1.66 loss | 1.203 (torch-compile) 0.94 win | old best a bf16 leg |
| rtx-3090 | llm_blkpre_t2048 | 13.61 (torch-bf16-compile) 3.38 loss | 49.42 (torch-compile) 0.93 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l2048 | 0.2483 (torch-bf16-compile) 1.12 loss | 0.3642 (torch-compile) 0.76 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l8192 | 0.316 (torch-bf16-compile) 1.20 loss | 0.4557 (torch-compile) 0.83 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l2048 | 0.7248 (torch-bf16-compile) 1.69 loss | 2.696 (torch-compile) 0.45 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192 | 2.407 (torch-bf16-compile) 1.78 loss | 10.83 (torch-compile) 0.40 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l8192 | 0.7373 (torch-bf16-compile) 2.44 loss | 2.939 (torch-compile) 0.61 win | old best a bf16 leg |
| rtx-3090 | pj_m128_n28672_k4096 | 0.4803 (torch-bf16) 2.57 loss | 1.265 (torch-compile) 0.98 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n18432_k7168 | 0.3287 (torch-bf16-compile) 1.92 loss | 0.6062 (lt-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k4096 | 1.093 (torch-bf16) 2.75 loss | 2.869 (torch-compile) 1.05 even | old best a bf16 leg |
| a100 | llm_blkpre_t2048 | 4.481 (torch-bf16-compile) 12.47 loss | 55.18 (torch-compile) 1.01 even | old best a bf16 leg |
| a100 | llm_dec_b1_l2048 | 0.1739 (torch-bf16-compile) 1.24 loss | 0.2488 (torch-compile) 0.87 win | old best a bf16 leg |
| a100 | llm_dec_b1_l8192 | 0.1977 (torch-bf16-compile) 1.39 loss | 0.3942 (torch-compile) 0.70 win | old best a bf16 leg |
| a100 | llm_dec_b32_l2048 | 0.4934 (torch-bf16) 1.74 loss | 1.842 (torch-compile) 0.47 win | old best a bf16 leg |
| a100 | llm_dec_b32_l8192 | 1.21 (torch-bf16) 2.10 loss | 6.525 (torch-compile) 0.39 win | old best a bf16 leg |
| a100 | llm_dec_b8_l2048 | 0.2289 (torch-bf16-compile) 1.74 loss | 0.6228 (torch-compile) 0.64 win | old best a bf16 leg |
| a100 | llm_dec_b8_l8192 | 0.4227 (torch-bf16) 1.91 loss | 1.823 (torch-compile) 0.44 win | old best a bf16 leg |
| a100 | llm_pre_t2048 | 1.046 (torch-bf16-compile) 14.46 loss | 15.01 (torch-compile) 1.01 even | old best a bf16 leg |
| a100 | pj_m128_n28672_k4096 | 0.2908 (torch-bf16-compile) 5.94 loss | 1.781 (cublas) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m16_n18432_k7168 | 0.213 (torch-bf16-compile) 2.04 loss | 0.4454 (torch-compile) 0.98 even | old best a bf16 leg |
| a100 | pj_m16_n28672_k4096 | 0.1874 (torch-bf16-compile) 2.24 loss | 0.426 (torch-compile) 0.99 even | old best a bf16 leg |
| a100 | pj_m16_n4096_k14336 | 0.1106 (torch-bf16-compile) 2.05 loss | 0.2509 (lt) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m2048_n28672_k4096 | 1.979 (torch-bf16) 13.57 loss | 26.02 (torch-compile) 1.03 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k14336 | 0.8991 (torch-bf16) 14.83 loss | 12.88 (torch-eager) 1.03 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k4096 | 0.3092 (torch-bf16) 12.51 loss | 3.709 (cublas-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m2048_n6144_k4096 | 0.4147 (torch-bf16-compile) 14.00 loss | 6.147 (lt) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_blkpre_t2048 | 5.614 (torch-bf16-compile) 3.31 loss | 18.98 (torch-compile) 0.98 even | old best a bf16 leg |
| rtx-4090 | llm_dec_b1_l2048 | 0.1716 (torch-bf16-compile) 1.41 loss | 0.2737 (torch-compile) 0.88 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b1_l8192 | 0.2076 (torch-bf16-compile) 1.66 loss | 0.3384 (torch-compile) 1.02 even | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l2048 | 0.4565 (torch-bf16-compile) 2.47 loss | 2.309 (torch-compile) 0.49 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l8192 | 1.356 (torch-bf16-compile) 2.78 loss | 9.405 (torch-compile) 0.40 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l2048 | 0.2382 (torch-bf16-compile) 2.08 loss | 0.7128 (torch-compile) 0.70 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l8192 | 0.4631 (torch-bf16-compile) 2.44 loss | 2.532 (torch-compile) 0.45 win | old best a bf16 leg |
| rtx-4090 | llm_pre_t2048 | 1.329 (torch-bf16-compile) 4.75 loss | 6.063 (torch-compile) 1.04 even | old best a bf16 leg |
| rtx-4090 | pj_m128_n28672_k4096 | 0.3154 (torch-bf16-compile) 2.01 loss | 0.6492 (lt-fresh) 0.98 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n4096_k4096_tf32 | 0.1085 (torch-compile-tf32) 0.88 win | 0.0666 (cublas-tf32-fresh) 1.43 loss | best now a gemm_lib leg |
| rtx-4090 | pj_m16_n18432_k7168 | 0.3399 (torch-bf16-compile) 1.87 loss | 0.6113 (lt-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m16_n2048_k7168 | 0.0492 (torch-bf16-compile) 1.08 even | 0.0307 (cublas-fresh) 1.74 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m16_n28672_k4096 | 0.3082 (torch-bf16-compile) 1.86 loss | 0.5509 (cublas-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m16_n4096_k4096 | 0.0543 (torch-bf16-compile) 0.99 even | 0.0348 (cublas-fresh) 1.54 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n2048_k7168 | 0.0503 (torch-bf16-compile) 0.66 win | 0.0276 (lt-fresh) 1.20 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n4096_k4096 | 0.0564 (torch-bf16-compile) 0.57 win | 0.0195 (lt-fresh) 1.66 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n28672_k4096 | 2.822 (torch-bf16) 3.07 loss | 8.29 (lt) 1.05 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k14336 | 1.499 (torch-bf16-compile) 2.68 loss | 4.236 (torch-compile) 0.95 win | old best a bf16 leg |
| rtx-4090 | pj_m2048_n4096_k4096 | 0.4332 (torch-bf16-compile) 2.69 loss | 1.173 (cublas-fresh) 0.99 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n6144_k4096 | 0.6011 (torch-bf16-compile) 2.91 loss | 1.791 (cublas) 0.98 even | old best a bf16 leg + best now a gemm_lib leg |
| h100 | llm_dec_b32_l2048 | 0.2181 (torch-bf16-compile) 2.16 loss | 1.182 (torch-compile) 0.40 win | old best a bf16 leg |
| h100 | llm_dec_b32_l8192 | 0.5537 (torch-bf16-compile) 1.87 loss | 3.991 (torch-compile) 0.26 win | old best a bf16 leg |
| h100 | llm_dec_b8_l8192 | 0.2852 (torch-bf16-compile) 1.64 loss | 1.147 (torch-compile) 0.41 win | old best a bf16 leg |

## Command

```
python3 benches/local/matrix.py \
  rtx-5080=.claude/raw-evidence/local-rtx-5080/model-scale/resA2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB1/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/onto-device/resE/results-rtx-5080.txt@8bba2a50,.claude/raw-evidence/local-rtx-5080/device-geometry/v3/model/results-rtx-5080.txt@5b6c0a29,.claude/raw-evidence/local-rtx-5080/large-gemm/resM/results-rtx-5080.txt@7a2b67bf,.claude/raw-evidence/local-rtx-5080/half-precision/run1/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/half-precision/run2/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/long-prefill/resL/results-rtx-5080.txt@c8b83028,.claude/raw-evidence/local-rtx-5080/long-prefill/suite/results-rtx-5080.txt@c03a52a3 \
  rtx-3090=.claude/raw-evidence/cross-arch/60f0f8ce/rtx-3090/results-rtx-3090.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/rtx-3090/results-rtx-3090-pass2.txt@60f0f8ce \
  a100=.claude/raw-evidence/cross-arch/60f0f8ce/a100/results-a100.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/a100/results-a100-pass2.txt@60f0f8ce \
  rtx-4090=.claude/raw-evidence/cross-arch/60f0f8ce/rtx-4090/results-rtx-4090.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/rtx-4090/results-rtx-4090-pass2.txt@60f0f8ce \
  h100=.claude/raw-evidence/cross-arch/60f0f8ce/h100/results-h100.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/h100/results-h100-pass2.txt@60f0f8ce
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
