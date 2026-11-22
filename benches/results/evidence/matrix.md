# Mapal vs the fastest competitor at the same precision

Built 2026-11-23 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations) vs the torch bf16 legs (bf16 throughout). The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch's pick, `-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials' matmul / softmax / fused attention, autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. **Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang as servers. The rented parts' columns carry their 60f0f8ce legs until a rented round runs the new legs: their new families read `not run`.

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | f7a74cdb | competitors/resS@f7a74cdb: 43; competitors/resM@f7a74cdb: 43 |
| rtx-3090 | 60f0f8ce | 60f0f8ce/rtx-3090@60f0f8ce: 83 |
| a100 | 60f0f8ce | 60f0f8ce/a100@60f0f8ce: 83 |
| rtx-4090 | 60f0f8ce | 60f0f8ce/rtx-4090@60f0f8ce: 83 |
| h100 | 60f0f8ce | 60f0f8ce/h100@60f0f8ce: 83 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 1.156 (31) | 1.178 (31) | 1.213 (31) | 1.195 (31) | 1.497 (31) |
| MLP chains | 1.097 (6) | 1.160 (6) | 1.236 (6) | 1.030 (6) | 1.615 (6) |
| Attention / softmax / long rows | 0.841 (14) | 0.924 (11) | 1.201 (11) | 0.922 (11) | 0.908 (11) |
| LLM decode / prefill / block (f32, tf32, bf16w) | 1.077 (19) | 1.166 (19) | 1.198 (19) | 1.223 (19) | 0.950 (19) |
| Conv / FIR / elementwise | 0.525 (7) | 0.877 (7) | 0.949 (7) | 0.735 (7) | 0.985 (7) |
| Fan-in/out, loops, CPU+GPU | 1.061 (9) | 1.100 (9) | 1.190 (9) | 1.005 (9) | 1.187 (9) |
| all f32 rows | 0.905 (66) | 1.041 (63) | 1.073 (63) | 0.996 (63) | 1.096 (63) |
| all tf32 rows | 1.303 (13) | 1.262 (13) | 1.628 (13) | 1.300 (13) | 2.034 (13) |
| all bf16w rows | 1.582 (7) | 1.409 (7) | 1.569 (7) | 1.613 (7) | 0.968 (7) |
| **all rows** | 1.001 (86) | 1.101 (83) | 1.182 (83) | 1.082 (83) | 1.195 (83) |
| rows timed on every part (83) | 0.996 (83) | 1.101 (83) | 1.182 (83) | 1.082 (83) | 1.195 (83) |
| win / even / loss | 18/22/46 | 17/13/53 | 16/13/54 | 17/26/40 | 18/1/64 |

Geomean cells: value (rows). Parts time different row sets (the RTX 5080 adds 5080-only rows, the rented parts the full model-scale set); compare columns on the "every part" row.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 1.04≈ 3.675/3.547 (torch-compile-ro) | 0.96≈ 5.596/5.828 (chain-lt) | 1.08 7.88/7.296 (chain-lt) | 0.99≈ 2.407/2.434 (chain-lt) | 1.13 2.999/2.663 (chain-lt) |
| mm_16x2048x7168 | f32 | 1.10 0.0879/0.0799 (cublas) | 1.02≈ 0.09005/0.08805 (cublas) | 1.02≈ 0.09215/0.0906 (cublas) ±6% | 1.13 0.1084/0.0963 (cublas) | 1.84 0.09575/0.05195 (cublas) |
| mm_4096x4096x4096 | f32 | 1.04≈ 3.812/3.652 (cublas) | 1.11 7.115/6.409 (cublas) | 1.04≈ 7.681/7.402 (cublas) | 0.97≈ 2.393/2.477 (cublas) | 1.18 3.136/2.658 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.08 0.8088/0.7516 (lt) | 0.98≈ 1.235/1.265 (torch-compile) | 0.97≈ 1.727/1.781 (cublas) | 0.98≈ 0.6337/0.6492 (lt-fresh) | 1.07 0.7124/0.6633 (lt-fresh) |
| pj_m128_n4096_k14336 | f32 | 1.13 0.4547/0.4014 (cublas-fresh) | 1.08 0.7261/0.6748 (torch-eager) | 1.08 0.9178/0.8509 (cublas-fresh) | 1.09 0.3881/0.3574 (lt-fresh) | 1.21 0.378/0.3131 (lt) |
| pj_m128_n4096_k4096 | f32 | 1.08 0.1456/0.1352 (cublas-fresh) | 1.23 0.2524/0.2048 (lt) | 1.18 0.3061/0.2601 (cublas) | 1.33 0.1357/0.1024 (lt-fresh) | 1.51 0.1476/0.098 (lt) |
| pj_m128_n6144_k4096 | f32 | 1.16 0.2095/0.1802 (cublas-fresh) | 1.20 0.3484/0.2908 (lt) | 1.16 0.4317/0.3727 (lt-fresh) | 1.22 0.1911/0.1567 (lt-fresh) | 1.36 0.2016/0.1486 (lt-fresh) |
| pj_m16_n18432_k7168 | f32 | 1.02≈ 0.6025/0.5878 (cublas-fresh) | 1.07 0.653/0.6113 (torch-compile) | 0.98≈ 0.4347/0.4454 (torch-compile) | 1.04≈ 0.6352/0.6113 (lt-fresh) | 1.12 0.2317/0.2072 (torch-compile) |
| pj_m16_n2048_k7168 | f32 | **0.94** 0.04612/0.0492 (cublas-fresh) | 1.43 0.1199/0.084 (torch-compile) | 1.19 0.1035/0.087 (cublas-fresh) | 1.74 0.05337/0.0307 (cublas-fresh) ±16% | 1.58 0.07647/0.0484 (lt-fresh) |
| pj_m16_n28672_k4096 | f32 | 1.03≈ 0.5395/0.5243 (lt-fresh) | 1.09 0.5942/0.5458 (torch-compile) | 0.99≈ 0.4203/0.426 (torch-compile) | 1.04≈ 0.5731/0.5509 (cublas-fresh) | 1.26 0.2216/0.1761 (torch-compile) |
| pj_m16_n4096_k14336 | f32 | 0.97≈ 0.2667/0.2744 (torch-compile) | 1.13 0.3119/0.2755 (torch-compile) | **0.90** 0.2269/0.2509 (lt) | 1.05 0.318/0.3021 (cublas-fresh) | 1.20 0.1393/0.1159 (lt) |
| pj_m16_n4096_k4096 | f32 | **0.94** 0.05224/0.0553 (cublas-fresh) | 1.60 0.1372/0.086 (torch-compile) | 1.13 0.1004/0.0891 (cublas-fresh) | 1.54 0.0535/0.0348 (cublas-fresh) | 1.48 0.07107/0.0479 (lt-fresh) |
| pj_m16_n6144_k4096 | f32 | 1.14 0.1009/0.0881 (lt-fresh) | 1.34 0.1664/0.1239 (torch-compile) | 1.11 0.1291/0.1167 (torch-compile) | 1.22 0.1221/0.1004 (cublas-fresh) | 1.55 0.08844/0.0572 (torch-compile) ±6% |
| pj_m1_n18432_k7168 | f32 | 0.99≈ 0.6051/0.6083 (torch-compile) | 1.04≈ 0.6312/0.6062 (lt-fresh) | 1.10 0.4308/0.3922 (lt-fresh) | 1.17 0.6274/0.5366 (lt-fresh) | 1.21 0.226/0.1868 (lt-fresh) |
| pj_m1_n2048_k7168 | f32 | 1.26 0.03351/0.0266 (cublas-fresh) | 1.39 0.1038/0.0748 (lt-fresh) | 1.99 0.08766/0.044 (cublas-fresh) | 1.20 0.0332/0.0276 (lt-fresh) ±6% | 2.17 0.05269/0.0243 (cublas-fresh) ±7% |
| pj_m1_n28672_k4096 | f32 | 1.12 0.5624/0.5038 (lt-fresh) | 1.06 0.5718/0.5376 (lt) | 1.10 0.3827/0.3492 (torch-compile) | 1.18 0.5685/0.4823 (lt-fresh) | 1.24 0.2046/0.1649 (lt-fresh) |
| pj_m1_n4096_k14336 | f32 | 1.10 0.2547/0.2314 (cublas-fresh) | 1.09 0.3025/0.2765 (lt) | 1.13 0.2021/0.1782 (cublas-fresh) | 1.47 0.2995/0.2038 (lt-fresh) | 1.56 0.1285/0.0824 (cublas-fresh) ±8% |
| pj_m1_n4096_k4096 | f32 | 1.59 0.03263/0.0205 (lt-fresh) ±10% | 1.31 0.113/0.086 (lt) | 1.75 0.0932/0.0532 (cublas-fresh) | 1.66 0.03228/0.0195 (lt-fresh) | 2.89 0.06915/0.0239 (lt-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.80 0.09199/0.0512 (cublas-fresh) | 1.23 0.151/0.1229 (lt-fresh) | 1.49 0.116/0.0778 (lt-fresh) | 1.43 0.08477/0.0592 (lt-fresh) | 1.90 0.06935/0.0365 (cublas-fresh) |
| pj_m2048_n28672_k4096 | f32 | 1.03≈ 12.63/12.21 (cublas-fresh) | 1.11 21.02/18.86 (torch-compile) | 1.03≈ 26.86/26.02 (torch-compile) | 1.05≈ 8.673/8.29 (lt) | 1.10 10.16/9.204 (torch-compile) |
| pj_m2048_n4096_k14336 | f32 | 0.99≈ 6.427/6.498 (torch-eager) | 1.06 11.09/10.47 (torch-compile) | 1.03≈ 13.34/12.88 (torch-eager) | **0.95** 4.017/4.236 (torch-compile) | 1.16 5.24/4.533 (torch-eager) |
| pj_m2048_n4096_k4096 | f32 | 1.04≈ 1.837/1.772 (torch-compile-cg) | 1.05≈ 3.01/2.869 (torch-compile) | 1.04≈ 3.867/3.709 (cublas-fresh) | 0.99≈ 1.167/1.173 (cublas-fresh) | 1.15 1.515/1.318 (torch-compile) |
| pj_m2048_n6144_k4096 | f32 | 1.02≈ 2.648/2.602 (torch-compile-cg) | 1.10 4.584/4.177 (cublas) | **0.94** 5.804/6.147 (lt) | 0.98≈ 1.751/1.791 (cublas) | 1.13 2.231/1.977 (torch-eager) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.18 0.6985/0.5898 (cublas-tf32-fresh) | 1.15 1.071/0.9298 (cublas-tf32) | 1.19 0.5376/0.4526 (lt-tf32) | 1.01≈ 0.5749/0.5704 (cublas-tf32-fresh) | 1.36 0.2515/0.1856 (cublas-tf32) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.48 0.4568/0.3092 (cublas-tf32-fresh) | 1.16 0.7229/0.6216 (torch-compile-tf32) | 2.02 0.4672/0.2314 (lt-tf32-fresh) | 1.24 0.38/0.3072 (lt-tf32-fresh) | 2.31 0.2466/0.1067 (lt-tf32) |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.66 0.1355/0.0819 (lt-tf32-fresh) | 1.31 0.219/0.1669 (lt-tf32-fresh) | 1.79 0.1482/0.0829 (cublas-tf32-fresh) | 1.43 0.0955/0.0666 (cublas-tf32-fresh) | 1.94 0.08557/0.0442 (cublas-tf32-fresh) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.90 0.2368/0.1249 (lt-tf32-fresh) | 1.89 0.4239/0.2243 (lt-tf32-fresh) | 1.57 0.1612/0.1024 (cublas-tf32-fresh) | 1.58 0.1808/0.1147 (cublas-tf32-fresh) | 2.94 0.1566/0.0532 (lt-tf32-fresh) |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.12 9.038/8.062 (lt-tf32-fresh) | 1.16 14.85/12.83 (torch-eager-tf32) | 1.43 5.928/4.153 (cublas-tf32-fresh) | 1.49 8.235/5.542 (torch-compile-tf32) | 1.57 2.037/1.294 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.15 4.898/4.244 (lt-tf32-fresh) | 1.19 8.366/7.007 (torch-eager-tf32) | 1.26 2.709/2.147 (cublas-tf32) | 1.19 3.31/2.788 (cublas-tf32-fresh) | 1.68 0.9885/0.5878 (torch-eager-tf32) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.22 1.416/1.161 (lt-tf32) | 1.18 2.408/2.038 (torch-compile-tf32) | 1.37 0.8018/0.5857 (lt-tf32) | 1.11 0.952/0.8581 (lt-tf32) | 1.77 0.3111/0.1759 (cublas-tf32-fresh) |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.15 2.017/1.748 (cublas-tf32-fresh) | 1.19 3.413/2.871 (torch-compile-tf32) | 1.53 1.246/0.8161 (torch-eager-tf32) | 1.19 1.423/1.196 (torch-eager-tf32) | 1.77 0.4608/0.2603 (lt-tf32-fresh) |
| geomean f32 | | 1.100 (23) | 1.150 (23) | 1.127 (23) | 1.171 (23) | 1.386 (23) |
| geomean tf32 | | 1.333 (8) | 1.262 (8) | 1.498 (8) | 1.265 (8) | 1.868 (8) |
| **geomean** | | 1.156 (31) | 1.178 (31) | 1.213 (31) | 1.195 (31) | 1.497 (31) |
| win / even / loss | | 2/10/19 | 0/5/26 | 2/8/21 | 1/9/21 | 0/0/31 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.10 1.943/1.765 (chain-lt2) | 1.09 3.253/2.98 (chain-lt2) | 1.17 4.399/3.76 (chain-streams) | 0.96≈ 1.184/1.232 (chain-cutlass) | 1.11 1.511/1.36 (chain-streams) |
| mlp_2048 | f32 | 1.03≈ 0.9414/0.9114 (chain-lt) | 1.06 1.621/1.532 (torch-compile) | 1.13 2.222/1.964 (chain-lt) | 0.96≈ 0.5925/0.6195 (chain-cutlass) | 1.11 0.761/0.6858 (chain-lt) |
| mlp_512 | f32 | 1.16 0.038/0.0328 (torch-compile) | 1.50 0.07958/0.0532 (chain-lt) | 1.24 0.07773/0.0625 (chain-lt) | 1.03≈ 0.03892/0.0379 (chain-lt) | 2.21 0.07836/0.0354 (chain-lt) ±12% |
| smlp_4096_256_1024 | f32 | 1.10 0.1645/0.1495 (torch-compile) | 1.23 0.2825/0.2294 (torch-compile) | 1.18 0.3455/0.2918 (chain-eager) | 0.99≈ 0.1083/0.1096 (torch-compile) | 1.21 0.1353/0.1116 (chain-lt) ±6% |
| mlp2_2048_tf32 | tf32 | 1.10 1.668/1.514 (torch-compile-ro-tf32) | 1.08 2.834/2.626 (torch-eager-tf32) | 1.28 0.932/0.7301 (torch-compile-tf32) | 1.14 0.9645/0.8489 (torch-compile-tf32) | 2.19 0.5347/0.2437 (torch-eager-tf32) |
| mlp_2048_tf32 | tf32 | 1.09 0.8298/0.7598 (torch-compile-ro-tf32) | 1.06 1.395/1.317 (torch-eager-tf32) | 1.43 0.5081/0.3553 (torch-eager-tf32) | 1.13 0.4891/0.4342 (torch-eager-tf32) | 2.45 0.2734/0.1117 (torch-eager-tf32) |
| geomean f32 | | 1.097 (4) | 1.208 (4) | 1.182 (4) | 0.983 (4) | 1.349 (4) |
| geomean tf32 | | 1.097 (2) | 1.069 (2) | 1.351 (2) | 1.131 (2) | 2.317 (2) |
| **geomean** | | 1.097 (6) | 1.160 (6) | 1.236 (6) | 1.030 (6) | 1.615 (6) |
| win / even / loss | | 0/1/5 | 0/0/6 | 0/0/6 | 0/4/2 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.36 0.03914/0.0287 (torch-compile-cg) | 1.83 0.1067/0.0584 (chain-fused) | 1.32 0.06356/0.0481 (torch-compile) | 1.47 0.04645/0.0317 (torch-compile) | 2.40 0.09197/0.0384 (chain-fused) ±9% |
| attn_2048_128 | f32 | 1.25 0.1177/0.0942 (torch-sdpa-compile-cg) | - | - | - | - |
| attn_2048_64 | f32 | 1.14 0.07953/0.0696 (torch-compile-ro) | - | - | - | - |
| causal_1024_256 | f32 | **0.40** 15.23/38.23 (torch-compile-cg) | **0.13** 25.04/199.7 (torch-compile) | **0.51** 36.8/72.58 (torch-compile) | **0.18** 18.38/101.2 (torch-compile) | **0.19** 38.36/201.1 (torch-compile) ±13% |
| decode_1024_256 | f32 | **0.54** 8.635/16.04 (chain-cache) | **0.79** 26.1/33.22 (chain-cache) | 0.96≈ 27.39/28.66 (chain-cache) | **0.75** 14.73/19.68 (chain-cache) | **0.73** 28.07/38.44 (chain-cache) ±6% |
| gen_1024_512_256 | f32 | **0.37** 7.685/20.84 (torch-compile-cg) | **0.13** 12.81/100.7 (torch-compile) | **0.48** 18.53/38.55 (torch-compile) | **0.18** 9.32/50.58 (torch-compile) | **0.13** 12.2/96.46 (torch-compile) ±10% |
| lsm_1_131072 | f32 | **0.70** 0.01582/0.0225 (torch-compile-cg) ±8% | 11.18 0.3208/0.0287 (torch-compile) | 10.80 0.3543/0.0328 (torch-compile) | 8.39 0.2232/0.0266 (torch-sdpa-compile) | 5.81 0.3387/0.0583 (torch-sdpa) |
| lsm_32_8192 | f32 | 1.51 0.01538/0.0102 (torch-compile) ±23% | 2.69 0.04146/0.0154 (torch-compile) | 1.67 0.0307/0.0184 (torch-compile) | 2.44 0.02688/0.011 (torch-compile) ±6% | 2.52 0.04112/0.0163 (torch-sdpa) ±8% |
| lsm_4_32768 | f32 | 1.16 0.0166/0.0143 (torch-compile) | 3.56 0.08762/0.0246 (torch-sdpa) | 2.38 0.0732/0.0307 (torch-sdpa) | 2.97 0.05782/0.0195 (torch-sdpa) | 2.83 0.06518/0.023 (torch-sdpa) ±15% |
| prefill_1024_512_256 | f32 | **0.40** 7.743/19.26 (torch-compile-cg) | **0.13** 13.16/98.09 (torch-compile) | **0.49** 18.87/38.52 (torch-compile) | **0.19** 9.51/50.58 (torch-compile) | **0.13** 13.2/99.58 (torch-compile) |
| rowsum_2048 | f32 | 1.00≈ 0.4593/0.4608 (chain-eager) | 1.10 0.8528/0.7731 (torch-compile) | 1.04≈ 1.035/0.9964 (chain-eager) | 0.98≈ 0.3122/0.3187 (torch-compile) | 1.19 0.4147/0.3478 (torch-compile) |
| rowsum_4096 | f32 | 1.07 3.733/3.486 (torch-compile) | 1.07 6.225/5.807 (chain-eager) | 1.06 7.749/7.339 (torch-compile) | 1.00≈ 2.379/2.387 (torch-eager) | 1.12 2.996/2.683 (torch-eager) |
| softmax_rows_2048 | f32 | 1.01≈ 0.4727/0.469 (torch-compile) | - | - | - | - |
| softmax_rows_4096 | f32 | 1.08 3.803/3.529 (torch-compile-ro) | 1.08 6.444/5.964 (torch-compile) | 1.06 7.817/7.378 (torch-compile) | 1.00≈ 2.442/2.451 (torch-compile) | 1.12 3.024/2.712 (torch-compile) |
| **geomean** | | 0.841 (14) | 0.924 (11) | 1.201 (11) | 0.922 (11) | 0.908 (11) |
| win / even / loss | | 5/2/7 | 4/0/7 | 3/2/6 | 4/3/4 | 4/0/7 |

## LLM decode / prefill / block (f32, tf32, bf16w)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 1.01≈ 1.071/1.066 (torch-compile) | **0.94** 1.133/1.203 (torch-compile) | 1.14 0.9588/0.8421 (torch-compile) | 1.42 1.494/1.053 (torch-compile) | 1.25 0.5689/0.4535 (torch-compile) |
| llm_blkpre_t2048 | f32 | 1.02≈ 27.28/26.67 (liger) | **0.93** 45.99/49.42 (torch-compile) | 1.01≈ 55.87/55.18 (torch-compile) | 0.98≈ 18.55/18.98 (torch-compile) | 1.02≈ 21.58/21.13 (torch-eager) |
| llm_dec_b1_l2048 | f32 | **0.95** 0.24/0.2532 (torch-compile-ro) | **0.76** 0.2774/0.3642 (torch-compile) | **0.87** 0.2153/0.2488 (torch-compile) | **0.88** 0.242/0.2737 (torch-compile) | **0.47** 0.1158/0.2469 (torch-compile) |
| llm_dec_b1_l8192 | f32 | 0.97≈ 0.3324/0.342 (torch-compile) | **0.83** 0.3777/0.4557 (torch-compile) | **0.70** 0.2756/0.3942 (torch-compile) | 1.02≈ 0.3439/0.3384 (torch-compile) | **0.49** 0.1659/0.3386 (torch-compile) |
| llm_dec_b32_l2048 | f32 | **0.46** 1.033/2.256 (torch-compile) | **0.45** 1.224/2.696 (torch-compile) | **0.47** 0.8594/1.842 (torch-compile) | **0.49** 1.129/2.309 (torch-compile) | **0.40** 0.4715/1.182 (torch-compile) |
| llm_dec_b32_l8192 | f32 | **0.37** 3.506/9.484 (torch-compile) | **0.40** 4.282/10.83 (torch-compile) | **0.39** 2.535/6.525 (torch-compile) | **0.40** 3.765/9.405 (torch-compile) | **0.26** 1.038/3.991 (torch-compile) |
| llm_dec_b8_l2048 | f32 | **0.66** 0.4385/0.6685 (torch-compile) | 1.15 1.023/0.8885 (torch-compile) | **0.64** 0.3983/0.6228 (torch-compile) | **0.70** 0.4967/0.7128 (torch-compile) | **0.44** 0.207/0.469 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.43** 1.067/2.468 (torch-compile-cg) | **0.61** 1.801/2.939 (torch-compile) | **0.44** 0.8091/1.823 (torch-compile) | **0.45** 1.132/2.532 (torch-compile) | **0.41** 0.4684/1.147 (torch-compile) |
| llm_pre_t2048 | f32 | 1.19 8.56/7.2 (liger) | 1.16 14.46/12.44 (torch-compile) | 1.01≈ 15.13/15.01 (torch-compile) | 1.04≈ 6.318/6.063 (torch-compile) | 1.12 6.153/5.517 (torch-compile) |
| llm_pre_t8192 | f32 | 1.53 80.68/52.83 (liger) | 13.74 1731/126 (torch-compile) | 12.30 1431/116.3 (torch-compile) | 13.50 753.2/55.8 (torch-compile) | 19.06 827.9/43.43 (torch-compile) |
| llm_blkpre_t2048_tf32 | tf32 | 1.21 21.83/18.08 (triton-tf32) | 1.05≈ 33.89/32.39 (torch-compile-tf32) | 1.47 16.37/11.13 (torch-compile-tf32) | 1.25 16.84/13.46 (torch-compile-tf32) | 1.91 6.719/3.51 (torch-compile-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.64 7.455/4.537 (triton-tf32) | 1.43 12.75/8.922 (torch-compile-tf32) | 2.03 7.412/3.655 (torch-compile-tf32) | 1.21 5.821/4.822 (torch-compile-tf32) | 2.12 3.242/1.529 (torch-compile-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.15 0.636/0.553 (torch-bf16-compile) | 1.05≈ 0.7123/0.6788 (torch-bf16-compile) | 1.75 0.7899/0.4515 (torch-bf16-compile) | 1.04≈ 0.5946/0.572 (torch-bf16-compile) | **0.82** 0.3523/0.4288 (torch-bf16-compile) |
| llm_dec_b1_l2048_bf16w | bf16w | 1.02≈ 0.1501/0.1471 (torch-bf16-compile) | **0.74** 0.1854/0.2502 (torch-bf16-compile) | 1.09 0.1698/0.1551 (torch-bf16-compile) | **0.89** 0.1542/0.1724 (torch-bf16-compile) | **0.33** 0.09614/0.2931 (torch-bf16-compile) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.28 0.2405/0.1883 (liger-bf16) | 0.95≈ 0.2869/0.3018 (torch-bf16-compile) | 1.16 0.2305/0.1981 (torch-bf16-compile) | 1.24 0.2575/0.2076 (torch-bf16-compile) | **0.84** 0.1412/0.1675 (torch-bf16-compile) |
| llm_dec_b32_l2048_bf16w | bf16w | 2.10 0.9551/0.4548 (torch-bf16-compile) | 1.65 1.196/0.7245 (torch-bf16-compile) | 1.83 0.8665/0.4747 (torch-bf16-compile) | 2.25 1.023/0.4542 (torch-bf16-compile) | 1.62 0.4911/0.3037 (torch-bf16-compile) |
| llm_dec_b32_l8192_bf16w | bf16w | 2.43 3.426/1.407 (flashinfer-bf16) | 1.76 4.244/2.412 (torch-bf16-compile) | 2.11 2.541/1.206 (torch-bf16) | 2.70 3.661/1.354 (torch-bf16-compile) | 1.88 1.054/0.5618 (torch-bf16-compile) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.56 0.339/0.217 (torch-bf16-compile) | 2.35 0.8375/0.3557 (torch-bf16-compile) | 1.52 0.3471/0.2285 (torch-bf16-compile) | 1.77 0.4174/0.2364 (torch-bf16-compile) | **0.74** 0.1897/0.2566 (torch-bf16-compile) |
| llm_dec_b8_l8192_bf16w | bf16w | 2.07 0.9781/0.4726 (torch-bf16-cudnn) | 2.18 1.615/0.7404 (torch-bf16-compile) | 1.80 0.7625/0.4232 (torch-bf16) | 2.29 1.051/0.4597 (torch-bf16-compile) | 1.56 0.4542/0.2903 (torch-bf16-compile) |
| geomean f32 | | 0.779 (10) | 1.011 (10) | 0.922 (10) | 1.007 (10) | 0.807 (10) |
| geomean tf32 | | 1.408 (2) | 1.222 (2) | 1.727 (2) | 1.229 (2) | 2.015 (2) |
| geomean bf16w | | 1.582 (7) | 1.409 (7) | 1.569 (7) | 1.613 (7) | 0.968 (7) |
| **geomean** | | 1.077 (19) | 1.166 (19) | 1.198 (19) | 1.223 (19) | 0.950 (19) |
| win / even / loss | | 5/4/10 | 8/3/8 | 6/2/11 | 6/4/9 | 10/1/8 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.44** 0.06647/0.1516 (torch-compile-cg) | **0.68** 0.1124/0.1649 (torch-compile) | **0.53** 0.07259/0.1372 (torch-eager) | **0.51** 0.04794/0.0932 (torch-eager) | **0.50** 0.04777/0.0946 (torch-compile) |
| convmc_8_64_56_3 | f32 | **0.68** 0.0809/0.1198 (torch-eager) | 1.87 0.1516/0.0809 (torch-eager) | 2.25 0.1958/0.087 (torch-eager) | 1.67 0.09405/0.0563 (torch-eager) | 1.76 0.1272/0.0724 (torch-eager) |
| ew_4096 | f32 | **0.33** 0.08065/0.2437 (chain-fused) ±5% | **0.65** 0.1559/0.2406 (chain-fused) | **0.76** 0.1178/0.1556 (torch-compile) | **0.34** 0.07694/0.2243 (chain-fused) | 1.38 0.1144/0.0828 (chain-fused) ±16% |
| ewconvew_2048_5 | f32 | **0.41** 0.07747/0.1884 (torch-compile) | **0.52** 0.1385/0.2673 (torch-compile) | **0.49** 0.08772/0.1772 (torch-compile) | **0.49** 0.0616/0.1249 (torch-compile) | **0.52** 0.06018/0.1167 (torch-compile) |
| fir_4194304_16 | f32 | **0.43** 0.04431/0.1024 (torch-compile-cg) | **0.67** 0.07665/0.1137 (torch-compile) | **0.59** 0.05038/0.085 (torch-compile) | **0.55** 0.03517/0.0645 (torch-compile) | **0.65** 0.03974/0.0611 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.48** 0.05244/0.1085 (torch-compile-ro) | **0.74** 0.1039/0.1403 (torch-compile) | **0.64** 0.06194/0.0963 (torch-compile) | **0.61** 0.04107/0.0676 (torch-compile) ±5% | **0.69** 0.04602/0.0664 (torch-compile) |
| convmc_8_64_56_3_tf32 | tf32 | 1.31 0.08029/0.0614 (torch-eager-tf32) | 1.87 0.1513/0.0809 (torch-eager-tf32) | 4.07 0.1915/0.0471 (torch-eager-tf32) | 2.40 0.1006/0.042 (torch-eager-tf32) | 3.16 0.1232/0.039 (torch-eager-tf32) |
| geomean f32 | | 0.451 (6) | 0.773 (6) | 0.744 (6) | 0.603 (6) | 0.811 (6) |
| geomean tf32 | | 1.308 (1) | 1.870 (1) | 4.067 (1) | 2.396 (1) | 3.160 (1) |
| **geomean** | | 0.525 (7) | 0.877 (7) | 0.949 (7) | 0.735 (7) | 0.985 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 5/0/2 | 5/0/2 | 4/0/3 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 1.05≈ 1.441/1.378 (chain-lt) | 1.04≈ 2.45/2.357 (chain-streams) | 1.10 3.246/2.949 (chain-lt) | 0.97≈ 0.8958/0.9247 (chain-cutlass) | 1.14 1.181/1.034 (chain-lt) |
| fanout_2048 | f32 | 1.08 0.9824/0.9112 (chain-streams) | 1.05 1.656/1.573 (chain-streams) | 1.15 2.253/1.967 (chain-streams) | 0.95≈ 0.6075/0.638 (chain-streams) | 1.15 0.7925/0.6865 (chain-streams) |
| indep_2048 | f32 | 1.09 0.9995/0.9175 (torch-compile) | 1.05≈ 1.65/1.575 (torch-compile) | 1.14 2.246/1.968 (torch-compile) | 0.95≈ 0.5937/0.6226 (chain-cutlass) | 1.16 0.8044/0.693 (chain-lt) |
| indep_512 | f32 | 1.19 0.03893/0.0328 (torch-compile) | 1.65 0.08637/0.0522 (chain-streams) | 1.32 0.07592/0.0573 (chain-streams) | 1.09≈ 0.04115/0.0379 (chain-streams) ±35% | 1.85 0.06053/0.0328 (chain-streams) ±7% |
| loop_4096 | f32 | 0.99≈ 28.25/28.55 (chain-lt) | 0.99≈ 46.87/47.55 (chain-lt) | 1.07 62.97/59.03 (torch-eager) | 0.96≈ 18.75/19.46 (chain-cutlass) | 1.12 23.81/21.26 (chain-lt) |
| loopmm_1024_8 | f32 | 1.10 0.649/0.5878 (torch-compile-cg) | 1.11 1.109/0.9984 (torch-compile) | 1.37 1.435/1.044 (torch-eager) | 1.18 0.4281/0.3625 (torch-compile) | 1.15 0.5319/0.4614 (torch-eager) |
| rt_mlp2_1024 | f32 | 1.01≈ 2.557/2.527 (torch-compile) | 1.03≈ 4.303/4.195 (torch-compile) | 1.43 6.834/4.774 (torch-compile) | 1.13 1.687/1.492 (torch-compile) | 1.05 2.146/2.043 (torch-compile) |
| rt_mlp2_2048 | f32 | 1.04≈ 16.19/15.53 (torch-compile-cg) | 0.99≈ 26.37/26.7 (torch-compile) | 1.16 37.17/31.95 (torch-compile) | **0.89** 9.677/10.92 (torch-compile) | 1.05 12.05/11.44 (torch-compile) |
| twin_2048 | f32 | 1.02≈ 0.9325/0.9175 (torch-compile) | 1.12 1.636/1.462 (torch-compile) | 1.03≈ 2.021/1.964 (torch-eager) | 0.96≈ 0.5949/0.6175 (torch-compile) | 1.16 0.792/0.6845 (torch-compile) |
| **geomean** | | 1.061 (9) | 1.100 (9) | 1.190 (9) | 1.005 (9) | 1.187 (9) |
| win / even / loss | | 0/5/4 | 0/5/4 | 0/1/8 | 1/6/2 | 0/0/9 |

## bf16 real-world reference

The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3174 (triton-bf16) · f32 2.55 | 0.4803 (bf16) · f32 2.57 | 0.2908 (bf16-compile) · f32 5.94 | 0.3154 (bf16-compile) · f32 2.01 | 0.095 (bf16-compile) · f32 7.50 |
| pj_m128_n4096_k14336 | 0.172 (bf16) · f32 2.64 | 0.2785 (bf16) · f32 2.61 | 0.126 (bf16-compile) · f32 7.28 | 0.1894 (bf16-compile) · f32 2.05 | 0.0596 (bf16-compile) · f32 6.34 |
| pj_m128_n4096_k4096 | 0.0532 (bf16) · f32 2.74 | 0.085 (bf16) · f32 2.97 | 0.0573 (bf16-compile) · f32 5.34 | 0.0604 (bf16-compile) · f32 2.25 | 0.025 (bf16) · f32 5.90 |
| pj_m128_n6144_k4096 | 0.0881 (bf16) · f32 2.38 | 0.128 (bf16) · f32 2.72 | 0.0676 (bf16-compile) · f32 6.39 | 0.0819 (bf16-compile) · f32 2.33 | 0.0339 (bf16) · f32 5.95 |
| pj_m16_n18432_k7168 | 0.3092 (bf16) · f32 1.95 | 0.3082 (bf16-compile) · f32 2.12 | 0.213 (bf16-compile) · f32 2.04 | 0.3399 (bf16-compile) · f32 1.87 | 0.0988 (bf16) · f32 2.35 |
| pj_m16_n2048_k7168 | 0.0389 (bf16-compile) · f32 1.19 | 0.044 (bf16-compile) · f32 2.73 | 0.044 (bf16-compile) · f32 2.35 | 0.0492 (bf16-compile) · f32 1.08 | 0.0231 (bf16) · f32 3.31 |
| pj_m16_n28672_k4096 | 0.2744 (bf16-compile) · f32 1.97 | 0.2734 (bf16-compile) · f32 2.17 | 0.1874 (bf16-compile) · f32 2.24 | 0.3082 (bf16-compile) · f32 1.86 | 0.0911 (bf16-compile) · f32 2.43 |
| pj_m16_n4096_k14336 | 0.1393 (bf16-compile) · f32 1.91 | 0.1423 (bf16-compile) · f32 2.19 | 0.1106 (bf16-compile) · f32 2.05 | 0.177 (bf16-compile) · f32 1.80 | 0.0562 (bf16) · f32 2.48 |
| pj_m16_n4096_k4096 | 0.043 (bf16-compile-cg) · f32 1.21 | 0.0512 (bf16) · f32 2.68 | 0.044 (bf16-compile) · f32 2.28 | 0.0543 (bf16-compile) · f32 0.99 | 0.0243 (bf16) · f32 2.92 |
| pj_m16_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.59 | 0.0666 (bf16-compile) · f32 2.50 | 0.0604 (bf16-compile) · f32 2.14 | 0.0788 (bf16-compile) · f32 1.55 | 0.0302 (bf16) · f32 2.93 |
| pj_m1_n18432_k7168 | 0.3072 (bf16-compile-ro) · f32 1.97 | 0.3287 (bf16-compile) · f32 1.92 | 0.212 (bf16-compile) · f32 2.03 | 0.3031 (bf16-compile) · f32 2.07 | 0.0986 (bf16) · f32 2.29 |
| pj_m1_n2048_k7168 | 0.043 (bf16-compile) · f32 0.78 | 0.0471 (bf16-compile) · f32 2.20 | 0.043 (bf16) · f32 2.04 | 0.0503 (bf16-compile) · f32 0.66 | 0.0222 (bf16) · f32 2.37 |
| pj_m1_n28672_k4096 | 0.2744 (bf16-compile) · f32 2.05 | 0.2806 (bf16-compile) · f32 2.04 | 0.1833 (bf16-compile) · f32 2.09 | 0.2733 (bf16-compile) · f32 2.08 | 0.0922 (bf16-compile) · f32 2.22 |
| pj_m1_n4096_k14336 | 0.1413 (bf16) · f32 1.80 | 0.1485 (bf16-compile) · f32 2.04 | 0.1014 (bf16-compile) · f32 1.99 | 0.1516 (bf16-compile) · f32 1.98 | 0.0555 (bf16) · f32 2.31 |
| pj_m1_n4096_k4096 | 0.0492 (bf16-compile) · f32 0.66 | 0.0492 (bf16-compile) · f32 2.30 | 0.0399 (bf16-compile) · f32 2.34 | 0.0564 (bf16-compile) · f32 0.57 | 0.0228 (bf16) · f32 3.03 |
| pj_m1_n6144_k4096 | 0.0635 (bf16-compile-cg) · f32 1.45 | 0.0646 (bf16-compile) · f32 2.34 | 0.0553 (bf16-compile) · f32 2.10 | 0.0799 (bf16) · f32 1.06 | 0.0323 (bf16) · f32 2.15 |
| pj_m2048_n28672_k4096 | 3.976 (bf16-compile-cg) · f32 3.18 | 6.386 (bf16-compile) · f32 3.29 | 1.979 (bf16) · f32 13.57 | 2.822 (bf16) · f32 3.07 | 0.6197 (bf16-compile) · f32 16.39 |
| pj_m2048_n4096_k14336 | 2.091 (bf16-compile-cg) · f32 3.07 | 3.777 (bf16) · f32 2.94 | 0.8991 (bf16) · f32 14.83 | 1.499 (bf16-compile) · f32 2.68 | 0.3059 (bf16) · f32 17.13 |
| pj_m2048_n4096_k4096 | 0.6001 (bf16-compile-cg) · f32 3.06 | 1.093 (bf16) · f32 2.75 | 0.3092 (bf16) · f32 12.51 | 0.4332 (bf16-compile) · f32 2.69 | 0.0941 (bf16) · f32 16.10 |
| pj_m2048_n6144_k4096 | 0.8827 (bf16-compile-ro) · f32 3.00 | 1.395 (bf16-compile) · f32 3.29 | 0.4147 (bf16-compile) · f32 14.00 | 0.6011 (bf16-compile) · f32 2.91 | 0.1392 (bf16) · f32 16.03 |
| llm_blkdec_b1_l4096 | 0.5538 (bf16-compile) · f32 1.93 · bf16w 1.15 | 0.6837 (bf16-compile) · f32 1.66 · bf16w 1.04 | 0.4522 (bf16-compile) · f32 2.12 · bf16w 1.75 | 0.5718 (bf16-compile) · f32 2.61 · bf16w 1.04 | 0.4299 (bf16-compile) · f32 1.32 · bf16w 0.82 |
| llm_blkpre_t2048 | 8.029 (bf16-compile-cg) · f32 3.40 | 13.61 (bf16-compile) · f32 3.38 | 4.481 (bf16-compile) · f32 12.47 | 5.614 (bf16-compile) · f32 3.31 | - |
| llm_dec_b1_l2048 | 0.1469 (bf16-compile) · f32 1.63 · bf16w 1.02 | 0.2483 (bf16-compile) · f32 1.12 · bf16w 0.75 | 0.1739 (bf16-compile) · f32 1.24 · bf16w 0.98 | 0.1716 (bf16-compile) · f32 1.41 · bf16w 0.90 | 0.2791 (bf16-compile) · f32 0.42 · bf16w 0.34 |
| llm_dec_b1_l8192 | 0.1953 (bf16-compile) · f32 1.70 · bf16w 1.23 | 0.316 (bf16-compile) · f32 1.20 · bf16w 0.91 | 0.1977 (bf16-compile) · f32 1.39 · bf16w 1.17 | 0.2076 (bf16-compile) · f32 1.66 · bf16w 1.24 | 0.2733 (bf16-compile) · f32 0.61 · bf16w 0.52 |
| llm_dec_b32_l2048 | 0.4563 (bf16-compile) · f32 2.27 · bf16w 2.09 | 0.7248 (bf16-compile) · f32 1.69 · bf16w 1.65 | 0.4934 (bf16) · f32 1.74 · bf16w 1.76 | 0.4565 (bf16-compile) · f32 2.47 · bf16w 2.24 | 0.2181 (bf16-compile) · f32 2.16 · bf16w 2.25 |
| llm_dec_b32_l8192 | 1.416 (flashinfer-bf16) · f32 2.48 · bf16w 2.42 | 2.407 (bf16-compile) · f32 1.78 · bf16w 1.76 | 1.21 (bf16) · f32 2.10 · bf16w 2.10 | 1.356 (bf16-compile) · f32 2.78 · bf16w 2.70 | 0.5537 (bf16-compile) · f32 1.87 · bf16w 1.90 |
| llm_dec_b8_l2048 | 0.2166 (bf16-compile) · f32 2.02 · bf16w 1.57 | 0.3464 (bf16-compile) · f32 2.95 · bf16w 2.42 | 0.2289 (bf16-compile) · f32 1.74 · bf16w 1.52 | 0.2382 (bf16-compile) · f32 2.08 · bf16w 1.75 | 0.2476 (bf16-compile) · f32 0.84 · bf16w 0.77 |
| llm_dec_b8_l8192 | 0.4675 (liger-bf16) · f32 2.28 · bf16w 2.09 | 0.7373 (bf16-compile) · f32 2.44 · bf16w 2.19 | 0.4227 (bf16) · f32 1.91 · bf16w 1.80 | 0.4631 (bf16-compile) · f32 2.44 · bf16w 2.27 | 0.2852 (bf16-compile) · f32 1.64 · bf16w 1.59 |
| llm_pre_t2048 | 1.915 (bf16-compile-cg) · f32 4.47 | 3.22 (bf16-compile) · f32 4.49 | 1.046 (bf16-compile) · f32 14.46 | 1.329 (bf16-compile) · f32 4.75 | 0.4071 (bf16-compile) · f32 15.12 |
| llm_pre_t8192 | 11.38 (bf16-compile) · f32 7.09 | 18.12 (bf16-compile) · f32 95.53 | 6.524 (bf16-compile) · f32 219.28 | 7.848 (bf16-compile) · f32 95.98 | 2.007 (bf16-compile) · f32 412.44 |

## Verdict flips vs the mixed-precision gap tables

gap.py's rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables (gap-<part>.md, gap.md) row for row. Rows a part's gap table did not cover (the RTX 5080's model-scale rows, reported in model-scale.md) are judged against that rule applied to the same data.

| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |
|---|---|---|---|---|
| rtx-5080 | llm_blkdec_b1_l4096 | 0.5538 (torch-bf16-compile) 1.93 loss | 1.066 (torch-compile) 1.01 even | old best a bf16 leg |
| rtx-5080 | llm_blkpre_t2048 | 8.074 (torch-bf16-compile) 3.38 loss | 26.67 (liger) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_dec_b1_l2048 | 0.1469 (torch-bf16-compile) 1.63 loss | 0.2532 (torch-compile-ro) 0.95 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l8192 | 0.1953 (torch-bf16-compile) 1.70 loss | 0.342 (torch-compile) 0.97 even | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l2048 | 0.4563 (torch-bf16-compile) 2.27 loss | 2.256 (torch-compile) 0.46 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l8192 | 1.5 (torch-bf16-compile) 2.34 loss | 9.484 (torch-compile) 0.37 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l2048 | 0.2166 (torch-bf16-compile) 2.02 loss | 0.6685 (torch-compile) 0.66 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l8192 | 0.489 (torch-bf16-compile) 2.18 loss | 2.468 (torch-compile-cg) 0.43 win | old best a bf16 leg |
| rtx-5080 | pj_m16_n18432_k7168 | 0.3092 (torch-bf16) 1.95 loss | 0.5878 (cublas-fresh) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n2048_k7168 | 0.0389 (torch-bf16-compile) 1.19 loss | 0.0492 (cublas-fresh) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n28672_k4096 | 0.2744 (torch-bf16-compile) 1.97 loss | 0.5243 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n4096_k14336 | 0.1393 (torch-bf16-compile) 1.91 loss | 0.2744 (torch-compile) 0.97 even | old best a bf16 leg |
| rtx-5080 | pj_m16_n4096_k4096 | 0.0451 (torch-bf16-compile) 1.16 loss | 0.0553 (cublas-fresh) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n18432_k7168 | 0.3092 (torch-bf16) 1.96 loss | 0.6083 (torch-compile) 0.99 even | old best a bf16 leg |
| rtx-5080 | pj_m1_n2048_k7168 | 0.043 (torch-bf16-compile) 0.78 win | 0.0266 (cublas-fresh) 1.26 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n4096_k4096 | 0.0492 (torch-bf16-compile) 0.66 win | 0.0205 (lt-fresh) 1.59 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096 | 3.984 (torch-bf16-compile) 3.17 loss | 12.21 (cublas-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k14336 | 2.093 (torch-bf16-compile) 3.07 loss | 6.498 (torch-eager) 0.99 even | old best a bf16 leg |
| rtx-5080 | pj_m2048_n4096_k4096 | 0.6021 (torch-bf16-compile) 3.05 loss | 1.772 (torch-compile-cg) 1.04 even | old best a bf16 leg |
| rtx-5080 | pj_m2048_n6144_k4096 | 0.8868 (torch-bf16-compile) 2.99 loss | 2.602 (torch-compile-cg) 1.02 even | old best a bf16 leg |
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

## What the new legs change

Rows whose best leg is one the competitors step added (CUDA-graph torch.compile, SDPA backends, hand Triton, Liger, FlashAttention-2/3, FlashInfer), against the best over the legs before it (`base`).

| part | workload | prec | base: best (leg) ratio | now: best (leg) ratio | verdict |
|---|---|---|---|---|---|
| rtx-5080 | epi_4096 | f32 | 3.558 (torch-compile) 1.03 | 3.547 (torch-compile-ro) 1.04 | even → even |
| rtx-5080 | pj_m2048_n4096_k4096 | f32 | 1.776 (torch-eager) 1.03 | 1.772 (torch-compile-cg) 1.04 | even → even |
| rtx-5080 | pj_m2048_n6144_k4096 | f32 | 2.605 (torch-eager) 1.02 | 2.602 (torch-compile-cg) 1.02 | even → even |
| rtx-5080 | mlp2_2048_tf32 | tf32 | 1.546 (torch-eager-tf32) 1.08 | 1.514 (torch-compile-ro-tf32) 1.10 | loss → loss |
| rtx-5080 | mlp_2048_tf32 | tf32 | 0.769 (torch-eager-tf32) 1.08 | 0.7598 (torch-compile-ro-tf32) 1.09 | loss → loss |
| rtx-5080 | attn_1024_64 | f32 | 0.0297 (torch-compile) 1.32 | 0.0287 (torch-compile-cg) 1.36 | loss → loss |
| rtx-5080 | attn_2048_128 | f32 | 0.0953 (torch-sdpa-compile) 1.24 | 0.0942 (torch-sdpa-compile-cg) 1.25 | loss → loss |
| rtx-5080 | attn_2048_64 | f32 | 0.0748 (torch-sdpa-compile) 1.06 | 0.0696 (torch-compile-ro) 1.14 | loss → loss |
| rtx-5080 | causal_1024_256 | f32 | 44.52 (torch-compile) 0.34 | 38.23 (torch-compile-cg) 0.40 | win → win |
| rtx-5080 | gen_1024_512_256 | f32 | 22.53 (torch-compile) 0.34 | 20.84 (torch-compile-cg) 0.37 | win → win |
| rtx-5080 | lsm_1_131072 | f32 | 0.0246 (torch-compile) 0.64 | 0.0225 (torch-compile-cg) 0.70 | win → win |
| rtx-5080 | prefill_1024_512_256 | f32 | 22.49 (torch-compile) 0.34 | 19.26 (torch-compile-cg) 0.40 | win → win |
| rtx-5080 | softmax_rows_4096 | f32 | 3.54 (torch-compile) 1.07 | 3.529 (torch-compile-ro) 1.08 | loss → loss |
| rtx-5080 | llm_blkpre_t2048 | f32 | 28.09 (torch-compile) 0.97 | 26.67 (liger) 1.02 | even → even |
| rtx-5080 | llm_blkpre_t2048_tf32 | tf32 | 18.56 (torch-compile-tf32) 1.18 | 18.08 (triton-tf32) 1.21 | loss → loss |
| rtx-5080 | llm_dec_b1_l2048 | f32 | 0.2536 (torch-compile) 0.95 | 0.2532 (torch-compile-ro) 0.95 | win → win |
| rtx-5080 | llm_dec_b1_l8192_bf16w | bf16w | 0.1913 (torch-bf16-compile) 1.26 | 0.1883 (liger-bf16) 1.28 | loss → loss |
| rtx-5080 | llm_dec_b32_l8192_bf16w | bf16w | 1.49 (torch-bf16-compile) 2.30 | 1.407 (flashinfer-bf16) 2.43 | loss → loss |
| rtx-5080 | llm_dec_b8_l8192 | f32 | 2.488 (torch-compile) 0.43 | 2.468 (torch-compile-cg) 0.43 | win → win |
| rtx-5080 | llm_dec_b8_l8192_bf16w | bf16w | 0.485 (torch-bf16-compile) 2.02 | 0.4726 (torch-bf16-cudnn) 2.07 | loss → loss |
| rtx-5080 | llm_pre_t2048 | f32 | 8.367 (torch-compile) 1.02 | 7.2 (liger) 1.19 | even → loss |
| rtx-5080 | llm_pre_t2048_tf32 | tf32 | 6.062 (torch-compile-tf32) 1.23 | 4.537 (triton-tf32) 1.64 | loss → loss |
| rtx-5080 | llm_pre_t8192 | f32 | 84.52 (torch-compile) 0.95 | 52.83 (liger) 1.53 | even → loss |
| rtx-5080 | conv_2048_5 | f32 | 0.1536 (torch-eager) 0.43 | 0.1516 (torch-compile-cg) 0.44 | win → win |
| rtx-5080 | fir_4194304_16 | f32 | 0.1044 (torch-eager) 0.42 | 0.1024 (torch-compile-cg) 0.43 | win → win |
| rtx-5080 | firew_rowsum_2048x2048_16 | f32 | 0.1106 (torch-compile) 0.47 | 0.1085 (torch-compile-ro) 0.48 | win → win |
| rtx-5080 | loopmm_1024_8 | f32 | 0.5898 (torch-compile) 1.10 | 0.5878 (torch-compile-cg) 1.10 | loss → loss |
| rtx-5080 | rt_mlp2_2048 | f32 | 15.55 (torch-compile) 1.04 | 15.53 (torch-compile-cg) 1.04 | even → even |

Mapal wins that disappear: 0.

## Every competitor leg

Per row and part, every leg at the row's precision, grouped by competitor family: `leg ms (Mapal/leg)`; **bold** = the row's fastest leg; `✗gate` = timed but outside the value gate (not counted); `cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg for this row on another part; `unavailable` = the part's run reported the package missing. Families: vendor libs; torch eager; torch.compile; compile + CUDA graphs; reduce-overhead; SDPA; FlashAttention-2/3; FlashInfer; hand Triton; Liger; TensorRT-LLM; hand CUDA. Columns with no leg on any row of a table are left out.

### GEMM cells & projections

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| epi_4096 | rtx-5080 | 3.675 | chain-lt 3.567 (1.03)<br>chain-eager 3.7 (0.99)<br>chain-cutlass 3.761 (0.98) | torch-eager 3.747 (0.98) | torch-compile 3.558 (1.03) | torch-compile-cg 3.55 (1.04) | **torch-compile-ro 3.547 (1.04)** | triton 4.344 (0.85) |
| epi_4096 | rtx-3090 | 5.596 | **chain-lt 5.828 (0.96)**<br>chain-cutlass 5.956 (0.94)<br>chain-eager 5.976 (0.94) | torch-eager 6.083 (0.92) | torch-compile 5.979 (0.94) | not run | not run | not run |
| epi_4096 | a100 | 7.88 | **chain-lt 7.296 (1.08)**<br>chain-eager 7.389 (1.07)<br>chain-cutlass 7.639 (1.03) | torch-eager 7.48 (1.05) | torch-compile 7.608 (1.04) | not run | not run | not run |
| epi_4096 | rtx-4090 | 2.407 | **chain-lt 2.434 (0.99)**<br>chain-cutlass 2.44 (0.99)<br>chain-eager 2.543 (0.95) | torch-eager 2.618 (0.92) | torch-compile 2.492 (0.97) | not run | not run | not run |
| epi_4096 | h100 | 2.999 | **chain-lt 2.663 (1.13)**<br>chain-eager 2.722 (1.10)<br>chain-cutlass 3.351 (0.90) | torch-eager 2.76 (1.09) | torch-compile 2.711 (1.11) | not run | not run | not run |
| mm_16x2048x7168 | rtx-5080 | 0.0879 | **cublas 0.0799 (1.10)**<br>cutlass-tuned 0.385 (0.23)<br>cutlass 0.526 (0.17) |  |  |  |  | triton 0.1536 (0.57) |
| mm_16x2048x7168 | rtx-3090 | 0.09005 | **cublas 0.08805 (1.02)**<br>cutlass-tuned 0.2155 (0.42)<br>cutlass 0.6974 (0.13) |  |  |  |  | not run |
| mm_16x2048x7168 | a100 | 0.09215 | **cublas 0.0906 (1.02)**<br>cutlass-tuned 0.3502 (0.26)<br>cutlass 1.351 (0.07) |  |  |  |  | not run |
| mm_16x2048x7168 | rtx-4090 | 0.1084 | **cublas 0.0963 (1.13)**<br>cutlass-tuned 0.2032 (0.53)<br>cutlass 0.4966 (0.22) |  |  |  |  | not run |
| mm_16x2048x7168 | h100 | 0.09575 | **cublas 0.05195 (1.84)**<br>cutlass-tuned 0.2816 (0.34)<br>cutlass 0.7594 (0.13) |  |  |  |  | not run |
| mm_4096x4096x4096 | rtx-5080 | 3.812 | **cublas 3.652 (1.04)**<br>cutlass 3.725 (1.02)<br>cutlass-tuned 9.051 (0.42) |  |  |  |  | triton 4.23 (0.90) |
| mm_4096x4096x4096 | rtx-3090 | 7.115 | **cublas 6.409 (1.11)**<br>cutlass 6.504 (1.09)<br>cutlass-tuned 17.16 (0.41) |  |  |  |  | not run |
| mm_4096x4096x4096 | a100 | 7.681 | **cublas 7.402 (1.04)**<br>cutlass 7.711 (1.00)<br>cutlass-tuned 12.01 (0.64) |  |  |  |  | not run |
| mm_4096x4096x4096 | rtx-4090 | 2.393 | **cublas 2.477 (0.97)**<br>cutlass 2.491 (0.96)<br>cutlass-tuned 6.71 (0.36) |  |  |  |  | not run |
| mm_4096x4096x4096 | h100 | 3.136 | **cublas 2.658 (1.18)**<br>cutlass 3.323 (0.94)<br>cutlass-tuned 6.757 (0.46) |  |  |  |  | not run |
| pj_m128_n28672_k4096 | rtx-5080 | 0.8088 | **lt 0.7516 (1.08)**<br>lt-fresh 0.7516 (1.08)<br>cublas 0.7947 (1.02)<br>cublas-fresh 0.7967 (1.02) | torch-eager 0.8008 (1.01) | torch-compile 0.8008 (1.01) | torch-compile-cg 0.8008 (1.01) | torch-compile-ro 0.8008 (1.01) | triton 1.067 (0.76) |
| pj_m128_n28672_k4096 | rtx-3090 | 1.235 | cublas 1.351 (0.91)<br>lt 1.352 (0.91)<br>cublas-fresh 1.352 (0.91)<br>lt-fresh 1.352 (0.91) | torch-eager 1.362 (0.91) | **torch-compile 1.265 (0.98)** | not run | not run | not run |
| pj_m128_n28672_k4096 | a100 | 1.727 | **cublas 1.781 (0.97)**<br>lt 1.781 (0.97)<br>cublas-fresh 1.781 (0.97)<br>lt-fresh 1.782 (0.97) | torch-eager 2.241 (0.77) | torch-compile 2.241 (0.77) | not run | not run | not run |
| pj_m128_n28672_k4096 | rtx-4090 | 0.6337 | **lt-fresh 0.6492 (0.98)**<br>lt 0.6555 (0.97)<br>cublas-fresh 0.6831 (0.93)<br>cublas 0.6892 (0.92) | torch-eager 0.6932 (0.91) | torch-compile 0.6932 (0.91) | not run | not run | not run |
| pj_m128_n28672_k4096 | h100 | 0.7124 | **lt-fresh 0.6633 (1.07)**<br>lt 0.6639 (1.07)<br>cublas 0.6812 (1.05)<br>cublas-fresh 0.6817 (1.05) | torch-eager 0.723 (0.99) | torch-compile 0.6818 (1.04) | not run | not run | not run |
| pj_m128_n4096_k14336 | rtx-5080 | 0.4547 | **cublas-fresh 0.4014 (1.13)**<br>lt-fresh 0.4014 (1.13)<br>cublas 0.4055 (1.12)<br>lt 0.4055 (1.12) | torch-eager 0.4117 (1.10) | torch-compile 0.4117 (1.10) | torch-compile-cg 0.4117 (1.10) | torch-compile-ro 0.4117 (1.10) | triton 0.5898 (0.77) |
| pj_m128_n4096_k14336 | rtx-3090 | 0.7261 | lt-fresh 0.681 (1.07)<br>lt 0.682 (1.06)<br>cublas 0.684 (1.06)<br>cublas-fresh 0.684 (1.06) | **torch-eager 0.6748 (1.08)** | torch-compile 0.6779 (1.07) | not run | not run | not run |
| pj_m128_n4096_k14336 | a100 | 0.9178 | **cublas-fresh 0.8509 (1.08)**<br>lt-fresh 0.8509 (1.08)<br>cublas 0.854 (1.07)<br>lt 0.854 (1.07) | torch-eager 1.104 (0.83) | torch-compile 1.102 (0.83) | not run | not run | not run |
| pj_m128_n4096_k14336 | rtx-4090 | 0.3881 | **lt-fresh 0.3574 (1.09)**<br>lt 0.3655 (1.06)<br>cublas-fresh 0.3717 (1.04)<br>cublas 0.3779 (1.03) | torch-eager 0.3912 (0.99) | torch-compile 0.3809 (1.02) | not run | not run | not run |
| pj_m128_n4096_k14336 | h100 | 0.378 | **lt 0.3131 (1.21)**<br>cublas 0.3139 (1.20)<br>cublas-fresh 0.3144 (1.20)<br>lt-fresh 0.3144 (1.20) | torch-eager 0.314 (1.20) | torch-compile 0.3146 (1.20) | not run | not run | not run |
| pj_m128_n4096_k4096 | rtx-5080 | 0.1456 | **cublas-fresh 0.1352 (1.08)**<br>lt-fresh 0.1352 (1.08)<br>cublas 0.1434 (1.02)<br>lt 0.1434 (1.02) | torch-eager 0.1454 (1.00) | torch-compile 0.1454 (1.00) | torch-compile-cg 0.1454 (1.00) | torch-compile-ro 0.1454 (1.00) | triton 0.1761 (0.83) |
| pj_m128_n4096_k4096 | rtx-3090 | 0.2524 | **lt 0.2048 (1.23)**<br>lt-fresh 0.2058 (1.23)<br>cublas 0.2263 (1.12)<br>cublas-fresh 0.2263 (1.12) | torch-eager 0.2263 (1.12) | torch-compile 0.2273 (1.11) | not run | not run | not run |
| pj_m128_n4096_k4096 | a100 | 0.3061 | **cublas 0.2601 (1.18)**<br>lt 0.2611 (1.17)<br>cublas-fresh 0.2611 (1.17)<br>lt-fresh 0.2611 (1.17) | torch-eager 0.3348 (0.91) | torch-compile 0.3328 (0.92) | not run | not run | not run |
| pj_m128_n4096_k4096 | rtx-4090 | 0.1357 | **lt-fresh 0.1024 (1.33)**<br>cublas-fresh 0.1065 (1.27)<br>lt 0.124 (1.09)<br>cublas 0.1393 (0.97) | torch-eager 0.1435 (0.95) | torch-compile 0.1178 (1.15) | not run | not run | not run |
| pj_m128_n4096_k4096 | h100 | 0.1476 | **lt 0.098 (1.51)**<br>cublas 0.0988 (1.49)<br>cublas-fresh 0.0988 (1.49)<br>lt-fresh 0.0992 (1.49) | torch-eager 0.0988 (1.49) | torch-compile 0.0995 (1.48) | not run | not run | not run |
| pj_m128_n6144_k4096 | rtx-5080 | 0.2095 | **cublas-fresh 0.1802 (1.16)**<br>lt-fresh 0.1802 (1.16)<br>cublas 0.1864 (1.12)<br>lt 0.1864 (1.12) | torch-eager 0.1884 (1.11) | torch-compile 0.1884 (1.11) | torch-compile-cg 0.1884 (1.11) | torch-compile-ro 0.1884 (1.11) | triton 0.2949 (0.71) |
| pj_m128_n6144_k4096 | rtx-3090 | 0.3484 | **lt 0.2908 (1.20)**<br>lt-fresh 0.2918 (1.19)<br>cublas 0.3205 (1.09)<br>cublas-fresh 0.3205 (1.09) | torch-eager 0.3195 (1.09) | torch-compile 0.3195 (1.09) | not run | not run | not run |
| pj_m128_n6144_k4096 | a100 | 0.4317 | **lt-fresh 0.3727 (1.16)**<br>lt 0.3738 (1.15)<br>cublas 0.4557 (0.95)<br>cublas-fresh 0.4557 (0.95) | torch-eager 0.5816 (0.74) | torch-compile 0.5427 (0.80) | not run | not run | not run |
| pj_m128_n6144_k4096 | rtx-4090 | 0.1911 | **lt-fresh 0.1567 (1.22)**<br>cublas-fresh 0.1588 (1.20)<br>lt 0.1782 (1.07)<br>cublas 0.1874 (1.02) | torch-eager 0.1935 (0.99) | torch-compile 0.1843 (1.04) | not run | not run | not run |
| pj_m128_n6144_k4096 | h100 | 0.2016 | **lt-fresh 0.1486 (1.36)**<br>cublas-fresh 0.1493 (1.35)<br>lt 0.1495 (1.35)<br>cublas 0.1496 (1.35) | torch-eager 0.1495 (1.35) | torch-compile 0.1521 (1.33) | not run | not run | not run |
| pj_m16_n18432_k7168 | rtx-5080 | 0.6025 | **cublas-fresh 0.5878 (1.02)**<br>lt-fresh 0.5878 (1.02)<br>cublas 0.6185 (0.97)<br>lt 0.6185 (0.97) | torch-eager 0.6185 (0.97) | torch-compile 0.6083 (0.99) | torch-compile-cg 0.6083 (0.99) | torch-compile-ro 0.6185 (0.97) | triton 0.6185 (0.97) |
| pj_m16_n18432_k7168 | rtx-3090 | 0.653 | cublas 0.6246 (1.05)<br>lt 0.6246 (1.05)<br>cublas-fresh 0.6246 (1.05)<br>lt-fresh 0.6246 (1.05) | torch-eager 0.6236 (1.05) | **torch-compile 0.6113 (1.07)** | not run | not run | not run |
| pj_m16_n18432_k7168 | a100 | 0.4347 | lt 0.5048 (0.86)<br>lt-fresh 0.5059 (0.86)<br>cublas 0.5284 (0.82)<br>cublas-fresh 0.5304 (0.82) | torch-eager 0.6584 (0.66) | **torch-compile 0.4454 (0.98)** | not run | not run | not run |
| pj_m16_n18432_k7168 | rtx-4090 | 0.6352 | **lt-fresh 0.6113 (1.04)**<br>lt 0.6216 (1.02)<br>cublas-fresh 0.6298 (1.01)<br>cublas 0.638 (1.00) | torch-eager 0.638 (1.00) | torch-compile 0.6175 (1.03) | not run | not run | not run |
| pj_m16_n18432_k7168 | h100 | 0.2317 | lt 0.2369 (0.98)<br>lt-fresh 0.2382 (0.97)<br>cublas 0.2602 (0.89)<br>cublas-fresh 0.2612 (0.89) | torch-eager 0.2628 (0.88) | **torch-compile 0.2072 (1.12)** | not run | not run | not run |
| pj_m16_n2048_k7168 | rtx-5080 | 0.04612 | **cublas-fresh 0.0492 (0.94)**<br>lt-fresh 0.0492 (0.94)<br>lt 0.0798 (0.58)<br>cublas 0.0799 (0.58) | torch-eager 0.0799 (0.58) | torch-compile 0.0778 (0.59) | torch-compile-cg 0.0778 (0.59) | torch-compile-ro 0.0799 (0.58) | triton 0.1536 (0.30) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1199 | lt 0.085 (1.41)<br>lt-fresh 0.085 (1.41)<br>cublas 0.0891 (1.35)<br>cublas-fresh 0.0891 (1.35) | torch-eager 0.0942 (1.27) | **torch-compile 0.084 (1.43)** | not run | not run | not run |
| pj_m16_n2048_k7168 | a100 | 0.1035 | **cublas-fresh 0.087 (1.19)**<br>lt-fresh 0.087 (1.19)<br>cublas 0.0911 (1.14)<br>lt 0.0911 (1.14) | torch-eager 0.1096 (0.94) | torch-compile 0.1096 (0.94) | not run | not run | not run |
| pj_m16_n2048_k7168 | rtx-4090 | 0.05337 | **cublas-fresh 0.0307 (1.74)**<br>lt-fresh 0.0307 (1.74)<br>cublas 0.0963 (0.55)<br>lt 0.0963 (0.55) | torch-eager 0.1004 (0.53) | torch-compile 0.0942 (0.57) | not run | not run | not run |
| pj_m16_n2048_k7168 | h100 | 0.07647 | **lt-fresh 0.0484 (1.58)**<br>lt 0.049 (1.56)<br>cublas-fresh 0.0503 (1.52)<br>cublas 0.0513 (1.49) | torch-eager 0.0542 (1.41) | torch-compile 0.0831 (0.92) | not run | not run | not run |
| pj_m16_n28672_k4096 | rtx-5080 | 0.5395 | **lt-fresh 0.5243 (1.03)**<br>lt 0.5509 (0.98)<br>cublas-fresh 1.859 (0.29)<br>cublas 1.926 (0.28) | torch-eager 1.927 (0.28) | torch-compile 0.5427 (0.99) | torch-compile-cg 0.5427 (0.99) | torch-compile-ro 1.927 (0.28) | triton 0.5509 (0.98) |
| pj_m16_n28672_k4096 | rtx-3090 | 0.5942 | cublas 0.555 (1.07)<br>lt 0.5559 (1.07)<br>cublas-fresh 0.556 (1.07)<br>lt-fresh 0.556 (1.07) | torch-eager 0.554 (1.07) | **torch-compile 0.5458 (1.09)** | not run | not run | not run |
| pj_m16_n28672_k4096 | a100 | 0.4203 | cublas 0.471 (0.89)<br>lt 0.471 (0.89)<br>cublas-fresh 0.4721 (0.89)<br>lt-fresh 0.4731 (0.89) | torch-eager 0.5847 (0.72) | **torch-compile 0.426 (0.99)** | not run | not run | not run |
| pj_m16_n28672_k4096 | rtx-4090 | 0.5731 | **cublas-fresh 0.5509 (1.04)**<br>lt-fresh 0.5509 (1.04)<br>cublas 0.5571 (1.03)<br>lt 0.5571 (1.03) | torch-eager 0.5601 (1.02) | torch-compile 0.555 (1.03) | not run | not run | not run |
| pj_m16_n28672_k4096 | h100 | 0.2216 | lt 0.217 (1.02)<br>lt-fresh 0.2188 (1.01)<br>cublas 0.2397 (0.92)<br>cublas-fresh 0.2405 (0.92) | torch-eager 0.2411 (0.92) | **torch-compile 0.1761 (1.26)** | not run | not run | not run |
| pj_m16_n4096_k14336 | rtx-5080 | 0.2667 | cublas-fresh 0.3174 (0.84)<br>lt-fresh 0.3174 (0.84)<br>cublas 0.3318 (0.80)<br>lt 0.3318 (0.80) | torch-eager 0.3338 (0.80) | **torch-compile 0.2744 (0.97)** | torch-compile-cg 0.2744 (0.97) | torch-compile-ro 0.3338 (0.80) | triton 0.3052 (0.87) |
| pj_m16_n4096_k14336 | rtx-3090 | 0.3119 | lt 0.2898 (1.08)<br>lt-fresh 0.2898 (1.08)<br>cublas-fresh 0.2939 (1.06)<br>cublas 0.2949 (1.06) | torch-eager 0.3144 (0.99) | **torch-compile 0.2755 (1.13)** | not run | not run | not run |
| pj_m16_n4096_k14336 | a100 | 0.2269 | **lt 0.2509 (0.90)**<br>lt-fresh 0.2509 (0.90)<br>cublas 0.2529 (0.90)<br>cublas-fresh 0.2529 (0.90) | torch-eager 0.3174 (0.71) | torch-compile 0.3164 (0.72) | not run | not run | not run |
| pj_m16_n4096_k14336 | rtx-4090 | 0.318 | **cublas-fresh 0.3021 (1.05)**<br>lt-fresh 0.3041 (1.05)<br>lt 0.3133 (1.01)<br>cublas 0.3144 (1.01) | torch-eager 0.3369 (0.94) | torch-compile 0.3103 (1.02) | not run | not run | not run |
| pj_m16_n4096_k14336 | h100 | 0.1393 | **lt 0.1159 (1.20)**<br>lt-fresh 0.1168 (1.19)<br>cublas 0.1329 (1.05)<br>cublas-fresh 0.1334 (1.04) | torch-eager 0.1351 (1.03) | torch-compile 0.1441 (0.97) | not run | not run | not run |
| pj_m16_n4096_k4096 | rtx-5080 | 0.05224 | **cublas-fresh 0.0553 (0.94)**<br>lt-fresh 0.0553 (0.94)<br>lt 0.0942 (0.55)<br>cublas 0.0963 (0.54) | torch-eager 0.0963 (0.54) | torch-compile 0.084 (0.62) | torch-compile-cg 0.0819 (0.64) | torch-compile-ro 0.0983 (0.53) | triton 0.0922 (0.57) |
| pj_m16_n4096_k4096 | rtx-3090 | 0.1372 | lt 0.1004 (1.37)<br>lt-fresh 0.1024 (1.34)<br>cublas 0.1034 (1.33)<br>cublas-fresh 0.1034 (1.33) | torch-eager 0.1085 (1.26) | **torch-compile 0.086 (1.60)** | not run | not run | not run |
| pj_m16_n4096_k4096 | a100 | 0.1004 | **cublas-fresh 0.0891 (1.13)**<br>lt-fresh 0.0891 (1.13)<br>lt 0.0922 (1.09)<br>cublas 0.0932 (1.08) | torch-eager 0.1085 (0.93) | torch-compile 0.1075 (0.93) | not run | not run | not run |
| pj_m16_n4096_k4096 | rtx-4090 | 0.0535 | **cublas-fresh 0.0348 (1.54)**<br>lt-fresh 0.0399 (1.34)<br>lt 0.1085 (0.49)<br>cublas 0.1126 (0.48) | torch-eager 0.1167 (0.46) | torch-compile 0.1044 (0.51) | not run | not run | not run |
| pj_m16_n4096_k4096 | h100 | 0.07107 | **lt-fresh 0.0479 (1.48)**<br>lt 0.048 (1.48)<br>cublas-fresh 0.091 (0.78)<br>cublas 0.0913 (0.78) | torch-eager 0.0919 (0.77) | torch-compile 0.056 (1.27) | not run | not run | not run |
| pj_m16_n6144_k4096 | rtx-5080 | 0.1009 | **lt-fresh 0.0881 (1.14)**<br>cublas-fresh 0.0963 (1.05)<br>lt 0.129 (0.78)<br>cublas 0.1413 (0.71) | torch-eager 0.1393 (0.72) | torch-compile 0.1208 (0.83) | torch-compile-cg 0.1208 (0.83) | torch-compile-ro 0.1413 (0.71) | triton 0.129 (0.78) |
| pj_m16_n6144_k4096 | rtx-3090 | 0.1664 | lt 0.1311 (1.27)<br>lt-fresh 0.1321 (1.26)<br>cublas 0.1434 (1.16)<br>cublas-fresh 0.1444 (1.15) | torch-eager 0.1465 (1.14) | **torch-compile 0.1239 (1.34)** | not run | not run | not run |
| pj_m16_n6144_k4096 | a100 | 0.1291 | lt-fresh 0.1208 (1.07)<br>lt 0.1229 (1.05)<br>cublas-fresh 0.1341 (0.96)<br>cublas 0.1362 (0.95) | torch-eager 0.1475 (0.88) | **torch-compile 0.1167 (1.11)** | not run | not run | not run |
| pj_m16_n6144_k4096 | rtx-4090 | 0.1221 | **cublas-fresh 0.1004 (1.22)**<br>lt-fresh 0.1076 (1.13)<br>lt 0.1587 (0.77)<br>cublas 0.1597 (0.76) | torch-eager 0.1597 (0.76) | torch-compile 0.1536 (0.79) | not run | not run | not run |
| pj_m16_n6144_k4096 | h100 | 0.08844 | lt 0.0677 (1.31)<br>lt-fresh 0.068 (1.30)<br>cublas-fresh 0.0943 (0.94)<br>cublas 0.0949 (0.93) | torch-eager 0.0945 (0.94) | **torch-compile 0.0572 (1.55)** | not run | not run | not run |
| pj_m1_n18432_k7168 | rtx-5080 | 0.6051 | cublas-fresh 0.6103 (0.99)<br>lt-fresh 0.6124 (0.99)<br>cublas 0.6185 (0.98)<br>lt 0.6185 (0.98) | torch-eager 0.6164 (0.98) | **torch-compile 0.6083 (0.99)** | torch-compile-cg 0.6083 (0.99) | torch-compile-ro 0.6185 (0.98) | triton 0.6369 (0.95) |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6312 | **lt-fresh 0.6062 (1.04)**<br>cublas 0.6072 (1.04)<br>lt 0.6072 (1.04)<br>cublas-fresh 0.6072 (1.04) | torch-eager 0.6062 (1.04) | torch-compile 0.6083 (1.04) | not run | not run | not run |
| pj_m1_n18432_k7168 | a100 | 0.4308 | **lt-fresh 0.3922 (1.10)**<br>lt 0.3973 (1.08)<br>cublas-fresh 0.4014 (1.07)<br>cublas 0.4198 (1.03) | torch-eager 0.4311 (1.00) | torch-compile 0.4004 (1.08) | not run | not run | not run |
| pj_m1_n18432_k7168 | rtx-4090 | 0.6274 | **lt-fresh 0.5366 (1.17)**<br>lt 0.596 (1.05)<br>cublas-fresh 0.6183 (1.01)<br>cublas 0.6267 (1.00) | torch-eager 0.6267 (1.00) | torch-compile 0.5775 (1.09) | not run | not run | not run |
| pj_m1_n18432_k7168 | h100 | 0.226 | **lt-fresh 0.1868 (1.21)**<br>cublas-fresh 0.1894 (1.19)<br>lt 0.1941 (1.16)<br>cublas 0.1964 (1.15) | torch-eager 0.1964 (1.15) | torch-compile 0.1964 (1.15) | not run | not run | not run |
| pj_m1_n2048_k7168 | rtx-5080 | 0.03351 | **cublas-fresh 0.0266 (1.26)**<br>lt-fresh 0.0266 (1.26)<br>cublas 0.0758 (0.44)<br>lt 0.0758 (0.44) | torch-eager 0.0758 (0.44) | torch-compile 0.0758 (0.44) | torch-compile-cg 0.0758 (0.44) | torch-compile-ro 0.0758 (0.44) | triton 0.2028 (0.17) |
| pj_m1_n2048_k7168 | rtx-3090 | 0.1038 | **lt-fresh 0.0748 (1.39)**<br>lt 0.0778 (1.33)<br>cublas-fresh 0.0901 (1.15)<br>cublas 0.0911 (1.14) | torch-eager 0.0942 (1.10) | torch-compile 0.084 (1.24) | not run | not run | not run |
| pj_m1_n2048_k7168 | a100 | 0.08766 | **cublas-fresh 0.044 (1.99)**<br>lt-fresh 0.044 (1.99)<br>cublas 0.0553 (1.59)<br>lt 0.0553 (1.59) | torch-eager 0.0584 (1.50) | torch-compile 0.0614 (1.43) | not run | not run | not run |
| pj_m1_n2048_k7168 | rtx-4090 | 0.0332 | **lt-fresh 0.0276 (1.20)**<br>cublas-fresh 0.044 (0.75)<br>lt 0.086 (0.39)<br>cublas 0.1064 (0.31) | torch-eager 0.1075 (0.31) | torch-compile 0.0901 (0.37) | not run | not run | not run |
| pj_m1_n2048_k7168 | h100 | 0.05269 | **cublas-fresh 0.0243 (2.17)**<br>lt-fresh 0.0254 (2.07)<br>lt 0.0305 (1.73)<br>cublas 0.0328 (1.61) | torch-eager 0.0363 (1.45) | torch-compile 0.061 (0.86) | not run | not run | not run |
| pj_m1_n28672_k4096 | rtx-5080 | 0.5624 | **lt-fresh 0.5038 (1.12)**<br>cublas-fresh 0.5202 (1.08)<br>lt 0.5407 (1.04)<br>cublas 0.5591 (1.01) | torch-eager 0.5591 (1.01) | torch-compile 0.5386 (1.04) | torch-compile-cg 0.5386 (1.04) | torch-compile-ro 0.5591 (1.01) | triton 0.5693 (0.99) |
| pj_m1_n28672_k4096 | rtx-3090 | 0.5718 | **lt 0.5376 (1.06)**<br>cublas 0.5386 (1.06)<br>cublas-fresh 0.5386 (1.06)<br>lt-fresh 0.5386 (1.06) | torch-eager 0.5386 (1.06) | torch-compile 0.5417 (1.06) | not run | not run | not run |
| pj_m1_n28672_k4096 | a100 | 0.3827 | lt-fresh 0.3594 (1.06)<br>lt 0.3697 (1.04)<br>cublas 0.4485 (0.85)<br>cublas-fresh 0.4485 (0.85) | torch-eager 0.4751 (0.81) | **torch-compile 0.3492 (1.10)** | not run | not run | not run |
| pj_m1_n28672_k4096 | rtx-4090 | 0.5685 | **lt-fresh 0.4823 (1.18)**<br>lt 0.5304 (1.07)<br>cublas-fresh 0.5622 (1.01)<br>cublas 0.5673 (1.00) | torch-eager 0.5693 (1.00) | torch-compile 0.5161 (1.10) | not run | not run | not run |
| pj_m1_n28672_k4096 | h100 | 0.2046 | **lt-fresh 0.1649 (1.24)**<br>cublas-fresh 0.1651 (1.24)<br>cublas 0.1724 (1.19)<br>lt 0.1724 (1.19) | torch-eager 0.1729 (1.18) | torch-compile 0.1732 (1.18) | not run | not run | not run |
| pj_m1_n4096_k14336 | rtx-5080 | 0.2547 | **cublas-fresh 0.2314 (1.10)**<br>lt-fresh 0.2314 (1.10)<br>cublas 0.2744 (0.93)<br>lt 0.2744 (0.93) | torch-eager 0.2744 (0.93) | torch-compile 0.2765 (0.92) | torch-compile-cg 0.2765 (0.92) | torch-compile-ro 0.2765 (0.92) | triton 0.4096 (0.62) |
| pj_m1_n4096_k14336 | rtx-3090 | 0.3025 | **lt 0.2765 (1.09)**<br>lt-fresh 0.2775 (1.09)<br>cublas 0.2785 (1.09)<br>cublas-fresh 0.2785 (1.09) | torch-eager 0.3 (1.01) | torch-compile 0.2847 (1.06) | not run | not run | not run |
| pj_m1_n4096_k14336 | a100 | 0.2021 | **cublas-fresh 0.1782 (1.13)**<br>lt-fresh 0.1782 (1.13)<br>lt 0.1843 (1.10)<br>cublas 0.1935 (1.04) | torch-eager 0.2007 (1.01) | torch-compile 0.1843 (1.10) | not run | not run | not run |
| pj_m1_n4096_k14336 | rtx-4090 | 0.2995 | **lt-fresh 0.2038 (1.47)**<br>lt 0.2693 (1.11)<br>cublas-fresh 0.298 (1.00)<br>cublas 0.3154 (0.95) | torch-eager 0.3389 (0.88) | torch-compile 0.2755 (1.09) | not run | not run | not run |
| pj_m1_n4096_k14336 | h100 | 0.1285 | **cublas-fresh 0.0824 (1.56)**<br>lt-fresh 0.0824 (1.56)<br>cublas 0.0908 (1.41)<br>lt 0.0908 (1.41) | torch-eager 0.0912 (1.41) | torch-compile 0.0998 (1.29) | not run | not run | not run |
| pj_m1_n4096_k4096 | rtx-5080 | 0.03263 | **lt-fresh 0.0205 (1.59)**<br>cublas-fresh 0.0246 (1.33)<br>lt 0.084 (0.39)<br>cublas 0.0881 (0.37) | torch-eager 0.0881 (0.37) | torch-compile 0.086 (0.38) | torch-compile-cg 0.0881 (0.37) | torch-compile-ro 0.0881 (0.37) | triton 0.127 (0.26) |
| pj_m1_n4096_k4096 | rtx-3090 | 0.113 | **lt 0.086 (1.31)**<br>lt-fresh 0.086 (1.31)<br>cublas-fresh 0.0911 (1.24)<br>cublas 0.0932 (1.21) | torch-eager 0.0973 (1.16) | torch-compile 0.0922 (1.23) | not run | not run | not run |
| pj_m1_n4096_k4096 | a100 | 0.0932 | **cublas-fresh 0.0532 (1.75)**<br>lt-fresh 0.0532 (1.75)<br>cublas 0.0676 (1.38)<br>lt 0.0676 (1.38) | torch-eager 0.0707 (1.32) | torch-compile 0.0676 (1.38) | not run | not run | not run |
| pj_m1_n4096_k4096 | rtx-4090 | 0.03228 | **lt-fresh 0.0195 (1.66)**<br>cublas-fresh 0.0223 (1.45)<br>lt 0.0952 (0.34)<br>cublas 0.0974 (0.33) | torch-eager 0.1022 (0.32) | torch-compile 0.0993 (0.33) | not run | not run | not run |
| pj_m1_n4096_k4096 | h100 | 0.06915 | **lt-fresh 0.0239 (2.89)**<br>cublas-fresh 0.024 (2.88)<br>lt 0.0316 (2.19)<br>cublas 0.0317 (2.18) | torch-eager 0.0318 (2.17) | torch-compile 0.0606 (1.14) | not run | not run | not run |
| pj_m1_n6144_k4096 | rtx-5080 | 0.09199 | **cublas-fresh 0.0512 (1.80)**<br>lt-fresh 0.0512 (1.80)<br>lt 0.1229 (0.75)<br>cublas 0.1249 (0.74) | torch-eager 0.1249 (0.74) | torch-compile 0.1208 (0.76) | torch-compile-cg 0.1229 (0.75) | torch-compile-ro 0.1229 (0.75) | triton 0.1434 (0.64) |
| pj_m1_n6144_k4096 | rtx-3090 | 0.151 | **lt-fresh 0.1229 (1.23)**<br>lt 0.1239 (1.22)<br>cublas 0.1393 (1.08)<br>cublas-fresh 0.1403 (1.08) | torch-eager 0.1403 (1.08) | torch-compile 0.1229 (1.23) | not run | not run | not run |
| pj_m1_n6144_k4096 | a100 | 0.116 | **lt-fresh 0.0778 (1.49)**<br>cublas-fresh 0.0788 (1.47)<br>lt 0.0901 (1.29)<br>cublas 0.0922 (1.26) | torch-eager 0.0932 (1.24) | torch-compile 0.0901 (1.29) | not run | not run | not run |
| pj_m1_n6144_k4096 | rtx-4090 | 0.08477 | **lt-fresh 0.0592 (1.43)**<br>cublas-fresh 0.1178 (0.72)<br>lt 0.1268 (0.67)<br>cublas 0.1597 (0.53) | torch-eager 0.1597 (0.53) | torch-compile 0.1332 (0.64) | not run | not run | not run |
| pj_m1_n6144_k4096 | h100 | 0.06935 | **cublas-fresh 0.0365 (1.90)**<br>lt-fresh 0.0366 (1.89)<br>cublas 0.0438 (1.58)<br>lt 0.0439 (1.58) | torch-eager 0.0433 (1.60) | torch-compile 0.0516 (1.34) | not run | not run | not run |
| pj_m2048_n28672_k4096 | rtx-5080 | 12.63 | **cublas-fresh 12.21 (1.03)**<br>cublas 12.21 (1.03)<br>lt-fresh 12.21 (1.03)<br>lt 12.22 (1.03) | torch-eager 12.45 (1.01) | torch-compile 12.42 (1.02) | torch-compile-cg 12.46 (1.01) | torch-compile-ro 12.46 (1.01) | triton 15.8 (0.80) |
| pj_m2048_n28672_k4096 | rtx-3090 | 21.02 | cublas-fresh 23.59 (0.89)<br>lt-fresh 23.61 (0.89)<br>cublas 23.61 (0.89)<br>lt 23.62 (0.89) | torch-eager 19.64 (1.07) | **torch-compile 18.86 (1.11)** | not run | not run | not run |
| pj_m2048_n28672_k4096 | a100 | 26.86 | cublas-fresh 26.24 (1.02)<br>cublas 26.25 (1.02)<br>lt 26.38 (1.02)<br>lt-fresh 26.39 (1.02) | torch-eager 26.46 (1.02) | **torch-compile 26.02 (1.03)** | not run | not run | not run |
| pj_m2048_n28672_k4096 | rtx-4090 | 8.673 | **lt 8.29 (1.05)**<br>cublas 8.294 (1.05)<br>cublas-fresh 8.307 (1.04)<br>lt-fresh 8.339 (1.04) | torch-eager 8.463 (1.02) | torch-compile 8.295 (1.05) | not run | not run | not run |
| pj_m2048_n28672_k4096 | h100 | 10.16 | cublas-fresh 9.263 (1.10)<br>lt-fresh 9.263 (1.10)<br>lt 9.266 (1.10)<br>cublas 9.267 (1.10) | torch-eager 9.266 (1.10) | **torch-compile 9.204 (1.10)** | not run | not run | not run |
| pj_m2048_n4096_k14336 | rtx-5080 | 6.427 | cublas-fresh 9.066 (0.71)<br>cublas 9.475 (0.68)<br>lt 9.48 (0.68)<br>lt-fresh 9.503 (0.68) | **torch-eager 6.498 (0.99)** | torch-compile 6.515 (0.99) | torch-compile-cg 6.499 (0.99) | torch-compile-ro 6.508 (0.99) | triton 8.21 (0.78) |
| pj_m2048_n4096_k14336 | rtx-3090 | 11.09 | lt 10.6 (1.05)<br>lt-fresh 10.6 (1.05)<br>cublas 11.29 (0.98)<br>cublas-fresh 11.3 (0.98) | torch-eager 11.07 (1.00) | **torch-compile 10.47 (1.06)** | not run | not run | not run |
| pj_m2048_n4096_k14336 | a100 | 13.34 | lt-fresh 13.08 (1.02)<br>lt 13.08 (1.02)<br>cublas-fresh 13.08 (1.02)<br>cublas 13.08 (1.02) | **torch-eager 12.88 (1.04)** | torch-compile 13.63 (0.98) | not run | not run | not run |
| pj_m2048_n4096_k14336 | rtx-4090 | 4.017 | cublas 4.238 (0.95)<br>lt 4.238 (0.95)<br>cublas-fresh 4.239 (0.95)<br>lt-fresh 4.239 (0.95) | torch-eager 4.329 (0.93) | **torch-compile 4.236 (0.95)** | not run | not run | not run |
| pj_m2048_n4096_k14336 | h100 | 5.24 | cublas-fresh 4.624 (1.13)<br>lt-fresh 4.624 (1.13)<br>cublas 4.625 (1.13)<br>lt 4.625 (1.13) | **torch-eager 4.533 (1.16)** | torch-compile 4.635 (1.13) | not run | not run | not run |
| pj_m2048_n4096_k4096 | rtx-5080 | 1.837 | lt-fresh 1.785 (1.03)<br>cublas-fresh 1.787 (1.03)<br>cublas 1.792 (1.03)<br>lt 1.792 (1.03) | torch-eager 1.776 (1.03) | torch-compile 1.776 (1.03) | **torch-compile-cg 1.772 (1.04)** | torch-compile-ro 1.776 (1.03) | triton 2.335 (0.79) |
| pj_m2048_n4096_k4096 | rtx-3090 | 3.01 | lt-fresh 2.897 (1.04)<br>lt 2.898 (1.04)<br>cublas 2.899 (1.04)<br>cublas-fresh 2.899 (1.04) | torch-eager 2.922 (1.03) | **torch-compile 2.869 (1.05)** | not run | not run | not run |
| pj_m2048_n4096_k4096 | a100 | 3.867 | **cublas-fresh 3.709 (1.04)**<br>lt 3.716 (1.04)<br>cublas 3.721 (1.04)<br>lt-fresh 3.737 (1.03) | torch-eager 3.75 (1.03) | torch-compile 4.811 (0.80) | not run | not run | not run |
| pj_m2048_n4096_k4096 | rtx-4090 | 1.167 | **cublas-fresh 1.173 (0.99)**<br>lt-fresh 1.175 (0.99)<br>cublas 1.176 (0.99)<br>lt 1.176 (0.99) | torch-eager 1.177 (0.99) | torch-compile 1.177 (0.99) | not run | not run | not run |
| pj_m2048_n4096_k4096 | h100 | 1.515 | lt-fresh 1.324 (1.14)<br>cublas-fresh 1.325 (1.14)<br>lt 1.327 (1.14)<br>cublas 1.328 (1.14) | torch-eager 1.327 (1.14) | **torch-compile 1.318 (1.15)** | not run | not run | not run |
| pj_m2048_n6144_k4096 | rtx-5080 | 2.648 | cublas-fresh 2.634 (1.01)<br>lt-fresh 2.634 (1.01)<br>cublas 2.64 (1.00)<br>lt 2.642 (1.00) | torch-eager 2.605 (1.02) | torch-compile 2.606 (1.02) | **torch-compile-cg 2.602 (1.02)** | torch-compile-ro 2.608 (1.02) | triton 3.42 (0.77) |
| pj_m2048_n6144_k4096 | rtx-3090 | 4.584 | **cublas 4.177 (1.10)**<br>lt 4.177 (1.10)<br>cublas-fresh 4.177 (1.10)<br>lt-fresh 4.178 (1.10) | torch-eager 4.186 (1.09) | torch-compile 4.187 (1.09) | not run | not run | not run |
| pj_m2048_n6144_k4096 | a100 | 5.804 | **lt 6.147 (0.94)**<br>cublas 6.148 (0.94)<br>lt-fresh 6.148 (0.94)<br>cublas-fresh 6.149 (0.94) | torch-eager 6.149 (0.94) | torch-compile 6.15 (0.94) | not run | not run | not run |
| pj_m2048_n6144_k4096 | rtx-4090 | 1.751 | **cublas 1.791 (0.98)**<br>cublas-fresh 1.792 (0.98)<br>lt-fresh 1.792 (0.98)<br>lt 1.793 (0.98) | torch-eager 1.798 (0.97) | torch-compile 1.862 (0.94) | not run | not run | not run |
| pj_m2048_n6144_k4096 | h100 | 2.231 | cublas-fresh 1.988 (1.12)<br>lt-fresh 1.988 (1.12)<br>lt 1.99 (1.12)<br>cublas 1.991 (1.12) | **torch-eager 1.977 (1.13)** | torch-compile 1.978 (1.13) | not run | not run | not run |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.6985 | **cublas-tf32-fresh 0.5898 (1.18)**<br>lt-tf32-fresh 0.5898 (1.18)<br>cublas-tf32 0.6226 (1.12)<br>lt-tf32 0.6226 (1.12) | torch-eager-tf32 0.6472 (1.08) | torch-compile-tf32 0.6267 (1.11) | torch-compile-cg-tf32 0.6287 (1.11) | torch-compile-ro-tf32 0.6451 (1.08) | triton-tf32 0.6267 (1.11) |
| pj_m128_n28672_k4096_tf32 | rtx-3090 | 1.071 | **cublas-tf32 0.9298 (1.15)**<br>lt-tf32 0.9298 (1.15)<br>cublas-tf32-fresh 0.9298 (1.15)<br>lt-tf32-fresh 0.9298 (1.15) | torch-eager-tf32 0.9554 (1.12) | torch-compile-tf32 0.9564 (1.12) | not run | not run | not run |
| pj_m128_n28672_k4096_tf32 | a100 | 0.5376 | **lt-tf32 0.4526 (1.19)**<br>lt-tf32-fresh 0.4526 (1.19)<br>cublas-tf32-fresh 0.4628 (1.16)<br>cublas-tf32 0.4639 (1.16) | torch-eager-tf32 0.5253 (1.02) | torch-compile-tf32 0.5018 (1.07) | not run | not run | not run |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 0.5749 | **cublas-tf32-fresh 0.5704 (1.01)**<br>lt-tf32-fresh 0.5714 (1.01)<br>cublas-tf32 0.5775 (1.00)<br>lt-tf32 0.5775 (1.00) | torch-eager-tf32 0.6236 (0.92) | torch-compile-tf32 0.599 (0.96) | not run | not run | not run |
| pj_m128_n28672_k4096_tf32 | h100 | 0.2515 | **cublas-tf32 0.1856 (1.36)**<br>lt-tf32 0.1856 (1.36)<br>lt-tf32-fresh 0.1856 (1.36)<br>cublas-tf32-fresh 0.1857 (1.35) | torch-eager-tf32 0.1887 (1.33) | torch-compile-tf32 0.2009 (1.25) | not run | not run | not run |
| pj_m128_n4096_k14336_tf32 | rtx-5080 | 0.4568 | **cublas-tf32-fresh 0.3092 (1.48)**<br>lt-tf32-fresh 0.3092 (1.48)<br>cublas-tf32 0.3195 (1.43)<br>lt-tf32 0.3195 (1.43) | torch-eager-tf32 0.3113 (1.47) | torch-compile-tf32 0.3113 (1.47) | torch-compile-cg-tf32 0.3113 (1.47) | torch-compile-ro-tf32 0.3113 (1.47) | triton-tf32 0.3707 (1.23) |
| pj_m128_n4096_k14336_tf32 | rtx-3090 | 0.7229 | lt-tf32 0.8817 (0.82)<br>lt-tf32-fresh 0.8817 (0.82)<br>cublas-tf32-fresh 1.067 (0.68)<br>cublas-tf32 1.068 (0.68) | torch-eager-tf32 1.097 (0.66) | **torch-compile-tf32 0.6216 (1.16)** | not run | not run | not run |
| pj_m128_n4096_k14336_tf32 | a100 | 0.4672 | **lt-tf32-fresh 0.2314 (2.02)**<br>cublas-tf32 0.2324 (2.01)<br>lt-tf32 0.2324 (2.01)<br>cublas-tf32-fresh 0.2324 (2.01) | torch-eager-tf32 0.2642 (1.77) | torch-compile-tf32 0.257 (1.82) | not run | not run | not run |
| pj_m128_n4096_k14336_tf32 | rtx-4090 | 0.38 | **lt-tf32-fresh 0.3072 (1.24)**<br>cublas-tf32-fresh 0.3103 (1.22)<br>lt-tf32 0.3236 (1.17)<br>cublas-tf32 0.3267 (1.16) | torch-eager-tf32 0.3308 (1.15) | torch-compile-tf32 0.3195 (1.19) | not run | not run | not run |
| pj_m128_n4096_k14336_tf32 | h100 | 0.2466 | **lt-tf32 0.1067 (2.31)**<br>lt-tf32-fresh 0.1082 (2.28)<br>cublas-tf32 0.109 (2.26)<br>cublas-tf32-fresh 0.1095 (2.25) | torch-eager-tf32 0.1109 (2.22) | torch-compile-tf32 0.1189 (2.07) | not run | not run | not run |
| pj_m128_n4096_k4096_tf32 | rtx-5080 | 0.1355 | **lt-tf32-fresh 0.0819 (1.65)**<br>cublas-tf32-fresh 0.084 (1.61)<br>lt-tf32 0.1085 (1.25)<br>cublas-tf32 0.1106 (1.23) | torch-eager-tf32 0.1106 (1.23) | torch-compile-tf32 0.1126 (1.20) | torch-compile-cg-tf32 0.1106 (1.23) | torch-compile-ro-tf32 0.1106 (1.23) | triton-tf32 0.1106 (1.23) |
| pj_m128_n4096_k4096_tf32 | rtx-3090 | 0.219 | **lt-tf32-fresh 0.1669 (1.31)**<br>lt-tf32 0.1679 (1.30)<br>cublas-tf32-fresh 0.169 (1.30)<br>cublas-tf32 0.17 (1.29) | torch-eager-tf32 0.17 (1.29) | torch-compile-tf32 0.17 (1.29) | not run | not run | not run |
| pj_m128_n4096_k4096_tf32 | a100 | 0.1482 | **cublas-tf32-fresh 0.0829 (1.79)**<br>lt-tf32-fresh 0.0829 (1.79)<br>cublas-tf32 0.085 (1.74)<br>lt-tf32 0.085 (1.74) | torch-eager-tf32 0.0952 (1.56) | torch-compile-tf32 0.0942 (1.57) | not run | not run | not run |
| pj_m128_n4096_k4096_tf32 | rtx-4090 | 0.0955 | **cublas-tf32-fresh 0.0666 (1.43)**<br>lt-tf32-fresh 0.0666 (1.43)<br>lt-tf32 0.1087 (0.88)<br>cublas-tf32 0.1097 (0.87) | torch-eager-tf32 0.1096 (0.87) | torch-compile-tf32 0.1085 (0.88) | not run | not run | not run |
| pj_m128_n4096_k4096_tf32 | h100 | 0.08557 | **cublas-tf32-fresh 0.0442 (1.94)**<br>lt-tf32-fresh 0.0446 (1.92)<br>cublas-tf32 0.045 (1.90)<br>lt-tf32 0.0463 (1.85) | torch-eager-tf32 0.0462 (1.85) | torch-compile-tf32 0.0691 (1.24) | not run | not run | not run |
| pj_m128_n6144_k4096_tf32 | rtx-5080 | 0.2368 | **lt-tf32-fresh 0.1249 (1.90)**<br>lt-tf32 0.1597 (1.48)<br>cublas-tf32-fresh 0.1802 (1.31)<br>cublas-tf32 0.1987 (1.19) | torch-eager-tf32 0.1905 (1.24) | torch-compile-tf32 0.172 (1.38) | torch-compile-cg-tf32 0.172 (1.38) | torch-compile-ro-tf32 0.1905 (1.24) | triton-tf32 0.2028 (1.17) |
| pj_m128_n6144_k4096_tf32 | rtx-3090 | 0.4239 | **lt-tf32-fresh 0.2243 (1.89)**<br>lt-tf32 0.2253 (1.88)<br>cublas-tf32 0.3267 (1.30)<br>cublas-tf32-fresh 0.3277 (1.29) | torch-eager-tf32 0.3277 (1.29) | torch-compile-tf32 0.2826 (1.50) | not run | not run | not run |
| pj_m128_n6144_k4096_tf32 | a100 | 0.1612 | **cublas-tf32-fresh 0.1024 (1.57)**<br>cublas-tf32 0.1034 (1.56)<br>lt-tf32 0.1034 (1.56)<br>lt-tf32-fresh 0.1034 (1.56) | torch-eager-tf32 0.1096 (1.47) | torch-compile-tf32 0.1096 (1.47) | not run | not run | not run |
| pj_m128_n6144_k4096_tf32 | rtx-4090 | 0.1808 | **cublas-tf32-fresh 0.1147 (1.58)**<br>lt-tf32-fresh 0.1147 (1.58)<br>lt-tf32 0.1577 (1.15)<br>cublas-tf32 0.1587 (1.14) | torch-eager-tf32 0.1597 (1.13) | torch-compile-tf32 0.1577 (1.15) | not run | not run | not run |
| pj_m128_n6144_k4096_tf32 | h100 | 0.1566 | **lt-tf32-fresh 0.0532 (2.94)**<br>lt-tf32 0.054 (2.90)<br>cublas-tf32-fresh 0.0623 (2.51)<br>cublas-tf32 0.0643 (2.44) | torch-eager-tf32 0.0643 (2.44) | torch-compile-tf32 0.0908 (1.73) | not run | not run | not run |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 9.038 | **lt-tf32-fresh 8.062 (1.12)**<br>cublas-tf32-fresh 8.063 (1.12)<br>cublas-tf32 8.065 (1.12)<br>lt-tf32 8.066 (1.12) | torch-eager-tf32 8.14 (1.11) | torch-compile-tf32 8.141 (1.11) | torch-compile-cg-tf32 8.142 (1.11) | torch-compile-ro-tf32 8.144 (1.11) | triton-tf32 8.494 (1.06) |
| pj_m2048_n28672_k4096_tf32 | rtx-3090 | 14.85 | cublas-tf32 13.5 (1.10)<br>lt-tf32 13.58 (1.09)<br>cublas-tf32-fresh 13.85 (1.07)<br>lt-tf32-fresh 13.85 (1.07) | **torch-eager-tf32 12.83 (1.16)** | torch-compile-tf32 12.84 (1.16) | not run | not run | not run |
| pj_m2048_n28672_k4096_tf32 | a100 | 5.928 | **cublas-tf32-fresh 4.153 (1.43)**<br>lt-tf32-fresh 4.153 (1.43)<br>lt-tf32 4.154 (1.43)<br>cublas-tf32 4.155 (1.43) | torch-eager-tf32 4.391 (1.35) | torch-compile-tf32 4.396 (1.35) | not run | not run | not run |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 8.235 | cublas-tf32 5.558 (1.48)<br>cublas-tf32-fresh 5.562 (1.48)<br>lt-tf32 5.619 (1.47)<br>lt-tf32-fresh 5.621 (1.47) | torch-eager-tf32 5.56 (1.48) | **torch-compile-tf32 5.542 (1.49)** | not run | not run | not run |
| pj_m2048_n28672_k4096_tf32 | h100 | 2.037 | **lt-tf32 1.294 (1.57)**<br>lt-tf32-fresh 1.294 (1.57)<br>cublas-tf32 1.52 (1.34)<br>cublas-tf32-fresh 1.521 (1.34) | torch-eager-tf32 1.465 (1.39) | torch-compile-tf32 1.417 (1.44) | not run | not run | not run |
| pj_m2048_n4096_k14336_tf32 | rtx-5080 | 4.898 | **lt-tf32-fresh 4.244 (1.15)**<br>cublas-tf32-fresh 4.245 (1.15)<br>cublas-tf32 4.248 (1.15)<br>lt-tf32 4.253 (1.15) | torch-eager-tf32 4.533 (1.08) | torch-compile-tf32 4.311 (1.14) | torch-compile-cg-tf32 4.31 (1.14) | torch-compile-ro-tf32 4.54 (1.08) | triton-tf32 4.63 (1.06) |
| pj_m2048_n4096_k14336_tf32 | rtx-3090 | 8.366 | cublas-tf32 7.241 (1.16)<br>lt-tf32 7.241 (1.16)<br>cublas-tf32-fresh 7.242 (1.16)<br>lt-tf32-fresh 7.242 (1.16) | **torch-eager-tf32 7.007 (1.19)** | torch-compile-tf32 7.419 (1.13) | not run | not run | not run |
| pj_m2048_n4096_k14336_tf32 | a100 | 2.709 | **cublas-tf32 2.147 (1.26)**<br>lt-tf32 2.148 (1.26)<br>cublas-tf32-fresh 2.148 (1.26)<br>lt-tf32-fresh 2.148 (1.26) | torch-eager-tf32 2.301 (1.18) | torch-compile-tf32 2.299 (1.18) | not run | not run | not run |
| pj_m2048_n4096_k14336_tf32 | rtx-4090 | 3.31 | **cublas-tf32-fresh 2.788 (1.19)**<br>cublas-tf32 2.789 (1.19)<br>lt-tf32-fresh 2.892 (1.14)<br>lt-tf32 2.893 (1.14) | torch-eager-tf32 2.979 (1.11) | torch-compile-tf32 2.987 (1.11) | not run | not run | not run |
| pj_m2048_n4096_k14336_tf32 | h100 | 0.9885 | cublas-tf32 0.5968 (1.66)<br>lt-tf32 0.5969 (1.66)<br>cublas-tf32-fresh 0.597 (1.66)<br>lt-tf32-fresh 0.5978 (1.65) | **torch-eager-tf32 0.5878 (1.68)** | torch-compile-tf32 0.5935 (1.67) | not run | not run | not run |
| pj_m2048_n4096_k4096_tf32 | rtx-5080 | 1.416 | **lt-tf32 1.161 (1.22)**<br>lt-tf32-fresh 1.161 (1.22)<br>cublas-tf32-fresh 1.163 (1.22)<br>cublas-tf32 1.173 (1.21) | torch-eager-tf32 1.309 (1.08) | torch-compile-tf32 1.225 (1.16) | torch-compile-cg-tf32 1.229 (1.15) | torch-compile-ro-tf32 1.3 (1.09) | triton-tf32 1.348 (1.05) |
| pj_m2048_n4096_k4096_tf32 | rtx-3090 | 2.408 | lt-tf32 2.283 (1.06)<br>lt-tf32-fresh 2.283 (1.06)<br>cublas-tf32 2.283 (1.05)<br>cublas-tf32-fresh 2.284 (1.05) | torch-eager-tf32 2.508 (0.96) | **torch-compile-tf32 2.038 (1.18)** | not run | not run | not run |
| pj_m2048_n4096_k4096_tf32 | a100 | 0.8018 | **lt-tf32 0.5857 (1.37)**<br>lt-tf32-fresh 0.5857 (1.37)<br>cublas-tf32 0.6001 (1.34)<br>cublas-tf32-fresh 0.6001 (1.34) | torch-eager-tf32 0.6922 (1.16) | torch-compile-tf32 0.6912 (1.16) | not run | not run | not run |
| pj_m2048_n4096_k4096_tf32 | rtx-4090 | 0.952 | **lt-tf32 0.8581 (1.11)**<br>lt-tf32-fresh 0.8591 (1.11)<br>cublas-tf32-fresh 0.8623 (1.10)<br>cublas-tf32 0.8632 (1.10) | torch-eager-tf32 0.8663 (1.10) | torch-compile-tf32 0.8612 (1.11) | not run | not run | not run |
| pj_m2048_n4096_k4096_tf32 | h100 | 0.3111 | **cublas-tf32-fresh 0.1759 (1.77)**<br>lt-tf32-fresh 0.1759 (1.77)<br>cublas-tf32 0.1768 (1.76)<br>lt-tf32 0.1769 (1.76) | torch-eager-tf32 0.1802 (1.73) | torch-compile-tf32 0.1788 (1.74) | not run | not run | not run |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.017 | **cublas-tf32-fresh 1.748 (1.15)**<br>lt-tf32-fresh 1.75 (1.15)<br>cublas-tf32 1.759 (1.15)<br>lt-tf32 1.759 (1.15) | torch-eager-tf32 1.862 (1.08) | torch-compile-tf32 1.8 (1.12) | torch-compile-cg-tf32 1.8 (1.12) | torch-compile-ro-tf32 1.86 (1.08) | triton-tf32 1.972 (1.02) |
| pj_m2048_n6144_k4096_tf32 | rtx-3090 | 3.413 | cublas-tf32 2.917 (1.17)<br>lt-tf32 2.917 (1.17)<br>cublas-tf32-fresh 2.918 (1.17)<br>lt-tf32-fresh 2.918 (1.17) | torch-eager-tf32 2.952 (1.16) | **torch-compile-tf32 2.871 (1.19)** | not run | not run | not run |
| pj_m2048_n6144_k4096_tf32 | a100 | 1.246 | cublas-tf32 0.9257 (1.35)<br>lt-tf32-fresh 0.9257 (1.35)<br>lt-tf32 0.9267 (1.35)<br>cublas-tf32-fresh 0.9267 (1.35) | **torch-eager-tf32 0.8161 (1.53)** | torch-compile-tf32 0.8243 (1.51) | not run | not run | not run |
| pj_m2048_n6144_k4096_tf32 | rtx-4090 | 1.423 | cublas-tf32 1.284 (1.11)<br>lt-tf32 1.284 (1.11)<br>cublas-tf32-fresh 1.284 (1.11)<br>lt-tf32-fresh 1.284 (1.11) | **torch-eager-tf32 1.196 (1.19)** | torch-compile-tf32 1.2 (1.19) | not run | not run | not run |
| pj_m2048_n6144_k4096_tf32 | h100 | 0.4608 | **lt-tf32-fresh 0.2603 (1.77)**<br>lt-tf32 0.2608 (1.77)<br>cublas-tf32-fresh 0.3186 (1.45)<br>cublas-tf32 0.3199 (1.44) | torch-eager-tf32 0.3242 (1.42) | torch-compile-tf32 0.3232 (1.43) | not run | not run | not run |

### MLP chains

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| mlp2_2048 | rtx-5080 | 1.943 | **chain-lt2 1.765 (1.10)**<br>chain-streams 1.789 (1.09)<br>chain-lt 1.846 (1.05)<br>chain-eager 1.871 (1.04)<br>chain-cutlass 2.208 (0.88) | torch-eager 1.875 (1.04) | torch-compile 1.855 (1.05) | torch-compile-cg 1.839 (1.06) | torch-compile-ro 1.855 (1.05) | triton 2.41 (0.81) |
| mlp2_2048 | rtx-3090 | 3.253 | **chain-lt2 2.98 (1.09)**<br>chain-streams 3.097 (1.05)<br>chain-lt 3.163 (1.03)<br>chain-eager 3.242 (1.00)<br>chain-cutlass 3.681 (0.88) | torch-eager 3.324 (0.98) | torch-compile 3.289 (0.99) | not run | not run | not run |
| mlp2_2048 | a100 | 4.399 | **chain-streams 3.76 (1.17)**<br>chain-lt2 3.792 (1.16)<br>chain-eager 3.974 (1.11)<br>chain-lt 4.034 (1.09)<br>chain-cutlass 4.616 (0.95) | torch-eager 4.005 (1.10) | torch-compile 3.956 (1.11) | not run | not run | not run |
| mlp2_2048 | rtx-4090 | 1.184 | **chain-cutlass 1.232 (0.96)**<br>chain-lt2 1.236 (0.96)<br>chain-lt 1.265 (0.94)<br>chain-streams 1.266 (0.94)<br>chain-eager 1.289 (0.92) | torch-eager 1.26 (0.94) | torch-compile 1.28 (0.93) | not run | not run | not run |
| mlp2_2048 | h100 | 1.511 | **chain-streams 1.36 (1.11)**<br>chain-lt2 1.36 (1.11)<br>chain-lt 1.371 (1.10)<br>chain-eager 1.394 (1.08)<br>chain-cutlass 1.693 (0.89) | torch-eager 1.413 (1.07) | torch-compile 1.407 (1.07) | not run | not run | not run |
| mlp_2048 | rtx-5080 | 0.9414 | **chain-lt 0.9114 (1.03)**<br>chain-eager 0.9236 (1.02)<br>chain-cutlass 1.089 (0.86) | torch-eager 0.9257 (1.02) | torch-compile 0.9257 (1.02) | torch-compile-cg 0.9236 (1.02) | torch-compile-ro 0.9236 (1.02) | triton 1.198 (0.79) |
| mlp_2048 | rtx-3090 | 1.621 | chain-lt 1.554 (1.04)<br>chain-eager 1.597 (1.01)<br>chain-cutlass 1.812 (0.89) | torch-eager 1.575 (1.03) | **torch-compile 1.532 (1.06)** | not run | not run | not run |
| mlp_2048 | a100 | 2.222 | **chain-lt 1.964 (1.13)**<br>chain-eager 1.988 (1.12)<br>chain-cutlass 2.307 (0.96) | torch-eager 2.044 (1.09) | torch-compile 1.978 (1.12) | not run | not run | not run |
| mlp_2048 | rtx-4090 | 0.5925 | **chain-cutlass 0.6195 (0.96)**<br>chain-lt 0.639 (0.93)<br>chain-eager 0.6502 (0.91) | torch-eager 0.6308 (0.94) | torch-compile 0.6482 (0.91) | not run | not run | not run |
| mlp_2048 | h100 | 0.761 | **chain-lt 0.6858 (1.11)**<br>chain-eager 0.699 (1.09)<br>chain-cutlass 0.8424 (0.90) | torch-eager 0.6892 (1.10) | torch-compile 0.6895 (1.10) | not run | not run | not run |
| mlp_512 | rtx-5080 | 0.038 | chain-eager 0.043 (0.88)<br>chain-lt 0.043 (0.88)<br>chain-cutlass 0.1024 (0.37) | torch-eager 0.0451 (0.84) | **torch-compile 0.0328 (1.16)** | torch-compile-cg 0.0328 (1.16) | torch-compile-ro 0.0451 (0.84) | triton 0.0328 (1.16) |
| mlp_512 | rtx-3090 | 0.07958 | **chain-lt 0.0532 (1.50)**<br>chain-eager 0.0573 (1.39)<br>chain-cutlass 0.1239 (0.64) | torch-eager 0.0563 (1.41) | torch-compile 0.1618 (0.49) | not run | not run | not run |
| mlp_512 | a100 | 0.07773 | **chain-lt 0.0625 (1.24)**<br>chain-eager 0.0655 (1.19)<br>chain-cutlass 0.2058 (0.38) | torch-eager 0.0666 (1.17) | torch-compile 0.0655 (1.19) | not run | not run | not run |
| mlp_512 | rtx-4090 | 0.03892 | **chain-lt 0.0379 (1.03)**<br>chain-eager 0.0399 (0.98)<br>chain-cutlass 0.0891 (0.44) | torch-eager 0.0389 (1.00) | torch-compile 0.0389 (1.00) | not run | not run | not run |
| mlp_512 | h100 | 0.07836 | **chain-lt 0.0354 (2.21)**<br>chain-eager 0.0387 (2.02)<br>chain-cutlass 0.1197 (0.65) | torch-eager 0.0371 (2.11) | torch-compile 0.1271 (0.62) | not run | not run | not run |
| smlp_4096_256_1024 | rtx-5080 | 0.1645 | chain-lt 0.1556 (1.06)<br>chain-cutlass 0.1577 (1.04)<br>chain-eager 0.1679 (0.98) | torch-eager 0.1772 (0.93) | **torch-compile 0.1495 (1.10)** | torch-compile-cg 0.1495 (1.10) | torch-compile-ro 0.1638 (1.00) | triton 0.1782 (0.92) |
| smlp_4096_256_1024 | rtx-3090 | 0.2825 | chain-cutlass 0.2499 (1.13)<br>chain-eager 0.2785 (1.01)<br>chain-lt 0.3308 (0.85) | torch-eager 0.3031 (0.93) | **torch-compile 0.2294 (1.23)** | not run | not run | not run |
| smlp_4096_256_1024 | a100 | 0.3455 | **chain-eager 0.2918 (1.18)**<br>chain-lt 0.3348 (1.03)<br>chain-cutlass 0.3574 (0.97) | torch-eager 0.3348 (1.03) | torch-compile 0.3451 (1.00) | not run | not run | not run |
| smlp_4096_256_1024 | rtx-4090 | 0.1083 | chain-eager 0.1341 (0.81)<br>chain-cutlass 0.1403 (0.77)<br>chain-lt 0.2017 (0.54) | torch-eager 0.1464 (0.74) | **torch-compile 0.1096 (0.99)** | not run | not run | not run |
| smlp_4096_256_1024 | h100 | 0.1353 | **chain-lt 0.1116 (1.21)**<br>chain-eager 0.1272 (1.06)<br>chain-cutlass 0.1754 (0.77) | torch-eager 0.132 (1.02) | torch-compile 0.117 (1.16) | not run | not run | not run |
| mlp2_2048_tf32 | rtx-5080 | 1.668 |  | torch-eager-tf32 1.546 (1.08) | torch-compile-tf32 1.28 ✗gate | torch-compile-cg-tf32 1.282 ✗gate | **torch-compile-ro-tf32 1.514 (1.10)** | triton-tf32 1.429 ✗gate |
| mlp2_2048_tf32 | rtx-3090 | 2.834 |  | **torch-eager-tf32 2.626 (1.08)** | torch-compile-tf32 2.222 ✗gate | not run | not run | not run |
| mlp2_2048_tf32 | a100 | 0.932 |  | torch-eager-tf32 0.7557 (1.23) | **torch-compile-tf32 0.7301 (1.28)** | not run | not run | not run |
| mlp2_2048_tf32 | rtx-4090 | 0.9645 |  | torch-eager-tf32 0.9185 (1.05) | **torch-compile-tf32 0.8489 (1.14)** | not run | not run | not run |
| mlp2_2048_tf32 | h100 | 0.5347 |  | **torch-eager-tf32 0.2437 (2.19)** | torch-compile-tf32 0.2562 (2.09) | not run | not run | not run |
| mlp_2048_tf32 | rtx-5080 | 0.8298 |  | torch-eager-tf32 0.769 (1.08) | torch-compile-tf32 0.6369 ✗gate | torch-compile-cg-tf32 0.6369 ✗gate | **torch-compile-ro-tf32 0.7598 (1.09)** | triton-tf32 0.7229 ✗gate |
| mlp_2048_tf32 | rtx-3090 | 1.395 |  | **torch-eager-tf32 1.317 (1.06)** | torch-compile-tf32 1.029 ✗gate | not run | not run | not run |
| mlp_2048_tf32 | a100 | 0.5081 |  | **torch-eager-tf32 0.3553 (1.43)** | torch-compile-tf32 0.3615 (1.41) | not run | not run | not run |
| mlp_2048_tf32 | rtx-4090 | 0.4891 |  | **torch-eager-tf32 0.4342 (1.13)** | torch-compile-tf32 0.4342 (1.13) | not run | not run | not run |
| mlp_2048_tf32 | h100 | 0.2734 |  | **torch-eager-tf32 0.1117 (2.45)** | torch-compile-tf32 0.1136 (2.41) | not run | not run | not run |

### Attention / softmax / long rows

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | hand Triton | hand CUDA |
|---|---|---:|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-5080 | 0.03914 | chain-eager 0.041 (0.95) | torch-eager 0.0492 (0.80) | torch-compile 0.0297 (1.32) | **torch-compile-cg 0.0287 (1.36)**<br>torch-sdpa-compile-cg 0.0287 (1.36) | torch-compile-ro 0.0287 (1.36) | torch-sdpa-compile 0.0307 (1.27)<br>torch-sdpa 0.0512 (0.76) | triton 0.1516 (0.26) | chain-fused 0.0348 (1.12) |
| attn_1024_64 | rtx-3090 | 0.1067 | chain-eager 0.0727 (1.47) | torch-eager 0.0901 (1.18) | torch-compile 0.0584 (1.83) | not run | not run | torch-sdpa-compile 0.0717 (1.49)<br>torch-sdpa 0.084 (1.27) | not run | **chain-fused 0.0584 (1.83)** |
| attn_1024_64 | a100 | 0.06356 | chain-eager 0.0676 (0.94) | torch-eager 0.1106 (0.57) | **torch-compile 0.0481 (1.32)** | not run | not run | torch-sdpa-compile 0.0492 (1.29)<br>torch-sdpa 0.0799 (0.80) | not run | chain-fused 0.0604 (1.05) |
| attn_1024_64 | rtx-4090 | 0.04645 | chain-eager 0.043 (1.08) | torch-eager 0.0522 (0.89) | **torch-compile 0.0317 (1.47)** | not run | not run | torch-sdpa-compile 0.0338 (1.37)<br>torch-sdpa 0.0471 (0.99) | not run | chain-fused 0.0379 (1.23) |
| attn_1024_64 | h100 | 0.09197 | chain-eager 0.0444 (2.07) | torch-eager 0.097 (0.95) | torch-compile 0.0838 (1.10) | not run | not run | torch-sdpa-compile 0.0644 (1.43)<br>torch-sdpa 0.1094 (0.84) | not run | **chain-fused 0.0384 (2.40)** |
| attn_2048_128 | rtx-5080 | 0.1177 | chain-eager 0.1393 (0.85) | torch-eager 0.1495 (0.79) | torch-compile 0.0993 (1.19) | **torch-sdpa-compile-cg 0.0942 (1.25)**<br>torch-compile-cg 0.1004 (1.17) | torch-compile-ro 0.1004 (1.17) | torch-sdpa-compile 0.0953 (1.24)<br>torch-sdpa 0.1413 (0.83) | triton 0.7741 (0.15) | chain-fused 0.127 (0.93) |
| attn_2048_64 | rtx-5080 | 0.07953 | chain-eager 0.1065 (0.75) | torch-eager 0.1208 (0.66) | torch-compile 0.0778 (1.02) | torch-sdpa-compile-cg 0.0737 (1.08)<br>torch-compile-cg 0.0758 (1.05) | **torch-compile-ro 0.0696 (1.14)** | torch-sdpa-compile 0.0748 (1.06)<br>torch-sdpa 0.1116 (0.71) | triton 0.2929 (0.27) | chain-fused 0.0942 (0.84) |
| causal_1024_256 | rtx-5080 | 15.23 |  | torch-eager 90.53 (0.17) | torch-compile 44.52 (0.34) | **torch-compile-cg 38.23 (0.40)** | torch-compile-ro 42.56 (0.36) |  |  |  |
| causal_1024_256 | rtx-3090 | 25.04 |  | torch-eager 356.1 (0.07) | **torch-compile 199.7 (0.13)** | not run | not run |  |  |  |
| causal_1024_256 | a100 | 36.8 |  | torch-eager 150.6 (0.24) | **torch-compile 72.58 (0.51)** | not run | not run |  |  |  |
| causal_1024_256 | rtx-4090 | 18.38 |  | torch-eager 189.9 (0.10) | **torch-compile 101.2 (0.18)** | not run | not run |  |  |  |
| causal_1024_256 | h100 | 38.36 |  | torch-eager 286.4 (0.13) | **torch-compile 201.1 (0.19)** | not run | not run |  |  |  |
| decode_1024_256 | rtx-5080 | 8.635 | chain-nocache 23.73 (0.36) | torch-eager 47.94 (0.18) | torch-compile 44.2 (0.20) | torch-compile-cg 39.42 (0.22) | torch-compile-ro 36.74 (0.24) |  |  | **chain-cache 16.04 (0.54)** |
| decode_1024_256 | rtx-3090 | 26.1 | chain-nocache 37.77 (0.69) | torch-eager 214 (0.12) | torch-compile 183.2 (0.14) | not run | not run |  |  | **chain-cache 33.22 (0.79)** |
| decode_1024_256 | a100 | 27.39 | chain-nocache 53.95 (0.51) | torch-eager 82.39 (0.33) | torch-compile 66.19 (0.41) | not run | not run |  |  | **chain-cache 28.66 (0.96)** |
| decode_1024_256 | rtx-4090 | 14.73 | chain-nocache 22.34 (0.66) | torch-eager 93.77 (0.16) | torch-compile 89.32 (0.16) | not run | not run |  |  | **chain-cache 19.68 (0.75)** |
| decode_1024_256 | h100 | 28.07 | chain-nocache 46.91 (0.60) | torch-eager 200.9 (0.14) | torch-compile 215.9 (0.13) | not run | not run |  |  | **chain-cache 38.44 (0.73)** |
| gen_1024_512_256 | rtx-5080 | 7.685 |  | torch-eager 39.88 (0.19) | torch-compile 22.53 (0.34) | **torch-compile-cg 20.84 (0.37)** | torch-compile-ro 21.58 (0.36) |  |  |  |
| gen_1024_512_256 | rtx-3090 | 12.81 |  | torch-eager 191.7 (0.07) | **torch-compile 100.7 (0.13)** | not run | not run |  |  |  |
| gen_1024_512_256 | a100 | 18.53 |  | torch-eager 78.43 (0.24) | **torch-compile 38.55 (0.48)** | not run | not run |  |  |  |
| gen_1024_512_256 | rtx-4090 | 9.32 |  | torch-eager 95.03 (0.10) | **torch-compile 50.58 (0.18)** | not run | not run |  |  |  |
| gen_1024_512_256 | h100 | 12.2 |  | torch-eager 164.8 (0.07) | **torch-compile 96.46 (0.13)** | not run | not run |  |  |  |
| lsm_1_131072 | rtx-5080 | 0.01582 |  | torch-eager 0.0348 (0.45) | torch-compile 0.0246 (0.64) | **torch-compile-cg 0.0225 (0.70)**<br>torch-sdpa-compile-cg 0.0225 (0.70) | torch-compile-ro 0.0225 (0.70) | torch-sdpa-compile 0.0256 (0.62)<br>torch-sdpa 0.0512 (0.31) |  |  |
| lsm_1_131072 | rtx-3090 | 0.3208 |  | torch-eager 0.0471 (6.81) | **torch-compile 0.0287 (11.18)** | not run | not run | torch-sdpa-compile 0.0317 (10.12)<br>torch-sdpa 0.0737 (4.35) |  |  |
| lsm_1_131072 | a100 | 0.3543 |  | torch-eager 0.0666 (5.32) | **torch-compile 0.0328 (10.80)** | not run | not run | torch-sdpa-compile 0.0338 (10.48)<br>torch-sdpa 0.086 (4.12) |  |  |
| lsm_1_131072 | rtx-4090 | 0.2232 |  | torch-eager 0.0389 (5.74) | torch-compile 0.0276 (8.09) | not run | not run | **torch-sdpa-compile 0.0266 (8.39)**<br>torch-sdpa 0.0533 (4.19) |  |  |
| lsm_1_131072 | h100 | 0.3387 |  | torch-eager 0.0703 (4.82) | torch-compile 0.1145 (2.96) | not run | not run | **torch-sdpa 0.0583 (5.81)**<br>torch-sdpa-compile 0.1531 (2.21) |  |  |
| lsm_32_8192 | rtx-5080 | 0.01538 |  | torch-eager 0.0215 (0.72) | **torch-compile 0.0102 (1.51)** | torch-compile-cg 0.0102 (1.51)<br>torch-sdpa-compile-cg 0.0102 (1.51) | torch-compile-ro 0.0102 (1.51) | torch-sdpa-compile 0.0102 (1.51)<br>torch-sdpa 0.0123 (1.25) | triton 0.0102 (1.51) |  |
| lsm_32_8192 | rtx-3090 | 0.04146 |  | torch-eager 0.0317 (1.31) | **torch-compile 0.0154 (2.69)** | not run | not run | torch-sdpa 0.0154 (2.69)<br>torch-sdpa-compile 0.0164 (2.53) | not run |  |
| lsm_32_8192 | a100 | 0.0307 |  | torch-eager 0.042 (0.73) | **torch-compile 0.0184 (1.67)** | not run | not run | torch-sdpa-compile 0.0184 (1.67)<br>torch-sdpa 0.0205 (1.50) | not run |  |
| lsm_32_8192 | rtx-4090 | 0.02688 |  | torch-eager 0.0225 (1.19) | **torch-compile 0.011 (2.44)** | not run | not run | torch-sdpa 0.0113 (2.38)<br>torch-sdpa-compile 0.0113 (2.38) | not run |  |
| lsm_32_8192 | h100 | 0.04112 |  | torch-eager 0.0632 (0.65) | torch-compile 0.0728 (0.56) | not run | not run | **torch-sdpa 0.0163 (2.52)**<br>torch-sdpa-compile 0.0689 (0.60) | not run |  |
| lsm_4_32768 | rtx-5080 | 0.0166 |  | torch-eager 0.0256 (0.65) | **torch-compile 0.0143 (1.16)** | torch-compile-cg 0.0143 (1.16)<br>torch-sdpa-compile-cg 0.0143 (1.16) | torch-compile-ro 0.0143 (1.16) | torch-sdpa-compile 0.0143 (1.16)<br>torch-sdpa 0.0184 (0.90) |  |  |
| lsm_4_32768 | rtx-3090 | 0.08762 |  | torch-eager 0.0399 (2.20) | torch-compile 0.043 (2.04) | not run | not run | **torch-sdpa 0.0246 (3.56)**<br>torch-sdpa-compile 0.0379 (2.31) |  |  |
| lsm_4_32768 | a100 | 0.0732 |  | torch-eager 0.0512 (1.43) | torch-compile 0.0328 (2.23) | not run | not run | **torch-sdpa 0.0307 (2.38)**<br>torch-sdpa-compile 0.0328 (2.23) |  |  |
| lsm_4_32768 | rtx-4090 | 0.05782 |  | torch-eager 0.0287 (2.01) | torch-compile 0.0212 (2.73) | not run | not run | **torch-sdpa 0.0195 (2.97)**<br>torch-sdpa-compile 0.0266 (2.17) |  |  |
| lsm_4_32768 | h100 | 0.06518 |  | torch-eager 0.049 (1.33) | torch-compile 0.082 (0.79) | not run | not run | **torch-sdpa 0.023 (2.83)**<br>torch-sdpa-compile 0.0938 (0.69) |  |  |
| prefill_1024_512_256 | rtx-5080 | 7.743 |  | torch-eager 39.81 (0.19) | torch-compile 22.49 (0.34) | **torch-compile-cg 19.26 (0.40)** | torch-compile-ro 21.52 (0.36) |  |  |  |
| prefill_1024_512_256 | rtx-3090 | 13.16 |  | torch-eager 179.5 (0.07) | **torch-compile 98.09 (0.13)** | not run | not run |  |  |  |
| prefill_1024_512_256 | a100 | 18.87 |  | torch-eager 78.04 (0.24) | **torch-compile 38.52 (0.49)** | not run | not run |  |  |  |
| prefill_1024_512_256 | rtx-4090 | 9.51 |  | torch-eager 95.88 (0.10) | **torch-compile 50.58 (0.19)** | not run | not run |  |  |  |
| prefill_1024_512_256 | h100 | 13.2 |  | torch-eager 165.1 (0.08) | **torch-compile 99.58 (0.13)** | not run | not run |  |  |  |
| rowsum_2048 | rtx-5080 | 0.4593 | **chain-eager 0.4608 (1.00)**<br>chain-gemv 0.4628 (0.99) | torch-eager 0.4649 (0.99) | torch-compile 0.4669 (0.98) | torch-compile-cg 0.4669 (0.98) | torch-compile-ro 0.4649 (0.99) |  |  |  |
| rowsum_2048 | rtx-3090 | 0.8528 | chain-gemv 0.7752 (1.10)<br>chain-eager 0.7782 (1.10) | torch-eager 0.7813 (1.09) | **torch-compile 0.7731 (1.10)** | not run | not run |  |  |  |
| rowsum_2048 | a100 | 1.035 | **chain-eager 0.9964 (1.04)**<br>chain-gemv 1 (1.03) | torch-eager 1.002 (1.03) | torch-compile 0.9964 (1.04) | not run | not run |  |  |  |
| rowsum_2048 | rtx-4090 | 0.3122 | chain-eager 0.3277 (0.95)<br>chain-gemv 0.3297 (0.95) | torch-eager 0.3318 (0.94) | **torch-compile 0.3187 (0.98)** | not run | not run |  |  |  |
| rowsum_2048 | h100 | 0.4147 | chain-eager 0.3498 (1.19)<br>chain-gemv 0.3516 (1.18) | torch-eager 0.3482 (1.19) | **torch-compile 0.3478 (1.19)** | not run | not run |  |  |  |
| rowsum_4096 | rtx-5080 | 3.733 | chain-gemv 3.597 (1.04)<br>chain-eager 3.606 (1.04) | torch-eager 3.514 (1.06) | **torch-compile 3.486 (1.07)** | torch-compile-cg 3.488 (1.07) | torch-compile-ro 3.49 (1.07) |  |  |  |
| rowsum_4096 | rtx-3090 | 6.225 | **chain-eager 5.807 (1.07)**<br>chain-gemv 5.842 (1.07) | torch-eager 5.865 (1.06) | torch-compile 5.894 (1.06) | not run | not run |  |  |  |
| rowsum_4096 | a100 | 7.749 | chain-eager 7.346 (1.05)<br>chain-gemv 7.349 (1.05) | torch-eager 7.351 (1.05) | **torch-compile 7.339 (1.06)** | not run | not run |  |  |  |
| rowsum_4096 | rtx-4090 | 2.379 | chain-gemv 2.441 (0.97)<br>chain-eager 2.46 (0.97) | **torch-eager 2.387 (1.00)** | torch-compile 2.473 (0.96) | not run | not run |  |  |  |
| rowsum_4096 | h100 | 2.996 | chain-eager 2.692 (1.11)<br>chain-gemv 2.696 (1.11) | **torch-eager 2.683 (1.12)** | torch-compile 2.698 (1.11) | not run | not run |  |  |  |
| softmax_rows_2048 | rtx-5080 | 0.4727 | chain-eager 0.5018 (0.94) | torch-eager 0.5202 (0.91) | **torch-compile 0.469 (1.01)** | torch-compile-cg 0.471 (1.00) | torch-compile-ro 0.4731 (1.00) |  | triton 0.6042 (0.78) | chain-fused 0.4895 (0.97) |
| softmax_rows_4096 | rtx-5080 | 3.803 | chain-eager 3.907 (0.97) | torch-eager 4.117 (0.92) | torch-compile 3.54 (1.07) | torch-compile-cg 3.541 (1.07) | **torch-compile-ro 3.529 (1.08)** |  | triton 4.36 (0.87) | chain-fused 3.748 (1.01) |
| softmax_rows_4096 | rtx-3090 | 6.444 | chain-eager 6.229 (1.03) | torch-eager 6.488 (0.99) | **torch-compile 5.964 (1.08)** | not run | not run |  | not run | chain-fused 6.118 (1.05) |
| softmax_rows_4096 | a100 | 7.817 | chain-eager 7.631 (1.02) | torch-eager 7.827 (1.00) | **torch-compile 7.378 (1.06)** | not run | not run |  | not run | chain-fused 7.568 (1.03) |
| softmax_rows_4096 | rtx-4090 | 2.442 | chain-eager 2.681 (0.91) | torch-eager 2.936 (0.83) | **torch-compile 2.451 (1.00)** | not run | not run |  | not run | chain-fused 2.573 (0.95) |
| softmax_rows_4096 | h100 | 3.024 | chain-eager 2.849 (1.06) | torch-eager 2.95 (1.03) | **torch-compile 2.712 (1.12)** | not run | not run |  | not run | chain-fused 2.833 (1.07) |

### LLM decode / prefill / block (f32, tf32, bf16w)

| workload | part | Mapal ms | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | FlashAttention-2/3 | FlashInfer | hand Triton | Liger |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | rtx-5080 | 1.071 | torch-eager 1.137 (0.94) | **torch-compile 1.066 (1.01)** | torch-compile-cg 1.078 (0.99) | torch-compile-ro 1.11 (0.96) | torch-sdpa 1.555 (0.69)<br>torch-sdpa-efficient 1.867 (0.57) |  |  |  | liger 1.864 (0.57) |
| llm_blkdec_b1_l4096 | rtx-3090 | 1.133 | torch-eager 1.569 (0.72) | **torch-compile 1.203 (0.94)** | not run | not run | torch-sdpa 1.979 (0.57) |  |  |  | not run |
| llm_blkdec_b1_l4096 | a100 | 0.9588 | torch-eager 1.058 (0.91) | **torch-compile 0.8421 (1.14)** | not run | not run | torch-sdpa 1.318 (0.73) |  |  |  | not run |
| llm_blkdec_b1_l4096 | rtx-4090 | 1.494 | torch-eager 1.236 (1.21) | **torch-compile 1.053 (1.42)** | not run | not run | torch-sdpa 1.565 (0.95) |  |  |  | not run |
| llm_blkdec_b1_l4096 | h100 | 0.5689 | torch-eager 0.6957 (0.82) | **torch-compile 0.4535 (1.25)** | not run | not run | torch-sdpa 0.7366 (0.77) |  |  |  | not run |
| llm_blkpre_t2048 | rtx-5080 | 27.28 | torch-eager 33.04 (0.83) | torch-compile 28.09 (0.97) | torch-compile-cg 27.97 (0.98) | torch-compile-ro 29.04 (0.94) | torch-sdpa-efficient 27.16 (1.00)<br>torch-sdpa 33.45 (0.82) |  |  | triton 29.45 (0.93) | **liger 26.67 (1.02)** |
| llm_blkpre_t2048 | rtx-3090 | 45.99 | torch-eager 53.11 (0.87) | **torch-compile 49.42 (0.93)** | not run | not run | torch-sdpa 59.31 (0.78) |  |  | not run | not run |
| llm_blkpre_t2048 | a100 | 55.87 | torch-eager 61.5 (0.91) | **torch-compile 55.18 (1.01)** | not run | not run | torch-sdpa 61.88 (0.90) |  |  | not run | not run |
| llm_blkpre_t2048 | rtx-4090 | 18.55 | torch-eager 22.79 (0.81) | **torch-compile 18.98 (0.98)** | not run | not run | torch-sdpa 22.57 (0.82) |  |  | not run | not run |
| llm_blkpre_t2048 | h100 | 21.58 | **torch-eager 21.13 (1.02)** | not run | not run | not run | not run |  |  | not run | not run |
| llm_dec_b1_l2048 | rtx-5080 | 0.24 | torch-eager 0.2804 (0.86) | torch-compile 0.2536 (0.95) | torch-compile-cg 0.2636 (0.91) | **torch-compile-ro 0.2532 (0.95)** | torch-sdpa 0.4747 (0.51)<br>torch-sdpa-efficient 0.6538 (0.37) |  |  |  | liger 0.6613 (0.36) |
| llm_dec_b1_l2048 | rtx-3090 | 0.2774 | torch-eager 0.6875 (0.40) | **torch-compile 0.3642 (0.76)** | not run | not run | torch-sdpa 0.8194 (0.34) |  |  |  | not run |
| llm_dec_b1_l2048 | a100 | 0.2153 | torch-eager 0.341 (0.63) | **torch-compile 0.2488 (0.87)** | not run | not run | torch-sdpa 0.5087 (0.42) |  |  |  | not run |
| llm_dec_b1_l2048 | rtx-4090 | 0.242 | torch-eager 0.4217 (0.57) | **torch-compile 0.2737 (0.88)** | not run | not run | torch-sdpa 0.5089 (0.48) |  |  |  | not run |
| llm_dec_b1_l2048 | h100 | 0.1158 | torch-eager 0.5078 (0.23) | **torch-compile 0.2469 (0.47)** | not run | not run | torch-sdpa 0.5076 (0.23) |  |  |  | not run |
| llm_dec_b1_l8192 | rtx-5080 | 0.3324 | torch-eager 0.4058 (0.82) | **torch-compile 0.342 (0.97)** | torch-compile-cg 0.3468 (0.96) | torch-compile-ro 0.3619 (0.92) | torch-sdpa 1.274 (0.26)<br>torch-sdpa-efficient 1.914 (0.17) |  |  |  | liger 1.917 (0.17) |
| llm_dec_b1_l8192 | rtx-3090 | 0.3777 | torch-eager 0.7839 (0.48) | **torch-compile 0.4557 (0.83)** | not run | not run | torch-sdpa 1.822 (0.21) |  |  |  | not run |
| llm_dec_b1_l8192 | a100 | 0.2756 | torch-eager 0.548 (0.50) | **torch-compile 0.3942 (0.70)** | not run | not run | torch-sdpa 1.035 (0.27) |  |  |  | not run |
| llm_dec_b1_l8192 | rtx-4090 | 0.3439 | torch-eager 0.4934 (0.70) | **torch-compile 0.3384 (1.02)** | not run | not run | torch-sdpa 1.232 (0.28) |  |  |  | not run |
| llm_dec_b1_l8192 | h100 | 0.1659 | torch-eager 0.7677 (0.22) | **torch-compile 0.3386 (0.49)** | not run | not run | torch-sdpa 0.7815 (0.21) |  |  |  | not run |
| llm_dec_b32_l2048 | rtx-5080 | 1.033 | torch-eager 2.591 (0.40) | **torch-compile 2.256 (0.46)** | torch-compile-cg 2.272 (0.45) | torch-compile-ro 2.357 (0.44) | torch-sdpa-efficient 7.964 (0.13)<br>torch-sdpa 8.786 (0.12) |  |  |  | liger 7.983 (0.13) |
| llm_dec_b32_l2048 | rtx-3090 | 1.224 | torch-eager 2.935 (0.42) | **torch-compile 2.696 (0.45)** | not run | not run | torch-sdpa 9.054 (0.14) |  |  |  | not run |
| llm_dec_b32_l2048 | a100 | 0.8594 | torch-eager 1.989 (0.43) | **torch-compile 1.842 (0.47)** | not run | not run | torch-sdpa 6.103 (0.14) |  |  |  | not run |
| llm_dec_b32_l2048 | rtx-4090 | 1.129 | torch-eager 2.449 (0.46) | **torch-compile 2.309 (0.49)** | not run | not run | torch-sdpa 7.975 (0.14) |  |  |  | not run |
| llm_dec_b32_l2048 | h100 | 0.4715 | torch-eager 1.471 (0.32) | **torch-compile 1.182 (0.40)** | not run | not run | torch-sdpa 3.783 (0.12) |  |  |  | not run |
| llm_dec_b32_l8192 | rtx-5080 | 3.506 | torch-eager 10.68 (0.33) | **torch-compile 9.484 (0.37)** | torch-compile-cg 9.494 (0.37) | torch-compile-ro 9.948 (0.35) | torch-sdpa-efficient 29.97 (0.12) |  |  |  |  |
| llm_dec_b32_l8192 | rtx-3090 | 4.282 | torch-eager 11.95 (0.36) | **torch-compile 10.83 (0.40)** | not run | not run | torch-sdpa 41.17 (0.10) |  |  |  |  |
| llm_dec_b32_l8192 | a100 | 2.535 | torch-eager 6.995 (0.36) | **torch-compile 6.525 (0.39)** | not run | not run | torch-sdpa 23.74 (0.11) |  |  |  |  |
| llm_dec_b32_l8192 | rtx-4090 | 3.765 | torch-eager 10.78 (0.35) | **torch-compile 9.405 (0.40)** | not run | not run | torch-sdpa 30.6 (0.12) |  |  |  |  |
| llm_dec_b32_l8192 | h100 | 1.038 | torch-eager 4.502 (0.23) | **torch-compile 3.991 (0.26)** | not run | not run | torch-sdpa 13.71 (0.08) |  |  |  |  |
| llm_dec_b8_l2048 | rtx-5080 | 0.4385 | torch-eager 0.7942 (0.55) | **torch-compile 0.6685 (0.66)** | torch-compile-cg 0.6806 (0.64) | torch-compile-ro 0.7235 (0.61) | torch-sdpa 2.369 (0.19)<br>torch-sdpa-efficient 2.402 (0.18) |  |  |  | liger 2.411 (0.18) |
| llm_dec_b8_l2048 | rtx-3090 | 1.023 | torch-eager 1.144 (0.89) | **torch-compile 0.8885 (1.15)** | not run | not run | torch-sdpa 2.689 (0.38) |  |  |  | not run |
| llm_dec_b8_l2048 | a100 | 0.3983 | torch-eager 0.8001 (0.50) | **torch-compile 0.6228 (0.64)** | not run | not run | torch-sdpa 1.818 (0.22) |  |  |  | not run |
| llm_dec_b8_l2048 | rtx-4090 | 0.4967 | torch-eager 0.8186 (0.61) | **torch-compile 0.7128 (0.70)** | not run | not run | torch-sdpa 2.238 (0.22) |  |  |  | not run |
| llm_dec_b8_l2048 | h100 | 0.207 | torch-eager 0.7129 (0.29) | **torch-compile 0.469 (0.44)** | not run | not run | torch-sdpa 1.255 (0.16) |  |  |  | not run |
| llm_dec_b8_l8192 | rtx-5080 | 1.067 | torch-eager 2.831 (0.38) | torch-compile 2.488 (0.43) | **torch-compile-cg 2.468 (0.43)** | torch-compile-ro 2.529 (0.42) | torch-sdpa-efficient 8.634 (0.12)<br>torch-sdpa 8.663 (0.12) |  |  |  | liger 8.659 (0.12) |
| llm_dec_b8_l8192 | rtx-3090 | 1.801 | torch-eager 3.404 (0.53) | **torch-compile 2.939 (0.61)** | not run | not run | torch-sdpa 10.71 (0.17) |  |  |  | not run |
| llm_dec_b8_l8192 | a100 | 0.8091 | torch-eager 2.04 (0.40) | **torch-compile 1.823 (0.44)** | not run | not run | torch-sdpa 6.159 (0.13) |  |  |  | not run |
| llm_dec_b8_l8192 | rtx-4090 | 1.132 | torch-eager 2.934 (0.39) | **torch-compile 2.532 (0.45)** | not run | not run | torch-sdpa 7.903 (0.14) |  |  |  | not run |
| llm_dec_b8_l8192 | h100 | 0.4684 | torch-eager 1.529 (0.31) | **torch-compile 1.147 (0.41)** | not run | not run | torch-sdpa 3.825 (0.12) |  |  |  | not run |
| llm_pre_t2048 | rtx-5080 | 8.56 | torch-eager 13.43 (0.64) | torch-compile 8.367 (1.02) | torch-compile-cg 8.321 (1.03) | torch-compile-ro 9.35 (0.92) | torch-sdpa-efficient 7.286 (1.17)<br>torch-sdpa 13.68 (0.63) |  |  | triton 9.424 (0.91) | **liger 7.2 (1.19)** |
| llm_pre_t2048 | rtx-3090 | 14.46 | torch-eager 16.94 (0.85) | **torch-compile 12.44 (1.16)** | not run | not run | torch-sdpa 17.15 (0.84) |  |  | not run | not run |
| llm_pre_t2048 | a100 | 15.13 | torch-eager 17.62 (0.86) | **torch-compile 15.01 (1.01)** | not run | not run | torch-sdpa 17.9 (0.85) |  |  | not run | not run |
| llm_pre_t2048 | rtx-4090 | 6.318 | torch-eager 9.913 (0.64) | **torch-compile 6.063 (1.04)** | not run | not run | torch-sdpa 9.686 (0.65) |  |  | not run | not run |
| llm_pre_t2048 | h100 | 6.153 | torch-eager 7.016 (0.88) | **torch-compile 5.517 (1.12)** | not run | not run | torch-sdpa 7.149 (0.86) |  |  | not run | not run |
| llm_pre_t8192 | rtx-5080 | 80.68 | torch-eager 166 (0.49) | torch-compile 84.52 (0.95) | torch-compile-cg 83.79 (0.96) | torch-compile-ro 100.1 (0.81) | torch-sdpa-efficient 53.58 (1.51) |  |  | triton 84.28 (0.96) | **liger 52.83 (1.53)** |
| llm_pre_t8192 | rtx-3090 | 1731 | torch-eager 176.9 (9.78) | **torch-compile 126 (13.74)** | not run | not run | torch-sdpa 199.2 (8.69) |  |  | not run | not run |
| llm_pre_t8192 | a100 | 1431 | torch-eager 163.8 (8.73) | **torch-compile 116.3 (12.30)** | not run | not run | torch-sdpa 160.4 (8.92) |  |  | not run | not run |
| llm_pre_t8192 | rtx-4090 | 753.2 | torch-eager 117.6 (6.40) | **torch-compile 55.8 (13.50)** | not run | not run | torch-sdpa 119.2 (6.32) |  |  | not run | not run |
| llm_pre_t8192 | h100 | 827.9 | torch-eager 68.05 (12.17) | **torch-compile 43.43 (19.06)** | not run | not run | torch-sdpa 70.46 (11.75) |  |  | not run | not run |
| llm_blkpre_t2048_tf32 | rtx-5080 | 21.83 | torch-eager-tf32 24.26 (0.90) | torch-compile-tf32 18.56 (1.18) | torch-compile-cg-tf32 18.58 (1.17) | torch-compile-ro-tf32 19.56 (1.12) | torch-sdpa-tf32 23.94 (0.91) |  |  | **triton-tf32 18.08 (1.21)** | liger-tf32 19.12 (1.14) |
| llm_blkpre_t2048_tf32 | rtx-3090 | 33.89 | torch-eager-tf32 35.23 (0.96) | **torch-compile-tf32 32.39 (1.05)** | not run | not run | torch-sdpa-tf32 37.6 (0.90) |  |  | not run | not run |
| llm_blkpre_t2048_tf32 | a100 | 16.37 | torch-eager-tf32 13.46 (1.22) | **torch-compile-tf32 11.13 (1.47)** | not run | not run | torch-sdpa-tf32 13.73 (1.19) |  |  | not run | not run |
| llm_blkpre_t2048_tf32 | rtx-4090 | 16.84 | torch-eager-tf32 17.88 (0.94) | **torch-compile-tf32 13.46 (1.25)** | not run | not run | torch-sdpa-tf32 17.68 (0.95) |  |  | not run | not run |
| llm_blkpre_t2048_tf32 | h100 | 6.719 | torch-eager-tf32 5.066 (1.33) | **torch-compile-tf32 3.51 (1.91)** | not run | not run | torch-sdpa-tf32 5.197 (1.29) |  |  | not run | not run |
| llm_pre_t2048_tf32 | rtx-5080 | 7.455 | torch-eager-tf32 10.76 (0.69) | torch-compile-tf32 6.062 (1.23) | torch-compile-cg-tf32 6.091 (1.22) | torch-compile-ro-tf32 6.382 (1.17) | torch-sdpa-tf32 10.39 (0.72) |  |  | **triton-tf32 4.537 (1.64)** | liger-tf32 5.926 (1.26) |
| llm_pre_t2048_tf32 | rtx-3090 | 12.75 | torch-eager-tf32 14 (0.91) | **torch-compile-tf32 8.922 (1.43)** | not run | not run | torch-sdpa-tf32 14.19 (0.90) |  |  | not run | not run |
| llm_pre_t2048_tf32 | a100 | 7.412 | torch-eager-tf32 6.469 (1.15) | **torch-compile-tf32 3.655 (2.03)** | not run | not run | torch-sdpa-tf32 6.731 (1.10) |  |  | not run | not run |
| llm_pre_t2048_tf32 | rtx-4090 | 5.821 | torch-eager-tf32 8.816 (0.66) | **torch-compile-tf32 4.822 (1.21)** | not run | not run | torch-sdpa-tf32 8.609 (0.68) |  |  | not run | not run |
| llm_pre_t2048_tf32 | h100 | 3.242 | torch-eager-tf32 2.968 (1.09) | **torch-compile-tf32 1.529 (2.12)** | not run | not run | torch-sdpa-tf32 3.079 (1.05) |  |  | not run | not run |
| llm_blkdec_b1_l4096_bf16w | rtx-5080 | 0.636 | torch-bf16 0.6244 (1.02) | **torch-bf16-compile 0.553 (1.15)** | torch-bf16-compile-cg 0.5636 (1.13) | torch-bf16-compile-ro 0.9708 (0.66) | torch-bf16-cudnn 0.6171 (1.03)<br>torch-bf16-efficient 0.9613 (0.66) | fa2-bf16 0.6159 (1.03) | flashinfer-bf16 0.7054 (0.90) |  | liger-bf16 0.5893 (1.08) |
| llm_blkdec_b1_l4096_bf16w | rtx-3090 | 0.7123 | torch-bf16 1.086 (0.66) | **torch-bf16-compile 0.6788 (1.05)** | not run | not run | not run | not run | not run |  | not run |
| llm_blkdec_b1_l4096_bf16w | a100 | 0.7899 | torch-bf16 0.5719 (1.38) | **torch-bf16-compile 0.4515 (1.75)** | not run | not run | not run | not run | not run |  | not run |
| llm_blkdec_b1_l4096_bf16w | rtx-4090 | 0.5946 | torch-bf16 0.735 (0.81) | **torch-bf16-compile 0.572 (1.04)** | not run | not run | not run | not run | not run |  | not run |
| llm_blkdec_b1_l4096_bf16w | h100 | 0.3523 | torch-bf16 0.8354 (0.42) | **torch-bf16-compile 0.4288 (0.82)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16w | rtx-5080 | 0.1501 | torch-bf16 0.1962 (0.77) | **torch-bf16-compile 0.1471 (1.02)** | torch-bf16-compile-cg 0.1542 (0.97) | torch-bf16-compile-ro 0.2913 (0.52) | torch-bf16-cudnn 0.1853 (0.81)<br>torch-bf16-efficient 0.3393 (0.44) | fa2-bf16 0.1848 (0.81) | flashinfer-bf16 0.254 (0.59) |  | liger-bf16 0.1781 (0.84) |
| llm_dec_b1_l2048_bf16w | rtx-3090 | 0.1854 | torch-bf16 0.5496 (0.34) | **torch-bf16-compile 0.2502 (0.74)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16w | a100 | 0.1698 | torch-bf16 0.2415 (0.70) | **torch-bf16-compile 0.1551 (1.09)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16w | rtx-4090 | 0.1542 | torch-bf16 0.3055 (0.50) | **torch-bf16-compile 0.1724 (0.89)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16w | h100 | 0.09614 | torch-bf16 0.4988 (0.19) | **torch-bf16-compile 0.2931 (0.33)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16w | rtx-5080 | 0.2405 | torch-bf16 0.2184 (1.10) | torch-bf16-compile 0.1913 (1.26) | torch-bf16-compile-cg 0.1983 (1.21) | torch-bf16-compile-ro 1.006 (0.24) | torch-bf16-cudnn 0.1921 (1.25)<br>torch-bf16-efficient 0.8834 (0.27) | fa2-bf16 0.2003 (1.20) | flashinfer-bf16 0.2932 (0.82) |  | **liger-bf16 0.1883 (1.28)** |
| llm_dec_b1_l8192_bf16w | rtx-3090 | 0.2869 | torch-bf16 0.5645 (0.51) | **torch-bf16-compile 0.3018 (0.95)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16w | a100 | 0.2305 | torch-bf16 0.2605 (0.88) | **torch-bf16-compile 0.1981 (1.16)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16w | rtx-4090 | 0.2575 | torch-bf16 0.3134 (0.82) | **torch-bf16-compile 0.2076 (1.24)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16w | h100 | 0.1412 | torch-bf16 0.2744 (0.51) | **torch-bf16-compile 0.1675 (0.84)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16w | rtx-5080 | 0.9551 | torch-bf16 0.5885 (1.62) | **torch-bf16-compile 0.4548 (2.10)** | torch-bf16-compile-cg 0.4687 (2.04) | torch-bf16-compile-ro 7.621 (0.13) | torch-bf16-cudnn 0.4881 (1.96)<br>torch-bf16-efficient 3.743 (0.26) | fa2-bf16 0.5237 (1.82) | flashinfer-bf16 0.5133 (1.86) |  | liger-bf16 0.5728 (1.67) |
| llm_dec_b32_l2048_bf16w | rtx-3090 | 1.196 | torch-bf16 1.146 (1.04) | **torch-bf16-compile 0.7245 (1.65)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16w | a100 | 0.8665 | torch-bf16 0.5014 (1.73) | **torch-bf16-compile 0.4747 (1.83)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16w | rtx-4090 | 1.023 | torch-bf16 0.5914 (1.73) | **torch-bf16-compile 0.4542 (2.25)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16w | h100 | 0.4911 | torch-bf16 0.3348 (1.47) | **torch-bf16-compile 0.3037 (1.62)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16w | rtx-5080 | 3.426 | torch-bf16 1.802 (1.90) | torch-bf16-compile 1.49 (2.30) | torch-bf16-compile-cg 1.477 (2.32) | torch-bf16-compile-ro 29.82 (0.11) | torch-bf16-cudnn 1.417 (2.42)<br>torch-bf16-efficient 14.28 (0.24) | fa2-bf16 1.555 (2.20) | **flashinfer-bf16 1.407 (2.43)** |  | liger-bf16 1.793 (1.91) |
| llm_dec_b32_l8192_bf16w | rtx-3090 | 4.244 | torch-bf16 3.027 (1.40) | **torch-bf16-compile 2.412 (1.76)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16w | a100 | 2.541 | **torch-bf16 1.206 (2.11)** | torch-bf16-compile 1.728 (1.47) | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16w | rtx-4090 | 3.661 | torch-bf16 1.448 (2.53) | **torch-bf16-compile 1.354 (2.70)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16w | h100 | 1.054 | torch-bf16 0.7816 (1.35) | **torch-bf16-compile 0.5618 (1.88)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16w | rtx-5080 | 0.339 | torch-bf16 0.2551 (1.33) | **torch-bf16-compile 0.217 (1.56)** | torch-bf16-compile-cg 0.2295 (1.48) | torch-bf16-compile-ro 1.928 (0.18) | torch-bf16-cudnn 0.2515 (1.35)<br>torch-bf16-efficient 1.163 (0.29) | fa2-bf16 0.2529 (1.34) | flashinfer-bf16 0.3125 (1.09) |  | liger-bf16 0.2431 (1.39) |
| llm_dec_b8_l2048_bf16w | rtx-3090 | 0.8375 | torch-bf16 0.6158 (1.36) | **torch-bf16-compile 0.3557 (2.35)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16w | a100 | 0.3471 | torch-bf16 0.2831 (1.23) | **torch-bf16-compile 0.2285 (1.52)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16w | rtx-4090 | 0.4174 | torch-bf16 0.3724 (1.12) | **torch-bf16-compile 0.2364 (1.77)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16w | h100 | 0.1897 | torch-bf16 0.5525 (0.34) | **torch-bf16-compile 0.2566 (0.74)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16w | rtx-5080 | 0.9781 | torch-bf16 0.492 (1.99) | torch-bf16-compile 0.485 (2.02) | torch-bf16-compile-cg 0.4918 (1.99) | torch-bf16-compile-ro 7.687 (0.13) | **torch-bf16-cudnn 0.4726 (2.07)**<br>torch-bf16-efficient 4.057 (0.24) | fa2-bf16 0.4845 (2.02) | flashinfer-bf16 0.5405 (1.81) |  | liger-bf16 0.4883 (2.00) |
| llm_dec_b8_l8192_bf16w | rtx-3090 | 1.615 | torch-bf16 0.8316 (1.94) | **torch-bf16-compile 0.7404 (2.18)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16w | a100 | 0.7625 | **torch-bf16 0.4232 (1.80)** | torch-bf16-compile 0.5364 (1.42) | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16w | rtx-4090 | 1.051 | torch-bf16 0.5861 (1.79) | **torch-bf16-compile 0.4597 (2.29)** | not run | not run | not run | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16w | h100 | 0.4542 | torch-bf16 0.5171 (0.88) | **torch-bf16-compile 0.2903 (1.56)** | not run | not run | not run | not run | not run |  | not run |

### Conv / FIR / elementwise

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand CUDA |
|---|---|---:|---|---|---|---|---|---|
| conv_2048_5 | rtx-5080 | 0.06647 |  | torch-eager 0.1536 (0.43) | torch-compile 0.1536 (0.43) | **torch-compile-cg 0.1516 (0.44)** | torch-compile-ro 0.1536 (0.43) |  |
| conv_2048_5 | rtx-3090 | 0.1124 |  | torch-eager 0.1823 (0.62) | **torch-compile 0.1649 (0.68)** | not run | not run |  |
| conv_2048_5 | a100 | 0.07259 |  | **torch-eager 0.1372 (0.53)** | torch-compile 0.1382 (0.53) | not run | not run |  |
| conv_2048_5 | rtx-4090 | 0.04794 |  | **torch-eager 0.0932 (0.51)** | torch-compile 0.0942 (0.51) | not run | not run |  |
| conv_2048_5 | h100 | 0.04777 |  | torch-eager 0.0949 (0.50) | **torch-compile 0.0946 (0.50)** | not run | not run |  |
| convmc_8_64_56_3 | rtx-5080 | 0.0809 |  | **torch-eager 0.1198 (0.68)** | torch-compile 0.1628 (0.50) | torch-compile-cg 0.1577 (0.51) | torch-compile-ro 0.1577 (0.51) |  |
| convmc_8_64_56_3 | rtx-3090 | 0.1516 |  | **torch-eager 0.0809 (1.87)** | torch-compile 0.169 (0.90) | not run | not run |  |
| convmc_8_64_56_3 | a100 | 0.1958 |  | **torch-eager 0.087 (2.25)** | torch-compile 0.1823 (1.07) | not run | not run |  |
| convmc_8_64_56_3 | rtx-4090 | 0.09405 |  | **torch-eager 0.0563 (1.67)** | torch-compile 0.0922 (1.02) | not run | not run |  |
| convmc_8_64_56_3 | h100 | 0.1272 |  | **torch-eager 0.0724 (1.76)** | torch-compile 0.1731 (0.74) | not run | not run |  |
| ew_4096 | rtx-5080 | 0.08065 | chain-eager 0.5652 (0.14) | torch-eager 0.5734 (0.14) | torch-compile 0.2458 (0.33) | torch-compile-cg 0.2437 (0.33) | torch-compile-ro 0.2437 (0.33) | **chain-fused 0.2437 (0.33)** |
| ew_4096 | rtx-3090 | 0.1559 | chain-eager 0.5642 (0.28) | torch-eager 0.5642 (0.28) | torch-compile 0.2406 (0.65) | not run | not run | **chain-fused 0.2406 (0.65)** |
| ew_4096 | a100 | 0.1178 | chain-eager 0.3666 (0.32) | torch-eager 0.3584 (0.33) | **torch-compile 0.1556 (0.76)** | not run | not run | chain-fused 0.1608 (0.73) |
| ew_4096 | rtx-4090 | 0.07694 | chain-eager 0.514 (0.15) | torch-eager 0.5233 (0.15) | torch-compile 0.2263 (0.34) | not run | not run | **chain-fused 0.2243 (0.34)** |
| ew_4096 | h100 | 0.1144 | chain-eager 0.1995 (0.57) | torch-eager 0.1688 (0.68) | torch-compile 0.1331 (0.86) | not run | not run | **chain-fused 0.0828 (1.38)** |
| ewconvew_2048_5 | rtx-5080 | 0.07747 |  | torch-eager 0.2079 (0.37) | **torch-compile 0.1884 (0.41)** | torch-compile-cg 0.1884 (0.41) | torch-compile-ro 0.1905 (0.41) |  |
| ewconvew_2048_5 | rtx-3090 | 0.1385 |  | torch-eager 0.2918 (0.47) | **torch-compile 0.2673 (0.52)** | not run | not run |  |
| ewconvew_2048_5 | a100 | 0.08772 |  | torch-eager 0.1946 (0.45) | **torch-compile 0.1772 (0.50)** | not run | not run |  |
| ewconvew_2048_5 | rtx-4090 | 0.0616 |  | torch-eager 0.1454 (0.42) | **torch-compile 0.1249 (0.49)** | not run | not run |  |
| ewconvew_2048_5 | h100 | 0.06018 |  | torch-eager 0.1258 (0.48) | **torch-compile 0.1167 (0.52)** | not run | not run |  |
| fir_4194304_16 | rtx-5080 | 0.04431 |  | torch-eager 0.1044 (0.42) | torch-compile 0.1044 (0.42) | **torch-compile-cg 0.1024 (0.43)** | torch-compile-ro 0.1044 (0.42) |  |
| fir_4194304_16 | rtx-3090 | 0.07665 |  | torch-eager 0.1147 (0.67) | **torch-compile 0.1137 (0.67)** | not run | not run |  |
| fir_4194304_16 | a100 | 0.05038 |  | torch-eager 0.086 (0.59) | **torch-compile 0.085 (0.59)** | not run | not run |  |
| fir_4194304_16 | rtx-4090 | 0.03517 |  | torch-eager 0.0655 (0.54) | **torch-compile 0.0645 (0.55)** | not run | not run |  |
| fir_4194304_16 | h100 | 0.03974 |  | **torch-eager 0.0611 (0.65)** | torch-compile 0.0612 (0.65) | not run | not run |  |
| firew_rowsum_2048x2048_16 | rtx-5080 | 0.05244 |  | torch-eager 0.128 (0.41) | torch-compile 0.1106 (0.47) | torch-compile-cg 0.1106 (0.47) | **torch-compile-ro 0.1085 (0.48)** |  |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1039 |  | torch-eager 0.2212 (0.47) | **torch-compile 0.1403 (0.74)** | not run | not run |  |
| firew_rowsum_2048x2048_16 | a100 | 0.06194 |  | torch-eager 0.1311 (0.47) | **torch-compile 0.0963 (0.64)** | not run | not run |  |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.04107 |  | torch-eager 0.0932 (0.44) | **torch-compile 0.0676 (0.61)** | not run | not run |  |
| firew_rowsum_2048x2048_16 | h100 | 0.04602 |  | torch-eager 0.0876 (0.53) | **torch-compile 0.0664 (0.69)** | not run | not run |  |
| convmc_8_64_56_3_tf32 | rtx-5080 | 0.08029 |  | **torch-eager-tf32 0.0614 (1.31)** | torch-compile-tf32 0.0922 (0.87) | torch-compile-cg-tf32 0.0922 (0.87) | torch-compile-ro-tf32 0.0901 (0.89) |  |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1513 |  | **torch-eager-tf32 0.0809 (1.87)** | torch-compile-tf32 0.1372 (1.10) | not run | not run |  |
| convmc_8_64_56_3_tf32 | a100 | 0.1915 |  | **torch-eager-tf32 0.0471 (4.07)** | torch-compile-tf32 0.0829 (2.31) | not run | not run |  |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.1006 |  | **torch-eager-tf32 0.042 (2.40)** | torch-compile-tf32 0.0809 (1.24) | not run | not run |  |
| convmc_8_64_56_3_tf32 | h100 | 0.1232 |  | **torch-eager-tf32 0.039 (3.16)** | torch-compile-tf32 0.1167 (1.06) | not run | not run |  |

### Fan-in/out, loops, CPU+GPU

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead |
|---|---|---:|---|---|---|---|---|
| diamond_2048 | rtx-5080 | 1.441 | **chain-lt 1.378 (1.05)**<br>chain-streams 1.397 (1.03)<br>chain-eager 1.405 (1.03)<br>chain-cutlass 1.647 (0.88) | torch-eager 1.384 (1.04) | torch-compile 1.382 (1.04) | torch-compile-cg 1.382 (1.04) | torch-compile-ro 1.397 (1.03) |
| diamond_2048 | rtx-3090 | 2.45 | **chain-streams 2.357 (1.04)**<br>chain-lt 2.398 (1.02)<br>chain-eager 2.408 (1.02)<br>chain-cutlass 2.737 (0.90) | torch-eager 2.446 (1.00) | torch-compile 2.419 (1.01) | not run | not run |
| diamond_2048 | a100 | 3.246 | **chain-lt 2.949 (1.10)**<br>chain-streams 2.953 (1.10)<br>chain-eager 2.983 (1.09)<br>chain-cutlass 3.465 (0.94) | torch-eager 2.979 (1.09) | torch-compile 3.793 (0.86) | not run | not run |
| diamond_2048 | rtx-4090 | 0.8958 | **chain-cutlass 0.9247 (0.97)**<br>chain-lt 0.9483 (0.94)<br>chain-streams 0.9574 (0.94)<br>chain-eager 0.9667 (0.93) | torch-eager 0.9695 (0.92) | torch-compile 0.9492 (0.94) | not run | not run |
| diamond_2048 | h100 | 1.181 | **chain-lt 1.034 (1.14)**<br>chain-streams 1.042 (1.13)<br>chain-eager 1.045 (1.13)<br>chain-cutlass 1.267 (0.93) | torch-eager 1.042 (1.13) | torch-compile 1.035 (1.14) | not run | not run |
| fanout_2048 | rtx-5080 | 0.9824 | **chain-streams 0.9112 (1.08)**<br>chain-eager 0.9236 (1.06) | torch-eager 0.938 (1.05) | torch-compile 0.9236 (1.06) | torch-compile-cg 0.9257 (1.06) | torch-compile-ro 0.9257 (1.06) |
| fanout_2048 | rtx-3090 | 1.656 | **chain-streams 1.573 (1.05)**<br>chain-eager 1.607 (1.03) | torch-eager 1.646 (1.01) | torch-compile 1.645 (1.01) | not run | not run |
| fanout_2048 | a100 | 2.253 | **chain-streams 1.967 (1.15)**<br>chain-eager 1.987 (1.13) | torch-eager 1.996 (1.13) | torch-compile 2.046 (1.10) | not run | not run |
| fanout_2048 | rtx-4090 | 0.6075 | **chain-streams 0.638 (0.95)**<br>chain-eager 0.6441 (0.94) | torch-eager 0.638 (0.95) | torch-compile 0.6574 (0.92) | not run | not run |
| fanout_2048 | h100 | 0.7925 | **chain-streams 0.6865 (1.15)**<br>chain-eager 0.698 (1.14) | torch-eager 0.7024 (1.13) | torch-compile 0.6997 (1.13) | not run | not run |
| indep_2048 | rtx-5080 | 0.9995 | chain-lt 0.9216 (1.08)<br>chain-streams 0.938 (1.07)<br>chain-eager 0.9441 (1.06)<br>chain-cutlass 1.102 (0.91) | torch-eager 0.9298 (1.07) | **torch-compile 0.9175 (1.09)** | torch-compile-cg 0.9175 (1.09) | torch-compile-ro 0.9298 (1.07) |
| indep_2048 | rtx-3090 | 1.65 | chain-streams 1.577 (1.05)<br>chain-lt 1.618 (1.02)<br>chain-eager 1.63 (1.01)<br>chain-cutlass 1.83 (0.90) | torch-eager 1.624 (1.02) | **torch-compile 1.575 (1.05)** | not run | not run |
| indep_2048 | a100 | 2.246 | chain-lt 1.969 (1.14)<br>chain-streams 1.972 (1.14)<br>chain-eager 2.002 (1.12)<br>chain-cutlass 2.313 (0.97) | torch-eager 1.997 (1.12) | **torch-compile 1.968 (1.14)** | not run | not run |
| indep_2048 | rtx-4090 | 0.5937 | **chain-cutlass 0.6226 (0.95)**<br>chain-lt 0.64 (0.93)<br>chain-streams 0.6431 (0.92)<br>chain-eager 0.6533 (0.91) | torch-eager 0.6564 (0.90) | torch-compile 0.6398 (0.93) | not run | not run |
| indep_2048 | h100 | 0.8044 | **chain-lt 0.693 (1.16)**<br>chain-streams 0.7013 (1.15)<br>chain-eager 0.7046 (1.14)<br>chain-cutlass 0.8525 (0.94) | torch-eager 0.6994 (1.15) | torch-compile 0.6951 (1.16) | not run | not run |
| indep_512 | rtx-5080 | 0.03893 | chain-streams 0.0365 (1.07)<br>chain-lt 0.0451 (0.86)<br>chain-eager 0.0471 (0.83)<br>chain-cutlass 0.1044 (0.37) | torch-eager 0.0451 (0.86) | **torch-compile 0.0328 (1.19)** | torch-compile-cg 0.0328 (1.19) | torch-compile-ro 0.0451 (0.86) |
| indep_512 | rtx-3090 | 0.08637 | **chain-streams 0.0522 (1.65)**<br>chain-lt 0.0543 (1.59)<br>chain-eager 0.0573 (1.51)<br>chain-cutlass 0.1249 (0.69) | torch-eager 0.0584 (1.48) | torch-compile 0.1638 (0.53) | not run | not run |
| indep_512 | a100 | 0.07592 | **chain-streams 0.0573 (1.32)**<br>chain-lt 0.0655 (1.16)<br>chain-eager 0.0666 (1.14)<br>chain-cutlass 0.2079 (0.37) | torch-eager 0.0666 (1.14) | torch-compile 0.085 (0.89) | not run | not run |
| indep_512 | rtx-4090 | 0.04115 | **chain-streams 0.0379 (1.09)**<br>chain-lt 0.0379 (1.09)<br>chain-eager 0.0399 (1.03)<br>chain-cutlass 0.0911 (0.45) | torch-eager 0.0399 (1.03) | torch-compile 0.0399 (1.03) | not run | not run |
| indep_512 | h100 | 0.06053 | **chain-streams 0.0328 (1.85)**<br>chain-lt 0.0351 (1.72)<br>chain-eager 0.038 (1.59)<br>chain-cutlass 0.1199 (0.50) | torch-eager 0.0381 (1.59) | torch-compile 0.1355 (0.45) | not run | not run |
| loop_4096 | rtx-5080 | 28.25 | **chain-lt 28.55 (0.99)**<br>chain-cutlass 28.9 (0.98)<br>chain-eager 30.19 (0.94) | torch-eager 29.28 (0.96) | torch-compile 29.17 (0.97) | torch-compile-cg 29.3 (0.96) | torch-compile-ro 29.22 (0.97) |
| loop_4096 | rtx-3090 | 46.87 | **chain-lt 47.55 (0.99)**<br>chain-cutlass 49.15 (0.95)<br>chain-eager 49.98 (0.94) | torch-eager 48.94 (0.96) | torch-compile 49.16 (0.95) | not run | not run |
| loop_4096 | a100 | 62.97 | chain-eager 59.11 (1.07)<br>chain-lt 60.44 (1.04)<br>chain-cutlass 61.06 (1.03) | **torch-eager 59.03 (1.07)** | torch-compile 61.22 (1.03) | not run | not run |
| loop_4096 | rtx-4090 | 18.75 | **chain-cutlass 19.46 (0.96)**<br>chain-lt 19.58 (0.96)<br>chain-eager 20.61 (0.91) | torch-eager 20.03 (0.94) | torch-compile 19.69 (0.95) | not run | not run |
| loop_4096 | h100 | 23.81 | **chain-lt 21.26 (1.12)**<br>chain-eager 21.72 (1.10)<br>chain-cutlass 26.94 (0.88) | torch-eager 21.63 (1.10) | torch-compile 21.72 (1.10) | not run | not run |
| loopmm_1024_8 | rtx-5080 | 0.649 |  | torch-eager 0.6615 (0.98) | torch-compile 0.5898 (1.10) | **torch-compile-cg 0.5878 (1.10)** | torch-compile-ro 0.6246 (1.04) |
| loopmm_1024_8 | rtx-3090 | 1.109 |  | torch-eager 1.041 (1.07) | **torch-compile 0.9984 (1.11)** | not run | not run |
| loopmm_1024_8 | a100 | 1.435 |  | **torch-eager 1.044 (1.37)** | torch-compile 1.353 (1.06) | not run | not run |
| loopmm_1024_8 | rtx-4090 | 0.4281 |  | torch-eager 0.3645 (1.17) | **torch-compile 0.3625 (1.18)** | not run | not run |
| loopmm_1024_8 | h100 | 0.5319 |  | **torch-eager 0.4614 (1.15)** | torch-compile 0.4751 (1.12) | not run | not run |
| rt_mlp2_1024 | rtx-5080 | 2.557 |  | torch-eager 3.356 (0.76) | **torch-compile 2.527 (1.01)** | torch-compile-cg 2.531 (1.01) | torch-compile-ro 2.616 (0.98) |
| rt_mlp2_1024 | rtx-3090 | 4.303 |  | torch-eager 6.459 (0.67) | **torch-compile 4.195 (1.03)** | not run | not run |
| rt_mlp2_1024 | a100 | 6.834 |  | torch-eager 5.453 (1.25) | **torch-compile 4.774 (1.43)** | not run | not run |
| rt_mlp2_1024 | rtx-4090 | 1.687 |  | torch-eager 2.342 (0.72) | **torch-compile 1.492 (1.13)** | not run | not run |
| rt_mlp2_1024 | h100 | 2.146 |  | torch-eager 2.96 (0.73) | **torch-compile 2.043 (1.05)** | not run | not run |
| rt_mlp2_2048 | rtx-5080 | 16.19 |  | torch-eager 19.28 (0.84) | torch-compile 15.55 (1.04) | **torch-compile-cg 15.53 (1.04)** | torch-compile-ro 15.65 (1.03) |
| rt_mlp2_2048 | rtx-3090 | 26.37 |  | torch-eager 35.51 (0.74) | **torch-compile 26.7 (0.99)** | not run | not run |
| rt_mlp2_2048 | a100 | 37.17 |  | torch-eager 38.32 (0.97) | **torch-compile 31.95 (1.16)** | not run | not run |
| rt_mlp2_2048 | rtx-4090 | 9.677 |  | torch-eager 14.04 (0.69) | **torch-compile 10.92 (0.89)** | not run | not run |
| rt_mlp2_2048 | h100 | 12.05 |  | torch-eager 14.42 (0.84) | **torch-compile 11.44 (1.05)** | not run | not run |
| twin_2048 | rtx-5080 | 0.9325 |  | torch-eager 0.9421 (0.99) | **torch-compile 0.9175 (1.02)** | torch-compile-cg 0.9175 (1.02) | torch-compile-ro 0.9175 (1.02) |
| twin_2048 | rtx-3090 | 1.636 |  | torch-eager 1.567 (1.04) | **torch-compile 1.462 (1.12)** | not run | not run |
| twin_2048 | a100 | 2.021 |  | **torch-eager 1.964 (1.03)** | torch-compile 1.964 (1.03) | not run | not run |
| twin_2048 | rtx-4090 | 0.5949 |  | torch-eager 0.6482 (0.92) | **torch-compile 0.6175 (0.96)** | not run | not run |
| twin_2048 | h100 | 0.792 |  | torch-eager 0.6846 (1.16) | **torch-compile 0.6845 (1.16)** | not run | not run |

## Command

```
python3 benches/local/matrix.py \
  rtx-5080=.claude/raw-evidence/local-rtx-5080/model-scale/resA2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB1/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/onto-device/resE/results-rtx-5080.txt@8bba2a50,.claude/raw-evidence/local-rtx-5080/device-geometry/v3/model/results-rtx-5080.txt@5b6c0a29,.claude/raw-evidence/local-rtx-5080/large-gemm/resM/results-rtx-5080.txt@7a2b67bf,.claude/raw-evidence/local-rtx-5080/half-precision/run1/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/half-precision/run2/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/long-prefill/resL/results-rtx-5080.txt@c8b83028,.claude/raw-evidence/local-rtx-5080/long-prefill/suite/results-rtx-5080.txt@c03a52a3,.claude/raw-evidence/local-rtx-5080/competitors/resS/results-rtx-5080.txt@f7a74cdb,.claude/raw-evidence/local-rtx-5080/competitors/resM/results-rtx-5080.txt@f7a74cdb,.claude/raw-evidence/local-rtx-5080/competitors/resM2/results-rtx-5080.txt@f7a74cdb \
  rtx-3090=.claude/raw-evidence/cross-arch/60f0f8ce/rtx-3090/results-rtx-3090.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/rtx-3090/results-rtx-3090-pass2.txt@60f0f8ce \
  a100=.claude/raw-evidence/cross-arch/60f0f8ce/a100/results-a100.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/a100/results-a100-pass2.txt@60f0f8ce \
  rtx-4090=.claude/raw-evidence/cross-arch/60f0f8ce/rtx-4090/results-rtx-4090.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/rtx-4090/results-rtx-4090-pass2.txt@60f0f8ce \
  h100=.claude/raw-evidence/cross-arch/60f0f8ce/h100/results-h100.txt@60f0f8ce,.claude/raw-evidence/cross-arch/60f0f8ce/h100/results-h100-pass2.txt@60f0f8ce
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
