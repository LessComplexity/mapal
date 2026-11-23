# Mapal vs the fastest competitor at the same precision

Built 2026-11-23 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations) vs the torch bf16 legs (bf16 throughout). The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch's pick, `-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials' matmul / softmax / fused attention, autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. **Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang as servers. Every part ran the competitor legs (FlashAttention-3 on sm_90a only; `unavailable` = the part's run reported the package missing or its build capped).

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | f7a74cdb | competitors/resS@f7a74cdb: 43; competitors/resM@f7a74cdb: 43 |
| rtx-3090 | 19a8a44e | 19a8a44e/rtx-3090@19a8a44e: 86 |
| a100 | 19a8a44e | 19a8a44e/a100@19a8a44e: 86 |
| rtx-4090 | 19a8a44e | 19a8a44e/rtx-4090@19a8a44e: 86 |
| h100 | 19a8a44e | 19a8a44e/h100@19a8a44e: 86 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 1.156 (31) | 1.113 (31) | 1.229 (31) | 1.230 (31) | 1.430 (31) |
| MLP chains | 1.097 (6) | 1.142 (6) | 1.249 (6) | 1.236 (6) | 1.594 (6) |
| Attention / softmax / long rows | 0.841 (14) | 1.051 (14) | 1.318 (14) | 0.969 (14) | 1.096 (14) |
| LLM decode / prefill / block (f32, tf32, bf16w) | 1.077 (19) | 1.041 (19) | 1.149 (19) | 1.011 (19) | 0.991 (19) |
| Conv / FIR / elementwise | 0.525 (7) | 0.868 (7) | 0.953 (7) | 0.838 (7) | 0.894 (7) |
| Fan-in/out, loops, CPU+GPU | 1.061 (9) | 0.995 (9) | 1.206 (9) | 1.052 (9) | 1.207 (9) |
| all f32 rows | 0.905 (66) | 0.995 (66) | 1.080 (66) | 1.010 (66) | 1.073 (66) |
| all tf32 rows | 1.303 (13) | 1.212 (13) | 1.721 (13) | 1.359 (13) | 2.051 (13) |
| all bf16w rows | 1.582 (7) | 1.408 (7) | 1.633 (7) | 1.337 (7) | 1.318 (7) |
| **all rows** | 1.001 (86) | 1.054 (86) | 1.199 (86) | 1.081 (86) | 1.203 (86) |
| rows timed on every part (86) | 1.001 (86) | 1.054 (86) | 1.199 (86) | 1.081 (86) | 1.203 (86) |
| win / even / loss | 18/22/46 | 32/9/45 | 16/13/57 | 24/13/49 | 16/1/69 |

Geomean cells: value (rows). Every part times the same rows.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 1.04≈ 3.675/3.547 (torch-compile-ro) | **0.80** 5.512/6.855 (torch-compile-cg) | 1.08 7.882/7.295 (chain-lt) | 0.99≈ 2.593/2.614 (torch-compile-ro) | 1.12 2.98/2.66 (chain-lt) |
| mm_16x2048x7168 | f32 | 1.10 0.0879/0.0799 (cublas) | 1.03≈ 0.09165/0.0886 (cublas) | 1.01≈ 0.09215/0.0911 (cublas) ±6% | 1.12 0.105/0.09375 (cublas) | 1.85 0.09525/0.05155 (cublas) |
| mm_4096x4096x4096 | f32 | 1.04≈ 3.812/3.652 (cublas) | 1.18 9.525/8.091 (cublas) | 1.04≈ 7.68/7.375 (cublas) | 1.02≈ 3.503/3.425 (cublas) | 1.18 3.143/2.663 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.08 0.8088/0.7516 (lt) | 0.99≈ 1.227/1.235 (torch-compile-ro) | 0.99≈ 1.727/1.747 (torch-compile-ro) | 1.00≈ 0.6884/0.6861 (lt-fresh) | 1.03≈ 0.7008/0.6772 (lt) |
| pj_m128_n4096_k14336 | f32 | 1.13 0.4547/0.4014 (cublas-fresh) | 1.08 0.7108/0.6564 (torch-compile-cg) | 1.08 0.9177/0.8509 (cublas-fresh) | 1.08 0.3975/0.3666 (cublas-fresh) | 1.18 0.3695/0.3123 (lt) |
| pj_m128_n4096_k4096 | f32 | 1.08 0.1456/0.1352 (cublas-fresh) | 1.11 0.2442/0.2191 (torch-compile-cg) | 1.17 0.3046/0.2601 (cublas) | 1.38 0.1467/0.1065 (lt-fresh) | 1.45 0.1428/0.0982 (lt) |
| pj_m128_n6144_k4096 | f32 | 1.16 0.2095/0.1802 (cublas-fresh) | 1.09 0.3396/0.3103 (torch-eager) | 1.15 0.4298/0.3738 (lt) | 1.24 0.2068/0.1669 (cublas-fresh) | 1.32 0.1975/0.1494 (torch-compile-cg) |
| pj_m16_n18432_k7168 | f32 | 1.02≈ 0.6025/0.5878 (cublas-fresh) | 1.06 0.6458/0.6113 (torch-compile) | 1.00≈ 0.4434/0.4454 (torch-compile) | 1.05 0.6256/0.5949 (lt-fresh) | 1.06 0.2207/0.2088 (torch-compile) |
| pj_m16_n2048_k7168 | f32 | **0.94** 0.04612/0.0492 (cublas-fresh) | 1.40 0.1145/0.0819 (torch-compile-cg) | 1.19 0.1036/0.087 (cublas-fresh) | 2.01 0.06373/0.0317 (cublas-fresh) | 1.43 0.06898/0.0481 (lt-fresh) ±8% |
| pj_m16_n28672_k4096 | f32 | 1.03≈ 0.5395/0.5243 (lt-fresh) | 1.09 0.5905/0.5427 (torch-compile) | 1.03≈ 0.4179/0.4076 (torch-compile) | 1.06 0.5661/0.5356 (cublas-fresh) | 1.17 0.2053/0.1757 (torch-compile-cg) |
| pj_m16_n4096_k14336 | f32 | 0.97≈ 0.2667/0.2744 (torch-compile) | 1.11 0.306/0.2755 (torch-compile) | **0.94** 0.2347/0.2509 (lt) | 1.08 0.3183/0.2939 (cublas-fresh) | 1.13 0.1319/0.1164 (lt) |
| pj_m16_n4096_k4096 | f32 | **0.94** 0.05224/0.0553 (cublas-fresh) | 1.52 0.131/0.086 (torch-compile) | 1.12 0.0994/0.0891 (cublas-fresh) | 1.83 0.06558/0.0358 (lt-fresh) | 1.35 0.06435/0.0477 (lt) |
| pj_m16_n6144_k4096 | f32 | 1.14 0.1009/0.0881 (lt-fresh) | 1.33 0.1629/0.1229 (torch-compile-cg) | 1.12 0.1293/0.1157 (torch-compile-cg) | 1.27 0.1276/0.1004 (cublas-fresh) | 1.40 0.08028/0.0572 (torch-compile) |
| pj_m1_n18432_k7168 | f32 | 0.99≈ 0.6051/0.6083 (torch-compile) | 1.03≈ 0.6233/0.6072 (cublas) | 1.10 0.4309/0.3912 (lt-fresh) | 1.19 0.6203/0.5222 (lt-fresh) | 1.14 0.2183/0.1916 (lt-fresh) |
| pj_m1_n2048_k7168 | f32 | 1.26 0.03351/0.0266 (cublas-fresh) | 1.30 0.09693/0.0748 (lt-fresh) | 2.00 0.08791/0.044 (cublas-fresh) | 1.42 0.03918/0.0276 (lt-fresh) ±7% | 2.06 0.05008/0.0243 (cublas-fresh) |
| pj_m1_n28672_k4096 | f32 | 1.12 0.5624/0.5038 (lt-fresh) | 1.05 0.5658/0.5366 (torch-compile-ro) | 1.09 0.3794/0.3482 (torch-compile-cg) | 1.20 0.5613/0.468 (lt-fresh) | 1.13 0.1913/0.169 (lt-fresh) |
| pj_m1_n4096_k14336 | f32 | 1.10 0.2547/0.2314 (cublas-fresh) | 1.09 0.3006/0.2755 (torch-compile-ro) | 1.13 0.2009/0.1772 (lt-fresh) | 1.52 0.3031/0.1997 (lt-fresh) | 1.36 0.115/0.0847 (cublas-fresh) |
| pj_m1_n4096_k4096 | f32 | 1.59 0.03263/0.0205 (lt-fresh) ±10% | 1.29 0.1105/0.086 (lt) | 1.69 0.08984/0.0532 (cublas-fresh) | 2.12 0.04339/0.0205 (lt-fresh) | 2.22 0.05306/0.0239 (lt-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.80 0.09199/0.0512 (cublas-fresh) | 1.22 0.1485/0.1219 (torch-compile-cg) | 1.48 0.1154/0.0778 (lt-fresh) | 1.59 0.08952/0.0564 (lt-fresh) | 1.72 0.06406/0.0372 (cublas-fresh) |
| pj_m2048_n28672_k4096 | f32 | 1.03≈ 12.63/12.21 (cublas-fresh) | **0.92** 20.74/22.48 (torch-compile-ro) | 1.03≈ 26.87/26.02 (torch-compile) | **0.86** 9.519/11.06 (lt) | 1.10 10.12/9.235 (torch-compile) |
| pj_m2048_n4096_k14336 | f32 | 0.99≈ 6.427/6.498 (torch-eager) | **0.94** 11.05/11.75 (lt) | 1.03≈ 13.34/12.9 (torch-eager) | **0.77** 4.41/5.721 (torch-compile-cg) | 1.14 5.268/4.623 (cublas) |
| pj_m2048_n4096_k4096 | f32 | 1.04≈ 1.837/1.772 (torch-compile-cg) | **0.92** 3.125/3.393 (lt) | 1.04≈ 3.868/3.704 (torch-compile-cg) | 1.07 1.289/1.2 (torch-compile) | 1.14 1.514/1.328 (cublas-fresh) |
| pj_m2048_n6144_k4096 | f32 | 1.02≈ 2.648/2.602 (torch-compile-cg) | **0.93** 4.56/4.906 (torch-eager) | 1.00≈ 5.806/5.784 (torch-compile-cg) | 1.06 1.93/1.827 (torch-compile-ro) | 1.12 2.21/1.977 (torch-eager) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.18 0.6985/0.5898 (cublas-tf32-fresh) | 1.23 1.044/0.8499 (torch-compile-ro-tf32) | 1.29 0.5405/0.4178 (triton-tf32) | 1.01≈ 0.5646/0.5591 (lt-tf32-fresh) | 1.32 0.2497/0.1889 (cublas-tf32) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.48 0.4568/0.3092 (cublas-tf32-fresh) | 1.16 0.7027/0.6042 (torch-compile-tf32) | 2.03 0.4688/0.2314 (cublas-tf32-fresh) | 1.26 0.3781/0.3 (lt-tf32-fresh) | 2.22 0.2397/0.1079 (lt-tf32) |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.66 0.1355/0.0819 (lt-tf32-fresh) | 1.30 0.2123/0.1638 (torch-compile-cg-tf32) | 1.80 0.1495/0.0829 (cublas-tf32-fresh) | 1.55 0.1051/0.0676 (cublas-tf32-fresh) | 1.97 0.08655/0.044 (cublas-tf32-fresh) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.90 0.2368/0.1249 (lt-tf32-fresh) | 1.68 0.4119/0.2458 (lt-tf32) | 1.61 0.1604/0.0993 (triton-tf32) | 1.55 0.1852/0.1198 (lt-tf32-fresh) | 2.86 0.1496/0.0524 (lt-tf32-fresh) |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.12 9.038/8.062 (lt-tf32-fresh) | 1.01≈ 14.47/14.35 (torch-compile-tf32) | 1.57 5.908/3.773 (torch-eager-tf32) | 1.37 8.124/5.936 (lt-tf32-fresh) | 1.58 2.031/1.282 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.15 4.898/4.244 (lt-tf32-fresh) | 1.06 8.154/7.68 (torch-compile-cg-tf32) | 1.33 2.711/2.044 (torch-compile-ro-tf32) | 1.10 3.311/3.003 (torch-compile-ro-tf32) | 1.70 0.9886/0.5811 (torch-compile-cg-tf32) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.22 1.416/1.161 (lt-tf32) | 0.96≈ 2.345/2.442 (torch-compile-ro-tf32) | 1.37 0.7999/0.5827 (lt-tf32-fresh) | 1.17 0.9701/0.8294 (lt-tf32) | 1.76 0.3094/0.1757 (lt-tf32-fresh) |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.15 2.017/1.748 (cublas-tf32-fresh) | 1.04≈ 3.325/3.199 (torch-eager-tf32) | 1.55 1.248/0.8069 (torch-compile-cg-tf32) | 1.23 1.431/1.162 (torch-compile-ro-tf32) | 1.74 0.4528/0.2596 (lt-tf32-fresh) |
| geomean f32 | | 1.100 (23) | 1.096 (23) | 1.133 (23) | 1.217 (23) | 1.308 (23) |
| geomean tf32 | | 1.333 (8) | 1.162 (8) | 1.552 (8) | 1.267 (8) | 1.849 (8) |
| **geomean** | | 1.156 (31) | 1.113 (31) | 1.229 (31) | 1.230 (31) | 1.430 (31) |
| win / even / loss | | 2/10/19 | 5/6/20 | 1/9/21 | 2/4/25 | 0/1/30 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.10 1.943/1.765 (chain-lt2) | **0.89** 3.203/3.581 (chain-lt2) | 1.17 4.399/3.76 (chain-streams) | 0.97≈ 1.308/1.343 (torch-compile-ro) | 1.11 1.496/1.353 (chain-lt2) |
| mlp_2048 | f32 | 1.03≈ 0.9414/0.9114 (chain-lt) | 0.99≈ 1.607/1.628 (torch-compile-ro) | 1.13 2.224/1.964 (chain-lt) | 0.98≈ 0.641/0.6543 (torch-compile-cg) | 1.13 0.7728/0.6844 (chain-lt) |
| mlp_512 | f32 | 1.16 0.038/0.0328 (torch-compile) | 1.71 0.07699/0.0451 (triton) ±9% | 1.26 0.07869/0.0625 (chain-lt) | 2.16 0.06188/0.0287 (torch-compile-cg) | 1.86 0.05912/0.0317 (torch-compile-cg) |
| smlp_4096_256_1024 | f32 | 1.10 0.1645/0.1495 (torch-compile) | 1.23 0.2733/0.2222 (torch-compile-cg) | 1.18 0.3442/0.2918 (chain-eager) | 1.11 0.1205/0.1085 (triton) | 1.17 0.1306/0.1119 (chain-lt) |
| mlp2_2048_tf32 | tf32 | 1.10 1.668/1.514 (torch-compile-ro-tf32) | 1.04≈ 2.758/2.648 (torch-compile-ro-tf32) | 1.34 0.9324/0.6973 (torch-compile-cg-tf32) | 1.27 1.033/0.8141 (torch-compile-cg-tf32) | 2.47 0.5295/0.2145 (torch-compile-cg-tf32) |
| mlp_2048_tf32 | tf32 | 1.09 0.8298/0.7598 (torch-compile-ro-tf32) | 1.15 1.359/1.184 (torch-compile-ro-tf32) | 1.45 0.5078/0.3512 (torch-compile-cg-tf32) | 1.23 0.5029/0.4096 (torch-compile-cg-tf32) | 2.44 0.2701/0.1107 (torch-compile-cg-tf32) |
| geomean f32 | | 1.097 (4) | 1.167 (4) | 1.184 (4) | 1.229 (4) | 1.284 (4) |
| geomean tf32 | | 1.097 (2) | 1.094 (2) | 1.390 (2) | 1.248 (2) | 2.454 (2) |
| **geomean** | | 1.097 (6) | 1.142 (6) | 1.249 (6) | 1.236 (6) | 1.594 (6) |
| win / even / loss | | 0/1/5 | 1/2/3 | 0/0/6 | 0/2/4 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.36 0.03914/0.0287 (torch-compile-cg) | 2.18 0.0982/0.0451 (torch-compile-ro) | 1.17 0.06356/0.0543 (torch-sdpa-compile-cg) | 2.27 0.0603/0.0266 (torch-sdpa-compile-cg) | 2.10 0.0562/0.0268 (torch-compile-ro) |
| attn_2048_128 | f32 | 1.25 0.1177/0.0942 (torch-sdpa-compile-cg) | 1.43 0.2664/0.1864 (torch-sdpa-compile-cg) | 1.58 0.2613/0.1659 (torch-compile) | 1.21 0.1019/0.084 (torch-sdpa-compile-cg) | 1.31 0.1035/0.0789 (torch-compile-ro) |
| attn_2048_64 | f32 | 1.14 0.07953/0.0696 (torch-compile-ro) | 1.24 0.1655/0.1331 (torch-compile-ro) | 1.45 0.1573/0.1085 (torch-compile) | 1.18 0.0759/0.0645 (torch-compile-ro) | 1.45 0.08729/0.0603 (torch-compile-ro) |
| causal_1024_256 | f32 | **0.40** 15.23/38.23 (torch-compile-cg) | **0.14** 24.55/175.2 (torch-compile-cg) | **0.50** 36.83/74.11 (torch-compile-cg) | **0.10** 18.6/178.1 (torch-compile-cg) | **0.18** 21.44/120 (torch-compile-cg) |
| decode_1024_256 | f32 | **0.54** 8.635/16.04 (chain-cache) | **0.76** 25.58/33.6 (chain-cache) ±7% | **0.93** 27.32/29.32 (chain-cache) | **0.82** 25.78/31.47 (chain-cache) | **0.82** 18.61/22.59 (chain-cache) |
| gen_1024_512_256 | f32 | **0.37** 7.685/20.84 (torch-compile-cg) | **0.14** 12.58/88.2 (torch-compile-cg) | **0.49** 18.57/37.71 (torch-compile) | **0.11** 9.474/88.5 (torch-compile-ro) | **0.18** 10.83/61.59 (torch-compile-cg) |
| lsm_1_131072 | f32 | **0.70** 0.01582/0.0225 (torch-compile-cg) ±8% | 12.00 0.3071/0.0256 (torch-compile-cg) | 13.84 0.3543/0.0256 (torch-compile-cg) | 9.92 0.2342/0.0236 (torch-compile-cg) | 13.72 0.3114/0.0227 (torch-compile-cg) |
| lsm_32_8192 | f32 | 1.51 0.01538/0.0102 (torch-compile) ±23% | 3.53 0.03987/0.0113 (torch-compile-ro) | 2.48 0.03052/0.0123 (torch-sdpa-compile-cg) | 4.87 0.03994/0.0082 (torch-compile-cg) | 2.84 0.03233/0.0114 (torch-compile) |
| lsm_4_32768 | f32 | 1.16 0.0166/0.0143 (torch-compile) | 4.98 0.08167/0.0164 (torch-sdpa-compile-cg) | 3.81 0.0743/0.0195 (torch-sdpa-compile-cg) | 4.94 0.06574/0.0133 (torch-compile-cg) | 2.83 0.06055/0.0214 (torch-compile-ro) |
| prefill_1024_512_256 | f32 | **0.40** 7.743/19.26 (torch-compile-cg) | **0.15** 13.05/87.9 (torch-compile-cg) | **0.50** 18.68/37.47 (torch-compile-cg) | **0.10** 9.618/95.96 (torch-compile-cg) | **0.18** 11.12/61.68 (torch-compile-ro) |
| rowsum_2048 | f32 | 1.00≈ 0.4593/0.4608 (chain-eager) | 1.12 0.8423/0.7537 (torch-compile-ro) | 1.04≈ 1.036/0.9943 (torch-compile-ro) | 0.99≈ 0.328/0.3328 (chain-eager) | 1.18 0.4102/0.3467 (torch-compile-cg) |
| rowsum_4096 | f32 | 1.07 3.733/3.486 (torch-compile) | **0.92** 6.174/6.742 (torch-eager) | 1.06 7.75/7.337 (torch-compile-ro) | 1.01≈ 2.61/2.596 (torch-eager) | 1.12 2.986/2.661 (torch-compile-ro) |
| softmax_rows_2048 | f32 | 1.01≈ 0.4727/0.469 (torch-compile) | 1.17 0.8804/0.7516 (torch-compile-cg) | 1.03≈ 1.053/1.019 (triton) | 1.01≈ 0.3377/0.3359 (torch-compile-cg) | 1.19 0.4216/0.3549 (torch-compile-cg) |
| softmax_rows_4096 | f32 | 1.08 3.803/3.529 (torch-compile-ro) | **0.91** 6.337/6.945 (torch-compile-ro) | 1.06 7.818/7.375 (torch-compile-ro) | **0.91** 2.66/2.912 (torch-compile-cg) | 1.12 3.025/2.702 (torch-compile-cg) |
| **geomean** | | 0.841 (14) | 1.051 (14) | 1.318 (14) | 0.969 (14) | 1.096 (14) |
| win / even / loss | | 5/2/7 | 6/0/8 | 4/2/8 | 5/3/6 | 4/0/10 |

## LLM decode / prefill / block (f32, tf32, bf16w)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 1.01≈ 1.071/1.066 (torch-compile) | **0.90** 1.131/1.251 (torch-compile) | 1.17 0.9575/0.8206 (torch-compile) | 1.39 1.486/1.069 (torch-compile) | 1.20 0.5697/0.4745 (torch-compile-cg) |
| llm_blkpre_t2048 | f32 | 1.02≈ 27.28/26.67 (liger) | 1.23 61.08/49.54 (liger) | 1.02≈ 55.88/54.57 (torch-compile-cg) | **0.87** 20.1/23.03 (liger) | 1.16 21.38/18.46 (liger) |
| llm_dec_b1_l2048 | f32 | **0.95** 0.24/0.2532 (torch-compile-ro) | **0.76** 0.276/0.3629 (torch-compile) | **0.86** 0.2155/0.2501 (torch-compile) | **0.76** 0.2374/0.3105 (torch-compile) | **0.57** 0.115/0.2003 (torch-compile) |
| llm_dec_b1_l8192 | f32 | 0.97≈ 0.3324/0.342 (torch-compile) | **0.77** 0.379/0.4934 (torch-compile) | **0.70** 0.2774/0.3949 (torch-compile) | **0.89** 0.3319/0.372 (torch-compile) | **0.51** 0.1565/0.3047 (torch-compile) |
| llm_dec_b32_l2048 | f32 | **0.46** 1.033/2.256 (torch-compile) | **0.41** 1.218/2.943 (torch-compile-ro) | **0.47** 0.8605/1.832 (torch-compile-cg) | **0.46** 1.062/2.293 (torch-compile) | **0.41** 0.4756/1.149 (torch-compile) |
| llm_dec_b32_l8192 | f32 | **0.37** 3.506/9.484 (torch-compile) | **0.38** 4.27/11.18 (torch-compile-ro) | **0.39** 2.535/6.499 (torch-compile-cg) | **0.39** 3.502/8.986 (torch-compile) | **0.27** 1.052/3.878 (torch-compile) |
| llm_dec_b8_l2048 | f32 | **0.66** 0.4385/0.6685 (torch-compile) | 1.07 1.007/0.9416 (torch-compile) | **0.64** 0.3931/0.6163 (torch-compile) | **0.63** 0.4762/0.7501 (torch-compile) | **0.51** 0.2096/0.4127 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.43** 1.067/2.468 (torch-compile-cg) | **0.60** 1.787/2.957 (torch-compile) | **0.45** 0.8098/1.806 (torch-compile) | **0.43** 1.063/2.444 (torch-compile) | **0.43** 0.4752/1.096 (torch-compile) |
| llm_pre_t2048 | f32 | 1.19 8.56/7.2 (liger) | 1.08 14.34/13.27 (torch-sdpa-efficient) | 1.27 15.13/11.96 (liger) | 1.14 6.499/5.689 (liger) | 1.36 6.121/4.512 (liger) |
| llm_pre_t8192 | f32 | 1.53 80.68/52.83 (liger) | 1.58 153.7/97.29 (liger) | 2.17 132.3/61.05 (liger) | 1.61 64.53/40.13 (liger) | 1.98 51.58/25.99 (liger) |
| llm_blkpre_t2048_tf32 | tf32 | 1.21 21.83/18.08 (triton-tf32) | 1.14 35.17/30.99 (triton-tf32) | 1.65 16.36/9.895 (triton-tf32) | 1.27 16.52/13.06 (triton-tf32) | 1.92 6.678/3.47 (torch-compile-cg-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.64 7.455/4.537 (triton-tf32) | 1.52 12.62/8.286 (triton-tf32) | 2.62 7.426/2.832 (triton-tf32) | 1.69 5.738/3.398 (triton-tf32) | 2.35 3.217/1.372 (triton-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.15 0.636/0.553 (torch-bf16-compile) | 1.01≈ 0.7061/0.6983 (torch-bf16-compile) | 1.73 0.7864/0.4551 (torch-bf16-compile-cg) | **0.94** 0.5813/0.618 (torch-bf16-compile) | 1.22 0.3538/0.2899 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16w | bf16w | 1.02≈ 0.1501/0.1471 (torch-bf16-compile) | **0.72** 0.1846/0.2548 (torch-bf16-compile) | 1.07 0.1693/0.1576 (torch-bf16-compile) | **0.64** 0.1523/0.2379 (torch-bf16-compile-cg) | **0.63** 0.09451/0.1492 (torch-bf16-compile-cg) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.28 0.2405/0.1883 (liger-bf16) | **0.94** 0.2872/0.3039 (torch-bf16-compile) | 1.17 0.2314/0.198 (torch-bf16-compile) | **0.93** 0.2488/0.2688 (torch-bf16-compile-cg) | **0.87** 0.1359/0.1558 (torch-bf16-compile-cg) |
| llm_dec_b32_l2048_bf16w | bf16w | 2.10 0.9551/0.4548 (torch-bf16-compile) | 1.44 1.192/0.8302 (torch-bf16-compile-cg) | 2.03 0.866/0.4262 (torch-bf16-cudnn) | 1.97 0.9615/0.4882 (torch-bf16-compile) | 2.11 0.4895/0.2324 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16w | bf16w | 2.43 3.426/1.407 (flashinfer-bf16) | 2.39 4.256/1.778 (torch-bf16-cudnn) | 2.61 2.543/0.975 (torch-bf16-cudnn) | 2.44 3.404/1.398 (torch-bf16-compile-cg) | 2.16 1.065/0.494 (torch-bf16-compile-ro) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.56 0.339/0.217 (torch-bf16-compile) | 2.32 0.8274/0.356 (torch-bf16-compile) | 1.49 0.3439/0.2302 (torch-bf16-compile) | 1.43 0.3974/0.2776 (torch-bf16-compile) | 1.15 0.1901/0.166 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16w | bf16w | 2.07 0.9781/0.4726 (torch-bf16-cudnn) | 1.99 1.607/0.8092 (liger-bf16) | 1.81 0.7614/0.4216 (torch-bf16) | 1.99 0.9876/0.4954 (torch-bf16-compile) | 1.96 0.4534/0.2307 (torch-bf16-compile-cg) |
| geomean f32 | | 0.779 (10) | 0.804 (10) | 0.797 (10) | 0.773 (10) | 0.698 (10) |
| geomean tf32 | | 1.408 (2) | 1.315 (2) | 2.082 (2) | 1.462 (2) | 2.125 (2) |
| geomean bf16w | | 1.582 (7) | 1.408 (7) | 1.633 (7) | 1.337 (7) | 1.318 (7) |
| **geomean** | | 1.077 (19) | 1.041 (19) | 1.149 (19) | 1.011 (19) | 0.991 (19) |
| win / even / loss | | 5/4/10 | 8/1/10 | 6/1/12 | 10/0/9 | 8/0/11 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.44** 0.06647/0.1516 (torch-compile-cg) | **0.68** 0.1094/0.1608 (torch-compile) | **0.53** 0.07205/0.1372 (torch-compile-cg) | **0.62** 0.05751/0.0932 (torch-compile-cg) | **0.48** 0.0451/0.0943 (torch-eager) |
| convmc_8_64_56_3 | f32 | **0.68** 0.0809/0.1198 (torch-eager) | 1.91 0.147/0.0768 (torch-eager) | 2.22 0.193/0.087 (torch-eager) | 1.82 0.104/0.0573 (torch-eager) | 1.63 0.1166/0.0717 (torch-eager) |
| ew_4096 | f32 | **0.33** 0.08065/0.2437 (chain-fused) ±5% | **0.66** 0.1594/0.2406 (torch-compile-cg) | **0.76** 0.1171/0.1536 (torch-compile) | **0.45** 0.09937/0.2202 (chain-fused) | 1.17 0.0856/0.0731 (torch-compile) |
| ewconvew_2048_5 | f32 | **0.41** 0.07747/0.1884 (torch-compile) | **0.50** 0.1301/0.2601 (torch-compile-cg) | **0.51** 0.088/0.1731 (torch-compile-cg) | **0.52** 0.06835/0.1311 (torch-compile-cg) | **0.51** 0.05784/0.1142 (torch-compile-cg) |
| fir_4194304_16 | f32 | **0.43** 0.04431/0.1024 (torch-compile-cg) | **0.66** 0.0738/0.1126 (torch-compile-cg) | **0.59** 0.04992/0.085 (torch-compile-cg) | **0.64** 0.04025/0.0625 (torch-compile-ro) | **0.54** 0.03304/0.0607 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.48** 0.05244/0.1085 (torch-compile-ro) | **0.75** 0.1015/0.1362 (torch-compile-cg) | **0.65** 0.06162/0.0942 (torch-compile-ro) | **0.70** 0.04683/0.0666 (torch-compile-cg) | **0.59** 0.03877/0.0653 (torch-compile-cg) ±7% |
| convmc_8_64_56_3_tf32 | tf32 | 1.31 0.08029/0.0614 (torch-eager-tf32) | 1.77 0.1485/0.084 (torch-eager-tf32) | 4.13 0.1943/0.0471 (torch-eager-tf32) | 2.43 0.1045/0.043 (torch-eager-tf32) | 3.07 0.1164/0.0379 (torch-eager-tf32) |
| geomean f32 | | 0.451 (6) | 0.771 (6) | 0.747 (6) | 0.702 (6) | 0.728 (6) |
| geomean tf32 | | 1.308 (1) | 1.768 (1) | 4.126 (1) | 2.431 (1) | 3.071 (1) |
| **geomean** | | 0.525 (7) | 0.868 (7) | 0.953 (7) | 0.838 (7) | 0.894 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 5/0/2 | 5/0/2 | 4/0/3 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 1.05≈ 1.441/1.378 (chain-lt) | **0.87** 2.415/2.79 (chain-streams) | 1.10 3.247/2.95 (chain-lt) | 1.01≈ 0.9712/0.9626 (torch-compile-cg) | 1.13 1.166/1.029 (chain-lt) |
| fanout_2048 | f32 | 1.08 0.9824/0.9112 (chain-streams) | **0.95** 1.633/1.724 (torch-compile) | 1.15 2.252/1.966 (chain-streams) | 1.00≈ 0.6574/0.6573 (chain-streams) | 1.14 0.782/0.686 (chain-streams) |
| indep_2048 | f32 | 1.09 0.9995/0.9175 (torch-compile) | **0.88** 1.63/1.848 (torch-compile-ro) | 1.14 2.245/1.966 (torch-compile-cg) | 1.00≈ 0.6509/0.6492 (torch-compile-cg) | 1.15 0.7914/0.6886 (chain-lt) |
| indep_512 | f32 | 1.19 0.03893/0.0328 (torch-compile) | 1.82 0.08198/0.0451 (torch-compile-cg) | 1.33 0.07609/0.0573 (chain-streams) | 2.12 0.06089/0.0287 (torch-compile) | 2.10 0.05957/0.0284 (torch-compile) ±6% |
| loop_4096 | f32 | 0.99≈ 28.25/28.55 (chain-lt) | **0.92** 48.69/52.87 (chain-lt) | 1.07 62.97/58.98 (torch-compile-ro) | **0.76** 18.68/24.52 (torch-compile-ro) | 1.12 23.65/21.13 (chain-lt) |
| loopmm_1024_8 | f32 | 1.10 0.649/0.5878 (torch-compile-cg) | 1.16 1.073/0.9267 (torch-compile-ro) | 1.37 1.434/1.044 (torch-eager) | 1.21 0.4325/0.3584 (torch-compile-cg) | 1.16 0.5228/0.4513 (torch-compile-cg) |
| rt_mlp2_1024 | f32 | 1.01≈ 2.557/2.527 (torch-compile) | **0.85** 4.204/4.935 (torch-compile-cg) | 1.59 6.836/4.293 (torch-compile-cg) | 1.07 1.703/1.588 (torch-compile-cg) | 1.09 2.143/1.962 (torch-compile-ro) |
| rt_mlp2_2048 | f32 | 1.04≈ 16.19/15.53 (torch-compile-cg) | **0.85** 26.77/31.59 (torch-compile) | 1.17 37.18/31.75 (torch-compile-cg) | **0.78** 10.37/13.35 (torch-compile-ro) | 1.08 12.05/11.18 (torch-compile) |
| twin_2048 | f32 | 1.02≈ 0.9325/0.9175 (torch-compile) | **0.94** 1.605/1.7 (torch-compile) | 1.03≈ 2.022/1.962 (torch-compile-cg) | 0.96≈ 0.648/0.6758 (torch-compile-cg) | 1.14 0.7818/0.6844 (torch-compile-ro) |
| **geomean** | | 1.061 (9) | 0.995 (9) | 1.206 (9) | 1.052 (9) | 1.207 (9) |
| win / even / loss | | 0/5/4 | 7/0/2 | 0/1/8 | 2/4/3 | 0/0/9 |

## bf16 real-world reference

The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3174 (triton-bf16) · f32 2.55 | 0.4659 (bf16-compile-ro) · f32 2.63 | 0.2181 (triton-bf16) · f32 7.92 | 0.3082 (bf16-compile) · f32 2.23 | 0.0931 (triton-bf16) · f32 7.53 |
| pj_m128_n4096_k14336 | 0.172 (bf16) · f32 2.64 | 0.2703 (bf16-compile-cg) · f32 2.63 | 0.1229 (bf16-compile-cg) · f32 7.47 | 0.1833 (bf16-compile-cg) · f32 2.17 | 0.0586 (bf16-compile) · f32 6.30 |
| pj_m128_n4096_k4096 | 0.0532 (bf16) · f32 2.74 | 0.0829 (bf16-compile-ro) · f32 2.95 | 0.0522 (triton-bf16) · f32 5.83 | 0.0584 (triton-bf16) · f32 2.51 | 0.0239 (bf16) · f32 5.97 |
| pj_m128_n6144_k4096 | 0.0881 (bf16) · f32 2.38 | 0.1239 (bf16-compile-ro) · f32 2.74 | 0.0625 (triton-bf16) · f32 6.88 | 0.0799 (triton-bf16) · f32 2.59 | 0.0324 (bf16-compile) · f32 6.10 |
| pj_m16_n18432_k7168 | 0.3092 (bf16) · f32 1.95 | 0.3072 (bf16-compile) · f32 2.10 | 0.212 (triton-bf16) · f32 2.09 | 0.3308 (bf16-compile) · f32 1.89 | 0.1015 (triton-bf16) · f32 2.17 |
| pj_m16_n2048_k7168 | 0.0389 (bf16-compile) · f32 1.19 | 0.0471 (bf16-compile) · f32 2.43 | 0.043 (bf16-compile-cg) · f32 2.41 | 0.0471 (bf16-compile) · f32 1.35 | 0.022 (bf16) · f32 3.14 |
| pj_m16_n28672_k4096 | 0.2744 (bf16-compile) · f32 1.97 | 0.2734 (bf16-compile) · f32 2.16 | 0.1884 (bf16-compile) · f32 2.22 | 0.3 (bf16-compile) · f32 1.89 | 0.0905 (bf16-compile) · f32 2.27 |
| pj_m16_n4096_k14336 | 0.1393 (bf16-compile) · f32 1.91 | 0.1423 (bf16-compile-cg) · f32 2.15 | 0.1096 (bf16-compile-cg) · f32 2.14 | 0.171 (bf16-compile) · f32 1.86 | 0.0559 (bf16) · f32 2.36 |
| pj_m16_n4096_k4096 | 0.043 (bf16-compile-cg) · f32 1.21 | 0.0471 (bf16-compile) · f32 2.78 | 0.043 (bf16-compile-cg) · f32 2.31 | 0.0532 (bf16-compile-cg) · f32 1.23 | 0.0237 (bf16) · f32 2.72 |
| pj_m16_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.59 | 0.0655 (bf16-compile-cg) · f32 2.49 | 0.0594 (bf16-compile-cg) · f32 2.18 | 0.0768 (bf16-compile-cg) · f32 1.66 | 0.029 (bf16-compile) · f32 2.77 |
| pj_m1_n18432_k7168 | 0.3072 (bf16-compile-ro) · f32 1.97 | 0.3185 (bf16-compile-cg) · f32 1.96 | 0.214 (bf16-compile) · f32 2.01 | 0.2929 (bf16-compile-cg) · f32 2.12 | 0.1026 (bf16-compile-ro) · f32 2.13 |
| pj_m1_n2048_k7168 | 0.043 (bf16-compile) · f32 0.78 | 0.0451 (bf16-compile) · f32 2.15 | 0.0358 (bf16-compile-cg) · f32 2.46 | 0.0481 (bf16-compile-cg) · f32 0.81 | 0.022 (bf16) · f32 2.28 |
| pj_m1_n28672_k4096 | 0.2744 (bf16-compile) · f32 2.05 | 0.2796 (bf16-compile) · f32 2.02 | 0.1833 (bf16-compile) · f32 2.07 | 0.2632 (bf16-compile-cg) · f32 2.13 | 0.0919 (bf16-compile) · f32 2.08 |
| pj_m1_n4096_k14336 | 0.1413 (bf16) · f32 1.80 | 0.1465 (bf16) · f32 2.05 | 0.0993 (bf16-compile-cg) · f32 2.02 | 0.1475 (bf16-compile-cg) · f32 2.05 | 0.0547 (bf16-compile-cg) · f32 2.10 |
| pj_m1_n4096_k4096 | 0.0492 (bf16-compile) · f32 0.66 | 0.0492 (bf16-compile-ro) · f32 2.25 | 0.0399 (bf16-compile-cg) · f32 2.25 | 0.0543 (bf16-compile-cg) · f32 0.80 | 0.0224 (bf16) · f32 2.37 |
| pj_m1_n6144_k4096 | 0.0635 (bf16-compile-cg) · f32 1.45 | 0.0686 (bf16-compile) · f32 2.16 | 0.0532 (bf16-compile-cg) · f32 2.17 | 0.0768 (bf16-compile-cg) · f32 1.17 | 0.03 (bf16-compile-cg) · f32 2.14 |
| pj_m2048_n28672_k4096 | 3.976 (bf16-compile-cg) · f32 3.18 | 7.069 (bf16-compile-ro) · f32 2.93 | 1.954 (bf16-compile-ro) · f32 13.75 | 2.968 (bf16-compile-ro) · f32 3.21 | 0.6093 (bf16-compile) · f32 16.60 |
| pj_m2048_n4096_k14336 | 2.091 (bf16-compile-cg) · f32 3.07 | 4.096 (triton-bf16) · f32 2.70 | 0.9462 (bf16-compile-ro) · f32 14.10 | 1.61 (bf16-compile) · f32 2.74 | 0.3008 (bf16-compile-cg) · f32 17.51 |
| pj_m2048_n4096_k4096 | 0.6001 (bf16-compile-cg) · f32 3.06 | 1.061 (bf16-compile-cg) · f32 2.95 | 0.3092 (bf16) · f32 12.51 | 0.4362 (bf16-compile) · f32 2.96 | 0.0927 (bf16-compile-ro) · f32 16.33 |
| pj_m2048_n6144_k4096 | 0.8827 (bf16-compile-ro) · f32 3.00 | 1.329 (bf16-compile) · f32 3.43 | 0.4096 (bf16-compile-cg) · f32 14.17 | 0.598 (bf16-compile) · f32 3.23 | 0.1373 (bf16-compile-cg) · f32 16.10 |
| llm_blkdec_b1_l4096 | 0.5538 (bf16-compile) · f32 1.93 · bf16w 1.15 | 0.7022 (bf16-compile) · f32 1.61 · bf16w 1.01 | 0.4541 (bf16-compile) · f32 2.11 · bf16w 1.73 | 0.6213 (bf16-compile-cg) · f32 2.39 · bf16w 0.94 | 0.2903 (bf16-compile-cg) · f32 1.96 · bf16w 1.22 |
| llm_blkpre_t2048 | 8.029 (bf16-compile-cg) · f32 3.40 | 15.2 (bf16-compile) · f32 4.02 | 4.451 (bf16-compile) · f32 12.55 | 6.295 (bf16-compile-cg) · f32 3.19 | 1.319 (bf16-compile-cg) · f32 16.22 |
| llm_dec_b1_l2048 | 0.1469 (bf16-compile) · f32 1.63 · bf16w 1.02 | 0.2555 (bf16-compile) · f32 1.08 · bf16w 0.72 | 0.1574 (bf16-compile) · f32 1.37 · bf16w 1.08 | 0.2397 (bf16-compile-cg) · f32 0.99 · bf16w 0.64 | 0.1455 (bf16-compile-cg) · f32 0.79 · bf16w 0.65 |
| llm_dec_b1_l8192 | 0.1953 (bf16-compile) · f32 1.70 · bf16w 1.23 | 0.2997 (bf16-compile) · f32 1.26 · bf16w 0.96 | 0.1956 (bf16-compile) · f32 1.42 · bf16w 1.18 | 0.2578 (bf16-compile) · f32 1.29 · bf16w 0.97 | 0.1589 (bf16-compile-cg) · f32 0.99 · bf16w 0.86 |
| llm_dec_b32_l2048 | 0.4563 (bf16-compile) · f32 2.27 · bf16w 2.09 | 0.8455 (bf16-compile) · f32 1.44 · bf16w 1.41 | 0.4314 (bf16-cudnn) · f32 1.99 · bf16w 2.01 | 0.4904 (bf16-compile) · f32 2.17 · bf16w 1.96 | 0.2325 (bf16-compile-ro) · f32 2.05 · bf16w 2.11 |
| llm_dec_b32_l8192 | 1.416 (flashinfer-bf16) · f32 2.48 · bf16w 2.42 | 1.792 (bf16-cudnn) · f32 2.38 · bf16w 2.38 | 0.9729 (bf16-cudnn) · f32 2.61 · bf16w 2.61 | 1.385 (bf16-compile) · f32 2.53 · bf16w 2.46 | 0.4943 (bf16-compile-ro) · f32 2.13 · bf16w 2.16 |
| llm_dec_b8_l2048 | 0.2166 (bf16-compile) · f32 2.02 · bf16w 1.57 | 0.3608 (bf16-compile) · f32 2.79 · bf16w 2.29 | 0.2293 (bf16-compile) · f32 1.71 · bf16w 1.50 | 0.2728 (bf16-compile) · f32 1.75 · bf16w 1.46 | 0.1693 (bf16-compile-cg) · f32 1.24 · bf16w 1.12 |
| llm_dec_b8_l8192 | 0.4675 (liger-bf16) · f32 2.28 · bf16w 2.09 | 0.817 (liger-bf16) · f32 2.19 · bf16w 1.97 | 0.4219 (bf16) · f32 1.92 · bf16w 1.80 | 0.4973 (bf16-compile) · f32 2.14 · bf16w 1.99 | 0.2332 (bf16-compile-cg) · f32 2.04 · bf16w 1.94 |
| llm_pre_t2048 | 1.915 (bf16-compile-cg) · f32 4.47 | 3.569 (bf16-compile-cg) · f32 4.02 | 1.046 (bf16-compile-cg) · f32 14.47 | 1.319 (bf16-compile-cg) · f32 4.93 | 0.3511 (bf16-compile-cg) · f32 17.43 |
| llm_pre_t8192 | 11.38 (bf16-compile) · f32 7.09 | 20.51 (bf16-compile) · f32 7.49 | 6.483 (bf16-compile) · f32 20.41 | 8.5 (bf16-compile) · f32 7.59 | 1.923 (bf16-compile) · f32 26.82 |

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
| rtx-3090 | llm_blkdec_b1_l4096 | 0.7022 (torch-bf16-compile) 1.61 loss | 1.251 (torch-compile) 0.90 win | old best a bf16 leg |
| rtx-3090 | llm_blkpre_t2048_tf32 | 34.59 (torch-compile-tf32) 1.02 even | 30.99 (triton-tf32) 1.14 loss | best now a gemm_lib leg |
| rtx-3090 | llm_dec_b1_l2048 | 0.2555 (torch-bf16-compile) 1.08 loss | 0.3629 (torch-compile) 0.76 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l8192 | 0.2997 (torch-bf16-compile) 1.26 loss | 0.4934 (torch-compile) 0.77 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l2048 | 0.8455 (torch-bf16-compile) 1.44 loss | 2.943 (torch-compile-ro) 0.41 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192 | 3.223 (torch-bf16) 1.32 loss | 11.18 (torch-compile-ro) 0.38 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l8192 | 0.8285 (torch-bf16) 2.16 loss | 2.957 (torch-compile) 0.60 win | old best a bf16 leg |
| rtx-3090 | mlp_2048 | 1.829 (chain-lt) 0.88 win | 1.628 (torch-compile-ro) 0.99 even | other |
| rtx-3090 | mlp_2048_tf32 | 1.556 (torch-eager-tf32) 0.87 win | 1.184 (torch-compile-ro-tf32) 1.15 loss | other |
| rtx-3090 | pj_m128_n28672_k4096 | 0.468 (torch-bf16-compile) 2.62 loss | 1.235 (torch-compile-ro) 0.99 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n18432_k7168 | 0.3205 (torch-bf16-compile) 1.94 loss | 0.6072 (cublas) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n28672_k4096 | 7.071 (torch-bf16) 2.93 loss | 22.48 (torch-compile-ro) 0.92 win | old best a bf16 leg |
| rtx-3090 | pj_m2048_n4096_k14336 | 4.342 (torch-bf16-compile) 2.54 loss | 11.75 (lt) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k4096 | 1.138 (torch-bf16-compile) 2.75 loss | 3.393 (lt) 0.92 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k4096_tf32 | 2.486 (torch-eager-tf32) 0.94 win | 2.442 (torch-compile-ro-tf32) 0.96 even | other |
| rtx-3090 | pj_m2048_n6144_k4096 | 1.329 (torch-bf16-compile) 3.43 loss | 4.906 (torch-eager) 0.93 win | old best a bf16 leg |
| a100 | llm_blkpre_t2048 | 4.451 (torch-bf16-compile) 12.55 loss | 54.57 (torch-compile-cg) 1.02 even | old best a bf16 leg |
| a100 | llm_dec_b1_l2048 | 0.1574 (torch-bf16-compile) 1.37 loss | 0.2501 (torch-compile) 0.86 win | old best a bf16 leg |
| a100 | llm_dec_b1_l8192 | 0.1956 (torch-bf16-compile) 1.42 loss | 0.3949 (torch-compile) 0.70 win | old best a bf16 leg |
| a100 | llm_dec_b32_l2048 | 0.4747 (torch-bf16-compile) 1.81 loss | 1.832 (torch-compile-cg) 0.47 win | old best a bf16 leg |
| a100 | llm_dec_b32_l8192 | 1.209 (torch-bf16) 2.10 loss | 6.499 (torch-compile-cg) 0.39 win | old best a bf16 leg |
| a100 | llm_dec_b8_l2048 | 0.2293 (torch-bf16-compile) 1.71 loss | 0.6163 (torch-compile) 0.64 win | old best a bf16 leg |
| a100 | llm_dec_b8_l8192 | 0.4219 (torch-bf16) 1.92 loss | 1.806 (torch-compile) 0.45 win | old best a bf16 leg |
| a100 | pj_m128_n28672_k4096 | 0.2918 (torch-bf16-compile) 5.92 loss | 1.747 (torch-compile-ro) 0.99 even | old best a bf16 leg |
| a100 | pj_m16_n18432_k7168 | 0.214 (torch-bf16-compile) 2.07 loss | 0.4454 (torch-compile) 1.00 even | old best a bf16 leg |
| a100 | pj_m16_n28672_k4096 | 0.1884 (torch-bf16-compile) 2.22 loss | 0.4076 (torch-compile) 1.03 even | old best a bf16 leg |
| a100 | pj_m16_n4096_k14336 | 0.1106 (torch-bf16-compile) 2.12 loss | 0.2509 (lt) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m2048_n28672_k4096 | 2.002 (torch-bf16) 13.42 loss | 26.02 (torch-compile) 1.03 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k14336 | 0.9615 (torch-bf16) 13.87 loss | 12.9 (torch-eager) 1.03 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k4096 | 0.3092 (torch-bf16) 12.51 loss | 3.704 (torch-compile-cg) 1.04 even | old best a bf16 leg |
| a100 | pj_m2048_n6144_k4096 | 0.4147 (torch-bf16-compile) 14.00 loss | 5.784 (torch-compile-cg) 1.00 even | old best a bf16 leg |
| a100 | softmax_rows_4096 | 7.569 (chain-fused) 1.03 even | 7.375 (torch-compile-ro) 1.06 loss | other |
| rtx-4090 | attn_2048_64 | 0.0788 (torch-sdpa-compile) 0.96 even | 0.0645 (torch-compile-ro) 1.18 loss | other |
| rtx-4090 | epi_4096 | 3.276 (torch-compile) 0.79 win | 2.614 (torch-compile-ro) 0.99 even | other |
| rtx-4090 | llm_blkpre_t2048 | 6.325 (torch-bf16-compile) 3.18 loss | 23.03 (liger) 0.87 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_dec_b1_l8192 | 0.2578 (torch-bf16-compile) 1.29 loss | 0.372 (torch-compile) 0.89 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l2048 | 0.4904 (torch-bf16-compile) 2.17 loss | 2.293 (torch-compile) 0.46 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l8192 | 1.385 (torch-bf16-compile) 2.53 loss | 8.986 (torch-compile) 0.39 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l2048 | 0.2728 (torch-bf16-compile) 1.75 loss | 0.7501 (torch-compile) 0.63 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l8192 | 0.4973 (torch-bf16-compile) 2.14 loss | 2.444 (torch-compile) 0.43 win | old best a bf16 leg |
| rtx-4090 | mlp2_2048 | 1.377 (torch-compile) 0.95 win | 1.343 (torch-compile-ro) 0.97 even | other |
| rtx-4090 | pj_m128_n28672_k4096 | 0.3082 (torch-bf16-compile) 2.23 loss | 0.6861 (lt-fresh) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n4096_k4096_tf32 | 0.1055 (torch-compile-tf32) 1.00 even | 0.0676 (cublas-tf32-fresh) 1.55 loss | best now a gemm_lib leg |
| rtx-4090 | pj_m1_n2048_k7168 | 0.0502 (torch-bf16-compile) 0.78 win | 0.0276 (lt-fresh) 1.42 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n4096_k4096 | 0.0553 (torch-bf16-compile) 0.78 win | 0.0205 (lt-fresh) 2.12 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n28672_k4096 | 3.18 (torch-bf16-compile) 2.99 loss | 11.06 (lt) 0.86 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k14336 | 1.61 (torch-bf16-compile) 2.74 loss | 5.721 (torch-compile-cg) 0.77 win | old best a bf16 leg |
| rtx-4090 | rt_mlp2_1024 | 1.85 (torch-compile) 0.92 win | 1.588 (torch-compile-cg) 1.07 loss | other |
| h100 | llm_dec_b32_l2048 | 0.2543 (torch-bf16-compile) 1.87 loss | 1.149 (torch-compile) 0.41 win | old best a bf16 leg |
| h100 | llm_dec_b32_l8192 | 0.5183 (torch-bf16-compile) 2.03 loss | 3.878 (torch-compile) 0.27 win | old best a bf16 leg |
| h100 | llm_dec_b8_l2048 | 0.2172 (torch-bf16-compile) 0.97 even | 0.4127 (torch-compile) 0.51 win | old best a bf16 leg |
| h100 | llm_dec_b8_l2048_bf16w | 0.2062 (torch-bf16-compile) 0.92 win | 0.166 (torch-bf16-compile-cg) 1.15 loss | other |
| h100 | llm_dec_b8_l8192 | 0.2619 (torch-bf16-compile) 1.81 loss | 1.096 (torch-compile) 0.43 win | old best a bf16 leg |
| h100 | pj_m128_n28672_k4096 | 0.0937 (torch-bf16-compile) 7.48 loss | 0.6772 (lt) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |

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
| rtx-3090 | epi_4096 | f32 | 6.909 (torch-compile) 0.80 | 6.855 (torch-compile-cg) 0.80 | win → win |
| rtx-3090 | pj_m128_n28672_k4096 | f32 | 1.535 (torch-compile) 0.80 | 1.235 (torch-compile-ro) 0.99 | win → even |
| rtx-3090 | pj_m128_n28672_k4096_tf32 | tf32 | 0.9339 (torch-compile-tf32) 1.12 | 0.8499 (torch-compile-ro-tf32) 1.23 | loss → loss |
| rtx-3090 | pj_m128_n4096_k14336 | f32 | 0.6574 (torch-compile) 1.08 | 0.6564 (torch-compile-cg) 1.08 | loss → loss |
| rtx-3090 | pj_m128_n4096_k4096 | f32 | 0.2202 (torch-eager) 1.11 | 0.2191 (torch-compile-cg) 1.11 | loss → loss |
| rtx-3090 | pj_m128_n4096_k4096_tf32 | tf32 | 0.1649 (torch-eager-tf32) 1.29 | 0.1638 (torch-compile-cg-tf32) 1.30 | loss → loss |
| rtx-3090 | pj_m16_n2048_k7168 | f32 | 0.0829 (torch-compile) 1.38 | 0.0819 (torch-compile-cg) 1.40 | loss → loss |
| rtx-3090 | pj_m16_n6144_k4096 | f32 | 0.1239 (torch-compile) 1.31 | 0.1229 (torch-compile-cg) 1.33 | loss → loss |
| rtx-3090 | pj_m1_n28672_k4096 | f32 | 0.5386 (torch-eager) 1.05 | 0.5366 (torch-compile-ro) 1.05 | loss → loss |
| rtx-3090 | pj_m1_n4096_k14336 | f32 | 0.2775 (lt) 1.08 | 0.2755 (torch-compile-ro) 1.09 | loss → loss |
| rtx-3090 | pj_m1_n6144_k4096 | f32 | 0.1229 (torch-compile) 1.21 | 0.1219 (torch-compile-cg) 1.22 | loss → loss |
| rtx-3090 | pj_m2048_n28672_k4096 | f32 | 22.51 (torch-compile) 0.92 | 22.48 (torch-compile-ro) 0.92 | win → win |
| rtx-3090 | pj_m2048_n4096_k14336_tf32 | tf32 | 7.691 (torch-compile-tf32) 1.06 | 7.68 (torch-compile-cg-tf32) 1.06 | loss → loss |
| rtx-3090 | pj_m2048_n4096_k4096_tf32 | tf32 | 2.46 (cublas-tf32-fresh) 0.95 | 2.442 (torch-compile-ro-tf32) 0.96 | even → even |
| rtx-3090 | mlp2_2048_tf32 | tf32 | 2.745 (torch-eager-tf32) 1.00 | 2.648 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-3090 | mlp_2048 | f32 | 1.829 (chain-lt) 0.88 | 1.628 (torch-compile-ro) 0.99 | win → even |
| rtx-3090 | mlp_2048_tf32 | tf32 | 1.556 (torch-eager-tf32) 0.87 | 1.184 (torch-compile-ro-tf32) 1.15 | win → loss |
| rtx-3090 | mlp_512 | f32 | 0.0502 (torch-compile) 1.53 | 0.0451 (triton) 1.71 | loss → loss |
| rtx-3090 | smlp_4096_256_1024 | f32 | 0.2232 (torch-compile) 1.22 | 0.2222 (torch-compile-cg) 1.23 | loss → loss |
| rtx-3090 | attn_1024_64 | f32 | 0.0573 (chain-fused) 1.71 | 0.0451 (torch-compile-ro) 2.18 | loss → loss |
| rtx-3090 | attn_2048_128 | f32 | 0.1894 (torch-sdpa-compile) 1.41 | 0.1864 (torch-sdpa-compile-cg) 1.43 | loss → loss |
| rtx-3090 | attn_2048_64 | f32 | 0.1546 (torch-compile) 1.07 | 0.1331 (torch-compile-ro) 1.24 | loss → loss |
| rtx-3090 | causal_1024_256 | f32 | 199.4 (torch-compile) 0.12 | 175.2 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | gen_1024_512_256 | f32 | 98.05 (torch-compile) 0.13 | 88.2 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | lsm_1_131072 | f32 | 0.0297 (torch-compile) 10.34 | 0.0256 (torch-compile-cg) 12.00 | loss → loss |
| rtx-3090 | lsm_32_8192 | f32 | 0.0113 (torch-sdpa-compile) 3.53 | 0.0113 (torch-compile-ro) 3.53 | loss → loss |
| rtx-3090 | lsm_4_32768 | f32 | 0.0205 (torch-compile) 3.98 | 0.0164 (torch-sdpa-compile-cg) 4.98 | loss → loss |
| rtx-3090 | prefill_1024_512_256 | f32 | 98.99 (torch-compile) 0.13 | 87.9 (torch-compile-cg) 0.15 | win → win |
| rtx-3090 | rowsum_2048 | f32 | 0.7567 (torch-compile) 1.11 | 0.7537 (torch-compile-ro) 1.12 | loss → loss |
| rtx-3090 | softmax_rows_2048 | f32 | 0.7547 (torch-compile) 1.17 | 0.7516 (torch-compile-cg) 1.17 | loss → loss |
| rtx-3090 | softmax_rows_4096 | f32 | 6.954 (torch-compile) 0.91 | 6.945 (torch-compile-ro) 0.91 | win → win |
| rtx-3090 | llm_blkpre_t2048 | f32 | 51.09 (torch-compile) 1.20 | 49.54 (liger) 1.23 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_tf32 | tf32 | 34.59 (torch-compile-tf32) 1.02 | 30.99 (triton-tf32) 1.14 | even → loss |
| rtx-3090 | llm_dec_b32_l2048 | f32 | 2.991 (torch-compile) 0.41 | 2.943 (torch-compile-ro) 0.41 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16w | bf16w | 0.8413 (torch-bf16-compile) 1.42 | 0.8302 (torch-bf16-compile-cg) 1.44 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192 | f32 | 11.31 (torch-compile) 0.38 | 11.18 (torch-compile-ro) 0.38 | win → win |
| rtx-3090 | llm_dec_b32_l8192_bf16w | bf16w | 3.225 (torch-bf16) 1.32 | 1.778 (torch-bf16-cudnn) 2.39 | loss → loss |
| rtx-3090 | llm_dec_b8_l8192_bf16w | bf16w | 0.8253 (torch-bf16) 1.95 | 0.8092 (liger-bf16) 1.99 | loss → loss |
| rtx-3090 | llm_pre_t2048 | f32 | 15.35 (torch-compile) 0.93 | 13.27 (torch-sdpa-efficient) 1.08 | win → loss |
| rtx-3090 | llm_pre_t2048_tf32 | tf32 | 10.96 (torch-compile-tf32) 1.15 | 8.286 (triton-tf32) 1.52 | loss → loss |
| rtx-3090 | llm_pre_t8192 | f32 | 147.9 (torch-compile) 1.04 | 97.29 (liger) 1.58 | even → loss |
| rtx-3090 | ew_4096 | f32 | 0.2417 (chain-fused) 0.66 | 0.2406 (torch-compile-cg) 0.66 | win → win |
| rtx-3090 | ewconvew_2048_5 | f32 | 0.2621 (torch-compile) 0.50 | 0.2601 (torch-compile-cg) 0.50 | win → win |
| rtx-3090 | fir_4194304_16 | f32 | 0.1137 (torch-eager) 0.65 | 0.1126 (torch-compile-cg) 0.66 | win → win |
| rtx-3090 | firew_rowsum_2048x2048_16 | f32 | 0.1372 (torch-compile) 0.74 | 0.1362 (torch-compile-cg) 0.75 | win → win |
| rtx-3090 | indep_2048 | f32 | 1.866 (chain-streams) 0.87 | 1.848 (torch-compile-ro) 0.88 | win → win |
| rtx-3090 | indep_512 | f32 | 0.0461 (torch-compile) 1.78 | 0.0451 (torch-compile-cg) 1.82 | loss → loss |
| rtx-3090 | loopmm_1024_8 | f32 | 0.9769 (torch-compile) 1.10 | 0.9267 (torch-compile-ro) 1.16 | loss → loss |
| rtx-3090 | rt_mlp2_1024 | f32 | 4.981 (torch-compile) 0.84 | 4.935 (torch-compile-cg) 0.85 | win → win |
| a100 | pj_m128_n28672_k4096 | f32 | 1.781 (lt) 0.97 | 1.747 (torch-compile-ro) 0.99 | even → even |
| a100 | pj_m128_n28672_k4096_tf32 | tf32 | 0.4516 (lt-tf32) 1.20 | 0.4178 (triton-tf32) 1.29 | loss → loss |
| a100 | pj_m128_n6144_k4096_tf32 | tf32 | 0.1034 (cublas-tf32) 1.55 | 0.0993 (triton-tf32) 1.61 | loss → loss |
| a100 | pj_m16_n6144_k4096 | f32 | 0.1167 (torch-compile) 1.11 | 0.1157 (torch-compile-cg) 1.12 | loss → loss |
| a100 | pj_m1_n28672_k4096 | f32 | 0.3492 (torch-compile) 1.09 | 0.3482 (torch-compile-cg) 1.09 | loss → loss |
| a100 | pj_m2048_n4096_k14336_tf32 | tf32 | 2.095 (torch-eager-tf32) 1.29 | 2.044 (torch-compile-ro-tf32) 1.33 | loss → loss |
| a100 | pj_m2048_n4096_k4096 | f32 | 3.74 (cublas) 1.03 | 3.704 (torch-compile-cg) 1.04 | even → even |
| a100 | pj_m2048_n6144_k4096 | f32 | 5.787 (torch-compile) 1.00 | 5.784 (torch-compile-cg) 1.00 | even → even |
| a100 | pj_m2048_n6144_k4096_tf32 | tf32 | 0.8141 (torch-compile-tf32) 1.53 | 0.8069 (torch-compile-cg-tf32) 1.55 | loss → loss |
| a100 | mlp2_2048_tf32 | tf32 | 0.7055 (torch-compile-tf32) 1.32 | 0.6973 (torch-compile-cg-tf32) 1.34 | loss → loss |
| a100 | mlp_2048_tf32 | tf32 | 0.3533 (torch-compile-tf32) 1.44 | 0.3512 (torch-compile-cg-tf32) 1.45 | loss → loss |
| a100 | attn_1024_64 | f32 | 0.0604 (chain-fused) 1.05 | 0.0543 (torch-sdpa-compile-cg) 1.17 | loss → loss |
| a100 | causal_1024_256 | f32 | 75.36 (torch-compile) 0.49 | 74.11 (torch-compile-cg) 0.50 | win → win |
| a100 | lsm_1_131072 | f32 | 0.0328 (torch-compile) 10.80 | 0.0256 (torch-compile-cg) 13.84 | loss → loss |
| a100 | lsm_32_8192 | f32 | 0.0143 (torch-sdpa-compile) 2.13 | 0.0123 (torch-sdpa-compile-cg) 2.48 | loss → loss |
| a100 | lsm_4_32768 | f32 | 0.0266 (torch-sdpa-compile) 2.79 | 0.0195 (torch-sdpa-compile-cg) 3.81 | loss → loss |
| a100 | prefill_1024_512_256 | f32 | 38.03 (torch-compile) 0.49 | 37.47 (torch-compile-cg) 0.50 | win → win |
| a100 | rowsum_2048 | f32 | 0.9964 (torch-compile) 1.04 | 0.9943 (torch-compile-ro) 1.04 | even → even |
| a100 | rowsum_4096 | f32 | 7.347 (chain-eager) 1.05 | 7.337 (torch-compile-ro) 1.06 | loss → loss |
| a100 | softmax_rows_2048 | f32 | 1.044 (chain-fused) 1.01 | 1.019 (triton) 1.03 | even → even |
| a100 | softmax_rows_4096 | f32 | 7.569 (chain-fused) 1.03 | 7.375 (torch-compile-ro) 1.06 | even → loss |
| a100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.4916 (torch-bf16-compile) 1.60 | 0.4551 (torch-bf16-compile-cg) 1.73 | loss → loss |
| a100 | llm_blkpre_t2048 | f32 | 54.6 (torch-compile) 1.02 | 54.57 (torch-compile-cg) 1.02 | even → even |
| a100 | llm_blkpre_t2048_tf32 | tf32 | 10.54 (torch-compile-tf32) 1.55 | 9.895 (triton-tf32) 1.65 | loss → loss |
| a100 | llm_dec_b32_l2048 | f32 | 1.839 (torch-compile) 0.47 | 1.832 (torch-compile-cg) 0.47 | win → win |
| a100 | llm_dec_b32_l2048_bf16w | bf16w | 0.5003 (torch-bf16) 1.73 | 0.4262 (torch-bf16-cudnn) 2.03 | loss → loss |
| a100 | llm_dec_b32_l8192 | f32 | 6.558 (torch-compile) 0.39 | 6.499 (torch-compile-cg) 0.39 | win → win |
| a100 | llm_dec_b32_l8192_bf16w | bf16w | 1.206 (torch-bf16) 2.11 | 0.975 (torch-bf16-cudnn) 2.61 | loss → loss |
| a100 | llm_pre_t2048 | f32 | 14.65 (torch-compile) 1.03 | 11.96 (liger) 1.27 | even → loss |
| a100 | llm_pre_t2048_tf32 | tf32 | 3.709 (torch-compile-tf32) 2.00 | 2.832 (triton-tf32) 2.62 | loss → loss |
| a100 | llm_pre_t8192 | f32 | 115.7 (torch-compile) 1.14 | 61.05 (liger) 2.17 | loss → loss |
| a100 | conv_2048_5 | f32 | 0.1382 (torch-eager) 0.52 | 0.1372 (torch-compile-cg) 0.53 | win → win |
| a100 | ewconvew_2048_5 | f32 | 0.1761 (torch-compile) 0.50 | 0.1731 (torch-compile-cg) 0.51 | win → win |
| a100 | fir_4194304_16 | f32 | 0.086 (torch-eager) 0.58 | 0.085 (torch-compile-cg) 0.59 | win → win |
| a100 | firew_rowsum_2048x2048_16 | f32 | 0.0973 (torch-compile) 0.63 | 0.0942 (torch-compile-ro) 0.65 | win → win |
| a100 | indep_2048 | f32 | 1.968 (chain-lt) 1.14 | 1.966 (torch-compile-cg) 1.14 | loss → loss |
| a100 | loop_4096 | f32 | 59.02 (torch-eager) 1.07 | 58.98 (torch-compile-ro) 1.07 | loss → loss |
| a100 | rt_mlp2_1024 | f32 | 5.46 (torch-eager) 1.25 | 4.293 (torch-compile-cg) 1.59 | loss → loss |
| a100 | rt_mlp2_2048 | f32 | 31.86 (torch-compile) 1.17 | 31.75 (torch-compile-cg) 1.17 | loss → loss |
| a100 | twin_2048 | f32 | 1.965 (torch-compile) 1.03 | 1.962 (torch-compile-cg) 1.03 | even → even |
| rtx-4090 | epi_4096 | f32 | 3.276 (torch-compile) 0.79 | 2.614 (torch-compile-ro) 0.99 | win → even |
| rtx-4090 | pj_m2048_n4096_k14336 | f32 | 5.735 (torch-eager) 0.77 | 5.721 (torch-compile-cg) 0.77 | win → win |
| rtx-4090 | pj_m2048_n4096_k14336_tf32 | tf32 | 3.053 (cublas-tf32-fresh) 1.08 | 3.003 (torch-compile-ro-tf32) 1.10 | loss → loss |
| rtx-4090 | pj_m2048_n6144_k4096 | f32 | 2.312 (cublas) 0.83 | 1.827 (torch-compile-ro) 1.06 | win → loss |
| rtx-4090 | pj_m2048_n6144_k4096_tf32 | tf32 | 1.256 (cublas-tf32) 1.14 | 1.162 (torch-compile-ro-tf32) 1.23 | loss → loss |
| rtx-4090 | mlp2_2048 | f32 | 1.377 (torch-compile) 0.95 | 1.343 (torch-compile-ro) 0.97 | win → even |
| rtx-4090 | mlp2_2048_tf32 | tf32 | 0.8192 (torch-compile-tf32) 1.26 | 0.8141 (torch-compile-cg-tf32) 1.27 | loss → loss |
| rtx-4090 | mlp_2048 | f32 | 0.6584 (torch-compile) 0.97 | 0.6543 (torch-compile-cg) 0.98 | even → even |
| rtx-4090 | mlp_2048_tf32 | tf32 | 0.4127 (torch-compile-tf32) 1.22 | 0.4096 (torch-compile-cg-tf32) 1.23 | loss → loss |
| rtx-4090 | mlp_512 | f32 | 0.0297 (torch-compile) 2.08 | 0.0287 (torch-compile-cg) 2.16 | loss → loss |
| rtx-4090 | smlp_4096_256_1024 | f32 | 0.1087 (torch-compile) 1.11 | 0.1085 (triton) 1.11 | loss → loss |
| rtx-4090 | attn_1024_64 | f32 | 0.0316 (torch-sdpa-compile) 1.91 | 0.0266 (torch-sdpa-compile-cg) 2.27 | loss → loss |
| rtx-4090 | attn_2048_128 | f32 | 0.0891 (torch-compile) 1.14 | 0.084 (torch-sdpa-compile-cg) 1.21 | loss → loss |
| rtx-4090 | attn_2048_64 | f32 | 0.0788 (torch-sdpa-compile) 0.96 | 0.0645 (torch-compile-ro) 1.18 | even → loss |
| rtx-4090 | causal_1024_256 | f32 | 218 (torch-compile) 0.09 | 178.1 (torch-compile-cg) 0.10 | win → win |
| rtx-4090 | gen_1024_512_256 | f32 | 99.81 (torch-compile) 0.09 | 88.5 (torch-compile-ro) 0.11 | win → win |
| rtx-4090 | lsm_1_131072 | f32 | 0.0276 (torch-compile) 8.49 | 0.0236 (torch-compile-cg) 9.92 | loss → loss |
| rtx-4090 | lsm_32_8192 | f32 | 0.009 (torch-compile) 4.44 | 0.0082 (torch-compile-cg) 4.87 | loss → loss |
| rtx-4090 | lsm_4_32768 | f32 | 0.0174 (torch-compile) 3.78 | 0.0133 (torch-compile-cg) 4.94 | loss → loss |
| rtx-4090 | prefill_1024_512_256 | f32 | 103.6 (torch-compile) 0.09 | 95.96 (torch-compile-cg) 0.10 | win → win |
| rtx-4090 | softmax_rows_2048 | f32 | 0.3492 (torch-compile) 0.97 | 0.3359 (torch-compile-cg) 1.01 | even → even |
| rtx-4090 | softmax_rows_4096 | f32 | 3.319 (chain-fused) 0.80 | 2.912 (torch-compile-cg) 0.91 | win → win |
| rtx-4090 | llm_blkpre_t2048 | f32 | 25.4 (torch-compile) 0.79 | 23.03 (liger) 0.87 | win → win |
| rtx-4090 | llm_blkpre_t2048_tf32 | tf32 | 14.66 (torch-compile-tf32) 1.13 | 13.06 (triton-tf32) 1.27 | loss → loss |
| rtx-4090 | llm_dec_b1_l2048_bf16w | bf16w | 0.2545 (torch-bf16-compile) 0.60 | 0.2379 (torch-bf16-compile-cg) 0.64 | win → win |
| rtx-4090 | llm_dec_b1_l8192_bf16w | bf16w | 0.2695 (torch-bf16-compile) 0.92 | 0.2688 (torch-bf16-compile-cg) 0.93 | win → win |
| rtx-4090 | llm_dec_b32_l8192_bf16w | bf16w | 1.399 (torch-bf16-compile) 2.43 | 1.398 (torch-bf16-compile-cg) 2.44 | loss → loss |
| rtx-4090 | llm_pre_t2048 | f32 | 7.779 (torch-compile) 0.84 | 5.689 (liger) 1.14 | win → loss |
| rtx-4090 | llm_pre_t2048_tf32 | tf32 | 4.998 (torch-compile-tf32) 1.15 | 3.398 (triton-tf32) 1.69 | loss → loss |
| rtx-4090 | llm_pre_t8192 | f32 | 75.7 (torch-compile) 0.85 | 40.13 (liger) 1.61 | win → loss |
| rtx-4090 | conv_2048_5 | f32 | 0.0951 (torch-eager) 0.60 | 0.0932 (torch-compile-cg) 0.62 | win → win |
| rtx-4090 | ewconvew_2048_5 | f32 | 0.1321 (torch-compile) 0.52 | 0.1311 (torch-compile-cg) 0.52 | win → win |
| rtx-4090 | fir_4194304_16 | f32 | 0.0645 (torch-eager) 0.62 | 0.0625 (torch-compile-ro) 0.64 | win → win |
| rtx-4090 | firew_rowsum_2048x2048_16 | f32 | 0.0686 (torch-compile) 0.68 | 0.0666 (torch-compile-cg) 0.70 | win → win |
| rtx-4090 | diamond_2048 | f32 | 0.9656 (torch-compile) 1.01 | 0.9626 (torch-compile-cg) 1.01 | even → even |
| rtx-4090 | indep_2048 | f32 | 0.6502 (torch-compile) 1.00 | 0.6492 (torch-compile-cg) 1.00 | even → even |
| rtx-4090 | loop_4096 | f32 | 24.56 (torch-eager) 0.76 | 24.52 (torch-compile-ro) 0.76 | win → win |
| rtx-4090 | loopmm_1024_8 | f32 | 0.3635 (torch-compile) 1.19 | 0.3584 (torch-compile-cg) 1.21 | loss → loss |
| rtx-4090 | rt_mlp2_1024 | f32 | 1.85 (torch-compile) 0.92 | 1.588 (torch-compile-cg) 1.07 | win → loss |
| rtx-4090 | rt_mlp2_2048 | f32 | 13.37 (torch-compile) 0.78 | 13.35 (torch-compile-ro) 0.78 | win → win |
| rtx-4090 | twin_2048 | f32 | 0.6769 (torch-compile) 0.96 | 0.6758 (torch-compile-cg) 0.96 | even → even |
| h100 | pj_m128_n6144_k4096 | f32 | 0.1496 (lt-fresh) 1.32 | 0.1494 (torch-compile-cg) 1.32 | loss → loss |
| h100 | pj_m16_n28672_k4096 | f32 | 0.176 (torch-compile) 1.17 | 0.1757 (torch-compile-cg) 1.17 | loss → loss |
| h100 | pj_m2048_n4096_k14336_tf32 | tf32 | 0.5827 (torch-eager-tf32) 1.70 | 0.5811 (torch-compile-cg-tf32) 1.70 | loss → loss |
| h100 | mlp2_2048_tf32 | tf32 | 0.2261 (torch-compile-tf32) 2.34 | 0.2145 (torch-compile-cg-tf32) 2.47 | loss → loss |
| h100 | mlp_2048_tf32 | tf32 | 0.1123 (torch-eager-tf32) 2.40 | 0.1107 (torch-compile-cg-tf32) 2.44 | loss → loss |
| h100 | mlp_512 | f32 | 0.0328 (torch-compile) 1.80 | 0.0317 (torch-compile-cg) 1.86 | loss → loss |
| h100 | attn_1024_64 | f32 | 0.0325 (torch-compile) 1.73 | 0.0268 (torch-compile-ro) 2.10 | loss → loss |
| h100 | attn_2048_128 | f32 | 0.0906 (torch-sdpa-compile) 1.14 | 0.0789 (torch-compile-ro) 1.31 | loss → loss |
| h100 | attn_2048_64 | f32 | 0.0656 (torch-sdpa-compile) 1.33 | 0.0603 (torch-compile-ro) 1.45 | loss → loss |
| h100 | causal_1024_256 | f32 | 133.2 (torch-compile) 0.16 | 120 (torch-compile-cg) 0.18 | win → win |
| h100 | gen_1024_512_256 | f32 | 67.77 (torch-compile) 0.16 | 61.59 (torch-compile-cg) 0.18 | win → win |
| h100 | lsm_1_131072 | f32 | 0.0425 (torch-compile) 7.33 | 0.0227 (torch-compile-cg) 13.72 | loss → loss |
| h100 | lsm_4_32768 | f32 | 0.0218 (torch-sdpa) 2.78 | 0.0214 (torch-compile-ro) 2.83 | loss → loss |
| h100 | prefill_1024_512_256 | f32 | 67.57 (torch-compile) 0.16 | 61.68 (torch-compile-ro) 0.18 | win → win |
| h100 | rowsum_2048 | f32 | 0.3485 (torch-compile) 1.18 | 0.3467 (torch-compile-cg) 1.18 | loss → loss |
| h100 | rowsum_4096 | f32 | 2.667 (torch-compile) 1.12 | 2.661 (torch-compile-ro) 1.12 | loss → loss |
| h100 | softmax_rows_2048 | f32 | 0.3564 (torch-compile) 1.18 | 0.3549 (torch-compile-cg) 1.19 | loss → loss |
| h100 | softmax_rows_4096 | f32 | 2.704 (torch-compile) 1.12 | 2.702 (torch-compile-cg) 1.12 | loss → loss |
| h100 | llm_blkdec_b1_l4096 | f32 | 0.481 (torch-compile) 1.18 | 0.4745 (torch-compile-cg) 1.20 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.3316 (torch-bf16-compile) 1.07 | 0.2899 (torch-bf16-compile-cg) 1.22 | loss → loss |
| h100 | llm_blkpre_t2048 | f32 | 19.54 (torch-compile) 1.09 | 18.46 (liger) 1.16 | loss → loss |
| h100 | llm_blkpre_t2048_tf32 | tf32 | 3.605 (torch-compile-tf32) 1.85 | 3.47 (torch-compile-cg-tf32) 1.92 | loss → loss |
| h100 | llm_dec_b1_l2048_bf16w | bf16w | 0.2034 (torch-bf16-compile) 0.46 | 0.1492 (torch-bf16-compile-cg) 0.63 | win → win |
| h100 | llm_dec_b1_l8192_bf16w | bf16w | 0.2066 (torch-bf16-compile) 0.66 | 0.1558 (torch-bf16-compile-cg) 0.87 | win → win |
| h100 | llm_dec_b32_l2048_bf16w | bf16w | 0.2475 (torch-bf16-compile) 1.98 | 0.2324 (torch-bf16-compile-cg) 2.11 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16w | bf16w | 0.5084 (torch-bf16-compile) 2.10 | 0.494 (torch-bf16-compile-ro) 2.16 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16w | bf16w | 0.2062 (torch-bf16-compile) 0.92 | 0.166 (torch-bf16-compile-cg) 1.15 | win → loss |
| h100 | llm_dec_b8_l8192_bf16w | bf16w | 0.2514 (torch-bf16-compile) 1.80 | 0.2307 (torch-bf16-compile-cg) 1.96 | loss → loss |
| h100 | llm_pre_t2048 | f32 | 5.47 (torch-compile) 1.12 | 4.512 (liger) 1.36 | loss → loss |
| h100 | llm_pre_t2048_tf32 | tf32 | 1.514 (torch-compile-tf32) 2.12 | 1.372 (triton-tf32) 2.35 | loss → loss |
| h100 | llm_pre_t8192 | f32 | 43.96 (torch-compile) 1.17 | 25.99 (liger) 1.98 | loss → loss |
| h100 | ewconvew_2048_5 | f32 | 0.1164 (torch-compile) 0.50 | 0.1142 (torch-compile-cg) 0.51 | win → win |
| h100 | firew_rowsum_2048x2048_16 | f32 | 0.0663 (torch-compile) 0.58 | 0.0653 (torch-compile-cg) 0.59 | win → win |
| h100 | loopmm_1024_8 | f32 | 0.4561 (torch-compile) 1.15 | 0.4513 (torch-compile-cg) 1.16 | loss → loss |
| h100 | rt_mlp2_1024 | f32 | 2.017 (torch-compile) 1.06 | 1.962 (torch-compile-ro) 1.09 | loss → loss |
| h100 | twin_2048 | f32 | 0.6849 (torch-compile) 1.14 | 0.6844 (torch-compile-ro) 1.14 | loss → loss |

Mapal wins that disappear: 11 — rtx-3090 llm_pre_t2048 (torch-sdpa-efficient), rtx-3090 mlp_2048 (torch-compile-ro), rtx-3090 mlp_2048_tf32 (torch-compile-ro-tf32), rtx-3090 pj_m128_n28672_k4096 (torch-compile-ro), rtx-4090 epi_4096 (torch-compile-ro), rtx-4090 llm_pre_t2048 (liger), rtx-4090 llm_pre_t8192 (liger), rtx-4090 mlp2_2048 (torch-compile-ro), rtx-4090 pj_m2048_n6144_k4096 (torch-compile-ro), rtx-4090 rt_mlp2_1024 (torch-compile-cg), h100 llm_dec_b8_l2048_bf16w (torch-bf16-compile-cg).

## Every competitor leg

Per row and part, every leg at the row's precision, grouped by competitor family: `leg ms (Mapal/leg)`; **bold** = the row's fastest leg; `✗gate` = timed but outside the value gate (not counted); `cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg for this row on another part; `unavailable` = the part's run reported the package missing. Families: vendor libs; torch eager; torch.compile; compile + CUDA graphs; reduce-overhead; SDPA; FlashAttention-2/3; FlashInfer; hand Triton; Liger; TensorRT-LLM; hand CUDA. Columns with no leg on any row of a table are left out.

### GEMM cells & projections

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| epi_4096 | rtx-5080 | 3.675 | chain-lt 3.567 (1.03)<br>chain-eager 3.7 (0.99)<br>chain-cutlass 3.761 (0.98) | torch-eager 3.747 (0.98) | torch-compile 3.558 (1.03) | torch-compile-cg 3.55 (1.04) | **torch-compile-ro 3.547 (1.04)** | triton 4.344 (0.85) |
| epi_4096 | rtx-3090 | 5.512 | chain-lt 6.937 (0.79)<br>chain-cutlass 7.076 (0.78)<br>chain-eager 7.08 (0.78) | torch-eager 6.952 (0.79) | torch-compile 6.909 (0.80) | **torch-compile-cg 6.855 (0.80)** | torch-compile-ro 6.858 (0.80) | triton 7.976 (0.69) |
| epi_4096 | a100 | 7.882 | **chain-lt 7.295 (1.08)**<br>chain-eager 7.388 (1.07)<br>chain-cutlass 7.641 (1.03) | torch-eager 7.479 (1.05) | torch-compile 7.667 (1.03) | torch-compile-cg 7.589 (1.04) | torch-compile-ro 7.375 (1.07) | triton 7.444 (1.06) |
| epi_4096 | rtx-4090 | 2.593 | chain-lt 3.336 (0.78)<br>chain-cutlass 3.356 (0.77)<br>chain-eager 3.447 (0.75) | torch-eager 3.338 (0.78) | torch-compile 3.276 (0.79) | torch-compile-cg 3.245 (0.80) | **torch-compile-ro 2.614 (0.99)** | triton 3.741 (0.69) |
| epi_4096 | h100 | 2.98 | **chain-lt 2.66 (1.12)**<br>chain-eager 2.719 (1.10)<br>chain-cutlass 3.355 (0.89) | torch-eager 2.753 (1.08) | torch-compile 2.704 (1.10) | torch-compile-cg 2.703 (1.10) | torch-compile-ro 2.702 (1.10) | triton 3.024 (0.99) |
| mm_16x2048x7168 | rtx-5080 | 0.0879 | **cublas 0.0799 (1.10)**<br>cutlass-tuned 0.385 (0.23)<br>cutlass 0.526 (0.17) |  |  |  |  | triton 0.1536 (0.57) |
| mm_16x2048x7168 | rtx-3090 | 0.09165 | **cublas 0.0886 (1.03)**<br>cutlass-tuned 0.2238 (0.41)<br>cutlass 0.7337 (0.12) |  |  |  |  | triton 0.2079 (0.44) |
| mm_16x2048x7168 | a100 | 0.09215 | **cublas 0.0911 (1.01)**<br>cutlass-tuned 0.3507 (0.26)<br>cutlass 1.351 (0.07) |  |  |  |  | triton 0.385 (0.24) |
| mm_16x2048x7168 | rtx-4090 | 0.105 | **cublas 0.09375 (1.12)**<br>cutlass-tuned 0.1925 (0.55)<br>cutlass 0.4813 (0.22) |  |  |  |  | triton 0.1495 (0.70) |
| mm_16x2048x7168 | h100 | 0.09525 | **cublas 0.05155 (1.85)**<br>cutlass-tuned 0.2802 (0.34)<br>cutlass 0.7521 (0.13) |  |  |  |  | triton 0.1942 (0.49) |
| mm_4096x4096x4096 | rtx-5080 | 3.812 | **cublas 3.652 (1.04)**<br>cutlass 3.725 (1.02)<br>cutlass-tuned 9.051 (0.42) |  |  |  |  | triton 4.23 (0.90) |
| mm_4096x4096x4096 | rtx-3090 | 9.525 | **cublas 8.091 (1.18)**<br>cutlass 8.536 (1.12)<br>cutlass-tuned 20.69 (0.46) |  |  |  |  | triton 8.262 (1.15) |
| mm_4096x4096x4096 | a100 | 7.68 | **cublas 7.375 (1.04)**<br>cutlass 7.704 (1.00)<br>cutlass-tuned 11.99 (0.64) |  |  |  |  | triton 7.546 (1.02) |
| mm_4096x4096x4096 | rtx-4090 | 3.503 | **cublas 3.425 (1.02)**<br>cutlass 3.489 (1.00)<br>cutlass-tuned 9.598 (0.36) |  |  |  |  | triton 3.905 (0.90) |
| mm_4096x4096x4096 | h100 | 3.143 | **cublas 2.663 (1.18)**<br>cutlass 3.339 (0.94)<br>cutlass-tuned 6.758 (0.47) |  |  |  |  | triton 3.016 (1.04) |
| pj_m128_n28672_k4096 | rtx-5080 | 0.8088 | **lt 0.7516 (1.08)**<br>lt-fresh 0.7516 (1.08)<br>cublas 0.7947 (1.02)<br>cublas-fresh 0.7967 (1.02) | torch-eager 0.8008 (1.01) | torch-compile 0.8008 (1.01) | torch-compile-cg 0.8008 (1.01) | torch-compile-ro 0.8008 (1.01) | triton 1.067 (0.76) |
| pj_m128_n28672_k4096 | rtx-3090 | 1.227 | cublas 1.808 (0.68)<br>lt 1.808 (0.68)<br>cublas-fresh 1.808 (0.68)<br>lt-fresh 1.808 (0.68) | torch-eager 2.125 (0.58) | torch-compile 1.535 (0.80) | torch-compile-cg 1.965 (0.62) | **torch-compile-ro 1.235 (0.99)** | triton 2.057 (0.60) |
| pj_m128_n28672_k4096 | a100 | 1.727 | lt 1.781 (0.97)<br>cublas 1.782 (0.97)<br>cublas-fresh 1.782 (0.97)<br>lt-fresh 1.782 (0.97) | torch-eager 1.788 (0.97) | torch-compile 2.352 (0.73) | torch-compile-cg 1.821 (0.95) | **torch-compile-ro 1.747 (0.99)** | triton 1.887 (0.92) |
| pj_m128_n28672_k4096 | rtx-4090 | 0.6884 | **lt-fresh 0.6861 (1.00)**<br>lt 0.6922 (0.99)<br>cublas-fresh 0.7137 (0.96)<br>cublas 0.7168 (0.96) | torch-eager 0.7025 (0.98) | torch-compile 0.7014 (0.98) | torch-compile-cg 0.6994 (0.98) | torch-compile-ro 0.6994 (0.98) | triton 1.144 (0.60) |
| pj_m128_n28672_k4096 | h100 | 0.7008 | **lt 0.6772 (1.03)**<br>cublas 0.6787 (1.03)<br>cublas-fresh 0.6787 (1.03)<br>lt-fresh 0.6806 (1.03) | torch-eager 0.6857 (1.02) | torch-compile 0.6819 (1.03) | torch-compile-cg 0.6832 (1.03) | torch-compile-ro 0.6838 (1.02) | triton 0.8065 (0.87) |
| pj_m128_n4096_k14336 | rtx-5080 | 0.4547 | **cublas-fresh 0.4014 (1.13)**<br>lt-fresh 0.4014 (1.13)<br>cublas 0.4055 (1.12)<br>lt 0.4055 (1.12) | torch-eager 0.4117 (1.10) | torch-compile 0.4117 (1.10) | torch-compile-cg 0.4117 (1.10) | torch-compile-ro 0.4117 (1.10) | triton 0.5898 (0.77) |
| pj_m128_n4096_k14336 | rtx-3090 | 0.7108 | lt 0.8458 (0.84)<br>lt-fresh 0.8458 (0.84)<br>cublas 0.8509 (0.84)<br>cublas-fresh 0.852 (0.83) | torch-eager 0.7178 (0.99) | torch-compile 0.6574 (1.08) | **torch-compile-cg 0.6564 (1.08)** | torch-compile-ro 0.6605 (1.08) | triton 1.152 (0.62) |
| pj_m128_n4096_k14336 | a100 | 0.9177 | **cublas-fresh 0.8509 (1.08)**<br>lt-fresh 0.8509 (1.08)<br>lt 0.854 (1.07)<br>cublas 0.855 (1.07) | torch-eager 0.854 (1.07) | torch-compile 1.101 (0.83) | torch-compile-cg 1.102 (0.83) | torch-compile-ro 1.102 (0.83) | triton 1.441 (0.64) |
| pj_m128_n4096_k14336 | rtx-4090 | 0.3975 | **cublas-fresh 0.3666 (1.08)**<br>lt-fresh 0.3697 (1.08)<br>lt 0.3749 (1.06)<br>cublas 0.3768 (1.05) | torch-eager 0.3799 (1.05) | torch-compile 0.3799 (1.05) | torch-compile-cg 0.3779 (1.05) | torch-compile-ro 0.3789 (1.05) | triton 0.4823 (0.82) |
| pj_m128_n4096_k14336 | h100 | 0.3695 | **lt 0.3123 (1.18)**<br>cublas 0.3133 (1.18)<br>lt-fresh 0.3134 (1.18)<br>cublas-fresh 0.3135 (1.18) | torch-eager 0.3131 (1.18) | torch-compile 0.3136 (1.18) | torch-compile-cg 0.3132 (1.18) | torch-compile-ro 0.3131 (1.18) | triton 0.3964 (0.93) |
| pj_m128_n4096_k4096 | rtx-5080 | 0.1456 | **cublas-fresh 0.1352 (1.08)**<br>lt-fresh 0.1352 (1.08)<br>cublas 0.1434 (1.02)<br>lt 0.1434 (1.02) | torch-eager 0.1454 (1.00) | torch-compile 0.1454 (1.00) | torch-compile-cg 0.1454 (1.00) | torch-compile-ro 0.1454 (1.00) | triton 0.1761 (0.83) |
| pj_m128_n4096_k4096 | rtx-3090 | 0.2442 | lt 0.2386 (1.02)<br>lt-fresh 0.2386 (1.02)<br>cublas 0.2632 (0.93)<br>cublas-fresh 0.2632 (0.93) | torch-eager 0.2202 (1.11) | torch-compile 0.2212 (1.10) | **torch-compile-cg 0.2191 (1.11)** | torch-compile-ro 0.2202 (1.11) | triton 0.3154 (0.77) |
| pj_m128_n4096_k4096 | a100 | 0.3046 | **cublas 0.2601 (1.17)**<br>cublas-fresh 0.2601 (1.17)<br>lt 0.2611 (1.17)<br>lt-fresh 0.2611 (1.17) | torch-eager 0.3328 (0.92) | torch-compile 0.3328 (0.92) | torch-compile-cg 0.3318 (0.92) | torch-compile-ro 0.3318 (0.92) | triton 0.4178 (0.73) |
| pj_m128_n4096_k4096 | rtx-4090 | 0.1467 | **lt-fresh 0.1065 (1.38)**<br>cublas-fresh 0.1116 (1.31)<br>lt 0.1249 (1.17)<br>cublas 0.1423 (1.03) | torch-eager 0.1413 (1.04) | torch-compile 0.126 (1.16) | torch-compile-cg 0.126 (1.16) | torch-compile-ro 0.1413 (1.04) | triton 0.1208 (1.21) |
| pj_m128_n4096_k4096 | h100 | 0.1428 | **lt 0.0982 (1.45)**<br>cublas 0.099 (1.44)<br>lt-fresh 0.099 (1.44)<br>cublas-fresh 0.0991 (1.44) | torch-eager 0.0997 (1.43) | torch-compile 0.1845 (0.77) | torch-compile-cg 0.0995 (1.43) | torch-compile-ro 0.0988 (1.44) | triton 0.1183 (1.21) |
| pj_m128_n6144_k4096 | rtx-5080 | 0.2095 | **cublas-fresh 0.1802 (1.16)**<br>lt-fresh 0.1802 (1.16)<br>cublas 0.1864 (1.12)<br>lt 0.1864 (1.12) | torch-eager 0.1884 (1.11) | torch-compile 0.1884 (1.11) | torch-compile-cg 0.1884 (1.11) | torch-compile-ro 0.1884 (1.11) | triton 0.2949 (0.71) |
| pj_m128_n6144_k4096 | rtx-3090 | 0.3396 | lt 0.3625 (0.94)<br>lt-fresh 0.3625 (0.94)<br>cublas 0.4004 (0.85)<br>cublas-fresh 0.4004 (0.85) | **torch-eager 0.3103 (1.09)** | torch-compile 0.3113 (1.09) | torch-compile-cg 0.3103 (1.09) | torch-compile-ro 0.3103 (1.09) | triton 0.4803 (0.71) |
| pj_m128_n6144_k4096 | a100 | 0.4298 | **lt 0.3738 (1.15)**<br>lt-fresh 0.3738 (1.15)<br>cublas 0.4557 (0.94)<br>cublas-fresh 0.4557 (0.94) | torch-eager 0.5816 (0.74) | torch-compile 0.5437 (0.79) | torch-compile-cg 0.5427 (0.79) | torch-compile-ro 0.5816 (0.74) | triton 0.4495 (0.96) |
| pj_m128_n6144_k4096 | rtx-4090 | 0.2068 | **cublas-fresh 0.1669 (1.24)**<br>lt-fresh 0.1679 (1.23)<br>lt 0.1761 (1.17)<br>cublas 0.1905 (1.09) | torch-eager 0.1896 (1.09) | torch-compile 0.1864 (1.11) | torch-compile-cg 0.1864 (1.11) | torch-compile-ro 0.1894 (1.09) | triton 0.1853 (1.12) |
| pj_m128_n6144_k4096 | h100 | 0.1975 | lt-fresh 0.1496 (1.32)<br>cublas 0.1503 (1.31)<br>lt 0.1503 (1.31)<br>cublas-fresh 0.1504 (1.31) | torch-eager 0.1502 (1.32) | torch-compile 0.1504 (1.31) | **torch-compile-cg 0.1494 (1.32)** | torch-compile-ro 0.1513 (1.31) | triton 0.2023 (0.98) |
| pj_m16_n18432_k7168 | rtx-5080 | 0.6025 | **cublas-fresh 0.5878 (1.02)**<br>lt-fresh 0.5878 (1.02)<br>cublas 0.6185 (0.97)<br>lt 0.6185 (0.97) | torch-eager 0.6185 (0.97) | torch-compile 0.6083 (0.99) | torch-compile-cg 0.6083 (0.99) | torch-compile-ro 0.6185 (0.97) | triton 0.6185 (0.97) |
| pj_m16_n18432_k7168 | rtx-3090 | 0.6458 | cublas-fresh 0.7485 (0.86)<br>lt-fresh 0.7485 (0.86)<br>cublas 0.7496 (0.86)<br>lt 0.7506 (0.86) | torch-eager 0.6748 (0.96) | **torch-compile 0.6113 (1.06)** | torch-compile-cg 0.6185 (1.04) | torch-compile-ro 0.6236 (1.04) | triton 0.8479 (0.76) |
| pj_m16_n18432_k7168 | a100 | 0.4434 | lt 0.5038 (0.88)<br>lt-fresh 0.5048 (0.88)<br>cublas 0.5284 (0.84)<br>cublas-fresh 0.5304 (0.84) | torch-eager 0.6615 (0.67) | **torch-compile 0.4454 (1.00)** | torch-compile-cg 0.469 (0.95) | torch-compile-ro 0.5253 (0.84) | triton 0.7301 (0.61) |
| pj_m16_n18432_k7168 | rtx-4090 | 0.6256 | **lt-fresh 0.5949 (1.05)**<br>lt 0.6042 (1.04)<br>cublas-fresh 0.6124 (1.02)<br>cublas 0.6186 (1.01) | torch-eager 0.6656 (0.94) | torch-compile 0.6011 (1.04) | torch-compile-cg 0.6001 (1.04) | torch-compile-ro 0.6226 (1.00) | triton 0.6849 (0.91) |
| pj_m16_n18432_k7168 | h100 | 0.2207 | lt 0.2385 (0.93)<br>lt-fresh 0.2402 (0.92)<br>cublas 0.2623 (0.84)<br>cublas-fresh 0.2636 (0.84) | torch-eager 0.2631 (0.84) | **torch-compile 0.2088 (1.06)** | torch-compile-cg 0.209 (1.06) | torch-compile-ro 0.2628 (0.84) | triton 0.3511 (0.63) |
| pj_m16_n2048_k7168 | rtx-5080 | 0.04612 | **cublas-fresh 0.0492 (0.94)**<br>lt-fresh 0.0492 (0.94)<br>lt 0.0798 (0.58)<br>cublas 0.0799 (0.58) | torch-eager 0.0799 (0.58) | torch-compile 0.0778 (0.59) | torch-compile-cg 0.0778 (0.59) | torch-compile-ro 0.0799 (0.58) | triton 0.1536 (0.30) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1145 | lt 0.086 (1.33)<br>lt-fresh 0.086 (1.33)<br>cublas 0.0891 (1.28)<br>cublas-fresh 0.0891 (1.28) | torch-eager 0.0901 (1.27) | torch-compile 0.0829 (1.38) | **torch-compile-cg 0.0819 (1.40)** | torch-compile-ro 0.0881 (1.30) | triton 0.2171 (0.53) |
| pj_m16_n2048_k7168 | a100 | 0.1036 | **cublas-fresh 0.087 (1.19)**<br>lt-fresh 0.087 (1.19)<br>cublas 0.0901 (1.15)<br>lt 0.0911 (1.14) | torch-eager 0.1096 (0.95) | torch-compile 0.1096 (0.95) | torch-compile-cg 0.1065 (0.97) | torch-compile-ro 0.1075 (0.96) | triton 0.3727 (0.28) |
| pj_m16_n2048_k7168 | rtx-4090 | 0.06373 | **cublas-fresh 0.0317 (2.01)**<br>lt-fresh 0.0317 (2.01)<br>lt 0.0932 (0.68)<br>cublas 0.0942 (0.68) | torch-eager 0.0942 (0.68) | torch-compile 0.0932 (0.68) | torch-compile-cg 0.0922 (0.69) | torch-compile-ro 0.0932 (0.68) | triton 0.1464 (0.44) |
| pj_m16_n2048_k7168 | h100 | 0.06898 | **lt-fresh 0.0481 (1.43)**<br>lt 0.0491 (1.40)<br>cublas-fresh 0.0504 (1.37)<br>cublas 0.0516 (1.34) | torch-eager 0.0518 (1.33) | torch-compile 0.0516 (1.34) | torch-compile-cg 0.0508 (1.36) | torch-compile-ro 0.0503 (1.37) | triton 0.1913 (0.36) |
| pj_m16_n28672_k4096 | rtx-5080 | 0.5395 | **lt-fresh 0.5243 (1.03)**<br>lt 0.5509 (0.98)<br>cublas-fresh 1.859 (0.29)<br>cublas 1.926 (0.28) | torch-eager 1.927 (0.28) | torch-compile 0.5427 (0.99) | torch-compile-cg 0.5427 (0.99) | torch-compile-ro 1.927 (0.28) | triton 0.5509 (0.98) |
| pj_m16_n28672_k4096 | rtx-3090 | 0.5905 | cublas 0.642 (0.92)<br>lt 0.6431 (0.92)<br>cublas-fresh 0.6431 (0.92)<br>lt-fresh 0.6431 (0.92) | torch-eager 0.5581 (1.06) | **torch-compile 0.5427 (1.09)** | torch-compile-cg 0.5458 (1.08) | torch-compile-ro 0.555 (1.06) | triton 0.6799 (0.87) |
| pj_m16_n28672_k4096 | a100 | 0.4179 | cublas 0.471 (0.89)<br>lt 0.471 (0.89)<br>cublas-fresh 0.4731 (0.88)<br>lt-fresh 0.4731 (0.88) | torch-eager 0.5857 (0.71) | **torch-compile 0.4076 (1.03)** | torch-compile-cg 0.4168 (1.00) | torch-compile-ro 0.5847 (0.71) | triton 0.6175 (0.68) |
| pj_m16_n28672_k4096 | rtx-4090 | 0.5661 | **cublas-fresh 0.5356 (1.06)**<br>lt-fresh 0.5356 (1.06)<br>cublas 0.5417 (1.04)<br>lt 0.5417 (1.04) | torch-eager 0.5458 (1.04) | torch-compile 0.5407 (1.05) | torch-compile-cg 0.5396 (1.05) | torch-compile-ro 0.5448 (1.04) | triton 0.555 (1.02) |
| pj_m16_n28672_k4096 | h100 | 0.2053 | lt 0.218 (0.94)<br>lt-fresh 0.2197 (0.93)<br>cublas 0.2399 (0.86)<br>cublas-fresh 0.2409 (0.85) | torch-eager 0.2409 (0.85) | torch-compile 0.176 (1.17) | **torch-compile-cg 0.1757 (1.17)** | torch-compile-ro 0.2415 (0.85) | triton 0.219 (0.94) |
| pj_m16_n4096_k14336 | rtx-5080 | 0.2667 | cublas-fresh 0.3174 (0.84)<br>lt-fresh 0.3174 (0.84)<br>cublas 0.3318 (0.80)<br>lt 0.3318 (0.80) | torch-eager 0.3338 (0.80) | **torch-compile 0.2744 (0.97)** | torch-compile-cg 0.2744 (0.97) | torch-compile-ro 0.3338 (0.80) | triton 0.3052 (0.87) |
| pj_m16_n4096_k14336 | rtx-3090 | 0.306 | lt 0.3 (1.02)<br>lt-fresh 0.3 (1.02)<br>cublas-fresh 0.3011 (1.02)<br>cublas 0.3021 (1.01) | torch-eager 0.2949 (1.04) | **torch-compile 0.2755 (1.11)** | torch-compile-cg 0.2755 (1.11) | torch-compile-ro 0.2929 (1.04) | triton 0.4854 (0.63) |
| pj_m16_n4096_k14336 | a100 | 0.2347 | **lt 0.2509 (0.94)**<br>lt-fresh 0.2509 (0.94)<br>cublas-fresh 0.2529 (0.93)<br>cublas 0.254 (0.92) | torch-eager 0.257 (0.91) | torch-compile 0.3164 (0.74) | torch-compile-cg 0.3133 (0.75) | torch-compile-ro 0.3133 (0.75) | triton 0.7393 (0.32) |
| pj_m16_n4096_k14336 | rtx-4090 | 0.3183 | **cublas-fresh 0.2939 (1.08)**<br>lt-fresh 0.2959 (1.08)<br>lt 0.3031 (1.05)<br>cublas 0.3041 (1.05) | torch-eager 0.3092 (1.03) | torch-compile 0.3021 (1.05) | torch-compile-cg 0.3011 (1.06) | torch-compile-ro 0.3072 (1.04) | triton 0.3204 (0.99) |
| pj_m16_n4096_k14336 | h100 | 0.1319 | **lt 0.1164 (1.13)**<br>lt-fresh 0.1173 (1.12)<br>cublas-fresh 0.1342 (0.98)<br>cublas 0.1343 (0.98) | torch-eager 0.1344 (0.98) | torch-compile 0.1333 (0.99) | torch-compile-cg 0.1352 (0.98) | torch-compile-ro 0.1352 (0.98) | triton 0.3793 (0.35) |
| pj_m16_n4096_k4096 | rtx-5080 | 0.05224 | **cublas-fresh 0.0553 (0.94)**<br>lt-fresh 0.0553 (0.94)<br>lt 0.0942 (0.55)<br>cublas 0.0963 (0.54) | torch-eager 0.0963 (0.54) | torch-compile 0.084 (0.62) | torch-compile-cg 0.0819 (0.64) | torch-compile-ro 0.0983 (0.53) | triton 0.0922 (0.57) |
| pj_m16_n4096_k4096 | rtx-3090 | 0.131 | cublas 0.1044 (1.25)<br>lt 0.1044 (1.25)<br>cublas-fresh 0.1055 (1.24)<br>lt-fresh 0.1055 (1.24) | torch-eager 0.1024 (1.28) | **torch-compile 0.086 (1.52)** | torch-compile-cg 0.086 (1.52) | torch-compile-ro 0.1014 (1.29) | triton 0.1403 (0.93) |
| pj_m16_n4096_k4096 | a100 | 0.0994 | **cublas-fresh 0.0891 (1.12)**<br>lt-fresh 0.0891 (1.12)<br>cublas 0.0922 (1.08)<br>lt 0.0922 (1.08) | torch-eager 0.1075 (0.92) | torch-compile 0.1075 (0.92) | torch-compile-cg 0.1055 (0.94) | torch-compile-ro 0.1055 (0.94) | triton 0.2171 (0.46) |
| pj_m16_n4096_k4096 | rtx-4090 | 0.06558 | **lt-fresh 0.0358 (1.83)**<br>cublas-fresh 0.0389 (1.69)<br>lt 0.1055 (0.62)<br>cublas 0.1126 (0.58) | torch-eager 0.1126 (0.58) | torch-compile 0.1014 (0.65) | torch-compile-cg 0.1004 (0.65) | torch-compile-ro 0.1106 (0.59) | triton 0.1024 (0.64) |
| pj_m16_n4096_k4096 | h100 | 0.06435 | **lt 0.0477 (1.35)**<br>lt-fresh 0.0478 (1.35)<br>cublas-fresh 0.0905 (0.71)<br>cublas 0.0908 (0.71) | torch-eager 0.0909 (0.71) | torch-compile 0.0509 (1.26) | torch-compile-cg 0.0513 (1.25) | torch-compile-ro 0.0911 (0.71) | triton 0.1129 (0.57) |
| pj_m16_n6144_k4096 | rtx-5080 | 0.1009 | **lt-fresh 0.0881 (1.14)**<br>cublas-fresh 0.0963 (1.05)<br>lt 0.129 (0.78)<br>cublas 0.1413 (0.71) | torch-eager 0.1393 (0.72) | torch-compile 0.1208 (0.83) | torch-compile-cg 0.1208 (0.83) | torch-compile-ro 0.1413 (0.71) | triton 0.129 (0.78) |
| pj_m16_n6144_k4096 | rtx-3090 | 0.1629 | lt 0.1352 (1.20)<br>lt-fresh 0.1352 (1.20)<br>cublas 0.1454 (1.12)<br>cublas-fresh 0.1454 (1.12) | torch-eager 0.1464 (1.11) | torch-compile 0.1239 (1.31) | **torch-compile-cg 0.1229 (1.33)** | torch-compile-ro 0.1444 (1.13) | triton 0.1597 (1.02) |
| pj_m16_n6144_k4096 | a100 | 0.1293 | lt-fresh 0.1208 (1.07)<br>lt 0.1229 (1.05)<br>cublas-fresh 0.1341 (0.96)<br>cublas 0.1362 (0.95) | torch-eager 0.1475 (0.88) | torch-compile 0.1167 (1.11) | **torch-compile-cg 0.1157 (1.12)** | torch-compile-ro 0.1454 (0.89) | triton 0.2181 (0.59) |
| pj_m16_n6144_k4096 | rtx-4090 | 0.1276 | **cublas-fresh 0.1004 (1.27)**<br>lt-fresh 0.1065 (1.20)<br>lt 0.1536 (0.83)<br>cublas 0.1565 (0.82) | torch-eager 0.1587 (0.80) | torch-compile 0.1485 (0.86) | torch-compile-cg 0.1475 (0.86) | torch-compile-ro 0.1567 (0.81) | triton 0.1516 (0.84) |
| pj_m16_n6144_k4096 | h100 | 0.08028 | lt 0.0668 (1.20)<br>lt-fresh 0.0668 (1.20)<br>cublas 0.095 (0.85)<br>cublas-fresh 0.0951 (0.84) | torch-eager 0.0943 (0.85) | **torch-compile 0.0572 (1.40)** | torch-compile-cg 0.0575 (1.40) | torch-compile-ro 0.0961 (0.84) | triton 0.1139 (0.70) |
| pj_m1_n18432_k7168 | rtx-5080 | 0.6051 | cublas-fresh 0.6103 (0.99)<br>lt-fresh 0.6124 (0.99)<br>cublas 0.6185 (0.98)<br>lt 0.6185 (0.98) | torch-eager 0.6164 (0.98) | **torch-compile 0.6083 (0.99)** | torch-compile-cg 0.6083 (0.99) | torch-compile-ro 0.6185 (0.98) | triton 0.6369 (0.95) |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6233 | **cublas 0.6072 (1.03)**<br>cublas-fresh 0.6072 (1.03)<br>lt 0.6103 (1.02)<br>lt-fresh 0.6113 (1.02) | torch-eager 0.6646 (0.94) | torch-compile 0.6083 (1.02) | torch-compile-cg 0.6083 (1.02) | torch-compile-ro 0.6185 (1.01) | triton 0.8407 (0.74) |
| pj_m1_n18432_k7168 | a100 | 0.4309 | **lt-fresh 0.3912 (1.10)**<br>lt 0.3973 (1.08)<br>cublas-fresh 0.4014 (1.07)<br>cublas 0.4209 (1.02) | torch-eager 0.4444 (0.97) | torch-compile 0.3994 (1.08) | torch-compile-cg 0.3994 (1.08) | torch-compile-ro 0.4332 (0.99) | triton 0.7588 (0.57) |
| pj_m1_n18432_k7168 | rtx-4090 | 0.6203 | **lt-fresh 0.5222 (1.19)**<br>lt 0.5774 (1.07)<br>cublas-fresh 0.6011 (1.03)<br>cublas 0.6062 (1.02) | torch-eager 0.6604 (0.94) | torch-compile 0.5612 (1.11) | torch-compile-cg 0.5601 (1.11) | torch-compile-ro 0.6083 (1.02) | triton 0.6881 (0.90) |
| pj_m1_n18432_k7168 | h100 | 0.2183 | **lt-fresh 0.1916 (1.14)**<br>cublas-fresh 0.1962 (1.11)<br>lt 0.1985 (1.10)<br>cublas 0.2037 (1.07) | torch-eager 0.2047 (1.07) | torch-compile 0.1965 (1.11) | torch-compile-cg 0.1966 (1.11) | torch-compile-ro 0.2045 (1.07) | triton 0.3526 (0.62) |
| pj_m1_n2048_k7168 | rtx-5080 | 0.03351 | **cublas-fresh 0.0266 (1.26)**<br>lt-fresh 0.0266 (1.26)<br>cublas 0.0758 (0.44)<br>lt 0.0758 (0.44) | torch-eager 0.0758 (0.44) | torch-compile 0.0758 (0.44) | torch-compile-cg 0.0758 (0.44) | torch-compile-ro 0.0758 (0.44) | triton 0.2028 (0.17) |
| pj_m1_n2048_k7168 | rtx-3090 | 0.09693 | **lt-fresh 0.0748 (1.30)**<br>lt 0.0778 (1.25)<br>cublas-fresh 0.0901 (1.08)<br>cublas 0.0911 (1.06) | torch-eager 0.0901 (1.08) | torch-compile 0.0809 (1.20) | torch-compile-cg 0.0819 (1.18) | torch-compile-ro 0.0901 (1.08) | triton 0.2324 (0.42) |
| pj_m1_n2048_k7168 | a100 | 0.08791 | **cublas-fresh 0.044 (2.00)**<br>lt-fresh 0.0451 (1.95)<br>cublas 0.0553 (1.59)<br>lt 0.0553 (1.59) | torch-eager 0.0584 (1.51) | torch-compile 0.0614 (1.43) | torch-compile-cg 0.0594 (1.48) | torch-compile-ro 0.0543 (1.62) | triton 0.3727 (0.24) |
| pj_m1_n2048_k7168 | rtx-4090 | 0.03918 | **lt-fresh 0.0276 (1.42)**<br>cublas-fresh 0.0451 (0.87)<br>lt 0.0809 (0.48)<br>cublas 0.1034 (0.38) | torch-eager 0.1024 (0.38) | torch-compile 0.0879 (0.45) | torch-compile-cg 0.086 (0.46) | torch-compile-ro 0.1004 (0.39) | triton 0.1484 (0.26) |
| pj_m1_n2048_k7168 | h100 | 0.05008 | **cublas-fresh 0.0243 (2.06)**<br>lt-fresh 0.025 (2.00)<br>lt 0.0305 (1.64)<br>cublas 0.0329 (1.52) | torch-eager 0.0326 (1.54) | torch-compile 0.0332 (1.51) | torch-compile-cg 0.0315 (1.59) | torch-compile-ro 0.0329 (1.52) | triton 0.1895 (0.26) |
| pj_m1_n28672_k4096 | rtx-5080 | 0.5624 | **lt-fresh 0.5038 (1.12)**<br>cublas-fresh 0.5202 (1.08)<br>lt 0.5407 (1.04)<br>cublas 0.5591 (1.01) | torch-eager 0.5591 (1.01) | torch-compile 0.5386 (1.04) | torch-compile-cg 0.5386 (1.04) | torch-compile-ro 0.5591 (1.01) | triton 0.5693 (0.99) |
| pj_m1_n28672_k4096 | rtx-3090 | 0.5658 | cublas 0.5396 (1.05)<br>lt 0.5396 (1.05)<br>lt-fresh 0.5396 (1.05)<br>cublas-fresh 0.5407 (1.05) | torch-eager 0.5386 (1.05) | torch-compile 0.5417 (1.04) | torch-compile-cg 0.5417 (1.04) | **torch-compile-ro 0.5366 (1.05)** | triton 0.8325 (0.68) |
| pj_m1_n28672_k4096 | a100 | 0.3794 | lt-fresh 0.3594 (1.06)<br>lt 0.3697 (1.03)<br>cublas 0.4485 (0.85)<br>cublas-fresh 0.4495 (0.84) | torch-eager 0.51 (0.74) | torch-compile 0.3492 (1.09) | **torch-compile-cg 0.3482 (1.09)** | torch-compile-ro 0.4905 (0.77) | triton 0.6369 (0.60) |
| pj_m1_n28672_k4096 | rtx-4090 | 0.5613 | **lt-fresh 0.468 (1.20)**<br>lt 0.5183 (1.08)<br>cublas-fresh 0.5448 (1.03)<br>cublas 0.5519 (1.02) | torch-eager 0.5519 (1.02) | torch-compile 0.5018 (1.12) | torch-compile-cg 0.4998 (1.12) | torch-compile-ro 0.5509 (1.02) | triton 0.5622 (1.00) |
| pj_m1_n28672_k4096 | h100 | 0.1913 | **lt-fresh 0.169 (1.13)**<br>cublas-fresh 0.1698 (1.13)<br>lt 0.1772 (1.08)<br>cublas 0.1778 (1.08) | torch-eager 0.1782 (1.07) | torch-compile 0.1722 (1.11) | torch-compile-cg 0.1725 (1.11) | torch-compile-ro 0.1793 (1.07) | triton 0.2542 (0.75) |
| pj_m1_n4096_k14336 | rtx-5080 | 0.2547 | **cublas-fresh 0.2314 (1.10)**<br>lt-fresh 0.2314 (1.10)<br>cublas 0.2744 (0.93)<br>lt 0.2744 (0.93) | torch-eager 0.2744 (0.93) | torch-compile 0.2765 (0.92) | torch-compile-cg 0.2765 (0.92) | torch-compile-ro 0.2765 (0.92) | triton 0.4096 (0.62) |
| pj_m1_n4096_k14336 | rtx-3090 | 0.3006 | lt 0.2775 (1.08)<br>cublas 0.2785 (1.08)<br>lt-fresh 0.2785 (1.08)<br>cublas-fresh 0.2796 (1.07) | torch-eager 0.2775 (1.08) | torch-compile 0.2836 (1.06) | torch-compile-cg 0.2806 (1.07) | **torch-compile-ro 0.2755 (1.09)** | triton 0.5458 (0.55) |
| pj_m1_n4096_k14336 | a100 | 0.2009 | **lt-fresh 0.1772 (1.13)**<br>cublas-fresh 0.1782 (1.13)<br>lt 0.1843 (1.09)<br>cublas 0.1935 (1.04) | torch-eager 0.1997 (1.01) | torch-compile 0.1843 (1.09) | torch-compile-cg 0.1823 (1.10) | torch-compile-ro 0.1915 (1.05) | triton 0.7393 (0.27) |
| pj_m1_n4096_k14336 | rtx-4090 | 0.3031 | **lt-fresh 0.1997 (1.52)**<br>lt 0.2601 (1.17)<br>cublas-fresh 0.2908 (1.04)<br>cublas 0.3061 (0.99) | torch-eager 0.3103 (0.98) | torch-compile 0.2693 (1.13) | torch-compile-cg 0.2652 (1.14) | torch-compile-ro 0.3092 (0.98) | triton 0.3246 (0.93) |
| pj_m1_n4096_k14336 | h100 | 0.115 | **cublas-fresh 0.0847 (1.36)**<br>lt-fresh 0.0851 (1.35)<br>cublas 0.0933 (1.23)<br>lt 0.0936 (1.23) | torch-eager 0.0934 (1.23) | torch-compile 0.0939 (1.23) | torch-compile-cg 0.0928 (1.24) | torch-compile-ro 0.094 (1.22) | triton 0.373 (0.31) |
| pj_m1_n4096_k4096 | rtx-5080 | 0.03263 | **lt-fresh 0.0205 (1.59)**<br>cublas-fresh 0.0246 (1.33)<br>lt 0.084 (0.39)<br>cublas 0.0881 (0.37) | torch-eager 0.0881 (0.37) | torch-compile 0.086 (0.38) | torch-compile-cg 0.0881 (0.37) | torch-compile-ro 0.0881 (0.37) | triton 0.127 (0.26) |
| pj_m1_n4096_k4096 | rtx-3090 | 0.1105 | **lt 0.086 (1.29)**<br>lt-fresh 0.087 (1.27)<br>cublas-fresh 0.0901 (1.23)<br>cublas 0.0922 (1.20) | torch-eager 0.0922 (1.20) | torch-compile 0.0932 (1.19) | torch-compile-cg 0.0932 (1.19) | torch-compile-ro 0.0922 (1.20) | triton 0.1905 (0.58) |
| pj_m1_n4096_k4096 | a100 | 0.08984 | **cublas-fresh 0.0532 (1.69)**<br>lt-fresh 0.0532 (1.69)<br>cublas 0.0676 (1.33)<br>lt 0.0676 (1.33) | torch-eager 0.0686 (1.31) | torch-compile 0.0655 (1.37) | torch-compile-cg 0.0635 (1.41) | torch-compile-ro 0.0655 (1.37) | triton 0.2181 (0.41) |
| pj_m1_n4096_k4096 | rtx-4090 | 0.04339 | **lt-fresh 0.0205 (2.12)**<br>cublas-fresh 0.0225 (1.93)<br>lt 0.0922 (0.47)<br>cublas 0.0952 (0.46) | torch-eager 0.0963 (0.45) | torch-compile 0.0963 (0.45) | torch-compile-cg 0.0952 (0.46) | torch-compile-ro 0.0952 (0.46) | triton 0.1075 (0.40) |
| pj_m1_n4096_k4096 | h100 | 0.05306 | **lt-fresh 0.0239 (2.22)**<br>cublas-fresh 0.0241 (2.20)<br>lt 0.032 (1.66)<br>cublas 0.0321 (1.65) | torch-eager 0.0321 (1.65) | torch-compile 0.0384 (1.38) | torch-compile-cg 0.0386 (1.37) | torch-compile-ro 0.0325 (1.63) | triton 0.1117 (0.48) |
| pj_m1_n6144_k4096 | rtx-5080 | 0.09199 | **cublas-fresh 0.0512 (1.80)**<br>lt-fresh 0.0512 (1.80)<br>lt 0.1229 (0.75)<br>cublas 0.1249 (0.74) | torch-eager 0.1249 (0.74) | torch-compile 0.1208 (0.76) | torch-compile-cg 0.1229 (0.75) | torch-compile-ro 0.1229 (0.75) | triton 0.1434 (0.64) |
| pj_m1_n6144_k4096 | rtx-3090 | 0.1485 | lt 0.1239 (1.20)<br>lt-fresh 0.1239 (1.20)<br>cublas 0.1403 (1.06)<br>cublas-fresh 0.1403 (1.06) | torch-eager 0.1393 (1.07) | torch-compile 0.1229 (1.21) | **torch-compile-cg 0.1219 (1.22)** | torch-compile-ro 0.1372 (1.08) | triton 0.2447 (0.61) |
| pj_m1_n6144_k4096 | a100 | 0.1154 | **lt-fresh 0.0778 (1.48)**<br>cublas-fresh 0.0788 (1.46)<br>lt 0.0901 (1.28)<br>cublas 0.0922 (1.25) | torch-eager 0.0983 (1.17) | torch-compile 0.0911 (1.27) | torch-compile-cg 0.0891 (1.29) | torch-compile-ro 0.0911 (1.27) | triton 0.2202 (0.52) |
| pj_m1_n6144_k4096 | rtx-4090 | 0.08952 | **lt-fresh 0.0564 (1.59)**<br>cublas-fresh 0.1147 (0.78)<br>lt 0.1239 (0.72)<br>cublas 0.1567 (0.57) | torch-eager 0.1546 (0.58) | torch-compile 0.1303 (0.69) | torch-compile-cg 0.129 (0.69) | torch-compile-ro 0.1536 (0.58) | triton 0.1608 (0.56) |
| pj_m1_n6144_k4096 | h100 | 0.06406 | **cublas-fresh 0.0372 (1.72)**<br>lt-fresh 0.0372 (1.72)<br>cublas 0.0443 (1.45)<br>lt 0.0445 (1.44) | torch-eager 0.0441 (1.45) | torch-compile 0.0491 (1.30) | torch-compile-cg 0.0471 (1.36) | torch-compile-ro 0.0447 (1.43) | triton 0.1127 (0.57) |
| pj_m2048_n28672_k4096 | rtx-5080 | 12.63 | **cublas-fresh 12.21 (1.03)**<br>cublas 12.21 (1.03)<br>lt-fresh 12.21 (1.03)<br>lt 12.22 (1.03) | torch-eager 12.45 (1.01) | torch-compile 12.42 (1.02) | torch-compile-cg 12.46 (1.01) | torch-compile-ro 12.46 (1.01) | triton 15.8 (0.80) |
| pj_m2048_n28672_k4096 | rtx-3090 | 20.74 | lt-fresh 23.04 (0.90)<br>cublas-fresh 23.05 (0.90)<br>lt 23.05 (0.90)<br>cublas 23.05 (0.90) | torch-eager 22.51 (0.92) | torch-compile 22.51 (0.92) | torch-compile-cg 22.73 (0.91) | **torch-compile-ro 22.48 (0.92)** | triton 29.33 (0.71) |
| pj_m2048_n28672_k4096 | a100 | 26.87 | cublas 26.24 (1.02)<br>cublas-fresh 26.24 (1.02)<br>lt-fresh 26.39 (1.02)<br>lt 26.4 (1.02) | torch-eager 26.69 (1.01) | **torch-compile 26.02 (1.03)** | torch-compile-cg 26.02 (1.03) | torch-compile-ro 26.71 (1.01) | triton 26.18 (1.03) |
| pj_m2048_n28672_k4096 | rtx-4090 | 9.519 | **lt 11.06 (0.86)**<br>cublas 11.06 (0.86)<br>cublas-fresh 11.09 (0.86)<br>lt-fresh 11.23 (0.85) | torch-eager 11.34 (0.84) | torch-compile 13.37 (0.71) | torch-compile-cg 13.51 (0.70) | torch-compile-ro 11.44 (0.83) | triton 13.17 (0.72) |
| pj_m2048_n28672_k4096 | h100 | 10.12 | lt-fresh 9.264 (1.09)<br>cublas-fresh 9.265 (1.09)<br>lt 9.267 (1.09)<br>cublas 9.267 (1.09) | torch-eager 9.263 (1.09) | **torch-compile 9.235 (1.10)** | torch-compile-cg 9.265 (1.09) | torch-compile-ro 9.268 (1.09) | triton 10.21 (0.99) |
| pj_m2048_n4096_k14336 | rtx-5080 | 6.427 | cublas-fresh 9.066 (0.71)<br>cublas 9.475 (0.68)<br>lt 9.48 (0.68)<br>lt-fresh 9.503 (0.68) | **torch-eager 6.498 (0.99)** | torch-compile 6.515 (0.99) | torch-compile-cg 6.499 (0.99) | torch-compile-ro 6.508 (0.99) | triton 8.21 (0.78) |
| pj_m2048_n4096_k14336 | rtx-3090 | 11.05 | **lt 11.75 (0.94)**<br>lt-fresh 11.75 (0.94)<br>cublas 12.54 (0.88)<br>cublas-fresh 12.54 (0.88) | torch-eager 12.47 (0.89) | torch-compile 12.32 (0.90) | torch-compile-cg 12.23 (0.90) | torch-compile-ro 12.23 (0.90) | triton 15.16 (0.73) |
| pj_m2048_n4096_k14336 | a100 | 13.34 | cublas 13.08 (1.02)<br>lt 13.08 (1.02)<br>cublas-fresh 13.08 (1.02)<br>lt-fresh 13.09 (1.02) | **torch-eager 12.9 (1.03)** | torch-compile 13.62 (0.98) | torch-compile-cg 13.63 (0.98) | torch-compile-ro 13.05 (1.02) | triton 13.73 (0.97) |
| pj_m2048_n4096_k14336 | rtx-4090 | 4.41 | lt 6.365 (0.69)<br>cublas-fresh 6.365 (0.69)<br>lt-fresh 6.365 (0.69)<br>cublas 6.366 (0.69) | torch-eager 5.735 (0.77) | torch-compile 5.735 (0.77) | **torch-compile-cg 5.721 (0.77)** | torch-compile-ro 5.784 (0.76) | triton 7.024 (0.63) |
| pj_m2048_n4096_k14336 | h100 | 5.268 | **cublas 4.623 (1.14)**<br>lt-fresh 4.623 (1.14)<br>lt 4.623 (1.14)<br>cublas-fresh 4.623 (1.14) | torch-eager 4.656 (1.13) | torch-compile 4.704 (1.12) | torch-compile-cg 4.627 (1.14) | torch-compile-ro 4.628 (1.14) | triton 5.296 (0.99) |
| pj_m2048_n4096_k4096 | rtx-5080 | 1.837 | lt-fresh 1.785 (1.03)<br>cublas-fresh 1.787 (1.03)<br>cublas 1.792 (1.03)<br>lt 1.792 (1.03) | torch-eager 1.776 (1.03) | torch-compile 1.776 (1.03) | **torch-compile-cg 1.772 (1.04)** | torch-compile-ro 1.776 (1.03) | triton 2.335 (0.79) |
| pj_m2048_n4096_k4096 | rtx-3090 | 3.125 | **lt 3.393 (0.92)**<br>cublas-fresh 3.393 (0.92)<br>lt-fresh 3.393 (0.92)<br>cublas 3.393 (0.92) | torch-eager 3.441 (0.91) | torch-compile 3.483 (0.90) | torch-compile-cg 3.395 (0.92) | torch-compile-ro 3.421 (0.91) | triton 4.2 (0.74) |
| pj_m2048_n4096_k4096 | a100 | 3.868 | cublas 3.74 (1.03)<br>lt 3.742 (1.03)<br>cublas-fresh 3.742 (1.03)<br>lt-fresh 3.742 (1.03) | torch-eager 3.791 (1.02) | torch-compile 4.813 (0.80) | **torch-compile-cg 3.704 (1.04)** | torch-compile-ro 3.705 (1.04) | triton 3.93 (0.98) |
| pj_m2048_n4096_k4096 | rtx-4090 | 1.289 | cublas 1.469 (0.88)<br>lt 1.47 (0.88)<br>cublas-fresh 1.472 (0.88)<br>lt-fresh 1.472 (0.88) | torch-eager 1.371 (0.94) | **torch-compile 1.2 (1.07)** | torch-compile-cg 1.204 (1.07) | torch-compile-ro 1.21 (1.07) | triton 2.154 (0.60) |
| pj_m2048_n4096_k4096 | h100 | 1.514 | **cublas-fresh 1.328 (1.14)**<br>lt-fresh 1.328 (1.14)<br>cublas 1.331 (1.14)<br>lt 1.331 (1.14) | torch-eager 1.337 (1.13) | torch-compile 1.328 (1.14) | torch-compile-cg 1.329 (1.14) | torch-compile-ro 1.328 (1.14) | triton 1.522 (0.99) |
| pj_m2048_n6144_k4096 | rtx-5080 | 2.648 | cublas-fresh 2.634 (1.01)<br>lt-fresh 2.634 (1.01)<br>cublas 2.64 (1.00)<br>lt 2.642 (1.00) | torch-eager 2.605 (1.02) | torch-compile 2.606 (1.02) | **torch-compile-cg 2.602 (1.02)** | torch-compile-ro 2.608 (1.02) | triton 3.42 (0.77) |
| pj_m2048_n6144_k4096 | rtx-3090 | 4.56 | lt 4.908 (0.93)<br>cublas-fresh 4.908 (0.93)<br>lt-fresh 4.908 (0.93)<br>cublas 4.91 (0.93) | **torch-eager 4.906 (0.93)** | torch-compile 4.932 (0.92) | torch-compile-cg 4.978 (0.92) | torch-compile-ro 4.91 (0.93) | triton 6.205 (0.73) |
| pj_m2048_n6144_k4096 | a100 | 5.806 | lt-fresh 5.839 (0.99)<br>lt 5.84 (0.99)<br>cublas 5.886 (0.99)<br>cublas-fresh 5.886 (0.99) | torch-eager 6.145 (0.94) | torch-compile 5.787 (1.00) | **torch-compile-cg 5.784 (1.00)** | torch-compile-ro 6.228 (0.93) | triton 5.87 (0.99) |
| pj_m2048_n6144_k4096 | rtx-4090 | 1.93 | cublas 2.312 (0.83)<br>lt-fresh 2.312 (0.83)<br>lt 2.313 (0.83)<br>cublas-fresh 2.313 (0.83) | torch-eager 2.422 (0.80) | torch-compile 2.384 (0.81) | torch-compile-cg 2.402 (0.80) | **torch-compile-ro 1.827 (1.06)** | triton 3.055 (0.63) |
| pj_m2048_n6144_k4096 | h100 | 2.21 | lt-fresh 1.989 (1.11)<br>cublas-fresh 1.99 (1.11)<br>cublas 1.992 (1.11)<br>lt 1.992 (1.11) | **torch-eager 1.977 (1.12)** | torch-compile 1.978 (1.12) | torch-compile-cg 1.979 (1.12) | torch-compile-ro 1.978 (1.12) | triton 2.275 (0.97) |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.6985 | **cublas-tf32-fresh 0.5898 (1.18)**<br>lt-tf32-fresh 0.5898 (1.18)<br>cublas-tf32 0.6226 (1.12)<br>lt-tf32 0.6226 (1.12) | torch-eager-tf32 0.6472 (1.08) | torch-compile-tf32 0.6267 (1.11) | torch-compile-cg-tf32 0.6287 (1.11) | torch-compile-ro-tf32 0.6451 (1.08) | triton-tf32 0.6267 (1.11) |
| pj_m128_n28672_k4096_tf32 | rtx-3090 | 1.044 | cublas-tf32 1.096 (0.95)<br>lt-tf32 1.096 (0.95)<br>cublas-tf32-fresh 1.096 (0.95)<br>lt-tf32-fresh 1.096 (0.95) | torch-eager-tf32 1.017 (1.03) | torch-compile-tf32 0.9339 (1.12) | torch-compile-cg-tf32 0.9318 (1.12) | **torch-compile-ro-tf32 0.8499 (1.23)** | triton-tf32 1.151 (0.91) |
| pj_m128_n28672_k4096_tf32 | a100 | 0.5405 | lt-tf32 0.4516 (1.20)<br>lt-tf32-fresh 0.4516 (1.20)<br>cublas-tf32-fresh 0.4628 (1.17)<br>cublas-tf32 0.4649 (1.16) | torch-eager-tf32 0.5253 (1.03) | torch-compile-tf32 0.5038 (1.07) | torch-compile-cg-tf32 0.5243 (1.03) | torch-compile-ro-tf32 0.5243 (1.03) | **triton-tf32 0.4178 (1.29)** |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 0.5646 | **lt-tf32-fresh 0.5591 (1.01)**<br>cublas-tf32-fresh 0.562 (1.00)<br>cublas-tf32 0.5642 (1.00)<br>lt-tf32 0.5652 (1.00) | torch-eager-tf32 0.5654 (1.00) | torch-compile-tf32 0.5652 (1.00) | torch-compile-cg-tf32 0.5642 (1.00) | torch-compile-ro-tf32 0.5652 (1.00) | triton-tf32 0.6187 (0.91) |
| pj_m128_n28672_k4096_tf32 | h100 | 0.2497 | **cublas-tf32 0.1889 (1.32)**<br>lt-tf32-fresh 0.1892 (1.32)<br>cublas-tf32-fresh 0.1899 (1.31)<br>lt-tf32 0.1903 (1.31) | torch-eager-tf32 0.1922 (1.30) | torch-compile-tf32 0.1916 (1.30) | torch-compile-cg-tf32 0.1915 (1.30) | torch-compile-ro-tf32 0.1916 (1.30) | triton-tf32 0.2629 (0.95) |
| pj_m128_n4096_k14336_tf32 | rtx-5080 | 0.4568 | **cublas-tf32-fresh 0.3092 (1.48)**<br>lt-tf32-fresh 0.3092 (1.48)<br>cublas-tf32 0.3195 (1.43)<br>lt-tf32 0.3195 (1.43) | torch-eager-tf32 0.3113 (1.47) | torch-compile-tf32 0.3113 (1.47) | torch-compile-cg-tf32 0.3113 (1.47) | torch-compile-ro-tf32 0.3113 (1.47) | triton-tf32 0.3707 (1.23) |
| pj_m128_n4096_k14336_tf32 | rtx-3090 | 0.7027 | lt-tf32 0.9554 (0.74)<br>lt-tf32-fresh 0.9554 (0.74)<br>cublas-tf32 1.155 (0.61)<br>cublas-tf32-fresh 1.155 (0.61) | torch-eager-tf32 1.066 (0.66) | **torch-compile-tf32 0.6042 (1.16)** | torch-compile-cg-tf32 0.6042 (1.16) | torch-compile-ro-tf32 1.066 (0.66) | triton-tf32 0.6564 (1.07) |
| pj_m128_n4096_k14336_tf32 | a100 | 0.4688 | **cublas-tf32-fresh 0.2314 (2.03)**<br>cublas-tf32 0.2324 (2.02)<br>lt-tf32 0.2324 (2.02)<br>lt-tf32-fresh 0.2324 (2.02) | torch-eager-tf32 0.257 (1.82) | torch-compile-tf32 0.257 (1.82) | torch-compile-cg-tf32 0.256 (1.83) | torch-compile-ro-tf32 0.256 (1.83) | triton-tf32 0.2611 (1.80) |
| pj_m128_n4096_k14336_tf32 | rtx-4090 | 0.3781 | **lt-tf32-fresh 0.3 (1.26)**<br>cublas-tf32-fresh 0.3021 (1.25)<br>lt-tf32 0.3134 (1.21)<br>cublas-tf32 0.3164 (1.19) | torch-eager-tf32 0.3205 (1.18) | torch-compile-tf32 0.3123 (1.21) | torch-compile-cg-tf32 0.3113 (1.21) | torch-compile-ro-tf32 0.3185 (1.19) | triton-tf32 0.3318 (1.14) |
| pj_m128_n4096_k14336_tf32 | h100 | 0.2397 | **lt-tf32 0.1079 (2.22)**<br>lt-tf32-fresh 0.11 (2.18)<br>cublas-tf32 0.1108 (2.16)<br>cublas-tf32-fresh 0.1111 (2.16) | torch-eager-tf32 0.1109 (2.16) | torch-compile-tf32 0.1106 (2.17) | torch-compile-cg-tf32 0.1106 (2.17) | torch-compile-ro-tf32 0.1108 (2.16) | triton-tf32 0.145 (1.65) |
| pj_m128_n4096_k4096_tf32 | rtx-5080 | 0.1355 | **lt-tf32-fresh 0.0819 (1.65)**<br>cublas-tf32-fresh 0.084 (1.61)<br>lt-tf32 0.1085 (1.25)<br>cublas-tf32 0.1106 (1.23) | torch-eager-tf32 0.1106 (1.23) | torch-compile-tf32 0.1126 (1.20) | torch-compile-cg-tf32 0.1106 (1.23) | torch-compile-ro-tf32 0.1106 (1.23) | triton-tf32 0.1106 (1.23) |
| pj_m128_n4096_k4096_tf32 | rtx-3090 | 0.2123 | lt-tf32 0.1761 (1.21)<br>lt-tf32-fresh 0.1761 (1.21)<br>cublas-tf32 0.1782 (1.19)<br>cublas-tf32-fresh 0.1782 (1.19) | torch-eager-tf32 0.1649 (1.29) | torch-compile-tf32 0.1659 (1.28) | **torch-compile-cg-tf32 0.1638 (1.30)** | torch-compile-ro-tf32 0.1649 (1.29) | triton-tf32 0.1884 (1.13) |
| pj_m128_n4096_k4096_tf32 | a100 | 0.1495 | **cublas-tf32-fresh 0.0829 (1.80)**<br>lt-tf32-fresh 0.0829 (1.80)<br>cublas-tf32 0.085 (1.76)<br>lt-tf32 0.085 (1.76) | torch-eager-tf32 0.0942 (1.59) | torch-compile-tf32 0.0952 (1.57) | torch-compile-cg-tf32 0.0942 (1.59) | torch-compile-ro-tf32 0.0942 (1.59) | triton-tf32 0.086 (1.74) |
| pj_m128_n4096_k4096_tf32 | rtx-4090 | 0.1051 | **cublas-tf32-fresh 0.0676 (1.55)**<br>lt-tf32-fresh 0.0676 (1.55)<br>cublas-tf32 0.1065 (0.99)<br>lt-tf32 0.1065 (0.99) | torch-eager-tf32 0.1065 (0.99) | torch-compile-tf32 0.1055 (1.00) | torch-compile-cg-tf32 0.1045 (1.01) | torch-compile-ro-tf32 0.1044 (1.01) | triton-tf32 0.1065 (0.99) |
| pj_m128_n4096_k4096_tf32 | h100 | 0.08655 | **cublas-tf32-fresh 0.044 (1.97)**<br>lt-tf32-fresh 0.0441 (1.96)<br>cublas-tf32 0.045 (1.92)<br>lt-tf32 0.0459 (1.89) | torch-eager-tf32 0.0457 (1.89) | torch-compile-tf32 0.0452 (1.91) | torch-compile-cg-tf32 0.0458 (1.89) | torch-compile-ro-tf32 0.0458 (1.89) | triton-tf32 0.0494 (1.75) |
| pj_m128_n6144_k4096_tf32 | rtx-5080 | 0.2368 | **lt-tf32-fresh 0.1249 (1.90)**<br>lt-tf32 0.1597 (1.48)<br>cublas-tf32-fresh 0.1802 (1.31)<br>cublas-tf32 0.1987 (1.19) | torch-eager-tf32 0.1905 (1.24) | torch-compile-tf32 0.172 (1.38) | torch-compile-cg-tf32 0.172 (1.38) | torch-compile-ro-tf32 0.1905 (1.24) | triton-tf32 0.2028 (1.17) |
| pj_m128_n6144_k4096_tf32 | rtx-3090 | 0.4119 | **lt-tf32 0.2458 (1.68)**<br>lt-tf32-fresh 0.2458 (1.68)<br>cublas-tf32-fresh 0.3584 (1.15)<br>cublas-tf32 0.3604 (1.14) | torch-eager-tf32 0.3185 (1.29) | torch-compile-tf32 0.2755 (1.49) | torch-compile-cg-tf32 0.2744 (1.50) | torch-compile-ro-tf32 0.3185 (1.29) | triton-tf32 0.3553 (1.16) |
| pj_m128_n6144_k4096_tf32 | a100 | 0.1604 | cublas-tf32 0.1034 (1.55)<br>lt-tf32 0.1034 (1.55)<br>cublas-tf32-fresh 0.1034 (1.55)<br>lt-tf32-fresh 0.1034 (1.55) | torch-eager-tf32 0.1085 (1.48) | torch-compile-tf32 0.1096 (1.46) | torch-compile-cg-tf32 0.1085 (1.48) | torch-compile-ro-tf32 0.1075 (1.49) | **triton-tf32 0.0993 (1.61)** |
| pj_m128_n6144_k4096_tf32 | rtx-4090 | 0.1852 | **lt-tf32-fresh 0.1198 (1.55)**<br>cublas-tf32-fresh 0.1199 (1.54)<br>cublas-tf32 0.1536 (1.21)<br>lt-tf32 0.1546 (1.20) | torch-eager-tf32 0.1536 (1.21) | torch-compile-tf32 0.1526 (1.21) | torch-compile-cg-tf32 0.1516 (1.22) | torch-compile-ro-tf32 0.1536 (1.21) | triton-tf32 0.1526 (1.21) |
| pj_m128_n6144_k4096_tf32 | h100 | 0.1496 | **lt-tf32-fresh 0.0524 (2.86)**<br>lt-tf32 0.0531 (2.82)<br>cublas-tf32-fresh 0.0615 (2.43)<br>cublas-tf32 0.0639 (2.34) | torch-eager-tf32 0.0638 (2.35) | torch-compile-tf32 0.0636 (2.35) | torch-compile-cg-tf32 0.0647 (2.31) | torch-compile-ro-tf32 0.0638 (2.35) | triton-tf32 0.0654 (2.29) |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 9.038 | **lt-tf32-fresh 8.062 (1.12)**<br>cublas-tf32-fresh 8.063 (1.12)<br>cublas-tf32 8.065 (1.12)<br>lt-tf32 8.066 (1.12) | torch-eager-tf32 8.14 (1.11) | torch-compile-tf32 8.141 (1.11) | torch-compile-cg-tf32 8.142 (1.11) | torch-compile-ro-tf32 8.144 (1.11) | triton-tf32 8.494 (1.06) |
| pj_m2048_n28672_k4096_tf32 | rtx-3090 | 14.47 | cublas-tf32 14.49 (1.00)<br>lt-tf32-fresh 14.49 (1.00)<br>cublas-tf32-fresh 14.49 (1.00)<br>lt-tf32 14.49 (1.00) | torch-eager-tf32 14.36 (1.01) | **torch-compile-tf32 14.35 (1.01)** | torch-compile-cg-tf32 14.36 (1.01) | torch-compile-ro-tf32 14.35 (1.01) | triton-tf32 15.23 (0.95) |
| pj_m2048_n28672_k4096_tf32 | a100 | 5.908 | cublas-tf32 4.156 (1.42)<br>cublas-tf32-fresh 4.156 (1.42)<br>lt-tf32 4.157 (1.42)<br>lt-tf32-fresh 4.157 (1.42) | **torch-eager-tf32 3.773 (1.57)** | torch-compile-tf32 4.395 (1.34) | torch-compile-cg-tf32 4.271 (1.38) | torch-compile-ro-tf32 4.273 (1.38) | triton-tf32 5.01 (1.18) |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 8.124 | **lt-tf32-fresh 5.936 (1.37)**<br>cublas-tf32-fresh 5.937 (1.37)<br>cublas-tf32 5.939 (1.37)<br>lt-tf32 5.939 (1.37) | torch-eager-tf32 5.977 (1.36) | torch-compile-tf32 5.945 (1.37) | torch-compile-cg-tf32 6.003 (1.35) | torch-compile-ro-tf32 5.972 (1.36) | triton-tf32 6.346 (1.28) |
| pj_m2048_n28672_k4096_tf32 | h100 | 2.031 | **lt-tf32 1.282 (1.58)**<br>lt-tf32-fresh 1.283 (1.58)<br>cublas-tf32 1.506 (1.35)<br>cublas-tf32-fresh 1.508 (1.35) | torch-eager-tf32 1.388 (1.46) | torch-compile-tf32 1.384 (1.47) | torch-compile-cg-tf32 1.385 (1.47) | torch-compile-ro-tf32 1.457 (1.39) | triton-tf32 5.082 (0.40) |
| pj_m2048_n4096_k14336_tf32 | rtx-5080 | 4.898 | **lt-tf32-fresh 4.244 (1.15)**<br>cublas-tf32-fresh 4.245 (1.15)<br>cublas-tf32 4.248 (1.15)<br>lt-tf32 4.253 (1.15) | torch-eager-tf32 4.533 (1.08) | torch-compile-tf32 4.311 (1.14) | torch-compile-cg-tf32 4.31 (1.14) | torch-compile-ro-tf32 4.54 (1.08) | triton-tf32 4.63 (1.06) |
| pj_m2048_n4096_k14336_tf32 | rtx-3090 | 8.154 | lt-tf32-fresh 7.758 (1.05)<br>cublas-tf32-fresh 7.759 (1.05)<br>cublas-tf32 7.764 (1.05)<br>lt-tf32 7.767 (1.05) | torch-eager-tf32 7.698 (1.06) | torch-compile-tf32 7.691 (1.06) | **torch-compile-cg-tf32 7.68 (1.06)** | torch-compile-ro-tf32 7.682 (1.06) | triton-tf32 8.148 (1.00) |
| pj_m2048_n4096_k14336_tf32 | a100 | 2.711 | cublas-tf32 2.148 (1.26)<br>cublas-tf32-fresh 2.148 (1.26)<br>lt-tf32-fresh 2.148 (1.26)<br>lt-tf32 2.149 (1.26) | torch-eager-tf32 2.095 (1.29) | torch-compile-tf32 2.298 (1.18) | torch-compile-cg-tf32 2.084 (1.30) | **torch-compile-ro-tf32 2.044 (1.33)** | triton-tf32 2.512 (1.08) |
| pj_m2048_n4096_k14336_tf32 | rtx-4090 | 3.311 | cublas-tf32-fresh 3.053 (1.08)<br>lt-tf32-fresh 3.053 (1.08)<br>cublas-tf32 3.054 (1.08)<br>lt-tf32 3.054 (1.08) | torch-eager-tf32 3.09 (1.07) | torch-compile-tf32 3.105 (1.07) | torch-compile-cg-tf32 3.083 (1.07) | **torch-compile-ro-tf32 3.003 (1.10)** | triton-tf32 3.242 (1.02) |
| pj_m2048_n4096_k14336_tf32 | h100 | 0.9886 | cublas-tf32 0.5858 (1.69)<br>lt-tf32 0.5859 (1.69)<br>cublas-tf32-fresh 0.5877 (1.68)<br>lt-tf32-fresh 0.5878 (1.68) | torch-eager-tf32 0.5827 (1.70) | torch-compile-tf32 0.6808 (1.45) | **torch-compile-cg-tf32 0.5811 (1.70)** | torch-compile-ro-tf32 0.583 (1.70) | triton-tf32 2.337 (0.42) |
| pj_m2048_n4096_k4096_tf32 | rtx-5080 | 1.416 | **lt-tf32 1.161 (1.22)**<br>lt-tf32-fresh 1.161 (1.22)<br>cublas-tf32-fresh 1.163 (1.22)<br>cublas-tf32 1.173 (1.21) | torch-eager-tf32 1.309 (1.08) | torch-compile-tf32 1.225 (1.16) | torch-compile-cg-tf32 1.229 (1.15) | torch-compile-ro-tf32 1.3 (1.09) | triton-tf32 1.348 (1.05) |
| pj_m2048_n4096_k4096_tf32 | rtx-3090 | 2.345 | cublas-tf32-fresh 2.46 (0.95)<br>cublas-tf32 2.461 (0.95)<br>lt-tf32 2.461 (0.95)<br>lt-tf32-fresh 2.462 (0.95) | torch-eager-tf32 2.486 (0.94) | torch-compile-tf32 2.606 (0.90) | torch-compile-cg-tf32 2.638 (0.89) | **torch-compile-ro-tf32 2.442 (0.96)** | triton-tf32 2.489 (0.94) |
| pj_m2048_n4096_k4096_tf32 | a100 | 0.7999 | **lt-tf32-fresh 0.5827 (1.37)**<br>lt-tf32 0.5837 (1.37)<br>cublas-tf32 0.597 (1.34)<br>cublas-tf32-fresh 0.598 (1.34) | torch-eager-tf32 0.6922 (1.16) | torch-compile-tf32 0.6912 (1.16) | torch-compile-cg-tf32 0.6902 (1.16) | torch-compile-ro-tf32 0.6902 (1.16) | triton-tf32 0.7004 (1.14) |
| pj_m2048_n4096_k4096_tf32 | rtx-4090 | 0.9701 | **lt-tf32 0.8294 (1.17)**<br>lt-tf32-fresh 0.8294 (1.17)<br>cublas-tf32-fresh 0.8346 (1.16)<br>cublas-tf32 0.8356 (1.16) | torch-eager-tf32 0.8714 (1.11) | torch-compile-tf32 0.8643 (1.12) | torch-compile-cg-tf32 0.8632 (1.12) | torch-compile-ro-tf32 0.8684 (1.12) | triton-tf32 0.9492 (1.02) |
| pj_m2048_n4096_k4096_tf32 | h100 | 0.3094 | **lt-tf32-fresh 0.1757 (1.76)**<br>cublas-tf32-fresh 0.1759 (1.76)<br>cublas-tf32 0.1761 (1.76)<br>lt-tf32 0.1763 (1.75) | torch-eager-tf32 0.1782 (1.74) | torch-compile-tf32 0.178 (1.74) | torch-compile-cg-tf32 0.178 (1.74) | torch-compile-ro-tf32 0.1777 (1.74) | triton-tf32 0.6639 (0.47) |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.017 | **cublas-tf32-fresh 1.748 (1.15)**<br>lt-tf32-fresh 1.75 (1.15)<br>cublas-tf32 1.759 (1.15)<br>lt-tf32 1.759 (1.15) | torch-eager-tf32 1.862 (1.08) | torch-compile-tf32 1.8 (1.12) | torch-compile-cg-tf32 1.8 (1.12) | torch-compile-ro-tf32 1.86 (1.08) | triton-tf32 1.972 (1.02) |
| pj_m2048_n6144_k4096_tf32 | rtx-3090 | 3.325 | cublas-tf32 3.227 (1.03)<br>lt-tf32 3.227 (1.03)<br>cublas-tf32-fresh 3.227 (1.03)<br>lt-tf32-fresh 3.227 (1.03) | **torch-eager-tf32 3.199 (1.04)** | torch-compile-tf32 3.692 (0.90) | torch-compile-cg-tf32 3.67 (0.91) | torch-compile-ro-tf32 3.225 (1.03) | triton-tf32 3.473 (0.96) |
| pj_m2048_n6144_k4096_tf32 | a100 | 1.248 | cublas-tf32 0.9359 (1.33)<br>lt-tf32 0.9359 (1.33)<br>lt-tf32-fresh 0.9359 (1.33)<br>cublas-tf32-fresh 0.937 (1.33) | torch-eager-tf32 0.8724 (1.43) | torch-compile-tf32 0.8141 (1.53) | **torch-compile-cg-tf32 0.8069 (1.55)** | torch-compile-ro-tf32 0.8141 (1.53) | triton-tf32 1.116 (1.12) |
| pj_m2048_n6144_k4096_tf32 | rtx-4090 | 1.431 | cublas-tf32 1.256 (1.14)<br>lt-tf32 1.256 (1.14)<br>cublas-tf32-fresh 1.256 (1.14)<br>lt-tf32-fresh 1.256 (1.14) | torch-eager-tf32 1.271 (1.13) | torch-compile-tf32 1.357 (1.05) | torch-compile-cg-tf32 1.216 (1.18) | **torch-compile-ro-tf32 1.162 (1.23)** | triton-tf32 1.402 (1.02) |
| pj_m2048_n6144_k4096_tf32 | h100 | 0.4528 | **lt-tf32-fresh 0.2596 (1.74)**<br>lt-tf32 0.2607 (1.74)<br>cublas-tf32-fresh 0.3174 (1.43)<br>cublas-tf32 0.3192 (1.42) | torch-eager-tf32 0.3216 (1.41) | torch-compile-tf32 0.3205 (1.41) | torch-compile-cg-tf32 0.3216 (1.41) | torch-compile-ro-tf32 0.322 (1.41) | triton-tf32 1.057 (0.43) |

### MLP chains

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| mlp2_2048 | rtx-5080 | 1.943 | **chain-lt2 1.765 (1.10)**<br>chain-streams 1.789 (1.09)<br>chain-lt 1.846 (1.05)<br>chain-eager 1.871 (1.04)<br>chain-cutlass 2.208 (0.88) | torch-eager 1.875 (1.04) | torch-compile 1.855 (1.05) | torch-compile-cg 1.839 (1.06) | torch-compile-ro 1.855 (1.05) | triton 2.41 (0.81) |
| mlp2_2048 | rtx-3090 | 3.203 | **chain-lt2 3.581 (0.89)**<br>chain-streams 3.663 (0.87)<br>chain-lt 3.742 (0.86)<br>chain-eager 3.816 (0.84)<br>chain-cutlass 4.458 (0.72) | torch-eager 3.97 (0.81) | torch-compile 3.892 (0.82) | torch-compile-cg 3.97 (0.81) | torch-compile-ro 3.912 (0.82) | triton 4.591 (0.70) |
| mlp2_2048 | a100 | 4.399 | **chain-streams 3.76 (1.17)**<br>chain-lt2 3.793 (1.16)<br>chain-eager 3.974 (1.11)<br>chain-lt 4.034 (1.09)<br>chain-cutlass 4.617 (0.95) | torch-eager 4.005 (1.10) | torch-compile 3.959 (1.11) | torch-compile-cg 3.95 (1.11) | torch-compile-ro 3.95 (1.11) | triton 3.99 (1.10) |
| mlp2_2048 | rtx-4090 | 1.308 | chain-cutlass 1.57 (0.83)<br>chain-lt2 1.592 (0.82)<br>chain-lt 1.613 (0.81)<br>chain-streams 1.617 (0.81)<br>chain-eager 1.641 (0.80) | torch-eager 1.544 (0.85) | torch-compile 1.377 (0.95) | torch-compile-cg 1.372 (0.95) | **torch-compile-ro 1.343 (0.97)** | triton 1.787 (0.73) |
| mlp2_2048 | h100 | 1.496 | **chain-lt2 1.353 (1.11)**<br>chain-streams 1.356 (1.10)<br>chain-lt 1.367 (1.09)<br>chain-eager 1.393 (1.07)<br>chain-cutlass 1.695 (0.88) | torch-eager 1.413 (1.06) | torch-compile 1.384 (1.08) | torch-compile-cg 1.381 (1.08) | torch-compile-ro 1.381 (1.08) | triton 1.661 (0.90) |
| mlp_2048 | rtx-5080 | 0.9414 | **chain-lt 0.9114 (1.03)**<br>chain-eager 0.9236 (1.02)<br>chain-cutlass 1.089 (0.86) | torch-eager 0.9257 (1.02) | torch-compile 0.9257 (1.02) | torch-compile-cg 0.9236 (1.02) | torch-compile-ro 0.9236 (1.02) | triton 1.198 (0.79) |
| mlp_2048 | rtx-3090 | 1.607 | chain-lt 1.829 (0.88)<br>chain-eager 1.873 (0.86)<br>chain-cutlass 2.18 (0.74) | torch-eager 2.064 (0.78) | torch-compile 2.121 (0.76) | torch-compile-cg 1.651 (0.97) | **torch-compile-ro 1.628 (0.99)** | triton 2.313 (0.69) |
| mlp_2048 | a100 | 2.224 | **chain-lt 1.964 (1.13)**<br>chain-eager 1.987 (1.12)<br>chain-cutlass 2.308 (0.96) | torch-eager 2.003 (1.11) | torch-compile 1.979 (1.12) | torch-compile-cg 1.976 (1.13) | torch-compile-ro 1.976 (1.13) | triton 1.995 (1.11) |
| mlp_2048 | rtx-4090 | 0.641 | chain-cutlass 0.7096 (0.90)<br>chain-lt 0.7332 (0.87)<br>chain-eager 0.7455 (0.86) | torch-eager 0.6799 (0.94) | torch-compile 0.6584 (0.97) | **torch-compile-cg 0.6543 (0.98)** | torch-compile-ro 0.6564 (0.98) | triton 0.7895 (0.81) |
| mlp_2048 | h100 | 0.7728 | **chain-lt 0.6844 (1.13)**<br>chain-eager 0.6973 (1.11)<br>chain-cutlass 0.8396 (0.92) | torch-eager 0.6916 (1.12) | torch-compile 0.6911 (1.12) | torch-compile-cg 0.7414 (1.04) | torch-compile-ro 0.6887 (1.12) | triton 0.8379 (0.92) |
| mlp_512 | rtx-5080 | 0.038 | chain-eager 0.043 (0.88)<br>chain-lt 0.043 (0.88)<br>chain-cutlass 0.1024 (0.37) | torch-eager 0.0451 (0.84) | **torch-compile 0.0328 (1.16)** | torch-compile-cg 0.0328 (1.16) | torch-compile-ro 0.0451 (0.84) | triton 0.0328 (1.16) |
| mlp_512 | rtx-3090 | 0.07699 | chain-lt 0.0512 (1.50)<br>chain-eager 0.0563 (1.37)<br>chain-cutlass 0.1208 (0.64) | torch-eager 0.0553 (1.39) | torch-compile 0.0502 (1.53) | torch-compile-cg 0.0492 (1.56) | torch-compile-ro 0.0522 (1.47) | **triton 0.0451 (1.71)** |
| mlp_512 | a100 | 0.07869 | **chain-lt 0.0625 (1.26)**<br>chain-eager 0.0655 (1.20)<br>chain-cutlass 0.2058 (0.38) | torch-eager 0.0655 (1.20) | torch-compile 0.0748 (1.05) | torch-compile-cg 0.0717 (1.10) | torch-compile-ro 0.0788 (1.00) | triton 0.0758 (1.04) |
| mlp_512 | rtx-4090 | 0.06188 | chain-lt 0.0358 (1.73)<br>chain-eager 0.0389 (1.59)<br>chain-cutlass 0.0891 (0.69) | torch-eager 0.041 (1.51) | torch-compile 0.0297 (2.08) | **torch-compile-cg 0.0287 (2.16)** | torch-compile-ro 0.0379 (1.63) | triton 0.0307 (2.02) |
| mlp_512 | h100 | 0.05912 | chain-lt 0.0344 (1.72)<br>chain-eager 0.0373 (1.58)<br>chain-cutlass 0.1171 (0.50) | torch-eager 0.0368 (1.61) | torch-compile 0.0328 (1.80) | **torch-compile-cg 0.0317 (1.86)** | torch-compile-ro 0.0338 (1.75) | triton 0.0389 (1.52) |
| smlp_4096_256_1024 | rtx-5080 | 0.1645 | chain-lt 0.1556 (1.06)<br>chain-cutlass 0.1577 (1.04)<br>chain-eager 0.1679 (0.98) | torch-eager 0.1772 (0.93) | **torch-compile 0.1495 (1.10)** | torch-compile-cg 0.1495 (1.10) | torch-compile-ro 0.1638 (1.00) | triton 0.1782 (0.92) |
| smlp_4096_256_1024 | rtx-3090 | 0.2733 | chain-cutlass 0.2437 (1.12)<br>chain-eager 0.2734 (1.00)<br>chain-lt 0.3256 (0.84) | torch-eager 0.3185 (0.86) | torch-compile 0.2232 (1.22) | **torch-compile-cg 0.2222 (1.23)** | torch-compile-ro 0.2683 (1.02) | triton 0.2939 (0.93) |
| smlp_4096_256_1024 | a100 | 0.3442 | **chain-eager 0.2918 (1.18)**<br>chain-lt 0.3348 (1.03)<br>chain-cutlass 0.3574 (0.96) | torch-eager 0.3379 (1.02) | torch-compile 0.3461 (0.99) | torch-compile-cg 0.3441 (1.00) | torch-compile-ro 0.3645 (0.94) | triton 0.3072 (1.12) |
| smlp_4096_256_1024 | rtx-4090 | 0.1205 | chain-eager 0.1372 (0.88)<br>chain-cutlass 0.1403 (0.86)<br>chain-lt 0.1987 (0.61) | torch-eager 0.1495 (0.81) | torch-compile 0.1087 (1.11) | torch-compile-cg 0.1106 (1.09) | torch-compile-ro 0.1311 (0.92) | **triton 0.1085 (1.11)** |
| smlp_4096_256_1024 | h100 | 0.1306 | **chain-lt 0.1119 (1.17)**<br>chain-eager 0.1276 (1.02)<br>chain-cutlass 0.1751 (0.75) | torch-eager 0.134 (0.97) | torch-compile 0.1168 (1.12) | torch-compile-cg 0.1163 (1.12) | torch-compile-ro 0.1182 (1.11) | triton 0.1219 (1.07) |
| mlp2_2048_tf32 | rtx-5080 | 1.668 |  | torch-eager-tf32 1.546 (1.08) | torch-compile-tf32 1.28 ✗gate | torch-compile-cg-tf32 1.282 ✗gate | **torch-compile-ro-tf32 1.514 (1.10)** | triton-tf32 1.429 ✗gate |
| mlp2_2048_tf32 | rtx-3090 | 2.758 |  | torch-eager-tf32 2.745 (1.00) | torch-compile-tf32 2.686 ✗gate | torch-compile-cg-tf32 2.635 ✗gate | **torch-compile-ro-tf32 2.648 (1.04)** | triton-tf32 2.45 ✗gate |
| mlp2_2048_tf32 | a100 | 0.9324 |  | torch-eager-tf32 0.7567 (1.23) | torch-compile-tf32 0.7055 (1.32) | **torch-compile-cg-tf32 0.6973 (1.34)** | torch-compile-ro-tf32 0.6994 (1.33) | triton-tf32 0.7731 ✗gate |
| mlp2_2048_tf32 | rtx-4090 | 1.033 |  | torch-eager-tf32 0.9677 (1.07) | torch-compile-tf32 0.8192 (1.26) | **torch-compile-cg-tf32 0.8141 (1.27)** | torch-compile-ro-tf32 0.8141 (1.27) | triton-tf32 0.9656 ✗gate |
| mlp2_2048_tf32 | h100 | 0.5295 |  | torch-eager-tf32 0.246 (2.15) | torch-compile-tf32 0.2261 (2.34) | **torch-compile-cg-tf32 0.2145 (2.47)** | torch-compile-ro-tf32 0.2148 (2.47) | triton-tf32 0.6575 ✗gate |
| mlp_2048_tf32 | rtx-5080 | 0.8298 |  | torch-eager-tf32 0.769 (1.08) | torch-compile-tf32 0.6369 ✗gate | torch-compile-cg-tf32 0.6369 ✗gate | **torch-compile-ro-tf32 0.7598 (1.09)** | triton-tf32 0.7229 ✗gate |
| mlp_2048_tf32 | rtx-3090 | 1.359 |  | torch-eager-tf32 1.556 (0.87) | torch-compile-tf32 1.022 ✗gate | torch-compile-cg-tf32 0.9738 ✗gate | **torch-compile-ro-tf32 1.184 (1.15)** | triton-tf32 1.238 ✗gate |
| mlp_2048_tf32 | a100 | 0.5078 |  | torch-eager-tf32 0.3543 (1.43) | torch-compile-tf32 0.3533 (1.44) | **torch-compile-cg-tf32 0.3512 (1.45)** | torch-compile-ro-tf32 0.3512 (1.45) | triton-tf32 0.3707 ✗gate |
| mlp_2048_tf32 | rtx-4090 | 0.5029 |  | torch-eager-tf32 0.4291 (1.17) | torch-compile-tf32 0.4127 (1.22) | **torch-compile-cg-tf32 0.4096 (1.23)** | torch-compile-ro-tf32 0.4096 (1.23) | triton-tf32 0.4884 ✗gate |
| mlp_2048_tf32 | h100 | 0.2701 |  | torch-eager-tf32 0.1123 (2.40) | torch-compile-tf32 0.1123 (2.40) | **torch-compile-cg-tf32 0.1107 (2.44)** | torch-compile-ro-tf32 0.111 (2.43) | triton-tf32 0.3312 ✗gate |

### Attention / softmax / long rows

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | hand Triton | hand CUDA |
|---|---|---:|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-5080 | 0.03914 | chain-eager 0.041 (0.95) | torch-eager 0.0492 (0.80) | torch-compile 0.0297 (1.32) | **torch-compile-cg 0.0287 (1.36)**<br>torch-sdpa-compile-cg 0.0287 (1.36) | torch-compile-ro 0.0287 (1.36) | torch-sdpa-compile 0.0307 (1.27)<br>torch-sdpa 0.0512 (0.76) | triton 0.1516 (0.26) | chain-fused 0.0348 (1.12) |
| attn_1024_64 | rtx-3090 | 0.0982 | chain-eager 0.0707 (1.39) | torch-eager 0.086 (1.14) | torch-compile 0.0573 (1.71) | torch-compile-cg 0.0543 (1.81)<br>torch-sdpa-compile-cg 0.084 (1.17) | **torch-compile-ro 0.0451 (2.18)** | torch-sdpa-compile 0.0809 (1.21)<br>torch-sdpa 0.0819 (1.20) | triton 0.2191 (0.45) | chain-fused 0.0573 (1.71) |
| attn_1024_64 | a100 | 0.06356 | chain-eager 0.0676 (0.94) | torch-eager 0.087 (0.73) | torch-compile 0.0614 (1.04) | **torch-sdpa-compile-cg 0.0543 (1.17)**<br>torch-compile-cg 0.0563 (1.13) | torch-compile-ro 0.0563 (1.13) | torch-sdpa-compile 0.0625 (1.02)<br>torch-sdpa 0.1014 (0.63) | triton 0.2376 (0.27) | chain-fused 0.0604 (1.05) |
| attn_1024_64 | rtx-4090 | 0.0603 | chain-eager 0.043 (1.40) | torch-eager 0.0522 (1.16) | torch-compile 0.0328 (1.84) | **torch-sdpa-compile-cg 0.0266 (2.27)**<br>torch-compile-cg 0.0297 (2.03) | torch-compile-ro 0.0297 (2.03) | torch-sdpa-compile 0.0316 (1.91)<br>torch-sdpa 0.0501 (1.20) | triton 0.1526 (0.40) | chain-fused 0.0379 (1.59) |
| attn_1024_64 | h100 | 0.0562 | chain-eager 0.0422 (1.33) | torch-eager 0.0564 (1.00) | torch-compile 0.0325 (1.73) | torch-sdpa-compile-cg 0.029 (1.94)<br>torch-compile-cg 0.0305 (1.84) | **torch-compile-ro 0.0268 (2.10)** | torch-sdpa-compile 0.0401 (1.40)<br>torch-sdpa 0.0536 (1.05) | triton 0.2357 (0.24) | chain-fused 0.0365 (1.54) |
| attn_2048_128 | rtx-5080 | 0.1177 | chain-eager 0.1393 (0.85) | torch-eager 0.1495 (0.79) | torch-compile 0.0993 (1.19) | **torch-sdpa-compile-cg 0.0942 (1.25)**<br>torch-compile-cg 0.1004 (1.17) | torch-compile-ro 0.1004 (1.17) | torch-sdpa-compile 0.0953 (1.24)<br>torch-sdpa 0.1413 (0.83) | triton 0.7741 (0.15) | chain-fused 0.127 (0.93) |
| attn_2048_128 | rtx-3090 | 0.2664 | chain-eager 0.2867 (0.93) | torch-eager 0.3676 (0.72) | torch-compile 0.2171 (1.23) | **torch-sdpa-compile-cg 0.1864 (1.43)**<br>torch-compile-cg 0.2355 (1.13) | torch-compile-ro 0.1956 (1.36) | torch-sdpa-compile 0.1894 (1.41)<br>torch-sdpa 0.2877 (0.93) | triton 0.9144 (0.29) | chain-fused 0.258 (1.03) |
| attn_2048_128 | a100 | 0.2613 | chain-eager 0.2263 (1.15) | torch-eager 0.3277 (0.80) | **torch-compile 0.1659 (1.58)** | torch-compile-cg 0.2089 (1.25)<br>torch-sdpa-compile-cg 0.2191 (1.19) | torch-compile-ro 0.2099 (1.24) | torch-sdpa-compile 0.2284 (1.14)<br>torch-sdpa 0.3011 (0.87) | triton 1.422 (0.18) | chain-fused 0.2191 (1.19) |
| attn_2048_128 | rtx-4090 | 0.1019 | chain-eager 0.1311 (0.78) | torch-eager 0.1485 (0.69) | torch-compile 0.0891 (1.14) | **torch-sdpa-compile-cg 0.084 (1.21)**<br>torch-compile-cg 0.087 (1.17) | torch-compile-ro 0.085 (1.20) | torch-sdpa-compile 0.0921 (1.11)<br>torch-sdpa 0.1474 (0.69) | triton 0.6083 (0.17) | chain-fused 0.1219 (0.84) |
| attn_2048_128 | h100 | 0.1035 | chain-eager 0.1114 (0.93) | torch-eager 0.1354 (0.76) | torch-compile 0.1037 (1.00) | torch-sdpa-compile-cg 0.0901 (1.15)<br>torch-compile-cg 0.1018 (1.02) | **torch-compile-ro 0.0789 (1.31)** | torch-sdpa-compile 0.0906 (1.14)<br>torch-sdpa 0.1211 (0.85) | triton 1.09 (0.09) | chain-fused 0.0986 (1.05) |
| attn_2048_64 | rtx-5080 | 0.07953 | chain-eager 0.1065 (0.75) | torch-eager 0.1208 (0.66) | torch-compile 0.0778 (1.02) | torch-sdpa-compile-cg 0.0737 (1.08)<br>torch-compile-cg 0.0758 (1.05) | **torch-compile-ro 0.0696 (1.14)** | torch-sdpa-compile 0.0748 (1.06)<br>torch-sdpa 0.1116 (0.71) | triton 0.2929 (0.27) | chain-fused 0.0942 (0.84) |
| attn_2048_64 | rtx-3090 | 0.1655 | chain-eager 0.2304 (0.72) | torch-eager 0.3133 (0.53) | torch-compile 0.1546 (1.07) | torch-compile-cg 0.1516 (1.09)<br>torch-sdpa-compile-cg 0.1843 (0.90) | **torch-compile-ro 0.1331 (1.24)** | torch-sdpa-compile 0.1731 (0.96)<br>torch-sdpa 0.2232 (0.74) | triton 0.383 (0.43) | chain-fused 0.2007 (0.82) |
| attn_2048_64 | a100 | 0.1573 | chain-eager 0.171 (0.92) | torch-eager 0.254 (0.62) | **torch-compile 0.1085 (1.45)** | torch-sdpa-compile-cg 0.1157 (1.36)<br>torch-compile-cg 0.1341 (1.17) | torch-compile-ro 0.1341 (1.17) | torch-sdpa-compile 0.1147 (1.37)<br>torch-sdpa 0.2253 (0.70) | triton 0.4669 (0.34) | chain-fused 0.1638 (0.96) |
| attn_2048_64 | rtx-4090 | 0.0759 | chain-eager 0.1116 (0.68) | torch-eager 0.1362 (0.56) | torch-compile 0.0829 (0.92) | torch-sdpa-compile-cg 0.0686 (1.11)<br>torch-compile-cg 0.0778 (0.98) | **torch-compile-ro 0.0645 (1.18)** | torch-sdpa-compile 0.0788 (0.96)<br>torch-sdpa 0.128 (0.59) | triton 0.2703 (0.28) | chain-fused 0.1014 (0.75) |
| attn_2048_64 | h100 | 0.08729 | chain-eager 0.094 (0.93) | torch-eager 0.1186 (0.74) | torch-compile 0.0728 (1.20) | torch-sdpa-compile-cg 0.0618 (1.41)<br>torch-compile-cg 0.0687 (1.27) | **torch-compile-ro 0.0603 (1.45)** | torch-sdpa-compile 0.0656 (1.33)<br>torch-sdpa 0.1035 (0.84) | triton 0.4552 (0.19) | chain-fused 0.0786 (1.11) |
| causal_1024_256 | rtx-5080 | 15.23 |  | torch-eager 90.53 (0.17) | torch-compile 44.52 (0.34) | **torch-compile-cg 38.23 (0.40)** | torch-compile-ro 42.56 (0.36) |  |  |  |
| causal_1024_256 | rtx-3090 | 24.55 |  | torch-eager 366.4 (0.07) | torch-compile 199.4 (0.12) | **torch-compile-cg 175.2 (0.14)** | torch-compile-ro 177.6 (0.14) |  |  |  |
| causal_1024_256 | a100 | 36.83 |  | torch-eager 150.8 (0.24) | torch-compile 75.36 (0.49) | **torch-compile-cg 74.11 (0.50)** | torch-compile-ro 80.56 (0.46) |  |  |  |
| causal_1024_256 | rtx-4090 | 18.6 |  | torch-eager 380.8 (0.05) | torch-compile 218 (0.09) | **torch-compile-cg 178.1 (0.10)** | torch-compile-ro 179.6 (0.10) |  |  |  |
| causal_1024_256 | h100 | 21.44 |  | torch-eager 195.1 (0.11) | torch-compile 133.2 (0.16) | **torch-compile-cg 120 (0.18)** | torch-compile-ro 123.7 (0.17) |  |  |  |
| decode_1024_256 | rtx-5080 | 8.635 | chain-nocache 23.73 (0.36) | torch-eager 47.94 (0.18) | torch-compile 44.2 (0.20) | torch-compile-cg 39.42 (0.22) | torch-compile-ro 36.74 (0.24) |  |  | **chain-cache 16.04 (0.54)** |
| decode_1024_256 | rtx-3090 | 25.58 | chain-nocache 37.85 (0.68) | torch-eager 188.9 (0.14) | torch-compile 175.1 (0.15) | torch-compile-cg 175 (0.15) | torch-compile-ro 175.8 (0.15) |  |  | **chain-cache 33.6 (0.76)** |
| decode_1024_256 | a100 | 27.32 | chain-nocache 54.06 (0.51) | torch-eager 80.27 (0.34) | torch-compile 67.6 (0.40) | torch-compile-cg 73.71 (0.37) | torch-compile-ro 76.75 (0.36) |  |  | **chain-cache 29.32 (0.93)** |
| decode_1024_256 | rtx-4090 | 25.78 | chain-nocache 32.46 (0.79) | torch-eager 207.4 (0.12) | torch-compile 186.3 (0.14) | torch-compile-cg 178.8 (0.14) | torch-compile-ro 179.1 (0.14) |  |  | **chain-cache 31.47 (0.82)** |
| decode_1024_256 | h100 | 18.61 | chain-nocache 30.21 (0.62) | torch-eager 103.8 (0.18) | torch-compile 119.6 (0.16) | torch-compile-cg 123 (0.15) | torch-compile-ro 121.1 (0.15) |  |  | **chain-cache 22.59 (0.82)** |
| gen_1024_512_256 | rtx-5080 | 7.685 |  | torch-eager 39.88 (0.19) | torch-compile 22.53 (0.34) | **torch-compile-cg 20.84 (0.37)** | torch-compile-ro 21.58 (0.36) |  |  |  |
| gen_1024_512_256 | rtx-3090 | 12.58 |  | torch-eager 173.9 (0.07) | torch-compile 98.05 (0.13) | **torch-compile-cg 88.2 (0.14)** | torch-compile-ro 88.32 (0.14) |  |  |  |
| gen_1024_512_256 | a100 | 18.57 |  | torch-eager 75.63 (0.25) | **torch-compile 37.71 (0.49)** | torch-compile-cg 37.77 (0.49) | torch-compile-ro 40.59 (0.46) |  |  |  |
| gen_1024_512_256 | rtx-4090 | 9.474 |  | torch-eager 177.5 (0.05) | torch-compile 99.81 (0.09) | torch-compile-cg 89.13 (0.11) | **torch-compile-ro 88.5 (0.11)** |  |  |  |
| gen_1024_512_256 | h100 | 10.83 |  | torch-eager 97.69 (0.11) | torch-compile 67.77 (0.16) | **torch-compile-cg 61.59 (0.18)** | torch-compile-ro 62.24 (0.17) |  |  |  |
| lsm_1_131072 | rtx-5080 | 0.01582 |  | torch-eager 0.0348 (0.45) | torch-compile 0.0246 (0.64) | **torch-compile-cg 0.0225 (0.70)**<br>torch-sdpa-compile-cg 0.0225 (0.70) | torch-compile-ro 0.0225 (0.70) | torch-sdpa-compile 0.0256 (0.62)<br>torch-sdpa 0.0512 (0.31) |  |  |
| lsm_1_131072 | rtx-3090 | 0.3071 |  | torch-eager 0.0543 (5.66) | torch-compile 0.0297 (10.34) | **torch-compile-cg 0.0256 (12.00)**<br>torch-sdpa-compile-cg 0.0256 (12.00) | torch-compile-ro 0.0256 (12.00) | torch-sdpa-compile 0.0307 (10.00)<br>torch-sdpa 0.0717 (4.28) |  |  |
| lsm_1_131072 | a100 | 0.3543 |  | torch-eager 0.0666 (5.32) | torch-compile 0.0328 (10.80) | **torch-compile-cg 0.0256 (13.84)**<br>torch-sdpa-compile-cg 0.0256 (13.84) | torch-compile-ro 0.0256 (13.84) | torch-sdpa-compile 0.0328 (10.80)<br>torch-sdpa 0.086 (4.12) |  |  |
| lsm_1_131072 | rtx-4090 | 0.2342 |  | torch-eager 0.0399 (5.87) | torch-compile 0.0276 (8.49) | **torch-compile-cg 0.0236 (9.92)**<br>torch-sdpa-compile-cg 0.0246 (9.52) | torch-compile-ro 0.0236 (9.92) | torch-sdpa-compile 0.0287 (8.16)<br>torch-sdpa 0.0532 (4.40) |  |  |
| lsm_1_131072 | h100 | 0.3114 |  | torch-eager 0.0442 (7.04) | torch-compile 0.0425 (7.33) | **torch-compile-cg 0.0227 (13.72)**<br>torch-sdpa-compile-cg 0.024 (12.97) | torch-compile-ro 0.0237 (13.14) | torch-sdpa-compile 0.044 (7.08)<br>torch-sdpa 0.058 (5.37) |  |  |
| lsm_32_8192 | rtx-5080 | 0.01538 |  | torch-eager 0.0215 (0.72) | **torch-compile 0.0102 (1.51)** | torch-compile-cg 0.0102 (1.51)<br>torch-sdpa-compile-cg 0.0102 (1.51) | torch-compile-ro 0.0102 (1.51) | torch-sdpa-compile 0.0102 (1.51)<br>torch-sdpa 0.0123 (1.25) | triton 0.0102 (1.51) |  |
| lsm_32_8192 | rtx-3090 | 0.03987 |  | torch-eager 0.0317 (1.26) | torch-compile 0.0143 (2.79) | torch-sdpa-compile-cg 0.0123 (3.24)<br>torch-compile-cg 0.0174 (2.29) | **torch-compile-ro 0.0113 (3.53)** | torch-sdpa-compile 0.0113 (3.53)<br>torch-sdpa 0.0174 (2.29) | triton 0.0133 (3.00) |  |
| lsm_32_8192 | a100 | 0.03052 |  | torch-eager 0.041 (0.74) | torch-compile 0.0184 (1.66) | **torch-sdpa-compile-cg 0.0123 (2.48)**<br>torch-compile-cg 0.0154 (1.98) | torch-compile-ro 0.0174 (1.75) | torch-sdpa-compile 0.0143 (2.13)<br>torch-sdpa 0.0256 (1.19) | triton 0.0164 (1.86) |  |
| lsm_32_8192 | rtx-4090 | 0.03994 |  | torch-eager 0.0215 (1.86) | torch-compile 0.009 (4.44) | **torch-compile-cg 0.0082 (4.87)**<br>torch-sdpa-compile-cg 0.0082 (4.87) | torch-compile-ro 0.0082 (4.87) | torch-sdpa-compile 0.009 (4.44)<br>torch-sdpa 0.0133 (3.00) | triton 0.01 (3.99) |  |
| lsm_32_8192 | h100 | 0.03233 |  | torch-eager 0.032 (1.01) | **torch-compile 0.0114 (2.84)** | torch-sdpa-compile-cg 0.0219 (1.48)<br>torch-compile-cg 0.0256 (1.26) | torch-compile-ro 0.0233 (1.39) | torch-sdpa-compile 0.0115 (2.81)<br>torch-sdpa 0.0164 (1.97) | triton 0.0145 (2.23) |  |
| lsm_4_32768 | rtx-5080 | 0.0166 |  | torch-eager 0.0256 (0.65) | **torch-compile 0.0143 (1.16)** | torch-compile-cg 0.0143 (1.16)<br>torch-sdpa-compile-cg 0.0143 (1.16) | torch-compile-ro 0.0143 (1.16) | torch-sdpa-compile 0.0143 (1.16)<br>torch-sdpa 0.0184 (0.90) | not run |  |
| lsm_4_32768 | rtx-3090 | 0.08167 |  | torch-eager 0.0389 (2.10) | torch-compile 0.0205 (3.98) | **torch-sdpa-compile-cg 0.0164 (4.98)**<br>torch-compile-cg 0.0174 (4.69) | torch-compile-ro 0.0174 (4.69) | torch-sdpa-compile 0.0205 (3.98)<br>torch-sdpa 0.0246 (3.32) | not run |  |
| lsm_4_32768 | a100 | 0.0743 |  | torch-eager 0.0512 (1.45) | torch-compile 0.0328 (2.27) | **torch-sdpa-compile-cg 0.0195 (3.81)**<br>torch-compile-cg 0.0205 (3.62) | torch-compile-ro 0.0215 (3.46) | torch-sdpa-compile 0.0266 (2.79)<br>torch-sdpa 0.0317 (2.34) | triton 0.0307 (2.42) |  |
| lsm_4_32768 | rtx-4090 | 0.06574 |  | torch-eager 0.0297 (2.21) | torch-compile 0.0174 (3.78) | **torch-compile-cg 0.0133 (4.94)**<br>torch-sdpa-compile-cg 0.0133 (4.94) | torch-compile-ro 0.0143 (4.60) | torch-sdpa-compile 0.0174 (3.78)<br>torch-sdpa 0.0195 (3.37) | not run |  |
| lsm_4_32768 | h100 | 0.06055 |  | torch-eager 0.0386 (1.57) | torch-compile 0.0451 (1.34) | torch-sdpa-compile-cg 0.0225 (2.69)<br>torch-compile-cg 0.023 (2.63) | **torch-compile-ro 0.0214 (2.83)** | torch-sdpa 0.0218 (2.78)<br>torch-sdpa-compile 0.0435 (1.39) | not run |  |
| prefill_1024_512_256 | rtx-5080 | 7.743 |  | torch-eager 39.81 (0.19) | torch-compile 22.49 (0.34) | **torch-compile-cg 19.26 (0.40)** | torch-compile-ro 21.52 (0.36) |  |  |  |
| prefill_1024_512_256 | rtx-3090 | 13.05 |  | torch-eager 174.6 (0.07) | torch-compile 98.99 (0.13) | **torch-compile-cg 87.9 (0.15)** | torch-compile-ro 90.39 (0.14) |  |  |  |
| prefill_1024_512_256 | a100 | 18.68 |  | torch-eager 75.67 (0.25) | torch-compile 38.03 (0.49) | **torch-compile-cg 37.47 (0.50)** | torch-compile-ro 40.65 (0.46) |  |  |  |
| prefill_1024_512_256 | rtx-4090 | 9.618 |  | torch-eager 179.7 (0.05) | torch-compile 103.6 (0.09) | **torch-compile-cg 95.96 (0.10)** | torch-compile-ro 96.16 (0.10) |  |  |  |
| prefill_1024_512_256 | h100 | 11.12 |  | torch-eager 106.2 (0.10) | torch-compile 67.57 (0.16) | torch-compile-cg 61.85 (0.18) | **torch-compile-ro 61.68 (0.18)** |  |  |  |
| rowsum_2048 | rtx-5080 | 0.4593 | **chain-eager 0.4608 (1.00)**<br>chain-gemv 0.4628 (0.99) | torch-eager 0.4649 (0.99) | torch-compile 0.4669 (0.98) | torch-compile-cg 0.4669 (0.98) | torch-compile-ro 0.4649 (0.99) |  |  |  |
| rowsum_2048 | rtx-3090 | 0.8423 | chain-gemv 0.9544 (0.88)<br>chain-eager 0.9595 (0.88) | torch-eager 0.8489 (0.99) | torch-compile 0.7567 (1.11) | torch-compile-cg 0.8212 (1.03) | **torch-compile-ro 0.7537 (1.12)** |  |  |  |
| rowsum_2048 | a100 | 1.036 | chain-eager 0.9974 (1.04)<br>chain-gemv 1 (1.04) | torch-eager 1.002 (1.03) | torch-compile 0.9964 (1.04) | torch-compile-cg 0.9953 (1.04) | **torch-compile-ro 0.9943 (1.04)** |  |  |  |
| rowsum_2048 | rtx-4090 | 0.328 | **chain-eager 0.3328 (0.99)**<br>chain-gemv 0.3348 (0.98) | torch-eager 0.3359 (0.98) | torch-compile 0.3471 (0.95) | torch-compile-cg 0.3441 (0.95) | torch-compile-ro 0.3451 (0.95) |  |  |  |
| rowsum_2048 | h100 | 0.4102 | chain-eager 0.3501 (1.17)<br>chain-gemv 0.3523 (1.16) | torch-eager 0.3493 (1.17) | torch-compile 0.3485 (1.18) | **torch-compile-cg 0.3467 (1.18)** | torch-compile-ro 0.3498 (1.17) |  |  |  |
| rowsum_4096 | rtx-5080 | 3.733 | chain-gemv 3.597 (1.04)<br>chain-eager 3.606 (1.04) | torch-eager 3.514 (1.06) | **torch-compile 3.486 (1.07)** | torch-compile-cg 3.488 (1.07) | torch-compile-ro 3.49 (1.07) |  |  |  |
| rowsum_4096 | rtx-3090 | 6.174 | chain-eager 6.781 (0.91)<br>chain-gemv 6.815 (0.91) | **torch-eager 6.742 (0.92)** | torch-compile 6.806 (0.91) | torch-compile-cg 6.767 (0.91) | torch-compile-ro 6.788 (0.91) |  |  |  |
| rowsum_4096 | a100 | 7.75 | chain-eager 7.347 (1.05)<br>chain-gemv 7.35 (1.05) | torch-eager 7.351 (1.05) | torch-compile 7.659 (1.01) | torch-compile-cg 7.634 (1.02) | **torch-compile-ro 7.337 (1.06)** |  |  |  |
| rowsum_4096 | rtx-4090 | 2.61 | chain-gemv 3.239 (0.81)<br>chain-eager 3.254 (0.80) | **torch-eager 2.596 (1.01)** | torch-compile 3.283 (0.79) | torch-compile-cg 3.284 (0.79) | torch-compile-ro 2.64 (0.99) |  |  |  |
| rowsum_4096 | h100 | 2.986 | chain-eager 2.692 (1.11)<br>chain-gemv 2.698 (1.11) | torch-eager 2.691 (1.11) | torch-compile 2.667 (1.12) | torch-compile-cg 2.668 (1.12) | **torch-compile-ro 2.661 (1.12)** |  |  |  |
| softmax_rows_2048 | rtx-5080 | 0.4727 | chain-eager 0.5018 (0.94) | torch-eager 0.5202 (0.91) | **torch-compile 0.469 (1.01)** | torch-compile-cg 0.471 (1.00) | torch-compile-ro 0.4731 (1.00) |  | triton 0.6042 (0.78) | chain-fused 0.4895 (0.97) |
| softmax_rows_2048 | rtx-3090 | 0.8804 | chain-eager 1.074 (0.82) | torch-eager 0.9851 (0.89) | torch-compile 0.7547 (1.17) | **torch-compile-cg 0.7516 (1.17)** | torch-compile-ro 0.7537 (1.17) |  | triton 0.8848 (1.00) | chain-fused 1.042 (0.84) |
| softmax_rows_2048 | a100 | 1.053 | chain-eager 1.055 (1.00) | torch-eager 1.402 (0.75) | torch-compile 1.22 (0.86) | torch-compile-cg 1.22 (0.86) | torch-compile-ro 1.067 (0.99) |  | **triton 1.019 (1.03)** | chain-fused 1.044 (1.01) |
| softmax_rows_2048 | rtx-4090 | 0.3377 | chain-eager 0.3668 (0.92) | torch-eager 0.3932 (0.86) | torch-compile 0.3492 (0.97) | **torch-compile-cg 0.3359 (1.01)** | torch-compile-ro 0.3359 (1.01) |  | triton 0.3994 (0.85) | chain-fused 0.3594 (0.94) |
| softmax_rows_2048 | h100 | 0.4216 | chain-eager 0.3855 (1.09) | torch-eager 0.4098 (1.03) | torch-compile 0.3564 (1.18) | **torch-compile-cg 0.3549 (1.19)** | torch-compile-ro 0.3555 (1.19) |  | triton 0.4329 (0.97) | chain-fused 0.3685 (1.14) |
| softmax_rows_4096 | rtx-5080 | 3.803 | chain-eager 3.907 (0.97) | torch-eager 4.117 (0.92) | torch-compile 3.54 (1.07) | torch-compile-cg 3.541 (1.07) | **torch-compile-ro 3.529 (1.08)** |  | triton 4.36 (0.87) | chain-fused 3.748 (1.01) |
| softmax_rows_4096 | rtx-3090 | 6.337 | chain-eager 7.293 (0.87) | torch-eager 7.602 (0.83) | torch-compile 6.954 (0.91) | torch-compile-cg 7.079 (0.90) | **torch-compile-ro 6.945 (0.91)** |  | triton 8.163 (0.78) | chain-fused 7.182 (0.88) |
| softmax_rows_4096 | a100 | 7.818 | chain-eager 7.631 (1.02) | torch-eager 8.315 (0.94) | torch-compile 7.754 (1.01) | torch-compile-cg 7.67 (1.02) | **torch-compile-ro 7.375 (1.06)** |  | triton 7.558 (1.03) | chain-fused 7.569 (1.03) |
| softmax_rows_4096 | rtx-4090 | 2.66 | chain-eager 3.437 (0.77) | torch-eager 3.625 (0.73) | torch-compile 3.364 (0.79) | **torch-compile-cg 2.912 (0.91)** | torch-compile-ro 3.315 (0.80) |  | triton 3.909 (0.68) | chain-fused 3.319 (0.80) |
| softmax_rows_4096 | h100 | 3.025 | chain-eager 2.85 (1.06) | torch-eager 2.946 (1.03) | torch-compile 2.704 (1.12) | **torch-compile-cg 2.702 (1.12)** | torch-compile-ro 2.703 (1.12) |  | triton 3.05 (0.99) | chain-fused 2.835 (1.07) |

### LLM decode / prefill / block (f32, tf32, bf16w)

| workload | part | Mapal ms | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | FlashAttention-2/3 | FlashInfer | hand Triton | Liger |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | rtx-5080 | 1.071 | torch-eager 1.137 (0.94) | **torch-compile 1.066 (1.01)** | torch-compile-cg 1.078 (0.99) | torch-compile-ro 1.11 (0.96) | torch-sdpa 1.555 (0.69)<br>torch-sdpa-efficient 1.867 (0.57) |  |  |  | liger 1.864 (0.57) |
| llm_blkdec_b1_l4096 | rtx-3090 | 1.131 | torch-eager 1.593 (0.71) | **torch-compile 1.251 (0.90)** | torch-compile-cg 1.261 (0.90) | torch-compile-ro 1.436 (0.79) | torch-sdpa 2.003 (0.56)<br>torch-sdpa-efficient 2.729 (0.41) |  |  |  | liger 2.674 (0.42) |
| llm_blkdec_b1_l4096 | a100 | 0.9575 | torch-eager 1.058 (0.90) | **torch-compile 0.8206 (1.17)** | torch-compile-cg 0.8237 (1.16) | torch-compile-ro 0.9508 (1.01) | torch-sdpa 1.317 (0.73)<br>torch-sdpa-efficient 1.601 (0.60) |  |  |  | liger 1.589 (0.60) |
| llm_blkdec_b1_l4096 | rtx-4090 | 1.486 | torch-eager 1.457 (1.02) | **torch-compile 1.069 (1.39)** | torch-compile-cg 1.089 (1.36) | torch-compile-ro 1.136 (1.31) | torch-sdpa 1.726 (0.86)<br>torch-sdpa-efficient 2.106 (0.71) |  |  |  | liger 2.074 (0.72) |
| llm_blkdec_b1_l4096 | h100 | 0.5697 | torch-eager 0.764 (0.75) | torch-compile 0.481 (1.18) | **torch-compile-cg 0.4745 (1.20)** | torch-compile-ro 0.5542 (1.03) | torch-sdpa 0.8019 (0.71)<br>torch-sdpa-efficient 1.074 (0.53) |  |  |  | liger 1.061 (0.54) |
| llm_blkpre_t2048 | rtx-5080 | 27.28 | torch-eager 33.04 (0.83) | torch-compile 28.09 (0.97) | torch-compile-cg 27.97 (0.98) | torch-compile-ro 29.04 (0.94) | torch-sdpa-efficient 27.16 (1.00)<br>torch-sdpa 33.45 (0.82) |  |  | triton 29.45 (0.93) | **liger 26.67 (1.02)** |
| llm_blkpre_t2048 | rtx-3090 | 61.08 | torch-eager 58.25 (1.05) | torch-compile 51.09 (1.20) | torch-compile-cg 51.36 (1.19) | torch-compile-ro 51.14 (1.19) | torch-sdpa-efficient 50.76 (1.20)<br>torch-sdpa 56.69 (1.08) |  |  | triton 53.97 (1.13) | **liger 49.54 (1.23)** |
| llm_blkpre_t2048 | a100 | 55.88 | torch-eager 61.58 (0.91) | torch-compile 54.6 (1.02) | **torch-compile-cg 54.57 (1.02)** | torch-compile-ro 59.05 (0.95) | torch-sdpa-efficient 56.38 (0.99)<br>torch-sdpa 61.88 (0.90) |  |  | triton 60.6 (0.92) | liger 55.85 (1.00) |
| llm_blkpre_t2048 | rtx-4090 | 20.1 | torch-eager 28.68 (0.70) | torch-compile 25.4 (0.79) | torch-compile-cg 25.34 (0.79) | torch-compile-ro 25.57 (0.79) | torch-sdpa-efficient 25.65 (0.78)<br>torch-sdpa 29.98 (0.67) |  |  | triton 24.97 (0.81) | **liger 23.03 (0.87)** |
| llm_blkpre_t2048 | h100 | 21.38 | torch-eager 21.14 (1.01) | torch-compile 19.54 (1.09) | torch-compile-cg 19.54 (1.09) | torch-compile-ro 19.63 (1.09) | torch-sdpa-efficient 18.83 (1.14)<br>torch-sdpa 21.4 (1.00) |  |  | triton 22.78 (0.94) | **liger 18.46 (1.16)** |
| llm_dec_b1_l2048 | rtx-5080 | 0.24 | torch-eager 0.2804 (0.86) | torch-compile 0.2536 (0.95) | torch-compile-cg 0.2636 (0.91) | **torch-compile-ro 0.2532 (0.95)** | torch-sdpa 0.4747 (0.51)<br>torch-sdpa-efficient 0.6538 (0.37) |  |  |  | liger 0.6613 (0.36) |
| llm_dec_b1_l2048 | rtx-3090 | 0.276 | torch-eager 0.6676 (0.41) | **torch-compile 0.3629 (0.76)** | torch-compile-cg 0.3836 (0.72) | torch-compile-ro 0.3931 (0.70) | torch-sdpa 0.8145 (0.34)<br>torch-sdpa-efficient 1.128 (0.24) |  |  |  | liger 1.105 (0.25) |
| llm_dec_b1_l2048 | a100 | 0.2155 | torch-eager 0.3444 (0.63) | **torch-compile 0.2501 (0.86)** | torch-compile-cg 0.2541 (0.85) | torch-compile-ro 0.2538 (0.85) | torch-sdpa 0.5085 (0.42)<br>torch-sdpa-efficient 0.6249 (0.34) |  |  |  | liger 0.6271 (0.34) |
| llm_dec_b1_l2048 | rtx-4090 | 0.2374 | torch-eager 0.6579 (0.36) | **torch-compile 0.3105 (0.76)** | torch-compile-cg 0.3357 (0.71) | torch-compile-ro 0.3553 (0.67) | torch-sdpa 0.6895 (0.34)<br>torch-sdpa-efficient 0.9135 (0.26) |  |  |  | liger 0.8962 (0.26) |
| llm_dec_b1_l2048 | h100 | 0.115 | torch-eager 0.3998 (0.29) | **torch-compile 0.2003 (0.57)** | torch-compile-cg 0.2052 (0.56) | torch-compile-ro 0.2146 (0.54) | torch-sdpa 0.435 (0.26)<br>torch-sdpa-efficient 0.5582 (0.21) |  |  |  | liger 0.5684 (0.20) |
| llm_dec_b1_l8192 | rtx-5080 | 0.3324 | torch-eager 0.4058 (0.82) | **torch-compile 0.342 (0.97)** | torch-compile-cg 0.3468 (0.96) | torch-compile-ro 0.3619 (0.92) | torch-sdpa 1.274 (0.26)<br>torch-sdpa-efficient 1.914 (0.17) |  |  |  | liger 1.917 (0.17) |
| llm_dec_b1_l8192 | rtx-3090 | 0.379 | torch-eager 0.7624 (0.50) | **torch-compile 0.4934 (0.77)** | torch-compile-cg 0.5331 (0.71) | torch-compile-ro 0.517 (0.73) | torch-sdpa 1.826 (0.21)<br>torch-sdpa-efficient 3.113 (0.12) |  |  |  | liger 3.098 (0.12) |
| llm_dec_b1_l8192 | a100 | 0.2774 | torch-eager 0.5484 (0.51) | **torch-compile 0.3949 (0.70)** | torch-compile-cg 0.401 (0.69) | torch-compile-ro 0.4579 (0.61) | torch-sdpa 1.035 (0.27)<br>torch-sdpa-efficient 1.65 (0.17) |  |  |  | liger 1.654 (0.17) |
| llm_dec_b1_l8192 | rtx-4090 | 0.3319 | torch-eager 0.7256 (0.46) | **torch-compile 0.372 (0.89)** | torch-compile-cg 0.3943 (0.84) | torch-compile-ro 0.4517 (0.73) | torch-sdpa 1.391 (0.24)<br>torch-sdpa-efficient 2.178 (0.15) |  |  |  | liger 2.155 (0.15) |
| llm_dec_b1_l8192 | h100 | 0.1565 | torch-eager 0.6457 (0.24) | **torch-compile 0.3047 (0.51)** | torch-compile-cg 0.31 (0.50) | torch-compile-ro 0.4775 (0.33) | torch-sdpa 0.709 (0.22)<br>torch-sdpa-efficient 1.286 (0.12) |  |  |  | liger 1.296 (0.12) |
| llm_dec_b32_l2048 | rtx-5080 | 1.033 | torch-eager 2.591 (0.40) | **torch-compile 2.256 (0.46)** | torch-compile-cg 2.272 (0.45) | torch-compile-ro 2.357 (0.44) | torch-sdpa-efficient 7.964 (0.13)<br>torch-sdpa 8.786 (0.12) |  |  |  | liger 7.983 (0.13) |
| llm_dec_b32_l2048 | rtx-3090 | 1.218 | torch-eager 3.18 (0.38) | torch-compile 2.991 (0.41) | torch-compile-cg 3.003 (0.41) | **torch-compile-ro 2.943 (0.41)** | torch-sdpa 9.491 (0.13)<br>torch-sdpa-efficient 11.65 (0.10) |  |  |  | liger 11.59 (0.11) |
| llm_dec_b32_l2048 | a100 | 0.8605 | torch-eager 2.001 (0.43) | torch-compile 1.839 (0.47) | **torch-compile-cg 1.832 (0.47)** | torch-compile-ro 1.863 (0.46) | torch-sdpa-efficient 5.799 (0.15)<br>torch-sdpa 6.104 (0.14) |  |  |  | liger 5.745 (0.15) |
| llm_dec_b32_l2048 | rtx-4090 | 1.062 | torch-eager 2.564 (0.41) | **torch-compile 2.293 (0.46)** | torch-compile-cg 2.335 (0.45) | torch-compile-ro 2.314 (0.46) | torch-sdpa-efficient 6.411 (0.17)<br>torch-sdpa 7.951 (0.13) |  |  |  | liger 6.353 (0.17) |
| llm_dec_b32_l2048 | h100 | 0.4756 | torch-eager 1.395 (0.34) | **torch-compile 1.149 (0.41)** | torch-compile-cg 1.156 (0.41) | torch-compile-ro 1.21 (0.39) | torch-sdpa-efficient 3.512 (0.14)<br>torch-sdpa 3.626 (0.13) |  |  |  | liger 3.553 (0.13) |
| llm_dec_b32_l8192 | rtx-5080 | 3.506 | torch-eager 10.68 (0.33) | **torch-compile 9.484 (0.37)** | torch-compile-cg 9.494 (0.37) | torch-compile-ro 9.948 (0.35) | torch-sdpa-efficient 29.97 (0.12) |  |  |  | not run |
| llm_dec_b32_l8192 | rtx-3090 | 4.27 | torch-eager 12 (0.36) | torch-compile 11.31 (0.38) | torch-compile-cg 11.38 (0.38) | **torch-compile-ro 11.18 (0.38)** | torch-sdpa 41.55 (0.10)<br>torch-sdpa-efficient 52.23 (0.08) |  |  |  | liger 51.73 (0.08) |
| llm_dec_b32_l8192 | a100 | 2.535 | torch-eager 6.985 (0.36) | torch-compile 6.558 (0.39) | **torch-compile-cg 6.499 (0.39)** | torch-compile-ro 6.552 (0.39) | torch-sdpa-efficient 22.47 (0.11)<br>torch-sdpa 23.76 (0.11) |  |  |  | liger 22.25 (0.11) |
| llm_dec_b32_l8192 | rtx-4090 | 3.502 | torch-eager 10.32 (0.34) | **torch-compile 8.986 (0.39)** | torch-compile-cg 9.007 (0.39) | torch-compile-ro 9.025 (0.39) | torch-sdpa-efficient 24.14 (0.15)<br>torch-sdpa 29.89 (0.12) |  |  |  | liger 23.74 (0.15) |
| llm_dec_b32_l8192 | h100 | 1.052 | torch-eager 4.498 (0.23) | **torch-compile 3.878 (0.27)** | torch-compile-cg 3.913 (0.27) | torch-compile-ro 4.149 (0.25) | torch-sdpa-efficient 12.72 (0.08)<br>torch-sdpa 13.46 (0.08) |  |  |  | liger 12.78 (0.08) |
| llm_dec_b8_l2048 | rtx-5080 | 0.4385 | torch-eager 0.7942 (0.55) | **torch-compile 0.6685 (0.66)** | torch-compile-cg 0.6806 (0.64) | torch-compile-ro 0.7235 (0.61) | torch-sdpa 2.369 (0.19)<br>torch-sdpa-efficient 2.402 (0.18) |  |  |  | liger 2.411 (0.18) |
| llm_dec_b8_l2048 | rtx-3090 | 1.007 | torch-eager 1.177 (0.86) | **torch-compile 0.9416 (1.07)** | torch-compile-cg 0.9547 (1.05) | torch-compile-ro 1.015 (0.99) | torch-sdpa 2.654 (0.38)<br>torch-sdpa-efficient 3.643 (0.28) |  |  |  | liger 3.628 (0.28) |
| llm_dec_b8_l2048 | a100 | 0.3931 | torch-eager 0.8007 (0.49) | **torch-compile 0.6163 (0.64)** | torch-compile-cg 0.6259 (0.63) | torch-compile-ro 0.6926 (0.57) | torch-sdpa-efficient 1.79 (0.22)<br>torch-sdpa 1.817 (0.22) |  |  |  | liger 1.785 (0.22) |
| llm_dec_b8_l2048 | rtx-4090 | 0.4762 | torch-eager 0.9843 (0.48) | **torch-compile 0.7501 (0.63)** | torch-compile-cg 0.7595 (0.63) | torch-compile-ro 0.7848 (0.61) | torch-sdpa-efficient 1.96 (0.24)<br>torch-sdpa 2.38 (0.20) |  |  |  | liger 1.945 (0.24) |
| llm_dec_b8_l2048 | h100 | 0.2096 | torch-eager 0.6212 (0.34) | **torch-compile 0.4127 (0.51)** | torch-compile-cg 0.4338 (0.48) | torch-compile-ro 0.4923 (0.43) | torch-sdpa-efficient 1.139 (0.18)<br>torch-sdpa 1.15 (0.18) |  |  |  | liger 1.172 (0.18) |
| llm_dec_b8_l8192 | rtx-5080 | 1.067 | torch-eager 2.831 (0.38) | torch-compile 2.488 (0.43) | **torch-compile-cg 2.468 (0.43)** | torch-compile-ro 2.529 (0.42) | torch-sdpa-efficient 8.634 (0.12)<br>torch-sdpa 8.663 (0.12) |  |  |  | liger 8.659 (0.12) |
| llm_dec_b8_l8192 | rtx-3090 | 1.787 | torch-eager 3.438 (0.52) | **torch-compile 2.957 (0.60)** | torch-compile-cg 2.967 (0.60) | torch-compile-ro 3.063 (0.58) | torch-sdpa 10.76 (0.17)<br>torch-sdpa-efficient 14.51 (0.12) |  |  |  | liger 14.49 (0.12) |
| llm_dec_b8_l8192 | a100 | 0.8098 | torch-eager 2.053 (0.39) | **torch-compile 1.806 (0.45)** | torch-compile-cg 1.817 (0.45) | torch-compile-ro 1.894 (0.43) | torch-sdpa 6.166 (0.13)<br>torch-sdpa-efficient 6.263 (0.13) |  |  |  | liger 6.226 (0.13) |
| llm_dec_b8_l8192 | rtx-4090 | 1.063 | torch-eager 2.965 (0.36) | **torch-compile 2.444 (0.43)** | torch-compile-cg 2.461 (0.43) | torch-compile-ro 2.472 (0.43) | torch-sdpa-efficient 6.332 (0.17)<br>torch-sdpa 7.862 (0.14) |  |  |  | liger 6.284 (0.17) |
| llm_dec_b8_l8192 | h100 | 0.4752 | torch-eager 1.458 (0.33) | **torch-compile 1.096 (0.43)** | torch-compile-cg 1.123 (0.42) | torch-compile-ro 1.231 (0.39) | torch-sdpa-efficient 3.509 (0.14)<br>torch-sdpa 3.639 (0.13) |  |  |  | liger 3.537 (0.13) |
| llm_pre_t2048 | rtx-5080 | 8.56 | torch-eager 13.43 (0.64) | torch-compile 8.367 (1.02) | torch-compile-cg 8.321 (1.03) | torch-compile-ro 9.35 (0.92) | torch-sdpa-efficient 7.286 (1.17)<br>torch-sdpa 13.68 (0.63) |  |  | triton 9.424 (0.91) | **liger 7.2 (1.19)** |
| llm_pre_t2048 | rtx-3090 | 14.34 | torch-eager 19.48 (0.74) | torch-compile 15.35 (0.93) | torch-compile-cg 15.43 (0.93) | torch-compile-ro 15.04 (0.95) | **torch-sdpa-efficient 13.27 (1.08)**<br>torch-sdpa 19.52 (0.73) |  |  | triton 18.56 (0.77) | liger 13.36 (1.07) |
| llm_pre_t2048 | a100 | 15.13 | torch-eager 17.73 (0.85) | torch-compile 14.65 (1.03) | torch-compile-cg 14.73 (1.03) | torch-compile-ro 14.95 (1.01) | torch-sdpa-efficient 12.14 (1.25)<br>torch-sdpa 17.98 (0.84) |  |  | triton 16.31 (0.93) | **liger 11.96 (1.27)** |
| llm_pre_t2048 | rtx-4090 | 6.499 | torch-eager 10.21 (0.64) | torch-compile 7.779 (0.84) | torch-compile-cg 7.826 (0.83) | torch-compile-ro 7.666 (0.85) | torch-sdpa-efficient 5.808 (1.12)<br>torch-sdpa 9.991 (0.65) |  |  | triton 7.371 (0.88) | **liger 5.689 (1.14)** |
| llm_pre_t2048 | h100 | 6.121 | torch-eager 7.031 (0.87) | torch-compile 5.47 (1.12) | torch-compile-cg 5.433 (1.13) | torch-compile-ro 5.538 (1.11) | torch-sdpa-efficient 4.572 (1.34)<br>torch-sdpa 7.135 (0.86) |  |  | triton 8.469 (0.72) | **liger 4.512 (1.36)** |
| llm_pre_t8192 | rtx-5080 | 80.68 | torch-eager 166 (0.49) | torch-compile 84.52 (0.95) | torch-compile-cg 83.79 (0.96) | torch-compile-ro 100.1 (0.81) | torch-sdpa-efficient 53.58 (1.51) |  |  | triton 84.28 (0.96) | **liger 52.83 (1.53)** |
| llm_pre_t8192 | rtx-3090 | 153.7 | torch-eager 203.9 (0.75) | torch-compile 147.9 (1.04) | torch-compile-cg 148 (1.04) | torch-compile-ro 142.4 (1.08) | torch-sdpa-efficient 98.03 (1.57)<br>torch-sdpa 210.9 (0.73) |  |  | triton 179.6 (0.86) | **liger 97.29 (1.58)** |
| llm_pre_t8192 | a100 | 132.3 | torch-eager 163.8 (0.81) | torch-compile 115.7 (1.14) | torch-compile-cg 115.5 (1.15) | torch-compile-ro 119 (1.11) | torch-sdpa-efficient 61.58 (2.15)<br>torch-sdpa 160.4 (0.82) |  |  | triton 121.5 (1.09) | **liger 61.05 (2.17)** |
| llm_pre_t8192 | rtx-4090 | 64.53 | torch-eager 118.5 (0.54) | torch-compile 75.7 (0.85) | torch-compile-cg 75.54 (0.85) | torch-compile-ro 75.55 (0.85) | torch-sdpa-efficient 41.11 (1.57)<br>torch-sdpa 121.4 (0.53) |  |  | triton 66.25 (0.97) | **liger 40.13 (1.61)** |
| llm_pre_t8192 | h100 | 51.58 | torch-eager 68.25 (0.76) | torch-compile 43.96 (1.17) | torch-compile-cg 43.86 (1.18) | torch-compile-ro 43.98 (1.17) | torch-sdpa-efficient 26.37 (1.96)<br>torch-sdpa 70.98 (0.73) |  |  | triton 74.2 (0.70) | **liger 25.99 (1.98)** |
| llm_blkpre_t2048_tf32 | rtx-5080 | 21.83 | torch-eager-tf32 24.26 (0.90) | torch-compile-tf32 18.56 (1.18) | torch-compile-cg-tf32 18.58 (1.17) | torch-compile-ro-tf32 19.56 (1.12) | torch-sdpa-tf32 23.94 (0.91) |  |  | **triton-tf32 18.08 (1.21)** | liger-tf32 19.12 (1.14) |
| llm_blkpre_t2048_tf32 | rtx-3090 | 35.17 | torch-eager-tf32 38.53 (0.91) | torch-compile-tf32 34.59 (1.02) | torch-compile-cg-tf32 34.61 (1.02) | torch-compile-ro-tf32 33.19 (1.06) | torch-sdpa-tf32 38.88 (0.90) |  |  | **triton-tf32 30.99 (1.14)** | liger-tf32 32.87 (1.07) |
| llm_blkpre_t2048_tf32 | a100 | 16.36 | torch-eager-tf32 13.45 (1.22) | torch-compile-tf32 10.54 (1.55) | torch-compile-cg-tf32 10.51 (1.56) | torch-compile-ro-tf32 10.53 (1.55) | torch-sdpa-tf32 13.72 (1.19) |  |  | **triton-tf32 9.895 (1.65)** | liger-tf32 10.44 (1.57) |
| llm_blkpre_t2048_tf32 | rtx-4090 | 16.52 | torch-eager-tf32 18.73 (0.88) | torch-compile-tf32 14.66 (1.13) | torch-compile-cg-tf32 14.76 (1.12) | torch-compile-ro-tf32 14.49 (1.14) | torch-sdpa-tf32 18.57 (0.89) |  |  | **triton-tf32 13.06 (1.27)** | liger-tf32 13.55 (1.22) |
| llm_blkpre_t2048_tf32 | h100 | 6.678 | torch-eager-tf32 5.125 (1.30) | torch-compile-tf32 3.605 (1.85) | **torch-compile-cg-tf32 3.47 (1.92)** | torch-compile-ro-tf32 3.56 (1.88) | torch-sdpa-tf32 5.237 (1.28) |  |  | triton-tf32 3.482 (1.92) | liger-tf32 3.802 (1.76) |
| llm_pre_t2048_tf32 | rtx-5080 | 7.455 | torch-eager-tf32 10.76 (0.69) | torch-compile-tf32 6.062 (1.23) | torch-compile-cg-tf32 6.091 (1.22) | torch-compile-ro-tf32 6.382 (1.17) | torch-sdpa-tf32 10.39 (0.72) |  |  | **triton-tf32 4.537 (1.64)** | liger-tf32 5.926 (1.26) |
| llm_pre_t2048_tf32 | rtx-3090 | 12.62 | torch-eager-tf32 15.23 (0.83) | torch-compile-tf32 10.96 (1.15) | torch-compile-cg-tf32 11 (1.15) | torch-compile-ro-tf32 10.44 (1.21) | torch-sdpa-tf32 15.33 (0.82) |  |  | **triton-tf32 8.286 (1.52)** | liger-tf32 10.35 (1.22) |
| llm_pre_t2048_tf32 | a100 | 7.426 | torch-eager-tf32 6.486 (1.14) | torch-compile-tf32 3.709 (2.00) | torch-compile-cg-tf32 3.685 (2.01) | torch-compile-ro-tf32 3.74 (1.99) | torch-sdpa-tf32 6.729 (1.10) |  |  | **triton-tf32 2.832 (2.62)** | liger-tf32 3.574 (2.08) |
| llm_pre_t2048_tf32 | rtx-4090 | 5.738 | torch-eager-tf32 8.65 (0.66) | torch-compile-tf32 4.998 (1.15) | torch-compile-cg-tf32 5.002 (1.15) | torch-compile-ro-tf32 5.02 (1.14) | torch-sdpa-tf32 8.458 (0.68) |  |  | **triton-tf32 3.398 (1.69)** | liger-tf32 4.309 (1.33) |
| llm_pre_t2048_tf32 | h100 | 3.217 | torch-eager-tf32 3.008 (1.07) | torch-compile-tf32 1.514 (2.12) | torch-compile-cg-tf32 1.504 (2.14) | torch-compile-ro-tf32 1.52 (2.12) | torch-sdpa-tf32 3.119 (1.03) |  |  | **triton-tf32 1.372 (2.35)** | liger-tf32 1.707 (1.88) |
| llm_blkdec_b1_l4096_bf16w | rtx-5080 | 0.636 | torch-bf16 0.6244 (1.02) | **torch-bf16-compile 0.553 (1.15)** | torch-bf16-compile-cg 0.5636 (1.13) | torch-bf16-compile-ro 0.9708 (0.66) | torch-bf16-cudnn 0.6171 (1.03)<br>torch-bf16-efficient 0.9613 (0.66) | fa2-bf16 0.6159 (1.03) | flashinfer-bf16 0.7054 (0.90) |  | liger-bf16 0.5893 (1.08) |
| llm_blkdec_b1_l4096_bf16w | rtx-3090 | 0.7061 | torch-bf16 1.054 (0.67) | **torch-bf16-compile 0.6983 (1.01)** | torch-bf16-compile-cg 0.7003 (1.01) | torch-bf16-compile-ro 1.265 (0.56) | torch-bf16-cudnn 1.112 (0.64)<br>torch-bf16-efficient 1.347 (0.52) | fa2-bf16 1.037 (0.68) | flashinfer-bf16 1.209 (0.58) |  | liger-bf16 1.023 (0.69) |
| llm_blkdec_b1_l4096_bf16w | a100 | 0.7864 | torch-bf16 0.57 (1.38) | torch-bf16-compile 0.4916 (1.60) | **torch-bf16-compile-cg 0.4551 (1.73)** | torch-bf16-compile-ro 0.7867 (1.00) | torch-bf16-cudnn 0.5862 (1.34)<br>torch-bf16-efficient 0.8721 (0.90) | fa2-bf16 0.5761 (1.37) | flashinfer-bf16 0.6804 (1.16) |  | liger-bf16 0.586 (1.34) |
| llm_blkdec_b1_l4096_bf16w | rtx-4090 | 0.5813 | torch-bf16 1.012 (0.57) | **torch-bf16-compile 0.618 (0.94)** | torch-bf16-compile-cg 0.6206 (0.94) | torch-bf16-compile-ro 0.9873 (0.59) | torch-bf16-cudnn 1.059 (0.55)<br>torch-bf16-efficient 1.179 (0.49) | fa2-bf16 0.9857 (0.59) | flashinfer-bf16 1.162 (0.50) |  | liger-bf16 0.9812 (0.59) |
| llm_blkdec_b1_l4096_bf16w | h100 | 0.3538 | torch-bf16 0.5247 (0.67) | torch-bf16-compile 0.3316 (1.07) | **torch-bf16-compile-cg 0.2899 (1.22)** | torch-bf16-compile-ro 0.2945 (1.20) | torch-bf16-cudnn 0.5874 (0.60)<br>torch-bf16-efficient 0.7242 (0.49) | fa2-bf16 0.5095 (0.69)<br>fa3-bf16 0.5581 (0.63) | flashinfer-bf16 0.6634 (0.53) |  | liger-bf16 0.5771 (0.61) |
| llm_dec_b1_l2048_bf16w | rtx-5080 | 0.1501 | torch-bf16 0.1962 (0.77) | **torch-bf16-compile 0.1471 (1.02)** | torch-bf16-compile-cg 0.1542 (0.97) | torch-bf16-compile-ro 0.2913 (0.52) | torch-bf16-cudnn 0.1853 (0.81)<br>torch-bf16-efficient 0.3393 (0.44) | fa2-bf16 0.1848 (0.81) | flashinfer-bf16 0.254 (0.59) |  | liger-bf16 0.1781 (0.84) |
| llm_dec_b1_l2048_bf16w | rtx-3090 | 0.1846 | torch-bf16 0.5288 (0.35) | **torch-bf16-compile 0.2548 (0.72)** | torch-bf16-compile-cg 0.2817 (0.66) | torch-bf16-compile-ro 0.521 (0.35) | torch-bf16-cudnn 0.5739 (0.32)<br>torch-bf16-efficient 0.6857 (0.27) | fa2-bf16 0.5156 (0.36) | flashinfer-bf16 0.6974 (0.26) |  | liger-bf16 0.5038 (0.37) |
| llm_dec_b1_l2048_bf16w | a100 | 0.1693 | torch-bf16 0.2389 (0.71) | **torch-bf16-compile 0.1576 (1.07)** | torch-bf16-compile-cg 0.1655 (1.02) | torch-bf16-compile-ro 0.326 (0.52) | torch-bf16-cudnn 0.2592 (0.65)<br>torch-bf16-efficient 0.4006 (0.42) | fa2-bf16 0.2398 (0.71) | flashinfer-bf16 0.3288 (0.51) |  | liger-bf16 0.2558 (0.66) |
| llm_dec_b1_l2048_bf16w | rtx-4090 | 0.1523 | torch-bf16 0.5364 (0.28) | torch-bf16-compile 0.2545 (0.60) | **torch-bf16-compile-cg 0.2379 (0.64)** | torch-bf16-compile-ro 0.3736 (0.41) | torch-bf16-cudnn 0.5871 (0.26)<br>torch-bf16-efficient 0.6251 (0.24) | fa2-bf16 0.5204 (0.29) | flashinfer-bf16 0.7177 (0.21) |  | liger-bf16 0.513 (0.30) |
| llm_dec_b1_l2048_bf16w | h100 | 0.09451 | torch-bf16 0.3462 (0.27) | torch-bf16-compile 0.2034 (0.46) | **torch-bf16-compile-cg 0.1492 (0.63)** | torch-bf16-compile-ro 0.1497 (0.63) | torch-bf16-cudnn 0.3787 (0.25)<br>torch-bf16-efficient 0.4417 (0.21) | fa2-bf16 0.2974 (0.32)<br>fa3-bf16 0.3505 (0.27) | flashinfer-bf16 0.4483 (0.21) |  | liger-bf16 0.3422 (0.28) |
| llm_dec_b1_l8192_bf16w | rtx-5080 | 0.2405 | torch-bf16 0.2184 (1.10) | torch-bf16-compile 0.1913 (1.26) | torch-bf16-compile-cg 0.1983 (1.21) | torch-bf16-compile-ro 1.006 (0.24) | torch-bf16-cudnn 0.1921 (1.25)<br>torch-bf16-efficient 0.8834 (0.27) | fa2-bf16 0.2003 (1.20) | flashinfer-bf16 0.2932 (0.82) |  | **liger-bf16 0.1883 (1.28)** |
| llm_dec_b1_l8192_bf16w | rtx-3090 | 0.2872 | torch-bf16 0.5512 (0.52) | **torch-bf16-compile 0.3039 (0.94)** | torch-bf16-compile-cg 0.3327 (0.86) | torch-bf16-compile-ro 1.361 (0.21) | torch-bf16-cudnn 0.5821 (0.49)<br>torch-bf16-efficient 1.234 (0.23) | fa2-bf16 0.5411 (0.53) | flashinfer-bf16 0.7978 (0.36) |  | liger-bf16 0.5103 (0.56) |
| llm_dec_b1_l8192_bf16w | a100 | 0.2314 | torch-bf16 0.259 (0.89) | **torch-bf16-compile 0.198 (1.17)** | torch-bf16-compile-cg 0.2045 (1.13) | torch-bf16-compile-ro 0.8473 (0.27) | torch-bf16-cudnn 0.2756 (0.84)<br>torch-bf16-efficient 0.8505 (0.27) | fa2-bf16 0.2648 (0.87) | flashinfer-bf16 0.4051 (0.57) |  | liger-bf16 0.2716 (0.85) |
| llm_dec_b1_l8192_bf16w | rtx-4090 | 0.2488 | torch-bf16 0.548 (0.45) | torch-bf16-compile 0.2695 (0.92) | **torch-bf16-compile-cg 0.2688 (0.93)** | torch-bf16-compile-ro 1.095 (0.23) | torch-bf16-cudnn 0.5911 (0.42)<br>torch-bf16-efficient 0.985 (0.25) | fa2-bf16 0.517 (0.48) | flashinfer-bf16 0.745 (0.33) |  | liger-bf16 0.4947 (0.50) |
| llm_dec_b1_l8192_bf16w | h100 | 0.1359 | torch-bf16 0.3432 (0.40) | torch-bf16-compile 0.2066 (0.66) | **torch-bf16-compile-cg 0.1558 (0.87)** | torch-bf16-compile-ro 0.1601 (0.85) | torch-bf16-cudnn 0.3825 (0.36)<br>torch-bf16-efficient 0.7729 (0.18) | fa2-bf16 0.3042 (0.45)<br>fa3-bf16 0.3512 (0.39) | flashinfer-bf16 0.5068 (0.27) |  | liger-bf16 0.341 (0.40) |
| llm_dec_b32_l2048_bf16w | rtx-5080 | 0.9551 | torch-bf16 0.5885 (1.62) | **torch-bf16-compile 0.4548 (2.10)** | torch-bf16-compile-cg 0.4687 (2.04) | torch-bf16-compile-ro 7.621 (0.13) | torch-bf16-cudnn 0.4881 (1.96)<br>torch-bf16-efficient 3.743 (0.26) | fa2-bf16 0.5237 (1.82) | flashinfer-bf16 0.5133 (1.86) |  | liger-bf16 0.5728 (1.67) |
| llm_dec_b32_l2048_bf16w | rtx-3090 | 1.192 | torch-bf16 1.137 (1.05) | torch-bf16-compile 0.8413 (1.42) | **torch-bf16-compile-cg 0.8302 (1.44)** | torch-bf16-compile-ro 8.987 (0.13) | torch-bf16-cudnn 0.8529 (1.40)<br>torch-bf16-efficient 5.32 (0.22) | fa2-bf16 0.8762 (1.36) | flashinfer-bf16 0.9709 (1.23) |  | liger-bf16 1.099 (1.08) |
| llm_dec_b32_l2048_bf16w | a100 | 0.866 | torch-bf16 0.5003 (1.73) | torch-bf16-compile 0.5477 (1.58) | torch-bf16-compile-cg 0.4838 (1.79) | torch-bf16-compile-ro 5.438 (0.16) | **torch-bf16-cudnn 0.4262 (2.03)**<br>torch-bf16-efficient 3.4 (0.25) | fa2-bf16 0.4566 (1.90) | flashinfer-bf16 0.4971 (1.74) |  | liger-bf16 0.4911 (1.76) |
| llm_dec_b32_l2048_bf16w | rtx-4090 | 0.9615 | torch-bf16 0.79 (1.22) | **torch-bf16-compile 0.4882 (1.97)** | torch-bf16-compile-cg 0.5141 (1.87) | torch-bf16-compile-ro 7.264 (0.13) | torch-bf16-cudnn 0.8159 (1.18)<br>torch-bf16-efficient 3.129 (0.31) | fa2-bf16 0.7711 (1.25) | flashinfer-bf16 0.9416 (1.02) |  | liger-bf16 0.7709 (1.25) |
| llm_dec_b32_l2048_bf16w | h100 | 0.4895 | torch-bf16 0.3979 (1.23) | torch-bf16-compile 0.2475 (1.98) | **torch-bf16-compile-cg 0.2324 (2.11)** | torch-bf16-compile-ro 0.2324 (2.11) | torch-bf16-cudnn 0.4243 (1.15)<br>torch-bf16-efficient 2.204 (0.22) | fa2-bf16 0.3768 (1.30)<br>fa3-bf16 0.4475 (1.09) | flashinfer-bf16 0.5132 (0.95) |  | liger-bf16 0.4097 (1.19) |
| llm_dec_b32_l8192_bf16w | rtx-5080 | 3.426 | torch-bf16 1.802 (1.90) | torch-bf16-compile 1.49 (2.30) | torch-bf16-compile-cg 1.477 (2.32) | torch-bf16-compile-ro 29.82 (0.11) | torch-bf16-cudnn 1.417 (2.42)<br>torch-bf16-efficient 14.28 (0.24) | fa2-bf16 1.555 (2.20) | **flashinfer-bf16 1.407 (2.43)** |  | liger-bf16 1.793 (1.91) |
| llm_dec_b32_l8192_bf16w | rtx-3090 | 4.256 | torch-bf16 3.225 (1.32) | torch-bf16-compile 3.676 (1.16) | torch-bf16-compile-cg 3.753 (1.13) | torch-bf16-compile-ro 36.28 (0.12) | **torch-bf16-cudnn 1.778 (2.39)**<br>torch-bf16-efficient 20.47 (0.21) | fa2-bf16 2.013 (2.11) | flashinfer-bf16 1.878 (2.27) |  | liger-bf16 3.146 (1.35) |
| llm_dec_b32_l8192_bf16w | a100 | 2.543 | torch-bf16 1.206 (2.11) | torch-bf16-compile 1.711 (1.49) | torch-bf16-compile-cg 1.727 (1.47) | torch-bf16-compile-ro 21.44 (0.12) | **torch-bf16-cudnn 0.975 (2.61)**<br>torch-bf16-efficient 12.97 (0.20) | fa2-bf16 1.098 (2.32) | flashinfer-bf16 1.059 (2.40) |  | liger-bf16 1.215 (2.09) |
| llm_dec_b32_l8192_bf16w | rtx-4090 | 3.404 | torch-bf16 1.613 (2.11) | torch-bf16-compile 1.399 (2.43) | **torch-bf16-compile-cg 1.398 (2.44)** | torch-bf16-compile-ro 27.72 (0.12) | torch-bf16-cudnn 1.63 (2.09)<br>torch-bf16-efficient 10.63 (0.32) | fa2-bf16 1.596 (2.13) | flashinfer-bf16 1.761 (1.93) |  | liger-bf16 1.593 (2.14) |
| llm_dec_b32_l8192_bf16w | h100 | 1.065 | torch-bf16 0.6546 (1.63) | torch-bf16-compile 0.5084 (2.10) | torch-bf16-compile-cg 0.4954 (2.15) | **torch-bf16-compile-ro 0.494 (2.16)** | torch-bf16-cudnn 0.6886 (1.55)<br>torch-bf16-efficient 7.848 (0.14) | fa2-bf16 0.631 (1.69)<br>fa3-bf16 0.708 (1.50) | flashinfer-bf16 0.7743 (1.38) |  | liger-bf16 0.6691 (1.59) |
| llm_dec_b8_l2048_bf16w | rtx-5080 | 0.339 | torch-bf16 0.2551 (1.33) | **torch-bf16-compile 0.217 (1.56)** | torch-bf16-compile-cg 0.2295 (1.48) | torch-bf16-compile-ro 1.928 (0.18) | torch-bf16-cudnn 0.2515 (1.35)<br>torch-bf16-efficient 1.163 (0.29) | fa2-bf16 0.2529 (1.34) | flashinfer-bf16 0.3125 (1.09) |  | liger-bf16 0.2431 (1.39) |
| llm_dec_b8_l2048_bf16w | rtx-3090 | 0.8274 | torch-bf16 0.5965 (1.39) | **torch-bf16-compile 0.356 (2.32)** | torch-bf16-compile-cg 0.3814 (2.17) | torch-bf16-compile-ro 2.421 (0.34) | torch-bf16-cudnn 0.6098 (1.36)<br>torch-bf16-efficient 1.678 (0.49) | fa2-bf16 0.6031 (1.37) | flashinfer-bf16 0.7429 (1.11) |  | liger-bf16 0.5866 (1.41) |
| llm_dec_b8_l2048_bf16w | a100 | 0.3439 | torch-bf16 0.2821 (1.22) | **torch-bf16-compile 0.2302 (1.49)** | torch-bf16-compile-cg 0.2425 (1.42) | torch-bf16-compile-ro 1.488 (0.23) | torch-bf16-cudnn 0.2858 (1.20)<br>torch-bf16-efficient 1.087 (0.32) | fa2-bf16 0.2923 (1.18) | flashinfer-bf16 0.3495 (0.98) |  | liger-bf16 0.3123 (1.10) |
| llm_dec_b8_l2048_bf16w | rtx-4090 | 0.3974 | torch-bf16 0.5761 (0.69) | **torch-bf16-compile 0.2776 (1.43)** | torch-bf16-compile-cg 0.3005 (1.32) | torch-bf16-compile-ro 1.932 (0.21) | torch-bf16-cudnn 0.6086 (0.65)<br>torch-bf16-efficient 1.193 (0.33) | fa2-bf16 0.5871 (0.68) | flashinfer-bf16 0.737 (0.54) |  | liger-bf16 0.5622 (0.71) |
| llm_dec_b8_l2048_bf16w | h100 | 0.1901 | torch-bf16 0.3455 (0.55) | torch-bf16-compile 0.2062 (0.92) | **torch-bf16-compile-cg 0.166 (1.15)** | torch-bf16-compile-ro 0.1673 (1.14) | torch-bf16-cudnn 0.3783 (0.50)<br>torch-bf16-efficient 0.8137 (0.23) | fa2-bf16 0.3157 (0.60)<br>fa3-bf16 0.3682 (0.52) | flashinfer-bf16 0.4491 (0.42) |  | liger-bf16 0.3651 (0.52) |
| llm_dec_b8_l8192_bf16w | rtx-5080 | 0.9781 | torch-bf16 0.492 (1.99) | torch-bf16-compile 0.485 (2.02) | torch-bf16-compile-cg 0.4918 (1.99) | torch-bf16-compile-ro 7.687 (0.13) | **torch-bf16-cudnn 0.4726 (2.07)**<br>torch-bf16-efficient 4.057 (0.24) | fa2-bf16 0.4845 (2.02) | flashinfer-bf16 0.5405 (1.81) |  | liger-bf16 0.4883 (2.00) |
| llm_dec_b8_l8192_bf16w | rtx-3090 | 1.607 | torch-bf16 0.8253 (1.95) | torch-bf16-compile 0.8666 (1.85) | torch-bf16-compile-cg 0.8371 (1.92) | torch-bf16-compile-ro 9.168 (0.18) | torch-bf16-cudnn 0.8334 (1.93)<br>torch-bf16-efficient 5.544 (0.29) | fa2-bf16 0.8282 (1.94) | flashinfer-bf16 0.9681 (1.66) |  | **liger-bf16 0.8092 (1.99)** |
| llm_dec_b8_l8192_bf16w | a100 | 0.7614 | **torch-bf16 0.4216 (1.81)** | torch-bf16-compile 0.535 (1.42) | torch-bf16-compile-cg 0.5453 (1.40) | torch-bf16-compile-ro 5.462 (0.14) | torch-bf16-cudnn 0.4289 (1.78)<br>torch-bf16-efficient 3.572 (0.21) | fa2-bf16 0.4308 (1.77) | flashinfer-bf16 0.4888 (1.56) |  | liger-bf16 0.4444 (1.71) |
| llm_dec_b8_l8192_bf16w | rtx-4090 | 0.9876 | torch-bf16 0.7835 (1.26) | **torch-bf16-compile 0.4954 (1.99)** | torch-bf16-compile-cg 0.5234 (1.89) | torch-bf16-compile-ro 7.15 (0.14) | torch-bf16-cudnn 0.8099 (1.22)<br>torch-bf16-efficient 3.069 (0.32) | fa2-bf16 0.7988 (1.24) | flashinfer-bf16 0.9357 (1.06) |  | liger-bf16 0.7735 (1.28) |
| llm_dec_b8_l8192_bf16w | h100 | 0.4534 | torch-bf16 0.4063 (1.12) | torch-bf16-compile 0.2514 (1.80) | **torch-bf16-compile-cg 0.2307 (1.96)** | torch-bf16-compile-ro 0.2314 (1.96) | torch-bf16-cudnn 0.4308 (1.05)<br>torch-bf16-efficient 2.22 (0.20) | fa2-bf16 0.3805 (1.19)<br>fa3-bf16 0.433 (1.05) | flashinfer-bf16 0.5178 (0.88) |  | liger-bf16 0.4133 (1.10) |

### Conv / FIR / elementwise

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand CUDA |
|---|---|---:|---|---|---|---|---|---|
| conv_2048_5 | rtx-5080 | 0.06647 |  | torch-eager 0.1536 (0.43) | torch-compile 0.1536 (0.43) | **torch-compile-cg 0.1516 (0.44)** | torch-compile-ro 0.1536 (0.43) |  |
| conv_2048_5 | rtx-3090 | 0.1094 |  | torch-eager 0.1782 (0.61) | **torch-compile 0.1608 (0.68)** | torch-compile-cg 0.1608 (0.68) | torch-compile-ro 0.1608 (0.68) |  |
| conv_2048_5 | a100 | 0.07205 |  | torch-eager 0.1382 (0.52) | torch-compile 0.1382 (0.52) | **torch-compile-cg 0.1372 (0.53)** | torch-compile-ro 0.1372 (0.53) |  |
| conv_2048_5 | rtx-4090 | 0.05751 |  | torch-eager 0.0951 (0.60) | torch-compile 0.0952 (0.60) | **torch-compile-cg 0.0932 (0.62)** | torch-compile-ro 0.0942 (0.61) |  |
| conv_2048_5 | h100 | 0.0451 |  | **torch-eager 0.0943 (0.48)** | torch-compile 0.0944 (0.48) | torch-compile-cg 0.0948 (0.48) | torch-compile-ro 0.0949 (0.48) |  |
| convmc_8_64_56_3 | rtx-5080 | 0.0809 |  | **torch-eager 0.1198 (0.68)** | torch-compile 0.1628 (0.50) | torch-compile-cg 0.1577 (0.51) | torch-compile-ro 0.1577 (0.51) |  |
| convmc_8_64_56_3 | rtx-3090 | 0.147 |  | **torch-eager 0.0768 (1.91)** | torch-compile 0.1659 (0.89) | torch-compile-cg 0.1577 (0.93) | torch-compile-ro 0.1587 (0.93) |  |
| convmc_8_64_56_3 | a100 | 0.193 |  | **torch-eager 0.087 (2.22)** | torch-compile 0.1823 (1.06) | torch-compile-cg 0.17 (1.14) | torch-compile-ro 0.1567 (1.23) |  |
| convmc_8_64_56_3 | rtx-4090 | 0.104 |  | **torch-eager 0.0573 (1.82)** | torch-compile 0.1065 (0.98) | torch-compile-cg 0.0942 (1.10) | torch-compile-ro 0.0952 (1.09) |  |
| convmc_8_64_56_3 | h100 | 0.1166 |  | **torch-eager 0.0717 (1.63)** | torch-compile 0.1122 (1.04) | torch-compile-cg 0.1061 (1.10) | torch-compile-ro 0.1052 (1.11) |  |
| ew_4096 | rtx-5080 | 0.08065 | chain-eager 0.5652 (0.14) | torch-eager 0.5734 (0.14) | torch-compile 0.2458 (0.33) | torch-compile-cg 0.2437 (0.33) | torch-compile-ro 0.2437 (0.33) | **chain-fused 0.2437 (0.33)** |
| ew_4096 | rtx-3090 | 0.1594 | chain-eager 0.5642 (0.28) | torch-eager 0.5642 (0.28) | torch-compile 0.2417 (0.66) | **torch-compile-cg 0.2406 (0.66)** | torch-compile-ro 0.2406 (0.66) | chain-fused 0.2417 (0.66) |
| ew_4096 | a100 | 0.1171 | chain-eager 0.3666 (0.32) | torch-eager 0.3594 (0.33) | **torch-compile 0.1536 (0.76)** | torch-compile-cg 0.1546 (0.76) | torch-compile-ro 0.1556 (0.75) | chain-fused 0.1608 (0.73) |
| ew_4096 | rtx-4090 | 0.09937 | chain-eager 0.5048 (0.20) | torch-eager 0.51 (0.19) | torch-compile 0.2202 (0.45) | torch-compile-cg 0.2212 (0.45) | torch-compile-ro 0.2202 (0.45) | **chain-fused 0.2202 (0.45)** |
| ew_4096 | h100 | 0.0856 | chain-eager 0.2003 (0.43) | torch-eager 0.1701 (0.50) | **torch-compile 0.0731 (1.17)** | torch-compile-cg 0.0738 (1.16) | torch-compile-ro 0.0737 (1.16) | chain-fused 0.0833 (1.03) |
| ewconvew_2048_5 | rtx-5080 | 0.07747 |  | torch-eager 0.2079 (0.37) | **torch-compile 0.1884 (0.41)** | torch-compile-cg 0.1884 (0.41) | torch-compile-ro 0.1905 (0.41) |  |
| ewconvew_2048_5 | rtx-3090 | 0.1301 |  | torch-eager 0.2949 (0.44) | torch-compile 0.2621 (0.50) | **torch-compile-cg 0.2601 (0.50)** | torch-compile-ro 0.2601 (0.50) |  |
| ewconvew_2048_5 | a100 | 0.088 |  | torch-eager 0.1956 (0.45) | torch-compile 0.1761 (0.50) | **torch-compile-cg 0.1731 (0.51)** | torch-compile-ro 0.1741 (0.51) |  |
| ewconvew_2048_5 | rtx-4090 | 0.06835 |  | torch-eager 0.1413 (0.48) | torch-compile 0.1321 (0.52) | **torch-compile-cg 0.1311 (0.52)** | torch-compile-ro 0.1311 (0.52) |  |
| ewconvew_2048_5 | h100 | 0.05784 |  | torch-eager 0.1254 (0.46) | torch-compile 0.1164 (0.50) | **torch-compile-cg 0.1142 (0.51)** | torch-compile-ro 0.1143 (0.51) |  |
| fir_4194304_16 | rtx-5080 | 0.04431 |  | torch-eager 0.1044 (0.42) | torch-compile 0.1044 (0.42) | **torch-compile-cg 0.1024 (0.43)** | torch-compile-ro 0.1044 (0.42) |  |
| fir_4194304_16 | rtx-3090 | 0.0738 |  | torch-eager 0.1137 (0.65) | torch-compile 0.1137 (0.65) | **torch-compile-cg 0.1126 (0.66)** | torch-compile-ro 0.1126 (0.66) |  |
| fir_4194304_16 | a100 | 0.04992 |  | torch-eager 0.086 (0.58) | torch-compile 0.086 (0.58) | **torch-compile-cg 0.085 (0.59)** | torch-compile-ro 0.085 (0.59) |  |
| fir_4194304_16 | rtx-4090 | 0.04025 |  | torch-eager 0.0645 (0.62) | torch-compile 0.0645 (0.62) | torch-compile-cg 0.0627 (0.64) | **torch-compile-ro 0.0625 (0.64)** |  |
| fir_4194304_16 | h100 | 0.03304 |  | **torch-eager 0.0607 (0.54)** | torch-compile 0.0608 (0.54) | torch-compile-cg 0.0608 (0.54) | torch-compile-ro 0.061 (0.54) |  |
| firew_rowsum_2048x2048_16 | rtx-5080 | 0.05244 |  | torch-eager 0.128 (0.41) | torch-compile 0.1106 (0.47) | torch-compile-cg 0.1106 (0.47) | **torch-compile-ro 0.1085 (0.48)** |  |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1015 |  | torch-eager 0.2222 (0.46) | torch-compile 0.1372 (0.74) | **torch-compile-cg 0.1362 (0.75)** | torch-compile-ro 0.1362 (0.75) |  |
| firew_rowsum_2048x2048_16 | a100 | 0.06162 |  | torch-eager 0.1311 (0.47) | torch-compile 0.0973 (0.63) | torch-compile-cg 0.0952 (0.65) | **torch-compile-ro 0.0942 (0.65)** |  |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.04683 |  | torch-eager 0.0922 (0.51) | torch-compile 0.0686 (0.68) | **torch-compile-cg 0.0666 (0.70)** | torch-compile-ro 0.0666 (0.70) |  |
| firew_rowsum_2048x2048_16 | h100 | 0.03877 |  | torch-eager 0.0869 (0.45) | torch-compile 0.0663 (0.58) | **torch-compile-cg 0.0653 (0.59)** | torch-compile-ro 0.0656 (0.59) |  |
| convmc_8_64_56_3_tf32 | rtx-5080 | 0.08029 |  | **torch-eager-tf32 0.0614 (1.31)** | torch-compile-tf32 0.0922 (0.87) | torch-compile-cg-tf32 0.0922 (0.87) | torch-compile-ro-tf32 0.0901 (0.89) |  |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1485 |  | **torch-eager-tf32 0.084 (1.77)** | torch-compile-tf32 0.1444 (1.03) | torch-compile-cg-tf32 0.1372 (1.08) | torch-compile-ro-tf32 0.1393 (1.07) |  |
| convmc_8_64_56_3_tf32 | a100 | 0.1943 |  | **torch-eager-tf32 0.0471 (4.13)** | torch-compile-tf32 0.084 (2.31) | torch-compile-cg-tf32 0.0748 (2.60) | torch-compile-ro-tf32 0.0748 (2.60) |  |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.1045 |  | **torch-eager-tf32 0.043 (2.43)** | torch-compile-tf32 0.0798 (1.31) | torch-compile-cg-tf32 0.0758 (1.38) | torch-compile-ro-tf32 0.0788 (1.33) |  |
| convmc_8_64_56_3_tf32 | h100 | 0.1164 |  | **torch-eager-tf32 0.0379 (3.07)** | torch-compile-tf32 0.0496 (2.35) | torch-compile-cg-tf32 0.0459 (2.54) | torch-compile-ro-tf32 0.0442 (2.63) |  |

### Fan-in/out, loops, CPU+GPU

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead |
|---|---|---:|---|---|---|---|---|
| diamond_2048 | rtx-5080 | 1.441 | **chain-lt 1.378 (1.05)**<br>chain-streams 1.397 (1.03)<br>chain-eager 1.405 (1.03)<br>chain-cutlass 1.647 (0.88) | torch-eager 1.384 (1.04) | torch-compile 1.382 (1.04) | torch-compile-cg 1.382 (1.04) | torch-compile-ro 1.397 (1.03) |
| diamond_2048 | rtx-3090 | 2.415 | **chain-streams 2.79 (0.87)**<br>chain-lt 2.82 (0.86)<br>chain-eager 2.833 (0.85)<br>chain-cutlass 3.308 (0.73) | torch-eager 3.032 (0.80) | torch-compile 2.901 (0.83) | torch-compile-cg 2.894 (0.83) | torch-compile-ro 2.915 (0.83) |
| diamond_2048 | a100 | 3.247 | **chain-lt 2.95 (1.10)**<br>chain-streams 2.954 (1.10)<br>chain-eager 2.983 (1.09)<br>chain-cutlass 3.465 (0.94) | torch-eager 2.986 (1.09) | torch-compile 3.795 (0.86) | torch-compile-cg 2.952 (1.10) | torch-compile-ro 2.969 (1.09) |
| diamond_2048 | rtx-4090 | 0.9712 | chain-cutlass 1.134 (0.86)<br>chain-lt 1.165 (0.83)<br>chain-streams 1.177 (0.83)<br>chain-eager 1.183 (0.82) | torch-eager 0.9841 (0.99) | torch-compile 0.9656 (1.01) | **torch-compile-cg 0.9626 (1.01)** | torch-compile-ro 0.9728 (1.00) |
| diamond_2048 | h100 | 1.166 | **chain-lt 1.029 (1.13)**<br>chain-streams 1.038 (1.12)<br>chain-eager 1.042 (1.12)<br>chain-cutlass 1.264 (0.92) | torch-eager 1.041 (1.12) | torch-compile 1.032 (1.13) | torch-compile-cg 1.031 (1.13) | torch-compile-ro 1.041 (1.12) |
| fanout_2048 | rtx-5080 | 0.9824 | **chain-streams 0.9112 (1.08)**<br>chain-eager 0.9236 (1.06) | torch-eager 0.938 (1.05) | torch-compile 0.9236 (1.06) | torch-compile-cg 0.9257 (1.06) | torch-compile-ro 0.9257 (1.06) |
| fanout_2048 | rtx-3090 | 1.633 | chain-streams 1.946 (0.84)<br>chain-eager 1.982 (0.82) | torch-eager 1.809 (0.90) | **torch-compile 1.724 (0.95)** | torch-compile-cg 2.021 (0.81) | torch-compile-ro 2.019 (0.81) |
| fanout_2048 | a100 | 2.252 | **chain-streams 1.966 (1.15)**<br>chain-eager 1.987 (1.13) | torch-eager 1.995 (1.13) | torch-compile 2.047 (1.10) | torch-compile-cg 2.046 (1.10) | torch-compile-ro 1.977 (1.14) |
| fanout_2048 | rtx-4090 | 0.6574 | **chain-streams 0.6573 (1.00)**<br>chain-eager 0.6625 (0.99) | torch-eager 0.6799 (0.97) | torch-compile 0.6748 (0.97) | torch-compile-cg 0.6758 (0.97) | torch-compile-ro 0.6574 (1.00) |
| fanout_2048 | h100 | 0.782 | **chain-streams 0.686 (1.14)**<br>chain-eager 0.6979 (1.12) | torch-eager 0.7036 (1.11) | torch-compile 0.6935 (1.13) | torch-compile-cg 0.6928 (1.13) | torch-compile-ro 0.6935 (1.13) |
| indep_2048 | rtx-5080 | 0.9995 | chain-lt 0.9216 (1.08)<br>chain-streams 0.938 (1.07)<br>chain-eager 0.9441 (1.06)<br>chain-cutlass 1.102 (0.91) | torch-eager 0.9298 (1.07) | **torch-compile 0.9175 (1.09)** | torch-compile-cg 0.9175 (1.09) | torch-compile-ro 0.9298 (1.07) |
| indep_2048 | rtx-3090 | 1.63 | chain-streams 1.866 (0.87)<br>chain-lt 1.899 (0.86)<br>chain-eager 1.908 (0.85)<br>chain-cutlass 2.207 (0.74) | torch-eager 2.041 (0.80) | torch-compile 2.007 (0.81) | torch-compile-cg 1.948 (0.84) | **torch-compile-ro 1.848 (0.88)** |
| indep_2048 | a100 | 2.245 | chain-lt 1.968 (1.14)<br>chain-streams 1.971 (1.14)<br>chain-eager 2.001 (1.12)<br>chain-cutlass 2.313 (0.97) | torch-eager 1.998 (1.12) | torch-compile 1.969 (1.14) | **torch-compile-cg 1.966 (1.14)** | torch-compile-ro 1.982 (1.13) |
| indep_2048 | rtx-4090 | 0.6509 | chain-cutlass 0.7117 (0.91)<br>chain-lt 0.7352 (0.89)<br>chain-streams 0.7424 (0.88)<br>chain-eager 0.7506 (0.87) | torch-eager 0.6676 (0.98) | torch-compile 0.6502 (1.00) | **torch-compile-cg 0.6492 (1.00)** | torch-compile-ro 0.6595 (0.99) |
| indep_2048 | h100 | 0.7914 | **chain-lt 0.6886 (1.15)**<br>chain-streams 0.6964 (1.14)<br>chain-eager 0.7011 (1.13)<br>chain-cutlass 0.8497 (0.93) | torch-eager 0.6995 (1.13) | torch-compile 0.6928 (1.14) | torch-compile-cg 0.6925 (1.14) | torch-compile-ro 0.6999 (1.13) |
| indep_512 | rtx-5080 | 0.03893 | chain-streams 0.0365 (1.07)<br>chain-lt 0.0451 (0.86)<br>chain-eager 0.0471 (0.83)<br>chain-cutlass 0.1044 (0.37) | torch-eager 0.0451 (0.86) | **torch-compile 0.0328 (1.19)** | torch-compile-cg 0.0328 (1.19) | torch-compile-ro 0.0451 (0.86) |
| indep_512 | rtx-3090 | 0.08198 | chain-streams 0.0512 (1.60)<br>chain-lt 0.0532 (1.54)<br>chain-eager 0.0553 (1.48)<br>chain-cutlass 0.1229 (0.67) | torch-eager 0.0645 (1.27) | torch-compile 0.0461 (1.78) | **torch-compile-cg 0.0451 (1.82)** | torch-compile-ro 0.0532 (1.54) |
| indep_512 | a100 | 0.07609 | **chain-streams 0.0573 (1.33)**<br>chain-lt 0.0655 (1.16)<br>chain-eager 0.0676 (1.13)<br>chain-cutlass 0.2079 (0.37) | torch-eager 0.0666 (1.14) | torch-compile 0.0788 (0.97) | torch-compile-cg 0.0788 (0.97) | torch-compile-ro 0.0809 (0.94) |
| indep_512 | rtx-4090 | 0.06089 | chain-lt 0.0369 (1.65)<br>chain-streams 0.0378 (1.61)<br>chain-eager 0.039 (1.56)<br>chain-cutlass 0.0901 (0.68) | torch-eager 0.044 (1.38) | **torch-compile 0.0287 (2.12)** | torch-compile-cg 0.0287 (2.12) | torch-compile-ro 0.042 (1.45) |
| indep_512 | h100 | 0.05957 | chain-streams 0.0315 (1.89)<br>chain-lt 0.0346 (1.72)<br>chain-eager 0.0377 (1.58)<br>chain-cutlass 0.1192 (0.50) | torch-eager 0.0369 (1.61) | **torch-compile 0.0284 (2.10)** | torch-compile-cg 0.0286 (2.08) | torch-compile-ro 0.0351 (1.70) |
| loop_4096 | rtx-5080 | 28.25 | **chain-lt 28.55 (0.99)**<br>chain-cutlass 28.9 (0.98)<br>chain-eager 30.19 (0.94) | torch-eager 29.28 (0.96) | torch-compile 29.17 (0.97) | torch-compile-cg 29.3 (0.96) | torch-compile-ro 29.22 (0.97) |
| loop_4096 | rtx-3090 | 48.69 | **chain-lt 52.87 (0.92)**<br>chain-cutlass 56.46 (0.86)<br>chain-eager 57.02 (0.85) | torch-eager 53.66 (0.91) | torch-compile 54.46 (0.89) | torch-compile-cg 54.84 (0.89) | torch-compile-ro 53.78 (0.91) |
| loop_4096 | a100 | 62.97 | chain-eager 59.11 (1.07)<br>chain-lt 60.43 (1.04)<br>chain-cutlass 61.07 (1.03) | torch-eager 59.02 (1.07) | torch-compile 61.2 (1.03) | torch-compile-cg 61.18 (1.03) | **torch-compile-ro 58.98 (1.07)** |
| loop_4096 | rtx-4090 | 18.68 | chain-cutlass 26.43 (0.71)<br>chain-lt 26.6 (0.70)<br>chain-eager 29.49 (0.63) | torch-eager 24.56 (0.76) | torch-compile 24.68 (0.76) | torch-compile-cg 24.76 (0.75) | **torch-compile-ro 24.52 (0.76)** |
| loop_4096 | h100 | 23.65 | **chain-lt 21.13 (1.12)**<br>chain-eager 21.61 (1.09)<br>chain-cutlass 26.94 (0.88) | torch-eager 21.62 (1.09) | torch-compile 21.61 (1.09) | torch-compile-cg 21.57 (1.10) | torch-compile-ro 21.57 (1.10) |
| loopmm_1024_8 | rtx-5080 | 0.649 |  | torch-eager 0.6615 (0.98) | torch-compile 0.5898 (1.10) | **torch-compile-cg 0.5878 (1.10)** | torch-compile-ro 0.6246 (1.04) |
| loopmm_1024_8 | rtx-3090 | 1.073 |  | torch-eager 1.057 (1.02) | torch-compile 0.9769 (1.10) | torch-compile-cg 0.9626 (1.12) | **torch-compile-ro 0.9267 (1.16)** |
| loopmm_1024_8 | a100 | 1.434 |  | **torch-eager 1.044 (1.37)** | torch-compile 1.356 (1.06) | torch-compile-cg 1.347 (1.07) | torch-compile-ro 1.347 (1.07) |
| loopmm_1024_8 | rtx-4090 | 0.4325 |  | torch-eager 0.4116 (1.05) | torch-compile 0.3635 (1.19) | **torch-compile-cg 0.3584 (1.21)** | torch-compile-ro 0.3584 (1.21) |
| loopmm_1024_8 | h100 | 0.5228 |  | torch-eager 0.4599 (1.14) | torch-compile 0.4561 (1.15) | **torch-compile-cg 0.4513 (1.16)** | torch-compile-ro 0.4553 (1.15) |
| rt_mlp2_1024 | rtx-5080 | 2.557 |  | torch-eager 3.356 (0.76) | **torch-compile 2.527 (1.01)** | torch-compile-cg 2.531 (1.01) | torch-compile-ro 2.616 (0.98) |
| rt_mlp2_1024 | rtx-3090 | 4.204 |  | torch-eager 7.308 (0.58) | torch-compile 4.981 (0.84) | **torch-compile-cg 4.935 (0.85)** | torch-compile-ro 5.437 (0.77) |
| rt_mlp2_1024 | a100 | 6.836 |  | torch-eager 5.46 (1.25) | torch-compile 5.63 (1.21) | **torch-compile-cg 4.293 (1.59)** | torch-compile-ro 4.33 (1.58) |
| rt_mlp2_1024 | rtx-4090 | 1.703 |  | torch-eager 2.615 (0.65) | torch-compile 1.85 (0.92) | **torch-compile-cg 1.588 (1.07)** | torch-compile-ro 1.599 (1.06) |
| rt_mlp2_1024 | h100 | 2.143 |  | torch-eager 2.923 (0.73) | torch-compile 2.017 (1.06) | torch-compile-cg 1.991 (1.08) | **torch-compile-ro 1.962 (1.09)** |
| rt_mlp2_2048 | rtx-5080 | 16.19 |  | torch-eager 19.28 (0.84) | torch-compile 15.55 (1.04) | **torch-compile-cg 15.53 (1.04)** | torch-compile-ro 15.65 (1.03) |
| rt_mlp2_2048 | rtx-3090 | 26.77 |  | torch-eager 40.18 (0.67) | **torch-compile 31.59 (0.85)** | torch-compile-cg 31.85 (0.84) | torch-compile-ro 32.11 (0.83) |
| rt_mlp2_2048 | a100 | 37.18 |  | torch-eager 38.32 (0.97) | torch-compile 31.86 (1.17) | **torch-compile-cg 31.75 (1.17)** | torch-compile-ro 31.92 (1.16) |
| rt_mlp2_2048 | rtx-4090 | 10.37 |  | torch-eager 15.69 (0.66) | torch-compile 13.37 (0.78) | torch-compile-cg 13.35 (0.78) | **torch-compile-ro 13.35 (0.78)** |
| rt_mlp2_2048 | h100 | 12.05 |  | torch-eager 14.41 (0.84) | **torch-compile 11.18 (1.08)** | torch-compile-cg 11.19 (1.08) | torch-compile-ro 11.26 (1.07) |
| twin_2048 | rtx-5080 | 0.9325 |  | torch-eager 0.9421 (0.99) | **torch-compile 0.9175 (1.02)** | torch-compile-cg 0.9175 (1.02) | torch-compile-ro 0.9175 (1.02) |
| twin_2048 | rtx-3090 | 1.605 |  | torch-eager 1.916 (0.84) | **torch-compile 1.7 (0.94)** | torch-compile-cg 2.068 (0.78) | torch-compile-ro 1.958 (0.82) |
| twin_2048 | a100 | 2.022 |  | torch-eager 1.966 (1.03) | torch-compile 1.965 (1.03) | **torch-compile-cg 1.962 (1.03)** | torch-compile-ro 1.962 (1.03) |
| twin_2048 | rtx-4090 | 0.648 |  | torch-eager 0.7055 (0.92) | torch-compile 0.6769 (0.96) | **torch-compile-cg 0.6758 (0.96)** | torch-compile-ro 0.6799 (0.95) |
| twin_2048 | h100 | 0.7818 |  | torch-eager 0.6856 (1.14) | torch-compile 0.6849 (1.14) | torch-compile-cg 0.696 (1.12) | **torch-compile-ro 0.6844 (1.14)** |

## Command

```
python3 benches/local/matrix.py --skip 'llm_kv[xn]_' \
  rtx-5080=.claude/raw-evidence/local-rtx-5080/model-scale/resA2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB1/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/model-scale/resB2/results-rtx-5080.txt@5fb1b3b0,.claude/raw-evidence/local-rtx-5080/onto-device/resE/results-rtx-5080.txt@8bba2a50,.claude/raw-evidence/local-rtx-5080/device-geometry/v3/model/results-rtx-5080.txt@5b6c0a29,.claude/raw-evidence/local-rtx-5080/large-gemm/resM/results-rtx-5080.txt@7a2b67bf,.claude/raw-evidence/local-rtx-5080/half-precision/run1/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/half-precision/run2/results-rtx-5080.txt@d1927ac1,.claude/raw-evidence/local-rtx-5080/long-prefill/resL/results-rtx-5080.txt@c8b83028,.claude/raw-evidence/local-rtx-5080/long-prefill/suite/results-rtx-5080.txt@c03a52a3,.claude/raw-evidence/local-rtx-5080/competitors/resS/results-rtx-5080.txt@f7a74cdb,.claude/raw-evidence/local-rtx-5080/competitors/resM/results-rtx-5080.txt@f7a74cdb,.claude/raw-evidence/local-rtx-5080/competitors/resM2/results-rtx-5080.txt@f7a74cdb \
  rtx-3090=.claude/raw-evidence/cross-arch/19a8a44e/rtx-3090/results-rtx-3090.txt@19a8a44e \
  a100=.claude/raw-evidence/cross-arch/19a8a44e/a100/results-a100.txt@19a8a44e \
  rtx-4090=.claude/raw-evidence/cross-arch/19a8a44e/rtx-4090/results-rtx-4090.txt@19a8a44e \
  h100=.claude/raw-evidence/cross-arch/19a8a44e/h100/results-h100.txt@19a8a44e
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
