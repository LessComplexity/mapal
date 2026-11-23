# Mapal vs the fastest competitor at the same precision

Built 2026-11-23 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations), `_bf16kv` rows (weights and KV cache bf16, f32 compute) and `_bf16` rows (bf16 storage, `: bf16 | f32` products) vs the bf16 legs (bf16 throughout); a bf16-storage row without bf16 legs of its own takes its twin's (`_bf16w`, else the f32 program: the same bf16 computation); `_f8w` rows (fp8 weights, f32 compute) vs fp8 legs, of which none is installed: `(no leg)`, their `_bf16w` twin's bf16 leg is a reference only. The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch's pick, `-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials' matmul / softmax / fused attention, autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. **Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang as servers. Every part ran the competitor legs (FlashAttention-3 on sm_90a only; `unavailable` = the part's run reported the package missing or its build capped).

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | integration | integration/resM-final@integration: 81; integration/resS-final@integration: 43; integration/keep-ab@integration: 3 |
| rtx-3090 | 3a8e654f | 3a8e654f/rtx-3090@3a8e654f: 120 |
| a100 | 3a8e654f | 3a8e654f/a100@3a8e654f: 120 |
| rtx-4090 | 3a8e654f | 3a8e654f/rtx-4090@3a8e654f: 120 |
| h100 | 3a8e654f | 3a8e654f/h100@3a8e654f: 120 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 1.115 (47) | 1.113 (47) | 1.383 (47) | 1.195 (47) | 1.647 (47) |
| MLP chains | 1.095 (6) | 1.133 (6) | 1.248 (6) | 1.205 (6) | 1.545 (6) |
| Attention / softmax / long rows | 0.883 (14) | 0.764 (14) | 0.992 (14) | 0.742 (14) | 0.986 (14) |
| LLM decode / prefill / block (f32, tf32, bf16 storage) | 1.211 (37) | 1.458 (37) | 2.285 (37) | 1.217 (37) | 2.101 (37) |
| Conv / FIR / elementwise | 0.549 (7) | 0.741 (7) | 0.950 (7) | 0.815 (7) | 0.876 (7) |
| Fan-in/out, loops, CPU+GPU | 1.081 (9) | 1.057 (9) | 1.205 (9) | 1.105 (9) | 1.165 (9) |
| all f32 rows | 0.924 (66) | 0.924 (66) | 1.040 (66) | 0.978 (66) | 1.051 (66) |
| all tf32 rows | 1.198 (13) | 1.172 (13) | 1.702 (13) | 1.356 (13) | 1.918 (13) |
| all bf16w rows | 1.861 (9) | 2.066 (9) | 3.559 (9) | 1.807 (9) | 3.168 (9) |
| all bf16kv rows | 1.252 (7) | 1.623 (7) | 2.990 (7) | 1.122 (7) | 2.900 (7) |
| all bf16 rows | 1.139 (25) | 1.356 (25) | 2.206 (25) | 1.146 (25) | 2.634 (25) |
| all f8w rows | - | - | - | - | - |
| **all rows** | 1.065 (120) | 1.127 (120) | 1.496 (120) | 1.106 (120) | 1.566 (120) |
| rows timed on every part (120) | 1.065 (120) | 1.127 (120) | 1.496 (120) | 1.106 (120) | 1.566 (120) |
| win / even / loss | 23/29/68 | 16/21/83 | 13/14/93 | 25/24/71 | 14/4/102 |

Geomean cells: value (rows). Parts time different row sets; compare columns on the "every part" row.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 1.09 4.07/3.748 (torch-eager) | **0.92** 5.326/5.771 (chain-lt) | 1.08 7.883/7.297 (chain-lt) | 1.01≈ 2.427/2.415 (torch-compile) | **0.92** 3.624/3.935 (chain-lt) |
| mm_16x2048x7168 | f32 | 1.10 0.0881/0.0799 (cublas) | 1.03≈ 0.0886/0.086 (cublas) | 1.01≈ 0.09215/0.0911 (cublas) ±7% | 1.13 0.1075/0.09525 (cublas) | 1.59 0.1017/0.06395 (cublas) |
| mm_4096x4096x4096 | f32 | 1.04≈ 3.819/3.659 (cublas) | 1.04≈ 6.274/6.034 (cublas) | 1.05 7.68/7.313 (cublas) | 0.97≈ 2.411/2.473 (cublas) | 1.15 4.769/4.144 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.08 0.8098/0.7496 (lt) | 0.98≈ 1.204/1.226 (torch-eager) | 0.99≈ 1.729/1.748 (torch-compile-ro) | 1.01≈ 0.661/0.6513 (lt-fresh) | 1.07 0.989/0.9224 (torch-compile) |
| pj_m128_n4096_k14336 | f32 | 1.14 0.4594/0.4014 (cublas-fresh) | 1.07 0.6789/0.6369 (torch-eager) | 1.08 0.9203/0.8509 (cublas-fresh) | 1.07 0.3792/0.3543 (lt-fresh) | 1.22 0.511/0.4181 (torch-compile-cg) |
| pj_m128_n4096_k4096 | f32 | 1.11 0.1501/0.1352 (cublas-fresh) | 1.07 0.2168/0.2028 (cublas) | 1.17 0.305/0.2601 (cublas) | 1.40 0.1359/0.0973 (lt-fresh) | 1.48 0.1749/0.1178 (cublas-fresh) ±5% |
| pj_m128_n6144_k4096 | f32 | 1.20 0.2164/0.1802 (cublas-fresh) | 1.03≈ 0.318/0.3092 (cublas-fresh) | 1.15 0.4302/0.3727 (lt) | 1.27 0.1942/0.1526 (lt-fresh) | 1.26 0.2376/0.1884 (cublas) |
| pj_m16_n18432_k7168 | f32 | 1.03≈ 0.6049/0.5878 (lt-fresh) | 1.02≈ 0.627/0.6124 (torch-compile) | 0.97≈ 0.43/0.4454 (torch-compile) | 1.06 0.6464/0.6103 (lt-fresh) | 0.97≈ 0.3269/0.3356 (torch-compile-cg) |
| pj_m16_n2048_k7168 | f32 | 1.03≈ 0.05084/0.0492 (cublas-fresh) | 1.17 0.09591/0.0819 (lt) | 1.16 0.1007/0.087 (cublas-fresh) | 2.11 0.06263/0.0297 (cublas-fresh) | 1.40 0.0683/0.0487 (torch-compile-ro) |
| pj_m16_n28672_k4096 | f32 | 1.03≈ 0.5392/0.5243 (lt-fresh) | 1.06 0.5762/0.5417 (torch-compile) | 0.99≈ 0.4215/0.426 (torch-compile) | 1.06 0.5827/0.5499 (cublas-fresh) | 1.08 0.2809/0.2602 (torch-compile) |
| pj_m16_n4096_k14336 | f32 | 0.97≈ 0.2662/0.2744 (torch-compile) | 1.06 0.2895/0.2734 (torch-compile-cg) | **0.91** 0.2276/0.2509 (lt) | 1.09 0.3279/0.3011 (cublas-fresh) | 0.98≈ 0.1651/0.1691 (torch-compile) |
| pj_m16_n4096_k4096 | f32 | 1.01≈ 0.05609/0.0553 (cublas-fresh) | 1.30 0.1102/0.085 (torch-compile-cg) | 1.13 0.1005/0.0891 (cublas-fresh) | 2.16 0.0707/0.0328 (lt-fresh) ±10% | 1.29 0.06357/0.0494 (torch-compile) ±5% |
| pj_m16_n6144_k4096 | f32 | 1.18 0.1042/0.0881 (lt-fresh) | 1.17 0.1428/0.1221 (torch-compile-cg) | 1.12 0.13/0.1157 (torch-compile-cg) | 1.38 0.1399/0.1016 (lt-fresh) | 1.34 0.07577/0.0567 (torch-compile) |
| pj_m1_n18432_k7168 | f32 | 0.98≈ 0.5959/0.6103 (torch-compile) | 1.00≈ 0.6078/0.6052 (lt) | 1.12 0.4381/0.3922 (lt-fresh) | 1.19 0.639/0.5376 (lt-fresh) | 1.11 0.2162/0.1956 (torch-compile) |
| pj_m1_n2048_k7168 | f32 | 1.36 0.03616/0.0266 (cublas-fresh) | 1.11 0.08139/0.0733 (lt-fresh) | 1.81 0.07979/0.044 (cublas-fresh) | 1.93 0.05146/0.0266 (lt-fresh) ±19% | 1.94 0.04453/0.0229 (cublas-fresh) |
| pj_m1_n28672_k4096 | f32 | 1.04≈ 0.5251/0.5038 (lt-fresh) | 1.01≈ 0.543/0.5366 (torch-compile-ro) | 1.14 0.3957/0.3471 (torch-compile) | 1.20 0.5735/0.4792 (lt-fresh) | 1.18 0.2018/0.1713 (torch-compile) |
| pj_m1_n4096_k14336 | f32 | 1.11 0.2575/0.2314 (cublas-fresh) | 1.02≈ 0.2799/0.2744 (torch-compile-ro) | 1.27 0.2254/0.1772 (lt-fresh) | 1.48 0.3008/0.2038 (lt-fresh) | 1.34 0.1098/0.0821 (cublas-fresh) |
| pj_m1_n4096_k4096 | f32 | 1.89 0.0388/0.0205 (lt-fresh) | 1.07 0.09101/0.085 (lt) | 1.54 0.08204/0.0532 (cublas-fresh) | 2.23 0.04346/0.0195 (lt-fresh) | 2.17 0.04837/0.0223 (cublas-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.74 0.08914/0.0512 (cublas-fresh) | 1.06 0.1284/0.1208 (torch-compile) | 1.32 0.1023/0.0778 (lt-fresh) | 1.77 0.1036/0.0584 (lt-fresh) | 2.04 0.07067/0.0347 (lt-fresh) |
| pj_m2048_n28672_k4096 | f32 | 0.96≈ 11.78/12.21 (lt-fresh) | 1.02≈ 19.22/18.85 (torch-eager) | 1.03≈ 26.86/26.02 (torch-compile-cg) | 1.09 8.956/8.183 (lt) | 1.17 16.2/13.87 (cublas-fresh) |
| pj_m2048_n4096_k14336 | f32 | 1.01≈ 6.45/6.413 (lt-fresh) | 1.11 10.99/9.916 (lt-fresh) | 1.04≈ 13.34/12.89 (torch-compile-ro) | 0.97≈ 4.078/4.185 (cublas) | 1.30 8.435/6.474 (torch-compile) |
| pj_m2048_n4096_k4096 | f32 | 1.04≈ 1.842/1.778 (torch-compile-cg) | 1.02≈ 2.875/2.829 (torch-compile-ro) | 1.04≈ 3.868/3.704 (torch-compile-ro) | 1.03≈ 1.192/1.158 (torch-compile-ro) | 1.08 1.827/1.688 (torch-compile) |
| pj_m2048_n6144_k4096 | f32 | 1.01≈ 2.651/2.615 (torch-compile-cg) | 1.11 4.547/4.099 (torch-compile-ro) | 1.00≈ 5.804/5.785 (torch-compile) | 1.02≈ 1.786/1.756 (torch-compile) | 0.96≈ 2.749/2.878 (torch-compile) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.18 0.6977/0.5898 (cublas-tf32-fresh) | 1.04≈ 0.9326/0.8929 (torch-eager-tf32) | 1.28 0.5329/0.4178 (triton-tf32) | 1.02≈ 0.5784/0.5693 (cublas-tf32-fresh) | 1.40 0.3591/0.2567 (lt-tf32) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.34 0.4157/0.3092 (cublas-tf32-fresh) | 1.14 0.6505/0.5714 (triton-tf32) | 2.02 0.4685/0.2314 (cublas-tf32) | 1.22 0.3755/0.3072 (lt-tf32-fresh) | 2.34 0.3411/0.1458 (torch-compile-tf32) |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.65 0.1387/0.084 (cublas-tf32-fresh) | 1.20 0.1851/0.154 (cublas-tf32) | 1.80 0.149/0.0829 (cublas-tf32-fresh) | 1.53 0.09878/0.0644 (cublas-tf32-fresh) | 2.10 0.09087/0.0433 (cublas-tf32-fresh) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.47 0.1899/0.129 (lt-tf32-fresh) | 1.51 0.3684/0.2437 (lt-tf32-fresh) | 1.63 0.1618/0.0993 (triton-tf32) | 1.57 0.1837/0.1167 (cublas-tf32-fresh) | 2.99 0.1563/0.0522 (lt-tf32-fresh) |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.12 9.042/8.059 (cublas-tf32-fresh) | 1.04≈ 13.22/12.72 (torch-eager-tf32) | 1.42 5.909/4.15 (lt-tf32-fresh) | 1.54 8.344/5.406 (torch-compile-cg-tf32) | 1.52 3.928/2.58 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.03≈ 4.356/4.241 (lt-tf32-fresh) | 1.06 7.285/6.896 (torch-compile-ro-tf32) | 1.31 2.71/2.071 (torch-compile-ro-tf32) | 1.17 3.178/2.718 (torch-compile-cg-tf32) | 1.55 1.643/1.057 (torch-eager-tf32) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.16 1.341/1.161 (lt-tf32-fresh) | 1.03≈ 2.095/2.035 (torch-compile-tf32) | 1.38 0.8024/0.5806 (cublas-tf32) | 1.13 0.9283/0.8243 (lt-tf32) | 1.40 0.4583/0.3284 (lt-tf32-fresh) |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.16 2.022/1.749 (cublas-tf32-fresh) | 1.06 2.986/2.821 (torch-compile-cg-tf32) | 1.54 1.248/0.809 (torch-eager-tf32) | 1.18 1.378/1.166 (torch-compile-ro-tf32) | 1.27 0.7186/0.5644 (lt-tf32) |
| pj_m128_n28672_k4096_bf16 | bf16 | 1.21 0.3844/0.3174 (triton-bf16) | 1.07 0.4946/0.4618 (triton-bf16) | 1.75 0.3812/0.2181 (triton-bf16) | 1.07 0.3384/0.3154 (torch-bf16-compile) | 2.01 0.2607/0.13 (torch-bf16-compile-ro) |
| pj_m128_n4096_k14336_bf16 | bf16 | 1.46 0.2487/0.17 (torch-bf16-compile-cg) | 1.40 0.3786/0.2714 (torch-bf16) | 2.71 0.3353/0.1239 (torch-bf16-compile-cg) | 1.24 0.2342/0.1884 (torch-bf16-compile-cg) | 3.77 0.2853/0.0757 (torch-bf16-compile-ro) |
| pj_m128_n4096_k4096_bf16 | bf16 | 1.45 0.07729/0.0532 (torch-bf16-compile-ro) | 1.35 0.1083/0.0799 (triton-bf16) | 2.06 0.1098/0.0532 (triton-bf16) | 1.13 0.06718/0.0593 (triton-bf16) | 3.03 0.06663/0.022 (torch-bf16) |
| pj_m128_n6144_k4096_bf16 | bf16 | 1.22 0.1073/0.0881 (torch-bf16) | 1.35 0.169/0.1249 (torch-bf16-compile-cg) | 1.86 0.1159/0.0625 (triton-bf16) | 1.17 0.09603/0.0819 (torch-bf16-compile) | 3.46 0.1056/0.0305 (torch-bf16) ±6% |
| pj_m16_n28672_k4096_bf16 | bf16 | 1.08 0.2966/0.2744 (torch-bf16-compile) | 1.10 0.3007/0.2724 (torch-bf16-compile) | 1.07 0.2059/0.1925 (torch-bf16-compile) | 1.05 0.3237/0.3082 (torch-bf16-compile) | 2.27 0.2044/0.0902 (torch-bf16-compile-cg) |
| pj_m16_n4096_k14336_bf16 | bf16 | **0.94** 0.1305/0.1393 (torch-bf16-compile) | 1.17 0.1658/0.1423 (torch-bf16-compile) | 2.94 0.3217/0.1096 (torch-bf16-compile) | **0.91** 0.1609/0.1772 (torch-bf16-compile) | 2.20 0.1181/0.0538 (torch-bf16-compile) |
| pj_m16_n4096_k4096_bf16 | bf16 | 1.23 0.05285/0.043 (torch-bf16-compile) | 1.57 0.07414/0.0471 (torch-bf16-compile) | 2.51 0.1078/0.043 (torch-bf16-compile-cg) | 1.12 0.06311/0.0563 (torch-bf16-compile-cg) ±7% | 2.50 0.05409/0.0216 (torch-bf16-compile) |
| pj_m16_n6144_k4096_bf16 | bf16 | **0.93** 0.05933/0.0635 (torch-bf16-compile) | 1.35 0.08774/0.0649 (torch-bf16-compile-cg) | 1.90 0.1128/0.0594 (torch-bf16-compile-cg) | 0.95≈ 0.0772/0.0809 (torch-bf16-compile-ro) ±8% | 2.24 0.06172/0.0276 (torch-bf16-compile) |
| pj_m1_n28672_k4096_bf16 | bf16 | **0.94** 0.2586/0.2744 (torch-bf16-compile) | 1.01≈ 0.2808/0.2785 (torch-bf16-compile) | 1.28 0.2346/0.1833 (torch-bf16-compile) | 1.15 0.3141/0.2724 (torch-bf16-compile) | 1.31 0.1197/0.0914 (torch-bf16) |
| pj_m1_n4096_k14336_bf16 | bf16 | **0.78** 0.1099/0.1413 (torch-bf16) | 1.04≈ 0.1535/0.1475 (torch-bf16-compile-cg) | 1.28 0.1373/0.1075 (torch-bf16-compile) | **0.81** 0.1214/0.1505 (torch-bf16-compile-cg) | 1.51 0.07918/0.0524 (torch-bf16-compile-cg) |
| pj_m1_n4096_k4096_bf16 | bf16 | **0.77** 0.03644/0.0471 (torch-bf16-compile-cg) ±14% | 1.18 0.05816/0.0492 (torch-bf16-compile-cg) | 1.07 0.04823/0.0451 (torch-bf16-compile-cg) | **0.55** 0.03053/0.0553 (torch-bf16-compile-cg) | 1.58 0.03195/0.0202 (torch-bf16-compile-ro) |
| pj_m1_n6144_k4096_bf16 | bf16 | **0.54** 0.03456/0.0635 (torch-bf16-compile) | 1.10 0.07446/0.0676 (torch-bf16-compile) | 1.38 0.07335/0.0532 (torch-bf16-compile-cg) | **0.72** 0.05672/0.0788 (torch-bf16-compile-cg) ±16% | 1.61 0.0484/0.03 (torch-bf16) |
| pj_m2048_n28672_k4096_bf16 | bf16 | 1.18 4.665/3.963 (torch-bf16-compile-cg) | 1.08 6.801/6.283 (torch-bf16-compile-cg) | 1.80 3.509/1.954 (torch-bf16-compile-cg) | 1.58 4.343/2.748 (torch-bf16-compile-ro) | 3.13 3.967/1.267 (torch-bf16) |
| pj_m2048_n4096_k14336_bf16 | bf16 | 1.09 2.283/2.097 (torch-bf16-compile-cg) | 1.12 3.976/3.556 (triton-bf16) | 2.17 1.966/0.9052 (torch-bf16-compile-ro) | 1.22 1.676/1.377 (triton-bf16) | 2.69 1.671/0.6216 (torch-bf16) |
| pj_m2048_n4096_k4096_bf16 | bf16 | 1.19 0.7108/0.598 (torch-bf16-compile-cg) | 1.09 1.113/1.023 (triton-bf16) | 1.94 0.5995/0.3092 (torch-bf16) | 1.27 0.5076/0.4004 (triton-bf16) | 2.85 0.4782/0.1681 (torch-bf16) |
| pj_m2048_n6144_k4096_bf16 | bf16 | 1.22 1.08/0.8865 (torch-bf16-compile-ro) | 1.12 1.538/1.369 (torch-bf16-compile-ro) | 1.99 0.8165/0.4096 (torch-bf16-compile-cg) | 1.25 0.7327/0.5857 (torch-bf16-compile) | 2.77 0.737/0.2662 (torch-bf16) |
| geomean f32 | | 1.120 (23) | 1.061 (23) | 1.122 (23) | 1.280 (23) | 1.269 (23) |
| geomean tf32 | | 1.250 (8) | 1.126 (8) | 1.530 (8) | 1.280 (8) | 1.746 (8) |
| geomean bf16 | | 1.047 (16) | 1.185 (16) | 1.776 (16) | 1.045 (16) | 2.327 (16) |
| **geomean** | | 1.115 (47) | 1.113 (47) | 1.383 (47) | 1.195 (47) | 1.647 (47) |
| win / even / loss | | 6/13/28 | 1/15/31 | 1/8/38 | 4/8/35 | 1/3/43 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.11 1.943/1.749 (chain-lt2) | 1.06 3.142/2.975 (chain-lt2) | 1.17 4.401/3.759 (chain-streams) | 0.98≈ 1.205/1.227 (chain-lt2) | 1.06 1.893/1.793 (torch-compile) |
| mlp_2048 | f32 | 1.04≈ 0.9459/0.9114 (chain-lt) | 1.09 1.584/1.449 (torch-eager) | 1.13 2.223/1.964 (chain-lt) | 0.96≈ 0.589/0.6113 (chain-cutlass) | 1.10 0.9285/0.8412 (torch-compile-ro) |
| mlp_512 | f32 | 1.24 0.04079/0.0328 (torch-compile) | 1.38 0.05491/0.0399 (triton) | 1.24 0.07764/0.0625 (chain-lt) | 2.08 0.05532/0.0266 (torch-compile-cg) | 1.76 0.053/0.0301 (torch-compile-cg) |
| smlp_4096_256_1024 | f32 | 1.12 0.1672/0.1495 (torch-compile) | 1.22 0.2687/0.2212 (chain-cutlass) | 1.18 0.3459/0.2929 (chain-eager) | 1.10 0.1131/0.1026 (triton) | 1.33 0.1666/0.1248 (chain-lt) |
| mlp2_2048_tf32 | tf32 | 1.04≈ 1.572/1.513 (torch-compile-ro-tf32) | 1.04≈ 2.443/2.353 (torch-compile-ro-tf32) | 1.34 0.9325/0.6984 (torch-compile-cg-tf32) | 1.20 0.9832/0.8172 (torch-compile-cg-tf32) | 2.13 0.7855/0.3685 (torch-compile-ro-tf32) |
| mlp_2048_tf32 | tf32 | 1.04≈ 0.787/0.7598 (torch-compile-ro-tf32) | 1.05 1.207/1.145 (torch-compile-ro-tf32) | 1.45 0.5092/0.3502 (torch-compile-cg-tf32) | 1.17 0.4813/0.4106 (torch-compile-ro-tf32) | 2.33 0.4059/0.1742 (torch-compile-ro-tf32) |
| geomean f32 | | 1.125 (4) | 1.179 (4) | 1.181 (4) | 1.214 (4) | 1.287 (4) |
| geomean tf32 | | 1.037 (2) | 1.046 (2) | 1.393 (2) | 1.188 (2) | 2.229 (2) |
| **geomean** | | 1.095 (6) | 1.133 (6) | 1.248 (6) | 1.205 (6) | 1.545 (6) |
| win / even / loss | | 0/3/3 | 0/1/5 | 0/0/6 | 0/2/4 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.50 0.04307/0.0287 (torch-compile) | 1.70 0.07689/0.0451 (torch-compile-ro) | 1.18 0.064/0.0543 (torch-sdpa-compile-cg) | 1.87 0.05166/0.0276 (torch-sdpa-compile-cg) | 2.18 0.05382/0.0247 (torch-compile-cg) |
| attn_2048_128 | f32 | 1.27 0.12/0.0942 (torch-sdpa-compile) | 1.37 0.2406/0.1761 (torch-sdpa-compile-cg) | 1.50 0.2609/0.1741 (torch-compile) | 1.19 0.09487/0.0799 (torch-compile-ro) | 1.49 0.1267/0.0848 (torch-compile-cg) |
| attn_2048_64 | f32 | 1.18 0.08184/0.0696 (torch-compile-ro) | 1.17 0.1511/0.1293 (torch-compile-ro) | 1.41 0.1555/0.1106 (torch-compile) | 1.23 0.07445/0.0604 (torch-compile-ro) | 1.48 0.08674/0.0586 (torch-compile-ro) |
| causal_1024_256 | f32 | **0.36** 15.37/42.76 (torch-compile-cg) | **0.14** 21.52/155.2 (torch-compile-cg) | **0.50** 36.87/74.15 (torch-compile) | **0.10** 17.94/177 (torch-compile-cg) | **0.18** 21.32/116 (torch-compile-ro) |
| decode_1024_256 | f32 | **0.53** 8.729/16.44 (chain-cache) | **0.75** 14.77/19.62 (chain-cache) | 0.96≈ 28.05/29.33 (chain-cache) | **0.41** 12.46/30.29 (chain-cache) | 1.45 28.57/19.71 (chain-cache) |
| gen_1024_512_256 | f32 | **0.32** 7.762/24.48 (torch-compile-cg) | **0.14** 10.86/77.58 (torch-compile-cg) | **0.50** 18.76/37.22 (torch-compile-cg) | **0.10** 9.14/88.22 (torch-compile-cg) | **0.19** 10.8/56.81 (torch-compile-cg) |
| lsm_1_131072 | f32 | **0.92** 0.02269/0.0246 (torch-compile-ro) | 0.96≈ 0.02453/0.0256 (torch-compile-cg) | 0.97≈ 0.02483/0.0256 (torch-compile-cg) | 1.18 0.02793/0.0236 (torch-compile-cg) | 1.69 0.03181/0.0188 (torch-compile-cg) ±15% |
| lsm_32_8192 | f32 | 2.37 0.02416/0.0102 (torch-compile-cg) | 2.56 0.02351/0.0092 (torch-compile-cg) | 2.25 0.02762/0.0123 (torch-compile-cg) | 5.09 0.0417/0.0082 (torch-compile-cg) | 2.78 0.02699/0.0097 (torch-compile-ro) |
| lsm_4_32768 | f32 | 1.49 0.02134/0.0143 (torch-compile-cg) | 1.13 0.01861/0.0164 (torch-compile-cg) | 1.11 0.02164/0.0195 (torch-compile-cg) | 2.01 0.02672/0.0133 (torch-compile-cg) | 1.79 0.02757/0.0154 (torch-sdpa-compile-cg) ±19% |
| prefill_1024_512_256 | f32 | **0.33** 8.345/24.98 (torch-compile-cg) | **0.14** 11.31/78.32 (torch-compile-ro) | **0.51** 18.97/37.31 (torch-compile-cg) | **0.10** 9.319/89.74 (torch-compile-cg) | **0.19** 10.95/57.04 (torch-compile-ro) |
| rowsum_2048 | f32 | 1.00≈ 0.4651/0.4628 (chain-eager) | 1.08 0.8224/0.7619 (torch-compile) | 1.04≈ 1.035/0.9943 (torch-compile-ro) | 0.99≈ 0.3055/0.3092 (torch-compile-ro) | 1.21 0.5099/0.4199 (torch-compile-cg) |
| rowsum_4096 | f32 | 1.07≈ 3.738/3.498 (torch-compile-cg) ±12% | 1.07 6.163/5.743 (torch-compile-ro) | 1.06 7.749/7.336 (torch-compile-ro) | 1.02≈ 2.426/2.375 (torch-compile-cg) | 1.15 4.606/4.006 (torch-compile-ro) |
| softmax_rows_2048 | f32 | 1.02≈ 0.4801/0.4707 (torch-compile-cg) | 1.16 0.8558/0.7404 (torch-compile) | 1.05≈ 1.052/1.004 (torch-compile) | 1.01≈ 0.3148/0.3132 (torch-compile) | 1.22 0.5219/0.4272 (torch-compile-ro) |
| softmax_rows_4096 | f32 | 1.08≈ 3.838/3.547 (torch-compile-ro) ±11% | 1.09 6.344/5.821 (torch-compile) | 1.06 7.816/7.376 (torch-compile-ro) | 1.02≈ 2.487/2.428 (torch-compile-cg) | 1.22 4.639/3.8 (torch-compile) |
| **geomean** | | 0.883 (14) | 0.764 (14) | 0.992 (14) | 0.742 (14) | 0.986 (14) |
| win / even / loss | | 5/4/5 | 4/1/9 | 3/4/7 | 4/4/6 | 3/0/11 |

## LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 1.00≈ 1.073/1.07 (torch-compile) | 1.15 1.38/1.203 (torch-compile-cg) | 1.52 1.251/0.8228 (torch-compile) | 0.96≈ 1.044/1.088 (torch-compile) | 1.40 0.7931/0.5657 (torch-compile) |
| llm_blkpre_t2048 | f32 | **0.95** 25.38/26.84 (liger) | 1.10 45.42/41.26 (liger) | 1.02≈ 55.87/54.53 (torch-compile) | 1.05≈ 18.8/17.95 (liger) | 1.06 29.74/27.99 (liger) |
| llm_dec_b1_l2048 | f32 | **0.93** 0.2407/0.2589 (torch-compile) | **0.94** 0.3259/0.3454 (torch-compile) | 1.88 0.4713/0.2511 (torch-compile) | **0.77** 0.2429/0.3161 (torch-compile) | 1.57 0.3256/0.2068 (torch-compile-cg) |
| llm_dec_b1_l8192 | f32 | 0.97≈ 0.3324/0.3439 (torch-compile) | **0.95** 0.4267/0.4502 (torch-compile) | 1.34 0.5297/0.3951 (torch-compile) | **0.92** 0.3452/0.3747 (torch-compile) | 1.12 0.3661/0.3272 (torch-compile) |
| llm_dec_b32_l2048 | f32 | **0.46** 1.033/2.262 (torch-compile) | **0.46** 1.213/2.658 (torch-compile) | **0.47** 0.8597/1.835 (torch-compile-cg) | **0.48** 1.127/2.329 (torch-compile) | **0.39** 0.5586/1.417 (torch-compile-cg) |
| llm_dec_b32_l8192 | f32 | **0.34** 3.515/10.27 (torch-compile) | **0.39** 4.278/10.85 (torch-compile) | **0.39** 2.534/6.513 (torch-compile-cg) | **0.40** 3.76/9.407 (torch-compile) | **0.27** 1.48/5.389 (torch-compile-cg) |
| llm_dec_b8_l2048 | f32 | **0.60** 0.4103/0.6816 (torch-compile-cg) | 1.03≈ 0.9218/0.8917 (torch-compile) | **0.64** 0.3983/0.6182 (torch-compile) | **0.67** 0.5001/0.7461 (torch-compile) | **0.44** 0.2183/0.4996 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.37** 1.069/2.856 (torch-compile-cg) | **0.58** 1.702/2.925 (torch-compile) | **0.45** 0.8105/1.807 (torch-compile) | **0.44** 1.128/2.56 (torch-compile) | **0.35** 0.5075/1.437 (torch-compile-cg) |
| llm_pre_t2048 | f32 | 1.00≈ 7.222/7.219 (liger) ±7% | 1.22 14.05/11.48 (liger) | 1.25 15.14/12.1 (liger) | 1.25 6.305/5.064 (liger) | 1.30 8.564/6.565 (liger) |
| llm_pre_t8192 | f32 | 1.69 89.55/52.91 (liger) ±6% | 1.51 129.9/85.87 (liger) | 2.17 132.3/61.06 (liger) | 1.84 67.87/36.84 (liger) | 1.81 71.33/39.46 (liger) |
| llm_blkpre_t2048_tf32 | tf32 | 1.10 19.9/18.07 (triton-tf32) | 1.20 33.35/27.78 (triton-tf32) | 1.67 16.37/9.807 (triton-tf32) | 1.25 16.54/13.24 (triton-tf32) | 1.64 10.38/6.315 (liger-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.34 6.072/4.543 (triton-tf32) | 1.55 11.74/7.568 (triton-tf32) | 2.62 7.419/2.829 (triton-tf32) | 1.79 5.669/3.17 (triton-tf32) | 2.04 4.453/2.188 (triton-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.15 0.6369/0.5556 (torch-bf16-compile) | 1.68 1.135/0.6744 (torch-bf16-compile) | 4.57 2.062/0.4513 (torch-bf16-compile-cg) | 1.03≈ 0.6472/0.6301 (torch-bf16-compile-cg) | 5.05 1.525/0.3021 (torch-bf16-compile-ro) |
| llm_blkpre_t2048_bf16w | bf16w | 3.19 25.67/8.043 (torch-bf16-compile-cg) | 3.28 43.99/13.41 (torch-bf16-compile-ro) | 12.53 56.07/4.476 (torch-bf16-compile-cg) | 3.29 18.39/5.594 (torch-bf16-compile-cg) | 9.92 27.72/2.794 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16w | bf16w | 1.03≈ 0.15/0.1455 (torch-bf16-compile) | 1.23 0.2866/0.2338 (torch-bf16-compile) | 2.84 0.4649/0.164 (torch-bf16-compile-cg) | **0.69** 0.1665/0.2425 (torch-bf16-compile-cg) | 2.19 0.3256/0.1484 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.35 0.2454/0.1814 (torch-bf16-compile) | 1.33 0.3873/0.2906 (torch-bf16-compile) | 2.66 0.5225/0.1961 (torch-bf16-compile) | 1.08 0.272/0.2525 (torch-bf16-compile) | 2.29 0.3666/0.16 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16w | bf16w | 2.05 0.9497/0.4622 (torch-bf16-compile) | 1.64 1.167/0.7098 (torch-bf16-compile) | 2.02 0.8662/0.4281 (torch-bf16-cudnn) | 2.00 1.022/0.5118 (torch-bf16-compile) | 2.07 0.5528/0.2671 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16w | bf16w | 2.34 3.438/1.47 (torch-bf16-cudnn) | 2.49 4.229/1.697 (torch-bf16-cudnn) | 2.60 2.538/0.9763 (torch-bf16-cudnn) | 2.62 3.659/1.398 (torch-bf16-compile) | 2.52 1.452/0.5752 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.41 0.3143/0.2221 (torch-bf16-compile) | 2.07 0.7375/0.3564 (torch-bf16-compile) | 1.53 0.3486/0.2275 (torch-bf16-compile) | 1.56 0.4149/0.2652 (torch-bf16-compile) | 1.11 0.1858/0.1675 (torch-bf16-compile-ro) |
| llm_dec_b8_l8192_bf16w | bf16w | 2.08 1.012/0.4862 (torch-bf16-compile) | 2.08 1.517/0.7305 (torch-bf16-compile) | 1.81 0.7605/0.4207 (torch-bf16) | 2.09 1.049/0.5009 (torch-bf16-compile) | 1.78 0.4798/0.2691 (torch-bf16-compile-ro) |
| llm_pre_t2048_bf16w | bf16w | 3.70 7.148/1.932 (torch-bf16-compile-cg) | 4.33 13.93/3.22 (torch-bf16-compile) | 14.54 15.15/1.041 (torch-bf16-compile-cg) | 4.81 6.23/1.294 (torch-bf16-compile-cg) | 12.34 8.006/0.649 (torch-bf16-compile-cg) |
| llm_blkdec_b1_l4096_bf16kv | bf16kv | 1.11 0.6157/0.5556 (torch-bf16-compile) | 1.70 1.147/0.6744 (torch-bf16-compile) | 4.63 2.089/0.4513 (torch-bf16-compile-cg) | 1.00≈ 0.6293/0.6301 (torch-bf16-compile-cg) | 5.09 1.536/0.3021 (torch-bf16-compile-ro) |
| llm_dec_b1_l2048_bf16kv | bf16kv | 0.97≈ 0.1404/0.1455 (torch-bf16-compile) | 1.25 0.2921/0.2338 (torch-bf16-compile) | 2.88 0.4721/0.164 (torch-bf16-compile-cg) | **0.66** 0.16/0.2425 (torch-bf16-compile-cg) | 2.23 0.3315/0.1484 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16kv | bf16kv | 1.09 0.1976/0.1814 (torch-bf16-compile) | 1.28 0.3717/0.2906 (torch-bf16-compile) | 2.86 0.5599/0.1961 (torch-bf16-compile) | **0.88** 0.2217/0.2525 (torch-bf16-compile) | 2.50 0.3999/0.16 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16kv | bf16kv | 1.41 0.6535/0.4622 (torch-bf16-compile) | 1.44 1.02/0.7098 (torch-bf16-compile) | 2.69 1.151/0.4281 (torch-bf16-cudnn) | 1.31 0.6686/0.5118 (torch-bf16-compile) | 3.37 0.9015/0.2671 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16kv | bf16kv | 1.64 2.415/1.47 (torch-bf16-cudnn) | 2.03 3.452/1.697 (torch-bf16-cudnn) | 3.69 3.603/0.9763 (torch-bf16-cudnn) | 1.57 2.2/1.398 (torch-bf16-compile) | 3.44 1.978/0.5752 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16kv | bf16kv | 1.08 0.2406/0.2221 (torch-bf16-compile) | 2.06 0.7325/0.3564 (torch-bf16-compile) | 2.27 0.5163/0.2275 (torch-bf16-compile) | 1.38 0.3658/0.2652 (torch-bf16-compile) | 1.68 0.2819/0.1675 (torch-bf16-compile-ro) |
| llm_dec_b8_l8192_bf16kv | bf16kv | 1.64 0.7981/0.4862 (torch-bf16-compile) ±12% | 1.82 1.328/0.7305 (torch-bf16-compile) | 2.50 1.05/0.4207 (torch-bf16) | 1.37 0.6857/0.5009 (torch-bf16-compile) | 3.11 0.8377/0.2691 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_bf16 | bf16 | 1.11 0.615/0.5556 (torch-bf16-compile) | 1.70 1.146/0.6744 (torch-bf16-compile) | 4.63 2.09/0.4513 (torch-bf16-compile-cg) | 1.00≈ 0.6273/0.6301 (torch-bf16-compile-cg) | 5.09 1.538/0.3021 (torch-bf16-compile-ro) |
| llm_blkpre_t2048_bf16 | bf16 | 1.44 11.62/8.043 (torch-bf16-compile-cg) | 1.55 20.72/13.41 (torch-bf16-compile-ro) | 2.87 12.86/4.476 (torch-bf16-compile-cg) | 1.83 10.26/5.594 (torch-bf16-compile-cg) | 3.61 10.07/2.794 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16 | bf16 | 0.96≈ 0.1401/0.1455 (torch-bf16-compile) | 1.25 0.2923/0.2338 (torch-bf16-compile) | 2.88 0.4718/0.164 (torch-bf16-compile-cg) | **0.66** 0.1602/0.2425 (torch-bf16-compile-cg) | 2.24 0.3318/0.1484 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16 | bf16 | 1.10 0.1996/0.1814 (torch-bf16-compile) | 1.29 0.3745/0.2906 (torch-bf16-compile) | 2.85 0.5595/0.1961 (torch-bf16-compile) | **0.88** 0.2233/0.2525 (torch-bf16-compile) | 2.48 0.3965/0.16 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16 | bf16 | 1.41 0.6536/0.4622 (torch-bf16-compile) | 1.44 1.02/0.7098 (torch-bf16-compile) | 2.66 1.138/0.4281 (torch-bf16-cudnn) | 1.31 0.669/0.5118 (torch-bf16-compile) | 3.51 0.9381/0.2671 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16 | bf16 | 1.63 2.393/1.47 (torch-bf16-cudnn) | 2.04 3.457/1.697 (torch-bf16-cudnn) | 3.67 3.586/0.9763 (torch-bf16-cudnn) | 1.57 2.199/1.398 (torch-bf16-compile) | 3.50 2.016/0.5752 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16 | bf16 | 1.08 0.2403/0.2221 (torch-bf16-compile) | 2.06 0.7329/0.3564 (torch-bf16-compile) | 2.27 0.5168/0.2275 (torch-bf16-compile) | 1.38 0.3664/0.2652 (torch-bf16-compile) | 1.67 0.2802/0.1675 (torch-bf16-compile-ro) |
| llm_dec_b8_l8192_bf16 | bf16 | 1.44 0.6996/0.4862 (torch-bf16-compile) | 1.82 1.328/0.7305 (torch-bf16-compile) | 2.50 1.051/0.4207 (torch-bf16) | 1.37 0.6864/0.5009 (torch-bf16-compile) | 3.11 0.8382/0.2691 (torch-bf16-compile-ro) |
| llm_pre_t2048_bf16 | bf16 | 2.04 3.942/1.932 (torch-bf16-compile-cg) ±11% | 2.91 9.37/3.22 (torch-bf16-compile) | 6.54 6.81/1.041 (torch-bf16-compile-cg) | 3.55 4.595/1.294 (torch-bf16-compile-cg) | 6.79 4.407/0.649 (torch-bf16-compile-cg) |
| llm_blkdec_b1_l4096_f8w | f8w | 0.3717 (no leg) | - | - | - | - |
| llm_dec_b1_l2048_f8w | f8w | 0.05575 (no leg) | - | - | - | - |
| llm_dec_b1_l8192_f8w | f8w | 0.1998 (no leg) | - | - | - | - |
| llm_dec_b32_l2048_f8w | f8w | 0.9476 (no leg) | - | - | - | - |
| llm_dec_b32_l8192_f8w | f8w | 3.538 (no leg) | - | - | - | - |
| llm_dec_b8_l2048_f8w | f8w | 0.2703 (no leg) | - | - | - | - |
| llm_dec_b8_l8192_f8w | f8w | 0.986 (no leg) | - | - | - | - |
| geomean f32 | | 0.742 (10) | 0.862 (10) | 0.944 (10) | 0.789 (10) | 0.797 (10) |
| geomean tf32 | | 1.213 (2) | 1.364 (2) | 2.092 (2) | 1.495 (2) | 1.829 (2) |
| geomean bf16w | | 1.861 (9) | 2.066 (9) | 3.559 (9) | 1.807 (9) | 3.168 (9) |
| geomean bf16kv | | 1.252 (7) | 1.623 (7) | 2.990 (7) | 1.122 (7) | 2.900 (7) |
| geomean bf16 | | 1.323 (9) | 1.724 (9) | 3.242 (9) | 1.349 (9) | 3.283 (9) |
| **geomean** | | 1.211 (37) | 1.458 (37) | 2.285 (37) | 1.217 (37) | 2.101 (37) |
| win / even / loss | | 6/6/25 | 5/1/31 | 4/1/32 | 11/5/21 | 4/0/33 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.46** 0.06927/0.1516 (torch-eager) | **0.59** 0.09289/0.1577 (torch-compile) | **0.52** 0.07186/0.1372 (torch-compile-cg) | **0.60** 0.05365/0.0901 (torch-compile-cg) | **0.44** 0.04278/0.0972 (torch-eager) |
| convmc_8_64_56_3 | f32 | **0.76** 0.08369/0.1096 (torch-eager) | 1.68 0.127/0.0758 (torch-eager) | 2.25 0.1958/0.087 (torch-eager) | 1.75 0.09697/0.0553 (torch-eager) | 1.70 0.1196/0.0702 (torch-eager) |
| ew_4096 | f32 | **0.38** 0.09179/0.2437 (chain-fused) | **0.43** 0.1039/0.2396 (torch-compile) | **0.77** 0.1182/0.1536 (torch-compile) | **0.46** 0.1044/0.2253 (chain-fused) | 1.11≈ 0.07908/0.0712 (torch-compile-cg) ±19% |
| ewconvew_2048_5 | f32 | **0.44** 0.08131/0.1864 (torch-compile-cg) | **0.47** 0.1128/0.2386 (torch-compile-cg) | **0.51** 0.08807/0.1731 (torch-compile-ro) | **0.51** 0.06402/0.126 (torch-compile-cg) | **0.47** 0.05446/0.1159 (torch-compile-cg) |
| fir_4194304_16 | f32 | **0.46** 0.04817/0.1044 (torch-eager) | **0.58** 0.0616/0.1055 (torch-eager) | **0.59** 0.04998/0.085 (torch-eager) | **0.61** 0.03793/0.0625 (torch-compile) | **0.53** 0.03144/0.0596 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.51** 0.05537/0.1085 (torch-compile) | **0.69** 0.08745/0.126 (torch-compile) | **0.66** 0.06182/0.0942 (torch-compile-ro) | **0.69** 0.04532/0.0655 (torch-compile-cg) | **0.58** 0.03754/0.0644 (torch-compile-ro) |
| convmc_8_64_56_3_tf32 | tf32 | 1.12 0.08357/0.0748 (torch-eager-tf32) | 1.50 0.1274/0.085 (torch-eager-tf32) | 3.93 0.1889/0.0481 (torch-eager-tf32) | 2.31 0.09696/0.042 (torch-eager-tf32) | 3.30 0.1192/0.0361 (torch-eager-tf32) |
| geomean f32 | | 0.488 (6) | 0.659 (6) | 0.750 (6) | 0.685 (6) | 0.703 (6) |
| geomean tf32 | | 1.117 (1) | 1.498 (1) | 3.927 (1) | 2.308 (1) | 3.301 (1) |
| **geomean** | | 0.549 (7) | 0.741 (7) | 0.950 (7) | 0.815 (7) | 0.876 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 5/0/2 | 5/0/2 | 4/1/2 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 1.06 1.45/1.368 (torch-compile-cg) | 1.00≈ 2.358/2.354 (chain-streams) | 1.10 3.247/2.945 (torch-compile-cg) | 0.99≈ 0.9067/0.9175 (torch-compile-cg) | 1.13 1.437/1.271 (torch-eager) |
| fanout_2048 | f32 | 1.08 0.9854/0.9133 (chain-streams) | 1.11 1.616/1.451 (torch-compile) | 1.15 2.255/1.966 (chain-streams) | 0.98≈ 0.6126/0.6226 (chain-streams) | 1.15 0.9683/0.8452 (torch-compile-cg) |
| indep_2048 | f32 | 1.09 1.001/0.9155 (chain-lt) | 1.05 1.571/1.494 (torch-compile) | 1.14 2.246/1.964 (torch-compile-cg) | 0.99≈ 0.6077/0.6144 (chain-cutlass) | 1.17 0.9889/0.8423 (torch-compile-cg) |
| indep_512 | f32 | 1.24 0.04073/0.0328 (torch-compile) | 1.22 0.05629/0.0461 (torch-compile) | 1.32 0.0757/0.0573 (chain-streams) | 2.14 0.05487/0.0256 (torch-compile-cg) | 2.09 0.05621/0.0269 (torch-compile) |
| loop_4096 | f32 | 0.99≈ 28.24/28.53 (chain-lt) | **0.93** 42.15/45.42 (chain-lt) | 1.07 62.96/59 (torch-compile-ro) | 0.96≈ 18.41/19.2 (chain-cutlass) | **0.87** 26.98/31.13 (chain-lt) |
| loopmm_1024_8 | f32 | 1.11 0.6523/0.5878 (torch-compile-cg) | 1.20 1.082/0.898 (torch-eager) | 1.37 1.434/1.044 (torch-eager) | 1.20 0.4096/0.342 (torch-compile-cg) | 1.26 0.624/0.4938 (torch-compile-cg) |
| rt_mlp2_1024 | f32 | 1.11 2.794/2.518 (torch-compile-cg) | 0.96≈ 3.964/4.125 (torch-compile-cg) | 1.59 6.835/4.299 (torch-compile-cg) | 1.12 1.624/1.448 (torch-compile-cg) | 1.07 2.612/2.435 (torch-compile-cg) |
| rt_mlp2_2048 | f32 | 1.04≈ 16.24/15.66 (torch-compile-ro) | 0.96≈ 25.51/26.61 (torch-compile-cg) | 1.17 37.17/31.75 (torch-compile-cg) | **0.94** 9.742/10.4 (torch-compile) | **0.93** 15.16/16.33 (torch-compile-ro) |
| twin_2048 | f32 | 1.02≈ 0.941/0.9185 (torch-compile-cg) | 1.11 1.583/1.422 (torch-compile-ro) | 1.03≈ 2.02/1.961 (torch-compile-ro) | 0.99≈ 0.603/0.6093 (torch-compile-cg) | 1.14 0.9591/0.8394 (torch-compile) |
| **geomean** | | 1.081 (9) | 1.057 (9) | 1.205 (9) | 1.105 (9) | 1.165 (9) |
| win / even / loss | | 0/3/6 | 1/3/5 | 0/1/8 | 1/5/3 | 2/0/7 |

## bf16 real-world reference

The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3174 (triton-bf16) · f32 2.55 | 0.4618 (triton-bf16) · f32 2.61 | 0.2181 (triton-bf16) · f32 7.93 | 0.3154 (bf16-compile) · f32 2.10 | 0.13 (bf16-compile-ro) · f32 7.61 |
| pj_m128_n4096_k14336 | 0.17 (bf16-compile-cg) · f32 2.70 | 0.2714 (bf16) · f32 2.50 | 0.1239 (bf16-compile-cg) · f32 7.43 | 0.1884 (bf16-compile-cg) · f32 2.01 | 0.0757 (bf16-compile-ro) · f32 6.75 |
| pj_m128_n4096_k4096 | 0.0532 (bf16-compile-ro) · f32 2.82 | 0.0799 (triton-bf16) · f32 2.71 | 0.0532 (triton-bf16) · f32 5.73 | 0.0593 (triton-bf16) · f32 2.29 | 0.022 (bf16) · f32 7.95 |
| pj_m128_n6144_k4096 | 0.0881 (bf16) · f32 2.46 | 0.1249 (bf16-compile-cg) · f32 2.55 | 0.0625 (triton-bf16) · f32 6.88 | 0.0819 (bf16-compile) · f32 2.37 | 0.0305 (bf16) · f32 7.79 |
| pj_m16_n18432_k7168 | 0.3092 (bf16-compile) · f32 1.96 | 0.3072 (bf16-compile) · f32 2.04 | 0.212 (triton-bf16) · f32 2.03 | 0.3389 (bf16-compile) · f32 1.91 | 0.1006 (bf16-compile) · f32 3.25 |
| pj_m16_n2048_k7168 | 0.0389 (bf16-compile) · f32 1.31 | 0.0471 (bf16-compile-cg) · f32 2.04 | 0.043 (bf16-compile-cg) · f32 2.34 | 0.0492 (bf16-compile) · f32 1.27 | 0.0199 (bf16-compile) · f32 3.43 |
| pj_m16_n28672_k4096 | 0.2744 (bf16-compile) · f32 1.96 | 0.2724 (bf16-compile) · f32 2.12 | 0.1925 (bf16-compile) · f32 2.19 | 0.3082 (bf16-compile) · f32 1.89 | 0.0902 (bf16-compile-cg) · f32 3.11 |
| pj_m16_n4096_k14336 | 0.1393 (bf16-compile) · f32 1.91 | 0.1423 (bf16-compile) · f32 2.03 | 0.1096 (bf16-compile) · f32 2.08 | 0.1772 (bf16-compile) · f32 1.85 | 0.0538 (bf16-compile) · f32 3.07 |
| pj_m16_n4096_k4096 | 0.043 (bf16-compile) · f32 1.30 | 0.0471 (bf16-compile) · f32 2.34 | 0.043 (bf16-compile-cg) · f32 2.34 | 0.0563 (bf16-compile-cg) · f32 1.26 | 0.0216 (bf16-compile) · f32 2.94 |
| pj_m16_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.64 | 0.0649 (bf16-compile-cg) · f32 2.20 | 0.0594 (bf16-compile-cg) · f32 2.19 | 0.0809 (bf16-compile-ro) · f32 1.73 | 0.0276 (bf16-compile) · f32 2.75 |
| pj_m1_n18432_k7168 | 0.3092 (bf16) · f32 1.93 | 0.3195 (bf16-compile) · f32 1.90 | 0.212 (bf16-compile) · f32 2.07 | 0.3013 (bf16-compile-cg) · f32 2.12 | 0.0999 (bf16) · f32 2.16 |
| pj_m1_n2048_k7168 | 0.043 (bf16-compile) · f32 0.84 | 0.0442 (bf16-compile) · f32 1.84 | 0.0348 (bf16-compile-cg) · f32 2.29 | 0.0502 (bf16-compile-cg) · f32 1.03 | 0.0192 (bf16-compile-cg) · f32 2.32 |
| pj_m1_n28672_k4096 | 0.2744 (bf16-compile) · f32 1.91 | 0.2785 (bf16-compile) · f32 1.95 | 0.1833 (bf16-compile) · f32 2.16 | 0.2724 (bf16-compile) · f32 2.11 | 0.0914 (bf16) · f32 2.21 |
| pj_m1_n4096_k14336 | 0.1413 (bf16) · f32 1.82 | 0.1475 (bf16-compile-cg) · f32 1.90 | 0.1075 (bf16-compile) · f32 2.10 | 0.1505 (bf16-compile-cg) · f32 2.00 | 0.0524 (bf16-compile-cg) · f32 2.10 |
| pj_m1_n4096_k4096 | 0.0471 (bf16-compile-cg) · f32 0.82 | 0.0492 (bf16-compile-cg) · f32 1.85 | 0.0451 (bf16-compile-cg) · f32 1.82 | 0.0553 (bf16-compile-cg) · f32 0.79 | 0.0202 (bf16-compile-ro) · f32 2.39 |
| pj_m1_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.40 | 0.0676 (bf16-compile) · f32 1.90 | 0.0532 (bf16-compile-cg) · f32 1.92 | 0.0788 (bf16-compile-cg) · f32 1.31 | 0.03 (bf16) · f32 2.36 |
| pj_m2048_n28672_k4096 | 3.963 (bf16-compile-cg) · f32 2.97 | 6.283 (bf16-compile-cg) · f32 3.06 | 1.954 (bf16-compile-cg) · f32 13.75 | 2.748 (bf16-compile-ro) · f32 3.26 | 1.267 (bf16) · f32 12.79 |
| pj_m2048_n4096_k14336 | 2.097 (bf16-compile-cg) · f32 3.08 | 3.556 (triton-bf16) · f32 3.09 | 0.9052 (bf16-compile-ro) · f32 14.74 | 1.377 (triton-bf16) · f32 2.96 | 0.6216 (bf16) · f32 13.57 |
| pj_m2048_n4096_k4096 | 0.598 (bf16-compile-cg) · f32 3.08 | 1.023 (triton-bf16) · f32 2.81 | 0.3092 (bf16) · f32 12.51 | 0.4004 (triton-bf16) · f32 2.98 | 0.1681 (bf16) · f32 10.87 |
| pj_m2048_n6144_k4096 | 0.8865 (bf16-compile-ro) · f32 2.99 | 1.369 (bf16-compile-ro) · f32 3.32 | 0.4096 (bf16-compile-cg) · f32 14.17 | 0.5857 (bf16-compile) · f32 3.05 | 0.2662 (bf16) · f32 10.33 |
| llm_blkdec_b1_l4096 | 0.5498 (bf16-compile) · f32 1.95 · bf16w 1.16 | 0.6757 (bf16-compile) · f32 2.04 · bf16w 1.68 | 0.4491 (bf16-compile) · f32 2.79 · bf16w 4.59 | 0.6298 (bf16-compile) · f32 1.66 · bf16w 1.03 | 0.3087 (bf16-compile-ro) · f32 2.57 · bf16w 4.94 |
| llm_blkpre_t2048 | 8.043 (bf16-compile-cg) · f32 3.16 · bf16w 3.19 | 13.41 (bf16-compile-ro) · f32 3.39 · bf16w 3.28 | 4.476 (bf16-compile-cg) · f32 12.48 · bf16w 12.53 | 5.594 (bf16-compile-cg) · f32 3.36 · bf16w 3.29 | 2.794 (bf16-compile-cg) · f32 10.64 · bf16w 9.92 |
| llm_dec_b1_l2048 | 0.143 (bf16-compile) · f32 1.68 · bf16w 1.05 | 0.2327 (bf16-compile) · f32 1.40 · bf16w 1.23 | 0.1581 (bf16-compile) · f32 2.98 · bf16w 2.94 | 0.2401 (bf16-compile-cg) · f32 1.01 · bf16w 0.69 | 0.156 (bf16-compile-ro) · f32 2.09 · bf16w 2.09 |
| llm_dec_b1_l8192 | 0.1863 (bf16-compile) · f32 1.78 · bf16w 1.32 | 0.293 (bf16-compile) · f32 1.46 · bf16w 1.32 | 0.1926 (bf16-compile) · f32 2.75 · bf16w 2.71 | 0.2547 (bf16-compile) · f32 1.36 · bf16w 1.07 | 0.1621 (bf16-compile-ro) · f32 2.26 · bf16w 2.26 |
| llm_dec_b32_l2048 | 0.4613 (bf16-compile) · f32 2.24 · bf16w 2.06 | 0.7273 (bf16-compile) · f32 1.67 · bf16w 1.60 | 0.4312 (bf16-cudnn) · f32 1.99 · bf16w 2.01 | 0.5082 (bf16-compile) · f32 2.22 · bf16w 2.01 | 0.2762 (bf16-compile-ro) · f32 2.02 · bf16w 2.00 |
| llm_dec_b32_l8192 | 1.498 (flashinfer-bf16) · f32 2.35 · bf16w 2.30 | 1.694 (bf16-cudnn) · f32 2.52 · bf16w 2.50 | 0.9747 (bf16-cudnn) · f32 2.60 · bf16w 2.60 | 1.4 (bf16-compile) · f32 2.68 · bf16w 2.61 | 0.5796 (bf16-compile-ro) · f32 2.55 · bf16w 2.51 |
| llm_dec_b8_l2048 | 0.237 (bf16-compile-cg) · f32 1.73 · bf16w 1.33 | 0.3526 (bf16-compile) · f32 2.61 · bf16w 2.09 | 0.2281 (bf16-compile) · f32 1.75 · bf16w 1.53 | 0.2796 (bf16-compile) · f32 1.79 · bf16w 1.48 | 0.1738 (bf16-compile-ro) · f32 1.26 · bf16w 1.07 |
| llm_dec_b8_l8192 | 0.4938 (fa2-bf16) · f32 2.17 · bf16w 2.05 | 0.7316 (bf16) · f32 2.33 · bf16w 2.07 | 0.4212 (bf16) · f32 1.92 · bf16w 1.81 | 0.5018 (bf16-compile) · f32 2.25 · bf16w 2.09 | 0.2713 (bf16-compile-ro) · f32 1.87 · bf16w 1.77 |
| llm_pre_t2048 | 1.932 (bf16-compile-cg) · f32 3.74 · bf16w 3.70 | 3.22 (bf16-compile) · f32 4.36 · bf16w 4.33 | 1.041 (bf16-compile-cg) · f32 14.53 · bf16w 14.54 | 1.294 (bf16-compile-cg) · f32 4.87 · bf16w 4.81 | 0.649 (bf16-compile-cg) · f32 13.20 · bf16w 12.34 |
| llm_pre_t8192 | 11.36 (bf16-compile) · f32 7.88 | 17.92 (bf16-compile) · f32 7.25 | 6.562 (bf16-compile-ro) · f32 20.16 | 7.656 (bf16-compile-cg) · f32 8.86 | 4.074 (bf16-compile) · f32 17.51 |

## Verdict flips vs the mixed-precision gap tables

gap.py's rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables (gap-<part>.md, gap.md) row for row. Rows a part's gap table did not cover (the RTX 5080's model-scale rows, reported in model-scale.md) are judged against that rule applied to the same data.

| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |
|---|---|---|---|---|
| rtx-5080 | llm_blkdec_b1_l4096 | 0.5498 (torch-bf16-compile) 1.95 loss | 1.07 (torch-compile) 1.00 even | old best a bf16 leg |
| rtx-5080 | llm_blkpre_t2048 | 8.088 (torch-bf16-compile) 3.14 loss | 26.84 (liger) 0.95 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_dec_b1_l2048 | 0.143 (torch-bf16-compile) 1.68 loss | 0.2589 (torch-compile) 0.93 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l8192 | 0.1863 (torch-bf16-compile) 1.78 loss | 0.3439 (torch-compile) 0.97 even | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l2048 | 0.4613 (torch-bf16-compile) 2.24 loss | 2.262 (torch-compile) 0.46 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l8192 | 1.507 (torch-bf16-compile) 2.33 loss | 10.27 (torch-compile) 0.34 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l2048 | 0.2578 (torch-bf16-compile) 1.59 loss | 0.6816 (torch-compile-cg) 0.60 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l8192 | 0.5165 (torch-bf16-compile) 2.07 loss | 2.856 (torch-compile-cg) 0.37 win | old best a bf16 leg |
| rtx-5080 | llm_pre_t2048 | 1.933 (torch-bf16-compile) 3.74 loss | 7.219 (liger) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_pre_t2048_tf32 | 6.122 (torch-compile-tf32) 0.99 even | 4.543 (triton-tf32) 1.34 loss | best now a gemm_lib leg |
| rtx-5080 | pj_m16_n18432_k7168 | 0.3092 (torch-bf16-compile) 1.96 loss | 0.5878 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n2048_k7168 | 0.0389 (torch-bf16-compile) 1.31 loss | 0.0492 (cublas-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n28672_k4096 | 0.2744 (torch-bf16-compile) 1.97 loss | 0.5243 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n4096_k14336 | 0.1393 (torch-bf16-compile) 1.91 loss | 0.2744 (torch-compile) 0.97 even | old best a bf16 leg |
| rtx-5080 | pj_m16_n4096_k4096 | 0.043 (torch-bf16-compile) 1.30 loss | 0.0553 (cublas-fresh) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n18432_k7168 | 0.3092 (torch-bf16) 1.93 loss | 0.6103 (torch-compile) 0.98 even | old best a bf16 leg |
| rtx-5080 | pj_m1_n2048_k7168 | 0.043 (torch-bf16-compile) 0.84 win | 0.0266 (cublas-fresh) 1.36 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n28672_k4096 | 0.2744 (torch-bf16-compile) 1.91 loss | 0.5038 (lt-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n4096_k4096 | 0.0492 (torch-bf16-compile) 0.79 win | 0.0205 (lt-fresh) 1.89 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096 | 3.964 (torch-bf16-compile) 2.97 loss | 12.21 (lt-fresh) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k14336 | 2.101 (torch-bf16-compile) 3.07 loss | 6.413 (lt-fresh) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k4096 | 0.6021 (torch-bf16-compile) 3.06 loss | 1.778 (torch-compile-cg) 1.04 even | old best a bf16 leg |
| rtx-5080 | pj_m2048_n6144_k4096 | 0.8868 (torch-bf16) 2.99 loss | 2.615 (torch-compile-cg) 1.01 even | old best a bf16 leg |
| rtx-5080 | rt_mlp2_2048 | 17.97 (torch-compile) 0.90 win | 15.66 (torch-compile-ro) 1.04 even | other |
| rtx-3090 | llm_dec_b1_l2048 | 0.2327 (torch-bf16-compile) 1.40 loss | 0.3454 (torch-compile) 0.94 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l8192 | 0.293 (torch-bf16-compile) 1.46 loss | 0.4502 (torch-compile) 0.95 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l2048 | 0.7273 (torch-bf16-compile) 1.67 loss | 2.658 (torch-compile) 0.46 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192 | 2.442 (torch-bf16-compile) 1.75 loss | 10.85 (torch-compile) 0.39 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l2048 | 0.3526 (torch-bf16-compile) 2.61 loss | 0.8917 (torch-compile) 1.03 even | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l8192 | 0.7316 (torch-bf16) 2.33 loss | 2.925 (torch-compile) 0.58 win | old best a bf16 leg |
| rtx-3090 | lsm_1_131072 | 0.0287 (torch-compile) 0.85 win | 0.0256 (torch-compile-cg) 0.96 even | other |
| rtx-3090 | lsm_4_32768 | 0.0205 (torch-compile) 0.91 win | 0.0164 (torch-compile-cg) 1.13 loss | other |
| rtx-3090 | mlp_2048_tf32 | 1.276 (torch-eager-tf32) 0.95 win | 1.145 (torch-compile-ro-tf32) 1.05 loss | other |
| rtx-3090 | pj_m128_n28672_k4096 | 0.4751 (torch-bf16-compile) 2.53 loss | 1.226 (torch-eager) 0.98 even | old best a bf16 leg |
| rtx-3090 | pj_m128_n6144_k4096 | 0.126 (torch-bf16) 2.52 loss | 0.3092 (cublas-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m16_n18432_k7168 | 0.3072 (torch-bf16-compile) 2.04 loss | 0.6124 (torch-compile) 1.02 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n18432_k7168 | 0.3195 (torch-bf16-compile) 1.90 loss | 0.6052 (lt) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m1_n28672_k4096 | 0.2785 (torch-bf16-compile) 1.95 loss | 0.5366 (torch-compile-ro) 1.01 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n4096_k14336 | 0.1476 (torch-bf16) 1.90 loss | 0.2744 (torch-compile-ro) 1.02 even | old best a bf16 leg |
| rtx-3090 | pj_m2048_n28672_k4096 | 6.341 (torch-bf16) 3.03 loss | 18.85 (torch-eager) 1.02 even | old best a bf16 leg |
| rtx-3090 | pj_m2048_n4096_k4096 | 1.079 (torch-bf16-compile) 2.66 loss | 2.829 (torch-compile-ro) 1.02 even | old best a bf16 leg |
| rtx-3090 | pj_m2048_n6144_k4096_tf32 | 2.845 (torch-compile-tf32) 1.05 even | 2.821 (torch-compile-cg-tf32) 1.06 loss | other |
| rtx-3090 | rt_mlp2_1024 | 4.183 (torch-compile) 0.95 win | 4.125 (torch-compile-cg) 0.96 even | other |
| rtx-3090 | twin_2048 | 1.538 (torch-eager) 1.03 even | 1.422 (torch-compile-ro) 1.11 loss | other |
| a100 | llm_blkpre_t2048 | 4.521 (torch-bf16-compile) 12.36 loss | 54.53 (torch-compile) 1.02 even | old best a bf16 leg |
| a100 | llm_dec_b32_l2048 | 0.4765 (torch-bf16-compile) 1.80 loss | 1.835 (torch-compile-cg) 0.47 win | old best a bf16 leg |
| a100 | llm_dec_b32_l8192 | 1.21 (torch-bf16) 2.09 loss | 6.513 (torch-compile-cg) 0.39 win | old best a bf16 leg |
| a100 | llm_dec_b8_l2048 | 0.2281 (torch-bf16-compile) 1.75 loss | 0.6182 (torch-compile) 0.64 win | old best a bf16 leg |
| a100 | llm_dec_b8_l8192 | 0.4212 (torch-bf16) 1.92 loss | 1.807 (torch-compile) 0.45 win | old best a bf16 leg |
| a100 | lsm_1_131072 | 0.0328 (torch-sdpa-compile) 0.76 win | 0.0256 (torch-compile-cg) 0.97 even | other |
| a100 | lsm_4_32768 | 0.0297 (torch-compile) 0.73 win | 0.0195 (torch-compile-cg) 1.11 loss | other |
| a100 | pj_m128_n28672_k4096 | 0.2908 (torch-bf16-compile) 5.95 loss | 1.748 (torch-compile-ro) 0.99 even | old best a bf16 leg |
| a100 | pj_m16_n18432_k7168 | 0.213 (torch-bf16-compile) 2.02 loss | 0.4454 (torch-compile) 0.97 even | old best a bf16 leg |
| a100 | pj_m16_n28672_k4096 | 0.1925 (torch-bf16-compile) 2.19 loss | 0.426 (torch-compile) 0.99 even | old best a bf16 leg |
| a100 | pj_m16_n4096_k14336 | 0.1096 (torch-bf16-compile) 2.08 loss | 0.2509 (lt) 0.91 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m2048_n28672_k4096 | 2.081 (torch-bf16) 12.91 loss | 26.02 (torch-compile-cg) 1.03 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k14336 | 0.9103 (torch-bf16) 14.66 loss | 12.89 (torch-compile-ro) 1.04 even | old best a bf16 leg |
| a100 | pj_m2048_n4096_k4096 | 0.3092 (torch-bf16) 12.51 loss | 3.704 (torch-compile-ro) 1.04 even | old best a bf16 leg |
| a100 | pj_m2048_n6144_k4096 | 0.4147 (torch-bf16-compile) 14.00 loss | 5.785 (torch-compile) 1.00 even | old best a bf16 leg |
| a100 | softmax_rows_4096 | 7.567 (chain-fused) 1.03 even | 7.376 (torch-compile-ro) 1.06 loss | other |
| rtx-4090 | attn_2048_64 | 0.0737 (torch-sdpa-compile) 1.01 even | 0.0604 (torch-compile-ro) 1.23 loss | other |
| rtx-4090 | llm_blkdec_b1_l4096 | 0.6298 (torch-bf16-compile) 1.66 loss | 1.088 (torch-compile) 0.96 even | old best a bf16 leg |
| rtx-4090 | llm_blkpre_t2048 | 5.676 (torch-bf16-compile) 3.31 loss | 17.95 (liger) 1.05 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_dec_b1_l2048 | 0.2538 (torch-bf16-compile) 0.96 even | 0.3161 (torch-compile) 0.77 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b1_l8192 | 0.2547 (torch-bf16-compile) 1.36 loss | 0.3747 (torch-compile) 0.92 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l2048 | 0.5082 (torch-bf16-compile) 2.22 loss | 2.329 (torch-compile) 0.48 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l8192 | 1.4 (torch-bf16-compile) 2.68 loss | 9.407 (torch-compile) 0.40 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l2048 | 0.2796 (torch-bf16-compile) 1.79 loss | 0.7461 (torch-compile) 0.67 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l8192 | 0.5018 (torch-bf16-compile) 2.25 loss | 2.56 (torch-compile) 0.44 win | old best a bf16 leg |
| rtx-4090 | lsm_1_131072 | 0.0276 (torch-compile) 1.01 even | 0.0236 (torch-compile-cg) 1.18 loss | other |
| rtx-4090 | pj_m128_n28672_k4096 | 0.3154 (torch-bf16-compile) 2.10 loss | 0.6513 (lt-fresh) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n4096_k4096_tf32 | 0.1085 (torch-compile-tf32) 0.91 win | 0.0644 (cublas-tf32-fresh) 1.53 loss | best now a gemm_lib leg |
| rtx-4090 | pj_m1_n2048_k7168 | 0.0512 (torch-bf16-compile) 1.01 even | 0.0266 (lt-fresh) 1.93 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n4096_k4096 | 0.0563 (torch-bf16-compile) 0.77 win | 0.0195 (lt-fresh) 2.23 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k14336 | 1.434 (torch-bf16-compile) 2.84 loss | 4.185 (cublas) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k4096 | 0.4148 (torch-bf16-compile) 2.87 loss | 1.158 (torch-compile-ro) 1.03 even | old best a bf16 leg |
| rtx-4090 | pj_m2048_n6144_k4096 | 0.5857 (torch-bf16-compile) 3.05 loss | 1.756 (torch-compile) 1.02 even | old best a bf16 leg |
| h100 | llm_dec_b32_l2048 | 0.3036 (torch-bf16-compile) 1.84 loss | 1.417 (torch-compile-cg) 0.39 win | old best a bf16 leg |
| h100 | llm_dec_b32_l8192 | 0.5995 (torch-bf16-compile) 2.47 loss | 5.389 (torch-compile-cg) 0.27 win | old best a bf16 leg |
| h100 | llm_dec_b8_l2048_bf16w | 0.2517 (torch-bf16-compile) 0.74 win | 0.1675 (torch-bf16-compile-ro) 1.11 loss | other |
| h100 | llm_dec_b8_l8192 | 0.3058 (torch-bf16-compile) 1.66 loss | 1.437 (torch-compile-cg) 0.35 win | old best a bf16 leg |
| h100 | pj_m16_n18432_k7168 | 0.1006 (torch-bf16-compile) 3.25 loss | 0.3356 (torch-compile-cg) 0.97 even | old best a bf16 leg |
| h100 | pj_m16_n4096_k14336 | 0.0538 (torch-bf16-compile) 3.07 loss | 0.1691 (torch-compile) 0.98 even | old best a bf16 leg |
| h100 | pj_m2048_n6144_k4096 | 0.2662 (torch-bf16) 10.33 loss | 2.878 (torch-compile) 0.96 even | old best a bf16 leg |
| h100 | rt_mlp2_1024 | 2.489 (torch-compile) 1.05 even | 2.435 (torch-compile-cg) 1.07 loss | other |

## What the new legs change

Rows whose best leg is one the competitors step added (CUDA-graph torch.compile, SDPA backends, hand Triton, Liger, FlashAttention-2/3, FlashInfer), against the best over the legs before it (`base`).

| part | workload | prec | base: best (leg) ratio | now: best (leg) ratio | verdict |
|---|---|---|---|---|---|
| rtx-5080 | pj_m128_n28672_k4096_bf16 | bf16 | 0.3215 (torch-bf16-compile) 1.20 | 0.3174 (triton-bf16) 1.21 | loss → loss |
| rtx-5080 | pj_m128_n4096_k14336_bf16 | bf16 | 0.172 (torch-bf16) 1.45 | 0.17 (torch-bf16-compile-cg) 1.46 | loss → loss |
| rtx-5080 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0553 (torch-bf16) 1.40 | 0.0532 (torch-bf16-compile-ro) 1.45 | loss → loss |
| rtx-5080 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0492 (torch-bf16-compile) 0.74 | 0.0471 (torch-bf16-compile-cg) 0.77 | win → win |
| rtx-5080 | pj_m2048_n28672_k4096_bf16 | bf16 | 3.964 (torch-bf16-compile) 1.18 | 3.963 (torch-bf16-compile-cg) 1.18 | loss → loss |
| rtx-5080 | pj_m2048_n4096_k14336_bf16 | bf16 | 2.101 (torch-bf16-compile) 1.09 | 2.097 (torch-bf16-compile-cg) 1.09 | loss → loss |
| rtx-5080 | pj_m2048_n4096_k4096 | f32 | 1.78 (torch-eager) 1.04 | 1.778 (torch-compile-cg) 1.04 | even → even |
| rtx-5080 | pj_m2048_n4096_k4096_bf16 | bf16 | 0.6021 (torch-bf16-compile) 1.18 | 0.598 (torch-bf16-compile-cg) 1.19 | loss → loss |
| rtx-5080 | pj_m2048_n6144_k4096 | f32 | 2.617 (torch-eager) 1.01 | 2.615 (torch-compile-cg) 1.01 | even → even |
| rtx-5080 | pj_m2048_n6144_k4096_bf16 | bf16 | 0.8868 (torch-bf16) 1.22 | 0.8865 (torch-bf16-compile-ro) 1.22 | loss → loss |
| rtx-5080 | mlp2_2048_tf32 | tf32 | 1.545 (torch-eager-tf32) 1.02 | 1.513 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-5080 | mlp_2048_tf32 | tf32 | 0.7629 (torch-eager-tf32) 1.03 | 0.7598 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-5080 | attn_2048_64 | f32 | 0.0748 (torch-sdpa-compile) 1.09 | 0.0696 (torch-compile-ro) 1.18 | loss → loss |
| rtx-5080 | causal_1024_256 | f32 | 51.28 (torch-compile) 0.30 | 42.76 (torch-compile-cg) 0.36 | win → win |
| rtx-5080 | gen_1024_512_256 | f32 | 29.85 (torch-compile) 0.26 | 24.48 (torch-compile-cg) 0.32 | win → win |
| rtx-5080 | lsm_1_131072 | f32 | 0.0266 (torch-compile) 0.85 | 0.0246 (torch-compile-ro) 0.92 | win → win |
| rtx-5080 | lsm_32_8192 | f32 | 0.0102 (torch-sdpa-compile) 2.37 | 0.0102 (torch-compile-cg) 2.37 | loss → loss |
| rtx-5080 | lsm_4_32768 | f32 | 0.0164 (torch-compile) 1.30 | 0.0143 (torch-compile-cg) 1.49 | loss → loss |
| rtx-5080 | prefill_1024_512_256 | f32 | 29.73 (torch-compile) 0.28 | 24.98 (torch-compile-cg) 0.33 | win → win |
| rtx-5080 | rowsum_4096 | f32 | 3.609 (chain-gemv) 1.04 | 3.498 (torch-compile-cg) 1.07 | even → even |
| rtx-5080 | softmax_rows_2048 | f32 | 0.4731 (torch-compile) 1.01 | 0.4707 (torch-compile-cg) 1.02 | even → even |
| rtx-5080 | softmax_rows_4096 | f32 | 3.562 (torch-compile) 1.08 | 3.547 (torch-compile-ro) 1.08 | even → even |
| rtx-5080 | llm_blkpre_t2048 | f32 | 28.15 (torch-compile) 0.90 | 26.84 (liger) 0.95 | win → win |
| rtx-5080 | llm_blkpre_t2048_bf16 | bf16 | 8.088 (torch-bf16-compile) 1.44 | 8.043 (torch-bf16-compile-cg) 1.44 | loss → loss |
| rtx-5080 | llm_blkpre_t2048_bf16w | bf16w | 8.088 (torch-bf16-compile) 3.17 | 8.043 (torch-bf16-compile-cg) 3.19 | loss → loss |
| rtx-5080 | llm_blkpre_t2048_tf32 | tf32 | 18.6 (torch-compile-tf32) 1.07 | 18.07 (triton-tf32) 1.10 | loss → loss |
| rtx-5080 | llm_dec_b32_l8192_bf16 | bf16 | 1.478 (torch-bf16-compile) 1.62 | 1.47 (torch-bf16-cudnn) 1.63 | loss → loss |
| rtx-5080 | llm_dec_b32_l8192_bf16kv | bf16kv | 1.478 (torch-bf16-compile) 1.63 | 1.47 (torch-bf16-cudnn) 1.64 | loss → loss |
| rtx-5080 | llm_dec_b32_l8192_bf16w | bf16w | 1.478 (torch-bf16-compile) 2.33 | 1.47 (torch-bf16-cudnn) 2.34 | loss → loss |
| rtx-5080 | llm_dec_b8_l2048 | f32 | 0.7077 (torch-compile) 0.58 | 0.6816 (torch-compile-cg) 0.60 | win → win |
| rtx-5080 | llm_dec_b8_l8192 | f32 | 2.889 (torch-compile) 0.37 | 2.856 (torch-compile-cg) 0.37 | win → win |
| rtx-5080 | llm_pre_t2048 | f32 | 8.44 (torch-compile) 0.86 | 7.219 (liger) 1.00 | win → even |
| rtx-5080 | llm_pre_t2048_bf16 | bf16 | 1.933 (torch-bf16-compile) 2.04 | 1.932 (torch-bf16-compile-cg) 2.04 | loss → loss |
| rtx-5080 | llm_pre_t2048_bf16w | bf16w | 1.933 (torch-bf16-compile) 3.70 | 1.932 (torch-bf16-compile-cg) 3.70 | loss → loss |
| rtx-5080 | llm_pre_t2048_tf32 | tf32 | 6.122 (torch-compile-tf32) 0.99 | 4.543 (triton-tf32) 1.34 | even → loss |
| rtx-5080 | llm_pre_t8192 | f32 | 85.01 (torch-compile) 1.05 | 52.91 (liger) 1.69 | even → loss |
| rtx-5080 | ewconvew_2048_5 | f32 | 0.1874 (torch-compile) 0.43 | 0.1864 (torch-compile-cg) 0.44 | win → win |
| rtx-5080 | diamond_2048 | f32 | 1.37 (torch-compile) 1.06 | 1.368 (torch-compile-cg) 1.06 | loss → loss |
| rtx-5080 | loopmm_1024_8 | f32 | 0.5898 (torch-compile) 1.11 | 0.5878 (torch-compile-cg) 1.11 | loss → loss |
| rtx-5080 | rt_mlp2_1024 | f32 | 2.544 (torch-compile) 1.10 | 2.518 (torch-compile-cg) 1.11 | loss → loss |
| rtx-5080 | rt_mlp2_2048 | f32 | 17.97 (torch-compile) 0.90 | 15.66 (torch-compile-ro) 1.04 | win → even |
| rtx-5080 | twin_2048 | f32 | 0.9236 (torch-compile) 1.02 | 0.9185 (torch-compile-cg) 1.02 | even → even |
| rtx-3090 | pj_m128_n28672_k4096_bf16 | bf16 | 0.4751 (torch-bf16-compile) 1.04 | 0.4618 (triton-bf16) 1.07 | even → loss |
| rtx-3090 | pj_m128_n4096_k14336_tf32 | tf32 | 0.6144 (torch-compile-tf32) 1.06 | 0.5714 (triton-tf32) 1.14 | loss → loss |
| rtx-3090 | pj_m128_n4096_k4096_bf16 | bf16 | 0.084 (torch-bf16) 1.29 | 0.0799 (triton-bf16) 1.35 | loss → loss |
| rtx-3090 | pj_m128_n6144_k4096_bf16 | bf16 | 0.126 (torch-bf16) 1.34 | 0.1249 (torch-bf16-compile-cg) 1.35 | loss → loss |
| rtx-3090 | pj_m16_n4096_k14336 | f32 | 0.2744 (torch-compile) 1.05 | 0.2734 (torch-compile-cg) 1.06 | loss → loss |
| rtx-3090 | pj_m16_n4096_k4096 | f32 | 0.0851 (torch-compile) 1.30 | 0.085 (torch-compile-cg) 1.30 | loss → loss |
| rtx-3090 | pj_m16_n6144_k4096 | f32 | 0.1229 (torch-compile) 1.16 | 0.1221 (torch-compile-cg) 1.17 | loss → loss |
| rtx-3090 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0655 (torch-bf16-compile) 1.34 | 0.0649 (torch-bf16-compile-cg) 1.35 | loss → loss |
| rtx-3090 | pj_m1_n28672_k4096 | f32 | 0.5376 (cublas) 1.01 | 0.5366 (torch-compile-ro) 1.01 | even → even |
| rtx-3090 | pj_m1_n4096_k14336 | f32 | 0.2755 (lt) 1.02 | 0.2744 (torch-compile-ro) 1.02 | even → even |
| rtx-3090 | pj_m1_n4096_k14336_bf16 | bf16 | 0.1476 (torch-bf16) 1.04 | 0.1475 (torch-bf16-compile-cg) 1.04 | even → even |
| rtx-3090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0502 (torch-bf16) 1.16 | 0.0492 (torch-bf16-compile-cg) 1.18 | loss → loss |
| rtx-3090 | pj_m2048_n28672_k4096_bf16 | bf16 | 6.341 (torch-bf16) 1.07 | 6.283 (torch-bf16-compile-cg) 1.08 | loss → loss |
| rtx-3090 | pj_m2048_n4096_k14336_bf16 | bf16 | 3.713 (torch-bf16) 1.07 | 3.556 (triton-bf16) 1.12 | loss → loss |
| rtx-3090 | pj_m2048_n4096_k14336_tf32 | tf32 | 6.909 (torch-eager-tf32) 1.05 | 6.896 (torch-compile-ro-tf32) 1.06 | loss → loss |
| rtx-3090 | pj_m2048_n4096_k4096 | f32 | 2.84 (torch-compile) 1.01 | 2.829 (torch-compile-ro) 1.02 | even → even |
| rtx-3090 | pj_m2048_n4096_k4096_bf16 | bf16 | 1.079 (torch-bf16-compile) 1.03 | 1.023 (triton-bf16) 1.09 | even → loss |
| rtx-3090 | pj_m2048_n6144_k4096 | f32 | 4.138 (lt) 1.10 | 4.099 (torch-compile-ro) 1.11 | loss → loss |
| rtx-3090 | pj_m2048_n6144_k4096_bf16 | bf16 | 1.424 (torch-bf16-compile) 1.08 | 1.369 (torch-bf16-compile-ro) 1.12 | loss → loss |
| rtx-3090 | pj_m2048_n6144_k4096_tf32 | tf32 | 2.845 (torch-compile-tf32) 1.05 | 2.821 (torch-compile-cg-tf32) 1.06 | even → loss |
| rtx-3090 | mlp2_2048_tf32 | tf32 | 2.534 (torch-eager-tf32) 0.96 | 2.353 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-3090 | mlp_2048_tf32 | tf32 | 1.276 (torch-eager-tf32) 0.95 | 1.145 (torch-compile-ro-tf32) 1.05 | win → loss |
| rtx-3090 | mlp_512 | f32 | 0.045 (chain-lt) 1.22 | 0.0399 (triton) 1.38 | loss → loss |
| rtx-3090 | attn_1024_64 | f32 | 0.0492 (torch-compile) 1.56 | 0.0451 (torch-compile-ro) 1.70 | loss → loss |
| rtx-3090 | attn_2048_128 | f32 | 0.1802 (torch-compile) 1.34 | 0.1761 (torch-sdpa-compile-cg) 1.37 | loss → loss |
| rtx-3090 | attn_2048_64 | f32 | 0.1331 (torch-compile) 1.14 | 0.1293 (torch-compile-ro) 1.17 | loss → loss |
| rtx-3090 | causal_1024_256 | f32 | 170.4 (torch-compile) 0.13 | 155.2 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | gen_1024_512_256 | f32 | 85.08 (torch-compile) 0.13 | 77.58 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | lsm_1_131072 | f32 | 0.0287 (torch-compile) 0.85 | 0.0256 (torch-compile-cg) 0.96 | win → even |
| rtx-3090 | lsm_32_8192 | f32 | 0.0102 (torch-compile) 2.31 | 0.0092 (torch-compile-cg) 2.56 | loss → loss |
| rtx-3090 | lsm_4_32768 | f32 | 0.0205 (torch-compile) 0.91 | 0.0164 (torch-compile-cg) 1.13 | win → loss |
| rtx-3090 | prefill_1024_512_256 | f32 | 86.33 (torch-compile) 0.13 | 78.32 (torch-compile-ro) 0.14 | win → win |
| rtx-3090 | rowsum_4096 | f32 | 5.746 (torch-eager) 1.07 | 5.743 (torch-compile-ro) 1.07 | loss → loss |
| rtx-3090 | llm_blkdec_b1_l4096 | f32 | 1.209 (torch-compile) 1.14 | 1.203 (torch-compile-cg) 1.15 | loss → loss |
| rtx-3090 | llm_blkpre_t2048 | f32 | 42.23 (torch-compile) 1.08 | 41.26 (liger) 1.10 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_bf16 | bf16 | 13.68 (torch-bf16-compile) 1.52 | 13.41 (torch-bf16-compile-ro) 1.55 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_bf16w | bf16w | 13.68 (torch-bf16-compile) 3.22 | 13.41 (torch-bf16-compile-ro) 3.28 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_tf32 | tf32 | 29.64 (torch-compile-tf32) 1.13 | 27.78 (triton-tf32) 1.20 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16 | bf16 | 2.407 (torch-bf16-compile) 1.44 | 1.697 (torch-bf16-cudnn) 2.04 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16kv | bf16kv | 2.407 (torch-bf16-compile) 1.43 | 1.697 (torch-bf16-cudnn) 2.03 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16w | bf16w | 2.407 (torch-bf16-compile) 1.76 | 1.697 (torch-bf16-cudnn) 2.49 | loss → loss |
| rtx-3090 | llm_pre_t2048 | f32 | 12.55 (torch-compile) 1.12 | 11.48 (liger) 1.22 | loss → loss |
| rtx-3090 | llm_pre_t2048_tf32 | tf32 | 9.03 (torch-compile-tf32) 1.30 | 7.568 (triton-tf32) 1.55 | loss → loss |
| rtx-3090 | llm_pre_t8192 | f32 | 109.6 (torch-compile) 1.19 | 85.87 (liger) 1.51 | loss → loss |
| rtx-3090 | ewconvew_2048_5 | f32 | 0.2417 (torch-compile) 0.47 | 0.2386 (torch-compile-cg) 0.47 | win → win |
| rtx-3090 | rt_mlp2_1024 | f32 | 4.183 (torch-compile) 0.95 | 4.125 (torch-compile-cg) 0.96 | win → even |
| rtx-3090 | rt_mlp2_2048 | f32 | 26.68 (torch-compile) 0.96 | 26.61 (torch-compile-cg) 0.96 | even → even |
| rtx-3090 | twin_2048 | f32 | 1.538 (torch-eager) 1.03 | 1.422 (torch-compile-ro) 1.11 | even → loss |
| a100 | pj_m128_n28672_k4096 | f32 | 1.781 (cublas) 0.97 | 1.748 (torch-compile-ro) 0.99 | even → even |
| a100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.2908 (torch-bf16-compile) 1.31 | 0.2181 (triton-bf16) 1.75 | loss → loss |
| a100 | pj_m128_n28672_k4096_tf32 | tf32 | 0.4516 (lt-tf32) 1.18 | 0.4178 (triton-tf32) 1.28 | loss → loss |
| a100 | pj_m128_n4096_k14336_bf16 | bf16 | 0.126 (torch-bf16-compile) 2.66 | 0.1239 (torch-bf16-compile-cg) 2.71 | loss → loss |
| a100 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0573 (torch-bf16-compile) 1.92 | 0.0532 (triton-bf16) 2.06 | loss → loss |
| a100 | pj_m128_n6144_k4096_bf16 | bf16 | 0.0676 (torch-bf16-compile) 1.72 | 0.0625 (triton-bf16) 1.86 | loss → loss |
| a100 | pj_m128_n6144_k4096_tf32 | tf32 | 0.1034 (cublas-tf32) 1.57 | 0.0993 (triton-tf32) 1.63 | loss → loss |
| a100 | pj_m16_n4096_k4096_bf16 | bf16 | 0.044 (torch-bf16-compile) 2.45 | 0.043 (torch-bf16-compile-cg) 2.51 | loss → loss |
| a100 | pj_m16_n6144_k4096 | f32 | 0.1167 (torch-compile) 1.11 | 0.1157 (torch-compile-cg) 1.12 | loss → loss |
| a100 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0604 (torch-bf16-compile) 1.87 | 0.0594 (torch-bf16-compile-cg) 1.90 | loss → loss |
| a100 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0461 (torch-bf16) 1.05 | 0.0451 (torch-bf16-compile-cg) 1.07 | even → loss |
| a100 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0553 (torch-bf16-compile) 1.33 | 0.0532 (torch-bf16-compile-cg) 1.38 | loss → loss |
| a100 | pj_m2048_n28672_k4096 | f32 | 26.02 (torch-compile) 1.03 | 26.02 (torch-compile-cg) 1.03 | even → even |
| a100 | pj_m2048_n28672_k4096_bf16 | bf16 | 2.081 (torch-bf16) 1.69 | 1.954 (torch-bf16-compile-cg) 1.80 | loss → loss |
| a100 | pj_m2048_n4096_k14336 | f32 | 12.9 (torch-eager) 1.03 | 12.89 (torch-compile-ro) 1.04 | even → even |
| a100 | pj_m2048_n4096_k14336_bf16 | bf16 | 0.9103 (torch-bf16) 2.16 | 0.9052 (torch-bf16-compile-ro) 2.17 | loss → loss |
| a100 | pj_m2048_n4096_k14336_tf32 | tf32 | 2.146 (cublas-tf32) 1.26 | 2.071 (torch-compile-ro-tf32) 1.31 | loss → loss |
| a100 | pj_m2048_n4096_k4096 | f32 | 3.708 (cublas-fresh) 1.04 | 3.704 (torch-compile-ro) 1.04 | even → even |
| a100 | pj_m2048_n6144_k4096_bf16 | bf16 | 0.4147 (torch-bf16-compile) 1.97 | 0.4096 (torch-bf16-compile-cg) 1.99 | loss → loss |
| a100 | mlp2_2048_tf32 | tf32 | 0.7066 (torch-compile-tf32) 1.32 | 0.6984 (torch-compile-cg-tf32) 1.34 | loss → loss |
| a100 | mlp_2048_tf32 | tf32 | 0.3543 (torch-eager-tf32) 1.44 | 0.3502 (torch-compile-cg-tf32) 1.45 | loss → loss |
| a100 | attn_1024_64 | f32 | 0.0604 (chain-fused) 1.06 | 0.0543 (torch-sdpa-compile-cg) 1.18 | loss → loss |
| a100 | gen_1024_512_256 | f32 | 37.67 (torch-compile) 0.50 | 37.22 (torch-compile-cg) 0.50 | win → win |
| a100 | lsm_1_131072 | f32 | 0.0328 (torch-sdpa-compile) 0.76 | 0.0256 (torch-compile-cg) 0.97 | win → even |
| a100 | lsm_32_8192 | f32 | 0.0143 (torch-compile) 1.93 | 0.0123 (torch-compile-cg) 2.25 | loss → loss |
| a100 | lsm_4_32768 | f32 | 0.0297 (torch-compile) 0.73 | 0.0195 (torch-compile-cg) 1.11 | win → loss |
| a100 | prefill_1024_512_256 | f32 | 37.8 (torch-compile) 0.50 | 37.31 (torch-compile-cg) 0.51 | win → win |
| a100 | rowsum_2048 | f32 | 0.9964 (chain-eager) 1.04 | 0.9943 (torch-compile-ro) 1.04 | even → even |
| a100 | rowsum_4096 | f32 | 7.346 (chain-eager) 1.05 | 7.336 (torch-compile-ro) 1.06 | loss → loss |
| a100 | softmax_rows_4096 | f32 | 7.567 (chain-fused) 1.03 | 7.376 (torch-compile-ro) 1.06 | even → loss |
| a100 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.4915 (torch-bf16-compile) 4.25 | 0.4513 (torch-bf16-compile-cg) 4.63 | loss → loss |
| a100 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.4915 (torch-bf16-compile) 4.25 | 0.4513 (torch-bf16-compile-cg) 4.63 | loss → loss |
| a100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.4915 (torch-bf16-compile) 4.20 | 0.4513 (torch-bf16-compile-cg) 4.57 | loss → loss |
| a100 | llm_blkpre_t2048_bf16 | bf16 | 4.521 (torch-bf16-compile) 2.84 | 4.476 (torch-bf16-compile-cg) 2.87 | loss → loss |
| a100 | llm_blkpre_t2048_bf16w | bf16w | 4.521 (torch-bf16-compile) 12.40 | 4.476 (torch-bf16-compile-cg) 12.53 | loss → loss |
| a100 | llm_blkpre_t2048_tf32 | tf32 | 10.51 (torch-compile-tf32) 1.56 | 9.807 (triton-tf32) 1.67 | loss → loss |
| a100 | llm_dec_b1_l2048_bf16 | bf16 | 0.1754 (torch-bf16-compile) 2.69 | 0.164 (torch-bf16-compile-cg) 2.88 | loss → loss |
| a100 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.1754 (torch-bf16-compile) 2.69 | 0.164 (torch-bf16-compile-cg) 2.88 | loss → loss |
| a100 | llm_dec_b1_l2048_bf16w | bf16w | 0.1754 (torch-bf16-compile) 2.65 | 0.164 (torch-bf16-compile-cg) 2.84 | loss → loss |
| a100 | llm_dec_b32_l2048 | f32 | 1.844 (torch-compile) 0.47 | 1.835 (torch-compile-cg) 0.47 | win → win |
| a100 | llm_dec_b32_l2048_bf16 | bf16 | 0.4738 (torch-bf16-compile) 2.40 | 0.4281 (torch-bf16-cudnn) 2.66 | loss → loss |
| a100 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.4738 (torch-bf16-compile) 2.43 | 0.4281 (torch-bf16-cudnn) 2.69 | loss → loss |
| a100 | llm_dec_b32_l2048_bf16w | bf16w | 0.4738 (torch-bf16-compile) 1.83 | 0.4281 (torch-bf16-cudnn) 2.02 | loss → loss |
| a100 | llm_dec_b32_l8192 | f32 | 6.558 (torch-compile) 0.39 | 6.513 (torch-compile-cg) 0.39 | win → win |
| a100 | llm_dec_b32_l8192_bf16 | bf16 | 1.208 (torch-bf16) 2.97 | 0.9763 (torch-bf16-cudnn) 3.67 | loss → loss |
| a100 | llm_dec_b32_l8192_bf16kv | bf16kv | 1.208 (torch-bf16) 2.98 | 0.9763 (torch-bf16-cudnn) 3.69 | loss → loss |
| a100 | llm_dec_b32_l8192_bf16w | bf16w | 1.208 (torch-bf16) 2.10 | 0.9763 (torch-bf16-cudnn) 2.60 | loss → loss |
| a100 | llm_pre_t2048 | f32 | 14.59 (torch-compile) 1.04 | 12.1 (liger) 1.25 | even → loss |
| a100 | llm_pre_t2048_bf16 | bf16 | 1.049 (torch-bf16-compile) 6.49 | 1.041 (torch-bf16-compile-cg) 6.54 | loss → loss |
| a100 | llm_pre_t2048_bf16w | bf16w | 1.049 (torch-bf16-compile) 14.44 | 1.041 (torch-bf16-compile-cg) 14.54 | loss → loss |
| a100 | llm_pre_t2048_tf32 | tf32 | 3.689 (torch-compile-tf32) 2.01 | 2.829 (triton-tf32) 2.62 | loss → loss |
| a100 | llm_pre_t8192 | f32 | 116.1 (torch-compile) 1.14 | 61.06 (liger) 2.17 | loss → loss |
| a100 | conv_2048_5 | f32 | 0.1382 (torch-eager) 0.52 | 0.1372 (torch-compile-cg) 0.52 | win → win |
| a100 | ewconvew_2048_5 | f32 | 0.1772 (torch-compile) 0.50 | 0.1731 (torch-compile-ro) 0.51 | win → win |
| a100 | firew_rowsum_2048x2048_16 | f32 | 0.0963 (torch-compile) 0.64 | 0.0942 (torch-compile-ro) 0.66 | win → win |
| a100 | diamond_2048 | f32 | 2.949 (chain-lt) 1.10 | 2.945 (torch-compile-cg) 1.10 | loss → loss |
| a100 | indep_2048 | f32 | 1.967 (torch-compile) 1.14 | 1.964 (torch-compile-cg) 1.14 | loss → loss |
| a100 | loop_4096 | f32 | 59.02 (torch-eager) 1.07 | 59 (torch-compile-ro) 1.07 | loss → loss |
| a100 | rt_mlp2_1024 | f32 | 5.454 (torch-eager) 1.25 | 4.299 (torch-compile-cg) 1.59 | loss → loss |
| a100 | rt_mlp2_2048 | f32 | 31.85 (torch-compile) 1.17 | 31.75 (torch-compile-cg) 1.17 | loss → loss |
| a100 | twin_2048 | f32 | 1.964 (torch-eager) 1.03 | 1.961 (torch-compile-ro) 1.03 | even → even |
| rtx-4090 | pj_m128_n4096_k14336_bf16 | bf16 | 0.1894 (torch-bf16-compile) 1.24 | 0.1884 (torch-bf16-compile-cg) 1.24 | loss → loss |
| rtx-4090 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0614 (torch-bf16) 1.09 | 0.0593 (triton-bf16) 1.13 | loss → loss |
| rtx-4090 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0573 (torch-bf16-compile) 1.10 | 0.0563 (torch-bf16-compile-cg) 1.12 | loss → loss |
| rtx-4090 | pj_m16_n6144_k4096_bf16 | bf16 | 0.081 (torch-bf16-compile) 0.95 | 0.0809 (torch-bf16-compile-ro) 0.95 | even → even |
| rtx-4090 | pj_m1_n4096_k14336_bf16 | bf16 | 0.1525 (torch-bf16-compile) 0.80 | 0.1505 (torch-bf16-compile-cg) 0.81 | win → win |
| rtx-4090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0563 (torch-bf16-compile) 0.54 | 0.0553 (torch-bf16-compile-cg) 0.55 | win → win |
| rtx-4090 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0799 (torch-bf16) 0.71 | 0.0788 (torch-bf16-compile-cg) 0.72 | win → win |
| rtx-4090 | pj_m2048_n28672_k4096_bf16 | bf16 | 2.774 (torch-bf16) 1.57 | 2.748 (torch-bf16-compile-ro) 1.58 | loss → loss |
| rtx-4090 | pj_m2048_n28672_k4096_tf32 | tf32 | 5.418 (torch-eager-tf32) 1.54 | 5.406 (torch-compile-cg-tf32) 1.54 | loss → loss |
| rtx-4090 | pj_m2048_n4096_k14336_bf16 | bf16 | 1.434 (torch-bf16-compile) 1.17 | 1.377 (triton-bf16) 1.22 | loss → loss |
| rtx-4090 | pj_m2048_n4096_k14336_tf32 | tf32 | 2.724 (torch-eager-tf32) 1.17 | 2.718 (torch-compile-cg-tf32) 1.17 | loss → loss |
| rtx-4090 | pj_m2048_n4096_k4096 | f32 | 1.161 (torch-compile) 1.03 | 1.158 (torch-compile-ro) 1.03 | even → even |
| rtx-4090 | pj_m2048_n4096_k4096_bf16 | bf16 | 0.4148 (torch-bf16-compile) 1.22 | 0.4004 (triton-bf16) 1.27 | loss → loss |
| rtx-4090 | pj_m2048_n6144_k4096_tf32 | tf32 | 1.168 (torch-eager-tf32) 1.18 | 1.166 (torch-compile-ro-tf32) 1.18 | loss → loss |
| rtx-4090 | mlp2_2048_tf32 | tf32 | 0.8223 (torch-compile-tf32) 1.20 | 0.8172 (torch-compile-cg-tf32) 1.20 | loss → loss |
| rtx-4090 | mlp_2048_tf32 | tf32 | 0.4139 (torch-compile-tf32) 1.16 | 0.4106 (torch-compile-ro-tf32) 1.17 | loss → loss |
| rtx-4090 | mlp_512 | f32 | 0.0284 (torch-compile) 1.95 | 0.0266 (torch-compile-cg) 2.08 | loss → loss |
| rtx-4090 | smlp_4096_256_1024 | f32 | 0.1065 (torch-compile) 1.06 | 0.1026 (triton) 1.10 | loss → loss |
| rtx-4090 | attn_1024_64 | f32 | 0.0317 (torch-compile) 1.63 | 0.0276 (torch-sdpa-compile-cg) 1.87 | loss → loss |
| rtx-4090 | attn_2048_128 | f32 | 0.0819 (torch-compile) 1.16 | 0.0799 (torch-compile-ro) 1.19 | loss → loss |
| rtx-4090 | attn_2048_64 | f32 | 0.0737 (torch-sdpa-compile) 1.01 | 0.0604 (torch-compile-ro) 1.23 | even → loss |
| rtx-4090 | causal_1024_256 | f32 | 196.4 (torch-compile) 0.09 | 177 (torch-compile-cg) 0.10 | win → win |
| rtx-4090 | gen_1024_512_256 | f32 | 97.54 (torch-compile) 0.09 | 88.22 (torch-compile-cg) 0.10 | win → win |
| rtx-4090 | lsm_1_131072 | f32 | 0.0276 (torch-compile) 1.01 | 0.0236 (torch-compile-cg) 1.18 | even → loss |
| rtx-4090 | lsm_32_8192 | f32 | 0.0092 (torch-compile) 4.53 | 0.0082 (torch-compile-cg) 5.09 | loss → loss |
| rtx-4090 | lsm_4_32768 | f32 | 0.0174 (torch-compile) 1.54 | 0.0133 (torch-compile-cg) 2.01 | loss → loss |
| rtx-4090 | prefill_1024_512_256 | f32 | 99.97 (torch-compile) 0.09 | 89.74 (torch-compile-cg) 0.10 | win → win |
| rtx-4090 | rowsum_2048 | f32 | 0.3144 (chain-eager) 0.97 | 0.3092 (torch-compile-ro) 0.99 | even → even |
| rtx-4090 | rowsum_4096 | f32 | 2.432 (torch-compile) 1.00 | 2.375 (torch-compile-cg) 1.02 | even → even |
| rtx-4090 | softmax_rows_4096 | f32 | 2.46 (torch-compile) 1.01 | 2.428 (torch-compile-cg) 1.02 | even → even |
| rtx-4090 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.6304 (torch-bf16-compile) 1.00 | 0.6301 (torch-bf16-compile-cg) 1.00 | even → even |
| rtx-4090 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.6304 (torch-bf16-compile) 1.00 | 0.6301 (torch-bf16-compile-cg) 1.00 | even → even |
| rtx-4090 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.6304 (torch-bf16-compile) 1.03 | 0.6301 (torch-bf16-compile-cg) 1.03 | even → even |
| rtx-4090 | llm_blkpre_t2048 | f32 | 18.97 (torch-compile) 0.99 | 17.95 (liger) 1.05 | even → even |
| rtx-4090 | llm_blkpre_t2048_bf16 | bf16 | 5.676 (torch-bf16-compile) 1.81 | 5.594 (torch-bf16-compile-cg) 1.83 | loss → loss |
| rtx-4090 | llm_blkpre_t2048_bf16w | bf16w | 5.676 (torch-bf16-compile) 3.24 | 5.594 (torch-bf16-compile-cg) 3.29 | loss → loss |
| rtx-4090 | llm_blkpre_t2048_tf32 | tf32 | 13.28 (torch-compile-tf32) 1.25 | 13.24 (triton-tf32) 1.25 | loss → loss |
| rtx-4090 | llm_dec_b1_l2048_bf16 | bf16 | 0.2547 (torch-bf16-compile) 0.63 | 0.2425 (torch-bf16-compile-cg) 0.66 | win → win |
| rtx-4090 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.2547 (torch-bf16-compile) 0.63 | 0.2425 (torch-bf16-compile-cg) 0.66 | win → win |
| rtx-4090 | llm_dec_b1_l2048_bf16w | bf16w | 0.2547 (torch-bf16-compile) 0.65 | 0.2425 (torch-bf16-compile-cg) 0.69 | win → win |
| rtx-4090 | llm_pre_t2048 | f32 | 6.028 (torch-compile) 1.05 | 5.064 (liger) 1.25 | even → loss |
| rtx-4090 | llm_pre_t2048_bf16 | bf16 | 1.3 (torch-bf16-compile) 3.54 | 1.294 (torch-bf16-compile-cg) 3.55 | loss → loss |
| rtx-4090 | llm_pre_t2048_bf16w | bf16w | 1.3 (torch-bf16-compile) 4.79 | 1.294 (torch-bf16-compile-cg) 4.81 | loss → loss |
| rtx-4090 | llm_pre_t2048_tf32 | tf32 | 4.78 (torch-compile-tf32) 1.19 | 3.17 (triton-tf32) 1.79 | loss → loss |
| rtx-4090 | llm_pre_t8192 | f32 | 55.96 (torch-compile) 1.21 | 36.84 (liger) 1.84 | loss → loss |
| rtx-4090 | conv_2048_5 | f32 | 0.0911 (torch-eager) 0.59 | 0.0901 (torch-compile-cg) 0.60 | win → win |
| rtx-4090 | ewconvew_2048_5 | f32 | 0.128 (torch-compile) 0.50 | 0.126 (torch-compile-cg) 0.51 | win → win |
| rtx-4090 | firew_rowsum_2048x2048_16 | f32 | 0.0676 (torch-compile) 0.67 | 0.0655 (torch-compile-cg) 0.69 | win → win |
| rtx-4090 | diamond_2048 | f32 | 0.9196 (torch-compile) 0.99 | 0.9175 (torch-compile-cg) 0.99 | even → even |
| rtx-4090 | indep_512 | f32 | 0.0266 (torch-compile) 2.06 | 0.0256 (torch-compile-cg) 2.14 | loss → loss |
| rtx-4090 | loopmm_1024_8 | f32 | 0.3471 (torch-compile) 1.18 | 0.342 (torch-compile-cg) 1.20 | loss → loss |
| rtx-4090 | rt_mlp2_1024 | f32 | 1.499 (torch-compile) 1.08 | 1.448 (torch-compile-cg) 1.12 | loss → loss |
| rtx-4090 | twin_2048 | f32 | 0.6135 (torch-compile) 0.98 | 0.6093 (torch-compile-cg) 0.99 | even → even |
| h100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.1308 (torch-bf16) 1.99 | 0.13 (torch-bf16-compile-ro) 2.01 | loss → loss |
| h100 | pj_m128_n4096_k14336 | f32 | 0.4262 (torch-compile) 1.20 | 0.4181 (torch-compile-cg) 1.22 | loss → loss |
| h100 | pj_m128_n4096_k14336_bf16 | bf16 | 0.0763 (torch-bf16) 3.74 | 0.0757 (torch-bf16-compile-ro) 3.77 | loss → loss |
| h100 | pj_m16_n18432_k7168 | f32 | 0.3361 (torch-compile) 0.97 | 0.3356 (torch-compile-cg) 0.97 | even → even |
| h100 | pj_m16_n2048_k7168 | f32 | 0.0499 (torch-compile) 1.37 | 0.0487 (torch-compile-ro) 1.40 | loss → loss |
| h100 | pj_m16_n28672_k4096_bf16 | bf16 | 0.0918 (torch-bf16-compile) 2.23 | 0.0902 (torch-bf16-compile-cg) 2.27 | loss → loss |
| h100 | pj_m1_n4096_k14336_bf16 | bf16 | 0.053 (torch-bf16-compile) 1.49 | 0.0524 (torch-bf16-compile-cg) 1.51 | loss → loss |
| h100 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0203 (torch-bf16) 1.57 | 0.0202 (torch-bf16-compile-ro) 1.58 | loss → loss |
| h100 | mlp2_2048_tf32 | tf32 | 0.3729 (torch-compile-tf32) 2.11 | 0.3685 (torch-compile-ro-tf32) 2.13 | loss → loss |
| h100 | mlp_2048 | f32 | 0.8576 (torch-compile) 1.08 | 0.8412 (torch-compile-ro) 1.10 | loss → loss |
| h100 | mlp_2048_tf32 | tf32 | 0.1868 (torch-eager-tf32) 2.17 | 0.1742 (torch-compile-ro-tf32) 2.33 | loss → loss |
| h100 | mlp_512 | f32 | 0.0312 (torch-compile) 1.70 | 0.0301 (torch-compile-cg) 1.76 | loss → loss |
| h100 | attn_1024_64 | f32 | 0.036 (chain-fused) 1.50 | 0.0247 (torch-compile-cg) 2.18 | loss → loss |
| h100 | attn_2048_128 | f32 | 0.0879 (torch-compile) 1.44 | 0.0848 (torch-compile-cg) 1.49 | loss → loss |
| h100 | attn_2048_64 | f32 | 0.0634 (torch-sdpa-compile) 1.37 | 0.0586 (torch-compile-ro) 1.48 | loss → loss |
| h100 | causal_1024_256 | f32 | 118 (torch-compile) 0.18 | 116 (torch-compile-ro) 0.18 | win → win |
| h100 | gen_1024_512_256 | f32 | 58.21 (torch-compile) 0.19 | 56.81 (torch-compile-cg) 0.19 | win → win |
| h100 | lsm_1_131072 | f32 | 0.0225 (torch-compile) 1.41 | 0.0188 (torch-compile-cg) 1.69 | loss → loss |
| h100 | lsm_32_8192 | f32 | 0.0099 (torch-sdpa-compile) 2.73 | 0.0097 (torch-compile-ro) 2.78 | loss → loss |
| h100 | lsm_4_32768 | f32 | 0.02 (torch-compile) 1.38 | 0.0154 (torch-sdpa-compile-cg) 1.79 | loss → loss |
| h100 | prefill_1024_512_256 | f32 | 58.94 (torch-compile) 0.19 | 57.04 (torch-compile-ro) 0.19 | win → win |
| h100 | rowsum_2048 | f32 | 0.4213 (torch-compile) 1.21 | 0.4199 (torch-compile-cg) 1.21 | loss → loss |
| h100 | rowsum_4096 | f32 | 4.016 (torch-compile) 1.15 | 4.006 (torch-compile-ro) 1.15 | loss → loss |
| h100 | softmax_rows_2048 | f32 | 0.4284 (torch-compile) 1.22 | 0.4272 (torch-compile-ro) 1.22 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.3788 (torch-bf16-compile) 4.06 | 0.3021 (torch-bf16-compile-ro) 5.09 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.3788 (torch-bf16-compile) 4.06 | 0.3021 (torch-bf16-compile-ro) 5.09 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.3788 (torch-bf16-compile) 4.03 | 0.3021 (torch-bf16-compile-ro) 5.05 | loss → loss |
| h100 | llm_blkpre_t2048 | f32 | 29.68 (torch-compile) 1.00 | 27.99 (liger) 1.06 | even → loss |
| h100 | llm_blkpre_t2048_bf16 | bf16 | 2.892 (torch-bf16-compile) 3.48 | 2.794 (torch-bf16-compile-cg) 3.61 | loss → loss |
| h100 | llm_blkpre_t2048_bf16w | bf16w | 2.892 (torch-bf16-compile) 9.58 | 2.794 (torch-bf16-compile-cg) 9.92 | loss → loss |
| h100 | llm_blkpre_t2048_tf32 | tf32 | 7.278 (torch-compile-tf32) 1.43 | 6.315 (liger-tf32) 1.64 | loss → loss |
| h100 | llm_dec_b1_l2048 | f32 | 0.2219 (torch-compile) 1.47 | 0.2068 (torch-compile-cg) 1.57 | loss → loss |
| h100 | llm_dec_b1_l2048_bf16 | bf16 | 0.2258 (torch-bf16-compile) 1.47 | 0.1484 (torch-bf16-compile-ro) 2.24 | loss → loss |
| h100 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.2258 (torch-bf16-compile) 1.47 | 0.1484 (torch-bf16-compile-ro) 2.23 | loss → loss |
| h100 | llm_dec_b1_l2048_bf16w | bf16w | 0.2258 (torch-bf16-compile) 1.44 | 0.1484 (torch-bf16-compile-ro) 2.19 | loss → loss |
| h100 | llm_dec_b1_l8192_bf16 | bf16 | 0.2256 (torch-bf16-compile) 1.76 | 0.16 (torch-bf16-compile-ro) 2.48 | loss → loss |
| h100 | llm_dec_b1_l8192_bf16kv | bf16kv | 0.2256 (torch-bf16-compile) 1.77 | 0.16 (torch-bf16-compile-ro) 2.50 | loss → loss |
| h100 | llm_dec_b1_l8192_bf16w | bf16w | 0.2256 (torch-bf16-compile) 1.62 | 0.16 (torch-bf16-compile-ro) 2.29 | loss → loss |
| h100 | llm_dec_b32_l2048 | f32 | 1.429 (torch-compile) 0.39 | 1.417 (torch-compile-cg) 0.39 | win → win |
| h100 | llm_dec_b32_l2048_bf16 | bf16 | 0.2886 (torch-bf16-compile) 3.25 | 0.2671 (torch-bf16-compile-cg) 3.51 | loss → loss |
| h100 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.2886 (torch-bf16-compile) 3.12 | 0.2671 (torch-bf16-compile-cg) 3.37 | loss → loss |
| h100 | llm_dec_b32_l2048_bf16w | bf16w | 0.2886 (torch-bf16-compile) 1.92 | 0.2671 (torch-bf16-compile-cg) 2.07 | loss → loss |
| h100 | llm_dec_b32_l8192 | f32 | 5.567 (torch-compile) 0.27 | 5.389 (torch-compile-cg) 0.27 | win → win |
| h100 | llm_dec_b32_l8192_bf16 | bf16 | 0.5941 (torch-bf16-compile) 3.39 | 0.5752 (torch-bf16-compile-cg) 3.50 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16kv | bf16kv | 0.5941 (torch-bf16-compile) 3.33 | 0.5752 (torch-bf16-compile-cg) 3.44 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16w | bf16w | 0.5941 (torch-bf16-compile) 2.44 | 0.5752 (torch-bf16-compile-cg) 2.52 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16 | bf16 | 0.2517 (torch-bf16-compile) 1.11 | 0.1675 (torch-bf16-compile-ro) 1.67 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16kv | bf16kv | 0.2517 (torch-bf16-compile) 1.12 | 0.1675 (torch-bf16-compile-ro) 1.68 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16w | bf16w | 0.2517 (torch-bf16-compile) 0.74 | 0.1675 (torch-bf16-compile-ro) 1.11 | win → loss |
| h100 | llm_dec_b8_l8192 | f32 | 1.455 (torch-compile) 0.35 | 1.437 (torch-compile-cg) 0.35 | win → win |
| h100 | llm_dec_b8_l8192_bf16 | bf16 | 0.2946 (torch-bf16-compile) 2.84 | 0.2691 (torch-bf16-compile-ro) 3.11 | loss → loss |
| h100 | llm_dec_b8_l8192_bf16kv | bf16kv | 0.2946 (torch-bf16-compile) 2.84 | 0.2691 (torch-bf16-compile-ro) 3.11 | loss → loss |
| h100 | llm_dec_b8_l8192_bf16w | bf16w | 0.2946 (torch-bf16-compile) 1.63 | 0.2691 (torch-bf16-compile-ro) 1.78 | loss → loss |
| h100 | llm_pre_t2048 | f32 | 8.505 (torch-compile) 1.01 | 6.565 (liger) 1.30 | even → loss |
| h100 | llm_pre_t2048_bf16 | bf16 | 0.6811 (torch-bf16-compile) 6.47 | 0.649 (torch-bf16-compile-cg) 6.79 | loss → loss |
| h100 | llm_pre_t2048_bf16w | bf16w | 0.6811 (torch-bf16-compile) 11.76 | 0.649 (torch-bf16-compile-cg) 12.34 | loss → loss |
| h100 | llm_pre_t2048_tf32 | tf32 | 2.852 (torch-compile-tf32) 1.56 | 2.188 (triton-tf32) 2.04 | loss → loss |
| h100 | llm_pre_t8192 | f32 | 70.2 (torch-compile) 1.02 | 39.46 (liger) 1.81 | even → loss |
| h100 | ew_4096 | f32 | 0.0715 (torch-compile) 1.11 | 0.0712 (torch-compile-cg) 1.11 | even → even |
| h100 | ewconvew_2048_5 | f32 | 0.1172 (torch-compile) 0.46 | 0.1159 (torch-compile-cg) 0.47 | win → win |
| h100 | firew_rowsum_2048x2048_16 | f32 | 0.0655 (torch-compile) 0.57 | 0.0644 (torch-compile-ro) 0.58 | win → win |
| h100 | fanout_2048 | f32 | 0.8459 (torch-compile) 1.14 | 0.8452 (torch-compile-cg) 1.15 | loss → loss |
| h100 | indep_2048 | f32 | 0.845 (torch-compile) 1.17 | 0.8423 (torch-compile-cg) 1.17 | loss → loss |
| h100 | loopmm_1024_8 | f32 | 0.4985 (torch-compile) 1.25 | 0.4938 (torch-compile-cg) 1.26 | loss → loss |
| h100 | rt_mlp2_1024 | f32 | 2.489 (torch-compile) 1.05 | 2.435 (torch-compile-cg) 1.07 | even → loss |
| h100 | rt_mlp2_2048 | f32 | 16.46 (torch-compile) 0.92 | 16.33 (torch-compile-ro) 0.93 | win → win |

Mapal wins that disappear: 9 — rtx-5080 llm_pre_t2048 (liger), rtx-5080 rt_mlp2_2048 (torch-compile-ro), rtx-3090 lsm_1_131072 (torch-compile-cg), rtx-3090 lsm_4_32768 (torch-compile-cg), rtx-3090 mlp_2048_tf32 (torch-compile-ro-tf32), rtx-3090 rt_mlp2_1024 (torch-compile-cg), a100 lsm_1_131072 (torch-compile-cg), a100 lsm_4_32768 (torch-compile-cg), h100 llm_dec_b8_l2048_bf16w (torch-bf16-compile-ro).

## Every competitor leg

Per row and part, every leg at the row's precision, grouped by competitor family: `leg ms (Mapal/leg)`; **bold** = the row's fastest leg; `✗gate` = timed but outside the value gate (not counted); `cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg for this row on another part; `unavailable` = the part's run reported the package missing. Families: vendor libs; torch eager; torch.compile; compile + CUDA graphs; reduce-overhead; SDPA; FlashAttention-2/3; FlashInfer; hand Triton; Liger; TensorRT-LLM; hand CUDA. Columns with no leg on any row of a table are left out.

### GEMM cells & projections

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| epi_4096 | rtx-5080 | 4.07 | chain-lt 3.762 (1.08)<br>chain-eager 3.903 (1.04)<br>chain-cutlass 3.927 (1.04) | **torch-eager 3.748 (1.09)** | torch-compile 3.834 (1.06) | torch-compile-cg 3.851 (1.06) | torch-compile-ro 3.944 (1.03) | triton 4.355 (0.93) |
| epi_4096 | rtx-3090 | 5.326 | **chain-lt 5.771 (0.92)**<br>chain-cutlass 5.899 (0.90)<br>chain-eager 5.92 (0.90) | torch-eager 5.974 (0.89) | torch-compile 5.879 (0.91) | torch-compile-cg 5.825 (0.91) | torch-compile-ro 5.829 (0.91) | triton 6.839 (0.78) |
| epi_4096 | a100 | 7.883 | **chain-lt 7.297 (1.08)**<br>chain-eager 7.388 (1.07)<br>chain-cutlass 7.64 (1.03) | torch-eager 7.48 (1.05) | torch-compile 7.67 (1.03) | torch-compile-cg 7.589 (1.04) | torch-compile-ro 7.375 (1.07) | triton 7.444 (1.06) |
| epi_4096 | rtx-4090 | 2.427 | chain-lt 2.423 (1.00)<br>chain-cutlass 2.454 (0.99)<br>chain-eager 2.538 (0.96) | torch-eager 2.615 (0.93) | **torch-compile 2.415 (1.01)** | torch-compile-cg 2.456 (0.99) | torch-compile-ro 2.469 (0.98) | triton 2.92 (0.83) |
| epi_4096 | h100 | 3.624 | **chain-lt 3.935 (0.92)**<br>chain-eager 4.003 (0.91)<br>chain-cutlass 4.534 (0.80) | torch-eager 4.083 (0.89) | torch-compile 4.057 (0.89) | torch-compile-cg 4.04 (0.90) | torch-compile-ro 4.026 (0.90) | triton 4.595 (0.79) |
| mm_16x2048x7168 | rtx-5080 | 0.0881 | **cublas 0.0799 (1.10)**<br>cutlass-tuned 0.385 (0.23)<br>cutlass 0.5243 (0.17) |  |  |  |  | triton 0.1536 (0.57) |
| mm_16x2048x7168 | rtx-3090 | 0.0886 | **cublas 0.086 (1.03)**<br>cutlass-tuned 0.2119 (0.42)<br>cutlass 0.6727 (0.13) |  |  |  |  | triton 0.2079 (0.43) |
| mm_16x2048x7168 | a100 | 0.09215 | **cublas 0.0911 (1.01)**<br>cutlass-tuned 0.3502 (0.26)<br>cutlass 1.351 (0.07) |  |  |  |  | triton 0.3983 (0.23) |
| mm_16x2048x7168 | rtx-4090 | 0.1075 | **cublas 0.09525 (1.13)**<br>cutlass-tuned 0.2007 (0.54)<br>cutlass 0.5002 (0.21) |  |  |  |  | triton 0.1485 (0.72) |
| mm_16x2048x7168 | h100 | 0.1017 | **cublas 0.06395 (1.59)**<br>cutlass-tuned 0.2863 (0.36)<br>cutlass 0.7724 (0.13) |  |  |  |  | triton 0.2111 (0.48) |
| mm_4096x4096x4096 | rtx-5080 | 3.819 | **cublas 3.659 (1.04)**<br>cutlass 3.736 (1.02)<br>cutlass-tuned 9.09 (0.42) |  |  |  |  | triton 4.298 (0.89) |
| mm_4096x4096x4096 | rtx-3090 | 6.274 | **cublas 6.034 (1.04)**<br>cutlass 6.178 (1.02)<br>cutlass-tuned 16 (0.39) |  |  |  |  | triton 7 (0.90) |
| mm_4096x4096x4096 | a100 | 7.68 | **cublas 7.313 (1.05)**<br>cutlass 7.676 (1.00)<br>cutlass-tuned 12.01 (0.64) |  |  |  |  | triton 7.47 (1.03) |
| mm_4096x4096x4096 | rtx-4090 | 2.411 | **cublas 2.473 (0.97)**<br>cutlass 2.497 (0.97)<br>cutlass-tuned 6.722 (0.36) |  |  |  |  | triton 2.938 (0.82) |
| mm_4096x4096x4096 | h100 | 4.769 | **cublas 4.144 (1.15)**<br>cutlass 4.8 (0.99)<br>cutlass-tuned 10.83 (0.44) |  |  |  |  | triton 4.928 (0.97) |
| pj_m128_n28672_k4096 | rtx-5080 | 0.8098 | **lt 0.7496 (1.08)**<br>lt-fresh 0.7516 (1.08)<br>cublas 0.7926 (1.02)<br>cublas-fresh 0.7967 (1.02) | torch-eager 0.8049 (1.01) | torch-compile 0.8049 (1.01) | torch-compile-cg 0.8028 (1.01) | torch-compile-ro 0.8008 (1.01) | triton 1.065 (0.76) |
| pj_m128_n28672_k4096 | rtx-3090 | 1.204 | cublas 1.345 (0.90)<br>lt 1.345 (0.90)<br>cublas-fresh 1.345 (0.90)<br>lt-fresh 1.345 (0.89) | **torch-eager 1.226 (0.98)** | torch-compile 1.242 (0.97) | torch-compile-cg 1.25 (0.96) | torch-compile-ro 1.356 (0.89) | triton 1.669 (0.72) |
| pj_m128_n28672_k4096 | a100 | 1.729 | cublas 1.781 (0.97)<br>lt 1.781 (0.97)<br>cublas-fresh 1.781 (0.97)<br>lt-fresh 1.781 (0.97) | torch-eager 1.8 (0.96) | torch-compile 2.354 (0.73) | torch-compile-cg 1.821 (0.95) | **torch-compile-ro 1.748 (0.99)** | triton 1.887 (0.92) |
| pj_m128_n28672_k4096 | rtx-4090 | 0.661 | **lt-fresh 0.6513 (1.01)**<br>lt 0.6595 (1.00)<br>cublas-fresh 0.6861 (0.96)<br>cublas 0.6932 (0.95) | torch-eager 0.6932 (0.95) | torch-compile 0.6943 (0.95) | torch-compile-cg 0.7291 (0.91) | torch-compile-ro 0.7291 (0.91) | triton 0.7608 (0.87) |
| pj_m128_n28672_k4096 | h100 | 0.989 | cublas 1.03 (0.96)<br>lt 1.03 (0.96)<br>lt-fresh 1.041 (0.95)<br>cublas-fresh 1.042 (0.95) | torch-eager 1.069 (0.93) | **torch-compile 0.9224 (1.07)** | torch-compile-cg 0.9258 (1.07) | torch-compile-ro 0.9232 (1.07) | triton 1.272 (0.78) |
| pj_m128_n4096_k14336 | rtx-5080 | 0.4594 | **cublas-fresh 0.4014 (1.14)**<br>lt-fresh 0.4014 (1.14)<br>cublas 0.4055 (1.13)<br>lt 0.4055 (1.13) | torch-eager 0.4117 (1.12) | torch-compile 0.4117 (1.12) | torch-compile-cg 0.4117 (1.12) | torch-compile-ro 0.4117 (1.12) | triton 0.5898 (0.78) |
| pj_m128_n4096_k14336 | rtx-3090 | 0.6789 | cublas 0.6799 (1.00)<br>lt 0.6799 (1.00)<br>cublas-fresh 0.681 (1.00)<br>lt-fresh 0.681 (1.00) | **torch-eager 0.6369 (1.07)** | torch-compile 0.6707 (1.01) | torch-compile-cg 0.6707 (1.01) | torch-compile-ro 0.6707 (1.01) | triton 0.8919 (0.76) |
| pj_m128_n4096_k14336 | a100 | 0.9203 | **cublas-fresh 0.8509 (1.08)**<br>lt-fresh 0.8509 (1.08)<br>cublas 0.854 (1.08)<br>lt 0.854 (1.08) | torch-eager 1.101 (0.84) | torch-compile 1.102 (0.84) | torch-compile-cg 1.103 (0.83) | torch-compile-ro 1.103 (0.83) | triton 1.44 (0.64) |
| pj_m128_n4096_k14336 | rtx-4090 | 0.3792 | **lt-fresh 0.3543 (1.07)**<br>lt 0.3615 (1.05)<br>cublas-fresh 0.3686 (1.03)<br>cublas 0.3758 (1.01) | torch-eager 0.3738 (1.01) | torch-compile 0.3779 (1.00) | torch-compile-cg 0.3758 (1.01) | torch-compile-ro 0.3768 (1.01) | triton 0.3697 (1.03) |
| pj_m128_n4096_k14336 | h100 | 0.511 | lt 0.4454 (1.15)<br>cublas 0.4462 (1.15)<br>cublas-fresh 0.4512 (1.13)<br>lt-fresh 0.4519 (1.13) | torch-eager 0.4408 (1.16) | torch-compile 0.4262 (1.20) | **torch-compile-cg 0.4181 (1.22)** | torch-compile-ro 0.4223 (1.21) | triton 0.5465 (0.94) |
| pj_m128_n4096_k4096 | rtx-5080 | 0.1501 | **cublas-fresh 0.1352 (1.11)**<br>lt-fresh 0.1352 (1.11)<br>cublas 0.1454 (1.03)<br>lt 0.1454 (1.03) | torch-eager 0.1454 (1.03) | torch-compile 0.1454 (1.03) | torch-compile-cg 0.1454 (1.03) | torch-compile-ro 0.1454 (1.03) | triton 0.1761 (0.85) |
| pj_m128_n4096_k4096 | rtx-3090 | 0.2168 | **cublas 0.2028 (1.07)**<br>lt 0.2028 (1.07)<br>cublas-fresh 0.2028 (1.07)<br>lt-fresh 0.2028 (1.07) | torch-eager 0.2243 (0.97) | torch-compile 0.2243 (0.97) | torch-compile-cg 0.2232 (0.97) | torch-compile-ro 0.2232 (0.97) | triton 0.2529 (0.86) |
| pj_m128_n4096_k4096 | a100 | 0.305 | **cublas 0.2601 (1.17)**<br>cublas-fresh 0.2601 (1.17)<br>lt 0.2611 (1.17)<br>lt-fresh 0.2611 (1.17) | torch-eager 0.3328 (0.92) | torch-compile 0.3328 (0.92) | torch-compile-cg 0.3318 (0.92) | torch-compile-ro 0.3318 (0.92) | triton 0.4178 (0.73) |
| pj_m128_n4096_k4096 | rtx-4090 | 0.1359 | **lt-fresh 0.0973 (1.40)**<br>cublas-fresh 0.1014 (1.34)<br>lt 0.1239 (1.10)<br>cublas 0.1362 (1.00) | torch-eager 0.1372 (0.99) | torch-compile 0.1126 (1.21) | torch-compile-cg 0.1118 (1.22) | torch-compile-ro 0.1362 (1.00) | triton 0.1137 (1.20) |
| pj_m128_n4096_k4096 | h100 | 0.1749 | **cublas-fresh 0.1178 (1.48)**<br>lt-fresh 0.1182 (1.48)<br>lt 0.1253 (1.40)<br>cublas 0.1292 (1.35) | torch-eager 0.1263 (1.38) | torch-compile 0.1277 (1.37) | torch-compile-cg 0.1263 (1.38) | torch-compile-ro 0.1254 (1.39) | triton 0.1635 (1.07) |
| pj_m128_n6144_k4096 | rtx-5080 | 0.2164 | **cublas-fresh 0.1802 (1.20)**<br>lt-fresh 0.1802 (1.20)<br>cublas 0.1864 (1.16)<br>lt 0.1864 (1.16) | torch-eager 0.1884 (1.15) | torch-compile 0.1884 (1.15) | torch-compile-cg 0.1884 (1.15) | torch-compile-ro 0.1884 (1.15) | triton 0.2949 (0.73) |
| pj_m128_n6144_k4096 | rtx-3090 | 0.318 | **cublas-fresh 0.3092 (1.03)**<br>lt-fresh 0.31 (1.03)<br>lt 0.3101 (1.03)<br>cublas 0.3103 (1.02) | torch-eager 0.3164 (1.01) | torch-compile 0.3164 (1.01) | torch-compile-cg 0.3154 (1.01) | torch-compile-ro 0.3154 (1.01) | triton 0.4157 (0.77) |
| pj_m128_n6144_k4096 | a100 | 0.4302 | **lt 0.3727 (1.15)**<br>lt-fresh 0.3727 (1.15)<br>cublas 0.4557 (0.94)<br>cublas-fresh 0.4557 (0.94) | torch-eager 0.5816 (0.74) | torch-compile 0.5417 (0.79) | torch-compile-cg 0.5417 (0.79) | torch-compile-ro 0.5806 (0.74) | triton 0.4506 (0.95) |
| pj_m128_n6144_k4096 | rtx-4090 | 0.1942 | **lt-fresh 0.1526 (1.27)**<br>cublas-fresh 0.1536 (1.26)<br>lt 0.1802 (1.08)<br>cublas 0.1894 (1.03) | torch-eager 0.1894 (1.03) | torch-compile 0.1782 (1.09) | torch-compile-cg 0.1772 (1.10) | torch-compile-ro 0.1884 (1.03) | triton 0.171 (1.14) |
| pj_m128_n6144_k4096 | h100 | 0.2376 | **cublas 0.1884 (1.26)**<br>lt-fresh 0.1891 (1.26)<br>cublas-fresh 0.1894 (1.25)<br>lt 0.1902 (1.25) | torch-eager 0.1915 (1.24) | torch-compile 0.1907 (1.25) | torch-compile-cg 0.1898 (1.25) | torch-compile-ro 0.1902 (1.25) | triton 0.2917 (0.81) |
| pj_m16_n18432_k7168 | rtx-5080 | 0.6049 | **lt-fresh 0.5878 (1.03)**<br>cublas-fresh 0.5898 (1.03)<br>lt 0.6185 (0.98)<br>cublas 0.6205 (0.97) | torch-eager 0.6226 (0.97) | torch-compile 0.6103 (0.99) | torch-compile-cg 0.6103 (0.99) | torch-compile-ro 0.6205 (0.97) | triton 0.6205 (0.97) |
| pj_m16_n18432_k7168 | rtx-3090 | 0.627 | cublas 0.6226 (1.01)<br>lt-fresh 0.6226 (1.01)<br>lt 0.6227 (1.01)<br>cublas-fresh 0.6236 (1.01) | torch-eager 0.6748 (0.93) | **torch-compile 0.6124 (1.02)** | torch-compile-cg 0.6164 (1.02) | torch-compile-ro 0.6226 (1.01) | triton 0.6318 (0.99) |
| pj_m16_n18432_k7168 | a100 | 0.43 | lt 0.5007 (0.86)<br>lt-fresh 0.5028 (0.86)<br>cublas 0.5263 (0.82)<br>cublas-fresh 0.5274 (0.82) | torch-eager 0.6595 (0.65) | **torch-compile 0.4454 (0.97)** | torch-compile-cg 0.4588 (0.94) | torch-compile-ro 0.5427 (0.79) | triton 0.7301 (0.59) |
| pj_m16_n18432_k7168 | rtx-4090 | 0.6464 | **lt-fresh 0.6103 (1.06)**<br>lt 0.6216 (1.04)<br>cublas-fresh 0.6287 (1.03)<br>cublas 0.6359 (1.02) | torch-eager 0.686 (0.94) | torch-compile 0.6175 (1.05) | torch-compile-cg 0.6728 (0.96) | torch-compile-ro 0.6861 (0.94) | triton 0.7014 (0.92) |
| pj_m16_n18432_k7168 | h100 | 0.3269 | lt 0.3684 (0.89)<br>lt-fresh 0.3931 (0.83)<br>cublas 0.4055 (0.81)<br>cublas-fresh 0.4268 (0.77) | torch-eager 0.4251 (0.77) | torch-compile 0.3361 (0.97) | **torch-compile-cg 0.3356 (0.97)** | torch-compile-ro 0.422 (0.77) | triton 0.5612 (0.58) |
| pj_m16_n2048_k7168 | rtx-5080 | 0.05084 | **cublas-fresh 0.0492 (1.03)**<br>lt-fresh 0.0492 (1.03)<br>cublas 0.0799 (0.64)<br>lt 0.0799 (0.64) | torch-eager 0.0799 (0.64) | torch-compile 0.0799 (0.64) | torch-compile-cg 0.0796 (0.64) | torch-compile-ro 0.0816 (0.62) | triton 0.1556 (0.33) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.09591 | **lt 0.0819 (1.17)**<br>lt-fresh 0.0819 (1.17)<br>cublas 0.086 (1.12)<br>cublas-fresh 0.086 (1.12) | torch-eager 0.0891 (1.08) | torch-compile 0.0881 (1.09) | torch-compile-cg 0.087 (1.10) | torch-compile-ro 0.087 (1.10) | triton 0.2048 (0.47) |
| pj_m16_n2048_k7168 | a100 | 0.1007 | **cublas-fresh 0.087 (1.16)**<br>lt-fresh 0.087 (1.16)<br>cublas 0.0901 (1.12)<br>lt 0.0911 (1.11) | torch-eager 0.1096 (0.92) | torch-compile 0.1085 (0.93) | torch-compile-cg 0.1065 (0.95) | torch-compile-ro 0.1065 (0.95) | triton 0.3727 (0.27) |
| pj_m16_n2048_k7168 | rtx-4090 | 0.06263 | **cublas-fresh 0.0297 (2.11)**<br>lt-fresh 0.0297 (2.11)<br>lt 0.0952 (0.66)<br>cublas 0.0963 (0.65) | torch-eager 0.0963 (0.65) | torch-compile 0.0963 (0.65) | torch-compile-cg 0.0952 (0.66) | torch-compile-ro 0.0952 (0.66) | triton 0.1466 (0.43) |
| pj_m16_n2048_k7168 | h100 | 0.0683 | lt 0.0511 (1.34)<br>lt-fresh 0.0523 (1.31)<br>cublas 0.0552 (1.24)<br>cublas-fresh 0.0553 (1.24) | torch-eager 0.0504 (1.36) | torch-compile 0.0499 (1.37) | torch-compile-cg 0.0488 (1.40) | **torch-compile-ro 0.0487 (1.40)** | triton 0.2004 (0.34) |
| pj_m16_n28672_k4096 | rtx-5080 | 0.5392 | **lt-fresh 0.5243 (1.03)**<br>lt 0.5489 (0.98)<br>cublas-fresh 1.863 (0.29)<br>cublas 1.928 (0.28) | torch-eager 1.925 (0.28) | torch-compile 0.5448 (0.99) | torch-compile-cg 0.5427 (0.99) | torch-compile-ro 1.929 (0.28) | triton 0.553 (0.98) |
| pj_m16_n28672_k4096 | rtx-3090 | 0.5762 | lt 0.5537 (1.04)<br>cublas 0.554 (1.04)<br>cublas-fresh 0.554 (1.04)<br>lt-fresh 0.554 (1.04) | torch-eager 0.555 (1.04) | **torch-compile 0.5417 (1.06)** | torch-compile-cg 0.544 (1.06) | torch-compile-ro 0.555 (1.04) | triton 0.5645 (1.02) |
| pj_m16_n28672_k4096 | a100 | 0.4215 | cublas 0.4721 (0.89)<br>lt 0.4721 (0.89)<br>cublas-fresh 0.4731 (0.89)<br>lt-fresh 0.4731 (0.89) | torch-eager 0.5847 (0.72) | **torch-compile 0.426 (0.99)** | torch-compile-cg 0.4393 (0.96) | torch-compile-ro 0.5693 (0.74) | triton 0.6164 (0.68) |
| pj_m16_n28672_k4096 | rtx-4090 | 0.5827 | **cublas-fresh 0.5499 (1.06)**<br>lt-fresh 0.5499 (1.06)<br>cublas 0.5568 (1.05)<br>lt 0.5571 (1.05) | torch-eager 0.6031 (0.97) | torch-compile 0.5548 (1.05) | torch-compile-cg 0.5878 (0.99) | torch-compile-ro 0.6031 (0.97) | triton 0.5704 (1.02) |
| pj_m16_n28672_k4096 | h100 | 0.2809 | lt 0.3389 (0.83)<br>lt-fresh 0.3534 (0.79)<br>cublas 0.3582 (0.78)<br>cublas-fresh 0.3694 (0.76) | torch-eager 0.3843 (0.73) | **torch-compile 0.2602 (1.08)** | torch-compile-cg 0.2604 (1.08) | torch-compile-ro 0.3812 (0.74) | triton 0.3685 (0.76) |
| pj_m16_n4096_k14336 | rtx-5080 | 0.2662 | cublas-fresh 0.3154 (0.84)<br>lt-fresh 0.3154 (0.84)<br>cublas 0.3318 (0.80)<br>lt 0.3318 (0.80) | torch-eager 0.3338 (0.80) | **torch-compile 0.2744 (0.97)** | torch-compile-cg 0.2744 (0.97) | torch-compile-ro 0.3318 (0.80) | triton 0.3052 (0.87) |
| pj_m16_n4096_k14336 | rtx-3090 | 0.2895 | lt 0.2877 (1.01)<br>lt-fresh 0.2877 (1.01)<br>cublas 0.2908 (1.00)<br>cublas-fresh 0.2908 (1.00) | torch-eager 0.2908 (1.00) | torch-compile 0.2744 (1.05) | **torch-compile-cg 0.2734 (1.06)** | torch-compile-ro 0.2898 (1.00) | triton 0.427 (0.68) |
| pj_m16_n4096_k14336 | a100 | 0.2276 | **lt 0.2509 (0.91)**<br>lt-fresh 0.2509 (0.91)<br>cublas-fresh 0.2529 (0.90)<br>cublas 0.254 (0.90) | torch-eager 0.257 (0.89) | torch-compile 0.3154 (0.72) | torch-compile-cg 0.3123 (0.73) | torch-compile-ro 0.3123 (0.73) | triton 0.7393 (0.31) |
| pj_m16_n4096_k14336 | rtx-4090 | 0.3279 | **cublas-fresh 0.3011 (1.09)**<br>lt-fresh 0.3021 (1.09)<br>lt 0.3092 (1.06)<br>cublas 0.3113 (1.05) | torch-eager 0.3165 (1.04) | torch-compile 0.3092 (1.06) | torch-compile-cg 0.3092 (1.06) | torch-compile-ro 0.3154 (1.04) | triton 0.3277 (1.00) |
| pj_m16_n4096_k14336 | h100 | 0.1651 | lt-fresh 0.1784 (0.93)<br>lt 0.1877 (0.88)<br>cublas-fresh 0.1911 (0.86)<br>cublas 0.2066 (0.80) | torch-eager 0.1916 (0.86) | **torch-compile 0.1691 (0.98)** | torch-compile-cg 0.1693 (0.97) | torch-compile-ro 0.19 (0.87) | triton 0.4817 (0.34) |
| pj_m16_n4096_k4096 | rtx-5080 | 0.05609 | **cublas-fresh 0.0553 (1.01)**<br>lt-fresh 0.0553 (1.01)<br>cublas 0.0963 (0.58)<br>lt 0.0963 (0.58) | torch-eager 0.0963 (0.58) | torch-compile 0.0819 (0.68) | torch-compile-cg 0.0819 (0.68) | torch-compile-ro 0.0983 (0.57) | triton 0.0922 (0.61) |
| pj_m16_n4096_k4096 | rtx-3090 | 0.1102 | cublas 0.1003 (1.10)<br>cublas-fresh 0.1004 (1.10)<br>lt-fresh 0.1004 (1.10)<br>lt 0.1014 (1.09) | torch-eager 0.1024 (1.08) | torch-compile 0.0851 (1.30) | **torch-compile-cg 0.085 (1.30)** | torch-compile-ro 0.1004 (1.10) | triton 0.1239 (0.89) |
| pj_m16_n4096_k4096 | a100 | 0.1005 | **cublas-fresh 0.0891 (1.13)**<br>lt-fresh 0.0891 (1.13)<br>lt 0.0922 (1.09)<br>cublas 0.0932 (1.08) | torch-eager 0.1075 (0.93) | torch-compile 0.1075 (0.93) | torch-compile-cg 0.1055 (0.95) | torch-compile-ro 0.1044 (0.96) | triton 0.2171 (0.46) |
| pj_m16_n4096_k4096 | rtx-4090 | 0.0707 | **lt-fresh 0.0328 (2.16)**<br>cublas-fresh 0.0358 (1.97)<br>lt 0.1085 (0.65)<br>cublas 0.1137 (0.62) | torch-eager 0.1145 (0.62) | torch-compile 0.1046 (0.68) | torch-compile-cg 0.1044 (0.68) | torch-compile-ro 0.1126 (0.63) | triton 0.1055 (0.67) |
| pj_m16_n4096_k4096 | h100 | 0.06357 | lt 0.0507 (1.25)<br>lt-fresh 0.0511 (1.24)<br>cublas-fresh 0.0919 (0.69)<br>cublas 0.092 (0.69) | torch-eager 0.0899 (0.71) | **torch-compile 0.0494 (1.29)** | torch-compile-cg 0.0495 (1.28) | torch-compile-ro 0.0905 (0.70) | triton 0.1537 (0.41) |
| pj_m16_n6144_k4096 | rtx-5080 | 0.1042 | **lt-fresh 0.0881 (1.18)**<br>cublas-fresh 0.0963 (1.08)<br>lt 0.129 (0.81)<br>cublas 0.1413 (0.74) | torch-eager 0.1413 (0.74) | torch-compile 0.1208 (0.86) | torch-compile-cg 0.1208 (0.86) | torch-compile-ro 0.1413 (0.74) | triton 0.129 (0.81) |
| pj_m16_n6144_k4096 | rtx-3090 | 0.1428 | lt 0.129 (1.11)<br>lt-fresh 0.13 (1.10)<br>cublas 0.1393 (1.03)<br>cublas-fresh 0.1403 (1.02) | torch-eager 0.1463 (0.98) | torch-compile 0.1229 (1.16) | **torch-compile-cg 0.1221 (1.17)** | torch-compile-ro 0.1444 (0.99) | triton 0.1364 (1.05) |
| pj_m16_n6144_k4096 | a100 | 0.13 | lt-fresh 0.1208 (1.08)<br>lt 0.1229 (1.06)<br>cublas-fresh 0.1341 (0.97)<br>cublas 0.1362 (0.95) | torch-eager 0.1475 (0.88) | torch-compile 0.1167 (1.11) | **torch-compile-cg 0.1157 (1.12)** | torch-compile-ro 0.1444 (0.90) | triton 0.2181 (0.60) |
| pj_m16_n6144_k4096 | rtx-4090 | 0.1399 | **lt-fresh 0.1016 (1.38)**<br>cublas-fresh 0.1022 (1.37)<br>lt 0.1589 (0.88)<br>cublas 0.1597 (0.88) | torch-eager 0.1587 (0.88) | torch-compile 0.1526 (0.92) | torch-compile-cg 0.1526 (0.92) | torch-compile-ro 0.1577 (0.89) | triton 0.1587 (0.88) |
| pj_m16_n6144_k4096 | h100 | 0.07577 | lt-fresh 0.0766 (0.99)<br>lt 0.0944 (0.80)<br>cublas-fresh 0.1071 (0.71)<br>cublas 0.1165 (0.65) | torch-eager 0.1124 (0.67) | **torch-compile 0.0567 (1.34)** | torch-compile-cg 0.057 (1.33) | torch-compile-ro 0.1164 (0.65) | triton 0.1468 (0.52) |
| pj_m1_n18432_k7168 | rtx-5080 | 0.5959 | cublas-fresh 0.6124 (0.97)<br>lt-fresh 0.6144 (0.97)<br>cublas 0.6185 (0.96)<br>lt 0.6205 (0.96) | torch-eager 0.6185 (0.96) | **torch-compile 0.6103 (0.98)** | torch-compile-cg 0.6161 (0.97) | torch-compile-ro 0.6185 (0.96) | triton 0.639 (0.93) |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6078 | **lt 0.6052 (1.00)**<br>lt-fresh 0.6058 (1.00)<br>cublas 0.6113 (0.99)<br>cublas-fresh 0.6113 (0.99) | torch-eager 0.6625 (0.92) | torch-compile 0.6072 (1.00) | torch-compile-cg 0.6062 (1.00) | torch-compile-ro 0.6175 (0.98) | triton 0.7854 (0.77) |
| pj_m1_n18432_k7168 | a100 | 0.4381 | **lt-fresh 0.3922 (1.12)**<br>lt 0.3973 (1.10)<br>cublas-fresh 0.4024 (1.09)<br>cublas 0.4209 (1.04) | torch-eager 0.4434 (0.99) | torch-compile 0.4004 (1.09) | torch-compile-cg 0.3983 (1.10) | torch-compile-ro 0.4332 (1.01) | triton 0.7588 (0.58) |
| pj_m1_n18432_k7168 | rtx-4090 | 0.639 | **lt-fresh 0.5376 (1.19)**<br>lt 0.5949 (1.07)<br>cublas-fresh 0.6175 (1.03)<br>cublas 0.6246 (1.02) | torch-eager 0.6788 (0.94) | torch-compile 0.5778 (1.11) | torch-compile-cg 0.5775 (1.11) | torch-compile-ro 0.6799 (0.94) | triton 0.6933 (0.92) |
| pj_m1_n18432_k7168 | h100 | 0.2162 | lt-fresh 0.2004 (1.08)<br>cublas-fresh 0.2033 (1.06)<br>lt 0.2106 (1.03)<br>cublas 0.224 (0.97) | torch-eager 0.2191 (0.99) | **torch-compile 0.1956 (1.11)** | torch-compile-cg 0.1977 (1.09) | torch-compile-ro 0.2185 (0.99) | triton 0.5257 (0.41) |
| pj_m1_n2048_k7168 | rtx-5080 | 0.03616 | **cublas-fresh 0.0266 (1.36)**<br>lt-fresh 0.0266 (1.36)<br>cublas 0.0758 (0.48)<br>lt 0.0758 (0.48) | torch-eager 0.0758 (0.48) | torch-compile 0.0758 (0.48) | torch-compile-cg 0.0758 (0.48) | torch-compile-ro 0.0758 (0.48) | triton 0.2028 (0.18) |
| pj_m1_n2048_k7168 | rtx-3090 | 0.08139 | **lt-fresh 0.0733 (1.11)**<br>lt 0.0761 (1.07)<br>cublas-fresh 0.087 (0.94)<br>cublas 0.0881 (0.92) | torch-eager 0.0878 (0.93) | torch-compile 0.0839 (0.97) | torch-compile-cg 0.0809 (1.01) | torch-compile-ro 0.0891 (0.91) | triton 0.2109 (0.39) |
| pj_m1_n2048_k7168 | a100 | 0.07979 | **cublas-fresh 0.044 (1.81)**<br>lt-fresh 0.044 (1.81)<br>cublas 0.0543 (1.47)<br>lt 0.0553 (1.44) | torch-eager 0.0584 (1.37) | torch-compile 0.0614 (1.30) | torch-compile-cg 0.0594 (1.34) | torch-compile-ro 0.0543 (1.47) | triton 0.3727 (0.21) |
| pj_m1_n2048_k7168 | rtx-4090 | 0.05146 | **lt-fresh 0.0266 (1.93)**<br>cublas-fresh 0.042 (1.23)<br>lt 0.0829 (0.62)<br>cublas 0.1053 (0.49) | torch-eager 0.1044 (0.49) | torch-compile 0.0901 (0.57) | torch-compile-cg 0.0891 (0.58) | torch-compile-ro 0.1026 (0.50) | triton 0.1495 (0.34) |
| pj_m1_n2048_k7168 | h100 | 0.04453 | **cublas-fresh 0.0229 (1.94)**<br>lt-fresh 0.024 (1.86)<br>lt 0.0288 (1.55)<br>cublas 0.031 (1.44) | torch-eager 0.0305 (1.46) | torch-compile 0.0308 (1.45) | torch-compile-cg 0.0294 (1.51) | torch-compile-ro 0.0309 (1.44) | triton 0.1925 (0.23) |
| pj_m1_n28672_k4096 | rtx-5080 | 0.5251 | **lt-fresh 0.5038 (1.04)**<br>cublas-fresh 0.5222 (1.01)<br>lt 0.5386 (0.98)<br>cublas 0.555 (0.95) | torch-eager 0.5591 (0.94) | torch-compile 0.5386 (0.98) | torch-compile-cg 0.5382 (0.98) | torch-compile-ro 0.5571 (0.94) | triton 0.5673 (0.93) |
| pj_m1_n28672_k4096 | rtx-3090 | 0.543 | cublas 0.5376 (1.01)<br>lt 0.5376 (1.01)<br>cublas-fresh 0.5376 (1.01)<br>lt-fresh 0.5376 (1.01) | torch-eager 0.5376 (1.01) | torch-compile 0.5398 (1.01) | torch-compile-cg 0.5396 (1.01) | **torch-compile-ro 0.5366 (1.01)** | triton 0.6554 (0.83) |
| pj_m1_n28672_k4096 | a100 | 0.3957 | lt-fresh 0.3594 (1.10)<br>lt 0.3697 (1.07)<br>cublas 0.4485 (0.88)<br>cublas-fresh 0.4485 (0.88) | torch-eager 0.51 (0.78) | **torch-compile 0.3471 (1.14)** | torch-compile-cg 0.3471 (1.14) | torch-compile-ro 0.4905 (0.81) | triton 0.638 (0.62) |
| pj_m1_n28672_k4096 | rtx-4090 | 0.5735 | **lt-fresh 0.4792 (1.20)**<br>lt 0.5304 (1.08)<br>cublas-fresh 0.5612 (1.02)<br>cublas 0.5663 (1.01) | torch-eager 0.6072 (0.94) | torch-compile 0.5161 (1.11) | torch-compile-cg 0.5161 (1.11) | torch-compile-ro 0.6083 (0.94) | triton 0.5806 (0.99) |
| pj_m1_n28672_k4096 | h100 | 0.2018 | cublas 0.1795 (1.12)<br>lt 0.1814 (1.11)<br>cublas-fresh 0.1837 (1.10)<br>lt-fresh 0.185 (1.09) | torch-eager 0.19 (1.06) | **torch-compile 0.1713 (1.18)** | torch-compile-cg 0.1881 (1.07) | torch-compile-ro 0.1881 (1.07) | triton 0.373 (0.54) |
| pj_m1_n4096_k14336 | rtx-5080 | 0.2575 | **cublas-fresh 0.2314 (1.11)**<br>lt-fresh 0.2314 (1.11)<br>lt 0.2744 (0.94)<br>cublas 0.2764 (0.93) | torch-eager 0.2744 (0.94) | torch-compile 0.2744 (0.94) | torch-compile-cg 0.2744 (0.94) | torch-compile-ro 0.2744 (0.94) | triton 0.4076 (0.63) |
| pj_m1_n4096_k14336 | rtx-3090 | 0.2799 | lt 0.2755 (1.02)<br>lt-fresh 0.2764 (1.01)<br>cublas 0.2765 (1.01)<br>cublas-fresh 0.2767 (1.01) | torch-eager 0.2755 (1.02) | torch-compile 0.2847 (0.98) | torch-compile-cg 0.2836 (0.99) | **torch-compile-ro 0.2744 (1.02)** | triton 0.47 (0.60) |
| pj_m1_n4096_k14336 | a100 | 0.2254 | **lt-fresh 0.1772 (1.27)**<br>cublas-fresh 0.1782 (1.26)<br>lt 0.1843 (1.22)<br>cublas 0.1935 (1.16) | torch-eager 0.1997 (1.13) | torch-compile 0.1843 (1.22) | torch-compile-cg 0.1823 (1.24) | torch-compile-ro 0.1905 (1.18) | triton 0.7393 (0.30) |
| pj_m1_n4096_k14336 | rtx-4090 | 0.3008 | **lt-fresh 0.2038 (1.48)**<br>lt 0.2673 (1.13)<br>cublas-fresh 0.297 (1.01)<br>cublas 0.3125 (0.96) | torch-eager 0.3174 (0.95) | torch-compile 0.2736 (1.10) | torch-compile-cg 0.2724 (1.10) | torch-compile-ro 0.3164 (0.95) | triton 0.3308 (0.91) |
| pj_m1_n4096_k14336 | h100 | 0.1098 | **cublas-fresh 0.0821 (1.34)**<br>lt-fresh 0.0838 (1.31)<br>lt 0.093 (1.18)<br>cublas 0.0934 (1.18) | torch-eager 0.0932 (1.18) | torch-compile 0.0984 (1.12) | torch-compile-cg 0.0902 (1.22) | torch-compile-ro 0.0934 (1.18) | triton 0.4888 (0.22) |
| pj_m1_n4096_k4096 | rtx-5080 | 0.0388 | **lt-fresh 0.0205 (1.89)**<br>cublas-fresh 0.0246 (1.58)<br>cublas 0.086 (0.45)<br>lt 0.086 (0.45) | torch-eager 0.0881 (0.44) | torch-compile 0.086 (0.45) | torch-compile-cg 0.086 (0.45) | torch-compile-ro 0.0881 (0.44) | triton 0.127 (0.31) |
| pj_m1_n4096_k4096 | rtx-3090 | 0.09101 | **lt 0.085 (1.07)**<br>lt-fresh 0.085 (1.07)<br>cublas-fresh 0.0881 (1.03)<br>cublas 0.0901 (1.01) | torch-eager 0.0911 (1.00) | torch-compile 0.0915 (0.99) | torch-compile-cg 0.0942 (0.97) | torch-compile-ro 0.0911 (1.00) | triton 0.1567 (0.58) |
| pj_m1_n4096_k4096 | a100 | 0.08204 | **cublas-fresh 0.0532 (1.54)**<br>lt-fresh 0.0532 (1.54)<br>cublas 0.0676 (1.21)<br>lt 0.0686 (1.20) | torch-eager 0.0727 (1.13) | torch-compile 0.0676 (1.21) | torch-compile-cg 0.0645 (1.27) | torch-compile-ro 0.0666 (1.23) | triton 0.2191 (0.37) |
| pj_m1_n4096_k4096 | rtx-4090 | 0.04346 | **lt-fresh 0.0195 (2.23)**<br>cublas-fresh 0.0215 (2.02)<br>lt 0.0963 (0.45)<br>cublas 0.0983 (0.44) | torch-eager 0.0983 (0.44) | torch-compile 0.0993 (0.44) | torch-compile-cg 0.0983 (0.44) | torch-compile-ro 0.0973 (0.45) | triton 0.1096 (0.40) |
| pj_m1_n4096_k4096 | h100 | 0.04837 | **cublas-fresh 0.0223 (2.17)**<br>lt-fresh 0.0223 (2.17)<br>lt 0.0298 (1.62)<br>cublas 0.0299 (1.62) | torch-eager 0.0299 (1.62) | torch-compile 0.0334 (1.45) | torch-compile-cg 0.0322 (1.50) | torch-compile-ro 0.0299 (1.62) | triton 0.1496 (0.32) |
| pj_m1_n6144_k4096 | rtx-5080 | 0.08914 | **cublas-fresh 0.0512 (1.74)**<br>lt-fresh 0.0512 (1.74)<br>cublas 0.1229 (0.73)<br>lt 0.1229 (0.73) | torch-eager 0.1249 (0.71) | torch-compile 0.1188 (0.75) | torch-compile-cg 0.1188 (0.75) | torch-compile-ro 0.1229 (0.73) | triton 0.1454 (0.61) |
| pj_m1_n6144_k4096 | rtx-3090 | 0.1284 | lt-fresh 0.1219 (1.05)<br>lt 0.1229 (1.05)<br>cublas 0.1362 (0.94)<br>cublas-fresh 0.1362 (0.94) | torch-eager 0.1382 (0.93) | **torch-compile 0.1208 (1.06)** | torch-compile-cg 0.1208 (1.06) | torch-compile-ro 0.1362 (0.94) | triton 0.1997 (0.64) |
| pj_m1_n6144_k4096 | a100 | 0.1023 | **lt-fresh 0.0778 (1.32)**<br>cublas-fresh 0.0788 (1.30)<br>lt 0.0901 (1.14)<br>cublas 0.0922 (1.11) | torch-eager 0.0932 (1.10) | torch-compile 0.0901 (1.14) | torch-compile-cg 0.0881 (1.16) | torch-compile-ro 0.0911 (1.12) | triton 0.2202 (0.46) |
| pj_m1_n6144_k4096 | rtx-4090 | 0.1036 | **lt-fresh 0.0584 (1.77)**<br>cublas-fresh 0.1167 (0.89)<br>lt 0.128 (0.81)<br>cublas 0.1597 (0.65) | torch-eager 0.1597 (0.65) | torch-compile 0.1331 (0.78) | torch-compile-cg 0.1321 (0.78) | torch-compile-ro 0.1587 (0.65) | triton 0.1597 (0.65) |
| pj_m1_n6144_k4096 | h100 | 0.07067 | **lt-fresh 0.0347 (2.04)**<br>cublas-fresh 0.0348 (2.03)<br>cublas 0.042 (1.68)<br>lt 0.042 (1.68) | torch-eager 0.0419 (1.69) | torch-compile 0.046 (1.54) | torch-compile-cg 0.0451 (1.57) | torch-compile-ro 0.0419 (1.69) | triton 0.1425 (0.50) |
| pj_m2048_n28672_k4096 | rtx-5080 | 11.78 | **lt-fresh 12.21 (0.96)**<br>cublas-fresh 12.21 (0.96)<br>cublas 12.21 (0.96)<br>lt 12.21 (0.96) | torch-eager 12.48 (0.94) | torch-compile 12.44 (0.95) | torch-compile-cg 12.49 (0.94) | torch-compile-ro 12.46 (0.95) | triton 15.76 (0.75) |
| pj_m2048_n28672_k4096 | rtx-3090 | 19.22 | cublas 18.88 (1.02)<br>lt-fresh 18.88 (1.02)<br>cublas-fresh 18.88 (1.02)<br>lt 18.88 (1.02) | **torch-eager 18.85 (1.02)** | torch-compile 18.85 (1.02) | torch-compile-cg 18.85 (1.02) | torch-compile-ro 18.85 (1.02) | triton 24.59 (0.78) |
| pj_m2048_n28672_k4096 | a100 | 26.86 | cublas-fresh 26.23 (1.02)<br>cublas 26.23 (1.02)<br>lt 26.39 (1.02)<br>lt-fresh 26.39 (1.02) | torch-eager 26.46 (1.02) | torch-compile 26.02 (1.03) | **torch-compile-cg 26.02 (1.03)** | torch-compile-ro 26.75 (1.00) | triton 26.13 (1.03) |
| pj_m2048_n28672_k4096 | rtx-4090 | 8.956 | **lt 8.183 (1.09)**<br>cublas-fresh 8.184 (1.09)<br>cublas 8.185 (1.09)<br>lt-fresh 8.23 (1.09) | torch-eager 8.236 (1.09) | torch-compile 8.185 (1.09) | torch-compile-cg 8.322 (1.08) | torch-compile-ro 8.345 (1.07) | triton 10.52 (0.85) |
| pj_m2048_n28672_k4096 | h100 | 16.2 | **cublas-fresh 13.87 (1.17)**<br>lt-fresh 13.87 (1.17)<br>cublas 13.87 (1.17)<br>lt 13.87 (1.17) | torch-eager 13.88 (1.17) | torch-compile 13.87 (1.17) | torch-compile-cg 14.38 (1.13) | torch-compile-ro 13.93 (1.16) | triton 16.69 (0.97) |
| pj_m2048_n4096_k14336 | rtx-5080 | 6.45 | **lt-fresh 6.413 (1.01)**<br>cublas-fresh 6.414 (1.01)<br>cublas 6.418 (1.00)<br>lt 6.421 (1.00) | torch-eager 6.499 (0.99) | torch-compile 6.506 (0.99) | torch-compile-cg 6.495 (0.99) | torch-compile-ro 6.496 (0.99) | triton 8.219 (0.78) |
| pj_m2048_n4096_k14336 | rtx-3090 | 10.99 | **lt-fresh 9.916 (1.11)**<br>lt 9.916 (1.11)<br>cublas 10.56 (1.04)<br>cublas-fresh 10.56 (1.04) | torch-eager 10.49 (1.05) | torch-compile 10.39 (1.06) | torch-compile-cg 10.39 (1.06) | torch-compile-ro 10.39 (1.06) | triton 12.62 (0.87) |
| pj_m2048_n4096_k14336 | a100 | 13.34 | lt 13.08 (1.02)<br>cublas-fresh 13.08 (1.02)<br>cublas 13.08 (1.02)<br>lt-fresh 13.09 (1.02) | torch-eager 12.9 (1.03) | torch-compile 13.63 (0.98) | torch-compile-cg 13.63 (0.98) | **torch-compile-ro 12.89 (1.04)** | triton 13.73 (0.97) |
| pj_m2048_n4096_k14336 | rtx-4090 | 4.078 | **cublas 4.185 (0.97)**<br>lt 4.186 (0.97)<br>lt-fresh 4.187 (0.97)<br>cublas-fresh 4.188 (0.97) | torch-eager 4.296 (0.95) | torch-compile 4.193 (0.97) | torch-compile-cg 4.233 (0.96) | torch-compile-ro 4.261 (0.96) | triton 5.325 (0.77) |
| pj_m2048_n4096_k14336 | h100 | 8.435 | lt 7.093 (1.19)<br>cublas-fresh 7.108 (1.19)<br>cublas 7.109 (1.19)<br>lt-fresh 7.109 (1.19) | torch-eager 6.87 (1.23) | **torch-compile 6.474 (1.30)** | torch-compile-cg 6.5 (1.30) | torch-compile-ro 6.893 (1.22) | triton 9.025 (0.93) |
| pj_m2048_n4096_k4096 | rtx-5080 | 1.842 | cublas-fresh 1.783 (1.03)<br>lt-fresh 1.786 (1.03)<br>cublas 1.792 (1.03)<br>lt 1.792 (1.03) | torch-eager 1.78 (1.04) | torch-compile 1.788 (1.03) | **torch-compile-cg 1.778 (1.04)** | torch-compile-ro 1.782 (1.03) | triton 2.325 (0.79) |
| pj_m2048_n4096_k4096 | rtx-3090 | 2.875 | cublas-fresh 2.865 (1.00)<br>lt-fresh 2.865 (1.00)<br>cublas 2.866 (1.00)<br>lt 2.866 (1.00) | torch-eager 2.895 (0.99) | torch-compile 2.84 (1.01) | torch-compile-cg 2.833 (1.01) | **torch-compile-ro 2.829 (1.02)** | triton 3.573 (0.80) |
| pj_m2048_n4096_k4096 | a100 | 3.868 | cublas-fresh 3.708 (1.04)<br>lt 3.709 (1.04)<br>lt-fresh 3.709 (1.04)<br>cublas 3.711 (1.04) | torch-eager 3.823 (1.01) | torch-compile 4.381 (0.88) | torch-compile-cg 3.857 (1.00) | **torch-compile-ro 3.704 (1.04)** | triton 3.929 (0.98) |
| pj_m2048_n4096_k4096 | rtx-4090 | 1.192 | cublas 1.163 (1.02)<br>lt 1.165 (1.02)<br>lt-fresh 1.166 (1.02)<br>cublas-fresh 1.167 (1.02) | torch-eager 1.162 (1.03) | torch-compile 1.161 (1.03) | torch-compile-cg 1.159 (1.03) | **torch-compile-ro 1.158 (1.03)** | triton 1.502 (0.79) |
| pj_m2048_n4096_k4096 | h100 | 1.827 | lt-fresh 1.982 (0.92)<br>cublas-fresh 1.982 (0.92)<br>lt 1.984 (0.92)<br>cublas 1.984 (0.92) | torch-eager 2.167 (0.84) | **torch-compile 1.688 (1.08)** | torch-compile-cg 1.691 (1.08) | torch-compile-ro 1.747 (1.05) | triton 2.561 (0.71) |
| pj_m2048_n6144_k4096 | rtx-5080 | 2.651 | cublas-fresh 2.634 (1.01)<br>lt-fresh 2.637 (1.01)<br>lt 2.643 (1.00)<br>cublas 2.644 (1.00) | torch-eager 2.617 (1.01) | torch-compile 2.621 (1.01) | **torch-compile-cg 2.615 (1.01)** | torch-compile-ro 2.621 (1.01) | triton 3.418 (0.78) |
| pj_m2048_n6144_k4096 | rtx-3090 | 4.547 | lt 4.138 (1.10)<br>cublas-fresh 4.138 (1.10)<br>cublas 4.139 (1.10)<br>lt-fresh 4.139 (1.10) | torch-eager 4.138 (1.10) | torch-compile 4.14 (1.10) | torch-compile-cg 4.103 (1.11) | **torch-compile-ro 4.099 (1.11)** | triton 5.283 (0.86) |
| pj_m2048_n6144_k4096 | a100 | 5.804 | lt 5.808 (1.00)<br>lt-fresh 5.816 (1.00)<br>cublas-fresh 5.828 (1.00)<br>cublas 5.831 (1.00) | torch-eager 6.15 (0.94) | **torch-compile 5.785 (1.00)** | torch-compile-cg 5.787 (1.00) | torch-compile-ro 6.22 (0.93) | triton 5.871 (0.99) |
| pj_m2048_n6144_k4096 | rtx-4090 | 1.786 | lt-fresh 1.781 (1.00)<br>lt 1.781 (1.00)<br>cublas-fresh 1.782 (1.00)<br>cublas 1.785 (1.00) | torch-eager 1.762 (1.01) | **torch-compile 1.756 (1.02)** | torch-compile-cg 1.76 (1.01) | torch-compile-ro 1.764 (1.01) | triton 2.252 (0.79) |
| pj_m2048_n6144_k4096 | h100 | 2.749 | cublas-fresh 2.975 (0.92)<br>lt-fresh 2.975 (0.92)<br>cublas 2.976 (0.92)<br>lt 2.976 (0.92) | torch-eager 2.98 (0.92) | **torch-compile 2.878 (0.96)** | torch-compile-cg 2.886 (0.95) | torch-compile-ro 2.925 (0.94) | triton 3.926 (0.70) |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.6977 | **cublas-tf32-fresh 0.5898 (1.18)**<br>lt-tf32-fresh 0.5898 (1.18)<br>lt-tf32 0.6205 (1.12)<br>cublas-tf32 0.6226 (1.12) | torch-eager-tf32 0.6472 (1.08) | torch-compile-tf32 0.641 (1.09) | torch-compile-cg-tf32 0.639 (1.09) | torch-compile-ro-tf32 0.6472 (1.08) | triton-tf32 0.6246 (1.12) |
| pj_m128_n28672_k4096_tf32 | rtx-3090 | 0.9326 | lt-tf32 0.9148 (1.02)<br>lt-tf32-fresh 0.9151 (1.02)<br>cublas-tf32 0.9155 (1.02)<br>cublas-tf32-fresh 0.9155 (1.02) | **torch-eager-tf32 0.8929 (1.04)** | torch-compile-tf32 0.9462 (0.99) | torch-compile-cg-tf32 0.9472 (0.98) | torch-compile-ro-tf32 0.9462 (0.99) | triton-tf32 0.9882 (0.94) |
| pj_m128_n28672_k4096_tf32 | a100 | 0.5329 | lt-tf32 0.4516 (1.18)<br>lt-tf32-fresh 0.4516 (1.18)<br>cublas-tf32-fresh 0.4628 (1.15)<br>cublas-tf32 0.4639 (1.15) | torch-eager-tf32 0.4905 (1.09) | torch-compile-tf32 0.5038 (1.06) | torch-compile-cg-tf32 0.5253 (1.01) | torch-compile-ro-tf32 0.5253 (1.01) | **triton-tf32 0.4178 (1.28)** |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 0.5784 | **cublas-tf32-fresh 0.5693 (1.02)**<br>lt-tf32-fresh 0.5693 (1.02)<br>cublas-tf32 0.5765 (1.00)<br>lt-tf32 0.5775 (1.00) | torch-eager-tf32 0.6195 (0.93) | torch-compile-tf32 0.5816 (0.99) | torch-compile-cg-tf32 0.6257 (0.92) | torch-compile-ro-tf32 0.6216 (0.93) | triton-tf32 0.5888 (0.98) |
| pj_m128_n28672_k4096_tf32 | h100 | 0.3591 | **lt-tf32 0.2567 (1.40)**<br>cublas-tf32 0.2602 (1.38)<br>cublas-tf32-fresh 0.2613 (1.37)<br>lt-tf32-fresh 0.2634 (1.36) | torch-eager-tf32 0.274 (1.31) | torch-compile-tf32 0.2717 (1.32) | torch-compile-cg-tf32 0.2703 (1.33) | torch-compile-ro-tf32 0.27 (1.33) | triton-tf32 0.733 (0.49) |
| pj_m128_n4096_k14336_tf32 | rtx-5080 | 0.4157 | **cublas-tf32-fresh 0.3092 (1.34)**<br>lt-tf32-fresh 0.3092 (1.34)<br>cublas-tf32 0.3195 (1.30)<br>lt-tf32 0.3195 (1.30) | torch-eager-tf32 0.3133 (1.33) | torch-compile-tf32 0.3133 (1.33) | torch-compile-cg-tf32 0.3113 (1.34) | torch-compile-ro-tf32 0.3133 (1.33) | triton-tf32 0.3686 (1.13) |
| pj_m128_n4096_k14336_tf32 | rtx-3090 | 0.6505 | lt-tf32-fresh 0.8325 (0.78)<br>lt-tf32 0.8335 (0.78)<br>cublas-tf32-fresh 1.014 (0.64)<br>cublas-tf32 1.015 (0.64) | torch-eager-tf32 1.085 (0.60) | torch-compile-tf32 0.6144 (1.06) | torch-compile-cg-tf32 0.6134 (1.06) | torch-compile-ro-tf32 1.084 (0.60) | **triton-tf32 0.5714 (1.14)** |
| pj_m128_n4096_k14336_tf32 | a100 | 0.4685 | **cublas-tf32 0.2314 (2.02)**<br>cublas-tf32-fresh 0.2314 (2.02)<br>lt-tf32-fresh 0.2314 (2.02)<br>lt-tf32 0.2324 (2.02) | torch-eager-tf32 0.257 (1.82) | torch-compile-tf32 0.257 (1.82) | torch-compile-cg-tf32 0.256 (1.83) | torch-compile-ro-tf32 0.256 (1.83) | triton-tf32 0.2611 (1.79) |
| pj_m128_n4096_k14336_tf32 | rtx-4090 | 0.3755 | **lt-tf32-fresh 0.3072 (1.22)**<br>cublas-tf32-fresh 0.3113 (1.21)<br>lt-tf32 0.3228 (1.16)<br>cublas-tf32 0.3256 (1.15) | torch-eager-tf32 0.3277 (1.15) | torch-compile-tf32 0.3195 (1.18) | torch-compile-cg-tf32 0.3195 (1.18) | torch-compile-ro-tf32 0.3256 (1.15) | triton-tf32 0.3297 (1.14) |
| pj_m128_n4096_k14336_tf32 | h100 | 0.3411 | lt-tf32-fresh 0.1479 (2.31)<br>cublas-tf32-fresh 0.1481 (2.30)<br>lt-tf32 0.1537 (2.22)<br>cublas-tf32 0.1555 (2.19) | torch-eager-tf32 0.1475 (2.31) | **torch-compile-tf32 0.1458 (2.34)** | torch-compile-cg-tf32 0.1463 (2.33) | torch-compile-ro-tf32 0.1463 (2.33) | triton-tf32 0.3191 (1.07) |
| pj_m128_n4096_k4096_tf32 | rtx-5080 | 0.1387 | **cublas-tf32-fresh 0.084 (1.65)**<br>lt-tf32-fresh 0.084 (1.65)<br>lt-tf32 0.1085 (1.28)<br>cublas-tf32 0.1126 (1.23) | torch-eager-tf32 0.1106 (1.25) | torch-compile-tf32 0.1085 (1.28) | torch-compile-cg-tf32 0.1085 (1.28) | torch-compile-ro-tf32 0.1126 (1.23) | triton-tf32 0.1106 (1.25) |
| pj_m128_n4096_k4096_tf32 | rtx-3090 | 0.1851 | **cublas-tf32 0.154 (1.20)**<br>lt-tf32 0.154 (1.20)<br>lt-tf32-fresh 0.154 (1.20)<br>cublas-tf32-fresh 0.1546 (1.20) | torch-eager-tf32 0.1669 (1.11) | torch-compile-tf32 0.1669 (1.11) | torch-compile-cg-tf32 0.1669 (1.11) | torch-compile-ro-tf32 0.1669 (1.11) | triton-tf32 0.1608 (1.15) |
| pj_m128_n4096_k4096_tf32 | a100 | 0.149 | **cublas-tf32-fresh 0.0829 (1.80)**<br>lt-tf32-fresh 0.0829 (1.80)<br>cublas-tf32 0.085 (1.75)<br>lt-tf32 0.085 (1.75) | torch-eager-tf32 0.0952 (1.56) | torch-compile-tf32 0.0952 (1.56) | torch-compile-cg-tf32 0.0942 (1.58) | torch-compile-ro-tf32 0.0942 (1.58) | triton-tf32 0.086 (1.73) |
| pj_m128_n4096_k4096_tf32 | rtx-4090 | 0.09878 | **cublas-tf32-fresh 0.0644 (1.53)**<br>lt-tf32-fresh 0.0645 (1.53)<br>lt-tf32 0.1086 (0.91)<br>cublas-tf32 0.1096 (0.90) | torch-eager-tf32 0.1096 (0.90) | torch-compile-tf32 0.1085 (0.91) | torch-compile-cg-tf32 0.1075 (0.92) | torch-compile-ro-tf32 0.1075 (0.92) | triton-tf32 0.1096 (0.90) |
| pj_m128_n4096_k4096_tf32 | h100 | 0.09087 | **cublas-tf32-fresh 0.0433 (2.10)**<br>lt-tf32-fresh 0.0434 (2.09)<br>cublas-tf32 0.0451 (2.01)<br>lt-tf32 0.0461 (1.97) | torch-eager-tf32 0.0438 (2.07) | torch-compile-tf32 0.0435 (2.09) | torch-compile-cg-tf32 0.0436 (2.08) | torch-compile-ro-tf32 0.0437 (2.08) | triton-tf32 0.0898 (1.01) |
| pj_m128_n6144_k4096_tf32 | rtx-5080 | 0.1899 | **lt-tf32-fresh 0.129 (1.47)**<br>lt-tf32 0.1516 (1.25)<br>cublas-tf32-fresh 0.1782 (1.07)<br>cublas-tf32 0.2007 (0.95) | torch-eager-tf32 0.1905 (1.00) | torch-compile-tf32 0.172 (1.10) | torch-compile-cg-tf32 0.172 (1.10) | torch-compile-ro-tf32 0.1905 (1.00) | triton-tf32 0.2028 (0.94) |
| pj_m128_n6144_k4096_tf32 | rtx-3090 | 0.3684 | **lt-tf32-fresh 0.2437 (1.51)**<br>lt-tf32 0.2447 (1.51)<br>cublas-tf32 0.3072 (1.20)<br>cublas-tf32-fresh 0.3072 (1.20) | torch-eager-tf32 0.3236 (1.14) | torch-compile-tf32 0.2787 (1.32) | torch-compile-cg-tf32 0.2785 (1.32) | torch-compile-ro-tf32 0.3236 (1.14) | triton-tf32 0.2836 (1.30) |
| pj_m128_n6144_k4096_tf32 | a100 | 0.1618 | cublas-tf32 0.1034 (1.57)<br>lt-tf32 0.1034 (1.57)<br>cublas-tf32-fresh 0.1034 (1.57)<br>lt-tf32-fresh 0.1034 (1.57) | torch-eager-tf32 0.1085 (1.49) | torch-compile-tf32 0.1096 (1.48) | torch-compile-cg-tf32 0.1075 (1.51) | torch-compile-ro-tf32 0.1085 (1.49) | **triton-tf32 0.0993 (1.63)** |
| pj_m128_n6144_k4096_tf32 | rtx-4090 | 0.1837 | **cublas-tf32-fresh 0.1167 (1.57)**<br>lt-tf32-fresh 0.1167 (1.57)<br>cublas-tf32 0.1587 (1.16)<br>lt-tf32 0.1587 (1.16) | torch-eager-tf32 0.1587 (1.16) | torch-compile-tf32 0.1587 (1.16) | torch-compile-cg-tf32 0.1577 (1.16) | torch-compile-ro-tf32 0.1587 (1.16) | triton-tf32 0.1579 (1.16) |
| pj_m128_n6144_k4096_tf32 | h100 | 0.1563 | **lt-tf32-fresh 0.0522 (2.99)**<br>lt-tf32 0.0552 (2.83)<br>cublas-tf32-fresh 0.06 (2.60)<br>cublas-tf32 0.0687 (2.27) | torch-eager-tf32 0.0661 (2.36) | torch-compile-tf32 0.0669 (2.34) | torch-compile-cg-tf32 0.0655 (2.39) | torch-compile-ro-tf32 0.0657 (2.38) | triton-tf32 0.1263 (1.24) |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 9.042 | **cublas-tf32-fresh 8.059 (1.12)**<br>cublas-tf32 8.06 (1.12)<br>lt-tf32-fresh 8.06 (1.12)<br>lt-tf32 8.063 (1.12) | torch-eager-tf32 8.143 (1.11) | torch-compile-tf32 8.143 (1.11) | torch-compile-cg-tf32 8.144 (1.11) | torch-compile-ro-tf32 8.143 (1.11) | triton-tf32 8.503 (1.06) |
| pj_m2048_n28672_k4096_tf32 | rtx-3090 | 13.22 | lt-tf32 12.78 (1.04)<br>cublas-tf32 12.78 (1.03)<br>lt-tf32-fresh 12.8 (1.03)<br>cublas-tf32-fresh 12.81 (1.03) | **torch-eager-tf32 12.72 (1.04)** | torch-compile-tf32 12.72 (1.04) | torch-compile-cg-tf32 12.73 (1.04) | torch-compile-ro-tf32 12.73 (1.04) | triton-tf32 13.74 (0.96) |
| pj_m2048_n28672_k4096_tf32 | a100 | 5.909 | **lt-tf32-fresh 4.15 (1.42)**<br>lt-tf32 4.151 (1.42)<br>cublas-tf32-fresh 4.151 (1.42)<br>cublas-tf32 4.152 (1.42) | torch-eager-tf32 4.27 (1.38) | torch-compile-tf32 4.395 (1.34) | torch-compile-cg-tf32 4.229 (1.40) | torch-compile-ro-tf32 4.271 (1.38) | triton-tf32 5.024 (1.18) |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 8.344 | cublas-tf32-fresh 5.434 (1.54)<br>cublas-tf32 5.435 (1.54)<br>lt-tf32 5.501 (1.52)<br>lt-tf32-fresh 5.503 (1.52) | torch-eager-tf32 5.418 (1.54) | torch-compile-tf32 5.704 (1.46) | **torch-compile-cg-tf32 5.406 (1.54)** | torch-compile-ro-tf32 5.406 (1.54) | triton-tf32 5.884 (1.42) |
| pj_m2048_n28672_k4096_tf32 | h100 | 3.928 | **lt-tf32 2.58 (1.52)**<br>lt-tf32-fresh 2.611 (1.50)<br>cublas-tf32-fresh 3.049 (1.29)<br>cublas-tf32 3.074 (1.28) | torch-eager-tf32 3.307 (1.19) | torch-compile-tf32 3.35 (1.17) | torch-compile-cg-tf32 3.349 (1.17) | torch-compile-ro-tf32 3.275 (1.20) | triton-tf32 8.131 (0.48) |
| pj_m2048_n4096_k14336_tf32 | rtx-5080 | 4.356 | **lt-tf32-fresh 4.241 (1.03)**<br>cublas-tf32-fresh 4.244 (1.03)<br>cublas-tf32 4.252 (1.02)<br>lt-tf32 4.253 (1.02) | torch-eager-tf32 4.539 (0.96) | torch-compile-tf32 4.308 (1.01) | torch-compile-cg-tf32 4.306 (1.01) | torch-compile-ro-tf32 4.537 (0.96) | triton-tf32 4.638 (0.94) |
| pj_m2048_n4096_k14336_tf32 | rtx-3090 | 7.285 | cublas-tf32-fresh 6.952 (1.05)<br>cublas-tf32 6.953 (1.05)<br>lt-tf32 6.954 (1.05)<br>lt-tf32-fresh 6.954 (1.05) | torch-eager-tf32 6.909 (1.05) | torch-compile-tf32 7.327 (0.99) | torch-compile-cg-tf32 7.362 (0.99) | **torch-compile-ro-tf32 6.896 (1.06)** | triton-tf32 7.313 (1.00) |
| pj_m2048_n4096_k14336_tf32 | a100 | 2.71 | cublas-tf32 2.146 (1.26)<br>lt-tf32 2.146 (1.26)<br>lt-tf32-fresh 2.146 (1.26)<br>cublas-tf32-fresh 2.147 (1.26) | torch-eager-tf32 2.149 (1.26) | torch-compile-tf32 2.299 (1.18) | torch-compile-cg-tf32 2.298 (1.18) | **torch-compile-ro-tf32 2.071 (1.31)** | triton-tf32 2.511 (1.08) |
| pj_m2048_n4096_k14336_tf32 | rtx-4090 | 3.178 | lt-tf32-fresh 2.73 (1.16)<br>lt-tf32 2.731 (1.16)<br>cublas-tf32-fresh 2.734 (1.16)<br>cublas-tf32 2.735 (1.16) | torch-eager-tf32 2.724 (1.17) | torch-compile-tf32 2.864 (1.11) | **torch-compile-cg-tf32 2.718 (1.17)** | torch-compile-ro-tf32 2.72 (1.17) | triton-tf32 2.914 (1.09) |
| pj_m2048_n4096_k14336_tf32 | h100 | 1.643 | lt-tf32 1.29 (1.27)<br>cublas-tf32 1.29 (1.27)<br>lt-tf32-fresh 1.316 (1.25)<br>cublas-tf32-fresh 1.316 (1.25) | **torch-eager-tf32 1.057 (1.55)** | torch-compile-tf32 1.224 (1.34) | torch-compile-cg-tf32 1.234 (1.33) | torch-compile-ro-tf32 1.234 (1.33) | triton-tf32 3.824 (0.43) |
| pj_m2048_n4096_k4096_tf32 | rtx-5080 | 1.341 | **lt-tf32-fresh 1.161 (1.15)**<br>lt-tf32 1.163 (1.15)<br>cublas-tf32-fresh 1.163 (1.15)<br>cublas-tf32 1.173 (1.14) | torch-eager-tf32 1.313 (1.02) | torch-compile-tf32 1.225 (1.10) | torch-compile-cg-tf32 1.225 (1.10) | torch-compile-ro-tf32 1.302 (1.03) | triton-tf32 1.345 (1.00) |
| pj_m2048_n4096_k4096_tf32 | rtx-3090 | 2.095 | cublas-tf32 2.235 (0.94)<br>cublas-tf32-fresh 2.235 (0.94)<br>lt-tf32-fresh 2.235 (0.94)<br>lt-tf32 2.236 (0.94) | torch-eager-tf32 2.484 (0.84) | **torch-compile-tf32 2.035 (1.03)** | torch-compile-cg-tf32 2.051 (1.02) | torch-compile-ro-tf32 2.2 (0.95) | triton-tf32 2.156 (0.97) |
| pj_m2048_n4096_k4096_tf32 | a100 | 0.8024 | **cublas-tf32 0.5806 (1.38)**<br>lt-tf32 0.5857 (1.37)<br>lt-tf32-fresh 0.5857 (1.37)<br>cublas-tf32-fresh 0.6011 (1.33) | torch-eager-tf32 0.6922 (1.16) | torch-compile-tf32 0.6912 (1.16) | torch-compile-cg-tf32 0.6902 (1.16) | torch-compile-ro-tf32 0.6902 (1.16) | triton-tf32 0.7004 (1.15) |
| pj_m2048_n4096_k4096_tf32 | rtx-4090 | 0.9283 | **lt-tf32 0.8243 (1.13)**<br>lt-tf32-fresh 0.8245 (1.13)<br>cublas-tf32 0.8294 (1.12)<br>cublas-tf32-fresh 0.8294 (1.12) | torch-eager-tf32 0.8316 (1.12) | torch-compile-tf32 0.8274 (1.12) | torch-compile-cg-tf32 0.8264 (1.12) | torch-compile-ro-tf32 0.8315 (1.12) | triton-tf32 0.8489 (1.09) |
| pj_m2048_n4096_k4096_tf32 | h100 | 0.4583 | **lt-tf32-fresh 0.3284 (1.40)**<br>cublas-tf32-fresh 0.3285 (1.40)<br>lt-tf32 0.3428 (1.34)<br>cublas-tf32 0.3468 (1.32) | torch-eager-tf32 0.3502 (1.31) | torch-compile-tf32 0.348 (1.32) | torch-compile-cg-tf32 0.3483 (1.32) | torch-compile-ro-tf32 0.3475 (1.32) | triton-tf32 1.192 (0.38) |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.022 | **cublas-tf32-fresh 1.749 (1.16)**<br>lt-tf32-fresh 1.749 (1.16)<br>lt-tf32 1.757 (1.15)<br>cublas-tf32 1.759 (1.15) | torch-eager-tf32 1.862 (1.09) | torch-compile-tf32 1.798 (1.12) | torch-compile-cg-tf32 1.798 (1.12) | torch-compile-ro-tf32 1.861 (1.09) | triton-tf32 1.974 (1.02) |
| pj_m2048_n6144_k4096_tf32 | rtx-3090 | 2.986 | cublas-tf32 2.869 (1.04)<br>lt-tf32 2.87 (1.04)<br>cublas-tf32-fresh 2.87 (1.04)<br>lt-tf32-fresh 2.87 (1.04) | torch-eager-tf32 2.85 (1.05) | torch-compile-tf32 2.845 (1.05) | **torch-compile-cg-tf32 2.821 (1.06)** | torch-compile-ro-tf32 2.848 (1.05) | triton-tf32 3.151 (0.95) |
| pj_m2048_n6144_k4096_tf32 | a100 | 1.248 | cublas-tf32 0.9257 (1.35)<br>lt-tf32 0.9257 (1.35)<br>cublas-tf32-fresh 0.9257 (1.35)<br>lt-tf32-fresh 0.9257 (1.35) | **torch-eager-tf32 0.809 (1.54)** | torch-compile-tf32 0.8151 (1.53) | torch-compile-cg-tf32 0.8724 (1.43) | torch-compile-ro-tf32 0.8253 (1.51) | triton-tf32 1.131 (1.10) |
| pj_m2048_n6144_k4096_tf32 | rtx-4090 | 1.378 | cublas-tf32 1.234 (1.12)<br>lt-tf32 1.234 (1.12)<br>cublas-tf32-fresh 1.236 (1.11)<br>lt-tf32-fresh 1.236 (1.11) | torch-eager-tf32 1.168 (1.18) | torch-compile-tf32 1.177 (1.17) | torch-compile-cg-tf32 1.178 (1.17) | **torch-compile-ro-tf32 1.166 (1.18)** | triton-tf32 1.272 (1.08) |
| pj_m2048_n6144_k4096_tf32 | h100 | 0.7186 | **lt-tf32 0.5644 (1.27)**<br>lt-tf32-fresh 0.5853 (1.23)<br>cublas-tf32 0.6824 (1.05)<br>cublas-tf32-fresh 0.7009 (1.03) | torch-eager-tf32 0.6483 (1.11) | torch-compile-tf32 0.6648 (1.08) | torch-compile-cg-tf32 0.6579 (1.09) | torch-compile-ro-tf32 0.6528 (1.10) | triton-tf32 1.713 (0.42) |
| pj_m128_n28672_k4096_bf16 | rtx-5080 | 0.3844 |  | torch-bf16 0.3256 (1.18) | torch-bf16-compile 0.3215 (1.20) | torch-bf16-compile-cg 0.3215 (1.20) | torch-bf16-compile-ro 0.3318 (1.16) | **triton-bf16 0.3174 (1.21)** |
| pj_m128_n28672_k4096_bf16 | rtx-3090 | 0.4946 |  | torch-bf16 0.5089 (0.97) | torch-bf16-compile 0.4751 (1.04) | torch-bf16-compile-cg 0.4741 (1.04) | torch-bf16-compile-ro 0.4741 (1.04) | **triton-bf16 0.4618 (1.07)** |
| pj_m128_n28672_k4096_bf16 | a100 | 0.3812 |  | torch-bf16 0.2959 (1.29) | torch-bf16-compile 0.2908 (1.31) | torch-bf16-compile-cg 0.2918 (1.31) | torch-bf16-compile-ro 0.2918 (1.31) | **triton-bf16 0.2181 (1.75)** |
| pj_m128_n28672_k4096_bf16 | rtx-4090 | 0.3384 |  | torch-bf16 0.3411 (0.99) | **torch-bf16-compile 0.3154 (1.07)** | torch-bf16-compile-cg 0.3379 (1.00) | torch-bf16-compile-ro 0.3615 (0.94) | triton-bf16 0.3276 (1.03) |
| pj_m128_n28672_k4096_bf16 | h100 | 0.2607 |  | torch-bf16 0.1308 (1.99) | torch-bf16-compile 0.134 (1.95) | torch-bf16-compile-cg 0.1342 (1.94) | **torch-bf16-compile-ro 0.13 (2.01)** | triton-bf16 0.1516 (1.72) |
| pj_m128_n4096_k14336_bf16 | rtx-5080 | 0.2487 |  | torch-bf16 0.172 (1.45) | torch-bf16-compile 0.172 (1.45) | **torch-bf16-compile-cg 0.17 (1.46)** | torch-bf16-compile-ro 0.172 (1.45) | triton-bf16 0.1843 (1.35) |
| pj_m128_n4096_k14336_bf16 | rtx-3090 | 0.3786 |  | **torch-bf16 0.2714 (1.40)** | torch-bf16-compile 0.2755 (1.37) | torch-bf16-compile-cg 0.2755 (1.37) | torch-bf16-compile-ro 0.2744 (1.38) | triton-bf16 0.2755 (1.37) |
| pj_m128_n4096_k14336_bf16 | a100 | 0.3353 |  | torch-bf16 0.127 (2.64) | torch-bf16-compile 0.126 (2.66) | **torch-bf16-compile-cg 0.1239 (2.71)** | torch-bf16-compile-ro 0.1239 (2.71) | triton-bf16 0.1444 (2.32) |
| pj_m128_n4096_k14336_bf16 | rtx-4090 | 0.2342 |  | torch-bf16 0.1935 (1.21) | torch-bf16-compile 0.1894 (1.24) | **torch-bf16-compile-cg 0.1884 (1.24)** | torch-bf16-compile-ro 0.1935 (1.21) | triton-bf16 0.1884 (1.24) |
| pj_m128_n4096_k14336_bf16 | h100 | 0.2853 |  | torch-bf16 0.0763 (3.74) | torch-bf16-compile 0.0771 (3.70) | torch-bf16-compile-cg 0.078 (3.66) | **torch-bf16-compile-ro 0.0757 (3.77)** | triton-bf16 0.106 (2.69) |
| pj_m128_n4096_k4096_bf16 | rtx-5080 | 0.07729 |  | torch-bf16 0.0553 (1.40) | torch-bf16-compile 0.0553 (1.40) | torch-bf16-compile-cg 0.0553 (1.40) | **torch-bf16-compile-ro 0.0532 (1.45)** | triton-bf16 0.0573 (1.35) |
| pj_m128_n4096_k4096_bf16 | rtx-3090 | 0.1083 |  | torch-bf16 0.084 (1.29) | torch-bf16-compile 0.084 (1.29) | torch-bf16-compile-cg 0.0831 (1.30) | torch-bf16-compile-ro 0.084 (1.29) | **triton-bf16 0.0799 (1.35)** |
| pj_m128_n4096_k4096_bf16 | a100 | 0.1098 |  | torch-bf16 0.0584 (1.88) | torch-bf16-compile 0.0573 (1.92) | torch-bf16-compile-cg 0.0563 (1.95) | torch-bf16-compile-ro 0.0563 (1.95) | **triton-bf16 0.0532 (2.06)** |
| pj_m128_n4096_k4096_bf16 | rtx-4090 | 0.06718 |  | torch-bf16 0.0614 (1.09) | torch-bf16-compile 0.0614 (1.09) | torch-bf16-compile-cg 0.0594 (1.13) | torch-bf16-compile-ro 0.0594 (1.13) | **triton-bf16 0.0593 (1.13)** |
| pj_m128_n4096_k4096_bf16 | h100 | 0.06663 |  | **torch-bf16 0.022 (3.03)** | torch-bf16-compile 0.0226 (2.95) | torch-bf16-compile-cg 0.0223 (2.99) | torch-bf16-compile-ro 0.0224 (2.97) | triton-bf16 0.0348 (1.91) |
| pj_m128_n6144_k4096_bf16 | rtx-5080 | 0.1073 |  | **torch-bf16 0.0881 (1.22)** | torch-bf16-compile 0.0881 (1.22) | torch-bf16-compile-cg 0.0881 (1.22) | torch-bf16-compile-ro 0.0881 (1.22) | triton-bf16 0.1024 (1.05) |
| pj_m128_n6144_k4096_bf16 | rtx-3090 | 0.169 |  | torch-bf16 0.126 (1.34) | torch-bf16-compile 0.126 (1.34) | **torch-bf16-compile-cg 0.1249 (1.35)** | torch-bf16-compile-ro 0.1253 (1.35) | triton-bf16 0.1318 (1.28) |
| pj_m128_n6144_k4096_bf16 | a100 | 0.1159 |  | torch-bf16 0.0737 (1.57) | torch-bf16-compile 0.0676 (1.72) | torch-bf16-compile-cg 0.0666 (1.74) | torch-bf16-compile-ro 0.0727 (1.59) | **triton-bf16 0.0625 (1.86)** |
| pj_m128_n6144_k4096_bf16 | rtx-4090 | 0.09603 |  | torch-bf16 0.0922 (1.04) | **torch-bf16-compile 0.0819 (1.17)** | torch-bf16-compile-cg 0.0819 (1.17) | torch-bf16-compile-ro 0.0912 (1.05) | triton-bf16 0.0819 (1.17) |
| pj_m128_n6144_k4096_bf16 | h100 | 0.1056 |  | **torch-bf16 0.0305 (3.46)** | torch-bf16-compile 0.0305 (3.46) | torch-bf16-compile-cg 0.0307 (3.44) | torch-bf16-compile-ro 0.0309 (3.42) | triton-bf16 0.0409 (2.58) |
| pj_m16_n28672_k4096_bf16 | rtx-5080 | 0.2966 |  | torch-bf16 0.2806 (1.06) | **torch-bf16-compile 0.2744 (1.08)** | torch-bf16-compile-cg 0.2744 (1.08) | torch-bf16-compile-ro 0.2826 (1.05) | triton-bf16 0.2806 (1.06) |
| pj_m16_n28672_k4096_bf16 | rtx-3090 | 0.3007 |  | torch-bf16 0.3164 (0.95) | **torch-bf16-compile 0.2724 (1.10)** | torch-bf16-compile-cg 0.2939 (1.02) | torch-bf16-compile-ro 0.3297 (0.91) | triton-bf16 0.2826 (1.06) |
| pj_m16_n28672_k4096_bf16 | a100 | 0.2059 |  | torch-bf16 0.2232 (0.92) | **torch-bf16-compile 0.1925 (1.07)** | torch-bf16-compile-cg 0.2068 (1.00) | torch-bf16-compile-ro 0.2191 (0.94) | triton-bf16 0.1997 (1.03) |
| pj_m16_n28672_k4096_bf16 | rtx-4090 | 0.3237 |  | torch-bf16 0.3144 (1.03) | **torch-bf16-compile 0.3082 (1.05)** | torch-bf16-compile-cg 0.3215 (1.01) | torch-bf16-compile-ro 0.3328 (0.97) | triton-bf16 0.3113 (1.04) |
| pj_m16_n28672_k4096_bf16 | h100 | 0.2044 |  | torch-bf16 0.097 (2.11) | torch-bf16-compile 0.0918 (2.23) | **torch-bf16-compile-cg 0.0902 (2.27)** | torch-bf16-compile-ro 0.093 (2.20) | triton-bf16 0.1307 (1.56) |
| pj_m16_n4096_k14336_bf16 | rtx-5080 | 0.1305 |  | torch-bf16 0.1823 (0.72) | **torch-bf16-compile 0.1393 (0.94)** | torch-bf16-compile-cg 0.1393 (0.94) | torch-bf16-compile-ro 0.1823 (0.72) | triton-bf16 0.1843 (0.71) |
| pj_m16_n4096_k14336_bf16 | rtx-3090 | 0.1658 |  | torch-bf16 0.1468 (1.13) | **torch-bf16-compile 0.1423 (1.17)** | torch-bf16-compile-cg 0.1423 (1.17) | torch-bf16-compile-ro 0.1464 (1.13) | triton-bf16 0.1526 (1.09) |
| pj_m16_n4096_k14336_bf16 | a100 | 0.3217 |  | torch-bf16 0.1167 (2.76) | **torch-bf16-compile 0.1096 (2.94)** | torch-bf16-compile-cg 0.1096 (2.94) | torch-bf16-compile-ro 0.1147 (2.80) | triton-bf16 0.129 (2.49) |
| pj_m16_n4096_k14336_bf16 | rtx-4090 | 0.1609 |  | torch-bf16 0.182 (0.88) | **torch-bf16-compile 0.1772 (0.91)** | torch-bf16-compile-cg 0.1772 (0.91) | torch-bf16-compile-ro 0.1802 (0.89) | triton-bf16 0.1864 (0.86) |
| pj_m16_n4096_k14336_bf16 | h100 | 0.1181 |  | torch-bf16 0.0539 (2.19) | **torch-bf16-compile 0.0538 (2.20)** | torch-bf16-compile-cg 0.0543 (2.18) | torch-bf16-compile-ro 0.0545 (2.17) | triton-bf16 0.0943 (1.25) |
| pj_m16_n4096_k4096_bf16 | rtx-5080 | 0.05285 |  | torch-bf16 0.0573 (0.92) | **torch-bf16-compile 0.043 (1.23)** | torch-bf16-compile-cg 0.043 (1.23) | torch-bf16-compile-ro 0.0553 (0.96) | triton-bf16 0.0614 (0.86) |
| pj_m16_n4096_k4096_bf16 | rtx-3090 | 0.07414 |  | torch-bf16 0.0502 (1.48) | **torch-bf16-compile 0.0471 (1.57)** | torch-bf16-compile-cg 0.0471 (1.57) | torch-bf16-compile-ro 0.0502 (1.48) | triton-bf16 0.049 (1.51) |
| pj_m16_n4096_k4096_bf16 | a100 | 0.1078 |  | torch-bf16 0.0481 (2.24) | torch-bf16-compile 0.044 (2.45) | **torch-bf16-compile-cg 0.043 (2.51)** | torch-bf16-compile-ro 0.0471 (2.29) | triton-bf16 0.0512 (2.10) |
| pj_m16_n4096_k4096_bf16 | rtx-4090 | 0.06311 |  | torch-bf16 0.0635 (0.99) | torch-bf16-compile 0.0573 (1.10) | **torch-bf16-compile-cg 0.0563 (1.12)** | torch-bf16-compile-ro 0.0635 (0.99) | triton-bf16 0.0614 (1.03) |
| pj_m16_n4096_k4096_bf16 | h100 | 0.05409 |  | torch-bf16 0.0223 (2.43) | **torch-bf16-compile 0.0216 (2.50)** | torch-bf16-compile-cg 0.0223 (2.43) | torch-bf16-compile-ro 0.0223 (2.43) | triton-bf16 0.0346 (1.56) |
| pj_m16_n6144_k4096_bf16 | rtx-5080 | 0.05933 |  | torch-bf16 0.0655 (0.91) | **torch-bf16-compile 0.0635 (0.93)** | torch-bf16-compile-cg 0.0635 (0.93) | torch-bf16-compile-ro 0.0655 (0.91) | triton-bf16 0.0676 (0.88) |
| pj_m16_n6144_k4096_bf16 | rtx-3090 | 0.08774 |  | torch-bf16 0.086 (1.02) | torch-bf16-compile 0.0655 (1.34) | **torch-bf16-compile-cg 0.0649 (1.35)** | torch-bf16-compile-ro 0.086 (1.02) | triton-bf16 0.0666 (1.32) |
| pj_m16_n6144_k4096_bf16 | a100 | 0.1128 |  | torch-bf16 0.0686 (1.64) | torch-bf16-compile 0.0604 (1.87) | **torch-bf16-compile-cg 0.0594 (1.90)** | torch-bf16-compile-ro 0.0666 (1.69) | triton-bf16 0.0635 (1.78) |
| pj_m16_n6144_k4096_bf16 | rtx-4090 | 0.0772 |  | torch-bf16 0.0819 (0.94) | torch-bf16-compile 0.081 (0.95) | torch-bf16-compile-cg 0.0819 (0.94) | **torch-bf16-compile-ro 0.0809 (0.95)** | triton-bf16 0.0819 (0.94) |
| pj_m16_n6144_k4096_bf16 | h100 | 0.06172 |  | torch-bf16 0.028 (2.20) | **torch-bf16-compile 0.0276 (2.24)** | torch-bf16-compile-cg 0.0277 (2.23) | torch-bf16-compile-ro 0.0282 (2.19) | triton-bf16 0.0395 (1.56) |
| pj_m1_n28672_k4096_bf16 | rtx-5080 | 0.2586 |  | torch-bf16 0.2847 (0.91) | **torch-bf16-compile 0.2744 (0.94)** | torch-bf16-compile-cg 0.2744 (0.94) | torch-bf16-compile-ro 0.2847 (0.91) | triton-bf16 0.3052 (0.85) |
| pj_m1_n28672_k4096_bf16 | rtx-3090 | 0.2808 |  | torch-bf16 0.299 (0.94) | **torch-bf16-compile 0.2785 (1.01)** | torch-bf16-compile-cg 0.2785 (1.01) | torch-bf16-compile-ro 0.2989 (0.94) | triton-bf16 0.3891 (0.72) |
| pj_m1_n28672_k4096_bf16 | a100 | 0.2346 |  | torch-bf16 0.2099 (1.12) | **torch-bf16-compile 0.1833 (1.28)** | torch-bf16-compile-cg 0.1843 (1.27) | torch-bf16-compile-ro 0.2089 (1.12) | triton-bf16 0.2048 (1.15) |
| pj_m1_n28672_k4096_bf16 | rtx-4090 | 0.3141 |  | torch-bf16 0.2857 (1.10) | **torch-bf16-compile 0.2724 (1.15)** | torch-bf16-compile-cg 0.2724 (1.15) | torch-bf16-compile-ro 0.3052 (1.03) | triton-bf16 0.3185 (0.99) |
| pj_m1_n28672_k4096_bf16 | h100 | 0.1197 |  | **torch-bf16 0.0914 (1.31)** | torch-bf16-compile 0.0961 (1.25) | torch-bf16-compile-cg 0.1 (1.20) | torch-bf16-compile-ro 0.092 (1.30) | triton-bf16 0.1269 (0.94) |
| pj_m1_n4096_k14336_bf16 | rtx-5080 | 0.1099 |  | **torch-bf16 0.1413 (0.78)** | torch-bf16-compile 0.1472 (0.75) | torch-bf16-compile-cg 0.1454 (0.76) | torch-bf16-compile-ro 0.1413 (0.78) | triton-bf16 0.2007 (0.55) |
| pj_m1_n4096_k14336_bf16 | rtx-3090 | 0.1535 |  | torch-bf16 0.1476 (1.04) | torch-bf16-compile 0.1487 (1.03) | **torch-bf16-compile-cg 0.1475 (1.04)** | torch-bf16-compile-ro 0.1475 (1.04) | triton-bf16 0.2263 (0.68) |
| pj_m1_n4096_k14336_bf16 | a100 | 0.1373 |  | torch-bf16 0.1126 (1.22) | **torch-bf16-compile 0.1075 (1.28)** | torch-bf16-compile-cg 0.1075 (1.28) | torch-bf16-compile-ro 0.1157 (1.19) | triton-bf16 0.1516 (0.91) |
| pj_m1_n4096_k14336_bf16 | rtx-4090 | 0.1214 |  | torch-bf16 0.1812 (0.67) | torch-bf16-compile 0.1525 (0.80) | **torch-bf16-compile-cg 0.1505 (0.81)** | torch-bf16-compile-ro 0.1802 (0.67) | triton-bf16 0.1946 (0.62) |
| pj_m1_n4096_k14336_bf16 | h100 | 0.07918 |  | torch-bf16 0.0535 (1.48) | torch-bf16-compile 0.053 (1.49) | **torch-bf16-compile-cg 0.0524 (1.51)** | torch-bf16-compile-ro 0.0537 (1.47) | triton-bf16 0.1116 (0.71) |
| pj_m1_n4096_k4096_bf16 | rtx-5080 | 0.03644 |  | torch-bf16 0.0696 (0.52) | torch-bf16-compile 0.0492 (0.74) | **torch-bf16-compile-cg 0.0471 (0.77)** | torch-bf16-compile-ro 0.0696 (0.52) | triton-bf16 0.0717 (0.51) |
| pj_m1_n4096_k4096_bf16 | rtx-3090 | 0.05816 |  | torch-bf16 0.0502 (1.16) | torch-bf16-compile 0.0504 (1.15) | **torch-bf16-compile-cg 0.0492 (1.18)** | torch-bf16-compile-ro 0.0492 (1.18) | triton-bf16 0.0922 (0.63) |
| pj_m1_n4096_k4096_bf16 | a100 | 0.04823 |  | torch-bf16 0.0461 (1.05) | torch-bf16-compile 0.0461 (1.05) | **torch-bf16-compile-cg 0.0451 (1.07)** | torch-bf16-compile-ro 0.0451 (1.07) | triton-bf16 0.0594 (0.81) |
| pj_m1_n4096_k4096_bf16 | rtx-4090 | 0.03053 |  | torch-bf16 0.0625 (0.49) | torch-bf16-compile 0.0563 (0.54) | **torch-bf16-compile-cg 0.0553 (0.55)** | torch-bf16-compile-ro 0.0635 (0.48) | triton-bf16 0.0676 (0.45) |
| pj_m1_n4096_k4096_bf16 | h100 | 0.03195 |  | torch-bf16 0.0203 (1.57) | torch-bf16-compile 0.0219 (1.46) | torch-bf16-compile-cg 0.0208 (1.54) | **torch-bf16-compile-ro 0.0202 (1.58)** | triton-bf16 0.0399 (0.80) |
| pj_m1_n6144_k4096_bf16 | rtx-5080 | 0.03456 |  | torch-bf16 0.0799 (0.43) | **torch-bf16-compile 0.0635 (0.54)** | torch-bf16-compile-cg 0.0635 (0.54) | torch-bf16-compile-ro 0.0778 (0.44) | triton-bf16 0.084 (0.41) |
| pj_m1_n6144_k4096_bf16 | rtx-3090 | 0.07446 |  | torch-bf16 0.0748 (1.00) | **torch-bf16-compile 0.0676 (1.10)** | torch-bf16-compile-cg 0.0676 (1.10) | torch-bf16-compile-ro 0.084 (0.89) | triton-bf16 0.1311 (0.57) |
| pj_m1_n6144_k4096_bf16 | a100 | 0.07335 |  | torch-bf16 0.0645 (1.14) | torch-bf16-compile 0.0553 (1.33) | **torch-bf16-compile-cg 0.0532 (1.38)** | torch-bf16-compile-ro 0.0635 (1.16) | triton-bf16 0.0748 (0.98) |
| pj_m1_n6144_k4096_bf16 | rtx-4090 | 0.05672 |  | torch-bf16 0.0799 (0.71) | torch-bf16-compile 0.0809 (0.70) | **torch-bf16-compile-cg 0.0788 (0.72)** | torch-bf16-compile-ro 0.0799 (0.71) | triton-bf16 0.087 (0.65) |
| pj_m1_n6144_k4096_bf16 | h100 | 0.0484 |  | **torch-bf16 0.03 (1.61)** | torch-bf16-compile 0.031 (1.56) | torch-bf16-compile-cg 0.03 (1.61) | torch-bf16-compile-ro 0.0303 (1.60) | triton-bf16 0.0467 (1.04) |
| pj_m2048_n28672_k4096_bf16 | rtx-5080 | 4.665 |  | torch-bf16 4.11 (1.14) | torch-bf16-compile 3.964 (1.18) | **torch-bf16-compile-cg 3.963 (1.18)** | torch-bf16-compile-ro 4.068 (1.15) | triton-bf16 4.143 (1.13) |
| pj_m2048_n28672_k4096_bf16 | rtx-3090 | 6.801 |  | torch-bf16 6.341 (1.07) | torch-bf16-compile 6.824 (1.00) | **torch-bf16-compile-cg 6.283 (1.08)** | torch-bf16-compile-ro 6.331 (1.07) | triton-bf16 6.523 (1.04) |
| pj_m2048_n28672_k4096_bf16 | a100 | 3.509 |  | torch-bf16 2.081 (1.69) | torch-bf16-compile 2.226 (1.58) | **torch-bf16-compile-cg 1.954 (1.80)** | torch-bf16-compile-ro 1.975 (1.78) | triton-bf16 2.29 (1.53) |
| pj_m2048_n28672_k4096_bf16 | rtx-4090 | 4.343 |  | torch-bf16 2.774 (1.57) | torch-bf16-compile 2.859 (1.52) | torch-bf16-compile-cg 2.774 (1.57) | **torch-bf16-compile-ro 2.748 (1.58)** | triton-bf16 2.788 (1.56) |
| pj_m2048_n28672_k4096_bf16 | h100 | 3.967 |  | **torch-bf16 1.267 (3.13)** | torch-bf16-compile 1.578 (2.51) | torch-bf16-compile-cg 1.377 (2.88) | torch-bf16-compile-ro 1.34 (2.96) | triton-bf16 1.91 (2.08) |
| pj_m2048_n4096_k14336_bf16 | rtx-5080 | 2.283 |  | torch-bf16 2.118 (1.08) | torch-bf16-compile 2.101 (1.09) | **torch-bf16-compile-cg 2.097 (1.09)** | torch-bf16-compile-ro 2.097 (1.09) | triton-bf16 2.169 (1.05) |
| pj_m2048_n4096_k14336_bf16 | rtx-3090 | 3.976 |  | torch-bf16 3.713 (1.07) | torch-bf16-compile 3.743 (1.06) | torch-bf16-compile-cg 3.723 (1.07) | torch-bf16-compile-ro 3.724 (1.07) | **triton-bf16 3.556 (1.12)** |
| pj_m2048_n4096_k14336_bf16 | a100 | 1.966 |  | torch-bf16 0.9103 (2.16) | torch-bf16-compile 1.147 (1.71) | torch-bf16-compile-cg 1.146 (1.72) | **torch-bf16-compile-ro 0.9052 (2.17)** | triton-bf16 1.207 (1.63) |
| pj_m2048_n4096_k14336_bf16 | rtx-4090 | 1.676 |  | torch-bf16 1.588 (1.06) | torch-bf16-compile 1.434 (1.17) | torch-bf16-compile-cg 1.434 (1.17) | torch-bf16-compile-ro 1.556 (1.08) | **triton-bf16 1.377 (1.22)** |
| pj_m2048_n4096_k14336_bf16 | h100 | 1.671 |  | **torch-bf16 0.6216 (2.69)** | torch-bf16-compile 0.6729 (2.48) | torch-bf16-compile-cg 0.6699 (2.50) | torch-bf16-compile-ro 0.6754 (2.47) | triton-bf16 0.9609 (1.74) |
| pj_m2048_n4096_k4096_bf16 | rtx-5080 | 0.7108 |  | torch-bf16 0.6062 (1.17) | torch-bf16-compile 0.6021 (1.18) | **torch-bf16-compile-cg 0.598 (1.19)** | torch-bf16-compile-ro 0.6001 (1.18) | triton-bf16 0.6205 (1.15) |
| pj_m2048_n4096_k4096_bf16 | rtx-3090 | 1.113 |  | torch-bf16 1.098 (1.01) | torch-bf16-compile 1.079 (1.03) | torch-bf16-compile-cg 1.078 (1.03) | torch-bf16-compile-ro 1.078 (1.03) | **triton-bf16 1.023 (1.09)** |
| pj_m2048_n4096_k4096_bf16 | a100 | 0.5995 |  | **torch-bf16 0.3092 (1.94)** | torch-bf16-compile 0.3994 (1.50) | torch-bf16-compile-cg 0.3983 (1.51) | torch-bf16-compile-ro 0.3973 (1.51) | triton-bf16 0.3287 (1.82) |
| pj_m2048_n4096_k4096_bf16 | rtx-4090 | 0.5076 |  | torch-bf16 0.4628 (1.10) | torch-bf16-compile 0.4148 (1.22) | torch-bf16-compile-cg 0.4147 (1.22) | torch-bf16-compile-ro 0.4588 (1.11) | **triton-bf16 0.4004 (1.27)** |
| pj_m2048_n4096_k4096_bf16 | h100 | 0.4782 |  | **torch-bf16 0.1681 (2.85)** | torch-bf16-compile 0.1733 (2.76) | torch-bf16-compile-cg 0.173 (2.76) | torch-bf16-compile-ro 0.1731 (2.76) | triton-bf16 0.2583 (1.85) |
| pj_m2048_n6144_k4096_bf16 | rtx-5080 | 1.08 |  | torch-bf16 0.8868 (1.22) | torch-bf16-compile 0.8888 (1.22) | torch-bf16-compile-cg 0.8885 (1.22) | **torch-bf16-compile-ro 0.8865 (1.22)** | triton-bf16 0.9073 (1.19) |
| pj_m2048_n6144_k4096_bf16 | rtx-3090 | 1.538 |  | torch-bf16 1.532 (1.00) | torch-bf16-compile 1.424 (1.08) | torch-bf16-compile-cg 1.369 (1.12) | **torch-bf16-compile-ro 1.369 (1.12)** | triton-bf16 1.558 (0.99) |
| pj_m2048_n6144_k4096_bf16 | a100 | 0.8165 |  | torch-bf16 0.4352 (1.88) | torch-bf16-compile 0.4147 (1.97) | **torch-bf16-compile-cg 0.4096 (1.99)** | torch-bf16-compile-ro 0.4096 (1.99) | triton-bf16 0.4936 (1.65) |
| pj_m2048_n6144_k4096_bf16 | rtx-4090 | 0.7327 |  | torch-bf16 0.7004 (1.05) | **torch-bf16-compile 0.5857 (1.25)** | torch-bf16-compile-cg 0.5857 (1.25) | torch-bf16-compile-ro 0.6636 (1.10) | triton-bf16 0.596 (1.23) |
| pj_m2048_n6144_k4096_bf16 | h100 | 0.737 |  | **torch-bf16 0.2662 (2.77)** | torch-bf16-compile 0.2769 (2.66) | torch-bf16-compile-cg 0.2757 (2.67) | torch-bf16-compile-ro 0.2765 (2.67) | triton-bf16 0.3397 (2.17) |

### MLP chains

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| mlp2_2048 | rtx-5080 | 1.943 | **chain-lt2 1.749 (1.11)**<br>chain-streams 1.773 (1.10)<br>chain-eager 1.865 (1.04)<br>chain-lt 2.05 (0.95)<br>chain-cutlass 2.641 (0.74) | torch-eager 1.878 (1.03) | torch-compile 1.857 (1.05) | torch-compile-cg 1.844 (1.05) | torch-compile-ro 2.264 (0.86) | triton 2.415 (0.80) |
| mlp2_2048 | rtx-3090 | 3.142 | **chain-lt2 2.975 (1.06)**<br>chain-streams 3.077 (1.02)<br>chain-lt 3.145 (1.00)<br>chain-eager 3.224 (0.97)<br>chain-cutlass 3.682 (0.85) | torch-eager 3.277 (0.96) | torch-compile 3.23 (0.97) | torch-compile-cg 3.253 (0.97) | torch-compile-ro 3.253 (0.97) | triton 3.686 (0.85) |
| mlp2_2048 | a100 | 4.401 | **chain-streams 3.759 (1.17)**<br>chain-lt2 3.793 (1.16)<br>chain-eager 3.972 (1.11)<br>chain-lt 4.032 (1.09)<br>chain-cutlass 4.617 (0.95) | torch-eager 5.137 (0.86) | torch-compile 3.956 (1.11) | torch-compile-cg 3.946 (1.12) | torch-compile-ro 3.949 (1.11) | triton 3.989 (1.10) |
| mlp2_2048 | rtx-4090 | 1.205 | **chain-lt2 1.227 (0.98)**<br>chain-cutlass 1.227 (0.98)<br>chain-lt 1.246 (0.97)<br>chain-streams 1.25 (0.96)<br>chain-eager 1.271 (0.95) | torch-eager 1.25 (0.96) | torch-compile 1.232 (0.98) | torch-compile-cg 1.229 (0.98) | torch-compile-ro 1.228 (0.98) | triton 1.449 (0.83) |
| mlp2_2048 | h100 | 1.893 | chain-lt2 1.989 (0.95)<br>chain-streams 1.99 (0.95)<br>chain-lt 2.007 (0.94)<br>chain-eager 2.036 (0.93)<br>chain-cutlass 2.257 (0.84) | torch-eager 1.898 (1.00) | **torch-compile 1.793 (1.06)** | torch-compile-cg 1.793 (1.06) | torch-compile-ro 1.852 (1.02) | triton 2.407 (0.79) |
| mlp_2048 | rtx-5080 | 0.9459 | **chain-lt 0.9114 (1.04)**<br>chain-eager 0.9257 (1.02)<br>chain-cutlass 1.089 (0.87) | torch-eager 0.9257 (1.02) | torch-compile 0.9257 (1.02) | torch-compile-cg 0.9233 (1.02) | torch-compile-ro 0.9254 (1.02) | triton 1.214 (0.78) |
| mlp_2048 | rtx-3090 | 1.584 | chain-lt 1.539 (1.03)<br>chain-eager 1.58 (1.00)<br>chain-cutlass 1.813 (0.87) | **torch-eager 1.449 (1.09)** | torch-compile 1.59 (1.00) | torch-compile-cg 1.458 (1.09) | torch-compile-ro 1.464 (1.08) | triton 1.864 (0.85) |
| mlp_2048 | a100 | 2.223 | **chain-lt 1.964 (1.13)**<br>chain-eager 1.987 (1.12)<br>chain-cutlass 2.307 (0.96) | torch-eager 1.979 (1.12) | torch-compile 1.978 (1.12) | torch-compile-cg 1.975 (1.13) | torch-compile-ro 1.975 (1.13) | triton 1.994 (1.12) |
| mlp_2048 | rtx-4090 | 0.589 | **chain-cutlass 0.6113 (0.96)**<br>chain-lt 0.6164 (0.96)<br>chain-eager 0.6277 (0.94) | torch-eager 0.6194 (0.95) | torch-compile 0.6185 (0.95) | torch-compile-cg 0.6175 (0.95) | torch-compile-ro 0.6216 (0.95) | triton 0.7139 (0.83) |
| mlp_2048 | h100 | 0.9285 | chain-lt 0.8852 (1.05)<br>chain-eager 0.8935 (1.04)<br>chain-cutlass 1.017 (0.91) | torch-eager 0.9067 (1.02) | torch-compile 0.8576 (1.08) | torch-compile-cg 0.8812 (1.05) | **torch-compile-ro 0.8412 (1.10)** | triton 1.099 (0.84) |
| mlp_512 | rtx-5080 | 0.04079 | chain-eager 0.043 (0.95)<br>chain-lt 0.043 (0.95)<br>chain-cutlass 0.1024 (0.40) | torch-eager 0.0451 (0.90) | **torch-compile 0.0328 (1.24)** | torch-compile-cg 0.0328 (1.24) | torch-compile-ro 0.043 (0.95) | triton 0.0328 (1.24) |
| mlp_512 | rtx-3090 | 0.05491 | chain-lt 0.045 (1.22)<br>chain-eager 0.0481 (1.14)<br>chain-cutlass 0.1074 (0.51) | torch-eager 0.0482 (1.14) | torch-compile 0.0502 (1.09) | torch-compile-cg 0.0492 (1.12) | torch-compile-ro 0.0522 (1.05) | **triton 0.0399 (1.38)** |
| mlp_512 | a100 | 0.07764 | **chain-lt 0.0625 (1.24)**<br>chain-eager 0.0655 (1.19)<br>chain-cutlass 0.2058 (0.38) | torch-eager 0.0655 (1.19) | torch-compile 0.0748 (1.04) | torch-compile-cg 0.0727 (1.07) | torch-compile-ro 0.0788 (0.99) | triton 0.0758 (1.02) |
| mlp_512 | rtx-4090 | 0.05532 | chain-lt 0.0358 (1.55)<br>chain-eager 0.038 (1.46)<br>chain-cutlass 0.086 (0.64) | torch-eager 0.0379 (1.46) | torch-compile 0.0284 (1.95) | **torch-compile-cg 0.0266 (2.08)** | torch-compile-ro 0.0348 (1.59) | triton 0.0317 (1.75) |
| mlp_512 | h100 | 0.053 | chain-lt 0.0341 (1.55)<br>chain-eager 0.0427 (1.24)<br>chain-cutlass 0.1185 (0.45) | torch-eager 0.0358 (1.48) | torch-compile 0.0312 (1.70) | **torch-compile-cg 0.0301 (1.76)** | torch-compile-ro 0.0318 (1.67) | triton 0.0528 (1.00) |
| smlp_4096_256_1024 | rtx-5080 | 0.1672 | chain-lt 0.1553 (1.08)<br>chain-cutlass 0.1577 (1.06)<br>chain-eager 0.1679 (1.00) | torch-eager 0.1772 (0.94) | **torch-compile 0.1495 (1.12)** | torch-compile-cg 0.1495 (1.12) | torch-compile-ro 0.1638 (1.02) | triton 0.1782 (0.94) |
| smlp_4096_256_1024 | rtx-3090 | 0.2687 | **chain-cutlass 0.2212 (1.21)**<br>chain-eager 0.2499 (1.08)<br>chain-lt 0.3 (0.90) | torch-eager 0.2949 (0.91) | torch-compile 0.2273 (1.18) | torch-compile-cg 0.2273 (1.18) | torch-compile-ro 0.2734 (0.98) | triton 0.2499 (1.08) |
| smlp_4096_256_1024 | a100 | 0.3459 | **chain-eager 0.2929 (1.18)**<br>chain-lt 0.3359 (1.03)<br>chain-cutlass 0.3574 (0.97) | torch-eager 0.343 (1.01) | torch-compile 0.3471 (1.00) | torch-compile-cg 0.3451 (1.00) | torch-compile-ro 0.3645 (0.95) | triton 0.3072 (1.13) |
| smlp_4096_256_1024 | rtx-4090 | 0.1131 | chain-eager 0.129 (0.88)<br>chain-cutlass 0.1352 (0.84)<br>chain-lt 0.1935 (0.58) | torch-eager 0.1436 (0.79) | torch-compile 0.1065 (1.06) | torch-compile-cg 0.1044 (1.08) | torch-compile-ro 0.1249 (0.91) | **triton 0.1026 (1.10)** |
| smlp_4096_256_1024 | h100 | 0.1666 | **chain-lt 0.1248 (1.33)**<br>chain-eager 0.1352 (1.23)<br>chain-cutlass 0.1832 (0.91) | torch-eager 0.2902 (0.57) | torch-compile 0.1346 (1.24) | torch-compile-cg 0.1332 (1.25) | torch-compile-ro 0.13 (1.28) | triton 0.163 (1.02) |
| mlp2_2048_tf32 | rtx-5080 | 1.572 |  | torch-eager-tf32 1.545 (1.02) | torch-compile-tf32 1.282 ✗gate | torch-compile-cg-tf32 1.28 ✗gate | **torch-compile-ro-tf32 1.513 (1.04)** | triton-tf32 1.437 ✗gate |
| mlp2_2048_tf32 | rtx-3090 | 2.443 |  | torch-eager-tf32 2.534 (0.96) | torch-compile-tf32 2.183 ✗gate | torch-compile-cg-tf32 2.162 ✗gate | **torch-compile-ro-tf32 2.353 (1.04)** | triton-tf32 2.407 ✗gate |
| mlp2_2048_tf32 | a100 | 0.9325 |  | torch-eager-tf32 0.7567 (1.23) | torch-compile-tf32 0.7066 (1.32) | **torch-compile-cg-tf32 0.6984 (1.34)** | torch-compile-ro-tf32 0.6984 (1.34) | triton-tf32 0.7875 ✗gate |
| mlp2_2048_tf32 | rtx-4090 | 0.9832 |  | torch-eager-tf32 0.852 (1.15) | torch-compile-tf32 0.8223 (1.20) | **torch-compile-cg-tf32 0.8172 (1.20)** | torch-compile-ro-tf32 0.8172 (1.20) | triton-tf32 0.9298 ✗gate |
| mlp2_2048_tf32 | h100 | 0.7855 |  | torch-eager-tf32 0.3975 (1.98) | torch-compile-tf32 0.3729 (2.11) | torch-compile-cg-tf32 0.3704 (2.12) | **torch-compile-ro-tf32 0.3685 (2.13)** | triton-tf32 0.8894 ✗gate |
| mlp_2048_tf32 | rtx-5080 | 0.787 |  | torch-eager-tf32 0.7629 (1.03) | torch-compile-tf32 0.639 ✗gate | torch-compile-cg-tf32 0.6369 ✗gate | **torch-compile-ro-tf32 0.7598 (1.04)** | triton-tf32 0.7291 ✗gate |
| mlp_2048_tf32 | rtx-3090 | 1.207 |  | torch-eager-tf32 1.276 (0.95) | torch-compile-tf32 0.9705 ✗gate | torch-compile-cg-tf32 0.9677 ✗gate | **torch-compile-ro-tf32 1.145 (1.05)** | triton-tf32 1.216 ✗gate |
| mlp_2048_tf32 | a100 | 0.5092 |  | torch-eager-tf32 0.3543 (1.44) | torch-compile-tf32 0.3543 (1.44) | **torch-compile-cg-tf32 0.3502 (1.45)** | torch-compile-ro-tf32 0.3512 (1.45) | triton-tf32 0.3686 ✗gate |
| mlp_2048_tf32 | rtx-4090 | 0.4813 |  | torch-eager-tf32 0.43 (1.12) | torch-compile-tf32 0.4139 (1.16) | torch-compile-cg-tf32 0.4116 (1.17) | **torch-compile-ro-tf32 0.4106 (1.17)** | triton-tf32 0.47 ✗gate |
| mlp_2048_tf32 | h100 | 0.4059 |  | torch-eager-tf32 0.1868 (2.17) | torch-compile-tf32 0.1902 (2.13) | torch-compile-cg-tf32 0.2146 (1.89) | **torch-compile-ro-tf32 0.1742 (2.33)** | triton-tf32 0.4678 ✗gate |

### Attention / softmax / long rows

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | hand Triton | hand CUDA |
|---|---|---:|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-5080 | 0.04307 | chain-eager 0.041 (1.05) | torch-eager 0.0492 (0.88) | **torch-compile 0.0287 (1.50)** | torch-compile-cg 0.0287 (1.50)<br>torch-sdpa-compile-cg 0.0307 (1.40) | torch-compile-ro 0.0287 (1.50) | torch-sdpa-compile 0.0307 (1.40)<br>torch-sdpa 0.0451 (0.95) | triton 0.1516 (0.28) | chain-fused 0.0348 (1.24) |
| attn_1024_64 | rtx-3090 | 0.07689 | chain-eager 0.0645 (1.19) | torch-eager 0.085 (0.90) | torch-compile 0.0492 (1.56) | torch-compile-cg 0.0471 (1.63)<br>torch-sdpa-compile-cg 0.0543 (1.42) | **torch-compile-ro 0.0451 (1.70)** | torch-sdpa-compile 0.0584 (1.32)<br>torch-sdpa 0.0819 (0.94) | triton 0.2284 (0.34) | chain-fused 0.0512 (1.50) |
| attn_1024_64 | a100 | 0.064 | chain-eager 0.0686 (0.93) | torch-eager 0.087 (0.74) | torch-compile 0.0614 (1.04) | **torch-sdpa-compile-cg 0.0543 (1.18)**<br>torch-compile-cg 0.0563 (1.14) | torch-compile-ro 0.0563 (1.14) | torch-sdpa-compile 0.0614 (1.04)<br>torch-sdpa 0.1014 (0.63) | triton 0.2376 (0.27) | chain-fused 0.0604 (1.06) |
| attn_1024_64 | rtx-4090 | 0.05166 | chain-eager 0.043 (1.20) | torch-eager 0.0481 (1.07) | torch-compile 0.0317 (1.63) | **torch-sdpa-compile-cg 0.0276 (1.87)**<br>torch-compile-cg 0.0297 (1.74) | torch-compile-ro 0.0297 (1.74) | torch-sdpa-compile 0.0317 (1.63)<br>torch-sdpa 0.0481 (1.07) | triton 0.1475 (0.35) | chain-fused 0.0369 (1.40) |
| attn_1024_64 | h100 | 0.05382 | chain-eager 0.0418 (1.29) | torch-eager 0.0618 (0.87) | torch-compile 0.041 (1.31) | **torch-compile-cg 0.0247 (2.18)**<br>torch-sdpa-compile-cg 0.0271 (1.99) | torch-compile-ro 0.0249 (2.16) | torch-sdpa-compile 0.0409 (1.32)<br>torch-sdpa 0.0534 (1.01) | triton 0.2301 (0.23) | chain-fused 0.036 (1.50) |
| attn_2048_128 | rtx-5080 | 0.12 | chain-eager 0.1393 (0.86) | torch-eager 0.1516 (0.79) | torch-compile 0.1004 (1.20) | torch-sdpa-compile-cg 0.0942 (1.27)<br>torch-compile-cg 0.1065 (1.13) | torch-compile-ro 0.1004 (1.20) | **torch-sdpa-compile 0.0942 (1.27)**<br>torch-sdpa 0.1362 (0.88) | triton 0.7885 (0.15) | chain-fused 0.129 (0.93) |
| attn_2048_128 | rtx-3090 | 0.2406 | chain-eager 0.2724 (0.88) | torch-eager 0.3697 (0.65) | torch-compile 0.1802 (1.34) | **torch-sdpa-compile-cg 0.1761 (1.37)**<br>torch-compile-cg 0.1853 (1.30) | torch-compile-ro 0.1925 (1.25) | torch-sdpa-compile 0.1812 (1.33)<br>torch-sdpa 0.2857 (0.84) | triton 0.8796 (0.27) | chain-fused 0.2427 (0.99) |
| attn_2048_128 | a100 | 0.2609 | chain-eager 0.2263 (1.15) | torch-eager 0.3277 (0.80) | **torch-compile 0.1741 (1.50)** | torch-compile-cg 0.1925 (1.36)<br>torch-sdpa-compile-cg 0.2202 (1.18) | torch-compile-ro 0.1935 (1.35) | torch-sdpa-compile 0.1772 (1.47)<br>torch-sdpa 0.2806 (0.93) | triton 1.422 (0.18) | chain-fused 0.2202 (1.18) |
| attn_2048_128 | rtx-4090 | 0.09487 | chain-eager 0.1311 (0.72) | torch-eager 0.1505 (0.63) | torch-compile 0.0819 (1.16) | torch-sdpa-compile-cg 0.0809 (1.17)<br>torch-compile-cg 0.0819 (1.16) | **torch-compile-ro 0.0799 (1.19)** | torch-sdpa-compile 0.088 (1.08)<br>torch-sdpa 0.1404 (0.68) | triton 0.5868 (0.16) | chain-fused 0.1198 (0.79) |
| attn_2048_128 | h100 | 0.1267 | chain-eager 0.1203 (1.05) | torch-eager 0.1403 (0.90) | torch-compile 0.0879 (1.44) | **torch-compile-cg 0.0848 (1.49)**<br>torch-sdpa-compile-cg 0.0976 (1.30) | torch-compile-ro 0.0848 (1.49) | torch-sdpa-compile 0.0992 (1.28)<br>torch-sdpa 0.1244 (1.02) | triton 1.087 (0.12) | chain-fused 0.109 (1.16) |
| attn_2048_64 | rtx-5080 | 0.08184 | chain-eager 0.1085 (0.75) | torch-eager 0.1219 (0.67) | torch-compile 0.0758 (1.08) | torch-compile-cg 0.0758 (1.08)<br>torch-sdpa-compile-cg 0.0758 (1.08) | **torch-compile-ro 0.0696 (1.18)** | torch-sdpa-compile 0.0748 (1.09)<br>torch-sdpa 0.1055 (0.78) | triton 0.2949 (0.28) | chain-fused 0.0942 (0.87) |
| attn_2048_64 | rtx-3090 | 0.1511 | chain-eager 0.2202 (0.69) | torch-eager 0.298 (0.51) | torch-compile 0.1331 (1.14) | torch-compile-cg 0.1321 (1.14)<br>torch-sdpa-compile-cg 0.1864 (0.81) | **torch-compile-ro 0.1293 (1.17)** | torch-sdpa-compile 0.17 (0.89)<br>torch-sdpa 0.2171 (0.70) | triton 0.382 (0.40) | chain-fused 0.1915 (0.79) |
| attn_2048_64 | a100 | 0.1555 | chain-eager 0.17 (0.91) | torch-eager 0.2529 (0.62) | **torch-compile 0.1106 (1.41)** | torch-compile-cg 0.1341 (1.16)<br>torch-sdpa-compile-cg 0.1393 (1.12) | torch-compile-ro 0.1341 (1.16) | torch-sdpa-compile 0.1147 (1.36)<br>torch-sdpa 0.2243 (0.69) | triton 0.468 (0.33) | chain-fused 0.1638 (0.95) |
| attn_2048_64 | rtx-4090 | 0.07445 | chain-eager 0.1096 (0.68) | torch-eager 0.1331 (0.56) | torch-compile 0.0778 (0.96) | torch-sdpa-compile-cg 0.0696 (1.07)<br>torch-compile-cg 0.0748 (1.00) | **torch-compile-ro 0.0604 (1.23)** | torch-sdpa-compile 0.0737 (1.01)<br>torch-sdpa 0.1249 (0.60) | triton 0.2725 (0.27) | chain-fused 0.0973 (0.77) |
| attn_2048_64 | h100 | 0.08674 | chain-eager 0.0972 (0.89) | torch-eager 0.1183 (0.73) | torch-compile 0.0646 (1.34) | torch-compile-cg 0.0588 (1.48)<br>torch-sdpa-compile-cg 0.0612 (1.42) | **torch-compile-ro 0.0586 (1.48)** | torch-sdpa-compile 0.0634 (1.37)<br>torch-sdpa 0.1018 (0.85) | triton 0.4528 (0.19) | chain-fused 0.0817 (1.06) |
| causal_1024_256 | rtx-5080 | 15.37 |  | torch-eager 96.9 (0.16) | torch-compile 51.28 (0.30) | **torch-compile-cg 42.76 (0.36)** | torch-compile-ro 51.86 (0.30) |  |  |  |
| causal_1024_256 | rtx-3090 | 21.52 |  | torch-eager 276 (0.08) | torch-compile 170.4 (0.13) | **torch-compile-cg 155.2 (0.14)** | torch-compile-ro 155.8 (0.14) |  |  |  |
| causal_1024_256 | a100 | 36.87 |  | torch-eager 149.1 (0.25) | **torch-compile 74.15 (0.50)** | torch-compile-cg 74.51 (0.49) | torch-compile-ro 80.63 (0.46) |  |  |  |
| causal_1024_256 | rtx-4090 | 17.94 |  | torch-eager 338.3 (0.05) | torch-compile 196.4 (0.09) | **torch-compile-cg 177 (0.10)** | torch-compile-ro 177.4 (0.10) |  |  |  |
| causal_1024_256 | h100 | 21.32 |  | torch-eager 201.9 (0.11) | torch-compile 118 (0.18) | torch-compile-cg 117 (0.18) | **torch-compile-ro 116 (0.18)** |  |  |  |
| decode_1024_256 | rtx-5080 | 8.729 | chain-nocache 23.75 (0.37) | torch-eager 51.55 (0.17) | torch-compile 51.18 (0.17) | torch-compile-cg 52.8 (0.17) | torch-compile-ro 49.84 (0.18) |  |  | **chain-cache 16.44 (0.53)** |
| decode_1024_256 | rtx-3090 | 14.77 | chain-nocache 36.76 (0.40) | torch-eager 155.7 (0.09) | torch-compile 151.5 (0.10) | torch-compile-cg 156.6 (0.09) | torch-compile-ro 155 (0.10) |  |  | **chain-cache 19.62 (0.75)** |
| decode_1024_256 | a100 | 28.05 | chain-nocache 53.99 (0.52) | torch-eager 79.24 (0.35) | torch-compile 65.82 (0.43) | torch-compile-cg 73.05 (0.38) | torch-compile-ro 75.86 (0.37) |  |  | **chain-cache 29.33 (0.96)** |
| decode_1024_256 | rtx-4090 | 12.46 | chain-nocache 32.26 (0.39) | torch-eager 184.4 (0.07) | torch-compile 173.5 (0.07) | torch-compile-cg 175.5 (0.07) | torch-compile-ro 183.1 (0.07) |  |  | **chain-cache 30.29 (0.41)** |
| decode_1024_256 | h100 | 28.57 | chain-nocache 30.11 (0.95) | torch-eager 103.1 (0.28) | torch-compile 101.8 (0.28) | torch-compile-cg 112 (0.26) | torch-compile-ro 110.4 (0.26) |  |  | **chain-cache 19.71 (1.45)** |
| gen_1024_512_256 | rtx-5080 | 7.762 |  | torch-eager 48.9 (0.16) | torch-compile 29.85 (0.26) | **torch-compile-cg 24.48 (0.32)** | torch-compile-ro 26.21 (0.30) |  |  |  |
| gen_1024_512_256 | rtx-3090 | 10.86 |  | torch-eager 138.3 (0.08) | torch-compile 85.08 (0.13) | **torch-compile-cg 77.58 (0.14)** | torch-compile-ro 78.28 (0.14) |  |  |  |
| gen_1024_512_256 | a100 | 18.76 |  | torch-eager 75.09 (0.25) | torch-compile 37.67 (0.50) | **torch-compile-cg 37.22 (0.50)** | torch-compile-ro 40.3 (0.47) |  |  |  |
| gen_1024_512_256 | rtx-4090 | 9.14 |  | torch-eager 180.1 (0.05) | torch-compile 97.54 (0.09) | **torch-compile-cg 88.22 (0.10)** | torch-compile-ro 91.48 (0.10) |  |  |  |
| gen_1024_512_256 | h100 | 10.8 |  | torch-eager 103.4 (0.10) | torch-compile 58.21 (0.19) | **torch-compile-cg 56.81 (0.19)** | torch-compile-ro 57.21 (0.19) |  |  |  |
| lsm_1_131072 | rtx-5080 | 0.02269 |  | torch-eager 0.0369 (0.61) | torch-compile 0.0266 (0.85) | torch-sdpa-compile-cg 0.0246 (0.92)<br>torch-compile-cg 0.0263 (0.86) | **torch-compile-ro 0.0246 (0.92)** | torch-sdpa-compile 0.0276 (0.82)<br>torch-sdpa 0.0532 (0.43) |  |  |
| lsm_1_131072 | rtx-3090 | 0.02453 |  | torch-eager 0.0471 (0.52) | torch-compile 0.0287 (0.85) | **torch-compile-cg 0.0256 (0.96)**<br>torch-sdpa-compile-cg 0.0317 (0.77) | torch-compile-ro 0.0256 (0.96) | torch-sdpa-compile 0.0328 (0.75)<br>torch-sdpa 0.0717 (0.34) |  |  |
| lsm_1_131072 | a100 | 0.02483 |  | torch-eager 0.0655 (0.38) | torch-compile 0.0338 (0.73) | **torch-compile-cg 0.0256 (0.97)**<br>torch-sdpa-compile-cg 0.0266 (0.93) | torch-compile-ro 0.0266 (0.93) | torch-sdpa-compile 0.0328 (0.76)<br>torch-sdpa 0.0922 (0.27) |  |  |
| lsm_1_131072 | rtx-4090 | 0.02793 |  | torch-eager 0.0379 (0.74) | torch-compile 0.0276 (1.01) | **torch-compile-cg 0.0236 (1.18)**<br>torch-sdpa-compile-cg 0.0246 (1.14) | torch-compile-ro 0.0236 (1.18) | torch-sdpa-compile 0.0287 (0.97)<br>torch-sdpa 0.0532 (0.53) |  |  |
| lsm_1_131072 | h100 | 0.03181 |  | torch-eager 0.0432 (0.74) | torch-compile 0.0225 (1.41) | **torch-compile-cg 0.0188 (1.69)**<br>torch-sdpa-compile-cg 0.0229 (1.39) | torch-compile-ro 0.0191 (1.67) | torch-sdpa-compile 0.0514 (0.62)<br>torch-sdpa 0.0596 (0.53) |  |  |
| lsm_32_8192 | rtx-5080 | 0.02416 |  | torch-eager 0.0215 (1.12) | torch-compile 0.0123 (1.96) | **torch-compile-cg 0.0102 (2.37)**<br>torch-sdpa-compile-cg 0.0102 (2.37) | torch-compile-ro 0.0102 (2.37) | torch-sdpa-compile 0.0102 (2.37)<br>torch-sdpa 0.0123 (1.96) | triton 0.0102 (2.37) |  |
| lsm_32_8192 | rtx-3090 | 0.02351 |  | torch-eager 0.0307 (0.77) | torch-compile 0.0102 (2.30) | **torch-compile-cg 0.0092 (2.56)**<br>torch-sdpa-compile-cg 0.0102 (2.30) | torch-compile-ro 0.0113 (2.08) | torch-sdpa-compile 0.0113 (2.08)<br>torch-sdpa 0.0164 (1.43) | triton 0.0123 (1.91) |  |
| lsm_32_8192 | a100 | 0.02762 |  | torch-eager 0.042 (0.66) | torch-compile 0.0143 (1.93) | **torch-compile-cg 0.0123 (2.25)**<br>torch-sdpa-compile-cg 0.0123 (2.25) | torch-compile-ro 0.0133 (2.08) | torch-sdpa-compile 0.0154 (1.79)<br>torch-sdpa 0.0205 (1.35) | triton 0.0164 (1.68) |  |
| lsm_32_8192 | rtx-4090 | 0.0417 |  | torch-eager 0.0225 (1.85) | torch-compile 0.0092 (4.53) | **torch-compile-cg 0.0082 (5.09)**<br>torch-sdpa-compile-cg 0.0082 (5.09) | torch-compile-ro 0.0082 (5.09) | torch-sdpa-compile 0.0092 (4.53)<br>torch-sdpa 0.0133 (3.14) | triton 0.0102 (4.09) |  |
| lsm_32_8192 | h100 | 0.02699 |  | torch-eager 0.0304 (0.89) | torch-compile 0.0324 (0.83) | torch-sdpa-compile-cg 0.0108 (2.50)<br>torch-compile-cg 0.0137 (1.97) | **torch-compile-ro 0.0097 (2.78)** | torch-sdpa-compile 0.0099 (2.73)<br>torch-sdpa 0.0142 (1.90) | triton 0.0116 (2.33) |  |
| lsm_4_32768 | rtx-5080 | 0.02134 |  | torch-eager 0.0256 (0.83) | torch-compile 0.0164 (1.30) | **torch-compile-cg 0.0143 (1.49)**<br>torch-sdpa-compile-cg 0.0143 (1.49) | torch-compile-ro 0.0143 (1.49) | torch-sdpa-compile 0.0164 (1.30)<br>torch-sdpa 0.0184 (1.16) | not run |  |
| lsm_4_32768 | rtx-3090 | 0.01861 |  | torch-eager 0.0382 (0.49) | torch-compile 0.0205 (0.91) | **torch-compile-cg 0.0164 (1.13)**<br>torch-sdpa-compile-cg 0.0164 (1.13) | torch-compile-ro 0.0164 (1.13) | torch-sdpa 0.0236 (0.79)<br>torch-sdpa-compile 0.0236 (0.79) | not run |  |
| lsm_4_32768 | a100 | 0.02164 |  | torch-eager 0.0532 (0.41) | torch-compile 0.0297 (0.73) | **torch-compile-cg 0.0195 (1.11)**<br>torch-sdpa-compile-cg 0.0246 (0.88) | torch-compile-ro 0.0205 (1.06) | torch-sdpa 0.0297 (0.73)<br>torch-sdpa-compile 0.0338 (0.64) | triton 0.0297 (0.73) |  |
| lsm_4_32768 | rtx-4090 | 0.02672 |  | torch-eager 0.0297 (0.90) | torch-compile 0.0174 (1.54) | **torch-compile-cg 0.0133 (2.01)**<br>torch-sdpa-compile-cg 0.0143 (1.87) | torch-compile-ro 0.0143 (1.87) | torch-sdpa-compile 0.0174 (1.54)<br>torch-sdpa 0.0195 (1.37) | not run |  |
| lsm_4_32768 | h100 | 0.02757 |  | torch-eager 0.0376 (0.73) | torch-compile 0.02 (1.38) | **torch-sdpa-compile-cg 0.0154 (1.79)**<br>torch-compile-cg 0.0163 (1.69) | torch-compile-ro 0.0164 (1.68) | torch-sdpa-compile 0.0206 (1.34)<br>torch-sdpa 0.0213 (1.29) | not run |  |
| prefill_1024_512_256 | rtx-5080 | 8.345 |  | torch-eager 51.83 (0.16) | torch-compile 29.73 (0.28) | **torch-compile-cg 24.98 (0.33)** | torch-compile-ro 26.5 (0.31) |  |  |  |
| prefill_1024_512_256 | rtx-3090 | 11.31 |  | torch-eager 139.9 (0.08) | torch-compile 86.33 (0.13) | torch-compile-cg 78.35 (0.14) | **torch-compile-ro 78.32 (0.14)** |  |  |  |
| prefill_1024_512_256 | a100 | 18.97 |  | torch-eager 75.09 (0.25) | torch-compile 37.8 (0.50) | **torch-compile-cg 37.31 (0.51)** | torch-compile-ro 40.42 (0.47) |  |  |  |
| prefill_1024_512_256 | rtx-4090 | 9.319 |  | torch-eager 177.4 (0.05) | torch-compile 99.97 (0.09) | **torch-compile-cg 89.74 (0.10)** | torch-compile-ro 93.96 (0.10) |  |  |  |
| prefill_1024_512_256 | h100 | 10.95 |  | torch-eager 103.2 (0.11) | torch-compile 58.94 (0.19) | torch-compile-cg 59.34 (0.18) | **torch-compile-ro 57.04 (0.19)** |  |  |  |
| rowsum_2048 | rtx-5080 | 0.4651 | **chain-eager 0.4628 (1.00)**<br>chain-gemv 0.4649 (1.00) | torch-eager 0.4659 (1.00) | torch-compile 0.4649 (1.00) | torch-compile-cg 0.4649 (1.00) | torch-compile-ro 0.4649 (1.00) |  |  |  |
| rowsum_2048 | rtx-3090 | 0.8224 | chain-gemv 0.7967 (1.03)<br>chain-eager 0.8008 (1.03) | torch-eager 0.7668 (1.07) | **torch-compile 0.7619 (1.08)** | torch-compile-cg 0.7619 (1.08) | torch-compile-ro 0.7629 (1.08) |  |  |  |
| rowsum_2048 | a100 | 1.035 | chain-eager 0.9964 (1.04)<br>chain-gemv 1 (1.04) | torch-eager 1.002 (1.03) | torch-compile 0.9964 (1.04) | torch-compile-cg 0.9953 (1.04) | **torch-compile-ro 0.9943 (1.04)** |  |  |  |
| rowsum_2048 | rtx-4090 | 0.3055 | chain-eager 0.3144 (0.97)<br>chain-gemv 0.3164 (0.97) | torch-eager 0.3164 (0.97) | torch-compile 0.3144 (0.97) | torch-compile-cg 0.3123 (0.98) | **torch-compile-ro 0.3092 (0.99)** |  |  |  |
| rowsum_2048 | h100 | 0.5099 | chain-eager 0.426 (1.20)<br>chain-gemv 0.4277 (1.19) | torch-eager 0.4222 (1.21) | torch-compile 0.4213 (1.21) | **torch-compile-cg 0.4199 (1.21)** | torch-compile-ro 0.4203 (1.21) |  |  |  |
| rowsum_4096 | rtx-5080 | 3.738 | chain-gemv 3.609 (1.04)<br>chain-eager 3.62 (1.03) | torch-eager 3.911 (0.96) | torch-compile 3.918 (0.95) | **torch-compile-cg 3.498 (1.07)** | torch-compile-ro 3.907 (0.96) |  |  |  |
| rowsum_4096 | rtx-3090 | 6.163 | chain-eager 5.789 (1.06)<br>chain-gemv 5.791 (1.06) | torch-eager 5.746 (1.07) | torch-compile 5.793 (1.06) | torch-compile-cg 5.792 (1.06) | **torch-compile-ro 5.743 (1.07)** |  |  |  |
| rowsum_4096 | a100 | 7.749 | chain-eager 7.346 (1.05)<br>chain-gemv 7.35 (1.05) | torch-eager 7.351 (1.05) | torch-compile 7.655 (1.01) | torch-compile-cg 7.71 (1.01) | **torch-compile-ro 7.336 (1.06)** |  |  |  |
| rowsum_4096 | rtx-4090 | 2.426 | chain-gemv 2.438 (0.99)<br>chain-eager 2.454 (0.99) | torch-eager 2.489 (0.97) | torch-compile 2.432 (1.00) | **torch-compile-cg 2.375 (1.02)** | torch-compile-ro 2.379 (1.02) |  |  |  |
| rowsum_4096 | h100 | 4.606 | chain-eager 4.018 (1.15)<br>chain-gemv 4.024 (1.14) | torch-eager 4.028 (1.14) | torch-compile 4.016 (1.15) | torch-compile-cg 4.059 (1.13) | **torch-compile-ro 4.006 (1.15)** |  |  |  |
| softmax_rows_2048 | rtx-5080 | 0.4801 | chain-eager 0.5018 (0.96) | torch-eager 0.5222 (0.92) | torch-compile 0.4731 (1.01) | **torch-compile-cg 0.4707 (1.02)** | torch-compile-ro 0.4731 (1.01) |  | triton 0.6103 (0.79) | chain-fused 0.4915 (0.98) |
| softmax_rows_2048 | rtx-3090 | 0.8558 | chain-eager 0.8274 (1.03) | torch-eager 0.9267 (0.92) | **torch-compile 0.7404 (1.16)** | torch-compile-cg 0.7854 (1.09) | torch-compile-ro 0.7894 (1.08) |  | triton 0.8741 (0.98) | chain-fused 0.7997 (1.07) |
| softmax_rows_2048 | a100 | 1.052 | chain-eager 1.055 (1.00) | torch-eager 1.402 (0.75) | **torch-compile 1.004 (1.05)** | torch-compile-cg 1.267 (0.83) | torch-compile-ro 1.271 (0.83) |  | triton 1.019 (1.03) | chain-fused 1.044 (1.01) |
| softmax_rows_2048 | rtx-4090 | 0.3148 | chain-eager 0.3483 (0.90) | torch-eager 0.3697 (0.85) | **torch-compile 0.3132 (1.01)** | torch-compile-cg 0.3144 (1.00) | torch-compile-ro 0.3144 (1.00) |  | triton 0.3617 (0.87) | chain-fused 0.341 (0.92) |
| softmax_rows_2048 | h100 | 0.5219 | chain-eager 0.4626 (1.13) | torch-eager 0.4821 (1.08) | torch-compile 0.4284 (1.22) | torch-compile-cg 0.4292 (1.22) | **torch-compile-ro 0.4272 (1.22)** |  | triton 0.5188 (1.01) | chain-fused 0.4467 (1.17) |
| softmax_rows_4096 | rtx-5080 | 3.838 | chain-eager 3.924 (0.98) | torch-eager 4.148 (0.93) | torch-compile 3.562 (1.08) | torch-compile-cg 3.556 (1.08) | **torch-compile-ro 3.547 (1.08)** |  | triton 4.583 (0.84) | chain-fused 3.762 (1.02) |
| softmax_rows_4096 | rtx-3090 | 6.344 | chain-eager 6.218 (1.02) | torch-eager 6.435 (0.99) | **torch-compile 5.821 (1.09)** | torch-compile-cg 5.869 (1.08) | torch-compile-ro 5.872 (1.08) |  | triton 6.914 (0.92) | chain-fused 6.109 (1.04) |
| softmax_rows_4096 | a100 | 7.816 | chain-eager 7.628 (1.02) | torch-eager 7.824 (1.00) | torch-compile 7.671 (1.02) | torch-compile-cg 7.669 (1.02) | **torch-compile-ro 7.376 (1.06)** |  | triton 7.56 (1.03) | chain-fused 7.567 (1.03) |
| softmax_rows_4096 | rtx-4090 | 2.487 | chain-eager 2.676 (0.93) | torch-eager 2.949 (0.84) | torch-compile 2.46 (1.01) | **torch-compile-cg 2.428 (1.02)** | torch-compile-ro 2.438 (1.02) |  | triton 3.01 (0.83) | chain-fused 2.564 (0.97) |
| softmax_rows_4096 | h100 | 4.639 | chain-eager 4.2 (1.10) | torch-eager 4.381 (1.06) | **torch-compile 3.8 (1.22)** | torch-compile-cg 4.081 (1.14) | torch-compile-ro 4.023 (1.15) |  | triton 5.092 (0.91) | chain-fused 4.191 (1.11) |

### LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | part | Mapal ms | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | FlashAttention-2/3 | FlashInfer | hand Triton | Liger |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | rtx-5080 | 1.073 | torch-eager 1.149 (0.93) | **torch-compile 1.07 (1.00)** | torch-compile-cg 1.079 (0.99) | torch-compile-ro 1.11 (0.97) | torch-sdpa 1.557 (0.69)<br>torch-sdpa-efficient 1.878 (0.57) |  |  |  | liger 1.914 (0.56) |
| llm_blkdec_b1_l4096 | rtx-3090 | 1.38 | torch-eager 1.452 (0.95) | torch-compile 1.209 (1.14) | **torch-compile-cg 1.203 (1.15)** | torch-compile-ro 1.286 (1.07) | torch-sdpa 1.863 (0.74)<br>torch-sdpa-efficient 2.482 (0.56) |  |  |  | liger 2.485 (0.56) |
| llm_blkdec_b1_l4096 | a100 | 1.251 | torch-eager 1.057 (1.18) | **torch-compile 0.8228 (1.52)** | torch-compile-cg 0.8228 (1.52) | torch-compile-ro 0.9505 (1.32) | torch-sdpa 1.316 (0.95)<br>torch-sdpa-efficient 1.6 (0.78) |  |  |  | liger 1.59 (0.79) |
| llm_blkdec_b1_l4096 | rtx-4090 | 1.044 | torch-eager 1.457 (0.72) | **torch-compile 1.088 (0.96)** | torch-compile-cg 1.105 (0.94) | torch-compile-ro 1.16 (0.90) | torch-sdpa 1.733 (0.60)<br>torch-sdpa-efficient 2.108 (0.50) |  |  |  | liger 2.102 (0.50) |
| llm_blkdec_b1_l4096 | h100 | 0.7931 | torch-eager 0.8155 (0.97) | **torch-compile 0.5657 (1.40)** | torch-compile-cg 0.6055 (1.31) | torch-compile-ro 0.6664 (1.19) | torch-sdpa 0.8965 (0.88)<br>torch-sdpa-efficient 1.24 (0.64) |  |  |  | liger 1.273 (0.62) |
| llm_blkpre_t2048 | rtx-5080 | 25.38 | torch-eager 33.06 (0.77) | torch-compile 28.15 (0.90) | torch-compile-cg 28.09 (0.90) | torch-compile-ro 29.14 (0.87) | torch-sdpa-efficient 27.14 (0.94)<br>torch-sdpa 33.43 (0.76) |  |  | triton 29.56 (0.86) | **liger 26.84 (0.95)** |
| llm_blkpre_t2048 | rtx-3090 | 45.42 | torch-eager 47.42 (0.96) | torch-compile 42.23 (1.08) | torch-compile-cg 42.21 (1.08) | torch-compile-ro 42.41 (1.07) | torch-sdpa-efficient 42.24 (1.08)<br>torch-sdpa 47.54 (0.96) |  |  | triton 46.25 (0.98) | **liger 41.26 (1.10)** |
| llm_blkpre_t2048 | a100 | 55.87 | torch-eager 61.47 (0.91) | **torch-compile 54.53 (1.02)** | torch-compile-cg 54.53 (1.02) | torch-compile-ro 58.7 (0.95) | torch-sdpa-efficient 56.43 (0.99)<br>torch-sdpa 61.85 (0.90) |  |  | triton 60.61 (0.92) | liger 55.83 (1.00) |
| llm_blkpre_t2048 | rtx-4090 | 18.8 | torch-eager 22.78 (0.83) | torch-compile 18.97 (0.99) | torch-compile-cg 18.95 (0.99) | torch-compile-ro 19.04 (0.99) | torch-sdpa-efficient 18.4 (1.02)<br>torch-sdpa 22.6 (0.83) |  |  | triton 20.18 (0.93) | **liger 17.95 (1.05)** |
| llm_blkpre_t2048 | h100 | 29.74 | torch-eager 34.12 (0.87) | torch-compile 29.68 (1.00) | torch-compile-cg 29.63 (1.00) | torch-compile-ro 30.19 (0.98) | torch-sdpa-efficient 29.45 (1.01)<br>torch-sdpa 32.78 (0.91) |  |  | triton 32.46 (0.92) | **liger 27.99 (1.06)** |
| llm_dec_b1_l2048 | rtx-5080 | 0.2407 | torch-eager 0.3038 (0.79) | **torch-compile 0.2589 (0.93)** | torch-compile-cg 0.2701 (0.89) | torch-compile-ro 0.2592 (0.93) | torch-sdpa 0.4879 (0.49)<br>torch-sdpa-efficient 0.6694 (0.36) |  |  |  | liger 0.708 (0.34) |
| llm_dec_b1_l2048 | rtx-3090 | 0.3259 | torch-eager 0.5654 (0.58) | **torch-compile 0.3454 (0.94)** | torch-compile-cg 0.3663 (0.89) | torch-compile-ro 0.3755 (0.87) | torch-sdpa 0.7275 (0.45)<br>torch-sdpa-efficient 1.026 (0.32) |  |  |  | liger 1.04 (0.31) |
| llm_dec_b1_l2048 | a100 | 0.4713 | torch-eager 0.3422 (1.38) | **torch-compile 0.2511 (1.88)** | torch-compile-cg 0.2534 (1.86) | torch-compile-ro 0.2525 (1.87) | torch-sdpa 0.5061 (0.93)<br>torch-sdpa-efficient 0.6253 (0.75) |  |  |  | liger 0.6299 (0.75) |
| llm_dec_b1_l2048 | rtx-4090 | 0.2429 | torch-eager 0.6366 (0.38) | **torch-compile 0.3161 (0.77)** | torch-compile-cg 0.3367 (0.72) | torch-compile-ro 0.3604 (0.67) | torch-sdpa 0.6806 (0.36)<br>torch-sdpa-efficient 0.9031 (0.27) |  |  |  | liger 0.9069 (0.27) |
| llm_dec_b1_l2048 | h100 | 0.3256 | torch-eager 0.4179 (0.78) | torch-compile 0.2219 (1.47) | **torch-compile-cg 0.2068 (1.57)** | torch-compile-ro 0.2245 (1.45) | torch-sdpa 0.4541 (0.72)<br>torch-sdpa-efficient 0.5851 (0.56) |  |  |  | liger 0.627 (0.52) |
| llm_dec_b1_l8192 | rtx-5080 | 0.3324 | torch-eager 0.4181 (0.79) | **torch-compile 0.3439 (0.97)** | torch-compile-cg 0.3586 (0.93) | torch-compile-ro 0.3471 (0.96) | torch-sdpa 1.287 (0.26)<br>torch-sdpa-efficient 1.927 (0.17) |  |  |  | liger 1.957 (0.17) |
| llm_dec_b1_l8192 | rtx-3090 | 0.4267 | torch-eager 0.6474 (0.66) | **torch-compile 0.4502 (0.95)** | torch-compile-cg 0.4603 (0.93) | torch-compile-ro 0.4939 (0.86) | torch-sdpa 1.73 (0.25)<br>torch-sdpa-efficient 2.967 (0.14) |  |  |  | liger 2.963 (0.14) |
| llm_dec_b1_l8192 | a100 | 0.5297 | torch-eager 0.5485 (0.97) | **torch-compile 0.3951 (1.34)** | torch-compile-cg 0.3989 (1.33) | torch-compile-ro 0.4584 (1.16) | torch-sdpa 1.033 (0.51)<br>torch-sdpa-efficient 1.65 (0.32) |  |  |  | liger 1.654 (0.32) |
| llm_dec_b1_l8192 | rtx-4090 | 0.3452 | torch-eager 0.7098 (0.49) | **torch-compile 0.3747 (0.92)** | torch-compile-cg 0.396 (0.87) | torch-compile-ro 0.4603 (0.75) | torch-sdpa 1.399 (0.25)<br>torch-sdpa-efficient 2.178 (0.16) |  |  |  | liger 2.18 (0.16) |
| llm_dec_b1_l8192 | h100 | 0.3661 | torch-eager 0.6705 (0.55) | **torch-compile 0.3272 (1.12)** | torch-compile-cg 0.3358 (1.09) | torch-compile-ro 0.5082 (0.72) | torch-sdpa 0.817 (0.45)<br>torch-sdpa-efficient 1.402 (0.26) |  |  |  | liger 1.428 (0.26) |
| llm_dec_b32_l2048 | rtx-5080 | 1.033 | torch-eager 2.672 (0.39) | **torch-compile 2.262 (0.46)** | torch-compile-cg 2.281 (0.45) | torch-compile-ro 2.367 (0.44) | torch-sdpa-efficient 7.965 (0.13)<br>torch-sdpa 8.797 (0.12) |  |  |  | liger 8.023 (0.13) |
| llm_dec_b32_l2048 | rtx-3090 | 1.213 | torch-eager 2.84 (0.43) | **torch-compile 2.658 (0.46)** | torch-compile-cg 2.669 (0.45) | torch-compile-ro 2.674 (0.45) | torch-sdpa 8.973 (0.14)<br>torch-sdpa-efficient 10.52 (0.12) |  |  |  | liger 10.43 (0.12) |
| llm_dec_b32_l2048 | a100 | 0.8597 | torch-eager 2.001 (0.43) | torch-compile 1.844 (0.47) | **torch-compile-cg 1.835 (0.47)** | torch-compile-ro 1.863 (0.46) | torch-sdpa-efficient 5.807 (0.15)<br>torch-sdpa 6.104 (0.14) |  |  |  | liger 5.743 (0.15) |
| llm_dec_b32_l2048 | rtx-4090 | 1.127 | torch-eager 2.59 (0.43) | **torch-compile 2.329 (0.48)** | torch-compile-cg 2.344 (0.48) | torch-compile-ro 2.35 (0.48) | torch-sdpa-efficient 6.221 (0.18)<br>torch-sdpa 8.127 (0.14) |  |  |  | liger 6.249 (0.18) |
| llm_dec_b32_l2048 | h100 | 0.5586 | torch-eager 1.845 (0.30) | torch-compile 1.429 (0.39) | **torch-compile-cg 1.417 (0.39)** | torch-compile-ro 1.558 (0.36) | torch-sdpa 5.269 (0.11)<br>torch-sdpa-efficient 5.417 (0.10) |  |  |  | liger 5.346 (0.10) |
| llm_dec_b32_l8192 | rtx-5080 | 3.515 | torch-eager 12.25 (0.29) | **torch-compile 10.27 (0.34)** | torch-compile-cg 10.57 (0.33) | torch-compile-ro 10.93 (0.32) | torch-sdpa-efficient 30.69 (0.11) |  |  |  | liger 32.71 (0.11) |
| llm_dec_b32_l8192 | rtx-3090 | 4.278 | torch-eager 11.99 (0.36) | **torch-compile 10.85 (0.39)** | torch-compile-cg 10.87 (0.39) | torch-compile-ro 11.23 (0.38) | torch-sdpa 41.07 (0.10)<br>torch-sdpa-efficient 47.22 (0.09) |  |  |  | liger 46.65 (0.09) |
| llm_dec_b32_l8192 | a100 | 2.534 | torch-eager 6.985 (0.36) | torch-compile 6.558 (0.39) | **torch-compile-cg 6.513 (0.39)** | torch-compile-ro 6.552 (0.39) | torch-sdpa-efficient 22.45 (0.11)<br>torch-sdpa 23.75 (0.11) |  |  |  | liger 22.26 (0.11) |
| llm_dec_b32_l8192 | rtx-4090 | 3.76 | torch-eager 11.09 (0.34) | **torch-compile 9.407 (0.40)** | torch-compile-cg 9.424 (0.40) | torch-compile-ro 9.511 (0.40) | torch-sdpa-efficient 22.99 (0.16)<br>torch-sdpa 30.67 (0.12) |  |  |  | liger 23.01 (0.16) |
| llm_dec_b32_l8192 | h100 | 1.48 | torch-eager 6.579 (0.22) | torch-compile 5.567 (0.27) | **torch-compile-cg 5.389 (0.27)** | torch-compile-ro 5.732 (0.26) | torch-sdpa 20 (0.07)<br>torch-sdpa-efficient 20.97 (0.07) |  |  |  | liger 20.8 (0.07) |
| llm_dec_b8_l2048 | rtx-5080 | 0.4103 | torch-eager 0.8128 (0.50) | torch-compile 0.7077 (0.58) | **torch-compile-cg 0.6816 (0.60)** | torch-compile-ro 0.7592 (0.54) | torch-sdpa-efficient 2.455 (0.17)<br>torch-sdpa 2.737 (0.15) |  |  |  | liger 2.456 (0.17) |
| llm_dec_b8_l2048 | rtx-3090 | 0.9218 | torch-eager 1.055 (0.87) | **torch-compile 0.8917 (1.03)** | torch-compile-cg 0.9098 (1.01) | torch-compile-ro 0.9396 (0.98) | torch-sdpa 2.599 (0.35)<br>torch-sdpa-efficient 3.307 (0.28) |  |  |  | liger 3.342 (0.28) |
| llm_dec_b8_l2048 | a100 | 0.3983 | torch-eager 0.8004 (0.50) | **torch-compile 0.6182 (0.64)** | torch-compile-cg 0.6257 (0.64) | torch-compile-ro 0.6921 (0.58) | torch-sdpa-efficient 1.791 (0.22)<br>torch-sdpa 1.817 (0.22) |  |  |  | liger 1.786 (0.22) |
| llm_dec_b8_l2048 | rtx-4090 | 0.5001 | torch-eager 0.9788 (0.51) | **torch-compile 0.7461 (0.67)** | torch-compile-cg 0.7681 (0.65) | torch-compile-ro 0.7966 (0.63) | torch-sdpa-efficient 1.931 (0.26)<br>torch-sdpa 2.411 (0.21) |  |  |  | liger 1.985 (0.25) |
| llm_dec_b8_l2048 | h100 | 0.2183 | torch-eager 0.7 (0.31) | **torch-compile 0.4996 (0.44)** | torch-compile-cg 0.5252 (0.42) | torch-compile-ro 0.5748 (0.38) | torch-sdpa 1.467 (0.15)<br>torch-sdpa-efficient 1.589 (0.14) |  |  |  | liger 1.605 (0.14) |
| llm_dec_b8_l8192 | rtx-5080 | 1.069 | torch-eager 3.252 (0.33) | torch-compile 2.889 (0.37) | **torch-compile-cg 2.856 (0.37)** | torch-compile-ro 3.005 (0.36) | torch-sdpa-efficient 8.772 (0.12)<br>torch-sdpa 8.805 (0.12) |  |  |  | liger 8.743 (0.12) |
| llm_dec_b8_l8192 | rtx-3090 | 1.702 | torch-eager 3.284 (0.52) | **torch-compile 2.925 (0.58)** | torch-compile-cg 2.931 (0.58) | torch-compile-ro 3.042 (0.56) | torch-sdpa 10.61 (0.16)<br>torch-sdpa-efficient 13.38 (0.13) |  |  |  | liger 13.41 (0.13) |
| llm_dec_b8_l8192 | a100 | 0.8105 | torch-eager 2.054 (0.39) | **torch-compile 1.807 (0.45)** | torch-compile-cg 1.818 (0.45) | torch-compile-ro 1.895 (0.43) | torch-sdpa 6.162 (0.13)<br>torch-sdpa-efficient 6.264 (0.13) |  |  |  | liger 6.218 (0.13) |
| llm_dec_b8_l8192 | rtx-4090 | 1.128 | torch-eager 3.098 (0.36) | **torch-compile 2.56 (0.44)** | torch-compile-cg 2.572 (0.44) | torch-compile-ro 2.581 (0.44) | torch-sdpa-efficient 6.131 (0.18)<br>torch-sdpa 8.038 (0.14) |  |  |  | liger 6.331 (0.18) |
| llm_dec_b8_l8192 | h100 | 0.5075 | torch-eager 1.734 (0.29) | torch-compile 1.455 (0.35) | **torch-compile-cg 1.437 (0.35)** | torch-compile-ro 1.486 (0.34) | torch-sdpa 5.28 (0.10)<br>torch-sdpa-efficient 5.418 (0.09) |  |  |  | liger 5.491 (0.09) |
| llm_pre_t2048 | rtx-5080 | 7.222 | torch-eager 13.47 (0.54) | torch-compile 8.44 (0.86) | torch-compile-cg 8.43 (0.86) | torch-compile-ro 9.445 (0.76) | torch-sdpa-efficient 7.298 (0.99)<br>torch-sdpa 13.87 (0.52) |  |  | triton 9.437 (0.77) | **liger 7.219 (1.00)** |
| llm_pre_t2048 | rtx-3090 | 14.05 | torch-eager 17.3 (0.81) | torch-compile 12.55 (1.12) | torch-compile-cg 12.5 (1.12) | torch-compile-ro 12.76 (1.10) | torch-sdpa-efficient 11.75 (1.20)<br>torch-sdpa 17.24 (0.81) |  |  | triton 16.25 (0.86) | **liger 11.48 (1.22)** |
| llm_pre_t2048 | a100 | 15.14 | torch-eager 17.73 (0.85) | torch-compile 14.59 (1.04) | torch-compile-cg 14.65 (1.03) | torch-compile-ro 14.97 (1.01) | torch-sdpa-efficient 12.13 (1.25)<br>torch-sdpa 17.93 (0.84) |  |  | triton 16.29 (0.93) | **liger 12.1 (1.25)** |
| llm_pre_t2048 | rtx-4090 | 6.305 | torch-eager 10.08 (0.63) | torch-compile 6.028 (1.05) | torch-compile-cg 6.032 (1.05) | torch-compile-ro 6.106 (1.03) | torch-sdpa-efficient 5.181 (1.22)<br>torch-sdpa 9.76 (0.65) |  |  | triton 6.786 (0.93) | **liger 5.064 (1.25)** |
| llm_pre_t2048 | h100 | 8.564 | torch-eager 10.32 (0.83) | torch-compile 8.505 (1.01) | torch-compile-cg 8.471 (1.01) | torch-compile-ro 8.645 (0.99) | torch-sdpa-efficient 8.46 (1.01)<br>torch-sdpa 10.51 (0.81) |  |  | triton 11.02 (0.78) | **liger 6.565 (1.30)** |
| llm_pre_t8192 | rtx-5080 | 89.55 | torch-eager 166.3 (0.54) | torch-compile 85.01 (1.05) | torch-compile-cg 84.67 (1.06) | torch-compile-ro 101 (0.89) | torch-sdpa-efficient 53.6 (1.67) |  |  | triton 84.32 (1.06) | **liger 52.91 (1.69)** |
| llm_pre_t8192 | rtx-3090 | 129.9 | torch-eager 174.9 (0.74) | torch-compile 109.6 (1.19) | torch-compile-cg 110.1 (1.18) | torch-compile-ro 112.9 (1.15) | torch-sdpa-efficient 86.31 (1.50)<br>torch-sdpa 175.3 (0.74) |  |  | triton 157.2 (0.83) | **liger 85.87 (1.51)** |
| llm_pre_t8192 | a100 | 132.3 | torch-eager 164.1 (0.81) | torch-compile 116.1 (1.14) | torch-compile-cg 116.2 (1.14) | torch-compile-ro 119 (1.11) | torch-sdpa-efficient 61.6 (2.15)<br>torch-sdpa 160.4 (0.82) |  |  | triton 121.2 (1.09) | **liger 61.06 (2.17)** |
| llm_pre_t8192 | rtx-4090 | 67.87 | torch-eager 118 (0.58) | torch-compile 55.96 (1.21) | torch-compile-cg 55.89 (1.21) | torch-compile-ro 57.06 (1.19) | torch-sdpa-efficient 37.51 (1.81)<br>torch-sdpa 119.3 (0.57) |  |  | triton 60.47 (1.12) | **liger 36.84 (1.84)** |
| llm_pre_t8192 | h100 | 71.33 | torch-eager 104.5 (0.68) | torch-compile 70.2 (1.02) | torch-compile-cg 69.5 (1.03) | torch-compile-ro 70.23 (1.02) | torch-sdpa-efficient 41.36 (1.72)<br>torch-sdpa 107 (0.67) |  |  | triton 109.4 (0.65) | **liger 39.46 (1.81)** |
| llm_blkpre_t2048_tf32 | rtx-5080 | 19.9 | torch-eager-tf32 24.23 (0.82) | torch-compile-tf32 18.6 (1.07) | torch-compile-cg-tf32 18.6 (1.07) | torch-compile-ro-tf32 19.55 (1.02) | torch-sdpa-tf32 23.88 (0.83) |  |  | **triton-tf32 18.07 (1.10)** | liger-tf32 19.13 (1.04) |
| llm_blkpre_t2048_tf32 | rtx-3090 | 33.35 | torch-eager-tf32 34.71 (0.96) | torch-compile-tf32 29.64 (1.13) | torch-compile-cg-tf32 29.73 (1.12) | torch-compile-ro-tf32 29.53 (1.13) | torch-sdpa-tf32 34.92 (0.96) |  |  | **triton-tf32 27.78 (1.20)** | liger-tf32 29.37 (1.14) |
| llm_blkpre_t2048_tf32 | a100 | 16.37 | torch-eager-tf32 13.44 (1.22) | torch-compile-tf32 10.51 (1.56) | torch-compile-cg-tf32 10.48 (1.56) | torch-compile-ro-tf32 10.53 (1.55) | torch-sdpa-tf32 13.72 (1.19) |  |  | **triton-tf32 9.807 (1.67)** | liger-tf32 10.42 (1.57) |
| llm_blkpre_t2048_tf32 | rtx-4090 | 16.54 | torch-eager-tf32 17.63 (0.94) | torch-compile-tf32 13.28 (1.25) | torch-compile-cg-tf32 13.27 (1.25) | torch-compile-ro-tf32 13.45 (1.23) | torch-sdpa-tf32 17.41 (0.95) |  |  | **triton-tf32 13.24 (1.25)** | liger-tf32 13.34 (1.24) |
| llm_blkpre_t2048_tf32 | h100 | 10.38 | torch-eager-tf32 9.396 (1.11) | torch-compile-tf32 7.278 (1.43) | torch-compile-cg-tf32 7.214 (1.44) | torch-compile-ro-tf32 7.182 (1.45) | torch-sdpa-tf32 9.141 (1.14) |  |  | triton-tf32 6.838 (1.52) | **liger-tf32 6.315 (1.64)** |
| llm_pre_t2048_tf32 | rtx-5080 | 6.072 | torch-eager-tf32 10.81 (0.56) | torch-compile-tf32 6.122 (0.99) | torch-compile-cg-tf32 6.127 (0.99) | torch-compile-ro-tf32 7.583 (0.80) | torch-sdpa-tf32 10.55 (0.58) |  |  | **triton-tf32 4.543 (1.34)** | liger-tf32 5.931 (1.02) |
| llm_pre_t2048_tf32 | rtx-3090 | 11.74 | torch-eager-tf32 14.03 (0.84) | torch-compile-tf32 9.03 (1.30) | torch-compile-cg-tf32 9.007 (1.30) | torch-compile-ro-tf32 9.416 (1.25) | torch-sdpa-tf32 14.2 (0.83) |  |  | **triton-tf32 7.568 (1.55)** | liger-tf32 9.448 (1.24) |
| llm_pre_t2048_tf32 | a100 | 7.419 | torch-eager-tf32 6.454 (1.15) | torch-compile-tf32 3.689 (2.01) | torch-compile-cg-tf32 3.665 (2.02) | torch-compile-ro-tf32 3.713 (2.00) | torch-sdpa-tf32 6.726 (1.10) |  |  | **triton-tf32 2.829 (2.62)** | liger-tf32 3.57 (2.08) |
| llm_pre_t2048_tf32 | rtx-4090 | 5.669 | torch-eager-tf32 8.753 (0.65) | torch-compile-tf32 4.78 (1.19) | torch-compile-cg-tf32 4.771 (1.19) | torch-compile-ro-tf32 4.87 (1.16) | torch-sdpa-tf32 8.563 (0.66) |  |  | **triton-tf32 3.17 (1.79)** | liger-tf32 4.063 (1.40) |
| llm_pre_t2048_tf32 | h100 | 4.453 | torch-eager-tf32 4.514 (0.99) | torch-compile-tf32 2.852 (1.56) | torch-compile-cg-tf32 2.655 (1.68) | torch-compile-ro-tf32 2.765 (1.61) | torch-sdpa-tf32 4.484 (0.99) |  |  | **triton-tf32 2.188 (2.04)** | liger-tf32 2.8 (1.59) |
| llm_blkdec_b1_l4096_bf16w | rtx-5080 | 0.6369 | torch-bf16 0.646 (0.99) | **torch-bf16-compile 0.5556 (1.15)** | torch-bf16-compile-cg 0.5613 (1.13) | torch-bf16-compile-ro 0.9771 (0.65) | torch-bf16-cudnn 0.6846 (0.93)<br>torch-bf16-efficient 1.014 (0.63) | fa2-bf16 0.6303 (1.01) | flashinfer-bf16 0.7693 (0.83) |  | liger-bf16 0.6587 (0.97) |
| llm_blkdec_b1_l4096_bf16w | rtx-3090 | 1.135 | torch-bf16 0.9324 (1.22) | **torch-bf16-compile 0.6744 (1.68)** | torch-bf16-compile-cg 0.6883 (1.65) | torch-bf16-compile-ro 1.17 (0.97) | torch-bf16-cudnn 0.9905 (1.15)<br>torch-bf16-efficient 1.23 (0.92) | fa2-bf16 0.93 (1.22) | flashinfer-bf16 1.088 (1.04) |  | liger-bf16 0.958 (1.18) |
| llm_blkdec_b1_l4096_bf16w | a100 | 2.062 | torch-bf16 0.5724 (3.60) | torch-bf16-compile 0.4915 (4.20) | **torch-bf16-compile-cg 0.4513 (4.57)** | torch-bf16-compile-ro 0.7862 (2.62) | torch-bf16-cudnn 0.5868 (3.51)<br>torch-bf16-efficient 0.8734 (2.36) | fa2-bf16 0.5756 (3.58) | flashinfer-bf16 0.6805 (3.03) |  | liger-bf16 0.5929 (3.48) |
| llm_blkdec_b1_l4096_bf16w | rtx-4090 | 0.6472 | torch-bf16 1.005 (0.64) | torch-bf16-compile 0.6304 (1.03) | **torch-bf16-compile-cg 0.6301 (1.03)** | torch-bf16-compile-ro 0.9945 (0.65) | torch-bf16-cudnn 1.064 (0.61)<br>torch-bf16-efficient 1.174 (0.55) | fa2-bf16 0.9892 (0.65) | flashinfer-bf16 1.159 (0.56) |  | liger-bf16 0.9825 (0.66) |
| llm_blkdec_b1_l4096_bf16w | h100 | 1.525 | torch-bf16 0.5607 (2.72) | torch-bf16-compile 0.3788 (4.03) | torch-bf16-compile-cg 0.3346 (4.56) | **torch-bf16-compile-ro 0.3021 (5.05)** | torch-bf16-cudnn 0.5992 (2.55)<br>torch-bf16-efficient 0.7378 (2.07) | fa2-bf16 0.5335 (2.86)<br>fa3-bf16 0.5535 (2.76) | flashinfer-bf16 0.6789 (2.25) |  | liger-bf16 0.6127 (2.49) |
| llm_blkpre_t2048_bf16w | rtx-5080 | 25.67 | torch-bf16 8.871 (2.89) | torch-bf16-compile 8.088 (3.17) | **torch-bf16-compile-cg 8.043 (3.19)** | torch-bf16-compile-ro 8.375 (3.06) | torch-bf16-cudnn 8.812 (2.91)<br>torch-bf16-efficient 9.355 (2.74) | fa2-bf16 8.872 (2.89) | flashinfer-bf16 8.886 (2.89) | triton-bf16 8.924 (2.88) | liger-bf16 8.509 (3.02) |
| llm_blkpre_t2048_bf16w | rtx-3090 | 43.99 | torch-bf16 14.33 (3.07) | torch-bf16-compile 13.68 (3.22) | torch-bf16-compile-cg 13.65 (3.22) | **torch-bf16-compile-ro 13.41 (3.28)** | torch-bf16-cudnn 14.29 (3.08)<br>torch-bf16-efficient 14.69 (3.00) | fa2-bf16 14.22 (3.09) | flashinfer-bf16 14.25 (3.09) | triton-bf16 14.35 (3.07) | liger-bf16 13.57 (3.24) |
| llm_blkpre_t2048_bf16w | a100 | 56.07 | torch-bf16 5.08 (11.04) | torch-bf16-compile 4.521 (12.40) | **torch-bf16-compile-cg 4.476 (12.53)** | torch-bf16-compile-ro 4.486 (12.50) | torch-bf16-cudnn 5.014 (11.18)<br>torch-bf16-efficient 5.264 (10.65) | fa2-bf16 5.017 (11.18) | flashinfer-bf16 5.755 (9.74) | triton-bf16 5.18 (10.82) | liger-bf16 4.626 (12.12) |
| llm_blkpre_t2048_bf16w | rtx-4090 | 18.39 | torch-bf16 6.536 (2.81) | torch-bf16-compile 5.676 (3.24) | **torch-bf16-compile-cg 5.594 (3.29)** | torch-bf16-compile-ro 6.091 (3.02) | torch-bf16-cudnn 6.49 (2.83)<br>torch-bf16-efficient 6.625 (2.78) | fa2-bf16 6.496 (2.83) | flashinfer-bf16 6.686 (2.75) | triton-bf16 6.678 (2.75) | liger-bf16 6.185 (2.97) |
| llm_blkpre_t2048_bf16w | h100 | 27.72 | torch-bf16 3.481 (7.96) | torch-bf16-compile 2.892 (9.58) | **torch-bf16-compile-cg 2.794 (9.92)** | torch-bf16-compile-ro 3.106 (8.92) | torch-bf16-cudnn 3.401 (8.15)<br>torch-bf16-efficient 3.698 (7.50) | fa3-bf16 3.429 (8.08)<br>fa2-bf16 3.527 (7.86) | flashinfer-bf16 3.488 (7.95) | triton-bf16 3.566 (7.77) | liger-bf16 3.239 (8.56) |
| llm_dec_b1_l2048_bf16w | rtx-5080 | 0.15 | torch-bf16 0.2109 (0.71) | **torch-bf16-compile 0.1455 (1.03)** | torch-bf16-compile-cg 0.152 (0.99) | torch-bf16-compile-ro 0.2949 (0.51) | torch-bf16-cudnn 0.2401 (0.62)<br>torch-bf16-efficient 0.3759 (0.40) | fa2-bf16 0.1884 (0.80) | flashinfer-bf16 0.3084 (0.49) |  | liger-bf16 0.2295 (0.65) |
| llm_dec_b1_l2048_bf16w | rtx-3090 | 0.2866 | torch-bf16 0.4284 (0.67) | **torch-bf16-compile 0.2338 (1.23)** | torch-bf16-compile-cg 0.2572 (1.11) | torch-bf16-compile-ro 0.4923 (0.58) | torch-bf16-cudnn 0.4712 (0.61)<br>torch-bf16-efficient 0.6061 (0.47) | fa2-bf16 0.4305 (0.67) | flashinfer-bf16 0.5996 (0.48) |  | liger-bf16 0.437 (0.66) |
| llm_dec_b1_l2048_bf16w | a100 | 0.4649 | torch-bf16 0.2411 (1.93) | torch-bf16-compile 0.1754 (2.65) | **torch-bf16-compile-cg 0.164 (2.84)** | torch-bf16-compile-ro 0.3262 (1.43) | torch-bf16-cudnn 0.2598 (1.79)<br>torch-bf16-efficient 0.4015 (1.16) | fa2-bf16 0.2397 (1.94) | flashinfer-bf16 0.3319 (1.40) |  | liger-bf16 0.2552 (1.82) |
| llm_dec_b1_l2048_bf16w | rtx-4090 | 0.1665 | torch-bf16 0.5154 (0.32) | torch-bf16-compile 0.2547 (0.65) | **torch-bf16-compile-cg 0.2425 (0.69)** | torch-bf16-compile-ro 0.3754 (0.44) | torch-bf16-cudnn 0.565 (0.29)<br>torch-bf16-efficient 0.6135 (0.27) | fa2-bf16 0.5044 (0.33) | flashinfer-bf16 0.6757 (0.25) |  | liger-bf16 0.487 (0.34) |
| llm_dec_b1_l2048_bf16w | h100 | 0.3256 | torch-bf16 0.3651 (0.89) | torch-bf16-compile 0.2258 (1.44) | torch-bf16-compile-cg 0.1502 (2.17) | **torch-bf16-compile-ro 0.1484 (2.19)** | torch-bf16-cudnn 0.3968 (0.82)<br>torch-bf16-efficient 0.4492 (0.72) | fa2-bf16 0.3299 (0.99)<br>fa3-bf16 0.3539 (0.92) | flashinfer-bf16 0.4735 (0.69) |  | liger-bf16 0.3848 (0.85) |
| llm_dec_b1_l8192_bf16w | rtx-5080 | 0.2454 | torch-bf16 0.2167 (1.13) | **torch-bf16-compile 0.1814 (1.35)** | torch-bf16-compile-cg 0.1926 (1.27) | torch-bf16-compile-ro 1.009 (0.24) | torch-bf16-cudnn 0.2591 (0.95)<br>torch-bf16-efficient 0.9416 (0.26) | fa2-bf16 0.2095 (1.17) | flashinfer-bf16 0.3736 (0.66) |  | liger-bf16 0.2495 (0.98) |
| llm_dec_b1_l8192_bf16w | rtx-3090 | 0.3873 | torch-bf16 0.4601 (0.84) | **torch-bf16-compile 0.2906 (1.33)** | torch-bf16-compile-cg 0.3082 (1.26) | torch-bf16-compile-ro 1.244 (0.31) | torch-bf16-cudnn 0.473 (0.82)<br>torch-bf16-efficient 1.118 (0.35) | fa2-bf16 0.4625 (0.84) | flashinfer-bf16 0.706 (0.55) |  | liger-bf16 0.4636 (0.84) |
| llm_dec_b1_l8192_bf16w | a100 | 0.5225 | torch-bf16 0.2598 (2.01) | **torch-bf16-compile 0.1961 (2.66)** | torch-bf16-compile-cg 0.199 (2.62) | torch-bf16-compile-ro 0.8473 (0.62) | torch-bf16-cudnn 0.2756 (1.90)<br>torch-bf16-efficient 0.8524 (0.61) | fa2-bf16 0.2654 (1.97) | flashinfer-bf16 0.4056 (1.29) |  | liger-bf16 0.273 (1.91) |
| llm_dec_b1_l8192_bf16w | rtx-4090 | 0.272 | torch-bf16 0.5172 (0.53) | **torch-bf16-compile 0.2525 (1.08)** | torch-bf16-compile-cg 0.2704 (1.01) | torch-bf16-compile-ro 1.091 (0.25) | torch-bf16-cudnn 0.5663 (0.48)<br>torch-bf16-efficient 0.9747 (0.28) | fa2-bf16 0.5052 (0.54) | flashinfer-bf16 0.738 (0.37) |  | liger-bf16 0.4875 (0.56) |
| llm_dec_b1_l8192_bf16w | h100 | 0.3666 | torch-bf16 0.3544 (1.03) | torch-bf16-compile 0.2256 (1.62) | torch-bf16-compile-cg 0.1622 (2.26) | **torch-bf16-compile-ro 0.16 (2.29)** | torch-bf16-cudnn 0.4029 (0.91)<br>torch-bf16-efficient 0.8113 (0.45) | fa2-bf16 0.3308 (1.11)<br>fa3-bf16 0.3536 (1.04) | flashinfer-bf16 0.5167 (0.71) |  | liger-bf16 0.3836 (0.96) |
| llm_dec_b32_l2048_bf16w | rtx-5080 | 0.9497 | torch-bf16 0.6217 (1.53) | **torch-bf16-compile 0.4622 (2.05)** | torch-bf16-compile-cg 0.4758 (2.00) | torch-bf16-compile-ro 7.622 (0.12) | torch-bf16-cudnn 0.5436 (1.75)<br>torch-bf16-efficient 3.808 (0.25) | fa2-bf16 0.5404 (1.76) | flashinfer-bf16 0.6066 (1.57) |  | liger-bf16 0.6592 (1.44) |
| llm_dec_b32_l2048_bf16w | rtx-3090 | 1.167 | torch-bf16 1.021 (1.14) | **torch-bf16-compile 0.7098 (1.64)** | torch-bf16-compile-cg 0.7293 (1.60) | torch-bf16-compile-ro 8.25 (0.14) | torch-bf16-cudnn 0.7449 (1.57)<br>torch-bf16-efficient 4.448 (0.26) | fa2-bf16 0.7953 (1.47) | flashinfer-bf16 0.8771 (1.33) |  | liger-bf16 1.021 (1.14) |
| llm_dec_b32_l2048_bf16w | a100 | 0.8662 | torch-bf16 0.4959 (1.75) | torch-bf16-compile 0.4738 (1.83) | torch-bf16-compile-cg 0.4831 (1.79) | torch-bf16-compile-ro 5.437 (0.16) | **torch-bf16-cudnn 0.4281 (2.02)**<br>torch-bf16-efficient 3.4 (0.25) | fa2-bf16 0.4568 (1.90) | flashinfer-bf16 0.4986 (1.74) |  | liger-bf16 0.4915 (1.76) |
| llm_dec_b32_l2048_bf16w | rtx-4090 | 1.022 | torch-bf16 0.7804 (1.31) | **torch-bf16-compile 0.5118 (2.00)** | torch-bf16-compile-cg 0.5238 (1.95) | torch-bf16-compile-ro 7.312 (0.14) | torch-bf16-cudnn 0.8036 (1.27)<br>torch-bf16-efficient 3.159 (0.32) | fa2-bf16 0.7789 (1.31) | flashinfer-bf16 0.9419 (1.08) |  | liger-bf16 0.7759 (1.32) |
| llm_dec_b32_l2048_bf16w | h100 | 0.5528 | torch-bf16 0.4196 (1.32) | torch-bf16-compile 0.2886 (1.92) | **torch-bf16-compile-cg 0.2671 (2.07)** | torch-bf16-compile-ro 0.2697 (2.05) | torch-bf16-cudnn 0.4469 (1.24)<br>torch-bf16-efficient 3.174 (0.17) | fa2-bf16 0.4372 (1.26)<br>fa3-bf16 0.482 (1.15) | flashinfer-bf16 0.5459 (1.01) |  | liger-bf16 0.4619 (1.20) |
| llm_dec_b32_l8192_bf16w | rtx-5080 | 3.438 | torch-bf16 1.817 (1.89) | torch-bf16-compile 1.478 (2.33) | torch-bf16-compile-cg 1.488 (2.31) | torch-bf16-compile-ro 29.8 (0.12) | **torch-bf16-cudnn 1.47 (2.34)**<br>torch-bf16-efficient 14.33 (0.24) | fa2-bf16 1.572 (2.19) | flashinfer-bf16 1.495 (2.30) |  | liger-bf16 1.883 (1.83) |
| llm_dec_b32_l8192_bf16w | rtx-3090 | 4.229 | torch-bf16 2.883 (1.47) | torch-bf16-compile 2.407 (1.76) | torch-bf16-compile-cg 2.441 (1.73) | torch-bf16-compile-ro 33.56 (0.13) | **torch-bf16-cudnn 1.697 (2.49)**<br>torch-bf16-efficient 17.31 (0.24) | fa2-bf16 1.889 (2.24) | flashinfer-bf16 1.783 (2.37) |  | liger-bf16 2.826 (1.50) |
| llm_dec_b32_l8192_bf16w | a100 | 2.538 | torch-bf16 1.208 (2.10) | torch-bf16-compile 1.663 (1.53) | torch-bf16-compile-cg 1.675 (1.52) | torch-bf16-compile-ro 21.42 (0.12) | **torch-bf16-cudnn 0.9763 (2.60)**<br>torch-bf16-efficient 12.99 (0.20) | fa2-bf16 1.098 (2.31) | flashinfer-bf16 1.061 (2.39) |  | liger-bf16 1.217 (2.09) |
| llm_dec_b32_l8192_bf16w | rtx-4090 | 3.659 | torch-bf16 1.629 (2.25) | **torch-bf16-compile 1.398 (2.62)** | torch-bf16-compile-cg 1.413 (2.59) | torch-bf16-compile-ro 28.52 (0.13) | torch-bf16-cudnn 1.645 (2.22)<br>torch-bf16-efficient 10.89 (0.34) | fa2-bf16 1.618 (2.26) | flashinfer-bf16 1.781 (2.05) |  | liger-bf16 1.627 (2.25) |
| llm_dec_b32_l8192_bf16w | h100 | 1.452 | torch-bf16 0.6945 (2.09) | torch-bf16-compile 0.5941 (2.44) | **torch-bf16-compile-cg 0.5752 (2.52)** | torch-bf16-compile-ro 0.5789 (2.51) | torch-bf16-cudnn 0.7243 (2.00)<br>torch-bf16-efficient 12.01 (0.12) | fa2-bf16 0.8319 (1.75)<br>fa3-bf16 0.9329 (1.56) | flashinfer-bf16 0.8352 (1.74) |  | liger-bf16 0.7424 (1.96) |
| llm_dec_b8_l2048_bf16w | rtx-5080 | 0.3143 | torch-bf16 0.2732 (1.15) | **torch-bf16-compile 0.2221 (1.41)** | torch-bf16-compile-cg 0.2313 (1.36) | torch-bf16-compile-ro 1.924 (0.16) | torch-bf16-cudnn 0.3026 (1.04)<br>torch-bf16-efficient 1.214 (0.26) | fa2-bf16 0.2672 (1.18) | flashinfer-bf16 0.4018 (0.78) |  | liger-bf16 0.3243 (0.97) |
| llm_dec_b8_l2048_bf16w | rtx-3090 | 0.7375 | torch-bf16 0.5228 (1.41) | **torch-bf16-compile 0.3564 (2.07)** | torch-bf16-compile-cg 0.3567 (2.07) | torch-bf16-compile-ro 2.267 (0.33) | torch-bf16-cudnn 0.5277 (1.40)<br>torch-bf16-efficient 1.479 (0.50) | fa2-bf16 0.5277 (1.40) | flashinfer-bf16 0.6705 (1.10) |  | liger-bf16 0.5378 (1.37) |
| llm_dec_b8_l2048_bf16w | a100 | 0.3486 | torch-bf16 0.2819 (1.24) | **torch-bf16-compile 0.2275 (1.53)** | torch-bf16-compile-cg 0.24 (1.45) | torch-bf16-compile-ro 1.487 (0.23) | torch-bf16-cudnn 0.2875 (1.21)<br>torch-bf16-efficient 1.09 (0.32) | fa2-bf16 0.2917 (1.19) | flashinfer-bf16 0.351 (0.99) |  | liger-bf16 0.3128 (1.11) |
| llm_dec_b8_l2048_bf16w | rtx-4090 | 0.4149 | torch-bf16 0.5613 (0.74) | **torch-bf16-compile 0.2652 (1.56)** | torch-bf16-compile-cg 0.3015 (1.38) | torch-bf16-compile-ro 1.964 (0.21) | torch-bf16-cudnn 0.5891 (0.70)<br>torch-bf16-efficient 1.184 (0.35) | fa2-bf16 0.5752 (0.72) | flashinfer-bf16 0.7332 (0.57) |  | liger-bf16 0.5641 (0.74) |
| llm_dec_b8_l2048_bf16w | h100 | 0.1858 | torch-bf16 0.3621 (0.51) | torch-bf16-compile 0.2517 (0.74) | torch-bf16-compile-cg 0.1691 (1.10) | **torch-bf16-compile-ro 0.1675 (1.11)** | torch-bf16-cudnn 0.3933 (0.47)<br>torch-bf16-efficient 0.99 (0.19) | fa2-bf16 0.3402 (0.55)<br>fa3-bf16 0.3615 (0.51) | flashinfer-bf16 0.4746 (0.39) |  | liger-bf16 0.4054 (0.46) |
| llm_dec_b8_l8192_bf16w | rtx-5080 | 1.012 | torch-bf16 0.5007 (2.02) | **torch-bf16-compile 0.4862 (2.08)** | torch-bf16-compile-cg 0.4997 (2.02) | torch-bf16-compile-ro 7.681 (0.13) | torch-bf16-cudnn 0.5256 (1.92)<br>torch-bf16-efficient 4.117 (0.25) | fa2-bf16 0.5109 (1.98) | flashinfer-bf16 0.6337 (1.60) |  | liger-bf16 0.5747 (1.76) |
| llm_dec_b8_l8192_bf16w | rtx-3090 | 1.517 | torch-bf16 0.7692 (1.97) | **torch-bf16-compile 0.7305 (2.08)** | torch-bf16-compile-cg 0.7494 (2.02) | torch-bf16-compile-ro 8.523 (0.18) | torch-bf16-cudnn 0.758 (2.00)<br>torch-bf16-efficient 4.828 (0.31) | fa2-bf16 0.75 (2.02) | flashinfer-bf16 0.8752 (1.73) |  | liger-bf16 0.7629 (1.99) |
| llm_dec_b8_l8192_bf16w | a100 | 0.7605 | **torch-bf16 0.4207 (1.81)** | torch-bf16-compile 0.5214 (1.46) | torch-bf16-compile-cg 0.5245 (1.45) | torch-bf16-compile-ro 5.464 (0.14) | torch-bf16-cudnn 0.43 (1.77)<br>torch-bf16-efficient 3.575 (0.21) | fa2-bf16 0.4306 (1.77) | flashinfer-bf16 0.4901 (1.55) |  | liger-bf16 0.4465 (1.70) |
| llm_dec_b8_l8192_bf16w | rtx-4090 | 1.049 | torch-bf16 0.7749 (1.35) | **torch-bf16-compile 0.5009 (2.09)** | torch-bf16-compile-cg 0.5242 (2.00) | torch-bf16-compile-ro 7.213 (0.15) | torch-bf16-cudnn 0.7995 (1.31)<br>torch-bf16-efficient 3.113 (0.34) | fa2-bf16 0.7885 (1.33) | flashinfer-bf16 0.9444 (1.11) |  | liger-bf16 0.7789 (1.35) |
| llm_dec_b8_l8192_bf16w | h100 | 0.4798 | torch-bf16 0.4285 (1.12) | torch-bf16-compile 0.2946 (1.63) | torch-bf16-compile-cg 0.2699 (1.78) | **torch-bf16-compile-ro 0.2691 (1.78)** | torch-bf16-cudnn 0.454 (1.06)<br>torch-bf16-efficient 3.181 (0.15) | fa2-bf16 0.442 (1.09)<br>fa3-bf16 0.47 (1.02) | flashinfer-bf16 0.5567 (0.86) |  | liger-bf16 0.4642 (1.03) |
| llm_pre_t2048_bf16w | rtx-5080 | 7.148 | torch-bf16 2.216 (3.23) | torch-bf16-compile 1.933 (3.70) | **torch-bf16-compile-cg 1.932 (3.70)** | torch-bf16-compile-ro 1.97 (3.63) | torch-bf16-cudnn 2.167 (3.30)<br>torch-bf16-efficient 2.722 (2.63) | fa2-bf16 2.211 (3.23) | flashinfer-bf16 2.226 (3.21) | triton-bf16 2.291 (3.12) | liger-bf16 2.072 (3.45) |
| llm_pre_t2048_bf16w | rtx-3090 | 13.93 | torch-bf16 3.703 (3.76) | **torch-bf16-compile 3.22 (4.33)** | torch-bf16-compile-cg 3.283 (4.24) | torch-bf16-compile-ro 3.226 (4.32) | torch-bf16-cudnn 3.728 (3.73)<br>torch-bf16-efficient 4.044 (3.44) | fa2-bf16 3.702 (3.76) | flashinfer-bf16 3.763 (3.70) | triton-bf16 3.846 (3.62) | liger-bf16 3.471 (4.01) |
| llm_pre_t2048_bf16w | a100 | 15.15 | torch-bf16 1.474 (10.28) | torch-bf16-compile 1.049 (14.44) | **torch-bf16-compile-cg 1.041 (14.54)** | torch-bf16-compile-ro 1.078 (14.05) | torch-bf16-cudnn 1.465 (10.34)<br>torch-bf16-efficient 1.674 (9.05) | fa2-bf16 1.427 (10.61) | flashinfer-bf16 1.445 (10.48) | triton-bf16 1.577 (9.60) | liger-bf16 1.26 (12.02) |
| llm_pre_t2048_bf16w | rtx-4090 | 6.23 | torch-bf16 1.654 (3.77) | torch-bf16-compile 1.3 (4.79) | **torch-bf16-compile-cg 1.294 (4.81)** | torch-bf16-compile-ro 1.463 (4.26) | torch-bf16-cudnn 1.7 (3.66)<br>torch-bf16-efficient 1.828 (3.41) | fa2-bf16 1.661 (3.75) | flashinfer-bf16 1.723 (3.61) | triton-bf16 1.742 (3.58) | liger-bf16 1.535 (4.06) |
| llm_pre_t2048_bf16w | h100 | 8.006 | torch-bf16 0.9656 (8.29) | torch-bf16-compile 0.6811 (11.76) | **torch-bf16-compile-cg 0.649 (12.34)** | torch-bf16-compile-ro 0.6498 (12.32) | torch-bf16-cudnn 0.9689 (8.26)<br>torch-bf16-efficient 1.346 (5.95) | fa3-bf16 1.212 (6.61)<br>fa2-bf16 1.502 (5.33) | flashinfer-bf16 1.389 (5.76) | triton-bf16 1.083 (7.40) | liger-bf16 0.9266 (8.64) |
| llm_blkdec_b1_l4096_bf16kv | rtx-5080 | 0.6157 | torch-bf16 0.646 (0.95) | **torch-bf16-compile 0.5556 (1.11)** | torch-bf16-compile-cg 0.5613 (1.10) | torch-bf16-compile-ro 0.9771 (0.63) | torch-bf16-cudnn 0.6846 (0.90)<br>torch-bf16-efficient 1.014 (0.61) | fa2-bf16 0.6303 (0.98) | flashinfer-bf16 0.7693 (0.80) |  | liger-bf16 0.6587 (0.93) |
| llm_blkdec_b1_l4096_bf16kv | rtx-3090 | 1.147 | torch-bf16 0.9324 (1.23) | **torch-bf16-compile 0.6744 (1.70)** | torch-bf16-compile-cg 0.6883 (1.67) | torch-bf16-compile-ro 1.17 (0.98) | torch-bf16-cudnn 0.9905 (1.16)<br>torch-bf16-efficient 1.23 (0.93) | fa2-bf16 0.93 (1.23) | flashinfer-bf16 1.088 (1.05) |  | liger-bf16 0.958 (1.20) |
| llm_blkdec_b1_l4096_bf16kv | a100 | 2.089 | torch-bf16 0.5724 (3.65) | torch-bf16-compile 0.4915 (4.25) | **torch-bf16-compile-cg 0.4513 (4.63)** | torch-bf16-compile-ro 0.7862 (2.66) | torch-bf16-cudnn 0.5868 (3.56)<br>torch-bf16-efficient 0.8734 (2.39) | fa2-bf16 0.5756 (3.63) | flashinfer-bf16 0.6805 (3.07) |  | liger-bf16 0.5929 (3.52) |
| llm_blkdec_b1_l4096_bf16kv | rtx-4090 | 0.6293 | torch-bf16 1.005 (0.63) | torch-bf16-compile 0.6304 (1.00) | **torch-bf16-compile-cg 0.6301 (1.00)** | torch-bf16-compile-ro 0.9945 (0.63) | torch-bf16-cudnn 1.064 (0.59)<br>torch-bf16-efficient 1.174 (0.54) | fa2-bf16 0.9892 (0.64) | flashinfer-bf16 1.159 (0.54) |  | liger-bf16 0.9825 (0.64) |
| llm_blkdec_b1_l4096_bf16kv | h100 | 1.536 | torch-bf16 0.5607 (2.74) | torch-bf16-compile 0.3788 (4.06) | torch-bf16-compile-cg 0.3346 (4.59) | **torch-bf16-compile-ro 0.3021 (5.09)** | torch-bf16-cudnn 0.5992 (2.56)<br>torch-bf16-efficient 0.7378 (2.08) | fa2-bf16 0.5335 (2.88)<br>fa3-bf16 0.5535 (2.78) | flashinfer-bf16 0.6789 (2.26) |  | liger-bf16 0.6127 (2.51) |
| llm_dec_b1_l2048_bf16kv | rtx-5080 | 0.1404 | torch-bf16 0.2109 (0.67) | **torch-bf16-compile 0.1455 (0.97)** | torch-bf16-compile-cg 0.152 (0.92) | torch-bf16-compile-ro 0.2949 (0.48) | torch-bf16-cudnn 0.2401 (0.58)<br>torch-bf16-efficient 0.3759 (0.37) | fa2-bf16 0.1884 (0.75) | flashinfer-bf16 0.3084 (0.46) |  | liger-bf16 0.2295 (0.61) |
| llm_dec_b1_l2048_bf16kv | rtx-3090 | 0.2921 | torch-bf16 0.4284 (0.68) | **torch-bf16-compile 0.2338 (1.25)** | torch-bf16-compile-cg 0.2572 (1.14) | torch-bf16-compile-ro 0.4923 (0.59) | torch-bf16-cudnn 0.4712 (0.62)<br>torch-bf16-efficient 0.6061 (0.48) | fa2-bf16 0.4305 (0.68) | flashinfer-bf16 0.5996 (0.49) |  | liger-bf16 0.437 (0.67) |
| llm_dec_b1_l2048_bf16kv | a100 | 0.4721 | torch-bf16 0.2411 (1.96) | torch-bf16-compile 0.1754 (2.69) | **torch-bf16-compile-cg 0.164 (2.88)** | torch-bf16-compile-ro 0.3262 (1.45) | torch-bf16-cudnn 0.2598 (1.82)<br>torch-bf16-efficient 0.4015 (1.18) | fa2-bf16 0.2397 (1.97) | flashinfer-bf16 0.3319 (1.42) |  | liger-bf16 0.2552 (1.85) |
| llm_dec_b1_l2048_bf16kv | rtx-4090 | 0.16 | torch-bf16 0.5154 (0.31) | torch-bf16-compile 0.2547 (0.63) | **torch-bf16-compile-cg 0.2425 (0.66)** | torch-bf16-compile-ro 0.3754 (0.43) | torch-bf16-cudnn 0.565 (0.28)<br>torch-bf16-efficient 0.6135 (0.26) | fa2-bf16 0.5044 (0.32) | flashinfer-bf16 0.6757 (0.24) |  | liger-bf16 0.487 (0.33) |
| llm_dec_b1_l2048_bf16kv | h100 | 0.3315 | torch-bf16 0.3651 (0.91) | torch-bf16-compile 0.2258 (1.47) | torch-bf16-compile-cg 0.1502 (2.21) | **torch-bf16-compile-ro 0.1484 (2.23)** | torch-bf16-cudnn 0.3968 (0.84)<br>torch-bf16-efficient 0.4492 (0.74) | fa2-bf16 0.3299 (1.00)<br>fa3-bf16 0.3539 (0.94) | flashinfer-bf16 0.4735 (0.70) |  | liger-bf16 0.3848 (0.86) |
| llm_dec_b1_l8192_bf16kv | rtx-5080 | 0.1976 | torch-bf16 0.2167 (0.91) | **torch-bf16-compile 0.1814 (1.09)** | torch-bf16-compile-cg 0.1926 (1.03) | torch-bf16-compile-ro 1.009 (0.20) | torch-bf16-cudnn 0.2591 (0.76)<br>torch-bf16-efficient 0.9416 (0.21) | fa2-bf16 0.2095 (0.94) | flashinfer-bf16 0.3736 (0.53) |  | liger-bf16 0.2495 (0.79) |
| llm_dec_b1_l8192_bf16kv | rtx-3090 | 0.3717 | torch-bf16 0.4601 (0.81) | **torch-bf16-compile 0.2906 (1.28)** | torch-bf16-compile-cg 0.3082 (1.21) | torch-bf16-compile-ro 1.244 (0.30) | torch-bf16-cudnn 0.473 (0.79)<br>torch-bf16-efficient 1.118 (0.33) | fa2-bf16 0.4625 (0.80) | flashinfer-bf16 0.706 (0.53) |  | liger-bf16 0.4636 (0.80) |
| llm_dec_b1_l8192_bf16kv | a100 | 0.5599 | torch-bf16 0.2598 (2.15) | **torch-bf16-compile 0.1961 (2.86)** | torch-bf16-compile-cg 0.199 (2.81) | torch-bf16-compile-ro 0.8473 (0.66) | torch-bf16-cudnn 0.2756 (2.03)<br>torch-bf16-efficient 0.8524 (0.66) | fa2-bf16 0.2654 (2.11) | flashinfer-bf16 0.4056 (1.38) |  | liger-bf16 0.273 (2.05) |
| llm_dec_b1_l8192_bf16kv | rtx-4090 | 0.2217 | torch-bf16 0.5172 (0.43) | **torch-bf16-compile 0.2525 (0.88)** | torch-bf16-compile-cg 0.2704 (0.82) | torch-bf16-compile-ro 1.091 (0.20) | torch-bf16-cudnn 0.5663 (0.39)<br>torch-bf16-efficient 0.9747 (0.23) | fa2-bf16 0.5052 (0.44) | flashinfer-bf16 0.738 (0.30) |  | liger-bf16 0.4875 (0.45) |
| llm_dec_b1_l8192_bf16kv | h100 | 0.3999 | torch-bf16 0.3544 (1.13) | torch-bf16-compile 0.2256 (1.77) | torch-bf16-compile-cg 0.1622 (2.47) | **torch-bf16-compile-ro 0.16 (2.50)** | torch-bf16-cudnn 0.4029 (0.99)<br>torch-bf16-efficient 0.8113 (0.49) | fa2-bf16 0.3308 (1.21)<br>fa3-bf16 0.3536 (1.13) | flashinfer-bf16 0.5167 (0.77) |  | liger-bf16 0.3836 (1.04) |
| llm_dec_b32_l2048_bf16kv | rtx-5080 | 0.6535 | torch-bf16 0.6217 (1.05) | **torch-bf16-compile 0.4622 (1.41)** | torch-bf16-compile-cg 0.4758 (1.37) | torch-bf16-compile-ro 7.622 (0.09) | torch-bf16-cudnn 0.5436 (1.20)<br>torch-bf16-efficient 3.808 (0.17) | fa2-bf16 0.5404 (1.21) | flashinfer-bf16 0.6066 (1.08) |  | liger-bf16 0.6592 (0.99) |
| llm_dec_b32_l2048_bf16kv | rtx-3090 | 1.02 | torch-bf16 1.021 (1.00) | **torch-bf16-compile 0.7098 (1.44)** | torch-bf16-compile-cg 0.7293 (1.40) | torch-bf16-compile-ro 8.25 (0.12) | torch-bf16-cudnn 0.7449 (1.37)<br>torch-bf16-efficient 4.448 (0.23) | fa2-bf16 0.7953 (1.28) | flashinfer-bf16 0.8771 (1.16) |  | liger-bf16 1.021 (1.00) |
| llm_dec_b32_l2048_bf16kv | a100 | 1.151 | torch-bf16 0.4959 (2.32) | torch-bf16-compile 0.4738 (2.43) | torch-bf16-compile-cg 0.4831 (2.38) | torch-bf16-compile-ro 5.437 (0.21) | **torch-bf16-cudnn 0.4281 (2.69)**<br>torch-bf16-efficient 3.4 (0.34) | fa2-bf16 0.4568 (2.52) | flashinfer-bf16 0.4986 (2.31) |  | liger-bf16 0.4915 (2.34) |
| llm_dec_b32_l2048_bf16kv | rtx-4090 | 0.6686 | torch-bf16 0.7804 (0.86) | **torch-bf16-compile 0.5118 (1.31)** | torch-bf16-compile-cg 0.5238 (1.28) | torch-bf16-compile-ro 7.312 (0.09) | torch-bf16-cudnn 0.8036 (0.83)<br>torch-bf16-efficient 3.159 (0.21) | fa2-bf16 0.7789 (0.86) | flashinfer-bf16 0.9419 (0.71) |  | liger-bf16 0.7759 (0.86) |
| llm_dec_b32_l2048_bf16kv | h100 | 0.9015 | torch-bf16 0.4196 (2.15) | torch-bf16-compile 0.2886 (3.12) | **torch-bf16-compile-cg 0.2671 (3.37)** | torch-bf16-compile-ro 0.2697 (3.34) | torch-bf16-cudnn 0.4469 (2.02)<br>torch-bf16-efficient 3.174 (0.28) | fa2-bf16 0.4372 (2.06)<br>fa3-bf16 0.482 (1.87) | flashinfer-bf16 0.5459 (1.65) |  | liger-bf16 0.4619 (1.95) |
| llm_dec_b32_l8192_bf16kv | rtx-5080 | 2.415 | torch-bf16 1.817 (1.33) | torch-bf16-compile 1.478 (1.63) | torch-bf16-compile-cg 1.488 (1.62) | torch-bf16-compile-ro 29.8 (0.08) | **torch-bf16-cudnn 1.47 (1.64)**<br>torch-bf16-efficient 14.33 (0.17) | fa2-bf16 1.572 (1.54) | flashinfer-bf16 1.495 (1.62) |  | liger-bf16 1.883 (1.28) |
| llm_dec_b32_l8192_bf16kv | rtx-3090 | 3.452 | torch-bf16 2.883 (1.20) | torch-bf16-compile 2.407 (1.43) | torch-bf16-compile-cg 2.441 (1.41) | torch-bf16-compile-ro 33.56 (0.10) | **torch-bf16-cudnn 1.697 (2.03)**<br>torch-bf16-efficient 17.31 (0.20) | fa2-bf16 1.889 (1.83) | flashinfer-bf16 1.783 (1.94) |  | liger-bf16 2.826 (1.22) |
| llm_dec_b32_l8192_bf16kv | a100 | 3.603 | torch-bf16 1.208 (2.98) | torch-bf16-compile 1.663 (2.17) | torch-bf16-compile-cg 1.675 (2.15) | torch-bf16-compile-ro 21.42 (0.17) | **torch-bf16-cudnn 0.9763 (3.69)**<br>torch-bf16-efficient 12.99 (0.28) | fa2-bf16 1.098 (3.28) | flashinfer-bf16 1.061 (3.40) |  | liger-bf16 1.217 (2.96) |
| llm_dec_b32_l8192_bf16kv | rtx-4090 | 2.2 | torch-bf16 1.629 (1.35) | **torch-bf16-compile 1.398 (1.57)** | torch-bf16-compile-cg 1.413 (1.56) | torch-bf16-compile-ro 28.52 (0.08) | torch-bf16-cudnn 1.645 (1.34)<br>torch-bf16-efficient 10.89 (0.20) | fa2-bf16 1.618 (1.36) | flashinfer-bf16 1.781 (1.24) |  | liger-bf16 1.627 (1.35) |
| llm_dec_b32_l8192_bf16kv | h100 | 1.978 | torch-bf16 0.6945 (2.85) | torch-bf16-compile 0.5941 (3.33) | **torch-bf16-compile-cg 0.5752 (3.44)** | torch-bf16-compile-ro 0.5789 (3.42) | torch-bf16-cudnn 0.7243 (2.73)<br>torch-bf16-efficient 12.01 (0.16) | fa2-bf16 0.8319 (2.38)<br>fa3-bf16 0.9329 (2.12) | flashinfer-bf16 0.8352 (2.37) |  | liger-bf16 0.7424 (2.66) |
| llm_dec_b8_l2048_bf16kv | rtx-5080 | 0.2406 | torch-bf16 0.2732 (0.88) | **torch-bf16-compile 0.2221 (1.08)** | torch-bf16-compile-cg 0.2313 (1.04) | torch-bf16-compile-ro 1.924 (0.13) | torch-bf16-cudnn 0.3026 (0.80)<br>torch-bf16-efficient 1.214 (0.20) | fa2-bf16 0.2672 (0.90) | flashinfer-bf16 0.4018 (0.60) |  | liger-bf16 0.3243 (0.74) |
| llm_dec_b8_l2048_bf16kv | rtx-3090 | 0.7325 | torch-bf16 0.5228 (1.40) | **torch-bf16-compile 0.3564 (2.06)** | torch-bf16-compile-cg 0.3567 (2.05) | torch-bf16-compile-ro 2.267 (0.32) | torch-bf16-cudnn 0.5277 (1.39)<br>torch-bf16-efficient 1.479 (0.50) | fa2-bf16 0.5277 (1.39) | flashinfer-bf16 0.6705 (1.09) |  | liger-bf16 0.5378 (1.36) |
| llm_dec_b8_l2048_bf16kv | a100 | 0.5163 | torch-bf16 0.2819 (1.83) | **torch-bf16-compile 0.2275 (2.27)** | torch-bf16-compile-cg 0.24 (2.15) | torch-bf16-compile-ro 1.487 (0.35) | torch-bf16-cudnn 0.2875 (1.80)<br>torch-bf16-efficient 1.09 (0.47) | fa2-bf16 0.2917 (1.77) | flashinfer-bf16 0.351 (1.47) |  | liger-bf16 0.3128 (1.65) |
| llm_dec_b8_l2048_bf16kv | rtx-4090 | 0.3658 | torch-bf16 0.5613 (0.65) | **torch-bf16-compile 0.2652 (1.38)** | torch-bf16-compile-cg 0.3015 (1.21) | torch-bf16-compile-ro 1.964 (0.19) | torch-bf16-cudnn 0.5891 (0.62)<br>torch-bf16-efficient 1.184 (0.31) | fa2-bf16 0.5752 (0.64) | flashinfer-bf16 0.7332 (0.50) |  | liger-bf16 0.5641 (0.65) |
| llm_dec_b8_l2048_bf16kv | h100 | 0.2819 | torch-bf16 0.3621 (0.78) | torch-bf16-compile 0.2517 (1.12) | torch-bf16-compile-cg 0.1691 (1.67) | **torch-bf16-compile-ro 0.1675 (1.68)** | torch-bf16-cudnn 0.3933 (0.72)<br>torch-bf16-efficient 0.99 (0.28) | fa2-bf16 0.3402 (0.83)<br>fa3-bf16 0.3615 (0.78) | flashinfer-bf16 0.4746 (0.59) |  | liger-bf16 0.4054 (0.70) |
| llm_dec_b8_l8192_bf16kv | rtx-5080 | 0.7981 | torch-bf16 0.5007 (1.59) | **torch-bf16-compile 0.4862 (1.64)** | torch-bf16-compile-cg 0.4997 (1.60) | torch-bf16-compile-ro 7.681 (0.10) | torch-bf16-cudnn 0.5256 (1.52)<br>torch-bf16-efficient 4.117 (0.19) | fa2-bf16 0.5109 (1.56) | flashinfer-bf16 0.6337 (1.26) |  | liger-bf16 0.5747 (1.39) |
| llm_dec_b8_l8192_bf16kv | rtx-3090 | 1.328 | torch-bf16 0.7692 (1.73) | **torch-bf16-compile 0.7305 (1.82)** | torch-bf16-compile-cg 0.7494 (1.77) | torch-bf16-compile-ro 8.523 (0.16) | torch-bf16-cudnn 0.758 (1.75)<br>torch-bf16-efficient 4.828 (0.28) | fa2-bf16 0.75 (1.77) | flashinfer-bf16 0.8752 (1.52) |  | liger-bf16 0.7629 (1.74) |
| llm_dec_b8_l8192_bf16kv | a100 | 1.05 | **torch-bf16 0.4207 (2.50)** | torch-bf16-compile 0.5214 (2.01) | torch-bf16-compile-cg 0.5245 (2.00) | torch-bf16-compile-ro 5.464 (0.19) | torch-bf16-cudnn 0.43 (2.44)<br>torch-bf16-efficient 3.575 (0.29) | fa2-bf16 0.4306 (2.44) | flashinfer-bf16 0.4901 (2.14) |  | liger-bf16 0.4465 (2.35) |
| llm_dec_b8_l8192_bf16kv | rtx-4090 | 0.6857 | torch-bf16 0.7749 (0.88) | **torch-bf16-compile 0.5009 (1.37)** | torch-bf16-compile-cg 0.5242 (1.31) | torch-bf16-compile-ro 7.213 (0.10) | torch-bf16-cudnn 0.7995 (0.86)<br>torch-bf16-efficient 3.113 (0.22) | fa2-bf16 0.7885 (0.87) | flashinfer-bf16 0.9444 (0.73) |  | liger-bf16 0.7789 (0.88) |
| llm_dec_b8_l8192_bf16kv | h100 | 0.8377 | torch-bf16 0.4285 (1.96) | torch-bf16-compile 0.2946 (2.84) | torch-bf16-compile-cg 0.2699 (3.10) | **torch-bf16-compile-ro 0.2691 (3.11)** | torch-bf16-cudnn 0.454 (1.85)<br>torch-bf16-efficient 3.181 (0.26) | fa2-bf16 0.442 (1.90)<br>fa3-bf16 0.47 (1.78) | flashinfer-bf16 0.5567 (1.50) |  | liger-bf16 0.4642 (1.80) |
| llm_blkdec_b1_l4096_bf16 | rtx-5080 | 0.615 | torch-bf16 0.646 (0.95) | **torch-bf16-compile 0.5556 (1.11)** | torch-bf16-compile-cg 0.5613 (1.10) | torch-bf16-compile-ro 0.9771 (0.63) | torch-bf16-cudnn 0.6846 (0.90)<br>torch-bf16-efficient 1.014 (0.61) | fa2-bf16 0.6303 (0.98) | flashinfer-bf16 0.7693 (0.80) |  | liger-bf16 0.6587 (0.93) |
| llm_blkdec_b1_l4096_bf16 | rtx-3090 | 1.146 | torch-bf16 0.9324 (1.23) | **torch-bf16-compile 0.6744 (1.70)** | torch-bf16-compile-cg 0.6883 (1.66) | torch-bf16-compile-ro 1.17 (0.98) | torch-bf16-cudnn 0.9905 (1.16)<br>torch-bf16-efficient 1.23 (0.93) | fa2-bf16 0.93 (1.23) | flashinfer-bf16 1.088 (1.05) |  | liger-bf16 0.958 (1.20) |
| llm_blkdec_b1_l4096_bf16 | a100 | 2.09 | torch-bf16 0.5724 (3.65) | torch-bf16-compile 0.4915 (4.25) | **torch-bf16-compile-cg 0.4513 (4.63)** | torch-bf16-compile-ro 0.7862 (2.66) | torch-bf16-cudnn 0.5868 (3.56)<br>torch-bf16-efficient 0.8734 (2.39) | fa2-bf16 0.5756 (3.63) | flashinfer-bf16 0.6805 (3.07) |  | liger-bf16 0.5929 (3.52) |
| llm_blkdec_b1_l4096_bf16 | rtx-4090 | 0.6273 | torch-bf16 1.005 (0.62) | torch-bf16-compile 0.6304 (1.00) | **torch-bf16-compile-cg 0.6301 (1.00)** | torch-bf16-compile-ro 0.9945 (0.63) | torch-bf16-cudnn 1.064 (0.59)<br>torch-bf16-efficient 1.174 (0.53) | fa2-bf16 0.9892 (0.63) | flashinfer-bf16 1.159 (0.54) |  | liger-bf16 0.9825 (0.64) |
| llm_blkdec_b1_l4096_bf16 | h100 | 1.538 | torch-bf16 0.5607 (2.74) | torch-bf16-compile 0.3788 (4.06) | torch-bf16-compile-cg 0.3346 (4.60) | **torch-bf16-compile-ro 0.3021 (5.09)** | torch-bf16-cudnn 0.5992 (2.57)<br>torch-bf16-efficient 0.7378 (2.08) | fa2-bf16 0.5335 (2.88)<br>fa3-bf16 0.5535 (2.78) | flashinfer-bf16 0.6789 (2.26) |  | liger-bf16 0.6127 (2.51) |
| llm_blkpre_t2048_bf16 | rtx-5080 | 11.62 | torch-bf16 8.871 (1.31) | torch-bf16-compile 8.088 (1.44) | **torch-bf16-compile-cg 8.043 (1.44)** | torch-bf16-compile-ro 8.375 (1.39) | torch-bf16-cudnn 8.812 (1.32)<br>torch-bf16-efficient 9.355 (1.24) | fa2-bf16 8.872 (1.31) | flashinfer-bf16 8.886 (1.31) | triton-bf16 8.924 (1.30) | liger-bf16 8.509 (1.37) |
| llm_blkpre_t2048_bf16 | rtx-3090 | 20.72 | torch-bf16 14.33 (1.45) | torch-bf16-compile 13.68 (1.52) | torch-bf16-compile-cg 13.65 (1.52) | **torch-bf16-compile-ro 13.41 (1.55)** | torch-bf16-cudnn 14.29 (1.45)<br>torch-bf16-efficient 14.69 (1.41) | fa2-bf16 14.22 (1.46) | flashinfer-bf16 14.25 (1.45) | triton-bf16 14.35 (1.44) | liger-bf16 13.57 (1.53) |
| llm_blkpre_t2048_bf16 | a100 | 12.86 | torch-bf16 5.08 (2.53) | torch-bf16-compile 4.521 (2.84) | **torch-bf16-compile-cg 4.476 (2.87)** | torch-bf16-compile-ro 4.486 (2.87) | torch-bf16-cudnn 5.014 (2.56)<br>torch-bf16-efficient 5.264 (2.44) | fa2-bf16 5.017 (2.56) | flashinfer-bf16 5.755 (2.23) | triton-bf16 5.18 (2.48) | liger-bf16 4.626 (2.78) |
| llm_blkpre_t2048_bf16 | rtx-4090 | 10.26 | torch-bf16 6.536 (1.57) | torch-bf16-compile 5.676 (1.81) | **torch-bf16-compile-cg 5.594 (1.83)** | torch-bf16-compile-ro 6.091 (1.68) | torch-bf16-cudnn 6.49 (1.58)<br>torch-bf16-efficient 6.625 (1.55) | fa2-bf16 6.496 (1.58) | flashinfer-bf16 6.686 (1.53) | triton-bf16 6.678 (1.54) | liger-bf16 6.185 (1.66) |
| llm_blkpre_t2048_bf16 | h100 | 10.07 | torch-bf16 3.481 (2.89) | torch-bf16-compile 2.892 (3.48) | **torch-bf16-compile-cg 2.794 (3.61)** | torch-bf16-compile-ro 3.106 (3.24) | torch-bf16-cudnn 3.401 (2.96)<br>torch-bf16-efficient 3.698 (2.72) | fa3-bf16 3.429 (2.94)<br>fa2-bf16 3.527 (2.86) | flashinfer-bf16 3.488 (2.89) | triton-bf16 3.566 (2.82) | liger-bf16 3.239 (3.11) |
| llm_dec_b1_l2048_bf16 | rtx-5080 | 0.1401 | torch-bf16 0.2109 (0.66) | **torch-bf16-compile 0.1455 (0.96)** | torch-bf16-compile-cg 0.152 (0.92) | torch-bf16-compile-ro 0.2949 (0.48) | torch-bf16-cudnn 0.2401 (0.58)<br>torch-bf16-efficient 0.3759 (0.37) | fa2-bf16 0.1884 (0.74) | flashinfer-bf16 0.3084 (0.45) |  | liger-bf16 0.2295 (0.61) |
| llm_dec_b1_l2048_bf16 | rtx-3090 | 0.2923 | torch-bf16 0.4284 (0.68) | **torch-bf16-compile 0.2338 (1.25)** | torch-bf16-compile-cg 0.2572 (1.14) | torch-bf16-compile-ro 0.4923 (0.59) | torch-bf16-cudnn 0.4712 (0.62)<br>torch-bf16-efficient 0.6061 (0.48) | fa2-bf16 0.4305 (0.68) | flashinfer-bf16 0.5996 (0.49) |  | liger-bf16 0.437 (0.67) |
| llm_dec_b1_l2048_bf16 | a100 | 0.4718 | torch-bf16 0.2411 (1.96) | torch-bf16-compile 0.1754 (2.69) | **torch-bf16-compile-cg 0.164 (2.88)** | torch-bf16-compile-ro 0.3262 (1.45) | torch-bf16-cudnn 0.2598 (1.82)<br>torch-bf16-efficient 0.4015 (1.18) | fa2-bf16 0.2397 (1.97) | flashinfer-bf16 0.3319 (1.42) |  | liger-bf16 0.2552 (1.85) |
| llm_dec_b1_l2048_bf16 | rtx-4090 | 0.1602 | torch-bf16 0.5154 (0.31) | torch-bf16-compile 0.2547 (0.63) | **torch-bf16-compile-cg 0.2425 (0.66)** | torch-bf16-compile-ro 0.3754 (0.43) | torch-bf16-cudnn 0.565 (0.28)<br>torch-bf16-efficient 0.6135 (0.26) | fa2-bf16 0.5044 (0.32) | flashinfer-bf16 0.6757 (0.24) |  | liger-bf16 0.487 (0.33) |
| llm_dec_b1_l2048_bf16 | h100 | 0.3318 | torch-bf16 0.3651 (0.91) | torch-bf16-compile 0.2258 (1.47) | torch-bf16-compile-cg 0.1502 (2.21) | **torch-bf16-compile-ro 0.1484 (2.24)** | torch-bf16-cudnn 0.3968 (0.84)<br>torch-bf16-efficient 0.4492 (0.74) | fa2-bf16 0.3299 (1.01)<br>fa3-bf16 0.3539 (0.94) | flashinfer-bf16 0.4735 (0.70) |  | liger-bf16 0.3848 (0.86) |
| llm_dec_b1_l8192_bf16 | rtx-5080 | 0.1996 | torch-bf16 0.2167 (0.92) | **torch-bf16-compile 0.1814 (1.10)** | torch-bf16-compile-cg 0.1926 (1.04) | torch-bf16-compile-ro 1.009 (0.20) | torch-bf16-cudnn 0.2591 (0.77)<br>torch-bf16-efficient 0.9416 (0.21) | fa2-bf16 0.2095 (0.95) | flashinfer-bf16 0.3736 (0.53) |  | liger-bf16 0.2495 (0.80) |
| llm_dec_b1_l8192_bf16 | rtx-3090 | 0.3745 | torch-bf16 0.4601 (0.81) | **torch-bf16-compile 0.2906 (1.29)** | torch-bf16-compile-cg 0.3082 (1.22) | torch-bf16-compile-ro 1.244 (0.30) | torch-bf16-cudnn 0.473 (0.79)<br>torch-bf16-efficient 1.118 (0.33) | fa2-bf16 0.4625 (0.81) | flashinfer-bf16 0.706 (0.53) |  | liger-bf16 0.4636 (0.81) |
| llm_dec_b1_l8192_bf16 | a100 | 0.5595 | torch-bf16 0.2598 (2.15) | **torch-bf16-compile 0.1961 (2.85)** | torch-bf16-compile-cg 0.199 (2.81) | torch-bf16-compile-ro 0.8473 (0.66) | torch-bf16-cudnn 0.2756 (2.03)<br>torch-bf16-efficient 0.8524 (0.66) | fa2-bf16 0.2654 (2.11) | flashinfer-bf16 0.4056 (1.38) |  | liger-bf16 0.273 (2.05) |
| llm_dec_b1_l8192_bf16 | rtx-4090 | 0.2233 | torch-bf16 0.5172 (0.43) | **torch-bf16-compile 0.2525 (0.88)** | torch-bf16-compile-cg 0.2704 (0.83) | torch-bf16-compile-ro 1.091 (0.20) | torch-bf16-cudnn 0.5663 (0.39)<br>torch-bf16-efficient 0.9747 (0.23) | fa2-bf16 0.5052 (0.44) | flashinfer-bf16 0.738 (0.30) |  | liger-bf16 0.4875 (0.46) |
| llm_dec_b1_l8192_bf16 | h100 | 0.3965 | torch-bf16 0.3544 (1.12) | torch-bf16-compile 0.2256 (1.76) | torch-bf16-compile-cg 0.1622 (2.44) | **torch-bf16-compile-ro 0.16 (2.48)** | torch-bf16-cudnn 0.4029 (0.98)<br>torch-bf16-efficient 0.8113 (0.49) | fa2-bf16 0.3308 (1.20)<br>fa3-bf16 0.3536 (1.12) | flashinfer-bf16 0.5167 (0.77) |  | liger-bf16 0.3836 (1.03) |
| llm_dec_b32_l2048_bf16 | rtx-5080 | 0.6536 | torch-bf16 0.6217 (1.05) | **torch-bf16-compile 0.4622 (1.41)** | torch-bf16-compile-cg 0.4758 (1.37) | torch-bf16-compile-ro 7.622 (0.09) | torch-bf16-cudnn 0.5436 (1.20)<br>torch-bf16-efficient 3.808 (0.17) | fa2-bf16 0.5404 (1.21) | flashinfer-bf16 0.6066 (1.08) |  | liger-bf16 0.6592 (0.99) |
| llm_dec_b32_l2048_bf16 | rtx-3090 | 1.02 | torch-bf16 1.021 (1.00) | **torch-bf16-compile 0.7098 (1.44)** | torch-bf16-compile-cg 0.7293 (1.40) | torch-bf16-compile-ro 8.25 (0.12) | torch-bf16-cudnn 0.7449 (1.37)<br>torch-bf16-efficient 4.448 (0.23) | fa2-bf16 0.7953 (1.28) | flashinfer-bf16 0.8771 (1.16) |  | liger-bf16 1.021 (1.00) |
| llm_dec_b32_l2048_bf16 | a100 | 1.138 | torch-bf16 0.4959 (2.29) | torch-bf16-compile 0.4738 (2.40) | torch-bf16-compile-cg 0.4831 (2.35) | torch-bf16-compile-ro 5.437 (0.21) | **torch-bf16-cudnn 0.4281 (2.66)**<br>torch-bf16-efficient 3.4 (0.33) | fa2-bf16 0.4568 (2.49) | flashinfer-bf16 0.4986 (2.28) |  | liger-bf16 0.4915 (2.31) |
| llm_dec_b32_l2048_bf16 | rtx-4090 | 0.669 | torch-bf16 0.7804 (0.86) | **torch-bf16-compile 0.5118 (1.31)** | torch-bf16-compile-cg 0.5238 (1.28) | torch-bf16-compile-ro 7.312 (0.09) | torch-bf16-cudnn 0.8036 (0.83)<br>torch-bf16-efficient 3.159 (0.21) | fa2-bf16 0.7789 (0.86) | flashinfer-bf16 0.9419 (0.71) |  | liger-bf16 0.7759 (0.86) |
| llm_dec_b32_l2048_bf16 | h100 | 0.9381 | torch-bf16 0.4196 (2.24) | torch-bf16-compile 0.2886 (3.25) | **torch-bf16-compile-cg 0.2671 (3.51)** | torch-bf16-compile-ro 0.2697 (3.48) | torch-bf16-cudnn 0.4469 (2.10)<br>torch-bf16-efficient 3.174 (0.30) | fa2-bf16 0.4372 (2.15)<br>fa3-bf16 0.482 (1.95) | flashinfer-bf16 0.5459 (1.72) |  | liger-bf16 0.4619 (2.03) |
| llm_dec_b32_l8192_bf16 | rtx-5080 | 2.393 | torch-bf16 1.817 (1.32) | torch-bf16-compile 1.478 (1.62) | torch-bf16-compile-cg 1.488 (1.61) | torch-bf16-compile-ro 29.8 (0.08) | **torch-bf16-cudnn 1.47 (1.63)**<br>torch-bf16-efficient 14.33 (0.17) | fa2-bf16 1.572 (1.52) | flashinfer-bf16 1.495 (1.60) |  | liger-bf16 1.883 (1.27) |
| llm_dec_b32_l8192_bf16 | rtx-3090 | 3.457 | torch-bf16 2.883 (1.20) | torch-bf16-compile 2.407 (1.44) | torch-bf16-compile-cg 2.441 (1.42) | torch-bf16-compile-ro 33.56 (0.10) | **torch-bf16-cudnn 1.697 (2.04)**<br>torch-bf16-efficient 17.31 (0.20) | fa2-bf16 1.889 (1.83) | flashinfer-bf16 1.783 (1.94) |  | liger-bf16 2.826 (1.22) |
| llm_dec_b32_l8192_bf16 | a100 | 3.586 | torch-bf16 1.208 (2.97) | torch-bf16-compile 1.663 (2.16) | torch-bf16-compile-cg 1.675 (2.14) | torch-bf16-compile-ro 21.42 (0.17) | **torch-bf16-cudnn 0.9763 (3.67)**<br>torch-bf16-efficient 12.99 (0.28) | fa2-bf16 1.098 (3.27) | flashinfer-bf16 1.061 (3.38) |  | liger-bf16 1.217 (2.95) |
| llm_dec_b32_l8192_bf16 | rtx-4090 | 2.199 | torch-bf16 1.629 (1.35) | **torch-bf16-compile 1.398 (1.57)** | torch-bf16-compile-cg 1.413 (1.56) | torch-bf16-compile-ro 28.52 (0.08) | torch-bf16-cudnn 1.645 (1.34)<br>torch-bf16-efficient 10.89 (0.20) | fa2-bf16 1.618 (1.36) | flashinfer-bf16 1.781 (1.24) |  | liger-bf16 1.627 (1.35) |
| llm_dec_b32_l8192_bf16 | h100 | 2.016 | torch-bf16 0.6945 (2.90) | torch-bf16-compile 0.5941 (3.39) | **torch-bf16-compile-cg 0.5752 (3.50)** | torch-bf16-compile-ro 0.5789 (3.48) | torch-bf16-cudnn 0.7243 (2.78)<br>torch-bf16-efficient 12.01 (0.17) | fa2-bf16 0.8319 (2.42)<br>fa3-bf16 0.9329 (2.16) | flashinfer-bf16 0.8352 (2.41) |  | liger-bf16 0.7424 (2.72) |
| llm_dec_b8_l2048_bf16 | rtx-5080 | 0.2403 | torch-bf16 0.2732 (0.88) | **torch-bf16-compile 0.2221 (1.08)** | torch-bf16-compile-cg 0.2313 (1.04) | torch-bf16-compile-ro 1.924 (0.12) | torch-bf16-cudnn 0.3026 (0.79)<br>torch-bf16-efficient 1.214 (0.20) | fa2-bf16 0.2672 (0.90) | flashinfer-bf16 0.4018 (0.60) |  | liger-bf16 0.3243 (0.74) |
| llm_dec_b8_l2048_bf16 | rtx-3090 | 0.7329 | torch-bf16 0.5228 (1.40) | **torch-bf16-compile 0.3564 (2.06)** | torch-bf16-compile-cg 0.3567 (2.05) | torch-bf16-compile-ro 2.267 (0.32) | torch-bf16-cudnn 0.5277 (1.39)<br>torch-bf16-efficient 1.479 (0.50) | fa2-bf16 0.5277 (1.39) | flashinfer-bf16 0.6705 (1.09) |  | liger-bf16 0.5378 (1.36) |
| llm_dec_b8_l2048_bf16 | a100 | 0.5168 | torch-bf16 0.2819 (1.83) | **torch-bf16-compile 0.2275 (2.27)** | torch-bf16-compile-cg 0.24 (2.15) | torch-bf16-compile-ro 1.487 (0.35) | torch-bf16-cudnn 0.2875 (1.80)<br>torch-bf16-efficient 1.09 (0.47) | fa2-bf16 0.2917 (1.77) | flashinfer-bf16 0.351 (1.47) |  | liger-bf16 0.3128 (1.65) |
| llm_dec_b8_l2048_bf16 | rtx-4090 | 0.3664 | torch-bf16 0.5613 (0.65) | **torch-bf16-compile 0.2652 (1.38)** | torch-bf16-compile-cg 0.3015 (1.22) | torch-bf16-compile-ro 1.964 (0.19) | torch-bf16-cudnn 0.5891 (0.62)<br>torch-bf16-efficient 1.184 (0.31) | fa2-bf16 0.5752 (0.64) | flashinfer-bf16 0.7332 (0.50) |  | liger-bf16 0.5641 (0.65) |
| llm_dec_b8_l2048_bf16 | h100 | 0.2802 | torch-bf16 0.3621 (0.77) | torch-bf16-compile 0.2517 (1.11) | torch-bf16-compile-cg 0.1691 (1.66) | **torch-bf16-compile-ro 0.1675 (1.67)** | torch-bf16-cudnn 0.3933 (0.71)<br>torch-bf16-efficient 0.99 (0.28) | fa2-bf16 0.3402 (0.82)<br>fa3-bf16 0.3615 (0.78) | flashinfer-bf16 0.4746 (0.59) |  | liger-bf16 0.4054 (0.69) |
| llm_dec_b8_l8192_bf16 | rtx-5080 | 0.6996 | torch-bf16 0.5007 (1.40) | **torch-bf16-compile 0.4862 (1.44)** | torch-bf16-compile-cg 0.4997 (1.40) | torch-bf16-compile-ro 7.681 (0.09) | torch-bf16-cudnn 0.5256 (1.33)<br>torch-bf16-efficient 4.117 (0.17) | fa2-bf16 0.5109 (1.37) | flashinfer-bf16 0.6337 (1.10) |  | liger-bf16 0.5747 (1.22) |
| llm_dec_b8_l8192_bf16 | rtx-3090 | 1.328 | torch-bf16 0.7692 (1.73) | **torch-bf16-compile 0.7305 (1.82)** | torch-bf16-compile-cg 0.7494 (1.77) | torch-bf16-compile-ro 8.523 (0.16) | torch-bf16-cudnn 0.758 (1.75)<br>torch-bf16-efficient 4.828 (0.28) | fa2-bf16 0.75 (1.77) | flashinfer-bf16 0.8752 (1.52) |  | liger-bf16 0.7629 (1.74) |
| llm_dec_b8_l8192_bf16 | a100 | 1.051 | **torch-bf16 0.4207 (2.50)** | torch-bf16-compile 0.5214 (2.02) | torch-bf16-compile-cg 0.5245 (2.00) | torch-bf16-compile-ro 5.464 (0.19) | torch-bf16-cudnn 0.43 (2.44)<br>torch-bf16-efficient 3.575 (0.29) | fa2-bf16 0.4306 (2.44) | flashinfer-bf16 0.4901 (2.14) |  | liger-bf16 0.4465 (2.35) |
| llm_dec_b8_l8192_bf16 | rtx-4090 | 0.6864 | torch-bf16 0.7749 (0.89) | **torch-bf16-compile 0.5009 (1.37)** | torch-bf16-compile-cg 0.5242 (1.31) | torch-bf16-compile-ro 7.213 (0.10) | torch-bf16-cudnn 0.7995 (0.86)<br>torch-bf16-efficient 3.113 (0.22) | fa2-bf16 0.7885 (0.87) | flashinfer-bf16 0.9444 (0.73) |  | liger-bf16 0.7789 (0.88) |
| llm_dec_b8_l8192_bf16 | h100 | 0.8382 | torch-bf16 0.4285 (1.96) | torch-bf16-compile 0.2946 (2.84) | torch-bf16-compile-cg 0.2699 (3.11) | **torch-bf16-compile-ro 0.2691 (3.11)** | torch-bf16-cudnn 0.454 (1.85)<br>torch-bf16-efficient 3.181 (0.26) | fa2-bf16 0.442 (1.90)<br>fa3-bf16 0.47 (1.78) | flashinfer-bf16 0.5567 (1.51) |  | liger-bf16 0.4642 (1.81) |
| llm_pre_t2048_bf16 | rtx-5080 | 3.942 | torch-bf16 2.216 (1.78) | torch-bf16-compile 1.933 (2.04) | **torch-bf16-compile-cg 1.932 (2.04)** | torch-bf16-compile-ro 1.97 (2.00) | torch-bf16-cudnn 2.167 (1.82)<br>torch-bf16-efficient 2.722 (1.45) | fa2-bf16 2.211 (1.78) | flashinfer-bf16 2.226 (1.77) | triton-bf16 2.291 (1.72) | liger-bf16 2.072 (1.90) |
| llm_pre_t2048_bf16 | rtx-3090 | 9.37 | torch-bf16 3.703 (2.53) | **torch-bf16-compile 3.22 (2.91)** | torch-bf16-compile-cg 3.283 (2.85) | torch-bf16-compile-ro 3.226 (2.90) | torch-bf16-cudnn 3.728 (2.51)<br>torch-bf16-efficient 4.044 (2.32) | fa2-bf16 3.702 (2.53) | flashinfer-bf16 3.763 (2.49) | triton-bf16 3.846 (2.44) | liger-bf16 3.471 (2.70) |
| llm_pre_t2048_bf16 | a100 | 6.81 | torch-bf16 1.474 (4.62) | torch-bf16-compile 1.049 (6.49) | **torch-bf16-compile-cg 1.041 (6.54)** | torch-bf16-compile-ro 1.078 (6.32) | torch-bf16-cudnn 1.465 (4.65)<br>torch-bf16-efficient 1.674 (4.07) | fa2-bf16 1.427 (4.77) | flashinfer-bf16 1.445 (4.71) | triton-bf16 1.577 (4.32) | liger-bf16 1.26 (5.40) |
| llm_pre_t2048_bf16 | rtx-4090 | 4.595 | torch-bf16 1.654 (2.78) | torch-bf16-compile 1.3 (3.54) | **torch-bf16-compile-cg 1.294 (3.55)** | torch-bf16-compile-ro 1.463 (3.14) | torch-bf16-cudnn 1.7 (2.70)<br>torch-bf16-efficient 1.828 (2.51) | fa2-bf16 1.661 (2.77) | flashinfer-bf16 1.723 (2.67) | triton-bf16 1.742 (2.64) | liger-bf16 1.535 (2.99) |
| llm_pre_t2048_bf16 | h100 | 4.407 | torch-bf16 0.9656 (4.56) | torch-bf16-compile 0.6811 (6.47) | **torch-bf16-compile-cg 0.649 (6.79)** | torch-bf16-compile-ro 0.6498 (6.78) | torch-bf16-cudnn 0.9689 (4.55)<br>torch-bf16-efficient 1.346 (3.27) | fa3-bf16 1.212 (3.64)<br>fa2-bf16 1.502 (2.93) | flashinfer-bf16 1.389 (3.17) | triton-bf16 1.083 (4.07) | liger-bf16 0.9266 (4.76) |
| llm_blkdec_b1_l4096_f8w | rtx-5080 | 0.3717 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-5080 | 0.05575 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-5080 | 0.1998 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-5080 | 0.9476 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-5080 | 3.538 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-5080 | 0.2703 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-5080 | 0.986 |  |  |  |  |  |  |  |  |  |

### Conv / FIR / elementwise

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand CUDA |
|---|---|---:|---|---|---|---|---|---|
| conv_2048_5 | rtx-5080 | 0.06927 |  | **torch-eager 0.1516 (0.46)** | torch-compile 0.1516 (0.46) | torch-compile-cg 0.1516 (0.46) | torch-compile-ro 0.1516 (0.46) |  |
| conv_2048_5 | rtx-3090 | 0.09289 |  | torch-eager 0.1812 (0.51) | **torch-compile 0.1577 (0.59)** | torch-compile-cg 0.1577 (0.59) | torch-compile-ro 0.1577 (0.59) |  |
| conv_2048_5 | a100 | 0.07186 |  | torch-eager 0.1382 (0.52) | torch-compile 0.1382 (0.52) | **torch-compile-cg 0.1372 (0.52)** | torch-compile-ro 0.1372 (0.52) |  |
| conv_2048_5 | rtx-4090 | 0.05365 |  | torch-eager 0.0911 (0.59) | torch-compile 0.0911 (0.59) | **torch-compile-cg 0.0901 (0.60)** | torch-compile-ro 0.0901 (0.60) |  |
| conv_2048_5 | h100 | 0.04278 |  | **torch-eager 0.0972 (0.44)** | torch-compile 0.0975 (0.44) | torch-compile-cg 0.0978 (0.44) | torch-compile-ro 0.0977 (0.44) |  |
| convmc_8_64_56_3 | rtx-5080 | 0.08369 |  | **torch-eager 0.1096 (0.76)** | torch-compile 0.1628 (0.51) | torch-compile-cg 0.1577 (0.53) | torch-compile-ro 0.1577 (0.53) |  |
| convmc_8_64_56_3 | rtx-3090 | 0.127 |  | **torch-eager 0.0758 (1.68)** | torch-compile 0.1659 (0.77) | torch-compile-cg 0.1587 (0.80) | torch-compile-ro 0.1597 (0.80) |  |
| convmc_8_64_56_3 | a100 | 0.1958 |  | **torch-eager 0.087 (2.25)** | torch-compile 0.1833 (1.07) | torch-compile-cg 0.17 (1.15) | torch-compile-ro 0.1577 (1.24) |  |
| convmc_8_64_56_3 | rtx-4090 | 0.09697 |  | **torch-eager 0.0553 (1.75)** | torch-compile 0.0973 (1.00) | torch-compile-cg 0.0993 (0.98) | torch-compile-ro 0.1004 (0.97) |  |
| convmc_8_64_56_3 | h100 | 0.1196 |  | **torch-eager 0.0702 (1.70)** | torch-compile 0.1132 (1.06) | torch-compile-cg 0.1085 (1.10) | torch-compile-ro 0.1084 (1.10) |  |
| ew_4096 | rtx-5080 | 0.09179 | chain-eager 0.5673 (0.16) | torch-eager 0.5724 (0.16) | torch-compile 0.2437 (0.38) | torch-compile-cg 0.2454 (0.37) | torch-compile-ro 0.2437 (0.38) | **chain-fused 0.2437 (0.38)** |
| ew_4096 | rtx-3090 | 0.1039 | chain-eager 0.5622 (0.18) | torch-eager 0.5622 (0.18) | **torch-compile 0.2396 (0.43)** | torch-compile-cg 0.2396 (0.43) | torch-compile-ro 0.2396 (0.43) | chain-fused 0.2406 (0.43) |
| ew_4096 | a100 | 0.1182 | chain-eager 0.3666 (0.32) | torch-eager 0.3584 (0.33) | **torch-compile 0.1536 (0.77)** | torch-compile-cg 0.1536 (0.77) | torch-compile-ro 0.1556 (0.76) | chain-fused 0.1608 (0.74) |
| ew_4096 | rtx-4090 | 0.1044 | chain-eager 0.5171 (0.20) | torch-eager 0.5222 (0.20) | torch-compile 0.2263 (0.46) | torch-compile-cg 0.2263 (0.46) | torch-compile-ro 0.2253 (0.46) | **chain-fused 0.2253 (0.46)** |
| ew_4096 | h100 | 0.07908 | chain-eager 0.2075 (0.38) | torch-eager 0.168 (0.47) | torch-compile 0.0715 (1.11) | **torch-compile-cg 0.0712 (1.11)** | torch-compile-ro 0.0716 (1.10) | chain-fused 0.0828 (0.96) |
| ewconvew_2048_5 | rtx-5080 | 0.08131 |  | torch-eager 0.2079 (0.39) | torch-compile 0.1874 (0.43) | **torch-compile-cg 0.1864 (0.44)** | torch-compile-ro 0.1884 (0.43) |  |
| ewconvew_2048_5 | rtx-3090 | 0.1128 |  | torch-eager 0.2826 (0.40) | torch-compile 0.2417 (0.47) | **torch-compile-cg 0.2386 (0.47)** | torch-compile-ro 0.2621 (0.43) |  |
| ewconvew_2048_5 | a100 | 0.08807 |  | torch-eager 0.1956 (0.45) | torch-compile 0.1772 (0.50) | torch-compile-cg 0.1741 (0.51) | **torch-compile-ro 0.1731 (0.51)** |  |
| ewconvew_2048_5 | rtx-4090 | 0.06402 |  | torch-eager 0.1413 (0.45) | torch-compile 0.128 (0.50) | **torch-compile-cg 0.126 (0.51)** | torch-compile-ro 0.127 (0.50) |  |
| ewconvew_2048_5 | h100 | 0.05446 |  | torch-eager 0.1296 (0.42) | torch-compile 0.1172 (0.46) | **torch-compile-cg 0.1159 (0.47)** | torch-compile-ro 0.1174 (0.46) |  |
| fir_4194304_16 | rtx-5080 | 0.04817 |  | **torch-eager 0.1044 (0.46)** | torch-compile 0.1044 (0.46) | torch-compile-cg 0.1044 (0.46) | torch-compile-ro 0.1044 (0.46) |  |
| fir_4194304_16 | rtx-3090 | 0.0616 |  | **torch-eager 0.1055 (0.58)** | torch-compile 0.1055 (0.58) | torch-compile-cg 0.1055 (0.58) | torch-compile-ro 0.1055 (0.58) |  |
| fir_4194304_16 | a100 | 0.04998 |  | **torch-eager 0.085 (0.59)** | torch-compile 0.086 (0.58) | torch-compile-cg 0.085 (0.59) | torch-compile-ro 0.085 (0.59) |  |
| fir_4194304_16 | rtx-4090 | 0.03793 |  | torch-eager 0.0627 (0.60) | **torch-compile 0.0625 (0.61)** | torch-compile-cg 0.0625 (0.61) | torch-compile-ro 0.0625 (0.61) |  |
| fir_4194304_16 | h100 | 0.03144 |  | **torch-eager 0.0596 (0.53)** | torch-compile 0.0596 (0.53) | torch-compile-cg 0.0597 (0.53) | torch-compile-ro 0.0596 (0.53) |  |
| firew_rowsum_2048x2048_16 | rtx-5080 | 0.05537 |  | torch-eager 0.128 (0.43) | **torch-compile 0.1085 (0.51)** | torch-compile-cg 0.1085 (0.51) | torch-compile-ro 0.1085 (0.51) |  |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.08745 |  | torch-eager 0.213 (0.41) | **torch-compile 0.126 (0.69)** | torch-compile-cg 0.1372 (0.64) | torch-compile-ro 0.1382 (0.63) |  |
| firew_rowsum_2048x2048_16 | a100 | 0.06182 |  | torch-eager 0.1311 (0.47) | torch-compile 0.0963 (0.64) | torch-compile-cg 0.0952 (0.65) | **torch-compile-ro 0.0942 (0.66)** |  |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.04532 |  | torch-eager 0.0901 (0.50) | torch-compile 0.0676 (0.67) | **torch-compile-cg 0.0655 (0.69)** | torch-compile-ro 0.0655 (0.69) |  |
| firew_rowsum_2048x2048_16 | h100 | 0.03754 |  | torch-eager 0.0857 (0.44) | torch-compile 0.0655 (0.57) | torch-compile-cg 0.0645 (0.58) | **torch-compile-ro 0.0644 (0.58)** |  |
| convmc_8_64_56_3_tf32 | rtx-5080 | 0.08357 |  | **torch-eager-tf32 0.0748 (1.12)** | torch-compile-tf32 0.0952 (0.88) | torch-compile-cg-tf32 0.0922 (0.91) | torch-compile-ro-tf32 0.0901 (0.93) |  |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1274 |  | **torch-eager-tf32 0.085 (1.50)** | torch-compile-tf32 0.1444 (0.88) | torch-compile-cg-tf32 0.1393 (0.91) | torch-compile-ro-tf32 0.1393 (0.91) |  |
| convmc_8_64_56_3_tf32 | a100 | 0.1889 |  | **torch-eager-tf32 0.0481 (3.93)** | torch-compile-tf32 0.0819 (2.31) | torch-compile-cg-tf32 0.0768 (2.46) | torch-compile-ro-tf32 0.0748 (2.53) |  |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.09696 |  | **torch-eager-tf32 0.042 (2.31)** | torch-compile-tf32 0.0788 (1.23) | torch-compile-cg-tf32 0.0778 (1.25) | torch-compile-ro-tf32 0.0799 (1.21) |  |
| convmc_8_64_56_3_tf32 | h100 | 0.1192 |  | **torch-eager-tf32 0.0361 (3.30)** | torch-compile-tf32 0.0593 (2.01) | torch-compile-cg-tf32 0.0439 (2.71) | torch-compile-ro-tf32 0.0426 (2.80) |  |

### Fan-in/out, loops, CPU+GPU

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead |
|---|---|---:|---|---|---|---|---|
| diamond_2048 | rtx-5080 | 1.45 | chain-lt 1.383 (1.05)<br>chain-streams 1.401 (1.04)<br>chain-eager 1.407 (1.03)<br>chain-cutlass 1.653 (0.88) | torch-eager 1.382 (1.05) | torch-compile 1.37 (1.06) | **torch-compile-cg 1.368 (1.06)** | torch-compile-ro 1.391 (1.04) |
| diamond_2048 | rtx-3090 | 2.358 | **chain-streams 2.354 (1.00)**<br>chain-lt 2.39 (0.99)<br>chain-eager 2.403 (0.98)<br>chain-cutlass 2.746 (0.86) | torch-eager 2.38 (0.99) | torch-compile 2.366 (1.00) | torch-compile-cg 2.415 (0.98) | torch-compile-ro 2.432 (0.97) |
| diamond_2048 | a100 | 3.247 | chain-lt 2.949 (1.10)<br>chain-streams 2.953 (1.10)<br>chain-eager 2.982 (1.09)<br>chain-cutlass 3.465 (0.94) | torch-eager 2.977 (1.09) | torch-compile 3.22 (1.01) | **torch-compile-cg 2.945 (1.10)** | torch-compile-ro 2.96 (1.10) |
| diamond_2048 | rtx-4090 | 0.9067 | chain-cutlass 0.9206 (0.98)<br>chain-lt 0.9277 (0.98)<br>chain-streams 0.939 (0.97)<br>chain-eager 0.9441 (0.96) | torch-eager 0.9411 (0.96) | torch-compile 0.9196 (0.99) | **torch-compile-cg 0.9175 (0.99)** | torch-compile-ro 0.9318 (0.97) |
| diamond_2048 | h100 | 1.437 | chain-lt 1.441 (1.00)<br>chain-streams 1.451 (0.99)<br>chain-eager 1.454 (0.99)<br>chain-cutlass 1.638 (0.88) | **torch-eager 1.271 (1.13)** | torch-compile 1.363 (1.05) | torch-compile-cg 1.338 (1.07) | torch-compile-ro 1.346 (1.07) |
| fanout_2048 | rtx-5080 | 0.9854 | **chain-streams 0.9133 (1.08)**<br>chain-eager 0.9257 (1.06) | torch-eager 0.937 (1.05) | torch-compile 0.9277 (1.06) | torch-compile-cg 0.9236 (1.07) | torch-compile-ro 0.9233 (1.07) |
| fanout_2048 | rtx-3090 | 1.616 | chain-streams 1.585 (1.02)<br>chain-eager 1.618 (1.00) | torch-eager 1.572 (1.03) | **torch-compile 1.451 (1.11)** | torch-compile-cg 1.479 (1.09) | torch-compile-ro 1.626 (0.99) |
| fanout_2048 | a100 | 2.255 | **chain-streams 1.966 (1.15)**<br>chain-eager 1.987 (1.14) | torch-eager 1.995 (1.13) | torch-compile 2.047 (1.10) | torch-compile-cg 2.048 (1.10) | torch-compile-ro 1.977 (1.14) |
| fanout_2048 | rtx-4090 | 0.6126 | **chain-streams 0.6226 (0.98)**<br>chain-eager 0.6298 (0.97) | torch-eager 0.6431 (0.95) | torch-compile 0.64 (0.96) | torch-compile-cg 0.639 (0.96) | torch-compile-ro 0.6236 (0.98) |
| fanout_2048 | h100 | 0.9683 | chain-streams 0.8499 (1.14)<br>chain-eager 0.855 (1.13) | torch-eager 0.8591 (1.13) | torch-compile 0.8459 (1.14) | **torch-compile-cg 0.8452 (1.15)** | torch-compile-ro 0.8492 (1.14) |
| indep_2048 | rtx-5080 | 1.001 | **chain-lt 0.9155 (1.09)**<br>chain-streams 0.9359 (1.07)<br>chain-eager 0.94 (1.07)<br>chain-cutlass 1.094 (0.92) | torch-eager 0.9277 (1.08) | torch-compile 0.9175 (1.09) | torch-compile-cg 0.9175 (1.09) | torch-compile-ro 0.9274 (1.08) |
| indep_2048 | rtx-3090 | 1.571 | chain-streams 1.565 (1.00)<br>chain-lt 1.606 (0.98)<br>chain-eager 1.613 (0.97)<br>chain-cutlass 1.818 (0.86) | torch-eager 1.588 (0.99) | **torch-compile 1.494 (1.05)** | torch-compile-cg 1.521 (1.03) | torch-compile-ro 1.511 (1.04) |
| indep_2048 | a100 | 2.246 | chain-lt 1.968 (1.14)<br>chain-streams 1.971 (1.14)<br>chain-eager 2.001 (1.12)<br>chain-cutlass 2.313 (0.97) | torch-eager 1.997 (1.12) | torch-compile 1.967 (1.14) | **torch-compile-cg 1.964 (1.14)** | torch-compile-ro 1.981 (1.13) |
| indep_2048 | rtx-4090 | 0.6077 | **chain-cutlass 0.6144 (0.99)**<br>chain-lt 0.6175 (0.98)<br>chain-streams 0.6244 (0.97)<br>chain-eager 0.6298 (0.96) | torch-eager 0.6318 (0.96) | torch-compile 0.6195 (0.98) | torch-compile-cg 0.6164 (0.99) | torch-compile-ro 0.6246 (0.97) |
| indep_2048 | h100 | 0.9889 | chain-lt 0.898 (1.10)<br>chain-streams 0.9073 (1.09)<br>chain-eager 0.9107 (1.09)<br>chain-cutlass 1.035 (0.96) | torch-eager 0.8558 (1.16) | torch-compile 0.845 (1.17) | **torch-compile-cg 0.8423 (1.17)** | torch-compile-ro 0.8536 (1.16) |
| indep_512 | rtx-5080 | 0.04073 | chain-streams 0.0365 (1.12)<br>chain-eager 0.0451 (0.90)<br>chain-lt 0.0451 (0.90)<br>chain-cutlass 0.1065 (0.38) | torch-eager 0.0451 (0.90) | **torch-compile 0.0328 (1.24)** | torch-compile-cg 0.0328 (1.24) | torch-compile-ro 0.0451 (0.90) |
| indep_512 | rtx-3090 | 0.05629 | chain-streams 0.0471 (1.20)<br>chain-lt 0.0471 (1.20)<br>chain-eager 0.0492 (1.14)<br>chain-cutlass 0.1085 (0.52) | torch-eager 0.0563 (1.00) | **torch-compile 0.0461 (1.22)** | torch-compile-cg 0.0461 (1.22) | torch-compile-ro 0.0532 (1.06) |
| indep_512 | a100 | 0.0757 | **chain-streams 0.0573 (1.32)**<br>chain-lt 0.0655 (1.16)<br>chain-eager 0.0666 (1.14)<br>chain-cutlass 0.2079 (0.36) | torch-eager 0.0666 (1.14) | torch-compile 0.0788 (0.96) | torch-compile-cg 0.0778 (0.97) | torch-compile-ro 0.0809 (0.94) |
| indep_512 | rtx-4090 | 0.05487 | chain-lt 0.0358 (1.53)<br>chain-streams 0.0377 (1.46)<br>chain-eager 0.0389 (1.41)<br>chain-cutlass 0.0871 (0.63) | torch-eager 0.0399 (1.38) | torch-compile 0.0266 (2.06) | **torch-compile-cg 0.0256 (2.14)** | torch-compile-ro 0.0389 (1.41) |
| indep_512 | h100 | 0.05621 | chain-streams 0.0315 (1.78)<br>chain-lt 0.0335 (1.68)<br>chain-eager 0.0431 (1.30)<br>chain-cutlass 0.1205 (0.47) | torch-eager 0.0391 (1.44) | **torch-compile 0.0269 (2.09)** | torch-compile-cg 0.0269 (2.09) | torch-compile-ro 0.0329 (1.71) |
| loop_4096 | rtx-5080 | 28.24 | **chain-lt 28.53 (0.99)**<br>chain-cutlass 28.84 (0.98)<br>chain-eager 30.29 (0.93) | torch-eager 29.27 (0.96) | torch-compile 31.66 (0.89) | torch-compile-cg 32.39 (0.87) | torch-compile-ro 29.8 (0.95) |
| loop_4096 | rtx-3090 | 42.15 | **chain-lt 45.42 (0.93)**<br>chain-cutlass 47.08 (0.90)<br>chain-eager 48.37 (0.87) | torch-eager 45.7 (0.92) | torch-compile 45.88 (0.92) | torch-compile-cg 45.9 (0.92) | torch-compile-ro 46.08 (0.91) |
| loop_4096 | a100 | 62.96 | chain-eager 59.09 (1.07)<br>chain-lt 60.43 (1.04)<br>chain-cutlass 61.04 (1.03) | torch-eager 59.02 (1.07) | torch-compile 61.18 (1.03) | torch-compile-cg 61.18 (1.03) | **torch-compile-ro 59 (1.07)** |
| loop_4096 | rtx-4090 | 18.41 | **chain-cutlass 19.2 (0.96)**<br>chain-lt 19.49 (0.94)<br>chain-eager 20.48 (0.90) | torch-eager 19.72 (0.93) | torch-compile 19.52 (0.94) | torch-compile-cg 19.51 (0.94) | torch-compile-ro 19.37 (0.95) |
| loop_4096 | h100 | 26.98 | **chain-lt 31.13 (0.87)**<br>chain-eager 31.51 (0.86)<br>chain-cutlass 36.27 (0.74) | torch-eager 31.26 (0.86) | torch-compile 31.95 (0.84) | torch-compile-cg 31.49 (0.86) | torch-compile-ro 31.46 (0.86) |
| loopmm_1024_8 | rtx-5080 | 0.6523 |  | torch-eager 0.641 (1.02) | torch-compile 0.5898 (1.11) | **torch-compile-cg 0.5878 (1.11)** | torch-compile-ro 0.6246 (1.04) |
| loopmm_1024_8 | rtx-3090 | 1.082 |  | **torch-eager 0.898 (1.20)** | torch-compile 0.9072 (1.19) | torch-compile-cg 0.9011 (1.20) | torch-compile-ro 0.9011 (1.20) |
| loopmm_1024_8 | a100 | 1.434 |  | **torch-eager 1.044 (1.37)** | torch-compile 1.354 (1.06) | torch-compile-cg 1.345 (1.07) | torch-compile-ro 1.345 (1.07) |
| loopmm_1024_8 | rtx-4090 | 0.4096 |  | torch-eager 0.3584 (1.14) | torch-compile 0.3471 (1.18) | **torch-compile-cg 0.342 (1.20)** | torch-compile-ro 0.342 (1.20) |
| loopmm_1024_8 | h100 | 0.624 |  | torch-eager 0.6521 (0.96) | torch-compile 0.4985 (1.25) | **torch-compile-cg 0.4938 (1.26)** | torch-compile-ro 0.5109 (1.22) |
| rt_mlp2_1024 | rtx-5080 | 2.794 |  | torch-eager 3.653 (0.76) | torch-compile 2.544 (1.10) | **torch-compile-cg 2.518 (1.11)** | torch-compile-ro 2.609 (1.07) |
| rt_mlp2_1024 | rtx-3090 | 3.964 |  | torch-eager 6.357 (0.62) | torch-compile 4.183 (0.95) | **torch-compile-cg 4.125 (0.96)** | torch-compile-ro 4.26 (0.93) |
| rt_mlp2_1024 | a100 | 6.835 |  | torch-eager 5.454 (1.25) | torch-compile 5.629 (1.21) | **torch-compile-cg 4.299 (1.59)** | torch-compile-ro 4.33 (1.58) |
| rt_mlp2_1024 | rtx-4090 | 1.624 |  | torch-eager 2.471 (0.66) | torch-compile 1.499 (1.08) | **torch-compile-cg 1.448 (1.12)** | torch-compile-ro 1.504 (1.08) |
| rt_mlp2_1024 | h100 | 2.612 |  | torch-eager 3.554 (0.73) | torch-compile 2.489 (1.05) | **torch-compile-cg 2.435 (1.07)** | torch-compile-ro 2.572 (1.02) |
| rt_mlp2_2048 | rtx-5080 | 16.24 |  | torch-eager 19.86 (0.82) | torch-compile 17.97 (0.90) | torch-compile-cg 17.88 (0.91) | **torch-compile-ro 15.66 (1.04)** |
| rt_mlp2_2048 | rtx-3090 | 25.51 |  | torch-eager 35.17 (0.73) | torch-compile 26.68 (0.96) | **torch-compile-cg 26.61 (0.96)** | torch-compile-ro 26.81 (0.95) |
| rt_mlp2_2048 | a100 | 37.17 |  | torch-eager 38.31 (0.97) | torch-compile 31.85 (1.17) | **torch-compile-cg 31.75 (1.17)** | torch-compile-ro 31.91 (1.16) |
| rt_mlp2_2048 | rtx-4090 | 9.742 |  | torch-eager 13.94 (0.70) | **torch-compile 10.4 (0.94)** | torch-compile-cg 10.42 (0.93) | torch-compile-ro 10.4 (0.94) |
| rt_mlp2_2048 | h100 | 15.16 |  | torch-eager 19.67 (0.77) | torch-compile 16.46 (0.92) | torch-compile-cg 16.4 (0.92) | **torch-compile-ro 16.33 (0.93)** |
| twin_2048 | rtx-5080 | 0.941 |  | torch-eager 0.9482 (0.99) | torch-compile 0.9236 (1.02) | **torch-compile-cg 0.9185 (1.02)** | torch-compile-ro 0.9192 (1.02) |
| twin_2048 | rtx-3090 | 1.583 |  | torch-eager 1.538 (1.03) | torch-compile 1.578 (1.00) | torch-compile-cg 1.528 (1.04) | **torch-compile-ro 1.422 (1.11)** |
| twin_2048 | a100 | 2.02 |  | torch-eager 1.964 (1.03) | torch-compile 1.964 (1.03) | torch-compile-cg 1.962 (1.03) | **torch-compile-ro 1.961 (1.03)** |
| twin_2048 | rtx-4090 | 0.603 |  | torch-eager 0.6175 (0.98) | torch-compile 0.6135 (0.98) | **torch-compile-cg 0.6093 (0.99)** | torch-compile-ro 0.6113 (0.99) |
| twin_2048 | h100 | 0.9591 |  | torch-eager 1.008 (0.95) | **torch-compile 0.8394 (1.14)** | torch-compile-cg 0.8414 (1.14) | torch-compile-ro 0.8397 (1.14) |

## Command

```
python3 benches/local/matrix.py --skip 'llm_kv[xn]_' \
  rtx-5080=.claude/raw-evidence/cross-arch/3a8e654f/rtx-5080/resS/results-rtx-5080.txt@3a8e654f,.claude/raw-evidence/cross-arch/3a8e654f/rtx-5080/resM/results-rtx-5080.txt@3a8e654f,.claude/raw-evidence/cross-arch/3a8e654f/rtx-5080/resS2/results-rtx-5080.txt@3a8e654f,.claude/raw-evidence/cross-arch/3a8e654f/rtx-5080/resM2/results-rtx-5080.txt@3a8e654f,.claude/raw-evidence/local-rtx-5080/integration/resS-final/results-rtx-5080.txt@integration,.claude/raw-evidence/local-rtx-5080/integration/resM-final/results-rtx-5080.txt@integration,.claude/raw-evidence/local-rtx-5080/integration/keep-ab/results-rtx-5080.txt@integration \
  rtx-3090=.claude/raw-evidence/cross-arch/3a8e654f/rtx-3090/results-rtx-3090.txt@3a8e654f \
  a100=.claude/raw-evidence/cross-arch/3a8e654f/a100/results-a100.txt@3a8e654f \
  rtx-4090=.claude/raw-evidence/cross-arch/3a8e654f/rtx-4090/results-rtx-4090.txt@3a8e654f \
  h100=.claude/raw-evidence/cross-arch/3a8e654f/h100/results-h100.txt@3a8e654f
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
