# Mapal vs the fastest competitor at the same precision

Built 2026-11-25 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations), `_bf16kv` rows (weights and KV cache bf16, f32 compute) and `_bf16` rows (bf16 storage, `: bf16 | f32` products) vs the bf16 legs (bf16 throughout); a bf16-storage row without bf16 legs of its own takes its twin's (`_bf16w`, else the f32 program: the same bf16 computation); `_f8w` rows (fp8 weights, f32 compute) vs fp8 legs, of which none is installed: `(no leg)`, their `_bf16w` twin's bf16 leg is a reference only. The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch's pick, `-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials' matmul / softmax / fused attention, autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. **Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang as servers. Every part ran the competitor legs (FlashAttention-3 on sm_90a only; `unavailable` = the part's run reported the package missing or its build capped).

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | 39cd2c87, 5fcb401b | 5fcb401b/rtx-5080/resM@5fcb401b: 73; 5fcb401b/rtx-5080/resS@5fcb401b: 40; confirmation-fixes/changed-resM@39cd2c87: 11; confirmation-fixes/changed-resS@39cd2c87: 3 |
| rtx-3090 | 5fcb401b | 5fcb401b/rtx-3090@5fcb401b: 127 |
| a100 | 5fcb401b | 5fcb401b/a100@5fcb401b: 126 |
| rtx-4090 | 5fcb401b | 5fcb401b/rtx-4090@5fcb401b: 120 |
| h100 | 5fcb401b | 5fcb401b/h100@5fcb401b: 127 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 1.045 (47) | 1.099 (47) | 1.288 (46) | 1.104 (47) | 1.623 (47) |
| MLP chains | 1.054 (6) | 1.119 (6) | 1.537 (6) | 1.155 (6) | 1.617 (6) |
| Attention / softmax / long rows | 0.855 (14) | 0.827 (14) | 1.029 (14) | 0.600 (14) | 0.935 (14) |
| LLM decode / prefill / block (f32, tf32, bf16 storage) | 1.035 (37) | 1.004 (37) | 1.232 (37) | 0.908 (37) | 1.215 (37) |
| Conv / FIR / elementwise | 0.529 (7) | 0.833 (7) | 0.978 (7) | 0.739 (7) | 0.942 (7) |
| Fan-in/out, loops, CPU+GPU | 1.032 (9) | 0.977 (9) | 1.494 (9) | 1.102 (9) | 1.201 (9) |
| all f32 rows | 0.869 (66) | 0.893 (66) | 1.104 (65) | 0.868 (66) | 1.006 (66) |
| all tf32 rows | 1.186 (13) | 1.148 (13) | 1.646 (13) | 1.366 (13) | 2.625 (13) |
| all bf16w rows | 1.655 (9) | 1.498 (9) | 2.156 (9) | 1.379 (9) | 2.194 (9) |
| all bf16kv rows | 1.043 (7) | 1.000 (7) | 1.124 (7) | 0.826 (7) | 1.103 (7) |
| all bf16 rows | 0.982 (25) | 1.134 (25) | 1.228 (25) | 0.898 (25) | 1.646 (25) |
| all f8w rows | - | - | - | - | - |
| **all rows** | 0.978 (120) | 1.009 (120) | 1.242 (119) | 0.948 (120) | 1.318 (120) |
| rows timed on every part (119) | 0.977 (119) | 1.009 (119) | 1.242 (119) | 0.948 (119) | 1.321 (119) |
| win / even / loss | 30/39/51 | 43/17/60 | 24/9/86 | 36/27/57 | 22/3/95 |

Geomean cells: value (rows). Parts time different row sets; compare columns on the "every part" row.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 1.02≈ 3.794/3.724 (torch-compile-ro) | **0.84** 5.939/7.084 (torch-compile-ro) | 1.29 10.06/7.821 (triton) | 1.03≈ 2.496/2.433 (chain-lt) | 1.09 2.882/2.642 (chain-lt) |
| mm_16x2048x7168 | f32 | 1.10 0.08785/0.0799 (cublas) | 1.04≈ 0.09165/0.0881 (cublas) | 1.03≈ 0.09265/0.0901 (cublas) ±7% | 1.14 0.1208/0.1059 (cublas) | 1.85 0.0947/0.0513 (cublas) |
| mm_4096x4096x4096 | f32 | 1.05≈ 3.845/3.671 (cublas) | 1.09≈ 8.841/8.115 (cublas) ±10% | - | 0.98≈ 2.417/2.466 (cublas) | 1.08 2.88/2.673 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.08 0.8057/0.7476 (lt-fresh) | **0.89** 1.198/1.345 (torch-compile-ro) | 1.34 2.466/1.845 (torch-compile-ro) | 0.97≈ 0.6885/0.7096 (lt-fresh) | 1.06 0.6929/0.6563 (lt) |
| pj_m128_n4096_k14336 | f32 | 1.12 0.4489/0.4014 (cublas-fresh) | 1.03≈ 0.6983/0.6769 (torch-compile-ro) | 1.69 1.563/0.9226 (lt-fresh) | 1.14 0.4507/0.3952 (lt-fresh) | 1.15 0.358/0.3103 (torch-compile-cg) |
| pj_m128_n4096_k4096 | f32 | 1.03≈ 0.1392/0.1352 (cublas-fresh) | 1.16 0.2413/0.2079 (torch-eager) | 1.82 0.4719/0.2591 (cublas) | 1.36 0.1347/0.0994 (lt-fresh) | 1.38 0.1349/0.0977 (torch-compile-cg) |
| pj_m128_n6144_k4096 | f32 | 1.14 0.2055/0.1802 (cublas-fresh) | 1.05 0.3372/0.3205 (torch-compile-cg) | 1.91 0.7521/0.3942 (lt) | 1.37 0.2323/0.169 (cublas-fresh) | 1.25 0.1867/0.149 (lt-fresh) |
| pj_m16_n18432_k7168 | f32 | 1.02≈ 0.6004/0.5898 (cublas-fresh) | 1.03≈ 0.6317/0.6134 (torch-compile) | 0.98≈ 0.434/0.4413 (torch-compile-cg) | 1.05≈ 0.7317/0.6973 (torch-compile-cg) | 1.01≈ 0.2071/0.2057 (torch-compile) |
| pj_m16_n2048_k7168 | f32 | **0.85** 0.04163/0.0492 (cublas-fresh) ±6% | 1.23 0.1032/0.084 (torch-compile) | 1.18 0.1018/0.086 (cublas-fresh) | 1.80 0.05359/0.0297 (cublas-fresh) | 1.25 0.06012/0.048 (lt-fresh) ±7% |
| pj_m16_n28672_k4096 | f32 | 1.04≈ 0.5428/0.5243 (lt-fresh) | 1.08 0.5876/0.5448 (torch-compile) | 1.03≈ 0.4139/0.4024 (torch-compile-cg) | 1.01≈ 0.6371/0.6277 (cublas-fresh) | 1.13 0.1943/0.1725 (torch-compile-cg) |
| pj_m16_n4096_k14336 | f32 | **0.94** 0.2597/0.2765 (torch-compile) | 1.07 0.2953/0.2755 (torch-compile) | **0.88** 0.2249/0.256 (lt) | 1.06 0.3636/0.343 (cublas-fresh) | 1.07 0.1232/0.1155 (lt) |
| pj_m16_n4096_k4096 | f32 | **0.90** 0.04951/0.0553 (cublas-fresh) | 1.40 0.119/0.085 (torch-compile-cg) | 1.10 0.09687/0.0881 (cublas-fresh) | 1.69 0.05707/0.0338 (lt-fresh) | 1.15 0.05455/0.0474 (lt-fresh) |
| pj_m16_n6144_k4096 | f32 | 1.11 0.09791/0.0881 (lt-fresh) | 1.20 0.1489/0.1239 (torch-compile) | 1.09 0.1249/0.1147 (torch-compile) | 1.17 0.1332/0.1137 (cublas-fresh) | 1.25 0.0719/0.0573 (torch-compile) |
| pj_m1_n18432_k7168 | f32 | 1.05≈ 0.5848/0.5571 (lt-fresh) | 1.01≈ 0.6126/0.6052 (cublas) | 1.06 0.4152/0.3922 (lt-fresh) | 1.10 0.713/0.6495 (torch-compile-cg) | 1.05 0.1955/0.1855 (lt-fresh) |
| pj_m1_n2048_k7168 | f32 | 0.95≈ 0.02538/0.0266 (cublas-fresh) ±8% | 1.16 0.08656/0.0748 (lt-fresh) | 1.64 0.07228/0.044 (cublas-fresh) | 1.18 0.03212/0.0273 (lt-fresh) | 1.66 0.04012/0.0242 (cublas-fresh) ±6% |
| pj_m1_n28672_k4096 | f32 | 1.03≈ 0.5186/0.5018 (lt-fresh) | 1.01≈ 0.5474/0.5417 (cublas) | 1.09 0.3792/0.3471 (torch-compile-cg) | 1.21 0.6407/0.5315 (lt-fresh) | 1.08 0.1766/0.1634 (lt-fresh) |
| pj_m1_n4096_k14336 | f32 | 1.08 0.2481/0.2294 (cublas-fresh) | 1.01≈ 0.282/0.2785 (cublas) | 1.12 0.1978/0.1772 (lt-fresh) | 1.32 0.3278/0.2478 (lt-fresh) | 1.25 0.102/0.0819 (cublas-fresh) ±7% |
| pj_m1_n4096_k4096 | f32 | 1.76 0.03959/0.0225 (lt-fresh) ±5% | 1.10 0.09445/0.086 (lt) | 1.46 0.07791/0.0532 (cublas-fresh) | 1.77 0.03453/0.0195 (lt-fresh) | 1.80 0.04248/0.0236 (cublas-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.61 0.08265/0.0512 (lt-fresh) | 1.10 0.1341/0.1219 (torch-compile) | 1.31 0.1017/0.0778 (lt-fresh) | 1.46 0.09443/0.0645 (lt-fresh) | 1.48 0.05328/0.0361 (lt-fresh) |
| pj_m2048_n28672_k4096 | f32 | 0.96≈ 11.77/12.2 (lt-fresh) | **0.83** 19.33/23.32 (torch-compile-ro) | 1.27 35.37/27.88 (torch-compile) | 1.17 9.59/8.182 (lt) | 1.09 10/9.206 (torch-eager) |
| pj_m2048_n4096_k14336 | f32 | 1.00≈ 6.419/6.415 (cublas-fresh) | **0.90** 10.97/12.14 (lt-fresh) | 1.29 18.02/13.97 (torch-eager) | 1.01≈ 4.237/4.186 (cublas) | 1.12 5.168/4.596 (lt-fresh) |
| pj_m2048_n4096_k4096 | f32 | 1.03≈ 1.83/1.774 (torch-eager) | **0.83** 2.891/3.478 (torch-compile) | 1.32 5.213/3.959 (torch-compile-cg) | 1.06 1.235/1.169 (lt) | 1.13 1.499/1.327 (torch-compile) |
| pj_m2048_n6144_k4096 | f32 | 1.02≈ 2.66/2.616 (torch-eager) | **0.90** 4.553/5.074 (lt-fresh) | 1.16 7.714/6.676 (torch-compile-cg) | 1.04≈ 1.862/1.789 (cublas) | 1.10 2.18/1.976 (torch-compile-ro) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.19 0.7019/0.5898 (cublas-tf32-fresh) | 1.23 1.076/0.8735 (torch-compile-ro-tf32) | 1.17 0.5375/0.4588 (lt-tf32) | 1.40 0.8771/0.6277 (lt-tf32-fresh) | 1.81 0.3351/0.1853 (cublas-tf32-fresh) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.31 0.405/0.3092 (cublas-tf32-fresh) | 1.08 0.6761/0.6246 (torch-compile-tf32) | 1.47 0.3493/0.2376 (cublas-tf32) | 1.09 0.3899/0.3564 (cublas-tf32-fresh) | 3.36 0.3538/0.1053 (lt-tf32) ±7% |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.66 0.1393/0.084 (cublas-tf32-fresh) | 1.29 0.2187/0.17 (torch-compile-cg-tf32) | 1.58 0.1311/0.0829 (cublas-tf32-fresh) | 1.59 0.1041/0.0655 (cublas-tf32-fresh) | 4.82 0.2122/0.044 (cublas-tf32-fresh) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.43 0.1821/0.127 (lt-tf32-fresh) | 1.08 0.2758/0.256 (lt-tf32) | 1.62 0.1676/0.1034 (cublas-tf32) | 1.75 0.2189/0.1249 (cublas-tf32-fresh) | 4.48 0.2349/0.0524 (lt-tf32-fresh) |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.12 9.048/8.057 (lt-tf32) | 1.01≈ 14.92/14.76 (torch-compile-cg-tf32) | 1.31 5.934/4.521 (torch-eager-tf32) | 1.73 9.506/5.5 (torch-compile-ro-tf32) | 1.53 1.965/1.282 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.03≈ 4.359/4.244 (cublas-tf32-fresh) | 1.07 8.433/7.85 (torch-compile-tf32) | 1.26 2.86/2.275 (torch-compile-tf32) | 1.19 3.27/2.759 (lt-tf32-fresh) | 1.87 1.075/0.5746 (lt-tf32-fresh) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.15 1.333/1.159 (lt-tf32-fresh) | 1.04≈ 2.578/2.489 (torch-compile-ro-tf32) | 1.47 0.8916/0.6083 (lt-tf32) | 1.13 0.9572/0.8479 (lt-tf32) | 2.99 0.5192/0.1738 (lt-tf32-fresh) ±19% |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.15 2.018/1.75 (lt-tf32-fresh) | 1.05≈ 3.436/3.286 (cublas-tf32) | 1.48 1.271/0.8581 (torch-compile-ro-tf32) | 1.19 1.411/1.186 (cublas-tf32) | 2.20 0.5655/0.2569 (lt-tf32-fresh) |
| pj_m128_n28672_k4096_bf16 | bf16 | 1.27 0.4024/0.3174 (triton-bf16) | 1.12 0.5212/0.4669 (torch-bf16-compile-cg) | 1.24 0.2918/0.2355 (triton-bf16) ±13% | 1.08 0.3884/0.3584 (torch-bf16-compile) | 1.55 0.1436/0.0924 (triton-bf16) |
| pj_m128_n4096_k14336_bf16 | bf16 | 1.29 0.2209/0.1717 (torch-bf16-compile-ro) | 1.51 0.4226/0.2806 (torch-bf16-compile) | 1.69 0.2083/0.1229 (torch-bf16-compile-ro) | 1.02≈ 0.2264/0.2222 (torch-bf16-compile) | 2.86 0.1652/0.0578 (torch-bf16-compile-cg) |
| pj_m128_n4096_k4096_bf16 | bf16 | 1.25 0.06639/0.0532 (torch-bf16) | 1.40 0.1193/0.085 (torch-bf16-compile-ro) | 1.57 0.08835/0.0563 (torch-bf16-compile-ro) | 1.05≈ 0.06972/0.0666 (torch-bf16) | 2.55 0.06077/0.0238 (torch-bf16-compile-ro) |
| pj_m128_n6144_k4096_bf16 | bf16 | 1.02≈ 0.08742/0.086 (torch-bf16-compile-cg) | 1.39 0.1774/0.128 (torch-bf16-compile-cg) | 1.56 0.1056/0.0676 (torch-bf16-compile-cg) | 1.20 0.1078/0.0901 (torch-bf16-compile-cg) | 2.39 0.07726/0.0323 (torch-bf16-compile) |
| pj_m16_n28672_k4096_bf16 | bf16 | 1.02≈ 0.2804/0.2744 (torch-bf16-compile-cg) | 1.11 0.3052/0.2744 (torch-bf16-compile) | 1.13 0.211/0.1864 (torch-bf16-compile-cg) | 1.06 0.3679/0.3484 (torch-bf16-compile) | 1.62 0.1459/0.0901 (torch-bf16-compile) |
| pj_m16_n4096_k14336_bf16 | bf16 | **0.85** 0.1178/0.1393 (torch-bf16-compile-cg) | 1.17 0.168/0.1434 (torch-bf16-compile) | 1.12 0.123/0.1096 (torch-bf16-compile-cg) | **0.73** 0.1437/0.1956 (torch-bf16-compile) | 1.66 0.09177/0.0553 (torch-bf16) |
| pj_m16_n4096_k4096_bf16 | bf16 | 0.98≈ 0.04212/0.043 (torch-bf16-compile-cg) | 1.71 0.08065/0.0471 (torch-bf16-compile-cg) | 0.96≈ 0.04142/0.043 (torch-bf16-compile-cg) | **0.82** 0.04778/0.0585 (torch-bf16-compile-cg) | 2.33 0.05546/0.0238 (torch-bf16-compile-ro) |
| pj_m16_n6144_k4096_bf16 | bf16 | **0.77** 0.04904/0.0635 (torch-bf16-compile) | 1.42 0.09434/0.0666 (torch-bf16-compile-cg) | 1.07 0.06373/0.0594 (torch-bf16-compile-cg) | **0.72** 0.06278/0.087 (torch-bf16-compile) | 1.99 0.05744/0.0288 (torch-bf16-compile) |
| pj_m1_n28672_k4096_bf16 | bf16 | **0.90** 0.2497/0.2765 (torch-bf16-compile) | 1.03≈ 0.2859/0.2785 (torch-bf16-compile-cg) | 1.33 0.2438/0.1833 (torch-bf16-compile) | 1.11 0.3371/0.3031 (torch-bf16-compile-cg) | 1.26 0.1144/0.0911 (torch-bf16) |
| pj_m1_n4096_k14336_bf16 | bf16 | **0.72** 0.1015/0.1413 (torch-bf16) | 1.09 0.1589/0.1464 (torch-bf16) | 1.17 0.1262/0.1075 (torch-bf16-compile) | **0.73** 0.1226/0.169 (torch-bf16-compile-cg) | 1.34 0.07014/0.0525 (torch-bf16-compile-cg) |
| pj_m1_n4096_k4096_bf16 | bf16 | **0.55** 0.02687/0.0492 (torch-bf16-compile) ±12% | 1.26 0.06214/0.0492 (torch-bf16-compile-cg) | **0.94** 0.04231/0.0451 (torch-bf16-compile) | **0.33** 0.02121/0.0635 (torch-bf16-compile-cg) | 1.01≈ 0.02212/0.0218 (torch-bf16-compile-ro) ±7% |
| pj_m1_n6144_k4096_bf16 | bf16 | **0.45** 0.02839/0.0635 (torch-bf16-compile-cg) | 1.21 0.08146/0.0676 (torch-bf16-compile) | 1.31 0.06971/0.0532 (torch-bf16-compile-cg) | **0.31** 0.028/0.0891 (torch-bf16-compile-cg) | 1.22 0.03508/0.0287 (torch-bf16-compile-cg) |
| pj_m2048_n28672_k4096_bf16 | bf16 | 1.14 4.555/4 (torch-bf16-compile-cg) | 0.98≈ 7.049/7.199 (torch-bf16-compile) | 1.43 3.084/2.159 (torch-bf16-compile-ro) | 1.79 4.946/2.757 (torch-bf16-compile-cg) | 2.73 1.66/0.6085 (torch-bf16-compile-ro) |
| pj_m2048_n4096_k14336_bf16 | bf16 | 1.08 2.267/2.094 (torch-bf16-compile-cg) | **0.91** 3.958/4.332 (triton-bf16) | 1.35 1.446/1.073 (torch-bf16) | 1.10 1.545/1.407 (torch-bf16-compile) | 2.60 0.7741/0.2978 (torch-bf16-compile-cg) |
| pj_m2048_n4096_k4096_bf16 | bf16 | 1.13 0.6742/0.598 (torch-bf16-compile-cg) | 1.14 1.252/1.099 (torch-bf16-compile-cg) | 1.49 0.5039/0.3389 (torch-bf16) | 1.14 0.4557/0.4004 (torch-bf16-compile-cg) | 2.59 0.2379/0.0917 (torch-bf16-compile) |
| pj_m2048_n6144_k4096_bf16 | bf16 | 1.06 0.9403/0.8868 (torch-bf16-compile-ro) | 1.02≈ 1.63/1.592 (torch-bf16-compile-ro) | 1.48 0.6271/0.425 (torch-bf16-compile-cg) | 1.13 0.6773/0.597 (torch-bf16-compile-cg) | 2.54 0.3476/0.1367 (torch-bf16-compile) |
| geomean f32 | | 1.067 (23) | 1.033 (23) | 1.249 (22) | 1.200 (23) | 1.219 (23) |
| geomean tf32 | | 1.242 (8) | 1.102 (8) | 1.412 (8) | 1.360 (8) | 2.656 (8) |
| geomean bf16 | | 0.931 (16) | 1.199 (16) | 1.284 (16) | 0.883 (16) | 1.916 (16) |
| **geomean** | | 1.045 (47) | 1.099 (47) | 1.288 (46) | 1.104 (47) | 1.623 (47) |
| win / even / loss | | 9/16/22 | 7/13/27 | 2/4/40 | 6/9/32 | 0/2/45 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.03≈ 1.826/1.771 (chain-lt2) | **0.86** 3.165/3.664 (chain-lt2) | 1.37 5.436/3.976 (chain-streams) | 1.03≈ 1.276/1.233 (chain-lt2) | 1.12 1.509/1.35 (chain-lt2) |
| mlp_2048 | f32 | 0.97≈ 0.8858/0.9134 (chain-lt) | **0.93** 1.604/1.716 (torch-compile) | 1.34 2.718/2.022 (torch-compile-cg) | 0.98≈ 0.6197/0.6328 (chain-lt) | 1.12 0.7587/0.6798 (chain-lt) |
| mlp_512 | f32 | 1.20 0.03946/0.0328 (torch-compile) | 1.42 0.06422/0.0451 (triton) | 2.28 0.1401/0.0614 (chain-lt) | 1.60 0.04583/0.0287 (torch-compile-cg) | 1.47 0.04757/0.0323 (torch-compile-cg) |
| smlp_4096_256_1024 | f32 | 1.06 0.1591/0.1495 (torch-compile) | 1.22 0.2826/0.2315 (torch-compile) | 1.34 0.3902/0.2918 (chain-eager) | 1.05 0.1101/0.1044 (triton) | 1.14 0.127/0.1111 (chain-lt) |
| mlp2_2048_tf32 | tf32 | 1.04≈ 1.57/1.513 (torch-compile-ro-tf32) | 1.13 3.013/2.674 (torch-compile-ro-tf32) | 1.54 1.12/0.7291 (torch-compile-cg-tf32) | 1.20 1.009/0.8428 (torch-compile-cg-tf32) | 2.64 0.5558/0.2106 (torch-compile-cg-tf32) |
| mlp_2048_tf32 | tf32 | 1.03≈ 0.7846/0.7598 (torch-compile-ro-tf32) | 1.24 1.487/1.197 (torch-compile-ro-tf32) | 1.53 0.5632/0.3676 (torch-compile-cg-tf32) | 1.16 0.4937/0.4239 (torch-compile-ro-tf32) | 3.22 0.35/0.1087 (torch-compile-ro-tf32) |
| geomean f32 | | 1.064 (4) | 1.088 (4) | 1.539 (4) | 1.143 (4) | 1.204 (4) |
| geomean tf32 | | 1.035 (2) | 1.183 (2) | 1.534 (2) | 1.181 (2) | 2.915 (2) |
| **geomean** | | 1.054 (6) | 1.119 (6) | 1.537 (6) | 1.155 (6) | 1.617 (6) |
| win / even / loss | | 0/4/2 | 2/0/4 | 0/0/6 | 0/2/4 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.28 0.03686/0.0287 (torch-compile-cg) ±7% | 1.39 0.06421/0.0461 (torch-compile-ro) | 1.12 0.06188/0.0553 (torch-sdpa-compile-cg) | 0.95≈ 0.02727/0.0287 (torch-sdpa-compile-cg) | 1.68 0.04478/0.0267 (torch-compile-ro) ±21% |
| attn_2048_128 | f32 | 1.24 0.1163/0.0942 (torch-sdpa-compile-cg) | 1.75 0.3152/0.1802 (torch-compile-cg) | 4.06 0.7319/0.1802 (torch-sdpa-compile-cg) | 1.13 0.09286/0.0819 (torch-compile-ro) | 1.29 0.1022/0.0792 (torch-compile-ro) |
| attn_2048_64 | f32 | 1.15 0.08032/0.0696 (torch-compile-ro) | 1.44 0.1887/0.1311 (torch-sdpa-compile-cg) | 1.23 0.1453/0.1178 (torch-sdpa-compile-cg) | 1.33 0.08429/0.0635 (torch-compile-ro) | 1.58 0.09448/0.0599 (torch-compile-ro) |
| causal_1024_256 | f32 | **0.34** 15.16/44.09 (torch-compile-ro) | **0.18** 25.15/138.1 (torch-compile-cg) | **0.35** 37.16/105 (torch-compile-cg) | **0.07** 18.06/253.3 (torch-compile-cg) | **0.20** 21.29/107.9 (torch-compile-cg) |
| decode_1024_256 | f32 | **0.53** 8.654/16.36 (chain-cache) | **0.74** 17.46/23.55 (chain-cache) | **0.79** 26.7/33.94 (chain-cache) | **0.35** 11.91/34.45 (chain-cache) | 1.41 28.4/20.14 (chain-cache) |
| gen_1024_512_256 | f32 | **0.36** 7.769/21.4 (torch-compile-ro) | **0.19** 12.77/67.65 (torch-compile-cg) | **0.36** 18.71/51.44 (torch-compile-cg) | **0.07** 9.177/127.4 (torch-compile-ro) | **0.20** 10.75/54.06 (torch-compile-cg) |
| lsm_1_131072 | f32 | 0.78≈ 0.01755/0.0225 (torch-compile-cg) ±26% | 1.07 0.02859/0.0266 (torch-compile-ro) | 1.00≈ 0.02985/0.0297 (torch-compile-cg) ±12% | **0.87** 0.02146/0.0246 (torch-compile-ro) | 1.45 0.02892/0.02 (torch-compile-cg) |
| lsm_32_8192 | f32 | 2.33 0.02372/0.0102 (torch-compile) ±24% | 2.91 0.02965/0.0102 (torch-compile-cg) | 1.90 0.0271/0.0143 (torch-sdpa-compile) | 2.87 0.02357/0.0082 (torch-compile-cg) | 2.55 0.02859/0.0112 (torch-compile-ro) ±5% |
| lsm_4_32768 | f32 | 1.18 0.01687/0.0143 (torch-compile) | 1.39 0.02424/0.0174 (torch-compile-ro) | 1.16 0.02269/0.0195 (torch-compile-cg) | 1.54 0.02045/0.0133 (torch-sdpa-compile-cg) | 1.68 0.02752/0.0164 (torch-compile-cg) |
| prefill_1024_512_256 | f32 | **0.34** 7.777/22.57 (torch-compile-cg) | **0.19** 13/69.86 (torch-compile-cg) | **0.37** 19.53/52.72 (torch-compile-cg) | **0.07** 9.435/127.8 (torch-compile-cg) | **0.20** 11.05/54.25 (torch-compile-cg) |
| rowsum_2048 | f32 | 0.99≈ 0.4592/0.4639 (chain-eager) | 1.07 0.8282/0.7772 (torch-compile-ro) | 1.38 1.365/0.9902 (chain-eager) | 0.98≈ 0.3214/0.327 (chain-eager) | 1.17 0.4093/0.3493 (chain-eager) |
| rowsum_4096 | f32 | 1.12 3.93/3.498 (torch-compile) | **0.88** 6.157/7.005 (torch-compile) | 1.29 10.13/7.846 (torch-compile-ro) | 1.04≈ 2.527/2.423 (torch-compile) | 1.09 2.917/2.684 (torch-compile-ro) |
| softmax_rows_2048 | f32 | 1.00≈ 0.472/0.471 (torch-compile) | 1.16 0.8616/0.7424 (torch-compile-cg) | 1.39 1.384/0.9953 (torch-compile) | 1.01≈ 0.3316/0.3287 (torch-compile) | 1.18 0.4196/0.3548 (torch-compile-ro) |
| softmax_rows_4096 | f32 | 1.12 3.973/3.54 (torch-compile-ro) | **0.88** 6.284/7.123 (torch-compile) | 1.29 10.2/7.921 (torch-compile-ro) | 1.06 2.595/2.46 (torch-compile-ro) | 1.09 2.966/2.714 (torch-compile-cg) |
| **geomean** | | 0.855 (14) | 0.827 (14) | 1.029 (14) | 0.600 (14) | 0.935 (14) |
| win / even / loss | | 4/3/7 | 6/0/8 | 4/1/9 | 5/4/5 | 3/0/11 |

## LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 0.98≈ 1.066/1.089 (torch-compile) | **0.93** 1.131/1.216 (torch-compile) | 1.13 0.9517/0.8452 (torch-compile) | **0.91** 1.176/1.286 (torch-compile-cg) | 1.18 0.5538/0.4681 (torch-compile-cg) |
| llm_blkpre_t2048 | f32 | **0.94** 25.17/26.83 (liger) | 1.17 58.61/50.15 (liger) | 1.27 75.41/59.15 (torch-compile) | 0.99≈ 17.87/18.04 (liger) | 1.14 21.18/18.66 (liger) |
| llm_dec_b1_l2048 | f32 | **0.92** 0.2395/0.259 (torch-compile-ro) | **0.77** 0.2693/0.3477 (torch-compile) | **0.80** 0.2143/0.2671 (torch-compile) | **0.66** 0.2704/0.4096 (torch-compile) | **0.57** 0.1098/0.193 (torch-compile-cg) |
| llm_dec_b1_l8192 | f32 | **0.87** 0.3024/0.3462 (torch-compile) | **0.75** 0.3372/0.4481 (torch-compile) | **0.65** 0.2686/0.4154 (torch-compile) | **0.72** 0.344/0.48 (torch-compile-cg) | **0.46** 0.1391/0.2995 (torch-compile-cg) |
| llm_dec_b32_l2048 | f32 | **0.39** 0.8996/2.293 (torch-compile) | **0.33** 0.984/2.989 (torch-compile-ro) | **0.40** 0.7924/1.962 (torch-compile-cg) | **0.39** 1.013/2.569 (torch-compile) | **0.31** 0.3553/1.163 (torch-compile) |
| llm_dec_b32_l8192 | f32 | **0.29** 2.779/9.728 (torch-compile-cg) | **0.28** 3.15/11.33 (torch-compile-ro) | **0.30** 2.11/7.043 (torch-compile-ro) | **0.31** 3.216/10.5 (torch-compile) | **0.26** 1.028/3.931 (torch-compile-cg) |
| llm_dec_b8_l2048 | f32 | **0.51** 0.3876/0.7657 (torch-compile-cg) | **0.89** 0.8352/0.9394 (torch-compile) | **0.52** 0.3588/0.695 (torch-compile-cg) | **0.49** 0.4321/0.8825 (torch-compile) | **0.43** 0.1796/0.4141 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.35** 0.8684/2.513 (torch-compile-cg) | **0.46** 1.382/2.979 (torch-compile-cg) | **0.33** 0.6829/2.098 (torch-compile-cg) | **0.35** 0.9969/2.873 (torch-compile) | **0.32** 0.3534/1.114 (torch-compile) |
| llm_pre_t2048 | f32 | **0.90** 6.458/7.216 (liger) | **0.82** 11.11/13.59 (liger) | 1.49 19.28/12.94 (liger) | **0.92** 4.756/5.164 (liger) | 1.31 5.985/4.552 (liger) |
| llm_pre_t8192 | f32 | **0.84** 44.65/52.89 (liger) | 1.06 104.3/98.78 (liger) | 3.15 208.6/66.2 (liger) | **0.93** 34.83/37.53 (liger) | 3.35 86.56/25.82 (liger) |
| llm_blkpre_t2048_tf32 | tf32 | 1.09 19.74/18.08 (triton-tf32) | 1.05≈ 33.22/31.79 (triton-tf32) | 1.60 17.41/10.9 (triton-tf32) | 1.32 16.3/12.37 (triton-tf32) | 1.89 6.507/3.44 (triton-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.18 5.342/4.543 (triton-tf32) | 1.12 9.455/8.418 (triton-tf32) | 2.79 8.595/3.085 (triton-tf32) | 1.24 4.051/3.276 (triton-tf32) | 2.28 3.131/1.373 (triton-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.13 0.627/0.5569 (torch-bf16-compile) | 1.05≈ 0.7056/0.6743 (torch-bf16-compile) | 1.65 0.7915/0.4794 (torch-bf16-compile-cg) | **0.94** 0.7082/0.7551 (torch-bf16-compile-cg) | 1.72 0.4931/0.286 (torch-bf16-compile-cg) |
| llm_blkpre_t2048_bf16w | bf16w | 3.15 25.54/8.103 (torch-bf16-compile) | 3.19 49.78/15.62 (torch-bf16-compile-ro) | 11.84 57.23/4.833 (torch-bf16-compile) | 3.08 17.05/5.542 (torch-bf16-compile) | 16.88 22.07/1.308 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16w | bf16w | 1.01≈ 0.1477/0.1455 (torch-bf16-compile) | **0.82** 0.1821/0.223 (torch-bf16-compile) | **0.93** 0.1637/0.1762 (torch-bf16-compile) | **0.57** 0.1792/0.3121 (torch-bf16-compile-cg) | **0.61** 0.08862/0.1455 (torch-bf16-compile-cg) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.15 0.2152/0.1868 (torch-bf16-compile) | **0.86** 0.2509/0.2924 (torch-bf16-compile) | **0.94** 0.2157/0.2292 (torch-bf16-compile) | **0.74** 0.2552/0.3467 (torch-bf16-compile-cg) | **0.78** 0.1154/0.1472 (torch-bf16-compile-cg) |
| llm_dec_b32_l2048_bf16w | bf16w | 1.74 0.8071/0.4644 (torch-bf16-compile) | 1.29 0.9582/0.74 (torch-bf16-cudnn) | 1.41 0.7394/0.5243 (torch-bf16-cudnn) | 1.50 0.8987/0.5992 (torch-bf16-compile) | 1.62 0.3662/0.2266 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16w | bf16w | 1.89 2.821/1.489 (torch-bf16-compile) | 1.86 3.135/1.683 (torch-bf16-cudnn) | 1.90 2.058/1.082 (torch-bf16-cudnn) | 1.91 3.118/1.629 (torch-bf16-compile) | 2.11 1.028/0.4866 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.33 0.2948/0.222 (torch-bf16-compile) | 1.55 0.5245/0.3388 (torch-bf16-compile) | 1.16 0.3114/0.2689 (torch-bf16-compile-cg) | 1.00≈ 0.347/0.3483 (torch-bf16-compile) | 0.96≈ 0.1552/0.1615 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16w | bf16w | 1.56 0.7715/0.4946 (torch-bf16-compile) | 1.50 1.075/0.7144 (liger-bf16) | 1.26 0.6365/0.5052 (torch-bf16) | 1.48 0.9045/0.6121 (torch-bf16-compile) | 1.48 0.3361/0.2278 (torch-bf16-compile-cg) |
| llm_pre_t2048_bf16w | bf16w | 3.30 6.365/1.931 (torch-bf16-compile-cg) | 2.90 10.85/3.742 (torch-bf16-compile-ro) | 15.03 16.27/1.082 (torch-bf16-compile-cg) | 3.49 4.6/1.319 (torch-bf16-compile-cg) | 17.53 6.13/0.3497 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_bf16kv | bf16kv | 1.10 0.6116/0.5569 (torch-bf16-compile) | 1.02≈ 0.6876/0.6743 (torch-bf16-compile) | 1.60 0.7672/0.4794 (torch-bf16-compile-cg) | **0.92** 0.6906/0.751 (torch-bf16-compile-cg) | 1.72 0.4905/0.286 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16kv | bf16kv | 0.96≈ 0.1392/0.1455 (torch-bf16-compile) | **0.78** 0.1732/0.223 (torch-bf16-compile) | **0.90** 0.1593/0.1762 (torch-bf16-compile) | **0.55** 0.1682/0.3065 (torch-bf16-compile-cg) | **0.60** 0.08693/0.1455 (torch-bf16-compile-cg) |
| llm_dec_b1_l8192_bf16kv | bf16kv | 0.96≈ 0.1792/0.1868 (torch-bf16-compile) | **0.75** 0.2202/0.2924 (torch-bf16-compile) | **0.88** 0.2016/0.2292 (torch-bf16-compile) | **0.60** 0.2105/0.3505 (torch-bf16-compile-cg) | **0.77** 0.1137/0.1472 (torch-bf16-compile-cg) |
| llm_dec_b32_l2048_bf16kv | bf16kv | 1.10 0.5105/0.4644 (torch-bf16-compile) | **0.91** 0.6754/0.74 (torch-bf16-cudnn) | 1.19 0.6216/0.5243 (torch-bf16-cudnn) | 0.99≈ 0.5943/0.5975 (torch-bf16-compile) | 1.40 0.3177/0.2266 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16kv | bf16kv | 1.16 1.728/1.489 (torch-bf16-compile) | 1.18 1.993/1.683 (torch-bf16-cudnn) | 1.45 1.566/1.082 (torch-bf16-cudnn) | 1.17 1.922/1.638 (torch-bf16-compile) | 1.63 0.7925/0.4866 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16kv | bf16kv | 1.01≈ 0.2241/0.222 (torch-bf16-compile) | 1.39 0.4704/0.3388 (torch-bf16-compile) | 1.01≈ 0.2717/0.2689 (torch-bf16-compile-cg) | **0.77** 0.2706/0.353 (torch-bf16-compile) | **0.89** 0.1434/0.1615 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16kv | bf16kv | 1.04≈ 0.5126/0.4946 (torch-bf16-compile) | 1.12 0.7996/0.7144 (liger-bf16) | 1.03≈ 0.52/0.5052 (torch-bf16) | 0.97≈ 0.5934/0.6136 (torch-bf16-compile) | 1.24 0.2817/0.2278 (torch-bf16-compile-cg) |
| llm_blkdec_b1_l4096_bf16 | bf16 | 1.03≈ 0.5738/0.5569 (torch-bf16-compile) | **0.95** 0.6397/0.6743 (torch-bf16-compile) | 1.11 0.5308/0.4794 (torch-bf16-compile-cg) | **0.87** 0.6554/0.7508 (torch-bf16-compile-cg) | **0.93** 0.2649/0.286 (torch-bf16-compile-cg) |
| llm_blkpre_t2048_bf16 | bf16 | 1.18 9.586/8.103 (torch-bf16-compile) | 1.04≈ 16.18/15.62 (torch-bf16-compile-ro) | 1.47 7.109/4.833 (torch-bf16-compile) | 1.38 7.616/5.527 (torch-bf16-compile-cg) | 2.93 3.828/1.308 (torch-bf16-compile-cg) |
| llm_dec_b1_l2048_bf16 | bf16 | 0.95≈ 0.1385/0.1455 (torch-bf16-compile) | **0.79** 0.177/0.223 (torch-bf16-compile) | **0.90** 0.1582/0.1762 (torch-bf16-compile) | **0.55** 0.1688/0.3082 (torch-bf16-compile-cg) | **0.57** 0.08364/0.1455 (torch-bf16-compile-cg) |
| llm_dec_b1_l8192_bf16 | bf16 | **0.95** 0.1768/0.1868 (torch-bf16-compile) | **0.76** 0.2233/0.2924 (torch-bf16-compile) | **0.85** 0.1956/0.2292 (torch-bf16-compile) | **0.60** 0.2113/0.3507 (torch-bf16-compile-cg) | **0.74** 0.1092/0.1472 (torch-bf16-compile-cg) |
| llm_dec_b32_l2048_bf16 | bf16 | 1.09 0.5048/0.4644 (torch-bf16-compile) | **0.91** 0.6751/0.74 (torch-bf16-cudnn) | **0.94** 0.4909/0.5243 (torch-bf16-cudnn) | 0.99≈ 0.5931/0.5996 (torch-bf16-compile) | 1.17 0.2652/0.2266 (torch-bf16-compile-cg) |
| llm_dec_b32_l8192_bf16 | bf16 | 1.17 1.746/1.489 (torch-bf16-compile) | 1.19 1.997/1.683 (torch-bf16-cudnn) | 1.33 1.442/1.082 (torch-bf16-cudnn) | 1.17 1.923/1.642 (torch-bf16-compile) | 1.51 0.7362/0.4866 (torch-bf16-compile-cg) |
| llm_dec_b8_l2048_bf16 | bf16 | 1.01≈ 0.2239/0.222 (torch-bf16-compile) | 1.38 0.469/0.3388 (torch-bf16-compile) | 1.01≈ 0.271/0.2689 (torch-bf16-compile-cg) | **0.78** 0.2709/0.346 (torch-bf16-compile) | **0.88** 0.1419/0.1615 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16 | bf16 | 1.04≈ 0.5133/0.4946 (torch-bf16-compile) | 1.12 0.8015/0.7144 (liger-bf16) | 1.02≈ 0.5165/0.5052 (torch-bf16) | 0.97≈ 0.5948/0.6134 (torch-bf16-compile) | 1.23 0.2793/0.2278 (torch-bf16-compile-cg) |
| llm_pre_t2048_bf16 | bf16 | 1.37 2.65/1.931 (torch-bf16-compile-cg) | 1.26 4.7/3.742 (torch-bf16-compile-ro) | 1.94 2.103/1.082 (torch-bf16-compile-cg) | 1.41 1.862/1.319 (torch-bf16-compile-cg) | 3.53 1.234/0.3497 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_f8w | f8w | 0.3615 (no leg) | 0.5482 (no leg) | 0.4754 (no leg) | - | 0.2082 (no leg) |
| llm_dec_b1_l2048_f8w | f8w | 0.0538 (no leg) | 0.1586 (no leg) | 0.151 (no leg) | - | 0.07836 (no leg) |
| llm_dec_b1_l8192_f8w | f8w | 0.1676 (no leg) | 0.2293 (no leg) | 0.2046 (no leg) | - | 0.1047 (no leg) |
| llm_dec_b32_l2048_f8w | f8w | 0.7909 (no leg) | 0.9839 (no leg) | 0.7592 (no leg) | - | 0.3522 (no leg) |
| llm_dec_b32_l8192_f8w | f8w | 2.813 (no leg) | 3.161 (no leg) | 2.079 (no leg) | - | 1.022 (no leg) |
| llm_dec_b8_l2048_f8w | f8w | 0.2496 (no leg) | 0.812 (no leg) | 0.3186 (no leg) | - | 0.1307 (no leg) |
| llm_dec_b8_l8192_f8w | f8w | 0.7264 (no leg) | 1.362 (no leg) | 0.6478 (no leg) | - | 0.3038 (no leg) |
| geomean f32 | | 0.637 (10) | 0.678 (10) | 0.768 (10) | 0.613 (10) | 0.666 (10) |
| geomean tf32 | | 1.133 (2) | 1.083 (2) | 2.110 (2) | 1.276 (2) | 2.077 (2) |
| geomean bf16w | | 1.655 (9) | 1.498 (9) | 2.156 (9) | 1.379 (9) | 2.194 (9) |
| geomean bf16kv | | 1.043 (7) | 1.000 (7) | 1.124 (7) | 0.826 (7) | 1.103 (7) |
| geomean bf16 | | 1.081 (9) | 1.026 (9) | 1.135 (9) | 0.924 (9) | 1.256 (9) |
| **geomean** | | 1.035 (37) | 1.004 (37) | 1.232 (37) | 0.908 (37) | 1.215 (37) |
| win / even / loss | | 10/10/17 | 17/4/16 | 13/4/20 | 20/6/11 | 15/1/21 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.44** 0.06689/0.1516 (torch-eager) | **0.69** 0.1102/0.1608 (torch-compile-cg) | **0.54** 0.0734/0.1362 (torch-compile-cg) | **0.57** 0.05355/0.0932 (torch-compile-cg) | **0.49** 0.04574/0.094 (torch-eager) |
| convmc_8_64_56_3 | f32 | **0.68** 0.08304/0.1219 (torch-eager) | 1.88 0.148/0.0788 (torch-eager) | 2.17 0.1886/0.087 (torch-eager) | 1.64 0.09594/0.0584 (torch-eager) | 1.61 0.1148/0.0715 (torch-eager) |
| ew_4096 | f32 | **0.37** 0.08924/0.2437 (torch-compile-cg) ±8% | **0.53** 0.1267/0.2396 (torch-compile-ro) | **0.87** 0.1341/0.1536 (torch-compile-cg) | **0.32** 0.08164/0.253 (torch-compile-cg) | 1.65 0.1202/0.0728 (torch-compile) |
| ewconvew_2048_5 | f32 | **0.41** 0.07788/0.1881 (torch-compile-cg) | **0.53** 0.1304/0.2447 (torch-compile) | **0.52** 0.08923/0.1731 (torch-compile-ro) | **0.50** 0.06455/0.13 (torch-compile-cg) | **0.50** 0.05628/0.1118 (torch-compile-cg) |
| fir_4194304_16 | f32 | **0.44** 0.04556/0.1044 (torch-eager) | **0.62** 0.07393/0.1188 (torch-compile-cg) | **0.60** 0.05142/0.085 (torch-compile-cg) | **0.57** 0.03763/0.0655 (torch-compile-cg) | **0.54** 0.03295/0.0605 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.49** 0.05363/0.1085 (torch-compile) | **0.73** 0.1018/0.1393 (torch-compile-cg) | **0.68** 0.0643/0.0942 (torch-compile-ro) | **0.64** 0.04429/0.0696 (torch-compile-cg) | **0.60** 0.03921/0.0655 (torch-compile-ro) |
| convmc_8_64_56_3_tf32 | tf32 | 1.18 0.08321/0.0707 (torch-eager-tf32) | 1.68 0.1481/0.088 (torch-eager-tf32) | 3.93 0.1892/0.0481 (torch-eager-tf32) | 2.17 0.09545/0.0439 (torch-eager-tf32) | 3.10 0.1178/0.038 (torch-eager-tf32) |
| geomean f32 | | 0.463 (6) | 0.741 (6) | 0.775 (6) | 0.617 (6) | 0.772 (6) |
| geomean tf32 | | 1.177 (1) | 1.683 (1) | 3.933 (1) | 2.174 (1) | 3.099 (1) |
| **geomean** | | 0.529 (7) | 0.833 (7) | 0.978 (7) | 0.739 (7) | 0.942 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 5/0/2 | 5/0/2 | 4/0/3 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 1.02≈ 1.386/1.366 (torch-compile-cg) | **0.84** 2.401/2.874 (chain-streams) | 1.34 4.077/3.05 (torch-compile-cg) | 1.00≈ 0.9404/0.9411 (torch-compile) | 1.14 1.165/1.025 (chain-lt) |
| fanout_2048 | f32 | 1.00≈ 0.9163/0.9132 (chain-streams) | **0.88** 1.616/1.844 (torch-compile-ro) | 1.34 2.721/2.023 (torch-eager) | 1.01≈ 0.6424/0.6368 (chain-streams) | 1.14 0.7799/0.6813 (chain-streams) |
| indep_2048 | f32 | 1.04≈ 0.9508/0.9155 (torch-compile) | **0.84** 1.609/1.907 (chain-streams) | 1.37 2.732/1.995 (torch-compile) | 0.99≈ 0.6291/0.6342 (torch-compile) | 1.15 0.7879/0.6868 (chain-lt) |
| indep_512 | f32 | 1.16 0.03816/0.0328 (torch-compile) ±6% | 1.55 0.07292/0.0471 (torch-compile) | 2.46 0.1412/0.0573 (chain-streams) | 1.96 0.05657/0.0288 (torch-compile-cg) ±10% | 2.09 0.05934/0.0284 (torch-compile) |
| loop_4096 | f32 | **0.94** 26.93/28.72 (chain-lt) | 1.20 66.08/54.85 (chain-lt) | 1.32 82.56/62.35 (torch-eager) | 0.98≈ 18.82/19.22 (chain-cutlass) | 1.08 22.9/21.12 (chain-lt) |
| loopmm_1024_8 | f32 | 1.05≈ 0.617/0.5878 (torch-compile-cg) | 1.20 1.107/0.9226 (torch-compile-ro) | 1.44 1.575/1.093 (torch-eager) | 1.17 0.4151/0.3533 (torch-compile-ro) | 1.14 0.5147/0.4503 (torch-compile-cg) |
| rt_mlp2_1024 | f32 | 1.09 2.753/2.516 (torch-compile-cg) | **0.80** 4.313/5.396 (torch-compile) | 1.75 7.889/4.502 (torch-compile-cg) | 1.10 1.67/1.519 (torch-compile-ro) | 1.08 2.11/1.96 (torch-compile-ro) |
| rt_mlp2_2048 | f32 | 0.99≈ 15.45/15.62 (torch-compile) | **0.80** 26.34/32.76 (torch-compile) | 1.33 44.17/33.23 (torch-compile-cg) | 0.99≈ 10.3/10.44 (torch-compile-cg) | 1.10 12.26/11.18 (torch-compile-cg) |
| twin_2048 | f32 | 1.01≈ 0.9286/0.9175 (torch-compile-cg) | **0.92** 1.589/1.734 (torch-compile-ro) | 1.37 2.7/1.97 (torch-compile) | 0.98≈ 0.6312/0.6451 (torch-eager) | 1.14 0.7814/0.6843 (torch-compile) |
| **geomean** | | 1.032 (9) | 0.977 (9) | 1.494 (9) | 1.102 (9) | 1.201 (9) |
| win / even / loss | | 1/6/2 | 6/0/3 | 0/0/9 | 0/6/3 | 0/0/9 |

## bf16 real-world reference

The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3174 (triton-bf16) · f32 2.54 | 0.4669 (bf16-compile-cg) · f32 2.57 | 0.2355 (triton-bf16) · f32 10.47 | 0.3594 (bf16-compile) · f32 1.92 | 0.0924 (triton-bf16) · f32 7.50 |
| pj_m128_n4096_k14336 | 0.1717 (bf16-compile-ro) · f32 2.61 | 0.2806 (bf16-compile) · f32 2.49 | 0.1229 (bf16-compile-ro) · f32 12.72 | 0.2048 (triton-bf16) · f32 2.20 | 0.0578 (bf16-compile-cg) · f32 6.19 |
| pj_m128_n4096_k4096 | 0.0532 (bf16) · f32 2.62 | 0.085 (bf16-compile-ro) · f32 2.84 | 0.0563 (bf16-compile-ro) · f32 8.38 | 0.0625 (triton-bf16) · f32 2.16 | 0.0238 (bf16-compile-ro) · f32 5.67 |
| pj_m128_n6144_k4096 | 0.086 (bf16-compile-cg) · f32 2.39 | 0.128 (bf16-compile-cg) · f32 2.63 | 0.0676 (bf16-compile-cg) · f32 11.13 | 0.0901 (bf16-compile-cg) · f32 2.58 | 0.0323 (bf16-compile) · f32 5.78 |
| pj_m16_n18432_k7168 | 0.3092 (bf16-compile) · f32 1.94 | 0.307 (bf16-compile) · f32 2.06 | 0.2171 (bf16-compile) · f32 2.00 | 0.379 (bf16-compile-cg) · f32 1.93 | 0.1017 (bf16-compile) · f32 2.04 |
| pj_m16_n2048_k7168 | 0.0389 (bf16-compile) · f32 1.07 | 0.042 (bf16-compile-cg) · f32 2.46 | 0.043 (bf16-compile-cg) · f32 2.37 | 0.0532 (bf16-compile) · f32 1.01 | 0.022 (bf16-compile-ro) · f32 2.73 |
| pj_m16_n28672_k4096 | 0.2744 (bf16-compile-cg) · f32 1.98 | 0.2744 (bf16-compile) · f32 2.14 | 0.1864 (bf16-compile-cg) · f32 2.22 | 0.3502 (bf16-compile) · f32 1.82 | 0.0901 (bf16-compile) · f32 2.16 |
| pj_m16_n4096_k14336 | 0.1393 (bf16-compile-cg) · f32 1.86 | 0.1434 (bf16-compile) · f32 2.06 | 0.1096 (bf16-compile-cg) · f32 2.05 | 0.1946 (bf16-compile-cg) · f32 1.87 | 0.0553 (bf16) · f32 2.23 |
| pj_m16_n4096_k4096 | 0.043 (bf16-compile-cg) · f32 1.15 | 0.0471 (bf16-compile-cg) · f32 2.53 | 0.043 (bf16-compile-cg) · f32 2.25 | 0.0594 (bf16-compile) · f32 0.96 | 0.0238 (bf16-compile-ro) · f32 2.29 |
| pj_m16_n6144_k4096 | 0.0635 (bf16-compile) · f32 1.54 | 0.0666 (bf16-compile-cg) · f32 2.24 | 0.0594 (bf16-compile-cg) · f32 2.10 | 0.087 (bf16-compile) · f32 1.53 | 0.0288 (bf16-compile) · f32 2.50 |
| pj_m1_n18432_k7168 | 0.3092 (bf16-compile-ro) · f32 1.89 | 0.3185 (bf16-compile-cg) · f32 1.92 | 0.212 (bf16-compile) · f32 1.96 | 0.34 (bf16-compile) · f32 2.10 | 0.1019 (bf16-compile-ro) · f32 1.92 |
| pj_m1_n2048_k7168 | 0.043 (bf16-compile) · f32 0.59 | 0.043 (bf16-compile-cg) · f32 2.01 | 0.0379 (bf16-compile-cg) · f32 1.91 | 0.0573 (bf16-compile-cg) · f32 0.56 | 0.0207 (bf16-compile-cg) · f32 1.94 |
| pj_m1_n28672_k4096 | 0.2765 (bf16-compile) · f32 1.88 | 0.2785 (bf16-compile-cg) · f32 1.97 | 0.1833 (bf16-compile) · f32 2.07 | 0.3052 (bf16-compile-cg) · f32 2.10 | 0.0911 (bf16) · f32 1.94 |
| pj_m1_n4096_k14336 | 0.1413 (bf16) · f32 1.76 | 0.1464 (bf16) · f32 1.93 | 0.1075 (bf16-compile) · f32 1.84 | 0.17 (bf16-compile-cg) · f32 1.93 | 0.0525 (bf16-compile-cg) · f32 1.94 |
| pj_m1_n4096_k4096 | 0.0492 (bf16-compile) · f32 0.80 | 0.0492 (bf16-compile-cg) · f32 1.92 | 0.0451 (bf16-compile) · f32 1.73 | 0.0635 (bf16-compile-cg) · f32 0.54 | 0.0218 (bf16-compile-ro) · f32 1.95 |
| pj_m1_n6144_k4096 | 0.0635 (bf16-compile-cg) · f32 1.30 | 0.0676 (bf16-compile) · f32 1.98 | 0.0532 (bf16-compile-cg) · f32 1.91 | 0.0891 (bf16-compile-cg) · f32 1.06 | 0.0287 (bf16-compile-cg) · f32 1.86 |
| pj_m2048_n28672_k4096 | 4 (bf16-compile-cg) · f32 2.94 | 7.199 (bf16-compile) · f32 2.69 | 2.159 (bf16-compile-ro) · f32 16.39 | 2.757 (triton-bf16) · f32 3.48 | 0.6085 (bf16-compile-ro) · f32 16.44 |
| pj_m2048_n4096_k14336 | 2.094 (bf16-compile-cg) · f32 3.07 | 4.332 (triton-bf16) · f32 2.53 | 1.073 (bf16) · f32 16.79 | 1.392 (triton-bf16) · f32 3.04 | 0.2978 (bf16-compile-cg) · f32 17.35 |
| pj_m2048_n4096_k4096 | 0.598 (bf16-compile-cg) · f32 3.06 | 1.099 (bf16-compile-cg) · f32 2.63 | 0.3389 (bf16) · f32 15.38 | 0.4037 (triton-bf16) · f32 3.06 | 0.0917 (bf16-compile) · f32 16.34 |
| pj_m2048_n6144_k4096 | 0.8868 (bf16-compile-ro) · f32 3.00 | 1.592 (bf16-compile-ro) · f32 2.86 | 0.425 (bf16-compile-cg) · f32 18.15 | 0.6001 (triton-bf16) · f32 3.10 | 0.1367 (bf16-compile) · f32 15.95 |
| llm_blkdec_b1_l4096 | 0.5625 (bf16-compile) · f32 1.90 · bf16w 1.11 | 0.6717 (bf16-compile) · f32 1.68 · bf16w 1.05 | 0.4715 (bf16-compile) · f32 2.02 · bf16w 1.68 | 0.7531 (bf16-compile-cg) · f32 1.56 · bf16w 0.94 | 0.2852 (bf16-compile-cg) · f32 1.94 · bf16w 1.73 |
| llm_blkpre_t2048 | 8.103 (bf16-compile) · f32 3.11 · bf16w 3.15 | 15.62 (bf16-compile-ro) · f32 3.75 · bf16w 3.19 | 4.833 (bf16-compile) · f32 15.60 · bf16w 11.84 | 5.547 (bf16-compile-cg) · f32 3.22 · bf16w 3.07 | 1.308 (bf16-compile-cg) · f32 16.20 · bf16w 16.88 |
| llm_dec_b1_l2048 | 0.1461 (bf16-compile) · f32 1.64 · bf16w 1.01 | 0.2251 (bf16-compile) · f32 1.20 · bf16w 0.81 | 0.1757 (bf16-compile) · f32 1.22 · bf16w 0.93 | 0.3062 (bf16-compile-cg) · f32 0.88 · bf16w 0.59 | 0.1446 (bf16-compile-cg) · f32 0.76 · bf16w 0.61 |
| llm_dec_b1_l8192 | 0.1874 (bf16-compile) · f32 1.61 · bf16w 1.15 | 0.2984 (bf16-compile) · f32 1.13 · bf16w 0.84 | 0.2233 (bf16-compile) · f32 1.20 · bf16w 0.97 | 0.35 (bf16-compile-cg) · f32 0.98 · bf16w 0.73 | 0.151 (bf16-compile-cg) · f32 0.92 · bf16w 0.76 |
| llm_dec_b32_l2048 | 0.4665 (bf16-compile) · f32 1.93 · bf16w 1.73 | 0.7252 (bf16-cudnn) · f32 1.36 · bf16w 1.32 | 0.5251 (bf16-cudnn) · f32 1.51 · bf16w 1.41 | 0.6008 (bf16-compile) · f32 1.69 · bf16w 1.50 | 0.2288 (bf16-compile-ro) · f32 1.55 · bf16w 1.60 |
| llm_dec_b32_l8192 | 1.468 (bf16-cudnn) · f32 1.89 · bf16w 1.92 | 1.692 (bf16-cudnn) · f32 1.86 · bf16w 1.85 | 1.102 (bf16-cudnn) · f32 1.91 · bf16w 1.87 | 1.645 (bf16-compile) · f32 1.96 · bf16w 1.90 | 0.489 (bf16-compile-ro) · f32 2.10 · bf16w 2.10 |
| llm_dec_b8_l2048 | 0.22 (bf16-compile) · f32 1.76 · bf16w 1.34 | 0.3467 (bf16-compile) · f32 2.41 · bf16w 1.51 | 0.2702 (bf16-compile-cg) · f32 1.33 · bf16w 1.15 | 0.351 (bf16-compile) · f32 1.23 · bf16w 0.99 | 0.1653 (bf16-compile-ro) · f32 1.09 · bf16w 0.94 |
| llm_dec_b8_l8192 | 0.5445 (fa2-bf16) · f32 1.59 · bf16w 1.42 | 0.7327 (liger-bf16) · f32 1.89 · bf16w 1.47 | 0.5114 (fa2-bf16) · f32 1.34 · bf16w 1.24 | 0.6157 (bf16-compile) · f32 1.62 · bf16w 1.47 | 0.2302 (bf16-compile-cg) · f32 1.54 · bf16w 1.46 |
| llm_pre_t2048 | 1.931 (bf16-compile-cg) · f32 3.34 · bf16w 3.30 | 3.742 (bf16-compile-ro) · f32 2.97 · bf16w 2.90 | 1.082 (bf16-compile-cg) · f32 17.81 · bf16w 15.03 | 1.324 (bf16-compile) · f32 3.59 · bf16w 3.47 | 0.3497 (bf16-compile-ro) · f32 17.11 · bf16w 17.53 |
| llm_pre_t8192 | 11.48 (bf16-compile) · f32 3.89 | 21.3 (bf16-compile-cg) · f32 4.90 | 7.143 (bf16-compile) · f32 29.21 | 7.803 (bf16-compile-cg) · f32 4.46 | 1.92 (bf16-compile) · f32 45.09 |

## Verdict flips vs the mixed-precision gap tables

gap.py's rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables (gap-<part>.md, gap.md) row for row. Rows a part's gap table did not cover (the RTX 5080's model-scale rows, reported in model-scale.md) are judged against that rule applied to the same data.

| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |
|---|---|---|---|---|
| rtx-5080 | attn_2048_64 | 0.0768 (torch-sdpa-compile) 1.05 even | 0.0696 (torch-compile-ro) 1.15 loss | other |
| rtx-5080 | llm_blkdec_b1_l4096 | 0.5625 (torch-bf16-compile) 1.90 loss | 1.089 (torch-compile) 0.98 even | old best a bf16 leg |
| rtx-5080 | llm_blkpre_t2048 | 8.103 (torch-bf16-compile) 3.11 loss | 26.83 (liger) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_blkpre_t2048_tf32 | 18.96 (torch-compile-tf32) 1.04 even | 18.08 (triton-tf32) 1.09 loss | best now a gemm_lib leg |
| rtx-5080 | llm_dec_b1_l2048 | 0.1461 (torch-bf16-compile) 1.64 loss | 0.259 (torch-compile-ro) 0.92 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l8192 | 0.1874 (torch-bf16-compile) 1.61 loss | 0.3462 (torch-compile) 0.87 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l2048 | 0.4665 (torch-bf16-compile) 1.93 loss | 2.293 (torch-compile) 0.39 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l8192 | 1.506 (torch-bf16-compile) 1.85 loss | 9.728 (torch-compile-cg) 0.29 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l2048 | 0.22 (torch-bf16-compile) 1.76 loss | 0.7657 (torch-compile-cg) 0.51 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l8192 | 0.5883 (torch-bf16) 1.48 loss | 2.513 (torch-compile-cg) 0.35 win | old best a bf16 leg |
| rtx-5080 | llm_pre_t2048 | 1.937 (torch-bf16-compile) 3.33 loss | 7.216 (liger) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_pre_t2048_tf32 | 6.117 (torch-compile-tf32) 0.87 win | 4.543 (triton-tf32) 1.18 loss | best now a gemm_lib leg |
| rtx-5080 | llm_pre_t8192 | 11.48 (torch-bf16-compile) 3.89 loss | 52.89 (liger) 0.84 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | lsm_1_131072 | 0.0246 (torch-sdpa-compile) 0.71 win | 0.0225 (torch-compile-cg) 0.78 even | other |
| rtx-5080 | pj_m128_n4096_k4096 | 0.0532 (torch-bf16) 2.62 loss | 0.1352 (cublas-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n18432_k7168 | 0.3092 (torch-bf16-compile) 1.94 loss | 0.5898 (cublas-fresh) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n2048_k7168 | 0.0389 (torch-bf16-compile) 1.07 loss | 0.0492 (cublas-fresh) 0.85 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n28672_k4096 | 0.2765 (torch-bf16-compile) 1.96 loss | 0.5243 (lt-fresh) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n4096_k14336 | 0.1413 (torch-bf16-compile) 1.84 loss | 0.2765 (torch-compile) 0.94 win | old best a bf16 leg |
| rtx-5080 | pj_m16_n4096_k4096 | 0.0451 (torch-bf16-compile) 1.10 loss | 0.0553 (cublas-fresh) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n18432_k7168 | 0.3113 (torch-bf16) 1.88 loss | 0.5571 (lt-fresh) 1.05 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n2048_k7168 | 0.043 (torch-bf16-compile) 0.59 win | 0.0266 (cublas-fresh) 0.95 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n28672_k4096 | 0.2765 (torch-bf16-compile) 1.88 loss | 0.5018 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n4096_k4096 | 0.0492 (torch-bf16-compile) 0.80 win | 0.0225 (lt-fresh) 1.76 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096 | 4.111 (torch-bf16) 2.86 loss | 12.2 (lt-fresh) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k14336 | 2.099 (torch-bf16-compile) 3.06 loss | 6.415 (cublas-fresh) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k4096 | 0.6021 (torch-bf16-compile) 3.04 loss | 1.774 (torch-eager) 1.03 even | old best a bf16 leg |
| rtx-5080 | pj_m2048_n6144_k4096 | 0.8888 (torch-bf16-compile) 2.99 loss | 2.616 (torch-eager) 1.02 even | old best a bf16 leg |
| rtx-3090 | llm_blkdec_b1_l4096 | 0.6717 (torch-bf16-compile) 1.68 loss | 1.216 (torch-compile) 0.93 win | old best a bf16 leg |
| rtx-3090 | llm_blkpre_t2048_tf32 | 39.35 (torch-eager-tf32) 0.84 win | 31.79 (triton-tf32) 1.05 even | best now a gemm_lib leg |
| rtx-3090 | llm_dec_b1_l2048 | 0.2251 (torch-bf16-compile) 1.20 loss | 0.3477 (torch-compile) 0.77 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l8192 | 0.2984 (torch-bf16-compile) 1.13 loss | 0.4481 (torch-compile) 0.75 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l2048 | 0.913 (torch-bf16-compile) 1.08 loss | 2.989 (torch-compile-ro) 0.33 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192 | 3.224 (torch-bf16) 0.98 even | 11.33 (torch-compile-ro) 0.28 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192_bf16w | 3.211 (torch-bf16) 0.98 even | 1.683 (torch-bf16-cudnn) 1.86 loss | other |
| rtx-3090 | llm_dec_b8_l2048 | 0.3467 (torch-bf16-compile) 2.41 loss | 0.9394 (torch-compile) 0.89 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l8192 | 0.7336 (torch-bf16) 1.88 loss | 2.979 (torch-compile-cg) 0.46 win | old best a bf16 leg |
| rtx-3090 | llm_pre_t2048 | 4.146 (torch-bf16-compile) 2.68 loss | 13.59 (liger) 0.82 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | llm_pre_t2048_tf32 | 11.39 (torch-compile-tf32) 0.83 win | 8.418 (triton-tf32) 1.12 loss | best now a gemm_lib leg |
| rtx-3090 | lsm_1_131072 | 0.0307 (torch-sdpa-compile) 0.93 win | 0.0266 (torch-compile-ro) 1.07 loss | other |
| rtx-3090 | mlp_2048_tf32 | 1.422 (torch-eager-tf32) 1.05 even | 1.197 (torch-compile-ro-tf32) 1.24 loss | other |
| rtx-3090 | pj_m128_n28672_k4096 | 0.4833 (torch-bf16-compile) 2.48 loss | 1.345 (torch-compile-ro) 0.89 win | old best a bf16 leg |
| rtx-3090 | pj_m128_n4096_k14336 | 0.2806 (torch-bf16-compile) 2.49 loss | 0.6769 (torch-compile-ro) 1.03 even | old best a bf16 leg |
| rtx-3090 | pj_m128_n6144_k4096_tf32 | 0.2847 (torch-compile-tf32) 0.97 even | 0.256 (lt-tf32) 1.08 loss | best now a gemm_lib leg |
| rtx-3090 | pj_m16_n18432_k7168 | 0.307 (torch-bf16-compile) 2.06 loss | 0.6134 (torch-compile) 1.03 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n18432_k7168 | 0.3236 (torch-bf16-compile) 1.89 loss | 0.6052 (cublas) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m1_n28672_k4096 | 0.2806 (torch-bf16-compile) 1.95 loss | 0.5417 (cublas) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m1_n4096_k14336 | 0.1464 (torch-bf16) 1.93 loss | 0.2785 (cublas) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n28672_k4096 | 7.199 (torch-bf16-compile) 2.69 loss | 23.32 (torch-compile-ro) 0.83 win | old best a bf16 leg |
| rtx-3090 | pj_m2048_n4096_k14336 | 4.527 (torch-bf16-compile) 2.42 loss | 12.14 (lt-fresh) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k4096 | 1.1 (torch-bf16-compile) 2.63 loss | 3.478 (torch-compile) 0.83 win | old best a bf16 leg |
| rtx-3090 | pj_m2048_n6144_k4096 | 1.786 (torch-bf16) 2.55 loss | 5.074 (lt-fresh) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | attn_1024_64 | 0.0604 (chain-fused) 1.02 even | 0.0553 (torch-sdpa-compile-cg) 1.12 loss | other |
| a100 | attn_2048_64 | 0.1444 (torch-compile) 1.01 even | 0.1178 (torch-sdpa-compile-cg) 1.23 loss | other |
| a100 | llm_dec_b1_l2048 | 0.1757 (torch-bf16-compile) 1.22 loss | 0.2671 (torch-compile) 0.80 win | old best a bf16 leg |
| a100 | llm_dec_b1_l8192 | 0.2233 (torch-bf16-compile) 1.20 loss | 0.4154 (torch-compile) 0.65 win | old best a bf16 leg |
| a100 | llm_dec_b32_l2048 | 0.5267 (torch-bf16-compile) 1.50 loss | 1.962 (torch-compile-cg) 0.40 win | old best a bf16 leg |
| a100 | llm_dec_b32_l8192 | 1.53 (torch-bf16) 1.38 loss | 7.043 (torch-compile-ro) 0.30 win | old best a bf16 leg |
| a100 | llm_dec_b8_l2048 | 0.2783 (torch-bf16-compile) 1.29 loss | 0.695 (torch-compile-cg) 0.52 win | old best a bf16 leg |
| a100 | llm_dec_b8_l8192 | 0.5114 (torch-bf16) 1.34 loss | 2.098 (torch-compile-cg) 0.33 win | old best a bf16 leg |
| a100 | lsm_1_131072 | 0.0369 (torch-compile) 0.81 win | 0.0297 (torch-compile-cg) 1.00 even | other |
| a100 | lsm_4_32768 | 0.0256 (torch-compile) 0.89 win | 0.0195 (torch-compile-cg) 1.16 loss | other |
| a100 | pj_m16_n18432_k7168 | 0.2171 (torch-bf16-compile) 2.00 loss | 0.4413 (torch-compile-cg) 0.98 even | old best a bf16 leg |
| a100 | pj_m16_n28672_k4096 | 0.1884 (torch-bf16-compile) 2.20 loss | 0.4024 (torch-compile-cg) 1.03 even | old best a bf16 leg |
| a100 | pj_m16_n4096_k14336 | 0.1106 (torch-bf16-compile) 2.03 loss | 0.256 (lt) 0.88 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | attn_1024_64 | 0.0328 (torch-sdpa-compile) 0.83 win | 0.0287 (torch-sdpa-compile-cg) 0.95 even | other |
| rtx-4090 | llm_blkdec_b1_l4096 | 0.7938 (torch-bf16-compile) 1.48 loss | 1.286 (torch-compile-cg) 0.91 win | old best a bf16 leg |
| rtx-4090 | llm_blkpre_t2048 | 5.836 (torch-bf16-compile) 3.06 loss | 18.04 (liger) 0.99 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_dec_b32_l2048 | 0.6008 (torch-bf16-compile) 1.69 loss | 2.569 (torch-compile) 0.39 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l8192 | 1.645 (torch-bf16-compile) 1.96 loss | 10.5 (torch-compile) 0.31 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l2048 | 0.351 (torch-bf16-compile) 1.23 loss | 0.8825 (torch-compile) 0.49 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l8192 | 0.6157 (torch-bf16-compile) 1.62 loss | 2.873 (torch-compile) 0.35 win | old best a bf16 leg |
| rtx-4090 | llm_pre_t2048 | 1.324 (torch-bf16-compile) 3.59 loss | 5.164 (liger) 0.92 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_pre_t2048_tf32 | 5.124 (torch-compile-tf32) 0.79 win | 3.276 (triton-tf32) 1.24 loss | best now a gemm_lib leg |
| rtx-4090 | llm_pre_t8192 | 7.807 (torch-bf16-compile) 4.46 loss | 37.53 (liger) 0.93 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n28672_k4096 | 0.3594 (torch-bf16-compile) 1.92 loss | 0.7096 (lt-fresh) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n4096_k4096_tf32 | 0.1219 (torch-compile-tf32) 0.85 win | 0.0655 (cublas-tf32-fresh) 1.59 loss | best now a gemm_lib leg |
| rtx-4090 | pj_m16_n18432_k7168 | 0.3809 (torch-bf16-compile) 1.92 loss | 0.6973 (torch-compile-cg) 1.05 even | old best a bf16 leg |
| rtx-4090 | pj_m16_n2048_k7168 | 0.0532 (torch-bf16-compile) 1.01 even | 0.0297 (cublas-fresh) 1.80 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m16_n28672_k4096 | 0.3502 (torch-bf16-compile) 1.82 loss | 0.6277 (cublas-fresh) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m16_n4096_k4096 | 0.0594 (torch-bf16-compile) 0.96 even | 0.0338 (lt-fresh) 1.69 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n2048_k7168 | 0.0583 (torch-bf16-compile) 0.55 win | 0.0273 (lt-fresh) 1.18 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n4096_k4096 | 0.0654 (torch-bf16-compile) 0.53 win | 0.0195 (lt-fresh) 1.77 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n6144_k4096 | 0.0901 (torch-bf16-compile) 1.05 even | 0.0645 (lt-fresh) 1.46 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k14336 | 1.483 (torch-bf16-compile) 2.86 loss | 4.186 (cublas) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n6144_k4096 | 0.6042 (torch-bf16-compile) 3.08 loss | 1.789 (cublas) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | rt_mlp2_1024 | 1.612 (torch-compile) 1.04 even | 1.519 (torch-compile-ro) 1.10 loss | other |
| rtx-4090 | smlp_4096_256_1024 | 0.1065 (torch-compile) 1.03 even | 0.1044 (triton) 1.05 loss | best now a gemm_lib leg |
| rtx-4090 | softmax_rows_4096 | 2.497 (torch-compile) 1.04 even | 2.46 (torch-compile-ro) 1.06 loss | other |
| h100 | llm_dec_b32_l2048 | 0.316 (torch-bf16-compile) 1.12 loss | 1.163 (torch-compile) 0.31 win | old best a bf16 leg |
| h100 | llm_dec_b32_l8192 | 0.5296 (torch-bf16-compile) 1.94 loss | 3.931 (torch-compile-cg) 0.26 win | old best a bf16 leg |
| h100 | llm_dec_b8_l2048_bf16w | 0.2247 (torch-bf16-compile) 0.69 win | 0.1615 (torch-bf16-compile-cg) 0.96 even | other |
| h100 | llm_dec_b8_l8192 | 0.2768 (torch-bf16-compile) 1.28 loss | 1.114 (torch-compile) 0.32 win | old best a bf16 leg |
| h100 | pj_m16_n18432_k7168 | 0.1017 (torch-bf16-compile) 2.04 loss | 0.2057 (torch-compile) 1.01 even | old best a bf16 leg |
| h100 | rt_mlp2_1024 | 2.047 (torch-compile) 1.03 even | 1.96 (torch-compile-ro) 1.08 loss | other |

## What the new legs change

Rows whose best leg is one the competitors step added (CUDA-graph torch.compile, SDPA backends, hand Triton, Liger, FlashAttention-2/3, FlashInfer), against the best over the legs before it (`base`).

| part | workload | prec | base: best (leg) ratio | now: best (leg) ratio | verdict |
|---|---|---|---|---|---|
| rtx-5080 | epi_4096 | f32 | 3.727 (chain-lt) 1.02 | 3.724 (torch-compile-ro) 1.02 | even → even |
| rtx-5080 | pj_m128_n28672_k4096_bf16 | bf16 | 0.3236 (torch-bf16-compile) 1.24 | 0.3174 (triton-bf16) 1.27 | loss → loss |
| rtx-5080 | pj_m128_n4096_k14336_bf16 | bf16 | 0.172 (torch-bf16) 1.28 | 0.1717 (torch-bf16-compile-ro) 1.29 | loss → loss |
| rtx-5080 | pj_m128_n6144_k4096_bf16 | bf16 | 0.0881 (torch-bf16) 0.99 | 0.086 (torch-bf16-compile-cg) 1.02 | even → even |
| rtx-5080 | pj_m16_n28672_k4096_bf16 | bf16 | 0.2765 (torch-bf16-compile) 1.01 | 0.2744 (torch-bf16-compile-cg) 1.02 | even → even |
| rtx-5080 | pj_m16_n4096_k14336_bf16 | bf16 | 0.1413 (torch-bf16-compile) 0.83 | 0.1393 (torch-bf16-compile-cg) 0.85 | win → win |
| rtx-5080 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0451 (torch-bf16-compile) 0.93 | 0.043 (torch-bf16-compile-cg) 0.98 | win → even |
| rtx-5080 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0655 (torch-bf16-compile) 0.43 | 0.0635 (torch-bf16-compile-cg) 0.45 | win → win |
| rtx-5080 | pj_m2048_n28672_k4096_bf16 | bf16 | 4.111 (torch-bf16) 1.11 | 4 (torch-bf16-compile-cg) 1.14 | loss → loss |
| rtx-5080 | pj_m2048_n4096_k14336_bf16 | bf16 | 2.099 (torch-bf16-compile) 1.08 | 2.094 (torch-bf16-compile-cg) 1.08 | loss → loss |
| rtx-5080 | pj_m2048_n4096_k4096_bf16 | bf16 | 0.6021 (torch-bf16-compile) 1.12 | 0.598 (torch-bf16-compile-cg) 1.13 | loss → loss |
| rtx-5080 | pj_m2048_n6144_k4096_bf16 | bf16 | 0.8888 (torch-bf16-compile) 1.06 | 0.8868 (torch-bf16-compile-ro) 1.06 | loss → loss |
| rtx-5080 | mlp2_2048_tf32 | tf32 | 1.547 (torch-eager-tf32) 1.01 | 1.513 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-5080 | mlp_2048_tf32 | tf32 | 0.7629 (torch-eager-tf32) 1.03 | 0.7598 (torch-compile-ro-tf32) 1.03 | even → even |
| rtx-5080 | attn_1024_64 | f32 | 0.0297 (torch-compile) 1.24 | 0.0287 (torch-compile-cg) 1.28 | loss → loss |
| rtx-5080 | attn_2048_128 | f32 | 0.0973 (torch-sdpa-compile) 1.20 | 0.0942 (torch-sdpa-compile-cg) 1.24 | loss → loss |
| rtx-5080 | attn_2048_64 | f32 | 0.0768 (torch-sdpa-compile) 1.05 | 0.0696 (torch-compile-ro) 1.15 | even → loss |
| rtx-5080 | causal_1024_256 | f32 | 57.21 (torch-compile) 0.27 | 44.09 (torch-compile-ro) 0.34 | win → win |
| rtx-5080 | gen_1024_512_256 | f32 | 28.12 (torch-compile) 0.28 | 21.4 (torch-compile-ro) 0.36 | win → win |
| rtx-5080 | lsm_1_131072 | f32 | 0.0246 (torch-sdpa-compile) 0.71 | 0.0225 (torch-compile-cg) 0.78 | win → even |
| rtx-5080 | prefill_1024_512_256 | f32 | 29.3 (torch-compile) 0.27 | 22.57 (torch-compile-cg) 0.34 | win → win |
| rtx-5080 | softmax_rows_4096 | f32 | 3.559 (torch-compile) 1.12 | 3.54 (torch-compile-ro) 1.12 | loss → loss |
| rtx-5080 | llm_blkpre_t2048 | f32 | 31.93 (torch-compile) 0.79 | 26.83 (liger) 0.94 | win → win |
| rtx-5080 | llm_blkpre_t2048_tf32 | tf32 | 18.96 (torch-compile-tf32) 1.04 | 18.08 (triton-tf32) 1.09 | even → loss |
| rtx-5080 | llm_dec_b1_l2048 | f32 | 0.2636 (torch-compile) 0.91 | 0.259 (torch-compile-ro) 0.92 | win → win |
| rtx-5080 | llm_dec_b32_l8192 | f32 | 9.767 (torch-compile) 0.28 | 9.728 (torch-compile-cg) 0.29 | win → win |
| rtx-5080 | llm_dec_b8_l2048 | f32 | 0.7713 (torch-compile) 0.50 | 0.7657 (torch-compile-cg) 0.51 | win → win |
| rtx-5080 | llm_dec_b8_l8192 | f32 | 2.964 (torch-compile) 0.29 | 2.513 (torch-compile-cg) 0.35 | win → win |
| rtx-5080 | llm_pre_t2048 | f32 | 8.448 (torch-compile) 0.76 | 7.216 (liger) 0.90 | win → win |
| rtx-5080 | llm_pre_t2048_bf16 | bf16 | 1.937 (torch-bf16-compile) 1.37 | 1.931 (torch-bf16-compile-cg) 1.37 | loss → loss |
| rtx-5080 | llm_pre_t2048_bf16w | bf16w | 1.937 (torch-bf16-compile) 3.29 | 1.931 (torch-bf16-compile-cg) 3.30 | loss → loss |
| rtx-5080 | llm_pre_t2048_tf32 | tf32 | 6.117 (torch-compile-tf32) 0.87 | 4.543 (triton-tf32) 1.18 | win → loss |
| rtx-5080 | llm_pre_t8192 | f32 | 85.07 (torch-compile) 0.52 | 52.89 (liger) 0.84 | win → win |
| rtx-5080 | ew_4096 | f32 | 0.2447 (torch-compile) 0.36 | 0.2437 (torch-compile-cg) 0.37 | win → win |
| rtx-5080 | ewconvew_2048_5 | f32 | 0.1884 (torch-compile) 0.41 | 0.1881 (torch-compile-cg) 0.41 | win → win |
| rtx-5080 | diamond_2048 | f32 | 1.368 (torch-compile) 1.01 | 1.366 (torch-compile-cg) 1.02 | even → even |
| rtx-5080 | loopmm_1024_8 | f32 | 0.5908 (torch-compile) 1.04 | 0.5878 (torch-compile-cg) 1.05 | even → even |
| rtx-5080 | rt_mlp2_1024 | f32 | 2.538 (torch-compile) 1.08 | 2.516 (torch-compile-cg) 1.09 | loss → loss |
| rtx-5080 | twin_2048 | f32 | 0.9196 (torch-compile) 1.01 | 0.9175 (torch-compile-cg) 1.01 | even → even |
| rtx-3090 | epi_4096 | f32 | 7.085 (torch-compile) 0.84 | 7.084 (torch-compile-ro) 0.84 | win → win |
| rtx-3090 | pj_m128_n28672_k4096 | f32 | 1.986 (cublas) 0.60 | 1.345 (torch-compile-ro) 0.89 | win → win |
| rtx-3090 | pj_m128_n28672_k4096_bf16 | bf16 | 0.4833 (torch-bf16-compile) 1.08 | 0.4669 (torch-bf16-compile-cg) 1.12 | loss → loss |
| rtx-3090 | pj_m128_n28672_k4096_tf32 | tf32 | 0.9626 (torch-compile-tf32) 1.12 | 0.8735 (torch-compile-ro-tf32) 1.23 | loss → loss |
| rtx-3090 | pj_m128_n4096_k14336 | f32 | 0.683 (torch-compile) 1.02 | 0.6769 (torch-compile-ro) 1.03 | even → even |
| rtx-3090 | pj_m128_n4096_k4096_bf16 | bf16 | 0.086 (torch-bf16) 1.39 | 0.085 (torch-bf16-compile-ro) 1.40 | loss → loss |
| rtx-3090 | pj_m128_n4096_k4096_tf32 | tf32 | 0.1709 (torch-compile-tf32) 1.28 | 0.17 (torch-compile-cg-tf32) 1.29 | loss → loss |
| rtx-3090 | pj_m128_n6144_k4096 | f32 | 0.3226 (torch-compile) 1.05 | 0.3205 (torch-compile-cg) 1.05 | even → loss |
| rtx-3090 | pj_m128_n6144_k4096_bf16 | bf16 | 0.129 (torch-bf16) 1.37 | 0.128 (torch-bf16-compile-cg) 1.39 | loss → loss |
| rtx-3090 | pj_m16_n4096_k4096 | f32 | 0.086 (torch-compile) 1.38 | 0.085 (torch-compile-cg) 1.40 | loss → loss |
| rtx-3090 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0481 (torch-bf16-compile) 1.68 | 0.0471 (torch-bf16-compile-cg) 1.71 | loss → loss |
| rtx-3090 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0676 (torch-bf16-compile) 1.40 | 0.0666 (torch-bf16-compile-cg) 1.42 | loss → loss |
| rtx-3090 | pj_m1_n28672_k4096_bf16 | bf16 | 0.2806 (torch-bf16-compile) 1.02 | 0.2785 (torch-bf16-compile-cg) 1.03 | even → even |
| rtx-3090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0502 (torch-bf16) 1.24 | 0.0492 (torch-bf16-compile-cg) 1.26 | loss → loss |
| rtx-3090 | pj_m2048_n28672_k4096 | f32 | 23.6 (torch-compile) 0.82 | 23.32 (torch-compile-ro) 0.83 | win → win |
| rtx-3090 | pj_m2048_n28672_k4096_tf32 | tf32 | 14.77 (lt-tf32-fresh) 1.01 | 14.76 (torch-compile-cg-tf32) 1.01 | even → even |
| rtx-3090 | pj_m2048_n4096_k14336_bf16 | bf16 | 4.527 (torch-bf16-compile) 0.87 | 4.332 (triton-bf16) 0.91 | win → win |
| rtx-3090 | pj_m2048_n4096_k4096_bf16 | bf16 | 1.1 (torch-bf16-compile) 1.14 | 1.099 (torch-bf16-compile-cg) 1.14 | loss → loss |
| rtx-3090 | pj_m2048_n4096_k4096_tf32 | tf32 | 2.508 (cublas-tf32-fresh) 1.03 | 2.489 (torch-compile-ro-tf32) 1.04 | even → even |
| rtx-3090 | pj_m2048_n6144_k4096_bf16 | bf16 | 1.786 (torch-bf16) 0.91 | 1.592 (torch-bf16-compile-ro) 1.02 | win → even |
| rtx-3090 | mlp2_2048_tf32 | tf32 | 2.843 (torch-eager-tf32) 1.06 | 2.674 (torch-compile-ro-tf32) 1.13 | loss → loss |
| rtx-3090 | mlp_2048_tf32 | tf32 | 1.422 (torch-eager-tf32) 1.05 | 1.197 (torch-compile-ro-tf32) 1.24 | even → loss |
| rtx-3090 | mlp_512 | f32 | 0.0512 (torch-compile) 1.25 | 0.0451 (triton) 1.42 | loss → loss |
| rtx-3090 | attn_1024_64 | f32 | 0.0502 (torch-compile) 1.28 | 0.0461 (torch-compile-ro) 1.39 | loss → loss |
| rtx-3090 | attn_2048_128 | f32 | 0.1833 (torch-compile) 1.72 | 0.1802 (torch-compile-cg) 1.75 | loss → loss |
| rtx-3090 | attn_2048_64 | f32 | 0.1331 (torch-compile) 1.42 | 0.1311 (torch-sdpa-compile-cg) 1.44 | loss → loss |
| rtx-3090 | causal_1024_256 | f32 | 159.8 (torch-compile) 0.16 | 138.1 (torch-compile-cg) 0.18 | win → win |
| rtx-3090 | gen_1024_512_256 | f32 | 77.77 (torch-compile) 0.16 | 67.65 (torch-compile-cg) 0.19 | win → win |
| rtx-3090 | lsm_1_131072 | f32 | 0.0307 (torch-sdpa-compile) 0.93 | 0.0266 (torch-compile-ro) 1.07 | win → loss |
| rtx-3090 | lsm_32_8192 | f32 | 0.0113 (torch-sdpa-compile) 2.62 | 0.0102 (torch-compile-cg) 2.91 | loss → loss |
| rtx-3090 | lsm_4_32768 | f32 | 0.0215 (torch-sdpa-compile) 1.13 | 0.0174 (torch-compile-ro) 1.39 | loss → loss |
| rtx-3090 | prefill_1024_512_256 | f32 | 79 (torch-compile) 0.16 | 69.86 (torch-compile-cg) 0.19 | win → win |
| rtx-3090 | rowsum_2048 | f32 | 0.7782 (torch-compile) 1.06 | 0.7772 (torch-compile-ro) 1.07 | loss → loss |
| rtx-3090 | softmax_rows_2048 | f32 | 0.7434 (torch-compile) 1.16 | 0.7424 (torch-compile-cg) 1.16 | loss → loss |
| rtx-3090 | llm_blkpre_t2048 | f32 | 52.87 (torch-compile) 1.11 | 50.15 (liger) 1.17 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_bf16 | bf16 | 16.19 (torch-bf16-compile) 1.00 | 15.62 (torch-bf16-compile-ro) 1.04 | even → even |
| rtx-3090 | llm_blkpre_t2048_bf16w | bf16w | 16.19 (torch-bf16-compile) 3.07 | 15.62 (torch-bf16-compile-ro) 3.19 | loss → loss |
| rtx-3090 | llm_blkpre_t2048_tf32 | tf32 | 39.35 (torch-eager-tf32) 0.84 | 31.79 (triton-tf32) 1.05 | win → even |
| rtx-3090 | llm_dec_b32_l2048 | f32 | 2.992 (torch-compile) 0.33 | 2.989 (torch-compile-ro) 0.33 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16 | bf16 | 0.9067 (torch-bf16-compile) 0.74 | 0.74 (torch-bf16-cudnn) 0.91 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.9067 (torch-bf16-compile) 0.74 | 0.74 (torch-bf16-cudnn) 0.91 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16w | bf16w | 0.9067 (torch-bf16-compile) 1.06 | 0.74 (torch-bf16-cudnn) 1.29 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192 | f32 | 11.47 (torch-compile) 0.27 | 11.33 (torch-compile-ro) 0.28 | win → win |
| rtx-3090 | llm_dec_b32_l8192_bf16 | bf16 | 3.211 (torch-bf16) 0.62 | 1.683 (torch-bf16-cudnn) 1.19 | win → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16kv | bf16kv | 3.211 (torch-bf16) 0.62 | 1.683 (torch-bf16-cudnn) 1.18 | win → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16w | bf16w | 3.211 (torch-bf16) 0.98 | 1.683 (torch-bf16-cudnn) 1.86 | even → loss |
| rtx-3090 | llm_dec_b8_l8192 | f32 | 3.04 (torch-compile) 0.45 | 2.979 (torch-compile-cg) 0.46 | win → win |
| rtx-3090 | llm_dec_b8_l8192_bf16 | bf16 | 0.7267 (torch-bf16) 1.10 | 0.7144 (liger-bf16) 1.12 | loss → loss |
| rtx-3090 | llm_dec_b8_l8192_bf16kv | bf16kv | 0.7267 (torch-bf16) 1.10 | 0.7144 (liger-bf16) 1.12 | loss → loss |
| rtx-3090 | llm_dec_b8_l8192_bf16w | bf16w | 0.7267 (torch-bf16) 1.48 | 0.7144 (liger-bf16) 1.50 | loss → loss |
| rtx-3090 | llm_pre_t2048 | f32 | 15.94 (torch-compile) 0.70 | 13.59 (liger) 0.82 | win → win |
| rtx-3090 | llm_pre_t2048_bf16 | bf16 | 4.146 (torch-bf16-compile) 1.13 | 3.742 (torch-bf16-compile-ro) 1.26 | loss → loss |
| rtx-3090 | llm_pre_t2048_bf16w | bf16w | 4.146 (torch-bf16-compile) 2.62 | 3.742 (torch-bf16-compile-ro) 2.90 | loss → loss |
| rtx-3090 | llm_pre_t2048_tf32 | tf32 | 11.39 (torch-compile-tf32) 0.83 | 8.418 (triton-tf32) 1.12 | win → loss |
| rtx-3090 | llm_pre_t8192 | f32 | 151 (torch-compile) 0.69 | 98.78 (liger) 1.06 | win → loss |
| rtx-3090 | conv_2048_5 | f32 | 0.1618 (torch-eager) 0.68 | 0.1608 (torch-compile-cg) 0.69 | win → win |
| rtx-3090 | ew_4096 | f32 | 0.2406 (chain-fused) 0.53 | 0.2396 (torch-compile-ro) 0.53 | win → win |
| rtx-3090 | fir_4194304_16 | f32 | 0.1198 (torch-eager) 0.62 | 0.1188 (torch-compile-cg) 0.62 | win → win |
| rtx-3090 | firew_rowsum_2048x2048_16 | f32 | 0.1413 (torch-compile) 0.72 | 0.1393 (torch-compile-cg) 0.73 | win → win |
| rtx-3090 | fanout_2048 | f32 | 1.946 (chain-streams) 0.83 | 1.844 (torch-compile-ro) 0.88 | win → win |
| rtx-3090 | loopmm_1024_8 | f32 | 0.9277 (torch-compile) 1.19 | 0.9226 (torch-compile-ro) 1.20 | loss → loss |
| rtx-3090 | twin_2048 | f32 | 1.905 (torch-eager) 0.83 | 1.734 (torch-compile-ro) 0.92 | win → win |
| a100 | epi_4096 | f32 | 7.892 (chain-lt) 1.27 | 7.821 (triton) 1.29 | loss → loss |
| a100 | pj_m128_n28672_k4096 | f32 | 1.953 (torch-eager) 1.26 | 1.845 (torch-compile-ro) 1.34 | loss → loss |
| a100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.2683 (torch-bf16-compile) 1.09 | 0.2355 (triton-bf16) 1.24 | even → loss |
| a100 | pj_m128_n4096_k14336_bf16 | bf16 | 0.126 (torch-bf16) 1.65 | 0.1229 (torch-bf16-compile-ro) 1.69 | loss → loss |
| a100 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0584 (torch-bf16) 1.51 | 0.0563 (torch-bf16-compile-ro) 1.57 | loss → loss |
| a100 | pj_m128_n6144_k4096_bf16 | bf16 | 0.0696 (torch-bf16-compile) 1.52 | 0.0676 (torch-bf16-compile-cg) 1.56 | loss → loss |
| a100 | pj_m16_n18432_k7168 | f32 | 0.4434 (torch-compile) 0.98 | 0.4413 (torch-compile-cg) 0.98 | even → even |
| a100 | pj_m16_n28672_k4096 | f32 | 0.4055 (torch-compile) 1.02 | 0.4024 (torch-compile-cg) 1.03 | even → even |
| a100 | pj_m16_n28672_k4096_bf16 | bf16 | 0.1884 (torch-bf16-compile) 1.12 | 0.1864 (torch-bf16-compile-cg) 1.13 | loss → loss |
| a100 | pj_m16_n4096_k14336_bf16 | bf16 | 0.1106 (torch-bf16-compile) 1.11 | 0.1096 (torch-bf16-compile-cg) 1.12 | loss → loss |
| a100 | pj_m16_n4096_k4096_bf16 | bf16 | 0.044 (torch-bf16-compile) 0.94 | 0.043 (torch-bf16-compile-cg) 0.96 | win → even |
| a100 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0604 (torch-bf16-compile) 1.05 | 0.0594 (torch-bf16-compile-cg) 1.07 | loss → loss |
| a100 | pj_m1_n28672_k4096 | f32 | 0.3482 (torch-compile) 1.09 | 0.3471 (torch-compile-cg) 1.09 | loss → loss |
| a100 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0553 (torch-bf16-compile) 1.26 | 0.0532 (torch-bf16-compile-cg) 1.31 | loss → loss |
| a100 | pj_m2048_n28672_k4096_bf16 | bf16 | 2.201 (torch-bf16) 1.40 | 2.159 (torch-bf16-compile-ro) 1.43 | loss → loss |
| a100 | pj_m2048_n4096_k4096 | f32 | 4.054 (cublas) 1.29 | 3.959 (torch-compile-cg) 1.32 | loss → loss |
| a100 | pj_m2048_n6144_k4096 | f32 | 6.678 (torch-compile) 1.16 | 6.676 (torch-compile-cg) 1.16 | loss → loss |
| a100 | pj_m2048_n6144_k4096_bf16 | bf16 | 0.4291 (torch-bf16-compile) 1.46 | 0.425 (torch-bf16-compile-cg) 1.48 | loss → loss |
| a100 | pj_m2048_n6144_k4096_tf32 | tf32 | 0.8653 (torch-compile-tf32) 1.47 | 0.8581 (torch-compile-ro-tf32) 1.48 | loss → loss |
| a100 | mlp2_2048_tf32 | tf32 | 0.7424 (torch-compile-tf32) 1.51 | 0.7291 (torch-compile-cg-tf32) 1.54 | loss → loss |
| a100 | mlp_2048 | f32 | 2.026 (torch-compile) 1.34 | 2.022 (torch-compile-cg) 1.34 | loss → loss |
| a100 | mlp_2048_tf32 | tf32 | 0.3707 (torch-compile-tf32) 1.52 | 0.3676 (torch-compile-cg-tf32) 1.53 | loss → loss |
| a100 | attn_1024_64 | f32 | 0.0604 (chain-fused) 1.02 | 0.0553 (torch-sdpa-compile-cg) 1.12 | even → loss |
| a100 | attn_2048_128 | f32 | 0.214 (torch-compile) 3.42 | 0.1802 (torch-sdpa-compile-cg) 4.06 | loss → loss |
| a100 | attn_2048_64 | f32 | 0.1444 (torch-compile) 1.01 | 0.1178 (torch-sdpa-compile-cg) 1.23 | even → loss |
| a100 | causal_1024_256 | f32 | 108.4 (torch-compile) 0.34 | 105 (torch-compile-cg) 0.35 | win → win |
| a100 | gen_1024_512_256 | f32 | 53.98 (torch-compile) 0.35 | 51.44 (torch-compile-cg) 0.36 | win → win |
| a100 | lsm_1_131072 | f32 | 0.0369 (torch-compile) 0.81 | 0.0297 (torch-compile-cg) 1.00 | win → even |
| a100 | lsm_4_32768 | f32 | 0.0256 (torch-compile) 0.89 | 0.0195 (torch-compile-cg) 1.16 | win → loss |
| a100 | prefill_1024_512_256 | f32 | 53.92 (torch-compile) 0.36 | 52.72 (torch-compile-cg) 0.37 | win → win |
| a100 | rowsum_4096 | f32 | 7.859 (torch-eager) 1.29 | 7.846 (torch-compile-ro) 1.29 | loss → loss |
| a100 | softmax_rows_4096 | f32 | 8.168 (chain-fused) 1.25 | 7.921 (torch-compile-ro) 1.29 | loss → loss |
| a100 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.5082 (torch-bf16-compile) 1.04 | 0.4794 (torch-bf16-compile-cg) 1.11 | even → loss |
| a100 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.5082 (torch-bf16-compile) 1.51 | 0.4794 (torch-bf16-compile-cg) 1.60 | loss → loss |
| a100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.5082 (torch-bf16-compile) 1.56 | 0.4794 (torch-bf16-compile-cg) 1.65 | loss → loss |
| a100 | llm_blkpre_t2048_tf32 | tf32 | 11.6 (torch-compile-tf32) 1.50 | 10.9 (triton-tf32) 1.60 | loss → loss |
| a100 | llm_dec_b32_l2048 | f32 | 1.98 (torch-compile) 0.40 | 1.962 (torch-compile-cg) 0.40 | win → win |
| a100 | llm_dec_b32_l2048_bf16 | bf16 | 0.5413 (torch-bf16-compile) 0.91 | 0.5243 (torch-bf16-cudnn) 0.94 | win → win |
| a100 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.5413 (torch-bf16-compile) 1.15 | 0.5243 (torch-bf16-cudnn) 1.19 | loss → loss |
| a100 | llm_dec_b32_l2048_bf16w | bf16w | 0.5413 (torch-bf16-compile) 1.37 | 0.5243 (torch-bf16-cudnn) 1.41 | loss → loss |
| a100 | llm_dec_b32_l8192 | f32 | 7.11 (torch-compile) 0.30 | 7.043 (torch-compile-ro) 0.30 | win → win |
| a100 | llm_dec_b32_l8192_bf16 | bf16 | 1.499 (torch-bf16) 0.96 | 1.082 (torch-bf16-cudnn) 1.33 | even → loss |
| a100 | llm_dec_b32_l8192_bf16kv | bf16kv | 1.499 (torch-bf16) 1.04 | 1.082 (torch-bf16-cudnn) 1.45 | even → loss |
| a100 | llm_dec_b32_l8192_bf16w | bf16w | 1.499 (torch-bf16) 1.37 | 1.082 (torch-bf16-cudnn) 1.90 | loss → loss |
| a100 | llm_dec_b8_l2048 | f32 | 0.7727 (torch-compile) 0.46 | 0.695 (torch-compile-cg) 0.52 | win → win |
| a100 | llm_dec_b8_l2048_bf16 | bf16 | 0.2812 (torch-bf16-compile) 0.96 | 0.2689 (torch-bf16-compile-cg) 1.01 | even → even |
| a100 | llm_dec_b8_l2048_bf16kv | bf16kv | 0.2812 (torch-bf16-compile) 0.97 | 0.2689 (torch-bf16-compile-cg) 1.01 | even → even |
| a100 | llm_dec_b8_l2048_bf16w | bf16w | 0.2812 (torch-bf16-compile) 1.11 | 0.2689 (torch-bf16-compile-cg) 1.16 | loss → loss |
| a100 | llm_dec_b8_l8192 | f32 | 2.212 (torch-compile) 0.31 | 2.098 (torch-compile-cg) 0.33 | win → win |
| a100 | llm_pre_t2048 | f32 | 17.56 (torch-compile) 1.10 | 12.94 (liger) 1.49 | loss → loss |
| a100 | llm_pre_t2048_bf16 | bf16 | 1.227 (torch-bf16-compile) 1.71 | 1.082 (torch-bf16-compile-cg) 1.94 | loss → loss |
| a100 | llm_pre_t2048_bf16w | bf16w | 1.227 (torch-bf16-compile) 13.26 | 1.082 (torch-bf16-compile-cg) 15.03 | loss → loss |
| a100 | llm_pre_t2048_tf32 | tf32 | 4.171 (torch-compile-tf32) 2.06 | 3.085 (triton-tf32) 2.79 | loss → loss |
| a100 | llm_pre_t8192 | f32 | 127.1 (torch-compile) 1.64 | 66.2 (liger) 3.15 | loss → loss |
| a100 | conv_2048_5 | f32 | 0.1372 (torch-compile) 0.54 | 0.1362 (torch-compile-cg) 0.54 | win → win |
| a100 | ew_4096 | f32 | 0.1556 (torch-compile) 0.86 | 0.1536 (torch-compile-cg) 0.87 | win → win |
| a100 | ewconvew_2048_5 | f32 | 0.1772 (torch-compile) 0.50 | 0.1731 (torch-compile-ro) 0.52 | win → win |
| a100 | fir_4194304_16 | f32 | 0.087 (torch-compile) 0.59 | 0.085 (torch-compile-cg) 0.60 | win → win |
| a100 | firew_rowsum_2048x2048_16 | f32 | 0.0973 (torch-compile) 0.66 | 0.0942 (torch-compile-ro) 0.68 | win → win |
| a100 | diamond_2048 | f32 | 3.085 (chain-lt) 1.32 | 3.05 (torch-compile-cg) 1.34 | loss → loss |
| a100 | rt_mlp2_1024 | f32 | 5.52 (torch-compile) 1.43 | 4.502 (torch-compile-cg) 1.75 | loss → loss |
| a100 | rt_mlp2_2048 | f32 | 33.36 (torch-compile) 1.32 | 33.23 (torch-compile-cg) 1.33 | loss → loss |
| rtx-4090 | pj_m128_n6144_k4096_bf16 | bf16 | 0.0903 (torch-bf16-compile) 1.19 | 0.0901 (torch-bf16-compile-cg) 1.20 | loss → loss |
| rtx-4090 | pj_m16_n18432_k7168 | f32 | 0.6994 (torch-compile) 1.05 | 0.6973 (torch-compile-cg) 1.05 | even → even |
| rtx-4090 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0594 (torch-bf16-compile) 0.80 | 0.0585 (torch-bf16-compile-cg) 0.82 | win → win |
| rtx-4090 | pj_m1_n18432_k7168 | f32 | 0.6502 (torch-compile) 1.10 | 0.6495 (torch-compile-cg) 1.10 | loss → loss |
| rtx-4090 | pj_m1_n28672_k4096_bf16 | bf16 | 0.3062 (torch-bf16-compile) 1.10 | 0.3031 (torch-bf16-compile-cg) 1.11 | loss → loss |
| rtx-4090 | pj_m1_n4096_k14336_bf16 | bf16 | 0.171 (torch-bf16-compile) 0.72 | 0.169 (torch-bf16-compile-cg) 0.73 | win → win |
| rtx-4090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0645 (torch-bf16-compile) 0.33 | 0.0635 (torch-bf16-compile-cg) 0.33 | win → win |
| rtx-4090 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0904 (torch-bf16-compile) 0.31 | 0.0891 (torch-bf16-compile-cg) 0.31 | win → win |
| rtx-4090 | pj_m2048_n28672_k4096_bf16 | bf16 | 2.763 (torch-bf16-compile) 1.79 | 2.757 (torch-bf16-compile-cg) 1.79 | loss → loss |
| rtx-4090 | pj_m2048_n28672_k4096_tf32 | tf32 | 5.502 (torch-eager-tf32) 1.73 | 5.5 (torch-compile-ro-tf32) 1.73 | loss → loss |
| rtx-4090 | pj_m2048_n4096_k4096_bf16 | bf16 | 0.4014 (torch-bf16-compile) 1.14 | 0.4004 (torch-bf16-compile-cg) 1.14 | loss → loss |
| rtx-4090 | pj_m2048_n6144_k4096_bf16 | bf16 | 0.598 (torch-bf16-compile) 1.13 | 0.597 (torch-bf16-compile-cg) 1.13 | loss → loss |
| rtx-4090 | mlp2_2048_tf32 | tf32 | 0.8489 (torch-compile-tf32) 1.19 | 0.8428 (torch-compile-cg-tf32) 1.20 | loss → loss |
| rtx-4090 | mlp_2048_tf32 | tf32 | 0.4352 (torch-eager-tf32) 1.13 | 0.4239 (torch-compile-ro-tf32) 1.16 | loss → loss |
| rtx-4090 | mlp_512 | f32 | 0.0297 (torch-compile) 1.54 | 0.0287 (torch-compile-cg) 1.60 | loss → loss |
| rtx-4090 | smlp_4096_256_1024 | f32 | 0.1065 (torch-compile) 1.03 | 0.1044 (triton) 1.05 | even → loss |
| rtx-4090 | attn_1024_64 | f32 | 0.0328 (torch-sdpa-compile) 0.83 | 0.0287 (torch-sdpa-compile-cg) 0.95 | win → even |
| rtx-4090 | attn_2048_128 | f32 | 0.0852 (torch-compile) 1.09 | 0.0819 (torch-compile-ro) 1.13 | loss → loss |
| rtx-4090 | attn_2048_64 | f32 | 0.0758 (torch-sdpa-compile) 1.11 | 0.0635 (torch-compile-ro) 1.33 | loss → loss |
| rtx-4090 | causal_1024_256 | f32 | 279.7 (torch-compile) 0.06 | 253.3 (torch-compile-cg) 0.07 | win → win |
| rtx-4090 | gen_1024_512_256 | f32 | 139.1 (torch-compile) 0.07 | 127.4 (torch-compile-ro) 0.07 | win → win |
| rtx-4090 | lsm_1_131072 | f32 | 0.0285 (torch-sdpa-compile) 0.75 | 0.0246 (torch-compile-ro) 0.87 | win → win |
| rtx-4090 | lsm_32_8192 | f32 | 0.0092 (torch-compile) 2.56 | 0.0082 (torch-compile-cg) 2.87 | loss → loss |
| rtx-4090 | lsm_4_32768 | f32 | 0.0174 (torch-compile) 1.18 | 0.0133 (torch-sdpa-compile-cg) 1.54 | loss → loss |
| rtx-4090 | prefill_1024_512_256 | f32 | 139.5 (torch-compile) 0.07 | 127.8 (torch-compile-cg) 0.07 | win → win |
| rtx-4090 | softmax_rows_4096 | f32 | 2.497 (torch-compile) 1.04 | 2.46 (torch-compile-ro) 1.06 | even → loss |
| rtx-4090 | llm_blkdec_b1_l4096 | f32 | 1.305 (torch-compile) 0.90 | 1.286 (torch-compile-cg) 0.91 | win → win |
| rtx-4090 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.7761 (torch-bf16-compile) 0.84 | 0.7508 (torch-bf16-compile-cg) 0.87 | win → win |
| rtx-4090 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.7731 (torch-bf16-compile) 0.89 | 0.751 (torch-bf16-compile-cg) 0.92 | win → win |
| rtx-4090 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.7886 (torch-bf16-compile) 0.90 | 0.7551 (torch-bf16-compile-cg) 0.94 | win → win |
| rtx-4090 | llm_blkpre_t2048 | f32 | 19.25 (torch-compile) 0.93 | 18.04 (liger) 0.99 | win → even |
| rtx-4090 | llm_blkpre_t2048_bf16 | bf16 | 5.532 (torch-bf16-compile) 1.38 | 5.527 (torch-bf16-compile-cg) 1.38 | loss → loss |
| rtx-4090 | llm_blkpre_t2048_tf32 | tf32 | 13.64 (torch-compile-tf32) 1.19 | 12.37 (triton-tf32) 1.32 | loss → loss |
| rtx-4090 | llm_dec_b1_l2048_bf16 | bf16 | 0.3642 (torch-bf16-compile) 0.46 | 0.3082 (torch-bf16-compile-cg) 0.55 | win → win |
| rtx-4090 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.3631 (torch-bf16-compile) 0.46 | 0.3065 (torch-bf16-compile-cg) 0.55 | win → win |
| rtx-4090 | llm_dec_b1_l2048_bf16w | bf16w | 0.3602 (torch-bf16-compile) 0.50 | 0.3121 (torch-bf16-compile-cg) 0.57 | win → win |
| rtx-4090 | llm_dec_b1_l8192 | f32 | 0.4818 (torch-compile) 0.71 | 0.48 (torch-compile-cg) 0.72 | win → win |
| rtx-4090 | llm_dec_b1_l8192_bf16 | bf16 | 0.3581 (torch-bf16-compile) 0.59 | 0.3507 (torch-bf16-compile-cg) 0.60 | win → win |
| rtx-4090 | llm_dec_b1_l8192_bf16kv | bf16kv | 0.3805 (torch-bf16-compile) 0.55 | 0.3505 (torch-bf16-compile-cg) 0.60 | win → win |
| rtx-4090 | llm_dec_b1_l8192_bf16w | bf16w | 0.3654 (torch-bf16-compile) 0.70 | 0.3467 (torch-bf16-compile-cg) 0.74 | win → win |
| rtx-4090 | llm_pre_t2048 | f32 | 6.332 (torch-compile) 0.75 | 5.164 (liger) 0.92 | win → win |
| rtx-4090 | llm_pre_t2048_bf16 | bf16 | 1.328 (torch-bf16-compile) 1.40 | 1.319 (torch-bf16-compile-cg) 1.41 | loss → loss |
| rtx-4090 | llm_pre_t2048_bf16w | bf16w | 1.322 (torch-bf16-compile) 3.48 | 1.319 (torch-bf16-compile-cg) 3.49 | loss → loss |
| rtx-4090 | llm_pre_t2048_tf32 | tf32 | 5.124 (torch-compile-tf32) 0.79 | 3.276 (triton-tf32) 1.24 | win → loss |
| rtx-4090 | llm_pre_t8192 | f32 | 59.89 (torch-compile) 0.58 | 37.53 (liger) 0.93 | win → win |
| rtx-4090 | conv_2048_5 | f32 | 0.0942 (torch-eager) 0.57 | 0.0932 (torch-compile-cg) 0.57 | win → win |
| rtx-4090 | ew_4096 | f32 | 0.2532 (chain-fused) 0.32 | 0.253 (torch-compile-cg) 0.32 | win → win |
| rtx-4090 | ewconvew_2048_5 | f32 | 0.131 (torch-compile) 0.49 | 0.13 (torch-compile-cg) 0.50 | win → win |
| rtx-4090 | fir_4194304_16 | f32 | 0.0666 (torch-eager) 0.57 | 0.0655 (torch-compile-cg) 0.57 | win → win |
| rtx-4090 | firew_rowsum_2048x2048_16 | f32 | 0.0727 (torch-compile) 0.61 | 0.0696 (torch-compile-cg) 0.64 | win → win |
| rtx-4090 | indep_512 | f32 | 0.0297 (torch-compile) 1.90 | 0.0288 (torch-compile-cg) 1.96 | loss → loss |
| rtx-4090 | loopmm_1024_8 | f32 | 0.3619 (torch-eager) 1.15 | 0.3533 (torch-compile-ro) 1.17 | loss → loss |
| rtx-4090 | rt_mlp2_1024 | f32 | 1.612 (torch-compile) 1.04 | 1.519 (torch-compile-ro) 1.10 | even → loss |
| rtx-4090 | rt_mlp2_2048 | f32 | 10.47 (torch-compile) 0.98 | 10.44 (torch-compile-cg) 0.99 | even → even |
| h100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.0939 (torch-bf16-compile) 1.53 | 0.0924 (triton-bf16) 1.55 | loss → loss |
| h100 | pj_m128_n4096_k14336 | f32 | 0.3105 (lt) 1.15 | 0.3103 (torch-compile-cg) 1.15 | loss → loss |
| h100 | pj_m128_n4096_k14336_bf16 | bf16 | 0.0579 (torch-bf16-compile) 2.85 | 0.0578 (torch-bf16-compile-cg) 2.86 | loss → loss |
| h100 | pj_m128_n4096_k4096 | f32 | 0.0979 (lt) 1.38 | 0.0977 (torch-compile-cg) 1.38 | loss → loss |
| h100 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0241 (torch-bf16) 2.52 | 0.0238 (torch-bf16-compile-ro) 2.55 | loss → loss |
| h100 | pj_m16_n28672_k4096 | f32 | 0.1761 (torch-compile) 1.10 | 0.1725 (torch-compile-cg) 1.13 | loss → loss |
| h100 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0239 (torch-bf16) 2.32 | 0.0238 (torch-bf16-compile-ro) 2.33 | loss → loss |
| h100 | pj_m1_n4096_k14336_bf16 | bf16 | 0.0536 (torch-bf16-compile) 1.31 | 0.0525 (torch-bf16-compile-cg) 1.34 | loss → loss |
| h100 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0223 (torch-bf16) 0.99 | 0.0218 (torch-bf16-compile-ro) 1.01 | even → even |
| h100 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0311 (torch-bf16) 1.13 | 0.0287 (torch-bf16-compile-cg) 1.22 | loss → loss |
| h100 | pj_m2048_n28672_k4096_bf16 | bf16 | 0.6086 (torch-bf16-compile) 2.73 | 0.6085 (torch-bf16-compile-ro) 2.73 | loss → loss |
| h100 | pj_m2048_n4096_k14336_bf16 | bf16 | 0.2996 (torch-bf16) 2.58 | 0.2978 (torch-bf16-compile-cg) 2.60 | loss → loss |
| h100 | pj_m2048_n6144_k4096 | f32 | 1.976 (torch-compile) 1.10 | 1.976 (torch-compile-ro) 1.10 | loss → loss |
| h100 | mlp2_2048_tf32 | tf32 | 0.215 (torch-compile-tf32) 2.59 | 0.2106 (torch-compile-cg-tf32) 2.64 | loss → loss |
| h100 | mlp_2048_tf32 | tf32 | 0.1114 (torch-eager-tf32) 3.14 | 0.1087 (torch-compile-ro-tf32) 3.22 | loss → loss |
| h100 | mlp_512 | f32 | 0.0336 (torch-compile) 1.42 | 0.0323 (torch-compile-cg) 1.47 | loss → loss |
| h100 | attn_1024_64 | f32 | 0.0322 (torch-sdpa-compile) 1.39 | 0.0267 (torch-compile-ro) 1.68 | loss → loss |
| h100 | attn_2048_128 | f32 | 0.0917 (torch-sdpa-compile) 1.11 | 0.0792 (torch-compile-ro) 1.29 | loss → loss |
| h100 | attn_2048_64 | f32 | 0.0638 (torch-sdpa-compile) 1.48 | 0.0599 (torch-compile-ro) 1.58 | loss → loss |
| h100 | causal_1024_256 | f32 | 112.2 (torch-compile) 0.19 | 107.9 (torch-compile-cg) 0.20 | win → win |
| h100 | gen_1024_512_256 | f32 | 56.34 (torch-compile) 0.19 | 54.06 (torch-compile-cg) 0.20 | win → win |
| h100 | lsm_1_131072 | f32 | 0.0237 (torch-compile) 1.22 | 0.02 (torch-compile-cg) 1.45 | loss → loss |
| h100 | lsm_32_8192 | f32 | 0.0117 (torch-compile) 2.44 | 0.0112 (torch-compile-ro) 2.55 | loss → loss |
| h100 | lsm_4_32768 | f32 | 0.0217 (torch-sdpa-compile) 1.27 | 0.0164 (torch-compile-cg) 1.68 | loss → loss |
| h100 | prefill_1024_512_256 | f32 | 56.4 (torch-compile) 0.20 | 54.25 (torch-compile-cg) 0.20 | win → win |
| h100 | rowsum_4096 | f32 | 2.69 (chain-eager) 1.08 | 2.684 (torch-compile-ro) 1.09 | loss → loss |
| h100 | softmax_rows_2048 | f32 | 0.3566 (torch-compile) 1.18 | 0.3548 (torch-compile-ro) 1.18 | loss → loss |
| h100 | softmax_rows_4096 | f32 | 2.725 (torch-compile) 1.09 | 2.714 (torch-compile-cg) 1.09 | loss → loss |
| h100 | llm_blkdec_b1_l4096 | f32 | 0.4835 (torch-compile) 1.15 | 0.4681 (torch-compile-cg) 1.18 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.3395 (torch-bf16-compile) 0.78 | 0.286 (torch-bf16-compile-cg) 0.93 | win → win |
| h100 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.3395 (torch-bf16-compile) 1.44 | 0.286 (torch-bf16-compile-cg) 1.72 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.3395 (torch-bf16-compile) 1.45 | 0.286 (torch-bf16-compile-cg) 1.72 | loss → loss |
| h100 | llm_blkpre_t2048 | f32 | 19.58 (torch-compile) 1.08 | 18.66 (liger) 1.14 | loss → loss |
| h100 | llm_blkpre_t2048_bf16 | bf16 | 1.355 (torch-bf16-compile) 2.82 | 1.308 (torch-bf16-compile-cg) 2.93 | loss → loss |
| h100 | llm_blkpre_t2048_bf16w | bf16w | 1.355 (torch-bf16-compile) 16.29 | 1.308 (torch-bf16-compile-cg) 16.88 | loss → loss |
| h100 | llm_blkpre_t2048_tf32 | tf32 | 3.604 (torch-compile-tf32) 1.81 | 3.44 (triton-tf32) 1.89 | loss → loss |
| h100 | llm_dec_b1_l2048 | f32 | 0.2143 (torch-compile) 0.51 | 0.193 (torch-compile-cg) 0.57 | win → win |
| h100 | llm_dec_b1_l2048_bf16 | bf16 | 0.1997 (torch-bf16-compile) 0.42 | 0.1455 (torch-bf16-compile-cg) 0.57 | win → win |
| h100 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.1997 (torch-bf16-compile) 0.44 | 0.1455 (torch-bf16-compile-cg) 0.60 | win → win |
| h100 | llm_dec_b1_l2048_bf16w | bf16w | 0.1997 (torch-bf16-compile) 0.44 | 0.1455 (torch-bf16-compile-cg) 0.61 | win → win |
| h100 | llm_dec_b1_l8192 | f32 | 0.3121 (torch-compile) 0.45 | 0.2995 (torch-compile-cg) 0.46 | win → win |
| h100 | llm_dec_b1_l8192_bf16 | bf16 | 0.1998 (torch-bf16-compile) 0.55 | 0.1472 (torch-bf16-compile-cg) 0.74 | win → win |
| h100 | llm_dec_b1_l8192_bf16kv | bf16kv | 0.1998 (torch-bf16-compile) 0.57 | 0.1472 (torch-bf16-compile-cg) 0.77 | win → win |
| h100 | llm_dec_b1_l8192_bf16w | bf16w | 0.1998 (torch-bf16-compile) 0.58 | 0.1472 (torch-bf16-compile-cg) 0.78 | win → win |
| h100 | llm_dec_b32_l2048_bf16 | bf16 | 0.2607 (torch-bf16-compile) 1.02 | 0.2266 (torch-bf16-compile-cg) 1.17 | even → loss |
| h100 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.2607 (torch-bf16-compile) 1.22 | 0.2266 (torch-bf16-compile-cg) 1.40 | loss → loss |
| h100 | llm_dec_b32_l2048_bf16w | bf16w | 0.2607 (torch-bf16-compile) 1.40 | 0.2266 (torch-bf16-compile-cg) 1.62 | loss → loss |
| h100 | llm_dec_b32_l8192 | f32 | 3.938 (torch-compile) 0.26 | 3.931 (torch-compile-cg) 0.26 | win → win |
| h100 | llm_dec_b32_l8192_bf16 | bf16 | 0.512 (torch-bf16-compile) 1.44 | 0.4866 (torch-bf16-compile-cg) 1.51 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16kv | bf16kv | 0.512 (torch-bf16-compile) 1.55 | 0.4866 (torch-bf16-compile-cg) 1.63 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16w | bf16w | 0.512 (torch-bf16-compile) 2.01 | 0.4866 (torch-bf16-compile-cg) 2.11 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16 | bf16 | 0.2247 (torch-bf16-compile) 0.63 | 0.1615 (torch-bf16-compile-cg) 0.88 | win → win |
| h100 | llm_dec_b8_l2048_bf16kv | bf16kv | 0.2247 (torch-bf16-compile) 0.64 | 0.1615 (torch-bf16-compile-cg) 0.89 | win → win |
| h100 | llm_dec_b8_l2048_bf16w | bf16w | 0.2247 (torch-bf16-compile) 0.69 | 0.1615 (torch-bf16-compile-cg) 0.96 | win → even |
| h100 | llm_dec_b8_l8192_bf16 | bf16 | 0.2601 (torch-bf16-compile) 1.07 | 0.2278 (torch-bf16-compile-cg) 1.23 | loss → loss |
| h100 | llm_dec_b8_l8192_bf16kv | bf16kv | 0.2601 (torch-bf16-compile) 1.08 | 0.2278 (torch-bf16-compile-cg) 1.24 | loss → loss |
| h100 | llm_dec_b8_l8192_bf16w | bf16w | 0.2601 (torch-bf16-compile) 1.29 | 0.2278 (torch-bf16-compile-cg) 1.48 | loss → loss |
| h100 | llm_pre_t2048 | f32 | 5.502 (torch-compile) 1.09 | 4.552 (liger) 1.31 | loss → loss |
| h100 | llm_pre_t2048_bf16 | bf16 | 0.3576 (torch-bf16-compile) 3.45 | 0.3497 (torch-bf16-compile-ro) 3.53 | loss → loss |
| h100 | llm_pre_t2048_bf16w | bf16w | 0.3576 (torch-bf16-compile) 17.14 | 0.3497 (torch-bf16-compile-ro) 17.53 | loss → loss |
| h100 | llm_pre_t2048_tf32 | tf32 | 1.53 (torch-compile-tf32) 2.05 | 1.373 (triton-tf32) 2.28 | loss → loss |
| h100 | llm_pre_t8192 | f32 | 43.87 (torch-compile) 1.97 | 25.82 (liger) 3.35 | loss → loss |
| h100 | ewconvew_2048_5 | f32 | 0.1147 (torch-compile) 0.49 | 0.1118 (torch-compile-cg) 0.50 | win → win |
| h100 | firew_rowsum_2048x2048_16 | f32 | 0.0673 (torch-compile) 0.58 | 0.0655 (torch-compile-ro) 0.60 | win → win |
| h100 | loopmm_1024_8 | f32 | 0.4558 (torch-compile) 1.13 | 0.4503 (torch-compile-cg) 1.14 | loss → loss |
| h100 | rt_mlp2_1024 | f32 | 2.047 (torch-compile) 1.03 | 1.96 (torch-compile-ro) 1.08 | even → loss |
| h100 | rt_mlp2_2048 | f32 | 11.29 (torch-compile) 1.09 | 11.18 (torch-compile-cg) 1.10 | loss → loss |

Mapal wins that disappear: 17 — rtx-5080 llm_pre_t2048_tf32 (triton-tf32), rtx-5080 lsm_1_131072 (torch-compile-cg), rtx-5080 pj_m16_n4096_k4096_bf16 (torch-bf16-compile-cg), rtx-3090 llm_blkpre_t2048_tf32 (triton-tf32), rtx-3090 llm_dec_b32_l8192_bf16 (torch-bf16-cudnn), rtx-3090 llm_dec_b32_l8192_bf16kv (torch-bf16-cudnn), rtx-3090 llm_pre_t2048_tf32 (triton-tf32), rtx-3090 llm_pre_t8192 (liger), rtx-3090 lsm_1_131072 (torch-compile-ro), rtx-3090 pj_m2048_n6144_k4096_bf16 (torch-bf16-compile-ro), a100 lsm_1_131072 (torch-compile-cg), a100 lsm_4_32768 (torch-compile-cg), a100 pj_m16_n4096_k4096_bf16 (torch-bf16-compile-cg), rtx-4090 attn_1024_64 (torch-sdpa-compile-cg), rtx-4090 llm_blkpre_t2048 (liger), rtx-4090 llm_pre_t2048_tf32 (triton-tf32), h100 llm_dec_b8_l2048_bf16w (torch-bf16-compile-cg).

## Every competitor leg

Per row and part, every leg at the row's precision, grouped by competitor family: `leg ms (Mapal/leg)`; **bold** = the row's fastest leg; `✗gate` = timed but outside the value gate (not counted); `cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg for this row on another part; `unavailable` = the part's run reported the package missing. Families: vendor libs; torch eager; torch.compile; compile + CUDA graphs; reduce-overhead; SDPA; FlashAttention-2/3; FlashInfer; hand Triton; Liger; TensorRT-LLM; hand CUDA. Columns with no leg on any row of a table are left out.

### GEMM cells & projections

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| epi_4096 | rtx-5080 | 3.794 | chain-lt 3.727 (1.02)<br>chain-eager 3.861 (0.98)<br>chain-cutlass 3.901 (0.97) | torch-eager 3.793 (1.00) | torch-compile 3.741 (1.01) | torch-compile-cg 3.742 (1.01) | **torch-compile-ro 3.724 (1.02)** | triton 4.297 (0.88) |
| epi_4096 | rtx-3090 | 5.939 | chain-lt 7.103 (0.84)<br>chain-cutlass 7.237 (0.82)<br>chain-eager 7.251 (0.82) | torch-eager 7.246 (0.82) | torch-compile 7.085 (0.84) | torch-compile-cg 7.09 (0.84) | **torch-compile-ro 7.084 (0.84)** | triton 8.29 (0.72) |
| epi_4096 | a100 | 10.06 | chain-lt 7.892 (1.27)<br>chain-eager 7.984 (1.26)<br>chain-cutlass 8.258 (1.22) | torch-eager 7.985 (1.26) | torch-compile 9.202 (1.09) | torch-compile-cg 9.199 (1.09) | torch-compile-ro 7.885 (1.28) | **triton 7.821 (1.29)** |
| epi_4096 | rtx-4090 | 2.496 | **chain-lt 2.433 (1.03)**<br>chain-cutlass 2.479 (1.01)<br>chain-eager 2.561 (0.97) | torch-eager 2.692 (0.93) | torch-compile 2.477 (1.01) | torch-compile-cg 2.488 (1.00) | torch-compile-ro 2.473 (1.01) | triton 2.947 (0.85) |
| epi_4096 | h100 | 2.882 | **chain-lt 2.642 (1.09)**<br>chain-eager 2.7 (1.07)<br>chain-cutlass 3.33 (0.87) | torch-eager 2.753 (1.05) | torch-compile 2.715 (1.06) | torch-compile-cg 2.708 (1.06) | torch-compile-ro 2.711 (1.06) | triton 3.003 (0.96) |
| mm_16x2048x7168 | rtx-5080 | 0.08785 | **cublas 0.0799 (1.10)**<br>cutlass-tuned 0.3891 (0.23)<br>cutlass 0.5302 (0.17) |  |  |  |  | triton 0.1536 (0.57) |
| mm_16x2048x7168 | rtx-3090 | 0.09165 | **cublas 0.0881 (1.04)**<br>cutlass-tuned 0.2248 (0.41)<br>cutlass 0.7439 (0.12) |  |  |  |  | triton 0.2191 (0.42) |
| mm_16x2048x7168 | a100 | 0.09265 | **cublas 0.0901 (1.03)**<br>cutlass-tuned 0.3477 (0.27)<br>cutlass 1.337 (0.07) |  |  |  |  | triton 0.5335 (0.17) |
| mm_16x2048x7168 | rtx-4090 | 0.1208 | **cublas 0.1059 (1.14)**<br>cutlass-tuned 0.2082 (0.58)<br>cutlass 0.5014 (0.24) |  |  |  |  | triton 0.1495 (0.81) |
| mm_16x2048x7168 | h100 | 0.0947 | **cublas 0.0513 (1.85)**<br>cutlass-tuned 0.2799 (0.34)<br>cutlass 0.7574 (0.13) |  |  |  |  | triton 0.1873 (0.51) |
| mm_4096x4096x4096 | rtx-5080 | 3.845 | **cublas 3.671 (1.05)**<br>cutlass 3.743 (1.03)<br>cutlass-tuned 9.173 (0.42) |  |  |  |  | triton 4.306 (0.89) |
| mm_4096x4096x4096 | rtx-3090 | 8.841 | **cublas 8.115 (1.09)**<br>cutlass 8.706 (1.02)<br>cutlass-tuned 21.41 (0.41) |  |  |  |  | triton 8.526 (1.04) |
| mm_4096x4096x4096 | rtx-4090 | 2.417 | **cublas 2.466 (0.98)**<br>cutlass 2.491 (0.97)<br>cutlass-tuned 6.661 (0.36) |  |  |  |  | triton 2.924 (0.83) |
| mm_4096x4096x4096 | h100 | 2.88 | **cublas 2.673 (1.08)**<br>cutlass 3.329 (0.87)<br>cutlass-tuned 6.782 (0.42) |  |  |  |  | triton 2.942 (0.98) |
| pj_m128_n28672_k4096 | rtx-5080 | 0.8057 | **lt-fresh 0.7476 (1.08)**<br>lt 0.7496 (1.07)<br>cublas 0.7906 (1.02)<br>cublas-fresh 0.7926 (1.02) | torch-eager 0.8028 (1.00) | torch-compile 0.8049 (1.00) | torch-compile-cg 0.8008 (1.01) | torch-compile-ro 0.8028 (1.00) | triton 1.061 (0.76) |
| pj_m128_n28672_k4096 | rtx-3090 | 1.198 | cublas 1.986 (0.60)<br>cublas-fresh 1.987 (0.60)<br>lt 1.992 (0.60)<br>lt-fresh 1.992 (0.60) | torch-eager 2.342 (0.51) | torch-compile 2.037 (0.59) | torch-compile-cg 1.541 (0.78) | **torch-compile-ro 1.345 (0.89)** | triton 2.176 (0.55) |
| pj_m128_n28672_k4096 | a100 | 2.466 | cublas 1.955 (1.26)<br>lt 1.955 (1.26)<br>cublas-fresh 1.955 (1.26)<br>lt-fresh 1.956 (1.26) | torch-eager 1.953 (1.26) | torch-compile 2.214 (1.11) | torch-compile-cg 1.865 (1.32) | **torch-compile-ro 1.845 (1.34)** | triton 2.715 (0.91) |
| pj_m128_n28672_k4096 | rtx-4090 | 0.6885 | **lt-fresh 0.7096 (0.97)**<br>lt 0.7158 (0.96)<br>cublas-fresh 0.7516 (0.92)<br>cublas 0.7558 (0.91) | torch-eager 0.7547 (0.91) | torch-compile 0.7455 (0.92) | torch-compile-cg 0.7467 (0.92) | torch-compile-ro 0.7629 (0.90) | triton 0.7793 (0.88) |
| pj_m128_n28672_k4096 | h100 | 0.6929 | **lt 0.6563 (1.06)**<br>lt-fresh 0.6606 (1.05)<br>cublas 0.6766 (1.02)<br>cublas-fresh 0.6772 (1.02) | torch-eager 0.6867 (1.01) | torch-compile 0.6804 (1.02) | torch-compile-cg 0.681 (1.02) | torch-compile-ro 0.6818 (1.02) | triton 0.8065 (0.86) |
| pj_m128_n4096_k14336 | rtx-5080 | 0.4489 | **cublas-fresh 0.4014 (1.12)**<br>lt-fresh 0.4014 (1.12)<br>cublas 0.4066 (1.10)<br>lt 0.4076 (1.10) | torch-eager 0.4117 (1.09) | torch-compile 0.4117 (1.09) | torch-compile-cg 0.4117 (1.09) | torch-compile-ro 0.4117 (1.09) | triton 0.5898 (0.76) |
| pj_m128_n4096_k14336 | rtx-3090 | 0.6983 | lt 0.8653 (0.81)<br>lt-fresh 0.8653 (0.81)<br>cublas 0.8704 (0.80)<br>cublas-fresh 0.8714 (0.80) | torch-eager 0.8161 (0.86) | torch-compile 0.683 (1.02) | torch-compile-cg 0.6861 (1.02) | **torch-compile-ro 0.6769 (1.03)** | triton 1.165 (0.60) |
| pj_m128_n4096_k14336 | a100 | 1.563 | **lt-fresh 0.9226 (1.69)**<br>cublas-fresh 0.9308 (1.68)<br>lt 0.9339 (1.67)<br>cublas 0.9349 (1.67) | torch-eager 1.084 (1.44) | torch-compile 1.084 (1.44) | torch-compile-cg 1.082 (1.44) | torch-compile-ro 1.082 (1.44) | triton 1.761 (0.89) |
| pj_m128_n4096_k14336 | rtx-4090 | 0.4507 | **lt-fresh 0.3952 (1.14)**<br>cublas-fresh 0.3963 (1.14)<br>lt 0.3994 (1.13)<br>cublas 0.4055 (1.11) | torch-eager 0.4076 (1.11) | torch-compile 0.4076 (1.11) | torch-compile-cg 0.4055 (1.11) | torch-compile-ro 0.4055 (1.11) | triton 0.4045 (1.11) |
| pj_m128_n4096_k14336 | h100 | 0.358 | lt 0.3105 (1.15)<br>cublas 0.3115 (1.15)<br>lt-fresh 0.3118 (1.15)<br>cublas-fresh 0.3119 (1.15) | torch-eager 0.3123 (1.15) | torch-compile 0.3119 (1.15) | **torch-compile-cg 0.3103 (1.15)** | torch-compile-ro 0.3117 (1.15) | triton 0.3943 (0.91) |
| pj_m128_n4096_k4096 | rtx-5080 | 0.1392 | **cublas-fresh 0.1352 (1.03)**<br>lt-fresh 0.1352 (1.03)<br>cublas 0.1434 (0.97)<br>lt 0.1434 (0.97) | torch-eager 0.1454 (0.96) | torch-compile 0.1454 (0.96) | torch-compile-cg 0.1454 (0.96) | torch-compile-ro 0.1454 (0.96) | triton 0.1761 (0.79) |
| pj_m128_n4096_k4096 | rtx-3090 | 0.2413 | lt-fresh 0.2437 (0.99)<br>lt 0.2447 (0.99)<br>cublas 0.2693 (0.90)<br>cublas-fresh 0.2693 (0.90) | **torch-eager 0.2079 (1.16)** | torch-compile 0.2284 (1.06) | torch-compile-cg 0.2273 (1.06) | torch-compile-ro 0.2273 (1.06) | triton 0.3052 (0.79) |
| pj_m128_n4096_k4096 | a100 | 0.4719 | **cublas 0.2591 (1.82)**<br>cublas-fresh 0.2591 (1.82)<br>lt-fresh 0.2591 (1.82)<br>lt 0.2601 (1.81) | torch-eager 0.3277 (1.44) | torch-compile 0.3287 (1.44) | torch-compile-cg 0.3287 (1.44) | torch-compile-ro 0.3287 (1.44) | triton 0.4864 (0.97) |
| pj_m128_n4096_k4096 | rtx-4090 | 0.1347 | **lt-fresh 0.0994 (1.36)**<br>cublas-fresh 0.1044 (1.29)<br>lt 0.1372 (0.98)<br>cublas 0.1502 (0.90) | torch-eager 0.1516 (0.89) | torch-compile 0.1249 (1.08) | torch-compile-cg 0.1239 (1.09) | torch-compile-ro 0.1516 (0.89) | triton 0.127 (1.06) |
| pj_m128_n4096_k4096 | h100 | 0.1349 | lt 0.0979 (1.38)<br>cublas 0.0987 (1.37)<br>cublas-fresh 0.0988 (1.37)<br>lt-fresh 0.099 (1.36) | torch-eager 0.0984 (1.37) | torch-compile 0.0985 (1.37) | **torch-compile-cg 0.0977 (1.38)** | torch-compile-ro 0.0978 (1.38) | triton 0.1188 (1.14) |
| pj_m128_n6144_k4096 | rtx-5080 | 0.2055 | **cublas-fresh 0.1802 (1.14)**<br>lt-fresh 0.1802 (1.14)<br>cublas 0.1864 (1.10)<br>lt 0.1864 (1.10) | torch-eager 0.1884 (1.09) | torch-compile 0.1884 (1.09) | torch-compile-cg 0.1884 (1.09) | torch-compile-ro 0.1884 (1.09) | triton 0.2929 (0.70) |
| pj_m128_n6144_k4096 | rtx-3090 | 0.3372 | lt-fresh 0.3666 (0.92)<br>lt 0.3676 (0.92)<br>cublas 0.4055 (0.83)<br>cublas-fresh 0.4055 (0.83) | torch-eager 0.3267 (1.03) | torch-compile 0.3226 (1.05) | **torch-compile-cg 0.3205 (1.05)** | torch-compile-ro 0.3226 (1.05) | triton 0.5028 (0.67) |
| pj_m128_n6144_k4096 | a100 | 0.7521 | **lt 0.3942 (1.91)**<br>lt-fresh 0.3942 (1.91)<br>cublas 0.4782 (1.57)<br>cublas-fresh 0.4782 (1.57) | torch-eager 0.5704 (1.32) | torch-compile 0.5315 (1.41) | torch-compile-cg 0.5304 (1.42) | torch-compile-ro 0.5693 (1.32) | triton 0.5755 (1.31) |
| pj_m128_n6144_k4096 | rtx-4090 | 0.2323 | **cublas-fresh 0.169 (1.37)**<br>lt-fresh 0.169 (1.37)<br>lt 0.1997 (1.16)<br>cublas 0.2039 (1.14) | torch-eager 0.2058 (1.13) | torch-compile 0.1833 (1.27) | torch-compile-cg 0.1833 (1.27) | torch-compile-ro 0.2048 (1.13) | triton 0.1782 (1.30) |
| pj_m128_n6144_k4096 | h100 | 0.1867 | **lt-fresh 0.149 (1.25)**<br>lt 0.1499 (1.25)<br>cublas 0.15 (1.24)<br>cublas-fresh 0.1501 (1.24) | torch-eager 0.1499 (1.25) | torch-compile 0.1502 (1.24) | torch-compile-cg 0.1492 (1.25) | torch-compile-ro 0.1499 (1.25) | triton 0.2023 (0.92) |
| pj_m16_n18432_k7168 | rtx-5080 | 0.6004 | **cublas-fresh 0.5898 (1.02)**<br>lt-fresh 0.5898 (1.02)<br>cublas 0.6185 (0.97)<br>lt 0.6185 (0.97) | torch-eager 0.6195 (0.97) | torch-compile 0.6103 (0.98) | torch-compile-cg 0.6083 (0.99) | torch-compile-ro 0.6205 (0.97) | triton 0.6226 (0.96) |
| pj_m16_n18432_k7168 | rtx-3090 | 0.6317 | cublas 0.767 (0.82)<br>lt 0.767 (0.82)<br>cublas-fresh 0.767 (0.82)<br>lt-fresh 0.767 (0.82) | torch-eager 0.6748 (0.94) | **torch-compile 0.6134 (1.03)** | torch-compile-cg 0.6595 (0.96) | torch-compile-ro 0.6738 (0.94) | triton 0.897 (0.70) |
| pj_m16_n18432_k7168 | a100 | 0.434 | lt 0.5468 (0.79)<br>lt-fresh 0.5468 (0.79)<br>cublas 0.5734 (0.76)<br>cublas-fresh 0.5734 (0.76) | torch-eager 0.6472 (0.67) | torch-compile 0.4434 (0.98) | **torch-compile-cg 0.4413 (0.98)** | torch-compile-ro 0.6441 (0.67) | triton 0.9789 (0.44) |
| pj_m16_n18432_k7168 | rtx-4090 | 0.7317 | lt-fresh 0.7059 (1.04)<br>lt 0.7086 (1.03)<br>cublas-fresh 0.7117 (1.03)<br>cublas 0.7158 (1.02) | torch-eager 0.7772 (0.94) | torch-compile 0.6994 (1.05) | **torch-compile-cg 0.6973 (1.05)** | torch-compile-ro 0.7178 (1.02) | triton 0.7864 (0.93) |
| pj_m16_n18432_k7168 | h100 | 0.2071 | lt 0.2371 (0.87)<br>lt-fresh 0.2387 (0.87)<br>cublas 0.2603 (0.80)<br>cublas-fresh 0.2619 (0.79) | torch-eager 0.2605 (0.80) | **torch-compile 0.2057 (1.01)** | torch-compile-cg 0.2059 (1.01) | torch-compile-ro 0.2605 (0.80) | triton 0.3482 (0.59) |
| pj_m16_n2048_k7168 | rtx-5080 | 0.04163 | **cublas-fresh 0.0492 (0.85)**<br>lt-fresh 0.0492 (0.85)<br>cublas 0.0799 (0.52)<br>lt 0.0799 (0.52) | torch-eager 0.0799 (0.52) | torch-compile 0.0799 (0.52) | torch-compile-cg 0.0799 (0.52) | torch-compile-ro 0.0799 (0.52) | triton 0.1536 (0.27) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1032 | lt 0.085 (1.21)<br>lt-fresh 0.086 (1.20)<br>cublas 0.0891 (1.16)<br>cublas-fresh 0.0891 (1.16) | torch-eager 0.0942 (1.10) | **torch-compile 0.084 (1.23)** | torch-compile-cg 0.084 (1.23) | torch-compile-ro 0.0891 (1.16) | triton 0.2222 (0.46) |
| pj_m16_n2048_k7168 | a100 | 0.1018 | **cublas-fresh 0.086 (1.18)**<br>lt-fresh 0.086 (1.18)<br>cublas 0.0901 (1.13)<br>lt 0.0911 (1.12) | torch-eager 0.1085 (0.94) | torch-compile 0.1075 (0.95) | torch-compile-cg 0.1055 (0.97) | torch-compile-ro 0.1044 (0.98) | triton 0.4342 (0.23) |
| pj_m16_n2048_k7168 | rtx-4090 | 0.05359 | **cublas-fresh 0.0297 (1.80)**<br>lt-fresh 0.0297 (1.80)<br>lt 0.1055 (0.51)<br>cublas 0.1065 (0.50) | torch-eager 0.1116 (0.48) | torch-compile 0.1033 (0.52) | torch-compile-cg 0.1024 (0.52) | torch-compile-ro 0.1055 (0.51) | triton 0.1488 (0.36) |
| pj_m16_n2048_k7168 | h100 | 0.06012 | **lt-fresh 0.048 (1.25)**<br>lt 0.0486 (1.24)<br>cublas-fresh 0.0504 (1.19)<br>cublas 0.0514 (1.17) | torch-eager 0.0514 (1.17) | torch-compile 0.0821 (0.73) | torch-compile-cg 0.0824 (0.73) | torch-compile-ro 0.0503 (1.20) | triton 0.1927 (0.31) |
| pj_m16_n28672_k4096 | rtx-5080 | 0.5428 | **lt-fresh 0.5243 (1.04)**<br>lt 0.553 (0.98)<br>cublas 1.984 (0.27)<br>cublas-fresh 2.064 (0.26) | torch-eager 1.94 (0.28) | torch-compile 0.5448 (1.00) | torch-compile-cg 0.5448 (1.00) | torch-compile-ro 1.937 (0.28) | triton 0.5489 (0.99) |
| pj_m16_n28672_k4096 | rtx-3090 | 0.5876 | cublas 0.6666 (0.88)<br>lt 0.6666 (0.88)<br>cublas-fresh 0.6666 (0.88)<br>lt-fresh 0.6666 (0.88) | torch-eager 0.598 (0.98) | **torch-compile 0.5448 (1.08)** | torch-compile-cg 0.5888 (1.00) | torch-compile-ro 0.597 (0.98) | triton 0.7086 (0.83) |
| pj_m16_n28672_k4096 | a100 | 0.4139 | lt 0.5069 (0.82)<br>cublas 0.5079 (0.81)<br>cublas-fresh 0.5079 (0.81)<br>lt-fresh 0.5079 (0.81) | torch-eager 0.5755 (0.72) | torch-compile 0.4055 (1.02) | **torch-compile-cg 0.4024 (1.03)** | torch-compile-ro 0.5745 (0.72) | triton 0.8284 (0.50) |
| pj_m16_n28672_k4096 | rtx-4090 | 0.6371 | **cublas-fresh 0.6277 (1.02)**<br>lt-fresh 0.6277 (1.02)<br>cublas 0.6285 (1.01)<br>lt 0.6287 (1.01) | torch-eager 0.6861 (0.93) | torch-compile 0.6328 (1.01) | torch-compile-cg 0.6308 (1.01) | torch-compile-ro 0.6308 (1.01) | triton 0.6523 (0.98) |
| pj_m16_n28672_k4096 | h100 | 0.1943 | lt 0.2167 (0.90)<br>lt-fresh 0.2182 (0.89)<br>cublas 0.2385 (0.81)<br>cublas-fresh 0.2394 (0.81) | torch-eager 0.2383 (0.82) | torch-compile 0.1761 (1.10) | **torch-compile-cg 0.1725 (1.13)** | torch-compile-ro 0.2381 (0.82) | triton 0.2194 (0.89) |
| pj_m16_n4096_k14336 | rtx-5080 | 0.2597 | cublas-fresh 0.3174 (0.82)<br>lt-fresh 0.3174 (0.82)<br>cublas 0.3338 (0.78)<br>lt 0.3338 (0.78) | torch-eager 0.3338 (0.78) | **torch-compile 0.2765 (0.94)** | torch-compile-cg 0.2765 (0.94) | torch-compile-ro 0.3338 (0.78) | triton 0.3052 (0.85) |
| pj_m16_n4096_k14336 | rtx-3090 | 0.2953 | lt-fresh 0.301 (0.98)<br>lt 0.3011 (0.98)<br>cublas-fresh 0.3011 (0.98)<br>cublas 0.3031 (0.97) | torch-eager 0.3133 (0.94) | **torch-compile 0.2755 (1.07)** | torch-compile-cg 0.297 (0.99) | torch-compile-ro 0.3133 (0.94) | triton 0.4998 (0.59) |
| pj_m16_n4096_k14336 | a100 | 0.2249 | **lt 0.256 (0.88)**<br>lt-fresh 0.256 (0.88)<br>cublas 0.257 (0.87)<br>cublas-fresh 0.257 (0.87) | torch-eager 0.3082 (0.73) | torch-compile 0.3092 (0.73) | torch-compile-cg 0.3062 (0.73) | torch-compile-ro 0.3062 (0.73) | triton 0.9277 (0.24) |
| pj_m16_n4096_k14336 | rtx-4090 | 0.3636 | **cublas-fresh 0.343 (1.06)**<br>lt-fresh 0.3451 (1.05)<br>lt 0.3504 (1.04)<br>cublas 0.3526 (1.03) | torch-eager 0.3564 (1.02) | torch-compile 0.3462 (1.05) | torch-compile-cg 0.3461 (1.05) | torch-compile-ro 0.3543 (1.03) | triton 0.3553 (1.02) |
| pj_m16_n4096_k14336 | h100 | 0.1232 | **lt 0.1155 (1.07)**<br>lt-fresh 0.1163 (1.06)<br>cublas 0.1329 (0.93)<br>cublas-fresh 0.1337 (0.92) | torch-eager 0.1333 (0.92) | torch-compile 0.1325 (0.93) | torch-compile-cg 0.133 (0.93) | torch-compile-ro 0.1334 (0.92) | triton 0.3796 (0.32) |
| pj_m16_n4096_k4096 | rtx-5080 | 0.04951 | **cublas-fresh 0.0553 (0.90)**<br>lt-fresh 0.0553 (0.90)<br>lt 0.0963 (0.51)<br>cublas 0.0983 (0.50) | torch-eager 0.0983 (0.50) | torch-compile 0.082 (0.60) | torch-compile-cg 0.0819 (0.60) | torch-compile-ro 0.0983 (0.50) | triton 0.0922 (0.54) |
| pj_m16_n4096_k4096 | rtx-3090 | 0.119 | lt 0.1024 (1.16)<br>cublas 0.1044 (1.14)<br>lt-fresh 0.1044 (1.14)<br>cublas-fresh 0.1055 (1.13) | torch-eager 0.1024 (1.16) | torch-compile 0.086 (1.38) | **torch-compile-cg 0.085 (1.40)** | torch-compile-ro 0.1014 (1.17) | triton 0.1454 (0.82) |
| pj_m16_n4096_k4096 | a100 | 0.09687 | **cublas-fresh 0.0881 (1.10)**<br>lt-fresh 0.0881 (1.10)<br>lt 0.0911 (1.06)<br>cublas 0.0922 (1.05) | torch-eager 0.1055 (0.92) | torch-compile 0.1055 (0.92) | torch-compile-cg 0.1044 (0.93) | torch-compile-ro 0.1034 (0.94) | triton 0.255 (0.38) |
| pj_m16_n4096_k4096 | rtx-4090 | 0.05707 | **lt-fresh 0.0338 (1.69)**<br>cublas-fresh 0.0369 (1.55)<br>lt 0.1219 (0.47)<br>cublas 0.1272 (0.45) | torch-eager 0.128 (0.45) | torch-compile 0.1147 (0.50) | torch-compile-cg 0.1147 (0.50) | torch-compile-ro 0.1251 (0.46) | triton 0.1167 (0.49) |
| pj_m16_n4096_k4096 | h100 | 0.05455 | **lt-fresh 0.0474 (1.15)**<br>lt 0.0476 (1.15)<br>cublas-fresh 0.0902 (0.60)<br>cublas 0.0907 (0.60) | torch-eager 0.0911 (0.60) | torch-compile 0.0509 (1.07) | torch-compile-cg 0.051 (1.07) | torch-compile-ro 0.0911 (0.60) | triton 0.1129 (0.48) |
| pj_m16_n6144_k4096 | rtx-5080 | 0.09791 | **lt-fresh 0.0881 (1.11)**<br>cublas-fresh 0.0963 (1.02)<br>lt 0.129 (0.76)<br>cublas 0.1393 (0.70) | torch-eager 0.1393 (0.70) | torch-compile 0.1208 (0.81) | torch-compile-cg 0.1208 (0.81) | torch-compile-ro 0.1413 (0.69) | triton 0.129 (0.76) |
| pj_m16_n6144_k4096 | rtx-3090 | 0.1489 | lt 0.1331 (1.12)<br>lt-fresh 0.1341 (1.11)<br>cublas 0.1434 (1.04)<br>cublas-fresh 0.1444 (1.03) | torch-eager 0.1464 (1.02) | **torch-compile 0.1239 (1.20)** | torch-compile-cg 0.1239 (1.20) | torch-compile-ro 0.1454 (1.02) | triton 0.1761 (0.85) |
| pj_m16_n6144_k4096 | a100 | 0.1249 | lt-fresh 0.1208 (1.03)<br>lt 0.1219 (1.02)<br>cublas-fresh 0.1341 (0.93)<br>cublas 0.1362 (0.92) | torch-eager 0.1454 (0.86) | **torch-compile 0.1147 (1.09)** | torch-compile-cg 0.1147 (1.09) | torch-compile-ro 0.1434 (0.87) | triton 0.3246 (0.38) |
| pj_m16_n6144_k4096 | rtx-4090 | 0.1332 | **cublas-fresh 0.1137 (1.17)**<br>lt-fresh 0.1205 (1.11)<br>lt 0.1783 (0.75)<br>cublas 0.1833 (0.73) | torch-eager 0.1844 (0.72) | torch-compile 0.1711 (0.78) | torch-compile-cg 0.171 (0.78) | torch-compile-ro 0.1833 (0.73) | triton 0.178 (0.75) |
| pj_m16_n6144_k4096 | h100 | 0.0719 | lt-fresh 0.0663 (1.08)<br>lt 0.0664 (1.08)<br>cublas-fresh 0.0932 (0.77)<br>cublas 0.0945 (0.76) | torch-eager 0.0942 (0.76) | **torch-compile 0.0573 (1.25)** | torch-compile-cg 0.0573 (1.25) | torch-compile-ro 0.0941 (0.76) | triton 0.1132 (0.64) |
| pj_m1_n18432_k7168 | rtx-5080 | 0.5848 | **lt-fresh 0.5571 (1.05)**<br>cublas-fresh 0.6144 (0.95)<br>cublas 0.6164 (0.95)<br>lt 0.6205 (0.94) | torch-eager 0.6205 (0.94) | torch-compile 0.6083 (0.96) | torch-compile-cg 0.6083 (0.96) | torch-compile-ro 0.6185 (0.95) | triton 0.639 (0.92) |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6126 | **cublas 0.6052 (1.01)**<br>lt 0.6052 (1.01)<br>cublas-fresh 0.6052 (1.01)<br>lt-fresh 0.6052 (1.01) | torch-eager 0.6564 (0.93) | torch-compile 0.6093 (1.01) | torch-compile-cg 0.6083 (1.01) | torch-compile-ro 0.6636 (0.92) | triton 0.8878 (0.69) |
| pj_m1_n18432_k7168 | a100 | 0.4152 | **lt-fresh 0.3922 (1.06)**<br>lt 0.3963 (1.05)<br>cublas-fresh 0.4035 (1.03)<br>cublas 0.4229 (0.98) | torch-eager 0.4434 (0.94) | torch-compile 0.3994 (1.04) | torch-compile-cg 0.3994 (1.04) | torch-compile-ro 0.4168 (1.00) | triton 0.9605 (0.43) |
| pj_m1_n18432_k7168 | rtx-4090 | 0.713 | lt-fresh 0.6883 (1.04)<br>lt 0.6935 (1.03)<br>cublas-fresh 0.7035 (1.01)<br>cublas 0.7052 (1.01) | torch-eager 0.7701 (0.93) | torch-compile 0.6502 (1.10) | **torch-compile-cg 0.6495 (1.10)** | torch-compile-ro 0.7045 (1.01) | triton 0.7844 (0.91) |
| pj_m1_n18432_k7168 | h100 | 0.1955 | **lt-fresh 0.1855 (1.05)**<br>cublas-fresh 0.1884 (1.04)<br>lt 0.1923 (1.02)<br>cublas 0.1949 (1.00) | torch-eager 0.1963 (1.00) | torch-compile 0.1979 (0.99) | torch-compile-cg 0.1973 (0.99) | torch-compile-ro 0.1965 (0.99) | triton 0.355 (0.55) |
| pj_m1_n2048_k7168 | rtx-5080 | 0.02538 | **cublas-fresh 0.0266 (0.95)**<br>lt-fresh 0.0266 (0.95)<br>cublas 0.0758 (0.33)<br>lt 0.0758 (0.33) | torch-eager 0.0758 (0.33) | torch-compile 0.0758 (0.33) | torch-compile-cg 0.0758 (0.33) | torch-compile-ro 0.0758 (0.33) | triton 0.2028 (0.13) |
| pj_m1_n2048_k7168 | rtx-3090 | 0.08656 | **lt-fresh 0.0748 (1.16)**<br>lt 0.0778 (1.11)<br>cublas 0.0911 (0.95)<br>cublas-fresh 0.0911 (0.95) | torch-eager 0.0942 (0.92) | torch-compile 0.0809 (1.07) | torch-compile-cg 0.0799 (1.08) | torch-compile-ro 0.0901 (0.96) | triton 0.2335 (0.37) |
| pj_m1_n2048_k7168 | a100 | 0.07228 | **cublas-fresh 0.044 (1.64)**<br>lt-fresh 0.0451 (1.60)<br>cublas 0.0553 (1.31)<br>lt 0.0553 (1.31) | torch-eager 0.0553 (1.31) | torch-compile 0.0614 (1.18) | torch-compile-cg 0.0594 (1.22) | torch-compile-ro 0.0543 (1.33) | triton 0.3994 (0.18) |
| pj_m1_n2048_k7168 | rtx-4090 | 0.03212 | **lt-fresh 0.0273 (1.18)**<br>cublas-fresh 0.043 (0.75)<br>lt 0.0954 (0.34)<br>cublas 0.1158 (0.28) | torch-eager 0.119 (0.27) | torch-compile 0.1004 (0.32) | torch-compile-cg 0.0993 (0.32) | torch-compile-ro 0.1137 (0.28) | triton 0.1516 (0.21) |
| pj_m1_n2048_k7168 | h100 | 0.04012 | **cublas-fresh 0.0242 (1.66)**<br>lt-fresh 0.0251 (1.60)<br>lt 0.03 (1.34)<br>cublas 0.0326 (1.23) | torch-eager 0.0323 (1.24) | torch-compile 0.0323 (1.24) | torch-compile-cg 0.0313 (1.28) | torch-compile-ro 0.0324 (1.24) | triton 0.1899 (0.21) |
| pj_m1_n28672_k4096 | rtx-5080 | 0.5186 | **lt-fresh 0.5018 (1.03)**<br>cublas-fresh 0.5222 (0.99)<br>lt 0.5427 (0.96)<br>cublas 0.5591 (0.93) | torch-eager 0.5612 (0.92) | torch-compile 0.5407 (0.96) | torch-compile-cg 0.5407 (0.96) | torch-compile-ro 0.5612 (0.92) | triton 0.5673 (0.91) |
| pj_m1_n28672_k4096 | rtx-3090 | 0.5474 | **cublas 0.5417 (1.01)**<br>lt 0.5417 (1.01)<br>lt-fresh 0.5417 (1.01)<br>cublas-fresh 0.5427 (1.01) | torch-eager 0.5816 (0.94) | torch-compile 0.5427 (1.01) | torch-compile-cg 0.5417 (1.01) | torch-compile-ro 0.5837 (0.94) | triton 0.8356 (0.66) |
| pj_m1_n28672_k4096 | a100 | 0.3792 | lt-fresh 0.3574 (1.06)<br>lt 0.3676 (1.03)<br>cublas 0.4465 (0.85)<br>cublas-fresh 0.4475 (0.85) | torch-eager 0.468 (0.81) | torch-compile 0.3482 (1.09) | **torch-compile-cg 0.3471 (1.09)** | torch-compile-ro 0.4444 (0.85) | triton 0.8212 (0.46) |
| pj_m1_n28672_k4096 | rtx-4090 | 0.6407 | **lt-fresh 0.5315 (1.21)**<br>lt 0.5897 (1.09)<br>cublas 0.6543 (0.98)<br>cublas-fresh 0.6554 (0.98) | torch-eager 0.7055 (0.91) | torch-compile 0.5796 (1.11) | torch-compile-cg 0.5786 (1.11) | torch-compile-ro 0.6543 (0.98) | triton 0.6605 (0.97) |
| pj_m1_n28672_k4096 | h100 | 0.1766 | **lt-fresh 0.1634 (1.08)**<br>cublas-fresh 0.1647 (1.07)<br>lt 0.1716 (1.03)<br>cublas 0.1717 (1.03) | torch-eager 0.1725 (1.02) | torch-compile 0.1716 (1.03) | torch-compile-cg 0.1722 (1.03) | torch-compile-ro 0.1733 (1.02) | triton 0.2568 (0.69) |
| pj_m1_n4096_k14336 | rtx-5080 | 0.2481 | **cublas-fresh 0.2294 (1.08)**<br>lt-fresh 0.2294 (1.08)<br>cublas 0.2744 (0.90)<br>lt 0.2744 (0.90) | torch-eager 0.2765 (0.90) | torch-compile 0.2765 (0.90) | torch-compile-cg 0.2765 (0.90) | torch-compile-ro 0.2765 (0.90) | triton 0.4076 (0.61) |
| pj_m1_n4096_k14336 | rtx-3090 | 0.282 | **cublas 0.2785 (1.01)**<br>lt 0.2785 (1.01)<br>cublas-fresh 0.2796 (1.01)<br>lt-fresh 0.2796 (1.01) | torch-eager 0.2991 (0.94) | torch-compile 0.2816 (1.00) | torch-compile-cg 0.2796 (1.01) | torch-compile-ro 0.299 (0.94) | triton 0.5632 (0.50) |
| pj_m1_n4096_k14336 | a100 | 0.1978 | **lt-fresh 0.1772 (1.12)**<br>cublas-fresh 0.1782 (1.11)<br>lt 0.1864 (1.06)<br>cublas 0.1935 (1.02) | torch-eager 0.1925 (1.03) | torch-compile 0.1833 (1.08) | torch-compile-cg 0.1894 (1.04) | torch-compile-ro 0.1976 (1.00) | triton 0.8704 (0.23) |
| pj_m1_n4096_k14336 | rtx-4090 | 0.3278 | **lt-fresh 0.2478 (1.32)**<br>lt 0.3226 (1.02)<br>cublas-fresh 0.341 (0.96)<br>cublas 0.3594 (0.91) | torch-eager 0.3623 (0.90) | torch-compile 0.3085 (1.06) | torch-compile-cg 0.3072 (1.07) | torch-compile-ro 0.3604 (0.91) | triton 0.3596 (0.91) |
| pj_m1_n4096_k14336 | h100 | 0.102 | **cublas-fresh 0.0819 (1.25)**<br>lt-fresh 0.082 (1.24)<br>cublas 0.0903 (1.13)<br>lt 0.0903 (1.13) | torch-eager 0.0904 (1.13) | torch-compile 0.0941 (1.08) | torch-compile-cg 0.093 (1.10) | torch-compile-ro 0.0905 (1.13) | triton 0.3758 (0.27) |
| pj_m1_n4096_k4096 | rtx-5080 | 0.03959 | **lt-fresh 0.0225 (1.76)**<br>cublas-fresh 0.0246 (1.61)<br>lt 0.084 (0.47)<br>cublas 0.0881 (0.45) | torch-eager 0.0881 (0.45) | torch-compile 0.0881 (0.45) | torch-compile-cg 0.086 (0.46) | torch-compile-ro 0.0881 (0.45) | triton 0.1249 (0.32) |
| pj_m1_n4096_k4096 | rtx-3090 | 0.09445 | **lt 0.086 (1.10)**<br>lt-fresh 0.087 (1.09)<br>cublas-fresh 0.0911 (1.04)<br>cublas 0.0922 (1.02) | torch-eager 0.0922 (1.02) | torch-compile 0.0932 (1.01) | torch-compile-cg 0.0891 (1.06) | torch-compile-ro 0.0922 (1.02) | triton 0.1905 (0.50) |
| pj_m1_n4096_k4096 | a100 | 0.07791 | **cublas-fresh 0.0532 (1.46)**<br>lt-fresh 0.0532 (1.46)<br>cublas 0.0676 (1.15)<br>lt 0.0676 (1.15) | torch-eager 0.0717 (1.09) | torch-compile 0.0707 (1.10) | torch-compile-cg 0.0645 (1.21) | torch-compile-ro 0.0666 (1.17) | triton 0.2437 (0.32) |
| pj_m1_n4096_k4096 | rtx-4090 | 0.03453 | **lt-fresh 0.0195 (1.77)**<br>cublas-fresh 0.0216 (1.60)<br>lt 0.1065 (0.32)<br>cublas 0.1085 (0.32) | torch-eager 0.1147 (0.30) | torch-compile 0.1106 (0.31) | torch-compile-cg 0.1096 (0.32) | torch-compile-ro 0.1137 (0.30) | triton 0.1219 (0.28) |
| pj_m1_n4096_k4096 | h100 | 0.04248 | **cublas-fresh 0.0236 (1.80)**<br>lt-fresh 0.0236 (1.80)<br>lt 0.0312 (1.36)<br>cublas 0.0313 (1.36) | torch-eager 0.0316 (1.34) | torch-compile 0.0364 (1.17) | torch-compile-cg 0.0348 (1.22) | torch-compile-ro 0.0316 (1.34) | triton 0.1116 (0.38) |
| pj_m1_n6144_k4096 | rtx-5080 | 0.08265 | **lt-fresh 0.0512 (1.61)**<br>cublas-fresh 0.0532 (1.55)<br>cublas 0.1249 (0.66)<br>lt 0.1249 (0.66) | torch-eager 0.1249 (0.66) | torch-compile 0.1208 (0.68) | torch-compile-cg 0.1208 (0.68) | torch-compile-ro 0.1249 (0.66) | triton 0.1454 (0.57) |
| pj_m1_n6144_k4096 | rtx-3090 | 0.1341 | lt-fresh 0.1239 (1.08)<br>lt 0.1248 (1.07)<br>cublas 0.1403 (0.96)<br>cublas-fresh 0.1403 (0.96) | torch-eager 0.1402 (0.96) | **torch-compile 0.1219 (1.10)** | torch-compile-cg 0.1219 (1.10) | torch-compile-ro 0.1372 (0.98) | triton 0.2458 (0.55) |
| pj_m1_n6144_k4096 | a100 | 0.1017 | **lt-fresh 0.0778 (1.31)**<br>cublas-fresh 0.0788 (1.29)<br>lt 0.0901 (1.13)<br>cublas 0.0922 (1.10) | torch-eager 0.0973 (1.04) | torch-compile 0.0901 (1.13) | torch-compile-cg 0.0881 (1.15) | torch-compile-ro 0.0911 (1.12) | triton 0.2601 (0.39) |
| pj_m1_n6144_k4096 | rtx-4090 | 0.09443 | **lt-fresh 0.0645 (1.46)**<br>cublas-fresh 0.1311 (0.72)<br>lt 0.1516 (0.62)<br>cublas 0.1802 (0.52) | torch-eager 0.1799 (0.52) | torch-compile 0.1516 (0.62) | torch-compile-cg 0.1505 (0.63) | torch-compile-ro 0.1772 (0.53) | triton 0.1772 (0.53) |
| pj_m1_n6144_k4096 | h100 | 0.05328 | **lt-fresh 0.0361 (1.48)**<br>cublas-fresh 0.0363 (1.47)<br>cublas 0.0436 (1.22)<br>lt 0.0436 (1.22) | torch-eager 0.0431 (1.24) | torch-compile 0.0476 (1.12) | torch-compile-cg 0.0464 (1.15) | torch-compile-ro 0.0436 (1.22) | triton 0.1124 (0.47) |
| pj_m2048_n28672_k4096 | rtx-5080 | 11.77 | **lt-fresh 12.2 (0.96)**<br>cublas-fresh 12.2 (0.96)<br>cublas 12.21 (0.96)<br>lt 12.21 (0.96) | torch-eager 12.94 (0.91) | torch-compile 12.85 (0.92) | torch-compile-cg 12.82 (0.92) | torch-compile-ro 12.84 (0.92) | triton 16.26 (0.72) |
| pj_m2048_n28672_k4096 | rtx-3090 | 19.33 | lt 23.89 (0.81)<br>lt-fresh 23.9 (0.81)<br>cublas-fresh 23.9 (0.81)<br>cublas 23.9 (0.81) | torch-eager 23.61 (0.82) | torch-compile 23.6 (0.82) | torch-compile-cg 23.32 (0.83) | **torch-compile-ro 23.32 (0.83)** | triton 30.68 (0.63) |
| pj_m2048_n28672_k4096 | a100 | 35.37 | cublas-fresh 27.94 (1.27)<br>cublas 27.97 (1.26)<br>lt-fresh 28.93 (1.22)<br>lt 28.95 (1.22) | torch-eager 28.7 (1.23) | **torch-compile 27.88 (1.27)** | torch-compile-cg 28.21 (1.25) | torch-compile-ro 29.04 (1.22) | triton 42.62 (0.83) |
| pj_m2048_n28672_k4096 | rtx-4090 | 9.59 | **lt 8.182 (1.17)**<br>cublas-fresh 8.186 (1.17)<br>cublas 8.201 (1.17)<br>lt-fresh 8.233 (1.16) | torch-eager 8.234 (1.16) | torch-compile 8.238 (1.16) | torch-compile-cg 8.341 (1.15) | torch-compile-ro 8.238 (1.16) | triton 10.59 (0.91) |
| pj_m2048_n28672_k4096 | h100 | 10 | lt-fresh 9.27 (1.08)<br>cublas-fresh 9.271 (1.08)<br>lt 9.273 (1.08)<br>cublas 9.274 (1.08) | **torch-eager 9.206 (1.09)** | torch-compile 9.261 (1.08) | torch-compile-cg 9.264 (1.08) | torch-compile-ro 9.264 (1.08) | triton 10.18 (0.98) |
| pj_m2048_n4096_k14336 | rtx-5080 | 6.419 | **cublas-fresh 6.415 (1.00)**<br>lt-fresh 6.415 (1.00)<br>lt 6.418 (1.00)<br>cublas 6.425 (1.00) | torch-eager 6.536 (0.98) | torch-compile 6.548 (0.98) | torch-compile-cg 6.511 (0.99) | torch-compile-ro 6.643 (0.97) | triton 8.651 (0.74) |
| pj_m2048_n4096_k14336 | rtx-3090 | 10.97 | **lt-fresh 12.14 (0.90)**<br>lt 12.15 (0.90)<br>cublas-fresh 12.85 (0.85)<br>cublas 12.86 (0.85) | torch-eager 12.84 (0.85) | torch-compile 12.62 (0.87) | torch-compile-cg 12.65 (0.87) | torch-compile-ro 12.67 (0.87) | triton 15.41 (0.71) |
| pj_m2048_n4096_k14336 | a100 | 18.02 | cublas 13.97 (1.29)<br>lt 13.99 (1.29)<br>cublas-fresh 13.99 (1.29)<br>lt-fresh 13.99 (1.29) | **torch-eager 13.97 (1.29)** | torch-compile 14.65 (1.23) | torch-compile-cg 14.65 (1.23) | torch-compile-ro 13.97 (1.29) | triton 19.96 (0.90) |
| pj_m2048_n4096_k14336 | rtx-4090 | 4.237 | **cublas 4.186 (1.01)**<br>cublas-fresh 4.188 (1.01)<br>lt 4.214 (1.01)<br>lt-fresh 4.218 (1.00) | torch-eager 4.292 (0.99) | torch-compile 4.226 (1.00) | torch-compile-cg 4.215 (1.01) | torch-compile-ro 4.268 (0.99) | triton 5.391 (0.79) |
| pj_m2048_n4096_k14336 | h100 | 5.168 | **lt-fresh 4.596 (1.12)**<br>cublas-fresh 4.597 (1.12)<br>lt 4.597 (1.12)<br>cublas 4.597 (1.12) | torch-eager 4.661 (1.11) | torch-compile 4.646 (1.11) | torch-compile-cg 4.646 (1.11) | torch-compile-ro 4.65 (1.11) | triton 5.288 (0.98) |
| pj_m2048_n4096_k4096 | rtx-5080 | 1.83 | cublas-fresh 1.782 (1.03)<br>lt-fresh 1.782 (1.03)<br>cublas 1.79 (1.02)<br>lt 1.79 (1.02) | **torch-eager 1.774 (1.03)** | torch-compile 1.784 (1.03) | torch-compile-cg 1.78 (1.03) | torch-compile-ro 1.779 (1.03) | triton 2.314 (0.79) |
| pj_m2048_n4096_k4096 | rtx-3090 | 2.891 | lt 3.511 (0.82)<br>lt-fresh 3.512 (0.82)<br>cublas 3.513 (0.82)<br>cublas-fresh 3.513 (0.82) | torch-eager 3.571 (0.81) | **torch-compile 3.478 (0.83)** | torch-compile-cg 3.591 (0.81) | torch-compile-ro 3.563 (0.81) | triton 4.299 (0.67) |
| pj_m2048_n4096_k4096 | a100 | 5.213 | cublas 4.054 (1.29)<br>cublas-fresh 4.054 (1.29)<br>lt 4.055 (1.29)<br>lt-fresh 4.055 (1.29) | torch-eager 4.06 (1.28) | torch-compile 4.715 (1.11) | **torch-compile-cg 3.959 (1.32)** | torch-compile-ro 3.985 (1.31) | triton 6.092 (0.86) |
| pj_m2048_n4096_k4096 | rtx-4090 | 1.235 | **lt 1.169 (1.06)**<br>cublas 1.169 (1.06)<br>lt-fresh 1.169 (1.06)<br>cublas-fresh 1.17 (1.06) | torch-eager 1.175 (1.05) | torch-compile 1.183 (1.04) | torch-compile-cg 1.183 (1.04) | torch-compile-ro 1.183 (1.04) | triton 1.501 (0.82) |
| pj_m2048_n4096_k4096 | h100 | 1.499 | cublas-fresh 1.331 (1.13)<br>lt-fresh 1.331 (1.13)<br>lt 1.333 (1.12)<br>cublas 1.334 (1.12) | torch-eager 1.336 (1.12) | **torch-compile 1.327 (1.13)** | torch-compile-cg 1.327 (1.13) | torch-compile-ro 1.327 (1.13) | triton 1.521 (0.99) |
| pj_m2048_n6144_k4096 | rtx-5080 | 2.66 | cublas-fresh 2.631 (1.01)<br>lt-fresh 2.632 (1.01)<br>lt 2.638 (1.01)<br>cublas 2.64 (1.01) | **torch-eager 2.616 (1.02)** | torch-compile 2.617 (1.02) | torch-compile-cg 2.616 (1.02) | torch-compile-ro 2.616 (1.02) | triton 3.401 (0.78) |
| pj_m2048_n6144_k4096 | rtx-3090 | 4.553 | **lt-fresh 5.074 (0.90)**<br>cublas 5.075 (0.90)<br>lt 5.076 (0.90)<br>cublas-fresh 5.076 (0.90) | torch-eager 5.079 (0.90) | torch-compile 5.139 (0.89) | torch-compile-cg 5.094 (0.89) | torch-compile-ro 5.135 (0.89) | triton 6.426 (0.71) |
| pj_m2048_n6144_k4096 | a100 | 7.714 | lt 6.759 (1.14)<br>cublas 6.76 (1.14)<br>cublas-fresh 6.76 (1.14)<br>lt-fresh 6.76 (1.14) | torch-eager 6.681 (1.15) | torch-compile 6.678 (1.16) | **torch-compile-cg 6.676 (1.16)** | torch-compile-ro 6.676 (1.16) | triton 9.261 (0.83) |
| pj_m2048_n6144_k4096 | rtx-4090 | 1.862 | **cublas 1.789 (1.04)**<br>lt 1.79 (1.04)<br>lt-fresh 1.79 (1.04)<br>cublas-fresh 1.793 (1.04) | torch-eager 1.804 (1.03) | torch-compile 1.836 (1.01) | torch-compile-cg 1.816 (1.03) | torch-compile-ro 1.804 (1.03) | triton 2.277 (0.82) |
| pj_m2048_n6144_k4096 | h100 | 2.18 | cublas-fresh 1.994 (1.09)<br>lt-fresh 1.995 (1.09)<br>cublas 1.997 (1.09)<br>lt 1.997 (1.09) | torch-eager 1.977 (1.10) | torch-compile 1.976 (1.10) | torch-compile-cg 1.977 (1.10) | **torch-compile-ro 1.976 (1.10)** | triton 2.274 (0.96) |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.7019 | **cublas-tf32-fresh 0.5898 (1.19)**<br>lt-tf32-fresh 0.5898 (1.19)<br>cublas-tf32 0.6226 (1.13)<br>lt-tf32 0.6246 (1.12) | torch-eager-tf32 0.6513 (1.08) | torch-compile-tf32 0.641 (1.10) | torch-compile-cg-tf32 0.6387 (1.10) | torch-compile-ro-tf32 0.6451 (1.09) | triton-tf32 0.6246 (1.12) |
| pj_m128_n28672_k4096_tf32 | rtx-3090 | 1.076 | lt-tf32-fresh 1.126 (0.96)<br>cublas-tf32 1.127 (0.95)<br>lt-tf32 1.127 (0.95)<br>cublas-tf32-fresh 1.127 (0.95) | torch-eager-tf32 1.065 (1.01) | torch-compile-tf32 0.9626 (1.12) | torch-compile-cg-tf32 0.9636 (1.12) | **torch-compile-ro-tf32 0.8735 (1.23)** | triton-tf32 1.188 (0.91) |
| pj_m128_n28672_k4096_tf32 | a100 | 0.5375 | **lt-tf32 0.4588 (1.17)**<br>lt-tf32-fresh 0.4588 (1.17)<br>cublas-tf32 0.4721 (1.14)<br>cublas-tf32-fresh 0.4721 (1.14) | torch-eager-tf32 0.5007 (1.07) | torch-compile-tf32 0.5007 (1.07) | torch-compile-cg-tf32 0.4987 (1.08) | torch-compile-ro-tf32 0.4987 (1.08) | triton-tf32 0.4823 (1.11) |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 0.8771 | **lt-tf32-fresh 0.6277 (1.40)**<br>lt-tf32 0.6309 (1.39)<br>cublas-tf32-fresh 0.642 (1.37)<br>cublas-tf32 0.6472 (1.36) | torch-eager-tf32 0.6912 (1.27) | torch-compile-tf32 0.6454 (1.36) | torch-compile-cg-tf32 0.6474 (1.35) | torch-compile-ro-tf32 0.6482 (1.35) | triton-tf32 0.6711 (1.31) |
| pj_m128_n28672_k4096_tf32 | h100 | 0.3351 | **cublas-tf32-fresh 0.1853 (1.81)**<br>cublas-tf32 0.1855 (1.81)<br>lt-tf32-fresh 0.1855 (1.81)<br>lt-tf32 0.1858 (1.80) | torch-eager-tf32 0.1857 (1.80) | torch-compile-tf32 0.1865 (1.80) | torch-compile-cg-tf32 0.1859 (1.80) | torch-compile-ro-tf32 0.1862 (1.80) | triton-tf32 0.2629 (1.27) |
| pj_m128_n4096_k14336_tf32 | rtx-5080 | 0.405 | **cublas-tf32-fresh 0.3092 (1.31)**<br>lt-tf32-fresh 0.3092 (1.31)<br>cublas-tf32 0.3215 (1.26)<br>lt-tf32 0.3215 (1.26) | torch-eager-tf32 0.3133 (1.29) | torch-compile-tf32 0.3133 (1.29) | torch-compile-cg-tf32 0.3133 (1.29) | torch-compile-ro-tf32 0.3133 (1.29) | triton-tf32 0.3645 (1.11) |
| pj_m128_n4096_k14336_tf32 | rtx-3090 | 0.6761 | lt-tf32 0.98 (0.69)<br>lt-tf32-fresh 0.98 (0.69)<br>cublas-tf32 1.18 (0.57)<br>cublas-tf32-fresh 1.18 (0.57) | torch-eager-tf32 1.015 (0.67) | **torch-compile-tf32 0.6246 (1.08)** | torch-compile-cg-tf32 0.6246 (1.08) | torch-compile-ro-tf32 0.9759 (0.69) | triton-tf32 0.6892 (0.98) |
| pj_m128_n4096_k14336_tf32 | a100 | 0.3493 | **cublas-tf32 0.2376 (1.47)**<br>lt-tf32 0.2376 (1.47)<br>cublas-tf32-fresh 0.2376 (1.47)<br>lt-tf32-fresh 0.2376 (1.47) | torch-eager-tf32 0.2621 (1.33) | torch-compile-tf32 0.2621 (1.33) | torch-compile-cg-tf32 0.2601 (1.34) | torch-compile-ro-tf32 0.2591 (1.35) | triton-tf32 0.3174 (1.10) |
| pj_m128_n4096_k14336_tf32 | rtx-4090 | 0.3899 | **cublas-tf32-fresh 0.3564 (1.09)**<br>lt-tf32-fresh 0.3615 (1.08)<br>lt-tf32 0.3625 (1.08)<br>cublas-tf32 0.3739 (1.04) | torch-eager-tf32 0.3768 (1.03) | torch-compile-tf32 0.3649 (1.07) | torch-compile-cg-tf32 0.3639 (1.07) | torch-compile-ro-tf32 0.3768 (1.03) | triton-tf32 0.3727 (1.05) |
| pj_m128_n4096_k14336_tf32 | h100 | 0.3538 | **lt-tf32 0.1053 (3.36)**<br>lt-tf32-fresh 0.1067 (3.32)<br>cublas-tf32 0.1082 (3.27)<br>cublas-tf32-fresh 0.1085 (3.26) | torch-eager-tf32 0.1078 (3.28) | torch-compile-tf32 0.1075 (3.29) | torch-compile-cg-tf32 0.1076 (3.29) | torch-compile-ro-tf32 0.1075 (3.29) | triton-tf32 0.1452 (2.44) |
| pj_m128_n4096_k4096_tf32 | rtx-5080 | 0.1393 | **cublas-tf32-fresh 0.084 (1.66)**<br>lt-tf32-fresh 0.084 (1.66)<br>lt-tf32 0.1085 (1.28)<br>cublas-tf32 0.1147 (1.21) | torch-eager-tf32 0.1106 (1.26) | torch-compile-tf32 0.1106 (1.26) | torch-compile-cg-tf32 0.1106 (1.26) | torch-compile-ro-tf32 0.1106 (1.26) | triton-tf32 0.1085 (1.28) |
| pj_m128_n4096_k4096_tf32 | rtx-3090 | 0.2187 | lt-tf32 0.1802 (1.21)<br>lt-tf32-fresh 0.1802 (1.21)<br>cublas-tf32-fresh 0.1823 (1.20)<br>cublas-tf32 0.1833 (1.19) | torch-eager-tf32 0.171 (1.28) | torch-compile-tf32 0.1709 (1.28) | **torch-compile-cg-tf32 0.17 (1.29)** | torch-compile-ro-tf32 0.17 (1.29) | triton-tf32 0.1925 (1.14) |
| pj_m128_n4096_k4096_tf32 | a100 | 0.1311 | **cublas-tf32-fresh 0.0829 (1.58)**<br>lt-tf32-fresh 0.0829 (1.58)<br>cublas-tf32 0.085 (1.54)<br>lt-tf32 0.085 (1.54) | torch-eager-tf32 0.0952 (1.38) | torch-compile-tf32 0.0952 (1.38) | torch-compile-cg-tf32 0.0942 (1.39) | torch-compile-ro-tf32 0.0942 (1.39) | triton-tf32 0.0973 (1.35) |
| pj_m128_n4096_k4096_tf32 | rtx-4090 | 0.1041 | **cublas-tf32-fresh 0.0655 (1.59)**<br>lt-tf32-fresh 0.0655 (1.59)<br>lt-tf32 0.1222 (0.85)<br>cublas-tf32 0.1229 (0.85) | torch-eager-tf32 0.1239 (0.84) | torch-compile-tf32 0.1219 (0.85) | torch-compile-cg-tf32 0.1208 (0.86) | torch-compile-ro-tf32 0.1219 (0.85) | triton-tf32 0.1198 (0.87) |
| pj_m128_n4096_k4096_tf32 | h100 | 0.2122 | **cublas-tf32-fresh 0.044 (4.82)**<br>lt-tf32-fresh 0.0442 (4.80)<br>cublas-tf32 0.0448 (4.74)<br>lt-tf32 0.0459 (4.62) | torch-eager-tf32 0.0455 (4.66) | torch-compile-tf32 0.0639 (3.32) | torch-compile-cg-tf32 0.0644 (3.30) | torch-compile-ro-tf32 0.0449 (4.73) | triton-tf32 0.0496 (4.28) |
| pj_m128_n6144_k4096_tf32 | rtx-5080 | 0.1821 | **lt-tf32-fresh 0.127 (1.43)**<br>lt-tf32 0.1577 (1.15)<br>cublas-tf32-fresh 0.1802 (1.01)<br>cublas-tf32 0.2007 (0.91) | torch-eager-tf32 0.1905 (0.96) | torch-compile-tf32 0.172 (1.06) | torch-compile-cg-tf32 0.172 (1.06) | torch-compile-ro-tf32 0.1905 (0.96) | triton-tf32 0.2028 (0.90) |
| pj_m128_n6144_k4096_tf32 | rtx-3090 | 0.2758 | **lt-tf32 0.256 (1.08)**<br>lt-tf32-fresh 0.257 (1.07)<br>cublas-tf32 0.3727 (0.74)<br>cublas-tf32-fresh 0.3727 (0.74) | torch-eager-tf32 0.3297 (0.84) | torch-compile-tf32 0.2847 (0.97) | torch-compile-cg-tf32 0.2836 (0.97) | torch-compile-ro-tf32 0.3297 (0.84) | triton-tf32 0.3441 (0.80) |
| pj_m128_n6144_k4096_tf32 | a100 | 0.1676 | **cublas-tf32 0.1034 (1.62)**<br>lt-tf32 0.1034 (1.62)<br>cublas-tf32-fresh 0.1034 (1.62)<br>lt-tf32-fresh 0.1034 (1.62) | torch-eager-tf32 0.1096 (1.53) | torch-compile-tf32 0.1106 (1.52) | torch-compile-cg-tf32 0.1085 (1.54) | torch-compile-ro-tf32 0.1085 (1.54) | triton-tf32 0.1137 (1.47) |
| pj_m128_n6144_k4096_tf32 | rtx-4090 | 0.2189 | **cublas-tf32-fresh 0.1249 (1.75)**<br>lt-tf32-fresh 0.1251 (1.75)<br>cublas-tf32 0.1802 (1.21)<br>lt-tf32 0.1811 (1.21) | torch-eager-tf32 0.1802 (1.21) | torch-compile-tf32 0.1754 (1.25) | torch-compile-cg-tf32 0.1751 (1.25) | torch-compile-ro-tf32 0.1802 (1.21) | triton-tf32 0.1782 (1.23) |
| pj_m128_n6144_k4096_tf32 | h100 | 0.2349 | **lt-tf32-fresh 0.0524 (4.48)**<br>lt-tf32 0.0529 (4.44)<br>cublas-tf32-fresh 0.0608 (3.86)<br>cublas-tf32 0.0631 (3.72) | torch-eager-tf32 0.0632 (3.72) | torch-compile-tf32 0.0894 (2.63) | torch-compile-cg-tf32 0.0901 (2.61) | torch-compile-ro-tf32 0.0626 (3.75) | triton-tf32 0.0649 (3.62) |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 9.048 | **lt-tf32 8.057 (1.12)**<br>cublas-tf32-fresh 8.058 (1.12)<br>lt-tf32-fresh 8.058 (1.12)<br>cublas-tf32 8.059 (1.12) | torch-eager-tf32 8.389 (1.08) | torch-compile-tf32 8.469 (1.07) | torch-compile-cg-tf32 8.395 (1.08) | torch-compile-ro-tf32 8.383 (1.08) | triton-tf32 8.759 (1.03) |
| pj_m2048_n28672_k4096_tf32 | rtx-3090 | 14.92 | lt-tf32-fresh 14.77 (1.01)<br>cublas-tf32 14.77 (1.01)<br>cublas-tf32-fresh 14.77 (1.01)<br>lt-tf32 14.77 (1.01) | torch-eager-tf32 14.78 (1.01) | torch-compile-tf32 14.77 (1.01) | **torch-compile-cg-tf32 14.76 (1.01)** | torch-compile-ro-tf32 14.77 (1.01) | triton-tf32 15.51 (0.96) |
| pj_m2048_n28672_k4096_tf32 | a100 | 5.934 | lt-tf32 4.644 (1.28)<br>cublas-tf32-fresh 4.644 (1.28)<br>cublas-tf32 4.645 (1.28)<br>lt-tf32-fresh 4.645 (1.28) | **torch-eager-tf32 4.521 (1.31)** | torch-compile-tf32 4.655 (1.27) | torch-compile-cg-tf32 4.684 (1.27) | torch-compile-ro-tf32 4.706 (1.26) | triton-tf32 9.933 (0.60) |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 9.506 | cublas-tf32 5.505 (1.73)<br>cublas-tf32-fresh 5.509 (1.73)<br>lt-tf32-fresh 5.642 (1.68)<br>lt-tf32 5.645 (1.68) | torch-eager-tf32 5.502 (1.73) | torch-compile-tf32 5.867 (1.62) | torch-compile-cg-tf32 5.505 (1.73) | **torch-compile-ro-tf32 5.5 (1.73)** | triton-tf32 5.961 (1.59) |
| pj_m2048_n28672_k4096_tf32 | h100 | 1.965 | **lt-tf32 1.282 (1.53)**<br>lt-tf32-fresh 1.286 (1.53)<br>cublas-tf32-fresh 1.511 (1.30)<br>cublas-tf32 1.514 (1.30) | torch-eager-tf32 1.381 (1.42) | torch-compile-tf32 1.38 (1.42) | torch-compile-cg-tf32 1.373 (1.43) | torch-compile-ro-tf32 1.374 (1.43) | triton-tf32 5.094 (0.39) |
| pj_m2048_n4096_k14336_tf32 | rtx-5080 | 4.359 | **cublas-tf32-fresh 4.244 (1.03)**<br>lt-tf32-fresh 4.244 (1.03)<br>cublas-tf32 4.251 (1.03)<br>lt-tf32 4.251 (1.03) | torch-eager-tf32 4.545 (0.96) | torch-compile-tf32 4.313 (1.01) | torch-compile-cg-tf32 4.301 (1.01) | torch-compile-ro-tf32 4.543 (0.96) | triton-tf32 4.652 (0.94) |
| pj_m2048_n4096_k14336_tf32 | rtx-3090 | 8.433 | lt-tf32-fresh 7.913 (1.07)<br>lt-tf32 7.921 (1.06)<br>cublas-tf32-fresh 7.921 (1.06)<br>cublas-tf32 7.923 (1.06) | torch-eager-tf32 7.912 (1.07) | **torch-compile-tf32 7.85 (1.07)** | torch-compile-cg-tf32 7.912 (1.07) | torch-compile-ro-tf32 7.925 (1.06) | triton-tf32 8.313 (1.01) |
| pj_m2048_n4096_k14336_tf32 | a100 | 2.86 | lt-tf32-fresh 2.36 (1.21)<br>lt-tf32 2.361 (1.21)<br>cublas-tf32-fresh 2.361 (1.21)<br>cublas-tf32 2.362 (1.21) | torch-eager-tf32 2.372 (1.21) | **torch-compile-tf32 2.275 (1.26)** | torch-compile-cg-tf32 2.293 (1.25) | torch-compile-ro-tf32 2.291 (1.25) | triton-tf32 4.669 (0.61) |
| pj_m2048_n4096_k14336_tf32 | rtx-4090 | 3.27 | **lt-tf32-fresh 2.759 (1.19)**<br>lt-tf32 2.761 (1.18)<br>cublas-tf32-fresh 2.763 (1.18)<br>cublas-tf32 2.764 (1.18) | torch-eager-tf32 2.802 (1.17) | torch-compile-tf32 2.951 (1.11) | torch-compile-cg-tf32 2.802 (1.17) | torch-compile-ro-tf32 2.801 (1.17) | triton-tf32 2.952 (1.11) |
| pj_m2048_n4096_k14336_tf32 | h100 | 1.075 | **lt-tf32-fresh 0.5746 (1.87)**<br>cublas-tf32-fresh 0.5747 (1.87)<br>lt-tf32 0.5748 (1.87)<br>cublas-tf32 0.575 (1.87) | torch-eager-tf32 0.5768 (1.86) | torch-compile-tf32 0.575 (1.87) | torch-compile-cg-tf32 0.5754 (1.87) | torch-compile-ro-tf32 0.5771 (1.86) | triton-tf32 2.316 (0.46) |
| pj_m2048_n4096_k4096_tf32 | rtx-5080 | 1.333 | **lt-tf32-fresh 1.159 (1.15)**<br>lt-tf32 1.161 (1.15)<br>cublas-tf32-fresh 1.163 (1.15)<br>cublas-tf32 1.178 (1.13) | torch-eager-tf32 1.311 (1.02) | torch-compile-tf32 1.225 (1.09) | torch-compile-cg-tf32 1.223 (1.09) | torch-compile-ro-tf32 1.3 (1.02) | triton-tf32 1.348 (0.99) |
| pj_m2048_n4096_k4096_tf32 | rtx-3090 | 2.578 | cublas-tf32-fresh 2.508 (1.03)<br>cublas-tf32 2.509 (1.03)<br>lt-tf32 2.509 (1.03)<br>lt-tf32-fresh 2.509 (1.03) | torch-eager-tf32 2.524 (1.02) | torch-compile-tf32 2.635 (0.98) | torch-compile-cg-tf32 2.634 (0.98) | **torch-compile-ro-tf32 2.489 (1.04)** | triton-tf32 2.536 (1.02) |
| pj_m2048_n4096_k4096_tf32 | a100 | 0.8916 | **lt-tf32 0.6083 (1.47)**<br>lt-tf32-fresh 0.6083 (1.47)<br>cublas-tf32-fresh 0.6226 (1.43)<br>cublas-tf32 0.6236 (1.43) | torch-eager-tf32 0.6871 (1.30) | torch-compile-tf32 0.6881 (1.30) | torch-compile-cg-tf32 0.684 (1.30) | torch-compile-ro-tf32 0.682 (1.31) | triton-tf32 1.323 (0.67) |
| pj_m2048_n4096_k4096_tf32 | rtx-4090 | 0.9572 | **lt-tf32 0.8479 (1.13)**<br>lt-tf32-fresh 0.8491 (1.13)<br>cublas-tf32-fresh 0.854 (1.12)<br>cublas-tf32 0.855 (1.12) | torch-eager-tf32 0.8561 (1.12) | torch-compile-tf32 0.8561 (1.12) | torch-compile-cg-tf32 0.855 (1.12) | torch-compile-ro-tf32 0.8552 (1.12) | triton-tf32 0.8591 (1.11) |
| pj_m2048_n4096_k4096_tf32 | h100 | 0.5192 | **lt-tf32-fresh 0.1738 (2.99)**<br>cublas-tf32-fresh 0.1742 (2.98)<br>cublas-tf32 0.1744 (2.98)<br>lt-tf32 0.1746 (2.97) | torch-eager-tf32 0.1768 (2.94) | torch-compile-tf32 0.1757 (2.95) | torch-compile-cg-tf32 0.1759 (2.95) | torch-compile-ro-tf32 0.176 (2.95) | triton-tf32 0.664 (0.78) |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.018 | **lt-tf32-fresh 1.75 (1.15)**<br>cublas-tf32-fresh 1.752 (1.15)<br>cublas-tf32 1.763 (1.14)<br>lt-tf32 1.763 (1.14) | torch-eager-tf32 1.864 (1.08) | torch-compile-tf32 1.798 (1.12) | torch-compile-cg-tf32 1.798 (1.12) | torch-compile-ro-tf32 1.862 (1.08) | triton-tf32 1.972 (1.02) |
| pj_m2048_n6144_k4096_tf32 | rtx-3090 | 3.436 | **cublas-tf32 3.286 (1.05)**<br>lt-tf32 3.286 (1.05)<br>cublas-tf32-fresh 3.287 (1.05)<br>lt-tf32-fresh 3.287 (1.05) | torch-eager-tf32 3.288 (1.04) | torch-compile-tf32 3.921 (0.88) | torch-compile-cg-tf32 3.873 (0.89) | torch-compile-ro-tf32 3.286 (1.05) | triton-tf32 3.56 (0.96) |
| pj_m2048_n6144_k4096_tf32 | a100 | 1.271 | cublas-tf32 0.9165 (1.39)<br>lt-tf32 0.9165 (1.39)<br>cublas-tf32-fresh 0.9165 (1.39)<br>lt-tf32-fresh 0.9165 (1.39) | torch-eager-tf32 0.9236 (1.38) | torch-compile-tf32 0.8653 (1.47) | torch-compile-cg-tf32 0.8806 (1.44) | **torch-compile-ro-tf32 0.8581 (1.48)** | triton-tf32 2.233 (0.57) |
| pj_m2048_n6144_k4096_tf32 | rtx-4090 | 1.411 | **cublas-tf32 1.186 (1.19)**<br>lt-tf32 1.186 (1.19)<br>cublas-tf32-fresh 1.188 (1.19)<br>lt-tf32-fresh 1.189 (1.19) | torch-eager-tf32 1.204 (1.17) | torch-compile-tf32 1.207 (1.17) | torch-compile-cg-tf32 1.206 (1.17) | torch-compile-ro-tf32 1.203 (1.17) | triton-tf32 1.285 (1.10) |
| pj_m2048_n6144_k4096_tf32 | h100 | 0.5655 | **lt-tf32-fresh 0.2569 (2.20)**<br>lt-tf32 0.2573 (2.20)<br>cublas-tf32-fresh 0.3143 (1.80)<br>cublas-tf32 0.3153 (1.79) | torch-eager-tf32 0.3182 (1.78) | torch-compile-tf32 0.3182 (1.78) | torch-compile-cg-tf32 0.3183 (1.78) | torch-compile-ro-tf32 0.3169 (1.78) | triton-tf32 1.053 (0.54) |
| pj_m128_n28672_k4096_bf16 | rtx-5080 | 0.4024 |  | torch-bf16 0.3297 (1.22) | torch-bf16-compile 0.3236 (1.24) | torch-bf16-compile-cg 0.3215 (1.25) | torch-bf16-compile-ro 0.3338 (1.21) | **triton-bf16 0.3174 (1.27)** |
| pj_m128_n28672_k4096_bf16 | rtx-3090 | 0.5212 |  | torch-bf16 0.5663 (0.92) | torch-bf16-compile 0.4833 (1.08) | **torch-bf16-compile-cg 0.4669 (1.12)** | torch-bf16-compile-ro 0.469 (1.11) | triton-bf16 0.5888 (0.89) |
| pj_m128_n28672_k4096_bf16 | a100 | 0.2918 |  | torch-bf16 0.299 (0.98) | torch-bf16-compile 0.2683 (1.09) | torch-bf16-compile-cg 0.2662 (1.10) | torch-bf16-compile-ro 0.3154 (0.93) | **triton-bf16 0.2355 (1.24)** |
| pj_m128_n28672_k4096_bf16 | rtx-4090 | 0.3884 |  | torch-bf16 0.3963 (0.98) | **torch-bf16-compile 0.3584 (1.08)** | torch-bf16-compile-cg 0.3891 (1.00) | torch-bf16-compile-ro 0.4219 (0.92) | not run |
| pj_m128_n28672_k4096_bf16 | h100 | 0.1436 |  | torch-bf16 0.0945 (1.52) | torch-bf16-compile 0.0939 (1.53) | torch-bf16-compile-cg 0.0939 (1.53) | torch-bf16-compile-ro 0.0941 (1.53) | **triton-bf16 0.0924 (1.55)** |
| pj_m128_n4096_k14336_bf16 | rtx-5080 | 0.2209 |  | torch-bf16 0.172 (1.28) | torch-bf16-compile 0.172 (1.28) | torch-bf16-compile-cg 0.172 (1.28) | **torch-bf16-compile-ro 0.1717 (1.29)** | triton-bf16 0.1843 (1.20) |
| pj_m128_n4096_k14336_bf16 | rtx-3090 | 0.4226 |  | torch-bf16 0.3553 (1.19) | **torch-bf16-compile 0.2806 (1.51)** | torch-bf16-compile-cg 0.2806 (1.51) | torch-bf16-compile-ro 0.2806 (1.51) | triton-bf16 0.3174 (1.33) |
| pj_m128_n4096_k14336_bf16 | a100 | 0.2083 |  | torch-bf16 0.126 (1.65) | torch-bf16-compile 0.126 (1.65) | torch-bf16-compile-cg 0.1239 (1.68) | **torch-bf16-compile-ro 0.1229 (1.69)** | triton-bf16 0.1874 (1.11) |
| pj_m128_n4096_k14336_bf16 | rtx-4090 | 0.2264 |  | torch-bf16 0.2253 (1.00) | **torch-bf16-compile 0.2222 (1.02)** | torch-bf16-compile-cg 0.2458 (0.92) | torch-bf16-compile-ro 0.2427 (0.93) | not run |
| pj_m128_n4096_k14336_bf16 | h100 | 0.1652 |  | torch-bf16 0.062 (2.66) | torch-bf16-compile 0.0579 (2.85) | **torch-bf16-compile-cg 0.0578 (2.86)** | torch-bf16-compile-ro 0.0626 (2.64) | triton-bf16 0.0805 (2.05) |
| pj_m128_n4096_k4096_bf16 | rtx-5080 | 0.06639 |  | **torch-bf16 0.0532 (1.25)** | torch-bf16-compile 0.0553 (1.20) | torch-bf16-compile-cg 0.0532 (1.25) | torch-bf16-compile-ro 0.0532 (1.25) | triton-bf16 0.0573 (1.16) |
| pj_m128_n4096_k4096_bf16 | rtx-3090 | 0.1193 |  | torch-bf16 0.086 (1.39) | torch-bf16-compile 0.0922 (1.29) | torch-bf16-compile-cg 0.0911 (1.31) | **torch-bf16-compile-ro 0.085 (1.40)** | triton-bf16 0.0922 (1.29) |
| pj_m128_n4096_k4096_bf16 | a100 | 0.08835 |  | torch-bf16 0.0584 (1.51) | torch-bf16-compile 0.0584 (1.51) | torch-bf16-compile-cg 0.0573 (1.54) | **torch-bf16-compile-ro 0.0563 (1.57)** | triton-bf16 0.0573 (1.54) |
| pj_m128_n4096_k4096_bf16 | rtx-4090 | 0.06972 |  | **torch-bf16 0.0666 (1.05)** | torch-bf16-compile 0.0666 (1.05) | torch-bf16-compile-cg 0.067 (1.04) | torch-bf16-compile-ro 0.0666 (1.05) | not run |
| pj_m128_n4096_k4096_bf16 | h100 | 0.06077 |  | torch-bf16 0.0241 (2.52) | torch-bf16-compile 0.0249 (2.44) | torch-bf16-compile-cg 0.025 (2.43) | **torch-bf16-compile-ro 0.0238 (2.55)** | triton-bf16 0.0324 (1.88) |
| pj_m128_n6144_k4096_bf16 | rtx-5080 | 0.08742 |  | torch-bf16 0.0881 (0.99) | torch-bf16-compile 0.0881 (0.99) | **torch-bf16-compile-cg 0.086 (1.02)** | torch-bf16-compile-ro 0.0881 (0.99) | triton-bf16 0.1044 (0.84) |
| pj_m128_n6144_k4096_bf16 | rtx-3090 | 0.1774 |  | torch-bf16 0.129 (1.37) | torch-bf16-compile 0.129 (1.37) | **torch-bf16-compile-cg 0.128 (1.39)** | torch-bf16-compile-ro 0.128 (1.39) | triton-bf16 0.1546 (1.15) |
| pj_m128_n6144_k4096_bf16 | a100 | 0.1056 |  | torch-bf16 0.0737 (1.43) | torch-bf16-compile 0.0696 (1.52) | **torch-bf16-compile-cg 0.0676 (1.56)** | torch-bf16-compile-ro 0.0727 (1.45) | triton-bf16 0.0676 (1.56) |
| pj_m128_n6144_k4096_bf16 | rtx-4090 | 0.1078 |  | torch-bf16 0.1044 (1.03) | torch-bf16-compile 0.0903 (1.19) | **torch-bf16-compile-cg 0.0901 (1.20)** | torch-bf16-compile-ro 0.1044 (1.03) | not run |
| pj_m128_n6144_k4096_bf16 | h100 | 0.07726 |  | torch-bf16 0.033 (2.34) | **torch-bf16-compile 0.0323 (2.39)** | torch-bf16-compile-cg 0.0328 (2.36) | torch-bf16-compile-ro 0.0324 (2.38) | triton-bf16 0.0383 (2.02) |
| pj_m16_n28672_k4096_bf16 | rtx-5080 | 0.2804 |  | torch-bf16 0.2826 (0.99) | torch-bf16-compile 0.2765 (1.01) | **torch-bf16-compile-cg 0.2744 (1.02)** | torch-bf16-compile-ro 0.2826 (0.99) | triton-bf16 0.2805 (1.00) |
| pj_m16_n28672_k4096_bf16 | rtx-3090 | 0.3052 |  | torch-bf16 0.3215 (0.95) | **torch-bf16-compile 0.2744 (1.11)** | torch-bf16-compile-cg 0.2949 (1.03) | torch-bf16-compile-ro 0.3318 (0.92) | triton-bf16 0.298 (1.02) |
| pj_m16_n28672_k4096_bf16 | a100 | 0.211 |  | torch-bf16 0.215 (0.98) | torch-bf16-compile 0.1884 (1.12) | **torch-bf16-compile-cg 0.1864 (1.13)** | torch-bf16-compile-ro 0.212 (1.00) | triton-bf16 0.2109 (1.00) |
| pj_m16_n28672_k4096_bf16 | rtx-4090 | 0.3679 |  | torch-bf16 0.3615 (1.02) | **torch-bf16-compile 0.3484 (1.06)** | torch-bf16-compile-cg 0.3779 (0.97) | torch-bf16-compile-ro 0.386 (0.95) | not run |
| pj_m16_n28672_k4096_bf16 | h100 | 0.1459 |  | torch-bf16 0.0935 (1.56) | **torch-bf16-compile 0.0901 (1.62)** | torch-bf16-compile-cg 0.0901 (1.62) | torch-bf16-compile-ro 0.0936 (1.56) | triton-bf16 0.0915 (1.59) |
| pj_m16_n4096_k14336_bf16 | rtx-5080 | 0.1178 |  | torch-bf16 0.1802 (0.65) | torch-bf16-compile 0.1413 (0.83) | **torch-bf16-compile-cg 0.1393 (0.85)** | torch-bf16-compile-ro 0.1802 (0.65) | triton-bf16 0.1843 (0.64) |
| pj_m16_n4096_k14336_bf16 | rtx-3090 | 0.168 |  | torch-bf16 0.1516 (1.11) | **torch-bf16-compile 0.1434 (1.17)** | torch-bf16-compile-cg 0.1536 (1.09) | torch-bf16-compile-ro 0.1587 (1.06) | triton-bf16 0.172 (0.98) |
| pj_m16_n4096_k14336_bf16 | a100 | 0.123 |  | torch-bf16 0.1178 (1.04) | torch-bf16-compile 0.1106 (1.11) | **torch-bf16-compile-cg 0.1096 (1.12)** | torch-bf16-compile-ro 0.1147 (1.07) | triton-bf16 0.1526 (0.81) |
| pj_m16_n4096_k14336_bf16 | rtx-4090 | 0.1437 |  | torch-bf16 0.2156 (0.67) | **torch-bf16-compile 0.1956 (0.73)** | torch-bf16-compile-cg 0.2048 (0.70) | torch-bf16-compile-ro 0.2171 (0.66) | not run |
| pj_m16_n4096_k14336_bf16 | h100 | 0.09177 |  | **torch-bf16 0.0553 (1.66)** | torch-bf16-compile 0.0559 (1.64) | torch-bf16-compile-cg 0.0563 (1.63) | torch-bf16-compile-ro 0.056 (1.64) | triton-bf16 0.0807 (1.14) |
| pj_m16_n4096_k4096_bf16 | rtx-5080 | 0.04212 |  | torch-bf16 0.0553 (0.76) | torch-bf16-compile 0.0451 (0.93) | **torch-bf16-compile-cg 0.043 (0.98)** | torch-bf16-compile-ro 0.0553 (0.76) | triton-bf16 0.0594 (0.71) |
| pj_m16_n4096_k4096_bf16 | rtx-3090 | 0.08065 |  | torch-bf16 0.0512 (1.58) | torch-bf16-compile 0.0481 (1.68) | **torch-bf16-compile-cg 0.0471 (1.71)** | torch-bf16-compile-ro 0.0512 (1.58) | triton-bf16 0.0522 (1.55) |
| pj_m16_n4096_k4096_bf16 | a100 | 0.04142 |  | torch-bf16 0.0481 (0.86) | torch-bf16-compile 0.044 (0.94) | **torch-bf16-compile-cg 0.043 (0.96)** | torch-bf16-compile-ro 0.0471 (0.88) | triton-bf16 0.0532 (0.78) |
| pj_m16_n4096_k4096_bf16 | rtx-4090 | 0.04778 |  | torch-bf16 0.0666 (0.72) | torch-bf16-compile 0.0594 (0.80) | **torch-bf16-compile-cg 0.0585 (0.82)** | torch-bf16-compile-ro 0.0659 (0.72) | not run |
| pj_m16_n4096_k4096_bf16 | h100 | 0.05546 |  | torch-bf16 0.0239 (2.32) | torch-bf16-compile 0.0246 (2.25) | torch-bf16-compile-cg 0.025 (2.22) | **torch-bf16-compile-ro 0.0238 (2.33)** | triton-bf16 0.0316 (1.76) |
| pj_m16_n6144_k4096_bf16 | rtx-5080 | 0.04904 |  | torch-bf16 0.0655 (0.75) | **torch-bf16-compile 0.0635 (0.77)** | torch-bf16-compile-cg 0.0635 (0.77) | torch-bf16-compile-ro 0.0655 (0.75) | triton-bf16 0.0676 (0.73) |
| pj_m16_n6144_k4096_bf16 | rtx-3090 | 0.09434 |  | torch-bf16 0.0881 (1.07) | torch-bf16-compile 0.0676 (1.40) | **torch-bf16-compile-cg 0.0666 (1.42)** | torch-bf16-compile-ro 0.0872 (1.08) | triton-bf16 0.0676 (1.40) |
| pj_m16_n6144_k4096_bf16 | a100 | 0.06373 |  | torch-bf16 0.0686 (0.93) | torch-bf16-compile 0.0604 (1.06) | **torch-bf16-compile-cg 0.0594 (1.07)** | torch-bf16-compile-ro 0.0666 (0.96) | triton-bf16 0.0666 (0.96) |
| pj_m16_n6144_k4096_bf16 | rtx-4090 | 0.06278 |  | torch-bf16 0.0897 (0.70) | **torch-bf16-compile 0.087 (0.72)** | torch-bf16-compile-cg 0.087 (0.72) | torch-bf16-compile-ro 0.0891 (0.70) | not run |
| pj_m16_n6144_k4096_bf16 | h100 | 0.05744 |  | torch-bf16 0.0302 (1.90) | **torch-bf16-compile 0.0288 (1.99)** | torch-bf16-compile-cg 0.029 (1.98) | torch-bf16-compile-ro 0.0298 (1.93) | triton-bf16 0.0373 (1.54) |
| pj_m1_n28672_k4096_bf16 | rtx-5080 | 0.2497 |  | torch-bf16 0.2867 (0.87) | **torch-bf16-compile 0.2765 (0.90)** | torch-bf16-compile-cg 0.2785 (0.90) | torch-bf16-compile-ro 0.2867 (0.87) | triton-bf16 0.3031 (0.82) |
| pj_m1_n28672_k4096_bf16 | rtx-3090 | 0.2859 |  | torch-bf16 0.3041 (0.94) | torch-bf16-compile 0.2806 (1.02) | **torch-bf16-compile-cg 0.2785 (1.03)** | torch-bf16-compile-ro 0.3267 (0.88) | triton-bf16 0.4065 (0.70) |
| pj_m1_n28672_k4096_bf16 | a100 | 0.2438 |  | torch-bf16 0.1976 (1.23) | **torch-bf16-compile 0.1833 (1.33)** | torch-bf16-compile-cg 0.1833 (1.33) | torch-bf16-compile-ro 0.1966 (1.24) | triton-bf16 0.2345 (1.04) |
| pj_m1_n28672_k4096_bf16 | rtx-4090 | 0.3371 |  | torch-bf16 0.3564 (0.95) | torch-bf16-compile 0.3062 (1.10) | **torch-bf16-compile-cg 0.3031 (1.11)** | torch-bf16-compile-ro 0.385 (0.88) | not run |
| pj_m1_n28672_k4096_bf16 | h100 | 0.1144 |  | **torch-bf16 0.0911 (1.26)** | torch-bf16-compile 0.0911 (1.26) | torch-bf16-compile-cg 0.0914 (1.25) | torch-bf16-compile-ro 0.0916 (1.25) | triton-bf16 0.1113 (1.03) |
| pj_m1_n4096_k14336_bf16 | rtx-5080 | 0.1015 |  | **torch-bf16 0.1413 (0.72)** | torch-bf16-compile 0.1475 (0.69) | torch-bf16-compile-cg 0.1475 (0.69) | torch-bf16-compile-ro 0.1413 (0.72) | triton-bf16 0.2007 (0.51) |
| pj_m1_n4096_k14336_bf16 | rtx-3090 | 0.1589 |  | **torch-bf16 0.1464 (1.09)** | torch-bf16-compile 0.1567 (1.01) | torch-bf16-compile-cg 0.1567 (1.01) | torch-bf16-compile-ro 0.1608 (0.99) | triton-bf16 0.2324 (0.68) |
| pj_m1_n4096_k14336_bf16 | a100 | 0.1262 |  | torch-bf16 0.1167 (1.08) | **torch-bf16-compile 0.1075 (1.17)** | torch-bf16-compile-cg 0.1075 (1.17) | torch-bf16-compile-ro 0.1157 (1.09) | triton-bf16 0.1659 (0.76) |
| pj_m1_n4096_k14336_bf16 | rtx-4090 | 0.1226 |  | torch-bf16 0.2028 (0.60) | torch-bf16-compile 0.171 (0.72) | **torch-bf16-compile-cg 0.169 (0.73)** | torch-bf16-compile-ro 0.2153 (0.57) | not run |
| pj_m1_n4096_k14336_bf16 | h100 | 0.07014 |  | torch-bf16 0.0551 (1.27) | torch-bf16-compile 0.0536 (1.31) | **torch-bf16-compile-cg 0.0525 (1.34)** | torch-bf16-compile-ro 0.0556 (1.26) | triton-bf16 0.11 (0.64) |
| pj_m1_n4096_k4096_bf16 | rtx-5080 | 0.02687 |  | torch-bf16 0.0696 (0.39) | **torch-bf16-compile 0.0492 (0.55)** | torch-bf16-compile-cg 0.0573 (0.47) | torch-bf16-compile-ro 0.0696 (0.39) | triton-bf16 0.0717 (0.37) |
| pj_m1_n4096_k4096_bf16 | rtx-3090 | 0.06214 |  | torch-bf16 0.0502 (1.24) | torch-bf16-compile 0.0625 (0.99) | **torch-bf16-compile-cg 0.0492 (1.26)** | torch-bf16-compile-ro 0.0502 (1.24) | triton-bf16 0.0963 (0.65) |
| pj_m1_n4096_k4096_bf16 | a100 | 0.04231 |  | torch-bf16 0.0461 (0.92) | **torch-bf16-compile 0.0451 (0.94)** | torch-bf16-compile-cg 0.0481 (0.88) | torch-bf16-compile-ro 0.0471 (0.90) | triton-bf16 0.0635 (0.67) |
| pj_m1_n4096_k4096_bf16 | rtx-4090 | 0.02121 |  | torch-bf16 0.0716 (0.30) | torch-bf16-compile 0.0645 (0.33) | **torch-bf16-compile-cg 0.0635 (0.33)** | torch-bf16-compile-ro 0.0727 (0.29) | not run |
| pj_m1_n4096_k4096_bf16 | h100 | 0.02212 |  | torch-bf16 0.0223 (0.99) | torch-bf16-compile 0.0234 (0.95) | torch-bf16-compile-cg 0.0224 (0.99) | **torch-bf16-compile-ro 0.0218 (1.01)** | triton-bf16 0.0412 (0.54) |
| pj_m1_n6144_k4096_bf16 | rtx-5080 | 0.02839 |  | torch-bf16 0.0799 (0.36) | torch-bf16-compile 0.0655 (0.43) | **torch-bf16-compile-cg 0.0635 (0.45)** | torch-bf16-compile-ro 0.0799 (0.36) | triton-bf16 0.084 (0.34) |
| pj_m1_n6144_k4096_bf16 | rtx-3090 | 0.08146 |  | torch-bf16 0.0778 (1.05) | **torch-bf16-compile 0.0676 (1.21)** | torch-bf16-compile-cg 0.0676 (1.21) | torch-bf16-compile-ro 0.0758 (1.07) | triton-bf16 0.1485 (0.55) |
| pj_m1_n6144_k4096_bf16 | a100 | 0.06971 |  | torch-bf16 0.0645 (1.08) | torch-bf16-compile 0.0553 (1.26) | **torch-bf16-compile-cg 0.0532 (1.31)** | torch-bf16-compile-ro 0.0635 (1.10) | triton-bf16 0.0819 (0.85) |
| pj_m1_n6144_k4096_bf16 | rtx-4090 | 0.028 |  | torch-bf16 0.0922 (0.30) | torch-bf16-compile 0.0904 (0.31) | **torch-bf16-compile-cg 0.0891 (0.31)** | torch-bf16-compile-ro 0.0942 (0.30) | not run |
| pj_m1_n6144_k4096_bf16 | h100 | 0.03508 |  | torch-bf16 0.0311 (1.13) | torch-bf16-compile 0.0322 (1.09) | **torch-bf16-compile-cg 0.0287 (1.22)** | torch-bf16-compile-ro 0.0319 (1.10) | triton-bf16 0.0476 (0.74) |
| pj_m2048_n28672_k4096_bf16 | rtx-5080 | 4.555 |  | torch-bf16 4.111 (1.11) | torch-bf16-compile 4.195 (1.09) | **torch-bf16-compile-cg 4 (1.14)** | torch-bf16-compile-ro 4.274 (1.07) | triton-bf16 4.122 (1.11) |
| pj_m2048_n28672_k4096_bf16 | rtx-3090 | 7.049 |  | torch-bf16 7.267 (0.97) | **torch-bf16-compile 7.199 (0.98)** | torch-bf16-compile-cg 7.266 (0.97) | torch-bf16-compile-ro 7.267 (0.97) | triton-bf16 7.431 (0.95) |
| pj_m2048_n28672_k4096_bf16 | a100 | 3.084 |  | torch-bf16 2.201 (1.40) | torch-bf16-compile 2.201 (1.40) | torch-bf16-compile-cg 2.19 (1.41) | **torch-bf16-compile-ro 2.159 (1.43)** | triton-bf16 4.775 (0.65) |
| pj_m2048_n28672_k4096_bf16 | rtx-4090 | 4.946 |  | torch-bf16 2.771 (1.78) | torch-bf16-compile 2.763 (1.79) | **torch-bf16-compile-cg 2.757 (1.79)** | torch-bf16-compile-ro 2.763 (1.79) | not run |
| pj_m2048_n28672_k4096_bf16 | h100 | 1.66 |  | torch-bf16 0.6104 (2.72) | torch-bf16-compile 0.6086 (2.73) | torch-bf16-compile-cg 0.609 (2.73) | **torch-bf16-compile-ro 0.6085 (2.73)** | triton-bf16 0.6476 (2.56) |
| pj_m2048_n4096_k14336_bf16 | rtx-5080 | 2.267 |  | torch-bf16 2.122 (1.07) | torch-bf16-compile 2.099 (1.08) | **torch-bf16-compile-cg 2.094 (1.08)** | torch-bf16-compile-ro 2.094 (1.08) | triton-bf16 2.158 (1.05) |
| pj_m2048_n4096_k14336_bf16 | rtx-3090 | 3.958 |  | torch-bf16 4.545 (0.87) | torch-bf16-compile 4.527 (0.87) | torch-bf16-compile-cg 4.53 (0.87) | torch-bf16-compile-ro 4.577 (0.86) | **triton-bf16 4.332 (0.91)** |
| pj_m2048_n4096_k14336_bf16 | a100 | 1.446 |  | **torch-bf16 1.073 (1.35)** | torch-bf16-compile 1.134 (1.28) | torch-bf16-compile-cg 1.131 (1.28) | torch-bf16-compile-ro 1.129 (1.28) | triton-bf16 2.796 (0.52) |
| pj_m2048_n4096_k14336_bf16 | rtx-4090 | 1.545 |  | torch-bf16 1.589 (0.97) | **torch-bf16-compile 1.407 (1.10)** | torch-bf16-compile-cg 1.407 (1.10) | torch-bf16-compile-ro 1.558 (0.99) | not run |
| pj_m2048_n4096_k14336_bf16 | h100 | 0.7741 |  | torch-bf16 0.2996 (2.58) | torch-bf16-compile 0.2997 (2.58) | **torch-bf16-compile-cg 0.2978 (2.60)** | torch-bf16-compile-ro 0.2984 (2.59) | triton-bf16 0.3045 (2.54) |
| pj_m2048_n4096_k4096_bf16 | rtx-5080 | 0.6742 |  | torch-bf16 0.6103 (1.10) | torch-bf16-compile 0.6021 (1.12) | **torch-bf16-compile-cg 0.598 (1.13)** | torch-bf16-compile-ro 0.598 (1.13) | triton-bf16 0.6205 (1.09) |
| pj_m2048_n4096_k4096_bf16 | rtx-3090 | 1.252 |  | torch-bf16 1.265 (0.99) | torch-bf16-compile 1.1 (1.14) | **torch-bf16-compile-cg 1.099 (1.14)** | torch-bf16-compile-ro 1.099 (1.14) | triton-bf16 1.292 (0.97) |
| pj_m2048_n4096_k4096_bf16 | a100 | 0.5039 |  | **torch-bf16 0.3389 (1.49)** | torch-bf16-compile 0.3973 (1.27) | torch-bf16-compile-cg 0.3953 (1.27) | torch-bf16-compile-ro 0.3953 (1.27) | triton-bf16 0.8315 (0.61) |
| pj_m2048_n4096_k4096_bf16 | rtx-4090 | 0.4557 |  | torch-bf16 0.4495 (1.01) | torch-bf16-compile 0.4014 (1.14) | **torch-bf16-compile-cg 0.4004 (1.14)** | torch-bf16-compile-ro 0.4485 (1.02) | not run |
| pj_m2048_n4096_k4096_bf16 | h100 | 0.2379 |  | torch-bf16 0.0922 (2.58) | **torch-bf16-compile 0.0917 (2.59)** | torch-bf16-compile-cg 0.092 (2.59) | torch-bf16-compile-ro 0.092 (2.59) | triton-bf16 0.0965 (2.47) |
| pj_m2048_n6144_k4096_bf16 | rtx-5080 | 0.9403 |  | torch-bf16 0.9032 (1.04) | torch-bf16-compile 0.8888 (1.06) | torch-bf16-compile-cg 0.8885 (1.06) | **torch-bf16-compile-ro 0.8868 (1.06)** | triton-bf16 0.9032 (1.04) |
| pj_m2048_n6144_k4096_bf16 | rtx-3090 | 1.63 |  | torch-bf16 1.786 (0.91) | torch-bf16-compile 1.988 (0.82) | torch-bf16-compile-cg 1.964 (0.83) | **torch-bf16-compile-ro 1.592 (1.02)** | triton-bf16 1.787 (0.91) |
| pj_m2048_n6144_k4096_bf16 | a100 | 0.6271 |  | torch-bf16 0.4762 (1.32) | torch-bf16-compile 0.4291 (1.46) | **torch-bf16-compile-cg 0.425 (1.48)** | torch-bf16-compile-ro 0.427 (1.47) | triton-bf16 1.162 (0.54) |
| pj_m2048_n6144_k4096_bf16 | rtx-4090 | 0.6773 |  | torch-bf16 0.682 (0.99) | torch-bf16-compile 0.598 (1.13) | **torch-bf16-compile-cg 0.597 (1.13)** | torch-bf16-compile-ro 0.6769 (1.00) | not run |
| pj_m2048_n6144_k4096_bf16 | h100 | 0.3476 |  | torch-bf16 0.1371 (2.54) | **torch-bf16-compile 0.1367 (2.54)** | torch-bf16-compile-cg 0.1372 (2.53) | torch-bf16-compile-ro 0.1371 (2.54) | triton-bf16 0.1432 (2.43) |

### MLP chains

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| mlp2_2048 | rtx-5080 | 1.826 | **chain-lt2 1.771 (1.03)**<br>chain-streams 1.797 (1.02)<br>chain-lt 1.848 (0.99)<br>chain-eager 1.875 (0.97)<br>chain-cutlass 2.211 (0.83) | torch-eager 2.213 (0.82) | torch-compile 2.244 (0.81) | torch-compile-cg 2.205 (0.83) | torch-compile-ro 2.241 (0.81) | triton 2.43 (0.75) |
| mlp2_2048 | rtx-3090 | 3.165 | **chain-lt2 3.664 (0.86)**<br>chain-streams 3.748 (0.84)<br>chain-lt 3.842 (0.82)<br>chain-eager 3.897 (0.81)<br>chain-cutlass 4.596 (0.69) | torch-eager 3.997 (0.79) | torch-compile 3.929 (0.81) | torch-compile-cg 4.001 (0.79) | torch-compile-ro 3.966 (0.80) | triton 4.462 (0.71) |
| mlp2_2048 | a100 | 5.436 | **chain-streams 3.976 (1.37)**<br>chain-lt2 4.009 (1.36)<br>chain-eager 4.202 (1.29)<br>chain-lt 4.259 (1.28)<br>chain-cutlass 4.877 (1.11) | torch-eager 4.235 (1.28) | torch-compile 4.096 (1.33) | torch-compile-cg 4.128 (1.32) | torch-compile-ro 4.13 (1.32) | triton 4.048 (1.34) |
| mlp2_2048 | rtx-4090 | 1.276 | **chain-lt2 1.233 (1.03)**<br>chain-lt 1.25 (1.02)<br>chain-cutlass 1.26 (1.01)<br>chain-streams 1.262 (1.01)<br>chain-eager 1.281 (1.00) | torch-eager 1.326 (0.96) | torch-compile 1.308 (0.98) | torch-compile-cg 1.279 (1.00) | torch-compile-ro 1.305 (0.98) | triton 1.437 (0.89) |
| mlp2_2048 | h100 | 1.509 | **chain-lt2 1.35 (1.12)**<br>chain-streams 1.35 (1.12)<br>chain-lt 1.361 (1.11)<br>chain-eager 1.385 (1.09)<br>chain-cutlass 1.684 (0.90) | torch-eager 1.411 (1.07) | torch-compile 1.385 (1.09) | torch-compile-cg 1.385 (1.09) | torch-compile-ro 1.388 (1.09) | triton 1.669 (0.90) |
| mlp_2048 | rtx-5080 | 0.8858 | **chain-lt 0.9134 (0.97)**<br>chain-eager 0.9257 (0.96)<br>chain-cutlass 1.089 (0.81) | torch-eager 0.9236 (0.96) | torch-compile 0.9257 (0.96) | torch-compile-cg 0.9236 (0.96) | torch-compile-ro 0.9236 (0.96) | triton 1.207 (0.73) |
| mlp_2048 | rtx-3090 | 1.604 | chain-lt 1.869 (0.86)<br>chain-eager 1.912 (0.84)<br>chain-cutlass 2.229 (0.72) | torch-eager 2.023 (0.79) | **torch-compile 1.716 (0.93)** | torch-compile-cg 2.112 (0.76) | torch-compile-ro 1.842 (0.87) | triton 2.39 (0.67) |
| mlp_2048 | a100 | 2.718 | chain-lt 2.034 (1.34)<br>chain-eager 2.056 (1.32)<br>chain-cutlass 2.386 (1.14) | torch-eager 2.102 (1.29) | torch-compile 2.026 (1.34) | **torch-compile-cg 2.022 (1.34)** | torch-compile-ro 2.023 (1.34) | triton 2.044 (1.33) |
| mlp_2048 | rtx-4090 | 0.6197 | **chain-lt 0.6328 (0.98)**<br>chain-cutlass 0.6392 (0.97)<br>chain-eager 0.6441 (0.96) | torch-eager 0.6615 (0.94) | torch-compile 0.6613 (0.94) | torch-compile-cg 0.6576 (0.94) | torch-compile-ro 0.6558 (0.94) | triton 0.724 (0.86) |
| mlp_2048 | h100 | 0.7587 | **chain-lt 0.6798 (1.12)**<br>chain-eager 0.6928 (1.10)<br>chain-cutlass 0.8345 (0.91) | torch-eager 0.6886 (1.10) | torch-compile 0.6913 (1.10) | torch-compile-cg 0.6907 (1.10) | torch-compile-ro 0.6878 (1.10) | triton 0.8372 (0.91) |
| mlp_512 | rtx-5080 | 0.03946 | chain-eager 0.043 (0.92)<br>chain-lt 0.043 (0.92)<br>chain-cutlass 0.1085 (0.36) | torch-eager 0.043 (0.92) | **torch-compile 0.0328 (1.20)** | torch-compile-cg 0.0328 (1.20) | torch-compile-ro 0.043 (0.92) | triton 0.0328 (1.20) |
| mlp_512 | rtx-3090 | 0.06422 | chain-lt 0.0532 (1.21)<br>chain-eager 0.0573 (1.12)<br>chain-cutlass 0.124 (0.52) | torch-eager 0.0573 (1.12) | torch-compile 0.0512 (1.25) | torch-compile-cg 0.0502 (1.28) | torch-compile-ro 0.0543 (1.18) | **triton 0.0451 (1.42)** |
| mlp_512 | a100 | 0.1401 | **chain-lt 0.0614 (2.28)**<br>chain-eager 0.0655 (2.14)<br>chain-cutlass 0.2038 (0.69) | torch-eager 0.0666 (2.10) | torch-compile 0.0748 (1.87) | torch-compile-cg 0.0717 (1.95) | torch-compile-ro 0.0788 (1.78) | triton 0.0748 (1.87) |
| mlp_512 | rtx-4090 | 0.04583 | chain-lt 0.0369 (1.24)<br>chain-eager 0.0389 (1.18)<br>chain-cutlass 0.0881 (0.52) | torch-eager 0.044 (1.04) | torch-compile 0.0297 (1.54) | **torch-compile-cg 0.0287 (1.60)** | torch-compile-ro 0.042 (1.09) | triton 0.0298 (1.54) |
| mlp_512 | h100 | 0.04757 | chain-lt 0.034 (1.40)<br>chain-eager 0.0372 (1.28)<br>chain-cutlass 0.118 (0.40) | torch-eager 0.0372 (1.28) | torch-compile 0.0336 (1.42) | **torch-compile-cg 0.0323 (1.47)** | torch-compile-ro 0.0335 (1.42) | triton 0.0382 (1.25) |
| smlp_4096_256_1024 | rtx-5080 | 0.1591 | chain-lt 0.1556 (1.02)<br>chain-cutlass 0.1577 (1.01)<br>chain-eager 0.1679 (0.95) | torch-eager 0.1772 (0.90) | **torch-compile 0.1495 (1.06)** | torch-compile-cg 0.1495 (1.06) | torch-compile-ro 0.1638 (0.97) | triton 0.1782 (0.89) |
| smlp_4096_256_1024 | rtx-3090 | 0.2826 | chain-cutlass 0.2621 (1.08)<br>chain-eager 0.2826 (1.00)<br>chain-lt 0.3369 (0.84) | torch-eager 0.3277 (0.86) | **torch-compile 0.2315 (1.22)** | torch-compile-cg 0.2324 (1.22) | torch-compile-ro 0.2775 (1.02) | triton 0.299 (0.95) |
| smlp_4096_256_1024 | a100 | 0.3902 | **chain-eager 0.2918 (1.34)**<br>chain-lt 0.3287 (1.19)<br>chain-cutlass 0.3553 (1.10) | torch-eager 0.3727 (1.05) | torch-compile 0.34 (1.15) | torch-compile-cg 0.3389 (1.15) | torch-compile-ro 0.3604 (1.08) | triton 0.3052 (1.28) |
| smlp_4096_256_1024 | rtx-4090 | 0.1101 | chain-eager 0.1382 (0.80)<br>chain-cutlass 0.1425 (0.77)<br>chain-lt 0.2028 (0.54) | torch-eager 0.1526 (0.72) | torch-compile 0.1065 (1.03) | torch-compile-cg 0.1065 (1.03) | torch-compile-ro 0.1341 (0.82) | **triton 0.1044 (1.05)** |
| smlp_4096_256_1024 | h100 | 0.127 | **chain-lt 0.1111 (1.14)**<br>chain-eager 0.1265 (1.00)<br>chain-cutlass 0.1744 (0.73) | torch-eager 0.1314 (0.97) | torch-compile 0.1175 (1.08) | torch-compile-cg 0.116 (1.09) | torch-compile-ro 0.1177 (1.08) | triton 0.1224 (1.04) |
| mlp2_2048_tf32 | rtx-5080 | 1.57 |  | torch-eager-tf32 1.547 (1.01) | torch-compile-tf32 1.282 ✗gate | torch-compile-cg-tf32 1.28 ✗gate | **torch-compile-ro-tf32 1.513 (1.04)** | triton-tf32 1.437 ✗gate |
| mlp2_2048_tf32 | rtx-3090 | 3.013 |  | torch-eager-tf32 2.843 (1.06) | torch-compile-tf32 2.721 ✗gate | torch-compile-cg-tf32 2.663 ✗gate | **torch-compile-ro-tf32 2.674 (1.13)** | triton-tf32 2.516 ✗gate |
| mlp2_2048_tf32 | a100 | 1.12 |  | torch-eager-tf32 0.8069 (1.39) | torch-compile-tf32 0.7424 (1.51) | **torch-compile-cg-tf32 0.7291 (1.54)** | torch-compile-ro-tf32 0.7342 (1.53) | triton-tf32 0.8305 ✗gate |
| mlp2_2048_tf32 | rtx-4090 | 1.009 |  | torch-eager-tf32 0.9134 (1.10) | torch-compile-tf32 0.8489 (1.19) | **torch-compile-cg-tf32 0.8428 (1.20)** | torch-compile-ro-tf32 0.8428 (1.20) | triton-tf32 0.9533 ✗gate |
| mlp2_2048_tf32 | h100 | 0.5558 |  | torch-eager-tf32 0.242 (2.30) | torch-compile-tf32 0.215 (2.59) | **torch-compile-cg-tf32 0.2106 (2.64)** | torch-compile-ro-tf32 0.2115 (2.63) | triton-tf32 0.6575 ✗gate |
| mlp_2048_tf32 | rtx-5080 | 0.7846 |  | torch-eager-tf32 0.7629 (1.03) | torch-compile-tf32 0.6369 ✗gate | torch-compile-cg-tf32 0.6369 ✗gate | **torch-compile-ro-tf32 0.7598 (1.03)** | triton-tf32 0.7311 ✗gate |
| mlp_2048_tf32 | rtx-3090 | 1.487 |  | torch-eager-tf32 1.422 (1.05) | torch-compile-tf32 0.9984 ✗gate | torch-compile-cg-tf32 1.023 ✗gate | **torch-compile-ro-tf32 1.197 (1.24)** | triton-tf32 1.272 ✗gate |
| mlp_2048_tf32 | a100 | 0.5632 |  | torch-eager-tf32 0.3799 (1.48) | torch-compile-tf32 0.3707 (1.52) | **torch-compile-cg-tf32 0.3676 (1.53)** | torch-compile-ro-tf32 0.3686 (1.53) | triton-tf32 0.3891 ✗gate |
| mlp_2048_tf32 | rtx-4090 | 0.4937 |  | torch-eager-tf32 0.4352 (1.13) | torch-compile-tf32 0.4159 ✗gate | torch-compile-cg-tf32 0.4147 ✗gate | **torch-compile-ro-tf32 0.4239 (1.16)** | triton-tf32 0.4833 ✗gate |
| mlp_2048_tf32 | h100 | 0.35 |  | torch-eager-tf32 0.1114 (3.14) | torch-compile-tf32 0.1221 (2.87) | torch-compile-cg-tf32 0.1395 (2.51) | **torch-compile-ro-tf32 0.1087 (3.22)** | triton-tf32 0.3353 ✗gate |

### Attention / softmax / long rows

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | hand Triton | hand CUDA |
|---|---|---:|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-5080 | 0.03686 | chain-eager 0.041 (0.90) | torch-eager 0.0492 (0.75) | torch-compile 0.0297 (1.24) | **torch-compile-cg 0.0287 (1.28)**<br>torch-sdpa-compile-cg 0.0287 (1.28) | torch-compile-ro 0.0307 (1.20) | torch-sdpa-compile 0.0348 (1.06)<br>torch-sdpa 0.0532 (0.69) | triton 0.1516 (0.24) | chain-fused 0.0348 (1.06) |
| attn_1024_64 | rtx-3090 | 0.06421 | chain-eager 0.0727 (0.88) | torch-eager 0.0932 (0.69) | torch-compile 0.0502 (1.28) | torch-compile-cg 0.0471 (1.36)<br>torch-sdpa-compile-cg 0.0563 (1.14) | **torch-compile-ro 0.0461 (1.39)** | torch-sdpa-compile 0.0594 (1.08)<br>torch-sdpa 0.0841 (0.76) | triton 0.2294 (0.28) | chain-fused 0.0594 (1.08) |
| attn_1024_64 | a100 | 0.06188 | chain-eager 0.0676 (0.92) | torch-eager 0.0963 (0.64) | torch-compile 0.0614 (1.01) | **torch-sdpa-compile-cg 0.0553 (1.12)**<br>torch-compile-cg 0.0563 (1.10) | torch-compile-ro 0.0563 (1.10) | torch-sdpa-compile 0.0614 (1.01)<br>torch-sdpa 0.1004 (0.62) | triton 0.2417 (0.26) | chain-fused 0.0604 (1.02) |
| attn_1024_64 | rtx-4090 | 0.02727 | chain-eager 0.045 (0.61) | torch-eager 0.0522 (0.52) | torch-compile 0.0335 (0.81) | **torch-sdpa-compile-cg 0.0287 (0.95)**<br>torch-compile-cg 0.0317 (0.86) | torch-compile-ro 0.0307 (0.89) | torch-sdpa-compile 0.0328 (0.83)<br>torch-sdpa 0.0505 (0.54) | triton 0.1506 (0.18) | chain-fused 0.0379 (0.72) |
| attn_1024_64 | h100 | 0.04478 | chain-eager 0.0415 (1.08) | torch-eager 0.0569 (0.79) | torch-compile 0.0337 (1.33) | torch-sdpa-compile-cg 0.0288 (1.55)<br>torch-compile-cg 0.0313 (1.43) | **torch-compile-ro 0.0267 (1.68)** | torch-sdpa-compile 0.0322 (1.39)<br>torch-sdpa 0.0543 (0.82) | triton 0.228 (0.20) | chain-fused 0.0359 (1.25) |
| attn_2048_128 | rtx-5080 | 0.1163 | chain-eager 0.1393 (0.84) | torch-eager 0.1516 (0.77) | torch-compile 0.1055 (1.10) | **torch-sdpa-compile-cg 0.0942 (1.24)**<br>torch-compile-cg 0.1004 (1.16) | torch-compile-ro 0.1004 (1.16) | torch-sdpa-compile 0.0973 (1.20)<br>torch-sdpa 0.1434 (0.81) | triton 0.7803 (0.15) | chain-fused 0.127 (0.92) |
| attn_2048_128 | rtx-3090 | 0.3152 | chain-eager 0.2929 (1.08) | torch-eager 0.3737 (0.84) | torch-compile 0.1833 (1.72) | **torch-compile-cg 0.1802 (1.75)**<br>torch-sdpa-compile-cg 0.1823 (1.73) | torch-compile-ro 0.1864 (1.69) | torch-sdpa-compile 0.1833 (1.72)<br>torch-sdpa 0.2785 (1.13) | triton 1.101 (0.29) | chain-fused 0.2621 (1.20) |
| attn_2048_128 | a100 | 0.7319 | chain-eager 0.2253 (3.25) | torch-eager 0.3256 (2.25) | torch-compile 0.214 (3.42) | **torch-sdpa-compile-cg 0.1802 (4.06)**<br>torch-compile-cg 0.2079 (3.52) | torch-compile-ro 0.2079 (3.52) | torch-sdpa-compile 0.2253 (3.25)<br>torch-sdpa 0.299 (2.45) | triton 1.39 (0.53) | chain-fused 0.2284 (3.20) |
| attn_2048_128 | rtx-4090 | 0.09286 | chain-eager 0.1382 (0.67) | torch-eager 0.1536 (0.60) | torch-compile 0.0852 (1.09) | torch-compile-cg 0.0829 (1.12)<br>torch-sdpa-compile-cg 0.0829 (1.12) | **torch-compile-ro 0.0819 (1.13)** | torch-sdpa-compile 0.0911 (1.02)<br>torch-sdpa 0.1512 (0.61) | triton 0.6033 (0.15) | chain-fused 0.1249 (0.74) |
| attn_2048_128 | h100 | 0.1022 | chain-eager 0.1099 (0.93) | torch-eager 0.1352 (0.76) | torch-compile 0.1065 (0.96) | torch-sdpa-compile-cg 0.0864 (1.18)<br>torch-compile-cg 0.1061 (0.96) | **torch-compile-ro 0.0792 (1.29)** | torch-sdpa-compile 0.0917 (1.12)<br>torch-sdpa 0.1202 (0.85) | triton 1.085 (0.09) | chain-fused 0.098 (1.04) |
| attn_2048_64 | rtx-5080 | 0.08032 | chain-eager 0.1065 (0.75) | torch-eager 0.1229 (0.65) | torch-compile 0.084 (0.96) | torch-sdpa-compile-cg 0.0737 (1.09)<br>torch-compile-cg 0.0778 (1.03) | **torch-compile-ro 0.0696 (1.15)** | torch-sdpa-compile 0.0768 (1.05)<br>torch-sdpa 0.1126 (0.71) | triton 0.2949 (0.27) | chain-fused 0.0942 (0.85) |
| attn_2048_64 | rtx-3090 | 0.1887 | chain-eager 0.2335 (0.81) | torch-eager 0.3041 (0.62) | torch-compile 0.1331 (1.42) | **torch-sdpa-compile-cg 0.1311 (1.44)**<br>torch-compile-cg 0.1321 (1.43) | torch-compile-ro 0.1331 (1.42) | torch-sdpa-compile 0.1382 (1.37)<br>torch-sdpa 0.222 (0.85) | triton 0.3912 (0.48) | chain-fused 0.2079 (0.91) |
| attn_2048_64 | a100 | 0.1453 | chain-eager 0.17 (0.85) | torch-eager 0.2161 (0.67) | torch-compile 0.1444 (1.01) | **torch-sdpa-compile-cg 0.1178 (1.23)**<br>torch-compile-cg 0.1341 (1.08) | torch-compile-ro 0.1331 (1.09) | torch-sdpa-compile 0.1464 (0.99)<br>torch-sdpa 0.2232 (0.65) | triton 0.4639 (0.31) | chain-fused 0.1659 (0.88) |
| attn_2048_64 | rtx-4090 | 0.08429 | chain-eager 0.1208 (0.70) | torch-eager 0.1454 (0.58) | torch-compile 0.0807 (1.04) | torch-sdpa-compile-cg 0.0727 (1.16)<br>torch-compile-cg 0.0788 (1.07) | **torch-compile-ro 0.0635 (1.33)** | torch-sdpa-compile 0.0758 (1.11)<br>torch-sdpa 0.1362 (0.62) | triton 0.2806 (0.30) | chain-fused 0.1075 (0.78) |
| attn_2048_64 | h100 | 0.09448 | chain-eager 0.0923 (1.02) | torch-eager 0.1182 (0.80) | torch-compile 0.0733 (1.29) | torch-sdpa-compile-cg 0.0617 (1.53)<br>torch-compile-cg 0.0685 (1.38) | **torch-compile-ro 0.0599 (1.58)** | torch-sdpa-compile 0.0638 (1.48)<br>torch-sdpa 0.1026 (0.92) | triton 0.4478 (0.21) | chain-fused 0.0771 (1.23) |
| causal_1024_256 | rtx-5080 | 15.16 |  | torch-eager 87.82 (0.17) | torch-compile 57.21 (0.27) | torch-compile-cg 46.44 (0.33) | **torch-compile-ro 44.09 (0.34)** |  |  |  |
| causal_1024_256 | rtx-3090 | 25.15 |  | torch-eager 268.8 (0.09) | torch-compile 159.8 (0.16) | **torch-compile-cg 138.1 (0.18)** | torch-compile-ro 139.9 (0.18) |  |  |  |
| causal_1024_256 | a100 | 37.16 |  | torch-eager 216.6 (0.17) | torch-compile 108.4 (0.34) | **torch-compile-cg 105 (0.35)** | torch-compile-ro 110.3 (0.34) |  |  |  |
| causal_1024_256 | rtx-4090 | 18.06 |  | torch-eager 482.5 (0.04) | torch-compile 279.7 (0.06) | **torch-compile-cg 253.3 (0.07)** | torch-compile-ro 253.8 (0.07) |  |  |  |
| causal_1024_256 | h100 | 21.29 |  | torch-eager 204.5 (0.10) | torch-compile 112.2 (0.19) | **torch-compile-cg 107.9 (0.20)** | torch-compile-ro 109.8 (0.19) |  |  |  |
| decode_1024_256 | rtx-5080 | 8.654 | chain-nocache 23.75 (0.36) | torch-eager 46.81 (0.18) | torch-compile 44.09 (0.20) | torch-compile-cg 39.88 (0.22) | torch-compile-ro 41.6 (0.21) |  |  | **chain-cache 16.36 (0.53)** |
| decode_1024_256 | rtx-3090 | 17.46 | chain-nocache 38.61 (0.45) | torch-eager 148.5 (0.12) | torch-compile 141 (0.12) | torch-compile-cg 139.3 (0.13) | torch-compile-ro 138.9 (0.13) |  |  | **chain-cache 23.55 (0.74)** |
| decode_1024_256 | a100 | 26.7 | chain-nocache 52.5 (0.51) | torch-eager 113.9 (0.23) | torch-compile 94 (0.28) | torch-compile-cg 102.8 (0.26) | torch-compile-ro 105 (0.25) |  |  | **chain-cache 33.94 (0.79)** |
| decode_1024_256 | rtx-4090 | 11.91 | chain-nocache 35.91 (0.33) | torch-eager 269.3 (0.04) | torch-compile 248.8 (0.05) | torch-compile-cg 254.4 (0.05) | torch-compile-ro 254 (0.05) |  |  | **chain-cache 34.45 (0.35)** |
| decode_1024_256 | h100 | 28.4 | chain-nocache 28.91 (0.98) | torch-eager 98.75 (0.29) | torch-compile 98.07 (0.29) | torch-compile-cg 106.7 (0.27) | torch-compile-ro 107.8 (0.26) |  |  | **chain-cache 20.14 (1.41)** |
| gen_1024_512_256 | rtx-5080 | 7.769 |  | torch-eager 55.28 (0.14) | torch-compile 28.12 (0.28) | torch-compile-cg 21.7 (0.36) | **torch-compile-ro 21.4 (0.36)** |  |  |  |
| gen_1024_512_256 | rtx-3090 | 12.77 |  | torch-eager 131.1 (0.10) | torch-compile 77.77 (0.16) | **torch-compile-cg 67.65 (0.19)** | torch-compile-ro 69.19 (0.18) |  |  |  |
| gen_1024_512_256 | a100 | 18.71 |  | torch-eager 108.7 (0.17) | torch-compile 53.98 (0.35) | **torch-compile-cg 51.44 (0.36)** | torch-compile-ro 55.06 (0.34) |  |  |  |
| gen_1024_512_256 | rtx-4090 | 9.177 |  | torch-eager 240.9 (0.04) | torch-compile 139.1 (0.07) | torch-compile-cg 127.5 (0.07) | **torch-compile-ro 127.4 (0.07)** |  |  |  |
| gen_1024_512_256 | h100 | 10.75 |  | torch-eager 102.3 (0.11) | torch-compile 56.34 (0.19) | **torch-compile-cg 54.06 (0.20)** | torch-compile-ro 54.26 (0.20) |  |  |  |
| lsm_1_131072 | rtx-5080 | 0.01755 |  | torch-eager 0.0348 (0.50) | torch-compile 0.0266 (0.66) | **torch-compile-cg 0.0225 (0.78)**<br>torch-sdpa-compile-cg 0.0246 (0.71) | torch-compile-ro 0.0246 (0.71) | torch-sdpa-compile 0.0246 (0.71)<br>torch-sdpa 0.0512 (0.34) |  |  |
| lsm_1_131072 | rtx-3090 | 0.02859 |  | torch-eager 0.0502 (0.57) | torch-compile 0.0338 (0.85) | torch-sdpa-compile-cg 0.0276 (1.04)<br>torch-compile-cg 0.0307 (0.93) | **torch-compile-ro 0.0266 (1.07)** | torch-sdpa-compile 0.0307 (0.93)<br>torch-sdpa 0.0739 (0.39) |  |  |
| lsm_1_131072 | a100 | 0.02985 |  | torch-eager 0.0553 (0.54) | torch-compile 0.0369 (0.81) | **torch-compile-cg 0.0297 (1.00)**<br>torch-sdpa-compile-cg 0.0307 (0.97) | torch-compile-ro 0.0297 (1.00) | torch-sdpa-compile 0.043 (0.69)<br>torch-sdpa 0.1065 (0.28) |  |  |
| lsm_1_131072 | rtx-4090 | 0.02146 |  | torch-eager 0.0399 (0.54) | torch-compile 0.0287 (0.75) | torch-compile-cg 0.0256 (0.84)<br>torch-sdpa-compile-cg 0.0266 (0.81) | **torch-compile-ro 0.0246 (0.87)** | torch-sdpa-compile 0.0285 (0.75)<br>torch-sdpa 0.0553 (0.39) |  |  |
| lsm_1_131072 | h100 | 0.02892 |  | torch-eager 0.0431 (0.67) | torch-compile 0.0237 (1.22) | **torch-compile-cg 0.02 (1.45)**<br>torch-sdpa-compile-cg 0.0478 (0.61) | torch-compile-ro 0.02 (1.45) | torch-sdpa-compile 0.0528 (0.55)<br>torch-sdpa 0.0586 (0.49) |  |  |
| lsm_32_8192 | rtx-5080 | 0.02372 |  | torch-eager 0.0215 (1.10) | **torch-compile 0.0102 (2.33)** | torch-compile-cg 0.0102 (2.33)<br>torch-sdpa-compile-cg 0.0102 (2.33) | torch-compile-ro 0.0102 (2.33) | torch-sdpa-compile 0.0102 (2.33)<br>torch-sdpa 0.0123 (1.93) | triton 0.0102 (2.33) |  |
| lsm_32_8192 | rtx-3090 | 0.02965 |  | torch-eager 0.0328 (0.90) | torch-compile 0.0143 (2.07) | **torch-compile-cg 0.0102 (2.91)**<br>torch-sdpa-compile-cg 0.0123 (2.41) | torch-compile-ro 0.0113 (2.62) | torch-sdpa-compile 0.0113 (2.62)<br>torch-sdpa 0.0174 (1.70) | triton 0.0132 (2.25) |  |
| lsm_32_8192 | a100 | 0.0271 |  | torch-eager 0.0522 (0.52) | torch-compile 0.0184 (1.47) | torch-compile-cg 0.0154 (1.76)<br>torch-sdpa-compile-cg 0.0164 (1.65) | torch-compile-ro 0.0174 (1.56) | **torch-sdpa-compile 0.0143 (1.90)**<br>torch-sdpa 0.0256 (1.06) | triton 0.0164 (1.65) |  |
| lsm_32_8192 | rtx-4090 | 0.02357 |  | torch-eager 0.0218 (1.08) | torch-compile 0.0092 (2.56) | **torch-compile-cg 0.0082 (2.87)**<br>torch-sdpa-compile-cg 0.0082 (2.87) | torch-compile-ro 0.0082 (2.87) | torch-sdpa-compile 0.0092 (2.56)<br>torch-sdpa 0.0123 (1.92) | triton 0.0092 (2.56) |  |
| lsm_32_8192 | h100 | 0.02859 |  | torch-eager 0.0321 (0.89) | torch-compile 0.0117 (2.44) | torch-compile-cg 0.012 (2.38)<br>torch-sdpa-compile-cg 0.0184 (1.55) | **torch-compile-ro 0.0112 (2.55)** | torch-sdpa 0.0167 (1.71)<br>torch-sdpa-compile 0.0294 (0.97) | triton 0.0572 (0.50) |  |
| lsm_4_32768 | rtx-5080 | 0.01687 |  | torch-eager 0.0256 (0.66) | **torch-compile 0.0143 (1.18)** | torch-compile-cg 0.0143 (1.18)<br>torch-sdpa-compile-cg 0.0143 (1.18) | torch-compile-ro 0.0143 (1.18) | torch-sdpa-compile 0.0154 (1.10)<br>torch-sdpa 0.0184 (0.92) | not run |  |
| lsm_4_32768 | rtx-3090 | 0.02424 |  | torch-eager 0.0399 (0.61) | torch-compile 0.0236 (1.03) | torch-sdpa-compile-cg 0.0174 (1.39)<br>torch-compile-cg 0.0195 (1.24) | **torch-compile-ro 0.0174 (1.39)** | torch-sdpa-compile 0.0215 (1.13)<br>torch-sdpa 0.0256 (0.95) | not run |  |
| lsm_4_32768 | a100 | 0.02269 |  | torch-eager 0.0635 (0.36) | torch-compile 0.0256 (0.89) | **torch-compile-cg 0.0195 (1.16)**<br>torch-sdpa-compile-cg 0.0236 (0.96) | torch-compile-ro 0.0205 (1.11) | torch-sdpa-compile 0.0297 (0.76)<br>torch-sdpa 0.0307 (0.74) | triton 0.0297 (0.76) |  |
| lsm_4_32768 | rtx-4090 | 0.02045 |  | torch-eager 0.0307 (0.67) | torch-compile 0.0174 (1.18) | **torch-sdpa-compile-cg 0.0133 (1.54)**<br>torch-compile-cg 0.0143 (1.43) | torch-compile-ro 0.0143 (1.43) | torch-sdpa-compile 0.0174 (1.18)<br>torch-sdpa 0.0202 (1.01) | not run |  |
| lsm_4_32768 | h100 | 0.02752 |  | torch-eager 0.0385 (0.71) | torch-compile 0.022 (1.25) | **torch-compile-cg 0.0164 (1.68)**<br>torch-sdpa-compile-cg 0.0167 (1.65) | torch-compile-ro 0.018 (1.53) | torch-sdpa-compile 0.0217 (1.27)<br>torch-sdpa 0.0228 (1.21) | not run |  |
| prefill_1024_512_256 | rtx-5080 | 7.777 |  | torch-eager 44.06 (0.18) | torch-compile 29.3 (0.27) | **torch-compile-cg 22.57 (0.34)** | torch-compile-ro 23.22 (0.33) |  |  |  |
| prefill_1024_512_256 | rtx-3090 | 13 |  | torch-eager 138 (0.09) | torch-compile 79 (0.16) | **torch-compile-cg 69.86 (0.19)** | torch-compile-ro 75.13 (0.17) |  |  |  |
| prefill_1024_512_256 | a100 | 19.53 |  | torch-eager 108.7 (0.18) | torch-compile 53.92 (0.36) | **torch-compile-cg 52.72 (0.37)** | torch-compile-ro 55.58 (0.35) |  |  |  |
| prefill_1024_512_256 | rtx-4090 | 9.435 |  | torch-eager 242.8 (0.04) | torch-compile 139.5 (0.07) | **torch-compile-cg 127.8 (0.07)** | torch-compile-ro 128.2 (0.07) |  |  |  |
| prefill_1024_512_256 | h100 | 11.05 |  | torch-eager 103.1 (0.11) | torch-compile 56.4 (0.20) | **torch-compile-cg 54.25 (0.20)** | torch-compile-ro 55.12 (0.20) |  |  |  |
| rowsum_2048 | rtx-5080 | 0.4592 | **chain-eager 0.4639 (0.99)**<br>chain-gemv 0.4649 (0.99) | torch-eager 0.4669 (0.98) | torch-compile 0.4669 (0.98) | torch-compile-cg 0.4649 (0.99) | torch-compile-ro 0.4649 (0.99) |  |  |  |
| rowsum_2048 | rtx-3090 | 0.8282 | chain-gemv 0.8571 (0.97)<br>chain-eager 0.8612 (0.96) | torch-eager 0.7813 (1.06) | torch-compile 0.7782 (1.06) | torch-compile-cg 0.7803 (1.06) | **torch-compile-ro 0.7772 (1.07)** |  |  |  |
| rowsum_2048 | a100 | 1.365 | **chain-eager 0.9902 (1.38)**<br>chain-gemv 0.9943 (1.37) | torch-eager 1.12 (1.22) | torch-compile 1.115 (1.22) | torch-compile-cg 1.113 (1.23) | torch-compile-ro 0.9984 (1.37) |  |  |  |
| rowsum_2048 | rtx-4090 | 0.3214 | **chain-eager 0.327 (0.98)**<br>chain-gemv 0.3287 (0.98) | torch-eager 0.3277 (0.98) | torch-compile 0.34 (0.95) | torch-compile-cg 0.3338 (0.96) | torch-compile-ro 0.3297 (0.97) |  |  |  |
| rowsum_2048 | h100 | 0.4093 | **chain-eager 0.3493 (1.17)**<br>chain-gemv 0.3511 (1.17) | torch-eager 0.3507 (1.17) | torch-compile 0.3501 (1.17) | torch-compile-cg 0.3494 (1.17) | torch-compile-ro 0.3495 (1.17) |  |  |  |
| rowsum_4096 | rtx-5080 | 3.93 | chain-gemv 3.742 (1.05)<br>chain-eager 3.753 (1.05) | torch-eager 3.514 (1.12) | **torch-compile 3.498 (1.12)** | torch-compile-cg 3.562 (1.10) | torch-compile-ro 3.498 (1.12) |  |  |  |
| rowsum_4096 | rtx-3090 | 6.157 | chain-eager 7.009 (0.88)<br>chain-gemv 7.009 (0.88) | torch-eager 7.012 (0.88) | **torch-compile 7.005 (0.88)** | torch-compile-cg 7.082 (0.87) | torch-compile-ro 7.026 (0.88) |  |  |  |
| rowsum_4096 | a100 | 10.13 | chain-eager 7.943 (1.28)<br>chain-gemv 7.946 (1.28) | torch-eager 7.859 (1.29) | torch-compile 9.322 (1.09) | torch-compile-cg 9.311 (1.09) | **torch-compile-ro 7.846 (1.29)** |  |  |  |
| rowsum_4096 | rtx-4090 | 2.527 | chain-gemv 2.428 (1.04)<br>chain-eager 2.448 (1.03) | torch-eager 2.479 (1.02) | **torch-compile 2.423 (1.04)** | torch-compile-cg 2.424 (1.04) | torch-compile-ro 2.451 (1.03) |  |  |  |
| rowsum_4096 | h100 | 2.917 | chain-eager 2.69 (1.08)<br>chain-gemv 2.696 (1.08) | torch-eager 2.701 (1.08) | torch-compile 2.691 (1.08) | torch-compile-cg 2.685 (1.09) | **torch-compile-ro 2.684 (1.09)** |  |  |  |
| softmax_rows_2048 | rtx-5080 | 0.472 | chain-eager 0.5035 (0.94) | torch-eager 0.5192 (0.91) | **torch-compile 0.471 (1.00)** | torch-compile-cg 0.471 (1.00) | torch-compile-ro 0.4731 (1.00) |  | triton 0.6083 (0.78) | chain-fused 0.4915 (0.96) |
| softmax_rows_2048 | rtx-3090 | 0.8616 | chain-eager 1.055 (0.82) | torch-eager 1.034 (0.83) | torch-compile 0.7434 (1.16) | **torch-compile-cg 0.7424 (1.16)** | torch-compile-ro 0.7434 (1.16) |  | triton 0.8848 (0.97) | chain-fused 1.021 (0.84) |
| softmax_rows_2048 | a100 | 1.384 | chain-eager 1.05 (1.32) | torch-eager 1.381 (1.00) | **torch-compile 0.9953 (1.39)** | torch-compile-cg 1.273 (1.09) | torch-compile-ro 1.271 (1.09) |  | triton 1.013 (1.37) | chain-fused 1.041 (1.33) |
| softmax_rows_2048 | rtx-4090 | 0.3316 | chain-eager 0.3594 (0.92) | torch-eager 0.385 (0.86) | **torch-compile 0.3287 (1.01)** | torch-compile-cg 0.3287 (1.01) | torch-compile-ro 0.3287 (1.01) |  | triton 0.3748 (0.88) | chain-fused 0.3512 (0.94) |
| softmax_rows_2048 | h100 | 0.4196 | chain-eager 0.3842 (1.09) | torch-eager 0.4098 (1.02) | torch-compile 0.3566 (1.18) | torch-compile-cg 0.3556 (1.18) | **torch-compile-ro 0.3548 (1.18)** |  | triton 0.4329 (0.97) | chain-fused 0.3674 (1.14) |
| softmax_rows_4096 | rtx-5080 | 3.973 | chain-eager 3.895 (1.02) | torch-eager 4.165 (0.95) | torch-compile 3.559 (1.12) | torch-compile-cg 3.552 (1.12) | **torch-compile-ro 3.54 (1.12)** |  | triton 4.574 (0.87) | chain-fused 3.741 (1.06) |
| softmax_rows_4096 | rtx-3090 | 6.284 | chain-eager 7.41 (0.85) | torch-eager 7.728 (0.81) | **torch-compile 7.123 (0.88)** | torch-compile-cg 7.182 (0.87) | torch-compile-ro 7.165 (0.88) |  | triton 8.523 (0.74) | chain-fused 7.304 (0.86) |
| softmax_rows_4096 | a100 | 10.2 | chain-eager 8.235 (1.24) | torch-eager 8.336 (1.22) | torch-compile 9.287 (1.10) | torch-compile-cg 9.396 (1.09) | **torch-compile-ro 7.921 (1.29)** |  | triton 7.933 (1.29) | chain-fused 8.168 (1.25) |
| softmax_rows_4096 | rtx-4090 | 2.595 | chain-eager 2.735 (0.95) | torch-eager 3.054 (0.85) | torch-compile 2.497 (1.04) | torch-compile-cg 2.464 (1.05) | **torch-compile-ro 2.46 (1.06)** |  | triton 3.003 (0.86) | chain-fused 2.617 (0.99) |
| softmax_rows_4096 | h100 | 2.966 | chain-eager 2.847 (1.04) | torch-eager 2.945 (1.01) | torch-compile 2.725 (1.09) | **torch-compile-cg 2.714 (1.09)** | torch-compile-ro 2.726 (1.09) |  | triton 3.07 (0.97) | chain-fused 2.831 (1.05) |

### LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | part | Mapal ms | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | FlashAttention-2/3 | FlashInfer | hand Triton | Liger |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | rtx-5080 | 1.066 | torch-eager 1.179 (0.90) | **torch-compile 1.089 (0.98)** | torch-compile-cg 1.107 (0.96) | torch-compile-ro 1.117 (0.95) | torch-sdpa 1.582 (0.67)<br>torch-sdpa-efficient 1.916 (0.56) |  |  |  | liger 1.991 (0.54) |
| llm_blkdec_b1_l4096 | rtx-3090 | 1.131 | torch-eager 1.471 (0.77) | **torch-compile 1.216 (0.93)** | torch-compile-cg 1.232 (0.92) | torch-compile-ro 1.473 (0.77) | torch-sdpa 1.928 (0.59)<br>torch-sdpa-efficient 2.643 (0.43) |  |  |  | liger 2.621 (0.43) |
| llm_blkdec_b1_l4096 | a100 | 0.9517 | torch-eager 1.142 (0.83) | **torch-compile 0.8452 (1.13)** | torch-compile-cg 0.8492 (1.12) | torch-compile-ro 0.9787 (0.97) | torch-sdpa 1.37 (0.69)<br>torch-sdpa-efficient 1.662 (0.57) |  |  |  | liger 1.684 (0.57) |
| llm_blkdec_b1_l4096 | rtx-4090 | 1.176 | torch-eager 1.847 (0.64) | torch-compile 1.305 (0.90) | **torch-compile-cg 1.286 (0.91)** | torch-compile-ro 1.354 (0.87) | torch-sdpa 2.098 (0.56)<br>torch-sdpa-efficient 2.439 (0.48) |  |  |  | liger 2.446 (0.48) |
| llm_blkdec_b1_l4096 | h100 | 0.5538 | torch-eager 0.7628 (0.73) | torch-compile 0.4835 (1.15) | **torch-compile-cg 0.4681 (1.18)** | torch-compile-ro 0.5524 (1.00) | torch-sdpa 0.787 (0.70)<br>torch-sdpa-efficient 1.064 (0.52) |  |  |  | liger 1.073 (0.52) |
| llm_blkpre_t2048 | rtx-5080 | 25.17 | torch-eager 35.33 (0.71) | torch-compile 31.93 (0.79) | torch-compile-cg 31.95 (0.79) | torch-compile-ro 29.86 (0.84) | torch-sdpa-efficient 30.9 (0.81)<br>torch-sdpa 34.04 (0.74) |  |  | triton 29.55 (0.85) | **liger 26.83 (0.94)** |
| llm_blkpre_t2048 | rtx-3090 | 58.61 | torch-eager 59.29 (0.99) | torch-compile 52.87 (1.11) | torch-compile-cg 53.02 (1.11) | torch-compile-ro 53.16 (1.10) | torch-sdpa-efficient 51.46 (1.14)<br>torch-sdpa 58.89 (1.00) |  |  | triton 55.52 (1.06) | **liger 50.15 (1.17)** |
| llm_blkpre_t2048 | a100 | 75.41 | torch-eager 67.58 (1.12) | **torch-compile 59.15 (1.27)** | torch-compile-cg 59.17 (1.27) | torch-compile-ro 64.11 (1.18) | torch-sdpa-efficient 61.24 (1.23)<br>torch-sdpa 67.24 (1.12) |  |  | triton 64.19 (1.17) | liger 60.69 (1.24) |
| llm_blkpre_t2048 | rtx-4090 | 17.87 | torch-eager 23.91 (0.75) | torch-compile 19.25 (0.93) | torch-compile-cg 19.23 (0.93) | torch-compile-ro 19.41 (0.92) | torch-sdpa-efficient 18.46 (0.97)<br>torch-sdpa 23.62 (0.76) |  |  | triton 20.05 (0.89) | **liger 18.04 (0.99)** |
| llm_blkpre_t2048 | h100 | 21.18 | torch-eager 21.15 (1.00) | torch-compile 19.58 (1.08) | torch-compile-cg 19.53 (1.08) | torch-compile-ro 19.61 (1.08) | torch-sdpa-efficient 18.83 (1.12)<br>torch-sdpa 21.29 (0.99) |  |  | triton 22.57 (0.94) | **liger 18.66 (1.13)** |
| llm_dec_b1_l2048 | rtx-5080 | 0.2395 | torch-eager 0.3011 (0.80) | torch-compile 0.2636 (0.91) | torch-compile-cg 0.2708 (0.88) | **torch-compile-ro 0.259 (0.92)** | torch-sdpa 0.4895 (0.49)<br>torch-sdpa-efficient 0.6791 (0.35) |  |  |  | liger 0.7366 (0.33) |
| llm_dec_b1_l2048 | rtx-3090 | 0.2693 | torch-eager 0.5442 (0.49) | **torch-compile 0.3477 (0.77)** | torch-compile-cg 0.3654 (0.74) | torch-compile-ro 0.3681 (0.73) | torch-sdpa 0.7119 (0.38)<br>torch-sdpa-efficient 1.036 (0.26) |  |  |  | liger 1.025 (0.26) |
| llm_dec_b1_l2048 | a100 | 0.2143 | torch-eager 0.4302 (0.50) | **torch-compile 0.2671 (0.80)** | torch-compile-cg 0.2753 (0.78) | torch-compile-ro 0.2737 (0.78) | torch-sdpa 0.5586 (0.38)<br>torch-sdpa-efficient 0.6855 (0.31) |  |  |  | liger 0.7263 (0.30) |
| llm_dec_b1_l2048 | rtx-4090 | 0.2704 | torch-eager 0.8824 (0.31) | **torch-compile 0.4096 (0.66)** | torch-compile-cg 0.4176 (0.65) | torch-compile-ro 0.4436 (0.61) | torch-sdpa 0.8966 (0.30)<br>torch-sdpa-efficient 1.109 (0.24) |  |  |  | liger 1.135 (0.24) |
| llm_dec_b1_l2048 | h100 | 0.1098 | torch-eager 0.4127 (0.27) | torch-compile 0.2143 (0.51) | **torch-compile-cg 0.193 (0.57)** | torch-compile-ro 0.2126 (0.52) | torch-sdpa 0.4279 (0.26)<br>torch-sdpa-efficient 0.5571 (0.20) |  |  |  | liger 0.5847 (0.19) |
| llm_dec_b1_l8192 | rtx-5080 | 0.3024 | torch-eager 0.4075 (0.74) | **torch-compile 0.3462 (0.87)** | torch-compile-cg 0.358 (0.84) | torch-compile-ro 0.3547 (0.85) | torch-sdpa 1.303 (0.23)<br>torch-sdpa-efficient 1.941 (0.16) |  |  |  | liger 2.06 (0.15) |
| llm_dec_b1_l8192 | rtx-3090 | 0.3372 | torch-eager 0.6281 (0.54) | **torch-compile 0.4481 (0.75)** | torch-compile-cg 0.4595 (0.73) | torch-compile-ro 0.498 (0.68) | torch-sdpa 1.731 (0.19)<br>torch-sdpa-efficient 3.083 (0.11) |  |  |  | liger 3.049 (0.11) |
| llm_dec_b1_l8192 | a100 | 0.2686 | torch-eager 0.6098 (0.44) | **torch-compile 0.4154 (0.65)** | torch-compile-cg 0.4243 (0.63) | torch-compile-ro 0.4833 (0.56) | torch-sdpa 1.091 (0.25)<br>torch-sdpa-efficient 1.712 (0.16) |  |  |  | liger 1.755 (0.15) |
| llm_dec_b1_l8192 | rtx-4090 | 0.344 | torch-eager 0.9581 (0.36) | torch-compile 0.4818 (0.71) | **torch-compile-cg 0.48 (0.72)** | torch-compile-ro 0.5516 (0.62) | torch-sdpa 1.722 (0.20)<br>torch-sdpa-efficient 2.453 (0.14) |  |  |  | liger 2.478 (0.14) |
| llm_dec_b1_l8192 | h100 | 0.1391 | torch-eager 0.6657 (0.21) | torch-compile 0.3121 (0.45) | **torch-compile-cg 0.2995 (0.46)** | torch-compile-ro 0.4767 (0.29) | torch-sdpa 0.7052 (0.20)<br>torch-sdpa-efficient 1.294 (0.11) |  |  |  | liger 1.325 (0.11) |
| llm_dec_b32_l2048 | rtx-5080 | 0.8996 | torch-eager 2.644 (0.34) | **torch-compile 2.293 (0.39)** | torch-compile-cg 2.31 (0.39) | torch-compile-ro 2.395 (0.38) | torch-sdpa-efficient 8.042 (0.11)<br>torch-sdpa 8.887 (0.10) |  |  |  | liger 8.492 (0.11) |
| llm_dec_b32_l2048 | rtx-3090 | 0.984 | torch-eager 3.18 (0.31) | torch-compile 2.992 (0.33) | torch-compile-cg 3.006 (0.33) | **torch-compile-ro 2.989 (0.33)** | torch-sdpa 9.615 (0.10)<br>torch-sdpa-efficient 11.78 (0.08) |  |  |  | liger 11.64 (0.08) |
| llm_dec_b32_l2048 | a100 | 0.7924 | torch-eager 2.128 (0.37) | torch-compile 1.98 (0.40) | **torch-compile-cg 1.962 (0.40)** | torch-compile-ro 1.976 (0.40) | torch-sdpa-efficient 6.29 (0.13)<br>torch-sdpa 6.368 (0.12) |  |  |  | liger 6.232 (0.13) |
| llm_dec_b32_l2048 | rtx-4090 | 1.013 | torch-eager 3.019 (0.34) | **torch-compile 2.569 (0.39)** | torch-compile-cg 2.599 (0.39) | torch-compile-ro 2.631 (0.38) | torch-sdpa-efficient 6.976 (0.15)<br>torch-sdpa 9.569 (0.11) |  |  |  | liger 7.046 (0.14) |
| llm_dec_b32_l2048 | h100 | 0.3553 | torch-eager 1.407 (0.25) | **torch-compile 1.163 (0.31)** | torch-compile-cg 1.165 (0.30) | torch-compile-ro 1.223 (0.29) | torch-sdpa-efficient 3.561 (0.10)<br>torch-sdpa 3.649 (0.10) |  |  |  | liger 3.602 (0.10) |
| llm_dec_b32_l8192 | rtx-5080 | 2.779 | torch-eager 10.98 (0.25) | torch-compile 9.767 (0.28) | **torch-compile-cg 9.728 (0.29)** | torch-compile-ro 10.24 (0.27) | torch-sdpa-efficient 30.25 (0.09) |  |  |  | not run |
| llm_dec_b32_l8192 | rtx-3090 | 3.15 | torch-eager 12.15 (0.26) | torch-compile 11.47 (0.27) | torch-compile-cg 11.48 (0.27) | **torch-compile-ro 11.33 (0.28)** | torch-sdpa 42.79 (0.07)<br>torch-sdpa-efficient 53.32 (0.06) |  |  |  | liger 52.6 (0.06) |
| llm_dec_b32_l8192 | a100 | 2.11 | torch-eager 7.491 (0.28) | torch-compile 7.11 (0.30) | torch-compile-cg 7.044 (0.30) | **torch-compile-ro 7.043 (0.30)** | torch-sdpa 24.99 (0.08)<br>torch-sdpa-efficient 26.54 (0.08) |  |  |  | liger 24.09 (0.09) |
| llm_dec_b32_l8192 | rtx-4090 | 3.216 | torch-eager 12.07 (0.27) | **torch-compile 10.5 (0.31)** | torch-compile-cg 10.52 (0.31) | torch-compile-ro 10.6 (0.30) | torch-sdpa-efficient 25.34 (0.13)<br>torch-sdpa 35.8 (0.09) |  |  |  | liger 25.37 (0.13) |
| llm_dec_b32_l8192 | h100 | 1.028 | torch-eager 4.537 (0.23) | torch-compile 3.938 (0.26) | **torch-compile-cg 3.931 (0.26)** | torch-compile-ro 4.187 (0.25) | torch-sdpa-efficient 12.72 (0.08)<br>torch-sdpa 13.47 (0.08) |  |  |  | liger 12.74 (0.08) |
| llm_dec_b8_l2048 | rtx-5080 | 0.3876 | torch-eager 0.826 (0.47) | torch-compile 0.7713 (0.50) | **torch-compile-cg 0.7657 (0.51)** | torch-compile-ro 0.8316 (0.47) | torch-sdpa 2.407 (0.16)<br>torch-sdpa-efficient 2.602 (0.15) |  |  |  | liger 2.585 (0.15) |
| llm_dec_b8_l2048 | rtx-3090 | 0.8352 | torch-eager 1.129 (0.74) | **torch-compile 0.9394 (0.89)** | torch-compile-cg 0.9524 (0.88) | torch-compile-ro 1.018 (0.82) | torch-sdpa 2.639 (0.32)<br>torch-sdpa-efficient 3.623 (0.23) |  |  |  | liger 3.568 (0.23) |
| llm_dec_b8_l2048 | a100 | 0.3588 | torch-eager 0.9614 (0.37) | torch-compile 0.7727 (0.46) | **torch-compile-cg 0.695 (0.52)** | torch-compile-ro 0.7525 (0.48) | torch-sdpa 2.076 (0.17)<br>torch-sdpa-efficient 2.407 (0.15) |  |  |  | liger 1.973 (0.18) |
| llm_dec_b8_l2048 | rtx-4090 | 0.4321 | torch-eager 1.205 (0.36) | **torch-compile 0.8825 (0.49)** | torch-compile-cg 0.91 (0.47) | torch-compile-ro 0.9254 (0.47) | torch-sdpa-efficient 2.278 (0.19)<br>torch-sdpa 2.9 (0.15) |  |  |  | liger 2.343 (0.18) |
| llm_dec_b8_l2048 | h100 | 0.1796 | torch-eager 0.6233 (0.29) | **torch-compile 0.4141 (0.43)** | torch-compile-cg 0.4331 (0.41) | torch-compile-ro 0.4962 (0.36) | torch-sdpa-efficient 1.15 (0.16)<br>torch-sdpa 1.153 (0.16) |  |  |  | liger 1.2 (0.15) |
| llm_dec_b8_l8192 | rtx-5080 | 0.8684 | torch-eager 3.313 (0.26) | torch-compile 2.964 (0.29) | **torch-compile-cg 2.513 (0.35)** | torch-compile-ro 2.596 (0.33) | torch-sdpa-efficient 8.797 (0.10)<br>torch-sdpa 10.31 (0.08) |  |  |  | liger 9.206 (0.09) |
| llm_dec_b8_l8192 | rtx-3090 | 1.382 | torch-eager 3.343 (0.41) | torch-compile 3.04 (0.45) | **torch-compile-cg 2.979 (0.46)** | torch-compile-ro 3.074 (0.45) | torch-sdpa 10.89 (0.13)<br>torch-sdpa-efficient 14.7 (0.09) |  |  |  | liger 14.52 (0.10) |
| llm_dec_b8_l8192 | a100 | 0.6829 | torch-eager 2.515 (0.27) | torch-compile 2.212 (0.31) | **torch-compile-cg 2.098 (0.33)** | torch-compile-ro 2.203 (0.31) | torch-sdpa 8.395 (0.08)<br>torch-sdpa-efficient 8.529 (0.08) |  |  |  | liger 6.711 (0.10) |
| llm_dec_b8_l8192 | rtx-4090 | 0.9969 | torch-eager 3.503 (0.28) | **torch-compile 2.873 (0.35)** | torch-compile-cg 2.897 (0.34) | torch-compile-ro 2.914 (0.34) | torch-sdpa-efficient 6.862 (0.15)<br>torch-sdpa 9.449 (0.11) |  |  |  | liger 6.923 (0.14) |
| llm_dec_b8_l8192 | h100 | 0.3534 | torch-eager 1.42 (0.25) | **torch-compile 1.114 (0.32)** | torch-compile-cg 1.132 (0.31) | torch-compile-ro 1.253 (0.28) | torch-sdpa-efficient 3.531 (0.10)<br>torch-sdpa 3.652 (0.10) |  |  |  | liger 3.542 (0.10) |
| llm_pre_t2048 | rtx-5080 | 6.458 | torch-eager 13.5 (0.48) | torch-compile 8.448 (0.76) | torch-compile-cg 8.451 (0.76) | torch-compile-ro 9.474 (0.68) | torch-sdpa-efficient 7.266 (0.89)<br>torch-sdpa 13.74 (0.47) |  |  | triton 9.568 (0.67) | **liger 7.216 (0.90)** |
| llm_pre_t2048 | rtx-3090 | 11.11 | torch-eager 20.21 (0.55) | torch-compile 15.94 (0.70) | torch-compile-cg 16.02 (0.69) | torch-compile-ro 15.67 (0.71) | torch-sdpa-efficient 13.79 (0.81)<br>torch-sdpa 20.11 (0.55) |  |  | triton 18.74 (0.59) | **liger 13.59 (0.82)** |
| llm_pre_t2048 | a100 | 19.28 | torch-eager 26.51 (0.73) | torch-compile 17.56 (1.10) | torch-compile-cg 19.53 (0.99) | torch-compile-ro 17.21 (1.12) | torch-sdpa-efficient 20.33 (0.95)<br>torch-sdpa 23.36 (0.83) |  |  | triton 17.27 (1.12) | **liger 12.94 (1.49)** |
| llm_pre_t2048 | rtx-4090 | 4.756 | torch-eager 11.01 (0.43) | torch-compile 6.332 (0.75) | torch-compile-cg 6.332 (0.75) | torch-compile-ro 6.505 (0.73) | torch-sdpa-efficient 5.24 (0.91)<br>torch-sdpa 10.77 (0.44) |  |  | triton 6.944 (0.68) | **liger 5.164 (0.92)** |
| llm_pre_t2048 | h100 | 5.985 | torch-eager 7.023 (0.85) | torch-compile 5.502 (1.09) | torch-compile-cg 5.459 (1.10) | torch-compile-ro 5.575 (1.07) | torch-sdpa-efficient 4.565 (1.31)<br>torch-sdpa 7.145 (0.84) |  |  | triton 8.423 (0.71) | **liger 4.552 (1.31)** |
| llm_pre_t8192 | rtx-5080 | 44.65 | torch-eager 167.6 (0.27) | torch-compile 85.07 (0.52) | torch-compile-cg 85.51 (0.52) | torch-compile-ro 101.3 (0.44) | torch-sdpa-efficient 53.71 (0.83) |  |  | triton 85.06 (0.52) | **liger 52.89 (0.84)** |
| llm_pre_t8192 | rtx-3090 | 104.3 | torch-eager 210.1 (0.50) | torch-compile 151 (0.69) | torch-compile-cg 150.6 (0.69) | torch-compile-ro 145.3 (0.72) | torch-sdpa-efficient 99.72 (1.05)<br>torch-sdpa 213.2 (0.49) |  |  | triton 180.6 (0.58) | **liger 98.78 (1.06)** |
| llm_pre_t8192 | a100 | 208.6 | torch-eager 222.7 (0.94) | torch-compile 127.1 (1.64) | torch-compile-cg 127.2 (1.64) | torch-compile-ro 127.6 (1.64) | torch-sdpa-efficient 66.68 (3.13)<br>torch-sdpa 168.9 (1.24) |  |  | triton 146.8 (1.42) | **liger 66.2 (3.15)** |
| llm_pre_t8192 | rtx-4090 | 34.83 | torch-eager 131.6 (0.26) | torch-compile 59.89 (0.58) | torch-compile-cg 60.21 (0.58) | torch-compile-ro 60.77 (0.57) | torch-sdpa-efficient 38.19 (0.91) |  |  | triton 61.86 (0.56) | **liger 37.53 (0.93)** |
| llm_pre_t8192 | h100 | 86.56 | torch-eager 67.68 (1.28) | torch-compile 43.87 (1.97) | torch-compile-cg 43.82 (1.98) | torch-compile-ro 44.03 (1.97) | torch-sdpa-efficient 26 (3.33)<br>torch-sdpa 70.85 (1.22) |  |  | triton 73.5 (1.18) | **liger 25.82 (3.35)** |
| llm_blkpre_t2048_tf32 | rtx-5080 | 19.74 | torch-eager-tf32 24.88 (0.79) | torch-compile-tf32 18.96 (1.04) | torch-compile-cg-tf32 18.66 (1.06) | torch-compile-ro-tf32 19.59 (1.01) | torch-sdpa-tf32 24 (0.82) |  |  | **triton-tf32 18.08 (1.09)** | liger-tf32 19.13 (1.03) |
| llm_blkpre_t2048_tf32 | rtx-3090 | 33.22 | torch-eager-tf32 39.35 (0.84) | torch-compile-tf32 43.32 (0.77) | torch-compile-cg-tf32 43.52 (0.76) | torch-compile-ro-tf32 34.07 (0.98) | torch-sdpa-tf32 39.5 (0.84) |  |  | **triton-tf32 31.79 (1.05)** | liger-tf32 33.62 (0.99) |
| llm_blkpre_t2048_tf32 | a100 | 17.41 | torch-eager-tf32 14.57 (1.19) | torch-compile-tf32 11.6 (1.50) | torch-compile-cg-tf32 11.55 (1.51) | torch-compile-ro-tf32 11.65 (1.50) | torch-sdpa-tf32 14.69 (1.19) |  |  | **triton-tf32 10.9 (1.60)** | liger-tf32 11.38 (1.53) |
| llm_blkpre_t2048_tf32 | rtx-4090 | 16.3 | torch-eager-tf32 18.99 (0.86) | torch-compile-tf32 13.64 (1.19) | torch-compile-cg-tf32 13.61 (1.20) | torch-compile-ro-tf32 14.12 (1.15) | torch-sdpa-tf32 18.73 (0.87) |  |  | **triton-tf32 12.37 (1.32)** | liger-tf32 12.95 (1.26) |
| llm_blkpre_t2048_tf32 | h100 | 6.507 | torch-eager-tf32 5.067 (1.28) | torch-compile-tf32 3.604 (1.81) | torch-compile-cg-tf32 3.527 (1.84) | torch-compile-ro-tf32 3.457 (1.88) | torch-sdpa-tf32 5.181 (1.26) |  |  | **triton-tf32 3.44 (1.89)** | liger-tf32 3.725 (1.75) |
| llm_pre_t2048_tf32 | rtx-5080 | 5.342 | torch-eager-tf32 10.78 (0.50) | torch-compile-tf32 6.117 (0.87) | torch-compile-cg-tf32 6.099 (0.88) | torch-compile-ro-tf32 6.416 (0.83) | torch-sdpa-tf32 10.4 (0.51) |  |  | **triton-tf32 4.543 (1.18)** | liger-tf32 5.929 (0.90) |
| llm_pre_t2048_tf32 | rtx-3090 | 9.455 | torch-eager-tf32 15.42 (0.61) | torch-compile-tf32 11.39 (0.83) | torch-compile-cg-tf32 11.36 (0.83) | torch-compile-ro-tf32 10.79 (0.88) | torch-sdpa-tf32 15.69 (0.60) |  |  | **triton-tf32 8.418 (1.12)** | liger-tf32 10.44 (0.91) |
| llm_pre_t2048_tf32 | a100 | 8.595 | torch-eager-tf32 7.011 (1.23) | torch-compile-tf32 4.171 (2.06) | torch-compile-cg-tf32 4.129 (2.08) | torch-compile-ro-tf32 4.192 (2.05) | torch-sdpa-tf32 7.269 (1.18) |  |  | **triton-tf32 3.085 (2.79)** | liger-tf32 3.829 (2.24) |
| llm_pre_t2048_tf32 | rtx-4090 | 4.051 | torch-eager-tf32 9.883 (0.41) | torch-compile-tf32 5.124 (0.79) | torch-compile-cg-tf32 5.124 (0.79) | torch-compile-ro-tf32 5.364 (0.76) | torch-sdpa-tf32 9.704 (0.42) |  |  | **triton-tf32 3.276 (1.24)** | liger-tf32 4.177 (0.97) |
| llm_pre_t2048_tf32 | h100 | 3.131 | torch-eager-tf32 2.978 (1.05) | torch-compile-tf32 1.53 (2.05) | torch-compile-cg-tf32 1.497 (2.09) | torch-compile-ro-tf32 1.508 (2.08) | torch-sdpa-tf32 3.077 (1.02) |  |  | **triton-tf32 1.373 (2.28)** | liger-tf32 1.716 (1.82) |
| llm_blkdec_b1_l4096_bf16w | rtx-5080 | 0.627 | torch-bf16 0.6486 (0.97) | **torch-bf16-compile 0.5569 (1.13)** | torch-bf16-compile-cg 0.5591 (1.12) | torch-bf16-compile-ro 0.9885 (0.63) | torch-bf16-cudnn 0.6793 (0.92)<br>torch-bf16-efficient 0.9947 (0.63) | fa2-bf16 0.6803 (0.92) | flashinfer-bf16 0.822 (0.76) |  | liger-bf16 0.7106 (0.88) |
| llm_blkdec_b1_l4096_bf16w | rtx-3090 | 0.7056 | torch-bf16 0.934 (0.76) | **torch-bf16-compile 0.6743 (1.05)** | torch-bf16-compile-cg 0.6883 (1.03) | torch-bf16-compile-ro 1.257 (0.56) | torch-bf16-cudnn 0.9506 (0.74)<br>torch-bf16-efficient 1.258 (0.56) | fa2-bf16 0.9223 (0.77) | flashinfer-bf16 1.057 (0.67) |  | liger-bf16 0.9117 (0.77) |
| llm_blkdec_b1_l4096_bf16w | a100 | 0.7915 | torch-bf16 0.6659 (1.19) | torch-bf16-compile 0.5082 (1.56) | **torch-bf16-compile-cg 0.4794 (1.65)** | torch-bf16-compile-ro 0.8159 (0.97) | torch-bf16-cudnn 0.7171 (1.10)<br>torch-bf16-efficient 0.9582 (0.83) | fa2-bf16 0.6591 (1.20) | flashinfer-bf16 0.7987 (0.99) |  | liger-bf16 0.717 (1.10) |
| llm_blkdec_b1_l4096_bf16w | rtx-4090 | 0.7082 | torch-bf16 1.305 (0.54) | torch-bf16-compile 0.7886 (0.90) | **torch-bf16-compile-cg 0.7551 (0.94)** | torch-bf16-compile-ro 1.174 (0.60) | torch-bf16-cudnn 1.374 (0.52)<br>torch-bf16-efficient 1.46 (0.48) | fa2-bf16 1.313 (0.54) | flashinfer-bf16 1.564 (0.45) |  | liger-bf16 1.329 (0.53) |
| llm_blkdec_b1_l4096_bf16w | h100 | 0.4931 | torch-bf16 0.542 (0.91) | torch-bf16-compile 0.3395 (1.45) | **torch-bf16-compile-cg 0.286 (1.72)** | torch-bf16-compile-ro 0.2879 (1.71) | torch-bf16-cudnn 0.5835 (0.85)<br>torch-bf16-efficient 0.7069 (0.70) | fa2-bf16 0.5341 (0.92)<br>fa3-bf16 0.5788 (0.85) | flashinfer-bf16 0.6731 (0.73) |  | liger-bf16 0.6021 (0.82) |
| llm_blkpre_t2048_bf16w | rtx-5080 | 25.54 | torch-bf16 10.22 (2.50) | **torch-bf16-compile 8.103 (3.15)** | torch-bf16-compile-cg 9.426 (2.71) | torch-bf16-compile-ro 9.779 (2.61) | torch-bf16-cudnn 8.82 (2.90)<br>torch-bf16-efficient 10.83 (2.36) | fa2-bf16 8.878 (2.88) | flashinfer-bf16 8.881 (2.88) | triton-bf16 8.92 (2.86) | liger-bf16 8.524 (3.00) |
| llm_blkpre_t2048_bf16w | rtx-3090 | 49.78 | torch-bf16 16.56 (3.01) | torch-bf16-compile 16.19 (3.07) | torch-bf16-compile-cg 16.33 (3.05) | **torch-bf16-compile-ro 15.62 (3.19)** | torch-bf16-cudnn 16.53 (3.01)<br>torch-bf16-efficient 17.02 (2.93) | fa2-bf16 16.43 (3.03) | flashinfer-bf16 16.43 (3.03) | triton-bf16 16.51 (3.02) | liger-bf16 15.81 (3.15) |
| llm_blkpre_t2048_bf16w | a100 | 57.23 | torch-bf16 5.504 (10.40) | **torch-bf16-compile 4.833 (11.84)** | torch-bf16-compile-cg 4.863 (11.77) | torch-bf16-compile-ro 4.889 (11.71) | torch-bf16-cudnn 5.437 (10.52)<br>torch-bf16-efficient 5.617 (10.19) | fa2-bf16 5.321 (10.76) | flashinfer-bf16 5.688 (10.06) | triton-bf16 5.601 (10.22) | liger-bf16 4.99 (11.47) |
| llm_blkpre_t2048_bf16w | rtx-4090 | 17.05 | torch-bf16 6.633 (2.57) | **torch-bf16-compile 5.542 (3.08)** | torch-bf16-compile-cg 5.547 (3.07) | torch-bf16-compile-ro 6.138 (2.78) | torch-bf16-cudnn 6.63 (2.57)<br>torch-bf16-efficient 6.743 (2.53) | not run | not run | not run | not run |
| llm_blkpre_t2048_bf16w | h100 | 22.07 | torch-bf16 1.694 (13.03) | torch-bf16-compile 1.355 (16.29) | **torch-bf16-compile-cg 1.308 (16.88)** | torch-bf16-compile-ro 1.325 (16.66) | torch-bf16-cudnn 1.723 (12.81)<br>torch-bf16-efficient 1.952 (11.31) | fa3-bf16 1.68 (13.14)<br>fa2-bf16 1.792 (12.31) | flashinfer-bf16 1.728 (12.78) | triton-bf16 1.761 (12.54) | liger-bf16 1.475 (14.96) |
| llm_dec_b1_l2048_bf16w | rtx-5080 | 0.1477 | torch-bf16 0.2075 (0.71) | **torch-bf16-compile 0.1455 (1.01)** | torch-bf16-compile-cg 0.1567 (0.94) | torch-bf16-compile-ro 0.2947 (0.50) | torch-bf16-cudnn 0.2502 (0.59)<br>torch-bf16-efficient 0.3872 (0.38) | fa2-bf16 0.202 (0.73) | flashinfer-bf16 0.35 (0.42) |  | liger-bf16 0.2531 (0.58) |
| llm_dec_b1_l2048_bf16w | rtx-3090 | 0.1821 | torch-bf16 0.4225 (0.43) | **torch-bf16-compile 0.223 (0.82)** | torch-bf16-compile-cg 0.2391 (0.76) | torch-bf16-compile-ro 0.4951 (0.37) | torch-bf16-cudnn 0.4602 (0.40)<br>torch-bf16-efficient 0.5925 (0.31) | fa2-bf16 0.4136 (0.44) | flashinfer-bf16 0.5623 (0.32) |  | liger-bf16 0.3979 (0.46) |
| llm_dec_b1_l2048_bf16w | a100 | 0.1637 | torch-bf16 0.3257 (0.50) | **torch-bf16-compile 0.1762 (0.93)** | torch-bf16-compile-cg 0.1917 (0.85) | torch-bf16-compile-ro 0.3527 (0.46) | torch-bf16-cudnn 0.3678 (0.45)<br>torch-bf16-efficient 0.4884 (0.34) | fa2-bf16 0.3177 (0.52) | flashinfer-bf16 0.4538 (0.36) |  | liger-bf16 0.348 (0.47) |
| llm_dec_b1_l2048_bf16w | rtx-4090 | 0.1792 | torch-bf16 0.7104 (0.25) | torch-bf16-compile 0.3602 (0.50) | **torch-bf16-compile-cg 0.3121 (0.57)** | torch-bf16-compile-ro 0.4534 (0.40) | torch-bf16-cudnn 0.7836 (0.23)<br>torch-bf16-efficient 0.8259 (0.22) | fa2-bf16 0.7193 (0.25) | flashinfer-bf16 0.968 (0.19) |  | liger-bf16 0.7048 (0.25) |
| llm_dec_b1_l2048_bf16w | h100 | 0.08862 | torch-bf16 0.3467 (0.26) | torch-bf16-compile 0.1997 (0.44) | **torch-bf16-compile-cg 0.1455 (0.61)** | torch-bf16-compile-ro 0.1596 (0.56) | torch-bf16-cudnn 0.3824 (0.23)<br>torch-bf16-efficient 0.4249 (0.21) | fa2-bf16 0.3328 (0.27)<br>fa3-bf16 0.3669 (0.24) | flashinfer-bf16 0.4864 (0.18) |  | liger-bf16 0.3815 (0.23) |
| llm_dec_b1_l8192_bf16w | rtx-5080 | 0.2152 | torch-bf16 0.2327 (0.92) | **torch-bf16-compile 0.1868 (1.15)** | torch-bf16-compile-cg 0.1972 (1.09) | torch-bf16-compile-ro 1.022 (0.21) | torch-bf16-cudnn 0.2761 (0.78)<br>torch-bf16-efficient 0.961 (0.22) | fa2-bf16 0.2261 (0.95) | flashinfer-bf16 0.416 (0.52) |  | liger-bf16 0.2783 (0.77) |
| llm_dec_b1_l8192_bf16w | rtx-3090 | 0.2509 | torch-bf16 0.4364 (0.58) | **torch-bf16-compile 0.2924 (0.86)** | torch-bf16-compile-cg 0.3036 (0.83) | torch-bf16-compile-ro 1.327 (0.19) | torch-bf16-cudnn 0.447 (0.56)<br>torch-bf16-efficient 1.147 (0.22) | fa2-bf16 0.448 (0.56) | flashinfer-bf16 0.6571 (0.38) |  | liger-bf16 0.4257 (0.59) |
| llm_dec_b1_l8192_bf16w | a100 | 0.2157 | torch-bf16 0.3478 (0.62) | **torch-bf16-compile 0.2292 (0.94)** | torch-bf16-compile-cg 0.2375 (0.91) | torch-bf16-compile-ro 0.8785 (0.25) | torch-bf16-cudnn 0.3702 (0.58)<br>torch-bf16-efficient 0.9419 (0.23) | fa2-bf16 0.3423 (0.63) | flashinfer-bf16 0.5288 (0.41) |  | liger-bf16 0.3633 (0.59) |
| llm_dec_b1_l8192_bf16w | rtx-4090 | 0.2552 | torch-bf16 0.714 (0.36) | torch-bf16-compile 0.3654 (0.70) | **torch-bf16-compile-cg 0.3467 (0.74)** | torch-bf16-compile-ro 1.251 (0.20) | torch-bf16-cudnn 0.7806 (0.33)<br>torch-bf16-efficient 1.192 (0.21) | fa2-bf16 0.7177 (0.36) | flashinfer-bf16 1.013 (0.25) |  | liger-bf16 0.7042 (0.36) |
| llm_dec_b1_l8192_bf16w | h100 | 0.1154 | torch-bf16 0.3399 (0.34) | torch-bf16-compile 0.1998 (0.58) | **torch-bf16-compile-cg 0.1472 (0.78)** | torch-bf16-compile-ro 0.1568 (0.74) | torch-bf16-cudnn 0.3969 (0.29)<br>torch-bf16-efficient 0.7583 (0.15) | fa2-bf16 0.3339 (0.35)<br>fa3-bf16 0.3567 (0.32) | flashinfer-bf16 0.5158 (0.22) |  | liger-bf16 0.38 (0.30) |
| llm_dec_b32_l2048_bf16w | rtx-5080 | 0.8071 | torch-bf16 0.6149 (1.31) | **torch-bf16-compile 0.4644 (1.74)** | torch-bf16-compile-cg 0.475 (1.70) | torch-bf16-compile-ro 7.663 (0.11) | torch-bf16-cudnn 0.5587 (1.44)<br>torch-bf16-efficient 3.863 (0.21) | fa2-bf16 0.5842 (1.38) | flashinfer-bf16 0.6725 (1.20) |  | liger-bf16 0.6989 (1.15) |
| llm_dec_b32_l2048_bf16w | rtx-3090 | 0.9582 | torch-bf16 1.062 (0.90) | torch-bf16-compile 0.9067 (1.06) | torch-bf16-compile-cg 0.9052 (1.06) | torch-bf16-compile-ro 9.047 (0.11) | **torch-bf16-cudnn 0.74 (1.29)**<br>torch-bf16-efficient 5.334 (0.18) | fa2-bf16 0.7837 (1.22) | flashinfer-bf16 0.8248 (1.16) |  | liger-bf16 1.034 (0.93) |
| llm_dec_b32_l2048_bf16w | a100 | 0.7394 | torch-bf16 0.57 (1.30) | torch-bf16-compile 0.5413 (1.37) | torch-bf16-compile-cg 0.5411 (1.37) | torch-bf16-compile-ro 5.617 (0.13) | **torch-bf16-cudnn 0.5243 (1.41)**<br>torch-bf16-efficient 3.677 (0.20) | fa2-bf16 0.532 (1.39) | flashinfer-bf16 0.6223 (1.19) |  | liger-bf16 0.586 (1.26) |
| llm_dec_b32_l2048_bf16w | rtx-4090 | 0.8987 | torch-bf16 1.004 (0.90) | **torch-bf16-compile 0.5992 (1.50)** | torch-bf16-compile-cg 0.6287 (1.43) | torch-bf16-compile-ro 8.076 (0.11) | torch-bf16-cudnn 1.048 (0.86)<br>torch-bf16-efficient 3.747 (0.24) | fa2-bf16 1.024 (0.88) | flashinfer-bf16 1.26 (0.71) |  | liger-bf16 1.036 (0.87) |
| llm_dec_b32_l2048_bf16w | h100 | 0.3662 | torch-bf16 0.3925 (0.93) | torch-bf16-compile 0.2607 (1.40) | **torch-bf16-compile-cg 0.2266 (1.62)** | torch-bf16-compile-ro 0.2274 (1.61) | torch-bf16-cudnn 0.4286 (0.85)<br>torch-bf16-efficient 2.208 (0.17) | fa2-bf16 0.3954 (0.93)<br>fa3-bf16 0.4371 (0.84) | flashinfer-bf16 0.5202 (0.70) |  | liger-bf16 0.4352 (0.84) |
| llm_dec_b32_l8192_bf16w | rtx-5080 | 2.821 | torch-bf16 1.883 (1.50) | **torch-bf16-compile 1.489 (1.89)** | torch-bf16-compile-cg 1.507 (1.87) | torch-bf16-compile-ro 29.82 (0.09) | torch-bf16-cudnn 1.501 (1.88)<br>torch-bf16-efficient 16.65 (0.17) | fa2-bf16 1.668 (1.69) | flashinfer-bf16 1.576 (1.79) |  | liger-bf16 1.974 (1.43) |
| llm_dec_b32_l8192_bf16w | rtx-3090 | 3.135 | torch-bf16 3.211 (0.98) | torch-bf16-compile 3.495 (0.90) | torch-bf16-compile-cg 3.485 (0.90) | torch-bf16-compile-ro 36.94 (0.08) | **torch-bf16-cudnn 1.683 (1.86)**<br>torch-bf16-efficient 21.02 (0.15) | fa2-bf16 1.941 (1.62) | flashinfer-bf16 1.745 (1.80) |  | liger-bf16 3.149 (1.00) |
| llm_dec_b32_l8192_bf16w | a100 | 2.058 | torch-bf16 1.499 (1.37) | torch-bf16-compile 1.925 (1.07) | torch-bf16-compile-cg 1.94 (1.06) | torch-bf16-compile-ro 24.29 (0.08) | **torch-bf16-cudnn 1.082 (1.90)**<br>torch-bf16-efficient 17.59 (0.12) | fa2-bf16 1.177 (1.75) | flashinfer-bf16 1.193 (1.73) |  | liger-bf16 1.333 (1.54) |
| llm_dec_b32_l8192_bf16w | rtx-4090 | 3.118 | torch-bf16 1.982 (1.57) | **torch-bf16-compile 1.629 (1.91)** | torch-bf16-compile-cg 1.669 (1.87) | torch-bf16-compile-ro 31.1 (0.10) | torch-bf16-cudnn 2.033 (1.53)<br>torch-bf16-efficient 12.68 (0.25) | fa2-bf16 1.986 (1.57) | flashinfer-bf16 2.231 (1.40) |  | liger-bf16 2.015 (1.55) |
| llm_dec_b32_l8192_bf16w | h100 | 1.028 | torch-bf16 0.6608 (1.56) | torch-bf16-compile 0.512 (2.01) | **torch-bf16-compile-cg 0.4866 (2.11)** | torch-bf16-compile-ro 0.4882 (2.10) | torch-bf16-cudnn 0.6801 (1.51)<br>torch-bf16-efficient 7.899 (0.13) | fa2-bf16 0.6512 (1.58)<br>fa3-bf16 0.6943 (1.48) | flashinfer-bf16 0.7902 (1.30) |  | liger-bf16 0.7033 (1.46) |
| llm_dec_b8_l2048_bf16w | rtx-5080 | 0.2948 | torch-bf16 0.2872 (1.03) | **torch-bf16-compile 0.222 (1.33)** | torch-bf16-compile-cg 0.2362 (1.25) | torch-bf16-compile-ro 1.958 (0.15) | torch-bf16-cudnn 0.2889 (1.02)<br>torch-bf16-efficient 1.246 (0.24) | fa2-bf16 0.2892 (1.02) | flashinfer-bf16 0.4182 (0.70) |  | liger-bf16 0.3768 (0.78) |
| llm_dec_b8_l2048_bf16w | rtx-3090 | 0.5245 | torch-bf16 0.4905 (1.07) | **torch-bf16-compile 0.3388 (1.55)** | torch-bf16-compile-cg 0.356 (1.47) | torch-bf16-compile-ro 2.444 (0.21) | torch-bf16-cudnn 0.492 (1.07)<br>torch-bf16-efficient 1.628 (0.32) | fa2-bf16 0.5136 (1.02) | flashinfer-bf16 0.6094 (0.86) |  | liger-bf16 0.4984 (1.05) |
| llm_dec_b8_l2048_bf16w | a100 | 0.3114 | torch-bf16 0.362 (0.86) | torch-bf16-compile 0.2812 (1.11) | **torch-bf16-compile-cg 0.2689 (1.16)** | torch-bf16-compile-ro 1.565 (0.20) | torch-bf16-cudnn 0.3791 (0.82)<br>torch-bf16-efficient 1.208 (0.26) | fa2-bf16 0.3692 (0.84) | flashinfer-bf16 0.4734 (0.66) |  | liger-bf16 0.4105 (0.76) |
| llm_dec_b8_l2048_bf16w | rtx-4090 | 0.347 | torch-bf16 0.7576 (0.46) | **torch-bf16-compile 0.3483 (1.00)** | torch-bf16-compile-cg 0.3786 (0.92) | torch-bf16-compile-ro 2.199 (0.16) | torch-bf16-cudnn 0.8024 (0.43)<br>torch-bf16-efficient 1.474 (0.24) | fa2-bf16 0.7772 (0.45) | flashinfer-bf16 1.007 (0.34) |  | liger-bf16 0.7843 (0.44) |
| llm_dec_b8_l2048_bf16w | h100 | 0.1552 | torch-bf16 0.3381 (0.46) | torch-bf16-compile 0.2247 (0.69) | **torch-bf16-compile-cg 0.1615 (0.96)** | torch-bf16-compile-ro 0.1624 (0.96) | torch-bf16-cudnn 0.378 (0.41)<br>torch-bf16-efficient 0.793 (0.20) | fa2-bf16 0.3288 (0.47)<br>fa3-bf16 0.3499 (0.44) | flashinfer-bf16 0.4545 (0.34) |  | liger-bf16 0.3905 (0.40) |
| llm_dec_b8_l8192_bf16w | rtx-5080 | 0.7715 | torch-bf16 0.5299 (1.46) | **torch-bf16-compile 0.4946 (1.56)** | torch-bf16-compile-cg 0.5092 (1.51) | torch-bf16-compile-ro 7.778 (0.10) | torch-bf16-cudnn 0.5548 (1.39)<br>torch-bf16-efficient 4.241 (0.18) | fa2-bf16 0.5441 (1.42) | flashinfer-bf16 0.6576 (1.17) |  | liger-bf16 0.6043 (1.28) |
| llm_dec_b8_l8192_bf16w | rtx-3090 | 1.075 | torch-bf16 0.7267 (1.48) | torch-bf16-compile 0.8541 (1.26) | torch-bf16-compile-cg 0.8591 (1.25) | torch-bf16-compile-ro 9.324 (0.12) | torch-bf16-cudnn 0.7304 (1.47)<br>torch-bf16-efficient 5.682 (0.19) | fa2-bf16 0.7297 (1.47) | flashinfer-bf16 0.8244 (1.30) |  | **liger-bf16 0.7144 (1.50)** |
| llm_dec_b8_l8192_bf16w | a100 | 0.6365 | **torch-bf16 0.5052 (1.26)** | torch-bf16-compile 0.5837 (1.09) | torch-bf16-compile-cg 0.5976 (1.07) | torch-bf16-compile-ro 6.004 (0.11) | torch-bf16-cudnn 0.5265 (1.21)<br>torch-bf16-efficient 4.381 (0.15) | fa2-bf16 0.5096 (1.25) | flashinfer-bf16 0.6136 (1.04) |  | liger-bf16 0.5432 (1.17) |
| llm_dec_b8_l8192_bf16w | rtx-4090 | 0.9045 | torch-bf16 1.007 (0.90) | **torch-bf16-compile 0.6121 (1.48)** | torch-bf16-compile-cg 0.6408 (1.41) | torch-bf16-compile-ro 7.96 (0.11) | torch-bf16-cudnn 1.051 (0.86)<br>torch-bf16-efficient 3.713 (0.24) | fa2-bf16 1.026 (0.88) | flashinfer-bf16 1.251 (0.72) |  | liger-bf16 1.038 (0.87) |
| llm_dec_b8_l8192_bf16w | h100 | 0.3361 | torch-bf16 0.4078 (0.82) | torch-bf16-compile 0.2601 (1.29) | **torch-bf16-compile-cg 0.2278 (1.48)** | torch-bf16-compile-ro 0.2293 (1.47) | torch-bf16-cudnn 0.4303 (0.78)<br>torch-bf16-efficient 2.24 (0.15) | fa2-bf16 0.3912 (0.86)<br>fa3-bf16 0.4108 (0.82) | flashinfer-bf16 0.5274 (0.64) |  | liger-bf16 0.4336 (0.78) |
| llm_pre_t2048_bf16w | rtx-5080 | 6.365 | torch-bf16 2.213 (2.88) | torch-bf16-compile 1.937 (3.29) | **torch-bf16-compile-cg 1.931 (3.30)** | torch-bf16-compile-ro 1.971 (3.23) | torch-bf16-cudnn 2.164 (2.94)<br>torch-bf16-efficient 2.714 (2.35) | fa2-bf16 2.215 (2.87) | flashinfer-bf16 2.222 (2.86) | triton-bf16 2.294 (2.77) | liger-bf16 2.075 (3.07) |
| llm_pre_t2048_bf16w | rtx-3090 | 10.85 | torch-bf16 4.242 (2.56) | torch-bf16-compile 4.146 (2.62) | torch-bf16-compile-cg 3.87 (2.80) | **torch-bf16-compile-ro 3.742 (2.90)** | torch-bf16-cudnn 4.251 (2.55)<br>torch-bf16-efficient 4.598 (2.36) | fa2-bf16 4.195 (2.59) | flashinfer-bf16 4.231 (2.56) | triton-bf16 4.353 (2.49) | liger-bf16 3.954 (2.74) |
| llm_pre_t2048_bf16w | a100 | 16.27 | torch-bf16 2.522 (6.45) | torch-bf16-compile 1.227 (13.26) | **torch-bf16-compile-cg 1.082 (15.03)** | torch-bf16-compile-ro 1.121 (14.51) | torch-bf16-cudnn 1.625 (10.01)<br>torch-bf16-efficient 1.857 (8.76) | fa2-bf16 1.56 (10.43) | flashinfer-bf16 1.502 (10.83) | triton-bf16 1.702 (9.56) | liger-bf16 1.349 (12.06) |
| llm_pre_t2048_bf16w | rtx-4090 | 4.6 | torch-bf16 1.717 (2.68) | torch-bf16-compile 1.322 (3.48) | **torch-bf16-compile-cg 1.319 (3.49)** | torch-bf16-compile-ro 1.483 (3.10) | torch-bf16-cudnn 1.738 (2.65)<br>torch-bf16-efficient 1.875 (2.45) | not run | not run | not run | not run |
| llm_pre_t2048_bf16w | h100 | 6.13 | torch-bf16 0.5705 (10.75) | torch-bf16-compile 0.3576 (17.14) | torch-bf16-compile-cg 0.3509 (17.47) | **torch-bf16-compile-ro 0.3497 (17.53)** | torch-bf16-cudnn 0.574 (10.68)<br>torch-bf16-efficient 0.8057 (7.61) | fa3-bf16 0.5498 (11.15)<br>fa2-bf16 0.6388 (9.60) | flashinfer-bf16 0.5787 (10.59) | triton-bf16 0.6412 (9.56) | liger-bf16 0.4843 (12.66) |
| llm_blkdec_b1_l4096_bf16kv | rtx-5080 | 0.6116 | torch-bf16 0.6486 (0.94) | **torch-bf16-compile 0.5569 (1.10)** | torch-bf16-compile-cg 0.5591 (1.09) | torch-bf16-compile-ro 0.9885 (0.62) | torch-bf16-cudnn 0.6793 (0.90)<br>torch-bf16-efficient 0.9947 (0.61) | fa2-bf16 0.6803 (0.90) | flashinfer-bf16 0.822 (0.74) |  | liger-bf16 0.7106 (0.86) |
| llm_blkdec_b1_l4096_bf16kv | rtx-3090 | 0.6876 | torch-bf16 0.934 (0.74) | **torch-bf16-compile 0.6743 (1.02)** | torch-bf16-compile-cg 0.6883 (1.00) | torch-bf16-compile-ro 1.257 (0.55) | torch-bf16-cudnn 0.9506 (0.72)<br>torch-bf16-efficient 1.258 (0.55) | fa2-bf16 0.9223 (0.75) | flashinfer-bf16 1.057 (0.65) |  | liger-bf16 0.9117 (0.75) |
| llm_blkdec_b1_l4096_bf16kv | a100 | 0.7672 | torch-bf16 0.6659 (1.15) | torch-bf16-compile 0.5082 (1.51) | **torch-bf16-compile-cg 0.4794 (1.60)** | torch-bf16-compile-ro 0.8159 (0.94) | torch-bf16-cudnn 0.7171 (1.07)<br>torch-bf16-efficient 0.9582 (0.80) | fa2-bf16 0.6591 (1.16) | flashinfer-bf16 0.7987 (0.96) |  | liger-bf16 0.717 (1.07) |
| llm_blkdec_b1_l4096_bf16kv | rtx-4090 | 0.6906 | torch-bf16 1.317 (0.52) | torch-bf16-compile 0.7731 (0.89) | **torch-bf16-compile-cg 0.751 (0.92)** | torch-bf16-compile-ro 1.182 (0.58) | torch-bf16-cudnn 1.387 (0.50)<br>torch-bf16-efficient 1.478 (0.47) | not run | not run |  | not run |
| llm_blkdec_b1_l4096_bf16kv | h100 | 0.4905 | torch-bf16 0.542 (0.91) | torch-bf16-compile 0.3395 (1.44) | **torch-bf16-compile-cg 0.286 (1.72)** | torch-bf16-compile-ro 0.2879 (1.70) | torch-bf16-cudnn 0.5835 (0.84)<br>torch-bf16-efficient 0.7069 (0.69) | fa2-bf16 0.5341 (0.92)<br>fa3-bf16 0.5788 (0.85) | flashinfer-bf16 0.6731 (0.73) |  | liger-bf16 0.6021 (0.81) |
| llm_dec_b1_l2048_bf16kv | rtx-5080 | 0.1392 | torch-bf16 0.2075 (0.67) | **torch-bf16-compile 0.1455 (0.96)** | torch-bf16-compile-cg 0.1567 (0.89) | torch-bf16-compile-ro 0.2947 (0.47) | torch-bf16-cudnn 0.2502 (0.56)<br>torch-bf16-efficient 0.3872 (0.36) | fa2-bf16 0.202 (0.69) | flashinfer-bf16 0.35 (0.40) |  | liger-bf16 0.2531 (0.55) |
| llm_dec_b1_l2048_bf16kv | rtx-3090 | 0.1732 | torch-bf16 0.4225 (0.41) | **torch-bf16-compile 0.223 (0.78)** | torch-bf16-compile-cg 0.2391 (0.72) | torch-bf16-compile-ro 0.4951 (0.35) | torch-bf16-cudnn 0.4602 (0.38)<br>torch-bf16-efficient 0.5925 (0.29) | fa2-bf16 0.4136 (0.42) | flashinfer-bf16 0.5623 (0.31) |  | liger-bf16 0.3979 (0.44) |
| llm_dec_b1_l2048_bf16kv | a100 | 0.1593 | torch-bf16 0.3257 (0.49) | **torch-bf16-compile 0.1762 (0.90)** | torch-bf16-compile-cg 0.1917 (0.83) | torch-bf16-compile-ro 0.3527 (0.45) | torch-bf16-cudnn 0.3678 (0.43)<br>torch-bf16-efficient 0.4884 (0.33) | fa2-bf16 0.3177 (0.50) | flashinfer-bf16 0.4538 (0.35) |  | liger-bf16 0.348 (0.46) |
| llm_dec_b1_l2048_bf16kv | rtx-4090 | 0.1682 | torch-bf16 0.7272 (0.23) | torch-bf16-compile 0.3631 (0.46) | **torch-bf16-compile-cg 0.3065 (0.55)** | torch-bf16-compile-ro 0.4537 (0.37) | torch-bf16-cudnn 0.7959 (0.21)<br>torch-bf16-efficient 0.8383 (0.20) | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16kv | h100 | 0.08693 | torch-bf16 0.3467 (0.25) | torch-bf16-compile 0.1997 (0.44) | **torch-bf16-compile-cg 0.1455 (0.60)** | torch-bf16-compile-ro 0.1596 (0.54) | torch-bf16-cudnn 0.3824 (0.23)<br>torch-bf16-efficient 0.4249 (0.20) | fa2-bf16 0.3328 (0.26)<br>fa3-bf16 0.3669 (0.24) | flashinfer-bf16 0.4864 (0.18) |  | liger-bf16 0.3815 (0.23) |
| llm_dec_b1_l8192_bf16kv | rtx-5080 | 0.1792 | torch-bf16 0.2327 (0.77) | **torch-bf16-compile 0.1868 (0.96)** | torch-bf16-compile-cg 0.1972 (0.91) | torch-bf16-compile-ro 1.022 (0.18) | torch-bf16-cudnn 0.2761 (0.65)<br>torch-bf16-efficient 0.961 (0.19) | fa2-bf16 0.2261 (0.79) | flashinfer-bf16 0.416 (0.43) |  | liger-bf16 0.2783 (0.64) |
| llm_dec_b1_l8192_bf16kv | rtx-3090 | 0.2202 | torch-bf16 0.4364 (0.50) | **torch-bf16-compile 0.2924 (0.75)** | torch-bf16-compile-cg 0.3036 (0.73) | torch-bf16-compile-ro 1.327 (0.17) | torch-bf16-cudnn 0.447 (0.49)<br>torch-bf16-efficient 1.147 (0.19) | fa2-bf16 0.448 (0.49) | flashinfer-bf16 0.6571 (0.34) |  | liger-bf16 0.4257 (0.52) |
| llm_dec_b1_l8192_bf16kv | a100 | 0.2016 | torch-bf16 0.3478 (0.58) | **torch-bf16-compile 0.2292 (0.88)** | torch-bf16-compile-cg 0.2375 (0.85) | torch-bf16-compile-ro 0.8785 (0.23) | torch-bf16-cudnn 0.3702 (0.54)<br>torch-bf16-efficient 0.9419 (0.21) | fa2-bf16 0.3423 (0.59) | flashinfer-bf16 0.5288 (0.38) |  | liger-bf16 0.3633 (0.56) |
| llm_dec_b1_l8192_bf16kv | rtx-4090 | 0.2105 | torch-bf16 0.7293 (0.29) | torch-bf16-compile 0.3805 (0.55) | **torch-bf16-compile-cg 0.3505 (0.60)** | torch-bf16-compile-ro 1.253 (0.17) | torch-bf16-cudnn 0.7974 (0.26)<br>torch-bf16-efficient 1.205 (0.17) | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16kv | h100 | 0.1137 | torch-bf16 0.3399 (0.33) | torch-bf16-compile 0.1998 (0.57) | **torch-bf16-compile-cg 0.1472 (0.77)** | torch-bf16-compile-ro 0.1568 (0.72) | torch-bf16-cudnn 0.3969 (0.29)<br>torch-bf16-efficient 0.7583 (0.15) | fa2-bf16 0.3339 (0.34)<br>fa3-bf16 0.3567 (0.32) | flashinfer-bf16 0.5158 (0.22) |  | liger-bf16 0.38 (0.30) |
| llm_dec_b32_l2048_bf16kv | rtx-5080 | 0.5105 | torch-bf16 0.6149 (0.83) | **torch-bf16-compile 0.4644 (1.10)** | torch-bf16-compile-cg 0.475 (1.07) | torch-bf16-compile-ro 7.663 (0.07) | torch-bf16-cudnn 0.5587 (0.91)<br>torch-bf16-efficient 3.863 (0.13) | fa2-bf16 0.5842 (0.87) | flashinfer-bf16 0.6725 (0.76) |  | liger-bf16 0.6989 (0.73) |
| llm_dec_b32_l2048_bf16kv | rtx-3090 | 0.6754 | torch-bf16 1.062 (0.64) | torch-bf16-compile 0.9067 (0.74) | torch-bf16-compile-cg 0.9052 (0.75) | torch-bf16-compile-ro 9.047 (0.07) | **torch-bf16-cudnn 0.74 (0.91)**<br>torch-bf16-efficient 5.334 (0.13) | fa2-bf16 0.7837 (0.86) | flashinfer-bf16 0.8248 (0.82) |  | liger-bf16 1.034 (0.65) |
| llm_dec_b32_l2048_bf16kv | a100 | 0.6216 | torch-bf16 0.57 (1.09) | torch-bf16-compile 0.5413 (1.15) | torch-bf16-compile-cg 0.5411 (1.15) | torch-bf16-compile-ro 5.617 (0.11) | **torch-bf16-cudnn 0.5243 (1.19)**<br>torch-bf16-efficient 3.677 (0.17) | fa2-bf16 0.532 (1.17) | flashinfer-bf16 0.6223 (1.00) |  | liger-bf16 0.586 (1.06) |
| llm_dec_b32_l2048_bf16kv | rtx-4090 | 0.5943 | torch-bf16 1.024 (0.58) | **torch-bf16-compile 0.5975 (0.99)** | torch-bf16-compile-cg 0.6772 (0.88) | torch-bf16-compile-ro 8.08 (0.07) | torch-bf16-cudnn 1.062 (0.56)<br>torch-bf16-efficient 3.759 (0.16) | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16kv | h100 | 0.3177 | torch-bf16 0.3925 (0.81) | torch-bf16-compile 0.2607 (1.22) | **torch-bf16-compile-cg 0.2266 (1.40)** | torch-bf16-compile-ro 0.2274 (1.40) | torch-bf16-cudnn 0.4286 (0.74)<br>torch-bf16-efficient 2.208 (0.14) | fa2-bf16 0.3954 (0.80)<br>fa3-bf16 0.4371 (0.73) | flashinfer-bf16 0.5202 (0.61) |  | liger-bf16 0.4352 (0.73) |
| llm_dec_b32_l8192_bf16kv | rtx-5080 | 1.728 | torch-bf16 1.883 (0.92) | **torch-bf16-compile 1.489 (1.16)** | torch-bf16-compile-cg 1.507 (1.15) | torch-bf16-compile-ro 29.82 (0.06) | torch-bf16-cudnn 1.501 (1.15)<br>torch-bf16-efficient 16.65 (0.10) | fa2-bf16 1.668 (1.04) | flashinfer-bf16 1.576 (1.10) |  | liger-bf16 1.974 (0.88) |
| llm_dec_b32_l8192_bf16kv | rtx-3090 | 1.993 | torch-bf16 3.211 (0.62) | torch-bf16-compile 3.495 (0.57) | torch-bf16-compile-cg 3.485 (0.57) | torch-bf16-compile-ro 36.94 (0.05) | **torch-bf16-cudnn 1.683 (1.18)**<br>torch-bf16-efficient 21.02 (0.09) | fa2-bf16 1.941 (1.03) | flashinfer-bf16 1.745 (1.14) |  | liger-bf16 3.149 (0.63) |
| llm_dec_b32_l8192_bf16kv | a100 | 1.566 | torch-bf16 1.499 (1.04) | torch-bf16-compile 1.925 (0.81) | torch-bf16-compile-cg 1.94 (0.81) | torch-bf16-compile-ro 24.29 (0.06) | **torch-bf16-cudnn 1.082 (1.45)**<br>torch-bf16-efficient 17.59 (0.09) | fa2-bf16 1.177 (1.33) | flashinfer-bf16 1.193 (1.31) |  | liger-bf16 1.333 (1.18) |
| llm_dec_b32_l8192_bf16kv | rtx-4090 | 1.922 | torch-bf16 1.99 (0.97) | **torch-bf16-compile 1.638 (1.17)** | torch-bf16-compile-cg 1.673 (1.15) | torch-bf16-compile-ro 31.14 (0.06) | torch-bf16-cudnn 2.053 (0.94)<br>torch-bf16-efficient 12.7 (0.15) | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16kv | h100 | 0.7925 | torch-bf16 0.6608 (1.20) | torch-bf16-compile 0.512 (1.55) | **torch-bf16-compile-cg 0.4866 (1.63)** | torch-bf16-compile-ro 0.4882 (1.62) | torch-bf16-cudnn 0.6801 (1.17)<br>torch-bf16-efficient 7.899 (0.10) | fa2-bf16 0.6512 (1.22)<br>fa3-bf16 0.6943 (1.14) | flashinfer-bf16 0.7902 (1.00) |  | liger-bf16 0.7033 (1.13) |
| llm_dec_b8_l2048_bf16kv | rtx-5080 | 0.2241 | torch-bf16 0.2872 (0.78) | **torch-bf16-compile 0.222 (1.01)** | torch-bf16-compile-cg 0.2362 (0.95) | torch-bf16-compile-ro 1.958 (0.11) | torch-bf16-cudnn 0.2889 (0.78)<br>torch-bf16-efficient 1.246 (0.18) | fa2-bf16 0.2892 (0.77) | flashinfer-bf16 0.4182 (0.54) |  | liger-bf16 0.3768 (0.59) |
| llm_dec_b8_l2048_bf16kv | rtx-3090 | 0.4704 | torch-bf16 0.4905 (0.96) | **torch-bf16-compile 0.3388 (1.39)** | torch-bf16-compile-cg 0.356 (1.32) | torch-bf16-compile-ro 2.444 (0.19) | torch-bf16-cudnn 0.492 (0.96)<br>torch-bf16-efficient 1.628 (0.29) | fa2-bf16 0.5136 (0.92) | flashinfer-bf16 0.6094 (0.77) |  | liger-bf16 0.4984 (0.94) |
| llm_dec_b8_l2048_bf16kv | a100 | 0.2717 | torch-bf16 0.362 (0.75) | torch-bf16-compile 0.2812 (0.97) | **torch-bf16-compile-cg 0.2689 (1.01)** | torch-bf16-compile-ro 1.565 (0.17) | torch-bf16-cudnn 0.3791 (0.72)<br>torch-bf16-efficient 1.208 (0.22) | fa2-bf16 0.3692 (0.74) | flashinfer-bf16 0.4734 (0.57) |  | liger-bf16 0.4105 (0.66) |
| llm_dec_b8_l2048_bf16kv | rtx-4090 | 0.2706 | torch-bf16 0.7625 (0.35) | **torch-bf16-compile 0.353 (0.77)** | torch-bf16-compile-cg 0.3859 (0.70) | torch-bf16-compile-ro 2.191 (0.12) | torch-bf16-cudnn 0.8156 (0.33)<br>torch-bf16-efficient 1.491 (0.18) | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16kv | h100 | 0.1434 | torch-bf16 0.3381 (0.42) | torch-bf16-compile 0.2247 (0.64) | **torch-bf16-compile-cg 0.1615 (0.89)** | torch-bf16-compile-ro 0.1624 (0.88) | torch-bf16-cudnn 0.378 (0.38)<br>torch-bf16-efficient 0.793 (0.18) | fa2-bf16 0.3288 (0.44)<br>fa3-bf16 0.3499 (0.41) | flashinfer-bf16 0.4545 (0.32) |  | liger-bf16 0.3905 (0.37) |
| llm_dec_b8_l8192_bf16kv | rtx-5080 | 0.5126 | torch-bf16 0.5299 (0.97) | **torch-bf16-compile 0.4946 (1.04)** | torch-bf16-compile-cg 0.5092 (1.01) | torch-bf16-compile-ro 7.778 (0.07) | torch-bf16-cudnn 0.5548 (0.92)<br>torch-bf16-efficient 4.241 (0.12) | fa2-bf16 0.5441 (0.94) | flashinfer-bf16 0.6576 (0.78) |  | liger-bf16 0.6043 (0.85) |
| llm_dec_b8_l8192_bf16kv | rtx-3090 | 0.7996 | torch-bf16 0.7267 (1.10) | torch-bf16-compile 0.8541 (0.94) | torch-bf16-compile-cg 0.8591 (0.93) | torch-bf16-compile-ro 9.324 (0.09) | torch-bf16-cudnn 0.7304 (1.09)<br>torch-bf16-efficient 5.682 (0.14) | fa2-bf16 0.7297 (1.10) | flashinfer-bf16 0.8244 (0.97) |  | **liger-bf16 0.7144 (1.12)** |
| llm_dec_b8_l8192_bf16kv | a100 | 0.52 | **torch-bf16 0.5052 (1.03)** | torch-bf16-compile 0.5837 (0.89) | torch-bf16-compile-cg 0.5976 (0.87) | torch-bf16-compile-ro 6.004 (0.09) | torch-bf16-cudnn 0.5265 (0.99)<br>torch-bf16-efficient 4.381 (0.12) | fa2-bf16 0.5096 (1.02) | flashinfer-bf16 0.6136 (0.85) |  | liger-bf16 0.5432 (0.96) |
| llm_dec_b8_l8192_bf16kv | rtx-4090 | 0.5934 | torch-bf16 1.022 (0.58) | **torch-bf16-compile 0.6136 (0.97)** | torch-bf16-compile-cg 0.6425 (0.92) | torch-bf16-compile-ro 7.968 (0.07) | torch-bf16-cudnn 1.065 (0.56)<br>torch-bf16-efficient 3.727 (0.16) | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16kv | h100 | 0.2817 | torch-bf16 0.4078 (0.69) | torch-bf16-compile 0.2601 (1.08) | **torch-bf16-compile-cg 0.2278 (1.24)** | torch-bf16-compile-ro 0.2293 (1.23) | torch-bf16-cudnn 0.4303 (0.65)<br>torch-bf16-efficient 2.24 (0.13) | fa2-bf16 0.3912 (0.72)<br>fa3-bf16 0.4108 (0.69) | flashinfer-bf16 0.5274 (0.53) |  | liger-bf16 0.4336 (0.65) |
| llm_blkdec_b1_l4096_bf16 | rtx-5080 | 0.5738 | torch-bf16 0.6486 (0.88) | **torch-bf16-compile 0.5569 (1.03)** | torch-bf16-compile-cg 0.5591 (1.03) | torch-bf16-compile-ro 0.9885 (0.58) | torch-bf16-cudnn 0.6793 (0.84)<br>torch-bf16-efficient 0.9947 (0.58) | fa2-bf16 0.6803 (0.84) | flashinfer-bf16 0.822 (0.70) |  | liger-bf16 0.7106 (0.81) |
| llm_blkdec_b1_l4096_bf16 | rtx-3090 | 0.6397 | torch-bf16 0.934 (0.68) | **torch-bf16-compile 0.6743 (0.95)** | torch-bf16-compile-cg 0.6883 (0.93) | torch-bf16-compile-ro 1.257 (0.51) | torch-bf16-cudnn 0.9506 (0.67)<br>torch-bf16-efficient 1.258 (0.51) | fa2-bf16 0.9223 (0.69) | flashinfer-bf16 1.057 (0.61) |  | liger-bf16 0.9117 (0.70) |
| llm_blkdec_b1_l4096_bf16 | a100 | 0.5308 | torch-bf16 0.6659 (0.80) | torch-bf16-compile 0.5082 (1.04) | **torch-bf16-compile-cg 0.4794 (1.11)** | torch-bf16-compile-ro 0.8159 (0.65) | torch-bf16-cudnn 0.7171 (0.74)<br>torch-bf16-efficient 0.9582 (0.55) | fa2-bf16 0.6591 (0.81) | flashinfer-bf16 0.7987 (0.66) |  | liger-bf16 0.717 (0.74) |
| llm_blkdec_b1_l4096_bf16 | rtx-4090 | 0.6554 | torch-bf16 1.318 (0.50) | torch-bf16-compile 0.7761 (0.84) | **torch-bf16-compile-cg 0.7508 (0.87)** | torch-bf16-compile-ro 1.175 (0.56) | torch-bf16-cudnn 1.385 (0.47)<br>torch-bf16-efficient 1.471 (0.45) | not run | not run |  | not run |
| llm_blkdec_b1_l4096_bf16 | h100 | 0.2649 | torch-bf16 0.542 (0.49) | torch-bf16-compile 0.3395 (0.78) | **torch-bf16-compile-cg 0.286 (0.93)** | torch-bf16-compile-ro 0.2879 (0.92) | torch-bf16-cudnn 0.5835 (0.45)<br>torch-bf16-efficient 0.7069 (0.37) | fa2-bf16 0.5341 (0.50)<br>fa3-bf16 0.5788 (0.46) | flashinfer-bf16 0.6731 (0.39) |  | liger-bf16 0.6021 (0.44) |
| llm_blkpre_t2048_bf16 | rtx-5080 | 9.586 | torch-bf16 10.22 (0.94) | **torch-bf16-compile 8.103 (1.18)** | torch-bf16-compile-cg 9.426 (1.02) | torch-bf16-compile-ro 9.779 (0.98) | torch-bf16-cudnn 8.82 (1.09)<br>torch-bf16-efficient 10.83 (0.89) | fa2-bf16 8.878 (1.08) | flashinfer-bf16 8.881 (1.08) | triton-bf16 8.92 (1.07) | liger-bf16 8.524 (1.12) |
| llm_blkpre_t2048_bf16 | rtx-3090 | 16.18 | torch-bf16 16.56 (0.98) | torch-bf16-compile 16.19 (1.00) | torch-bf16-compile-cg 16.33 (0.99) | **torch-bf16-compile-ro 15.62 (1.04)** | torch-bf16-cudnn 16.53 (0.98)<br>torch-bf16-efficient 17.02 (0.95) | fa2-bf16 16.43 (0.98) | flashinfer-bf16 16.43 (0.98) | triton-bf16 16.51 (0.98) | liger-bf16 15.81 (1.02) |
| llm_blkpre_t2048_bf16 | a100 | 7.109 | torch-bf16 5.504 (1.29) | **torch-bf16-compile 4.833 (1.47)** | torch-bf16-compile-cg 4.863 (1.46) | torch-bf16-compile-ro 4.889 (1.45) | torch-bf16-cudnn 5.437 (1.31)<br>torch-bf16-efficient 5.617 (1.27) | fa2-bf16 5.321 (1.34) | flashinfer-bf16 5.688 (1.25) | triton-bf16 5.601 (1.27) | liger-bf16 4.99 (1.42) |
| llm_blkpre_t2048_bf16 | rtx-4090 | 7.616 | torch-bf16 6.617 (1.15) | torch-bf16-compile 5.532 (1.38) | **torch-bf16-compile-cg 5.527 (1.38)** | torch-bf16-compile-ro 6.165 (1.24) | torch-bf16-cudnn 6.606 (1.15)<br>torch-bf16-efficient 6.75 (1.13) | not run | not run | not run | not run |
| llm_blkpre_t2048_bf16 | h100 | 3.828 | torch-bf16 1.694 (2.26) | torch-bf16-compile 1.355 (2.82) | **torch-bf16-compile-cg 1.308 (2.93)** | torch-bf16-compile-ro 1.325 (2.89) | torch-bf16-cudnn 1.723 (2.22)<br>torch-bf16-efficient 1.952 (1.96) | fa3-bf16 1.68 (2.28)<br>fa2-bf16 1.792 (2.14) | flashinfer-bf16 1.728 (2.22) | triton-bf16 1.761 (2.17) | liger-bf16 1.475 (2.59) |
| llm_dec_b1_l2048_bf16 | rtx-5080 | 0.1385 | torch-bf16 0.2075 (0.67) | **torch-bf16-compile 0.1455 (0.95)** | torch-bf16-compile-cg 0.1567 (0.88) | torch-bf16-compile-ro 0.2947 (0.47) | torch-bf16-cudnn 0.2502 (0.55)<br>torch-bf16-efficient 0.3872 (0.36) | fa2-bf16 0.202 (0.69) | flashinfer-bf16 0.35 (0.40) |  | liger-bf16 0.2531 (0.55) |
| llm_dec_b1_l2048_bf16 | rtx-3090 | 0.177 | torch-bf16 0.4225 (0.42) | **torch-bf16-compile 0.223 (0.79)** | torch-bf16-compile-cg 0.2391 (0.74) | torch-bf16-compile-ro 0.4951 (0.36) | torch-bf16-cudnn 0.4602 (0.38)<br>torch-bf16-efficient 0.5925 (0.30) | fa2-bf16 0.4136 (0.43) | flashinfer-bf16 0.5623 (0.31) |  | liger-bf16 0.3979 (0.44) |
| llm_dec_b1_l2048_bf16 | a100 | 0.1582 | torch-bf16 0.3257 (0.49) | **torch-bf16-compile 0.1762 (0.90)** | torch-bf16-compile-cg 0.1917 (0.83) | torch-bf16-compile-ro 0.3527 (0.45) | torch-bf16-cudnn 0.3678 (0.43)<br>torch-bf16-efficient 0.4884 (0.32) | fa2-bf16 0.3177 (0.50) | flashinfer-bf16 0.4538 (0.35) |  | liger-bf16 0.348 (0.45) |
| llm_dec_b1_l2048_bf16 | rtx-4090 | 0.1688 | torch-bf16 0.7293 (0.23) | torch-bf16-compile 0.3642 (0.46) | **torch-bf16-compile-cg 0.3082 (0.55)** | torch-bf16-compile-ro 0.4539 (0.37) | torch-bf16-cudnn 0.7944 (0.21)<br>torch-bf16-efficient 0.8357 (0.20) | not run | not run |  | not run |
| llm_dec_b1_l2048_bf16 | h100 | 0.08364 | torch-bf16 0.3467 (0.24) | torch-bf16-compile 0.1997 (0.42) | **torch-bf16-compile-cg 0.1455 (0.57)** | torch-bf16-compile-ro 0.1596 (0.52) | torch-bf16-cudnn 0.3824 (0.22)<br>torch-bf16-efficient 0.4249 (0.20) | fa2-bf16 0.3328 (0.25)<br>fa3-bf16 0.3669 (0.23) | flashinfer-bf16 0.4864 (0.17) |  | liger-bf16 0.3815 (0.22) |
| llm_dec_b1_l8192_bf16 | rtx-5080 | 0.1768 | torch-bf16 0.2327 (0.76) | **torch-bf16-compile 0.1868 (0.95)** | torch-bf16-compile-cg 0.1972 (0.90) | torch-bf16-compile-ro 1.022 (0.17) | torch-bf16-cudnn 0.2761 (0.64)<br>torch-bf16-efficient 0.961 (0.18) | fa2-bf16 0.2261 (0.78) | flashinfer-bf16 0.416 (0.43) |  | liger-bf16 0.2783 (0.64) |
| llm_dec_b1_l8192_bf16 | rtx-3090 | 0.2233 | torch-bf16 0.4364 (0.51) | **torch-bf16-compile 0.2924 (0.76)** | torch-bf16-compile-cg 0.3036 (0.74) | torch-bf16-compile-ro 1.327 (0.17) | torch-bf16-cudnn 0.447 (0.50)<br>torch-bf16-efficient 1.147 (0.19) | fa2-bf16 0.448 (0.50) | flashinfer-bf16 0.6571 (0.34) |  | liger-bf16 0.4257 (0.52) |
| llm_dec_b1_l8192_bf16 | a100 | 0.1956 | torch-bf16 0.3478 (0.56) | **torch-bf16-compile 0.2292 (0.85)** | torch-bf16-compile-cg 0.2375 (0.82) | torch-bf16-compile-ro 0.8785 (0.22) | torch-bf16-cudnn 0.3702 (0.53)<br>torch-bf16-efficient 0.9419 (0.21) | fa2-bf16 0.3423 (0.57) | flashinfer-bf16 0.5288 (0.37) |  | liger-bf16 0.3633 (0.54) |
| llm_dec_b1_l8192_bf16 | rtx-4090 | 0.2113 | torch-bf16 0.7318 (0.29) | torch-bf16-compile 0.3581 (0.59) | **torch-bf16-compile-cg 0.3507 (0.60)** | torch-bf16-compile-ro 1.247 (0.17) | torch-bf16-cudnn 0.7973 (0.27)<br>torch-bf16-efficient 1.205 (0.18) | not run | not run |  | not run |
| llm_dec_b1_l8192_bf16 | h100 | 0.1092 | torch-bf16 0.3399 (0.32) | torch-bf16-compile 0.1998 (0.55) | **torch-bf16-compile-cg 0.1472 (0.74)** | torch-bf16-compile-ro 0.1568 (0.70) | torch-bf16-cudnn 0.3969 (0.28)<br>torch-bf16-efficient 0.7583 (0.14) | fa2-bf16 0.3339 (0.33)<br>fa3-bf16 0.3567 (0.31) | flashinfer-bf16 0.5158 (0.21) |  | liger-bf16 0.38 (0.29) |
| llm_dec_b32_l2048_bf16 | rtx-5080 | 0.5048 | torch-bf16 0.6149 (0.82) | **torch-bf16-compile 0.4644 (1.09)** | torch-bf16-compile-cg 0.475 (1.06) | torch-bf16-compile-ro 7.663 (0.07) | torch-bf16-cudnn 0.5587 (0.90)<br>torch-bf16-efficient 3.863 (0.13) | fa2-bf16 0.5842 (0.86) | flashinfer-bf16 0.6725 (0.75) |  | liger-bf16 0.6989 (0.72) |
| llm_dec_b32_l2048_bf16 | rtx-3090 | 0.6751 | torch-bf16 1.062 (0.64) | torch-bf16-compile 0.9067 (0.74) | torch-bf16-compile-cg 0.9052 (0.75) | torch-bf16-compile-ro 9.047 (0.07) | **torch-bf16-cudnn 0.74 (0.91)**<br>torch-bf16-efficient 5.334 (0.13) | fa2-bf16 0.7837 (0.86) | flashinfer-bf16 0.8248 (0.82) |  | liger-bf16 1.034 (0.65) |
| llm_dec_b32_l2048_bf16 | a100 | 0.4909 | torch-bf16 0.57 (0.86) | torch-bf16-compile 0.5413 (0.91) | torch-bf16-compile-cg 0.5411 (0.91) | torch-bf16-compile-ro 5.617 (0.09) | **torch-bf16-cudnn 0.5243 (0.94)**<br>torch-bf16-efficient 3.677 (0.13) | fa2-bf16 0.532 (0.92) | flashinfer-bf16 0.6223 (0.79) |  | liger-bf16 0.586 (0.84) |
| llm_dec_b32_l2048_bf16 | rtx-4090 | 0.5931 | torch-bf16 1.022 (0.58) | **torch-bf16-compile 0.5996 (0.99)** | torch-bf16-compile-cg 0.6297 (0.94) | torch-bf16-compile-ro 8.07 (0.07) | torch-bf16-cudnn 1.065 (0.56)<br>torch-bf16-efficient 3.758 (0.16) | not run | not run |  | not run |
| llm_dec_b32_l2048_bf16 | h100 | 0.2652 | torch-bf16 0.3925 (0.68) | torch-bf16-compile 0.2607 (1.02) | **torch-bf16-compile-cg 0.2266 (1.17)** | torch-bf16-compile-ro 0.2274 (1.17) | torch-bf16-cudnn 0.4286 (0.62)<br>torch-bf16-efficient 2.208 (0.12) | fa2-bf16 0.3954 (0.67)<br>fa3-bf16 0.4371 (0.61) | flashinfer-bf16 0.5202 (0.51) |  | liger-bf16 0.4352 (0.61) |
| llm_dec_b32_l8192_bf16 | rtx-5080 | 1.746 | torch-bf16 1.883 (0.93) | **torch-bf16-compile 1.489 (1.17)** | torch-bf16-compile-cg 1.507 (1.16) | torch-bf16-compile-ro 29.82 (0.06) | torch-bf16-cudnn 1.501 (1.16)<br>torch-bf16-efficient 16.65 (0.10) | fa2-bf16 1.668 (1.05) | flashinfer-bf16 1.576 (1.11) |  | liger-bf16 1.974 (0.88) |
| llm_dec_b32_l8192_bf16 | rtx-3090 | 1.997 | torch-bf16 3.211 (0.62) | torch-bf16-compile 3.495 (0.57) | torch-bf16-compile-cg 3.485 (0.57) | torch-bf16-compile-ro 36.94 (0.05) | **torch-bf16-cudnn 1.683 (1.19)**<br>torch-bf16-efficient 21.02 (0.09) | fa2-bf16 1.941 (1.03) | flashinfer-bf16 1.745 (1.14) |  | liger-bf16 3.149 (0.63) |
| llm_dec_b32_l8192_bf16 | a100 | 1.442 | torch-bf16 1.499 (0.96) | torch-bf16-compile 1.925 (0.75) | torch-bf16-compile-cg 1.94 (0.74) | torch-bf16-compile-ro 24.29 (0.06) | **torch-bf16-cudnn 1.082 (1.33)**<br>torch-bf16-efficient 17.59 (0.08) | fa2-bf16 1.177 (1.23) | flashinfer-bf16 1.193 (1.21) |  | liger-bf16 1.333 (1.08) |
| llm_dec_b32_l8192_bf16 | rtx-4090 | 1.923 | torch-bf16 1.996 (0.96) | **torch-bf16-compile 1.642 (1.17)** | torch-bf16-compile-cg 1.669 (1.15) | torch-bf16-compile-ro 31.14 (0.06) | torch-bf16-cudnn 2.053 (0.94)<br>torch-bf16-efficient 12.7 (0.15) | not run | not run |  | not run |
| llm_dec_b32_l8192_bf16 | h100 | 0.7362 | torch-bf16 0.6608 (1.11) | torch-bf16-compile 0.512 (1.44) | **torch-bf16-compile-cg 0.4866 (1.51)** | torch-bf16-compile-ro 0.4882 (1.51) | torch-bf16-cudnn 0.6801 (1.08)<br>torch-bf16-efficient 7.899 (0.09) | fa2-bf16 0.6512 (1.13)<br>fa3-bf16 0.6943 (1.06) | flashinfer-bf16 0.7902 (0.93) |  | liger-bf16 0.7033 (1.05) |
| llm_dec_b8_l2048_bf16 | rtx-5080 | 0.2239 | torch-bf16 0.2872 (0.78) | **torch-bf16-compile 0.222 (1.01)** | torch-bf16-compile-cg 0.2362 (0.95) | torch-bf16-compile-ro 1.958 (0.11) | torch-bf16-cudnn 0.2889 (0.77)<br>torch-bf16-efficient 1.246 (0.18) | fa2-bf16 0.2892 (0.77) | flashinfer-bf16 0.4182 (0.54) |  | liger-bf16 0.3768 (0.59) |
| llm_dec_b8_l2048_bf16 | rtx-3090 | 0.469 | torch-bf16 0.4905 (0.96) | **torch-bf16-compile 0.3388 (1.38)** | torch-bf16-compile-cg 0.356 (1.32) | torch-bf16-compile-ro 2.444 (0.19) | torch-bf16-cudnn 0.492 (0.95)<br>torch-bf16-efficient 1.628 (0.29) | fa2-bf16 0.5136 (0.91) | flashinfer-bf16 0.6094 (0.77) |  | liger-bf16 0.4984 (0.94) |
| llm_dec_b8_l2048_bf16 | a100 | 0.271 | torch-bf16 0.362 (0.75) | torch-bf16-compile 0.2812 (0.96) | **torch-bf16-compile-cg 0.2689 (1.01)** | torch-bf16-compile-ro 1.565 (0.17) | torch-bf16-cudnn 0.3791 (0.71)<br>torch-bf16-efficient 1.208 (0.22) | fa2-bf16 0.3692 (0.73) | flashinfer-bf16 0.4734 (0.57) |  | liger-bf16 0.4105 (0.66) |
| llm_dec_b8_l2048_bf16 | rtx-4090 | 0.2709 | torch-bf16 0.7702 (0.35) | **torch-bf16-compile 0.346 (0.78)** | torch-bf16-compile-cg 0.3813 (0.71) | torch-bf16-compile-ro 2.191 (0.12) | torch-bf16-cudnn 0.8154 (0.33)<br>torch-bf16-efficient 1.489 (0.18) | not run | not run |  | not run |
| llm_dec_b8_l2048_bf16 | h100 | 0.1419 | torch-bf16 0.3381 (0.42) | torch-bf16-compile 0.2247 (0.63) | **torch-bf16-compile-cg 0.1615 (0.88)** | torch-bf16-compile-ro 0.1624 (0.87) | torch-bf16-cudnn 0.378 (0.38)<br>torch-bf16-efficient 0.793 (0.18) | fa2-bf16 0.3288 (0.43)<br>fa3-bf16 0.3499 (0.41) | flashinfer-bf16 0.4545 (0.31) |  | liger-bf16 0.3905 (0.36) |
| llm_dec_b8_l8192_bf16 | rtx-5080 | 0.5133 | torch-bf16 0.5299 (0.97) | **torch-bf16-compile 0.4946 (1.04)** | torch-bf16-compile-cg 0.5092 (1.01) | torch-bf16-compile-ro 7.778 (0.07) | torch-bf16-cudnn 0.5548 (0.93)<br>torch-bf16-efficient 4.241 (0.12) | fa2-bf16 0.5441 (0.94) | flashinfer-bf16 0.6576 (0.78) |  | liger-bf16 0.6043 (0.85) |
| llm_dec_b8_l8192_bf16 | rtx-3090 | 0.8015 | torch-bf16 0.7267 (1.10) | torch-bf16-compile 0.8541 (0.94) | torch-bf16-compile-cg 0.8591 (0.93) | torch-bf16-compile-ro 9.324 (0.09) | torch-bf16-cudnn 0.7304 (1.10)<br>torch-bf16-efficient 5.682 (0.14) | fa2-bf16 0.7297 (1.10) | flashinfer-bf16 0.8244 (0.97) |  | **liger-bf16 0.7144 (1.12)** |
| llm_dec_b8_l8192_bf16 | a100 | 0.5165 | **torch-bf16 0.5052 (1.02)** | torch-bf16-compile 0.5837 (0.88) | torch-bf16-compile-cg 0.5976 (0.86) | torch-bf16-compile-ro 6.004 (0.09) | torch-bf16-cudnn 0.5265 (0.98)<br>torch-bf16-efficient 4.381 (0.12) | fa2-bf16 0.5096 (1.01) | flashinfer-bf16 0.6136 (0.84) |  | liger-bf16 0.5432 (0.95) |
| llm_dec_b8_l8192_bf16 | rtx-4090 | 0.5948 | torch-bf16 1.021 (0.58) | **torch-bf16-compile 0.6134 (0.97)** | torch-bf16-compile-cg 0.6429 (0.93) | torch-bf16-compile-ro 7.981 (0.07) | torch-bf16-cudnn 1.068 (0.56)<br>torch-bf16-efficient 3.723 (0.16) | not run | not run |  | not run |
| llm_dec_b8_l8192_bf16 | h100 | 0.2793 | torch-bf16 0.4078 (0.68) | torch-bf16-compile 0.2601 (1.07) | **torch-bf16-compile-cg 0.2278 (1.23)** | torch-bf16-compile-ro 0.2293 (1.22) | torch-bf16-cudnn 0.4303 (0.65)<br>torch-bf16-efficient 2.24 (0.12) | fa2-bf16 0.3912 (0.71)<br>fa3-bf16 0.4108 (0.68) | flashinfer-bf16 0.5274 (0.53) |  | liger-bf16 0.4336 (0.64) |
| llm_pre_t2048_bf16 | rtx-5080 | 2.65 | torch-bf16 2.213 (1.20) | torch-bf16-compile 1.937 (1.37) | **torch-bf16-compile-cg 1.931 (1.37)** | torch-bf16-compile-ro 1.971 (1.34) | torch-bf16-cudnn 2.164 (1.22)<br>torch-bf16-efficient 2.714 (0.98) | fa2-bf16 2.215 (1.20) | flashinfer-bf16 2.222 (1.19) | triton-bf16 2.294 (1.16) | liger-bf16 2.075 (1.28) |
| llm_pre_t2048_bf16 | rtx-3090 | 4.7 | torch-bf16 4.242 (1.11) | torch-bf16-compile 4.146 (1.13) | torch-bf16-compile-cg 3.87 (1.21) | **torch-bf16-compile-ro 3.742 (1.26)** | torch-bf16-cudnn 4.251 (1.11)<br>torch-bf16-efficient 4.598 (1.02) | fa2-bf16 4.195 (1.12) | flashinfer-bf16 4.231 (1.11) | triton-bf16 4.353 (1.08) | liger-bf16 3.954 (1.19) |
| llm_pre_t2048_bf16 | a100 | 2.103 | torch-bf16 2.522 (0.83) | torch-bf16-compile 1.227 (1.71) | **torch-bf16-compile-cg 1.082 (1.94)** | torch-bf16-compile-ro 1.121 (1.88) | torch-bf16-cudnn 1.625 (1.29)<br>torch-bf16-efficient 1.857 (1.13) | fa2-bf16 1.56 (1.35) | flashinfer-bf16 1.502 (1.40) | triton-bf16 1.702 (1.24) | liger-bf16 1.349 (1.56) |
| llm_pre_t2048_bf16 | rtx-4090 | 1.862 | torch-bf16 1.719 (1.08) | torch-bf16-compile 1.328 (1.40) | **torch-bf16-compile-cg 1.319 (1.41)** | torch-bf16-compile-ro 1.484 (1.25) | torch-bf16-cudnn 1.738 (1.07)<br>torch-bf16-efficient 1.875 (0.99) | not run | not run | not run | not run |
| llm_pre_t2048_bf16 | h100 | 1.234 | torch-bf16 0.5705 (2.16) | torch-bf16-compile 0.3576 (3.45) | torch-bf16-compile-cg 0.3509 (3.52) | **torch-bf16-compile-ro 0.3497 (3.53)** | torch-bf16-cudnn 0.574 (2.15)<br>torch-bf16-efficient 0.8057 (1.53) | fa3-bf16 0.5498 (2.24)<br>fa2-bf16 0.6388 (1.93) | flashinfer-bf16 0.5787 (2.13) | triton-bf16 0.6412 (1.92) | liger-bf16 0.4843 (2.55) |
| llm_blkdec_b1_l4096_f8w | rtx-5080 | 0.3615 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | rtx-3090 | 0.5482 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | a100 | 0.4754 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | h100 | 0.2082 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-5080 | 0.0538 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-3090 | 0.1586 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | a100 | 0.151 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | h100 | 0.07836 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-5080 | 0.1676 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-3090 | 0.2293 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | a100 | 0.2046 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | h100 | 0.1047 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-5080 | 0.7909 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-3090 | 0.9839 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | a100 | 0.7592 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | h100 | 0.3522 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-5080 | 2.813 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-3090 | 3.161 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | a100 | 2.079 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | h100 | 1.022 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-5080 | 0.2496 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-3090 | 0.812 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | a100 | 0.3186 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | h100 | 0.1307 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-5080 | 0.7264 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-3090 | 1.362 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | a100 | 0.6478 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | h100 | 0.3038 |  |  |  |  |  |  |  |  |  |

### Conv / FIR / elementwise

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand CUDA |
|---|---|---:|---|---|---|---|---|---|
| conv_2048_5 | rtx-5080 | 0.06689 |  | **torch-eager 0.1516 (0.44)** | torch-compile 0.1516 (0.44) | torch-compile-cg 0.1516 (0.44) | torch-compile-ro 0.1516 (0.44) |  |
| conv_2048_5 | rtx-3090 | 0.1102 |  | torch-eager 0.1618 (0.68) | torch-compile 0.1638 (0.67) | **torch-compile-cg 0.1608 (0.69)** | torch-compile-ro 0.1608 (0.69) |  |
| conv_2048_5 | a100 | 0.0734 |  | torch-eager 0.171 (0.43) | torch-compile 0.1372 (0.53) | **torch-compile-cg 0.1362 (0.54)** | torch-compile-ro 0.1362 (0.54) |  |
| conv_2048_5 | rtx-4090 | 0.05355 |  | torch-eager 0.0942 (0.57) | torch-compile 0.0942 (0.57) | **torch-compile-cg 0.0932 (0.57)** | torch-compile-ro 0.0934 (0.57) |  |
| conv_2048_5 | h100 | 0.04574 |  | **torch-eager 0.094 (0.49)** | torch-compile 0.0941 (0.49) | torch-compile-cg 0.0943 (0.49) | torch-compile-ro 0.0944 (0.48) |  |
| convmc_8_64_56_3 | rtx-5080 | 0.08304 |  | **torch-eager 0.1219 (0.68)** | torch-compile 0.1597 (0.52) | torch-compile-cg 0.1577 (0.53) | torch-compile-ro 0.1577 (0.53) |  |
| convmc_8_64_56_3 | rtx-3090 | 0.148 |  | **torch-eager 0.0788 (1.88)** | torch-compile 0.169 (0.88) | torch-compile-cg 0.1608 (0.92) | torch-compile-ro 0.1618 (0.91) |  |
| convmc_8_64_56_3 | a100 | 0.1886 |  | **torch-eager 0.087 (2.17)** | torch-compile 0.1812 (1.04) | torch-compile-cg 0.169 (1.12) | torch-compile-ro 0.17 (1.11) |  |
| convmc_8_64_56_3 | rtx-4090 | 0.09594 |  | **torch-eager 0.0584 (1.64)** | torch-compile 0.1055 (0.91) | torch-compile-cg 0.0993 (0.97) | torch-compile-ro 0.0993 (0.97) |  |
| convmc_8_64_56_3 | h100 | 0.1148 |  | **torch-eager 0.0715 (1.61)** | torch-compile 0.111 (1.03) | torch-compile-cg 0.1056 (1.09) | torch-compile-ro 0.1053 (1.09) |  |
| ew_4096 | rtx-5080 | 0.08924 | chain-eager 0.5734 (0.16) | torch-eager 0.5734 (0.16) | torch-compile 0.2447 (0.36) | **torch-compile-cg 0.2437 (0.37)** | torch-compile-ro 0.2437 (0.37) | chain-fused 0.2478 (0.36) |
| ew_4096 | rtx-3090 | 0.1267 | chain-eager 0.5632 (0.22) | torch-eager 0.5642 (0.22) | torch-compile 0.2406 (0.53) | torch-compile-cg 0.2406 (0.53) | **torch-compile-ro 0.2396 (0.53)** | chain-fused 0.2406 (0.53) |
| ew_4096 | a100 | 0.1341 | chain-eager 0.3666 (0.37) | torch-eager 0.3584 (0.37) | torch-compile 0.1556 (0.86) | **torch-compile-cg 0.1536 (0.87)** | torch-compile-ro 0.1556 (0.86) | chain-fused 0.1608 (0.83) |
| ew_4096 | rtx-4090 | 0.08164 | chain-eager 0.5806 (0.14) | torch-eager 0.6033 (0.14) | torch-compile 0.255 (0.32) | **torch-compile-cg 0.253 (0.32)** | torch-compile-ro 0.254 (0.32) | chain-fused 0.2532 (0.32) |
| ew_4096 | h100 | 0.1202 | chain-eager 0.1982 (0.61) | torch-eager 0.1694 (0.71) | **torch-compile 0.0728 (1.65)** | torch-compile-cg 0.0729 (1.65) | torch-compile-ro 0.073 (1.65) | chain-fused 0.0823 (1.46) |
| ewconvew_2048_5 | rtx-5080 | 0.07788 |  | torch-eager 0.2079 (0.37) | torch-compile 0.1884 (0.41) | **torch-compile-cg 0.1881 (0.41)** | torch-compile-ro 0.1884 (0.41) |  |
| ewconvew_2048_5 | rtx-3090 | 0.1304 |  | torch-eager 0.298 (0.44) | **torch-compile 0.2447 (0.53)** | torch-compile-cg 0.2662 (0.49) | torch-compile-ro 0.2662 (0.49) |  |
| ewconvew_2048_5 | a100 | 0.08923 |  | torch-eager 0.1946 (0.46) | torch-compile 0.1772 (0.50) | torch-compile-cg 0.1741 (0.51) | **torch-compile-ro 0.1731 (0.52)** |  |
| ewconvew_2048_5 | rtx-4090 | 0.06455 |  | torch-eager 0.1516 (0.43) | torch-compile 0.131 (0.49) | **torch-compile-cg 0.13 (0.50)** | torch-compile-ro 0.1311 (0.49) |  |
| ewconvew_2048_5 | h100 | 0.05628 |  | torch-eager 0.125 (0.45) | torch-compile 0.1147 (0.49) | **torch-compile-cg 0.1118 (0.50)** | torch-compile-ro 0.1143 (0.49) |  |
| fir_4194304_16 | rtx-5080 | 0.04556 |  | **torch-eager 0.1044 (0.44)** | torch-compile 0.1044 (0.44) | torch-compile-cg 0.1044 (0.44) | torch-compile-ro 0.1044 (0.44) |  |
| fir_4194304_16 | rtx-3090 | 0.07393 |  | torch-eager 0.1198 (0.62) | torch-compile 0.1198 (0.62) | **torch-compile-cg 0.1188 (0.62)** | torch-compile-ro 0.1188 (0.62) |  |
| fir_4194304_16 | a100 | 0.05142 |  | torch-eager 0.0881 (0.58) | torch-compile 0.087 (0.59) | **torch-compile-cg 0.085 (0.60)** | torch-compile-ro 0.085 (0.60) |  |
| fir_4194304_16 | rtx-4090 | 0.03763 |  | torch-eager 0.0666 (0.57) | torch-compile 0.0666 (0.57) | **torch-compile-cg 0.0655 (0.57)** | torch-compile-ro 0.0655 (0.57) |  |
| fir_4194304_16 | h100 | 0.03295 |  | **torch-eager 0.0605 (0.54)** | torch-compile 0.0605 (0.54) | torch-compile-cg 0.0608 (0.54) | torch-compile-ro 0.0607 (0.54) |  |
| firew_rowsum_2048x2048_16 | rtx-5080 | 0.05363 |  | torch-eager 0.128 (0.42) | **torch-compile 0.1085 (0.49)** | torch-compile-cg 0.1085 (0.49) | torch-compile-ro 0.1085 (0.49) |  |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1018 |  | torch-eager 0.2284 (0.45) | torch-compile 0.1413 (0.72) | **torch-compile-cg 0.1393 (0.73)** | torch-compile-ro 0.1393 (0.73) |  |
| firew_rowsum_2048x2048_16 | a100 | 0.0643 |  | torch-eager 0.13 (0.49) | torch-compile 0.0973 (0.66) | torch-compile-cg 0.0952 (0.68) | **torch-compile-ro 0.0942 (0.68)** |  |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.04429 |  | torch-eager 0.0932 (0.48) | torch-compile 0.0727 (0.61) | **torch-compile-cg 0.0696 (0.64)** | torch-compile-ro 0.0707 (0.63) |  |
| firew_rowsum_2048x2048_16 | h100 | 0.03921 |  | torch-eager 0.0871 (0.45) | torch-compile 0.0673 (0.58) | torch-compile-cg 0.0662 (0.59) | **torch-compile-ro 0.0655 (0.60)** |  |
| convmc_8_64_56_3_tf32 | rtx-5080 | 0.08321 |  | **torch-eager-tf32 0.0707 (1.18)** | torch-compile-tf32 0.0973 (0.86) | torch-compile-cg-tf32 0.0963 (0.86) | torch-compile-ro-tf32 0.0983 (0.85) |  |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1481 |  | **torch-eager-tf32 0.088 (1.68)** | torch-compile-tf32 0.1464 (1.01) | torch-compile-cg-tf32 0.1413 (1.05) | torch-compile-ro-tf32 0.1423 (1.04) |  |
| convmc_8_64_56_3_tf32 | a100 | 0.1892 |  | **torch-eager-tf32 0.0481 (3.93)** | torch-compile-tf32 0.0829 (2.28) | torch-compile-cg-tf32 0.0748 (2.53) | torch-compile-ro-tf32 0.0758 (2.50) |  |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.09545 |  | **torch-eager-tf32 0.0439 (2.17)** | torch-compile-tf32 0.0822 (1.16) | torch-compile-cg-tf32 0.0829 (1.15) | torch-compile-ro-tf32 0.0799 (1.19) |  |
| convmc_8_64_56_3_tf32 | h100 | 0.1178 |  | **torch-eager-tf32 0.038 (3.10)** | torch-compile-tf32 0.0467 (2.52) | torch-compile-cg-tf32 0.0438 (2.69) | torch-compile-ro-tf32 0.0442 (2.66) |  |

### Fan-in/out, loops, CPU+GPU

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead |
|---|---|---:|---|---|---|---|---|
| diamond_2048 | rtx-5080 | 1.386 | chain-lt 1.384 (1.00)<br>chain-streams 1.401 (0.99)<br>chain-eager 1.404 (0.99)<br>chain-cutlass 1.652 (0.84) | torch-eager 1.384 (1.00) | torch-compile 1.368 (1.01) | **torch-compile-cg 1.366 (1.02)** | torch-compile-ro 1.38 (1.00) |
| diamond_2048 | rtx-3090 | 2.401 | **chain-streams 2.874 (0.84)**<br>chain-eager 2.897 (0.83)<br>chain-lt 2.901 (0.83)<br>chain-cutlass 3.384 (0.71) | torch-eager 3 (0.80) | torch-compile 2.908 (0.83) | torch-compile-cg 2.991 (0.80) | torch-compile-ro 3.028 (0.79) |
| diamond_2048 | a100 | 4.077 | chain-lt 3.085 (1.32)<br>chain-streams 3.091 (1.32)<br>chain-eager 3.119 (1.31)<br>chain-cutlass 3.622 (1.13) | torch-eager 3.771 (1.08) | torch-compile 3.742 (1.09) | **torch-compile-cg 3.05 (1.34)** | torch-compile-ro 3.064 (1.33) |
| diamond_2048 | rtx-4090 | 0.9404 | chain-lt 0.9413 (1.00)<br>chain-cutlass 0.9533 (0.99)<br>chain-streams 0.9554 (0.98)<br>chain-eager 0.9595 (0.98) | torch-eager 0.9595 (0.98) | **torch-compile 0.9411 (1.00)** | torch-compile-cg 0.9411 (1.00) | torch-compile-ro 0.9513 (0.99) |
| diamond_2048 | h100 | 1.165 | **chain-lt 1.025 (1.14)**<br>chain-streams 1.034 (1.13)<br>chain-eager 1.037 (1.12)<br>chain-cutlass 1.256 (0.93) | torch-eager 1.04 (1.12) | torch-compile 1.033 (1.13) | torch-compile-cg 1.031 (1.13) | torch-compile-ro 1.04 (1.12) |
| fanout_2048 | rtx-5080 | 0.9163 | **chain-streams 0.9132 (1.00)**<br>chain-eager 0.9236 (0.99) | torch-eager 0.938 (0.98) | torch-compile 0.9247 (0.99) | torch-compile-cg 0.9233 (0.99) | torch-compile-ro 0.9236 (0.99) |
| fanout_2048 | rtx-3090 | 1.616 | chain-streams 1.946 (0.83)<br>chain-eager 1.978 (0.82) | torch-eager 2.024 (0.80) | torch-compile 2.045 (0.79) | torch-compile-cg 1.886 (0.86) | **torch-compile-ro 1.844 (0.88)** |
| fanout_2048 | a100 | 2.721 | chain-streams 2.058 (1.32)<br>chain-eager 2.077 (1.31) | **torch-eager 2.023 (1.34)** | torch-compile 2.603 (1.05) | torch-compile-cg 2.165 (1.26) | torch-compile-ro 2.024 (1.34) |
| fanout_2048 | rtx-4090 | 0.6424 | **chain-streams 0.6368 (1.01)**<br>chain-eager 0.6441 (1.00) | torch-eager 0.684 (0.94) | torch-compile 0.6595 (0.97) | torch-compile-cg 0.6605 (0.97) | torch-compile-ro 0.642 (1.00) |
| fanout_2048 | h100 | 0.7799 | **chain-streams 0.6813 (1.14)**<br>chain-eager 0.693 (1.13) | torch-eager 0.7028 (1.11) | torch-compile 0.6923 (1.13) | torch-compile-cg 0.6912 (1.13) | torch-compile-ro 0.6924 (1.13) |
| indep_2048 | rtx-5080 | 0.9508 | chain-lt 0.9196 (1.03)<br>chain-streams 0.9384 (1.01)<br>chain-eager 0.9421 (1.01)<br>chain-cutlass 1.096 (0.87) | torch-eager 0.9257 (1.03) | **torch-compile 0.9155 (1.04)** | torch-compile-cg 0.9175 (1.04) | torch-compile-ro 0.9277 (1.02) |
| indep_2048 | rtx-3090 | 1.609 | **chain-streams 1.907 (0.84)**<br>chain-lt 1.942 (0.83)<br>chain-eager 1.954 (0.82)<br>chain-cutlass 2.258 (0.71) | torch-eager 1.952 (0.82) | torch-compile 2.243 (0.72) | torch-compile-cg 1.94 (0.83) | torch-compile-ro 1.986 (0.81) |
| indep_2048 | a100 | 2.732 | chain-lt 2.038 (1.34)<br>chain-streams 2.041 (1.34)<br>chain-eager 2.072 (1.32)<br>chain-cutlass 2.392 (1.14) | torch-eager 2.526 (1.08) | **torch-compile 1.995 (1.37)** | torch-compile-cg 2.019 (1.35) | torch-compile-ro 2.007 (1.36) |
| indep_2048 | rtx-4090 | 0.6291 | chain-lt 0.6359 (0.99)<br>chain-cutlass 0.6431 (0.98)<br>chain-streams 0.6451 (0.98)<br>chain-eager 0.6513 (0.97) | torch-eager 0.6513 (0.97) | **torch-compile 0.6342 (0.99)** | torch-compile-cg 0.6349 (0.99) | torch-compile-ro 0.6442 (0.98) |
| indep_2048 | h100 | 0.7879 | **chain-lt 0.6868 (1.15)**<br>chain-streams 0.6974 (1.13)<br>chain-eager 0.6984 (1.13)<br>chain-cutlass 0.846 (0.93) | torch-eager 0.6993 (1.13) | torch-compile 0.6915 (1.14) | torch-compile-cg 0.6908 (1.14) | torch-compile-ro 0.7009 (1.12) |
| indep_512 | rtx-5080 | 0.03816 | chain-streams 0.0365 (1.05)<br>chain-lt 0.0451 (0.85)<br>chain-eager 0.0471 (0.81)<br>chain-cutlass 0.1065 (0.36) | torch-eager 0.043 (0.89) | **torch-compile 0.0328 (1.16)** | torch-compile-cg 0.0328 (1.16) | torch-compile-ro 0.043 (0.89) |
| indep_512 | rtx-3090 | 0.07292 | chain-streams 0.0532 (1.37)<br>chain-lt 0.0543 (1.34)<br>chain-eager 0.0573 (1.27)<br>chain-cutlass 0.126 (0.58) | torch-eager 0.0707 (1.03) | **torch-compile 0.0471 (1.55)** | torch-compile-cg 0.0471 (1.55) | torch-compile-ro 0.0543 (1.34) |
| indep_512 | a100 | 0.1412 | **chain-streams 0.0573 (2.46)**<br>chain-lt 0.0655 (2.16)<br>chain-eager 0.0666 (2.12)<br>chain-cutlass 0.2048 (0.69) | torch-eager 0.0686 (2.06) | torch-compile 0.0829 (1.70) | torch-compile-cg 0.0799 (1.77) | torch-compile-ro 0.0799 (1.77) |
| indep_512 | rtx-4090 | 0.05657 | chain-lt 0.038 (1.49)<br>chain-streams 0.0389 (1.45)<br>chain-eager 0.0401 (1.41)<br>chain-cutlass 0.0892 (0.63) | torch-eager 0.044 (1.29) | torch-compile 0.0297 (1.90) | **torch-compile-cg 0.0288 (1.96)** | torch-compile-ro 0.042 (1.35) |
| indep_512 | h100 | 0.05934 | chain-streams 0.0313 (1.90)<br>chain-lt 0.034 (1.75)<br>chain-eager 0.0369 (1.61)<br>chain-cutlass 0.1191 (0.50) | torch-eager 0.0367 (1.62) | **torch-compile 0.0284 (2.09)** | torch-compile-cg 0.0284 (2.09) | torch-compile-ro 0.0351 (1.69) |
| loop_4096 | rtx-5080 | 26.93 | **chain-lt 28.72 (0.94)**<br>chain-cutlass 28.99 (0.93)<br>chain-eager 30.37 (0.89) | torch-eager 29.27 (0.92) | torch-compile 29.17 (0.92) | torch-compile-cg 29.14 (0.92) | torch-compile-ro 29.16 (0.92) |
| loop_4096 | rtx-3090 | 66.08 | **chain-lt 54.85 (1.20)**<br>chain-cutlass 57.63 (1.15)<br>chain-eager 58.38 (1.13) | torch-eager 55 (1.20) | torch-compile 55.09 (1.20) | torch-compile-cg 55.14 (1.20) | torch-compile-ro 55.14 (1.20) |
| loop_4096 | a100 | 82.56 | chain-eager 62.85 (1.31)<br>chain-lt 64.39 (1.28)<br>chain-cutlass 65.13 (1.27) | **torch-eager 62.35 (1.32)** | torch-compile 72.61 (1.14) | torch-compile-cg 73.44 (1.12) | torch-compile-ro 63.08 (1.31) |
| loop_4096 | rtx-4090 | 18.82 | **chain-cutlass 19.22 (0.98)**<br>chain-lt 19.56 (0.96)<br>chain-eager 20.67 (0.91) | torch-eager 19.89 (0.95) | torch-compile 19.47 (0.97) | torch-compile-cg 19.47 (0.97) | torch-compile-ro 19.59 (0.96) |
| loop_4096 | h100 | 22.9 | **chain-lt 21.12 (1.08)**<br>chain-eager 21.64 (1.06)<br>chain-cutlass 26.9 (0.85) | torch-eager 21.64 (1.06) | torch-compile 21.74 (1.05) | torch-compile-cg 21.72 (1.05) | torch-compile-ro 21.72 (1.05) |
| loopmm_1024_8 | rtx-5080 | 0.617 |  | torch-eager 0.6308 (0.98) | torch-compile 0.5908 (1.04) | **torch-compile-cg 0.5878 (1.05)** | torch-compile-ro 0.6246 (0.99) |
| loopmm_1024_8 | rtx-3090 | 1.107 |  | torch-eager 1.131 (0.98) | torch-compile 0.9277 (1.19) | torch-compile-cg 1.072 (1.03) | **torch-compile-ro 0.9226 (1.20)** |
| loopmm_1024_8 | a100 | 1.575 |  | **torch-eager 1.093 (1.44)** | torch-compile 1.329 (1.18) | torch-compile-cg 1.321 (1.19) | torch-compile-ro 1.32 (1.19) |
| loopmm_1024_8 | rtx-4090 | 0.4151 |  | torch-eager 0.3619 (1.15) | torch-compile 0.3932 (1.06) | torch-compile-cg 0.3871 (1.07) | **torch-compile-ro 0.3533 (1.17)** |
| loopmm_1024_8 | h100 | 0.5147 |  | torch-eager 0.4595 (1.12) | torch-compile 0.4558 (1.13) | **torch-compile-cg 0.4503 (1.14)** | torch-compile-ro 0.4535 (1.14) |
| rt_mlp2_1024 | rtx-5080 | 2.753 |  | torch-eager 3.386 (0.81) | torch-compile 2.538 (1.08) | **torch-compile-cg 2.516 (1.09)** | torch-compile-ro 2.61 (1.05) |
| rt_mlp2_1024 | rtx-3090 | 4.313 |  | torch-eager 7.413 (0.58) | **torch-compile 5.396 (0.80)** | torch-compile-cg 5.492 (0.79) | torch-compile-ro 5.619 (0.77) |
| rt_mlp2_1024 | a100 | 7.889 |  | torch-eager 5.629 (1.40) | torch-compile 5.52 (1.43) | **torch-compile-cg 4.502 (1.75)** | torch-compile-ro 4.52 (1.75) |
| rt_mlp2_1024 | rtx-4090 | 1.67 |  | torch-eager 3.686 (0.45) | torch-compile 1.612 (1.04) | torch-compile-cg 1.53 (1.09) | **torch-compile-ro 1.519 (1.10)** |
| rt_mlp2_1024 | h100 | 2.11 |  | torch-eager 2.93 (0.72) | torch-compile 2.047 (1.03) | torch-compile-cg 1.991 (1.06) | **torch-compile-ro 1.96 (1.08)** |
| rt_mlp2_2048 | rtx-5080 | 15.45 |  | torch-eager 19.4 (0.80) | **torch-compile 15.62 (0.99)** | torch-compile-cg 15.64 (0.99) | torch-compile-ro 15.69 (0.99) |
| rt_mlp2_2048 | rtx-3090 | 26.34 |  | torch-eager 40.88 (0.64) | **torch-compile 32.76 (0.80)** | torch-compile-cg 32.77 (0.80) | torch-compile-ro 33.09 (0.80) |
| rt_mlp2_2048 | a100 | 44.17 |  | torch-eager 39.49 (1.12) | torch-compile 33.36 (1.32) | **torch-compile-cg 33.23 (1.33)** | torch-compile-ro 33.59 (1.31) |
| rt_mlp2_2048 | rtx-4090 | 10.3 |  | torch-eager 14.71 (0.70) | torch-compile 10.47 (0.98) | **torch-compile-cg 10.44 (0.99)** | torch-compile-ro 10.51 (0.98) |
| rt_mlp2_2048 | h100 | 12.26 |  | torch-eager 14.41 (0.85) | torch-compile 11.29 (1.09) | **torch-compile-cg 11.18 (1.10)** | torch-compile-ro 11.31 (1.08) |
| twin_2048 | rtx-5080 | 0.9286 |  | torch-eager 0.9462 (0.98) | torch-compile 0.9196 (1.01) | **torch-compile-cg 0.9175 (1.01)** | torch-compile-ro 0.9175 (1.01) |
| twin_2048 | rtx-3090 | 1.589 |  | torch-eager 1.905 (0.83) | torch-compile 2.15 (0.74) | torch-compile-cg 1.916 (0.83) | **torch-compile-ro 1.734 (0.92)** |
| twin_2048 | a100 | 2.7 |  | torch-eager 2.062 (1.31) | **torch-compile 1.97 (1.37)** | torch-compile-cg 1.988 (1.36) | torch-compile-ro 1.988 (1.36) |
| twin_2048 | rtx-4090 | 0.6312 |  | **torch-eager 0.6451 (0.98)** | torch-compile 0.6554 (0.96) | torch-compile-cg 0.6523 (0.97) | torch-compile-ro 0.6513 (0.97) |
| twin_2048 | h100 | 0.7814 |  | torch-eager 0.6851 (1.14) | **torch-compile 0.6843 (1.14)** | torch-compile-cg 0.6847 (1.14) | torch-compile-ro 0.6843 (1.14) |

## Command

```
python3 benches/local/matrix.py --skip 'llm_kv[xn]_' \
  rtx-5080=.claude/raw-evidence/cross-arch/5fcb401b/rtx-5080/resS/results-rtx-5080.txt@5fcb401b,.claude/raw-evidence/cross-arch/5fcb401b/rtx-5080/resM/results-rtx-5080.txt@5fcb401b,.claude/raw-evidence/local-rtx-5080/confirmation-fixes/changed-resS/results-rtx-5080.txt@39cd2c87,.claude/raw-evidence/local-rtx-5080/confirmation-fixes/changed-resM/results-rtx-5080.txt@39cd2c87 \
  rtx-3090=.claude/raw-evidence/cross-arch/5fcb401b/rtx-3090/results-rtx-3090.txt@5fcb401b \
  a100=.claude/raw-evidence/cross-arch/5fcb401b/a100/results-a100.txt@5fcb401b \
  rtx-4090=.claude/raw-evidence/cross-arch/5fcb401b/rtx-4090/results-rtx-4090.txt@5fcb401b \
  h100=.claude/raw-evidence/cross-arch/5fcb401b/h100/results-h100.txt@5fcb401b
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
