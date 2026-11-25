# Mapal vs the fastest competitor at the same precision

Built 2026-11-25 by `benches/local/matrix.py` from existing results files (command at the end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even (|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.

**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 activations), `_bf16kv` rows (weights and KV cache bf16, f32 compute) and `_bf16` rows (bf16 storage, `: bf16 | f32` products) vs the bf16 legs (bf16 throughout); a bf16-storage row without bf16 legs of its own takes its twin's (`_bf16w`, else the f32 program: the same bf16 computation); `_f8w` rows (fp8 weights, f32 compute) vs fp8 legs, of which none is installed: `(no leg)`, their `_bf16w` twin's bf16 leg is a reference only. The torch bf16 reference of an f32 / tf32 row is never its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count only with `within=yes`.

**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch's pick, `-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials' matmul / softmax / fused attention, autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. **Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang as servers. Every part ran the competitor legs (FlashAttention-3 on sm_90a only; `unavailable` = the part's run reported the package missing or its build capped).

**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, `chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; `cutlass` / `cutlass-tuned` the cells' CUTLASS stock / tuned.

| part | commit(s) | Mapal rows from (step@commit: rows) |
|---|---|---|
| rtx-5080 | 9abeb446, d450014d, d945c641 | gapfix-integrate/resM@d945c641: 51; gapfix-integrate/zeroab2@9abeb446: 47; gapfix-integrate/resS@d945c641: 26; gapfix-integrate/flyab@d450014d: 3 |
| rtx-3090 | f1d3a04f | f1d3a04f/rtx-3090@f1d3a04f: 127 |
| a100 | f1d3a04f | f1d3a04f/a100@f1d3a04f: 126 |
| rtx-4090 | f1d3a04f | f1d3a04f/rtx-4090@f1d3a04f: 127 |
| h100 | f1d3a04f | f1d3a04f/h100@f1d3a04f: 127 |

## Overall

| geomean Mapal/best | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 0.927 (47) | 1.116 (47) | 1.389 (46) | 1.075 (47) | 1.511 (47) |
| MLP chains | 0.988 (6) | 1.131 (6) | 1.263 (6) | 1.123 (6) | 1.381 (6) |
| Attention / softmax / long rows | 0.642 (14) | 0.742 (14) | 0.853 (14) | 0.678 (14) | 0.933 (14) |
| LLM decode / prefill / block (f32, tf32, bf16 storage) | 0.987 (37) | 0.962 (37) | 1.121 (37) | 0.903 (37) | 1.031 (37) |
| Conv / FIR / elementwise | 0.493 (7) | 0.879 (7) | 1.242 (7) | 0.879 (7) | 0.908 (7) |
| Fan-in/out, loops, CPU+GPU | 0.992 (9) | 1.026 (9) | 1.019 (9) | 1.040 (9) | 1.183 (9) |
| all f32 rows | 0.775 (66) | 0.891 (66) | 0.953 (65) | 0.869 (66) | 0.974 (66) |
| all tf32 rows | 1.154 (13) | 1.139 (13) | 2.087 (13) | 1.239 (13) | 1.688 (13) |
| all bf16w rows | 1.559 (9) | 1.419 (9) | 1.682 (9) | 1.377 (9) | 1.749 (9) |
| all bf16kv rows | 0.971 (7) | 0.938 (7) | 1.035 (7) | 0.863 (7) | 1.022 (7) |
| all bf16 rows | 0.846 (25) | 1.122 (25) | 1.423 (25) | 0.958 (25) | 1.620 (25) |
| all f8w rows | - | - | - | - | - |
| **all rows** | 0.880 (120) | 0.997 (120) | 1.185 (119) | 0.954 (120) | 1.204 (120) |
| rows timed on every part (119) | 0.879 (119) | 0.996 (119) | 1.185 (119) | 0.954 (119) | 1.205 (119) |
| win / even / loss | 47/42/31 | 42/18/60 | 41/6/72 | 42/26/52 | 25/7/88 |

Geomean cells: value (rows). Parts time different row sets; compare columns on the "every part" row.

## GEMM cells & projections

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| epi_4096 | f32 | 0.99≈ 3.538/3.57 (chain-lt) | **0.84** 5.567/6.654 (torch-compile-cg) | **0.91** 6.618/7.27 (chain-lt) | **0.95** 4.581/4.843 (chain-lt) | 1.13 4.408/3.912 (torch-compile-cg) |
| mm_16x2048x7168 | f32 | 1.00≈ 0.0799/0.0799 (cublas) | 1.03≈ 0.0901/0.08755 (cublas) | 1.02≈ 0.0963/0.0942 (cublas) | 1.15 0.1305/0.1137 (cublas) | 1.59 0.1017/0.0639 (cublas) |
| mm_4096x4096x4096 | f32 | 1.04≈ 3.818/3.678 (cublas) | 1.10 7.511/6.804 (cublas) | - | 0.97≈ 4.673/4.837 (cublas) | 1.17 4.84/4.134 (cublas) |
| pj_m128_n28672_k4096 | f32 | 1.07 0.8019/0.7501 (lt) | 0.98≈ 1.202/1.223 (torch-compile-ro) | **0.93** 1.616/1.739 (cublas) | **0.94** 1.105/1.18 (lt) | 1.06 0.9701/0.918 (torch-compile-ro) |
| pj_m128_n4096_k14336 | f32 | 1.09 0.4447/0.4071 (lt-fresh) | 1.03≈ 0.7032/0.684 (torch-compile) | 1.28 1.101/0.8591 (torch-eager) | 1.00≈ 0.6412/0.64 (torch-eager) | 1.21 0.5062/0.4187 (torch-compile-ro) |
| pj_m128_n4096_k4096 | f32 | 0.99≈ 0.1368/0.1383 (lt-fresh) | 1.10 0.2449/0.2222 (lt) | 1.18 0.3488/0.2959 (cublas) | 1.18 0.2367/0.2007 (lt-fresh) | 1.30 0.1592/0.1222 (lt-fresh) |
| pj_m128_n6144_k4096 | f32 | 1.07 0.1953/0.1826 (lt-fresh) | 1.05 0.3378/0.3205 (torch-compile-cg) | 1.55 0.6778/0.4383 (lt) | 1.12 0.3358/0.3 (torch-compile-cg) | 1.20 0.2288/0.1901 (torch-compile-ro) |
| pj_m16_n18432_k7168 | f32 | 1.01≈ 0.596/0.5904 (lt-fresh) | 1.04≈ 0.64/0.6124 (torch-compile) | 1.10 0.4718/0.4291 (torch-compile) | 1.04≈ 0.7444/0.7127 (torch-compile) | **0.95** 0.3194/0.3365 (torch-compile) |
| pj_m16_n2048_k7168 | f32 | **0.67** 0.03463/0.052 (lt-fresh) | 1.25 0.1053/0.084 (lt) | 1.32 0.1253/0.0952 (torch-eager) | 1.40 0.08037/0.0573 (cublas-fresh) ±10% | 1.22 0.05923/0.0487 (torch-compile-cg) |
| pj_m16_n28672_k4096 | f32 | 1.02≈ 0.5377/0.528 (lt-fresh) | 1.07 0.5816/0.5448 (torch-compile) | 1.17 0.4755/0.4055 (torch-compile) | 1.03≈ 0.6595/0.638 (lt-fresh) | 1.06 0.2744/0.2601 (torch-compile-cg) |
| pj_m16_n4096_k14336 | f32 | **0.80** 0.2339/0.294 (torch-compile) | 1.08 0.2968/0.2755 (torch-compile) | **0.87** 0.2708/0.3103 (torch-compile-cg) | 1.07 0.3697/0.3471 (cublas-fresh) | **0.94** 0.1593/0.1694 (torch-compile) |
| pj_m16_n4096_k4096 | f32 | **0.66** 0.03876/0.0591 (lt-fresh) | 1.42 0.1206/0.0851 (torch-compile-cg) | 1.14 0.1188/0.1044 (torch-compile-cg) | 1.28 0.08238/0.0645 (lt-fresh) | 1.08 0.053/0.049 (torch-compile) |
| pj_m16_n6144_k4096 | f32 | 0.96≈ 0.08607/0.0899 (lt-fresh) | 1.25 0.1542/0.1229 (torch-compile) | 1.22 0.1427/0.1167 (torch-compile-cg) | 1.26 0.1575/0.1249 (cublas-fresh) | 1.23 0.06957/0.0565 (torch-compile) |
| pj_m1_n18432_k7168 | f32 | 0.95≈ 0.5869/0.6173 (cublas-fresh) | 1.01≈ 0.6127/0.6083 (lt) | 1.08 0.4274/0.3953 (lt-fresh) | 1.17 0.7272/0.6195 (lt-fresh) | 1.04≈ 0.1992/0.1924 (torch-compile-cg) |
| pj_m1_n2048_k7168 | f32 | **0.80** 0.02402/0.03 (lt-fresh) ±11% | 1.16 0.08661/0.0748 (lt-fresh) | 1.63 0.08678/0.0532 (cublas-fresh) | 0.92≈ 0.05095/0.0553 (lt-fresh) ±14% | 1.66 0.0379/0.0228 (cublas-fresh) |
| pj_m1_n28672_k4096 | f32 | 1.03≈ 0.5219/0.5063 (lt-fresh) | 1.02≈ 0.5487/0.5366 (torch-compile-ro) | 1.12 0.3932/0.3512 (torch-compile-cg) | 1.09 0.6547/0.599 (torch-compile-cg) | 1.01≈ 0.1813/0.1801 (cublas) |
| pj_m1_n4096_k14336 | f32 | 0.96≈ 0.2238/0.2342 (lt-fresh) | 1.04≈ 0.2852/0.2755 (torch-compile-ro) | 1.16 0.2133/0.1833 (cublas-fresh) | 1.45 0.3351/0.2314 (lt-fresh) | 1.21 0.1008/0.0835 (cublas-fresh) |
| pj_m1_n4096_k4096 | f32 | 1.26 0.03005/0.0238 (lt-fresh) ±5% | 1.12 0.09665/0.086 (lt) | 1.53 0.09081/0.0594 (cublas-fresh) | 1.33 0.04615/0.0348 (lt-fresh) ±5% | 1.96 0.04356/0.0222 (cublas-fresh) |
| pj_m1_n6144_k4096 | f32 | 1.28 0.06874/0.0539 (lt-fresh) | 1.11 0.1355/0.1219 (torch-compile) | 1.33 0.1129/0.085 (cublas-fresh) | 1.45 0.1011/0.0696 (lt-fresh) ±6% | 1.55 0.05403/0.0348 (cublas-fresh) |
| pj_m2048_n28672_k4096 | f32 | 0.97≈ 11.8/12.23 (lt-fresh) | **0.90** 19.3/21.53 (lt-fresh) | **0.91** 23.22/25.48 (torch-compile) | 1.00≈ 16.4/16.45 (cublas-fresh) | 1.17 16.15/13.86 (torch-compile) |
| pj_m2048_n4096_k14336 | f32 | 1.00≈ 6.412/6.434 (cublas) | 0.98≈ 11.02/11.24 (lt-fresh) | **0.93** 11.76/12.65 (torch-compile-ro) | 0.96≈ 8.073/8.425 (cublas) | 1.34 8.443/6.322 (torch-compile) |
| pj_m2048_n4096_k4096 | f32 | 1.02≈ 1.832/1.798 (lt-fresh) | **0.88** 2.872/3.255 (cublas) | **0.91** 3.372/3.692 (torch-compile-cg) | 0.97≈ 2.348/2.408 (lt) | 1.45 2.446/1.686 (torch-compile-cg) |
| pj_m2048_n6144_k4096 | f32 | 1.00≈ 2.643/2.652 (lt-fresh) | 0.97≈ 4.576/4.699 (cublas-fresh) | **0.89** 5.094/5.711 (lt) | 0.96≈ 3.502/3.631 (cublas) | 0.98≈ 2.73/2.793 (torch-compile-ro) |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.17 0.6983/0.5952 (lt-tf32-fresh) | 1.12 1.078/0.9636 (torch-compile-tf32) | 1.57 0.651/0.4147 (triton-tf32) | 1.22 1.032/0.8448 (torch-compile-tf32) | 1.35 0.3335/0.2465 (cublas-tf32) |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.30 0.4065/0.3127 (lt-tf32-fresh) | 1.09 0.6794/0.6205 (torch-compile-cg-tf32) | 1.61 0.4112/0.255 (cublas-tf32) | 1.30 0.534/0.4096 (lt-tf32-fresh) | 2.32 0.3395/0.1464 (torch-compile-cg-tf32) |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.48 0.1297/0.0877 (lt-tf32-fresh) | 1.30 0.2177/0.168 (lt-tf32-fresh) | 2.27 0.1978/0.087 (triton-tf32) | 1.36 0.1742/0.128 (cublas-tf32-fresh) | 2.03 0.08789/0.0434 (torch-compile-cg-tf32) |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.38 0.1797/0.1304 (lt-tf32-fresh) | 1.16 0.2905/0.2499 (lt-tf32) | 2.07 0.2077/0.1004 (triton-tf32) | 1.30 0.2686/0.2068 (lt-tf32-fresh) | 2.37 0.1233/0.052 (lt-tf32-fresh) ±7% |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.06 8.532/8.072 (lt-tf32-fresh) | 1.07 14.94/13.97 (cublas-tf32-fresh) | 1.82 6.719/3.694 (cublas-tf32) | 1.08 13.46/12.43 (torch-compile-tf32) | 1.49 3.852/2.587 (lt-tf32) |
| pj_m2048_n4096_k14336_tf32 | tf32 | 1.02≈ 4.357/4.264 (lt-tf32-fresh) | 1.12 8.436/7.557 (lt-tf32) | 1.94 3.568/1.838 (torch-compile-ro-tf32) | 1.11 6.915/6.226 (lt-tf32-fresh) | 1.51 1.631/1.081 (torch-compile-tf32) |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.14 1.329/1.165 (lt-tf32-fresh) | 1.00≈ 2.392/2.381 (triton-tf32) | 1.91 1.012/0.5284 (lt-tf32) | 1.12 1.998/1.786 (lt-tf32) | 1.33 0.4543/0.341 (lt-tf32-fresh) |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.14 2.018/1.768 (lt-tf32-fresh) | 1.10 3.44/3.142 (cublas-tf32-fresh) | 1.90 1.533/0.8049 (torch-compile-ro-tf32) | 1.10 2.953/2.677 (lt-tf32) | 1.26 0.7127/0.5671 (lt-tf32) |
| pj_m128_n28672_k4096_bf16 | bf16 | 1.02≈ 0.3426/0.3354 (torch-bf16) | 1.08 0.52/0.4823 (torch-bf16-compile-cg) | 1.59 0.3453/0.2171 (triton-bf16) ±18% | 1.17 0.4928/0.4198 (torch-bf16-compile) | 1.66 0.2156/0.1295 (torch-bf16-compile-cg) |
| pj_m128_n4096_k14336_bf16 | bf16 | 1.24 0.2179/0.1759 (torch-bf16) | 1.43 0.402/0.2806 (torch-bf16-compile-cg) | 2.25 0.2762/0.1229 (torch-bf16-compile-cg) | 1.24 0.305/0.2468 (torch-bf16-compile-cg) | 2.88 0.2143/0.0745 (torch-bf16) |
| pj_m128_n4096_k4096_bf16 | bf16 | 1.04≈ 0.06252/0.0602 (torch-bf16) | 1.41 0.1199/0.085 (torch-bf16-compile-cg) | 2.05 0.1089/0.0532 (triton-bf16) | 1.33 0.1018/0.0768 (torch-bf16) | 3.00 0.06544/0.0218 (torch-bf16-compile) |
| pj_m128_n6144_k4096_bf16 | bf16 | **0.95** 0.08725/0.0921 (torch-bf16) | 1.47 0.1882/0.128 (torch-bf16-compile-cg) | 2.16 0.137/0.0635 (triton-bf16) | 1.45 0.163/0.1126 (torch-bf16-compile) | 2.53 0.0773/0.0305 (torch-bf16-compile) |
| pj_m16_n28672_k4096_bf16 | bf16 | **0.78** 0.2252/0.2871 (torch-bf16) | 1.12 0.3062/0.2734 (torch-bf16-compile) | 1.21 0.2292/0.1894 (torch-bf16-compile) | 1.05 0.3705/0.3523 (torch-bf16-compile) | 2.12 0.1941/0.0916 (torch-bf16-compile-cg) |
| pj_m16_n4096_k14336_bf16 | bf16 | **0.55** 0.08632/0.1558 (torch-bf16-compile) | 1.19 0.1695/0.1423 (torch-bf16-compile-cg) | 1.21 0.1341/0.1106 (torch-bf16-compile-cg) | **0.87** 0.1741/0.2007 (torch-bf16-compile-cg) | 1.94 0.104/0.0537 (torch-bf16) |
| pj_m16_n4096_k4096_bf16 | bf16 | **0.61** 0.03629/0.0598 (torch-bf16-compile) ±8% | 1.75 0.0822/0.0471 (torch-bf16-compile) | 1.25 0.05517/0.044 (torch-bf16-compile-cg) | 1.19 0.07332/0.0614 (torch-bf16-compile-cg) | 2.48 0.05467/0.022 (torch-bf16) |
| pj_m16_n6144_k4096_bf16 | bf16 | **0.75** 0.05299/0.071 (torch-bf16) | 1.52 0.09937/0.0655 (torch-bf16-compile-cg) | 1.28 0.07574/0.0594 (torch-bf16-compile-cg) | 1.08 0.09756/0.0901 (torch-bf16-compile) | 2.10 0.05759/0.0274 (torch-bf16-compile) |
| pj_m1_n28672_k4096_bf16 | bf16 | **0.73** 0.2104/0.2879 (torch-bf16) | 1.03≈ 0.2888/0.2806 (torch-bf16-compile) | 1.43 0.263/0.1843 (torch-bf16-compile-cg) | 1.05 0.3325/0.3164 (torch-bf16-compile) | 1.31 0.1218/0.0928 (torch-bf16-compile) |
| pj_m1_n4096_k14336_bf16 | bf16 | **0.53** 0.07798/0.1466 (torch-bf16) | 1.09 0.1595/0.1464 (torch-bf16-compile-ro) | 1.30 0.1408/0.1085 (torch-bf16-compile-cg) | **0.72** 0.1298/0.1802 (torch-bf16-compile-cg) | 1.38 0.07073/0.0514 (torch-bf16-compile-cg) |
| pj_m1_n4096_k4096_bf16 | bf16 | **0.31** 0.02224/0.0714 (torch-bf16-compile) ±10% | 1.26 0.06345/0.0502 (torch-bf16-compile-ro) | 1.24 0.05572/0.0451 (torch-bf16-compile-cg) | **0.51** 0.03381/0.0666 (torch-bf16-compile-cg) ±9% | 1.12 0.02269/0.0203 (torch-bf16-compile-ro) |
| pj_m1_n6144_k4096_bf16 | bf16 | **0.35** 0.02786/0.08 (torch-bf16-compile) | 1.26 0.08134/0.0645 (torch-bf16-compile-cg) | 1.39 0.08235/0.0594 (torch-bf16-compile-cg) | **0.48** 0.04508/0.0932 (torch-bf16) | 1.30 0.03629/0.0279 (torch-bf16-compile-cg) |
| pj_m2048_n28672_k4096_bf16 | bf16 | 1.00≈ 3.987/3.976 (torch-bf16-compile) | 0.99≈ 7.057/7.106 (triton-bf16) | 1.94 3.372/1.742 (torch-bf16-compile-ro) | 1.03≈ 6.35/6.174 (torch-bf16-compile-cg) | 2.69 3.464/1.288 (torch-bf16) |
| pj_m2048_n4096_k14336_bf16 | bf16 | 1.07 2.264/2.123 (torch-bf16-compile) | **0.94** 3.961/4.223 (torch-bf16-compile) | 2.06 1.828/0.8878 (torch-bf16) | 1.04≈ 3.251/3.119 (torch-bf16-compile) | 1.93 1.307/0.6779 (torch-bf16-compile-cg) |
| pj_m2048_n4096_k4096_bf16 | bf16 | 1.10 0.6779/0.6175 (torch-bf16) | 1.10 1.151/1.044 (torch-bf16-compile-ro) | 1.94 0.5977/0.3082 (triton-bf16) | 1.05 0.9449/0.898 (torch-bf16-compile) | 2.21 0.3732/0.1689 (torch-bf16) |
| pj_m2048_n6144_k4096_bf16 | bf16 | 1.03≈ 0.9364/0.9095 (torch-bf16-compile) | 1.02≈ 1.633/1.608 (torch-bf16-compile-cg) | 1.94 0.7945/0.4106 (torch-bf16) | 1.05≈ 1.401/1.337 (torch-bf16-compile-cg) | 2.21 0.5863/0.265 (torch-bf16) |
| geomean f32 | | 0.971 (23) | 1.056 (23) | 1.125 (22) | 1.105 (23) | 1.216 (23) |
| geomean tf32 | | 1.202 (8) | 1.117 (8) | 1.876 (8) | 1.196 (8) | 1.656 (8) |
| geomean bf16 | | 0.760 (16) | 1.210 (16) | 1.595 (16) | 0.980 (16) | 1.971 (16) |
| **geomean** | | 0.927 (47) | 1.116 (47) | 1.389 (46) | 1.075 (47) | 1.511 (47) |
| win / even / loss | | 13/19/15 | 4/13/30 | 7/1/38 | 6/12/29 | 2/3/42 |

## MLP chains

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| mlp2_2048 | f32 | 1.03≈ 1.816/1.769 (chain-lt2) | **0.88** 3.252/3.694 (torch-compile) | **0.95** 3.543/3.748 (chain-streams) | **0.95** 2.375/2.501 (chain-cutlass) | 1.12 1.886/1.687 (torch-compile-ro) |
| mlp_2048 | f32 | 0.96≈ 0.8779/0.9114 (chain-lt) | **0.94** 1.618/1.728 (torch-compile) | **0.90** 1.772/1.968 (torch-compile-ro) | **0.94** 1.182/1.251 (chain-cutlass) | 1.11 0.9296/0.8394 (torch-compile-ro) |
| mlp_512 | f32 | **0.90** 0.03871/0.043 (chain-lt) | 1.56 0.06864/0.044 (triton) | 1.58 0.1054/0.0666 (torch-eager) | 1.62 0.07976/0.0492 (torch-compile-cg) ±8% | 1.50 0.04546/0.0303 (torch-compile-cg) |
| smlp_4096_256_1024 | f32 | 1.01≈ 0.157/0.1556 (chain-lt) | 1.21 0.2814/0.2324 (torch-compile) | **0.93** 0.2877/0.3082 (triton) | 1.09 0.2083/0.1915 (torch-compile-cg) | 1.42 0.1765/0.1244 (chain-lt) |
| mlp2_2048_tf32 | tf32 | 1.01≈ 1.569/1.549 (torch-compile-ro-tf32) | 1.11 2.857/2.583 (torch-compile-ro-tf32) | 1.73 1.211/0.6984 (torch-compile-ro-tf32) | 1.14 2.103/1.846 (torch-compile-ro-tf32) | 1.56 0.5719/0.367 (torch-compile-ro-tf32) |
| mlp_2048_tf32 | tf32 | 1.02≈ 0.7861/0.7697 (torch-eager-tf32) | 1.21 1.406/1.158 (torch-compile-ro-tf32) | 1.86 0.656/0.3523 (torch-compile-ro-tf32) | 1.11 1.031/0.9257 (torch-compile-ro-tf32) | 1.69 0.2924/0.1733 (torch-compile-ro-tf32) |
| geomean f32 | | 0.973 (4) | 1.117 (4) | 1.059 (4) | 1.122 (4) | 1.274 (4) |
| geomean tf32 | | 1.017 (2) | 1.159 (2) | 1.797 (2) | 1.126 (2) | 1.622 (2) |
| **geomean** | | 0.988 (6) | 1.131 (6) | 1.263 (6) | 1.123 (6) | 1.381 (6) |
| win / even / loss | | 1/5/0 | 2/0/4 | 3/0/3 | 2/0/4 | 0/0/6 |

## Attention / softmax / long rows

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| attn_1024_64 | f32 | 1.02≈ 0.03558/0.0348 (chain-fused) ±6% | 1.06 0.04884/0.0461 (torch-compile-ro) | 1.18 0.06892/0.0584 (torch-compile-ro) ±9% | 1.09 0.04562/0.042 (torch-sdpa-compile-cg) | 1.75 0.04302/0.0246 (torch-compile-cg) |
| attn_2048_128 | f32 | 0.99≈ 0.1131/0.1141 (torch-sdpa-compile) | 1.49 0.2682/0.1802 (torch-compile-cg) | 1.61 0.2923/0.1812 (torch-sdpa-compile) | 1.15 0.1707/0.1485 (torch-compile-ro) | 1.49 0.1257/0.0841 (torch-compile-ro) |
| attn_2048_64 | f32 | **0.78** 0.07282/0.0938 (torch-sdpa-compile) | **0.95** 0.1244/0.1311 (torch-compile-ro) | 1.46 0.1844/0.126 (torch-compile) | 1.21 0.1238/0.1024 (torch-sdpa-compile-cg) ±5% | 1.16 0.06664/0.0573 (torch-compile-cg) |
| causal_1024_256 | f32 | **0.30** 13.45/45.22 (torch-compile-cg) | **0.14** 25.37/179.4 (torch-compile-cg) | **0.28** 53.11/192.8 (torch-compile-cg) | **0.14** 33.74/244.1 (torch-compile-cg) | **0.18** 21.44/116.7 (torch-compile-cg) |
| decode_1024_256 | f32 | **0.53** 8.674/16.41 (chain-cache) | **0.53** 17.58/33.16 (chain-cache) | 0.86≈ 45.81/53.39 (chain-nocache) ±24% | **0.41** 21.02/51.3 (chain-cache) | 1.45 28.48/19.66 (chain-cache) |
| gen_1024_512_256 | f32 | **0.30** 6.862/23.08 (torch-compile-cg) | **0.14** 12.93/93.44 (torch-compile-cg) | **0.26** 24.18/92.24 (torch-compile-cg) | **0.14** 17.01/120.9 (torch-compile-ro) | **0.19** 10.82/56.96 (torch-compile) |
| lsm_1_131072 | f32 | **0.39** 0.0165/0.0426 (torch-eager) ±28% | 1.24 0.03164/0.0256 (torch-compile-cg) | 1.16 0.03691/0.0317 (torch-compile-ro) | 0.93≈ 0.02959/0.0317 (torch-compile-ro) ±15% | 1.44 0.02653/0.0184 (torch-sdpa-compile-cg) |
| lsm_32_8192 | f32 | **0.84** 0.0162/0.0193 (torch-sdpa) | 3.16 0.03221/0.0102 (torch-compile-cg) | 2.25 0.0392/0.0174 (triton) | 2.66 0.03271/0.0123 (torch-compile-cg) | 2.86 0.02521/0.0088 (torch-compile-cg) |
| lsm_4_32768 | f32 | **0.57** 0.01451/0.0254 (torch-sdpa) | 1.62 0.02655/0.0164 (torch-sdpa-compile-cg) | 1.51 0.03402/0.0225 (torch-compile-cg) | 1.16 0.02384/0.0205 (torch-compile-cg) | 1.56 0.02406/0.0154 (torch-sdpa-compile-cg) ±6% |
| prefill_1024_512_256 | f32 | **0.30** 6.767/22.58 (torch-compile-cg) | **0.15** 13.09/90.17 (torch-compile-cg) | **0.26** 24.54/93.03 (torch-compile-cg) | **0.14** 17.04/121 (torch-compile-ro) | **0.19** 10.9/56.43 (torch-compile-ro) |
| rowsum_2048 | f32 | 0.98≈ 0.4533/0.4628 (chain-eager) | 1.06 0.8261/0.7803 (torch-compile) | **0.73** 0.9173/1.248 (torch-compile-ro) | **0.95** 0.6164/0.6513 (torch-compile-ro) | 1.22 0.5114/0.4195 (torch-compile-cg) |
| rowsum_4096 | f32 | 1.04≈ 3.744/3.604 (chain-gemv) | **0.94** 6.181/6.568 (chain-eager) | **0.93** 6.746/7.29 (torch-eager) | 0.97≈ 4.754/4.879 (torch-compile) | 1.14 4.532/3.986 (torch-compile-ro) |
| softmax_rows_2048 | f32 | **0.95** 0.4631/0.4895 (chain-fused) | 1.15 0.8582/0.7455 (torch-compile-cg) | **0.95** 0.9513/1.002 (torch-compile) | 0.97≈ 0.636/0.6554 (torch-compile-cg) | 1.25 0.5364/0.4275 (torch-compile-cg) |
| softmax_rows_4096 | f32 | 1.03≈ 3.804/3.682 (torch-compile-ro) | **0.95** 6.317/6.672 (torch-compile) | **0.93** 6.812/7.356 (torch-compile-ro) | 0.99≈ 4.84/4.891 (torch-compile-ro) | 1.14 4.576/4.002 (torch-compile-ro) |
| **geomean** | | 0.642 (14) | 0.742 (14) | 0.853 (14) | 0.678 (14) | 0.933 (14) |
| win / even / loss | | 9/5/0 | 7/0/7 | 7/1/6 | 5/4/5 | 3/0/11 |

## LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | f32 | 0.95≈ 1.034/1.087 (torch-compile) | **0.93** 1.132/1.217 (torch-compile) | 1.14 1.041/0.9139 (torch-compile) | **0.91** 1.228/1.354 (torch-compile) | 1.01≈ 0.5671/0.5639 (torch-compile) |
| llm_blkpre_t2048 | f32 | **0.94** 25.37/26.88 (liger) | 0.96≈ 45.78/47.5 (liger) | **0.73** 37.48/51.35 (liger) | **0.93** 33.23/35.76 (liger) | 1.03≈ 29.17/28.35 (liger) |
| llm_dec_b1_l2048 | f32 | **0.84** 0.2195/0.2623 (torch-compile-ro) | **0.75** 0.2702/0.3626 (torch-compile) | **0.81** 0.2619/0.324 (torch-compile-ro) | **0.72** 0.3029/0.4224 (torch-compile) | **0.52** 0.1104/0.2109 (torch-compile-cg) |
| llm_dec_b1_l8192 | f32 | **0.81** 0.2871/0.353 (torch-compile) | **0.75** 0.3502/0.4653 (torch-compile) | **0.62** 0.2971/0.4781 (torch-compile-cg) | **0.69** 0.385/0.5601 (torch-compile) | **0.43** 0.142/0.332 (torch-compile-cg) |
| llm_dec_b32_l2048 | f32 | **0.37** 0.8501/2.291 (torch-compile) | **0.34** 0.9856/2.858 (torch-compile-ro) | **0.40** 0.7602/1.917 (torch-compile) | **0.34** 1.062/3.102 (torch-compile) | **0.34** 0.481/1.415 (torch-compile) |
| llm_dec_b32_l8192 | f32 | **0.29** 2.761/9.531 (torch-compile) | **0.29** 3.145/10.9 (torch-compile) | **0.30** 1.995/6.628 (torch-compile-cg) | **0.29** 3.21/11.18 (torch-compile) | **0.25** 1.361/5.399 (torch-compile) |
| llm_dec_b8_l2048 | f32 | **0.54** 0.3726/0.6869 (torch-compile) | **0.89** 0.8314/0.9304 (torch-compile) | **0.64** 0.4325/0.6779 (torch-compile) ±7% | **0.45** 0.4735/1.048 (torch-compile) | **0.37** 0.1894/0.5135 (torch-compile) |
| llm_dec_b8_l8192 | f32 | **0.34** 0.8465/2.481 (torch-compile) | **0.47** 1.382/2.943 (torch-compile) | **0.40** 0.7525/1.863 (torch-compile) | **0.32** 1/3.105 (torch-compile) | **0.29** 0.4194/1.452 (torch-compile) |
| llm_pre_t2048 | f32 | **0.89** 6.492/7.279 (liger) | **0.83** 10.8/12.95 (torch-sdpa-efficient) | 1.13 12.91/11.43 (liger) | **0.88** 9.142/10.35 (liger) | 1.07 7.023/6.568 (liger) |
| llm_pre_t8192 | f32 | **0.84** 44.66/52.96 (liger) | 1.14 108/94.44 (torch-sdpa-efficient) | 1.59 93.69/58.94 (liger) | 1.04≈ 81.27/78.31 (liger) | 1.28 50.43/39.47 (liger) |
| llm_blkpre_t2048_tf32 | tf32 | 1.09 19.71/18.08 (triton-tf32) | 1.06 33.21/31.44 (triton-tf32) | 1.99 17.92/8.989 (triton-tf32) | 1.11 29.03/26.18 (triton-tf32) | 1.33 8.771/6.586 (liger-tf32) |
| llm_pre_t2048_tf32 | tf32 | 1.17 5.34/4.55 (triton-tf32) | **0.89** 9.169/10.28 (torch-compile-ro-tf32) | 2.80 7.381/2.637 (triton-tf32) | 1.26 8.332/6.619 (triton-tf32) | 1.34 2.903/2.162 (triton-tf32) |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.08 0.6066/0.5613 (torch-bf16-compile) | 1.01≈ 0.7059/0.6992 (torch-bf16-compile-cg) | 1.61 0.8763/0.5455 (torch-bf16-compile) | 1.02≈ 0.8057/0.789 (torch-bf16-compile) | 1.63 0.4981/0.3052 (torch-bf16-compile-ro) |
| llm_blkpre_t2048_bf16w | bf16w | 3.06 25.21/8.23 (torch-bf16-compile-cg) | 3.02 45.11/14.95 (torch-bf16-compile) | 5.90 23.67/4.01 (torch-bf16-compile-ro) | 2.83 34.79/12.29 (torch-bf16-compile) | 6.64 18.35/2.765 (torch-bf16-compile-ro) |
| llm_dec_b1_l2048_bf16w | bf16w | **0.88** 0.1337/0.1526 (torch-bf16-compile) | **0.75** 0.1828/0.2433 (torch-bf16-compile) | **0.71** 0.1853/0.2615 (torch-bf16-compile-cg) | **0.70** 0.2146/0.3054 (torch-bf16-compile) | **0.59** 0.08976/0.153 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16w | bf16w | 1.03≈ 0.1952/0.1889 (torch-bf16-compile) | **0.84** 0.2525/0.3002 (torch-bf16-compile) | **0.83** 0.2471/0.2979 (torch-bf16-compile-cg) | **0.87** 0.2963/0.339 (torch-bf16-compile) | **0.72** 0.117/0.1621 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16w | bf16w | 1.64 0.7774/0.4738 (torch-bf16-compile) | 1.14 0.9613/0.8439 (torch-bf16-compile-cg) | 1.26 0.7075/0.5621 (torch-bf16-compile) | 1.42 0.9656/0.6795 (torch-bf16-compile) | 1.50 0.4071/0.2717 (torch-bf16-compile-ro) |
| llm_dec_b32_l8192_bf16w | bf16w | 1.80 2.704/1.505 (torch-bf16-compile) | 1.74 3.12/1.794 (torch-bf16-cudnn) | 1.54 1.955/1.269 (torch-bf16-cudnn) | 1.66 3.121/1.885 (torch-bf16-compile) | 2.21 1.282/0.58 (torch-bf16-compile-ro) |
| llm_dec_b8_l2048_bf16w | bf16w | 1.25 0.2796/0.224 (torch-bf16-compile) | 1.52 0.5248/0.3452 (torch-bf16-compile) | 1.09 0.3729/0.3425 (torch-bf16-compile) | 1.00≈ 0.385/0.3858 (torch-bf16-compile) | 0.96≈ 0.1635/0.1706 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16w | bf16w | 1.53 0.7541/0.4932 (torch-bf16-compile) | 1.32 1.076/0.8122 (torch-bf16-compile) | 1.14 0.7064/0.6208 (torch-bf16-compile-cg) | 1.32 0.9158/0.6954 (torch-bf16-compile) | 1.45 0.3919/0.2707 (torch-bf16-compile-ro) |
| llm_pre_t2048_bf16w | bf16w | 3.23 6.367/1.971 (torch-bf16-compile-cg) | 3.04 10.59/3.486 (torch-bf16-compile-cg) | 8.05 8.321/1.033 (torch-bf16-compile-cg) | 3.24 9.44/2.91 (torch-bf16-compile-cg) | 7.28 4.738/0.6509 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_bf16kv | bf16kv | 1.05≈ 0.5891/0.5613 (torch-bf16-compile) | 0.99≈ 0.6891/0.6992 (torch-bf16-compile-cg) | 1.59 0.8675/0.5455 (torch-bf16-compile) | 0.99≈ 0.7833/0.789 (torch-bf16-compile) | 1.62 0.4955/0.3052 (torch-bf16-compile-ro) |
| llm_dec_b1_l2048_bf16kv | bf16kv | **0.82** 0.1256/0.1526 (torch-bf16-compile) | **0.72** 0.1751/0.2433 (torch-bf16-compile) | **0.75** 0.197/0.2615 (torch-bf16-compile-cg) | **0.67** 0.206/0.3054 (torch-bf16-compile) | **0.57** 0.08655/0.153 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16kv | bf16kv | **0.88** 0.1656/0.1889 (torch-bf16-compile) | **0.75** 0.2248/0.3002 (torch-bf16-compile) | **0.77** 0.2305/0.2979 (torch-bf16-compile-cg) | **0.77** 0.2614/0.339 (torch-bf16-compile) | **0.70** 0.1131/0.1621 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16kv | bf16kv | 1.05≈ 0.4959/0.4738 (torch-bf16-compile) | **0.80** 0.675/0.8439 (torch-bf16-compile-cg) | 1.09 0.614/0.5621 (torch-bf16-compile) | 0.96≈ 0.6494/0.6795 (torch-bf16-compile) | 1.19 0.3233/0.2717 (torch-bf16-compile-ro) |
| llm_dec_b32_l8192_bf16kv | bf16kv | 1.10 1.657/1.505 (torch-bf16-compile) | 1.10 1.975/1.794 (torch-bf16-cudnn) | 1.25 1.592/1.269 (torch-bf16-cudnn) ±7% | 1.02≈ 1.916/1.885 (torch-bf16-compile) | 1.56 0.9065/0.58 (torch-bf16-compile-ro) |
| llm_dec_b8_l2048_bf16kv | bf16kv | **0.93** 0.2077/0.224 (torch-bf16-compile) | 1.39 0.4784/0.3452 (torch-bf16-compile) | 1.01≈ 0.3458/0.3425 (torch-bf16-compile) ±6% | **0.80** 0.3088/0.3858 (torch-bf16-compile) | **0.86** 0.1461/0.1706 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16kv | bf16kv | 1.00≈ 0.4949/0.4932 (torch-bf16-compile) | 0.99≈ 0.8026/0.8122 (torch-bf16-compile) | 0.99≈ 0.6159/0.6208 (torch-bf16-compile-cg) | **0.89** 0.6155/0.6954 (torch-bf16-compile) | 1.14 0.3083/0.2707 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_bf16 | bf16 | 0.97≈ 0.5464/0.5613 (torch-bf16-compile) | **0.92** 0.6403/0.6992 (torch-bf16-compile-cg) | 1.06 0.5803/0.5455 (torch-bf16-compile) | **0.92** 0.729/0.789 (torch-bf16-compile) | **0.92** 0.2812/0.3052 (torch-bf16-compile-ro) |
| llm_blkpre_t2048_bf16 | bf16 | 1.17 9.596/8.23 (torch-bf16-compile-cg) | 1.08 16.17/14.95 (torch-bf16-compile) | 2.12 8.483/4.01 (torch-bf16-compile-ro) | 1.12 13.75/12.29 (torch-bf16-compile) | 2.24 6.2/2.765 (torch-bf16-compile-ro) |
| llm_dec_b1_l2048_bf16 | bf16 | **0.82** 0.1249/0.1526 (torch-bf16-compile) | **0.73** 0.1782/0.2433 (torch-bf16-compile) | **0.70** 0.1821/0.2615 (torch-bf16-compile-cg) | **0.66** 0.2002/0.3054 (torch-bf16-compile) | **0.55** 0.08362/0.153 (torch-bf16-compile-ro) |
| llm_dec_b1_l8192_bf16 | bf16 | **0.87** 0.1646/0.1889 (torch-bf16-compile) | **0.74** 0.223/0.3002 (torch-bf16-compile) | **0.76** 0.2265/0.2979 (torch-bf16-compile-cg) | **0.76** 0.257/0.339 (torch-bf16-compile) | **0.70** 0.1128/0.1621 (torch-bf16-compile-ro) |
| llm_dec_b32_l2048_bf16 | bf16 | 1.09 0.5164/0.4738 (torch-bf16-compile) | **0.80** 0.6747/0.8439 (torch-bf16-compile-cg) | 1.01≈ 0.5692/0.5621 (torch-bf16-compile) | 0.96≈ 0.6513/0.6795 (torch-bf16-compile) | 1.07 0.2919/0.2717 (torch-bf16-compile-ro) |
| llm_dec_b32_l8192_bf16 | bf16 | 1.11 1.672/1.505 (torch-bf16-compile) | 1.11 1.991/1.794 (torch-bf16-cudnn) | 1.29 1.643/1.269 (torch-bf16-cudnn) | 1.03≈ 1.949/1.885 (torch-bf16-compile) | 1.51 0.875/0.58 (torch-bf16-compile-ro) |
| llm_dec_b8_l2048_bf16 | bf16 | **0.92** 0.207/0.224 (torch-bf16-compile) | 1.35 0.4669/0.3452 (torch-bf16-compile) | **0.94** 0.3205/0.3425 (torch-bf16-compile) | **0.81** 0.3125/0.3858 (torch-bf16-compile) | **0.85** 0.1447/0.1706 (torch-bf16-compile-cg) |
| llm_dec_b8_l8192_bf16 | bf16 | 1.00≈ 0.4941/0.4932 (torch-bf16-compile) | 0.98≈ 0.7979/0.8122 (torch-bf16-compile) | 1.01≈ 0.6285/0.6208 (torch-bf16-compile-cg) | **0.90** 0.6241/0.6954 (torch-bf16-compile) | 1.13 0.3063/0.2707 (torch-bf16-compile-ro) |
| llm_pre_t2048_bf16 | bf16 | 1.33 2.627/1.971 (torch-bf16-compile-cg) | 1.34 4.655/3.486 (torch-bf16-compile-cg) | 2.62 2.707/1.033 (torch-bf16-compile-cg) | 1.27 3.69/2.91 (torch-bf16-compile-cg) | 2.71 1.764/0.6509 (torch-bf16-compile-ro) |
| llm_blkdec_b1_l4096_f8w | f8w | 0.3362 (no leg) | 0.5482 (no leg) | 0.5678 (no leg) | 0.4534 (no leg) | 0.2112 (no leg) |
| llm_dec_b1_l2048_f8w | f8w | 0.05436 (no leg) | 0.1607 (no leg) | 0.1895 (no leg) | 0.1019 (no leg) | 0.0797 (no leg) |
| llm_dec_b1_l8192_f8w | f8w | 0.1543 (no leg) | 0.2301 (no leg) | 0.2395 (no leg) | 0.2317 (no leg) | 0.1057 (no leg) |
| llm_dec_b32_l2048_f8w | f8w | 0.7509 (no leg) | 0.9826 (no leg) | 0.8763 (no leg) | 0.9651 (no leg) | 0.442 (no leg) |
| llm_dec_b32_l8192_f8w | f8w | 2.666 (no leg) | 3.135 (no leg) | 2.082 (no leg) | 3.107 (no leg) | 1.313 (no leg) |
| llm_dec_b8_l2048_f8w | f8w | 0.2358 (no leg) | 0.8136 (no leg) | 0.3927 (no leg) | 0.338 (no leg) | 0.1383 (no leg) |
| llm_dec_b8_l8192_f8w | f8w | 0.7094 (no leg) | 1.361 (no leg) | 0.7336 (no leg) | 0.8757 (no leg) | 0.3671 (no leg) |
| geomean f32 | | 0.625 (10) | 0.676 (10) | 0.686 (10) | 0.594 (10) | 0.557 (10) |
| geomean tf32 | | 1.131 (2) | 0.971 (2) | 2.362 (2) | 1.181 (2) | 1.337 (2) |
| geomean bf16w | | 1.559 (9) | 1.419 (9) | 1.682 (9) | 1.377 (9) | 1.749 (9) |
| geomean bf16kv | | 0.971 (7) | 0.938 (7) | 1.035 (7) | 0.863 (7) | 1.022 (7) |
| geomean bf16 | | 1.021 (9) | 0.982 (9) | 1.163 (9) | 0.919 (9) | 1.142 (9) |
| **geomean** | | 0.987 (37) | 0.962 (37) | 1.121 (37) | 0.903 (37) | 1.031 (37) |
| win / even / loss | | 16/7/14 | 18/5/14 | 14/4/19 | 20/8/9 | 15/3/19 |

## Conv / FIR / elementwise

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| conv_2048_5 | f32 | **0.43** 0.06647/0.1563 (torch-eager) | **0.71** 0.1139/0.1608 (torch-compile-cg) | **0.71** 0.09762/0.1382 (torch-compile-cg) | **0.60** 0.1009/0.1679 (torch-compile-cg) | **0.45** 0.04371/0.0972 (torch-eager) |
| convmc_8_64_56_3 | f32 | **0.67** 0.08205/0.1222 (torch-eager) | 1.93 0.1501/0.0778 (torch-eager) | 3.07 0.2707/0.0881 (torch-eager) ±8% | 2.05 0.1867/0.0911 (torch-eager) | 1.71 0.1193/0.0697 (torch-eager) |
| ew_4096 | f32 | **0.36** 0.08805/0.2478 (chain-fused) | **0.61** 0.1466/0.2406 (torch-compile) | 1.23 0.193/0.1567 (torch-compile) | **0.60** 0.156/0.2601 (chain-fused) | 1.32 0.09409/0.0712 (torch-compile) ±15% |
| ewconvew_2048_5 | f32 | **0.37** 0.07822/0.2092 (torch-compile) | **0.53** 0.131/0.2458 (torch-compile) | **0.60** 0.1167/0.1956 (torch-eager) | **0.55** 0.1194/0.2181 (torch-compile-cg) | **0.47** 0.05485/0.1162 (torch-compile-cg) |
| fir_4194304_16 | f32 | **0.42** 0.04549/0.1083 (torch-eager) | **0.69** 0.07618/0.1097 (torch-compile-ro) | **0.79** 0.06793/0.086 (torch-compile) | **0.65** 0.07014/0.1075 (torch-eager) | **0.53** 0.03159/0.0595 (torch-eager) |
| firew_rowsum_2048x2048_16 | f32 | **0.40** 0.05164/0.1294 (torch-compile) | **0.75** 0.1042/0.1393 (torch-compile-cg) | **0.70** 0.08268/0.1178 (torch-compile-cg) | **0.70** 0.08167/0.1167 (torch-compile-cg) | **0.59** 0.03816/0.0644 (torch-compile-ro) |
| convmc_8_64_56_3_tf32 | tf32 | 1.12 0.08025/0.0719 (torch-eager-tf32) | 1.76 0.1534/0.087 (torch-eager-tf32) | 5.17 0.2485/0.0481 (torch-eager-tf32) | 2.20 0.1868/0.085 (torch-eager-tf32) | 3.38 0.1187/0.0351 (torch-eager-tf32) |
| geomean f32 | | 0.430 (6) | 0.783 (6) | 0.980 (6) | 0.755 (6) | 0.730 (6) |
| geomean tf32 | | 1.116 (1) | 1.764 (1) | 5.166 (1) | 2.198 (1) | 3.382 (1) |
| **geomean** | | 0.493 (7) | 0.879 (7) | 1.242 (7) | 0.879 (7) | 0.908 (7) |
| win / even / loss | | 6/0/1 | 5/0/2 | 4/0/3 | 5/0/2 | 4/0/3 |

## Fan-in/out, loops, CPU+GPU

| workload | prec | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|---|
| diamond_2048 | f32 | 0.99≈ 1.369/1.378 (chain-lt) | **0.90** 2.419/2.678 (chain-streams) | **0.92** 2.701/2.939 (chain-lt) | 0.95≈ 1.79/1.877 (chain-cutlass) | 1.14 1.443/1.269 (torch-eager) |
| fanout_2048 | f32 | 1.00≈ 0.9136/0.9114 (chain-streams) | **0.92** 1.653/1.805 (chain-streams) | **0.91** 1.809/1.99 (torch-compile-cg) | **0.94** 1.214/1.285 (chain-streams) | 1.15 0.9712/0.8438 (torch-compile-ro) |
| indep_2048 | f32 | 1.02≈ 0.9372/0.9155 (chain-lt) | **0.92** 1.629/1.77 (chain-streams) | **0.90** 1.758/1.958 (torch-compile-cg) | 0.96≈ 1.207/1.253 (chain-cutlass) | 1.16 0.9733/0.8421 (torch-compile-cg) |
| indep_512 | f32 | 1.08 0.03936/0.0365 (chain-streams) | 1.59 0.07504/0.0471 (torch-compile) | 1.45 0.0981/0.0676 (torch-eager) | 1.54 0.07232/0.0471 (torch-compile) | 1.83 0.04893/0.0268 (torch-compile) |
| loop_4096 | f32 | **0.94** 26.9/28.51 (chain-lt) | 1.22 63.56/52.03 (torch-compile-cg) | **0.90** 52.81/58.44 (torch-compile-ro) | **0.94** 36.33/38.66 (chain-lt) | 1.04≈ 32.16/30.9 (chain-lt) |
| loopmm_1024_8 | f32 | 1.00≈ 0.6093/0.6106 (torch-compile) | 1.20 1.11/0.9277 (torch-compile-cg) | 1.18 1.234/1.044 (torch-eager) | 1.18 0.8405/0.7127 (torch-compile-cg) | 1.26 0.6196/0.4936 (torch-compile-cg) |
| rt_mlp2_1024 | f32 | 0.99≈ 2.554/2.575 (torch-compile) | **0.90** 4.348/4.831 (torch-compile) | 1.26 5.485/4.347 (torch-compile-ro) | 1.12 3.372/2.998 (torch-compile-cg) | 1.18 2.684/2.268 (torch-compile-cg) |
| rt_mlp2_2048 | f32 | **0.93** 14.66/15.69 (torch-compile-cg) | **0.87** 26.39/30.45 (torch-compile-cg) | **0.90** 28.51/31.52 (torch-compile-cg) | **0.91** 19.17/21 (torch-compile) | **0.92** 15.09/16.37 (torch-compile-cg) |
| twin_2048 | f32 | 0.97≈ 0.9203/0.9463 (torch-compile) | **0.91** 1.587/1.738 (torch-compile) | **0.90** 1.748/1.952 (torch-eager) | **0.94** 1.2/1.277 (torch-compile-cg) | 1.16 0.9686/0.8378 (torch-compile) |
| **geomean** | | 0.992 (9) | 1.026 (9) | 1.019 (9) | 1.040 (9) | 1.183 (9) |
| win / even / loss | | 2/6/1 | 6/0/3 | 6/0/3 | 4/2/3 | 1/1/7 |

## bf16 real-world reference

The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal's f32 values) next to Mapal's own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV cache to bf16 too.

| workload | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| pj_m128_n28672_k4096 | 0.3354 (bf16) · f32 2.39 | 0.4823 (bf16-compile-cg) · f32 2.49 | 0.2171 (triton-bf16) · f32 7.45 | 0.4198 (bf16-compile) · f32 2.63 | 0.1295 (bf16-compile-cg) · f32 7.49 |
| pj_m128_n4096_k14336 | 0.1759 (bf16) · f32 2.53 | 0.2806 (bf16-compile-cg) · f32 2.51 | 0.1229 (bf16-compile-cg) · f32 8.96 | 0.2468 (bf16-compile-cg) · f32 2.60 | 0.0745 (bf16) · f32 6.79 |
| pj_m128_n4096_k4096 | 0.0602 (bf16) · f32 2.27 | 0.085 (bf16-compile-cg) · f32 2.88 | 0.0532 (triton-bf16) · f32 6.56 | 0.0768 (bf16) · f32 3.08 | 0.0218 (bf16-compile) · f32 7.30 |
| pj_m128_n6144_k4096 | 0.0921 (bf16) · f32 2.12 | 0.128 (bf16-compile-cg) · f32 2.64 | 0.0635 (triton-bf16) · f32 10.67 | 0.1126 (bf16-compile) · f32 2.98 | 0.0305 (bf16-compile) · f32 7.50 |
| pj_m16_n18432_k7168 | 0.3156 (bf16) · f32 1.89 | 0.3072 (bf16-compile) · f32 2.08 | 0.212 (triton-bf16) · f32 2.23 | 0.3881 (bf16-compile) · f32 1.92 | 0.1008 (bf16-compile-ro) · f32 3.17 |
| pj_m16_n2048_k7168 | 0.0567 (bf16-compile) · f32 0.61 | 0.042 (bf16-compile-cg) · f32 2.51 | 0.044 (bf16-compile-cg) · f32 2.85 | 0.0543 (bf16-compile-cg) · f32 1.48 | 0.0199 (bf16-compile) · f32 2.98 |
| pj_m16_n28672_k4096 | 0.2871 (bf16) · f32 1.87 | 0.2734 (bf16-compile) · f32 2.13 | 0.1894 (bf16-compile) · f32 2.51 | 0.3523 (bf16-compile) · f32 1.87 | 0.0916 (bf16-compile-cg) · f32 3.00 |
| pj_m16_n4096_k14336 | 0.1558 (bf16-compile) · f32 1.50 | 0.1423 (bf16-compile-cg) · f32 2.09 | 0.1106 (bf16-compile-cg) · f32 2.45 | 0.2007 (bf16-compile-cg) · f32 1.84 | 0.0537 (bf16) · f32 2.97 |
| pj_m16_n4096_k4096 | 0.0598 (bf16-compile) · f32 0.65 | 0.0471 (bf16-compile) · f32 2.56 | 0.044 (bf16-compile-cg) · f32 2.70 | 0.0614 (bf16-compile-cg) · f32 1.34 | 0.022 (bf16) · f32 2.41 |
| pj_m16_n6144_k4096 | 0.071 (bf16) · f32 1.21 | 0.0655 (bf16-compile-cg) · f32 2.35 | 0.0594 (bf16-compile-cg) · f32 2.40 | 0.0901 (bf16-compile) · f32 1.75 | 0.0274 (bf16-compile) · f32 2.54 |
| pj_m1_n18432_k7168 | 0.3136 (bf16) · f32 1.87 | 0.3215 (bf16-compile-cg) · f32 1.91 | 0.213 (bf16-compile-cg) · f32 2.01 | 0.3492 (bf16-compile) · f32 2.08 | 0.1007 (bf16-compile-ro) · f32 1.98 |
| pj_m1_n2048_k7168 | 0.0515 (bf16) · f32 0.47 | 0.042 (bf16-compile-cg) · f32 2.06 | 0.0389 (bf16-compile-cg) · f32 2.23 | 0.0594 (bf16-compile-cg) · f32 0.86 | 0.019 (bf16-compile-cg) · f32 1.99 |
| pj_m1_n28672_k4096 | 0.2879 (bf16) · f32 1.81 | 0.2806 (bf16-compile) · f32 1.96 | 0.1843 (bf16-compile-cg) · f32 2.13 | 0.3164 (bf16-compile) · f32 2.07 | 0.0928 (bf16-compile) · f32 1.95 |
| pj_m1_n4096_k14336 | 0.1466 (bf16) · f32 1.53 | 0.1464 (bf16-compile-ro) · f32 1.95 | 0.1085 (bf16-compile-cg) · f32 1.97 | 0.1802 (bf16-compile-cg) · f32 1.86 | 0.0514 (bf16-compile-cg) · f32 1.96 |
| pj_m1_n4096_k4096 | 0.0714 (bf16-compile) · f32 0.42 | 0.0502 (bf16-compile-ro) · f32 1.93 | 0.0451 (bf16-compile-cg) · f32 2.01 | 0.0666 (bf16-compile-cg) · f32 0.69 | 0.0203 (bf16-compile-ro) · f32 2.15 |
| pj_m1_n6144_k4096 | 0.08 (bf16-compile) · f32 0.86 | 0.0645 (bf16-compile-cg) · f32 2.10 | 0.0594 (bf16-compile-cg) · f32 1.90 | 0.0932 (bf16) · f32 1.08 | 0.0279 (bf16-compile-cg) · f32 1.94 |
| pj_m2048_n28672_k4096 | 4.061 (bf16-compile-cg) · f32 2.91 | 7.106 (triton-bf16) · f32 2.72 | 1.742 (bf16-compile-ro) · f32 13.33 | 6.174 (bf16-compile-cg) · f32 2.66 | 1.288 (bf16) · f32 12.54 |
| pj_m2048_n4096_k14336 | 2.123 (bf16-compile) · f32 3.02 | 4.223 (bf16-compile) · f32 2.61 | 0.8878 (bf16) · f32 13.25 | 3.119 (bf16-compile) · f32 2.59 | 0.6779 (bf16-compile-cg) · f32 12.46 |
| pj_m2048_n4096_k4096 | 0.6175 (bf16) · f32 2.97 | 1.044 (bf16-compile-ro) · f32 2.75 | 0.3082 (triton-bf16) · f32 10.94 | 0.898 (bf16-compile) · f32 2.61 | 0.1689 (bf16) · f32 14.48 |
| pj_m2048_n6144_k4096 | 0.9095 (bf16-compile) · f32 2.91 | 1.608 (bf16-compile-cg) · f32 2.85 | 0.4106 (bf16) · f32 12.41 | 1.337 (bf16-compile-cg) · f32 2.62 | 0.265 (bf16) · f32 10.30 |
| llm_blkdec_b1_l4096 | 0.5619 (bf16-compile) · f32 1.84 · bf16w 1.08 | 0.6977 (bf16-compile) · f32 1.62 · bf16w 1.01 | 0.5439 (bf16-compile-cg) · f32 1.91 · bf16w 1.61 | 0.791 (bf16-compile) · f32 1.55 · bf16w 1.02 | 0.3113 (bf16-compile-ro) · f32 1.82 · bf16w 1.60 |
| llm_blkpre_t2048 | 8.23 (bf16-compile-cg) · f32 3.08 · bf16w 3.06 | 14.95 (bf16-compile) · f32 3.06 · bf16w 3.02 | 4.01 (bf16-compile-ro) · f32 9.35 · bf16w 5.90 | 12.29 (bf16-compile) · f32 2.70 · bf16w 2.83 | 2.765 (bf16-compile-ro) · f32 10.55 · bf16w 6.64 |
| llm_dec_b1_l2048 | 0.1474 (bf16-compile) · f32 1.49 · bf16w 0.91 | 0.2661 (bf16-compile) · f32 1.02 · bf16w 0.69 | 0.2474 (bf16-compile-cg) · f32 1.06 · bf16w 0.75 | 0.3089 (bf16-compile) · f32 0.98 · bf16w 0.69 | 0.1551 (bf16-compile-ro) · f32 0.71 · bf16w 0.58 |
| llm_dec_b1_l8192 | 0.192 (bf16-compile) · f32 1.50 · bf16w 1.02 | 0.3066 (bf16-compile) · f32 1.14 · bf16w 0.82 | 0.2814 (bf16-compile) · f32 1.06 · bf16w 0.88 | 0.3499 (bf16-compile) · f32 1.10 · bf16w 0.85 | 0.1676 (bf16-compile-ro) · f32 0.85 · bf16w 0.70 |
| llm_dec_b32_l2048 | 0.4698 (bf16-compile) · f32 1.81 · bf16w 1.65 | 0.8172 (bf16-compile) · f32 1.21 · bf16w 1.18 | 0.5573 (bf16-compile) · f32 1.36 · bf16w 1.27 | 0.6732 (bf16-compile) · f32 1.58 · bf16w 1.43 | 0.2781 (bf16-compile-ro) · f32 1.73 · bf16w 1.46 |
| llm_dec_b32_l8192 | 1.521 (bf16-compile-cg) · f32 1.82 · bf16w 1.78 | 1.809 (bf16-cudnn) · f32 1.74 · bf16w 1.72 | 1.277 (bf16-cudnn) · f32 1.56 · bf16w 1.53 | 1.87 (bf16-compile) · f32 1.72 · bf16w 1.67 | 0.5805 (bf16-compile-ro) · f32 2.35 · bf16w 2.21 |
| llm_dec_b8_l2048 | 0.2293 (bf16-compile) · f32 1.63 · bf16w 1.22 | 0.3526 (bf16-compile) · f32 2.36 · bf16w 1.49 | 0.2965 (bf16-compile) · f32 1.46 · bf16w 1.26 | 0.3917 (bf16-compile) · f32 1.21 · bf16w 0.98 | 0.1755 (bf16-compile-cg) · f32 1.08 · bf16w 0.93 |
| llm_dec_b8_l8192 | 0.5005 (bf16-compile) · f32 1.69 · bf16w 1.51 | 0.8087 (bf16-compile) · f32 1.71 · bf16w 1.33 | 0.5845 (bf16-compile) · f32 1.29 · bf16w 1.21 | 0.6917 (bf16-compile) · f32 1.45 · bf16w 1.32 | 0.2866 (bf16-compile-cg) · f32 1.46 · bf16w 1.37 |
| llm_pre_t2048 | 1.971 (bf16-compile-cg) · f32 3.29 · bf16w 3.23 | 3.486 (bf16-compile-cg) · f32 3.10 · bf16w 3.04 | 1.033 (bf16-compile-cg) · f32 12.50 · bf16w 8.05 | 2.91 (bf16-compile-cg) · f32 3.14 · bf16w 3.24 | 0.6509 (bf16-compile-ro) · f32 10.79 · bf16w 7.28 |
| llm_pre_t8192 | 11.44 (bf16-compile) · f32 3.91 | 20.18 (bf16-compile-cg) · f32 5.35 | 5.879 (bf16-compile-cg) · f32 15.94 | 17.16 (bf16-compile) · f32 4.74 | 4.148 (bf16-compile-ro) · f32 12.16 |

## Verdict flips vs the mixed-precision gap tables

gap.py's rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables (gap-<part>.md, gap.md) row for row. Rows a part's gap table did not cover (the RTX 5080's model-scale rows, reported in model-scale.md) are judged against that rule applied to the same data.

| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |
|---|---|---|---|---|
| rtx-5080 | llm_blkdec_b1_l4096 | 0.5619 (torch-bf16-compile) 1.84 loss | 1.087 (torch-compile) 0.95 even | old best a bf16 leg |
| rtx-5080 | llm_blkpre_t2048 | 8.232 (torch-bf16-compile) 3.08 loss | 26.88 (liger) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_dec_b1_l2048 | 0.1474 (torch-bf16-compile) 1.49 loss | 0.2623 (torch-compile-ro) 0.84 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b1_l8192 | 0.192 (torch-bf16-compile) 1.50 loss | 0.353 (torch-compile) 0.81 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b32_l2048 | 0.4698 (torch-bf16-compile) 1.81 loss | 2.291 (torch-compile) 0.37 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l2048 | 0.2293 (torch-bf16-compile) 1.63 loss | 0.6869 (torch-compile) 0.54 win | old best a bf16 leg |
| rtx-5080 | llm_dec_b8_l8192 | 0.5005 (torch-bf16-compile) 1.69 loss | 2.481 (torch-compile) 0.34 win | old best a bf16 leg |
| rtx-5080 | llm_pre_t2048 | 1.972 (torch-bf16-compile) 3.29 loss | 7.279 (liger) 0.89 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | llm_pre_t2048_tf32 | 6.171 (torch-compile-tf32) 0.87 win | 4.55 (triton-tf32) 1.17 loss | best now a gemm_lib leg |
| rtx-5080 | llm_pre_t8192 | 11.44 (torch-bf16-compile) 3.91 loss | 52.96 (liger) 0.84 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m128_n4096_k4096 | 0.0602 (torch-bf16) 2.27 loss | 0.1383 (lt-fresh) 0.99 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m128_n6144_k4096_tf32 | 0.1895 (torch-compile-tf32) 0.95 win | 0.1304 (lt-tf32-fresh) 1.38 loss | best now a gemm_lib leg |
| rtx-5080 | pj_m16_n18432_k7168 | 0.3156 (torch-bf16) 1.89 loss | 0.5904 (lt-fresh) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n28672_k4096 | 0.2871 (torch-bf16) 1.87 loss | 0.528 (lt-fresh) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m16_n4096_k14336 | 0.1558 (torch-bf16-compile) 1.50 loss | 0.294 (torch-compile) 0.80 win | old best a bf16 leg |
| rtx-5080 | pj_m16_n6144_k4096 | 0.071 (torch-bf16) 1.21 loss | 0.0899 (lt-fresh) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n18432_k7168 | 0.3136 (torch-bf16) 1.87 loss | 0.6173 (cublas-fresh) 0.95 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n28672_k4096 | 0.2879 (torch-bf16) 1.81 loss | 0.5063 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n4096_k14336 | 0.1466 (torch-bf16) 1.53 loss | 0.2342 (lt-fresh) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n4096_k4096 | 0.0714 (torch-bf16-compile) 0.42 win | 0.0238 (lt-fresh) 1.26 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m1_n6144_k4096 | 0.08 (torch-bf16-compile) 0.86 win | 0.0539 (lt-fresh) 1.28 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096 | 4.062 (torch-bf16-compile) 2.91 loss | 12.23 (lt-fresh) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n28672_k4096_tf32 | 8.172 (torch-eager-tf32) 1.04 even | 8.072 (lt-tf32-fresh) 1.06 loss | best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k14336 | 2.123 (torch-bf16-compile) 3.02 loss | 6.434 (cublas) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n4096_k4096 | 0.6175 (torch-bf16) 2.97 loss | 1.798 (lt-fresh) 1.02 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-5080 | pj_m2048_n6144_k4096 | 0.9095 (torch-bf16-compile) 2.91 loss | 2.652 (lt-fresh) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | attn_1024_64 | 0.0584 (chain-fused) 0.84 win | 0.0461 (torch-compile-ro) 1.06 loss | other |
| rtx-3090 | llm_blkdec_b1_l4096 | 0.6977 (torch-bf16-compile) 1.62 loss | 1.217 (torch-compile) 0.93 win | old best a bf16 leg |
| rtx-3090 | llm_blkpre_t2048 | 14.95 (torch-bf16-compile) 3.06 loss | 47.5 (liger) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | llm_blkpre_t2048_tf32 | 37.74 (torch-eager-tf32) 0.88 win | 31.44 (triton-tf32) 1.06 loss | best now a gemm_lib leg |
| rtx-3090 | llm_dec_b1_l2048 | 0.2661 (torch-bf16-compile) 1.02 even | 0.3626 (torch-compile) 0.75 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b1_l8192 | 0.3066 (torch-bf16-compile) 1.14 loss | 0.4653 (torch-compile) 0.75 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l2048 | 0.8172 (torch-bf16-compile) 1.21 loss | 2.858 (torch-compile-ro) 0.34 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192 | 3.076 (torch-bf16-compile) 1.02 even | 10.9 (torch-compile) 0.29 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b32_l8192_bf16w | 3.046 (torch-bf16-compile) 1.02 even | 1.794 (torch-bf16-cudnn) 1.74 loss | other |
| rtx-3090 | llm_dec_b8_l2048 | 0.3526 (torch-bf16-compile) 2.36 loss | 0.9304 (torch-compile) 0.89 win | old best a bf16 leg |
| rtx-3090 | llm_dec_b8_l8192 | 0.8087 (torch-bf16-compile) 1.71 loss | 2.943 (torch-compile) 0.47 win | old best a bf16 leg |
| rtx-3090 | llm_pre_t2048 | 3.506 (torch-bf16-compile) 3.08 loss | 12.95 (torch-sdpa-efficient) 0.83 win | old best a bf16 leg |
| rtx-3090 | lsm_1_131072 | 0.0307 (torch-compile) 1.03 even | 0.0256 (torch-compile-cg) 1.24 loss | other |
| rtx-3090 | mlp2_2048_tf32 | 2.746 (torch-eager-tf32) 1.04 even | 2.583 (torch-compile-ro-tf32) 1.11 loss | other |
| rtx-3090 | mlp_2048_tf32 | 1.411 (torch-eager-tf32) 1.00 even | 1.158 (torch-compile-ro-tf32) 1.21 loss | other |
| rtx-3090 | pj_m128_n28672_k4096 | 0.4833 (torch-bf16-compile) 2.49 loss | 1.223 (torch-compile-ro) 0.98 even | old best a bf16 leg |
| rtx-3090 | pj_m128_n4096_k14336 | 0.2807 (torch-bf16-compile) 2.51 loss | 0.684 (torch-compile) 1.03 even | old best a bf16 leg |
| rtx-3090 | pj_m128_n6144_k4096_tf32 | 0.2847 (torch-compile-tf32) 1.02 even | 0.2499 (lt-tf32) 1.16 loss | best now a gemm_lib leg |
| rtx-3090 | pj_m16_n18432_k7168 | 0.3072 (torch-bf16-compile) 2.08 loss | 0.6124 (torch-compile) 1.04 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n18432_k7168 | 0.3226 (torch-bf16-compile) 1.90 loss | 0.6083 (lt) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m1_n28672_k4096 | 0.2806 (torch-bf16-compile) 1.96 loss | 0.5366 (torch-compile-ro) 1.02 even | old best a bf16 leg |
| rtx-3090 | pj_m1_n4096_k14336 | 0.1495 (torch-bf16) 1.91 loss | 0.2755 (torch-compile-ro) 1.04 even | old best a bf16 leg |
| rtx-3090 | pj_m2048_n28672_k4096 | 7.424 (torch-bf16) 2.60 loss | 21.53 (lt-fresh) 0.90 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n28672_k4096_tf32 | 16.09 (torch-eager-tf32) 0.93 win | 13.97 (cublas-tf32-fresh) 1.07 loss | best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k14336 | 4.223 (torch-bf16-compile) 2.61 loss | 11.24 (lt-fresh) 0.98 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n4096_k4096 | 1.101 (torch-bf16-compile) 2.61 loss | 3.255 (cublas) 0.88 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-3090 | pj_m2048_n6144_k4096 | 1.671 (torch-bf16-compile) 2.74 loss | 4.699 (cublas-fresh) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| a100 | attn_1024_64 | 0.0717 (torch-compile) 0.96 even | 0.0584 (torch-compile-ro) 1.18 loss | other |
| a100 | llm_blkpre_t2048 | 4.615 (torch-bf16) 8.12 loss | 51.35 (liger) 0.73 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | llm_dec_b1_l8192 | 0.2814 (torch-bf16-compile) 1.06 loss | 0.4781 (torch-compile-cg) 0.62 win | old best a bf16 leg |
| a100 | llm_dec_b32_l2048 | 0.5573 (torch-bf16-compile) 1.36 loss | 1.917 (torch-compile) 0.40 win | old best a bf16 leg |
| a100 | llm_dec_b32_l8192 | 1.458 (torch-bf16) 1.37 loss | 6.628 (torch-compile-cg) 0.30 win | old best a bf16 leg |
| a100 | llm_dec_b8_l2048 | 0.2965 (torch-bf16-compile) 1.46 loss | 0.6779 (torch-compile) 0.64 win | old best a bf16 leg |
| a100 | llm_dec_b8_l8192 | 0.5845 (torch-bf16-compile) 1.29 loss | 1.863 (torch-compile) 0.40 win | old best a bf16 leg |
| a100 | lsm_1_131072 | 0.044 (torch-compile) 0.84 win | 0.0317 (torch-compile-ro) 1.16 loss | other |
| a100 | lsm_4_32768 | 0.0348 (torch-compile) 0.98 even | 0.0225 (torch-compile-cg) 1.51 loss | other |
| a100 | pj_m128_n28672_k4096 | 0.2468 (torch-bf16-compile) 6.55 loss | 1.739 (cublas) 0.93 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | pj_m16_n4096_k14336 | 0.1126 (torch-bf16-compile) 2.40 loss | 0.3103 (torch-compile-cg) 0.87 win | old best a bf16 leg |
| a100 | pj_m2048_n28672_k4096 | 1.76 (torch-bf16) 13.19 loss | 25.48 (torch-compile) 0.91 win | old best a bf16 leg |
| a100 | pj_m2048_n4096_k14336 | 0.8878 (torch-bf16) 13.25 loss | 12.65 (torch-compile-ro) 0.93 win | old best a bf16 leg |
| a100 | pj_m2048_n4096_k4096 | 0.3092 (torch-bf16) 10.91 loss | 3.692 (torch-compile-cg) 0.91 win | old best a bf16 leg |
| a100 | pj_m2048_n6144_k4096 | 0.4106 (torch-bf16) 12.41 loss | 5.711 (lt) 0.89 win | old best a bf16 leg + best now a gemm_lib leg |
| a100 | rt_mlp2_1024 | 5.471 (torch-eager) 1.00 even | 4.347 (torch-compile-ro) 1.26 loss | other |
| rtx-4090 | attn_1024_64 | 0.0471 (torch-compile) 0.97 even | 0.042 (torch-sdpa-compile-cg) 1.09 loss | other |
| rtx-4090 | llm_blkdec_b1_l4096 | 0.791 (torch-bf16-compile) 1.55 loss | 1.354 (torch-compile) 0.91 win | old best a bf16 leg |
| rtx-4090 | llm_blkpre_t2048 | 12.29 (torch-bf16-compile) 2.70 loss | 35.76 (liger) 0.93 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_dec_b1_l2048 | 0.3089 (torch-bf16-compile) 0.98 even | 0.4224 (torch-compile) 0.72 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b1_l8192 | 0.3499 (torch-bf16-compile) 1.10 loss | 0.5601 (torch-compile) 0.69 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l2048 | 0.6732 (torch-bf16-compile) 1.58 loss | 3.102 (torch-compile) 0.34 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b32_l8192 | 1.87 (torch-bf16-compile) 1.72 loss | 11.18 (torch-compile) 0.29 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l2048 | 0.3917 (torch-bf16-compile) 1.21 loss | 1.048 (torch-compile) 0.45 win | old best a bf16 leg |
| rtx-4090 | llm_dec_b8_l8192 | 0.6917 (torch-bf16-compile) 1.45 loss | 3.105 (torch-compile) 0.32 win | old best a bf16 leg |
| rtx-4090 | llm_pre_t2048 | 2.911 (torch-bf16-compile) 3.14 loss | 10.35 (liger) 0.88 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | llm_pre_t2048_tf32 | 8.056 (torch-compile-tf32) 1.03 even | 6.619 (triton-tf32) 1.26 loss | best now a gemm_lib leg |
| rtx-4090 | llm_pre_t8192 | 17.16 (torch-bf16-compile) 4.74 loss | 78.31 (liger) 1.04 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | lsm_1_131072 | 0.0389 (torch-compile) 0.76 win | 0.0317 (torch-compile-ro) 0.93 even | other |
| rtx-4090 | lsm_4_32768 | 0.0256 (torch-compile) 0.93 win | 0.0205 (torch-compile-cg) 1.16 loss | other |
| rtx-4090 | pj_m128_n28672_k4096 | 0.4198 (torch-bf16-compile) 2.63 loss | 1.18 (lt) 0.94 win | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m128_n4096_k14336 | 0.2478 (torch-bf16) 2.59 loss | 0.64 (torch-eager) 1.00 even | old best a bf16 leg |
| rtx-4090 | pj_m16_n18432_k7168 | 0.3881 (torch-bf16-compile) 1.92 loss | 0.7127 (torch-compile) 1.04 even | old best a bf16 leg |
| rtx-4090 | pj_m16_n28672_k4096 | 0.3523 (torch-bf16-compile) 1.87 loss | 0.638 (lt-fresh) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n2048_k7168 | 0.0604 (torch-bf16-compile) 0.84 win | 0.0553 (lt-fresh) 0.92 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m1_n4096_k4096 | 0.0686 (torch-bf16-compile) 0.67 win | 0.0348 (lt-fresh) 1.33 loss | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n28672_k4096 | 6.175 (torch-bf16-compile) 2.66 loss | 16.45 (cublas-fresh) 1.00 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k14336 | 3.119 (torch-bf16-compile) 2.59 loss | 8.425 (cublas) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n4096_k4096 | 0.898 (torch-bf16-compile) 2.61 loss | 2.408 (lt) 0.97 even | old best a bf16 leg + best now a gemm_lib leg |
| rtx-4090 | pj_m2048_n6144_k4096 | 1.338 (torch-bf16-compile) 2.62 loss | 3.631 (cublas) 0.96 even | old best a bf16 leg + best now a gemm_lib leg |
| h100 | llm_blkdec_b1_l4096 | 0.4022 (torch-bf16-compile) 1.41 loss | 0.5639 (torch-compile) 1.01 even | old best a bf16 leg |
| h100 | llm_blkpre_t2048 | 2.93 (torch-bf16-compile) 9.96 loss | 28.35 (liger) 1.03 even | old best a bf16 leg + best now a gemm_lib leg |
| h100 | llm_dec_b32_l2048 | 0.3048 (torch-bf16-compile) 1.58 loss | 1.415 (torch-compile) 0.34 win | old best a bf16 leg |
| h100 | llm_dec_b32_l8192 | 0.6003 (torch-bf16-compile) 2.27 loss | 5.399 (torch-compile) 0.25 win | old best a bf16 leg |
| h100 | llm_dec_b8_l2048_bf16w | 0.258 (torch-bf16-compile) 0.63 win | 0.1706 (torch-bf16-compile-cg) 0.96 even | other |
| h100 | llm_dec_b8_l8192 | 0.3062 (torch-bf16-compile) 1.37 loss | 1.452 (torch-compile) 0.29 win | old best a bf16 leg |
| h100 | pj_m16_n18432_k7168 | 0.1011 (torch-bf16-compile) 3.16 loss | 0.3365 (torch-compile) 0.95 win | old best a bf16 leg |
| h100 | pj_m16_n4096_k14336 | 0.0537 (torch-bf16) 2.97 loss | 0.1694 (torch-compile) 0.94 win | old best a bf16 leg |
| h100 | pj_m1_n18432_k7168 | 0.1009 (torch-bf16) 1.97 loss | 0.1924 (torch-compile-cg) 1.04 even | old best a bf16 leg |
| h100 | pj_m1_n28672_k4096 | 0.0928 (torch-bf16-compile) 1.95 loss | 0.1801 (cublas) 1.01 even | old best a bf16 leg + best now a gemm_lib leg |
| h100 | pj_m2048_n6144_k4096 | 0.265 (torch-bf16) 10.30 loss | 2.793 (torch-compile-ro) 0.98 even | old best a bf16 leg |

## What the new legs change

Rows whose best leg is one the competitors step added (CUDA-graph torch.compile, SDPA backends, hand Triton, Liger, FlashAttention-2/3, FlashInfer), against the best over the legs before it (`base`).

| part | workload | prec | base: best (leg) ratio | now: best (leg) ratio | verdict |
|---|---|---|---|---|---|
| rtx-5080 | mlp2_2048_tf32 | tf32 | 1.553 (torch-eager-tf32) 1.01 | 1.549 (torch-compile-ro-tf32) 1.01 | even → even |
| rtx-5080 | causal_1024_256 | f32 | 55.55 (torch-compile) 0.24 | 45.22 (torch-compile-cg) 0.30 | win → win |
| rtx-5080 | gen_1024_512_256 | f32 | 28.53 (torch-compile) 0.24 | 23.08 (torch-compile-cg) 0.30 | win → win |
| rtx-5080 | prefill_1024_512_256 | f32 | 28.36 (torch-compile) 0.24 | 22.58 (torch-compile-cg) 0.30 | win → win |
| rtx-5080 | softmax_rows_4096 | f32 | 3.689 (torch-compile) 1.03 | 3.682 (torch-compile-ro) 1.03 | even → even |
| rtx-5080 | llm_blkpre_t2048 | f32 | 28.22 (torch-compile) 0.90 | 26.88 (liger) 0.94 | win → win |
| rtx-5080 | llm_blkpre_t2048_bf16 | bf16 | 8.232 (torch-bf16-compile) 1.17 | 8.23 (torch-bf16-compile-cg) 1.17 | loss → loss |
| rtx-5080 | llm_blkpre_t2048_bf16w | bf16w | 8.232 (torch-bf16-compile) 3.06 | 8.23 (torch-bf16-compile-cg) 3.06 | loss → loss |
| rtx-5080 | llm_blkpre_t2048_tf32 | tf32 | 18.7 (torch-compile-tf32) 1.05 | 18.08 (triton-tf32) 1.09 | loss → loss |
| rtx-5080 | llm_dec_b1_l2048 | f32 | 0.2636 (torch-compile) 0.83 | 0.2623 (torch-compile-ro) 0.84 | win → win |
| rtx-5080 | llm_pre_t2048 | f32 | 8.495 (torch-compile) 0.76 | 7.279 (liger) 0.89 | win → win |
| rtx-5080 | llm_pre_t2048_bf16 | bf16 | 1.972 (torch-bf16-compile) 1.33 | 1.971 (torch-bf16-compile-cg) 1.33 | loss → loss |
| rtx-5080 | llm_pre_t2048_bf16w | bf16w | 1.972 (torch-bf16-compile) 3.23 | 1.971 (torch-bf16-compile-cg) 3.23 | loss → loss |
| rtx-5080 | llm_pre_t2048_tf32 | tf32 | 6.171 (torch-compile-tf32) 0.87 | 4.55 (triton-tf32) 1.17 | win → loss |
| rtx-5080 | llm_pre_t8192 | f32 | 84.97 (torch-compile) 0.53 | 52.96 (liger) 0.84 | win → win |
| rtx-5080 | rt_mlp2_2048 | f32 | 15.7 (torch-compile) 0.93 | 15.69 (torch-compile-cg) 0.93 | win → win |
| rtx-3090 | epi_4096 | f32 | 6.666 (chain-lt) 0.84 | 6.654 (torch-compile-cg) 0.84 | win → win |
| rtx-3090 | pj_m128_n28672_k4096 | f32 | 1.508 (torch-compile) 0.80 | 1.223 (torch-compile-ro) 0.98 | win → even |
| rtx-3090 | pj_m128_n28672_k4096_bf16 | bf16 | 0.4833 (torch-bf16-compile) 1.08 | 0.4823 (torch-bf16-compile-cg) 1.08 | loss → loss |
| rtx-3090 | pj_m128_n4096_k14336_bf16 | bf16 | 0.2807 (torch-bf16-compile) 1.43 | 0.2806 (torch-bf16-compile-cg) 1.43 | loss → loss |
| rtx-3090 | pj_m128_n4096_k14336_tf32 | tf32 | 0.6216 (torch-compile-tf32) 1.09 | 0.6205 (torch-compile-cg-tf32) 1.09 | loss → loss |
| rtx-3090 | pj_m128_n4096_k4096_bf16 | bf16 | 0.086 (torch-bf16) 1.39 | 0.085 (torch-bf16-compile-cg) 1.41 | loss → loss |
| rtx-3090 | pj_m128_n6144_k4096 | f32 | 0.3226 (torch-eager) 1.05 | 0.3205 (torch-compile-cg) 1.05 | even → loss |
| rtx-3090 | pj_m128_n6144_k4096_bf16 | bf16 | 0.129 (torch-bf16-compile) 1.46 | 0.128 (torch-bf16-compile-cg) 1.47 | loss → loss |
| rtx-3090 | pj_m16_n4096_k14336_bf16 | bf16 | 0.1434 (torch-bf16-compile) 1.18 | 0.1423 (torch-bf16-compile-cg) 1.19 | loss → loss |
| rtx-3090 | pj_m16_n4096_k4096 | f32 | 0.086 (torch-compile) 1.40 | 0.0851 (torch-compile-cg) 1.42 | loss → loss |
| rtx-3090 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0666 (torch-bf16-compile) 1.49 | 0.0655 (torch-bf16-compile-cg) 1.52 | loss → loss |
| rtx-3090 | pj_m1_n28672_k4096 | f32 | 0.5386 (torch-eager) 1.02 | 0.5366 (torch-compile-ro) 1.02 | even → even |
| rtx-3090 | pj_m1_n4096_k14336 | f32 | 0.2765 (lt) 1.03 | 0.2755 (torch-compile-ro) 1.04 | even → even |
| rtx-3090 | pj_m1_n4096_k14336_bf16 | bf16 | 0.1495 (torch-bf16) 1.07 | 0.1464 (torch-bf16-compile-ro) 1.09 | loss → loss |
| rtx-3090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0512 (torch-bf16) 1.24 | 0.0502 (torch-bf16-compile-ro) 1.26 | loss → loss |
| rtx-3090 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0676 (torch-bf16-compile) 1.20 | 0.0645 (torch-bf16-compile-cg) 1.26 | loss → loss |
| rtx-3090 | pj_m2048_n28672_k4096_bf16 | bf16 | 7.424 (torch-bf16) 0.95 | 7.106 (triton-bf16) 0.99 | even → even |
| rtx-3090 | pj_m2048_n4096_k4096_bf16 | bf16 | 1.101 (torch-bf16-compile) 1.05 | 1.044 (torch-bf16-compile-ro) 1.10 | even → loss |
| rtx-3090 | pj_m2048_n4096_k4096_tf32 | tf32 | 2.399 (torch-compile-tf32) 1.00 | 2.381 (triton-tf32) 1.00 | even → even |
| rtx-3090 | pj_m2048_n6144_k4096_bf16 | bf16 | 1.671 (torch-bf16-compile) 0.98 | 1.608 (torch-bf16-compile-cg) 1.02 | even → even |
| rtx-3090 | mlp2_2048_tf32 | tf32 | 2.746 (torch-eager-tf32) 1.04 | 2.583 (torch-compile-ro-tf32) 1.11 | even → loss |
| rtx-3090 | mlp_2048_tf32 | tf32 | 1.411 (torch-eager-tf32) 1.00 | 1.158 (torch-compile-ro-tf32) 1.21 | even → loss |
| rtx-3090 | mlp_512 | f32 | 0.0522 (torch-compile) 1.31 | 0.044 (triton) 1.56 | loss → loss |
| rtx-3090 | attn_1024_64 | f32 | 0.0584 (chain-fused) 0.84 | 0.0461 (torch-compile-ro) 1.06 | win → loss |
| rtx-3090 | attn_2048_128 | f32 | 0.1843 (torch-sdpa-compile) 1.46 | 0.1802 (torch-compile-cg) 1.49 | loss → loss |
| rtx-3090 | attn_2048_64 | f32 | 0.1372 (torch-compile) 0.91 | 0.1311 (torch-compile-ro) 0.95 | win → win |
| rtx-3090 | causal_1024_256 | f32 | 204.3 (torch-compile) 0.12 | 179.4 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | gen_1024_512_256 | f32 | 100.6 (torch-compile) 0.13 | 93.44 (torch-compile-cg) 0.14 | win → win |
| rtx-3090 | lsm_1_131072 | f32 | 0.0307 (torch-compile) 1.03 | 0.0256 (torch-compile-cg) 1.24 | even → loss |
| rtx-3090 | lsm_32_8192 | f32 | 0.0113 (torch-sdpa-compile) 2.85 | 0.0102 (torch-compile-cg) 3.16 | loss → loss |
| rtx-3090 | lsm_4_32768 | f32 | 0.0205 (torch-sdpa-compile) 1.30 | 0.0164 (torch-sdpa-compile-cg) 1.62 | loss → loss |
| rtx-3090 | prefill_1024_512_256 | f32 | 101.4 (torch-compile) 0.13 | 90.17 (torch-compile-cg) 0.15 | win → win |
| rtx-3090 | softmax_rows_2048 | f32 | 0.8038 (torch-compile) 1.07 | 0.7455 (torch-compile-cg) 1.15 | loss → loss |
| rtx-3090 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.7057 (torch-bf16-compile) 0.91 | 0.6992 (torch-bf16-compile-cg) 0.92 | win → win |
| rtx-3090 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.7057 (torch-bf16-compile) 0.98 | 0.6992 (torch-bf16-compile-cg) 0.99 | even → even |
| rtx-3090 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.7057 (torch-bf16-compile) 1.00 | 0.6992 (torch-bf16-compile-cg) 1.01 | even → even |
| rtx-3090 | llm_blkpre_t2048 | f32 | 48.57 (torch-compile) 0.94 | 47.5 (liger) 0.96 | win → even |
| rtx-3090 | llm_blkpre_t2048_tf32 | tf32 | 37.74 (torch-eager-tf32) 0.88 | 31.44 (triton-tf32) 1.06 | win → loss |
| rtx-3090 | llm_dec_b32_l2048 | f32 | 2.876 (torch-compile) 0.34 | 2.858 (torch-compile-ro) 0.34 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16 | bf16 | 0.8513 (torch-bf16-compile) 0.79 | 0.8439 (torch-bf16-compile-cg) 0.80 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.8513 (torch-bf16-compile) 0.79 | 0.8439 (torch-bf16-compile-cg) 0.80 | win → win |
| rtx-3090 | llm_dec_b32_l2048_bf16w | bf16w | 0.8513 (torch-bf16-compile) 1.13 | 0.8439 (torch-bf16-compile-cg) 1.14 | loss → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16 | bf16 | 3.046 (torch-bf16-compile) 0.65 | 1.794 (torch-bf16-cudnn) 1.11 | win → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16kv | bf16kv | 3.046 (torch-bf16-compile) 0.65 | 1.794 (torch-bf16-cudnn) 1.10 | win → loss |
| rtx-3090 | llm_dec_b32_l8192_bf16w | bf16w | 3.046 (torch-bf16-compile) 1.02 | 1.794 (torch-bf16-cudnn) 1.74 | even → loss |
| rtx-3090 | llm_pre_t2048 | f32 | 14.38 (torch-compile) 0.75 | 12.95 (torch-sdpa-efficient) 0.83 | win → win |
| rtx-3090 | llm_pre_t2048_bf16 | bf16 | 3.506 (torch-bf16-compile) 1.33 | 3.486 (torch-bf16-compile-cg) 1.34 | loss → loss |
| rtx-3090 | llm_pre_t2048_bf16w | bf16w | 3.506 (torch-bf16-compile) 3.02 | 3.486 (torch-bf16-compile-cg) 3.04 | loss → loss |
| rtx-3090 | llm_pre_t2048_tf32 | tf32 | 10.47 (torch-compile-tf32) 0.88 | 10.28 (torch-compile-ro-tf32) 0.89 | win → win |
| rtx-3090 | llm_pre_t8192 | f32 | 133.5 (torch-compile) 0.81 | 94.44 (torch-sdpa-efficient) 1.14 | win → loss |
| rtx-3090 | conv_2048_5 | f32 | 0.1618 (torch-compile) 0.70 | 0.1608 (torch-compile-cg) 0.71 | win → win |
| rtx-3090 | fir_4194304_16 | f32 | 0.1106 (torch-compile) 0.69 | 0.1097 (torch-compile-ro) 0.69 | win → win |
| rtx-3090 | firew_rowsum_2048x2048_16 | f32 | 0.1403 (torch-compile) 0.74 | 0.1393 (torch-compile-cg) 0.75 | win → win |
| rtx-3090 | loop_4096 | f32 | 52.05 (torch-compile) 1.22 | 52.03 (torch-compile-cg) 1.22 | loss → loss |
| rtx-3090 | loopmm_1024_8 | f32 | 0.9329 (torch-compile) 1.19 | 0.9277 (torch-compile-cg) 1.20 | loss → loss |
| rtx-3090 | rt_mlp2_2048 | f32 | 30.48 (torch-compile) 0.87 | 30.45 (torch-compile-cg) 0.87 | win → win |
| a100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.2468 (torch-bf16-compile) 1.40 | 0.2171 (triton-bf16) 1.59 | loss → loss |
| a100 | pj_m128_n28672_k4096_tf32 | tf32 | 0.4844 (lt-tf32) 1.34 | 0.4147 (triton-tf32) 1.57 | loss → loss |
| a100 | pj_m128_n4096_k14336_bf16 | bf16 | 0.1249 (torch-bf16) 2.21 | 0.1229 (torch-bf16-compile-cg) 2.25 | loss → loss |
| a100 | pj_m128_n4096_k4096_bf16 | bf16 | 0.0584 (torch-bf16-compile) 1.86 | 0.0532 (triton-bf16) 2.05 | loss → loss |
| a100 | pj_m128_n4096_k4096_tf32 | tf32 | 0.0901 (cublas-tf32-fresh) 2.20 | 0.087 (triton-tf32) 2.27 | loss → loss |
| a100 | pj_m128_n6144_k4096_bf16 | bf16 | 0.0686 (torch-bf16-compile) 2.00 | 0.0635 (triton-bf16) 2.16 | loss → loss |
| a100 | pj_m128_n6144_k4096_tf32 | tf32 | 0.1075 (cublas-tf32-fresh) 1.93 | 0.1004 (triton-tf32) 2.07 | loss → loss |
| a100 | pj_m16_n4096_k14336 | f32 | 0.3123 (torch-compile) 0.87 | 0.3103 (torch-compile-cg) 0.87 | win → win |
| a100 | pj_m16_n4096_k14336_bf16 | bf16 | 0.1126 (torch-bf16-compile) 1.19 | 0.1106 (torch-bf16-compile-cg) 1.21 | loss → loss |
| a100 | pj_m16_n4096_k4096 | f32 | 0.1055 (lt-fresh) 1.13 | 0.1044 (torch-compile-cg) 1.14 | loss → loss |
| a100 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0451 (torch-bf16-compile) 1.22 | 0.044 (torch-bf16-compile-cg) 1.25 | loss → loss |
| a100 | pj_m16_n6144_k4096 | f32 | 0.1178 (torch-compile) 1.21 | 0.1167 (torch-compile-cg) 1.22 | loss → loss |
| a100 | pj_m16_n6144_k4096_bf16 | bf16 | 0.0614 (torch-bf16-compile) 1.23 | 0.0594 (torch-bf16-compile-cg) 1.28 | loss → loss |
| a100 | pj_m1_n28672_k4096 | f32 | 0.3564 (torch-compile) 1.10 | 0.3512 (torch-compile-cg) 1.12 | loss → loss |
| a100 | pj_m1_n28672_k4096_bf16 | bf16 | 0.1884 (torch-bf16-compile) 1.40 | 0.1843 (torch-bf16-compile-cg) 1.43 | loss → loss |
| a100 | pj_m1_n4096_k14336_bf16 | bf16 | 0.1106 (torch-bf16-compile) 1.27 | 0.1085 (torch-bf16-compile-cg) 1.30 | loss → loss |
| a100 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0481 (torch-bf16) 1.16 | 0.0451 (torch-bf16-compile-cg) 1.24 | loss → loss |
| a100 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0625 (torch-bf16-compile) 1.32 | 0.0594 (torch-bf16-compile-cg) 1.39 | loss → loss |
| a100 | pj_m2048_n28672_k4096_bf16 | bf16 | 1.76 (torch-bf16) 1.92 | 1.742 (torch-bf16-compile-ro) 1.94 | loss → loss |
| a100 | pj_m2048_n4096_k14336 | f32 | 12.67 (torch-eager) 0.93 | 12.65 (torch-compile-ro) 0.93 | win → win |
| a100 | pj_m2048_n4096_k14336_tf32 | tf32 | 1.894 (cublas-tf32) 1.88 | 1.838 (torch-compile-ro-tf32) 1.94 | loss → loss |
| a100 | pj_m2048_n4096_k4096 | f32 | 3.692 (cublas) 0.91 | 3.692 (torch-compile-cg) 0.91 | win → win |
| a100 | pj_m2048_n4096_k4096_bf16 | bf16 | 0.3092 (torch-bf16) 1.93 | 0.3082 (triton-bf16) 1.94 | loss → loss |
| a100 | pj_m2048_n6144_k4096_tf32 | tf32 | 0.809 (torch-eager-tf32) 1.89 | 0.8049 (torch-compile-ro-tf32) 1.90 | loss → loss |
| a100 | mlp2_2048_tf32 | tf32 | 0.7066 (torch-compile-tf32) 1.71 | 0.6984 (torch-compile-ro-tf32) 1.73 | loss → loss |
| a100 | mlp_2048 | f32 | 1.974 (torch-eager) 0.90 | 1.968 (torch-compile-ro) 0.90 | win → win |
| a100 | mlp_2048_tf32 | tf32 | 0.3553 (torch-eager-tf32) 1.85 | 0.3523 (torch-compile-ro-tf32) 1.86 | loss → loss |
| a100 | smlp_4096_256_1024 | f32 | 0.3144 (torch-eager) 0.92 | 0.3082 (triton) 0.93 | win → win |
| a100 | attn_1024_64 | f32 | 0.0717 (torch-compile) 0.96 | 0.0584 (torch-compile-ro) 1.18 | even → loss |
| a100 | causal_1024_256 | f32 | 209 (torch-compile) 0.25 | 192.8 (torch-compile-cg) 0.28 | win → win |
| a100 | gen_1024_512_256 | f32 | 104 (torch-compile) 0.23 | 92.24 (torch-compile-cg) 0.26 | win → win |
| a100 | lsm_1_131072 | f32 | 0.044 (torch-compile) 0.84 | 0.0317 (torch-compile-ro) 1.16 | win → loss |
| a100 | lsm_32_8192 | f32 | 0.0184 (torch-compile) 2.13 | 0.0174 (triton) 2.25 | loss → loss |
| a100 | lsm_4_32768 | f32 | 0.0348 (torch-compile) 0.98 | 0.0225 (torch-compile-cg) 1.51 | even → loss |
| a100 | prefill_1024_512_256 | f32 | 104.6 (torch-compile) 0.23 | 93.03 (torch-compile-cg) 0.26 | win → win |
| a100 | rowsum_2048 | f32 | 1.268 (chain-eager) 0.72 | 1.248 (torch-compile-ro) 0.73 | win → win |
| a100 | softmax_rows_4096 | f32 | 7.547 (chain-fused) 0.90 | 7.356 (torch-compile-ro) 0.93 | win → win |
| a100 | llm_blkpre_t2048 | f32 | 53.82 (torch-compile) 0.70 | 51.35 (liger) 0.73 | win → win |
| a100 | llm_blkpre_t2048_bf16 | bf16 | 4.615 (torch-bf16) 1.84 | 4.01 (torch-bf16-compile-ro) 2.12 | loss → loss |
| a100 | llm_blkpre_t2048_bf16w | bf16w | 4.615 (torch-bf16) 5.13 | 4.01 (torch-bf16-compile-ro) 5.90 | loss → loss |
| a100 | llm_blkpre_t2048_tf32 | tf32 | 9.548 (torch-compile-tf32) 1.88 | 8.989 (triton-tf32) 1.99 | loss → loss |
| a100 | llm_dec_b1_l2048 | f32 | 0.3552 (torch-compile) 0.74 | 0.324 (torch-compile-ro) 0.81 | win → win |
| a100 | llm_dec_b1_l2048_bf16 | bf16 | 0.2649 (torch-bf16-compile) 0.69 | 0.2615 (torch-bf16-compile-cg) 0.70 | win → win |
| a100 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.2649 (torch-bf16-compile) 0.74 | 0.2615 (torch-bf16-compile-cg) 0.75 | win → win |
| a100 | llm_dec_b1_l2048_bf16w | bf16w | 0.2649 (torch-bf16-compile) 0.70 | 0.2615 (torch-bf16-compile-cg) 0.71 | win → win |
| a100 | llm_dec_b1_l8192 | f32 | 0.4856 (torch-compile) 0.61 | 0.4781 (torch-compile-cg) 0.62 | win → win |
| a100 | llm_dec_b1_l8192_bf16 | bf16 | 0.3134 (torch-bf16-compile) 0.72 | 0.2979 (torch-bf16-compile-cg) 0.76 | win → win |
| a100 | llm_dec_b1_l8192_bf16kv | bf16kv | 0.3134 (torch-bf16-compile) 0.74 | 0.2979 (torch-bf16-compile-cg) 0.77 | win → win |
| a100 | llm_dec_b1_l8192_bf16w | bf16w | 0.3134 (torch-bf16-compile) 0.79 | 0.2979 (torch-bf16-compile-cg) 0.83 | win → win |
| a100 | llm_dec_b32_l8192 | f32 | 6.662 (torch-compile) 0.30 | 6.628 (torch-compile-cg) 0.30 | win → win |
| a100 | llm_dec_b32_l8192_bf16 | bf16 | 1.462 (torch-bf16) 1.12 | 1.269 (torch-bf16-cudnn) 1.29 | loss → loss |
| a100 | llm_dec_b32_l8192_bf16kv | bf16kv | 1.462 (torch-bf16) 1.09 | 1.269 (torch-bf16-cudnn) 1.25 | loss → loss |
| a100 | llm_dec_b32_l8192_bf16w | bf16w | 1.462 (torch-bf16) 1.34 | 1.269 (torch-bf16-cudnn) 1.54 | loss → loss |
| a100 | llm_dec_b8_l8192_bf16 | bf16 | 0.6537 (torch-bf16-compile) 0.96 | 0.6208 (torch-bf16-compile-cg) 1.01 | even → even |
| a100 | llm_dec_b8_l8192_bf16kv | bf16kv | 0.6537 (torch-bf16-compile) 0.94 | 0.6208 (torch-bf16-compile-cg) 0.99 | win → even |
| a100 | llm_dec_b8_l8192_bf16w | bf16w | 0.6537 (torch-bf16-compile) 1.08 | 0.6208 (torch-bf16-compile-cg) 1.14 | loss → loss |
| a100 | llm_pre_t2048 | f32 | 14.37 (torch-compile) 0.90 | 11.43 (liger) 1.13 | win → loss |
| a100 | llm_pre_t2048_bf16 | bf16 | 1.041 (torch-bf16-compile) 2.60 | 1.033 (torch-bf16-compile-cg) 2.62 | loss → loss |
| a100 | llm_pre_t2048_bf16w | bf16w | 1.041 (torch-bf16-compile) 7.99 | 1.033 (torch-bf16-compile-cg) 8.05 | loss → loss |
| a100 | llm_pre_t2048_tf32 | tf32 | 3.416 (torch-compile-tf32) 2.16 | 2.637 (triton-tf32) 2.80 | loss → loss |
| a100 | llm_pre_t8192 | f32 | 113.9 (torch-compile) 0.82 | 58.94 (liger) 1.59 | win → loss |
| a100 | conv_2048_5 | f32 | 0.1393 (torch-compile) 0.70 | 0.1382 (torch-compile-cg) 0.71 | win → win |
| a100 | firew_rowsum_2048x2048_16 | f32 | 0.126 (torch-compile) 0.66 | 0.1178 (torch-compile-cg) 0.70 | win → win |
| a100 | fanout_2048 | f32 | 2.505 (chain-streams) 0.72 | 1.99 (torch-compile-cg) 0.91 | win → win |
| a100 | indep_2048 | f32 | 1.961 (chain-lt) 0.90 | 1.958 (torch-compile-cg) 0.90 | win → win |
| a100 | loop_4096 | f32 | 58.83 (torch-eager) 0.90 | 58.44 (torch-compile-ro) 0.90 | win → win |
| a100 | rt_mlp2_1024 | f32 | 5.471 (torch-eager) 1.00 | 4.347 (torch-compile-ro) 1.26 | even → loss |
| a100 | rt_mlp2_2048 | f32 | 31.73 (torch-compile) 0.90 | 31.52 (torch-compile-cg) 0.90 | win → win |
| rtx-4090 | pj_m128_n4096_k14336_bf16 | bf16 | 0.2478 (torch-bf16) 1.23 | 0.2468 (torch-bf16-compile-cg) 1.24 | loss → loss |
| rtx-4090 | pj_m128_n6144_k4096 | f32 | 0.3011 (torch-eager) 1.12 | 0.3 (torch-compile-cg) 1.12 | loss → loss |
| rtx-4090 | pj_m16_n4096_k14336_bf16 | bf16 | 0.2017 (torch-bf16-compile) 0.86 | 0.2007 (torch-bf16-compile-cg) 0.87 | win → win |
| rtx-4090 | pj_m16_n4096_k4096_bf16 | bf16 | 0.0625 (torch-bf16-compile) 1.17 | 0.0614 (torch-bf16-compile-cg) 1.19 | loss → loss |
| rtx-4090 | pj_m1_n28672_k4096 | f32 | 0.6001 (torch-compile) 1.09 | 0.599 (torch-compile-cg) 1.09 | loss → loss |
| rtx-4090 | pj_m1_n4096_k14336_bf16 | bf16 | 0.1823 (torch-bf16-compile) 0.71 | 0.1802 (torch-bf16-compile-cg) 0.72 | win → win |
| rtx-4090 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0686 (torch-bf16-compile) 0.49 | 0.0666 (torch-bf16-compile-cg) 0.51 | win → win |
| rtx-4090 | pj_m2048_n28672_k4096_bf16 | bf16 | 6.175 (torch-bf16-compile) 1.03 | 6.174 (torch-bf16-compile-cg) 1.03 | even → even |
| rtx-4090 | pj_m2048_n6144_k4096_bf16 | bf16 | 1.338 (torch-bf16-compile) 1.05 | 1.337 (torch-bf16-compile-cg) 1.05 | even → even |
| rtx-4090 | mlp2_2048_tf32 | tf32 | 1.888 (torch-eager-tf32) 1.11 | 1.846 (torch-compile-ro-tf32) 1.14 | loss → loss |
| rtx-4090 | mlp_2048_tf32 | tf32 | 0.9317 (torch-eager-tf32) 1.11 | 0.9257 (torch-compile-ro-tf32) 1.11 | loss → loss |
| rtx-4090 | mlp_512 | f32 | 0.0502 (torch-compile) 1.59 | 0.0492 (torch-compile-cg) 1.62 | loss → loss |
| rtx-4090 | smlp_4096_256_1024 | f32 | 0.1935 (torch-compile) 1.08 | 0.1915 (torch-compile-cg) 1.09 | loss → loss |
| rtx-4090 | attn_1024_64 | f32 | 0.0471 (torch-compile) 0.97 | 0.042 (torch-sdpa-compile-cg) 1.09 | even → loss |
| rtx-4090 | attn_2048_128 | f32 | 0.1597 (torch-compile) 1.07 | 0.1485 (torch-compile-ro) 1.15 | loss → loss |
| rtx-4090 | attn_2048_64 | f32 | 0.1055 (torch-sdpa-compile) 1.17 | 0.1024 (torch-sdpa-compile-cg) 1.21 | loss → loss |
| rtx-4090 | causal_1024_256 | f32 | 269.8 (torch-compile) 0.12 | 244.1 (torch-compile-cg) 0.14 | win → win |
| rtx-4090 | gen_1024_512_256 | f32 | 122.9 (torch-compile) 0.14 | 120.9 (torch-compile-ro) 0.14 | win → win |
| rtx-4090 | lsm_1_131072 | f32 | 0.0389 (torch-compile) 0.76 | 0.0317 (torch-compile-ro) 0.93 | win → even |
| rtx-4090 | lsm_32_8192 | f32 | 0.0133 (torch-sdpa-compile) 2.46 | 0.0123 (torch-compile-cg) 2.66 | loss → loss |
| rtx-4090 | lsm_4_32768 | f32 | 0.0256 (torch-compile) 0.93 | 0.0205 (torch-compile-cg) 1.16 | win → loss |
| rtx-4090 | prefill_1024_512_256 | f32 | 126 (torch-compile) 0.14 | 121 (torch-compile-ro) 0.14 | win → win |
| rtx-4090 | rowsum_2048 | f32 | 0.6515 (chain-eager) 0.95 | 0.6513 (torch-compile-ro) 0.95 | win → win |
| rtx-4090 | softmax_rows_2048 | f32 | 0.6564 (torch-compile) 0.97 | 0.6554 (torch-compile-cg) 0.97 | even → even |
| rtx-4090 | softmax_rows_4096 | f32 | 4.958 (torch-compile) 0.98 | 4.891 (torch-compile-ro) 0.99 | even → even |
| rtx-4090 | llm_blkpre_t2048 | f32 | 36.21 (torch-compile) 0.92 | 35.76 (liger) 0.93 | win → win |
| rtx-4090 | llm_blkpre_t2048_tf32 | tf32 | 27.16 (torch-compile-tf32) 1.07 | 26.18 (triton-tf32) 1.11 | loss → loss |
| rtx-4090 | llm_pre_t2048 | f32 | 10.59 (torch-compile) 0.86 | 10.35 (liger) 0.88 | win → win |
| rtx-4090 | llm_pre_t2048_bf16 | bf16 | 2.911 (torch-bf16-compile) 1.27 | 2.91 (torch-bf16-compile-cg) 1.27 | loss → loss |
| rtx-4090 | llm_pre_t2048_bf16w | bf16w | 2.911 (torch-bf16-compile) 3.24 | 2.91 (torch-bf16-compile-cg) 3.24 | loss → loss |
| rtx-4090 | llm_pre_t2048_tf32 | tf32 | 8.056 (torch-compile-tf32) 1.03 | 6.619 (triton-tf32) 1.26 | even → loss |
| rtx-4090 | llm_pre_t8192 | f32 | 92.77 (torch-compile) 0.88 | 78.31 (liger) 1.04 | win → even |
| rtx-4090 | conv_2048_5 | f32 | 0.169 (torch-eager) 0.60 | 0.1679 (torch-compile-cg) 0.60 | win → win |
| rtx-4090 | ewconvew_2048_5 | f32 | 0.2212 (torch-compile) 0.54 | 0.2181 (torch-compile-cg) 0.55 | win → win |
| rtx-4090 | firew_rowsum_2048x2048_16 | f32 | 0.1178 (torch-compile) 0.69 | 0.1167 (torch-compile-cg) 0.70 | win → win |
| rtx-4090 | loopmm_1024_8 | f32 | 0.7188 (torch-compile) 1.17 | 0.7127 (torch-compile-cg) 1.18 | loss → loss |
| rtx-4090 | rt_mlp2_1024 | f32 | 3.068 (torch-compile) 1.10 | 2.998 (torch-compile-cg) 1.12 | loss → loss |
| rtx-4090 | twin_2048 | f32 | 1.279 (torch-compile) 0.94 | 1.277 (torch-compile-cg) 0.94 | win → win |
| h100 | epi_4096 | f32 | 3.935 (chain-lt) 1.12 | 3.912 (torch-compile-cg) 1.13 | loss → loss |
| h100 | pj_m128_n28672_k4096 | f32 | 0.9249 (torch-compile) 1.05 | 0.918 (torch-compile-ro) 1.06 | even → loss |
| h100 | pj_m128_n28672_k4096_bf16 | bf16 | 0.1307 (torch-bf16-compile) 1.65 | 0.1295 (torch-bf16-compile-cg) 1.66 | loss → loss |
| h100 | pj_m128_n4096_k14336 | f32 | 0.4256 (torch-compile) 1.19 | 0.4187 (torch-compile-ro) 1.21 | loss → loss |
| h100 | pj_m128_n4096_k14336_tf32 | tf32 | 0.1468 (torch-compile-tf32) 2.31 | 0.1464 (torch-compile-cg-tf32) 2.32 | loss → loss |
| h100 | pj_m128_n4096_k4096_tf32 | tf32 | 0.0435 (torch-eager-tf32) 2.02 | 0.0434 (torch-compile-cg-tf32) 2.03 | loss → loss |
| h100 | pj_m128_n6144_k4096 | f32 | 0.1911 (lt-fresh) 1.20 | 0.1901 (torch-compile-ro) 1.20 | loss → loss |
| h100 | pj_m16_n2048_k7168 | f32 | 0.05 (torch-compile) 1.18 | 0.0487 (torch-compile-cg) 1.22 | loss → loss |
| h100 | pj_m16_n28672_k4096 | f32 | 0.2616 (torch-compile) 1.05 | 0.2601 (torch-compile-cg) 1.06 | even → loss |
| h100 | pj_m16_n28672_k4096_bf16 | bf16 | 0.0929 (torch-bf16-compile) 2.09 | 0.0916 (torch-bf16-compile-cg) 2.12 | loss → loss |
| h100 | pj_m1_n18432_k7168 | f32 | 0.1976 (torch-compile) 1.01 | 0.1924 (torch-compile-cg) 1.04 | even → even |
| h100 | pj_m1_n4096_k14336_bf16 | bf16 | 0.0524 (torch-bf16-compile) 1.35 | 0.0514 (torch-bf16-compile-cg) 1.38 | loss → loss |
| h100 | pj_m1_n4096_k4096_bf16 | bf16 | 0.0204 (torch-bf16) 1.11 | 0.0203 (torch-bf16-compile-ro) 1.12 | loss → loss |
| h100 | pj_m1_n6144_k4096_bf16 | bf16 | 0.0283 (torch-bf16-compile) 1.28 | 0.0279 (torch-bf16-compile-cg) 1.30 | loss → loss |
| h100 | pj_m2048_n4096_k14336_bf16 | bf16 | 0.6816 (torch-bf16-compile) 1.92 | 0.6779 (torch-bf16-compile-cg) 1.93 | loss → loss |
| h100 | pj_m2048_n4096_k4096 | f32 | 1.798 (torch-compile) 1.36 | 1.686 (torch-compile-cg) 1.45 | loss → loss |
| h100 | pj_m2048_n6144_k4096 | f32 | 2.914 (torch-compile) 0.94 | 2.793 (torch-compile-ro) 0.98 | win → even |
| h100 | mlp2_2048 | f32 | 1.778 (torch-compile) 1.06 | 1.687 (torch-compile-ro) 1.12 | loss → loss |
| h100 | mlp2_2048_tf32 | tf32 | 0.3792 (torch-compile-tf32) 1.51 | 0.367 (torch-compile-ro-tf32) 1.56 | loss → loss |
| h100 | mlp_2048 | f32 | 0.8398 (torch-eager) 1.11 | 0.8394 (torch-compile-ro) 1.11 | loss → loss |
| h100 | mlp_2048_tf32 | tf32 | 0.1752 (torch-compile-tf32) 1.67 | 0.1733 (torch-compile-ro-tf32) 1.69 | loss → loss |
| h100 | mlp_512 | f32 | 0.0315 (torch-compile) 1.44 | 0.0303 (torch-compile-cg) 1.50 | loss → loss |
| h100 | attn_1024_64 | f32 | 0.0359 (chain-fused) 1.20 | 0.0246 (torch-compile-cg) 1.75 | loss → loss |
| h100 | attn_2048_128 | f32 | 0.0868 (torch-compile) 1.45 | 0.0841 (torch-compile-ro) 1.49 | loss → loss |
| h100 | attn_2048_64 | f32 | 0.0614 (torch-compile) 1.09 | 0.0573 (torch-compile-cg) 1.16 | loss → loss |
| h100 | causal_1024_256 | f32 | 117.1 (torch-compile) 0.18 | 116.7 (torch-compile-cg) 0.18 | win → win |
| h100 | lsm_1_131072 | f32 | 0.0222 (torch-compile) 1.20 | 0.0184 (torch-sdpa-compile-cg) 1.44 | loss → loss |
| h100 | lsm_32_8192 | f32 | 0.0116 (torch-compile) 2.17 | 0.0088 (torch-compile-cg) 2.86 | loss → loss |
| h100 | lsm_4_32768 | f32 | 0.0203 (torch-sdpa) 1.19 | 0.0154 (torch-sdpa-compile-cg) 1.56 | loss → loss |
| h100 | prefill_1024_512_256 | f32 | 57.62 (torch-compile) 0.19 | 56.43 (torch-compile-ro) 0.19 | win → win |
| h100 | rowsum_2048 | f32 | 0.4206 (torch-compile) 1.22 | 0.4195 (torch-compile-cg) 1.22 | loss → loss |
| h100 | rowsum_4096 | f32 | 4.009 (torch-compile) 1.13 | 3.986 (torch-compile-ro) 1.14 | loss → loss |
| h100 | softmax_rows_2048 | f32 | 0.4463 (chain-fused) 1.20 | 0.4275 (torch-compile-cg) 1.25 | loss → loss |
| h100 | softmax_rows_4096 | f32 | 4.048 (torch-compile) 1.13 | 4.002 (torch-compile-ro) 1.14 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16 | bf16 | 0.3796 (torch-bf16-compile) 0.74 | 0.3052 (torch-bf16-compile-ro) 0.92 | win → win |
| h100 | llm_blkdec_b1_l4096_bf16kv | bf16kv | 0.3796 (torch-bf16-compile) 1.31 | 0.3052 (torch-bf16-compile-ro) 1.62 | loss → loss |
| h100 | llm_blkdec_b1_l4096_bf16w | bf16w | 0.3796 (torch-bf16-compile) 1.31 | 0.3052 (torch-bf16-compile-ro) 1.63 | loss → loss |
| h100 | llm_blkpre_t2048 | f32 | 29.6 (torch-compile) 0.99 | 28.35 (liger) 1.03 | even → even |
| h100 | llm_blkpre_t2048_bf16 | bf16 | 2.93 (torch-bf16-compile) 2.12 | 2.765 (torch-bf16-compile-ro) 2.24 | loss → loss |
| h100 | llm_blkpre_t2048_bf16w | bf16w | 2.93 (torch-bf16-compile) 6.26 | 2.765 (torch-bf16-compile-ro) 6.64 | loss → loss |
| h100 | llm_blkpre_t2048_tf32 | tf32 | 7.309 (torch-compile-tf32) 1.20 | 6.586 (liger-tf32) 1.33 | loss → loss |
| h100 | llm_dec_b1_l2048 | f32 | 0.2241 (torch-compile) 0.49 | 0.2109 (torch-compile-cg) 0.52 | win → win |
| h100 | llm_dec_b1_l2048_bf16 | bf16 | 0.2273 (torch-bf16-compile) 0.37 | 0.153 (torch-bf16-compile-ro) 0.55 | win → win |
| h100 | llm_dec_b1_l2048_bf16kv | bf16kv | 0.2273 (torch-bf16-compile) 0.38 | 0.153 (torch-bf16-compile-ro) 0.57 | win → win |
| h100 | llm_dec_b1_l2048_bf16w | bf16w | 0.2273 (torch-bf16-compile) 0.39 | 0.153 (torch-bf16-compile-ro) 0.59 | win → win |
| h100 | llm_dec_b1_l8192 | f32 | 0.3583 (torch-compile) 0.40 | 0.332 (torch-compile-cg) 0.43 | win → win |
| h100 | llm_dec_b1_l8192_bf16 | bf16 | 0.2345 (torch-bf16-compile) 0.48 | 0.1621 (torch-bf16-compile-ro) 0.70 | win → win |
| h100 | llm_dec_b1_l8192_bf16kv | bf16kv | 0.2345 (torch-bf16-compile) 0.48 | 0.1621 (torch-bf16-compile-ro) 0.70 | win → win |
| h100 | llm_dec_b1_l8192_bf16w | bf16w | 0.2345 (torch-bf16-compile) 0.50 | 0.1621 (torch-bf16-compile-ro) 0.72 | win → win |
| h100 | llm_dec_b32_l2048_bf16 | bf16 | 0.2905 (torch-bf16-compile) 1.01 | 0.2717 (torch-bf16-compile-ro) 1.07 | even → loss |
| h100 | llm_dec_b32_l2048_bf16kv | bf16kv | 0.2905 (torch-bf16-compile) 1.11 | 0.2717 (torch-bf16-compile-ro) 1.19 | loss → loss |
| h100 | llm_dec_b32_l2048_bf16w | bf16w | 0.2905 (torch-bf16-compile) 1.40 | 0.2717 (torch-bf16-compile-ro) 1.50 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16 | bf16 | 0.5939 (torch-bf16-compile) 1.47 | 0.58 (torch-bf16-compile-ro) 1.51 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16kv | bf16kv | 0.5939 (torch-bf16-compile) 1.53 | 0.58 (torch-bf16-compile-ro) 1.56 | loss → loss |
| h100 | llm_dec_b32_l8192_bf16w | bf16w | 0.5939 (torch-bf16-compile) 2.16 | 0.58 (torch-bf16-compile-ro) 2.21 | loss → loss |
| h100 | llm_dec_b8_l2048_bf16 | bf16 | 0.258 (torch-bf16-compile) 0.56 | 0.1706 (torch-bf16-compile-cg) 0.85 | win → win |
| h100 | llm_dec_b8_l2048_bf16kv | bf16kv | 0.258 (torch-bf16-compile) 0.57 | 0.1706 (torch-bf16-compile-cg) 0.86 | win → win |
| h100 | llm_dec_b8_l2048_bf16w | bf16w | 0.258 (torch-bf16-compile) 0.63 | 0.1706 (torch-bf16-compile-cg) 0.96 | win → even |
| h100 | llm_dec_b8_l8192_bf16 | bf16 | 0.298 (torch-bf16-compile) 1.03 | 0.2707 (torch-bf16-compile-ro) 1.13 | even → loss |
| h100 | llm_dec_b8_l8192_bf16kv | bf16kv | 0.298 (torch-bf16-compile) 1.03 | 0.2707 (torch-bf16-compile-ro) 1.14 | even → loss |
| h100 | llm_dec_b8_l8192_bf16w | bf16w | 0.298 (torch-bf16-compile) 1.31 | 0.2707 (torch-bf16-compile-ro) 1.45 | loss → loss |
| h100 | llm_pre_t2048 | f32 | 8.488 (torch-compile) 0.83 | 6.568 (liger) 1.07 | win → loss |
| h100 | llm_pre_t2048_bf16 | bf16 | 0.6815 (torch-bf16-compile) 2.59 | 0.6509 (torch-bf16-compile-ro) 2.71 | loss → loss |
| h100 | llm_pre_t2048_bf16w | bf16w | 0.6815 (torch-bf16-compile) 6.95 | 0.6509 (torch-bf16-compile-ro) 7.28 | loss → loss |
| h100 | llm_pre_t2048_tf32 | tf32 | 2.674 (torch-compile-tf32) 1.09 | 2.162 (triton-tf32) 1.34 | loss → loss |
| h100 | llm_pre_t8192 | f32 | 70.07 (torch-compile) 0.72 | 39.47 (liger) 1.28 | win → loss |
| h100 | ewconvew_2048_5 | f32 | 0.1257 (torch-compile) 0.44 | 0.1162 (torch-compile-cg) 0.47 | win → win |
| h100 | firew_rowsum_2048x2048_16 | f32 | 0.0661 (torch-compile) 0.58 | 0.0644 (torch-compile-ro) 0.59 | win → win |
| h100 | fanout_2048 | f32 | 0.8476 (torch-compile) 1.15 | 0.8438 (torch-compile-ro) 1.15 | loss → loss |
| h100 | indep_2048 | f32 | 0.8459 (torch-compile) 1.15 | 0.8421 (torch-compile-cg) 1.16 | loss → loss |
| h100 | loopmm_1024_8 | f32 | 0.4972 (torch-compile) 1.25 | 0.4936 (torch-compile-cg) 1.26 | loss → loss |
| h100 | rt_mlp2_1024 | f32 | 2.341 (torch-compile) 1.15 | 2.268 (torch-compile-cg) 1.18 | loss → loss |
| h100 | rt_mlp2_2048 | f32 | 16.47 (torch-compile) 0.92 | 16.37 (torch-compile-cg) 0.92 | win → win |

Mapal wins that disappear: 19 — rtx-5080 llm_pre_t2048_tf32 (triton-tf32), rtx-3090 attn_1024_64 (torch-compile-ro), rtx-3090 llm_blkpre_t2048 (liger), rtx-3090 llm_blkpre_t2048_tf32 (triton-tf32), rtx-3090 llm_dec_b32_l8192_bf16 (torch-bf16-cudnn), rtx-3090 llm_dec_b32_l8192_bf16kv (torch-bf16-cudnn), rtx-3090 llm_pre_t8192 (torch-sdpa-efficient), rtx-3090 pj_m128_n28672_k4096 (torch-compile-ro), a100 llm_dec_b8_l8192_bf16kv (torch-bf16-compile-cg), a100 llm_pre_t2048 (liger), a100 llm_pre_t8192 (liger), a100 lsm_1_131072 (torch-compile-ro), rtx-4090 llm_pre_t8192 (liger), rtx-4090 lsm_1_131072 (torch-compile-ro), rtx-4090 lsm_4_32768 (torch-compile-cg), h100 llm_dec_b8_l2048_bf16w (torch-bf16-compile-cg), h100 llm_pre_t2048 (liger), h100 llm_pre_t8192 (liger), h100 pj_m2048_n6144_k4096 (torch-compile-ro).

## Every competitor leg

Per row and part, every leg at the row's precision, grouped by competitor family: `leg ms (Mapal/leg)`; **bold** = the row's fastest leg; `✗gate` = timed but outside the value gate (not counted); `cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg for this row on another part; `unavailable` = the part's run reported the package missing. Families: vendor libs; torch eager; torch.compile; compile + CUDA graphs; reduce-overhead; SDPA; FlashAttention-2/3; FlashInfer; hand Triton; Liger; TensorRT-LLM; hand CUDA. Columns with no leg on any row of a table are left out.

### GEMM cells & projections

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| epi_4096 | rtx-5080 | 3.538 | **chain-lt 3.57 (0.99)**<br>chain-eager 3.701 (0.96)<br>chain-cutlass 3.76 (0.94) | torch-eager 3.821 (0.93) | torch-compile 3.68 (0.96) | torch-compile-cg 3.676 (0.96) | torch-compile-ro 3.663 (0.97) | triton 4.491 (0.79) |
| epi_4096 | rtx-3090 | 5.567 | chain-lt 6.666 (0.84)<br>chain-eager 6.81 (0.82)<br>chain-cutlass 6.814 (0.82) | torch-eager 6.803 (0.82) | torch-compile 6.709 (0.83) | **torch-compile-cg 6.654 (0.84)** | torch-compile-ro 6.734 (0.83) | triton 10.29 (0.54) |
| epi_4096 | a100 | 6.618 | **chain-lt 7.27 (0.91)**<br>chain-eager 7.364 (0.90)<br>chain-cutlass 7.609 (0.87) | torch-eager 7.42 (0.89) | torch-compile 7.557 (0.88) | torch-compile-cg 7.558 (0.88) | torch-compile-ro 7.359 (0.90) | triton 7.425 (0.89) |
| epi_4096 | rtx-4090 | 4.581 | **chain-lt 4.843 (0.95)**<br>chain-cutlass 4.905 (0.93)<br>chain-eager 4.971 (0.92) | torch-eager 5.141 (0.89) | torch-compile 4.928 (0.93) | torch-compile-cg 4.922 (0.93) | torch-compile-ro 4.924 (0.93) | triton 6.022 (0.76) |
| epi_4096 | h100 | 4.408 | chain-lt 3.935 (1.12)<br>chain-eager 3.971 (1.11)<br>chain-cutlass 4.532 (0.97) | torch-eager 4.073 (1.08) | torch-compile 4.044 (1.09) | **torch-compile-cg 3.912 (1.13)** | torch-compile-ro 4.005 (1.10) | triton 4.583 (0.96) |
| mm_16x2048x7168 | rtx-5080 | 0.0799 | **cublas 0.0799 (1.00)**<br>cutlass-tuned 0.3871 (0.21)<br>cutlass 0.5284 (0.15) |  |  |  |  | triton 0.1732 (0.46) |
| mm_16x2048x7168 | rtx-3090 | 0.0901 | **cublas 0.08755 (1.03)**<br>cutlass-tuned 0.2171 (0.42)<br>cutlass 0.7045 (0.13) |  |  |  |  | not run |
| mm_16x2048x7168 | a100 | 0.0963 | **cublas 0.0942 (1.02)**<br>cutlass-tuned 0.3635 (0.26)<br>cutlass 1.359 (0.07) |  |  |  |  | triton 0.3727 (0.26) |
| mm_16x2048x7168 | rtx-4090 | 0.1305 | **cublas 0.1137 (1.15)**<br>cutlass-tuned 0.3041 (0.43)<br>cutlass 1.111 (0.12) |  |  |  |  | triton 0.3328 (0.39) |
| mm_16x2048x7168 | h100 | 0.1017 | **cublas 0.0639 (1.59)**<br>cutlass-tuned 0.2859 (0.36)<br>cutlass 0.7714 (0.13) |  |  |  |  | triton 0.2753 (0.37) |
| mm_4096x4096x4096 | rtx-5080 | 3.818 | **cublas 3.678 (1.04)**<br>cutlass 3.735 (1.02)<br>cutlass-tuned 9.09 (0.42) |  |  |  |  | triton 4.479 (0.85) |
| mm_4096x4096x4096 | rtx-3090 | 7.511 | **cublas 6.804 (1.10)**<br>cutlass 7.014 (1.07)<br>cutlass-tuned 18.73 (0.40) |  |  |  |  | not run |
| mm_4096x4096x4096 | rtx-4090 | 4.673 | **cublas 4.837 (0.97)**<br>cutlass 4.902 (0.95)<br>cutlass-tuned 13.32 (0.35) |  |  |  |  | triton 5.905 (0.79) |
| mm_4096x4096x4096 | h100 | 4.84 | **cublas 4.134 (1.17)**<br>cutlass 4.734 (1.02)<br>cutlass-tuned 10.77 (0.45) |  |  |  |  | triton 4.99 (0.97) |
| pj_m128_n28672_k4096 | rtx-5080 | 0.8019 | **lt 0.7501 (1.07)**<br>lt-fresh 0.7555 (1.06)<br>cublas 0.7966 (1.01)<br>cublas-fresh 0.7976 (1.01) | torch-eager 0.8109 (0.99) | torch-compile 0.8239 (0.97) | torch-compile-cg 0.8288 (0.97) | torch-compile-ro 0.8287 (0.97) | triton 1.087 (0.74) |
| pj_m128_n28672_k4096 | rtx-3090 | 1.202 | cublas 2.994 (0.40)<br>lt 2.994 (0.40)<br>cublas-fresh 2.995 (0.40)<br>lt-fresh 2.995 (0.40) | torch-eager 2.07 (0.58) | torch-compile 1.508 (0.80) | torch-compile-cg 1.516 (0.79) | **torch-compile-ro 1.223 (0.98)** | triton 2.526 (0.48) |
| pj_m128_n28672_k4096 | a100 | 1.616 | **cublas 1.739 (0.93)**<br>cublas-fresh 1.739 (0.93)<br>lt-fresh 1.739 (0.93)<br>lt 1.74 (0.93) | torch-eager 1.742 (0.93) | torch-compile 2.224 (0.73) | torch-compile-cg 2.223 (0.73) | torch-compile-ro 1.74 (0.93) | triton 1.876 (0.86) |
| pj_m128_n28672_k4096 | rtx-4090 | 1.105 | **lt 1.18 (0.94)**<br>lt-fresh 1.181 (0.94)<br>cublas 1.197 (0.92)<br>cublas-fresh 1.198 (0.92) | torch-eager 1.199 (0.92) | torch-compile 1.197 (0.92) | torch-compile-cg 1.194 (0.93) | torch-compile-ro 1.195 (0.93) | triton 1.41 (0.78) |
| pj_m128_n28672_k4096 | h100 | 0.9701 | lt 1.027 (0.94)<br>cublas 1.028 (0.94)<br>lt-fresh 1.033 (0.94)<br>cublas-fresh 1.038 (0.93) | torch-eager 1.042 (0.93) | torch-compile 0.9249 (1.05) | torch-compile-cg 0.9245 (1.05) | **torch-compile-ro 0.918 (1.06)** | triton 1.252 (0.77) |
| pj_m128_n4096_k14336 | rtx-5080 | 0.4447 | **lt-fresh 0.4071 (1.09)**<br>cublas-fresh 0.4079 (1.09)<br>lt 0.4107 (1.08)<br>cublas 0.4112 (1.08) | torch-eager 0.4178 (1.06) | torch-compile 0.4282 (1.04) | torch-compile-cg 0.4354 (1.02) | torch-compile-ro 0.4356 (1.02) | triton 0.61 (0.73) |
| pj_m128_n4096_k14336 | rtx-3090 | 0.7032 | lt 0.8561 (0.82)<br>lt-fresh 0.8571 (0.82)<br>cublas-fresh 0.8612 (0.82)<br>cublas 0.8622 (0.82) | torch-eager 0.725 (0.97) | **torch-compile 0.684 (1.03)** | torch-compile-cg 0.684 (1.03) | torch-compile-ro 0.684 (1.03) | triton 1.138 (0.62) |
| pj_m128_n4096_k14336 | a100 | 1.101 | lt-fresh 0.9841 (1.12)<br>cublas-fresh 0.9851 (1.12)<br>cublas 0.9861 (1.12)<br>lt 0.9861 (1.12) | **torch-eager 0.8591 (1.28)** | torch-compile 1.094 (1.01) | torch-compile-cg 1.094 (1.01) | torch-compile-ro 1.094 (1.01) | triton 1.435 (0.77) |
| pj_m128_n4096_k14336 | rtx-4090 | 0.6412 | cublas 0.64 (1.00)<br>lt 0.64 (1.00)<br>cublas-fresh 0.641 (1.00)<br>lt-fresh 0.641 (1.00) | **torch-eager 0.64 (1.00)** | torch-compile 0.64 (1.00) | torch-compile-cg 0.64 (1.00) | torch-compile-ro 0.64 (1.00) | triton 0.6973 (0.92) |
| pj_m128_n4096_k14336 | h100 | 0.5062 | cublas 0.4401 (1.15)<br>lt 0.4401 (1.15)<br>lt-fresh 0.4449 (1.14)<br>cublas-fresh 0.4451 (1.14) | torch-eager 0.4311 (1.17) | torch-compile 0.4256 (1.19) | torch-compile-cg 0.43 (1.18) | **torch-compile-ro 0.4187 (1.21)** | triton 0.5723 (0.88) |
| pj_m128_n4096_k4096 | rtx-5080 | 0.1368 | **lt-fresh 0.1383 (0.99)**<br>cublas-fresh 0.139 (0.98)<br>lt 0.1459 (0.94)<br>cublas 0.1467 (0.93) | torch-eager 0.1509 (0.91) | torch-compile 0.1616 (0.85) | torch-compile-cg 0.1674 (0.82) | torch-compile-ro 0.1676 (0.82) | triton 0.1943 (0.70) |
| pj_m128_n4096_k4096 | rtx-3090 | 0.2449 | **lt 0.2222 (1.10)**<br>lt-fresh 0.2222 (1.10)<br>cublas 0.2447 (1.00)<br>cublas-fresh 0.2447 (1.00) | torch-eager 0.2284 (1.07) | torch-compile 0.2284 (1.07) | torch-compile-cg 0.2284 (1.07) | torch-compile-ro 0.2284 (1.07) | triton 0.3235 (0.76) |
| pj_m128_n4096_k4096 | a100 | 0.3488 | **cublas 0.2959 (1.18)**<br>cublas-fresh 0.297 (1.17)<br>lt-fresh 0.297 (1.17)<br>lt 0.298 (1.17) | torch-eager 0.3338 (1.05) | torch-compile 0.3328 (1.05) | torch-compile-cg 0.3318 (1.05) | torch-compile-ro 0.3318 (1.05) | triton 0.4168 (0.84) |
| pj_m128_n4096_k4096 | rtx-4090 | 0.2367 | **lt-fresh 0.2007 (1.18)**<br>lt 0.2037 (1.16)<br>cublas-fresh 0.212 (1.12)<br>cublas 0.2131 (1.11) | torch-eager 0.214 (1.11) | torch-compile 0.214 (1.11) | torch-compile-cg 0.213 (1.11) | torch-compile-ro 0.213 (1.11) | triton 0.2058 (1.15) |
| pj_m128_n4096_k4096 | h100 | 0.1592 | **lt-fresh 0.1222 (1.30)**<br>cublas-fresh 0.1227 (1.30)<br>lt 0.1281 (1.24)<br>cublas 0.1292 (1.23) | torch-eager 0.1305 (1.22) | torch-compile 0.1283 (1.24) | torch-compile-cg 0.1267 (1.26) | torch-compile-ro 0.1268 (1.26) | triton 0.1671 (0.95) |
| pj_m128_n6144_k4096 | rtx-5080 | 0.1953 | **lt-fresh 0.1826 (1.07)**<br>cublas-fresh 0.1837 (1.06)<br>lt 0.1879 (1.04)<br>cublas 0.1885 (1.04) | torch-eager 0.1934 (1.01) | torch-compile 0.2038 (0.96) | torch-compile-cg 0.2102 (0.93) | torch-compile-ro 0.2099 (0.93) | triton 0.3118 (0.63) |
| pj_m128_n6144_k4096 | rtx-3090 | 0.3378 | lt 0.3328 (1.01)<br>lt-fresh 0.3338 (1.01)<br>cublas 0.3666 (0.92)<br>cublas-fresh 0.3676 (0.92) | torch-eager 0.3226 (1.05) | torch-compile 0.3226 (1.05) | **torch-compile-cg 0.3205 (1.05)** | torch-compile-ro 0.3226 (1.05) | triton 0.4874 (0.69) |
| pj_m128_n6144_k4096 | a100 | 0.6778 | **lt 0.4383 (1.55)**<br>lt-fresh 0.4393 (1.54)<br>cublas-fresh 0.5284 (1.28)<br>cublas 0.5294 (1.28) | torch-eager 0.5775 (1.17) | torch-compile 0.5376 (1.26) | torch-compile-cg 0.5366 (1.26) | torch-compile-ro 0.5765 (1.18) | triton 0.4495 (1.51) |
| pj_m128_n6144_k4096 | rtx-4090 | 0.3358 | cublas 0.3011 (1.12)<br>lt 0.3011 (1.12)<br>lt-fresh 0.3021 (1.11)<br>cublas-fresh 0.3031 (1.11) | torch-eager 0.3011 (1.12) | torch-compile 0.3011 (1.12) | **torch-compile-cg 0.3 (1.12)** | torch-compile-ro 0.3 (1.12) | triton 0.374 (0.90) |
| pj_m128_n6144_k4096 | h100 | 0.2288 | lt-fresh 0.1911 (1.20)<br>lt 0.1915 (1.19)<br>cublas 0.1916 (1.19)<br>cublas-fresh 0.192 (1.19) | torch-eager 0.1914 (1.20) | torch-compile 0.1916 (1.19) | torch-compile-cg 0.1902 (1.20) | **torch-compile-ro 0.1901 (1.20)** | triton 0.2942 (0.78) |
| pj_m16_n18432_k7168 | rtx-5080 | 0.596 | **lt-fresh 0.5904 (1.01)**<br>cublas-fresh 0.5926 (1.01)<br>lt 0.6198 (0.96)<br>cublas 0.6207 (0.96) | torch-eager 0.6244 (0.95) | torch-compile 0.6268 (0.95) | torch-compile-cg 0.635 (0.94) | torch-compile-ro 0.6437 (0.93) | triton 0.6373 (0.94) |
| pj_m16_n18432_k7168 | rtx-3090 | 0.64 | cublas 0.6646 (0.96)<br>lt 0.6646 (0.96)<br>cublas-fresh 0.6646 (0.96)<br>lt-fresh 0.6646 (0.96) | torch-eager 0.6738 (0.95) | **torch-compile 0.6124 (1.05)** | torch-compile-cg 0.6277 (1.02) | torch-compile-ro 0.6226 (1.03) | triton 0.8325 (0.77) |
| pj_m16_n18432_k7168 | a100 | 0.4718 | lt 0.6113 (0.77)<br>lt-fresh 0.6113 (0.77)<br>cublas 0.64 (0.74)<br>cublas-fresh 0.641 (0.74) | torch-eager 0.6543 (0.72) | **torch-compile 0.4291 (1.10)** | torch-compile-cg 0.4547 (1.04) | torch-compile-ro 0.5263 (0.90) | triton 0.726 (0.65) |
| pj_m16_n18432_k7168 | rtx-4090 | 0.7444 | lt-fresh 0.7148 (1.04)<br>lt 0.7311 (1.02)<br>cublas-fresh 0.7424 (1.00)<br>cublas 0.7516 (0.99) | torch-eager 0.7885 (0.94) | **torch-compile 0.7127 (1.04)** | torch-compile-cg 0.7291 (1.02) | torch-compile-ro 0.7557 (0.99) | triton 0.9769 (0.76) |
| pj_m16_n18432_k7168 | h100 | 0.3194 | lt 0.365 (0.88)<br>lt-fresh 0.3946 (0.81)<br>cublas 0.4033 (0.79)<br>cublas-fresh 0.4258 (0.75) | torch-eager 0.4254 (0.75) | **torch-compile 0.3365 (0.95)** | torch-compile-cg 0.3365 (0.95) | torch-compile-ro 0.4225 (0.76) | triton 0.5628 (0.57) |
| pj_m16_n2048_k7168 | rtx-5080 | 0.03463 | **lt-fresh 0.052 (0.67)**<br>cublas-fresh 0.0529 (0.65)<br>lt 0.0815 (0.42)<br>cublas 0.0825 (0.42) | torch-eager 0.0851 (0.41) | torch-compile 0.0948 (0.37) | torch-compile-cg 0.1029 (0.34) | torch-compile-ro 0.1018 (0.34) | triton 0.1736 (0.20) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1053 | **lt 0.084 (1.25)**<br>lt-fresh 0.085 (1.24)<br>cublas 0.0891 (1.18)<br>cublas-fresh 0.0891 (1.18) | torch-eager 0.0942 (1.12) | torch-compile 0.0891 (1.18) | torch-compile-cg 0.0881 (1.19) | torch-compile-ro 0.0881 (1.19) | triton 0.2171 (0.48) |
| pj_m16_n2048_k7168 | a100 | 0.1253 | lt-fresh 0.1004 (1.25)<br>cublas-fresh 0.1034 (1.21)<br>cublas 0.1065 (1.18)<br>lt 0.1065 (1.18) | **torch-eager 0.0952 (1.32)** | torch-compile 0.1096 (1.14) | torch-compile-cg 0.1075 (1.17) | torch-compile-ro 0.1065 (1.18) | triton 0.3717 (0.34) |
| pj_m16_n2048_k7168 | rtx-4090 | 0.08037 | **cublas-fresh 0.0573 (1.40)**<br>lt-fresh 0.0573 (1.40)<br>cublas 0.1106 (0.73)<br>lt 0.1106 (0.73) | torch-eager 0.1134 (0.71) | torch-compile 0.1126 (0.71) | torch-compile-cg 0.1106 (0.73) | torch-compile-ro 0.1106 (0.73) | triton 0.3328 (0.24) |
| pj_m16_n2048_k7168 | h100 | 0.05923 | lt 0.051 (1.16)<br>lt-fresh 0.0525 (1.13)<br>cublas 0.0554 (1.07)<br>cublas-fresh 0.0557 (1.06) | torch-eager 0.0502 (1.18) | torch-compile 0.05 (1.18) | **torch-compile-cg 0.0487 (1.22)** | torch-compile-ro 0.0488 (1.21) | triton 0.1995 (0.30) |
| pj_m16_n28672_k4096 | rtx-5080 | 0.5377 | **lt-fresh 0.528 (1.02)**<br>lt 0.5534 (0.97)<br>cublas-fresh 1.865 (0.29)<br>cublas 1.931 (0.28) | torch-eager 1.931 (0.28) | torch-compile 0.5589 (0.96) | torch-compile-cg 0.5648 (0.95) | torch-compile-ro 1.955 (0.28) | triton 0.5704 (0.94) |
| pj_m16_n28672_k4096 | rtx-3090 | 0.5816 | cublas 0.5857 (0.99)<br>lt 0.5857 (0.99)<br>cublas-fresh 0.5857 (0.99)<br>lt-fresh 0.5857 (0.99) | torch-eager 0.5632 (1.03) | **torch-compile 0.5448 (1.07)** | torch-compile-cg 0.5458 (1.07) | torch-compile-ro 0.555 (1.05) | triton 0.6799 (0.86) |
| pj_m16_n28672_k4096 | a100 | 0.4755 | cublas 0.5786 (0.82)<br>lt 0.5796 (0.82)<br>cublas-fresh 0.5796 (0.82)<br>lt-fresh 0.5796 (0.82) | torch-eager 0.5827 (0.82) | **torch-compile 0.4055 (1.17)** | torch-compile-cg 0.4168 (1.14) | torch-compile-ro 0.5038 (0.94) | triton 0.6134 (0.78) |
| pj_m16_n28672_k4096 | rtx-4090 | 0.6595 | **lt-fresh 0.638 (1.03)**<br>cublas-fresh 0.639 (1.03)<br>lt 0.6451 (1.02)<br>cublas 0.6472 (1.02) | torch-eager 0.6821 (0.97) | torch-compile 0.6492 (1.02) | torch-compile-cg 0.6689 (0.99) | torch-compile-ro 0.6554 (1.01) | triton 0.6677 (0.99) |
| pj_m16_n28672_k4096 | h100 | 0.2744 | lt 0.3507 (0.78)<br>lt-fresh 0.367 (0.75)<br>cublas 0.3731 (0.74)<br>cublas-fresh 0.3863 (0.71) | torch-eager 0.3819 (0.72) | torch-compile 0.2616 (1.05) | **torch-compile-cg 0.2601 (1.06)** | torch-compile-ro 0.3819 (0.72) | triton 0.3674 (0.75) |
| pj_m16_n4096_k14336 | rtx-5080 | 0.2339 | lt-fresh 0.32 (0.73)<br>cublas-fresh 0.3219 (0.73)<br>lt 0.3354 (0.70)<br>cublas 0.3358 (0.70) | torch-eager 0.3367 (0.69) | **torch-compile 0.294 (0.80)** | torch-compile-cg 0.2975 (0.79) | torch-compile-ro 0.3531 (0.66) | triton 0.3236 (0.72) |
| pj_m16_n4096_k14336 | rtx-3090 | 0.2968 | lt 0.2918 (1.02)<br>lt-fresh 0.2929 (1.01)<br>cublas 0.2959 (1.00)<br>cublas-fresh 0.2959 (1.00) | torch-eager 0.2929 (1.01) | **torch-compile 0.2755 (1.08)** | torch-compile-cg 0.2755 (1.08) | torch-compile-ro 0.2918 (1.02) | triton 0.4741 (0.63) |
| pj_m16_n4096_k14336 | a100 | 0.2708 | lt-fresh 0.3164 (0.86)<br>lt 0.3236 (0.84)<br>cublas 0.3297 (0.82)<br>cublas-fresh 0.3297 (0.82) | torch-eager 0.3144 (0.86) | torch-compile 0.3123 (0.87) | **torch-compile-cg 0.3103 (0.87)** | torch-compile-ro 0.3103 (0.87) | triton 0.7363 (0.37) |
| pj_m16_n4096_k14336 | rtx-4090 | 0.3697 | **cublas-fresh 0.3471 (1.07)**<br>lt-fresh 0.3471 (1.07)<br>lt 0.3604 (1.03)<br>cublas 0.3615 (1.02) | torch-eager 0.3656 (1.01) | torch-compile 0.3625 (1.02) | torch-compile-cg 0.3615 (1.02) | torch-compile-ro 0.3666 (1.01) | triton 0.6605 (0.56) |
| pj_m16_n4096_k14336 | h100 | 0.1593 | lt 0.1811 (0.88)<br>lt-fresh 0.1829 (0.87)<br>cublas-fresh 0.1903 (0.84)<br>cublas 0.2087 (0.76) | torch-eager 0.1927 (0.83) | **torch-compile 0.1694 (0.94)** | torch-compile-cg 0.1699 (0.94) | torch-compile-ro 0.1907 (0.84) | triton 0.4787 (0.33) |
| pj_m16_n4096_k4096 | rtx-5080 | 0.03876 | **lt-fresh 0.0591 (0.66)**<br>cublas-fresh 0.06 (0.65)<br>lt 0.0981 (0.40)<br>cublas 0.0989 (0.39) | torch-eager 0.1088 (0.36) | torch-compile 0.0989 (0.39) | torch-compile-cg 0.1051 (0.37) | torch-compile-ro 0.1191 (0.33) | triton 0.1107 (0.35) |
| pj_m16_n4096_k4096 | rtx-3090 | 0.1206 | lt 0.1025 (1.18)<br>cublas 0.1034 (1.17)<br>cublas-fresh 0.1034 (1.17)<br>lt-fresh 0.1034 (1.17) | torch-eager 0.1085 (1.11) | torch-compile 0.086 (1.40) | **torch-compile-cg 0.0851 (1.42)** | torch-compile-ro 0.1014 (1.19) | triton 0.1372 (0.88) |
| pj_m16_n4096_k4096 | a100 | 0.1188 | lt-fresh 0.1055 (1.13)<br>cublas-fresh 0.1065 (1.12)<br>lt 0.1075 (1.11)<br>cublas 0.1085 (1.10) | torch-eager 0.1085 (1.10) | torch-compile 0.1075 (1.11) | **torch-compile-cg 0.1044 (1.14)** | torch-compile-ro 0.1055 (1.13) | triton 0.2171 (0.55) |
| pj_m16_n4096_k4096 | rtx-4090 | 0.08238 | **lt-fresh 0.0645 (1.28)**<br>cublas-fresh 0.0686 (1.20)<br>lt 0.128 (0.64)<br>cublas 0.1382 (0.60) | torch-eager 0.1393 (0.59) | torch-compile 0.1188 (0.69) | torch-compile-cg 0.1188 (0.69) | torch-compile-ro 0.1372 (0.60) | triton 0.1946 (0.42) |
| pj_m16_n4096_k4096 | h100 | 0.053 | lt 0.0508 (1.04)<br>lt-fresh 0.0511 (1.04)<br>cublas-fresh 0.0918 (0.58)<br>cublas 0.092 (0.58) | torch-eager 0.09 (0.59) | **torch-compile 0.049 (1.08)** | torch-compile-cg 0.0493 (1.08) | torch-compile-ro 0.0902 (0.59) | triton 0.1539 (0.34) |
| pj_m16_n6144_k4096 | rtx-5080 | 0.08607 | **lt-fresh 0.0899 (0.96)**<br>cublas-fresh 0.1003 (0.86)<br>lt 0.1306 (0.66)<br>cublas 0.1427 (0.60) | torch-eager 0.1436 (0.60) | torch-compile 0.1377 (0.63) | torch-compile-cg 0.1441 (0.60) | torch-compile-ro 0.1618 (0.53) | triton 0.149 (0.58) |
| pj_m16_n6144_k4096 | rtx-3090 | 0.1542 | lt 0.132 (1.17)<br>lt-fresh 0.1321 (1.17)<br>cublas-fresh 0.1444 (1.07)<br>cublas 0.1453 (1.06) | torch-eager 0.1464 (1.05) | **torch-compile 0.1229 (1.25)** | torch-compile-cg 0.1239 (1.24) | torch-compile-ro 0.1454 (1.06) | triton 0.1567 (0.98) |
| pj_m16_n6144_k4096 | a100 | 0.1427 | lt-fresh 0.1434 (1.00)<br>lt 0.1454 (0.98)<br>cublas-fresh 0.1597 (0.89)<br>cublas 0.1618 (0.88) | torch-eager 0.1475 (0.97) | torch-compile 0.1178 (1.21) | **torch-compile-cg 0.1167 (1.22)** | torch-compile-ro 0.1444 (0.99) | triton 0.2181 (0.65) |
| pj_m16_n6144_k4096 | rtx-4090 | 0.1575 | **cublas-fresh 0.1249 (1.26)**<br>lt-fresh 0.1362 (1.16)<br>lt 0.1843 (0.85)<br>cublas 0.1875 (0.84) | torch-eager 0.1925 (0.82) | torch-compile 0.1772 (0.89) | torch-compile-cg 0.1761 (0.89) | torch-compile-ro 0.1915 (0.82) | triton 0.1956 (0.81) |
| pj_m16_n6144_k4096 | h100 | 0.06957 | lt-fresh 0.0755 (0.92)<br>lt 0.095 (0.73)<br>cublas-fresh 0.1079 (0.64)<br>cublas 0.116 (0.60) | torch-eager 0.1136 (0.61) | **torch-compile 0.0565 (1.23)** | torch-compile-cg 0.0568 (1.22) | torch-compile-ro 0.1115 (0.62) | triton 0.1472 (0.47) |
| pj_m1_n18432_k7168 | rtx-5080 | 0.5869 | **cublas-fresh 0.6173 (0.95)**<br>lt-fresh 0.6188 (0.95)<br>cublas 0.62 (0.95)<br>lt 0.6206 (0.95) | torch-eager 0.6219 (0.94) | torch-compile 0.6252 (0.94) | torch-compile-cg 0.6294 (0.93) | torch-compile-ro 0.6428 (0.91) | triton 0.6574 (0.89) |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6127 | **lt 0.6083 (1.01)**<br>lt-fresh 0.6083 (1.01)<br>cublas 0.6093 (1.01)<br>cublas-fresh 0.6093 (1.01) | torch-eager 0.6533 (0.94) | torch-compile 0.6103 (1.00) | torch-compile-cg 0.6093 (1.01) | torch-compile-ro 0.6124 (1.00) | triton 0.8745 (0.70) |
| pj_m1_n18432_k7168 | a100 | 0.4274 | **lt-fresh 0.3953 (1.08)**<br>lt 0.3983 (1.07)<br>cublas-fresh 0.4106 (1.04)<br>cublas 0.4311 (0.99) | torch-eager 0.4454 (0.96) | torch-compile 0.4004 (1.07) | torch-compile-cg 0.3983 (1.07) | torch-compile-ro 0.4321 (0.99) | triton 0.7516 (0.57) |
| pj_m1_n18432_k7168 | rtx-4090 | 0.7272 | **lt-fresh 0.6195 (1.17)**<br>lt 0.6861 (1.06)<br>cublas-fresh 0.767 (0.95)<br>cublas 0.7762 (0.94) | torch-eager 0.8131 (0.89) | torch-compile 0.6666 (1.09) | torch-compile-cg 0.6656 (1.09) | torch-compile-ro 0.7885 (0.92) | triton 0.9759 (0.75) |
| pj_m1_n18432_k7168 | h100 | 0.1992 | lt-fresh 0.2028 (0.98)<br>cublas-fresh 0.2029 (0.98)<br>lt 0.207 (0.96)<br>cublas 0.2117 (0.94) | torch-eager 0.221 (0.90) | torch-compile 0.1976 (1.01) | **torch-compile-cg 0.1924 (1.04)** | torch-compile-ro 0.2171 (0.92) | triton 0.5138 (0.39) |
| pj_m1_n2048_k7168 | rtx-5080 | 0.02402 | **lt-fresh 0.03 (0.80)**<br>cublas-fresh 0.0309 (0.78)<br>lt 0.0779 (0.31)<br>cublas 0.0787 (0.31) | torch-eager 0.0801 (0.30) | torch-compile 0.0914 (0.26) | torch-compile-cg 0.0979 (0.25) | torch-compile-ro 0.0975 (0.25) | triton 0.2213 (0.11) |
| pj_m1_n2048_k7168 | rtx-3090 | 0.08661 | **lt-fresh 0.0748 (1.16)**<br>lt 0.0768 (1.13)<br>cublas-fresh 0.0901 (0.96)<br>cublas 0.0911 (0.95) | torch-eager 0.0901 (0.96) | torch-compile 0.0809 (1.07) | torch-compile-cg 0.0809 (1.07) | torch-compile-ro 0.0901 (0.96) | triton 0.2191 (0.40) |
| pj_m1_n2048_k7168 | a100 | 0.08678 | **cublas-fresh 0.0532 (1.63)**<br>lt-fresh 0.0532 (1.63)<br>cublas 0.0635 (1.37)<br>lt 0.0686 (1.27) | torch-eager 0.0594 (1.46) | torch-compile 0.0686 (1.27) | torch-compile-cg 0.0676 (1.28) | torch-compile-ro 0.0584 (1.49) | triton 0.3717 (0.23) |
| pj_m1_n2048_k7168 | rtx-4090 | 0.05095 | **lt-fresh 0.0553 (0.92)**<br>cublas-fresh 0.084 (0.61)<br>lt 0.0942 (0.54)<br>cublas 0.1311 (0.39) | torch-eager 0.1362 (0.37) | torch-compile 0.1044 (0.49) | torch-compile-cg 0.1034 (0.49) | torch-compile-ro 0.1331 (0.38) | triton 0.3369 (0.15) |
| pj_m1_n2048_k7168 | h100 | 0.0379 | **cublas-fresh 0.0228 (1.66)**<br>lt-fresh 0.024 (1.58)<br>lt 0.0287 (1.32)<br>cublas 0.0312 (1.21) | torch-eager 0.0306 (1.24) | torch-compile 0.0302 (1.25) | torch-compile-cg 0.0292 (1.30) | torch-compile-ro 0.031 (1.22) | triton 0.1941 (0.20) |
| pj_m1_n28672_k4096 | rtx-5080 | 0.5219 | **lt-fresh 0.5063 (1.03)**<br>cublas-fresh 0.5267 (0.99)<br>lt 0.5441 (0.96)<br>cublas 0.5619 (0.93) | torch-eager 0.5641 (0.93) | torch-compile 0.5562 (0.94) | torch-compile-cg 0.56 (0.93) | torch-compile-ro 0.5835 (0.89) | triton 0.5902 (0.88) |
| pj_m1_n28672_k4096 | rtx-3090 | 0.5487 | cublas 0.5386 (1.02)<br>lt 0.5386 (1.02)<br>cublas-fresh 0.5386 (1.02)<br>lt-fresh 0.5386 (1.02) | torch-eager 0.5386 (1.02) | torch-compile 0.5437 (1.01) | torch-compile-cg 0.5437 (1.01) | **torch-compile-ro 0.5366 (1.02)** | triton 0.7301 (0.75) |
| pj_m1_n28672_k4096 | a100 | 0.3932 | lt-fresh 0.3656 (1.08)<br>lt 0.3748 (1.05)<br>cublas 0.4669 (0.84)<br>cublas-fresh 0.4669 (0.84) | torch-eager 0.511 (0.77) | torch-compile 0.3564 (1.10) | **torch-compile-cg 0.3512 (1.12)** | torch-compile-ro 0.5007 (0.79) | triton 0.6615 (0.59) |
| pj_m1_n28672_k4096 | rtx-4090 | 0.6547 | lt-fresh 0.638 (1.03)<br>lt 0.6431 (1.02)<br>cublas-fresh 0.6748 (0.97)<br>cublas 0.6799 (0.96) | torch-eager 0.727 (0.90) | torch-compile 0.6001 (1.09) | **torch-compile-cg 0.599 (1.09)** | torch-compile-ro 0.6954 (0.94) | triton 0.7404 (0.88) |
| pj_m1_n28672_k4096 | h100 | 0.1813 | **cublas 0.1801 (1.01)**<br>cublas-fresh 0.1808 (1.00)<br>lt-fresh 0.182 (1.00)<br>lt 0.1823 (0.99) | torch-eager 0.1929 (0.94) | torch-compile 0.1928 (0.94) | torch-compile-cg 0.1924 (0.94) | torch-compile-ro 0.1898 (0.96) | triton 0.3708 (0.49) |
| pj_m1_n4096_k14336 | rtx-5080 | 0.2238 | **lt-fresh 0.2342 (0.96)**<br>cublas-fresh 0.235 (0.95)<br>lt 0.278 (0.81)<br>cublas 0.2781 (0.80) | torch-eager 0.2808 (0.80) | torch-compile 0.2918 (0.77) | torch-compile-cg 0.2977 (0.75) | torch-compile-ro 0.2985 (0.75) | triton 0.4268 (0.52) |
| pj_m1_n4096_k14336 | rtx-3090 | 0.2852 | lt 0.2765 (1.03)<br>lt-fresh 0.2775 (1.03)<br>cublas 0.2785 (1.02)<br>cublas-fresh 0.2785 (1.02) | torch-eager 0.2775 (1.03) | torch-compile 0.2826 (1.01) | torch-compile-cg 0.2806 (1.02) | **torch-compile-ro 0.2755 (1.04)** | triton 0.5356 (0.53) |
| pj_m1_n4096_k14336 | a100 | 0.2133 | **cublas-fresh 0.1833 (1.16)**<br>lt-fresh 0.1833 (1.16)<br>lt 0.1915 (1.11)<br>cublas 0.2007 (1.06) | torch-eager 0.2017 (1.06) | torch-compile 0.1925 (1.11) | torch-compile-cg 0.1864 (1.14) | torch-compile-ro 0.1997 (1.07) | triton 0.7383 (0.29) |
| pj_m1_n4096_k14336 | rtx-4090 | 0.3351 | **lt-fresh 0.2314 (1.45)**<br>lt 0.3062 (1.09)<br>cublas-fresh 0.343 (0.98)<br>cublas 0.3615 (0.93) | torch-eager 0.3738 (0.90) | torch-compile 0.3256 (1.03) | torch-compile-cg 0.3246 (1.03) | torch-compile-ro 0.3738 (0.90) | triton 0.6707 (0.50) |
| pj_m1_n4096_k14336 | h100 | 0.1008 | **cublas-fresh 0.0835 (1.21)**<br>lt-fresh 0.0844 (1.19)<br>lt 0.0935 (1.08)<br>cublas 0.0937 (1.08) | torch-eager 0.0933 (1.08) | torch-compile 0.1043 (0.97) | torch-compile-cg 0.0914 (1.10) | torch-compile-ro 0.0936 (1.08) | triton 0.4719 (0.21) |
| pj_m1_n4096_k4096 | rtx-5080 | 0.03005 | **lt-fresh 0.0238 (1.26)**<br>cublas-fresh 0.0285 (1.05)<br>lt 0.0861 (0.35)<br>cublas 0.0897 (0.34) | torch-eager 0.0921 (0.33) | torch-compile 0.0999 (0.30) | torch-compile-cg 0.1066 (0.28) | torch-compile-ro 0.1124 (0.27) | triton 0.1446 (0.21) |
| pj_m1_n4096_k4096 | rtx-3090 | 0.09665 | **lt 0.086 (1.12)**<br>lt-fresh 0.086 (1.12)<br>cublas-fresh 0.0911 (1.06)<br>cublas 0.0932 (1.04) | torch-eager 0.0922 (1.05) | torch-compile 0.0911 (1.06) | torch-compile-cg 0.0891 (1.08) | torch-compile-ro 0.0911 (1.06) | triton 0.17 (0.57) |
| pj_m1_n4096_k4096 | a100 | 0.09081 | **cublas-fresh 0.0594 (1.53)**<br>lt-fresh 0.0594 (1.53)<br>cublas 0.0727 (1.25)<br>lt 0.0737 (1.23) | torch-eager 0.0737 (1.23) | torch-compile 0.0707 (1.28) | torch-compile-cg 0.0707 (1.28) | torch-compile-ro 0.0707 (1.28) | triton 0.2181 (0.42) |
| pj_m1_n4096_k4096 | rtx-4090 | 0.04615 | **lt-fresh 0.0348 (1.33)**<br>cublas-fresh 0.0358 (1.29)<br>lt 0.1075 (0.43)<br>cublas 0.1116 (0.41) | torch-eager 0.1167 (0.40) | torch-compile 0.1157 (0.40) | torch-compile-cg 0.1137 (0.41) | torch-compile-ro 0.1147 (0.40) | triton 0.2007 (0.23) |
| pj_m1_n4096_k4096 | h100 | 0.04356 | **cublas-fresh 0.0222 (1.96)**<br>lt-fresh 0.0222 (1.96)<br>cublas 0.0299 (1.46)<br>lt 0.0299 (1.46) | torch-eager 0.0298 (1.46) | torch-compile 0.0335 (1.30) | torch-compile-cg 0.0327 (1.33) | torch-compile-ro 0.03 (1.45) | triton 0.1506 (0.29) |
| pj_m1_n6144_k4096 | rtx-5080 | 0.06874 | **lt-fresh 0.0539 (1.28)**<br>cublas-fresh 0.0547 (1.26)<br>lt 0.1261 (0.55)<br>cublas 0.1273 (0.54) | torch-eager 0.1295 (0.53) | torch-compile 0.1367 (0.50) | torch-compile-cg 0.1424 (0.48) | torch-compile-ro 0.1465 (0.47) | triton 0.1635 (0.42) |
| pj_m1_n6144_k4096 | rtx-3090 | 0.1355 | lt 0.1239 (1.09)<br>lt-fresh 0.1239 (1.09)<br>cublas 0.1403 (0.97)<br>cublas-fresh 0.1403 (0.97) | torch-eager 0.1393 (0.97) | **torch-compile 0.1219 (1.11)** | torch-compile-cg 0.1219 (1.11) | torch-compile-ro 0.1372 (0.99) | triton 0.2212 (0.61) |
| pj_m1_n6144_k4096 | a100 | 0.1129 | **cublas-fresh 0.085 (1.33)**<br>lt-fresh 0.085 (1.33)<br>lt 0.0973 (1.16)<br>cublas 0.0983 (1.15) | torch-eager 0.0993 (1.14) | torch-compile 0.0973 (1.16) | torch-compile-cg 0.0911 (1.24) | torch-compile-ro 0.0922 (1.23) | triton 0.2222 (0.51) |
| pj_m1_n6144_k4096 | rtx-4090 | 0.1011 | **lt-fresh 0.0696 (1.45)**<br>lt 0.1444 (0.70)<br>cublas-fresh 0.1485 (0.68)<br>cublas 0.1894 (0.53) | torch-eager 0.1925 (0.53) | torch-compile 0.1567 (0.65) | torch-compile-cg 0.1546 (0.65) | torch-compile-ro 0.1915 (0.53) | triton 0.2191 (0.46) |
| pj_m1_n6144_k4096 | h100 | 0.05403 | **cublas-fresh 0.0348 (1.55)**<br>lt-fresh 0.0348 (1.55)<br>lt 0.042 (1.29)<br>cublas 0.0421 (1.28) | torch-eager 0.0417 (1.30) | torch-compile 0.0461 (1.17) | torch-compile-cg 0.0455 (1.19) | torch-compile-ro 0.0422 (1.28) | triton 0.1412 (0.38) |
| pj_m2048_n28672_k4096 | rtx-5080 | 11.8 | **lt-fresh 12.23 (0.97)**<br>cublas-fresh 12.24 (0.96)<br>cublas 12.24 (0.96)<br>lt 12.24 (0.96) | torch-eager 12.5 (0.94) | torch-compile 12.5 (0.94) | torch-compile-cg 12.55 (0.94) | torch-compile-ro 12.52 (0.94) | triton 15.77 (0.75) |
| pj_m2048_n28672_k4096 | rtx-3090 | 19.3 | **lt-fresh 21.53 (0.90)**<br>lt 21.53 (0.90)<br>cublas-fresh 21.53 (0.90)<br>cublas 21.53 (0.90) | torch-eager 24.8 (0.78) | torch-compile 24.8 (0.78) | torch-compile-cg 24.8 (0.78) | torch-compile-ro 24.8 (0.78) | triton 27.76 (0.70) |
| pj_m2048_n28672_k4096 | a100 | 23.22 | lt-fresh 25.51 (0.91)<br>lt 25.51 (0.91)<br>cublas 25.52 (0.91)<br>cublas-fresh 25.52 (0.91) | torch-eager 25.48 (0.91) | **torch-compile 25.48 (0.91)** | torch-compile-cg 25.48 (0.91) | torch-compile-ro 25.53 (0.91) | triton 25.94 (0.89) |
| pj_m2048_n28672_k4096 | rtx-4090 | 16.4 | **cublas-fresh 16.45 (1.00)**<br>cublas 16.45 (1.00)<br>lt-fresh 16.45 (1.00)<br>lt 16.45 (1.00) | torch-eager 16.45 (1.00) | torch-compile 16.45 (1.00) | torch-compile-cg 16.45 (1.00) | torch-compile-ro 16.45 (1.00) | triton 20.69 (0.79) |
| pj_m2048_n28672_k4096 | h100 | 16.15 | cublas 13.89 (1.16)<br>cublas-fresh 13.9 (1.16)<br>lt-fresh 13.9 (1.16)<br>lt 13.9 (1.16) | torch-eager 13.87 (1.16) | **torch-compile 13.86 (1.17)** | torch-compile-cg 14.33 (1.13) | torch-compile-ro 13.97 (1.16) | triton 16.53 (0.98) |
| pj_m2048_n4096_k14336 | rtx-5080 | 6.412 | **cublas 6.434 (1.00)**<br>lt-fresh 6.435 (1.00)<br>lt 6.438 (1.00)<br>cublas-fresh 6.44 (1.00) | torch-eager 6.533 (0.98) | torch-compile 6.535 (0.98) | torch-compile-cg 6.539 (0.98) | torch-compile-ro 6.559 (0.98) | triton 8.233 (0.78) |
| pj_m2048_n4096_k14336 | rtx-3090 | 11.02 | **lt-fresh 11.24 (0.98)**<br>lt 11.25 (0.98)<br>cublas 11.97 (0.92)<br>cublas-fresh 11.98 (0.92) | torch-eager 15.99 (0.69) | torch-compile 12.14 (0.91) | torch-compile-cg 12.18 (0.90) | torch-compile-ro 12.22 (0.90) | triton 14.61 (0.75) |
| pj_m2048_n4096_k14336 | a100 | 11.76 | cublas-fresh 12.69 (0.93)<br>cublas 12.7 (0.93)<br>lt-fresh 13.07 (0.90)<br>lt 13.08 (0.90) | torch-eager 12.67 (0.93) | torch-compile 13.45 (0.87) | torch-compile-cg 13.44 (0.88) | **torch-compile-ro 12.65 (0.93)** | triton 13.66 (0.86) |
| pj_m2048_n4096_k14336 | rtx-4090 | 8.073 | **cublas 8.425 (0.96)**<br>lt 8.425 (0.96)<br>cublas-fresh 8.425 (0.96)<br>lt-fresh 8.425 (0.96) | torch-eager 8.427 (0.96) | torch-compile 8.427 (0.96) | torch-compile-cg 8.428 (0.96) | torch-compile-ro 8.428 (0.96) | triton 10.24 (0.79) |
| pj_m2048_n4096_k14336 | h100 | 8.443 | cublas-fresh 7.027 (1.20)<br>lt-fresh 7.029 (1.20)<br>lt 7.032 (1.20)<br>cublas 7.038 (1.20) | torch-eager 6.879 (1.23) | **torch-compile 6.322 (1.34)** | torch-compile-cg 6.929 (1.22) | torch-compile-ro 6.821 (1.24) | triton 8.7 (0.97) |
| pj_m2048_n4096_k4096 | rtx-5080 | 1.832 | **lt-fresh 1.798 (1.02)**<br>cublas-fresh 1.798 (1.02)<br>lt 1.801 (1.02)<br>cublas 1.802 (1.02) | torch-eager 1.8 (1.02) | torch-compile 1.81 (1.01) | torch-compile-cg 1.813 (1.01) | torch-compile-ro 1.815 (1.01) | triton 2.362 (0.78) |
| pj_m2048_n4096_k4096 | rtx-3090 | 2.872 | **cublas 3.255 (0.88)**<br>lt 3.256 (0.88)<br>cublas-fresh 3.256 (0.88)<br>lt-fresh 3.259 (0.88) | torch-eager 3.358 (0.86) | torch-compile 3.29 (0.87) | torch-compile-cg 3.315 (0.87) | torch-compile-ro 3.399 (0.84) | triton 4.065 (0.71) |
| pj_m2048_n4096_k4096 | a100 | 3.372 | cublas 3.692 (0.91)<br>cublas-fresh 3.692 (0.91)<br>lt 3.893 (0.87)<br>lt-fresh 3.894 (0.87) | torch-eager 3.818 (0.88) | torch-compile 4.76 (0.71) | **torch-compile-cg 3.692 (0.91)** | torch-compile-ro 3.694 (0.91) | triton 3.907 (0.86) |
| pj_m2048_n4096_k4096 | rtx-4090 | 2.348 | **lt 2.408 (0.97)**<br>cublas 2.409 (0.97)<br>lt-fresh 2.412 (0.97)<br>cublas-fresh 2.413 (0.97) | torch-eager 2.412 (0.97) | torch-compile 2.409 (0.97) | torch-compile-cg 2.41 (0.97) | torch-compile-ro 2.41 (0.97) | triton 2.956 (0.79) |
| pj_m2048_n4096_k4096 | h100 | 2.446 | cublas-fresh 1.983 (1.23)<br>lt-fresh 1.983 (1.23)<br>cublas 1.985 (1.23)<br>lt 1.985 (1.23) | torch-eager 2.362 (1.04) | torch-compile 1.798 (1.36) | **torch-compile-cg 1.686 (1.45)** | torch-compile-ro 1.836 (1.33) | triton 2.499 (0.98) |
| pj_m2048_n6144_k4096 | rtx-5080 | 2.643 | **lt-fresh 2.652 (1.00)**<br>cublas-fresh 2.656 (1.00)<br>cublas 2.66 (0.99)<br>lt 2.661 (0.99) | torch-eager 2.687 (0.98) | torch-compile 2.699 (0.98) | torch-compile-cg 2.689 (0.98) | torch-compile-ro 2.697 (0.98) | triton 3.43 (0.77) |
| pj_m2048_n6144_k4096 | rtx-3090 | 4.576 | **cublas-fresh 4.699 (0.97)**<br>cublas 4.7 (0.97)<br>lt 4.7 (0.97)<br>lt-fresh 4.7 (0.97) | torch-eager 4.751 (0.96) | torch-compile 4.847 (0.94) | torch-compile-cg 4.772 (0.96) | torch-compile-ro 4.749 (0.96) | triton 5.944 (0.77) |
| pj_m2048_n6144_k4096 | a100 | 5.094 | **lt 5.711 (0.89)**<br>lt-fresh 5.711 (0.89)<br>cublas-fresh 5.743 (0.89)<br>cublas 5.744 (0.89) | torch-eager 5.75 (0.89) | torch-compile 5.747 (0.89) | torch-compile-cg 5.744 (0.89) | torch-compile-ro 5.748 (0.89) | triton 5.835 (0.87) |
| pj_m2048_n6144_k4096 | rtx-4090 | 3.502 | **cublas 3.631 (0.96)**<br>lt 3.632 (0.96)<br>cublas-fresh 3.632 (0.96)<br>lt-fresh 3.632 (0.96) | torch-eager 3.634 (0.96) | torch-compile 3.636 (0.96) | torch-compile-cg 3.635 (0.96) | torch-compile-ro 3.635 (0.96) | triton 4.431 (0.79) |
| pj_m2048_n6144_k4096 | h100 | 2.73 | cublas-fresh 2.974 (0.92)<br>lt-fresh 2.975 (0.92)<br>cublas 2.976 (0.92)<br>lt 2.977 (0.92) | torch-eager 2.977 (0.92) | torch-compile 2.914 (0.94) | torch-compile-cg 2.874 (0.95) | **torch-compile-ro 2.793 (0.98)** | triton 3.881 (0.70) |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.6983 | **lt-tf32-fresh 0.5952 (1.17)**<br>cublas-tf32-fresh 0.5955 (1.17)<br>lt-tf32 0.6233 (1.12)<br>cublas-tf32 0.624 (1.12) | torch-eager-tf32 0.65 (1.07) | torch-compile-tf32 0.6552 (1.07) | torch-compile-cg-tf32 0.6605 (1.06) | torch-compile-ro-tf32 0.6701 (1.04) | triton-tf32 0.6442 (1.08) |
| pj_m128_n28672_k4096_tf32 | rtx-3090 | 1.078 | lt-tf32-fresh 1.161 (0.93)<br>cublas-tf32 1.162 (0.93)<br>lt-tf32 1.162 (0.93)<br>cublas-tf32-fresh 1.162 (0.93) | torch-eager-tf32 1.007 (1.07) | **torch-compile-tf32 0.9636 (1.12)** | torch-compile-cg-tf32 0.9636 (1.12) | torch-compile-ro-tf32 0.9656 (1.12) | triton-tf32 1.289 (0.84) |
| pj_m128_n28672_k4096_tf32 | a100 | 0.651 | lt-tf32 0.4844 (1.34)<br>lt-tf32-fresh 0.4844 (1.34)<br>cublas-tf32 0.5007 (1.30)<br>cublas-tf32-fresh 0.5007 (1.30) | torch-eager-tf32 0.4915 (1.32) | torch-compile-tf32 0.5018 (1.30) | torch-compile-cg-tf32 0.5243 (1.24) | torch-compile-ro-tf32 0.4895 (1.33) | **triton-tf32 0.4147 (1.57)** |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 1.032 | lt-tf32-fresh 0.8765 (1.18)<br>lt-tf32 0.8786 (1.17)<br>cublas-tf32 0.897 (1.15)<br>cublas-tf32-fresh 0.897 (1.15) | torch-eager-tf32 0.8991 (1.15) | **torch-compile-tf32 0.8448 (1.22)** | torch-compile-cg-tf32 0.8468 (1.22) | torch-compile-ro-tf32 0.897 (1.15) | triton-tf32 0.9718 (1.06) |
| pj_m128_n28672_k4096_tf32 | h100 | 0.3335 | **cublas-tf32 0.2465 (1.35)**<br>lt-tf32 0.2493 (1.34)<br>lt-tf32-fresh 0.251 (1.33)<br>cublas-tf32-fresh 0.2519 (1.32) | torch-eager-tf32 0.2749 (1.21) | torch-compile-tf32 0.2728 (1.22) | torch-compile-cg-tf32 0.2719 (1.23) | torch-compile-ro-tf32 0.2723 (1.22) | triton-tf32 0.5449 (0.61) |
| pj_m128_n4096_k14336_tf32 | rtx-5080 | 0.4065 | **lt-tf32-fresh 0.3127 (1.30)**<br>cublas-tf32-fresh 0.3135 (1.30)<br>lt-tf32 0.3211 (1.27)<br>cublas-tf32 0.3218 (1.26) | torch-eager-tf32 0.3173 (1.28) | torch-compile-tf32 0.3266 (1.24) | torch-compile-cg-tf32 0.3322 (1.22) | torch-compile-ro-tf32 0.3321 (1.22) | triton-tf32 0.3893 (1.04) |
| pj_m128_n4096_k14336_tf32 | rtx-3090 | 0.6794 | lt-tf32-fresh 0.9216 (0.74)<br>lt-tf32 0.9226 (0.74)<br>cublas-tf32 1.114 (0.61)<br>cublas-tf32-fresh 1.114 (0.61) | torch-eager-tf32 1.104 (0.62) | torch-compile-tf32 0.6216 (1.09) | **torch-compile-cg-tf32 0.6205 (1.09)** | torch-compile-ro-tf32 1.103 (0.62) | triton-tf32 0.6758 (1.01) |
| pj_m128_n4096_k14336_tf32 | a100 | 0.4112 | **cublas-tf32 0.255 (1.61)**<br>lt-tf32 0.255 (1.61)<br>cublas-tf32-fresh 0.256 (1.61)<br>lt-tf32-fresh 0.256 (1.61) | torch-eager-tf32 0.258 (1.59) | torch-compile-tf32 0.257 (1.60) | torch-compile-cg-tf32 0.256 (1.61) | torch-compile-ro-tf32 0.256 (1.61) | triton-tf32 0.2621 (1.57) |
| pj_m128_n4096_k14336_tf32 | rtx-4090 | 0.534 | **lt-tf32-fresh 0.4096 (1.30)**<br>lt-tf32 0.4147 (1.29)<br>cublas-tf32-fresh 0.4157 (1.28)<br>cublas-tf32 0.4219 (1.27) | torch-eager-tf32 0.4168 (1.28) | torch-compile-tf32 0.4178 (1.28) | torch-compile-cg-tf32 0.4157 (1.28) | torch-compile-ro-tf32 0.4157 (1.28) | triton-tf32 0.4947 (1.08) |
| pj_m128_n4096_k14336_tf32 | h100 | 0.3395 | cublas-tf32-fresh 0.1496 (2.27)<br>cublas-tf32 0.1502 (2.26)<br>lt-tf32 0.1545 (2.20)<br>lt-tf32-fresh 0.1613 (2.10) | torch-eager-tf32 0.1477 (2.30) | torch-compile-tf32 0.1468 (2.31) | **torch-compile-cg-tf32 0.1464 (2.32)** | torch-compile-ro-tf32 0.1468 (2.31) | triton-tf32 0.2827 (1.20) |
| pj_m128_n4096_k4096_tf32 | rtx-5080 | 0.1297 | **lt-tf32-fresh 0.0877 (1.48)**<br>cublas-tf32-fresh 0.0888 (1.46)<br>lt-tf32 0.1097 (1.18)<br>cublas-tf32 0.1157 (1.12) | torch-eager-tf32 0.1157 (1.12) | torch-compile-tf32 0.1296 (1.00) | torch-compile-cg-tf32 0.1317 (0.98) | torch-compile-ro-tf32 0.1314 (0.99) | triton-tf32 0.1294 (1.00) |
| pj_m128_n4096_k4096_tf32 | rtx-3090 | 0.2177 | **lt-tf32-fresh 0.168 (1.30)**<br>lt-tf32 0.169 (1.29)<br>cublas-tf32 0.171 (1.27)<br>cublas-tf32-fresh 0.171 (1.27) | torch-eager-tf32 0.17 (1.28) | torch-compile-tf32 0.17 (1.28) | torch-compile-cg-tf32 0.169 (1.29) | torch-compile-ro-tf32 0.17 (1.28) | triton-tf32 0.1915 (1.14) |
| pj_m128_n4096_k4096_tf32 | a100 | 0.1978 | cublas-tf32-fresh 0.0901 (2.20)<br>lt-tf32-fresh 0.0901 (2.20)<br>cublas-tf32 0.0922 (2.15)<br>lt-tf32 0.0922 (2.15) | torch-eager-tf32 0.0973 (2.03) | torch-compile-tf32 0.1085 (1.82) | torch-compile-cg-tf32 0.1075 (1.84) | torch-compile-ro-tf32 0.0952 (2.08) | **triton-tf32 0.087 (2.27)** |
| pj_m128_n4096_k4096_tf32 | rtx-4090 | 0.1742 | **cublas-tf32-fresh 0.128 (1.36)**<br>lt-tf32-fresh 0.128 (1.36)<br>cublas-tf32 0.1311 (1.33)<br>lt-tf32 0.1311 (1.33) | torch-eager-tf32 0.1311 (1.33) | torch-compile-tf32 0.1311 (1.33) | torch-compile-cg-tf32 0.13 (1.34) | torch-compile-ro-tf32 0.129 (1.35) | triton-tf32 0.1527 (1.14) |
| pj_m128_n4096_k4096_tf32 | h100 | 0.08789 | cublas-tf32-fresh 0.0435 (2.02)<br>lt-tf32-fresh 0.0436 (2.02)<br>cublas-tf32 0.045 (1.95)<br>lt-tf32 0.0456 (1.93) | torch-eager-tf32 0.0435 (2.02) | torch-compile-tf32 0.0436 (2.02) | **torch-compile-cg-tf32 0.0434 (2.03)** | torch-compile-ro-tf32 0.0437 (2.01) | triton-tf32 0.0891 (0.99) |
| pj_m128_n6144_k4096_tf32 | rtx-5080 | 0.1797 | **lt-tf32-fresh 0.1304 (1.38)**<br>lt-tf32 0.1518 (1.18)<br>cublas-tf32-fresh 0.1832 (0.98)<br>cublas-tf32 0.2022 (0.89) | torch-eager-tf32 0.1967 (0.91) | torch-compile-tf32 0.1895 (0.95) | torch-compile-cg-tf32 0.1948 (0.92) | torch-compile-ro-tf32 0.2129 (0.84) | triton-tf32 0.2216 (0.81) |
| pj_m128_n6144_k4096_tf32 | rtx-3090 | 0.2905 | **lt-tf32 0.2499 (1.16)**<br>lt-tf32-fresh 0.2508 (1.16)<br>cublas-tf32 0.3645 (0.80)<br>cublas-tf32-fresh 0.3645 (0.80) | torch-eager-tf32 0.3297 (0.88) | torch-compile-tf32 0.2847 (1.02) | torch-compile-cg-tf32 0.2847 (1.02) | torch-compile-ro-tf32 0.3297 (0.88) | triton-tf32 0.3564 (0.82) |
| pj_m128_n6144_k4096_tf32 | a100 | 0.2077 | cublas-tf32-fresh 0.1075 (1.93)<br>lt-tf32-fresh 0.1075 (1.93)<br>cublas-tf32 0.1085 (1.91)<br>lt-tf32 0.1085 (1.91) | torch-eager-tf32 0.1116 (1.86) | torch-compile-tf32 0.1116 (1.86) | torch-compile-cg-tf32 0.1096 (1.89) | torch-compile-ro-tf32 0.1096 (1.89) | **triton-tf32 0.1004 (2.07)** |
| pj_m128_n6144_k4096_tf32 | rtx-4090 | 0.2686 | **lt-tf32-fresh 0.2068 (1.30)**<br>lt-tf32 0.2099 (1.28)<br>cublas-tf32-fresh 0.2355 (1.14)<br>cublas-tf32 0.2358 (1.14) | torch-eager-tf32 0.2365 (1.14) | torch-compile-tf32 0.2365 (1.14) | torch-compile-cg-tf32 0.2355 (1.14) | torch-compile-ro-tf32 0.2355 (1.14) | triton-tf32 0.2499 (1.07) |
| pj_m128_n6144_k4096_tf32 | h100 | 0.1233 | **lt-tf32-fresh 0.052 (2.37)**<br>lt-tf32 0.055 (2.24)<br>cublas-tf32-fresh 0.0603 (2.04)<br>cublas-tf32 0.0678 (1.82) | torch-eager-tf32 0.0658 (1.87) | torch-compile-tf32 0.0661 (1.86) | torch-compile-cg-tf32 0.0662 (1.86) | torch-compile-ro-tf32 0.0664 (1.86) | triton-tf32 0.1257 (0.98) |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 8.532 | **lt-tf32-fresh 8.072 (1.06)**<br>cublas-tf32 8.073 (1.06)<br>cublas-tf32-fresh 8.075 (1.06)<br>lt-tf32 8.075 (1.06) | torch-eager-tf32 8.172 (1.04) | torch-compile-tf32 8.215 (1.04) | torch-compile-cg-tf32 8.183 (1.04) | torch-compile-ro-tf32 8.205 (1.04) | triton-tf32 8.535 (1.00) |
| pj_m2048_n28672_k4096_tf32 | rtx-3090 | 14.94 | **cublas-tf32-fresh 13.97 (1.07)**<br>cublas-tf32 13.97 (1.07)<br>lt-tf32 13.97 (1.07)<br>lt-tf32-fresh 13.97 (1.07) | torch-eager-tf32 16.09 (0.93) | torch-compile-tf32 18.42 (0.81) | torch-compile-cg-tf32 18.42 (0.81) | torch-compile-ro-tf32 16.63 (0.90) | triton-tf32 14.84 (1.01) |
| pj_m2048_n28672_k4096_tf32 | a100 | 6.719 | **cublas-tf32 3.694 (1.82)**<br>lt-tf32 3.694 (1.82)<br>cublas-tf32-fresh 3.694 (1.82)<br>lt-tf32-fresh 3.694 (1.82) | torch-eager-tf32 3.746 (1.79) | torch-compile-tf32 4.361 (1.54) | torch-compile-cg-tf32 3.7 (1.82) | torch-compile-ro-tf32 3.715 (1.81) | triton-tf32 4.602 (1.46) |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 13.46 | lt-tf32-fresh 12.44 (1.08)<br>lt-tf32 12.44 (1.08)<br>cublas-tf32 12.44 (1.08)<br>cublas-tf32-fresh 12.44 (1.08) | torch-eager-tf32 12.44 (1.08) | **torch-compile-tf32 12.43 (1.08)** | torch-compile-cg-tf32 12.43 (1.08) | torch-compile-ro-tf32 12.43 (1.08) | triton-tf32 13.47 (1.00) |
| pj_m2048_n28672_k4096_tf32 | h100 | 3.852 | **lt-tf32 2.587 (1.49)**<br>lt-tf32-fresh 2.609 (1.48)<br>cublas-tf32 3.023 (1.27)<br>cublas-tf32-fresh 3.047 (1.26) | torch-eager-tf32 3.423 (1.13) | torch-compile-tf32 3.425 (1.12) | torch-compile-cg-tf32 3.424 (1.13) | torch-compile-ro-tf32 3.424 (1.13) | triton-tf32 7.893 (0.49) |
| pj_m2048_n4096_k14336_tf32 | rtx-5080 | 4.357 | **lt-tf32-fresh 4.264 (1.02)**<br>cublas-tf32-fresh 4.265 (1.02)<br>lt-tf32 4.269 (1.02)<br>cublas-tf32 4.272 (1.02) | torch-eager-tf32 4.556 (0.96) | torch-compile-tf32 4.336 (1.00) | torch-compile-cg-tf32 4.341 (1.00) | torch-compile-ro-tf32 4.572 (0.95) | triton-tf32 4.695 (0.93) |
| pj_m2048_n4096_k14336_tf32 | rtx-3090 | 8.436 | **lt-tf32 7.557 (1.12)**<br>lt-tf32-fresh 7.559 (1.12)<br>cublas-tf32-fresh 7.563 (1.12)<br>cublas-tf32 7.564 (1.12) | torch-eager-tf32 7.628 (1.11) | torch-compile-tf32 7.646 (1.10) | torch-compile-cg-tf32 7.629 (1.11) | torch-compile-ro-tf32 7.578 (1.11) | triton-tf32 7.989 (1.06) |
| pj_m2048_n4096_k14336_tf32 | a100 | 3.568 | cublas-tf32 1.894 (1.88)<br>cublas-tf32-fresh 1.894 (1.88)<br>lt-tf32 1.895 (1.88)<br>lt-tf32-fresh 1.899 (1.88) | torch-eager-tf32 1.896 (1.88) | torch-compile-tf32 2.28 (1.56) | torch-compile-cg-tf32 1.927 (1.85) | **torch-compile-ro-tf32 1.838 (1.94)** | triton-tf32 2.3 (1.55) |
| pj_m2048_n4096_k14336_tf32 | rtx-4090 | 6.915 | **lt-tf32-fresh 6.226 (1.11)**<br>lt-tf32 6.227 (1.11)<br>cublas-tf32 6.241 (1.11)<br>cublas-tf32-fresh 6.241 (1.11) | torch-eager-tf32 6.246 (1.11) | torch-compile-tf32 6.245 (1.11) | torch-compile-cg-tf32 6.243 (1.11) | torch-compile-ro-tf32 6.243 (1.11) | triton-tf32 6.68 (1.04) |
| pj_m2048_n4096_k14336_tf32 | h100 | 1.631 | cublas-tf32 1.213 (1.34)<br>cublas-tf32-fresh 1.234 (1.32)<br>lt-tf32 1.361 (1.20)<br>lt-tf32-fresh 1.386 (1.18) | torch-eager-tf32 1.256 (1.30) | **torch-compile-tf32 1.081 (1.51)** | torch-compile-cg-tf32 1.242 (1.31) | torch-compile-ro-tf32 1.208 (1.35) | triton-tf32 3.868 (0.42) |
| pj_m2048_n4096_k4096_tf32 | rtx-5080 | 1.329 | **lt-tf32-fresh 1.165 (1.14)**<br>lt-tf32 1.166 (1.14)<br>cublas-tf32-fresh 1.168 (1.14)<br>cublas-tf32 1.18 (1.13) | torch-eager-tf32 1.313 (1.01) | torch-compile-tf32 1.25 (1.06) | torch-compile-cg-tf32 1.25 (1.06) | torch-compile-ro-tf32 1.329 (1.00) | triton-tf32 1.37 (0.97) |
| pj_m2048_n4096_k4096_tf32 | rtx-3090 | 2.392 | cublas-tf32 2.403 (1.00)<br>lt-tf32 2.403 (1.00)<br>cublas-tf32-fresh 2.403 (1.00)<br>lt-tf32-fresh 2.403 (1.00) | torch-eager-tf32 2.44 (0.98) | torch-compile-tf32 2.399 (1.00) | torch-compile-cg-tf32 2.419 (0.99) | torch-compile-ro-tf32 2.421 (0.99) | **triton-tf32 2.381 (1.00)** |
| pj_m2048_n4096_k4096_tf32 | a100 | 1.012 | **lt-tf32 0.5284 (1.91)**<br>lt-tf32-fresh 0.5284 (1.91)<br>cublas-tf32 0.5366 (1.89)<br>cublas-tf32-fresh 0.5386 (1.88) | torch-eager-tf32 0.6892 (1.47) | torch-compile-tf32 0.6902 (1.47) | torch-compile-cg-tf32 0.6881 (1.47) | torch-compile-ro-tf32 0.6871 (1.47) | triton-tf32 0.6666 (1.52) |
| pj_m2048_n4096_k4096_tf32 | rtx-4090 | 1.998 | **lt-tf32 1.786 (1.12)**<br>lt-tf32-fresh 1.786 (1.12)<br>cublas-tf32 1.801 (1.11)<br>cublas-tf32-fresh 1.801 (1.11) | torch-eager-tf32 1.805 (1.11) | torch-compile-tf32 1.797 (1.11) | torch-compile-cg-tf32 1.797 (1.11) | torch-compile-ro-tf32 1.805 (1.11) | triton-tf32 1.933 (1.03) |
| pj_m2048_n4096_k4096_tf32 | h100 | 0.4543 | **lt-tf32-fresh 0.341 (1.33)**<br>cublas-tf32-fresh 0.3421 (1.33)<br>lt-tf32 0.3475 (1.31)<br>cublas-tf32 0.3487 (1.30) | torch-eager-tf32 0.3503 (1.30) | torch-compile-tf32 0.3501 (1.30) | torch-compile-cg-tf32 0.3489 (1.30) | torch-compile-ro-tf32 0.3508 (1.30) | triton-tf32 1.204 (0.38) |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.018 | **lt-tf32-fresh 1.768 (1.14)**<br>cublas-tf32-fresh 1.768 (1.14)<br>lt-tf32 1.782 (1.13)<br>cublas-tf32 1.783 (1.13) | torch-eager-tf32 1.868 (1.08) | torch-compile-tf32 1.83 (1.10) | torch-compile-cg-tf32 1.834 (1.10) | torch-compile-ro-tf32 1.889 (1.07) | triton-tf32 2.002 (1.01) |
| pj_m2048_n6144_k4096_tf32 | rtx-3090 | 3.44 | **cublas-tf32-fresh 3.142 (1.10)**<br>lt-tf32-fresh 3.143 (1.09)<br>cublas-tf32 3.143 (1.09)<br>lt-tf32 3.143 (1.09) | torch-eager-tf32 3.171 (1.08) | torch-compile-tf32 3.566 (0.96) | torch-compile-cg-tf32 3.568 (0.96) | torch-compile-ro-tf32 3.17 (1.09) | triton-tf32 3.325 (1.03) |
| pj_m2048_n6144_k4096_tf32 | a100 | 1.533 | cublas-tf32 0.812 (1.89)<br>lt-tf32 0.812 (1.89)<br>cublas-tf32-fresh 0.8151 (1.88)<br>lt-tf32-fresh 0.8161 (1.88) | torch-eager-tf32 0.809 (1.89) | torch-compile-tf32 0.809 (1.89) | torch-compile-cg-tf32 0.8069 (1.90) | **torch-compile-ro-tf32 0.8049 (1.90)** | triton-tf32 1.059 (1.45) |
| pj_m2048_n6144_k4096_tf32 | rtx-4090 | 2.953 | **lt-tf32 2.677 (1.10)**<br>lt-tf32-fresh 2.677 (1.10)<br>cublas-tf32 2.679 (1.10)<br>cublas-tf32-fresh 2.679 (1.10) | torch-eager-tf32 2.68 (1.10) | torch-compile-tf32 2.681 (1.10) | torch-compile-cg-tf32 2.679 (1.10) | torch-compile-ro-tf32 2.679 (1.10) | triton-tf32 2.898 (1.02) |
| pj_m2048_n6144_k4096_tf32 | h100 | 0.7127 | **lt-tf32 0.5671 (1.26)**<br>lt-tf32-fresh 0.5892 (1.21)<br>cublas-tf32 0.6823 (1.04)<br>cublas-tf32-fresh 0.7063 (1.01) | torch-eager-tf32 0.6557 (1.09) | torch-compile-tf32 0.6748 (1.06) | torch-compile-cg-tf32 0.6613 (1.08) | torch-compile-ro-tf32 0.6639 (1.07) | triton-tf32 1.74 (0.41) |
| pj_m128_n28672_k4096_bf16 | rtx-5080 | 0.3426 |  | **torch-bf16 0.3354 (1.02)** | torch-bf16-compile 0.3375 (1.01) | torch-bf16-compile-cg 0.3424 (1.00) | torch-bf16-compile-ro 0.348 (0.98) | triton-bf16 0.3355 (1.02) |
| pj_m128_n28672_k4096_bf16 | rtx-3090 | 0.52 |  | torch-bf16 0.556 (0.94) | torch-bf16-compile 0.4833 (1.08) | **torch-bf16-compile-cg 0.4823 (1.08)** | torch-bf16-compile-ro 0.4823 (1.08) | triton-bf16 0.599 (0.87) |
| pj_m128_n28672_k4096_bf16 | a100 | 0.3453 |  | torch-bf16 0.2929 (1.18) | torch-bf16-compile 0.2468 (1.40) | torch-bf16-compile-cg 0.2478 (1.39) | torch-bf16-compile-ro 0.2929 (1.18) | **triton-bf16 0.2171 (1.59)** |
| pj_m128_n28672_k4096_bf16 | rtx-4090 | 0.4928 |  | torch-bf16 0.4239 (1.16) | **torch-bf16-compile 0.4198 (1.17)** | torch-bf16-compile-cg 0.4332 (1.14) | torch-bf16-compile-ro 0.4372 (1.13) | triton-bf16 0.4649 (1.06) |
| pj_m128_n28672_k4096_bf16 | h100 | 0.2156 |  | torch-bf16 0.1335 (1.61) | torch-bf16-compile 0.1307 (1.65) | **torch-bf16-compile-cg 0.1295 (1.66)** | torch-bf16-compile-ro 0.1305 (1.65) | triton-bf16 0.1524 (1.41) |
| pj_m128_n4096_k14336_bf16 | rtx-5080 | 0.2179 |  | **torch-bf16 0.1759 (1.24)** | torch-bf16-compile 0.1857 (1.17) | torch-bf16-compile-cg 0.1926 (1.13) | torch-bf16-compile-ro 0.1934 (1.13) | triton-bf16 0.2045 (1.07) |
| pj_m128_n4096_k14336_bf16 | rtx-3090 | 0.402 |  | torch-bf16 0.3082 (1.30) | torch-bf16-compile 0.2807 (1.43) | **torch-bf16-compile-cg 0.2806 (1.43)** | torch-bf16-compile-ro 0.2806 (1.43) | triton-bf16 0.3226 (1.25) |
| pj_m128_n4096_k14336_bf16 | a100 | 0.2762 |  | torch-bf16 0.1249 (2.21) | torch-bf16-compile 0.1249 (2.21) | **torch-bf16-compile-cg 0.1229 (2.25)** | torch-bf16-compile-ro 0.1229 (2.25) | triton-bf16 0.1464 (1.89) |
| pj_m128_n4096_k14336_bf16 | rtx-4090 | 0.305 |  | torch-bf16 0.2478 (1.23) | torch-bf16-compile 0.2478 (1.23) | **torch-bf16-compile-cg 0.2468 (1.24)** | torch-bf16-compile-ro 0.2468 (1.24) | triton-bf16 0.2652 (1.15) |
| pj_m128_n4096_k14336_bf16 | h100 | 0.2143 |  | **torch-bf16 0.0745 (2.88)** | torch-bf16-compile 0.0798 (2.69) | torch-bf16-compile-cg 0.0796 (2.69) | torch-bf16-compile-ro 0.0755 (2.84) | triton-bf16 0.1082 (1.98) |
| pj_m128_n4096_k4096_bf16 | rtx-5080 | 0.06252 |  | **torch-bf16 0.0602 (1.04)** | torch-bf16-compile 0.0706 (0.89) | torch-bf16-compile-cg 0.0757 (0.83) | torch-bf16-compile-ro 0.0763 (0.82) | triton-bf16 0.0764 (0.82) |
| pj_m128_n4096_k4096_bf16 | rtx-3090 | 0.1199 |  | torch-bf16 0.086 (1.39) | torch-bf16-compile 0.086 (1.39) | **torch-bf16-compile-cg 0.085 (1.41)** | torch-bf16-compile-ro 0.085 (1.41) | triton-bf16 0.0922 (1.30) |
| pj_m128_n4096_k4096_bf16 | a100 | 0.1089 |  | torch-bf16 0.0594 (1.83) | torch-bf16-compile 0.0584 (1.86) | torch-bf16-compile-cg 0.0573 (1.90) | torch-bf16-compile-ro 0.0573 (1.90) | **triton-bf16 0.0532 (2.05)** |
| pj_m128_n4096_k4096_bf16 | rtx-4090 | 0.1018 |  | **torch-bf16 0.0768 (1.33)** | torch-bf16-compile 0.0768 (1.33) | torch-bf16-compile-cg 0.0768 (1.33) | torch-bf16-compile-ro 0.0778 (1.31) | triton-bf16 0.085 (1.20) |
| pj_m128_n4096_k4096_bf16 | h100 | 0.06544 |  | torch-bf16 0.0222 (2.95) | **torch-bf16-compile 0.0218 (3.00)** | torch-bf16-compile-cg 0.0221 (2.96) | torch-bf16-compile-ro 0.0221 (2.96) | triton-bf16 0.0353 (1.85) |
| pj_m128_n6144_k4096_bf16 | rtx-5080 | 0.08725 |  | **torch-bf16 0.0921 (0.95)** | torch-bf16-compile 0.1031 (0.85) | torch-bf16-compile-cg 0.1087 (0.80) | torch-bf16-compile-ro 0.1084 (0.80) | triton-bf16 0.1222 (0.71) |
| pj_m128_n6144_k4096_bf16 | rtx-3090 | 0.1882 |  | torch-bf16 0.1311 (1.44) | torch-bf16-compile 0.129 (1.46) | **torch-bf16-compile-cg 0.128 (1.47)** | torch-bf16-compile-ro 0.128 (1.47) | triton-bf16 0.1485 (1.27) |
| pj_m128_n6144_k4096_bf16 | a100 | 0.137 |  | torch-bf16 0.0758 (1.81) | torch-bf16-compile 0.0686 (2.00) | torch-bf16-compile-cg 0.0676 (2.03) | torch-bf16-compile-ro 0.0737 (1.86) | **triton-bf16 0.0635 (2.16)** |
| pj_m128_n6144_k4096_bf16 | rtx-4090 | 0.163 |  | torch-bf16 0.1137 (1.43) | **torch-bf16-compile 0.1126 (1.45)** | torch-bf16-compile-cg 0.1126 (1.45) | torch-bf16-compile-ro 0.1126 (1.45) | triton-bf16 0.127 (1.28) |
| pj_m128_n6144_k4096_bf16 | h100 | 0.0773 |  | torch-bf16 0.0311 (2.49) | **torch-bf16-compile 0.0305 (2.53)** | torch-bf16-compile-cg 0.0307 (2.52) | torch-bf16-compile-ro 0.0309 (2.50) | triton-bf16 0.0411 (1.88) |
| pj_m16_n28672_k4096_bf16 | rtx-5080 | 0.2252 |  | **torch-bf16 0.2871 (0.78)** | torch-bf16-compile 0.2918 (0.77) | torch-bf16-compile-cg 0.2958 (0.76) | torch-bf16-compile-ro 0.3023 (0.74) | triton-bf16 0.3001 (0.75) |
| pj_m16_n28672_k4096_bf16 | rtx-3090 | 0.3062 |  | torch-bf16 0.3779 (0.81) | **torch-bf16-compile 0.2734 (1.12)** | torch-bf16-compile-cg 0.2939 (1.04) | torch-bf16-compile-ro 0.3308 (0.93) | triton-bf16 0.2878 (1.06) |
| pj_m16_n28672_k4096_bf16 | a100 | 0.2292 |  | torch-bf16 0.2232 (1.03) | **torch-bf16-compile 0.1894 (1.21)** | torch-bf16-compile-cg 0.2007 (1.14) | torch-bf16-compile-ro 0.2243 (1.02) | triton-bf16 0.1997 (1.15) |
| pj_m16_n28672_k4096_bf16 | rtx-4090 | 0.3705 |  | torch-bf16 0.3697 (1.00) | **torch-bf16-compile 0.3523 (1.05)** | torch-bf16-compile-cg 0.3676 (1.01) | torch-bf16-compile-ro 0.3881 (0.95) | triton-bf16 0.3625 (1.02) |
| pj_m16_n28672_k4096_bf16 | h100 | 0.1941 |  | torch-bf16 0.0979 (1.98) | torch-bf16-compile 0.0929 (2.09) | **torch-bf16-compile-cg 0.0916 (2.12)** | torch-bf16-compile-ro 0.0943 (2.06) | triton-bf16 0.1273 (1.52) |
| pj_m16_n4096_k14336_bf16 | rtx-5080 | 0.08632 |  | torch-bf16 0.1885 (0.46) | **torch-bf16-compile 0.1558 (0.55)** | torch-bf16-compile-cg 0.1621 (0.53) | torch-bf16-compile-ro 0.2047 (0.42) | triton-bf16 0.199 (0.43) |
| pj_m16_n4096_k14336_bf16 | rtx-3090 | 0.1695 |  | torch-bf16 0.1823 (0.93) | torch-bf16-compile 0.1434 (1.18) | **torch-bf16-compile-cg 0.1423 (1.19)** | torch-bf16-compile-ro 0.1495 (1.13) | triton-bf16 0.1628 (1.04) |
| pj_m16_n4096_k14336_bf16 | a100 | 0.1341 |  | torch-bf16 0.1188 (1.13) | torch-bf16-compile 0.1126 (1.19) | **torch-bf16-compile-cg 0.1106 (1.21)** | torch-bf16-compile-ro 0.1157 (1.16) | triton-bf16 0.13 (1.03) |
| pj_m16_n4096_k14336_bf16 | rtx-4090 | 0.1741 |  | torch-bf16 0.2099 (0.83) | torch-bf16-compile 0.2017 (0.86) | **torch-bf16-compile-cg 0.2007 (0.87)** | torch-bf16-compile-ro 0.2079 (0.84) | triton-bf16 0.2417 (0.72) |
| pj_m16_n4096_k14336_bf16 | h100 | 0.104 |  | **torch-bf16 0.0537 (1.94)** | torch-bf16-compile 0.0537 (1.94) | torch-bf16-compile-cg 0.0541 (1.92) | torch-bf16-compile-ro 0.0546 (1.90) | triton-bf16 0.0947 (1.10) |
| pj_m16_n4096_k4096_bf16 | rtx-5080 | 0.03629 |  | torch-bf16 0.0619 (0.59) | **torch-bf16-compile 0.0598 (0.61)** | torch-bf16-compile-cg 0.0664 (0.55) | torch-bf16-compile-ro 0.0776 (0.47) | triton-bf16 0.0809 (0.45) |
| pj_m16_n4096_k4096_bf16 | rtx-3090 | 0.0822 |  | torch-bf16 0.0512 (1.61) | **torch-bf16-compile 0.0471 (1.75)** | torch-bf16-compile-cg 0.0471 (1.75) | torch-bf16-compile-ro 0.0512 (1.61) | triton-bf16 0.0512 (1.61) |
| pj_m16_n4096_k4096_bf16 | a100 | 0.05517 |  | torch-bf16 0.0492 (1.12) | torch-bf16-compile 0.0451 (1.22) | **torch-bf16-compile-cg 0.044 (1.25)** | torch-bf16-compile-ro 0.0481 (1.15) | triton-bf16 0.0512 (1.08) |
| pj_m16_n4096_k4096_bf16 | rtx-4090 | 0.07332 |  | torch-bf16 0.1178 (0.62) | torch-bf16-compile 0.0625 (1.17) | **torch-bf16-compile-cg 0.0614 (1.19)** | torch-bf16-compile-ro 0.1178 (0.62) | triton-bf16 0.0758 (0.97) |
| pj_m16_n4096_k4096_bf16 | h100 | 0.05467 |  | **torch-bf16 0.022 (2.49)** | torch-bf16-compile 0.022 (2.49) | torch-bf16-compile-cg 0.0224 (2.44) | torch-bf16-compile-ro 0.0223 (2.45) | triton-bf16 0.0342 (1.60) |
| pj_m16_n6144_k4096_bf16 | rtx-5080 | 0.05299 |  | **torch-bf16 0.071 (0.75)** | torch-bf16-compile 0.081 (0.65) | torch-bf16-compile-cg 0.0851 (0.62) | torch-bf16-compile-ro 0.0867 (0.61) | triton-bf16 0.0867 (0.61) |
| pj_m16_n6144_k4096_bf16 | rtx-3090 | 0.09937 |  | torch-bf16 0.0881 (1.13) | torch-bf16-compile 0.0666 (1.49) | **torch-bf16-compile-cg 0.0655 (1.52)** | torch-bf16-compile-ro 0.0881 (1.13) | triton-bf16 0.0676 (1.47) |
| pj_m16_n6144_k4096_bf16 | a100 | 0.07574 |  | torch-bf16 0.0696 (1.09) | torch-bf16-compile 0.0614 (1.23) | **torch-bf16-compile-cg 0.0594 (1.28)** | torch-bf16-compile-ro 0.0666 (1.14) | triton-bf16 0.0635 (1.19) |
| pj_m16_n6144_k4096_bf16 | rtx-4090 | 0.09756 |  | torch-bf16 0.127 (0.77) | **torch-bf16-compile 0.0901 (1.08)** | torch-bf16-compile-cg 0.0901 (1.08) | torch-bf16-compile-ro 0.126 (0.77) | triton-bf16 0.0954 (1.02) |
| pj_m16_n6144_k4096_bf16 | h100 | 0.05759 |  | torch-bf16 0.0285 (2.02) | **torch-bf16-compile 0.0274 (2.10)** | torch-bf16-compile-cg 0.0277 (2.08) | torch-bf16-compile-ro 0.028 (2.06) | triton-bf16 0.0392 (1.47) |
| pj_m1_n28672_k4096_bf16 | rtx-5080 | 0.2104 |  | **torch-bf16 0.2879 (0.73)** | torch-bf16-compile 0.2903 (0.72) | torch-bf16-compile-cg 0.2976 (0.71) | torch-bf16-compile-ro 0.3065 (0.69) | triton-bf16 0.3236 (0.65) |
| pj_m1_n28672_k4096_bf16 | rtx-3090 | 0.2888 |  | torch-bf16 0.3195 (0.90) | **torch-bf16-compile 0.2806 (1.03)** | torch-bf16-compile-cg 0.2806 (1.03) | torch-bf16-compile-ro 0.3 (0.96) | triton-bf16 0.4014 (0.72) |
| pj_m1_n28672_k4096_bf16 | a100 | 0.263 |  | torch-bf16 0.213 (1.23) | torch-bf16-compile 0.1884 (1.40) | **torch-bf16-compile-cg 0.1843 (1.43)** | torch-bf16-compile-ro 0.2099 (1.25) | triton-bf16 0.213 (1.23) |
| pj_m1_n28672_k4096_bf16 | rtx-4090 | 0.3325 |  | torch-bf16 0.3308 (1.01) | **torch-bf16-compile 0.3164 (1.05)** | torch-bf16-compile-cg 0.3164 (1.05) | torch-bf16-compile-ro 0.3451 (0.96) | triton-bf16 0.3983 (0.83) |
| pj_m1_n28672_k4096_bf16 | h100 | 0.1218 |  | torch-bf16 0.0935 (1.30) | **torch-bf16-compile 0.0928 (1.31)** | torch-bf16-compile-cg 0.0959 (1.27) | torch-bf16-compile-ro 0.0937 (1.30) | triton-bf16 0.1271 (0.96) |
| pj_m1_n4096_k14336_bf16 | rtx-5080 | 0.07798 |  | **torch-bf16 0.1466 (0.53)** | torch-bf16-compile 0.1612 (0.48) | torch-bf16-compile-cg 0.1661 (0.47) | torch-bf16-compile-ro 0.1637 (0.48) | triton-bf16 0.2187 (0.36) |
| pj_m1_n4096_k14336_bf16 | rtx-3090 | 0.1595 |  | torch-bf16 0.1495 (1.07) | torch-bf16-compile 0.1505 (1.06) | torch-bf16-compile-cg 0.1475 (1.08) | **torch-bf16-compile-ro 0.1464 (1.09)** | triton-bf16 0.2304 (0.69) |
| pj_m1_n4096_k14336_bf16 | a100 | 0.1408 |  | torch-bf16 0.1167 (1.21) | torch-bf16-compile 0.1106 (1.27) | **torch-bf16-compile-cg 0.1085 (1.30)** | torch-bf16-compile-ro 0.1157 (1.22) | triton-bf16 0.1536 (0.92) |
| pj_m1_n4096_k14336_bf16 | rtx-4090 | 0.1298 |  | torch-bf16 0.2263 (0.57) | torch-bf16-compile 0.1823 (0.71) | **torch-bf16-compile-cg 0.1802 (0.72)** | torch-bf16-compile-ro 0.2253 (0.58) | triton-bf16 0.2662 (0.49) |
| pj_m1_n4096_k14336_bf16 | h100 | 0.07073 |  | torch-bf16 0.0534 (1.32) | torch-bf16-compile 0.0524 (1.35) | **torch-bf16-compile-cg 0.0514 (1.38)** | torch-bf16-compile-ro 0.054 (1.31) | triton-bf16 0.1125 (0.63) |
| pj_m1_n4096_k4096_bf16 | rtx-5080 | 0.02224 |  | torch-bf16 0.0725 (0.31) | **torch-bf16-compile 0.0714 (0.31)** | torch-bf16-compile-cg 0.0739 (0.30) | torch-bf16-compile-ro 0.0904 (0.25) | triton-bf16 0.0881 (0.25) |
| pj_m1_n4096_k4096_bf16 | rtx-3090 | 0.06345 |  | torch-bf16 0.0512 (1.24) | torch-bf16-compile 0.0543 (1.17) | torch-bf16-compile-cg 0.0532 (1.19) | **torch-bf16-compile-ro 0.0502 (1.26)** | triton-bf16 0.0952 (0.67) |
| pj_m1_n4096_k4096_bf16 | a100 | 0.05572 |  | torch-bf16 0.0481 (1.16) | torch-bf16-compile 0.0492 (1.13) | **torch-bf16-compile-cg 0.0451 (1.24)** | torch-bf16-compile-ro 0.0481 (1.16) | triton-bf16 0.0594 (0.94) |
| pj_m1_n4096_k4096_bf16 | rtx-4090 | 0.03381 |  | torch-bf16 0.085 (0.40) | torch-bf16-compile 0.0686 (0.49) | **torch-bf16-compile-cg 0.0666 (0.51)** | torch-bf16-compile-ro 0.084 (0.40) | triton-bf16 0.0973 (0.35) |
| pj_m1_n4096_k4096_bf16 | h100 | 0.02269 |  | torch-bf16 0.0204 (1.11) | torch-bf16-compile 0.0216 (1.05) | torch-bf16-compile-cg 0.0207 (1.10) | **torch-bf16-compile-ro 0.0203 (1.12)** | triton-bf16 0.0404 (0.56) |
| pj_m1_n6144_k4096_bf16 | rtx-5080 | 0.02786 |  | torch-bf16 0.0847 (0.33) | **torch-bf16-compile 0.08 (0.35)** | torch-bf16-compile-cg 0.0938 (0.30) | torch-bf16-compile-ro 0.1023 (0.27) | triton-bf16 0.1016 (0.27) |
| pj_m1_n6144_k4096_bf16 | rtx-3090 | 0.08134 |  | torch-bf16 0.086 (0.95) | torch-bf16-compile 0.0676 (1.20) | **torch-bf16-compile-cg 0.0645 (1.26)** | torch-bf16-compile-ro 0.086 (0.95) | triton-bf16 0.1321 (0.62) |
| pj_m1_n6144_k4096_bf16 | a100 | 0.08235 |  | torch-bf16 0.0655 (1.26) | torch-bf16-compile 0.0625 (1.32) | **torch-bf16-compile-cg 0.0594 (1.39)** | torch-bf16-compile-ro 0.0666 (1.24) | triton-bf16 0.0788 (1.05) |
| pj_m1_n6144_k4096_bf16 | rtx-4090 | 0.04508 |  | **torch-bf16 0.0932 (0.48)** | torch-bf16-compile 0.0944 (0.48) | torch-bf16-compile-cg 0.0932 (0.48) | torch-bf16-compile-ro 0.0952 (0.47) | triton-bf16 0.1219 (0.37) |
| pj_m1_n6144_k4096_bf16 | h100 | 0.03629 |  | torch-bf16 0.0301 (1.21) | torch-bf16-compile 0.0283 (1.28) | **torch-bf16-compile-cg 0.0279 (1.30)** | torch-bf16-compile-ro 0.0308 (1.18) | triton-bf16 0.047 (0.77) |
| pj_m2048_n28672_k4096_bf16 | rtx-5080 | 3.987 |  | torch-bf16 4.06 (0.98) | **torch-bf16-compile 3.976 (1.00)** | torch-bf16-compile-cg 3.984 (1.00) | torch-bf16-compile-ro 4.086 (0.98) | triton-bf16 4.095 (0.97) |
| pj_m2048_n28672_k4096_bf16 | rtx-3090 | 7.057 |  | torch-bf16 7.424 (0.95) | torch-bf16-compile 7.768 (0.91) | torch-bf16-compile-cg 7.768 (0.91) | torch-bf16-compile-ro 7.846 (0.90) | **triton-bf16 7.106 (0.99)** |
| pj_m2048_n28672_k4096_bf16 | a100 | 3.372 |  | torch-bf16 1.76 (1.92) | torch-bf16-compile 2.212 (1.52) | torch-bf16-compile-cg 1.747 (1.93) | **torch-bf16-compile-ro 1.742 (1.94)** | triton-bf16 2.084 (1.62) |
| pj_m2048_n28672_k4096_bf16 | rtx-4090 | 6.35 |  | torch-bf16 6.187 (1.03) | torch-bf16-compile 6.175 (1.03) | **torch-bf16-compile-cg 6.174 (1.03)** | torch-bf16-compile-ro 6.186 (1.03) | triton-bf16 6.217 (1.02) |
| pj_m2048_n28672_k4096_bf16 | h100 | 3.464 |  | **torch-bf16 1.288 (2.69)** | torch-bf16-compile 1.388 (2.50) | torch-bf16-compile-cg 1.395 (2.48) | torch-bf16-compile-ro 1.395 (2.48) | triton-bf16 1.809 (1.91) |
| pj_m2048_n4096_k14336_bf16 | rtx-5080 | 2.264 |  | torch-bf16 2.128 (1.06) | **torch-bf16-compile 2.123 (1.07)** | torch-bf16-compile-cg 2.13 (1.06) | torch-bf16-compile-ro 2.127 (1.06) | triton-bf16 2.189 (1.03) |
| pj_m2048_n4096_k14336_bf16 | rtx-3090 | 3.961 |  | torch-bf16 4.36 (0.91) | **torch-bf16-compile 4.223 (0.94)** | torch-bf16-compile-cg 4.278 (0.93) | torch-bf16-compile-ro 4.278 (0.93) | not run |
| pj_m2048_n4096_k14336_bf16 | a100 | 1.828 |  | **torch-bf16 0.8878 (2.06)** | torch-bf16-compile 1.141 (1.60) | torch-bf16-compile-cg 1.14 (1.60) | torch-bf16-compile-ro 1.055 (1.73) | triton-bf16 1.089 (1.68) |
| pj_m2048_n4096_k14336_bf16 | rtx-4090 | 3.251 |  | torch-bf16 3.132 (1.04) | **torch-bf16-compile 3.119 (1.04)** | torch-bf16-compile-cg 3.119 (1.04) | torch-bf16-compile-ro 3.13 (1.04) | triton-bf16 3.135 (1.04) |
| pj_m2048_n4096_k14336_bf16 | h100 | 1.307 |  | torch-bf16 0.8463 (1.54) | torch-bf16-compile 0.6816 (1.92) | **torch-bf16-compile-cg 0.6779 (1.93)** | torch-bf16-compile-ro 0.6785 (1.93) | triton-bf16 0.986 (1.33) |
| pj_m2048_n4096_k4096_bf16 | rtx-5080 | 0.6779 |  | **torch-bf16 0.6175 (1.10)** | torch-bf16-compile 0.6208 (1.09) | torch-bf16-compile-cg 0.624 (1.09) | torch-bf16-compile-ro 0.6236 (1.09) | triton-bf16 0.6417 (1.06) |
| pj_m2048_n4096_k4096_bf16 | rtx-3090 | 1.151 |  | torch-bf16 1.216 (0.95) | torch-bf16-compile 1.101 (1.05) | torch-bf16-compile-cg 1.099 (1.05) | **torch-bf16-compile-ro 1.044 (1.10)** | triton-bf16 1.133 (1.02) |
| pj_m2048_n4096_k4096_bf16 | a100 | 0.5977 |  | torch-bf16 0.3092 (1.93) | torch-bf16-compile 0.3983 (1.50) | torch-bf16-compile-cg 0.3963 (1.51) | torch-bf16-compile-ro 0.3963 (1.51) | **triton-bf16 0.3082 (1.94)** |
| pj_m2048_n4096_k4096_bf16 | rtx-4090 | 0.9449 |  | torch-bf16 0.8991 (1.05) | **torch-bf16-compile 0.898 (1.05)** | torch-bf16-compile-cg 0.898 (1.05) | torch-bf16-compile-ro 0.898 (1.05) | triton-bf16 0.9042 (1.04) |
| pj_m2048_n4096_k4096_bf16 | h100 | 0.3732 |  | **torch-bf16 0.1689 (2.21)** | torch-bf16-compile 0.1738 (2.15) | torch-bf16-compile-cg 0.1738 (2.15) | torch-bf16-compile-ro 0.1739 (2.15) | triton-bf16 0.267 (1.40) |
| pj_m2048_n6144_k4096_bf16 | rtx-5080 | 0.9364 |  | torch-bf16 0.9135 (1.03) | **torch-bf16-compile 0.9095 (1.03)** | torch-bf16-compile-cg 0.9177 (1.02) | torch-bf16-compile-ro 0.9118 (1.03) | triton-bf16 0.9257 (1.01) |
| pj_m2048_n6144_k4096_bf16 | rtx-3090 | 1.633 |  | torch-bf16 1.716 (0.95) | torch-bf16-compile 1.671 (0.98) | **torch-bf16-compile-cg 1.608 (1.02)** | torch-bf16-compile-ro 1.653 (0.99) | triton-bf16 1.718 (0.95) |
| pj_m2048_n6144_k4096_bf16 | a100 | 0.7945 |  | **torch-bf16 0.4106 (1.94)** | torch-bf16-compile 0.4116 (1.93) | torch-bf16-compile-cg 0.4106 (1.94) | torch-bf16-compile-ro 0.4106 (1.94) | triton-bf16 0.4639 (1.71) |
| pj_m2048_n6144_k4096_bf16 | rtx-4090 | 1.401 |  | torch-bf16 1.343 (1.04) | torch-bf16-compile 1.338 (1.05) | **torch-bf16-compile-cg 1.337 (1.05)** | torch-bf16-compile-ro 1.343 (1.04) | triton-bf16 1.348 (1.04) |
| pj_m2048_n6144_k4096_bf16 | h100 | 0.5863 |  | **torch-bf16 0.265 (2.21)** | torch-bf16-compile 0.2815 (2.08) | torch-bf16-compile-cg 0.2804 (2.09) | torch-bf16-compile-ro 0.2795 (2.10) | triton-bf16 0.3433 (1.71) |

### MLP chains

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand Triton |
|---|---|---:|---|---|---|---|---|---|
| mlp2_2048 | rtx-5080 | 1.816 | **chain-lt2 1.769 (1.03)**<br>chain-streams 1.796 (1.01)<br>chain-lt 1.849 (0.98)<br>chain-eager 1.875 (0.97)<br>chain-cutlass 2.211 (0.82) | torch-eager 1.89 (0.96) | torch-compile 1.888 (0.96) | torch-compile-cg 1.89 (0.96) | torch-compile-ro 1.895 (0.96) | triton 2.456 (0.74) |
| mlp2_2048 | rtx-3090 | 3.252 | chain-lt2 8.328 (0.39)<br>chain-lt 8.335 (0.39)<br>chain-streams 8.454 (0.38)<br>chain-eager 8.893 (0.37)<br>chain-cutlass 10.98 (0.30) | torch-eager 3.764 (0.86) | **torch-compile 3.694 (0.88)** | torch-compile-cg 3.735 (0.87) | torch-compile-ro 3.779 (0.86) | triton 4.202 (0.77) |
| mlp2_2048 | a100 | 3.543 | **chain-streams 3.748 (0.95)**<br>chain-lt2 3.783 (0.94)<br>chain-eager 3.957 (0.90)<br>chain-lt 4.015 (0.88)<br>chain-cutlass 4.604 (0.77) | torch-eager 4.899 (0.72) | torch-compile 3.983 (0.89) | torch-compile-cg 3.937 (0.90) | torch-compile-ro 3.932 (0.90) | triton 3.979 (0.89) |
| mlp2_2048 | rtx-4090 | 2.375 | **chain-cutlass 2.501 (0.95)**<br>chain-lt2 2.534 (0.94)<br>chain-streams 2.556 (0.93)<br>chain-lt 2.556 (0.93)<br>chain-eager 2.599 (0.91) | torch-eager 2.619 (0.91) | torch-compile 2.585 (0.92) | torch-compile-cg 2.578 (0.92) | torch-compile-ro 2.579 (0.92) | triton 3.039 (0.78) |
| mlp2_2048 | h100 | 1.886 | chain-streams 1.988 (0.95)<br>chain-lt2 1.989 (0.95)<br>chain-lt 2.003 (0.94)<br>chain-eager 2.033 (0.93)<br>chain-cutlass 2.258 (0.84) | torch-eager 1.793 (1.05) | torch-compile 1.778 (1.06) | torch-compile-cg 1.706 (1.11) | **torch-compile-ro 1.687 (1.12)** | triton 2.358 (0.80) |
| mlp_2048 | rtx-5080 | 0.8779 | **chain-lt 0.9114 (0.96)**<br>chain-eager 0.9257 (0.95)<br>chain-cutlass 1.089 (0.81) | torch-eager 0.9329 (0.94) | torch-compile 0.9523 (0.92) | torch-compile-cg 0.9574 (0.92) | torch-compile-ro 0.9564 (0.92) | triton 1.227 (0.72) |
| mlp_2048 | rtx-3090 | 1.618 | chain-lt 2.772 (0.58)<br>chain-eager 2.816 (0.57)<br>chain-cutlass 3.41 (0.47) | torch-eager 1.745 (0.93) | **torch-compile 1.728 (0.94)** | torch-compile-cg 1.908 (0.85) | torch-compile-ro 1.848 (0.88) | triton 2.186 (0.74) |
| mlp_2048 | a100 | 1.772 | chain-lt 2.502 (0.71)<br>chain-eager 2.527 (0.70)<br>chain-cutlass 2.966 (0.60) | torch-eager 1.974 (0.90) | torch-compile 1.978 (0.90) | torch-compile-cg 1.981 (0.89) | **torch-compile-ro 1.968 (0.90)** | triton 1.991 (0.89) |
| mlp_2048 | rtx-4090 | 1.182 | **chain-cutlass 1.251 (0.94)**<br>chain-lt 1.279 (0.92)<br>chain-eager 1.297 (0.91) | torch-eager 1.296 (0.91) | torch-compile 1.295 (0.91) | torch-compile-cg 1.293 (0.91) | torch-compile-ro 1.292 (0.91) | triton 1.522 (0.78) |
| mlp_2048 | h100 | 0.9296 | chain-lt 0.8812 (1.05)<br>chain-eager 0.8953 (1.04)<br>chain-cutlass 1.014 (0.92) | torch-eager 0.8398 (1.11) | torch-compile 0.8438 (1.10) | torch-compile-cg 0.8406 (1.11) | **torch-compile-ro 0.8394 (1.11)** | triton 1.152 (0.81) |
| mlp_512 | rtx-5080 | 0.03871 | **chain-lt 0.043 (0.90)**<br>chain-eager 0.0451 (0.86)<br>chain-cutlass 0.1065 (0.36) | torch-eager 0.0481 (0.80) | torch-compile 0.048 (0.81) | torch-compile-cg 0.055 (0.70) | torch-compile-ro 0.0666 (0.58) | triton 0.0564 (0.69) |
| mlp_512 | rtx-3090 | 0.06864 | chain-lt 0.0522 (1.31)<br>chain-eager 0.0573 (1.20)<br>chain-cutlass 0.1249 (0.55) | torch-eager 0.0563 (1.22) | torch-compile 0.0522 (1.31) | torch-compile-cg 0.0502 (1.37) | torch-compile-ro 0.0543 (1.26) | **triton 0.044 (1.56)** |
| mlp_512 | a100 | 0.1054 | chain-lt 0.0788 (1.34)<br>chain-eager 0.084 (1.26)<br>chain-cutlass 0.2611 (0.40) | **torch-eager 0.0666 (1.58)** | torch-compile 0.0768 (1.37) | torch-compile-cg 0.0737 (1.43) | torch-compile-ro 0.0799 (1.32) | triton 0.0768 (1.37) |
| mlp_512 | rtx-4090 | 0.07976 | chain-lt 0.0543 (1.47)<br>chain-eager 0.0573 (1.39)<br>chain-cutlass 0.1751 (0.46) | torch-eager 0.0573 (1.39) | torch-compile 0.0502 (1.59) | **torch-compile-cg 0.0492 (1.62)** | torch-compile-ro 0.0543 (1.47) | triton 0.0604 (1.32) |
| mlp_512 | h100 | 0.04546 | chain-lt 0.0334 (1.36)<br>chain-eager 0.0422 (1.08)<br>chain-cutlass 0.1179 (0.39) | torch-eager 0.0341 (1.33) | torch-compile 0.0315 (1.44) | **torch-compile-cg 0.0303 (1.50)** | torch-compile-ro 0.0319 (1.42) | triton 0.0538 (0.84) |
| smlp_4096_256_1024 | rtx-5080 | 0.157 | **chain-lt 0.1556 (1.01)**<br>chain-cutlass 0.1577 (1.00)<br>chain-eager 0.1679 (0.93) | torch-eager 0.1808 (0.87) | torch-compile 0.1687 (0.93) | torch-compile-cg 0.1723 (0.91) | torch-compile-ro 0.1882 (0.83) | triton 0.1976 (0.79) |
| smlp_4096_256_1024 | rtx-3090 | 0.2814 | chain-cutlass 0.258 (1.09)<br>chain-eager 0.2867 (0.98)<br>chain-lt 0.3389 (0.83) | torch-eager 0.3062 (0.92) | **torch-compile 0.2324 (1.21)** | torch-compile-cg 0.2324 (1.21) | torch-compile-ro 0.2775 (1.01) | triton 0.2734 (1.03) |
| smlp_4096_256_1024 | a100 | 0.2877 | chain-eager 0.3707 (0.78)<br>chain-lt 0.4198 (0.69)<br>chain-cutlass 0.4598 (0.63) | torch-eager 0.3144 (0.92) | torch-compile 0.3451 (0.83) | torch-compile-cg 0.342 (0.84) | torch-compile-ro 0.3635 (0.79) | **triton 0.3082 (0.93)** |
| smlp_4096_256_1024 | rtx-4090 | 0.2083 | chain-eager 0.2263 (0.92)<br>chain-cutlass 0.2632 (0.79)<br>chain-lt 0.2949 (0.71) | torch-eager 0.2417 (0.86) | torch-compile 0.1935 (1.08) | **torch-compile-cg 0.1915 (1.09)** | torch-compile-ro 0.2171 (0.96) | triton 0.2243 (0.93) |
| smlp_4096_256_1024 | h100 | 0.1765 | **chain-lt 0.1244 (1.42)**<br>chain-eager 0.135 (1.31)<br>chain-cutlass 0.1832 (0.96) | torch-eager 0.2714 (0.65) | torch-compile 0.1342 (1.31) | torch-compile-cg 0.1335 (1.32) | torch-compile-ro 0.1298 (1.36) | triton 0.1627 (1.08) |
| mlp2_2048_tf32 | rtx-5080 | 1.569 |  | torch-eager-tf32 1.553 (1.01) | torch-compile-tf32 1.313 ✗gate | torch-compile-cg-tf32 1.313 ✗gate | **torch-compile-ro-tf32 1.549 (1.01)** | triton-tf32 1.465 ✗gate |
| mlp2_2048_tf32 | rtx-3090 | 2.857 |  | torch-eager-tf32 2.746 (1.04) | torch-compile-tf32 2.47 ✗gate | torch-compile-cg-tf32 2.515 ✗gate | **torch-compile-ro-tf32 2.583 (1.11)** | triton-tf32 2.459 ✗gate |
| mlp2_2048_tf32 | a100 | 1.211 |  | torch-eager-tf32 0.7608 (1.59) | torch-compile-tf32 0.7066 (1.71) | torch-compile-cg-tf32 0.6994 (1.73) | **torch-compile-ro-tf32 0.6984 (1.73)** | triton-tf32 0.7711 ✗gate |
| mlp2_2048_tf32 | rtx-4090 | 2.103 |  | torch-eager-tf32 1.888 (1.11) | torch-compile-tf32 1.853 ✗gate | torch-compile-cg-tf32 1.848 ✗gate | **torch-compile-ro-tf32 1.846 (1.14)** | triton-tf32 1.981 ✗gate |
| mlp2_2048_tf32 | h100 | 0.5719 |  | torch-eager-tf32 0.3973 (1.44) | torch-compile-tf32 0.3792 (1.51) | torch-compile-cg-tf32 0.3912 (1.46) | **torch-compile-ro-tf32 0.367 (1.56)** | triton-tf32 0.9441 ✗gate |
| mlp_2048_tf32 | rtx-5080 | 0.7861 |  | **torch-eager-tf32 0.7697 (1.02)** | torch-compile-tf32 0.6569 ✗gate | torch-compile-cg-tf32 0.6635 ✗gate | torch-compile-ro-tf32 0.7854 (1.00) | triton-tf32 0.7535 ✗gate |
| mlp_2048_tf32 | rtx-3090 | 1.406 |  | torch-eager-tf32 1.411 (1.00) | torch-compile-tf32 1.228 ✗gate | torch-compile-cg-tf32 1.028 ✗gate | **torch-compile-ro-tf32 1.158 (1.21)** | triton-tf32 1.249 ✗gate |
| mlp_2048_tf32 | a100 | 0.656 |  | torch-eager-tf32 0.3553 (1.85) | torch-compile-tf32 0.3604 (1.82) | torch-compile-cg-tf32 0.3564 (1.84) | **torch-compile-ro-tf32 0.3523 (1.86)** | triton-tf32 0.4393 ✗gate |
| mlp_2048_tf32 | rtx-4090 | 1.031 |  | torch-eager-tf32 0.9317 (1.11) | torch-compile-tf32 0.9175 ✗gate | torch-compile-cg-tf32 0.9155 ✗gate | **torch-compile-ro-tf32 0.9257 (1.11)** | triton-tf32 0.9923 ✗gate |
| mlp_2048_tf32 | h100 | 0.2924 |  | torch-eager-tf32 0.1868 (1.57) | torch-compile-tf32 0.1752 (1.67) | torch-compile-cg-tf32 0.1756 (1.67) | **torch-compile-ro-tf32 0.1733 (1.69)** | triton-tf32 0.4859 ✗gate |

### Attention / softmax / long rows

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | hand Triton | hand CUDA |
|---|---|---:|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-5080 | 0.03558 | chain-eager 0.041 (0.87) | torch-eager 0.0548 (0.65) | torch-compile 0.0466 (0.76) | torch-compile-cg 0.0514 (0.69)<br>torch-sdpa-compile-cg 0.054 (0.66) | torch-compile-ro 0.0537 (0.66) | torch-sdpa-compile 0.0567 (0.63)<br>torch-sdpa 0.0632 (0.56) | triton 0.1969 (0.18) | **chain-fused 0.0348 (1.02)** |
| attn_1024_64 | rtx-3090 | 0.04884 | chain-eager 0.0717 (0.68) | torch-eager 0.0881 (0.55) | torch-compile 0.0625 (0.78) | torch-compile-cg 0.0573 (0.85)<br>torch-sdpa-compile-cg 0.0778 (0.63) | **torch-compile-ro 0.0461 (1.06)** | torch-sdpa 0.085 (0.57)<br>torch-sdpa-compile 0.085 (0.57) | triton 0.2273 (0.21) | chain-fused 0.0584 (0.84) |
| attn_1024_64 | a100 | 0.06892 | chain-eager 0.1137 (0.61) | torch-eager 0.0891 (0.77) | torch-compile 0.0717 (0.96) | torch-compile-cg 0.0676 (1.02)<br>torch-sdpa-compile-cg 0.0727 (0.95) | **torch-compile-ro 0.0584 (1.18)** | torch-sdpa 0.1024 (0.67)<br>torch-sdpa-compile 0.1311 (0.53) | triton 0.3154 (0.22) | chain-fused 0.085 (0.81) |
| attn_1024_64 | rtx-4090 | 0.04562 | chain-eager 0.0645 (0.71) | torch-eager 0.0829 (0.55) | torch-compile 0.0471 (0.97) | **torch-sdpa-compile-cg 0.042 (1.09)**<br>torch-compile-cg 0.044 (1.04) | torch-compile-ro 0.044 (1.04) | torch-sdpa-compile 0.0492 (0.93)<br>torch-sdpa 0.0727 (0.63) | triton 0.3174 (0.14) | chain-fused 0.0573 (0.80) |
| attn_1024_64 | h100 | 0.04302 | chain-eager 0.0418 (1.03) | torch-eager 0.0656 (0.66) | torch-compile 0.0391 (1.10) | **torch-compile-cg 0.0246 (1.75)**<br>torch-sdpa-compile-cg 0.0258 (1.67) | torch-compile-ro 0.0248 (1.73) | torch-sdpa-compile 0.0396 (1.09)<br>torch-sdpa 0.0512 (0.84) | triton 0.2316 (0.19) | chain-fused 0.0359 (1.20) |
| attn_2048_128 | rtx-5080 | 0.1131 | chain-eager 0.1393 (0.81) | torch-eager 0.1618 (0.70) | torch-compile 0.1203 (0.94) | torch-sdpa-compile-cg 0.1167 (0.97)<br>torch-compile-cg 0.1223 (0.93) | torch-compile-ro 0.1265 (0.89) | **torch-sdpa-compile 0.1141 (0.99)**<br>torch-sdpa 0.1538 (0.74) | triton 0.8353 (0.14) | chain-fused 0.127 (0.89) |
| attn_2048_128 | rtx-3090 | 0.2682 | chain-eager 0.2908 (0.92) | torch-eager 0.3738 (0.72) | torch-compile 0.1976 (1.36) | **torch-compile-cg 0.1802 (1.49)**<br>torch-sdpa-compile-cg 0.1802 (1.49) | torch-compile-ro 0.1853 (1.45) | torch-sdpa-compile 0.1843 (1.46)<br>torch-sdpa 0.2775 (0.97) | triton 0.9011 (0.30) | chain-fused 0.2611 (1.03) |
| attn_2048_128 | a100 | 0.2923 | chain-eager 0.2734 (1.07) | torch-eager 0.3277 (0.89) | torch-compile 0.1997 (1.46) | torch-compile-cg 0.2089 (1.40)<br>torch-sdpa-compile-cg 0.2253 (1.30) | torch-compile-ro 0.2089 (1.40) | **torch-sdpa-compile 0.1812 (1.61)**<br>torch-sdpa 0.3011 (0.97) | triton 1.402 (0.21) | chain-fused 0.2683 (1.09) |
| attn_2048_128 | rtx-4090 | 0.1707 | chain-eager 0.212 (0.81) | torch-eager 0.2417 (0.71) | torch-compile 0.1597 (1.07) | torch-compile-cg 0.1577 (1.08)<br>torch-sdpa-compile-cg 0.1628 (1.05) | **torch-compile-ro 0.1485 (1.15)** | torch-sdpa-compile 0.1679 (1.02)<br>torch-sdpa 0.2284 (0.75) | triton 1.264 (0.14) | chain-fused 0.2048 (0.83) |
| attn_2048_128 | h100 | 0.1257 | chain-eager 0.1205 (1.04) | torch-eager 0.1403 (0.90) | torch-compile 0.0868 (1.45) | torch-compile-cg 0.0842 (1.49)<br>torch-sdpa-compile-cg 0.0971 (1.29) | **torch-compile-ro 0.0841 (1.49)** | torch-sdpa-compile 0.0988 (1.27)<br>torch-sdpa 0.1236 (1.02) | triton 1.089 (0.12) | chain-fused 0.1091 (1.15) |
| attn_2048_64 | rtx-5080 | 0.07282 | chain-eager 0.1065 (0.68) | torch-eager 0.1292 (0.56) | torch-compile 0.0972 (0.75) | torch-sdpa-compile-cg 0.0966 (0.75)<br>torch-compile-cg 0.106 (0.69) | torch-compile-ro 0.0941 (0.77) | **torch-sdpa-compile 0.0938 (0.78)**<br>torch-sdpa 0.1241 (0.59) | triton 0.3434 (0.21) | chain-fused 0.0942 (0.77) |
| attn_2048_64 | rtx-3090 | 0.1244 | chain-eager 0.2345 (0.53) | torch-eager 0.3042 (0.41) | torch-compile 0.1372 (0.91) | torch-compile-cg 0.1352 (0.92)<br>torch-sdpa-compile-cg 0.1925 (0.65) | **torch-compile-ro 0.1311 (0.95)** | torch-sdpa-compile 0.1925 (0.65)<br>torch-sdpa 0.2222 (0.56) | triton 0.3901 (0.32) | chain-fused 0.2048 (0.61) |
| attn_2048_64 | a100 | 0.1844 | chain-eager 0.2109 (0.87) | torch-eager 0.254 (0.73) | **torch-compile 0.126 (1.46)** | torch-sdpa-compile-cg 0.1423 (1.30)<br>torch-compile-cg 0.1567 (1.18) | torch-compile-ro 0.1352 (1.36) | torch-sdpa-compile 0.1516 (1.22)<br>torch-sdpa 0.2253 (0.82) | triton 0.4659 (0.40) | chain-fused 0.2028 (0.91) |
| attn_2048_64 | rtx-4090 | 0.1238 | chain-eager 0.1731 (0.72) | torch-eager 0.2048 (0.60) | torch-compile 0.1096 (1.13) | **torch-sdpa-compile-cg 0.1024 (1.21)**<br>torch-compile-cg 0.1075 (1.15) | torch-compile-ro 0.1075 (1.15) | torch-sdpa-compile 0.1055 (1.17)<br>torch-sdpa 0.1853 (0.67) | triton 0.6267 (0.20) | chain-fused 0.1628 (0.76) |
| attn_2048_64 | h100 | 0.06664 | chain-eager 0.0976 (0.68) | torch-eager 0.118 (0.56) | torch-compile 0.0614 (1.09) | **torch-compile-cg 0.0573 (1.16)**<br>torch-sdpa-compile-cg 0.0603 (1.11) | torch-compile-ro 0.0589 (1.13) | torch-sdpa-compile 0.0646 (1.03)<br>torch-sdpa 0.1004 (0.66) | triton 0.4572 (0.15) | chain-fused 0.0816 (0.82) |
| causal_1024_256 | rtx-5080 | 13.45 |  | torch-eager 95.4 (0.14) | torch-compile 55.55 (0.24) | **torch-compile-cg 45.22 (0.30)** | torch-compile-ro 46.86 (0.29) |  |  |  |
| causal_1024_256 | rtx-3090 | 25.37 |  | torch-eager 358.5 (0.07) | torch-compile 204.3 (0.12) | **torch-compile-cg 179.4 (0.14)** | torch-compile-ro 180.3 (0.14) |  |  |  |
| causal_1024_256 | a100 | 53.11 |  | torch-eager 406.4 (0.13) | torch-compile 209 (0.25) | **torch-compile-cg 192.8 (0.28)** | torch-compile-ro 193.2 (0.27) |  |  |  |
| causal_1024_256 | rtx-4090 | 33.74 |  | torch-eager 395.9 (0.09) | torch-compile 269.8 (0.13) | **torch-compile-cg 244.1 (0.14)** | torch-compile-ro 245.5 (0.14) |  |  |  |
| causal_1024_256 | h100 | 21.44 |  | torch-eager 204.1 (0.11) | torch-compile 117.1 (0.18) | **torch-compile-cg 116.7 (0.18)** | torch-compile-ro 117.7 (0.18) |  |  |  |
| decode_1024_256 | rtx-5080 | 8.674 | chain-nocache 23.72 (0.37) | torch-eager 51.11 (0.17) | torch-compile 48.29 (0.18) | torch-compile-cg 45.82 (0.19) | torch-compile-ro 45.88 (0.19) |  |  | **chain-cache 16.41 (0.53)** |
| decode_1024_256 | rtx-3090 | 17.58 | chain-nocache 37.71 (0.47) | torch-eager 198.1 (0.09) | torch-compile 180.2 (0.10) | torch-compile-cg 179.5 (0.10) | torch-compile-ro 179.3 (0.10) |  |  | **chain-cache 33.16 (0.53)** |
| decode_1024_256 | a100 | 45.81 | **chain-nocache 53.39 (0.86)** | torch-eager 238.3 (0.19) | torch-compile 184.7 (0.25) | torch-compile-cg 184.6 (0.25) | torch-compile-ro 188.7 (0.24) |  |  | chain-cache 65.35 (0.70) |
| decode_1024_256 | rtx-4090 | 21.02 | chain-nocache 57.08 (0.37) | torch-eager 213.6 (0.10) | torch-compile 221 (0.10) | torch-compile-cg 238.9 (0.09) | torch-compile-ro 239 (0.09) |  |  | **chain-cache 51.3 (0.41)** |
| decode_1024_256 | h100 | 28.48 | chain-nocache 30.19 (0.94) | torch-eager 101 (0.28) | torch-compile 98.68 (0.29) | torch-compile-cg 112.3 (0.25) | torch-compile-ro 113.2 (0.25) |  |  | **chain-cache 19.66 (1.45)** |
| gen_1024_512_256 | rtx-5080 | 6.862 |  | torch-eager 48.57 (0.14) | torch-compile 28.53 (0.24) | **torch-compile-cg 23.08 (0.30)** | torch-compile-ro 23.37 (0.29) |  |  |  |
| gen_1024_512_256 | rtx-3090 | 12.93 |  | torch-eager 190.7 (0.07) | torch-compile 100.6 (0.13) | **torch-compile-cg 93.44 (0.14)** | torch-compile-ro 93.6 (0.14) |  |  |  |
| gen_1024_512_256 | a100 | 24.18 |  | torch-eager 202.9 (0.12) | torch-compile 104 (0.23) | **torch-compile-cg 92.24 (0.26)** | torch-compile-ro 92.28 (0.26) |  |  |  |
| gen_1024_512_256 | rtx-4090 | 17.01 |  | torch-eager 194.1 (0.09) | torch-compile 122.9 (0.14) | torch-compile-cg 121 (0.14) | **torch-compile-ro 120.9 (0.14)** |  |  |  |
| gen_1024_512_256 | h100 | 10.82 |  | torch-eager 102.6 (0.11) | **torch-compile 56.96 (0.19)** | torch-compile-cg 57.6 (0.19) | torch-compile-ro 58.68 (0.18) |  |  |  |
| lsm_1_131072 | rtx-5080 | 0.0165 |  | **torch-eager 0.0426 (0.39)** | torch-compile 0.0468 (0.35) | torch-compile-cg 0.045 (0.37)<br>torch-sdpa-compile-cg 0.0453 (0.36) | torch-compile-ro 0.0472 (0.35) | torch-sdpa-compile 0.0465 (0.35)<br>torch-sdpa 0.0565 (0.29) |  |  |
| lsm_1_131072 | rtx-3090 | 0.03164 |  | torch-eager 0.0481 (0.66) | torch-compile 0.0307 (1.03) | **torch-compile-cg 0.0256 (1.24)**<br>torch-sdpa-compile-cg 0.0399 (0.79) | torch-compile-ro 0.0256 (1.24) | torch-sdpa-compile 0.0584 (0.54)<br>torch-sdpa 0.0727 (0.44) |  |  |
| lsm_1_131072 | a100 | 0.03691 |  | torch-eager 0.0707 (0.52) | torch-compile 0.044 (0.84) | torch-compile-cg 0.0348 (1.06)<br>torch-sdpa-compile-cg 0.0348 (1.06) | **torch-compile-ro 0.0317 (1.16)** | torch-sdpa-compile 0.0461 (0.80)<br>torch-sdpa 0.1085 (0.34) |  |  |
| lsm_1_131072 | rtx-4090 | 0.02959 |  | torch-eager 0.0573 (0.52) | torch-compile 0.0389 (0.76) | torch-compile-cg 0.0328 (0.90)<br>torch-sdpa-compile-cg 0.0328 (0.90) | **torch-compile-ro 0.0317 (0.93)** | torch-sdpa-compile 0.0389 (0.76)<br>torch-sdpa 0.0993 (0.30) |  |  |
| lsm_1_131072 | h100 | 0.02653 |  | torch-eager 0.0414 (0.64) | torch-compile 0.0222 (1.20) | **torch-sdpa-compile-cg 0.0184 (1.44)**<br>torch-compile-cg 0.0198 (1.34) | torch-compile-ro 0.0193 (1.37) | torch-sdpa-compile 0.023 (1.15)<br>torch-sdpa 0.0604 (0.44) |  |  |
| lsm_32_8192 | rtx-5080 | 0.0162 |  | torch-eager 0.0338 (0.48) | torch-compile 0.028 (0.58) | torch-sdpa-compile-cg 0.033 (0.49)<br>torch-compile-cg 0.0331 (0.49) | torch-compile-ro 0.0325 (0.50) | **torch-sdpa 0.0193 (0.84)**<br>torch-sdpa-compile 0.0266 (0.61) | triton 0.0376 (0.43) |  |
| lsm_32_8192 | rtx-3090 | 0.03221 |  | torch-eager 0.0328 (0.98) | torch-compile 0.0122 (2.64) | **torch-compile-cg 0.0102 (3.16)**<br>torch-sdpa-compile-cg 0.0102 (3.16) | torch-compile-ro 0.0113 (2.85) | torch-sdpa-compile 0.0113 (2.85)<br>torch-sdpa 0.0164 (1.96) | triton 0.0132 (2.44) |  |
| lsm_32_8192 | a100 | 0.0392 |  | torch-eager 0.0532 (0.74) | torch-compile 0.0184 (2.13) | torch-compile-cg 0.0205 (1.91)<br>torch-sdpa-compile-cg 0.0215 (1.82) | torch-compile-ro 0.0195 (2.01) | torch-sdpa-compile 0.0195 (2.01)<br>torch-sdpa 0.0266 (1.47) | **triton 0.0174 (2.25)** |  |
| lsm_32_8192 | rtx-4090 | 0.03271 |  | torch-eager 0.0389 (0.84) | torch-compile 0.0143 (2.29) | **torch-compile-cg 0.0123 (2.66)**<br>torch-sdpa-compile-cg 0.0123 (2.66) | torch-compile-ro 0.0133 (2.46) | torch-sdpa-compile 0.0133 (2.46)<br>torch-sdpa 0.0205 (1.60) | triton 0.0154 (2.12) |  |
| lsm_32_8192 | h100 | 0.02521 |  | torch-eager 0.0398 (0.63) | torch-compile 0.0116 (2.17) | **torch-compile-cg 0.0088 (2.86)**<br>torch-sdpa-compile-cg 0.0132 (1.91) | torch-compile-ro 0.0099 (2.55) | torch-sdpa 0.0146 (1.73)<br>torch-sdpa-compile 0.0276 (0.91) | triton 0.0115 (2.19) |  |
| lsm_4_32768 | rtx-5080 | 0.01451 |  | torch-eager 0.0335 (0.43) | torch-compile 0.0465 (0.31) | torch-sdpa-compile-cg 0.0368 (0.39)<br>torch-compile-cg 0.0375 (0.39) | torch-compile-ro 0.0377 (0.38) | **torch-sdpa 0.0254 (0.57)**<br>torch-sdpa-compile 0.0476 (0.30) | not run |  |
| lsm_4_32768 | rtx-3090 | 0.02655 |  | torch-eager 0.0399 (0.67) | torch-compile 0.0225 (1.18) | **torch-sdpa-compile-cg 0.0164 (1.62)**<br>torch-compile-cg 0.0184 (1.44) | torch-compile-ro 0.0174 (1.53) | torch-sdpa-compile 0.0205 (1.30)<br>torch-sdpa 0.0256 (1.04) | not run |  |
| lsm_4_32768 | a100 | 0.03402 |  | torch-eager 0.0635 (0.54) | torch-compile 0.0348 (0.98) | **torch-compile-cg 0.0225 (1.51)**<br>torch-sdpa-compile-cg 0.0266 (1.28) | torch-compile-ro 0.0266 (1.28) | torch-sdpa-compile 0.0348 (0.98)<br>torch-sdpa 0.0389 (0.87) | triton 0.0573 (0.59) |  |
| lsm_4_32768 | rtx-4090 | 0.02384 |  | torch-eager 0.0502 (0.47) | torch-compile 0.0256 (0.93) | **torch-compile-cg 0.0205 (1.16)**<br>torch-sdpa-compile-cg 0.0205 (1.16) | torch-compile-ro 0.0215 (1.11) | torch-sdpa-compile 0.0256 (0.93)<br>torch-sdpa 0.0328 (0.73) | not run |  |
| lsm_4_32768 | h100 | 0.02406 |  | torch-eager 0.0372 (0.65) | torch-compile 0.0243 (0.99) | **torch-sdpa-compile-cg 0.0154 (1.56)**<br>torch-compile-cg 0.0162 (1.48) | torch-compile-ro 0.0161 (1.49) | torch-sdpa 0.0203 (1.19)<br>torch-sdpa-compile 0.0212 (1.13) | not run |  |
| prefill_1024_512_256 | rtx-5080 | 6.767 |  | torch-eager 48.92 (0.14) | torch-compile 28.36 (0.24) | **torch-compile-cg 22.58 (0.30)** | torch-compile-ro 23.13 (0.29) |  |  |  |
| prefill_1024_512_256 | rtx-3090 | 13.09 |  | torch-eager 188.4 (0.07) | torch-compile 101.4 (0.13) | **torch-compile-cg 90.17 (0.15)** | torch-compile-ro 90.28 (0.14) |  |  |  |
| prefill_1024_512_256 | a100 | 24.54 |  | torch-eager 201 (0.12) | torch-compile 104.6 (0.23) | **torch-compile-cg 93.03 (0.26)** | torch-compile-ro 93.24 (0.26) |  |  |  |
| prefill_1024_512_256 | rtx-4090 | 17.04 |  | torch-eager 195.9 (0.09) | torch-compile 126 (0.14) | torch-compile-cg 121.6 (0.14) | **torch-compile-ro 121 (0.14)** |  |  |  |
| prefill_1024_512_256 | h100 | 10.9 |  | torch-eager 102.6 (0.11) | torch-compile 57.62 (0.19) | torch-compile-cg 59.71 (0.18) | **torch-compile-ro 56.43 (0.19)** |  |  |  |
| rowsum_2048 | rtx-5080 | 0.4533 | **chain-eager 0.4628 (0.98)**<br>chain-gemv 0.4649 (0.98) | torch-eager 0.4719 (0.96) | torch-compile 0.4855 (0.93) | torch-compile-cg 0.4883 (0.93) | torch-compile-ro 0.4892 (0.93) |  |  |  |
| rowsum_2048 | rtx-3090 | 0.8261 | chain-gemv 0.8899 (0.93)<br>chain-eager 0.894 (0.92) | torch-eager 0.7885 (1.05) | **torch-compile 0.7803 (1.06)** | torch-compile-cg 0.7803 (1.06) | torch-compile-ro 0.7844 (1.05) |  |  |  |
| rowsum_2048 | a100 | 0.9173 | chain-eager 1.268 (0.72)<br>chain-gemv 1.273 (0.72) | torch-eager 1.276 (0.72) | torch-compile 1.269 (0.72) | torch-compile-cg 1.268 (0.72) | **torch-compile-ro 1.248 (0.73)** |  |  |  |
| rowsum_2048 | rtx-4090 | 0.6164 | chain-eager 0.6515 (0.95)<br>chain-gemv 0.6554 (0.94) | torch-eager 0.6533 (0.94) | torch-compile 0.6532 (0.94) | torch-compile-cg 0.6543 (0.94) | **torch-compile-ro 0.6513 (0.95)** |  |  |  |
| rowsum_2048 | h100 | 0.5114 | chain-eager 0.4252 (1.20)<br>chain-gemv 0.4269 (1.20) | torch-eager 0.4214 (1.21) | torch-compile 0.4206 (1.22) | **torch-compile-cg 0.4195 (1.22)** | torch-compile-ro 0.42 (1.22) |  |  |  |
| rowsum_4096 | rtx-5080 | 3.744 | **chain-gemv 3.604 (1.04)**<br>chain-eager 3.616 (1.04) | torch-eager 3.628 (1.03) | torch-compile 3.632 (1.03) | torch-compile-cg 3.63 (1.03) | torch-compile-ro 3.645 (1.03) |  |  |  |
| rowsum_4096 | rtx-3090 | 6.181 | **chain-eager 6.568 (0.94)**<br>chain-gemv 6.569 (0.94) | torch-eager 6.572 (0.94) | torch-compile 6.63 (0.93) | torch-compile-cg 6.625 (0.93) | torch-compile-ro 6.627 (0.93) |  |  |  |
| rowsum_4096 | a100 | 6.746 | chain-eager 7.324 (0.92)<br>chain-gemv 7.326 (0.92) | **torch-eager 7.29 (0.93)** | torch-compile 7.608 (0.89) | torch-compile-cg 7.607 (0.89) | torch-compile-ro 7.316 (0.92) |  |  |  |
| rowsum_4096 | rtx-4090 | 4.754 | chain-gemv 4.889 (0.97)<br>chain-eager 4.903 (0.97) | torch-eager 4.903 (0.97) | **torch-compile 4.879 (0.97)** | torch-compile-cg 4.892 (0.97) | torch-compile-ro 4.879 (0.97) |  |  |  |
| rowsum_4096 | h100 | 4.532 | chain-eager 4.012 (1.13)<br>chain-gemv 4.019 (1.13) | torch-eager 4.03 (1.12) | torch-compile 4.009 (1.13) | torch-compile-cg 3.987 (1.14) | **torch-compile-ro 3.986 (1.14)** |  |  |  |
| softmax_rows_2048 | rtx-5080 | 0.4631 | chain-eager 0.5018 (0.92) | torch-eager 0.5276 (0.88) | torch-compile 0.4899 (0.95) | torch-compile-cg 0.4956 (0.93) | torch-compile-ro 0.4996 (0.93) |  | triton 0.6297 (0.74) | **chain-fused 0.4895 (0.95)** |
| softmax_rows_2048 | rtx-3090 | 0.8582 | chain-eager 0.895 (0.96) | torch-eager 0.9943 (0.86) | torch-compile 0.8038 (1.07) | **torch-compile-cg 0.7455 (1.15)** | torch-compile-ro 0.7465 (1.15) |  | triton 0.8868 (0.97) | chain-fused 0.8622 (1.00) |
| softmax_rows_2048 | a100 | 0.9513 | chain-eager 1.34 (0.71) | torch-eager 1.387 (0.69) | **torch-compile 1.002 (0.95)** | torch-compile-cg 1.278 (0.74) | torch-compile-ro 1.278 (0.74) |  | triton 1.018 (0.93) | chain-fused 1.329 (0.72) |
| softmax_rows_2048 | rtx-4090 | 0.636 | chain-eager 0.7107 (0.89) | torch-eager 0.7434 (0.86) | torch-compile 0.6564 (0.97) | **torch-compile-cg 0.6554 (0.97)** | torch-compile-ro 0.6584 (0.97) |  | triton 0.7731 (0.82) | chain-fused 0.7004 (0.91) |
| softmax_rows_2048 | h100 | 0.5364 | chain-eager 0.4629 (1.16) | torch-eager 0.4817 (1.11) | torch-compile 0.4575 (1.17) | **torch-compile-cg 0.4275 (1.25)** | torch-compile-ro 0.4281 (1.25) |  | triton 0.5183 (1.03) | chain-fused 0.4463 (1.20) |
| softmax_rows_4096 | rtx-5080 | 3.804 | chain-eager 3.909 (0.97) | torch-eager 4.203 (0.91) | torch-compile 3.689 (1.03) | torch-compile-cg 3.697 (1.03) | **torch-compile-ro 3.682 (1.03)** |  | triton 4.62 (0.82) | chain-fused 3.753 (1.01) |
| softmax_rows_4096 | rtx-3090 | 6.317 | chain-eager 7.01 (0.90) | torch-eager 7.347 (0.86) | **torch-compile 6.672 (0.95)** | torch-compile-cg 6.725 (0.94) | torch-compile-ro 6.718 (0.94) |  | triton 8.032 (0.79) | chain-fused 6.9 (0.92) |
| softmax_rows_4096 | a100 | 6.812 | chain-eager 7.609 (0.90) | torch-eager 7.809 (0.87) | torch-compile 7.645 (0.89) | torch-compile-cg 7.609 (0.90) | **torch-compile-ro 7.356 (0.93)** |  | triton 7.545 (0.90) | chain-fused 7.547 (0.90) |
| softmax_rows_4096 | rtx-4090 | 4.84 | chain-eager 5.235 (0.92) | torch-eager 5.514 (0.88) | torch-compile 4.958 (0.98) | torch-compile-cg 4.961 (0.98) | **torch-compile-ro 4.891 (0.99)** |  | triton 6.04 (0.80) | chain-fused 5.103 (0.95) |
| softmax_rows_4096 | h100 | 4.576 | chain-eager 4.2 (1.09) | torch-eager 4.339 (1.05) | torch-compile 4.048 (1.13) | torch-compile-cg 4.069 (1.12) | **torch-compile-ro 4.002 (1.14)** |  | triton 4.936 (0.93) | chain-fused 4.181 (1.09) |

### LLM decode / prefill / block (f32, tf32, bf16 storage)

| workload | part | Mapal ms | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | SDPA | FlashAttention-2/3 | FlashInfer | hand Triton | Liger |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| llm_blkdec_b1_l4096 | rtx-5080 | 1.034 | torch-eager 1.2 (0.86) | **torch-compile 1.087 (0.95)** | torch-compile-cg 1.1 (0.94) | torch-compile-ro 1.138 (0.91) | torch-sdpa 1.593 (0.65)<br>torch-sdpa-efficient 1.944 (0.53) |  |  |  | liger 1.918 (0.54) |
| llm_blkdec_b1_l4096 | rtx-3090 | 1.132 | torch-eager 1.591 (0.71) | **torch-compile 1.217 (0.93)** | torch-compile-cg 1.231 (0.92) | torch-compile-ro 1.374 (0.82) | torch-sdpa 1.999 (0.57)<br>torch-sdpa-efficient 2.715 (0.42) |  |  |  | liger 2.669 (0.42) |
| llm_blkdec_b1_l4096 | a100 | 1.041 | torch-eager 1.383 (0.75) | **torch-compile 0.9139 (1.14)** | torch-compile-cg 0.9231 (1.13) | torch-compile-ro 1.023 (1.02) | torch-sdpa 1.563 (0.67)<br>torch-sdpa-efficient 1.853 (0.56) |  |  |  | liger 1.809 (0.58) |
| llm_blkdec_b1_l4096 | rtx-4090 | 1.228 | torch-eager 1.736 (0.71) | **torch-compile 1.354 (0.91)** | torch-compile-cg 1.373 (0.89) | torch-compile-ro 1.476 (0.83) | torch-sdpa 2.05 (0.60)<br>torch-sdpa-efficient 3.253 (0.38) |  |  |  | liger 3.318 (0.37) |
| llm_blkdec_b1_l4096 | h100 | 0.5671 | torch-eager 0.8207 (0.69) | **torch-compile 0.5639 (1.01)** | torch-compile-cg 0.5826 (0.97) | torch-compile-ro 0.6684 (0.85) | torch-sdpa 0.894 (0.63)<br>torch-sdpa-efficient 1.24 (0.46) |  |  |  | liger 1.266 (0.45) |
| llm_blkpre_t2048 | rtx-5080 | 25.37 | torch-eager 33.21 (0.76) | torch-compile 28.22 (0.90) | torch-compile-cg 28.26 (0.90) | torch-compile-ro 29.25 (0.87) | torch-sdpa-efficient 27.23 (0.93)<br>torch-sdpa 33.56 (0.76) |  |  | triton 29.48 (0.86) | **liger 26.88 (0.94)** |
| llm_blkpre_t2048 | rtx-3090 | 45.78 | torch-eager 53.98 (0.85) | torch-compile 48.57 (0.94) | torch-compile-cg 48.68 (0.94) | torch-compile-ro 48.78 (0.94) | torch-sdpa-efficient 48.42 (0.95)<br>torch-sdpa 53.99 (0.85) |  |  | triton 52.46 (0.87) | **liger 47.5 (0.96)** |
| llm_blkpre_t2048 | a100 | 37.48 | torch-eager 57.31 (0.65) | torch-compile 53.82 (0.70) | torch-compile-cg 54.16 (0.69) | torch-compile-ro 54.4 (0.69) | torch-sdpa-efficient 51.69 (0.73)<br>torch-sdpa 57.55 (0.65) |  |  | triton 56.82 (0.66) | **liger 51.35 (0.73)** |
| llm_blkpre_t2048 | rtx-4090 | 33.23 | torch-eager 40.87 (0.81) | torch-compile 36.21 (0.92) | torch-compile-cg 36.25 (0.92) | torch-compile-ro 36.31 (0.91) | torch-sdpa-efficient 36.18 (0.92)<br>torch-sdpa 40.61 (0.82) |  |  | triton 39.89 (0.83) | **liger 35.76 (0.93)** |
| llm_blkpre_t2048 | h100 | 29.17 | torch-eager 34 (0.86) | torch-compile 29.6 (0.99) | torch-compile-cg 30.39 (0.96) | torch-compile-ro 30.42 (0.96) | torch-sdpa-efficient 29.22 (1.00)<br>torch-sdpa 33.28 (0.88) |  |  | triton 32.38 (0.90) | **liger 28.35 (1.03)** |
| llm_dec_b1_l2048 | rtx-5080 | 0.2195 | torch-eager 0.318 (0.69) | torch-compile 0.2636 (0.83) | torch-compile-cg 0.2754 (0.80) | **torch-compile-ro 0.2623 (0.84)** | torch-sdpa 0.5017 (0.44)<br>torch-sdpa-efficient 0.7011 (0.31) |  |  |  | liger 0.7063 (0.31) |
| llm_dec_b1_l2048 | rtx-3090 | 0.2702 | torch-eager 0.7218 (0.37) | **torch-compile 0.3626 (0.75)** | torch-compile-cg 0.3947 (0.68) | torch-compile-ro 0.3957 (0.68) | torch-sdpa 0.8335 (0.32)<br>torch-sdpa-efficient 1.139 (0.24) |  |  |  | liger 1.132 (0.24) |
| llm_dec_b1_l2048 | a100 | 0.2619 | torch-eager 0.6747 (0.39) | torch-compile 0.3552 (0.74) | torch-compile-cg 0.3281 (0.80) | **torch-compile-ro 0.324 (0.81)** | torch-sdpa 0.7528 (0.35)<br>torch-sdpa-efficient 0.8807 (0.30) |  |  |  | liger 0.855 (0.31) |
| llm_dec_b1_l2048 | rtx-4090 | 0.3029 | torch-eager 0.7307 (0.41) | **torch-compile 0.4224 (0.72)** | torch-compile-cg 0.449 (0.67) | torch-compile-ro 0.4933 (0.61) | torch-sdpa 0.7782 (0.39)<br>torch-sdpa-efficient 1.411 (0.21) |  |  |  | liger 1.486 (0.20) |
| llm_dec_b1_l2048 | h100 | 0.1104 | torch-eager 0.4185 (0.26) | torch-compile 0.2241 (0.49) | **torch-compile-cg 0.2109 (0.52)** | torch-compile-ro 0.2281 (0.48) | torch-sdpa 0.4526 (0.24)<br>torch-sdpa-efficient 0.5843 (0.19) |  |  |  | liger 0.6308 (0.17) |
| llm_dec_b1_l8192 | rtx-5080 | 0.2871 | torch-eager 0.4221 (0.68) | **torch-compile 0.353 (0.81)** | torch-compile-cg 0.3603 (0.80) | torch-compile-ro 0.3685 (0.78) | torch-sdpa 1.302 (0.22)<br>torch-sdpa-efficient 1.971 (0.15) |  |  |  | liger 1.968 (0.15) |
| llm_dec_b1_l8192 | rtx-3090 | 0.3502 | torch-eager 0.7779 (0.45) | **torch-compile 0.4653 (0.75)** | torch-compile-cg 0.5007 (0.70) | torch-compile-ro 0.5207 (0.67) | torch-sdpa 1.838 (0.19)<br>torch-sdpa-efficient 3.132 (0.11) |  |  |  | liger 3.191 (0.11) |
| llm_dec_b1_l8192 | a100 | 0.2971 | torch-eager 0.839 (0.35) | torch-compile 0.4856 (0.61) | **torch-compile-cg 0.4781 (0.62)** | torch-compile-ro 0.5292 (0.56) | torch-sdpa 1.277 (0.23)<br>torch-sdpa-efficient 1.9 (0.16) |  |  |  | liger 1.876 (0.16) |
| llm_dec_b1_l8192 | rtx-4090 | 0.385 | torch-eager 0.8404 (0.46) | **torch-compile 0.5601 (0.69)** | torch-compile-cg 0.5889 (0.65) | torch-compile-ro 0.6493 (0.59) | torch-sdpa 1.611 (0.24)<br>torch-sdpa-efficient 4.04 (0.10) |  |  |  | liger 4.125 (0.09) |
| llm_dec_b1_l8192 | h100 | 0.142 | torch-eager 0.6716 (0.21) | torch-compile 0.3583 (0.40) | **torch-compile-cg 0.332 (0.43)** | torch-compile-ro 0.5087 (0.28) | torch-sdpa 0.8188 (0.17)<br>torch-sdpa-efficient 1.398 (0.10) |  |  |  | liger 1.434 (0.10) |
| llm_dec_b32_l2048 | rtx-5080 | 0.8501 | torch-eager 2.649 (0.32) | **torch-compile 2.291 (0.37)** | torch-compile-cg 2.317 (0.37) | torch-compile-ro 2.393 (0.36) | torch-sdpa-efficient 8.086 (0.11)<br>torch-sdpa 8.889 (0.10) |  |  |  | liger 8.068 (0.11) |
| llm_dec_b32_l2048 | rtx-3090 | 0.9856 | torch-eager 3.093 (0.32) | torch-compile 2.876 (0.34) | torch-compile-cg 2.881 (0.34) | **torch-compile-ro 2.858 (0.34)** | torch-sdpa 8.901 (0.11)<br>torch-sdpa-efficient 11.45 (0.09) |  |  |  | liger 17.01 (0.06) |
| llm_dec_b32_l2048 | a100 | 0.7602 | torch-eager 2.207 (0.34) | **torch-compile 1.917 (0.40)** | torch-compile-cg 1.928 (0.39) | torch-compile-ro 1.942 (0.39) | torch-sdpa-efficient 5.807 (0.13)<br>torch-sdpa 6.316 (0.12) |  |  |  | liger 5.812 (0.13) |
| llm_dec_b32_l2048 | rtx-4090 | 1.062 | torch-eager 3.284 (0.32) | **torch-compile 3.102 (0.34)** | torch-compile-cg 3.137 (0.34) | torch-compile-ro 3.134 (0.34) | torch-sdpa 9.359 (0.11)<br>torch-sdpa-efficient 10.49 (0.10) |  |  |  | liger 10.29 (0.10) |
| llm_dec_b32_l2048 | h100 | 0.481 | torch-eager 1.91 (0.25) | **torch-compile 1.415 (0.34)** | torch-compile-cg 1.427 (0.34) | torch-compile-ro 1.567 (0.31) | torch-sdpa 5.291 (0.09)<br>torch-sdpa-efficient 5.42 (0.09) |  |  |  | liger 5.434 (0.09) |
| llm_dec_b32_l8192 | rtx-5080 | 2.761 | torch-eager 10.77 (0.26) | **torch-compile 9.531 (0.29)** | torch-compile-cg 9.778 (0.28) | not run | not run |  |  |  | not run |
| llm_dec_b32_l8192 | rtx-3090 | 3.145 | torch-eager 11.9 (0.26) | **torch-compile 10.9 (0.29)** | torch-compile-cg 10.9 (0.29) | torch-compile-ro 11.34 (0.28) | torch-sdpa 41.29 (0.08)<br>torch-sdpa-efficient 50.66 (0.06) |  |  |  | liger 87.87 (0.04) |
| llm_dec_b32_l8192 | a100 | 1.995 | torch-eager 7.128 (0.28) | torch-compile 6.662 (0.30) | **torch-compile-cg 6.628 (0.30)** | torch-compile-ro 6.631 (0.30) | torch-sdpa-efficient 21.6 (0.09)<br>torch-sdpa 23.86 (0.08) |  |  |  | liger 21.61 (0.09) |
| llm_dec_b32_l8192 | rtx-4090 | 3.21 | torch-eager 12.13 (0.26) | **torch-compile 11.18 (0.29)** | torch-compile-cg 11.31 (0.28) | torch-compile-ro 11.25 (0.29) | torch-sdpa 35.57 (0.09)<br>torch-sdpa-efficient 38.85 (0.08) |  |  |  | liger 38.76 (0.08) |
| llm_dec_b32_l8192 | h100 | 1.361 | torch-eager 6.535 (0.21) | **torch-compile 5.399 (0.25)** | torch-compile-cg 5.572 (0.24) | torch-compile-ro 5.696 (0.24) | torch-sdpa 20.02 (0.07)<br>torch-sdpa-efficient 20.99 (0.06) |  |  |  | liger 20.57 (0.07) |
| llm_dec_b8_l2048 | rtx-5080 | 0.3726 | torch-eager 0.8293 (0.45) | **torch-compile 0.6869 (0.54)** | torch-compile-cg 0.7066 (0.53) | torch-compile-ro 0.7595 (0.49) | torch-sdpa 2.431 (0.15)<br>torch-sdpa-efficient 2.471 (0.15) |  |  |  | liger 2.48 (0.15) |
| llm_dec_b8_l2048 | rtx-3090 | 0.8314 | torch-eager 1.178 (0.71) | **torch-compile 0.9304 (0.89)** | torch-compile-cg 0.9501 (0.87) | torch-compile-ro 0.9963 (0.83) | torch-sdpa 2.752 (0.30)<br>torch-sdpa-efficient 3.609 (0.23) |  |  |  | liger 6.705 (0.12) |
| llm_dec_b8_l2048 | a100 | 0.4325 | torch-eager 1.002 (0.43) | **torch-compile 0.6779 (0.64)** | torch-compile-cg 0.6962 (0.62) | torch-compile-ro 0.7619 (0.57) | torch-sdpa-efficient 2.007 (0.22)<br>torch-sdpa 2.031 (0.21) |  |  |  | liger 2.017 (0.21) |
| llm_dec_b8_l2048 | rtx-4090 | 0.4735 | torch-eager 1.229 (0.39) | **torch-compile 1.048 (0.45)** | torch-compile-cg 1.078 (0.44) | torch-compile-ro 1.074 (0.44) | torch-sdpa 2.81 (0.17)<br>torch-sdpa-efficient 2.987 (0.16) |  |  |  | liger 3.072 (0.15) |
| llm_dec_b8_l2048 | h100 | 0.1894 | torch-eager 0.7059 (0.27) | **torch-compile 0.5135 (0.37)** | torch-compile-cg 0.5452 (0.35) | torch-compile-ro 0.5972 (0.32) | torch-sdpa 1.48 (0.13)<br>torch-sdpa-efficient 1.589 (0.12) |  |  |  | liger 1.613 (0.12) |
| llm_dec_b8_l8192 | rtx-5080 | 0.8465 | torch-eager 2.853 (0.30) | **torch-compile 2.481 (0.34)** | torch-compile-cg 2.509 (0.34) | torch-compile-ro 2.585 (0.33) | torch-sdpa 8.856 (0.10)<br>torch-sdpa-efficient 8.865 (0.10) |  |  |  | liger 8.744 (0.10) |
| llm_dec_b8_l8192 | rtx-3090 | 1.382 | torch-eager 3.394 (0.41) | **torch-compile 2.943 (0.47)** | torch-compile-cg 2.957 (0.47) | torch-compile-ro 3.05 (0.45) | torch-sdpa 10.75 (0.13)<br>torch-sdpa-efficient 14.31 (0.10) |  |  |  | liger 19.25 (0.07) |
| llm_dec_b8_l8192 | a100 | 0.7525 | torch-eager 2.253 (0.33) | **torch-compile 1.863 (0.40)** | torch-compile-cg 1.883 (0.40) | torch-compile-ro 1.961 (0.38) | torch-sdpa-efficient 6.354 (0.12)<br>torch-sdpa 6.369 (0.12) |  |  |  | liger 6.36 (0.12) |
| llm_dec_b8_l8192 | rtx-4090 | 1 | torch-eager 3.497 (0.29) | **torch-compile 3.105 (0.32)** | torch-compile-cg 3.132 (0.32) | torch-compile-ro 3.166 (0.32) | torch-sdpa 9.429 (0.11)<br>torch-sdpa-efficient 10.2 (0.10) |  |  |  | liger 10.19 (0.10) |
| llm_dec_b8_l8192 | h100 | 0.4194 | torch-eager 1.712 (0.24) | **torch-compile 1.452 (0.29)** | torch-compile-cg 1.46 (0.29) | torch-compile-ro 1.489 (0.28) | torch-sdpa 5.236 (0.08)<br>torch-sdpa-efficient 5.503 (0.08) |  |  |  | liger 5.356 (0.08) |
| llm_pre_t2048 | rtx-5080 | 6.492 | torch-eager 13.49 (0.48) | torch-compile 8.495 (0.76) | torch-compile-cg 8.506 (0.76) | torch-compile-ro 9.538 (0.68) | torch-sdpa-efficient 7.299 (0.89)<br>torch-sdpa 13.86 (0.47) |  |  | triton 9.463 (0.69) | **liger 7.279 (0.89)** |
| llm_pre_t2048 | rtx-3090 | 10.8 | torch-eager 18.93 (0.57) | torch-compile 14.38 (0.75) | torch-compile-cg 14.37 (0.75) | torch-compile-ro 14.4 (0.75) | **torch-sdpa-efficient 12.95 (0.83)**<br>torch-sdpa 18.99 (0.57) |  |  | triton 34.68 (0.31) | liger 22.75 (0.47) |
| llm_pre_t2048 | a100 | 12.91 | torch-eager 17.25 (0.75) | torch-compile 14.37 (0.90) | torch-compile-cg 14.35 (0.90) | torch-compile-ro 14.29 (0.90) | torch-sdpa-efficient 11.56 (1.12)<br>torch-sdpa 17.5 (0.74) |  |  | triton 16.89 (0.76) | **liger 11.43 (1.13)** |
| llm_pre_t2048 | rtx-4090 | 9.142 | torch-eager 15.21 (0.60) | torch-compile 10.59 (0.86) | torch-compile-cg 10.6 (0.86) | torch-compile-ro 10.64 (0.86) | torch-sdpa-efficient 10.47 (0.87)<br>torch-sdpa 14.97 (0.61) |  |  | triton 14.16 (0.65) | **liger 10.35 (0.88)** |
| llm_pre_t2048 | h100 | 7.023 | torch-eager 10.31 (0.68) | torch-compile 8.488 (0.83) | torch-compile-cg 8.537 (0.82) | torch-compile-ro 8.704 (0.81) | torch-sdpa-efficient 8.454 (0.83)<br>torch-sdpa 10.5 (0.67) |  |  | triton 11.03 (0.64) | **liger 6.568 (1.07)** |
| llm_pre_t8192 | rtx-5080 | 44.66 | torch-eager 166.4 (0.27) | torch-compile 84.97 (0.53) | torch-compile-cg 84.84 (0.53) | torch-compile-ro 101.2 (0.44) | torch-sdpa-efficient 53.61 (0.83) |  |  | triton 84.06 (0.53) | **liger 52.96 (0.84)** |
| llm_pre_t8192 | rtx-3090 | 108 | torch-eager 193.5 (0.56) | torch-compile 133.5 (0.81) | torch-compile-cg 133.2 (0.81) | torch-compile-ro 130.6 (0.83) | **torch-sdpa-efficient 94.44 (1.14)**<br>torch-sdpa 196.3 (0.55) |  |  | triton 273.8 (0.39) | liger 133.1 (0.81) |
| llm_pre_t8192 | a100 | 93.69 | torch-eager 162.2 (0.58) | torch-compile 113.9 (0.82) | torch-compile-cg 113.8 (0.82) | torch-compile-ro 117.3 (0.80) | torch-sdpa-efficient 59.52 (1.57)<br>torch-sdpa 158.2 (0.59) |  |  | triton 120.5 (0.78) | **liger 58.94 (1.59)** |
| llm_pre_t8192 | rtx-4090 | 81.27 | torch-eager 162 (0.50) | torch-compile 92.77 (0.88) | torch-compile-cg 92.77 (0.88) | torch-compile-ro 92.74 (0.88) | torch-sdpa-efficient 79.07 (1.03)<br>torch-sdpa 165 (0.49) |  |  | triton 130.5 (0.62) | **liger 78.31 (1.04)** |
| llm_pre_t8192 | h100 | 50.43 | torch-eager 103.8 (0.49) | torch-compile 70.07 (0.72) | torch-compile-cg 70.68 (0.71) | torch-compile-ro 70.59 (0.71) | torch-sdpa-efficient 41.36 (1.22)<br>torch-sdpa 106.9 (0.47) |  |  | triton 109.4 (0.46) | **liger 39.47 (1.28)** |
| llm_blkpre_t2048_tf32 | rtx-5080 | 19.71 | torch-eager-tf32 24.25 (0.81) | torch-compile-tf32 18.7 (1.05) | torch-compile-cg-tf32 18.71 (1.05) | torch-compile-ro-tf32 19.65 (1.00) | torch-sdpa-tf32 23.95 (0.82) |  |  | **triton-tf32 18.08 (1.09)** | liger-tf32 19.21 (1.03) |
| llm_blkpre_t2048_tf32 | rtx-3090 | 33.21 | torch-eager-tf32 37.74 (0.88) | torch-compile-tf32 39.67 (0.84) | torch-compile-cg-tf32 39.25 (0.85) | torch-compile-ro-tf32 32.6 (1.02) | torch-sdpa-tf32 38.06 (0.87) |  |  | **triton-tf32 31.44 (1.06)** | liger-tf32 32 (1.04) |
| llm_blkpre_t2048_tf32 | a100 | 17.92 | torch-eager-tf32 12.54 (1.43) | torch-compile-tf32 9.548 (1.88) | torch-compile-cg-tf32 9.453 (1.90) | torch-compile-ro-tf32 9.597 (1.87) | torch-sdpa-tf32 12.81 (1.40) |  |  | **triton-tf32 8.989 (1.99)** | liger-tf32 9.463 (1.89) |
| llm_blkpre_t2048_tf32 | rtx-4090 | 29.03 | torch-eager-tf32 32.45 (0.89) | torch-compile-tf32 27.16 (1.07) | torch-compile-cg-tf32 26.95 (1.08) | torch-compile-ro-tf32 27.53 (1.05) | torch-sdpa-tf32 32.14 (0.90) |  |  | **triton-tf32 26.18 (1.11)** | liger-tf32 28.03 (1.04) |
| llm_blkpre_t2048_tf32 | h100 | 8.771 | torch-eager-tf32 9.291 (0.94) | torch-compile-tf32 7.309 (1.20) | torch-compile-cg-tf32 7.261 (1.21) | torch-compile-ro-tf32 7.347 (1.19) | torch-sdpa-tf32 9.182 (0.96) |  |  | triton-tf32 7.047 (1.24) | **liger-tf32 6.586 (1.33)** |
| llm_pre_t2048_tf32 | rtx-5080 | 5.34 | torch-eager-tf32 10.76 (0.50) | torch-compile-tf32 6.171 (0.87) | torch-compile-cg-tf32 6.124 (0.87) | torch-compile-ro-tf32 6.415 (0.83) | torch-sdpa-tf32 10.52 (0.51) |  |  | **triton-tf32 4.55 (1.17)** | liger-tf32 5.995 (0.89) |
| llm_pre_t2048_tf32 | rtx-3090 | 9.169 | torch-eager-tf32 14.96 (0.61) | torch-compile-tf32 10.47 (0.88) | torch-compile-cg-tf32 10.45 (0.88) | **torch-compile-ro-tf32 10.28 (0.89)** | torch-sdpa-tf32 15.1 (0.61) |  |  | triton-tf32 11.06 (0.83) | liger-tf32 13.45 (0.68) |
| llm_pre_t2048_tf32 | a100 | 7.381 | torch-eager-tf32 6.366 (1.16) | torch-compile-tf32 3.416 (2.16) | torch-compile-cg-tf32 3.389 (2.18) | torch-compile-ro-tf32 3.431 (2.15) | torch-sdpa-tf32 6.653 (1.11) |  |  | **triton-tf32 2.637 (2.80)** | liger-tf32 3.341 (2.21) |
| llm_pre_t2048_tf32 | rtx-4090 | 8.332 | torch-eager-tf32 12.93 (0.64) | torch-compile-tf32 8.056 (1.03) | torch-compile-cg-tf32 8.046 (1.04) | torch-compile-ro-tf32 8.362 (1.00) | torch-sdpa-tf32 12.69 (0.66) |  |  | **triton-tf32 6.619 (1.26)** | liger-tf32 8.8 (0.95) |
| llm_pre_t2048_tf32 | h100 | 2.903 | torch-eager-tf32 4.585 (0.63) | torch-compile-tf32 2.674 (1.09) | torch-compile-cg-tf32 2.844 (1.02) | torch-compile-ro-tf32 2.675 (1.09) | torch-sdpa-tf32 4.507 (0.64) |  |  | **triton-tf32 2.162 (1.34)** | liger-tf32 2.868 (1.01) |
| llm_blkdec_b1_l4096_bf16w | rtx-5080 | 0.6066 | torch-bf16 0.6762 (0.90) | **torch-bf16-compile 0.5613 (1.08)** | torch-bf16-compile-cg 0.5747 (1.06) | torch-bf16-compile-ro 1.006 (0.60) | torch-bf16-cudnn 0.7528 (0.81)<br>torch-bf16-efficient 1.059 (0.57) | fa2-bf16 0.6538 (0.93) | flashinfer-bf16 0.8008 (0.76) |  | liger-bf16 0.6801 (0.89) |
| llm_blkdec_b1_l4096_bf16w | rtx-3090 | 0.7059 | torch-bf16 1.085 (0.65) | torch-bf16-compile 0.7057 (1.00) | **torch-bf16-compile-cg 0.6992 (1.01)** | torch-bf16-compile-ro 1.235 (0.57) | torch-bf16-cudnn 1.188 (0.59)<br>torch-bf16-efficient 1.362 (0.52) | fa2-bf16 1.114 (0.63) | flashinfer-bf16 1.267 (0.56) |  | liger-bf16 1.087 (0.65) |
| llm_blkdec_b1_l4096_bf16w | a100 | 0.8763 | torch-bf16 0.9298 (0.94) | **torch-bf16-compile 0.5455 (1.61)** | torch-bf16-compile-cg 0.5505 (1.59) | torch-bf16-compile-ro 0.8554 (1.02) | torch-bf16-cudnn 1.04 (0.84)<br>torch-bf16-efficient 1.162 (0.75) | fa2-bf16 0.9093 (0.96) | flashinfer-bf16 1.079 (0.81) |  | liger-bf16 0.9048 (0.97) |
| llm_blkdec_b1_l4096_bf16w | rtx-4090 | 0.8057 | torch-bf16 1.203 (0.67) | **torch-bf16-compile 0.789 (1.02)** | torch-bf16-compile-cg 0.8153 (0.99) | torch-bf16-compile-ro 1.358 (0.59) | torch-bf16-cudnn 1.253 (0.64)<br>torch-bf16-efficient 1.62 (0.50) | fa2-bf16 1.194 (0.68) | flashinfer-bf16 1.444 (0.56) |  | liger-bf16 1.284 (0.63) |
| llm_blkdec_b1_l4096_bf16w | h100 | 0.4981 | torch-bf16 0.5641 (0.88) | torch-bf16-compile 0.3796 (1.31) | torch-bf16-compile-cg 0.3309 (1.50) | **torch-bf16-compile-ro 0.3052 (1.63)** | torch-bf16-cudnn 0.6138 (0.81)<br>torch-bf16-efficient 0.7411 (0.67) | fa2-bf16 0.5355 (0.93)<br>fa3-bf16 0.5597 (0.89) | flashinfer-bf16 0.6816 (0.73) |  | liger-bf16 0.6267 (0.79) |
| llm_blkpre_t2048_bf16w | rtx-5080 | 25.21 | torch-bf16 8.89 (2.84) | torch-bf16-compile 8.232 (3.06) | **torch-bf16-compile-cg 8.23 (3.06)** | torch-bf16-compile-ro 8.493 (2.97) | torch-bf16-cudnn 8.86 (2.85)<br>torch-bf16-efficient 9.409 (2.68) | fa2-bf16 8.878 (2.84) | flashinfer-bf16 8.885 (2.84) | triton-bf16 8.931 (2.82) | liger-bf16 8.585 (2.94) |
| llm_blkpre_t2048_bf16w | rtx-3090 | 45.11 | torch-bf16 15.9 (2.84) | **torch-bf16-compile 14.95 (3.02)** | torch-bf16-compile-cg 14.97 (3.01) | torch-bf16-compile-ro 14.98 (3.01) | torch-bf16-cudnn 15.84 (2.85)<br>torch-bf16-efficient 16.34 (2.76) | fa2-bf16 15.63 (2.89) | flashinfer-bf16 15.62 (2.89) | triton-bf16 15.7 (2.87) | liger-bf16 15.13 (2.98) |
| llm_blkpre_t2048_bf16w | a100 | 23.67 | torch-bf16 4.615 (5.13) | torch-bf16-compile 4.966 (4.77) | torch-bf16-compile-cg 4.035 (5.87) | **torch-bf16-compile-ro 4.01 (5.90)** | torch-bf16-cudnn 4.58 (5.17)<br>torch-bf16-efficient 4.8 (4.93) | fa2-bf16 4.61 (5.13) | flashinfer-bf16 5.72 (4.14) | triton-bf16 5.903 (4.01) | liger-bf16 4.175 (5.67) |
| llm_blkpre_t2048_bf16w | rtx-4090 | 34.79 | torch-bf16 13.13 (2.65) | **torch-bf16-compile 12.29 (2.83)** | torch-bf16-compile-cg 12.38 (2.81) | torch-bf16-compile-ro 12.69 (2.74) | torch-bf16-cudnn 13.2 (2.64)<br>torch-bf16-efficient 13.47 (2.58) | fa2-bf16 13.13 (2.65) | flashinfer-bf16 13.25 (2.63) | triton-bf16 13.27 (2.62) | liger-bf16 12.72 (2.74) |
| llm_blkpre_t2048_bf16w | h100 | 18.35 | torch-bf16 3.38 (5.43) | torch-bf16-compile 2.93 (6.26) | torch-bf16-compile-cg 2.824 (6.50) | **torch-bf16-compile-ro 2.765 (6.64)** | torch-bf16-cudnn 3.44 (5.33)<br>torch-bf16-efficient 3.689 (4.97) | fa3-bf16 3.398 (5.40)<br>fa2-bf16 3.597 (5.10) | flashinfer-bf16 3.313 (5.54) | triton-bf16 3.443 (5.33) | liger-bf16 3.317 (5.53) |
| llm_dec_b1_l2048_bf16w | rtx-5080 | 0.1337 | torch-bf16 0.2215 (0.60) | **torch-bf16-compile 0.1526 (0.88)** | torch-bf16-compile-cg 0.1574 (0.85) | torch-bf16-compile-ro 0.2995 (0.45) | torch-bf16-cudnn 0.2837 (0.47)<br>torch-bf16-efficient 0.4118 (0.32) | fa2-bf16 0.201 (0.66) | flashinfer-bf16 0.3348 (0.40) |  | liger-bf16 0.2456 (0.54) |
| llm_dec_b1_l2048_bf16w | rtx-3090 | 0.1828 | torch-bf16 0.5483 (0.33) | **torch-bf16-compile 0.2433 (0.75)** | torch-bf16-compile-cg 0.2881 (0.63) | torch-bf16-compile-ro 0.5107 (0.36) | torch-bf16-cudnn 0.6015 (0.30)<br>torch-bf16-efficient 0.7059 (0.26) | fa2-bf16 0.5349 (0.34) | flashinfer-bf16 0.719 (0.25) |  | liger-bf16 0.5066 (0.36) |
| llm_dec_b1_l2048_bf16w | a100 | 0.1853 | torch-bf16 0.5629 (0.33) | torch-bf16-compile 0.2649 (0.70) | **torch-bf16-compile-cg 0.2615 (0.71)** | torch-bf16-compile-ro 0.3999 (0.46) | torch-bf16-cudnn 0.6091 (0.30)<br>torch-bf16-efficient 0.6947 (0.27) | fa2-bf16 0.5228 (0.35) | flashinfer-bf16 0.6977 (0.27) |  | liger-bf16 0.5046 (0.37) |
| llm_dec_b1_l2048_bf16w | rtx-4090 | 0.2146 | torch-bf16 0.5695 (0.38) | **torch-bf16-compile 0.3054 (0.70)** | torch-bf16-compile-cg 0.321 (0.67) | torch-bf16-compile-ro 0.5665 (0.38) | torch-bf16-cudnn 0.6253 (0.34)<br>torch-bf16-efficient 0.7873 (0.27) | fa2-bf16 0.5636 (0.38) | flashinfer-bf16 0.7946 (0.27) |  | liger-bf16 0.6035 (0.36) |
| llm_dec_b1_l2048_bf16w | h100 | 0.08976 | torch-bf16 0.37 (0.24) | torch-bf16-compile 0.2273 (0.39) | torch-bf16-compile-cg 0.1532 (0.59) | **torch-bf16-compile-ro 0.153 (0.59)** | torch-bf16-cudnn 0.4014 (0.22)<br>torch-bf16-efficient 0.4483 (0.20) | fa2-bf16 0.3324 (0.27)<br>fa3-bf16 0.3578 (0.25) | flashinfer-bf16 0.4817 (0.19) |  | liger-bf16 0.3954 (0.23) |
| llm_dec_b1_l8192_bf16w | rtx-5080 | 0.1952 | torch-bf16 0.2541 (0.77) | **torch-bf16-compile 0.1889 (1.03)** | torch-bf16-compile-cg 0.2049 (0.95) | torch-bf16-compile-ro 1.038 (0.19) | torch-bf16-cudnn 0.2936 (0.67)<br>torch-bf16-efficient 0.9945 (0.20) | fa2-bf16 0.2236 (0.87) | flashinfer-bf16 0.402 (0.49) |  | liger-bf16 0.2639 (0.74) |
| llm_dec_b1_l8192_bf16w | rtx-3090 | 0.2525 | torch-bf16 0.5738 (0.44) | **torch-bf16-compile 0.3002 (0.84)** | torch-bf16-compile-cg 0.3252 (0.78) | torch-bf16-compile-ro 1.316 (0.19) | torch-bf16-cudnn 0.5996 (0.42)<br>torch-bf16-efficient 1.24 (0.20) | fa2-bf16 0.565 (0.45) | flashinfer-bf16 0.8266 (0.31) |  | liger-bf16 0.5243 (0.48) |
| llm_dec_b1_l8192_bf16w | a100 | 0.2471 | torch-bf16 0.5656 (0.44) | torch-bf16-compile 0.3134 (0.79) | **torch-bf16-compile-cg 0.2979 (0.83)** | torch-bf16-compile-ro 0.9151 (0.27) | torch-bf16-cudnn 0.6132 (0.40)<br>torch-bf16-efficient 1.195 (0.21) | fa2-bf16 0.5245 (0.47) | flashinfer-bf16 0.758 (0.33) |  | liger-bf16 0.5055 (0.49) |
| llm_dec_b1_l8192_bf16w | rtx-4090 | 0.2963 | torch-bf16 0.5887 (0.50) | **torch-bf16-compile 0.339 (0.87)** | torch-bf16-compile-cg 0.3787 (0.78) | torch-bf16-compile-ro 1.527 (0.19) | torch-bf16-cudnn 0.6343 (0.47)<br>torch-bf16-efficient 1.533 (0.19) | fa2-bf16 0.5797 (0.51) | flashinfer-bf16 0.9508 (0.31) |  | liger-bf16 0.6154 (0.48) |
| llm_dec_b1_l8192_bf16w | h100 | 0.117 | torch-bf16 0.3597 (0.33) | torch-bf16-compile 0.2345 (0.50) | torch-bf16-compile-cg 0.1636 (0.72) | **torch-bf16-compile-ro 0.1621 (0.72)** | torch-bf16-cudnn 0.4087 (0.29)<br>torch-bf16-efficient 0.8163 (0.14) | fa2-bf16 0.336 (0.35)<br>fa3-bf16 0.36 (0.32) | flashinfer-bf16 0.5258 (0.22) |  | liger-bf16 0.3968 (0.29) |
| llm_dec_b32_l2048_bf16w | rtx-5080 | 0.7774 | torch-bf16 0.655 (1.19) | **torch-bf16-compile 0.4738 (1.64)** | torch-bf16-compile-cg 0.4812 (1.62) | torch-bf16-compile-ro 7.691 (0.10) | torch-bf16-cudnn 0.5924 (1.31)<br>torch-bf16-efficient 3.903 (0.20) | fa2-bf16 0.5621 (1.38) | flashinfer-bf16 0.6529 (1.19) |  | liger-bf16 0.6831 (1.14) |
| llm_dec_b32_l2048_bf16w | rtx-3090 | 0.9613 | torch-bf16 1.141 (0.84) | torch-bf16-compile 0.8513 (1.13) | **torch-bf16-compile-cg 0.8439 (1.14)** | torch-bf16-compile-ro 8.801 (0.11) | torch-bf16-cudnn 0.8758 (1.10)<br>torch-bf16-efficient 5.091 (0.19) | fa2-bf16 0.9149 (1.05) | flashinfer-bf16 0.9993 (0.96) |  | liger-bf16 1.198 (0.80) |
| llm_dec_b32_l2048_bf16w | a100 | 0.7075 | torch-bf16 0.7667 (0.92) | **torch-bf16-compile 0.5621 (1.26)** | torch-bf16-compile-cg 0.6378 (1.11) | torch-bf16-compile-ro 5.493 (0.13) | torch-bf16-cudnn 0.7265 (0.97)<br>torch-bf16-efficient 3.678 (0.19) | fa2-bf16 0.718 (0.99) | flashinfer-bf16 0.8587 (0.82) |  | liger-bf16 0.7268 (0.97) |
| llm_dec_b32_l2048_bf16w | rtx-4090 | 0.9656 | torch-bf16 1.01 (0.96) | **torch-bf16-compile 0.6795 (1.42)** | torch-bf16-compile-cg 0.7068 (1.37) | torch-bf16-compile-ro 10.25 (0.09) | torch-bf16-cudnn 0.9247 (1.04)<br>torch-bf16-efficient 4.372 (0.22) | fa2-bf16 0.8941 (1.08) | flashinfer-bf16 1.082 (0.89) |  | liger-bf16 1.092 (0.88) |
| llm_dec_b32_l2048_bf16w | h100 | 0.4071 | torch-bf16 0.4293 (0.95) | torch-bf16-compile 0.2905 (1.40) | torch-bf16-compile-cg 0.2727 (1.49) | **torch-bf16-compile-ro 0.2717 (1.50)** | torch-bf16-cudnn 0.4491 (0.91)<br>torch-bf16-efficient 3.15 (0.13) | fa2-bf16 0.4388 (0.93)<br>fa3-bf16 0.485 (0.84) | flashinfer-bf16 0.5539 (0.73) |  | liger-bf16 0.4723 (0.86) |
| llm_dec_b32_l8192_bf16w | rtx-5080 | 2.704 | torch-bf16 1.89 (1.43) | **torch-bf16-compile 1.505 (1.80)** | torch-bf16-compile-cg 1.526 (1.77) | torch-bf16-compile-ro 29.89 (0.09) | torch-bf16-cudnn 1.524 (1.77)<br>torch-bf16-efficient 14.53 (0.19) | fa2-bf16 1.604 (1.69) | flashinfer-bf16 1.534 (1.76) |  | liger-bf16 1.917 (1.41) |
| llm_dec_b32_l8192_bf16w | rtx-3090 | 3.12 | torch-bf16 3.199 (0.98) | torch-bf16-compile 3.046 (1.02) | torch-bf16-compile-cg 3.122 (1.00) | torch-bf16-compile-ro 34.97 (0.09) | **torch-bf16-cudnn 1.794 (1.74)**<br>torch-bf16-efficient 19.69 (0.16) | fa2-bf16 2.198 (1.42) | flashinfer-bf16 1.916 (1.63) |  | liger-bf16 3.543 (0.88) |
| llm_dec_b32_l8192_bf16w | a100 | 1.955 | torch-bf16 1.462 (1.34) | torch-bf16-compile 1.789 (1.09) | torch-bf16-compile-cg 1.731 (1.13) | torch-bf16-compile-ro 21.45 (0.09) | **torch-bf16-cudnn 1.269 (1.54)**<br>torch-bf16-efficient 13.1 (0.15) | fa2-bf16 1.361 (1.44) | flashinfer-bf16 1.415 (1.38) |  | liger-bf16 1.434 (1.36) |
| llm_dec_b32_l8192_bf16w | rtx-4090 | 3.121 | torch-bf16 2.536 (1.23) | **torch-bf16-compile 1.885 (1.66)** | torch-bf16-compile-cg 1.936 (1.61) | torch-bf16-compile-ro 39.56 (0.08) | torch-bf16-cudnn 1.91 (1.63)<br>torch-bf16-efficient 15.76 (0.20) | fa2-bf16 1.886 (1.65) | flashinfer-bf16 2.068 (1.51) |  | liger-bf16 2.562 (1.22) |
| llm_dec_b32_l8192_bf16w | h100 | 1.282 | torch-bf16 0.7052 (1.82) | torch-bf16-compile 0.5939 (2.16) | torch-bf16-compile-cg 0.5967 (2.15) | **torch-bf16-compile-ro 0.58 (2.21)** | torch-bf16-cudnn 0.7385 (1.74)<br>torch-bf16-efficient 12.07 (0.11) | fa2-bf16 0.8295 (1.55)<br>fa3-bf16 0.9242 (1.39) | flashinfer-bf16 0.8394 (1.53) |  | liger-bf16 0.7551 (1.70) |
| llm_dec_b8_l2048_bf16w | rtx-5080 | 0.2796 | torch-bf16 0.2936 (0.95) | **torch-bf16-compile 0.224 (1.25)** | torch-bf16-compile-cg 0.2351 (1.19) | torch-bf16-compile-ro 1.947 (0.14) | torch-bf16-cudnn 0.3533 (0.79)<br>torch-bf16-efficient 1.3 (0.21) | fa2-bf16 0.2853 (0.98) | flashinfer-bf16 0.4052 (0.69) |  | liger-bf16 0.3433 (0.81) |
| llm_dec_b8_l2048_bf16w | rtx-3090 | 0.5248 | torch-bf16 0.6315 (0.83) | **torch-bf16-compile 0.3452 (1.52)** | torch-bf16-compile-cg 0.3749 (1.40) | torch-bf16-compile-ro 2.378 (0.22) | torch-bf16-cudnn 0.6503 (0.81)<br>torch-bf16-efficient 1.668 (0.31) | fa2-bf16 0.6263 (0.84) | flashinfer-bf16 0.7707 (0.68) |  | liger-bf16 0.6009 (0.87) |
| llm_dec_b8_l2048_bf16w | a100 | 0.3729 | torch-bf16 0.5668 (0.66) | **torch-bf16-compile 0.3425 (1.09)** | torch-bf16-compile-cg 0.3515 (1.06) | torch-bf16-compile-ro 1.566 (0.24) | torch-bf16-cudnn 0.6399 (0.58)<br>torch-bf16-efficient 1.415 (0.26) | fa2-bf16 0.5598 (0.67) | flashinfer-bf16 0.7103 (0.52) |  | liger-bf16 0.55 (0.68) |
| llm_dec_b8_l2048_bf16w | rtx-4090 | 0.385 | torch-bf16 0.6427 (0.60) | **torch-bf16-compile 0.3858 (1.00)** | torch-bf16-compile-cg 0.4195 (0.92) | torch-bf16-compile-ro 2.813 (0.14) | torch-bf16-cudnn 0.6787 (0.57)<br>torch-bf16-efficient 1.54 (0.25) | fa2-bf16 0.6483 (0.59) | flashinfer-bf16 0.82 (0.47) |  | liger-bf16 0.6934 (0.56) |
| llm_dec_b8_l2048_bf16w | h100 | 0.1635 | torch-bf16 0.3678 (0.44) | torch-bf16-compile 0.258 (0.63) | **torch-bf16-compile-cg 0.1706 (0.96)** | torch-bf16-compile-ro 0.1706 (0.96) | torch-bf16-cudnn 0.4055 (0.40)<br>torch-bf16-efficient 0.9931 (0.16) | fa2-bf16 0.341 (0.48)<br>fa3-bf16 0.3677 (0.44) | flashinfer-bf16 0.4826 (0.34) |  | liger-bf16 0.4154 (0.39) |
| llm_dec_b8_l8192_bf16w | rtx-5080 | 0.7541 | torch-bf16 0.5451 (1.38) | **torch-bf16-compile 0.4932 (1.53)** | torch-bf16-compile-cg 0.515 (1.46) | torch-bf16-compile-ro 7.751 (0.10) | torch-bf16-cudnn 0.5893 (1.28)<br>torch-bf16-efficient 4.208 (0.18) | fa2-bf16 0.5144 (1.47) | flashinfer-bf16 0.6309 (1.20) |  | liger-bf16 0.5729 (1.32) |
| llm_dec_b8_l8192_bf16w | rtx-3090 | 1.076 | torch-bf16 0.8435 (1.28) | **torch-bf16-compile 0.8122 (1.32)** | torch-bf16-compile-cg 0.8232 (1.31) | torch-bf16-compile-ro 9.016 (0.12) | torch-bf16-cudnn 0.8607 (1.25)<br>torch-bf16-efficient 5.44 (0.20) | fa2-bf16 0.8595 (1.25) | flashinfer-bf16 0.9962 (1.08) |  | liger-bf16 0.8277 (1.30) |
| llm_dec_b8_l8192_bf16w | a100 | 0.7064 | torch-bf16 0.7354 (0.96) | torch-bf16-compile 0.6537 (1.08) | **torch-bf16-compile-cg 0.6208 (1.14)** | torch-bf16-compile-ro 5.541 (0.13) | torch-bf16-cudnn 0.7766 (0.91)<br>torch-bf16-efficient 3.897 (0.18) | fa2-bf16 0.6978 (1.01) | flashinfer-bf16 0.8489 (0.83) |  | liger-bf16 0.6883 (1.03) |
| llm_dec_b8_l8192_bf16w | rtx-4090 | 0.9158 | torch-bf16 0.8955 (1.02) | **torch-bf16-compile 0.6954 (1.32)** | torch-bf16-compile-cg 0.7276 (1.26) | torch-bf16-compile-ro 10.28 (0.09) | torch-bf16-cudnn 0.9191 (1.00)<br>torch-bf16-efficient 4.402 (0.21) | fa2-bf16 0.9012 (1.02) | flashinfer-bf16 1.063 (0.86) |  | liger-bf16 0.9457 (0.97) |
| llm_dec_b8_l8192_bf16w | h100 | 0.3919 | torch-bf16 0.4291 (0.91) | torch-bf16-compile 0.298 (1.31) | torch-bf16-compile-cg 0.2723 (1.44) | **torch-bf16-compile-ro 0.2707 (1.45)** | torch-bf16-cudnn 0.4538 (0.86)<br>torch-bf16-efficient 3.173 (0.12) | fa2-bf16 0.4436 (0.88)<br>fa3-bf16 0.4741 (0.83) | flashinfer-bf16 0.5649 (0.69) |  | liger-bf16 0.4727 (0.83) |
| llm_pre_t2048_bf16w | rtx-5080 | 6.367 | torch-bf16 2.225 (2.86) | torch-bf16-compile 1.972 (3.23) | **torch-bf16-compile-cg 1.971 (3.23)** | torch-bf16-compile-ro 2.006 (3.17) | torch-bf16-cudnn 2.184 (2.92)<br>torch-bf16-efficient 2.739 (2.33) | fa2-bf16 2.217 (2.87) | flashinfer-bf16 2.228 (2.86) | triton-bf16 2.293 (2.78) | liger-bf16 2.131 (2.99) |
| llm_pre_t2048_bf16w | rtx-3090 | 10.59 | torch-bf16 4.048 (2.62) | torch-bf16-compile 3.506 (3.02) | **torch-bf16-compile-cg 3.486 (3.04)** | torch-bf16-compile-ro 3.583 (2.96) | torch-bf16-cudnn 4.081 (2.59)<br>torch-bf16-efficient 4.398 (2.41) | fa2-bf16 6.575 (1.61) | flashinfer-bf16 6.694 (1.58) | triton-bf16 7.38 (1.43) | liger-bf16 6.399 (1.65) |
| llm_pre_t2048_bf16w | a100 | 8.321 | torch-bf16 1.435 (5.80) | torch-bf16-compile 1.041 (7.99) | **torch-bf16-compile-cg 1.033 (8.05)** | torch-bf16-compile-ro 1.065 (7.81) | torch-bf16-cudnn 1.422 (5.85)<br>torch-bf16-efficient 1.64 (5.07) | fa2-bf16 1.408 (5.91) | flashinfer-bf16 1.415 (5.88) | triton-bf16 1.558 (5.34) | liger-bf16 1.235 (6.74) |
| llm_pre_t2048_bf16w | rtx-4090 | 9.44 | torch-bf16 3.269 (2.89) | torch-bf16-compile 2.911 (3.24) | **torch-bf16-compile-cg 2.91 (3.24)** | torch-bf16-compile-ro 2.961 (3.19) | torch-bf16-cudnn 3.35 (2.82)<br>torch-bf16-efficient 3.623 (2.61) | fa2-bf16 3.271 (2.89) | flashinfer-bf16 3.403 (2.77) | triton-bf16 3.429 (2.75) | liger-bf16 3.105 (3.04) |
| llm_pre_t2048_bf16w | h100 | 4.738 | torch-bf16 0.9614 (4.93) | torch-bf16-compile 0.6815 (6.95) | torch-bf16-compile-cg 0.712 (6.65) | **torch-bf16-compile-ro 0.6509 (7.28)** | torch-bf16-cudnn 0.9414 (5.03)<br>torch-bf16-efficient 1.356 (3.50) | fa3-bf16 1.083 (4.37)<br>fa2-bf16 1.383 (3.42) | flashinfer-bf16 1.313 (3.61) | triton-bf16 1.092 (4.34) | liger-bf16 0.9394 (5.04) |
| llm_blkdec_b1_l4096_bf16kv | rtx-5080 | 0.5891 | torch-bf16 0.6762 (0.87) | **torch-bf16-compile 0.5613 (1.05)** | torch-bf16-compile-cg 0.5747 (1.02) | torch-bf16-compile-ro 1.006 (0.59) | torch-bf16-cudnn 0.7528 (0.78)<br>torch-bf16-efficient 1.059 (0.56) | fa2-bf16 0.6538 (0.90) | flashinfer-bf16 0.8008 (0.74) |  | liger-bf16 0.6801 (0.87) |
| llm_blkdec_b1_l4096_bf16kv | rtx-3090 | 0.6891 | torch-bf16 1.085 (0.64) | torch-bf16-compile 0.7057 (0.98) | **torch-bf16-compile-cg 0.6992 (0.99)** | torch-bf16-compile-ro 1.235 (0.56) | torch-bf16-cudnn 1.188 (0.58)<br>torch-bf16-efficient 1.362 (0.51) | fa2-bf16 1.114 (0.62) | flashinfer-bf16 1.267 (0.54) |  | liger-bf16 1.087 (0.63) |
| llm_blkdec_b1_l4096_bf16kv | a100 | 0.8675 | torch-bf16 0.9298 (0.93) | **torch-bf16-compile 0.5455 (1.59)** | torch-bf16-compile-cg 0.5505 (1.58) | torch-bf16-compile-ro 0.8554 (1.01) | torch-bf16-cudnn 1.04 (0.83)<br>torch-bf16-efficient 1.162 (0.75) | fa2-bf16 0.9093 (0.95) | flashinfer-bf16 1.079 (0.80) |  | liger-bf16 0.9048 (0.96) |
| llm_blkdec_b1_l4096_bf16kv | rtx-4090 | 0.7833 | torch-bf16 1.203 (0.65) | **torch-bf16-compile 0.789 (0.99)** | torch-bf16-compile-cg 0.8153 (0.96) | torch-bf16-compile-ro 1.358 (0.58) | torch-bf16-cudnn 1.253 (0.63)<br>torch-bf16-efficient 1.62 (0.48) | fa2-bf16 1.194 (0.66) | flashinfer-bf16 1.444 (0.54) |  | liger-bf16 1.284 (0.61) |
| llm_blkdec_b1_l4096_bf16kv | h100 | 0.4955 | torch-bf16 0.5641 (0.88) | torch-bf16-compile 0.3796 (1.31) | torch-bf16-compile-cg 0.3309 (1.50) | **torch-bf16-compile-ro 0.3052 (1.62)** | torch-bf16-cudnn 0.6138 (0.81)<br>torch-bf16-efficient 0.7411 (0.67) | fa2-bf16 0.5355 (0.93)<br>fa3-bf16 0.5597 (0.89) | flashinfer-bf16 0.6816 (0.73) |  | liger-bf16 0.6267 (0.79) |
| llm_dec_b1_l2048_bf16kv | rtx-5080 | 0.1256 | torch-bf16 0.2215 (0.57) | **torch-bf16-compile 0.1526 (0.82)** | torch-bf16-compile-cg 0.1574 (0.80) | torch-bf16-compile-ro 0.2995 (0.42) | torch-bf16-cudnn 0.2837 (0.44)<br>torch-bf16-efficient 0.4118 (0.31) | fa2-bf16 0.201 (0.62) | flashinfer-bf16 0.3348 (0.38) |  | liger-bf16 0.2456 (0.51) |
| llm_dec_b1_l2048_bf16kv | rtx-3090 | 0.1751 | torch-bf16 0.5483 (0.32) | **torch-bf16-compile 0.2433 (0.72)** | torch-bf16-compile-cg 0.2881 (0.61) | torch-bf16-compile-ro 0.5107 (0.34) | torch-bf16-cudnn 0.6015 (0.29)<br>torch-bf16-efficient 0.7059 (0.25) | fa2-bf16 0.5349 (0.33) | flashinfer-bf16 0.719 (0.24) |  | liger-bf16 0.5066 (0.35) |
| llm_dec_b1_l2048_bf16kv | a100 | 0.197 | torch-bf16 0.5629 (0.35) | torch-bf16-compile 0.2649 (0.74) | **torch-bf16-compile-cg 0.2615 (0.75)** | torch-bf16-compile-ro 0.3999 (0.49) | torch-bf16-cudnn 0.6091 (0.32)<br>torch-bf16-efficient 0.6947 (0.28) | fa2-bf16 0.5228 (0.38) | flashinfer-bf16 0.6977 (0.28) |  | liger-bf16 0.5046 (0.39) |
| llm_dec_b1_l2048_bf16kv | rtx-4090 | 0.206 | torch-bf16 0.5695 (0.36) | **torch-bf16-compile 0.3054 (0.67)** | torch-bf16-compile-cg 0.321 (0.64) | torch-bf16-compile-ro 0.5665 (0.36) | torch-bf16-cudnn 0.6253 (0.33)<br>torch-bf16-efficient 0.7873 (0.26) | fa2-bf16 0.5636 (0.37) | flashinfer-bf16 0.7946 (0.26) |  | liger-bf16 0.6035 (0.34) |
| llm_dec_b1_l2048_bf16kv | h100 | 0.08655 | torch-bf16 0.37 (0.23) | torch-bf16-compile 0.2273 (0.38) | torch-bf16-compile-cg 0.1532 (0.57) | **torch-bf16-compile-ro 0.153 (0.57)** | torch-bf16-cudnn 0.4014 (0.22)<br>torch-bf16-efficient 0.4483 (0.19) | fa2-bf16 0.3324 (0.26)<br>fa3-bf16 0.3578 (0.24) | flashinfer-bf16 0.4817 (0.18) |  | liger-bf16 0.3954 (0.22) |
| llm_dec_b1_l8192_bf16kv | rtx-5080 | 0.1656 | torch-bf16 0.2541 (0.65) | **torch-bf16-compile 0.1889 (0.88)** | torch-bf16-compile-cg 0.2049 (0.81) | torch-bf16-compile-ro 1.038 (0.16) | torch-bf16-cudnn 0.2936 (0.56)<br>torch-bf16-efficient 0.9945 (0.17) | fa2-bf16 0.2236 (0.74) | flashinfer-bf16 0.402 (0.41) |  | liger-bf16 0.2639 (0.63) |
| llm_dec_b1_l8192_bf16kv | rtx-3090 | 0.2248 | torch-bf16 0.5738 (0.39) | **torch-bf16-compile 0.3002 (0.75)** | torch-bf16-compile-cg 0.3252 (0.69) | torch-bf16-compile-ro 1.316 (0.17) | torch-bf16-cudnn 0.5996 (0.37)<br>torch-bf16-efficient 1.24 (0.18) | fa2-bf16 0.565 (0.40) | flashinfer-bf16 0.8266 (0.27) |  | liger-bf16 0.5243 (0.43) |
| llm_dec_b1_l8192_bf16kv | a100 | 0.2305 | torch-bf16 0.5656 (0.41) | torch-bf16-compile 0.3134 (0.74) | **torch-bf16-compile-cg 0.2979 (0.77)** | torch-bf16-compile-ro 0.9151 (0.25) | torch-bf16-cudnn 0.6132 (0.38)<br>torch-bf16-efficient 1.195 (0.19) | fa2-bf16 0.5245 (0.44) | flashinfer-bf16 0.758 (0.30) |  | liger-bf16 0.5055 (0.46) |
| llm_dec_b1_l8192_bf16kv | rtx-4090 | 0.2614 | torch-bf16 0.5887 (0.44) | **torch-bf16-compile 0.339 (0.77)** | torch-bf16-compile-cg 0.3787 (0.69) | torch-bf16-compile-ro 1.527 (0.17) | torch-bf16-cudnn 0.6343 (0.41)<br>torch-bf16-efficient 1.533 (0.17) | fa2-bf16 0.5797 (0.45) | flashinfer-bf16 0.9508 (0.27) |  | liger-bf16 0.6154 (0.42) |
| llm_dec_b1_l8192_bf16kv | h100 | 0.1131 | torch-bf16 0.3597 (0.31) | torch-bf16-compile 0.2345 (0.48) | torch-bf16-compile-cg 0.1636 (0.69) | **torch-bf16-compile-ro 0.1621 (0.70)** | torch-bf16-cudnn 0.4087 (0.28)<br>torch-bf16-efficient 0.8163 (0.14) | fa2-bf16 0.336 (0.34)<br>fa3-bf16 0.36 (0.31) | flashinfer-bf16 0.5258 (0.22) |  | liger-bf16 0.3968 (0.29) |
| llm_dec_b32_l2048_bf16kv | rtx-5080 | 0.4959 | torch-bf16 0.655 (0.76) | **torch-bf16-compile 0.4738 (1.05)** | torch-bf16-compile-cg 0.4812 (1.03) | torch-bf16-compile-ro 7.691 (0.06) | torch-bf16-cudnn 0.5924 (0.84)<br>torch-bf16-efficient 3.903 (0.13) | fa2-bf16 0.5621 (0.88) | flashinfer-bf16 0.6529 (0.76) |  | liger-bf16 0.6831 (0.73) |
| llm_dec_b32_l2048_bf16kv | rtx-3090 | 0.675 | torch-bf16 1.141 (0.59) | torch-bf16-compile 0.8513 (0.79) | **torch-bf16-compile-cg 0.8439 (0.80)** | torch-bf16-compile-ro 8.801 (0.08) | torch-bf16-cudnn 0.8758 (0.77)<br>torch-bf16-efficient 5.091 (0.13) | fa2-bf16 0.9149 (0.74) | flashinfer-bf16 0.9993 (0.68) |  | liger-bf16 1.198 (0.56) |
| llm_dec_b32_l2048_bf16kv | a100 | 0.614 | torch-bf16 0.7667 (0.80) | **torch-bf16-compile 0.5621 (1.09)** | torch-bf16-compile-cg 0.6378 (0.96) | torch-bf16-compile-ro 5.493 (0.11) | torch-bf16-cudnn 0.7265 (0.85)<br>torch-bf16-efficient 3.678 (0.17) | fa2-bf16 0.718 (0.86) | flashinfer-bf16 0.8587 (0.72) |  | liger-bf16 0.7268 (0.84) |
| llm_dec_b32_l2048_bf16kv | rtx-4090 | 0.6494 | torch-bf16 1.01 (0.64) | **torch-bf16-compile 0.6795 (0.96)** | torch-bf16-compile-cg 0.7068 (0.92) | torch-bf16-compile-ro 10.25 (0.06) | torch-bf16-cudnn 0.9247 (0.70)<br>torch-bf16-efficient 4.372 (0.15) | fa2-bf16 0.8941 (0.73) | flashinfer-bf16 1.082 (0.60) |  | liger-bf16 1.092 (0.59) |
| llm_dec_b32_l2048_bf16kv | h100 | 0.3233 | torch-bf16 0.4293 (0.75) | torch-bf16-compile 0.2905 (1.11) | torch-bf16-compile-cg 0.2727 (1.19) | **torch-bf16-compile-ro 0.2717 (1.19)** | torch-bf16-cudnn 0.4491 (0.72)<br>torch-bf16-efficient 3.15 (0.10) | fa2-bf16 0.4388 (0.74)<br>fa3-bf16 0.485 (0.67) | flashinfer-bf16 0.5539 (0.58) |  | liger-bf16 0.4723 (0.68) |
| llm_dec_b32_l8192_bf16kv | rtx-5080 | 1.657 | torch-bf16 1.89 (0.88) | **torch-bf16-compile 1.505 (1.10)** | torch-bf16-compile-cg 1.526 (1.09) | torch-bf16-compile-ro 29.89 (0.06) | torch-bf16-cudnn 1.524 (1.09)<br>torch-bf16-efficient 14.53 (0.11) | fa2-bf16 1.604 (1.03) | flashinfer-bf16 1.534 (1.08) |  | liger-bf16 1.917 (0.86) |
| llm_dec_b32_l8192_bf16kv | rtx-3090 | 1.975 | torch-bf16 3.199 (0.62) | torch-bf16-compile 3.046 (0.65) | torch-bf16-compile-cg 3.122 (0.63) | torch-bf16-compile-ro 34.97 (0.06) | **torch-bf16-cudnn 1.794 (1.10)**<br>torch-bf16-efficient 19.69 (0.10) | fa2-bf16 2.198 (0.90) | flashinfer-bf16 1.916 (1.03) |  | liger-bf16 3.543 (0.56) |
| llm_dec_b32_l8192_bf16kv | a100 | 1.592 | torch-bf16 1.462 (1.09) | torch-bf16-compile 1.789 (0.89) | torch-bf16-compile-cg 1.731 (0.92) | torch-bf16-compile-ro 21.45 (0.07) | **torch-bf16-cudnn 1.269 (1.25)**<br>torch-bf16-efficient 13.1 (0.12) | fa2-bf16 1.361 (1.17) | flashinfer-bf16 1.415 (1.13) |  | liger-bf16 1.434 (1.11) |
| llm_dec_b32_l8192_bf16kv | rtx-4090 | 1.916 | torch-bf16 2.536 (0.76) | **torch-bf16-compile 1.885 (1.02)** | torch-bf16-compile-cg 1.936 (0.99) | torch-bf16-compile-ro 39.56 (0.05) | torch-bf16-cudnn 1.91 (1.00)<br>torch-bf16-efficient 15.76 (0.12) | fa2-bf16 1.886 (1.02) | flashinfer-bf16 2.068 (0.93) |  | liger-bf16 2.562 (0.75) |
| llm_dec_b32_l8192_bf16kv | h100 | 0.9065 | torch-bf16 0.7052 (1.29) | torch-bf16-compile 0.5939 (1.53) | torch-bf16-compile-cg 0.5967 (1.52) | **torch-bf16-compile-ro 0.58 (1.56)** | torch-bf16-cudnn 0.7385 (1.23)<br>torch-bf16-efficient 12.07 (0.08) | fa2-bf16 0.8295 (1.09)<br>fa3-bf16 0.9242 (0.98) | flashinfer-bf16 0.8394 (1.08) |  | liger-bf16 0.7551 (1.20) |
| llm_dec_b8_l2048_bf16kv | rtx-5080 | 0.2077 | torch-bf16 0.2936 (0.71) | **torch-bf16-compile 0.224 (0.93)** | torch-bf16-compile-cg 0.2351 (0.88) | torch-bf16-compile-ro 1.947 (0.11) | torch-bf16-cudnn 0.3533 (0.59)<br>torch-bf16-efficient 1.3 (0.16) | fa2-bf16 0.2853 (0.73) | flashinfer-bf16 0.4052 (0.51) |  | liger-bf16 0.3433 (0.60) |
| llm_dec_b8_l2048_bf16kv | rtx-3090 | 0.4784 | torch-bf16 0.6315 (0.76) | **torch-bf16-compile 0.3452 (1.39)** | torch-bf16-compile-cg 0.3749 (1.28) | torch-bf16-compile-ro 2.378 (0.20) | torch-bf16-cudnn 0.6503 (0.74)<br>torch-bf16-efficient 1.668 (0.29) | fa2-bf16 0.6263 (0.76) | flashinfer-bf16 0.7707 (0.62) |  | liger-bf16 0.6009 (0.80) |
| llm_dec_b8_l2048_bf16kv | a100 | 0.3458 | torch-bf16 0.5668 (0.61) | **torch-bf16-compile 0.3425 (1.01)** | torch-bf16-compile-cg 0.3515 (0.98) | torch-bf16-compile-ro 1.566 (0.22) | torch-bf16-cudnn 0.6399 (0.54)<br>torch-bf16-efficient 1.415 (0.24) | fa2-bf16 0.5598 (0.62) | flashinfer-bf16 0.7103 (0.49) |  | liger-bf16 0.55 (0.63) |
| llm_dec_b8_l2048_bf16kv | rtx-4090 | 0.3088 | torch-bf16 0.6427 (0.48) | **torch-bf16-compile 0.3858 (0.80)** | torch-bf16-compile-cg 0.4195 (0.74) | torch-bf16-compile-ro 2.813 (0.11) | torch-bf16-cudnn 0.6787 (0.46)<br>torch-bf16-efficient 1.54 (0.20) | fa2-bf16 0.6483 (0.48) | flashinfer-bf16 0.82 (0.38) |  | liger-bf16 0.6934 (0.45) |
| llm_dec_b8_l2048_bf16kv | h100 | 0.1461 | torch-bf16 0.3678 (0.40) | torch-bf16-compile 0.258 (0.57) | **torch-bf16-compile-cg 0.1706 (0.86)** | torch-bf16-compile-ro 0.1706 (0.86) | torch-bf16-cudnn 0.4055 (0.36)<br>torch-bf16-efficient 0.9931 (0.15) | fa2-bf16 0.341 (0.43)<br>fa3-bf16 0.3677 (0.40) | flashinfer-bf16 0.4826 (0.30) |  | liger-bf16 0.4154 (0.35) |
| llm_dec_b8_l8192_bf16kv | rtx-5080 | 0.4949 | torch-bf16 0.5451 (0.91) | **torch-bf16-compile 0.4932 (1.00)** | torch-bf16-compile-cg 0.515 (0.96) | torch-bf16-compile-ro 7.751 (0.06) | torch-bf16-cudnn 0.5893 (0.84)<br>torch-bf16-efficient 4.208 (0.12) | fa2-bf16 0.5144 (0.96) | flashinfer-bf16 0.6309 (0.78) |  | liger-bf16 0.5729 (0.86) |
| llm_dec_b8_l8192_bf16kv | rtx-3090 | 0.8026 | torch-bf16 0.8435 (0.95) | **torch-bf16-compile 0.8122 (0.99)** | torch-bf16-compile-cg 0.8232 (0.98) | torch-bf16-compile-ro 9.016 (0.09) | torch-bf16-cudnn 0.8607 (0.93)<br>torch-bf16-efficient 5.44 (0.15) | fa2-bf16 0.8595 (0.93) | flashinfer-bf16 0.9962 (0.81) |  | liger-bf16 0.8277 (0.97) |
| llm_dec_b8_l8192_bf16kv | a100 | 0.6159 | torch-bf16 0.7354 (0.84) | torch-bf16-compile 0.6537 (0.94) | **torch-bf16-compile-cg 0.6208 (0.99)** | torch-bf16-compile-ro 5.541 (0.11) | torch-bf16-cudnn 0.7766 (0.79)<br>torch-bf16-efficient 3.897 (0.16) | fa2-bf16 0.6978 (0.88) | flashinfer-bf16 0.8489 (0.73) |  | liger-bf16 0.6883 (0.89) |
| llm_dec_b8_l8192_bf16kv | rtx-4090 | 0.6155 | torch-bf16 0.8955 (0.69) | **torch-bf16-compile 0.6954 (0.89)** | torch-bf16-compile-cg 0.7276 (0.85) | torch-bf16-compile-ro 10.28 (0.06) | torch-bf16-cudnn 0.9191 (0.67)<br>torch-bf16-efficient 4.402 (0.14) | fa2-bf16 0.9012 (0.68) | flashinfer-bf16 1.063 (0.58) |  | liger-bf16 0.9457 (0.65) |
| llm_dec_b8_l8192_bf16kv | h100 | 0.3083 | torch-bf16 0.4291 (0.72) | torch-bf16-compile 0.298 (1.03) | torch-bf16-compile-cg 0.2723 (1.13) | **torch-bf16-compile-ro 0.2707 (1.14)** | torch-bf16-cudnn 0.4538 (0.68)<br>torch-bf16-efficient 3.173 (0.10) | fa2-bf16 0.4436 (0.70)<br>fa3-bf16 0.4741 (0.65) | flashinfer-bf16 0.5649 (0.55) |  | liger-bf16 0.4727 (0.65) |
| llm_blkdec_b1_l4096_bf16 | rtx-5080 | 0.5464 | torch-bf16 0.6762 (0.81) | **torch-bf16-compile 0.5613 (0.97)** | torch-bf16-compile-cg 0.5747 (0.95) | torch-bf16-compile-ro 1.006 (0.54) | torch-bf16-cudnn 0.7528 (0.73)<br>torch-bf16-efficient 1.059 (0.52) | fa2-bf16 0.6538 (0.84) | flashinfer-bf16 0.8008 (0.68) |  | liger-bf16 0.6801 (0.80) |
| llm_blkdec_b1_l4096_bf16 | rtx-3090 | 0.6403 | torch-bf16 1.085 (0.59) | torch-bf16-compile 0.7057 (0.91) | **torch-bf16-compile-cg 0.6992 (0.92)** | torch-bf16-compile-ro 1.235 (0.52) | torch-bf16-cudnn 1.188 (0.54)<br>torch-bf16-efficient 1.362 (0.47) | fa2-bf16 1.114 (0.57) | flashinfer-bf16 1.267 (0.51) |  | liger-bf16 1.087 (0.59) |
| llm_blkdec_b1_l4096_bf16 | a100 | 0.5803 | torch-bf16 0.9298 (0.62) | **torch-bf16-compile 0.5455 (1.06)** | torch-bf16-compile-cg 0.5505 (1.05) | torch-bf16-compile-ro 0.8554 (0.68) | torch-bf16-cudnn 1.04 (0.56)<br>torch-bf16-efficient 1.162 (0.50) | fa2-bf16 0.9093 (0.64) | flashinfer-bf16 1.079 (0.54) |  | liger-bf16 0.9048 (0.64) |
| llm_blkdec_b1_l4096_bf16 | rtx-4090 | 0.729 | torch-bf16 1.203 (0.61) | **torch-bf16-compile 0.789 (0.92)** | torch-bf16-compile-cg 0.8153 (0.89) | torch-bf16-compile-ro 1.358 (0.54) | torch-bf16-cudnn 1.253 (0.58)<br>torch-bf16-efficient 1.62 (0.45) | fa2-bf16 1.194 (0.61) | flashinfer-bf16 1.444 (0.50) |  | liger-bf16 1.284 (0.57) |
| llm_blkdec_b1_l4096_bf16 | h100 | 0.2812 | torch-bf16 0.5641 (0.50) | torch-bf16-compile 0.3796 (0.74) | torch-bf16-compile-cg 0.3309 (0.85) | **torch-bf16-compile-ro 0.3052 (0.92)** | torch-bf16-cudnn 0.6138 (0.46)<br>torch-bf16-efficient 0.7411 (0.38) | fa2-bf16 0.5355 (0.53)<br>fa3-bf16 0.5597 (0.50) | flashinfer-bf16 0.6816 (0.41) |  | liger-bf16 0.6267 (0.45) |
| llm_blkpre_t2048_bf16 | rtx-5080 | 9.596 | torch-bf16 8.89 (1.08) | torch-bf16-compile 8.232 (1.17) | **torch-bf16-compile-cg 8.23 (1.17)** | torch-bf16-compile-ro 8.493 (1.13) | torch-bf16-cudnn 8.86 (1.08)<br>torch-bf16-efficient 9.409 (1.02) | fa2-bf16 8.878 (1.08) | flashinfer-bf16 8.885 (1.08) | triton-bf16 8.931 (1.07) | liger-bf16 8.585 (1.12) |
| llm_blkpre_t2048_bf16 | rtx-3090 | 16.17 | torch-bf16 15.9 (1.02) | **torch-bf16-compile 14.95 (1.08)** | torch-bf16-compile-cg 14.97 (1.08) | torch-bf16-compile-ro 14.98 (1.08) | torch-bf16-cudnn 15.84 (1.02)<br>torch-bf16-efficient 16.34 (0.99) | fa2-bf16 15.63 (1.03) | flashinfer-bf16 15.62 (1.04) | triton-bf16 15.7 (1.03) | liger-bf16 15.13 (1.07) |
| llm_blkpre_t2048_bf16 | a100 | 8.483 | torch-bf16 4.615 (1.84) | torch-bf16-compile 4.966 (1.71) | torch-bf16-compile-cg 4.035 (2.10) | **torch-bf16-compile-ro 4.01 (2.12)** | torch-bf16-cudnn 4.58 (1.85)<br>torch-bf16-efficient 4.8 (1.77) | fa2-bf16 4.61 (1.84) | flashinfer-bf16 5.72 (1.48) | triton-bf16 5.903 (1.44) | liger-bf16 4.175 (2.03) |
| llm_blkpre_t2048_bf16 | rtx-4090 | 13.75 | torch-bf16 13.13 (1.05) | **torch-bf16-compile 12.29 (1.12)** | torch-bf16-compile-cg 12.38 (1.11) | torch-bf16-compile-ro 12.69 (1.08) | torch-bf16-cudnn 13.2 (1.04)<br>torch-bf16-efficient 13.47 (1.02) | fa2-bf16 13.13 (1.05) | flashinfer-bf16 13.25 (1.04) | triton-bf16 13.27 (1.04) | liger-bf16 12.72 (1.08) |
| llm_blkpre_t2048_bf16 | h100 | 6.2 | torch-bf16 3.38 (1.83) | torch-bf16-compile 2.93 (2.12) | torch-bf16-compile-cg 2.824 (2.20) | **torch-bf16-compile-ro 2.765 (2.24)** | torch-bf16-cudnn 3.44 (1.80)<br>torch-bf16-efficient 3.689 (1.68) | fa3-bf16 3.398 (1.82)<br>fa2-bf16 3.597 (1.72) | flashinfer-bf16 3.313 (1.87) | triton-bf16 3.443 (1.80) | liger-bf16 3.317 (1.87) |
| llm_dec_b1_l2048_bf16 | rtx-5080 | 0.1249 | torch-bf16 0.2215 (0.56) | **torch-bf16-compile 0.1526 (0.82)** | torch-bf16-compile-cg 0.1574 (0.79) | torch-bf16-compile-ro 0.2995 (0.42) | torch-bf16-cudnn 0.2837 (0.44)<br>torch-bf16-efficient 0.4118 (0.30) | fa2-bf16 0.201 (0.62) | flashinfer-bf16 0.3348 (0.37) |  | liger-bf16 0.2456 (0.51) |
| llm_dec_b1_l2048_bf16 | rtx-3090 | 0.1782 | torch-bf16 0.5483 (0.32) | **torch-bf16-compile 0.2433 (0.73)** | torch-bf16-compile-cg 0.2881 (0.62) | torch-bf16-compile-ro 0.5107 (0.35) | torch-bf16-cudnn 0.6015 (0.30)<br>torch-bf16-efficient 0.7059 (0.25) | fa2-bf16 0.5349 (0.33) | flashinfer-bf16 0.719 (0.25) |  | liger-bf16 0.5066 (0.35) |
| llm_dec_b1_l2048_bf16 | a100 | 0.1821 | torch-bf16 0.5629 (0.32) | torch-bf16-compile 0.2649 (0.69) | **torch-bf16-compile-cg 0.2615 (0.70)** | torch-bf16-compile-ro 0.3999 (0.46) | torch-bf16-cudnn 0.6091 (0.30)<br>torch-bf16-efficient 0.6947 (0.26) | fa2-bf16 0.5228 (0.35) | flashinfer-bf16 0.6977 (0.26) |  | liger-bf16 0.5046 (0.36) |
| llm_dec_b1_l2048_bf16 | rtx-4090 | 0.2002 | torch-bf16 0.5695 (0.35) | **torch-bf16-compile 0.3054 (0.66)** | torch-bf16-compile-cg 0.321 (0.62) | torch-bf16-compile-ro 0.5665 (0.35) | torch-bf16-cudnn 0.6253 (0.32)<br>torch-bf16-efficient 0.7873 (0.25) | fa2-bf16 0.5636 (0.36) | flashinfer-bf16 0.7946 (0.25) |  | liger-bf16 0.6035 (0.33) |
| llm_dec_b1_l2048_bf16 | h100 | 0.08362 | torch-bf16 0.37 (0.23) | torch-bf16-compile 0.2273 (0.37) | torch-bf16-compile-cg 0.1532 (0.55) | **torch-bf16-compile-ro 0.153 (0.55)** | torch-bf16-cudnn 0.4014 (0.21)<br>torch-bf16-efficient 0.4483 (0.19) | fa2-bf16 0.3324 (0.25)<br>fa3-bf16 0.3578 (0.23) | flashinfer-bf16 0.4817 (0.17) |  | liger-bf16 0.3954 (0.21) |
| llm_dec_b1_l8192_bf16 | rtx-5080 | 0.1646 | torch-bf16 0.2541 (0.65) | **torch-bf16-compile 0.1889 (0.87)** | torch-bf16-compile-cg 0.2049 (0.80) | torch-bf16-compile-ro 1.038 (0.16) | torch-bf16-cudnn 0.2936 (0.56)<br>torch-bf16-efficient 0.9945 (0.17) | fa2-bf16 0.2236 (0.74) | flashinfer-bf16 0.402 (0.41) |  | liger-bf16 0.2639 (0.62) |
| llm_dec_b1_l8192_bf16 | rtx-3090 | 0.223 | torch-bf16 0.5738 (0.39) | **torch-bf16-compile 0.3002 (0.74)** | torch-bf16-compile-cg 0.3252 (0.69) | torch-bf16-compile-ro 1.316 (0.17) | torch-bf16-cudnn 0.5996 (0.37)<br>torch-bf16-efficient 1.24 (0.18) | fa2-bf16 0.565 (0.39) | flashinfer-bf16 0.8266 (0.27) |  | liger-bf16 0.5243 (0.43) |
| llm_dec_b1_l8192_bf16 | a100 | 0.2265 | torch-bf16 0.5656 (0.40) | torch-bf16-compile 0.3134 (0.72) | **torch-bf16-compile-cg 0.2979 (0.76)** | torch-bf16-compile-ro 0.9151 (0.25) | torch-bf16-cudnn 0.6132 (0.37)<br>torch-bf16-efficient 1.195 (0.19) | fa2-bf16 0.5245 (0.43) | flashinfer-bf16 0.758 (0.30) |  | liger-bf16 0.5055 (0.45) |
| llm_dec_b1_l8192_bf16 | rtx-4090 | 0.257 | torch-bf16 0.5887 (0.44) | **torch-bf16-compile 0.339 (0.76)** | torch-bf16-compile-cg 0.3787 (0.68) | torch-bf16-compile-ro 1.527 (0.17) | torch-bf16-cudnn 0.6343 (0.41)<br>torch-bf16-efficient 1.533 (0.17) | fa2-bf16 0.5797 (0.44) | flashinfer-bf16 0.9508 (0.27) |  | liger-bf16 0.6154 (0.42) |
| llm_dec_b1_l8192_bf16 | h100 | 0.1128 | torch-bf16 0.3597 (0.31) | torch-bf16-compile 0.2345 (0.48) | torch-bf16-compile-cg 0.1636 (0.69) | **torch-bf16-compile-ro 0.1621 (0.70)** | torch-bf16-cudnn 0.4087 (0.28)<br>torch-bf16-efficient 0.8163 (0.14) | fa2-bf16 0.336 (0.34)<br>fa3-bf16 0.36 (0.31) | flashinfer-bf16 0.5258 (0.21) |  | liger-bf16 0.3968 (0.28) |
| llm_dec_b32_l2048_bf16 | rtx-5080 | 0.5164 | torch-bf16 0.655 (0.79) | **torch-bf16-compile 0.4738 (1.09)** | torch-bf16-compile-cg 0.4812 (1.07) | torch-bf16-compile-ro 7.691 (0.07) | torch-bf16-cudnn 0.5924 (0.87)<br>torch-bf16-efficient 3.903 (0.13) | fa2-bf16 0.5621 (0.92) | flashinfer-bf16 0.6529 (0.79) |  | liger-bf16 0.6831 (0.76) |
| llm_dec_b32_l2048_bf16 | rtx-3090 | 0.6747 | torch-bf16 1.141 (0.59) | torch-bf16-compile 0.8513 (0.79) | **torch-bf16-compile-cg 0.8439 (0.80)** | torch-bf16-compile-ro 8.801 (0.08) | torch-bf16-cudnn 0.8758 (0.77)<br>torch-bf16-efficient 5.091 (0.13) | fa2-bf16 0.9149 (0.74) | flashinfer-bf16 0.9993 (0.68) |  | liger-bf16 1.198 (0.56) |
| llm_dec_b32_l2048_bf16 | a100 | 0.5692 | torch-bf16 0.7667 (0.74) | **torch-bf16-compile 0.5621 (1.01)** | torch-bf16-compile-cg 0.6378 (0.89) | torch-bf16-compile-ro 5.493 (0.10) | torch-bf16-cudnn 0.7265 (0.78)<br>torch-bf16-efficient 3.678 (0.15) | fa2-bf16 0.718 (0.79) | flashinfer-bf16 0.8587 (0.66) |  | liger-bf16 0.7268 (0.78) |
| llm_dec_b32_l2048_bf16 | rtx-4090 | 0.6513 | torch-bf16 1.01 (0.64) | **torch-bf16-compile 0.6795 (0.96)** | torch-bf16-compile-cg 0.7068 (0.92) | torch-bf16-compile-ro 10.25 (0.06) | torch-bf16-cudnn 0.9247 (0.70)<br>torch-bf16-efficient 4.372 (0.15) | fa2-bf16 0.8941 (0.73) | flashinfer-bf16 1.082 (0.60) |  | liger-bf16 1.092 (0.60) |
| llm_dec_b32_l2048_bf16 | h100 | 0.2919 | torch-bf16 0.4293 (0.68) | torch-bf16-compile 0.2905 (1.01) | torch-bf16-compile-cg 0.2727 (1.07) | **torch-bf16-compile-ro 0.2717 (1.07)** | torch-bf16-cudnn 0.4491 (0.65)<br>torch-bf16-efficient 3.15 (0.09) | fa2-bf16 0.4388 (0.67)<br>fa3-bf16 0.485 (0.60) | flashinfer-bf16 0.5539 (0.53) |  | liger-bf16 0.4723 (0.62) |
| llm_dec_b32_l8192_bf16 | rtx-5080 | 1.672 | torch-bf16 1.89 (0.88) | **torch-bf16-compile 1.505 (1.11)** | torch-bf16-compile-cg 1.526 (1.10) | torch-bf16-compile-ro 29.89 (0.06) | torch-bf16-cudnn 1.524 (1.10)<br>torch-bf16-efficient 14.53 (0.12) | fa2-bf16 1.604 (1.04) | flashinfer-bf16 1.534 (1.09) |  | liger-bf16 1.917 (0.87) |
| llm_dec_b32_l8192_bf16 | rtx-3090 | 1.991 | torch-bf16 3.199 (0.62) | torch-bf16-compile 3.046 (0.65) | torch-bf16-compile-cg 3.122 (0.64) | torch-bf16-compile-ro 34.97 (0.06) | **torch-bf16-cudnn 1.794 (1.11)**<br>torch-bf16-efficient 19.69 (0.10) | fa2-bf16 2.198 (0.91) | flashinfer-bf16 1.916 (1.04) |  | liger-bf16 3.543 (0.56) |
| llm_dec_b32_l8192_bf16 | a100 | 1.643 | torch-bf16 1.462 (1.12) | torch-bf16-compile 1.789 (0.92) | torch-bf16-compile-cg 1.731 (0.95) | torch-bf16-compile-ro 21.45 (0.08) | **torch-bf16-cudnn 1.269 (1.29)**<br>torch-bf16-efficient 13.1 (0.13) | fa2-bf16 1.361 (1.21) | flashinfer-bf16 1.415 (1.16) |  | liger-bf16 1.434 (1.15) |
| llm_dec_b32_l8192_bf16 | rtx-4090 | 1.949 | torch-bf16 2.536 (0.77) | **torch-bf16-compile 1.885 (1.03)** | torch-bf16-compile-cg 1.936 (1.01) | torch-bf16-compile-ro 39.56 (0.05) | torch-bf16-cudnn 1.91 (1.02)<br>torch-bf16-efficient 15.76 (0.12) | fa2-bf16 1.886 (1.03) | flashinfer-bf16 2.068 (0.94) |  | liger-bf16 2.562 (0.76) |
| llm_dec_b32_l8192_bf16 | h100 | 0.875 | torch-bf16 0.7052 (1.24) | torch-bf16-compile 0.5939 (1.47) | torch-bf16-compile-cg 0.5967 (1.47) | **torch-bf16-compile-ro 0.58 (1.51)** | torch-bf16-cudnn 0.7385 (1.18)<br>torch-bf16-efficient 12.07 (0.07) | fa2-bf16 0.8295 (1.05)<br>fa3-bf16 0.9242 (0.95) | flashinfer-bf16 0.8394 (1.04) |  | liger-bf16 0.7551 (1.16) |
| llm_dec_b8_l2048_bf16 | rtx-5080 | 0.207 | torch-bf16 0.2936 (0.71) | **torch-bf16-compile 0.224 (0.92)** | torch-bf16-compile-cg 0.2351 (0.88) | torch-bf16-compile-ro 1.947 (0.11) | torch-bf16-cudnn 0.3533 (0.59)<br>torch-bf16-efficient 1.3 (0.16) | fa2-bf16 0.2853 (0.73) | flashinfer-bf16 0.4052 (0.51) |  | liger-bf16 0.3433 (0.60) |
| llm_dec_b8_l2048_bf16 | rtx-3090 | 0.4669 | torch-bf16 0.6315 (0.74) | **torch-bf16-compile 0.3452 (1.35)** | torch-bf16-compile-cg 0.3749 (1.25) | torch-bf16-compile-ro 2.378 (0.20) | torch-bf16-cudnn 0.6503 (0.72)<br>torch-bf16-efficient 1.668 (0.28) | fa2-bf16 0.6263 (0.75) | flashinfer-bf16 0.7707 (0.61) |  | liger-bf16 0.6009 (0.78) |
| llm_dec_b8_l2048_bf16 | a100 | 0.3205 | torch-bf16 0.5668 (0.57) | **torch-bf16-compile 0.3425 (0.94)** | torch-bf16-compile-cg 0.3515 (0.91) | torch-bf16-compile-ro 1.566 (0.20) | torch-bf16-cudnn 0.6399 (0.50)<br>torch-bf16-efficient 1.415 (0.23) | fa2-bf16 0.5598 (0.57) | flashinfer-bf16 0.7103 (0.45) |  | liger-bf16 0.55 (0.58) |
| llm_dec_b8_l2048_bf16 | rtx-4090 | 0.3125 | torch-bf16 0.6427 (0.49) | **torch-bf16-compile 0.3858 (0.81)** | torch-bf16-compile-cg 0.4195 (0.74) | torch-bf16-compile-ro 2.813 (0.11) | torch-bf16-cudnn 0.6787 (0.46)<br>torch-bf16-efficient 1.54 (0.20) | fa2-bf16 0.6483 (0.48) | flashinfer-bf16 0.82 (0.38) |  | liger-bf16 0.6934 (0.45) |
| llm_dec_b8_l2048_bf16 | h100 | 0.1447 | torch-bf16 0.3678 (0.39) | torch-bf16-compile 0.258 (0.56) | **torch-bf16-compile-cg 0.1706 (0.85)** | torch-bf16-compile-ro 0.1706 (0.85) | torch-bf16-cudnn 0.4055 (0.36)<br>torch-bf16-efficient 0.9931 (0.15) | fa2-bf16 0.341 (0.42)<br>fa3-bf16 0.3677 (0.39) | flashinfer-bf16 0.4826 (0.30) |  | liger-bf16 0.4154 (0.35) |
| llm_dec_b8_l8192_bf16 | rtx-5080 | 0.4941 | torch-bf16 0.5451 (0.91) | **torch-bf16-compile 0.4932 (1.00)** | torch-bf16-compile-cg 0.515 (0.96) | torch-bf16-compile-ro 7.751 (0.06) | torch-bf16-cudnn 0.5893 (0.84)<br>torch-bf16-efficient 4.208 (0.12) | fa2-bf16 0.5144 (0.96) | flashinfer-bf16 0.6309 (0.78) |  | liger-bf16 0.5729 (0.86) |
| llm_dec_b8_l8192_bf16 | rtx-3090 | 0.7979 | torch-bf16 0.8435 (0.95) | **torch-bf16-compile 0.8122 (0.98)** | torch-bf16-compile-cg 0.8232 (0.97) | torch-bf16-compile-ro 9.016 (0.09) | torch-bf16-cudnn 0.8607 (0.93)<br>torch-bf16-efficient 5.44 (0.15) | fa2-bf16 0.8595 (0.93) | flashinfer-bf16 0.9962 (0.80) |  | liger-bf16 0.8277 (0.96) |
| llm_dec_b8_l8192_bf16 | a100 | 0.6285 | torch-bf16 0.7354 (0.85) | torch-bf16-compile 0.6537 (0.96) | **torch-bf16-compile-cg 0.6208 (1.01)** | torch-bf16-compile-ro 5.541 (0.11) | torch-bf16-cudnn 0.7766 (0.81)<br>torch-bf16-efficient 3.897 (0.16) | fa2-bf16 0.6978 (0.90) | flashinfer-bf16 0.8489 (0.74) |  | liger-bf16 0.6883 (0.91) |
| llm_dec_b8_l8192_bf16 | rtx-4090 | 0.6241 | torch-bf16 0.8955 (0.70) | **torch-bf16-compile 0.6954 (0.90)** | torch-bf16-compile-cg 0.7276 (0.86) | torch-bf16-compile-ro 10.28 (0.06) | torch-bf16-cudnn 0.9191 (0.68)<br>torch-bf16-efficient 4.402 (0.14) | fa2-bf16 0.9012 (0.69) | flashinfer-bf16 1.063 (0.59) |  | liger-bf16 0.9457 (0.66) |
| llm_dec_b8_l8192_bf16 | h100 | 0.3063 | torch-bf16 0.4291 (0.71) | torch-bf16-compile 0.298 (1.03) | torch-bf16-compile-cg 0.2723 (1.12) | **torch-bf16-compile-ro 0.2707 (1.13)** | torch-bf16-cudnn 0.4538 (0.67)<br>torch-bf16-efficient 3.173 (0.10) | fa2-bf16 0.4436 (0.69)<br>fa3-bf16 0.4741 (0.65) | flashinfer-bf16 0.5649 (0.54) |  | liger-bf16 0.4727 (0.65) |
| llm_pre_t2048_bf16 | rtx-5080 | 2.627 | torch-bf16 2.225 (1.18) | torch-bf16-compile 1.972 (1.33) | **torch-bf16-compile-cg 1.971 (1.33)** | torch-bf16-compile-ro 2.006 (1.31) | torch-bf16-cudnn 2.184 (1.20)<br>torch-bf16-efficient 2.739 (0.96) | fa2-bf16 2.217 (1.18) | flashinfer-bf16 2.228 (1.18) | triton-bf16 2.293 (1.15) | liger-bf16 2.131 (1.23) |
| llm_pre_t2048_bf16 | rtx-3090 | 4.655 | torch-bf16 4.048 (1.15) | torch-bf16-compile 3.506 (1.33) | **torch-bf16-compile-cg 3.486 (1.34)** | torch-bf16-compile-ro 3.583 (1.30) | torch-bf16-cudnn 4.081 (1.14)<br>torch-bf16-efficient 4.398 (1.06) | fa2-bf16 6.575 (0.71) | flashinfer-bf16 6.694 (0.70) | triton-bf16 7.38 (0.63) | liger-bf16 6.399 (0.73) |
| llm_pre_t2048_bf16 | a100 | 2.707 | torch-bf16 1.435 (1.89) | torch-bf16-compile 1.041 (2.60) | **torch-bf16-compile-cg 1.033 (2.62)** | torch-bf16-compile-ro 1.065 (2.54) | torch-bf16-cudnn 1.422 (1.90)<br>torch-bf16-efficient 1.64 (1.65) | fa2-bf16 1.408 (1.92) | flashinfer-bf16 1.415 (1.91) | triton-bf16 1.558 (1.74) | liger-bf16 1.235 (2.19) |
| llm_pre_t2048_bf16 | rtx-4090 | 3.69 | torch-bf16 3.269 (1.13) | torch-bf16-compile 2.911 (1.27) | **torch-bf16-compile-cg 2.91 (1.27)** | torch-bf16-compile-ro 2.961 (1.25) | torch-bf16-cudnn 3.35 (1.10)<br>torch-bf16-efficient 3.623 (1.02) | fa2-bf16 3.271 (1.13) | flashinfer-bf16 3.403 (1.08) | triton-bf16 3.429 (1.08) | liger-bf16 3.105 (1.19) |
| llm_pre_t2048_bf16 | h100 | 1.764 | torch-bf16 0.9614 (1.83) | torch-bf16-compile 0.6815 (2.59) | torch-bf16-compile-cg 0.712 (2.48) | **torch-bf16-compile-ro 0.6509 (2.71)** | torch-bf16-cudnn 0.9414 (1.87)<br>torch-bf16-efficient 1.356 (1.30) | fa3-bf16 1.083 (1.63)<br>fa2-bf16 1.383 (1.27) | flashinfer-bf16 1.313 (1.34) | triton-bf16 1.092 (1.62) | liger-bf16 0.9394 (1.88) |
| llm_blkdec_b1_l4096_f8w | rtx-5080 | 0.3362 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | rtx-3090 | 0.5482 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | a100 | 0.5678 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | rtx-4090 | 0.4534 |  |  |  |  |  |  |  |  |  |
| llm_blkdec_b1_l4096_f8w | h100 | 0.2112 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-5080 | 0.05436 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-3090 | 0.1607 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | a100 | 0.1895 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | rtx-4090 | 0.1019 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l2048_f8w | h100 | 0.0797 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-5080 | 0.1543 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-3090 | 0.2301 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | a100 | 0.2395 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | rtx-4090 | 0.2317 |  |  |  |  |  |  |  |  |  |
| llm_dec_b1_l8192_f8w | h100 | 0.1057 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-5080 | 0.7509 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-3090 | 0.9826 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | a100 | 0.8763 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | rtx-4090 | 0.9651 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l2048_f8w | h100 | 0.442 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-5080 | 2.666 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-3090 | 3.135 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | a100 | 2.082 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | rtx-4090 | 3.107 |  |  |  |  |  |  |  |  |  |
| llm_dec_b32_l8192_f8w | h100 | 1.313 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-5080 | 0.2358 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-3090 | 0.8136 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | a100 | 0.3927 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | rtx-4090 | 0.338 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l2048_f8w | h100 | 0.1383 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-5080 | 0.7094 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-3090 | 1.361 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | a100 | 0.7336 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | rtx-4090 | 0.8757 |  |  |  |  |  |  |  |  |  |
| llm_dec_b8_l8192_f8w | h100 | 0.3671 |  |  |  |  |  |  |  |  |  |

### Conv / FIR / elementwise

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead | hand CUDA |
|---|---|---:|---|---|---|---|---|---|
| conv_2048_5 | rtx-5080 | 0.06647 |  | **torch-eager 0.1563 (0.43)** | torch-compile 0.1672 (0.40) | torch-compile-cg 0.1757 (0.38) | torch-compile-ro 0.1758 (0.38) |  |
| conv_2048_5 | rtx-3090 | 0.1139 |  | torch-eager 0.1853 (0.61) | torch-compile 0.1618 (0.70) | **torch-compile-cg 0.1608 (0.71)** | torch-compile-ro 0.1609 (0.71) |  |
| conv_2048_5 | a100 | 0.09762 |  | torch-eager 0.1741 (0.56) | torch-compile 0.1393 (0.70) | **torch-compile-cg 0.1382 (0.71)** | torch-compile-ro 0.1382 (0.71) |  |
| conv_2048_5 | rtx-4090 | 0.1009 |  | torch-eager 0.169 (0.60) | torch-compile 0.169 (0.60) | **torch-compile-cg 0.1679 (0.60)** | torch-compile-ro 0.1679 (0.60) |  |
| conv_2048_5 | h100 | 0.04371 |  | **torch-eager 0.0972 (0.45)** | torch-compile 0.0974 (0.45) | torch-compile-cg 0.0977 (0.45) | torch-compile-ro 0.0975 (0.45) |  |
| convmc_8_64_56_3 | rtx-5080 | 0.08205 |  | **torch-eager 0.1222 (0.67)** | torch-compile 0.1793 (0.46) | torch-compile-cg 0.1821 (0.45) | torch-compile-ro 0.1826 (0.45) |  |
| convmc_8_64_56_3 | rtx-3090 | 0.1501 |  | **torch-eager 0.0778 (1.93)** | torch-compile 0.17 (0.88) | torch-compile-cg 0.1618 (0.93) | torch-compile-ro 0.1618 (0.93) |  |
| convmc_8_64_56_3 | a100 | 0.2707 |  | **torch-eager 0.0881 (3.07)** | torch-compile 0.1864 (1.45) | torch-compile-cg 0.1731 (1.56) | torch-compile-ro 0.171 (1.58) |  |
| convmc_8_64_56_3 | rtx-4090 | 0.1867 |  | **torch-eager 0.0911 (2.05)** | torch-compile 0.171 (1.09) | torch-compile-cg 0.169 (1.10) | torch-compile-ro 0.1669 (1.12) |  |
| convmc_8_64_56_3 | h100 | 0.1193 |  | **torch-eager 0.0697 (1.71)** | torch-compile 0.1129 (1.06) | torch-compile-cg 0.1084 (1.10) | torch-compile-ro 0.1087 (1.10) |  |
| ew_4096 | rtx-5080 | 0.08805 | chain-eager 0.5714 (0.15) | torch-eager 0.5763 (0.15) | torch-compile 0.2605 (0.34) | torch-compile-cg 0.2657 (0.33) | torch-compile-ro 0.2668 (0.33) | **chain-fused 0.2478 (0.36)** |
| ew_4096 | rtx-3090 | 0.1466 | chain-eager 0.5632 (0.26) | torch-eager 0.5632 (0.26) | **torch-compile 0.2406 (0.61)** | torch-compile-cg 0.2406 (0.61) | torch-compile-ro 0.2406 (0.61) | chain-fused 0.2406 (0.61) |
| ew_4096 | a100 | 0.193 | chain-eager 0.3871 (0.50) | torch-eager 0.3604 (0.54) | **torch-compile 0.1567 (1.23)** | torch-compile-cg 0.1567 (1.23) | torch-compile-ro 0.1567 (1.23) | chain-fused 0.1812 (1.07) |
| ew_4096 | rtx-4090 | 0.156 | chain-eager 0.5878 (0.27) | torch-eager 0.598 (0.26) | torch-compile 0.2651 (0.59) | torch-compile-cg 0.2632 (0.59) | torch-compile-ro 0.2611 (0.60) | **chain-fused 0.2601 (0.60)** |
| ew_4096 | h100 | 0.09409 | chain-eager 0.2072 (0.45) | torch-eager 0.1692 (0.56) | **torch-compile 0.0712 (1.32)** | torch-compile-cg 0.0719 (1.31) | torch-compile-ro 0.0716 (1.31) | chain-fused 0.0827 (1.14) |
| ewconvew_2048_5 | rtx-5080 | 0.07822 |  | torch-eager 0.2126 (0.37) | **torch-compile 0.2092 (0.37)** | torch-compile-cg 0.2132 (0.37) | torch-compile-ro 0.2107 (0.37) |  |
| ewconvew_2048_5 | rtx-3090 | 0.131 |  | torch-eager 0.2929 (0.45) | **torch-compile 0.2458 (0.53)** | torch-compile-cg 0.2662 (0.49) | torch-compile-ro 0.2662 (0.49) |  |
| ewconvew_2048_5 | a100 | 0.1167 |  | **torch-eager 0.1956 (0.60)** | torch-compile 0.2232 (0.52) | torch-compile-cg 0.2181 (0.54) | torch-compile-ro 0.213 (0.55) |  |
| ewconvew_2048_5 | rtx-4090 | 0.1194 |  | torch-eager 0.2458 (0.49) | torch-compile 0.2212 (0.54) | **torch-compile-cg 0.2181 (0.55)** | torch-compile-ro 0.2191 (0.55) |  |
| ewconvew_2048_5 | h100 | 0.05485 |  | torch-eager 0.1297 (0.42) | torch-compile 0.1257 (0.44) | **torch-compile-cg 0.1162 (0.47)** | torch-compile-ro 0.117 (0.47) |  |
| fir_4194304_16 | rtx-5080 | 0.04549 |  | **torch-eager 0.1083 (0.42)** | torch-compile 0.1194 (0.38) | torch-compile-cg 0.13 (0.35) | torch-compile-ro 0.1279 (0.36) |  |
| fir_4194304_16 | rtx-3090 | 0.07618 |  | torch-eager 0.1116 (0.68) | torch-compile 0.1106 (0.69) | torch-compile-cg 0.1106 (0.69) | **torch-compile-ro 0.1097 (0.69)** |  |
| fir_4194304_16 | a100 | 0.06793 |  | torch-eager 0.0881 (0.77) | **torch-compile 0.086 (0.79)** | torch-compile-cg 0.086 (0.79) | torch-compile-ro 0.086 (0.79) |  |
| fir_4194304_16 | rtx-4090 | 0.07014 |  | **torch-eager 0.1075 (0.65)** | torch-compile 0.1085 (0.65) | torch-compile-cg 0.1075 (0.65) | torch-compile-ro 0.1075 (0.65) |  |
| fir_4194304_16 | h100 | 0.03159 |  | **torch-eager 0.0595 (0.53)** | torch-compile 0.0596 (0.53) | torch-compile-cg 0.0599 (0.53) | torch-compile-ro 0.0595 (0.53) |  |
| firew_rowsum_2048x2048_16 | rtx-5080 | 0.05164 |  | torch-eager 0.1369 (0.38) | **torch-compile 0.1294 (0.40)** | torch-compile-cg 0.1302 (0.40) | torch-compile-ro 0.1321 (0.39) |  |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1042 |  | torch-eager 0.219 (0.48) | torch-compile 0.1403 (0.74) | **torch-compile-cg 0.1393 (0.75)** | torch-compile-ro 0.1403 (0.74) |  |
| firew_rowsum_2048x2048_16 | a100 | 0.08268 |  | torch-eager 0.1311 (0.63) | torch-compile 0.126 (0.66) | **torch-compile-cg 0.1178 (0.70)** | torch-compile-ro 0.1198 (0.69) |  |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.08167 |  | torch-eager 0.1516 (0.54) | torch-compile 0.1178 (0.69) | **torch-compile-cg 0.1167 (0.70)** | torch-compile-ro 0.1167 (0.70) |  |
| firew_rowsum_2048x2048_16 | h100 | 0.03816 |  | torch-eager 0.0853 (0.45) | torch-compile 0.0661 (0.58) | torch-compile-cg 0.0651 (0.59) | **torch-compile-ro 0.0644 (0.59)** |  |
| convmc_8_64_56_3_tf32 | rtx-5080 | 0.08025 |  | **torch-eager-tf32 0.0719 (1.12)** | torch-compile-tf32 0.1217 (0.66) | torch-compile-cg-tf32 0.1201 (0.67) | torch-compile-ro-tf32 0.1213 (0.66) |  |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1534 |  | **torch-eager-tf32 0.087 (1.76)** | torch-compile-tf32 0.1475 (1.04) | torch-compile-cg-tf32 0.1423 (1.08) | torch-compile-ro-tf32 0.1423 (1.08) |  |
| convmc_8_64_56_3_tf32 | a100 | 0.2485 |  | **torch-eager-tf32 0.0481 (5.17)** | torch-compile-tf32 0.084 (2.96) | torch-compile-cg-tf32 0.0963 (2.58) | torch-compile-ro-tf32 0.0952 (2.61) |  |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.1868 |  | **torch-eager-tf32 0.085 (2.20)** | torch-compile-tf32 0.1434 (1.30) | torch-compile-cg-tf32 0.1413 (1.32) | torch-compile-ro-tf32 0.1423 (1.31) |  |
| convmc_8_64_56_3_tf32 | h100 | 0.1187 |  | **torch-eager-tf32 0.0351 (3.38)** | torch-compile-tf32 0.0548 (2.17) | torch-compile-cg-tf32 0.0426 (2.79) | torch-compile-ro-tf32 0.0427 (2.78) |  |

### Fan-in/out, loops, CPU+GPU

| workload | part | Mapal ms | vendor libs | torch eager | torch.compile | compile + CUDA graphs | reduce-overhead |
|---|---|---:|---|---|---|---|---|
| diamond_2048 | rtx-5080 | 1.369 | **chain-lt 1.378 (0.99)**<br>chain-streams 1.395 (0.98)<br>chain-eager 1.403 (0.98)<br>chain-cutlass 1.647 (0.83) | torch-eager 1.401 (0.98) | torch-compile 1.402 (0.98) | torch-compile-cg 1.405 (0.97) | torch-compile-ro 1.417 (0.97) |
| diamond_2048 | rtx-3090 | 2.419 | **chain-streams 2.678 (0.90)**<br>chain-lt 2.708 (0.89)<br>chain-eager 2.724 (0.89)<br>chain-cutlass 3.168 (0.76) | torch-eager 2.759 (0.88) | torch-compile 2.747 (0.88) | torch-compile-cg 2.746 (0.88) | torch-compile-ro 2.927 (0.83) |
| diamond_2048 | a100 | 2.701 | **chain-lt 2.939 (0.92)**<br>chain-streams 2.943 (0.92)<br>chain-eager 2.971 (0.91)<br>chain-cutlass 3.457 (0.78) | torch-eager 3.784 (0.71) | torch-compile 3.755 (0.72) | torch-compile-cg 3.089 (0.87) | torch-compile-ro 2.948 (0.92) |
| diamond_2048 | rtx-4090 | 1.79 | **chain-cutlass 1.877 (0.95)**<br>chain-lt 1.919 (0.93)<br>chain-streams 1.934 (0.93)<br>chain-eager 1.942 (0.92) | torch-eager 1.94 (0.92) | torch-compile 1.92 (0.93) | torch-compile-cg 1.921 (0.93) | torch-compile-ro 1.936 (0.92) |
| diamond_2048 | h100 | 1.443 | chain-lt 1.486 (0.97)<br>chain-streams 1.496 (0.96)<br>chain-eager 1.499 (0.96)<br>chain-cutlass 1.677 (0.86) | **torch-eager 1.269 (1.14)** | torch-compile 1.339 (1.08) | torch-compile-cg 1.273 (1.13) | torch-compile-ro 1.277 (1.13) |
| fanout_2048 | rtx-5080 | 0.9136 | **chain-streams 0.9114 (1.00)**<br>chain-eager 0.9236 (0.99) | torch-eager 0.9439 (0.97) | torch-compile 0.951 (0.96) | torch-compile-cg 0.9536 (0.96) | torch-compile-ro 0.9562 (0.96) |
| fanout_2048 | rtx-3090 | 1.653 | **chain-streams 1.805 (0.92)**<br>chain-eager 1.84 (0.90) | torch-eager 1.876 (0.88) | torch-compile 1.857 (0.89) | torch-compile-cg 1.837 (0.90) | torch-compile-ro 1.923 (0.86) |
| fanout_2048 | a100 | 1.809 | chain-streams 2.505 (0.72)<br>chain-eager 2.527 (0.72) | torch-eager 2.539 (0.71) | torch-compile 2.533 (0.71) | **torch-compile-cg 1.99 (0.91)** | torch-compile-ro 1.992 (0.91) |
| fanout_2048 | rtx-4090 | 1.214 | **chain-streams 1.285 (0.94)**<br>chain-eager 1.297 (0.94) | torch-eager 1.318 (0.92) | torch-compile 1.325 (0.92) | torch-compile-cg 1.326 (0.92) | torch-compile-ro 1.297 (0.94) |
| fanout_2048 | h100 | 0.9712 | chain-streams 0.8508 (1.14)<br>chain-eager 0.8595 (1.13) | torch-eager 0.8576 (1.13) | torch-compile 0.8476 (1.15) | torch-compile-cg 0.8459 (1.15) | **torch-compile-ro 0.8438 (1.15)** |
| indep_2048 | rtx-5080 | 0.9372 | **chain-lt 0.9155 (1.02)**<br>chain-streams 0.9318 (1.01)<br>chain-eager 0.938 (1.00)<br>chain-cutlass 1.092 (0.86) | torch-eager 0.9327 (1.00) | torch-compile 0.94 (1.00) | torch-compile-cg 0.9432 (0.99) | torch-compile-ro 0.9564 (0.98) |
| indep_2048 | rtx-3090 | 1.629 | **chain-streams 1.77 (0.92)**<br>chain-lt 1.809 (0.90)<br>chain-eager 1.817 (0.90)<br>chain-cutlass 2.075 (0.79) | torch-eager 1.86 (0.88) | torch-compile 1.841 (0.89) | torch-compile-cg 1.864 (0.87) | torch-compile-ro 1.893 (0.86) |
| indep_2048 | a100 | 1.758 | chain-lt 1.961 (0.90)<br>chain-streams 1.962 (0.90)<br>chain-eager 1.995 (0.88)<br>chain-cutlass 2.307 (0.76) | torch-eager 2.536 (0.69) | torch-compile 1.97 (0.89) | **torch-compile-cg 1.958 (0.90)** | torch-compile-ro 1.973 (0.89) |
| indep_2048 | rtx-4090 | 1.207 | **chain-cutlass 1.253 (0.96)**<br>chain-lt 1.283 (0.94)<br>chain-streams 1.294 (0.93)<br>chain-eager 1.302 (0.93) | torch-eager 1.305 (0.93) | torch-compile 1.284 (0.94) | torch-compile-cg 1.282 (0.94) | torch-compile-ro 1.3 (0.93) |
| indep_2048 | h100 | 0.9733 | chain-lt 0.9438 (1.03)<br>chain-streams 0.9523 (1.02)<br>chain-eager 0.9557 (1.02)<br>chain-cutlass 1.073 (0.91) | torch-eager 0.8507 (1.14) | torch-compile 0.8459 (1.15) | **torch-compile-cg 0.8421 (1.16)** | torch-compile-ro 0.8522 (1.14) |
| indep_512 | rtx-5080 | 0.03936 | **chain-streams 0.0365 (1.08)**<br>chain-lt 0.043 (0.92)<br>chain-eager 0.0451 (0.87)<br>chain-cutlass 0.1044 (0.38) | torch-eager 0.0496 (0.79) | torch-compile 0.0491 (0.80) | torch-compile-cg 0.0556 (0.71) | torch-compile-ro 0.0673 (0.58) |
| indep_512 | rtx-3090 | 0.07504 | chain-streams 0.0532 (1.41)<br>chain-lt 0.0543 (1.38)<br>chain-eager 0.0573 (1.31)<br>chain-cutlass 0.126 (0.60) | torch-eager 0.0666 (1.13) | **torch-compile 0.0471 (1.59)** | torch-compile-cg 0.0471 (1.59) | torch-compile-ro 0.0553 (1.36) |
| indep_512 | a100 | 0.0981 | chain-streams 0.0676 (1.45)<br>chain-lt 0.0778 (1.26)<br>chain-eager 0.0799 (1.23)<br>chain-cutlass 0.2447 (0.40) | **torch-eager 0.0676 (1.45)** | torch-compile 0.084 (1.17) | torch-compile-cg 0.0809 (1.21) | torch-compile-ro 0.0809 (1.21) |
| indep_512 | rtx-4090 | 0.07232 | chain-lt 0.0553 (1.31)<br>chain-streams 0.0563 (1.28)<br>chain-eager 0.0594 (1.22)<br>chain-cutlass 0.1761 (0.41) | torch-eager 0.0614 (1.18) | **torch-compile 0.0471 (1.54)** | torch-compile-cg 0.0471 (1.54) | torch-compile-ro 0.0573 (1.26) |
| indep_512 | h100 | 0.04893 | chain-streams 0.0311 (1.57)<br>chain-lt 0.0336 (1.46)<br>chain-eager 0.0428 (1.14)<br>chain-cutlass 0.1201 (0.41) | torch-eager 0.0419 (1.17) | **torch-compile 0.0268 (1.83)** | torch-compile-cg 0.0269 (1.82) | torch-compile-ro 0.0332 (1.47) |
| loop_4096 | rtx-5080 | 26.9 | **chain-lt 28.51 (0.94)**<br>chain-cutlass 28.76 (0.94)<br>chain-eager 30.21 (0.89) | torch-eager 29.28 (0.92) | torch-compile 29.29 (0.92) | torch-compile-cg 29.24 (0.92) | torch-compile-ro 29.28 (0.92) |
| loop_4096 | rtx-3090 | 63.56 | chain-lt 109.8 (0.58)<br>chain-eager 117.4 (0.54)<br>chain-cutlass 137.6 (0.46) | torch-eager 52.06 (1.22) | torch-compile 52.05 (1.22) | **torch-compile-cg 52.03 (1.22)** | torch-compile-ro 52.14 (1.22) |
| loop_4096 | a100 | 52.81 | chain-eager 58.89 (0.90)<br>chain-lt 60.22 (0.88)<br>chain-cutlass 60.8 (0.87) | torch-eager 58.83 (0.90) | torch-compile 60.79 (0.87) | torch-compile-cg 60.76 (0.87) | **torch-compile-ro 58.44 (0.90)** |
| loop_4096 | rtx-4090 | 36.33 | **chain-lt 38.66 (0.94)**<br>chain-cutlass 39.19 (0.93)<br>chain-eager 39.73 (0.91) | torch-eager 39.78 (0.91) | torch-compile 39.75 (0.91) | torch-compile-cg 39.71 (0.92) | torch-compile-ro 39.55 (0.92) |
| loop_4096 | h100 | 32.16 | **chain-lt 30.9 (1.04)**<br>chain-eager 31.42 (1.02)<br>chain-cutlass 36.09 (0.89) | torch-eager 31.27 (1.03) | torch-compile 31.37 (1.03) | torch-compile-cg 31.32 (1.03) | torch-compile-ro 31.35 (1.03) |
| loopmm_1024_8 | rtx-5080 | 0.6093 |  | torch-eager 0.6372 (0.96) | **torch-compile 0.6106 (1.00)** | torch-compile-cg 0.6163 (0.99) | torch-compile-ro 0.6551 (0.93) |
| loopmm_1024_8 | rtx-3090 | 1.11 |  | torch-eager 1.05 (1.06) | torch-compile 0.9329 (1.19) | **torch-compile-cg 0.9277 (1.20)** | torch-compile-ro 0.9902 (1.12) |
| loopmm_1024_8 | a100 | 1.234 |  | **torch-eager 1.044 (1.18)** | torch-compile 1.35 (0.91) | torch-compile-cg 1.343 (0.92) | torch-compile-ro 1.343 (0.92) |
| loopmm_1024_8 | rtx-4090 | 0.8405 |  | torch-eager 0.7199 (1.17) | torch-compile 0.7188 (1.17) | **torch-compile-cg 0.7127 (1.18)** | torch-compile-ro 0.7127 (1.18) |
| loopmm_1024_8 | h100 | 0.6196 |  | torch-eager 0.6518 (0.95) | torch-compile 0.4972 (1.25) | **torch-compile-cg 0.4936 (1.26)** | torch-compile-ro 0.5106 (1.21) |
| rt_mlp2_1024 | rtx-5080 | 2.554 |  | torch-eager 3.363 (0.76) | **torch-compile 2.575 (0.99)** | torch-compile-cg 2.575 (0.99) | torch-compile-ro 2.655 (0.96) |
| rt_mlp2_1024 | rtx-3090 | 4.348 |  | torch-eager 7.051 (0.62) | **torch-compile 4.831 (0.90)** | torch-compile-cg 4.867 (0.89) | torch-compile-ro 4.992 (0.87) |
| rt_mlp2_1024 | a100 | 5.485 |  | torch-eager 5.471 (1.00) | torch-compile 5.618 (0.98) | torch-compile-cg 5.574 (0.98) | **torch-compile-ro 4.347 (1.26)** |
| rt_mlp2_1024 | rtx-4090 | 3.372 |  | torch-eager 4.382 (0.77) | torch-compile 3.068 (1.10) | **torch-compile-cg 2.998 (1.12)** | torch-compile-ro 3.037 (1.11) |
| rt_mlp2_1024 | h100 | 2.684 |  | torch-eager 3.528 (0.76) | torch-compile 2.341 (1.15) | **torch-compile-cg 2.268 (1.18)** | torch-compile-ro 2.457 (1.09) |
| rt_mlp2_2048 | rtx-5080 | 14.66 |  | torch-eager 19.41 (0.76) | torch-compile 15.7 (0.93) | **torch-compile-cg 15.69 (0.93)** | torch-compile-ro 15.78 (0.93) |
| rt_mlp2_2048 | rtx-3090 | 26.39 |  | torch-eager 38.92 (0.68) | torch-compile 30.48 (0.87) | **torch-compile-cg 30.45 (0.87)** | torch-compile-ro 30.81 (0.86) |
| rt_mlp2_2048 | a100 | 28.51 |  | torch-eager 38.2 (0.75) | torch-compile 31.73 (0.90) | **torch-compile-cg 31.52 (0.90)** | torch-compile-ro 31.68 (0.90) |
| rt_mlp2_2048 | rtx-4090 | 19.17 |  | torch-eager 25.8 (0.74) | **torch-compile 21 (0.91)** | torch-compile-cg 21.01 (0.91) | torch-compile-ro 21 (0.91) |
| rt_mlp2_2048 | h100 | 15.09 |  | torch-eager 19.63 (0.77) | torch-compile 16.47 (0.92) | **torch-compile-cg 16.37 (0.92)** | torch-compile-ro 16.49 (0.92) |
| twin_2048 | rtx-5080 | 0.9203 |  | torch-eager 0.9514 (0.97) | **torch-compile 0.9463 (0.97)** | torch-compile-cg 0.9475 (0.97) | torch-compile-ro 0.9492 (0.97) |
| twin_2048 | rtx-3090 | 1.587 |  | torch-eager 1.799 (0.88) | **torch-compile 1.738 (0.91)** | torch-compile-cg 1.749 (0.91) | torch-compile-ro 1.889 (0.84) |
| twin_2048 | a100 | 1.748 |  | **torch-eager 1.952 (0.90)** | torch-compile 1.957 (0.89) | torch-compile-cg 1.956 (0.89) | torch-compile-ro 1.954 (0.89) |
| twin_2048 | rtx-4090 | 1.2 |  | torch-eager 1.28 (0.94) | torch-compile 1.279 (0.94) | **torch-compile-cg 1.277 (0.94)** | torch-compile-ro 1.278 (0.94) |
| twin_2048 | h100 | 0.9686 |  | torch-eager 0.9972 (0.97) | **torch-compile 0.8378 (1.16)** | torch-compile-cg 0.8397 (1.15) | torch-compile-ro 0.8403 (1.15) |

## Command

```
python3 benches/local/matrix.py --skip 'llm_kv[xn]_' \
  rtx-5080=.claude/raw-evidence/local-rtx-5080/gapfix-integrate/resS/results-rtx-5080.txt@d945c641,.claude/raw-evidence/local-rtx-5080/gapfix-integrate/resM/results-rtx-5080.txt@d945c641,.claude/raw-evidence/local-rtx-5080/gapfix-integrate/flyab/results-rtx-5080.txt@d450014d,.claude/raw-evidence/local-rtx-5080/gapfix-integrate/zeroab2/results-rtx-5080.txt@9abeb446 \
  rtx-3090=.claude/raw-evidence/cross-arch/f1d3a04f/rtx-3090/results-rtx-3090-comp.txt@f1d3a04f,.claude/raw-evidence/cross-arch/f1d3a04f/rtx-3090/results-rtx-3090.txt@f1d3a04f \
  a100=.claude/raw-evidence/cross-arch/f1d3a04f/a100/results-a100-comp.txt@f1d3a04f,.claude/raw-evidence/cross-arch/f1d3a04f/a100/results-a100.txt@f1d3a04f \
  rtx-4090=.claude/raw-evidence/cross-arch/f1d3a04f/rtx-4090/results-rtx-4090-comp.txt@f1d3a04f,.claude/raw-evidence/cross-arch/f1d3a04f/rtx-4090/results-rtx-4090.txt@f1d3a04f \
  h100=.claude/raw-evidence/cross-arch/f1d3a04f/h100/results-h100-comp.txt@f1d3a04f,.claude/raw-evidence/cross-arch/f1d3a04f/h100/results-h100.txt@f1d3a04f
```

Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row (all fair legs, sources, the bf16 reference, the old rule's best).
