# Current results

Every number behind the README's charts: how the round was run, how correctness was checked, the generated headline
tables, the full model and the kernel suite. Back to the [README](../../README.md#results); the performance index:
[`docs/performance/`](README.md).

Five GPUs, 2026-11-26: an RTX 5080 at home and an RTX 3090, A100, RTX 4090 and H100 rented. The full model on all
five and the kernel suite on the rented four ran this round's final build (`b8429388`, re-timed on `9be9886d`'s
runtime); the RTX 5080's suite column is the gapfix integration's (`d945c641`). Type against type: the f32 program
against engines computing in f32, the bf16 program against bf16 engines, each suite row against competitors at its
own precision. No gain is claimed across precisions.

**Correctness.** Every timed program passes its value gate on every GPU before it is timed: within the stated error
bound on the default face, in any order of operations, and bit for bit with the interpreter in the development modes
([ADR-0049](../decisions/ADR-0049-correctness-is-the-flow-and-the-bound.md)). One bf16 program
(`px_llm_blkdec_b2_l16_bf16`) first read as outside its bound on the rented GPUs: the compiled result was right — bit
for bit the program with the bf16 stores its `: bf16 | f32` permits — and the gate's bound was the unsound part, so
the gate now checks a bound composed through every operation of the program
([ADR-0041](../decisions/ADR-0041-compute-precision-permission-candidate.md)). The model's generated tokens are
checked against the Hugging Face reference of the same class, run on the same box.

<!-- results-headline:start -->

**Kernel and layer suite**: programs where Mapal is faster, level or slower than the fastest competitor at the program's own precision, of the 119 timed on every GPU, both sides timed at the program boundary. Level: within the row's measured run-to-run variation or 5%, whichever is larger. No averages.

| GPU | faster | level | slower |
| --- | ---: | ---: | ---: |
| RTX 5080 | 46 | 43 | 30 |
| RTX 3090 | 39 | 30 | 50 |
| A100 | 57 | 11 | 51 |
| RTX 4090 | 49 | 24 | 46 |
| H100 | 30 | 8 | 81 |


**Llama-3.2-1B, the full model**: Mapal ÷ the fastest engine of the same class, prompts 128 / 512 / 2048 (batch 1, 128 generated tokens, medians of 5).

| GPU | class | time to first token | decode per token | at 512: Mapal vs the fastest engine |
| --- | --- | ---: | ---: | --- |
| RTX 5080 | f32 | **0.62** / **0.62** / **0.53** | **0.55** / **0.58** / **0.58** | 25.4 vs 41.0 ms (vLLM f32); 3.29 vs 5.72 ms / token (gpt-fast f32) |
| RTX 5080 | bf16 | 1.13 / 1.05 / 1.15 | 1.10 / 1.10 / 1.06 | 14.1 vs 13.5 ms (llama.cpp bf16); 3.29 vs 2.98 ms / token (gpt-fast) |
| RTX 3090 | f32 | 1.04 / 0.95 / **0.82** | **0.65** / **0.66** / **0.67** | 56.0 vs 58.8 ms (llama.cpp f32); 4.04 vs 6.15 ms / token (gpt-fast f32) |
| RTX 3090 | bf16 | 1.55 / 1.17 / 1.30 | 1.23 / 1.23 / 1.25 | 23.8 vs 20.4 ms (SGLang); 4.01 vs 3.27 ms / token (gpt-fast) |
| A100 | f32 | **0.73** / **0.52** / **0.43** | **0.86** / **0.91** / **0.89** | 40.0 vs 76.8 ms (vLLM f32); 3.82 vs 4.21 ms / token (gpt-fast f32) |
| A100 | bf16 | 1.87 / 1.44 / 2.12 | 1.56 / 1.55 / 1.57 | 16.1 vs 11.1 ms (SGLang); 3.69 vs 2.38 ms / token (gpt-fast) |
| RTX 4090 | f32 | **0.72** / **0.64** / **0.53** | **0.62** / **0.63** / **0.63** | 20.1 vs 31.6 ms (vLLM f32); 3.47 vs 5.54 ms / token (gpt-fast f32) |
| RTX 4090 | bf16 | 1.44 / 1.16 / 1.24 | 1.18 / 1.19 / 1.19 | 11.9 vs 10.3 ms (gpt-fast); 3.47 vs 2.93 ms / token (gpt-fast) |
| H100 | f32 | 1.04 / **0.74** / **0.50** | **0.93** / 0.95 / 0.95 | 21.8 vs 29.6 ms (HF eager f32); 2.06 vs 2.16 ms / token (gpt-fast f32) |
| H100 | bf16 | 1.30 / 1.88 / 2.37 | 1.49 / 1.56 / 1.51 | 7.60 vs 4.04 ms (gpt-fast); 2.06 vs 1.32 ms / token (gpt-fast) |
| H100, `wgmma` (another box) | bf16 | **0.75** / 1.25 / 1.58 | 1.49 / 1.49 / 1.45 | 5.60 vs 4.47 ms (gpt-fast); 2.00 vs 1.34 ms / token (gpt-fast) |

Below 1: Mapal faster; bold: faster by more than 5%. The **bf16** rows are like for like. The **f32 math** rows are not: Mapal computes in f32 on the model's bf16 weights (half the bytes per token), gpt-fast, vLLM, llama.cpp and Hugging Face in f32 on f32 weights with TF32 off (SGLang has no f32 path and TensorRT-LLM's f32 output is wrong: neither is compared); **bf16** (`fold: bf16 | f32`, bf16 KV cache) against gpt-fast, vLLM, SGLang, TensorRT-LLM, llama.cpp and Hugging Face in bf16. llama.cpp in f16 is another 16-bit format and is not compared. The H100 `wgmma` row is the round's code plus the 16-bit warpgroup tiles, measured on another H100 SXM against gpt-fast on that box (the H100's fastest bf16 engine in the round); the other engines were not rerun there.

<!-- results-headline:end -->

## The full model

The readable [`examples/llama.mapal`](../../examples/llama.mapal) on all five GPUs against gpt-fast, vLLM, SGLang,
TensorRT-LLM, llama.cpp and Hugging Face (eager, and static cache + `torch.compile`), in each precision class. Each
bar is Mapal's time over the fastest engine of its class on that GPU; the leader is named on the right.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../benches/results/charts/llm-bf16-dark.svg">
  <img alt="Llama-3.2-1B, bf16 class, five GPUs: Mapal's time over the fastest bf16 engine's, time to first token and decode at a 512-token prompt" src="../../benches/results/charts/llm-bf16-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../benches/results/charts/llm-f32-dark.svg">
  <img alt="Llama-3.2-1B, f32 class, five GPUs: Mapal's time over the fastest f32 engine's, time to first token and decode at a 512-token prompt" src="../../benches/results/charts/llm-f32-light.svg">
</picture>

- **f32 math, not like for like** (the program as written: f32 activations on the bf16 weights, while the f32
  engines read f32 weights, twice the bytes per token): ahead of every f32 engine on decode
  on all five GPUs (0.55–0.95×) and on time to first token from a 512-token prompt up (0.43–0.95×); at 128 tokens
  4 % behind on the RTX 3090 and H100. Its generated tokens equal the Hugging Face f32 reference's, 128 of 128, at
  every prompt length on every GPU. Peak GPU memory 2.7–3.5 GB, against 5.1 GB and more for every f32 engine.
- **bf16, like for like** (`fold: bf16 | f32`, bf16 KV cache): behind the fastest bf16 engine on every GPU, closest on the
  RTX 5080 (1.05–1.15× to the first token, 1.06–1.10× decode) and furthest on the A100 and H100 (up to 2.12× and
  2.37× to the first token at 2048, 1.49–1.57× decode), where gpt-fast's CUDA-graph decode and the TensorRT-LLM /
  SGLang prefill are strongest. Since this round the H100's 16-bit tiles run on its warpgroup matrix instruction
  (`wgmma`): on another H100 SXM, against gpt-fast on the same box, the first token at 128 / 512 / 2048 went from
  1.09 / 1.66 / 2.17× to 0.75 / 1.25 / 1.58× (2.70 / 5.60 / 18.11 ms against 3.58 / 4.47 / 11.45), decode unchanged
  ([wgmma-16bit](../../benches/results/evidence/hopper/wgmma-16bit.md)). Its tokens leave the bf16 reference after a near
  tie, as the bf16 engines' do among themselves (on the RTX 5080, 7 to 89 equal tokens at 512); on the RTX 5080 and
  A100 they can vary from run to run, because the tensor tile adds partial sums in arrival order. Peak memory
  2.7–3.3 GB, level with llama.cpp and HF eager, against 8.8–47 GB for the pooled servers (vLLM, SGLang,
  TensorRT-LLM).
- llama.cpp in f16 is another 16-bit format: recorded, compared with neither class. SGLang has no f32 path, and
  TensorRT-LLM's f32 build gives wrong tokens: neither is in the f32 class.

Every engine at every prompt length, the tokens and the peaks:
[`full-model.json`](../../benches/results/evidence/full-model.json); the RTX 5080 with where each side's time goes and
what is next: [`full-model.md`](../../benches/results/evidence/local-rtx-5080/full-model.md); the rented GPUs:
[`cross-arch`](../../benches/results/evidence/cross-arch/README.md#b8429388--9be9886d-five-gpu-round-bf16-integration-every-competitor-leg-at-the-program-boundary-the-full-model).

## The kernel and layer suite

GEMMs and projections, MLP chains, attention and softmax, LLM layers, convolution and FIR, fan-in / fan-out and
loops, on every GPU of the round, each program against the fastest of cuBLAS, cuBLASLt, CUTLASS, PyTorch eager and
`torch.compile` (with and without CUDA graphs), SDPA, FlashAttention-2/3, FlashInfer, hand-written Triton, Liger
and hand-written CUDA.

Both sides are timed at the program boundary: the device idle, then the host clock around the call, its dispatch
and launches inside. Each program is classed per GPU as faster, level (within its measured run-to-run variation or
5 %, whichever is larger) or slower than the fastest same-precision competitor; no average is taken. Mapal is faster
on 30 to 57 of the 119 programs per GPU, level on 8 to 43, slower on 30 to 81. Attention and softmax are faster on
9 / 9 / 12 / 12 / 5 of 14 programs and convolution, FIR and elementwise chains on 6 / 5 / 4 / 5 / 4 of 7 (RTX 5080 /
RTX 3090 / A100 / RTX 4090 / H100); the H100 trails on attention (slower on 8). GEMM cells and projections are the weak family: slower on 15 / 22 / 24 / 19 / 39 of 46
(RTX 5080 / RTX 3090 / A100 / RTX 4090 / H100). The TF32 rows (mostly GEMMs, against cuBLAS) are slower on 10 to 13 of
13 on every GPU, and the bf16-weight rows (LLM layers, against `torch.compile`) on 6 to 8 of 9.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../benches/results/charts/suite-gpus-dark.svg">
  <img alt="Every program's Mapal time over the fastest same-precision competitor's, per GPU, wins and losses apart" src="../../benches/results/charts/suite-gpus-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../benches/results/charts/suite-groups-dark.svg">
  <img alt="Programs faster, level and slower by workload family and by precision class, per GPU" src="../../benches/results/charts/suite-groups-light.svg">
</picture>

<!-- results-programs:start -->

**By family and precision class**: programs faster / level / slower, per GPU.

| group | programs | RTX 5080 | RTX 3090 | A100 | RTX 4090 | H100 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM cells and projections | 46 | 13 / 18 / 15 | 7 / 17 / 22 | 14 / 8 / 24 | 13 / 14 / 19 | 2 / 5 / 39 |
| MLP chains | 6 | 1 / 5 / 0 | 0 / 1 / 5 | 3 / 0 / 3 | 4 / 0 / 2 | 0 / 0 / 6 |
| Attention, softmax, long rows | 14 | 9 / 5 / 0 | 9 / 0 / 5 | 12 / 0 / 2 | 12 / 2 / 0 | 5 / 1 / 8 |
| LLM layers: decode, prefill, block | 37 | 16 / 7 / 14 | 18 / 5 / 14 | 18 / 2 / 17 | 10 / 6 / 21 | 19 / 1 / 17 |
| Conv, FIR, elementwise | 7 | 6 / 0 / 1 | 5 / 0 / 2 | 4 / 1 / 2 | 5 / 0 / 2 | 4 / 0 / 3 |
| Fan-in / fan-out, loops, CPU+GPU | 9 | 1 / 8 / 0 | 0 / 7 / 2 | 6 / 0 / 3 | 5 / 2 / 2 | 0 / 1 / 8 |
| f32 | 65 | 30 / 30 / 5 | 25 / 20 / 20 | 41 / 8 / 16 | 38 / 14 / 13 | 17 / 5 / 43 |
| TF32 | 13 | 0 / 3 / 10 | 0 / 0 / 13 | 0 / 0 / 13 | 0 / 1 / 12 | 0 / 0 / 13 |
| bf16 weights, f32 activations | 9 | 1 / 1 / 7 | 2 / 1 / 6 | 2 / 0 / 7 | 0 / 1 / 8 | 3 / 0 / 6 |
| bf16 weights and KV cache | 7 | 3 / 3 / 1 | 3 / 2 / 2 | 4 / 1 / 2 | 1 / 1 / 5 | 5 / 0 / 2 |
| bf16 storage, bf16 | f32 products | 25 | 12 / 6 / 7 | 9 / 7 / 9 | 10 / 2 / 13 | 10 / 7 / 8 | 5 / 3 / 17 |

**Every program**: Mapal's time ÷ the fastest same-precision competitor's, per GPU. Bold: faster; `≈`: level (within the row's measured variation or 5%); plain: slower. Leaders and times per row: [`matrix.md`](../../benches/results/evidence/matrix.md).

| program | family | precision | RTX 5080 | RTX 3090 | A100 | RTX 4090 | H100 |
|---|---|---|---:|---:|---:|---:|---:|
| `attn_1024_64` | attn | f32 | 0.97≈ | **0.74** | **0.82** | **0.61** | 1.17 |
| `attn_2048_128` | attn | f32 | 0.99≈ | 1.06 | 1.38 | **0.85** | 0.96≈ |
| `attn_2048_64` | attn | f32 | **0.78** | **0.61** | 1.88 | **0.64** | **0.85** |
| `causal_1024_256` | attn | f32 | **0.30** | **0.15** | **0.32** | **0.35** | **0.23** |
| `decode_1024_256` | attn | f32 | **0.53** | **0.67** | **0.84** | **0.88** | 1.08 |
| `gen_1024_512_256` | attn | f32 | **0.30** | **0.16** | **0.28** | **0.35** | **0.28** |
| `lsm_1_131072` | attn | f32 | **0.39** | **0.28** | **0.31** | **0.50** | **0.83** |
| `lsm_32_8192` | attn | f32 | **0.84** | **0.48** | **0.43** | **0.83** | 1.31 |
| `lsm_4_32768` | attn | f32 | **0.57** | **0.41** | **0.48** | **0.55** | 1.34 |
| `prefill_1024_512_256` | attn | f32 | **0.30** | **0.15** | **0.28** | **0.36** | **0.27** |
| `rowsum_2048` | attn | f32 | 0.97≈ | 1.08 | **0.81** | **0.92** | 1.13 |
| `rowsum_4096` | attn | f32 | 1.03≈ | 1.11 | **0.86** | 0.97≈ | 1.09 |
| `softmax_rows_2048` | attn | f32 | **0.95** | 1.09 | **0.87** | **0.91** | 1.10 |
| `softmax_rows_4096` | attn | f32 | 1.03≈ | 1.12 | **0.87** | 0.98≈ | 1.08 |
| `conv_2048_5` | conv | f32 | **0.43** | **0.55** | **0.59** | **0.52** | **0.49** |
| `convmc_8_64_56_3` | conv | f32 | **0.67** | 1.45 | 2.17 | 1.51 | 1.37 |
| `convmc_8_64_56_3_tf32` | conv | tf32 | 1.12 | 1.39 | 2.98 | 1.89 | 2.24 |
| `ew_4096` | conv | f32 | **0.35** | **0.58** | 1.07≈ | **0.33** | 1.35 |
| `ewconvew_2048_5` | conv | f32 | **0.37** | **0.41** | **0.54** | **0.40** | **0.48** |
| `fir_4194304_16` | conv | f32 | **0.42** | **0.57** | **0.60** | **0.51** | **0.52** |
| `firew_rowsum_2048x2048_16` | conv | f32 | **0.40** | **0.53** | **0.51** | **0.46** | **0.43** |
| `diamond_2048` | fan | f32 | 0.99≈ | 1.03≈ | **0.87** | 0.96≈ | 1.12 |
| `fanout_2048` | fan | f32 | 0.99≈ | 1.03≈ | **0.87** | **0.93** | 1.11 |
| `indep_2048` | fan | f32 | 1.01≈ | 1.04≈ | **0.88** | **0.94** | 1.13 |
| `indep_512` | fan | f32 | 0.97≈ | 1.14 | 1.17 | **0.78** | 1.20 |
| `loop_4096` | fan | f32 | 0.96≈ | 0.95≈ | **0.86** | 1.00≈ | 1.09 |
| `loopmm_1024_8` | fan | f32 | 1.00≈ | 1.09 | 1.11 | 1.13 | 1.10 |
| `rt_mlp2_1024` | fan | f32 | 0.99≈ | 1.00≈ | 1.46 | 1.09 | 1.03≈ |
| `rt_mlp2_2048` | fan | f32 | **0.93** | 0.96≈ | **0.87** | **0.92** | 1.09 |
| `twin_2048` | fan | f32 | 0.97≈ | 1.05≈ | **0.86** | **0.91** | 1.11 |
| `epi_4096` | gemm | f32 | 0.99≈ | **0.91** | **0.79** | 1.00≈ | 1.08 |
| `mm_16x2048x7168` | gemm | f32 | 0.96≈ | 1.00≈ | 0.97≈ | 1.13 | 1.73 |
| `pj_m128_n28672_k4096` | gemm | f32 | 1.07 | **0.87** | **0.86** | **0.95** | 1.04≈ |
| `pj_m128_n28672_k4096_bf16` | gemm | bf16 | 1.02≈ | 1.02≈ | 1.40 | **0.95** | 1.37 |
| `pj_m128_n28672_k4096_tf32` | gemm | tf32 | 1.17 | 1.13 | 1.38 | 0.99≈ | 1.12 |
| `pj_m128_n4096_k14336` | gemm | f32 | 1.09 | 1.02≈ | 1.25 | **0.93** | 1.15 |
| `pj_m128_n4096_k14336_bf16` | gemm | bf16 | 1.24 | 0.96≈ | 2.05 | 1.06 | 2.52 |
| `pj_m128_n4096_k14336_tf32` | gemm | tf32 | 1.30 | 1.09 | 1.67 | 1.15 | 1.83 |
| `pj_m128_n4096_k4096` | gemm | f32 | 0.99≈ | 1.16 | 1.27 | 1.16 | 1.37 |
| `pj_m128_n4096_k4096_bf16` | gemm | bf16 | 1.04≈ | 1.19 | 1.41 | 0.98≈ | 2.13 |
| `pj_m128_n4096_k4096_tf32` | gemm | tf32 | 1.48 | 1.27 | 1.98 | 1.45 | 1.80 |
| `pj_m128_n6144_k4096` | gemm | f32 | 1.07 | 1.12 | 1.66 | 1.12 | 1.24 |
| `pj_m128_n6144_k4096_bf16` | gemm | bf16 | **0.95** | 0.96≈ | 1.47 | 1.06 | 1.91 |
| `pj_m128_n6144_k4096_tf32` | gemm | tf32 | 1.38 | 1.20 | 1.95 | 1.57 | 1.82 |
| `pj_m16_n18432_k7168` | gemm | f32 | 1.01≈ | 1.00≈ | 0.97≈ | 1.02≈ | **0.89** |
| `pj_m16_n2048_k7168` | gemm | f32 | **0.67** | 1.14 | 1.01≈ | 1.42 | 1.21 |
| `pj_m16_n28672_k4096` | gemm | f32 | 1.02≈ | 1.05≈ | 0.98≈ | 1.03≈ | **0.88** |
| `pj_m16_n28672_k4096_bf16` | gemm | bf16 | **0.78** | **0.93** | **0.91** | **0.67** | 1.42 |
| `pj_m16_n4096_k14336` | gemm | f32 | **0.80** | 1.00≈ | 0.99≈ | **0.75** | 1.04≈ |
| `pj_m16_n4096_k14336_bf16` | gemm | bf16 | **0.55** | 1.00≈ | 1.01≈ | **0.43** | 1.51 |
| `pj_m16_n4096_k4096` | gemm | f32 | **0.66** | 1.08 | **0.94** | 1.23 | 1.16 |
| `pj_m16_n4096_k4096_bf16` | gemm | bf16 | **0.61** | 1.16 | **0.80** | **0.58** | 1.86 |
| `pj_m16_n6144_k4096` | gemm | f32 | 0.96≈ | 1.10 | **0.93** | **0.83** | 0.96≈ |
| `pj_m16_n6144_k4096_bf16` | gemm | bf16 | **0.75** | 0.95≈ | **0.90** | **0.64** | 1.49 |
| `pj_m1_n18432_k7168` | gemm | f32 | 0.95≈ | 0.99≈ | 1.06 | 1.15 | 1.06 |
| `pj_m1_n2048_k7168` | gemm | f32 | **0.80** | 1.09 | 1.08 | 1.21 | 1.21 |
| `pj_m1_n28672_k4096` | gemm | f32 | 1.03≈ | 1.01≈ | 1.04≈ | 1.15 | 1.09 |
| `pj_m1_n28672_k4096_bf16` | gemm | bf16 | **0.73** | **0.90** | 1.14 | **0.77** | 1.13 |
| `pj_m1_n4096_k14336` | gemm | f32 | 0.96≈ | 1.01≈ | 1.15 | 1.15 | 1.27 |
| `pj_m1_n4096_k14336_bf16` | gemm | bf16 | **0.53** | **0.91** | 1.03≈ | **0.46** | 1.10 |
| `pj_m1_n4096_k4096` | gemm | f32 | 1.26 | 1.06 | **0.92** | 1.16 | 1.44 |
| `pj_m1_n4096_k4096_bf16` | gemm | bf16 | **0.31** | **0.93** | **0.78** | **0.26** | 0.92≈ |
| `pj_m1_n6144_k4096` | gemm | f32 | 1.28 | 1.04≈ | 1.23 | 1.02≈ | 1.26 |
| `pj_m1_n6144_k4096_bf16` | gemm | bf16 | **0.35** | **0.88** | **0.94** | **0.35** | 1.02≈ |
| `pj_m2048_n28672_k4096` | gemm | f32 | 0.97≈ | 1.02≈ | **0.84** | 1.04≈ | 1.10 |
| `pj_m2048_n28672_k4096_bf16` | gemm | bf16 | 1.00≈ | 1.05≈ | 1.43 | 1.03≈ | 2.43 |
| `pj_m2048_n28672_k4096_tf32` | gemm | tf32 | 1.06 | 1.11 | 1.39 | 1.09 | 1.50 |
| `pj_m2048_n4096_k14336` | gemm | f32 | 1.00≈ | 1.16 | **0.87** | 0.95≈ | 1.12 |
| `pj_m2048_n4096_k14336_bf16` | gemm | bf16 | 1.07 | 1.07 | 1.66 | 1.03≈ | 2.48 |
| `pj_m2048_n4096_k14336_tf32` | gemm | tf32 | 1.02≈ | 1.14 | 1.53 | 1.11 | 1.62 |
| `pj_m2048_n4096_k4096` | gemm | f32 | 1.02≈ | 0.98≈ | **0.87** | 0.97≈ | 1.12 |
| `pj_m2048_n4096_k4096_bf16` | gemm | bf16 | 1.10 | 1.36 | 1.64 | 1.03≈ | 2.25 |
| `pj_m2048_n4096_k4096_tf32` | gemm | tf32 | 1.14 | 1.09 | 1.46 | 1.11 | 1.67 |
| `pj_m2048_n6144_k4096` | gemm | f32 | 1.00≈ | 1.12 | **0.80** | 0.96≈ | 1.10 |
| `pj_m2048_n6144_k4096_bf16` | gemm | bf16 | 1.03≈ | 1.06 | 1.48 | 1.03≈ | 2.26 |
| `pj_m2048_n6144_k4096_tf32` | gemm | tf32 | 1.14 | 1.13 | 1.61 | 1.10 | 1.62 |
| `llm_blkdec_b1_l4096` | llm | f32 | 0.95≈ | **0.92** | 1.15 | 0.99≈ | 1.05 |
| `llm_blkdec_b1_l4096_bf16` | llm | bf16 | 0.97≈ | **0.90** | 1.22 | 1.03≈ | **0.74** |
| `llm_blkdec_b1_l4096_bf16kv` | llm | bf16kv | 1.05≈ | 0.97≈ | 1.63 | 1.09 | 1.47 |
| `llm_blkdec_b1_l4096_bf16w` | llm | bf16w | 1.08 | 0.99≈ | 1.68 | 1.13 | 1.48 |
| `llm_blkpre_t2048` | llm | f32 | **0.94** | 0.99≈ | **0.68** | **0.92** | 1.10 |
| `llm_blkpre_t2048_bf16` | llm | bf16 | 1.17 | 1.15 | 1.67 | 1.24 | 2.76 |
| `llm_blkpre_t2048_bf16w` | llm | bf16w | 3.06 | 2.95 | 4.89 | 2.50 | 7.11 |
| `llm_blkpre_t2048_tf32` | llm | tf32 | 1.09 | 1.11 | 1.61 | 1.15 | 1.60 |
| `llm_dec_b1_l2048` | llm | f32 | **0.84** | **0.75** | **0.69** | **0.87** | **0.50** |
| `llm_dec_b1_l2048_bf16` | llm | bf16 | **0.82** | **0.70** | **0.63** | **0.91** | **0.47** |
| `llm_dec_b1_l2048_bf16kv` | llm | bf16kv | **0.82** | **0.68** | **0.64** | **0.92** | **0.50** |
| `llm_dec_b1_l2048_bf16w` | llm | bf16w | **0.88** | **0.73** | **0.67** | 1.00≈ | **0.47** |
| `llm_dec_b1_l8192` | llm | f32 | **0.81** | **0.69** | **0.55** | **0.90** | **0.41** |
| `llm_dec_b1_l8192_bf16` | llm | bf16 | **0.87** | **0.68** | **0.52** | 0.97≈ | **0.67** |
| `llm_dec_b1_l8192_bf16kv` | llm | bf16kv | **0.88** | **0.67** | **0.53** | 0.98≈ | **0.71** |
| `llm_dec_b1_l8192_bf16w` | llm | bf16w | 1.03≈ | **0.80** | **0.62** | 1.19 | **0.75** |
| `llm_dec_b32_l2048` | llm | f32 | **0.37** | **0.37** | **0.37** | **0.36** | **0.27** |
| `llm_dec_b32_l2048_bf16` | llm | bf16 | 1.09 | **0.89** | **0.87** | 1.17 | **0.84** |
| `llm_dec_b32_l2048_bf16kv` | llm | bf16kv | 1.05≈ | **0.91** | 0.97≈ | 1.17 | **0.88** |
| `llm_dec_b32_l2048_bf16w` | llm | bf16w | 1.64 | 1.37 | 1.33 | 1.79 | 1.10 |
| `llm_dec_b32_l8192` | llm | f32 | **0.29** | **0.29** | **0.29** | **0.29** | **0.23** |
| `llm_dec_b32_l8192_bf16` | llm | bf16 | 1.11 | 1.15 | 1.23 | 1.29 | 1.33 |
| `llm_dec_b32_l8192_bf16kv` | llm | bf16kv | 1.10 | 1.16 | 1.30 | 1.29 | 1.36 |
| `llm_dec_b32_l8192_bf16w` | llm | bf16w | 1.80 | 1.86 | 1.70 | 1.99 | 1.87 |
| `llm_dec_b8_l2048` | llm | f32 | **0.54** | **0.89** | **0.65** | **0.53** | **0.39** |
| `llm_dec_b8_l2048_bf16` | llm | bf16 | **0.92** | 1.27 | **0.88** | 1.08 | **0.88** |
| `llm_dec_b8_l2048_bf16kv` | llm | bf16kv | **0.93** | 1.27 | **0.92** | 1.08 | **0.81** |
| `llm_dec_b8_l2048_bf16w` | llm | bf16w | 1.25 | 1.51 | 1.07 | 1.41 | **0.90** |
| `llm_dec_b8_l8192` | llm | f32 | **0.34** | **0.45** | **0.36** | **0.33** | **0.29** |
| `llm_dec_b8_l8192_bf16` | llm | bf16 | 1.00≈ | 1.05≈ | **0.91** | 1.20 | 0.96≈ |
| `llm_dec_b8_l8192_bf16kv` | llm | bf16kv | 1.00≈ | 1.05≈ | **0.87** | 1.20 | **0.91** |
| `llm_dec_b8_l8192_bf16w` | llm | bf16w | 1.53 | 1.46 | 1.08 | 1.75 | 1.13 |
| `llm_pre_t2048` | llm | f32 | **0.89** | **0.95** | 1.02≈ | **0.86** | 1.14 |
| `llm_pre_t2048_bf16` | llm | bf16 | 1.33 | 1.38 | 2.12 | 1.25 | 2.92 |
| `llm_pre_t2048_bf16w` | llm | bf16w | 3.23 | 3.23 | 6.72 | 2.81 | 7.90 |
| `llm_pre_t2048_tf32` | llm | tf32 | 1.17 | 1.14 | 2.30 | 1.19 | 1.64 |
| `llm_pre_t8192` | llm | f32 | **0.84** | **0.84** | 1.46 | 1.03≈ | 1.56 |
| `mlp2_2048` | mlp | f32 | 1.02≈ | 1.08 | **0.91** | **0.95** | 1.12 |
| `mlp2_2048_tf32` | mlp | tf32 | 1.01≈ | 1.10 | 1.41 | 1.09 | 1.53 |
| `mlp_2048` | mlp | f32 | 0.95≈ | 1.03≈ | **0.87** | **0.94** | 1.11 |
| `mlp_2048_tf32` | mlp | tf32 | 1.02≈ | 1.07 | 1.62 | 1.07 | 1.64 |
| `mlp_512` | mlp | f32 | **0.85** | 1.19 | 1.09 | **0.82** | 1.21 |
| `smlp_4096_256_1024` | mlp | f32 | 1.00≈ | 1.10 | **0.78** | **0.84** | 1.19 |

<!-- results-programs:end -->

Every row, its legs and its leader: [`matrix.md`](../../benches/results/evidence/matrix.md). The rented GPUs, their
boxes and rounds: [`cross-arch`](../../benches/results/evidence/cross-arch/README.md). Earlier results (CPU against
NumPy and OpenBLAS, the October four-GPU table, plan-cost, tile efficiency):
[`docs/performance/earlier-readme-results.md`](earlier-readme-results.md). Method and history:
[`docs/performance/`](README.md).

The charts and the table above are generated. To refresh them after a run, update
[`matrix.json`](../../benches/results/evidence/matrix.json) (`benches/local/matrix.py`) and
[`full-model.json`](../../benches/results/evidence/full-model.json), then run
`python3 benches/results/charts/make_charts.py`: it redraws the charts and rewrites the tables between the
`results-headline` markers in this file.

## CPU against careful C++ and NumPy

AMD Threadripper 9970X (32 cores / 64 threads, 4.5 GHz cap), clang 23.1.5, 2026-11-28. Medians of 31 interleaved runs
(21 for the multi-threaded NumPy table), one process per run. Each cell is competitor time ÷ Mapal time (above 1: Mapal
faster) with its 95 % bootstrap interval; a side wins only when the interval excludes 1 and Mann-Whitney p < 0.05. The
C++ is written carefully (register-blocked tiles, blocked transpose, a thread pool started before the timer) and built
strict and with -ffast-math; the faster build is used. NumPy is the official wheel with OpenBLAS 0.3.34.

| workload | one thread vs C++ | one thread vs NumPy | whole machine vs C++ |
| --- | --- | --- | --- |
| FIR filter, 1M | 0.95× [0.94, 0.95] C++ faster | 6.23× [6.16, 6.24] Mapal faster | 0.94× [0.84, 1.07] no clear difference |
| conv2d 3x3, 1024² | 1.52× [1.50, 1.54] Mapal faster | 23.76× [23.53, 24.06] Mapal faster | 0.94× [0.88, 1.07] no clear difference |
| saxpy, 1M | 0.96× [0.93, 0.99] C++ faster | 2.09× [2.02, 2.14] Mapal faster | 0.95× [0.93, 0.99] C++ faster |
| sum reduction, 1M | 1.01× [0.99, 1.01] no clear difference | 4.31× [4.29, 4.35] Mapal faster | 0.44× [0.40, 0.46] C++ faster |
| transpose, 1024² | 0.80× [0.79, 0.81] C++ faster | 5.21× [4.69, 5.37] Mapal faster | 0.92× [0.82, 0.98] C++ faster |
| gather x[idx[i]], 1M | 0.81× [0.81, 0.82] C++ faster | 1.42× [1.38, 1.44] Mapal faster | 1.26× [1.17, 1.35] Mapal faster |

Whole machine against multi-threaded NumPy, each side at its better of 32 / 64 threads:

| workload | NumPy ÷ Mapal |
| --- | --- |
| matrix multiply 1024² | 1.09× [1.04, 1.16] Mapal faster |
| matrix multiply 2048² | 1.57× [1.48, 1.61] Mapal faster |
| matrix multiply 4096² | 0.62× [0.60, 0.64] NumPy faster |
| 2-layer network, width 512 | 1.25× [1.13, 1.37] Mapal faster |
| 4-layer network, width 512 | 1.33× [1.24, 1.46] Mapal faster |
| 2-layer network, width 1024 | 1.01× [0.98, 1.04] no clear difference |
| 4-layer network, width 1024 | 1.02× [0.99, 1.10] no clear difference |
| 2-layer network, width 2048 | 1.09× [1.04, 1.46] Mapal faster |
| 4-layer network, width 2048 | 1.27× [1.18, 1.33] Mapal faster |

Mapal's 64-thread sum at 1M elements is slower than its own one-thread run: an open problem. Tables, samples and
commands: [`shapes-fair.md`](../../benches/results/evidence/local-rtx-5080/cpu/shapes-fair.md),
[`numpy-mt.md`](../../benches/results/evidence/local-rtx-5080/cpu/numpy-mt.md) (data in the `.json` files beside them).
