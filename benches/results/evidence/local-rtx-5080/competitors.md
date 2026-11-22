# competitors — "best competitor" = the fastest real-world implementation (2026-11-23)

Branch `local-rtx-5080` @ f7a74cdb (Mapal unchanged; this step adds competitor legs and the matrix's every-leg view).
RTX 5080 (sm_120, 84 SMs, 64 MiB L2, 16 GB), driver 620.73.04, torch 2.11.0+cu130 (Triton 3.6.0, cuDNN 9.19), CUDA
13.5 toolchain. Timed runs under the bench protocol (lock, CPUs 0-7 idle and pinned, no game, Tctl 33–58 C): the
suite + long-rows bundle (41 programs + 2 cells, 945 s) and the model-scale bundle (43 programs: f32 / `_tf32` /
`_bf16w` layers and projections, 1147 s), both built at HEAD, every leg in the same session as Mapal. Matrix:
[`../matrix.md`](../matrix.md) (sections "What the new legs change" and "Every competitor leg").

## Legs added

| family | legs (precision) | rows | implementation | install: local / rented box |
| --- | --- | --- | --- | --- |
| torch + CUDA graphs | `compile-cg` (`torch.compile` mode `max-autotune`), `compile-ro` (`reduce-overhead`); `-tf32`, `bf16-` twins; `sdpa-compile-cg` | every torch row but the kv twins | every tensor the function reads (weights, KV cache, the decode loop's caches) a static address (`mark_static_address`, as nn.Module parameters / buffers are); a decode step marks the step begin and clones its output (gpt-fast's loop); the layer's input activations stay graph inputs (copied per replay); a leg prints `cg=on` or `cg=skipped(n)` (not counted) | ships with torch |
| SDPA backends | `sdpa-efficient` (f32), `bf16-efficient`, `bf16-cudnn` | LLM layers | `sdpa_kernel([backend])`; the efficient kernel takes no GQA: K / V repeated per query head (HF `repeat_kv`) | ships with torch (cuDNN 9.19) |
| hand Triton | `triton` (f32: `ieee` dots), `triton-tf32`, `triton-bf16` | GEMM cells, projections, epi / MLP chains (the tutorial's fused-activation slot carries the epilogue), softmax rows / long softmax, attention rows, LLM prefill attention | Triton v3.6.0 tutorials 02 / 03 / 06 (forward), autotuned by their own config lists — `benches/gpu/triton_kernels.py` lists each change (dtype, f32 precision, GQA, epilogues, softmax scale) | ships with torch |
| Liger | `liger`, `liger-tf32`, `liger-bf16` | LLM layers | Liger 0.8.5 RMSNorm, RoPE (HF cos / sin), SwiGLU; cuBLAS projections; SDPA (f32: efficient, bf16: flash) | `liger-kernel==0.8.5` (pip, with FlashInfer: 9.7 s locally) / box: pip in the torch install job |
| FlashAttention-2 | `fa2-bf16` | LLM layers (bf16) | flash-attn 2.8.3.post2: `flash_attn_func` (prefill, causal, GQA), `flash_attn_with_kvcache` (decode: appends the step's K / V, split-KV) | no wheel for torch 2.11 + CUDA 13: source build, `FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=6 NVCC_THREADS=2`, pinned: **742 s** / box: `comp_build.sh` (arch family 80 / 90 / 100 / 120, jobs = min(cores / 2, GB / 6)), capped `COMP_BUILD_MIN` 25 |
| FlashAttention-3 | `fa3-bf16` | LLM layers (bf16), sm_90a | the same tag's `hopper/` build: forward, head dim 128, bf16, split-KV, append-KV, packed GQA | box-only (H100): `comp_build.sh`; FA3's setup.py fetches NVIDIA's nvcc 12.8 (its host compiler is not this machine's gcc 16) — not built here |
| FlashInfer | `flashinfer-bf16` | LLM layers (bf16) | 0.7.5.post1: `BatchDecodeWithPagedKVCacheWrapper` (tensor-core kernel: GQA group 4, FlashInfer's guidance; each sequence's cache one page, planned per step from host tensors), `single_prefill_with_kv_cache` (causal) | `flashinfer-python` + prebuilt `flashinfer-jit-cache-sm120f` (flashinfer.ai cu130 index, 454 MB, 2.3 s) / box: the part's jit-cache package (sm80 / sm89 / sm90a / sm100a / sm120f) |
| TensorRT-LLM | `trtllm-bf16` | — | **not installable on this stack**: every release pins its own torch — 1.0.0 torch 2.7.1–2.8.0a0 (cuda-python < 13), 1.1.0 torch 2.9.0, 1.2.0 torch 2.9.1–2.10.0a0, 1.3.0 torch 2.14.0 (+ cuDNN 9.24, cuBLAS 13.1.1); with torch 2.11.0 pinned the resolver falls back to the 0.17.0.post1 stub (pypi.nvidia.com, CUDA 12). Installing 1.3.0 replaces the venv's torch, i.e. every other torch leg's libraries. Its standalone kernels (XQA / FMHA, the gemm plugin over cuBLASLt) build only with the whole C++ tree (TensorRT, NCCL): not built. `comp_bench.py` reports it unavailable | — |

Run: `comp_bench.py` (new; same `workload` / `llm_workload` tensors, `gate`, `timed` as torch_bench.py — 5 warm-ups,
min / median / max of 30, 256 MB L2 flush before each), the CUDA-graph and SDPA legs in `torch_bench.py`. Box flow:
`box_run.sh` installs the pip legs beside torch, builds FA2 / FA3 after every timed Mapal and torch section (nothing
timed runs beside a build), then runs `comp_bench.py` under `COMP_MIN` (default 20 min); `COMP=0` / `COMP_BUILD=0`
switch it off; `vast_run.sh` passes `COMP*` through. The rented parts keep their 60f0f8ce data: their new families
read `not run` in the matrix until a rented round runs these legs.

## Value gates (against Mapal's printed values, torch_bench's tolerances)

| legs | within / outside / failed | note |
| --- | --- | --- |
| `fa2-bf16`, `flashinfer-bf16`, `liger-bf16`, `torch-bf16-{cudnn,efficient,compile-cg,compile-ro}` | 17/0/0 each (37/0/0 for the bf16 compile twins) | bf16 throughout vs Mapal's f32 / bf16w values: 5e-2; max 1.7e-2 |
| `triton` | 36/0/2 | max 9.3e-6; failed: `lsm_4_32768`, `lsm_1_131072` — the tutorial holds a row in one program: 128 / 512 KiB of shared memory > 99 KiB |
| `triton-bf16` / `triton-tf32` | 23/0/0 / 10/2/0 | tf32 outside: `mlp_2048_tf32`, `mlp2_2048_tf32` at 1e-3 (3.7e-3 / 4.6e-3 — the torch `compile-tf32` legs' same miss; tf32 model-scale rows gate at 1e-2) |
| `liger` / `liger-tf32` | 9/0/1 / 2/0/0 | failed: `llm_dec_b32_l8192` — f32 K / V repeated per head is 4 GiB each (OOM); max 4.1e-7 |
| `torch-compile-cg` / `-ro` (+ `-tf32`, `sdpa-compile-cg`, `sdpa-efficient`) | 72/2/0 / 72/2/0 (…) | the outside ones are the f32 legs of `_tf32` rows, as `torch-compile`'s; every graph leg `cg=on` (0 skipped) |

## RTX 5080: the old best vs the new best

Old best = the fastest leg among those before this step (torch eager / compile / sdpa / bf16, gemm_lib, chain_bench,
cells' cuBLAS / CUTLASS), re-timed in the same session. 58 of 86 rows keep their best leg. The 28 that change:

| workload | prec | Mapal ms | old best (leg) Mapal/best | new best (leg) Mapal/best | verdict |
| --- | --- | ---: | --- | --- | --- |
| epi_4096 | f32 | 3.675 | 3.558 (compile) 1.03 | 3.547 (compile-ro) 1.04 | even |
| pj_m2048_n4096_k4096 | f32 | 1.837 | 1.776 (eager) 1.03 | 1.772 (compile-cg) 1.04 | even |
| pj_m2048_n6144_k4096 | f32 | 2.648 | 2.605 (eager) 1.02 | 2.602 (compile-cg) 1.02 | even |
| mlp2_2048_tf32 | tf32 | 1.668 | 1.546 (eager-tf32) 1.08 | 1.514 (compile-ro-tf32) 1.10 | loss |
| mlp_2048_tf32 | tf32 | 0.8298 | 0.769 (eager-tf32) 1.08 | 0.7598 (compile-ro-tf32) 1.09 | loss |
| attn_1024_64 | f32 | 0.03914 | 0.0297 (compile) 1.32 | 0.0287 (compile-cg) 1.36 | loss |
| attn_2048_128 | f32 | 0.1177 | 0.0953 (sdpa-compile) 1.24 | 0.0942 (sdpa-compile-cg) 1.25 | loss |
| attn_2048_64 | f32 | 0.07953 | 0.0748 (sdpa-compile) 1.06 | 0.0696 (compile-ro) 1.14 | loss |
| causal_1024_256 | f32 | 15.23 | 44.52 (compile) 0.34 | 38.23 (compile-cg) 0.40 | win |
| gen_1024_512_256 | f32 | 7.685 | 22.53 (compile) 0.34 | 20.84 (compile-cg) 0.37 | win |
| lsm_1_131072 | f32 | 0.01582 | 0.0246 (compile) 0.64 | 0.0225 (compile-cg) 0.70 | win |
| prefill_1024_512_256 | f32 | 7.743 | 22.49 (compile) 0.34 | 19.26 (compile-cg) 0.40 | win |
| softmax_rows_4096 | f32 | 3.803 | 3.54 (compile) 1.07 | 3.529 (compile-ro) 1.08 | loss |
| llm_blkpre_t2048 | f32 | 27.28 | 28.09 (compile) 0.97 | 26.67 (liger) 1.02 | even |
| llm_dec_b1_l2048 | f32 | 0.24 | 0.2536 (compile) 0.95 | 0.2532 (compile-ro) 0.95 | win |
| llm_dec_b8_l8192 | f32 | 1.067 | 2.488 (compile) 0.43 | 2.468 (compile-cg) 0.43 | win |
| llm_pre_t2048 | f32 | 8.56 | 8.367 (compile) 1.02 | 7.2 (liger) 1.19 | even → **loss** |
| llm_pre_t8192 | f32 | 80.68 | 84.52 (compile) 0.95 | 52.83 (liger) 1.53 | even → **loss** |
| llm_blkpre_t2048_tf32 | tf32 | 21.83 | 18.56 (compile-tf32) 1.18 | 18.08 (triton-tf32) 1.21 | loss |
| llm_pre_t2048_tf32 | tf32 | 7.455 | 6.062 (compile-tf32) 1.23 | 4.537 (triton-tf32) 1.64 | loss |
| llm_dec_b1_l8192_bf16w | bf16w | 0.2405 | 0.1913 (bf16-compile) 1.26 | 0.1883 (liger-bf16) 1.28 | loss |
| llm_dec_b32_l8192_bf16w | bf16w | 3.426 | 1.49 (bf16-compile) 2.30 | 1.407 (flashinfer-bf16) 2.43 | loss |
| llm_dec_b8_l8192_bf16w | bf16w | 0.9781 | 0.485 (bf16-compile) 2.02 | 0.4726 (bf16-cudnn) 2.07 | loss |
| conv_2048_5 | f32 | 0.06647 | 0.1536 (eager) 0.43 | 0.1516 (compile-cg) 0.44 | win |
| fir_4194304_16 | f32 | 0.04431 | 0.1044 (eager) 0.42 | 0.1024 (compile-cg) 0.43 | win |
| firew_rowsum_2048x2048_16 | f32 | 0.05244 | 0.1106 (compile) 0.47 | 0.1085 (compile-ro) 0.48 | win |
| loopmm_1024_8 | f32 | 0.649 | 0.5898 (compile) 1.10 | 0.5878 (compile-cg) 1.10 | loss |
| rt_mlp2_2048 | f32 | 16.19 | 15.55 (compile) 1.04 | 15.53 (compile-cg) 1.04 | even |

Geomean Mapal / best over the 86 rows: 0.980 against the old legs → **1.001** against every leg; win / even / loss
18 / 24 / 44 → 18 / 22 / 46. By family (old → new): attention / softmax 0.806 → 0.841, LLM 1.016 → 1.077, conv /
FIR 0.521 → 0.525, MLP 1.091 → 1.097, GEMM 1.156 → 1.156, fan / loops 1.061 → 1.061. New best legs by family:
CUDA graphs 20 (`-cg` 13, `-ro` 7), Liger 4, hand Triton 2, FlashInfer 1, SDPA cuDNN 1.

**Mapal wins that disappear: none** (every win keeps its verdict; the decode loops stay 2.5–2.7× ahead). Two even rows
become losses — `llm_pre_t2048` (1.02 → 1.19) and `llm_pre_t8192` (0.95 → 1.53) — and the TF32 prefill's loss deepens
(1.23 → 1.64), all from fused attention kernels (below). The rented parts are unchanged until their round runs the legs.

## Why (same-box ABBA, CUPTI)

ABBA (`comp_abba.sh`: old best, new best, new best, old best; one process each) confirms every move that changes a
ratio by > 2 %: new / old = 0.625 (`llm_pre_t8192`, compile → liger), 0.855 (`llm_pre_t2048`), 0.751
(`llm_pre_t2048_tf32`, compile-tf32 → triton-tf32), 0.950 (`llm_blkpre_t2048`), 0.953 (`llm_dec_b32_l8192_bf16w`,
→ FlashInfer), 0.972 (`llm_dec_b8_l8192_bf16w`, → cuDNN), 0.885 (`causal_1024_256`, → compile-cg), 0.945
(`attn_2048_64`, → compile-ro). One does not hold: `llm_dec_b1_l8192_bf16w` liger-bf16 / bf16-compile = 1.057 in ABBA
(0.98 in the run) — the two are within each other's noise there; the row is a loss either way. The < 2 % moves
(epi, pj_m2048, softmax_rows_4096, fir, conv, …) change no verdict and were not re-timed.

| row | new best's kernels (CUPTI, per call) | Mapal / the old best | the difference |
| --- | --- | --- | --- |
| llm_pre_t8192 f32 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80` 32.6 ms (128 × 32 blocks, 250 regs, 84 KiB smem) + SIMT sgemm 2 × 8.3 ms | Mapal: per KV-head group QKᵀ 2.19 + softmax flow 2.67 + P·V 2.92 ms = 7.8 ms × 8 = 62.2 ms attention, projections 10.3 + 7.3 ms | the memory-efficient kernel streams key blocks with the scores on chip and skips the causally masked blocks; Mapal's scores and `pe` cross DRAM (~4 GiB a group) and its QKᵀ computes every masked cell — long-prefill.md open 1 (flash-style streaming) and 2 (causal block skip), now the deciding gap |
| llm_pre_t2048 f32 | the same fmha kernel 2.26 ms + sgemm 2 × 2.13 ms | Mapal: QKᵀ 1.32 + softmax 1.30 + P·V 1.17 = 3.79 ms; projections 2.58 + 1.93 ms | attention 3.79 vs 2.26 ms (same cause), projections 4.51 vs 4.26 ms |
| llm_pre_t2048_tf32 | Triton `_attn_fwd` (tf32 mma, 32 × 32 blocks, 255 regs, 41 KiB smem) ≈ 1.2 ms + cuBLAS tf32 tensor-op GEMMs 1.84 + 1.29 ms | Mapal tf32: QKᵀ 1.32 + softmax 1.30 + P·V 1.16 ms; projections 2.00 + 1.44 ms | the tutorial's flash loop (online softmax, causal stage split) vs the three-kernel attention |
| llm_dec_b32_l8192_bf16w | FlashInfer `BatchPrefillWithPagedKVCacheKernel` (its tensor-core decode: the GQA group as rows) 1.18 ms a step | Inductor (bf16-compile): two Triton reductions, QKᵀ 0.64 + softmax·V 0.67 ms a step | one fused pass over the cache instead of two |
| causal_1024_256 | the same 5 Inductor kernels a step as `compile` (2.97 / 2.14 / 1.49 / 0.72 µs …) replayed as one graph | compile: device busy 17 of 44.5 ms a run | the graph removes the launches; the step stays CPU-bound (cudagraph-tree bookkeeping, the clone, the step index's H2D copy: 38 µs a step for 12.5 µs of kernels); Mapal runs the whole loop on the device (15 µs a step) |

`reduce-overhead` on the bf16 decode rows is 2–20× slower than `max-autotune` (e.g. b32 l8192: 29.8 vs 1.48 ms a
step): without max-autotune Inductor lowers the masked GQA attention by materializing the expanded cache
(`triton_poi_fused_…_clone_expand`, 0.87 ms a step at b8 l2048) and calls cuBLAS gemv on it; max-autotune fuses the
bmm into Triton reductions. torch's own behaviour, never a best leg. (A first pass had the layer's input activation
`x0` a static address: cudagraph trees re-recorded every step; fixed — activations stay inputs — and the LLM graph
legs re-timed, `resM2`.)

## Open

1. **Mapal's f32 / tf32 prefill attention** is now the largest LLM gap on this part (1.19–1.64×): the missing facts are
   long-prefill.md's open 1 (a tile operand read through an element law — `pe` dissolved — and the streamed
   row-block unit with the running `(den, O)` rescale) and 2 (a guard's affine form over a tile block, so a fully
   masked causal block skips its `k` loop).
2. TensorRT-LLM: not installable with torch 2.11 (above); a separate torch 2.14 venv would time it on another torch
   build than every other leg — not done.
3. FlashAttention-3 and the box install path (`comp_build.sh`, FlashInfer's per-part jit cache) run first on the next
   rented round; FA3's build time on the H100 box is unmeasured (capped at 25 min).
4. The Triton softmax tutorial fails on rows wider than ~25 K f32 (its one-program-per-row shared memory); the tf32
   Triton legs of the non-model-scale `_tf32` rows sit outside the 1e-3 gate like torch's `compile-tf32`.

Raw (local, not in git): `.claude/raw-evidence/local-rtx-5080/competitors/` — `resS`, `resM` (full timed runs),
`resM2` (the LLM graph legs re-timed), `abba.txt`, `trace/` (CUPTI summaries), `commands.txt`, `env.sh`, stems.
