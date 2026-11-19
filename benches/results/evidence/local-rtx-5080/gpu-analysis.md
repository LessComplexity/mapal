# local-rtx-5080 — why Mapal loses on the RTX 5080 (gpu-analysis, 2026-11-19)

Inputs: run 2 of the local suite (`gpu/gap-run2.md`), the bundle's PTX / plan lines / SASS / ptxas
(`evidence/`), and **CUPTI activity traces** of the timed runs (`benches/local/ktrace.cpp`, loaded with
`CUDA_INJECTION64_PATH`: every kernel, memset and driver call with device timestamps; no nsys / ncu on this box).
Competitor kernels are captured the same way from the actual launches (chain_bench, torch_bench, torch matmul).

## 1. The biggest cause is the runtime, not a kernel: the first site decision is inside the timed region

`mapal_gpu_site` calls `choice_init` (`crates/mapal-rt/gpu/mapal_gpu.c:874-885`: `cuDeviceGetName`,
two attributes, open + parse the wisdom file) lazily on a program's **first** site. A program runs one timed
iteration per process, so every program with a site table pays it between `t0` and `t1`. CUPTI driver-API
timeline of `pt_indep_512` (`iter ms` 0.105): sync at 0 us, `cuDeviceGetName` at 19 us, first `cuLaunchKernel`
at 83 us, two tile kernels 16.1 + 14.1 us, sync ends at 121 us. Kernels are 30 us of a 105 us timed region.

Experiment (scratch copy of `mapal_gpu.c`, not committed): `choice_init` called at the end of `load_module`
(outside every timed region), all 35 programs relinked, values compared, ABBA 2×6 processes each (`ab_rt.tsv`):

| program (site tables) | shipped runtime ms | wisdom read at load ms | best other | ratio shipped → at-load |
|---|---:|---:|---|---|
| indep_512 | 0.1040 | 0.0395 | 0.0366 streams | 2.84 → **1.08** |
| mlp_512 | 0.1076 | 0.0391 | 0.0410 cuBLASLt | 2.62 → **0.95** |
| attn_1024_64 | 0.0950 | 0.0430 | 0.0297 torch-compile | 3.20 → 1.45 |
| convmc_8_64_56_3_tf32 | 0.1422 | 0.0813 | 0.0645 torch-eager-tf32 | 2.21 → 1.26 |
| convmc_8_64_56_3 | 0.1494 | 0.0827 | 0.1075 torch-eager | 1.39 → **0.77** |
| pj_m16_n2048_k7168 | 0.0997 | 0.0441 | 0.0799 torch-eager | 1.25 → **0.55** ¹ |
| pj_m1_n2048_k7168 | 0.0950 | 0.0307 | 0.0758 torch-eager | 1.25 → **0.41** ¹ |
| smlp_4096_256_1024 | 0.2254 | 0.1651 | 0.1495 torch-compile | 1.51 → 1.10 |
| 2048 chains (mlp, mlp2, indep, diamond, twin, fanout), loopmm, rowsum_2048, pj_*_n18432 | — | −0.05…−0.09 ms each | — | 1.11–1.23 → 1.00–1.12 |

Every program with a site table saves 50–90 us. The programs without one (ew, fir, conv, ewconvew, firew, the
TF32 MLPs) do not move, and neither do the long looped programs beyond noise. Output values are the same: the
default face's split-K / atomic folds vary in the last bits run to run in the shipped build too. This is also most of the
"5090 → 5080 regression" on small programs: the README's 5090 column predates per-site-choice (no
`mapal_gpu_site` then); plan-cost E1 measured `pt_indep_512` at 55.5 us on the 4090 before it.

¹ **Boundary caveat**: torch_bench flushes L2 before every rep (a 128 MB `FillFunctor`, CUPTI), while a Mapal
program's inputs are generated on the device right before `t0`, so they sit in L2. The pj weights (58.7 MB) fit
the 5080's 64 MB L2: Mapal's pj_m1 kernel reads them at 2.55 TB/s (> 0.9 TB/s DRAM). Rows whose working set
fits L2 (pj_*_n2048, attn, *_512, convmc) compare an L2-warm Mapal with a cold-L2 competitor.

## 2. Kernel-level causes (Mapal kernels vs the competitor's captured kernels)

| workload | Mapal kernels (CUPTI us) | competitor kernels (CUPTI us) | cause (SASS / plan) |
|---|---|---|---|
| attn_1024_64 | QKᵀ 12.3 + softmax flow 7.4 + P·V 13.1 = 32.8 | cutlass simt sgemm 128x64 10.3 + triton fused softmax 5.3 + sgemm 64x64 split-K 8 9.1 + reduce 1.5 = 26.2 | QKᵀ is `face=SeparateMulAdd`: 512 FMUL + 512 FADD, **no FFMA** (`Arithmetic::scalar`, gpu.rs:357: a non-radix site fuses only when `b.clane == 1`; Kᵀ's strided lane loses FMA even on the order-free face). P·V: S=8 atomic split with **1-warp CTAs** (block 32, 1024 CTAs) + a zero pass; `fill-wave` intent mismatch (plan 13 blocks/SM, ptxas 16). Flow kernel: tree over 256 lanes through smem, 19 BAR, 0 SHFL |
| convmc_8_64_56_3_tf32 | FFMA implicit-GEMM tile 71.6 (163 regs, 392 CTAs × 64 thr, no HMMA) | cuDNN `cutlass_tensorop_s1688fprop_optimized` 44.5 (TF32 HMMA, 72 KB smem) + NCHW↔NHWC 2 × 4.6 | `tf32-radix` OFF by the rtx-5090 row (arch rule). Forced ON here (value-gated, K=576 within bound): its mma.sync kernel is **not faster** (83 vs 79 us traced): 16 HMMA but the digit-gathered image operand moves by 4-byte `cp.async` (16 `LDGSTS.E` vs 4 `.128`), 143 IMAD, 64 `STG.32`. cuDNN pays a layout pass to get 16-byte copies |
| 2048 MLP rows (mlp, mlp2, indep, diamond, twin, fanout) | stream-k tile (168 CTAs, 233 regs, tm8×tn16) 436–452; **epilogue-fused tile 497–540** | cuBLASLt `cutlass_80_simt_sgemm_256x128_8x4` split-K 3: 471.5 | Mapal's mainloop beats the library's; the loss is the epi=1 kernels: `combine=none`, 16×16 = 256 tiles on 84 SMs × 2 = 168 slots = **1.52 waves** (the tail wave half full). Epilogue operand reads are 64–128 `LDG.32` per thread (stores are `STG.128`) |
| loopmm_1024_8, smlp | 1024² tile 80 us × 8 (128 CTAs); smlp 69.9 + 83.8 (128 CTAs) | torch-compile ~74 us per matmul | 128 CTAs on 168 slots = **0.76 wave**: 40 of 84 SMs hold one block. No stream-k at 1024 |
| 4096³ cell | candidate 0 4.15 ms (stream-k, 256 thr, 128 regs); best candidate 3.78–3.81 | cuBLAS 3.55–3.72 | every Mapal tile is `stages=1` (register prefetch, no `cp.async`, bk 16, 16.6 KB smem); the cutlass simt 256x128_8x4 the library uses is a 4-stage `cp.async` pipeline, 48 KB smem |
| 16x2048x7168 cell | candidate 0 0.1126 ms: stream-k, **84 one-warp CTAs** (231 regs) = 521 GB/s | cuBLAS 0.0799 | too few bytes in flight (84 warps GPU-wide; Little's law at 573 ns × 0.89 TB/s needs ~510 KB). Its own candidates are faster: S=56 atomic 0.086, the exact face's 0.0737 (< cuBLAS). The programs' measured choice picks those; the cell times candidate 0 |
| decode_1024_256 | 6144 launches + 4096 memsets per run: Σ kernels 8.0–8.9 ms of 17–19 ms | hand CUDA `cache` 15.5 ms | launch/issue bound: per step 24 launches (2048 single-block cell kernels of 0.7–0.8 us) + 16 zero passes for S=32 / S=128 atomic naive GEMVs (one of them `SeparateMulAdd`, same clane rule) |
| softmax_rows_4096 | stream-k GEMM 3884 + flow 127 | cuBLAS GEMM + fused 3764 | the GEMM (stages=1) and the flow kernel above |

## 3. What the compiler decided, and what the 5080's facts change

- Rungs: the 5080 has no row of its own; `gpu_rungs_on`'s architecture rule applies rtx-5090's `GPU_RUNGS_OFF`
  rows (e.g. `tf32-radix` OFF, `kmajor-a` gated). The `tf32-radix` row is confirmed on this part (above).
- Sites: per-site-choice tables carry 2–8 candidates; the box tune picks per site (`wisdom-rtx-5080.tsv`). The
  rules' candidate 0 decides the cells and every untuned run.
- DeviceCost: **calibrated now** (`calib/`, profile.rs `RTX_5080`), same methods as rtx-5090:

| fact | 5080 | 5090 | method |
|---|---:|---:|---|
| issue_ns | 1690 | 2370 | E1 empty-kernel launch back to back, 1.69 us [1.68–1.75] |
| gap_ns | 260 | 300 | CUPTI device gap between dependent kernels (loopmm, rt_mlp2), median |
| zero_ns | 790 | 739 | CUPTI memset 1 MB 1024 ns − 1 MB at the L2 rate |
| atomic_ops_per_ns | 400 | 560 | E1 RED M=262144 S=2…64: 388–403 Gatom/s |
| l2_bytes_per_ns | 4500 | 6500 | `__ldcg` 16–48 MB plateau 4.45–4.62 TB/s |
| dram_bytes_per_ns | 890 | 1690 | `__ldcg` 536 MB 894 GB/s, calib B peak 899 (2 GB point 751: open) |
| lat_ns | 573 | 510 | calib B: bytes in flight / DRAM rate below 90 % of peak, median of 125 [557–604] (method reproduces 5090's 510) |
| lsu_ipc_pm | 500 | 500 | calib B `lds`, 64 B/clk/SM LDS.32 |
| flops_per_ns / bytes_per_ns / launch_ns | 32200 / 2553 / 11130 | 53410 / 494 / 17015 | mm_4096 contract cell; pj_m1_n2048 site kernel (L2-resident); pt_pc_64 all-device / 3 sites |
| eff_max_pm / eff_half | 800 / 6 | 800 / 6 | not refit: rtx-5090's fit (same sm_120 SM, a per-SM fraction) |
| clock_khz | 2 450 000 | 2 410 000 | E1 peak FFMA 52.7 TF / (84·128·2) |

  Not added to `GPU_MEASURED`: that set means the rung table was swept on the part, and no rung sweep ran here.
- **A/B, calibrated vs uncalibrated profile** (both bundles timed back to back, `LADDER=1`): candidate 0 is
  unchanged in every program; 11 programs and the skinny cell gain `+plan-cost` / `+full-tile` candidates (some
  tables, capped at 8, drop `+async-fill-wave` / `+bank-plan-row-fit`). The measured choice picked the **same
  candidates** as before in every changed program. Timing change: none beyond the cross-run null (the 25
  programs with byte-identical emission drift median 1.7 %, max 25 % on sub-0.2 ms programs). Verdict: neutral;
  the facts are in place for the plan-cost decisions below.

## 4. Missed optimizations / missing facts, ranked by expected gain

| # | what | evidence | fact / place it belongs | est. gain |
|---|---|---|---|---|
| 1 | read the wisdom (device name + file) at module load, not at the first site | §1 ABBA | runtime ordering (`load_module`), no new fact | −50…−90 us on every tabled program: 8 rows cross to ≤ 1.1 (indep_512 2.84 → 1.08, mlp_512 → 0.95) |
| 2 | wave quantization for tiles that cannot stream-k (fused epilogue) on 84 SMs | 2048 rows: epi tile 497–540 vs stream-k 436–452 us | `TargetProfile` sms × blocks_per_sm (exists) + plan-cost's wave term (now priced); missing: an epilogue-capable stream-k / tile-shape candidate | −50…−90 us per epi GEMM: mlp/mlp2/indep/diamond/twin/fanout 2048 → ~1.0 |
| 3 | FMA on the order-free face for strided-lane sites | attn QKᵀ 1024 FP instrs instead of 512; decode GEMV | `Arithmetic::scalar` should take `free` for non-radix sites as it does for radix (ADR-0040 D2 bound) — no new fact | attn QKᵀ ~12 → ~9 us |
| 4 | under-filled waves at 1024 (128 CTAs on 168 slots) | loopmm, smlp, attn P·V (1-warp CTAs) | same wave fact; stream-k / split candidates for 1024 sites | loopmm 1.10 → ~0.9–1.0 |
| 5 | 16-byte operand staging for implicit GEMM (radix) | tf32 mma tile: 16 × 4-byte `LDGSTS`, 143 IMAD | a data-move site (`move_plan`/`footprint_plan`) that makes the k-axis contiguous once, as cuDNN's NHWC pass does | convmc_tf32 tile 71.6 → ~45 us (cuDNN) |
| 6 | launch / zero-pass count in loop bodies | decode: 24 launches + 16 memsets per step, Σ kernels < half the run | issue_ns / gap_ns / zero_ns now calibrated → plan-cost prices zero passes; structural: graph replay of a `loop_plan` body | decode 1.10 → < 1.0 plausible (host time is 55 %) |
| 7 | warp-shuffle row-fold tree in flow kernels | softmax flow 7.4 vs triton 5.3 us; 19 BAR, 0 SHFL | `fold_tree_step` lanes → SHFL before smem | attn −2 us, softmax −0.03 ms |
| 8 | multi-stage `cp.async` for compute-bound GEMMs | every tile `stages=1`; library 4-stage | `async` / `async-depth` rungs exist, gated by rows from other parts | 4096³ 1.04–1.07 → ~1.0 |
| 9 | vectorized epilogue operand loads | 64–128 `LDG.32` vs `STG.128` | elem_plan's contiguity of the epilogue operand | small (once per tile) |

## Open

- Equal L2 state for both sides (flush in Mapal's probe or no flush in torch_bench) before reading rows whose
  working set fits L2.
- The DRAM 2 GB `__ldcg` point (751 GB/s) vs 536 MB / calib B (894–899 GB/s).
- eff_max_pm / eff_half refit on this part (plancost.py over a plan-cost sweep).
- 428 / 399 evidence mismatches (cal / uncal runs): same classes as per-site-choice's open alternatives item.
- Cross-run drift up to 25 % on sub-0.2 ms programs: decisions on them need same-run ABBA.
