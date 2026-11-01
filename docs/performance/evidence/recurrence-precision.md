# recurrence-precision: end check and remaining gaps

Raw data: [`benches/results/evidence/recurrence-precision/`](../../../benches/results/evidence/recurrence-precision/)
(sweep `5fd470c`, 4 archs; smoke `c1c0ac0`, rtx-4090). Only the 25 programs whose emission differs from
`cpu-gpu-boundary-latest` were re-run (`changed.tsv`); every other row is **carried (emission identical)** from
[cpu-gpu-boundary.md](cpu-gpu-boundary.md). ON/OFF = the program's `iter ms`, shipped build vs its `.off-<rung>` arm,
ABBA x5, same box; null = the shipped build run twice. Launch counts and per-kernel ms are `probe_kernels` (rep 1) from
`sweep-<arch>/results-<arch>.txt.gz`; kernel geometry is nsys (`nsys-losers.md`, `prof-<arch>.tgz`).

## End check (CLAUDE.md evidence pass, step 5)

Cells: intent rows ok/total (`evidence-<arch>.tsv.gz`, ON and OFF arms) · ON vs OFF ms on the program named.

| optimization | intended | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | decision (the difference that explains it) |
|---|---|---|---|---|---|---|
| `trip-batch` (1a) | the loop enters at the skipped interval's last trip; one device cell writes the batched trips | 12/12 · gen 20.45 vs 39.25 | 12/12 · 16.84 vs 33.57 | 14/14 · 27.91 vs 55.48 | 16/16 · 22.63 vs 44.66 | **ON.** ON makes 4618 launches per run vs 9216 OFF on all 4 archs (`host.batch=1`, `dev.cell_batch=1`): the 512 prompt trips are one cell. 1.92–1.99, null ≤ 2.6 % |
| `range-cell` (1b) | an in-place range map is a device cell over its interval; no written row crosses per trip | 56/56 · decode 31.98 vs 53.48 | 56/56 · 29.68 vs 33.87 | 64/64 · 51.91 vs 68.75 | 68/68 · 36.88 vs 50.41 | **ON.** ON `host.move.h2d_b=0` per trip vs OFF 1024 B (the written row, plus the host range map's sync); nsys decode: 0 H2D / 0 D2H per trip on every arch (one D2D per run). ON adds 1 launch per trip (6144 vs 5120). Without it `trip-batch` cannot fire (gen OFF: 8192 launches, 2.39–2.94) |
| `scalar-fold` (1b) | a whole fold of a device array runs on the device; no copy-out per trip | 40/40 · causal 40.48 vs 55.46 | 40/40 · 33.57 vs 45.96 | 46/46 · 55.53 vs 74.27 | 52/52 · 44.75 vs 64.36 | **ON (default face).** ON `host.move.d2h_b=0` per trip vs OFF 8192 B (`s`, `p` out for the host's `mx`/`den`); ON makes 9216 launches vs 7168. 1.34–1.44. **Contract face: open** (below) |
| `cell-lanes` (2) | a nest cell with fewer cells than resident threads folds across a warp, ≥ 5 butterfly shuffles | 13/13 · causal 40.48 vs 82.38 | 13/13 · 33.57 vs 83.52 | 16/16 · 55.53 vs 142.5 | 19/19 · 44.75 vs 137.4 | **ON.** Same launches (9216). OFF's top kernels are the one-thread-a-cell `s`/`o` nest cells (H100 `eea92378` 1024 × 64.0 µs, `586f1a0c` 1024 × 42.9 µs); ON they leave the top two (`dev.cell_lanes_shfl=2`). 2.04–3.17 |
| `fold-lanes` (2) | a device row fold with idle resident threads is a warp a row | 6/6 · rowsum_2048 0.3176 vs 0.3100, rowsum_4096 2.560 vs 2.536 | emission = OFF | emission = OFF | emission = OFF | **OFF row on rtx-4090.** Realized (`dev.rowfold_lanes_shfl=1`), but the rowfold kernel is slower ON: 0.023 vs 0.020 ms (2048), 0.088 vs 0.075 ms (4096). Its operand is column-major (`tree-gate`), so one thread a row is already coalesced and a warp a row strides its lanes across columns. Smoke 1.004 / 0.994: no positive sign |
| `radix-tile` | a per-digit affine conv fold is a staged FMA tile site; no div/rem | 16/16 · convmc 0.1124 vs 1.003 | 16/16 · 0.0615 vs 0.7059 | 16/16 · 0.1049 vs 1.639 | 14/16 · 0.172 vs 2.661 | **ON.** OFF is the one-thread-per-output nest cell (4090 `16ec8f32` 0.95 ms); ON is an 8×4 (4090, 5090) / 8×8 (H100, 3090) register tile, 512 FFMA per panel trip, `ld.shared.v4` 48, `div_rem=0`. 8.9–17.2. **Mismatch on 3090:** 72 B spill stores / 40 B loads (`cea45c7b`, `6345d960`) |
| `tf32-tile` (ADR-0041) | a product that admits `tf32` takes the tensor-core tile | 15/15 · mlp_tf32 0.506 vs 0.630 | 15/15 · 0.441 vs 0.549 | 15/15 · 0.369 vs 1.249 | 15/15 · 1.701 vs 2.009 | **ON.** 64 `mma.sync` / HMMA per hot loop on 4090, 5090, 3090; 4 `wgmma` / HGMMA on H100. 3.38–3.40 on H100 vs 1.14–1.26 elsewhere: H100 PCIe's dense TF32 tensor peak is about 7× its FP32 peak, the GeForce parts' about 1×, so there the gain is the FFMA tile's issue cost (4090 GEMM 0.31–0.32 → 0.26 ms); the H100 stops at 3.4 because of T1 |
| zero-fill (integration) | a `(0.0, n) -> fill` loop state is zeroed on the device, never copied in | all `rg_*`/`rt_*` rows ok | ok | ok | ok | **Kept (not a rung).** rt_mlp2_2048 0.75 / 1.00 / 1.17 / 0.88 vs 1.12 / 1.12 / 1.27 / 1.00 at cpu-gpu-boundary; nsys: the 16.8 MB pageable H2D (N4: 1.48 / 0.97 ms on 4090 / 5090) is now a 16.8 MB memset of 0.01 ms |

Other intent mismatches: 4090 px_rowsum_128 (unroll, face-registers: 4 rows, an untimed px program); 3090 loop
programs' `off-streamk+splitk-atomic` arm (8–16 B spill) and convmc `_c` prefetch LDG cover 0. None is on a shipped new
rung except the 3090 radix-tile spill.

## GPU_RUNGS_OFF changes (`crates/backends/llvm/src/profile.rs`)

| part | rung | change | evidence |
|---|---|---|---|
| rtx-4090 | `fold-lanes` | new row (OFF) | rowfold kernel ON vs OFF 0.024/0.023 vs 0.020/0.020 ms (rowsum_2048), 0.088/0.088 vs 0.075/0.075 ms (rowsum_4096); program OFF/ON 0.98 (null 1.3 %) / 0.99 (null 0.6 %); smoke 1.004 / 0.994 |

No other row: every other new rung is positive with null ≤ 2.6 % on all four parts (convmc on H100: null 9.7 %, ratio
15.6). The 4090 rowsum rows below are the measured `.off-fold-lanes` arm, which is now the shipped emission.

## Gap table (Mapal / best other; below 1 = Mapal faster)

54 of 120 cells at or below 1.0. On the 96 cells shared with cpu-gpu-boundary it is 48 (was 41). † = carried (emission
identical). `_tf32` rows compare against torch with TF32 on; every other row against FP32 legs. Full table:
[`table.md`](../../../benches/results/evidence/recurrence-precision/table.md).

| workload | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | cpu-gpu-boundary (4090 / 5090 / H100 / 3090) |
|---|---|---|---|---|---|
| decode_1024_256 | 1.03 | 1.52 | 2.48 | 1.12 | 1.66 / 1.60 / 2.95 / 1.69 |
| convmc_8_64_56_3_tf32 | 2.07 | 1.36 | 2.38 | 1.69 | new |
| attn_1024_64 † | 1.97 | 1.39 | 2.00 | 1.77 | same |
| mlp2_2048_tf32 | 1.16 | 1.15 | 1.87 | 1.17 | new |
| mlp_2048_tf32 | 1.06 | 1.16 | 1.75 | 1.40 | new |
| convmc_8_64_56_3 | 1.66 | 1.04 | 1.29 | 1.68 | 17.77 / 12.36 / 20.33 / 27.57 |
| loopmm_1024_8 † | 1.31 | 1.25 | 1.12 | 1.62 | same |
| pj_m16_n2048_k7168 † | **0.86** | 1.21 | 1.35 | 1.56 | same |
| rt_mlp2_1024 | 1.14 | 1.20 | 1.45 | **0.99** | new |
| mlp2_2048 † | 1.01 | 1.12 | 1.39 | 1.03 | same |
| indep_2048 † | **0.99** | 1.08 | 1.35 | 1.09 | same |
| diamond_2048 † | **0.98** | 1.11 | 1.30 | 1.01 | same |
| smlp_4096_256_1024 † | 1.09 | 1.17 | 1.19 | 1.29 | same |
| fanout_2048 † | **0.97** | 1.08 | 1.23 | 1.02 | same |
| mlp_2048 † | **0.97** | 1.09 | 1.22 | 1.04 | same |
| rowsum_2048 | **0.93** | 1.08 | 1.21 | 1.11 | new |
| twin_2048 † | **0.95** | 1.05 | 1.21 | 1.01 | same |
| rt_mlp2_2048 | **0.75** | **1.00** | 1.17 | **0.88** | 1.12 / 1.12 / 1.27 / 1.00 |
| softmax_rows_4096 † | 1.03 | 1.09 | **0.97** | **0.96** | same |
| rowsum_4096 | **0.99** | 1.07 | **0.96** | **0.96** | 1.05 / 1.10 / 0.97 / 0.95 |
| epi_4096 † | **1.00** | 1.06 | **0.97** | **0.88** | same |
| ew_4096 † | **0.46** | **0.45** | 1.01 | **0.61** | same |
| loop_4096 † | **0.95** | **0.97** | **0.97** | **0.96** | same |
| firew_rowsum_2048x2048_16 † | **0.66** | **0.55** | **0.57** | **0.72** | same |
| conv_2048_5 † | **0.63** | **0.45** | **0.50** | **0.68** | same |
| fir_4194304_16 † | **0.61** | **0.51** | **0.56** | **0.63** | same |
| causal_1024_256 | **0.21** | **0.40** | **0.54** | **0.16** | 0.62 / 1.30 / 1.41 / 0.75 |
| prefill_1024_512_256 | **0.21** | **0.38** | **0.53** | **0.16** | 0.64 / 1.25 / 1.30 / 0.81 |
| ewconvew_2048_5 † | **0.53** | **0.51** | **0.51** | **0.53** | same |
| gen_1024_512_256 | **0.16** | **0.44** | **0.52** | **0.12** | new |

## What moved

| workload | 4090 | 5090 | H100 | 3090 | by |
|---|---|---|---|---|---|
| causal_1024_256 | 0.62 → 0.21 | 1.30 → 0.40 | 1.41 → 0.54 | 0.75 → 0.16 | `cell-lanes` 2.04–3.07, `scalar-fold` 1.34–1.44, `range-cell` 1.20–1.47 |
| prefill_1024_512_256 | 0.64 → 0.21 | 1.25 → 0.38 | 1.30 → 0.53 | 0.81 → 0.16 | same three rungs |
| convmc_8_64_56_3 | 17.77 → 1.66 | 12.36 → 1.04 | 20.33 → 1.29 | 27.57 → 1.68 | `radix-tile` 8.9–15.6 |
| decode_1024_256 | 1.66 → 1.03 | 1.60 → 1.52 | 2.95 → 2.48 | 1.69 → 1.12 | `range-cell` 1.14–1.67: no row crosses per trip |
| rt_mlp2_2048 | 1.12 → 0.75 | 1.12 → 1.00 | 1.27 → 1.17 | 1.00 → 0.88 | zero-fill: the in-clock 16.8 MB upload is gone |
| gen_1024_512_256 (new) | 0.16 | 0.44 | 0.52 | 0.12 | `trip-batch` 1.92–1.99 on top of the prefill rungs; gen ÷ prefill same box 0.95–0.99 |

## Top 10 gaps (each workload at its worst arch)

| # | workload | worst arch: gap ms | other archs: gap ms | cause | ms recoverable (basis) |
|---|---|---|---|---|---|
| 1 | decode_1024_256 | H100: 30.95 | 5090 10.19, 3090 3.83, 4090 0.96 | **D1**: the per-trip K/V one-row site runs on one CTA (grid 1×1×1, 256 threads): H100 `3881eb4b` 2046 × 12.9 µs = 26.4 ms, 5090 `e2f8ae46` 2046 × 4.2, 4090 `fee5d054` 2046 × 4.6. The cache leg's gemv uses 64–256 CTAs at 2.9–3.2 µs. Plus 6–7 launches and 1–2 memsets per trip. No copy per trip remains | H100 ≤ 20.5 (2046 × (12.9 − 2.9) µs); 5090 ≤ 8.6 (site total) |
| 2 | rt_mlp2_2048 | H100: 2.94 | 5090 0.04 | C6a: 24 GEMMs at 632–649 µs (16×16 CTAs, 256 threads, 122 regs, 17.4 KB smem: the 64-accumulator 1-stage tile) | ≤ 2.94 |
| 3 | rt_mlp2_1024 | H100: 1.21 | 5090 0.31, 4090 0.25 | C6c: 1024² GEMMs on 16×8 = 128 CTAs over 114 SMs (two waves, the second 14 CTAs), 124.5–133 µs vs torch's FP32 GEMM (split-K 4, 1024 CTAs) 79.6 µs | ≤ 1.77 (Σ gaps) |
| 4 | mlp2_2048 † | H100: 0.707 | 5090 0.124, 3090 0.110 | C6a | 0.83 |
| 5 | loopmm_1024_8 † | 3090: 0.558 | 4090 0.117, 5090 0.090, H100 0.073 | C6c | 0.84 |
| 6 | `_tf32` GEMMs (mlp, mlp2) | 3090: 0.403 (mlp2) | H100 0.346 / 0.157, 4090 0.134, 5090 0.118 | **T1** H100: 16×8 = 128 CTAs of 205.8 KB smem, one per SM over 114 SMs: two waves, the second 14 CTAs; 173–181 µs per GEMM (≈ 97 TFLOP/s). 3090: no trace (O5) | H100 ≤ 0.50 |
| 7 | diamond_2048 † | H100: 0.440 | 5090 0.091, 3090 0.033 | C6a | 0.56 |
| 8 | indep_2048 † | H100: 0.331 | 3090 0.134, 5090 0.044 | C6a | 0.51 |
| 9 | convmc_8_64_56_3 | 3090: 0.070 | 4090 0.045, H100 0.024, 5090 0.002 | **R1**: 784 CTAs × 64 threads (2 warps), 7 blocks/SM = 12.25 resident warps of 48 on the 4090; 100.7 µs ≈ 18 TFLOP/s (22 % of FP32 peak) vs cuDNN FP32 67.6 µs. 3090: 72 B spill | ≤ 0.14 |
| 10 | convmc_8_64_56_3_tf32 | H100: 0.055 | 4090 0.056, 3090 0.060, 5090 0.016 | **R2**: `: tf32` does not reach a radix-tile site: it emits FFMA (`sass.hot.ffma=512`, no HMMA/HGMMA) while cuDNN's TF32 leg runs on tensor cores (H100 39.6 µs) | ≤ 0.19 |

Carried gaps below these (C6a fanout / mlp / twin / rowsum_2048 on H100, C6b on 5090, attn C9/N3, pj C10): see
[cpu-gpu-boundary.md](cpu-gpu-boundary.md#causes); unchanged emission.

## Causes (new or changed)

| id | rows | evidence | missing fact / realization |
|---|---|---|---|
| D1 | decode ×4 | One-CTA one-row site, 2046 launches per run; 6–7 kernels and 1–2 memsets per trip | Split the one-row site's fold across CTAs or lanes when its cells are fewer than the resident threads (`cell-lanes` reaches nest cells, not the retained tile site) |
| R1 | convmc ×4 | 2-warp CTAs, 12.25 resident warps per SM; 3090 spills 72 B | Radix-tile block shape from `TargetProfile` (more warps per CTA); a 3090 tile that fits its registers |
| R2 | convmc_tf32 ×4 | FFMA only on a program that admits `tf32` | `tf32-tile` for radix-tile sites |
| T1 | `_tf32` on H100 | 128 CTAs over 114 SMs at one CTA per SM | Grid-share / stream-K for the wgmma tile |
| C5 | decode, causal, prefill | **Closed.** No H2D/D2H per trip left (`range-cell`, `scalar-fold`) | – |
| N1 | causal, prefill | **Closed** by `cell-lanes`: the `s`/`o` nest cells (H100 1024 × (64.0 + 42.9) µs OFF) leave the top two kernels | – |
| N2 | convmc | **Closed** by `radix-tile` (17.8 → 1.66 on the 4090); R1/R2 remain | – |
| N4 | rt_mlp2 on 4090 / 5090 | **Closed** by zero-fill | – |

## Open

| id | what | evidence that closes it |
|---|---|---|
| O1 | `scalar-fold` on the contract face: 0.96 (5090), 0.97 (3090), 0.98–1.00 (H100), but 1.16 on the 4090. The usage difference is the same on every part (ON 9216 launches with two in-order whole folds per trip vs OFF 7168 launches plus 8 KB of copy-outs), so the sign by part is unexplained, and no contract null was timed. No row | nsys of the `_c` ON and OFF arms, plus a contract null |
| O2 | `fold-lanes` was measured only on a column-major operand; its gate reads no layout fact | Gate lanes on `k` being the unit-stride axis, then re-measure a row-major rowfold and drop the 4090 row if positive |
| O3 | gen on the 4090: Mapal 14.1 ms in the torch-pass box vs 20.45 ms in the pass-1 box (same build) | Same-box re-time; the gen/best ratio uses the torch-pass box's own legs |
| O4 | `cell-lanes` changed `rowsum_*_xabs` on all 4 parts; not timed | ON/OFF of `pt_rowsum_2048_xabs` / `_4096_xabs` |
| O5 | 3090 profile archive truncated: no kernel split for its `_tf32` gaps (1.17–1.40) | A 3090 nsys of `pt_mlp_2048_tf32` |
| O6 | convmc on H100: null 9.7 % | Re-time at 9 reps |
| O7 | Older opens (O3, O7, O8, O11–O15 of cpu-gpu-boundary) are unchanged | see [cpu-gpu-boundary.md](cpu-gpu-boundary.md#open) |
