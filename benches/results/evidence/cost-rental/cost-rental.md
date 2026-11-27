# cost-rental — the cost model's primitives on five rented parts, the replay re-scored, ABBA per part (ADR-0050)

The rental round of the cost-model step (`benches/results/evidence/local-rtx-5080/cost-model.md` §6): on an RTX
3090, an A100, an RTX 4090, an H100 SXM and an RTX 5090, one `place_calib.sh` run each (every primitive: its saturated
rate, one use's latency, the concurrency that saturates it), the facts into each part's `PlaceFacts` (measured
provenance), the replay re-scored with every part's facts, and the `machine-place` A/B (ON against OFF, value gates
first, ABBA) of the suite programs the search moves on that part. Spec: ADR-0050; raw output (never in git):
`.claude/raw-evidence/cost-rental/` (rows of `benches/results/evidence/RAW-manifest.tsv`, asset
`pending:raw-cost-rental.tar.gz`).

**Verdict: `machine-place` stays OFF.** (a) The replay: the model picks 32 sites against the rules' 59 (28 before
the rented facts). (b) On every part some moved program is slower than OFF beyond the null (round 2, the final code):
the per-site trial's candidates the ON arm forgoes, the TF32 MLP sites, the one-warp FMA tile. (c) Round 1's value
gates failed on the A100 (three programs) and the H100 (one cell's fixed-order faces): four lowering bugs, fixed with
tests (§6); round 2's gates pass on every part.

## 1. The round

| part | box | GPU power limit / clock lock | host CPU | driver | $/h | rented |
| --- | --- | --- | --- | --- | ---: | --- |
| RTX 3090 | vast 62832819 (California) | 280 W (stock 350 W) / none | EPYC 7K62 | 615.71.09 | 0.159 | calib + A/B |
| A100-SXM4-40GB | vast 62832820 (Alberta) | 300 W / none | Xeon Gold 6138 | 615.71.09 | 0.523 | calib + A/B |
| RTX 4090 | vast 62833449 (Oregon) | 450 W / none | EPYC 7702 | 615.71.09 | 0.441 | calib + A/B |
| H100 80GB HBM3 SXM | vast 62832815 (Washington) | 700 W / none | EPYC 9354 | 610.57.04 | 1.788 | calib + A/B |
| RTX 5090 | vast 62834959 (California) | 500 W / none | EPYC 7R32 | 580.126.09 | 0.509 | calib + A/B |

Every box: image `nvidia/cuda:13.0.2-devel-ubuntu22.04` (nvcc 13.0), one box a part held from the calibration to the
A/B (the A/B bundle needs the part's measured facts), labeled `mapal-cost-<part>`, destroyed when its run ended
(`xcost.sh`; a watchdog over our instance ids capped the round at $14). Two offers never started (an RTX 4090 and an
RTX 5090 box) and were destroyed for the next offer. Spend: §8 (≈ $5.7).

## 2. The calibration (one `place_calib.sh` run a part)

`calib/<part>.place` holds each run's calibration line (`place_calib_facts.py`), its first line the box.

| primitive | rtx-3090 | a100 | rtx-4090 | h100 | rtx-5090 | rtx-5080 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SM clock sustained, GHz | 1.799 | 1.394 | 2.550 | 1.830 | 2.821 | 2.879 |
| FFMA flop/clk/SM / clk | 250 / 4.52 | 128 / 4.51 | 250 / 4.54 | 250 / 4.53 | 250 / 4.56 | 250 / 4.55 |
| mma tf32 flop/clk/SM / clk | 256 / 33 | 1008 / 24.1 | 256 / 33 | 1354 / 24.8 | 256 / 34.8 | 256 / 35.1 |
| mma bf16 | 512 / 32.9 | 2030 / 24.2 | 512 / 33 | 2706 / 24.9 | 512 / 34.9 | 512 / 35.2 |
| mma e4m3 | – | – | 1023 / 33.2 | 17170 / 21.5 | 1024 / 34.8 | 1024 / 35.2 |
| wgmma m64n128k16 bf16 | – | – | – | 4094 / 122 | – | – |
| ld.global.16 L2 B/ns/SM / ns | 26.3 / 142 | 29.1 / 221 | 34.8 / 122 | 49.5 / 157 | 36.5 / 134 | 56.1 / 132 |
| ld.global.16 DRAM | 10.7 / 287 | 13.6 / 409 | 7.46 / 240 | 23.2 / 360 | 9.95 / 316 | 10.9 / 313 |
| cp.async.16 L2 | 26.3 / 154 | 39 / 174 | 34.2 / 131 | 49.4 / 169 | 36.5 / 142 | 56 / 140 |
| cp.async.16 DRAM | 10.7 / 298 | 13.7 / 426 | 7.47 / 248 | 23.3 / 372 | 9.94 / 323 | 10.9 / 321 |
| TMA 16 KiB box L2 | – | – | – | 60.5 / 255 | 42.7 / 401 | 56.7 / 400 |
| TMA 16 KiB box DRAM | – | – | – | 22.9 / 550 | 10.6 / 402 | 11 / 400 |
| ld.shared.16 B/clk/SM / clk | 247 / 66.1 | 247 / 66.1 | 247 / 68.1 | 252 / 66 | 254 / 67.8 | 254 / 67.8 |
| ldmatrix.x4 | 128 | 128 | 128 | 291 | 289 | 288 |
| st.global.16 DRAM B/ns/SM | 10.6 | 13.8 | 7.33 | 23.9 | 9.85 | 10.9 |
| st.global.16 L2 | 23.3 | 22 | 18.7 | 28.2 | 19.9 | 28.7 |
| red.add.f32 contiguous / frag / sector op/ns/SM | 4.74 / 0.689 / 0.689 | 4 / 1.15 / 0.566 | 3.75 / 0.848 / 0.484 | 4.97 / 1.27 / 0.767 | 4.19 / 1.74 / 0.612 | 5.7 / 2.6 / 0.829 |
| bar.sync 256 clk | 38.7 | 37.8 | 38.7 | 28.7 | 28.8 | 28.7 |
| mbarrier phase clk | 92.1 | 92.1 | 106 | 98.5 | 100 | 146 |
| grid sync ns | 871 | 1590 | 793 | 1520 | 939 | 798 |
| launch issue / gap ns | 3603 / 1024 | 1694 / 1966 | 2926 / 1024 | 1973 / 1668 | 3069 / 2550 | 1634 / 584 |
| L2 keep (cyclic / rotated / strided), MiB | 6.1 / 6.3 / 6.3 | 40.0 / 42.0 / 42.0 | 72.0 / 73.8 / 67.5 | 31.2 / 37.5 / 31.2 | 98.4 / 98.4 / 90.0 | 65.6 / 65.6 / 65.6 |
| start past the first wave | lockstep | lockstep | lockstep | lockstep | lockstep | lockstep |

Rates are a sweep's best (min / max over three runs the interval), latencies one use (a dependent chain), the RTX
5080's from `cost-model.md` beside them. Checks against the parts' published figures: DRAM 10.7 / 13.6 / 7.46 / 23.2 /
9.95 B/ns/SM × SMs = 878 GB/s, 1.47, 0.95, 3.07, 1.69 TB/s; the H100's warpgroup bf16 product 4094 flop/clk/SM (989
TFLOPS at 1.83 GHz, its dense peak); the A100's `mma.sync` bf16 2030 (its 312 TFLOPS); the 3090's L2 keep 6.1 MiB of
its 6 MiB. A capability was taken as a module the driver's JIT accepts: fp8 `mma.sync` (compute_89 PTX) on the 4090
/ H100 / 5090 / 5080, TMA boxes (compute_90) on the H100 / 5090 / 5080, `wgmma` (compute_90a) on the H100 alone. That
is not sound (review, §9): the H100's JIT accepted the fp8 module and ran it emulated. `place_calib.sh` now keeps a
module only where the box's own SASS of it holds the probed instruction (an MMA reading E4M3, `UTMALDG`, `HGMMA`;
offline: sm_90 gives `F2FP.F16.E4M3.UNPACK_B` + `HMMA.16816.F32`, sm_89 / sm_120 `QMMA.16832.F32.E4M3.E4M3`).

**The `ld.shared` / `ldmatrix` latencies above are inflated** (review, §9): the probe's chain branched on a run-time
kind inside the loop, moved its address by an `IMAD` a step, and its `lane × 16` words gave `ld.shared.u32` a 4-way
bank conflict. The pure pointer chase on the RTX 5080 measures 28.0 / 41.9 / 35.9 clk (`ld.shared.4` / `.16` /
`ldmatrix.x4`) against 67.8 / 67.8 / 69.0; the five rented parts' rows are open (§9).

**Two probe defects.** (1) The warpgroup sweep launched 2–4 warpgroups into a kernel bounded at 128 threads and the
driver-API launch aborted the run (`place_calib.cu:1137 driver error 1`, the first H100 run): the kernels are now
bounded at 512 / 256 lanes (c = 1 / 2 chains; c = 2's accumulators take 128 registers a lane) and a refused point is
skipped like any footprint the SM cannot hold (`eff_launch_failed`). (2) The H100's fp8 `mma.sync` row measures 17 175
flop/clk/SM — above the part's dense fp8 peak: sm_90 has no fp8 warp product, ptxas lowers it to an f16 `HMMA` after an
`F2FP` unpack (sm_89: `QMMA`), and the probe's loop is not that work. It is left out of `PLACE_H100` and (review,
§9) out of `calib/h100.place`, with the SASS check above in `place_calib.sh`.

## 3. The facts

`PLACE_RTX_3090`, `PLACE_A100`, `PLACE_RTX_4090`, `PLACE_H100`, `PLACE_RTX_5090` (`crates/backends/llvm/src/profile.rs`)
gain the run's start timing, L2 keeps and hit curve, efficiency points, dependent latencies, sustained clock and
`prim` rows, each cited to its box (`cite`), every value measured. `place::machine_of` reads them as measured
`Fact`s: no part's cost is derived from another part's primitives any more.

## 4. The replay re-scored (`place_replay_scores`, 76 sites)

| part (sites) | model before: picked / missed / open / null | model after | rules: picked / missed / null |
| --- | --- | --- | --- |
| RTX 3090 (6) | 2 / 1 / 3 / 0 | **5 / 1 / 0 / 0** | 6 / 0 / 0 |
| A100 (8) | 2 / 3 / 3 / 0 | 1 / 2 / 5 / 0 | 8 / 0 / 0 |
| RTX 4090 (7) | 3 / 0 / 3 / 1 | **5 / 1 / 0 / 1** | 4 / 2 / 1 |
| H100 (10) | 1 / 0 / 9 / 0 | 1 / 0 / 9 / 0 | 10 / 0 / 0 |
| RTX 5090 (1) | 0 / 0 / 1 / 0 | 0 / 0 / 1 / 0 | 1 / 0 / 0 |
| RTX 5080 (44) | 20 / 5 / 17 / 2 | 20 / 5 / 17 / 2 | 30 / 12 / 2 |
| **all (76)** | **28 / 9 / 36 / 3** | **32 / 9 / 32 / 3** | **59 / 14 / 3** |

Residuals (`log(measured / model)` over the µs-measured arms; pairwise `|log((model_b/model_a)/(meas_b/meas_a))|`
against each site's measured best):

`bias` here is the **mean** of `log(measured / model)`; `cost-model.md` §4 reports the signed **median** under the same
name (review, §9: the same state reads mean +0.157 against median +0.055 — the RTX 4090's and 5090's relative-unit
outliers pull the mean).

| | absolute: bias (mean) / median abs / p80 | pairwise: median / p80 |
| --- | --- | --- |
| all, before | +0.157 / 0.107 / 0.409 (126) | 0.088 / 0.193 (88) |
| **all, after** | **+0.157 / 0.105 / 0.409** | **0.074 / 0.174** |
| RTX 3090 before → after | −0.012 / 0.095 → +0.054 / 0.054 (2) | 0.157 → 0.044 (6) |
| A100 | −0.035 / 0.183 → −0.053 / 0.184 (6) | 0.151 → 0.167 (8) |
| RTX 4090 | +0.696 / 0.140 → +0.695 / 0.123 (6) | 0.118 → 0.100 (7) |
| H100 | −0.200 / 0.286 → −0.198 / 0.282 (18) | 0.090 → 0.090 (10) |

The band re-derived (`PLACE_BAND_PM`, test `place_band_is_the_replays_residuals`): all 176 → 160 ‰, compute 143 →
127, DRAM 217, cache 247 → 253; `MODEL_FLOOR` 28 → 32 (review, §9: the floor is now the picked sites themselves,
`REPLAY_PICKED`, against the pre-cost-model level 37).

**Every miss and the term it lacks** (`REPLAY_OPEN`, test-checked):

| site | measured | the model | the term it lacks |
| --- | --- | --- | --- |
| cut-8 / -9 (A100 128 × 4096 × 14336 bf16 / TF32) | cut 8 by 10 / 25 % | uncut 7 / 3 % under | the overlap of near-equal flows: the uncut's L2 168 µs against the ring's latency 134 and the scratchpad 124 (all under one max) |
| cut-12 (4090, the same site, bf16) | cut 2 by 8 % | uncut 7 % under | the same: the uncut's DRAM 127 µs against its latency 123 |
| cut-2 (3090, the same site, bf16) | uncut by 7.8 % | cut 8 6 % under | the same, the other way: the cut's L2 218 against compute 207 and DRAM 197 |
| hyb-0, hyb-1, walk-16, walk-6 / -7 (5080) | — | — | unchanged (`cost-model.md` §4) |

cut-0 (3090 2048³ TF32) and cut-10 (A100 2048 × 4096 × 14336) left the list: the 3090's measured reductions price
cut-0's partial adds and the model picks the uncut grid; cut-10 is a tie within the null.

**Tried (not committed): the overlap of flows.** The four cut misses share one structure — the losing arm's flows sit
close under its bound, the winner's apart — and the steady state takes their max (a perfect overlap). A trip's steady
state as `(Σ tᵢᵖ)^(1/p)` instead (an experiment behind an environment variable, reverted): p = 2: 44 picked / 8 missed;
p = 3: 41 / 6; p = 4: 37 / 5 (all four cut misses and hyb-0 picked; cut-11 missed); p = 6: 33 / 7; p = ∞ (today): 32 /
9. The term separates the cases, but `p` would be a fit of the replay (ADR-0050: no fits read by the evaluator): the
overlap law of two concurrent flows is a primitive to measure (a two-flow probe per part: compute beside an L2 / DRAM
stream at controlled ratios, and two movers together), not a parameter to tune. Even p = 2 stays below the rules' 59.

**The H100's nine opens are ties**: its warpgroup arms (`wgmma-ws` against `wgmma`, the rasters) price alike — every
arm compute-bound at the warpgroup rate. The measured TMA (60.5 B/ns/SM from L2) against `cp.async` (49.4) did **not**
separate the movers (review, §9): no measured box matched the lowering's declared box (21 845 / 32 768 B against the
4096 / 16 384 B rows) and `Machine::cost` had no nearest-row path for a bulk copy, so every warp-specialized arm
priced its mover from the derived reach facts; it now borrows the largest measured box at most its own (marked,
naming the row). The TF32 warpgroup product's rate is unmeasured (the probe measures bf16, and its `n = 128` rows were
read by no machine row until §9), and the bf16w arms' f32 `a` operand conversion is an issue term the model does not
separate.

## 5. ABBA (`machine-place` ON against OFF)

`gpu_changed.sh <out> --on machine-place --arch <part>` (with `PART=<part>:<sm>`, new in this step: the local bundle
scripts build for another part) on each part's measured facts, then `box_run.sh` on the box (`LADDER=1`, value gates
first, 5 processes an arm, the order reversed on even repetitions; the cells in `tile_bench --splitk`, two passes).
`ON/OFF` is the median of the program's own `iter ms` (untraced), `kernel` the traced kernels' sum, `null` the
bundle's equal-text copy against OFF. Programs within 3 % on both and their null are left out of the tables.

Two passes. **Round 1** (the bundles built from the measured facts, before §6's fixes) found the value failures and
the rest of §6: A100 — `causal` / `decode` / `gen` printed nothing (§6.2); H100 — the 16 × 2048 × 7168 cell's
fixed-order faces outside the bound (§6.3), and the TF32 MLPs' PTX refused before the run (§6.1); the 3090's FMA
128 × 256 tile spilled (`mm_4096³` 3.44 ×, `rowsum_4096` / `softmax_rows_4096` 3.2 ×). **Round 2** (below) times the
final code on fresh boxes, every bundle from the committed emitter. `ON / rules` is the ON arm against candidate 0 of
the OFF build's site table (`MAPAL_GPU_PICK=rules`: the rules' launch without the runtime trial); OFF itself runs
each site's wisdom (the per-site-choice trial's pick).

**rtx-3090** — round 2 not run: two 3090 boxes reset the connection mid-upload (`ssh`: "Connection reset by peer"
after ~4 MB, a chunked upload too; other parts' uploads went through), both destroyed. Round 1 (before §6's fixes):
value gates pass; slower beyond the null `rowsum_4096` 3.27 ×, `softmax_rows_4096` 3.21 ×, `mm_4096³` 3.44 × (the
spilling FMA tile), `twin_2048` 1.32, `rowsum_2048` 1.30, `decode_1024_256` 1.28, `diamond_2048` 1.20, `smlp` 1.20,
`mlp_2048` / `mlp2_2048` / `indep_2048` / `loopmm` / `fanout_2048` 1.13–1.15 (null 0.99–1.04).
**a100** — 14 programs timed, 8 within 3 % of OFF; value gates: all pass.

| program | OFF iter ms | ON iter ms | ON / OFF | ON / OFF kernel | ON / rules | null / OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_convmc_8_64_56_3 | 0.2443 | 0.3574 | **1.463** | 1.476 | 0.990 | 1.023 |
| pt_convmc_8_64_56_3_tf32 | 0.2487 | 0.3576 | **1.438** | 1.437 | 0.993 | 1.045 |
| pt_indep_512 | 0.0950 | 0.1026 | **1.080** | 1.040 | 0.957 | 1.015 |
| pt_mlp2_2048_tf32 | 1.2024 | 1.6205 | **1.348** | 1.339 | – | 1.004 |
| pt_mlp_2048_tf32 | 0.6657 | 0.8101 | **1.217** | 1.215 | – | 1.001 |
| pt_pj_m16_n18432_k7168 | 0.4719 | 0.5058 | **1.072** | 1.082 | 1.001 | 1.003 |

**rtx-4090** — 27 programs timed, 21 within 3 % of OFF; value gates: all pass.

| program | OFF iter ms | ON iter ms | ON / OFF | ON / OFF kernel | ON / rules | null / OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_decode_1024_256 | 11.6847 | 21.6401 | **1.852** | 1.537 | 1.851 | 1.001 |
| pt_indep_512 | 0.0363 | 0.0320 | **0.884** | 1.137 | 0.768 | 0.999 |
| pt_loopmm_1024_8 | 0.4012 | 0.4227 | **1.054** | 1.003 | 1.060 | 0.993 |
| pt_mlp_512 | 0.0348 | 0.0326 | **0.937** | 0.949 | 0.862 | 1.024 |
| pt_pj_m16_n2048_k7168 | 0.0542 | 0.0508 | **0.937** | 1.011 | 0.933 | 1.000 |
| pt_pj_m1_n2048_k7168 | 0.0326 | 0.0309 | **0.948** | 0.979 | 0.924 | 1.015 |

Cells (`tile_bench --splitk`, ON / OFF): mm_16x2048x7168_c 0.253 (null nan, bound yes); mm_16x2048x7168_rc 0.968 (null nan, bound yes); mm_16x2048x7168_x 0.454 (null nan, bound yes).

**h100** — 30 programs timed, 10 within 3 % of OFF; value gates: all pass.

| program | OFF iter ms | ON iter ms | ON / OFF | ON / OFF kernel | ON / rules | null / OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_attn_1024_64 | 0.0418 | 0.0373 | **0.891** | 0.873 | 0.884 | 1.021 |
| pt_convmc_8_64_56_3 | 0.1198 | 0.0914 | **0.763** | 0.763 | 0.763 | 1.001 |
| pt_convmc_8_64_56_3_tf32 | 0.1195 | 0.0912 | **0.763** | 0.767 | 0.761 | 1.001 |
| pt_diamond_2048 | 1.2479 | 1.3393 | **1.073** | 1.057 | 1.001 | 1.000 |
| pt_fanout_2048 | 0.8318 | 0.9080 | **1.092** | 1.074 | 1.007 | 1.000 |
| pt_indep_2048 | 0.8394 | 0.8979 | **1.070** | 1.059 | 0.999 | 1.000 |
| pt_indep_512 | 0.0455 | 0.0391 | **0.859** | 1.044 | 0.734 | 0.993 |
| pt_loopmm_1024_8 | 0.5458 | 0.5977 | **1.095** | 1.089 | 0.933 | 1.002 |
| pt_mlp2_2048 | 1.6310 | 1.7726 | **1.087** | 1.077 | 1.000 | 1.001 |
| pt_mlp2_2048_tf32 | 0.3640 | 0.4127 | **1.134** | 1.124 | – | 1.001 |
| pt_mlp_2048 | 0.8152 | 0.8903 | **1.092** | 1.082 | 1.001 | 1.000 |
| pt_mlp_2048_tf32 | 0.1851 | 0.1988 | **1.074** | 1.076 | – | 0.998 |
| pt_mlp_512 | 0.0435 | 0.0373 | **0.858** | 0.944 | 0.742 | 1.010 |
| pt_pj_m16_n18432_k7168 | 0.2057 | 0.2178 | **1.059** | 1.057 | 0.993 | 1.000 |
| pt_pj_m16_n2048_k7168 | 0.0528 | 0.0412 | **0.780** | 0.742 | 0.415 | 0.995 |
| pt_rowsum_2048 | 0.4251 | 0.4548 | **1.070** | 1.068 | 1.001 | 0.998 |
| pt_rt_mlp2_1024 | 2.2509 | 2.4792 | **1.101** | 1.069 | 0.957 | 1.000 |
| pt_rt_mlp2_2048 | 13.1972 | 14.3277 | **1.086** | 1.073 | 1.000 | 1.000 |
| pt_smlp_4096_256_1024 | 0.1335 | 0.1406 | **1.053** | 1.033 | 0.971 | 1.001 |
| pt_twin_2048 | 0.8268 | 0.8920 | **1.079** | 1.065 | 1.002 | 1.000 |

Cells (`tile_bench --splitk`, ON / OFF): mm_16x2048x7168 0.387 (null 1.001, bound yes).

**rtx-5090** — 30 programs timed, 18 within 3 % of OFF; value gates: all pass.

| program | OFF iter ms | ON iter ms | ON / OFF | ON / OFF kernel | ON / rules | null / OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pt_attn_1024_64 | 0.0517 | 0.0383 | **0.741** | 0.779 | 0.758 | 0.934 |
| pt_convmc_8_64_56_3 | 0.0699 | 0.0669 | **0.957** | 0.970 | 0.959 | 0.999 |
| pt_decode_1024_256 | 26.9349 | 25.3935 | **0.943** | 1.004 | 0.951 | 0.987 |
| pt_indep_512 | 0.0484 | 0.0295 | **0.609** | 0.955 | 0.597 | 0.881 |
| pt_loopmm_1024_8 | 0.4092 | 0.4413 | **1.078** | 0.991 | 0.993 | 1.001 |
| pt_mlp2_2048_tf32 | 0.9110 | 1.1867 | **1.303** | 1.255 | – | 0.997 |
| pt_mlp_2048_tf32 | 0.4513 | 0.5930 | **1.314** | 1.284 | – | 1.000 |
| pt_mlp_512 | 0.0350 | 0.0303 | **0.866** | 0.888 | 0.753 | 1.051 |
| pt_pj_m16_n2048_k7168 | 0.0373 | 0.0339 | **0.909** | 0.965 | 0.619 | 1.008 |
| pt_pj_m1_n2048_k7168 | 0.0333 | 0.0314 | **0.943** | 1.021 | 0.911 | 1.009 |
| pt_rt_mlp2_1024 | 1.7023 | 1.9110 | **1.123** | 1.025 | 1.001 | 1.000 |
| pt_smlp_4096_256_1024 | 0.1118 | 0.1171 | **1.047** | 1.068 | 0.794 | 0.994 |

Cells (`tile_bench --splitk`, ON / OFF): mm_16x2048x7168 0.726 (null 1.016, bound yes); mm_16x2048x7168_c 0.775 (null nan, bound yes); mm_16x2048x7168_r 0.699 (null nan, bound yes); mm_16x2048x7168_rc 0.699 (null nan, bound yes); mm_16x2048x7168_x 0.756 (null nan, bound yes).

**What the round-2 A/B says.**

- **Against the rules' launch the model's picks win on the under-filled sites**: the H100's `pj_m16_n2048`
  0.415 ×, `mlp_512` 0.74, `convmc` 0.76, `attn_1024_64` 0.88; the 5090's `pj_m16_n2048` 0.62, `mlp_512` 0.75, `attn`
  0.76, `smlp` 0.79; the 4090's `mlp_512` 0.86; the 16 × 2048 × 7168 cell 0.39 × on the H100 (and its contract and
  separate faces 0.43 / 0.53: the cells table labels their kernel "same" by its name, which hashes the site, but the
  box log shows a moved kernel — 64 lanes, 49 920 B of shared memory, 96 / 62 registers against 256 / 9 216 / 32 / 39),
  0.70–0.78 × on the 5090 on every face, and 0.25 / 0.45 × on the 4090's sequential and exact faces (every value
  within its bound).
- **Open, not wins** (review, §9): `indep_512`'s iteration time falls (H100 0.73 against rules, 0.859 against OFF;
  5090 0.60 / 0.609; 4090 0.77 / 0.884) while its traced kernels do not (ON / OFF kernel 1.044 / 0.955 / 1.137; the
  5090's null 0.881): the kernels do not explain the iteration time, so no decision reads these ratios.
- **Against OFF — the rules plus the per-site trial, the shipped default — the trial wins where the model kept the
  rules' tile**: the H100's 2048² programs 7–10 % (`ON / rules` 1.00: the trial picked candidates 3–4), the A100's
  `convmc` 1.44–1.46 × (`ON / rules` 0.99). A placed site's table is its pick alone (ADR-0050 C4): the run-time
  confirmation among the search's top (C6) is not built, so the ON arm forgoes the trial's candidates.
- **Losses against both**: the TF32 MLPs (A100 1.22 / 1.35, H100 1.07 / 1.13, 5090 1.31 / 1.30 — an FMA walk or a
  split TF32 warp tile in place of the rules' tensor tile; these sites have no table, OFF is the rules'), `decode_1024_256`
  on the 4090 (1.85 ×: its one-warp 1 × 32 tile), `loopmm_1024_8` on the 4090 (1.05).

## 6. Four lowering bugs the rented A/B found (fixed, tested)

1. **A direct TF32 warpgroup tile** (H100, `pt_mlp_2048_tf32`, `pt_mlp2_2048_tf32`): the search's `wgmma-ws:op7`
   with a lanes-major `b` — ptxas: `Arguments mismatch for instruction 'wgmma.mma_async'`. A K-major-only instruction
   (`MatrixOp::mn_major` false: TF32, fp8 wgmma) takes the transposed product, as the rules lower it; the lowering
   now declares it (`Refusal`: "a K-major-only instruction takes the transposed product"; test
   `tf32_wgmma_direct_product_refused`).
2. **The host launched a kernel the device did not emit** (A100, `pt_causal_1024_256`, `pt_decode_1024_256`,
   `pt_gen_1024_512_256`: no output; reproduced on the local 5080 from the A100's sm_80 PTX, compute-sanitizer: an
   invalid 16-byte global write in `mapal_tile_743210b332aabb82`): the host took the pick's 32 × 32 FMA tile (grid 8 ×
   32) while the kernel kept the rules' 64 × 32 tile — `placed_for` keeps the rules' tile where it stores
   column-major and the pick cannot; `placed` (the host) now keeps it too.
3. **The kernel depended on which was emitted first** (H100 `mm_16x2048x7168`, the fixed-order faces `_r` / `_rc`:
   maxrel 43.5, outside the bound): `placed_for` searched only the order the arms imply (`free` false → sequential)
   and fell back to the rules' kernel where the host's fixed-order pick was not yet cached. The emission's order is now
   a scope (`place::OrderScope`, set beside `WarmScope`): the kernel of a realization is the pick of the order the
   host places the site under, whichever comes first.
4. **A kept pick ran the host's copy of the rules' launch**: at the kernel the rules' launch of a realization is its
   own resize (`r`: the register rule, a walk's full tile); the device took the host's `Launch` instead (on the
   3090's `decode_1024_256` site it differs from the OFF arm's kernel). A kept pick now returns `r`. (Round 1's
   2048² losses on the H100 and the 3090 were not this: round 2 shows them equal to the rules' launch — §5.)

Test `placed_host_launch_is_the_device_kernel` (four programs × six parts): the launched geometry is the kernel's,
and a kept site's kernel is the OFF arm's; it fails without fix 2 or fix 4. It did **not** fail without fix 3 (review,
§9): its fixed-order cell compared the tile only where the host's realization matched the pick's, and fix 2 kept the
rules' tile on that cell on every part. Fix 3's test is `placed_fixed_order_kernel_is_the_host_launch` (four
fixed-order sites × six parts, the kernel of the host's realization emitted first, the whole launch compared): it
fails without `OrderScope` (16 × 18432 × 7168 on the A100).


## 7. The flip rule (ADR-0050) and what blocks it

| condition | result | blocked by |
| --- | --- | --- |
| (a) model ≥ rules on the replay | **no: 32 against 59** | the overlap of concurrent flows (the cut misses on three parts; the hyb-0 DRAM run); the walk's per-unit cost and a small warp tile's rate (5080); the H100's ties (§4) |
| (b) no moved program slower beyond the null, every part | **no** | the per-site trial's candidates the ON arm forgoes (a placed site's table is its pick: C6's run-time confirmation is not built) — H100 2048² 7–10 %, A100 `convmc` 1.45 × (both equal to the rules' launch); the TF32 MLP sites (A100 1.22–1.35 ×, H100 1.07–1.13, 5090 1.30–1.31: an FMA walk or a split warp tile priced under the rules' tensor tile); the one-warp FMA tile (`decode_1024_256` 1.85 × on the 4090); the FMA register tile at 128 accumulators a thread (round 1, 3090 4096³: `ptxas` 255 registers + 208 bytes of spill, 3.2–3.4 ×; the declared registers do not cover `ptxas`; `--place-gate=ptxas` refuses it) |
| (c) value gates pass | **round 1 no; round 2 yes on the four parts it ran** | §6.2 (A100) and §6.3 (H100), fixed: every round-2 gate passes on the A100, RTX 4090, H100 and RTX 5090; the RTX 3090 had no round 2 (§5), and the RTX 5080's local ABBA (`cost-model.md`) predates the four fixes |

## 8. Spend

| part | instances (min) | $ |
| --- | --- | ---: |
| RTX 3090 | 63.6 + 34.6 + 35 (upload failed) | 0.35 |
| A100 | 49.9 + 37.9 | 0.73 |
| RTX 4090 | 5.4 (never started) + 29.8 + 17.7 | 0.37 |
| H100 SXM | 65.5 + 37.8 | 3.08 |
| RTX 5090 | 17.5 (never started) + 116.3 (held while its bundle was rebuilt for §6's fixes) | 1.12 |
| **round** | | **≈ 5.7** |

Every instance of ours destroyed (`vastai show instances`: none labeled `mapal-cost-*` left).

## 9. Review fixes (2026-11-27, no rental)

Every finding of the round's review confirmed; each fix has a test that fails without it (checked by reverting).

| finding | fix | test |
| --- | --- | --- |
| The search's floor above the value on the RTX 3090 / A100 (their memory knee latency, 383 / 1198 ns for `cp.async.16`, below the cache's 624 / 1679): the miss mix fell with misses, the branch and bound could prune a cheaper schedule | A miss's path is a hit's and then the memory's: it waits at least a hit's loaded latency (`eval.rs`, `miss_loaded`). The latency never falls as misses rise; the floor (misses at their least) is admissible on every part | `place_floor_is_admissible` (every replay arm, three bounds; without the fix: `c11-5` on the A100), Q4's floor check and `bnb_is_the_exhaustive_best` on 3090-like rows |
| Fix 3 (`OrderScope`) had no test that catches its absence | — | `placed_fixed_order_kernel_is_the_host_launch` (§6) |
| The H100's `wg.m64n128k16.bf16` rows (rate, efficiency, latency) read by no machine row | A row measures an operation family (`n` the schedule's) at a member `n` it admits, every other part of the geometry equal (`place::measures`), its cite naming the member; the `.rs` form (a from registers) stays unmeasured | `every_measured_row_is_read` (every prim / eff / latency row of every part maps) |
| The measured TMA rows never read | `Machine::near`: a bulk copy borrows the largest measured box at most its own; the warp-specialized arms now read `tma.box16384` / `tma.box4096` (terms `prim.mover.cache<-tma.box16384.l2`) | `a_borrowed_row_is_marked_names_its_row_and_errs_slow` |
| A borrowed row presented as measured, some on the optimistic side | Every borrowed row is marked and names its row; the side that bounds the request where one is measured (narrower copy, smaller box, shorter run, fewer bytes in flight, more lanes, more sectors), else the nearest scaled by its mechanism's bound (bytes, Little's law, sector throughput, lanes arriving) | the same |
| `ld.shared` / `ldmatrix` latency ~2× (the probe's branch, `IMAD` and bank conflict) | `lds_chase_k<KIND>`: a pure pointer chase (each lane's next address the word it loads). RTX 5080 re-measured: 28.0 / 41.9 / 35.9 clk (`ld.shared.4` / `.16` / `ldmatrix.x4`) against 67.8 / 67.8 / 69.0; every other row of the rerun within 6 % of the first | `place_calib_chase_is_pure` (the SASS for sm_80 / 86 / 89 / 90 / 120: each load's address the previous load's destination, nothing between writing it, the wide loads wide, unrolled) |
| Q4 weaker than stated | Q4 over the rows a term reads only (no Panel, no op row), rate / latency / concurrency each raised, at every bound, in both latency regimes, against `lower_bound` and `evaluate_floor` | Q4 (fails without the floor fix) |
| JIT acceptance taken as a capability | `place_calib.sh` keeps a module only where the box's SASS holds the probed instruction; the fp8 rows left `calib/h100.place` | offline check: sm_90 fp8 `F2FP` + `HMMA` → skipped; sm_89 / 120 `QMMA` kept; TMA / wgmma by `UTMALDG` / `HGMMA` |
| The report's overstatements | §2, §4, §5, §6, §7 corrected in place (value-gate coverage, the `indep_512` ratios open, the H100 cells' "same" label, both `bias` definitions, the TMA claim, fix 3's test) | — |
| `MODEL_FLOOR` lowered to fit (37 → 28 → 32) | `REPLAY_PICKED`: the picked sites themselves, asserted like the misses (no pick traded silently under a total), the pre-cost-model 37 reported beside | `place_replay_scores` |
| `PLACE_RTX_4090` mixing two boxes, stale rate provenance | `lat_l2_cite`: the L2 latency cites its own box (an earlier 4090, 5090; the 5080's own); the datasheet rate labels say the machine model reads the prim row | `every_measured_row_is_read` (the 4090's L2 latency cites its box; every op with a prim row is measured) |

**The replay after the fixes** (`place_replay_scores`): model **32 / 7 / 34 / 3** (picked / missed / open / null)
against the rules' 59 / 14 / 3 — the A100's cut-8 / cut-9 left the misses: with a miss waiting at least a hit's
latency the model's value picks the measured winner (cut 8) on both, open where the corners flip it. Picks unchanged
(`REPLAY_PICKED`). Band (`PLACE_BAND_PM`): all 161, compute 126, DRAM 217, cache 253 → 202. Residuals over the 126
µs arms: mean +0.155, signed median +0.051, median |log| 0.110, p80 0.404 (before: +0.157 / +0.055 / 0.105 / 0.409);
pairwise median 0.073, p80 0.176 over 88 pairs. `machine-place` stays OFF (§7).

**Open (each needs a rented part; no rental in this step):**

- The five rented parts' scratchpad rows are the old probe's: run `benches/local/place_calib.sh <out> prim` on an
  RTX 3090, A100, RTX 4090, H100 and RTX 5090 and replace `ld.shared.4`, `ld.shared.16`, `ldmatrix.x4` in each
  `PLACE_*` and `calib/<part>.place` (the 5080's rerun changed no other row beyond 6 %).
- The H100's caps still declare the fp8 warp product (`CAPS_WARPGROUP_MMA` ⊃ `MATRIX_WARP_F8[3..4]`, predating this
  round), which `Caps::matrix_op` lists before the native warpgroup fp8 form: dropping it changes the H100's fp8
  emission, so it needs `gpu_changed.sh --arch h100` on the fp8 programs (value gates, then ABBA) with sm_90's caps
  without the warp fp8 product. The generic direction: the matrix caps from the same SASS check as `place_calib.sh`,
  not the PTX ISA table.

**The workspace** (release, once, `TMPDIR` on disk — the shared `/tmp` tmpfs ran out mid-run and the linker died of
`SIGBUS`, failing 70 nvcc / clang-based tests on a first attempt): **1859 passed, 1 failed, 11 ignored**. The failure,
`machine_place` R3 (`#2 4096 x 8192 x 8192 tf32 | f32`: no `; mapal.place` line on five parts), is §6's fix 2 and
fails identically at the reviewed HEAD (an export of `34e01cf6` built apart): the filter kept the rules' tile wherever
the rules' launch *could* store column-major, so a site that stores row-major (its rules an FMA tile, the pick a TF32
tensor tile) lost its pick and no kernel carried a line. Fixed: the layout is the site's — `place::ColScope` (the row
folds' `colmajor` sites, set beside `WarmScope` at emission) drops a pick for a site that stores column-major alone;
outside an emission the conservative reading stays. R3 passes (96 emissions placed, 32 changed, none of the search's
kernels spilling); after it the backend lib (154 / 0) and `machine_place` (R3, C5 values and picks through the
emulator: 3 / 0) pass again.

## Proposed CLAUDE.md facts-table rows (not applied — CLAUDE.md changes need the user)

| query | what it proves |
| --- | --- |
| `place::OrderScope` | the emission's order: the kernel of a placed realization is the pick of the order the host placed its site under (host launch = device kernel, whichever is emitted first) |
