# plan-cost — end check

Sprint `plan-cost` (branch `plan-cost`, 2026-11-08/09). One cost model (`TargetProfile::site_ns`, design
`.claude/sprints/plan-cost/design.md` §2) picks each tile site's plan (combine × S × geometry) and `linear` launches a
sum of tile sites over one `k` range as one site table. Sweep: rtx-4090 and rtx-5090 (the only parts whose emission
changed; h100-pcie / rtx-3090 emit as before on every face), arms ON / OFF (`--gpu-off=plan-cost,linear`) / BASE
(d87ab13 emitter, same runtime), 9 ABBA reps. Tables: `benches/results/evidence/plan-cost/sweep/`; raw:
`raw-plan-cost-sweep.tar.gz` (release `evidence-raw-2026-11-09`, `RAW-manifest.tsv`).

## Verdicts

| rung | rtx-4090 | rtx-5090 | h100-pcie / rtx-3090 |
| --- | --- | --- | --- |
| `plan-cost` | **gated OFF** (stopgap row; mixed: 24 W / 8 L / 13 null) | **gated OFF** (stopgap row; mixed: 23 W / 34 L / 20 null) | no emission change (no plan rates) |
| `linear` | kept ON: emits nothing different on any of 191 programs (plan-cost on or off) | **gated OFF**: fires on 2 programs, both null (`pt_indep_4096` 0.994, `pt_diamond_4096` 0.999 ON/BASE) | no emission change |
| step 9 (delete the 7 / 14 plan rows) | **open**: hold-out 22/33 (67 %) < 90 % | **open**: 27/43 (63 %) < 90 % | — |
| `eff` rates in `profile.rs` | unchanged: the fit lands on the grid edge (100, 1) | unchanged: grid edge (1000, 12) | — |

Main ships the rules: the new default emits the sweep's OFF arm on all 112 changed programs on both parts (emission
diff, kernel ids), and OFF / BASE is 1.000 (4090) / 0.997 (5090) geomean. One program's OFF emission differs from
BASE and is unmeasured: `px_diamond_128` on 5090 (no row in the sweep).

## End check: intended vs realized

| rung / arch | intent (`benches/gpu/intent.py`) | realized (`evidence-<arch>.tsv`, nsys `ops-<arch>.tsv`, ptxas) | time ON / BASE |
| --- | --- | --- | --- |
| plan-cost 4090 | plan line carries pick, ns, runner-up, crossover; pick ns ≤ runner-up; `_512` chains: zero passes → 0, fewer ops, no spills | 240 ok; ops 3→2 indep/mlp/fanout, 4→2 twin, 5→3 diamond, 6→4 mlp2; zero passes 1–2 → 0; idle gaps 6–8 → 0.6–2 µs; 0 spills | geomean 1.026 (45 changed); `_512` 0.77–0.92; M=1 GEMV 2.2–3.6×, decode 1.94×, `pj_m16_n2048` 1.25× |
| plan-cost 5090 | same | 364 ok; same op cuts on `_512`; 0 spills | geomean 1.033 (77 changed); `_512` 0.70–0.95; 2048 sizes 1.07–1.16, 4096 1.02–1.035, `mm_128x1024x2048` 1.64×, M=1 2–3.4×, decode 1.32× |
| linear 5090 | site table where it beats the sites' plans | 8 ok, 4 n/a | null on its 2 programs |

The 3 MISMATCH rows per part (`skinny_8x4096x256_f64` hot-loop count 0) belong to no plan-cost rung. The checker now
gates on the plan line's own rungs (`evidence.py` fix), so plan-cost / linear intents are evaluated.

## Gap: Mapal / best other leg, BASE → ON (chains)

| chain | 4090 512 | 4090 1024 / 2048 | 5090 512 | 5090 1024 | 5090 2048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| indep | 1.242 → 1.072 | 1.361 / 1.205 (=) | 1.754 → 1.294 | 1.506 (=) | 1.103 → 1.178 |
| twin | 1.258 → **0.970** | 1.336 / 1.213 (=) | 1.711 → 1.193 | 1.368 (=) | 1.073 → 1.248 |
| mlp | 1.165 → 1.070 | 1.366 / 1.207 (=) | 1.333 → 1.150 | 1.355 (=) | 1.089 → 1.195 |
| mlp2 | 1.365 → 1.239 | 1.440 / 1.230 (=) | 1.859 → 1.766 | 1.655 (=) | 1.097 → 1.202 |
| fanout | 1.169 → 1.067 | 1.338 / 1.221 (=) | 1.384 → 1.232 | 1.322 (=) | 1.162 → 1.263 |
| diamond | 1.249 → 1.033 | 1.308 / 1.220 (=) | 1.524 → 1.208 | 1.412 (=) | 1.085 → 1.195 |

(=) the plan equals BASE's. 4090 box: Ryzen 3 3300X host (not E1's), so library legs differ from earlier sweeps.

## What moved and why

**Wins (`_512` chains, both parts).** The model prices launches (`gap_ns`) and zero passes (`zero_ns`), so it
drops atomic combines for unsplit / workspace plans and absorbs folds: 4090 `pt_indep_512` 3 ops / 1 zero pass /
6.6 µs idle → 2 / 0 / 0.6 µs, span 38.7 → 29.2 µs; kernel time 30.6 → 28.6 µs. The launch side of the model is right.

**Losses: the kernel side picks grids that leave SMs idle.** Plan lines (`; mapal.cand`, rates as shipped):

| class | program / arch | ON pick (pred ns) | rules' plan (pred ns) | measured ON / OFF | the term that is wrong |
| --- | --- | --- | --- | --- | --- |
| A DRAM feed | `mm_1x1024x4096` 4090 | naive splitk2, 8 blocks on 128 SMs, D-bound 17 127 → 18 340 | atomic32, 128 blocks → 20 655 | 84.0 / 30.7 µs | `D / dram` is the chip's bandwidth at any grid: 8 SMs pull ≈ 25 B/ns each (16.7 MB / 84 µs / 8) |
| A | `pt_decode_1024_256` 4090 | GEMV sites `none` / splitk4 on 4 blocks; tile site 1 block/SM, bk 64 | atomic32 / atomic128 (128 blocks), tile 2 blocks/SM atomic2 | 46.5 / 23.7 ms | same, per decode step |
| A | `pt_pj_m16_n2048_k7168` 4090 | `none`, 128 blocks, 1 block/SM, wps 0.5, D-bound 60 459 → 61 059 | atomic4, 512 blocks, wps 2 → 62 728+ | 70.7 / 56.7 µs | same: 2 warps per SM cannot keep 29 MB of weights in flight |
| B per-SM balance | `pt_mlp_2048` 5090 | 2nd GEMM `none`, 256 tiles, 1 wave → 1 097 965 | stream-K 338×97 → 1 118 192 | 0.617 / 0.563 ms | 256 tiles on 170 SMs: 86 SMs run 2 blocks, 84 run 1; the wave term averages (1.51 per SM) instead of loading the busiest SM (2). 4090: 256 = 2·128, no imbalance, plan unchanged |
| C latency depth | `mm_128x1024x2048` 5090 | splitk4, 7 blocks/SM, wps 1.5, F 35 541 → 37 693 | atomic8, 13 blocks/SM, wps 3.0, F 34 833 → 39 376 | 73.7 / 45.0 µs | `eff(wps·acc)` saturates (eff_half 1): 1-warp blocks, bk 8, 1 stage are latency-bound and scale ~linearly with wps; model says F equal |

Classes A and C are one root: the rate a site achieves is bounded by the bytes each SM keeps in flight
(warps × stages × panel bytes per k-step) over memory latency; the model has a whole-chip DRAM roofline and an
FFMA-efficiency curve, neither of which sees resident warps per SM below saturation. B is site-geometry's open G4.

## Hold-out (design §3: split by program hash, stratified by rung class; `eff` fitted on the fit half only)

Pool: site-geometry probes + this sweep's ladder and changed programs (`@pc` rows; the same program stays in one
half). `benches/results/evidence/plan-cost/holdout-<arch>.tsv`, `decisions-<arch>.tsv`, `compiler-<arch>.tsv`.

| arch | decisions fit / hold | rates | hold-out picks | pairwise signs | median \|log err\| | compiler picks (hold-out) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| rtx-4090 | 35 / 33 | seed eff (475, 6) | 18/33 (55 %) | 188/261 (72 %) | 0.075 | 16/25 (64 %), 8 unknown |
| | | fitted (100, 1) | 22/33 (67 %) | 206/261 (79 %) | 0.108 | |
| rtx-5090 | 41 / 43 | seed (150, 1) | 24/43 (56 %) | 117/187 (63 %) | 0.170 | 18/32 (56 %), 11 unknown |
| | | fitted (1000, 12) | 27/43 (63 %) | 126/187 (67 %) | 0.130 | |

The fits sit on opposite grid edges and the 4090's error grows: two `eff` parameters cannot carry the missing
terms, so the rates stay as shipped (no re-timing owed).

## Missing facts (ranked by the regressions they explain)

1. **In-flight bytes per SM (Little's law)** — `TargetProfile` fact: memory latency (DRAM, L2) and the bytes a
   resident warp keeps in flight per k-step (`stages · (bm + bn) · bk · e / warps`); the site's feed rate is
   `min(dram, Σ_SM inflight / latency)` and its FFMA rate is capped the same way. Explains A + C: 4090 M=1 2.2–3.6×,
   decode 1.94×, `pj_m16` 1.25×; 5090 M=1 2–3.4×, decode 1.32×, `mm_128x1024x{1024,2048}` 1.40× / 1.64×.
2. **Per-SM load of a wave (site-geometry G4)**: makespan of the busiest SM, `⌈B_w / sms⌉` blocks at that SM's wps,
   not `B_w / sms`. Explains every 5090 2048 (1.07–1.16) and 4096 (1.02–1.035) loss and why the 4090 is unchanged
   there. The stream-K makespan rule had it; the cost model's wave term dropped it.
3. **k-depth of the panel loop (G3)**: panel trips per partition `(k/S)/bk` × per-trip latency at the site's
   stages. Folded into 1 for the 1-stage tiles; needed for split counts that shorten the loop below the pipeline.

Each needs its property test over `testgen` programs before the gate rows go; then re-time only the rows whose
emission changes (`benches/gpu/changed.py`).

## plan-cost v2 judge (2026-11-09): calibrated `lat_ns`, refit `eff`, verdict

Box runs `920d3df` (rtx-4090, rtx-5090): `benches/results/evidence/plan-cost/v2/`, kernel tables
`benches/results/evidence/tile-efficiency/`. ON = `--gpu-on=plan-cost,linear` at the v2 rates (4090 `(700, 6, 150)`,
5090 `(1000, 8, 300)`); tolerance per program max(2 %, null arm, half the spread).

### Rates

`lat_ns` is the model's own quantity (`site_ns`: feed rate `Σ fly·panel / lat_ns`), so it is calibrated as bytes in
flight over the achieved rate, below 90 % of peak (`benches/gpu/calib.cu`, `B-bw-<arch>.tsv`, valid points):

| arch | DRAM median [IQR] (points) | L2 median [IQR] (points) | idle chase DRAM / L2 | shipped `lat_ns` |
| --- | --- | --- | --- | ---: |
| rtx-4090 | 380 [369–407] (87) | 240 [233–281] (113) | 146 / 106 | 150 → **380** |
| rtx-5090 | 510 [500–528] (111) | 290 [282–350] (129) | 195 / 141 | 300 → **510** |

DRAM, because the term's cap is the DRAM roofline (`D / dram`); the L2 value is what the open residency fact would
select. `dram_bytes_per_ns` / `l2_bytes_per_ns` stay at E1 (B agrees within 2.3 % / 8.8 % on the 4090, 0.7 % / 2.8 %
on the 5090; they feed shipped rungs). `eff` refit with lat fixed (`plancost.py --lat N`, 6 program folds, same pool
as "Review fixes"): 4090 `(525, 0)` (folds 400–600, eff_half 0–1), 5090 `(800, 6)` (folds 700–950, 6–8), no rate on a
grid edge (the 5090 fit sat on eff_max 1000 before). `holdout-<arch>.tsv`, `decisions-`, `compiler-<arch>.tsv`.

| arch | decisions | no-terms hold picks mean (min–max) | terms, lat calibrated | terms − no-terms per fold | compiler picks, fold-0 hold |
| --- | ---: | ---: | ---: | --- | --- |
| rtx-4090 | 68 | 71.2 % (61.8–81.8) | 74.1 % (65.7–81.8) | −1, +2, 0, +4, 0, +1 | 9/14 (19 unknown) |
| rtx-5090 | 84 | 58.4 % (41.0–71.1) | **78.6 % (71.4–85.7)** | +9, +7, +9, +7, +3, +15 | 13/17 (24 unknown) |

Fixing lat at the calibrated value beats the free-lat fit on both (4090 72.6 → 74.1 %, 5090 73.7 → 78.6 %).

### Box list and blind set (ON vs shipped, at the v2 rates)

| arch | box list timed: geomean, W / L / null | blind set (40 programs) differing: geomean, W / L / null | blind score |
| --- | --- | --- | --- |
| rtx-4090 | 25: 0.945, 17 / 5 / 3 | 8: 0.960, 2 / 2 / 4 | faster 2 of 4 decisive (6/8 within null) |
| rtx-5090 | 47: 1.033, 9 / 13 / 25 | 19: **0.788**, 10 / 6 / 3 | faster 10 of 16 decisive (13/19 within null) |

0 `VALUE-FAIL`. Losses: 4090 `pt_pj_m1_n2048_k7168` 1.50, `mm_1x1024x4096` 1.40, `pt_decode_1024_256` 1.32,
`pt_pj_m1_n1024_k4096` 1.24, `mm_1x2048x7168` 1.16; blind b08 1.10, b22 1.04. 5090 `mm_1x1024x4096` 1.74,
`pt_pj_m16_n2048_k7168` 1.44, `mm_1x2048x7168` 1.36, `mm_16x2048x7168` 1.29, `pt_pj_m16_n1024_k4096` 1.27,
`mm_128x256x256` 1.24, `pt_pj_m1_n1024_k4096` 1.17; blind b37 1.29, b08 1.23, b22 1.18, b17 1.13, b23 1.10, b38 1.09.

### Verdict: both `GPU_RUNGS_OFF` plan-cost rows stay (open, missing facts)

Neither arch meets "no loss beyond null on the box list and the blind set agrees". The losses are the skinny
cells (M = 1 GEMV, M = 16, decode); the wins are the `_512` chains, `mm_128x1024x*`, `mm_512x512x512`, attention.

1. **L2-resident vs DRAM operand latency.** Calibrated: the effective latency of an L2-resident stream is 0.63×
   (4090) / 0.57× (5090) the DRAM one, and loaded latency climbs steeply toward peak (4090 218 ns at 50 %, 2437 ns at
   90 %; 5090 472 / 1154 ns): one `lat_ns` cannot price both. Same plan, same shape, different residency:
   `pt_pj_m1_n1024_k4096` 1.24 vs `mm_1x1024x4096` 1.40 (4090). Needed: the operand bytes that persist in L2 across
   trips (`ResidentPlan` + `l2_bytes` + `loop_plan` reuse) → `l2_lat_ns`, `dram_lat_ns` weighted by missed bytes.
   The M = 1 losers' own SASS was not captured in v2, so their per-kernel cause is **open**.
2. **Shared-memory bytes per FMA.** Calibrated LDS: 64 / 126 / 127 B per clock per SM (.32 / .64 / .128; both parts),
   i.e. one non-broadcast warp LDS.128 per 4 clocks against 4 warp FFMA per clock: a tile below 16 FFMA per LDS.128 is LDS-bound.
   SASS (`D-sass-<arch>.tsv`): Mapal's 512/1024 hot loops have 16 accumulators and 7.6–10 FFMA per LDS (cap
   0.48–0.63 of FFMA peak); the cutlass 128×128×8 loop has 64 and 16 (cap 1.0); the 5090 2048 loop reaches 64 / 16
   and ties or wins. The skinny 5090 cells sit at that cap: 128×1024×1024 0.57, 128×2048×7168 0.54, 16×1024×4096
   0.62 of the best library. Needed: `eff` capped by `lds_bytes_per_clk` against the tile's FFMA per LDS
   (`(tm + tn) / (tm · tn)` at its load width) — a `TargetProfile` fact plus the tile's shape, no new IR fact.

Unresolved in this run: `evidence.tsv` intent mismatches (4090 104, 5090 192) not triaged; the 4090 chase vs cold
sweep gap (146 vs 253 ns) unexplained; probe timings single-shot over 5 reps.

### Re-timing owed

Shipped emission is unchanged (`site_ns` is only reached when `plan-cost` is ON; both parts ship it OFF). ON at the
new rates vs ON at the v2 rates over the 243 bundle programs (`v2/retime-<arch>.tsv`):
4090 **25** of 243 moved, 5090 **37**. They hold every 4090 loss above (5/5 box list, 2/2 blind) and
6/7 + 4/6 of the 5090 ones (not `mm_128x256x256`, b17, b23), so the v2 losses do not judge the calibrated rates: the
next box run times these rows, ON vs shipped (host IR moved nowhere else).

- rtx-4090: `decode_1024_256`, `deep_16x1024x4096`, `mm_128x2048x7168`, `mm_1536`, `mm_1536x1536x1536`, `mm_16x1024x4096`, `mm_1x1024x4096`, `mm_1x2048x7168`, `pt_bl_b02`, `pt_bl_b04`, `pt_bl_b08`, `pt_bl_b11`, `pt_bl_b12`, `pt_bl_b13`, `pt_bl_b18`, `pt_bl_b22`, `pt_bl_b24`, `pt_bl_b29`, `pt_bl_b31`, `pt_bl_b39`, `pt_decode_1024_256`, `pt_pj_m1_n1024_k4096`, `pt_pj_m1_n2048_k7168`, `pt_pj_m128_n2048_k7168`, `pt_pj_m16_n1024_k4096`
- rtx-5090: `chain_256`, `deep_16x1024x4096`, `deep_16x2048x7168`, `mm_128x512x256`, `mm_16x1024x4096`, `mm_16x2048x7168`, `mm_1x1024x4096`, `mm_1x18432x7168`, `mm_1x2048x7168`, `pt_bl_b03`, `pt_bl_b04`, `pt_bl_b05`, `pt_bl_b08`, `pt_bl_b12`, `pt_bl_b18`, `pt_bl_b22`, `pt_bl_b25`, `pt_bl_b32`, `pt_bl_b33`, `pt_bl_b34`, `pt_bl_b37`, `pt_bl_b38`, `pt_bl_b39`, `pt_diamond_2048`, `pt_indep_2048`, `pt_pj_m1_n1024_k4096`, `pt_pj_m1_n18432_k7168`, `pt_pj_m1_n2048_k7168`, `pt_pj_m16_n1024_k4096`, `pt_pj_m16_n2048_k7168`, `pt_rg_indep_2048`, `pt_rg_mlp2_2048`, `pt_rt_indep_2048`, `pt_rt_mlp2_2048`, `rm_256`, `share_256`, `skinny_1x16384x512`
