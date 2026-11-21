# local-rtx-5080 — large GEMM: the libraries' geometry as rules over properties (2026-11-21)

Machine: Threadripper 9970X + RTX 5080 (sm_120, 84 SMs, 100 KiB smem / SM, 64 MiB L2, 16 GB, 2.45 GHz FFMA clock
under load), CUDA 13 (cuBLAS 13.8), clang/LLVM 23, branch `local-rtx-5080`. Protocol: every timed step behind the
gate (load < 3, no `MapleStory.exe`, the shared bench lock), Tctl 33–60 C at the starts; cells in one `tile_bench`
process with cuBLAS / CUTLASS legs (30 reps a leg, value gates first); the suite `gpu_run.sh` ON / OFF / ON
(`LADDER=1`); the model-scale GEMM rows `model_scale_run.sh` (pinned CPUs 0–7, arms interleaved, `gemm_lib` cuBLAS /
cuBLASLt legs on the same box); ABBA 16 processes an arm (candidate 0, `MAPAL_GPU_TUNE=0`); CUPTI by
`benches/local/ktrace.cpp`. OFF = `--gpu-off=wide-fma,k-fill,streamk-fill` (on this part the pre-step emission:
candidate-0 kernels byte-identical to HEAD a354f221 but for metadata numbering).

## 1. What the libraries run (captured from the launches, CUPTI)

| cell | cuBLAS / CUTLASS kernel | grid, block, regs, smem | kernel us | Mapal before (HEAD) | kernel us |
|---|---|---|---:|---|---:|
| 4096³ | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` (256 × 128 tile, bk 8, 4 `cp.async` stages) | 256×2×**3** (split-K 3: 1536 CTAs = 18.3 waves at 1 CTA/SM), 256 thr, 210 regs, 48 KB | 3544 | stream-K, `8 × 8` thread tile, 256 thr, 168 CTAs, 128 regs, 16.6 KB, register staged | 3954 |
| 4096³ | CUTLASS stock `MmaPipelined` 128 × 128 × 8, 2 buffers | 32×32, 256 thr, 118 regs, 16.6 KB (6.1 waves, no split) | 3624 | | |
| 128 × 4096 × 4096 | `cutlass_80_simt_sgemm_128x64_8x5` (5 stages) | 16×4×**5** (split-K 5: 320 CTAs, one wave at 4 CTA/SM), 128 thr, 128 regs, 30 KB | 138 | `4 × 4` tile (warp-fill), 64×2 = 128 CTAs, no split, 77 regs | 212 |
| 16 × 2048 × 7168 | `cutlass_80_simt_sgemm_128x32_8x5` + `splitKreduce` | 8×2×**9** (split-K 9: 144 CTAs × 4 warps), 80 regs, 25.6 KB | 63.4 + 1.4 | stream-K over **84 one-warp CTAs** (one an SM), 231 regs | 86.8 |
| TF32 128 × 4096 × 4096 | `cutlass_80_tensorop_s1688gemm_64x64_16x6` | 16×8×**8** (split-K 8: 1024 CTAs), 6 stages, 48 KB | 101 | `mma.sync` 2 × 2 warp tiles, 128 CTAs, 2 stages, 37 KB, no split | 122 |
| TF32 2048 × 28672 × 4096 | `cutlass_80_tensorop_s1688gemm_128x128_32x3` | 128×28, 128 thr, 234 regs, **96 KB (3 stages bk 32, 1 CTA/SM)** | 7982 | `mma.sync` 4 × 8, bk 16, 2 stages, 37 KB, 2 CTA/SM | 9072 |

What separates them is not the inner loop (Mapal's stream-K mainloop at m 2048 already beats cuBLAS's) but three
geometry choices: (a) the **128-accumulator thread tile** (8 × 16; cuBLAS's 256 × 128 / 256 threads) — fewer shared
loads per FFMA; (b) **filling the part along `k`** — every library kernel above splits K (3, 5, 8, 9) so its grid
fills the SMs at the full tile; Mapal shrank the tile (`4 × 4`: 2 `LDS.128` per 16 FFMA, LDS-bound) or ran stream-K
at one block an SM; (c) for TF32, deeper pipelines (3–6 stages) and split-K — open (§6).

## 2. Rules over facts and properties (each with an OFF arm)

| rule | property / fact it reads | where | OFF arm |
|---|---|---|---|
| **`wide-fma` on compute-rich parts**: the whole-file `8 × 16` tile on a contract face where the part issues ≥ `FILL_BALANCE_PM` (45) FFMA per DRAM byte (5080 54.9, 4090 81.9, 5090 58.5; off: 3090 38.0, H100 25.1, A100 12.5 — every measured loss), or where an SM's DRAM share is scarce (as before) | `Gpu::ffma_per_dram_byte_pm` | `rules.rs` | `--gpu-off=wide-fma` |
| … and only where its grid gives the part its fill target at its own geometry (`blocks × warps ≥ fill_warps`): half the S52 tile's warps a block, it neither streams nor splits otherwise (256 × 6144 × 8192: 96 four-warp blocks; the 5090's 2048²) | `Gpu::fill_warps` | `profile.rs` `tile_split_at` | `--gpu-off=streamk-fill` |
| **`streamk-fill`**: stream-K walks a launched tile grid only where that grid fills the part at its own geometry; a grid short of `fill_warps` (16 one-warp tiles) takes the `fill-wave` split, which reaches it | the launched grid's warps vs `Gpu::fill_warps` | `func/gpu.rs` `combine` | `--gpu-off=streamk-fill` |
| **`k-fill`**: where the rules shrank the tile or its threads to fill the part (`warp-fill`, step-4 halving), the full S52 tile is kept and the part is filled along `k` by stream-K over **every slot the tile holds** (`sms × residency` at its registers / threads — not `grid-share`'s one block an SM: 128 × 6144 × 4096 84 CTAs 0.231 ms, 167 CTAs 0.205), where each CTA folds ≥ `K_FILL_PANELS` = 32 panels (a CTA's fixup — two tile pieces of atomic adds — is ~2.5 panels of its FFMA time here; measured winner 48.8, loser 24.4 panels: 1024³ keeps the shrunk tile). Index width the filled class's (`idx32`) | `TargetProfile::tile_capacity`, `fma_tile_side`, `fill_grid`'s `wps` | `func/gpu.rs` `k_fill`, `combine`, `resized` | `--gpu-off=k-fill`; `--geom=kfill=<panels>` forces the floor (measurement / emulator) |

Replay (`rung_replay.tsv`, test `rung_rules_replay`): three new lines (wide-fma / k-fill / streamk-fill on the
rtx-5080), all picked: **105 picked / 67 null / 9 mixed / 16 open** (was 102 / 67 / 9 / 16). The 5090's
`wide-fma` line stays "mixed" (its 4096 disjoint loss and 8192 gain, measured before stream-K and the fill check);
the rule now ships it on there at 4096 and off at 2048² (its 256 four-warp blocks are short of 1360).

## 3. Intent → realized (SASS / ptxas, final emitter; evidence.py with the 5080's limits)

| kernel (cell) | intent | realized (ptxas -v, `cuobjdump -sass` hot loop) |
|---|---|---|
| 4096³ ON (`wide-fma`, stream-K 168 × 128 thr) | 128 accumulators, 6 `LDS.128` per 128 FFMA, 2 blocks / SM | 233 regs, 0 spill; hot loop 2048 FFMA / 96 LDS.128 (21.3 FFMA per LDS vs 16), 8 LDG.128, 20 STS, 2 BAR; 128 REDG |
| 4096³ OFF | — | 128 regs; 1024 FFMA / 64 LDS (16 per LDS) |
| 128 × 4096 × 4096 ON (`k-fill`) | `8 × 8` × 256 thr, stream-K 168 CTAs (2 / SM), ≥ 32 panels / CTA | 128 regs (cap), 1024 FFMA / 64 LDS.128 hot, iters 48; 24 B spill **outside** the panel loop (stream-K piece setup: 3 STL / 3 LDL, 0 in the hot loop) |
| 128 × 4096 × 4096 OFF | — | `4 × 4`, 77 regs, 256 FFMA / 32 LDS (8 per LDS: LDS-bound) |
| 16 × 2048 × 7168 ON (`streamk-fill` → fill split S = 56) | 896 one-warp CTAs (≥ 672 warps) | 155 regs, 0 spill, 512 FFMA / 32 LDS hot, 64 REDG |
| 16 × 2048 × 7168 OFF | — | stream-K 84 CTAs (one warp an SM), 231 regs |

Intent registry (`intent.py`): new `k-fill` (full tile, slots filled ≥ 90 % in one wave, ≥ 32 panels, residency, no
local memory in the hot loop) and `streamk-fill` (stream-K grid fills at its own geometry); `evidence.py` knows the
two rungs and the 5080's limits (`PARTS`; its `occ.sms` was 0 before, which failed every 5080 `streamk` /
`fill-wave` one-wave check). Static check of the shipped bundles: k-fill 15 / 15 ok, streamk-fill 20 / 20, streamk
122 / 122, fill-wave 21 / 21 one-wave ok; `wide-fma` mismatches: "no spill" only on exact-face stems whose site table
forces the `+wide-fma` alternative (50, not the rules' plan), and "ptxas gave the thread what the tile holds" 8 ×
(ptxas used 204–206 of the 208 the estimate counts — conservative by ≤ 4 registers, no spill).

## 4. Results

### Cells (`tile_bench`, one process a cell, ms medians; ON / OFF the same process)

| cell | OFF | ON | cuBLAS | ON / cuBLAS | before | explanation (CUPTI kernel us OFF → ON; cuBLAS) |
|---|---:|---:|---:|---:|---:|---|
| 4096³ | 4.171 | **3.841** | 3.787 | **1.01** | 1.14 | wide tile 3954 → 3663 (cuBLAS 3544, 1.03) |
| 2048 × 4096 × 4096 | 2.010 | **1.908** | 1.821 | 1.05 | 1.10 | wide tile |
| 2048³ | 0.508 | **0.487** | 0.459 | 1.06 | 1.11 | wide tile |
| 16 × 2048 × 7168 | 0.1126 | **0.0881** | 0.0799 | 1.10 | 1.41 | 86.8 → 59.4 us kernel (cuBLAS 63.4 + 1.4): faster than the library's kernel; the event gap is the zero pass + launch |
| 16 × 18432 × 7168 | 0.6349 | **0.6144** | 0.6185 | **0.99** | 1.03 | stream-K 168 one-warp CTAs → S = 7 split (1008) |
| 128 × 4096 × 4096 | 0.2191 | **0.1516** | 0.1434 | 1.06 | 1.53 | 212 → 137 us (cuBLAS 138: **0.99** kernel) |
| 128 × 6144 × 4096 | 0.3236 | **0.2109** | 0.1864 | 1.13 | 1.74 | k-fill 167 CTAs |
| 128 × 4096 × 14336 | 0.7578 | **0.4441** | 0.4076 | 1.09 | 1.86 | k-fill |
| 128 × 2048 × 7168 | 0.2458 | **0.1413** | 0.1393 | **1.01** | 1.76 | k-fill |
| 128 × 28672 × 4096 | 0.8500 | **0.8049** | 0.8008 | **1.01** | 1.06 | wide tile (224 blocks fill) |
| 256 × 6144 × 8192 | 0.7148 | 0.7148 | 0.6431 | 1.11 | 1.11 | unchanged by design (S52 + stream-K: the wide grid would be short) |
| 1024³ | 0.0901 | 0.0901 | 0.0819 | 1.10 | 1.10 | unchanged (24 panels a CTA < 32: shrunk tile) |

Wide vs S52, with and without stream-K (cells-tail.txt): 4096³ 3.875 / 4.270 (stream-K) and 4.075 / 4.399
(none); 2048 × 4096² 1.883 / 1.996 and 2.045 / 2.246; 2048³ 0.487 / 0.508 and 0.528 / 0.610 — the whole-file tile
wins with and without stream-K in the cells.

### Suite (35 programs, ON1 / OFF / ON2, `iter ms`, shipped build = tuned pick; best f32 competitor)

Changed beyond the run-to-run null, ABBA where marked (candidate 0, 16 processes an arm, ON / OFF):

| workload | OFF | ON (ON1 / ON2) | OFF / ON | ABBA ON / OFF | best f32 (leg) | ON / best | explanation (CUPTI) |
|---|---:|---:|---:|---:|---|---:|---|
| fanout_2048 | 1.081 | 0.985 / 0.986 | 1.097 | | 0.913 (streams) | 1.08 | wide tiles (epilogue + stream-K sites) |
| rowsum_4096 | 4.051 | 3.728 / 3.742 | 1.085 | **0.926** | 3.502 (torch-compile) | 1.07 | stream-K tile 3896 → 3589 us |
| softmax_rows_4096 | 4.106 | 3.798 / 3.795 | 1.082 | | 3.75 (fused) | 1.01 | as rowsum_4096 |
| mlp_2048 | 1.001 | 0.944 / 0.945 | 1.059 | **0.944** | 0.913 (lt) | 1.03 | epilogue tile 522 → 494 us, stream-K tile 460 → 431 us |
| rowsum_2048 | 0.483 | 0.458 / 0.463 | 1.048 | | 0.463 (eager) | 0.99 | wide tile |
| mlp2 / rt_mlp2 / twin / diamond / indep_2048 | | | 1.035–1.042 | | lt / torch-compile | 1.01–1.10 | wide tiles |
| pj_m16_n18432_k7168 | 0.616 | 0.605 / 0.604 | 1.019 | | 0.614 (torch-compile) | **0.98** | fill split S = 7 (rules 0.604 vs OFF rules 0.626) |
| pj_m16_n2048_k7168 | 0.0487 | 0.0496 / 0.0485 | 0.993 | **0.586** | 0.0778 (torch-compile) | 0.63 | rules 0.0476 vs OFF rules 0.0814: the tune found the split before; candidate 0 now is it |
| epi_4096 | 3.583 | 3.679 / 3.674 | **0.975** | **1.027** | 3.577 (lt) | 1.03 | epilogue tile 3565 → 3680 us: **loss** (open, §6) |
| loop_4096 | 27.68 | 28.38 / 28.27 | **0.977** | **1.024** | 28.55 (lt) | 0.99 | the same epilogue site, 4096³ in a loop (open) |
| attn_1024_64, pj_m1_n2048_k7168, indep_512, mlp_512 | | | 0.96–1.03 | 1.008, 0.957, 1.021, — | | | candidate 0 identical; tuned picks differ (spread 14–45 %): null |

The other 17 programs: candidate 0 byte-identical modulo kernel names, ratios 0.996–1.007 (null 0–0.8 %; the
tabled programs' site tables differ only by the `+wide-fma` axis's alternative). Cells in the suite run: 4096³
4.155 → 3.80 / 3.82 (cuBLAS 3.663, **1.04**), 16 × 2048 × 7168 0.1124 → 0.0881 (**1.10**). Value gates: 0 failures
in each of the three runs (339 / 302 / 339 gate lines).

### Model-scale GEMM rows (projections and the 2048 prefills; ms, shipped = tuned; rules = candidate 0)

| row | m | ON shipped | rules | OFF rules | OFF / rules | before (shipped, `model-scale.md`) | best library (leg) | ON / library |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| QKV n6144 | 128 | **0.2086** | 0.2083 | 0.3308 | 1.59 | 0.2449 (1.36×) | 0.1802 (cuBLAS fresh) | 1.16 |
| O n4096 | 128 | **0.1445** | 0.1446 | 0.2162 | 1.50 | 0.1909 (1.41×) | 0.1352 (cuBLAS fresh) | **1.07** |
| gate/up n28672 | 128 | 0.8082 | 0.8091 | 0.8561 | 1.06 | 0.8053 (1.07×) | 0.7496 (Lt) | 1.08 |
| down k14336 | 128 | **0.457** | 0.4512 | 0.7728 | 1.71 | 0.4824 (1.20×) | 0.4014 (cuBLAS fresh) | 1.14 |
| QKV | 16 | 0.0956 | 0.0973 | 0.1792 | 1.84 | 0.0954 | 0.0881 (Lt fresh) | 1.09 |
| O | 16 | 0.0529 | 0.0617 | 0.0996 | 1.61 | 0.0572 | 0.0553 (cuBLAS fresh) | **0.96** |
| gate/up | 16 | 0.5483 | 0.5459 | 0.5729 | 1.05 | 0.5766 | 0.5222 (Lt fresh) | 1.05 |
| down | 16 | 0.2668 | 0.2662 | 0.3686 | 1.38 | 0.2576 | 0.3154 (cuBLAS fresh) | **0.85** |
| QKV / O / gate/up / down | 2048 | 2.653 / 1.843 / 11.81 / 6.437 | = | 2.779 / 1.941 / 12.62 / 7.532 | 1.05 / 1.06 / 1.07 / 1.17 | 2.648 / 1.840 / 11.79 / 6.433 | 2.633 / 1.785 / 12.23 / 6.42 | 1.01 / 1.03 / **0.97** / 1.00 |
| m = 1 (4 rows) | 1 | 0.1 / 0.0497 / 0.558 / 0.2545 | = | = | 0.996–1.014 | unchanged | | 1.10–2.21 (§6) |
| prefill T2048 | | **8.626** | 9.489 | 10.18 | 1.07 | 8.821 | 8.322 (torch-compile, 11-20) | 1.04 |
| block prefill T2048 | | **27.37** | 28.69 | 29.81 | 1.04 | 32.38 | 27.93 (torch-compile, 11-20) | **0.98** |
| TF32 (8 rows) | 128 / 2048 | 0.24 / 0.135 / 0.697 / 0.456; 2.01 / 1.42 / 9.04 / 4.90 | — | — | (no change) | the same | cuBLAS TF32 fresh | 1.13–1.92 (§6) |

"before" is the projection table's shipped (tuned) column of `model-scale.md` (2026-11-21, same box); the m 16 and
m 2048 rows' tuned picks had already found the split / the wide candidate at run time — the rules now take them
without tuning (OFF / rules 1.05–1.84). Value gates: every arm within D2 of its exact-face device oracle
(`.xref` / `.xabsref`), 229 gate lines, 0 failures; null 0.0–4.1 %.

## 5. Tests

- `tests/large_gemm.rs` (new; generator `testgen::gemm_prog_strategy`: skinny / under-filled deep / filled / odd
  extents, `b` row-major or transposed, integer operands in −2..2):
  - `large_gemm_rules_on_generated_shapes`: 96 generated GEMMs (plans only, model-scale extents): every stream-K
    kernel is `k-fill`'s (full tile, ≥ 90 % of the slots in one wave, ≥ 32 panels) or fills at its own geometry;
    each OFF arm's plan as specified. Coverage: 6 k-fill kernels (6 shrunk under the OFF arm), 34 filled stream-K, 10
    stream-K grids declined for the fill split, 24 whole-file tiles (counters asserted > 0).
  - `large_gemm_values_through_the_emulator`: 12 generated programs at runnable extents (k-fill through
    `--geom=kfill=1`), the interpreter as oracle, ON under `MAPAL_EMU_SERIAL=1`, `=2` and `MAPAL_EMU_REVERSE=1`,
    the OFF arm too: 5 k-fill + 6 fill-split programs, 33 emulated runs, all equal.
- Unit: the chooser's measured cells (the 3090's 16 × 18432 × 7168 now takes the fill split, its OFF arm the
  one-wave stream-K), the whole-file tile's fill (5080 / 4090 keep it at 2048², the 5090 not), the replay.
- Pinned: `c0-rules.tsv` candidate-0 golden with the two site rules off, regolded for `wide-fma` (7 rtx-5090
  default / contract rows move — the property rule); `tiled_combines_emulated` pins the S63 grid (site rules off).
- Workspace (`cargo test --release --workspace --no-fail-fast`, 111 binaries): **1431 passed, 1 failed** — the
  pre-existing `streams.rs pdl_is_hopper_and_later_only` (below).

## 6. Open

- **`wide-fma` on 6-wave epilogue tiles** (mixed): epi_4096 / loop_4096 lose 2.4–2.7 % (ABBA 1.027 / 1.024; CUPTI
  3565 → 3680 us), while the 1.5-wave epilogue tile (mlp_2048, 522 → 494 us), every stream-K site (−6 to −8 %) and
  the cell without stream-K (4.40 → 4.07 ms) win. Same block tile, waves, raster and stores (16 vs 32 `STG.128` a
  thread for half the threads); the separating fact is not found (candidates: the program's L2 state after its
  operand generation vs the cell's, the four-warp block's latency cover in a 6-wave launch).
- **TF32** (1.13–1.92×): the `mma.sync` tile has no split / stream-K (m 128: 128 CTAs vs cuBLAS's split-K 8 over
  1024) and a 2-stage bk-16 pipeline at 2 CTAs / SM where cutlass runs 3 × bk 32 at one (96 KB). Both need the
  `cp.async` pipeline to take a runtime trip count (the FMA tile's "ponytail": its stream-K runs register staging
  only) — the same change would let the FMA `cp.async` panels stream-K (4096³: async without stream-K 4.15 ms vs
  register-staged stream-K 3.76).
- **Pipeline depth from latency × bandwidth**: measured, not shipped. The skinny site's best plan is a 2-stage
  `cp.async` panel at S = 7 (0.0778 ms, cuBLAS 0.0799) against the fill split's S = 56 (0.0881): the fill target in
  warps (`fill_warps`) over-splits a DRAM-bound site whose few warps keep enough bytes in flight; the in-flight
  bytes the split needs (Little's law at the measured `lat_ns` 573 × 890 B/ns = 510 KB predicts S = 4, measured
  S ≥ 7 with `cp.async`, ≥ 14 register-staged) is the missing fact.
- m = 1 L2-resident GEMVs (1.95–2.21×): the runtime site decision inside the timed region (`model-scale.md`).
- 1024³ (1.10×) and 256 × 6144 × 8192 (1.11×): S52 tile + stream-K / shrunk tile; cuBLAS splits K at 256 × 128.
- Pre-existing failure unrelated to this step: `streams.rs pdl_is_hopper_and_later_only` (expects `pdl` on a
  straight-line program; the cross-arch rule 7eef8687 ships it only on repeating launches).

## Commands

`benches/local/gpu_bundle.sh` (ON: the shipped emitter; OFF: `emit … --gpu-off=wide-fma,k-fill,streamk-fill`),
`LADDER=1 gpu_run.sh` ON / OFF / ON, `model_scale_run.sh` with `PROBE_ARMS="on rules off-wide-fma+k-fill+streamk-fill
null"`, `tile_bench --splitk M N K 30 rtx-5080 <arms>`, ABBA `abba.py`, CUPTI `ktrace.so`. Raw (local, not in git):
`.claude/raw-evidence/local-rtx-5080/large-gemm/` (`commands.txt`, cells, suite runs, model run, ABBA, traces,
static evidence with the 5080's limits).
