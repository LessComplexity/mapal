# cross-arch rules — the rules the four-part round showed wrong, each as a property (2026-11-21)

Branch `local-rtx-5080`. Input: the first cross-architecture round (`README.md` here: snapshot 8bba2a50 on rented RTX
3090 / A100 40GB / RTX 4090 / H100 SXM; raw `.claude/raw-evidence/cross-arch/<part>/`). No rentals in this step: every
change is validated by **replay** against that round's measured ON/OFF arms (and the earlier cited measurements,
`rung_replay.tsv`), by property tests over generated programs, and — where the RTX 5080 has the mechanism — by timed
runs on it. A final cross-arch round validates the shipped rules on the parts.

Raw (not in git): `.claude/raw-evidence/local-rtx-5080/cross-arch-rules/` (calibration, the three-arm PDL run, the
ABBA, the tune runs, the replay scripts and their output).

## Rule → property → replay

| # | rule | before (what the round showed) | property it reads now | replay before → after |
|---|---|---|---|---|
| 1 | `wide-fma`, `async`, `kmajor-a` | FFMA per DRAM byte ≥ 70 ‰ (`BALANCE_PM`): the 250 W 4090 box's measured clock (1.955 GHz) puts it at 63.6 ‰ → all three off, yet ON beat OFF 17 / 3 / 10 programs, 0 losses | **DRAM bytes per µs per SM < 9 200** (`DRAM_PER_SM_LOW`; clock-free) | 4090 at its measured clock: 3 lines lost → picked (17 + 3 + 10 wins kept); every earlier line unchanged (async 8 / 1 / 0 / 0, kmajor-a 6 / 0 / 0 / 0, wide-fma 5 / 0 / 1 / 0) |
| 2 | `pdl` | on wherever the measured programmatic launch is cheaper (H100 734 < 1722 ns): W 6 / null 57 / L 8 on 71 programs, geomean 1.005 | **the module's launches repeat** (a tile, cell or row-fold site in a canonical loop: `loop_structure` / `loop_plan` × `tile_plan` / `cell_plan` / `row_fold`), and the tile kernel's dependent launches at its **exit** (`launch_dependents` at entry only under the new arm `pdl-early`) | H100: straight-line 55 programs off (6 losses avoided — smlp 1.489, indep_512 1.293, blkpre_tf32 1.120, mlp_512 1.069, fir 1.040, pj_m1_n4096 1.034 —, 1 win forgone: pj_m1_n2048 0.964); loops 16 on (gen 0.788, causal 0.790, prefill 0.796, decode 0.922, dec_b1 0.972 kept; rt_mlp2_1024 1.189 and blkdec 1.099 open); geomean vs plain 1.005 → **0.992**. 5080 lines: on-loops picked, short programs mixed → off picked |
| 3 | `row-split` | DRAM bytes per µs per SM ≥ 14 600: on on the H100 SXM, 4 / 4 lose (decode 0.811, gen 0.860, causal 0.866, prefill 0.869; 6144 vs 4098 launches) | the property rule **and** the site's price (`row_split_pays`): the unsplit pass (one SM streaming the row's `k · c` bytes alone, `sm_bytes_per_ns`) saves more than the split's 2 dependent launches (`dep_launch_ns`) | H100 rows (256 × 256 f32: 1.70 µs alone vs 3.44 µs of launches) off: 4 losses → 4 wins (geomean 0.851 of the old); H100 PCIe (launch unmeasured: unpriced) and every other line unchanged |
| 4 | `min-blocks` | off filled at ≤ 32 regs / resident thread: H100 SXM 18 L / 7 W, A100 6 L / 2 W | unchanged — **open** (below) | 3 cross-arch lines recorded open |
| 5 | `warp-fill` | mixed on 3090 / A100 / H100 | unchanged — **open** (below) | recorded mixed (3 lines) |
| 6 | `grid-share`, `panel-regs` | off on the 3090 (75 KiB L2 / SM < 128 KiB) | unchanged — the round **supports** the 3090's off (below) | grid-share 3090 filled: picked; under-filled: mixed; panel-regs: mixed |
| 7 | tune trial | candidate minima of a trial whose clock drifts (+3.5 % a trial run on the capped boxes' large sites) | **steady state first** (candidate 0 back to back until the median of three runs is within 1 % of the three before, ≤ 1 s or 64 runs) and the head-to-head **pairs decide** (product over the 3 adjacent pairs < 0.97³ and > 3 µs) | archived trials (4 parts, 308 sites, the shipped rule reproduced 308 / 308): 5 of 16 steady-state-slower picks reverted (a100 pj_m128_n28672, pj_m2048_n4096_k14336, rowsum_4096; 4090 loop_4096, rowsum_4096), 4 of 82 measured wins reverted (2.4–4.5 %), none added |

`rung_replay.tsv` + `rules::tests::rung_rules_replay`: **194 lines — 102 picked, 67 null, 9 mixed, 16 open** (was 179:
95 / 67 / 4 / 13). The 15 new lines are this round's; the open set is the 13 earlier ones plus the three `min-blocks`.
A line may now carry the box's measured property (`clock_khz=1955131`) and the site's side of a priced rule
(`row_bytes=`, `repeats=`).

## 1. The operand-stream rungs read the DRAM rate an SM shares

The measured clock under load moved only the clock-dependent balance: rtx-4090 81.9 → 63.6 ‰ (box), rtx-5090 58.5,
5080 54.9, 3090 38.0 → 35.0, H100 SXM 18.2, A100 12.5. A threshold on it would have to sit between 58.5 and 63.6 — a
margin the 4090's own power cap moved through. The parts' FFMA rate per SM is the same (128 lanes) on all but the A100,
so the clock is the only thing the balance adds to **DRAM bytes per µs per SM**, and that does not move with a power
cap: 4090 7 875 (wins), 5090 10 541 (async / kmajor-a lost, wide-fma mixed), 3090 / 5080 11 414, A100 14 398, H100 PCIe
17 885, H100 SXM 25 393 (lost or ship off). The mechanism the SASS shows: `wide-fma`'s 8 × 16 tile and the `cp.async`
panel trade occupancy for fewer loaded bytes per FFMA; that pays where each SM's DRAM share is the bottleneck,
whatever the clock. `rungs_read_properties_never_names` asserts the capped 4090 (1.955 GHz) takes the same three rungs
as the part at 2.52 GHz.

**Sustained clock as a measured property** (`calib.cu ffma`): after the burst peak, the FFMA kernel back to back for
4 s in 100 ms windows → `sustained_clock_khz` (median of the last second) and `settle_ms` (from which window every one
is within 2 %). It is the profile's `clock_khz` (documented on `Gpu::clock_khz`): rates and plan cost read it, no
rung rule reads it alone. RTX 5080 (360 W): burst 2.859 GHz, sustained 2.809 GHz, settle 0 ms (the profile keeps its
E1 2.45 GHz measured under the tile kernel; the gap is a measurement-method difference, open). Where the clock matters
in a decision now is the tune trial (§7), and there the measurement waits for it rather than reading it.

## 2. `pdl`: the chain, and the trigger at the exit

What the losers share (H100 traces): straight-line programs of one to four launches — every one lost or was null on
both parts (5080 mlp_512 / convmc / fir +2.0 / +1.2 / +0.7 µs; H100 fir / pj_m1 / mlp_512 +1.5 / +1.4 / +3.4 µs) — and,
on the H100 only, three short chains of warp-filled tile kernels (smlp, indep_512, rt_mlp2_1024: +17 to +68 µs, far
beyond any launch cost; the traced per-kernel times are unchanged, so it is the untraced overlap). What the winners
share: launches repeated by a loop (1024-trip decode chains, the model decodes): each dependent launch saves
`dep − pdl` (H100 988 ns, 5080 164 ns), the head is paid once.

`calib.cu head` (new): a tiny kernel then 1 / 2 / 4 kernels right after a synchronize, PDL vs plain, 400 reps: 5080
−180 / −29 / +10 ns — the launch call is not the head cost; the cost is the programs' own (open: which part of it).

**Three-arm run on the 5080** (10 rounds rotated, `iter ms` medians, no-pdl = 1): rules (entry trigger) / exit
trigger — decode 0.918 / **0.905**, mlp_512 1.043 / 1.040, convmc 1.015 / 1.017, fir 1.013 / 1.013, indep_512 1.017 /
0.953 (spread 17 %), smlp 1.004 / 1.006, rt_mlp2_1024 1.000 / 0.998, loopmm 0.996 / 0.998. The exit trigger keeps the
loops' gain; the entry trigger buys nothing measurable here and is what lets the dependent's blocks sit resident
beside the producer — the candidate for the H100's tile-chain losses (open: measured there with the entry trigger).

The emission: `pdl(ir, …)` = the part's rule (`pdl_pays`) and `launches_repeat(ir)`; a module without it launches
plainly and its kernels carry no `griddepcontrol` (the waits stripped: its PTX equals the `--gpu-off=pdl` arm's, checked
for the 8 programs below). `--gpu-on=pdl` forces any module; `--gpu-on=pdl-early` restores the entry trigger.

## 3. `row-split` priced at the site

H100 SXM trace (decode, 1023 trips): the split's 2 × 9.1 µs kernels + memsets replaced one merged 13.0 µs kernel holding
both unsplit passes — the pass alone is ~3 µs; the split adds a launch and a zero pass and loses the horizontal merge.
The rule now prices it: `saved = bytes / sm_bytes_per_ns − bytes / dram_bytes_per_ns` against `2 · dep_launch_ns`.
`sm_bytes_per_ns` is new (`DeviceCost`; `calib.cu bw` prints `one_sm`: the best `sms=1` L2 point): H100 SXM 147, A100 81,
4090 77, 3090 62, 5080 94 B/ns. The rows that won on the H100 PCIe (site-geometry G2, 1.41–1.56×) are the same four
programs — measured before the merged grids, with its launch unmeasured: that line stays (unpriced → property rule).
The split stays a per-site choice axis (`row-split` in `CHOICE_AXES`).

## 4–6. Open, with what the round separates

- **`min-blocks` (32 regs a resident thread).** H100 SXM ON wins 17 filled programs by 2–5 % (epi, loop, pj_m2048 ×4,
  mlp_2048, softmax, rowsum …) where the H100 PCIe's filled cells lost (the cap spilled 8–16 B, an older tile); OFF wins
  the m1 / m16 projections and mlp_512 (1.02–1.03), pj_m128_n6144 1.10; block decode 0.696 is its under-filled 112-block
  tiles. A100 ON wins the m16 projections (3–5 %). The parts share every per-SM fact: the separating fact is per site
  — the derived residency's register cap (`regs_per_sm / (blocks · threads)`) against the tile's own registers
  (`fma_held_regs`), and the 1107 `face-registers` evidence mismatches on the H100 are where to read it. Not changed
  without a measured site-level replay.
- **`warp-fill`.** At one warp a scheduler the losers are the m16 projections, attn, convmc and the model decodes; the
  winners loopmm, smlp, rt_mlp2, pj_m128_n6144 — on every part; FFMA per DRAM byte does not order them. Candidate:
  the fitted grid's wave quantization against the SMs (the m16 projections' 16–96 blocks). Recorded mixed.
- **`grid-share` / `panel-regs` on the 3090.** The round supports the shipped off: the rule wins 12 / loses 6
  (geomean 1.045 for ON), filled m16 projections by 1.22–2.20× (the cap drops `blocks_per_sm` 16 → 1). The losses are
  the 1024 × 256 × 256 site (decode 0.864, inside its 11 % null, while gen / causal at the same site win 1.29 / 1.30)
  and convmc (wps 2, 0.87–0.91). The L2-per-SM threshold separates no measured part wrongly; the convmc site is the
  open per-site fact.

## 7. The tune trial in steady state

The archived trials show the drift directly: on the A100 / 4090 boxes a large site's run times climb through the trial
(fitted slopes up to +35 × 10⁻³ a trial run; H100 / 3090 ≈ 0), so a candidate measured early wins on minima. The
replay (`tune_replay.py`) reproduces the shipped picks on all 308 sites, then re-decides with the pairs: 5 of the 16
steady-state-slower picks revert to candidate 0; 4 of 82 measured wins revert (3090 twin 0.955, pj_m16_n2048 0.981; H100
epi 0.976, loop 0.972); no new pick appears. A linear-drift fit over the whole trial was also replayed and rejected: it
moved 47 sites to other candidates nobody measured. The settle before the trial is not replayable (no rental):
5080 tune runs (6 programs × 2) settle in 5–10 runs a site (6–35 a program), the same wall time (~230 ms a process),
picks unchanged but for two marginal sites (rt_mlp2_1024 9dccf0 4 → 0 once, mlp_512 3e77a9 4 → 5 once).

## Properties updated (measured on the boxes, how)

| part | values now in `profile.rs` (all `calib.cu` on the box, 2026-11-21) | kept |
|---|---|---|
| h100 (SXM) | `cost`: DRAM 3186 / L2 3990 B/ns, one SM 147, gsync 910, dep launch 1722, pdl 734, coop 1839 ns; clock 1.807 GHz (FFMA under load, 700 W) | — |
| a100 | `cost`: DRAM 1488 / L2 2793, one SM 81, gsync 1113, dep 2050, coop 4714 | clock 1.41 GHz (measured 1.409) |
| rtx-3090 | DRAM 891 / L2 2447, one SM 62, gsync 514, dep 1472, coop 2866; clock 1.561 GHz (stock 350 W) | S62 place rates |
| rtx-4090 | one SM 77, gsync 502, dep 1440, coop 2660 | clock 2.52 GHz (the box's 1.955 is its 250 W cap: its calibration line carries it), DRAM 981 |
| rtx-5080 | one SM 94 (`calib/rtx-5080.calib` too) | clock 2.45 GHz (E1) |

Hosts' pageable ratios stay replay conditions (a property of the box, not the part). The H100 SXM joins `GPU_PARTS`.
Consequence: the 3090, 4090 and a100 priced `calls` (kernel-merging persistent chains) with their measured barrier
and launch costs alone — 32 of 208 candidate-0 hashes moved (4090 18, 3090 14: decode, attn, mlp_2048_tf32,
rq_mlp2_512, smlp, twin_512, px_mlp2_64_f64; the a100 formed causal / gen / prefill), unmeasured in time on those parts.
Reverted by the review fix (`../local-rtx-5080/review-fixes.md` §3): the rule also prices the bodies' measured change
inside one kernel (`coop_body_ns`), which none of them has yet — no chain forms there until their forced suite chains
measure it.

## RTX 5080 — this step's emission against HEAD (faa1f06f)

Rules that move here: `pdl` (straight-line modules plain; loops with the exit trigger) and the trial. PTX equality
(normalized names / labels): the 5 straight-line programs = the `--gpu-off=pdl` arm, the 3 loop programs = the
exit-trigger arm. ABBA, 8 rounds (16 processes an arm), paired new / old medians [IQR], old–old null:

| workload | HEAD (ms) | new (ms) | new / HEAD | null | what moved it |
|---|---:|---:|---:|---:|---|
| decode_1024_256 | 9.344 | 9.066 | 0.981 [0.964–0.986] | 3.0 % | exit trigger (three-arm: 0.905 vs 0.918 of no-pdl) |
| mlp_512 | 0.0460 | 0.0454 | 0.990 | 7.5 % | plain launches (three-arm: no-pdl 0.959 of rules) |
| fir_4194304_16 | 0.0486 | 0.0478 | 0.989 | 3.5 % | plain launches |
| convmc_8_64_56_3 | 0.0832 | 0.0827 | 0.989 | 2.4 % | plain launches |
| indep_512, smlp, rt_mlp2_1024 | — | — | 0.991 / 1.003 / 1.010 | 11 / 1.8 / 11 % | null |
| loopmm_1024_8 | 0.6490 | 0.6513 | 1.003 [1.001–1.006] | 0.2 % | exit trigger: 0.3 % (the one tile chain the entry trigger helped) |

Best competitor rows are unchanged from `local-rtx-5080/hardware-properties.md` (no tile plan moved on this part).

## Tests

| test | result |
|---|---|
| `rules::tests::rung_rules_replay` | 194 lines: 102 picked / 67 null / 9 mixed / 16 open, the open set asserted |
| `rules::tests::rungs_read_properties_never_names` | + the capped 4090 takes the 2.52 GHz part's operand-stream rungs |
| `tests/hardware_properties.rs` rung rules | 36 generated programs on drawn parts: DRAM-scarce sides 14 / 22, regs 15 / 21, L2 24 / 12, DRAM/SM 22 / 14; 36 emissions moved; 144 emulator runs (free, `MAPAL_EMU_SERIAL=1`, `=2`, `MAPAL_EMU_REVERSE=1`) equal the every-rung arm |
| `tests/hardware_properties.rs` pdl / row-split (new, `calls_prog_strategy`) | 24 programs on the H100: plain 18 / programmatic 6 (exactly the looped ones), no entry trigger, `pdl-early` adds 28 triggers and nothing else; retained-row splits kept 6 (one SM at 1 B/ns) / priced off 6; 24 emulator runs of priced vs split within the bound |
| `site_choice` | the trial's settle runs counted (5 on the fake clock); `c0-rules.tsv` regolded (32 / 208) |
| workspace | backend suite: every test passes after the updates (calls.rs / the calls unit test name the 5090 as the unmeasured part; knobs.json skips the SXM H100: no offline search) |

## Open

1. `min-blocks` per site (register cap vs the tile's registers); block decode's under-filled branch.
2. `warp-fill` at one warp a scheduler: wave quantization of the fitted grid.
3. The H100's tile-chain PDL losses (rt_mlp2_1024, blkdec; smlp / indep_512 now plain): re-measure with the exit
   trigger at the final round; `pdl-early`'s ON arm measures the entry trigger.
4. PDL's head cost on short programs: not the launch call (`calib.cu head`); its source is unmeasured.
5. 3090 / 4090 `calls` chains now priced from measured costs: timed at the final round.
6. The settle loop on a capped part (A100 / 4090 boxes): its cost and the picks it changes, at the final round.
7. The 5080's FFMA clock: E1 2.45 GHz vs `calib.cu ffma` 2.81 sustained.
