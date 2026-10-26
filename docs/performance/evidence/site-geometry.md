# site-geometry: end check and remaining gaps

Raw data: [`benches/results/evidence/site-geometry/sweep/`](../../../benches/results/evidence/site-geometry/sweep/)
(`report.md`, per arch `results-`, `evidence-`, `best-<arch>.tsv`; sweep `78ea7d4`, 4 archs; smoke under `../smoke*`).
Only programs whose emission differs from `recurrence-precision-latest` were re-run; every other row is **carried
(emission identical)** from [recurrence-precision.md](recurrence-precision.md). ON/OFF = the program's `iter ms`
(cells: `tile_bench` ms over its own cuBLAS leg), shipped build vs its arm, ABBA, same box; null = the shipped build
twice. The judge's table changes were checked by emission: 49 program × face rows move, and each equals the measured
arm it ships (`--gpu-off=` / `--gpu-on=` on the sweep build); 0 CPU rows move.

## End check (CLAUDE.md evidence pass, step 5)

Cells: intent rows ok/total on the shipped build and its arm (`evidence-<arch>.tsv`) · time **ON vs OFF** (ms) on the
program named.

| optimization | intended | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | decision (the difference that explains it) |
|---|---|---|---|---|---|---|
| `fill-wave` (C9, C10, C6c) | a grid below `fill_warps` cuts its free fold into the fewest partitions that fill the part, counted at the tile's own residency; split panels sized for the split grid | 53/53 (+139 n/a) · ON/OFF geomean probes 0.99 (9W 9L), cells 1.02 (3W 5L) | 44/44 (+157 n/a) · 1.09 (4W 15L), 0.93 (8W 2L) | 9/9 (+51 n/a) · 0.99 (5W 3L), 1.09 (1W 4L) | 10/10 (+26 n/a) · 1.01 (1W 3L), 0.98 (1W 0L) | **ON, open (G3).** Realized as intended: ON splits S = 2–16 with an atomic combine, panels bk' = 32 (4090), 8–16 (5090), 4 (3090) vs OFF S = 1, bk 64–512. Wins and losses on the same part are separated by facts the launch does not read (G3). n/a = ptxas / occupancy not captured on probes |
| `fit-first` (R1) | an FMA tile taller than its rows reshapes (`ty`/2, `tx`·2, threads kept) before the fill step halves its threads | 0/0 · convmc 0.0948 vs 0.1107; mm_16x18432x7168 1.053 vs 0.972 | 0/0 · convmc 0.0857 vs 0.0673 | 0/0 · pj_m16_n18432 0.6217 vs 0.5794 | 0/0 · convmc 0.1653 vs 0.1461 | **OFF rows on 5090 / H100 / 3090 (stopgap, G4); ON on 4090 (mixed, G4).** ON: 2 blocks per SM, bk 16–64, 95–108 registers; OFF: 5–8 per SM, bk 8–16, 41–77 registers. Grid 196 vs 784 blocks (convmc, 4090 and 5090), 144 vs 576 (16x18432, 4090); unchanged on 3090 and H100. **Mismatch:** the intent reads `plan.c` and `occ.sms`, which no probe records, so it never evaluated |
| `row-split` (D1) | a retained site's one-row pass runs on more than one CTA, its partials into a zeroed row or a reduced workspace | 0/0 decode family (2/2 on GEMV sites) · decode 36.08 vs 30.65 | same · 47.00 vs 34.95 | same · 34.75 vs 54.05 | same · 38.74 vs 32.85 | **OFF rows on 4090 / 5090 / 3090 (stopgap, G2); ON on H100.** 4090/5090/3090 take the atomic combine: a naive kernel with `atom.add`, 1–2 FMA in its hot loop vs 8 unrolled OFF, and 2 more zero launches per decode trip (`host.calls.zero` 8 vs 6). H100 (no `splitk-atomic`) takes the workspace split S = 128, no atomics: 1.41–1.56. **Mismatch:** the intent applies to `plan.kind == 'tile'`; the decode one-row pass is `naive`, so it never evaluated there |
| `tf32-radix` (R2) | a mixed-radix site whose products admit TF32 takes the `mma.sync` tile | 2/2 (+1 n/a) · convmc_tf32 0.0864 vs 0.0949 | 2/2 (+1 n/a) · 0.0700 vs 0.0852 (fitted FFMA tile 0.0680) | not timed | 2/2 (+1 n/a) · 0.1327 vs 0.1671 (fitted FFMA 0.1477) | **ON on 4090 (shipped) and 3090 (row lifted); OFF on 5090, H100, A100.** ON: 16 / 32 `mma.sync` + 27 / 54 `cp.async` replace the 512 / 1024-FFMA hot loop; values within the a-priori bound (K=576) on all three boxes. On the 5090 the FFMA tile with `fit-first` off (shipped there now) is 3 % faster than the `mma.sync` tile (null 1.0 %): GeForce TF32 ≈ FP32 peak. n/a = SASS HMMA not captured on probes |
| geometry table (ADR-0034 search) | a per-part knob row from the coordinate search, shipped where it beats the seed row beyond null | `resident` 1: 0.915, `warps` 8: 0.911 (null 0.046) | none | `split` 16: 0.713 (0.006) | `warps` 4: 0.855 (0.003) | **Not shipped (G1).** Each pick wins on its 3–7 timed programs with no loser, but a table knob is a part constant: it moves 467 / 418 (4090), 104 (H100), 362 (3090) program × face emissions, of which 3–7 were timed |

Other evidence findings: the legacy cuBLAS timing section of `box_run.sh` read the default-face module as `_exact`, so
every cell printed `MAPAL VALUES WRONG` (fixed, `18b1ffb`; the gated `_x` legs are bit-exact). H100's
`decode_1024_256.off-arch-rungs` arm failed its value gate by about 0.8 % on the default and `_c` faces (G6).

## GPU_RUNGS_OFF changes (`crates/backends/llvm/src/profile.rs`)

| part | rung | change | evidence (ms, ON vs OFF) |
|---|---|---|---|
| rtx-4090 | `row-split` | new row (stopgap G2) | decode 36.08 vs 30.65, prefill 22.69 vs 19.53, gen 21.95 vs 18.60, causal 43.97 vs 37.12 (null 0.4–2.8 %) |
| rtx-5090 | `row-split` | new row (stopgap G2) | decode 47.00 vs 34.95, prefill 24.95 vs 22.41; gen and causal inside an 11–16 % null |
| rtx-3090 | `row-split` | new row (stopgap G2) | decode 38.74 vs 32.85, prefill 27.71 vs 23.37, gen 25.93 vs 22.30, causal 46.92 vs 40.21 (causal null 25 %) |
| rtx-5090 | `fit-first` | new row (stopgap G4) | convmc 0.0857 vs 0.0673 (null 0.3 %), `_c` 0.1099 vs 0.0869 |
| h100-pcie | `fit-first` | new row (stopgap G4) | pj_m16_n18432_k7168 0.6217 vs 0.5794 (null 0.4 %), mm_16x18432x7168 cell 0.609 vs 0.571 |
| rtx-3090 | `fit-first` | new row (stopgap G4) | convmc 0.1653 vs 0.1461, `_c` 0.3296 vs 0.2579; mm_16x18432x7168 `_c` 0.893 vs 0.922 (ON faster on that face) |
| rtx-3090 | `tf32-radix` | row lifted (ON) | 0.1327 vs 0.1671 (null 1.1 %), values within bound |
| rtx-5090 | `tf32-radix` | row kept, new cite | values within bound; 0.0700 vs 0.0852, but the fitted FFMA tile runs 0.0680 |

Not changed: `fill-wave` (mixed on every part, G3); `fit-first` on the 4090 (mixed, G4); forced-ON `row-fit` wins on
H100 (1.50) and 3090 (1.43 cell, 1.02 probe) and `wave-gate` loses on the 3090 (0.89–0.90), but each moves 9–39 rows
of which 2 were timed, and `row-fit` on H100 shares its one site with the unshipped `split` 16 (G5).

Moved without a timing of their own (each is the OFF arm of a rung new this sprint): the exact faces of convmc,
convmc_tf32, pj_m16_n18432_k7168 and mm_16x18432x7168, `skinny_16x16384x512` on H100 / 3090, and the
non-probe `decode_1024_256` on 4090 / 5090 / 3090.

## Gap table (Mapal / best other; below 1 = Mapal faster)

Default face. ‡ = the judge's shipped emission is a measured arm of this sweep (its ms used). † = carried (emission
identical, earlier sweep). n/t = not timed on that part (emission identical, no earlier row). 3090 torch legs are
carried from recurrence-precision (its torch install failed a hash check).

| workload | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | recurrence-precision (4090 / 5090 / H100 / 3090) |
|---|---|---|---|---|---|
| convmc_8_64_56_3_tf32 | 1.91 | 1.44‡ | 2.62 | 1.53‡ | 2.07 / 1.36 / 2.38 / 1.69 |
| mm_16x1024x4096 (cell) | 1.06 | 1.70 | n/t | 2.62 | new |
| indep_512 | 2.18 | 2.18 | n/t | n/t | new |
| mlp_512 | 2.10 | 1.58 | n/t | n/t | new |
| attn_1024_64 | 2.09 | **0.82** | 2.00† | 1.77† | 1.97 / 1.39 / 2.00 / 1.77 |
| fanout_512 | 1.93 | 1.68 | n/t | n/t | new |
| twin_512 | 1.92 | 1.86 | n/t | n/t | new |
| mm_128x1024x1024 (cell) | 1.12 | 1.87 | n/t | n/t | new |
| pj_m16_n18432_k7168 | 1.08 | n/t | 1.86‡ | no leg | new |
| pj_m16_n1024_k4096 | 1.19 | 1.86 | n/t | no leg | new |
| diamond_512 | 1.86 | 1.82 | n/t | n/t | new |
| mm_128x2048x7168 (cell) | 1.08 | 1.82 | n/t | 1.40 | new |
| mlp2_512 | 1.79 | 1.52 | n/t | n/t | new |
| attn_1024_128 | 1.71 | 1.64 | n/t | n/t | new |
| pj_m16_n2048_k7168 | **0.69** | 1.30 | 1.58 | 1.56† | 0.86 / 1.21 / 1.35 / 1.56 |
| pj_m128_n2048_k7168 | 1.17 | 1.51 | n/t | no leg | new |
| decode_1024_256 | 1.01‡ | **0.82‡** | 1.51 | **1.00‡** | 1.03 / 1.52 / 2.48 / 1.12 |
| convmc_8_64_56_3 | 1.38 | **0.89‡** | 1.18 | 1.43‡ | 1.66 / 1.04 / 1.29 / 1.68 |
| pj_m1_n2048_k7168 | **0.42** | **0.77** | 1.40 | no leg | new |
| mm_16x18432x7168 (cell) | 1.09 | n/t | 1.39‡ | **0.98‡** | new |
| attn_2048_128 | 1.38 | 1.28 | n/t | n/t | new |
| pj_m128_n1024_k4096 | 1.31 | 1.27 | n/t | n/t | new |
| pj_m1_n1024_k4096 | 1.17 | 1.28 | 1.19 | no leg | new |
| pj_m1_n18432_k7168 | 1.01 | **0.93** | 1.21 | no leg | new |
| attn_2048_64 | 1.18 | 1.14 | n/t | n/t | new |
| prefill_1024_512_256 | **0.18‡** | **0.22‡** | **0.26** | **0.15‡** | 0.21 / 0.38 / 0.53 / 0.16 |
| causal_1024_256 | **0.18‡** | **0.26‡** | **0.25** | **0.14‡** | 0.21 / 0.40 / 0.54 / 0.16 |
| gen_1024_512_256 | **0.17‡** | **0.24‡** | **0.25** | **0.15‡** | 0.16 / 0.44 / 0.52 / 0.12 |

The other mm cells (13) and the rq_* / rmlp probes (no library leg) are in `sweep/report.md`.

## What moved

Across sprints the boxes differ, so only each row's ratio over its own box's best leg is compared.

| workload | before → after (4090 / 5090 / H100 / 3090) | the measured difference that moved it |
|---|---|---|
| decode_1024_256 | 1.03 / 1.52 / 2.48 / 1.12 → 1.01 / 0.82 / 1.51 / 1.00 | H100: `row-split` workspace split 1.56 and `fill-wave` 1.30 on this sweep's build; 4090 / 5090 / 3090: `row-split` OFF (atomic combine removed) 1.18 / 1.34 / 1.18 |
| causal, prefill, gen | 5090 0.38–0.44 → 0.22–0.26; H100 0.52–0.54 → 0.25–0.26 | H100: `row-split` 1.41–1.45 × `fill-wave` 1.09–1.11; 5090: `fill-wave` 1.12–1.17 on gen / causal |
| convmc_8_64_56_3 | 1.66 / 1.04 / 1.29 / 1.68 → 1.38 / 0.89 / 1.18 / 1.43 | 4090: `fit-first` ON 1.17 (784 → 196 blocks of 8 warps); 5090 / 3090: `fit-first` OFF 1.27 / 1.13 |
| convmc_8_64_56_3_tf32 | 2.07 / 1.36 / 2.38 / 1.69 → 1.91 / 1.44 / 2.62 / 1.53 | 4090 / 3090: `tf32-radix` 1.10 / 1.26 (`mma.sync`); 5090: the fitted FFMA tile (`fit-first` OFF); H100: Mapal 0.0944 → 0.1164 ms on another box, not attributed (no arm of this sweep moves it) |
| pj_m16_n2048_k7168 | 0.86 / 1.21 / 1.35 → 0.69 / 1.30 / 1.58 | 4090: `fill-wave` 1.25 (S = 4, k/S = 1792); H100 / 5090: `fill-wave` 0.89 / 0.99 |
| attn_1024_64 | 1.97 / 1.39 → 2.09 / 0.82 | 5090: competitor leg 0.0638 ms this box vs Mapal 0.0526; `fill-wave` 0.80 (loses) |

## Missing facts and open items (top first)

| id | what | evidence | the fact that would separate it / what closes it |
|---|---|---|---|
| G1 | The ADR-0034 search times its 3–7 GEMM programs, but a table knob moves every site of the part (4090 `resident` 1: 467 rows, `warps` 8: 418; H100 `split` 16: 104; 3090 `warps` 4: 362) | `best-<arch>.tsv`, emission reach of `--geom=` over the 189 bench programs × 3 faces | `search.py plan` takes its program list from the knob's emission reach (every row it moves), not a fixed list; then ship. The one-knob-at-a-time picks also overlap on the same sites (4090 `resident` and `warps`; H100 `split`, `row-fit`, `fit-first` at 16x18432x7168) |
| G2 | `row-split`: the atomic combine on a one-row pass loses 14–26 %; the workspace split wins 41–56 % | decode family, all four parts; `host.calls.zero` 8 vs 6, `atom.add` naive kernel | The combine cost of a one-row site: an atomic combine pays one zero pass per loop trip (`loop_plan`) and S-way contention on `cells` addresses (the row's extent); the workspace split pays neither. Choose the combine from these, then drop the three rows |
| G3 | `fill-wave` wins and loses on the same part | 4090 (every unsplit grid exactly one block per SM): losers have k/S ≤ 512 at bk' = 32 (≤ 16 panel trips per CTA: 512 chains 0.83–0.95, mm_128x1024x1024 0.80), winners k/S ≥ 1024 (pj_m128_n1024 1.26, pj_m16_n2048 1.25). 5090: the split grid holds 6–12 blocks per SM, which cuts bk' to 8; the same kernel pair flips between the L2-flushed cell and the warm probe (128x1024x4096 1.26 vs 0.80, 16x2048x7168 1.27 vs 0.99). Unexplained: rq_indep / rq_mlp 1.20 / 1.08 with indep's kernel (0.87). The 4090 `resident` 1 pick (fill target halved) points the same way | Two facts: panel trips per partition, (k/S)/bk' with bk' from the split grid's residency (`tile_plan` k + `TargetProfile` smem / registers); and where the site's operands are read from at launch (L2-resident: `footprint_plan` bytes vs `TargetProfile.l2_bytes` and whether the previous trip or producer touched them), the fact `asyncstage` also lacks |
| G4 | `fit-first` wins on 4090 convmc only | ON wins where the reshaped grid stays balanced over the SMs (4090 convmc 196 blocks on 128 SMs: the average SM carries 77 % of the busiest SM's blocks), loses at 56–58 % (5090 convmc 196 on 170, 4090 16x18432 144 on 128) and where the grid is unchanged and only residency drops (3090, H100: 2 vs 5–8 blocks per SM) | The reshaped grid's last-wave balance, blocks / (sms · ⌈blocks / sms⌉), against the shrunk grid's (`tile_plan` extents + `TargetProfile.sms`); fit first only when it does not drop it. Also make the intent evaluable (G7) |
| G5 | Forced-ON `row-fit` wins on H100 (1.50) and 3090 (1.43 cell); `wave-gate` ON loses on the 3090 (0.89–0.90) | `sweep/report.md` ON:row-fit, wave-gate rows | Time the 39 / 9 / 31 rows each moves (with G1's reach list) in one box run per part, row-fit jointly with `split` 16 on H100 |
| G6 | H100 `decode_1024_256.off-arch-rungs` (default and `_c`) failed the value gate by about 0.8 % | `h100-pcie/run-h100-pcie.attempt1-valuefail.out` | An emitter bug on the generic (all arch rungs off) path (ADR-0034 D4); reproduce on the interpreter with that arm's flags |
| G7 | Probes record no ptxas, occupancy or SASS facts, so `fit-first` (reads `plan.c`, `occ.sms`) and `row-split` (decode's pass is `naive`) never evaluate, and the fill-wave one-wave / spill checks are n/a | `evidence-<arch>.tsv` | Capture `ptxas -v` and `cuobjdump -sass` for probe kernels (the cells already have them in `prof-<arch>.tgz`); key `row-split` on the one-row site, not the tile kind |
| G8 | Noise and carried legs: 5090 nulls 11–16 % on the decode family; 3090 mm_16x18432x7168 cell spread 0.63–1.42 ms; 3090 torch legs carried; H100 square-cell section cut off; `pass2` files empty | `results-<arch>.txt` | Re-time on a quiet box |

## Top remaining gaps (each workload at its worst part)

| # | workload | worst | cause |
|---|---|---|---|
| 1 | convmc_8_64_56_3_tf32 | H100 2.62 | R2/T1: `tf32-radix` OFF on H100 (no box value gate), the FFMA radix tile vs cuDNN TF32 on tensor cores (H100 TF32 ≈ 7× FP32 peak) |
| 2 | mm_16x1024x4096 (cell) | 3090 2.62 | Open: no kernel capture for the 3090 cell; `wave-gate` OFF is 1.13 faster there (G5) |
| 3 | indep / mlp / twin / fanout / diamond _512 | 4090 1.86–2.18 | k = 512 split into k/S = 256 partitions loses 5–17 % (G3); the rest is unexplained (no nsys of these probes) |
| 4 | attn_1024_64 | 4090 2.09 | Carried C9 / N3 (cpu-gpu-boundary); `fill-wave` 1.04 |
| 5 | mm_128x1024x1024, mm_128x2048x7168, pj_m16_n1024_k4096 | 5090 1.82–1.87 | 5090 GEMM tile vs cuBLAS; the search's `warps` / `tn` picks win the geomean (0.84 / 0.94) but lose up to 1.80× on one program: a missing fact, not a table entry |
| 6 | pj_m16_n18432_k7168 | H100 1.86 | `split` 16 measures 0.67 here but is not shipped (G1) |
| 7 | decode_1024_256 | H100 1.51 | vs the KV-cache leg (22.99 ms); this sweep: `row-split` + `fill-wave` took it from 2.48 |
| 8 | convmc_8_64_56_3 | 3090 1.43 | R1 (2-warp blocks, 3090 spill), now with `fit-first` OFF |
