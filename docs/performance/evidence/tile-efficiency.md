# tile-efficiency — end check

Sweep: commit `881cb0b`, raw `evidence-raw-2026-10-29/raw-raw.tar.gz`, tables
[`benches/results/evidence/tile-efficiency/sweep/`](../../../benches/results/evidence/tile-efficiency/sweep/).
Design and build: `.claude/sprints/tile-efficiency/{design,build}.md`. Base = the plan-cost emitter `6698976`.
Ratios are time, shipped (ON) / OFF unless named; W / L are beyond each part's null (cells 4.4–12.8 %, programs
2.6–7.5 %; `null/shipped` geomean 0.96–1.01). Value gates passed before timing on every part (differential 376 / 285 /
256 / 296, 0 fail; 3090 without the 35 `px_*_tf` objects, whose own reference arm fails the TF32 bound).
H100: the box dropped during the probes; its rows come from the box stdout and SASS is from local ptxas.

## Intended vs realized, per rung per arch

| rung | intent | 4090 | 5090 | 3090 | H100 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `kmajor-a` (T2) | 16-B A copies only; STS.128 only; LDS unchanged | async panels: 48 LDGSTS.32 → 0 (24 LDGSTS.128), FFMA/LSU 4.2 → 6.6, cells 0.56–0.72. Staged wide tile: regs 220 → 255, bk 16 → 8 (FFMA 2080 → 1056), mm_4096 1.22, 2048³ 1.25 vs base; 128x18432 spill 8 → 40 B, 1.14 | staged: STS 15 → 6, regs 77 → 83, LDS / LDG / FFMA / BAR equal; programs 1.03 (4 / 28), 128x2048x7168 1.21 | staged: STS 27 → 9; mm_4096 regs 220 → 255, bk halved, 1.36; programs 1.14 (2 / 41); skinny wins 0.68–0.91 (LDS.32 128 → LDS.128 16) | staged: regs 128 → 167 (mlp2_1024), 111 → 151 (mm_4096); programs 1.16 (3 / 50) | **keep** on streamed panels (4090 Under / Partial); **gate** register-staged: rows 4090 Filled, 5090 / 3090 / H100 Any (stopgap, missing fact 1) |
| `full-tile` (T1) | acc ≥ 64, ≥ 128 t, FFMA/LDS ≥ 16 on split picks | PC+FT / PC 1.01: same tile on 3 of 4 cells | PC+FT / PC 0.97; 128x2048x7168 8x8 256 t, FFMA/LSU 14.2 vs 6.4, 33.4 vs 28.2 TFLOP/s | no rates: dormant | no rates: dormant | **keep** (acts only under plan-cost, which stays gated); 205 intent mismatches are the rules' tile the model keeps |
| `async-depth` (T3) | bk ≥ 16 / tm·tn·bk ≥ 1024; 1 BAR + i32 counter | 1 BAR a trip, no `urem`; OFF arm did not separate (17 moved programs inside null) | no streamed rows moved | — | — | **keep** (no effect measured); 92 intent mismatches = shallow streamed rows where staging is not deeper (by rule) |
| `async` re-judge | — | Under: cells 0.85–0.92 (3 / 0); rmlp programs 1.22–1.33 — their async panels keep 4-B copies (640 LDGSTS.32, FFMA/LSU 3.3 vs 4.0 staged; k-major does not apply, tx < 8). Filled ON: 2048³ 0.84 (bk 8 → 16, BAR 2 → 1) | ON: cells 0.80 (5 / 0): 128x2048x7168 0.50, 512³ 0.73 (LDG+STS → 12 LDGSTS.128, BAR 2 → 1, bk 8 → 16); 2048³ 1.13 (null) with 4 B spill | — | — | **open**: rows unchanged; ON measured on cells only, no program arm (breadth) |
| `plan-cost`+`linear` (T4) | `lsu=` printed; pick FFMA/LSU ≥ 13 | ON / shipped cells 0.96 (3 / 1), programs 0.93 (8 / 2): pj_m16_n1024_k4096 1.37, decode 1.14, mm_1x1024x4096 1.10 | cells 0.92 (3 / 1), programs 0.93 (11 / 10): M=1 / M=16 projections 1.07–1.52 | no rates | no rates | **rows kept** on 4090 and 5090: every loss is an M ≤ 16 DRAM-streaming site (missing fact 2) |

## Gap: Mapal / best library (cuBLAS), time, base → shipped `881cb0b`

| shape | 4090 | 5090 | 3090 | H100 |
| --- | --- | --- | --- | --- |
| 512³ | 1.10 → 0.86 | 1.00 → 1.00 | 2.43 → 2.43 | 1.23 → 1.27 |
| 1024³ | 1.28 → 1.06 | 1.29 → 1.43 | 1.56 → 3.13 | 1.03 → 1.19 |
| 2048³ | 0.98 → 1.22 | 1.11 → 1.14 | 1.16 → 1.52 | 1.37 → 1.62 |
| 128x1024x1024 | 1.11 → 0.64 | 1.87 → 1.87 | 2.07 → 2.07 | 1.58 → 1.52 |
| 128x2048x7168 | 1.07 → 0.76 | 1.80 → 2.18 | 1.49 → 1.49 | 2.20 → 2.46 |
| 16x1024x4096 | 1.09 → 1.09 | 1.59 → 1.59 | 2.51 → 1.70 | 1.39 → 1.11 |
| 16x18432x7168 | 1.10 → 1.09 | 1.25 → 1.25 | 0.99 → 1.02 | 1.38 → 1.43 |

The `kmajor-a` rows (judge, after the sweep) return every register-staged kernel to the measured `off-kmajor-a`
arm: 5090 / 3090 / H100 ship that arm whole; the 4090 keeps k-major on its streamed panels (the 128-row and 512³ /
1024³ wins) and drops it on the filled wide tile. Not re-timed: the arm is already measured
(`rungs-detail-*.tsv`), except 4090 2048³, whose OFF arm was measured only as base (56.5 TFLOP/s).

Chains, Mapal / best library: 4090 512 1.03–1.17 (base 1.19–1.34), 1024 1.08–1.37 (base 1.25–1.40); torch-compile
512 0.86–0.99, 1024 1.08–1.19. 5090 1024 indep 117 → 127 µs (1.55). 3090 1024 1.5–1.7, 512 2.2–2.9. H100 1024
1.55–1.89 (no torch leg). Single-shot at ~50 µs.

## What moved and why

- **4090 128-row and 512³ / 1024³ cells won** (0.56–0.83): the A panel now streams as 16-B `cp.async` (48 LDGSTS.32 →
  0), and the unpadded panel doubles bk (32 → 64 at 1024³), so FFMA/LSU rises 6.7 → 9.4 at one BAR a trip. Mapal now
  beats cuBLAS at 128x1024x1024 (14.6 vs 9.3 TFLOP/s) and 128x2048x7168 (32.5 vs 24.8).
- **Register-staged kernels lost on every part**: k-major removes 2/3 of the STS but adds the fragment registers
  (`(lanes−1)·tm` held). Where that crosses a budget it shows in ptxas: the 8x16 wide tile hits 255 registers and the
  prefetch budget halves bk (2 BAR per 1024 FFMA instead of per 2048), the 4090 128x18432 kernel spills 40 B, and the
  H100 kernels cross 128 registers (occupancy). On 5090 / 3090 at +1…+6 registers the counts favour ON and time does
  not: the staged k-major store is unpadded (`pad` skipped), so bank conflicts are the candidate. ncu is blocked.
- **Full tile** shows once, 5090 128x2048x7168 (33.4 vs 18.7 TFLOP/s shipped); elsewhere the cost model keeps the
  rules' tile, so it stays behind the plan-cost rows.

## Missing facts, ranked

1. **Register-staged k-major cost** (gates `kmajor-a` on 4 parts): the panel's held fragment registers against the
   register budget and prefetch bk, and the bank geometry of an unpadded staged k-major store. Winners and losers split
   on copy mode in every row measured; the skinny 3090 wins (A fragment LDS.32 → LDS.128) show the rule must read
   both. Needs ncu bank-conflict counters (blocked on vast).
2. **Loaded DRAM latency in plan-cost `c_ns`** (gates plan-cost on 4090 / 5090): all re-time losses are M ≤ 16 sites.
3. **k-major where tx < 8** (build open 5): the 4090 rmlp async panels still copy A 4 B at a time (FFMA/LSU 3.3).
4. **`async` rows**: ON wins every 5090 cell and the 4090 2048³ cell; a program-suite arm decides the rows.
5. 3090 / H100 have no plan-cost rates, so T1 / T4 are dormant there.
