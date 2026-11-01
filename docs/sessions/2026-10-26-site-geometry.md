# 2026-10-26 — site-geometry sprint

Immutable. Base `recurrence-precision-latest` @ f6e3f88; main checkout ends on `site-geometry-latest` (9ea0a61 merges
the CLAUDE.md rule commit 765da70; 8b5bfce adds the result cache). Nothing pushed. Scope:
`.claude/sprints/site-geometry-scope.md`. Evidence: `docs/performance/evidence/site-geometry.md`,
`benches/results/evidence/site-geometry/`.

## Runs

| step | outcome |
| --- | --- |
| A (tooling) | `changed.py` (emission diff + result cache keyed by emission hash), changed-only bundles + `preflight.py` before renting, ADR-0034 search driver (`search.py`), `device_loops` over job slots (117 → 43 s), `tests/targets.py` |
| B (compiler) | geometry table `Gpu::geom` (9 knobs, `--geom=`), `fill-wave`, `fit-first`, `row-split`, `tf32-radix`, GEMM candidates as table entries |
| reviews → integrate | 20 findings fixed; 3 new bugs from the 4090 smoke fixed (epilogue group smem, bundle faces, evidence regex); 1295 tests pass |
| sweep | interrupted twice (macOS denied `/Volumes/LessComplex`; laptop checkpoint) and resumed from committed evidence; 61 changed programs; 8 rentals, 4 without a complete result |
| judge | applied the new rule (mixed result = missing fact): no table entries shipped, stopgap rows only, G1–G8 open |

Tokens: 1.52 M + 0.53 M ≈ 2.05 M subagent tokens.

## Decisions

| item | decision (evidence) |
| --- | --- |
| `tf32-radix` | ON 4090 (1.10) and 3090 (1.26, row removed); 5090 row kept (fitted FFMA tile 0.068 vs mma 0.070 ms); H100/A100 OFF (not measured) |
| `row-split` | OFF on 4090/5090/3090 — stopgap for **G2** (atomic combine on a one-row pass loses 14–26 %; the H100 workspace split wins 41–56 %) |
| `fit-first` | OFF on 5090/H100/3090 — stopgap for **G4** (wins only where the reshaped grid stays balanced over SMs) |
| `fill-wave` | no row; mixed on every part — **G3** (panel trips per partition, L2 residency at launch) |
| ADR-0034 search picks | none shipped — **G1**: a knob moves 104–467 rows per part, the search timed 3–7 programs |
| rule | CLAUDE.md: "a mixed result is a missing fact" (Sapir) |

## Results (Mapal / best; measured rows, default face)

| workload | 4090 | 5090 | H100 | 3090 | was |
| --- | --- | --- | --- | --- | --- |
| causal | 0.22 | 0.25 | 0.25 | 0.16 | 0.21 / 0.40 / 0.54 / 0.16 |
| prefill | 0.21 | 0.24 | 0.26 | 0.18 | 0.21 / 0.38 / 0.53 / 0.16 |
| gen | 0.20 | 0.24 | 0.25 | 0.17 | 0.16 / 0.44 / 0.52 / 0.12 |
| decode (sweep, rungs ON) | 1.19 | 1.10 | 1.51 | 1.18 | 1.03 / 1.52 / 2.48 / 1.12 (row-split now OFF on 4090/5090/3090) |
| convmc | 1.38 | 1.13 | 1.18 | 1.61 | 1.66 / 1.04 / 1.29 / 1.68 |
| convmc_tf32 (vs torch TF32) | 1.91 | 1.81 | 2.62 | 1.92 | 2.07 / 1.36 / 2.38 / 1.69 |

## Open — the missing facts (top first; details in the evidence doc)

| id | missing fact |
| --- | --- |
| G1 | the search must time every row a knob moves (emission reach), not a fixed program list |
| G2 | combine cost of a one-row site: atomic (zero pass per trip, contention on the row) vs workspace |
| G3 | panel trips per partition (k/S)/bk' from the split grid's residency; operands' L2 residency at launch |
| G4 | last-wave balance of the reshaped grid, blocks / (sms · ⌈blocks/sms⌉) |
| G5–G8 | forced row-fit / wave-gate rows untimed; H100 generic-path value-gate failure (emitter bug); probes lack ptxas/SASS; noise / carried legs |

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours |
| worktrees | sprint worktrees removed |
| local jobs | none |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3            # site-geometry-latest
cat docs/performance/evidence/site-geometry.md docs/next-session.md
python3 benches/gpu/changed.py diff /tmp/changed --base site-geometry-latest   # measure only what changed
```
