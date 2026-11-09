# 2026-11-10 — tile-efficiency

Branch `tile-efficiency` from plan-cost `6698976`. Not pushed. End check:
[`tile-efficiency.md`](../performance/evidence/tile-efficiency.md).

## What was done

| step | commit | result |
| --- | --- | --- |
| T4 plan-cost LSU term | `b7a33cf` | `PlanShape.lsu`, `DeviceCost.lsu_ipc_pm`; PTX count equals the model's |
| T2 `kmajor-a` | `5de58a4` | A panel untransposed, 16-B copies; emulator `cp.async` queue |
| T1 `full-tile` | `54c7fb7` | full register tile as a split candidate under plan-cost |
| T3 `async-depth` | `2bc99a4`, `fb4efee` | bk ≥ 16 on streamed panels unless staging is deeper; i32 stage counter |
| sweep | `881cb0b` | 4090 / 5090 / 3090 / H100 (H100 partial: ssh dropped in probes); value gates 0 fail |
| judge | this commit | `kmajor-a` rows: 4090 Filled, 5090 / 3090 / H100 Any; plan-cost rows kept |

## Verdicts

| rung | 4090 | 5090 | 3090 | H100 |
| --- | --- | --- | --- | --- |
| `kmajor-a` | keep streamed, gate Filled | gate | gate | gate |
| `full-tile` | keep (dormant: plan-cost row) | keep (dormant) | dormant (no rates) | dormant |
| `async-depth` | keep (no effect) | keep | — | — |
| `async` rows | open (cells only) | open (ON wins 5 / 0 cells) | — | — |
| `plan-cost` row | kept (L: pj_m16, decode, M=1) | kept (10 program L) | — | — |

## Tests

`cargo test --workspace --release --no-fail-fast` with the rows: 102 binaries ok, 2 lib tests failed
(`gpu_rungs_are_the_measured_table`: cites need raw `.txt` + ms; `skinny_projections_take_the_ordered_cooperative_tile`:
the 5090 row-fit pin now forces `kmajor-a`, shipped pinned at staged bk64×1). Fixed; `mapal-backend-llvm --lib` 92 / 92.
`cargo fmt --all --check` ok.

## Live state at close

- Rentals: 0 `tile-eff` instances on default and itur.
- Raw: release `evidence-raw-2026-10-29` (`raw-raw.tar.gz`); tables in `benches/results/evidence/tile-efficiency/sweep/`.
- Judge scratch: `.claude/sprints/tile-efficiency/scratch/judge/` (`sass2.py` counts any arm's SASS from the bundle).

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal && git switch tile-efficiency && git log --oneline -3
python3 .claude/sprints/tile-efficiency/scratch/judge/sass2.py rtx-5090:pt_mlp2_1024:ship,off-kmajor-a
```
