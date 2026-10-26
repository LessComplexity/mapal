# 2026-10-25 → 26 — recurrence-precision sprint (+ harness overload fix)

Immutable. Base `cpu-gpu-boundary-latest`; main checkout ends on `recurrence-precision-latest` @ 835b972 (+ this log).
Nothing pushed. Scope: `.claude/sprints/recurrence-precision-scope.md`. Evidence: `docs/performance/evidence/recurrence-precision.md`.

## What happened (runs)

| run | outcome |
| --- | --- |
| `wf_a892a0c9-7ba` | builders died with the Claude process (nvim terminal hung) — WIP saved as `wip:` commits |
| `wf_cfbfca41-ff6`, `wf_2e9a9b78-395` | died again; first blamed on the volume, real cause found below |
| fix `579b0c2` | **the differential harness oversubscribed the laptop**: sweeps started 14 workers each running binaries with 14-thread pools (≈196 runnable threads on 14 cores), emulated runs had no timeout → machine hot and stuck. Now one child-process limit per test binary ((cores−2)/2 or `MAPAL_TEST_JOBS`), 2-lane pools + `MAPAL_SPIN=0` for sweep children, 120 s emulator timeout. Differential 41/41, 694 s, 1.9 cores avg, peak 595 % |
| `3fb5b7b` | `mapal-interp` at opt-level 3 in dev/test builds (checks kept): differential 777 → 683 s |
| `wf_2977f68d-b31` | builders + reviews + integrate (4090 smoke); integrate cut once by a network outage and resumed; sweep **narrowed by Sapir** to programs whose emission changed (138 of 1528 rows, 25 of 192 programs; rest carried); judge |

Tokens: `wf_2977f68d-b31` 1.41 M + 0.47 M = 1.88 M, plus the three interrupted runs (uncounted) — over the ~1.5 M approved.
Rentals: smoke 2 boxes; sweep **28 rentals, 15 without results** (two bundle mistakes: arms without an oracle, cutlass
dir missing from the torch-only bundle) — all destroyed and confirmed.

## Decisions (ON/OFF same-box ABBA, OFF ÷ ON; null ≤ 2.6 % except convmc H100 9.7 %)

| rung | 4090 / 5090 / H100 / 3090 | decision |
| --- | --- | --- |
| `trip-batch` (gen) | 1.92 / 1.99 / 1.99 / 1.97 | ON (4618 launches vs 9216) |
| `range-cell` (decode … gen) | 1.14–2.94 | ON (per-trip upload 1024 B → 0) |
| `scalar-fold` (causal/prefill/gen) | 1.34–1.44 | ON default face; contract face open (0.96–1.00 on 3 archs, no null) |
| `cell-lanes` | 2.04–3.17 | ON |
| `fold-lanes` (rowsum) | 0.98–0.99 on 4090, no arm elsewhere | **OFF row rtx-4090** (column-major input already coalesced) |
| `radix-tile` (convmc) | 8.9 / 11.5 / 15.6 / 15.5 | ON (3090 kernel spills 72 B) |
| `tf32-tile` (`: tf32 \| f32` mlp) | 1.25 / 1.25 / 3.38 / 1.18 | ON (HMMA; HGMMA on H100) |

ADR-0041 implemented with the surface `: compute` on steps, blocks and functions (`fn f(a) -> T: tf32 | f32 { }`).

## Results (Mapal / best of library legs and torch)

| workload | 4090 | 5090 | H100 | 3090 | was (cpu-gpu-boundary) |
| --- | --- | --- | --- | --- | --- |
| causal | 0.21 | 0.40 | 0.54 | 0.16 | 0.62 / 1.30 / 1.41 / 0.75 |
| prefill | 0.21 | 0.38 | 0.53 | 0.16 | 0.64 / 1.25 / 1.30 / 0.81 |
| gen (new, natural one loop) | 0.16 | 0.44 | 0.52 | 0.12 | — ; gen ÷ prefill 0.95–0.99 |
| decode | 1.03 | 1.52 | 2.48 | 1.12 | 1.66 / 1.60 / 2.95 / 1.69 |
| convmc (vs torch FP32) | 1.66 | 1.04 | 1.29 | 1.68 | 17.8 / 12.4 / 20.3 / 27.6 |
| rt_mlp2_2048 | 0.75 | 1.00 | 1.17 | 0.88 | 1.12 / 1.12 / 1.27 / 1.00 |

54 of 120 cells at or below the best other; on the 96 cells shared with the last sweep 48 (was 41).

## Tests

Integrate 1289 / 0 / 3; CPU-only emission identical (174/174 emit sweep, 561/561 benches); drift 0 dead. Smoke value
gates 101/103 → all pass after the M-oracle gate fix (offline). Sweep value gates 18/18 timed programs × 4 archs.
Judge full workspace 1289 / 0 / 3.

## Open

| id | what |
| --- | --- |
| R1 | decode @ H100 31 ms gap: per-trip K/V row kernel = 1 block × 256 threads (12.9 µs × 2046) vs library GEMV 64–256 blocks |
| R2 | H100 FP32 tile too small (rt_mlp2, mlp2, diamond, indep); rt_mlp2_1024 second wave (128 blocks on 114 SMs) |
| R3 | `: tf32` does not reach the radix tile (convmc_tf32 = FP32 FMAs); `_tf32` GEMMs trail torch TF32 (H100 1.75–1.87: two waves) |
| R4 | convmc: 2-warp blocks, ~22 % FP32 peak; 3090 spill |
| R5 | 1c checkpoint seeding not done; `pt_gen` launches 82 vs prefill 78 in the emulator (twin check) |
| R6 | `scalar-fold` contract face; `fold-lanes` should read the input layout (fact) instead of a per-arch row |
| R7 | test speed: `differential_device_loops` serial (≈ 600–750 s); macOS first exec 0.5 s per new binary; measure-only-what-changed as the default for builders (affected tests) and every box run |
| R8 | runtime spin rule ignores external load (latent; needs its own measurement) |

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours (9 `stocklab-*`, untouched) |
| worktrees | sprint worktrees removed; older agent worktrees remain (prunable) |
| local jobs | none |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3            # recurrence-precision-latest
cat docs/performance/evidence/recurrence-precision.md docs/next-session.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
