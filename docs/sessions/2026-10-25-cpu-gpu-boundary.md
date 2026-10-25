# 2026-10-25 — cpu-gpu-boundary sprint

Immutable. One workflow run (`wf_233c0550-71b`, 7 agents, 7.3 h, **1.87 M subagent tokens against ~300 k approved**).
Base `flow-latest` @ 5a49ab1. Main checkout ends on `cpu-gpu-boundary-latest` @ c167c3b (+ the doc fix in this
session's commit). Nothing pushed. Plan: `.claude/sprints/cpu-gpu-boundary-scope.md` § Plan.

## Branches

| branch | head | what |
| --- | --- | --- |
| `cpu-gpu-boundary-placement` | 78a3344 | builder A: `uniform`, `nest-cell`, `tree-gate`, retained-site fold veto dropped |
| `cpu-gpu-boundary-runtime` | 50c2a4f | builder B: `preload`, `entry-arena`, `async-h2d`, `stage`, `chain-h2d` |
| `cpu-gpu-boundary` | 66ebce8 | merge + 12 review fixes (`retained-fold` rung, `CellSite::work` fact, intents, property tests) + 4090 smoke |
| `cpu-gpu-boundary-sweep` | 7beca78 | 4-arch sweep evidence `benches/results/evidence/cpu-gpu-boundary/` |
| `cpu-gpu-boundary-latest` | c167c3b | rows: H100 `retain-resident` ON, `async-h2d` OFF ×4; `docs/performance/evidence/cpu-gpu-boundary.md`; README |

## Decisions (measured, ON/OFF same-box ABBA)

| rung | decision | evidence |
| --- | --- | --- |
| `uniform` (loop scalars as kernel args) | ON | prefill 65.5 vs 309.3 ms (4090); rt_mlp2 17 H2D → 1 |
| `nest-cell` (map folding over captures as device cell) | ON | convmc 0.98 vs 351.9 ms (4090); causal D2H 2148 → 9.2 MB (H100) |
| `retained-fold` | ON | prefill 0.25–0.37 ON/OFF |
| `tree-gate` | ON | rowsum_4096 4090 0.944 (closes flow's 1.056) |
| `preload` | ON | 7–19 programs faster per arch, 0 slower |
| `entry-arena` | ON | ew_4096 0.23–0.34 ON/OFF |
| `async-h2d`, `stage` | OFF row ×4 | no win: in-clock uploads are 1 KB/trip behind a host-read copy-out |
| `chain-h2d` | open | never emitted (streams ship OFF) |
| H100 `retain-resident` | row removed (ON) | causal 0.922, prefill 0.949, decode 0.940 (null 0.077); same sign as flow sweep |

Not delivered: 1c decode in-place row as device cell; 2 steady trip without host syncs (only upload waits removed);
1d mixed-radix fact (convmc runs as one thread per output, not implicit GEMM); CUDA graphs (none; if ever, built from
the static loop plan, never stream capture — only if nsys shows launch rate as the remaining gap).

## Results (Mapal / best of library legs and torch; flow sweep → this sweep)

41 of 96 workload × arch cells at or below best other (flow sweep: 26 of 92). Full table
`benches/results/evidence/cpu-gpu-boundary/table.md`.

| workload | 4090 | 5090 | H100 | 3090 |
| --- | --- | --- | --- | --- |
| prefill | 5.57 → 0.64 | 5.15 → 1.25 | 8.62 → 1.30 | 5.51 → 0.81 |
| causal | 2.87 → 0.62 | 2.74 → 1.30 | 5.62 → 1.41 | 2.89 → 0.75 |
| attn | 8.76 → 1.97 | 7.6 → 1.39 | 5.62 → 2.00 | 4.15 → 1.77 |
| ew_4096 | 2.59 → 0.46 | 5.33 → 0.45 | 4.75 → 1.01 | 2.27 → 0.61 |
| rt_mlp2 | 4.08 → 1.12 | 5.6 → 1.12 | 2.27 → 1.27 | 2.78 → 1.00 |
| decode | 1.96 → 1.66 | 1.34 → 1.60 | 3.58 → 2.95 | 1.89 → 1.69 |
| convmc | 5178 → 17.8 | 4041 → 12.4 | 2214 → 20.3 | 5023 → 27.6 |

H100 causal/prefill/decode were measured with retention OFF; it now ships ON (≈ −15 ms causal).

## Tests

Integrate: 1266 passed / 0 failed / 3 ignored; CPU-only emission 740/740 byte-identical vs flow-latest; drift 0 dead.
Smoke (4090): differential 602/602, traps 14/14, probe value gates 99/99. Sweep: value gates 25/25 on all 4 archs.
Judge changed rows and re-ran only the 9 broken tests; full workspace re-run on `cpu-gpu-boundary-latest`
@ c167c3b: 1266 passed / 0 failed / 3 ignored.

## Open

| id | what |
| --- | --- |
| N1 | decode @ H100 46 ms gap: row t written on host → 1 H2D + 1 D2H per trip (1c) |
| N2 | causal/prefill: `mx`/`den` whole scalar folds stay on host → 3 D2H + 1 H2D per trip; nest cells run 1–4 CTAs with a serial fold per thread |
| N3 | convmc 12–28× behind torch: mixed-radix fact → implicit GEMM (1d) |
| N4 | H100 64-accumulator GEMM tile (mlp2, diamond, rt_mlp2, fanout, indep) — deferred C6a |
| N5 | untraced: 3090 epi 0.79 → 0.88 (and small 3090 rises on mlp/mlp2/twin/diamond/fanout); smoke ew_4096 async 1.108 (O10); O2 0.678 → 0.922 shrink explained only by nest-cell removing D2H |
| N6 | `entry-arena` gives every slot a whole-function interval (two loops never share bytes); iota-there rule limited to nest cells but not to `nested` slots |
| N7 | 110 smoke `evidence.py` mismatches (face-fma/face-separate/arena on `.fs`) not compared against flow-latest; sweep says same 21 intent rows as flow |

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours (4 smoke + 4 sweep boxes destroyed and confirmed); `stocklab-*` untouched |
| worktrees | `.claude/worktrees/wf_233c0550-71b-{1,2,5,6}` hold the four sprint branches (clean, prunable) |
| local jobs | none |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3          # cpu-gpu-boundary-latest
cat docs/performance/evidence/cpu-gpu-boundary.md docs/next-session.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
