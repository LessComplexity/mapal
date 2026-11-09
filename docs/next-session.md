# Continue after `plan-cost`

Main checkout: branch **`plan-cost`** (local, not pushed; `site-geometry-latest` has the open PR to `main`). Rules
every agent follows: [`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-11-08-plan-cost.md)
(previous: [site-geometry](sessions/2026-10-26-site-geometry.md)).

## Where Mapal stands (best-of library legs and torch)

End check, gaps and missing facts: [`plan-cost.md`](performance/evidence/plan-cost.md) (previous:
[`site-geometry.md`](performance/evidence/site-geometry.md)). Only changed programs are re-run
(`benches/gpu/changed.py`, result cache `benches/results/cache/`); unchanged rows are carried.

| | result (Mapal / best other) |
| --- | --- |
| shipped | the rules (plan-cost and 5090 `linear` gated OFF by stopgap rows); the README table holds |
| plan-cost ON, measured | `_512` chains 4090 0.97–1.24 (BASE 1.17–1.37), 5090 1.15–1.77 (BASE 1.33–1.86) |
| plan-cost ON, why gated | M=1 GEMV 2–3.6×, decode 1.94× / 1.32×, 5090 2048 sizes 1.07–1.16 vs BASE |

## Candidate next sprint (not yet approved) — the missing facts that gate plan-cost

The launch side of the cost model is right (`_512` chains: zero passes → 0, ops 3→2, idle gaps 6.6 → 0.6 µs); the
kernel side picks grids that leave SMs idle. Add, each as one fact with its `testgen` property test, then drop the
stopgap rows and re-time only the rows whose emission changes:

1. **In-flight bytes per SM** (`TargetProfile`: DRAM / L2 latency; per warp `stages · (bm+bn) · bk · e` in flight):
   feed and FFMA rate capped by `Σ_SM inflight / latency`. Closes M=1 GEMV, decode, `pj_m16` (4090), 5090
   `mm_128x1024x{1024,2048}`.
2. **Per-SM load of a wave** (site-geometry G4): the busiest SM's `⌈B_w/sms⌉` blocks. Closes 5090 2048/4096 (256 tiles
   on 170 SMs; stream-K's balance lost).
3. **k-depth** (G3): panel trips per partition `(k/S)/bk` × per-trip latency at the site's stages.
4. Then hold-out ≥ 90 % per part → delete the 4090 / 5090 plan rows (design step 9: 7 / 14).
5. Still open from the plan-cost design: grid union 6(b) (one grid vs 2 launches vs streams), tile flags (c) and the
   zero-free split with epilogue (step 8), the prologue, CUDA graph / PDL for loops (`pdl` needs sm ≥ 90; OFF on
   5090 by row).
6. Unmeasured: `px_diamond_128` 5090 rules emission ≠ BASE; 20 changed programs with no library leg (`NO_LEG`);
   `skinny_8x4096x256_f64` hot-loop intent MISMATCH (not a plan-cost rung). Site-geometry G1, G2, G6, G7 stay open.

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours; only `stocklab-*` (another project — never touch) |
| worktrees | older agent worktrees under `.claude/worktrees/` (finished; prunable) |
| scope files | `.claude/sprints/` (durable; `/private/tmp` is wiped on reboot) |
| local load | test harness bounded (`MAPAL_TEST_JOBS`, default (cores−2)/2); run Claude outside nvim's terminal |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git switch plan-cost && git status --short && git log --oneline -3
cat CLAUDE.md docs/performance/evidence/plan-cost.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
