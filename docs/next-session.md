# Continue after `site-geometry`

Main checkout: branch **`site-geometry-latest`**. Nothing is pushed. Rules every agent follows:
[`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-10-26-site-geometry.md).

## Where Mapal stands (best-of library legs and torch)

End check, gaps and missing facts: [`site-geometry.md`](performance/evidence/site-geometry.md) (previous:
[`recurrence-precision.md`](performance/evidence/recurrence-precision.md)). Only changed programs are re-run
(`benches/gpu/changed.py`, result cache `benches/results/cache/`); unchanged rows are carried.

| | result (Mapal / best other) |
| --- | --- |
| ahead | causal 0.16–0.25, prefill 0.18–0.26, natural one-loop gen 0.17–0.25 on all four GPUs; conv / FIR / ew chains |
| about even | rowsum, softmax, 2048 MLPs on 4090 |
| behind | convmc_tf32 1.81–2.62, attn 4090 2.09, 512 chains 4090 1.86–2.18, 5090 GEMM cells 1.8, decode 1.1–1.5, convmc 1.13–1.61 |

## Candidate next sprint (not yet approved) — the missing facts (site-geometry G1–G4)

The site-geometry rungs split by program on every GPU. Per CLAUDE.md ("a mixed result is a missing fact") the next
step is the facts that separate winners from losers, each read by its rule; then the stopgap rows go:

1. **G1 search reach**: `search.py plan` times every row a knob moves; then ship the picks (4090 `resident` 1 0.915,
   `warps` 8 0.911; H100 `split` 16 0.713; 3090 `warps` 4 0.855).
2. **G2 combine cost** of a one-row site (atomic: a zero pass per trip + contention on the row; workspace: neither) →
   `row-split` per case (H100 workspace wins 41–56 %, atomic loses 14–26 % elsewhere).
3. **G3** panel trips per partition (k/S)/bk' from the split grid's residency, and operands' L2 residency at launch →
   `fill-wave`.
4. **G4** last-wave balance of the reshaped grid, blocks / (sms · ⌈blocks/sms⌉) → `fit-first`.
5. G6 emitter bug on H100's generic path (value gate −0.8 % on decode `off-arch-rungs`); G7 ptxas/SASS for probes;
   runtime spin rule vs external load (measure first).

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
git status --short && git log --oneline -3
cat CLAUDE.md docs/performance/evidence/site-geometry.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
