# Continue after `site-geometry`

Main checkout: branch **`site-geometry-latest`**. Nothing is pushed. Rules every agent follows:
[`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-10-26-site-geometry.md). Latest evidence: [`site-geometry.md`](performance/evidence/site-geometry.md).

## Where Mapal stands (recurrence-precision sweep, best-of library legs and torch)

End check and gaps: [`recurrence-precision.md`](performance/evidence/recurrence-precision.md) · raw:
[`benches/results/evidence/recurrence-precision/`](../benches/results/evidence/recurrence-precision/) (only changed
programs re-run; other rows carried, emission identical). 54 of 120 workload × arch cells at or below the best other.

| | result (Mapal / best other) |
| --- | --- |
| ahead | causal 0.16–0.54, prefill 0.16–0.53, natural one-loop gen 0.12–0.52 (= hand-written prefill), conv / FIR / ew, rt_mlp2 on 4090/3090 |
| about even | rowsum, softmax, 2048 MLPs on 4090/5090, convmc on 5090 (1.04) |
| behind | decode 1.03–2.48 (H100 worst), convmc 1.29–1.68, H100 FP32 G## Candidate next sprint (not yet approved) — the missing facts (site-geometry G1–G4)

The site-geometry rungs split by program on every GPU; per CLAUDE.md ("a mixed result is a missing fact") the next step
is the facts that separate winners from losers, each read by its rule, then drop the stopgap rows:

1. **G1 search reach**: `search.py plan` times every row a knob moves; then ship the picks (4090 `resident` 1 0.915,
   `warps` 8 0.911; H100 `split` 16 0.713; 3090 `warps` 4 0.855).
2. **G2 combine cost** for one-row sites (atomic vs workspace) → `row-split` right per part (H100 wins 41–56 %).
3. **G3** panel trips per partition + operands' L2 residency at launch → `fill-wave`.
4. **G4** last-wave balance of the reshaped grid → `fit-first`.
5. G6 emitter bug on H100's generic path (value gate −0.8 %); G7 ptxas/SASS for probes.
Still behind after that: convmc_tf32 H100 2.62, decode H100 1.51, attn 4090 2.09, 5090 GEMM cells 1.8.

runtime spin rule vs external load (R8, measure first).

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
cat CLAUDE.md docs/performance/evidence/recurrence-precision.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
