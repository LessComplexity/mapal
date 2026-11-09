# Continue after `tile-efficiency`

Main checkout: branch **`tile-efficiency`** (local, not pushed). Rules every agent follows: [`CLAUDE.md`](../CLAUDE.md)
(`AGENTS.md` → symlink). History: [session log](sessions/2026-11-10-tile-efficiency.md) (previous:
[plan-cost](sessions/2026-11-08-plan-cost.md)).

## Where Mapal stands

End check, gaps and missing facts: [`tile-efficiency.md`](performance/evidence/tile-efficiency.md) (previous:
[`plan-cost.md`](performance/evidence/plan-cost.md)). Only changed programs are re-run (`benches/gpu/changed.py`).

| | result (Mapal / cuBLAS, time) |
| --- | --- |
| shipped | the rules + `kmajor-a` on streamed (`cp.async`) panels; stopgap rows gate it on register-staged panels (4090 Filled, 5090 / 3090 / H100 Any) and keep plan-cost OFF on 4090 / 5090 |
| 4090 wins | 128x1024x1024 1.11 → 0.64, 128x2048x7168 1.07 → 0.76, 512³ 1.10 → 0.86, 1024³ 1.28 → 1.06 |
| still behind | 5090 128-row 1.87–2.18, 3090 512³ / 128-row 2.1–2.4, H100 128x2048x7168 2.2 |

## Candidate next sprint (not yet approved) — the missing facts behind the rows

1. **Register-staged k-major cost**: held fragment registers vs register budget / prefetch bk, and bank geometry of the
   unpadded staged k-major store (needs ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}`; ncu is
   blocked on vast — rent a box that permits it). Then the rule reads the fact and the 4 `kmajor-a` rows go.
2. **Loaded DRAM latency in plan-cost `c_ns`** (all plan-cost re-time losses are M ≤ 16): then drop the 4090 / 5090
   plan-cost rows (plan-cost.md items 1–3 still open: in-flight bytes per SM, per-SM wave load, k-depth).
3. **k-major where tx < 8** (4090 rmlp async panels copy A 4 B at a time, FFMA/LSU 3.3).
4. **`async` rows re-judge** with a program-suite arm (`--gpu-on=async`): ON won every 5090 cell (0.50–0.81) and 4090
   2048³ (0.84) on cells alone.
5. Not run: 4090 `off-kmajor-a` at 2048³, `async-depth` post-review ON/OFF, H100 timing section / torch / SASS from
   the box, 3090 TF32 `px_*_tf` (reference arm fails the 1e-4 bound: atomic order).

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours (`tile-eff` destroyed on default and itur); `stocklab-*` belongs to another project — never touch |
| raw output | release `evidence-raw-2026-10-29`, `raw-raw.tar.gz` |
| scope files | `.claude/sprints/tile-efficiency/` (design, build, scratch per role) |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git switch tile-efficiency && git status --short && git log --oneline -3
cat CLAUDE.md docs/performance/evidence/tile-efficiency.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
