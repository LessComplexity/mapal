# Continue: sprint `cpu-gpu-boundary`

Main checkout: branch **`flow-latest` @ `6369853`** (flow ON on all four parts). Nothing is pushed. Rules every agent
follows: [`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-10-25-optimization-sprints-handoff.md).

## Where Mapal stands (latest sweep, best-of library legs and torch)

Tables: [`benches/results/evidence/flow/table.md`](../benches/results/evidence/flow/table.md) ·
causes: [`why-slow.md`](performance/evidence/why-slow.md).

| | result |
| --- | --- |
| at or ahead of best other, every part | conv 5×5, ew→conv→ew, FIR, loop_4096; most 3090 rows |
| within ~1.0–1.4× | mlp / mlp2 / twin / diamond / fanout / softmax / rowsum / epi |
| far behind | prefill 5.2–8.6×, causal 2.7–5.6×, attention 4.2–8.8×, rt_mlp2 2.3–5.6×, ew_4096 2.3–5.3×, pj 1.6–4.3×, convmc (host only) |

**Why:** the big gaps are the CPU↔GPU boundary, not kernels — work left on the host because a fact/recognizer
rejects it, host round-trips every loop step, allocation / lazy loading inside the clock, pageable copies.

## Next sprint `cpu-gpu-boundary` (approved; ~300k tokens; Opus build, Sonnet sweep)

Scope file: `.claude/sprints/cpu-gpu-boundary-scope.md` (copy below). Every item through the evidence pass
(kernel + movement + launch intents; CPU-only emission must stay byte-identical or be measured ≥ parity).

1. **Complete placement** — what the graph proves can run on the device runs there:
   guard_plan predicate into strided cells + drop the retained-site fold veto (`resident.rs:1091`) → attention /
   causal / prefill; scalar captures as kernel args in device generation (`resident.rs:612`) → rt_mlp2, decode `q`;
   decode's in-place row map as a device cell; a mixed-radix index fact in mapal-ir → convmc as implicit-GEMM site.
2. **No host round-trip inside a loop** unless a host morphism observes: a steady trip with the same launch sequence
   (loop_plan) is submitted without host syncs; CUDA-graph capture where it pays.
3. **No allocation / loading in the timed path:** arena allocated at function entry (static shapes); eager module load.
4. **Transfers:** pinned host buffers; async copies overlapped where path_plan proves independence.
5. Small: gate for the lone tree-fold class (rtx-4090 rowsum_4096 1.06 with flow ON); check H100 retain-resident ON on
   decode, then flip the row (causal 0.678, prefill 0.787 already measured).
Deferred: H100 128-accumulator cp.async tile (mlp2/diamond); tile sites as flow-unit members; conv baselines (direct
cuDNN `cudnnFind`, CUTLASS conv, naive CUDA); StableHLO/PJRT coverage only after parity on all workloads.

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours; only `stocklab-*` (another project — never touch) |
| worktrees | 19 agent worktrees under `.claude/worktrees/` (finished; prunable) |
| scope files | `.claude/sprints/` (durable; `/private/tmp` is wiped on reboot) |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3
cat CLAUDE.md docs/performance/evidence/why-slow.md .claude/sprints/cpu-gpu-boundary-scope.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
