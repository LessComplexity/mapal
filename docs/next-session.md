# Continue after `cpu-gpu-boundary`

Main checkout: branch **`cpu-gpu-boundary-latest`**. Nothing is pushed. Rules every agent follows:
[`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-10-25-cpu-gpu-boundary.md).

## Where Mapal stands (cpu-gpu-boundary sweep, best-of library legs and torch)

Tables: [`benches/results/evidence/cpu-gpu-boundary/table.md`](../benches/results/evidence/cpu-gpu-boundary/table.md) ·
end check and gaps: [`cpu-gpu-boundary.md`](performance/evidence/cpu-gpu-boundary.md). 41 of 96 workload × arch cells
at or below the best other (flow sweep 26 of 92).

| | result (Mapal / best other) |
| --- | --- |
| at or ahead | conv, ew→conv→ew, FIR, firew, loop_4096, ew_4096 (not H100), causal + prefill on 4090/3090, most 2048 MLPs on 4090 |
| within ~1.0–1.4× | mlp / mlp2 / twin / diamond / fanout / indep / softmax / rowsum / epi / rt_mlp2 / smlp / loopmm; causal + prefill on 5090/H100 |
| far behind | convmc 12–28× (direct conv, no implicit GEMM), decode 1.6–3.0×, attn 1.4–2.0× |

## Candidate next sprint (not yet approved)

Ranked by measured gap ms (H100 worst):

1. **Device loop for attention-style decode** — decode's in-place row map as a device cell (N1, 46 ms), `mx`/`den`
   whole scalar folds on the device (N2, causal 56 ms / prefill 22 ms); then the steady trip submits without host syncs.
2. **Parallel nest cells** — cells now run 1–4 CTAs with one thread folding serially; lanes/tree per cell on the
   default face (order free).
3. **Mixed-radix index fact** → convmc as an implicit-GEMM tile site (N3).
4. H100 128-accumulator cp.async tile (N4, mlp2/diamond/rt_mlp2/fanout/indep).
5. Trace the untraced (N5–N7): 3090 epi/MLP rises, ew_4096 async 1.108, smoke evidence mismatches vs flow-latest.

CUDA graphs: none planned. If nsys ever shows launch rate as the remaining gap, build the graph from the static loop
plan at load time (never stream capture).

## Live state

| item | state |
| --- | --- |
| vast rentals | none of ours; only `stocklab-*` (another project — never touch) |
| worktrees | `.claude/worktrees/wf_233c0550-71b-{1,2,5,6}` + older agent worktrees (finished; prunable) |
| scope files | `.claude/sprints/` (durable; `/private/tmp` is wiped on reboot) |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3
cat CLAUDE.md docs/performance/evidence/cpu-gpu-boundary.md
vastai show instances --raw | python3 -c 'import json,sys; print([(x["id"],x.get("label")) for x in json.load(sys.stdin)])'
```
