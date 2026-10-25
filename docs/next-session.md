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

1. **Prefill and decode emerge from the loop's recurrence (no workload cases)** — Sapir 2026-10-25: "they should be
   proven from the datagraph … based on caching/reusing alone". (a) *Trip batching:* a loop node outside the
   recurrence SCC (`sccs`, `carried_by`), per guard arm on the trip index (`guard_plan`), lifts to a map over `t`
   computed before/beside the loop — a carried array's rows written by such trips (`write_delta`) are filled in one
   batched site; a guard `r <= t` becomes a masked tile. (b) *Recurrent trips on the device:* every node in the SCC is
   placed on the device (decode's row map N1, `mx`/`den` N2), so no host morphism observes inside the loop and the
   steady trip submits without syncs. (c) *Checkpoints:* a carried array whose rows are final at loop entry seeds the
   cache (prefill's `k0` = trip 1's `kk` rows) — compute once across the loop boundary. Missing fact: the per-loop
   trip split (one query in mapal-ir). Acceptance: a natural one-loop generator (guard `t < 512` picks prompt vs
   generated row) must match `pt_prefill`'s hand-written prefill (twin test); causal/decode are fully recurrent
   (`x` row t reads `o` of trip t−1), so their win comes from (b).
2. **Parallel nest cells** — cells now run 1–4 CTAs with one thread folding serially; lanes/tree per cell on the
   default face (order free).
3. **Mixed-radix index fact** → convmc as an implicit-GEMM tile site (N3).
4. H100 128-accumulator cp.async tile (N4, mlp2/diamond/rt_mlp2/fanout/indep).
5. Trace the untraced (N5–N7): 3090 epi/MLP rises, ew_4096 async 1.108, smoke evidence mismatches vs flow-latest.

6. **Test speed** (measured 2026-10-25, differential suite 683 s with the oracle optimized): `differential_device_loops`
   runs 64 programs × 2 arms serially (≈ 600–750 s, ≈ 5 s per emulated run) — spread over the job slots like
   `differential_natural_twins` (≈ 630 s, 256 programs); profile one emulated run (emulator vs per-binary cost);
   macOS first exec of each new binary costs 0.50 s (second exec 0.00 s) — batch binaries or exempt the terminal
   (Developer Tools). The compiler itself: 0.02–0.19 s per bench program (debug build), not the bottleneck.

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
