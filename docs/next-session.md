# Continue after `recurrence-precision`

Main checkout: branch **`recurrence-precision-latest`**. Nothing is pushed. Rules every agent follows:
[`CLAUDE.md`](../CLAUDE.md) (`AGENTS.md` → symlink). History: [session log](sessions/2026-10-26-recurrence-precision.md).

## Where Mapal stands (recurrence-precision sweep, best-of library legs and torch)

End check and gaps: [`recurrence-precision.md`](performance/evidence/recurrence-precision.md) · raw:
[`benches/results/evidence/recurrence-precision/`](../benches/results/evidence/recurrence-precision/) (only changed
programs re-run; other rows carried, emission identical). 54 of 120 workload × arch cells at or below the best other.

| | result (Mapal / best other) |
| --- | --- |
| ahead | causal 0.16–0.54, prefill 0.16–0.53, natural one-loop gen 0.12–0.52 (= hand-written prefill), conv / FIR / ew, rt_mlp2 on 4090/3090 |
| about even | rowsum, softmax, 2048 MLPs on 4090/5090, convmc on 5090 (1.04) |
| behind | decode 1.03–2.48 (H100 worst), convmc 1.29–1.68, H100 FP32 GEMMs 1.17–1.39, `_tf32` GEMMs vs torch TF32 1.06–1.87 |

## Candidate next sprint (not yet approved) — ranked by measured gap

1. **Measure only what changed, by default** (R7): builders run the affected crates' tests while iterating (full suite
   once at integrate); every box run keys builds and timings by emission hash × arch × face and re-times only changed
   rows (the narrowed sweep did 3–30 min per box vs ~2 h); `differential_device_loops` over the job slots; fewer new
   binaries (macOS first exec 0.5 s each). Sweep bundles validated locally before renting (15 of 28 rentals wasted).
2. **Decode's per-trip GEMV on more blocks** (R1, H100 31 ms): the row kernel runs 1 block × 256 threads; the
   library's GEMV runs 64–256 blocks.
3. **H100 FP32 tile** (R2, C6a): 128 accumulators + cp.async stages as on 4090/3090; wave quantization on 114 SMs.
4. **`: tf32` through the radix tile and TF32 GEMM waves** (R3, R4): convmc_tf32 emits FP32 FMAs; 2-warp convmc blocks.
5. Small: 1c checkpoint seeding and the `pt_gen` launch twin (R5); `fold-lanes` reads the input layout as a fact
   (R6); runtime spin rule vs external load (R8, measure first).

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
