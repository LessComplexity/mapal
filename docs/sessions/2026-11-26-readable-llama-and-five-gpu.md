# 2026-11-24 → 2026-11-26 — the readable Llama, four language ADRs, five-GPU round, Hopper wgmma

Branch `local-rtx-5080` (never pushed). Continues `2026-11-24-local-rtx-5080-compiler-sprints.md`. Machine: Threadripper
9970X + RTX 5080, CPU boost capped at 4.5 GHz (see Machine notes). Rented GPUs on vast.ai. Times are Singapore (UTC+8).
166 commits since that log's `54afa0b1`.

## Live state at the end of this log (2026-11-26 14:25)

| what | where | resume / inspect |
| --- | --- | --- |
| workflow `hopper-wgmma` (run `wf_795b722f-5e8`, task `wmyczwhj9`) — implementers DONE (cached), **integrate running** (restarted after a CLI exit), then ONE review, then fix | script `~/.claude/projects/-mnt-data-Personal-mapal/8e535edb-996a-45a2-85a6-64c8c6d81eab/workflows/scripts/hopper-wgmma-wf_a0d49bd2-c77.js`; journal `…/subagents/workflows/wf_795b722f-5e8/journal.jsonl` | if it died: `Workflow({scriptPath: <script>, resumeFromRunId: "wf_795b722f-5e8"})` (implementers replay from cache). Progress: count `"type":"result"` lines (2 done; integrate, review, fix to come) |
| main checkout | HEAD `16f46432` (both hopper branches merged); **uncommitted edits** in README.md, open-rules.md, func/gpu.rs, rules.rs, rung_replay.tsv = the interrupted integrate's start on the mma-walk Ampere fix; saved at `.claude/raw-evidence/hopper/integrate-interrupted/wip.patch`; the resumed integrate owns them | `git status`, `git log --oneline -5` |
| rentals | **none running**; spend since the $100 cap was set ≈ $80 (confirmation round $15.67, fix agents $3.82, gap-fix ~$0.1, five-GPU $30.09, hopper-wgmma ~$25–30); all rounds since 2026-11-19 ≈ $131 | `vastai show instances --raw` — only `mapal-*` labels are ours |
| still to do after the workflow | (1) refresh `docs/summaries/2026-11-25-compiler-and-llm-summary.md` (untracked; the presentation data summary, follow-up to `TND + Mapal Update.pdf` of 2026-10-28) with the wgmma + regression results; (2) README results: H100 bf16 after wgmma (re-run `benches/results/charts/make_charts.py` after updating `full-model.json`); (3) next sprint (Open work below) | — |

## Results at the end of this log

**Full model** — Llama-3.2-1B-Instruct, real weights, readable `examples/llama.mapal` (404 lines), batch 1, prompts
128/512/2048, 128 greedy tokens. Mapal ÷ **best engine of the same class** (the fastest of gpt-fast, vLLM, SGLang,
TensorRT-LLM, llama.cpp, HF eager, HF static+compile per cell; type to type). Data: `benches/results/evidence/full-model.json`.

| GPU | f32 TTFT 128/512/2048 | f32 decode | bf16 TTFT 128/512/2048 | bf16 decode |
|---|---|---|---|---|
| RTX 5080 | 0.62 / 0.62 / 0.53 | 0.55–0.58 | 1.13 / 1.05 / 1.15 | 1.06–1.10 |
| RTX 3090 | 1.04 / 0.95 / 0.82 | 0.65–0.67 | 1.55 / 1.17 / 1.30 | 1.23–1.25 |
| A100 | 0.73 / 0.52 / 0.43 | 0.86–0.91 | 1.87 / 1.44 / 2.12 | 1.55–1.57 |
| RTX 4090 | 0.72 / 0.64 / 0.53 | 0.62–0.63 | 1.44 / 1.16 / 1.24 | 1.18–1.19 |
| H100 | 1.04 / 0.74 / 0.50 | 0.93–0.95 | 1.30 / 1.88 / 2.37 → **after wgmma 0.75 / 1.25 / 1.58** (gpt-fast same box; gpt-fast was the H100's best everywhere) | 1.49–1.56 |

f32 class: tokens 128/128 = HF f32 on every GPU; Mapal leads because it streams the 2.5 GB bf16 weights and widens them
exactly in registers (f32 engines stream a 4.9 GB copy). SGLang has no f32 path; TensorRT-LLM's f32 output is wrong
(0/128). RTX 5080 absolute (final): f32 TTFT 8.54/25.39/95.48 ms, decode 3.24–3.45 ms/token; bf16 TTFT 5.58/14.07/52.14,
decode 3.26–3.40. H100 after wgmma: bf16 TTFT 2.70/5.60/18.11 ms vs gpt-fast 3.58/4.47/11.45 (same box).

**Kernel suite** (120 programs × 5 GPUs, Mapal ÷ fastest competitor at the same precision, program-boundary timing):
RTX 5080 0.876 (46/44/30) · RTX 3090 0.920 (39/30/51) · A100 1.006 (57/11/51) · RTX 4090 0.911 (49/25/46) · H100 1.101
(30/8/82); attention 0.52–0.79, f32 0.77–0.90, tf32 1.15–1.72, bf16-weight rows 1.46–1.63; GEMM the weak family on
3090/A100/H100. Cells at or below the best: 279 of 599. `benches/results/evidence/matrix.{md,json}`.

## User decisions made in this stretch (all in CLAUDE.md and/or memory)

- Rental cap **$100 total** (no per-round asking below it).
- **Full model** in readable Mapal (descriptive names, small functions, like `examples/sepia.mapal`); Llama-3.2-1B-Instruct
  via the ungated mirror; real weights via a new **`"file" -> read`** builtin.
- **Type to type:** bf16 class only vs bf16 engines, f32 only vs f32-compute engines; never a gain across classes; compare
  Mapal vs **the best** engine, not a named one.
- **Language surface:** `x -> sqrt / sin / cos` (builtins = contract + reference definition + per-target overrides admitted
  by the contract, chosen by properties — CPU/GPU/ASIC); size-generic functions `fn project<rows, columns>(…)` (declared
  size list); no `const` — everything immutable unless `mut`, top-level `2048 -> hidden_size;`; the compiler computes
  any non-mut value the dataflow proves computable beforehand; `type T { _: … }` — `_` is a type's default interpretation
  (alias); size-generic types `Matrix<rows, columns>`.
- **ADR-0049 — correctness is the flow and the bound, not bit-identity:** order free (atomics, stream-K OK; last bits may
  differ between runs); never trade speed for reproducibility; exact/contract/fixed are development modes. Overrides the
  bit-identity framing (CLAUDE.md rule updated).
- **Disassembly comparison** against the leader is the standard way to study losses and regressions ("from compile path to
  asm/ptx to runtime").
- **Token efficiency:** one agent per implementation, one review per batch; never message a running workflow agent (it
  spawns a duplicate — memory `workflow-agent-messaging`).
- **README structure:** philosophy first (the language IS the flow; data category-theory based; data+flow beats control;
  … "AI agents can write and verify" last), then example, syntax, short results with charts + link to the full matrix,
  roadmap. **Roadmap:** AMD backend + PyTorch importer next; then dev → deploy → verification → synthesis (ASIC lab).
- **CPU heat:** the user set a 4.5 GHz boost cap (`echo 4500000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq`;
  undo with 5487604; resets on reboot).

## What was done, in order (merge commits on `local-rtx-5080`)

1. **confirmation-fixes** (fp8 PTX ISA as a capability, measured properties, tensor-core splits, Hopper launch path,
   attention/chains) 530c5231…3bd55d42; review fixes 2972c3d1 (5080 column corrected 0.965 → 0.978).
2. **Rented confirmation round** f1d3a04f, merged f41d4a83 ($15.67).
3. **Gap analysis by disassembly** (51 loss rows on the 5080 vs the leaders' SASS, 14 causes) fac7691b; **gap fixes**
   57f3ea8f / c98931a1 / d945c641 + harness program-boundary timing (H) → 5080 0.978 → 0.878; review fixes 3a586696.
4. **Language:** ADR-0045 `read`, ADR-0046 sqrt/sin/cos + builtin-realization framework, ADR-0047 size generics, ADR-0048
   top-level bindings / compile-time evaluation / `_` / generic types — merged via model/language de12816c.
5. **Model-ready** (structure: arrays by address, nested loops via calls, ordering fixes; placement: decode on device)
   fa147d70 / d9244769; full model with real weights 16d668c6; zero-hoist miscompile fix 151953b5.
6. **Prefill fixes** (precision: shared-body store fix, split-rate; residency: once-range, one weight copy, carry-take,
   calibration supply; mask: flash attention through `positions[t]`) 4420f00c / 06b708ed / a29a219b; token-claim checker
   92420a89.
7. **bf16 sprint** (mma-walk stream-K tensor tile; causal skip with finite-check; bf16 KV cache windows; round-post;
   device-trap; cand-made) 08faa02c / b80bd696 / e3a31d1e; review fixes 9be9886d (device-trap hang, checker bugs, safety
   net).
8. **Five-GPU round** c808622c ($30.09); **bound fix** 9a8bcd07 (the gate's bound was unsound — a composed bound,
   `mapal run --interp --bound`, ADR-0041 amended); **ADR-0049** 47346059; **README rewrite** aa372fce (charts from
   `matrix.json` + `full-model.json` via `benches/results/charts/make_charts.py`; old results in
   `docs/performance/earlier-readme-results.md`).
9. **Hopper wgmma + regression study** (running workflow): 16-bit `wgmma.m64nNk16` tiles as capability geometry, one tile
   family with TF32 wgmma (779f4a15; H100 bf16 projections 1.13–2.23×, 2048×28672×4096 1.05× cuBLAS; merged-launch
   crash worked around — warp-specialized kernels launch alone, root cause open); regression study 16f46432: **one Mapal
   regression** — mma-walk on Ampere (3090 pj_m2048_n4096_k4096_bf16 1.10 → 1.48; A100 down projection 788 vs 761 µs: the
   walk loop has half the static HMMA + RED + a zero pass) — being fixed by the resumed integrate; everything else box
   properties (4090 last round clock-locked at 1.2 GHz; H100 host round trips 2–3×, 500 → 700 W; A100 ±50 % noise + tuner
   reverting good picks).

## Open work (measured, not closed)

- bf16 class behind on the rented GPUs (table above); H100 still 1.25 / 1.58 at 512/2048 after wgmma; next: flash
  attention on wgmma (FA3-style), persistent walk, the merged-launch crash's root cause.
- The mma-walk Ampere regression (in the resumed integrate).
- Decode vs gpt-fast: argmax on the host (75–126 µs/step), K/V projections at ~500 GB/s (16-byte loads), q/k/v not merged
  with their rotations (walk order), 288–323 launches/step.
- Properties: host round-trip cost as a measured property; tuner revert only when pairs agree beyond null.
- Bound quality: the composed bound is sound but ~50× the actual error on one program; tighten (running error analysis);
  f32 gates not switched to it (decision pending).
- tf32 rows and GEMM on A100/H100; multi-channel conv; first-launch host cost; split-combine leftovers.
- Roadmap: AMD backend, PyTorch importer, then the ecosystem; batch > 1 and serving features for the model.

## Machine notes

- Tools: CUPTI ktrace / modcap / sass_of.sh (`.claude/raw-evidence/local-rtx-5080/gaps-disasm/` and scratch), no
  nsys/ncu locally (ncu also refused on vast boxes); competitor venvs under `~/.cache/mapal/` (torch, vllm, sglang,
  trtllm), llama.cpp in `~/.cache/mapal/llama.cpp`, harness `~/.cache/mapal/model-compare/`, weights
  `~/.cache/mapal/models/llama-3.2-1b-instruct/` (+ `mapal/` raw bf16, `-gguf/` bf16/f16/f32).
- `/tmp` is RAM; keep cargo target dirs on disk. Old worktrees under `.claude/worktrees/` (43) can be pruned once merged.
- Calibration: `native-gpu` reads `MAPAL_GPU_CALIB`, the user cache, then the repo's committed lines (warns loudly if none).

## Evidence index

`benches/results/evidence/` — `matrix.md/json`, `full-model.json`, `local-rtx-5080/{full-model,gaps-disasm,prefill-*,bf16-*,
integration}.md`, `cross-arch/README.md` (every rented round incl. "b8429388 / 9be9886d" and the hopper section to come),
`hopper/{wgmma-16bit,disasm-rented}.md`; ADRs `docs/decisions/ADR-0045…0049`; presentation data summary
`docs/summaries/2026-11-25-compiler-and-llm-summary.md` (untracked); raw under `.claude/raw-evidence/` (not in git).
