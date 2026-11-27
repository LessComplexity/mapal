# 2026-11-27 — machine model step 1, format generics, the language server, the Language Exposition, publish

Previous log: `2026-11-26-readable-llama-and-five-gpu.md`. Written for a compaction: everything needed to continue.

## Live state at handoff (2026-11-27 ~16:55 SGT)

| item | state | how to check / resume |
| --- | --- | --- |
| **cost-model sprint** — workflow `cost-model-calibration`, run `wf_699b1432-f7a`, task `w3naeob4p` | stage 1 "model" finishing its final `cargo test --release --workspace`; stages left: **rental** (≤ $15: calibrate + ABBA on 3090 / A100 / 4090 / H100), **review**, **fix**. Stage 1 committed `41b3f940` (one calibration run measures every primitive) and `1677b7ca` (primitives as machine facts; the evaluator's pipeline calculation); 9 files still uncommitted in the tree are its WIP | journal `~/.claude/projects/-mnt-data-Personal-mapal/8e535edb-996a-45a2-85a6-64c8c6d81eab/subagents/workflows/wf_699b1432-f7a/journal.jsonl`; script `…/workflows/scripts/cost-model-calibration-wf_699b1432-f7a.js`; resume `Workflow({scriptPath: <script>, resumeFromRunId: "wf_699b1432-f7a"})` (never SendMessage a running workflow agent) |
| **CPU re-measurement** for the Exposition's CPU chart — bash task `bi3euupmb` | waits until no `cargo test --release --workspace` runs, then `P1CPU=12 bash benches/local/cpu_shapes.sh <scratchpad>/cpu-shapes-<MMDD-HHMM>` (log `<dir>.log`) | its task output prints the result dir; scratchpad = `/tmp/claude-1000/-mnt-data-Personal-mapal/8e535edb-996a-45a2-85a6-64c8c6d81eab/scratchpad/` |
| **CPU page update — APPROVED TO PUSH** (user: "push the CI fix and the CPU page update") | not started: after the harness → one Sonnet agent replaces the page's CPU chart (today the old M4 Pro table: Mapal threaded vs NumPy 1-thread; sum shows 5× slower) with the fresh Threadripper results, fair legs side by side (1t vs 1t, whole box vs whole box; NumPy, C++ baselines), generated + `--check`, label machine and date → I review screenshots → cherry-pick the docs commit onto `origin/main` in a temp worktree, push, remove the worktree | `python3 docs/tools/exposition.py --check`, `docs/tools/highlight_mapal.py --check` (`MAPAL_BIN=$PWD/target/release/mapal`) |
| **CI** on `main` `fbed8ce9` — watcher task `b0gka8pxk` | running (test jobs now 300 min; the previous run was cancelled at 75 min mid-suite, no failures) | `gh run list --repo LessComplexity/mapal --limit 4` |
| **GitHub Pages** | live: https://lesscomplexity.github.io/mapal/ (index → `language-exposition.html`), from `main` `/docs` | `gh api repos/LessComplexity/mapal/pages/builds/latest` |
| **branches** | `origin/main` = `fbed8ce9` (reviewed, published). `origin/local-rtx-5080` = `d08dd90e` (pushed as backup; contains the sprint's unreviewed `41b3f940`, `1677b7ca` and copies of the commits cherry-picked to main) | after the sprint's review: merge `local-rtx-5080` into `main` (the cherry-picked copies are patch-equal) |
| **GitHub CLI** | active account **LessComplexity** (the user asked to switch and stay); git's credential helper is `gh auth git-credential`, so pushes use it; the other account has no push permission | `gh auth status` |
| **rentals** | none running (only `mapal-*` labels are ours); ≈ $80 of the $100 cap spent before the sprint's ≤ $15 round | `vastai show instances --raw` |
| **CPU boost** | still capped at 4.5 GHz; restore: `echo 5487604 \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null` (thermal reason: CCD1 ~90 °C under many agents) | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq` |
| **worktrees** | none but `.claude/worktrees/_archive-s54-20261021` (old snapshot with raw results not in git — keep) | `git worktree list` |

## Waiting on the user

1. **Trim the sprint's rental to calibration only** (recommended: the pipeline model picks worse than the rules, so ABBA of its picks is predictable; calibration on the four parts ≈ $5–8, ≈ 1 h; then add the missing low-occupancy efficiency fact locally; ABBA only once model ≥ rules). Unanswered → the workflow runs the full round. To trim: stop the workflow right after stage 1's `result` is in the journal, edit the rental prompt in the script, resume from the run id (stage 1 is served from cache).
2. **`mapal build|run --target=native` emits generic x86-64 (SSE2)** — no `target-cpu` in the IR, so `addps %xmm` where the Zen 5 has AVX-512; the benchmark harness passes `-march=native` itself. Fix: give LLVM the host CPU and features when the target is native. Recommended, unanswered.
3. **CLAUDE.md facts-table rows** proposed by the branches (wgmma: `wgmma_half_per_clk`, `Caps::matrix_group`, `MatrixOp::mn_major`, `rules::wgmma16_pays`; mapal-place: `reuse::reads`, `Combine::StreamK { whole }`, …; bounds_proof's `widen_range` and the `%` rule). Not applied. Note: the format-generics prep agent DID rename `to_bf16` → `to<bf16>` in one CLAUDE.md row (flagged to the user, no objection yet).

## Results and findings

- **Cost model, stage 1** (`benches/results/evidence/local-rtx-5080/cost-model.md`): the pipeline over measured primitives made predicted times better (5080 median error 12 % → 8.7 %, bias +12 → +8 %) but **picks worse**: 28 / 76 against the rules' 59 (before: 37). Every loss is one pattern: small / one-warp FMA tiles priced cheaper than they run — the FMA tile's realized rate at one warp a slot and short panels is not captured by per-primitive calibration → needs a measured per-kernel efficiency at low occupancy. **With the rung ON, the local ABBA produced two wrong values** (rung OFF by default, no user impact) — the review must find and fix it.
- **GEMM step 1 (ADR-0050)** — `30f1a32f` crate `mapal-place`, `9d7cebaa` translation + lowerings, `1ee980e8` replay + probes, `c18ce8bf`, `6cb71680` search + rung `machine-place` (OFF), `abd40618` hybrid traversal, `f2c2c68e` review fixes (OFF arm byte-identical on 280/280, one shared verdict: honest replay 37 vs 59, warm operands from `residence_plan`, a values test that caught a single-stage tile storing zeros). **Hybrid ABBA on the RTX 5080: wins all 10 sites, −2.7 % to −13 %** (2048×4096×4096 668.5 → 581.3 µs, past CUTLASS's 590.6). Only the model picks it; the rules don't — so not shipped by default yet.
- **CPU sum (1M f32, this Threadripper 9970X, best of 7):** NumPy 1t 0.140 ms; Mapal `mapal build` 1t 0.070 ms (2×); with `-march=native` (AVX-512) 0.046 ms (3×). The fold is vectorized + regrouped (several accumulators) — the same `step_algebra` regroup the GPU uses; order free under ADR-0040/0049. Whole box no faster at 1M.
- **Fair CPU comparisons that exist** (`docs/performance/cpu/scheduling/dispatch-threshold.md`, M4 Pro / i9, NN chains): multithreaded Mapal beats NumPy on OpenBLAS / Accelerate 1.26–1.36× at d = 2048; single-threaded Mapal is 0.60–0.87× of NumPy; small sizes NumPy.
- **Memory (five-GPU round):** f32 class 2.7–3.4 GB vs ≥ 5.1 GB for every f32 engine (mostly bf16 weight storage); bf16 class level with llama.cpp / HF eager, 20–30 % below gpt-fast; pooled servers reserve 9–47 GB by design. Mapal's 3090 peak unrecorded; SGLang 3090 measured as growth.

## What landed today (in order)

- ADR-0050 the machine model (`1cbcfe89`; design check in `.claude/sprints/machine-model/design-check.json`).
- Language server: `mapal lsp` (crate `mapal-lsp`), one class table for Neovim and VS Code, gd through LazyVim's Snacks picker (`27c30064`). VS Code extension 0.4.0 bundles the server for linux-x64 (`eaffd809`, `04a60d76`).
- ADR-0051 format generics: `to<target>` (rounds) / `widen<target>` (exact, proven), `name: kind` parameters (float / narrow / int / number), explicit stage arguments as a prefix, the rest inferred; seven old names removed (L1713); every source rewritten, same IR (`43a1b61d` ADR, merged `d600b0dc`). Generic KV cache in `examples/llama.mapal`.
- GEMM step 1 (above).
- The Language Exposition `docs/language-exposition.html` (old explorer redirects): example-first, demos with real outputs read from data files, performance by problem class (GPU families, named samples, CPU vs NumPy — to be refreshed), the architecture as a steppable equation, components, artifacts walk, the "where Mapal is different" map, In depth. Dark by default, the real logo, 18 px text, sections up to 1600 px. Generated by `docs/tools/exposition.py` + `highlight_mapal.py` (+ `mermaid_render.mjs`: diagrams pre-rendered with mermaid 12.3.0, mermaid.live's engine and font, identical placement), `--check` for every region.
- Diagram modes from the compiler: `mapal dump-ir --mermaid` = condensed **string diagram (default)**; `=string`, `=category[,condensed]`, `=graph` (stored graph), `=flow`.
- README rewritten short (`408c64ef`, `803f9900`, `3c07aac6`); details moved to `docs/getting-started.md`, `syntax.md`, `philosophy.md`, `roadmap.md`, `performance/results.md` (the chart script writes its tables there), `STATUS.md`, the editor READMEs, `CONTRIBUTING.md` (`962682fa`).
- Pre-publish fixes: CUDA differential harness caps GPU runs by free memory (`fbaccff8`); testgen old names (`42c5836e`); rustfmt (`6396cb50`); evidence CSVs with a local process line redacted (`fada4a06`, user chose clean-in-tree, history keeps the original); font license (`f50c22e4`).
- Published `main` = `6396cb50` → Pages; then `2854a26a` (emulator: `emu_smem` 32 KiB alignment for macOS + descriptor decode relative to the buffer's low bits; 600 s emulated-run guard), `fbed8ce9` (CI 300 min; "made for AI agents" removed — a side effect, not an idea).

## User decisions today (binding; also in memory)

- **Hardware = abstract compute + placement facts; the dataflow is mapped onto them by one cost model + search; lowering after; target-specific steps after that, minimal** (ADR-0050). The cost model is a **calculation**, not a simulation; parameters: memory placement, movement, compute capabilities, result movement.
- **Format generics** as above; two verbs; kinds; old names removed.
- **One tree; worktrees only for completely different parallel work**, merged and removed when done.
- **Docs / pages / mechanical work: one Sonnet agent, Opus (me) reviews** — no multi-agent workflows for them; workflows for compiler work.
- **Exposition style:** examples and demos first, big text, dark default, real logo, wide sections, architecture as diagrams, depth last; demo outputs generated.
- **README:** short plain exposition (first look, ideas, results charts + three sentences, where it stands, try it, learn more); no Llama listing, no jargon, no "AI agents" idea.
- **IR diagrams:** the condensed string diagram is the default (CLI and pages); the category diagram second.
- **Publishing:** user approves pushes to `main`; the active gh account is LessComplexity.

## Open work after the sprint

- Decide the cost model from the sprint's review (add the low-occupancy efficiency fact; model ≥ rules on the replay; ABBA; then default ON — which ships the hybrid).
- The resource round (local 5080, after GPU work): GPU memory split, host RSS, cold/warm start incl. compile / engine build, energy per token, install footprint, model source lines, launches per token.
- The shorter generic Llama: format aliases at the top (`type CacheElement { _: bf16 }`), `Matrix<…, element: float>`, precision annotations taking a format parameter → one file for both classes, `variant.py`'s rewriting gone.
- Language roadmap: coproducts ADR next (sum types + a flow `match`), then function values (specialized away), abstract types with contracts (maps/sets), bounded run-time-sized arrays; from the Abstract comparison: spelled-out float formats (`f32s1e8m23`) and declared value ranges as checked facts.
- ADR-0051 leftovers: `u8`/`i64` sources, `f64 → f32`, float → int, a map body's expected element type as an inference source.
- String diagram: loops as boxes (today a `⟲` node).
- ADR-0050 later steps: attention / streamed units, cells and folds, co-schedule and residence, the CPU translation, AMD.

## Machine notes

Threadripper 9970X (64 threads) + RTX 5080 16 GB; `CARGO_BUILD_JOBS` up to 24 ok. CUDA differential tests: GPU runs capped by free memory (`MAPAL_GPU_JOBS` overrides). Emulator tests are slow (minutes each). `~/.local/bin/mapal` = the merged build (has `lsp`, ADR-0051 syntax). VS Code: Code - OSS, extension 0.4.0 with the bundled server.
