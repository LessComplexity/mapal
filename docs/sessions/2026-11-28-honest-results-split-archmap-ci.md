# 2026-11-28 — honest results, the 1,000-line split, the Architecture Map, short CI

Previous log: `2026-11-27-machine-model-format-generics-exposition-publish.md`.

## Live state at close

| What | State | Resume / inspect |
| --- | --- | --- |
| Branch | `main`, tracking `origin/main` (`local-rtx-5080` retired, same commit) | `git status -sb` |
| GitHub `main` | `a639d079`, CI green, Pages deployed | `gh run list --repo LessComplexity/mapal --limit 5` |
| Uncommitted work | this session's results / page / tooling changes (below) — **the user commits** | `git status --short` |
| Raw evidence release | `evidence-raw-2026-11-28`: the five-GPU + gapfix-integrate round, 1,586 files, redacted (username / local paths), rows in `RAW-manifest.tsv` (uncommitted) | `gh release view evidence-raw-2026-11-28 --repo LessComplexity/mapal` |
| Rentals | none (≈ $86 of $100 spent overall) | `vastai show instances` |
| Running jobs | none | — |
| CPU | the user capped clocks at 4.5 GHz after a 95 °C run (my oversubscribed nextest run) | `sensors \| grep Tctl` |
| NumPy for benchmarks | the system NumPy is linked to the **reference BLAS** (≈ 250 ms for a 1024² matmul): use a venv with the official wheel (OpenBLAS) | `uv venv np && VIRTUAL_ENV=np uv pip install numpy threadpoolctl` |

## Landed (pushed to main during the session)

- **Identity:** every `sapiritur` commit rewritten to `Sapir Shemer <lesscomplexity@gmail.com>` (user ran the rewrite scripts); a stale local `main` was force-pushed by my script and lost 4 commits — rebuilt and fast-forwarded (`45422a3c`). Repo git config now sets the identity. Backup of the old history: `/mnt/data/Personal/mapal-backup-before-identity-rewrite.git` (delete when satisfied).
- **Cost-model sprint merged** (machine-place stays OFF: the model picks 32 of 76 sites, the rules 59); `mapal repl`; the clearer L1605 message.
- **Every code file ≤ 1,000 lines** (pure-move split, byte-identical emission on 325 programs × 9 profiles; CI `scripts/check_file_size.sh`).
- **Architecture Map** (`docs/architecture/map.json` → `docs/tools/archmap.py` → `docs/architecture-map.{html,md}`, 871 code references, CI `--check`), linked from the README and the Exposition, on Pages.
- **Short CI:** per push `checks` + `nextest --profile quick` (1,731 tests; Ubuntu 5.5 min, macOS 10.5 min); nightly `Full tests` (all tests, sharded). Slow list `.config/slow-tests.txt` → `scripts/slow_tests.py` → `.config/nextest.toml` (locally the slow tests run two at a time).
- VISION.md removed (and other "slop" removals by the user).

## Uncommitted at close (the user commits)

1. **Llama results relabelled** (README, Exposition, results.md, `full-model.json` labels, charts): like for like is bf16 — Mapal behind on all five GPUs (1.06–1.57× per token, 1.05–2.37× to the first token). The "f32 class" is Mapal's f32 math on the bf16 weights against engines on f32 weights: **not like for like** (Mapal reads half the bytes). The H100 "faster for short prompts" line (a later run against gpt-fast only) removed.
2. **GPU suite without averages:** wins / level / losses per GPU (5080 46/43/30, 3090 39/30/50, A100 57/11/51, 4090 49/24/46, H100 30/8/81 of 119), every program a bar; class blurbs rewritten from counts (matrix products the weak class: 15–40 of 47 slower; attention's step-loop wins are against a PyTorch Python loop).
3. **CPU results rebuilt with statistics:**
   - Baselines fixed: FIR/conv2d register-blocked tiles (a run-time tile size made clang reload every output per tap: 2.6 vs 0.95 ms), blocked transpose, a thread pool before the timer (`benches/shapes/pool.h`), the threaded sum's 4 MB-per-thread copy bug fixed, strict and `-ffast-math` builds (the faster used).
   - `benches/local/stats.py` (median ratio, 95 % bootstrap interval, Mann–Whitney; a verdict only when the interval excludes 1 and p < 0.05), `shapes_summary.py`, `cpu_numpy_mt.sh` (refuses a NumPy without a threaded BLAS).
   - Results (31 runs; Threadripper 9970X, 4.5 GHz cap): one thread vs careful C++ — Mapal wins conv2d 1.52×, C++ wins transpose 1.25× / gather 1.23× / FIR 1.05× / saxpy 1.04×, sum level; vs NumPy 1t Mapal 1.4–24× on all six; whole machine vs C++ — sum C++ 2.27× (Mapal's 64-thread sum slower than its 1-thread: open), gather Mapal 1.26×, FIR/conv2d level. Whole machine vs multi-threaded NumPy (OpenBLAS, 21 runs): Mapal 1.09× (mm 1024), 1.57× (2048), NumPy 1.62× (4096), networks 1.25–1.33× (512), level (1024), 1.09–1.27× (2048).
4. CI nightly split 8 ways (four ways ran 1–4 h; part 2 hit 240 min).
5. `archive_raw.sh` `RAW_ALL=1` (the default filter skipped the evidence tables).

## Decisions (the user's)

- The user commits and pushes; I leave changes uncommitted. Work on `main`.
- No code file over 1,000 lines; tests in their own modules ([[file-size-cap]] memory). The user will refactor the architecture by hand — no broad AI refactors unless asked.
- No means/geomeans in published results: winners and losers apart; every comparison with an interval and a significance test; baselines written carefully, not naive.
- Not added to CLAUDE.md for now (file-size rule); `GpuRungs` macro chain kept.

## Open

- **Fair GPU attention baselines:** `causal_*` / `gen_*` / `prefill_*` / `decode_*` need an SDPA / FlashAttention leg and a whole-loop CUDA-graph leg (`benches/gpu/torch_bench.py:708-724`).
- **Both GPU clocks shown:** the program-boundary clock (current) and the event bracket (`dev_med_ms`) per row.
- **A 3×TF32 competitor leg** (CUTLASS) for f32 products, so the tensor-core f32 split meets a same-accuracy rival.
- **GPU suite statistics:** intervals from per-run samples, replacing the fixed max(null, 5 %) band.
- **CPU open problems:** Mapal's whole-machine sum slower than its one-thread (parallel dispatch at small sizes); 4096 matmul k-blocking (`kc_nest` OFF; the packed B panel fills L2) — diagnosed in `benches/results/evidence/local-rtx-5080/cpu-analysis.md` §4.
- Two pre-existing dead drift rows: `backend-nvptx/IMPLEMENTATION.md → profile.rs:RungOff`, `backend-llvm/IMPLEMENTATION.md → func/mod.rs:checkpoint_injection`.
- Delete the identity backup mirror when satisfied; `examples/softmax.mapal` carries the user's uncommitted edit (the regenerated Exposition shows it).
