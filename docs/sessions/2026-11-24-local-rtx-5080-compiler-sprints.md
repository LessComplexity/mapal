# 2026-11-19 → 2026-11-24 — local RTX 5080 box: competitor runs, compiler sprints, five-GPU matrix

Branch `local-rtx-5080` (never pushed). Machine: Threadripper 9970X (32C/64T, 4 CCDs, Zen 5) + RTX 5080 (sm_120,
84 SMs, 64 MiB L2), Arch, CUDA 13 (`/opt/cuda`, driver 620.73 since a system update), clang/LLVM 23. Rented GPUs via
vast.ai (IturAI account; the `vastai` CLI here has no `--profile`). Times are Singapore (UTC+8).

## Live state at the end of this log (2026-11-24 ~17:30)

| what | where | resume / inspect |
| --- | --- | --- |
| workflow `confirmation-fixes` (task `wk8nbdqx2`, run `wf_04f1c895-2ee`) — 4 parallel worktree agents → integrate → review → fix | script `~/.claude/projects/-mnt-data-Personal-mapal/8e535edb-996a-45a2-85a6-64c8c6d81eab/workflows/scripts/confirmation-fixes-wf_04f1c895-2ee.js`; journal `…/subagents/workflows/wf_04f1c895-2ee/journal.jsonl` | `Workflow({scriptPath: <script>, resumeFromRunId: "wf_04f1c895-2ee"})` (completed agents replay from cache); progress: `grep -c '"type":"result"' <journal>` (4 parallel + integrate + review [+ fix]) |
| branches being written | `fix/bugs-properties` (CPU 0-7,32-39, no rentals), `fix/tensor-cores` (16-23,48-55, $5 A100/H100), `fix/hopper-launch` (24-31,56-63, $7 H100), `fix/attention-chains` (8-15,40-47, $3 A100) in `.claude/worktrees/wf_04f1c895-2ee-*` | `git log --oneline 8ff66d34..fix/<name>` |
| rentals | user-approved cap **$15 total** across the four agents (labels `mapal-tc-*`, `mapal-hl-*`, `mapal-ac-*`) | `vastai show instances --raw` — only touch `mapal-*`; `sapir v4-*`, `ci-bench-*`, `stocklab-*` belong to other projects |
| next after the workflow | merge check, then ASK the user before a confirmation round on 3090/A100/4090/H100 (last one cost $14.12) | rounds follow `benches/results/evidence/cross-arch/README.md` sections |

Expected finish of the workflow: Wed 2026-11-25 ~01:00–04:00.

## Machine notes (also in memory `local-rtx-box`)

- `/tmp` is tmpfs (RAM). Session scratch (`/tmp/claude-1000/...`) grows to 40+ GB of bundles; delete finished steps'
  dirs (`rm -rf "${P:?}/${d:?}"` form). The torch competitor venv (torch 2.11+cu130, FlashAttention-2 built from
  source, FlashInfer, Liger, Triton) lives durably at `~/.cache/mapal/venv-torch-cu130` (`TORCH_PY`; use
  `python -m pip`, its console-script shebangs point at the old scratch path).
- Thermal: CCD1 (CPUs 0-7,32-39) runs hottest; tests spawn programs with 64-lane pools → 91 °C once. Rules: pin heavy
  work (`taskset`), `RUST_TEST_THREADS≈8–16`, `MAPAL_PAR=2–4`, `MAPAL_TESTGEN_JOBS`, pause above 85 °C.
- GPU timing: one 5080 shared through `flock <scratch>/perf/bench.lock`; wait for no `MapleStory.exe` (the user's game)
  and load < 3; the game added 26–34 % noise.
- `~/.cache/mapal/native-dispatch.tsv`: the native host's measured dispatch cost; a busy or debug measurement is no
  longer cached (13e57ad4); a bad 32-lane line was removed (backup `.bak-20261122`).
- Docker container `keel` (another project): never stop/restart it. Its keel-gate hit Telegram HTTP 429 (no second
  poller on this machine). A clean `claude --remote-control keel` session was started in a kitty window
  "claude rc: keel (clean)"; a first attempt that inherited this session's env ("claude rc: keel") may still be
  open — the user closes it.
- Old worktrees under `.claude/worktrees/` (agent-*, wf_*) can be pruned once their branches are merged.

## User rules added this session (CLAUDE.md + memory)

- **Hardware is physical properties, never a name/version** (f3a9e792): no per-architecture on/off; `GPU_RUNGS_OFF`
  removed (24a92033); every measurement is a replay line (`rung_replay.tsv`); `native-gpu` = device query + calib.
- **Capabilities are geometry** (34bc9705): mma shapes, operand/accumulator homes (TMEM), TMA boxes, clusters, barriers.
- **No runtime graph recording** — GPU work is merged at compile time, vertical (calls) and horizontal (data).
- **`exp` is a pipeline builtin** `x -> exp`, never `exp(x)` (ADR-0042).
- **Open problems are attacked, not just recorded** (e85f7c82); generic first; tailored fixes only when no abstraction
  exists, listed in `docs/tailored-fixes.md` (still empty).
- Evidence flow: changed-programs ABBA (`benches/local/gpu_changed.sh`) + one full suite per step; `BUNDLE_CACHE`;
  parallel generated tests (user kept this rewrite of the evidence rule).
- Token efficiency: no fan-out for simple audits; workflows for big parallel work (ultracode).

## What was done, in order (commits on `local-rtx-5080`)

1. **Local competitor runs + analysis**: harness + profile d6e266c; evidence 461085bb, 78bce3c9; CPU dispatch fact
   1ffe0137 (lanes = √(T/w)); GPU analysis + calibrated rtx-5080 DeviceCost ba15bbda; site choice at module load
   0fa2643f (removed 50–90 µs per program).
2. **fold-algebra** (`step_algebra`: FMA pairs, regroup) 76199924, 74960f5d; CPU 1M sum 14× (4.4× faster than NumPy);
   attention claim corrected 4206028f.
3. **compile-time-graph** 2521a051, 0ca6347e, 49430fc8 (row-block units; CUDA-graph replay later removed per user).
4. **kernel-merging**: groundwork 0e8adcdf/f18bd484 (access_plan, residence_plan, zero-fold; decode 0.88×);
   rows 66a10923; horizontal aad8e3e3 (decode 0.66×, causal/gen 0.77×); calls 47ad30aa/fb37a9fb/11b124a2.
5. **properties-and-big-models**: GPU_RUNGS_OFF → property rules 24a92033/fc56a298; model-scale Llama-layer suite
   5fdace7c/5fb1b3b0; long rows (`fold_combine`) cf40019b; onto-device (bounds_proof by guards/loop ranges, i64,
   host-par) 7942e6d7/8bba2a50 (decode 218 → 0.67 ms); device geometry (tile-post, head-group, piece-delta,
   row-tree) 6c372a58/5b6c0a29; exp b2d1c1c9/faa1f06f; cross-arch rules 7eef8687/a354f221; large GEMM
   ed684a45/7a2b67bf (4096³ 1.01× cuBLAS); half precision bf16/f16 storage (ADR-0043) 3b404428/d1927ac1; review fixes
   af0da8a5/87780c58; calls priced by calibrated+derived body cost 65edf4c4/ddfc0378; named-host dispatch
   13e57ad4/60f0f8ce; streams test 7c227fd3; bench programs formatted by the generator 180e99c5.
6. **gaps-and-tensor-cores**: faster harness b9893dee/52b0b18f/6a763e51; long prefill (`uniform-base`) 8354b6de…
   (T8192 1065 → 82 ms); caching-memory (`unchanged_plan`, automatic KV cache at model scale, peak memory)
   3cdb0b9d…f7a74cdb; competitors (CUDA-graph torch, SDPA backends, hand Triton, Liger, FA2/FA3, FlashInfer; TRT-LLM
   not installable) 6dfd0d95/e3f841c1/19a8a44e; tensor cores bf16/f16 mma + bf16 KV cache fedc07ca/c4b0f62b;
   GEMV in-flight + TMA capability 27258765/5dccfc07; open rules 8aee53e7/d3d9dd76; review fixes 3a8e654f;
   one-matrix tooling 9e27d09c.
7. **flash attention** (streamed single pass, `stream_plan`, GuardBox, finite_plan, split-KV) 55ba3e42.
8. **parallel-sprint** (worktrees): regressions 370637ed; f32 on tensor cores (split forms) dfdcb5e1; Hopper wgmma-ws +
   TMA 3092b270 ($1.79 H100); fp8 storage + calls-issue + tune-context + call_keep ad451a13; integration
   851de1ba/a31cd014 (`l2_keep_bytes`); sweeps 083d9b8d/80e6d240/d7cae90a/4c29387e (T8192 prefill 47.9 ms vs Liger
   53.2, 0.41× torch memory; round-store); sweep 2 0992dea6/43af51cf/683ae4d6/d950efb3 (stream-mma, kmajor-b,
   warm-zero, streamk-price/epi); review fixes 5fcb401b.
9. **Docs/rules**: CLAUDE.md f3a9e792, 34bc9705, e85f7c82; `docs/tailored-fixes.md`.

## Rented rounds (all value gates passed unless noted; every mapal instance destroyed)

| round (code) | cost | notes |
| --- | ---: | --- |
| 8bba2a50 | $5.60 | first cross-arch; property rule gaps found |
| 7a2b67bf | $4.88 | + gap analysis; a self-matching `pgrep` wait loop hung the agent 10 h (killed) |
| 60f0f8ce | $5.68 | chains on 4090/3090 win 1.16–1.38× |
| 19a8a44e (all competitors) | $11.55 | A100 replaced mid-run |
| 3a8e654f (final code then) | $9.19 | found B1-decode regression (fixed in 370637ed) |
| 5fcb401b (confirmation) | $14.12 | fp8 PTX 7.8 vs 8.1 bug on sm_89 (being fixed) |

## Final matrix (5fcb401b, every competitor, same precision; `benches/results/evidence/matrix.md`)

| geomean Mapal / fastest | RTX 5080 | RTX 3090 | A100 | RTX 4090 | H100 |
| --- | --- | --- | --- | --- | --- |
| all rows (120) | **0.979** | 1.009 | 1.242 | **0.948** | 1.318 |
| f32 / tf32 | 0.87 / 1.19 | 0.89 / 1.15 | 1.10 / 1.65 | 0.87 / 1.37 | 1.01 / 2.63 |
| bf16 weights / bf16 KV / bf16 storage | 1.66 / 1.04 / 0.98 | 1.50 / 1.00 / 1.13 | 2.16 / 1.12 / 1.23 | 1.38 / 0.83 / 0.90 | 2.19 / 1.10 / 1.65 |
| previous round (3a8e654f) | 1.076 | 1.127 | 1.501 | 1.106 | 1.566 |

## Open problems (being attacked by `confirmation-fixes`, then a review)

1. fp8 PTX ISA gating + ptxas preflight (bug, 4090). 2. Measured vs derived properties (H100 mma.sync 2.5× the
profile; bar_ns, latency, L2 keep). 3. split-tile on the A100 (registers instead of cp.async). 4. H100 16-bit tiles on
wgmma. 5. bf16-weight prefill 12–17× on A100/H100. 6. H100 TMA path: 120–270 µs per call outside the kernel; a
persistent scheduler. 7. Streamed attention priced wrong at long rows on HBM parts. 8. A100 MLP chains / fan-loops
regression. 9. mma-combine mixed; streamk-epi loses on 3090. Not yet scheduled: small-m GEMVs vs cuBLASLt (host launch
time), fp8 tiles' measurement, TensorRT-LLM (not installable on this stack), the ~1 µs runtime residual.

## Evidence index

`benches/results/evidence/local-rtx-5080/` (README links every step's file), `benches/results/evidence/cross-arch/README.md`
(every rented round), `benches/results/evidence/matrix.{md,json}` (rebuild: command at the end of matrix.md),
raw outputs under `.claude/raw-evidence/` (not in git; `archive_raw.sh` not run — it publishes a GitHub release).
