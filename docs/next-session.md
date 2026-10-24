# Continue from the fusion-natural-forms sweep

Branch `fusion-natural-forms` (generic-fusion + natural-forms, integrated at `332d20f`, swept on four parts). Not pushed.
Results: [evidence README, sweep](performance/evidence/README.md#fusion-natural-forms-sweep-the-four-parts-on-hardware) ·
raw: [`benches/results/evidence/fusion-natural-forms/`](../benches/results/evidence/fusion-natural-forms/) ·
log: [session](sessions/2026-10-24-fusion-natural-forms.md) · tables: `benches/gpu/fusion_natural_report.py`,
`benches/gpu/evidence_report.py`.

## State

| item | state |
| --- | --- |
| `fuse` | **ships OFF on all four parts** (`GPU_RUNGS_OFF` × 4, `a5a5368`): ON slower outside null on 19 of 24 program × part cells on the main boxes and 22 of 24 on the second hosts, faster on none. The unit kernel (one thread per row) is 1.79–4.82× the four kernels it replaces. Code, tests (`generic_fusion.rs`, `fuse` forced on) and the OFF arm stay |
| `window` | ON on all four: conv / ew→conv→ew 9.7–48× their OFF arm (host conv) |
| `range-map` | ON on all four: decode 2.3–4.4× its OFF arm (host map over 262144 cells per trip) |
| `lane-stride` 2-D (`q·Kᵀ`) | ON on all four: attention 106–693× its OFF arm |
| end check | the build with the fuse rows = the measured build for 25 programs × 4 parts (fused programs = their off-fuse arm) |
| tests | workspace release 1254 passed, 0 failed, 3 ignored (93 binaries) at `a5a5368` |
| harness | chain_bench: attention (cuBLAS eager / warp-per-row softmax) and smlp (eager, cuBLASLt RELU_BIAS, CUTLASS) legs; torch_bench: attention, smlp, FIR→ew→rowsum, ew→conv→ew; `B()` stops on an undersized name reuse |
| rentals | none live. Protected, never touch: 57065574, 57100404, 57126023, 57133432, 57285249, 57415164, 57450102 |

## Remaining gaps, ranked by measured time lost (the build as it ships now; main boxes, 3090 = S65's own host)

| # | workload | ms lost, Σ 4 parts (Mapal / best other) | explained by | next |
| --- | --- | --- | --- | --- |
| 1 | decode_1024_256 | 118.3 (1.99 / 1.27 / 3.69 / 1.77× the cache or no-cache harness) | 4090: 4096 launches of one-block kernels (`gx` 1 or 4, 256 threads, 7–17 µs each; the one-row K/V site 2046 × 10.8 µs) + per step 2 h2d / 1 d2h synchronous copies (8.9 + 9.5 ms); the harness issues 6 async launches per step, no copies | the per-step sites in one launch (fuse-tile / a step kernel), the per-step copies off the critical path |
| 2 | rt_mlp2_1024 | 51.5 (13.55 / 10.75 / 3.51 / 3.22× torch compile) | traced kernel / copy / host ms 2.71 / 11.99 / 8.34 (4090), 2.26 / 6.43 / 8.37 (5090), 4.04 / 2.80 / 2.31 (H100): 68 MB of host-made request inputs cross per run on EPYC hosts; 5.49× on the Ryzen 9 4090 host | make the request inputs on the device (S65 finding, still open) |
| 3 | loop_2048, loop_4096 | 2.7, 1.3 (1.14–1.26×, 0.94–1.03× cuBLASLt / CUTLASS) | the 2048³ FMA tile per step against cuBLASLt RELU epilogue | tile class at 2048 (unchanged from S65) |
| 4 | ew_2048, ew_4096 | 1.7, 1.6 (6.7–15.3×, 2.3–5.0× the fused kernel) | traced host 0.35–0.66 ms of a 0.39–0.95 ms run (one launch, kernel 0.04–0.15 ms) on the EPYC main boxes; the whole run 0.22 ms on the Ryzen 9 4090 / 5090 hosts | the fixed per-run host cost |
| 5 | softmax_rows_4096 / 2048 (fuse OFF) | 1.7, 0.9 (1.07–1.18×, 1.37–1.60× compile) | two one-thread-per-row rowfold kernels (0.08–0.22 ms each at 4096 rows) | a block-per-row reduction (rowfold and the fuse unit alike) |
| 6 | loopmm_1024_8 | 1.6 (1.30–2.00× compile) | as S65 (3090 `splitk-atomic` zero-fill per trip) | as S65 |
| 7 | attention (4 sizes, fuse OFF) | 1.1 + 1.1 + 0.8 + 0.7 (2.85–8.66× compile) | kernels 0.21–0.54 ms at 2048 × 128: strided `q·Kᵀ` tile (separate mul/add, 5090 / 3090 spills), two thread-per-row rowfolds, two cell kernels, `p·V` tile; compile and the warp-per-row harness are 0.03–0.26 ms | block-per-row softmax unit; the strided tile's FMA face and registers |
| 8 | rm_m128 (2 sizes), smlp, firew_rowsum, rowsum | 0.1–0.6 each | small-site host overhead and the MLP epilogue (as S64 / S65) | as before |

Measured faster than the best other on every part: conv_2048_5 (0.11–0.87×), ewconvew_2048_5 (0.13–0.82×),
fir_4194304_16 (0.63–0.92×).

## Findings, open

| finding | evidence | next |
| --- | --- | --- |
| fuse unit geometry: one thread per row, members serial, `ex` twice | unit kernel 1.79–4.82× the replaced kernels, all parts; d = 128 attention 3.63–4.82× vs d = 64 2.11–2.84×: the unit kernel carries 6 `div.rn.f32` at d = 128 (÷11.3) against 2 and 4 multiplies by 0.125 at d = 64 (÷8.0) | block per row, warp reductions, `ex` once into registers / shared memory; re-measure with the existing OFF arm, then remove the rows |
| `window`: LLVM unrolls the 25-tap walk only partially; ptxas completes it | remarks `full_unroll=0` on every part (intent mismatch, 2 per part) | the design's two counted loops |
| strided 2-D tiles: separate mul/add on the default face, 5090 / 3090 spills, H100 occupancy 1 < 2, prefetch cover | box intent check (mismatch table) | as the static finding 4 |
| 5090 JIT ≠ ptxas (registers 225 vs 227, local 152 vs 208 B) | `exec:on` 16 mismatches on both 5090 boxes (driver 580 / 610) | as S64 |
| boxes: 3090 main capped at 220 W (cuBLAS 4096³ 10.06 vs 6.04 ms at 350 W), a 3090 at ~27% speed, a 5090 that lost clock mid-run | `confirm-throttled/`, boxes table | rent by host id where a known-good host is free (`run_offer`-style) |
| `fuse-tile`, footprint rect, `L1101` (integration findings 3, 5, 6) | unchanged | as the integration report |

## Resume commands

```sh
cd <worktree on fusion-natural-forms>
git status --short && git log --oneline -15
python3 benches/gpu/fusion_natural_report.py benches/results/evidence/fusion-natural-forms
python3 benches/gpu/evidence_report.py benches/results/evidence/fusion-natural-forms
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
vastai show instances   # expect only the protected ids
# a re-take of one part (same bundle recipe: integrated_bundle.sh with SUITE / ARMS as in the session log):
PROFILE=1 E2E=0 TIMING=0 PROBE_REPS=9 RUN_MIN=75 benches/gpu/vast_part.sh rtx-4090 sm_89 "$B" "$R/rtx-4090" 30 RTX_4090
```
