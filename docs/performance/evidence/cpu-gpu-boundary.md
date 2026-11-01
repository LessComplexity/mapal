# cpu-gpu-boundary: end check and remaining gaps

Raw data: [`benches/results/evidence/cpu-gpu-boundary/`](../../../benches/results/evidence/cpu-gpu-boundary/) (sweep
`7beca78`, 4 archs; smoke `66ebce8`, rtx-4090). ON/OFF is the program's own `iter ms`, shipped build vs its
`.off-<rung>` arm, ABBA x5 (sweep) or x3 (smoke); null = the shipped build run twice. The gap split follows
[why-slow.md](why-slow.md): one nsys trace per program (`prof-<arch>.tgz`); **kernel** excludes `mapal_gen_*`, **copy**
counts copies before the last kernel, **rest** = median iter − kernel − copy − memset (it mixes a median with a single
trace, so it can go negative). Previous state: [why-slow.md](why-slow.md) (flow sweep).

## End check (CLAUDE.md evidence pass, step 5)

Cells: intent rows ok/total from `evidence-<arch>.tsv.gz` · best program ON vs OFF ms · arms faster / slower beyond
their null (25 programs per arch unless noted).

| optimization (scope item) | intended | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | decision (the difference that explains it) |
|---|---|---|---|---|---|---|
| `uniform` (1b) | number captures become by-value `i64` kernel params; decode/causal/prefill `q` and rt_mlp2 `a`/`f` built on the device; per-trip H2D only the facts' (1 KB `x` row) | 8/8 · prefill 65.5 vs 309.3 · 3 / 0 | 8/8 · 52.6 vs 159.5 · 4 / 0 | 8/8 · 98.0 vs 425.4 · 3 / 0 | 8/8 · 80.2 vs 323.6 · 3 / 0 | **ON.** OFF builds the laws on the host and uploads them each trip (flow sweep rt_mlp2: 17 H2D, 285 MB per run). ON rt_mlp2 makes 1 H2D, and decode/causal 1 KB per trip. Decode stays at 0.89–1.09 because each trip still waits on the `o` copy-out (C5) |
| `nest-cell` (1a, 1d) | a map that folds over captured arrays becomes a one-thread-per-cell kernel walking the fold (PTX loop); K/V rows stop crossing | 6/6 · convmc 0.98 vs 351.9 · 3 / 0 | 6/6 · 0.73 vs 262.5 · 3 / 0 | 6/6 · 1.64 vs 361.6 · 3 / 0 | 6/6 · 2.23 vs 446.6 · 3 / 0 | **ON.** OFF runs causal/prefill `s`, `p`, `o` and the convmc fold on the host: in the flow sweep the GPU sat idle 414–572 ms per run and convmc had no kernel. With ON, causal D2H is 9.2 MB per run (the H100 moved 2148 MB in the flow sweep). **1d mismatch:** convmc was meant to become a `tile_plan` implicit GEMM. It runs as one thread per output instead (N2) |
| `retained-fold` (1a) | standalone row folds stay device sites in a function with retained sites (prefill `pmx`/`pden`) | check n/a: `dev.rowfold` = 0 on prefill. Arm diff ON / OFF: 8 / 5 launches, 4 / 3 nest cells, 1 / 0 flow unit · 65.5 vs 266.2 · 1 / 0 | n/a · 52.6 vs 143.0 · 1 / 0 | no retained site while `retain-resident` was OFF; active in the measured `.on-retain-resident` arm | n/a · 80.2 vs 294.2 · 1 / 0 | **ON.** OFF returns one placed fold and the flow unit to the host. **Finding:** the intent row counts `dev.rowfold`, which never sees prefill's folds (they are nest cells in the flow unit), so this rung is realized only through the arm diff |
| `tree-gate` (5) | under `flow`, a lone fold over a column-major operand keeps its rowfold kernel | rowsum_4096: the rowfold is launched (`dev.rowfold` 1) · 2.612 vs 2.766 · 1 / 0 (null 0.004) | emission unchanged | emission unchanged | emission unchanged | **ON.** Closes flow's rowsum_4096 at 1.056: 0.944 ≈ 1/1.056, with the 68.6 µs rowfold kept beside the GEMM. The intent row is still note-only |
| `preload` (3) | every kernel resolved at module load (init flag 8) | 25/25 · attn 0.071 vs 0.231 · 19 / 0 | 25/25 · 0.037 vs 0.067 · 7 / 0 | 25/25 · 0.066 vs 0.245 · 15 / 0 | 25/25 · 0.103 vs 0.207 · 16 / 0 | **ON.** OFF resolves each function at its first launch (`cuModuleGetFunction` plus a lazy load, inside the clock; in the flow sweep, C7 left the GPU idle 13–352 µs before first launches). This is also the O4 eager-load arm |
| `entry-arena` (3) | the arena is made in the entry block from the static plan, and loop-cone buffers go in it | 25/25 · ew 0.103 vs 0.429 · 3 / 0 | 25/25 · 0.059 vs 0.158 · 3 / 0 | 25/25 · 0.116 vs 0.527 · 4 / 0 | 25/25 · 0.148 vs 0.603 · 4 / 0 | **ON.** OFF calls the arena's `cudaMalloc` after `t0` (flow C8: alloc_ms 0.35–0.56), and loopmm makes 3 loop-cone allocations. ON makes one allocation, in the entry block (the `evidence.py` alloc prediction matches). Ceiling: slots get whole-function intervals, so two loops' buffers never share bytes |
| `async-h2d` (4) | no upload waits for the device | 5/5 (`h2d` = 0) · Σ25 281.5 vs 281.7 · 0 / 0 | 5/5 · 228.0 vs 228.0 · 0 / 0 | 5/5 · 436.1 vs 433.7 · 0 / 0 | 5/5 · 387.4 vs 372.9 · 0 / 0 | **OFF row on all 4 parts** (S63 rule: no win means a row). Realized but saves no time: the only in-clock uploads are the 1 KB row per trip (1024 × 0.3–1.0 µs), each ahead of a copy-out the host reads. In the smoke run, ew_4096 contract was 1.108 slower, unexplained (O10) |
| `stage` (4) | page-locked staging ring (init flag 16) | 5/5 · Σ 281.5 vs 280.5 · 0 / 0 | 5/5 · 228.0 vs 227.9 · 0 / 0 | 5/5 · 436.1 vs 429.3 · 0 / 0 | 5/5 · 387.4 vs 370.2 · 0 / 0 | Inert while `async-h2d` is OFF, so it has no row of its own. Same movement explanation as `async-h2d` |
| `chain-h2d` (4) | an upload read by only one chain crosses on that chain's stream | never emitted: `streams` ships OFF on all 4 parts, and no bench program changes even with streams forced | same | same | same | **Open:** not measured. It fires only in the randomized property test (42 chain uploads) |
| `retain-resident`, H100 (5) | a one-row K/V site per trip with the output resident: no full rebuild, no memset | – | – | the ON arm emits the one-row site `3881eb4b` (1 CTA, 8 LDG + 4 FFMA per hot-loop trip) in place of stream-K `8065b046` (228 CTAs) · ON vs OFF: causal 178.3 vs 193.0, prefill 92.99 vs 97.95, decode 67.03 vs 69.99 (the shipped build was the OFF arm here) · 1 / 0 (prefill and causal ranges disjoint) | – | **Row removed: ships ON.** Same sign in two sessions (flow sweep 0.678 / 0.787; now 0.922 / 0.949, decode 0.940 inside null 0.077). The OFF trace's per-run rebuild (23.7 ms) and memset (2.9–4.5 ms) are larger than each saving. The win shrank because nest-cell already removed OFF's 2148 MB D2H |
| in-place row map on device (1c) | no per-trip `o` D2H or `x` H2D in decode | not done: 1024 H2D + 1023 D2H per run on all 4 archs | same | same | same | **Open** (C5) |
| steady trip without host syncs (2) | a loop trip submitted with no host wait unless a host morphism observes it; CUDA graph as its own rung | only upload waits removed (`async-h2d`). causal/prefill still do 3 D2H + 1 H2D per trip, decode 1 + 1 | same | same | same | **Open** (C5) |

Intent mismatches on shipped programs: the same 21 rows as the flow sweep's list (face-fma on firew_rowsum, the 5090
pj 208 B spill, H100 attn prefetch cover, window full-unroll on conv/ewconvew, 4090 attn LDGSTS). None is on a new
rung (`report.md`, "Findings").

## GPU_RUNGS_OFF changes (`crates/backends/llvm/src/profile.rs`)

| part | rung | change | evidence |
|---|---|---|---|
| h100-pcie | `retain-resident` | row removed (ON) | ON arm vs shipped: causal 178.3 vs 193.0 ms, prefill 92.99 vs 97.95, decode 67.03 vs 69.99; `8065b046` rebuild + 1 MB memsets gone in favor of the one-row site `3881eb4b` |
| rtx-4090, rtx-5090, h100-pcie, rtx-3090 | `async-h2d` | new row (OFF) | ON/OFF median 1.001 / 1.003 / 1.001 / 0.999, 0 faster beyond null; 1 KB uploads per trip behind a host-read copy-out |

## What moved (Mapal / best other; flow sweep in brackets; below 1 = Mapal faster)

41 of 96 cells at or below 1.0 (flow sweep: 26 of 92). Full table: [`table.md`](../../../benches/results/evidence/cpu-gpu-boundary/table.md).

| workload | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | Mapal ms 4090 (best other) |
|---|---|---|---|---|---|
| convmc_8_64_56_3 | 17.77 (5178.39) | 12.36 (4041.26) | 20.33 (2213.74) | 27.57 (5022.66) | 0.9827 (0.0553 (torch)) |
| decode_1024_256 | 1.66 (1.96) | 1.60 (1.34) | 2.95 (3.58) | 1.69 (1.89) | 49.94 (29.999 (cache)) |
| attn_1024_64 | 1.97 (8.76) | 1.39 (7.6) | 2.00 (5.62) | 1.77 (4.15) | 0.0707 (0.0358 (torch-compile)) |
| loopmm_1024_8 | 1.31 (1.7) | 1.25 (2.03) | 1.12 (1.26) | 1.62 (1.72) | 0.4904 (0.3738 (torch-compile)) |
| pj_m16_n2048_k7168 | **0.86** (1.6) | 1.21 (2.27) | 1.35 (3.43) | 1.56 (4.26) | 0.0811 (0.0942 (torch-compile)) |
| causal_1024_256 | **0.62** (2.87) | 1.30 (2.74) | 1.41 (5.62) | **0.75** (2.89) | 118.85 (190.4353 (torch-compile)) |
| mlp2_2048 | 1.01 (1.0) | 1.12 (1.14) | 1.39 (1.38) | 1.03 (0.95) | 1.2895 (1.281 (lt2)) |
| indep_2048 | **0.99** (-) | 1.08 (-) | 1.35 (-) | 1.09 (-) | 0.6511 (0.6605 (cutlass)) |
| diamond_2048 | **0.98** (1.08) | 1.11 (1.15) | 1.30 (1.32) | 1.01 (0.93) | 0.9555 (0.98 (cutlass)) |
| prefill_1024_512_256 | **0.64** (5.57) | 1.25 (5.15) | 1.30 (8.62) | **0.81** (5.51) | 65.51 (102.9018 (torch-compile)) |
| smlp_4096_256_1024 | 1.09 (2.57) | 1.17 (1.84) | 1.19 (1.49) | 1.29 (1.69) | 0.1224 (0.1126 (torch-compile)) |
| rt_mlp2_2048 | 1.12 (4.08) | 1.12 (5.6) | 1.27 (2.27) | **1.00** (2.78) | 12.17 (10.8984 (torch-compile)) |
| fanout_2048 | **0.97** (1.09) | 1.08 (1.18) | 1.23 (1.26) | 1.02 (0.92) | 0.6415 (0.6615 (streams)) |
| mlp_2048 | **0.97** (1.12) | 1.09 (1.18) | 1.22 (1.25) | 1.04 (0.96) | 0.6351 (0.6543 (cutlass)) |
| twin_2048 | **0.95** (1.12) | 1.05 (1.15) | 1.21 (1.25) | 1.01 (1.03) | 0.6337 (0.6697 (torch)) |
| rowsum_4096 | 1.05 (1.05) | 1.10 (1.12) | **0.97** (0.98) | **0.95** (0.86) | 2.6121 (2.4956 (gemv)) |
| softmax_rows_4096 | 1.03 (1.03) | 1.09 (1.18) | **0.97** (1.08) | **0.96** (0.92) | 2.6301 (2.5446 (torch-compile)) |
| epi_4096 | **1.00** (0.89) | 1.06 (1.07) | **0.97** (0.95) | **0.88** (0.79) | 2.5192 (2.5262 (lt)) |
| ew_4096 | **0.46** (2.59) | **0.45** (5.33) | 1.01 (4.75) | **0.61** (2.27) | 0.1029 (0.2243 (fused)) |
| loop_4096 | **0.95** (0.9) | **0.97** (0.96) | **0.97** (0.94) | **0.96** (0.94) | 19.74 (20.8425 (cutlass)) |
| firew_rowsum_2048x2048_16 | **0.66** (1.31) | **0.55** (1.56) | **0.57** (1.34) | **0.72** (1.07) | 0.0454 (0.0686 (torch-compile)) |
| conv_2048_5 | **0.63** (0.8) | **0.45** (0.71) | **0.50** (0.56) | **0.68** (0.81) | 0.0559 (0.0891 (torch-compile)) |
| fir_4194304_16 | **0.61** (0.85) | **0.51** (0.84) | **0.56** (0.63) | **0.63** (0.74) | 0.0396 (0.0645 (torch-compile)) |
| ewconvew_2048_5 | **0.53** (0.79) | **0.51** (0.82) | **0.51** (0.6) | **0.53** (0.62) | 0.0674 (0.126 (torch-compile)) |

## Top 10 gaps (each workload at its worst arch)

| # | workload | worst arch (gap ms) | other archs (gap ms) | where the time goes | causes | ms recoverable (basis) |
|---|---|---|---|---|---|---|
| 1 | causal_1024_256 | H100 (56.3) | 5090 23.2 (4090, 3090 win) | Kernels 133.6 ms: the `s`/`o` nest cells take 1024 × (67.0 + 38.2) µs on 1 and 4 CTAs. The K/V rebuild takes 2048 × 11.6 µs, plus 2048 × 1 MB memset. Rest 49.0 ms: each trip does 3 D2H (`s`, `p`, `o`: 9 KB) and 1 H2D (`x`) around the host's `mx`/`den` | N1, C5, C2 | C2 measured 14.7 (now ON). N1 ≤ 107.7 kernel. C5 ≤ 49.0 rest |
| 2 | decode_1024_256 | H100 (46.3) | 3090 20.3, 4090 19.9, 5090 12.4 | Kernels 38.3: rebuild 23.75, memset 4.5. Rest 24.1: each trip does an H2D of the `x` row and a D2H of `o`, because the host writes row t | C5, C2 | C2 measured 3.0 (inside null). C5 ≤ rest: 24.1 / 22.8 / 26.9 / 14.2 |
| 3 | prefill_1024_512_256 | H100 (22.4) | 5090 10.4 (4090, 3090 win) | Nest cells 512 × (66.2 + 43.1) µs, rebuild 1026 × 11.6 µs, rest 23.2 | N1, C5, C2 | C2 measured 5.0 |
| 4 | rt_mlp2_2048 | H100 (4.76) | 4090 1.27, 5090 1.13 (3090 ties) | H100: 32 GEMMs at 630–647 µs (64-accumulator tile, 122 regs). 4090/5090: one pageable 16.8 MB H2D inside the clock (1.48 / 0.97 ms) | C6a, N4 | H100 ≤ 4.76. 4090/5090 ≤ copy (2.40) |
| 5 | convmc_8_64_56_3 | 3090 (2.14) | H100 1.56, 4090 0.93, 5090 0.68 | One nest-cell kernel: 6272 CTAs × 256 threads, 40 regs, 0 smem, 576 taps walked per output. 0.73–2.2 ms vs cuDNN 0.055–0.081 ms (TF32, O6) | N2 | ≤ 5.31 (Σ gaps) |
| 6 | mlp2_2048 | H100 (0.707) | 5090 0.124, 3090 0.110 (4090 inside null) | 3 GEMMs at 616–622 µs vs cuBLASLt 508 | C6a, C6b | 0.83 |
| 7 | loopmm_1024_8 | 3090 (0.558) | 4090 0.117, 5090 0.090, H100 0.073 | 3090: 8 × 173.7 µs (16×16 CTAs, 128 threads, 127 regs) plus 8 memsets. torch-compile: 0.897 total | C6c | 0.84 |
| 8 | diamond_2048 | H100 (0.440) | 5090 0.091, 3090 0.033 | 3 GEMMs at 622–644 µs | C6a, C6b | 0.56 |
| 9 | indep_2048 | H100 (0.331) | 3090 0.134, 5090 0.044 | 2 GEMMs at 615–641 µs | C6a | 0.51 |
| 10 | fanout_2048 | H100 (0.230) | 5090 0.044, 3090 0.027 | 2 GEMMs at 621–628 µs | C6a, C6b | 0.30 |

## Every losing workload × arch (sorted by gap)

| workload | arch | Mapal ms | best ms (leg) | gap ms | GPU kernel ms | copy ms | memset ms | rest ms | H2D/D2H in clock | top Mapal kernels (n x median us) | causes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| causal_1024_256 | h100-pcie | 193 | 136.7 (torch-compile) | 56.33 | 133.6 | 7.52 | 2.88 | 49.01 | 1026/3071 | eea92378: 1024x66.9, 586f1a0c: 1024x42.1 | N1, C5, C2 |
| decode_1024_256 | h100-pcie | 69.99 | 23.69 (cache) | 46.3 | 38.25 | 3.14 | 4.52 | 24.08 | 1024/1023 | aa4e3a84: 2048x11.6, a156cb5e: 1024x7.9 | C5, C2 |
| causal_1024_256 | rtx-5090 | 101.5 | 78.36 (torch-compile) | 23.17 | 67.9 | 1.6 | 0 | 32.03 | 1026/3071 | 46817d56: 1024x33.4, ce77349a: 1024x25.7 | N1, C5 |
| prefill_1024_512_256 | h100-pcie | 97.95 | 75.59 (torch-compile) | 22.36 | 69.47 | 3.79 | 1.45 | 23.23 | 515/1537 | 095314a1: 512x66.2, f5d7e482: 512x43.1 | N1, C5, C2 |
| decode_1024_256 | rtx-3090 | 49.64 | 29.39 (cache) | 20.25 | 23.14 | 1.83 | 1.92 | 22.75 | 1024/1023 | 0648dc17: 2046x6.4, 9f8b9499: 1024x4.4 | C5 |
| decode_1024_256 | rtx-4090 | 49.94 | 30 (cache) | 19.94 | 19.97 | 1.65 | 1.43 | 26.89 | 1024/1023 | 184d5db4: 2046x4.6, e34e679c: 1024x5.7 | C5 |
| decode_1024_256 | rtx-5090 | 33.13 | 20.72 (cache) | 12.41 | 16.89 | 0.691 | 1.35 | 14.2 | 1024/1023 | 8c059815: 2046x4.1, 20b8101f: 1024x4.8 | C5 |
| prefill_1024_512_256 | rtx-5090 | 52.56 | 42.15 (torch-compile) | 10.41 | 35.62 | 0.852 | 0 | 16.09 | 515/1537 | 9bfd34b5: 512x33.4, 9b155244: 512x25.9 | N1, C5 |
| rt_mlp2_2048 | h100-pcie | 22.22 | 17.46 (torch-compile) | 4.755 | 20.43 | 1.49 | 0 | 0.2952 | 1/0 | c31419c7: 8x647.0, 8d1b6cf1: 8x630.7 | C6a |
| convmc_8_64_56_3 | rtx-3090 | 2.225 | 0.0807 (torch) | 2.144 | 2.2 | 0.000608 | 0 | 0.02379 | 1/0 | 09f9b4e5: 1x2200.5 | N2 |
| convmc_8_64_56_3 | h100-pcie | 1.641 | 0.0807 (torch) | 1.56 | 1.597 | 0.000928 | 0 | 0.0427 | 1/0 | d5d8e1a3: 1x1597.2 | N2 |
| rt_mlp2_2048 | rtx-4090 | 12.17 | 10.9 (torch-compile) | 1.272 | 10.45 | 1.48 | 0 | 0.2361 | 1/0 | a2524dfa: 8x328.2, 29c1dd2d: 8x323.0 | N4 |
| rt_mlp2_2048 | rtx-5090 | 10.66 | 9.527 (torch-compile) | 1.133 | 9.573 | 0.966 | 0.0471 | 0.07356 | 1/0 | 7d9568ae: 8x314.9, 041e4d09: 8x297.5 | N4 |
| convmc_8_64_56_3 | rtx-4090 | 0.9827 | 0.0553 (torch) | 0.9274 | 0.958 | 0.000448 | 0 | 0.02423 | 1/0 | 16ec8f32: 1x958.0 | N2 |
| mlp2_2048 | h100-pcie | 2.507 | 1.8 (lt2) | 0.7073 | 2.476 | 0 | 0 | 0.03066 | 0/0 | f2c69fb4: 2x616.6, 8d1b6cf1: 1x621.9 | C6a |
| convmc_8_64_56_3 | rtx-5090 | 0.7344 | 0.0594 (torch) | 0.675 | 0.7252 | 0.000352 | 0 | 0.008892 | 1/0 | 97eaeec3: 1x725.2 | N2 |
| loopmm_1024_8 | rtx-3090 | 1.455 | 0.897 (torch-compile) | 0.5584 | 1.39 | 0 | 0.0462 | 0.01952 | 0/0 | fee9769e: 8x173.5 | C6c |
| diamond_2048 | h100-pcie | 1.884 | 1.444 (streams) | 0.4397 | 1.889 | 0 | 0 | -0.004719 | 0/0 | f2c69fb4: 2x622.4, 0c928e3f: 1x644.0 | C6a |
| indep_2048 | h100-pcie | 1.269 | 0.9377 (streams) | 0.3312 | 1.256 | 0 | 0 | 0.01295 | 0/0 | f006c502: 1x640.6, f2c69fb4: 1x615.3 | C6a |
| fanout_2048 | h100-pcie | 1.25 | 1.02 (streams) | 0.2303 | 1.249 | 0 | 0 | 0.001592 | 0/0 | 7e9ac22e: 1x628.2, f2c69fb4: 1x620.7 | C6a |
| mlp_2048 | h100-pcie | 1.248 | 1.021 (lt) | 0.2269 | 1.24 | 0 | 0 | 0.007496 | 0/0 | 2e05664d: 1x620.5, f2c69fb4: 1x619.8 | C6a |
| twin_2048 | h100-pcie | 1.241 | 1.022 (torch-compile) | 0.2191 | 1.227 | 0 | 0 | 0.01409 | 0/0 | f2c69fb4: 2x613.7 | C6a |
| rowsum_4096 | rtx-5090 | 2.309 | 2.107 (torch-compile) | 0.202 | 2.272 | 0 | 0.0301 | 0.007465 | 0/0 | a437b31e: 1x2234.3, 13e9e014: 1x37.5 | C6b |
| softmax_rows_4096 | rtx-5090 | 2.325 | 2.126 (torch-compile) | 0.1997 | 2.287 | 0 | 0.0287 | 0.009548 | 0/0 | a437b31e: 1x2225.5, f707d404: 1x61.8 | C6b |
| indep_2048 | rtx-3090 | 1.7 | 1.567 (torch-compile) | 0.1338 | 1.701 | 0 | 0 | -0.000601 | 0/0 | 9f761422: 1x877.6, c50c7414: 1x823.5 | O11 |
| epi_4096 | rtx-5090 | 2.247 | 2.119 (torch-compile) | 0.1285 | 2.232 | 0 | 0 | 0.01554 | 0/0 | 1da7ca3b: 1x2231.7 | C6b |
| mlp2_2048 | rtx-5090 | 1.205 | 1.081 (streams) | 0.1244 | 1.156 | 0 | 0.0118 | 0.03749 | 0/0 | a86bc84f: 2x269.6, 7d9568ae: 1x310.6 | C6b |
| loopmm_1024_8 | rtx-4090 | 0.4904 | 0.3738 (torch-compile) | 0.1166 | 0.4727 | 0 | 0 | 0.01769 | 0/0 | 02536326: 8x58.7 | C6c |
| rowsum_4096 | rtx-4090 | 2.612 | 2.496 (gemv) | 0.1165 | 2.591 | 0 | 0 | 0.02156 | 0/0 | b6d53a54: 1x2521.9, 9e8c05c9: 1x68.6 | C6 |
| mlp2_2048 | rtx-3090 | 3.29 | 3.179 (lt2) | 0.1102 | 3.263 | 0 | 0 | 0.02664 | 0/0 | c50c7414: 2x825.6, 248d9158: 1x808.2 | O11 |
| diamond_2048 | rtx-5090 | 0.8902 | 0.7988 (torch-compile) | 0.0914 | 0.8524 | 0 | 0.0121 | 0.02578 | 0/0 | a86bc84f: 2x277.4, aec4ae21: 1x297.6 | C6b |
| loopmm_1024_8 | rtx-5090 | 0.4537 | 0.3635 (torch-compile) | 0.0902 | 0.4493 | 0 | 0 | 0.004385 | 0/0 | 07e662e7: 8x56.2 | C6c |
| softmax_rows_4096 | rtx-4090 | 2.63 | 2.545 (torch-compile) | 0.0855 | 2.623 | 0 | 0 | 0.00685 | 0/0 | db04347e: 1x2506.9, 04060495: 1x116.4 | C6 |
| loopmm_1024_8 | h100-pcie | 0.6872 | 0.6145 (torch-compile) | 0.0727 | 0.6255 | 0 | 0.0186 | 0.04307 | 0/0 | 9b98b88a: 8x77.5 | C6c |
| smlp_4096_256_1024 | rtx-3090 | 0.3115 | 0.2406 (torch-compile) | 0.0709 | 0.2885 | 0 | 0.00634 | 0.01671 | 0/0 | d9a22392: 1x170.6, 348d40c1: 1x117.8 | C6c |
| mlp_2048 | rtx-3090 | 1.699 | 1.631 (lt) | 0.0679 | 1.69 | 0 | 0 | 0.009072 | 0/0 | 78fdd7ad: 1x859.9, c50c7414: 1x830.1 | O11 |
| mlp_2048 | rtx-5090 | 0.575 | 0.5264 (lt) | 0.0486 | 0.5795 | 0 | 0.00605 | -0.01057 | 0/0 | 414beaef: 1x295.4, a86bc84f: 1x284.1 | C6b |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1278 | 0.0819 (torch-compile) | 0.0459 | 0.1178 | 0 | 0 | 0.009975 | 0/0 | b5cd6bd4: 1x115.5, 0f1b1523: 1x2.3 | C10 |
| attn_1024_64 | rtx-3090 | 0.1033 | 0.0584 (fused) | 0.0449 | 0.08874 | 0 | 0.00176 | 0.0128 | 0/0 | 2a067910: 1x55.6, 92e252b2: 1x20.2 | C9, N3 |
| indep_2048 | rtx-5090 | 0.5763 | 0.5325 (lt) | 0.0438 | 0.5816 | 0 | 0.00579 | -0.01109 | 0/0 | aa8ff818: 1x310.4, a86bc84f: 1x271.2 | C6b |
| fanout_2048 | rtx-5090 | 0.5721 | 0.5286 (streams) | 0.0435 | 0.5795 | 0 | 0.00598 | -0.01338 | 0/0 | 79563630: 1x299.6, a86bc84f: 1x279.8 | C6b |
| attn_1024_64 | rtx-4090 | 0.0707 | 0.0358 (torch-compile) | 0.0349 | 0.04977 | 0 | 0.00269 | 0.01824 | 0/0 | 868c20ca: 1x25.5, 10ab6f60: 1x17.8 | C9, N3 |
| diamond_2048 | rtx-3090 | 2.534 | 2.501 (streams) | 0.0334 | 2.526 | 0 | 0 | 0.008251 | 0/0 | c50c7414: 2x829.5, c874c2df: 1x866.7 | O11 |
| attn_1024_64 | h100-pcie | 0.0665 | 0.0333 (torch-compile) | 0.0332 | 0.04371 | 0 | 0.00125 | 0.02154 | 0/0 | 5fb73ea4: 1x19.0, a1b8421a: 1x17.6 | C9 |
| smlp_4096_256_1024 | h100-pcie | 0.185 | 0.1554 (torch-compile) | 0.0296 | 0.1646 | 0 | 0.00237 | 0.01805 | 0/0 | 256dece7: 1x86.7, 18bdc1ae: 1x77.9 | C6c |
| fanout_2048 | rtx-3090 | 1.684 | 1.657 (streams) | 0.0269 | 1.669 | 0 | 0 | 0.01453 | 0/0 | 8f30e00e: 1x846.6, c50c7414: 1x822.6 | O11 |
| twin_2048 | rtx-5090 | 0.555 | 0.5294 (torch-compile) | 0.0256 | 0.554 | 0 | 0.0121 | -0.01105 | 0/0 | a86bc84f: 2x277.0 | C6b |
| pj_m16_n2048_k7168 | h100-pcie | 0.0939 | 0.0695 (torch-compile) | 0.0244 | 0.07824 | 0 | 0 | 0.01566 | 0/0 | 1364f56e: 1x76.6, 0e98010d: 1x1.7 | C10 |
| twin_2048 | rtx-3090 | 1.657 | 1.64 (torch-compile) | 0.0163 | 1.656 | 0 | 0 | 0.000973 | 0/0 | c50c7414: 2x827.9 | O11 |
| smlp_4096_256_1024 | rtx-5090 | 0.1034 | 0.0881 (torch-compile) | 0.0153 | 0.09293 | 0 | 0 | 0.01047 | 0/0 | d020cefb: 1x55.5, f39d46f7: 1x37.4 | C6c |
| pj_m16_n2048_k7168 | rtx-5090 | 0.067 | 0.0553 (torch) | 0.0117 | 0.06189 | 0 | 0 | 0.005111 | 0/0 | 69cbcc55: 1x61.9 | C10 |
| attn_1024_64 | rtx-5090 | 0.0371 | 0.0266 (torch-compile) | 0.0105 | 0.02822 | 0 | 0 | 0.008876 | 0/0 | b16cc161: 1x14.5, 7897f338: 1x8.7 | C9 |
| smlp_4096_256_1024 | rtx-4090 | 0.1224 | 0.1126 (torch-compile) | 0.0098 | 0.1079 | 0 | 0 | 0.01446 | 0/0 | 510aec4a: 1x60.4, 439d6096: 1x47.6 | C6c |
| mlp2_2048 | rtx-4090 | 1.29 | 1.281 (lt2) | 0.0085 | 1.285 | 0 | 0 | 0.004005 | 0/0 | 17ac8177: 2x310.5, 51089bac: 1x344.9 | inside null |
| ew_4096 | h100-pcie | 0.1155 | 0.1148 (torch-compile) | 0.0007 | 0.07226 | 0 | 0 | 0.04324 | 0/0 | 9895b0ef: 1x72.3 | inside null |

## Causes

| id | rows | evidence | missing fact / realization |
|---|---|---|---|
| C2 | causal, prefill, decode on H100 | The OFF trace rebuilds K/V every trip (`8065b046`, 228 CTAs, 11.6 µs) and memsets 1 MB each time | **Fixed in this judge** (row removed). Measured ON arms: 0.922 / 0.949 / 0.940. The shipped build has not been re-timed |
| C5 | decode ×4, causal/prefill on 5090 and H100 | decode: 1 H2D (`x`) + 1 D2H (`o`) per trip, because the host writes row t (range map). causal/prefill: 3 D2H (`s`, `p`, `o`: 9 KB) + 1 H2D per trip, because `mx`/`den` are whole folds the host computes. Rest is 14–49 ms. The cache leg runs about 12 async launches per trip | A device cell for the in-place range map (`map_law` piecewise + `Outside`, item 1c). A device site for a whole fold over a device array, read as a `uniform` by the next cell. Then the trip needs no host sync (item 2) |
| N1 | causal, prefill | The `o` nest cell has 256 cells, each folding 1024 steps; `s` has 1024 cells folding 256 steps. They run on 1 and 4 CTAs of 256 threads, one thread per cell. Per trip that is 59.6 µs on 4090, 56.5 on 5090 and 105 on H100. torch-compile's step uses gemv over 16 CTAs × 512 threads | When cells < resident threads (`TargetProfile`), split a nest cell's fold across lanes as a tree fold, the way `flow` already does for row folds. The graph proves the fold reassociable on the default face (ADR-0040) |
| N2 | convmc ×4 | One thread per output walks 576 taps, with no smem reuse of `img` or the weights: 958 µs on 4090, about 0.6 TFLOP/s | The mixed-radix fact (constant div/mod split of `k` and `t`) is still missing, so the fold is not a `tile_plan` implicit-GEMM site (1d) |
| N3 | attn on 4090 / 3090, loopmm on 3090 | 4090 attn ships the atomic split `7e5df615` (16×16 CTAs, 69 regs, 38.9 KB smem, 25.5 µs) plus a memset. It measures 0.0707 ms vs 0.0493 with `splitk-atomic` off and 0.0446 with all splits off. 3090: attn 1.588, loopmm 1.173 beyond null. The kernel + memset difference is about 9 of the 22 µs | Open: the iteration difference is larger than the kernel difference, so no row yet (`splitk-atomic`, Under/Contract) |
| N4 | rt_mlp2 on 4090 / 5090 | One host-built 2048² operand (16.8 MB, pageable) is uploaded inside the clock: 1.48 ms on 4090. Every per-trip law is on the device (`uniform`) | That operand's law is not a device generation (`gen_law`) |
| C6a | H100: mlp, mlp2, diamond, fanout, twin, indep, rt_mlp2 | Unchanged since the flow sweep: a 64-accumulator, 1-stage tile at 613–647 µs vs cuBLASLt 508 µs | The 128-accumulator cp.async tile on sm_90a (deferred) |
| C6b | 5090: rowsum, softmax, epi, mlp-family | The 4096 GEMM takes 2225–2234 µs (torch-compile ≈ 2.1 ms total). The 2048 epilogue GEMMs take 295–311 µs vs stream-K 270–284 | Stream-K is not combined with the epilogue site. Tile size |
| C6c | loopmm ×4, smlp | 3090 takes 173.7 µs per 1024³ GEMM (split-K atomic + memset); 4090 58.7, 5090 56.2, H100 77.5 | The tile choice at K = 1024 |
| C9 | attn ×4 | The rowfold `e594df4e` runs 4×32 CTAs × 32 threads, 117 regs, 88 KB smem: 17.8 µs on 4090 | A tree row fold with more lanes per row |
| C10 | pj on 3090 / H100 / 5090 | 3090 takes 115.5 µs (498 GB/s); the 5090 kernel spills 208 B | Too few warps per SM in the GEMV tile (finding unchanged) |
| C6 | rowsum, softmax, epi on 4090 | 1.03–1.05: 4096 GEMM at 2507–2522 µs vs the gemv / torch-compile legs | Tile at 4096 (inside 5 %) |
| O11 | mlp-family on 3090 | 1.01–1.09; up from 0.92–1.03 in the flow sweep, inside null (epi rose 0.79 → 0.88, still a win) | Untraced |

Closed since why-slow.md: C1 (nest-cell + uniform: prefill 5.57 → 0.64 on 4090), C3 (convmc 5178 → 17.8×), C4 (rt_mlp2
4.08 → 1.12), C7 (preload), C8 (entry-arena: ew 2.59 → 0.46), C12 and rowsum on 4090 (tree-gate). C11 (5090 rest of
0.39–0.43 ms) did not reproduce: rest is 0.007–0.016 ms on the same box's trace.

## Fixes ranked by ms recoverable

| rank | fix | causes | rows | ms recoverable (sum over archs) | basis |
|---|---|---|---|---:|---|
| 1 | Device site for whole folds over device arrays + device cell for the in-place range map, so no host sync per trip | C5 | decode ×4, causal/prefill 5090 + H100 | ≤ 181.6 | Σ min(gap, rest) |
| 2 | Split a nest cell's fold across lanes when cells < resident threads | N1 | causal, prefill on 5090 / H100 | ≤ 112.3 (overlaps 1) | Σ gaps; nest-cell kernel time is 56–108 ms per row |
| 3 | H100 `retain-resident` ON | C2 | causal, prefill, decode on H100 | 22.7 (done) | ON arms: 14.7 + 5.0 + 3.0 |
| 4 | Mixed-radix index fact, so convmc becomes an implicit-GEMM `tile_plan` site | N2 | convmc ×4 | ≤ 5.3 | Σ gaps |
| 5 | 128-accumulator cp.async tile on H100 | C6a | 7 H100 rows | ≤ 6.9 | Σ gaps |
| 6 | Generate rt_mlp2's host-built operand on the device | N4 | rt_mlp2 4090 / 5090 | ≤ 2.4 | copy time, capped at the gap |
| 7 | Tiles at K = 1024 | C6c | loopmm ×4, smlp | 0.84 + 0.13 | Σ gaps |
| 8 | Stream-K with the epilogue; 4096 tile on 5090 | C6b | 5090 mlp-family, rowsum, softmax, epi | 0.85 | Σ gaps |
| 9 | `splitk-atomic` class on attn / loopmm (N3, after it is explained) | N3 | attn 4090 / 3090, loopmm 3090 | ≤ 0.1 | ON vs off-split arms |
| 10 | Tree row fold for attn; GEMV warps per SM | C9, C10 | attn ×4, pj ×3 | ≤ 0.21 | Σ gaps |

## Open

| id | what | evidence that closes it |
|---|---|---|
| O2 | **Closed:** H100 `retain-resident` on decode measured 0.940 (inside null, not slower); the row is removed on causal/prefill | – |
| O3 | Library kernel SASS (C6a/C6b cite nsys geometry only) | cuobjdump of the captured cuBLASLt/CUTLASS images |
| O4 | **Closed:** the eager-load arm is `.off-preload` (attn 0.071 vs 0.231 ms on 4090) | – |
| O6 | A same-face convmc ratio | torch leg with `allow_tf32 = False` |
| O7 | DRAM GB/s and FP32 % are derived, not counted | ncu (vast refuses counters) |
| O8 | No lib-leg trace for causal, prefill, convmc | Add lib traces |
| O9 | H100 one-row site cost: the ON arm saves 14.7 ms on causal against 26.6 ms of rebuild + memset removed | nsys of `.on-retain-resident` on H100 |
| O10 | `async-h2d`/`stage` on ew_4096: 1.03–1.16 on 7 of 8 sweep arms and 1.108 in the smoke, where only init flag 16 differs | Moot while `async-h2d` ships OFF; an API trace of the ring allocation |
| O11 | 3090 rows up since the flow sweep (diamond 0.93 → 1.01, fanout 0.92 → 1.02, epi 0.79 → 0.88) | Same-box A/B of flow-latest vs this build |
| O12 | Intent rows: `tree-gate` is note-only; the `retained-fold` counter is blind on prefill | Count units/folds per arm in `intent.py` |
| O13 | `chain-h2d` not measured (streams OFF everywhere) | A program with two independent chains reading uploads, with streams ON |
| O14 | The sweep took 5 ABBA reps (9 were planned), and its null parse was loose | Re-time the decided rows at 9 reps |
| O15 | The smoke's 110 exact/contract-face mismatches (face-fma 69, face-separate 35, `arena:on` on `.fs` 6) have not been compared to flow-latest | `rejudge.py` on both builds |
