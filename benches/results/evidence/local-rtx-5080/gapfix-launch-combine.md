# gapfix-launch-combine — C1 host path, H harness boundary, C5 split price (2026-11-25)

Branch `gapfix/launch-combine` off `fac7691b` (the `gaps-disasm.md` merge). Causes C1, H and C5 of
[`gaps-disasm.md`](gaps-disasm.md); C9 not reached (open, below). Raw output (never in git):
`.claude/raw-evidence/local-rtx-5080/gapfix-launch-combine/` — `rtx-4090/` (the rented box: `results.txt`, `runs.tgz`,
`rent.log`), `local-5080-abba-v1.txt`, `local-5080-abba-attn2048-fix.txt`, `torch-legs-5080.out`, `kt.*.tsv` (CUPTI),
`zero.ptx` + `zero_sm_{120,89,86}.sass`.

**Method.** Host-path arms are one-call programs (the suite's own boundary: `mapal_gpu_sync`, clock, sites, sync,
clock). Each arm is the shipped build against `--gpu-off=zero-host,zero-hoist,load-dispatch,split-price` linked with
the base runtime (`fac7691b`'s `mapal_gpu.c`); both runtimes instrumented with host timestamps (`HP`: clock start,
site entry, zero call, kernel lookup, `cuLaunchKernel` in/out, sync) so every time difference is located on the host
timeline. Both arms read one wisdom (same tables and PTX: one key; the OFF arm tuned it). ABBA, one process per run,
values gated ON against OFF (1e-4 relative) before timing. Decisive box: a rented **RTX 4090** (EPYC 7K62, quiet;
20 runs an arm); the local RTX 5080 (shared, load 23-49) for coarse checks (16 runs an arm).

## C1 — the host path inside the clock

| change | fact / property / capability / rule | OFF arm |
|---|---|---|
| zero kernel | capability: the runtime's own zero kernel (`mapal_rt_zero`: grid-stride `STG.E.128` + `STG.E.U8` tail, 48 SASS instructions, 14–16 registers on sm_120 / 89 / 86), loaded from its PTX at init where the module zeroes (`GPU_INIT_ZERO_KERNEL`) — an atomic output's zero is one `cuLaunchKernel`, never `cuMemsetD8Async` | `--gpu-off=zero-host` |
| hoisted zero | rule over the zero-fold walk (`access_plan` storage classes, arena offsets): a zero no kernel takes is issued right after the last step touching the output's bytes (or the sequence start), arena members only; `mapal_gpu_zero_for` skips it where the site's decided pick adds nothing (`@<fn>.zh<j>` = the site's table) | `--gpu-off=zero-hoist` |
| dispatch at load | the host registers its site tables before `mapal_gpu_init` (`@mapal_sites`); the init thread resolves each table's stored / forced pick (`choice_preset`, trace `at=load`), its candidates' kernels with their dynamic shared-memory limit (`cuFuncSetAttribute` off the clock), a cooperative table's occupancy; preloaded kernels found by id (one hash probe) | `--gpu-off=load-dispatch` |

**Intent → realized.** Host IR (`pt_pj_m1_n4096_k4096`, rtx-5080): `; mapal.zero.host sites=1 … none=1 hoisted=1`,
the `mapal_gpu_zero_for` call after the first step — before both `mapal_gpu_gen` launches and the clock — and the
site called with flags 3 (`RERUN | ZEROED`); `call void @mapal_gpu_sites(ptr @mapal_sites, i32 1)` before
`mapal_gpu_init(i32 169)`. CUPTI (`kt.*.tsv`): OFF runs 5 memsets (3 warm-zero, the program's 16 KiB / 64 MiB zero),
ON none — 5 `mapal_rt_zero` grids (4 warm, then the program's: 4 blocks for 16 KiB, 672 = 8 × 84 SMs for
`softmax_rows_4096`'s 64 MiB), all before the input generators, so the timed region holds only the sites' kernels.

**Explain (host timeline, medians, µs from the clock's start; `results.txt`).**

| program | part | OFF: zero call | OFF: first launch returns | ON: first launch returns | where it went |
|---|---|---:|---:|---:|---|
| pj_m1_n4096_k4096 | 4090 box | 2.76 → 9.54 | 16.59 | 8.29 | memset API 6.8 µs gone (hoisted), dispatch 2.6 → 0.8 |
| pj_m1_n4096_k4096 | 5080 | 1.42 → 4.48 | 9.01 | 5.01 | memset 3.1 µs gone, dispatch 1.2 → 0.9 |
| mlp_512 | 4090 box | — | 24.91 | 9.00 | first launch's `cuFuncSetAttribute` (14.3 µs in `kernel()`) moved to load; 2nd table's dispatch 7.7 → 0.6 |
| lsm_32_8192 | 4090 box | — | 11.24 | 9.24 | 3 launches 2.0 µs sooner each (dispatch at load); `cuLaunchKernel` itself 5–8 µs on this host |
| attn_2048_128 | 5080 | — | 6.67 | 5.64 | (first build regressed 1.30×: the arena-layout trial counted hoists as folds and dropped a fold — fixed, `zero_folds` counts tails only) |

**Time (ON / OFF, the shipped build against the three OFF arms; every value gate passed).**

| workload | 4090 box OFF → ON µs | ratio | 5080 OFF → ON µs | ratio |
|---|---|---:|---|---:|
| pj_m1_n4096_k4096 | 37.09 → 29.95 | 0.808 | 35.22 → 30.70 | 0.872 |
| pj_m1_n6144_k4096 | 85.01 → 76.98 | 0.906 | 86.02 → 80.04 | 0.931 |
| pj_m16_n6144_k4096 | 141.54 → 132.51 | 0.936 | 98.70 → 98.01 | 0.993 |
| pj_m1_n4096_k14336 | 286.29 → 275.86 | 0.964 | 249.43 → 241.89 | 0.970 |
| pj_m128_n28672_k4096 | 669.06 → 665.36 | 0.995 | 807.83 → 787.92 | 0.975 |
| smlp_4096_256_1024 | 119.37 → 115.02 | 0.964 | 161.31 → 160.19 | 0.993 |
| mlp_512 | 56.50 → 36.15 | 0.640 | 40.58 → 39.70 | 0.979 |
| indep_512 | 55.22 → 36.07 | 0.653 | 39.80 → 39.13 | 0.983 |
| lsm_32_8192 | 35.55 → 29.86 | 0.840 | 22.96 → 18.97 | 0.826 |
| lsm_4_32768 | 29.27 → 27.24 | 0.931 | 18.29 → 17.54 | 0.959 |
| attn_1024_64 | 30.69 → 29.85 | 0.973 | 39.65 → 38.19 | 0.963 |
| attn_2048_128 | 99.02 → 96.81 | 0.978 | 118.70 → 116.62 | 0.983 |
| softmax_rows_4096 | 2630.9 → 2629.5 | 0.9995 | 4080.0 → 3913.0 | 0.959 |
| rowsum_4096 | 2569.4 → 2565.8 | 0.999 | 3766.3 → 3688.6 | 0.979 |
| pj_m128_n4096_k4096_bf16 | 73.81 → 65.97 | 0.894 | 64.54 → 64.54 (same text) | 1.000 |

No program slower on either part. The 4090 box's wins are larger where its host is slower (`cuLaunchKernel`
5–8 µs on the EPYC 7K62 against 3 on the 5080's Threadripper): the host path is the box's CPU and driver, so every
part gains by its host. `softmax_rows_4096` / `rowsum_4096` gain on the 5080 (the 64 MiB zero, 41–72 µs of device
time, leaves the clock) and not on the 4090, whose stream-K plan zeroes nothing there.

**Open in C1.** (1) The cold first launch: a one-call program's first `cuLaunchKernel` costs 2–3× the profile's
steady `issue_ns` (5080 2.5–4.4 µs against 1.69; 4090 box 5–8 against 2.85) — `calls-issue` prices the steady
value; a first-call property (`calib.cu`, per box) is the generic next step, not attempted. (2) The merge rule for
short dependent launches (`lsm_32_8192`: 3 launches of 2–3 µs kernels) is unchanged: with dispatch at load the
gaps are the launches themselves; the one-kernel form stays refused by cost on the 5080 (33.7 vs 23.0 µs forced,
`gaps-disasm.md`), so pricing the host enqueue would not flip it there — the 4090 box (7 µs launches) is where it
could; not measured.

## H — the harness boundary

`torch_bench.py` `timed` (and `comp_bench.py`'s legs through it), `gemm_lib.cu` `time_once`, `tile_bench.cu`:
each repetition is timed as a Mapal program times itself — the flush (or the fresh operands) done and the device
idle, the host clock around the call and a synchronize. The former bracket (CUDA events behind the 256 MB flush:
the leg's host enqueue hidden under the flush's device time) still runs every repetition and prints beside as
`dev_med_ms`; `BENCH_BOUNDARY=event` (a measurement flag) restores it as the time. Not changed: `chain_bench.cu`,
`mlp_bench.cu` (no loss row reads them).

| leg (row) | part | event bracket ms | wall at the boundary ms | host dispatch inside |
|---|---|---:|---:|---:|
| torch-compile (lsm_32_8192) | 5080 | 0.0102 | 0.0635 | 53 µs |
| torch-compile (mlp_512) | 5080 | 0.0328 | 0.0732 | 40 µs |
| torch-compile (indep_512) | 5080 | 0.0328 | 0.0747 | 42 µs |
| torch-compile (smlp_4096_256_1024) | 5080 | 0.1495 | 0.2070 | 58 µs |
| torch-compile-cg (lsm_4_32768) | 5080 | 0.0143 | 0.0697 | 55 µs |
| torch-compile-cg (attn_1024_64) | 5080 | 0.0287 | 0.0747 | 46 µs |
| lt-fresh (pj_m1_n4096_k4096) | 5080 / 4090 box | 0.0205 / 0.0205 | 0.0269 / 0.0257 | 6.4 / 5.2 µs |
| cublas-fresh (pj_m1_n6144_k4096) | 5080 / 4090 box | 0.0512 / 0.1147 | 0.0582 / 0.1228 | 7.0 / 8.1 µs |
| lt-fresh (pj_m16_n6144_k4096) | 5080 / 4090 box | 0.0881 / 0.1044 | 0.0955 / 0.1096 | 7.4 / 5.2 µs |
| lt-fresh (pj_m1_n4096_k14336) | 5080 / 4090 box | 0.2314 / 0.1997 | 0.2383 / 0.2043 | 6.9 / 4.6 µs |
| lt-fresh (pj_m128_n28672_k4096) | 5080 / 4090 box | 0.7496 / 0.6595 | 0.7649 / 0.6674 | 15.3 / 7.9 µs |
| cublas (mm_16x2048x7168 cell) | 5080 | 0.0809 | 0.0881 | 7.2 µs |

The CUDA-graph legs (`-cg`) pay 46–55 µs of host time a replay: the cudagraph-tree bookkeeping and Python around
one graph launch, which the event bracket hid. Re-based rows below use the row's former leader re-timed at the
boundary; **every leg of every row must re-run with this harness before the matrix re-bases** (another leg may now
lead — e.g. a CUDA-graph or eager leg with less Python per call).

## C5 — the split / combine price

| change | fact / property / rule | OFF arm |
|---|---|---|
| priced split | property `DeviceCost::split_cell_ps` — what one more `k` partition of a DRAM-streaming atomic split costs per output cell (its partial's `red`, its blocks' prologue / epilogue as the stream issues them), rtx-5080 13 ps (fit: S = 14 / 28 at one grid spread, 79.9 / 86.0 µs); rule `TargetProfile::split_priced` = the cheapest `bytes / dram_rate · grid_spread_pm(blocks·S) + S · cells · split_cell_ps` over the counts dividing the panels inside the wave, taken by `combine` for an under-filled FMA tile below the part's FFMA balance; an unmeasured part keeps the fill rule | `--gpu-off=split-price` |

**Intent → realized.** `mm_16x2048x7168` (rtx-5080): fill rule S = 56 (896 one-warp blocks, 1.84 M `RED`), priced
S = 14 (224 blocks, smem 9472 B: the re-sized split tile), `.launch` `mapal_tile_3aebd4e2e06d7129 16 14 32 9472 14
atomic`. The other two suite cells (16 × 1024 × 4096, 16 × 18432 × 7168) keep their counts (8, 7): their price
agrees with the fill rule.

| S | blocks | spread over 84 SMs | partial cells | priced µs | measured µs (device, events) | wall (tile_bench) |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 112 | 1.500 | 229 k | 102.9 | 100.4 | 103.2 |
| 8 | 128 | 1.313 | 262 k | 90.8 | 88.1 | 90.3 |
| **14** | 224 | 1.125 | 459 k | **80.9** | **79.9** | **81.7** |
| 28 | 448 | 1.125 | 918 k | 86.8 | 86.0 | 86.6 |
| 56 (fill) | 896 | 1.031 | 1835 k | 92.5 | 88.1 | 89.1 |
| cuBLAS | — | — | — | — | 80.9 | 88.1 |

Explained: below S = 14 the grid leaves SMs with one block where others hold two (spread 1.31–1.50 — the stream
runs at the busiest SM's share); above it every extra partition adds its partials and its blocks' prologue and
epilogue beside the stream. The row: 1.086 → ~0.99 at the event bracket, ~0.90 against cuBLAS's wall.

**Open in C5 (not attempted, scope / budget):** data-parallel + stream-K (owned tiles `STG`, remainder tiles through a
workspace + ticket) — the doc's fix for `softmax_rows_4096` / `rowsum_4096`'s whole-output zero (here the zero's
host issue and device time leave the clock through the hoist instead, 0.959 / 0.979 on the 5080, but the `REDG`
combine stays); non-divisor (uneven last slice) splits (`attn_2048_128` wants S = 5); an ordered serial fixup for
the tensor k-cuts; `split_cell_ps` on other parts (one sweep a part: `tile_bench --splitk` over S at one cell). C9
(the merged epilogue losing its producer's split) not reached.

## Re-based rows (RTX 5080; Mapal = the matrix's number × this branch's ON / OFF)

| workload | before | Mapal only (event-bracket leader) | at the equal boundary (former leader at wall) |
|---|---:|---:|---:|
| pj_m1_n4096_k4096 | 1.540 | 1.342 | 1.023 |
| pj_m1_n6144_k4096 | 1.621 | 1.509 | 1.327 |
| pj_m16_n6144_k4096 | 1.096 | 1.089 | 1.004 |
| pj_m1_n4096_k14336 | 1.067 | 1.035 | 0.996 |
| pj_m128_n28672_k4096 | 1.076 | 1.049 | 1.022 |
| mm_16x2048x7168 | 1.086 | 0.996 | 0.903 |
| smlp_4096_256_1024 | 1.061 | 1.054 | 0.761 |
| mlp_512 | 1.166 | 1.141 | 0.511 |
| indep_512 | 1.253 | 1.232 | 0.541 |
| lsm_32_8192 | 1.882 | 1.555 | 0.250 |
| lsm_4_32768 | 1.184 | 1.135 | 0.233 |
| attn_1024_64 | 1.284 | 1.237 | 0.475 |
| attn_2048_128 | 1.235 | 1.213 | (sdpa leg not re-timed) |
| softmax_rows_4096 | 1.076 | 1.032 | (ms-scale: boundary null) |
| rowsum_4096 | 1.071 | 1.049 | (ms-scale: boundary null) |

Column (120 rows, the reviewed 0.978): Mapal only → ~0.972; with the boundary → ~0.917, **provisional** until every
leg re-runs with the new harness.

## Tests

`cargo test -p mapal-backend-llvm`, all 48 binaries green on the branch head (~495 tests; the full run on the
pre-fix head failed 3 — `site_choice` ×2, `splitk` ×1 — that predated `mapal_gpu_zero_for` in their assertions;
fixed and re-run with `zero_fold`, `streams`, `tf32_tile`, `tile_sites_pin`, the lib). Coverage:
`zero_fold_on_generated_programs` 120 programs, 167 atomic sites, 139 folded, 18 hoisted, 256 valued runs;
`split_price_counted_and_valued_on_unseen_programs` 48 programs, 7 splits moved (all valued), 41 kept. The hoist is
the straight-line walk only: the generator's loop trips fold every zero they can, so a loop hoist had no coverage
and is not emitted. New / extended: unit
`site_registry_lists_each_table_once_before_the_init`, `warm_zero_flag_follows_the_modules_zeros` (+ the zero-kernel
bit), `split_price_replays_the_mm16_sweep`; `site_choice` `the_measured_pick_is_stored_and_reused` (stored and forced
decisions `at=load`, a fresh wisdom still trials at the call) and the pre-sprint golden under the three OFF arms;
`zero_fold_on_generated_programs` (hoisted zeros counted in straight-line code and loop trips, values against the
interpreter and the OFF arm under every pick and serial order, zero passes never more); `site_geometry`
`split_price_counted_and_valued_on_unseen_programs` (moved splits walk whole panels, values equal the fill rule's, a
part without the property byte-equal).

## Rental

One RTX 4090 (vast offer 59820959, EPYC 7K62, $0.417/h), label `mapal-gf-lc-4090`, instance 62399718: up 03:32,
destroyed 03:38 (verified gone): ~$0.05.

## For integration

- **`benches/results/evidence/local-rtx-5080/README.md` / `matrix.*`:** re-time every competitor leg with the H
  harness (`med_ms` is now the wall at the program's boundary, `dev_med_ms` the former bracket) before re-basing; then
  the rows above. `llm_dec_b8_l8192_*` (fa2 variance) still needs its re-time (`gaps-disasm.md`).
- **CLAUDE.md facts table, backend row:** add `DeviceCost::split_cell_ps` (a DRAM-streaming atomic split's cost per
  partition-cell; `TargetProfile::split_priced`: the split by the stream at the grid's spread against the partials)
  to the `Gpu` properties list; add the site registry (`@mapal_sites` / `mapal_gpu_sites`: picks, kernels and
  occupancy resolved at load) and `mapal_gpu_zero_for` (a hoisted zero skipped where the pick adds nothing) to the
  backend runtime row.
- **`docs/tailored-fixes.md`:** none (no tailored fix).
- **Other parts:** `split_cell_ps` is measured on the rtx-5080 only; other parts keep the fill rule until their sweep
  (`tile_bench --splitk` at one memory-bound cell over S) adds the line. Every part gets C1 by its host.
