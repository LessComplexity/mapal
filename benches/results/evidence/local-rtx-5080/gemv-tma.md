# GEMV and TMA — matrix-vector sites from the in-flight knee, the bulk-copy engine as geometry (RTX 5080)

Cross-arch gap items 3–4 (`../cross-arch/README.md`, gap analysis): the m = 1 projections and the decode layers'
GEMVs keep too few bytes in flight to pull DRAM at the part's rate (worst on the H100: 1.16–2.37× cuBLAS), and
Hopper's TMA / `mbarrier` pipeline is not modeled. Raw output: `.claude/raw-evidence/local-rtx-5080/gemv-stream/`
(tests, runs `gemv-tma` / `gemv-tma2`, probes).

## What changed

| kind | name | what it is |
|---|---|---|
| property | `DeviceCost::fly_bytes_per_sm` (`Gpu::stream_fly_per_sm`) | bytes in flight per SM at which a grid on every SM first reads DRAM at 90 % of the part's measured peak (`calib.cu bw`, now printed as its `fly` line); unmeasured: Little's law at `lat_ns` over the DRAM peak an SM shares; neither: 0 |
| rule | `TargetProfile::split_stream` (`stream-fly`) | a streaming one-thread-per-cell site (flops per operand byte below the FFMA balance) is cut into the fewest `k` partitions whose grid puts a block on every SM with the knee in flight on each, inside one wave (the larger of it and the warp fill) |
| rule | `vec_fit` / `vec_takes` (`vec-cells`) | a thread folds the `16 / e_b` adjacent lanes one 16-byte load of `b` covers, where the layout facts prove the load (`b` unit-stride along the lane; base, row and `k` strides whole 16-byte granules; `c` whole vectors; `a` the same for every lane) and the split its grid needs leaves each partition ≥ `VEC_MIN_TRIPS` (8) trips of `NAIVE_UNROLL` loads; the trip's loads issue before its FMAs; each cell keeps the oracle's `k` order |
| capability | `Caps::bulk: Option<BulkCopy>` (`BULK_SM90`) | the bulk-copy engine as the geometry it executes: 16-byte granule, tensor boxes of ≤ 5 dims × 256 elements, swizzle spans 32 / 64 / 128 B, `mbarrier` transaction counts ≤ 2²⁰ − 1 B, clusters of 8 with multicast; `caps_of_cc`: compute capability ≥ 9.0 (H100, RTX 5080 / 5090; not the 4090 / A100 / 3090) |
| rule | `bulk_stage` + `BulkCopy::box_fits` (`bulk-stage`) | the warp TF32 `mma` tile's panels by the engine where its geometry covers every panel row: one warp's `cp.async.bulk` row copies, a stage's bytes counted on its `mbarrier` (`expect_tx`), every thread waiting on the phase (`try_wait.parity`); **shipped off on every part** (measured below) |

OFF arms: `--gpu-off=vec-cells`, `--gpu-off=stream-fly` (the warp fill alone), `--gpu-on=bulk-stage` (its ON arm).
Intents (`intent.py`): `vec-cells` (LDG.128 in the hot loop, no more scalar loads than vectors, `cells` FFMA per
vector, no spill; OFF: no LDG.128), `stream-fly` (a block on every SM, the knee in flight on each, the trip's loads
together), `bulk-stage` (UBLKCP + SYNCS, no LDGSTS, HMMA; OFF: LDGSTS, no UBLKCP). Static evidence of both bundles: every
check of the three intents `ok` (31 vec-cells, 33 stream-fly, 8 bulk-stage kernels in the first).

## The property: bytes in flight per SM (`calib.cu bw`, `sms = all` rows)

| part | DRAM peak | knee (90 %) | Little at the loaded latency | rules' GEMV grid before (warp fill) |
|---|---:|---:|---:|---:|
| RTX 4090 | 958 GB/s | 4 KiB | 2.9 KiB (367 ns) | 8 KiB |
| RTX 3090 | 892 GB/s | 8 KiB | 5.2 KiB (456 ns) | 8 KiB |
| RTX 5080 | 899 GB/s | 8 KiB | 6.5 KiB (573 ns) | 8 KiB |
| RTX 5090 | 1701 GB/s | 8 KiB | 5.1 KiB (510 ns) | 8 KiB |
| A100 | 1483 GB/s | 16 KiB | 10.6 KiB (735 ns) | 8 KiB |
| H100 SXM | 3177 GB/s | 32 KiB (16 KiB: 80 %) | 16 KiB (632 ns) | 8 KiB |

The warp-fill split put 8 warps of 4-byte loads (8 a thread) on every SM: 8 KiB — the 5080's and 3090's knee, a
quarter of the H100's. That is the gap's mechanism: 1 × 4096 × 4096 on the H100 ran at S = 16 (8 KiB an SM, about half
its DRAM rate); `stream-fly` now cuts it into 64 vector partitions (62 KiB an SM, LDG.128, emitted and through
`ptxas -arch=sm_90a` without spills — intents only, the H100 is validated on rented hardware).

## Design probe (`probe.cu`, `probe16.cu`: hand kernels of the same shapes, CUDA events, median)

f32 `o = a·B`, B `k × n`, split S with atomics. Scalar (one cell a thread, 8 loads in flight) against 16-byte cells:

| shape | regime | scalar best (S) | 16-byte best (S) | scalar at S = 4 | note |
|---|---|---:|---:|---:|---|
| 4096 × 4096 f32 | DRAM | 84.0 us (8) | 84.0 (32) | 114.7 | equal once the grid covers the SMs |
| 4096 × 4096 f32 | L2 | 15.6 (128) | 16.5 (32) | 38.7 | |
| 28672 × 4096 f32 | DRAM | 548.5 (4) | 540.7 (4) | 548.9 | +1.5 % |
| 4096 × 4096 bf16 | DRAM | 51.2 (16) | 46.8 (32) | 75.8 | +9 % |
| 28672 × 4096 bf16 | DRAM | 292.9 (128) | 278.5 (8) | 317.4 | +5 %, 16× fewer partitions |

At equal partitions the bytes in flight per SM are `rows·c·S·u·e / sms`, the same for either load width: the width
buys fewer LSU instructions per byte (4× for f32, 8× for bf16) and, for 2-byte weights, twice the bytes per load; the
split buys the coverage. A 2-byte scalar weight keeps half the 4-byte one's bytes in flight — the bf16 rows gain most.

## Timed (changed programs only, ABBA, iter ms, median of 20; null = the null arm's spread)

Run `gemv-tma2` (the shipped rules; 15 programs whose text differs between ON and `--gpu-off=vec-cells,stream-fly`;
value gates: 0 failed). Geometry: candidate 0's plan line (`cells`, `S`, bytes of `b` in flight an SM). Best
competitor: same precision and boundary — cuBLAS / cuBLASLt `-fresh` (operands regenerated before each repetition, as
Mapal's) for f32 projections, torch bf16 for bf16 weights, torch f32 for f32 layers.

| workload | OFF | ON | OFF/ON (null) | best competitor | Mapal/best | explanation (ON vs OFF geometry; SASS) |
|---|---:|---:|---:|---|---:|---|
| pj_m1_n4096_k14336_bf16 | 0.1550 | **0.1083** | 1.43 (6.0 %) | torch-bf16 0.1413 | **0.77** | 8 cells, S 7 → 56, 5.3 → 42.7 KiB an SM (knee 8); LDG.U16 → LDG.128 (8 lanes); kernel 0.145 → 0.097 ms |
| pj_m1_n6144_k4096_bf16 | 0.0435 | **0.0332** | 1.31 (2.3 %) | torch-bf16-compile 0.0635 | **0.52** | 8 cells, S 4 → 32, 4.6 → 36.6 KiB |
| pj_m1_n28672_k4096_bf16 | 0.3012 | **0.2600** | 1.16 (3.9 %) | torch-bf16-compile-cg 0.2744 | **0.95** | 8 cells, S 4 → 8; both above the knee: 8× fewer loads per byte; 235 MB at 941 GB/s |
| pj_m1_n4096_k4096_bf16 | 0.0314 | 0.0328 | 0.96 (5.7 %) | torch-bf16-compile 0.0471 | 0.70 | null: 8 trips a partition (the `VEC_MIN_TRIPS` edge), kernel 22 vs 20.5 us |
| pj_m1_n6144_k4096 | 0.1014 | **0.0886** | 1.14 (16 %) | cuBLASLt-fresh 0.0512 | 1.73 | 4 cells, S 4 → 16, 9.1 → 36.6 KiB; kernel 91 → 78 us (cuBLASLt reads its fresh 100 MB mostly from L2: open 3) |
| pj_m1_n28672_k4096 | 0.5522 | **0.5234** | 1.06 (1.4 %) | cuBLASLt-fresh 0.5018 | 1.04 | 4 cells at the same S 4: LDG.128, 4× fewer loads per byte; kernel 0.543 → 0.512 ms (918 GB/s) |
| pj_m1_n18432_k7168 | 0.6045 | 0.5907 | 1.02 (2.9 %) | cuBLAS-fresh 0.6124 | 0.96 | 4 cells, S 2 → 7, 13.7 → 48 KiB; within null |
| pj_m1_n4096_k14336 | 0.2509 | 0.2487 | 1.01 (4.0 %) | cuBLAS-fresh 0.2294 | 1.08 | 4 cells, S 7 → 28; null |
| pj_m1_n2048_k7168 | 0.0326 | 0.0300 | 1.09 (4.6 %) | cuBLAS-fresh 0.0266 | 1.13 | 4 cells, S 14 → 56; kernel 21.5 → 21.0 us (L2-resident, 0.79× cuBLAS's kernel; the rest is the runtime's site decision) |
| pj_m1_n4096_k4096 | 0.0371 | 0.0353 | 1.05 (21 %) | cuBLASLt-fresh 0.0205 | 1.72 | 4 cells, S 8 → 32; kernel 24.5 vs 25.0 us: null (L2 knee: open 3) |
| llm_dec_b1_l2048 | 3.919 | **3.803** | 1.03 (0.3 %) | torch-compile-ro 4.005 | **0.95** | its gate / up / QKV GEMVs as above; the O site's ordered epilogue kernel unchanged (open 2) |
| llm_dec_b1_l2048_bf16w | 2.502 | **2.398** | 1.04 (0.3 %) | torch-bf16-compile 2.236 | 1.07 | bf16 weights: 8 cells |
| llm_blkdec_b1_l4096 | 17.27 | **17.11** | 1.01 (0.1 %) | torch-compile 17.05 | 1.00 | |
| llm_blkdec_b1_l4096_bf16w | 10.29 | **10.15** | 1.01 (0.2 %) | torch-bf16-compile 8.82 | 1.15 | |
| llm_dec_b8_l2048 | 7.005 | 7.008 | 1.00 (0.2 %) | torch-compile 10.71 | 0.65 | B 8: tile sites, its one changed GEMV a small share |

The first run (`gemv-tma`, before `VEC_MIN_TRIPS`) found the two losers that fact separates: 1 × 1024 × 4096 at 0.87×
(one vector block a partition needed 128 partitions of 4 trips against the scalar cells' 32) and the 1024 / 256-wide
GEMVs of decode_1024_256's loop at 0.78× (1–2 trips a partition, a quarter of each block's threads idle); with the
fact both are the OFF emission byte for byte (no longer in the changed set).

Safety net: the full suite at 27258765 (35 programs + 2 cells, `gemv-stream/full/`): value gates 0 failed; against
the last full-suite run (`long-prefill/suite`) every row within 0.93–1.08 — the two changed suite rows (pj_m1_n2048,
n18432) 0.97, the ends (ew_4096 0.93, attn_1024_64 1.08) programs whose text did not change (cross-run variation).

## TMA on the RTX 5080 (sm_120)

Probed (`tma_probe.cu`, `mc_probe.cu`): `cp.async.bulk.tensor.2d` with a `cuTensorMapEncodeTiled` box lands bit-exact
(SASS `UTMALDG.2D`, `SYNCS.ARRIVE.TRANS64`, `SYNCS.PHASECHK.TRANS64.TRYWAIT`); clusters of 1–8 launch (16 refused:
non-portable); a `.multicast::cluster` box reaches both blocks of a 2-cluster bit-exact, ptxas advising that the
modifier "should be used on sm_90a / sm_100a … reduced performance" elsewhere. So `BULK_SM90` lists multicast (it runs)
and the rule never asks for it yet.

`bulk-stage` (row copies) is emitted for sm_120 and sm_90a (the H100's `wgmma` arm off; `ptxas -v` clean, 6 UBLKCP /
2 SYNCS.ARRIVE / 2 SYNCS.PHASECHK in the 5080's kernel) and is exact: on 23 generated TF32 GEMMs the emulated device
equals the `cp.async` arm byte for byte in every emulator mode (the emulator applies a bulk copy only when a thread
waits on its barrier; a wrong-parity mutation fails the test). Timed ON / OFF (first run, `gemv-tma`):

| workload | OFF (cp.async) | ON (bulk rows) | kernel ON / OFF |
|---|---:|---:|---:|
| pj_m128_n4096_k4096_tf32 | 0.136 | 1.961 | 1.961 / 0.135 ms |
| pj_m128_n28672_k4096_tf32 | 0.697 | 4.774 | |
| pj_m2048_n4096_k4096_tf32 | 1.419 | 7.570 | |
| mlp_2048_tf32 / mlp2_2048_tf32 | 0.832 / 1.670 | 3.609 / 6.800 | |

4.3–14.5× slower: one bulk copy per 64–128-byte panel row (96 a stage at 64 × 32) issued from one warp — the
engine's per-copy issue against 256 threads' 16-byte `LDGSTS` each. The rule ships it off on every part
(`rung_replay.tsv`); the capability, the geometry test and the exact emission stay as its ON arm.

## Tests

`gemv_stream.rs`: the rules per part and arm (5080 S = 32 four cells a thread, H100 S = 64; each OFF arm), and 64
generated GEMVs (`testgen::GemvProg`: f32 / bf16 / f16 weights, unit, strided, offset and per-row layouts, a shared
activation, an epilogue) — exact face byte for byte, default face within its bound, in every emulator mode at 1, 2, 8
and 84 SMs and under the `vec-cells`, `stream-fly`, `splitk-atomic` OFF arms: coverage 64 programs: 7 f32 and 6 bf16 / f16 vector kernels, 10 split (atomic and workspace), 2 with an epilogue, 6 ordered, 2 with rows > 1; 4 strided and 7 misaligned scalar fallbacks, 28 scalar streaming splits.
`bulk_stage.rs`: the capability per part (5080, H100 with `wgmma` off: bulk; 4090: none; rules: off) and 48 generated
TF32 GEMMs (23 bulk-staged). `profile.rs` `split_stream_reads_the_in_flight_knee`. A vector-lane mutation and a
barrier-parity mutation each fail their test. Backend suite: 447 passed, 45 binaries.

## Open

1. TMA tensor boxes: one `cp.async.bulk.tensor` a panel (swizzled for `ldmatrix`) needs a host-encoded descriptor
   per operand pointer (`cuTensorMapEncodeTiled` in the runtime, a 128-byte `__grid_constant__` parameter, its
   emulation); the per-copy issue cost of the engine is then the property the staging rule reads. Wgmma fed from
   TMA-staged smem and warp-specialized producers follow from it (not built: the H100 path cannot be validated here).
2. The ordered GEMV of an epilogue group (decode's `c = 4096, k = 4096` site with its residual, `combine=none`): no
   split, 16 blocks — the knee is not reached on any part (8 KiB an SM: the H100's quarter).
3. L2-resident operands: the knee is DRAM's; an operand the residence plan keeps in L2 needs the L2 knee (5080 32 KiB:
   1 × 4096 × 4096 at 64 MB, S = 16–128 → 15.6 us against 19.8 at S = 8) — `residence_plan` at the launch.
4. Weights stored `n × k` (PyTorch's layout) read column-strided by the one-thread-per-cell kernel: a warp a row with
   a shuffle tree (`row_fold`) is the coalesced form; not generated by the suite.
