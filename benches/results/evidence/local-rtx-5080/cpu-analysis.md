# local-rtx-5080 — CPU: dispatch fact, warm boundary, why Mapal loses (2026-11-19)

Threadripper 9970X (Zen 5, 32C/64T, AVX-512, L1d 48 KB, L2 1 MB/core, L3 4 × 32 MB). Code: `1ffe0137`.
Raw: `.claude/raw-evidence/local-rtx-5080/` (not in git).

## 1. The fact: what a dispatch costs on this host

Before: Linux `native` had `dispatch: None` → every split task woke all 64 lanes. The runtime's own probe
(`crates/mapal-rt/examples/dispatch_cost.rs`, `run 1 task / 256 slices`): **24 / 50 / 149 / 263 µs at 8 / 16 / 32 /
64 lanes** (M4: ~8 µs). A wave's cost is per lane woken (`Pool::notify_lane`, one wake per filled lane).

Now: `native` reads `DispatchCost { dispatch_ns, lanes, flops_per_ns, bytes_per_ns }` from
`~/.cache/mapal/native-dispatch.tsv`, measured on a miss (`mapal_rt::calibrate::wave`: begin → split task, one
slice per lane → finish, workers parked, minus the same task inline; median of 64). Here, three measurements:
**89.7 / 95.2 / 100.3 µs at 64 lanes** (1.5 µs a lane), 282–284 flops/ns (one-lane AVX-512 FMA), 117–121 B/ns
(one lane, L3-resident stream). The emitter registers each split's `width = round(sqrt(T / w))` (`T` = the
tile- or cell-site roofline, `w` = `dispatch_ns / lanes`); 1 = inline, ≥ lanes = whole pool. `MAPAL_NATIVE_DISPATCH=off`
is the OFF arm. Hand-measured profiles have `lanes = 0`: unchanged byte for byte.

Tests: width/cache/probe units; `tests/dispatch_width.rs` (48 testgen chains on an 8-lane pool: 376 partial
widths, 780 inlined splits, values = interpreter bit for bit); workspace 1347 pass (cuda nvcc sweep: pre-existing,
first run on a host with nvcc, sm_89 binaries JIT on sm_120 → 15 s timeouts).

## 2. ON vs OFF (same box, interleaved, medians)

Shapes, fma face, 64 lanes, 25 cycles (ms). Null: the 1t legs (same work in both builds) moved 0–6 % (15 % on
saxpy's 0.1 ms leg); NumPy-1t 1–3 %.

| workload | OFF | ON | ON/OFF | Mapal 1t (median) |
| --- | ---: | ---: | ---: | ---: |
| FIR 1M | 0.365 | 0.277 | **1.32×** | 0.994 |
| conv2d 1024² | 0.326 | 0.143 | **2.28×** | 0.147 |
| saxpy 1M | 0.377 | 0.116 | **3.25×** | 0.139 |
| sum 1M | 0.485 | 0.467 | 1.04× (fold not split) | 0.465 |
| transpose 1024² | 0.392 | 0.277 | **1.41×** | 1.188 |
| gather 1M | 0.452 | 0.305 | **1.48×** | 0.575 |

Threaded is no longer slower than one lane on any shape (conv2d / saxpy: equal within noise).

NN forward, Mapal threaded (ms, 21 cycles): d=128 nn4 0.315 → 0.175 (1.8×), d=256 nn4 0.587 → 0.266 (2.2×),
nn2x2 0.429 → 0.185 (2.3×), d=512 nn4 1.003 → 0.781 (1.28×); d ≥ 1024 equal. **Still open:** at d=128 ON
(0.081 ms nn2) is slower than Mapal 1t (0.047): width 3 is chosen from the empty-slice wave (1.5 µs a lane); a
program's handoff costs more (saxpy at 64 lanes paid ~4 µs a lane: data written on the host, read on cold cores).

Matmul 1024–4096: ON 0.7–5 % slower than OFF on every leg; NumPy's own run-to-run swing was 3.5 % → within noise,
direction consistent (the B pack at 4096 gets width 29: its DRAM-resident bytes are priced at the L3 rate). Open.

## 3. Competitors, fair boundary

`nn_numpy.py` timed one cold call per process (8.70 ms vs 1.52 warm at nn2 d=1024): now 3 untimed passes first
(`NN_WARMUP=0` = old arm). `numpy_bench.py` (matmul) already warmed up 3×. Mapal's pool is created at
`mapal_par_begin`, before its `time` bracket, and used by data generation: its legs were already warm.

OpenBLAS at 64 threads is handicapped by SMT: 4096 runs **23.0 ms at 32 threads vs 33.5 at 64**. Best of 32 / 64
threads per side (7-rep medians, NumPy warm, ON build):

| workload | Mapal best | NumPy best | NumPy / Mapal |
| --- | ---: | ---: | ---: |
| matmul 1024 | 0.767 (64) | 0.876 (32) | **1.14×** |
| matmul 2048 | 3.594 (64) | 5.333 (32) | **1.48×** |
| matmul 4096 | 35.85 (32) | 23.0 (32) | 0.64× — NumPy 1.56× faster |
| nn2 / nn4 d=512 | 0.444 / 0.873 | 0.498 / 0.955 | **1.12× / 1.09×** |
| nn2 / nn4 d=1024 | 1.588 / 3.075 | 1.515 / 2.999 | 0.95× / 0.98× (parity) |
| nn2 / nn4 d=2048 | 7.270 / 14.49 | 10.62 / 21.60 | **1.46× / 1.49×** |
| NN d=128 (1t vs 1t) | 0.047 nn2 | 0.038 | 0.80× — NumPy faster |
| sum 1M (1t) | 0.459 | 0.144 | 0.31× — NumPy 3.2× faster |

The earlier "NN 1.3–5.4×" was the cold NumPy call; superseded.

## 4. Decompiled: why Mapal loses

**Sum reduction (3.2× behind NumPy).** The default face (ADR-0040: order unspecified) emits the whole fold as
task `kind 0, n 1` with a strict `fadd` (no `reassoc`); `clang -O3 -Rpass-missed=loop-vectorize`: *loop not
vectorized*; asm: 32 × `vaddss` (scalar, latency-bound: 0.46 ms ≈ C++ scalar 0.46). The facts that license
lane partials and a tree (`whole_fold`, `fold_tree_step`) are read only by the GPU path (`func/resident.rs`).
Headroom: 4 MB at 117 B/ns ≈ 0.04 ms on one lane.

**Matmul 4096 (NumPy 1.56× at best threads).** The micro-kernel is sound: 4 rows × 4 zmm = 16 accumulators,
16 `vfmadd231ps zmm` per 4 `vmovups` + 4 `vbroadcastss`, B packed; one lane 278 GFLOP/s at 1024 (= the 284 peak
probe). But the k loop runs the whole K (`cmp $0xffe`): the packed B panel is K × 64 × 4 B = **1 MiB = all of L2**
at 4096 (512 KiB at 2048). One lane: 276 GFLOP/s at 2048 → **235 at 4096 (−15 %)**; threaded 4,780 → 3,830
(−20 %) while OpenBLAS (k-blocked) rises to ~5,975. `tile_kc` derives kc = 256 here, but `kc_nest` ships off
(`lib.rs` EmitOpts: it lost at every depth on i9 / M4 as implemented).

## 5. Missed optimizations / missing facts, ranked

| # | what | evidence | status |
| --- | --- | --- | --- |
| 1 | order-free whole fold → vector lane partials + tree split on CPU (`fold_tree_step`) | scalar `vaddss`, not vectorized; 3.2× behind NumPy, ~12× headroom | recommend |
| 2 | k-blocking when the packed B panel reaches L2 (re-examine why `kc_nest` lost: C reload per k-block?) | 1t −15 % at 4096; NumPy 1.56× | recommend |
| 3 | SMT as a fact: lanes per core for compute-bound tile sites | OpenBLAS +46 % at 32 thr; Mapal 64 better at ≤2048, ≈ at 4096 | open (missing fact) |
| 4 | the wave's per-lane cost with data handoff (host-written data read on cold cores, per-CCD L3) | d=128 NN threaded 1.7× slower than 1t; probe 1.5 µs vs ~4 µs seen | open |
| 5 | rate by residency (DRAM vs L3) in the width estimate | 4096 pack width 29; matmul ON 2–5 % (within noise) | open |
| 6 | register tile uses 16 of 32 zmm (`tile_i` policy ×2 headroom) | asm | recommend (policy search, ADR-0034) |
