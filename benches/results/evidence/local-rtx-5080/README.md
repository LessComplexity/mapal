# local-rtx-5080 — first local competitor run (2026-11-19)

Machine: AMD Ryzen Threadripper 9970X (Zen 5, 32C/64T, AVX-512, 4 × 32 MB L3), RTX 5080 (sm_120, 84 SMs,
64 MB L2, 16 GB), Arch Linux, clang/LLVM 23, CUDA 13 at `/opt/cuda`, driver 620.70. The GPU also drives the
display; a game was closed before timing, Docker Desktop's VM and a browser stayed up (~9 % CPU each).

## Commands

| pass | command |
| --- | --- |
| GPU bundle | `benches/local/gpu_bundle.sh <bundle>` (README "GPU workloads" suite, 35 programs + 2 cells, `rtx-5080` profile) |
| GPU run | `LADDER=1 LOCAL_GPU=<dir> benches/local/gpu_run.sh <bundle>` (`LADDER=1` times "carried" programs too; run 1 lacked it, so 7 rows are run 2 only) |
| CPU matmul | `CYCLES=15 PYTHON=<venv>/bin/python benches/local/cpu_matmul.sh <out>` |
| CPU shapes | `CYCLES=25 PYTHON=<venv>/bin/python benches/local/cpu_shapes.sh <out>` |
| CPU NN | `TARGET=native NN_CFLAGS="-O2 -march=native" NN_CONF_FLAGS=--no-contract PAR=64 CYCLES=41 OUT=<out> PYTHON=<venv>/bin/python benches/nn/nn_ab.sh` |

Competitors: cuBLAS / cuBLASLt / CUTLASS `b2f249a` (stock + tuned) / hand CUDA (chain_bench), PyTorch 2.11.0+cu130
eager and `torch.compile` (TF32 rows with TF32 on). CPU: NumPy 2.5.5 on scipy-openblas 0.3.34 in a venv (the
system NumPy links reference cblas, 250 ms at 1024² — not a valid opponent), C++ naive 1t / mt.

## Files

`gpu/gap-run1.md`, `gpu/gap-run2.md` (Mapal / best other per workload, null), `gpu/evidence-rtx-5080.tsv`
(run 2: 10695 checks, 401 mismatch, 66 n/a — not yet triaged), `gpu/wisdom-rtx-5080.tsv`, GPU app snapshots;
`cpu/matmul-summary.md`, `cpu/shapes-summary.md`, `cpu/nn-report.txt`.

## Known invalid / open

- **The CPU NN NumPy leg was a cold single iteration per process** (`nn_numpy.py`, no warm-up). Warm, NumPy nn2
  d=1024 is 1.52 ms vs 8.70 ms in the harness, so the NN "1.3–5.4× Mapal" ratios were a boundary artifact. The
  matmul NumPy leg was warm (`numpy_bench.py` runs 3 untimed passes); its ratios stand at 64 threads. Corrected
  tables (warm NumPy, each side at its best thread count — OpenBLAS is faster at 32 threads): `cpu-analysis.md`.
- Mapal threaded < Mapal 1t on conv2d / saxpy / reduce: the Linux `native` profile had no dispatch-cost facts
  (fixed in 1ffe0137, `cpu-analysis.md`).
- GPU: explained in `gpu-analysis.md` (CUPTI traces + SASS of both sides). The largest share of the small-program
  losses is the runtime's first site decision (wisdom read) inside the timed region, 50–90 us per program; the
  kernel causes, the calibrated `rtx-5080` DeviceCost (`calib/`) and its neutral A/B are there. Rows whose working set
  fits the 64 MB L2 compare an L2-warm Mapal with torch's flushed L2: open.

**compile-time-graph** (`compile-time-graph.md`): `unit_plan` row-block units (attention one kernel when forced; the
plan model keeps the boundary — the in-unit q·Kᵀ reads Kᵀ at half a sector a lane) and compile-time launch regions
replayed as CUDA graphs (rt_mlp2_1024 0.92× by overlapping independent chains; decode family neutral: a 10-node
graph launch costs as much as its eager ops on this driver). Intended vs realized, CUPTI, ABBA, open items there.

**kernel-merging** (`kernel-merging.md`): runtime graph replay removed; `access_plan` (the compile-time dependency fact)
and `residence_plan` (where a value's bytes live) in mapal-ir; the row-block model prices attention by one rule on the
4090 / 5090 / 5080; `zero-fold` puts an atomic site's zero pass into an earlier kernel the dependency fact allows —
decode_1024_256 0.878× (ABBA), now 0.95× hand CUDA. Intended vs realized, CUPTI, ABBA, open items there.
Rows step: `row-stage` stages a row-block unit's tile panels through shared memory (`cp.async`, swizzled 16-B
chunks, one or two buffers, column passes, priced by the site model). attn_1024_64 forced as one unit: 32.9 → 28.4 us
(model pick) / 25.7 us (best), against 30.8 us of split kernels; iteration ABBA 0.932 / 0.857 of the split. The plan
model still ranks the unit above the split (missing facts listed), so row-block stays off by default.
Horizontal step: independent launches of a trip (no `access_plan` edge) run as one grid dispatching on block index to
each member's kernel as a device function, where no member loses a wave and their last waves share one — decode
0.658×, causal 0.774×, gen 0.776×, prefill 0.782× (ABBA; the KV rows and the query side by side); decode now 0.62×
hand CUDA.
Calls step: a chain of dependent launches as one persistent kernel (device-function bodies over a co-resident grid,
grid barriers only where `access_plan` orders, an atomic site's zero pass in the kernel), priced by measured
properties (`calib.cu gsync`: barrier 553 ns, plain / programmatic / cooperative dependent launch 650 / 486 / 2044
ns). The cooperative launch's own cost makes the rule refuse every suite chain on this part; forced, they measure
1.03–1.32× slower (ABBA), so the shipped code is the OFF arm's. row-block's per-part rows are gone (its plan model
decides; on this part it still refuses the attention unit that measured faster — the missing facts, single-wave phase
serialization and row-segment latency, are in the section).

**hardware-properties** (`hardware-properties.md`): the per-architecture rung table (`GPU_RUNGS_OFF`, 131 rows) is
gone; every rung is decided by the rung rules over physical properties (`rules.rs`: FFMA per DRAM byte, registers per
resident thread, L2 and DRAM per SM, measured launch and copy costs, capabilities as properties; no name or version
compare), replayed against every cited measurement (`rung_replay.tsv`: 95 picked, 67 null, 13 open with the
missing fact named; 4 mixed). `native-gpu` is the device query + the calibration file (`calib/rtx-5080.calib`). On this part:
`pdl` ships (decode / causal / gen / prefill 0.91–0.92×, ABBA; short programs 1.5–2.5 % slower: mixed, open);
every kernel now waits under PDL (it raced before).

**model-scale** (`model-scale.md`): one Llama-3-8B-like layer (d 4096, 32 × 128 heads, 8 KV heads, SwiGLU 14336,
RMSNorm, RoPE) in idiomatic Mapal — decode steps (B 1/8/32 × KV 2048/8192), prefill 2048/8192, the whole block, the
projections at m 1–2048, TF32 twins — against cuBLAS / cuBLASLt, torch eager / compile / SDPA and a bf16 reference,
value-gated at both scales. Projections: m 2048 at parity, TF32 1.12–1.22×, m 128 1.07–1.41×. The natural layer
programs run their attention on the host (the bounds proof uses neither guard conditions nor loop-carried ranges:
RoPE's indices stay trap-guarded) — 669–2053× torch per decode step, minutes per prefill; with those indices
written provable (`_pv` diagnostics) the device path is 1.32–3.96× (decode) and 7.8–14× (prefill) torch-compile f32,
from per-cell kernels for GEMM + residual and for attention, a whole-cache update map per step and one-thread
RMSNorm. Ten open findings with minimal repros there.

**onto-device** (`model-scale.md` [Onto device](model-scale.md#onto-device-2026-11-21)): the natural layer programs on
the device — `bounds_proof` ranges indices by the conditions where they run and by loop counters (`cond-bounds`), loop
exit values readable in later bodies, `carry-made` placement, `host-par`, i64 extents. Decode 217.8 → 0.668 ms/step
(B1 L2048; 1.32–3.96× torch-compile f32 over the suite), prefill 2048 280 s → 86.7 ms (10.4×), the 8192 prefill 1.01 s
(12.1×; no device form before); the gaps are findings 6–9.

**device-geometry** (`model-scale.md` [Device geometry](model-scale.md#device-geometry-2026-11-21)): findings 6–9 —
`tile_plan_post` (a GEMM whose map body goes on past its fold is a tile site; the body after the fold is its store's
epilogue, or a split's reduce), batch digits of mixed-radix sites (a KV head's query heads share one staged K/V panel:
`row_group`, `gpu_group_tile`), `Delta::Pieces` (a per-sequence cache append written as B pieces in place) and the
row-tree fold beside a norm's root. Decode 0.668 → 0.240 ms/step at B1 L2048 (0.93× torch-compile f32), 0.37–0.96×
torch-compile over B 1–32 × L 2048–8192; block decode 1.00×; prefill 2048 86.8 → 8.82 ms (1.06×), block prefill
276.6 → 32.4 ms (1.16×); the 8192 prefill unchanged (open). Main suite: prefill_1024_512_256 0.966 (ABBA), 34
programs byte-identical; the first version's loss on decode-shaped loops (1.05–1.08) is the k-major lone-row rule.

Raw output (results txt, box.out, CPU logs) is kept locally in `.claude/raw-evidence/local-rtx-5080/`; not yet
uploaded with `benches/gpu/archive_raw.sh` (that publishes a GitHub release).

**long-rows** (`long-rows.md`): `fold_combine` in mapal-ir — how a fold's state combines across blocks of its axis and
the law (factor, shift, or phase) each per-row value its term reads gets from the dataflow; the `row-slices` rung runs a
flow unit's few long rows over several blocks (partials + an in-order tree combine, phases at grid barriers of one
cooperative launch), priced by two new measured properties (`row_iter_ns` 87 ns, `coop_first_ns` 38 us).
pt_lsm_1_131072 (softmax over one 131072-long row): flow kernel 182 → 7.6 us, iteration ABBA 0.290; every other
program's emission unchanged. No exponential law in the IR: attention's `den` is a phase, so flash-style attention
stays open (reasons and the model-scale rows there).

**exp** (`exp.md`): `x -> exp` is a pipeline builtin (ADR-0042) — the oracle's algorithm `E_b` (exhaustively ≤ 1.221
ulp in f32) byte for byte on the exact faces (interpreter = host = RTX 5080 = CUDA backend), and on the default face
the host's vectorizable contracted polynomial and the device's `ex2.approx` form (exhaustively ≤ 2.068 ulp; admitted by
the measured `Caps::ex2` against the 2⁻²¹ bound); `fold_combine`'s rescale law makes a row-sliced softmax fold its sum
beside its max (lsm_1_131072: 3 → 2 phases, flow kernel 7.94 → 5.89 us). The softmax / attention / model-scale
programs use `-> exp`; torch legs (incl. SDPA) now value-gate within 10⁻⁶. Suite ON vs OFF: lsm_4_32768 0.950 (ABBA),
the rest null; flash-style attention and the walk-cost fact (lsm_4_32768 vs the polynomial program) are open.
