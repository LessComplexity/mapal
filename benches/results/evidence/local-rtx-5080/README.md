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

Cross-part matrix — Mapal vs the fastest same-precision competitor per family, this box and the rented parts: [matrix.md](../matrix.md).

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

**harness** (`harness.md`): cheaper steps, same checks — the emulator's serialized modes hand the turn to one thread
(a block-wide broadcast per hand-off was their whole cost) and generated programs are checked in parallel
(`MAPAL_TESTGEN_JOBS`, deterministic draws, index-ordered results): 14 test binaries 3790 → 558 s, every coverage
counter identical on the fixed seed; changed-programs GPU runs per A/B (`gpu_changed.sh`: only programs whose host or
device text differs, ABBA) plus one full suite per step; incremental bundles (`BUNDLE_CACHE`: 155 → 47 s, the same
files). Flow: `benches/local/README.md`.

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

**large-gemm** (`large-gemm.md`): what cuBLAS / CUTLASS run on the large multiplies (captured: the 128-accumulator
256 × 128 SIMT tile, split-K 3 / 5 / 8 / 9 to fill 84 SMs, 3–6 `cp.async` stages) as rules over properties —
`wide-fma` where the part issues ≥ 45 FFMA per DRAM byte (and its grid fills the part), `streamk-fill` (stream-K
only over a grid that fills the part at its own geometry; a skinny grid takes the fill split), `k-fill` (an
under-filled site keeps the full tile and streams `k` over every slot the tile holds). Cells vs cuBLAS: 4096³
1.14 → 1.01–1.04, 16 × 2048 × 7168 1.41 → 1.10, 128 × 4096 × 4096 1.53 → 1.06; model-scale m 128 projections 1.07–1.16
(were 1.07–1.41), prefill 2048 8.82 → 8.63 ms, block prefill 32.4 → 27.4 ms; suite 2048/4096 GEMMs 4–10 % faster.
Open: the 6-wave epilogue tile's wide-tile loss (2.5 %), TF32 split / depth, a cp.async pipeline with stream-K.

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

**cross-arch rules** (`../cross-arch/rules.md`): the rules the four-part round (`../cross-arch/README.md`) showed wrong,
each as a property, replayed against its ON/OFF arms — `wide-fma` / `async` / `kmajor-a` read DRAM bytes per µs per SM
(clock-free: the 250 W 4090's 17 / 3 / 10 wins kept at its measured 1.96 GHz); `pdl` only where a module's launches
repeat in a loop, the dependent triggered at the producer's exit (`pdl-early` arm; H100 geomean vs plain 1.005 →
0.992); `row-split` priced at the site (one SM's `sm_bytes_per_ns` against the launches it adds: the H100's 4 losses
off); the tune trial settles to steady state and its head-to-head pairs decide (5 / 16 steady-state-slower picks
reverted); `calib.cu` gains the sustained clock, the PDL chain head and the one-SM rate; the rented parts' profiles
carry their measured costs. `min-blocks`, `warp-fill` and the 3090's `grid-share` convmc site stay open. RTX 5080:
decode 0.981, short programs 0.989–0.990 (inside null), loopmm 1.003.

**review-fixes** (`review-fixes.md`): the four findings of the independent review after f3a9e792 — the rescale law
over wholly masked slices (NaN → the sequential fold's value; lsm_1_131072's sliced kernel 5.888 vs 5.888 us), a
loop's next state made once (no device carry: the device printed the initial value), `calls`' missing body price
(`coop_body_cycles`: `calib.cu body` measures it here, 4390 cycles = 1792 ns a chain; every other part derives it at its
own clock — 4090 / 3090 form causal / gen / prefill, nothing re-forms the H100's smlp / firew losers; this part
unchanged), and `intent.py` entries for `row-slices`, `exp-hw`, `exp-rescale` (44 checks ok). The review after
60f0f8ce (same file, second part): a `bf16` operand's pointer moved by a loop counter in the product's 4-byte elements
(`uniform-base` × `narrow-radix`: wrong values; now its own 2-byte elements, and its copy size on the tile launch path),
and a callee's device loop that returned its array without copying it out (`escapes` is now a host use) — both fixed
with generated tests (`SliceProg::narrow`, `HistoryProg::callee`); no shipped suite emission moves, nothing timed.

**half-precision** (`half-precision.md`): ADR-0043 — `bf16` / `f16` storage types (`to_bf16` / `to_f16` round an f32 once,
`widen_f32` widens exactly; no arithmetic on narrow values; the interpreter's bits on every face and backend),
`tile_plan_narrow` sites on the one-thread-per-cell kernel (`LDG.U16`) and the register-staged tile (panels widened as
copied), operand bytes per format in the cost model; every non-narrow program emits byte-identical modules. The
model-scale decode rows with bf16 weights: 0.59–0.98× the f32 program (B1 L2048 0.149 ms/step against torch bf16-compile's
0.143; projections at half their bytes, CUPTI), the remaining gap attention over an f32 cache; 16-bit `mma` and fp8 open.

**tensor-cores** (`tensor-cores.md`): ADR-0043 slice 2 — `: bf16 | f32` / `: f16 | f32` compute members (D3's bound
with `u_t` 2⁻⁸ / 2⁻¹¹); matrix instructions as capability geometry (`Caps::matrix`: shape, operand format, operand and
accumulator homes, the accumulator measured — `mma_calib.cu`: every warp format truncates, so `{f32}` never takes a
tensor core); `mma.sync.m16n8k16` bf16 / f16 tiles (`cp.async` panels in storage format, `ldmatrix`, f32 accumulate):
projections at m ≥ 128 1.6–2.7× the FMA tile, 1.17–1.47× cuBLASLt bf16; prefill bf16 projections 2.4–2.7×; the bf16 KV
cache as mixed-radix tile sites (`narrow-radix`): B ≥ 8 decode −13–26 %, 1.36–1.81× torch bf16 (was 1.6–2.5×);
`narrow-async`, `staged-regs` (the block decode's 216–244 B spills gone), a host frame-size fix for narrow arrays. fp8
types, the 16-bit tile over radix sites and split-k open.

**long-prefill** (`long-prefill.md`): cross-arch gap items 1–2. The fact that kept the 8192-token prefill per-cell was
`tile_plan_radix` refusing an address term on any map capture but the cell index — the KV-group loop's counter `g`
(every index fit `i32` and was proven in bounds). `uniform-base` gives such terms a digit form checked at the capture's
range (`BoundsProof::capture_range`); the launch moves the operand pointer. With `carry-take` (a carry takes its
dying init's device copy: the 134 MB `y` upload gone) the T8192 prefill is **1064.8 → 82.06 ms, 0.97× torch-compile**
(was 12.1×), values unchanged. `slice-launches` meets a sliced unit's phases at kernel boundaries where launches
undercut a cooperative launch (outside a loop its first one): lsm 1 × 131072 / 4 × 32768 0.65× / 0.87× the best torch
leg (were 2.3× / 3.6×); 32 × 8192 1.80×. Open: flash-style streaming (an operand read through `exp(s − m)`, then the
streamed row-block unit), causal block skips, attn_1024_64.

**caching-memory** (`caching-memory.md`): the automatic KV cache and memory reuse, time and peak device memory
(`MAPAL_GPU_PEAK`, against `torch.cuda.max_memory_allocated` one process a leg). A Llama-layer decode written
naively (each step recomputes RMSNorm, the fused projection and RoPE over the whole carried history) gets its cache
from `unchanged_plan`: after the first trip it runs the explicit-cache program's kernels at their cost (CUPTI), 4.5–12.5×
faster than torch running the same naive program at 0.84–0.96× its memory; 1.19–2.18× the explicit program (the
first trip projects all 6144 columns — the missing fact is demand). The explicit-cache program: 0.65–1.00×
torch-compile's HF-style time at 0.69–0.96× its peak. New: `retain-same` (a value the same every trip launched /
uploaded once), `cone-span` (loop buffers take the bytes of what died before the loop: −19 % peak on the explicit
cache's B ≥ 8 rows), `arena-fit` (largest-first packing where it is smaller and folds as many zeros: prefill T2048
−9 %); fixed: a seedless back-edge value claimed "unchanged", a widened delta's first piece skipped below row 0.
Open: demand, the gathered-rows split rule, prefill peak (1.34–1.54× torch: materialized scores).

**competitors** (`competitors.md`): "best competitor" now means the fastest real-world implementation — torch.compile
with CUDA graphs (`max-autotune`, `reduce-overhead`), SDPA's efficient and cuDNN backends, hand-written Triton (the
tutorials' matmul / softmax / fused attention, autotuned), Liger, FlashAttention-2 (built from source: no wheel for
torch 2.11 + CUDA 13), FlashAttention-3 (sm_90a boxes), FlashInfer; TensorRT-LLM is not installable on this stack
(every release pins its own torch: 1.3.0 → 2.14). Every leg value-gated, timed like the torch legs, in the box flow
(`comp_bench.py`, `comp_build.sh`); the matrix lists every leg per row and part by family. On the 5080, 28 of 86 rows
get a new best (CUDA graphs 20, Liger 4, Triton 2, FlashInfer 1, cuDNN 1): geomean Mapal/best 0.980 → 1.001, no win
disappears; the f32 prefills go even → loss (1.19, 1.53) and the TF32 prefill 1.23 → 1.64 — fused attention kernels
(the CUTLASS memory-efficient f32 kernel, the Triton flash tutorial) against Mapal's three-kernel attention with the
scores through DRAM (long-prefill open 1–2). ABBA and CUPTI for every move > 2 %.

**gemv-tma** (`gemv-tma.md`): cross-arch gap items 3–4. A measured property — the bytes in flight per SM at which a
grid on every SM reads DRAM at 90 % of peak (`calib.cu bw` `fly`: 4090 4 KiB, 5080 / 3090 / 5090 8, A100 16, H100 32;
the warp fill gave every part 8) — sizes a streaming one-thread-per-cell site's `k` split (`stream-fly`), and the
layout facts give a thread the lanes one 16-byte load of `b` covers (`vec-cells`, LDG.128, at least `VEC_MIN_TRIPS`
trips a partition). RTX 5080: bf16-weight GEMVs 1.16–1.43× (0.52–0.95× torch bf16), f32 1.01–1.14× (0.96–1.73×
cuBLASLt-fresh), decode B1 1.01–1.04×; the H100's 1 × 4096 × 4096 now 64 vector partitions (intents only). TMA as
capability geometry (`Caps::bulk`: granule, box dims, swizzle spans, transaction bytes, clusters of 8, multicast;
probed on sm_120: UTMALDG, SYNCS, clusters, multicast); its row-copy staging of the TF32 tile (`bulk-stage`) is exact
but 4.3–14.5× slower, shipped off. Open: tensor-box staging (a host descriptor), the ordered epilogue GEMV's split,
the L2 knee, `n × k` weights.

**open-rules** (`open-rules.md`): the cross-arch open rules against their replay (rounds 7a2b67bf, 60f0f8ce; no
rentals). Bug: a `tile-post` site's candidates were planned under other arms than the emission draws them, so the
4090 block decode's chain formed with no cooperative launch (`site_choice`; and a merged unit's copied-out member,
`hz_copied`) — formed = cooperative launches in 2990 sweep emissions and asserted on generated programs. `calls`: the
bodies' change per round of the persistent grid (one round: none) and the phases' length (22 ‰ of the members' work at
the part's peaks); unmeasured boundaries derived from the 5080's cycles — decode's chain forms again on the 3090 /
4090 (won 1.48 / 1.20), the H100's long-phase chains stay refused: 18 / 20 measured signs. `wide-fma`: FP32 lanes per
SM and, where an SM's DRAM share ≥ 14.6 KB/µs, the wide grid beyond one wave (H100 5 / 5 lines). `row-slices`: derived
walk / latency on every part (the rented parts' lsm rows now slice). Intents: face from the host line, `pdl` by the
module's programmatic flag (H100 89 / 89), `calls-kernel` by the IR's barrier count. Open: `min-blocks`, `warp-fill`,
`grid-share`, the tune trial, the A100's one-round phases; no timed 5080 program changes code (A/B null by
construction).

**flash-attention** (`flash-attention.md`): attention as one pass over key windows from facts — `stream_plan` (roles,
each fold's `fold_combine` law, a rescale's shift the running extreme; `row-slices` reads the same shift test),
`unit_plan`'s extended groups over epilogue and mixed-radix sites (P·V's rows a digit permutation of the scores',
`RowMap`; the KV head's query heads one panel), `fold_combine` over every tile plan, a causal mask's box classes
(`GuardBox`) with the absorb vote and `finite_plan`. The 2048-token prefill's attention: 3 kernels / 3.79 ms → one
streamed kernel / 1.91 ms (scores never stored, masked windows skipped); the layer 8.60 → 6.62 ms ABBA, **0.91× torch SDPA
efficient** (was 1.17); `_tf32` 1.62 → 1.20, block prefill 1.01 → 0.94. Split-KV by a last-block ticket, P1 `bar_ns` 16.9 ns
(`calib.cu bar`). Open: the split's price on L2-resident attention (attn rows stay split by cost; forced `s = 8` is
15 % faster on `attn_1024_64`), decode (narrow cache + captured position), tensor cores, F8.

**regressions** (`regressions.md`): the cross-arch 3a8e654f round's losers. (1) B1 decode: a `tile-post` site's
planned workspace split was dropped by the one-candidate launch (the ordered 16-block kernel, H100 235 vs 29 µs) —
it launches through its table; `stream-fly`'s intent checks post sites, `evidence.py` reads the A100 / H100 SMs from
their profiles. (2) bf16 KV cache on HBM: the scores kernel's 2-byte loads (`align 2`; 256 a thread, 4× the f32
cache's) — a narrow operand carries its allocation's alignment: 32 16-byte loads; the emulator checks every declared
alignment at each launch; RTX 5080 bf16 decode 1.05–1.29× ABBA. (3) `vec-cells`: the DRAM share an SM
(`DRAM_PER_SM_SPREAD`) against the vector grid's spread over the SMs (`vec_cells_pay`, OFF arm `vec-spread`), 11
replay lines, 10 / 12 HBM cells picked. (4) the 4090's GEMVs / `lsm_32_8192`: 1.33× kernel × 15–29 µs of per-call
host time; one fused lsm kernel measured slower (33.7 vs 23.0 µs) — open: the box's per-launch host time.

**f32-tensor-cores** (`f32-tensor-cores.md`): the warp accumulator keeps two guard bits on every format (the member
bound is sound) and the 5080's TF32 rate equals its FFMA rate (256 flops / SM clk; 16-bit 512) — `mapal_ir::split`
(split forms' a-priori coefficient against D2), `split-tile` (narrow1: `{f32}` over two bf16 operands 2.1–2.4× the
FFMA tile, 0.40–0.47× torch f32; tf32x3 taken by the rate rule on the A100 only), `mma-combine` (TF32 / 16-bit tiles
cut along `k` by a wave model: 10 programs 0.77–0.985, mlp_2048_tf32 1.09 → 1.03× the best TF32 leg),
`mma-depth` measured losing (off).

**hopper-wgmma-tma** (`hopper.md`): warp-specialized TF32 wgmma fed by 2 tensor boxes a stage (R1 price on the
measured `tma_copy_ns`: 5080 197 / H100 159 ns), producer-rounded aT, setmaxnreg; H100 SXM: MLP tf32 1.17–1.25×,
m128 0.87–1.21×, m2048 1.01–1.07× of OFF; $1.79.

**fp8-leftovers** (`fp8-leftovers.md`): fp8 storage (f8e4m3 / f8e5m2, saturating, C1 cvt), `_f8w` decode rows (B1
L2048 0.056 ms/step, 0.37× bf16w; blkdec 0.37 ms, 0.57×), the one-candidate split post table, calls-issue (issue
queue), tune-context (cold / event trials: dec B8 L2048 bf16w 0.92×), the `call_keep` fact; open: fp8 MMA tiles,
retain-call's consumer.

**integration** (`integration.md`): regressions + f32-tensor-cores + hopper + fp8-leftovers on one branch. The one-candidate
epilogue split fixed twice became one rung (`post-table`, OFF arm the emission before; `--cands=1` keeps the plan);
narrow1 takes a format only where its warp product has the 16-bit geometry (fp8 × fp8 reached it); one matrix rate
(P1) instead of two. The full GPU suite found two regressions no step's changed-programs run could see: hopper's
per-kernel tensor-map lookup (+20–30 µs a program; now read once from the PTX at load) and the trial's cold flag at the
L2's edge (pj_m16_n4096_k4096 1.74×; measured `l2_keep_bytes`, the cyclic working set the L2 keeps: 1.025× on the 5080).
5080 column 1.076 → 1.065 (all 120 rows), 25/21/74 → 23/29/68; every step's row reproduces; session-controlled ABBA
against the 3a8e654f binaries 0.92–1.055. Open: a ~1 µs runtime residual, the tune's pick variance (pre_t8192).

**sweep-attention** (`sweep-attention.md`): `llm_pre_t8192` (f32) 89.55 → **48.89 ms**, 0.92× Liger (was 1.69×): each
KV-head group's attention is one streamed kernel (`unit-uniform`: a tile whose operand address a loop counter moves
joins the stream unit, the launch moves the pointer; `TilePost::after`), the stream kernel's blocks per SM declared
from its geometry (`nvvm.minctasm`: 168 registers, no spill). Three fact bugs the generated head-group loops found
(`d = T` taken for the producer, one-head row reads proved by digit forms, a carried sum after the consumer).
OFF arm `--gpu-off=unit-uniform`. Open: bf16-weight prefill (`Bf16x3n` unemitted), tensor cores in the stream kernel,
decode attention (captured position, batch-moved K window, bf16 cache windows), the tune's pick variance.

**sweep-gemm** (`sweep-gemm.md`): round-store — an f32 array read only by bf16 products is stored rounded by its
producer (`precision_plan.store`, `narrow_stored` + the site-plan guard; OFF arm `--gpu-off=round-store`); 25 `_bf16`
programs: m128 bf16 1.11–1.18×, m2048 n6144 1.15×, m1 GEMVs 1.16–1.19×, LLM bf16 rows 1.01–1.14×, m128 n28672 0.97×
(panel depth, open); sm_120's `FHFMA.BF16` counted as the FFMA it is (92 false mismatches). Open: the m128 n28672
panel depth, the m1 GEMVs' per-call zero, m128 TF32.

**sweep-runtime** (`sweep-runtime.md`): tests 3–4× faster per emulated binary (the emulator compiled once, lld) and
run side by side (`benches/local/test_par.py`); site_geometry's loops in parallel; the runtime alloc check's false
mismatch fixed (282 → 0); bundles no longer stop on a stale `mapal` CLI. Prefill peak: T2048 0.42×, block prefill
0.74× torch since flash-attention. Epilogue groups forfeit stream-K on 2048 sites (fan / MLP rows 4–9 %): measured,
the priced rule replayed 9 / 13, open (stream-K epilogue or the plan model's refit).

**integration — sweep merge** (`integration.md` § Sweep merge, `open-problems.md`): the three sweeps on one branch
(one conflict: both new rungs in the rung table), every timed program's emission the superposition of its step's
change, 1524 tests (`test_par.py`: 16 min, was 45), the full suite + model scale + every competitor leg in one
session. 5080 column 1.065 → **1.021** (28 / 31 / 61); `llm_pre_t8192` 47.9 ms, 0.90× Liger, peak 0.41× torch.
Found: `test_par.py`'s shared log names, and the timed scripts' busy check on the wrong CPUs — the 9970X's CCD 0
cores have the highest clock ceiling, and host-bound programs run 0.81–1.00× as long there as on CCD 1 (integration 1
timed on CCD 1): `timed_cpus.sh` records the set. Open: the tune's pick variance (a third row), the f8w B1 decode's
session spread (0.89 / 1.15 ms on equal code).

**sweep-attention-2** (`sweep-attention-2.md`): the streamed attention kernel folds its products on
`mma.sync.m16n8k16` where every product admits bf16 (`stream-mma`: `stream::mma_fmt`, `TileSite::half` ×
`Caps::matrix_op`, priced by `stream_ns` on `mma_half_per_clk`) — `llm_pre_t2048_bf16` 3.77 → **2.65 ms** (1.95× →
1.37× torch-bf16-compile-cg), `llm_blkpre_t2048_bf16` 10.74 → 9.61 (1.34× → 1.20×). The decode rows' scores kernel
copies its keys along their rows (`kmajor-b`: a register-staged tile's k-major `b`) — `llm_dec_b32_l8192_bf16kv` 2.43 →
**1.68 ms** a step (1.60× → 1.10× torch-bf16-cudnn), B8 1.51× → 1.05×, B1 0.96× → 0.85×, f32 B32 L8192 0.37× → 0.29×;
the explicit-cache twin B32 L2048 69.9 → 49.2 ms (0.65× torch's HF cache). OFF arms `--gpu-off=stream-mma`,
`--gpu-off=kmajor-b`. Open: bf16-weight prefill (A100 / H100 rates), TF32 stream, decode as one stream, small programs,
the naive-cache demand fact; stream-mma at 40 % of the bf16 peak (f32-staged windows, no `ldmatrix`).

**sweep-gemm-2** (`sweep-gemm-2.md`): warm-zero — the driver's lazy memset init (13–17 µs a process's first ≥ 8 KiB
fill) moved before the clock: one-call atomic-split programs 0.77–0.98× (pj_m1 GEMVs 0.77–0.81×); streamk-price — an
under-filled site's stream-K priced by the part's FMA tile / `red` / zero rates (site-level wave quantization):
loopmm 0.94×, rt_mlp2_1024 0.94×, smlp 0.96×. OFF arms `--gpu-off=warm-zero`, `--gpu-off=streamk-price`. Open: the m1
GEMV L2-edge geometry, TF32 per-fragment cvt (a TF32-exact storage fact), the 3090 / A100 / H100 `red` rate.

**sweep-runtime-2** (`sweep-runtime-2.md`): an epilogue group keeps its site's stream-K walk (`streamk-epi`: whole
tiles stored, shared tiles through a ticket) — rt_mlp2_2048 0.906, fanout 0.937, mlp2 0.940, mlp 0.943, loop_4096
0.951, indep 0.954, diamond 0.964, epi_4096 0.965 (mlp_2048, rt_mlp2_2048 and loop_4096 ahead of their fastest
competitor); the tune's trial picks confirmed in their program (`confirm_pick.py`, OFF `CONFIRM=0`); an ordered launch
behind a stream-K plan fixed; the GPU's P-state and clocks recorded through every timed run (`gpu-clocks.txt`). OFF
arm `--gpu-off=streamk-epi`. Open: the f8w B1 decode's session spread (clocks now recorded), the ~1 µs residual.

**integration — second sweep merge** (`integration.md` § Second sweep merge, `open-problems.md`): the three second
sweeps on one branch; two interactions found by comparing each program's merged emission with each step's and handled
(a group at a priced stream-K site takes the ticket: generated plans reach it, `rt_mlp2_1024` 0.982 ON/OFF;
`warm-zero` now counts the ticket's workspace zero), a stale `horizontal` expectation, five evidence-checker
misreadings of the new kernels (the hot loop of a stream-K piece loop, an epilogue group's order, one unroll count
among several, table-launched groups, `kmajor-b`'s 8-byte bf16 vectors: bS 321 → 229, bM 5561 → 5481). 1531 tests;
the full suite + model scale + every competitor leg in one session, the GPU's clocks now recorded by the model-scale
script too. 5080 column 1.021 → **0.973** (30 / 38 / 52); equal-code programs 0.990. Open: `kmajor-b`'s per-row
64-bit addresses and B1 prefetch cover, the f8w B1 decode's slow state (0.848 ms here at P0 / P1).

**review-fixes-2** (`review-fixes-2.md`): the second sweep's review. `round-store` no longer stores an array that
escapes its function (a kept callee returned `[bf16; n]` to an f32 caller); found with it, an epilogue group's output
that escapes through a return was never stored (the device left it zero). The priced rungs no longer fall back where a
rate is unmeasured: `Gpu::plan_cost` derives each missing plan rate from the 5080's at the part's properties
(`streamk-price`, the `stream` rule on the 3090 / A100 / H100), and the first cooperative launch is priced — derived
where unmeasured, shared by a counted loop's trips (`trip_split`), not free in any loop (calls-issue's two 5080
programs had lost 1.77× / 1.48× to it; refused now). 13 replay lines (243: 140 / 72 / 10 / 21). The streamk-epi table
is 5 reps (not 8); round-store's m128 n28672 loss re-measured, 12 reps: OFF 0.3846 / ON 0.3996. 5080 text unchanged
on all 199 programs; 18–50 programs move on each rented part (unmeasured: next rented round).
