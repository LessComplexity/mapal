# local-rtx-5080 — caching and memory (2026-11-23)

Machine: Threadripper 9970X + RTX 5080 (sm_120, 84 SMs, 100 KiB smem / SM, 64 MiB L2, 16 GB, ~960 GB/s), CUDA 13,
clang/LLVM 23, PyTorch 2.11.0+cu130, branch `local-rtx-5080` (`3cdb0b9d`, `834e64cb`, `e966e535`). Raw output:
`.claude/raw-evidence/local-rtx-5080/caching-memory/` (`commands.txt`, `resKV*/`, `resMS/`, `trace/`, never in git).

The question: does the compiler find a KV cache in a program that never wrote one, reuse memory as well as a
framework's allocator, and what does each of its caching / memory mechanisms buy in time **and** peak device memory?
Everything below is deduced from dataflow facts — no cache API, no workload-named code.

## 1. Facts and the mechanisms that read them

| fact / rule | what it proves or decides | where | OFF arm |
|---|---|---|---|
| **`unchanged_plan`** → `trip`, `same`, `inputs` (`3cdb0b9d`, stored on the graph, CLAUDE.md row) | per loop-body array the cells that may differ from its previous trip — `write_delta`'s deltas carried through row-local maps (row folds, per-row scales, projections, rotations, windows widened to the neighbour row): `Rows` / `Pieces` / `Top`; values equal on every trip (`same`); each value's input slots (across calls) | `mapal-ir/src/algo/unchanged.rs` | `--gpu-off=unchanged` (fact arm: `write_delta`'s deltas only) |
| fix (this step): a back-edge value whose view carries no seed (`write_delta` keyed the write on an image) takes its own delta, never "unchanged" — it defaulted to `same`, so a map over the new history and everything reading it were claimed equal on every trip | | `unchanged.rs` (test `unchanged_back_edge_value_is_never_same`, `retain_delta`'s scaled decode, `recurrence`) | — |
| fix: a carried write's back-edge value is a trip delta with the derivation off too | | `unchanged.rs` (`834e64cb`) | — |
| **`retain-delta`** (S53) on the derived deltas: a tile site over a changed operand recomputes only the delta rows | first trip whole, then the row | `func/retain.rs` | `--gpu-off=retain-delta,retain-resident` |
| **`retain-pieces`**: one changed row per sequence (`Delta::Pieces`) → the gathered rows as one site into a compact buffer, placed by `mapal_gpu_copy2d` | | `func/retain.rs` | `--gpu-off=retain-pieces` |
| **`retain-cell`**: a device cell whose output changes in a delta launches over the delta after the loop's first trip | | `func/resident.rs` | `--gpu-off=retain-cell` |
| fix (this step): a `Pieces` delta whose first piece starts below 0 but ends inside (a window widened to row −1) was skipped — its cells inside are the delta's too; the kernel clips below 0 as it clips past `n` | | `resident.rs` `emit_cell_retained` | — |
| **`retain-same`** (new): a loop value the same on every trip (a transform of a carried array the loop never writes) is launched (device cell) or uploaded (host map a device site reads) on the loop's first trip only (`; mapal.move … same=N`) | `unchanged_plan.same` | `resident.rs` `emit_cell_once`, `delta_uploads` | `--gpu-off=retain-same` |
| `piece-delta`, `resident`, `carry-take`, `delta-h2d` (existing) | in-place piece writes of a carried cache; device residency of carried state; a carry takes its dying init's buffer; row uploads | `resident.rs` | `--gpu-off=<rung>` |
| **`arena`** + **`cone-span`** (new): a buffer freed only at exit lives in the entry arena from its first writer — a loop's body: the outermost loop's `LoopEnter` position; an epilogue's site, a unit's tile site, a kernel its zero pass rides — placed first-fit among the straight-line members (a member a carry took: to exit), so it takes the bytes of what died before | `last_use_plan` positions, `zero-fold`'s tails, `carry-take`'s takes | `resident.rs` `cone_birth`, `cone_fit` | `--gpu-off=cone-span` (whole function, after the members) |
| **`arena-fit`** (new): the members also placed largest first; the smaller of the two arenas kept (neither order wins every plan) | the members' `(birth, death, bytes)` | `resident.rs` `color_by` | `--gpu-off=arena-fit` (birth order) |
| compute once / loop-invariant | one graph node per value: a shared value is computed once, a value of loop-invariant inputs outside the loop's cone | the IR itself (`emission_plan`, `loop_plan`) | none — no recompute path exists; torch's `naive` legs are the recompute baseline |
| runtime: `MAPAL_GPU_PEAK=1` prints `mapal_gpu_peak` — every `cuMemAlloc` less every `cuMemFree`, the high-water mark (box_run's traced processes) | | `mapal-rt/gpu/mapal_gpu.c` | — |

Torch's peak (`torch_bench.py`): `torch.cuda.max_memory_allocated` over the leg, counted from the bytes the process
held before the leg's inputs were made (the L2-flush buffer and anything earlier legs left are not the leg's), one
process per (program, leg) (`benches/local/torch_peak_proc.sh`). Both sides count inputs (weights, the history), temporaries and
outputs. Time: box_run's `iter ms`, median of 3 processes per arm (ABBA), torch median of 30. Protocol as
`model-scale.md` (bench lock, CPUs 0–7 idle and pinned, no game, Tctl < 85 C at start; `conditions.txt` per run).
Run 2 overlapped a test build that took Tctl to 95 C for a few minutes (its start / end 56 / 69 C); its rows that
run 3 repeats agree within 1.1 %. Standing evidence item, not this step's: `evidence.py`'s runtime check predicts two
allocations where the entry arena makes one (`trace allocs == predicted`, every arena program since `entry-arena`).

## 2. The automatic KV cache at model scale

`gen_tiles.py` `llm_decode_kv`: one Llama-3-8B-like attention half (d 4096, 32 × 128 query heads, 8 KV heads), a
history `hx0` of L positions per sequence, S = 16 decode steps writing positions L − 16 + t, f32.

- **explicit** (`pt_llm_kvx_*`): RMSNorm and the K|V columns of the fused projection over the history, RoPE by
  position — the cache, built once — then each step appends its row to the carried `kc` / `vc` (HF-style).
- **naive** (`pt_llm_kvn_*`): no cache. The loop carries the history itself; each step writes its input into it
  and recomputes RMSNorm, the whole fused projection (Q, K, V for every position) and RoPE over all of it, then
  attends with the current row's q. Nothing in the program says "cache".

Value gates: the interpreter on the `px_llm_kv{x,n}{,_blkdec}_b2_l16` twins (exact face bit for bit, default face
within ADR-0040 D2's bound, also GPU-only: 24 / 24 pass); torch's legs against Mapal's printed values at full size
(every leg `within=yes`).

| shape (B, L) | Mapal explicit ms / MB | Mapal naive ms / MB | torch HF-cache f32 ms / MB (leg) | torch naive ms / MB (leg) | bf16 ref ms / MB | naive: Mapal vs torch naive | naive: Mapal vs torch HF-cache |
|---|---:|---:|---:|---:|---:|---|---|
| b1_l2048 | 4.82 / 230 | 6.40 / 286 | 5.21 / 298 (compile) | 47.2 / 321 (naive-compile) | 2.84 / 189 | time Mapal (0.14×), memory Mapal (0.89×) | time torch (1.23×), memory Mapal (0.96×) |
| b1_l8192 | 9.83 / 425 | 16.18 / 651 | 10.48 / 556 (compile) | 181.3 / 748 (naive-compile) | 5.46 / 410 | time Mapal (0.09×), memory Mapal (0.87×) | time torch (1.54×), memory torch (1.17×) |
| b8_l2048 | 15.36 / 688 | 28.56 / 1140 | 20.47 / 937 (compile) | 361.3 / 1322 (naive-compile) | 8.27 / 729 | time Mapal (0.08×), memory Mapal (0.86×) | time torch (1.40×), memory torch (1.22×) |
| b8_l8192 | 51.45 / 2223 | 110.13 / 4031 | 78.60 / 3244 (compile) | 1435.1 / 4781 (naive-compile) | 27.32 / 2650 | time Mapal (0.08×), memory Mapal (0.84×) | time torch (1.40×), memory torch (1.24×) |
| b32_l2048 | 50.47 / 2220 | 109.83 / 4030 | 75.86 / 3242 (compile) | 1432.6 / 4779 (naive-compile) | 27.05 / 2649 | time Mapal (0.08×), memory Mapal (0.84×) | time torch (1.45×), memory torch (1.24×) |
| blkdec_b1_l4096 | 19.43 / 967 | 23.09 / 1080 | 19.48 / 1035 (compile) | 104.5 / 1130 (naive-compile) | 9.94 / 585 | time Mapal (0.22×), memory Mapal (0.96×) | time torch (1.19×), memory torch (1.04×) |

**Who wins.** Against the same math in torch f32: the explicit-cache program takes 0.65–1.00× torch-compile's
HF-style time at 0.69–0.96× its peak (it wins most at B ≥ 8, where torch-compile re-lays the cache out every step:
`triton_poi_fused_clone_permute` 3.4 + 2.1 ms per B8 L2048 run, CUPTI), and the naive program is **4.5–12.5×
faster than torch running the same naive program at 0.84–0.96× its memory** (torch recomputes the full projection
every step: the cutlass SGEMM 20.9 ms × 16 per B8 L2048 run), and 1.19–1.54× the HF-style torch time at 0.96–1.24×
its memory. The bf16 reference stays ~2× faster than any f32 program (half the bytes); it is reported apart.

**Intent → realized (CUPTI, `trace/`).** The naive program's steps after the first run the explicit program's
kernels at the explicit program's cost: B8 L2048, the gathered-rows QKV site 120.3 us × 15 (explicit's one-row
QKV 121.0 us × 16), QKᵀ 127 vs 143 us, P·V 79.6 vs 80.0 us, O 82.8 vs 78.3 us; `h2d = d2h = 0` per trip on both
(`; mapal.move`), one arena allocation each. The whole difference is the **first trip**: the naive projection over
the history computes all 6144 columns for all 16384 rows — 20.22 ms (825 GFLOP at 40.8 TFLOPS) against the explicit
cache build's 2048 K|V columns, 7.11 ms (275 GFLOP, 38.7 TFLOPS); 13.1 of the 13.2 ms gap. B1 L2048: 2.59 vs
0.90 ms, 1.69 ms against a 1.58 ms gap (the naive steps' other kernels run marginally faster). Without the derivation (`.off-unchanged`) the projection runs whole on every trip:
16 × 29.4 ms (B8 L2048, 16.9× the shipped time), 16 × 2.60 ms (B1, 7.1×).

**Memory.** The naive program holds what it computes: the history, its normalized copy and the full
`[B·L][6144]` projection persist (the retained buffers the delta passes update), against the explicit program's
`[B·L][1024]` K and V. B8 L2048: 1140 MB against 688 MB (explicit) and torch's 1322 MB (naive) /
937 MB (HF-style). **Open — the missing fact is demand:** which cells of a map its readers read per trip.
The naive program reads the projection's Q columns only at the written row; with that fact the first pass would
compute the K|V columns of every row plus the Q columns of the delta row (the explicit build's cost) and store no Q
columns but the row's (the explicit program's memory). That needs a column-window tile site (an output row stride
apart from `c`, in every kernel flavour) or a demand-split of the map — not done in this step.

## 3. Each mechanism's OFF arm: time and peak memory

| workload | arm (OFF) | run | Mapal ms | vs ON | peak MB | vs ON |
|---|---|---|---:|---:|---:|---:|
| pt_causal_1024_256 | arena | 1 | 15.15 | 1.00× | 4 | -0 |
| pt_causal_1024_256 | resident | 1 | 838.82 | 55.34× | 2 | -1 |
| pt_decode_1024_256 | arena | 1 | 8.70 | 1.01× | 4 | -0 |
| pt_decode_1024_256 | resident | 1 | 797.48 | 92.73× | 2 | -1 |
| pt_gen_1024_512_256 | arena | 1 | 7.67 | 1.01× | 4 | -0 |
| pt_gen_1024_512_256 | resident | 1 | 866.19 | 113.57× | 2 | -1 |
| pt_llm_kvn_b1_l2048 | arena | 1 | 6.45 | 1.00× | 284 | -2 |
| pt_llm_kvn_b1_l2048 | resident | 1 | 5115.20 | 796.12× | 176 | -110 |
| pt_llm_kvn_b1_l2048 | retain-cell | 1 | 8.40 | 1.31× | 286 | +0 |
| pt_llm_kvn_b1_l2048 | retain-delta+retain-resident | 1 | 45.47 | 7.08× | 286 | +0 |
| pt_llm_kvn_b8_l2048 | arena | 1 | 28.42 | 0.99× | 1136 | -4 |
| pt_llm_kvn_b8_l2048 | resident | 1 | 20679.78 | 723.06× | 736 | -404 |
| pt_llm_kvn_b8_l2048 | retain-cell | 1 | 58.95 | 2.06× | 1140 | +0 |
| pt_llm_kvn_b8_l2048 | retain-delta+retain-resident | 1 | 479.76 | 16.77× | 1140 | -0 |
| pt_llm_kvn_b8_l8192 | retain-pieces | 1 | 1891.57 | 17.25× | 4031 | -0 |
| pt_llm_kvn_b8_l8192 | unchanged | 1 | 2024.03 | 18.45× | 4031 | -0 |
| pt_llm_kvn_blkdec_b1_l4096 | unchanged | 1 | 113.28 | 4.90× | 1080 | +0 |
| pt_llm_kvx_b1_l2048 | arena | 1 | 5.17 | 1.07× | 225 | -21 |
| pt_llm_kvx_b1_l2048 | resident | 1 | 3620.32 | 751.51× | 144 | -102 |
| pt_llm_kvx_b8_l2048 | carry-take | 1 | 29.28 | 1.90× | 948 | +128 |
| pt_llm_kvx_b8_l2048 | resident | 1 | 16496.25 | 1071.66× | 480 | -340 |
| pt_prefill_1024_512_256 | carry-take | 1 | 7.84 | 1.01× | 13 | +1 |
| pt_prefill_1024_512_256 | resident | 1 | 501.51 | 64.54× | 2 | -10 |
| pt_llm_blkdec_b1_l4096 | arena | 2 | 17.16 | 1.00× | 870 | -2 |
| pt_llm_dec_b1_l2048 | arena | 2 | 3.85 | 1.00× | 180 | -2 |
| pt_llm_dec_b1_l2048 | resident | 2 | 2891.18 | 750.87× | 96 | -86 |
| pt_llm_dec_b8_l2048 | arena | 2 | 6.92 | 1.00× | 304 | -4 |
| pt_llm_dec_b8_l2048 | piece-delta | 2 | 11.77 | 1.70× | 436 | +128 |
| pt_llm_dec_b8_l2048 | resident | 2 | 16588.24 | 2394.20× | 96 | -212 |
| pt_decode_1024_256 | retain-delta+retain-resident | 3 | 50.22 | 5.83× | 4 | +0 |
| pt_gen_1024_512_256 | retain-delta+retain-resident | 3 | 18.25 | 2.37× | 4 | +0 |
| pt_llm_blkpre_t2048 | arena | 3 | 32.48 | 1.19× | 1840 | -192 |
| pt_llm_blkpre_t2048 | arena-fit | 3 | 27.38 | 1.01× | 2161 | +129 |
| pt_llm_kvn_b1_l2048 | unchanged | 3 | 45.64 | 7.13× | 286 | +0 |
| pt_llm_kvn_b8_l2048 | retain-pieces | 3 | 463.35 | 16.22× | 1140 | -0 |
| pt_llm_kvn_b8_l2048 | unchanged | 3 | 482.24 | 16.88× | 1140 | -0 |
| pt_llm_kvx_b1_l2048 | cone-span | 3 | 4.82 | 1.00× | 246 | +17 |
| pt_llm_kvx_b1_l8192 | cone-span | 3 | 9.85 | 1.00× | 491 | +66 |
| pt_llm_kvx_b32_l2048 | cone-span | 3 | 50.66 | 1.00× | 2751 | +531 |
| pt_llm_kvx_b8_l2048 | arena | 3 | 16.88 | 1.10× | 673 | -14 |
| pt_llm_kvx_b8_l2048 | cone-span | 3 | 15.42 | 1.00× | 820 | +133 |
| pt_llm_kvx_b8_l2048 | piece-delta | 3 | 21.85 | 1.42× | 688 | +0 |
| pt_llm_kvx_b8_l8192 | cone-span | 3 | 51.03 | 0.99× | 2751 | +529 |
| pt_llm_kvx_blkdec_b1_l4096 | cone-span | 3 | 19.46 | 1.00× | 1000 | +33 |
| pt_llm_pre_t2048 | arena | 3 | 12.17 | 1.42× | 1168 | -128 |
| pt_llm_pre_t2048 | arena-fit | 3 | 8.62 | 1.01× | 1425 | +129 |
| pt_llm_pre_t8192 | arena | 3 | 83.68 | 1.02× | 2736 | -116 |
| pt_llm_pre_t8192 | arena-fit | 3 | 81.93 | 1.00× | 2852 | +0 |
| pt_llm_pre_t8192 | cone-span | 3 | 82.27 | 1.00× | 3028 | +176 |
| pt_prefill_1024_512_256 | arena | 3 | 7.98 | 1.03× | 7 | -2 |
| pt_prefill_1024_512_256 | arena-fit | 3 | 7.72 | 0.99× | 10 | +1 |
| pt_prefill_1024_512_256 | cone-span | 3 | 7.73 | 1.00× | 11 | +2 |

Runs: 1 = `834e64cb` (the arena before `cone-span` / `arena-fit`), the KV twins and the small suite; 2 = the same
build, the other model-scale rows; 3 = this step's emission, the KV twins and the prefills. A row run 3 repeats is
shown from run 3 only. Same-box ABBA, 3 processes an arm; the `null` copy of the shipped build moved ≤ 1 % on every row above 1 ms but
naive B8 L8192 (2.5 %, run 3), up to 8.5 % on the sub-0.1 ms projections: differences inside that are null.

Explanations (CUPTI / the emitted host IR):

- **`unchanged`, `retain-delta` + `retain-resident`, `retain-pieces`** (naive twins, 5–18×): with the derived delta
  gone, the projection over the history is a whole site every trip (CUPTI above). `retain-pieces` off at B > 1: the
  one-changed-row-per-sequence delta has no gathered site, so the whole site runs. Peak unchanged (the retained
  buffers exist either way). The small suite's `decode_1024_256` / `gen_1024_512_256` are naive by construction (K, V
  of the whole history in the loop): 5.83× / 2.37× without retention.
- **`retain-cell`** (1.31× B1, 2.06× B8): RMSNorm's row folds, the normalized history and the RoPE'd K computed
  whole every trip instead of over the delta rows.
- **`piece-delta`** (explicit B8 1.42×, `dec` B8 1.70×): the cache append becomes a whole-cache map every step
  (`Delta::Top`): 2 × 67 MB written per step instead of 8 rows, into next buffers of their own (+128 MB; with
  `cone-span` they take the cache build's dead bytes, +0 MB).
- **`carry-take`** (explicit B8, 1.90×, +128 MB in run 1): the prebuilt `kc0` / `vc0` are copied into separate carry
  buffers instead of becoming them.
- **`resident`** (700–1100×): every trip's operands cross PCIe and the attention runs per launch on host buffers;
  its peak is lower only because the state lives on the host.
- **`arena`** (whole arena vs `cuMemAlloc` at first use / `cuMemFree` at death): 1.03–1.42× slower (allocations in the
  timed region; prefill T2048: 12.2 vs 8.6 ms) but **lower peak in runs 1–2: −147 MB explicit B8, −257 MB
  prefill T2048** — the arena held more than the live maximum. Two causes, both fixed:
  - loop buffers were placed over the whole function after every straight-line member, so they could not take the
    bytes of the cache build's inputs (dead at the loop's entry) → **`cone-span`**: explicit B8 L2048 820 → 688 MB,
    B32 L2048 / B8 L8192 2751 → 2220 MB (−19 %), prefill T8192 3028 → 2852 MB, at equal time;
  - first-fit in birth order stacked a large late member above small early ones → **`arena-fit`**: prefill T2048
    1425 → 1296 MB, block prefill 2161 → 2032 MB.
  The arena still holds more than free-at-death on the prefills (1296 vs 1168 MB): first-fit is not optimal packing
  (open); it keeps one allocation and the time.

## 4. Every other model-scale row and the suite's decode / causal / prefill / gen rows

| workload | Mapal ms | Mapal peak MB | best torch f32 ms (leg) | its peak MB | bf16 ref ms / MB | time | memory |
|---|---:|---:|---:|---:|---:|---|---|
| pt_llm_blkdec_b1_l4096 | 17.140 | 872 | 17.195 (compile) | 930 | 8.84 / 498 | Mapal 1.00× | Mapal 0.94× |
| pt_llm_blkdec_b1_l4096_bf16w | 10.164 | 457 | 17.163 (compile) | 930 | 8.86 / 498 | Mapal 0.59× | Mapal 0.49× |
| pt_llm_dec_b1_l2048 | 3.850 | 182 | 4.065 (compile) | 242 | 2.28 / 157 | Mapal 0.95× | Mapal 0.75× |
| pt_llm_dec_b1_l8192 | 5.411 | 235 | 5.500 (compile) | 292 | 3.12 / 178 | Mapal 0.98× | Mapal 0.80× |
| pt_llm_dec_b32_l2048 | 16.469 | 702 | 36.258 (compile) | 939 | 7.38 / 346 | Mapal 0.45× | Mapal 0.75× |
| pt_llm_dec_b32_l8192 | 56.932 | 2291 | 151.503 (compile) | 3270 | 23.77 / 1234 | Mapal 0.38× | Mapal 0.70× |
| pt_llm_dec_b8_l2048 | 6.929 | 308 | 10.804 (compile) | 356 | 3.44 / 209 | Mapal 0.64× | Mapal 0.87× |
| pt_llm_dec_b8_l8192 | 17.111 | 703 | 39.843 (compile) | 941 | 7.61 / 347 | Mapal 0.43× | Mapal 0.75× |
| pt_pj_m128_n28672_k4096 | 0.804 | 464 | 0.801 (compile) | 528 | 0.32 / 288 | torch 1.00× | Mapal 0.88× |
| pt_pj_m128_n4096_k14336 | 0.452 | 233 | 0.412 (eager) | 250 | 0.17 / 133 | torch 1.10× | Mapal 0.93× |
| pt_pj_m128_n4096_k4096 | 0.141 | 68 | 0.145 (eager) | 84 | 0.05 / 50 | Mapal 0.97× | Mapal 0.81× |
| pt_pj_m128_n6144_k4096 | 0.207 | 101 | 0.188 (eager) | 121 | 0.09 / 71 | torch 1.10× | Mapal 0.83× |
| pt_pj_m16_n28672_k4096 | 0.541 | 450 | 0.543 (compile) | 457 | 0.27 / 232 | Mapal 1.00× | Mapal 0.98× |
| pt_pj_m16_n4096_k14336 | 0.263 | 226 | 0.274 (compile) | 226 | 0.14 / 114 | Mapal 0.96× | Mapal 1.00× |
| pt_pj_m16_n4096_k4096 | 0.052 | 65 | 0.084 (compile) | 66 | 0.05 / 33 | Mapal 0.62× | Mapal 0.99× |
| pt_pj_m16_n6144_k4096 | 0.099 | 97 | 0.121 (compile) | 98 | 0.06 / 50 | Mapal 0.82× | Mapal 0.99× |
| pt_pj_m1_n28672_k4096 | 0.549 | 448 | 0.539 (compile) | 449 | 0.27 / 224 | torch 1.02× | Mapal 1.00× |
| pt_pj_m1_n4096_k14336 | 0.253 | 224 | 0.274 (eager) | 232 | 0.14 / 120 | Mapal 0.92× | Mapal 0.96× |
| pt_pj_m1_n4096_k4096 | 0.032 | 64 | 0.086 (compile) | 128 | 0.05 / 96 | Mapal 0.37× | Mapal 0.50× |
| pt_pj_m1_n6144_k4096 | 0.091 | 96 | 0.123 (eager) | 104 | 0.07 / 48 | Mapal 0.74× | Mapal 0.92× |
| pt_pj_m2048_n28672_k4096 | 11.819 | 704 | 12.352 (eager) | 1608 | 3.95 / 1248 | Mapal 0.96× | Mapal 0.44× |
| pt_pj_m2048_n4096_k14336 | 6.411 | 368 | 6.457 (compile) | 504 | 2.09 / 320 | Mapal 0.99× | Mapal 0.73× |
| pt_pj_m2048_n4096_k4096 | 1.836 | 128 | 1.755 (compile) | 264 | 0.60 / 200 | torch 1.05× | Mapal 0.48× |
| pt_pj_m2048_n6144_k4096 | 2.650 | 176 | 2.595 (eager) | 376 | 0.88 / 288 | torch 1.02× | Mapal 0.47× |
| pt_decode_1024_256 | 8.613 | 4 | 45.366 (compile) | 70 | — | Mapal 0.19× | Mapal 0.05× |
| pt_gen_1024_512_256 | 7.686 | 4 | 26.908 (compile) | 78 | — | Mapal 0.29× | Mapal 0.05× |
| pt_prefill_1024_512_256 | 7.767 | 9 | 26.894 (compile) | 77 | — | Mapal 0.29× | Mapal 0.11× |
| pt_llm_pre_t2048 | 8.569 | 1296 | 8.444 (compile) | 841 | 1.94 / 249 | torch 1.01× | torch 1.54× |
| pt_llm_blkpre_t2048 | 27.216 | 2032 | 27.659 (compile) | 1513 | 7.98 / 601 | Mapal 0.98× | torch 1.34× |
| pt_llm_pre_t8192 | 82.144 | 2852 | 85.040 (compile) | 1916 | 11.48 / 722 | Mapal 0.97× | torch 1.49× |
| pt_causal_1024_256 | 15.159 | 4 | 52.365 (compile) | 70 | — | Mapal 0.29× | Mapal 0.05× |

The explicit-cache decode rows (`pt_llm_dec_*`, a closed-form cache) take 0.38–0.98× torch-compile's time at
0.70–0.87× its peak; the projections match cuBLAS-backed torch in time (0.37–1.10×) at 0.44–1.00× its memory (one
arena, no allocator slack). **The prefills hold 1.34–1.54× torch's peak** (T2048 1296 vs 841 MB): Mapal
materializes the 32-head score array (32 × 2048 × 2048 f32 = 512 MiB; at T8192 per KV-head group) where
torch-compile's fused softmax streams it — the flash-style streaming item `long-prefill.md` leaves open. The small
suite programs run in 4–9 MB (their arena) against torch's 70–78 MB, 3.4–5.3× faster.

## 5. Tests

- `crates/backends/llvm/tests/caching_memory.rs` (10 tests): the fact's deltas, pieces, views, the OFF arm, the
  rows form, constants and calls; the back edge never `same`; `same_every_trip_launches_once` (h2d 0 per trip ON,
  256 B OFF, every emulator mode); the two-sequence history on the emulator; **96 generated programs**
  (`testgen::history_strategy`, extended with prebuilt histories whose transforms die before the loop and small
  row values of different lifetimes), each against the interpreter in all four emulator modes
  (`MAPAL_EMU_SERIAL=0/1/2`, `MAPAL_EMU_REVERSE=1`), the CPU build and one of six OFF arms in rotation. Counters
  (all asserted non-zero but `fit-larger`): programs 96, delta arrays 521, derived facts 96, rows facts 18, piece
  facts 78, same arrays 46, retained 283, retained-cell kernels 233, gathered sites 30, **same cells 46**, **cone
  shared 42**, **fit smaller 4** (fit larger 1: the members' smaller coloring, then the loop's buffers), OFF-arm runs
  32. The generator's new variant found the two fixed bugs (§1).
- `resident.rs` unit test `color_by_size_packs_large_first`; the backend suites (`mapal-backend-llvm`, every test
  file) and `mapal-ir` pass: 49 test binaries, 670 tests, 0 failed (6 ignored); the candidate-0 golden
  (`site_choice`) regolded for the two programs whose emission the fixes and `arena-fit` change (`decode_1024_256`,
  `pt_attn_1024_64` under `zero-fold` off).
- GPU: the px differentials (24 / 24) and the probe value gates of every timed run (0 failed).

## 6. Open

1. **Demand** (§2): the cells a map's readers read per trip, so a naive program's first pass and storage match the
   explicit cache (naive / explicit: 1.19–2.18× time, 1.12–1.82× memory).
2. **The gathered-rows site's rule**: `row-split` reads DRAM bytes per SM only (off on the 5080); the gathered QKV
   rows of B = 8 sequences (24 CTAs on 84 SMs) run 2.6× faster split (118.6 vs 310.9 us per launch, the tune's
   pick; rules-only arm 39.6 vs 28.6 ms). The fill of the gathered grid is the missing property.
3. Arena packing on the prefills (§3) and attention-score streaming (§4).
4. Across calls: `inputs` is computed and tested; no runtime keeps buffers between calls of a function yet, so it
   has no consumer.
