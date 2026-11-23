# sweep-attention-2 — the attention area's open problems, second sweep (2026-11-24)

Branch `sprint/sweep-attention-2` from `local-rtx-5080` @ `4c29387e` (integration 2). RTX 5080 (sm_120, 84 SMs, 99 KiB
opt-in smem, 64 MiB L2), CUDA 13, clang/LLVM 23. Timed runs: changed-programs ABBA (`gpu_changed.sh`), one `flock` hold
each, `taskset -c 8-15,40-47`, `nice -n 5`, Tctl at start in `conditions.txt`. Raw output (bundles, ABBA results,
test logs): `.claude/raw-evidence/local-rtx-5080/sweep-attention-2/` (not in git).

**Result.** Two items closed generically, each its own commit with an OFF arm, generated-program tests and a
changed-programs ABBA:

- `stream-mma` (`3b6b6d11`): the streamed attention kernel folds its products on `mma.sync.m16n8k16` where they
  admit the `bf16` member — `llm_pre_t2048_bf16` 3.775 → **2.645 ms** (1.95× → **1.37×** the fastest competitor),
  `llm_blkpre_t2048_bf16` 10.74 → 9.61 (1.34× → 1.20×).
- `kmajor-b` (`3bf53861`): a register-staged tile copies a k-major `b` (a decode step's keys) along its rows — the
  decode scores kernel 1.515 → 0.780 ms a step at B32 L8192 (bf16 cache); the 36 decode programs 1.01–1.45× faster
  (1.10–1.45× at B ≥ 8 or L8192), the 12 KV-cache twins 1.02–1.45×, the six block decodes within null; e.g.
  `llm_dec_b32_l8192_bf16kv` 1.60× → **1.10×** torch-bf16-cudnn, `llm_dec_b8_l8192_bf16kv` 1.51× → **1.05×**,
  `llm_dec_b1_l8192_bf16kv` 0.96× → **0.85×**, the f32 B32 L8192 decode 0.37× → **0.29×**; no row slower beyond its
  null.

## The area's items, ranked by cost, and what happened to each

Cost = time lost to the fastest same-precision competitor over the five parts (`matrix.md`; `sweep-attention.md`'s
ranking, its items re-read against this sweep's findings).

| # | item | rows | status |
| --- | --- | --- | --- |
| 1 | bf16-weight prefill: f32 activations × bf16 weights on FFMA (`Bf16x3n` unemitted) | `_bf16w` prefill / block prefill (3.3–14.5×) | attempted-open (§4): the rate property is right to refuse it on every GeForce part; the A100 / H100 rows need their plan rates first |
| 2 | decode attention over the cache (the decode gap is the cache read) | the 42 decode / block-decode programs (`_bf16kv`, `_bf16`, `_bf16w`, f32, `_f8w`, `_pv`), the 12 KV-cache twins | **closed generic** for the cache read (`kmajor-b`, §2): the scores kernel's k-major copies |
| 3 | `: bf16 \| f32` prefill attention streams on FFMA (tensor cores inside the streamed kernel) | `llm_pre_t2048_bf16`, `llm_blkpre_t2048_bf16` | **closed generic** (`stream-mma`, §1) |
| 4 | TF32 prefill attention on FFMA | `_tf32` prefill / block prefill | attempted-open (§5): no part in the set has a TF32 rate above FFMA where the stream is priced |
| 5 | decode attention as a stream (captured position, batch-moved window, narrow windows) | decode rows | attempted-open (§6): §2 removes the larger cost it was meant to attack |
| 6 | small programs: lsm (3 launches), `attn_1024_64` (split under-priced) | 6 rows | attempted-open (§7) |
| 7 | naive-cache twins vs the HF-cache program (1.19–1.54× on the 5080) | `llm_kvn_*` | open (§8): the demand fact; not reached |
| 8 | the tune's pick between sessions | — | the runtime sweep's (not this area's) |

## 1. `stream-mma`: tensor cores inside the streamed attention kernel (closed generic)

**What was missing.** A stream unit (`stream_plan`: `q·Kᵀ → softmax → P·V` as one pass over key windows) folded both
products by FFMA in a lane layout (`rt` rows a thread, columns `x + g·z`). The `: bf16 | f32` programs' products admit
the `bf16` member, which the projections already fold on `mma.sync.m16n8k16` (`half-tile`); the streamed kernel
did not, so the 2048-token prefill's attention ran at 18 TFLOPS on FFMA (1.89 ms of 3.77).

**The capability, as geometry.** `stream::mma_fmt` reads each producer's and consumer's `TileSite::half` (the
16-bit member its product admits, `precision_plan`) and the part's matrix capability (`Caps::matrix_op`: a warp
product of that format, `m16n8k16`, its accumulator meeting the member's `u_a = 2⁻²³`, a rate the part states:
`mma_half_per_clk`). Where every product admits one format and the geometry divides (`d` whole 16-element
`k`-steps, `c` whole 8-column tiles, 16-row warps, 16-key windows), the rule (`stream_choice`) prices the
matrix-product geometries beside the FFMA ones on the same plan model (`stream_ns`: the products' flops at the
matrix rate and the shipped tile's fraction of peak, the fragment loads' wavefronts and LSU issue, the packs) and
takes the cheapest. No version or part name is read.

**The emission** (`func/stream.rs`). The threads take the accumulator's fragment layout: `rt = 2`, `g = 4`, warp `w`'s
lane `l` holds rows `16w + l%32÷4` and `+8`, window columns `8·(z÷2) + 2·(l%4) + z%2`, the consumer's columns
likewise — so the cells (mask, scale, `exp`), the row folds (a quad butterfly, `g = 4`) and the rescales run
unchanged over the new coordinates. The producer's row operand (`q`) is loaded once into its `a` fragments
(registers; no shared-memory panel), each window's `b` fragments come from the key window (two values rounded once
into a register, `cvt.rn.bf16x2.f32`); the producer's cells in the accumulator layout are already the consumer's `a`
fragments (FlashAttention-2's register pass), so the row window of `P` in shared memory is gone; the value window
gives the consumer's `b` fragments. Window rows are padded so the fragment loads hit distinct banks (`d + 8`,
`c + 4`). OFF arm: `--gpu-off=stream-mma` (the FFMA stream, the emission before).

**Intended → realized** (`llm_pre_t2048_bf16`, `b32xkb16:rt2g4t64:mma:cp`, `evidence.tsv`, intent `stream-mma`):

| what | intent | realized |
| --- | --- | --- |
| instruction mix | HMMA for both products, the operand pairs packed in registers | 32 `HMMA.16816.F32.BF16`, 100 `F2FP` a kernel; `mma.sync` in PTX |
| data movement | K / V windows by `cp.async`, nothing axis-wide stored, `q` read once a block | `LDGSTS.128`; `st.global` only for `attf`; no `q` panel (smem 17.2 KB, was 76.3 KB) |
| resources | no spill | 163 registers, 0 spill |
| model vs measured | — | 0.611 ms priced, 0.758 ms measured (1.24×); the FFMA stream's 1.443 / 1.893 |
| checks | `stream-mma` ON / OFF | ok on all four stems (the ON pair after the plan line names the rung) |

**Measurements** (RTX 5080, `gpu_changed.sh --off stream-mma`, `PROBE_REPS=9`, Tctl 65 °C, value gate 0 failed):

| workload | OFF ms | ON ms | null ms | fastest competitor (leg) | Mapal/best before → now | explanation |
| --- | ---: | ---: | ---: | --- | --- | --- |
| llm_pre_t2048_bf16 | 3.775 | **2.645** | 2.643 | 1.931 (torch-bf16-compile-cg) | 1.95 → **1.37** | the stream kernel 1.893 → 0.758 ms (45 TFLOPS); the projections unchanged (0.94 + 0.81 ms) |
| llm_blkpre_t2048_bf16 | 10.742 | **9.611** | 9.610 | 8.036 (torch-bf16-compile-cg) | 1.34 → **1.20** | the same kernel, one a block |

Both rows still lose: the attention kernel reaches 40 % of the part's bf16 peak (FlashAttention-2 ~60–70 %): its
windows are f32 in shared memory (the `b` fragments read by four 4-byte loads and two packs, no `ldmatrix`), and the
geometry the model picks (32-row blocks, 16-key windows) re-reads each K / V window from L2 per 32 rows.

**Replay on the other parts** (the rule over each profile): the 4090 takes it (`b64xkb32:rt2g4t128:mma:cp`, priced
0.47 ms against the split's 3.81), the 5090 takes it (`b32xkb16`); the 3090, A100 and H100 keep `stream=s:unpriced`
(no plan rates: flash-attention open 5) — their prefill emission is unchanged.

**Scope** (`suite_diff.py --off stream-mma` over every `gen_tiles.py` program): only `pt_llm_pre_t2048_bf16` and
`pt_llm_blkpre_t2048_bf16` change code.

**Tests.** `flash_attention.rs` `stream_mma_on_generated_attention`: 48 generated programs
(`testgen::flash_half_strategy`: 16–64 tokens, 1–4 heads of 16 / 32 in groups of 1–4, causal or not with the mask
`−1e9` / `−8`, the head-group loop or not, products `: bf16 | f32`; data exact in `bf16`, so the probabilities'
rounding is the member's whole error), each forced onto the matrix product, its OFF arm and split-KV where the
windows divide: 250 emulator runs in four modes, every value within the member's bound. Counters: 48 streamed on the
matrix product, 24 causal, 18 with skipped windows, 19 loops, 27 at `d = 32`, 24 grouped, 21 with 32-key windows, 6
with tails, 10 split-KV, 48 OFF arms on FFMA; the rule's own pick (512 tokens, 8 heads of 64) is the matrix product,
OFF the FFMA stream. The whole `mapal-backend-llvm`, `mapal-ir`, `mapal-rewrite` suites on this commit's code: 77
binaries, 821 passed (the one failure, a first version of the rule-pick counter at sizes the rule leaves split, is
the fixed-size check above: `flash_attention` 5 / 5 on the final code).

## 2. `kmajor-b`: the decode step's keys copied along their rows (closed generic)

**What was missing.** The decode scores `s[b][h][r] = q[b][h]·K[b][r][h÷G]` are a mixed-radix tile site (its `b`
panel shared by a KV head's `G` query heads: `head-group`) whose `b` is k-major: a key's 128 elements are
contiguous, its lanes (the keys) 1024 apart. The register-staged FMA tile copied a `b` panel one element a copy
down a column (`LaneMap::RowMajor { per_row: bn, unit: 1 }` over `(k, n)`): each thread of a warp a different key's
row, one 2-byte (bf16) or 4-byte load a thread, a memory request a row. At B32 L8192 the bf16 scores kernel read its
512 MiB cache at 350 GB/s (1.515 ms a step, 2.7× the DRAM time) while the values kernel (`P·V`, `b` along its lanes)
read the same bytes at 830 GB/s; the f32 scores at 540 GB/s.

**The realization** (`func/gpu.rs` `kvec_b`, rung `kmajor-b`). Where `b` is unit-stride along `k` and not along its
lanes (`b_k_major`), every other term of its address a whole vector (a mixed-radix `b`: one unit `k` digit whose radix
the vector divides, its row and lane digits whole vectors) and `bk` whole vectors, the copies walk each key's row —
`LaneMap::RowMajor { per_row: bk / lanes, unit: lanes }` over `(n, k)` — a vector of `k` a copy (16 bytes of f32, 8 of
bf16, widened in registers as before), scattered into the `[k][n]` panel the fragments read (as a transposed `a`
panel already is). The fragment reads are unchanged. `lsu_per_panel` and `panel_excess` count the scattered stores;
the plan line says `kvb=1`. It is a realization, not a priced choice: every part takes it where the layout allows (the
replay: the 4090, 3090, A100, H100 and 5090 emissions of the B32 L8192 decode carry `kvb=1` on all three candidates).
`launch_key` drops the field, so every candidate keeps its id (its arms differ in text, as candidate 0's always
could). OFF arm: `--gpu-off=kmajor-b`.

**Intended → realized** (`llm_dec_b32_l8192_bf16kv`'s scores kernel, `evidence.tsv` intent `kmajor-b`):

| what | intent | realized |
| --- | --- | --- |
| data movement | each key's row read coalesced: a vector a copy | `LDG.E.64.CONSTANT` (8 B of bf16) for the `b` copies, scalar `STS` scatter; `ld.global.v` in PTX |
| per call | the cache read at the DRAM rate | 0.780 ms a step for 512 MiB (690 GB/s, 72 % of peak; was 1.515) |
| checks | ON: a vector global load, no spill; OFF: no vector copy | ON 623 / 623 ok (one no-spill mismatch: an `attn_2048_64` candidate at ptxas' 128-register cap, 24 bytes spilled ON, 20 OFF); OFF 1379 / 1379 ok |
| the bundle's evidence | no new mismatch class | 146 289 checks, 5 896 mismatches, every class a standing one (`kmajor-a`, `full-tile`, `face-registers`, `unroll`, `prefetch`, `fill-wave`, `grid-share`, …) on both arms; `arch-rungs` 150 as built, + 300 on the `.c1` / `.off-kmajor-b` arms only after the plan lines were re-read with the final kernel rung list (a post-hoc patch of a bundle built before it; a fresh bundle prints them) |

**Measurements** (RTX 5080, `gpu_changed.sh --off kmajor-b` over the 59 programs it changes, one `flock` hold,
`PROBE_REPS=5` ABBA, Tctl 84 °C at start with the other sweeps loading the host (load 55); value gate 0 failed;
decode rows ms a step = a run's 16 steps ÷ 16; the competitor is the matrix's fastest same-precision leg):

| workload | OFF | ON | fastest competitor (leg) | Mapal/best (matrix → now) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| llm_dec_b32_l8192_bf16kv | 2.425 | **1.676** | 1.521 (torch-bf16-cudnn) | 1.60 → **1.10** | scores kernel 1.515 → 0.780 ms a step; `P·V` 0.645 unchanged |
| llm_dec_b32_l8192_bf16 | 2.454 | **1.689** | 1.521 (torch-bf16-cudnn) | 1.60 → **1.11** | the same kernel |
| llm_dec_b32_l8192 (f32) | 3.797 | **2.816** | 9.739 (torch-compile-cg) | 0.37 → **0.29** | f32 scores: 16-byte copies |
| llm_dec_b32_l8192_bf16w | 3.716 | **2.724** | 1.521 (torch-bf16-cudnn) | 2.31 → **1.79** | f32 cache (twice torch's bytes) |
| llm_dec_b32_l2048_bf16kv | 0.6551 | **0.5032** | 0.465 (torch-bf16-compile) | 1.40 → **1.08** | |
| llm_dec_b32_l2048 (f32) | 1.097 | **0.8625** | 2.302 (torch-compile) | 0.48 → **0.37** | |
| llm_dec_b8_l8192_bf16kv | 0.6979 | **0.5117** | 0.4883 (torch-bf16) | 1.51 → **1.05** | |
| llm_dec_b8_l8192 (f32) | 1.109 | **0.8712** | 2.779 (torch-compile-cg) | 0.38 → **0.31** | |
| llm_dec_b8_l2048_bf16kv | 0.2455 | **0.2232** | 0.2285 (torch-bf16-compile) | 1.05 → **0.98** | |
| llm_dec_b1_l8192_bf16kv | 0.2264 | **0.1766** | 0.2085 (torch-bf16-compile-cg) | 0.96 → **0.85** | |
| llm_dec_b1_l8192_bf16w | 0.248 | **0.2095** | 0.2085 (torch-bf16-compile-cg) | 1.16 → **1.00** | |
| llm_dec_b1_l2048_bf16kv | 0.1411 | 0.1392 | 0.1521 (torch-bf16-compile-cg) | 0.92 → 0.92 | the step is the projections' |
| llm_blkdec_b1_l4096_bf16kv | 0.6499 | 0.6444 | 0.5576 (torch-bf16-compile) | 1.10 → 1.16 | within null (this session's blkdec null 3–8 %) |
| attn_2048_128 | 0.1756 | 0.1187 | 0.0983 (torch-sdpa-compile) | 1.21 → 1.21 | the tuned pick: ON's best candidate copies its keys by vector; OFF's tune picked another (0.167 rules) |
| attn_1024_64 | 0.0411 | 0.0375 | 0.0287 (torch-compile-cg) | 1.51 → 1.31 | |
| pt_llm_kvx_b32_l2048 (run ms) | 69.90 | **49.21** | 75.86 (torch HF-cache f32, `caching-memory.md`) | this session 0.92 → **0.65** | the explicit-cache decode; `caching-memory.md` measured 50.47 for it earlier — the shipped code times 69.9 here (its rules pick 66.4), ON is back at that level |
| pt_llm_kvn_b8_l8192 (run ms) | 151.5 | **108.1** | 78.60 (torch HF-cache f32) | this session 1.93 → **1.38** | the naive twin (earlier 110.1); its first trip is §8's |

All 59: OFF/ON 0.97–1.45; the four below 1.00 (`blkdec_bf16` 0.978, `blkdec_f8w` 0.968, `blkdec_pv` 0.997,
`prefill_1024_512_256` 0.974) sit inside their own null/ON (0.953, 0.920, 0.982, 0.970) — host noise of this
session, not the change. Full table: `raw-evidence/.../sweep-attention-2/kb1/abba.txt`.

**Scope** (`suite_diff.py`, every `gen_tiles.py` program): 59 programs change under `kmajor-b` — the four `attn_*`,
every decode / block-decode (f32, `_bf16`, `_bf16kv`, `_bf16w`, `_f8w`, `_pv`), the 12 KV-cache twins and
`prefill_1024_512_256`. Against `4c29387e`, the branch's emitter changes exactly these 59 plus §1's two.

**Tests.** `flash_attention.rs` `kmajor_b_on_generated_decode`: 40 generated decode-score programs
(`testgen::kdec_strategy`: 1–4 sequences, 4–8 query heads in groups of 1–4, 64–512 keys of 16–128 elements, f32 or
bf16 cache, a run-time position masking the rest) — ON and OFF through the emulator, every mode on ON, values exact
against the interpreter (107 runs). Counters: 61 vector-copied tiles in 9 programs, 7 bf16, 2 f32, 9 grouped, 3 at
`d = 128`. The `mapal-backend-llvm` suite: 49 binaries, 473 passed (the candidate-0 golden takes `kmajor-b` off like
every rung after it, `site_choice.rs`); `--lib` 121, `golden_ll`, `tile_sites_pin`, `plan_cost`, `perf_baseline` on
the final code.

## 4. bf16-weight prefill (attempted-open)

`_bf16w` multiplies f32 activations by bf16 weights at f32 precision; its legs are bf16 throughout (activations
rounded). The exact tensor-core realization is `Bf16x3n` (three bf16 parts of each f32 activation, three exact
products). Its rate condition reads the part's properties: on the 5080 three `m16n8k16` products at 512 flops/clk/SM
cost 1.5 FFMA-equivalents (256) per useful flop, so FFMA is the faster exact realization — the refusal is right on
every GeForce part (the 3090 / 4090 run their f32-accumulating bf16 product at half rate). On the A100 / H100 the
rule would take it, but those parts' prefill sites are priced only after a rented round measures their plan rates
(flash-attention open 5), and no part here could time the kernel. Not built: there is no measurement that could close
it in this sweep. What would: the rented round (`calib.cu bar` + the plan rates), then the emission (f32 `a`
fragments cut into three bf16 parts against `narrow1`'s `ldmatrix` `b` panel, `tf32x3`'s cold guard).

## 5. TF32 attention on the matrix product (attempted-open)

`stream-mma`'s layout carries to TF32 except one step: the `m16n8k8` TF32 `a` fragment pairs columns `t`, `t+4`,
while the accumulator holds `2t`, `2t+1` — `P` would need a quad permutation (shuffles) before `P·V`. The rate
property decides whether it pays: TF32 equals FFMA on the 5080 and 4090 (256 flops/clk/SM both), and the parts where
it is 8× (A100, H100) have no plan rates for the stream. So no part in the set would take it; it was not built.

## 6. Decode attention as a stream (attempted-open)

Unchanged: the three gaps `sweep-attention.md` §5 names (a captured position after the fold, a batch digit in the
window operand, narrow windows). The data movement a stream saves is the scores and probabilities (32 MB a step at
B32 L8192), 7–15 % of the step; §2 attacks the larger cost, the cache read itself.

## 7. Small programs (attempted-open)

- **`lsm_32_8192`** (17.5 vs 10.2 µs): the device runs 9.9 µs (the scores tile 1.9, the sliced softmax 2 × 3.06); the
  rest is the host's per-launch and synchronize time, which no calibration measures yet (`regressions.md` §4: the
  missing property is the host time a timed call pays per dependent launch). The row-block unit that would make it one
  kernel is refused correctly (32 rows = 32 blocks on 84 SMs; forced: 33.7 against 23.0 µs).
- **`attn_1024_64`** (1.51×): the split is priced 2.2× optimistic by the plan model's small-tile `site_ns`, shared
  by every tile decision (`flash-attention.md` open 1); the stream rule is right given its inputs. Not attempted
  here: a fix there moves every tile decision and belongs to a plan-model step.

## 8. Naive-cache twins (open, not reached)

The missing fact is demand (`caching-memory.md`): which cells of a map its readers read per trip. The naive program's
first trip computes the projection's Q columns for every row although only the delta row's are read (13.1 of the
13.2 ms gap at B8 L2048). It needs a column-window tile site or a demand split of the map; not reached in this
sweep.

## For integration

- **README line (local results):** `llm_pre_t2048_bf16` 3.77 → **2.65 ms** (1.95× → 1.37× torch-bf16-compile-cg),
  `llm_blkpre_t2048_bf16` 10.74 → 9.61 (1.34× → 1.20×): the streamed attention kernel folds its products on
  `mma.sync.m16n8k16` (`stream-mma`).
  The decode rows' scores kernel copies its keys along their rows (`kmajor-b`): `llm_dec_b32_l8192_bf16kv` 2.43 → **1.68
  ms** a step (1.60× → 1.10× torch-bf16-cudnn), `llm_dec_b8_l8192_bf16kv` 1.51× → 1.05×, `llm_dec_b1_l8192_bf16kv` 0.96×
  → 0.85×, f32 B32 L8192 0.37× → 0.29×; the explicit-cache twin B32 L2048 69.9 → 49.2 ms (0.65× torch's HF-cache).
- **CLAUDE.md facts table:** no new IR fact. Backend capability rows (the `Caps` / rules line): `stream::mma_fmt` —
  a stream pass's products on the warp matrix product every one admits (`TileSite::half` × `Caps::matrix_op`, its
  geometry and accumulator), priced by `stream_ns` on `mma_half_per_clk`. `gpu.rs kvec_b` (`kmajor-b`): a register-staged tile's k-major `b` copied
  along its rows — a layout realization over `TileRead` / `RadixAddr` (`b_k_major`), no new fact.
- **OFF arms:** `--gpu-off=stream-mma`. `--gpu-off=kmajor-b` (also off in the candidate-0 golden).
- **Intents:** `stream-mma` (`intent.py`; `evidence.py` `flow.stream_mma`), the `stream` intent's product check
  accepts HMMA. `kmajor-b` (`evidence.py` PTX `ld_global_vec`; `kmajor-b` in the kernel rung list).
- **Tailored-fix registry:** nothing (no tailored fix).
