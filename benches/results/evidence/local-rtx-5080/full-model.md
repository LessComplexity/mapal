# local-rtx-5080 — the full model after the prefill integration: readable Llama-3.2-1B-Instruct in Mapal against six inference stacks, both precision classes (2026-11-25, evening)

`local-rtx-5080` (the builds measured here: `8bf54fb3`): the merges of prefill/precision, prefill/residency and prefill/mask-frontend, and
three fixes the integration found. The program is the readable `examples/llama.mapal` (403 lines, 56 of them
comments), unchanged except as mask-frontend wrote it (`attend` reads the cache through the struct). The f32 class
decodes HF f32's 128 tokens at every length; the bf16 class agrees with the bf16 references as far as they agree
with each other (below). Raw output (never in git):
`.claude/raw-evidence/local-rtx-5080/prefill-integrate/` (`mapal/` programs and builds, `time/` the timed runs,
`competitors/` the sweep, `trace/` CUPTI traces, `changed*/` the suite emission diff and timing, `tools/`, logs).

## Method

- **RTX 5080** (sm_120, 84 SMs, 16 GB, ~960 GB/s), Threadripper 9970X (shared; boost capped at 4.5 GHz), driver
  620.73.04, CUDA 13.5. Every timed process held the bench lock and ran on CPUs 16-23,48-55 (`nice 10`); the
  workspace test suite ran at the same time on CPUs 24-31,56-63 (moved there for the timing).
- **Definitions** (the harness's `common.py`): TTFT = submit → first token on the host; decode = (t₁₂₈ − t₂) / 126;
  e2e = t₁₂₈; peak = the process's GPU memory (nvidia-smi). Greedy, batch 1, 128 generated, warmup 2 + 5 reps,
  medians [min, max] in `time/`.
- **Mapal**: `benches/model/run_mapal.py` (one process per repetition; weights load and upload before `() -> time`;
  `tools/time_mapal.sh` holds the lock per length).
  Builds: `MAPAL_GPU_CALIB=benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib mapal build
  --target=native-gpu` (the native-gpu profile is the device query + this box's calibration line; this box has no
  `~/.cache/mapal/gpu-calib`, so the same build without the variable is also reported, "no calib file").
  Programs from `benches/model/variant.py` (`prompt_length=N`; `--compute bf16` for the bf16 class).
- **Competitors**: `~/.cache/mapal/model-compare/compare.py --batches 1 --lens 128,512,2048` (all eight legs, their
  own warmup + 5 reps), 18:06–18:13, load 323 → 36 at start / end (a burst at the start; most
  rows within 10 % of the afternoon's quiet sweep, vLLM at 2048 17 % slower, llama.cpp's 128 / 2048 rows faster).
  Mapal: the table is the pass beside the sweep (`time/final2`, 18:13–18:16, load 36 → 28); the pass before it
  (`time/final`, 18:01–18:04, load 68 → 18) and a third after the test suite (`time/final3`, 18:43–18:47, load 11 → 8) read TTFT up to 22 % and decode up to 17 %
  slower — the decode spread is the host (below), not the load average.
  The f32-compute competitor legs are the precision step's (`compare.py --runner-arg=--dtype --runner-arg=f32`,
  TF32 off; 16:59–17:05, load 5–18; raw `prefill-precision/competitors-f32/`).

**Precision classes.** No gain is claimed across classes.
- **f32 class** — the program as written (Mapal's default face): f32 activations × the bf16 weights,
  f32-accurate within ADR-0040 D2's bound (the projections on the split tile `bf16x2n` where the calibrated rule
  takes it, else the FFMA tile). Compared with the f32-compute legs.
- **bf16 class** — the competitors' precision: `fold: bf16 | f32` on the three sums of products (`project`, the
  scores, `P·V`; `variant.py --compute bf16`): operands rounded to bf16, f32 accumulation (ADR-0041). Norms,
  softmax, RoPE and the cache stay f32. Compared with the bf16 legs.

## Tokens (greedy, 128, batch 1; tokens equal before the first difference)

| program | 128 | 512 | 2048 |
|---|---|---|---|
| Mapal f32 class (calibrated and no calib file) vs HF f32 (`hf_reference.py`, eager attention, TF32 off) | **128** | **128** | **128** |
| Mapal bf16 class vs HF bf16 eager attention (`hf_reference.py`) | 128 | 67 | 61 |
| Mapal bf16 class vs the harness's `hf-eager` (bf16, SDPA) | 49 | **128** | 57 |
| the competitors vs the harness's `hf-eager` | gpt-fast 49, llama.cpp 49, SGLang 109, TRT-LLM 128, vLLM 128 | 32, 67 (bf16) / 32 (f16), 95, 95, 128 | 57 for every tool |

HF's two bf16 runs agree with each other on 49 / 67 / 57 tokens and HF f32 with `hf-eager` on 49 / 32 / 57: past a
near tie every bf16 run follows its own kernels. The bf16-class Mapal run sits inside that spread (first-step top
logit 18.37 against HF f32's 18.40 at 512). The f32-compute legs give 128 / 128 of HF f32 at every length.

## Time (RTX 5080, batch 1, 128 generated, medians)

### f32 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal**, calibrated (projections on `bf16x2n`) | **10.40** | **28.91** | **125.3** | **3.760** | **3.594** | **3.997** | **482** | **2 716 / 2 984 / 4 074** |
| Mapal, no calib file (FFMA tile) | 14.14 | 40.61 | 162.9 | 3.797 | 3.705 | 3.715 | 508 | 3 014 / 3 498 / 4 578 |
| Mapal at 16d668c6 (no calib file; quiet machine) | 28.3 | 85.2 | 332.7 | 3.336 | 3.396 | 3.548 | — | 4 882 / 5 086 / 6 180 |
| gpt-fast f32 (max-autotune, CUDA graphs) | 13.76 | 47.99 | 213.9 | 5.729 | 5.743 | 5.919 | 778 | 6 298 / 6 628 / 7 690 |
| HF eager f32 | 14.41 | 41.75 | 241.9 | 7.163 | 7.478 | 8.160 | 991 | 5 104 / 5 204 / 6 494 |
| HF static cache + compile f32 | 17.13 | 44.35 | 245.5 | 6.771 | 7.552 | 10.494 | 1 010 | 5 266 / 5 372 / 6 874 |

**Mapal leads the f32 class everywhere**: TTFT 1.32 / 1.44 / 1.71× faster than the best f32 tool, decode 1.48–1.60×
faster than gpt-fast f32, end to end 1.6× faster, at less than half its memory. The decode lead is the weights'
bytes (Mapal reads the bf16 bytes and widens in registers; the f32 tools store the widened weights, 4.9 GB a step);
the TTFT lead is the split tile (bf16x2n: two bf16 products of the f32 activation's parts against the exact bf16
weights, within D2's bound, against SGEMM).

### bf16 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`) | 6.95 | 18.12 | 74.05 | 3.528 | 3.673 | 4.012 | 481 | 2 718 / 2 706 / 3 498 |
| gpt-fast (max-autotune, CUDA graphs) | **4.91** | 13.88 | 51.96 | **2.965** | **2.979** | 3.344 | **393** | 3 410 / 3 532 / 4 074 |
| SGLang 0.5.23 | 6.85 | 14.54 | 45.47 | 3.333 | 3.118 | **3.164** | 410 | 8 614 / 8 710 / 8 746 (reserved) |
| TensorRT-LLM 1.3.0 | 10.20 | 14.38 | 45.76 | 3.048 | 3.074 | 3.203 | 405 | 8 666 / 8 670 / 8 744 (reserved) |
| vLLM 0.33.0 | 6.96 | 13.75 | 53.16 | 3.433 | 3.466 | 3.621 | 454 | 9 236 (reserved) |
| llama.cpp bf16 | 5.91 | 14.01 | 49.75 | 3.183 | 3.214 | 3.258 | 422 | 2 804 / 2 846 / 2 896 |
| llama.cpp f16 | 6.07 | **10.91** | **37.04** | 3.183 | 3.230 | 3.262 | 422 | 2 804 / 2 848 / 2 898 |
| HF eager | 6.57 | 14.29 | 47.98 | 4.427 | 4.481 | 4.480 | 584 | 2 726 / 2 780 / 2 962 |
| HF static cache + compile | 8.40 | 15.70 | 49.07 | 3.714 | 4.038 | 5.353 | 529 | 2 822 / 2 872 / 3 090 |

**bf16 class**: TTFT 1.42 / 1.66 / 2.00× behind the best tool (gpt-fast at 128, llama.cpp f16 at 512 and 2048); at
128 Mapal is within 6 % of SGLang / vLLM / HF eager and ahead of TensorRT-LLM and HF static. Decode 1.19 / 1.23 /
1.27× behind the best (gpt-fast, gpt-fast, SGLang), ahead of HF eager and HF static at every length. Peak memory is
the lowest of every tool at 128 and 512 (one copy of the weights since CSE value scope; the f32 cache and
activations put it above llama.cpp and HF at 2048). From 16d668c6 to here, the readable program's TTFT went 28.3 /
85.2 / 332.7 → 10.4 / 28.9 / 125 ms as written and → 6.95 / 18.1 / 74.1 ms in the competitors' precision.

**Decode is bound by the host's launches.** CUPTI shows the step at 3.36 ms of kernels (288 launches, 0.32 ms idle;
without the calibration file 3.27 ms / 323 launches / 0.17 ms). Untraced, one bf16-class run at 512 steps in 3.8–5.4
ms (median 3.98; GPU at P1, 2.75–2.90 GHz): 226 of the 288 kernels run under 4 µs, shorter than the host's issue
cost (`issue_ns` 1 690 ns, ~0.49 ms of host time a step), so the GPU waits on the CPU and the step follows the
cores' state — 3.53–4.30 ms across the three passes for the same build. gpt-fast replays one CUDA graph a step
(2.951 quiet, 2.965 here). The fix is fewer launches (below), not a graph.

## Where each side's time goes (CUPTI, `tools/ktgroup.py`; the prompt pass from the first launch after the weights to the logits copy)

| prompt kernels, ms | 128 | 512 | 2048 |
|---|---:|---:|---:|
| Mapal f32 class, calibrated: kernels (span) | 9.62 (11.1) | 26.89 (29.9) | 122.6 |
| Mapal f32 class, no calib file | — | 35.54 (38.4) | 165.5 |
| Mapal bf16 class | 5.79 (6.9) | 15.56 (18.3) | 69.1 (71.7) |

**bf16 class, 512** (15.56 ms of kernels, 274 launches): gate + up 6.19 ms (32 × 193 µs, 89 Tflop/s — 85 % of the
calibrated bf16 pipe, `mma_half_per_clk` 512 × 84 SMs × 2.45 GHz = 105 Tflop/s), down 3.75 (73 Tflop/s), query +
output 1.91 (69–75), key + value 0.62 (56), the streamed attention 1.47 (16 × 92 µs, `stream-mma`), 143 elementwise
cells 0.99 (norms, RoPE, SiLU·up, residuals), the output head 0.61 (one row against the 525 MB embedding, 866 GB/s).
The projections are within ~6 % of the precision step's cuBLAS bf16 per-layer times (779 vs 734 µs a layer).
**The 2.5 ms between span and kernels is the host**: the embedding gather (`embedding[ids[t]·H + c]`) and layer 0's
RMSNorm run on the host (0.95 ms + 1.51 ms gaps, then 4 MB + 2 MB uploads). **2048** (69.1 ms): gate + up 23.2
(merged, 94 Tflop/s), attention 16.3 (1.02 ms a layer: the whole 2176-key context every row, no causal skip, 36
Tflop/s), down 12.8, query + output 8.2, cells 5.0, key + value 3.0, head 0.6.

**f32 class, 512** (26.89 ms): the projections on `bf16x2n` 23.0 ms (gate + up 11.4, down 7.0, query + output 3.6,
key + value 1.0), attention 2.67 (the FFMA stream: f32 products), cells 0.48, head 0.59; at 2048 the attention is
34.1 of 122.6 ms.

**Decode step** (512, 3.36 ms of kernels, 288 launches): the merged horizontal GEMVs 1.34 ms (46 launches), the
per-row cells and GEMV cells 1.42 ms (224 launches), the output head 0.58 ms. The 2.47 GB of weights at 960 GB/s are
2.57 ms: the kernels run at 76 % of that roofline; gpt-fast's whole step is 2.97 ms (87 %).

**The competitors**: bf16 activations end to end (half Mapal's activation and cache bytes), cuBLAS / CUTLASS /
FlashAttention / FlashInfer kernels (llama.cpp its own; its f16 build the fastest prompt here), causal attention
that skips the masked half, fused norms and SiLU, and CUDA graphs for the decode step (gpt-fast, vLLM, SGLang,
TensorRT-LLM, HF static).

## The integration

- **Merges**: `4420f00c` prefill/precision, `06b708ed` prefill/residency (incl. its once-range commit `67f440c8`;
  `zero.rs` conflict with `151953b5` resolved to `d4dedf0d`'s form — every touching step's launch, a superset;
  both tests kept), `a29a219b` prefill/mask-frontend (rung names 157 / 159 → 160). The CLAUDE.md edits from the
  three "For integration" sections are in the merge commits, plus `docs/components/lower/DESIGN.md` LD29.
- **Found by the integration and fixed** (each with a generated-program test that fails without it):
  1. `bf97ad67` **stream-mma**: the warp-matrix stream loaded its row operand at the group row, not the producer's
     own row numbering (mask-frontend's `out_rows` permutation; the FFMA stream's `stage_panel` maps it). The bf16
     class with attention admitted decoded 0 / 128 tokens (first logit 14.08); after, the table above.
     `flash_half_strategy` now draws the token-major forms (`tm` 1–4): `stream_mma_on_generated_attention` counts
     `mma token-major` 22 / 48, `mma run-time mask` 8.
  2. `934ee80a` **carry-take**: a taken loop init read by a host fold before the loop lost its copy out
     (`bulk.rs` "fold array ptr"): `generic_fusion` and `row_block` generated chains, `site_choice` calibrated
     parts. A taken init's other readers now count (`edge_uses`).
  3. `8bf54fb3` **stream_plan**: where the head width equals the key axis (`d = T`) the output projection after
     `P·V` was taken as a second consumer of the pass (index out of bounds in the kernel) — reached once
     `stream-numbers` streams a mask over a captured number (`placement::once_range_on_generated_prompts`). A
     consumer's row is now a producer's or a cell's.

## Tests

- Full workspace (`cargo test --workspace --release --no-fail-fast`, `RUST_TEST_THREADS=16 MAPAL_PAR=4`,
  `MAPAL_TESTGEN_JOBS=4`) on the merge commit `a29a219b` (`test-full.log`): 154 test binaries, **1700 passed, 7
  failed, 10 ignored**. The 7 failures, in 5 targets, are the three integration faults above and one golden:
  `generic_fusion` (2) and `plan_cost` (2) and `row_block` (1) — carry-take's "fold array ptr"; `placement`
  `once_range_on_generated_prompts` — the `d = T` stream; `site_choice` `candidate_zero_is_the_pre_sprint_emission`
  — 36 golden rows (px_mlp2_64_f64, pt_rq_mlp2_512, pt_smlp_4096_256_1024 × 4 parts × 3 faces) that differ from
  16d668c6 only by precision's `_u<slots>` variant name (host and device text equal with the names, kernel ids and
  metadata ids normalized: the 24 default / exact (part, face) pairs checked): regolded in `716bc739`.
- Re-run at `8bf54fb3` (site_choice after the regold): those five targets (generic_fusion 12/12, placement 2/2, plan_cost 3/3, row_block 9/9,
  site_choice 9/9), the fixes' neighbours (flash_attention 6/6 with the extended `stream_mma_on_generated_attention`,
  zero_fold 5/5, caching_memory 11/11, epilogue 13/13) and `mapal-ir` + `mapal-rewrite` (395 passed); each fix's
  test fails without it (`fix*-*.log`).
- **Suite emission** (`gpu_changed.sh --base 16d668c6`): same = 37, differs = 8 (`pt_indep_512`, `pt_indep_2048`,
  `pt_mlp2_2048`, `pt_mlp2_2048_tf32`, `pt_rt_mlp2_1024`, `pt_rt_mlp2_2048`, `pt_diamond_2048`,
  `pt_smlp_4096_256_1024`); equal at `8bf54fb3` and at the merge. All eight differ only by precision's `_u<slots>`
  variant name (`0d9686a6`): with symbol names and metadata ids normalized, host and device IR are identical — the
  same kernels, null by construction. The ladder run (`LADDER=1 gpu_run.sh`, value gates pass on every arm) times
  them at 0.95–1.12× the best library leg (`changed/results/gap.md`).

## Where Mapal still loses, why, and what is next (generic)

1. **The prompt's host chain** (bf16 class: 2.5 of 18.1 ms at 512, 1.1 of 6.9 at 128). The embedding gather's index
   `ids[t]·H + c` is proven only at run time, so the map is no `cell_plan` cell (`body_is_classifiable`: every
   `Index` proven) and runs on the host, serially; layer 0's RMSNorm follows it there by the byte rule (its operand
   is a host value). Next: a device cell whose index is unproven needs a device trap — a flag the kernel sets and
   the next sync reports as the host's trap would (a runtime fact and a placement rule, not a model rule). The norm's
   half is the residency worktree's in-progress `shared-upload` (an operand another device reader uploads anyway
   adds no bytes).
2. **Attention at 2048** (16.3 of 69 ms, bf16 class; 34 of 123 ms f32 class): every row reads the whole context —
   the causal skip needs the cache proven finite (mask-frontend's open item: a run-time finite check, one reduction
   a launch) — and the `stream-mma` geometry (32-row blocks, 16-key windows, 64 threads) reaches 36 Tflop/s.
3. **Decode**: 288 launches, 3.36 ms of kernels at 76 % of the weights' DRAM roofline; ~200 small per-row kernels
   (residency step: `flow` refuses the decode groups as `phase`; `horizontal` does not price latency-bound GEMVs).
   **Calibrated, the decode step moves layer 0's chain (norm, key / value projections, rotated key) to the host**:
   `device_pays` prices a per-trip cell's device side at the isolated `launch_ns` (11.1 µs: launch + completion)
   where a launch queued behind the trip's device work costs `gap_ns` (260 ns) — +0.08 ms busy and +0.17 ms idle a
   step against the uncalibrated build. Next: price a cell queued behind device work at the stream's dependent-launch
   cost (a `DeviceCost` property already measured), with the suite A/B it needs.
4. **Activation bytes**: the bf16 class still keeps f32 activations between the sums of products (cells 1.0 ms at
   512, 5.0 ms at 2048; `attended` f32 into the output projection) where the competitors keep bf16.
5. **The bf16 class builds slowly** (150–210 s against 10–20 s for the f32 class) — not traced here.

No tailored fixes (`docs/tailored-fixes.md` unchanged).

## The program

`examples/llama.mapal` — top-level sizes (`2048 -> hidden_size;`, prompt length and generated tokens one line each),
`type Matrix<rows, columns> { _: [bf16; rows * columns] }`, per-layer weight structs (`DecoderLayerWeights`) inside a
`Model` read whole (`"model.bin" -> read -> model: Model;`), one key/value cache struct (read inside `attend`'s maps as `cache.keys[…]` since mask-frontend), size-generic blocks
(`rms_norm<tokens>`, one `project<tokens, rows, columns>`, `rotate<tokens, width>`, `attend<tokens>`,
`feed_forward<tokens>`, `decoder_layer<tokens>`), and **one `forward<tokens>`** — a chunk of new tokens through a
layer loop over `model.layers[layer]` — used for the whole prompt (`tokens` = prompt length) and for each generated
token (`tokens` = 1), as inference engines do. `sqrt`, `sin`, `cos`, `exp` are the builtins; the llama3 RoPE
frequencies are a top-level table the compiler computes before the run (ADR-0048: 32 values, embedded). Every
activation is f32, the weights stay bf16 (Mapal's default face). Weights: `benches/model/export_weights.py` writes
the checkpoint's tensors bit for bit as the `Model` value's bytes (2 471 628 800 bytes) and the prompt ids (row 0
of the harness's 2048-token prompt; 128 and 512 are its prefixes). Run: `MAPAL_DATA_DIR=<export dir> mapal run
examples/llama.mapal`; each line is `token logit ms-since-submit`.

### The first step (16d668c6): what the readable program needed

| problem the readable program hit | change | where |
|---|---|---|
| `read` took 1-D arrays only — a `Model` struct of per-layer structs could not be read whole | **a record read**: any array / struct / tuple whose leaves are readable scalars with no padding is the file's bytes, leaves in order (interpreter decodes nested values; the backends read straight into the value's storage) | `mapal-ir/src/read.rs` (`readable`, `file_bytes`, `leaf_runs`), `mapal-interp` (`decode_value`), L1617 text |
| `project<tokens, rows, columns>` (rows·columns, tokens·columns, tokens·rows) had no unknown alone in any equation: L1706 | **size inference, D3.3′**: the whole assignments of a system of products, by the divisors of one product's value, taken when exactly one fits; two fit: an error naming both | `mapal-lower/src/sizes.rs` (`joint`) |
| `() -> time` inside a loop: seal failed (`TokenNotLinear`) — the loop's token fork accepted only `Pair` consumers, and `TimeMs` takes the bare token | the fork's consumers may be `TimeMs` | `mapal-ir` `builder.rs` / `validate.rs` (`is_loop_fork`) |
| a layer loop inside `forward`, called inside the decode loop, could neither inline (nested loops) nor stay a call (every array crossing the call each token); at the top level each trip's `model.layers[layer]` was a fresh view, uploaded each trip | **R-UN**: a counted loop with constant trips is replayed trip by trip when a trip takes a record of an invariant array of records at the counter (`model.layers[layer]`, directly or inside its maps): each view at a constant index, made once and placed once, and the function loop-free (it inlines, also from inside another loop). Token-free state, array work in the body, ≤ 128 trips, ≤ 2¹⁶ morphisms. Tried first, and narrowed: also every loop of a function called inside a loop, and rows of arrays of arrays — it unrolled the generated caching programs' history loops (`caching_memory`) | `mapal-rewrite/src/unroll.rs`, `replay.rs` (`reconstruct_unrolled_loop`) |
| the unrolled `forward` (thousands of morphisms) stayed a call: the inline cap | **inline policy**: a callee with one call site inlines whatever its size (nothing is copied twice). Tried first, and dropped: a loop-bearing callee inlining at a site in no loop — it moved two suites (`caching_memory`'s generated callee loops, the CPU matmul4 example's packed tile) and R-UN covers the model without it | `mapal-rewrite/src/inline.rs` |
| the cache append `t = cell ÷ 512 − layer·ctx − first; (t ≥ 0 ∧ t < tokens)` was no range law: 32 whole-cache (21 MB) copies a step (decode 4.64 ms) | **`map_law` `Outside`**: a bound on a row `t ÷ d` shifted by a captured affine term — rows `[a·e + lo, a·e + hi)` of `d` cells — so a chunk's rows written at a captured position are an in-place range map (decode 4.64 → 3.37 ms) | `mapal-ir/src/algo.rs` (`t_bound`, `t_rows`) |
| a prompt's RMSNorm read a 4 KB gain the host holds and 4 MB the device holds; a straight-line cell needed every operand on the device, so 4 MB went down and back each layer | rung **`once-bytes`**: `bytes-place`'s rule (run where fewer bytes move) for a cell outside every loop too (TTFT 143 → 85 ms). OFF arm `--gpu-off=once-bytes` | `backends/llvm` `resident.rs` (`cell_round`), `lib.rs` |
| attention read `cache.keys` through a struct captured by its map body: no `cell_plan` read, the decode's attention and cache on the host (decode 112 ms) | not changed in the compiler: `attend` takes the keys and values (the textbook `attention(q, k, v)` signature) — the struct-capture read stays open (below) | the program |

Diagnostic: `MAPAL_DEBUG_PLACE=1` prints each placement round's dropped cells with the reason (operands there / on
the device, a host reader, the byte rule) — how the cache and RMSNorm causes above were found.

