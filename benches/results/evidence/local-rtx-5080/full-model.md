# local-rtx-5080 — the full model: readable Llama-3.2-1B-Instruct in Mapal against six inference stacks (2026-11-25)

The real checkpoint, written the way a person writes it (`examples/llama.mapal`: 404 lines, 56 of them comments, 19 functions and types),
read from one file, run end to end on the RTX 5080 (sm_120, 84 SMs, 16 GB, ~960 GB/s; driver 620.73.04, CUDA 13.5) and
timed with the competitors' own harness definitions. Machine: Threadripper 9970X, shared; Mapal and competitor runs
held the bench lock, timed processes on CPUs 16-23 (`nice 10`); load 0.6–3.0 and Tctl 34–44 C during every timed
run (the final Mapal runs at load 2–3: another agent's work winding down; an earlier run of the same programs at load
0.6–1.2 gave 28.9 / 85.0 / 332.1 ms and 3.322 / 3.369 / 3.531 ms — within 1 %). Raw output (never in git): `.claude/raw-evidence/local-rtx-5080/model-ready-full-model/` (`competitors/` the
sweep, `mapal/` the programs, builds and `mapal_n*_b1.json`, `hf/` the references, `changed/` the suite emission
diff, `tests/`).

## The program

`examples/llama.mapal` — top-level sizes (`2048 -> hidden_size;`, prompt length and generated tokens one line each),
`type Matrix<rows, columns> { _: [bf16; rows * columns] }`, per-layer weight structs (`DecoderLayerWeights`) inside a
`Model` read whole (`"model.bin" -> read -> model: Model;`), one key/value cache struct, size-generic blocks
(`rms_norm<tokens>`, one `project<tokens, rows, columns>`, `rotate<tokens, width>`, `attend<tokens>`,
`feed_forward<tokens>`, `decoder_layer<tokens>`), and **one `forward<tokens>`** — a chunk of new tokens through a
layer loop over `model.layers[layer]` — used for the whole prompt (`tokens` = prompt length) and for each generated
token (`tokens` = 1), as inference engines do. `sqrt`, `sin`, `cos`, `exp` are the builtins; the llama3 RoPE
frequencies are a top-level table the compiler computes before the run (ADR-0048: 32 values, embedded). Every
activation is f32, the weights stay bf16 (Mapal's default face). Weights: `benches/model/export_weights.py` writes
the checkpoint's tensors bit for bit as the `Model` value's bytes (2 471 628 800 bytes) and the prompt ids (row 0
of the harness's 2048-token prompt; 128 and 512 are its prefixes). Run: `MAPAL_DATA_DIR=<export dir> mapal run
examples/llama.mapal`; each line is `token logit ms-since-submit`.

## Correctness (greedy, 128 tokens, batch 1)

| prompt | vs HF transformers f32 math (bf16 weights) | vs HF bf16 eager | first-step top logit, Mapal − HF f32 | max over the 128 steps |
|---|---|---|---:|---:|
| 128 | **128 / 128** | 128 / 128 | 3.2e-5 | 3.6e-5 |
| 512 | **128 / 128** | 32 / 128 | 1.9e-6 | 3.7e-5 |
| 2048 | **128 / 128** | 61 / 128 | 1.5e-5 | 1.7e-4 |

(`benches/model/hf_reference.py`: eager attention, TF32 off; logits ~18–26.) Mapal computes the f32 model, so it
follows the f32 reference all the way; every bf16 tool leaves it where a near tie flips. For scale, the competitors
against the harness's `hf-eager` (bf16, SDPA) — the same split points: gpt-fast 49 / 32 / 57, llama.cpp 49 / 67 /
57, SGLang 109 / 95 / 57, TensorRT-LLM 128 / 95 / 57, vLLM 128 / 128 / 57; **HF f32 itself agrees with `hf-eager`
49 / 32 / 57 — exactly Mapal's agreement with it**.

## Time (RTX 5080, batch 1, 128 generated, warmup 2 + 5 reps, medians)

Definitions (the harness's `common.py`): TTFT = submit → first token on the host; decode = (t₁₂₈ − t₂) / 126;
peak = the process's GPU memory (NVML / nvidia-smi). Mapal: one process per repetition (`benches/model/run_mapal.py`:
the weights load and upload before `() -> time`; the prompt is read after it), spreads ≤ 1.3 % (TTFT) and ≤ 0.2 %
(decode). Competitors: `compare.py --batches 1 --lens 128,512,2048` the same afternoon, same lock and CPUs.

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | peak MB (512) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Mapal** (f32 activations) | 28.3 | 85.2 | 332.7 | 3.336 | 3.396 | 3.548 | 5 086 |
| gpt-fast (max-autotune, CUDA graphs) | 4.5 | 13.8 | 51.6 | **2.951** | **2.967** | **3.045** | 3 532 |
| SGLang 0.5.23 | 6.4 | 13.6 | **44.9** | 3.064 | 3.066 | 3.147 | 9 338 (reserved) |
| TensorRT-LLM 1.3.0 | 9.6 | 14.2 | 45.5 | 3.032 | 3.061 | 3.176 | 9 198 (reserved) |
| vLLM 0.33.0 | 6.4 | 13.4 | 45.2 | 3.159 | 3.164 | 3.233 | 9 236 (reserved) |
| llama.cpp bf16 | 11.9 | 13.2 | 55.9 | 3.144 | 3.177 | 3.229 | 2 846 |
| llama.cpp f16 | 5.7 | **10.2** | 45.9 | 3.145 | 3.172 | 3.236 | 2 848 |
| HF eager | 6.4 | 14.2 | 47.8 | 4.460 | 4.466 | 4.458 | 2 780 |
| HF static cache + compile | 8.2 | 15.2 | 48.8 | 3.680 | 3.998 | 5.316 | 2 872 |

**Decode**: Mapal beats HF eager (1.26–1.34×) and HF static cache + compile (1.10 / 1.18 / 1.50×); it is 6–10 %
behind llama.cpp and vLLM, 9–13 % behind SGLang and TensorRT-LLM, 13–17 % behind gpt-fast. **TTFT**: 2.4–8.4×
behind every tool (6.3 / 8.4 / 7.4× the fastest at 128 / 512 / 2048). **Peak**: two device copies of the weights
(below).

### Precision classes (prompt 512)

| program | products | TTFT ms | decode ms | tokens vs HF f32 / bf16 |
|---|---|---:|---:|---|
| as written (default face) | f32 activations × bf16 weights, f32-accurate | 85.2 | 3.396 | 128 / 32 |
| `fold: bf16 \| f32` on `project`'s fold (one line; `mapal/llama_bf16p_n512.mapal`) | bf16 products admitted: the prompt's projections on `mma m16n8k16.bf16` (`HalfMma`), activations stored bf16 | **28.8** | 3.415 | **0 / 0 — wrong output** (token 84707 repeated; first-step top logit 9.46 against 18.5) |

The bf16-permitted build is a **miscompilation found here, not fixed**: HF's bf16 run keeps the first 32 tokens of
the f32 reference, so a correct bf16-product build would too. The build stores `normed` / `mixed` / `attended` bf16
(`precision_plan.store`, `narrow=a:bf16`) and runs the prompt's projections as HalfMma tiles; the decode steps'
one-row GEMV cells read the same arrays — the first suspect is a reader taking a narrow-stored array as f32. Open,
first in line: it is also the competitors' precision and a 3× TTFT.

### Where each side's time goes

- **Mapal decode step (512)**: 291 launches, one sync (the logits, 513 KB, for the host's argmax), no upload.
  `MAPAL_GPU_TRACE` (every launch synchronized, so small kernels read high): gate+up 1.28 ms (16 merged two-lane
  kernels, ~800 GB/s), down 0.69 (~780 GB/s), output head 0.58 (875 GB/s), query/output 0.48 (~560 GB/s), key/value
  0.28 (~230 GB/s), attention scores 0.21 + weighted sum 0.19 (the f32 cache over the whole 640-position context,
  masked positions included), the RMSNorm sums of squares 0.20, the cache appends 0.16 (range cells, in place), and
  ~100 single-row elementwise kernels (RoPE, residuals, SiLU·up, norm scales) at launch-latency each. The 2.47 GB of
  weights at the DRAM rate is 2.57 ms; the rest is launches and the f32 cache (2× the bytes of a bf16 one: decode
  grows 3.34 → 3.55 ms from 128 to 2048).
- **Mapal prefill (512)**, traced on the real build (`mapal/trace_n512.trace.gz`): 38 ms of kernels in the 85 ms —
  the projections on the FFMA tile ~32 ms (gate / up / down 7.6–8.7 ms each over 16 layers, query / output ~2.5,
  key / value 1.8): the default face keeps f32 activations, and the 5080's bf16 split tile is refused by rule
  (tensor-cores-2's open item; forced `--gpu-on=split-tile` takes 6 sites to bf16x2n: TTFT 85 → 77 ms, tokens
  unchanged); the attention's scores already a guarded tile site (16 × 81 µs) and P·V a tile (16 × 170 µs). The rest
  is the prompt's 32 cache writes run through the host: 34 uploads of the 21 MB cache (~28 ms at 25 GB/s) and the
  rows down — the straight-line chain of range maps (open item 3).
- **The competitors**: bf16 activations and bf16 tensor cores (cuBLAS / CUTLASS / FlashAttention / FlashInfer
  kernels; llama.cpp's own), a bf16 KV cache read only up to the current position, CUDA graphs (gpt-fast, vLLM,
  SGLang, TensorRT-LLM, HF static) or fused small kernels; gpt-fast's whole decode step is one captured graph
  of torch.compile max-autotune kernels. Their prefill is ~1 Tflop on a ~120 Tflop/s bf16 pipe; Mapal's f32-accurate
  prefill needs split forms of at least two bf16 products (bf16x2n) or the FFMA pipe (~60 Tflop/s peak).

## What changed on the way (all generic; tests below)

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

## Tests

- New: `mapal-rewrite/tests/unroll.rs` (a layer loop over records called per step unrolls and inlines; a plain
  counted loop stays; **48 generated programs** — layer count 1–6, steps 1–3, width 1–9, record views read inside
  the maps or stacked reads: values equal to the oracle, no call left where records are read; every counter > 0), `backends/llvm/tests/read.rs`
  `a_record_read_whole_on_every_backend` (a struct of a bf16 matrix, per-layer structs with nested arrays and an f32
  vector, read whole, a layer loop over `model.layers[layer]`: interpreter raw and rewritten, native exact and
  default, the GPU build in every emulator mode; a short file fails), `mapal-lower` `size_generics`
  `product_systems_solve_with_the_results_type` (the unique solution; the ambiguous call's error),
  `structural` `a_clock_read_in_a_loop_body`, `read_builtin` (nested / tuple / struct records accepted, a padded
  record L1617), `mapal-ir` `read` unit tests and the §5.1 typing table (a nested array read is well typed).
- Changed expectations: the inline policy tests (`oversized_callee_stays_a_call` now two sites, `+
  oversized_callee_with_one_site_inlines`); `zero_fold`'s generated study pins `once-bytes` off beside `place` (with
  it a straight-line generator runs on the device and the "no kernel before the zero" case leaves the set).
- `cargo test --release -p mapal-ir -p mapal-lower -p mapal-interp -p mapal-rewrite`: all pass; the backend, CLI and
  runtime (`test_par.py --bins 4`, `RUST_TEST_THREADS=8 MAPAL_PAR=2 MAPAL_TESTGEN_JOBS=4`): **66 binaries, 568
  passed, 0 failed** (1655 s; `tests/backend.log`).
- **Suite**: `suite_diff.py --base HEAD` — 35 programs + 2 cells × 5 faces, **same = 45**: no suite program's host or
  device text changes (the suite programs are single-function, loop-free-callee programs), so nothing to time.

## Open (attempted or measured; why it did not close here)

1. **Prefill on tensor cores** (TTFT): the f32-activation projections need a split form; the 5080's bf16x2n is
   refused by `split_rate_pays` (bf16 pipe 2 × the FFMA rate, θ 1.6) though forced it wins (tensor-cores-2's open
   item: the tile's realized fraction vs its issue is the missing property). Forced here: 85 → 77 ms. A program that
   admits bf16 products (`fold: bf16 | f32`) would run the competitors' precision — not the default face.
2. **The causal mask through `positions[t]`** (the user asked for it in this step, not deferred): measured first —
   on the real build (640 positions) the prompt's scores are a guarded tile site with the mask as written (16 × 81 µs,
   1.3 ms of the 38 ms), so the mask is not where the prompt's time goes; a streamed (flash) form would fold the
   scores, softmax and P·V (≈ 4.5 ms) into one pass. Rewriting the mask as `past <= t` (a digit of the cell index)
   gives the same plan here. The general fact the stream plan's `GuardBox` needs is a side "affine in the row index
   plus a per-launch value": `positions[t]` read through a capture whose `elem_plan` law is `first + t` is that form
   with `first` the launch's uniform (the `ua` / `ub` capture terms `tile_plan_radix` already carries for operand
   addresses); a runtime position vector with no law is a per-row bound read once per row block. Neither is
   implemented here (no time left after the cache chain's diagnosis); the uniform-term form is the general one —
   it covers the closed-form arrays, top-level bindings and the decode step's runtime position alike.
3. **The prompt's cache writes** go through the host (a straight-line chain of range maps over a host-made zero
   fill: 34 × 21 MB up at 512 — ~28 ms of the 85 ms TTFT, the largest non-kernel cost). `range-chain` writes in
   place only into a loop carry: a straight-line range cell is `homeless` (`cell_round`), and the fix is a
   straight-line alias in `ResidentPlan` (the cell's output takes its dead source's device buffer; `alias()`, the
   arena's coloring and the frees must see it as one object — the carry-take move, `store 0` into the source slot,
   is the realization). Tried: a `+0.0` fill counted as made
   on the device in the byte rule (zeroed, as a carry's init already is) — the first write's operand is the cache
   struct's field view, not the fill, so nothing moved; the view → fill forward in straight-line code is the next step.
4. **Two device copies of the weights** (peak 4.9–6.3 GB): the prompt's and the decode step's `model.layers[l]`
   are two view objects — CSE keys feeders by object and the two constant indices `l` are two objects. Keying
   constants by value merges them (tried: one upload), but merges across guard arms broke the generated
   `property` programs (an alias to an arm's object) — needs the arm-scoped form.
5. **Decode launches** (291 a step, ~100 one-row elementwise kernels at launch latency) and the K / V projections
   (~230 GB/s, 512 rows on a warp each): the placement step's items 4–5; the attention reads the whole context's f32
   cache including masked positions (`GuardBox` does not skip their bytes).
6. **Struct captures** in map bodies (`cache.keys[...]` inside `attend`'s maps) are not `cell_plan` reads; the
   readable program passes the arrays.
7. Found: a struct literal written straight to `ret.1` in a callee inlined into an unrolled loop panics replay
   (`reconstruct_temp`, an object with no writer); binding it first works — not reduced to a repro yet.
8. **Rented boxes**: not run (this step is local). `~/.cache/mapal/model-compare/setup_box.sh` installs every
   competitor; for Mapal copy the repo, `cargo build --release -p mapal-cli`, run `export_weights.py`, build the
   three variants (`prompt_length` = 128 / 512 / 2048) with `MAPAL_CUDA=<cuda> mapal build --target=native-gpu`,
   then `run_mapal.py --exe <build> --n <N>` beside `compare.py`.

No tailored fixes (`docs/tailored-fixes.md` unchanged).

## For integration

CLAUDE.md facts table (edits to existing rows, not made here — CLAUDE.md is not this step's to change):

- `elem_plan`, `map_law`: "… piecewise when every cell off a row / range / column / cell / row-every-`s`-rows … is a
  capture's own" — add "or off a range of rows `t ÷ d` shifted by a captured affine term (`lo ≤ t ÷ d − X < hi`: a
  chunk's rows written at a captured position)".
- backend row: the rung `once-bytes` (full-model).

Rewriter (DESIGN `docs/components/rewrite`): R-UN (`unroll.rs`) beside R-LF / R-LM; Move 1's policy — a single-site
callee inlines whatever its size. ADRs: 0045 and 0047 carry
amendments (records; products of unknowns).
