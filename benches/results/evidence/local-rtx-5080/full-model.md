# local-rtx-5080 — the full model after the bf16 integration: readable Llama-3.2-1B-Instruct in Mapal against the inference stacks, type to type (2026-11-25, night)

`local-rtx-5080` (the builds measured here: `3c629486`) merges bf16/tensor-tile (`mma-walk`), bf16/class (`finite-check`,
`narrow-window` with the bf16 KV cache, `round-post`, `device-trap`) and prefill/residency's last three commits
(residency calibration, carry-take's way out, `fly-unroll`; the `decode-done` flag named `6bbc6fa8`, its tip). The
integration found one placement fault between `device-trap` and the calibrated pricing and fixed it generically
(`cand-made`, below). The program is the readable `examples/llama.mapal`, unchanged; the bf16 class is
`benches/model/variant.py --compute bf16` (now with the bf16 KV cache). **Type to type**: the bf16 class is compared
only with bf16 engines, the f32 class only with f32-compute engines; no gain is claimed across classes. Raw output
(never in git): `.claude/raw-evidence/local-rtx-5080/bf16-integrate/` (`mapal/` programs and builds, `time/` the
timed passes, `competitors/{bf16,f32}/` the sweeps, `trace/` CUPTI traces, `changed*/` the suite emission and its
ABBA, `tools/`, `test-*.log`).

## Method

- **RTX 5080** (sm_120, 84 SMs, 16 GB), Threadripper 9970X (shared with other agents; boost capped), driver 620.73.04,
  CUDA 13.5. Every timed process held the bench lock and ran on CPUs 16-23,48-55 (`nice 10`); during the passes this
  step ran only one targeted test and one build, on CPUs 24-31,56-63 (the workspace tests ran after the passes).
- **Load**: before every run (each Mapal class × length, each engine × length) the 1-minute load had to fall below 10,
  waiting at most 120 s; the load is logged per run (`time/*/load.log`, the engines' raw JSON `load`). Three engine
  runs met a spike from other processes (vLLM bf16 2048 at load 53, llama.cpp f32 128 at 46, 512 after a 100 s wait);
  they and the f32 512 legs whose spread exceeded 10 % (gpt-fast, vLLM, HF eager, HF static) were re-run at load 4.4
  (`rerun.log`; the first runs kept in `raw-first/`). Every value in the tables ran at load 4–9.
- **Definitions** (the harness's `common.py`): TTFT = submit → first token on the host; decode = (t₁₂₈ − t₂) / 126;
  e2e = t₁₂₈; peak = the process's GPU memory (nvidia-smi, 20 Hz). Greedy, batch 1, 128 generated, warmup 2 + 5 reps,
  medians.
- **Mapal**: `benches/model/run_mapal.py` (one process per repetition; weights load and upload before `() -> time`),
  builds `MAPAL_GPU_CALIB=benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib mapal build --target=native-gpu`.
  Re-timed back to back with the engines: pass A before the bf16 sweep, B right after it, C right after the f32 sweep,
  D right after the re-runs (23:22–23:24, load 4.2–4.4). The tables give pass D; B and C agree within 0.5 % except bf16
  2048 in B (55.1 ms, spread 52.3–55.8 at load 6).
- **Engines**: `~/.cache/mapal/model-compare/compare.py` (own warmup + 5 reps) with the precision step's load-wait
  patch. bf16: the harness's runners. f32: the precision step's runner copies with `--dtype f32` (TF32 off;
  `prefill-precision/competitors-f32/`). **SGLang has no f32 mode. TensorRT-LLM's f32 build runs but its output is
  wrong** (0 of 128 tokens equal HF f32 at every length; its first ids repeat one token, `prefill-precision`'s raw
  `trtllm_n*_b1.json`): no f32 row. llama.cpp f16 is a 16-bit engine of another type (f16): shown for reference, not
  compared.

**Precision classes.**
- **f32 class** — the program as written (Mapal's default face): f32 activations × the bf16 weights, within ADR-0040
  D2's bound (the projections on the split tile `bf16x2n`). Against the f32-compute engines.
- **bf16 class** — `fold: bf16 | f32` on the sums of products (projections, scores, `P·V`): operands rounded to bf16,
  f32 accumulation (ADR-0041); the key/value cache stored bf16 (`[bf16; …]`, rounded once on write); arrays whose every
  read is a bf16 multiplicand stored bf16 (`round-store`, now also an epilogue's array: the attention output). Norms,
  softmax, RoPE and the residual stream stay f32. Against the bf16 engines.

## Tokens (greedy, 128, batch 1; tokens equal before the first difference)

| program | 128 | 512 | 2048 |
|---|---|---|---|
| Mapal f32 class vs HF f32 (`hf_reference.py`, eager, TF32 off) | **128** | **128** | **128** <!-- tokens prefix(integ-f32,hf-f32)@128,512,2048 --> |
| every f32-compute engine vs HF f32 (gpt-fast, vLLM, llama.cpp, HF eager, HF static) | 128 | 128 | 128 |
| Mapal bf16 class (pass D, first repetition) vs HF bf16 eager (`hf_reference.py`) | 128 | 32 | 61 <!-- tokens prefix(integ-bf16,hf-bf16-eager)@128,512,2048 --> |
| Mapal bf16 class, the five repetitions at 512 vs HF bf16 eager | | 32, 67, 89, 100 (4 sequences) | |
| the bf16 engines vs HF bf16 eager | gpt-fast 128, llama.cpp 128, SGLang / TRT-LLM / vLLM / HF eager / HF static 49 | 24, 89, 67, 67, 67, 67, 7 | 61, 61, 61, 61, 89, 57, 57 |

HF f32 and HF bf16 eager agree on 128 / 32 / 61<!-- tokens prefix(hf-f32,hf-bf16-eager)@128,512,2048 --> tokens:
past a near tie every bf16 run follows its own kernels, and the bf16 engines spread 7–89 at 512 among themselves.
The Mapal bf16 class sits inside that spread at every length. **At 512 its tokens vary from run to run** (below:
the tensor tile's stream-K walk adds three or more partials of a tile in arrival order; with `--gpu-off=mma-walk`
the same build is deterministic). The f32 class and the bf16 class at 128 and 2048 give the same tokens every run.
Every marked count is `token-agreement.tsv` (`benches/model/token_agree.py`, runs `integ-*`).

## Time (RTX 5080, batch 1, 128 generated, medians)

### f32 class — against the f32-compute engines

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (default face; projections on `bf16x2n`) | **8.54** | **25.39** | **95.48** | **3.241** | **3.294** | **3.447** | **441** | **2 722 / 2 730 / 3 018** |
| gpt-fast f32 (max-autotune, CUDA graphs) | 13.70 | 47.71 | 213.7 | 5.849 | 5.718 | 5.896 | 774 | 6 296 / 6 626 / 7 688 |
| vLLM 0.33.0 f32 | 14.57 | 41.03 | 178.8 | 6.225 | 5.935 | 6.220 | 795 | 9 188 / 9 188 / 9 236 (reserved) |
| llama.cpp f32 | 14.21 | 41.27 | 203.9 | 6.028 | 6.312 | 7.751 | 843 | 5 148 / 5 200 / 5 396 |
| HF eager f32 | 14.45 | 41.71 | 263.5 | 7.733 | 7.432 | 8.836 | 986 | 5 104 / 5 204 / 6 494 |
| HF static cache + compile f32 | 16.85 | 44.17 | 244.8 | 7.389 | 7.502 | 10.418 | 997 | 5 202 / 5 306 / 6 782 |

**Mapal leads the f32 class everywhere**: TTFT **1.60 / 1.62 / 1.87× faster** than the best f32 engine (gpt-fast,
vLLM, vLLM), decode **1.80 / 1.74 / 1.71×** faster than gpt-fast f32, end to end 1.75× faster at 512, at about half
the lowest engine's memory. Against the previous integration (10.40 / 28.91 / 125.3 ms and 3.76 / 3.59 / 4.00 ms/token,
a louder sitting): the causal skip over a run-time-finite cache (attention at 2048 34.1 → 16.3 ms), the prompt's host
chain gone (`cand-made`), and the decode step's calibrated placement (residency).

### bf16 class — against the bf16 engines

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`, bf16 KV cache) | 5.58 | 14.07 | 52.14 | 3.258 | 3.294 | 3.395 | 429 | **2 652** / 2 870 / 3 228 |
| gpt-fast (max-autotune, CUDA graphs) | **4.94** | 13.73 | 56.11 | **2.972** | **2.984** | 3.327 | **393** | 3 410 / 3 532 / 4 074 |
| SGLang 0.5.23 | 6.83 | 13.80 | 49.30 | 3.336 | 3.078 | 3.506 | 405 | 8 902 / 8 838 / 8 906 (reserved) |
| TensorRT-LLM 1.3.0 | 9.49 | 15.99 | 45.67 | 3.313 | 3.491 | **3.190** | 459 | 8 810 / 8 790 / 8 784 (reserved) |
| vLLM 0.33.0 | 6.79 | 14.94 | **45.30** | 3.490 | 3.508 | 3.236 | 461 | 9 236 (reserved) |
| llama.cpp bf16 | 5.61 | **13.46** | 49.23 | 3.157 | 3.172 | 3.234 | 417 | 2 804 / 2 846 / **2 896** |
| HF eager | 6.63 | 15.50 | 55.96 | 4.674 | 4.743 | 5.033 | 618 | 2 726 / **2 780** / 2 962 |
| HF static cache + compile | 8.81 | 17.60 | 56.05 | 4.043 | 4.561 | 6.030 | 597 | 2 822 / 2 872 / 3 090 |
| *(llama.cpp f16 — another type, not compared)* | *5.66* | *10.27* | *36.74* | *3.153* | *3.180* | *3.235* | *415* | *2 804 / 2 848 / 2 898* |

**bf16 class**: TTFT **1.13 / 1.05 / 1.15× behind** the best bf16 engine (gpt-fast at 128, llama.cpp bf16 at 512,
vLLM at 2048; was 1.42 / 1.66 / 2.00× at the previous integration, then against the best tool including f16). At 128
Mapal ties llama.cpp bf16 and is ahead of SGLang, vLLM, TensorRT-LLM and both HF; at 512 within 2–5 % of gpt-fast,
SGLang and llama.cpp and ahead of vLLM, TensorRT-LLM and both HF; at 2048 ahead of gpt-fast and both HF, 6–15 % behind
llama.cpp, SGLang, TensorRT-LLM and vLLM. Decode **1.10 / 1.10 / 1.06× behind** the best (gpt-fast, gpt-fast,
TensorRT-LLM); ahead of vLLM and TensorRT-LLM at 128 / 512, of SGLang at 128 / 2048 and of both HF everywhere.
Peak memory is the lowest of every tool at 128; at 512 / 2048 HF eager and llama.cpp hold less (the walk's workspaces,
below). Previous integration → now: TTFT 6.95 / 18.12 / 74.05 → 5.58 / 14.07 / 52.14 ms, decode 3.53 / 3.67 / 4.01 →
3.26 / 3.29 / 3.40 ms/token.

## Where each side's time goes (CUPTI, `tools/ktgroup.py`, `ktgaps.py`; the prompt pass from the first launch after the weights to the logits copy)

| prompt, ms | 128 | 512 | 2048 |
|---|---:|---:|---:|
| Mapal bf16 class: kernels / launches / host gaps > 20 µs | 5.42 / 276 / 0.10 | 13.96 / 324 / 0.09 | 52.21 / 308 / 0.08 |
| Mapal f32 class: kernels / launches / host gaps > 20 µs | 8.37 / 261 / 0.06 | 25.24 / 277 / 0.06 | 95.48 / 277 / 0.06 |

**bf16 class, 512** (13.96 ms of kernels): gate + up 5.71 ms (32 × 178 µs, walked: 96 Tflop/s, 91 % of the
calibrated 105 Tflop/s pipe), down 2.89 (16 × 180 µs, walked, 95 Tflop/s), query + output 1.93 (71 Tflop/s), the
elementwise cells 0.99 (146: norms, RoPE, SiLU·up, residuals), the streamed attention 0.89 (16 × 56 µs, causal skip),
key + value 0.61 (56 Tflop/s), the output head 0.60 (one row against the 525 MB embedding), the walk's zero passes
0.34 (48). The host chain (embedding gather and layer 0's norm, 2.5 ms at the previous integration) is gone.
**2048** (52.21 ms): merged gate + up 19.62 (16 × 1.23 ms, at the pipe's rate), down 11.07 (16 × 692 µs walked, 99
Tflop/s), attention 7.91 (16 × 494 µs: the causal skip on a run-time-finite cache halves it, was 16.3 ms; ~37 Tflop/s
on the unmasked half), query + output 5.96 (92 Tflop/s), cells 4.94, key + value 1.80, head 0.60. **128** (5.42 ms):
the projections 4.35 ms (nothing walks: 128-row tiles), head 0.59, cells 0.27, attention 0.14.

**f32 class, 512** (25.24 ms): projections on `bf16x2n` 22.3 ms (gate + up 10.79, down 6.98, query + output 3.52,
key + value 1.04), attention 1.70 (the FFMA stream), head 0.59, cells 0.49; at 2048 the attention is 16.3 of 95.5 ms.

**Decode step** (512, both classes: 3.21–3.24 ms of kernels, 324 launches, 0.03 ms of gaps): the merged horizontal
GEMVs 1.35 ms (48), per-row and GEMV cells 1.15 ms (194), output head 0.58 ms, the decode attention's tiles 0.10 ms.
Untraced the step is 3.29 ms: the kernels, not the host, bound it now. The 2.47 GB of weights at the measured 890 GB/s
are 2.78 ms; gpt-fast's step is 2.97 ms.

**The engines**: bf16 activations end to end, cuBLAS / CUTLASS / FlashAttention / FlashInfer kernels (llama.cpp its
own), causal attention that skips the masked half, fused norms and SiLU, CUDA graphs for the decode step (gpt-fast,
vLLM, SGLang, TensorRT-LLM, HF static). The f32 engines run SGEMM-class f32 kernels (TF32 off) on f32 weights (4.9 GB
read a decode step, against Mapal's 2.47 GB of bf16 weights widened in registers).

## The integration

- **Merges**: `08faa02c` bf16/tensor-tile (clean); `b80bd696` bf16/class (rung list 161 / 163 → 164); `e3a31d1e`
  prefill/residency `30181082..6bbc6fa8` (the `decode-done` flag's commit). Conflicts: the rung list (`mma-walk` and
  `fly-unroll`, 165); carry-take, where this branch's `934ee80a` (a taken init's uses by its out-edges, `edge_uses`)
  met residency's `6badc822` (`uses` without the state's escape, the into-edges over every taking loop) — kept
  residency's into-edges and `edge_uses` (the same uses, no escape entry to remove); `device-trap`'s checking body
  against `fly-unroll`'s depth in the cell kernels — a trap cell has one lane, so its body takes no unroll.
- **CLAUDE.md**: the facts-table rows of both "For integration" sections (`StreamGuard::check`, `TilePost::round`,
  `precision_plan` `store` on an epilogue's array, the `trap_cell_plan` row, the backend rungs and runtime hooks, the
  `mma_walk` / `MmaWalk` price) and `cand-made`.
- **Found by the integration**:
  1. **`cand-made`** (`3c629486`, rung `cand-made`, OFF arm `--gpu-off=cand-made`). Calibrated, the prompt at 128 and
     512 still gathered the embedding rows and ran layer 0's norm on the host (`MAPAL_DEBUG_PLACE`: the gather dropped
     in round 1 for a host reader): the norm's scale map was no placement candidate, its own roofline not paying at the
     calibrated launch cost, and nothing tile-made upstream (`device_made` sees tile sites only), so it read the
     gathered rows on the host, which kept the gather and the row fold there too — each map priced alone. S62's rule
     "on the device when an operand is device-produced" now reads the candidates too: a map an operand of which a
     candidate cell or device site makes is a candidate, and `cell_sites`' fixed point decides the chain whole (bytes,
     readers). Prompt host gaps 1.45 → 0.06 ms (f32 512), 2.5 → 0.09 ms (bf16 512); the kernel suite's emission is
     equal on both arms (45 of 45); generated test below.
  2. **`mma-walk` × `round-post`: no interaction by construction** — a tile site whose epilogue is its own body (`post`,
     where `TilePost::round` lives) is never walked, so the walk's `red` partials never meet a bf16 store; a split
     epilogue (its own map over the site's f32 output, stored rounded) walks. `half_tile::round_post_under_the_walk`.
  3. **The walk makes the bf16 class's tokens run-dependent at 512** and adds memory (open below).
  4. **The suite bundle linked a stale runtime**: `integrated_bundle.sh` ships `target/x86_64-unknown-linux-gnu`'s
     `libmapal_rt.a`, which nothing rebuilt; without bf16/class's `mapal_rt_checkpoint_hook` every arm failed to link and
     `box_run.sh` timed nothing (`arms=0`). `benches/local/gpu_bundle.sh` now builds both runtimes from the tree.
  5. `run_mapal.py` keeps each repetition's tokens (`tokens_per_rep`, `tokens_distinct`) instead of aborting when they
     differ: the default face's order is unspecified (ADR-0040).

## Tests

- Full workspace (`cargo test --workspace --release --no-fail-fast`, `RUST_TEST_THREADS=16 MAPAL_PAR=4
  MAPAL_TESTGEN_JOBS=4`) on the final tree (`3c629486`, `test-full.log`, 8 120 s): **165 test binaries, 1716 passed,
  0 failed, 10 ignored** — the walk's and the bf16 class's generated tests (`mma_walk`, `flash_attention`,
  `half_tile`, `onto_device`) and residency's (`placement`) among them.
- New, generated: `placement::cand_made_places_generated_prompt_heads` — 12 prompt heads (`prompt_head_src`: a gather
  through run-time ids no proof bounds, its rows' sum of squares, the norm's scale, a projection, a residual; widths
  256–2048, 8–512 rows) placed against x86-64-v3's measured host and the 5080: candidates only grow (a gather the
  roofline put on the device stays), values within the default face's bound in every emulator mode and on the OFF arm.
  Counters: 11 gathers on the device, **5 moved there by `cand-made`** (0 without it: the test fails), 9 with the norm,
  8 with the residual, 8 OFF-arm runs. `half_tile::round_post_under_the_walk` — 4 rounding post sites kept ordered
  under the forced walk, 3 split epilogues walked, 22 emulated runs within the bound.
- **Kernel suite emission vs `bb9bbed7`** (`gpu_changed.sh --base bb9bbed7`): 35 programs + 2 cells × 5 faces, host and
  device text equal on 42; `pt_causal_1024_256`, `pt_prefill_1024_512_256`, `pt_gen_1024_512_256` differ
  (residency's `fly-unroll`). The runtime differs (the checkpoint hook) on all. ABBA of the three, 5 repetitions, value
  gates pass (`changed/results`, load 7 → 4): causal 12.61 vs 12.75 ms (0.989), prefill 6.48 vs 6.50 (0.998), gen 6.57
  vs 6.44 (1.019, ranges 6.45–6.60 vs 6.44–6.60 overlap; residency's −13 % on gen did not reproduce here). `cand-made`
  on / off: 45 of 45 equal. The full-suite safety net for the runtime change did not run in this step.
- In the model (ABBA, bf16 512, 3 + 3 runs): the walk on 14.06–14.08 ms TTFT, off 15.07–15.10 (−1.02 ms), decode
  equal, peak 2 870 vs 2 700 MB.

## Where Mapal still loses, why, and what is next (generic)

1. **bf16 TTFT at 2048** (52.1 vs vLLM 45.3, TRT-LLM 45.7): attention 7.9 ms (FlashAttention-class kernels do the
   causal half in ~3 ms; the `stream-mma` geometry at 2048 is 32-row blocks × 16-key windows, 64 threads), and 4.9 ms of
   f32 elementwise cells (norms, RoPE, SiLU·up, residuals over f32 activations; the engines keep them bf16). Next:
   the stream's block geometry from the pipe's rate (as the tensor tile's), and the activations' storage format from
   their readers (`round-store` takes only arrays read solely as multiplicands).
2. **bf16 TTFT at 128** (5.58 vs gpt-fast 4.94): the projections are 4.35 of 5.42 ms; at 128 rows nothing walks and
   the 128 × N tiles fill 84 SMs in partial rounds (the tensor-tile report's open 1: the ordered tile's achieved pipe
   share is no property yet, so the price misses wins at small grids).
3. **Decode** (3.26–3.40 vs gpt-fast 2.97): 3.21 ms of kernels, 324 launches; the GEMVs run near the DRAM rate, the
   rest is ~240 small kernels (residency step: merging q/k/v needs the inlined calls' walk order — open there).
4. **The walk's side effects** (bf16 class): (a) at 512 the greedy tokens differ between runs (4 sequences in 5): a
   shared tile's 3+ partials are added by `red` in arrival order, so the f32 sums round differently run to run and a
   near tie flips (2 partials into the zeroed output commute exactly: 128 and 2048 are deterministic); (b) +170 MB peak
   at 512 (and the 2048 peak above bf16/class's 2 954 MB): each walked site's workspace is its own buffer, not colored
   by death across the 16 layers. Next, generic: the ticket sums each tile's partials in piece order (the tensor-tile
   report's open 2, "per-block slots summed at the ticket") — deterministic and without the zero pass — and the
   workspaces as arena values with their kernel's lifetime.
5. **Peak memory** at 512 / 2048 above HF eager and llama.cpp (item 4b plus the f32 activations).

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

