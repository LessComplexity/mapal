<!-- markdownlint-disable MD033 MD041 -->
<div align = center>

<img src="assets/logo-wordmark.svg" width="320" alt="Mapal">

<br>

[![Badge Workflow]][Workflow]
[![Badge License]][License]
![Badge Language]
[![Badge Pull Requests]][Pull Requests]
[![Badge Issues]][Issues]
![Badge Numerics]<br>

<br>

</div>

<!--toc:start-->

- [Philosophy](#philosophy)
- [Example: Llama-3.2-1B](#example-llama-32-1b)
- [Syntax](#syntax)
- [Current results](#current-results)
  - [The full model](#the-full-model)
  - [The kernel and layer suite](#the-kernel-and-layer-suite)
- [Roadmap](#roadmap)
- [Status](#status)
- [Trying it](#trying-it)
  - [Formatting](#formatting)
  - [Editor support](#editor-support)
- [How the project works](#how-the-project-works)
- [License](#license)

<!--toc:end-->

**A dataflow programming language.** You write what the data is and how it flows; the compiler works out how
it runs: on the CPU, on the GPU, and across both. No threads, streams, kernels, tile sizes or intrinsics appear
in the source.

---

## Philosophy

- **The language is the flow.** A program states data and how it flows, `data -> transform -> result;`, not the
  control that moves it. A `map` says every cell is independent, a `fold` says how values combine, a binding says
  who reads what.
- **Data is category-theoretic.** Values are objects, transforms are morphisms, a program is their composition.
  The compiler's IR is that graph, and its rewrites are the category's laws
  ([`docs/spec/category-ir.md`](docs/spec/category-ir.md)).
- **Expressing the flow beats expressing control.** Because the source states only what depends on what, the
  compiler derives what runs in parallel, what merges into one kernel, what stays in registers, shared memory or
  cache, and what moves between devices, from facts on the graph (independence, reuse, death points, footprints)
  against the hardware's properties. Control written by hand fixes each of those decisions for one machine.

What follows from it:

- **Hardware is physical properties, never names.** The rules read SM counts, register and shared-memory sizes,
  cache capacities, bandwidths, latencies and matrix-instruction shapes, queried from the device or measured by
  calibration. A new chip is a new property set, not a new code path.
- **No runtime graph recording.** Which work merges into one kernel, vertically along a chain and horizontally
  across independent branches, is decided at compile time from the dataflow. Nothing is traced or captured while
  the program runs.
- **Correct numerics: the flow and the bound, not bit-identity.** Every result is within an error bound stated before
  the program runs, at full precision ([ADR-0040](docs/decisions/ADR-0040-default-float-face.md)). The compiler orders
  and combines operations freely for speed and placement, so parallel sums may differ in the last bits between runs —
  correctness is a correct flow of operations, not a repeated bit pattern
  ([ADR-0049](docs/decisions/ADR-0049-correctness-is-the-flow-and-the-bound.md)). Bit-reproducible runs exist as a
  development mode for testing and debugging. Reduced precision is opt-in, per product.
- **Readable, idiomatic source.** Programs read like the math they compute. If natural code compiles worse than a
  hand-shaped version, the compiler is fixed, not the program.
- **A language AI agents can write and verify.** Deterministic semantics, precise diagnostics with stable codes, one
  formatter (`mapal fmt`), and the interpreter as an exact reference that any compiled run is checked against.

---

## Example: Llama-3.2-1B

[`examples/llama.mapal`](examples/llama.mapal) is Llama-3.2-1B-Instruct in 403 lines, written to be read: the
checkpoint's weights are one `read`, and one `forward` runs the whole prompt and then each generated token through
the 16 layers. Excerpts, unedited.

RMSNorm, for any number of tokens:

```flow
// RMSNorm of each vector: x / sqrt(mean(x^2) + epsilon), times a learned gain per feature.
fn rms_norm<tokens>(x: [f32; tokens * hidden_size], gain: [bf16; hidden_size]) -> [f32; tokens * hidden_size] {
    hidden_size -> iota -> features;
    hidden_size -> widen_f32 -> width;
    tokens -> iota -> map {
        t -> (0.0, features) -> fold {
            sum, i -> sum + x[t * hidden_size + i] * x[t * hidden_size + i]
        }
    } -> sum_of_squares;
    tokens * hidden_size -> iota -> map {
        cell ->
        sum_of_squares[cell / hidden_size] / width + rms_epsilon -> sqrt -> root;
        gain[cell % hidden_size] -> widen_f32 -> g;
        x[cell] * (1.0 / root) * g
    } -> ret;
}
```

One decoder layer, as a flow from the hidden state to the next hidden state and the updated key/value cache:

```flow
// One decoder layer over the chunk: attention with a residual connection, then the feed-forward block with
// a residual connection. The chunk's keys and values enter the cache before its queries attend.
fn decoder_layer<tokens>(
    hidden: [f32; tokens * hidden_size],
    weights: DecoderLayerWeights,
    layer: i32,
    cache: KeyValueCache,
    positions: [i32; tokens],
) -> ([f32; tokens * hidden_size], KeyValueCache) {
    (hidden, weights.attention_norm) -> rms_norm -> normed;
    (weights.query, normed) -> project -> query: [f32; tokens * query_size];
    (weights.key, normed) -> project -> key: [f32; tokens * key_value_size];
    (weights.value, normed) -> project -> value: [f32; tokens * key_value_size];
    (query, positions) -> rotate -> rotated_query;
    (key, positions) -> rotate -> rotated_key;
    (cache.keys, rotated_key, layer, positions[0]) -> write_rows -> keys;
    (cache.values, value, layer, positions[0]) -> write_rows -> values;
    KeyValueCache {
        keys: keys,
        values: values,
    } -> updated_cache;
    (rotated_query, positions, updated_cache, layer) -> attend -> attended;
    (weights.output, attended) -> project -> attention_output: [f32; tokens * hidden_size];
    (hidden, attention_output) -> add -> after_attention;
    (after_attention, weights.feed_forward_norm) -> rms_norm -> normed_again;
    (normed_again, weights) -> feed_forward -> feed_forward_output;
    (after_attention, feed_forward_output) -> add -> ret.0;
    updated_cache -> ret.1;
}
```

<details>
<summary>Attention (<code>attend</code>): grouped-query, causal, softmax as three maps and two folds</summary>

```flow
// Causal grouped-query attention over layer `layer`'s cached keys and values: each token's query head h reads
// key/value head h / 4 of every cached position up to its own, scores are scaled by 1 / sqrt(head_size),
// softmax over each head's scores, and the result is the weighted sum of the cached values.
fn attend<tokens>(
    query: [f32; tokens * query_size],
    positions: [i32; tokens],
    cache: KeyValueCache,
    layer: i32,
) -> [f32; tokens * query_size] {
    head_size -> iota -> dimensions;
    context_length -> iota -> past_positions;
    head_size -> widen_f32 -> sqrt -> scale_root;
    tokens * query_heads * context_length -> iota -> map {
        cell ->
        cell / context_length -> token_head;
        cell % context_length -> past;
        token_head / query_heads -> t;
        token_head % query_heads / heads_per_key_value_head -> key_value_head;
        (past <= positions[t]) -> {
            -true-> {
                (layer * context_length + past) * key_value_heads + key_value_head -> key_row;
                (0.0, dimensions) -> fold {
                    sum, d -> sum + query[token_head * head_size + d] * cache.keys[key_row * head_size + d]
                } -> dot;
                dot / scale_root
            }
            -false-> -1000000000.0;
        }
    } -> scores;
    tokens * query_heads -> iota -> token_heads;
    token_heads -> map {
        token_head -> (-1000000000.0, past_positions) -> fold {
            best, past -> (best, scores[token_head * context_length + past]) -> larger
        }
    } -> row_maximum;
    tokens * query_heads * context_length -> iota -> map {
        cell -> scores[cell] - row_maximum[cell / context_length] -> exp
    } -> exponentials;
    token_heads -> map {
        token_head -> (0.0, past_positions) -> fold {
            sum, past -> sum + exponentials[token_head * context_length + past]
        }
    } -> row_total;
    tokens * query_size -> iota -> map {
        cell ->
        cell / head_size -> token_head;
        cell % head_size -> d;
        token_head % query_heads / heads_per_key_value_head -> key_value_head;
        (0.0, past_positions) -> fold {
            sum, past ->
            (layer * context_length + past) * key_value_heads + key_value_head -> value_row;
            sum + exponentials[token_head * context_length + past] * cache.values[value_row * head_size + d]
        } -> weighted;
        weighted / row_total[token_head]
    } -> ret;
}
```

</details>

Greedy decoding, inside `main`: each step feeds the last token back through the same `forward`:

```flow
    // Each later token: feed the previous one at the next position.
    mut token: i32 <- first_token;
    mut cache: KeyValueCache <- prefilled.1;
    mut position: i32 <- prompt_length;
    loop {
        (position < prompt_length + generated_tokens - 1) -> {
            -true-> {
                [token] -> ids;
                (ids, position, model, cache) -> forward -> stepped;
                stepped.0 -> most_likely_token -> token;
                stepped.1 -> cache;
                () -> time -> now;
                token -> print;
                " " -> print;
                stepped.0[token] -> print;
                " " -> print;
                now - submitted -> println;
                position + 1 -> position;
                -> loop;
            }
            -false-> position -> last_position;
        }
    }
```

The compiler turns the folds in `project` into tiled matrix products, streams the attention, keeps the cache on
the GPU and writes each step's new rows in place, and merges what it can into kernels. None of it is in the source.

---

## Syntax

The core forms in one whole program. It reads 12 `bf16` values from `weights.bin` (the directory comes from
`MAPAL_DATA_DIR`) and prints one number.

```flow
// Top-level values: immutable (there is no `const`), computed before the program runs.
3 -> rows;
4 -> columns;

// `_` is a type's default interpretation: a Matrix is its array wherever an array fits.
type Matrix<r, c> {
    _: [bf16; r * c]
}

// Size-generic: `r` and `c` are inferred from the arguments at each call.
fn project<r, c>(weight: Matrix<r, c>, x: [f32; c]) -> [f32; r] {
    c -> iota -> along;
    r -> iota -> map {
        row -> (0.0, along) -> fold {
            sum, column ->
            weight[row * c + column] -> widen_f32 -> w;
            sum + w * x[column]
        }
    } -> ret;
}

// A guard: one arm runs, chosen by the condition.
fn relu(v: f32) -> f32 {
    (v > 0.0) -> {
        -true-> v;
        -false-> 0.0;
    } -> ret;
}

// A loop: `mut` state, and a guard that either goes round again or leaves.
fn total<n>(values: [f32; n]) -> f32 {
    mut step: i32 <- 0;
    mut sum: f32 <- 0.0;
    loop {
        (step < n) -> {
            -true-> {
                sum + values[step] -> sum;
                step + 1 -> step;
                -> loop;
            }
            -false-> sum -> ret;
        }
    }
}

fn main() {
    "weights.bin" -> read -> weight: Matrix<rows, columns>;
    columns -> iota -> map {
        i -> i -> widen_f32 -> sqrt
    } -> x;
    (weight, x) -> project -> y;
    y -> map {
        v -> v -> relu
    } -> activated;
    activated -> total -> println;
}
```

| form | what it says | decision |
| --- | --- | --- |
| `x -> f -> y;` | a pipeline: `x` flows through `f`, the result is bound as `y` | [ADR-0005](docs/decisions/ADR-0005-flow-is-a-statement.md) |
| `2048 -> hidden_size;` | a top-level binding: immutable (no `const`; only `mut` values change), computed before the program runs when its dataflow reads no input | [ADR-0048](docs/decisions/ADR-0048-top-level-bindings-compile-time-evaluation.md) |
| `type Matrix<rows, columns> { _: [bf16; rows * columns] }` | `_` is the type's default interpretation: with no other field the type is an alias, with others it reads as `_` wherever only that fits | [ADR-0048](docs/decisions/ADR-0048-top-level-bindings-compile-time-evaluation.md) |
| `fn project<rows, columns>(…)` | size-generic: the sizes are inferred at each call, one fixed-size function per size tuple | [ADR-0047](docs/decisions/ADR-0047-size-generic-functions.md) |
| `x -> exp`, `-> sqrt`, `-> sin`, `-> cos` | math builtins in pipeline position (no call syntax), bit-defined on the exact faces | [ADR-0042](docs/decisions/ADR-0042-exp-builtin.md), [ADR-0046](docs/decisions/ADR-0046-math-builtins-and-target-realizations.md) |
| `"file" -> read -> w: [bf16; n];` | load a value's raw bytes from a file: an array, or a whole struct of arrays | [ADR-0045](docs/decisions/ADR-0045-read-builtin.md) |
| `bf16`, `f16` | narrow storage: one rounding in (`to_bf16`), exact widening out (`widen_f32`), f32 compute | [ADR-0043](docs/decisions/ADR-0043-narrow-float-storage.md) |
| `n -> iota -> map { i -> … }` | a value per cell, every cell independent | |
| `(seed, xs) -> fold { acc, x -> … }` | combine values; on the default face the order is the compiler's | [ADR-0040](docs/decisions/ADR-0040-default-float-face.md) |
| `(c) -> { -true-> a; -false-> b; }` | a guard: exactly one arm runs | |
| `mut s: T <- init; loop { … -> loop; }` | a loop: `mut` state, an arm that goes round again or leaves | |

Every decision record, with the alternatives it rejected: [`docs/decisions/`](docs/decisions/README.md).

---

## Current results

Five GPUs, 2026-11-26: an RTX 5080 at home and an RTX 3090, A100, RTX 4090 and H100 rented. The full model on all
five and the kernel suite on the rented four ran this round's final build (`b8429388`, re-timed on `9be9886d`'s
runtime); the RTX 5080's suite column is the gapfix integration's (`d945c641`). Type against type: the f32 program
against engines computing in f32, the bf16 program against bf16 engines, each suite row against competitors at its
own precision. No gain is claimed across precisions.

**Correctness.** Every timed program passes its value gate on every GPU before it is timed: within the stated error
bound on the default face, in any order of operations, and bit for bit with the interpreter in the development modes
([ADR-0049](docs/decisions/ADR-0049-correctness-is-the-flow-and-the-bound.md)). One bf16 program
(`px_llm_blkdec_b2_l16_bf16`) first read as outside its bound on the rented GPUs: the compiled result was right — bit
for bit the program with the bf16 stores its `: bf16 | f32` permits — and the gate's bound was the unsound part, so
the gate now checks a bound composed through every operation of the program
([ADR-0041](docs/decisions/ADR-0041-compute-precision-permission-candidate.md)). The model's generated tokens are
checked against the Hugging Face reference of the same class, run on the same box.

<!-- results-headline:start -->

**Kernel and layer suite**: Mapal ÷ the fastest competitor at the program's own precision, geomean over the 119 programs timed on every GPU, both sides timed at the program boundary.

| GPU | Mapal ÷ fastest | programs faster / even / slower |
| --- | ---: | ---: |
| RTX 5080 | **0.876** | 46 / 44 / 30 |
| RTX 3090 | **0.920** | 39 / 30 / 51 |
| A100 | 1.006 | 57 / 11 / 51 |
| RTX 4090 | **0.911** | 49 / 25 / 46 |
| H100 | 1.101 | 30 / 8 / 82 |

**Llama-3.2-1B, the full model**: Mapal ÷ the fastest engine of the same class, prompts 128 / 512 / 2048 (batch 1, 128 generated tokens, medians of 5).

| GPU | class | time to first token | decode per token | at 512: Mapal vs the fastest engine |
| --- | --- | ---: | ---: | --- |
| RTX 5080 | f32 | **0.62** / **0.62** / **0.53** | **0.55** / **0.58** / **0.58** | 25.4 vs 41.0 ms (vLLM f32); 3.29 vs 5.72 ms / token (gpt-fast f32) |
| RTX 5080 | bf16 | 1.13 / 1.05 / 1.15 | 1.10 / 1.10 / 1.06 | 14.1 vs 13.5 ms (llama.cpp bf16); 3.29 vs 2.98 ms / token (gpt-fast) |
| RTX 3090 | f32 | 1.04 / 0.95 / **0.82** | **0.65** / **0.66** / **0.67** | 56.0 vs 58.8 ms (llama.cpp f32); 4.04 vs 6.15 ms / token (gpt-fast f32) |
| RTX 3090 | bf16 | 1.55 / 1.17 / 1.30 | 1.23 / 1.23 / 1.25 | 23.8 vs 20.4 ms (SGLang); 4.01 vs 3.27 ms / token (gpt-fast) |
| A100 | f32 | **0.73** / **0.52** / **0.43** | **0.86** / **0.91** / **0.89** | 40.0 vs 76.8 ms (vLLM f32); 3.82 vs 4.21 ms / token (gpt-fast f32) |
| A100 | bf16 | 1.87 / 1.44 / 2.12 | 1.56 / 1.55 / 1.57 | 16.1 vs 11.1 ms (SGLang); 3.69 vs 2.38 ms / token (gpt-fast) |
| RTX 4090 | f32 | **0.72** / **0.64** / **0.53** | **0.62** / **0.63** / **0.63** | 20.1 vs 31.6 ms (vLLM f32); 3.47 vs 5.54 ms / token (gpt-fast f32) |
| RTX 4090 | bf16 | 1.44 / 1.16 / 1.24 | 1.18 / 1.19 / 1.19 | 11.9 vs 10.3 ms (gpt-fast); 3.47 vs 2.93 ms / token (gpt-fast) |
| H100 | f32 | 1.04 / **0.74** / **0.50** | **0.93** / 0.95 / 0.95 | 21.8 vs 29.6 ms (HF eager f32); 2.06 vs 2.16 ms / token (gpt-fast f32) |
| H100 | bf16 | 1.30 / 1.88 / 2.37 | 1.49 / 1.56 / 1.51 | 7.60 vs 4.04 ms (gpt-fast); 2.06 vs 1.32 ms / token (gpt-fast) |

Below 1: Mapal faster; bold: faster by more than 5%. Type to type: the **f32 class** is the program as written (f32 activations on the bf16 weights) against gpt-fast, vLLM, llama.cpp and Hugging Face computing in f32 with TF32 off (SGLang has no f32 path and TensorRT-LLM's f32 output is wrong: neither is compared); the **bf16 class** (`fold: bf16 | f32`, bf16 KV cache) against gpt-fast, vLLM, SGLang, TensorRT-LLM, llama.cpp and Hugging Face in bf16. llama.cpp in f16 is another 16-bit format and is not compared.

<!-- results-headline:end -->

### The full model

The readable [`examples/llama.mapal`](examples/llama.mapal) on all five GPUs against gpt-fast, vLLM, SGLang,
TensorRT-LLM, llama.cpp and Hugging Face (eager, and static cache + `torch.compile`), in each precision class. Each
bar is Mapal's time over the fastest engine of its class on that GPU; the leader is named on the right.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/llm-f32-dark.svg">
  <img alt="Llama-3.2-1B, f32 class, five GPUs: Mapal's time over the fastest f32 engine's, time to first token and decode at a 512-token prompt" src="benches/results/charts/llm-f32-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/llm-bf16-dark.svg">
  <img alt="Llama-3.2-1B, bf16 class, five GPUs: Mapal's time over the fastest bf16 engine's, time to first token and decode at a 512-token prompt" src="benches/results/charts/llm-bf16-light.svg">
</picture>

- **f32 class** (the program as written: f32 activations on the bf16 weights): ahead of every f32 engine on decode
  on all five GPUs (0.55–0.95×) and on time to first token from a 512-token prompt up (0.43–0.95×); at 128 tokens
  4 % behind on the RTX 3090 and H100. Its generated tokens equal the Hugging Face f32 reference's, 128 of 128, at
  every prompt length on every GPU. Peak GPU memory 2.7–3.5 GB, against 5.1 GB and more for every f32 engine.
- **bf16 class** (`fold: bf16 | f32`, bf16 KV cache): behind the fastest bf16 engine on every GPU, closest on the
  RTX 5080 (1.05–1.15× to the first token, 1.06–1.10× decode) and furthest on the A100 and H100 (up to 2.12× and
  2.37× to the first token at 2048, 1.49–1.57× decode), where gpt-fast's CUDA-graph decode and the TensorRT-LLM /
  SGLang prefill are strongest. Its tokens leave the bf16 reference after a near tie, as the bf16 engines' do among
  themselves (on the RTX 5080, 7 to 89 equal tokens at 512); on the RTX 5080 and A100 they can vary from run to run,
  because the tensor tile adds partial sums in arrival order. Peak memory 2.7–3.3 GB, level with llama.cpp and HF
  eager, against 8.8–47 GB for the pooled servers (vLLM, SGLang, TensorRT-LLM).
- llama.cpp in f16 is another 16-bit format: recorded, compared with neither class. SGLang has no f32 path, and
  TensorRT-LLM's f32 build gives wrong tokens: neither is in the f32 class.

Every engine at every prompt length, the tokens and the peaks:
[`full-model.json`](benches/results/evidence/full-model.json); the RTX 5080 with where each side's time goes and
what is next: [`full-model.md`](benches/results/evidence/local-rtx-5080/full-model.md); the rented GPUs:
[`cross-arch`](benches/results/evidence/cross-arch/README.md#b8429388--9be9886d-five-gpu-round-bf16-integration-every-competitor-leg-at-the-program-boundary-the-full-model).

### The kernel and layer suite

GEMMs and projections, MLP chains, attention and softmax, LLM layers, convolution and FIR, fan-in / fan-out and
loops, on every GPU of the round, each program against the fastest of cuBLAS, cuBLASLt, CUTLASS, PyTorch eager and
`torch.compile` (with and without CUDA graphs), SDPA, FlashAttention-2/3, FlashInfer, hand-written Triton, Liger
and hand-written CUDA.

Both sides are timed at the program boundary: the device idle, then the host clock around the call, its dispatch
and launches inside. Against the kernel-event bracket of earlier rounds this moves every GPU 5–11 % in Mapal's
favour (launch-bound torch legs); on that old bracket the rented GPUs read 1.011 / 1.134 / 0.964 / 1.233 (RTX 3090 /
A100 / RTX 4090 / H100) against the previous round's 0.997 / 1.185 / 0.954 / 1.204 (`f1d3a04f`). Attention,
softmax, convolution and elementwise chains lead on every GPU. GEMM cells and projections are the weak family:
behind on the RTX 3090, A100 and H100 (1.05 / 1.16 / 1.37), and the TF32 rows (mostly GEMMs, against cuBLAS) and
the bf16-weight rows (LLM layers, against `torch.compile`) trail on every GPU (1.15–1.72).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/suite-gpus-dark.svg">
  <img alt="Geomean of Mapal time over the fastest same-precision competitor's time, per GPU" src="benches/results/charts/suite-gpus-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/suite-groups-dark.svg">
  <img alt="The same geomean by workload family and by precision class, per GPU" src="benches/results/charts/suite-groups-light.svg">
</picture>

Every row, its legs and its leader: [`matrix.md`](benches/results/evidence/matrix.md). The rented GPUs, their
boxes and rounds: [`cross-arch`](benches/results/evidence/cross-arch/README.md). Earlier results (CPU against
NumPy and OpenBLAS, the October four-GPU table, plan-cost, tile efficiency):
[`docs/performance/earlier-readme-results.md`](docs/performance/earlier-readme-results.md). Method and history:
[`docs/performance/`](docs/performance/).

The charts and the table above are generated. To refresh them after a run, update
[`matrix.json`](benches/results/evidence/matrix.json) (`benches/local/matrix.py`) and
[`full-model.json`](benches/results/evidence/full-model.json), then run
`python3 benches/results/charts/make_charts.py`.

---

## Roadmap

1. **An AMD backend.** The same graph and facts on a second GPU vendor; its compute units, wavefront width, LDS
   and matrix instructions are one more property set.
2. **A PyTorch importer**, to bring existing models into Mapal.
3. **Then the ecosystem around the language:** development → deployment → verification → synthesis.

Done so far, and how each step turned out, including what was tried and did not work:
[`docs/sessions/`](docs/sessions/).

---

## Status

|                     |                                                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language core       | working: pipelines, top-level bindings, size-generic functions and types, guards, loops, `map` / `fold` / `zip` / `enumerate`, tuples, structs, fixed arrays |
| Numbers             | `i32`, `i64`, `u8`, `f32`, `f64`; `bf16` / `f16` storage; `exp`, `sqrt`, `sin`, `cos`; reduced-precision products opt-in (`: tf32 \| f32`, `: bf16 \| f32`)   |
| Effects             | `print` / `println`, `time`, `read`                                                                                                                             |
| Interpreter         | working: _defines_ the language; every backend is tested against it                                                                                            |
| CPU backend (LLVM)  | working: threading, vectorization, cache blocking                                                                                                               |
| GPU backend (CUDA)  | working: value gates pass on the RTX 5080, RTX 3090, A100, RTX 4090 and H100 ([Current results](#current-results)); tensor cores, `wgmma` / TMA where the part has them |
| FPGA backend        | not started                                                                                                                                                     |
| Command-line tool   | `mapal build`, `run`, `test`, `dump-ir`, `fmt`                                                                                                                  |
| Tests               | about 1,700 across the workspace; the GPU tests skip without a CUDA toolchain                                                                                   |
| CI                  | `cargo fmt` + the full suite on Linux and macOS, per push; a skipped LLVM differential fails the run                                                            |

**Checked against the interpreter.** On the default face, compiled values are checked against the interpreter within
the stated bound, in whatever order the compiler chose. In the development modes (the exact faces,
[ADR-0049](docs/decisions/ADR-0049-correctness-is-the-flow-and-the-bound.md)) every compiled program is checked to
print exactly what the interpreter prints, for the examples and for programs from the random generator, at any
thread count and on any device: the mode for testing and debugging.

**Not in the language yet:** recursion, sum types, pattern matching, strings beyond printing and file paths,
closures, dynamic arrays, type-generic functions, modules.

Per-component state: [`docs/STATUS.md`](docs/STATUS.md). Research compiler; no packaged release.

---

## Trying it

Needs a recent Rust toolchain and `clang`; the GPU backend needs an NVIDIA GPU and CUDA.

```sh
# the test suite = the correctness argument
cargo test --workspace --release

# the CLI and the runtime it links against
cargo build --release -p mapal-cli -p mapal-rt
target/release/mapal run --interp examples/pipeline.mapal   # the interpreter, the reference
target/release/mapal run examples/fir.mapal                 # compiled for this machine
target/release/mapal test examples/fir.mapal                # compiled vs the interpreter, byte for byte

# the Llama example on a CUDA GPU (weights exported outside the repository)
python3 benches/model/export_weights.py
MAPAL_DATA_DIR=~/.cache/mapal/models/llama-3.2-1b-instruct/mapal \
    target/release/mapal run --target=native-gpu examples/llama.mapal
```

Start in [`examples/`](examples/): `pipeline.mapal` for syntax, `sepia.mapal` for most of the language,
`fir.mapal` for a loop, `llama.mapal` for a whole model.

### Formatting

```sh
target/release/mapal fmt path/to/program.mapal           # format in place
target/release/mapal fmt --check path/to/program.mapal   # check without writing
target/release/mapal fmt --stdout path/to/program.mapal  # preview without writing
```

The formatter uses four spaces and expands brace bodies onto multiple lines, preserving comments and literal
spelling. `mapal fmt -` reads stdin and writes stdout, so editors and generators share the same style. Invalid
syntax is reported without replacing the source. LLVM and GPU tools are not needed for formatting.

### Editor support

Both editors get highlighting, formatting through `mapal fmt`, and a file icon. Neither is published to a
registry, so both load from disk.

|                 | Neovim                                     | VS Code                                      |
| --------------- | ------------------------------------------ | -------------------------------------------- |
| Highlighting    | Vimscript syntax file                      | TextMate grammar                             |
| File icon       | font glyph                                 | the real SVG logo                            |
| Formatting      | `:MapalFormat` or `gggqG`                  | **Format Document**                          |
| Binding vs call | resolved by scanning for `fn` declarations | lexical only - `-> name;` reads as a binding |
| Details         | [`editors/nvim/`](editors/nvim/)           | [`editors/vscode/`](editors/vscode/)         |

**Neovim** - with lazy.nvim:

```lua
{ dir = "/path/to/mapal/editors/nvim", ft = "mapal" }
```

or without a plugin manager:

```lua
vim.opt.runtimepath:append("/path/to/mapal/editors/nvim")
require("mapal.icon").setup()          -- optional: nvim-web-devicons / mini.icons
```

**VS Code / Cursor** - build the extension and install it:

```sh
python3 editors/vscode/package-vsix.py
code --install-extension editors/vscode/mapal-lang-0.2.0.vsix   # or `cursor`
```

Then restart. A `.mapal` file should show **Mapal** in the status bar. Copying the folder into
`~/.vscode/extensions` does _not_ work: VS Code reads `extensions.json` as its registry and ignores unregistered
directories silently.

For formatting, install the CLI with `cargo install --path crates/mapal-cli --locked`, or point the plugins at
your built `target/release/mapal`: set `vim.g.mapal_executable` in Neovim or `mapal.executablePath` in VS Code.
Formatting uses unsaved buffer text and supports format-on-save through the editor's configuration; see the
plugin READMEs for setup.

**The logo as a terminal glyph.** The Rust and C++ marks in a file tree are font glyphs, not images - Nerd Fonts
ships those brand logos as characters. So Mapal's mark ships as a single-glyph font,
[`assets/font/MapalIcons.ttf`](assets/font/): install it and add it as a terminal _fallback_ font, rather than us
patching and redistributing someone else's Nerd Font. Then:

```lua
local icon = require("mapal.icon")
icon.setup({ glyph = icon.logo })     -- the real mark at U+F8F0
```

Without it, the closest glyph the Nerd Font already has is used. Neither editor has an LSP yet, so neither
resolves names the way the compiler does (ADR-0008).

---

## How the project works

Everything is written down, including what turned out wrong:

- [`docs/STATUS.md`](docs/STATUS.md) - what is built, with test counts
- [`docs/decisions/`](docs/decisions/README.md) - every significant decision and the alternatives it rejected,
  indexed (ADR = Architecture Decision Record)
- [`docs/architecture/`](docs/architecture/) - how the compiler derives what it does
- [`docs/performance/`](docs/performance/) and [`benches/results/`](benches/results/README.md) - every benchmark:
  machine, method, evidence, failures
- [`docs/sessions/`](docs/sessions/) - dated log of every work session, mistakes included

**Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) - the model-first workflow, the measurement rules, and the
[open ADRs](docs/decisions/README.md) anyone can pick up (dynamic arrays, type generics, sum types, an
external-backend SDK, co-execution, `scan`). Recursion, modules and closures are not in the language yet and have
no decision record - writing one is the contribution that unblocks the code. Forks welcome. One house rule: a
change arrives with the evidence of what it did, and names the published numbers it moves.

---

## License

[Apache License 2.0 with the LLVM exception](LICENSE) - the license LLVM and Swift use, for the same reason.
Apache-2.0 carries an explicit patent grant; the exception keeps the runtime Mapal links into your binaries from
imposing attribution requirements on _your_ program's output.

<!----------------------------------{ Badges }---------------------------------->

<!-- The license badge is static because GitHub classifies Apache-2.0 WITH
     LLVM-exception as NOASSERTION, so shields' dynamic one renders the words
     "not identifiable by github". The exception is real; the badge says so. -->

[Badge Workflow]: https://github.com/LessComplexity/mapal/actions/workflows/ci.yml/badge.svg
[Badge License]: https://img.shields.io/badge/license-Apache--2.0_WITH_LLVM--exception-blue
[Badge Language]: https://img.shields.io/github/languages/top/LessComplexity/mapal
[Badge Pull Requests]: https://img.shields.io/github/issues-pr/LessComplexity/mapal
[Badge Issues]: https://img.shields.io/github/issues/LessComplexity/mapal
[Badge Numerics]: https://img.shields.io/badge/numerics-bounded-14b8a6

<!-----------------------------------{ Links }----------------------------------->

[Workflow]: https://github.com/LessComplexity/mapal/actions/workflows/ci.yml
[License]: LICENSE
[Pull Requests]: https://github.com/LessComplexity/mapal/pulls
[Issues]: https://github.com/LessComplexity/mapal/issues
