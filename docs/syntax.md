# Syntax

The language, form by form. Back to the [README](../README.md); to build and run the examples:
[getting started](getting-started.md).

## The core forms in one program

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

## Form by form

| form | what it says | decision |
| --- | --- | --- |
| `x -> f -> y;` | a pipeline: `x` flows through `f`, the result is bound as `y` | [ADR-0005](decisions/ADR-0005-flow-is-a-statement.md) |
| `2048 -> hidden_size;` | a top-level binding: immutable (no `const`; only `mut` values change), computed before the program runs when its dataflow reads no input | [ADR-0048](decisions/ADR-0048-top-level-bindings-compile-time-evaluation.md) |
| `type Matrix<rows, columns> { _: [bf16; rows * columns] }` | `_` is the type's default interpretation: with no other field the type is an alias, with others it reads as `_` wherever only that fits | [ADR-0048](decisions/ADR-0048-top-level-bindings-compile-time-evaluation.md) |
| `fn project<rows, columns>(…)` | size-generic: the sizes are inferred at each call, one fixed-size function per size tuple | [ADR-0047](decisions/ADR-0047-size-generic-functions.md) |
| `x -> exp`, `-> sqrt`, `-> sin`, `-> cos` | math builtins in pipeline position (no call syntax), bit-defined on the exact faces | [ADR-0042](decisions/ADR-0042-exp-builtin.md), [ADR-0046](decisions/ADR-0046-math-builtins-and-target-realizations.md) |
| `"file" -> read -> w: [bf16; n];` | load a value's raw bytes from a file: an array, or a whole struct of arrays | [ADR-0045](decisions/ADR-0045-read-builtin.md) |
| `bf16`, `f16` | narrow storage: one rounding in (`to_bf16`), exact widening out (`widen_f32`), f32 compute | [ADR-0043](decisions/ADR-0043-narrow-float-storage.md) |
| `n -> iota -> map { i -> … }` | a value per cell, every cell independent | |
| `(seed, xs) -> fold { acc, x -> … }` | combine values; on the default face the order is the compiler's | [ADR-0040](decisions/ADR-0040-default-float-face.md) |
| `(c) -> { -true-> a; -false-> b; }` | a guard: exactly one arm runs | |
| `mut s: T <- init; loop { … -> loop; }` | a loop: `mut` state, an arm that goes round again or leaves | |

Every decision record, with the alternatives it rejected: [`docs/decisions/`](decisions/README.md).

## A step of a real model

RMSNorm, one of the steps of a transformer, for any number of tokens (`hidden_size` and `rms_epsilon` are top-level
values of the model's file):

```flow
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

No threads, tiles or kernels in the source: the compiler derives that every `map` cell is independent, how the `fold`
combines, and where each value lives. The whole Llama-3.2-1B is one readable file,
[`examples/llama.mapal`](../examples/llama.mapal) (403 lines).
