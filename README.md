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

**Describe how data flows. Mapal places it where it runs best: CPUs and NVIDIA GPUs today, and new hardware is described, not rewritten for.**

</div>

## A first look

```flow
// softmax: turn scores into probabilities that add up to 1.
fn softmax<n>(scores: [f32; n]) -> [f32; n] {
    n -> iota -> positions;
    (scores[0], positions) -> fold {
        top, i -> (scores[i] > top) -> {
            -true-> scores[i];
            -false-> top;
        }
    } -> largest;
    positions -> map {
        i -> scores[i] - largest -> exp
    } -> weights;
    (0.0, positions) -> fold {
        sum, i -> sum + weights[i]
    } -> total;
    positions -> map {
        i -> weights[i] / total
    } -> ret;
}

fn main() {
    [1.0, 2.0, 3.0, 4.0] -> softmax -> probabilities;
    probabilities[3] -> println;    // 0.6439143
}
```

`a -> f -> b` means "a flows through f and becomes b". A `map` does something to every item, and the items are
independent. A `fold` combines many values into one. That is all the program says, so the compiler is free to decide
the rest: what runs in parallel, where the data lives, and whether it runs on the CPU, the GPU or both. There are no
threads, kernels or tile sizes in the source.

A whole language model, Llama-3.2-1B, is one readable file:
[`examples/llama.mapal`](examples/llama.mapal).

## The ideas

- **The program is the flow.** You describe the data and how it moves, not the steps a machine should take.
- **Data composes like mathematics.** Values and the transformations between them are built on category theory, so
  the compiler can reason about a program exactly.
- **Describing the flow beats describing control.** Code that spells out the control is tuned for one machine. Code
  that states the flow can be compiled well for each machine.
- **The compiler knows both sides.** It knows your program (what depends on what) and the hardware (where data can
  live, what can compute it), and it matches the two. A new chip of a kind it already targets only needs its measured
  properties; a new kind of hardware needs its own backend. Programs never change.
- **Correct within a stated error.** Results are guaranteed within a precision you can see, not bit for bit, which
  leaves the compiler free to choose the fastest correct way.
- **Readable code is the target.** If the natural version of a program runs slower, the compiler gets fixed.

## Results

The Llama-3.2-1B model above, against the fastest of gpt-fast, vLLM, SGLang, TensorRT-LLM, llama.cpp and Hugging
Face on five NVIDIA GPUs. A bar below 1.0 means Mapal is faster.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/llm-bf16-dark.svg">
  <img alt="Llama-3.2-1B in bf16, like for like: Mapal's time over the fastest engine's, on five GPUs" src="benches/results/charts/llm-bf16-light.svg">
</picture>

- **Like for like (bf16):** Mapal is behind the fastest engine on all five GPUs: 1.06–1.57× slower per generated
  token and 1.05–2.37× slower to the first token. Closing this is the current work.
- **Memory:** 2.7–3.5 GB of GPU memory, level with the leanest engines and far below the serving engines.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/llm-f32-dark.svg">
  <img alt="Llama-3.2-1B, f32 math but not like for like: Mapal on the bf16 weights against engines on f32 weights" src="benches/results/charts/llm-f32-light.svg">
</picture>

- **f32 math, not like for like:** by default Mapal computes in f32 on the model's bf16 weights, as the model ships.
  The f32 engines convert the weights to f32 and read twice the bytes for every token. Against them Mapal is faster
  on every GPU, but mostly because it reads half the bytes, so this is not a same-precision result.

Across 119 kernels and model layers (matrix products, attention, convolution, reductions and more), each bar below
is one program against the best library or compiler. Mapal is faster on 30 to 57 of them per GPU, level on 8 to 43,
and slower on 30 to 81; the chart names the largest wins and losses.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benches/results/charts/suite-gpus-dark.svg">
  <img alt="Every program's Mapal time over the fastest competitor's, per GPU: wins and losses apart" src="benches/results/charts/suite-gpus-light.svg">
</picture>

All numbers, how they were measured, and every engine: [`docs/performance/results.md`](docs/performance/results.md).

## Where it stands

Working today: the language, an interpreter that defines it, a CPU backend, an NVIDIA GPU backend, and editor
support for Neovim and VS Code. Next: a model of the hardware the compiler places programs on, an AMD backend, a
PyTorch importer, and then hardware design (ASICs) from the same programs. A research compiler; no packaged release
yet. Details: [`docs/STATUS.md`](docs/STATUS.md) and [`docs/roadmap.md`](docs/roadmap.md).

## Try it

```sh
cargo build --release -p mapal-cli -p mapal-rt
target/release/mapal run examples/softmax.mapal
```

More, including the GPU, the tests and editor setup: [`docs/getting-started.md`](docs/getting-started.md).

## Learn more

- [The Language Exposition](https://lesscomplexity.github.io/mapal/language-exposition.html): Mapal by examples, then how the compiler works
- [Architecture Map](https://lesscomplexity.github.io/mapal/architecture-map.html): the compiler as data — every component, its categories and transformations, each linked to its code
- [Syntax](docs/syntax.md): the language, form by form
- [Decisions](docs/decisions/README.md): every design decision and what it replaced
- [Contributing](CONTRIBUTING.md)

## License

[Apache License 2.0 with the LLVM exception](LICENSE), the license LLVM and Swift use.

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
