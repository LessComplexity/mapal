<!-- markdownlint-disable MD033 MD041 -->
<div align = center>

<img src="assets/logo-wordmark.svg" width="320" alt="Mapal">

<br>

[![Badge Workflow]][Workflow]
[![Badge License]][License]
![Badge Language]
[![Badge Pull Requests]][Pull Requests]
[![Badge Issues]][Issues]
![Badge Determinism]<br>

<br>

</div>
<!-- markdownlint-enable MD033 -->

<!--toc:start-->

- [The idea](#the-idea)
  - [What the compiler works out for you](#what-the-compiler-works-out-for-you)
- [Results](#results)
  - [Matrix multiply, f32 - M4 Pro, threaded](#matrix-multiply-f32--m4-pro-threaded)
  - [Other shapes - M4 Pro, threaded](#other-shapes--m4-pro-threaded)
  - [The same shapes on an i9-14900F, threaded](#the-same-shapes-on-an-i9-14900f-threaded)
  - [matmul vs OpenBLAS on an i9-14900F, 1024² f32](#matmul-vs-openblas-on-an-i9-14900f-1024-f32)
  - [Three builds](#three-builds)
- [Status](#status)
- [Scope](#scope)
- [Trying it](#trying-it)
  - [Editor support](#editor-support)
- [What is next](#what-is-next)
- [How the project works](#how-the-project-works)
- [License](#license)

<!--toc:end-->

**A programming language that parallelizes your code for you.**

You write what you want computed. Mapal works out what can run at the same time, how to split it
across cores, how to use the vector and matrix hardware, and which bounds checks it can safely
drop. No threads, no locks, no intrinsics, no tuning pragmas - none of it appears in the source.

The same source runs on CPU and GPU, with FPGA planned.

```flow
fn main() {
    1048576 -> iota -> ts;
    ts -> map { t -> t * 2 } -> doubled;
    doubled[1048575] -> println;
}
```

That runs across every core of the machine. Nothing in the source says so.

---

## The idea

Fast code is usually fast because a human wrote machine details into it: tile sizes, thread
counts, vector widths, memory layouts. Those details are what stop it moving to other hardware.

Mapal keeps them out of your source. The program says what depends on what; the machine details are
**read off the machine you are compiling on** - cache sizes, line sizes, vector widths, whether
there is a matrix unit. Change the machine, keep the source.

### What the compiler works out for you

- What can run at the same time, and how to spread it across cores
- Which loops are matrix-shaped, and how to tile them for the cache and the vector registers
- Which data is read repeatedly, so it gets laid out once instead of re-fetched
- When an access pattern will fight the cache, and how to reorder it - from the machine's own
  cache geometry, which it detects
- Which bounds checks are provably unnecessary - and only those

How it does that: [`docs/architecture/`](docs/architecture/).

---

## Results

Method, machine specs, raw logs: [`docs/performance/`](docs/performance/).

### Matrix multiply, f32 - M4 Pro, threaded

|    N |       Mapal |   NumPy |                 |
| ---: | ----------: | ------: | --------------- |
| 1024 |     0.72 ms | 0.67 ms | NumPy 1.07×     |
| 2048 | **5.03 ms** | 5.33 ms | **Mapal 1.06×** |
| 4096 | **38.7 ms** | 44.1 ms | **Mapal 1.14×** |

3,556 GFLOP/s vs 3,115 at 4096, disjoint. Naive threaded C++: 33,439 ms.
Single-threaded NumPy is ~2× ahead
([why](docs/performance/cpu/memory/operand-residency-and-thermal-effects.md)).

### Other shapes - M4 Pro, threaded

| workload               | Mapal FMA off | Mapal FMA on | C++ naive-mt | NumPy 1t |
| ---------------------- | ------------: | -----------: | -----------: | -------: |
| FIR filter, 1M         |      0.305 ms | **0.289 ms** |         1.50 |     6.35 |
| conv2d 3×3, 1024²      |      0.084 ms | **0.083 ms** |         0.16 |     1.72 |
| saxpy, 1M              |      0.070 ms | **0.067 ms** |         0.12 |     0.18 |
| sum reduction, 1M      |  **0.560 ms** |     0.562 ms |         0.92 | **0.11** |
| transpose, 1024²       |      0.137 ms | **0.124 ms** |         0.26 |     0.86 |
| gather `x[idx[i]]`, 1M |      0.139 ms | **0.137 ms** |         0.16 |     2.19 |

Every row above moved this session, 1.00–1.31×, from scheduling alone — the kernels
are unchanged ([what changed](docs/performance/cpu/scheduling/roofline-dispatch-threshold.md)).

### The same shapes on an i9-14900F, threaded

| workload               | Mapal FMA off | Mapal FMA on | C++ naive-mt | NumPy 1t |
| ---------------------- | ------------: | -----------: | -----------: | -------: |
| FIR filter, 1M         |      0.225 ms | **0.187 ms** |         1.24 |     5.44 |
| conv2d 3×3, 1024²      |      0.118 ms | **0.084 ms** |         0.41 |     2.85 |
| saxpy, 1M              |      0.137 ms | **0.130 ms** |         0.42 |     0.47 |
| sum reduction, 1M      |  **0.394 ms** |     0.391 ms |         7.32 | **0.14** |
| transpose, 1024²       |      0.200 ms | **0.176 ms** |         0.47 |     2.22 |
| gather `x[idx[i]]`, 1M |      0.236 ms | **0.203 ms** |         0.48 |     1.33 |

Transpose single-threaded on this box: 2.42 → 1.09 ms, −55%, disjoint
([what changed](docs/performance/cpu/memory/l1-panel-conflicts.md)).

### matmul vs OpenBLAS on an i9-14900F, 1024² f32

|                     |    Mapal |    NumPy |         |
| ------------------- | -------: | -------: | ------- |
| single-threaded     | 14.88 ms | 12.21 ms | 1.22×   |
| threaded, 8 P-cores |  2.09 ms |  1.74 ms | 1.20×   |
| threaded, whole box |  1.52 ms |  1.50 ms | **tie** |

The whole-box tie is heterogeneity tolerance, not a better kernel - same box,
8 threads per row ([detail](docs/performance/matmul/history/timing-boundary-and-openblas-comparison.md)):

| 8 threads on…       |   Mapal | NumPy |               |
| ------------------- | ------: | ----: | ------------- |
| 8 E-cores (uniform) | 5.89 ms |  5.59 | NumPy 5%      |
| 8 P-cores (uniform) | 2.44 ms |  1.72 | **NumPy 41%** |
| 4 P + 4 E (mixed)   | 3.38 ms |  5.57 | **Mapal 65%** |

### Linear neural networks - M4 Pro, threaded

Forward pass only, square (batch = width = `d`), every layer `d×d`. Values are
bit-identical to NumPy's on every cell. NumPy is the better of its 1-thread and
multi-thread legs. One run, 41 interleaved cycles
([full table + noise spreads](docs/performance/cpu/scheduling/roofline-dispatch-threshold.md)).

| d | 4-layer net | NumPy | | 2 nets at once | NumPy | |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 128 | 0.081 ms | **0.023** | NumPy 3.5× | 0.076 ms | **0.023** | NumPy 3.3× |
| 256 | 0.261 ms | **0.092** | NumPy 2.8× | 0.192 ms | **0.091** | NumPy 2.1× |
| 512 | **0.633 ms** | 0.645 | tie | **0.473 ms** | 0.644 | **Mapal 1.36×** |
| 1024 | **2.930 ms** | 3.764 | **Mapal 1.28×** | **2.771 ms** | 4.064 | **Mapal 1.47×** |
| 2048 | **20.36 ms** | 24.97 | **Mapal 1.23×** | **19.68 ms** | 25.65 | **Mapal 1.30×** |

Two independent nets are the same total work as one 4-layer net. Nothing in the
source says "run these in parallel"
([why that is free](docs/performance/cpu/scheduling/independent-networks-and-dependency-cycles.md)).

| d | one 4-layer net | two 2-layer nets | gain from the graph alone |
| ---: | ---: | ---: | ---: |
| 256 | 0.261 ms | 0.192 ms | **1.36×** |
| 512 | 0.633 ms | 0.473 ms | **1.34×** |
| 1024 | 2.930 ms | 2.771 ms | 1.06× |

Small sizes are now a single-thread kernel gap, not a scheduling one
([breakdown](docs/performance/cpu/scheduling/dispatch-threshold.md)).

### GPU workloads - 4 GPUs, October 2026

Mapal time / the best of cuBLAS, cuBLASLt, CUTLASS, PyTorch eager and `torch.compile` (`_tf32` rows: torch with TF32 on);
below 1 = Mapal faster. Values pass on every program and arch. 54 of 120 cells at or below 1.0. † = carried from the
previous sweep (emission identical)
([end check, gaps and causes](docs/performance/evidence/recurrence-precision.md)).

| workload | RTX 4090 | RTX 5090 | H100 PCIe | RTX 3090 |
| --- | ---: | ---: | ---: | ---: |
| decode_1024_256 | 1.03 | 1.52 | 2.48 | 1.12 |
| convmc_8_64_56_3_tf32 | 2.07 | 1.36 | 2.38 | 1.69 |
| attn_1024_64 † | 1.97 | 1.39 | 2.00 | 1.77 |
| mlp2_2048_tf32 | 1.16 | 1.15 | 1.87 | 1.17 |
| mlp_2048_tf32 | 1.06 | 1.16 | 1.75 | 1.40 |
| convmc_8_64_56_3 | 1.66 | 1.04 | 1.29 | 1.68 |
| loopmm_1024_8 † | 1.31 | 1.25 | 1.12 | 1.62 |
| pj_m16_n2048_k7168 † | **0.86** | 1.21 | 1.35 | 1.56 |
| rt_mlp2_1024 | 1.14 | 1.20 | 1.45 | **0.99** |
| mlp2_2048 † | 1.01 | 1.12 | 1.39 | 1.03 |
| indep_2048 † | **0.99** | 1.08 | 1.35 | 1.09 |
| diamond_2048 † | **0.98** | 1.11 | 1.30 | 1.01 |
| smlp_4096_256_1024 † | 1.09 | 1.17 | 1.19 | 1.29 |
| fanout_2048 † | **0.97** | 1.08 | 1.23 | 1.02 |
| mlp_2048 † | **0.97** | 1.09 | 1.22 | 1.04 |
| rowsum_2048 | **0.93** | 1.08 | 1.21 | 1.11 |
| twin_2048 † | **0.95** | 1.05 | 1.21 | 1.01 |
| rt_mlp2_2048 | **0.75** | **1.00** | 1.17 | **0.88** |
| softmax_rows_4096 † | 1.03 | 1.09 | **0.97** | **0.96** |
| rowsum_4096 | **0.99** | 1.07 | **0.96** | **0.96** |
| epi_4096 † | **1.00** | 1.06 | **0.97** | **0.88** |
| ew_4096 † | **0.46** | **0.45** | 1.01 | **0.61** |
| loop_4096 † | **0.95** | **0.97** | **0.97** | **0.96** |
| firew_rowsum_2048x2048_16 † | **0.66** | **0.55** | **0.57** | **0.72** |
| conv_2048_5 † | **0.63** | **0.45** | **0.50** | **0.68** |
| fir_4194304_16 † | **0.61** | **0.51** | **0.56** | **0.63** |
| causal_1024_256 | **0.21** | **0.40** | **0.54** | **0.16** |
| prefill_1024_512_256 | **0.21** | **0.38** | **0.53** | **0.16** |
| ewconvew_2048_5 † | **0.53** | **0.51** | **0.51** | **0.53** |
| gen_1024_512_256 | **0.16** | **0.44** | **0.52** | **0.12** |

### Three builds

| build                                        | what you get                                                     |
| -------------------------------------------- | ---------------------------------------------------------------- |
| **FMA off** (default)                        | bit-identical to the interpreter, always                         |
| **FMA on** (`--contract`)                    | lets the hardware fuse multiply and add - faster, last bit moves |
| **SME** (`--contract --target=apple-m4-sme`) | same as FMA on, plus the matrix unit; needs an Apple M4 or newer |

---

## Status

|                        |                                                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Language core          | working - functions, pipelines, parallel fanout, guards, loops, `map`/`fold`/`zip`/`enumerate`, tuples, fixed arrays, `print` |
| Interpreter            | working - _defines_ the language; every backend is tested against it                                                          |
| CPU backend (LLVM)     | working - automatic threading, vectorization, cache blocking                                                                  |
| GPU backend (CUDA)     | working - value gates pass on RTX 4090, RTX 5090, H100 PCIe and RTX 3090, October 2026. No `time` builtin                     |
| FPGA backend (Verilog) | not started                                                                                                                   |
| Command-line tool      | **not built** - `mapal` prints "not yet implemented" and exits 1                                                              |
| Tests                  | 1,047; 161 are CUDA's and skip without `nvcc`. Green                                                                          |
| CI                     | `cargo fmt` + full suite on Linux and macOS, per push. Cannot pass vacuously - a skipped LLVM differential fails the run      |

**"Byte-identical" means:** every compiled program is checked to print exactly what the
interpreter prints - 10 examples plus 320 generated programs, at two optimization levels, 1,280
comparisons on every run. Changing the thread count does not change the answer.

Per-component state and test counts: [`docs/STATUS.md`](docs/STATUS.md).

Research compiler. No packaged release.

---

## Scope

Kernels are the proving ground: byte-identical output is easiest to check against code everyone
already has a fast version of.

**Not in the language yet:** recursion, sum types, pattern matching, strings beyond printing,
closures, dynamic arrays, modules. Exactly two effects. General-purpose in shape, not yet in
surface.

**Where it is going:** one program whose parts run on several processors at once - CPU and GPU
together - with the data movement between them handled by the compiler rather than by hand. Same
answer, byte for byte, however the work gets split.

---

## Trying it

Needs a recent Rust toolchain and `clang`.

```sh
# the test suite = the correctness argument
cargo test --workspace --release

# run a program on the interpreter
cargo run --release -p mapal-interp --example run -- examples/pipeline.mapal

# compile a program to a native binary
cargo build --release -p mapal-rt          # the runtime it links against
cargo run --release -p mapal-backend-llvm --example emit -- \
    examples/fir.mapal - --rewrite > fir.ll
clang -O2 fir.ll target/release/libmapal_rt.a -o fir && ./fir
```

Start in [`examples/`](examples/) - `pipeline.mapal` for syntax, `sepia.mapal` for most of the
language, `fir.mapal` for a loop.

### Editor support

Both editors get highlighting and a file icon. Neither is published to a registry, so both load
from disk.

|                 | Neovim                                     | VS Code                                      |
| --------------- | ------------------------------------------ | -------------------------------------------- |
| Highlighting    | Vimscript syntax file                      | TextMate grammar                             |
| File icon       | font glyph                                 | the real SVG logo                            |
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
code --install-extension editors/vscode/mapal-lang-0.1.0.vsix   # or `cursor`
```

Then restart. A `.mapal` file should show **Mapal** in the status bar. Copying the folder into
`~/.vscode/extensions` does _not_ work: VS Code reads `extensions.json` as its registry and
ignores unregistered directories silently.

**The logo as a terminal glyph.** The Rust and C++ marks in a file tree are font glyphs, not
images - Nerd Fonts ships those brand logos as characters. So Mapal's mark ships as a
single-glyph font, [`assets/font/MapalIcons.ttf`](assets/font/): install it and add it as a
terminal _fallback_ font, rather than us patching and redistributing someone else's Nerd Font.
Then:

```lua
local icon = require("mapal.icon")
icon.setup({ glyph = icon.logo })     -- the real mark at U+F8F0
```

Without it, the closest glyph the Nerd Font already has is used. Neither editor has an LSP yet,
so neither resolves names the way the compiler does (ADR-0008).

---

## What is next

1. **Close the single-threaded gap**, where NumPy is still ~2× ahead. Threaded is solved; this is a
   different bottleneck ([what we measured](docs/performance/cpu/memory/operand-residency-and-thermal-effects.md)).
2. **Pick the thread count from the program** instead of defaulting to every core
   ([plan](docs/components/backend-llvm/plans/plan-s32-deduced-scheduling.md)).
3. **Intel AMX**, so matrix units are not an Apple-only story.
4. **GPUs in earnest**, then running one program across several processors at once.

Done so far, and how each turned out, is in
[`docs/sessions/`](docs/sessions/) - including the things that were tried and did not work.

---

## How the project works

Everything is written down, including what turned out wrong:

- [`docs/STATUS.md`](docs/STATUS.md) - what is built, with test counts
- [`docs/decisions/`](docs/decisions/README.md) - every significant decision and the
  alternatives it rejected, indexed (ADR = Architecture Decision Record)
- [`docs/performance/`](docs/performance/) - every benchmark: machine, method, failures
- [`docs/sessions/`](docs/sessions/) - dated log of every work session, mistakes included

**Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) - the model-first workflow, the
measurement rules, and the [open ADRs](docs/decisions/README.md) anyone can pick up (dynamic
arrays, generics, sum types, an external-backend SDK, co-execution, `scan`). Recursion, modules
and closures are not in the language yet and have no decision record - writing one is the
contribution that unblocks the code. Forks welcome. One house rule: a change arrives with the evidence of what it
did, and names the published numbers it moves.

---

## License

[Apache License 2.0 with the LLVM exception](LICENSE) - the license LLVM and Swift use, for the
same reason. Apache-2.0 carries an explicit patent grant; the exception keeps the runtime Mapal
links into your binaries from imposing attribution requirements on _your_ program's output.

<!----------------------------------{ Badges }---------------------------------->

<!-- The license badge is static because GitHub classifies Apache-2.0 WITH
     LLVM-exception as NOASSERTION, so shields' dynamic one renders the words
     "not identifiable by github". The exception is real; the badge says so. -->

[Badge Workflow]: https://github.com/LessComplexity/mapal/actions/workflows/ci.yml/badge.svg
[Badge License]: https://img.shields.io/badge/license-Apache--2.0_WITH_LLVM--exception-blue
[Badge Language]: https://img.shields.io/github/languages/top/LessComplexity/mapal
[Badge Pull Requests]: https://img.shields.io/github/issues-pr/LessComplexity/mapal
[Badge Issues]: https://img.shields.io/github/issues/LessComplexity/mapal
[Badge Determinism]: https://img.shields.io/badge/output-byte--identical-14b8a6

<!-----------------------------------{ Links }----------------------------------->

[Workflow]: https://github.com/LessComplexity/mapal/actions/workflows/ci.yml
[License]: LICENSE
[Pull Requests]: https://github.com/LessComplexity/mapal/pulls
[Issues]: https://github.com/LessComplexity/mapal/issues
