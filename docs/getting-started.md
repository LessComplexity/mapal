# Getting started

Build Mapal, run the examples, test the compiler against its interpreter, and set up an editor. Back to the
[README](../README.md).

## Requirements

Needs a recent Rust toolchain and `clang`; the GPU backend needs an NVIDIA GPU and CUDA. The GPU tests skip without
a CUDA toolchain.

## Build, test and run

```sh
# the test suite = the correctness argument
cargo test --workspace --release

# the CLI and the runtime it links against
cargo build --release -p mapal-cli -p mapal-rt
target/release/mapal run --interp examples/pipeline.mapal   # the interpreter, the reference
target/release/mapal run examples/fir.mapal                 # compiled for this machine
target/release/mapal test examples/fir.mapal                # compiled vs the interpreter, byte for byte
```

What CI runs, and how to make sure the LLVM differential suite runs instead of skipping:
[CONTRIBUTING](../CONTRIBUTING.md#setup-and-the-gates).

## The Llama example on a CUDA GPU

The weights are exported outside the repository; `MAPAL_DATA_DIR` names the directory a program's `read` loads its
files from.

```sh
# the Llama example on a CUDA GPU (weights exported outside the repository)
python3 benches/model/export_weights.py
MAPAL_DATA_DIR=~/.cache/mapal/models/llama-3.2-1b-instruct/mapal \
    target/release/mapal run --target=native-gpu examples/llama.mapal
```

## Which example to start with

Start in [`examples/`](../examples/): `softmax.mapal` for a first look (the README's), `pipeline.mapal` for syntax,
`sepia.mapal` for most of the language, `fir.mapal` for a loop, `llama.mapal` for a whole model. The forms
themselves, one by one: [syntax](syntax.md).

## Interactive

`mapal repl` is a prompt on the interpreter. Type a function, a value or an expression; it answers at once.
A definition is kept (typing it again replaces it), a value you name with `-> name` is kept, and an expression is
shown and forgotten. A line that fails prints what is wrong, pointing at what you typed, and changes nothing.
`:help` lists the commands: `:load` a file's functions, `:list` or `:save` the session as a formatted program,
`:reset`, `:bound`, `:quit`.

```text
mapal> :load examples/softmax.mapal
loaded 1 item(s) from examples/softmax.mapal: softmax
mapal> [1.0, 2.0, 3.0, 4.0] -> softmax -> p
p = [0.032058604, 0.08714432, 0.23688284, 0.6439143]
mapal> p[3] * 2.0
1.2878286
```

## The command-line tool

`mapal build`, `run`, `test`, `dump-ir`, `fmt`, `repl`, `lsp`:

| command | what it does |
| --- | --- |
| `mapal run file.mapal` | compile for this machine and run it; `--interp` runs the interpreter, the reference |
| `mapal build file.mapal -o exe` | compile and link an executable (`--emit-only`: the LLVM IR, and the PTX for a GPU) |
| `mapal test file.mapal` | the compiled program against the interpreter, byte for byte |
| `mapal dump-ir file.mapal` | the program's graph at a stage (`--stage=lowered`, `rewritten`, `llvm`, `device`) |
| `mapal dump-ir --mermaid file.mapal` | the program as a Mermaid string diagram (steps are boxes, named values are wires); `--mermaid=category` draws it as a commutative diagram, `--mermaid=graph` is the stored graph, `=flow` the named values; `string` / `category` take `,condensed` |
| `mapal fmt files…` | format source ([below](#formatting)) |
| `mapal repl` | an interactive session on the interpreter ([below](#interactive)) |
| `mapal lsp` | the language server, over stdin / stdout (editors start it) |

`--target=native-gpu` compiles for the CUDA GPU in this machine. `mapal` with no arguments lists every command and
option.

## Formatting

```sh
target/release/mapal fmt path/to/program.mapal           # format in place
target/release/mapal fmt --check path/to/program.mapal   # check without writing
target/release/mapal fmt --stdout path/to/program.mapal  # preview without writing
```

The formatter uses four spaces and expands brace bodies onto multiple lines, preserving comments and literal
spelling. `mapal fmt -` reads stdin and writes stdout, so editors and generators share the same style. Invalid
syntax is reported without replacing the source. LLVM and GPU tools are not needed for formatting.

## Editors

Both editors get the language server (`mapal lsp`: go to definition, references, hover, rename,
completion, diagnostics, semantic highlighting), highlighting, formatting through `mapal fmt`, and a file
icon. Neither is published to a registry, so both load from disk.

Formatting uses unsaved buffer text and supports format-on-save through the editor's configuration; see the
plugin READMEs for setup: [Neovim](../editors/nvim/README.md) and [VS Code / Cursor](../editors/vscode/README.md).
