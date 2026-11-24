# ADR-0045: `read` — a pipeline builtin that loads an array from a file

Date: 2026-11-25 · Status: **accepted-for-implementation** (implemented on `feat/read-builtin`; the user chose the
surface: `"weights/embedding.bin" -> read -> token_embedding: [bf16; 262668288];` — the standard pipeline form,
never call syntax).
Builds on: ADR-0013 (the IO token, effects as token-threaded morphisms), ADR-0015 (the print family; string
literals feed only effects), ADR-0029 (bare-stage builtins resolved by name), ADR-0043 (`bf16` / `f16` storage), the
`time` builtin (plan-time-builtin: a value-producing effect, `(IoToken, value)` pairs).

## Context

1. **A model needs its weights.** Every model-scale benchmark program (`benches/algorithms/pt_llm_*`) synthesizes its
   weights in-program from closed-form laws (`25165824 -> iota -> i; i -> map { t -> (t % 103 * 11 + 57) % 103 … }`).
   A real model — an LLM with its embedding table, attention and MLP weights — cannot be written that way: its
   parameters are data, hundreds of megabytes to gigabytes of it, produced by training and stored in files.
2. **Synthetic weights are the wrong benchmark.** A law-built array is an `elem_plan` the compiler can rebuild where it
   is read (step 3b elides such arrays; a device site builds them "where the site runs, never uploaded"). A program
   whose weights come from a law measures something no inference engine does — and the comparison with torch /
   cuBLAS, which load real tensors, is skewed in whatever direction the law happens to favour.
3. **There was no input at all.** `main` takes no parameters (L1002) and the only effects were `print` / `println`
   and `time`: a program could not observe anything outside its source text.

## Decision

**D1 — Surface: an effectful pipeline builtin.** `read` is resolved by name in pipeline position like `time`:

```
"weights/embedding.bin" -> read -> token_embedding: [bf16; 262668288];
```

The wire into `read` is the path: a **string literal** (the rule that string literals feed only `print` / `println`
now reads "only `print`, `println` and `read`"); anything else is **L1618**. The next stage is the binding whose
**annotation is required** and types the array; a missing annotation, a binding without one, or a type that is not a
one-dimensional array of a readable element type (D2) is **L1617**. `read` is a reserved name (L1009). It threads the
IO token, so it is legal in `main` and in effectful functions and illegal where every effect is: in map and fold
bodies (L1605), guard arms (L1404) and parallel fanout branches (T0201). There is no call syntax (`read("…")` is not
Mapal). The formatter keeps the stage as written.

**D2 — Element types and shape.** One-dimensional arrays of `i32`, `i64`, `u8`, `f32`, `f64`, `bf16`, `f16`
(`mapal_ir::read::readable`). fp8, `bool`, tuples, structs and nested arrays are L1617. Sizes past 2³¹ cells are
ordinary array types; their indices take `Ty::index_ty` (`i64`), as every other extent does.

**D3 — File format.** Raw little-endian elements, exactly `n · |T|` bytes (`|T|` = 4, 8, 1, 4, 8, 2, 2), no header.
A converter (numpy's `tofile`, `safetensors` slices) writes it; the compiler never parses a container format.

**D4 — Failures.** A file that cannot be opened (missing, a directory, unreadable) or whose length is not exactly
`n · |T|` fails the run, identically on every backend: stdout flushed, one line on stderr, **exit 103** (traps are
101, GPU runtime failures 102):

```
mapal: read failed: <resolved path>: expected <n·|T|> bytes, found <actual>
mapal: read failed: <resolved path>: <operating-system error>          (e.g. "No such file or directory (os error 2)")
```

A read failure observes like a print: every output and every trap the oracle produces before the read is produced
first; nothing after it runs. (The interpreter's outcome is `Outcome::ReadFailed(message)`; like a trap its stdout
is not compared — `native_observable` gives `(None, 103)`.)

**D5 — Path resolution, decided at run time from a base recorded at compile time.**

1. An **absolute** literal is used as written.
2. A relative literal is joined to **`$MAPAL_DATA_DIR`** when that variable is set and non-empty;
3. else to the **program's source directory**, which the driver records when it compiles: `mapal` canonicalizes the
   source file's directory and stores it on the graph (`CategoryIr::with_data_dir` / `data_dir`, carried through
   every rewrite); the LLVM emitter bakes it into the binary (`@.mapal_data_dir`) and passes it with the literal to
   the runtime at each read;
4. else (a graph built without a source file: a test, a string handed to `lower`) to the **working directory**.

So a built binary run from any directory finds the data beside its source (`built_program_runs_from_another_directory`);
moving the binary to another machine, or the data elsewhere, takes `MAPAL_DATA_DIR`. The rule is one function,
`mapal_ir::read::resolve_with`; `mapal-rt` (which links into every program and depends on nothing) carries the same
function, and its tests replay the IR's cases against it.

**D6 — IR: one operation.** `Operation::Read : (IoToken × Str) → (IoToken × [T; n])`. Its source is the print
pair's shape — the token and the path literal, a `Str` constant (I9s now admits the `(IoToken, Str)` pair feeding
`Print` or `Read`); its target is the `time` pair's shape — the next token and the array, which lowering projects
apart (slot 0 rebinds the token, slot 1 is the binding). `FnBuilder::read` (`IrError::InvalidRead`), validate,
mermaid, replay. Being token-threaded it is never reordered against other effects, folded, CSE'd or DCE'd.

**D7 — After the read, an immutable array like any other.** The facts see the array as made outside every loop:
`unchanged_plan` gives it no input slot (the same at every call — a literal path, a fixed base, a file the run does
not change) and the same value on every trip of a loop it is read in, so `call_keep` and residency treat it as they
treat any loop-invariant array. `access_plan` records the read as writing the array's bytes, and the
`(token, array)` pair and its array as one storage class (the token is erased; the file's bytes land in the array).
A program that changes the file while it runs is outside the semantics.

**D8 — Placement on the parallel host.** A task reading a value the host spine writes races that write (the hazard
`time`'s consumers were kept on the spine for). A model's weights cannot stay on the spine, so a read is a **task**
of `path_plan` — the `Read` and the `Proj` of its array, nothing else — and its readers depend on it by ordinary
dataflow edges. It is not pinned: having no inputs, it starts when the run launches, beside the work that does not
read it (a weight file loads while earlier, independent work computes). A read is a **trap site**
(`path_trap_capable`): in a task its failure zero-fills the array, is recorded at the read's topo position
(`mapal_read_task` → `mapal_par_trap`, kind 2, the message kept) and is delivered by the first checkpoint past it —
the run's earliest trap wins, so every output and every trap the oracle produces before the read comes first, and
nothing the oracle would not reach is printed. Between the failure and that checkpoint the read's readers compute on
zeros (a trap's readers compute on its guard's value the same way). A read left on the spine (inside an effectful
loop) is a checkpoint like a print and exits at once; so does every read of a sequential function and of a GPU
build's host. (A first cut pinned the read to the spine behind a checkpoint; a checkpoint hoisted to an earlier
value's boundary then waited for the pinned read the walk had not reached — a deadlock a generated program found.)

**D9 — Data movement (the evidence rule).** The facts require the file's bytes to move once into the array and,
on a GPU, once more into the array's device buffer — nothing else.

- **Native CPU.** One `read` from the page cache straight into the array's storage (`mapal_read` in `mapal-rt`):
  the `(token, array)` pair has no storage of its own (`prepare_storage` elides it) and the `Proj` of the array is a
  no-op, so the host IR copies the array nowhere (`read.rs` asserts no `llvm.memcpy` names it). Arrays past 256 KB
  are the entry function's heap blocks (`mapal_rt_alloc`, made resident at the prologue), so the page faults are
  not charged to the read or to a later `time` bracket. `mmap` was not taken: it would make the array's bytes the
  file's (a file changed during the run changes a value) and moves the page faults into the first kernel that
  reads them.
- **GPU.** The host read as above, then the residency machinery's own upload — **one** `mapal_gpu_h2d` into the
  array's device buffer, placed where any host-made array's upload is: after its producer and, the array being the
  same on every trip, before every loop that reads it. No new allocator, no new copy path. Measured on the RTX 5080
  (`benches/local/read_gpu.sh`, `local-rtx-5080/read-builtin.md`): a 1 GiB weight file moves 1 073 741 824 bytes
  once into the host array (111.7 ms, 9.6 GB/s from the page cache) and once to the device (41.7 ms), at 1 and at 3
  trips of the loop that reads it; the host IR has no upload inside the loop (`evidence.py` `move_facts`:
  `h2d_n = 0` per trip). A read written before a `time` bracket
  is uploaded before the bracket opens (the clock fence orders it).
- **Staged or pinned copies** were to be taken only where they measurably help; they do not. The pageable upload
  of 1 GiB runs at 25.7 GB/s (41.7 ms) on the RTX 5080 — the bus's pinned rate on that part — so a pinned source
  saves nothing per byte, and registering 1 GiB costs more than it could (the existing `pinned` rule, `resident.rs`
  `pin`: a copy that runs once is not pinned). What remains is overlap: the upload follows the 111.7 ms read, and a
  chunked read-and-upload of a device-only array would hide it (~27 % of the load, outside any bracket). Open: it
  needs the read to write the device buffer, a placement the residency machinery does not express.

**D10 — Backends.** The interpreter (the oracle) reads through `mapal_ir::read::load`. The LLVM backend's host —
the CPU build (sequential and the parallel entry), the GPU build's host and the GPU emulator's (the emulator
replaces only the device runtime) — calls `mapal_read` in `mapal-rt`. The CUDA C++ backend has no host read seam
and reports `Unsupported` (as it does for `time`). `mapal fmt` keeps the stage; `editors/nvim` and the VS Code
grammar highlight `read` as a builtin.

## Consequences

- Programs can load real tensors: a model's weights are data, read once, uploaded once, and then exactly as
  loop-invariant to every fact as an array built before the loop.
- Benchmarks can stop synthesizing weights. A law-built operand and the same values read from a file are now two
  programs the generator draws side by side (`testgen::read_chain_strategy`): the oracle gives both the same
  output; the compiled read program gives the oracle's.
- The graph carries one non-identity attribute, `data_dir`, that no fact reads; rewrites carry it.
- Exit code 103 joins the defined exits (0, 101, 102).

## Alternatives rejected

| alternative | why not |
| --- | --- |
| **Inputs to `main`** (`fn main(weights: [bf16; n])`, the driver binding arguments to files) | `main`'s shape is fixed (L1002) and the entry protocol, the oracle's seed and every backend's wrapper would change; a function's arguments are values with no source, so the binding of parameter to file would live outside the program — the program would not say what it reads, and ordering a read against prints and clocks would have no meaning |
| **Synthetic weights** (keep building them from laws) | not a model: a law is an `elem_plan` the compiler rebuilds where it is read, so it never measures loading, uploading or the bandwidth of real data, and it favours whatever the law makes easy |
| Call syntax `read("…")` | Mapal has no call syntax for builtins; the pipeline form is the language's |
| The type from the file (a header, or the file's size) | types are static; a file's size is known at run time only. The annotation states the type; the size check enforces it |
| A container format (`.npy`, `safetensors`) | a parser in every runtime for metadata the annotation already states; raw little-endian is exact and any converter writes it |
| `mmap` as the semantics | the array would be the file: a file changed during the run changes a value; the read's cost moves into whichever kernel first touches a page |
| A pure read (no token) | a read observes the world: it must be ordered against prints and clocks, and its failure must not fire speculatively |
| Resolving relative paths against the working directory only | a built binary would find its data only when run from one directory; the source directory is where the program says the data is |

## What would falsify it

A backend whose output differs from the interpreter's on a program that reads (golden files of every element type,
generated read programs); a failure whose exit code or message differs between backends; a GPU run whose trace
shows a read array uploaded more than once, or inside a loop; a host IR that copies a read array; a built program
that does not find its data when run from another directory.

## The interpreter

`Operation::Read` resolves the literal against `CategoryIr::data_dir` (D5), checks the length, and decodes the
little-endian elements into the array (`mapal_ir::read::{load, decode}`); a failure is `Outcome::ReadFailed` with
D4's message. `mapal run --interp` prints the message and exits 103.
