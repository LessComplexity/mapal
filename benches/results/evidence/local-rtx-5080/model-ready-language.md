# local-rtx-5080 — model-ready language: `read`, math builtins, size generics merged; ADR-0048 (2026-11-25)

Branch `model/language` (from `fac7691b`): the three language branches merged — `feat/read-builtin` (ADR-0045),
`feat/math-builtins` (ADR-0046), `feat/size-generics` (ADR-0047) — and ADR-0048 implemented on top: top-level
bindings, compile-time evaluation by dataflow (the `static_plan` fact), `_` as a type's default interpretation,
size-generic types. Machine: Threadripper 9970X + RTX 5080, shared with other agents (CPU set 16-19,24-27,48-51,56-59,
`nice 10`); GPU runs held the bench lock. Raw output: `.claude/raw-evidence/local-rtx-5080/model-ready-language/`
(not in git).

## (A) The three branches together

- **Merge.** Code merged without conflicts; the shared tables conflicted only in docs (ADR index, CHANGES 9.3–9.6,
  mapal-as-implemented), the editors (Neovim builtin list, fixture and its line-numbered cases, the VS Code grammar's
  three builtin lists) and two code tables (`LCode`: L1617–L1618 beside L1701–L1707; `mapal-check` effects tests).
  Resolved as unions; the Neovim cases re-numbered to the merged fixture; ADR-0047's index row added (it had none).
- **What the combination broke.** One interaction: `read`'s array is the same at every call (`unchanged_plan` mask 0,
  ADR-0045 D7) — the mask ADR-0048 reads as "computable" — so a value computed from a file looked computable before
  the program runs. Fixed in `static_plan` (a `read`'s array is never computable; tested: `fold_static` leaves a
  read-fed fold, folds `sqrt` of a top-level size).
- **Self-check, adversarial small programs** (`adversarial/`): every `read` element type incl. `-0.0`, a subnormal,
  `i32::MIN`, `i64::MAX`, `u8` 255, `bf16`; a read inside a loop and in a helper; a short file and a missing file
  (exit 103, identical message); `sqrt` / `sin` / `cos` at `±0`, a subnormal, `3·10³⁸`, a negative (NaN), `−10³⁰`,
  `f64` at 10³⁰⁰ (Payne–Hanek); math in kernels (a map and a fold-per-row); size inference by bisection (`n * n`),
  `n + 2`, a two-size product, a generic in a loop, sizes past 2¹⁶; generic types over `_` and a `read`. **9 programs ×
  {native CPU, RTX 5080} × {compile-time evaluation on, off}: 36/36 `mapal test` PASS** (fixed face, bytes against the
  interpreter); values checked by hand / against Python.
- **Tests on the merged tree** (front end, IR, interpreter, rewrite, CLI, runtime; `--no-fail-fast`): all pass. The
  LLVM backend suite: see the test table below.

## (B) ADR-0048

`docs/decisions/ADR-0048-top-level-bindings-compile-time-evaluation.md`. Facts / rules / language:

| piece | where | what |
| --- | --- | --- |
| top-level bindings | `mapal-syntax` (`Item::Bind`), `mapal-lower/src/globals.rs` | `2048 -> hidden_size;`; dependency graph through functions and types; L1801–L1807 |
| compile-time evaluation, top level | `mapal-lower` `evaluate_globals` | synthetic `main` per dependency round, `mapal_interp::eval_values`; scalars become literals, positive integers sizes; L1808 |
| `static_plan` (new fact) | `mapal-ir/src/algo/statics.rs`, stored on the graph | computable = `unchanged_plan.inputs` 0, no `read`, a loop only whole; roots; bytes / cells / work (maps, folds, calls, counted loops by `trip_split`); `Embed` (≤ 1 KiB, ≤ 2²⁰ steps, work > cells, no `elem_plan` law) / `Attempt` (uncounted loop, 2¹⁸ steps once) / `Compute` |
| `fold_static` | `mapal-rewrite/src/statics.rs`; CLI before `rewrite`; OFF `--static=off` / `MAPAL_STATIC=off` | oracle values replace embedded roots (`constify`, new `embed` channel); evaluated cones dropped, whole loops too (new `drop_loops` channel) |
| `_` | `tys.rs`, `emit.rs` (`read_default`, `coerce`), `typing.rs` | only `_`: transparent alias; else one-way read where the own type does not fit (call args per component, generic inference, annotations, index, map/fold, fields); errors say when `_` applied |
| size-generic types | parser (`TypeDecl.sizes`, `TyKind::Generic`), `tys.rs`, `sizes.rs` | `type Matrix<rows, columns>`, `Matrix<64, n>`; struct literals infer sizes; generic functions infer through them; L1809 |
| map literal width | `typing.rs` | a map whose body value is an unpinned literal settles its width at the end of the walk (`map { i -> 1.0 } -> t: [f32; 4]` was `[f64; 4]` + L1201) |
| `unchanged_plan` fix | `algo/unchanged.rs` | a whole-input read of a tuple input depended on no slot (counted "same at every call"); now on every slot |
| formatter, editors | `format.rs`, `editors/nvim` | tight `type T<a, b>` / `T<a, b>` lists, no line break at their commas; `_`, size args highlighted (6 new cases) |

**Tests.** Syntax 5 (`top_level.rs`), lower 16 (every L1801–L1809, L1702/L1706/L1802 for types, `_` both ways and its
error note, generic types, literal widths), rewrite 7 (`static_eval.rs`: the rule's realizations, a trap left to run,
a counted loop priced and dropped, an uncounted loop attempted, a struct embedded; **384 generated programs of the
general generator** — closed/open, with/without traps: 101 changed, 113 roots embedded, every outcome kept),
**96 generated top-level programs** (`testgen_tops.rs`: 835 bindings, 295 computed, 576 `Vec<…>` alias uses, 268 `_`
reads, 244 table reads; the pure rendering lowers to the plain program's IR, names/spans aside; the full one computes
its values on the interpreter, after `rewrite` and after `fold_static` + `rewrite`; 840 roots embedded), **48 generated
top-level programs on the native CPU and the emulated device in 3 modes, compile-time evaluation on and off**
(`backends/llvm/tests/top_level.rs`), Neovim 6 cases.

**The suite under compile-time evaluation** (`suite_static.txt`, 308 programs of `benches/algorithms`): 7 change — the
small twins `prefill_16_8_8`, `px_convmc_2_4_6_3{,_tf32}`, `px_ew_128`, `px_ewc_128`, `px_ewconvew_32_5`,
`px_firew_rowsum_32x32_16` (their printed values are computed at compile time); 301 unchanged (3541 roots `Compute`:
laws, too large, too much work). Front-end time summed over the suite 17.8 s (debug build), max 0.71 s.
Calibration of the rule on the way: a 2²⁴-step budget cost 40 s on `px_mlp2_64_f64` (the oracle runs ≈ 10⁶ steps/s
on array programs) → 2²⁰; failed attempts retried per root → once; a 16 KiB embed cap added 12 s to the real-size
Llama's build (2063 packed cells) → 1 KiB; a law-built array embedded as 16 KiB of constants replaced a law the
backends rebuild where read → arrays with an `elem_plan` law stay laws.

## The scout's programs, before / after

The readable model scout rendered one text per size from `llama.mapal.tmpl` with a Python generator (no named
constants). `llama/to_tops.py` renders it once with every placeholder a top-level binding (9 base sizes, 26 computed
sizes, 7 float constants via `sqrt` in f64, the prompt); the tiny and the real program now differ only in those 9
numbers, the prompt and one decode line per layer (the layer loop inside the decode loop is L1504, still open).

| program | front end (lower+eval+rewrite, release) | build (native-gpu) | run on RTX 5080 | output |
| --- | --- | --- | --- | --- |
| tiny, scout text, static on | 0.23 s (off 0.17 s; 6 roots, 335 cells embedded) | 2.6 s | 1.0 s (CUDA start) | 8 tokens = interpreter; logits within 6.1·10⁻⁶ of it |
| tiny, top-level text | 0.22 s (same 6 roots) | 2.5 s | 0.34 s | interpreter output **byte-identical** to the scout text's (8 tokens, 256 logits); GPU logits within 6.3·10⁻⁶ |
| real, scout text, static off | 1.13 s | 5.9 s | 23.1 s | 32 tokens = the scout's (`98006 98006 118186 …`) |
| real, scout text, static on, 16 KiB cap (calibration) | 1.12 s (5 roots, 2063 cells) | 18.3 s | 24.5 s | same tokens |
| real, top-level text, static on (1 KiB cap, shipped) | 1.08 s (4 roots, 15 cells) | 8.1 s | 24.5 s | same tokens, same output bytes as static off |

Every real-size run moved 22.0 GB to the device in 1689 copies (≈ 0.9 s) with 2160–2460 launches and 130–165 ms of
kernels; wall 22.6–24.5 s on the shared machine (the scout measured 18.6–19.6 s on a quieter one). The language work
does not change where the per-element maps and 1-row projections run — that is the placement step's (`model/placement`)
— so the real-size time is unchanged within noise; what changed is the program text (one source for every size,
sizes by name) and that values computable before the run are computed by the compiler.

## Open

- A layer loop inside the decode loop (L1504) — the real program still lists one decode line per layer.
- Embedding as a constant global (one load, no per-cell stores) would let values past 1 KiB embed; the packed-constant
  realization is the only one today.
- A fold's seed literal and an array literal's elements read before their width is pinned stay standing-alone typed.
- The native CPU build of both scout programs segfaults at run time (CPU task ordering with the host spine — the
  scout's blocker 3, fixed on `model/structure` `40553be9`, not on this branch); GPU builds run.
- Benchmarks meant to time a kernel on law-built inputs: the 7 changed twins above now print compile-time values; a
  harness that wants their kernels passes `--static=off` (or reads its inputs).

## For integration

**CLAUDE.md facts table** — add a row:

| query | what it proves |
| --- | --- |
| `static_plan` (`StaticPlan`: `fixed`, `roots` → `StaticRoot { bytes, cells, work, realize }`, `work`; `Realize::{Embed, Attempt, Compute}`) | ADR-0048 D3: the values computable before the program runs (`unchanged_plan.inputs` 0, no `read` in the cone, a loop only whole); the roots a run-time value reads; each root realized by bytes against work — embedded by the oracle (≤ `EMBED_BYTES` 1 KiB, ≤ `EVAL_WORK` 2²⁰ steps, more steps than cells, no `elem_plan` law), attempted (an uncounted loop, `ATTEMPT_WORK` 2¹⁸) or computed at run time; work through maps, folds, calls and counted loops (`trip_split`) |

**docs/tailored-fixes.md** — nothing: no tailored fix was added.

**Merge notes.** `model/structure` (`1f682344`) also touches literal widths in `typing.rs` and products of arrays in
the backend; this branch's typing change is the map-body literal only (`pending_maps`). `model/placement` carries the
scout's two local fixes; this branch carries none of them (`inline_return` is ADR-0047's, merged).
