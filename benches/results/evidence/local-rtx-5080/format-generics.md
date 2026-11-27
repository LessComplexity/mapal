# Format generics (ADR-0051): the rewrite lowers to the same IR, emits the same text

Date: 2026-11-26 · Branch `lang/format-generics` · Machine: the local RTX 5080 desktop.

ADR-0051 D6 removes `to_bf16`, `to_f16`, `to_f8e4m3`, `to_f8e5m2`, `widen_i64`, `widen_f32`, `widen_f64` and
rewrites every Mapal source: `to_X` → `to<X>`, `widen_X` → `widen<X>`, or `to<f32>` where D4 does not admit
`widen` (an `i32` the facts do not prove within ±2²⁴). The rewrite must lower to the same IR.

**Method.** *Old*: the compiler at `43a1b61d` (`mapal` CLI and the `emit` example, release) on the sources before
the rewrite. *New*: this branch on the rewritten sources. Each program's lowered IR (`mapal dump-ir --stage=lowered`,
`Debug` text with source spans removed — the spellings differ in length) and, for the benchmark sources, its host
and device text (`emit … --rewrite`, less the `; mapal.` evidence lines, as `benches/gpu/changed.py` compares) are
hashed and compared. The scratch tools: `irdump.py`, `emitdump.py`, `extract.py` (the literal programs of the Rust
tests), `rewrite.py` / `fixwiden.py` (the rewrite itself: every `widen<f32>` the new compiler reports as not proven,
L1614, becomes `to<f32>`).

## Same IR

| programs | count | equal | differ | invalid in both |
| --- | ---: | ---: | ---: | ---: |
| every `.mapal` file of the repository (examples, benches incl. the suite and the ladder, test fixtures, results) | 503 | 497 | 0 | 6 (parse-error fixtures, `examples/vector.mapal`) |
| literal programs of the Rust tests holding a conversion (`fn main` string literals without format placeholders) | 121 | 105 | 0 | 16 (rejection tests) |
| generated programs, `testgen_dump` (10 strategies × 60, same seeds) | 600 | 600 | 0 | 0 |

## Same emission

| profile | programs (`benches/**`, `examples/`) | equal host + device text | differ | invalid in both |
| --- | ---: | ---: | ---: | ---: |
| `rtx-5080` (`--host=x86-64-v3`) | 473 | 472 | 0 | 1 (`examples/vector.mapal`) |
| `x86-64-v3` | 473 | 472 | 0 | 1 |

The suite (`benches/local/suite.txt`, 35 programs) and the full model (`examples/llama.mapal`) are among them.
Every rewritten `.mapal` file is in `mapal fmt`'s layout (157 of the 478 were not at `43a1b61d`); the tables compare
the formatted files.

## The rewrite

| name | occurrences | | name | occurrences |
| --- | ---: | --- | --- | ---: |
| `widen_f32` | 2,617 | | `to_bf16` | 270 |
| `widen_f64` | 80 | | `to_f8e4m3` | 33 |
| `widen_i64` | 16 | | `to_f16` | 30 |
| | | | `to_f8e5m2` | 7 |

3,053 occurrences in `.mapal` (2,262), Rust (649: tests, the testgen generators and their emitted strings), Python
generators (65), docs (76) and one shell script. Of the `widen_f32` of an `i32`, the facts prove all but 11 exact
(`bounds_proof`'s `widen_range`, or the compile-time value); those are `to<f32>` now — the conversion they always
were (`sitofp`):

| site | why not exact |
| --- | --- |
| `benches/algorithms/e2e_trap.mapal`, `gen_tiles.py` (`1000 / (t - 77)`) | a quotient: no range |
| `examples/llama.mapal` ×2 (`pair * 2`, `positions[t]`), `llama_tiny_readable.mapal` | a function input, an array element |
| `hoisted_zero_merged.mapal` ×2, `element_view_projection.mapal` | a loop's carried state, an element read |
| `compute_type.rs` (`zz`), testgen `recur` (`100 / (it - 0)`) | a loop sum, a quotient |
| `golden.rs` `16777217 -> to<f32>` | the value is past 2²⁴: the old `widen_f32` rounded it |

One fact grew on the way: `bounds_proof` ranges `x % k` without a range of `x` (any `i32`, a wrapped product
included, leaves the remainder inside `(−|k|, |k|)`), which proves the ladder's `(t * 11 + 57) % 103 - 51` at
`t < 16384²` (the product wraps). The tables above include it: no lowered IR, host or device text moved.

Generator note: `testgen`'s tile laws drew a coefficient from the length of the program text so far; the rewrite's
one extra character per law (`widen<f32>`) would have changed every draw, so the draw takes the length less one per
line — the generated set is the one before the rewrite (the 600-program row).

## The readable Llama with a format parameter (separate commit)

`examples/llama.mapal` declares its key/value cache format once: `type KeyValueCache<element: float>`, the
functions over the cache take `element` (inferred from the cache they are given), `write_rows` rounds each new
element with `to<element>`, `attend` reads with `widen<f32>`, and `main` chooses the format in one line
(`0.0 -> to<f32> -> zero;`). `benches/model/variant.py --cache bf16` is now that one token (`to<bf16>`) instead of
six text substitutions. Run on the local RTX 5080, `prompt_length=128`, 128 generated tokens, built with
`mapal build --target=rtx-5080`:

| class | old (`43a1b61d`, `variant.py` then) | new (generic program) | tokens | logits | host `.ll` | device `.ll` / PTX |
| --- | --- | --- | --- | --- | --- | --- |
| default (f32 cache, default face) | `variant.py prompt_length=128` | same | 128 / 128 identical | within 3.7e-5 (a rerun of the old binary: 4.2e-5 — the default face's free order) | identical | identical |
| bf16 (`--compute bf16`, bf16 cache) | `variant.py prompt_length=128 --compute bf16` | same | 128 / 128 identical | identical | differs | differs |

At the program's own sizes (`prompt_length` 512) the generic `examples/llama.mapal` emits the old program's host and
device text byte for byte on `rtx-5080` and `x86-64-v3` (the table's method).

The bf16 class differs only in the order the attention folds read their two arrays: the old variant's substitution
read the cache before the query (`cache.keys[…] -> widen<f32> -> key; sum + query[…] * key`), the f32 program the
query first, and one generic source has one order (it keeps the f32 class's). The old compiler on the old bf16
program written in the generic program's order (`query[…] -> q;` first) gives the new host text, device text and PTX
byte for byte — the generic instance is the hand-specialized bf16 program.
