# ADR-0048: Top-level bindings, compile-time evaluation by dataflow, `_` as a type's default interpretation, size-generic types

Date: 2026-11-25 · Status: **accepted-for-implementation** (implemented on `model/language`; the surface is the user's
design and fixed: "`2048 -> hidden_size;` … the compiler, using the dataflow of a non-mut object can determine if it is
compile-time-computable or not, no? then just compute it beforehand").
Builds on: ADR-0005 (flows are statements), ADR-0013 (the IO token: effects are token-threaded), ADR-0040 (the
interpreter is the oracle, exact on every face), ADR-0047 (size-generic functions: size expressions, inference,
monomorphization). Siblings: ADR-0045 (`read`), ADR-0046 (`sqrt` / `sin` / `cos`).

## Context

1. **Numbers repeated as literals.** The readable Llama model (`llama_tiny_ideal.mapal`, the model scout) writes
   `2048`, `64`, `8192` in every signature and body; the real-size program had to be rendered from a template by a
   Python generator because the language could not name a size. ADR-0047 made functions size-generic, but a model's
   sizes are *one* set of numbers shared by every function — a size parameter per function is the wrong scope.
2. **The same type spelled out everywhere.** `[f32; 2048]` (a hidden state) and `[bf16; 2048 * 2048]` (a weight
   matrix) are the model's vocabulary; written structurally they say nothing about what they are.
3. **Values known before the program runs are computed while it runs.** A RoPE frequency table, `1 / sqrt(head_size)`,
   the head count's square root, a norm's constant — every such value has a dataflow cone with no input and no effect,
   yet it is recomputed on every call (the existing constant folding stops at one scalar operation: no map, fold, call
   or loop).

## Decision

**D1 — Top-level bindings.** A chain statement at the top level binds program-wide values:

```
2048 -> hidden_size;
hidden_size / query_heads -> head_size;
0.00001 -> rms_epsilon;
32 -> query_heads;
```

There is no `const` keyword: everything is immutable unless `mut`, and a top-level binding is never `mut` (**L1801**).
A top-level chain is any chain (literals, arithmetic, calls of pure functions, `iota`, `map`, `fold`, …) whose binding
stages (`-> name` that is no function or builtin, `-> name: T`) name the values; it must bind at least one (**L1805**)
and has no `ret` / `loop` stage (**L1803**); any other statement at the top level stays ⟦P0012⟧. The bindings are
**order-independent**: item `i` reads item `j` when its chain — or a function or type it reaches — names one of `j`'s
names, and the item graph must be acyclic (**L1807**, naming the cycle; a size in a function's signature counts). A
top-level name is bound once and is not a function's (**L1804**) or a reserved name (L1009). **No effects** at the top
level (**L1806**): `print`, `println`, `time`, `read` and effectful functions belong in a function. **A local may not
reuse a top-level name** (**L1802**): a parameter, a binding, a map / fold parameter, a function's or a type's size
parameter. The formatter keeps top-level chains as written (one per line, a blank line before the next `fn` / `type`);
editors highlight them as bindings.

**D2 — Where a top-level value is visible.** Everywhere: every function body (map / fold bodies included), every size
position (an array length, a size-generic type's sizes), every top-level chain.

**D3 — Compile-time evaluation by dataflow.** *A non-`mut` value whose dataflow cone holds no effect and no run-time
input is computed before the program runs, by the interpreter — the oracle, so the value is exact on every face — and
replaced by its value.* Two places apply the rule:

- *Top-level values, during lowering.* Their cones have no input and no effect by D1. Lowering computes them in
  rounds: each round lowers a synthetic program — the types and functions whose sizes are known so far and a `main`
  holding the bindings that can be lowered, in dependency order — and runs each scalar binding's cone on the
  interpreter (`mapal_interp::eval_values`, budget `2²⁴` steps). A bare integer literal needs no evaluation. A scalar
  that comes out is a **literal** from then on: unannotated, a literal in every respect (an integer's width from
  context, `Ty::index_ty` past 2³¹; a float's from context — a one-literal binding `0.1 -> tenth;` is that literal's
  own text at each use, a computed float its `f64` value rounded once to the use's format); annotated
  (`0.5 -> half: f32;`), a constant of its type. A positive integer is usable as a **size** (ADR-0047 D2), so
  `[f32; hidden_size]`, `fn f(x: [f32; head_size])`, `Matrix<hidden_size, 4>`. **Sizes must come out**: a size that
  names a top-level value which is not an integer ≥ 1 computed this way (a float, a negative, a binding whose
  evaluation traps or runs out of budget) is **L1808**.
  Every other top-level value (an array, a tuple, a struct, a scalar that did not come out) is realized where a
  function reads it: its chain is emitted at the function's outermost scope (map / fold bodies capture it like any
  enclosing value) — and the graph rule below decides whether its value is embedded or computed at run time.
- *Every value, on the graph* (`mapal_rewrite::fold_static`, in the shipped pipeline before `rewrite`; OFF arm
  `--static=off` / `MAPAL_STATIC=off`). The fact is **`static_plan`** (mapal-ir, stored on the graph, one per
  function): a value is **computable** when `unchanged_plan(f).inputs` is 0 — it is a function of no input slot, and
  the IO token is an input slot, so every `print` / `time` / `read` result and everything computed from one is not —
  and, for a loop's value, when the whole loop is (its every decide, advance and exit value: the oracle runs a loop as
  a unit). Its **roots** are the computable values a run-time value reads (or that leave the function); a product
  assembled slot by slot is not a root, its computable components are. Each root carries its bytes, cells and the
  oracle's work over its cone (a map or fold `n` times its body's, a call its callee's; unknown through a loop), and
  the **rule — bytes against work** — realizes it:

  | realization | when | what happens |
  | --- | --- | --- |
  | `Embed` | bytes ≤ 16 KiB, work ≤ 2²⁴ steps, work > cells | the oracle computes it; a scalar becomes a constant, a product a pack of constants; its cone, evaluated without a trap, is dropped |
  | `Attempt` | bytes ≤ 16 KiB, work unknown (a loop) | evaluated under 2²⁴ steps; embedded when it comes out |
  | `Compute` | otherwise: too large to embed, too much work while compiling, or no cheaper as a value than as its law — an array with a closed-form element law (`elem_plan`: the backends rebuild it where it is read, a device site where it runs), or no more work than cells (`iota`, `fill`, a pack of literals) | stays computed at run time — once, where the existing facts (`unchanged_plan`, `call_keep`, `elem_plan`) place it — and its computable operands are roots in turn |

  A root whose evaluation traps stays computed at run time (the trap is the program's meaning). 16 KiB bounds the
  realization (one constant and one store per cell in the emitted code); 2²⁴ bounds the compiler's own time. Neither
  is a hardware property.

**D4 — Size-generic types.** `type Matrix<rows, columns> { _: [bf16; rows * columns] }` declares size parameters
after the type name (ADR-0047 D1's list; each declared once, **L1702**; used by a field, **L1706**; not a top-level
name, L1802); a field's sizes are size expressions over them and the top-level integers. A use states its sizes,
`Matrix<hidden_size, hidden_size>` — one size expression per parameter (**L1809** for a wrong count, for a generic
type used bare, for sizes on a fixed type); a type argument that is a type (`Vec<f32>`, `Result<T, E>`) stays ⟦P0103⟧.
An instance is a struct named `Matrix<2048, 2048>` (or, for an alias, its `_` type). In a size-generic function's
signature a generic type's sizes are inferred like an array's (ADR-0047 D3: the declaration's field types with the
arguments substituted are matched against the argument); a struct literal of a generic type takes its sizes from its
fields' types. The formatter writes the lists tight; editors highlight them.

**D5 — `_`, a type's default interpretation.** A field named `_`:

- *Only `_`* — `type Hidden { _: [f32; hidden_size] }` — is a **transparent two-way alias**: the type *is* the `_`
  field's type (zero cost: no struct in the IR; `Hidden { _: x }` is `x`).
- *With other fields* — `type Scaled { _: [f32; 3], scale: f32 }` — a value used where its own type does not fit but
  its `_`'s type does **reads as `x._`** (one way): a call's argument (each component of a tuple argument), a generic
  call's inference, a binding's annotation, an index `x[i]`, a `map` source, a `fold`'s array, a field `x.f` that is
  `_`'s field. **When both fit, the value is taken as is.** Chains are followed deterministically: `x`, then `x._`,
  then `x._._`, … the first that fits. A mismatch after the chain is reported where the value is used (L1201) with
  the type it has.

**D6 — The interpreter.** Unchanged in meaning: a top-level value is the value of its chain; a `_` read is a
projection; a generic type's instance a fixed struct. It gains one entry, `eval_values(ir, f, roots, budget)`: the
values of objects of `f` computed from their cones alone (no input, no IO token in the cone), the rest of `f` never
run — what both halves of D3 call.

## Alternatives rejected (by the user)

| alternative | why not |
| --- | --- |
| A `const` keyword (`const HIDDEN = 2048;`) | everything is immutable unless `mut`: a keyword would mark the default; whether a value is known before the program runs is a property of its dataflow, which the compiler can see — the user does not declare it |
| Pipeline-form declarations with a keyword (`let`, `def` at the top level) | the binding form already exists (`expr -> name;`); a second form for the same binding is two languages |
| Constants without type aliases | the model's types (`Hidden`, `Matrix<…>`) are as repeated as its numbers; naming the sizes but not the types leaves every signature structural |
| `type Hidden = [f32; 2048];` alias syntax | `_` gives one mechanism for both the transparent alias (only `_`) and the default interpretation of a richer struct; a second declaration form would split them |
| Compile-time evaluation only of a marked subset (`comptime`) | the cone already says it; a marker can only disagree with the dataflow |
| Always embedding every computable value | a 500 MB weight table built from a law is cheaper as its law (computed where read, or once); compile time is a resource too — the rule weighs bytes against work |

## Consequences

- The readable model names its sizes once and its types by role; `project<rows, columns>` takes `Matrix<rows, columns>`.
- A program whose operands are laws now computes its printed values at compile time when they are cheap enough —
  the benchmark suite's small programs among them (measured: `model-ready-language.md`). A benchmark meant to time a
  kernel takes run-time inputs (`read`) or the OFF arm; one test (`gpu_profile_build_writes_device_unit`) now passes
  `--static=off` because its whole 64³ program is computable.
- `unchanged_plan`'s input masks were wrong for a function whose input is a tuple read whole (not through slot
  projections): such a value counted as the same at every call. Fixed: a whole-input read depends on every slot.
- Found on the way, recorded as open (not ADR-0048's): a float literal in a top-level `map` body is typed standing
  alone (`f64`), as in a function — an annotation on the binding does not reach the body (the scout's literal-width
  item).

## What would falsify it

A program whose value differs between compile-time evaluation on and off (`fold_static` is checked against the
interpreter on generated programs, native CPU and every emulator mode); a top-level program that does not lower to the
IR of its twin written with literals (generated twins); a size accepted that does not come out of evaluation; lowering
time superlinear in the number of top-level bindings.
