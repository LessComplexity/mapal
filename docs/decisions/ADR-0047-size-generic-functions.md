# ADR-0047: Size-generic functions — `fn f<n, m>(…)`, sizes inferred at each call, one fixed-size function per size tuple

Date: 2026-11-25 · Status: **accepted-for-implementation** (sprint step "Generics": "add … size-generic functions too …
we are cross architecture/hardware/and even asic"; the surface below is the user's choice and fixed).
Builds on: ADR-0005 (flows are statements, calls are tuple flows), ADR-0029 / ADR-0031 (`iota` / `fill` take a
literal count), ADR-0043 (narrow storage). Siblings: ADR-0045 (`read`), ADR-0046 (`sqrt` / `sin` / `cos`).

## Context

1. **One function per shape.** An array's size is part of its type (`[T; n]`, `mapal-as-implemented` §2.1), so a
   projection written once for a `4096 × 4096` weight cannot take the `1024 × 4096` key / value weights or the
   `14336 × 4096` MLP weights. The readable Llama model (the model step's `llama_tiny_ideal.mapal`) spells the same
   matrix-vector product once per shape — `project_64_to_64`, `project_64_to_32`, … — identical bodies with
   different literals (and `rms_norm`, `embed`, the rotations likewise), and every new shape is another copy.
2. **Sizes must stay static.** Every fact the compiler decides from — `tile_plan` geometry, `bounds_proof` ranges,
   `residence_plan` bytes, `unit_plan` rows, the rung rules' wave fill — reads concrete extents. A function over
   unsized arrays would push every one of them to runtime (ADR-0023 territory) or force the conservative choice.
3. **Cross-target.** The same program runs on the interpreter, the native CPU, the GPU and (later) an ASIC; a size
   resolved before any target sees the program keeps every backend unchanged and every target-specific decision
   (tiles, kernels, an ASIC's fixed datapath) keyed on concrete shapes.

## Decision

**D1 — Surface.** Size parameters are declared once, in `<…>` after the function name; array sizes in the signature
are expressions over them. There is no call-site instantiation syntax: the call stays a flow.

```
fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> column_indices;
    rows -> iota -> map { row -> (0.0, column_indices) -> fold { sum, column -> … } } -> ret;
}
(layer.query, normed) -> project -> query;
```

The list holds one or more names (a trailing comma is allowed). A size name is fresh: declared once, not a
parameter's or a function's name (**L1702**), not a reserved name (L1009). Every size parameter must appear in a
parameter or output type (**L1706** at the declaration). `fn main` takes none (L1002). A name in a size position that
is not a declared size parameter is **L1701** — there are no implicit size parameters. The formatter writes the list
tight, `fn project<rows, columns>(`; editors highlight it.

**D2 — Size expressions.** Any array length — in a signature, a binding annotation in the body, a `type` field — is

```
size := term (('+' | '-') term)*      term := atom ('*' atom)*      atom := INT | IDENT | '(' size ')'
```

with the usual precedence, left-associative. `+`, `-` and `*` are the operators whose one unknown operand is
recovered exactly from the result and the other operand; `/` and `%` are not (`n / 2 = 3` holds for `n = 6` and
`n = 7`), so they are a parse error (**P0119**), as is any other form. A dimension evaluates in exact integer
arithmetic and must lie in `1 ..= 2⁶³ − 1` (**L1707**; a literal `0` stays L1208). Literal arithmetic works anywhere
(`[f32; 64 * 64]`), names only inside a size-generic function.

**D3 — Inference, at each call, from the argument's type.** Let the call's argument have IR type `A` (the wire: the
one parameter's value, or the tuple of all of them).

1. *Match.* Walk each parameter type against its component of `A`, outermost dimension first, parameters left to
   right; each array dimension gives an equation `e = N` (`e` the declared size expression, `N` the argument's
   length). A shape that does not match (an array against a scalar, a tuple of another arity) is L1201.
2. *Bind.* An equation `p = N` with `e` a bare name binds `p`; a second, different value is **L1704** (conflict).
3. *Solve,* to a fixpoint: an equation whose expression holds exactly one unbound parameter, occurring once, is
   inverted along the path to it — `a + b = N ⇒ a = N − b`; `a − b = N ⇒ a = N + b`, `b = a − N`; `a · b = N ⇒
   a = N / b`, which must divide exactly (**L1705**, "`rows` = 4096 / 3 is not a whole number"). One unknown
   occurring several times (`[f32; n * n]`, a square matrix) is solved when the expression is non-decreasing in it
   for `p ≥ 1` (no occurrence in a subtrahend, every factor non-negative at `p = 1`): the one whole solution by
   bisection over `1 ..= 2⁶³ − 1`, or L1705 when none exists ("no whole `n` gives 15"). A solved size must lie in
   `1 ..= 2⁶³ − 1` (L1707). An equation with two unknowns waits for another to bind one of them.
4. *Annotation, where needed.* If parameters are still unbound and the call's result is bound with an annotation
   (`x -> zeros -> z: [f32; 8]`) or written to `ret` (the function's declared output), the output type is matched
   against it the same way, and solving resumes.
5. *Check.* A parameter still unbound is **L1706** (it names the parameter and the annotation that would fix it).
   Every equation is evaluated with the binding: a mismatch is L1704 ("`rows * columns` is 64 · 3 = 192, the
   argument has 200"); every dimension of the output type is evaluated too (L1707), so an inferred instance always has
   a well-formed signature.

The whole algorithm is exact integer arithmetic over `i128` with checked products, deterministic, and linear in the
number of dimensions times the number of parameters.

**D4 — A size value in the body is an integer literal.** Inside the body a size parameter is the instance's value as
an integer literal in every respect: usable wherever one is (`n -> iota`, `(x, n) -> fill`, index arithmetic,
comparisons, guards, a `-n` folds to a constant), its width taken from context like any literal (L1203 on a conflict),
and **unpinned it is `Ty::index_ty(n)`** — `i32` up to `2³¹ − 1`, `i64` past it (where a bare literal that large
would need its context to pin it). A size expression in value position — `+`, `-`, `*` over size parameters and
integer literals, at least one parameter: `rows * columns -> iota`, `x[n - 1]`, `(t < n * n)` — is one literal of
its value too, as the hand-specialized program writes `16`; arithmetic over literals alone stays arithmetic. It is never a capture of a map / fold body (a constant there, like a literal) and
never rebound: a parameter, binding, map / fold parameter or loop exit reusing its name is **L1703**.

**D5 — Monomorphization in `mapal-lower`.** Lowering instantiates one fixed-size function per distinct pair
(generic function, size tuple) that the program reaches from `main`: its signature and body are the generic's with
every size replaced by its value — the IR the hand-specialized program lowers to. Tested: the same lowered IR,
names and source spans aside, on every generated program; with one instance per generic function also the same
rewritten IR, host LLVM text, device unit and PTX (five GPU profiles). Two instances of one generic share its
source spans, and the rewriter orders objects tied on source position by id, so against a hand-written program
whose copies sit at different positions the PTX is still identical but the host text may order tied objects
differently (the two-shape `project`: 8 host lines, the same stores reordered) — the same values everywhere.

Calls are resolved where the argument's type is exact, at emission (the builder's object types); the typing walk
(D1 typing, `typing.rs`) infers from its own types where it can, so a map or fold downstream of a generic call sees
its result type, and takes emission's binding of a call site when its own types do not reach. Lowering repeats
until the instance set and the call sites' bindings are stable — one pass for every program whose calls the typing
walk types (every generated program), one more per level where it cannot (an argument out of a `seq` block or a
fanout join, which the walk leaves untyped). Where the walk's errors may be artifacts of such an untyped call (a map
after it typed with the fallback element), emission runs as a probe past unit errors to learn the binding, and
those errors are reported only if it teaches nothing. Bounded at 32 rounds (then the last complete result stands,
or L1901 if none). A program with no generic function takes exactly the original single pass.
Instances no call reaches are dropped, so the IR holds exactly the instances the calls use. A generic
function no call reaches lowers to nothing: its signature is checked (D1, D2), its body only per instance —
an error in it is reported at the generic body's span with "in `project<64, 3>`" appended, once.

**D6 — Naming.** An instance is named `name<v₁, v₂, …>` — the values in the declared order, decimal, `", "`-separated:
`project<4096, 4096>`; its map / fold bodies `project<4096, 4096>::map@0`. The name is not an identifier, so no
source function collides with it. Instances are declared at the generic's source position, ordered by their size
tuple (numerically, lexicographically), so the IR's function order — the backends' `@fn{n}` ordinals — is
deterministic and matches a hand-written program that declares its specializations in that order. Backends name
symbols by ordinal and kernels by id: the name reaches diagnostics, dumps and `mapal-check` (which resolves an
instance's effect by its base name) only.

**D7 — Facts and backends.** After lowering, every fact and backend sees ordinary fixed-size functions: no IR,
fact or backend change. Two calls at the same sizes call one function — `call_keep`, `unchanged_plan` and inlining
treat it as any shared callee; calls at different sizes are different callees, each decided for its own shape (the
reason to specialize: sizes are inputs to every fact). An instance is effectful iff its generic is (effects do not
depend on sizes). The call graph is checked acyclic on surface names (L1008), so the instances are finite: each
instance's sizes are a function of its call site's argument types, and the call graph is a DAG.

**D8 — Out of scope (recorded).** Size parameters on `type` declarations (a name in a field's size is L1701), generic
element types, explicit instantiation (`f<4, 8>` at a call), size-polymorphic recursion, sizes known only at run time.

## Alternatives rejected

| alternative | why not |
| --- | --- |
| Implicit size names (an unknown name in a size position becomes a parameter) | rejected by the user: a typo silently becomes a parameter; the declared list is the function's contract |
| Unsized arrays `[T]` and a `length` builtin | rejected by the user: sizes leave the type, so every fact that reads an extent (tiles, bounds, residency, waves) would need runtime shapes |
| Call-site instantiation `x -> project<4096, 4096>` | redundant: the argument's type determines the sizes (D3), and a stated value could only disagree with it |
| `/` and `%` in sizes | not injective, so not invertible: `n / 2 = 3` has two solutions; a divisibility side condition would be a second language |
| Checking the generic body once, symbolically | needs symbolic extents in every type rule, the builder and every fact; monomorphization reuses the fixed-size checker and keeps the IR concrete |
| Monomorphizing the syntax tree (substituting literals in the source) | spans and names are source slices; the instance's errors must point at the generic body, and inference needs exact argument types, which exist only at emission |
| A separate post-lowering specialization pass over a polymorphic IR | the IR has no symbolic types; a second representation for one feature (CLAUDE.md: no parallel module re-deriving a fact) |

## Consequences

- **Found on the way** (generated programs): the inliner dropped a callee whose Return is written by a call that is
  itself inlined (`fn norm(x) { (x, x) -> dot -> ret; }`, called twice) and replayed that call against the dropped
  function — a panic in `mapal-rewrite` `replay` for any such program, generic or not. Fixed generically
  (`inline_return` substitutes the nested call with the same destination), with a regression test.

- One `project` serves every shape; the model reads as one projection called at several sizes.
- Compile work grows with distinct instances (each is its own function with its own facts), as the hand-written
  program's did; equal sizes share one function.
- An uncalled generic body is not type-checked (D5); `mapal check` reports its signature errors only.
- **Per-target definitions.** Builtins that need per-target overrides (an ASIC's fixed-function `sqrt`, a device
  intrinsic) and library functions written once over arrays meet here: an instance is keyed by (generic, size tuple)
  before any target sees it, so a later per-target override table can key on exactly that pair and a target property
  set (never a target name) without touching inference. Nothing in this ADR selects a body per target.

## What would falsify it

A generic program whose IR, host text or PTX differs from the hand-specialized program's other than in function
names; an inference that accepts a call the hand-specialized signature rejects, or rejects one it accepts (given the
annotation D3.4 needs); a program whose lowering does not settle; lowering time superlinear in the number of
instances.

## The interpreter

It runs the monomorphized IR: nothing changes. `mapal run`, `mapal test` and the differential tests compare a generic
program with the interpreter as any other; the tests compare it with its hand-specialized twin on the interpreter,
the native CPU, the GPU and every emulator mode.

## Amendment (full-model, 2026-11-25): products of unknowns

D3's solving step gains one rule (D3.3′, `sizes.rs` `joint`): when no equation has an unknown alone and sizes are
still unbound — `project<tokens, rows, columns>(weight: [bf16; rows * columns], x: [f32; tokens * columns]) ->
[f32; tokens * rows]`, every dimension a product of two unknowns — one unknown of a product-only equation takes each
divisor of that equation's value in turn, the rest propagate as before, and an assignment is kept when every
equation holds. Exactly one assignment: it is the call's (the result's annotation usually decides it: `columns² =
W·X / Y`); two or more: L1706 names two of them (`… are ambiguous in this call: … and … both fit — bind the result
with its type`); none: the existing errors. Equations with `+` / `−` are not searched. Test: `size_generics`
`product_systems_solve_with_the_results_type`.
