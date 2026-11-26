# ADR-0051: Format generics — `to<target>` and `widen<target>`, format parameters, explicit arguments at a stage

Date: 2026-11-26 · Status: **accepted** (decided by the user). Extends: ADR-0047 (size-generic functions — lifts its
D8 "generic element types" and "explicit instantiation" out-of-scope items), ADR-0048 (size-generic types). Replaces
the conversion names of ADR-0029 (`widen_i64`, `widen_f32`, `widen_f64`) and ADR-0043 D2 (`to_bf16`, `to_f16`,
`to_f8e4m3`, `to_f8e5m2`). Realized through ADR-0046's builtin framework (contract, reference definition, target
realizations).

## Context

Seven builtin names carry a format in their name: `to_bf16`, `to_f16`, `to_f8e4m3`, `to_f8e5m2` (rounding, ADR-0043)
and `widen_i64`, `widen_f32`, `widen_f64` (ADR-0029). Each new format (fp4, a scaled fp8) would add more names, and no
user function can be written once for several formats: the readable Llama fixes its weight and KV-cache formats in every
function that touches them. Size generics (ADR-0047) already instantiate a function per distinct size; formats are the
same kind of compile-time parameter.

The user's decisions (2026-11-26): the format becomes a type argument — `x -> to<bf16>`, `x -> widen<f32>`; a function
may declare format parameters with their kind, `fn attend<positions, cache: float>`; two verbs stay (`widen` is exact,
`to` rounds); the old names are removed. And: "`to` is basically `to<TO, FROM>(a: FROM) -> TO`, but we can specify
`var -> to<TO>` and it will be inferred by var" — explicit arguments are a prefix, the rest is inferred.

## Decision

**D1 — Format parameters.** A declared generic list (ADR-0047 D1) holds size parameters and format parameters. A
format parameter is written with its **kind**: `name: kind`. A bare name stays a size parameter, so every existing list
keeps its meaning.

```
fn attend<positions, cache: float>(keys: [f32; positions * 64], …) -> … {
    keys -> map { key -> key -> to<cache> } -> key_cache;
    …
}
type Matrix<rows, columns, element: float> { _: [element; rows * columns] }
```

The kinds are closed sets of formats: `float` (`f64`, `f32`, `bf16`, `f16`, `f8e4m3`, `f8e5m2`), `narrow` (the floats
narrower than `f32`), `int` (`i32`, `i64`, `u8`), `number` (`float` ∪ `int`). A kind name is meaningful only in a
declared list (it is not a reserved word elsewhere). A format parameter may appear wherever a scalar type may: an array
element, a scalar parameter or output, a binding annotation, a `type` field, a stage's explicit argument. As with sizes,
every declared format parameter must be determined at each use (D3) and a name in a type position that is neither a
format nor a declared format parameter is an error (no implicit parameters — ADR-0047's reason: the declared list is the
contract).

**D2 — Explicit arguments at a stage.** A stage may name its generic arguments: `x -> to<bf16>`,
`(weight, x) -> project<4096, 2048>`. Arguments are positional, a **prefix** of the declared list, sizes as size
expressions (ADR-0047 D2) and formats as format names or the caller's own format parameters. Calls remain flows
(ADR-0047 D1): arguments are written only in stage position, where a name followed by `<` opens the list (a comparison
cannot stand as a stage). The formatter writes it tight: `to<bf16>`, `project<4096, 2048>`.

**D3 — Inference completes the rest.** The parameters not given explicitly are inferred, in this order: (1) from the
argument's type, ADR-0047 D3's matching extended so a format parameter in a type position binds to the argument's
scalar format there; (2) from the expected type of the result where the context fixes it — a binding's annotation
(`x -> to -> y: bf16`, as `read` is typed today, ADR-0045), a parameter or field the value flows into. A parameter still
undetermined is an error naming it; two different determinations conflict (ADR-0047 L1704's rule). A format outside the
parameter's kind is an error naming the kind.

**D4 — The conversion builtins as generic functions.** Their signatures are ordinary generic signatures (a prelude
declaration); their realizations are builtins (ADR-0046: contract, reference definition, per-target realizations chosen
by the facts):

```
fn to<target: number, source: number>(value: source) -> target       -- rounds
fn widen<target: number, source: number>(value: source) -> target    -- exact
```

- **`to`** rounds to the nearest value of `target` with ADR-0043 D3's contract (exact on every face as an operation;
  fp8 saturating to ±max; `NarrowForm`). Admitted pairs: `f32` → each narrow format (ADR-0043 D2); an `int` → a
  `float` (round to nearest; exact while the value fits the significand). `f64` → `f32` and `float` → `int` stay open
  (ADR-0043 D2's open item; no rounding-to-integer rule is decided).
- **`widen`** is the exact embedding: admitted where every value of `source` is a value of `target` — narrow → `f32` /
  `f64`, `f32` → `f64`, `u8` / `i32` → `i64`, `u8` / `i32` → `f64`, `u8` → `i32` — or where the facts prove the value
  is one (a compile-time value, ADR-0048 `static_plan`; a value range from the bounds facts within `target`'s exact
  range). Anything else is an error that suggests `to<target>` — the conversion that may round says so in the text.
- An identity conversion (`source = target`) is admitted and erased.

**D5 — Monomorphization and naming.** Lowering instantiates one function per distinct argument list, sizes and formats
together (ADR-0047 D5); an instance is named `name<a₁, a₂, …>` in the declared order (ADR-0047 D6), formats by their
names (`to<bf16, f32>`). Every fact and backend after lowering sees fixed-format functions, as today. The precision plan
and the a-priori bound are computed per instance (ADR-0041).

**D6 — The old names are removed.** `to_bf16`, `to_f16`, `to_f8e4m3`, `to_f8e5m2`, `widen_i64`, `widen_f32`,
`widen_f64` become errors that name the replacement (`to_bf16` → `to<bf16>`; `widen_f32` → `widen<f32>`, or `to<f32>`
where D4 does not admit `widen`). Every Mapal source in the repository (examples, benches, tests, generated-program
generators, docs) is rewritten; the rewrite of an existing use must lower to the same IR (a `widen_f32` of an integer the
facts cannot prove exact becomes `to<f32>`, which is the conversion it always was).

## The interpreter

Unchanged in meaning: it runs the lowered instances. `to` and `widen` keep the oracle's definitions (ADR-0043's
rounding, the exact embeddings); instances are named per D5.

## Alternatives rejected

- **One verb, `to<T>`, exact or rounding by the pair**: a lossy conversion and an exact one read the same; the program
  stops showing where it loses precision (the user kept two verbs).
- **Format parameters by use** (a bare `<cache>` whose kind follows from how it is used): the declared list stops being
  the contract (ADR-0047 rejected implicit size names for the same reason).
- **Keep the old names as aliases**: two spellings of one operation (the user chose one).
- **Call-site syntax `f<…>(x)`**: Mapal calls are flows; the stage form `x -> f<…>` needs no new call form.

## What would falsify it

- A use whose format cannot be named at the stage nor inferred from the argument or the context (then the inference of
  D3 is incomplete).
- A rewrite in D6 that changes the lowered IR of an existing program (then the new builtins do not express the old ones).

## Implementation (2026-11-26)

- **Syntax** (`mapal-syntax`): `FnDecl` / `TypeDecl` hold `generics: Vec<GenericParam>` (`name`, `kind: Option<Name>`);
  a stage `IDENT '<' size (',' size)* ','? '>'` that ends where the stage ends is `ExprKind::Generic { name, args }`
  (otherwise the expression stage it always was); a number format is a generic type's argument (`Matrix<64, 64,
  bf16>`; a capitalized name or `bool` stays ⟦P0103⟧). The formatter writes `to<bf16>`, `fn f<n, c: float>(`.
- **Lowering** (`mapal-lower`): `formats.rs` (formats, kinds, `Arg = Size | Format`, the prelude, D4's admission),
  `sizes.rs` (the explicit prefix, format equations beside size equations, kinds, instance names), `tys.rs` (a
  format parameter in any type position, a generic type's format arguments), `convert.rs` (the post-seal proof of
  `widen<f32>` of an `i32`). Codes: L1708 undetermined, L1709 conflict, L1710 out of kind (or a size where a format
  is declared, and back), L1711 unknown kind, L1712 explicit arguments on a stage that takes none, L1713 a removed
  name; L1614 / L1616 keep `widen` / `to`.
- **Facts** (`mapal-ir`): `BoundsProof::widen_range(m)` — the range of the integer a `Widen` converts where it runs;
  `%` by a constant is ranged without its dividend's range (any `i32`, a wrapped product included, leaves `x % k`
  inside `(−|k|, |k|)`). The suite's host and device text is unchanged on the RTX 5080 and x86-64-v3 profiles.
- **Readings of D4.** `to` admits the exact pairs too (rounding to nearest of a representable value is the value),
  so a generic `x -> to<cache>` stays valid at `cache = f32` (the identity) and `f64`. A float literal flowing into
  a conversion takes the target's width (an `f32` into a narrow target); an integer literal keeps `i32`.
- **Open (reported, not realized):** `u8` and `i64` sources (no IR conversion edge yet), an integer to a narrow
  format (one rounding; through `f32` two), `f64 → f32` / narrow and float → int (ADR-0043 D2's open item);
  inference of a conversion's target from a map body's expected element type (the body's tail has no expected
  type yet: write the argument).
