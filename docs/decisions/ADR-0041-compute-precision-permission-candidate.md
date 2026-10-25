# ADR-0041: Compute precision as a permission on compute nodes — `tf32 | f32`

Date: 2026-10-25 · Status: candidate — NOT decided · number provisional · changes nothing until accepted.
Direction agreed (Sapir, 2026-10-25): precision is attached to the compute, not to the stored type — "attaching to
compute type seems more relevant; this way on the graph path we can group by compute type on the dataflow graph".
Implements: ADR-0032 D1 (the `tf32-class` region contract) and D5 (no precision via ambient config). Amends: ADR-0039
(candidate) — replaces its `to_tf32` morphism and `block` fold class; keeps its `bf16`/`f16` storage types. Builds on:
ADR-0040 (default face: full precision, a-priori bound, order free, fastest per environment).

## Context

1. **The tensor-core kernels exist; the program cannot ask for them.** `Reduced::Tf32` (`crates/backends/llvm/src/lib.rs`
   `Reduced::Tf32`) emits `mma.sync.m16n8k8.tf32` on sm_80+ and Hopper `wgmma` on sm_90a, behind the emitter flag
   `--precision=tf32` (`flags.rs`), with its own `Face::Tf32` rung rows (`profile.rs`). On an RTX 4090 that kernel ran
   77.9 TF/s against the FP32 FMA kernel's 52.8 (ADR-0039 Context). A flag that changes rounding is what ADR-0032 D5
   forbids except as an interim form.
2. **ADR-0039 put TF32 on the data** (`to_tf32 : f32 → f32`, an explicit rounding morphism). That fixes one realization
   in the program text: the CPU must round although it has no TF32 unit, and the program cannot say "TF32 where it is
   faster, f32 elsewhere". ADR-0040 settled that the default face is a bound plus the fastest realization per
   environment; a precision choice should follow the same rule.
3. **TF32 is a property of a product, not of a value.** Storage stays 32-bit; only the multiplicands of a product are
   rounded to 10 explicit significand bits, and the accumulation stays f32. An `mma` rounds *both* operands, so a
   per-value type (`w : tf32`) cannot by itself license `x·w` with `x` in full f32.
4. **The comparison legs already use it.** PyTorch's convolutions run TF32 by default
   (`torch.backends.cudnn.allow_tf32 = True`); the convmc torch leg is TF32 while Mapal is full FP32 (why-slow.md O6).
   A same-face comparison needs either torch with TF32 off or Mapal with a TF32 permission.

## Decision

**D1 — A precision permission set on compute nodes.** Every float product node (`a * b`, including the products inside a
fold body) carries a set `P` of admitted product precisions. Members: `f32` (ADR-0040's default face), `tf32`
(multiplicands may be rounded to binary32 with ≤ 10 explicit significand bits, round to nearest), later `bf16` and
`f16` (multiplicands rounded to those significands; `f16` also needs a value-range proof, below). `P` says what is
*acceptable*; it is not an ordered preference and not "use when available". The default is `P = {f32}`.

**D2 — Surface: a `precision(...)` clause on a scope; innermost wins.** Provisional syntax:

```
fn main() precision(tf32 | f32) { ... }                       // every product in main
(0.0, k) -> fold precision(tf32 | f32) { acc, k -> acc + x[k] * w[k] } -> y   // this fold only
```

A scope's clause applies to every product node lowered from it, nested `map`/`fold` bodies included, unless a nested
clause overrides it. `precision(f32)` restores the default inside a relaxed scope. The clause is part of the function
(ADR-0032 semantics note), so the oracle and gates are per scope.

**D3 — Semantics = the weakest member's bound; the realization is free.** A product node with set `P` admits every
realization whose error stays within the bound of the weakest member of `P`. For a tile site of length `K` whose
products admit `tf32`, with `u_t = 2⁻¹¹`, accumulation roundoff `u_a` (`2⁻²⁴`; `2⁻²³` for a tensor-core accumulator,
whose rounding mode is not guaranteed round-to-nearest) and ADR-0040's `M = |seed| + Σ_k |a_k·b_k|`:

`|out − exact| ≤ ((1 + u_t)² · (1 + γ_{K+1}(u_a)) − 1) · M  ≈  (2·u_t + γ_{K+1}) · M`

(a TF32 × TF32 product has ≤ 22 significand bits and is exact in f32, so the only product error is the two operand
roundings). Members stronger than the weakest are always admitted: `{tf32, f32}` admits f32, f64, and any split
realization that meets the bound. Writing `| f32` is therefore implied; the syntax keeps it because it reads as intent
and extends to `bf16 | tf32 | f32`.

**D4 — Precision is one more axis of the face, orthogonal to steps and order.** ADR-0040 D1 has *steps*
(`separate`/`fma`) and *order* (`sequential`/`fixed`/`free`). `P` is the third axis. `exact` (separate, sequential)
with `tf32 ∈ P` is legal: operands rounded, the fold in order, bit-checkable against an interpreter that rounds the same
way. ADR-0039 D3's `block` class is not needed: on the default face the order inside a hardware k-step is already free
(ADR-0040).

**D5 — `precision_plan`: one IR fact, stored on the graph.** In `mapal-ir/src/algo.rs`, computed once on the sealed
graph:

| field | what it proves |
| --- | --- |
| `set(m)` | `P` of product node `m` (from the innermost clause) |
| `round_at(v)` | value `v` whose every product reader admits `tf32`: round once at the producer, not per use (the compute-once rule) |
| `cross(e)` | edges where a value flows from a region admitting `tf32` into one that does not (no rounded copy may cross) |
| `bound(o)` | each tile site's / output's a-priori bound from D3, for the gates |

Kernel boundaries stay a result (CLAUDE.md, "No fusion concept"): `cell_plan`, `footprint_plan` and `flow` read
`set(m)` as one more input. A work unit whose products all admit `tf32` may take the tensor-core realization; a unit
mixing members keeps FFMA for the f32-only products. No grouping pass is added.

**D6 — Realization is backend config, chosen by measurement.** The existing `Reduced::Tf32` kernels (`mma.sync` sm_80+,
`wgmma` sm_90a) are re-keyed from the flag to `set(m)`. Per `(part, face)` the choice between FFMA tile and tensor-core
tile follows `TargetProfile` and a cited `GPU_RUNGS_OFF` row (ADR-0040 D5); every realization has an OFF arm. On the
CPU, `P ∋ tf32` realizes as f32 (no TF32 unit; f32 meets the bound). The chosen member per site and part is
recorded in `facts.json` and the evidence intents.

**D7 — `--precision=tf32` is deleted when D1–D6 land** (ADR-0032 D5, ADR-0039 D5).

## Interpreter

The parser accepts the clause; the IR stores `P` per product node. The interpreter evaluates in f32 (the strongest
member), so `.expect` files do not change. `mapal test --weakest` evaluates each product with its weakest member
(operands rounded with the IEEE rule) to exercise D3 from the other side. Value gates on scopes admitting `tf32` accept
iff `|got − ref| ≤ D3 bound`, `ref` and `M` in f64 (ADR-0040 D2 pattern). `print` is unchanged.

## Later (not decided here)

- **Deduced narrow storage.** If every reader of a value admits `bf16` (or `f16`), the producer may store it in 16 bits:
  half the bytes moved, deduced from consumers like residency and caching. Needs ADR-0039's storage types and an
  amendment of ADR-0032 D2 (conversions explicit): the permission is the explicit statement, the conversion its
  realization. `f16` additionally needs a value-range fact (65504 overflow).
- **3xTF32 under `{f32}`.** Splitting each operand into TF32 hi + lo and summing three products reaches near-f32
  accuracy on tensor cores (H100: ≈ 378 / 3 TF/s against 51 FP32). Its per-product error (≈ 3·2⁻²²) and its 3K
  accumulations exceed ADR-0040 D2's `γ_{K+1}·M` by a constant, so it is **not** admitted by `{f32}` as written. It
  becomes admissible only if D2 gains a stated slack term — a separate decision, to be argued from an H100 measurement.
- **Path composition of bounds.** D5 `bound` is per site and per probe, as ADR-0040 D2. Composing bounds along a
  path through non-linear ops (`exp`, division) needs condition numbers; open.

## Alternatives rejected

| alternative | why not |
| --- | --- |
| `tf32` as a datatype on values | Not a storage format (ADR-0039 already rejected it as one); an `mma` rounds both operands, so a per-value type does not license the product; meaningless on the CPU |
| `to_tf32` explicit rounding morphism (ADR-0039 D2) | Fixes one realization in the source: the CPU must round with no gain, and the program cannot say "fastest of these" (ADR-0040) |
| Ordered preference "tf32 if available, else f32" | Availability is the wrong criterion (TF32 vs FFMA is a measured, per-part choice); the semantics must be the bound of the set, whatever is chosen |
| Keep `--precision=tf32` | ADR-0032 D5: a flag that changes bits is a second, invisible source program |
| Per-operand permission | Needs a split realization (2xTF32) to use it; no program needs it yet |

## What would falsify it

- A tensor-core realization that violates D3's bound on a program the testgen generator produces (e.g. accumulation
  worse than `u_a = 2⁻²³`).
- A program that needs different precisions for the two operands of one product, which a scope set cannot express.
- Measured: on every part, the tensor-core realization of a `tf32` site is not faster than the FFMA tile (the 4090
  measurement says it is: 77.9 vs 52.8 TF/s).
- A chain where every per-site gate passes but the program output exceeds the composed bound.
