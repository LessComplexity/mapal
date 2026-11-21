# ADR-0042: `exp` — a pipeline builtin, the oracle's algorithm, a bounded default face, the rescale law

Date: 2026-11-21 · Status: **accepted-for-implementation** (sprint step "Exp": "Add exp to the language as a STANDARD
PIPELINE BUILTIN, exactly like widen_f32 … written `x -> exp` … NEVER call syntax `exp(x)`").
Builds on: ADR-0029 (bare-stage builtins resolved by name), ADR-0040 (the default float face: full precision, an
a-priori bound, order free), ADR-0041 (precision as a property of compute nodes, `precision_plan`).

## Context

1. **Softmax, attention and SiLU had no exponential.** The benchmark programs spelled `eˣ` as a polynomial:
   `(1 + x/256)^256` by eight squarings (`gen_tiles.py` `expo`, attention, softmax, decode) and, at model scale,
   `e^(z/256)`'s degree-5 Taylor polynomial raised to the 256th power with a clamp at −80 (`llm_exp`). Relative error
   ~5·10⁻⁵, a domain cut, ten multiplies and a compare per element, and an operation torch, cuDNN and every inference
   engine run as one `expf` (MUFU.EX2 on the device).
2. **No combine law for an exponential.** `fold_combine` (`mapal-ir/src/algo/algebra.rs`, long-rows) gives a value
   constant along a fold's axis a *factor*, a *shift* or a *phase*. A softmax's denominator reads `exp(s − m)`; the
   polynomial is no homomorphism `(+) → (×)`, so the denominator was a phase after the max (`long-rows.md`):
   online softmax and flash-style attention could not fall out of row slices or row-block units.
3. **Determinism where the faces demand it.** The exact, contract and fixed faces compare every backend with the
   interpreter byte for byte (ADR-0040 D1). A libm `expf` differs between glibc, CUDA and LLVM's constant folder, so
   the language must define the bits on those faces itself.

## Decision

**D1 — Surface: a pure pipeline builtin.** `exp` is resolved by name in pipeline position like `widen_f32`
(`mapal-lower` `is_pure_builtin`): `x -> exp`, `x -> exp -> y`, in any chain — a function body, a map or fold body,
a fanout branch. There is no call syntax (`exp(x)` is not Mapal). The operand is an `f32` or `f64`; the result has
its width (a float literal flows through and takes the width it unifies with, `f64` unpinned). Any other operand is
**L1615**; `exp` is a reserved name (L1009). The formatter keeps the stage as written.

**D2 — IR: one operation.** `Operation::Exp : Float{b} → Float{b}`, pure (no token), total and trap-free, built by
`FnBuilder::exp` (`IrError::InvalidExp`), validated as a width-keeping float edge. It is an ordinary unary operation
for every fact: a body with `exp` is classifiable (`elem_plan` / `map_law`'s `Apply`), device-capable and
recomputable; `bounds_proof` ranges no float.

**D3 — The oracle's semantics: the algorithm `E_b`** (`mapal-ir/src/exp.rs`, constants as bit patterns in
`ExpForm`):

```
xc = x > hi ? hi : x;  xc = xc < lo ? lo : xc;  xc = xc == xc ? xc : 0
kf = (xc · log2e + shifter) − shifter            round to the nearest integer (shifter = 1.5·2^(p−1))
k  = (int) kf
r  = (xc − kf · ln2_hi) − kf · ln2_lo             Cody–Waite; kf·ln2_hi exact for every reachable k
p  = Horner of Σ_{i≤n} rⁱ/i!                      n = 7 (f32), 13 (f64), coefficients rounded to the width
y  = (p · 2^(k>>1)) · 2^(k − (k>>1))              both scales normal; the second product rounds once if subnormal
y  = x > hi ? +inf : y;  y = x < lo ? +0 : y;  y = x ≠ x ? x : y
```

`hi`/`lo`: f32 `0x42b17217` (88.7228317) / `0xc2cff1b4` (−103.9720764); f64 `0x40862e42fefa39ef` /
`0xc0874910d52d3051`. Every step is one IEEE operation rounded to nearest even. Measured: `E_32` exhaustively over
every f32 in `[lo, hi]` (2 239 849 421 arguments) ≤ **1.221 ulp** (relative 2^−23.21 on normal results); `E_64` ≤ 1.0
ulp of glibc's `exp` over 2·10⁸ samples. The interpreter computes `E_b`; the rewriter constant-folds `exp` with it.

**D4 — Faces.** On the exact, contract and fixed faces every backend computes `E_b` operation for operation (the
LLVM host and device modules define `@mapal.exp.f32/f64` with no fast-math flag, llc runs `-nvptx-fma-level=0`; the
CUDA backend's `mapal_exp_f32/f64` uses `__fmul_rn`/`__fadd_rn` on the device, so `-fmad=true` cannot contract it):
the interpreter's bytes. `exp` is never contracted outside the default face. On the **default face** any realization
is admitted that keeps the oracle's special values (`x > hi` → `+inf`, `x < lo` → `+0`, `NaN` → `NaN`) and meets,
for `x ∈ [lo, hi]`,

`|got − eˣ| ≤ 2^−21 · eˣ + 2^−149` (f32)  ·  `|got − eˣ| ≤ 2^−50 · eˣ + 2^−1074` (f64)

(the absolute term is the smallest subnormal: a subnormal result rounds once). `precision_plan.exp(m)` records each
`Exp`'s form and bound; a value gate on a path through `exp` adds the input's error times the condition number
`|x|` (the composition rule ADR-0041 left open, here for one operation).

**D5 — Realizations on the default face, each with its OFF arm** (`--gpu-off=exp-hw`, `--no-exp-hw`: `E_b` on every
face):

| side | realization | admitted because |
| --- | --- | --- |
| host | `E_b` with `contract` on the reduction and the polynomial (FMA where the target has it); straight-line selects, no call — a loop of it vectorizes | exhaustive f32 ≤ 0.939 ulp |
| device f32 | the reduction as two FMAs, `ex2.approx.ftz.f32` (MUFU.EX2) of `r·log2e ∈ [−½, ½]`, the oracle's scaling and specials | `(u_ex2 + 1)·2^−24 ≤ 2^−21`, `u_ex2` the part's measured `ex2` error in units of `2^−24` |
| device f64 | `E_64` contracted (no f64 hardware exponential) | as the host |

`u_ex2` is a `TargetProfile` capability property (`Caps::ex2`): measured exhaustively over `[−½, ½]` on the RTX 5080
(`benches/local/exp_calib.cu`): 2^−22.709 → 3 units; the composed form measured exhaustively over `[lo, hi]` ≤
**2.068 ulp** (CUDA's `expf` 1.931, `__expf` 63.8: its error grows with `|x|`, outside the bound). A part whose
measured `u_ex2` exceeds 7 takes the contracted polynomial — the rule reads the property, never a name or version.

**D6 — `precision_plan`.** Beside `set`, `round_at`, `cross` and the tile sites' `bound`, the fact carries `exp(m)`
for every `Exp`: its width's `ExpForm` (the oracle's constants and the default face's `rel`, `abs`).

**D7 — The rescale law (`fold_combine`).** A value `v` constant along a fold's axis enters the term by
`DepLaw::Rescale` when `⊕ = +` and the term is `exp(a(v)) · y` through products and quotients whose other operand
never reads `v`, with `a(v) = κ·(x − v)` (or `κ·(v − x)`): `x` free of `v`, `κ` a product of positive constants
(`RescaleArg`: the `Exp`, the side of `x`, the steps, `up = sign(κ·x)`, the array `x` is read from). Then for any
shift `s`

`exp(a(v)) = exp(a(s)) · exp(a(x := s, v))`,

the second factor the same for every term — `exp(a − m₁) = exp(a − m) · exp(m − m₁)`. Derived from the dataflow
(`rescale_law`, `exp_factor_path`), never a name. It composes through a cell whose output is a product path in
the inner cell's read (attention's `p = ex ÷ den` keeps `mx` a rescale for `P·V`). Real arithmetic: exact. Floats:
each rescale is one more `exp` and one product, `(2^−21 + 2^−24)` relative per term, inside D4 composed with ADR-0040
D2 — so only the default face takes it; the exact and fixed faces keep the phase (the multi-pass order). Off:
`--no-exp-rescale` (`FactArms::exp_rescale`, a fact arm: such a value is a phase).

**Consumers.** `row-slices` (`func/flow.rs`) takes a rescale where the value is an earlier fold of the unit over
`x` — the same array, row and slices — whose step keeps the extreme matching `up` (a max for `x − v`): the
denominator's partial over a slice takes that slice's max as its shift (every term ≤ 1, no overflow) in the same
phase as the max; the combine multiplies partial `r` by `exp(a(x := m_r, v := m))`. Softmax: three phases → two
(online softmax). Flash-style attention (P·V's rescale over key blocks of a row-block unit) is the same fact; its
streamed unit is not built (`exp.md`, open).

## Alternatives rejected

| alternative | why not |
| --- | --- |
| Call syntax `exp(x)` | Mapal has no call syntax for builtins; the pipeline form is the language's (`x -> widen_f32`) |
| libm / CUDA `expf` as the semantics | not the same bits across glibc, CUDA and LLVM's constant folder: no exact face |
| `__expf` (`ex2.approx(x·log2e)`) on the default face | 63.8 ulp measured: the rounded product's error grows with `|x|` — outside any fixed bound |
| A correctly rounded `exp` | costs a multi-word reduction for a bound no consumer needs; no hardware form would meet it |
| `exp2` instead of `exp` | the programs (softmax, SiLU, attention) are written in `e`; `exp2` can be added the same way |
| The program's polynomial | 5·10⁻⁵ relative, a clamped domain, ten multiplies; no homomorphism, so no rescale law |
| A rescale law by pattern ("softmax") | CLAUDE.md: laws come from the dataflow; the test set includes the forms that must stay phases |

## What would falsify it

A default-face output outside D4's bound (a realization's or the rescale's error); an exact-face output differing
from the interpreter on any backend; a part whose measured `ex2` error exceeds what `Caps::ex2` states; a program
where the rescale changes a value beyond the composed bound (overflow of a partial: the consumer's shift must keep
every term ≤ 1).

## The interpreter

It computes `E_b` (`mapal_ir::exp::exp_f32` / `exp_f64`) — the oracle on every face; `mapal test` and the
differential tests compare the exact face with it byte for byte, the default face within D4 (with ADR-0040 D2 for the
folds around it).
