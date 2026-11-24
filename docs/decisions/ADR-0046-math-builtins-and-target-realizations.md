# ADR-0046: `sqrt`, `sin`, `cos` — one builtin-realization framework: contracts, portable reference definitions, target realizations as data

Date: 2026-11-25 · Status: **accepted-for-implementation** (user request 2026-11-25: "add sqrt, sin, cos builtins … the
thing is bcs we are cross architecture/hardware/and even asic, some builtin function per target might require
definition overrides, so keep this in mind").
Generalizes: ADR-0042 (`exp`: the oracle's algorithm, a bounded default face, the `ex2` hardware form) — `exp` moves onto
this framework with no behaviour change. Builds on: ADR-0029 (bare-stage builtins resolved by name), ADR-0040 (the
default float face), ADR-0041 (`precision_plan`), CLAUDE.md "hardware is physical properties, never a name or version".

## Context

1. **The programs need more than `exp`.** Norms divide by a square root (RMSNorm, LayerNorm, the L2 of a row),
   rotary position embeddings take `sin` / `cos` of position × frequency, signal programs (FIR design, FFT twiddles)
   take both. None of them can be spelled today except as a hand-written polynomial with a clamped domain — the
   situation `exp` was in before ADR-0042.
2. **ADR-0042 built one builtin's machinery by hand.** `mapal-ir/src/exp.rs` holds the algorithm and its bound;
   `backends/llvm/src/exp.rs` re-spells the algorithm as LLVM text three times (oracle, contracted, `ex2` form);
   `backends/cuda/src/module.rs` spells it a fourth time as C; `Caps::ex2` is a field for one instruction. A second,
   third and fourth builtin copied this way is four more hand-written copies per backend, and a new target (an ASIC
   with its own function units, or none) would need code in every backend.
3. **Targets differ in what they can compute, not only in how fast.** A CPU has a correctly rounded `sqrt`
   instruction and no transcendental unit; an NVIDIA SM has `sqrt.rn` (correctly rounded, several instructions),
   `sqrt.approx` / `rsqrt.approx` / `sin.approx` / `cos.approx` / `ex2.approx` (MUFU, one instruction each, an error
   only measurement states); an accelerator ASIC may have no float function unit at all, a CORDIC unit, or a vendor
   routine in ROM. libm results differ between glibc, CUDA and LLVM's constant folder in the last bit, so the exact
   faces (ADR-0040 D1) cannot be defined by "the platform's `sinf`".

## Decision

**D1 — Surface.** `sqrt`, `sin` and `cos` are pure pipeline builtins exactly like `exp` (ADR-0042 D1): `x -> sqrt`,
`angle -> sin -> s`, legal in any chain (function bodies, map and fold bodies, fanout branches), never call syntax.
`Float{b} → Float{b}` for `b ∈ {32, 64}` (a literal takes the width it unifies with); any other operand is **L1615**
(the one code for every math builtin); the names are reserved (L1009). Effect-free, trap-free, fanout-legal.

**D2 — IR: one operation family.** `Operation::Math(MathFn)` with `MathFn ∈ {Exp, Sqrt, Sin, Cos}`
(`mapal-ir/src/math.rs`), built by `FnBuilder::math` (`IrError::InvalidExp` for a non-float operand), validated as a
width-keeping float edge. `Operation::Exp` is kept as an associated constant equal to `Math(Exp)`, so every reader of
ADR-0042's operation keeps its meaning. Every fact treats a math op as an ordinary pure unary op (classifiable,
device-capable, recomputable); `fold_combine`'s rescale law (ADR-0042 D7) stays `exp`'s alone. `range` gives `sin` /
`cos` `[−1, 1]` and `sqrt` `[√lo, √hi]` on a non-negative range.

**D3 — Each builtin has a contract** (the language's semantics; `math::contract`, `math::special`):

| builtin | exact / contract / fixed faces (ADR-0040 D1) | default face, finite `x` in the domain | special values (every face) |
| --- | --- | --- | --- |
| `exp` | `E_b` (ADR-0042 D3) bit for bit | `\|got − eˣ\| ≤ 2⁻²¹·eˣ + 2⁻¹⁴⁹` (f32) · `2⁻⁵⁰·eˣ + 2⁻¹⁰⁷⁴` (f64), `x ∈ [lo, hi]` | `x > hi → +∞`, `x < lo → +0`, NaN → NaN |
| `sqrt` | the correctly rounded `√x` (IEEE 754 `squareRoot`, round to nearest even) bit for bit | `\|got − √x\| ≤ 2⁻²²·√x` (f32) · `2⁻⁵¹·√x` (f64), `x > 0` incl. subnormal | `±0 → ±0`, `+∞ → +∞`, `x < 0 → NaN`, NaN → NaN |
| `sin` | `S_b` (D4) bit for bit | `\|got − sin x\| ≤ 2⁻²¹·\|sin x\| + 2⁻¹⁴⁹` (f32) · `2⁻⁵⁰·\|sin x\| + 2⁻¹⁰⁷⁴` (f64), **every finite `x`** | `±0 → ±0`, `±∞ → NaN`, NaN → NaN |
| `cos` | `C_b` (D4) bit for bit | `\|got − cos x\| ≤ 2⁻²¹·\|cos x\| + 2⁻¹⁴⁹` (f32) · `2⁻⁵⁰·\|cos x\| + 2⁻¹⁰⁷⁴` (f64), every finite `x` | `±0 → 1`, `±∞ → NaN`, NaN → NaN |

A NaN result is any NaN (no Mapal program observes a payload). The default face's bound is relative with only the
smallest subnormal as absolute term, also at the zeros of `sin` and `cos` far from the origin: the reduction is exact
enough there (D4), so the default face keeps ADR-0040's "full precision" — a form whose error is absolute (a MUFU
`sin.approx`: `2⁻²¹·⁴¹` absolute on `[−π, π]`) is refused, not the contract loosened.

**D4 — Each builtin has a reference definition, written once from plain operations** (`math::{exp_def, sqrt_def,
sin_def, cos_def}`): IEEE `+ − ×` rounded to nearest, comparisons, selects, integer `+ − × mulhi & | ^ << >> clz`,
bit casts and int ↔ float conversions — straight-line, no table, no loop, no call, no FMA, no division. It is what the
interpreter runs, what the rewriter folds constants with, and what any backend emits where the target declares nothing
better: a target with no float function unit at all (an ASIC) still has every builtin, with the exact faces' bits.

- `exp`: ADR-0042's `E_b`, unchanged (the same operations in the same order — the host and device modules' text is
  byte-identical to ADR-0042's).
- `sqrt`: the significand's integer square root by the restoring digit recurrence (two radicand bits a step, 25 steps
  for f32 / 54 for f64, a 64-bit remainder), one round-to-nearest from its last bit (a tie is impossible for a square
  root), subnormal operands normalized by `clz`: the correctly rounded `√x` — equal to every IEEE `sqrt` (exhaustively
  checked over every f32, sampled f64).
- `sin` / `cos`: **Payne–Hanek reduction for every `|x| ≥ π/4`, in integer arithmetic**: the significand times a
  96-bit (f32) / 192-bit (f64) window of `2/π`'s bits chosen by the exponent (words picked by selects from constant
  words, no memory table), the quadrant from the product's top bits, the remainder's fraction normalized by `clz` and
  multiplied by `π/2` as a fixed-point integer before one rounding to the width; `|x| < π/4` is its own reduced
  argument. Then a polynomial on `[−π/4, π/4]` (f32: Cephes' degree-7 `sin` / degree-8 `cos`; f64: fdlibm's
  `__kernel_sin` / `__kernel_cos`), the quadrant's sign and swap, the special values. The reduced argument carries
  one rounding (relative `2⁻²⁴` / `2⁻⁵³`) even at the worst-case cancellation, so the result's error is relative over
  the whole range.

**D5 — Realizations are capability data the target declares** (`math::Target`, on `TargetProfile::math` for the host
and `Caps::math` for a device; never code keyed on a target's name):

```text
Unit      { op: Ex2 | Sqrt | Sin | Cos, bits, sym, err: Exact | Bound{rel, abs}, dom: [lo, hi], cost, specials, evidence }
TargetDef { f, bits, body: Steps(fn(&mut Trace, Val) -> Val) | Symbol(sym), err, cost, evidence }
```

A **unit** is one hardware instruction or function unit the target computes `op` with: `sym` is what the backend
emits (an LLVM intrinsic), `err` its error over `dom` against the real function — `Exact` only for a correctly rounded
operation (IEEE `sqrt`: `llvm.sqrt`, `sqrt.rn`), else the measured maximum (`Bound`) — `cost` its issue slots,
`specials` whether it returns the contract's special values itself (else the form wraps it in selects), `evidence`
where `err` comes from (a proof, or the calibration line it was read from). A **target definition** is a routine the
target supplies: built from the same plain operations (`Steps`: it is traced and emitted like a reference) or an
external symbol the target links (`Symbol`), with its declared error and the evidence for it. A builtin composes
units into **forms** whose error follows a-priori from the unit's (`math::forms`):

| form | needs | error of the form |
| --- | --- | --- |
| reference | — | `Exact` |
| contracted reference | the default face contracts (`fma`); every form's marked steps fuse there too | within the contract (measured: `exp` f32 0.939 ulp exhaustive; the device probe for `sin` / `cos`) |
| `exp` ← `ex2` | `Ex2` f32 on `[−½, ½]` | `rel(ex2) + 2⁻²⁴` (ADR-0042 D5) |
| `sqrt` ← `sqrt` | `Sqrt` of the width on `(0, ∞)` | the unit's |
| `sin` / `cos` ← `sin`, `cos` | a unit for `sin r`, for `cos r`, or both (the other part the polynomial), on the remainder's range `[−π/4, π/4]` | `max rel + 2⁻²³` (the remainder's rounding; the polynomial's part `2⁻²²`), `max abs` |

**D6 — Admission and selection.** A candidate (the reference, the contracted reference on a contracting default
face, each form whose units the target declares, each target definition) is **admitted** on the exact, contract and
fixed faces only if its error is `Exact`; on the default face if it is `Exact` or its `Bound` is inside the contract
(`rel ≤ rel_c`, `abs ≤ abs_c`) and a measured bound cites its evidence. A `Steps` definition is also re-measured over
the contract's test domain (`math::test_args`: ±0, subnormals, the domain's edges, huge `sin` / `cos` arguments at
the f32 / f64 worst cancellations, ±∞, NaN, seeded samples) before it is admitted — a declaration its own trace
contradicts is refused (`Refusal::Measured`). Among the admitted, the **cheapest** is chosen (`cost`: the traced
definition's operations, a unit's declared slots), ties to the reference. Nothing reads a target's name; the
selection is `math::choose(f, bits, face, contract, hw, &Target)`.

**D7 — How a new target declares its realizations.** A CPU declares its correctly rounded `sqrt` (`Exact`,
`llvm.sqrt.f32` / `.f64`). A GPU declares each function unit with the error its calibration measured
(`benches/gpu/calib.cu math`: exhaustive over the unit's domain against the f64 function; the RTX 5080's values in its
profile, `profile::MATH_SM_RTX_5080`; every NVIDIA SM's `ex2.approx` and `sqrt.rn`, `profile::MATH_SM`) — a part never calibrated takes the reference (no `Bound` without evidence). An
ASIC with no float unit declares nothing and runs every reference from plain operations; one with a unit declares it
as a `Unit` (its error measured on the part); one whose vendor ships a routine declares a `TargetDef` (`Steps` when
the routine is plain operations — then it is checked in-process — or `Symbol` with a measured bound). On the exact
faces only `Exact` realizations are admitted, so every target computes the same bits.

**D8 — Calibration.** `calib.cu math` measures, for each device unit, the worst relative and absolute error over every
f32 in its domain against the f64 function, whether it returns the contract's special values, and its throughput
(cycles a warp instruction, clock64 over independent chains); and the composed forms' error over the whole f32 range
beside CUDA's `sqrtf` / `sinf` / `cosf`. The profile stores the measured values (`Bound` with the calibration line as
evidence); an unmeasured part's units carry no `Bound`, so it derives the reference.

**D9 — `precision_plan`.** Beside `exp(m)` (ADR-0042 D6, unchanged) the fact carries `math(m)` for every math op: the
builtin, its width and its contract. A value gate composes a default-face path through a builtin as ADR-0042 did for
`exp`: the output's error is the contract's plus the input's relative error times the condition number
(`MathForm::cond`: `|x|` for `exp`, `½` for `sqrt`, `|x·cot x|` for `sin`, `|x·tan x|` for `cos`).

**D10 — OFF arms.** `--no-math-hw` / `--gpu-off=math-hw`: every builtin takes the exact face's choice on every face
(the reference, or an `Exact` unit such as IEEE `sqrt`); `--no-exp-hw` / `--gpu-off=exp-hw` stays `exp`'s alone.

## Implementation and measurements (2026-11-25, `local-rtx-5080/math.md`)

| | measured | bound |
| --- | --- | --- |
| `sqrt_def` (plain operations) vs IEEE `sqrt`, every f32 | 0 differ | bit for bit |
| `S_32` / `C_32`, every f32 against the f64 truth | ≤ 1.586 / 1.580 ulp, 0 outside | `2⁻²¹` relative |
| `S_64` / `C_64`, 4·10⁵ samples of every exponent vs glibc | ≤ 2.5 ulp of libm | `2⁻⁵⁰` |
| RTX 5080 `sqrt.approx.f32`, every positive f32 | 2^−23.253, specials ok, 8.13 cycles a warp op | admitted (`2⁻²²`) |
| RTX 5080 `sin.approx.f32` on `[−0.7854, 0.7854]` | absolute 2^−21.457, relative 1 | refused |
| RTX 5080 `cos.approx.f32` on the same | relative 2^−21.580 | the `cos r` part (`2^−21.12`) |
| RTX 5080 default face, 65 536 cells to `3·10²⁵` | `sin` 2^−21.69, `cos` 2^−21.53, `sqrt` 2^−23.29 | 0 outside |
| RTX 5080 exact face, the same cells | the interpreter's bytes | bit for bit |

`exp`'s module text is ADR-0042's byte for byte (`backends/llvm/tests/golden/exp-adr-0042.ll`, dumped from the
pre-framework printer; every face × contract × arm × part combination compared before it was removed).

## Consequences

- One definition per builtin feeds the interpreter (native evaluation of the generic definition), the rewriter's
  constant folder, the LLVM host and device modules (one printer of traced definitions, `backends/llvm/src/math.rs`)
  and the CUDA backend (one C printer) — a new builtin is a reference, a contract and its forms in `mapal-ir`.
- A new target is a property set: its units and definitions as data; no backend code changes.
- The exact faces cost what the reference costs where no `Exact` unit exists: `sin` / `cos` take the Payne–Hanek
  reduction on every `|x| ≥ π/4` (≈ 70 operations f32, ≈ 150 f64) — the default face takes the contracted reference.

## Alternatives rejected

| alternative | why not |
| --- | --- |
| libm / CUDA `sinf`, `sqrtf` per target as the semantics | the bits differ between platforms on the exact faces (glibc vs CUDA vs LLVM's constant folder) |
| One hand-written copy per builtin per backend (ADR-0042's shape) | four copies of each algorithm, and an ASIC target would need code in every backend |
| A target switch in the backends (`if target == …`) | CLAUDE.md: hardware is properties; the switch hides the property (the unit's error) the choice depends on |
| Cody–Waite reduction only | its error grows with `\|x\|`: no relative bound over the whole range near the zeros of `sin` |
| An absolute default-face bound for `sin` / `cos` (admits MUFU `sin.approx`) | `sin(1e-5)` off by percent is not ADR-0040's full precision; the cost saved is a few FMAs |
| A memory table for `2/π` (fdlibm's `ipio2`) | a load a lane (a gather when vectorized) and a constant segment on every target; selects over constant words are plain operations |

## What would falsify it

An exact-face output that differs from the interpreter on any backend or emulator mode; a default-face output outside
D3's bound; a unit whose measured error exceeds what its profile declares; an admitted target definition whose trace
measures outside its declaration.

## The interpreter

It runs the reference definitions natively (`math::eval_f32` / `eval_f64`; `sqrt` as Rust's correctly rounded `sqrt`,
equal to `sqrt_def` by construction and by an exhaustive test) — the oracle on every face; `mapal test` and the
differential tests compare the exact faces with it byte for byte, the default face within D3.
