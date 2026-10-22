# ADR-0039: Reduced-precision formats — storage types and conversion morphisms

Date: 2026-10-20 · Status: candidate — NOT decided · number provisional · changes nothing until accepted
Implements: ADR-0032 D1 (precision lattice), D2 (conversions are explicit morphisms), D3 (capability
matrix), D5 (no precision via ambient config). Related: ADR-0029 (`widen_*`), ADR-0028 (fold order),
plan-s41 S52 revision (the PTX leg).

## Context

S52 built the GPU tile kernel (PTX via the shared emitter) and measured it on Ada, Hopper and consumer
Blackwell. Two facts force this ADR:

1. **The tensor cores are where model arithmetic runs, and Mapal cannot ask for them.** On an RTX 4090
   the TF32 `mma.sync` kernel runs 77.9 TF/s against the FP32 FMA kernel's 52.8 (`benches/results/gpu/kernels/initial-comparisons/
   v4-rtx-4090.txt`). It exists today only behind `emit --precision=tf32` — an **ambient flag that
   changes rounding**, which ADR-0032 D5 forbids except as an interim form (the `--contract`
   precedent).
2. **Hopper's tensor path needs a storage format the language does not have.** On H100 cuBLAS-TF32 runs
   234–255 TF/s through `wgmma`; Mapal's `mma.sync` reaches 76–88 (`v4-h100-pcie.txt`). `wgmma` with
   TF32 operands requires both operands K-major; a row-major `b` (lanes contiguous) is accepted only for
   16-bit operands (`tnspB`). Models store BF16/FP16 anyway — the format is the program's fact, not the
   backend's.

## Decision

**D1 — Two storage types, one format restriction.**

| type | storage | value set | arithmetic in the interpreter |
| --- | --- | --- | --- |
| `bf16` | 16 bits | IEEE-754 binary32 values with 8 significand bits | none — operands widen to `f32` (exact) |
| `f16` | 16 bits | IEEE-754 binary16 | none — operands widen to `f32` (exact) |
| `f32` restricted by `to_tf32` | 32 bits | binary32 values with ≤ 10 explicit significand bits | ordinary `f32` |

`bf16` and `f16` are **storage formats**: every arithmetic morphism on them is defined as widen → `f32` op.
That makes the oracle trivial and exact, and it is what the hardware does inside an `mma` (products of
16-bit operands are exact in `f32`).

**D2 — Conversions are morphisms, never implicit.** `to_bf16 : f32 → bf16`, `to_f16 : f32 → f16`,
`to_tf32 : f32 → f32` round to nearest, ties to even, in the interpreter; `widen_f32 : bf16|f16 → f32`
is exact (the existing ADR-0029 family gains two sources). A tile site whose operands are produced by
`to_tf32`/`to_bf16`/`to_f16` *is* a reduced-precision site; nothing else is.

**D3 — The accumulation contract is the fold's, declared, not inferred from the operand type.** A
reduced operand format does not by itself license reordering the per-cell fold (that would let a type
silently change what `exact` means). The fold carries ADR-0032's class: `exact` (in-order, today),
`contract` (fused steps), or `block` — **the order inside one hardware k-step is the unit's**, with the
k-steps themselves in order. `block` is what `mma.sync`/`wgmma` implement; its oracle check is a bound,
not equality: |result − in-order f64 fold| ≤ the format's rounding bound × Σ|aᵢbᵢ|, checked by the
differential on every program, the S24b rel-tol pattern.

**D4 — Realizations are capability-selected, configurable, never contract-violating.** The GPU leg's
realizations (`mma.sync` m16n8k8/k16, `wgmma` m64nNk8/k16, `cp.async`, TMA, the FMA tile, one thread per
cell) are backend config (ADR-0032 D4): deduced by default from the site's operand formats and fold
class and the part's capability facts (`Gpu::sm()`), each with a measurement-only OFF arm
(`emit --gpu-off=<rung>`) and — new — a FORCE arm that may only choose among realizations the
contract admits. Forcing `mma` on an `exact` fold is an emit error, not a fallback.

**D5 — `--precision=tf32` and `--contract` are deleted** when D1–D3 land: the program says it.

## Interpreter

`Ty::Float { bits: 16, kind: Ieee | Brain }` (or a `Ty::BFloat16` sibling — an IR detail). Values are
stored as their bit pattern; `to_*` round with the IEEE rule; every op widens first. `print` of a 16-bit
value prints its widened `f32`. The `block` fold class evaluates in order (the oracle stays a left
fold); its differential is the bound in D3.

## Alternatives rejected

| Alternative | Why not |
| --- | --- |
| Keep `--precision=tf32` | ADR-0032 D5: a flag that changes bits is a second, invisible source program. |
| Make `bf16` arithmetic native (round after every op) | No hardware unit Mapal targets does it; it would make the oracle disagree with every `mma` beyond its bound. |
| Infer `block` from operand format | A format change would silently relax an `exact` fold; the two are independent facts (TF32 operands with an in-order fold is a legal, slower program). |
| TF32 as a storage type | It is not one: no part stores 19-bit values; it is a rounding of `f32`, so it is a morphism on `f32`. |
| Transpose `b` in shared memory for TF32 `wgmma` | Possible later as a realization; it does not remove the need for 16-bit storage. |

## What would falsify it

- A tensor-core realization that cannot meet D3's bound on some program the differential generates.
- A 16-bit site where widen-then-`f32` (D1) disagrees with every target's unit beyond D3's bound.
- Measured: a model program where explicit `to_bf16` at the boundary costs more than the 2× byte
  reduction and the `wgmma` rate it unlocks.
