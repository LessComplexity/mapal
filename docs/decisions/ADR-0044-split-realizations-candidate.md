# ADR-0044 (candidate): Split realizations of the default face — `{f32}` products on tensor cores within D2

Date: 2026-11-23 · Status: **candidate** (sprint step `f32-tensor-cores`; implemented for `narrow1` and `tf32x3` on
`mma.sync`, `benches/results/evidence/local-rtx-5080/f32-tensor-cores.md`). Supersedes ADR-0041 "Later: 3xTF32 under
`{f32}`"; amends ADR-0043 D4 (`{f32}` over two narrow operands: "if the accumulator meets 2⁻²⁴" becomes "if the split
realization's coefficient meets D2"). Builds on ADR-0040 (default face, D2), ADR-0041 (D3 member bound), ADR-0043.

## Context

1. A matrix instruction multiplies parts exactly (TF32 × TF32: 22 bits; bf16 × bf16: 16) and accumulates on a
   truncating datapath. Chaining a whole `K` on it costs `(K/k)·c_g·u·M` — ADR-0041's "Later" kept 3K accumulations
   there and needed a slack term D2 does not have.
2. Measured (`benches/local/mma_calib.cu`, RTX 5080): every warp format truncates its aligned addends below
   `2^(e_max − 23 − g)` with **`g = 2`**, normalizes and truncates the sum; subnormal products, operands and `c` are
   kept.

## Decision

**D1 — The realization.** Per k-step (one instruction's `k`) and output tile, the kept part products are issued
smallest class first into a **zeroed** accumulator, then added to the f32 register accumulator by **one
round-to-nearest add** (`j` k-steps a flush). Pieces of a cut fold combine by round-to-nearest adds (ADR-0040 D4).

**D2 — The coefficient** (`mapal_ir::split`): `E = (e + C_j)(1 + γ_n) + γ_n/u`, `n` = flushes + combines, `e` the
form's split error, `C_j` the chain of one flush under the measured datapath (`c_g = 2((k+1)2⁻ᵍ + 1)` an instruction,
carried classes summed). **Admitted iff `E·u ≤ γ_{K+1}(u)`** — ADR-0040 D2 verbatim, the gates unchanged.

**D3 — Edges.** A non-finite operand, or a nonzero one below the form's `exact_above` (a part would round on the
subnormal grid), sets a flag; a warp any of whose threads flagged recomputes its tile by FFMA (IEEE products: the
oracle's value class; no absolute term enters the gate). A `hi` part rounded to infinity keeps the truncation.

**D4 — Faces.** Default face (free order) only: exact, contract and fixed faces keep the oracle's order. A member set
(`: tf32 | f32`, `: bf16 | f32`) keeps its member's realization (cheaper, its own D3 bound).

**D5 — Choice.** A property rule (`rules::split_rate_pays`: the part's matrix rate against `1.3·P` times its FFMA
rate) and the site's balance decide per site; `--gpu-off=split-tile` / `--gpu-on=split-tile` are measurement arms.

## Consequences

- RTX 5080: narrow1 (`{f32}` over two bf16 / f16 operands) 2.1–2.4× the FFMA tile; tf32x3 refused (TF32 = FFMA rate).
- ADR-0041 D3's member bound needs `g ≥ 1` (`AccModel::member_sound`): measured `g = 2` on every format.
- Later: bf16x6 / bf16x3n emission, the `wgmma` split, `j > 1` flushes, a value-range fact that drops the guard.
