# ADR-0040: The default float face — full precision, bounded, order-unspecified, fastest per environment

Date: 2026-10-23 (supersedes the 2026-10-22 text "the reassociation face — a deterministic split-K", kept below as
Context) · Status: **accepted (Sapir, 2026-10-23 — "the default should always be full precision, error bound and
the fastest possible per default chosen environment (e.g. machine with cpu/gpu)"; on determinism as a requirement:
"why SHOULD it be an exact requirement?")**
Amends: ADR-0032 D1 (the default class is `default`, below; `exact` is opt-in). ADR-0032 D5 stands: the face is a
language-level permission (interim form: emitter flags, the `--contract` precedent); only its *realization* is
backend config. Related: ADR-0028 D4 (float folds sequential-pinned — now the `exact`/`contract` faces' rule, not
the default's), ADR-0039 (reduced formats, untouched: the default never changes format).

## Context (history — the 2026-10-22 opt-in face)

At M16 × N2048 × K7168, the ordered kernels Mapal and tuned CUTLASS emit take 220.9 µs (tuned CUTLASS, the best
ordered kernel measured) against cuBLAS's 51.2 µs; at M16 × N1024 × K4096, Mapal 0.176 ms against cuBLAS 0.0266 ms
(`docs/performance/gpu/kernels/competitive-cutlass.md`, `small-m-geometry-comparison.md`). The captured cuBLAS
dispatch at the first cell is a GEMM plus a separately launched `splitKreduce_kernel`, and it reproduces 5 of 256
sampled ordered-FMA outputs (`docs/performance/gpu/disassembly/executed-library-images.md`).

The IR says why an ordered kernel cannot follow. The site is a `TileSite` (`mapal-ir/src/algo.rs`) whose output
domain `rows·c` is independent per cell (`TilePlan`), and whose body is one `k` fold with a recorded operand order
(`mul_a_first`, `add_acc_first`). At M16 × N2048 that domain is 32 768 cells — 128 blocks of the one-thread-per-cell
kernel — while the RTX 5090 holds 1020 such blocks at once (`Launch::wave`). Seven eighths of the part idles, and the
only other parallel axis is the fold, which the oracle pinned in order (ADR-0028 D4).

S62 shipped an opt-in face (`--precision=reassoc`): `S` contiguous partial folds into a workspace, one deterministic
reduce launch in partition order. 47 of 48 measured split cells beat their ordered arm beyond null
(`benches/results/gpu/s62/splitk-reassoc/`). Sapir then moved the line: the default is the fastest bounded
realization, and bit-exactness is a testing mode, not a semantic.

## Decision

**D1 — Faces.** Two axes: *steps* (`separate`: `fmul` + `fadd`, two roundings; `fma`: one) and *order* of a tile
site's `k` fold (`sequential`; `fixed`: D3's partition order, bit-replayable; `free`: unspecified, run-to-run bits
may differ).

| face | steps | order | flag | oracle / gate |
| --- | --- | --- | --- | --- |
| **default** | fma | free | none | D2's a-priori bound, never bits |
| exact | separate | sequential | `--exact` | interpreter, bit for bit |
| contract | fma | sequential | `--order=sequential` | the CPU contract build, bit for bit |
| fixed (test/debug) | separate | fixed | `--no-contract --order=fixed` | interpreter replays the compiled order (D3), bit for bit |
| fixed + fma | fma | fixed | `--order=fixed` | D3's order computed by the harness, bit for bit |

The default never changes format (FP32 stays FP32; TF32/BF16/FP8 only via `--precision=`). It permits one-rounding
`fma` steps and any evaluation order of a tile site's `k` fold whose combine is the fold's own `+`. `Reduced::Reassoc`
is gone; order is `EmitOpts.order`, steps `EmitOpts.contract` (default `true`).

**D2 — The a-priori bound, checked by every default-face value gate.** `u = 2⁻²⁴` (f32; `2⁻⁵³` f64),
`γ_n = n·u / (1 − n·u)` (Higham 2002, Lemma 3.1), `M = |seed| + Σ_k |a_k·b_k|`. For every realization the default
admits — sequential, fixed split into `S`, atomic split into `S` combined in any order, stream-K pieces:

`|out − exact| ≤ γ_{K+1} · M`

A term in piece `i` (length `kc_i`) meets at most one product rounding, `kc_i` fold additions and `pieces − 1` combine
additions; `kc_i + pieces − 1 ≤ K` because pieces are disjoint and non-empty, and Lemma 3.1 holds for any summation
order. The sharper realization bound `γ_{kc+S}·M` (the old D3) is reported, not gated.
**Gate:** `ref` and `M` in f64 over the same operands; accept iff `|got − ref| ≤ γ_{K+1}(2⁻²⁴)·M + γ_{K+1}(2⁻⁵³)·M`.
Whole-program probes: each printed value within relative `2·γ_{K+1}` of the exact-face `.expect`, `K` the program's
longest tile-site `k` (`benches/gpu/value_gate.py`; `ponytail:` read from the probe stem until `mapal dump-ir`
prints fold lengths).

**D3 — Fixed order (test/debug).** The 2026-10-22 order verbatim: for a site split into `S` (`kc = K/S`),

```
P_0 = fold(seed, k ∈ [0, kc))       P_z = fold(0.0, k ∈ [z·kc, (z+1)·kc))  z ≥ 1
out = ((P_0 + P_1) + P_2) + … + P_{S−1}
```

realized as partial folds into a workspace and one reduce launch; bits are a function of `(site, profile)`. The
emitter publishes the orders it compiled (`Emission.orders`: fold morphism, `S`), the interpreter replays them
(`mapal_interp::run_ordered`). `mapal test` runs this face: the oracle computes no `S`, it is handed the compiled one.

**D4 — Free-order realizations are rungs, each with an OFF arm:** `splitk` (D3's workspace + reduce, deterministic,
legal under `free`), `splitk-atomic` (partials `atomicrmw fadd` into a zeroed output: no workspace, no second
launch), `streamk` (a resident grid walks the `tiles × panels` iteration space in equal shares; a piece of a tile
ends in the same atomic add). Legality reads IR facts only: one writer per output cell (`TilePlan`), the step
`acc ⊕ a·b` with `⊕ = +` (`TileSite.{mul_a_first, add_acc_first}`), a constant seed carried by piece 0, `k` cut into
whole panels (`bounds_proof` covers every piece's reads), `f32`/`f64` (`red.global.add` exists on every profile's
`sm`), no epilogue group, row-major store, one stream. Forbidden on the device: fast-math flags on emitted
instructions (the text fixes the arithmetic; LLVM may not re-bracket). On the host: only `reassoc` and `contract`
flags, never `afn`, `arcp`, `ninf`, `nnan` or `nsz` — order may move, a step's rounding may not.

**D5 — Per-part default = measured.** A rung ships OFF for `(part, fill, face)` only through a `GPU_RUNGS_OFF` row
with a cite. The 3090 `splitk` rows (`Face::Exact`) govern only the fixed face on separate steps. Rows for
`splitk-atomic` and `streamk` come from `benches/results/gpu/s63/default-float/`.

## Alternatives rejected

| Alternative | Why not |
| --- | --- |
| Determinism as a requirement of the default | Sapir: "why SHOULD it be an exact requirement?" Bits stay checkable where needed (`exact`, `fixed`); the default is checked against a bound. |
| A reduced format by default (TF32) | Changes precision, not just order; the default stays full precision. |
| `atomicAdd` partials | Rejected for the *fixed* face only (order then depends on scheduling). On the default face order is unspecified by decision: it is `splitk-atomic`. |
| A tree inside one block (ADR-0028 D1 shape) | Parallelism the block already has; empty SMs need more blocks, not more threads per cell. |
| The interpreter derives `S` | `S` is a function of the part; the interpreter has no profile. It replays `Emission.orders`. |
| A tuned `S` table per shape (the library way) | A constant per (part, shape) that no fact explains; the derivation states why, measured OFF rows say where it does not pay. |
| Fast-math (`afn`, `nsz`, `ninf`, …) on the host | Changes a step's result or special values, not just order: outside D2. |

## What would falsify it

A default-face output outside D2's bound (a compiler bug); a `fixed` output differing from D3's order, or from itself
run to run; a rung whose ON arm loses beyond the box's null on a part without an OFF row.

## The interpreter

It stays the `exact` oracle and computes no `free` result. Given `Emission.orders` it computes the `fixed` face's
bits (`run_ordered`), which is what `mapal test` and `crates/backends/llvm/tests/splitk.rs` check.
