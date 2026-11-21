# ADR-0043: Narrow float storage — `bf16`, `f16`, `f8e4m3`, `f8e5m2`: one rounding in, exact widening out, f32 compute

Date: 2026-11-21 · Status: **accepted-for-implementation** (sprint step "Half precision": "Real models run in
bf16/fp16/fp8; Mapal has only f32/f64 (+ tf32 compute) … implement the first COMPLETE slice end to end"). Slice 1
(D1–D3, D5–D8 for `bf16` / `f16` storage with f32 compute) is implemented; slices 2 (16-bit tensor-core products, D4/D6)
and 3 (fp8) are decided here and open.
Amends: ADR-0039 (candidate) — adopts its storage types and explicit conversions (D1, D2) as pipeline builtins;
its D3 `block` class stays replaced by ADR-0041's compute sets, which gain the narrow members (D4). Builds on: ADR-0029
(`widen_*` builtins), ADR-0040 (default face, D2 bound), ADR-0041 (compute precision on compute nodes,
`precision_plan`), ADR-0042 (`exp`).

## Context

1. **Models store 16 and 8 bits.** Weights, activations and KV caches are `bf16` (most), `f16`, and increasingly fp8
   (e4m3 weights / activations, e5m2 gradients). A decode step is bandwidth-bound: it reads the layer's weights once.
   The model-scale suite's decode rows (one Llama-3-8B-like layer, `model-scale.md`) move 164 MB of f32 weights a step
   where torch's bf16 reference moves 84 MB; Mapal could only spell f32 / f64 (+ TF32 *compute*, ADR-0041).
2. **ADR-0039 proposed the storage types and was never accepted;** ADR-0041 moved reduced precision onto compute nodes
   and listed "16-bit storage types" and "deduced narrow storage" as later work that needs them.
3. **The hardware's 16-bit path is a product with f32 accumulation.** `mma.sync.m16n8k16.f32.{bf16,f16}` (every part
   with warp `mma` since sm_80, the RTX 5080 included), `wgmma.m64nNk16` (sm_90a, row-major `b` admitted for 16-bit
   operands only), `mma.sync.m16n8k32.f32.{e4m3,e5m2}` (sm_89, sm_120, sm_90a's `wgmma` k32). Products of 16-bit
   operands are exact in f32; only the accumulation rounds.

## Decision

**D1 — Storage types.** A narrow type is a set of binary32 values stored in fewer bytes. It takes **no arithmetic,
comparison or print**: values move (bindings, arrays, tuples, captures, guards, loop state, `Index`, `Update`, `fill`)
and are converted. IR: `Ty::Narrow(NarrowFloat)` (not numeric: every arithmetic rule rejects it), constants
`Value::Narrow(fmt, bits)`.

| type | bytes | sign / exp / fraction | precision `p` | `emin` / `emax` | largest finite | ∞ | slice |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| `bf16` | 2 | 1 / 8 / 7 | 8 | −126 / 127 | (2 − 2⁻⁷)·2¹²⁷ | yes | 1 |
| `f16` | 2 | 1 / 5 / 10 | 11 | −14 / 15 | 65504 | yes | 1 |
| `f8e4m3` | 1 | 1 / 4 / 3 (OCP FN) | 4 | −6 / 8 | 448 | no (S.1111.111 NaN) | 3 |
| `f8e5m2` | 1 | 1 / 5 / 2 | 3 | −14 / 15 | 57344 | yes | 3 |

**D2 — Conversions are pipeline builtins (ADR-0029 / ADR-0042 form).** `x -> to_bf16`, `x -> to_f16` (slice 3:
`to_f8e4m3`, `to_f8e5m2`) round an `f32` **once**: IEEE-754 `convertFormat`, round to nearest with ties to even,
subnormal results kept, `±0` / `±inf` keep their sign, a NaN stays a NaN (payload unspecified — no surface observes it).
Overflow: `bf16` / `f16` give `±inf` past the threshold (f16: `|x| ≥ 65520`); the fp8 conversions **saturate** to
`±max` (the hardware's `cvt.rn.satfinite`; e4m3 has no infinity). `x -> widen_f32` / `widen_f64` accept every narrow
type and are exact. Any other source is **L1616** (`to_*`) / **L1614** (`widen_*`); a literal is never narrow
(`0.5 -> to_bf16`, constant-folded with the oracle's rounding). An `f64` source is open: it needs one rounding, and a
legalizer that goes through f32 rounds twice (realization: round-to-odd to f32, then ties-to-even). IR:
`Operation::Narrow : f32 → Narrow(fmt)`; `Operation::Widen` gains `Narrow(_) → f32 | f64`.

**D3 — Semantics and bounds.** The interpreter computes the conversions (`mapal_ir::narrow`: `round`, `widen`); they
are exact operations on **every** face — every backend computes the oracle's bits (D5, D6), so they add nothing to any
face's bound. Arithmetic happens in `f32` (or `f64`) on widened values, under ADR-0040 / ADR-0041 unchanged. Hence:

- A product of widened narrow values has exact operands: a tile site whose operands are narrow storage has the f32
  site's bound, `γ_{K+1}(2⁻²⁴)·M` on the default face (`precision_plan.bound` covers `tile_plan_narrow`'s sites).
- **Through a rounding**, a default-face value gate composes: if the operand carries error `e` (`|x̃ − x| ≤ e`), then
  `|round(x̃) − round(x)| ≤ e + 2·(u·max(|x|, |x̃|) + abs)` — the rounding is monotone, and a flip across a tie costs at
  most one ulp. `precision_plan.narrow(m)` records each rounding's `NarrowForm` (`u = 2⁻ᵖ`: bf16 2⁻⁸, f16 2⁻¹¹; `abs` =
  half the least subnormal: 2⁻¹³⁴, 2⁻²⁵; `max`).
- **`exp` of a narrow input computes in f32**: `w -> widen_f32 -> exp` is `E_32` on the exact faces and ADR-0042 D4's
  `2⁻²¹·eˣ + 2⁻¹⁴⁹` on the default face; stored back (`-> to_bf16`), the rounding composes as above. There is no
  narrow `exp` (ADR-0042 D1: `f32` / `f64` operands).
- Nothing is rounded the program did not round: no realization may store an `f32` value narrow (that is "deduced narrow
  storage", ADR-0041 Later, which needs the program's permission).

**D4 — Narrow members of the compute set (slice 2).** ADR-0041's `P` gains `bf16`, `f16` (and slice 3 `f8e4m3`,
`f8e5m2`): `: bf16 | f32` admits products whose multiplicands are rounded to the member (`u_t` = its `u`), accumulated
in f32 — ADR-0041 D3's bound with that `u_t`. `f16` / fp8 members are admitted only where a value-range fact proves
the multiplicands below the member's `max` (no silent overflow; fp8 additionally needs the program's own scale,
`x / s -> to_f8e4m3`). Where both operands are **already narrow storage** (widened exactly), rounding them to a member
at least as wide is the identity: the tensor-core product is exact, so `{f32}` itself admits it **if** the part's
measured accumulator error meets `u_a = 2⁻²⁴` (a `TargetProfile` property, `Caps::mma_acc`, measured like
`Caps::ex2`); a tensor core that truncates its aligned addends (`u_a = 2⁻²³`) needs the program's `: bf16 | f32`.

**D5 — CPU lowering.** Loads and stores of 2 bytes; `widen` is `fpext` (a shift for `bf16`, F16C `vcvtph2ps` for
`f16` — both vectorize). `to_bf16` is the integer sequence `(b + 0x7fff + lsb) >> 16` (NaN quieted): exact with
subnormals, five integer operations, vectorizes — AVX512-BF16's `vcvtneps2bf16` flushes subnormal inputs (outside
every exact face) and the soft-float `__truncsfbf2` is a call per element. `to_f16` is `fptrunc`: F16C `vcvtps2ph`
(round to nearest even; checked against the oracle on every 13th f32) where the ISA has it, `__truncsfhf2` elsewhere.
Compute stays f32 on widened values (slice 1). Slice 2 reads host capability properties (`dpbf16`: VDPBF16PS pairs
with f32 accumulation; AMX-BF16 / AMX-FP16 tiles) under D4's permission — their subnormal flushing is an absolute term
the permission's bound must state; a host without the property widens and computes in f32.

**D6 — GPU lowering.** `ld/st.global.b16`; `cvt.rn.bf16.f32` / `cvt.rn.f16.f32` (round to nearest even, subnormals
kept) and `cvt.f32.bf16` / `cvt.f32.f16` — the oracle's bits, one instruction each. Tile sites over narrow operands
are an IR fact (`tile_plan_narrow`, D8). Slice 1 realizes them with the one-thread-per-cell kernel (`naive` and its
split / atomic / post forms), which loads each operand element in its format and widens it, and with the
register-staged FMA tile, which widens each panel as it copies it (`<4 x bf16>` → `fpext` → an f32 panel in shared
memory: every fragment read and FFMA after it is the f32 tile's); a `cp.async` copy moves bytes unconverted and the
TF32 `mma` / `wgmma` tiles stage `elem`-wide fragments, so they refuse a narrow operand. Slice 2: 16-byte `cp.async`
panels of 8 elements, `ldmatrix(.trans)` of `.b16` tiles, `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` (and `f16`),
`wgmma.m64nNk16` with `tnspB` on parts with `Caps::wgmma`; slice 3: `mma.sync.m16n8k32…e4m3` / `e5m2` and
`cvt.rn.satfinite.e4m3x2.f32`. Each instruction is a capability **property** (`Caps::mma16`, `Caps::mma_f8`: the
`m × n × k` shape a warp issues, `Caps::cvt16`), never a compare of an architecture number; the rung rules price the
realization from the shape and the bytes (D7).

**D7 — Storage in the cost model.** Every byte count reads the storage width, not the product's: `TileRead::elem_bytes`
(per operand), `PlanShape.{ea, eb}` (the plan model's L2 panel term `bw·(bm·ea + bn·eb)·kb` and DRAM term
`rows·k·ea + k·c·eb + out`), `profile::site_bytes` (the placement roofline), `row_split_pays` on `b`'s bytes, and
`residence_plan` / `unit_plan` / `elem_bytes` (2 for a narrow element). A decode GEMV over `bf16` weights is priced at
half its weight stream.

**D8 — Facts.** `tile_plan_narrow` (mapal-ir, stored on the graph): sum-of-products sites whose operand is the exact
widening of a narrow array's element (`acc + x[i·k + k'] · (w[k'·n + j] -> widen_f32)`), with strided lanes and
epilogues; each such read carries `TileRead::narrow`. Disjoint from every other tile plan, so their consumers read what
they always read. `precision_plan.narrow` (D3). OFF arm: `--gpu-off=narrow-tile` (such a map is a per-cell kernel).

## Alternatives rejected

| alternative | why not |
| --- | --- |
| Arithmetic directly on `bf16` (implicit widening) | ADR-0032 D2: conversions are explicit morphisms; the reader would not see where a value is exact and where it rounds |
| `bf16` arithmetic rounding after every operation | No unit Mapal targets does it; a K-step fold would round K times at 2⁻⁸ |
| `Ty::Float { bits: 16 }` | 16 bits is two formats (`bf16`, `f16`), and every `bits`-keyed rule would take a narrow value for a number |
| AVX512-BF16 `vcvtneps2bf16` for `to_bf16` | flushes subnormal inputs: not the oracle on the exact faces |
| A per-tensor scale inside `to_f8*` | a second operand hidden in a conversion; the program writes its scale (`x / s`) |
| Deduce narrow storage from the consumers now | needs a permission the program states (ADR-0041 Later); explicit storage first |

## What would falsify it

A backend whose `to_bf16` / `to_f16` differs from `mapal_ir::narrow::round` on a non-NaN `f32` (tested: every narrow
value round-trips, every 97th f32 against the integer sequence, every 13th against F16C, generated programs on the
host and the emulated device byte for byte on the exact face); a narrow tile site outside the f32 site's bound; a
bandwidth-bound decode whose `bf16` weights do not move its time toward the byte ratio.
