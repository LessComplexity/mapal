# ADR-0040: The reassociation face — a deterministic split-K for tile-site folds

Date: 2026-10-22 · Status: **accepted (Sapir, S62 — "add an OPT-IN arithmetic face … Exact/Contract stay ordered and default")**
Implements: ADR-0032 D1 (the reassociation entry of the precision lattice) in the interim emitter form ADR-0032 D5
allows (the `--contract` / `--precision=tf32` precedent). Related: ADR-0028 D4 (float folds stay sequential-pinned;
its deferred "canonical-tree re-pin" is the escape hatch this face opens, opt-in only), ADR-0039 (the reduced
formats, untouched here).

## Context

At M16 × N2048 × K7168, the ordered kernels Mapal and tuned CUTLASS emit take 220.9 µs (tuned CUTLASS, the best
ordered kernel measured) against cuBLAS's 51.2 µs; at M16 × N1024 × K4096, Mapal 0.176 ms against cuBLAS 0.0266 ms
(`docs/performance/gpu/kernels/competitive-cutlass.md`, `small-m-geometry-comparison.md`). The captured cuBLAS
dispatch at the first cell is a GEMM plus a separately launched `splitKreduce_kernel`, and it reproduces 5 of 256
sampled ordered-FMA outputs (`docs/performance/gpu/disassembly/executed-library-images.md`).

The IR says why an ordered kernel cannot follow. The site is a `TileSite` (`mapal-ir/src/algo.rs`) whose output
domain `rows·c` is independent per cell (`TilePlan`), and whose body is one `k` fold with a recorded operand order
(`mul_a_first`, `add_acc_first`). At M16 × N2048 that domain is 32 768 cells — 128 blocks of the one-thread-per-cell
kernel (`gpu.rs:launch`) — while the RTX 5090 holds `sms · min(max_threads_per_sm / 256, max_blocks_per_sm)` =
170 · 6 = 1020 such blocks at once (`Launch::wave`). Seven eighths of the part idles, and the only other parallel
axis is the fold, which the oracle pins in order (ADR-0028 D4). Cutting it changes bits, so by ADR-0032's dividing
rule it is a language permission, never a performance rung on Exact or Contract.

## Decision

**D1 — A third, opt-in face: `reassoc`.** Interim form `emit --precision=reassoc` (`Reduced::Reassoc`); ADR-0032
D5 absorbs it into the type lattice with the others. It permits exactly one thing: **a tile site's `k` fold may be
evaluated as `S` ordered partial folds over contiguous `k` ranges, combined in partition order.** It is not a
format: each step rounds as on the face it rides on — `reassoc` alone is `fmul` + `fadd` steps, `reassoc` with
`--contract` is `fma` steps. Exact and Contract stay ordered and stay the default. TF32 is not combined with it.

**D2 — The oracle of the face is derived, fixed and computable.** For a site `(rows, c, k, seed)` on profile `P`,
`S = P.gpu_splitk(blocks, wave, k, unit, rows·c, elem)` (D5) and `kc = k / S`:

```
P_0 = fold(seed, k ∈ [0, kc))                    each fold: the site's step, in k order,
P_z = fold(0.0,  k ∈ [z·kc, (z+1)·kc))  z ≥ 1     operands in the site's recorded order
out = ((P_0 + P_1) + P_2) + … + P_{S−1}
```

The bits are a function of `(site, profile)`: identical run to run, launch to launch, process to process. They
are **not** the same across profiles — `S` reads the part's facts — which is what the face permits and what an
opt-in declares. `S = 1` is the ordered fold exactly.

**D3 — The a-priori bound.** With `u = 2⁻²⁴` (f32; `2⁻⁵³` for f64) and `γ_n = n·u / (1 − n·u)` (Higham 2002,
Lemma 3.1; summation §4.2), every product and partial sum above is one rounding, so with `M = |seed| + Σ_k |a_k b_k|`:

- ordered face: `|out − exact| ≤ γ_{K+1} · M` (`γ_K` when the step is `fma` or the seed is 0);
- reassoc face: `|out − exact| ≤ γ_{kc+S} · M` — a term of partition 0 meets `kc` additions in its fold and `S − 1`
  in the combine.

Since `kc + S ≤ K` for `2 ≤ S ≤ K/2`, the reassoc bound is never looser than the ordered one (K = 7168, S = 8:
`γ_904` against `γ_7169`). The difference to the ordered result is checkable: `|out_reassoc − out_ordered| ≤
(γ_{K+1} + γ_{kc+S}) · M`.

**D4 — Forbidden.** No `fast`/`reassoc` fast-math flag in any emitted instruction (LLVM must not re-bracket what
the text fixes); no atomics (the combine is one deterministic launch); no split on Exact, Contract or TF32; no
runtime choice of `S`.

**D5 — The realization (rung `splitk`, `TargetProfile::gpu_splitk`) reads facts only.** `blocks` is the grid the
ordered geometry gives (`gx·gy`), `wave` the blocks the part keeps resident at that geometry (`Launch::wave`:
SMs × the tile's `blocks_per_sm`, or the thread/block-slot limits for one thread per cell), `unit` the `k` steps
of one panel (`GpuTile::bk`, 1 for one thread per cell). Double `S` while `blocks·S < wave` (slots the output
domain leaves empty), `k % (2·S·unit) = 0` (every partition whole panels: each partition's reads are the site's
affine reads moved `z·kc` along `k`, inside the bounds the site proved, no remainder path), and
`2·S·rows·c·elem ≤ l2_bytes` (the partials the combine reads back stay in L2). No constant enters. The split kernel
is the ordered kernel with `z = ctaid.y / gy` folded into its operand bases and its store, its loop bound `kc`, its
seed `select(z = 0, seed, 0)`; the combine is one thread per cell, `S` loads, `S − 1` `fadd`s straight-line.
`raster` and `splitk` are exclusive by derivation (a grid is grouped only above one wave, split only below).
Scope: the resident path, a site with no epilogue group, row-major store, one stream; elsewhere the face emits the
ordered kernel (`ponytail` in `gpu.rs:emit_tiled_map_gpu` — split those when a measured program needs it).

**D6 — Measured default per part.** `splitk` is a rung like every other: where its ON arm is not faster than its
OFF arm by more than the box's null variation, a `GPU_RUNGS_OFF` row for that part, fill and face ships the
ordered kernel there — the face stays available, it just has nothing to win. S62 measured 48 split cells
(RTX 5090, RTX 4090, H100 PCIe, RTX 3090 × six cells × two step faces). 47 beat their ordered arm beyond null. The one
loss, RTX 3090 16 × 2048 × 7168 on exact steps, is the row `{rtx-3090, splitk, Filled, Exact}`
(`benches/results/gpu/s62/splitk-reassoc/`).

## Alternatives rejected

| Alternative | Why not |
| --- | --- |
| Split under Contract (cuBLAS's default) | Contract is ordered `fma`; a split changes bits. ADR-0032: a bit-changing permission is the language's, opted in. |
| `atomicAdd` partials into the output | Order then depends on scheduling: bits change run to run, and no oracle can name them. |
| A tree inside one block (ADR-0028 D1 shape) | Parallelism the block already has; the empty SMs need more blocks, not more threads per cell. |
| The interpreter computes `S` partitions itself | `S` is a function of the part; the interpreter has no profile. The derived oracle (D2) is computed where the profile is known: `tests/splitk.rs`, `tile_bench.cu --splitk`. |
| A tuned `S` table per shape (the library way) | A constant per (part, shape) that no fact explains and no other shape inherits; the derivation states why `S` is what it is, and the measured OFF rows say where it does not pay. |

## What would falsify it

A cell where the split arm is not faster than the ordered arm beyond the box's null variation (that row goes to
`GPU_RUNGS_OFF`); a split output that differs from D2's order (a bit-exact gate failure — a compiler bug, not a
bound question); a run-to-run difference in the split output (determinism is the permission's price); or an
output outside D3's bound.

## The interpreter

Unchanged: it stays the Exact oracle and computes no `reassoc` result. The face's oracle is D2's order, computed by
a test or harness that knows the profile; the OFF arm (`--gpu-off=splitk`) is checked against the interpreter
bit for bit (`crates/backends/llvm/tests/splitk.rs`).
