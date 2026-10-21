# S54 research — the shapes the representation produces, and what each licenses on a GPU

Date: 2026-10-21 · Branch `s53-gpu-opt` @ 6d105b0 · Builds on `docs/notes/s53-research-{creative,artifacts,link,kernel}.md`. Gap evidence: `benches/results-s54-gpu/fusion-probe-static.txt` — of six multi-computation probes only the closed-form prologue fuses; every per-cell map runs on the host behind a whole d2h, because `gpu.rs:gpu_site` admits only `TileSite`s with `k > 0` and `resident.rs:Use::Host` then copies the output out whole.

## 0. What the IR knows that no library or framework knows

Objects are immutable, sizes static, every read is a morphism, the interpreter is the oracle. So for any value the graph answers exactly, at compile time: who reads it, which cells, in what order, how often, when it dies, whether it changed since the last read, whether recomputing it can trap. XLA has fusion, buffer and layout assignment; Inductor fuses pointwise chains and has opt-in epilogue templates; cuBLASLt has a fixed epilogue menu; CUDA/Triton see one kernel. None has: (a) bit-identity of every placement to an oracle — placement is numerics-free; (b) a trap-freedom proof per body as the licence to recompute instead of load (`algo.rs:elem_law`); (c) the per-iteration write delta of loop state (`algo.rs:write_delta`); (d) proven-constant partial reads that shrink the *producer* (`algo.rs:BoundsProof`, `resident.rs:Use::Elem`); (e) static death points with no runtime coherence (`algo.rs:LastUsePlan`; FRAMEWORK §4.5 law 1 discharged statically — every rule below is a placement, not a cache protocol).

## 1. The missing query: `Footprint`

Amendment 9's rule needs one fact that exists in pieces and nowhere whole: which cells of `x` does output cell `i` of reader `R` read?

| reader | source | class |
|---|---|---|
| tile-site operand | `TileRead{ci, clane, ck}` (`algo.rs:255`) | `Row` / `Column` (a panel) |
| per-cell map | `ElemSrc` leaves `Load{x}`, `Index` (`algo.rs:461`) | `Cell(i)` |
| host `Index` at a proven constant | `BoundsProof::proven` + the constant | `Const(k)` |
| `MoveSite` | `cq, cr` (`algo.rs:336`) | `Cell(σ(i))` |
| standalone `Fold` | the whole array, ordered | `Whole` |
| anything else | — | `Whole` |

`Footprint : (ObjectId, reader) → class`, ~80 lines beside `elem_plan`, geometry only (ADR-0032). It closes artifacts-note gap 2: a `Map` whose body is trap-free and whose array reads are `Index` at affine functions of `t` with proven bounds (`c[t] + bias[t % n]`) is a per-cell site. **The placement rule (amendment 9):**

1. consumer footprint `⊆ Cell(i)` of a device producer → computed **in the producer's store** (epilogue);
2. consumer is a device site, operand's law per-cell and trap-free → computed **in the consumer's load** (`g2s` prologue; `devgen` is the captures-free instance, shipped);
3. footprint is a panel or `Whole` → **launch boundary**, operand stays in `Gmem` (`resident`, shipped);
4. every reader `Const(k)` → **demand-restricted** producer (S7).

One rule, no pass; "fuse" is never spelled.

## 2. Shapes at whole-program scope

Each: fact (file:symbol) → GPU consequence → what others lack → effect → test. Every item is a `GpuRungs` field with `--gpu-off`, an arm in `bundle.sh`, shipped ON per arch only by `GPU_RUNGS_OFF` rows.

**S1 — per-cell map over a device output (epilogue).** `epi`, `bias`, `mlp`'s `h`, `indep`'s `g`. Fact: `ElemSrc::Apply{inner: Load{site out}}` under the trap-free gate; captured reads via `Footprint`. Consequence: the body runs on the accumulator before `STG`; the raw site output never exists. Others: cuBLASLt's menu (bias, ReLU, GELU) vs any clean body; Inductor only with `max-autotune` templates. Effect: one whole d2h (3.2 / 3.4 / 7.3 ms H100 / 4090 / 5090 at 4096) + a host pass + 2·n² bytes per map removed; largest on the PCIe parts. Test: differential N/N (per-cell ops are bit-exact wherever they run); OFF arm `epilogue`; vs cuBLASLt `EPILOGUE_BIAS`, eager `addmm`.

**S2 — per-cell chains, diamonds, fan-in (no fold).** `ew`: `x=a·3; y=x+2; z=y·b`. Fact: `elem_plan` composes laws through `Apply∘Apply∘Pair` to depth 16, so `z[t]` is one closed form of `a[t], b[t], t`; a diamond is `Pair(Apply{Load x}, Apply{Load x})` — one load. Consequence: a **per-cell device site** (`CellSite`: `gpu.rs`'s one-thread-per-cell kernel with the host body text `devgen` emits) over the composed law; `x`, `y` never materialize. Others: Inductor/XLA fuse pointwise chains — parity; vs eager 3 kernels + 2 intermediates. Mapal's extra: soundness by proof (trap-free gate), not by "pointwise ops don't fail". Test: OFF arm `cellsite`; vs `torch.compile`, eager.

**S3 — prologue with captures.** `pro` fuses today only because `a` is itself closed-form. When the operand is a parameter or loaded array with a per-cell law over it (`a2 = a·0.25`, `to_tf32(a)`), rule 2 computes it in `g2s`. Effect: one n² pass per converted operand; on H100 TF32 the per-fragment `cvt.rna` (kernel note cause 7) moves to the staging copy, once per element. Test: OFF arm `prologue`; TF32 value gate.

**S4 — chain of tile sites (`mlp`).** `c=a@b; h=c·0.5; d=h@e`: `h` is `Cell` over `c` and a panel operand of `d` — rules 1 and 3: `h` is `c`'s epilogue, resident, `d` launches on it. Today 2 d2h + 1 h2d + a host pass ≈ 12 ms vs 2 × 4.2 ms kernels (H100, 4096); expected ≈ 2× whole-program on every arch. Test: vs eager (3 kernels) and cuBLAS×2 + pointwise.

**S5 — independent sites (`indep`), overlap.** Fact: `PathPlan.tasks` with no dep path, `rank` (`algo.rs:161`); the fill key (`profile.rs:gpu_rungs`, warps per scheduler) says whether one grid fills the part. Consequence: (i) with S1, `g = c + d` is the epilogue of the second site reading the first's resident output — serial, transfer-free; (ii) the two sites launch concurrently (a stream per path, an event per cross-path edge) **iff** their grids together are under one wave — derived, not a flag: at 4096 each grid fills every part; at ≤ 1024 they fill idle schedulers. Others: streams are manual in CUDA/PyTorch. Effect: sub-wave sites only (CPU precedent 1.63×); H100 (132 SMs) most. Test: OFF arm `streams`; `indep` at 512/1024/4096. Programmatic dependent launch (sm_90) is the same edge at kernel granularity — speculative.

**S6 — reductions after tiles.** Every probe ends in `(0.0, d) -> fold`. ADR-0028 D4 pins float folds in order: a whole-array sum is one dependent chain anywhere — device ≈ 16.7M × 4-cycle adds ≈ 30 ms at 4096, host ≈ 10 ms after a 3–7 ms d2h; d2h-then-host is the correct placement. Two shapes are placeable within the rules: **row-local folds** (`map { i -> fold k -> acc + f(d[i·n+k]) }`: softmax denominators, norms) are `rows` independent in-order chains — a tile-site shape with no multiply, which `tile_site` refuses (it requires `a·b`); generalizing the recognizer to any trap-free per-step body over affine reads makes a per-row fold a device site, one thread per row, bit-identical. **Exact-op folds** (integer, `min`/`max`) may use ADR-0028 D1's tree now. The whole-float-sum tail waits for the deferred canonical-tree re-pin (Sapir's call). Test: `rowsum`, `softmax_rows`; OFF arm `foldsite`; vs `torch.sum(dim=1)`.

**S7 — demand: compute only consumed cells.** `sample`: `c=a@b; c[0], c[n²−1]`. Fact: every reader `Const(k)`, `c` dies after them, no `escapes`. Consequence: the grid is restricted to the rows read — `retain.rs:Retained.row` already builds that realization (a `rows = 1` `TileSite` offset to a row), driven here by the static read set instead of the delta. Others: nothing shrinks a GEMM by a downstream gather. Effect: n² → k·n. Honest note: a program that samples its output is not a GEMM benchmark, and a checksum fold defeats the rule by design — cheap and correct, not a headline. Test: differential; OFF arm `demand`; trace shows the restricted grid.

**S8 — loop-carried state, delta writes, invariants (`decode`, `loop_mm`).** Facts: `carried_by`, `LoopPlan.scc_objects`, `write_delta → Delta::Rows` (`algo.rs:81`). Shipped: `resident` and `retain-delta` — but exclusive, and `retain-delta` lost because it kept S52 round trips (S53 §4c). S54 queue #1: the retained output slot **is** a resident buffer; the row-restricted launch reads the carried `x` resident and writes one row of the resident `y`; `y` inherits `Δ`, so the rule propagates through row-local consumers and stops at the first panel reader. Others: every framework hand-writes this cache. Effect: per step O(d·T) instead of O(d·T²); baseline to beat is `resident`'s 0.63–1.11 s. Test: `decode_T` vs eager PyTorch with a hand-written cache.

**S9 — fan-out (`share`) and reader-chosen layout.** Fact: `EmissionClass::Named`, out-edges; `resident` already keeps one copy, freed at `death`. Open half: `MoveSite∘TileRead` composition folds a transpose into the consumer's strides (artifacts gap 10) and gives H100 `wgmma` its K-major `B` without a copy (kernel note R8); a TF32 operand read by two sites is rounded once (S3). Others: cuBLAS `transA` by hand; XLA layout assignment (parity). Test: OFF arm `fold-move`; `transpose_mm` vs `matmul(a, b.T)`.

**S10 — whole-program memory planning.** Facts: `death` + static sizes → interval coloring with compile-time offsets (the cuda crate's `arena_plan` precedent). Consequence: one device arena per function, one `cuMemAlloc` per process; `resident` stops allocating and paging 64 MB host buffers for device-only objects (S53 open #2). `pool` was rejected because the runtime allocator's first allocation cost 4–19 ms — a static arena has no such pass. Others: XLA buffer assignment (parity). Effect: 0.6–2.7 ms alloc/free per site plus host paging — part of 298 vs 232 ms `e2e_4096` on the 4090. Test: OFF arm `arena`; e2e wall.

**S11 — launch elimination (speculative).** `topo_order` and all extents are static → a function's device DAG is a CUDA graph instantiated once; a loop body, one graph per iteration. Bound: `decode` T=1024 × ~3 launches × ~8 µs ≈ 25–35 ms of 630–1100 ms (3–5%). Measure before building.

**S12 — precision demand (blocked on ADR-0039).** `Widen`/`to_tf32` are morphisms, so a fold class's demand on each operand is a graph fact; S3 places the conversion at the producer's store. No storage types, no flag deletion.

## 3. Multiple computations: the test matrix

Every probe holds several computations; the static count (launches / d2h / h2d, the `fusion-probe-static.txt` method) is the first measurement, before any box. Expected after §1's rule:

| probe | shape | today L/d2h/h2d | expected | rung |
|---|---|---|---|---|
| epi, bias | site → cell map → fold, 2 Index | 1/1/0, host map | 1/0/0 (+2 cells) | epilogue |
| ew | 3 cell maps | 0, all host | 1/0/0 | cellsite |
| pro | cell map → site | 1/1/0 fused | 1/0/0 | prologue |
| mlp | site → cell → site | 2/2/1 | 2/0/0 | epilogue + resident |
| indep | 2 sites ∥ → cell map | 2/2/0 serial | 2/0/0; concurrent at ≤ 1024 | epilogue, streams |
| diamond | x → 2 cell maps → combine | all host | 1/0/0 | cellsite |
| rowsum, softmax_rows | site → per-row fold | d2h + host | 2/0/0 | foldsite |
| sample | site read at 2 constants | 1/1/0 | 2 rows, 2 cells | demand |
| transpose_mm | move → site | copy + site | 1/0/0 | fold-move |
| loop_mm, decode | carried, invariant, delta | resident | row launches, 0 transfers | resident×retain |

Box protocol: differential N/N on both faces and every probe; OFF arm interleaved with ON; library legs (cuBLAS, cuBLASLt epilogue, CUTLASS, eager and `torch.compile` chains) in one process; five arches, Ampere (3090, A100) measured for the first time — `GPU_RUNGS_OFF` rows for `sm_80`/`sm_86` before anything ships ON there.

## 4. Rank

| # | item | new fact | certainty | effect |
|---|---|---|---|---|
| 1 | `Footprint` + `CellSite` + placement rule (S1–S4, fan-in) | `Footprint` | high | every chain probe → 1–2 launches, 0 transfers; beats eager and cuBLASLt's menu |
| 2 | `resident` × `retain-delta` (S8) | none | high | decode per-step O(d·T) |
| 3 | static arena, no host buffers for device-only objects (S10) | none | high | the alloc/paging share of e2e |
| 4 | row-local fold sites (S6) | recognizer generalization | medium | chains ending in per-row reductions stay on the device |
| 5 | reader-chosen layout, `MoveSite∘TileRead` (S9) | composition | medium | transposes free; H100 wgmma K-major |
| 6 | streams for sub-wave sites (S5) | none | medium | ≤ 1024 only |
| 7 | demand-restricted sites (S7) | none | high, low value | correctness-preserving; nil vs a checksum |
| 8 | CUDA graph per loop (S11) | none | speculative | ≤ 5% on decode; measure first |
| 9 | precision placement (S12) | ADR-0039 | blocked | — |
| 10 | whole-float-sum on device | ADR-0028 tree | blocked | — |

Items 1–3 need no decision. Item 1 is the S54 build; its first number is §3's static table, its second the box.
