# backend-nvptx — architecture

The GPU leg of the LLVM backend: `crates/backends/llvm/src/func/{gpu,resident,retain}.rs`,
`crates/backends/llvm/src/profile.rs` (device facts, rung table), `crates/mapal-rt/gpu/mapal_gpu.c`
(the transmissions). Plan of record: `plans/plan-s41-the-nvptx-leg.md` (S52 revision) and
`plans/plan-s54-fusion-placement.md`. FRAMEWORK.md terms throughout (§4.1 atoms, §4.2 placement
as a span, §4.5 laws).

Created S54 (2026-10-21). Before this file the component had a plan and a STATUS but no model of
what a *multi-computation* program looks like at the device: five parts measured
(`benches/results-s54-gpu/baseline-tables.md`) and nothing fused except the closed-form prologue.
Every per-cell consumer of a tile site ran on the host behind a whole copy-out; chains paid a
copy-out and a copy-in; independent sites ran one after another.

## 1. Why model it categorically

Because the expensive thing is not a kernel, it is a **transmission** the program never asked
for. On every probe the kernels alone are 0.62–1.39× the best library leg and the whole chain is
1.5–42× slower (`baseline-tables.md` §Multi-computation probes). The difference is `Trm`s —
`d2h`, `h2d`, host passes — placed where no fact required them. FRAMEWORK §4.5 law 1 says what
is required: a transformation reads data present at its location, delivered by a `Trm` or
materialised there. Fusion is what is left when no `Trm` is placed without a reason.

So: **fusion is not a pass** (plan-s41 S52 amendment 9). There is one placement rule (§5) over
the IR's facts, and a kernel boundary is where the rule had to place a `launch`.

## 2. `Dat` — the objects of a multi-computation program

Nothing here is new to the GPU leg: the IR's objects are immutable, sized statically, every
read is a morphism. The facts the placement reads are all deduced by `mapal-ir` (ADR-0032: no
machine fact enters the IR).

| Object | Semantics | Fact (file:symbol) |
| --- | --- | --- |
| array `x: [T; n]` | immutable, static `n` | `Ty::Array` |
| tile site `S` | `out[i,j] = fold_k a·b` with affine reads, `k > 0` | `algo.rs:TileSite`, `TileRead{ci,clane,ck}` |
| per-cell morphism `m` | a `Map` with no inner fold whose cell `t` reads a fixed set of cells | `algo.rs:ElemSrc::Apply` (law), **`cell_plan` (S54: the read footprint)** |
| row fold `R` | a `Map` over rows whose body is one in-order `Fold` over an affine read | **`row_fold` (S54)** |
| element read | a host `Index` at a proven constant | `BoundsProof::proven` |
| carried state | a loop-state component and the rows one iteration writes | `LoopPlan`, `write_delta → Delta::Rows` |
| death | the position after which no morphism reads `x` | `LastUsePlan::death`, `escapes` |
| independence | two sites with no dependency path | `PathPlan.tasks` |

The **footprint** is the one fact that existed in pieces and nowhere whole. For a reader `R` of
`x` and output cell `t`:

| footprint | meaning | who has it |
| --- | --- | --- |
| `Same` | `x[t]` | a captures-0 `Map` over `x`; a captured `x[t]` |
| `Row(c)` / `Col(c)` | `x[t ÷ c]` / `x[t % c]` | a bias, a per-row scale |
| `Const(k)` | one cell | a host `Index` at a constant |
| `Panel` | a row or column of `x` | a tile site operand (`TileRead`), a row fold |
| `Whole` | everything, in order | a whole-array fold; anything unrecognized |

`cell_plan : (fn) → MorphismId ↦ CellSite{extent, reads: [(slot, Footprint)], body}` is the
S54 query, ~120 lines beside `move_plan`, geometry only. Its gate is `gen_law`'s (a clean leaf
body, no `Call`/`Map`/`Fold`, no float `%`) plus every `Index` proven.

## 3. `Trn` — the transformations

| `Trn` | `t_from → t_to` | realizations (`TrnLoc` rows) |
| --- | --- | --- |
| tile site | `(a, b) → out` | host nest (`tile.rs`); device `naive` / `tiled` / `mma` / `wgmma` (`gpu.rs`) |
| per-cell morphism | `(x₁…xₙ) → out` | host loop (`bulk.rs:emit_map`); `devgen` kernel (captures 0, closed form); **S54: epilogue in a site's store; `cell` kernel** |
| row fold | `x → r` | host nest; **S54: `rowfold` kernel, one thread per row** |
| whole float fold | `x → scalar` | host only (ADR-0028 D4: one in-order chain; one device thread is slower than the copy-out) |
| element read | `x → T` | host load; one-element `d2h` (`resident.rs:Use::Elem`) |
| restricted site | the site over the rows `Δ` says changed | `retain.rs:Retained.row` |

One `Trn` placed twice is two `TrnLoc` rows and may be two texts (FRAMEWORK §4.2). The rule
that keeps them the same *value*: a per-cell body is emitted once (`drive.rs`) and the device
text is the host's text — the same `fmul`/`fadd` with no `contract` flag on the exact face,
`llc -nvptx-fma-level=0` as the belt (`bundle.sh`). Placement is therefore numerics-free: the
exact face is bit-identical to the interpreter wherever a morphism runs (S53 differential
238–246/238–246 on five parts).

## 4. `Loc` and `Trm`

`Loc`: `Host` (the frame arena), `Gmem` (device global, per-object slot `%d<k>`; under `arena`
a straight-line object's slot is a compile-time offset of the function's one allocation), `Smem` (one
block's shared memory), `Reg` (one thread). Adjacency: `Host ↔ Gmem` over PCIe; `Gmem → Smem →
Reg` inside an SM; `Reg → Gmem` by a store. A value in `Reg` is visible to one thread, in `Smem`
to one block, in `Gmem` to every block **after a grid-wide wait**.

| `Trm` | `c_from → c_to` | carries | cost fact (S52/S54 traces) |
| --- | --- | --- | --- |
| `h2d` | `Host → Gmem` | a whole array, or (S54) the rows `Δ` wrote | 64 MB: 3–5 ms pageable; `cuMemcpyHtoD` |
| `d2h` | `Gmem → Host` | a whole array, or one element | 64 MB: 3.2–7.3 ms; synchronous |
| `launch` | `Host → Gmem` (control) | a kernel and its pointer arguments | 5–10 µs; asynchronous on a stream |
| grid-wide wait | `Gmem → Gmem` | the fact that every block stored | implicit between launches on one stream; an event across streams |
| `g2s` (stage) | `Gmem → Smem` | a panel | `cp.async` / vector loads; completed by `barrier0` |
| `s2r` | `Smem → Reg` | a fragment | `ld.shared` / `ldmatrix` |
| store | `Reg → Gmem` | an output tile | one vector store per `lanes` columns |

Law 2: each of these crosses a boundary; a same-location flow (the host loop, the register
step) is a `Trn`. Law 1: the host fold reads `d` at `Host`, so `d` is delivered by `d2h` or the
fold does not run — there is no third option, which is why the whole-sum programs keep their
copy-out until ADR-0028's canonical tree exists.

## 5. The placement rule

Every morphism of a function is placed in topo order by the first clause that holds. The rule
reads the facts of §2 and the device facts of `profile.rs` (`sms`, `max_blocks_per_sm`,
`smem_per_sm`, `l2_bytes`), never a workload name.

- **R0 (anchor).** A tile site with `k > 0` and plain affine reads is a device site
  (`gpu.rs:gpu_site`). Its output cell exists in the owning thread's `Reg` before the store.
  A row fold is a device site (one thread per row, `k` in order). Every other `Trn` starts at
  `Host`.
- **R1 (fold into the store).** A per-cell morphism `m` runs **inside** device kernel `S`,
  between the accumulator and the store, when every operand of `m` is (i) `S`'s output at
  footprint `Same`, (ii) a member of `S`'s group already (a chain `x → y → z` composes by
  `elem_plan` to one law), or (iii) an array complete in `Gmem` before `S` launches, read at a
  footprint that is a function of the cell only (`Same`, `Row`, `Col`, `Const`): a bias, a
  residual, the other site of a fan-in. No `launch`, no `Gmem` for `m`'s input. The group
  stores each member some outside reader needs; a member with none never exists in memory.
- **R2 (per-cell device site).** Otherwise `m` is its own one-thread-per-cell kernel when every
  operand is in `Gmem` (device-produced, carried, or closed-form and so `devgen`-able) and every
  reader of its output is a device site, an element read, or a member of a later group. A reader
  that needs the whole array at `Host` (a whole fold, a print, an escape) forces `d2h`; then `m`
  runs where the bytes it moves are fewest — on the device when its operands are already there
  (one `d2h` either way), on the host when they are not.
- **R3 (kernel boundary).** A reader whose footprint is `Panel` or `Whole` of a device output
  needs every block's store: `Gmem` plus a grid-wide wait — a `launch`. This is amendment 9's
  sentence with the fact that decides it. The output stays in `Gmem` (`resident`); the reader
  launches on it.
- **R4 (host).** A whole-array float fold (ADR-0028 D4), an unproven index (device traps are
  not modelled), a body outside `gen_law`'s leaf gate, a morphism in a guard arm or loop its
  producer is not in: `Host`, delivered by `d2h`.
- **R5 (retention, part of object).** Inside a loop, a device site over carried `x` whose
  iteration wrote rows `Δ` (`write_delta`), with `w` invariant and row-local reads, recomputes
  only `Δ` into its **resident** output (`retain.rs` rule, now on the `resident` buffer). The
  output inherits `Δ`, so R5 propagates through `Same` consumers and stops at the first `Panel`
  reader. A host write to a carried array with `Δ::Rows` uploads `Δ`, not the array.
- **R6 (residency and death).** An object a device site reads is copied in once after its
  producer; a device output is copied out only for a host reader, whole or one element; every
  buffer is freed at `death` (`resident.rs`). Device-only objects get no host slot and live in
  one arena laid out at compile time from death positions.
- **R7 (concurrency).** Device sites on different `PathPlan` tasks launch on different streams;
  a cross-task edge is an event. Streams are opened only when the union of the concurrent grids
  is under one wave (`blocks < sms · max_blocks_per_sm`, from the launch geometry): a filled part
  has nothing to overlap. On sm_90+ a dependent launch in one stream may begin its prologue
  before the producer's grid drains (`griddepcontrol`).

**Kernel boundaries are the consequence.** A program's launches are exactly the R0/R2 sites
after R1's groups, and the `Trm`s are exactly R3's waits, R4's and R6's copies, R7's events.
"Fuse" is never spelled in the code; the probes' static counts (`benches/gpu/fusion_report.py`)
and the trace (`MAPAL_GPU_TRACE`) are what say whether a boundary exists.

| probe (`gen_tiles.py`) | shape | today (base) L / d2h / h2d | after R1–R7 |
| --- | --- | --- | --- |
| epi, bias, residual | site → per-cell → corners | 1 / whole / 0, host map | 1 / 8 B / 0 |
| ew | three per-cell maps, closed-form inputs | 0, four host passes | 1 gen / 8 B / 0 |
| pro | per-cell → site | 1 / 8 B / 0 (devgen) | unchanged |
| mlp | site → per-cell → site | 2 / whole / whole | 2 / 8 B / 0 |
| indep, diamond | sites ∥ → per-cell join | 2–3 serial / whole | 2–3, join in the last store; streams under one wave |
| fanout | one output, three readers | 2 / 3 × whole | 2 / whole once (the host reads `c`) |
| rowsum | site → per-row fold | 1 / whole | 2 / 4 B |
| loop | `x ← relu(x @ w)` × 8 | 8 / 8 × whole / 8 × whole | 8 / 0 / 0 |
| decode | carried rows, invariant weights | 3 whole launches per step, 1 MB each way | row launches, 1 KB each way |

## 6. The coherence laws this must satisfy (§4.5)

1. **Placement honesty.** R1's condition (iii) *is* law 1: an epilogue operand must be complete
   in `Gmem` before the launch — a fan-in's first site is topo-before the second (one stream)
   or joined by an event (R7). Reading a staged panel before `barrier0` is the same law failing
   one level down (STATUS.md's `__syncthreads` note).
2. **Transmission well-typing.** Every `Trm` row above carries a typed datum; a one-element
   `d2h` carries `T`, a delta `h2d` carries `[T; stride]`. The host loop and the register step
   are `Trn`, never `Trm`.
3. **Placement totality.** Every morphism has a `TrnLoc` after R0–R7; the fallthrough is R4,
   which is always defined. A `CellSite` the kernel cannot address falls to R4, not to a panic.
4. **Dependency mediation.** A device reader of a host-produced array is mediated by `h2d`
   (R6); a host reader of a device output by `d2h`; two streams by an event. The trace counts
   each; the static report counts the calls.
5. **Composition soundness.** A group (R1) is the composite of its members: its stores are the
   union of the members' outside-needed outputs, nothing redescribed. The device unit's text for
   the group is the host bodies' text, so `data(C) = ⊎ data(members)`.
6. **`runsAt` is a relation.** A per-cell body has up to three `TrnLoc` rows (host loop,
   epilogue, cell kernel); the differential checks all of them against one oracle.

## 7. What a rung is here

Every placement clause is a measurement-only arm (`GpuRungs` field, `emit --gpu-off=<name>`,
an arm in `bundle.sh`), shipped ON per part only by a `GPU_RUNGS_OFF` row whose cite names the
probe program and the ON/OFF medians: `epilogue` (R1), `devmap` (R2), `rowfold` and
`rowfold-layout` (R0 for folds, the store order), `retain-resident` and `delta-h2d` (R5),
`arena` (R6), `streams`, `pinned`, `pdl` (R7), `raster` and `l2-window` (the grid order and L2
policy from `l2_bytes`; `l2-gate` is `raster`'s measurement arm, the derivation without its L2
gate). OFF is today's text byte for byte. Neither `raster` nor `l2-window` is a `Trn` or a `Trm`:
the grid order is how one realization walks its blocks, the window a cache policy on a `launch`
over `Gmem` the graph proves is read again — `l2-window`'s output case is R6's residency seen from
the L2 (a chain's intermediate, read by the next launch). A rung's fill key is the grid it
ends up in: the host kernel for an epilogue, its own grid for a cell kernel, the union for
streams (`profile.rs:Fill`).

The table is a rung too (`arch-rungs`; `emit --gpu-off=arch-select`): OFF ships every rung ON,
the S52 text, so ON vs OFF prices the per-part selection on every kernel and probe. A row's face
key is the kernel the rung folds into: `Exact`, `Contract` (the FMA tile, and TF32 with it — a
contract face), `Tf32` (the `mma` kernel alone; S54: the 3090's `.minnctapersm` pays on the FMA
tile and costs 11% on `mma`). `benches/gpu/rungs.py` decides a rung measured for the first time
by the plan's rule (ON iff some size is > 1% faster in every session) and flips a decided row only
when every session measures the other sign; host-read rungs (`gpu.rs:host_rungs`) key fill `Any`.

## 8. What is not placed here

- The whole float sum on the device (ADR-0028 D4; the canonical-tree re-pin is Sapir's call).
- Precision placement (ADR-0039 is a candidate; no storage type, no flag deletion).
- Demand-restricted sites (every reader `Const(k)` → compute only those rows): correct and
  derivable, but it would hollow the timed probes (their outputs are read at the corners) into
  two-row GEMMs. It waits for a program that needs it and a harness that keeps a whole reader.
