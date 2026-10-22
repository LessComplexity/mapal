# S54 research: fusion as placement — the rule, against today's code

Date: 2026-10-21 · HEAD `s53-gpu-opt` @ 6d105b0 · Probe record `benches/results/gpu/fusion/static-probes/fusion-probe-static.txt`
(programs `scratchpad/s54/fuse/*.mapal`, n=512, rtx-4090, both faces identical).

## 1. What today's code does, and why nothing fuses

| probe | shape | today |
|---|---|---|
| epi | `c=a@b; d=c*0.5+1` | 1 launch, whole `c` d2h, `d` on host |
| bias | `d[t]=c[t]+bias[t%n]` | same |
| ew | three per-cell maps, no site | no device code |
| pro | `a2=a*0.25; c=a2@b` | fused — but by `rewrite::MapFusion` composing two captures-0 maps into one, then `devgen` |
| indep | `c=a@b; d=e@f; g=c+d` | 2 serial launches, 2 whole d2h, `g` on host |
| mlp | `c=a@b; h=c*0.5; d=h@e` | d2h → host → h2d → launch |

Three facts in the code decide all of it:

1. `gpu::gpu_site` (`func/gpu.rs:129`) is the only device placement and admits only a `TileSite` with `k > 0`.
   A fold-less `Map` is never a device site, so `resident::uses` (`resident.rs:149`) calls every per-cell reader
   `Use::Host`, which forbids `keep` and forces the whole d2h in `emit_site_resident`.
2. Every kernel store (`Kernel::tiled` `done:`, `gpu.rs:1860`; `naive` `store:`, `:886`; `mma`; `wgmma`) writes one
   array through the fixed `(a, b, o)` signature (`Kernel::header`, `mapal_gpu_launch`).
3. No IR fact says what a captured per-cell body reads: `elem_plan`'s `Apply{captures, inner}` leaves `c[t]`,
   `bias[t%n]` as opaque `Index`es in the body (artifacts note, gap 2); `MoveSite` admits one read and no
   arithmetic; `tile_site` requires a product fold (`algo.rs:1140`).

`pro` fused only because no read is captured: `MapFusion` composed the chain and `gen_law` took it.

## 2. The rule

Locations: `Host`, `Gmem`, `Smem`, `Reg`. Transmissions: `h2d`/`d2h` (Host↔Gmem), `stage` (Gmem→Smem→Reg),
`launch` + grid-wide wait (the only way a value written by one block reaches another). A value in `Reg` is
visible to one thread; in `Smem` to one block; in `Gmem` to every block after a grid-wide wait.

**Placement of a per-cell morphism `m` (a `Map` with no fold, clean leaf body) is decided by the location of
its read footprint, in topo order, by the first clause that holds:**

- **R0 (anchor).** A `TileSite` with `k > 0` runs at `Gmem/SM` (today's `gpu_site`); its output cell `(i,j)` exists
  in the owning thread's `Reg` before it is stored.
- **R1 (fold into the store).** `m` runs inside device kernel `S` — no launch, no `Gmem` for the value — when every
  operand of `m` is (i) `S`'s output read at footprint `Same` (`out[t]`), (ii) an object already in `S`'s group (a
  chain), or (iii) a `Gmem`-resident array complete before `S` launches, read at a footprint that is a function of
  the cell's coordinates only (`Same`, `Row(C)` = `t÷C`, `Col(C)` = `t%C`, `Const`): a bias, a residual, another
  site's output (`indep`'s `g = c + d` folds into `d`'s store with `c` loaded — cuBLASLt's `beta=1` epilogue,
  derived). The group stores each member some outside reader needs; `c` with none is never materialised.
- **R2 (device map).** Otherwise `m` is its own one-thread-per-cell kernel iff fewer bytes cross the bus there:
  `Σ bytes(host-only operands) + bytes(out)·[host reads out] < Σ bytes(Gmem-only operands) + bytes(out)·[device
  reader exists]`; ties to the device (launch ~10 µs, 64 MB page copy 3–5 ms, S52 trace). R2 chains with `Same`
  footprints compose by R1 into one kernel.
- **R3 (kernel boundary).** A reader whose footprint is not a function of its own cell inside one block's tile — a
  row fold over `c/bn` blocks, a tile site, a gather — needs `Gmem` and a grid-wide wait: a launch. Amendment 9's
  sentence, with the fact that decides it.
- **R4 (host).** A whole-array float fold (ADR-0028 D4: one chain; one device thread over 16.7M cells ≈ 30 ms vs a
  3.2 ms d2h), an unproven index (device traps are not modelled), a body with `Call`/`Map`/`Fold`/float `%`
  (`gen_law`'s leaf gate), a morphism in a guard arm or loop its producer is not in.

Residency is unchanged: `resident.rs` already copies in at the producer, keeps while a device reader exists, frees
at death — it only needs R1/R2 sites to count as device readers (`Use::Dev`) and producers.

## 3. New facts (crate `mapal-ir`, no machine fact — ADR-0032)

**`cell_plan(f) -> SecondaryMap<MorphismId, CellSite>`** — the per-cell read footprint of a captured `Map`
(`algo.rs`, beside `move_plan`, ~120 lines reusing `tile_split`/`TileAffine`):

```rust
pub struct CellSite { pub extent: u64, pub reads: Vec<CellRead>, pub body: FuncId }
pub struct CellRead { pub slot: u32, pub foot: Footprint }
pub enum Footprint { Same, Row { c: u64 }, Col { c: u64 }, Const(u64), Affine { .. } /* general */, Other }
```

Gate: no `Fold`/`Map`/`Call`/`Print`/`TimeMs` in the body, `tile_trap_free`, every `Index` bound proven, the mapped
array an `iota` of the output's extent or the bare captures-0 source (`Same`). The `Footprint` query the S53
creative note (#1, #4) asked for; also what `resident::slot_use` needs to say `Dev` for a map.

**`row_fold(f)`** (for (f)): a `Map` over `rows` whose body is one `Fold` over `k` with a single affine read
`base + ci·i + ck·k` and a float `acc ⊕ e` step — `tile_site` minus the product. `TileSite{b: None}` would touch
every CPU rung, so a separate `RowFoldSite{rows, k, a: TileRead, op, seed}`.

**`write_delta` through `Same` maps**: `Delta::Rows` of a carried `x` propagates to `m(x)` when `m` reads `x` at
`Same` (one match arm); this keeps a retained site's fused epilogue retained.

## 4. The cases

### (a) Epilogue — `epi`, `bias`

*Deciding facts:* `Map{captures: 0}` whose source is a device site's output (`epi`; no new fact needed), or a
`CellSite` whose read of the output is `Same` and whose other reads are `Row/Col/Const` over Gmem-resident
arrays (`bias`; `bias` itself is captures-0 closed-form, so `devgen` builds it once `uses` says `Dev`).

*Code.* `func/gpu.rs`: `Kernel` gets `group: &[Epi]` (`Epi{body, reads: Vec<(slot, Footprint, ptr_arg)>,
store: Option<out_arg>}`); `header` takes the extra `ptr addrspace(1)` params; the four store sites (`tiled`
`done:`, `naive` `store:`, `mma`, `wgmma`) replace `store acc` with, per `Epi` in chain order, `%v = call T
@fnN_cell({ T %acc, ptr %capK, i32 %t })`, then a store where a reader exists. `@fnN_cell` is the host body
(`drive.rs:133`) emitted with the `Same` slot typed as the element and `emit_index` on that slot returning the
value (`ops.rs:590`, one branch; ~40 lines). `kernel_id` keys on the group's body names; `device_module` appends
the bodies as `devgen` does (same `@mapal_` assertion). `emit_site_resident` passes the extra pointers;
`mapal_gpu.c` gets `mapal_gpu_launchv(id, n, ptrs[], gx, gy, block, smem)`. `resident_plan` records group members
as `fused` (skipped on the host, outputs device objects).

*Exact face.* The body is the host's text: `fmul`/`fadd` with no `contract` flag on either face; `llc
-nvptx-fma-level=0` is already in `bundle.sh:49` (must stay; add a test: no `contract` in the device unit, no
`fma.rn` in exact-face PTX). The contract face fuses only the site's own step (`llvm.fma`). Register pressure on
the 128-accumulator `wide-fma` tile with a per-cell capture load: measured; the OFF arm is the control.

*v0 (ponytail).* Skip the `_cell` body variant: store `acc` to `o_c`, call the unmodified host body with the
generic pointer (`addrspacecast`), the same thread reloads its own store (L1 hit, program order). Saves the d2h,
host loop and launch; keeps `c`'s store traffic. v1 removes the store when `c` has no outside reader.

*Probe:* `epi`, `bias` → 1 launch, 0 d2h except the 4-byte `Index` reads (`MAPAL_GPU_TRACE bytes_out=8`), differential N/N both faces.
*OFF arm:* rung `epilogue` (today's text byte-for-byte).

### (b) Per-cell maps on the device — `ew`, `mlp`

*Deciding facts:* R2's byte inequality over `cell_plan` sites, operand locations from `ResidentPlan`. `mlp`:
`h` reads `c` (Gmem), nothing from the host → device; R1 then folds it into `c`'s store (`c` has no other reader:
never stored); site 2 reads `h` → `keep`, zero bus bytes. `ew`: nothing is on the device and `z`'s only reader is a
host fold → every map stays on the host (the inequality is strict) — the honest answer; the CPU's own fusion of
that chain is `APPLY_INLINE`, out of scope.

*Code.* `resident::device_sites` returns `enum DevSite { Tile, Cell(CellSite), RowFold }`; `slot_use` answers
`Dev` for a `Cell` read; `Kernel::cellmap` is `gen_kernel_text` generalised to `n` capture pointers and the
`Same`/`Row`/`Col` address forms; `emit_map` (`bulk.rs:86`) dispatches to it before the host loop. Placement and
R1 grouping (a map folds into the latest device producer of its `Same` operand) are computed in `resident_plan`
before the walk.

*Probe:* `mlp` → 2 launches, 0 d2h/h2d between; `ew` → unchanged (negative control, must stay byte-identical).
*OFF arm:* rung `devmap` (per-cell maps never device sites; `epilogue` depends on it only for captured forms).

### (c) Prologue beyond devgen

A per-cell producer `a2 = g(a)` of a tile operand where `a` has no closed form (a parameter, a loop view, an
earlier site's output). When `a` is a device output, R1 makes `g` its epilogue — once per cell, against a
prologue's `c/bn` applications per cell (exact use count decides the side). Prologue proper is `a` host-resident
or carried: `Kernel::tiled`'s `frag` closure (`gpu.rs:1810`, the Smem→Reg hop; `cp.async` bypasses registers, so
that is the only hop) applies `g`'s body per staged element. Same scalar op, same bits. After (a)/(b); OFF arm =
the (b) kernel. Rung `prologue`.

### (d) Fan-out, fan-in

Fan-out (`c` read by an epilogue and by a later site or the host): the group stores `c` too (R1's "every member
with an outside reader"); `resident` keeps/copies it by its own rules. Fan-in (`g = c + d`): R1(iii) — `c` is
Gmem-resident and complete (topo-before `d`'s site; with streams, an event). `indep` → 2 launches, `g` folded
into the second, 8 bytes out. No new fact beyond `cell_plan`.

### (e) Independent sites concurrently

*Facts:* `path_plan` tasks/deps (dropped on GPU profiles at `lib.rs:455`); the grid's fill from `launch`.
Streams pay only under one wave (`blocks < sms·max_blocks_per_sm`) or to overlap copies; at n ≥ 2048 one grid
fills every measured part, so expect ~0 there. *Code:* `mapal_gpu.c` — a non-blocking stream per chain, `launchv`
takes a stream index, `cuEventRecord`/`cuStreamWaitEvent` per cross-chain edge, `cuMemcpyDtoHAsync` +
`cuStreamSynchronize` before the host reader; `mapal_gpu_sync` stays `cuCtxSynchronize`. Chain id = the
`path_plan` component among device sites, assigned in `resident_plan`. sm_90+: `cuLaunchKernelEx` with
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` and `griddepcontrol.wait` before the dependent kernel's
first global load — rung `pdl`, H100 only, on the `mlp` chain. *Probe:* `indep` (two streams, one event).
*OFF arms:* `streams`, `pdl`.

### (f) Reductions after a site

Per-row in-order fold (`RowFoldSite`): one thread per row, `k` in order — ADR-0028 D4 holds because each chain is
one thread (a warp per row is legal only for D1's exact ops). Consecutive threads read `n` apart: uncoalesced.
The use fact fixes it: when the fold is the only reader of the site's output, the site stores in the reader's
order (column-major `o`) — the deduced-layout half of caching, rung `rowfold-layout`. R3 gives the fold its own
launch (a row spans `c/bn` blocks). The whole-array `sum` in every probe stays on the host with the whole d2h (R4)
until ADR-0028's canonical tree exists; the results say so. Kernel: `Kernel::rowfold` (~60 lines, `naive` without
`lane`). *Probe:* `rowsum` (`c = a@b; r[i] = fold_k c[i·n+k]`), host `sum` as the negative control. *OFF arm:* `rowfold`.

## 5. Items

| item | files | depends on | independent? |
|---|---|---|---|
| **A `cell_plan` + `write_delta` through `Same`** | `mapal-ir/src/algo.rs`, tests | — | yes (pure fact; consumer-coverage test) |
| **B epilogue** (a, d) | `gpu.rs` store sites, `header`, `kernel_id`, `device_module`; `resident.rs` `uses`/`resident_plan`; `drive.rs`/`ops.rs` `_cell` body variant; `mapal_gpu.c` `launchv` | A for captured forms; captures-0 (`epi`) needs nothing | yes, with v0 first |
| **C devmap** (b, c) | `resident.rs` `DevSite`, `Kernel::cellmap`, `bulk.rs` dispatch; prologue in `Kernel::tiled` | A; B's `launchv` | yes, but merges touch `resident_plan` with B: build B first, C on top |
| **D streams/pdl** (e) | `mapal_gpu.c`, `resident_plan` chain ids, `lib.rs` path_plan on GPU | `launchv` (shared with B; land it as its own 30-line commit first) | yes |
| **E rowfold** (f) | `algo.rs` `row_fold`, `Kernel::rowfold`, `tile.rs` dispatch, layout rung | — | yes |

First commit: `mapal_gpu_launchv` + `Kernel::header` over `n` pointers (B, C, D need it). B and C share
`resident_plan`'s group construction: one item at the plan level, separate kernels. Rungs for `GpuRungs`,
`emit --gpu-off` and `bundle.sh`'s whole-program arms: `epilogue devmap prologue streams pdl rowfold
rowfold-layout`; ON per part only on a positive, stable-sign measurement. Probes join `gen_tiles.py` at 256
(differential, both faces) and 4096 (timed vs cuBLASLt's fused-epilogue `sgemm` and an eager two-kernel chain
in `tile_bench`).

## 6. Expected, honestly

S52's trace at 4096: a fused epilogue removes 3.2–7.3 ms d2h, a host pass and (mlp) 5.5–15 ms h2d against a
2.3–4.2 ms kernel — chain programs go from ~3× the kernel to ~1×. The kernel gains nothing; the exact-face
0.5–0.66× gap stays. Streams ~0 at ≥2048, unknown below; prologue and `pdl` µs-scale. Every number lands in
`benches/results/gpu/` as it arrives.
