# plan-s54 — fusion as placement, on five parts

Date: 2026-10-21 · Base: `s54-base` @ e8c1967 (local) · Model: `../ARCHITECTURE.md` (R0–R7)
· Inputs: `docs/notes/s54-research-{shapes,fusion,arch,prior}.md`, the S53 notes,
`benches/results-s54-gpu/baseline-tables.md` (five parts, 12 probes, library legs),
`benches/results-s54-gpu/fusion-report-base.txt` (static counts).

Sapir's directives for the session are binding: fusion is a consequence of placement, every
optimization is derived from the dataflow shapes, multiple computations are tested together,
five architectures with per-arch selection, beat the libraries and report every loss, caching
is general (no named cache). The relayed request — *"also ensure on multiple computations that
can happen"* — is §3's test matrix: every item is measured on the 12 multi-computation probes.

## 1. What the baseline says the gap is

From `baseline-tables.md`, n = 4096, Mapal's own clock (`iter`) against the best library chain
leg, and the Mapal kernel time alone on the contract face against the same leg:

| probe | 3090 iter → kernel | A100 | 4090 | H100 | 5090 | where the loss is (static report) |
| --- | --- | --- | --- | --- | --- | --- |
| epi | 0.29 → 1.36 | 0.22 → 0.77 | 0.19 → 1.02 | 0.24 → 1.03 | 0.12 → 0.96 | whole d2h + host map |
| mlp | 0.36 → 1.39 | 0.24 → 0.72 | 0.22 → 1.05 | 0.29 → 1.03 | 0.16 → 0.97 | d2h + host map + h2d |
| indep | 0.30 → 1.35 | 0.22 → 0.74 | 0.20 → 1.06 | 0.25 → 0.94 | 0.14 → 0.96 | 2 whole d2h + host join; serial |
| rowsum | 0.17 → 1.29 | 0.18 → 0.76 | 0.12 → 1.03 | 0.16 → 1.04 | 0.08 → 0.97 | whole d2h + host fold |
| loop | 0.13 → 1.21 | 0.11 → 0.68 | 0.06 → 1.14 | 0.11 → 1.04 | 0.05 → 1.02 | 8 × (d2h + host relu + h2d) |
| ew | 0.013 | 0.006 | 0.014 | 0.006 | 0.005 | no device code at all |

The second number of each pair is the ceiling placement can reach on that part: once no `Trm`
is placed without a reason, the chain costs its kernels. On the exact face the kernel ceiling is
0.5–0.66× (two instructions per MAC, S53), a kernel loss this plan does not touch.

Decode (`decode_1024_256`, shipped `resident`): 0.83 s (4090), 0.94 s (5090), 1.08 s (H100),
1.47 s (3090), 1.31 s (A100) wall against a hand-written cuBLAS KV cache at 16 / 50 / 22 / 42 /
66 ms device time. The trace on the 4090: kernel 76 ms (K and V recomputed whole every step),
h2d 50 ms (the carried `x` re-uploaded whole after a one-row host write), d2h 100 ms, load 139 ms,
the rest host.

## 2. Every idea, ranked

Score = evidence × feasibility × derivability × uniqueness, each 0–3. Evidence is the measured
gap on the suite (3: a probe loses ≥ 4× to it; 2: a measured share of a probe; 1: a mechanism
with a bound; 0: nothing on the suite). Feasibility is for one builder on `s54-base` in one
worktree. Derivability: 3 = a fact the IR already proves; 2 = a fact plus a device fact; 1 = a
policy. Uniqueness: 3 = no language or library can know it; 2 = Mapal adds exactness or a proof
to a known technique; 1 = parity.

| # | idea (report) | ev | fe | de | un | score | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `cell_plan` / `Footprint` fact (shapes §1, fusion §3) | 3 | 3 | 3 | 2 | 54 | **build** — the seam of items 1–2 |
| 2 | epilogue: per-cell consumer in the site's store (shapes S1, fusion a/d, arch, prior) | 3 | 3 | 3 | 2 | 54 | **build 1** |
| 3 | per-cell device sites, chains, fan-in by `Trm` crossings (shapes S2/S4, fusion b/d) | 3 | 3 | 3 | 2 | 54 | **build 2** |
| 4 | `resident` × `retain-delta` + delta `h2d` (shapes S8, prior O2, S53 queue #1) | 3 | 2 | 3 | 3 | 54 | **build 5** |
| 5 | row-local fold sites (shapes S6, fusion f) | 3 | 2 | 3 | 2 | 36 | **build 4** |
| 6 | reader-chosen store order for a fold-only output (fusion f, prior O3) | 1 | 2 | 3 | 3 | 18 | **build 4**, as its second arm |
| 7 | demand-restricted sites (shapes S7) | 0 | 3 | 3 | 3 | 0 | reject (§5) |
| 8 | L2 window from reuse distance (arch, prior O4) | 2 | 3 | 2 | 2 | 24 | **build 7** |
| 9 | grid raster order from `l2_bytes` (arch, S53 kernel R9) | 2 | 3 | 3 | 1 | 18 | **build 7**, same item |
| 10 | streams per `PathPlan` task, events on cross edges (shapes S5, fusion e, prior) | 2 | 3 | 2 | 1 | 12 | **build 3** (required) |
| 11 | pinned host memory for arrays that cross the bus (S53 creative #3, prior) | 2 | 3 | 2 | 1 | 12 | **build 3**, prerequisite of async copies |
| 12 | programmatic dependent launch, sm_90+ (arch, prior) | 1 | 2 | 2 | 1 | 4 | **build 3**, H100 + 5090 arm only |
| 13 | static device arena from death points; no host slot for device-only objects (shapes S10, S53 open #2) | 2 | 2 | 3 | 1 | 12 | **build 6** |
| 14 | per-arch selection of every new rung; Ampere open rows (arch §3–4, baseline Open) | 3 | 3 | 2 | 1 | 18 | **build 8** (required) |
| 15 | prologue with captures in `g2s` (shapes S3, fusion c) | 0 | 2 | 3 | 2 | 0 | reject |
| 16 | `MoveSite ∘ TileRead` layout, free transposes, wgmma K-major (shapes S9) | 0 | 2 | 2 | 1 | 0 | reject |
| 17 | CUDA graph per loop body (shapes S11, arch) | 1 | 2 | 2 | 1 | 4 | reject, revisit after item 5 |
| 18 | precision placement (shapes S12) | — | — | — | — | — | blocked, ADR-0039 |
| 19 | whole float sum on the device, canonical tree (shapes rank 10, amendment 10) | — | — | — | — | — | blocked, ADR-0028 D4 |
| 20 | chunked d2h overlapped by panel order (prior O3) | 1 | 2 | 3 | 3 | 18 | reject for S54 (§5) |
| 21 | `cp.async.bulk` / TMA, cluster multicast, warp specialization on H100 (arch §5) | 3 | 1 | 2 | 1 | 6 | reject for S54: kernel work, its own session |
| 22 | `.L2::128B` prefetch qualifier (arch §5) | 1 | 3 | 1 | 1 | 3 | reject |
| 23 | `mma.m16n8k4`, `ld.global.nc` on the tile kernel, `tcgen05`, clusters on sm_120 (arch §5) | 0 | — | — | — | 0 | reject (the arch note's own verdict) |
| 24 | grouped / merged-grid launch of independent same-geometry sites (prior: grouped GEMM, combo kernels) | 1 | 2 | 3 | 1 | 6 | reject; streams first, merge when streams measure ~0 under-filled |
| 25 | FlashAttention derived (S53 creative #11) | — | — | — | — | — | blocked by 19 |
| 26 | persistent megakernel / MPK-style task graph (prior) | 1 | 0 | 2 | 1 | 0 | reject |
| 27 | library legs: cuBLASLt epilogues, EVT, streams, eager, graphs (prior §3) | — | — | — | — | — | done in base (`chain_bench.cu`); extended in §3 |

Items 1–3 need no decision. Item 4 is S53's queue #1. Items 5–8 are the measured-gap
items past the three required ones.

## 3. The test matrix (every item)

- **Differential.** `px_<probe>_128` (`px_loop_256`), both faces, every arm, interpreter oracle:
  N/N on every part. Trap-before-site 10/10.
- **Static.** `python3 benches/gpu/fusion_report.py <profile> <face> <n>` before any box: the
  launch / gen / d2h / h2d / host-loop counts per probe, committed under
  `benches/results-s54-gpu/fusion-report-<item>.txt`. The expected rows are in ARCHITECTURE §5.
- **Box.** `PROBE_ARMS="on off-<rung>"` through `box_run.sh`: each probe at 2048 and 4096, both
  faces, ON and OFF interleaved, five reps, one untraced process (wall, `iter ms`) and one trace;
  then `chain_bench` on the same values. Streams add `pt_*_1024` and `pt_*_512` (under one wave on
  every part). `python3 benches/gpu/rungs.py` decides; the rule is the S54 one (ships ON in a
  (fill, face) class only when some size is > 1% faster ON in every session, cuBLAS-normalized).
- **Library.** Kernels: cuBLAS and CUTLASS (`tile_bench`). Probes: eager and the best fused leg
  (`lt`, `lt2`, `cutlass`, `streams`, `gemv`, `fused`). Report best-library / Mapal iter and
  / Mapal kernel, every probe, every part, every loss.
- **Five parts, one session each per item** (`vast_part.sh`): rtx-3090:sm_86, a100:sm_80,
  rtx-4090:sm_89, h100-pcie:sm_90a, rtx-5090:sm_120; a confirm box for any row that rests on
  < 10%. Every number appended to `benches/results-s54-gpu/<item>-<part>.txt` as it lands.
- **Probe additions** (item 8 owns them; items may add their own): `pt_ew_*` already reads the
  corners, so R2 places it on the device; `px_*` keep the checksum fold (the whole-d2h path stays
  tested); `rowsum` gains `softmax_rows` (max, exp, per-row sum); streams gain `indep` and
  `mlp2` at 512/1024.

## 4. The build list

Each item is one branch `s54/<key>` off `s54-base`, one rung (`GpuRungs` field + `--gpu-off`
name + `bundle.sh` arm), OFF = today's text byte for byte. Items 1–3 share one seam: **the first
two commits of item 1** — `cell_plan` in `mapal-ir` (pure fact, its own tests, no backend edit)
and `mapal_gpu_launchv(id, stream, nptr, ptrs[], gx, gy, block, smem)` in `mapal_gpu.c` +
`gpu::RESIDENT_DECLS` (a superset of `mapal_gpu_launch`, which stays) — are cherry-picked by
items 2 and 3 before their own work. `resident_plan` is touched by items 1, 2, 5 and 6: each adds
**one new field** to `ResidentPlan` and **one new function** called from one new line in
`resident_plan`, so the integrator merges added lines, never edited ones.

### 4.1 `epilogue` — per-cell consumers in the tile site's store (R1)

**Shape.** `ElemSrc::Apply{captures: 0, inner: Load{c}}` over a device site's output `c`
(`epi`, `mlp`'s `h`, `loop`'s relu), and a `CellSite` whose read of `c` is `Same` and whose other
reads are `Row/Col/Const/Same` over `Gmem`-complete arrays (`bias`, `residual`, `indep`'s and
`diamond`'s join, `mlp2`'s bias + relu). Proof: `cell_plan` + `gen_law`'s leaf gate; the operand's
completeness is topo order (one stream) or an event (item 3).

**Design.**
- `mapal-ir/src/algo.rs`: `pub fn cell_plan(&self, f) -> SecondaryMap<MorphismId, CellSite>`,
  `CellSite{extent, reads: Vec<CellRead{slot, foot: Footprint}>, body}`,
  `Footprint::{Same, Row{c}, Col{c}, Const(u64), Other}`; reuses `tile_split`'s affine walk on
  the body's `Index` morphisms over the map index. Gate = `gen_law`'s plus every `Index` proven.
  Tests: one per footprint class, a trap-capable body refused, a nested `Fold` refused.
- `resident.rs`: `ResidentPlan.groups: SecondaryMap<MorphismId, Epi>` with
  `Epi{site: MorphismId, chain: Vec<(FuncId, Vec<(slot, Footprint, ObjectId)>)>, store_site_out: bool}`;
  `fn epilogues(&self, sites) -> groups` walks each site's output readers: a per-cell reader
  whose `Same` operand is the output and whose other operands are device-complete joins the
  group; chains join by composing (`y` over `x` over `c`); the site's own output is stored only
  if a reader outside the group exists. `slot_use` answers `Use::Dev` for a group member's
  reads, so `uses` no longer says `Host` and `keep` holds. Group members are in `skip` for the
  host walk; their outputs are device objects with their own `dslot`.
- `gpu.rs`: `Kernel` gets `epi: Option<&Epi>`; `header` takes `n` extra
  `ptr addrspace(1)` params and one extra output pointer per stored member; the four store sites
  (`naive` `store:`, `tiled` `done:`, `mma`, `wgmma`) replace `store acc` with, per chain
  member, `%v = call T @fn_cell(T %acc, ptr %cap_k ..., i32 %t)` then the stores that exist.
  `@fn_cell` is the host body (`drive.rs`) emitted with the `Same` slot typed as the element
  and `emit_index` on that slot returning the value (`ops.rs`, one branch). `kernel_id` keys on
  the group's body names and footprints; `device_module` appends the bodies as `devgen` does
  (same `@mapal_` assertion). `emit_site_resident` calls `launchv` with the extra pointers.
- v0 first (ponytail): store `acc`, call the unmodified host body through an `addrspacecast`
  pointer, the same thread reloads its own store (program order, L1). Saves the d2h, the host
  loop and a launch; keeps `c`'s store traffic. v1 removes the store when `c` has no outside
  reader. Both are the same rung; v1 is measured against v0 once.
- Exact face: the body text carries no `contract`; `-nvptx-fma-level=0` stays; a unit test
  greps the device unit for `contract` and the exact-face PTX for `fma.rn` outside the site's
  own step.

**OFF arm.** `epilogue`: no group is formed; today's placement.

**Tests.** Unit: `cell_plan` footprints; `epilogues` on `epi`/`bias`/`indep`/`mlp2` graphs
(group membership, stored members). Static: epi/bias/residual 1 / 8 B / 0; indep 2 / 8 B / 0;
mlp2 4 / 16 B / 0; fanout 2 / whole once (the host reads `c`); loop 8 / 0 / 0. Box: §3 on all
five parts; vs `lt` (cuBLASLt `EPILOGUE_BIAS`, `RELU_BIAS`, beta·C), `cutlass`, eager.

**Expected (contract face, iter → kernel ceiling from §1).** epi 4096: 3090 0.29 → ≈1.3×,
A100 0.22 → ≈0.75×, 4090 0.19 → ≈1.0×, H100 0.24 → ≈1.0×, 5090 0.12 → ≈0.95×. loop 4096 from
0.05–0.13 to ≈ the kernel ratio (1.0–1.2×, A100 0.68×). Register pressure on the 128-register
`wide-fma` tile with a capture load is the unknown: the OFF arm is the control, per part.

### 4.2 `devmap` — per-cell maps and chains as device sites by `Trm` crossings (R2)

**Shape.** A per-cell morphism (`cell_plan`, or `gen_law` for the closed-form case) whose
operands are all in `Gmem` (device output, carry, or closed-form) and whose readers are device
sites, element reads, or later groups. `ew` (closed-form inputs, corner reads), `mlp` without
item 1 (`h` as its own kernel), a per-cell map over a carried array before a site, a
`Row/Col`-footprint reader that R1 cannot take because its producer is not a site.

**Design.**
- `resident.rs`: `ResidentPlan.cells: SecondaryMap<MorphismId, Cell>` with
  `Cell{site: CellSite, id: u64, reads: Vec<(ObjectId, Footprint)>}`; `fn cell_sites(&self)`
  applies R2 in topo order after `device_sites()`: operands' location from the plan so far
  (`upload`/`keep`/`devgen`/carry views), readers' location from `uses` with cells counted as
  `Use::Dev`. The byte clause: a whole-host reader forces a `d2h` either way, so the cell runs
  on the device iff its operands are already there. `slot_use` returns `Dev` for a cell's read.
  `resident_instead` emits `launchv(cell.id, ...)` instead of the host loop (the dispatch point
  `ops.rs:emit_morphism` already exists; `bulk.rs:emit_map` is not touched).
- `gpu.rs`: `Kernel::cell` = `gen_kernel_text` generalized to `n` capture pointers and the
  `Same` / `Row{c}` / `Col{c}` / `Const` address forms (`t`, `t / c`, `t % c`, `k`); block
  `NAIVE_THREADS`; grid from `extent`. `kernel_id` keys on body name, footprints, extent.
- Chains: `elem_plan` already composes `Apply ∘ Apply ∘ Pair` to depth 16, so `z` over `y`
  over `x` is one body call per cell with one load of each leaf (`ew` is one kernel).
- Prologue is not built: when the operand is a device output, R1 (item 1) or R2 computes it
  once per cell; a per-staged-element prologue would compute it `c / bn` times.

**OFF arm.** `devmap`: no cell site; `epilogue` (item 1) still forms groups over site outputs.

**Tests.** Unit: `cell_sites` on `ew` (device: closed-form inputs, corner readers) and on an
`ew` variant whose output is read whole (host: the negative control, byte-identical text). Static:
ew 1 gen / 8 B / 0; mlp 2 + 1 cell / 8 B / 0 with item 1 OFF, 2 / 8 B / 0 with it ON. Box: §3,
vs `fused` (one kernel) and eager; `ew` reported as GB/s against the part's peak.

**Expected.** ew 4096 from 16–36 ms to the fused kernel's 0.13–0.45 ms ± one launch: from
0.005–0.014× to ≈0.5–1× of the library's fused kernel (one body call per cell against a
hand-fused kernel; the gap, if any, is the body's register use). mlp with item 1 OFF: iter →
kernel + one cell kernel (≈ +0.2–0.5 ms at 4096), the A/B that prices R1 against R2.

### 4.3 `streams` — concurrent independent sites, pinned copies, PDL on sm_90+ (R7)

**Shape.** `PathPlan.tasks` with no dependency path among device sites (`indep`, `diamond`,
`fanout`, `mlp2`); the fill of the union grid from `launch`'s geometry
(`blocks < sms · max_blocks_per_sm`). `LastUsePlan` says which host arrays cross the bus
(`upload`, the final `d2h`), so they are pinned once.

**Design.**
- `mapal_gpu.c`: `MAPAL_GPU_STREAMS` non-blocking streams created in `init_all`; `launchv`
  takes a stream index; `mapal_gpu_event(record_stream, wait_stream)` =
  `cuEventRecord` + `cuStreamWaitEvent`; `d2h` becomes `cuMemcpyDtoHAsync` on the producer's
  stream + `cuStreamSynchronize` (pageable memory would serialize: hence `pinned`);
  `mapal_gpu_pin(ptr, bytes)` = `cuMemHostRegister` once per host array that crosses
  (`resident_after` upload sites and the final `d2h` targets), unregistered at exit; the trace
  prints `op=event`, `op=pin`. `mapal_gpu_sync` stays `cuCtxSynchronize`.
- `resident.rs`: `ResidentPlan.chain: SecondaryMap<MorphismId, u32>` from `path_plan`'s task of
  each device site (a site inside a loop or guard stays on stream 0); `fn chains(&self)`.
  Opened only when the union grid is under one wave (derived, not a flag); otherwise every site
  is on stream 0 and the text is the OFF arm's. Cross-chain edges (a site reading another
  chain's output, an epilogue's fan-in operand) emit `mapal_gpu_event` before the consumer.
- `pdl` (sm_90+ only, `Gpu::sm() >= 90`): the dependent kernel of a chain (`mlp`'s second
  site) is launched with `cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`,
  and its kernel text has `griddepcontrol.wait` before the first global load of the dependent
  operand (its invariant operand's panels may stage before). `lib.rs:455`'s dropping of
  `path_plan` on GPU profiles is lifted for device sites only.

**OFF arms.** `streams` (one stream, no events, sync `d2h`), `pinned` (pageable copies), `pdl`.

**Tests.** Unit: `chains` on `indep`/`diamond`/`mlp2` (two chains, the join edge), a filled grid
gives one chain. Static: event count per probe. Box: `indep`, `diamond`, `mlp2`, `fanout` at
512 / 1024 / 2048 / 4096, both faces, vs the library `streams` leg and `lt2`; `pinned` read on
the `h2d`/`d2h` ms of every probe; `pdl` on `mlp`/`mlp2` on H100 and 5090 only.

**Expected.** At ≥ 2048 one grid fills every part: ~0 (the honest expectation; the row is
written either way). At 512 / 1024 the library's `streams` leg beats its eager by 2–12% on the
same probes, the bound for Mapal's. `pinned`: the remaining copies (`fanout`'s whole `c`, the
loop's initial upload, `px` checksum paths) at ~2× the pageable rate (S52 trace: 3–5 ms per
64 MB pageable). `pdl`: µs per dependent launch, likely within noise; measured because sm_90+
is where the arch report put it.

### 4.4 `rowfold` — row-local folds as device sites, and the reader-chosen store order (R0 for folds)

**Shape.** A `Map` over `rows` whose body is one `Fold` over `k` with a single affine read
`base + ci·i + ck·k` (`ci = n, ck = 1`) and a float `acc ⊕ e` step — `tile_site` minus the
product (`algo.rs:1140` requires `a·b`). ADR-0028 D4 holds: each row is one in-order chain on one
thread. `rowsum`, softmax denominators, norms. The second fact: when the fold is the only reader
of the site's output `c`, the order `c` is read in is known (`k` fastest per row), so the site may
store `c` in that order.

**Design.**
- `mapal-ir`: `RowFoldSite{rows, k, a: TileRead, op, seed, elem}` and `pub fn row_fold(&self, f)`
  (a separate recognizer; `TileSite{b: None}` would touch every CPU rung).
- `gpu.rs`: `Kernel::rowfold` (~60 lines: `naive` without `lane`, one thread per row, `k` in
  order). `rowfold-layout`: when `row_fold` is `c`'s only reader (`uses`), the producing site's
  store writes `o[j·rows + i]` (column-major) and the fold reads `c[i + k·rows]` — consecutive
  threads read consecutive words. Derived from the use count, not a flag.
- `resident.rs`: `DevSite::RowFold` in `device_sites`; `tile.rs`' dispatch gains one arm.

**OFF arms.** `rowfold` (host fold behind the whole `d2h`), `rowfold-layout` (row-major store,
strided fold reads).

**Tests.** Unit: `row_fold` on `rowsum`, a product body refused, a two-read body refused.
Differential: `px_rowsum_128`, `px_softmax_rows_128` both faces (per-row in-order chains are
bit-identical anywhere). Static: rowsum 2 / 4 B / 0. Box: vs `gemv` (sgemm + sgemv), eager
`torch.sum(dim=1)`-equivalent kernel, labelled "library order ≠ oracle order".

**Expected.** rowsum 4096 from 0.08–0.18× to the site's kernel ratio (1.0–1.3× contract; A100
0.76×) minus the fold kernel (a 64 MB read: 0.1–0.3 ms, bandwidth-bound once `rowfold-layout`
coalesces it; 3–10× slower strided). Per part the layout arm is the measurement: the 6 MB-L2 3090
should show the largest layout effect.

### 4.5 `retain-resident` — retention on the resident buffer, delta upload (R5)

**Shape.** Inside a loop: `write_delta → Delta::Rows{stride, start}` of the carried `x`
(`decode`: the `d` host `Update`s at `t·d + j`); the row-local sites `K = x·Wk`, `V = x·Wv`
(`retain.rs`'s three conditions); `s = q·Kᵀ` and `o = s·V` are `Panel` readers (R3) of the
retained outputs. The retained output's `Δ` is the same `Δ` (the rule in `retain.rs`), and it
propagates through `Same` consumers (`write_delta` gets one match arm).

**Design.**
- `retain.rs` + `resident.rs`: `ResidentPlan.retained: SecondaryMap<ObjectId, Retained>`; the
  retained output `y`'s `dslot` is the resident buffer — allocated once at loop entry
  (`resident_loop_enter`), never freed inside the loop, the whole site into it on the first
  pass, `emit_tiled_map_gpu_row` into it at offset `start·c` on later passes (the kernel exists:
  `Retained.row`). Today's exclusivity (`resident` or `retain-delta` takes the function) is
  removed: `retained_sites` runs under `resident` and marks its outputs `keep`.
- **Delta `h2d`** (new rung `delta-h2d`): a carried array the host writes with `Δ::Rows` and a
  device site reads is uploaded as `[start·stride, (start+1)·stride)` bytes
  (`mapal_gpu_h2d` at an offset: one new argument, the whole-array call is offset 0) instead of
  whole. The carry (F3) keeps the device copy; the host's row write is the only thing that
  crosses. The first iteration uploads whole.
- The fourth site of `decode`'s step (3 of 4 launch today; the trace shows ~1 MB `d2h` per
  step) is placed by R0: a `rows = 1` site is a device site; the builder reads the trace first.

**OFF arms.** `retain-resident` (today's `resident`-only decode), `delta-h2d` (whole upload).

**Tests.** Unit: the retained output in `keep`; `write_delta` through a `Same` map;
`loop_rewrite` (column writes) retains nothing. Differential: `decode_16_8`, `loop_rewrite_16_8`,
`px_loop_256`, both faces. Box: `decode_1024_256` both faces, ON / `off-retain-resident` /
`off-delta-h2d` / `off-retain-delta+resident` (the S53 arm), vs the hand-written cuBLAS KV cache
(device time) and an eager PyTorch-style loop with a hand-written cache (`chain_bench` gains the
leg).

**Expected.** Per step O(d²) instead of O(T·d²) for K/V: kernel 76 → ≈5–10 ms (launch-bound,
3–4 launches × 1024 × 5–10 µs); `h2d` 50 → ≈5 ms (1 KB × 1024, latency-bound); `d2h` 100 →
≈10 ms. 4090 wall 0.83 s → ≈0.2–0.3 s with the context (139 ms) in it; the KV cache's 16 ms
device time is the floor the launch count prevents reaching (item 17's graph would be next).
On the 3090 / A100 the base measured `retain-delta` ON at 2.7 / 2.2 s with 1.2 / 0.7 s of
alloc+free: the resident buffer removes exactly that.

### 4.6 `arena` — one device arena from death points; no host slot for device-only objects (R6)

**Shape.** Static sizes + `LastUsePlan` positions (`death`, `escapes`) → interval coloring with
compile-time offsets (`crates/backends/cuda/src/arena.rs:arena_plan` is the precedent). An
object in `keep ∪ devgen ∪ carry ∪ groups ∪ cells` with no host reader has no host `%Frame`
field (`frame.rs:mark_elided_arrays` is the precedent for arrays without a slot).

**Design.**
- `resident.rs`: `ResidentPlan.arena: Option<Arena{bytes, offset: SecondaryMap<ObjectId, u64>}>`
  from `fn arena(&self)` over the straight-line objects with a death; loop-scoped and
  death-less objects keep their own `cuMemAlloc` slot (today's path). `dalloc` of an arena object
  is `base + offset`, no runtime call; `free_after` skips arena objects.
- `mapal_gpu.c`: `mapal_gpu_arena(bytes) -> base` once per function entry (one `cuMemAlloc`),
  released at exit; the trace prints `op=arena`.
- `frame.rs`: device-only objects join `elided_arrays` so the host frame neither allocates nor
  pages them (S53 open #2: 64 MB each at 4096).

**OFF arm.** `arena`: per-object `cuMemAlloc` / `cuMemFree` and host slots as today.

**Tests.** Unit: two objects with disjoint lifetimes share an offset; overlapping ones do not;
an escaping object is not in the arena. Differential: every probe. Box: every probe's `alloc_ms`
+ `free_ms` and process wall; `e2e_4096`, `e2e_chain_4096` wall vs the cuBLAS floor
(4090: 233 vs 202 ms today).

**Expected.** 0.5–2.7 ms per site of alloc/free (S53) and the host paging of device-only
buffers: a few ms per probe at 4096, 5–10% of `e2e_4096`'s non-context wall. Honest: the context
(21–120 ms of 233–481 ms) is the larger share and is not this item.

### 4.7 `l2-raster` — grid order and L2 policy from `l2_bytes` and the reuse count (kernel, Ampere first)

**Shape.** `TileRead` strides give each operand's panel bytes; the grid gives the reuse count
(every B column-panel is read by `rows / bm` blocks, every A row-panel by `c / bn`); `l2_bytes`
(new `Gpu` field: 3090 6 MB, A100 40 MB, 4090 72 MB, H100 50 MB, 5090 96 MB, each checked
against the box's `facts` line) says whether one wave's working set fits. Evidence: the 3090
at 8192 drops to 0.84 contract / 0.79 TF32 from 0.96 / 1.00 at 4096 (B = 256 MB re-read from
HBM every wave); no other part drops.

**Design.**
- `profile.rs`: `Gpu.l2_bytes`; `fn raster(&self, g: GpuTile, rows, c, k) -> u64` = the block-row
  group height `h` such that `h` A row-panels + one wave's B column-panels fit `l2_bytes / 2`
  (half, the `tile_kc` rule at its L2 location): the largest power of two with
  `(h·bm + wave_cols·bn)·k·elem ≤ l2_bytes / 2`; 1 when the whole B fits (then the row-major
  order is kept, the OFF text).
- `gpu.rs`: the block index → `(row, col)` map at the kernel's head: `col = bid / h` within a
  group of `h` rows (the CUTLASS / cuBLAS swizzle, derived here).
- `l2-window`: `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` over the
  operand whose reuse count is highest and whose bytes fit `l2_bytes`, `hitProp = PERSISTING`;
  `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE)` once. For a chain, the window is the
  intermediate whose reuse distance (`LastUsePlan` positions between producer and consumer) is
  one launch.

**OFF arms.** `raster`, `l2-window`.

**As built (`s54/l2-raster`).** The formula above has no solution on any part at 8192 (one
128-row panel over `k` is 4 MB, so `h·bm + wave_cols·bn` panels never fit 3 MB of a 6 MB L2), and
"1 when the whole B fits" would group on the 4090 at 8192 too (B = 256 MB). Built instead, in
`Gpu::raster`: `h` = the power of two dividing `gy` that minimises the panels one wave reads,
`(h·bm + ⌈W/h⌉·bn)` (16 for a 164- or 256-block wave of 128×128 blocks), behind a gate that keeps
row-major while one block's panels over `k`, `(bm + bn)·k·elem`, fit `l2_bytes` — what the part
streams between the launch neighbours that share an `a` slab. That gate reproduces the measured
pattern (only the 3090 at 8192: 8 MB vs 6 MB) and the unit outcomes (3090: 1, 1, 16; every other
part 1). It is a model, so it is a rung: `l2-gate` OFF groups wherever the grid is more than one
wave, the arm that tests the gate on every part. `l2-window` takes the chain intermediate whose
next reader is the next launch, else the operand read by the most blocks (`a`: `gx`, `b`: `gy`),
when its bytes fit `l2_bytes`; `hitRatio` = set-aside / window, both device attributes.

**Measured** (`benches/results-s54-gpu/l2-raster-tables.md`, 21 sessions on five parts). The gate
mispredicted both ways:
- On the 3090 at 8192 grouping measured no stable sign: exact 0.986 / 1.029, FMA 0.992 / 1.017,
  TF32 0.991 / 0.956. **Raster ships OFF there.**
- On the 5090, which the gate excludes, ungated grouping at 8192 measured TF32 1.031 / 1.054 /
  1.014 in three sessions. **The gate ships OFF there on the contract faces.**
- On the A100, 4090 and H100 the ungated grouping was never faster.

`l2-window` ships ON only on H100 PCIe contract (mlp2_2048_c traced kernel 1.012 / 1.016). It costs
up to 7% on the 4090 and measured ~0 elsewhere. The remap's known bits turned the kernel's address
adds into `or`, which NVPTX does not fold into immediate offsets: +18 registers on the H100 FMA
tile. An opaque register move after the remap fixes that.

**Tests.** Unit: `raster` on the 3090 at 2048 / 4096 / 8192 (1, 1, > 1) and on the 4090 (1 at
every size). Box: `tile_bench` kernels at 1024–8192 all faces on all five parts (the arm is
one-line, every part gets a row); the chain probes at 4096 for `l2-window`.

**Expected.** 3090 at 8192: toward the 4096 ratio (+5–15% contract / TF32); every other part
~0 at ≤ 8192 (B fits). `l2-window`: ≤ 3–8% (the S53 kernel note's R9 bound) and only where the
operand fits; likely ~0 on the 72–96 MB parts.

### 4.8 `arch-select` — every new rung decided on five parts; the Ampere open rows

**Shape.** `GPU_RUNGS_OFF` rows (`RungOff{profile, rung, fill, face, cite}`) with the fill key
read from the grid the rung ends up in (ARCHITECTURE §7); `rungs.py`'s rule over two sessions.
`baseline-tables.md` §Open lists the Ampere rows that rest on one session: 3090 `warp-fill`
under-filled contract (0.94 / 0.96 and 0.88 / 0.89 OFF faster), `min-blocks` under register
staging (+9% exact, −11% TF32), `unroll` under-filled contract; A100 `idx32`, `wide-fma` filled
contract (sign changes); A100 TF32 0.57–0.69× everywhere.

**Design.**
- `rungs.py`: reads the probe rows (`probe prog=… arm=…`) as classes keyed by probe and size,
  the fill of the kernel the rung folds into; prints the `RungOff` rows to paste.
- `profile.rs`: the rows for `epilogue`, `devmap`, `streams`, `pinned`, `pdl`, `rowfold`,
  `rowfold-layout`, `retain-resident`, `delta-h2d`, `arena`, `raster`, `l2-window` per part,
  each citing `<item>-<part>.txt` and `-confirm.txt` medians; the 3090 / A100 open rows closed
  with a second session each. `Fill` separating TF32 from FMA within `Contract` is added only if a
  row needs it (the 3090 `min-blocks` sign split is the candidate).
- A100 TF32 is a kernel loss (the `mma` pipeline depth on a part whose tensor rate is 8× its
  FP32), named in the results; not fixed here.

**Tests.** `cargo test` pins: every new rung name in `GpuRungs::NAMES`, every `RungOff.rung`
resolves, `gpu_rungs_are_the_measured_table`. Box: the integrated build on all five parts
(`PROBE_ARMS` = on + every `off-<rung>`), two sessions where a row rests on < 10%.

**Expected.** Per-part tables with every loss: the exact face stays 0.5–0.66× (kernel); A100
TF32 stays 0.57–0.69×; the chain probes reach their kernel ratios where placement was the loss.

## 5. Rejected, one line each

- **Demand-restricted sites** (#7): no program in the suite reads a strict subset of a GEMM, and
  the timed probes read corners, so the rule would hollow them into two-row GEMMs; built only
  with a program that needs it and a harness that keeps a whole reader.
- **Prologue with captures** (#15): `pro` already fuses through `devgen`; no probe has a captured
  per-cell producer of a site operand; R1/R2 compute such a producer once per cell, a staged
  prologue would compute it `c / bn` times.
- **`MoveSite ∘ TileRead` layout** (#16): no transpose on the suite; the H100 wgmma K-major
  operand is kernel work (#21).
- **CUDA graph per loop** (#17): ≤ 3–5% of decode today; after item 5 the step is launch-bound
  and the bound becomes real — measure then.
- **Precision placement** (#18): ADR-0039 is a candidate, not decided.
- **Whole float sum on the device** (#19): ADR-0028 D4; the canonical-tree re-pin is Sapir's.
- **Chunked, overlapped d2h** (#20): after items 1–2 no timed program keeps a whole d2h; the
  `px` checksums are correctness runs; revisit with #19.
- **TMA / clusters / warp specialization** (#21): the H100 TF32 kernel loss is real (0.55–0.67×)
  and its own session; nothing in it is placement.
- **`.L2::128B` prefetch** (#22): ≤ 2%, below the noise floor.
- **k4 mma, `ld.global.nc` on the tile, `tcgen05`, sm_120 clusters** (#23): no effect or no
  part (the arch note's verdict).
- **Grouped / merged-grid launch** (#24): streams cover the independent case; a merged grid is
  the follow-up only where streams measure ~0 under-filled.
- **FlashAttention derived** (#25): needs #19.
- **Persistent megakernel** (#26): research, not a rung.
