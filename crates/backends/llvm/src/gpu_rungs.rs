//! [`GpuRungs`]: the GPU tile kernel's rungs as a set (one `bool` per rung).

/// The GPU tile kernel's rungs (S52), as a set: `true` = the rung is in it.
///
/// **What ships is the rung rules over the part's physical properties**
/// (hardware-properties, [`TargetProfile::gpu_rungs`], `rules.rs`): a site's
/// fill against the part's SMs, its face, and properties such as FFMA per
/// DRAM byte, registers per resident thread, L2 per SM and measured launch
/// costs — never a part's name or version. [`GpuArms`] forces a rung on or off
/// so its effect can be measured — an A/B needs both arms of the same
/// compiler, the `MovePanel::Off` precedent; an arm is a measurement flag,
/// never a per-part default. Values are identical either way: every rung
/// moves operands, never the order a cell folds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuRungs {
    /// Take the rung rules over the part's properties (`rules.rs`); off
    /// (`--gpu-off=arch-rungs`, the measurement arm) = every rung on (S52).
    pub arch: bool,
    /// Space a thread's register-tile chunks by the bank geometry.
    pub interleave: bool,
    /// Fully unroll the panel's `bk` register steps.
    pub unroll: bool,
    /// Emit `.minnctapersm` from the derived blocks per SM.
    pub min_blocks: bool,
    /// Shape each warp's threads by the shared-memory wavefront.
    pub warp: bool,
    /// Load the next panel into registers while this one is computed.
    pub prefetch: bool,
    /// On a reduced face, take the tensor-core realization (`mma.sync`).
    pub mma: bool,
    /// Stream FMA-tile panels through shared memory with `cp.async` (sm_80+).
    pub cp_async: bool,
    /// Tensor-core warp tile from the whole-thread register count (64×64).
    pub wide: bool,
    /// Load tensor-core A fragments with `ldmatrix.x4`.
    pub ldmatrix: bool,
    /// Shrink the register / warp tile until the grid gives every warp
    /// scheduler two warps (`warp-fill`, S53); OFF = blocks ≥ SMs only.
    pub fill: bool,
    /// S53 `fma-regs` (four rungs, FMA and one-thread-per-cell kernels):
    /// index arithmetic in `i32` when every index the kernel forms fits.
    pub idx32: bool,
    /// Panel copy `p` addressed as copy 0 plus a constant from the lane
    /// map's period: one base register per operand.
    pub affine: bool,
    /// Register-staged transposed-A store: the lane map and row pad under
    /// which one warp's stores hit 32 banks.
    pub pad: bool,
    /// FMA register tile from the whole-thread register count (8×16).
    pub wide_fma: bool,
    /// On `sm_90a`, the TF32 face takes Hopper's `wgmma` kernel (S53); off,
    /// the `mma.sync` kernel. Both are bounded by TF32, neither is bit-exact.
    pub wgmma: bool,
    /// Keep an object's device copy between sites (S53, plan-s53 §2.3): copy
    /// in at the producer, carry loop state on the device, copy out only what
    /// the host reads, free at death. Off: one `mapal_gpu_tile` round trip
    /// per site (S52). S62 retention probes (`pt_rt_{indep,mlp2}_{1024,2048}`,
    /// eight requests, `benches/results/gpu/s62/runtime-measurements/<part>/summary-*.md`,
    /// two sessions): off-resident/ON iter 1.43–3.08 (3090), 1.38–2.76 (4090),
    /// 1.13–3.36 (H100), 1.44–2.48 (5090), stable+ on 8 of 8 cells per part.
    pub resident: bool,
    /// Under `resident`, compute a closed-form producer that only device sites
    /// read on the device (`mapal_gen_<id>`) instead of copying it in.
    /// S62 retention probes (same files as `resident`): off-devgen/ON iter
    /// 0.86–1.10 (3090), 0.99–1.02 (4090), 0.83–1.01 (H100), 0.96–1.00
    /// (5090), no stable sign on any cell: ships ON (no stable-).
    pub devgen: bool,
    /// S53 retention at part-of-object scope: a row-local site over a carried
    /// array whose iteration writes one row ([`mapal_ir::Delta::Rows`])
    /// recomputes only that row into its retained output (`func/retain.rs`).
    /// Off on every part (no measured part found a gain, `rules.rs`): a
    /// function it fires in keeps S52 round trips, and `resident` alone ran decode 1.3–1.8× faster
    /// (`integrated-*.txt`). Since S54 it is the S52-round-trip arm only:
    /// `retain-resident` plans the two together.
    pub retain_delta: bool,
    /// Host runtime (S53): start the CUDA context and module load on a thread
    /// as `main`'s first call, so it overlaps the host work before the first
    /// site. Off: the first site creates them.
    pub eager_init: bool,
    /// Under `resident` (S54 R6): every straight-line device object with a
    /// death lives at a compile-time offset of one per-function allocation
    /// laid out by interval coloring over `LastUsePlan` positions, and a
    /// device-only object gets no host slot. Off: one `cuMemAlloc` /
    /// `cuMemFree` per object and a host slot for every array (S53).
    /// On on every part (measured positive on all five rented parts):
    /// `benches/results/gpu/memory/arenas/arena-<part>.txt` + `-confirm.txt`, iter ms
    /// stable+ on 38–44 of 50 rows per part and stable- on none; e2e_chain_4096
    /// wall OFF/ON 1.17/1.09 (3090), 1.03/1.06 (A100), 1.16/1.23 (4090),
    /// 1.04/1.06 (H100 PCIe), 1.06/1.30 (5090), exact face, ms in the files.
    /// S62 (`benches/results/gpu/s62/runtime-measurements/<part>/summary-*.md`, two
    /// sessions, 22 probe cells per part): off-arena/ON iter 1.56–20.72 (3090),
    /// 1.80–5.21 (4090), 1.27–10.35 (H100), 1.46–8.34 (5090), stable+ on all.
    pub arena: bool,
    /// S54: walk the tile grid in block-row groups (`TargetProfile` →
    /// [`profile::Gpu::raster`]): a wave reads fewer distinct panels. Off:
    /// the hardware's row-major order. Values are the same either way — a
    /// grid order moves no cell's fold.
    pub raster: bool,
    /// Under `raster`, keep row-major order while one block's panels fit the
    /// L2 (the derivation's gate). Off: group wherever the grid is more than
    /// one wave — the arm that tests the gate on every part.
    pub l2_gate: bool,
    /// S54: launch a resident site with an L2 access-policy window
    /// (`hitProp` persisting) over the object the graph proves read again —
    /// the output the next launch reads, else the operand read by the most
    /// blocks — when its bytes fit the L2 (`mapal_gpu_launch_l2`).
    pub l2_window: bool,
    /// S54 R5 under `resident`: a retained site (`func/retain.rs`'s rule)
    /// keeps its output in its resident buffer — the whole site into it on a
    /// loop entry's first pass, the written row into it after — and the
    /// function stays resident. Off: a function with a retained site takes the
    /// S52 round trips (`retain-delta` on) or retains nothing. Measured on
    /// decode_1024_256 (`retain-resident-*.txt`, two boxes per part): ON where
    /// the whole K/V site costs more than its row (3090, A100; 4090 and 5090
    /// since S65's by-ref removed the loop residue; H100 since cpu-gpu-boundary,
    /// on causal / prefill).
    pub retain_resident: bool,
    /// S54 R5's transmission: a host-written carried array with
    /// [`mapal_ir::Delta::Rows`] that a device site reads is uploaded as the
    /// rows the iteration wrote, after one whole upload per loop entry. Off:
    /// the whole array every iteration. Decode loop 1.10–1.40× faster on every
    /// part (1 MB → 1 KB per step).
    pub delta_h2d: bool,
    /// S54 R0 for one-row sites (and, since generic-fusion, 2-D ones: attention's
    /// `s = q·Kᵀ`): a site whose lane operand is read at a lane
    /// stride other than 1 (`q·Kᵀ`, [`mapal_ir::CategoryIr::tile_plan_strided`])
    /// is a device site. Off: it stays a host map behind a whole copy-out.
    /// Decode loop 1.30–1.58× faster on every part (no 1 MB copy-out of K, no
    /// host fold over it). Attention (fusion-natural-forms sweep): off-lane-stride/ON
    /// iter 106–693 on all four parts (the host `q·Kᵀ` of the OFF arm).
    pub lane_stride: bool,
    /// S54 R0 for folds (`func/rowfold.rs`): a row-local in-order fold whose
    /// read array a device site produced runs on the device, one thread per
    /// row. Off: the host fold behind the output's whole copy-out. Ships ON on
    /// every measured part: `rowfold-<part>.txt` rowsum 2.3–9.9× and
    /// softmax_rows 1.0–1.3× the OFF arm's own clock, two sessions each.
    pub rowfold: bool,
    /// Under `rowfold`, when the fold is the producing site's only reader, the
    /// site stores its output in the order the fold reads it (column-major),
    /// so a warp's reads coalesce. Off: row-major store, strided fold reads.
    /// Measured mostly no gain: the strided fold costs 0.01–0.1 ms at 4096
    /// (L1 serves each sector to the next steps) and the column-major store
    /// costs the FMA tile 0–6% (GDDR parts) to 7–22% (H100). Off on every
    /// part: its two 1–2 % wins (the 4090's contract face, the A100's exact
    /// face) are not separated from its losses by any property the rules read
    /// (open, `rung_replay.tsv`).
    pub rowfold_layout: bool,
    /// S54 R1, under `resident`: a per-cell morphism whose cell `t` reads a
    /// site's output at `t` (and other arrays complete on the device at a
    /// cell-only footprint) is evaluated in that site's store — no launch, no
    /// copy-out, and no store of the site's output when nothing else reads it
    /// (`func/resident.rs:epilogues`). Off: the morphism runs on the host
    /// behind the site's copy-out (S53).
    ///
    /// On on every part (measured positive on every rented part). On
    /// the nine probes it touches (epi, bias, residual, mlp, mlp2, indep,
    /// diamond, fanout, loop) at n = 2048 and 4096, both faces, the OFF arm's
    /// median clock was 1.55–3.24× ON's on the RTX 3090, 1.20–4.72× on the
    /// A100, 1.73–4.85× on the RTX 4090, 1.69–7.00× on the RTX 5090 and on
    /// the H100 PCIe 1.44–3.84× and 1.10–20.44× on two boxes. One H100 row
    /// read 0.26 because of host stalls in three of its five ON reps; its
    /// minima give 2.38. Every ratio of minima is at least 1.29, and the
    /// printed values are identical (`benches/results/gpu/fusion/epilogues/epilogue-*.txt`,
    /// `epilogue-tables.md`). Only the filled class (n ≥ 2048) was measured.
    pub epilogue: bool,
    /// S54 `devmap` (ARCHITECTURE R2): a per-cell map whose operands are
    /// already in device memory runs as its own one-thread-per-cell kernel
    /// instead of a host loop behind a whole copy-out. Off: every per-cell map
    /// on the host (the S53 placement).
    ///
    /// **On on every part** (measured positive on every rented part): arm
    /// `off-devmap` iter / ON iter, medians of 5, every multi-computation probe
    /// (bias, diamond, epi, fanout, indep, loop, mlp, residual) at 2048 and 4096
    /// on both faces, `benches/results/gpu/fusion/device-maps/devmap-<part>.txt`: RTX 3090
    /// 1.76–3.51, A100 1.30–2.53, RTX 4090 1.89–4.03, H100 PCIe 1.32–15.5, RTX
    /// 5090 1.76–4.76; `ew` 7–121 (the host loops it replaces).
    pub devmap: bool,
    /// S54 `cell-law`, under `devmap`: a captured array whose producer is
    /// closed-form in the index (`devgen`'s law) and which a cell reads only at
    /// its own index is evaluated in the cell, never stored or loaded. Off: the
    /// cell loads it from its `devgen` buffer.
    ///
    /// **On on every part** (measured positive on every rented part): `ew`, arm
    /// `off-cell-law` iter / ON iter at 4096 exact / contract, `devmap-<part>.txt`:
    /// RTX 3090 6.15 / 7.00, A100 1.27 / 1.19 (`-confirm.txt`, another box,
    /// 2.12 / 2.20), RTX 4090 8.70 / 9.32, H100 PCIe 1.11 / 3.16 (`-confirm.txt`
    /// 1.31 / 1.42), RTX 5090 1.99 / 1.99. On `residual` (the operand is also a
    /// tile site's, so it is built either way) no effect: 0.98–1.02 wherever the
    /// ranges are not swamped by the box's outliers.
    pub cell_law: bool,
    /// S54 R7: device sites with no `PathPlan` path between them launch on
    /// separate streams (one per chain), an event on every cross-chain edge,
    /// when every such site's output stays on the device. Off: one stream.
    pub streams: bool,
    /// Open the streams only where two independent grids' last waves fit one
    /// wave (the wave rule, `func/resident.rs:waves_merge`); off = whenever
    /// two chains exist. A measurement arm for the rule itself.
    pub stream_gate: bool,
    /// Page-lock (`cuMemHostRegister`) a host array that crosses the bus
    /// repeatedly (a copy inside a loop), once, before its first copy, where
    /// the host's measured pageable copy rate is below 0.9 of its page-locked
    /// one (`DeviceCost::pageable_pm`, `rules.rs`). Off: pageable copies.
    pub pinned: bool,
    /// Where the part has `griddepcontrol` and its measured programmatic
    /// launch is cheaper than a plain dependent one (`rules.rs`): a site
    /// launched right behind another kernel on its stream is a programmatic
    /// dependent launch (`griddepcontrol.wait` at kernel entry) — in a module
    /// whose launches repeat in a loop ([`rules::pdl_pays`]).
    pub pdl: bool,
    /// cross-arch: under `pdl`, a tile kernel triggers its dependent's launch
    /// at its entry (`griddepcontrol.launch_dependents`), so the dependent's
    /// blocks wait resident beside it. Off: the trigger is the kernel's exit
    /// (the launch still overlaps the drain). Off on every part (its ON arm
    /// measures it).
    pub pdl_early: bool,
    /// S62 `row-fit`: a thread grid taller than the site's rows (`ty·tm >
    /// rows`, `TileSite::rows`) trades rows of threads for columns until the
    /// block fits, keeping the thread count. Off: such a site takes the
    /// one-thread-per-cell kernel.
    pub row_fit: bool,
    /// S62 `wave-gate`, under `row-fit`: a row-fitted grid with fewer blocks
    /// than SMs ([`profile::Gpu::sms`]) takes the one-thread-per-cell kernel
    /// instead. Off: the row-fitted grid at any block count.
    pub wave_gate: bool,
    /// S62 `grid-share`: an SM holds at most `⌈blocks / sms⌉` of this launch's
    /// blocks, so the panel depth is sized against that share of its shared
    /// memory (and `minctasm` says so). Off: the resource-derived residency.
    pub grid_share: bool,
    /// S62 `panel-regs`: the register-staged prefetch panel takes what the
    /// derived residency leaves a thread after its tile
    /// ([`profile::fma_held_regs`]), at least one fragment pair. Off: one
    /// fragment pair (`tm + tn`, S52).
    pub panel_regs: bool,
    /// S62 `fill-rows`, under `warp-fill`: the fill rule shrinks the register
    /// tile's rows down to one (scalar `a` fragments); off, down to one vector.
    pub fill_rows: bool,
    /// S62: a shaped site no register tile divides runs one output per thread
    /// as a staged tile (square-most `bm × bn` block of cells,
    /// [`profile::TargetProfile::gpu_cell_tile`]). Off: one thread per cell,
    /// unstaged (`naive`). On on every part (measured positive on every
    /// rented part): `benches/results/gpu/s62/asyncstage/`, OFF/ON
    /// cuBLAS-normalized, two passes and two boxes per part, every M16 cell
    /// stable+ on both faces with L2 flushed: RTX 5090 1.90–3.25, RTX 4090
    /// 1.70–3.58, H100 1.61–5.69, RTX 3090 1.32–3.93; 1.01–6.09 with L2 warm
    /// (5090 16×1024×4096 the one cell inside its null).
    pub cell_fill: bool,
    /// S62: when the grid leaves an SM fewer blocks than the tile's smem was
    /// budgeted for, stream its panels with `cp.async` through as many stages
    /// as the share left holds ([`profile::TargetProfile::gpu_tile_stages`]).
    /// Off on every part (`rules.rs`): it hides DRAM latency — up to 4.0× with
    /// L2 flushed — but loses with operands in L2 and in the composed residual
    /// MLP, and the launch does not yet read whether its operands are in L2
    /// (open: `residence_plan` at the launch).
    pub async_stage: bool,
    /// S62 (ADR-0040 D3/D4), on the `fixed` and `free` orders: a resident site
    /// whose output domain cannot fill the part folds its `k` axis as
    /// [`TargetProfile::split_count`] contiguous partial folds into a
    /// workspace, then one deterministic reduce launch adds them in partition
    /// order. Off: the ordered kernel. Never fires on a sequential order.
    pub splitk: bool,
    /// S63 (ADR-0040 D4), on the `free` order only: the partial folds of
    /// `splitk` `atomicrmw fadd` into the zeroed output — no workspace, no
    /// reduce launch, combine order unspecified. Off: the next realization
    /// (`splitk`'s workspace, else the ordered kernel).
    pub splitk_atomic: bool,
    /// S63 (ADR-0040 D4), on the `free` order only: one wave of blocks walks
    /// the `tiles × panels` iteration space in equal shares; a tile cut
    /// between blocks combines by `atomicrmw fadd` into the zeroed output.
    /// Off: the next realization (`splitk-atomic`, `splitk`, ordered).
    pub streamk: bool,
    /// S62: a GPU build emits everything outside a device site against the
    /// host's CPU profile ([`EmitOpts::host`], [`profile::with_host`]), and a
    /// build whose every site stays on the host is that host's CPU build
    /// (parallel entry, tiling, packing). Off: the host takes [`GENERIC`]'s
    /// facts and runs sequentially — the S52–S54 GPU build.
    ///
    /// [`GENERIC`]: crate::profile::GENERIC
    pub hostopt: bool,
    /// S62, under `hostopt`: a capable site runs on the device only where the
    /// part's measured costs say it finishes sooner there
    /// ([`TargetProfile::device_pays`]), or where an operand is already
    /// device-produced (`func/gpu.rs:placement`). Off: every capable site.
    pub place: bool,
    /// S65 `by-ref` (rule R, `func/frame.rs`): forwarding is not observing —
    /// `Pair`, `Proj`, `Phi` and a loop's enter, back edge and exit move an
    /// array's address, so the bytes live where their producer writes them;
    /// an `Update` writes them in place when no view of them is read again
    /// (`LastUsePlan::view_dead_after`), and a loop's next state takes one
    /// buffer or two by whether the state it replaces is still read when it
    /// is written. A device-carried state has no host bytes at all. Any
    /// profile, CPU builds included. Off: products hold arrays inline (S64:
    /// five whole copies of the state per trip).
    pub by_ref: bool,
    /// `flow` (`func/flow.rs`): placed folds and cells whose
    /// read footprints stay inside one row of a group's unit (or a halo of a
    /// recomputable cell) run as one unit kernel; a member no outside reader
    /// needs is never stored. Off: the tile root only (`epilogue`, S54 R1).
    /// Values, launches and movement are proven by `tests/generic_fusion.rs`.
    /// The fusion-natural-forms sweep measured its first unit (the row softmax,
    /// one thread per row) 1.8–4.8× the four kernels it replaced; units since
    /// fold as trees (`tree-gate`).
    pub flow: bool,
    /// generic-fusion `window` (design §4): a tile site whose reads are affine
    /// in the fold's derived axes `(k÷div, k%div)` — a 2-D convolution's taps —
    /// is a device site; the one-thread-per-cell kernel walks `k` in order and
    /// addresses `cq·(k÷div) + cr·(k%div)`; and a weightless window over a
    /// device output (a box filter: a row fold with a split map index and
    /// split taps) is a device fold. Off: both stay the host's (`func/conv.rs`,
    /// the host fold). On on every part (measured positive on the four rented
    /// parts): off-window/ON iter, conv_2048_5 23.2 / 31.6 / 9.7 / 11.8 and
    /// ewconvew_2048_5 34.4 / 34.2 / 13.0 / 14.2 (4090 / 5090 / H100 / 3090,
    /// `benches/results/evidence/fusion-natural-forms/<part>/results-<part>.txt`).
    pub window: bool,
    /// generic-fusion natural forms: a map whose cells outside a set are a
    /// capture's own ([`mapal_ir::Outside`]: a row, a range, a column) runs in
    /// place over that set, under an `Update`'s legality (`in_place_source`).
    /// Any profile, CPU builds included. Off: the map runs over every cell
    /// into its own buffer. Ships ON on all four measured parts: decode_1024_256
    /// off-range-map/ON iter 3.96 / 4.39 / 2.50 / 4.15 (4090 / 5090 / H100 /
    /// 3090), the kernels' traced ms equal — the host map walks one row per trip,
    /// not 262144 cells (`fusion-natural-forms/<part>/results-<part>.txt`).
    pub range_map: bool,
    /// cpu-gpu-boundary 1b: a per-cell map with number captures
    /// (`CellSite::uniform` — a loop counter, a fold's result) is a device
    /// cell, each uniform a by-value kernel argument; one that reads no array
    /// is a closed form, built where its readers run (`devgen`'s rule). Off:
    /// such a map runs on the host and crosses the bus to its device readers.
    pub uniform: bool,
    /// cpu-gpu-boundary 1a `nest-cell`: a per-cell map whose body folds over
    /// captured arrays at any proven address (`CellSite::nested`: a guarded
    /// dot product, a fold per cell) is a device cell when it is no tile or
    /// row-fold site. Off: it runs on the host.
    pub nest_cell: bool,
    /// cpu-gpu-boundary item 5 `tree-gate`, under `flow`: a lone fold over a
    /// column-major operand (`rowfold-layout`'s sole-reader store) keeps its
    /// one-thread-a-row kernel instead of a tree unit — a row's cells are
    /// `rows` apart, so the tree's lanes would not coalesce. Off: a lone
    /// tree-able fold is a unit whatever the layout (rtx-4090 rowsum_4096,
    /// flow ON/OFF 1.056, why-slow C12).
    pub tree_gate: bool,
    /// cpu-gpu-boundary 1a `retained-fold`: a function with retained sites
    /// (S54 `retain-resident`) still places its standalone row folds on the
    /// device (prefill `pmx` / `pden`). Off: S65's veto — no row fold is a
    /// device site in such a function.
    pub retained_fold: bool,
    /// cpu-gpu-boundary `preload`: the runtime resolves every kernel of the
    /// module (`cuModuleGetFunction`, which loads it under CUDA lazy loading)
    /// when it loads the module — on the `eager-init` thread, before the
    /// clock — and launches from that table. Off: each launch resolves its
    /// kernel, so the first launch of each loads it inside the timed region.
    pub preload: bool,
    /// cpu-gpu-boundary `entry-arena`, under `arena`: the function's device arena is made at
    /// its entry from the static plan, and every buffer freed only at exit (a
    /// loop-cone buffer, a carry, a workspace) is placed in it too — each its
    /// own interval, so the existing coloring keeps it apart. Off: the arena
    /// on the first device op, loop-cone buffers by `mapal_gpu_alloc` on first
    /// use (inside the loop's first trip).
    pub entry_arena: bool,
    /// cpu-gpu-boundary `async-h2d`: every upload is enqueued without a host
    /// wait (`mapal_gpu_h2d_async`): the runtime stages the source at the call,
    /// so the host may rewrite it at once. A loop trip then waits for the
    /// device only where a host morphism reads a device value (a copy-out).
    /// Off: `cuMemcpyHtoD`, which waits for the device before each copy.
    /// Off on every part (null on the four rented parts, `rules.rs`).
    pub async_h2d: bool,
    /// cpu-gpu-boundary `stage`: the runtime keeps a ring of page-locked
    /// staging slots (made with the context, before the clock); an async
    /// upload that fits a slot is copied there and DMA'd from it, so a
    /// page-locked (`pinned`) source never waits either. Off: the driver's
    /// own staging (pageable) or a synchronous copy (a page-locked source).
    pub stage: bool,
    /// cpu-gpu-boundary `chain-h2d`, under `streams`: an upload whose every
    /// device reader is a site of one chain is copied in on that chain's
    /// stream (`mapal_gpu_h2d_on`), so the chains the `PathPlan` proves
    /// independent do not wait for it. Off: on the legacy stream, which
    /// every chain joins at its next launch.
    pub chain_h2d: bool,
    /// recurrence-precision 1b `range-cell`: a piecewise map written in place
    /// ([`mapal_ir::Outside`]: a row, a range, a column of the state) is a
    /// device cell over its interval alone — one thread per written cell,
    /// the interval's start a by-value argument — when its operands are on
    /// the device: the state stays there (decode/causal `x`). Off: the host's
    /// range map, the written row uploaded per trip (`delta-h2d`).
    pub range_cell: bool,
    /// recurrence-precision 1b `scalar-fold`: a whole fold of a device array
    /// ([`mapal_ir::CategoryIr::whole_fold`]: a row fold of one row) runs on
    /// the device into a one-element buffer, and a cell reading it as a
    /// number capture loads it there (a pointer, not a by-value argument):
    /// no host observes it (causal `mx`, `den`). Off: the array comes out
    /// and the host folds it.
    pub scalar_fold: bool,
    /// recurrence-precision 1a `trip-batch`: a loop whose first trip interval
    /// [`mapal_ir::TripPhase::skip`]s enters at its last trip, the interval's
    /// `batched` writes made first by one device cell over all their cells
    /// (each cell's trip `c ÷ d − b`). Off: every trip runs.
    pub trip_batch: bool,
    /// recurrence-precision item 2 `cell-lanes`: on the default face (order
    /// free) a nest cell whose cells are fewer than the part's resident
    /// threads folds across a warp's lanes — lane `l` takes every 32nd step
    /// from `l`, a shuffle tree combines — one warp a cell. Off: one thread a
    /// cell, the fold in order.
    pub cell_lanes: bool,
    /// recurrence-precision item 2 `fold-lanes`: on the default face a device
    /// row fold (whole folds included) whose step regroups (`+`, `×`, running
    /// max/min, f32) and whose rows leave the part's resident threads idle is a
    /// warp a row — lane `l` takes every 32nd `k` from `l`, a butterfly of
    /// shuffles combines, lane 0 stores. Off: one thread a row, `k` in order.
    /// Never over a column-major operand (`rowfold-layout`'s store): one
    /// thread a row already coalesces there — the rtx-4090 measurement that
    /// once kept it off on that part (`gpu::rowfold_lanes`).
    pub fold_lanes: bool,
    /// recurrence-precision `radix-tile`: a map/fold whose reads are affine
    /// per digit of the map and fold indices (`mapal_ir::RadixTile`, a
    /// convolution's implicit GEMM) is a device tile site — the staged FMA
    /// tile, addresses formed digit by digit. Off: it stays a `nest-cell`
    /// (one thread per output walking the fold).
    pub radix_tile: bool,
    /// recurrence-precision `tf32-tile` (ADR-0041 D6): a site whose product
    /// admits `tf32` (`precision_plan.set`) takes the tensor-core kernel
    /// (`mma.sync` sm_80+, `wgmma` sm_90a) where its shape fits. Off: the
    /// f32 FFMA tile, which `{tf32, f32}` admits as well.
    pub tf32_tile: bool,
    /// site-geometry `row-split` (D1): a retained site's one-row pass takes
    /// the fold chooser ([`Order`]-free faces: the atomic split or stream-K,
    /// the row zeroed first) instead of one block over the row. Off: the row
    /// in order on the one-thread-per-cell grid.
    pub row_split: bool,
    /// site-geometry `fill-wave` (C9, C10, C6c): a tile site whose grid
    /// holds fewer warps than [`profile::Gpu::fill_warps`] cuts its free fold
    /// into the fewest partitions that fill it, counted against the blocks
    /// the part can hold at the tile's own registers and threads (not the
    /// unsplit grid's share), and the split kernel's panels are sized for the
    /// split grid. Off: the largest split inside the unsplit grid's wave.
    pub fill_wave: bool,
    /// plan-cost (design §2): on a part with calibrated plan rates
    /// ([`profile::TargetProfile::plan_rates`]), a free-order site's fold
    /// realization and split count are the argmin of the predicted
    /// [`profile::TargetProfile::site_ns`] over the candidates the other rungs
    /// admit. Off, or on a part without the rates: the rules (`combine`).
    pub plan_cost: bool,
    /// plan-cost `linear` (design §4a concat): a sum of tile sites over one
    /// `k` range (`linear_plan`) runs as one site table launch — `z` selects
    /// term and partition, every partial added into the zeroed sum — where
    /// [`profile::TargetProfile::site_ns`] says it beats the sites' own plans.
    pub linear: bool,
    /// site-geometry `fit-first` (R1): an FMA tile's thread grid taller than
    /// the site (`ty·tm > rows`) takes `row-fit`'s reshape — `ty` halved, `tx`
    /// doubled, threads kept — before the fill-the-SMs step halves its
    /// threads (a 64-row convolution's 2-warp blocks). Off: the shrink first.
    pub fit_first: bool,
    /// site-geometry `tf32-radix` (R2): a mixed-radix site whose products
    /// admit `tf32` takes the `mma.sync` tile — its digit-affine operands
    /// copied element by element (`cp.async` 4 B), its digit-affine output
    /// stored cell by cell. Off: the FFMA radix tile.
    pub tf32_radix: bool,
    /// tile-efficiency T2: an `a` whose panel rows run along `k` (`a.ck == 1`,
    /// aligned) is staged untransposed, `As[m][k]`, by one 16-byte copy per
    /// vector (`cp.async.cg … 16` or one vector store); the kk loop reads `tm`
    /// 128-bit `a` fragments along `k` per `lanes` steps. Each cell's `k`
    /// order is unchanged. Only where one quarter-warp reads one panel row
    /// (`tx ≥ 8`, row-major store). Off: `a` transposed, 4-byte copies.
    /// On where the `cp.async` panel is (the `async` rule, `rules.rs`): it lost
    /// on every register-staged copy measured (rtx-4090 filled, rtx-5090 /
    /// rtx-3090 / h100-pcie) and won 21 W / 0 L on the async path.
    pub kmajor_a: bool,
    /// tile-efficiency T1: on a part with plan rates under `plan-cost`, a
    /// free-order split kernel (`S > 1`) keeps the full register tile and its
    /// threads — the machine is filled along `k`, not by shrinking the tile;
    /// the rules' shrunk tile stays the unsplit candidate. Off: a split
    /// kernel takes the rules' (`fill-wave`) tile.
    pub full_tile: bool,
    /// tile-efficiency T3: a `cp.async` panel below `tm·tn·bk ≥ 1024` FFMA
    /// per barrier yields to register staging where staging's panel is
    /// deeper ([`crate::func::gpu`] `launch_split`). Off: the deepest
    /// `cp.async` panel that fits, whatever its depth.
    pub async_depth: bool,
    /// per-site-choice D1: the shared-memory panels from the one bank degree
    /// function (`profile::bank_degree`): a k-major thread owns rows
    /// `trow + i·ty`, its `a` panel takes the bank model's pad, `lsu` counts
    /// each class's bank excess, and the register rule (A4) takes the k-major
    /// panel only where its `bk` and blocks/SM are not worse than the
    /// transposed panel's — under it `kmajor-a`'s rule does not apply. Off: the
    /// pre-sprint panels (row map, no k-major pad, no rule) and that rule.
    pub bank_plan: bool,
    /// compile-time-graph A: a row-block group may form where
    /// [`GpuArms::row_block`] allows it and the plan model, over the part's
    /// measured properties, prices the unit below the split launches (a part
    /// without plan rates forms none). No per-part rows (kernel-merging
    /// Calls); `--gpu-off=row-block` is the measurement arm, `--row-block=<b>`
    /// forces it.
    pub row_block: bool,
    /// kernel-merging `zero-fold`: an atomic site's zero pass runs inside an
    /// earlier kernel of the same sequence that the dependency fact
    /// (`access_plan`) orders after every earlier reader and writer of the
    /// output's bytes, where the plan rates price it below its own op
    /// (`zero_ns`). Off (`--gpu-off=zero-fold`): every zero pass is its own
    /// op, the emission before it.
    pub zero_fold: bool,
    /// kernel-merging `row-stage`: a row-block unit's tile phase stages its
    /// row-independent operand panels (and a row operand read from memory)
    /// through shared memory as coalesced 16-byte copies (`cp.async` where
    /// the part has it and the `async` rung is on), in a bank-conflict-free
    /// layout, double-buffered where the shared memory allows — each choice
    /// priced by the plan model. Off (`--gpu-off=row-stage`): every operand
    /// is read straight from memory, the emission before it. It acts only
    /// inside a row-block unit, which the plan model forms on no suite
    /// program yet; forced, the rtx-5080's pt_attn_1024_64 unit ran 0.908×
    /// its direct arm (ABBA, `local-rtx-5080/kernel-merging.md` Rows).
    pub row_stage: bool,
    /// kernel-merging `horizontal`: independent launches of one sequence
    /// that the dependency fact (`access_plan`) leaves unordered run as one
    /// grid, each block dispatching on its index to its launch's body, where
    /// the merged grid's footprint costs no member a wave and their last
    /// waves fit one (`func/horizontal.rs`). Off (`--gpu-off=horizontal`):
    /// every launch its own, the emission before it.
    pub horizontal: bool,
    /// kernel-merging `calls`: a chain of dependent launches of one sequence
    /// (single launches and `horizontal` groups, in walk order) runs as one
    /// persistent kernel — each launch's body a device function called over
    /// its blocks by a co-resident grid, a grid barrier between the levels
    /// the dependency fact (`access_plan`) orders — where the part's measured
    /// barrier and dependent-launch costs price the barriers at most the
    /// launches they replace (`func/calls.rs`). Off (`--gpu-off=calls`):
    /// every chain its launches, the emission before it.
    pub calls: bool,
    /// calls-issue (ADR plan-cost D2 in `calls`): the separate form of a chain
    /// pays each dependent launch's host issue where the host is slower than
    /// the member before it — `max(launch, issue − d_prev)` per boundary, the
    /// host's measured `issue_ns` against the members' device time
    /// (`func/calls.rs` `chain_ns`). Off (`--gpu-off=calls-issue`): the price
    /// without the host terms, the emission before it.
    pub calls_issue: bool,
    /// integration (regressions item 1 + fp8-leftovers): an epilogue group
    /// whose candidate 0 is a split (`group_order`: a workspace whose reduce
    /// evaluates the epilogue) launches through its site table even with one
    /// candidate — the plain launch runs an epilogue group's ordered kernel
    /// (the B1 decode's O projection on the H100 235 against 29 µs, the f8w
    /// decode's `wo` 108 against 39 µs a step). `--cands=1` keeps it (candidate
    /// 0 is the rules' plan, realized). Off (`--gpu-off=post-table`): the
    /// ordered kernel, the emission before it.
    pub post_table: bool,
    /// long-rows `row-slices`: a flow unit whose few rows leave the part's
    /// SMs idle runs each row over several blocks — every fold's partial
    /// state per block, combined by `fold_combine`'s law (its own `⊕`, a
    /// factor or shift applied after the combine), the phases its laws leave
    /// separated by grid barriers in one cooperative launch — where the
    /// part's measured stream, barrier and launch costs price it below one
    /// block a row (`func/flow.rs` `row_slices`). A part without them slices
    /// nothing. Off (`--gpu-off=row-slices`): one block a row.
    pub row_slices: bool,
    /// onto-device `cond-bounds`: `bounds_proof` ranges an index by the
    /// conditions that hold where it runs (a guard arm's, a loop's continue
    /// and exit conditions, a `Phi`'s per arm) and ranges loop-carried
    /// counters from `loop_plan` (init, affine step, continue condition), so
    /// a guarded read (RoPE's rotate_half partner) or a counter-indexed read
    /// (a decode step's position) carries no trap and its site is device
    /// capable. Off (`--gpu-off=cond-bounds`): the graph's facts are rebuilt
    /// under plain interval ranges (`mapal_ir::FactArms`), the proof before it.
    pub cond_bounds: bool,
    /// onto-device `carry-made`: a loop-carried value is device-made when the
    /// next state that replaces it is (and a loop's exit value when its
    /// payload is) — `place`'s "an operand is device-produced" rule sees
    /// through the back edge, so a step's small maps over a device-carried
    /// state (decode's RMSNorm) stay beside it instead of pulling the state
    /// to the host every trip. Off (`--gpu-off=carry-made`): the rule stops at
    /// a loop view, the placement before it.
    pub carry_made: bool,
    /// onto-device `host-par`: a GPU build's host map with the CPU build's
    /// split shape (clean body, static extent, run once, no device-only
    /// operand) runs as one split task on the pool where the host's measured
    /// dispatch cost says it pays (`func/hostpar.rs`), the CPU build's rule.
    /// Off (`--gpu-off=host-par`): the host walk runs it on one core.
    pub host_par: bool,
    /// device-geometry `tile-post`: a map whose body is a sum of products
    /// followed by elementwise terms of the same cell (a residual, a scale, a
    /// mask: [`mapal_ir::TilePost`]) is a tile site whose store evaluates the
    /// body after its fold (the epilogue, `func/resident.rs:epilogues`) —
    /// where its `b` panel serves more than one row or is read along its
    /// lanes: a lone row over a k-major `b` (a decode step's masked scores)
    /// stays the warp-a-cell kernel (`func/gpu.rs:post_sites`).
    /// Off (`--gpu-off=tile-post`): the map is a per-cell kernel (one thread
    /// a cell, its fold serial).
    pub tile_post: bool,
    /// device-geometry `row-tree` (a fact arm, `mapal_ir::FactArms`): a
    /// per-cell map whose body folds one row of an array beside folds that
    /// load nothing (RMSNorm's sum of squares, then its Heron root) splits
    /// that fold over a warp's lanes (`cell-lanes`). Off
    /// (`--gpu-off=row-tree`): only a body with one nested fold does; such a
    /// row is one thread's serial fold.
    pub row_tree: bool,
    /// device-geometry `piece-delta` (a fact arm, `mapal_ir::FactArms`): a
    /// map writing one row of each sequence of a `[B][L][…]` state,
    /// `((t ÷ d) % s == r) ? new : x[t]`, is piecewise ([`mapal_ir::Outside`]
    /// `w > 1`, [`mapal_ir::Delta::Pieces`]): on the device a range cell
    /// over its B pieces, in the state's buffer (`range-cell`). Off
    /// (`--gpu-off=piece-delta`): the map rewrites the whole state each trip.
    pub piece_delta: bool,
    /// device-geometry `head-group` (a fact arm, `mapal_ir::FactArms`): a
    /// mixed-radix tile site may have batch digits — digits of the map index
    /// both operands read (the query heads of one KV head, a sequence) —
    /// as the most significant row digits; a block's rows lie in one batch,
    /// so its `b` panel (one KV head's K or V rows) is loaded once for the
    /// group's query heads. Off (`--gpu-off=head-group`): such a map is a
    /// per-cell kernel (each query head re-reads its KV head's rows).
    pub head_group: bool,
    /// exp (ADR-0042 D5): on the default face `exp` takes the fastest form
    /// the bound admits — the host's contracted polynomial (vectorizes), the
    /// device's `ex2.approx` after a Cody–Waite reduction where the part's
    /// measured `ex2` error meets the bound (`profile::Caps::ex2`). Off
    /// (`--gpu-off=exp-hw`, `--no-exp-hw`): the oracle's steps on every face.
    pub exp_hw: bool,
    /// math builtins (ADR-0046 D10): on the default face every math builtin
    /// takes the cheapest realization its contract admits from the side's
    /// declared units and definitions (`sqrt.approx`, the `ex2` form, a
    /// target's routine). Off (`--gpu-off=math-hw`, `--no-math-hw`): the exact
    /// face's choice on every face — the reference, or an `Exact` unit (IEEE
    /// `sqrt`).
    pub math_hw: bool,
    /// exp (ADR-0042 D7, a fact arm, `mapal_ir::FactArms`): `fold_combine`'s
    /// rescale law — a value read only through `exp` of a difference lets a
    /// block's partial take its own shift (online softmax in `row-slices`).
    /// Off (`--gpu-off=exp-rescale`, `--no-exp-rescale`): such a value is a
    /// phase.
    pub exp_rescale: bool,
    /// half-precision (ADR-0043): a sum of products whose operand is the exact
    /// widening of a `bf16` / `f16` array element (`tile_plan_narrow`) is a
    /// device tile site whose kernel loads the 2-byte format and widens it.
    /// Off (`--gpu-off=narrow-tile`): such a map is a per-cell kernel (or the
    /// host's loop).
    pub narrow_tile: bool,
    /// large-gemm `streamk-fill`: stream-K walks the site's tiles over the
    /// slots its own launched grid fills — a grid whose blocks at their own
    /// geometry give the part fewer warps than [`profile::Gpu::fill_warps`]
    /// (a skinny site: 16 one-warp tiles) is no stream-K site; it takes the
    /// `fill-wave` split, which reaches that target. Off
    /// (`--gpu-off=streamk-fill`): stream-K wherever the fill class of the
    /// site's base grid allows it, at one block an SM.
    pub streamk_fill: bool,
    /// large-gemm `k-fill`: an under-filled site (no warp per scheduler at
    /// the base grid) whose free-order fold the `fill-wave` split can cut
    /// into enough partitions keeps the full register tile and its threads
    /// and fills the part along `k`, instead of shrinking the tile
    /// (`warp-fill`) or its threads — a tile below `8 × 8` issues a shared
    /// load per fewer than 16 FFMA. Off (`--gpu-off=k-fill`): the shrunk
    /// tile.
    pub k_fill: bool,
    /// long-prefill `uniform-base` (a fact arm, `mapal_ir::FactArms`): a
    /// mixed-radix tile read may take terms of a number its map captures (a
    /// loop counter slicing a head group or a weight panel); the launch
    /// moves the operand's pointer by them. Off (`--gpu-off=uniform-base`):
    /// such a map is a per-cell kernel.
    pub uniform_base: bool,
    /// long-prefill `carry-take`: a loop carry whose init's device copy is
    /// valid and dies at the loop's entry (`LastUsePlan`) takes that buffer
    /// as its first state, host-built or not — the S65 move of a
    /// device-built init, widened. Off (`--gpu-off=carry-take`): a host-built
    /// init is uploaded again into the carry's own buffer.
    pub carry_take: bool,
    /// long-prefill `slice-launches`: a sliced unit's phases may meet at
    /// kernel boundaries — one launch a phase — where the part's measured
    /// launches (`launch_at` × phases) undercut one cooperative launch and
    /// its grid barriers (outside a loop with the process's first
    /// cooperative launch, `coop_first_ns`). Off
    /// (`--gpu-off=slice-launches`): the cooperative launch only.
    pub slice_launches: bool,
    /// caching-memory `unchanged` (a fact arm, `mapal_ir::FactArms`): the
    /// trip deltas of `unchanged_plan` reach past `write_delta`'s carried
    /// arrays — through every map whose cells read rows of a changed array
    /// (a row fold, a per-row scale, a projection, a rotation) — so the
    /// retained sites and cells below see them. Off
    /// (`--gpu-off=unchanged`): `write_delta`'s deltas only.
    pub unchanged: bool,
    /// caching-memory `retain-cell`: a device cell map inside a loop whose
    /// output differs from its previous trip's only in a row delta
    /// (`unchanged_plan`) launches over that delta on every trip after the
    /// loop's first, in its own persisting buffer. Off
    /// (`--gpu-off=retain-cell`): every trip launches the whole map.
    pub retain_cell: bool,
    /// caching-memory `retain-pieces`: a retained tile site whose operand
    /// changes in one row of every sequence (`Delta::Pieces`) computes those
    /// rows as one site over the gathered rows into a compact buffer, then
    /// places them (`mapal_gpu_copy2d`). Off (`--gpu-off=retain-pieces`):
    /// such a site runs whole every trip.
    pub retain_pieces: bool,
    /// caching-memory `retain-same`: a device cell map inside a loop whose
    /// output is the same on every trip (`unchanged_plan`'s `same`: every
    /// input unchanged — a transform of a carried array the loop never
    /// writes) launches on the loop's first trip only, in its own persisting
    /// buffer. Off (`--gpu-off=retain-same`): every trip launches it.
    pub retain_same: bool,
    /// caching-memory `cone-span`, under `entry-arena`: a buffer freed only at
    /// exit lives in the arena from its first writer (a loop's body: its
    /// outermost loop's entry, the `LoopEnter`'s position; an epilogue's site,
    /// a unit's tile site, a kernel its zero pass rides) to exit — placed
    /// first-fit among the straight-line members' intervals (a member a
    /// carry took: to exit), so it takes the bytes of what died before (a
    /// prebuilt cache's inputs). Off (`--gpu-off=cone-span`): every such
    /// buffer lives over the whole function, after the members.
    pub cone_span: bool,
    /// caching-memory `arena-fit`: the arena's members are also placed
    /// largest first (then in birth order), first-fit, and that arena kept
    /// where it is smaller and `zero-fold` folds as many zero passes under
    /// it — largest first packs tighter where small early buffers would push
    /// a large late one above them, birth order where it does not (and a
    /// tighter packing may put a member between a site's zero and the
    /// site). Off (`--gpu-off=arena-fit`): birth order only.
    pub arena_fit: bool,
    /// tensor-cores (ADR-0043 D4 / D6): a tile site whose product admits a
    /// 16-bit member (`: bf16 | f32`, `: f16 | f32`) takes the part's
    /// matrix instruction for that format (`mma.sync.m16n8k16`, `Caps::matrix`):
    /// 16-byte `cp.async` panels of a narrow operand, an f32 operand rounded
    /// once as its panel is staged, `ldmatrix` fragments, f32 accumulators.
    /// Off (`--gpu-off=half-tile`): the f32 / narrow FMA tile it also admits.
    pub half_tile: bool,
    /// tensor-cores: a narrow (`bf16` / `f16`) operand's FMA-tile panel is
    /// copied in its own format by 16-byte `cp.async` and widened as each
    /// fragment is read from shared memory, on the `cp.async` path the rules
    /// pick (`async`). Off (`--gpu-off=narrow-async`): the register-staged
    /// tile that widens each panel as it copies it (slice 1).
    pub narrow_async: bool,
    /// tensor-cores `staged-regs`: a register-staged FMA tile whose panel
    /// (every copy's value live from load to store) and tile exceed the
    /// thread's registers is refused — the spilling geometry (the block
    /// decode's one-row narrow sites, 516 registers at `bk` 128). Off
    /// (`--gpu-off=staged-regs`): the rules' tile however large its panel.
    pub staged_regs: bool,
    /// tensor-cores `half-fill`: the 16-bit tile is taken where the site's
    /// flops per operand byte reach the part's FFMA balance or its own grid
    /// fills the warp schedulers; elsewhere the FMA tile, which splits `k`
    /// to fill the part. Off (`--gpu-off=half-fill`): the 16-bit tile
    /// wherever it fits.
    pub half_fill: bool,
    /// tensor-cores `narrow-radix` (a fact arm, `mapal_ir::FactArms`): a
    /// mixed-radix tile site may read the exact widening of a narrow element
    /// (attention over a `bf16` KV cache). Off (`--gpu-off=narrow-radix`):
    /// such a map is a per-cell kernel.
    pub narrow_radix: bool,
    /// gemv-stream `vec-cells`: the one-thread-per-cell kernel of a streaming
    /// site (flops per operand byte below the part's FFMA balance) gives a
    /// thread the adjacent cells one 16-byte load of `b` covers — `16 / e`
    /// lanes, where the layout facts prove the load (`b` unit-stride along
    /// the lane, its base and `k` stride whole 16-byte granules, the lanes a
    /// whole number of vectors) and `a` is the same for every lane: one
    /// `ld.global.nc.v4` and one `a` load per `16 / e` FMAs, the `k` steps'
    /// loads of one trip issued before their FMAs. Off
    /// (`--gpu-off=vec-cells`): one cell a thread, scalar loads.
    pub vec_cells: bool,
    /// gemv-stream `stream-fly`: a streaming one-thread-per-cell site's `k`
    /// split is the fewest partitions whose grid puts a block on every SM
    /// with the part's measured in-flight knee in flight on each
    /// ([`profile::Gpu::stream_fly_per_sm`],
    /// [`profile::TargetProfile::split_stream`]). Off
    /// (`--gpu-off=stream-fly`): the warp fill (`fill-wave`) alone.
    pub stream_fly: bool,
    /// regressions `vec-spread`: the vector cells (`vec-cells`) are taken
    /// where the part's DRAM share an SM asks for the grid's spread only
    /// when their grid holds the SMs as evenly as one cell a thread at the
    /// same split ([`rules::vec_cells_pay`]). Off (`--gpu-off=vec-spread`):
    /// the vector cells at any spread.
    pub vec_spread: bool,
    /// gemv-stream `bulk-stage` (TMA): the warp `mma` tile's panels are
    /// staged by the part's bulk-copy engine ([`profile::Caps::bulk`]) where
    /// its geometry covers every panel row ([`profile::BulkCopy::box_fits`]):
    /// one warp issues a row copy each (`cp.async.bulk`), a stage's
    /// transaction barrier (`mbarrier.expect_tx`) counts its bytes, and every
    /// thread waits on it (`mbarrier.try_wait.parity`) instead of on its own
    /// `cp.async` groups. Off (`--gpu-off=bulk-stage`): every thread's 16-byte
    /// `cp.async` copies.
    pub bulk_stage: bool,
    /// hopper-wgmma-tma R2 (`wgmma-ws`): the Hopper TF32 site as a
    /// warp-specialized `wgmma` kernel — one producer warpgroup copying each
    /// stage as whole tensor boxes (`cp.async.bulk.tensor`, a host-encoded
    /// map per operand) onto a full barrier and rounding the `aᵀ` box once,
    /// consumer warpgroups releasing each stage on an empty barrier once the
    /// group reading it completed (`wgmma.wait_group`)
    /// ([`profile::TargetProfile::gpu_wgmma_ws_tile`]). Off
    /// (`--gpu-off=wgmma-ws`): the S53 kernel.
    pub wgmma_ws: bool,
    /// hopper-wgmma-tma R3 (`ws-regs`): the warp-specialized kernel moves
    /// registers from its producer to its consumers (`setmaxnreg`) where the
    /// part reallocates ([`profile::Caps::reg_realloc`]). Off: every thread
    /// keeps the launch's even share (a smaller consumer tile).
    pub ws_regs: bool,
    /// hopper-wgmma-tma R1 (`tma-digits`): the `b` panel of a stage as ONE
    /// tensor box — its lane axis split into digits `lane = 32·q + r` (a
    /// 3-d map: the 32-word swizzle row, `k`, then `q` at 128 bytes), which
    /// lands the `bn/32` column slabs one after another exactly as `bn/32`
    /// 2-d boxes do: 2 copies a stage instead of `1 + bn/32` (the engine's
    /// measured cost is per copy, `tma_copy_ns`). Off: a 2-d box per slab.
    pub tma_digits: bool,
    /// wgmma-16bit (`wgmma16`): **a 16-bit tile on the part's warpgroup
    /// product** — where the part lists `wgmma.m64nNk16` for the site's
    /// format ([`profile::Caps::matrix_group`]) at a measured rate above its
    /// warp `mma` ([`crate::rules::wgmma16_pays`]), the bf16 / f16 site takes
    /// the warp-specialized tile family of `wgmma-ws` with the format a
    /// parameter: `c = a·b` as stored, `a` from a K-major descriptor (stored
    /// in the format) or registers (f32, rounded in its fragment load), `b`
    /// from a descriptor in either major, TMA boxes on an mbarrier ring
    /// ([`profile::TargetProfile::gpu_wgmma_ws_tile_of`]). Off
    /// (`--gpu-off=wgmma16`): the warp `mma.sync.m16n8k16` tile.
    pub wgmma16: bool,
    /// flash-attention `stream`: a row-block group `stream_plan` proves one
    /// pass over windows of its axis (attention's `q·Kᵀ → softmax → P·V`)
    /// runs as one kernel walking the axis window by window — each fold's
    /// state carried by its law, the axis-wide values never stored, each
    /// operand window read once a block — where the plan model prices it
    /// below the split launches (`func/stream.rs`). Off
    /// (`--gpu-off=stream`): the split realization, the emission before it.
    pub stream: bool,
    /// flash-attention `stream` (a fact arm, `mapal_ir::FactArms::stream`):
    /// `stream_plan` forms its groups. Off (`--gpu-off=stream-fact`,
    /// `--no-stream`): no stream group.
    pub stream_fact: bool,
    /// flash-attention `split-kv` (R3): a stream unit whose row blocks leave
    /// the part's SMs idle splits each row block's windows over `s` blocks;
    /// the row block's last block to finish combines the partial states by
    /// their laws (a ticket: a fence and one atomic, no second launch). Off
    /// (`--gpu-off=split-kv`): one block a row block.
    pub split_kv: bool,
    /// sweep-attention-2 `stream-mma`: a stream unit whose products each
    /// admit a 16-bit member the part has a warp matrix product for (its
    /// `m16n8k16` geometry, the accumulator meeting the member's `u_a`) folds
    /// them on it — the threads in the accumulator's fragment layout (two
    /// rows and a quad's columns a thread), the row operand's fragments held
    /// in registers, the producer's window cells passed to the consumer as
    /// its `a` fragments — where the plan model prices it below the FFMA
    /// stream. Off (`--gpu-off=stream-mma`): the FFMA stream.
    pub stream_mma: bool,
    /// sweep-attention-2 `kmajor-b`: a register-staged tile whose `b` is
    /// unit-stride along `k` and not along its lanes (a decode step's keys,
    /// `K[key][u]`) copies its panel along those rows — a vector of `k` a
    /// copy, a key's row coalesced, scattered into the `[k][n]` panel the
    /// fragments read — instead of one element a copy down a column (each
    /// thread a different key's row). Off (`--gpu-off=kmajor-b`): the
    /// element copies.
    pub kmajor_b: bool,
    /// flash-attention F5 (a fact arm, `mapal_ir::FactArms::guard_box`): a
    /// stream group's masked producer has its mask's form, and a wholly
    /// masked window is skipped where F6 allows. Off (`--gpu-off=guard-box`,
    /// `--no-guard-box`): every window computed.
    pub guard_box: bool,
    /// mask-frontend (a fact arm, `mapal_ir::FactArms::guard_terms`): a
    /// mask's side through a captured array — its element law, else a
    /// per-row load — or a per-launch number. Off (`--gpu-off=guard-terms`):
    /// digit forms of the cell index alone.
    pub guard_terms: bool,
    /// mask-frontend (a fact arm, `mapal_ir::FactArms::walk_order`): a
    /// stream group's blocks walk its rows in the seed's tile order. Off
    /// (`--gpu-off=walk-order`): in the rows' own order.
    pub walk_order: bool,
    /// mask-frontend `stream-numbers`: a streamed unit's epilogue numbers
    /// are launch arguments. Off (`--gpu-off=stream-numbers`): a unit whose
    /// epilogue reads a number after its fold is no stream (as before).
    pub stream_numbers: bool,
    /// bf16-class `finite-check`: a stream group's masked windows are
    /// skipped where a consumer's column operand is finite only at run time
    /// (`mapal_ir::StreamGuard::check`) — the kernel reads the skipped
    /// windows' column elements and runs those windows where one is not
    /// finite. Off (`--gpu-off=finite-check`): such a group skips none.
    pub finite_check: bool,
    /// bf16-class `narrow-window`: a stream group whose column operand is
    /// stored bf16 (a bf16 KV cache) streams on the matrix product, its
    /// windows holding the operand's own bytes (half an f32 window's) and
    /// its fragments taking them as they stand. Off
    /// (`--gpu-off=narrow-window`): such a group is no stream.
    pub narrow_window: bool,
    /// bf16-class `device-trap`: a map whose only possible trap is an index
    /// `bounds_proof` cannot prove (`mapal_ir::CategoryIr::trap_cell_plan`:
    /// an embedding's row through a run-time token id) may run as a device
    /// cell — each such read checked in the kernel (out of bounds: the trap
    /// recorded in the module's word, element 0 read), the trap raised at
    /// the next host checkpoint (a print, the clock, a trap, the exit).
    /// Off (`--gpu-off=device-trap`): such a map runs on the host.
    pub device_trap: bool,
    /// flash-attention F7 (a fact arm, `mapal_ir::FactArms::finite`):
    /// `finite_plan` proves values finite. Off (`--gpu-off=finite`,
    /// `--no-finite`): nothing is, so no masked window is skipped.
    pub finite: bool,
    /// sweep-attention `unit-uniform` (a fact arm,
    /// `mapal_ir::FactArms::unit_uniform`): a tile whose operand address a
    /// captured number moves (a loop counter slicing a KV head group) joins
    /// `unit_plan`'s extended groups, the unit's launch moving that pointer
    /// — attention inside a head-group loop streams. Off
    /// (`--gpu-off=unit-uniform`): such a tile launches alone, as before.
    pub unit_uniform: bool,
    /// f32-tensor-cores R-T1 `mma-combine`: a tensor-core tile site (TF32,
    /// 16-bit, split) cuts its `k` into `S` pieces whose partials add into
    /// the zeroed output (`splitk-atomic`) or a workspace (`splitk`) — the
    /// split the FMA tile takes (ADR-0040 D4), now on the `mma.sync` kernels:
    /// each piece's `cp.async` pipeline over its own `k / S`, the seed on
    /// piece 0. Off (`--gpu-off=mma-combine`): tensor tiles never split.
    pub mma_combine: bool,
    /// f32-tensor-cores R-F `split-tile`: a `{f32}` sum of products on the
    /// default face (free order) takes a split realization on the part's
    /// matrix instruction (`mapal_ir::split`: tf32x3, narrow1) where its
    /// coefficient meets D2 at the site's `K` and the part's matrix rate pays
    /// for the form's passes against its FFMA rate. Off (`--gpu-off=split-tile`):
    /// the FFMA tile (the emission before it); on (`--gpu-on=split-tile`):
    /// the split wherever the bound admits it, past the rate rule.
    pub split_tile: bool,
    /// tensor-cores-2 `split-issue`: a split tile's realization at its
    /// issue rate — each f32 element cut by integer rounding with the value
    /// classes left to the guard (which now also takes an element whose
    /// leading part rounds to infinity), and each tile's zeroed chain
    /// carried over the k-steps the bound admits before its round-to-nearest
    /// add ([`func::gpu::split_flush`]). Off (`--gpu-off=split-issue`): the
    /// 5fcb401b realization (`cvt.rna`, the overflow select, a flush every
    /// k-step) — the A100's tf32x3 loser (1612 SASS instructions a panel of
    /// 96 `HMMA` against 792).
    pub split_issue: bool,
    /// tensor-cores-2 `split-least`: an f32 × bf16 product takes the split
    /// form of fewest part products its bound admits at the site's `K` —
    /// bf16x2n (two bf16 parts of the f32 operand, split error `2⁻¹⁶`)
    /// before bf16x3n (three, exact). Off (`--gpu-off=split-least`):
    /// bf16x3n wherever the split is taken.
    pub split_least: bool,
    /// f32-tensor-cores R-T2 `mma-depth`: a tensor-core tile takes the panel
    /// depth keeping the most operand bytes in flight a block at its
    /// residency (`profile::mma_depth_pick`: `bk` 16 × 5 stages against `bk`
    /// 32 × 2) instead of the deepest panel two stages fit (S52). The rules
    /// ship it off on every part (it lost every measured site); `--gpu-on=
    /// mma-depth` measures it.
    pub mma_depth: bool,
    /// gapfix tensor-tile (C3) `mma-fly`: where a tensor tile's panel loads
    /// outlast the products that cover them (an operand past the L2 keep,
    /// fragments read as stored, the S52 panel's DRAM time within its
    /// products' but not their latency: `profile::TargetProfile::gpu_mma_fly`)
    /// its ring keeps the part's in-flight bytes
    /// (`profile::Gpu::stream_fly_per_sm`) past its wait on every SM — the
    /// residency, panel depth and stages of `profile::mma_fly_pick`;
    /// elsewhere the S52 panel. Off (`--gpu-off=mma-fly`): the S52 panel on
    /// every site.
    pub mma_fly: bool,
    /// sweep-gemm `round-store` (ADR-0041 D5 `round_at` over arrays): an f32
    /// array a map produces whose every element read is a multiplicand of a
    /// product admitting `bf16` ([`mapal_ir::PrecisionPlan::store`]) is
    /// stored rounded once by its producer ([`CategoryIr::narrow_stored`]):
    /// its tile panels copy 2-byte elements (`ldmatrix`, no conversion per
    /// fragment, half the shared memory). Default (order-free) face only.
    /// Off (`--gpu-off=round-store`): each reader rounds its copy.
    pub round_store: bool,
    /// sweep-gemm-2 `warm-zero`: a module whose launches zero an output
    /// (an atomic split's or stream-K's zero op, `mapal_gpu_zero`) has the
    /// runtime run one zero of a scratch buffer on the `eager-init` thread,
    /// before the clock: the driver initializes its memset path on the
    /// process's first fill of 8 KiB or more (CUDA 13, rtx-5080: 13–17 µs
    /// that call against 2–4 µs after), which a one-call program otherwise
    /// pays inside its timed region. Off: the first zero pays it.
    pub warm_zero: bool,
    /// sweep-gemm-2 `streamk-price`: on a part whose rates price a stream-K
    /// fixup (`DeviceCost`: the FMA tile rate, the `red` rate, the zero
    /// pass), a site's stream-K is weighed against the S63 count by those
    /// prices ([`crate::func::gpu`] `streamk_priced`: the pieces' partials and
    /// the zero pass against the panels the walk saves) — under-filled sites
    /// too, whose class ships `streamk` off (site-level wave quantization).
    /// Off (`--gpu-off=streamk-price`): the panel count (`iters + 1 < split`)
    /// in the classes that ship `streamk`.
    pub streamk_price: bool,
    /// sweep-runtime-2 `streamk-epi`: an epilogue group (S54, no `tile-post`
    /// body) whose site the rules walk by stream-K keeps it — every piece
    /// adds its partials into a zeroed workspace and the CTA that completes a
    /// tile (a per-tile ticket: the stream-K walk names how many CTAs share
    /// it) evaluates the group's members on the whole sums. Off
    /// (`--gpu-off=streamk-epi`): the group's ordered kernel, as before (a
    /// measurement flag; the stream-K rule decides where it applies).
    pub streamk_epi: bool,
    /// bf16 tensor-tile `mma-walk`: a 16-bit tensor tile (no split form)
    /// whose grid rounds over the SMs' matrix pipes (`⌈tiles / sms⌉` tiles
    /// on the busiest SM) cost more than the same products spread evenly is
    /// walked by stream-K over every resident slot at its whole warp tile
    /// (`func::gpu::mma_walk`, priced by `func::gpu::MmaWalk`); the shared
    /// tiles' partials add into the zeroed output, or an epilogue group's
    /// workspace and ticket (`streamk-epi`). Off (`--gpu-off=mma-walk`): the
    /// ordered tile or its `k` cut, as before (a measurement flag; the price
    /// decides where it applies). `--gpu-on=mma-walk` walks every site the
    /// walk admits, priced or not (the measurement and emulator arm).
    pub mma_walk: bool,
    /// hopper-launch `tma-warm`: a module carrying a kernel that reads
    /// tensor maps (the `<kernel>_tmap` recipes the runtime finds in its
    /// text at load) has the runtime launch its own one-box TMA kernel once
    /// on the `eager-init` thread, before the clock: the driver readies a
    /// context's tensor-copy path at the first launch of a kernel holding a
    /// `cp.async.bulk.tensor` (CUDA 13: 1.34–1.49 ms that host call on the
    /// RTX 5080, against 3–4 µs after; on the H100 the 120–270 µs a
    /// `wgmma-ws` call spent outside its kernel, cross-arch 5fcb401b), which
    /// a one-call program otherwise pays inside its timed region. Host only
    /// (the module's flag word); off (`--gpu-off=tma-warm`): the first TMA
    /// launch pays it.
    pub tma_warm: bool,
    /// hopper-launch `ws-persist`: the warp-specialized `wgmma` kernel as a
    /// persistent grid — `sms × blocks_per_sm` CTAs each walking the tiles
    /// `cta, cta + ctas, …` in raster order, the stage ring and its barrier
    /// phases running on across tiles, so a CTA's producer copies the next
    /// tile's boxes under this tile's epilogue and the pipeline fills once
    /// a CTA — where [`crate::rules::ws_persist_pays`] prices the fills and
    /// epilogues the walk overlaps above its tail. The rules ship it off (its
    /// H100 measurement is mixed, the separating fact open: `rules::rungs`);
    /// `--gpu-on=ws-persist` measures the walk. Off: a CTA a tile.
    pub ws_persist: bool,
    /// gaps-disasm C1 `load-dispatch`: the module registers its site tables
    /// before the runtime's init (`mapal_gpu_sites`), so each table's pick
    /// (where one is decided without a run: forced, stored), its candidates'
    /// kernels and shared-memory limits and a cooperative table's occupancy
    /// are resolved when the module loads — a one-call program's first site
    /// goes from its clock to its first driver call without hashing its
    /// table or scanning the wisdom (the runtime finds every preloaded kernel
    /// by its id in one hash probe either way). Off
    /// (`--gpu-off=load-dispatch`): no registry; all of it at the first call.
    pub load_dispatch: bool,
    /// gaps-disasm C1 `zero-host`: an atomic output's zero pass that no
    /// earlier kernel folded (`zero-fold`) is the runtime's zero kernel — one
    /// launch on the host — never the driver's memset API (RTX 5080: 3.7–5.9
    /// µs a call inside a one-call program's clock, against 2.5–3.9 µs a
    /// launch). Off (`--gpu-off=zero-host`): `cuMemsetD8Async`.
    pub zero_host: bool,
    /// gaps-disasm C1 `zero-hoist`: an atomic site's zero that no earlier
    /// kernel takes (`zero-fold`) is issued at the earliest point the
    /// zero-fold walk allows — right after the last step that touches the
    /// output's bytes, or the sequence's start — instead of between the
    /// site's dispatch and its launch, where its host issue delays the
    /// kernel (an arena member, in the straight-line walk; a loop trip keeps
    /// it at the site), skipped at run time where the site's decided pick
    /// adds nothing (`mapal_gpu_zero_for`). Off (`--gpu-off=zero-hoist`): at the
    /// site.
    pub zero_hoist: bool,
    /// gaps-disasm C5 `split-price`: an under-filled FMA tile whose operands
    /// stream from DRAM takes the atomic `k` split its price gives — the
    /// stream at the split grid's spread over the SMs against every
    /// partition's partial cells at the part's measured
    /// `DeviceCost::split_cell_ps` (`TargetProfile::split_priced`) — where the
    /// fill rule took the fewest partitions that reach `fill_warps`. A part
    /// without the property keeps the fill rule. Off
    /// (`--gpu-off=split-price`): the fill rule.
    pub split_price: bool,
    /// gapfix `iota-law`: an array a body fn captures whose law
    /// (`elem_plan` through captures: an `iota` passed in) is closed is
    /// computed at its read, never loaded (a norm's fold over a captured
    /// `iota`). Off: the capture is loaded.
    pub iota_law: bool,
    /// gapfix `stream-ld`: a tile operand the grid reads once whose bytes
    /// exceed what the L2 keeps (`Gpu::l2_keep_bytes`) loads with the
    /// streaming policy (`ld.global.cs`: L2 evict-first) where the part has
    /// it (`Caps::l2_evict`). Off: the default policy.
    pub stream_ld: bool,
    /// gapfix `shfl-tree`: a row unit's lane tree combines inside a warp
    /// by register shuffles and across warps through one shared exchange
    /// (one barrier), where its type is 32-bit. Off: a barrier per level.
    pub shfl_tree: bool,
    /// gapfix `kvec-swizzle`: a register-staged k-major `b` panel's words
    /// spread over the shared-memory banks by an XOR of the stored row's
    /// word group (the bank geometry: `Gpu::smem_banks`). Off: the dense
    /// panel.
    pub kvec_swizzle: bool,
    /// model-ready placement `trip-cells`: a cell a loop runs every trip is a
    /// candidate whatever its size, and an operand made once (outside every
    /// loop and guard arm) is in device memory for it (uploaded once at its
    /// producer) — the fixed point keeps it where it moves no bytes. Off
    /// (`--gpu-off=trip-cells`): a cell is a candidate only where it pays
    /// (`place`) or is fed, the placement before it.
    pub trip_cells: bool,
    /// model-ready placement `bytes-place`: a per-trip cell runs where it
    /// moves fewer bytes per trip (uploads of operands the device lacks and
    /// the copy-out of a host-read output, against the copy-out of operands
    /// only the device holds and the upload of a device-read output; a tie
    /// to the device when a device kernel reads the output). Off: every
    /// operand must already be on the device.
    pub bytes_place: bool,
    /// full-model `once-bytes`: `bytes-place`'s rule for a cell outside every
    /// loop too — it runs where it moves fewer bytes, once (a prompt's RMSNorm
    /// reads a 4 KB gain the host holds and 4 MB the device does: the gain
    /// goes up, not the 4 MB down and back). Off: such a cell needs every
    /// operand on the device already.
    pub once_bytes: bool,
    /// bf16-integrate `cand-made`: a map an operand of which a candidate
    /// device cell or site makes is a placement candidate too (S62's
    /// "operand device-produced" over the candidates, the fixed point
    /// deciding it), not only where its own roofline pays — a prompt's gather,
    /// its norm's row fold and the norm's scale are decided as one chain. Off
    /// (`--gpu-off=cand-made`): the roofline or a tile site's output alone
    /// makes a candidate (as before).
    pub cand_made: bool,
    /// full-model `once-range`: a piecewise range map (`map_law` `Outside`)
    /// outside every loop that no carry writes in place is a whole-extent
    /// device cell, run once. Off: such a map is no cell (the host runs it).
    pub once_range: bool,
    /// model-ready placement `range-chain`: a chain of in-place range maps
    /// over a carried state (a trip's appends to one cache) writes the
    /// carry's buffer link by link, each written array dead past its
    /// writer. Off: only a range map writing the next state over the view.
    pub range_chain: bool,
    /// model-ready placement `stride-lanes`: a nest cell whose fold reads an
    /// operand at unit stride along the fold (`step_algebra`'s regroup
    /// reads) takes a warp a cell — coalesced — even when its cells fill
    /// the part. Off: a warp a cell only where the cells leave threads idle.
    pub stride_lanes: bool,
    /// model-ready placement `carry-fwd`: a cell's output that is a device
    /// carry's next state forwards its pointer into the back edge — no host
    /// read there. Off: the back edge's pack is a host read.
    pub carry_fwd: bool,
    /// model-ready integration `lanes-unroll`: a warp-a-cell fold loop
    /// (`cell-lanes` / `stride-lanes`) is unrolled `NAIVE_UNROLL` trips
    /// (`llvm.loop.unroll.count`) where each lane walks at least that many —
    /// the loads in flight a lane keeps, as the stream rules price a naive
    /// thread (`NAIVE_UNROLL`), stated in the IR instead of left to `ptxas`
    /// (which unrolls the loop only when its index is loaded: an `iota-law`
    /// index left it rolled, one weight load in flight a warp). Off
    /// (`--gpu-off=lanes-unroll`): no loop metadata.
    pub lanes_unroll: bool,
    /// residency `fly-unroll`: a warp-a-cell fold whose grid holds too few
    /// warps an SM to keep the part's stream bytes in flight
    /// (`Gpu::stream_fly_per_sm`) at `NAIVE_UNROLL` loads a lane unrolls
    /// deeper — the power of two that keeps them, at most the lane's trips
    /// (a 512-row one-row projection: 6 warps an SM). Off
    /// (`--gpu-off=fly-unroll`): every lane loop `NAIVE_UNROLL`.
    pub fly_unroll: bool,
    /// ADR-0050 C4 `machine-place`: a tile site's launch is the machine
    /// model's search ([`crate::place::placed`]: the schedule the one
    /// evaluator prices lowest over every lowering's space, through the
    /// post-lowering gate, kept to the rules' launch within the model's
    /// error band), lowered by `launch_of`. Off (the default until ADR-0050
    /// C6 confirms it): the rule chain.
    pub machine_place: bool,
}
