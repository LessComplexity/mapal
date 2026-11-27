//! [`GpuRungs`](super::GpuRungs)'s first fields, in declaration order: the
//! tile kernel's rungs from S52 (`arch`) through the `calls` rounds
//! (`row_slices`).

/// Appends this part's fields to the field list it is handed and passes the
/// list to the next macro in `[..]` (the chain `super` starts).
macro_rules! kernel_rounds {
    ([$next:ident $(, $rest:ident)*] $($fields:tt)*) => {
        $next! {
            [$($rest),*]
            $($fields)*
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
        }
    };
}
