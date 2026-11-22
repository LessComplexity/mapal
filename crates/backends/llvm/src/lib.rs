//! LLVM backend: emits textual LLVM IR (`.ll`) piped to `clang` (ADR-0020 §1;
//! backend-llvm DESIGN). The entry point [`emit`] realizes the functor
//! `F_LLVM : Mapal-Cat → LLVM-Cat` as a String — the translation unit **is** the
//! artifact (ADR-0020). Semantics are the interpreter oracle's by construction:
//! every `Print`/trap routes through the shared `mapal-rt` runtime (ADR-0020 §2),
//! integer arithmetic wraps, `Div`/`Mod`/`Index`/`Update` are guarded (an S20
//! `bounds_proof`-proven `Index` elides its statically-dead guard), and floats
//! are IEEE at width.
//!
//! Emission is deterministic **per resolved profile** (L2): all names come from
//! per-function ordinals and a rising counter, never slotmap bits, so one `(IR,
//! EmitOpts, TargetProfile)` triple always yields the same text. The default
//! target is `native` (S46, ratified ADR-0038): it reads host facts, so default
//! emitted text is machine-dependent *by design* — reproducible emission names a
//! profile explicitly, and [`emit_detailed`] reports which profile was resolved.
//! The one capability gap (L3) is the non-canonical loop shape (multi-merge
//! SCC), rejected as [`EmitError::Unsupported`] — the same scope boundary as
//! interp M1 / rewrite.

mod exp;
pub mod flags;
mod func;
mod loops;
mod module;
mod narrow;
mod profile;
mod reuse;
mod rules;
mod ty;

pub use crate::profile::TargetProfile;
pub use crate::profile::resolve as resolve_profile;
pub use crate::profile::{CostOver, DeviceCost};

/// The profile a target name resolves to (`native` included), `None` for an
/// unknown name — what [`EmitOpts::target`] and [`EmitOpts::host`] accept.
pub fn target_profile(name: &str) -> Option<&'static TargetProfile> {
    profile::resolve(name)
}

/// hardware-properties: the rungs the rules ship on GPU profile `target`, one
/// line per fill class and face — `wps\tface\trungs on` (wps 0, 1 and 2 for
/// "2 or more"; faces `Exact`, `Contract`, `Tf32`) — for the evidence pass
/// (`benches/gpu/evidence.py`). `None` for an unknown or CPU profile.
pub fn rung_table(target: &str) -> Option<String> {
    let p = profile::resolve(target)?;
    p.gpu()?;
    let mut out = String::new();
    for wps in [0, 1, 2] {
        for face in [
            profile::Face::Exact,
            profile::Face::Contract,
            profile::Face::Tf32,
        ] {
            let r = p.gpu_rungs_on(wps, face);
            let on: Vec<&str> = GpuRungs::NAMES
                .iter()
                .copied()
                .filter(|n| r.get(n))
                .collect();
            out += &format!("{wps}\t{face:?}\t{}\n", on.join(","));
        }
    }
    Some(out)
}

use mapal_ir::{CategoryIr, FuncId, MorphismId, Operation, PathPlan, SourceLoc, TaskKind};
use slotmap::SecondaryMap;

use crate::func::{FnAttrs, FnEmit, packing_site};
use crate::module::{
    HEAP_DECLS, PAR_DECLS, PERF_DECLS, PREFETCH_DECL, PanelWrite, RT_DECLS, SME_DECLS,
    collect_str_globals, emit_main_wrapper, emit_str_globals, sme_panel,
};

/// A structured, renderer-free emission error (ADR-0020 §1; C3).
#[derive(Clone, Debug, PartialEq)]
pub enum EmitError {
    /// A capability-matrix rejection (the ✋ cell): a feature outside the
    /// realized set — here, the non-canonical (multi-merge) loop shape (L3).
    Unsupported { feature: String, loc: SourceLoc },
    /// An internal invariant violation (should not occur for sealed IR).
    Internal(String),
}

/// How the move-panel traversal is decided (S45). See [`EmitOpts::move_panel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MovePanel {
    /// **The default and the shipped path.** Fire where a recognized
    /// [`mapal_ir::MoveSite`] and the target profile's L1D geometry say it pays,
    /// with a derived block. A profile without L1D facts (`generic`) can never
    /// fire it, so the default target's emission is unchanged.
    #[default]
    Deduce,
    /// Never fire. The OFF arm of a measurement, and an escape hatch.
    Off,
    /// Force this `(w, b)` on every eligible map, ignoring both records — S44's
    /// behaviour exactly, kept **only** so a block sweep is still possible.
    Force(u64, u64),
}

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
}

impl GpuRungs {
    /// Every rung on — what S52 shipped on every part.
    pub const ALL: Self = Self {
        arch: true,
        interleave: true,
        unroll: true,
        min_blocks: true,
        warp: true,
        prefetch: true,
        mma: true,
        cp_async: true,
        wide: true,
        ldmatrix: true,
        fill: true,
        idx32: true,
        affine: true,
        pad: true,
        wide_fma: true,
        wgmma: true,
        resident: true,
        devgen: true,
        retain_delta: true,
        eager_init: true,
        arena: true,
        raster: true,
        l2_gate: true,
        l2_window: true,
        retain_resident: true,
        delta_h2d: true,
        lane_stride: true,
        rowfold: true,
        rowfold_layout: true,
        epilogue: true,
        devmap: true,
        cell_law: true,
        streams: true,
        stream_gate: true,
        pinned: true,
        pdl: true,
        pdl_early: true,
        row_fit: true,
        wave_gate: true,
        grid_share: true,
        panel_regs: true,
        fill_rows: true,
        cell_fill: true,
        async_stage: true,
        splitk: true,
        splitk_atomic: true,
        streamk: true,
        hostopt: true,
        place: true,
        by_ref: true,
        flow: true,
        window: true,
        range_map: true,
        uniform: true,
        nest_cell: true,
        tree_gate: true,
        retained_fold: true,
        preload: true,
        entry_arena: true,
        async_h2d: true,
        stage: true,
        chain_h2d: true,
        range_cell: true,
        scalar_fold: true,
        trip_batch: true,
        cell_lanes: true,
        fold_lanes: true,
        radix_tile: true,
        tf32_tile: true,
        row_split: true,
        fill_wave: true,
        plan_cost: true,
        linear: true,
        fit_first: true,
        tf32_radix: true,
        kmajor_a: true,
        full_tile: true,
        async_depth: true,
        bank_plan: true,
        row_block: true,
        zero_fold: true,
        row_stage: true,
        horizontal: true,
        calls: true,
        row_slices: true,
        cond_bounds: true,
        carry_made: true,
        host_par: true,
        tile_post: true,
        row_tree: true,
        piece_delta: true,
        head_group: true,
        exp_hw: true,
        exp_rescale: true,
        narrow_tile: true,
        streamk_fill: true,
        k_fill: true,
        uniform_base: true,
        carry_take: true,
        slice_launches: true,
        unchanged: true,
        retain_cell: true,
        retain_pieces: true,
        retain_same: true,
        cone_span: true,
        arena_fit: true,
    };

    /// The rung names `emit --gpu-off=` / `--gpu-on=` and the profile's rung
    /// table use.
    pub const NAMES: [&'static str; 106] = [
        "arch-rungs",
        "interleave",
        "unroll",
        "min-blocks",
        "warp",
        "prefetch",
        "mma",
        "async",
        "wide",
        "ldmatrix",
        "warp-fill",
        "idx32",
        "affine",
        "pad",
        "wide-fma",
        "wgmma-hopper",
        "resident",
        "devgen",
        "retain-delta",
        "eager-init",
        "arena",
        "raster",
        "l2-gate",
        "l2-window",
        "retain-resident",
        "delta-h2d",
        "lane-stride",
        "rowfold",
        "rowfold-layout",
        "epilogue",
        "devmap",
        "cell-law",
        "streams",
        "stream-gate",
        "pinned",
        "pdl",
        "pdl-early",
        "row-fit",
        "wave-gate",
        "grid-share",
        "panel-regs",
        "fill-rows",
        "cell-fill",
        "asyncstage",
        "splitk",
        "splitk-atomic",
        "streamk",
        "hostopt",
        "place",
        "by-ref",
        "flow",
        "window",
        "range-map",
        "uniform",
        "nest-cell",
        "tree-gate",
        "retained-fold",
        "preload",
        "entry-arena",
        "async-h2d",
        "stage",
        "chain-h2d",
        "range-cell",
        "scalar-fold",
        "trip-batch",
        "cell-lanes",
        "fold-lanes",
        "radix-tile",
        "tf32-tile",
        "row-split",
        "fill-wave",
        "plan-cost",
        "linear",
        "fit-first",
        "tf32-radix",
        "kmajor-a",
        "full-tile",
        "async-depth",
        "bank-plan",
        "row-block",
        "zero-fold",
        "row-stage",
        "horizontal",
        "calls",
        "row-slices",
        "cond-bounds",
        "carry-made",
        "host-par",
        "tile-post",
        "row-tree",
        "piece-delta",
        "head-group",
        "exp-hw",
        "exp-rescale",
        "narrow-tile",
        "streamk-fill",
        "k-fill",
        "uniform-base",
        "carry-take",
        "slice-launches",
        "unchanged",
        "retain-cell",
        "retain-pieces",
        "retain-same",
        "cone-span",
        "arena-fit",
    ];

    /// The rung called `name`, `None` for an unknown name.
    pub fn rung(&mut self, name: &str) -> Option<&mut bool> {
        Some(match name {
            "arch-rungs" => &mut self.arch,
            "interleave" => &mut self.interleave,
            "unroll" => &mut self.unroll,
            "min-blocks" => &mut self.min_blocks,
            "warp" => &mut self.warp,
            "prefetch" => &mut self.prefetch,
            "mma" => &mut self.mma,
            "async" => &mut self.cp_async,
            "wide" => &mut self.wide,
            "ldmatrix" => &mut self.ldmatrix,
            "warp-fill" => &mut self.fill,
            "idx32" => &mut self.idx32,
            "affine" => &mut self.affine,
            "pad" => &mut self.pad,
            "wide-fma" => &mut self.wide_fma,
            "wgmma-hopper" => &mut self.wgmma,
            "resident" => &mut self.resident,
            "devgen" => &mut self.devgen,
            "retain-delta" => &mut self.retain_delta,
            "eager-init" => &mut self.eager_init,
            "arena" => &mut self.arena,
            "raster" => &mut self.raster,
            "l2-gate" => &mut self.l2_gate,
            "l2-window" => &mut self.l2_window,
            "retain-resident" => &mut self.retain_resident,
            "delta-h2d" => &mut self.delta_h2d,
            "lane-stride" => &mut self.lane_stride,
            "rowfold" => &mut self.rowfold,
            "rowfold-layout" => &mut self.rowfold_layout,
            "epilogue" => &mut self.epilogue,
            "devmap" => &mut self.devmap,
            "cell-law" => &mut self.cell_law,
            "streams" => &mut self.streams,
            "stream-gate" => &mut self.stream_gate,
            "pinned" => &mut self.pinned,
            "pdl" => &mut self.pdl,
            "pdl-early" => &mut self.pdl_early,
            "row-fit" => &mut self.row_fit,
            "wave-gate" => &mut self.wave_gate,
            "grid-share" => &mut self.grid_share,
            "panel-regs" => &mut self.panel_regs,
            "fill-rows" => &mut self.fill_rows,
            "cell-fill" => &mut self.cell_fill,
            "asyncstage" => &mut self.async_stage,
            "splitk" => &mut self.splitk,
            "splitk-atomic" => &mut self.splitk_atomic,
            "streamk" => &mut self.streamk,
            "hostopt" => &mut self.hostopt,
            "place" => &mut self.place,
            "by-ref" => &mut self.by_ref,
            "flow" => &mut self.flow,
            "window" => &mut self.window,
            "range-map" => &mut self.range_map,
            "uniform" => &mut self.uniform,
            "nest-cell" => &mut self.nest_cell,
            "tree-gate" => &mut self.tree_gate,
            "retained-fold" => &mut self.retained_fold,
            "preload" => &mut self.preload,
            "entry-arena" => &mut self.entry_arena,
            "async-h2d" => &mut self.async_h2d,
            "stage" => &mut self.stage,
            "chain-h2d" => &mut self.chain_h2d,
            "range-cell" => &mut self.range_cell,
            "scalar-fold" => &mut self.scalar_fold,
            "trip-batch" => &mut self.trip_batch,
            "cell-lanes" => &mut self.cell_lanes,
            "fold-lanes" => &mut self.fold_lanes,
            "radix-tile" => &mut self.radix_tile,
            "tf32-tile" => &mut self.tf32_tile,
            "row-split" => &mut self.row_split,
            "fill-wave" => &mut self.fill_wave,
            "plan-cost" => &mut self.plan_cost,
            "linear" => &mut self.linear,
            "fit-first" => &mut self.fit_first,
            "tf32-radix" => &mut self.tf32_radix,
            "kmajor-a" => &mut self.kmajor_a,
            "full-tile" => &mut self.full_tile,
            "async-depth" => &mut self.async_depth,
            "bank-plan" => &mut self.bank_plan,
            "row-block" => &mut self.row_block,
            "zero-fold" => &mut self.zero_fold,
            "row-stage" => &mut self.row_stage,
            "horizontal" => &mut self.horizontal,
            "calls" => &mut self.calls,
            "row-slices" => &mut self.row_slices,
            "cond-bounds" => &mut self.cond_bounds,
            "carry-made" => &mut self.carry_made,
            "host-par" => &mut self.host_par,
            "tile-post" => &mut self.tile_post,
            "row-tree" => &mut self.row_tree,
            "piece-delta" => &mut self.piece_delta,
            "head-group" => &mut self.head_group,
            "exp-hw" => &mut self.exp_hw,
            "exp-rescale" => &mut self.exp_rescale,
            "narrow-tile" => &mut self.narrow_tile,
            "streamk-fill" => &mut self.streamk_fill,
            "k-fill" => &mut self.k_fill,
            "uniform-base" => &mut self.uniform_base,
            "carry-take" => &mut self.carry_take,
            "slice-launches" => &mut self.slice_launches,
            "unchanged" => &mut self.unchanged,
            "retain-cell" => &mut self.retain_cell,
            "retain-pieces" => &mut self.retain_pieces,
            "retain-same" => &mut self.retain_same,
            "cone-span" => &mut self.cone_span,
            "arena-fit" => &mut self.arena_fit,
            _ => return None,
        })
    }

    pub fn get(mut self, name: &str) -> bool {
        self.rung(name).is_some_and(|r| *r)
    }

    /// `(self ∪ on) ∖ off`.
    pub fn with(mut self, arms: GpuArms) -> Self {
        for n in Self::NAMES {
            let r = self.rung(n).expect("a listed name");
            *r = (*r || arms.on.get(n)) && !arms.off.get(n);
        }
        self
    }
}

/// Measurement arms over the profile's rung table (S53): a rung in `on` is
/// forced on, one in `off` forced off. Both empty by default — the shipped path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuArms {
    pub on: GpuRungs,
    pub off: GpuRungs,
    /// S62 measurement override (`emit --splitk=<S>`, the `MovePanel::Force`
    /// precedent): where `splitk` or `splitk-atomic` fires, partition into `S`
    /// instead of the derived count, when `S` cuts `k` into whole units.
    /// 0 = derive.
    pub splitk: u64,
    /// plan-cost `--plan-cands`: one `; mapal.cand` comment line per tile
    /// site and candidate plan on the device unit, beside its `; mapal.plan`
    /// line (the fit harness's features). Comments only: no kernel changes.
    pub cands: bool,
    /// The emission's order is free (ADR-0040 D1), set from
    /// [`EmitOpts::order`] by [`emit_detailed`]: a device-only site may take
    /// the fused step the bound admits (`radix-tile`).
    pub free: bool,
    /// per-site-choice `--cands=N` (B1): the most candidates one site's table
    /// holds, the rules' plan first. 1 = the rules' launch alone (the OFF arm:
    /// the emission before candidates, byte for byte).
    pub site_cands: u64,
    /// per-site-choice: every split kernel takes the full register tile where
    /// it cuts `k / S` whole — `plan-cost`'s argmin among full-tile plans, one
    /// site candidate (`profile::CHOICE_AXES`). Never a flag.
    pub full: bool,
    /// fold-algebra: the step algebra's product–accumulate pairs fuse where the
    /// face contracts — `contract` on the host's (and device cells') pairs, and
    /// every tile site's step fused on the device whatever its lane stride.
    /// `false` (`--no-fma-pairs`) is the OFF arm: separate generic steps and the
    /// unit-lane / radix rule. One switch for both: the host oracle and the
    /// device realize the same arithmetic.
    pub pairs: bool,
    /// fold-algebra: a fold step the step algebra lets regroup takes `reassoc`
    /// where the order is free (lane partials + a tree). `false`
    /// (`--no-fold-regroup`) is the OFF arm.
    pub regroup: bool,
    /// compile-time-graph A: a row-block group `unit_plan` proves closed (tile
    /// sites with the folds and cells reading their rows) runs as one unit
    /// kernel of `b` rows per block when it fits the part, its rung
    /// ([`GpuRungs::row_block`]) is on and the plan model prices it below the
    /// split launches. `false` (`--no-row-block`) is the OFF arm: the emission
    /// before it, byte for byte.
    pub row_block: bool,
    /// Measurement override (`--row-block=<b>`, the `--splitk` precedent): a
    /// row-block group that fits takes `b` rows a unit, the rung table and
    /// the cost test aside. 0 = derive.
    pub row_block_b: u64,
    /// Measurement override (`--row-stage=<kc>x<stages>p<passes>`, the
    /// `--row-block` precedent): every staged tile of a row-block unit takes
    /// this panel depth, buffers and column passes where they fit, the
    /// plan model's pick aside. `None` = derive.
    pub row_stage_force: Option<(u64, u64, u64)>,
    /// Measurement override (`--hz=force`, the `--row-block` precedent): a
    /// `horizontal` group takes every independent adjacent launch whose
    /// arguments, block and shared memory one launch holds, the occupancy and
    /// wave tests aside.
    pub hz_force: bool,
    /// Measurement override (`--calls=force`, the `--hz=force` precedent): a
    /// `calls` chain forms whatever its prices (the fit tests stay).
    pub calls_force: bool,
    /// Measurement override (`--row-slices=<s>`, the `--row-block`
    /// precedent): a flow unit the slicing admits takes `s` blocks a row,
    /// the prices aside. 0 = derive; 1 = one block a row.
    pub row_slices_s: u64,
}

/// [`GpuArms::site_cands`]'s default (design B1).
pub const SITE_CANDS: u64 = 8;

impl Default for GpuArms {
    fn default() -> Self {
        Self {
            on: GpuRungs::default(),
            off: GpuRungs::default(),
            splitk: 0,
            cands: false,
            free: false,
            site_cands: SITE_CANDS,
            full: false,
            pairs: true,
            regroup: true,
            row_block: true,
            row_block_b: 0,
            row_stage_force: None,
            hz_force: false,
            calls_force: false,
            row_slices_s: 0,
        }
    }
}

/// A reduced-precision face — ADR-0032 D1's lattice below `contract`. Unlike
/// every rung, it **changes values**, by the declared bound of its format, and
/// is what admits the tensor-core realizations: they cannot reproduce an f32
/// fold bit for bit. Interim form of the type-level contract (the `--contract`
/// precedent); the oracle check for it is a bound, not bit-equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduced {
    /// f32 storage, operands rounded to TF32 (10-bit mantissa, `cvt.rna`),
    /// f32 accumulation — `mma.sync.m16n8k8.tf32` on sm_80+.
    Tf32,
}

/// The order a tile site's `k` fold may be evaluated in (ADR-0040 D1): the
/// second axis of the float face, beside the steps ([`EmitOpts::contract`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Increasing `k`, one chain per cell: the interpreter's left fold.
    Sequential,
    /// ADR-0040 D3: `S` contiguous partial folds added in partition order,
    /// `S` the profile's; bits a function of `(site, profile)`, replayed by
    /// the interpreter from [`Emission::orders`]. The test/debug face.
    Fixed,
    /// Unspecified (the default): any realization within ADR-0040 D2's bound —
    /// atomic split, stream-K, the fixed split, sequential — per part by
    /// measurement. Run-to-run bits may differ.
    Free,
}

/// One split fold the emitter compiled under [`Order::Fixed`]: the `Fold`
/// morphism `fold` of function `func` (a tile site's map body) folds in `s`
/// contiguous partitions, added in partition order (ADR-0040 D3). The
/// interpreter replays exactly these (`mapal_interp::run_ordered`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldOrder {
    pub func: FuncId,
    pub fold: MorphismId,
    pub s: u64,
}

/// Emission options. [`emit`] delegates to these product defaults.
#[derive(Clone, Copy, Debug)]
pub struct EmitOpts {
    /// Bracket `mapal_main` with the mapal-rt compute timer.
    pub perf_timing: bool,
    /// Tile recognized matmul-shaped map sites.
    pub tiling: bool,
    /// Pack tiled two-dimensional right-hand operands.
    pub packing: bool,
    /// Steps (ADR-0040 D1): single-rounding `fma` per fold step. Default
    /// `true` (the default face); `false` is the separate `fmul` + `fadd` of
    /// the `exact` and `fixed` faces.
    pub contract: bool,
    /// Order of a tile site's `k` fold (ADR-0040 D1); default [`Order::Free`].
    pub order: Order,
    /// Split deep packed tile sites into k-panels — the OpenBLAS (jc, kc, ic)
    /// nest with an A-panel pack and partial sums parked in `out`.
    ///
    /// **Default OFF: measured a 3× LOSS** on M4 Pro at 1024 f32 (S29 —
    /// `fma` 59.8 ms on / 19.8 ms off; the parking traffic outweighs the A
    /// re-read it removes at this size and cache hierarchy).
    ///
    /// **S42: SWEPT on the box, and it loses at every depth — the last argument
    /// for this lever is gone.** It was kept because it "was designed against
    /// BOX-scale traffic (16 GB of A re-reads at 4096 on zen3) where it has not
    /// yet been measured".
    ///
    /// Measured on an i9-14900F (24C/32T, 48 KB L1d, 2 MB per-core L2, 36 MB L3,
    /// AVX2, governor `performance`), N=4096, values identical to the Mac's at
    /// every depth. For `zen3` at f32 the chain is `lanes 8 → tile_j 32 → nc
    /// 1024`, so `tile_kc = l2_bytes / 8192`; the sweep varies `l2_bytes`:
    ///
    /// | depth | blocks | 1 thread | threaded (32) |
    /// | ---: | ---: | ---: | ---: |
    /// | unblocked | 1 | **926.759 ms** | **112.919 ms** |
    /// | 4096 | 1 | 0.978× (gate closed ⇒ *is* unblocked) | 1.000× |
    /// | 2048 | 2 | 0.548× | 0.595× |
    /// | 1024 | 4 | 0.557× | 0.759× |
    /// | 512 | 8 | 0.551× | **0.771×** best |
    /// | 256 | 16 | 0.560× | 0.767× |
    /// | 64 | 64 | 0.454× | 0.668× |
    ///
    /// **A step function, not a curve**: any blocking at all costs ~1.8× at one
    /// thread and ~1.3× at best threaded, at *every* depth from 2 to 64 blocks.
    /// Unlike the SME k-panel — where one wrong depth made the technique look
    /// worthless (`func/sme.rs`) — here there is no optimum to find.
    ///
    /// ⇒ **`kc_nest` has now lost on every machine available, swept rather than
    /// spot-checked.** "Unmeasured on the box" is no longer a reason to keep it.
    /// Either a machine justifies it in writing, or delete the lever.
    ///
    /// A pure performance tailor in ADR-0032's sense: bit-exact either way, which
    /// the differential suite enforces — and which the cross-machine run
    /// independently confirmed, ARM and x86 printing identical values.
    pub kc_nest: bool,
    /// **The move-panel traversal — S44's rung, S45's deduction.** Visits a bulk
    /// `map`'s iteration space in `b × b` tiles of a `w`-wide 2-D geometry
    /// instead of in linear order, when that pays.
    ///
    /// **S44 shipped this as `--move-panel=W:B`: two numbers typed by a human.**
    /// `W` was program geometry the emitter could not see (`tile_site` requires a
    /// fold in the map body, and a transpose has none) and `B` was a swept
    /// machine constant. S45 deleted both: `W` comes from
    /// [`mapal_ir::MoveSite`], a fold-less move-site record; the cache geometry
    /// comes from [`crate::profile::L1d`], detected; and firing and `B` are
    /// arithmetic over the two in [`crate::profile::TargetProfile::move_block`].
    ///
    /// [`MovePanel::Deduce`] is the default and the shipped path. The other two
    /// exist so the transform can still be **measured** — an A/B needs an OFF arm
    /// of the same binary, and a B-sweep needs to force a block — and they carry
    /// no machine constant into any emitted default.
    ///
    /// A pure ADR-0032 D4 performance tailor: the transform is a **permutation of
    /// the loop counter**, so every element is still visited exactly once and
    /// values are bit-identical, which the differential suite enforces.
    pub move_panel: MovePanel,
    /// The machine facts the emitter tiles against, selected **by name**
    /// (plan-s31-target-profiles): `generic` (the pre-S46 literals, byte-
    /// identical on every host), `apple-m`, `apple-m4-sme`, `zen3`,
    /// `raptorlake`, `cuda-ada`.
    ///
    /// **The default is `native` (S46, ratified ADR-0038): probe the host, and
    /// fall back to `generic` when a fact cannot be read.** Default emission is
    /// therefore machine-dependent *by design* — anything that needs
    /// reproducible emitted text across machines (structural tests, artifact
    /// hashes) pins a profile by name, and [`emit_detailed`] reports which
    /// profile was resolved.
    ///
    /// A pure ADR-0032 D4 performance tailor: every profile field is
    /// value-invariant, which the differential suite enforces by running under
    /// a non-default profile.
    pub target: &'static str,
    /// S62 `hostopt`: on a GPU `target`, the CPU profile everything outside a
    /// device site is emitted against — resolved like `target` (`native` by
    /// default, so default emission is machine-dependent here too: tests pin
    /// it). A CPU `target` ignores it; the `emit` CLI rejects the pair.
    pub host: &'static str,
    /// Measurement arms over the GPU rung table; see [`GpuArms`].
    pub gpu: GpuArms,
    /// ADR-0034 D1: `--geom=<knob>=<value>` over the GPU part's geometry table
    /// ([`profile::Geom`]); all `None` = the table.
    pub geom: profile::GeomOver,
    /// plan-cost: `--cost=<rate>=<value>` over the GPU part's plan rates
    /// ([`profile::DeviceCost::NAMES`]); all `None` = the profile's.
    pub cost: profile::CostOver,
}

impl Default for EmitOpts {
    fn default() -> Self {
        Self {
            perf_timing: false,
            tiling: true,
            packing: true,
            // ADR-0040 (Sapir, 2026-10-23): the default face is full
            // precision, fma steps, order unspecified, bounded.
            contract: true,
            order: Order::Free,
            kc_nest: false,
            move_panel: MovePanel::Deduce,
            // S46 (Sapir): the DEFAULT detects the machine. Every rung that
            // needs machine facts asks the profile for them, so a default build
            // now gets the cache-aware ones instead of silently declining.
            // `native` falls back to `generic` when detection fails, and tests
            // that must be reproducible across machines pin `generic` by name.
            target: "native",
            host: "native",
            gpu: GpuArms::default(),
            geom: Default::default(),
            cost: [None; profile::DeviceCost::NAMES.len()],
        }
    }
}

impl EmitOpts {
    /// The `exact` face (`--exact`): separate `fmul` + `fadd`, sequential —
    /// the interpreter's bits. What every bit-for-bit oracle check pins.
    pub fn exact() -> Self {
        Self {
            contract: false,
            order: Order::Sequential,
            ..Self::default()
        }
    }
}

/// One emitted translation unit **plus the profile it was emitted against**
/// (ADR-0038 D2). The default target is `native`, so the text alone does not
/// say which machine facts shaped it — `profile` is the resolved profile's
/// name (`"native"` when detection succeeded, `"generic"` on fallback or by
/// name, or the named profile), and artifact identity is
/// `(source, EmitOpts, profile)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emission {
    /// The `.ll` text — the translation unit itself (ADR-0020).
    pub text: String,
    /// The **device** translation unit (`nvptx64`), when the profile is a GPU
    /// and some tile site offloads: `llc` turns it into the PTX the runtime
    /// loads (`MAPAL_PTX`). `None` on every CPU profile.
    pub device: Option<String>,
    /// The resolved [`TargetProfile`]'s name.
    pub profile: &'static str,
    /// The split folds compiled under [`Order::Fixed`] (ADR-0040 D3), for the
    /// interpreter to replay. Empty on every other order.
    pub orders: Vec<FoldOrder>,
}

/// Emit one LLVM translation unit for `ir` (ADR-0020 §1). `Ok` is the `.ll`
/// text; `Err(Unsupported)` for a non-canonical loop (L3). Uses
/// [`EmitOpts::default()`] — target `native` (S46); [`emit_detailed`] reports
/// which profile that resolved to on this host.
pub fn emit(ir: &CategoryIr) -> Result<String, EmitError> {
    emit_with_opts(ir, &EmitOpts::default())
}

/// [`emit`] with options (see [`EmitOpts`]); the text half of [`emit_detailed`].
pub fn emit_with_opts(ir: &CategoryIr, opts: &EmitOpts) -> Result<String, EmitError> {
    Ok(emit_detailed(ir, opts)?.text)
}

/// [`emit_with_opts`] with the resolved profile named in the artifact
/// (ADR-0038 D2). This is the honest signature of the transform: the profile
/// is an *input* — ambient when the target is `native`, explicit otherwise —
/// and naming it in the output is what makes the input visible.
pub fn emit_detailed(ir: &CategoryIr, opts: &EmitOpts) -> Result<Emission, EmitError> {
    // onto-device `cond-bounds` off: the facts rebuilt under plain intervals
    // (a fact arm, `mapal_ir::FactArms` — every fact derived afresh).
    let arms = mapal_ir::FactArms {
        cond_bounds: !opts.gpu.off.cond_bounds,
        row_tree: !opts.gpu.off.row_tree,
        piece_delta: !opts.gpu.off.piece_delta,
        head_group: !opts.gpu.off.head_group,
        exp_rescale: !opts.gpu.off.exp_rescale,
        uniform_base: !opts.gpu.off.uniform_base,
        unchanged: !opts.gpu.off.unchanged,
    };
    if arms != ir.fact_arms() {
        return emit_detailed(&ir.refacted(arms), opts);
    }
    // Machine facts, by name (plan-s31-target-profiles rule 3). An unknown name
    // is an error, never a silent fall back to `generic` — a typo that quietly
    // emits the default profile's numbers is the exact failure the table exists
    // to remove.
    let unknown = |name: &str| {
        EmitError::Internal(format!(
            "unknown target profile `{name}`; known: {}",
            profile::names()
        ))
    };
    let mut profile = profile::resolve(opts.target).ok_or_else(|| unknown(opts.target))?;
    // ADR-0034 D1: a measurement arm's geometry overrides, on a copy of the part.
    if opts.geom.iter().any(Option::is_some)
        && let profile::Machine::Gpu(g) = profile.machine
    {
        let mut p = *profile;
        p.machine = profile::Machine::Gpu(profile::Gpu {
            geom: g.geom.with(&opts.geom),
            ..g
        });
        profile = Box::leak(Box::new(p));
    }
    // plan-cost: a fit arm's rates, on a copy of the part.
    if opts.cost.iter().any(Option::is_some)
        && let profile::Machine::Gpu(g) = profile.machine
    {
        let mut p = *profile;
        p.machine = profile::Machine::Gpu(g.with_cost(&opts.cost));
        profile = Box::leak(Box::new(p));
    }
    if profile.gpu().is_none() {
        return emit_module(ir, opts, profile);
    }
    // ADR-0041 D4: a tensor core's k-step sums in the hardware's order, so a
    // `tf32` product takes it only where the order is free; a sequential or
    // fixed face realizes it as f32 (a member of every set): the FFMA tile.
    let mut face = *opts;
    face.gpu.free = opts.order == Order::Free;
    if !face.gpu.free {
        face.gpu.off.tf32_tile = true;
    }
    let opts = &face;
    // S62 `hostopt`: everything outside a device site is emitted against the
    // host's CPU facts; off, against `generic`'s — the S52–S54 GPU build.
    let rungs = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
    if !rungs.hostopt {
        return emit_module(ir, opts, profile).map(|em| host_line(ir, opts, profile, em));
    }
    let host = profile::resolve(opts.host).ok_or_else(|| unknown(opts.host))?;
    if host.gpu().is_some() {
        return Err(EmitError::Internal(format!(
            "host profile `{}` is a GPU profile",
            host.name
        )));
    }
    let mut both = profile::with_host(profile, host);
    // `place` off: no measured cost, so every capable site is a device site.
    // (A site admitting `tf32` is placed by the same roofline: the host's f32
    // realization is admitted too, ADR-0041 D6.)
    if !rungs.place
        && let profile::Machine::Gpu(g) = &mut both.machine
    {
        // plan-cost: the plan rates stay — `place` decides where a site
        // runs, not how (a zero placement rate is "no measured cost").
        g.cost = g
            .cost
            .map(|d| profile::DeviceCost {
                flops_per_ns: 0,
                bytes_per_ns: 0,
                launch_ns: 0,
                ..d
            })
            .filter(|d| d.issue_ns > 0);
    }
    let both: &'static TargetProfile = Box::leak(Box::new(both));
    let name: &'static str = format!("{}+{}", profile.name, host.name).leak();
    let em = emit_module(ir, opts, both)?;
    // A GPU build of a program whose every site stays on the host IS the
    // host's CPU build: parallel entry, the whole CPU ladder, no runtime.
    // ponytail: per site, not per program — a one-shot program pays the
    // device's start (context, module) once; gating on it needs the host cost
    // of what `devgen` builds on the device, which no profile fact carries.
    let em = if em.device.is_none() {
        emit_module(ir, opts, host)?
    } else {
        em
    };
    Ok(host_line(
        ir,
        opts,
        both,
        Emission {
            profile: name,
            ..em
        },
    ))
}

/// S64 evidence pass: a GPU build's host module ends with one `; mapal.host`
/// comment — the profiles it was emitted against, its face, and how many
/// capable tile sites `place` put on the device and kept on the host
/// (`func/gpu.rs:placement`, the walk's own answer). `benches/gpu/evidence.py`
/// reads it; a comment, so the object is unchanged.
fn host_line(
    ir: &CategoryIr,
    opts: &EmitOpts,
    profile: &'static TargetProfile,
    mut em: Emission,
) -> Emission {
    // A build with no device unit IS the host's CPU build, byte for byte.
    if em.device.is_none() {
        return em;
    }
    let rungs = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
    let (mut dev, mut host) = (0, 0);
    for (f, _) in ir.funcs() {
        let plan = crate::func::gpu::tile_plan(ir, f, profile, opts.contract, opts.gpu);
        let placed =
            crate::func::gpu::placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
        for (m, s) in plan.sites.iter() {
            if crate::func::gpu::gpu_site(s, profile, rungs.window) {
                if placed.contains_key(m) {
                    dev += 1;
                } else {
                    host += 1;
                }
            }
        }
    }
    em.text.push_str(&format!(
        "; mapal.host profile={} fma={} order={:?} device_sites={dev} host_sites={host}\n",
        em.profile,
        u8::from(opts.contract),
        opts.order
    ));
    em
}

/// One module against one resolved profile — a CPU profile, or a GPU profile
/// whose CPU-shaped fields describe the host ([`emit_detailed`]).
fn emit_module(
    ir: &CategoryIr,
    opts: &EmitOpts,
    profile: &'static TargetProfile,
) -> Result<Emission, EmitError> {
    // Capability gate (L3): canonical loops only — every SCC has exactly one
    // merge and a well-formed `loop_plan`. Same predicate as rewrite's
    // `is_canonical` / interp M1 (BL6/BL7).
    for (f, _) in ir.funcs() {
        for lscc in ir.loop_structure(f) {
            let canonical = lscc.merges.len() == 1 && ir.loop_plan(f, lscc.merges[0]).is_some();
            if !canonical {
                let loc = lscc
                    .merges
                    .first()
                    .and_then(|&m| ir.object(m))
                    .map(|o| o.loc)
                    .unwrap_or(SourceLoc { start: 0, end: 0 });
                return Err(EmitError::Unsupported {
                    feature: "nested loops".into(),
                    loc,
                });
            }
        }
    }

    let strings = collect_str_globals(ir);

    // Suggestions #7's attribute-capability pre-pass (func/mod.rs `FnAttrs`):
    // clean fns get `readonly nounwind` (+ `willreturn`), unclean fns nothing.
    let attrs = FnAttrs::analyze(ir);

    // Deterministic function names: entry → @mapal_main, others → @fn{ordinal}.
    let mut fnames: SecondaryMap<mapal_ir::FuncId, String> = SecondaryMap::new();
    let entry = ir.entry();
    for (ord, (id, _)) in ir.funcs().enumerate() {
        let name = if id == entry {
            "mapal_main".to_string()
        } else {
            format!("fn{ord}")
        };
        fnames.insert(id, name);
    }

    // ponytail: malformed/cyclic plans stay sequential; delete this fallback
    // when path_plan's total-DAG contract is enforced before backend entry.
    let path_plan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ir.path_plan(entry)))
        .ok()
        .filter(path_plan_is_acyclic);
    // On a GPU profile the device is the parallelism: the host runs the program
    // around the offloaded sites sequentially (S62: a GPU build with no device
    // site is re-emitted as the host's CPU build, `emit_detailed`, so it takes
    // this parallel entry). ponytail: host threads alongside the device is
    // co-execution (ADR-0035) — add when a program needs both.
    let parallel = profile.gpu().is_none()
        && path_plan
            .as_ref()
            .is_some_and(|plan| !plan.is_single_path());
    let body_sites = if parallel {
        parallel_body_sites(ir, entry, path_plan.as_ref().expect("parallel plan"))
    } else {
        SecondaryMap::new()
    };
    let prefetch = opts.tiling
        && opts.packing
        && ir.funcs().any(|(f, _)| {
            let plan = ir.tile_plan(f);
            let r = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
            let dev = crate::func::gpu::placement(ir, f, &plan, profile, r.window, r.carry_made);
            plan.sites
                .iter()
                .any(|(m, site)| !dev.contains_key(m) && packing_site(site, profile, opts.contract))
        });

    // Functions first: the arena declarations are gated on the emitted text
    // (below), so the bodies have to exist before the header is assembled.
    let mut funcs = String::new();
    // `devgen` (S53): the producers each function computes on the device, and
    // every function's emitted text — a device-built producer's body is the
    // host's own, appended to the device unit.
    let mut gens: Vec<(u64, mapal_ir::FuncId, String)> = Vec::new();
    // `epilogue` (S54): the grouped sites, and the bodies their kernels call.
    let mut epis: Vec<crate::func::gpu::EpiSite> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut epi_calls: Vec<(
        String,
        mapal_ir::FuncId,
        Vec<u32>,
        Option<mapal_ir::MorphismId>,
    )> = Vec::new();
    // `devmap` (S54): each function's cell kernels and the bodies they call.
    let mut cells: Vec<crate::func::gpu::CellKernel> = Vec::new();
    // S65: the row folds each function runs on the device.
    let mut folds: Vec<mapal_ir::RowFoldSite> = Vec::new();
    // `flow`: each function's unit kernels.
    let mut units: Vec<crate::func::flow::UnitKernel> = Vec::new();
    let mut texts: SecondaryMap<mapal_ir::FuncId, String> = SecondaryMap::new();
    let mut orders: Vec<FoldOrder> = Vec::new();
    for (id, _) in ir.funcs() {
        if id == entry && parallel {
            funcs.push_str(&FnEmit::emit_parallel(
                ir,
                id,
                &fnames,
                &strings,
                &attrs,
                path_plan.as_ref().expect("parallel plan"),
                opts.perf_timing,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.order,
                opts.kc_nest,
                opts.move_panel,
                profile,
                opts.gpu,
            ));
        } else {
            let mut fe = FnEmit::new(
                ir,
                id,
                &fnames,
                &strings,
                &attrs,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.order,
                opts.kc_nest,
                opts.move_panel,
                profile,
            );
            if id == entry {
                fe.set_perf_timing(opts.perf_timing);
                // S54 `streams`: the device sites' independence (the host
                // stays sequential on a GPU profile).
                fe.set_path_plan(path_plan.as_ref().filter(|_| profile.gpu().is_some()));
            }
            fe.set_gpu_arms(opts.gpu);
            if let Some(&site) = body_sites.get(id) {
                fe.set_task_body_site(site);
            }
            gens.extend(fe.gen_kernels());
            let (es, cs) = fe.epi_sites();
            epis.extend(es);
            epi_calls.extend(cs);
            cells.extend(fe.cell_kernels());
            folds.extend(fe.rowfold_kernels());
            units.extend(fe.unit_kernels());
            let (text, os) = fe.emit_ordered();
            orders.extend(os);
            funcs.push_str(&text);
            texts.insert(id, text);
        }
        funcs.push('\n');
    }

    let mut out = String::new();
    out.push_str("; mapal-backend-llvm emitted module\n");
    out.push_str(RT_DECLS);
    if prefetch {
        out.push_str(PREFETCH_DECL);
    }
    if opts.perf_timing {
        out.push_str(PERF_DECLS);
    }
    // onto-device `host-par`: a GPU build's walk dispatches split tasks too.
    if parallel || funcs.contains("@mapal_par_begin(") {
        out.push_str(PAR_DECLS);
    }
    // ponytail: gate the arena declarations on the emitted call itself, not on
    // a re-derived predicate — the call IS the requirement, so the two cannot
    // drift the way a `prefetch`-style pre-pass could. (PAR/PERF must gate on a
    // predicate: they are decided before any body exists.)
    if funcs.contains("@mapal_rt_alloc") {
        out.push_str(HEAP_DECLS);
    }
    // fold-algebra: an explicit-face pair is one `llvm.fma` (gated on the call).
    for (call, decl) in [
        (
            "@llvm.fma.f32(",
            "declare float @llvm.fma.f32(float, float, float)\n",
        ),
        (
            "@llvm.fma.f64(",
            "declare double @llvm.fma.f64(double, double, double)\n",
        ),
    ] {
        if funcs.contains(call) {
            out.push_str(decl);
        }
    }
    // ADR-0042: the host's `exp` definitions, for the widths a body calls.
    {
        let hw = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu).exp_hw;
        let free = opts.order == Order::Free;
        let real = crate::exp::choose(free, opts.contract, hw, None);
        out.push_str(&crate::exp::defs(&funcs, real));
    }
    // ADR-0043: the host's narrow roundings, for the formats a body rounds to.
    out.push_str(&crate::narrow::defs(&funcs, false));
    // Same rule for the SME leg: the emitted call IS the requirement, so the
    // three intrinsic declarations and the one streaming panel kernel appear
    // exactly when some tile site took the SME rung. A module without one is
    // byte-identical to what it was before the rung existed.
    // Two kernels, detected independently: `@mapal_sme_panel(` cannot match
    // `@mapal_sme_panel_acc(` because of the paren, so the KC variant appears
    // only where the KC nest actually called it.
    // A module talks to the GPU runtime when a site calls it: the S52 round
    // trip (`mapal_gpu_tile`) or `resident`'s split entries (`mapal_gpu_alloc`
    // precedes every one of them).
    // `arena` (S54): a function whose every device object sits in its arena
    // calls `mapal_gpu_arena` and no `mapal_gpu_alloc`.
    let arena = funcs.contains("@mapal_gpu_arena(");
    let resident = funcs.contains("@mapal_gpu_alloc(") || arena;
    let round_trip = funcs.contains("@mapal_gpu_tile(");
    let gpu_host = resident || round_trip;
    if gpu_host {
        out.push_str(crate::func::gpu::GPU_INIT_DECL);
        if round_trip {
            out.push_str(crate::func::gpu::GPU_TILE_DECL);
        }
        if funcs.contains(crate::func::gpu::GPU_SYNC) {
            out.push_str(crate::func::gpu::GPU_SYNC_DECL);
        }
    } else {
        // No device work in the module: the clock fence's sync waits for nothing.
        funcs = funcs.replace(&format!("  {}\n", crate::func::gpu::GPU_SYNC), "");
    }
    if resident {
        out.push_str(crate::func::gpu::RESIDENT_DECLS);
        if funcs.contains("@mapal_gpu_launch_l2(") {
            out.push_str(crate::func::gpu::GPU_L2_DECL);
        }
        // `streams`, `pinned` (S54): each entry declared only when called.
        for (call, decl) in crate::func::gpu::S54_DECLS {
            if funcs.contains(call) {
                out.push_str(decl);
            }
        }
    }
    if arena {
        out.push_str(crate::func::gpu::ARENA_DECL);
    }
    // `epilogue`, `devmap` and `streams` launch over n pointers: one declaration.
    if funcs.contains("@mapal_gpu_launchv(") {
        out.push_str(crate::func::gpu::LAUNCHV_DECL);
    }
    // kernel-merging `calls`: a persistent kernel's cooperative launch.
    if funcs.contains("@mapal_gpu_launch_coop(") {
        out.push_str(crate::func::gpu::COOP_DECL);
    }
    // per-site-choice: a site with candidates launches its table.
    if funcs.contains("@mapal_gpu_site(") {
        out.push_str(crate::func::gpu::SITE_DECL);
    }
    let sme_store = funcs.contains("@mapal_sme_panel(");
    let sme_acc = funcs.contains("@mapal_sme_panel_acc(");
    let sme = sme_store || sme_acc;
    if sme {
        out.push_str(SME_DECLS);
    }
    out.push('\n');

    let sg = emit_str_globals(&strings);
    if !sg.is_empty() {
        out.push_str(&sg);
        out.push('\n');
    }

    out.push_str(&funcs);
    if sme {
        // The panel side and the tile-block arrangement are the profile's,
        // never literals here: `svl_bytes / sizeof(f32)` and the square-most
        // split of the ZA tile count. The rung only fires at f32.
        let f32 = mapal_ir::Ty::f32();
        let expect = "the SME rung only fires on a profile with an SME unit";
        let t = profile.sme_tile_side(&f32).expect(expect);
        let (ti, tj) = profile.sme_block(&f32).expect(expect);
        if sme_store {
            out.push_str(&sme_panel(t, ti, tj, PanelWrite::Store));
        }
        if sme_acc {
            out.push_str(&sme_panel(t, ti, tj, PanelWrite::Accumulate));
        }
    }
    // The runtime's init is `main`'s first call exactly when a site calls the
    // runtime: the same gate as `GPU_INIT_DECL`.
    let prologue = if gpu_host {
        crate::func::gpu::init_call(
            crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu),
            // Streams are opened where a launch names one: `launchv` on stream 0
            // is also `epilogue`'s and `devmap`'s (S54 integration).
            funcs
                .split("@mapal_gpu_launchv(i64 ")
                .skip(1)
                .any(|t| t.split(", ").nth(1).is_some_and(|s| s != "i32 0")),
            crate::func::gpu::pdl(ir, profile, opts.contract, opts.gpu),
        )
    } else {
        String::new()
    };
    out.push_str(&emit_main_wrapper(ir, &prologue));
    // A device body's nested bodies are the host's text (generic-fusion
    // natural forms: a fold inside a cell), carried with the epilogues'.
    let nested: Vec<mapal_ir::FuncId> = gens
        .iter()
        .map(|g| g.1)
        .chain(epi_calls.iter().map(|e| e.1))
        .flat_map(|b| crate::func::resident::nested_bodies(ir, b))
        .collect();
    let gens: Vec<crate::func::gpu::GenKernel> = gens
        .into_iter()
        .map(|(gid, body, elem)| (gid, fnames[body].clone(), elem, texts[body].clone()))
        .collect();
    // An epilogue calls the host's body text, or its device variant: the
    // same emitter with the cell-read slots as one-cell buffers.
    let mut epi_bodies: Vec<(String, String)> = Vec::new();
    for nb in nested {
        if !epi_bodies.iter().any(|(c, _)| *c == fnames[nb]) {
            epi_bodies.push((fnames[nb].clone(), texts[nb].clone()));
        }
    }
    for (callee, body, slots, post) in epi_calls {
        if epi_bodies.iter().any(|(c, _)| *c == callee) {
            continue;
        }
        let text = if slots.is_empty() && post.is_none() {
            texts[body].clone()
        } else {
            let mut fe = FnEmit::new(
                ir,
                body,
                &fnames,
                &strings,
                &attrs,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.order,
                opts.kc_nest,
                opts.move_panel,
                profile,
            );
            fe.set_gpu_arms(opts.gpu);
            fe.set_cell_slots(slots);
            fe.set_post_fold(post);
            fe.emit()
        };
        epi_bodies.push((callee, text));
    }
    let device = opts
        .tiling
        .then(|| {
            crate::func::gpu::device_module(
                ir,
                profile,
                opts.contract,
                opts.gpu,
                opts.order,
                &gens,
                &epis,
                &epi_bodies,
                &cells,
                &folds,
                &units,
                &crate::func::gpu::launched_kernels(&out),
                &crate::func::horizontal::hz_specs(&out),
            )
        })
        .flatten();
    Ok(Emission {
        text: out,
        device,
        profile: profile.name,
        orders,
    })
}

fn path_plan_is_acyclic(plan: &PathPlan) -> bool {
    let mut remaining = plan
        .tasks
        .iter()
        .map(|task| task.deps.len())
        .collect::<Vec<_>>();
    let mut ready = remaining
        .iter()
        .enumerate()
        .filter_map(|(task, &deps)| (deps == 0).then_some(task))
        .collect::<Vec<_>>();
    let mut cursor = 0;
    while cursor < ready.len() {
        let before = ready[cursor];
        cursor += 1;
        for (after, task) in plan.tasks.iter().enumerate() {
            if task.deps.contains(&before) {
                remaining[after] -= 1;
                if remaining[after] == 0 {
                    ready.push(after);
                }
            }
        }
    }
    ready.len() == plan.tasks.len()
}

/// Map every Map/Fold body reached from a parallel entry site to that site's
/// topo position. Lowering mints body functions per site, so the map is
/// single-valued; nested bodies inherit the outer entry site's trap position.
fn parallel_body_sites(
    ir: &CategoryIr,
    entry: FuncId,
    plan: &PathPlan,
) -> SecondaryMap<FuncId, u32> {
    let topo = ir.topo_order(entry);
    let mut topo_pos: SecondaryMap<MorphismId, u32> = SecondaryMap::new();
    for (i, &m) in topo.iter().enumerate() {
        topo_pos.insert(m, i as u32);
    }

    let mut out = SecondaryMap::new();
    for task in &plan.tasks {
        let members: &[MorphismId] = match &task.kind {
            TaskKind::Split { site, .. } => std::slice::from_ref(site),
            TaskKind::Seq { morphisms } => morphisms,
        };
        for &m in members {
            if let Operation::Map { body, .. } | Operation::Fold { body, .. } =
                ir.morphism(m).expect("morphism resolves").op
            {
                mark_body_closure(ir, body, topo_pos[m], &mut out);
            }
        }
    }
    out
}

fn mark_body_closure(
    ir: &CategoryIr,
    body: FuncId,
    site_topo: u32,
    out: &mut SecondaryMap<FuncId, u32>,
) {
    if out.insert(body, site_topo).is_some() {
        return;
    }
    for &m in &ir.func(body).expect("body resolves").morphisms {
        if let Operation::Map { body, .. } | Operation::Fold { body, .. } =
            ir.morphism(m).expect("morphism resolves").op
        {
            mark_body_closure(ir, body, site_topo, out);
        }
    }
}

#[cfg(test)]
mod place_tests {
    //! S62 `place`, the mechanism: one predicate decides every reader of a
    //! site's placement. Synthetic costs (not a part's), so the verdicts are
    //! fixed whatever the calibration tables say.
    use super::*;
    use crate::profile::{DeviceCost, DispatchCost, Machine, RTX_4090, X86_64_V3, with_host};

    /// rtx-4090's device and x86-64-v3's host, with round costs: a launch is
    /// 20 µs, the device 20 Tflop/s; one host lane 50 Gflop/s.
    fn synth(place: bool) -> &'static TargetProfile {
        let mut p = with_host(&RTX_4090, &X86_64_V3);
        p.dispatch = Some(DispatchCost {
            flops_per_ns: 50,
            bytes_per_ns: 10,
            dispatch_ns: 10_000,
            lanes: 0,
        });
        if let Machine::Gpu(g) = &mut p.machine {
            g.cost = place.then_some(DeviceCost {
                flops_per_ns: 20_000,
                bytes_per_ns: 500,
                launch_ns: 20_000,
                ..crate::profile::NO_PLAN
            });
        }
        Box::leak(Box::new(p))
    }

    fn lower(src: &str) -> CategoryIr {
        let po = mapal_syntax::parse(src);
        assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
        mapal_rewrite::rewrite(mapal_lower::lower(src, &po.program).expect("lowers")).ir
    }

    fn law(name: &str, len: u64, mul: u64) -> String {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> {name};\n"
        )
    }

    fn site(out: &str, a: &str, b: &str, m: u64, n: u64, k: u64) -> String {
        format!(
            "    {mn} -> iota -> o_{out};\n    {k} -> iota -> k_{out};\n    o_{out} -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, k_{out}) -> fold {{ acc, k -> acc + {a}[i * {k} + k] * {b}[k * {n} + j] }} }} -> {out};\n",
            mn = m * n
        )
    }

    /// An 8³ chain (three sites) written first, then a 256³ chain, then a
    /// small site (8×8, k = 256) over the large chain's output.
    fn mixed() -> CategoryIr {
        let mut s = String::from("fn main() {\n");
        for (x, mul) in [("sa", 7), ("sb", 11), ("se", 5), ("sf", 3)] {
            s += &law(x, 64, mul);
        }
        for (x, mul) in [("a", 7), ("b", 11), ("e", 5)] {
            s += &law(x, 65536, mul);
        }
        s += &law("w", 2048, 13);
        s += &site("c1", "sa", "sb", 8, 8, 8);
        s += &site("c2", "c1", "se", 8, 8, 8);
        s += &site("c3", "c2", "sf", 8, 8, 8);
        s += &site("c", "a", "b", 256, 256, 256);
        s += "    c -> map { v -> v * 0.5 } -> h;\n";
        s += &site("d", "h", "e", 256, 256, 256);
        s += &site("g", "d", "w", 8, 8, 256);
        s += "    c3[63] -> println;\n    d[65535] -> println;\n    g[63] -> println;\n}\n";
        lower(&s)
    }

    /// The rules' plan (`--cands=1`): placement is what this checks (the
    /// row-block arm off: its chain of 8×8 sites is `row_block.rs`'s; the
    /// `horizontal` arm off: the three independent 8³ sites would launch as
    /// one, `horizontal.rs`'s).
    fn emit_on(ir: &CategoryIr, place: bool) -> Emission {
        let mut gpu = GpuArms {
            site_cands: 1,
            row_block: false,
            ..GpuArms::default()
        };
        *gpu.off.rung("horizontal").expect("a rung") = true;
        *gpu.off.rung("calls").expect("a rung") = true;
        let opts = EmitOpts {
            gpu,
            ..EmitOpts::exact()
        };
        emit_module(ir, &opts, synth(place)).expect("emits")
    }

    fn count(text: &str, pat: &str) -> usize {
        text.matches(pat).count()
    }

    /// The three 8³ sites move to the host — no launch, no kernel, no device
    /// generation of their operands — while the 256³ chain and the small site
    /// it feeds stay on the device.
    #[test]
    fn place_is_one_predicate() {
        let ir = mixed();
        let (on, off) = (emit_on(&ir, true), emit_on(&ir, false));
        let (don, doff) = (
            on.device.expect("a device chain"),
            off.device.expect("device"),
        );
        let kernels = |d: &str| count(d, "define ptx_kernel void @mapal_tile_");
        // The host walk, the residency plan and the device unit agree: the
        // three small sites (one kernel: the same shape) lost their kernel
        // AND their launches together.
        assert_eq!(kernels(&doff) - kernels(&don), 1, "{don}");
        let launches = |t: &str| count(t, "call void @mapal_gpu_launch");
        assert_eq!(launches(&off.text) - launches(&on.text), 3);
        // Left on the device: `c` (with `h` in its store), `d`, and `g` — an
        // 8×8 site its roofline puts on the host, kept on the device because
        // its operand `d` is device-produced (`fed`): `d` never crosses whole.
        assert_eq!(launches(&on.text), 3, "{}", on.text);
        // Their closed-form operands are no longer built on the device.
        let gens = |d: &str| count(d, "define ptx_kernel void @mapal_gen_");
        assert_eq!(gens(&doff) - gens(&don), 4);
    }

    /// Without measured costs (`place` off, or a part without a `cost` row)
    /// the placement is S52–S54's: every capable site.
    #[test]
    fn without_costs_every_capable_site_is_placed() {
        let ir = mixed();
        let mut gpu = GpuArms {
            site_cands: 1,
            row_block: false,
            ..GpuArms::default()
        };
        *gpu.off.rung("horizontal").expect("a rung") = true;
        *gpu.off.rung("calls").expect("a rung") = true;
        let opts = EmitOpts {
            gpu,
            ..EmitOpts::exact()
        };
        let today = emit_module(&ir, &opts, &RTX_4090).expect("emits");
        assert_eq!(
            emit_on(&ir, false).device,
            today.device,
            "the device unit is S54's"
        );
    }
}
