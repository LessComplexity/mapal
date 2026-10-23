//! the GPU realization of a tile site (plan-s41 steps 2–5, S52): the same
//! `Trn` at a second `Loc`.
//!
//! One cohesive unit, selected in one place (`emit_tiled_map`), never an `if`
//! inside a shared emitter (plan-s41 §2.2 rule 2). It has two halves:
//!
//! - **host** — [`FnEmit::emit_tiled_map_gpu`]: at the site, the host moves the
//!   two operands to the device, launches, and moves the output back, through
//!   one runtime call (`mapal_gpu_tile`, `crates/mapal-rt/gpu/mapal_gpu.c`).
//!   Those are the plan's `h2d` / `launch` / `d2h` transmissions.
//! - **device** — [`device_module`]: one `nvptx64` kernel per site, in a second
//!   translation unit that `llc` turns into PTX.
//!
//! **Values are the oracle's.** Every output cell accumulates its own products
//! in increasing `k`, one thread, one register, exactly the interpreter's left
//! fold — the tiles change *where* operands come from (shared memory, then
//! registers), never the order a cell combines them. The conformance face emits
//! a separate `fmul` and `fadd` (PTX `mul.rn` / `add.rn`, which `ptxas` may not
//! fuse); `--contract` emits `llvm.fma` per step, the CPU contract face's fused
//! form.
//!
//! The geometry is [`TargetProfile::gpu_tile`]'s: derived from device facts and
//! the site's extents, never a literal here.

use super::resident::Window;
use super::*;
use crate::Reduced;
use crate::profile::{
    Face, GPU_RESIDENT_BLOCKS, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, MMA_K, MMA_M,
    MMA_N, WARPGROUP_WARPS, WGMMA_ATOM_ROWS, WGMMA_K, WGMMA_M, WGMMA_SWIZZLE_BYTES, fma_held_regs,
};
use mapal_ir::{RowFoldOp, RowFoldSite};

/// Bytes one shared-memory wavefront serves: 32 banks × 4 B. An ISA rule on
/// every NVIDIA part since Kepler.
const GPU_WAVEFRONT_BYTES: u64 = 128;

/// The device data layout and triple `clang --target=nvptx64-nvidia-cuda`
/// emits (LLVM 22).
const DEVICE_PREAMBLE: &str = "\
target datalayout = \"e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64\"\n\
target triple = \"nvptx64-nvidia-cuda\"\n\
\n\
@mapal_smem = external addrspace(3) global [0 x i8], align 16\n\
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n\
declare void @llvm.nvvm.barrier0()\n\
declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.commit.group()\n\
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg)\n\
declare float @llvm.fma.f32(float, float, float)\n\
declare double @llvm.fma.f64(double, double, double)\n";

/// The host-side runtime entries, each declared only when the module calls it.
pub(crate) const GPU_INIT_DECL: &str = "declare void @mapal_gpu_init(i32)\n";
pub(crate) const GPU_TILE_DECL: &str =
    "declare void @mapal_gpu_tile(i64, ptr, i64, ptr, i64, ptr, i64, i32, i32, i32, i32)\n";

/// `mapal_gpu_init`'s flag bits; `mapal_gpu.c` reads the same ones.
pub(crate) const GPU_INIT_EAGER: u32 = 1;
/// S54: the module launches on chain streams (the runtime creates them).
pub(crate) const GPU_INIT_STREAMS: u32 = 2;
/// S54: every tile kernel opens with `griddepcontrol.wait` ([`pdl`]), so the
/// runtime may launch one right behind another as a dependent launch.
pub(crate) const GPU_INIT_PDL: u32 = 4;

/// `main`'s first call in a module with a GPU site: hands the runtime its host
/// rungs. Off, the runtime does nothing until the first site, the S52 path.
pub(crate) fn init_call(rungs: crate::GpuRungs, streams: bool, pdl: bool) -> String {
    let flags = u32::from(rungs.eager_init) * GPU_INIT_EAGER
        + u32::from(streams) * GPU_INIT_STREAMS
        + u32::from(pdl) * GPU_INIT_PDL;
    format!("  call void @mapal_gpu_init(i32 {flags})\n")
}

/// S54 `pdl`: programmatic dependent launch exists from sm_90 (PTX ISA:
/// `griddepcontrol` requires sm_90). One answer for the kernel text and the
/// runtime flag — a kernel launched dependent without the wait would race.
pub(crate) fn pdl(profile: &TargetProfile, contract: bool, arms: crate::GpuArms) -> bool {
    profile.gpu().is_some_and(|g| g.sm() >= 90) && host_rungs(profile, contract, arms).pdl
}

/// The rungs a GPU profile ships for one decision (S53): its measured table at
/// `wps` warps per scheduler on this face — every rung under the arm
/// `--gpu-off=arch-rungs` — then the measurement arms.
pub(crate) fn rungs_at(
    profile: &TargetProfile,
    wps: u64,
    face: Face,
    arms: crate::GpuArms,
) -> crate::GpuRungs {
    let table = if arms.off.arch {
        crate::GpuRungs::ALL
    } else {
        profile.gpu_rungs_on(wps, face)
    };
    table.with(arms)
}

/// The host rungs (`resident`, `devgen`, `retain-delta`, `eager-init`) are
/// decided per module, not per site: their table rows cover every fill, so any
/// fill reads them.
pub(crate) fn host_rungs(
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> crate::GpuRungs {
    rungs_at(profile, GPU_RESIDENT_BLOCKS, Face::of(contract, None), arms)
}

/// The clock fence's device half: a `time` read on a GPU profile waits for
/// every launch before it (`FnEmit::emit_time_ms`). Declared only beside a
/// tile call, so a clock-free module's text is what it was before the fence.
pub(crate) const GPU_SYNC: &str = "call void @mapal_gpu_sync()";
pub(crate) const GPU_SYNC_DECL: &str = "declare void @mapal_gpu_sync()\n";

/// S54 `l2-window`: a resident launch with an L2 access-policy window over
/// one of its buffers. Declared only when a launch calls it, so a module
/// without one is the text it was before the rung.
pub(crate) const GPU_L2_DECL: &str =
    "declare void @mapal_gpu_launch_l2(i64, i64, i64, i64, i32, i32, i32, i32, i64, i64)\n";

/// The split entries `resident` emits (`mapal_gpu.c`): a device pointer is an
/// `i64`, a device slot a `ptr` to one.
pub(crate) const RESIDENT_DECLS: &str = "\
declare i64 @mapal_gpu_alloc(ptr, i64)
declare void @mapal_gpu_h2d(i64, ptr, i64)
declare void @mapal_gpu_d2h(ptr, i64, i64, i64)
declare void @mapal_gpu_launch(i64, i64, i64, i64, i32, i32, i32, i32)
declare void @mapal_gpu_gen(i64, i64, i64)
declare void @mapal_gpu_free(ptr)
";

/// `arena` (S54): a function's one device allocation, made lazily into its
/// slot (`mapal_gpu.c`); declared only beside a call, so the OFF arm's text
/// is S53's.
pub(crate) const ARENA_DECL: &str = "declare i64 @mapal_gpu_arena(ptr, i64)\n";

/// `mapal_gpu_launchv(id, stream, nptr, ptrs, gx, gy, block, smem)` (S54): a
/// launch over any number of device pointers. Declared only when called, so a
/// module without one is the text it was before.
pub(crate) const LAUNCHV_DECL: &str =
    "declare void @mapal_gpu_launchv(i64, i32, i32, ptr, i32, i32, i32, i32)\n";

/// The most pointers one launch passes (`mapal_gpu.c`'s argument array).
pub(crate) const LAUNCHV_MAX: usize = 16;

/// S54 entries (`streams`, `pinned`), each declared only where it is called,
/// so a module that never opens a stream or pins keeps its S53 text.
/// (`launchv` is [`LAUNCHV_DECL`], shared with `epilogue` and `devmap`.)
pub(crate) const S54_DECLS: [(&str, &str); 4] = [
    (
        "@mapal_gpu_event(",
        "declare void @mapal_gpu_event(i32, i32)\n",
    ),
    (
        "@mapal_gpu_d2h_on(",
        "declare void @mapal_gpu_d2h_on(i32, ptr, i64, i64, i64)\n",
    ),
    ("@mapal_gpu_pin(", "declare void @mapal_gpu_pin(ptr, i64)\n"),
    ("@mapal_gpu_unpin(", "declare void @mapal_gpu_unpin()\n"),
];

/// Threads per block of a `devgen` kernel — the runtime's `MAPAL_GEN_THREADS`
/// launches it with exactly this many, so the two must agree.
/// ponytail: the naive kernel's literal, for the same reason (a one-element-
/// per-thread fill is bandwidth-bound at any block size a part accepts).
const GEN_THREADS: u64 = NAIVE_THREADS;

/// Threads per block of the one-thread-per-cell kernel.
/// ponytail: a literal, not derived — the fallback is a correctness path; derive
/// it when an untiled site turns out to matter for speed.
const NAIVE_THREADS: u64 = 256;

/// Can this site run on the GPU at all? Any float site whose reads are plain
/// affine in `(i, lane, k)`. The `k÷div` decomposition (conv taps) has no
/// device encoding yet and stays on the host.
pub(crate) fn gpu_site(site: &TileSite, profile: &TargetProfile) -> bool {
    profile.gpu().is_some()
        && is_float(&site.elem)
        && site.a.ksplit.is_none()
        && site.b.ksplit.is_none()
        && site.k > 0
        && site.rows * site.c > 0
}

/// The tile sites of `f` a GPU profile places (S54 `lane-stride`): the CPU's
/// [`CategoryIr::tile_plan`] plus the one-row strided-lane sites
/// ([`CategoryIr::tile_plan_strided`]) that are device sites — the
/// one-thread-per-cell kernel addresses any affine read. One answer for the
/// host walk and the device unit.
pub(crate) fn tile_plan(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    rungs: crate::GpuRungs,
) -> TilePlan {
    if !(rungs.lane_stride && profile.gpu().is_some()) {
        return ir.tile_plan(f);
    }
    let strict = ir.tile_plan(f);
    let mut plan = ir.tile_plan_strided(f);
    let extra: Vec<MorphismId> = plan
        .sites
        .iter()
        .filter(|&(m, s)| !strict.sites.contains_key(m) && !gpu_site(s, profile))
        .map(|(m, _)| m)
        .collect();
    for m in extra {
        plan.sites.remove(m);
    }
    plan
}

/// The arithmetic the selected kernel actually emits, distinct from the
/// module's permission to contract or use TF32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arithmetic {
    SeparateMulAdd,
    OrderedFma,
    Tf32Mma,
    Tf32Wgmma,
}

impl Arithmetic {
    fn scalar(site: &TileSite, contract: bool) -> Self {
        // Strided-lane sites are host maps whose step stays separate even
        // under Contract. Device placement must preserve that arithmetic.
        if contract && site.b.clane == 1 {
            Self::OrderedFma
        } else {
            Self::SeparateMulAdd
        }
    }

    fn face(self) -> Face {
        match self {
            Self::SeparateMulAdd => Face::Exact,
            Self::OrderedFma => Face::Contract,
            Self::Tf32Mma | Self::Tf32Wgmma => Face::Tf32,
        }
    }
}

/// Which kernel a GPU site gets, and how it is launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    pub arithmetic: Arithmetic,
    pub tiled: Option<GpuTile>,
    /// The tensor-core kernel, on a reduced face where the part has one.
    pub mma: Option<GpuMmaTile>,
    /// Hopper's asynchronous warpgroup tensor-core kernel (TF32 face, sm_90a).
    pub wgmma: Option<GpuWgmmaTile>,
    pub gx: u64,
    pub gy: u64,
    pub block: u64,
    pub smem: u64,
    /// The block-row group height the kernel walks its grid in (S54
    /// `raster`, [`Gpu::raster`]); 1 = row-major, no remap emitted.
    pub raster: u64,
    /// The rungs this site's kernel is emitted with: the profile's table at
    /// the site's fill, under the measurement arms.
    pub rungs: crate::GpuRungs,
    /// S62 `splitk` (ADR-0040): the `k` partitions the split variant folds,
    /// `z` riding in `ctaid.y` above the unsplit `gy`; 1 = never split.
    pub splitk: u64,
}

impl Launch {
    /// Blocks one wave holds: every SM times the blocks resident at this
    /// geometry (the derived tile's own count; for the one-thread-per-cell
    /// kernel, the thread and block limits of one SM).
    pub(crate) fn wave(&self, gpu: &Gpu) -> u64 {
        let per_sm = self
            .wgmma
            .map(|w| w.blocks_per_sm)
            .or(self.mma.map(|m| m.blocks_per_sm))
            .or(self.tiled.map(|t| t.blocks_per_sm))
            .unwrap_or_else(|| (gpu.max_threads_per_sm / self.block).min(gpu.max_blocks_per_sm));
        gpu.sms * per_sm.max(1)
    }
}

/// The staged kernel needs the separable shape the CPU's register-blocked rung
/// cashes (`b.ci == 0`): `a` ignores the lane and `b` ignores the row, so a
/// block's row panel of `a` and lane panel of `b` are each read once per `k`
/// step and shared by every cell of the tile. Strides are free — each operand
/// is staged with vector loads when its fast axis is unit-stride and aligned
/// ([`vector_a`], [`vector_b`]) and element by element otherwise. Anything else
/// takes the one-thread-per-cell kernel.
///
/// The rungs are decided here, once per site (S53): the site's fill is its
/// grid at the base FMA geometry (or the one-thread-per-cell grid), in warps
/// per scheduler; the profile's measured table at that fill and face, then the
/// arms.
pub(crate) fn launch(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
) -> Launch {
    let e = elem_bytes_of(&site.elem);
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    // The fill key reads the S52 geometry (no S53 geometry rung): the table's
    // rows were measured on grids classified by it.
    let base = shaped
        .then(|| profile.gpu_tile(site.rows, site.c, site.k, &site.elem, false, false))
        .flatten();
    let (blocks, threads) = match base {
        Some(g) => ((site.rows / g.bm()) * (site.c / g.bn()), g.threads()),
        None => ((site.rows * site.c).div_ceil(NAIVE_THREADS), NAIVE_THREADS),
    };
    let wps = gpu.warps_per_scheduler(blocks, threads);
    let raster = |rungs: crate::GpuRungs, bm: u64, bn: u64, per_sm: u64| match rungs.raster {
        true => gpu.raster(
            bm,
            bn,
            gpu.sms * per_sm,
            site.rows,
            site.c,
            site.k,
            e,
            rungs.l2_gate,
        ),
        false => 1,
    };
    // Attempt tensor geometry under its own policy. Permission alone is not
    // an emitted face: rejection must fall through to the scalar policy.
    if reduced == Some(Reduced::Tf32)
        && site.elem == Ty::f32()
        && shaped
        && vector_a(site)
        && vector_b(site)
    {
        let rungs = rungs_at(profile, wps, Face::Tf32, arms);
        if rungs.mma {
            let wgmma = rungs
                .wgmma
                .then(|| profile.gpu_wgmma_tile(site.rows, site.c, site.k))
                .flatten();
            if let Some(w) = wgmma {
                return Launch {
                    arithmetic: Arithmetic::Tf32Wgmma,
                    tiled: None,
                    mma: None,
                    wgmma: Some(w),
                    gx: site.c / w.bn(),
                    gy: site.rows / w.bm(),
                    block: w.threads(gpu.warp),
                    smem: w.smem_bytes(),
                    raster: raster(rungs, w.bm(), w.bn(), w.blocks_per_sm),
                    rungs,
                    splitk: 1,
                };
            }
            let mma = profile.gpu_mma_tile(site.rows, site.c, site.k, rungs.wide, rungs.fill);
            if let Some(m) = mma {
                return Launch {
                    arithmetic: Arithmetic::Tf32Mma,
                    tiled: None,
                    mma: Some(m),
                    wgmma: None,
                    gx: site.c / m.bn(),
                    gy: site.rows / m.bm(),
                    block: m.threads(gpu.warp),
                    smem: m.smem_bytes(),
                    raster: raster(rungs, m.bm(), m.bn(), m.blocks_per_sm),
                    rungs,
                    splitk: 1,
                };
            }
        }
    }
    let arithmetic = Arithmetic::scalar(site, contract);
    let rungs = rungs_at(profile, wps, arithmetic.face(), arms);
    let tiled = shaped
        .then(|| {
            profile.gpu_tile_with(site.rows, site.c, site.k, &site.elem, rungs.wide_fma, rungs)
        })
        .flatten()
        // S62 `cell-fill`: no register tile divides the site — one output per
        // thread, still staged ([`TargetProfile::gpu_cell_tile`]).
        .or_else(|| {
            (shaped && rungs.cell_fill)
                .then(|| profile.gpu_cell_tile(site.rows, site.c, site.k, &site.elem))
                .flatten()
        })
        .map(|g| {
            let streamed = rungs
                .cp_async
                .then(|| profile.gpu_tile_async(g, site.k, &site.elem))
                .flatten();
            if let Some(a) = streamed {
                return a;
            }
            // Register staging: the prefetch panel takes what a whole-file tile
            // leaves of the file (`wide-fma`), else (S62 `panel-regs`) what the
            // derived residency leaves after the tile, else one fragment pair.
            // `panel-regs` only on a grid that leaves a warp scheduler without a
            // warp: there the panel's independent loads are the only latency
            // hiding; with a warp per scheduler, warps hide it and the deeper
            // panel costs registers (S62 v1, the 5090, 4090, H100 PCIe and 3090:
            // 35 of 38 measured cells on this side of the line).
            let grid = (site.rows / g.bm()) * (site.c / g.bn());
            let g = match rungs.prefetch {
                true if g.held > 0 => {
                    prefetch_depth(g, e, gpu.regs_per_thread.saturating_sub(g.held))
                }
                true if rungs.panel_regs && gpu.warps_per_scheduler(grid, g.threads()) == 0 => {
                    prefetch_depth(g, e, panel_regs(gpu, &g, e))
                }
                true => prefetch_depth(g, e, g.tm + g.tn),
                false => g,
            };
            if rungs.pad && vector_a(site) {
                profile.gpu_tile_sync_a(g, &site.elem)
            } else {
                g
            }
        })
        // S62 `asyncstage`: stages from the blocks this grid puts on an SM.
        // Legal for the unpredicated copies above: `rows % bm`, `c % bn`,
        // `k % bk` are 0 (`TileSite` extents) and `shaped` makes each panel
        // one read per block; the panel order `0..k/bk` and each cell's step
        // order are unchanged ([`Kernel::step`]).
        .map(|g| {
            let blocks = (site.rows / g.bm()) * (site.c / g.bn());
            rungs
                .async_stage
                .then(|| profile.gpu_tile_stages(g, blocks, site.k, &site.elem))
                .flatten()
                .unwrap_or(g)
        });
    let mut l = match tiled {
        Some(g) => Launch {
            arithmetic,
            tiled: Some(g),
            mma: None,
            wgmma: None,
            gx: site.c / g.bn(),
            gy: site.rows / g.bm(),
            block: g.threads(),
            smem: g.smem_bytes(e),
            raster: raster(rungs, g.bm(), g.bn(), g.blocks_per_sm),
            rungs,
            splitk: 1,
        },
        None => Launch {
            arithmetic,
            tiled: None,
            mma: None,
            wgmma: None,
            gx: blocks,
            gy: 1,
            block: NAIVE_THREADS,
            smem: 0,
            raster: 1,
            rungs,
            splitk: 1,
        },
    };
    // S62 `splitk`: only the reassociation face permits a partitioned fold
    // (ADR-0040), and only the scalar kernels have one. `raster` and `splitk`
    // are exclusive by derivation — a grid is grouped only above one wave,
    // split only below — so a forced count never meets a remapped grid.
    if reduced == Some(Reduced::Reassoc) && rungs.splitk && l.raster == 1 {
        let unit = l.tiled.map_or(1, |g| g.bk);
        let derived = || {
            profile.gpu_splitk(
                l.gx * l.gy,
                l.wave(gpu),
                site.k,
                unit,
                site.rows * site.c,
                e,
            )
        };
        l.splitk = match arms.splitk {
            s if s > 0 && site.k.is_multiple_of(s * unit) => s,
            _ => derived(),
        };
    }
    l
}

/// The prefetch rung holds the next panel in registers while the current one
/// is computed, so its depth is bounded by registers, not shared memory: one
/// panel's elements per thread, `(bm + bn)·bk / threads`, may take at most
/// `budget` registers — one fragment pair's worth (`tm + tn`, S52), or what
/// the register tile leaves of the file (`wide-fma`, S53). Halve `bk` until it
/// fits; a halved power of two still divides `k` and still loads in whole
/// vectors per thread.
fn prefetch_depth(g: GpuTile, elem_bytes: u64, budget: u64) -> GpuTile {
    let lanes = GPU_VEC_BYTES / elem_bytes;
    let regs = |bk: u64| (g.bm() + g.bn()) * bk * elem_bytes.div_ceil(4) / g.threads();
    let whole = |bk: u64| {
        (g.bm() * bk / lanes).is_multiple_of(g.threads())
            && (bk * g.bn() / lanes).is_multiple_of(g.threads())
    };
    let mut bk = g.bk;
    while regs(bk) > budget && bk / 2 >= lanes && whole(bk / 2) {
        bk /= 2;
    }
    GpuTile { bk, ..g }
}

/// S62 `panel-regs`: the registers a register-staged thread has free for its
/// prefetch panel — the file the derived residency gives it (`regs_per_sm`
/// over `threads × blocks_per_sm`, at most `regs_per_thread`; the bound
/// `minctasm` holds ptxas to) less what its tile holds ([`fma_held_regs`]),
/// and never less than S52's one fragment pair. The S52 8×8 tile at 256
/// threads, two per SM, holds its whole 128-register share: the pair.
fn panel_regs(gpu: &Gpu, g: &GpuTile, elem_bytes: u64) -> u64 {
    let file = (gpu.regs_per_sm / (g.threads() * g.blocks_per_sm.max(1))).min(gpu.regs_per_thread);
    file.saturating_sub(fma_held_regs(g.tm, g.tn, elem_bytes))
        .max(g.tm + g.tn)
}

fn aligned(site: &TileSite, x: u64) -> bool {
    x.is_multiple_of(GPU_VEC_BYTES / elem_bytes_of(&site.elem))
}

/// `a`'s panel rows are contiguous along `k` and vector-aligned.
fn vector_a(site: &TileSite) -> bool {
    site.a.ck == 1 && aligned(site, site.a.ci) && aligned(site, site.a.base)
}

/// `b`'s panel rows are contiguous along the lane and vector-aligned.
fn vector_b(site: &TileSite) -> bool {
    site.b.clane == 1 && aligned(site, site.b.ck) && aligned(site, site.b.base)
}

fn elem_bytes_of(elem: &Ty) -> u64 {
    match elem {
        Ty::Float { bits } => u64::from(*bits) / 8,
        _ => 4,
    }
}

/// A stable id for a site's kernel: FNV-1a over everything the kernel text
/// depends on. The host passes it to the runtime, which looks the kernel up as
/// `mapal_tile_<id>` — so the two translation units agree without sharing state.
pub(crate) fn kernel_id(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    reduced: Option<Reduced>,
) -> u64 {
    kernel_id_with(site, profile, contract, reduced, None)
}

/// [`kernel_id`] of a site whose store evaluates `epi` (S54): the key also
/// names the group — its bodies, how each slot is fed, what is stored.
pub(crate) fn kernel_id_with(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    reduced: Option<Reduced>,
    epi: Option<&EpiKernel>,
) -> u64 {
    let mut key = format!("{site:?}|{}|{contract}|{reduced:?}", profile.name);
    if let Some(e) = epi {
        key.push_str(&format!("|{e:?}"));
    }
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// [`kernel_id`] of the site's kernel storing its output column-major
/// (`rowfold-layout`, [`col_store`]); the row-major id unchanged.
pub(crate) fn kernel_id_col(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    reduced: Option<Reduced>,
    col: bool,
) -> u64 {
    if !col {
        return kernel_id(site, profile, contract, reduced);
    }
    let key = format!("{site:?}|{}|{contract}|{reduced:?}|col", profile.name);
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// S62 `splitk`: the ids of a site's split kernel (`reduce` false: partial
/// folds into the workspace) and its reduce kernel (true) — the `col`
/// precedent. The ordered kernel keeps [`kernel_id`], so a function that does
/// not split the site (non-resident, an epilogue, a stream) launches that.
pub(crate) fn kernel_id_split(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    reduced: Option<Reduced>,
    reduce: bool,
) -> u64 {
    let part = if reduce { "reduce" } else { "splitk" };
    let key = format!("{site:?}|{}|{contract}|{reduced:?}|{part}", profile.name);
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Can this launch store its output column-major (`o[j·rows + i]`)? The FMA
/// tile writes `lanes` consecutive rows of one column per vector, which needs
/// whole vectors of rows (`tm % lanes == 0`); the one-thread-per-cell kernel
/// and the `cell-fill` tile (`tn < lanes`, the kernel's `narrow` store) one
/// word. A `fill-rows` tile (`tm < lanes ≤ tn`) stores row vectors: row-major.
/// ponytail: the tensor-core kernels keep row-major — no program reads a TF32
/// site through a fold; add their store when one does.
pub(crate) fn col_store(l: &Launch, site: &TileSite) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&site.elem);
    l.mma.is_none()
        && l.wgmma.is_none()
        && l.tiled
            .is_none_or(|g| g.tm.is_multiple_of(lanes) || g.tn < lanes)
        && site.rows.is_multiple_of(lanes)
}

/// **S54 R1, the device half: the per-cell consumers a site's store
/// evaluates** (`resident.rs:Epi`), in the terms the kernel needs. Group value
/// 0 is the site's accumulator, `k + 1` call `k`'s result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EpiKernel {
    /// The site's own output is stored (a reader outside the group).
    pub store_out: bool,
    /// Device pointers to complete operands, after `a, b, out` (`%x<i>`).
    pub caps: usize,
    pub calls: Vec<EpiCall>,
}

/// One member: `ret @callee(arg)`, its input assembled field by field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EpiCall {
    pub callee: String,
    /// The body's input type: a scalar element, or a `{ … }` struct.
    pub arg: String,
    pub ret: String,
    pub fields: Vec<EpiArg>,
    /// Stored (`%y<j>`, in call order among the stored).
    pub store: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpiArg {
    /// Group value `k`, by value (a mapped element).
    Value(usize),
    /// Group value `k` through a one-cell buffer: a captured array the body
    /// reads at `t`, which its device variant reads at 0.
    Cell(usize),
    /// Operand pointer `%x<i>`.
    Cap(usize),
    /// The cell index `t`, `i32`.
    Index,
}

/// A grouped site for the device unit: its function and morphism (its plain
/// kernel is not emitted), geometry and epilogue.
pub(crate) struct EpiSite {
    pub f: mapal_ir::FuncId,
    pub m: mapal_ir::MorphismId,
    pub site: TileSite,
    pub k: EpiKernel,
}

pub(crate) fn kernel_name(id: u64) -> String {
    format!("mapal_tile_{id:016x}")
}

/// A `devgen` kernel's id: FNV-1a over its body's name, length and profile —
/// the host passes it, the device unit names the kernel `mapal_gen_<id>`.
pub(crate) fn gen_id(body: &str, n: u64, profile: &TargetProfile) -> u64 {
    format!("gen|{body}|{n}|{}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A `devgen` producer for the device unit: `(id, body fn name, element
/// type, the body's own emitted text)`.
pub(crate) type GenKernel = (u64, String, String, String);

/// `out[i] = body(i)`, one thread per element: the host's own body text runs
/// on the device (a clean leaf fn is scalar IR with no host symbol), so the
/// element is the host's bit for bit.
fn gen_kernel_text(id: u64, callee: &str, elem: &str) -> String {
    format!(
        "define ptx_kernel void @mapal_gen_{id:016x}(ptr addrspace(1) %o, i64 %n) \
         \"nvvm.maxntid\"=\"{GEN_THREADS}\" {{\n\
         entry:\n\
         \x20 %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %t to i64\n\
         \x20 %b64 = zext i32 %b to i64\n\
         \x20 %base = mul i64 %b64, {GEN_THREADS}\n\
         \x20 %i = add i64 %base, %t64\n\
         \x20 %in = icmp ult i64 %i, %n\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         \x20 %x = trunc i64 %i to i32\n\
         \x20 %v = call {elem} @{callee}(i32 %x)\n\
         \x20 %p = getelementptr inbounds {elem}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {elem} %v, ptr addrspace(1) %p\n\
         \x20 br label %done\n\
         done:\n\
         \x20 ret void\n\
         }}\n"
    )
}

/// A device row fold's kernel id (S54 `rowfold`): FNV-1a over the site (its
/// read already re-addressed for a column-major producer) and the profile.
/// Named `mapal_tile_<id>` like a site's, so the runtime's `launch` finds it.
pub(crate) fn rowfold_id(site: &RowFoldSite, profile: &TargetProfile) -> u64 {
    format!("rowfold|{site:?}|{}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A cell kernel's id (`devmap`): FNV-1a over its body's name, extent and
/// profile. The kernel is named like a site's (`mapal_tile_<id>`), which is
/// how `mapal_gpu_launchv` finds it; the `cell` key keeps the two apart.
pub(crate) fn cell_id(body: &str, n: u64, profile: &TargetProfile) -> u64 {
    format!("cell|{body}|{n}|{}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A row fold's launch `(blocks, threads)`: one thread per row, and the rows
/// spread over every SM — a block of `rows / sms` threads rounded up to a
/// warp, at most [`NAIVE_THREADS`]. A fold has one thread per row and no more
/// (ADR-0028 D4), so few threads that the SMs they land on, not the block
/// size, bound how many loads are in flight.
pub(crate) fn rowfold_grid(site: &RowFoldSite, profile: &TargetProfile) -> (u64, u64) {
    let g = profile.gpu().expect("a device fold has a GPU profile");
    let threads = site
        .rows
        .div_ceil(g.sms)
        .next_multiple_of(g.warp)
        .min(NAIVE_THREADS);
    (site.rows.div_ceil(threads), threads)
}

/// `o[i] = fold_k (seed, ⊕) a[base + ci·i + ck·k]`, one thread per row, `k`
/// in order: the host fold's chain, so its bits. The step is the host's own
/// instruction (`fadd`/`fsub`/`fmul`, or `fcmp` + `select` for a running
/// max/min) in the source's operand order, never `contract`ed. The kernel
/// takes the site ABI's three pointers; `%b` is unused.
fn rowfold_kernel_text(name: &str, s: &RowFoldSite, profile: &TargetProfile) -> String {
    let (_, threads) = rowfold_grid(s, profile);
    let t = lower_ty(&s.elem).expect("float elem lowers");
    let e = elem_bytes_of(&s.elem);
    let seed = const_literal(&s.seed);
    let step = match s.op {
        RowFoldOp::Arith { op, acc_first } => {
            let code = match op {
                Operation::Add => "fadd",
                Operation::Sub => "fsub",
                _ => "fmul",
            };
            let (x, y) = if acc_first {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            format!("  %accn = {code} {t} {x}, {y}\n")
        }
        RowFoldOp::Select {
            cmp,
            acc_lhs,
            acc_if_true,
        } => {
            let pred = match cmp {
                Operation::Lt => "olt",
                Operation::Gt => "ogt",
                Operation::Le => "ole",
                _ => "oge",
            };
            let (x, y) = if acc_lhs {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            let (p, q) = if acc_if_true {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            format!("  %c = fcmp {pred} {t} {x}, {y}\n  %accn = select i1 %c, {t} {p}, {t} {q}\n")
        }
    };
    let (rows, k, ci, ck, base) = (s.rows, s.k, s.a.ci, s.a.ck, s.a.base);
    format!(
        "define ptx_kernel void @{name}(\
         ptr addrspace(1) noalias readonly align {e} %a, \
         ptr addrspace(1) noalias readonly align {e} %b, \
         ptr addrspace(1) noalias align {e} %o) \"nvvm.maxntid\"=\"{threads}\" {{\n\
         entry:\n\
         \x20 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %tid to i64\n\
         \x20 %b64 = zext i32 %bid to i64\n\
         \x20 %row0 = mul i64 %b64, {threads}\n\
         \x20 %i = add i64 %row0, %t64\n\
         \x20 %in = icmp ult i64 %i, {rows}\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         \x20 %ri = mul i64 %i, {ci}\n\
         \x20 %r0 = add i64 %ri, {base}\n\
         \x20 br label %kloop\n\
         kloop:\n\
         \x20 %k = phi i64 [0, %body], [%kn, %kloop]\n\
         \x20 %acc = phi {t} [{seed}, %body], [%accn, %kloop]\n\
         \x20 %kk = mul i64 %k, {ck}\n\
         \x20 %x = add i64 %r0, %kk\n\
         \x20 %p = getelementptr inbounds {t}, ptr addrspace(1) %a, i64 %x\n\
         \x20 %v = load {t}, ptr addrspace(1) %p, align {e}\n\
         {step}\
         \x20 %kn = add i64 %k, 1\n\
         \x20 %fin = icmp eq i64 %kn, {k}\n\
         \x20 br i1 %fin, label %store, label %kloop\n\
         store:\n\
         \x20 %q = getelementptr inbounds {t}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {t} %accn, ptr addrspace(1) %q, align {e}\n\
         \x20 br label %done\n\
         done:\n\
         \x20 ret void\n\
         }}\n"
    )
}

/// A cell kernel's launch: one thread per cell, [`NAIVE_THREADS`] per block
/// (a one-load, one-store-per-thread kernel is bandwidth-bound at any block
/// size a part accepts — the `devgen` reason), as many blocks as cover `n`.
pub(crate) fn cell_grid(n: u64) -> (u64, u64) {
    (n.div_ceil(NAIVE_THREADS), NAIVE_THREADS)
}

/// `cell-law`'s device variant of body `name` (`FnEmit::cell_vals`).
pub(crate) fn cell_variant(name: &str) -> String {
    format!("{name}_cell")
}

/// `out[i] = body(caps…, x[i])`, one thread per cell (`devmap`): the host's
/// body text runs on the device, so the cell is the host's bit for bit. The
/// parameters are the output, then the mapped array when it is loaded (`elem`
/// is its element type; `None` forms the element from the index, an `iota`),
/// then each captured array the cell loads — generic pointers to the body, as
/// the host passes them by reference. A capture `Some((law, ty))` is not a
/// parameter: `cell-law` evaluates `law(i)` and `callee` (the body's variant)
/// takes the value after its input.
pub(crate) fn cell_kernel_text(
    id: u64,
    callee: &str,
    n: u64,
    out: &str,
    arg: &str,
    elem: Option<&str>,
    caps: &[Option<(String, String)>],
) -> String {
    let mut params = vec!["ptr addrspace(1) %o".to_string()];
    if elem.is_some() {
        params.push("ptr addrspace(1) %x".into());
    }
    params.extend(
        caps.iter()
            .enumerate()
            .filter(|(_, c)| c.is_none())
            .map(|(k, _)| format!("ptr addrspace(1) %c{k}")),
    );
    let mut body = String::new();
    let e_ty = match elem {
        Some(t) => {
            body.push_str(&format!(
                "  %xp = getelementptr inbounds {t}, ptr addrspace(1) %x, i64 %i\n  %e = load {t}, ptr addrspace(1) %xp\n"
            ));
            t
        }
        None => {
            body.push_str("  %e = trunc i64 %i to i32\n");
            "i32"
        }
    };
    let mut call_arg = if caps.is_empty() {
        format!("{e_ty} %e")
    } else {
        let mut prev = "poison".to_string();
        for (k, c) in caps.iter().enumerate() {
            if c.is_some() {
                continue; // the variant never reads the pointer
            }
            body.push_str(&format!(
                "  %g{k} = addrspacecast ptr addrspace(1) %c{k} to ptr\n  %a{k} = insertvalue {arg} {prev}, ptr %g{k}, {k}\n"
            ));
            prev = format!("%a{k}");
        }
        body.push_str(&format!(
            "  %a = insertvalue {arg} {prev}, {e_ty} %e, {}\n",
            caps.len()
        ));
        format!("{arg} %a")
    };
    for (k, c) in caps.iter().enumerate() {
        if let Some((law, t)) = c {
            body.push_str(&format!("  %v{k} = call {t} @{law}(i32 %e)\n"));
            call_arg.push_str(&format!(", {t} %v{k}"));
        }
    }
    format!(
        "define ptx_kernel void @{name}({params}) \"nvvm.maxntid\"=\"{NAIVE_THREADS}\" {{\n\
         entry:\n\
         \x20 %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %t to i64\n\
         \x20 %b64 = zext i32 %b to i64\n\
         \x20 %base = mul i64 %b64, {NAIVE_THREADS}\n\
         \x20 %i = add i64 %base, %t64\n\
         \x20 %in = icmp ult i64 %i, {n}\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         {body}\
         \x20 %v = call {out} @{callee}({call_arg})\n\
         \x20 %p = getelementptr inbounds {out}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {out} %v, ptr addrspace(1) %p\n\
         \x20 br label %done\n\
         done:\n\
         \x20 ret void\n\
         }}\n",
        name = kernel_name(id),
        params = params.join(", "),
    )
}

/// A `devmap` cell for the device unit: the functions its kernel calls
/// (`(name, emitted text)`), then the kernel text.
pub(crate) type CellKernel = (Vec<(String, String)>, String);

/// The device translation unit: one kernel per GPU tile site in `ir`, in name
/// order. `None` when no site offloads (every CPU profile, and a GPU profile
/// on a program with no tile site).
pub(crate) fn device_module(
    ir: &CategoryIr,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
    gens: &[GenKernel],
    epis: &[EpiSite],
    epi_bodies: &[(String, String)],
    cells: &[CellKernel],
) -> Option<String> {
    let mut kernels = std::collections::BTreeMap::new();
    for (f, _) in ir.funcs() {
        let rungs = host_rungs(profile, contract, arms);
        let plan = tile_plan(ir, f, profile, rungs);
        // A retained site's restricted pass (S53) launches its one-row kernel.
        let retained = super::retain::retained_sites(ir, f, profile, rungs);
        // S54 `rowfold`: the device folds, and the sites that store for them.
        let folds = super::rowfold::rowfold_sites(ir, f, profile, contract, arms, reduced);
        let col = |m: MorphismId| folds.values().any(|r| r.colmajor == Some(m));
        // A grouped site (S54 `epilogue`) launches its epilogue kernel instead.
        let sites = plan
            .sites
            .iter()
            .filter(|&(m, _)| !epis.iter().any(|e| e.f == f && e.m == m))
            .map(|(m, s)| (s, col(m), true));
        let rows = retained.values().map(|r| (&r.row, false, false));
        for (site, col, whole) in sites.chain(rows) {
            if gpu_site(site, profile) {
                let name = kernel_name(kernel_id_col(site, profile, contract, reduced, col));
                kernels.entry(name.clone()).or_insert_with(|| {
                    kernel_text(
                        &name, site, profile, contract, arms, reduced, col, None, false,
                    )
                });
                // S62 `splitk`: a whole site that may split gets its split and
                // reduce kernels too; the host picks per function (resident,
                // no epilogue, one stream), so the ordered one stays.
                if whole && !col && launch(site, profile, contract, arms, reduced).splitk > 1 {
                    let id = |r| kernel_name(kernel_id_split(site, profile, contract, reduced, r));
                    let (split, red) = (id(false), id(true));
                    kernels.entry(split.clone()).or_insert_with(|| {
                        kernel_text(
                            &split, site, profile, contract, arms, reduced, false, None, true,
                        )
                    });
                    kernels.entry(red.clone()).or_insert_with(|| {
                        reduce_kernel_text(&red, site, profile, contract, arms, reduced)
                    });
                }
            }
        }
        for r in folds.values() {
            let name = kernel_name(rowfold_id(&r.site, profile));
            kernels
                .entry(name.clone())
                .or_insert_with(|| rowfold_kernel_text(&name, &r.site, profile));
        }
    }
    for e in epis {
        let id = kernel_id_with(&e.site, profile, contract, reduced, Some(&e.k));
        let name = kernel_name(id);
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(
                &name,
                &e.site,
                profile,
                contract,
                arms,
                reduced,
                false,
                Some(&e.k),
                false,
            )
        });
    }
    if kernels.is_empty() && gens.is_empty() && cells.is_empty() {
        return None;
    }
    let mut out = String::from("; mapal-backend-llvm device module\n");
    out.push_str(DEVICE_PREAMBLE);
    let mut md = String::new();
    for (i, k) in kernels.values().enumerate() {
        out.push('\n');
        if k.contains("!@LOOP@") {
            let n = 2 * i;
            out.push_str(&k.replace("!@LOOP@", &format!("!{n}")));
            md.push_str(&format!(
                "!{n} = distinct !{{!{n}, !{}}}\n!{} = !{{!\"llvm.loop.unroll.full\"}}\n",
                n + 1,
                n + 1
            ));
        } else {
            out.push_str(k);
        }
    }
    // `devgen`: each body once, then its kernel. A body is a clean leaf, so
    // its text names no host symbol — checked, since a stray one would fail
    // only at `llc` on the bundle machine.
    let mut bodies = std::collections::BTreeSet::new();
    for (id, callee, elem, text) in gens {
        assert!(
            !text.contains("@mapal_"),
            "devgen body {callee} names a host runtime symbol"
        );
        if bodies.insert(callee.clone()) {
            out.push('\n');
            out.push_str(text);
        }
        out.push('\n');
        out.push_str(&gen_kernel_text(*id, callee, elem));
    }
    // The bodies an epilogue calls (S54): the host's text, or its device
    // variant (`_cell`); clean leaves, under the same check.
    for (callee, text) in epi_bodies {
        assert!(
            !text.contains("@mapal_"),
            "epilogue body {callee} names a host runtime symbol"
        );
        if bodies.insert(callee.clone()) {
            out.push('\n');
            out.push_str(text);
        }
    }

    // `devmap`: each body once (shared with `devgen`'s), then its kernel.
    for (fns, kernel) in cells {
        for (callee, text) in fns {
            assert!(
                !text.contains("@mapal_"),
                "cell body {callee} names a host runtime symbol"
            );
            if bodies.insert(callee.clone()) {
                out.push('\n');
                out.push_str(text);
            }
        }
        out.push_str("\n; cell\n");
        out.push_str(kernel);
    }
    out.push('\n');
    out.push_str(&md);
    // A `wgmma` kernel's swizzle atoms start on 1024-byte boundaries.
    if out.contains("wgmma.mma_async") {
        out = out.replacen("[0 x i8], align 16", "[0 x i8], align 1024", 1);
    }
    Some(out)
}

fn kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
    col: bool,
    epi: Option<&EpiKernel>,
    split: bool,
) -> String {
    let mut k = kernel(site, profile, contract, arms, reduced, col, epi, split);
    let l = launch(site, profile, contract, arms, reduced);
    debug_assert!(
        !col || col_store(&l, site),
        "a column-major store this kernel has"
    );
    debug_assert!(!split || (l.splitk > 1 && !col && epi.is_none()));
    match (l.wgmma, l.mma, l.tiled) {
        (Some(w), _, _) => k.wgmma(name, w),
        (None, Some(m), _) => k.mma(name, m),
        (None, None, Some(g)) => k.tiled(name, g),
        (None, None, None) => k.naive(name, l.block),
    }
    k.out
}

/// S62 `splitk`'s second launch: `o[t] = ((P₀[t] + P₁[t]) + …) + P_{S−1}[t]`,
/// one thread per cell over the workspace `%a` (`S` partitions of `rows·c`),
/// partition order fixed, straight-line — ADR-0040's combine, so the bits are
/// a function of the site and the profile. No fast-math flag: LLVM may not
/// re-bracket what the text fixes. `%b` is unused.
fn reduce_kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
) -> String {
    let s = launch(site, profile, contract, arms, reduced).splitk;
    let mut k = kernel(site, profile, contract, arms, reduced, false, None, true);
    let (t, ix, cells) = (k.t.clone(), k.idx, site.rows * site.c);
    let e = elem_bytes_of(&site.elem);
    k.header(name, NAIVE_THREADS, e, 0);
    let tid = k.sreg("tid.x");
    let bx = k.sreg("ctaid.x");
    let cell0 = k.op("mul", &bx, NAIVE_THREADS);
    let cell = k.op("add", &cell0, &tid);
    let inside = k.tmp();
    k.line(format!("{inside} = icmp ult {ix} {cell}, {cells}"));
    k.line(format!("br i1 {inside}, label %body, label %done"));
    k.out.push_str("body:\n");
    let mut acc = String::new();
    for z in 0..s {
        let i = k.plus(cell.clone(), z * cells);
        let p = k.gep(1, "%a", &i);
        let v = k.tmp();
        k.line(format!("{v} = load {t}, ptr addrspace(1) {p}, align {e}"));
        acc = if z == 0 {
            v
        } else {
            let r = k.tmp();
            k.line(format!("{r} = fadd {t} {acc}, {v}"));
            r
        };
    }
    let p = k.gep(1, "%o", &cell);
    k.line(format!("store {t} {acc}, ptr addrspace(1) {p}, align {e}"));
    k.line("br label %done");
    k.out.push_str("done:\n  ret void\n}\n");
    k.out
}

/// The kernel builder for a site's launch: its index width, arithmetic,
/// rungs and, for the split variant (`split`), the partition count.
#[allow(clippy::too_many_arguments)]
fn kernel<'s>(
    site: &'s TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
    col: bool,
    epi: Option<&'s EpiKernel>,
    split: bool,
) -> Kernel<'s> {
    let l = launch(site, profile, contract, arms, reduced);
    let rungs = l.rungs;
    let splitk = if split { l.splitk } else { 1 };
    // `idx32` is an FMA / one-thread-per-cell rung: the tensor-core kernels
    // (`mma.sync`, `wgmma`) form their indices in `i64`.
    let tensor = l.mma.is_some() || l.wgmma.is_some();
    let idx = if rungs.idx32 && !tensor && max_index(site, &l, splitk) < 1 << 31 {
        "i32"
    } else {
        "i64"
    };
    Kernel {
        out: String::new(),
        next: 0,
        idx,
        t: lower_ty(&site.elem).expect("float elem lowers"),
        site,
        arithmetic: l.arithmetic,
        rungs: l.rungs,
        profile_gpu: *profile.gpu().expect("a GPU site has a GPU profile"),
        pdl: pdl(profile, contract, arms),
        async_tail: None,
        sync_tail: None,
        raster: (l.gx, l.raster),
        col,
        epi,
        splitk,
        gy_out: l.gy,
        z: None,
    }
}

/// The largest index a kernel forms (S53 rung `idx32`): each operand's last
/// element — `base + ci·(rows−1) + clane·(c−1) + ck·(k−1)`, site extents —
/// every cell a launched thread can name (the one-thread-per-cell grid rounds
/// up to whole blocks), and `k`. Every intermediate of the index arithmetic is
/// a partial sum of one of these, so below 2³¹ the kernel indexes in `i32`
/// with `nuw nsw`; pointers stay 64-bit. A split kernel (S62) names its
/// workspace cell `z·rows·c + cell`, `z < splitk`; a partition's reads stay
/// inside the operand's own last element.
fn max_index(site: &TileSite, l: &Launch, splitk: u64) -> u64 {
    let last = |r: &mapal_ir::TileRead| {
        r.base + r.ci * (site.rows - 1) + r.clane * (site.c - 1) + r.ck * (site.k - 1)
    };
    let cells = (site.rows * site.c).max(l.gx * l.gy * l.block) * splitk;
    last(&site.a).max(last(&site.b)).max(cells).max(site.k)
}

/// The integer arithmetic a panel lane map is written in, so one definition
/// both emits the kernel's index IR ([`Kernel`]) and evaluates on numbers
/// ([`Num`]) — the evaluation is what proves a copy affine ([`affine_delta`]).
trait Arith {
    type V: Clone;
    fn imm(&mut self, code: &str, a: &Self::V, c: u64) -> Self::V;
    fn sum(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
}

struct Num;

impl Arith for Num {
    type V = u64;
    fn imm(&mut self, code: &str, a: &u64, c: u64) -> u64 {
        match code {
            "add" => a + c,
            "mul" => a * c,
            "udiv" => a / c,
            "urem" => a % c,
            _ => unreachable!("lane maps use add/mul/udiv/urem"),
        }
    }
    fn sum(&mut self, a: &u64, b: &u64) -> u64 {
        a + b
    }
}

impl Arith for Kernel<'_> {
    type V = String;
    fn imm(&mut self, code: &str, a: &String, c: u64) -> String {
        self.op(code, a, c)
    }
    fn sum(&mut self, a: &String, b: &String) -> String {
        self.op("add", a, b)
    }
}

/// S54 `raster`: the block `(bx, by)` of a `gx`-wide grid, launched as
/// `by·gx + bx`, walked `h` block-rows per column: launch groups of `h·gx`,
/// each down `h` rows, then the next column. A bijection on the grid when `h`
/// divides its height ([`Gpu::raster`] only derives such an `h`).
fn raster_block<A: Arith>(ar: &mut A, bx: &A::V, by: &A::V, gx: u64, h: u64) -> (A::V, A::V) {
    let row = ar.imm("mul", by, gx);
    let lin = ar.sum(&row, bx);
    let group = ar.imm("udiv", &lin, h * gx);
    let local = ar.imm("urem", &lin, h * gx);
    let top = ar.imm("mul", &group, h);
    let dy = ar.imm("urem", &local, h);
    (ar.imm("udiv", &local, h), ar.sum(&top, &dy))
}

/// Where linear copy `li` (`tid + p·threads`) of a staged panel lands, as
/// (panel row, panel column): `a` → (`m`, `k`), `b` → (`k`, `n`).
#[derive(Clone, Copy, Debug)]
enum LaneMap {
    /// `per_row` consecutive copies along a row, `unit` elements each.
    RowMajor { per_row: u64, unit: u64 },
    /// Consecutive copies down a `rows`-long column (`a` column-major).
    ColMajor { rows: u64 },
    /// `l` copies along a row, then down `rows` rows, then the next `l·unit`
    /// columns: the `cp.async` element copy (`l = lanes`, `unit` 1) and the
    /// `pad` rung's vector copy (`l` from [`TargetProfile::gpu_tile_sync_a`]).
    Split { l: u64, rows: u64, unit: u64 },
}

impl LaneMap {
    fn coords<A: Arith>(self, ar: &mut A, li: &A::V) -> (A::V, A::V) {
        let scaled = |ar: &mut A, v: A::V, unit: u64| {
            if unit == 1 {
                v
            } else {
                ar.imm("mul", &v, unit)
            }
        };
        match self {
            LaneMap::RowMajor { per_row, unit } => {
                let r = ar.imm("udiv", li, per_row);
                let c = ar.imm("urem", li, per_row);
                (r, scaled(ar, c, unit))
            }
            LaneMap::ColMajor { rows } => {
                let c = ar.imm("udiv", li, rows);
                (ar.imm("urem", li, rows), c)
            }
            LaneMap::Split { l, rows, unit } => {
                let lo = ar.imm("urem", li, l);
                let q = ar.imm("udiv", li, l);
                let r = ar.imm("urem", &q, rows);
                let hi = ar.imm("udiv", &q, rows);
                let hi = ar.imm("mul", &hi, l);
                let c = ar.sum(&hi, &lo);
                (r, scaled(ar, c, unit))
            }
        }
    }
}

/// S53 rung `affine`: copy `p` is copy 0 moved by a constant when the lane map
/// is periodic in `threads` — `map(tid + p·threads) = map(tid) + map(p·threads)`
/// for every `tid` (`map(0) = (0, 0)`). With power-of-two extents it is
/// (`Q = threads / l`: `Q | rows` or `rows | Q`); it is checked here for every
/// thread rather than assumed, and a copy that fails keeps its own
/// `udiv`/`urem` chain.
fn affine_delta(map: LaneMap, threads: u64, p: u64) -> Option<(u64, u64)> {
    let at = |li: u64| map.coords(&mut Num, &li);
    let d = at(p * threads);
    (0..threads)
        .all(|t| {
            let (r, c) = at(t);
            at(t + p * threads) == (r + d.0, c + d.1)
        })
        .then_some(d)
}

/// The tiled kernel's two staged copies, `[a, b]`, each as (lane map, copies
/// per thread per panel). A vector operand moves `lanes` elements along its
/// fast axis per copy; a strided one moves one element, consecutive threads
/// taking consecutive panel rows/lanes. Async staging copies `a` element by
/// element (its panel is transposed in shared memory, which a 16-byte copy
/// cannot do): `lanes` lanes along `k` within one row, then rows — four lanes
/// read one 16-byte global run, a warp covers 8 rows, and `pad_a` keeps those
/// 32 writes on 32 banks. Register staging under `pad`: `a_row_lanes` per row.
fn lane_maps(s: &TileSite, g: &GpuTile) -> [(LaneMap, u64); 2] {
    let e = elem_bytes_of(&s.elem);
    let lanes = GPU_VEC_BYTES / e;
    let (bm, bn, bk, threads) = (g.bm(), g.bn(), g.bk, g.threads());
    let (kv, nv) = (bk / lanes, bn / lanes);
    let streamed = g.stages >= 2;
    let a = if streamed {
        let map = LaneMap::Split {
            l: lanes,
            rows: bm,
            unit: 1,
        };
        (map, bm * bk / threads)
    } else if !vector_a(s) {
        (LaneMap::ColMajor { rows: bm }, bm * bk / threads)
    } else if g.a_row_lanes > 0 && g.a_row_lanes < kv {
        let map = LaneMap::Split {
            l: g.a_row_lanes,
            rows: bm,
            unit: lanes,
        };
        (map, bm * kv / threads)
    } else {
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bm * kv / threads)
    };
    let b = if vector_b(s) {
        let map = LaneMap::RowMajor {
            per_row: nv,
            unit: lanes,
        };
        (map, bk * nv / threads)
    } else {
        let map = LaneMap::RowMajor {
            per_row: bn,
            unit: 1,
        };
        (map, bk * bn / threads)
    };
    [a, b]
}

/// String builder for one kernel — the device twin of [`FnEmit`]'s
/// `line`/`tmp`, kept separate because a kernel shares no state with the host
/// function around the site.
struct Kernel<'s> {
    out: String,
    next: u32,
    /// The index type: `i64`, or `i32` under `idx32` ([`max_index`]).
    idx: &'static str,
    /// `float` or `double`.
    t: String,
    site: &'s TileSite,
    arithmetic: Arithmetic,
    rungs: crate::GpuRungs,
    profile_gpu: Gpu,
    /// S54 `pdl`: open with `griddepcontrol.wait` (and let a dependent launch).
    pdl: bool,
    /// The panel loop's induction values, threaded from its head to its latch.
    async_tail: Option<(String, String, u64)>,
    sync_tail: Option<(String, String)>,
    /// `(gx, h)`: the grid width and the group height of [`raster_block`].
    raster: (u64, u64),
    /// Store the output column-major, `o[j·rows + i]` (S54 `rowfold-layout`):
    /// the order its only reader, a row fold, reads it in.
    col: bool,
    /// S54: the per-cell consumers evaluated in this kernel's store.
    epi: Option<&'s EpiKernel>,
    /// S62 `splitk`: this kernel folds partition `z = ctaid.y / gy_out` of
    /// `splitk` contiguous `k` ranges into workspace slab `z`; 1 = ordered.
    splitk: u64,
    gy_out: u64,
    z: Option<String>,
}

impl Kernel<'_> {
    fn tmp(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    /// `dst = op IDX lhs, rhs`, returning `dst`. In `i32` every value is a
    /// partial sum of an index below 2³¹ ([`max_index`]), so `add`/`mul` carry
    /// `nuw nsw` — what lets LLVM fold a copy's constant into the address.
    fn op(&mut self, code: &str, lhs: &str, rhs: impl std::fmt::Display) -> String {
        let t = self.tmp();
        let flags = match code {
            "add" | "mul" if self.idx == "i32" => " nuw nsw",
            _ => "",
        };
        self.line(format!("{t} = {code}{flags} {} {lhs}, {rhs}", self.idx));
        t
    }

    /// The block's `(column, row)` in the grid: `ctaid.x`, `ctaid.y`, remapped
    /// by [`raster_block`] when the derived group height is above 1.
    fn block_ids(&mut self) -> (String, String) {
        // Under the column-major store (S54 `rowfold-layout`) the grid's
        // fastest axis runs down the rows too (the host swaps `gx`/`gy`): one
        // wave's blocks write consecutive column runs, not 512-byte pieces
        // 16 KB apart. That order is its own raster.
        if self.col {
            let by = self.sreg("ctaid.x");
            return (self.sreg("ctaid.y"), by);
        }
        let bx = self.sreg("ctaid.x");
        let by = self.sreg("ctaid.y");
        // S62 `splitk`: the partition rides above the output rows in `y`
        // (`gy_out` a constant: LLVM folds the division). No raster here —
        // the two are exclusive by derivation (`launch`).
        if self.splitk > 1 {
            let row = self.op("urem", &by, self.gy_out);
            self.z = Some(self.op("udiv", &by, self.gy_out));
            return (bx, row);
        }
        let (gx, h) = self.raster;
        if h < 2 {
            return (bx, by);
        }
        self.line(format!("; raster: {h} block-rows per group"));
        let (c, r) = raster_block(self, &bx, &by, gx, h);
        (self.opaque(&c), self.opaque(&r))
    }

    /// `v` through a register move LLVM cannot see into. The remap bounds a
    /// block's row and column, so InstCombine turns the kernel's address adds
    /// into `or disjoint`, which NVPTX does not fold into immediate offsets:
    /// the H100 FMA tile went from 127 to 145 registers and one block per SM
    /// (`l2-raster-h100-pcie.txt`, `off-l2-gate`). Unbounded, the addresses
    /// are the row-major kernel's.
    fn opaque(&mut self, v: &str) -> String {
        let (ty, bits, c) = match self.idx {
            "i32" => ("i32", 32, "r"),
            _ => ("i64", 64, "l"),
        };
        let t = self.tmp();
        self.line(format!(
            "{t} = call {ty} asm \"mov.b{bits} $0, $1;\", \"={c},{c}\"({ty} {v})"
        ));
        t
    }

    fn sreg(&mut self, reg: &str) -> String {
        let r32 = self.tmp();
        self.line(format!("{r32} = call i32 @llvm.nvvm.read.ptx.sreg.{reg}()"));
        if self.idx == "i32" {
            return r32;
        }
        let r = self.tmp();
        self.line(format!("{r} = zext i32 {r32} to i64"));
        r
    }

    /// `ptr addrspace(AS) gep T, base, idx`. An `i32` index is zero-extended
    /// (it is non-negative): a GEP would sign-extend it, and copy 0's bare
    /// index then extends differently from copy `p`'s `index + constant`.
    fn gep(&mut self, space: u32, base: &str, idx: &str) -> String {
        let t = self.t.clone();
        self.gep_ty(space, &t, base, idx)
    }

    /// [`Self::gep`] over elements of type `ty`.
    fn gep_ty(&mut self, space: u32, ty: &str, base: &str, idx: &str) -> String {
        let idx = if self.idx == "i32" {
            let z = self.tmp();
            self.line(format!("{z} = zext nneg i32 {idx} to i64"));
            z
        } else {
            idx.to_string()
        };
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace({space}) {base}, i64 {idx}"
        ));
        p
    }

    /// The group values' types: the accumulator's, then each call's result.
    fn epi_tys(&self, epi: &EpiKernel) -> Vec<String> {
        std::iter::once(self.t.clone())
            .chain(epi.calls.iter().map(|c| c.ret.clone()))
            .collect()
    }

    /// The epilogue's extra parameters (S54): operand pointers `%x<i>`, then
    /// one output pointer `%y<j>` per stored member. Empty without one.
    fn epi_params(&self, align: u64) -> String {
        let Some(epi) = self.epi else {
            return String::new();
        };
        let mut s = String::new();
        for i in 0..epi.caps {
            s.push_str(&format!(", ptr addrspace(1) readonly %x{i}"));
        }
        let stored = epi.calls.iter().filter(|c| c.store).count();
        for j in 0..stored {
            s.push_str(&format!(", ptr addrspace(1) noalias align {align} %y{j}"));
        }
        s
    }

    /// Entry-block state of the epilogue: each operand pointer as the generic
    /// pointer a host body takes, and one one-cell buffer per `Cell` field —
    /// in the entry block, so inlining the body leaves a value in a register.
    fn epi_entry(&mut self) {
        let Some(epi) = self.epi else {
            return;
        };
        for i in 0..epi.caps {
            self.line(format!(
                "%xg{i} = addrspacecast ptr addrspace(1) %x{i} to ptr"
            ));
        }
        let tys = self.epi_tys(epi);
        for (k, c) in epi.calls.iter().enumerate() {
            for (f, a) in c.fields.iter().enumerate() {
                if let EpiArg::Cell(v) = *a {
                    self.line(format!("%cell{k}_{f} = alloca {}", tys[v]));
                }
            }
        }
    }

    /// The group's results at one cell (S54): each member's body called on
    /// the accumulator `acc` of output cell `cell` (an index value), in chain
    /// order. The bodies are the host's (or their `_cell` variants): the
    /// values are the host loop's, bit for bit.
    fn epi_values(&mut self, acc: &str, cell: &str) -> Vec<String> {
        let epi = self.epi.expect("an epilogue");
        let t32 = if self.idx == "i32" {
            cell.to_string()
        } else {
            let x = self.tmp();
            self.line(format!("{x} = trunc i64 {cell} to i32"));
            x
        };
        let tys = self.epi_tys(epi);
        let mut vals = vec![acc.to_string()];
        for (k, c) in epi.calls.iter().enumerate() {
            let arg = if c.arg.starts_with('{') {
                let mut agg = "poison".to_string();
                for (f, a) in c.fields.iter().enumerate() {
                    let (ty, v) = match *a {
                        EpiArg::Value(i) => (tys[i].clone(), vals[i].clone()),
                        EpiArg::Cell(i) => {
                            let p = format!("%cell{k}_{f}");
                            self.line(format!("store {} {}, ptr {p}", tys[i], vals[i]));
                            ("ptr".to_string(), p)
                        }
                        EpiArg::Cap(i) => ("ptr".to_string(), format!("%xg{i}")),
                        EpiArg::Index => ("i32".to_string(), t32.clone()),
                    };
                    let n = self.tmp();
                    self.line(format!("{n} = insertvalue {} {agg}, {ty} {v}, {f}", c.arg));
                    agg = n;
                }
                agg
            } else {
                let [EpiArg::Value(i)] = c.fields[..] else {
                    unreachable!("a scalar body input is one group value")
                };
                vals[i].clone()
            };
            let r = self.tmp();
            self.line(format!(
                "{r} = call {} @{}({} {arg})",
                c.ret, c.callee, c.arg
            ));
            vals.push(r);
        }
        vals.split_off(1)
    }

    /// Store `accs`, the accumulators of the consecutive output cells from
    /// index `oi`, through the epilogue: the site's output when stored, and
    /// each stored member's result — one vector store per output when
    /// `accs.len() > 1`, as the plain store.
    fn store_cells(&mut self, accs: &[String], oi: &str) {
        let epi = self.epi.expect("an epilogue");
        let mut per: Vec<Vec<String>> = vec![Vec::new(); epi.calls.len()];
        for (j, acc) in accs.iter().enumerate() {
            let cell = self.plus(oi.to_string(), j as u64);
            for (k, v) in self.epi_values(acc, &cell).into_iter().enumerate() {
                per[k].push(v);
            }
        }
        let mut outs: Vec<(String, String, Vec<String>)> = Vec::new();
        if epi.store_out {
            outs.push(("%o".to_string(), self.t.clone(), accs.to_vec()));
        }
        let mut y = 0;
        for (k, c) in epi.calls.iter().enumerate() {
            if c.store {
                outs.push((format!("%y{y}"), c.ret.clone(), std::mem::take(&mut per[k])));
                y += 1;
            }
        }
        let n = accs.len();
        for (p, ty, vals) in outs {
            let e = if ty == "double" { 8 } else { 4 };
            let (sty, v) = if n == 1 {
                (ty.clone(), vals[0].clone())
            } else {
                let vt = format!("<{n} x {ty}>");
                let mut vec = "poison".to_string();
                for (j, x) in vals.iter().enumerate() {
                    let nv = self.tmp();
                    self.line(format!(
                        "{nv} = insertelement {vt} {vec}, {ty} {x}, i64 {j}"
                    ));
                    vec = nv;
                }
                (vt, vec)
            };
            let gp = self.gep_ty(1, &ty, &p, oi);
            self.line(format!(
                "store {sty} {v}, ptr addrspace(1) {gp}, align {}",
                n * e
            ));
        }
    }

    /// `v + c`, or `v` itself when `c` is 0.
    fn plus(&mut self, v: String, c: u64) -> String {
        if c == 0 { v } else { self.op("add", &v, c) }
    }

    /// S62 `splitk`: `v + z·scale` in a split kernel — partition `z`'s
    /// offset along `k` (`scale = kc·ck`) or into the workspace
    /// (`scale = rows·c`) — and `v` itself in the ordered one.
    fn at_z(&mut self, v: String, scale: u64) -> String {
        match self.z.clone() {
            Some(z) => {
                let off = self.op("mul", &z, scale);
                self.op("add", &v, &off)
            }
            None => v,
        }
    }

    /// The fold's seed: partition 0 starts from the site's seed, every other
    /// partition from `0.0` (ADR-0040's partial folds), so the reduce adds
    /// the seed exactly once.
    fn split_seed(&mut self, seed: &str) -> String {
        let Some(z) = self.z.clone() else {
            return seed.to_string();
        };
        let (t, ix) = (self.t.clone(), self.idx);
        let (first, s) = (self.tmp(), self.tmp());
        self.line(format!("{first} = icmp eq {ix} {z}, 0"));
        self.line(format!("{s} = select i1 {first}, {t} {seed}, {t} 0.0"));
        s
    }

    /// Every staged copy's `(global index at k0 = 0, + constant, shared
    /// index, + constant)`; `at` turns panel coordinates into the two indices,
    /// `off` a coordinate delta into the two constants. Under `affine` a copy
    /// the lane map proves periodic ([`affine_delta`]) is copy 0's indices plus
    /// constants; otherwise (and on the OFF arm) each copy has its own chain.
    fn copies(
        &mut self,
        map: LaneMap,
        n: u64,
        threads: u64,
        tid: &str,
        at: impl Fn(&mut Self, &str, &str) -> (String, String),
        off: impl Fn(u64, u64) -> (u64, u64),
    ) -> Vec<(String, u64, String, u64)> {
        let mut base: Option<(String, String)> = None;
        let mut out = Vec::new();
        for p in 0..n {
            let delta = self.rungs.affine.then(|| affine_delta(map, threads, p));
            if let Some(Some((d0, d1))) = delta {
                let (g, s) = match &base {
                    Some(b) => b.clone(),
                    None => {
                        let (r, c) = map.coords(self, &tid.to_string());
                        base.insert(at(self, &r, &c)).clone()
                    }
                };
                let (go, so) = off(d0, d1);
                out.push((g, go, s, so));
                continue;
            }
            let li = self.op("add", tid, p * threads);
            let (r, c) = map.coords(self, &li);
            let (g, s) = at(self, &r, &c);
            out.push((g, 0, s, 0));
        }
        out
    }

    /// One accumulation step, in the oracle's operand order:
    /// `acc ⊕ (a ⊗ b)` with each side placed as the source wrote it.
    fn step(&mut self, acc: &str, a: &str, b: &str) -> String {
        let t = self.t.clone();
        let r = self.tmp();
        debug_assert!(matches!(
            self.arithmetic,
            Arithmetic::SeparateMulAdd | Arithmetic::OrderedFma
        ));
        if self.arithmetic == Arithmetic::OrderedFma {
            let bits = if t == "float" { 32 } else { 64 };
            self.line(format!(
                "{r} = call {t} @llvm.fma.f{bits}({t} {a}, {t} {b}, {t} {acc})"
            ));
            return r;
        }
        let (x, y) = if self.site.mul_a_first {
            (a, b)
        } else {
            (b, a)
        };
        let p = self.tmp();
        self.line(format!("{p} = fmul {t} {x}, {y}"));
        let (x, y) = if self.site.add_acc_first {
            (acc, p.as_str())
        } else {
            (p.as_str(), acc)
        };
        self.line(format!("{r} = fadd {t} {x}, {y}"));
        r
    }

    /// `blocks` > 0 also emits `.minnctapersm`: the geometry assumed that many
    /// resident blocks per SM when it sized the register tile and the panels,
    /// so ptxas is held to the register budget that makes it true (without
    /// it, sm_90 allocated 166 registers and one block fit — S52 v1).
    fn header(&mut self, name: &str, threads: u64, align: u64, blocks: u64) {
        let min_blocks = if blocks > 0 {
            format!(" \"nvvm.minctasm\"=\"{blocks}\"")
        } else {
            String::new()
        };
        self.out.push_str(&format!(
            "define ptx_kernel void @{name}(\
             ptr addrspace(1) noalias readonly align {align} %a, \
             ptr addrspace(1) noalias readonly align {align} %b, \
             ptr addrspace(1) noalias align {align} %o{}) \"nvvm.maxntid\"=\"{threads}\"{min_blocks} {{\n\
             entry:\n",
            self.epi_params(align)
        ));
        self.epi_entry();
        // A dependent launch may start while its producer drains: nothing is
        // read before the producer's grid is complete. Then let this grid's
        // own dependent start. Both are no-ops on a plain launch.
        if self.pdl {
            self.asm("griddepcontrol.wait;");
            self.asm("griddepcontrol.launch_dependents;");
        }
    }

    /// One thread per output cell, `k` walked in order — the correctness path
    /// for any affine site the tiled geometry does not cover.
    fn naive(&mut self, name: &str, threads: u64) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        self.header(name, threads, e, 0);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        // S62 `splitk`: the grid's `y` is the partition.
        if self.splitk > 1 {
            self.z = Some(self.sreg("ctaid.y"));
        }
        let cell0 = self.op("mul", &bx, threads);
        let cell = self.op("add", &cell0, &tid);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult {ix} {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let base = |k: &mut Self, r: &mapal_ir::TileRead| {
            let x = k.op("mul", &i, r.ci);
            let y = k.op("mul", &lane, r.clane);
            let z = k.op("add", &x, &y);
            k.op("add", &z, r.base)
        };
        // A split kernel folds `kc` steps from `z·kc`: the same loop over
        // operands moved `z·kc` along `k` (in bounds: `z·kc + kc ≤ k`).
        let kc = s.k / self.splitk;
        let abase = base(self, &s.a);
        let abase = self.at_z(abase, kc * s.a.ck);
        let bbase = base(self, &s.b);
        let bbase = self.at_z(bbase, kc * s.b.ck);
        let seed = self.split_seed(&seed);
        self.line("br label %kloop");
        self.out.push_str("kloop:\n");
        let (kv, acc, kn) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{kv} = phi {ix} [0, %body], [{kn}, %kloop]"));
        // The back-edge value is the step's result, known only after the body
        // is emitted; patched in below (the marker cannot collide with a `%v`).
        self.line(format!(
            "{acc} = phi {t} [{seed}, %body], [@ACC_NEXT@, %kloop]"
        ));
        let load = |k: &mut Self, arr: &str, base: &str, ck: u64| {
            let off = k.op("mul", &kv, ck);
            let idx = k.op("add", base, &off);
            let p = k.gep(1, arr, &idx);
            let v = k.tmp();
            k.line(format!("{v} = load {t}, ptr addrspace(1) {p}, align {e}"));
            v
        };
        let av = load(self, "%a", &abase, s.a.ck);
        let bv = load(self, "%b", &bbase, s.b.ck);
        let accn = self.step(&acc, &av, &bv);
        self.out = self.out.replace("@ACC_NEXT@", &accn);
        self.line(format!("{kn} = add {ix} {kv}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq {ix} {kn}, {kc}"));
        self.line(format!("br i1 {fin}, label %store, label %kloop"));
        self.out.push_str("store:\n");
        let cell = if self.col {
            let jr = self.op("mul", &lane, s.rows);
            self.op("add", &jr, &i)
        } else {
            cell
        };
        // A split kernel stores its partial into workspace slab `z`.
        let cell = self.at_z(cell, s.rows * s.c);
        if self.epi.is_some() {
            self.store_cells(std::slice::from_ref(&accn), &cell);
        } else {
            let p = self.gep(1, "%o", &cell);
            self.line(format!("store {t} {accn}, ptr addrspace(1) {p}, align {e}"));
        }
        self.line("br label %done");
        self.out.push_str("done:\n  ret void\n}\n");
    }

    /// Issue panel `tile` (an i64 value) into stage `stage` (an i64 value):
    /// one 16-byte `cp.async` per `(is_a, global index at k0 = 0, shared
    /// word in stage 0)` copy.
    fn issue16(
        &mut self,
        copies: &[(bool, String, String)],
        tile: &str,
        stage: &str,
        bk: u64,
        stage_words: u64,
    ) {
        let s = self.site;
        let k0 = self.op("mul", tile, bk);
        let ka = self.op("mul", &k0, s.a.ck);
        let kb = self.op("mul", &k0, s.b.ck);
        let so = self.op("mul", stage, stage_words);
        for (is_a, gsrc, sdst) in copies {
            let (arr, off) = if *is_a { ("%a", &ka) } else { ("%b", &kb) };
            let gi = self.op("add", gsrc, off);
            let gp = self.gep(1, arr, &gi);
            let si = self.op("add", sdst, &so);
            let sp = self.gep(3, "@mapal_smem", &si);
            self.line(format!(
                "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
            ));
        }
    }

    /// `cvt.rn.tf32.f32`: round to nearest, ties to even — ADR-0039's
    /// `to_tf32`. `wgmma` itself truncates the low 13 bits (PTX ISA), so an
    /// operand it reads must already be rounded. `sideeffect` pins the write
    /// before the `wgmma.fence` that orders it.
    fn round_tf32(&mut self, x: &str, pinned: bool) -> String {
        let r = self.tmp();
        let se = if pinned { " sideeffect" } else { "" };
        self.line(format!(
            "{r} = call i32 asm{se} \"cvt.rn.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
        ));
        r
    }

    fn asm(&mut self, text: &str) {
        self.line(format!(
            "call void asm sideeffect \"{text}\", \"~{{memory}}\"()"
        ));
    }

    /// **Hopper TF32 through `wgmma` (S53 `wgmma-hopper`).** The site's
    /// product is computed transposed, `cᵀ = bᵀ·aᵀ` ([`GpuWgmmaTile`]): warpgroup
    /// `wg` of block `(bx, by)` owns `c` columns `bx·bn + 64·wg ..+64` (the
    /// instruction's M) × rows `by·n ..+n` (its N).
    ///
    /// - `a` panels land by `cp.async`, K-major in 128-byte swizzle rows (the
    ///   descriptor's mode 1; SBO one 8-row atom). `wgmma` reads shared memory
    ///   as it is, truncating, so each thread rounds the chunks it copied once
    ///   (RNE) and fences them to the async proxy before the panel barrier.
    ///   That pass costs a third of the kernel at 4096 (S53 diag4: 0.89 ms,
    ///   0.60 without it), and two ways around it measured worse on H100
    ///   PCIe (nsys 4096, this kernel 0.80 ms): staging `a` through registers,
    ///   rounded on the way, 0.94 ms (latency of one panel in flight, not
    ///   two); two MMA groups in flight (`wait_group 1`, doubled fragments,
    ///   which moves the derivation to 128 × 128 at bk 64) 0.98 ms, and it
    ///   failed the TF32 value gate at 1024 with four stages.
    ///   `benches/results/gpu/kernels/optimization-ladder/wgmma-hopper-h100-pcie-run{2,3}-*.txt`.
    /// - `b` panels land row-major, rows padded; each thread loads its `bᵀ`
    ///   fragments (`a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4) of the
    ///   warp's 16 rows, PTX ISA m64nNk8) and rounds them in the load.
    /// - Pipeline: panel `it + stages − 2` is copied into the stage panel
    ///   `it − 2` used — the group issued at `it − 1` may still be reading
    ///   its own — and the wait for that group sits after the barrier, right
    ///   before its fragment registers are rewritten.
    ///
    /// Accumulator `d[4j + 2h + e]` holds `cᵀ(16w + g + 8h, 8j + 2t + e)`;
    /// each warp-wide store fills four whole 32-byte sectors of `c`.
    fn wgmma(&mut self, name: &str, g: GpuWgmmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, ldb, n) = (g.bm(), g.bn(), g.bk, g.ldb(), g.n);
        let (stage_words, a_words) = (g.stage_words(), g.a_words());
        let nk = s.k / bk;
        let dist = g.stages - 2;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";

        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        let wg = self.op("udiv", &tid, WARPGROUP_WARPS * warp);
        let warp_id = self.op("udiv", &tid, warp);
        let w = self.op("urem", &warp_id, WARPGROUP_WARPS);
        let lane = self.op("urem", &tid, warp);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = (n / 2) as usize;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }

        // Copies, in words: `a` row m, 16-byte chunk kc — atom column
        // ka = kc / 8, chunk c = kc % 8 — lands at ka·bm·128 + m·128 +
        // (c ⊕ m%8)·16 bytes (`Swizzle<3,4,3>`): eight lanes read one whole
        // 128-byte line and write eight distinct bank groups. `b` as the mma
        // kernel stages it.
        let kcs = bk * 4 / GPU_VEC_BYTES;
        let row_chunks = WGMMA_SWIZZLE_BYTES / GPU_VEC_BYTES;
        let row_w = WGMMA_SWIZZLE_BYTES / 4;
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        for p in 0..bm * kcs / threads {
            let li = self.op("add", &tid, p * threads);
            let kc = self.op("urem", &li, kcs);
            let m = self.op("udiv", &li, kcs);
            let grow = self.op("add", &row0, &m);
            let gr = self.op("mul", &grow, s.a.ci);
            let gk = self.op("mul", &kc, 4);
            let g0 = self.op("add", &gr, &gk);
            let gsrc = self.op("add", &g0, s.a.base);
            let ka = self.op("udiv", &kc, row_chunks);
            let c = self.op("urem", &kc, row_chunks);
            let r8 = self.op("urem", &m, WGMMA_ATOM_ROWS);
            let sw = self.op("xor", &c, &r8);
            let sa = self.op("mul", &ka, bm * row_w);
            let sr = self.op("mul", &m, row_w);
            let sc = self.op("mul", &sw, 4);
            let sdst = self.op("add", &sa, &sr);
            let sdst = self.op("add", &sdst, &sc);
            copies.push((true, gsrc, sdst));
        }
        let bc = bn / 4;
        for p in 0..bk * bc / threads {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, a_words);
            copies.push((false, gsrc, sdst));
        }
        for st in 0..dist.min(nk) {
            self.issue16(&copies, &st.to_string(), &st.to_string(), bk, stage_words);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        for _ in nk..dist {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        // This thread's bᵀ fragment origin inside a stage: row k = t, column
        // n = 64·wg + 16·w + g.
        let fr = self.op("mul", &t4, ldb);
        let fc = self.op("mul", &wg, WGMMA_M);
        let fw = self.op("mul", &w, WGMMA_M / WARPGROUP_WARPS);
        let fo = self.op("add", &fr, &fc);
        let fo = self.op("add", &fo, &fw);
        let fo = self.op("add", &fo, &grp);
        let b_org = self.op("add", &fo, a_words);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %issued]"));
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            dist - 1
        ));
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        // Round this thread's own `a` chunks of panel `it`, once.
        for (_, _, sdst) in copies.iter().filter(|c| c.0) {
            let si = self.op("add", sdst, &cbase);
            let sp = self.gep(3, smem, &si);
            let v = self.tmp();
            self.line(format!(
                "{v} = load <4 x float>, ptr addrspace(3) {sp}, align 16"
            ));
            let mut r = "poison".to_string();
            for j in 0..4 {
                let x = self.tmp();
                self.line(format!("{x} = extractelement <4 x float> {v}, i64 {j}"));
                let y = self.round_tf32(&x, false);
                let nr = self.tmp();
                self.line(format!(
                    "{nr} = insertelement <4 x i32> {r}, i32 {y}, i64 {j}"
                ));
                r = nr;
            }
            self.line(format!(
                "store <4 x i32> {r}, ptr addrspace(3) {sp}, align 16"
            ));
        }
        self.asm("fence.proxy.async.shared::cta;");
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, dist);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        self.issue16(&copies, &nxt, &nst, bk, stage_words);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        // The previous group done: its fragment registers may be rewritten.
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let fb = self.op("add", &cbase, &b_org);
        let mut frags = Vec::new();
        for kk in 0..bk / WGMMA_K {
            let mut f = Vec::new();
            for off in [0, 8, 4 * ldb, 4 * ldb + 8] {
                let i = self.op("add", &fb, kk * WGMMA_K * ldb + off);
                let p = self.gep(3, smem, &i);
                let x = self.tmp();
                self.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
                f.push(self.round_tf32(&x, true));
            }
            frags.push(f);
        }
        self.asm("wgmma.fence.sync.aligned;");
        // Descriptor of this stage's `a` panel; k8 step `kk` starts in atom
        // column `kk·32 / 128`, `kk·32 % 128` bytes into its swizzled rows (the
        // hardware applies the XOR to the address it forms).
        let ap = self.gep(3, smem, &cbase);
        let a32 = self.tmp();
        self.line(format!("{a32} = ptrtoint ptr addrspace(3) {ap} to i32"));
        let a64 = self.tmp();
        self.line(format!("{a64} = zext i32 {a32} to i64"));
        let enc = self.op("lshr", &a64, 4);
        let enc = self.op("and", &enc, 0x3fff);
        let desc0 = self.op("or", &enc, g.desc_fields());
        let k_off = |kk: u64| {
            let b = kk * WGMMA_K * 4;
            (b / WGMMA_SWIZZLE_BYTES * bm * WGMMA_SWIZZLE_BYTES + b % WGMMA_SWIZZLE_BYTES) >> 4
        };
        let sty = format!("{{ {} }}", vec!["float"; accs].join(", "));
        let mut text = format!("wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32 {{");
        text.push_str(
            &(0..accs)
                .map(|i| format!("${i}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push_str(&format!(
            "}}, {{${},${},${},${}}}, ${}, 1, 1, 1;",
            accs,
            accs + 1,
            accs + 2,
            accs + 3,
            accs + 4
        ));
        let mut cons = vec!["=f".to_string(); accs];
        cons.extend(["r", "r", "r", "r", "l"].map(String::from));
        cons.extend((0..accs).map(|i| i.to_string()));
        let cons = cons.join(",");
        for (kk, f) in frags.iter().enumerate() {
            let desc = self.op("add", &desc0, k_off(kk as u64));
            let mut args = format!(
                "i32 {}, i32 {}, i32 {}, i32 {}, i64 {desc}",
                f[0], f[1], f[2], f[3]
            );
            let mut slots = Vec::new();
            for q in 0..accs {
                let p = acc_slot(self, q);
                let v = self.tmp();
                self.line(format!("{v} = load float, ptr {p}, align 4"));
                args.push_str(&format!(", float {v}"));
                slots.push(p);
            }
            let r = self.tmp();
            self.line(format!(
                "{r} = call {sty} asm sideeffect \"{text}\", \"{cons}\"({args})"
            ));
            for (q, p) in slots.iter().enumerate() {
                let d = self.tmp();
                self.line(format!("{d} = extractvalue {sty} {r}, {q}"));
                self.line(format!("store float {d}, ptr {p}, align 4"));
            }
        }
        self.asm("wgmma.commit_group.sync.aligned;");
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        self.out.push_str("done:\n");
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let er = self.op("mul", &t4, 2);
        let er = self.op("add", &er, &row0);
        let er = self.op("mul", &er, s.c);
        let ec = self.op("add", &col0, &fc);
        let ec = self.op("add", &ec, &fw);
        let ec = self.op("add", &ec, &grp);
        let ebase = self.op("add", &er, &ec);
        for j in 0..n / 8 {
            for h in 0..2u64 {
                for e in 0..2u64 {
                    let q = (4 * j + 2 * h + e) as usize;
                    let p = acc_slot(self, q);
                    let x = self.tmp();
                    self.line(format!("{x} = load float, ptr {p}, align 4"));
                    let oi = self.op("add", &ebase, (8 * j + e) * s.c + 8 * h);
                    if self.epi.is_some() {
                        self.store_cells(&[x], &oi);
                        continue;
                    }
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store float {x}, ptr addrspace(1) {op}, align 4"));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// The tensor-core kernel (TF32 face): block `(bx, by)` owns a `bm × bn`
    /// output tile; warp `(wr, wc)` a `wm × wn` tile of `mt × nt` m16n8 tiles
    /// whose f32 accumulators live in registers. The reduction is staged in
    /// `stages` shared-memory buffers filled by `cp.async` — panel `it + stages
    /// - 1` is in flight while panel `it` is consumed — and each `bk` panel is
    /// `bk / 8` steps of `mma.sync.m16n8k8`, operands rounded with `cvt.rna`.
    ///
    /// Fragment layouts are the PTX ISA's for m16n8k8 (`group = lane / 4`,
    /// `t = lane % 4`): A `a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4);
    /// B `b0, b1` = (k t, n g), (k t+4, n g); C `c0..c3` = (g, 2t), (g, 2t+1),
    /// (g+8, 2t), (g+8, 2t+1).
    fn mma(&mut self, name: &str, g: GpuMmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let stage_words = g.stage_words();
        let nk = s.k / bk;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";

        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        let warp_id = self.op("udiv", &tid, warp);
        let lane = self.op("urem", &tid, warp);
        let wr = self.op("udiv", &warp_id, g.warps_x);
        let wc = self.op("urem", &warp_id, g.warps_x);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = mt * nt * 4;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }

        // cp.async coordinates: (global index at k0 = 0, shared word in stage 0),
        // one per 16-byte chunk this thread copies, A then B.
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        let (ac, bc) = (bk / 4, bn / 4);
        for p in 0..bm * bk / 4 / threads {
            let li = self.op("add", &tid, p * threads);
            let r = self.op("udiv", &li, ac);
            let kc0 = self.op("urem", &li, ac);
            let kc = self.op("mul", &kc0, 4);
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let g0 = self.op("add", &gr, &kc);
            let gsrc = self.op("add", &g0, s.a.base);
            let sr = self.op("mul", &r, lda);
            let sdst = self.op("add", &sr, &kc);
            copies.push((true, gsrc, sdst));
        }
        for p in 0..bk * bn / 4 / threads {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, bm * lda);
            copies.push((false, gsrc, sdst));
        }
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            k.issue16(&copies, tile, stage, bk, stage_words)
        };
        for st in 0..(g.stages - 1).min(nk) {
            issue(self, &st.to_string(), &st.to_string());
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        for _ in nk..g.stages - 1 {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        // Per-thread fragment origins inside a stage.
        let a_r = self.op("mul", &wr, g.wm());
        let a_r = self.op("add", &a_r, &grp);
        let a_r = self.op("mul", &a_r, lda);
        let a_org = self.op("add", &a_r, &t4);
        let b_r = self.op("mul", &t4, ldb);
        let b_c = self.op("mul", &wc, g.wn());
        let b_o = self.op("add", &b_r, &b_c);
        let b_o = self.op("add", &b_o, &grp);
        let b_org = self.op("add", &b_o, bm * lda);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %issued]"));
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            g.stages - 2
        ));
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, g.stages - 1);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        issue(self, &nxt, &nst);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        let a_base = self.op("add", &cbase, &a_org);
        let b_base = self.op("add", &cbase, &b_org);
        let lds_tf32 = |k: &mut Self, base: &str, off: u64| {
            let i = k.op("add", base, off);
            let p = k.gep(3, smem, &i);
            let x = k.tmp();
            k.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
            ));
            r
        };
        // `ldmatrix.x4` (rung `ldmatrix`): the four 8×8 b16 matrices of one
        // m16k8 TF32 fragment — rows 0–7 / 8–15 × k 0–3 / 4–7 — are exactly
        // a0..a3 when each 32-bit element is read as a b16 pair. Lane `l`
        // supplies the row address of matrix `l / 8`, row `l % 8`.
        let lm_base = if self.rungs.ldmatrix {
            let q = self.op("udiv", &lane, 8);
            let i = self.op("urem", &lane, 8);
            let qr = self.op("urem", &q, 2);
            let qk = self.op("udiv", &q, 2);
            let r = self.op("mul", &qr, 8);
            let r = self.op("add", &r, &i);
            let wrow = self.op("mul", &wr, g.wm());
            let r = self.op("add", &r, &wrow);
            let r = self.op("mul", &r, lda);
            let kq = self.op("mul", &qk, 4);
            let o = self.op("add", &r, &kq);
            Some(self.op("add", &cbase, &o))
        } else {
            None
        };
        for kk in 0..bk / MMA_K {
            let k8 = kk * MMA_K;
            let mut af = Vec::new();
            for mi in 0..mt as u64 {
                let r = mi * MMA_M * lda + k8;
                if let Some(lb) = &lm_base {
                    let i = self.op("add", lb, r);
                    let p = self.gep(3, smem, &i);
                    let a32 = self.tmp();
                    self.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
                    let m = self.tmp();
                    self.line(format!(
                        "{m} = call {{ i32, i32, i32, i32 }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0,$1,$2,$3}}, [$4];\", \"=r,=r,=r,=r,r\"(i32 {a32})"
                    ));
                    let mut q = Vec::new();
                    for j in 0..4 {
                        let x = self.tmp();
                        self.line(format!(
                            "{x} = extractvalue {{ i32, i32, i32, i32 }} {m}, {j}"
                        ));
                        let r = self.tmp();
                        self.line(format!(
                            "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,r\"(i32 {x})"
                        ));
                        q.push(r);
                    }
                    af.push([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
                    continue;
                }
                af.push([
                    lds_tf32(self, &a_base, r),
                    lds_tf32(self, &a_base, r + 8 * lda),
                    lds_tf32(self, &a_base, r + 4),
                    lds_tf32(self, &a_base, r + 8 * lda + 4),
                ]);
            }
            let mut bf = Vec::new();
            for ni in 0..nt as u64 {
                let c = k8 * ldb + ni * MMA_N;
                bf.push([
                    lds_tf32(self, &b_base, c),
                    lds_tf32(self, &b_base, c + 4 * ldb),
                ]);
            }
            for mi in 0..mt {
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4;
                    let mut cur = Vec::new();
                    for q in 0..4 {
                        let p = acc_slot(self, base + q);
                        let v = self.tmp();
                        self.line(format!("{v} = load float, ptr {p}, align 4"));
                        cur.push((p, v));
                    }
                    let r = self.tmp();
                    let [a0, a1, a2, a3] = &af[mi];
                    let [b0, b1] = &bf[ni];
                    self.line(format!(
                        "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {a0}, i32 {a1}, i32 {a2}, i32 {a3}, i32 {b0}, i32 {b1}, float {}, float {}, float {}, float {})",
                        cur[0].1, cur[1].1, cur[2].1, cur[3].1
                    ));
                    for (q, (p, _)) in cur.iter().enumerate() {
                        let d = self.tmp();
                        self.line(format!(
                            "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                        ));
                        self.line(format!("store float {d}, ptr {p}, align 4"));
                    }
                }
            }
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        // C fragments: (g, 2t..2t+1) and (g+8, 2t..2t+1) of every m16n8 tile.
        self.out.push_str("done:\n");
        let orow = self.op("mul", &wr, g.wm());
        let orow = self.op("add", &orow, &row0);
        let orow = self.op("add", &orow, &grp);
        let ocol = self.op("mul", &wc, g.wn());
        let ocol = self.op("add", &ocol, &col0);
        let t2 = self.op("mul", &t4, 2);
        let ocol = self.op("add", &ocol, &t2);
        for mi in 0..mt {
            for half in 0..2u64 {
                let r = self.op("add", &orow, mi as u64 * MMA_M + half * 8);
                let rr = self.op("mul", &r, s.c);
                let rc = self.op("add", &rr, &ocol);
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4 + half as usize * 2;
                    if self.epi.is_some() {
                        let mut xs = Vec::new();
                        for j in 0..2 {
                            let p = acc_slot(self, base + j);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            xs.push(x);
                        }
                        let oi = self.op("add", &rc, ni as u64 * MMA_N);
                        self.store_cells(&xs, &oi);
                        continue;
                    }
                    let mut vec = "poison".to_string();
                    for j in 0..2 {
                        let p = acc_slot(self, base + j);
                        let x = self.tmp();
                        self.line(format!("{x} = load float, ptr {p}, align 4"));
                        let nv = self.tmp();
                        self.line(format!(
                            "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                        ));
                        vec = nv;
                    }
                    let oi = self.op("add", &rc, ni as u64 * MMA_N);
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!(
                        "store <2 x float> {vec}, ptr addrspace(1) {op}, align 8"
                    ));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// The tiled kernel: block `(bx, by)` owns the `bm × bn` output tile at
    /// `(by·bm, bx·bn)`; thread `(trow, tcol)` owns a `tm × tn` register tile
    /// inside it. Per `bk` step of the reduction, the block stages
    ///
    /// - `A[rows][k0 .. k0+bk]` **transposed** into `As[kk][m]`, so a thread's
    ///   `tm` rows at one `kk` are contiguous (one vector read each), and
    /// - `B[k0 .. k0+bk][cols]` as `Bs[kk][n]`,
    ///
    /// with `ld.global.nc.v4`, then a barrier (the morphism that completes the
    /// `g2s` transmission — plan-s41 §1.4), then every thread folds `bk` steps
    /// into its registers, then a barrier before the panels are overwritten.
    fn tiled(&mut self, name: &str, g: GpuTile) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        let lanes = GPU_VEC_BYTES / e;
        let vt = format!("<{lanes} x {t}>");
        let (bm, bn, bk, tm, tn, threads) = (g.bm(), g.bn(), g.bk, g.tm, g.tn, g.threads());
        // S62 `splitk`: the `k` extent this kernel folds — partition `z`'s
        // whole panels (`launch` derives `S` so `bk | k / S`), its copies'
        // global sources moved `z·kext` along `k`.
        let kext = s.k / self.splitk;

        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        // Warp shape (S52 rung `warp`): lay a warp's 32 lanes out as
        // `warp / wx` rows × `wx` columns of threads, `wx` = the lanes one
        // shared-memory wavefront serves with vector loads (128 B / 16 B = 8).
        // A warp then reads 4 + 8 distinct thread fragments per `k` step
        // instead of 2 + 16 — two-thirds of the bytes. Off arm / a block too
        // narrow for it: row-major threads.
        let gpu = self.profile_gpu;
        let w = GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES;
        // S54 `rowfold-layout`: a column-major store runs a warp's fastest
        // lanes down the rows — the row-major rule mirrored, so one store
        // instruction still writes whole 128-byte lines (`col_store`). The
        // shared-memory side is symmetric: `w` fragments of one panel, the
        // rest of the other.
        let (wx, wy) = if self.col {
            (gpu.warp / w, w)
        } else {
            (w, gpu.warp / w)
        };
        let (trow, tcol) = if self.rungs.warp && g.tx % wx == 0 && g.ty % wy == 0 && g.tx > wx {
            let warp = self.op("udiv", &tid, gpu.warp);
            let lane = self.op("urem", &tid, gpu.warp);
            let warps_x = g.tx / wx;
            let wr = self.op("udiv", &warp, warps_x);
            let wc = self.op("urem", &warp, warps_x);
            let (lr, lc) = if self.col {
                (self.op("urem", &lane, wy), self.op("udiv", &lane, wy))
            } else {
                (self.op("udiv", &lane, wx), self.op("urem", &lane, wx))
            };
            let r0 = self.op("mul", &wr, wy);
            let c0 = self.op("mul", &wc, wx);
            (self.op("add", &r0, &lr), self.op("add", &c0, &lc))
        } else if self.col {
            (self.op("urem", &tid, g.ty), self.op("udiv", &tid, g.ty))
        } else {
            (self.op("udiv", &tid, g.tx), self.op("urem", &tid, g.tx))
        };
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);
        let smem = "@mapal_smem";
        // `sa`: words between two `k` rows of the transposed A panel; `bs_off`:
        // where B starts inside a stage. Async staging pads A's rows and keeps
        // `stages` buffers; register staging has one buffer, padded under the
        // `pad` rung ([`TargetProfile::gpu_tile_sync_a`]).
        let streamed = g.stages >= 2;
        let sa = g.sa();
        let bs_off = bk * sa;
        let stage_elems = g.stage_elems();

        let accs = (tm * tn) as usize;
        let acc = self.tmp();
        self.line(format!(
            "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
        ));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x {t}], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        let seed = self.split_seed(&seed);
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store {t} {seed}, ptr {p}, align {e}"));
        }

        // Loop-invariant load coordinates, one set per load this thread issues:
        // `(global index at k0 = 0, + constant, shared index, + constant)`.
        let (a_vec, b_vec) = (vector_a(s), vector_b(s));
        let [(a_map, a_loads), (b_map, b_loads)] = lane_maps(s, &g);
        let a_src = self.copies(
            a_map,
            a_loads,
            threads,
            &tid,
            |k, r, kc| {
                let grow = k.op("add", &row0, r);
                let gr = k.op("mul", &grow, s.a.ci);
                let gk = k.op("mul", kc, s.a.ck);
                let g0 = k.op("add", &gr, &gk);
                let gsrc = k.op("add", &g0, s.a.base);
                let gsrc = k.at_z(gsrc, kext * s.a.ck);
                let skc = k.op("mul", kc, sa);
                (gsrc, k.op("add", &skc, r))
            },
            |dr, dk| (dr * s.a.ci + dk * s.a.ck, dk * sa + dr),
        );
        let b_src = self.copies(
            b_map,
            b_loads,
            threads,
            &tid,
            |k, kr, nc| {
                let gk = k.op("mul", kr, s.b.ck);
                let gc = k.op("add", &col0, nc);
                let gl = k.op("mul", &gc, s.b.clane);
                let g0 = k.op("add", &gk, &gl);
                let gsrc = k.op("add", &g0, s.b.base);
                let gsrc = k.at_z(gsrc, kext * s.b.ck);
                let sk = k.op("mul", kr, bn);
                let sn = k.op("add", &sk, nc);
                (gsrc, k.op("add", &sn, bs_off))
            },
            |dk, dn| (dk * s.b.ck + dn * s.b.clane, dk * bn + dn),
        );
        // Bank geometry: a thread's register tile is `tm/lanes × tn/lanes`
        // vector chunks spaced one thread-row of chunks apart (`ty·lanes` rows,
        // `tx·lanes` columns), not one contiguous `tm × tn` block. Consecutive
        // threads then read consecutive 16-byte chunks of a panel row, which is
        // what a shared-memory wavefront serves without a conflict — and their
        // output stores coalesce. The CPU's move-panel rung answers the same
        // question for cache sets; here the geometry is the 32 × 4 B banks.
        // (Off arm: one contiguous `tm × tn` block per thread, chunk = lanes.)
        // S62 `fill-rows`: a register tile `tm` rows tall, `tm < lanes`, reads
        // its `a` fragment as one `tm`-wide chunk (`aw`); `tn` keeps whole
        // vectors (the output's vector store). A tile narrower than a vector
        // in `tn` (`cell-fill`'s 1 × 1) has no chunks to space: each thread's
        // cells are contiguous and stored one word each (`narrow`).
        let aw = tm.min(lanes);
        let narrow = tn < lanes;
        let (a_step, b_step, a_chunk, b_chunk) = if self.rungs.interleave && !narrow {
            (aw, lanes, g.ty * aw, g.tx * lanes)
        } else {
            (tm, tn, aw, lanes)
        };
        let a_frag0 = self.op("mul", &trow, a_step);
        let b_frag0 = self.op("mul", &tcol, b_step);

        // --- panel staging: global → (registers) → shared memory
        let a_ty = if a_vec { vt.clone() } else { t.clone() };
        let b_ty = if b_vec { vt.clone() } else { t.clone() };
        let a_al = if a_vec { GPU_VEC_BYTES } else { e };
        let b_al = if b_vec { GPU_VEC_BYTES } else { e };
        let load_panels = |k: &mut Self, k0: &str| -> Vec<String> {
            let ka = k.op("mul", k0, s.a.ck);
            let kb = k.op("mul", k0, s.b.ck);
            let mut vals = Vec::new();
            for (gsrc, goff, _, _) in &a_src {
                let gi = k.op("add", gsrc, &ka);
                let gi = k.plus(gi, *goff);
                let gp = k.gep(1, "%a", &gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {a_ty}, ptr addrspace(1) {gp}, align {a_al}"
                ));
                vals.push(v);
            }
            for (gsrc, goff, _, _) in &b_src {
                let gi = k.op("add", gsrc, &kb);
                let gi = k.plus(gi, *goff);
                let gp = k.gep(1, "%b", &gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {b_ty}, ptr addrspace(1) {gp}, align {b_al}"
                ));
                vals.push(v);
            }
            vals
        };
        let store_panels = |k: &mut Self, vals: &[String]| {
            for ((_, _, sdst, soff), v) in a_src.iter().zip(vals) {
                if !a_vec {
                    let si = k.plus(sdst.clone(), *soff);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!("store {t} {v}, ptr addrspace(3) {sp}, align {e}"));
                    continue;
                }
                // `a` is staged transposed: one vector along k scatters to
                // `lanes` panel rows.
                for j in 0..lanes {
                    let x = k.tmp();
                    k.line(format!("{x} = extractelement {vt} {v}, i64 {j}"));
                    let si = k.op("add", sdst, soff + j * sa);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!("store {t} {x}, ptr addrspace(3) {sp}, align {e}"));
                }
            }
            for ((_, _, sdst, soff), v) in b_src.iter().zip(&vals[a_src.len()..]) {
                let si = k.plus(sdst.clone(), *soff);
                let sp = k.gep(3, smem, &si);
                k.line(format!(
                    "store {b_ty} {v}, ptr addrspace(3) {sp}, align {b_al}"
                ));
            }
        };
        let tys: Vec<&String> = a_src
            .iter()
            .map(|_| &a_ty)
            .chain(b_src.iter().map(|_| &b_ty))
            .collect();

        // `cbase`: the stage the compute reads (async), else the one buffer.
        let (cbase, pre) = if streamed {
            // Issue panel `tile` into stage `stage`: one `cp.async` per unit.
            let issue = |k: &mut Self, tile: &str, stage: &str| {
                let k0 = k.op("mul", tile, bk);
                let ka = k.op("mul", &k0, s.a.ck);
                let kb = k.op("mul", &k0, s.b.ck);
                let so = k.op("mul", stage, stage_elems);
                for (gsrc, goff, sdst, soff) in &a_src {
                    let gi = k.op("add", gsrc, &ka);
                    let gi = k.plus(gi, *goff);
                    let gp = k.gep(1, "%a", &gi);
                    let si = k.op("add", sdst, &so);
                    let si = k.plus(si, *soff);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.ca.shared.global.{e}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
                for (gsrc, goff, sdst, soff) in &b_src {
                    let gi = k.op("add", gsrc, &kb);
                    let gi = k.plus(gi, *goff);
                    let gp = k.gep(1, "%b", &gi);
                    let si = k.op("add", sdst, &so);
                    let si = k.plus(si, *soff);
                    let sp = k.gep(3, smem, &si);
                    let (kind, n) = if b_vec {
                        ("cg", GPU_VEC_BYTES)
                    } else {
                        ("ca", e)
                    };
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.{kind}.shared.global.{n}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
            };
            let nk = kext / bk;
            for st in 0..(g.stages - 1).min(nk) {
                issue(self, &st.to_string(), &st.to_string());
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            for _ in nk..g.stages - 1 {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (it, itn) = (self.tmp(), self.tmp());
            self.line(format!("{it} = phi {ix} [0, %entry], [{itn}, %ktl]"));
            self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                g.stages - 2
            ));
            // One barrier per panel: panel `it` is visible to every thread, and
            // every thread is done with the stage the next issue overwrites.
            self.line("call void @llvm.nvvm.barrier0()");
            let nxt = self.op("add", &it, g.stages - 1);
            let more = self.tmp();
            self.line(format!("{more} = icmp ult {ix} {nxt}, {nk}"));
            self.line(format!("br i1 {more}, label %issue, label %issued"));
            self.out.push_str("issue:\n");
            let nst = self.op("urem", &nxt, g.stages);
            issue(self, &nxt, &nst);
            self.line("br label %issued");
            self.out.push_str("issued:\n");
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cst = self.op("urem", &it, g.stages);
            let cbase = self.op("mul", &cst, stage_elems);
            self.line("br label %kk");
            self.async_tail = Some((it, itn, nk));
            (cbase, "issued")
        } else {
            // Prefetch rung: panel 0 is loaded before the loop; inside it, the
            // registers holding panel `k0` are stored, and panel `k0 + bk` is
            // requested before the compute starts, so its latency hides behind
            // `bk` steps of arithmetic. The last trip re-reads its own panel
            // (in bounds, never stored) rather than branching.
            let first = if self.rungs.prefetch {
                load_panels(self, "0")
            } else {
                Vec::new()
            };
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (k0, k0n) = (self.tmp(), self.tmp());
            self.line(format!("{k0} = phi {ix} [0, %entry], [{k0n}, %ktl]"));
            if self.rungs.prefetch {
                let held: Vec<String> = first.iter().map(|_| self.tmp()).collect();
                for (i, (h, f)) in held.iter().zip(&first).enumerate() {
                    self.line(format!(
                        "{h} = phi {} [{f}, %entry], [@NEXT{i}@, %ktl]",
                        tys[i]
                    ));
                }
                store_panels(self, &held);
                self.line("call void @llvm.nvvm.barrier0()");
                let nk = self.op("add", &k0, bk);
                let more = self.tmp();
                self.line(format!("{more} = icmp ult {ix} {nk}, {kext}"));
                let kl = self.tmp();
                self.line(format!("{kl} = select i1 {more}, {ix} {nk}, {ix} {k0}"));
                let next = load_panels(self, &kl);
                for (i, n) in next.iter().enumerate() {
                    self.out = self.out.replace(&format!("@NEXT{i}@"), n);
                }
            } else {
                let vals = load_panels(self, &k0);
                store_panels(self, &vals);
                self.line("call void @llvm.nvvm.barrier0()");
            }
            self.line("br label %kk");
            self.sync_tail = Some((k0, k0n));
            ("0".to_string(), "kt")
        };

        // --- bk register steps over the staged panels
        self.out.push_str("kk:\n");
        let (kk, kkn) = (self.tmp(), self.tmp());
        self.line(format!("{kk} = phi {ix} [0, %{pre}], [{kkn}, %kk]"));
        let frag = |k: &mut Self, row: u64, first: &str, n: u64, off: u64, chunk: u64| {
            let r = k.op("mul", &kk, row);
            let r = k.op("add", &r, first);
            let r = k.op("add", &r, off);
            let r = k.op("add", &r, &cbase);
            // `w` elements per load: a vector, or (`fill-rows`) the whole
            // fragment when it is narrower than one — `w = 1` a scalar.
            let w = n.min(lanes);
            let wt = if w == 1 {
                t.clone()
            } else {
                format!("<{w} x {t}>")
            };
            let mut vals = Vec::new();
            for v in 0..n / w {
                let si = k.op("add", &r, v * chunk);
                let sp = k.gep(3, smem, &si);
                let x = k.tmp();
                k.line(format!(
                    "{x} = load {wt}, ptr addrspace(3) {sp}, align {}",
                    w * e
                ));
                if w == 1 {
                    vals.push(x);
                    continue;
                }
                for j in 0..w {
                    let y = k.tmp();
                    k.line(format!("{y} = extractelement {wt} {x}, i64 {j}"));
                    vals.push(y);
                }
            }
            vals
        };
        let av = frag(self, sa, &a_frag0, tm, 0, a_chunk);
        let bv = frag(self, bn, &b_frag0, tn, bs_off, b_chunk);
        for i in 0..tm as usize {
            for j in 0..tn as usize {
                let p = acc_slot(self, i * tn as usize + j);
                let cur = self.tmp();
                self.line(format!("{cur} = load {t}, ptr {p}, align {e}"));
                let r = self.step(&cur, &av[i], &bv[j]);
                self.line(format!("store {t} {r}, ptr {p}, align {e}"));
            }
        }
        self.line(format!("{kkn} = add {ix} {kk}, 1"));
        let kkd = self.tmp();
        self.line(format!("{kkd} = icmp eq {ix} {kkn}, {bk}"));
        // Fully unrolled: every shared-memory offset becomes an immediate.
        let unroll = if self.rungs.unroll {
            ", !llvm.loop !@LOOP@"
        } else {
            ""
        };
        self.line(format!("br i1 {kkd}, label %ktl, label %kk{unroll}"));

        self.out.push_str("ktl:\n");
        let ktd = self.tmp();
        if let Some((it, itn, nk)) = self.async_tail.take() {
            // No barrier here: the next panel's barrier orders the overwrite.
            self.line(format!("{itn} = add {ix} {it}, 1"));
            self.line(format!("{ktd} = icmp eq {ix} {itn}, {nk}"));
        } else {
            let (k0, k0n) = self.sync_tail.take().expect("sync panel loop");
            self.line("call void @llvm.nvvm.barrier0()");
            self.line(format!("{k0n} = add {ix} {k0}, {bk}"));
            self.line(format!("{ktd} = icmp eq {ix} {k0n}, {kext}"));
        }
        self.line(format!("br i1 {ktd}, label %done, label %kt"));

        // --- one vector store per `lanes` columns of each owned row
        self.out.push_str("done:\n");
        if narrow {
            // One scalar store per owned cell (`cell-fill`), row- or
            // column-major; the epilogue's store takes one cell at a time.
            for i in 0..tm {
                for j in 0..tn {
                    let p = acc_slot(self, (i * tn + j) as usize);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                    let r = self.op("add", &row0, &a_frag0);
                    let r = self.plus(r, i);
                    let cc = self.op("add", &col0, &b_frag0);
                    let cc = self.plus(cc, j);
                    let oi = if self.col {
                        let cr = self.op("mul", &cc, s.rows);
                        self.op("add", &cr, &r)
                    } else {
                        let rr = self.op("mul", &r, s.c);
                        self.op("add", &rr, &cc)
                    };
                    if self.epi.is_some() {
                        self.store_cells(&[x], &oi);
                        continue;
                    }
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store {t} {x}, ptr addrspace(1) {op}, align {e}"));
                }
            }
            self.out.push_str("  ret void\n}\n");
            return;
        }
        if self.col {
            // Column-major (`rowfold-layout`): one vector store per `lanes`
            // rows of each owned column — a thread's rows come in runs of
            // `lanes` consecutive ones (`col_store`).
            for ic in 0..tm / lanes {
                let r = self.op("add", &row0, &a_frag0);
                let r = self.op("add", &r, ic * a_chunk);
                for v in 0..tn / lanes {
                    for j in 0..lanes {
                        let cc = self.op("add", &col0, &b_frag0);
                        let cc = self.op("add", &cc, v * b_chunk + j);
                        let cr = self.op("mul", &cc, s.rows);
                        let oi = self.op("add", &cr, &r);
                        let mut vec = "poison".to_string();
                        for il in 0..lanes {
                            let i = ic * lanes + il;
                            let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                            let nv = self.tmp();
                            self.line(format!(
                                "{nv} = insertelement {vt} {vec}, {t} {x}, i64 {il}"
                            ));
                            vec = nv;
                        }
                        let op = self.gep(1, "%o", &oi);
                        self.line(format!(
                            "store {vt} {vec}, ptr addrspace(1) {op}, align {GPU_VEC_BYTES}"
                        ));
                    }
                }
            }
            self.out.push_str("  ret void\n}\n");
            return;
        }
        for i in 0..tm {
            let r = self.op("add", &row0, &a_frag0);
            let r = self.op("add", &r, (i / aw) * a_chunk + i % aw);
            let rr = self.op("mul", &r, s.c);
            let c0 = self.op("add", &rr, &col0);
            let c0 = self.op("add", &c0, &b_frag0);
            // A split kernel stores its partials into workspace slab `z`.
            let c0 = self.at_z(c0, s.rows * s.c);
            for v in 0..tn / lanes {
                if self.epi.is_some() {
                    let mut xs = Vec::new();
                    for j in 0..lanes {
                        let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                        let x = self.tmp();
                        self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                        xs.push(x);
                    }
                    let oi = self.op("add", &c0, v * b_chunk);
                    self.store_cells(&xs, &oi);
                    continue;
                }
                let mut vec = "poison".to_string();
                for j in 0..lanes {
                    let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                    let nv = self.tmp();
                    self.line(format!("{nv} = insertelement {vt} {vec}, {t} {x}, i64 {j}"));
                    vec = nv;
                }
                let oi = self.op("add", &c0, v * b_chunk);
                let op = self.gep(1, "%o", &oi);
                self.line(format!(
                    "store {vt} {vec}, ptr addrspace(1) {op}, align {GPU_VEC_BYTES}"
                ));
            }
        }
        self.out.push_str("  ret void\n}\n");
    }
}

impl<'a> FnEmit<'a> {
    /// Host half: hand the site's operands and output to the runtime, which
    /// copies them in, launches [`kernel_name`]`(id)`, and copies the output
    /// back before returning — so everything after the site sees the values a
    /// host loop would have stored.
    pub(super) fn emit_tiled_map_gpu(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
    ) {
        if let Some(plan) = &self.resident {
            // S54 `rowfold-layout`: stored in the order its only reader reads.
            let m = *self.ir.in_edges(target).first().expect("site morphism");
            let col = plan.rowfold.values().any(|r| r.colmajor == Some(m));
            let streams = !plan.chain.is_empty();
            // S54 `epilogue`: the per-cell consumers this site's store evaluates
            // (none when a fold is its only reader: `col` and a group exclude).
            let epi = self.epi_kernel(m);
            let id = if col {
                kernel_id_col(site, self.profile, self.contract, self.reduced, true)
            } else {
                kernel_id_with(
                    site,
                    self.profile,
                    self.contract,
                    self.reduced,
                    epi.as_ref(),
                )
            } as i64;
            let l = launch(
                site,
                self.profile,
                self.contract,
                self.gpu_arms,
                self.reduced,
            );
            // S62 `splitk` (ADR-0040): the split kernel folds `S` partitions
            // into a workspace (`gy·S` blocks), then the reduce kernel adds
            // them into the output in partition order, both on stream 0. The
            // workspace slot is made on the first launch and kept to function
            // exit. ponytail: an epilogue group, a column-major store and a
            // function on streams keep the ordered kernel (and an L2 window
            // is not given to the split launch); split them when a measured
            // program needs it.
            if l.splitk > 1 && epi.is_none() && !col && !streams {
                let s = l.splitk;
                let id = |r| kernel_id_split(site, self.profile, self.contract, self.reduced, r);
                let (sid, rid) = (id(false) as i64, id(true) as i64);
                let (rgx, rblock) = cell_grid(site.rows * site.c);
                let bytes = s * site.rows * site.c * elem_bytes_of(&site.elem);
                let slot = format!("%d{}", self.fresh());
                self.allocas.push_str(&format!(
                    "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                ));
                self.split_slots.push(slot.clone());
                let ws = self.tmp();
                self.line(format!(
                    "{ws} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {bytes})"
                ));
                let geom = format!(
                    "i32 {}, i32 {}, i32 {}, i32 {}",
                    l.gx,
                    l.gy * s,
                    l.block,
                    l.smem
                );
                self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], None, |ops, o| {
                    format!(
                        "call void @mapal_gpu_launch(i64 {sid}, i64 {}, i64 {}, i64 {ws}, {geom}) ; splitk={s}\n  \
                         call void @mapal_gpu_launch(i64 {rid}, i64 {ws}, i64 0, i64 {o}, i32 {rgx}, i32 1, i32 {rblock}, i32 0)",
                        ops[0], ops[1]
                    )
                });
                return;
            }
            // ponytail: a grouped launch (`launchv`) and a function on streams
            // (whose launches order through `launch_on`) carry no L2 window; give
            // them one when a part ships both on one program.
            let window = (epi.is_none()
                && !streams
                && host_rungs(self.profile, self.contract, self.gpu_arms).l2_window)
                .then(|| self.l2_window(m, site, &l))
                .flatten();
            // The column-major tile kernel takes its row block from `ctaid.x`.
            let (gx, gy) = if col && l.tiled.is_some() {
                (l.gy, l.gx)
            } else {
                (l.gx, l.gy)
            };
            let geom = format!("i32 {gx}, i32 {gy}, i32 {}, i32 {}", l.block, l.smem);
            let grouped = (id, geom.clone());
            self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], Some(grouped), |ops, o| {
                let (a, b) = (ops[0].as_str(), ops[1].as_str());
                match window {
                    None => format!(
                        "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, {geom})"
                    ),
                    Some((w, bytes)) => {
                        let p = match w {
                            Window::A => a,
                            Window::B => b,
                            Window::Out => o,
                        };
                        format!(
                            "call void @mapal_gpu_launch_l2(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                             {geom}, i64 {p}, i64 {bytes})"
                        )
                    }
                }
            });
            return;
        }
        let src_ty = self.obj_ty(source);
        let bytes = |ty: &Ty| {
            let (_, n) = array_parts(ty);
            n * elem_bytes_of(&site.elem)
        };
        let a_bytes = bytes(src_ty.component_ty(site.a.slot).expect("tile a array"));
        let b_bytes = bytes(src_ty.component_ty(site.b.slot).expect("tile b array"));
        let o_bytes = bytes(&self.obj_ty(target));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        let o_ptr = self.slot(target).expect("tile output slot");
        // LLVM reads an `i64` literal as signed; the runtime prints the same
        // 64 bits back as hex, so the two units agree on the name.
        let id = kernel_id(site, self.profile, self.contract, self.reduced) as i64;
        let l = launch(
            site,
            self.profile,
            self.contract,
            self.gpu_arms,
            self.reduced,
        );
        self.line(format!(
            "call void @mapal_gpu_tile(i64 {id}, ptr {a_ptr}, i64 {a_bytes}, ptr {b_ptr}, \
             i64 {b_bytes}, ptr {o_ptr}, i64 {o_bytes}, i32 {}, i32 {}, i32 {}, i32 {})",
            l.gx, l.gy, l.block, l.smem
        ));
    }

    /// Site `m`'s epilogue as its kernel takes it (S54), `None` without one.
    pub(super) fn epi_kernel(&self, m: MorphismId) -> Option<EpiKernel> {
        let g = self.groups.get(m)?;
        let caps = g.caps();
        let calls = g
            .chain
            .iter()
            .map(|mb| {
                let captures = match self.ir.morphism(mb.m).expect("member").op {
                    Operation::Map { captures, .. } => captures,
                    _ => unreachable!("an epilogue member is a map"),
                };
                let in_ty = self.obj_ty(self.ir.func(mb.body).expect("body").input);
                let arg = crate::ty::lower_body_input_ty(&in_ty, captures)
                    .expect("a member's input lowers");
                let fields: Vec<EpiArg> = mb
                    .srcs
                    .iter()
                    .enumerate()
                    .map(|(slot, s)| match *s {
                        resident::EpiSrc::Value(k) if (slot as u32) < captures => EpiArg::Cell(k),
                        resident::EpiSrc::Value(k) => EpiArg::Value(k),
                        resident::EpiSrc::Gmem(o) => {
                            EpiArg::Cap(caps.iter().position(|&x| x == o).expect("an operand"))
                        }
                        resident::EpiSrc::Index => EpiArg::Index,
                    })
                    .collect();
                let cell = fields.iter().any(|a| matches!(a, EpiArg::Cell(_)));
                let (ret, _) = array_parts(&self.obj_ty(mb.out));
                EpiCall {
                    callee: format!(
                        "{}{}",
                        self.fnames[mb.body],
                        if cell { "_cell" } else { "" }
                    ),
                    arg,
                    ret,
                    fields,
                    store: mb.store,
                }
            })
            .collect();
        Some(EpiKernel {
            store_out: g.store_out,
            caps: caps.len(),
            calls,
        })
    }

    /// This function's grouped sites for the device unit, and the bodies
    /// their kernels call: `(callee, body, captured slots read at the cell)`
    /// — an empty slot list is the host's own text.
    pub(crate) fn epi_sites(&self) -> (Vec<EpiSite>, Vec<(String, FuncId, Vec<u32>)>) {
        let mut sites = Vec::new();
        let mut bodies = Vec::new();
        for (m, g) in self.groups.iter() {
            let site = self
                .tile_plan
                .as_ref()
                .and_then(|p| p.sites.get(m))
                .expect("a grouped site is a tile site")
                .clone();
            let k = self.epi_kernel(m).expect("a group");
            for (mb, c) in g.chain.iter().zip(&k.calls) {
                let slots = c
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| matches!(a, EpiArg::Cell(_)))
                    .map(|(s, _)| s as u32)
                    .collect();
                bodies.push((c.callee.clone(), mb.body, slots));
            }
            sites.push(EpiSite {
                f: self.f,
                m,
                site,
                k,
            });
        }
        (sites, bodies)
    }

    /// A retained site's restricted pass (S53, `func/retain.rs`): output row
    /// `row` (an `i64` SSA value) of `site`, as the one-row site `one` whose
    /// `a` is read from the row's first cell and whose output lands at the
    /// row's first cell. Only the row's `a` span and output cross the bus.
    pub(super) fn emit_tiled_map_gpu_row(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        one: &TileSite,
        row: &str,
    ) {
        if self.resident.is_some() {
            let id = kernel_id(one, self.profile, self.contract, self.reduced) as i64;
            let l = launch(
                one,
                self.profile,
                self.contract,
                self.gpu_arms,
                self.reduced,
            );
            let call = |a: &str, b: &str, o: &str| {
                format!(
                    "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 {}, i32 {}, i32 {})",
                    l.gx, l.gy, l.block, l.smem
                )
            };
            if !self.emit_row_resident(source, target, site, row, call) {
                self.emit_tiled_map_gpu(source, target, site);
            }
            return;
        }
        let eb = elem_bytes_of(&site.elem);
        let elem_llt = lower_ty(&site.elem).expect("tile element lowers");
        let src_ty = self.obj_ty(source);
        let (_, b_len) = array_parts(src_ty.component_ty(site.b.slot).expect("tile b array"));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        let o_ptr = self.slot(target).expect("tile output slot");
        let a_cells = self.tmp();
        self.line(format!("{a_cells} = mul i64 {row}, {}", site.a.ci));
        let a_off = self.tmp();
        self.line(format!("{a_off} = add i64 {a_cells}, {}", site.a.base));
        let a_row = self.tmp();
        self.line(format!(
            "{a_row} = getelementptr {elem_llt}, ptr {a_ptr}, i64 {a_off}"
        ));
        let o_off = self.tmp();
        self.line(format!("{o_off} = mul i64 {row}, {}", site.c));
        let o_row = self.tmp();
        self.line(format!(
            "{o_row} = getelementptr {elem_llt}, ptr {o_ptr}, i64 {o_off}"
        ));
        let a_bytes = ((one.k - 1) * one.a.ck + 1) * eb;
        let (b_bytes, o_bytes) = (b_len * eb, one.c * eb);
        let id = kernel_id(one, self.profile, self.contract, self.reduced) as i64;
        let l = launch(
            one,
            self.profile,
            self.contract,
            self.gpu_arms,
            self.reduced,
        );
        self.line(format!(
            "call void @mapal_gpu_tile(i64 {id}, ptr {a_row}, i64 {a_bytes}, ptr {b_ptr}, \
             i64 {b_bytes}, ptr {o_row}, i64 {o_bytes}, i32 {}, i32 {}, i32 {}, i32 {})",
            l.gx, l.gy, l.block, l.smem
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuRungs;
    use crate::profile::{H100, H100_PCIE, RTX_4090, RTX_5090};
    use mapal_ir::TileRead;
    use proptest::prelude::*;

    /// A `m × n × k` site as `benches/gpu/gen_tiles.py` writes it: `a[i·K + k]`
    /// (or column-major `a[k·M + i]`), `b[k·N + j]` (or `b[j·K + k]`).
    fn site(m: u64, n: u64, k: u64, a_rm: bool, b_rm: bool, elem: Ty) -> TileSite {
        let read = |slot, ci, ck, clane| TileRead {
            slot,
            base: 0,
            ci,
            ck,
            clane,
            ksplit: None,
        };
        let a = if a_rm {
            read(0, k, 1, 0)
        } else {
            read(0, 1, m, 0)
        };
        let b = if b_rm {
            read(1, 0, n, 1)
        } else {
            read(1, 0, 1, k)
        };
        let seed = if elem == Ty::f32() {
            Value::F32(0.0)
        } else {
            Value::F64(0.0)
        };
        TileSite {
            rows: m,
            c: n,
            k,
            a,
            b,
            seed,
            elem,
            mul_a_first: true,
            add_acc_first: true,
        }
    }

    /// Every S52 box site: the four timed squares and the six differential
    /// programs.
    fn s52_sites() -> Vec<TileSite> {
        let f32 = Ty::f32;
        let mut v: Vec<TileSite> = [1024, 2048, 4096, 8192]
            .map(|n| site(n, n, n, true, true, f32()))
            .into();
        v.extend([
            site(256, 256, 256, true, true, f32()),
            site(128, 384, 96, true, true, f32()),
            site(256, 256, 128, false, true, f32()),
            site(256, 256, 128, true, false, f32()),
            site(100, 100, 100, true, true, f32()),
            site(128, 128, 128, true, true, Ty::f64()),
        ]);
        v
    }

    /// Arms that emit exactly the rung set `r`, whatever the profile's table.
    fn force(r: GpuRungs) -> crate::GpuArms {
        let mut off = GpuRungs::ALL;
        for n in GpuRungs::NAMES {
            *off.rung(n).expect("a listed name") = !r.get(n);
        }
        off.arch = true;
        crate::GpuArms {
            on: GpuRungs::default(),
            off,
            splitk: 0,
        }
    }

    const PARTS: [&TargetProfile; 4] = [&RTX_4090, &H100_PCIE, &H100, &RTX_5090];

    #[test]
    fn actual_arithmetic_strided_contract_selects_exact_policy() {
        let s = site(1, 256, 128, true, false, Ty::f32());
        let arms = crate::GpuArms::default();
        let text = kernel_text(
            "strided", &s, &H100_PCIE, true, arms, None, false, None, false,
        );
        assert!(text.contains(" = fmul float ") && text.contains(" = fadd float "));
        assert!(!text.contains("call float @llvm.fma"));
        let exact = launch(&s, &H100_PCIE, false, arms, None);
        let requested = launch(&s, &H100_PCIE, true, arms, None);
        assert_eq!(
            requested.rungs, exact.rungs,
            "separate arithmetic needs Exact policy"
        );
    }

    #[test]
    fn actual_arithmetic_rejected_tf32_selects_scalar_policy() {
        use crate::profile::RTX_3090;
        let s = site(256, 256, 128, false, true, Ty::f32());
        let arms = crate::GpuArms::default();
        let text = kernel_text(
            "fallback",
            &s,
            &RTX_3090,
            true,
            arms,
            Some(Reduced::Tf32),
            false,
            None,
            false,
        );
        assert!(text.contains("call float @llvm.fma"));
        assert!(!text.contains("mma.sync") && !text.contains("wgmma.mma_async"));
        let scalar = launch(&s, &RTX_3090, true, arms, None);
        let requested = launch(&s, &RTX_3090, true, arms, Some(Reduced::Tf32));
        assert_eq!(
            requested.rungs, scalar.rungs,
            "rejected TF32 needs scalar policy"
        );
    }

    #[test]
    fn actual_arithmetic_scalar_modes_preserve_order_and_geometry() {
        for p in PARTS {
            for n in [7, 256] {
                // naive and cooperative scalar realizations
                for elem in [Ty::f32(), Ty::f64()] {
                    let mut s = site(n, n, 128, true, true, elem);
                    s.mul_a_first = false;
                    s.add_acc_first = false;
                    s.seed = if s.elem == Ty::f32() {
                        Value::F32(2.0)
                    } else {
                        Value::F64(2.0)
                    };
                    for contract in [false, true] {
                        let arms = crate::GpuArms::default();
                        let l = launch(&s, p, contract, arms, None);
                        let expected = if contract {
                            Arithmetic::OrderedFma
                        } else {
                            Arithmetic::SeparateMulAdd
                        };
                        assert_eq!(l.arithmetic, expected);
                        assert_eq!(l.tiled.is_some(), n == 256, "{} {s:?}", p.name);
                        let text =
                            kernel_text("scalar", &s, p, contract, arms, None, false, None, false);
                        assert_eq!(text.contains("@llvm.fma"), contract);
                        assert_eq!(text.contains(" = fmul "), !contract);
                        assert_eq!(text.contains(" = fadd "), !contract);
                        // Exercise the same step builder with named values so
                        // the source order is checked independently of SSA ids.
                        let mut k = Kernel {
                            out: String::new(),
                            next: 0,
                            idx: "i64",
                            t: lower_ty(&s.elem).unwrap(),
                            site: &s,
                            arithmetic: l.arithmetic,
                            rungs: l.rungs,
                            profile_gpu: *p.gpu().unwrap(),
                            pdl: false,
                            async_tail: None,
                            sync_tail: None,
                            raster: (l.gx, l.raster),
                            col: false,
                            epi: None,
                            splitk: 1,
                            gy_out: l.gy,
                            z: None,
                        };
                        let result = k.step("%acc", "%a", "%b");
                        let t = &k.t;
                        if contract {
                            assert!(k.out.contains(&format!("({t} %a, {t} %b, {t} %acc)")));
                        } else {
                            assert!(k.out.contains(&format!("fmul {t} %b, %a")));
                            assert!(
                                k.out.contains(&format!("{result} = fadd {t} %v2, %acc")),
                                "{}",
                                k.out
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn actual_arithmetic_tensor_rejections_use_complete_scalar_plan() {
        use crate::profile::{Machine, RTX_3090};
        let old = TargetProfile {
            machine: Machine::Gpu(Gpu {
                arch: "sm_70",
                ..*RTX_4090.gpu().unwrap()
            }),
            ..RTX_4090
        };
        let ordinary = site(256, 256, 128, true, true, Ty::f32());
        let mut cases = vec![
            ("type", site(256, 256, 128, true, true, Ty::f64())),
            ("a-layout", site(256, 256, 128, false, true, Ty::f32())),
            ("b-layout", site(256, 256, 128, true, false, Ty::f32())),
            ("rows", site(1, 256, 128, true, true, Ty::f32())),
            ("columns", site(256, 7, 128, true, true, Ty::f32())),
            ("reduction", site(256, 256, 7, true, true, Ty::f32())),
        ];
        for (name, field) in [
            ("a-base", 0),
            ("b-base", 1),
            ("a-row", 2),
            ("b-row", 3),
            ("a-lane", 4),
            ("b-row-dependence", 5),
        ] {
            let mut s = ordinary.clone();
            match field {
                0 => s.a.base = 1,
                1 => s.b.base = 1,
                2 => s.a.ci += 1,
                3 => s.b.ck += 1,
                4 => s.a.clane = 1,
                _ => s.b.ci = 1,
            }
            cases.push((name, s));
        }
        let off = force(GpuRungs {
            mma: false,
            ..GpuRungs::ALL
        });
        for p in [&RTX_3090, &RTX_4090, &H100_PCIE, &H100, &RTX_5090, &old] {
            for (why, s) in &cases {
                for contract in [false, true] {
                    for arms in [crate::GpuArms::default(), force(GpuRungs::ALL), off] {
                        let scalar = launch(s, p, contract, arms, None);
                        let l = launch(s, p, contract, arms, Some(Reduced::Tf32));
                        assert_eq!(l, scalar, "{} {why} contract={contract}", p.name);
                        let text = kernel_text(
                            "rejected",
                            s,
                            p,
                            contract,
                            arms,
                            Some(Reduced::Tf32),
                            false,
                            None,
                            false,
                        );
                        assert!(
                            !text.contains("tf32")
                                && !text.contains("mma.sync")
                                && !text.contains("wgmma.mma_async")
                        );
                        assert_eq!(
                            text.contains("@llvm.fma"),
                            l.arithmetic == Arithmetic::OrderedFma
                        );
                    }
                }
            }
            // Valid operands with MMA forced off, and an unsupported device
            // with every optimization forced on, must still use scalar policy.
            for arms in if p.gpu().unwrap().sm() < 80 {
                vec![off, force(GpuRungs::ALL)]
            } else {
                vec![off]
            } {
                for contract in [false, true] {
                    assert_eq!(
                        launch(&ordinary, p, contract, arms, Some(Reduced::Tf32)),
                        launch(&ordinary, p, contract, arms, None),
                    );
                }
            }
        }
    }

    #[test]
    fn actual_arithmetic_supported_tensor_controls_keep_tf32_policy() {
        let s = site(1024, 1024, 1024, true, true, Ty::f32());
        let mma = force(GpuRungs {
            wgmma: false,
            ..GpuRungs::ALL
        });
        for (p, arms, expected, instruction) in [
            (
                &RTX_4090,
                force(GpuRungs::ALL),
                Arithmetic::Tf32Mma,
                "mma.sync.aligned",
            ),
            (&H100_PCIE, mma, Arithmetic::Tf32Mma, "mma.sync.aligned"),
            (
                &H100_PCIE,
                force(GpuRungs::ALL),
                Arithmetic::Tf32Wgmma,
                "wgmma.mma_async",
            ),
        ] {
            for contract in [false, true] {
                let l = launch(&s, p, contract, arms, Some(Reduced::Tf32));
                assert_eq!(l.arithmetic, expected);
                assert_eq!(l.arithmetic.face(), Face::Tf32);
                assert_eq!(l.rungs, GpuRungs::ALL.with(arms));
                assert!(l.block <= p.gpu().unwrap().max_threads_per_block);
                assert!(l.smem <= p.gpu().unwrap().smem_per_block);
                let text = kernel_text(
                    "tensor",
                    &s,
                    p,
                    contract,
                    arms,
                    Some(Reduced::Tf32),
                    false,
                    None,
                    false,
                );
                assert!(text.contains(instruction));
                assert!(!text.contains("@llvm.fma") && !text.contains(" = fmul "));
            }
        }
    }

    #[test]
    fn actual_arithmetic_mixed_sites_keep_module_pdl_agreement() {
        let sites = [
            site(1, 256, 128, true, false, Ty::f32()),
            site(256, 256, 128, true, true, Ty::f32()),
        ];
        for p in [&H100_PCIE, &RTX_4090] {
            for enabled in [false, true] {
                let arms = force(GpuRungs {
                    pdl: enabled,
                    ..GpuRungs::ALL
                });
                let wait = pdl(p, true, arms);
                assert_eq!(wait, enabled && p.gpu().unwrap().sm() >= 90);
                let host = init_call(host_rungs(p, true, arms), false, wait);
                let flags: u32 = host
                    .split("i32 ")
                    .nth(1)
                    .unwrap()
                    .split(')')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(flags & GPU_INIT_PDL != 0, wait);
                for (s, expected) in sites
                    .iter()
                    .zip([Arithmetic::SeparateMulAdd, Arithmetic::OrderedFma])
                {
                    assert_eq!(launch(s, p, true, arms, None).arithmetic, expected);
                    let text = kernel_text("mixed", s, p, true, arms, None, false, None, false);
                    assert_eq!(text.contains("griddepcontrol.wait"), wait);
                }
            }
        }
    }

    /// S62: the skinny M16 projections take the cooperative ordered tile on the
    /// 5090 — on S53 code both took the one-thread-per-cell kernel (`None`).
    /// 2048 columns: `fill-rows` 1×4 at 8 × 4 threads, 256 blocks; two per SM
    /// (`grid-share`) ⇒ bk 512; `panel-regs` (255 − 46 registers free for
    /// the panel at ¾ register per `k`) ⇒ bk 256. 18432 columns (Filled):
    /// the 5090 ships `row-fit` OFF on the exact face (no effect measured), so
    /// the vector floor does not fit 16 rows and `fill-rows` takes 1×4 at
    /// 16 × 16 threads, 288 blocks, `cp.async`; with `row-fit` the floor 4×4
    /// row-fits to 16 × 32. The contract face ships `warp-fill` and
    /// `wave-gate` OFF there: 8×8 row-fits to 144 blocks (gated: the
    /// one-thread-per-cell kernel, 4.5× slower on the 5090). `panel-regs`
    /// reads only grids that leave a scheduler without a warp: not 1024² or
    /// 128 × 18432 (3 warps per scheduler), whose launch is its OFF arm's.
    #[test]
    fn skinny_projections_take_the_ordered_cooperative_tile() {
        let geo = |n, contract, arms| {
            let s = site(16, n, 7168, true, true, Ty::f32());
            launch(&s, &RTX_5090, contract, arms, None)
                .tiled
                .map(|g| (g.tm, g.tn, g.bm(), g.bn(), g.threads(), g.bk, g.stages))
        };
        let shipped = crate::GpuArms::default();
        let off = |r: &str| {
            let mut a = crate::GpuArms::default();
            *a.off.rung(r).expect("a rung") = true;
            a
        };
        let on = |r: &str| {
            let mut a = crate::GpuArms::default();
            *a.on.rung(r).expect("a rung") = true;
            a
        };
        for contract in [false, true] {
            assert_eq!(
                geo(2048, contract, shipped),
                Some((1, 4, 8, 16, 32, 256, 1))
            );
        }
        assert_eq!(geo(18432, false, shipped), Some((1, 4, 16, 64, 256, 64, 2)));
        assert_eq!(
            geo(18432, false, on("row-fit")),
            Some((4, 4, 16, 32, 32, 32, 3))
        );
        assert_eq!(geo(18432, true, shipped), Some((8, 8, 16, 128, 32, 16, 1)));
        // Gated, the site has no register tile: `cell-fill` (S62 asyncstage
        // item) takes it, else the one-thread-per-cell kernel.
        assert_eq!(
            geo(18432, true, on("wave-gate")),
            Some((1, 1, 16, 16, 256, 64, 1))
        );
        let mut gated = on("wave-gate");
        gated.off.cell_fill = true;
        assert_eq!(geo(18432, true, gated), None);
        assert_eq!(
            geo(2048, false, off("panel-regs")),
            Some((1, 4, 8, 16, 32, 16, 1))
        );
        for (m, n, k) in [(1024, 1024, 1024), (128, 18432, 7168)] {
            for contract in [false, true] {
                let at = |arms| {
                    let l = launch(
                        &site(m, n, k, true, true, Ty::f32()),
                        &RTX_5090,
                        contract,
                        arms,
                        None,
                    );
                    (l.tiled, l.smem)
                };
                assert_eq!(at(shipped), at(off("panel-regs")), "{m}x{n}x{k} {contract}");
            }
        }
    }

    /// Every staged copy of a skinny site's panels lands once, in the panel,
    /// on every rung setting the S62 geometry rungs reach (row-fitted, gated
    /// off, scalar-row register tiles, f32 and f64, every operand layout).
    #[test]
    fn skinny_cooperative_copies_cover_each_panel_once() {
        use crate::profile::{A100, RTX_3090};
        use std::collections::BTreeSet;
        let geometry = |r: GpuRungs, names: &[&str]| {
            let mut r = r;
            for n in names {
                *r.rung(n).expect("a rung") = false;
            }
            r
        };
        for p in [&A100, &RTX_3090].into_iter().chain(PARTS) {
            for m in [1, 16, 128] {
                for (n, depth) in [(18432, 7168), (2048, 7168), (1024, 4096)] {
                    for (arm, brm) in [(true, true), (false, true), (true, false)] {
                        for elem in [Ty::f32(), Ty::f64()] {
                            let s = site(m, n, depth, arm, brm, elem);
                            let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
                            for base in arms() {
                                for names in [
                                    &[][..],
                                    &["fill-rows"],
                                    &["fill-rows", "wave-gate"],
                                    &["grid-share"],
                                    &["panel-regs"],
                                ] {
                                    let r = force(geometry(base, names));
                                    let Some(g) = launch(&s, p, false, r, None).tiled else {
                                        continue;
                                    };
                                    for (i, (map, loads)) in
                                        lane_maps(&s, &g).into_iter().enumerate()
                                    {
                                        let vec = if i == 0 {
                                            vector_a(&s) && g.stages < 2
                                        } else {
                                            vector_b(&s)
                                        };
                                        let width = if vec { lanes } else { 1 };
                                        let mut seen = BTreeSet::new();
                                        for copy in 0..loads {
                                            for tid in 0..g.threads() {
                                                let li = tid + copy * g.threads();
                                                let (row, col) = map.coords(&mut Num, &li);
                                                for v in 0..width {
                                                    assert!(seen.insert((row, col + v)), "{g:?}");
                                                }
                                            }
                                        }
                                        let (rows, cols) = if i == 0 {
                                            (g.bm(), g.bk)
                                        } else {
                                            (g.bk, g.bn())
                                        };
                                        assert_eq!(seen.len() as u64, rows * cols, "{g:?}");
                                        assert!(seen.iter().all(|&(r, c)| r < rows && c < cols));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The rung settings whose geometry differs: shipped, register staging,
    /// and each with the whole-file tile off.
    fn arms() -> Vec<GpuRungs> {
        let on = GpuRungs::ALL;
        let sync = GpuRungs {
            cp_async: false,
            ..on
        };
        [on, sync]
            .into_iter()
            .flat_map(|r| {
                [
                    r,
                    GpuRungs {
                        wide_fma: false,
                        ..r
                    },
                ]
            })
            .collect()
    }

    /// S54 `raster` at the kernels the box times, on the tile each face
    /// actually takes. The derivation (every rung on, `arch-rungs` off): the
    /// 3090 groups at 8192 only (one block's panels, 8 MB, past its 6 MB L2),
    /// every other part keeps row-major; without the gate every part groups
    /// where the grid is more than a wave. What ships is the measured table:
    /// raster OFF on the 3090; every other part keeps the derived L2 gate,
    /// including the 5090 after S56 selection. All shipped cases stay row-major;
    /// the forced-ungated arm still exercises grouping.
    #[test]
    fn raster_fires_where_one_blocks_panels_leave_the_l2() {
        use crate::profile::{A100, RTX_3090};
        let parts: [&TargetProfile; 6] =
            [&RTX_3090, &A100, &RTX_4090, &H100_PCIE, &H100, &RTX_5090];
        let faces = [(false, None), (true, None), (true, Some(Reduced::Tf32))];
        let arms = |gate_off: bool| {
            let mut off = GpuRungs::default();
            off.arch = true;
            off.l2_gate = gate_off;
            crate::GpuArms {
                on: GpuRungs::default(),
                off,
                splitk: 0,
            }
        };
        for p in parts {
            for n in [1024, 2048, 4096, 8192] {
                for (contract, reduced) in faces {
                    let s = site(n, n, n, true, true, Ty::f32());
                    let at = |a| launch(&s, p, contract, a, reduced);
                    let why = format!("{} {n} {contract} {reduced:?}", p.name);
                    let derived = at(arms(false));
                    assert_eq!(
                        derived.raster > 1,
                        p.name == "rtx-3090" && n == 8192,
                        "{why}"
                    );
                    let ungated = at(arms(true));
                    if n == 8192 {
                        assert!(ungated.raster > 1, "{why}");
                    }
                    assert!(ungated.gy % ungated.raster == 0, "{why}");
                    let shipped = at(crate::GpuArms::default());
                    assert_eq!(shipped.raster, 1, "{why}");
                }
            }
        }
    }

    /// S62 `splitk` on the measured cells and parts (`--nocapture` prints the
    /// table the results' `derived-S.txt` records): only the reassociation face
    /// splits; a split grid was under one wave and its partitions are whole
    /// panels; a grid that fills the part (16 × 18432 × 7168 on the 5090, the
    /// 4096³ control) keeps the ordered kernel.
    #[test]
    fn splitk_derivation_on_the_measured_cells() {
        let cells = [
            (16, 2048, 7168),
            (16, 1024, 4096),
            (16, 18432, 7168),
            (1, 2048, 7168),
            (1, 1024, 4096),
            (128, 2048, 7168),
            (128, 1024, 4096),
            (4096, 4096, 4096),
        ];
        let parts = [&RTX_5090, &RTX_4090, &H100_PCIE, &crate::profile::RTX_3090];
        let reassoc = Some(Reduced::Reassoc);
        let arms = crate::GpuArms::default();
        for p in parts {
            let g = p.gpu().unwrap();
            for (m, n, k) in cells {
                let s = site(m, n, k, true, true, Ty::f32());
                for contract in [false, true] {
                    let l = launch(&s, p, contract, arms, reassoc);
                    let unit = l.tiled.map_or(1, |t| t.bk);
                    let (blocks, wave) = (l.gx * l.gy, l.wave(g));
                    let base = p.gpu_tile(m, n, k, &s.elem, false, false);
                    let (fb, ft) = base.map_or((blocks, l.block), |t| {
                        ((m / t.bm()) * (n / t.bn()), t.threads())
                    });
                    let wps = g.warps_per_scheduler(fb, ft);
                    println!(
                        "{} {m}x{n}x{k} {} kernel={} grid={}x{} block={} blocks={blocks} wave={wave} \
                         fill={} S={} kc={}",
                        p.name,
                        if contract { "_rc" } else { "_r" },
                        if l.tiled.is_some() { "tiled" } else { "naive" },
                        l.gx,
                        l.gy,
                        l.block,
                        if wps < GPU_RESIDENT_BLOCKS {
                            "under"
                        } else {
                            "filled"
                        },
                        l.splitk,
                        k / l.splitk,
                    );
                    assert!(l.splitk == 1 || blocks < wave);
                    assert!(k.is_multiple_of(l.splitk * unit));
                    assert_eq!(launch(&s, p, contract, arms, None).splitk, 1);
                    let off = crate::GpuArms {
                        off: GpuRungs {
                            splitk: true,
                            ..GpuRungs::default()
                        },
                        ..arms
                    };
                    assert_eq!(launch(&s, p, contract, off, reassoc).splitk, 1);
                }
            }
        }
        // The S the splitk item measured (S62) were derived on the pre-S62
        // tiles: the geometry and cell-fill rungs off. With them on (the
        // integrated build) a skinny site's grid is re-tiled and `grid-share`
        // sizes residency to it, so fewer slots stand empty (5090 M16 × 2048:
        // 256 blocks on 340 slots, S = 2) — the invariants above still hold.
        let mut arms = arms;
        for r in [
            "row-fit",
            "wave-gate",
            "grid-share",
            "panel-regs",
            "fill-rows",
            "cell-fill",
        ] {
            *arms.off.rung(r).expect("a rung") = true;
        }
        let at = |p, (m, n, k)| {
            launch(
                &site(m, n, k, true, true, Ty::f32()),
                p,
                true,
                arms,
                reassoc,
            )
            .splitk
        };
        assert_eq!(
            at(&RTX_5090, (16, 2048, 7168)),
            8,
            "128 blocks, 170 × 6 slots"
        );
        assert_eq!(at(&RTX_5090, (16, 1024, 4096)), 16);
        assert_eq!(
            at(&RTX_5090, (16, 18432, 7168)),
            1,
            "1152 blocks fill 1020 slots"
        );
        assert_eq!(at(&RTX_5090, (4096, 4096, 4096)), 1);
        // The measured table: the 3090 ships the ordered kernel at (filled,
        // exact) — its one split cell there lost — and splits on the fma face.
        let s16 = site(16, 2048, 7168, true, true, Ty::f32());
        let r3090 = &crate::profile::RTX_3090;
        assert_eq!(launch(&s16, r3090, false, arms, reassoc).splitk, 1);
        assert_eq!(launch(&s16, r3090, true, arms, reassoc).splitk, 4);
    }

    /// The block map is a bijection on the grid for every height it is
    /// derived for (a power of two dividing `gy`), evaluated by the same code
    /// that emits it.
    #[test]
    fn raster_block_is_a_bijection_on_the_grid() {
        for (gx, gy) in [(2, 2), (12, 8), (64, 64), (32, 16), (3, 32), (1, 4)] {
            for h in (0..=u64::ilog2(gy)).map(|p| 1u64 << p) {
                let mut seen = vec![false; (gx * gy) as usize];
                for by in 0..gy {
                    for bx in 0..gx {
                        let (c, r) = raster_block(&mut Num, &bx, &by, gx, h);
                        assert!(c < gx && r < gy, "{gx}x{gy} h={h}");
                        assert!(!std::mem::replace(&mut seen[(r * gx + c) as usize], true));
                    }
                }
                // Within a group, consecutive launches walk down a column.
                let (c0, r0) = raster_block(&mut Num, &0, &0, gx, h);
                let (c1, r1) = raster_block(&mut Num, &1, &0, gx, h);
                if h > 1 && gx > 1 {
                    assert_eq!((c0, r0, c1, r1), (0, 0, 0, 1));
                }
            }
        }
    }

    /// `devmap`: one thread per cell, the block the naive kernel's, the grid
    /// the fewest blocks that cover the extent; a cell's id never names a gen
    /// kernel of the same body (different key, different prefix anyway).
    #[test]
    fn cell_grid_covers_the_extent_one_thread_per_cell() {
        for n in [1, 255, 256, 257, 128 * 128, 4096 * 4096] {
            let (gx, block) = cell_grid(n);
            assert_eq!(block, NAIVE_THREADS);
            assert!(gx * block >= n && (gx - 1) * block < n, "n={n}");
        }
        let p = crate::profile::resolve("rtx-4090").expect("profile");
        assert_ne!(cell_id("fn2", 16384, p), gen_id("fn2", 16384, p));
        let caps = [None, Some(("fn3".to_string(), "float".to_string()))];
        let k = cell_kernel_text(
            1,
            "fn2_cell",
            300,
            "float",
            "{ ptr, ptr, i32 }",
            None,
            &caps,
        );
        assert!(
            k.contains("(ptr addrspace(1) %o, ptr addrspace(1) %c0)"),
            "{k}"
        );
        assert!(k.contains("%v1 = call float @fn3(i32 %e)"), "{k}");
        assert!(
            k.contains("@fn2_cell({ ptr, ptr, i32 } %a, float %v1)"),
            "{k}"
        );
        assert!(k.contains("icmp ult i64 %i, 300"), "{k}");
    }

    /// `idx32`: every S52 site indexes in `i32`; an operand past 2³¹ elements
    /// keeps `i64`.
    #[test]
    fn idx32_takes_i32_exactly_when_every_index_fits() {
        for p in PARTS {
            for s in s52_sites() {
                let l = launch(&s, p, false, force(GpuRungs::ALL), None);
                assert!(max_index(&s, &l, 1) < 1 << 31, "{} {s:?}", p.name);
                let k = kernel_text(
                    "k",
                    &s,
                    p,
                    false,
                    force(GpuRungs::ALL),
                    None,
                    false,
                    None,
                    false,
                );
                assert!(k.contains("add nuw nsw i32"), "{} {s:?}", p.name);
            }
        }
        // 2¹⁶ rows of 2¹⁶: `a`'s last element is 2³² − 1.
        let big = site(1 << 16, 1 << 16, 1 << 16, true, true, Ty::f32());
        let l = launch(&big, &RTX_4090, false, force(GpuRungs::ALL), None);
        assert!(max_index(&big, &l, 1) >= 1 << 31);
        let k = kernel_text(
            "k",
            &big,
            &RTX_4090,
            false,
            force(GpuRungs::ALL),
            None,
            false,
            None,
            false,
        );
        assert!(!k.contains("nuw nsw i32") && k.contains("add i64"));
    }

    /// `affine`: for every S52 site, part and arm, every copy `p` of both
    /// panels is copy 0 plus a constant — the lane map evaluated at
    /// `tid + p·threads` equals the `udiv`/`urem` chain at `tid` plus the delta,
    /// for every `tid` ([`affine_delta`] checks exactly that).
    #[test]
    fn affine_decomposition_equals_udiv_urem_for_every_copy_and_thread() {
        let mut copies = 0;
        for p in PARTS {
            for r in arms() {
                for s in s52_sites() {
                    let Some(g) = launch(&s, p, false, force(r), None).tiled else {
                        continue;
                    };
                    for (map, n) in lane_maps(&s, &g) {
                        for c in 0..n {
                            let d = affine_delta(map, g.threads(), c);
                            assert!(d.is_some(), "{} {s:?} {map:?} copy {c}", p.name);
                            copies += 1;
                        }
                    }
                }
            }
        }
        assert!(
            copies > 1000,
            "the sweep reached the tiled kernels: {copies}"
        );
    }

    proptest! {
        /// Power-of-two extents make every lane map periodic in `threads`
        /// (`Q = threads / l` divides `rows` or `rows` divides `Q`).
        #[test]
        fn power_of_two_lane_maps_are_affine(
            t in 5u32..11, l in 0u32..4, rows in 0u32..9, unit in 0u32..3, p in 0u64..64,
        ) {
            let (threads, l, rows, unit) = (1u64 << t, 1u64 << l, 1u64 << rows, 1u64 << unit);
            for map in [
                LaneMap::RowMajor { per_row: l, unit },
                LaneMap::ColMajor { rows },
                LaneMap::Split { l, rows, unit },
            ] {
                prop_assert!(affine_delta(map, threads, p).is_some(), "{map:?}");
            }
        }
    }

    /// `pad`: on register staging each of a warp's `lanes` transposed-A store
    /// instructions hits 32 distinct banks; the S52 map (all `bk/lanes` lanes
    /// of a row in one warp, no pad) is a `bk/lanes`-way conflict — 4-way at
    /// the 4096² geometry the SASS review read.
    #[test]
    fn sync_a_store_is_conflict_free_under_pad() {
        let worst_way = |s: &TileSite, g: &GpuTile| {
            let [(map, n), _] = lane_maps(s, g);
            let lanes = GPU_VEC_BYTES / 4;
            let mut worst = 0;
            for warp in 0..g.threads() / 32 {
                for c in 0..n {
                    for j in 0..lanes {
                        let mut hits = [0; 32];
                        for lane in 0..32 {
                            let li = warp * 32 + lane + c * g.threads();
                            let (r, kc) = map.coords(&mut Num, &li);
                            hits[(((kc + j) * g.sa() + r) % 32) as usize] += 1;
                        }
                        worst = worst.max(*hits.iter().max().unwrap());
                    }
                }
            }
            worst
        };
        for p in PARTS {
            for wide_fma in [true, false] {
                let on = GpuRungs {
                    cp_async: false,
                    async_stage: false,
                    wide_fma,
                    ..GpuRungs::ALL
                };
                for n in [1024, 2048, 4096, 8192] {
                    let s = site(n, n, n, true, true, Ty::f32());
                    let g = launch(&s, p, false, force(on), None).tiled.expect("tiled");
                    assert_eq!((g.a_row_lanes, g.pad_a), (2, 4), "{} {n}", p.name);
                    assert_eq!(worst_way(&s, &g), 1, "{} {n} pad on", p.name);
                    // `bk / lanes` lanes per row ⇒ that many ways.
                    let off = launch(&s, p, false, force(GpuRungs { pad: false, ..on }), None);
                    let g = off.tiled.expect("tiled");
                    let ways = g.bk / (GPU_VEC_BYTES / 4);
                    assert_eq!(worst_way(&s, &g), ways, "{} {n} S52 map", p.name);
                    assert!(ways > 1);
                }
            }
        }
    }

    /// Arms with `names` forced off, the profile's table otherwise.
    fn off(names: &[&str]) -> crate::GpuArms {
        let mut off = GpuRungs::default();
        for n in names {
            *off.rung(n).expect("a listed name") = true;
        }
        crate::GpuArms {
            on: GpuRungs::default(),
            off,
            splitk: 0,
        }
    }

    /// S62: the M16 projection (16 × 2048 × 7168) was a one-thread-per-cell
    /// launch on every part (`gpu_tile` halves its thread grid past 16 rows).
    /// Now one output per thread in a 16 × 16 block of the same 128 × 256
    /// grid (`cell-fill`, shipped) — and, under `asyncstage` (an arm: its
    /// table rows ship it OFF), its panels streamed by `cp.async` through the
    /// stages the share left by ≤ 2 blocks per SM holds, each bound pinned.
    /// S62 integration: the geometry rungs (`fill-rows`, `row-fit`) give this
    /// site a register tile first; `cell-fill` is what it takes without them.
    #[test]
    fn starved_m16_site_takes_the_cell_tile_and_streams() {
        use crate::profile::RTX_3090;
        let s = site(16, 2048, 7168, true, true, Ty::f32());
        let base = off(&["fill-rows", "row-fit"]);
        for p in [&RTX_5090, &RTX_4090, &H100_PCIE, &RTX_3090] {
            let gpu = p.gpu().unwrap();
            for contract in [false, true] {
                let shipped = launch(&s, p, contract, base, None);
                let g = shipped
                    .tiled
                    .unwrap_or_else(|| panic!("{} cell tile", p.name));
                assert_eq!(
                    (g.tm, g.tn, g.threads(), g.stages),
                    (1, 1, 256, 1),
                    "{}",
                    p.name
                );
                assert_eq!(shipped.gx * shipped.gy, 128);
                let mut arms = base;
                arms.on.async_stage = true;
                let l = launch(&s, p, contract, arms, None);
                let g = l.tiled.unwrap_or_else(|| panic!("{} tiled", p.name));
                assert_eq!(
                    (g.tm, g.tn, g.threads(), l.gx * l.gy),
                    (1, 1, 256, 128),
                    "{}",
                    p.name
                );
                assert!(g.stages >= 2, "{} streams", p.name);
                let w = 128u64.div_ceil(gpu.sms);
                assert_eq!(g.blocks_per_sm, w, "{}", p.name);
                let share =
                    (gpu.smem_per_sm / w - gpu.smem_reserved_per_block).min(gpu.smem_per_block);
                let stage = g.stage_elems() * 4;
                assert!(g.stages * stage <= share, "{} fits", p.name);
                assert!(
                    g.stages == s.k / g.bk || (g.stages + 1) * stage > share,
                    "{} fills",
                    p.name
                );
                assert!(g.stages <= s.k / g.bk);
                assert_eq!(l.smem, g.stages * stage);
                let text = kernel_text("m16", &s, p, contract, arms, None, false, None, false);
                assert!(
                    text.contains("@llvm.nvvm.cp.async.cg.shared.global.16"),
                    "{}",
                    p.name
                );
                let wait = format!("@llvm.nvvm.cp.async.wait.group(i32 {})", g.stages - 2);
                assert!(text.contains(&wait), "{}", p.name);
                // The fold stays one in-order chain per cell.
                assert_eq!(text.contains("@llvm.fma.f32"), contract);
            }
        }
    }

    /// Every S62 tile rung off (these two and the geometry rungs) reproduces
    /// the launches emitted before S62 (the selection table's literals): one
    /// thread per cell at M16, the register-staged 4 × 4 tile at M128 on the
    /// RTX 5090.
    #[test]
    fn the_off_arms_reproduce_todays_launch() {
        let arms = off(&[
            "cell-fill",
            "asyncstage",
            "row-fit",
            "wave-gate",
            "grid-share",
            "panel-regs",
            "fill-rows",
        ]);
        let s = site(16, 2048, 7168, true, true, Ty::f32());
        let l = launch(&s, &RTX_5090, false, arms, None);
        assert_eq!(
            (l.tiled, l.gx, l.gy, l.block, l.smem),
            (None, 128, 1, 256, 0)
        );
        let s = site(128, 2048, 7168, true, true, Ty::f32());
        for (contract, smem) in [(false, 2048), (true, 2176)] {
            let l = launch(&s, &RTX_5090, contract, arms, None);
            let g = l.tiled.expect("tiled");
            assert_eq!(
                (g.stages, l.gx, l.gy, l.block, l.smem),
                (1, 64, 4, 64, smem)
            );
        }
    }

    /// A grid whose wave fills every block slot its tile was budgeted for
    /// keeps the text it had: `asyncstage` derives nothing there.
    #[test]
    fn filled_grids_are_untouched() {
        use crate::profile::GPU_MEASURED;
        for p in GPU_MEASURED {
            for (m, n, k) in [(4096, 4096, 4096), (16, 18432, 7168)] {
                let s = site(m, n, k, true, true, Ty::f32());
                for contract in [false, true] {
                    let text = |arms| kernel_text("f", &s, p, contract, arms, None, false, None, false);
                    assert_eq!(
                        text(crate::GpuArms::default()),
                        text(off(&["asyncstage"])),
                        "{} {m}x{n}x{k}",
                        p.name
                    );
                }
            }
        }
    }

    /// One-row sites keep the one-thread-per-cell kernel: a 1 × 256 block
    /// copies whole vectors only at `bk ≥ 1024`, which no share holds.
    #[test]
    fn m1_keeps_the_naive_kernel() {
        use crate::profile::GPU_MEASURED;
        let s = site(1, 2048, 7168, true, true, Ty::f32());
        for p in GPU_MEASURED {
            for contract in [false, true] {
                let l = launch(&s, p, contract, crate::GpuArms::default(), None);
                assert_eq!(l.tiled, None, "{}", p.name);
            }
        }
    }
}
