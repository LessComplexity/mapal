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
use crate::profile::{
    Face, GPU_RESIDENT_BLOCKS, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, MMA_K, MMA_M,
    MMA_N, NAIVE_UNROLL, PlanComb, PlanCost, PlanShape, WARPGROUP_WARPS, WGMMA_ATOM_ROWS, WGMMA_K,
    WGMMA_M, WGMMA_SWIZZLE_BYTES, bank_degree, fma_held_regs,
};
use crate::{Order, Reduced};
use mapal_ir::{RowFoldOp, RowFoldSite};

/// Bytes one shared-memory wavefront serves: 32 banks × 4 B. An ISA rule on
/// every NVIDIA part since Kepler.
const GPU_WAVEFRONT_BYTES: u64 = 128;
/// The lanes of one warp instruction the bank model ([`panel_excess`]) reads:
/// 32 on every NVIDIA part, as are the 32 banks of [`bank_degree`].
const WARP: u64 = 32;

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
/// cpu-gpu-boundary `preload`: resolve every kernel when the module loads.
pub(crate) const GPU_INIT_PRELOAD: u32 = 8;
/// cpu-gpu-boundary `stage`: the page-locked staging ring for async uploads.
pub(crate) const GPU_INIT_STAGE: u32 = 16;

/// `main`'s first call in a module with a GPU site: hands the runtime its host
/// rungs. Off, the runtime does nothing until the first site, the S52 path.
pub(crate) fn init_call(rungs: crate::GpuRungs, streams: bool, pdl: bool) -> String {
    let flags = u32::from(rungs.eager_init) * GPU_INIT_EAGER
        + u32::from(streams) * GPU_INIT_STREAMS
        + u32::from(pdl) * GPU_INIT_PDL
        + u32::from(rungs.preload) * GPU_INIT_PRELOAD
        + u32::from(rungs.stage && rungs.async_h2d) * GPU_INIT_STAGE;
    format!("  call void @mapal_gpu_init(i32 {flags})\n")
}

/// S54 `pdl`: a programmatic dependent launch needs `griddepcontrol`
/// ([`crate::profile::Caps::griddepcontrol`]). One answer for the kernel text
/// and the runtime flag — a kernel launched dependent without the wait would
/// race.
pub(crate) fn pdl(profile: &TargetProfile, contract: bool, arms: crate::GpuArms) -> bool {
    profile.gpu().is_some_and(|g| g.has_griddepcontrol()) && host_rungs(profile, contract, arms).pdl
}

/// The rungs a GPU profile ships for one decision: the rung rules over its
/// physical properties at `wps` warps per scheduler on this face
/// ([`crate::rules::rungs`]) — every rung under the arm
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
    let r = table.with(arms);
    // per-site-choice D1: `bank-plan` on, the register rule (A4,
    // [`launch_split`]) decides the k-major panel instead of `kmajor-a`'s rule.
    crate::GpuRungs {
        kmajor_a: r.kmajor_a || (r.bank_plan && !arms.off.kmajor_a),
        ..r
    }
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

/// kernel-merging `calls`: `mapal_gpu_launch_coop` — `launchv`'s arguments,
/// launched cooperatively (every block co-resident). Declared only when called.
pub(crate) const COOP_DECL: &str =
    "declare void @mapal_gpu_launch_coop(i64, i32, i32, ptr, i32, i32, i32, i32)\n";

/// S54 entries (`streams`, `pinned`), each declared only where it is called,
/// so a module that never opens a stream or pins keeps its S53 text.
/// (`launchv` is [`LAUNCHV_DECL`], shared with `epilogue` and `devmap`.)
pub(crate) const S54_DECLS: [(&str, &str); 7] = [
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
    // S63 `splitk-atomic` / `streamk`: the output zeroed before its partials add.
    (
        "@mapal_gpu_zero(",
        "declare void @mapal_gpu_zero(i64, i64)\n",
    ),
    // cpu-gpu-boundary `async-h2d`: an upload enqueued without a host wait.
    (
        "@mapal_gpu_h2d_async(",
        "declare void @mapal_gpu_h2d_async(i64, ptr, i64)\n",
    ),
    // `chain-h2d`: an upload on a chain's stream.
    (
        "@mapal_gpu_h2d_on(",
        "declare void @mapal_gpu_h2d_on(i32, i64, ptr, i64)\n",
    ),
];

/// Threads per block of a `devgen` kernel — the runtime's `MAPAL_GEN_THREADS`
/// launches it with exactly this many, so the two must agree.
/// ponytail: the naive kernel's literal, for the same reason (a one-element-
/// per-thread fill is bandwidth-bound at any block size a part accepts).
const GEN_THREADS: u64 = NAIVE_THREADS;

/// Threads per block of the one-thread-per-cell kernel.
/// ponytail: a literal, not derived — the fallback is a correctness path; derive
/// it when an untiled site turns out to matter for speed.
pub(crate) const NAIVE_THREADS: u64 = 256;

/// Can this site run on the GPU at all? Any float site whose reads are affine
/// in `(i, lane, k)`, or (generic-fusion `window`) in the fold's derived axes
/// `(k÷div, k%div)` — conv taps, which the one-thread-per-cell kernel
/// addresses. `window` off: those stay on the host.
pub(crate) fn gpu_site(site: &TileSite, profile: &TargetProfile, window: bool) -> bool {
    profile.gpu().is_some()
        && is_float(&site.elem)
        && (window || (site.a.ksplit.is_none() && site.b.ksplit.is_none()))
        && site.k > 0
        && site.rows * site.c > 0
}

/// **S62 `place`: the tile sites of `f` that run on the device** — of
/// `plan`, the sites as the caller walks them. One answer for every reader
/// (the host walk, the residency plan, retention, row folds, the device
/// unit): a site one of them placed and another did not would be a kernel
/// reading a buffer nobody wrote — the [`super::packing_site`] lesson.
///
/// In walk order, a capable site ([`gpu_site`]) is a device site when an
/// operand is device-produced (`fed`: the host placement would copy it out —
/// the resident plan's whole point) or when its roofline says the device
/// finishes first ([`TargetProfile::device_pays`]: `2·rows·c·k` flops over
/// [`crate::profile::tile_bytes`], the dispatch rule's arithmetic). Without
/// measured costs every capable site is placed: the S52–S54 rule.
pub(crate) fn placement(
    ir: &CategoryIr,
    f: FuncId,
    plan: &TilePlan,
    profile: &TargetProfile,
    window: bool,
) -> SecondaryMap<MorphismId, ()> {
    let mut dev = SecondaryMap::new();
    if profile.gpu().is_none() {
        return dev;
    }
    for m in ir.topo_order(f) {
        let Some(site) = plan.sites.get(m).filter(|s| gpu_site(s, profile, window)) else {
            continue;
        };
        let src = ir.morphism(m).expect("site resolves").source;
        let fed = [site.a.slot, site.b.slot]
            .into_iter()
            .any(|k| super::pair_source_ir(ir, src, k).is_some_and(|o| device_made(ir, &dev, o)));
        let flops = 2u64
            .saturating_mul(site.rows)
            .saturating_mul(site.c)
            .saturating_mul(site.k);
        let bytes = crate::profile::tile_bytes(site.rows, site.c, site.k, &site.elem);
        if fed || profile.device_pays(flops, bytes) {
            dev.insert(m, ());
        }
    }
    dev
}

/// `o` is produced on the device: by a placed site, or by a map or fold over
/// an array that is (a per-cell kernel, a row fold, an epilogue member — the
/// runs `devmap`, `rowfold` and `epilogue` keep beside their producer).
/// ponytail: a map that reads a device array but stays on the host counts too;
/// that over-approximation only ever keeps a reader on the device (today).
pub(crate) fn device_made(
    ir: &CategoryIr,
    dev: &SecondaryMap<MorphismId, ()>,
    o: ObjectId,
) -> bool {
    let Some(&p) = ir.in_edges(o).first() else {
        return false;
    };
    if dev.contains_key(p) {
        return true;
    }
    let morph = ir.morphism(p).expect("morphism resolves");
    if !matches!(morph.op, Operation::Map { .. } | Operation::Fold { .. }) {
        return false;
    }
    let pairs: Vec<ObjectId> = ir
        .in_edges(morph.source)
        .iter()
        .map(|&q| ir.morphism(q).expect("morphism resolves"))
        .filter(|q| matches!(q.op, Operation::Pair { .. }))
        .map(|q| q.source)
        .collect();
    if pairs.is_empty() {
        device_made(ir, dev, morph.source)
    } else {
        pairs.into_iter().any(|x| device_made(ir, dev, x))
    }
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
    contract: bool,
    arms: crate::GpuArms,
) -> TilePlan {
    let rungs = host_rungs(profile, contract, arms);
    if profile.gpu().is_none() {
        return ir.tile_plan(f);
    }
    let strict = ir.tile_plan(f);
    let mut plan = if rungs.lane_stride {
        ir.tile_plan_strided(f)
    } else {
        strict.clone()
    };
    // recurrence-precision `radix-tile`: a mixed-radix site where the staged
    // tile takes it (its kernels address digits; the naive kernel too, but a
    // site the tile refuses keeps the `nest-cell` it had).
    if rungs.radix_tile {
        for (m, s) in ir.tile_plan_radix(f) {
            let l = launch(&s, profile, contract, arms);
            let staged = l.tiled.is_some() || l.mma.is_some();
            if staged && gpu_site(&s, profile, rungs.window) {
                plan.sites.insert(m, s);
            }
        }
    }
    // S62: an extra is a site only where it is placed on the device.
    let dev = placement(ir, f, &plan, profile, rungs.window);
    let extra: Vec<MorphismId> = plan
        .sites
        .iter()
        .filter(|&(m, _)| !strict.sites.contains_key(m) && !dev.contains_key(m))
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
    fn scalar(site: &TileSite, contract: bool, free: bool, pairs: bool) -> Self {
        // fold-algebra: a tile site's step is `acc + a·b` with the product read
        // once — a `step_algebra` FMA pair by construction — and the host fuses
        // every such pair under `contract` (`FnEmit::emit_arith`, the same
        // emitter the device cells use), so the device fuses whatever B's lane
        // stride or radix form: admission is the algebra, not the geometry.
        // OFF arm (`--no-fma-pairs`): the host's generic steps stay separate,
        // so a strided-lane site does too; a radix site fuses where the order
        // is free (recurrence-precision `radix-tile`: its `_c` oracle replays
        // separate steps).
        let fused = pairs
            || match site.radix {
                Some(_) => free,
                None => site.b.clane == 1,
            };
        if contract && fused {
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
    /// How the site's `k` fold is realized (ADR-0040 D4): in order, or cut
    /// and combined. [`launch`] gives `None`; [`launch_in`] chooses under
    /// the emission's order.
    pub combine: Combine,
}

/// The realization of a tile site's `k` fold (ADR-0040 D3/D4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Combine {
    /// One chain per cell, increasing `k`: the ordered kernel.
    None,
    /// S62 `splitk`: `s` partial folds (`z` in `ctaid.y` above the unsplit
    /// `gy`) into a workspace, then one reduce launch in partition order.
    Workspace { s: u64 },
    /// S63 `splitk-atomic`: the same `s` partial folds, each added into the
    /// zeroed output by `atomicrmw fadd` (combine order unspecified).
    Atomic { s: u64 },
    /// S63 `streamk`: `ctas` blocks, each walking `iters` consecutive units
    /// of the `tiles × panels` iteration space; a tile's pieces add into the
    /// zeroed output by `atomicrmw fadd`.
    StreamK { ctas: u64, iters: u64 },
}

impl Combine {
    /// The partitions a split kernel folds (`z` extent); 1 otherwise.
    pub(crate) fn splits(self) -> u64 {
        match self {
            Combine::Workspace { s } | Combine::Atomic { s } => s,
            _ => 1,
        }
    }
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
) -> Launch {
    launch_split(site, profile, contract, arms, 1, false)
}

/// [`launch`] for the kernel that folds `split` partitions (site-geometry
/// `fill-wave`): the FMA tile's panels and stages are sized for the
/// `blocks · split` grid ([`TargetProfile::gpu_tile_split`]). 1 = [`launch`].
/// `full` (tile-efficiency T1): the split grid keeps the full register tile.
pub(crate) fn launch_split(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    split: u64,
    full: bool,
) -> Launch {
    let e = elem_bytes_of(&site.elem);
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    let (blocks, threads) = fill_grid(site, profile);
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
    // recurrence-precision `tf32-tile`: the permission is the site's product
    // set (ADR-0041 D6); off, the f32 FFMA tile it also admits.
    // A mixed-radix site stores digit by digit: site-geometry `tf32-radix`
    // gives the `mma.sync` tile element copies and stores for its
    // digit-affine addresses (`wgmma`'s swizzled 16-byte panels have none).
    let rx = site.radix.as_deref();
    let radix_ok = |rungs: crate::GpuRungs| rx.is_none() || rungs.tf32_radix;
    if site_reduced(site) == Some(Reduced::Tf32)
        && site.elem == Ty::f32()
        && shaped
        && (vector_a(site) || rx.is_some_and(|x| x.a.is_some()))
        && (vector_b(site) || rx.is_some_and(|x| x.b.is_some()))
    {
        let rungs = rungs_at(profile, wps, Face::Tf32, arms);
        if rungs.mma && rungs.tf32_tile && radix_ok(rungs) {
            let wgmma = (rungs.wgmma && rx.is_none())
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
                    combine: Combine::None,
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
                    combine: Combine::None,
                };
            }
        }
    }
    let arithmetic = Arithmetic::scalar(site, contract, arms.free, arms.pairs);
    let rungs = rungs_at(profile, wps, arithmetic.face(), arms);
    // tile-efficiency T2: `a`'s panel rows run along `k` (`vector_a`), so it
    // can stay untransposed — where one quarter-warp's fragment read is one
    // panel row (`tx` ≥ the lanes a wavefront serves, [`Kernel::tiled`]).
    let kmajor = rungs.kmajor_a && vector_a(site) && rx.is_none_or(|x| x.a.is_none());
    let tile_rungs = crate::GpuRungs {
        kmajor_a: kmajor,
        full_tile: full,
        ..rungs
    };
    // per-site-choice A2: the k-major panel's bank pad, where it fits — on
    // each pipeline's panel before they are weighed, and on the final stages.
    let kpad = |g: GpuTile| match g.kmajor && rungs.bank_plan {
        true => {
            profile.gpu_tile_kmajor_pad(g, site.k, &site.elem, |at| kmajor_pad(site, at, &rungs))
        }
        false => g,
    };
    let tiled = shaped
        .then(|| {
            profile.gpu_tile_split(
                site.rows,
                site.c,
                site.k,
                &site.elem,
                rungs.wide_fma,
                tile_rungs,
                split,
            )
        })
        .flatten()
        // S62 `cell-fill`: no register tile divides the site — one output per
        // thread, still staged ([`TargetProfile::gpu_cell_tile`]).
        .or_else(|| {
            (shaped && rungs.cell_fill)
                .then(|| profile.gpu_cell_tile(site.rows, site.c, site.k, &site.elem))
                .flatten()
        })
        .map(|g| GpuTile {
            kmajor: kmajor && g.tx >= GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES,
            ..g
        })
        .map(|g| {
            // ADR-0034 D1: a forced stage count picks the pipeline (1 =
            // register staging, >= 2 = `cp.async`) over the rung.
            let stream = match gpu.geom.stages {
                0 => rungs.cp_async,
                s => s >= 2,
            };
            let streamed = stream
                .then(|| profile.gpu_tile_async(g, site.k, &site.elem))
                .flatten()
                .map(kpad);
            // tile-efficiency T3 (`async-depth`): a thread does `tm·tn·bk ≥
            // 1024` FFMA per barrier and wait. A `cp.async` panel shallower
            // than that yields to register staging only where staging's
            // panel is deeper (`k = 8·odd` offers neither a deeper one); a
            // forced ADR-0034 depth stands.
            let floor = 1024 / (g.tm * g.tn).max(1);
            // per-site-choice D2: a forced `async` arm takes the panel anyway.
            let shallow = |a: &GpuTile| {
                rungs.async_depth && gpu.geom.bk == 0 && a.bk < floor && !arms.on.cp_async
            };
            if let Some(a) = streamed.filter(|a| !shallow(a)) {
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
            let grid = (site.rows / g.bm()) * (site.c / g.bn()) * split;
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
            let staged = if rungs.pad && vector_a(site) && !g.kmajor {
                profile.gpu_tile_sync_a(g, &site.elem)
            } else {
                kpad(g)
            };
            match streamed {
                Some(a) if a.bk >= staged.bk => a,
                _ => staged,
            }
        })
        // S62 `asyncstage`: stages from the blocks this grid puts on an SM.
        // Legal for the unpredicated copies above: `rows % bm`, `c % bn`,
        // `k % bk` are 0 (`TileSite` extents) and `shaped` makes each panel
        // one read per block; the panel order `0..k/bk` and each cell's step
        // order are unchanged ([`Kernel::step`]).
        .map(|g| {
            let blocks = (site.rows / g.bm()) * (site.c / g.bn()) * split;
            // per-site-choice D2: `async` forced off is register staging.
            (rungs.async_stage && gpu.geom.stages == 0 && !arms.off.cp_async)
                .then(|| profile.gpu_tile_stages(g, blocks, site.k, &site.elem))
                .flatten()
                .unwrap_or(g)
        })
        .map(kpad);
    // per-site-choice A4 (register rule): the k-major panel is taken where its
    // geometry is no worse than the transposed panel's — `bk` and blocks/SM,
    // which carry its `lanes − 1` extra fragment registers a row
    // ([`TargetProfile::gpu_tile_split`]) — and its `a` panel conflict-free
    // (`adeg` 1: no pad fit, [`TargetProfile::gpu_tile_kmajor_pad`]).
    // Elsewhere the transposed tile, the kernel's rungs unchanged. An arm
    // that forces `kmajor-a` on takes the k-major tile wherever it applies:
    // the rule's alternative (per-site-choice Part B) and its ON arm. A tile
    // too narrow for the k-major read (`tx`) is the transposed one either way.
    if let Some(k) = tiled.filter(|_| kmajor && rungs.bank_plan) {
        let [a, _, st, _] = panel_excess(site, &k, &rungs, false);
        let mut flat = arms;
        *flat.on.rung("kmajor-a").expect("a rung") = false;
        *flat.off.rung("kmajor-a").expect("a rung") = true;
        let t = launch_split(site, profile, contract, flat, split, full);
        let ruled = |t: &GpuTile| {
            arms.on.kmajor_a
                || (a.max(st) == 1 && k.bk >= t.bk && k.blocks_per_sm >= t.blocks_per_sm)
        };
        if t.tiled.is_some_and(|t| !(k.kmajor && ruled(&t))) {
            return Launch { rungs, ..t };
        }
    }
    match tiled {
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
            combine: Combine::None,
        },
        None => Launch {
            arithmetic,
            tiled: None,
            mma: None,
            wgmma: None,
            // One thread per cell: its own grid, not the fill key's (which
            // keys a tile the launch may not take — a `--geom=` arm).
            gx: (site.rows * site.c).div_ceil(NAIVE_THREADS),
            gy: 1,
            block: NAIVE_THREADS,
            smem: 0,
            raster: 1,
            rungs,
            combine: Combine::None,
        },
    }
}

/// The grid a site's fill is keyed by (`launch`'s `wps`): the S52 geometry
/// (no S53 geometry rung) — the table's rows were measured on grids
/// classified by it — or the one-thread-per-cell grid.
fn fill_grid(site: &TileSite, profile: &TargetProfile) -> (u64, u64) {
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    let base = shaped
        .then(|| profile.gpu_tile(site.rows, site.c, site.k, &site.elem, false, false))
        .flatten();
    match base {
        Some(g) => ((site.rows / g.bm()) * (site.c / g.bn()), g.threads()),
        None => ((site.rows * site.c).div_ceil(NAIVE_THREADS), NAIVE_THREADS),
    }
}

/// S64 evidence pass: one `; mapal.plan` comment line per kernel — what the
/// emitter derived for it (geometry, fill key, face, index width, the
/// kernel's rungs), the intent `benches/gpu/evidence.py` checks the PTX and
/// SASS against. A comment: `llc` drops it, the kernel is unchanged. Only the
/// kernel rungs are listed, so a host rung's arm leaves the device text alone.
#[allow(clippy::too_many_arguments)]
fn plan_line(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    l: &Launch,
    idx: &str,
    combine: Combine,
    col: bool,
    epi: bool,
) -> String {
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let (blocks, threads) = fill_grid(site, profile);
    let kind = match (l.wgmma, l.mma, l.tiled) {
        (Some(_), _, _) => "wgmma",
        (None, Some(_), _) => "mma",
        (None, None, Some(_)) => "tile",
        _ => "naive",
    };
    let mut f = format!(
        "; mapal.plan kernel={name} kind={kind} face={:?} idx={idx} rows={} c={} k={} wps={} \
         gx={} gy={} block={} smem={} raster={}",
        l.arithmetic,
        site.rows,
        site.c,
        site.k,
        gpu.warps_per_scheduler(blocks, threads),
        l.gx,
        l.gy,
        l.block,
        l.smem,
        l.raster
    );
    if let Some(g) = l.tiled {
        f += &format!(
            " tm={} tn={} tx={} ty={} bk={} stages={} pad_a={} held={} blocks_per_sm={} lsu={} \
             lsu_issue={} kmajor={} adeg={}",
            g.tm,
            g.tn,
            g.tx,
            g.ty,
            g.bk,
            g.stages,
            g.pad_a,
            g.held,
            g.blocks_per_sm,
            lsu_per_panel(site, &g, &l.rungs, col),
            // per-site-choice D4: the instructions alone, what SASS shows
            // (intent.py `lsu`).
            lsu_per_panel(
                site,
                &g,
                &crate::GpuRungs {
                    bank_plan: false,
                    ..l.rungs
                },
                col
            ),
            u8::from(g.kmajor && !col),
            // per-site-choice A5: the `a` panel's worst bank excess, fragment
            // read and copy write (intent.py `kmajor-a`: 1).
            {
                let [a, _, st, _] = panel_excess(site, &g, &l.rungs, col);
                a.max(st)
            }
        );
    }
    if let Some(m) = l.mma {
        f += &format!(
            " mt={} nt={} warps_x={} warps_y={} bk={} stages={} blocks_per_sm={}",
            m.mt, m.nt, m.warps_x, m.warps_y, m.bk, m.stages, m.blocks_per_sm
        );
    }
    if let Some(w) = l.wgmma {
        f += &format!(
            " n={} groups={} bk={} stages={} blocks_per_sm={}",
            w.n, w.groups, w.bk, w.stages, w.blocks_per_sm
        );
    }
    // generic-fusion `window`: a window site's tap grid, `(k÷div) × div`
    // (intent.py `window`); no other kernel's line changes.
    if let Some(ks) = site.a.ksplit.as_ref().or(site.b.ksplit.as_ref()) {
        f += &format!(" taps={}x{}", site.k / ks.div, ks.div);
    }
    // recurrence-precision `radix-tile`: which addresses are mixed-radix
    // (intent.py `radix`); no other kernel's line changes.
    if let Some(r) = site.radix.as_deref() {
        let parts = [("a", &r.a), ("b", &r.b), ("out", &r.out)];
        let on: Vec<&str> = parts
            .iter()
            .filter(|(_, x)| x.is_some())
            .map(|(n, _)| *n)
            .collect();
        f += &format!(" radix={}", on.join(","));
    }
    // recurrence-precision `tf32-tile`: the site's product set when it admits
    // TF32 (ADR-0041 D5 `set`, intent.py `tf32-tile`); f32-only lines unchanged.
    if site.tf32 {
        f += " set=tf32|f32";
    }
    let (c, s, ctas, iters) = match combine {
        Combine::None => ("none", 1, 0, 0),
        Combine::Workspace { s } => ("splitk", s, 0, 0),
        Combine::Atomic { s } => ("atomic", s, 0, 0),
        Combine::StreamK { ctas, iters } => ("streamk", 1, ctas, iters),
    };
    // site-geometry `fill-wave`: the target its split grid reaches (intent.py).
    if l.rungs.fill_wave && s > 1 {
        f += &format!(" fill_warps={}", gpu.fill_warps());
    }
    let on: Vec<&str> = KERNEL_RUNGS
        .iter()
        .copied()
        .filter(|n| l.rungs.get(n))
        .collect();
    format!(
        "{f} combine={c} S={s} ctas={ctas} iters={iters} col={} epi={} rungs_on={}\n",
        u8::from(col),
        u8::from(epi),
        on.join(",")
    )
}

/// The rungs a kernel's text reads ([`plan_line`]); the rest are host rungs.
const KERNEL_RUNGS: [&str; 37] = [
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
    "raster",
    "l2-gate",
    "pdl",
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
    "row-split",
    "fill-wave",
    "fit-first",
    "tf32-radix",
    "kmajor-a",
    "full-tile",
    "async-depth",
    "bank-plan",
];

/// [`launch`] with the fold's realization chosen under `order` (ADR-0040 D4).
pub(crate) fn launch_in(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> Launch {
    let l = launch(site, profile, contract, arms);
    let cands = match l.rungs.plan_cost {
        true => plan_cands(&l, site, profile, contract, arms, order),
        false => Vec::new(),
    };
    let c = match cands.first() {
        Some(pick) => pick.combine,
        None => combine(&l, site, profile, order, arms),
    };
    let mut l = resized(site, profile, contract, arms, l, c);
    l.combine = c;
    l
}

/// One candidate plan of a site's chooser: its fold realization, the launch
/// it emits and the cost model's reading of both (plan-cost §2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cand {
    pub combine: Combine,
    pub shape: PlanShape,
    pub cost: PlanCost,
}

impl Cand {
    /// The candidate's name on the plan line.
    pub(crate) fn label(&self) -> String {
        match self.combine {
            Combine::None => "none".into(),
            Combine::Atomic { s } => format!("atomic{s}"),
            Combine::Workspace { s } => format!("splitk{s}"),
            Combine::StreamK { ctas, iters } => format!("streamk{ctas}x{iters}"),
        }
    }
}

/// **plan-cost: the chooser.** On a part with calibrated plan rates
/// ([`TargetProfile::plan_rates`]), a free-order scalar
/// site's candidates — the ordered kernel, the atomic and the workspace split
/// at every `S` dividing the panels, stream-K, each at the geometry its split
/// grid takes ([`resized`]), each admitted by its own rung as in [`combine`],
/// and the rules' own pick — ranked by [`TargetProfile::site_ns`], cheapest
/// first (a tie keeps the earlier: the ordered kernel, then the rules').
/// A forced count leaves only its own splits. Empty where the rules decide:
/// another order, a part without the rates, the tensor-core kernels, a window
/// (whose `k` is never cut) and a forced count no split admits. [`launch_in`]
/// takes the first under rung `plan-cost`; its OFF arm keeps the rules' plan
/// and still prints the ranking (the fit harness reads every arm's).
pub(crate) fn plan_cands(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> Vec<Cand> {
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || site.radix.is_some();
    let rungs = l.rungs;
    if order != Order::Free
        || profile.plan_rates().is_none()
        || l.mma.is_some()
        || l.wgmma.is_some()
        || window
    {
        return Vec::new();
    }
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let e = elem_bytes_of(&site.elem);
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let blocks = l.gx * l.gy;
    let wave = l.wave(gpu);
    let atomic = e == 4 || gpu.caps.red_f64;
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    // A forced count (`--splitk=S`, the part's `Geom::split`) is the only
    // plan: the ordered kernel and the rules' pick are not candidates.
    let mut combos = match forced {
        0 => vec![Combine::None, combine(l, site, profile, order, arms)],
        _ => Vec::new(),
    };
    for s in (2..=panels.min(64)).filter(|&s| panels.is_multiple_of(s)) {
        if forced > 0 && s != forced {
            continue;
        }
        if rungs.splitk_atomic && atomic && l.raster == 1 {
            combos.push(Combine::Atomic { s });
        }
        if rungs.splitk && l.raster == 1 && s * site.rows * site.c * e <= gpu.l2_bytes {
            combos.push(Combine::Workspace { s });
        }
    }
    // Stream-K at the rules' one wave (see [`combine`]).
    if forced == 0
        && rungs.streamk
        && atomic
        && l.tiled.is_some_and(|g| g.stages < 2)
        && panels >= 2
        && blocks * panels > wave
    {
        let iters = (blocks * panels).div_ceil(wave);
        combos.push(Combine::StreamK {
            ctas: (blocks * panels).div_ceil(iters),
            iters,
        });
    }
    let mut seen = Vec::new();
    let mut cands: Vec<Cand> = combos
        .into_iter()
        .filter(|c| {
            let new = !seen.contains(c);
            seen.push(*c);
            new
        })
        .filter_map(|c| {
            let r = resized(site, profile, contract, arms, *l, c);
            let shape = plan_shape(site, &r, c, gpu);
            let cost = profile.site_ns(&shape)?;
            Some(Cand {
                combine: c,
                shape,
                cost,
            })
        })
        .collect();
    cands.sort_by(|a, b| a.cost.ns.total_cmp(&b.cost.ns));
    cands
}

/// The cost model's reading of launch `l` folding the site as `c`.
pub(crate) fn plan_shape(site: &TileSite, l: &Launch, c: Combine, gpu: &Gpu) -> PlanShape {
    // A cp.async pipeline keeps `stages − 1` panels in flight; the
    // register-staged tile its one prefetch panel; the naive kernel one
    // unrolled trip.
    let (bm, bn, acc, per_sm, bk, step, fly, lsu) = match l.tiled {
        Some(g) => (
            g.bm(),
            g.bn(),
            g.tm * g.tn,
            g.blocks_per_sm,
            g.bk,
            g.bk,
            g.stages.saturating_sub(1).max(1),
            lsu_per_panel(site, &g, &l.rungs, false),
        ),
        None => (1, l.block, 1, l.wave(gpu) / gpu.sms, 1, NAIVE_UNROLL, 1, 0),
    };
    let tiles = l.gx * l.gy;
    let (blocks, kb, s, comb) = match c {
        Combine::None => (tiles, site.k, 1, PlanComb::None),
        Combine::Atomic { s } => (tiles * s, site.k / s, s, PlanComb::Atomic),
        Combine::Workspace { s } => (tiles * s, site.k / s, s, PlanComb::Workspace),
        Combine::StreamK { ctas, iters } => (ctas, iters * bk, 1, PlanComb::StreamK),
    };
    PlanShape {
        rows: site.rows,
        c: site.c,
        k: site.k,
        e: elem_bytes_of(&site.elem),
        bm,
        bn,
        threads: l.block,
        per_sm,
        acc,
        tiles,
        blocks,
        kb,
        s,
        comb,
        bk: step,
        fly,
        lsu,
    }
}

/// plan-cost: what a site's plan line adds — ` pick= ns= alt=<cand>:<ns>
/// cross=x<scale>` (the runner-up and the scale where it takes over,
/// [`TargetProfile::crossover`]) — and, under `--plan-cands` on a row-major
/// kernel of the site (`cands_here`), one `; mapal.cand` line per candidate,
/// keyed by the site's ordered [`kernel_id`].
/// Empty where the rules decide.
fn plan_pick(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    cands_here: bool,
) -> String {
    let order = if arms.free {
        Order::Free
    } else {
        Order::Sequential
    };
    let l = launch(site, profile, contract, arms);
    let cands = plan_cands(&l, site, profile, contract, arms, order);
    let Some(pick) = cands.first() else {
        return String::new();
    };
    let id = kernel_id(site, profile, contract);
    let mut f = format!(
        " site={id:016x} pick={} ns={:.1}",
        pick.label(),
        pick.cost.ns
    );
    match cands.get(1) {
        Some(alt) => {
            let cross = profile
                .crossover(&pick.shape, &alt.shape)
                .map_or("none".into(), |x| format!("x{x}"));
            f += &format!(" alt={}:{:.1} cross={cross}", alt.label(), alt.cost.ns);
        }
        None => f += " alt=none cross=none",
    }
    if arms.cands && cands_here {
        // The rates the candidates were costed at (the harness's mirror).
        let g = profile.gpu().expect("a GPU site has a GPU profile");
        f += "\n; mapal.rates";
        for name in crate::profile::DeviceCost::NAMES {
            f += &format!(" {name}={}", g.rate(name));
        }
        f += &format!(
            " sms={} warp={} schedulers={}\n",
            g.sms,
            g.warp,
            g.schedulers()
        );
        for (i, c) in cands.iter().enumerate() {
            let (p, t) = (c.shape, c.cost);
            f += &format!(
                "; mapal.cand site={:016x} cand={} pick={} ns={:.1} rows={} c={} k={} e={} \
                 bm={} bn={} threads={} per_sm={} acc={} tiles={} blocks={} kb={} s={} \
                 bk={} fly={} lsu={} ffma_lsu={:.2} waves={} tail={} wps={:.3} f={:.1} l={:.1} \
                 d={:.1} chain={:.1} u={:.1} kernel={:.1} combine={:.1} fixed={:.1} launches={} \
                 zeros={} atomics={}",
                id,
                c.label(),
                u8::from(i == 0),
                t.ns,
                p.rows,
                p.c,
                p.k,
                p.e,
                p.bm,
                p.bn,
                p.threads,
                p.per_sm,
                p.acc,
                p.tiles,
                p.blocks,
                p.kb,
                p.s,
                p.bk,
                p.fly,
                p.lsu,
                (p.acc * p.bk) as f64 / p.lsu.max(1) as f64,
                t.waves,
                t.tail,
                t.wps,
                t.f_ns,
                t.l_ns,
                t.d_ns,
                t.c_ns,
                t.u_ns,
                t.kernel_ns,
                t.combine_ns,
                t.fixed_ns,
                t.launches,
                t.zeros,
                t.atomics
            );
            if i + 1 < cands.len() {
                f.push('\n');
            }
        }
    }
    f
}

/// `l` re-sized for the split grid of `combine` under `fill-wave`
/// ([`launch_split`]) — kept only when the re-sized panel still cuts each
/// partition's `k / S` whole (the split kernel walks `kext / bk` panels).
fn resized(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    l: Launch,
    combine: Combine,
) -> Launch {
    let s = combine.splits();
    let whole = |r: &Launch| r.tiled.is_some_and(|g| (site.k / s).is_multiple_of(g.bk));
    let rule = match l.rungs.fill_wave && s > 1 {
        true => Some(launch_split(site, profile, contract, arms, s, false))
            .filter(whole)
            .unwrap_or(l),
        false => l,
    };
    // tile-efficiency T1: where the plan is the cost model's (`plan-cost`, a
    // part with its rates) and the order free, a split kernel may fill the
    // machine along `k` with the full register tile instead — a candidate
    // beside the rules' (shrunk) tile at the same `S`, taken where
    // [`TargetProfile::site_ns`] says it is cheaper (never a smaller tile).
    let full = s > 1 && arms.free && l.rungs.full_tile && l.rungs.plan_cost && l.tiled.is_some();
    let Some(gpu) = profile.gpu().filter(|_| full) else {
        return rule;
    };
    let r = launch_split(site, profile, contract, arms, s, true);
    let acc = |x: &Launch| x.tiled.map_or(0, |g| g.tm * g.tn);
    let ns = |x: &Launch| {
        profile
            .site_ns(&plan_shape(site, x, combine, gpu))
            .map(|c| c.ns)
    };
    // per-site-choice: `arms.full` (a site candidate) takes it wherever it
    // divides — the cost model then ranks the full-tile plans.
    match (ns(&r), ns(&rule)) {
        (Some(a), Some(b)) if whole(&r) && acc(&r) > acc(&rule) && (a < b || arms.full) => r,
        _ => rule,
    }
}

/// The launch a kernel of realization `combine` is emitted with: under
/// `fill-wave` a split kernel's tile is the split grid's ([`launch_split`]).
fn launch_for(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    combine: Combine,
) -> Launch {
    resized(
        site,
        profile,
        contract,
        arms,
        launch(site, profile, contract, arms),
        combine,
    )
}

/// per-site-choice B2: a site's candidate table (`mapal_choice.h`'s
/// `struct mapal_cand`) and the one call that runs its pick. Declared only
/// beside a call, so a module without a table keeps its text.
pub(crate) const SITE_DECL: &str = "%mapal_cand = type { i64, i64, i64, i64, i32, i32, i32, i32, i32, i32 }\n\
     declare void @mapal_gpu_site(i64, i32, i32, i32, ptr, i32, ptr, i32, ptr)\n";

/// `mapal_gpu_site`'s flag: running the site again is value-safe.
const SITE_RERUN: u32 = 1;

/// **per-site-choice B1: one candidate of a site's table** — the launch it
/// realizes, the arms its kernels are emitted under, and the axes flipped
/// against candidate 0 (`rules` for candidate 0 itself).
#[derive(Clone, Debug)]
pub(crate) struct SiteCand {
    pub arms: crate::GpuArms,
    pub launch: Launch,
    pub axes: String,
}

/// per-site-choice B2: a site's table on its way to the launch point
/// (`emit_site_resident`, `emit_row_resident`), which emits it through
/// [`FnEmit::site_call`] in place of the single launch.
#[derive(Clone, Debug)]
pub(crate) struct SiteTable {
    /// Candidate 0's main kernel id.
    pub site: u64,
    pub flags: u32,
    /// One `%mapal_cand` constant per candidate.
    pub rows: Vec<String>,
    /// The largest workspace a candidate takes (0: none).
    pub ws: u64,
}

/// The launch a site realizes under `arms`: [`launch_in`]'s. A retained
/// row's one-row pass (`row`) cuts its fold only under `row-split` — a
/// workspace only on a free order (a fixed face replays one count per fold) —
/// and launches unsplit otherwise.
fn realized(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
) -> Launch {
    let l = launch_in(site, profile, contract, arms, order);
    let split = match l.combine {
        _ if !row => true,
        Combine::None => false,
        Combine::Workspace { .. } => l.rungs.row_split && order == Order::Free,
        _ => l.rungs.row_split,
    };
    match split {
        true => l,
        false => launch(site, profile, contract, arms),
    }
}

/// `arms` with axis `ax` applied against what launch `from` realized, and its
/// label (`+async`: forced on). per-site-choice D2: the alternative is the
/// opposite of `from`'s realized state ([`crate::profile::ChoiceAxis::realized`]),
/// forced — a forced arm bypasses the rule that declined it (`kmajor-a`'s
/// register rule, `async-depth`). `None` where a measurement arm pins the rung:
/// an A/B arm stays what it forces, in every candidate.
fn axis_arms(
    arms: crate::GpuArms,
    ax: &crate::profile::ChoiceAxis,
    from: &Launch,
) -> Option<(crate::GpuArms, String)> {
    let mut a = arms;
    if ax.full {
        if arms.off.get("plan-cost") || arms.off.get("full-tile") {
            return None;
        }
        *a.on.rung("plan-cost")? = true;
        *a.on.rung("full-tile")? = true;
        a.full = true;
        return Some((a, "+full-tile".into()));
    }
    if arms.on.get(ax.rung) || arms.off.get(ax.rung) {
        return None;
    }
    let on = !(ax.realized)(from);
    *(if on {
        a.on.rung(ax.rung)
    } else {
        a.off.rung(ax.rung)
    })? = true;
    Some((a, format!("{}{}", if on { '+' } else { '-' }, ax.rung)))
}

/// **per-site-choice B1: a site's candidates.** Candidate 0 is exactly the
/// rules' plan ([`realized`] under `arms`); then each
/// [`crate::profile::CHOICE_AXES`] entry flipped against its rungs, then the
/// pairs of two flips that each changed the launch (async × kmajor-a). A
/// candidate is kept when its launch — kernel and grid; a flipped rung shows
/// only through them — differs from every one before it, and on an order
/// other than free only with candidate 0's `Combine` (a fixed face replays one
/// partition count, an exact one has none). At most `arms.site_cands`.
/// `row`: the retained one-row pass of [`realized`].
pub(crate) fn site_cands(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
) -> Vec<SiteCand> {
    let at = |a: crate::GpuArms| realized(site, profile, contract, a, order, row);
    let base = at(arms);
    let mut out = vec![SiteCand {
        arms,
        launch: base,
        axes: "rules".into(),
    }];
    let cap = usize::try_from(arms.site_cands).unwrap_or(usize::MAX);
    if cap <= 1 {
        return out;
    }
    // `bank-plan` is the one axis rung the kernel text reads beyond the
    // launch: a k-major panel's row map ([`frag_layout`]).
    let map = |l: &Launch| l.rungs.bank_plan && l.tiled.is_some_and(|t| t.kmajor);
    let same = |a: &Launch, b: &Launch| {
        Launch {
            rungs: b.rungs,
            ..*a
        } == *b
            && map(a) == map(b)
    };
    // D2: no candidate where the axis's realized state did not change.
    let flip = |ax: &crate::profile::ChoiceAxis, fa: crate::GpuArms, from: &Launch| {
        let (a, axes) = axis_arms(fa, ax, from)?;
        let launch = at(a);
        (ax.full || (ax.realized)(&launch) != (ax.realized)(from)).then_some(SiteCand {
            arms: a,
            launch,
            axes,
        })
    };
    let keep = |out: &mut Vec<SiteCand>, c: SiteCand| {
        let order_ok = order == Order::Free || c.launch.combine == base.combine;
        if out.len() < cap && order_ok && !out.iter().any(|o| same(&o.launch, &c.launch)) {
            out.push(c);
        }
    };
    let singles: Vec<(&crate::profile::ChoiceAxis, SiteCand)> = crate::profile::CHOICE_AXES
        .iter()
        .filter_map(|ax| Some((ax, flip(ax, arms, &base)?)))
        .collect();
    for (_, c) in &singles {
        keep(&mut out, c.clone());
    }
    let moved: Vec<&(&crate::profile::ChoiceAxis, SiteCand)> = singles
        .iter()
        .filter(|(_, c)| !same(&c.launch, &base))
        .collect();
    for (i, (_, x)) in moved.iter().enumerate() {
        for (ay, _) in &moved[i + 1..] {
            if out.len() >= cap {
                return out;
            }
            if let Some(c) = flip(ay, x.arms, &x.launch) {
                let axes = format!("{}{}", x.axes, c.axes);
                keep(&mut out, SiteCand { axes, ..c });
            }
        }
    }
    out
}

/// per-site-choice B2: no buffer a launch writes is one it reads, under the
/// residency plan's `alias` (the slot an object shares): `RERUN`.
fn apart<T: Copy + PartialEq>(reads: &[T], writes: &[T], alias: impl Fn(T) -> T) -> bool {
    reads
        .iter()
        .all(|&r| writes.iter().all(|&w| alias(r) != alias(w)))
}

/// FNV-1a of `key`: the kernel ids' hash.
pub(crate) fn fnv(key: &str) -> u64 {
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// What a candidate kernel's id keys beyond its kind: the launch with only
/// its kernel rungs (the ones its plan line lists).
fn launch_key(l: &Launch) -> String {
    let mut r = crate::GpuRungs::default();
    for n in KERNEL_RUNGS.iter().filter(|n| l.rungs.get(n)) {
        *r.rung(n).expect("a kernel rung") = true;
    }
    // The ids predate the host-only `row-block`, `zero-fold`, `row-stage`, `horizontal` and `calls`
    // rungs (off here: no kernel rung): their fields are not part of a kernel's key, so
    // tuned picks and wisdom keep their ids.
    format!("{:?}", Launch { rungs: r, ..*l })
        .replace(", row_block: false", "")
        .replace(", zero_fold: false", "")
        .replace(", row_stage: false", "")
        .replace(", horizontal: false", "")
        .replace(", calls: false", "")
}

/// A candidate's kernel ids `(main, reduce)`, `reduce` 0 without a
/// workspace. Candidate 0 (`first`) keeps the ids the site had before
/// candidates; another keys its launch onto them.
fn cand_ids(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    epi: Option<&EpiKernel>,
    l: &Launch,
    first: bool,
) -> (u64, u64) {
    let split = |part| kernel_id_split(site, profile, contract, part);
    let (main, rid) = match l.combine {
        Combine::None => (kernel_id_with(site, profile, contract, epi), 0),
        Combine::Atomic { .. } => (split("atomic"), 0),
        Combine::StreamK { .. } => (split("streamk"), 0),
        Combine::Workspace { .. } => (split("splitk"), split("reduce")),
    };
    if first {
        return (main, rid);
    }
    let key = launch_key(l);
    let id = |x: u64| fnv(&format!("{x:016x}|{key}"));
    (id(main), if rid == 0 { 0 } else { id(rid) })
}

/// A candidate's launch recipe (B2) as the host's `%mapal_cand` constant,
/// and its workspace bytes.
fn recipe(site: &TileSite, l: &Launch, (id, rid): (u64, u64)) -> (String, u64) {
    let cells = site.rows * site.c;
    let bytes_o = cells * elem_bytes_of(&site.elem);
    let (cgx, cblock) = cell_grid(cells);
    let (zero, ws, gx, gy, rgx, rblock) = match l.combine {
        Combine::None => (0, 0, l.gx, l.gy, 0, 0),
        Combine::Atomic { s } => (bytes_o, 0, l.gx, l.gy * s, 0, 0),
        Combine::StreamK { ctas, .. } => (bytes_o, 0, ctas, 1, 0, 0),
        Combine::Workspace { s } => (0, s * bytes_o, l.gx, l.gy * s, cgx, cblock),
    };
    let text = format!(
        "%mapal_cand {{ i64 {}, i64 {}, i64 {zero}, i64 {ws}, i32 {gx}, i32 {gy}, i32 {}, i32 {}, \
         i32 {rgx}, i32 {rblock} }}",
        id as i64, rid as i64, l.block, l.smem
    );
    (text, ws)
}

/// A table's evidence line (B1): candidate 0's id, the count, every
/// candidate's main kernel and the axes it flipped.
fn site_line(ids: &[(u64, u64)], cands: &[SiteCand]) -> String {
    let ks: Vec<String> = ids.iter().map(|(i, _)| format!("{i:016x}")).collect();
    let axes: Vec<&str> = cands.iter().map(|c| c.axes.as_str()).collect();
    format!(
        "; mapal.site site={:016x} n={} cands={} axes={}\n",
        ids[0].0,
        ids.len(),
        ks.join(","),
        axes.join(",")
    )
}

/// The device half of a site's table: every candidate's kernels but candidate
/// 0's (the site's own, emitted as before), and the site's `; mapal.site`
/// line — only where the host's table names them (`live`).
#[allow(clippy::too_many_arguments)]
fn cand_kernels(
    kernels: &mut std::collections::BTreeMap<String, String>,
    lines: &mut std::collections::BTreeSet<String>,
    live: &std::collections::BTreeSet<String>,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
    epi: Option<&EpiKernel>,
) {
    let cands = site_cands(site, profile, contract, arms, order, row);
    let ids: Vec<(u64, u64)> = cands
        .iter()
        .enumerate()
        .map(|(i, c)| cand_ids(site, profile, contract, epi, &c.launch, i == 0))
        .collect();
    if !ids[1..]
        .iter()
        .any(|(i, _)| live.contains(&kernel_name(*i)))
    {
        return;
    }
    for (c, &(id, rid)) in cands.iter().zip(&ids).skip(1) {
        let name = kernel_name(id);
        let comb = c.launch.combine;
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(&name, site, profile, contract, c.arms, false, epi, comb)
        });
        if let Combine::Workspace { s } = comb {
            let red = kernel_name(rid);
            kernels.entry(red.clone()).or_insert_with(|| {
                format!(
                    "; mapal.plan kernel={red} kind=reduce rows={} c={} S={s}\n",
                    site.rows, site.c
                ) + &reduce_kernel_text(&red, site, profile, contract, c.arms, s)
            });
        }
    }
    lines.insert(site_line(&ids, &cands));
}

/// **plan-cost `linear` (§4a concat): the site table of `terms` sites of
/// `site`'s shape** — one launch, `z = term·S + part`, every partial added
/// into the zeroed sum — at the cheapest `S` by [`TargetProfile::site_ns`]:
/// `T×` the blocks and operand bytes, `S·T` adders per cell. A product goes
/// through `k/S` roundings in its partial and `S·T − 1` adds into the zeroed
/// sum: `S` is admitted only where `γ` of that depth stays within `bound`,
/// `linear_plan`'s source bound `γ_{k+T−1}` (D6; for `S > 1`, `S·T ≤ k`).
/// `None` where the scalar kernels do not run, the part has no plan rates, or
/// a launch's pointer table (≤ 16) is too short.
pub(crate) fn concat_plan(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    terms: u64,
    bound: f64,
) -> Option<(u64, PlanCost)> {
    let l = launch(site, profile, contract, arms);
    let gpu = profile.gpu()?;
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || site.radix.is_some();
    let atomic = elem_bytes_of(&site.elem) == 4 || gpu.caps.red_f64;
    if l.mma.is_some() || l.wgmma.is_some() || window || !atomic || l.raster != 1 {
        return None;
    }
    if !l.rungs.linear || 2 * terms + 1 > 16 || profile.plan_rates().is_none() {
        return None;
    }
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let mut best: Option<(u64, PlanCost)> = None;
    for s in (1..=panels.min(64)).filter(|&s| panels.is_multiple_of(s)) {
        let c = Combine::Atomic { s };
        let r = resized(site, profile, contract, arms, l, c);
        if mapal_ir::bound_coef(site.k / s + s * terms - 1, site.tf32) > bound
            || !(site.k / s).is_multiple_of(r.tiled.map_or(1, |g| g.bk))
        {
            continue;
        }
        let mut shape = plan_shape(site, &r, c, gpu);
        shape.blocks *= terms;
        shape.k *= terms;
        shape.s *= terms;
        let cost = profile.site_ns(&shape)?;
        if best.is_none_or(|(_, b)| cost.ns < b.ns) {
            best = Some((s, cost));
        }
    }
    best
}

/// The device text of [`concat_plan`]'s kernel: the atomic split kernel of
/// `S` partitions over a table of `terms` sites.
fn concat_kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    terms: u64,
    s: u64,
) -> String {
    let c = Combine::Atomic { s };
    let mut k = kernel(site, profile, contract, arms, false, None, c);
    k.terms = terms;
    let l = launch_for(site, profile, contract, arms, c);
    k.out = plan_line(name, site, profile, &l, k.idx, c, false, false);
    k.out.pop();
    k.out += &format!(" terms={terms}\n");
    match l.tiled {
        Some(g) => k.tiled(name, g),
        None => k.naive(name, l.block),
    }
    k.out
}

/// **The chooser (ADR-0040 D4, gap map M1/G2/G6)**: how a site's `k` fold is
/// realized, from IR facts (`TilePlan`'s independent cells, the site's
/// extents and step) and the part's (`Launch::wave`, `sm`, L2).
///
/// - `sequential`: in order.
/// - `fixed`: the deterministic workspace split (`splitk`), when the grid
///   leaves resident slots empty.
/// - `free`: stream-K when its makespan — `⌈blocks·P / wave⌉` panel steps per
///   block plus one fixup — beats the best split's
///   `⌈blocks·S / wave⌉ · P/S` (wave quantization, G6); else the atomic split
///   (`splitk-atomic`); else the workspace split (deterministic, legal under
///   `free`); else in order. Each step down is the rung above's OFF arm.
///
/// Only the scalar kernels have a fold to cut; `raster` and a split are
/// exclusive by derivation (a grid is grouped only above one wave, split only
/// below), stream-K walks tiles in the raster order instead. A forced
/// `--splitk=S` replaces the derived count where `S` divides the panels.
fn combine(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    order: Order,
    arms: crate::GpuArms,
) -> Combine {
    // A window's taps (`window`) are walked in order: its `k` is not cut.
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || site.radix.is_some();
    if order == Order::Sequential || l.mma.is_some() || l.wgmma.is_some() || window {
        return Combine::None;
    }
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let rungs = l.rungs;
    let e = elem_bytes_of(&site.elem);
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let blocks = l.gx * l.gy;
    let wave = l.wave(gpu);
    // site-geometry `fill-wave`: a grid with fewer warps than the fill rule's
    // target is cut into the fewest partitions that reach it, inside the wave
    // the part holds at the tile's own registers and threads.
    let warps = l.block.div_ceil(gpu.warp);
    let under = rungs.fill_wave && blocks * warps < gpu.fill_warps();
    let cap = match l.tiled {
        Some(g) if under => gpu.sms * profile.tile_capacity(&g, &site.elem),
        _ => wave,
    };
    // `--splitk=` (arms) over the part's table (`Geom::split`).
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    let count = |workspace| match forced {
        s if s > 0 && panels.is_multiple_of(s) => s,
        _ if under => {
            profile.split_fill(blocks, warps, cap, panels, site.rows * site.c, e, workspace)
        }
        _ => profile.split_count(blocks, wave, panels, site.rows * site.c, e, workspace),
    };
    let s_ws = if rungs.splitk && l.raster == 1 {
        count(true)
    } else {
        1
    };
    if order == Order::Fixed {
        return match s_ws {
            1 => Combine::None,
            s => Combine::Workspace { s },
        };
    }
    // `red.global.add.f32` exists from sm_20, `.f64` from sm_60.
    let atomic = e == 4 || gpu.caps.red_f64;
    let s_at = count(false);
    // `fill-wave` counts the split, not the ladder: stream-K is weighed
    // against the S63 count, as before.
    let s_cmp = match under && forced == 0 {
        true => profile.split_count(blocks, wave, panels, site.rows * site.c, e, false),
        false => s_at,
    };
    // ponytail: stream-K runs the register-staged tile (its panel loop takes
    // a runtime trip count); the cp.async pipeline's prologue is unrolled over
    // a constant one — add a predicated prologue when a streamed tile needs it.
    let streamk = rungs.streamk
        && atomic
        && l.tiled.is_some_and(|g| g.stages < 2)
        && panels >= 2
        && blocks * panels > wave;
    if streamk {
        let iters = (blocks * panels).div_ceil(wave);
        let split = (blocks * s_cmp).div_ceil(wave) * (panels / s_cmp);
        if iters + 1 < split {
            return Combine::StreamK {
                ctas: (blocks * panels).div_ceil(iters),
                iters,
            };
        }
    }
    if rungs.splitk_atomic && atomic && l.raster == 1 && s_at > 1 {
        return Combine::Atomic { s: s_at };
    }
    match s_ws {
        1 => Combine::None,
        s => Combine::Workspace { s },
    }
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

/// The site as its kernel ids key it: its `Debug` text, a plain site's
/// without the `radix` field it lacks (ids from before `radix-tile` stand).
fn site_key(site: &TileSite) -> String {
    format!("{site:?}")
        .replace(", radix: None", "")
        .replace(", tf32: false", "")
}

/// ADR-0041 D6: the reduced realization a site's products admit — `tf32` in
/// `precision_plan.set` of its product ([`TileSite::tf32`]) — keyed from the
/// program, never a flag (D7).
pub(crate) fn site_reduced(site: &TileSite) -> Option<Reduced> {
    site.tf32.then_some(Reduced::Tf32)
}

/// A stable id for a site's kernel: FNV-1a over everything the kernel text
/// depends on. The host passes it to the runtime, which looks the kernel up as
/// `mapal_tile_<id>` — so the two translation units agree without sharing state.
pub(crate) fn kernel_id(site: &TileSite, profile: &TargetProfile, contract: bool) -> u64 {
    kernel_id_with(site, profile, contract, None)
}

/// [`kernel_id`] of a site whose store evaluates `epi` (S54): the key also
/// names the group — its bodies, how each slot is fed, what is stored.
pub(crate) fn kernel_id_with(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    epi: Option<&EpiKernel>,
) -> u64 {
    let mut key = format!(
        "{}|{}|{contract}|{:?}",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
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
    col: bool,
) -> u64 {
    if !col {
        return kernel_id(site, profile, contract);
    }
    let key = format!(
        "{}|{}|{contract}|{:?}|col",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// S62/S63: the ids of a site's combine kernels — `part` is `splitk` (the
/// partial folds into the workspace), `reduce` (its reduce), `atomic` (the
/// partial folds added into the output) or `streamk` — the `col` precedent.
/// The ordered kernel keeps [`kernel_id`], so a function that does not split
/// the site (non-resident, an epilogue, a stream) launches that.
pub(crate) fn kernel_id_split(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    part: &str,
) -> u64 {
    let key = format!(
        "{}|{}|{contract}|{:?}|{part}",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
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
        && site.radix.is_none()
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
    /// plan-cost `linear` (§4a concat): the site table's terms and each
    /// term's partitions (`z = term·S + part`); 0 on an epilogue group.
    pub terms: u64,
    pub s: u64,
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
pub(crate) fn rowfold_id(site: &RowFoldSite, profile: &TargetProfile, lanes: u64) -> u64 {
    let l = if lanes > 1 {
        format!("|lanes={lanes}")
    } else {
        String::new()
    };
    format!("rowfold|{site:?}|{}{l}", profile.name)
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
pub(crate) fn rowfold_grid(site: &RowFoldSite, profile: &TargetProfile, lanes: u64) -> (u64, u64) {
    let g = profile.gpu().expect("a device fold has a GPU profile");
    if lanes > 1 {
        // `fold-lanes`: a warp a row, whole warps per block.
        return (
            site.rows.saturating_mul(lanes).div_ceil(NAIVE_THREADS),
            NAIVE_THREADS,
        );
    }
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
/// recurrence-precision item 2 `fold-lanes`: threads a row of a device row
/// fold folds over — a warp where the order is free (the default face,
/// ADR-0040), the step regroups (`+`, `×`, running max/min) over `f32`, the
/// fold is at least a warp long, plain (no window, no tap split) and its
/// rows leave the part's resident threads idle. 1 otherwise — and 1 over a
/// column-major operand (`colmajor`: its producing tile stores in the fold's
/// order, `rowfold-layout`): one thread a row already reads consecutive
/// addresses across a warp, where lanes along `k` would read `rows` apart
/// (the rtx-4090 measurement that once kept `fold-lanes` off there).
pub(crate) fn rowfold_lanes(
    site: &RowFoldSite,
    colmajor: bool,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> u64 {
    let regroups = matches!(
        site.op,
        RowFoldOp::Arith {
            op: Operation::Add | Operation::Mul,
            ..
        } | RowFoldOp::Select { .. }
    );
    match host_rungs(profile, contract, arms).fold_lanes
        && !colmajor
        && regroups
        && site.w == 1
        && site.a.ksplit.is_none()
    {
        true => warp_lanes(profile, order, &site.elem, site.k, site.rows),
        false => 1,
    }
}

/// The lane gate `fold-lanes` and `cell-lanes` share: a warp per unit (a row,
/// a cell) where the order is free (ADR-0040), the fold is f32 and at least a
/// warp long, and `units` warps fit the part's resident threads. 1 otherwise.
pub(crate) fn warp_lanes(
    profile: &TargetProfile,
    order: Order,
    elem: &Ty,
    k: u64,
    units: u64,
) -> u64 {
    let Some(g) = profile.gpu() else { return 1 };
    let fits = units.saturating_mul(g.warp) <= g.sms.saturating_mul(g.max_threads_per_sm);
    match order == Order::Free && g.warp == 32 && *elem == Ty::f32() && k >= g.warp && fits {
        true => g.warp,
        false => 1,
    }
}

/// A device row fold's kernel: its name, and its `; mapal.plan` line + text.
fn rowfold_entry(
    s: &RowFoldSite,
    colmajor: bool,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> (String, String) {
    let lanes = rowfold_lanes(s, colmajor, profile, contract, arms, order);
    let name = kernel_name(rowfold_id(s, profile, lanes));
    let (gx, block) = rowfold_grid(s, profile, lanes);
    let l = if lanes > 1 {
        format!(" lanes={lanes}")
    } else {
        String::new()
    };
    let text = format!(
        "; mapal.plan kernel={name} kind=rowfold rows={} k={} gx={gx} block={block}{l}\n",
        s.rows, s.k
    ) + &rowfold_kernel_text(&name, s, profile, lanes);
    (name, text)
}

/// `fold-lanes` (`lanes > 1`): a warp a row — lane `l` folds `k = l, l +
/// lanes, …` from the step's identity (a running max/min's from the seed),
/// a butterfly of shuffles combines the partials, `⊕` the seed once, lane 0
/// stores.
fn rowfold_kernel_text(name: &str, s: &RowFoldSite, profile: &TargetProfile, lanes: u64) -> String {
    let (_, threads) = rowfold_grid(s, profile, lanes);
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
    let (rows, k, ci, base) = (s.rows, s.k, s.a.ci, s.a.base);
    // generic-fusion `window`: the cell's row and lane `(t÷w, t%w)`, and the
    // fold's taps `(k÷div, k%div)`; the plain row fold's text otherwise.
    let row = if s.w > 1 {
        format!(
            "  %iq = udiv i64 %i, {w}\n  %il = urem i64 %i, {w}\n  %rq = mul i64 %iq, {ci}\n  \
             %rl = mul i64 %il, {cl}\n  %ri = add i64 %rq, %rl\n",
            w = s.w,
            cl = s.a.clane
        )
    } else {
        format!("  %ri = mul i64 %i, {ci}\n")
    };
    let taps = match s.a.ksplit {
        Some(ks) => format!(
            "  %kq = udiv i64 %k, {d}\n  %kr = urem i64 %k, {d}\n  %kq1 = mul i64 %kq, {cq}\n  \
             %kr1 = mul i64 %kr, {cr}\n  %kk = add i64 %kq1, %kr1\n",
            d = ks.div,
            cq = ks.cq,
            cr = ks.cr
        ),
        None => format!("  %kk = mul i64 %k, {}\n", s.a.ck),
    };
    // `fold-lanes`: the row is `g ÷ lanes`, the lane `g % lanes`; each lane
    // walks `k ≡ lane` from the identity, then the warp combines.
    let (gid, k0, acc0, kstep, fin, store) = if lanes > 1 {
        let comb = |a: &str, b: &str, r: &str| match s.op {
            RowFoldOp::Arith { op, .. } => {
                format!(
                    "  {r} = {} {t} {a}, {b}\n",
                    if op == Operation::Mul { "fmul" } else { "fadd" }
                )
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
                let (x, y) = if acc_lhs { (a, b) } else { (b, a) };
                let (p, q) = if acc_if_true { (a, b) } else { (b, a) };
                format!(
                    "  {r}c = fcmp {pred} {t} {x}, {y}\n  {r} = select i1 {r}c, {t} {p}, {t} {q}\n"
                )
            }
        };
        let (identity, seeded) = match s.op {
            RowFoldOp::Arith {
                op: Operation::Mul, ..
            } => ("1.0".to_string(), true),
            RowFoldOp::Arith { .. } => ("-0.0".to_string(), true),
            RowFoldOp::Select { .. } => (seed.clone(), false),
        };
        let mut tree = String::new();
        let mut cur = "%accn".to_string();
        let mut off = lanes / 2;
        while off >= 1 {
            tree.push_str(&format!(
                "  %sh{off} = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {cur}, i32 {off}, i32 31)\n"
            ));
            let r = format!("%cb{off}");
            tree.push_str(&comb(&cur, &format!("%sh{off}"), &r));
            cur = r;
            off /= 2;
        }
        if seeded {
            let first = matches!(
                s.op,
                RowFoldOp::Arith {
                    acc_first: true,
                    ..
                }
            );
            let (a, b) = if first {
                (seed.as_str(), cur.as_str())
            } else {
                (cur.as_str(), seed.as_str())
            };
            tree.push_str(&comb(a, b, "%tot"));
            cur = "%tot".into();
        }
        (
            format!(
                "  %g = add i64 %row0, %t64\n  %i = udiv i64 %g, {lanes}\n  %ln = urem i64 %g, {lanes}\n"
            ),
            "%ln".to_string(),
            identity,
            lanes,
            "uge",
            format!(
                "{tree}  %l0 = icmp eq i64 %ln, 0\n  br i1 %l0, label %st, label %done\nst:\n  \
                 %q = getelementptr inbounds {t}, ptr addrspace(1) %o, i64 %i\n  \
                 store {t} {cur}, ptr addrspace(1) %q, align {e}\n"
            ),
        )
    } else {
        (
            "  %i = add i64 %row0, %t64\n".to_string(),
            "0".to_string(),
            seed.clone(),
            1,
            "eq",
            format!(
                "  %q = getelementptr inbounds {t}, ptr addrspace(1) %o, i64 %i\n  \
                 store {t} %accn, ptr addrspace(1) %q, align {e}\n"
            ),
        )
    };
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
         {gid}\
         \x20 %in = icmp ult i64 %i, {rows}\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         {row}\
         \x20 %r0 = add i64 %ri, {base}\n\
         \x20 br label %kloop\n\
         kloop:\n\
         \x20 %k = phi i64 [{k0}, %body], [%kn, %kloop]\n\
         \x20 %acc = phi {t} [{acc0}, %body], [%accn, %kloop]\n\
         {taps}\
         \x20 %x = add i64 %r0, %kk\n\
         \x20 %p = getelementptr inbounds {t}, ptr addrspace(1) %a, i64 %x\n\
         \x20 %v = load {t}, ptr addrspace(1) %p, align {e}\n\
         {step}\
         \x20 %kn = add i64 %k, {kstep}\n\
         \x20 %fin = icmp {fin} i64 %kn, {k}\n\
         \x20 br i1 %fin, label %store, label %kloop\n\
         store:\n\
         {store}\
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

/// The `Fold` a tile site's map `m` evaluates per cell: the one fold of its
/// body (`CategoryIr::tile_site` admits exactly one).
fn site_fold(ir: &CategoryIr, m: MorphismId) -> (FuncId, MorphismId) {
    let Operation::Map { body, .. } = ir.morphism(m).expect("site morphism").op else {
        unreachable!("a tile site is a map");
    };
    let fold = ir
        .func(body)
        .expect("map body")
        .morphisms
        .iter()
        .copied()
        .find(|&x| matches!(ir.morphism(x).map(|g| g.op), Some(Operation::Fold { .. })))
        .expect("a tile site's body folds");
    (body, fold)
}

/// `cell-law`'s device variant of body `name` (`FnEmit::cell_vals`).
pub(crate) fn cell_variant(name: &str) -> String {
    format!("{name}_cell")
}

/// `cell-lanes`' device variant of body `name` (`FnEmit::lanes`).
pub(crate) fn lanes_variant(name: &str) -> String {
    format!("{name}_lanes")
}

/// `out[i] = body(caps…, x[i])`, one thread per cell (`devmap`): the host's
/// body text runs on the device, so the cell is the host's bit for bit. The
/// parameters are the output, then the mapped array when it is loaded (`elem`
/// is its element type; `None` forms the element from the index, an `iota`),
/// then each captured array the cell loads — generic pointers to the body, as
/// the host passes them by reference — or uniform it reads. A capture
/// [`CapArg::Law`] is not a parameter: `cell-law` evaluates `law(i)` and
/// `callee` (the body's variant) takes the value after its input. A
/// [`CapArg::Uniform`] arrives as the 8-byte argument `mapal_gpu_launchv`
/// passes, its low bits the host's value.
pub(crate) fn cell_kernel_text(
    id: u64,
    callee: &str,
    n: u64,
    out: &str,
    arg: &str,
    elem: Option<&str>,
    caps: &[CapArg],
    g: CellGeo,
) -> String {
    let mut params = vec!["ptr addrspace(1) %o".to_string()];
    if elem.is_some() {
        params.push("ptr addrspace(1) %x".into());
    }
    params.extend(caps.iter().enumerate().filter_map(|(k, c)| match c {
        CapArg::Ptr => Some(format!("ptr addrspace(1) %c{k}")),
        CapArg::Uniform(_) => Some(format!("i64 %c{k}")),
        CapArg::Dev(_) => Some(format!("ptr addrspace(1) %c{k}")),
        CapArg::Law(..) | CapArg::Trip(..) => None,
    }));
    if g.range.is_some() {
        params.push("i64 %rs".into());
        params.push("i64 %rn".into());
    }
    if g.zero > 0 {
        params.extend(super::zero::zero_params(g.zero));
    }
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
            match c {
                CapArg::Law(..) => continue, // the variant never reads the pointer
                CapArg::Ptr => body.push_str(&format!(
                    "  %g{k} = addrspacecast ptr addrspace(1) %c{k} to ptr\n  %a{k} = insertvalue {arg} {prev}, ptr %g{k}, {k}\n"
                )),
                CapArg::Uniform(t) => body.push_str(&format!(
                    "{}  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n",
                    uniform_from_i64(&format!("%g{k}"), &format!("%c{k}"), t)
                )),
                // `scalar-fold`: the number lives in device memory.
                CapArg::Dev(t) => body.push_str(&format!(
                    "  %g{k} = load {t}, ptr addrspace(1) %c{k}\n  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n"
                )),
                // `trip-batch`: the counter is the cell's own trip, `i ÷ d − b`.
                CapArg::Trip(t, d, b) => body.push_str(&format!(
                    "  %q{k} = sdiv i64 %i, {d}\n  %w{k} = sub i64 %q{k}, {b}\n{}  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n",
                    uniform_from_i64(&format!("%g{k}"), &format!("%w{k}"), t)
                )),
            }
            prev = format!("%a{k}");
        }
        body.push_str(&format!(
            "  %a = insertvalue {arg} {prev}, {e_ty} %e, {}\n",
            caps.len()
        ));
        format!("{arg} %a")
    };
    for (k, c) in caps.iter().enumerate() {
        if let CapArg::Law(law, t) = c {
            body.push_str(&format!("  %v{k} = call {t} @{law}(i32 %e)\n"));
            call_arg.push_str(&format!(", {t} %v{k}"));
        }
    }
    // `range-cell`: thread `k` is cell `rs + step·k` of the interval's `rn`
    // cells. `cell-lanes`: a cell is `lanes` consecutive threads (the body's
    // lane variant splits its fold over them), lane 0 stores.
    let (k, bound) = match g.range {
        None => ("%i", n.to_string()),
        Some(_) => ("%k", "%rn".to_string()),
    };
    let (gid, at) = match (g.lanes, g.range) {
        (1, None) => ("%i".to_string(), String::new()),
        (1, Some(step)) => (
            "%k".to_string(),
            format!("  %ks = mul i64 %k, {step}\n  %i = add i64 %rs, %ks\n"),
        ),
        (l, None) => ("%g".to_string(), format!("  %i = udiv i64 %g, {l}\n")),
        (l, Some(step)) => (
            "%g".to_string(),
            format!(
                "  %k = udiv i64 %g, {l}\n  %ks = mul i64 %k, {step}\n  %i = add i64 %rs, %ks\n"
            ),
        ),
    };
    // Lanes: the bound test is on the cell, after the division.
    let (head, guard) = if g.lanes == 1 {
        (
            String::new(),
            format!("  %in = icmp ult i64 {k}, {bound}\n"),
        )
    } else {
        (at.clone(), format!("  %in = icmp ult i64 {k}, {bound}\n"))
    };
    let at = if g.lanes == 1 { at } else { String::new() };
    let store = if g.lanes == 1 {
        String::new()
    } else {
        format!(
            "  %ln = urem i64 %g, {l}\n  %l0 = icmp eq i64 %ln, 0\n  br i1 %l0, label %st, label %done\nst:\n",
            l = g.lanes
        )
    };
    format!(
        "define ptx_kernel void @{name}({params}) \"nvvm.maxntid\"=\"{NAIVE_THREADS}\" {{\n\
         entry:\n\
         \x20 %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %t to i64\n\
         \x20 %b64 = zext i32 %b to i64\n\
         \x20 %base = mul i64 %b64, {NAIVE_THREADS}\n\
         \x20 {gid} = add i64 %base, %t64\n\
         {head}{guard}\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         {at}{body}\
         \x20 %v = call {out} @{callee}({call_arg})\n\
         {store}\
         \x20 %p = getelementptr inbounds {out}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {out} %v, ptr addrspace(1) %p\n\
         \x20 br label %done\n\
         done:\n\
         {tail}\
         \x20 ret void\n\
         }}\n",
        name = kernel_name(id),
        params = params.join(", "),
        tail = super::zero::zero_tail(g.zero, NAIVE_THREADS),
    )
}

/// One capture slot of a cell kernel ([`cell_kernel_text`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CapArg {
    /// A captured array, by device pointer.
    Ptr,
    /// `cell-law`: `(law fn, element type)`, evaluated in the cell.
    Law(String, String),
    /// A number capture of LLVM type `ty`, by value.
    Uniform(String),
    /// `scalar-fold`: a number capture of type `ty` in a one-element device
    /// buffer (a whole fold's result), by pointer.
    Dev(String),
    /// `trip-batch`: the loop counter, of type `ty`, as the cell's own trip
    /// `i ÷ d − b` (a batched row write: row `t + b` is trip `t`'s).
    Trip(String, u64, i64),
}

/// A cell kernel's geometry ([`cell_kernel_text`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CellGeo {
    /// `range-cell`: the interval's step; the kernel takes its start and
    /// count by value (`%rs`, `%rn`).
    pub range: Option<u64>,
    /// `cell-lanes`: threads a cell (1: one thread a cell).
    pub lanes: u64,
    /// kernel-merging `zero-fold`: the later sites' outputs this kernel
    /// zeroes ([`super::zero::zero_tail`]).
    pub zero: usize,
}

impl CellGeo {
    #[cfg(test)]
    pub(crate) const ONE: CellGeo = CellGeo {
        range: None,
        lanes: 1,
        zero: 0,
    };

    /// `(blocks, threads)` for `cells` cells.
    pub(crate) fn grid(self, cells: u64) -> (u64, u64) {
        cell_grid(cells.saturating_mul(self.lanes.max(1)))
    }
}

/// `dst = src` narrowed from the 8-byte launch argument to `ty` (`i64`
/// passes as is; a float is its bit pattern).
pub(crate) fn uniform_from_i64(dst: &str, src: &str, ty: &str) -> String {
    match ty {
        "i64" => format!("  {dst} = bitcast i64 {src} to i64\n"),
        "double" => format!("  {dst} = bitcast i64 {src} to double\n"),
        "float" => {
            format!("  {dst}.b = trunc i64 {src} to i32\n  {dst} = bitcast i32 {dst}.b to float\n")
        }
        t => format!("  {dst} = trunc i64 {src} to {t}\n"),
    }
}

/// The host's 8-byte launch argument for a uniform `val` of LLVM type `ty`
/// (the inverse of [`uniform_from_i64`]): the instructions, then the operand.
pub(crate) fn uniform_to_i64(dst: &str, val: &str, ty: &str) -> String {
    match ty {
        "i64" => format!("{dst} = bitcast i64 {val} to i64"),
        "double" => format!("{dst} = bitcast double {val} to i64"),
        "float" => {
            format!("{dst}.b = bitcast float {val} to i32\n  {dst} = zext i32 {dst}.b to i64")
        }
        t => format!("{dst} = zext {t} {val} to i64"),
    }
}

/// A `devmap` cell for the device unit: the functions its kernel calls
/// (`(name, emitted text)`), then the kernel text.
pub(crate) type CellKernel = (Vec<(String, String)>, String);

/// The kernels the host's launch sites name, by symbol: every
/// `mapal_gpu_{tile,launch,launch_l2,launchv}(id, …)` loads `mapal_tile_<id>`,
/// every `mapal_gpu_gen(id, …)` loads `mapal_gen_<id>` (`mapal_gpu.c`). A
/// device kernel outside this set has no use and is not emitted.
pub(crate) fn launched_kernels(host: &str) -> std::collections::BTreeSet<String> {
    let mut live = std::collections::BTreeSet::new();
    for (i, pat) in host.match_indices("call void @mapal_gpu_") {
        let rest = &host[i + pat.len()..];
        let Some((entry, args)) = rest.split_once("(i64 ") else {
            continue;
        };
        let prefix = match entry {
            "tile" | "launch" | "launch_l2" | "launchv" | "launch_coop" => "mapal_tile",
            "gen" => "mapal_gen",
            _ => continue,
        };
        let lit: String = args
            .chars()
            .take_while(|c| *c == '-' || c.is_ascii_digit())
            .collect();
        let id = lit
            .parse::<u64>()
            .or_else(|_| lit.parse::<i64>().map(|v| v as u64));
        if let Ok(id) = id {
            live.insert(format!("{prefix}_{id:016x}"));
        }
    }
    // per-site-choice: every kernel a site's table names
    // (`%mapal_cand { i64 <id>, i64 <reduce id>, … }`).
    for (i, pat) in host.match_indices("%mapal_cand { i64 ") {
        let mut f = host[i + pat.len()..].split(", ");
        let main = f.next();
        let red = f.next().and_then(|r| r.strip_prefix("i64 "));
        for lit in [main, red].into_iter().flatten() {
            if let Ok(id) = lit.parse::<i64>()
                && id != 0
            {
                live.insert(kernel_name(id as u64));
            }
        }
    }
    live
}

/// The symbol a kernel's text defines (its first `void @…(`).
fn kernel_symbol(text: &str) -> &str {
    text.split_once("void @")
        .and_then(|(_, r)| r.split_once('('))
        .map_or("?", |(n, _)| n)
}

/// The device translation unit: one kernel per GPU tile site in `ir`, in name
/// order, kept only where a host launch site names it (`live`, see
/// [`launched_kernels`]). `None` when no site offloads (every CPU profile, and
/// a GPU profile on a program with no tile site).
pub(crate) fn device_module(
    ir: &CategoryIr,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    gens: &[GenKernel],
    epis: &[EpiSite],
    epi_bodies: &[(String, String)],
    cells: &[CellKernel],
    folds: &[RowFoldSite],
    units: &[super::flow::UnitKernel],
    live: &std::collections::BTreeSet<String>,
    hz: &[super::horizontal::HzSpec],
) -> Option<String> {
    // kernel-merging `horizontal`: a merged kernel's members are composed
    // from their kernels' text (kept standalone only where `live` names them).
    let launched = live;
    let with_members: std::collections::BTreeSet<String> = live
        .iter()
        .cloned()
        .chain(super::horizontal::hz_members(hz))
        .collect();
    let live = &with_members;
    let mut kernels = std::collections::BTreeMap::new();
    // per-site-choice: each candidate table's `; mapal.site` line.
    let mut site_lines = std::collections::BTreeSet::new();
    // The module's reduced face, for the evidence header: any kernel realized on the tensor cores (ADR-0041 D6).
    let mut reduced = None;
    for (f, _) in ir.funcs() {
        let rungs = host_rungs(profile, contract, arms);
        let plan = tile_plan(ir, f, profile, contract, arms);
        let dev = placement(ir, f, &plan, profile, rungs.window);
        // A retained site's restricted pass (S53) launches its one-row kernel.
        let retained = super::retain::retained_sites(ir, f, profile, rungs);
        // S54 `rowfold`: the device folds, and the sites that store for them.
        let folds = super::rowfold::rowfold_sites(ir, f, profile, contract, arms);
        let col = |m: MorphismId| folds.values().any(|r| r.colmajor == Some(m));
        // A grouped site (S54 `epilogue`) launches its epilogue kernel instead.
        let sites = plan
            .sites
            .iter()
            .filter(|&(m, _)| dev.contains_key(m))
            .filter(|&(m, _)| !epis.iter().any(|e| e.f == f && e.m == m))
            .map(|(m, s)| (s, col(m), true));
        // A retained site's row kernel exists where the site is placed (S62).
        let rows = retained
            .iter()
            .filter(|&(y, _)| ir.in_edges(y).first().is_some_and(|&m| dev.contains_key(m)))
            .map(|(_, r)| (&r.row, false, false));
        for (site, col, whole) in sites.chain(rows) {
            if gpu_site(site, profile, rungs.window) {
                let l = launch(site, profile, contract, arms);
                if l.mma.is_some() || l.wgmma.is_some() {
                    reduced = site_reduced(site);
                }
                let name = kernel_name(kernel_id_col(site, profile, contract, col));
                kernels.entry(name.clone()).or_insert_with(|| {
                    kernel_text(
                        &name,
                        site,
                        profile,
                        contract,
                        arms,
                        col,
                        None,
                        Combine::None,
                    )
                });
                // per-site-choice: the candidates of a table (no column-major
                // store: its consumer reads the one layout).
                if !col {
                    cand_kernels(
                        &mut kernels,
                        &mut site_lines,
                        live,
                        site,
                        profile,
                        contract,
                        arms,
                        order,
                        !whole,
                        None,
                    );
                }
                // S62/S63 (ADR-0040 D4): a whole site whose fold is cut gets
                // its combine kernels too; the host picks per function
                // (resident, no epilogue, one stream), so the ordered one stays.
                let c = launch_in(site, profile, contract, arms, order).combine;
                // site-geometry `row-split`: a retained row's atomic split or
                // stream-K kernel (the host zeroes the row; no workspace).
                let row = l.rungs.row_split
                    && match c {
                        Combine::Workspace { .. } => order == Order::Free,
                        _ => true,
                    };
                if !(whole || row) || col || c == Combine::None {
                    continue;
                }
                let id = |part| kernel_name(kernel_id_split(site, profile, contract, part));
                let part = match c {
                    Combine::Workspace { .. } => "splitk",
                    Combine::Atomic { .. } => "atomic",
                    _ => "streamk",
                };
                let k = id(part);
                kernels.entry(k.clone()).or_insert_with(|| {
                    kernel_text(&k, site, profile, contract, arms, false, None, c)
                });
                if let Combine::Workspace { s } = c {
                    let red = id("reduce");
                    kernels.entry(red.clone()).or_insert_with(|| {
                        format!(
                            "; mapal.plan kernel={red} kind=reduce rows={} c={} S={s}\n",
                            site.rows, site.c
                        ) + &reduce_kernel_text(&red, site, profile, contract, arms, s)
                    });
                }
            }
        }
        // generic-fusion: a fold a unit holds launches inside the unit.
        let fused = |m: MorphismId| units.iter().any(|u| u.f == f && u.members.contains(&m));
        for (_, r) in folds.iter().filter(|&(m, _)| !fused(m)) {
            let (name, text) = rowfold_entry(
                &r.site,
                r.colmajor.is_some(),
                profile,
                contract,
                arms,
                order,
            );
            kernels.entry(name).or_insert(text);
        }
    }
    // S65: the folds a function placed over a device output no tile site
    // wrote (the host's plan names them).
    for f in folds {
        let (name, text) = rowfold_entry(f, false, profile, contract, arms, order);
        kernels.entry(name).or_insert(text);
    }
    for e in epis {
        if e.terms > 1 {
            let part = format!("concat{}x{}", e.terms, e.s);
            let name = kernel_name(kernel_id_split(&e.site, profile, contract, &part));
            kernels.entry(name.clone()).or_insert_with(|| {
                concat_kernel_text(&name, &e.site, profile, contract, arms, e.terms, e.s)
            });
            continue;
        }
        let l = launch(&e.site, profile, contract, arms);
        if l.mma.is_some() || l.wgmma.is_some() {
            reduced = site_reduced(&e.site);
        }
        let id = kernel_id_with(&e.site, profile, contract, Some(&e.k));
        let name = kernel_name(id);
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(
                &name,
                &e.site,
                profile,
                contract,
                arms,
                false,
                Some(&e.k),
                Combine::None,
            )
        });
        // An epilogue group's kernel is ordered: its candidates are too.
        cand_kernels(
            &mut kernels,
            &mut site_lines,
            live,
            &e.site,
            profile,
            contract,
            arms,
            Order::Sequential,
            false,
            Some(&e.k),
        );
    }
    if kernels.is_empty() && gens.is_empty() && cells.is_empty() && units.is_empty() {
        return None;
    }
    // Launch-site use counts: a kernel no launch site names is dead.
    kernels.retain(|name: &String, _| live.contains(name));
    let mut out = String::from("; mapal-backend-llvm device module\n");
    // S64 evidence: the module's face (each kernel's facts: `; mapal.plan`).
    out.push_str(&format!(
        "; mapal.device target={} fma={} order={order:?} reduced={:?}\n",
        profile.name,
        u8::from(contract),
        reduced
    ));
    for l in &site_lines {
        out.push_str(l);
    }
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
        if !live.contains(&format!("mapal_gen_{id:016x}")) {
            continue;
        }
        assert!(
            !text.contains("@mapal_"),
            "devgen body {callee} names a host runtime symbol"
        );
        if bodies.insert(callee.clone()) {
            out.push('\n');
            out.push_str(text);
        }
        out.push('\n');
        out.push_str(&format!(
            "; mapal.plan kernel=mapal_gen_{id:016x} kind=gen\n"
        ));
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
        let name = kernel_symbol(kernel);
        if !live.contains(name) {
            continue;
        }
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
        // `emit_cell`'s facts line, then the kernel; a uniform is an `i64`
        // parameter of its `define`.
        let (facts, kernel) = kernel
            .strip_prefix("; mapal.cell ")
            .and_then(|r| r.split_once('\n'))
            .unwrap_or(("", kernel));
        let uniform = kernel
            .lines()
            .find(|l| l.starts_with("define"))
            .map_or(0, |d| d.matches("i64 %c").count());
        out.push_str("\n; cell\n");
        out.push_str(&format!(
            "; mapal.plan kernel={name} kind=cell uniform={uniform} {facts}\n"
        ));
        out.push_str(kernel);
    }
    // generic-fusion: each unit's bodies once (shared with the others'), then
    // its kernel.
    for u in units {
        if !live.contains(kernel_symbol(&u.text)) {
            continue;
        }
        for (callee, text) in &u.fns {
            assert!(
                !text.contains("@mapal_"),
                "unit body {callee} names a host runtime symbol"
            );
            if bodies.insert(callee.clone()) {
                out.push('\n');
                out.push_str(text);
            }
        }
        out.push('\n');
        out.push_str(&u.text);
    }
    out.push('\n');
    out.push_str(&md);
    let mut out = super::horizontal::compose(out, hz, launched);
    // `pdl`: the runtime launches every kernel behind a kernel as a dependent
    // launch, so every kernel waits for its producer's grid before it reads.
    if pdl(profile, contract, arms) {
        out = pdl_wait_every_kernel(&out);
    }
    // A `wgmma` kernel's swizzle atoms start on 1024-byte boundaries.
    if out.contains("wgmma.mma_async") {
        out = out.replacen("[0 x i8], align 16", "[0 x i8], align 1024", 1);
    }
    Some(out)
}

/// hardware-properties `pdl`: `griddepcontrol.wait` at the entry of every
/// kernel of the module that does not already open with it (the tile kernel
/// does, after its index setup). The runtime launches any kernel that follows
/// a kernel on its stream as a programmatic dependent launch
/// (`mapal_gpu.c`), so a kernel without the wait (a cell, a row fold, a unit,
/// a merged grid, a reduce) would read its producer's output before that
/// grid completes. The wait is a no-op on a plain or cooperative launch.
pub(crate) fn pdl_wait_every_kernel(module: &str) -> String {
    const WAIT: &str = "  call void asm sideeffect \"griddepcontrol.wait;\", \"~{memory}\"()";
    let mut out = String::with_capacity(module.len() + 4096);
    let mut lines = module.lines().peekable();
    while let Some(l) = lines.next() {
        out.push_str(l);
        out.push('\n');
        if !l.starts_with("define ptx_kernel ") {
            continue;
        }
        // The kernel's text up to its closing brace: does it wait already?
        let body: Vec<&str> = std::iter::from_fn(|| {
            let n = lines.next()?;
            Some(n)
        })
        .take_while(|n| *n != "}")
        .collect();
        let waits = body.iter().any(|b| b.contains("griddepcontrol.wait"));
        let mut inserted = waits;
        for b in &body {
            out.push_str(b);
            out.push('\n');
            if !inserted && b.ends_with(':') && !b.starts_with(' ') {
                out.push_str(WAIT);
                out.push('\n');
                inserted = true;
            }
        }
        out.push_str("}\n");
    }
    if !module.ends_with('\n') {
        out.pop();
    }
    out
}

fn kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    col: bool,
    epi: Option<&EpiKernel>,
    combine: Combine,
) -> String {
    let mut k = kernel(site, profile, contract, arms, col, epi, combine);
    let l = launch_for(site, profile, contract, arms, combine);
    debug_assert!(
        !col || col_store(&l, site),
        "a column-major store this kernel has"
    );
    debug_assert!(combine == Combine::None || (!col && epi.is_none()));
    k.out = plan_line(name, site, profile, &l, k.idx, combine, col, epi.is_some());
    // plan-cost: the site's pick on every kernel's line (D3); its candidates
    // on every row-major one (the harness keys them by `site=`).
    let pick = plan_pick(site, profile, contract, arms, !col);
    if !pick.is_empty() {
        k.out.pop();
        k.out += &pick;
        k.out.push('\n');
    }
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
/// partition order fixed, straight-line — ADR-0040 D3's combine, so the bits
/// are a function of the site and the profile. No fast-math flag: LLVM may
/// not re-bracket what the text fixes. `%b` is unused.
fn reduce_kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    s: u64,
) -> String {
    let mut k = kernel(
        site,
        profile,
        contract,
        arms,
        false,
        None,
        Combine::Workspace { s },
    );
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
/// rungs and the fold's realization (`combine`; `None` for the ordered kernel).
#[allow(clippy::too_many_arguments)]
fn kernel<'s>(
    site: &'s TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    col: bool,
    epi: Option<&'s EpiKernel>,
    combine: Combine,
) -> Kernel<'s> {
    let l = launch_for(site, profile, contract, arms, combine);
    let rungs = l.rungs;
    let splitk = combine.splits();
    // `idx32` is an FMA / one-thread-per-cell rung: the tensor-core kernels
    // (`mma.sync`, `wgmma`) form their indices in `i64`.
    let tensor = l.mma.is_some() || l.wgmma.is_some();
    let idx = if rungs.idx32 && !tensor && max_index(site, &l, combine) < 1 << 31 {
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
        atomic: matches!(combine, Combine::Atomic { .. } | Combine::StreamK { .. }),
        terms: 1,
        streamk: match combine {
            Combine::StreamK { iters, .. } => Some(iters),
            _ => None,
        },
        p0: None,
        head: "entry",
    }
}

/// The largest index a kernel forms (S53 rung `idx32`): each operand's last
/// element — `base + ci·(rows−1) + clane·(c−1) + ck·(k−1)`, site extents —
/// every cell a launched thread can name (the one-thread-per-cell grid rounds
/// up to whole blocks), and `k`. Every intermediate of the index arithmetic is
/// a partial sum of one of these, so below 2³¹ the kernel indexes in `i32`
/// with `nuw nsw`; pointers stay 64-bit. A workspace split kernel (S62)
/// names its workspace cell `z·rows·c + cell`, `z < S`; a partition's reads
/// stay inside the operand's own last element. A stream-K kernel (S63) counts
/// its iteration space up to `ctas·iters` (`tiles × panels`, rounded up).
fn max_index(site: &TileSite, l: &Launch, combine: Combine) -> u64 {
    let rx = site.radix.as_deref();
    let last = |r: &mapal_ir::TileRead| {
        let ra = rx.and_then(|x| {
            if std::ptr::eq(r, &site.a) {
                x.a.as_ref()
            } else {
                x.b.as_ref()
            }
        });
        if let Some(ra) = ra {
            return ra.max();
        }
        let taps = r.ksplit.map_or(r.ck * (site.k - 1), |ks| {
            ks.cq * (site.k / ks.div - 1) + ks.cr * (ks.div - 1)
        });
        r.base + r.ci * (site.rows - 1) + r.clane * (site.c - 1) + taps
    };
    let slabs = match combine {
        Combine::Workspace { s } => s,
        _ => 1,
    };
    let walk = match combine {
        Combine::StreamK { ctas, iters } => ctas * iters,
        _ => 0,
    };
    let cells = (site.rows * site.c).max(l.gx * l.gy * l.block) * slabs;
    last(&site.a)
        .max(last(&site.b))
        .max(cells)
        .max(site.k)
        .max(walk)
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
fn lane_maps(s: &TileSite, g: &GpuTile, kmajor: bool) -> [(LaneMap, u64); 2] {
    let e = elem_bytes_of(&s.elem);
    let lanes = GPU_VEC_BYTES / e;
    let (bm, bn, bk, threads) = (g.bm(), g.bn(), g.bk, g.threads());
    let (kv, nv) = (bk / lanes, bn / lanes);
    let streamed = g.stages >= 2;
    // tile-efficiency T2: a k-major `a` panel is copied as it lies — `kv`
    // 16-byte copies along each row, async or register-staged alike.
    let a = if kmajor {
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bm * kv / threads)
    } else if streamed {
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

/// tile-efficiency T4: the LSU warp-instructions one thread of tile `g` issues
/// per panel trip, as [`Kernel::tiled`] emits them: `bk` steps of `a` and `b`
/// fragment loads, each copy's global load or `cp.async`, and the
/// register-staged copies' shared stores (`a` scattered `lanes` ways when it
/// is staged transposed). per-site-choice A3: each shared-memory class counts
/// its bank excess ([`panel_excess`]) times — a replayed wavefront is one more
/// LSU slot; a conflict-free kernel's count is its instruction count.
pub(crate) fn lsu_per_panel(s: &TileSite, g: &GpuTile, rungs: &crate::GpuRungs, col: bool) -> u64 {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    let kmaj = g.kmajor && !col;
    let [(_, a), (_, b)] = lane_maps(s, g, kmaj);
    // per-site-choice D1: the excess weights are `bank-plan`'s; off (and the
    // plan line's `lsu_issue`), every class counts its instructions once.
    let [xa, xb, sa, sb] = match rungs.bank_plan {
        true => panel_excess(s, g, rungs, col),
        false => [1; 4],
    };
    let lds_a = match kmaj {
        true => g.bk * g.tm / lanes,
        false => g.bk * g.tm / g.tm.min(lanes),
    };
    let lds_b = g.bk * g.tn / g.tn.min(lanes);
    let copies = match g.stages >= 2 {
        true => a * sa + b * sb,
        false if vector_a(s) && !kmaj => a + b + a * lanes * sa + b * sb,
        false => a + b + a * sa + b * sb,
    };
    lds_a * xa + lds_b * xb + copies
}

/// [`Kernel::tiled`]'s thread grid: `(trow, tcol)` of thread `tid`, in IR
/// ([`Kernel`]) or as numbers ([`Num`], the bank model). Warp shape (S52
/// rung `warp`): a warp's 32 lanes as `warp / wx` rows × `wx` columns of
/// threads, `wx` = the lanes one shared-memory wavefront serves with vector
/// loads (128 B / 16 B = 8) — 4 + 8 distinct fragments per `k` step instead
/// of 2 + 16. Off arm / a block too narrow for it: row-major threads. S54
/// `rowfold-layout` (`col`): the row-major rule mirrored, so one column-major
/// store still writes whole 128-byte lines (`col_store`).
fn thread_coords<A: Arith>(
    ar: &mut A,
    tid: &A::V,
    g: &GpuTile,
    warp_rung: bool,
    col: bool,
    warp: u64,
) -> (A::V, A::V) {
    let w = GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES;
    let (wx, wy) = if col { (warp / w, w) } else { (w, warp / w) };
    if warp_rung && g.tx % wx == 0 && g.ty % wy == 0 && g.tx > wx {
        let wp = ar.imm("udiv", tid, warp);
        let lane = ar.imm("urem", tid, warp);
        let warps_x = g.tx / wx;
        let wr = ar.imm("udiv", &wp, warps_x);
        let wc = ar.imm("urem", &wp, warps_x);
        let (lr, lc) = if col {
            (ar.imm("urem", &lane, wy), ar.imm("udiv", &lane, wy))
        } else {
            (ar.imm("udiv", &lane, wx), ar.imm("urem", &lane, wx))
        };
        let r0 = ar.imm("mul", &wr, wy);
        let c0 = ar.imm("mul", &wc, wx);
        return (ar.sum(&r0, &lr), ar.sum(&c0, &lc));
    }
    if col {
        (ar.imm("urem", tid, g.ty), ar.imm("udiv", tid, g.ty))
    } else {
        (ar.imm("udiv", tid, g.tx), ar.imm("urem", tid, g.tx))
    }
}

/// `sa`: elements between two rows of the `a` panel. Transposed (one row a
/// `k`): `bm + pad_a`, padded on `cp.async` and under `pad`
/// ([`TargetProfile::gpu_tile_sync_a`]). tile-efficiency T2: k-major
/// (`As[m][k]`): `bk + pad_a` — unless the kernel stores column-major (its
/// quarter-warps read eight panel rows), which keeps the transposed panel.
fn panel_sa(g: &GpuTile, kmaj: bool) -> u64 {
    match (g.kmajor, kmaj) {
        (true, true) => g.bk + g.pad_a,
        (true, false) => g.bm(),
        _ => g.sa(),
    }
}

/// Where a thread's register-tile fragments sit in the panels: its `i`-th
/// `a` row is `trow·a_step + (i / aw)·a_chunk + i % aw`, its `b` vectors
/// `tcol·b_step + v·b_chunk`.
struct Frag {
    aw: u64,
    a_step: u64,
    a_chunk: u64,
    b_step: u64,
    b_chunk: u64,
    narrow: bool,
}

/// Bank geometry: a thread's register tile is `tm/lanes × tn/lanes` vector
/// chunks spaced one thread-row of chunks apart (`ty·lanes` rows, `tx·lanes`
/// columns), not one contiguous `tm × tn` block. Consecutive threads then read
/// consecutive 16-byte chunks of a panel row, which is what a shared-memory
/// wavefront serves without a conflict — and their output stores coalesce.
/// (Off arm: one contiguous `tm × tn` block per thread, chunk = lanes.)
/// S62 `fill-rows`: a register tile `tm` rows tall, `tm < lanes`, reads its
/// `a` fragment as one `tm`-wide chunk (`aw`); `tn` keeps whole vectors (the
/// output's vector store). A tile narrower than a vector in `tn`
/// (`cell-fill`'s 1 × 1) has no chunks to space (`narrow`). A mixed-radix
/// output stores a vector where its unit lane digit runs `lanes`-aligned and
/// every other digit moves by whole vectors.
///
/// per-site-choice A1: a k-major thread owns rows `trow + i·ty`. One 16-byte
/// fragment load of a warp then reads its `warp / wx` consecutive rows, which
/// a row pad spreads over the banks ([`kmajor_pad`]); rows `aw` apart are
/// ≡ 0 mod 4 rows of a 16-byte-aligned stride and no pad separates them.
/// Each cell's `k` order is unchanged; the output rows follow the same map.
fn frag_layout(s: &TileSite, g: &GpuTile, kmaj: bool, interleave: bool) -> Frag {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    let (tm, tn) = (g.tm, g.tn);
    let aw = tm.min(lanes);
    let out_vec = s
        .radix
        .as_deref()
        .and_then(|x| x.out.as_ref())
        .is_none_or(|o| {
            let v = |c: u64| c.is_multiple_of(lanes);
            o.lane
                .first()
                .is_some_and(|d| d.stride == 1 && d.coef == 1 && v(d.radix))
                && o.i.iter().chain(&o.lane[1..]).all(|d| v(d.coef))
                && v(o.base)
        });
    let narrow = tn < lanes || !out_vec;
    let (a_step, b_step, a_chunk, b_chunk) = if interleave && !narrow {
        (aw, lanes, g.ty * aw, g.tx * lanes)
    } else {
        (tm, tn, aw, lanes)
    };
    let (aw, a_step, a_chunk) = match kmaj {
        true => (1, 1, g.ty),
        false => (aw, a_step, a_chunk),
    };
    Frag {
        aw,
        a_step,
        a_chunk,
        b_step,
        b_chunk,
        narrow,
    }
}

/// **per-site-choice A3: the bank excess of tile `g`'s four shared-memory
/// access classes** — `[a fragment read, b fragment read, a copy write,
/// b copy write]` (copy writes: register-staged stores or `cp.async`) — as
/// [`Kernel::tiled`] emits them ([`thread_coords`], [`frag_layout`],
/// [`lane_maps`]): the worst [`bank_degree`] ÷ ideal (rounded up) over every
/// warp and every instruction of one panel step; 1 = conflict-free. Steps
/// and stages move every lane by one constant: the same degree.
pub(crate) fn panel_excess(
    s: &TileSite,
    g: &GpuTile,
    rungs: &crate::GpuRungs,
    col: bool,
) -> [u64; 4] {
    let e = elem_bytes_of(&s.elem);
    let (lanes, wpe) = (GPU_VEC_BYTES / e, e.div_ceil(4));
    let threads = g.threads();
    let kmaj = g.kmajor && !col;
    let sa = panel_sa(g, kmaj);
    let f = frag_layout(s, g, kmaj && rungs.bank_plan, rungs.interleave);
    let at: Vec<(u64, u64)> = (0..threads)
        .map(|t| thread_coords(&mut Num, &t, g, rungs.warp, col, WARP))
        .collect();
    // The worst excess of `n` instructions, `acc(t, p)` thread `t`'s
    // (element, elements) in instruction `p`.
    let worst = |n: u64, acc: &dyn Fn(u64, u64) -> (u64, u64)| {
        let mut x = 1;
        for p in 0..n {
            for w0 in (0..threads).step_by(WARP as usize) {
                let words = (w0..threads.min(w0 + WARP)).map(|t| {
                    let (a, len) = acc(t, p);
                    (a * wpe, len * wpe)
                });
                let (d, i) = bank_degree(words);
                x = x.max(d.div_ceil(i));
            }
        }
        x
    };
    let lds_a = match kmaj {
        true => worst(g.tm, &|t, i| {
            let r = at[t as usize].0 * f.a_step + (i / f.aw) * f.a_chunk + i % f.aw;
            (r * sa, lanes)
        }),
        false => {
            let w = g.tm.min(lanes);
            worst(g.tm / w, &|t, v| {
                (at[t as usize].0 * f.a_step + v * f.a_chunk, w)
            })
        }
    };
    let w = g.tn.min(lanes);
    let lds_b = worst(g.tn / w, &|t, v| {
        (at[t as usize].1 * f.b_step + v * f.b_chunk, w)
    });
    let [(am, an), (bm, bn)] = lane_maps(s, g, kmaj);
    let rc = |m: LaneMap, t: u64, p: u64| m.coords(&mut Num, &(t + p * threads));
    let st_a = match (kmaj, g.stages < 2 && vector_a(s)) {
        (true, _) => worst(an, &|t, p| {
            let (r, kc) = rc(am, t, p);
            (r * sa + kc, lanes)
        }),
        // Transposed and register-staged: a vector along `k` scatters to
        // `lanes` panel rows, one store each.
        (false, true) => worst(an * lanes, &|t, p| {
            let (r, kc) = rc(am, t, p / lanes);
            ((kc + p % lanes) * sa + r, 1)
        }),
        (false, false) => worst(an, &|t, p| {
            let (r, kc) = rc(am, t, p);
            (kc * sa + r, 1)
        }),
    };
    let bw = if vector_b(s) { lanes } else { 1 };
    let st_b = worst(bn, &|t, p| {
        let (kr, nc) = rc(bm, t, p);
        (kr * g.bn() + nc, bw)
    });
    [lds_a, lds_b, st_a, st_b]
}

/// per-site-choice A2: the k-major `a` panel's row pad — the least, in
/// 16-byte steps under 32 words, that minimises the worst excess of its
/// fragment read and copy write ([`panel_excess`]); `b` does not move.
/// For f32 at the warp shape: `bk ≥ 16` ⇒ 4, `bk = 8` ⇒ 0.
fn kmajor_pad(s: &TileSite, g: &GpuTile, rungs: &crate::GpuRungs) -> u64 {
    let wpe = elem_bytes_of(&s.elem).div_ceil(4);
    let mut best = (u64::MAX, 0);
    for pad in (0..32 / wpe).step_by((GPU_VEC_BYTES / 4 / wpe).max(1) as usize) {
        let [a, _, st, _] = panel_excess(s, &GpuTile { pad_a: pad, ..*g }, rungs, false);
        best = best.min((a.max(st), pad));
        if best.0 == 1 {
            break;
        }
    }
    best.1
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
    /// S63 `splitk-atomic` / `streamk`: the partials add into `%o` itself
    /// (`atomicrmw fadd`), which the host zeroed; no workspace slab.
    atomic: bool,
    /// plan-cost `linear` (§4a concat): the site table's terms — term
    /// `ctaid.y ÷ (gy_out·S)` reads `%a<t>`, `%b<t>` and adds into `%o`; 1 =
    /// one site.
    terms: u64,
    /// S63 `streamk`: the units of the `tiles × panels` space one block walks.
    streamk: Option<u64>,
    /// S63 `streamk`: the current piece's first panel (the seed rides the
    /// piece that starts its tile, [`Kernel::split_seed`]).
    p0: Option<String>,
    /// The block the panel loop is entered from: `entry`, or `seg` in a
    /// stream-K kernel (one panel loop per piece).
    head: &'static str,
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
        if self.splitk > 1 || self.terms > 1 {
            let row = self.op("urem", &by, self.gy_out);
            let z = self.op("udiv", &by, self.gy_out);
            if self.splitk > 1 {
                // plan-cost `linear`: `z = term·S + part`.
                let part = match self.terms {
                    1 => z,
                    _ => self.op("urem", &z, self.splitk),
                };
                self.z = Some(part);
            }
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
    /// `Σ coef·((x ÷ stride) % radix)` — one axis of a mixed-radix address
    /// ([`mapal_ir::RadixAddr`]); constant divisors, which LLVM strength-reduces.
    fn digits(&mut self, ds: &[mapal_ir::AxisDigit], x: &str) -> String {
        let mut acc = "0".to_string();
        for d in ds {
            let q = if d.stride == 1 {
                x.to_string()
            } else {
                self.op("udiv", x, d.stride)
            };
            let r = self.op("urem", &q, d.radix);
            let v = self.op("mul", &r, d.coef);
            acc = self.op("add", &acc, &v);
        }
        acc
    }

    /// A mixed-radix address at `(i, lane)` without its `k` digits.
    fn radix_at(&mut self, r: &mapal_ir::RadixAddr, i: &str, lane: &str) -> String {
        let x = self.digits(&r.i, i);
        let y = self.digits(&r.lane, lane);
        let z = self.op("add", &x, &y);
        self.plus(z, r.base)
    }

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

    /// Output cell `v`'s index: workspace slab `z` in a workspace split
    /// kernel, `v` itself in an atomic one (its partials add into `%o`).
    fn slab(&mut self, v: String) -> String {
        if self.atomic {
            return v;
        }
        let cells = self.site.rows * self.site.c;
        self.at_z(v, cells)
    }

    /// Store `v` at `%o[oi]`: a plain store, or (S63 `splitk-atomic`,
    /// `streamk`) `atomicrmw fadd`, device scope, relaxed, result unused —
    /// `red.global.add` in the SASS. One instruction per element: a vector
    /// value is split lane by lane.
    fn store_o(&mut self, v: &[String], oi: &str) {
        let (t, e) = (self.t.clone(), elem_bytes_of(&self.site.elem));
        if !self.atomic {
            let n = v.len() as u64;
            let (ty, val) = if n == 1 {
                (t.clone(), v[0].clone())
            } else {
                let vt = format!("<{n} x {t}>");
                let mut vec = "poison".to_string();
                for (j, x) in v.iter().enumerate() {
                    let nv = self.tmp();
                    self.line(format!("{nv} = insertelement {vt} {vec}, {t} {x}, i64 {j}"));
                    vec = nv;
                }
                (vt, vec)
            };
            let p = self.gep(1, "%o", oi);
            self.line(format!(
                "store {ty} {val}, ptr addrspace(1) {p}, align {}",
                n * e
            ));
            return;
        }
        for (j, x) in v.iter().enumerate() {
            let i = self.plus(oi.to_string(), j as u64);
            let p = self.gep(1, "%o", &i);
            let r = self.tmp();
            self.line(format!(
                "{r} = atomicrmw fadd ptr addrspace(1) {p}, {t} {x} syncscope(\"device\") monotonic, align {e}"
            ));
        }
    }

    /// The fold's seed: partition 0 starts from the site's seed, every other
    /// partition from `0.0` (ADR-0040's partial folds), so the reduce adds
    /// the seed exactly once.
    fn split_seed(&mut self, seed: &str) -> String {
        let Some(z) = self.z.clone().or(self.p0.clone()) else {
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
        affine: bool,
        at: impl Fn(&mut Self, &str, &str) -> (String, String),
        off: impl Fn(u64, u64) -> (u64, u64),
    ) -> Vec<(String, u64, String, u64)> {
        let mut base: Option<(String, String)> = None;
        let mut out = Vec::new();
        for p in 0..n {
            let delta = (self.rungs.affine && affine).then(|| affine_delta(map, threads, p));
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
        // plan-cost `linear`: one `(a, b)` pair per term; `%a`, `%b` select
        // the block's term in the entry block.
        let ab = |t: u64| match t {
            0 if self.terms < 2 => ("%a".to_string(), "%b".to_string()),
            _ => (format!("%a{t}"), format!("%b{t}")),
        };
        let (a0, b0) = ab(0);
        let mut more = String::new();
        for t in 1..self.terms {
            let (a, b) = ab(t);
            more += &format!(
                ", ptr addrspace(1) readonly align {align} {a}, ptr addrspace(1) readonly align {align} {b}"
            );
        }
        let ro = if self.terms < 2 {
            "noalias readonly"
        } else {
            "readonly"
        };
        self.out.push_str(&format!(
            "define ptx_kernel void @{name}(\
             ptr addrspace(1) {ro} align {align} {a0}, \
             ptr addrspace(1) {ro} align {align} {b0}, \
             ptr addrspace(1) noalias align {align} %o{}{more}) \"nvvm.maxntid\"=\"{threads}\"{min_blocks} {{\n\
             entry:\n",
            self.epi_params(align)
        ));
        if self.terms > 1 {
            let y = self.tmp();
            self.line(format!("{y} = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()"));
            let term = self.tmp();
            self.line(format!(
                "{term} = udiv i32 {y}, {}",
                self.gy_out * self.splitk
            ));
            for (x, last) in [("a", "%a"), ("b", "%b")] {
                let mut prev = format!("%{x}0");
                for t in 1..self.terms {
                    let (hit, sel) = (self.tmp(), self.tmp());
                    let sel = if t + 1 == self.terms {
                        last.to_string()
                    } else {
                        sel
                    };
                    self.line(format!("{hit} = icmp eq i32 {term}, {t}"));
                    self.line(format!(
                        "{sel} = select i1 {hit}, ptr addrspace(1) %{x}{t}, ptr addrspace(1) {prev}"
                    ));
                    prev = sel;
                }
            }
        }
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
            let y = self.sreg("ctaid.y");
            self.z = Some(match self.terms {
                1 => y,
                _ => self.op("urem", &y, self.splitk),
            });
        }
        let cell0 = self.op("mul", &bx, threads);
        let cell = self.op("add", &cell0, &tid);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult {ix} {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let rx = s.radix.as_deref();
        let (ra, rb) = (rx.and_then(|x| x.a.as_ref()), rx.and_then(|x| x.b.as_ref()));
        let base = |k: &mut Self, r: &mapal_ir::TileRead| {
            if let Some(m) = if std::ptr::eq(r, &s.a) { ra } else { rb } {
                return k.radix_at(m, &i, &lane);
            }
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
        // generic-fusion `window`: a read affine in `(k÷div, k%div)` (conv
        // taps) addresses `cq·(k÷div) + cr·(k%div)`; constant divisors, a
        // constant trip count — LLVM folds the division away.
        let load = |k: &mut Self, arr: &str, base: &str, r: &mapal_ir::TileRead| {
            let m = if std::ptr::eq(r, &s.a) { ra } else { rb };
            let off = match r.ksplit {
                _ if m.is_some() => k.digits(&m.expect("radix").k, &kv),
                Some(ks) => {
                    let q = k.op("udiv", &kv, ks.div);
                    let m = k.op("urem", &kv, ks.div);
                    let x = k.op("mul", &q, ks.cq);
                    let y = k.op("mul", &m, ks.cr);
                    k.op("add", &x, &y)
                }
                None => k.op("mul", &kv, r.ck),
            };
            let idx = k.op("add", base, &off);
            let p = k.gep(1, arr, &idx);
            let v = k.tmp();
            k.line(format!("{v} = load {t}, ptr addrspace(1) {p}, align {e}"));
            v
        };
        let av = load(self, "%a", &abase, &s.a);
        let bv = load(self, "%b", &bbase, &s.b);
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
        } else if let Some(o) = rx.and_then(|x| x.out.as_ref()) {
            self.radix_at(o, &i, &lane)
        } else {
            cell
        };
        // A split kernel stores its partial into workspace slab `z`, or
        // (`splitk-atomic`) adds it into the output.
        let cell = self.slab(cell);
        if self.epi.is_some() {
            self.store_cells(std::slice::from_ref(&accn), &cell);
        } else {
            self.store_o(std::slice::from_ref(&accn), &cell);
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
        // site-geometry `tf32-radix`: a digit-affine operand's copies, one
        // element each — `(is_a, row/lane digits, shared word, its k)`; each
        // panel adds the `k` digits of `k0 + k` (`digits`).
        let rx = s.radix.as_deref();
        let (ra, rb, ro) = (
            rx.and_then(|x| x.a.as_ref()),
            rx.and_then(|x| x.b.as_ref()),
            rx.and_then(|x| x.out.as_ref()),
        );
        let mut radix: Vec<(bool, String, String, String)> = Vec::new();
        if let Some(m) = ra {
            for p in 0..bm * bk / threads {
                let li = self.op("add", &tid, p * threads);
                let r = self.op("udiv", &li, bk);
                let kc = self.op("urem", &li, bk);
                let grow = self.op("add", &row0, &r);
                let g = self.radix_at(m, &grow, "0");
                let sr = self.op("mul", &r, lda);
                radix.push((true, g, self.op("add", &sr, &kc), kc));
            }
        }
        if let Some(m) = rb {
            for p in 0..bk * bn / threads {
                let li = self.op("add", &tid, p * threads);
                let kr = self.op("udiv", &li, bn);
                let nc = self.op("urem", &li, bn);
                let gc = self.op("add", &col0, &nc);
                let g = self.radix_at(m, "0", &gc);
                let sk = self.op("mul", &kr, ldb);
                let sn = self.op("add", &sk, &nc);
                radix.push((false, g, self.op("add", &sn, bm * lda), kr));
            }
        }
        let a_chunks = if ra.is_some() {
            0
        } else {
            bm * bk / 4 / threads
        };
        for p in 0..a_chunks {
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
        let b_chunks = if rb.is_some() {
            0
        } else {
            bk * bn / 4 / threads
        };
        for p in 0..b_chunks {
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
            k.issue16(&copies, tile, stage, bk, stage_words);
            if radix.is_empty() {
                return;
            }
            let k0 = k.op("mul", tile, bk);
            let so = k.op("mul", stage, stage_words);
            for (is_a, g, sdst, kc) in &radix {
                let (arr, m) = if *is_a { ("%a", ra) } else { ("%b", rb) };
                let kv = k.op("add", &k0, kc);
                let kd = k.digits(&m.expect("a radix operand").k, &kv);
                let gi = k.op("add", g, &kd);
                let gp = k.gep(1, arr, &gi);
                let si = k.op("add", sdst, &so);
                let sp = k.gep(3, "@mapal_smem", &si);
                k.line(format!(
                    "call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                ));
            }
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
                    // `tf32-radix`: a digit-affine output, cell by cell.
                    if let Some(o) = ro {
                        for j in 0..2u64 {
                            let p = acc_slot(self, base + j as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            let cc = self.op("add", &ocol, ni as u64 * MMA_N + j);
                            let oi = self.radix_at(o, &r, &cc);
                            if self.epi.is_some() {
                                self.store_cells(&[x], &oi);
                            } else {
                                let op = self.gep(1, "%o", &oi);
                                self.line(format!(
                                    "store float {x}, ptr addrspace(1) {op}, align 4"
                                ));
                            }
                        }
                        continue;
                    }
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
        let accs = (tm * tn) as usize;
        let acc = self.tmp();
        // S63 `streamk`: the accumulators live across pieces (the entry block,
        // where they promote to registers); each piece re-seeds them.
        let seg = self.streamk.map(|iters| {
            self.line(format!(
                "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
            ));
            self.segment(iters, kext / bk, bk)
        });
        let (bx, by) = match &seg {
            Some(sg) => (sg.bx.clone(), sg.by.clone()),
            None => self.block_ids(),
        };
        let (kbeg, kend) = match &seg {
            Some(sg) => (sg.kbeg.clone(), sg.kend.clone()),
            None => ("0".to_string(), kext.to_string()),
        };
        // Warp shape (S52 rung `warp`): lay a warp's 32 lanes out as
        // `warp / wx` rows × `wx` columns of threads, `wx` = the lanes one
        // shared-memory wavefront serves with vector loads (128 B / 16 B = 8).
        // A warp then reads 4 + 8 distinct thread fragments per `k` step
        // instead of 2 + 16 — two-thirds of the bytes. Off arm / a block too
        // narrow for it: row-major threads.
        let (rungs, col, warp) = (self.rungs, self.col, self.profile_gpu.warp);
        let (trow, tcol) = thread_coords(self, &tid, &g, rungs.warp, col, warp);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);
        let smem = "@mapal_smem";
        let streamed = g.stages >= 2;
        let kmaj = g.kmajor && !self.col;
        let sa = panel_sa(&g, kmaj);
        let bs_off = g.a_elems();
        let stage_elems = g.stage_elems();

        if seg.is_none() {
            self.line(format!(
                "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
            ));
        }
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
        let [(a_map, a_loads), (b_map, b_loads)] = lane_maps(s, &g, kmaj);
        // recurrence-precision `radix-tile`: a mixed-radix operand's copy holds
        // its row (lane) digits at `k = 0` and its own `k` offset `kc`; each
        // panel adds the `k` digits of `k0 + kc` (not affine: no constant
        // delta between copies, no `k0·ck` stride).
        let rx = s.radix.as_deref();
        let (ra, rb, ro) = (
            rx.and_then(|x| x.a.as_ref()),
            rx.and_then(|x| x.b.as_ref()),
            rx.and_then(|x| x.out.as_ref()),
        );
        let a_kc = std::cell::RefCell::new(Vec::new());
        let b_kc = std::cell::RefCell::new(Vec::new());
        let a_src = self.copies(
            a_map,
            a_loads,
            threads,
            &tid,
            ra.is_none(),
            |k, r, kc| {
                let grow = k.op("add", &row0, r);
                if let Some(m) = ra {
                    a_kc.borrow_mut().push(kc.to_string());
                    let skc = k.op("mul", kc, sa);
                    return (k.radix_at(m, &grow, "0"), k.op("add", &skc, r));
                }
                let gr = k.op("mul", &grow, s.a.ci);
                let gk = k.op("mul", kc, s.a.ck);
                let g0 = k.op("add", &gr, &gk);
                let gsrc = k.op("add", &g0, s.a.base);
                let gsrc = k.at_z(gsrc, kext * s.a.ck);
                if kmaj {
                    let sr = k.op("mul", r, sa);
                    return (gsrc, k.op("add", &sr, kc));
                }
                let skc = k.op("mul", kc, sa);
                (gsrc, k.op("add", &skc, r))
            },
            |dr, dk| match kmaj {
                true => (dr * s.a.ci + dk * s.a.ck, dr * sa + dk),
                false => (dr * s.a.ci + dk * s.a.ck, dk * sa + dr),
            },
        );
        let b_src = self.copies(
            b_map,
            b_loads,
            threads,
            &tid,
            rb.is_none(),
            |k, kr, nc| {
                if let Some(m) = rb {
                    b_kc.borrow_mut().push(kr.to_string());
                    let gc = k.op("add", &col0, nc);
                    let sk = k.op("mul", kr, bn);
                    let sn = k.op("add", &sk, nc);
                    return (k.radix_at(m, "0", &gc), k.op("add", &sn, bs_off));
                }
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
        let Frag {
            aw,
            a_step,
            a_chunk,
            b_step,
            b_chunk,
            narrow,
        } = frag_layout(s, &g, kmaj && rungs.bank_plan, rungs.interleave);
        let a_frag0 = self.op("mul", &trow, a_step);
        let b_frag0 = self.op("mul", &tcol, b_step);

        // --- panel staging: global → (registers) → shared memory
        let a_ty = if a_vec { vt.clone() } else { t.clone() };
        let b_ty = if b_vec { vt.clone() } else { t.clone() };
        let a_al = if a_vec { GPU_VEC_BYTES } else { e };
        let b_al = if b_vec { GPU_VEC_BYTES } else { e };
        let (a_kc, b_kc) = (a_kc.into_inner(), b_kc.into_inner());
        // Copy `p`'s global index for panel `k0` (`kd` = `k0·ck` on a plain operand).
        let gidx = |k: &mut Self,
                    m: Option<&mapal_ir::RadixAddr>,
                    kcs: &[String],
                    p: usize,
                    gsrc: &str,
                    goff: u64,
                    k0: &str,
                    kd: &str| {
            if let Some(m) = m {
                let kk = k.op("add", k0, &kcs[p]);
                let kx = k.digits(&m.k, &kk);
                return k.op("add", gsrc, &kx);
            }
            let gi = k.op("add", gsrc, kd);
            k.plus(gi, goff)
        };
        let load_panels = |k: &mut Self, k0: &str| -> Vec<String> {
            let ka = k.op("mul", k0, s.a.ck);
            let kb = k.op("mul", k0, s.b.ck);
            let mut vals = Vec::new();
            for (p, (gsrc, goff, _, _)) in a_src.iter().enumerate() {
                let gi = gidx(k, ra, &a_kc, p, gsrc, *goff, k0, &ka);
                let gp = k.gep(1, "%a", &gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {a_ty}, ptr addrspace(1) {gp}, align {a_al}"
                ));
                vals.push(v);
            }
            for (p, (gsrc, goff, _, _)) in b_src.iter().enumerate() {
                let gi = gidx(k, rb, &b_kc, p, gsrc, *goff, k0, &kb);
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
                if kmaj {
                    let si = k.plus(sdst.clone(), *soff);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!(
                        "store {vt} {v}, ptr addrspace(3) {sp}, align {GPU_VEC_BYTES}"
                    ));
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
        let head = self.head;
        let (cbase, pre) = if streamed {
            debug_assert!(seg.is_none(), "stream-K runs the register-staged tile");
            // Issue panel `tile` into stage `stage`: one `cp.async` per unit.
            // `so`: the stage's first element (index type).
            let issue = |k: &mut Self, tile: &str, so: &str| {
                let k0 = k.op("mul", tile, bk);
                let ka = k.op("mul", &k0, s.a.ck);
                let kb = k.op("mul", &k0, s.b.ck);
                let so = so.to_string();
                for (p, (gsrc, goff, sdst, soff)) in a_src.iter().enumerate() {
                    let gi = gidx(k, ra, &a_kc, p, gsrc, *goff, &k0, &ka);
                    let gp = k.gep(1, "%a", &gi);
                    let si = k.op("add", sdst, &so);
                    let si = k.plus(si, *soff);
                    let sp = k.gep(3, smem, &si);
                    let (kind, n) = if kmaj {
                        ("cg", GPU_VEC_BYTES)
                    } else {
                        ("ca", e)
                    };
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.{kind}.shared.global.{n}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
                for (p, (gsrc, goff, sdst, soff)) in b_src.iter().enumerate() {
                    let gi = gidx(k, rb, &b_kc, p, gsrc, *goff, &k0, &kb);
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
                issue(self, &st.to_string(), &(st * stage_elems).to_string());
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            for _ in nk..g.stages - 1 {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (it, itn) = (self.tmp(), self.tmp());
            self.line(format!("{it} = phi {ix} [0, %entry], [{itn}, %ktl]"));
            // tile-efficiency T3: the stage the compute reads, a rotating i32
            // counter (no `urem`, no 64-bit stage arithmetic): `it mod stages`.
            let (cs, csn) = (self.tmp(), self.tmp());
            self.line(format!("{cs} = phi i32 [0, %entry], [{csn}, %ktl]"));
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
            // A stage's first element in the index type: `stage · stage_elems`
            // in i32 (every stage offset fits the 1 MiB-bounded shared memory).
            let stage_off = |k: &mut Self, st32: &str| {
                let x = k.tmp();
                k.line(format!("{x} = mul nuw i32 {st32}, {stage_elems}"));
                if ix == "i32" {
                    return x;
                }
                let y = k.tmp();
                k.line(format!("{y} = zext i32 {x} to {ix}"));
                y
            };
            self.out.push_str("issue:\n");
            // panel `it + stages − 1` lands in stage `(cs + stages − 1) mod stages`.
            let first = self.tmp();
            self.line(format!("{first} = icmp eq i32 {cs}, 0"));
            let back = self.tmp();
            self.line(format!("{back} = add i32 {cs}, -1"));
            let nst = self.tmp();
            self.line(format!(
                "{nst} = select i1 {first}, i32 {}, i32 {back}",
                g.stages - 1
            ));
            let nso = stage_off(self, &nst);
            issue(self, &nxt, &nso);
            self.line("br label %issued");
            self.out.push_str("issued:\n");
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cbase = stage_off(self, &cs);
            let up = self.tmp();
            self.line(format!("{up} = add nuw i32 {cs}, 1"));
            let wrap = self.tmp();
            self.line(format!("{wrap} = icmp eq i32 {up}, {}", g.stages));
            self.line(format!("{csn} = select i1 {wrap}, i32 0, i32 {up}"));
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
                load_panels(self, &kbeg)
            } else {
                Vec::new()
            };
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (k0, k0n) = (self.tmp(), self.tmp());
            self.line(format!("{k0} = phi {ix} [{kbeg}, %{head}], [{k0n}, %ktl]"));
            if self.rungs.prefetch {
                let held: Vec<String> = first.iter().map(|_| self.tmp()).collect();
                for (i, (h, f)) in held.iter().zip(&first).enumerate() {
                    self.line(format!(
                        "{h} = phi {} [{f}, %{head}], [@NEXT{i}@, %ktl]",
                        tys[i]
                    ));
                }
                store_panels(self, &held);
                self.line("call void @llvm.nvvm.barrier0()");
                let nk = self.op("add", &k0, bk);
                let more = self.tmp();
                self.line(format!("{more} = icmp ult {ix} {nk}, {kend}"));
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
        let frag = |k: &mut Self, ki: &str, row: u64, first: &str, n: u64, off: u64, chunk: u64| {
            let r = k.op("mul", ki, row);
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
        let fma = |k: &mut Self, av: &[String], bv: &[String]| {
            for i in 0..tm as usize {
                for j in 0..tn as usize {
                    let p = acc_slot(k, i * tn as usize + j);
                    let cur = k.tmp();
                    k.line(format!("{cur} = load {t}, ptr {p}, align {e}"));
                    let r = k.step(&cur, &av[i], &bv[j]);
                    k.line(format!("store {t} {r}, ptr {p}, align {e}"));
                }
            }
        };
        // tile-efficiency T2: a k-major trip takes `lanes` `k` steps — one
        // 128-bit `a` load along `k` per owned row, then per step its `b`
        // fragment and the `tm × tn` FMAs, `k` ascending as before.
        let steps = if kmaj { bk / lanes } else { bk };
        if kmaj {
            let k0 = self.op("mul", &kk, lanes);
            let k0 = self.op("add", &k0, &cbase);
            let mut rows = Vec::new();
            for i in 0..tm {
                let r = self.op("add", &a_frag0, (i / aw) * a_chunk + i % aw);
                let r = self.op("mul", &r, sa);
                let si = self.op("add", &r, &k0);
                let sp = self.gep(3, smem, &si);
                let x = self.tmp();
                self.line(format!(
                    "{x} = load {vt}, ptr addrspace(3) {sp}, align {GPU_VEC_BYTES}"
                ));
                let row: Vec<String> = (0..lanes)
                    .map(|j| {
                        let y = self.tmp();
                        self.line(format!("{y} = extractelement {vt} {x}, i64 {j}"));
                        y
                    })
                    .collect();
                rows.push(row);
            }
            for j in 0..lanes as usize {
                let kj = self.op("mul", &kk, lanes);
                let kj = self.op("add", &kj, j as u64);
                let bv = frag(self, &kj, bn, &b_frag0, tn, bs_off, b_chunk);
                let av: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                fma(self, &av, &bv);
            }
        } else {
            let av = frag(self, &kk, sa, &a_frag0, tm, 0, a_chunk);
            let bv = frag(self, &kk, bn, &b_frag0, tn, bs_off, b_chunk);
            fma(self, &av, &bv);
        }
        self.line(format!("{kkn} = add {ix} {kk}, 1"));
        let kkd = self.tmp();
        self.line(format!("{kkd} = icmp eq {ix} {kkn}, {steps}"));
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
            self.line(format!("{ktd} = icmp eq {ix} {k0n}, {kend}"));
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
                    let r = self.plus(r, (i / aw) * a_chunk + i % aw);
                    let cc = self.op("add", &col0, &b_frag0);
                    let cc = self.plus(cc, j);
                    let oi = if let Some(o) = ro {
                        self.radix_at(o, &r, &cc)
                    } else if self.col {
                        let cr = self.op("mul", &cc, s.rows);
                        self.op("add", &cr, &r)
                    } else {
                        let rr = self.op("mul", &r, s.c);
                        self.op("add", &rr, &cc)
                    };
                    // A split kernel stores its partials into workspace slab
                    // `z` (S62 integration: `cell-fill` × `splitk`), or adds
                    // them into the output (S63).
                    let oi = self.slab(oi);
                    if self.epi.is_some() {
                        self.store_cells(&[x], &oi);
                        continue;
                    }
                    self.store_o(&[x], &oi);
                }
            }
            self.finish(seg.as_ref());
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
            let rrow = ro.map(|_| r.clone());
            // A split kernel stores its partials into workspace slab `z`,
            // or adds them into the output (S63).
            let c0 = self.slab(c0);
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
                let mut xs = Vec::new();
                for j in 0..lanes {
                    let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                    xs.push(x);
                }
                let oi = match (ro, &rrow) {
                    (Some(o), Some(r)) => {
                        let cc = self.op("add", &col0, &b_frag0);
                        let cc = self.op("add", &cc, v * b_chunk);
                        self.radix_at(o, r, &cc)
                    }
                    _ => self.op("add", &c0, v * b_chunk),
                };
                self.store_o(&xs, &oi);
            }
        }
        self.finish(seg.as_ref());
    }

    /// **S63 `streamk`'s walk** (ADR-0040 D4): block `ctaid.x` owns units
    /// `[cta·iters, min((cta+1)·iters, tiles·P))` of the `tiles × panels`
    /// space, `P = panels`. Emits the entry's branch into the piece loop's
    /// head `seg`, and there the current piece: tile `it / P` (in the grid's
    /// raster order), panels `[p0, p1)`, `p1 = min(P, p0 + (end − it))`.
    /// Every block of the grid is resident (`ctas ≤ wave`), so the units a
    /// block walks are its whole share; a tile cut between blocks combines in
    /// the output ([`Kernel::store_o`]). The bounds sit inside the site's
    /// proved panels: the reads are the ordered kernel's.
    fn segment(&mut self, iters: u64, panels: u64, bk: u64) -> Segment {
        let ix = self.idx;
        let (gx, h) = self.raster;
        let total = gx * self.gy_out * panels;
        let cta = self.sreg("ctaid.x");
        let start = self.op("mul", &cta, iters);
        let end0 = self.op("add", &start, iters);
        let past = self.tmp();
        self.line(format!("{past} = icmp ult {ix} {end0}, {total}"));
        let end = self.tmp();
        self.line(format!(
            "{end} = select i1 {past}, {ix} {end0}, {ix} {total}"
        ));
        self.line("br label %seg");
        self.out.push_str("seg:\n");
        let it = self.tmp();
        self.line(format!(
            "{it} = phi {ix} [{start}, %entry], [@SEG_NEXT@, %segl]"
        ));
        let tile = self.op("udiv", &it, panels);
        let p0 = self.op("urem", &it, panels);
        let left = self.tmp();
        self.line(format!("{left} = sub {ix} {end}, {it}"));
        let p1a = self.op("add", &p0, &left);
        let fits = self.tmp();
        self.line(format!("{fits} = icmp ult {ix} {p1a}, {panels}"));
        let p1 = self.tmp();
        self.line(format!(
            "{p1} = select i1 {fits}, {ix} {p1a}, {ix} {panels}"
        ));
        let bx = self.op("urem", &tile, gx);
        let by = self.op("udiv", &tile, gx);
        let (bx, by) = if h < 2 {
            (bx, by)
        } else {
            self.line(format!("; raster: {h} block-rows per group"));
            let (c, r) = raster_block(self, &bx, &by, gx, h);
            (self.opaque(&c), self.opaque(&r))
        };
        let kbeg = self.op("mul", &p0, bk);
        let kend = self.op("mul", &p1, bk);
        let base = self.op("mul", &tile, panels);
        let next = self.op("add", &base, &p1);
        self.p0 = Some(p0);
        self.head = "seg";
        Segment {
            bx,
            by,
            kbeg,
            kend,
            next,
            end,
        }
    }

    /// The kernel's end: `ret`, or in a stream-K kernel the piece loop's
    /// latch — the next piece starts at the unit after this one's last.
    fn finish(&mut self, seg: Option<&Segment>) {
        let Some(sg) = seg else {
            self.out.push_str("  ret void\n}\n");
            return;
        };
        let ix = self.idx;
        self.line("br label %segl");
        self.out.push_str("segl:\n");
        let more = self.tmp();
        self.line(format!("{more} = icmp ult {ix} {}, {}", sg.next, sg.end));
        self.line(format!("br i1 {more}, label %seg, label %exit"));
        self.out.push_str("exit:\n  ret void\n}\n");
        self.out = self.out.replace("@SEG_NEXT@", &sg.next);
    }
}

/// One stream-K piece's values ([`Kernel::segment`]).
struct Segment {
    bx: String,
    by: String,
    kbeg: String,
    kend: String,
    /// The first unit after the piece, and the block's end.
    next: String,
    end: String,
}

impl<'a> FnEmit<'a> {
    /// per-site-choice B2: table `t` as a module constant and its one
    /// `mapal_gpu_site` call over the launch's pointers `ptrs` (the output
    /// third). The workspace slot, when a candidate takes one, is sized at the
    /// table's largest (`entry-arena` seeds it like any split workspace).
    pub(super) fn site_call(&mut self, t: SiteTable, ptrs: &[String]) {
        let global = format!("@{}.cands{}", self.fnames[self.f], self.fresh());
        self.cand_tables.push_str(&format!(
            "{global} = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
            t.rows.len(),
            t.rows.join(", ")
        ));
        let ws = match t.ws {
            0 => "null".to_string(),
            bytes => {
                let slot = format!("%d{}", self.fresh());
                self.allocas.push_str(&format!(
                    "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                ));
                self.split_slots.push(slot.clone());
                self.dallocs.push((slot.clone(), bytes));
                slot
            }
        };
        let n = ptrs.len();
        let arr = format!("%p{}", self.fresh());
        self.allocas
            .push_str(&format!("  {arr} = alloca [{n} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        self.line(format!(
            "call void @mapal_gpu_site(i64 {}, i32 {}, i32 0, i32 {}, ptr {global}, i32 {n}, \
             ptr {arr}, i32 2, ptr {ws})",
            t.site as i64,
            t.flags,
            t.rows.len()
        ));
    }

    /// per-site-choice: whether site `m`'s launch may run again (`RERUN`) —
    /// no buffer it writes (the output, an epilogue's stored results) is one
    /// it reads (an operand, an epilogue's complete operand). Every morphism
    /// writes a fresh object and the arena keeps a launch's objects apart
    /// (abi-map §3), so only a residency alias can join them.
    fn rerun_safe(
        &self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        m: MorphismId,
    ) -> bool {
        let alias = |o| self.resident.as_ref().map_or(o, |p| p.alias(o));
        let mut reads: Vec<ObjectId> = [site.a.slot, site.b.slot]
            .iter()
            .filter_map(|&k| self.pair_source(source, k))
            .collect();
        let mut writes = vec![target];
        if let Some(g) = self.groups.get(m) {
            reads.extend(g.caps());
            writes.extend(g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out));
        }
        apart(&reads, &writes, alias)
    }

    /// per-site-choice B1/B2: site `site`'s table, `one` its retained row's
    /// one-row site (`row`) or the whole site. `None` with fewer than two
    /// candidates.
    fn make_table(
        &self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        cands: &[SiteCand],
        one: &TileSite,
        epi: Option<&EpiKernel>,
    ) -> Option<SiteTable> {
        if cands.len() < 2 {
            return None;
        }
        let m = self.ir.in_edges(target)[0];
        let ids: Vec<(u64, u64)> = cands
            .iter()
            .enumerate()
            .map(|(i, c)| cand_ids(one, self.profile, self.contract, epi, &c.launch, i == 0))
            .collect();
        let (rows, ws): (Vec<String>, Vec<u64>) = cands
            .iter()
            .zip(&ids)
            .map(|(c, &id)| recipe(one, &c.launch, id))
            .unzip();
        Some(SiteTable {
            site: ids[0].0,
            flags: if self.rerun_safe(source, target, site, m) {
                SITE_RERUN
            } else {
                0
            },
            rows,
            ws: ws.into_iter().max().unwrap_or(0),
        })
    }

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
            let col = plan.rowfold.values().any(|r| r.colmajor == Some(m))
                || plan.units.values().flat_map(|u| &u.members).any(|mb| {
                    matches!(mb.kind, super::flow::UKind::Fold { colmajor: Some(c), .. } if c == m)
                });
            let streams = !plan.chain.is_empty();
            // plan-cost `linear`: the site table — the sum zeroed, one launch
            // (`emit_site_resident` passes every term's operands).
            if let Some(g) = self.groups.get(m).filter(|g| !g.terms.is_empty()) {
                let (t, s) = (g.terms.len() as u64 + 1, g.split);
                let id =
                    kernel_id_split(site, self.profile, self.contract, &format!("concat{t}x{s}"));
                let l = launch_for(
                    site,
                    self.profile,
                    self.contract,
                    self.gpu_arms,
                    Combine::Atomic { s },
                );
                let geom = format!(
                    "i32 {}, i32 {}, i32 {}, i32 {}",
                    l.gx,
                    l.gy * s * t,
                    l.block,
                    l.smem
                );
                self.emit_site_resident(
                    source,
                    target,
                    &[site.a.slot, site.b.slot],
                    Some((id as i64, geom)),
                    |_, _| unreachable!("a site table launches through its pointer table"),
                );
                return;
            }
            // S54 `epilogue`: the per-cell consumers this site's store evaluates
            // (none when a fold is its only reader: `col` and a group exclude).
            let epi = self.epi_kernel(m);
            let id = if col {
                kernel_id_col(site, self.profile, self.contract, true)
            } else {
                kernel_id_with(site, self.profile, self.contract, epi.as_ref())
            } as i64;
            let l = launch_in(site, self.profile, self.contract, self.gpu_arms, self.order);
            let cells = site.rows * site.c;
            let bytes_o = cells * elem_bytes_of(&site.elem);
            let combined = epi.is_none() && !col && !streams;
            // per-site-choice B1/B2: a site with candidates launches its
            // table — not a column-major store (its consumer reads the one
            // layout), a function on streams (its order is the events') or
            // the rules' ordered launch under an L2 window. An epilogue
            // group's kernel is ordered, so are its candidates.
            if !col && !streams {
                let order = match epi {
                    Some(_) => Order::Sequential,
                    None => self.order,
                };
                let arms = self.gpu_arms;
                let cands = site_cands(site, self.profile, self.contract, arms, order, false);
                let c0 = cands[0].launch;
                let windowed = epi.is_none()
                    && c0.combine == Combine::None
                    && host_rungs(self.profile, self.contract, arms).l2_window
                    && self.l2_window(m, site, &c0).is_some();
                let table = match windowed {
                    true => None,
                    false => self.make_table(source, target, site, &cands, site, epi.as_ref()),
                };
                if let Some(t) = table {
                    // ADR-0040 D3: every candidate of a fixed face has candidate
                    // 0's `Combine` — the count the interpreter replays.
                    if let (Order::Fixed, Combine::Workspace { s }) = (self.order, c0.combine) {
                        let (func, fold) = site_fold(self.ir, m);
                        self.orders.push(crate::FoldOrder { func, fold, s });
                    }
                    let geom = format!(
                        "i32 {}, i32 {}, i32 {}, i32 {}",
                        c0.gx, c0.gy, c0.block, c0.smem
                    );
                    self.site_table = Some(t);
                    self.emit_site_resident(
                        source,
                        target,
                        &[site.a.slot, site.b.slot],
                        Some((id, geom)),
                        |_, _| unreachable!("a table launches through mapal_gpu_site"),
                    );
                    debug_assert!(self.site_table.is_none(), "the table was launched");
                    return;
                }
            }
            // S63 (ADR-0040 D4) `splitk-atomic` / `streamk`: the output zeroed
            // on stream 0, then one launch whose partials add into it — no
            // workspace, no second kernel. Same exclusions as the workspace
            // split below.
            let atomic = match l.combine {
                Combine::Atomic { s } if combined => {
                    let geom = format!(
                        "i32 {}, i32 {}, i32 {}, i32 {}",
                        l.gx,
                        l.gy * s,
                        l.block,
                        l.smem
                    );
                    Some(("atomic", geom, format!(" ; splitk={s} atomic")))
                }
                Combine::StreamK { ctas, iters } if combined => {
                    let geom = format!("i32 {ctas}, i32 1, i32 {}, i32 {}", l.block, l.smem);
                    Some(("streamk", geom, format!(" ; streamk={ctas}x{iters}")))
                }
                _ => None,
            };
            if let Some((part, geom, tag)) = atomic {
                let aid = kernel_id_split(site, self.profile, self.contract, part) as i64;
                // kernel-merging `zero-fold`: an earlier kernel zeroed it.
                let zero = match plan.zeroed.contains_key(m) {
                    true => String::new(),
                    false => format!("call void @mapal_gpu_zero(i64 {{o}}, i64 {bytes_o})\n  "),
                };
                self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], None, |ops, o| {
                    format!(
                        "{}call void @mapal_gpu_launch(i64 {aid}, i64 {}, i64 {}, i64 {o}, {geom}){tag}",
                        zero.replace("{o}", o),
                        ops[0],
                        ops[1]
                    )
                });
                return;
            }
            // S62 `splitk` (ADR-0040): the split kernel folds `S` partitions
            // into a workspace (`gy·S` blocks), then the reduce kernel adds
            // them into the output in partition order, both on stream 0. The
            // workspace slot is made on the first launch and kept to function
            // exit. ponytail: an epilogue group, a column-major store and a
            // function on streams keep the ordered kernel (and an L2 window
            // is not given to the split launch); split them when a measured
            // program needs it.
            if let Combine::Workspace { s } = l.combine
                && combined
            {
                let id = |r| kernel_id_split(site, self.profile, self.contract, r);
                let (sid, rid) = (id("splitk") as i64, id("reduce") as i64);
                let (rgx, rblock) = cell_grid(cells);
                let bytes = s * bytes_o;
                // ADR-0040 D3: the order the interpreter replays (`fixed`).
                if self.order == Order::Fixed {
                    let (func, fold) = site_fold(self.ir, m);
                    self.orders.push(crate::FoldOrder { func, fold, s });
                }
                let slot = format!("%d{}", self.fresh());
                self.allocas.push_str(&format!(
                    "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                ));
                self.split_slots.push(slot.clone());
                self.dallocs.push((slot.clone(), bytes));
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
            // The ordered kernel (an epilogue group, a column-major store, a
            // function on streams, or no cut): the unsplit launch — `launch_in`
            // re-sizes the tile for a split (`fill-wave`) this kernel does not take.
            let l = if l.combine.splits() > 1 {
                launch(site, self.profile, self.contract, self.gpu_arms)
            } else {
                l
            };
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
        let id = kernel_id(site, self.profile, self.contract) as i64;
        let l = launch(site, self.profile, self.contract, self.gpu_arms);
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
            if !g.terms.is_empty() {
                sites.push(EpiSite {
                    f: self.f,
                    m,
                    site,
                    k: EpiKernel {
                        store_out: false,
                        caps: 0,
                        calls: Vec::new(),
                    },
                    terms: g.terms.len() as u64 + 1,
                    s: g.split,
                });
                continue;
            }
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
                terms: 0,
                s: 0,
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
        if let Some(plan) = &self.resident {
            // per-site-choice B1/B2: a row with candidates launches its table
            // (not on streams: their order is the events').
            if plan.chain.is_empty() {
                let (arms, order) = (self.gpu_arms, self.order);
                let cands = site_cands(one, self.profile, self.contract, arms, order, true);
                if let Some(t) = self.make_table(source, target, site, &cands, one, None) {
                    self.site_table = Some(t);
                    let done = self.emit_row_resident(source, target, site, row, |_, _, _| {
                        unreachable!("a table launches through mapal_gpu_site")
                    });
                    self.site_table = None;
                    if !done {
                        self.emit_tiled_map_gpu(source, target, site);
                    }
                    return;
                }
            }
            // site-geometry `row-split` (D1): the row's fold through the
            // chooser — zero the row, one launch whose partials add into it
            // (the whole site's `splitk-atomic` / `streamk`, one stream).
            // A workspace split (a part without the atomic one) on a free
            // order only: a fixed face replays one partition count per fold.
            let lc = launch_in(one, self.profile, self.contract, self.gpu_arms, self.order);
            let split = match lc.combine {
                _ if !lc.rungs.row_split || !plan.chain.is_empty() => None,
                Combine::Workspace { .. } if self.order != Order::Free => None,
                Combine::None => None,
                c => Some(c),
            };
            let bytes_o = one.c * elem_bytes_of(&one.elem);
            let part = match split {
                Some(Combine::Atomic { .. }) => "atomic",
                Some(Combine::StreamK { .. }) => "streamk",
                _ => "splitk",
            };
            let id = match split {
                Some(_) => kernel_id_split(one, self.profile, self.contract, part),
                None => kernel_id(one, self.profile, self.contract),
            } as i64;
            let ws = match split {
                Some(Combine::Workspace { s }) => {
                    let slot = format!("%d{}", self.fresh());
                    self.allocas.push_str(&format!(
                        "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                    ));
                    self.split_slots.push(slot.clone());
                    self.dallocs.push((slot.clone(), s * bytes_o));
                    let ws = self.tmp();
                    self.line(format!(
                        "{ws} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {})",
                        s * bytes_o
                    ));
                    ws
                }
                _ => String::new(),
            };
            let rid = kernel_id_split(one, self.profile, self.contract, "reduce") as i64;
            let l = launch(one, self.profile, self.contract, self.gpu_arms);
            let call = |a: &str, b: &str, o: &str| match split {
                Some(Combine::Atomic { s }) => format!(
                    "call void @mapal_gpu_zero(i64 {o}, i64 {bytes_o})\n  \
                     call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 {}, i32 {}, i32 {}) ; splitk={s} atomic",
                    lc.gx,
                    lc.gy * s,
                    lc.block,
                    lc.smem
                ),
                Some(Combine::StreamK { ctas, iters }) => format!(
                    "call void @mapal_gpu_zero(i64 {o}, i64 {bytes_o})\n  \
                     call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {ctas}, i32 1, i32 {}, i32 {}) ; streamk={ctas}x{iters}",
                    lc.block, lc.smem
                ),
                Some(Combine::Workspace { s }) => {
                    let (rgx, rblock) = cell_grid(one.c);
                    format!(
                        "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {ws}, \
                         i32 {}, i32 {}, i32 {}, i32 {}) ; splitk={s}\n  \
                         call void @mapal_gpu_launch(i64 {rid}, i64 {ws}, i64 0, i64 {o}, i32 {rgx}, i32 1, i32 {rblock}, i32 0)",
                        lc.gx,
                        lc.gy * s,
                        lc.block,
                        lc.smem
                    )
                }
                _ => format!(
                    "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 {}, i32 {}, i32 {})",
                    l.gx, l.gy, l.block, l.smem
                ),
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
        let id = kernel_id(one, self.profile, self.contract) as i64;
        let l = launch(one, self.profile, self.contract, self.gpu_arms);
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

    /// The tests' call shape from before ADR-0041 D6: `reduced` is now the
    /// site's own product set ([`TileSite::tf32`]), so it rides on the site.
    fn with_set(site: &TileSite, reduced: Option<Reduced>) -> TileSite {
        TileSite {
            tf32: reduced.is_some(),
            ..site.clone()
        }
    }

    fn launch(
        site: &TileSite,
        p: &TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
        reduced: Option<Reduced>,
    ) -> Launch {
        super::launch(&with_set(site, reduced), p, contract, arms)
    }

    fn launch_in(
        site: &TileSite,
        p: &TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
        reduced: Option<Reduced>,
        order: Order,
    ) -> Launch {
        super::launch_in(&with_set(site, reduced), p, contract, arms, order)
    }

    #[allow(clippy::too_many_arguments)]
    fn kernel_text(
        name: &str,
        site: &TileSite,
        p: &TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
        reduced: Option<Reduced>,
        col: bool,
        epi: Option<&EpiKernel>,
        combine: Combine,
    ) -> String {
        super::kernel_text(
            name,
            &with_set(site, reduced),
            p,
            contract,
            arms,
            col,
            epi,
            combine,
        )
    }

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
            radix: None,
            tf32: false,
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
            cands: false,
            free: false,
            ..crate::GpuArms::default()
        }
    }

    const PARTS: [&TargetProfile; 4] = [&RTX_4090, &H100_PCIE, &H100, &RTX_5090];

    /// fold-algebra: a strided-lane site's step is an FMA pair of the step
    /// algebra, and the host fuses it under `contract` — so the device does
    /// too, on the Contract policy. The OFF arm (`--no-fma-pairs`) keeps the
    /// unit-lane rule: separate `fmul` + `fadd`, the Exact policy.
    #[test]
    fn strided_contract_fuses_from_the_step_algebra() {
        let s = site(1, 256, 128, true, false, Ty::f32());
        for pairs in [true, false] {
            let arms = crate::GpuArms {
                pairs,
                ..crate::GpuArms::default()
            };
            let text = kernel_text(
                "strided",
                &s,
                &H100_PCIE,
                true,
                arms,
                None,
                false,
                None,
                Combine::None,
            );
            assert_eq!(text.contains("call float @llvm.fma"), pairs);
            assert_eq!(
                text.contains(" = fmul float ") && text.contains(" = fadd float "),
                !pairs
            );
            let exact = launch(&s, &H100_PCIE, false, arms, None);
            let requested = launch(&s, &H100_PCIE, true, arms, None);
            assert_eq!(
                requested.arithmetic,
                if pairs {
                    Arithmetic::OrderedFma
                } else {
                    Arithmetic::SeparateMulAdd
                }
            );
            if !pairs {
                assert_eq!(
                    requested.rungs, exact.rungs,
                    "separate arithmetic needs Exact policy"
                );
            }
        }
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
            Combine::None,
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
                        let text = kernel_text(
                            "scalar",
                            &s,
                            p,
                            contract,
                            arms,
                            None,
                            false,
                            None,
                            Combine::None,
                        );
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
                            atomic: false,
                            terms: 1,
                            streamk: None,
                            p0: None,
                            head: "entry",
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
                caps: crate::profile::caps_of_cc(70, false),
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
                            Combine::None,
                        );
                        // (The plan line names the site's set, `set=tf32|f32`.)
                        let code: String = text
                            .lines()
                            .filter(|l| !l.starts_with("; mapal.plan"))
                            .collect();
                        assert!(
                            !code.contains("tf32")
                                && !code.contains("mma.sync")
                                && !code.contains("wgmma.mma_async")
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
            for arms in if !p.gpu().unwrap().has_mma_sync() {
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
                    Combine::None,
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
                assert_eq!(wait, enabled && p.gpu().unwrap().has_griddepcontrol());
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
                // fold-algebra: the strided site fuses with the pairs arm on.
                let strided = if arms.pairs {
                    Arithmetic::OrderedFma
                } else {
                    Arithmetic::SeparateMulAdd
                };
                for (s, expected) in sites.iter().zip([strided, Arithmetic::OrderedFma]) {
                    assert_eq!(launch(s, p, true, arms, None).arithmetic, expected);
                    let text =
                        kernel_text("mixed", s, p, true, arms, None, false, None, Combine::None);
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
    /// tile-efficiency T3 `async-depth`: a `cp.async` panel below `tm·tn·bk
    /// ≥ 1024` yields to register staging only where staging's panel is
    /// deeper (4090 128×1024²: 4×4 at bk 16 → staged bk 32); at equal depth
    /// it stays (5090 16×18432×7168, bk 64). There is no `k` test: a `k`
    /// with no deeper panel gives staging none either. OFF: the deepest
    /// `cp.async` panel that fits.
    #[test]
    fn async_depth_yields_only_to_a_deeper_staged_panel() {
        let geo = |p: &TargetProfile, (m, n, k), off: bool| {
            let mut a = crate::GpuArms::default();
            a.off.async_depth = off;
            // The panel depth the launch's own SM share admits (`grid-share`,
            // on wherever the L2 share per SM holds deeper panels) is not
            // the yield this tests.
            a.off.grid_share = true;
            let s = site(m, n, k, true, true, Ty::f32());
            launch(&s, p, false, a, None)
                .tiled
                .map(|g| (g.tm * g.tn, g.bk, g.stages))
        };
        let mm = (128, 1024, 1024);
        assert_eq!(geo(&RTX_4090, mm, true), Some((16, 16, 2)));
        assert_eq!(geo(&RTX_4090, mm, false), Some((16, 32, 1)));
        let skinny = (16, 18432, 7168);
        assert_eq!(geo(&RTX_5090, skinny, false), geo(&RTX_5090, skinny, true));
    }

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
        // hardware-properties: `row-fit` ships filled on the exact face too
        // (its S62 exact cell here was null), so the 4 × 4 tile row-fits.
        assert_eq!(geo(18432, false, shipped), Some((4, 4, 16, 32, 32, 64, 1)));
        // tile-efficiency T2 / per-site-choice A2: the k-major panel's bank
        // pad (4 words a row at bk 64) leaves two `cp.async` stages of bk 64
        // over the share, so the padded async panel halves to bk 32 and
        // register staging's deeper panel — k-major, padded, conflict-free —
        // takes the site, as on the transposed arm.
        let mut kmajor = on("row-fit");
        *kmajor.on.rung("kmajor-a").expect("a rung") = true;
        // hardware-properties: the rules ship no `cp.async` panel filled (it
        // lost filled on every measured part), so both bank-plan arms take
        // register staging's panel.
        assert_eq!(geo(18432, false, kmajor), Some((4, 4, 16, 32, 32, 64, 1)));
        *kmajor.on.rung("bank-plan").expect("a rung") = true;
        assert_eq!(geo(18432, false, kmajor), Some((4, 4, 16, 32, 32, 64, 1)));
        let s = site(16, 18432, 7168, true, true, Ty::f32());
        let k = launch(&s, &RTX_5090, false, kmajor, None)
            .tiled
            .expect("tiled");
        assert_eq!((k.kmajor, k.pad_a), (true, 4));
        assert_eq!(panel_excess(&s, &k, &GpuRungs::ALL, false), [1; 4]);
        assert_eq!(
            geo(18432, false, on("row-fit")),
            Some((4, 4, 16, 32, 32, 64, 1))
        );
        assert_eq!(geo(18432, true, shipped), Some((8, 8, 16, 128, 32, 16, 1)));
        // hardware-properties: the gate forced on keeps the 144-block grid —
        // it fills 0.85 of the 5090's SM slots ([`crate::profile::wave_full`];
        // the S62 v2 cell: kept 0.46 ms, gated one thread a cell 2.07 ms).
        assert_eq!(
            geo(18432, true, on("wave-gate")),
            Some((8, 8, 16, 128, 32, 16, 1))
        );
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
                                        lane_maps(&s, &g, g.kmajor).into_iter().enumerate()
                                    {
                                        let vec = if i == 0 {
                                            vector_a(&s) && (g.stages < 2 || g.kmajor)
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
    /// where the grid is more than a wave. What ships is the rules' tile with
    /// raster on every part (the 3090's arms measured null): only the part
    /// whose L2 the panels leave groups; the forced-ungated arm still
    /// exercises grouping.
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
                cands: false,
                free: false,
                ..crate::GpuArms::default()
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
                    if shipped.raster > 1 {
                        assert!(p.name == "rtx-3090" && n == 8192, "{why}");
                    }
                    assert!(shipped.gy % shipped.raster == 0, "{why}");
                }
            }
        }
    }

    /// S62 integration: `cell-fill`'s one-output-per-thread tile under
    /// `splitk` (reached with `fill-rows` off: the 5090 session's LOO arm at
    /// M16 × 2048 × 7168) stores partition `z`'s partials at `z·rows·c` in the
    /// workspace. Its scalar store dropped that offset, so every partition
    /// raced on slab 0 (the box gate: not bit-exact, a different hash per
    /// process — `benches/results/gpu/s62/integrated/rtx-5090`). S63: the
    /// atomic variant of the same kernel adds into `%o` itself, no slab.
    #[test]
    fn cell_tile_split_stores_into_its_slab() {
        let s = site(16, 2048, 7168, true, true, Ty::f32());
        let mut arms = crate::GpuArms::default();
        arms.off.fill_rows = true;
        for contract in [false, true] {
            let l = launch_in(&s, &RTX_5090, contract, arms, None, Order::Fixed);
            let g = l.tiled.expect("tiled");
            let Combine::Workspace { s: parts } = l.combine else {
                panic!("the fixed order splits the cell tile: {:?}", l.combine)
            };
            assert_eq!((g.tm, g.tn), (1, 1), "the cell tile");
            for (c, slab_offset) in [
                (Combine::Workspace { s: parts }, true),
                (Combine::Atomic { s: parts }, false),
            ] {
                let text = kernel_text("k", &s, &RTX_5090, contract, arms, None, false, None, c);
                let z = text
                    .lines()
                    .find_map(|x| x.trim().split_once(" = udiv ").map(|(z, _)| z.to_string()))
                    .expect("z = ctaid.y / gy");
                let slab = format!("{z}, {}", 16 * 2048);
                assert_eq!(
                    text.lines()
                        .any(|x| x.contains(" = mul ") && x.contains(&slab)),
                    slab_offset,
                    "{c:?}: the workspace store offsets by z·rows·c, the atomic one not"
                );
                assert_eq!(
                    text.contains("atomicrmw fadd ptr addrspace(1)"),
                    !slab_offset,
                    "{c:?}"
                );
                assert!(!text.contains(" fast ") && !text.contains(" reassoc "));
            }
        }
    }

    /// S63 the chooser (ADR-0040 D4) on the measured cells and parts
    /// (`--nocapture` prints the table): the sequential order never cuts;
    /// the fixed order cuts only into a workspace; the free order picks
    /// stream-K, the atomic split, the workspace split or nothing, and each
    /// rung's OFF arm is the next realization down. Every split stays inside
    /// one wave and cuts whole panels (M1/G6).
    #[test]
    fn the_chooser_on_the_measured_cells() {
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
        // The rules (plan-cost's OFF arm; `plan_cost_picks_the_argmin` is ON's).
        let mut arms = crate::GpuArms::default();
        arms.off.plan_cost = true;
        let off = |names: &[&str]| {
            let mut a = arms;
            for n in names {
                *a.off.rung(n).expect("a rung") = true;
            }
            a
        };
        for p in parts {
            let g = p.gpu().unwrap();
            for (m, n, k) in cells {
                let s = site(m, n, k, true, true, Ty::f32());
                for contract in [false, true] {
                    let at = |a, o| launch_in(&s, p, contract, a, None, o);
                    let l = at(arms, Order::Free);
                    let unit = l.tiled.map_or(1, |t| t.bk);
                    let (blocks, wave) = (l.gx * l.gy, l.wave(g));
                    // The fill class the rung rules read (`launch`'s `wps`).
                    let base = p.gpu_tile(m, n, k, &s.elem, false, false);
                    let (fb, ft) = base
                        .map_or(((m * n).div_ceil(NAIVE_THREADS), NAIVE_THREADS), |t| {
                            ((m / t.bm()) * (n / t.bn()), t.threads())
                        });
                    let fill = if g.warps_per_scheduler(fb, ft) < GPU_RESIDENT_BLOCKS {
                        "under"
                    } else {
                        "filled"
                    };
                    println!(
                        "{} {m}x{n}x{k} contract={contract} fill={fill} kernel={} grid={}x{} \
                         block={} blocks={blocks} wave={wave} free={:?} fixed={:?}",
                        p.name,
                        if l.tiled.is_some() { "tiled" } else { "naive" },
                        l.gx,
                        l.gy,
                        l.block,
                        l.combine,
                        at(arms, Order::Fixed).combine,
                    );
                    assert_eq!(at(arms, Order::Sequential).combine, Combine::None);
                    for o in [Order::Free, Order::Fixed] {
                        // (A `fill-wave` split's wave is its own launch's.)
                        let lo = at(arms, o);
                        let c = lo.combine;
                        let parts_ = c.splits();
                        assert!(parts_ == 1 || blocks * parts_ <= lo.wave(g), "{c:?}");
                        assert!(k.is_multiple_of(parts_ * unit), "{c:?}");
                        if let Combine::StreamK { ctas, iters } = c {
                            assert!(ctas <= wave && ctas * iters >= blocks * (k / unit));
                        }
                    }
                    assert!(matches!(
                        at(arms, Order::Fixed).combine,
                        Combine::None | Combine::Workspace { .. }
                    ));
                    // Each OFF arm is the next realization down.
                    let no_sk = at(off(&["streamk"]), Order::Free).combine;
                    assert!(!matches!(no_sk, Combine::StreamK { .. }));
                    let no_at = at(off(&["streamk", "splitk-atomic"]), Order::Free).combine;
                    assert_eq!(no_at, at(arms, Order::Fixed).combine);
                    let none = at(off(&["streamk", "splitk-atomic", "splitk"]), Order::Free);
                    assert_eq!(none.combine, Combine::None);
                    let base = launch(&s, p, contract, arms, None);
                    let geom = |l: &Launch| (l.tiled, l.gx, l.gy, l.block, l.smem, l.raster);
                    assert_eq!(geom(&none), geom(&base), "the order never moves the tile");
                }
            }
        }
        // The one-wave clamp (G6): the 3090 at 16 × 18432 × 7168 holds 984
        // blocks of the 1152-block grid; no split fits one wave. Stream-K
        // would take the tail; its measured row (Filled) ships the grid.
        let r3090 = &crate::profile::RTX_3090;
        let s = site(16, 18432, 7168, true, true, Ty::f32());
        let l = launch_in(&s, r3090, true, arms, None, Order::Free);
        assert_eq!(l.combine.splits(), 1, "{:?}", l.combine);
        let s16 = site(16, 2048, 7168, true, true, Ty::f32());
        // S63 the shipped default per part (`GPU_RUNGS_OFF`, measured,
        // `benches/results/gpu/s63/default-float/`): each is a session arm
        // (site-geometry `fill-wave` off: the S63 chooser; its own arm is
        // `site_geometry.rs`'s).
        let s63 = off(&["fill-wave"]);
        let shipped = |p, (m, n, k)| {
            let c = launch_in(
                &site(m, n, k, true, true, Ty::f32()),
                p,
                true,
                s63,
                None,
                Order::Free,
            )
            .combine;
            match c {
                Combine::None => "none",
                Combine::Workspace { .. } => "workspace",
                Combine::Atomic { .. } => "atomic",
                Combine::StreamK { .. } => "streamk",
            }
        };
        // hardware-properties: the rules' picks (stream-K filled, `splitk`
        // off on the exact face below 128 KiB of L2 a SM, `splitk-atomic` on
        // every part) — the measured cells' winners are replayed in
        // `rung_replay.tsv`.
        let h100 = &H100_PCIE;
        for (p, cell, want) in [
            (&RTX_5090, (16, 2048, 7168), "none"),
            (&RTX_5090, (16, 18432, 7168), "streamk"),
            (&RTX_5090, (4096, 4096, 4096), "streamk"),
            (&RTX_4090, (16, 18432, 7168), "streamk"),
            (h100, (16, 1024, 4096), "atomic"),
            (h100, (16, 2048, 7168), "atomic"),
            (h100, (128, 2048, 7168), "none"),
            (h100, (4096, 4096, 4096), "streamk"),
            (r3090, (16, 1024, 4096), "atomic"),
            (r3090, (16, 2048, 7168), "streamk"),
            (r3090, (128, 2048, 7168), "atomic"),
            (r3090, (4096, 4096, 4096), "streamk"),
        ] {
            assert_eq!(shipped(p, cell), want, "{} {cell:?}", p.name);
        }
        // The measured table: the 3090 ships no workspace split at (filled,
        // exact) — its one split cell there lost (S62) — on the fixed face.
        let pre = off(&[
            "row-fit",
            "wave-gate",
            "grid-share",
            "panel-regs",
            "fill-rows",
            "cell-fill",
        ]);
        let l = launch_in(&s16, r3090, false, pre, None, Order::Fixed);
        assert_eq!(l.combine, Combine::None);
        let l = launch_in(&s16, r3090, true, pre, None, Order::Fixed);
        assert!(matches!(l.combine, Combine::Workspace { .. }));
    }

    /// plan-cost: on a part with plan rates, a free-order site's fold is the
    /// cheapest candidate the rungs admit — never dearer than the rules' pick
    /// (a candidate itself) — every candidate cuts `k` into whole panels, and
    /// each rung's OFF arm removes its candidates; on a part without the
    /// rates, and on the other orders, the plan is the rules' byte for byte
    /// (D8: the rules' launch, every cell, part, face and order).
    #[test]
    fn plan_cost_picks_the_argmin() {
        let cells = [
            (16, 2048, 7168),
            (16, 1024, 4096),
            (128, 2048, 7168),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ];
        // The rung forced on: the 4090/5090 rows ship it off (plan-cost stopgap).
        let mut on = crate::GpuArms::default();
        on.on.plan_cost = true;
        let mut rules = on;
        rules.on.plan_cost = false;
        rules.off.plan_cost = true;
        let mut no_atomic = on;
        no_atomic.off.splitk_atomic = true;
        no_atomic.off.streamk = true;
        let mut costed = 0;
        // The launch but for the rung set the arms differ in.
        let same = |a: Launch, b: Launch| {
            Launch {
                rungs: b.rungs,
                ..a
            } == b
        };
        for p in crate::profile::GPU_PARTS {
            for (m, n, k) in cells {
                let s = site(m, n, k, true, true, Ty::f32());
                for contract in [false, true] {
                    let l = launch(&s, p, contract, on, None);
                    let cands = plan_cands(&l, &s, p, contract, on, Order::Free);
                    for o in [Order::Sequential, Order::Fixed] {
                        assert!(plan_cands(&l, &s, p, contract, on, o).is_empty());
                        let at = |a| launch_in(&s, p, contract, a, None, o);
                        assert!(same(at(on), at(rules)), "{} {o:?}", p.name);
                    }
                    let at = |a| launch_in(&s, p, contract, a, None, Order::Free);
                    if p.plan_rates().is_none() {
                        assert!(cands.is_empty(), "{}", p.name);
                        assert!(same(at(on), at(rules)), "D8: {} {m}x{n}x{k}", p.name);
                        continue;
                    }
                    costed += 1;
                    let pick = cands.first().expect("the ordered kernel is a candidate");
                    assert_eq!(at(on).combine, pick.combine);
                    let rule = at(rules).combine;
                    let rule_ns = cands.iter().find(|c| c.combine == rule).expect("rules'");
                    assert!(pick.cost.ns <= rule_ns.cost.ns);
                    assert!(cands.windows(2).all(|w| w[0].cost.ns <= w[1].cost.ns));
                    for c in &cands {
                        let r = resized(&s, p, contract, on, l, c.combine);
                        let bk = r.tiled.map_or(1, |t| t.bk);
                        assert!(k.is_multiple_of(c.combine.splits() * bk), "{c:?}");
                    }
                    let l = launch(&s, p, contract, no_atomic, None);
                    let few = plan_cands(&l, &s, p, contract, no_atomic, Order::Free);
                    assert!(
                        few.iter().all(|c| matches!(
                            c.combine,
                            Combine::None | Combine::Workspace { .. }
                        ))
                    );
                }
            }
        }
        assert!(
            costed >= 20,
            "the calibrated parts reach the chooser ({costed})"
        );
    }

    /// A forced count (`--splitk=S`, the part's `Geom::split`) stays the plan
    /// under plan-cost: the chooser ranks only that count's splits, so the
    /// ordered kernel or the rules' pick never replaces it.
    #[test]
    fn a_forced_split_is_the_only_plan() {
        let mut forced = crate::GpuArms::default();
        forced.splitk = 4;
        forced.on.plan_cost = true;
        let mut ranked = 0;
        for p in [&RTX_4090, &RTX_5090] {
            for (m, n, k) in [(16, 2048, 7168), (512, 512, 512), (4096, 4096, 4096)] {
                let s = site(m, n, k, true, true, Ty::f32());
                let l = launch(&s, p, true, forced, None);
                let cands = plan_cands(&l, &s, p, true, forced, Order::Free);
                let got = launch_in(&s, p, true, forced, None, Order::Free).combine;
                assert!(cands.iter().all(|c| c.combine.splits() == 4), "{cands:?}");
                if !cands.is_empty() {
                    ranked += 1;
                    assert_eq!(got.splits(), 4, "{} {m}x{n}x{k}", p.name);
                }
            }
        }
        assert!(ranked > 0, "the forced count reaches the chooser");
    }

    /// plan-cost `linear` (D6): a site table's `S` keeps each product's depth
    /// — `k/S` roundings in its partial, `S·T − 1` adds into the zeroed sum —
    /// within the source's `k + T − 1` (`S·T ≤ k` for `S > 1`: `k = 4, T = 3`
    /// takes no `S = 2`), over skinny and short-`k` sites.
    #[test]
    fn site_table_split_stays_within_the_source_bound() {
        // `linear` forced on: the rules ship it off (no measured gain).
        let mut on = crate::GpuArms::default();
        on.on.linear = true;
        let (mut tables, mut split) = (0, 0);
        for p in [&RTX_4090, &RTX_5090] {
            for (m, n, k) in [1u64, 2, 16]
                .iter()
                .flat_map(|&m| [32u64, 128, 1024].map(move |n| (m, n)))
                .flat_map(|(m, n)| [4u64, 6, 8, 12, 16, 24, 64, 1024].map(move |k| (m, n, k)))
            {
                let s = site(m, n, k, true, true, Ty::f32());
                for t in 2..=7 {
                    let bound = mapal_ir::bound_coef(k + t - 1, false);
                    let Some((sp, _)) = concat_plan(&s, p, true, on, t, bound) else {
                        continue;
                    };
                    tables += 1;
                    split += u64::from(sp > 1);
                    assert!(
                        k / sp + sp * t - 1 <= k + t - 1,
                        "{} {m}x{n}x{k} T={t}: S={sp}",
                        p.name
                    );
                }
            }
        }
        assert!(tables > 0 && split > 0, "{tables} tables, {split} split");
    }

    /// plan-cost §5 pick stability, over sites of the generator's sizes
    /// (`testgen::geom_prog_strategy`: rows 1…1024, c and k 64…4096): ±10 %
    /// on any one rate moves every candidate's time by at most the factor the
    /// rate does, so it changes a pick only where the runner-up the plan line
    /// reports is within that band (`alt.ns ≤ 1.1/0.9 · pick.ns`).
    #[test]
    fn plan_pick_is_stable_under_rate_noise() {
        let on = crate::GpuArms::default();
        let (mut flips, mut checked) = (0, 0);
        for p in [&RTX_4090, &RTX_5090] {
            let g = *p.gpu().unwrap();
            for (m, n, k) in [1u64, 16, 64, 128, 256, 512, 1024]
                .iter()
                .flat_map(|&m| [64u64, 256, 1024, 4096].map(move |n| (m, n)))
                .flat_map(|(m, n)| [64u64, 512, 4096].map(move |k| (m, n, k)))
            {
                let s = site(m, n, k, true, true, Ty::f32());
                let l = launch(&s, p, true, on, None);
                let cands = plan_cands(&l, &s, p, true, on, Order::Free);
                let (Some(pick), Some(alt)) = (cands.first(), cands.get(1)) else {
                    continue;
                };
                // The rates (the physical-property names after `clock_khz`
                // reshape the part, not its rates).
                for (i, name) in crate::profile::DeviceCost::NAMES
                    .iter()
                    .enumerate()
                    .take(24)
                {
                    let v = g.rate(name);
                    for f in [0.9, 1.1] {
                        let mut over = crate::profile::CostOver::default();
                        over[i] = Some(((v as f64) * f).round() as u64);
                        let q = TargetProfile {
                            machine: crate::profile::Machine::Gpu(g.with_cost(&over)),
                            ..*p
                        };
                        let lq = launch(&s, &q, true, on, None);
                        let moved = plan_cands(&lq, &s, &q, true, on, Order::Free);
                        checked += 1;
                        if moved.first().map(|c| c.combine) != Some(pick.combine) {
                            flips += 1;
                            assert!(
                                alt.cost.ns <= pick.cost.ns * 1.1 / 0.9 + 1.0,
                                "{} {m}x{n}x{k} {name}×{f}: {} → {:?}, runner-up {} at {} vs {}",
                                p.name,
                                pick.label(),
                                moved.first().map(|c| c.label()),
                                alt.label(),
                                alt.cost.ns,
                                pick.cost.ns
                            );
                        }
                    }
                }
            }
        }
        eprintln!("plan-cost stability: {flips} flips in {checked} perturbed picks");
        assert!(checked > 1000);
    }

    /// S64 evidence pass: each kernel's `; mapal.plan` line reads back as the
    /// launch it was emitted from (the chooser's cells and parts, both step
    /// faces, every realization) — the intent `benches/gpu/evidence.py` checks.
    #[test]
    fn plan_line_reads_back_as_its_launch() {
        let cells = [
            (16, 2048, 7168),
            (16, 18432, 7168),
            (128, 2048, 7168),
            (4096, 4096, 4096),
        ];
        let parts = [&RTX_5090, &RTX_4090, &H100_PCIE, &crate::profile::RTX_3090];
        let arms = crate::GpuArms::default();
        for p in parts {
            for (m, n, k) in cells {
                let s = site(m, n, k, true, true, Ty::f32());
                for (contract, order) in [
                    (false, Order::Sequential),
                    (true, Order::Free),
                    (false, Order::Fixed),
                ] {
                    let l = launch_in(&s, p, contract, arms, None, order);
                    let text =
                        kernel_text("k", &s, p, contract, arms, None, false, None, l.combine);
                    let line = text.lines().next().expect("a plan line");
                    let f: std::collections::HashMap<&str, &str> = line
                        .strip_prefix("; mapal.plan ")
                        .expect("the first line")
                        .split(' ')
                        .filter_map(|kv| kv.split_once('='))
                        .collect();
                    let n = |key: &str| f[key].parse::<u64>().expect(key);
                    assert_eq!(
                        (n("gx"), n("gy"), n("block"), n("smem"), n("raster")),
                        (l.gx, l.gy, l.block, l.smem, l.raster),
                        "{line}"
                    );
                    let g = l.tiled.expect("the FMA tile");
                    assert_eq!(
                        (
                            n("tm"),
                            n("tn"),
                            n("tx"),
                            n("ty"),
                            n("bk"),
                            n("stages"),
                            n("blocks_per_sm")
                        ),
                        (g.tm, g.tn, g.tx, g.ty, g.bk, g.stages, g.blocks_per_sm)
                    );
                    assert_eq!(n("S"), l.combine.splits());
                    let on: Vec<&str> = f["rungs_on"].split(',').collect();
                    for r in KERNEL_RUNGS {
                        assert_eq!(on.contains(&r), l.rungs.get(r), "{r} in {line}");
                    }
                    assert_eq!(f["face"], format!("{:?}", l.arithmetic));
                }
            }
        }
    }

    /// S63 `streamk`: the kernel walks its share of the `tiles × panels`
    /// space piece by piece — one panel loop per piece entered from `seg`,
    /// the seed only on a tile's first piece, the store an atomic add, the
    /// latch back to `seg` — and keeps the register-staged tile's panel loop.
    #[test]
    fn streamk_kernel_walks_pieces() {
        let s = site(128, 2048, 7168, true, true, Ty::f32());
        let arms = crate::GpuArms::default();
        let l = launch(&s, &RTX_5090, true, arms, None);
        let g = l.tiled.expect("tiled");
        if g.stages >= 2 {
            return;
        }
        let panels = 7168 / g.bk;
        let blocks = l.gx * l.gy;
        let iters = (blocks * panels).div_ceil(l.wave(RTX_5090.gpu().unwrap()));
        let c = Combine::StreamK {
            ctas: (blocks * panels).div_ceil(iters),
            iters,
        };
        let text = kernel_text("k", &s, &RTX_5090, true, arms, None, false, None, c);
        assert!(text.contains("seg:\n") && text.contains("segl:\n") && text.contains("exit:\n"));
        assert!(text.contains("label %seg, label %exit"));
        assert!(text.contains(&format!(", {iters}\n")), "start = cta·iters");
        assert!(
            text.contains(", %seg], ["),
            "the panel loop is entered per piece"
        );
        assert!(text.contains("atomicrmw fadd ptr addrspace(1)"));
        assert!(
            !text
                .lines()
                .any(|x| x.trim_start().starts_with("store") && x.contains("addrspace(1)")),
            "no plain output store"
        );
        assert!(text.contains("select i1"), "the seed rides p0 = 0");
        assert!(!text.contains("@SEG_NEXT@") && !text.contains("@NEXT"));
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
        let caps = [
            CapArg::Ptr,
            CapArg::Law("fn3".to_string(), "float".to_string()),
        ];
        let k = cell_kernel_text(
            1,
            "fn2_cell",
            300,
            "float",
            "{ ptr, ptr, i32 }",
            None,
            &caps,
            CellGeo::ONE,
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
        // A uniform rides the launch's 8-byte slot and is narrowed in the cell.
        let caps = [CapArg::Uniform("i32".into()), CapArg::Ptr];
        let k = cell_kernel_text(
            2,
            "fn4",
            8,
            "float",
            "{ i32, ptr, i32 }",
            None,
            &caps,
            CellGeo::ONE,
        );
        assert!(
            k.contains("(ptr addrspace(1) %o, i64 %c0, ptr addrspace(1) %c1)"),
            "{k}"
        );
        assert!(k.contains("%g0 = trunc i64 %c0 to i32"), "{k}");
        assert!(
            k.contains("%a0 = insertvalue { i32, ptr, i32 } poison, i32 %g0, 0"),
            "{k}"
        );
    }

    /// `idx32`: every S52 site indexes in `i32`; an operand past 2³¹ elements
    /// keeps `i64`.
    #[test]
    fn idx32_takes_i32_exactly_when_every_index_fits() {
        for p in PARTS {
            for s in s52_sites() {
                let l = launch(&s, p, false, force(GpuRungs::ALL), None);
                assert!(
                    max_index(&s, &l, Combine::None) < 1 << 31,
                    "{} {s:?}",
                    p.name
                );
                let k = kernel_text(
                    "k",
                    &s,
                    p,
                    false,
                    force(GpuRungs::ALL),
                    None,
                    false,
                    None,
                    Combine::None,
                );
                assert!(k.contains("add nuw nsw i32"), "{} {s:?}", p.name);
            }
        }
        // 2¹⁶ rows of 2¹⁶: `a`'s last element is 2³² − 1.
        let big = site(1 << 16, 1 << 16, 1 << 16, true, true, Ty::f32());
        let l = launch(&big, &RTX_4090, false, force(GpuRungs::ALL), None);
        assert!(max_index(&big, &l, Combine::None) >= 1 << 31);
        let k = kernel_text(
            "k",
            &big,
            &RTX_4090,
            false,
            force(GpuRungs::ALL),
            None,
            false,
            None,
            Combine::None,
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
                    for (map, n) in lane_maps(&s, &g, g.kmajor) {
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
            let [(map, n), _] = lane_maps(s, g, g.kmajor);
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
                // tile-efficiency T2's k-major panel has no transposed store.
                let on = GpuRungs {
                    cp_async: false,
                    async_stage: false,
                    kmajor_a: false,
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

    /// A tile of the bank reference (`.claude/sprints/per-site-choice/
    /// scratch/bank/bank.py`): `(tm, tn, tx, ty, bk, stages)`, warp and
    /// interleave rungs, k-major or transposed.
    fn bank_tile(t: [u64; 6], warp: bool, interleave: bool, kmajor: bool) -> (GpuTile, GpuRungs) {
        let [tm, tn, tx, ty, bk, stages] = t;
        let g = GpuTile {
            tm,
            tn,
            tx,
            ty,
            bk,
            blocks_per_sm: 1,
            stages,
            pad_a: 0,
            a_row_lanes: 0,
            held: 0,
            kmajor,
        };
        let r = GpuRungs {
            warp,
            interleave,
            ..GpuRungs::ALL
        };
        (g, r)
    }

    /// per-site-choice D4: a hand-built alias — the output shares a carry's
    /// slot with an operand — is not `RERUN`; distinct slots are.
    #[test]
    fn an_aliased_output_is_not_rerun_safe() {
        let key = |o: u32| if o == 7 { 3 } else { o };
        assert!(apart(&[1, 2], &[5], key));
        assert!(!apart(&[1, 3], &[7], key), "7 shares 3's slot");
        assert!(!apart(&[4], &[4], |o| o));
    }

    /// per-site-choice A3: [`bank_degree`] / [`panel_excess`] reproduce the
    /// reference calculator's self-test (bank.py `selftest`).
    #[test]
    fn bank_model_matches_the_reference() {
        let s = site(4096, 4096, 4096, true, true, Ty::f32());
        // S52: register-staged transposed RowMajor copy, bm = 128 unpadded: 4-way.
        let (g, r) = bank_tile([8, 8, 16, 16, 16, 1], true, true, false);
        assert_eq!(panel_excess(&s, &g, &r, false)[2], 4);
        // S53 `pad`: 2 lanes a row, pad 4: conflict-free.
        let p = GpuTile {
            pad_a: 4,
            a_row_lanes: 2,
            ..g
        };
        assert_eq!(panel_excess(&s, &p, &r, false), [1; 4]);
        // async transposed element copy at its `conflict_free_pad` (8).
        assert_eq!(
            crate::profile::conflict_free_pad(128, |l| (l % 4, l / 4)),
            Some(8)
        );
        let a = GpuTile {
            stages: 3,
            pad_a: 8,
            ..g
        };
        assert_eq!(panel_excess(&s, &a, &r, false)[2], 1);
        // One 16-byte read of 4 rows 4 apart at a stride ≡ 0 mod 32 words:
        // 16 words on 4 banks ⇒ degree 4, ideal 1; a 16-byte copy of 32
        // consecutive chunks: 128 words, 4 a bank ⇒ degree 4 = ideal 4.
        assert_eq!(bank_degree((0..32).map(|l| ((l / 8) * 4 * 32, 4))), (4, 1));
        assert_eq!(bank_degree((0..32).map(|l| (4 * l, 4))), (4, 4));
        assert_eq!(bank_degree((0..32).map(|l| (l, 1))), (1, 1));
        assert_eq!(bank_degree((0..32).map(|_| (7, 1))), (1, 1), "a broadcast");
    }

    /// per-site-choice A1/A2: the 30 distinct k-major geometries the
    /// tile-efficiency sweep emitted (bank.md §3): with the shipped row map
    /// (`trow·aw + …`, unpadded) the fragment read is 2- or 4-way in 27;
    /// with rows `trow + i·ty` and [`kmajor_pad`] every `a` class is
    /// conflict-free. Before = [`bank_degree`] on the shipped map's
    /// addresses; after = [`panel_excess`] on the emitter's maps.
    #[test]
    fn kmajor_panel_is_conflict_free_on_the_swept_geometries() {
        #[rustfmt::skip]
        let geoms: [([u64; 6], bool, bool); 30] = [
            ([4, 4, 8, 8, 32, 3], true, true), ([8, 8, 16, 16, 16, 1], false, true),
            ([4, 4, 8, 16, 64, 2], true, true), ([2, 4, 32, 8, 64, 1], false, true),
            ([1, 4, 8, 8, 32, 1], false, true), ([8, 16, 8, 16, 8, 1], false, true),
            ([8, 4, 16, 16, 32, 2], true, true), ([8, 16, 8, 16, 8, 1], true, true),
            ([8, 4, 32, 8, 32, 2], true, true), ([8, 4, 16, 16, 64, 2], true, true),
            ([4, 4, 8, 16, 128, 2], true, true), ([4, 4, 8, 8, 128, 3], true, true),
            ([8, 4, 16, 16, 16, 1], true, true), ([4, 4, 8, 8, 8, 1], true, true),
            ([8, 8, 16, 2, 16, 1], true, true), ([8, 8, 16, 16, 16, 1], true, true),
            ([4, 4, 16, 16, 16, 1], true, true), ([8, 8, 16, 16, 16, 3], true, true),
            ([8, 4, 8, 8, 8, 1], true, true), ([4, 4, 8, 8, 128, 1], true, true),
            ([1, 1, 16, 16, 64, 1], false, false), ([2, 4, 8, 8, 16, 1], true, true),
            ([1, 4, 8, 8, 32, 1], true, true), ([8, 4, 16, 16, 16, 1], false, false),
            ([8, 8, 8, 8, 8, 1], true, true), ([8, 8, 8, 16, 8, 1], false, false),
            ([4, 4, 8, 8, 128, 1], false, true), ([4, 4, 8, 16, 16, 1], false, true),
            ([8, 8, 8, 8, 8, 1], false, true), ([8, 8, 8, 16, 8, 1], false, true),
        ];
        let s = site(4096, 4096, 4096, true, true, Ty::f32());
        let mut table =
            String::from("tm tn tx ty bk stages warp il | before lds st | pad after lds st\n");
        let mut conflicted = 0;
        for (t, warp, il) in geoms {
            let (g, r) = bank_tile(t, warp, il, true);
            // The shipped map: rows `trow·a_step + (i / aw)·a_chunk + i % aw`.
            let f = frag_layout(&s, &g, false, il);
            let threads = g.threads();
            let before = (0..g.tm)
                .flat_map(|i| (0..threads / WARP).map(move |w| (i, w)))
                .map(|(i, w)| {
                    let rows = (w * WARP..(w + 1) * WARP).map(|tid| {
                        let trow = thread_coords(&mut Num, &tid, &g, warp, false, WARP).0;
                        let row = trow * f.a_step + (i / f.aw) * f.a_chunk + i % f.aw;
                        (row * g.bk, 4)
                    });
                    let (d, id) = bank_degree(rows);
                    d.div_ceil(id)
                })
                .max()
                .unwrap();
            let [lds0, _, st0, _] = panel_excess(&s, &g, &r, false);
            assert_eq!(st0, 1, "{t:?}: the 16-byte copy is conflict-free unpadded");
            let pad = kmajor_pad(&s, &g, &r);
            let [lds, _, st, _] = panel_excess(&s, &GpuTile { pad_a: pad, ..g }, &r, false);
            assert!(pad.is_multiple_of(4) && pad < 32, "{t:?}: pad {pad}");
            assert_eq!(
                (lds, st),
                (1, 1),
                "{t:?} warp {warp} interleave {il}: pad {pad}"
            );
            conflicted += usize::from(before > 1);
            table += &format!(
                "{t:?} {warp} {il} | {before} {st0} | {pad} {lds} {st} (new map unpadded: {lds0})\n"
            );
        }
        eprintln!("{table}");
        assert!(
            conflicted >= 25,
            "the shipped map conflicts: {conflicted} of 30"
        );
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
            cands: false,
            free: false,
            ..crate::GpuArms::default()
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
                let text = kernel_text(
                    "m16",
                    &s,
                    p,
                    contract,
                    arms,
                    None,
                    false,
                    None,
                    Combine::None,
                );
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
            "kmajor-a",
            "full-tile",
        ]);
        let s = site(16, 2048, 7168, true, true, Ty::f32());
        let l = launch(&s, &RTX_5090, false, arms, None);
        assert_eq!(
            (l.tiled, l.gx, l.gy, l.block, l.smem),
            (None, 128, 1, 256, 0)
        );
        // hardware-properties: `pad` ships under-filled on both faces (the
        // H100 exact face measured it winning, the 5090's was null), so the
        // exact face's transposed `a` store is padded as the contract one.
        let s = site(128, 2048, 7168, true, true, Ty::f32());
        for (contract, smem) in [(false, 2176), (true, 2176)] {
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
        use crate::profile::GPU_PARTS;
        for p in GPU_PARTS {
            for (m, n, k) in [(4096, 4096, 4096), (16, 18432, 7168)] {
                let s = site(m, n, k, true, true, Ty::f32());
                for contract in [false, true] {
                    let text = |arms| {
                        kernel_text("f", &s, p, contract, arms, None, false, None, Combine::None)
                    };
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
        use crate::profile::GPU_PARTS;
        let s = site(1, 2048, 7168, true, true, Ty::f32());
        for p in GPU_PARTS {
            for contract in [false, true] {
                let l = launch(&s, p, contract, crate::GpuArms::default(), None);
                assert_eq!(l.tiled, None, "{}", p.name);
            }
        }
    }
}
