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
    Face, GPU_RESIDENT_BLOCKS, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, Home, MMA_K,
    MMA_M, MMA_N, MMA_SYNC_16, MmaFmt, NAIVE_UNROLL, PlanComb, PlanCost, PlanShape,
    WARPGROUP_WARPS, WGMMA_ATOM_ROWS, WGMMA_K, WGMMA_M, WGMMA_SWIZZLE_BYTES, bank_degree,
    fma_held_regs,
};
use crate::{Order, Reduced};
use mapal_ir::split::{SplitForm, SplitKind};
use mapal_ir::{RowFoldOp, RowFoldSite};

/// Bytes one shared-memory wavefront serves: 32 banks × 4 B. An ISA rule on
/// every NVIDIA part since Kepler.
pub(crate) const GPU_WAVEFRONT_BYTES: u64 = 128;
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
/// sweep-gemm-2 `warm-zero`: the module zeroes outputs; the init thread runs
/// one zero first, so the driver's memset path is initialized before the clock.
pub(crate) const GPU_INIT_WARM_ZERO: u32 = 32;

/// `main`'s first call in a module with a GPU site: hands the runtime its host
/// rungs. Off, the runtime does nothing until the first site, the S52 path.
/// `zeroes`: some launch of the module zeroes an output ([`module_zeroes`]).
pub(crate) fn init_call(rungs: crate::GpuRungs, streams: bool, pdl: bool, zeroes: bool) -> String {
    let flags = u32::from(rungs.eager_init) * GPU_INIT_EAGER
        + u32::from(streams) * GPU_INIT_STREAMS
        + u32::from(pdl) * GPU_INIT_PDL
        + u32::from(rungs.preload) * GPU_INIT_PRELOAD
        + u32::from(rungs.stage && rungs.async_h2d) * GPU_INIT_STAGE
        + u32::from(rungs.warm_zero && zeroes) * GPU_INIT_WARM_ZERO;
    format!("  call void @mapal_gpu_init(i32 {flags})\n")
}

/// sweep-gemm-2 `warm-zero`: does a launch of this host text zero device
/// bytes — a direct `mapal_gpu_zero` call, or a site table row (`%mapal_cand
/// { i64 id, i64 rid, i64 zero, i64 ws, … }`) whose zero bytes are not 0 (an
/// atomic split or a stream-K candidate), or whose workspace has no reduce
/// kernel (`rid` 0, `ws` not 0: sweep-runtime-2 `streamk-epi`'s sums and
/// tickets, which the runtime zeroes before the launch)?
pub(crate) fn module_zeroes(host: &str) -> bool {
    let field = |r: &str, i: usize| -> Option<u64> {
        r.split(", ")
            .nth(i)
            .and_then(|z| z.strip_prefix("i64 "))
            .and_then(|z| z.parse::<i64>().ok())
            .map(|z| z as u64)
    };
    host.contains("call void @mapal_gpu_zero(")
        || host
            .lines()
            .filter(|l| l.contains(" x %mapal_cand] ["))
            .flat_map(|l| l.split("%mapal_cand { ").skip(1))
            .any(|r| {
                field(r, 2).is_some_and(|z| z != 0)
                    || (field(r, 1) == Some(0) && field(r, 3).is_some_and(|w| w != 0))
            })
}

/// S54 `pdl`: a programmatic dependent launch needs `griddepcontrol`
/// ([`crate::profile::Caps::griddepcontrol`]). One answer for the kernel text
/// and the runtime flag — a kernel launched dependent without the wait would
/// race. cross-arch: the module's side of the price
/// ([`crate::rules::pdl_pays`]) is whether its launches repeat
/// ([`launches_repeat`]); a forced `--gpu-on=pdl` arm takes any module.
pub(crate) fn pdl(
    ir: &CategoryIr,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> bool {
    pdl_part(profile, contract, arms) && (arms.on.pdl || launches_repeat(ir))
}

/// The part's side of [`pdl`]: the capability and the rule over its measured
/// launches (the tile kernel's text reads it; the module strips its waits
/// where [`pdl`] is off).
pub(crate) fn pdl_part(profile: &TargetProfile, contract: bool, arms: crate::GpuArms) -> bool {
    profile.gpu().is_some_and(|g| g.has_griddepcontrol()) && host_rungs(profile, contract, arms).pdl
}

/// cross-arch `pdl`: **a device site inside a canonical loop** — a tile,
/// cell or row-fold site in the decide or advance order of a loop of any
/// function (`loop_structure`, `loop_plan`, `tile_plan`, `cell_plan`,
/// `row_fold`): the program's launches repeat with the loop's trips, so its
/// chain of dependent launches grows past the head a programmatic launch
/// pays once.
pub(crate) fn launches_repeat(ir: &CategoryIr) -> bool {
    ir.funcs().any(|(f, _)| {
        let tiles = ir.tile_plan(f);
        let cells = ir.cell_plan(f);
        let folds = ir.row_fold(f);
        ir.loop_structure(f).iter().any(|lscc| {
            lscc.merges.iter().any(|&merge| {
                ir.loop_plan(f, merge).is_some_and(|lp| {
                    lp.decide_order.iter().chain(&lp.advance_order).any(|&m| {
                        tiles.sites.contains_key(m)
                            || cells.contains_key(m)
                            || folds.contains_key(m)
                    })
                })
            })
        })
    })
}

/// The `griddepcontrol` lines of a module whose launches are plain ([`pdl`]
/// off where the part has it): the tile kernels' text reads the part's side
/// ([`pdl_part`]); without a dependent launch their wait is dropped, so the
/// module is the text it was without `pdl`.
pub(crate) fn strip_pdl(module: &str) -> String {
    module
        .lines()
        .filter(|l| !l.contains("griddepcontrol."))
        .map(|l| format!("{l}\n"))
        .collect()
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

/// [`rungs_at`] for one site: cross-arch `row-split` priced over the site's
/// row — a retained row's pass streams its `k · c` operand
/// ([`crate::rules::row_split_pays`]); a forced arm stays forced.
fn site_rungs(
    profile: &TargetProfile,
    wps: u64,
    face: Face,
    arms: crate::GpuArms,
    site: &TileSite,
) -> crate::GpuRungs {
    let r = rungs_at(profile, wps, face, arms);
    let bytes = site.k * site.c * site.b.elem_bytes(&site.elem);
    let priced = arms.on.row_split
        || profile
            .gpu()
            .is_none_or(|g| crate::rules::row_split_pays(g, bytes));
    crate::GpuRungs {
        row_split: r.row_split && priced,
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
pub(crate) const S54_DECLS: [(&str, &str); 8] = [
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
    // caching-memory `retain-pieces`: gathered rows placed at their stride.
    (
        "@mapal_gpu_copy2d(",
        "declare void @mapal_gpu_copy2d(i64, i64, i64, i64, i64, i64)\n",
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
        && uniform_aligned(site)
}

/// long-prefill `uniform-base`: a site whose operand address takes terms of a
/// number its map captures ([`mapal_ir::RadixTile::ua`]) — its launch moves
/// the operand's pointer by them, so it launches on its own.
pub(crate) fn uniform_site(site: &TileSite) -> bool {
    site.radix.as_deref().is_some_and(|r| r.uniform())
}

/// The pointer a launch moves by a captured number keeps the alignment every
/// kernel's widest load of that operand assumes: a vector of `lanes =
/// 16 / e` elements in the operand's own storage — 16 bytes of an `f32`
/// operand ([`GPU_VEC_BYTES`]); a narrow operand's register-staged vector is
/// the same `lanes` elements of its format (`<4 x bfloat>`, 8 bytes, ADR-0043),
/// and the kernels that copy 16 narrow bytes at once (`narrow-async`'s
/// `cp.async`, the 16-bit `mma` panels, `vec-cells`) take no mixed-radix site.
/// Each term's step, in the operand's bytes, is whole such vectors. Otherwise
/// the site is no device site.
fn uniform_aligned(site: &TileSite) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&site.elem);
    site.radix.as_deref().is_none_or(|r| {
        [(&r.ua, &site.a), (&r.ub, &site.b)]
            .iter()
            .all(|(ds, read)| {
                let e = read.elem_bytes(&site.elem);
                ds.iter().all(|d| (d.coef * e).is_multiple_of(lanes * e))
            })
    })
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
    carry: bool,
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
        let fed = [site.a.slot, site.b.slot].into_iter().any(|k| {
            super::pair_source_ir(ir, src, k).is_some_and(|o| device_made(ir, &dev, o, carry))
        });
        let flops = 2u64
            .saturating_mul(site.rows)
            .saturating_mul(site.c)
            .saturating_mul(site.k);
        let bytes = crate::profile::site_bytes(site);
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
///
/// `carry` (onto-device `carry-made`): a loop's carried view is made where the
/// next state replacing it is (the back edge's slot), and a loop's exit value
/// where its payload is — a value the device made last trip is device-made.
/// A cycle through the back edge alone proves nothing (each object is asked
/// once per query).
pub(crate) fn device_made(
    ir: &CategoryIr,
    dev: &SecondaryMap<MorphismId, ()>,
    o: ObjectId,
    carry: bool,
) -> bool {
    device_made_at(ir, dev, o, carry, &mut Vec::new())
}

fn device_made_at(
    ir: &CategoryIr,
    dev: &SecondaryMap<MorphismId, ()>,
    o: ObjectId,
    carry: bool,
    seen: &mut Vec<ObjectId>,
) -> bool {
    if seen.contains(&o) || seen.len() > 4096 {
        return false;
    }
    seen.push(o);
    if carry && let Some(next) = carried_from(ir, o) {
        return device_made_at(ir, dev, next, carry, seen);
    }
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
        device_made_at(ir, dev, morph.source, carry, seen)
    } else {
        pairs
            .into_iter()
            .any(|x| device_made_at(ir, dev, x, carry, seen))
    }
}

/// The value a loop view `o` carries in from the trip before (the back
/// edge's next-state slot of a canonical loop: `Proj{k}` of its merge, or the
/// merge of a one-value loop), or the payload a loop's exit value is.
fn carried_from(ir: &CategoryIr, o: ObjectId) -> Option<ObjectId> {
    let obj = ir.object(o)?;
    let (merge, k) = if obj.kind == mapal_ir::ObjectKind::LoopMerge {
        (o, None)
    } else {
        let &p = ir.in_edges(o).first()?;
        let morph = ir.morphism(p)?;
        match morph.op {
            Operation::Proj { index }
                if ir.object(morph.source)?.kind == mapal_ir::ObjectKind::LoopMerge =>
            {
                (morph.source, Some(index))
            }
            Operation::LoopExit => return super::pair_source_ir(ir, morph.source, 0),
            _ => return None,
        }
    };
    let plan = ir.loop_plan(ir.owner(merge), merge)?;
    let next = super::pair_source_ir(ir, plan.back_route, 0)?;
    match k {
        Some(k) => super::pair_source_ir(ir, next, k),
        None => Some(next),
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
        for (m, s) in ir
            .tile_plan_radix(f)
            .into_iter()
            .filter(|(_, s)| s.post.is_none())
        {
            let l = launch(&s, profile, contract, arms);
            let staged = l.tiled.is_some() || l.mma.is_some();
            if staged && gpu_site(&s, profile, rungs.window) {
                plan.sites.insert(m, s);
            }
        }
    }
    // device-geometry `tile-post`: a sum of products whose body goes on past
    // its fold is a site whose store evaluates the rest (`post_sites`).
    for (m, s) in post_sites(ir, f, profile, contract, arms) {
        plan.sites.insert(m, s);
    }
    // ADR-0043: a sum of products over narrow storage (`tile_plan_narrow`),
    // where the device loads the format (the epilogue ones in `post_sites`).
    if rungs.narrow_tile {
        for (m, s) in ir.tile_plan_narrow(f).sites.iter() {
            let strided = s.a.clane > 1 || s.b.clane > 1;
            if s.post.is_none()
                && (rungs.lane_stride || !strided)
                && gpu_site(s, profile, rungs.window)
            {
                plan.sites.insert(m, s.clone());
            }
        }
    }
    // S62: an extra is a site only where it is placed on the device.
    let dev = placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
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

/// A mixed-radix site whose `k` is not cut: every one but a `tile-post`
/// group's, whose partitions the staged and one-thread kernels offset
/// through its `k` digits (`z·kext` added before them) and whose
/// candidates are the ordered kernel and the workspace split
/// ([`group_order`]: no atomic, no stream-K).
fn radix_whole(site: &TileSite) -> bool {
    site.radix.is_some() && site.post.is_none()
}

/// device-geometry `head-group`: the rows a tile block may span — a batch's
/// (the least stride of `b`'s row digits, which every block-row start then
/// leaves unchanged within the block), else all of them.
pub(crate) fn row_group(site: &TileSite) -> u64 {
    site.radix
        .as_deref()
        .and_then(|r| r.b.as_ref())
        .and_then(|b| b.i.iter().map(|d| d.stride).min())
        .unwrap_or(site.rows)
}

/// device-geometry `tile-post`: `b` walks the fold contiguously (its unit
/// `k` step is one element) and its lanes not — the k-major layout a warp
/// folding one cell across its lanes reads coalesced (`cell-lanes`), where a
/// tile's lanes read it one element a row.
fn b_k_major(site: &TileSite) -> bool {
    let unit = |ds: &[mapal_ir::AxisDigit]| ds.iter().any(|d| d.stride == 1 && d.coef == 1);
    match site.radix.as_deref().and_then(|r| r.b.as_ref()) {
        Some(b) => unit(&b.k) && !unit(&b.lane),
        None => site.b.ck == 1 && site.b.clane != 1,
    }
}

/// device-geometry `tile-post`: the [`CategoryIr::tile_plan_post`] sites a
/// GPU build realizes — each one's epilogue evaluated in its store, which
/// exists exactly where an epilogue group can (`resident.rs:epilogues`): a
/// resident function (no retained site unless `retain-resident`), the
/// `epilogue` rung at the site's fill, the map outside every gated guard
/// arm, an `i32` cell index, numbers read by value that are the host's
/// (none a whole fold makes), a `b` panel shared by more than one row or
/// read along its lanes ([`b_k_major`]), a body the device runs with the host's text,
/// and its operand pointers within one launch's arguments. A site one of
/// these refuses stays the per-cell kernel it was.
pub(crate) fn post_sites(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> Vec<(MorphismId, TileSite)> {
    let rungs = host_rungs(profile, contract, arms);
    if profile.gpu().is_none()
        || !rungs.tile_post
        || !rungs.resident
        || !rungs.epilogue
        || !(rungs.retain_resident
            || super::retain::retained_sites(ir, f, profile, rungs).is_empty())
    {
        return Vec::new();
    }
    let gated: Vec<MorphismId> = ir
        .guard_plan(f)
        .into_iter()
        .filter(mapal_ir::GuardSite::gated)
        .flat_map(|g| g.on_true.own.into_iter().chain(g.on_false.own))
        .collect();
    let mut out = Vec::new();
    // A mixed-radix post site (`head-group` attention) where the staged tile
    // takes it, as `radix-tile`'s sites.
    let radix: Vec<(MorphismId, TileSite)> = match rungs.radix_tile {
        true => ir
            .tile_plan_radix(f)
            .into_iter()
            .filter(|(_, s)| s.post.is_some())
            .filter(|(_, s)| {
                let l = launch(s, profile, contract, arms);
                l.tiled.is_some() || l.mma.is_some()
            })
            .collect(),
        false => Vec::new(),
    };
    let posts = ir.tile_plan_post(f);
    let whole = ir.whole_fold(f);
    // ADR-0043: the narrow-operand sums of products with an epilogue.
    let narrow = match rungs.narrow_tile {
        true => ir.tile_plan_narrow(f),
        false => TilePlan {
            sites: SecondaryMap::new(),
        },
    };
    let narrow_posts = narrow.sites.iter().filter(|(_, s)| s.post.is_some());
    for (m, s) in posts
        .sites
        .iter()
        .chain(radix.iter().map(|(m, s)| (*m, s)))
        .chain(narrow_posts)
    {
        let Some(p) = &s.post else { continue };
        let morph = ir.morphism(m).expect("site resolves");
        let Operation::Map { body, captures } = morph.op else {
            continue;
        };
        let strided = s.a.clane > 1 || s.b.clane > 1;
        // A number the epilogue reads by value is the host's: one a whole
        // fold makes stays in its device buffer (`scalar-fold`), read by the
        // per-cell kernel there — a by-value read would wait on the device
        // each trip.
        let dev_number = p.uniform.iter().any(|&slot| {
            super::pair_source_ir(ir, morph.source, slot)
                .and_then(|o| ir.in_edges(o).first().copied())
                .is_some_and(|q| whole.contains_key(q))
        });
        let index_i32 = super::pair_source_ir(ir, morph.source, captures)
            .and_then(|o| ir.object(o))
            .is_some_and(|o| matches!(&o.ty, Ty::Array { elem, .. } if **elem == Ty::i32()));
        // One row a `b` panel (nothing to share it with) over a k-major `b`:
        // the per-cell kernel's warp a cell reads `b` coalesced and skips the
        // cells a guard masks; the tile would read it a row an element.
        let lone_k_major = row_group(s) == 1 && b_k_major(s);
        if gated.contains(&m)
            || dev_number
            || lone_k_major
            || (strided && !rungs.lane_stride)
            || !index_i32
            || !device_body(ir, body, 0)
            || 3 + p.reads.len() + 1 + p.uniform.len() > LAUNCHV_MAX
            || !gpu_site(s, profile, rungs.window)
            || !launch(s, profile, contract, arms).rungs.epilogue
        {
            continue;
        }
        out.push((m, s.clone()));
    }
    out
}

/// A body the device runs with the host's text, as far as the graph tells:
/// no `Call`, no float `%` (NVPTX's `frem` is not the host's `fmodf`), every
/// nested body the same. The emitter's `device_leaf` adds the function
/// attributes (clean) it alone knows.
fn device_body(ir: &CategoryIr, body: FuncId, depth: u32) -> bool {
    let Some(fd) = ir.func(body) else {
        return false;
    };
    depth < 8
        && fd.morphisms.iter().all(|&g| {
            let Some(x) = ir.morphism(g) else {
                return false;
            };
            let float_mod = x.op == Operation::Mod
                && ir
                    .object(x.source)
                    .and_then(|o| o.ty.component_ty(0).cloned())
                    .is_some_and(|t| is_float(&t));
            !float_mod
                && match x.op {
                    Operation::Call(_) => false,
                    Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                        device_body(ir, body, depth + 1)
                    }
                    _ => true,
                }
        })
}

/// The arithmetic the selected kernel actually emits, distinct from the
/// module's permission to contract or use TF32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arithmetic {
    SeparateMulAdd,
    OrderedFma,
    Tf32Mma,
    Tf32Wgmma,
    /// tensor-cores: `mma.sync.m16n8k16` over a 16-bit member (ADR-0043 D4).
    HalfMma,
    /// f32-tensor-cores `split-tile`: a `{f32}` product on a matrix
    /// instruction by a split form (`mapal_ir::split`), the default face's
    /// bound (ADR-0040 D2).
    SplitMma,
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
            Self::Tf32Mma | Self::Tf32Wgmma | Self::HalfMma | Self::SplitMma => Face::Tf32,
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
    /// gemv-stream `vec-cells`: the adjacent cells (lanes of one row) a
    /// thread of the one-thread-per-cell kernel folds, one 16-byte load of
    /// `b` a `k` step ([`vec_cells`]); 1 everywhere else.
    pub cells: u64,
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
    // ADR-0043: a narrow operand is widened as its panel is copied through
    // registers (the register-staged tile, the one-thread-per-cell kernel);
    // a `cp.async` copy moves bytes unconverted and the tensor-core kernels
    // stage `elem`-wide fragments, so those refuse it (open: 16-bit panels
    // and `mma.m16n8k16`, ADR-0043 slice 2).
    let narrow = site.a.narrow.is_some() || site.b.narrow.is_some();
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    // `head-group`: a block's rows within one batch (its `b` panel's rows):
    // the tile is chosen over one batch's rows.
    let grows = row_group(site);
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
    // tensor-cores (ADR-0043 D4 / D6): a product of 16-bit multiplicands on
    // the part's warp instruction for that format ([`half_operands`]): the
    // `m16n8` tile of the TF32 kernel at `k` 16, its panels at their storage
    // widths. Off (`half-tile`), or where no tile divides the site: the TF32
    // kernel the set may also admit, else the FMA tile.
    if let Some((fmt, ea, eb)) = half_operands(site, gpu)
        && shaped
        && rx.is_none()
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        let bt = site.b.clane != 1;
        let tile = (rungs.mma && rungs.half_tile)
            .then(|| {
                profile.gpu_mma16_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    fmt,
                    ea,
                    eb,
                    bt,
                    rungs.mma_depth,
                    None,
                )
            })
            .flatten();
        // tensor-cores: the 16-bit tile pays where the FMA tile would be
        // bound by its FFMA issue — the site's flops per operand byte at or
        // above the part's balance ([`Gpu::ffma_per_dram_byte_pm`]) — or
        // where its own grid gives every warp scheduler its warps
        // ([`Gpu::fill_warps`]). Below the balance with an under-filled grid
        // the bytes in flight decide, and the FMA tile splits `k` to fill
        // the part (RTX 5080, 16 × 6144 × 4096: 15.9 flop / B against 54.9,
        // 96 one-warp blocks: 50.5 us against the FMA tile's 36.9; 128 rows:
        // 116 flop / B, 768 warps: 110.8 against 174.0 us). Open: the split
        // 16-bit tile.
        let bytes = site.rows * site.k * ea + site.k * site.c * eb + site.rows * site.c * 4;
        let balanced = gpu
            .ffma_per_dram_byte_pm()
            .is_none_or(|b| 2 * site.rows * site.c * site.k * 1000 >= b * bytes);
        let filled = |m: &GpuMmaTile| {
            !rungs.half_fill
                || balanced
                || (site.rows / m.bm()) * (site.c / m.bn()) * m.warps_x * m.warps_y
                    >= gpu.fill_warps()
        };
        if let Some(m) = tile.filter(filled) {
            return Launch {
                arithmetic: Arithmetic::HalfMma,
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
                cells: 1,
            };
        }
    }
    if site.tf32
        && site.elem == Ty::f32()
        && shaped
        && !narrow
        && (vector_a(site) || rx.is_some_and(|x| x.a.is_some()))
        && (vector_b(site) || rx.is_some_and(|x| x.b.is_some()))
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        if rungs.mma && rungs.tf32_tile && radix_ok(rungs) {
            // hopper-wgmma-tma R2: the warp-specialized kernel where both
            // operands are tensor boxes of its panels, else the S53 kernel.
            let wgmma = (rungs.wgmma && rx.is_none())
                .then(|| {
                    rungs
                        .wgmma_ws
                        .then(|| {
                            profile.gpu_wgmma_ws_tile(
                                site.rows,
                                site.c,
                                site.k,
                                rungs.ws_regs,
                                rungs.tma_digits,
                            )
                        })
                        .flatten()
                        .filter(|w| ws_tensors(site, w, gpu, rungs.tma_digits).is_some())
                        .or_else(|| profile.gpu_wgmma_tile(site.rows, site.c, site.k))
                })
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
                    cells: 1,
                };
            }
            let mma = profile.gpu_mma_tile_at(
                grows,
                site.c,
                site.k,
                rungs.wide,
                rungs.fill,
                rungs.mma_depth,
            );
            if let Some(m) = mma {
                return Launch {
                    arithmetic: Arithmetic::Tf32Mma,
                    tiled: None,
                    mma: Some(m),
                    wgmma: None,
                    gx: site.c / m.bn(),
                    gy: site.rows / m.bm(),
                    block: m.threads(gpu.warp),
                    // TMA `bulk-stage`: a transaction barrier a stage.
                    smem: m.smem_bytes()
                        + if bulk_stage(site, &m, gpu, &rungs) {
                            BULK_BAR_BYTES * m.stages
                        } else {
                            0
                        },
                    raster: raster(rungs, m.bm(), m.bn(), m.blocks_per_sm),
                    rungs,
                    combine: Combine::None,
                    cells: 1,
                };
            }
        }
    }
    // f32-tensor-cores R-F `split-tile`: a `{f32}` product on the part's
    // matrix instruction by a split form, where the bound admits it and the
    // rate rule prices it ([`split_operands`], [`split_pays`]).
    let least = site_rungs(profile, wps, Face::Tf32, arms, site).split_least;
    if let Some((kind, fmt, ea, eb)) = split_operands(site, gpu, least)
        && shaped
        && rx.is_none()
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        let bt = site.b.clane != 1;
        let tile = (rungs.mma && rungs.split_tile)
            .then(|| match kind {
                SplitKind::Tf32x3 => profile.gpu_mma_split_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    rungs.mma_depth,
                ),
                _ => profile.gpu_mma16_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    fmt,
                    ea,
                    eb,
                    bt,
                    rungs.mma_depth,
                    Some(split_parts(kind, ea, eb)),
                ),
            })
            .flatten()
            .filter(|m| !bulk_stage(site, m, gpu, &rungs))
            .map(|m| GpuMmaTile {
                split: Some(kind),
                ..m
            })
            .filter(|m| split_pays(site, gpu, kind, m, arms.on.split_tile, rungs.split_issue));
        if let Some(m) = tile {
            return Launch {
                arithmetic: Arithmetic::SplitMma,
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
                cells: 1,
            };
        }
    }
    let arithmetic = Arithmetic::scalar(site, contract, arms.free, arms.pairs);
    let rungs = site_rungs(profile, wps, arithmetic.face(), arms, site);
    // open-rules `wide-fma`, the site's side ([`crate::rules::wide_fma_pays`]):
    // the wide tile's own grid (its split partitions included) against the
    // blocks one wave of it holds. The ON arm takes it anyway.
    let rungs = match rungs.wide_fma && !arms.on.wide_fma && shaped {
        true => {
            let wide =
                profile.gpu_tile_split(grows, site.c, site.k, &site.elem, true, rungs, split);
            let pays = wide.is_none_or(|w| {
                let blocks = (site.rows / w.bm()) * (site.c / w.bn()) * split;
                crate::rules::wide_fma_pays(gpu, blocks, gpu.sms * w.blocks_per_sm.max(1))
            });
            crate::GpuRungs {
                wide_fma: pays,
                ..rungs
            }
        }
        false => rungs,
    };
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
                grows,
                site.c,
                site.k,
                &site.elem,
                rungs.wide_fma,
                tile_rungs,
                split,
            )
        })
        .flatten()
        // device-geometry `head-group`: a row group below every derived
        // tile's rows — one block a group, `tm` its rows.
        .or_else(|| {
            (shaped && grows < site.rows)
                .then(|| profile.gpu_group_tile(grows, site.c, site.k, &site.elem))
                .flatten()
        })
        // S62 `cell-fill`: no register tile divides the site — one output per
        // thread, still staged ([`TargetProfile::gpu_cell_tile`]).
        .or_else(|| {
            (shaped && rungs.cell_fill)
                .then(|| profile.gpu_cell_tile(grows, site.c, site.k, &site.elem))
                .flatten()
        })
        .map(|g| GpuTile {
            kmajor: kmajor && g.tx >= GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES,
            ..g
        })
        .and_then(|g| {
            // ADR-0034 D1: a forced stage count picks the pipeline (1 =
            // register staging, >= 2 = `cp.async`) over the rung.
            let stream = match gpu.geom.stages {
                0 => rungs.cp_async,
                s => s >= 2,
            } && (!narrow || narrow_streams(site, &rungs));
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
                return Some(a);
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
                Some(a) if a.bk >= staged.bk => Some(a),
                // tensor-cores `staged-regs`: a register-staged panel the
                // thread cannot hold beside its tile — every copy's value
                // live from its load to its store, `(bm + bn)·bk / threads`
                // registers — spills (the block decode's narrow 1 × 4096 ×
                // 14336 site: one-row `a` panels whole at `bk` 128 over 32
                // threads, 516 registers, 216–244 B spilled on every rented
                // part). No such tile: the stream where the rules give one,
                // else the next kernel.
                _ if rungs.staged_regs && !staged_fits(gpu, &staged, e) => streamed,
                _ => Some(staged),
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
            (rungs.async_stage
                && gpu.geom.stages == 0
                && !arms.off.cp_async
                && (!narrow || narrow_streams(site, &rungs)))
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
            cells: 1,
        },
        None => {
            // gemv-stream `vec-cells`: the ordered kernel takes the vector
            // cells where their grid alone puts a block on every SM; a split
            // kernel takes them where the split grid does ([`resized`]).
            let v = vec_fit(site, gpu, &rungs);
            let cells = match vec_takes(site, gpu, &rungs, v, 1) {
                true => v,
                false => 1,
            };
            Launch {
                arithmetic,
                tiled: None,
                mma: None,
                wgmma: None,
                // One thread per cell: its own grid, not the fill key's (which
                // keys a tile the launch may not take — a `--geom=` arm).
                gx: cell_blocks(site, cells),
                gy: 1,
                block: NAIVE_THREADS,
                smem: 0,
                raster: 1,
                rungs,
                combine: Combine::None,
                cells,
            }
        }
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
    walk: Option<&str>,
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
    // half-precision (ADR-0043): the operands loaded in a narrow format (intent.py `narrow-tile`).
    let narrow: String = [("a", site.a.narrow), ("b", site.b.narrow)]
        .iter()
        .filter_map(|(o, n)| n.map(|n| format!("{o}:{}", n.name())))
        .collect::<Vec<_>>()
        .join("+");
    if !narrow.is_empty() {
        f += &format!(" narrow={narrow}");
    }
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
        // sweep-attention-2 `kmajor-b` (intent.py): `b`'s copies along its
        // k-major rows.
        if l.rungs.kmajor_b && kvec_b(site, &g) {
            f += " kvb=1";
        }
    }
    if let Some(m) = l.mma {
        f += &format!(
            " mt={} nt={} warps_x={} warps_y={} bk={} stages={} blocks_per_sm={}",
            m.mt, m.nt, m.warps_x, m.warps_y, m.bk, m.stages, m.blocks_per_sm
        );
        // tensor-cores: the instruction and each panel's storage bytes
        // (intent.py `half-tile`); the TF32 line unchanged.
        if m.fmt != MmaFmt::Tf32 {
            f += &format!(
                " op=m16n8k{}.{} ea={} eb={}",
                m.mma_k(),
                m.fmt.ptx(),
                m.ea,
                m.eb
            );
            if m.bt {
                f += " bt=1";
            }
        }
        // TMA `bulk-stage` (intent.py): the panels the bulk-copy engine stages.
        if bulk_stage(site, &m, gpu, &l.rungs) {
            f += " stage=bulk";
        }
        // f32-tensor-cores `split-tile` (intent.py): the split form.
        if let Some(k) = m.split {
            f += &format!(" split={}", split_name(k));
            // tensor-cores-2 `split-issue` (intent.py): the k-steps a zeroed
            // chain carries before its flush, and the issue ratio priced.
            let flush = match l.rungs.split_issue {
                true => split_flush(&m, site.k, combine.splits(), gpu),
                false => 1,
            };
            let (issue_pm, slots_pm) = split_cost(site, gpu, &m, k, l.rungs.split_issue);
            f += &format!(" flush={flush} issue_pm={issue_pm} slots_pm={slots_pm}");
        }
    }
    if let Some(w) = l.wgmma {
        f += &format!(
            " n={} groups={} bk={} stages={} blocks_per_sm={}",
            w.n, w.groups, w.bk, w.stages, w.blocks_per_sm
        );
        // hopper-wgmma-tma (intent.py `wgmma-ws`): the producer warpgroup,
        // the tensor boxes a stage, the registers each side keeps.
        if w.ws {
            let b = if l.rungs.tma_digits {
                1
            } else {
                w.ws_b_boxes()
            };
            f += &format!(" ws=1 stage=tma boxes={}", 1 + b);
            if let Some((p, c)) = w.realloc {
                f += &format!(" regs={p}/{c}");
            }
        }
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
        // long-prefill `uniform-base`: the operands whose pointer the launch
        // moves by a captured number (intent.py `uniform-base`).
        let moved: Vec<&str> = [("a", &r.ua), ("b", &r.ub)]
            .iter()
            .filter(|(_, ds)| !ds.is_empty())
            .map(|(n, _)| *n)
            .collect();
        if !moved.is_empty() {
            f += &format!(" ubase={}", moved.join(","));
        }
    }
    // device-geometry `head-group`: the rows one `b` panel serves (a KV
    // head's query heads), where fewer than the site's (intent.py
    // `head-group`); `tile-post`: the store evaluates the body after the fold.
    if row_group(site) < site.rows {
        f += &format!(" group={}", row_group(site));
    }
    if site.post.is_some() {
        f += " post=1";
    }
    // recurrence-precision `tf32-tile`: the site's product set when it admits
    // TF32 (ADR-0041 D5 `set`, intent.py `tf32-tile`); f32-only lines unchanged.
    if let Some(n) = site.half {
        f += &format!(
            " set={}{}|f32",
            n.name(),
            if site.tf32 { "|tf32" } else { "" }
        );
    } else if site.tf32 {
        f += " set=tf32|f32";
    }
    let (c, s, ctas, iters) = match combine {
        Combine::None => ("none", 1, 0, 0),
        Combine::Workspace { s } => ("splitk", s, 0, 0),
        Combine::Atomic { s } => ("atomic", s, 0, 0),
        Combine::StreamK { ctas, iters } => ("streamk", 1, ctas, iters),
    };
    // sweep-gemm-2: which rule walks a stream-K site (intent.py) — the class
    // that ships `streamk`, `k-fill`'s full tile, or `streamk-price`.
    if let Some(w) = walk {
        f += &format!(" walk={w}");
    }
    // site-geometry `fill-wave`: the target its split grid reaches (intent.py).
    if l.rungs.fill_wave && s > 1 {
        f += &format!(" fill_warps={}", gpu.fill_warps());
    }
    // gemv-stream (intent.py `vec-cells`, `stream-fly`): a streaming
    // one-thread-per-cell kernel's cells a thread, the bytes of `b` its
    // launched grid keeps in flight per SM it uses (`NAIVE_UNROLL` loads of
    // `cells · e_b` a thread), and the part's knee.
    if kind == "naive" && streaming(site, gpu) {
        let grid = l.gx * l.gy * s;
        let fly = grid * NAIVE_THREADS * NAIVE_UNROLL * l.cells * site.b.elem_bytes(&site.elem);
        f += &format!(
            " cells={} fly_sm={} knee={}",
            l.cells,
            fly / grid.min(gpu.sms).max(1),
            gpu.stream_fly_per_sm()
        );
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
const KERNEL_RUNGS: [&str; 46] = [
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
    "streamk-fill",
    "k-fill",
    "vec-cells",
    "stream-fly",
    "bulk-stage",
    "wgmma-ws",
    "ws-regs",
    "tma-digits",
    "kmajor-b",
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
        None => combine(&l, site, profile, contract, order, arms),
    };
    let mut l = resized(site, profile, contract, arms, l, c);
    l.combine = c;
    l
}

/// **large-gemm `k-fill`: an under-filled site keeps the full register tile
/// and fills the part along `k` with stream-K over the slots the tile
/// holds.** Where the rules' launch `l` shrank the tile or its threads to
/// fill the part (`warp-fill`, the fill rule's halving: fewer accumulators
/// a thread or fewer threads than the full S52 tile,
/// [`TargetProfile::gpu_tile_split`] under `full-tile`), the full tile's
/// grid is walked by stream-K at `sms × residency` CTAs — the residency at
/// the tile's own registers and threads ([`TargetProfile::tile_capacity`]),
/// not `grid-share`'s one block an SM, which leaves an SM's second slot
/// empty (128 × 6144 × 4096: 84 CTAs 0.231 ms, 167 CTAs 0.205) — where each
/// CTA folds at least [`K_FILL_PANELS`] panels. A shrunk tile issues more
/// shared loads per FFMA (`4 × 4`: 2 `LDS.128` per 16 FFMA, against 4 per
/// 64 at `8 × 8`), which bounds it at the part's shared-load rate; a CTA's
/// stream-K fixup (the atomic adds of at most two tile pieces) is the cost
/// on the other side, paid back over a long enough run (rtx-5080: 128 × 4096
/// × 4096 0.148 ms against 0.219 shrunk, 128 × 6144 × 4096 0.205 against
/// 0.324; 1024³ at 24 panels a CTA 0.098 against 0.088 shrunk — the shrunk
/// tile stays). Free order only (the fixup adds). The combine and the launch
/// its kernel runs ([`combine`] takes the first, [`resized`] the second).
fn k_fill(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> Option<(Combine, Launch)> {
    let r = l.rungs;
    let gpu = profile.gpu()?;
    let g = l.tiled?;
    let e = elem_bytes_of(&site.elem);
    let atomic = e == 4 || gpu.caps.red_f64;
    // (A whole-file tile is no shrunk one: it is taken only where its own
    // grid fills the part, `streamk-fill` in [`TargetProfile::gpu_tile_split`].)
    if !r.k_fill
        || !r.streamk_fill
        || !atomic
        || g.held > 0
        || arms.splitk > 0
        || gpu.geom.split > 0
    {
        return None;
    }
    // The full S52 tile (its `bm`, `bn`, threads), not the whole-file
    // `wide-fma` tile, which halves a block's warps where the grid is short
    // of them (128 × 4096 × 4096: `8 × 16` at `S = 8` 0.174 ms against
    // `8 × 8` at `S = 4` 0.162).
    let mut a = arms;
    *a.off.rung("wide-fma")? = true;
    let fg = launch_split(site, profile, contract, a, 2, true).tiled?;
    let side = profile.fma_tile_side(&site.elem)?;
    let full_side = fg.tm == side && fg.tn == side;
    if !full_side || fg.tm * fg.tn <= g.tm * g.tn && fg.threads() <= g.threads() || fg.stages >= 2 {
        return None;
    }
    let tiles = (site.rows / fg.bm()) * (site.c / fg.bn());
    let cap = gpu.sms * profile.tile_capacity(&fg, &site.elem);
    if tiles >= cap {
        return None;
    }
    // Sized for `cap` blocks: the panel share of the residency it runs at.
    let full = launch_split(site, profile, contract, a, cap.div_ceil(tiles), true);
    let t = full.tiled?;
    let panels = site.k / t.bk;
    let wave = gpu.sms * t.blocks_per_sm;
    let iters = (tiles * panels).div_ceil(wave);
    let floor = match gpu.geom.kfill {
        0 => K_FILL_PANELS,
        f => f,
    };
    if t.tm * t.tn != fg.tm * fg.tn || t.stages >= 2 || iters < floor || panels < 2 {
        return None;
    }
    let c = Combine::StreamK {
        ctas: (tiles * panels).div_ceil(iters),
        iters,
    };
    // The launch fills the part: the filled class's index width (`idx32`,
    // `contract && filled`) — the walk over a tile count that is no power of
    // two divides in `i32` (128 × 6144 × 4096 spilled 24 B at `i64`).
    let idx32 = rungs_at(profile, GPU_RESIDENT_BLOCKS, Face::of(contract, None), arms).idx32;
    let rungs = crate::GpuRungs {
        idx32,
        ..full.rungs
    };
    Some((
        c,
        Launch {
            combine: c,
            rungs,
            ..full
        },
    ))
}

/// large-gemm `k-fill`: the fewest panels a stream-K CTA of the full tile
/// folds. Between the rtx-5080's measured loser (1024³: 24.4 panels a CTA,
/// 0.098 ms against the shrunk tile's 0.088) and winners (128 × 4096 ×
/// 4096: 48.8, 0.148 against 0.219; 128 × 6144 × 4096: 73.6, 0.205
/// against 0.324): a CTA's fixup — the atomic adds of its two tile pieces,
/// `2 · bm · bn` at the part's per-SM atomic rate — costs about two and a
/// half panels of its FFMA time there, so the run must be long against it.
pub(crate) const K_FILL_PANELS: u64 = 32;

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
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || radix_whole(site);
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
        0 => vec![
            Combine::None,
            combine(l, site, profile, contract, order, arms),
        ],
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
        None => (
            1,
            l.block * l.cells,
            l.cells,
            l.wave(gpu) / gpu.sms,
            1,
            NAIVE_UNROLL,
            1,
            0,
        ),
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
        ea: site.a.elem_bytes(&site.elem),
        eb: site.b.elem_bytes(&site.elem),
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
    // large-gemm `k-fill`: the stream-K [`combine`] took over the full tile
    // runs that tile.
    if let Combine::StreamK { .. } = combine
        && let Some((c, full)) = k_fill(&l, site, profile, contract, arms)
        && c == combine
    {
        return full;
    }
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
        return split_cells(site, profile, rule, s);
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

/// gemv-stream `vec-cells`: a one-thread-per-cell kernel folding `s`
/// partitions takes the vector cells ([`vec_fit`]) where its split grid
/// puts a block on every SM (the ordered kernel's rule, [`launch_split`],
/// at `s` times the blocks); every other launch is `l`.
fn split_cells(site: &TileSite, profile: &TargetProfile, l: Launch, s: u64) -> Launch {
    let naive = l.tiled.is_none() && l.mma.is_none() && l.wgmma.is_none();
    let Some(gpu) = profile.gpu().filter(|_| naive && l.cells == 1 && s > 1) else {
        return l;
    };
    let v = vec_fit(site, gpu, &l.rungs);
    match vec_takes(site, gpu, &l.rungs, v, s) {
        true => Launch {
            cells: v,
            gx: cell_blocks(site, v),
            ..l
        },
        false => l,
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

/// The arms and order an epilogue group's candidates are realized under. A
/// `tile-post` group's fold may split into a workspace, whose reduce launch
/// evaluates the epilogue once every cell's sum is whole (the atomic and
/// stream-K splits are off). Any other group (sweep-runtime-2
/// `streamk-epi`) may take the stream-K walk the rules give its site, whose
/// tile-completing CTA evaluates the members on the whole sums
/// ([`Kernel::ticket`]); its atomic and workspace splits stay off — one
/// launch either way. `--gpu-off=streamk-epi`: the ordered kernel it
/// measured with (S54); so does a group whose walk would lose its panels'
/// L2 reuse ([`epi_walk`], bugs-properties).
pub(crate) fn group_order(
    epi: Option<&EpiKernel>,
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    order: Order,
) -> (crate::GpuArms, Order) {
    epi_order(
        epi.map(|e| e.post.is_some()),
        arms,
        order,
        epi_walk(site, profile, arms),
    )
}

/// bugs-properties `streamk-epi`: whether an epilogue group at `site` may
/// take the stream-K walk — its operands' L2 reuse on the part
/// ([`crate::rules::streamk_epi_pays`]); `--gpu-on=streamk-epi` forces it
/// (the measurement arm).
pub(crate) fn epi_walk(site: &TileSite, profile: &TargetProfile, arms: crate::GpuArms) -> bool {
    let operands = site.rows * site.k * site.a.elem_bytes(&site.elem)
        + site.k * site.c * site.b.elem_bytes(&site.elem);
    arms.on.get("streamk-epi")
        || profile
            .gpu()
            .is_none_or(|g| crate::rules::streamk_epi_pays(g, operands))
}

/// [`group_order`] by the group's shape alone: `post` — `None` without an
/// epilogue group, else whether it carries a `tile-post` body. The planning
/// passes (`horizontal`, `calls`) draw a site's candidates under this too,
/// so the launch they price is the one the emission's table holds.
pub(crate) fn epi_order(
    post: Option<bool>,
    arms: crate::GpuArms,
    order: Order,
    walk: bool,
) -> (crate::GpuArms, Order) {
    match post {
        Some(true) => {
            let mut a = arms;
            *a.off.rung("splitk-atomic").expect("a rung") = true;
            *a.off.rung("streamk").expect("a rung") = true;
            (a, order)
        }
        Some(_) if !arms.off.get("streamk-epi") && walk => {
            let mut a = arms;
            *a.off.rung("splitk-atomic").expect("a rung") = true;
            *a.off.rung("splitk").expect("a rung") = true;
            (a, order)
        }
        Some(_) => (arms, Order::Sequential),
        None => (arms, order),
    }
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
    // The ids predate the host-only `row-block`, `zero-fold`, `row-stage`, `horizontal`, `calls` and
    // `narrow-tile` rungs (off here: no kernel rung): their fields are not part of a kernel's key, so
    // tuned picks and wisdom keep their ids.
    format!("{:?}", Launch { rungs: r, ..*l })
        .replace(", row_block: false", "")
        .replace(", zero_fold: false", "")
        .replace(", row_stage: false", "")
        .replace(", horizontal: false", "")
        .replace(", calls: false", "")
        .replace(", calls_issue: false", "")
        // integration: `post-table` decides the host's launch path only.
        .replace(", post_table: false", "")
        // half-precision: `narrow-tile` decides which sites exist (no kernel text).
        .replace(", narrow_tile: false", "")
        // tensor-cores: `half-tile` / `narrow-async` choose the kernel (no
        // kernel rung), and a TF32 tile keys as it did before its format.
        .replace(", half_tile: false", "")
        .replace(", narrow_async: false", "")
        .replace(", staged_regs: false", "")
        .replace(", half_fill: false", "")
        .replace(", narrow_radix: false", "")
        .replace(", fmt: Tf32, ea: 4, eb: 4, bt: false", "")
        // gemv-stream: the rungs decide `cells` and the split (both keyed);
        // a launch of one cell a thread keys as it did before them.
        .replace(", vec_cells: true", "")
        .replace(", vec_cells: false", "")
        .replace(", stream_fly: true", "")
        .replace(", stream_fly: false", "")
        // regressions: `vec-spread` decides `cells` (keyed).
        .replace(", vec_spread: true", "")
        .replace(", vec_spread: false", "")
        // TMA: `bulk-stage` keys the mma tile's kernel through `smem` (its
        // stages' barriers); a launch it does not stage keys as before.
        .replace(", bulk_stage: true", "")
        .replace(", bulk_stage: false", "")
        // hopper-wgmma-tma: the warp-specialized kernel keys through its
        // tile (`ws`, `realloc`); the S53 kernel and every other launch key
        // as before.
        .replace(", wgmma_ws: true", "")
        .replace(", wgmma_ws: false", "")
        .replace(", ws_regs: true", "")
        .replace(", ws_regs: false", "")
        .replace(", tma_digits: true", "")
        .replace(", tma_digits: false", "")
        .replace(", ws: false, realloc: None", "")
        // sweep-attention: `unit-uniform` decides which units exist (no
        // kernel text of a tile site's).
        .replace(", unit_uniform: false", "")
        // sweep-attention-2: `stream-mma` decides a stream unit's kernel (no
        // tile site's text); `kmajor-b` changes how a candidate copies `b`,
        // not which candidate it is: its id is the arms' one (as candidate
        // 0's is), every launch keys as before.
        .replace(", stream_mma: false", "")
        .replace(", kmajor_b: true", "")
        .replace(", kmajor_b: false", "")
        // sweep-runtime-2: `streamk-epi` decides an epilogue group's
        // combine (keyed through its ticket kernel's id).
        .replace(", streamk_epi: false", "")
        .replace(", cells: 1 }", " }")
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
        // sweep-runtime-2 `streamk-epi`: a group's stream-K kernel evaluates
        // its members at each tile's ticket (its id names the group).
        Combine::StreamK { .. } => match epi {
            Some(e) => (fnv(&format!("{:016x}|ticket|{e:?}", split("streamk"))), 0),
            None => (split("streamk"), 0),
        },
        // `tile-post`: the reduce launch evaluates the epilogue (its id
        // names the group, as an ordered group kernel's does).
        Combine::Workspace { .. } => match epi {
            Some(e) => (
                split("splitk"),
                fnv(&format!("{:016x}|{e:?}", split("reduce"))),
            ),
            None => (split("splitk"), split("reduce")),
        },
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
fn recipe(site: &TileSite, l: &Launch, (id, rid): (u64, u64), epi: bool) -> (String, u64) {
    let cells = site.rows * site.c;
    let bytes_o = cells * elem_bytes_of(&site.elem);
    let (cgx, cblock) = cell_grid(cells);
    let (zero, ws, gx, gy, rgx, rblock) = match l.combine {
        Combine::None => (0, 0, l.gx, l.gy, 0, 0),
        Combine::Atomic { s } => (bytes_o, 0, l.gx, l.gy * s, 0, 0),
        // sweep-runtime-2 `streamk-epi`: the sums and the tiles' tickets in
        // a workspace the runtime zeroes (`ws` without a reduce kernel).
        Combine::StreamK { ctas, .. } if epi => (0, ticket_bytes(site, l), ctas, 1, 0, 0),
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

/// sweep-runtime-2 `streamk-epi`: a group's stream-K workspace — one tile
/// of sums a block (a shared tile's slot is its first block,
/// [`Kernel::ticket_split`]; 16-byte aligned), then one `i32` ticket a slot.
pub(crate) fn ticket_bytes(site: &TileSite, l: &Launch) -> u64 {
    let Combine::StreamK { ctas, .. } = l.combine else {
        unreachable!("a stream-K launch")
    };
    let g = l.tiled.expect("stream-K walks the FMA tile");
    let sums = ctas * g.bm() * g.bn() * elem_bytes_of(&site.elem);
    sums.div_ceil(GPU_VEC_BYTES) * GPU_VEC_BYTES + 4 * ctas
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
    // `tile-post`: candidate 0 of a post group may be a split, whose kernels
    // the plain group emission does not make: every candidate here — and
    // (`streamk-epi`) any group's stream-K candidate 0.
    let first = usize::from(
        !epi.is_some_and(|e| e.post.is_some() || cands[0].launch.combine != Combine::None),
    );
    if !ids[first..]
        .iter()
        .any(|(i, _)| live.contains(&kernel_name(*i)))
    {
        return;
    }
    for (c, &(id, rid)) in cands.iter().zip(&ids).skip(first) {
        let name = kernel_name(id);
        let comb = c.launch.combine;
        // A workspace split's partials are plain sums: the epilogue runs in
        // its reduce launch.
        let main_epi = match comb {
            Combine::Workspace { .. } => None,
            _ => epi,
        };
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(
                &name, site, profile, contract, c.arms, false, main_epi, comb,
            )
        });
        if let Combine::Workspace { s } = comb {
            let red = kernel_name(rid);
            kernels.entry(red.clone()).or_insert_with(|| {
                format!(
                    "; mapal.plan kernel={red} kind=reduce rows={} c={} S={s}{}\n",
                    site.rows,
                    site.c,
                    if epi.is_some() { " epi=1" } else { "" }
                ) + &reduce_kernel_text(&red, site, profile, contract, c.arms, s, epi)
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
        if mapal_ir::bound_coef_unit(site.k / s + s * terms - 1, site.unit()) > bound
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
    k.out = plan_line(name, site, profile, &l, k.idx, c, false, false, None);
    k.out.pop();
    k.out += &format!(" terms={terms}\n");
    match l.tiled {
        Some(g) => k.tiled(name, g),
        None if l.cells > 1 => k.naive_cells(name, l.block, l.cells),
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
/// **f32-tensor-cores R-T1 `mma-combine`: a tensor-core tile's `k` cut into
/// pieces** (ADR-0040 D4 on the `mma.sync` kernels, [`Kernel::mma`] /
/// [`Kernel::mma16`]): the partials of `S` pieces add into the zeroed output
/// (`splitk-atomic`) or a workspace (`splitk`). A forced count (`--splitk=S`,
/// the part's `Geom::split`) where it cuts the panels whole; else the rule
/// [`mma_pieces`]. Free order only (the tensor tiles exist only there); not
/// a mixed-radix site (its digit-affine copies take no piece offset) nor a
/// bulk-staged one (its box copies none either).
fn mma_combine(
    l: &Launch,
    m: &GpuMmaTile,
    site: &TileSite,
    profile: &TargetProfile,
    order: Order,
    arms: crate::GpuArms,
) -> Combine {
    let Some(gpu) = profile.gpu() else {
        return Combine::None;
    };
    let rungs = l.rungs;
    if !rungs.mma_combine
        || order != Order::Free
        || site.radix.is_some()
        || l.raster != 1
        || bulk_stage(site, m, gpu, &rungs)
    {
        return Combine::None;
    }
    let panels = site.k / m.bk;
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    let s = match forced {
        0 => mma_pieces(l, m, site, gpu),
        s if s > 1 && panels.is_multiple_of(s) => s,
        _ => 1,
    };
    // A split form's bound counts the pieces' combines (`S − 1` adds).
    let admitted = |s: u64| match (m.split, gpu.caps.matrix_op(m.fmt, Home::Regs)) {
        (Some(kind), Some(op)) => SplitForm::of(kind)
            .admits(site.k, s, op.shape.k, op.acc_model(), 1)
            .is_some(),
        _ => true,
    };
    let s = if admitted(s) { s } else { 1 };
    let cells = site.rows * site.c;
    match s {
        1 => Combine::None,
        s if rungs.splitk_atomic => Combine::Atomic { s },
        s if rungs.splitk && s * cells * 4 <= gpu.l2_bytes => Combine::Workspace { s },
        _ => Combine::None,
    }
}

/// f32-tensor-cores `mma-combine`: the fewest panels one piece of a split
/// tensor tile folds (the measured pieces: 16–112; none shorter was timed).
pub(crate) const MMA_PIECE_PANELS: u64 = 8;

/// f32-tensor-cores `mma-combine`: the most pieces a tensor tile is cut into
/// (16 lost the m128 cell, 0.1488 against 0.1373 ms).
pub(crate) const MMA_PIECES_MAX: u64 = 8;

/// f32-tensor-cores `mma-combine`: the predicted time a cut must save, per
/// mille — beyond the rtx-5080's null (2048 × 6144 × 4096: predicted 0.97,
/// measured 1.009; 128 × 4096 × 4096 predicted 0.97, measured 0.98 / 1.08 in
/// two sessions) and inside every measured gain (predicted ≤ 0.90).
pub(crate) const MMA_CUT_PM: u64 = 950;

/// **[`mma_combine`]'s rule: the pieces a tensor tile's grid takes** (1:
/// none). A grid of `blocks` tiles on the `slots` the tile holds (`sms ×` its
/// blocks an SM) runs `⌈blocks·S / slots⌉` waves of pieces of `P/S` panels
/// each, and every piece pays a fixed cost: its pipeline's prologue (`stages −
/// 1` panels landing before its first product) and its fixup (the tile's
/// partials added into the output, about one panel) — `stages` panels. The
/// cut `S` (dividing the panels, each piece at least [`MMA_PIECE_PANELS`], at
/// most [`MMA_PIECES_MAX`]) minimizing `⌈blocks·S / slots⌉·(P/S + stages)`
/// panel times plus the cut's own partials and zero pass, in ns at the
/// part's rates ([`MmaCut::ns`], bugs-properties `cut-price`), is taken
/// where it saves more than [`MMA_CUT_PM`] of the uncut time. A
/// grid of at least one wave over a site whose flops per operand byte stay
/// below twice the part's matrix balance (its matrix rate over its DRAM
/// rate: every SM already streams the operands from DRAM) is not cut: the
/// cut only adds its partials' traffic.
///
/// rtx-5080, TF32, ABBA (`f32-tensor-cores.md`): winners 128 × 6144 × 4096
/// (96 blocks on 168 slots, `S` = 8: 0.181 against 0.239 ms), 128 × 4096 ×
/// 14336 (0.411 / 0.458), 2048 × 4096 × 4096 (1.335 / 1.416), 2048 × 4096 ×
/// 14336 (4.43 / 4.90), mlp_2048 / mlp2_2048 (0.945 / 0.940); refused: 128 ×
/// 28672 × 4096 (224 blocks, 61.8 flop / B against 59.2: every cut lost,
/// 0.744–0.764 against 0.696 ms), 2048 × 6144 × 4096 (null), 128 × 4096 ×
/// 4096 (16 panels a piece: mixed).
fn mma_pieces(l: &Launch, m: &GpuMmaTile, site: &TileSite, gpu: &Gpu) -> u64 {
    let blocks = l.gx * l.gy;
    let slots = gpu.sms * m.blocks_per_sm.max(1);
    let panels = site.k / m.bk;
    let rate = gpu
        .caps
        .matrix_op(m.fmt, Home::Regs)
        .map_or(0, |op| gpu.matrix_flops_per_ns(&op));
    let dram = gpu
        .cost
        .map_or(0, |d| d.dram_bytes_per_ns)
        .max(gpu.dram_peak_bytes_per_ns());
    let bytes = site.rows * site.k * m.ea + site.k * site.c * m.eb + site.rows * site.c * 4;
    let flops = 2 * site.rows * site.c * site.k;
    // flops / bytes ≥ 2 · rate / dram, in integers.
    let compute = dram > 0 && rate > 0 && flops * dram >= 2 * rate * bytes;
    if blocks >= slots && !compute {
        return 1;
    }
    let cut = MmaCut {
        blocks,
        slots,
        panels,
        stages: m.stages,
        panel_flops: 2 * m.bm() * m.bn() * m.bk,
        blocks_per_sm: m.blocks_per_sm.max(1),
        cells: site.rows * site.c,
        bytes,
    };
    let time = |s: u64| cut.ns(gpu, rate, s);
    let mut best = (1, time(1));
    let mut s = 2;
    while s <= MMA_PIECES_MAX {
        if panels.is_multiple_of(s) && panels / s >= MMA_PIECE_PANELS && time(s) < best.1 {
            best = (s, time(s));
        }
        s *= 2;
    }
    match best.1 * 1000.0 <= MMA_CUT_PM as f64 * time(1) {
        true => best.0,
        false => 1,
    }
}

/// bugs-properties `cut-price`: **a tensor tile's grid and the price of
/// cutting its `k`** ([`mma_pieces`]) — `blocks` tiles on `slots` resident
/// places, `panels` panels of `panel_flops` each, a pipeline of `stages`,
/// `blocks_per_sm` blocks sharing an SM, `cells` outputs, `bytes` of
/// operands and output (its working set).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MmaCut {
    pub blocks: u64,
    pub slots: u64,
    pub panels: u64,
    pub stages: u64,
    pub panel_flops: u64,
    pub blocks_per_sm: u64,
    pub cells: u64,
    pub bytes: u64,
}

impl MmaCut {
    /// **The cut into `s` pieces, ns**: `⌈blocks·s / slots⌉` waves of
    /// pieces of `panels / s + stages` panels (the prologue and the fixup
    /// about `stages` panels), a panel at one block's share of the SM's
    /// matrix rate (`rate`, the part's flops per ns), and — where it cuts —
    /// the cut's own work: the output's zero pass (`zero_ns` and its bytes at
    /// the fill rate) and `s · cells` partials added by `red` at the part's
    /// `atomic_ops_per_ns` ([`crate::profile::Gpu::plan_cost`]'s rates, as
    /// `streamk_priced` prices a stream-K fixup) — where the cut's working
    /// set spills the L2 the part keeps ([`Gpu::l2_keep_bytes`]); inside it
    /// the adds resolve in the L2 beside the other pieces' products (`red`
    /// is issued and left), overlapped with the waves: the rtx-5080's 2048³
    /// MLP GEMM (48 MiB in its 66 MiB keep) won its cut 1.06× (local ABBA:
    /// 1.568 against 1.667 ms) where the serial price of its 33.5 M adds
    /// (84 µs) refused it. The panel count alone (the
    /// rule before) took every cut whose waves shrank: cross-arch 5fcb401b
    /// measured 9 of 20 such cuts losing (`mma-combine`, finding 5) — the
    /// A100's and 3090's 2048³ MLP GEMMs (`S` 4 / 8: 16.8–33.5 M adds at 248
    /// / 217 an ns, 68–155 µs against 62–68 µs of waves saved), the 4090's
    /// and H100's 128 × 6144 × 4096 (9.6–10.7 µs of partials against 2–10
    /// µs); priced, 18 of the 20 arms pick their measured side.
    pub fn ns(&self, gpu: &Gpu, rate: u64, s: u64) -> f64 {
        let waves = (self.blocks * s).div_ceil(self.slots) as f64;
        let per_sm = rate as f64 / gpu.sms.max(1) as f64;
        let panel = match per_sm > 0.0 {
            true => self.panel_flops as f64 * self.blocks_per_sm as f64 / per_sm,
            false => 1.0,
        };
        let run = waves * ((self.panels / s) as f64 + self.stages as f64) * panel;
        let d = gpu.plan_cost();
        let fill = match (d.fill_bytes_per_ns, d.l2_bytes_per_ns) {
            (0, l2) => l2,
            (f, _) => f,
        };
        let spills = self.bytes > gpu.l2_keep_bytes();
        let fixed = match (s, d.atomic_ops_per_ns, per_sm > 0.0) {
            (1, _, _) | (_, 0, _) | (_, _, false) => 0.0,
            (s, a, true) => {
                let adds = match spills {
                    true => (s * self.cells) as f64 / a as f64,
                    false => 0.0,
                };
                adds + d.zero_ns as f64 + (self.cells * 4) as f64 / fill.max(1) as f64
            }
        };
        run + fixed
    }
}

fn combine(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    order: Order,
    arms: crate::GpuArms,
) -> Combine {
    // A window's taps (`window`) are walked in order: its `k` is not cut.
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || radix_whole(site);
    if order == Order::Sequential || l.wgmma.is_some() || window {
        return Combine::None;
    }
    if let Some(m) = l.mma {
        return mma_combine(l, &m, site, profile, order, arms);
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
    // gemv-stream `stream-fly`: a streaming one-thread-per-cell site is cut
    // until its grid — at the vector cells its split takes ([`vec_fit`]) —
    // puts a block on every SM with the part's in-flight knee in flight on
    // each ([`TargetProfile::split_stream`]): a thread keeps `NAIVE_UNROLL`
    // loads of `cells · e_b` bytes of `b` in flight.
    let naive = l.tiled.is_none();
    let stream = naive && rungs.stream_fly && streaming(site, gpu);
    let count = |workspace| match forced {
        s if s > 0 && panels.is_multiple_of(s) => s,
        _ => {
            let rule = match under {
                true => {
                    profile.split_fill(blocks, warps, cap, panels, site.rows * site.c, e, workspace)
                }
                false => {
                    profile.split_count(blocks, wave, panels, site.rows * site.c, e, workspace)
                }
            };
            match stream {
                true => rule.max(stream_split(
                    site, profile, gpu, &rungs, wave, panels, e, workspace,
                )),
                false => rule,
            }
        }
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
    // large-gemm `k-fill`: an under-filled site's full tile, stream-K over
    // the slots it holds ([`k_fill`]); [`resized`] gives its kernel that tile.
    // (Its own rung: the table's `streamk` is the filled class's; an arm
    // forcing stream-K off turns it off too.)
    if !arms.off.get("streamk")
        && let Some((c, _)) = k_fill(l, site, profile, contract, arms)
    {
        return c;
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
    // large-gemm `streamk-fill`: stream-K spreads the tiles over the slots
    // the launched grid holds (`wave`); a grid that does not give the part
    // `fill_warps` at its own geometry holds one block an SM there (its
    // `grid-share` residency) — 84 one-warp blocks for 16 × 2048 × 7168,
    // 1.41× cuBLAS — so it takes the `fill-wave` split instead.
    // (With `fill-wave` off there is no fill split to take: stream-K as before.)
    let filled_grid =
        !(rungs.streamk_fill && rungs.fill_wave) || blocks * warps >= gpu.fill_warps();
    // sweep-gemm-2 `streamk-price`: where the part prices a stream-K fixup
    // ([`streamk_priced`]: its FMA tile rate, `red` rate and zero pass), an
    // under-filled site (its class ships `streamk` off) streams too when the
    // price says so.
    // (An arm that turns `streamk` off — a group's order, a measurement —
    // stays off.)
    let priced = rungs.streamk_price
        && !arms.off.get("streamk")
        && l.tiled.is_some_and(|g| {
            streamk_priced(profile, site, &g, wave, blocks, wave, 1, 1, 1).is_some()
        });
    let streamk = (rungs.streamk || priced)
        && filled_grid
        && atomic
        && l.tiled.is_some_and(|g| g.stages < 2)
        && panels >= 2
        && blocks * panels > wave;
    if streamk {
        let iters = (blocks * panels).div_ceil(wave);
        let split = (blocks * s_cmp).div_ceil(wave) * (panels / s_cmp);
        let ctas = (blocks * panels).div_ceil(iters);
        let g = l.tiled.expect("a streamed site is a tile");
        // A class that ships `streamk` keeps its panel count (its measured
        // wins, `rung_replay.tsv`); an under-filled site streams by its price.
        let pays = match rungs.streamk {
            true => iters + 1 < split,
            false => {
                streamk_priced(profile, site, &g, wave, blocks, ctas, iters, s_cmp, split)
                    == Some(true)
            }
        };
        if pays {
            return Combine::StreamK { ctas, iters };
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

/// **sweep-gemm-2 `streamk-price`: does stream-K beat the S63 count at this
/// site, its fixed costs priced by the part's rates?** The FMA tile's
/// `flops_per_ns`, `red.global.add.f32`'s `atomic_ops_per_ns` and the zero
/// pass's `zero_ns` where the part measured them, else derived from its
/// properties ([`Gpu::plan_cost`]: review-fixes — the 3090 / A100 / H100
/// profiles measured none of the three, and the panel-count rule this price
/// replaced, measured losing, stood there). `None` only where the element is
/// not `f32` (the rates are f32's) or the part gives no factor to derive
/// them from (no clock): the panel-count rule `iters + 1 < split` there.
///
/// One block's panel takes `t = 2·bm·bn·bk / (flops_per_ns / wave)` — the
/// tile's FMAs at its share of the part's FMA tile rate. Stream-K walks
/// `iters` panels a block, then every piece (`ctas + tiles` of them: a block's
/// walk crosses a tile edge once per tile it touches) adds its `bm·bn`
/// partials (`red`, at `atomic_ops_per_ns` over the device), after the
/// output's zero pass (`zero_ns` + its bytes at the fill rate). The S63 count
/// `s` runs `split` panels a block; when it cuts (`s > 1`) it pays `s·tiles`
/// tiles of partials and the same zero pass. A count without those costs
/// (`iters + 1 < split`, one panel of fixup) lost where pieces are many
/// against panels (rtx-5080: the 2048 × 2048 × 128 scores, 8 panels a tile,
/// 1.41×; the 512³ cells, 1.15–1.24×) and won where a tile holds 64 panels
/// (loopmm's 1024³, 0.95×).
#[allow(clippy::too_many_arguments)]
fn streamk_priced(
    profile: &TargetProfile,
    site: &TileSite,
    g: &GpuTile,
    wave: u64,
    tiles: u64,
    ctas: u64,
    iters: u64,
    s: u64,
    split: u64,
) -> Option<bool> {
    let d = profile.gpu()?.plan_cost();
    let e = elem_bytes_of(&site.elem);
    if e != 4 || d.flops_per_ns == 0 || d.atomic_ops_per_ns == 0 || d.zero_ns == 0 {
        return None;
    }
    let fill = match (d.fill_bytes_per_ns, d.l2_bytes_per_ns) {
        (0, 0) => return None,
        (0, l2) => l2,
        (f, _) => f,
    };
    let tile = (g.bm() * g.bn()) as f64;
    let panel = 2.0 * tile * g.bk as f64 * wave as f64 / d.flops_per_ns as f64;
    let partials = |pieces: u64| pieces as f64 * tile / d.atomic_ops_per_ns as f64;
    let zero = d.zero_ns as f64 + (site.rows * site.c * e) as f64 / fill as f64;
    let streamk = iters as f64 * panel + partials(ctas + tiles) + zero;
    let cut = split as f64 * panel
        + match s {
            1 => 0.0,
            s => partials(s * tiles) + zero,
        };
    Some(streamk < cut)
}

/// The prefetch rung holds the next panel in registers while the current one
/// is computed, so its depth is bounded by registers, not shared memory: one
/// panel's elements per thread, `(bm + bn)·bk / threads`, may take at most
/// `budget` registers — one fragment pair's worth (`tm + tn`, S52), or what
/// the register tile leaves of the file (`wide-fma`, S53). Halve `bk` until it
/// fits; a halved power of two still divides `k` and still loads in whole
/// vectors per thread.
/// tensor-cores `staged-regs`: a register-staged tile's panel — each copy's
/// value held from its load to its store, `(bm + bn)·bk` elements over the
/// block's threads — fits the registers a thread may hold. The panel alone:
/// ptxas shares the tile's fragment registers with the copies it has stored
/// (the B8 projection's 8 × 8 tile, 132 panel + 128 tile registers, compiles
/// without a spill), while a panel past the file spills whatever the tile
/// needs (the block decode's one-row sites, 516: 216–244 B; the 5090's 1 ×
/// 18432 × 7168 GEMV, 516: 2512 B). A cp.async tile holds no panel in
/// registers.
fn staged_fits(gpu: &Gpu, g: &GpuTile, elem_bytes: u64) -> bool {
    g.stages >= 2
        || (g.bm() + g.bn()) * g.bk * elem_bytes.div_ceil(4) / g.threads() <= gpu.regs_per_thread
}

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

/// **sweep-attention-2 `kmajor-b`: a register-staged tile's `b` panel copied
/// along its k-major rows** — a vector of `k` a copy (16 bytes of f32, 8 of
/// a 16-bit format), each key's row read coalesced, scattered into the
/// `[k][n]` panel the fragments read — where `b` is unit-stride along `k`
/// and not along its lanes ([`b_k_major`]), every other term of its address
/// a whole vector (a mixed-radix `b`: one unit `k` digit whose radix the
/// vector divides, its row and lane digits whole vectors), and `bk` whole
/// vectors. Otherwise ([`vector_b`] aside) the copies walk a column one
/// element each: every thread of a warp a different row, a request a row.
fn kvec_b(s: &TileSite, g: &GpuTile) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    if g.stages >= 2 || vector_b(s) || !g.bk.is_multiple_of(lanes) || !b_k_major(s) {
        return false;
    }
    match s.radix.as_deref().and_then(|r| r.b.as_ref()) {
        Some(b) => {
            matches!(b.k[..], [d] if d.stride == 1 && d.coef == 1 && d.radix.is_multiple_of(lanes))
                && aligned(s, b.base)
                && b.lane.iter().chain(&b.i).all(|d| aligned(s, d.coef))
        }
        None => {
            s.b.ck == 1 && aligned(s, s.b.clane) && aligned(s, s.b.base) && s.b.ksplit.is_none()
        }
    }
}

/// gemv-stream: the site's flops per operand byte fall below the part's
/// FFMA balance ([`Gpu::ffma_per_dram_byte_pm`]): its time is the bytes it
/// streams, not its FMAs (a matrix-vector product: `2·c·k` flops over
/// `c·k·e` bytes). Unknown balance: not streaming.
fn streaming(site: &TileSite, gpu: &Gpu) -> bool {
    let bytes = site.rows * site.k * site.a.elem_bytes(&site.elem)
        + site.k * site.c * site.b.elem_bytes(&site.elem)
        + site.rows * site.c * elem_bytes_of(&site.elem);
    gpu.ffma_per_dram_byte_pm()
        .is_some_and(|b| 2 * site.rows * site.c * site.k * 1000 < b * bytes)
}

/// **gemv-stream `vec-cells`: the adjacent cells one 16-byte load of `b`
/// covers** — `16 / e_b` lanes of one row, where the layout facts prove the
/// load: `b` unit-stride along the lane with its base, row and `k` strides
/// whole 16-byte granules of its storage, the row's lanes a whole number of
/// such vectors (so a thread's cells never cross a row and its output cells
/// are one aligned vector too), `a` the same for every lane (one `a` load a
/// `k` step serves every cell), plain affine reads, a streaming site. 1 (one
/// cell a thread) otherwise or with the rung off.
pub(crate) fn vec_fit(site: &TileSite, gpu: &Gpu, rungs: &crate::GpuRungs) -> u64 {
    let eb = site.b.elem_bytes(&site.elem);
    let v = GPU_VEC_BYTES / eb.max(1);
    let whole = |x: u64| (x * eb).is_multiple_of(GPU_VEC_BYTES);
    let fits = rungs.vec_cells
        && v > 1
        && site.radix.is_none()
        && site.a.ksplit.is_none()
        && site.b.ksplit.is_none()
        && site.a.clane == 0
        && site.b.clane == 1
        && whole(site.b.base)
        && whole(site.b.ci)
        && whole(site.b.ck)
        && site.c.is_multiple_of(v)
        && streaming(site, gpu);
    if fits { v } else { 1 }
}

/// Bytes of one `mbarrier` (a stage's transaction barrier).
const BULK_BAR_BYTES: u64 = 8;

/// **TMA `bulk-stage`: the staging rule** — the warp TF32 `mma` tile's
/// panels are copied by the part's bulk-copy engine
/// ([`crate::profile::Caps::bulk`]) where its geometry covers every panel
/// row ([`crate::profile::BulkCopy::box_fits`]: an `a` row of
/// `bk` elements, a `b` row of `bn`, each row's global address and its
/// shared row (`lda`, `ldb` words apart) whole 16-byte granules), a stage's
/// bytes fit one barrier phase, the tile streams (`stages ≥ 2`) and its
/// barriers fit beside the stages. Plain affine reads only (a mixed-radix
/// operand copies element by element).
pub(crate) fn bulk_stage(
    site: &TileSite,
    g: &GpuMmaTile,
    gpu: &Gpu,
    rungs: &crate::GpuRungs,
) -> bool {
    let Some(bc) = gpu.caps.bulk.filter(|_| rungs.bulk_stage) else {
        return false;
    };
    let e = 4;
    let whole = |x: u64| (x * e).is_multiple_of(bc.granule);
    g.fmt == MmaFmt::Tf32
        && !g.bt
        && g.stages >= 2
        && site.radix.is_none()
        && site.a.ck == 1
        && site.b.clane == 1
        && bc.box_fits(1, g.bk, site.a.ci, site.a.base, e, None)
        && bc.box_fits(1, g.bn(), site.b.ck, site.b.base, e, None)
        && whole(g.lda())
        && whole(g.ldb())
        && whole(g.stage_words())
        && (g.bm() * g.bk + g.bk * g.bn()) * e <= bc.tx_bytes
        && g.smem_bytes() + BULK_BAR_BYTES * g.stages <= gpu.smem_per_block
}

/// hopper-wgmma-tma R1 for the warp-specialized `wgmma` kernel: **its two
/// operands as tensor boxes** — `a` as the dense `k × rows` tensor its
/// [`mapal_ir::TileRead::tensor_form`] proves (the `aᵀ` box: `bk` words of
/// `n` rows, the 128-byte swizzle the descriptor reads), `b` as the dense
/// `c × k` tensor (`bk` rows of one 32-word swizzle span, `bn / 32` boxes a
/// stage; under `digits` one 3-d box: the lane axis split into its digits
/// `32·q + r`, `q` outermost at 32 words, which lands those slabs in the same
/// bytes) — every box within the engine's geometry
/// ([`crate::profile::BulkCopy::tensor_box_fits`]). The recipe the runtime
/// encodes each map from (`mapal_gpu.c` `tmap_args`): `[arg, elem bytes,
/// rank, swizzle, base bytes, dims[5], byte strides[4], box[5], 0]` per map.
/// `None`: plain f32 reads only (a narrow, strided-lane, digit or window
/// operand keeps the S53 kernel).
pub(crate) fn ws_tensors(
    site: &TileSite,
    w: &GpuWgmmaTile,
    gpu: &Gpu,
    digits: bool,
) -> Option<[[u64; 20]; 2]> {
    use mapal_ir::TensorAxis as X;
    let bc = gpu.caps.bulk?;
    if site.radix.is_some() || site.a.narrow.is_some() || site.b.narrow.is_some() {
        return None;
    }
    let e = 4u64;
    let span = WGMMA_SWIZZLE_BYTES;
    let a = site.a.tensor_form(site.rows, site.c, site.k)?;
    let b = site.b.tensor_form(site.rows, site.c, site.k)?;
    if a.axes() != [X::K, X::I] || b.axes() != [X::Lane, X::K] {
        return None;
    }
    // The `aᵀ` box `bk` deep: deeper than one swizzle row, its k axis as the
    // digits `k = 32·q + p` (`q` outermost, 32 words apart), which lands the
    // descriptor's atom columns `n·128` bytes apart.
    let r = span / e;
    let (a, asides, aexact) = if w.bk > r {
        let (kd, id) = (a.dims[0], a.dims[1]);
        let split = mapal_ir::TensorForm {
            base: a.base,
            dims: vec![
                mapal_ir::TensorDim { extent: r, ..kd },
                id,
                mapal_ir::TensorDim {
                    extent: kd.extent / r,
                    stride: r,
                    ..kd
                },
            ],
        };
        let exact = kd.extent.is_multiple_of(w.bk) && id.extent.is_multiple_of(w.n);
        (split, vec![r, w.n, w.bk / r], exact)
    } else {
        let (sides, exact) = a.box_of(|x| if x == X::K { w.bk } else { w.n });
        (a, sides, exact)
    };
    let (bsides, bexact) = b.box_of(|x| if x == X::Lane { span / e } else { w.bk });
    let dims = |f: &mapal_ir::TensorForm| {
        f.dims
            .iter()
            .map(|d| (d.extent, d.stride))
            .collect::<Vec<_>>()
    };
    let ok = aexact
        && bexact
        && bc.tensor_box_fits(&dims(&a), &asides, a.base, e, Some(span))
        && bc.tensor_box_fits(&dims(&b), &bsides, b.base, e, Some(span))
        && w.bn().is_multiple_of(span / e);
    if !ok {
        return None;
    }
    // `tma-digits`: the lane axis `c = (span/e) · (c·e/span)` as its two
    // digits, the outer one last (`span` bytes apart): one box of all the
    // stage's slabs.
    let b = match digits {
        true => {
            let (lane, kd) = (b.dims[0], b.dims[1]);
            let r = span / e;
            let split = mapal_ir::TensorForm {
                base: b.base,
                dims: vec![
                    mapal_ir::TensorDim { extent: r, ..lane },
                    kd,
                    mapal_ir::TensorDim {
                        extent: lane.extent / r,
                        stride: r,
                        ..lane
                    },
                ],
            };
            let sides = [r, w.bk, w.bn() / r];
            if !bc.tensor_box_fits(&dims(&split), &sides, split.base, e, Some(span)) {
                return None;
            }
            (split, sides.to_vec())
        }
        false => (b, bsides),
    };
    let (b, bsides) = b;
    let rec = |arg: u64, f: &mapal_ir::TensorForm, sides: &[u64]| {
        let mut r = [0u64; 20];
        r[0] = arg;
        r[1] = e;
        r[2] = f.dims.len() as u64;
        r[3] = span;
        r[4] = f.base * e;
        for (i, d) in f.dims.iter().enumerate() {
            r[5 + i] = d.extent;
            if i > 0 {
                r[9 + i] = d.stride * e;
            }
            r[14 + i] = sides[i];
        }
        r
    };
    Some([rec(0, &a, &asides), rec(1, &b, &bsides)])
}

/// The one-thread-per-cell grid at `cells` cells a thread.
fn cell_blocks(site: &TileSite, cells: u64) -> u64 {
    (site.rows * site.c).div_ceil(NAIVE_THREADS * cells)
}

/// gemv-stream `vec-cells`: the trips of `NAIVE_UNROLL` loads a thread's
/// partition must keep for the vector cells — between the RTX 5080's loser
/// (1 × 1024 × 4096: the 128 partitions its one vector block needs leave 4
/// trips, 0.87× the scalar cells at 32) and its winners (16–128 trips:
/// 1.03–1.42×); at 8 trips null (1 × 4096 × 4096 bf16, 1.03×). A partition
/// that short spends its time in its ramp and its adds, not its stream.
pub(crate) const VEC_MIN_TRIPS: u64 = 8;

/// Whether a one-thread-per-cell kernel folding `s` partitions takes `v`
/// cells a thread: its vector grid puts a block on every SM at `s`, each
/// partition keeps [`VEC_MIN_TRIPS`] trips, and (`vec-spread`) the grid
/// spreads over the SMs as the part's DRAM share asks
/// ([`crate::rules::vec_cells_pay`] against one cell a thread at the same
/// `s`).
fn vec_takes(site: &TileSite, gpu: &Gpu, rungs: &crate::GpuRungs, v: u64, s: u64) -> bool {
    let s = s.max(1);
    v > 1
        && cell_blocks(site, v) * s >= gpu.sms
        && site.k / s >= VEC_MIN_TRIPS * NAIVE_UNROLL
        && (!rungs.vec_spread
            || crate::rules::vec_cells_pay(gpu, cell_blocks(site, v) * s, cell_blocks(site, 1) * s))
}

/// gemv-stream `stream-fly`: the split of a streaming one-thread-per-cell
/// site — [`TargetProfile::split_stream`] on the vector grid where its cells
/// are taken at that split ([`vec_takes`]), else on one cell a thread (a
/// thread keeps `NAIVE_UNROLL` loads of `cells · e_b` bytes of `b` in flight).
#[allow(clippy::too_many_arguments)]
fn stream_split(
    site: &TileSite,
    profile: &TargetProfile,
    gpu: &Gpu,
    rungs: &crate::GpuRungs,
    wave: u64,
    panels: u64,
    e: u64,
    workspace: bool,
) -> u64 {
    let eb = site.b.elem_bytes(&site.elem);
    let at = |v: u64| {
        let fly = NAIVE_THREADS * NAIVE_UNROLL * v * eb;
        profile.split_stream(
            cell_blocks(site, v),
            fly,
            wave,
            panels,
            site.rows * site.c,
            e,
            workspace,
        )
    };
    let v = vec_fit(site, gpu, rungs);
    match v > 1 {
        true => Some(at(v)).filter(|&s| vec_takes(site, gpu, rungs, v, s)),
        false => None,
    }
    .unwrap_or_else(|| at(1))
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
        .replace(", half: None", "")
        .replace(", post: None", "")
}

/// ADR-0041 D6: the reduced realization a site's products admit — `tf32` in
/// `precision_plan.set` of its product ([`TileSite::tf32`]), or a 16-bit
/// member ([`TileSite::half`], ADR-0043 D4, the weaker) — keyed from the
/// program, never a flag (D7).
pub(crate) fn site_reduced(site: &TileSite) -> Option<Reduced> {
    site.half
        .map(Reduced::Half)
        .or(site.tf32.then_some(Reduced::Tf32))
}

/// tensor-cores `narrow-async`: the FMA tile streams a narrow `b` panel in
/// its own format — an 8-byte `cp.async` per 4-element vector (a 2-byte copy
/// has no `cp.async`), widened as each fragment is read from shared memory —
/// where `b` is the one narrow operand, its rows unit-stride and vector
/// aligned, and the rules stream the panel (`async`). A narrow `a` (the
/// transposed panel's scatter) stays register-staged.
fn narrow_streams(site: &TileSite, rungs: &crate::GpuRungs) -> bool {
    rungs.narrow_async
        && site.a.narrow.is_none()
        && site.b.narrow.is_some()
        && vector_b(site)
        && site.radix.is_none()
}

/// **The 16-bit matrix product a site's operands admit** (tensor-cores,
/// ADR-0043 D4): the format and each panel's storage width in shared
/// memory, or `None`.
///
/// - A site whose product admits a 16-bit member ([`TileSite::half`]) takes
///   it where the part lists a warp product of that format whose
///   accumulator meets the set's `u_a = 2⁻²³` ([`profile::MatrixOp::acc_meets`]):
///   a narrow operand of that format is copied as is (2 bytes), an f32
///   operand is rounded once to `bf16` (4 bytes, rounded as its fragment is
///   read). An `f16` member rounds no f32 operand: D4 admits it only where a
///   value-range fact proves the operand below 65504, and none is stated yet.
/// - A `{f32}` product of two operands of one narrow format is exact on the
///   tensor core; it takes the instruction only where its accumulator meets
///   the default face's `u_a = 2⁻²⁴` (every measured part truncates:
///   `mma_calib.cu`, so none does).
///
/// Each panel's rows must be 16-byte copies: unit stride along `k` (`a`;
/// `b` stored `n × k`, [`GpuMmaTile::bt`]) or the lane (`b` stored `k × n`),
/// stride and base whole 16-byte chunks of its storage.
/// f32-tensor-cores `split-tile`: the split form a `{f32}` tile site's
/// operands take, the instruction's format and each panel's storage bytes —
/// `Narrow1` where both are stored in one narrow format (their exact
/// widenings: one exact product a k-step), on that format's warp product
/// with [`half_operands`]' panel layouts; `Tf32x3` where both are f32, on
/// the TF32 warp product with the TF32 tile's 16-byte panels; `Bf16x3n`
/// where one operand is stored `bf16` and the other f32 (tensor-cores-2:
/// bf16 weights against f32 activations) — the f32 one cut into three exact
/// bf16 parts as its fragment is read, the narrow one its exact widening,
/// on the bf16 warp product (its panels at their storage widths, 4 and 2).
/// `None`: a reduced product set (its member's realization), an f32 × `f16`
/// pair (no three-part `f16` cut covers f32's range), or no such
/// instruction.
fn split_operands(
    site: &TileSite,
    gpu: &Gpu,
    least: bool,
) -> Option<(SplitKind, MmaFmt, u64, u64)> {
    if site.elem != Ty::f32() || site.tf32 || site.half.is_some() {
        return None;
    }
    match (site.a.narrow, site.b.narrow) {
        (Some(mapal_ir::NarrowFloat::Bf16), None) | (None, Some(mapal_ir::NarrowFloat::Bf16)) => {
            let op = gpu.caps.matrix_op(MmaFmt::Bf16, Home::Regs)?;
            if op.shape != MMA_SYNC_16 {
                return None;
            }
            let (ea, eb) = mma16_layout(site, MmaFmt::Bf16)?;
            // `split-least`: two parts where the bound covers their split
            // error at this `K`, else the exact three.
            let two = least
                && SplitForm::of(SplitKind::Bf16x2n)
                    .admits(site.k, 1, op.shape.k, op.acc_model(), 1)
                    .is_some();
            let kind = if two {
                SplitKind::Bf16x2n
            } else {
                SplitKind::Bf16x3n
            };
            Some((kind, MmaFmt::Bf16, ea, eb))
        }
        (Some(x), Some(y)) if x == y => {
            let fmt = MmaFmt::of_narrow(x);
            // The split tile issues the 16-bit warp product's geometry
            // (m16n8k16, two-byte fragments): a format whose warp product
            // has another (fp8's m16n8k32) has no split kernel.
            let op = gpu.caps.matrix_op(fmt, Home::Regs)?;
            if op.shape != MMA_SYNC_16 {
                return None;
            }
            let (ea, eb) = mma16_layout(site, fmt)?;
            Some((SplitKind::Narrow1, fmt, ea, eb))
        }
        (None, None) => {
            gpu.caps.matrix_op(MmaFmt::Tf32, Home::Regs)?;
            (vector_a(site) && vector_b(site)).then_some((SplitKind::Tf32x3, MmaFmt::Tf32, 4, 4))
        }
        _ => None,
    }
}

/// tensor-cores-2: **a split tile's flush depth** — the k-steps each
/// tile's zeroed chain carries before its round-to-nearest add into the f32
/// accumulator: the deepest the form's bound admits at the site's `K` and
/// pieces ([`SplitForm::admits`], a power of two), at most one panel's
/// k-steps (the panel loop is unrolled, so a flush falls inside it). 1 off
/// a split form. Each flush is `4·mt·nt` FADDs a thread: at one a k-step
/// they were as many issue slots as the three products' (sm_80 SASS).
pub(crate) fn split_flush(g: &GpuMmaTile, k: u64, pieces: u64, gpu: &Gpu) -> u64 {
    let (Some(kind), Some(op)) = (g.split, gpu.caps.matrix_op(g.fmt, Home::Regs)) else {
        return 1;
    };
    SplitForm::of(kind)
        .admits(k, pieces, op.shape.k, op.acc_model(), g.bk / g.mma_k())
        .unwrap_or(1)
}

/// The least `|x|` bits whose TF32 `rna` rounds to infinity (`2¹²⁸ − 2¹⁰³`).
pub(crate) const TF32_RNA_OVERFLOW: u32 = 0x7f7f_f000;
/// The least `|x|` bits whose bf16 `rn` (ties to even) rounds to infinity.
pub(crate) const BF16_RN_OVERFLOW: u32 = 0x7f7f_8000;

/// The parts of the `a` and `b` fragments a split tile's thread holds: an
/// f32 panel (4-byte elements) is cut into the form's parts, a narrow one is
/// its own exact widening.
pub(crate) fn split_parts(kind: SplitKind, ea: u64, eb: u64) -> (u64, u64) {
    let f = SplitForm::of(kind);
    let p = u64::from(f.parts_a.max(f.parts_b));
    let parts = |e: u64| if e == 4 { p } else { 1 };
    (parts(ea), parts(eb))
}

/// The plan line's name of a split form.
pub(crate) fn split_name(k: SplitKind) -> &'static str {
    match k {
        SplitKind::Tf32x3 => "tf32x3",
        SplitKind::Bf16x6 => "bf16x6",
        SplitKind::Bf16x3n => "bf16x3n",
        SplitKind::Bf16x2n => "bf16x2n",
        SplitKind::Narrow1 => "narrow1",
    }
}

/// **f32-tensor-cores R-F: whether a split tile `m` realizes the site.**
/// 1. *bound* — always: the form's coefficient at the site's `K` meets D2
///    ([`SplitForm::admits`], the accumulator's measured guard bits);
/// 2. *part* — the matrix rate the tile realizes at its instruction issue
///    ([`split_issue`]: its cuts, flushes and products a k-step against the
///    matrix pipe) pays for the form's `P` passes against the FFMA rate
///    ([`crate::rules::split_rate_pays`]);
/// 3. *site* — the flops per operand byte reach the part's FFMA balance
///    (a DRAM-bound site keeps the FFMA tile, which splits `k` to fill the
///    part) or the tile's grid fills the warp schedulers (`half-fill`'s
///    rule).
///
/// `forced` (`--gpu-on=split-tile`): the bound alone (the measurement arm).
fn split_pays(
    site: &TileSite,
    gpu: &Gpu,
    kind: SplitKind,
    m: &GpuMmaTile,
    forced: bool,
    issue: bool,
) -> bool {
    let Some(op) = gpu.caps.matrix_op(m.fmt, Home::Regs) else {
        return false;
    };
    let form = SplitForm::of(kind);
    if form
        .admits(site.k, 1, op.shape.k, op.acc_model(), 1)
        .is_none()
    {
        return false;
    }
    if forced {
        return true;
    }
    if !crate::rules::split_rate_pays(
        gpu,
        &op,
        form.products(),
        split_issue(site, gpu, m, kind, issue),
    ) {
        return false;
    }
    let bytes = site.rows * site.k * m.ea + site.k * site.c * m.eb + site.rows * site.c * 4;
    let balanced = gpu
        .ffma_per_dram_byte_pm()
        .is_none_or(|b| 2 * site.rows * site.c * site.k * 1000 >= b * bytes);
    balanced || (site.rows / m.bm()) * (site.c / m.bn()) * m.warps_x * m.warps_y >= gpu.fill_warps()
}

/// tensor-cores-2 `split-issue`: the split tile `m`'s issue ratio against
/// its matrix pipe ([`crate::rules::split_issue_pm`]) — its thread's cut
/// elements a k-step (an f32 `a` fragment's `k/2` of `m16 × k` a row tile,
/// an f32 `b` fragment's `k/4` a column tile), at the cut's measured slots
/// and the flush the bound admits; `issue` off: the 5fcb401b realization
/// (the `cvt.rna` cut, a flush every k-step).
fn split_issue(site: &TileSite, gpu: &Gpu, m: &GpuMmaTile, kind: SplitKind, issue: bool) -> u64 {
    split_cost(site, gpu, m, kind, issue).0
}

/// [`split_issue`] and the slots a matrix instruction it prices
/// ([`crate::rules::split_slots_pm`]), both per mille.
fn split_cost(
    site: &TileSite,
    gpu: &Gpu,
    m: &GpuMmaTile,
    kind: SplitKind,
    issue: bool,
) -> (u64, u64) {
    let Some(op) = gpu.caps.matrix_op(m.fmt, Home::Regs) else {
        return (u64::MAX, u64::MAX);
    };
    let k = op.shape.k;
    let cut = u64::from(m.ea == 4) * m.mt * k / 2 + u64::from(m.eb == 4) * m.nt * k / 4;
    let per_cut = match (kind, issue) {
        (SplitKind::Tf32x3, true) => crate::rules::SPLIT_CUT_TF32,
        (SplitKind::Tf32x3, false) => crate::rules::SPLIT_CUT_TF32_CVT,
        (SplitKind::Bf16x2n, _) => crate::rules::SPLIT_CUT_BF16X2,
        _ => crate::rules::SPLIT_CUT_BF16X3,
    };
    let flush = match issue {
        true => split_flush(m, site.k, 1, gpu),
        false => 1,
    };
    let products = SplitForm::of(kind).products();
    (
        crate::rules::split_issue_pm(gpu, &op, products, (m.mt, m.nt), cut, per_cut, flush),
        crate::rules::split_slots_pm(products, (m.mt, m.nt), cut, per_cut, flush),
    )
}

/// The 16-bit tile's panel widths for operands of `fmt` ([`half_operands`]'
/// layout rule): each panel's rows whole 16-byte copies.
fn mma16_layout(site: &TileSite, fmt: MmaFmt) -> Option<(u64, u64)> {
    let width = |r: &mapal_ir::TileRead| match r.narrow {
        Some(n) if MmaFmt::of_narrow(n) == fmt => Some(2),
        Some(_) => None,
        None if fmt == MmaFmt::Bf16 => Some(4),
        None => None,
    };
    let (ea, eb) = (width(&site.a)?, width(&site.b)?);
    let lanes = |e: u64| GPU_VEC_BYTES / e;
    let rows_a = site.a.ck == 1
        && site.a.ci.is_multiple_of(lanes(ea))
        && site.a.base.is_multiple_of(lanes(ea));
    let rows_b = site.b.base.is_multiple_of(lanes(eb))
        && match (site.b.clane, site.b.ck) {
            (1, ck) => ck.is_multiple_of(lanes(eb)),
            (cl, 1) => cl.is_multiple_of(lanes(eb)),
            _ => false,
        };
    (rows_a && rows_b).then_some((ea, eb))
}

fn half_operands(site: &TileSite, gpu: &Gpu) -> Option<(MmaFmt, u64, u64)> {
    if site.elem != Ty::f32() {
        return None;
    }
    let (fmt, u_a) = match site.half {
        Some(n) => (MmaFmt::of_narrow(n), -23),
        None => match (site.a.narrow, site.b.narrow) {
            (Some(x), Some(y)) if x == y => (MmaFmt::of_narrow(x), -24),
            _ => return None,
        },
    };
    let op = gpu.caps.matrix_op(fmt, Home::Regs)?;
    if !op.acc_meets(u_a) {
        return None;
    }
    let width = |r: &mapal_ir::TileRead| match r.narrow {
        Some(n) if MmaFmt::of_narrow(n) == fmt => Some(2),
        Some(_) => None,
        None if fmt == MmaFmt::Bf16 => Some(4),
        None => None,
    };
    let (ea, eb) = (width(&site.a)?, width(&site.b)?);
    let lanes = |e: u64| GPU_VEC_BYTES / e;
    let rows_a = site.a.ck == 1
        && site.a.ci.is_multiple_of(lanes(ea))
        && site.a.base.is_multiple_of(lanes(ea));
    let rows_b = site.b.base.is_multiple_of(lanes(eb))
        && match (site.b.clane, site.b.ck) {
            (1, ck) => ck.is_multiple_of(lanes(eb)),
            (cl, 1) => cl.is_multiple_of(lanes(eb)),
            _ => false,
        };
    (rows_a && rows_b).then_some((fmt, ea, eb))
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
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EpiKernel {
    /// The site's own output is stored (a reader outside the group).
    pub store_out: bool,
    /// Device pointers to complete operands, after `a, b, out` (`%x<i>`).
    pub caps: usize,
    pub calls: Vec<EpiCall>,
    /// device-geometry `tile-post`: the site's own body after its fold,
    /// `ret @callee(arg, acc)` — its result is group value 0.
    pub post: Option<EpiCall>,
    /// `tile-post`: the LLVM types of the numbers the epilogue reads by
    /// value (`%u<j>`, 8-byte arguments after the stored pointers).
    pub uniforms: Vec<String>,
}

// The kernel id hashes this text: a group without a post keeps the id it
// had before `tile-post` existed.
impl std::fmt::Debug for EpiKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("EpiKernel");
        d.field("store_out", &self.store_out)
            .field("caps", &self.caps)
            .field("calls", &self.calls);
        if let Some(p) = &self.post {
            d.field("post", p);
        }
        if !self.uniforms.is_empty() {
            d.field("uniforms", &self.uniforms);
        }
        d.finish()
    }
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
    /// `tile-post`: an array only the site's fold reads — the epilogue
    /// variant never touches it (`ptr null`).
    Null,
    /// `tile-post`: uniform `j` (`%uv<j>`), by value.
    Uniform(usize),
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

/// `tile-post`'s epilogue variant of map body `name` (`FnEmit::post_fold`).
pub(crate) fn post_variant(name: &str) -> String {
    format!("{name}_post")
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
    // `piece-delta`: cell `k` of the range is `(k ÷ w)·step + k % w` past `rs`.
    let offset = |step: u64| match g.piece {
        0 | 1 => format!("  %ks = mul i64 %k, {step}\n  %i = add i64 %rs, %ks\n"),
        w => format!(
            "  %kq = udiv i64 %k, {w}\n  %kr = urem i64 %k, {w}\n  %kqs = mul i64 %kq, {step}\n  \
             %ks = add i64 %kqs, %kr\n  %i = add i64 %rs, %ks\n"
        ),
    };
    let (gid, at) = match (g.lanes, g.range) {
        (1, None) => ("%i".to_string(), String::new()),
        (1, Some(step)) => ("%k".to_string(), offset(step)),
        (l, None) => ("%g".to_string(), format!("  %i = udiv i64 %g, {l}\n")),
        (l, Some(step)) => (
            "%g".to_string(),
            format!("  %k = udiv i64 %g, {l}\n{}", offset(step)),
        ),
    };
    // Lanes: the bound test is on the cell, after the division.
    // `retain-cell` (`clip`): the cell itself is tested against the extent
    // too, so its index is formed first.
    let (head, guard) = if g.clip && g.range.is_some() {
        (
            at.clone(),
            format!(
                "  %in0 = icmp ult i64 {k}, {bound}\n  %inc = icmp ult i64 %i, {n}\n  %in = and i1 %in0, %inc\n"
            ),
        )
    } else if g.lanes == 1 {
        (
            String::new(),
            format!("  %in = icmp ult i64 {k}, {bound}\n"),
        )
    } else {
        (at.clone(), format!("  %in = icmp ult i64 {k}, {bound}\n"))
    };
    let at = if g.lanes == 1 && !(g.clip && g.range.is_some()) {
        at
    } else {
        String::new()
    };
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
    /// `piece-delta`: each `step` position is a piece of this many cells
    /// (0 or 1: one cell). Thread `k` is cell `rs + (k ÷ w)·step + k % w`.
    pub piece: u64,
    /// `cell-lanes`: threads a cell (1: one thread a cell).
    pub lanes: u64,
    /// kernel-merging `zero-fold`: the later sites' outputs this kernel
    /// zeroes ([`super::zero::zero_tail`]).
    pub zero: usize,
    /// caching-memory `retain-cell`: a range cell past the extent stores
    /// nothing (a delta's window may reach past the array's end).
    pub clip: bool,
}

impl CellGeo {
    #[cfg(test)]
    pub(crate) const ONE: CellGeo = CellGeo {
        range: None,
        piece: 0,
        lanes: 1,
        zero: 0,
        clip: false,
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
        let dev = placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
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
                        ) + &reduce_kernel_text(&red, site, profile, contract, arms, s, None)
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
        // An epilogue group's kernel is ordered: its candidates are too — a
        // `tile-post` group's may split into a workspace (`group_order`).
        let (garms, gorder) = group_order(Some(&e.k), &e.site, profile, arms, order);
        cand_kernels(
            &mut kernels,
            &mut site_lines,
            live,
            &e.site,
            profile,
            contract,
            garms,
            gorder,
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
    // long-rows `row-slices`: a sliced unit's grid barrier, where no
    // persistent kernel brought it.
    if out.contains("call void @mapal_gsync(")
        && !out.contains("define internal void @mapal_gsync(")
    {
        let decls = super::calls::missing_decls(&out);
        out.push('\n');
        out.push_str(&decls);
        out.push_str(super::calls::GSYNC);
    }
    // ADR-0042: the device's `exp` definitions, for the widths a body calls —
    // the hardware form where the part's measured `ex2` error meets the bound.
    {
        let hw = host_rungs(profile, contract, arms).exp_hw;
        let ex2 = profile.gpu().map(|g| g.caps.ex2);
        let real = crate::exp::choose(order == Order::Free, contract, hw, ex2);
        let defs = crate::exp::defs(&out, real);
        out.push_str(&defs);
    }
    // ADR-0043: the device's narrow roundings (`cvt.rn`).
    let caps = profile.gpu().map(|g| g.caps);
    let mut ptx = caps.map(|c| c.ptx);
    {
        let cvt_f8 = caps.and_then(|c| c.cvt_f8);
        let (defs, hw) = crate::narrow::defs(&out, true, cvt_f8.is_some());
        out.push_str(&defs);
        // bugs-properties `ptx-isa`: the conversions' encoding on this part.
        if hw {
            ptx = ptx.max(cvt_f8);
        }
    }
    // `pdl`: the runtime launches every kernel behind a kernel as a dependent
    // launch, so every kernel waits for its producer's grid before it reads.
    if pdl(ir, profile, contract, arms) {
        out = pdl_wait_every_kernel(&out);
    } else if pdl_part(profile, contract, arms) {
        out = strip_pdl(&out);
    }
    // A `wgmma` kernel's swizzle atoms, and every swizzled tensor box,
    // start on 1024-byte boundaries.
    if out.contains("wgmma.mma_async") || out.contains("cp.async.bulk.tensor") {
        out = out.replacen("[0 x i8], align 16", "[0 x i8], align 1024", 1);
    }
    // bugs-properties `ptx-isa`: the module declares the largest encoding its
    // target and its instructions need — every `llc` step passes it.
    if let Some(p) = ptx {
        out = out.replacen(
            "; mapal-backend-llvm device module\n",
            &format!("; mapal-backend-llvm device module\n{}", p.declare()),
            1,
        );
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
    debug_assert!(
        combine == Combine::None
            || (!col && (epi.is_none() || matches!(combine, Combine::StreamK { .. })))
    );
    let walk = match combine {
        Combine::StreamK { .. } if l.rungs.streamk => Some("class"),
        Combine::StreamK { .. } => Some(
            match k_fill(
                &launch(site, profile, contract, arms),
                site,
                profile,
                contract,
                arms,
            ) {
                Some((c, _)) if c == combine => "fill",
                _ => "price",
            },
        ),
        _ => None,
    };
    k.out = plan_line(
        name,
        site,
        profile,
        &l,
        k.idx,
        combine,
        col,
        epi.is_some(),
        walk,
    );
    // plan-cost: the site's pick on every kernel's line (D3); its candidates
    // on every row-major one (the harness keys them by `site=`).
    let pick = plan_pick(site, profile, contract, arms, !col);
    if !pick.is_empty() {
        k.out.pop();
        k.out += &pick;
        k.out.push('\n');
    }
    match (l.wgmma, l.mma, l.tiled) {
        (Some(w), _, _) if w.ws => k.wgmma_ws(name, w),
        (Some(w), _, _) => k.wgmma(name, w),
        (None, Some(m), _) if m.fmt != MmaFmt::Tf32 => k.mma16(name, m),
        (None, Some(m), _) => k.mma(name, m),
        (None, None, Some(g)) => k.tiled(name, g),
        (None, None, None) if l.cells > 1 => k.naive_cells(name, l.block, l.cells),
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
    epi: Option<&EpiKernel>,
) -> String {
    let mut k = kernel(
        site,
        profile,
        contract,
        arms,
        false,
        epi,
        Combine::Workspace { s },
    );
    let (t, ix, cells) = (k.t.clone(), k.idx, site.rows * site.c);
    let e = elem_bytes_of(&site.elem);
    // `tile-post`: the header takes the group's pointers; the whole sum goes
    // through its store.
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
    if epi.is_some() {
        k.store_cells(std::slice::from_ref(&acc), &cell);
    } else {
        let p = k.gep(1, "%o", &cell);
        k.line(format!("store {t} {acc}, ptr addrspace(1) {p}, align {e}"));
    }
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
        pdl: pdl_part(profile, contract, arms),
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
        ticket: epi.is_some() && matches!(combine, Combine::StreamK { .. }),
        part: false,
        p0: None,
        head: "entry",
        tmap: Vec::new(),
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
    let cells = (site.rows * site.c).max(l.gx * l.gy * l.block * l.cells) * slabs;
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
fn lane_maps(s: &TileSite, g: &GpuTile, kmajor: bool, kvb: bool) -> [(LaneMap, u64); 2] {
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
    } else if kvb {
        // `kmajor-b`: the copies walk each key's row (`(n, k)` coordinates).
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bn * kv / threads)
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
    let kvb = rungs.kmajor_b && kvec_b(s, g);
    let [(_, a), (_, b)] = lane_maps(s, g, kmaj, kvb);
    // `kmajor-b`: each `b` copy scatters `lanes` stores.
    let bw = if kvb { lanes } else { 1 };
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
        false if vector_a(s) && !kmaj => a + b + a * lanes * sa + b * bw * sb,
        false => a + b + a * sa + b * bw * sb,
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
    let kvb = rungs.kmajor_b && kvec_b(s, g);
    let [(am, an), (bm, bn)] = lane_maps(s, g, kmaj, kvb);
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
    let st_b = match kvb {
        // `kmajor-b`: a vector along `k` scatters to `lanes` panel rows.
        true => worst(bn * lanes, &|t, p| {
            let (nc, kc) = rc(bm, t, p / lanes);
            ((kc + p % lanes) * g.bn() + nc, 1)
        }),
        false => worst(bn, &|t, p| {
            let (kr, nc) = rc(bm, t, p);
            (kr * g.bn() + nc, bw)
        }),
    };
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
    /// sweep-runtime-2 `streamk-epi`: a stream-K kernel of an epilogue group
    /// — its partials add into the workspace `%w` (the sums, then one `i32`
    /// ticket a tile), and the CTA that completes a tile evaluates the
    /// group's members on its whole sums ([`Kernel::ticket`]).
    ticket: bool,
    /// sweep-runtime-2 `streamk-epi`: the store pass of a shared tile's piece
    /// (its partials into the workspace slot), not a whole tile's.
    part: bool,
    /// S63 `streamk`: the current piece's first panel (the seed rides the
    /// piece that starts its tile, [`Kernel::split_seed`]).
    p0: Option<String>,
    /// The block the panel loop is entered from: `entry`, or `seg` in a
    /// stream-K kernel (one panel loop per piece).
    head: &'static str,
    /// hopper-wgmma-tma: the tensor maps this kernel takes by value after
    /// its pointers ([`ws_tensors`]'s recipes), carried to the runtime as
    /// the global `<kernel>_tmap`.
    tmap: Vec<[u64; 20]>,
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
        for j in 0..epi.uniforms.len() {
            s.push_str(&format!(", i64 %u{j}"));
        }
        // sweep-runtime-2 `streamk-epi`: the workspace the runtime appends
        // after the table's pointers (the sums, then the tiles' tickets).
        if self.ticket {
            s.push_str(&format!(
                ", ptr addrspace(1) noalias align {GPU_VEC_BYTES} %w"
            ));
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
        for (j, t) in epi.uniforms.iter().enumerate() {
            let conv = uniform_from_i64(&format!("%uv{j}"), &format!("%u{j}"), t);
            self.out.push_str(&conv);
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
        // `tile-post`: value 0 is the site's body after its fold.
        let acc = match &epi.post {
            Some(p) => {
                let mut agg = "poison".to_string();
                for (f, a) in p.fields.iter().enumerate() {
                    let (ty, v) = match *a {
                        EpiArg::Cap(i) => ("ptr".to_string(), format!("%xg{i}")),
                        EpiArg::Index => ("i32".to_string(), t32.clone()),
                        EpiArg::Null => ("ptr".to_string(), "null".to_string()),
                        EpiArg::Uniform(j) => (epi.uniforms[j].clone(), format!("%uv{j}")),
                        EpiArg::Value(_) | EpiArg::Cell(_) => {
                            unreachable!("an epilogue body reads no group value")
                        }
                    };
                    let n = self.tmp();
                    self.line(format!("{n} = insertvalue {} {agg}, {ty} {v}, {f}", p.arg));
                    agg = n;
                }
                let r = self.tmp();
                self.line(format!(
                    "{r} = call {} @{}({} {agg}, {} {acc})",
                    p.ret, p.callee, p.arg, self.t
                ));
                r
            }
            None => acc.to_string(),
        };
        let mut vals = vec![acc];
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
                        EpiArg::Null => ("ptr".to_string(), "null".to_string()),
                        EpiArg::Uniform(j) => (epi.uniforms[j].clone(), format!("%uv{j}")),
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
        vals
    }

    /// Store `accs`, the accumulators of the consecutive output cells from
    /// index `oi`, through the epilogue: the site's output when stored, and
    /// each stored member's result — one vector store per output when
    /// `accs.len() > 1`, as the plain store.
    fn store_cells(&mut self, accs: &[String], oi: &str) {
        let epi = self.epi.expect("an epilogue");
        // Value 0 (the site's: its accumulator, through a `tile-post` body),
        // then each call's.
        let mut per: Vec<Vec<String>> = vec![Vec::new(); epi.calls.len() + 1];
        for (j, acc) in accs.iter().enumerate() {
            let cell = self.plus(oi.to_string(), j as u64);
            for (k, v) in self.epi_values(acc, &cell).into_iter().enumerate() {
                per[k].push(v);
            }
        }
        let mut outs: Vec<(String, String, Vec<String>)> = Vec::new();
        if epi.store_out {
            outs.push((
                "%o".to_string(),
                self.t.clone(),
                std::mem::take(&mut per[0]),
            ));
        }
        let mut y = 0;
        for (k, c) in epi.calls.iter().enumerate() {
            if c.store {
                outs.push((
                    format!("%y{y}"),
                    c.ret.clone(),
                    std::mem::take(&mut per[k + 1]),
                ));
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
        let sums = if self.ticket { "%w" } else { "%o" };
        for (j, x) in v.iter().enumerate() {
            let i = self.plus(oi.to_string(), j as u64);
            let p = self.gep(1, sums, &i);
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
        // ADR-0043: a narrow operand's pointer is an allocation's (the
        // arena's `ARENA_ALIGN`, a device allocation's) like any operand's;
        // a launch that moves it by a captured number (`uniform-base`) moves
        // it by whole register-staged vectors of its format
        // ([`uniform_aligned`]: `lanes` elements, 8 bytes of bf16). Its
        // copies' adjacent loads then form 16-byte loads (cross-arch
        // 3a8e654f: the bf16 KV cache's scores kernel issued 2-byte loads
        // under `align 2` — 4× the f32 cache's load instructions, 2.6× its
        // time on the H100).
        let lanes = GPU_VEC_BYTES / elem_bytes_of(&self.site.elem);
        let narrow_align = |r: &mapal_ir::TileRead, moved: bool| {
            r.narrow.map_or(align, |n| match moved {
                true => align.min(lanes * n.bytes()),
                false => align,
            })
        };
        let moved = |a: bool| {
            self.site
                .radix
                .as_deref()
                .is_some_and(|x| !if a { &x.ua } else { &x.ub }.is_empty())
        };
        let (aa, ab_) = (
            narrow_align(&self.site.a, moved(true)),
            narrow_align(&self.site.b, moved(false)),
        );
        let mut more = String::new();
        for t in 1..self.terms {
            let (a, b) = ab(t);
            more += &format!(
                ", ptr addrspace(1) readonly align {aa} {a}, ptr addrspace(1) readonly align {ab_} {b}"
            );
        }
        let ro = if self.terms < 2 {
            "noalias readonly"
        } else {
            "readonly"
        };
        // hopper-wgmma-tma: each tensor map a 128-byte parameter by value.
        for m in 0..self.tmap.len() {
            more += &format!(", ptr byval([128 x i8]) align 64 \"nvvm.grid_constant\" %tm{m}");
        }
        self.out.push_str(&format!(
            "define ptx_kernel void @{name}(\
             ptr addrspace(1) {ro} align {aa} {a0}, \
             ptr addrspace(1) {ro} align {ab_} {b0}, \
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
            // cross-arch: the dependent launches at this kernel's exit unless
            // `pdl-early` (its blocks would wait resident beside this grid's).
            if self.rungs.pdl_early {
                self.asm("griddepcontrol.launch_dependents;");
            }
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
                // A split partition's `k` starts `z·kc` along the fold.
                _ if m.is_some() => {
                    let kz = k.at_z(kv.clone(), kc);
                    k.digits(&m.expect("radix").k, &kz)
                }
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
            // ADR-0043: a narrow operand loads its format and widens (exact).
            if let Some(n) = r.narrow {
                let nt = crate::narrow::llvm_ty(n);
                let p = k.gep_ty(1, nt, arr, &idx);
                let (w, v) = (k.tmp(), k.tmp());
                k.line(format!(
                    "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
                    n.bytes()
                ));
                let tt = if n.bytes() == 1 {
                    k.tmp()
                } else {
                    String::new()
                };
                for l in crate::narrow::widen_lines(n, nt, &w, &t, &v, &tt) {
                    k.line(l);
                }
                return v;
            }
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

    /// **gemv-stream `vec-cells`: `v` adjacent cells a thread** ([`vec_fit`])
    /// — the one-thread-per-cell kernel with a thread's `v` lanes of one row:
    /// each `k` step loads `a` once and `b`'s `v` lanes as one 16-byte vector
    /// (`ld.global.nc.v4`), its `v` cells each folding in `k` order (the
    /// oracle's chain per cell, unchanged). A trip issues the loads of `u`
    /// steps before their FMAs (`u` = [`NAIVE_UNROLL`], halved until it
    /// divides the partition): the bytes in flight the split was sized for
    /// ([`TargetProfile::split_stream`]). Splits, terms, the workspace slab,
    /// atomics, column-major stores and epilogues as [`Self::naive`].
    fn naive_cells(&mut self, name: &str, threads: u64, v: u64) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        self.header(name, threads, e, 0);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        if self.splitk > 1 {
            let y = self.sreg("ctaid.y");
            self.z = Some(match self.terms {
                1 => y,
                _ => self.op("urem", &y, self.splitk),
            });
        }
        let th0 = self.op("mul", &bx, threads);
        let th = self.op("add", &th0, &tid);
        let cell = self.op("mul", &th, v);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult {ix} {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let kc = s.k / self.splitk;
        // `a` ignores the lane (`clane` 0); `b`'s lanes are unit-stride.
        let ai = self.op("mul", &i, s.a.ci);
        let abase = self.plus(ai, s.a.base);
        let abase = self.at_z(abase, kc * s.a.ck);
        let bi = self.op("mul", &i, s.b.ci);
        let bl = self.op("add", &bi, &lane);
        let bbase = self.plus(bl, s.b.base);
        let bbase = self.at_z(bbase, kc * s.b.ck);
        let seed = self.split_seed(&seed);
        let mut u = NAIVE_UNROLL;
        while u > 1 && !kc.is_multiple_of(u) {
            u /= 2;
        }
        self.line("br label %kloop");
        self.out.push_str("kloop:\n");
        let (kv, kn) = (self.tmp(), self.tmp());
        self.line(format!("{kv} = phi {ix} [0, %body], [{kn}, %kloop]"));
        let accs: Vec<String> = (0..v).map(|_| self.tmp()).collect();
        for (j, acc) in accs.iter().enumerate() {
            self.line(format!(
                "{acc} = phi {t} [{seed}, %body], [@ACC_NEXT_{j}@, %kloop]"
            ));
        }
        // The trip's loads first: `u` of `a`, `u` vectors of `b`.
        let mut steps = Vec::new();
        for q in 0..u {
            let kq = self.plus(kv.clone(), q);
            let ao = self.op("mul", &kq, s.a.ck);
            let aidx = self.op("add", &abase, &ao);
            let av = match s.a.narrow {
                Some(n) => {
                    let nt = crate::narrow::llvm_ty(n);
                    let p = self.gep_ty(1, nt, "%a", &aidx);
                    let (w, x) = (self.tmp(), self.tmp());
                    self.line(format!(
                        "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
                        n.bytes()
                    ));
                    let tt = if n.bytes() == 1 {
                        self.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(n, nt, &w, &t, &x, &tt) {
                        self.line(l);
                    }
                    x
                }
                None => {
                    let p = self.gep(1, "%a", &aidx);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr addrspace(1) {p}, align {e}"));
                    x
                }
            };
            let bo = self.op("mul", &kq, s.b.ck);
            let bidx = self.op("add", &bbase, &bo);
            let vt = format!("<{v} x {t}>");
            let bv = match s.b.narrow {
                Some(n) => {
                    let nt = crate::narrow::llvm_ty(n);
                    let p = self.gep_ty(1, nt, "%b", &bidx);
                    let (w, x) = (self.tmp(), self.tmp());
                    self.line(format!(
                        "{w} = load <{v} x {nt}>, ptr addrspace(1) {p}, align {GPU_VEC_BYTES}"
                    ));
                    let nvt = format!("<{v} x {nt}>");
                    let tt = if n.bytes() == 1 {
                        self.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(n, &nvt, &w, &vt, &x, &tt) {
                        self.line(l);
                    }
                    x
                }
                None => {
                    let p = self.gep(1, "%b", &bidx);
                    let x = self.tmp();
                    self.line(format!(
                        "{x} = load {vt}, ptr addrspace(1) {p}, align {GPU_VEC_BYTES}"
                    ));
                    x
                }
            };
            steps.push((av, bv));
        }
        // Then each step's FMAs, `k` in order for every cell.
        let mut cur = accs.clone();
        for (av, bv) in steps {
            for (j, acc) in cur.iter_mut().enumerate() {
                let bj = self.tmp();
                self.line(format!("{bj} = extractelement <{v} x {t}> {bv}, i64 {j}"));
                *acc = self.step(acc, &av, &bj);
            }
        }
        for (j, next) in cur.iter().enumerate() {
            self.out = self.out.replace(&format!("@ACC_NEXT_{j}@"), next);
        }
        self.line(format!("{kn} = add {ix} {kv}, {u}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq {ix} {kn}, {kc}"));
        self.line(format!("br i1 {fin}, label %store, label %kloop"));
        self.out.push_str("store:\n");
        if self.col {
            // Column-major: each cell its own word (`o[j·rows + i]`).
            for (j, acc) in cur.iter().enumerate() {
                let lj = self.plus(lane.clone(), j as u64);
                let jr = self.op("mul", &lj, s.rows);
                let oc = self.op("add", &jr, &i);
                let oc = self.slab(oc);
                match self.epi.is_some() {
                    true => self.store_cells(std::slice::from_ref(acc), &oc),
                    false => self.store_o(std::slice::from_ref(acc), &oc),
                }
            }
        } else {
            // Whole 16-byte vectors of every stored output's type.
            let widest = self.epi.map_or(e, |ep| {
                ep.calls
                    .iter()
                    .filter(|c| c.store)
                    .map(|c| if c.ret == "double" { 8 } else { 4 })
                    .fold(e, u64::max)
            });
            let chunk = (GPU_VEC_BYTES / widest).clamp(1, v) as usize;
            let oc = self.slab(cell);
            for (n, part) in cur.chunks(chunk).enumerate() {
                let at = self.plus(oc.clone(), (n * chunk) as u64);
                match self.epi.is_some() {
                    true => self.store_cells(part, &at),
                    false => self.store_o(part, &at),
                }
            }
        }
        self.line("br label %done");
        self.out.push_str("done:\n  ret void\n}\n");
    }

    /// The 32-bit shared address `off` bytes past the barriers at `bars` (a
    /// shared pointer): each address its own cast of a pointer (the
    /// emulator maps it back, as `ldmatrix`'s).
    fn bar_at(&mut self, bars: &str, off: &str) -> String {
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds i8, ptr addrspace(3) {bars}, i64 {off}"
        ));
        let a = self.tmp();
        self.line(format!("{a} = ptrtoint ptr addrspace(3) {p} to i32"));
        a
    }

    /// An inline-asm statement with operands (`args`: typed values).
    fn asm_args(&mut self, text: &str, cons: &str, args: &[String]) {
        self.line(format!(
            "call void asm sideeffect \"{text}\", \"{cons},~{{memory}}\"({})",
            args.join(", ")
        ));
    }

    /// **TMA `bulk-stage`: panel `tile` into stage `stage`** (i64 values) of
    /// the TF32 `mma` tile by the bulk-copy engine — every row of the `a`
    /// panel (`bk` elements) and the `b` panel (`bn`) one `cp.async.bulk`
    /// into its shared row (`lda` / `ldb` words apart), warp 0's lane `l`
    /// copying rows `l, l + 32, …` (`a`'s `bm` rows, then `b`'s `bk`); lane 0
    /// first adds the stage's bytes to its barrier (`bars + 8·stage`,
    /// `mbarrier.arrive.expect_tx`). The stage was last read before the
    /// barrier every thread passed: `fence.proxy.async` orders those reads
    /// before the engine's writes.
    #[allow(clippy::too_many_arguments)]
    fn issue_bulk(
        &mut self,
        g: &GpuMmaTile,
        bars: &str,
        tid: &str,
        row0: &str,
        col0: &str,
        tile: &str,
        stage: &str,
    ) {
        let s = self.site;
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let k0 = self.op("mul", tile, bk);
        let so = self.op("mul", stage, g.stage_words());
        let boff = self.op("mul", stage, BULK_BAR_BYTES);
        let bar = self.bar_at(bars, &boff);
        let bytes = (bm * bk + bk * bn) * 4;
        let t32 = self.tmp();
        self.line(format!("{t32} = trunc i64 {tid} to i32"));
        self.asm_args(
            "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
            "r,r,r",
            &[format!("i32 {bar}"), format!("i32 {bytes}"), format!("i32 {t32}")],
        );
        self.asm("fence.proxy.async.shared::cta;");
        let rows = bm + bk;
        let warp = self.profile_gpu.warp;
        for j in 0..rows.div_ceil(warp) {
            let q = self.op("add", tid, j * warp);
            // Warp 0's lanes, rows below `bm + bk`.
            let (w0, inr) = (self.tmp(), self.tmp());
            self.line(format!("{w0} = icmp ult i64 {tid}, {warp}"));
            self.line(format!("{inr} = icmp ult i64 {q}, {rows}"));
            let on = self.tmp();
            self.line(format!("{on} = and i1 {w0}, {inr}"));
            let pred = self.tmp();
            self.line(format!("{pred} = zext i1 {on} to i32"));
            // An idle lane names row 0: every address it forms is in bounds.
            let q0 = self.tmp();
            self.line(format!("{q0} = select i1 {on}, i64 {q}, i64 0"));
            let q = q0;
            let is_a = self.tmp();
            self.line(format!("{is_a} = icmp ult i64 {q}, {bm}"));
            // `a` row `q`: (row0 + q)·ci + k0 + base, shared `q·lda`.
            let ar = self.op("add", row0, &q);
            let ag = self.op("mul", &ar, s.a.ci);
            let ag = self.op("add", &ag, &k0);
            let ag = self.plus(ag, s.a.base);
            let asw = self.op("mul", &q, lda);
            // `b` row `q − bm` (wrapping where `q < bm`: unused).
            let br = self.op("sub", &q, bm);
            let bkr = self.op("add", &k0, &br);
            let bg = self.op("mul", &bkr, s.b.ck);
            let bg = self.op("add", &bg, col0);
            let bg = self.plus(bg, s.b.base);
            let bsw = self.op("mul", &br, ldb);
            let bsw = self.plus(bsw, bm * lda);
            let ap = self.gep(1, "%a", &ag);
            let bp = self.gep(1, "%b", &bg);
            let gp = self.tmp();
            self.line(format!(
                "{gp} = select i1 {is_a}, ptr addrspace(1) {ap}, ptr addrspace(1) {bp}"
            ));
            let sw = self.tmp();
            self.line(format!("{sw} = select i1 {is_a}, i64 {asw}, i64 {bsw}"));
            let si = self.op("add", &sw, &so);
            let sp = self.gep(3, "@mapal_smem", &si);
            let s32 = self.tmp();
            self.line(format!("{s32} = ptrtoint ptr addrspace(3) {sp} to i32"));
            let n = self.tmp();
            self.line(format!(
                "{n} = select i1 {is_a}, i32 {}, i32 {}",
                bk * 4,
                bn * 4
            ));
            self.asm_args(
                "{ .reg .pred p; setp.ne.u32 p, $4, 0; @p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3]; }",
                "r,l,r,r,r",
                &[
                    format!("i32 {s32}"),
                    format!("ptr addrspace(1) {gp}"),
                    format!("i32 {n}"),
                    format!("i32 {bar}"),
                    format!("i32 {pred}"),
                ],
            );
        }
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

    /// An `mbarrier.try_wait.parity` spin on the barrier at the 32-bit shared
    /// address `bar` for the phase of parity `par` (an i64 value), in blocks
    /// `{label}w` (the spin) and `{label}d` (past it).
    fn mbar_wait(&mut self, bar: &str, par: &str, label: &str) {
        let p32 = self.tmp();
        self.line(format!("{p32} = trunc i64 {par} to i32"));
        self.line(format!("br label %{label}w"));
        self.out.push_str(&format!("{label}w:\n"));
        let ok = self.tmp();
        self.line(format!(
            "{ok} = call i32 asm sideeffect \"{{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.u32 $0, 1, 0, p; }}\", \"=r,r,r,~{{memory}}\"(i32 {bar}, i32 {p32})"
        ));
        let done = self.tmp();
        self.line(format!("{done} = icmp ne i32 {ok}, 0"));
        self.line(format!("br i1 {done}, label %{label}d, label %{label}w"));
        self.out.push_str(&format!("{label}d:\n"));
    }

    /// The 32-bit shared address `off` (an i64 value or literal) bytes into
    /// the dynamic shared memory.
    fn smem_at(&mut self, off: &str) -> String {
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {off}"
        ));
        let a = self.tmp();
        self.line(format!("{a} = ptrtoint ptr addrspace(3) {p} to i32"));
        a
    }

    /// `v` (an i64 value) as an i32.
    fn i32_of(&mut self, v: &str) -> String {
        let t = self.tmp();
        self.line(format!("{t} = trunc i64 {v} to i32"));
        t
    }

    /// **hopper-wgmma-tma K2: the warp-specialized Hopper TF32 kernel**
    /// ([`GpuWgmmaTile::ws`]) — S53's transposed product `cᵀ = bᵀ·aᵀ`
    /// ([`Self::wgmma`]: consumer warpgroup `cw` owns `c` columns
    /// `bx·bn + 64·cw ..+64` × rows `by·n ..+n`) fed by tensor boxes.
    ///
    /// - **Producer** (warpgroup 0, `setmaxnreg.dec` under `ws-regs`): step `j`
    ///   first rounds tile `j − (stages − 1)` in place — every thread waits
    ///   its stage's full barrier, rounds its 16-byte chunks of the `aᵀ` box
    ///   (RNE, `cvt.rn.tf32`: `wgmma` truncates what it reads), fences them
    ///   to the async proxy and arrives on the stage's ready barrier — then
    ///   thread 0 waits tile `j`'s stage empty, counts its bytes
    ///   (`expect_tx`) and issues one `aᵀ` box (`{bk, n}` at `(k0, row0)`) and
    ///   `bn/32` `b` boxes (`{32, bk}` at `(col0 + 32q, k0)`), all 128-byte
    ///   swizzled, onto its full barrier. `stages − 1` tiles are in flight
    ///   while one is rounded.
    /// - **Consumers** (`setmaxnreg.inc`): tile `i` waits its stage full (the
    ///   `b` boxes the fragments load) and ready (the rounded `aᵀ`), loads the
    ///   `bᵀ` fragments through the box swizzle (rounded in the load), issues
    ///   `bk/8` `wgmma` on the stage's descriptor and commits. Before its
    ///   fragment loads it waits for tile `i − 1`'s group (an `a` fragment
    ///   register may not be rewritten before then — PTX ISA), whose stage
    ///   each warp's lane 0 then releases (empty barrier): a stage is never
    ///   released while a group may still read it; the group in flight
    ///   computes through this tile's barrier waits, and the other consumer
    ///   warpgroup's through its fragment loads.
    /// - **Barriers** (after the stages): `full[s]` (one arrival + the bytes),
    ///   `ready[s]` (the producer's 128 threads), `empty[s]` (the consumer
    ///   warps); waits by phase parity.
    fn wgmma_ws(&mut self, name: &str, g: GpuWgmmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let wgt = WARPGROUP_WARPS * warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, n) = (g.bm(), g.bn(), g.bk, g.n);
        let (st, sb, ab) = (g.stages, g.ws_stage_bytes(), g.ws_a_bytes());
        let span = WGMMA_SWIZZLE_BYTES;
        let box_b = span * bk;
        let nk = s.k / bk;
        let lead = st - 1;
        let digits = self.rungs.tma_digits;
        let maps =
            ws_tensors(s, &g, &self.profile_gpu, digits).expect("the derivation checked the boxes");
        self.tmap = maps.to_vec();
        self.header(name, threads, GPU_VEC_BYTES, 1);

        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        let wg = self.op("udiv", &tid, wgt);
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
        // Barriers: full (kind 0), ready (1), empty (2), `stages` each.
        let bars = self.tmp();
        self.line(format!(
            "{bars} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {}",
            st * sb
        ));
        let t32 = self.i32_of(&tid);
        for kind in 0..3u64 {
            let count = [1, wgt, g.groups * WARPGROUP_WARPS][kind as usize];
            for x in 0..st {
                let a = self.bar_at(&bars, &((kind * st + x) * BULK_BAR_BYTES).to_string());
                self.asm_args(
                    "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared::cta.b64 [$0], $1; }",
                    "r,r,r",
                    &[format!("i32 {a}"), format!("i32 {count}"), format!("i32 {t32}")],
                );
            }
        }
        self.asm("fence.mbarrier_init.release.cluster;");
        self.line("call void @llvm.nvvm.barrier0()");
        let bar = |k: &mut Self, kind: u64, stage: &str| {
            let o = k.op("mul", stage, BULK_BAR_BYTES);
            let o = k.op("add", &o, kind * st * BULK_BAR_BYTES);
            k.bar_at(&bars, &o)
        };
        let prod = self.tmp();
        self.line(format!("{prod} = icmp eq i64 {wg}, 0"));
        self.line(format!("br i1 {prod}, label %prod, label %cons"));

        // ---- producer warpgroup ----
        self.out.push_str("prod:\n");
        if let Some((p, _)) = g.realloc {
            self.asm(&format!("setmaxnreg.dec.sync.aligned.u32 {p};"));
        }
        let tms: Vec<String> = (0..maps.len())
            .map(|m| {
                let t = self.tmp();
                self.line(format!("{t} = ptrtoint ptr %tm{m} to i64"));
                t
            })
            .collect();
        let r32 = self.i32_of(&row0);
        self.line("br label %pl");
        self.out.push_str("pl:\n");
        let (j, jn) = (self.tmp(), self.tmp());
        self.line(format!("{j} = phi i64 [0, %prod], [{jn}, %pn]"));
        let ge = self.tmp();
        self.line(format!("{ge} = icmp uge i64 {j}, {lead}"));
        self.line(format!("br i1 {ge}, label %pr, label %pi"));
        // Round tile `j − lead`'s `aᵀ` box in place.
        self.out.push_str("pr:\n");
        let i = self.op("sub", &j, lead);
        let x = self.op("urem", &i, st);
        let ph = self.op("udiv", &i, st);
        let ph = self.op("and", &ph, 1);
        let fb = bar(self, 0, &x);
        self.mbar_wait(&fb, &ph, "prf");
        let base = self.op("mul", &x, sb);
        let chunks = ab / GPU_VEC_BYTES;
        for p in 0..chunks.div_ceil(wgt) {
            let q = self.op("add", &tid, p * wgt);
            let tail = (p + 1) * wgt > chunks;
            if tail {
                let inr = self.tmp();
                self.line(format!("{inr} = icmp ult i64 {q}, {chunks}"));
                self.line(format!("br i1 {inr}, label %prc{p}, label %prn{p}"));
                self.out.push_str(&format!("prc{p}:\n"));
            }
            let o = self.op("mul", &q, GPU_VEC_BYTES);
            let o = self.op("add", &base, &o);
            let sp = self.tmp();
            self.line(format!(
                "{sp} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {o}"
            ));
            let v = self.tmp();
            self.line(format!(
                "{v} = load <4 x float>, ptr addrspace(3) {sp}, align 16"
            ));
            let mut r = "poison".to_string();
            for e in 0..4 {
                let xv = self.tmp();
                self.line(format!("{xv} = extractelement <4 x float> {v}, i64 {e}"));
                let y = self.round_tf32(&xv, false);
                let nr = self.tmp();
                self.line(format!(
                    "{nr} = insertelement <4 x i32> {r}, i32 {y}, i64 {e}"
                ));
                r = nr;
            }
            self.line(format!(
                "store <4 x i32> {r}, ptr addrspace(3) {sp}, align 16"
            ));
            if tail {
                self.line(format!("br label %prn{p}"));
                self.out.push_str(&format!("prn{p}:\n"));
            }
        }
        self.asm("fence.proxy.async.shared::cta;");
        let rb = bar(self, 1, &x);
        self.asm_args(
            "mbarrier.arrive.shared::cta.b64 _, [$0];",
            "r",
            &[format!("i32 {rb}")],
        );
        self.line("br label %pi");
        // Issue tile `j` (thread 0).
        self.out.push_str("pi:\n");
        let lt = self.tmp();
        self.line(format!("{lt} = icmp ult i64 {j}, {nk}"));
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 {tid}, 0"));
        let go = self.tmp();
        self.line(format!("{go} = and i1 {lt}, {t0}"));
        self.line(format!("br i1 {go}, label %pt, label %pn"));
        self.out.push_str("pt:\n");
        let x = self.op("urem", &j, st);
        let ph = self.op("udiv", &j, st);
        let ph = self.op("and", &ph, 1);
        let ph = self.op("xor", &ph, 1);
        let eb = bar(self, 2, &x);
        self.mbar_wait(&eb, &ph, "pe");
        let fb = bar(self, 0, &x);
        let bytes = ab + bn * bk * 4;
        self.asm_args(
            "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
            "r,r,r",
            &[format!("i32 {fb}"), format!("i32 {bytes}"), "i32 0".into()],
        );
        let k0 = self.op("mul", &j, bk);
        let k32 = self.i32_of(&k0);
        let base = self.op("mul", &x, sb);
        let ad = self.smem_at(&base);
        let copy = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];";
        if bk > span / 4 {
            // The k digits: coordinates (0, row0, k0 / 32).
            let kq = self.op("udiv", &k0, span / 4);
            let kq = self.i32_of(&kq);
            self.asm_args(
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
                "r,l,r,r,r,r",
                &[
                    format!("i32 {ad}"),
                    format!("i64 {}", tms[0]),
                    "i32 0".into(),
                    format!("i32 {r32}"),
                    format!("i32 {kq}"),
                    format!("i32 {fb}"),
                ],
            );
        } else {
            self.asm_args(
                copy,
                "r,l,r,r,r",
                &[
                    format!("i32 {ad}"),
                    format!("i64 {}", tms[0]),
                    format!("i32 {k32}"),
                    format!("i32 {r32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        if digits {
            // One box of every slab: coordinates (0, k0, col0 / 32).
            let o = self.op("add", &base, ab);
            let bd = self.smem_at(&o);
            let c = self.op("udiv", &col0, span / 4);
            let c32 = self.i32_of(&c);
            self.asm_args(
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
                "r,l,r,r,r,r",
                &[
                    format!("i32 {bd}"),
                    format!("i64 {}", tms[1]),
                    "i32 0".into(),
                    format!("i32 {k32}"),
                    format!("i32 {c32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        for q in 0..if digits { 0 } else { g.ws_b_boxes() } {
            let o = self.op("add", &base, ab + q * box_b);
            let bd = self.smem_at(&o);
            let c = self.op("add", &col0, q * span / 4);
            let c32 = self.i32_of(&c);
            self.asm_args(
                copy,
                "r,l,r,r,r",
                &[
                    format!("i32 {bd}"),
                    format!("i64 {}", tms[1]),
                    format!("i32 {c32}"),
                    format!("i32 {k32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        self.line("br label %pn");
        self.out.push_str("pn:\n");
        self.line(format!("{jn} = add i64 {j}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {jn}, {}", nk + lead));
        self.line(format!("br i1 {fin}, label %pend, label %pl"));
        self.out.push_str("pend:\n  ret void\n");

        // ---- consumer warpgroups ----
        self.out.push_str("cons:\n");
        if let Some((_, c)) = g.realloc {
            self.asm(&format!("setmaxnreg.inc.sync.aligned.u32 {c};"));
        }
        let cw = self.op("sub", &wg, 1);
        let ct = self.op("sub", &tid, wgt);
        let warp_id = self.op("udiv", &ct, warp);
        let w = self.op("urem", &warp_id, WARPGROUP_WARPS);
        let lane = self.op("urem", &tid, warp);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        // This thread's `bᵀ` fragment offsets in a stage at k-step 0: a_r =
        // (column nl = 64·cw + 16·w + g + 8h, k = t + 4e) of `b`, in box nl/32
        // at row k, byte 4·(nl % 32), 128-byte swizzled (the box's pattern;
        // a k-step adds 8 whole rows, which the pattern repeats over).
        let fc = self.op("mul", &cw, WGMMA_M);
        let fw = self.op("mul", &w, WGMMA_M / WARPGROUP_WARPS);
        let nl0 = self.op("add", &fc, &fw);
        let nl0 = self.op("add", &nl0, &grp);
        let mut offs = Vec::new();
        for (h, e) in [(0u64, 0u64), (1, 0), (0, 1), (1, 1)] {
            let nl = self.op("add", &nl0, 8 * h);
            let qb = self.op("udiv", &nl, span / 4);
            let qo = self.op("mul", &qb, box_b);
            let nb = self.op("urem", &nl, span / 4);
            let nb = self.op("mul", &nb, 4);
            let kr = self.op("add", &t4, 4 * e);
            let raw = self.op("mul", &kr, span);
            let raw = self.op("add", &raw, &nb);
            let sh = self.op("lshr", &raw, 7);
            let sh = self.op("and", &sh, 7);
            let sh = self.op("shl", &sh, 4);
            let sw = self.op("xor", &raw, &sh);
            let o = self.op("add", &qo, &sw);
            offs.push(self.op("add", &o, ab));
        }
        self.line("br label %ct");
        self.out.push_str("ct:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %cons], [{itn}, %cgo]"));
        let x = self.op("urem", &it, st);
        let ph = self.op("udiv", &it, st);
        let ph = self.op("and", &ph, 1);
        let fb = bar(self, 0, &x);
        self.mbar_wait(&fb, &ph, "cf");
        let rb = bar(self, 1, &x);
        self.mbar_wait(&rb, &ph, "cr");
        // PTX ISA: an `a` fragment register may not be rewritten before the
        // `wait_group` completing the `wgmma` that reads it — so tile `i − 1`'s
        // group is waited for here, after this tile's barriers (it computes
        // through them; the other consumer warpgroup's groups keep the
        // tensor core busy through the fragment loads), and its stage is
        // released at once: a stage is never released while a group reads it.
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let rel = self.tmp();
        self.line(format!("{rel} = icmp ugt i64 {it}, 0"));
        let l0 = self.tmp();
        self.line(format!("{l0} = icmp eq i64 {lane}, 0"));
        let go = self.tmp();
        self.line(format!("{go} = and i1 {rel}, {l0}"));
        self.line(format!("br i1 {go}, label %crel, label %cgo"));
        self.out.push_str("crel:\n");
        let prev = self.op("sub", &it, 1);
        let px = self.op("urem", &prev, st);
        let eb = bar(self, 2, &px);
        self.asm_args(
            "mbarrier.arrive.shared::cta.b64 _, [$0];",
            "r",
            &[format!("i32 {eb}")],
        );
        self.line("br label %cgo");
        self.out.push_str("cgo:\n");
        let base = self.op("mul", &x, sb);
        let mut frags = Vec::new();
        for kk in 0..bk / WGMMA_K {
            let mut f = Vec::new();
            for o in &offs {
                let i = self.op("add", &base, o);
                let i = self.op("add", &i, kk * WGMMA_K * span);
                let p = self.tmp();
                self.line(format!(
                    "{p} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {i}"
                ));
                let v = self.tmp();
                self.line(format!("{v} = load float, ptr addrspace(3) {p}, align 4"));
                f.push(self.round_tf32(&v, true));
            }
            frags.push(f);
        }
        self.asm("wgmma.fence.sync.aligned;");
        let a32 = self.smem_at(&base);
        let a64 = self.tmp();
        self.line(format!("{a64} = zext i32 {a32} to i64"));
        let enc = self.op("lshr", &a64, 4);
        let enc = self.op("and", &enc, 0x3fff);
        let desc0 = self.op("or", &enc, g.desc_fields());
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
            // k-step `kk` starts `kk·32` bytes into the k digits: atom column
            // `kk·32 / 128` (`n·128` bytes apart), `kk·32 % 128` into its rows.
            let kb = kk as u64 * WGMMA_K * 4;
            let off = kb / span * (n * span) + kb % span;
            let desc = self.op("add", &desc0, off >> 4);
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
        self.line(format!("br i1 {fin}, label %cd, label %ct"));

        self.out.push_str("cd:\n");
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let er = self.op("mul", &t4, 2);
        let er = self.op("add", &er, &row0);
        let er = self.op("mul", &er, s.c);
        let ec = self.op("add", &col0, &fc);
        let ec = self.op("add", &ec, &fw);
        let ec = self.op("add", &ec, &grp);
        let ebase = self.op("add", &er, &ec);
        for jj in 0..n / 8 {
            for h in 0..2u64 {
                for e in 0..2u64 {
                    let q = (4 * jj + 2 * h + e) as usize;
                    let p = acc_slot(self, q);
                    let xv = self.tmp();
                    self.line(format!("{xv} = load float, ptr {p}, align 4"));
                    let oi = self.op("add", &ebase, (8 * jj + e) * s.c + 8 * h);
                    if self.epi.is_some() {
                        self.store_cells(&[xv], &oi);
                        continue;
                    }
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store float {xv}, ptr addrspace(1) {op}, align 4"));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
        // The tensor maps' recipe (`mapal_gpu.c` `tmap_of`).
        let words: Vec<String> = std::iter::once(maps.len() as u64)
            .chain(maps.iter().flatten().copied())
            .map(|v| format!("i64 {v}"))
            .collect();
        self.out.push_str(&format!(
            "@{name}_tmap = addrspace(1) constant [{} x i64] [{}], align 8\n",
            words.len(),
            words.join(", ")
        ));
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
        // f32-tensor-cores `mma-combine`: a split kernel folds its piece
        // `z` of `S` — `k / S` from `z · k / S` ([`Kernel::at_z`]).
        let kext = s.k / self.splitk.max(1);
        let nk = kext / bk;
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
        let seed = self.split_seed(&seed);
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }
        // f32-tensor-cores `split-tile` (tf32x3): whether this thread split
        // an infinite operand (the cold path after the `k` loop).
        let split = g.split == Some(SplitKind::Tf32x3);
        let inf_slot = split.then(|| {
            let f = self.tmp();
            self.line(format!("{f} = alloca i32, align 4"));
            self.line(format!("store i32 0, ptr {f}, align 4"));
            f
        });

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
                // `head-group`: `b`'s row digits are the block's batch.
                let g = self.radix_at(m, &row0, &gc);
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
            let gsrc = self.at_z(gsrc, kext * s.a.ck);
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
            let gsrc = self.at_z(gsrc, kext * s.b.ck);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, bm * lda);
            copies.push((false, gsrc, sdst));
        }
        // TMA `bulk-stage`: warp 0 copies each panel row with the bulk-copy
        // engine, a stage's barrier counting the bytes ([`Self::issue_bulk`]).
        let bulk = bulk_stage(s, &g, &self.profile_gpu, &self.rungs);
        let bars = bulk.then(|| {
            let b = self.gep(3, smem, &(g.stages * stage_words).to_string());
            let t32 = self.tmp();
            self.line(format!("{t32} = trunc i64 {tid} to i32"));
            for st in 0..g.stages {
                let a = self.bar_at(&b, &format!("{}", st * BULK_BAR_BYTES));
                self.asm_args(
                    "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared::cta.b64 [$0], $1; }",
                    "r,r,r",
                    &[format!("i32 {a}"), "i32 1".into(), format!("i32 {t32}")],
                );
            }
            self.asm("fence.mbarrier_init.release.cluster;");
            self.line("call void @llvm.nvvm.barrier0()");
            b
        });
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            if let Some(b) = &bars {
                k.issue_bulk(&g, b, &tid, &row0, &col0, tile, stage);
                return;
            }
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
            if !bulk {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
        }
        for _ in nk..g.stages - 1 {
            if !bulk {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
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
        match &bars {
            // Panel `it`'s stage: its barrier's phase `it / stages` (parity).
            Some(b) => {
                let st = self.op("urem", &it, g.stages);
                let off = self.op("mul", &st, BULK_BAR_BYTES);
                let bar = self.bar_at(b, &off);
                let ph = self.op("udiv", &it, g.stages);
                let ph1 = self.op("and", &ph, 1);
                let par = self.tmp();
                self.line(format!("{par} = trunc i64 {ph1} to i32"));
                self.line("br label %mbw");
                self.out.push_str("mbw:\n");
                let ok = self.tmp();
                self.line(format!(
                    "{ok} = call i32 asm sideeffect \"{{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.u32 $0, 1, 0, p; }}\", \"=r,r,r,~{{memory}}\"(i32 {bar}, i32 {par})"
                ));
                let done = self.tmp();
                self.line(format!("{done} = icmp ne i32 {ok}, 0"));
                self.line(format!("br i1 {done}, label %mbd, label %mbw"));
                self.out.push_str("mbd:\n");
            }
            None => self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                g.stages - 2
            )),
        }
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
        if !bulk {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
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
        // A split form's flush depth and each tile's zeroed chain.
        let flush = match self.rungs.split_issue {
            true => split_flush(&g, s.k, self.splitk.max(1), &self.profile_gpu),
            false => 1,
        };
        let mut chain: Vec<Vec<String>> = vec![vec!["0.0".to_string(); 4]; mt * nt];
        for kk in 0..bk / MMA_K {
            let k8 = kk * MMA_K;
            if let Some(flag) = &inf_slot {
                // f32-tensor-cores `split-tile` (`mapal_ir::split`, tf32x3):
                // each element's `hi = rna(x)`, `lo = rna(x − hi)`; per m16n8
                // tile `lo·hi`, `hi·lo`, `hi·hi` from a zeroed accumulator,
                // then one round-to-nearest add into the f32 accumulator.
                let mut infs = Vec::new();
                let mut word = |k: &mut Self, x: String| -> [String; 2] {
                    let (h, l, inf) = match k.rungs.split_issue {
                        true => k.split_tf32(&x),
                        false => k.split_tf32_cvt(&x),
                    };
                    infs.push(inf);
                    [h, l]
                };
                let mut af = Vec::new();
                for mi in 0..mt as u64 {
                    let r = mi * MMA_M * lda + k8;
                    let xs: Vec<String> = match &lm_base {
                        Some(lb) => {
                            let i = self.op("add", lb, r);
                            let p = self.gep(3, smem, &i);
                            let a32 = self.tmp();
                            self.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
                            let m = self.tmp();
                            self.line(format!(
                                "{m} = call {{ i32, i32, i32, i32 }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0,$1,$2,$3}}, [$4];\", \"=r,=r,=r,=r,r\"(i32 {a32})"
                            ));
                            (0..4)
                                .map(|j| {
                                    let x = self.tmp();
                                    self.line(format!(
                                        "{x} = extractvalue {{ i32, i32, i32, i32 }} {m}, {j}"
                                    ));
                                    let f = self.tmp();
                                    self.line(format!("{f} = bitcast i32 {x} to float"));
                                    f
                                })
                                .collect()
                        }
                        None => [r, r + 8 * lda, r + 4, r + 8 * lda + 4]
                            .iter()
                            .map(|&o| {
                                let i = self.op("add", &a_base, o);
                                let p = self.gep(3, smem, &i);
                                let x = self.tmp();
                                self.line(format!(
                                    "{x} = load float, ptr addrspace(3) {p}, align 4"
                                ));
                                x
                            })
                            .collect(),
                    };
                    let parts: Vec<[String; 2]> = xs.into_iter().map(|x| word(self, x)).collect();
                    af.push(parts);
                }
                let mut bf = Vec::new();
                for ni in 0..nt as u64 {
                    let c = k8 * ldb + ni * MMA_N;
                    let parts: Vec<[String; 2]> = [c, c + 4 * ldb]
                        .iter()
                        .map(|&o| {
                            let i = self.op("add", &b_base, o);
                            let p = self.gep(3, smem, &i);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
                            word(self, x)
                        })
                        .collect();
                    bf.push(parts);
                }
                // The tile's zeroed chain carries `flush` k-steps before its
                // round-to-nearest add (tensor-cores-2, [`split_flush`]).
                let flushes = (kk + 1) % flush == 0;
                for mi in 0..mt {
                    for ni in 0..nt {
                        let a = |h: usize| -> Vec<String> {
                            af[mi].iter().map(|w| w[h].clone()).collect()
                        };
                        let b = |h: usize| -> Vec<String> {
                            bf[ni].iter().map(|w| w[h].clone()).collect()
                        };
                        // Smallest class first: lo·hi, hi·lo, hi·hi.
                        let mut t = std::mem::take(&mut chain[mi * nt + ni]);
                        for (x, y) in [(a(1), b(0)), (a(0), b(1)), (a(0), b(0))] {
                            t = self.mma_tf32_call(&x, &y, &t);
                        }
                        if !flushes {
                            chain[mi * nt + ni] = t;
                            continue;
                        }
                        let base = (mi * nt + ni) * 4;
                        for (q, tq) in t.iter().enumerate() {
                            let p = acc_slot(self, base + q);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr {p}, align 4"));
                            let d = self.tmp();
                            self.line(format!("{d} = fadd float {v}, {tq}"));
                            self.line(format!("store float {d}, ptr {p}, align 4"));
                        }
                        chain[mi * nt + ni] = vec!["0.0".to_string(); 4];
                    }
                }
                self.or_guards(flag, &infs);
                continue;
            }
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
        if let Some(flag) = &inf_slot {
            self.split_cold(flag, &orow, &ocol, &seed, kext, (mt, nt), &acc_slot);
        }
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
                    if self.splitk > 1 {
                        // `mma-combine`: the piece's partials into its slab
                        // (workspace) or added into the output (atomic).
                        let mut xs = Vec::new();
                        for j in 0..2 {
                            let p = acc_slot(self, base + j);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            xs.push(x);
                        }
                        let oi = self.op("add", &rc, ni as u64 * MMA_N);
                        let oi = self.slab(oi);
                        self.store_o(&xs, &oi);
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

    /// f32-tensor-cores `split-tile` (tf32x3): an f32 element `x`'s TF32
    /// parts as instruction operands, and whether the guard takes `x` — not
    /// finite, rounding to infinity (`|x| ≥ 2¹²⁸ − 2¹⁰³`: its `hi` would
    /// overflow), or nonzero below [`SplitForm::exact_above`] (its `lo` would
    /// round on the subnormal grid). Everything else is split by integer
    /// rounding (tensor-cores-2: `cvt.rna.tf32.f32`'s ties-away rounding as
    /// `(x + 2¹²) & ¬(2¹³ − 1)` on the bits, its value class checks left to
    /// the guard — 25 SASS instructions an element on sm_80 became 10):
    /// `hi = rna(x)`, `lo = rna(x − hi)` (`x − hi` exact). A guarded
    /// element's parts are never read: its warp recomputes the tile by FFMA.
    fn split_tf32(&mut self, x: &str) -> (String, String, String) {
        let rna = |k: &mut Self, b: &str| {
            let r = k.tmp();
            k.line(format!("{r} = add i32 {b}, 4096"));
            let h = k.tmp();
            k.line(format!("{h} = and i32 {r}, -8192"));
            h
        };
        let xb = self.tmp();
        self.line(format!("{xb} = bitcast float {x} to i32"));
        let thr = SplitForm::of(SplitKind::Tf32x3).exact_above as f32;
        let guard = self.split_guard(&xb, thr.to_bits(), TF32_RNA_OVERFLOW);
        let h = rna(self, &xb);
        let hf = self.tmp();
        self.line(format!("{hf} = bitcast i32 {h} to float"));
        let r = self.tmp();
        self.line(format!("{r} = fsub float {x}, {hf}"));
        let rb = self.tmp();
        self.line(format!("{rb} = bitcast float {r} to i32"));
        let l = rna(self, &rb);
        (h, l, guard)
    }

    /// `--gpu-off=split-issue`: the 5fcb401b tf32x3 cut — `cvt.rna` for
    /// both parts, the truncation select where `hi` overflows, the guard on
    /// infinities and subnormal-grid parts only.
    fn split_tf32_cvt(&mut self, x: &str) -> (String, String, String) {
        let cvt = |k: &mut Self, v: &str| {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(float {v})"
            ));
            r
        };
        let xb = self.tmp();
        self.line(format!("{xb} = bitcast float {x} to i32"));
        let ax = self.tmp();
        self.line(format!("{ax} = and i32 {xb}, 2147483647"));
        let inf = self.tmp();
        self.line(format!("{inf} = icmp eq i32 {ax}, 2139095040"));
        // `0 < |x| < exact_above`: `|x|`'s bits less one below its bits less
        // one, unsigned.
        let thr = SplitForm::of(SplitKind::Tf32x3).exact_above as f32;
        let am = self.tmp();
        self.line(format!("{am} = add i32 {ax}, -1"));
        let tiny = self.tmp();
        self.line(format!("{tiny} = icmp ult i32 {am}, {}", thr.to_bits() - 1));
        let guard = self.tmp();
        self.line(format!("{guard} = or i1 {inf}, {tiny}"));
        let hb = cvt(self, x);
        let hx = self.tmp();
        self.line(format!("{hx} = and i32 {hb}, 2147483647"));
        let hinf = self.tmp();
        self.line(format!("{hinf} = icmp eq i32 {hx}, 2139095040"));
        let fin = self.tmp();
        self.line(format!("{fin} = xor i1 {inf}, true"));
        let ovf = self.tmp();
        self.line(format!("{ovf} = and i1 {hinf}, {fin}"));
        let tb = self.tmp();
        self.line(format!("{tb} = and i32 {xb}, -8192"));
        let h = self.tmp();
        self.line(format!("{h} = select i1 {ovf}, i32 {tb}, i32 {hb}"));
        let hf = self.tmp();
        self.line(format!("{hf} = bitcast i32 {h} to float"));
        let r = self.tmp();
        self.line(format!("{r} = fsub float {x}, {hf}"));
        let lb = cvt(self, &r);
        let l = self.tmp();
        self.line(format!("{l} = select i1 {inf}, i32 0, i32 {lb}"));
        (h, l, guard)
    }

    /// The split guard of an element with bits `xb`: `|x| ≥ big` (not
    /// finite, or its leading part rounds to infinity) or `0 < |x| < tiny`
    /// (a part would round on the subnormal grid) — one unsigned range
    /// check: `|x| − 1 ≥ tiny − 1` and `|x| < big` keep it off the guard.
    fn split_guard(&mut self, xb: &str, tiny: u32, big: u32) -> String {
        let ax = self.tmp();
        self.line(format!("{ax} = and i32 {xb}, 2147483647"));
        let am = self.tmp();
        self.line(format!("{am} = add i32 {ax}, -1"));
        let small = self.tmp();
        self.line(format!("{small} = icmp ult i32 {am}, {}", tiny - 1));
        let large = self.tmp();
        self.line(format!("{large} = icmp uge i32 {ax}, {big}"));
        let g = self.tmp();
        self.line(format!("{g} = or i1 {small}, {large}"));
        g
    }

    /// tensor-cores-2 `bf16x3n` / `bf16x2n`: an f32 pair's first `parts`
    /// bf16 parts (`mapal_ir::split::bf16_parts`, the lower `k` in each
    /// register's low half) — `p₀ = rn(x)`, `p₁ = rn(x − p₀)`, `p₂ = rn(x −
    /// p₀ − p₁)`, each residual exact, `p₂` the whole rest for `|x| ≥
    /// 2⁻¹¹⁰` (three parts: exact) — and whether
    /// the guard takes either element (not finite, rounding to infinity, or
    /// nonzero below [`SplitForm::exact_above`]).
    fn split_bf16x2(&mut self, lo: &str, hi: &str, parts: usize) -> (Vec<String>, String) {
        let thr = SplitForm::of(SplitKind::Bf16x3n).exact_above as f32;
        let mut guards = Vec::new();
        for x in [lo, hi] {
            let xb = self.tmp();
            self.line(format!("{xb} = bitcast float {x} to i32"));
            guards.push(self.split_guard(&xb, thr.to_bits(), BF16_RN_OVERFLOW));
        }
        let guard = self.tmp();
        self.line(format!("{guard} = or i1 {}, {}", guards[0], guards[1]));
        let cvt = |k: &mut Self, l: &str, h: &str| {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rn.bf16x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {h}, float {l})"
            ));
            r
        };
        // A part register's two values as f32 (bf16 is f32's top half).
        let widen = |k: &mut Self, p: &str| {
            let (l, h) = (k.tmp(), k.tmp());
            k.line(format!("{l} = shl i32 {p}, 16"));
            k.line(format!("{h} = and i32 {p}, -65536"));
            let (lf, hf) = (k.tmp(), k.tmp());
            k.line(format!("{lf} = bitcast i32 {l} to float"));
            k.line(format!("{hf} = bitcast i32 {h} to float"));
            (lf, hf)
        };
        let sub = |k: &mut Self, a: &str, b: &str| {
            let r = k.tmp();
            k.line(format!("{r} = fsub float {a}, {b}"));
            r
        };
        // Each part the rounding of what the parts before it leave.
        let (mut l, mut h) = (lo.to_string(), hi.to_string());
        let mut out = Vec::new();
        for i in 0..parts {
            let p = cvt(self, &l, &h);
            if i + 1 < parts {
                let (w0, w1) = widen(self, &p);
                (l, h) = (sub(self, &l, &w0), sub(self, &h, &w1));
            }
            out.push(p);
        }
        (out, guard)
    }

    /// A split tile's per-thread guard flag `|=` this k-step's guards.
    fn or_guards(&mut self, flag: &str, guards: &[String]) {
        let Some(first) = guards.first() else {
            return;
        };
        let mut any = first.clone();
        for x in &guards[1..] {
            let o = self.tmp();
            self.line(format!("{o} = or i1 {any}, {x}"));
            any = o;
        }
        let (z, f0, f1) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{z} = zext i1 {any} to i32"));
        self.line(format!("{f0} = load i32, ptr {flag}, align 4"));
        self.line(format!("{f1} = or i32 {f0}, {z}"));
        self.line(format!("store i32 {f1}, ptr {flag}, align 4"));
    }

    /// One `mma.sync.m16n8k16` of the 16-bit format `ptx` into the
    /// accumulator `c` (four floats or literals); its four results.
    fn mma16_call(
        &mut self,
        ptx: &str,
        a: [&String; 4],
        b: [&String; 2],
        c: &[String],
    ) -> Vec<String> {
        let r = self.tmp();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{ptx}.{ptx}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        (0..4)
            .map(|q| {
                let d = self.tmp();
                self.line(format!(
                    "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                ));
                d
            })
            .collect()
    }

    /// One `mma.sync.m16n8k8` TF32 product into the accumulator `c` (four
    /// floats or literals); its four results.
    fn mma_tf32_call(&mut self, a: &[String], b: &[String], c: &[String]) -> Vec<String> {
        let r = self.tmp();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        (0..4)
            .map(|q| {
                let d = self.tmp();
                self.line(format!(
                    "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                ));
                d
            })
            .collect()
    }

    /// f32-tensor-cores `split-tile`: the guard's cold path (§2.5). A warp
    /// any of whose threads split an infinite operand (`vote.sync.any`) has
    /// its whole tile recomputed by FFMA from the operands in global memory
    /// over this piece's `k` (`kext` from `z·kext`) — IEEE products, so `inf ·
    /// b` keeps the oracle's value class where `hi·lo` would make `inf · 0`.
    #[allow(clippy::too_many_arguments)]
    fn split_cold(
        &mut self,
        flag: &str,
        orow: &str,
        ocol: &str,
        seed: &str,
        kext: u64,
        (mt, nt): (usize, usize),
        acc_slot: &dyn Fn(&mut Self, usize) -> String,
    ) {
        let s = self.site;
        let f = self.tmp();
        self.line(format!("{f} = load i32, ptr {flag}, align 4"));
        let v = self.tmp();
        self.line(format!(
            "{v} = call i32 asm sideeffect \"{{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; selp.u32 $0, 1, 0, q; }}\", \"=r,r\"(i32 {f})"
        ));
        let c = self.tmp();
        self.line(format!("{c} = icmp ne i32 {v}, 0"));
        self.line(format!("br i1 {c}, label %cold, label %warm"));
        self.out.push_str("cold:\n");
        for idx in 0..mt * nt * 4 {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }
        let kb = self.at_z("0".to_string(), kext);
        let kb = self.plus(kb, 0);
        let ke = self.op("add", &kb, kext);
        self.line("br label %ck");
        self.out.push_str("ck:\n");
        let (kq, kn) = (self.tmp(), self.tmp());
        self.line(format!("{kq} = phi i64 [{kb}, %cold], [{kn}, %ck]"));
        let mut av = Vec::new();
        for mi in 0..mt as u64 {
            for half in 0..2u64 {
                let r = self.op("add", orow, mi * MMA_M + half * 8);
                let ri = self.op("mul", &r, s.a.ci);
                let ki = self.op("mul", &kq, s.a.ck);
                let i = self.op("add", &ri, &ki);
                let i = self.op("add", &i, s.a.base);
                let x = self.cold_load("%a", s.a.narrow, &i);
                av.push(x);
            }
        }
        let mut bv = Vec::new();
        for ni in 0..nt as u64 {
            for j in 0..2u64 {
                let c = self.op("add", ocol, ni * MMA_N + j);
                let ci = self.op("mul", &c, s.b.clane);
                let ki = self.op("mul", &kq, s.b.ck);
                let i = self.op("add", &ci, &ki);
                let i = self.op("add", &i, s.b.base);
                let x = self.cold_load("%b", s.b.narrow, &i);
                bv.push(x);
            }
        }
        for mi in 0..mt {
            for ni in 0..nt {
                for half in 0..2 {
                    for j in 0..2 {
                        let idx = (mi * nt + ni) * 4 + half * 2 + j;
                        let p = acc_slot(self, idx);
                        let a = &av[mi * 2 + half];
                        let b = &bv[ni * 2 + j];
                        let o = self.tmp();
                        self.line(format!("{o} = load float, ptr {p}, align 4"));
                        let d = self.tmp();
                        self.line(format!(
                            "{d} = call float @llvm.fma.f32(float {a}, float {b}, float {o})"
                        ));
                        self.line(format!("store float {d}, ptr {p}, align 4"));
                    }
                }
            }
        }
        self.line(format!("{kn} = add i64 {kq}, 1"));
        let e = self.tmp();
        self.line(format!("{e} = icmp eq i64 {kn}, {ke}"));
        self.line(format!("br i1 {e}, label %warm, label %ck"));
        self.out.push_str("warm:\n");
    }

    /// The cold path's operand element at `i` of `arr`, as f32: a narrow
    /// operand's exact widening (tensor-cores-2 `bf16x3n`).
    fn cold_load(&mut self, arr: &str, narrow: Option<mapal_ir::NarrowFloat>, i: &str) -> String {
        let Some(n) = narrow else {
            let p = self.gep(1, arr, i);
            let x = self.tmp();
            self.line(format!("{x} = load float, ptr addrspace(1) {p}, align 4"));
            return x;
        };
        let nt = crate::narrow::llvm_ty(n);
        let p = self.gep_ty(1, nt, arr, i);
        let (w, v) = (self.tmp(), self.tmp());
        self.line(format!(
            "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
            n.bytes()
        ));
        let tt = if n.bytes() == 1 {
            self.tmp()
        } else {
            String::new()
        };
        for l in crate::narrow::widen_lines(n, nt, &w, "float", &v, &tt) {
            self.line(l);
        }
        v
    }

    /// **The 16-bit tensor-core kernel** (tensor-cores, ADR-0043 D6):
    /// [`Self::mma`]'s block, warp and `m16n8` tiles over
    /// `mma.sync.m16n8k16.row.col.f32.{bf16,f16}` with f32 accumulators in
    /// registers. Every panel lands by 16-byte `cp.async` in its storage
    /// format (`g.ea`, `g.eb`): a narrow panel's fragments are read by
    /// `ldmatrix` (`a`: `.x4`, row-major `m × k`; `b`: `.x4.trans` / `.x2.trans`
    /// of the `k × n` panel — two `n8` tiles a load); an f32 panel's pairs
    /// are loaded and rounded once to the format (`cvt.rn.bf16x2.f32`: the
    /// member the site's product admits, ADR-0043 D4).
    ///
    /// Fragment layouts are the PTX ISA's for m16n8k16 (`group = lane / 4`,
    /// `t = lane % 4`, two elements a register, the lower `k` in the low
    /// half): A `a0..a3` = (g, 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8); B
    /// `b0, b1` = (k 2t, n g), (k 2t+8, n g); C as m16n8k8's.
    fn mma16(&mut self, name: &str, g: GpuMmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let (ea, eb, ab) = (g.ea, g.eb, g.a_bytes());
        let stage_bytes = g.stage_bytes();
        let ptx = g.fmt.ptx();
        let mk = g.mma_k();
        // f32-tensor-cores `mma-combine`: a split kernel folds its piece
        // `z` of `S` — `k / S` from `z · k / S` ([`Kernel::at_z`]).
        let kext = s.k / self.splitk.max(1);
        let nk = kext / bk;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";
        let ety = |e: u64| if e == 2 { "i16" } else { "float" };

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
        let seed = self.split_seed(&seed);
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }
        // tensor-cores-2 `bf16x3n`: the f32 operand cut into three exact
        // bf16 parts as its fragment is read; whether this thread cut an
        // operand the guard takes (the cold path after the `k` loop).
        let cut = match g.split {
            Some(SplitKind::Bf16x3n) => 3,
            Some(SplitKind::Bf16x2n) => 2,
            _ => 0,
        };
        let inf_slot = (cut > 0).then(|| {
            let f = self.tmp();
            self.line(format!("{f} = alloca i32, align 4"));
            self.line(format!("store i32 0, ptr {f}, align 4"));
            f
        });
        // A split form's flush depth: the k-steps its zeroed chain carries
        // before the round-to-nearest add ([`split_flush`]).
        let flush = match self.rungs.split_issue {
            true => split_flush(&g, s.k, self.splitk.max(1), &self.profile_gpu),
            false => 1,
        };

        // cp.async: (is_a, global element index at k0 = 0, shared byte in
        // stage 0), one per 16-byte chunk this thread copies, A then B.
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        let (la, lb) = (GPU_VEC_BYTES / ea, GPU_VEC_BYTES / eb);
        for p in 0..bm * bk / la / threads {
            let li = self.op("add", &tid, p * threads);
            let r = self.op("udiv", &li, bk / la);
            let kc0 = self.op("urem", &li, bk / la);
            let kc = self.op("mul", &kc0, la);
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let g0 = self.op("add", &gr, &kc);
            let gsrc = self.op("add", &g0, s.a.base);
            let gsrc = self.at_z(gsrc, kext * s.a.ck);
            let sr = self.op("mul", &r, lda);
            let se = self.op("add", &sr, &kc);
            copies.push((true, gsrc, self.op("mul", &se, ea)));
        }
        for p in 0..bk * bn / lb / threads {
            let li = self.op("add", &tid, p * threads);
            // `bt`: rows `n`, 16-byte chunks along `k` (as `a`'s).
            let chunks = if g.bt { bk / lb } else { bn / lb };
            let r = self.op("udiv", &li, chunks);
            let c0 = self.op("urem", &li, chunks);
            let cc = self.op("mul", &c0, lb);
            let g0 = if g.bt {
                let gn = self.op("add", &col0, &r);
                let gr = self.op("mul", &gn, s.b.clane);
                self.op("add", &gr, &cc)
            } else {
                let gk = self.op("mul", &r, s.b.ck);
                let gc = self.op("add", &col0, &cc);
                self.op("add", &gk, &gc)
            };
            let gsrc = self.op("add", &g0, s.b.base);
            let gsrc = self.at_z(gsrc, kext * s.b.ck);
            let sk = self.op("mul", &r, ldb);
            let sn = self.op("add", &sk, &cc);
            let sb = self.op("mul", &sn, eb);
            copies.push((false, gsrc, self.op("add", &sb, ab)));
        }
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            let k0 = k.op("mul", tile, bk);
            let ka = k.op("mul", &k0, s.a.ck);
            let kb = k.op("mul", &k0, s.b.ck);
            let so = k.op("mul", stage, stage_bytes);
            for (is_a, gsrc, sdst) in &copies {
                let (arr, off, e) = if *is_a {
                    ("%a", &ka, ea)
                } else {
                    ("%b", &kb, eb)
                };
                let gi = k.op("add", gsrc, off);
                let gp = k.gep_ty(1, ety(e), arr, &gi);
                let si = k.op("add", sdst, &so);
                let sp = k.gep_ty(3, "i8", smem, &si);
                k.line(format!(
                    "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
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
        // Per-thread fragment origins inside a stage, in bytes. `ldmatrix`:
        // lane `l` gives row `l % 8` of matrix `l / 8`.
        let q = self.op("udiv", &lane, 8);
        let i8l = self.op("urem", &lane, 8);
        let qr = self.op("urem", &q, 2);
        let qk = self.op("udiv", &q, 2);
        let wrow = self.op("mul", &wr, g.wm());
        let wcol = self.op("mul", &wc, g.wn());
        let a_org = if ea == 2 {
            // a: matrices (rows 0-7 | 8-15) × (k 0-7 | 8-15), row-major.
            let r = self.op("mul", &qr, 8);
            let r = self.op("add", &r, &i8l);
            let r = self.op("add", &r, &wrow);
            let r = self.op("mul", &r, lda);
            let kq = self.op("mul", &qk, 8);
            let o = self.op("add", &r, &kq);
            self.op("mul", &o, 2)
        } else {
            // a: the pair (row g, k 2t..2t+1).
            let r = self.op("add", &wrow, &grp);
            let r = self.op("mul", &r, lda);
            let t2 = self.op("mul", &t4, 2);
            let o = self.op("add", &r, &t2);
            self.op("mul", &o, 4)
        };
        let b_org = if g.bt {
            // `n × k`: as `a` — `ldmatrix` matrix `q` is n rows `(q / 2)·8
            // ..+8` (two n tiles) at k `(q % 2)·8`; an f32 pair (n g, k 2t).
            let (n, kq, e) = if eb == 2 {
                let n = self.op("mul", &qk, MMA_N);
                let n = self.op("add", &n, &i8l);
                (n, self.op("mul", &qr, 8), 2)
            } else {
                (grp.clone(), self.op("mul", &t4, 2), 4)
            };
            let n = self.op("add", &n, &wcol);
            let r = self.op("mul", &n, ldb);
            let o = self.op("add", &r, &kq);
            let o = self.op("mul", &o, e);
            self.op("add", &o, ab)
        } else if eb == 2 {
            // b (`.trans`): matrix `q` is k rows `(q % 2)·8 ..+8` of n tile
            // `q / 2`; lane `l` gives its row `l % 8`.
            let kr = self.op("mul", &qr, 8);
            let kr = self.op("add", &kr, &i8l);
            let kr = self.op("mul", &kr, ldb);
            let nq = self.op("mul", &qk, MMA_N);
            let n = self.op("add", &wcol, &nq);
            let o = self.op("add", &kr, &n);
            let o = self.op("mul", &o, 2);
            self.op("add", &o, ab)
        } else {
            // b: the words (k 2t + j, n g).
            let kr = self.op("mul", &t4, 2);
            let kr = self.op("mul", &kr, ldb);
            let n = self.op("add", &wcol, &grp);
            let o = self.op("add", &kr, &n);
            let o = self.op("mul", &o, 4);
            self.op("add", &o, ab)
        };
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
        let cbase = self.op("mul", &cst, stage_bytes);
        let a_base = self.op("add", &cbase, &a_org);
        let b_base = self.op("add", &cbase, &b_org);
        // `ldmatrix` of `n` b16 matrices at shared byte `base + off`.
        let ldsm = |k: &mut Self, base: &str, off: u64, n: usize, trans: bool| -> Vec<String> {
            let i = k.op("add", base, off);
            let p = k.gep_ty(3, "i8", smem, &i);
            let a32 = k.tmp();
            k.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
            let ty = vec!["i32"; n].join(", ");
            let outs = (0..n)
                .map(|j| format!("${j}"))
                .collect::<Vec<_>>()
                .join(",");
            let cons = vec!["=r"; n].join(",");
            let m = k.tmp();
            k.line(format!(
                "{m} = call {{ {ty} }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x{n}{}.shared.b16 {{{outs}}}, [${n}];\", \"{cons},r\"(i32 {a32})",
                if trans { ".trans" } else { "" }
            ));
            (0..n)
                .map(|j| {
                    let x = k.tmp();
                    k.line(format!("{x} = extractvalue {{ {ty} }} {m}, {j}"));
                    x
                })
                .collect()
        };
        // Two f32 values rounded once into one register of the format, the
        // lower `k` in the low half.
        let pack = |k: &mut Self, lo: &str, hi: &str| -> String {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rn.{ptx}x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {hi}, float {lo})"
            ));
            r
        };
        let lds = |k: &mut Self, base: &str, off: u64| -> String {
            let i = k.op("add", base, off);
            let p = k.gep_ty(3, "i8", smem, &i);
            let x = k.tmp();
            k.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
            x
        };
        // The fragment registers of an f32 pair (the lower `k` in the low
        // half): rounded once to the format (the member's realization), or
        // — `bf16x3n` — its three exact bf16 parts, largest first.
        let mut infs: Vec<String> = Vec::new();
        let mut frag = |k: &mut Self, lo: &str, hi: &str| -> Vec<String> {
            if cut > 0 {
                let (parts, guard) = k.split_bf16x2(lo, hi, cut);
                infs.push(guard);
                return parts;
            }
            vec![pack(k, lo, hi)]
        };
        // Each tile's zeroed chain under a split form, carried `flush`
        // k-steps, then added into its accumulator by round-to-nearest adds.
        let zero = || vec!["0.0".to_string(); 4];
        let mut chain: Vec<Vec<String>> = vec![zero(); mt * nt];
        for kk in 0..bk / mk {
            let k16 = kk * mk;
            // af[mi][part] / bf[ni][part]: one part (a narrow panel, or the
            // member's rounding), three (`bf16x3n`'s f32 operand).
            let mut af: Vec<Vec<[String; 4]>> = Vec::new();
            for mi in 0..mt as u64 {
                if ea == 2 {
                    let f = ldsm(self, &a_base, (mi * MMA_M * lda + k16) * 2, 4, false);
                    af.push(vec![[
                        f[0].clone(),
                        f[1].clone(),
                        f[2].clone(),
                        f[3].clone(),
                    ]]);
                    continue;
                }
                let mut q: Vec<Vec<String>> = Vec::new();
                for (rr, kc) in [(0u64, 0u64), (8, 0), (0, 8), (8, 8)] {
                    let off = ((mi * MMA_M + rr) * lda + k16 + kc) * 4;
                    let i = self.op("add", &a_base, off);
                    let p = self.gep_ty(3, "i8", smem, &i);
                    let v = self.tmp();
                    self.line(format!(
                        "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                    ));
                    let (x0, x1) = (self.tmp(), self.tmp());
                    self.line(format!("{x0} = extractelement <2 x float> {v}, i64 0"));
                    self.line(format!("{x1} = extractelement <2 x float> {v}, i64 1"));
                    q.push(frag(self, &x0, &x1));
                }
                af.push(
                    (0..q[0].len())
                        .map(|pp| {
                            [
                                q[0][pp].clone(),
                                q[1][pp].clone(),
                                q[2][pp].clone(),
                                q[3][pp].clone(),
                            ]
                        })
                        .collect(),
                );
            }
            let mut bf: Vec<Vec<[String; 2]>> = Vec::new();
            let pairs = |q: Vec<Vec<String>>| -> Vec<[String; 2]> {
                (0..q[0].len())
                    .map(|pp| [q[0][pp].clone(), q[1][pp].clone()])
                    .collect()
            };
            let mut ni = 0u64;
            while ni < nt as u64 {
                let off = match g.bt {
                    true => ni * MMA_N * ldb + k16,
                    false => k16 * ldb + ni * MMA_N,
                };
                if g.bt && eb == 4 {
                    let mut q = Vec::new();
                    for kc in [0u64, 8] {
                        let i = self.op("add", &b_base, (off + kc) * 4);
                        let p = self.gep_ty(3, "i8", smem, &i);
                        let v = self.tmp();
                        self.line(format!(
                            "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                        ));
                        let (x0, x1) = (self.tmp(), self.tmp());
                        self.line(format!("{x0} = extractelement <2 x float> {v}, i64 0"));
                        self.line(format!("{x1} = extractelement <2 x float> {v}, i64 1"));
                        q.push(frag(self, &x0, &x1));
                    }
                    bf.push(pairs(q));
                    ni += 1;
                    continue;
                }
                if eb == 2 {
                    let pair = ni + 1 < nt as u64;
                    let f = ldsm(self, &b_base, off * 2, if pair { 4 } else { 2 }, !g.bt);
                    bf.push(vec![[f[0].clone(), f[1].clone()]]);
                    if pair {
                        bf.push(vec![[f[2].clone(), f[3].clone()]]);
                        ni += 1;
                    }
                    ni += 1;
                    continue;
                }
                let mut q = Vec::new();
                for kc in [0u64, 8] {
                    let x0 = lds(self, &b_base, (off + kc * ldb) * 4);
                    let x1 = lds(self, &b_base, (off + (kc + 1) * ldb) * 4);
                    q.push(frag(self, &x0, &x1));
                }
                bf.push(pairs(q));
                ni += 1;
            }
            let flushes = g.split.is_some() && (kk + 1) % flush == 0;
            for mi in 0..mt {
                for (ni, bp) in bf.iter().enumerate() {
                    let base = (mi * nt + ni) * 4;
                    let ap = &af[mi];
                    if g.split.is_some() {
                        // f32-tensor-cores `split-tile`: the exact part
                        // products, smallest class first, into the tile's
                        // zeroed chain (narrow1: one; bf16x3n: three).
                        let parts = ap.len().max(bp.len());
                        let mut t = std::mem::take(&mut chain[mi * nt + ni]);
                        for pp in (0..parts).rev() {
                            let [a0, a1, a2, a3] = &ap[pp.min(ap.len() - 1)];
                            let [b0, b1] = &bp[pp.min(bp.len() - 1)];
                            t = self.mma16_call(ptx, [a0, a1, a2, a3], [b0, b1], &t);
                        }
                        if !flushes {
                            chain[mi * nt + ni] = t;
                            continue;
                        }
                        // The flush: one round-to-nearest add a cell.
                        for (q, d) in t.iter().enumerate() {
                            let p = acc_slot(self, base + q);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr {p}, align 4"));
                            let x = self.tmp();
                            self.line(format!("{x} = fadd float {v}, {d}"));
                            self.line(format!("store float {x}, ptr {p}, align 4"));
                        }
                        chain[mi * nt + ni] = zero();
                        continue;
                    }
                    let mut cur = Vec::new();
                    for q in 0..4 {
                        let p = acc_slot(self, base + q);
                        let v = self.tmp();
                        self.line(format!("{v} = load float, ptr {p}, align 4"));
                        cur.push((p, v));
                    }
                    let r = self.tmp();
                    let [a0, a1, a2, a3] = &ap[0];
                    let [b0, b1] = &bp[0];
                    self.line(format!(
                        "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{ptx}.{ptx}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {a0}, i32 {a1}, i32 {a2}, i32 {a3}, i32 {b0}, i32 {b1}, float {}, float {}, float {}, float {})",
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
        if let Some(flag) = &inf_slot {
            self.or_guards(flag, &infs);
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        // C fragments: (g, 2t..2t+1) and (g+8, 2t..2t+1) of every m16n8 tile.
        self.out.push_str("done:\n");
        let orow = self.op("add", &wrow, &row0);
        let orow = self.op("add", &orow, &grp);
        let ocol = self.op("add", &wcol, &col0);
        let t2 = self.op("mul", &t4, 2);
        let ocol = self.op("add", &ocol, &t2);
        if let Some(flag) = &inf_slot {
            self.split_cold(flag, &orow, &ocol, &seed, kext, (mt, nt), &acc_slot);
        }
        for mi in 0..mt {
            for half in 0..2u64 {
                let r = self.op("add", &orow, mi as u64 * MMA_M + half * 8);
                let rr = self.op("mul", &r, s.c);
                let rc = self.op("add", &rr, &ocol);
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4 + half as usize * 2;
                    let mut xs = Vec::new();
                    for j in 0..2 {
                        let p = acc_slot(self, base + j);
                        let x = self.tmp();
                        self.line(format!("{x} = load float, ptr {p}, align 4"));
                        xs.push(x);
                    }
                    let oi = self.op("add", &rc, ni as u64 * MMA_N);
                    if self.epi.is_some() {
                        self.store_cells(&xs, &oi);
                        continue;
                    }
                    if self.splitk > 1 {
                        // `mma-combine`: as [`Self::mma`]'s.
                        let oi = self.slab(oi);
                        self.store_o(&xs, &oi);
                        continue;
                    }
                    let mut vec = "poison".to_string();
                    for (j, x) in xs.iter().enumerate() {
                        let nv = self.tmp();
                        self.line(format!(
                            "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                        ));
                        vec = nv;
                    }
                    let op = self.gep_ty(1, "float", "%o", &oi);
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
        // ADR-0043: a `cp.async` copy cannot widen: a streamed narrow `b`
        // panel stays narrow in shared memory (`narrow-async`), read and
        // widened per fragment; a narrow `a` is register-staged.
        assert!(
            !(streamed && s.a.narrow.is_some()),
            "a narrow `a` panel is register-staged"
        );
        let nb = s.b.narrow.filter(|_| streamed);
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
        // sweep-attention-2 `kmajor-b`: `b`'s copies along its k-major rows.
        let kvb = rungs.kmajor_b && kvec_b(s, &g);
        let [(a_map, a_loads), (b_map, b_loads)] = lane_maps(s, &g, kmaj, kvb);
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
            |k, r, c| {
                let (kr, nc) = if kvb { (c, r) } else { (r, c) };
                if let Some(m) = rb {
                    b_kc.borrow_mut().push(kr.to_string());
                    let gc = k.op("add", &col0, nc);
                    let sk = k.op("mul", kr, bn);
                    let sn = k.op("add", &sk, nc);
                    // `head-group`: `b`'s row digits are the block's batch.
                    return (k.radix_at(m, &row0, &gc), k.op("add", &sn, bs_off));
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
            |d0, d1| {
                let (dk, dn) = if kvb { (d1, d0) } else { (d0, d1) };
                (dk * s.b.ck + dn * s.b.clane, dk * bn + dn)
            },
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
        let b_ty = if b_vec || kvb { vt.clone() } else { t.clone() };
        let a_al = if a_vec { GPU_VEC_BYTES } else { e };
        let b_al = if b_vec || kvb { GPU_VEC_BYTES } else { e };
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
                // A split partition's `k` starts `z·kext` along the fold.
                let kk = k.at_z(kk, kext);
                let kx = k.digits(&m.k, &kk);
                return k.op("add", gsrc, &kx);
            }
            let gi = k.op("add", gsrc, kd);
            k.plus(gi, goff)
        };
        // ADR-0043: a narrow operand's copy loads its format (the same
        // element count: a vector is `lanes` narrow elements) and widens it
        // in registers — the panel in shared memory is `elem`, so every
        // fragment read and step after it is the f32 tile's.
        let load_op = |k: &mut Self,
                       arr: &str,
                       gi: &str,
                       vec: bool,
                       ty: &str,
                       al: u64,
                       r: &mapal_ir::TileRead| {
            let Some(n) = r.narrow else {
                let gp = k.gep(1, arr, gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {ty}, ptr addrspace(1) {gp}, align {al}"
                ));
                return v;
            };
            let nt = crate::narrow::llvm_ty(n);
            let (nty, nal) = match vec {
                true => (format!("<{lanes} x {nt}>"), lanes * n.bytes()),
                false => (nt.to_string(), n.bytes()),
            };
            let gp = k.gep_ty(1, nt, arr, gi);
            let (w, v) = (k.tmp(), k.tmp());
            k.line(format!(
                "{w} = load {nty}, ptr addrspace(1) {gp}, align {nal}"
            ));
            let tt = if n.bytes() == 1 {
                k.tmp()
            } else {
                String::new()
            };
            for l in crate::narrow::widen_lines(n, &nty, &w, ty, &v, &tt) {
                k.line(l);
            }
            v
        };
        let load_panels = |k: &mut Self, k0: &str| -> Vec<String> {
            let ka = k.op("mul", k0, s.a.ck);
            let kb = k.op("mul", k0, s.b.ck);
            let mut vals = Vec::new();
            for (p, (gsrc, goff, _, _)) in a_src.iter().enumerate() {
                let gi = gidx(k, ra, &a_kc, p, gsrc, *goff, k0, &ka);
                vals.push(load_op(k, "%a", &gi, a_vec, &a_ty, a_al, &s.a));
            }
            for (p, (gsrc, goff, _, _)) in b_src.iter().enumerate() {
                let gi = gidx(k, rb, &b_kc, p, gsrc, *goff, k0, &kb);
                vals.push(load_op(k, "%b", &gi, b_vec || kvb, &b_ty, b_al, &s.b));
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
                if kvb {
                    // `kmajor-b`: one vector along `k` scatters to `lanes`
                    // panel rows.
                    for j in 0..lanes {
                        let x = k.tmp();
                        k.line(format!("{x} = extractelement {vt} {v}, i64 {j}"));
                        let si = k.op("add", sdst, soff + j * bn);
                        let sp = k.gep(3, smem, &si);
                        k.line(format!("store {t} {x}, ptr addrspace(3) {sp}, align {e}"));
                    }
                    continue;
                }
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
                    // `narrow-async`: the `b` region keeps its place in the
                    // stage, its elements narrow: element `si` of the f32
                    // layout is narrow element `si + (4 / e − 1)·(so + bs_off)`
                    // (the region packed from its first byte; `e` 2 keeps the
                    // 16-bit formats' text).
                    if let Some(n) = nb {
                        let nt = crate::narrow::llvm_ty(n);
                        let gp = k.gep_ty(1, nt, "%b", &gi);
                        let si = k.op("add", sdst, &so);
                        let si = k.plus(si, *soff);
                        let (ni, f) = match n.bytes() {
                            2 => (k.op("add", &si, &so), 1),
                            e => {
                                let f = 4 / e - 1;
                                let c = k.op("mul", &so, f);
                                (k.op("add", &si, &c), f)
                            }
                        };
                        let ni = k.plus(ni, f * bs_off);
                        let sp = k.gep_ty(3, nt, smem, &ni);
                        k.line(format!(
                            "call void @llvm.nvvm.cp.async.ca.shared.global.{}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})",
                            lanes * n.bytes()
                        ));
                        continue;
                    }
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
        let frag = |k: &mut Self,
                    ki: &str,
                    row: u64,
                    first: &str,
                    n: u64,
                    off: u64,
                    chunk: u64,
                    narrow: Option<mapal_ir::NarrowFloat>| {
            let r = k.op("mul", ki, row);
            let r = k.op("add", &r, first);
            let r = k.op("add", &r, off);
            let r = k.op("add", &r, &cbase);
            // `narrow-async`: the stage's narrow `b` region (see `issue`).
            // The narrow region packs from the f32 region's first byte: f32
            // element `r` is narrow element `r + (4 / e − 1)·(stage + bs_off)`
            // (`e` 2: the 16-bit formats' text as it was; 1: fp8).
            let r = match narrow {
                Some(n) if n.bytes() == 2 => {
                    let x = k.op("add", &r, &cbase);
                    k.plus(x, bs_off)
                }
                Some(n) => {
                    let f = 4 / n.bytes() - 1;
                    let c = k.op("mul", &cbase, f);
                    let x = k.op("add", &r, &c);
                    k.plus(x, f * bs_off)
                }
                None => r,
            };
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
                let x = if let Some(nf) = narrow {
                    let nt = crate::narrow::llvm_ty(nf);
                    let nwt = if w == 1 {
                        nt.to_string()
                    } else {
                        format!("<{w} x {nt}>")
                    };
                    let sp = k.gep_ty(3, nt, smem, &si);
                    let (y, x) = (k.tmp(), k.tmp());
                    k.line(format!(
                        "{y} = load {nwt}, ptr addrspace(3) {sp}, align {}",
                        w * nf.bytes()
                    ));
                    let tt = if nf.bytes() == 1 {
                        k.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(nf, &nwt, &y, &wt, &x, &tt) {
                        k.line(l);
                    }
                    x
                } else {
                    let sp = k.gep(3, smem, &si);
                    let x = k.tmp();
                    k.line(format!(
                        "{x} = load {wt}, ptr addrspace(3) {sp}, align {}",
                        w * e
                    ));
                    x
                };
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
                let bv = frag(self, &kj, bn, &b_frag0, tn, bs_off, b_chunk, nb);
                let av: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                fma(self, &av, &bv);
            }
        } else {
            let av = frag(self, &kk, sa, &a_frag0, tm, 0, a_chunk, None);
            let bv = frag(self, &kk, bn, &b_frag0, tn, bs_off, b_chunk, nb);
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
        if self.col && !narrow {
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
        // sweep-runtime-2 `streamk-epi`: a piece that is its tile's whole
        // fold stores through the group's members as the ordered kernel does
        // (`st_whole`); a piece of a shared tile adds its partials into the
        // tile's workspace slot and takes the tile's ticket (`st_part`).
        let split = match (&seg, self.ticket) {
            (Some(sg), true) => Some(self.ticket_split(sg, kext / bk, (bm, bn))),
            _ => None,
        };
        let passes: &[bool] = if split.is_some() {
            &[false, true]
        } else {
            &[false]
        };
        for &part in passes {
            if split.is_some() {
                self.out
                    .push_str(if part { "st_part:\n" } else { "st_whole:\n" });
            }
            self.part = part;
            // The workspace cell of tile-local `(lr, lc)` in the piece's slot.
            let ws_at = |k: &mut Self, r: &str, lc: &str| -> String {
                let sp = split.as_ref().expect("a shared tile's slot");
                let lr = k.tmp();
                k.line(format!("{lr} = sub {} {r}, {row0}", k.idx));
                let lrb = k.op("mul", &lr, bn);
                let l = k.op("add", &lrb, lc);
                k.op("add", &sp.base, &l)
            };
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
                        if part {
                            let lc = self.plus(b_frag0.clone(), j);
                            let wi = ws_at(self, &r, &lc);
                            self.store_o(&[x], &wi);
                            continue;
                        }
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
                        if self.epi.is_some() && !self.part {
                            self.store_cells(&[x], &oi);
                            continue;
                        }
                        self.store_o(&[x], &oi);
                    }
                }
            } else {
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
                        let mut xs = Vec::new();
                        for j in 0..lanes {
                            let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                            xs.push(x);
                        }
                        if part {
                            let lc = self.plus(b_frag0.clone(), v * b_chunk);
                            let wi = ws_at(self, &r, &lc);
                            self.store_o(&xs, &wi);
                            continue;
                        }
                        // A digit-affine output (`tile-post` on a radix site): its
                        // cell index is its address, `lanes` consecutive cells
                        // (in slab `z` of a split's workspace, as `c0`).
                        let oi = match (ro, &rrow) {
                            (Some(o), Some(r)) => {
                                let cc = self.op("add", &col0, &b_frag0);
                                let cc = self.op("add", &cc, v * b_chunk);
                                let a = self.radix_at(o, r, &cc);
                                self.slab(a)
                            }
                            _ => self.op("add", &c0, v * b_chunk),
                        };
                        if self.epi.is_some() && !self.part {
                            self.store_cells(&xs, &oi);
                            continue;
                        }
                        self.store_o(&xs, &oi);
                    }
                }
            }
            if let Some(sp) = split.as_ref() {
                if part {
                    self.ticket(sp, (bm, bn, threads), (&row0, &col0, &tid));
                }
                self.line("br label %st_join");
            }
        }
        self.part = false;
        if split.is_some() {
            self.out.push_str("st_join:\n");
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
            tile,
            bx,
            by,
            kbeg,
            kend,
            next,
            end,
        }
    }

    /// **sweep-runtime-2 `streamk-epi`: is the piece its tile's whole
    /// fold?** The walk names the blocks that share tile `t`: `c₀ = ⌊t·P /
    /// iters⌋ … c₁ = ⌊((t+1)·P − 1) / iters⌋`. One block (`c₀ = c₁`): the
    /// piece is the tile, and its sums are whole in registers. Shared: the
    /// tile's workspace slot is `c₀` — a block starts at most one tile it
    /// does not finish, so the slots of shared tiles are distinct and the
    /// workspace is `ctas` tiles, not the output. Branches to `st_whole` /
    /// `st_part`.
    fn ticket_split(&mut self, sg: &Segment, panels: u64, (bm, bn): (u64, u64)) -> TicketSplit {
        let ix = self.idx;
        let iters = self.streamk.expect("a stream-K kernel");
        let first = self.op("mul", &sg.tile, panels);
        let c0 = self.op("udiv", &first, iters);
        let lastu = self.plus(first, panels - 1);
        let c1 = self.op("udiv", &lastu, iters);
        let shared = self.tmp();
        self.line(format!("{shared} = sub {ix} {c1}, {c0}"));
        let whole = self.tmp();
        self.line(format!("{whole} = icmp eq {ix} {c1}, {c0}"));
        let base = self.op("mul", &c0, bm * bn);
        self.line(format!("br i1 {whole}, label %st_whole, label %st_part"));
        let ctas = (self.raster.0 * self.gy_out * panels).div_ceil(iters);
        TicketSplit {
            base,
            slot: c0,
            shared,
            ctas,
        }
    }

    /// **sweep-runtime-2 `streamk-epi`: the shared tile's ticket.** After a
    /// piece of a shared tile has added its partials into the tile's
    /// workspace slot (`%w`, zeroed by the runtime), its block takes one
    /// ticket: release fence, barrier, thread 0's `atomicrmw add` (device
    /// scope, acq_rel) on the slot's `i32` past the slots, the old count
    /// broadcast through shared memory. The block that draws the last ticket
    /// (`c₁ − c₀`, [`Kernel::ticket_split`]) holds the whole sums: after an
    /// acquire fence it reads the slot's `bm × bn` cells (volatile: from L2,
    /// not a stale L1 line), `lanes` consecutive cells a thread a step, and
    /// evaluates the group's members on them ([`Kernel::store_cells`], the
    /// ordered kernel's store). Every other block goes on to its next piece.
    /// The tiled kernel's tiles divide the site, so no cell is guarded.
    fn ticket(
        &mut self,
        sp: &TicketSplit,
        (bm, bn, threads): (u64, u64, u64),
        (row0, col0, tid): (&str, &str, &str),
    ) {
        let ix = self.idx;
        let s = self.site;
        let (t, e) = (self.t.clone(), elem_bytes_of(&s.elem));
        let base = (sp.ctas * bm * bn * e).div_ceil(GPU_VEC_BYTES) * GPU_VEC_BYTES / 4;
        self.line("; streamk-epi: the tile's ticket");
        self.line("fence syncscope(\"device\") release");
        self.line("call void @llvm.nvvm.barrier0()");
        let lead = self.tmp();
        self.line(format!("{lead} = icmp eq {ix} {tid}, 0"));
        self.line(format!("br i1 {lead}, label %tk_draw, label %tk_join"));
        self.out.push_str("tk_draw:\n");
        let slot64 = if ix == "i32" {
            let z = self.tmp();
            self.line(format!("{z} = zext nneg i32 {} to i64", sp.slot));
            z
        } else {
            sp.slot.clone()
        };
        let ti = self.tmp();
        self.line(format!("{ti} = add nuw nsw i64 {slot64}, {base}"));
        let tp = self.tmp();
        self.line(format!(
            "{tp} = getelementptr inbounds i32, ptr addrspace(1) %w, i64 {ti}"
        ));
        let old = self.tmp();
        self.line(format!(
            "{old} = atomicrmw add ptr addrspace(1) {tp}, i32 1 syncscope(\"device\") acq_rel, align 4"
        ));
        self.line(format!(
            "store i32 {old}, ptr addrspace(3) @mapal_smem, align 4"
        ));
        self.line("br label %tk_join");
        self.out.push_str("tk_join:\n");
        self.line("call void @llvm.nvvm.barrier0()");
        let drawn = self.tmp();
        self.line(format!(
            "{drawn} = load i32, ptr addrspace(3) @mapal_smem, align 4"
        ));
        // The next piece's panels overwrite the slot only after every
        // thread has read it.
        self.line("call void @llvm.nvvm.barrier0()");
        let shared32 = if ix == "i32" {
            sp.shared.clone()
        } else {
            let z = self.tmp();
            self.line(format!("{z} = trunc i64 {} to i32", sp.shared));
            z
        };
        let last = self.tmp();
        self.line(format!("{last} = icmp eq i32 {drawn}, {shared32}"));
        self.line(format!("br i1 {last}, label %tk_epi, label %tk_end"));
        self.out.push_str("tk_epi:\n");
        self.line("fence syncscope(\"device\") acquire");
        self.line("br label %tk_cell");
        let lanes = GPU_VEC_BYTES / e;
        let lanes = match bn % lanes == 0 && (bm * bn) % (threads * lanes) == 0 {
            true => lanes,
            false => 1,
        };
        let steps = bm * bn / (threads * lanes);
        self.out.push_str("tk_cell:\n");
        let (j, jn) = (self.tmp(), self.tmp());
        self.line(format!("{j} = phi {ix} [0, %tk_epi], [{jn}, %tk_next]"));
        let q0 = self.op("mul", &j, threads);
        let q1 = self.op("add", &q0, tid);
        let q = self.op("mul", &q1, lanes);
        let r = self.op("udiv", &q, bn);
        let cc = self.op("urem", &q, bn);
        let gr = self.op("add", row0, &r);
        let gc = self.op("add", col0, &cc);
        let oi = match s.radix.as_deref().and_then(|x| x.out.as_ref()) {
            Some(o) => self.radix_at(o, &gr, &gc),
            None => {
                let rr = self.op("mul", &gr, s.c);
                self.op("add", &rr, &gc)
            }
        };
        let wi = self.op("add", &sp.base, &q);
        let p = self.gep(1, "%w", &wi);
        let xs: Vec<String> = if lanes == 1 {
            let x = self.tmp();
            self.line(format!(
                "{x} = load volatile {t}, ptr addrspace(1) {p}, align {e}"
            ));
            vec![x]
        } else {
            let vt = format!("<{lanes} x {t}>");
            let v = self.tmp();
            self.line(format!(
                "{v} = load volatile {vt}, ptr addrspace(1) {p}, align {}",
                lanes * e
            ));
            (0..lanes)
                .map(|l| {
                    let x = self.tmp();
                    self.line(format!("{x} = extractelement {vt} {v}, i64 {l}"));
                    x
                })
                .collect()
        };
        self.store_cells(&xs, &oi);
        self.line("br label %tk_next");
        self.out.push_str("tk_next:\n");
        self.line(format!("{jn} = add {ix} {j}, 1"));
        let more = self.tmp();
        self.line(format!("{more} = icmp ult {ix} {jn}, {steps}"));
        self.line(format!("br i1 {more}, label %tk_cell, label %tk_end"));
        self.out.push_str("tk_end:\n");
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

/// sweep-runtime-2 `streamk-epi`: a shared tile's workspace slot
/// ([`Kernel::ticket_split`]).
struct TicketSplit {
    /// The slot's first workspace cell (`c₀ · bm · bn`).
    base: String,
    /// The slot (`c₀`, the tile's first block): its ticket's index.
    slot: String,
    /// The tile's blocks less one: the last ticket's number.
    shared: String,
    /// The walk's blocks: the workspace's slots.
    ctas: u64,
}

/// One stream-K piece's values ([`Kernel::segment`]).
struct Segment {
    /// The piece's tile, in the walk's linear order (before `raster`).
    tile: String,
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
                self.dallocs.push((slot.clone(), bytes, 0));
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
    /// candidates — but for a `tile-post` group whose one candidate splits
    /// (`group_order`: a workspace whose reduce evaluates the epilogue): the
    /// group's plain launch is its ordered kernel, so the split reaches the
    /// device only through the table (the f8e4m3 decode's `wo` projection, a
    /// scale and a residual after the fold: its one candidate `splitk16`
    /// ran as 16 one-cell blocks, 108 against 39 µs a step for its bf16
    /// twin's two-candidate table). A one-candidate table never trials.
    fn make_table(
        &self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        cands: &[SiteCand],
        one: &TileSite,
        epi: Option<&EpiKernel>,
    ) -> Option<SiteTable> {
        // One candidate launches as the rules' kernel — except an epilogue
        // group's split (`group_order`): the single launch keeps the group
        // ordered, so its realized combine is the table's (cross-arch
        // 3a8e654f: the B1 decode's O projection, S 64 planned on the H100,
        // ran its ordered 16-block kernel, 235 against 29 us). `post-table`
        // (OFF: the ordered kernel, the emission before); `--cands=1` keeps
        // it — candidate 0 is the rules' plan, and this is its launch.
        let split_epi = host_rungs(self.profile, self.contract, self.gpu_arms).post_table
            && epi.is_some()
            && cands
                .first()
                .is_some_and(|c| c.launch.combine != Combine::None);
        if cands.len() < 2 && !split_epi {
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
            .map(|(c, &id)| recipe(one, &c.launch, id, epi.is_some()))
            .unzip();
        Some(SiteTable {
            site: ids[0].0,
            flags: if cands.len() > 1 && self.rerun_safe(source, target, site, m) {
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
                let (arms, order) =
                    group_order(epi.as_ref(), site, self.profile, self.gpu_arms, self.order);
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
                self.dallocs.push((slot.clone(), bytes, 0));
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
            // re-sizes the tile for a split (`fill-wave`) or a stream-K walk
            // (`k-fill`'s full tile) this kernel does not take. (A stream-K
            // combine counts one split: an ordered group kernel at a `k-fill`
            // site launched the full tile's grid and block over its shrunk
            // kernel — sweep-runtime-2's generated groups, `--cands=1`.)
            let l = if l.combine != Combine::None {
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
        // Each operand's bytes in its own storage (a narrow operand's 2-byte
        // elements, ADR-0043); the output's are `elem`'s.
        let (ea, eb) = (site.a.elem_bytes(&site.elem), site.b.elem_bytes(&site.elem));
        let bytes = |ty: &Ty, e: u64| {
            let (_, n) = array_parts(ty);
            n * e
        };
        let a_bytes = bytes(src_ty.component_ty(site.a.slot).expect("tile a array"), ea);
        let b_bytes = bytes(src_ty.component_ty(site.b.slot).expect("tile b array"), eb);
        let o_bytes = bytes(&self.obj_ty(target), elem_bytes_of(&site.elem));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        // long-prefill `uniform-base`: each operand copied from where its
        // captured numbers' terms move it (the copy ends where it did).
        let (mut a_ptr, mut b_ptr, mut a_bytes, mut b_bytes) =
            (a_ptr, b_ptr, a_bytes.to_string(), b_bytes.to_string());
        if let Some(r) = site.radix.as_deref() {
            for (ds, e, p, n) in [
                (&r.ua, ea, &mut a_ptr, &mut a_bytes),
                (&r.ub, eb, &mut b_ptr, &mut b_bytes),
            ] {
                if let Some(off) = self.uniform_bytes(source, ds, e) {
                    let (q, m) = (self.tmp(), self.tmp());
                    self.line(format!(
                        "{q} = getelementptr inbounds i8, ptr {p}, i64 {off}"
                    ));
                    self.line(format!("{m} = sub i64 {n}, {off}"));
                    (*p, *n) = (q, m);
                }
            }
        }
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

    /// long-prefill `uniform-base`: the bytes an operand's pointer moves by
    /// its captured numbers' terms `ds` ([`mapal_ir::uniform_at`]), each
    /// value loaded from `source` (non-negative where the site runs: the
    /// fact's range) — an `i64` operand, `None` without a term.
    pub(super) fn uniform_bytes(
        &mut self,
        source: ObjectId,
        ds: &[mapal_ir::Digit],
        elem_bytes: u64,
    ) -> Option<String> {
        let mut acc: Option<String> = None;
        for d in ds {
            let (t, v) = self
                .load_component(source, d.slot)
                .expect("a captured number loads");
            let w = self.tmp();
            self.line(uniform_to_i64(&w, &v, &t));
            let q = match d.stride {
                1 => w,
                s => self.op_i64("udiv", &w, s),
            };
            let r = match d.radix {
                0 => q,
                r => self.op_i64("urem", &q, r),
            };
            let term = self.op_i64("mul", &r, d.coef * elem_bytes);
            acc = Some(match acc {
                None => term,
                Some(a) => self.op_i64("add", &a, &term),
            });
        }
        acc
    }

    fn op_i64(&mut self, code: &str, x: &str, y: impl std::fmt::Display) -> String {
        let r = self.tmp();
        self.line(format!("{r} = {code} i64 {x}, {y}"));
        r
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
                        resident::EpiSrc::Uniform(_) => unreachable!("a member takes no uniform"),
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
        // `tile-post`: the site's body after its fold, slot by slot.
        let post = g.post.as_ref().map(|p| {
            let captures = match self.ir.morphism(m).expect("site").op {
                Operation::Map { captures, .. } => captures,
                _ => unreachable!("a tile site is a map"),
            };
            let in_ty = self.obj_ty(self.ir.func(p.body).expect("body").input);
            let arg = crate::ty::lower_body_input_ty(&in_ty, captures)
                .expect("a site body's input lowers");
            let fields = p
                .srcs
                .iter()
                .map(|s| match *s {
                    Some(resident::EpiSrc::Gmem(o)) => {
                        EpiArg::Cap(caps.iter().position(|&x| x == o).expect("an operand"))
                    }
                    Some(resident::EpiSrc::Index) => EpiArg::Index,
                    Some(resident::EpiSrc::Uniform(j)) => EpiArg::Uniform(j as usize),
                    Some(resident::EpiSrc::Value(_)) => unreachable!("no group value"),
                    None => EpiArg::Null,
                })
                .collect();
            let (ret, _) = array_parts(&self.obj_ty(self.ir.morphism(m).expect("site").target));
            EpiCall {
                callee: post_variant(&self.fnames[p.body]),
                arg,
                ret,
                fields,
                store: false,
            }
        });
        // `tile-post`: each uniform's type, from the site's source product.
        let source = self.ir.morphism(m).expect("site").source;
        let uniforms = g
            .uniforms()
            .into_iter()
            .map(|slot| {
                let t = self
                    .obj_ty(source)
                    .component_ty(slot)
                    .cloned()
                    .expect("a uniform");
                crate::ty::lower_ty(&t).expect("a number lowers")
            })
            .collect();
        Some(EpiKernel {
            store_out: g.store_out,
            caps: caps.len(),
            calls,
            post,
            uniforms,
        })
    }

    /// This function's grouped sites for the device unit, and the bodies
    /// their kernels call: `(callee, body, captured slots read at the cell)`
    /// — an empty slot list is the host's own text.
    #[allow(clippy::type_complexity)]
    pub(crate) fn epi_sites(
        &self,
    ) -> (
        Vec<EpiSite>,
        Vec<(String, FuncId, Vec<u32>, Option<MorphismId>)>,
    ) {
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
                        post: None,
                        uniforms: Vec::new(),
                    },
                    terms: g.terms.len() as u64 + 1,
                    s: g.split,
                });
                continue;
            }
            let k = self.epi_kernel(m).expect("a group");
            if let (Some(p), Some(c)) = (&g.post, &k.post) {
                bodies.push((c.callee.clone(), p.body, Vec::new(), Some(p.fold)));
            }
            for (mb, c) in g.chain.iter().zip(&k.calls) {
                let slots = c
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| matches!(a, EpiArg::Cell(_)))
                    .map(|(s, _)| s as u32)
                    .collect();
                bodies.push((c.callee.clone(), mb.body, slots, None));
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
            let bytes_o = one.rows * one.c * elem_bytes_of(&one.elem);
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
                    self.dallocs.push((slot.clone(), s * bytes_o, 0));
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
                    let (rgx, rblock) = cell_grid(one.rows * one.c);
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
            narrow: None,
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
            half: None,
            post: None,
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
                            ticket: false,
                            part: false,
                            p0: None,
                            head: "entry",
                            tmap: Vec::new(),
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
                let wait = pdl_part(p, true, arms);
                assert_eq!(wait, enabled && p.gpu().unwrap().has_griddepcontrol());
                let host = init_call(host_rungs(p, true, arms), false, wait, false);
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

    /// sweep-gemm-2 `streamk-price` on the rtx-5080's under-filled sites (the
    /// gen_tiles layouts): loopmm's 1024³ (128 tiles of 64 panels on 168
    /// slots) streams; 1024 × 1024 × 256 (16 panels a tile: the pieces'
    /// partials and the zero pass outweigh the three panels the walk saves)
    /// and the 512³ cells (32 panels of a 64 × 32 tile) do not; the OFF arm
    /// and a part without the rates (`DeviceCost` zero) keep the classes'
    /// rule (no under-filled stream-K).
    #[test]
    fn streamk_price_separates_the_measured_sites() {
        use crate::profile::{A100, H100, H100_PCIE, RTX_3090, RTX_4090, RTX_5080, RTX_5090};
        let free = |s: &TileSite, p: &TargetProfile, arms| {
            launch_in(s, p, true, arms, None, Order::Free).combine
        };
        let arms = crate::GpuArms::default();
        let mut unpriced = crate::GpuArms::default();
        *unpriced.off.rung("streamk-price").expect("a rung") = true;
        let sk = |c: Combine| matches!(c, Combine::StreamK { .. });
        let loopmm = site(1024, 1024, 1024, true, true, Ty::f32());
        assert!(sk(free(&loopmm, &RTX_5080, arms)), "loopmm streams");
        assert!(
            !sk(free(&loopmm, &RTX_5080, unpriced)),
            "OFF: the class's rule"
        );
        for (m, n, k) in [(1024, 1024, 256), (512, 512, 512)] {
            let s = site(m, n, k, true, true, Ty::f32());
            assert!(!sk(free(&s, &RTX_5080, arms)), "{m}x{n}x{k}");
        }
        // review-fixes: a part that measured none of the three rates is
        // priced on the ones its properties derive (`Gpu::plan_cost`) —
        // never left unpriced (the OFF arm is the class's rule); the 512³
        // cells' 8 panels a tile refuse there as on the 5080.
        for p in [&A100, &H100, &H100_PCIE, &RTX_3090] {
            let g = p.gpu().expect("gpu");
            assert!(
                g.cost
                    .is_none_or(|d| d.atomic_ops_per_ns == 0 || d.zero_ns == 0),
                "{}: unmeasured",
                p.name
            );
            let d = g.plan_cost();
            assert!(
                d.flops_per_ns > 0 && d.atomic_ops_per_ns > 0 && d.zero_ns > 0,
                "{}: {d:?}",
                p.name
            );
            // The FMA tile (the A100's contract face takes TF32x3 there).
            let mut fma = arms;
            *fma.off.rung("split-tile").expect("a rung") = true;
            let l = launch_in(&loopmm, p, true, fma, None, Order::Free);
            let t = l.tiled.expect("an FMA tile");
            let wave = l.wave(g);
            let blocks = l.gx * l.gy;
            assert!(
                streamk_priced(p, &loopmm, &t, wave, blocks, wave, 1, 1, 1).is_some(),
                "{}: priced",
                p.name
            );
            assert!(
                !sk(free(&loopmm, p, unpriced)),
                "{}: OFF the class's rule",
                p.name
            );
            let s = site(512, 512, 512, true, true, Ty::f32());
            assert!(!sk(free(&s, p, arms)), "{}: 512³", p.name);
        }
        // The measured parts' rates are their own.
        for p in [&RTX_5080, &RTX_4090, &RTX_5090] {
            let g = p.gpu().expect("gpu");
            let (d, m) = (g.plan_cost(), g.cost.expect("measured"));
            assert_eq!(
                (d.flops_per_ns, d.atomic_ops_per_ns, d.zero_ns),
                (m.flops_per_ns, m.atomic_ops_per_ns, m.zero_ns)
            );
        }
    }

    /// sweep-gemm-2 `warm-zero`: the init flag is set exactly for a module
    /// whose launches zero an output — a direct zero call or a site table row
    /// with zero bytes — and the rung is on.
    #[test]
    fn warm_zero_flag_follows_the_modules_zeros() {
        let row = |zero: u64| {
            format!(
                "%mapal_cand {{ i64 7, i64 0, i64 {zero}, i64 0, i32 4, i32 32, i32 256, i32 0, i32 0, i32 0 }}"
            )
        };
        let table = |zeros: &[u64]| {
            let rows: Vec<String> = zeros.iter().map(|&z| row(z)).collect();
            format!(
                "@f.cands1 = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
                zeros.len(),
                rows.join(", ")
            )
        };
        assert!(module_zeroes(&table(&[16384, 0])));
        assert!(module_zeroes(&table(&[0, 4096])));
        assert!(!module_zeroes(&table(&[0, 0])));
        // `streamk-epi`: a workspace without a reduce kernel is zeroed by the
        // runtime (integration 3: the merge of warm-zero with the ticket); a
        // split's workspace with its reduce kernel is not.
        let ws = |rid: i64, ws: u64| {
            format!(
                "@g.cands1 = private unnamed_addr constant [1 x %mapal_cand] [%mapal_cand {{ i64 7, i64 {rid}, \
                 i64 0, i64 {ws}, i32 84, i32 1, i32 256, i32 0, i32 0, i32 0 }}]\n"
            )
        };
        assert!(module_zeroes(&ws(0, 1 << 20)));
        assert!(!module_zeroes(&ws(-5, 1 << 20)));
        assert!(!module_zeroes(&ws(0, 0)));
        assert!(module_zeroes(
            "  call void @mapal_gpu_zero(i64 %d3, i64 65536)\n"
        ));
        assert!(!module_zeroes(
            "  call void @mapal_gpu_launchv(i64 1, i32 0)\n"
        ));
        let flags = |rungs: GpuRungs, zeroes: bool| -> u32 {
            init_call(rungs, false, false, zeroes)
                .split("i32 ")
                .nth(1)
                .and_then(|s| s.split(')').next())
                .and_then(|s| s.parse().ok())
                .expect("a flag word")
        };
        assert_ne!(flags(GpuRungs::ALL, true) & GPU_INIT_WARM_ZERO, 0);
        assert_eq!(flags(GpuRungs::ALL, false) & GPU_INIT_WARM_ZERO, 0);
        let off = GpuRungs {
            warm_zero: false,
            ..GpuRungs::ALL
        };
        assert_eq!(flags(off, true) & GPU_INIT_WARM_ZERO, 0);
        // Every other bit is the rung-off module's.
        assert_eq!(
            flags(GpuRungs::ALL, true) & !GPU_INIT_WARM_ZERO,
            flags(off, true)
        );
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
                                        lane_maps(&s, &g, g.kmajor, false).into_iter().enumerate()
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
    fn mma_pieces_on_the_measured_cells() {
        // f32-tensor-cores `mma-combine` on the rtx-5080's TF32 cells (ABBA,
        // `f32-tensor-cores.md`): the measured best cut where one won, none
        // where every cut lost; the OFF arm never cuts. bugs-properties
        // `cut-price`: 2048 × 4096 × 4096 takes the measured S = 4 (1.3346
        // against 1.4179 ms, `rung_replay.tsv`), priced with its partials;
        // the 2048³ MLP GEMM keeps its measured S = 8 (its adds in the L2).
        for (m, n, k, best) in [
            (128, 6144, 4096, 8),
            (128, 4096, 14336, 8),
            (128, 4096, 4096, 1),
            (128, 28672, 4096, 1),
            (2048, 4096, 4096, 4),
            (2048, 4096, 14336, 8),
            (2048, 6144, 4096, 1),
            (2048, 2048, 2048, 8),
        ] {
            let s = site(m, n, k, true, true, Ty::f32());
            let at = |arms| {
                launch_in(
                    &s,
                    &crate::profile::RTX_5080,
                    false,
                    arms,
                    Some(Reduced::Tf32),
                    Order::Free,
                )
                .combine
            };
            let on = at(crate::GpuArms::default());
            assert_eq!(on.splits(), best, "{m}x{n}x{k}: {on:?}");
            assert!(matches!(on, Combine::Atomic { .. } | Combine::None));
            let off = at(off(&["mma-combine"]));
            assert_eq!(off, Combine::None, "{m}x{n}x{k}");
        }
    }

    /// bugs-properties `cut-price`: **every measured `mma-combine` arm of
    /// cross-arch 5fcb401b, replayed** (`analysis/arms.md`, same-box OFF /
    /// ON, the arm's plan line: its tile, grid and cut `S`) — the priced cut
    /// ([`MmaCut::ns`]) must keep the measured side: cut where ON won, whole
    /// where it lost (a null either way). The wave count alone cut all 20
    /// (9 lost). Open: the 4090's 128 × 6144 × 4096 (bf16 / TF32, priced
    /// 1.18 / 1.14 and cut as before: its `b` stream at the DRAM rate bounds
    /// both arms, 55 / 105 µs against 44 / 83 µs of waves; a DRAM floor in
    /// the price misprices the A100's under-filled 128 × 4096 × 14336, whose
    /// uncut kernel runs at 2.8× its peak estimate — the tensor tile's
    /// efficiency at its occupancy is the missing fact) — `OPEN` below.
    #[test]
    fn mma_cut_replay() {
        use crate::profile::{A100, RTX_3090, RTX_5080};
        // part, m, n, k, tf32 (else bf16), mt, nt, warps_x, warps_y, bk,
        // stages, blocks an SM, S, OFF / ON.
        let arms: [(&TargetProfile, u64, u64, u64, bool, [u64; 8], f64); 21] = [
            (
                &RTX_3090,
                2048,
                2048,
                2048,
                true,
                [4, 8, 2, 2, 16, 2, 2, 8],
                0.946,
            ),
            (
                &RTX_3090,
                2048,
                2048,
                2048,
                true,
                [4, 8, 2, 2, 16, 2, 2, 8],
                0.943,
            ),
            (
                &RTX_3090,
                128,
                4096,
                14336,
                false,
                [2, 2, 4, 2, 64, 2, 2, 8],
                0.928,
            ),
            (
                &RTX_3090,
                128,
                4096,
                14336,
                true,
                [2, 2, 4, 2, 32, 2, 2, 8],
                1.074,
            ),
            (
                &RTX_3090,
                128,
                6144,
                4096,
                false,
                [4, 4, 2, 2, 16, 2, 4, 8],
                1.273,
            ),
            (
                &RTX_3090,
                128,
                6144,
                4096,
                true,
                [4, 4, 2, 2, 8, 2, 4, 8],
                1.494,
            ),
            (
                &A100,
                2048,
                2048,
                2048,
                true,
                [4, 8, 2, 2, 32, 2, 2, 4],
                0.836,
            ),
            (
                &A100,
                2048,
                2048,
                2048,
                true,
                [4, 8, 2, 2, 32, 2, 2, 4],
                0.914,
            ),
            (
                &A100,
                128,
                4096,
                14336,
                false,
                [2, 2, 4, 2, 128, 2, 2, 8],
                1.106,
            ),
            (
                &A100,
                128,
                4096,
                14336,
                true,
                [2, 2, 4, 2, 64, 2, 2, 8],
                1.342,
            ),
            (
                &A100,
                2048,
                4096,
                14336,
                false,
                [4, 8, 2, 2, 64, 2, 2, 2],
                1.090,
            ),
            (
                &A100,
                2048,
                4096,
                14336,
                true,
                [4, 8, 2, 2, 32, 2, 2, 8],
                0.956,
            ),
            (
                &RTX_4090,
                128,
                4096,
                14336,
                false,
                [2, 2, 4, 2, 64, 2, 2, 2],
                1.087,
            ),
            (
                &RTX_4090,
                128,
                4096,
                14336,
                true,
                [2, 2, 4, 2, 32, 2, 2, 2],
                1.020,
            ),
            (
                &RTX_4090,
                128,
                6144,
                4096,
                false,
                [2, 2, 4, 2, 64, 2, 2, 4],
                0.864,
            ),
            (
                &RTX_4090,
                128,
                6144,
                4096,
                true,
                [2, 2, 4, 2, 32, 2, 2, 4],
                0.922,
            ),
            (
                &H100,
                128,
                6144,
                4096,
                false,
                [2, 2, 4, 2, 128, 3, 2, 4],
                0.899,
            ),
            // The rtx-5080's (f32-tensor-cores.md, ABBA; ON / OFF inverted).
            (
                &RTX_5080,
                2048,
                2048,
                2048,
                true,
                [4, 8, 2, 2, 16, 2, 2, 8],
                1.0 / 0.945,
            ),
            (
                &RTX_5080,
                128,
                6144,
                4096,
                true,
                [2, 4, 4, 2, 16, 3, 2, 8],
                0.2389 / 0.1815,
            ),
            (
                &RTX_5080,
                128,
                4096,
                14336,
                true,
                [2, 2, 4, 2, 32, 2, 2, 8],
                0.4541 / 0.4163,
            ),
            (
                &RTX_5080,
                2048,
                4096,
                14336,
                true,
                [4, 8, 2, 2, 16, 2, 2, 8],
                4.90 / 4.43,
            ),
        ];
        const OPEN: [(&str, u64, u64, bool); 2] = [
            ("rtx-4090", 6144, 4096, true),
            ("rtx-4090", 6144, 4096, false),
        ];
        let (mut kept, mut open) = (0, 0);
        for (p, m, n, k, tf32, [mt, nt, wx, wy, bk, stages, bps, s], meas) in arms {
            let g = p.gpu().expect("gpu");
            let fmt = if tf32 { MmaFmt::Tf32 } else { MmaFmt::Bf16 };
            let op = g.caps.matrix_op(fmt, Home::Regs).expect("a warp product");
            let (bm, bn) = (MMA_M * mt * wy, MMA_N * nt * wx);
            let cut = MmaCut {
                blocks: (m / bm) * (n / bn),
                slots: g.sms * bps,
                panels: k / bk,
                stages,
                panel_flops: 2 * bm * bn * bk,
                blocks_per_sm: bps,
                cells: m * n,
                bytes: m * k * 4 + k * n * if tf32 { 4 } else { 2 } + m * n * 4,
            };
            let rate = g.matrix_flops_per_ns(&op);
            let priced = cut.ns(g, rate, 1) / cut.ns(g, rate, s);
            let cuts = priced * MMA_CUT_PM as f64 >= 1000.0;
            eprintln!(
                "{} {m}x{n}x{k} {} S={s}: priced {priced:.2}, measured {meas:.3}",
                p.name,
                if tf32 { "tf32" } else { "bf16" }
            );
            if (meas - 1.0).abs() <= 0.03 {
                continue;
            }
            if OPEN.contains(&(p.name, n, k, tf32)) && m == 128 {
                assert_ne!(cuts, meas > 1.0, "{} {m}x{n}x{k}: no longer open", p.name);
                open += 1;
                continue;
            }
            assert_eq!(
                cuts,
                meas > 1.0,
                "{} {m}x{n}x{k}: priced {priced:.2}",
                p.name
            );
            kept += 1;
        }
        assert_eq!((kept, open), (18, 2));
    }

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
        // would take the tail; its measured row (Filled) ships the grid —
        // where the grid fills the part. large-gemm `streamk-fill`: its 144
        // one-warp tiles give the part fewer than its 656 fill warps, so the
        // `fill-wave` split fills it (S = 7, 1008 blocks); the arm off, the
        // one-wave stream-K.
        let r3090 = &crate::profile::RTX_3090;
        let s = site(16, 18432, 7168, true, true, Ty::f32());
        let l = launch_in(&s, r3090, true, arms, None, Order::Free);
        assert_eq!(l.combine, Combine::Atomic { s: 7 }, "{:?}", l.combine);
        let l = launch_in(&s, r3090, true, off(&["streamk-fill"]), None, Order::Free);
        assert_eq!(l.combine.splits(), 1, "{:?}", l.combine);
        let s16 = site(16, 2048, 7168, true, true, Ty::f32());
        // S63 the shipped default per part (`GPU_RUNGS_OFF`, measured,
        // `benches/results/gpu/s63/default-float/`): each is a session arm
        // (site-geometry `fill-wave` off: the S63 chooser; its own arm is
        // `site_geometry.rs`'s; sweep-gemm-2 `streamk-price` postdates it, as
        // f32-tensor-cores `split-tile`: the H100's measured `mma.sync` rate
        // takes the 4096³ cell's tf32x3 tile, tensor-cores-2).
        let s63 = off(&[
            "fill-wave",
            "k-fill",
            "streamk-fill",
            "streamk-price",
            "split-tile",
        ]);
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
                        let mut over = [None; crate::profile::DeviceCost::NAMES.len()];
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
        // The FMA tile's launches: `split-tile` (tensor-cores-2: the H100's
        // measured TF32 rate takes 4096³) has its own read-back in
        // `tests/split_tile.rs`.
        let mut arms = crate::GpuArms::default();
        *arms.off.rung("split-tile").expect("a rung") = true;
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
                    for (map, n) in lane_maps(&s, &g, g.kmajor, false) {
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
            let [(map, n), _] = lane_maps(s, g, g.kmajor, false);
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
