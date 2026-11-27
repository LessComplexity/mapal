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
    Depth, Face, GPU_RESIDENT_BLOCKS, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, Home,
    MMA_K, MMA_M, MMA_N, MMA_SYNC_16, MmaFmt, NAIVE_UNROLL, PlanComb, PlanCost, PlanShape,
    WARPGROUP_WARPS, WGMMA_ATOM_ROWS, WGMMA_K, WGMMA_M, WGMMA_SWIZZLE_BYTES, WsForm, bank_degree,
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
/// hopper-launch `tma-warm`, off: the runtime leaves the context's
/// tensor-copy path cold. On (the bit clear), a module whose text carries a
/// tensor-map recipe has the init thread launch the runtime's one-box TMA
/// kernel before the clock — the runtime reads that fact from the module text
/// it loads, so the flag word of a module without one is what it was.
pub(crate) const GPU_INIT_TMA_COLD: u32 = 64;
/// gaps-disasm C1 `zero-host`: the module zeroes outputs, by the runtime's
/// zero kernel (loaded at init) rather than the driver's memset API.
pub(crate) const GPU_INIT_ZERO_KERNEL: u32 = 128;

/// `main`'s first call in a module with a GPU site: hands the runtime its host
/// rungs. Off, the runtime does nothing until the first site, the S52 path.
/// `zeroes`: some launch of the module zeroes an output ([`module_zeroes`]).
/// `load-dispatch`: the module's site tables ([`site_registry`]) registered
/// first, so the init resolves them.
pub(crate) fn init_call(
    rungs: crate::GpuRungs,
    streams: bool,
    pdl: bool,
    zeroes: bool,
    sites: usize,
) -> String {
    let flags = u32::from(rungs.eager_init) * GPU_INIT_EAGER
        + u32::from(streams) * GPU_INIT_STREAMS
        + u32::from(pdl) * GPU_INIT_PDL
        + u32::from(rungs.preload) * GPU_INIT_PRELOAD
        + u32::from(rungs.stage && rungs.async_h2d) * GPU_INIT_STAGE
        + u32::from(rungs.warm_zero && zeroes) * GPU_INIT_WARM_ZERO
        + u32::from(!rungs.tma_warm) * GPU_INIT_TMA_COLD
        + u32::from(rungs.zero_host && zeroes) * GPU_INIT_ZERO_KERNEL;
    let reg = match (rungs.load_dispatch, sites) {
        (true, n) if n > 0 => format!("  call void @mapal_gpu_sites(ptr @mapal_sites, i32 {n})\n"),
        _ => String::new(),
    };
    format!("{reg}  call void @mapal_gpu_init(i32 {flags})\n")
}

/// gaps-disasm C1 `load-dispatch`: the module's site tables, one
/// `%mapal_site_reg { i64 site, ptr table, i32 n, i32 flags }` each (its
/// first `mapal_gpu_site` call's site and flags; a table is one constant
/// however many calls launch it), as the `@mapal_sites` constant and its
/// declarations — empty when the host text launches no table.
pub(crate) fn site_registry(host: &str) -> (usize, String) {
    let mut rows: Vec<String> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for call in host.split("@mapal_gpu_site(i64 ").skip(1) {
        let mut f = call.split(", ");
        let (Some(site), Some(flags), Some(_stream), Some(n), Some(table)) =
            (f.next(), f.next(), f.next(), f.next(), f.next())
        else {
            continue;
        };
        let Some(table) = table.strip_prefix("ptr ") else {
            continue;
        };
        if seen.contains(&table) {
            continue;
        }
        seen.push(table);
        rows.push(format!(
            "%mapal_site_reg {{ i64 {site}, ptr {table}, {n}, {flags} }}"
        ));
    }
    if rows.is_empty() {
        return (0, String::new());
    }
    (
        rows.len(),
        format!(
            "{SITES_DECL}@mapal_sites = private unnamed_addr constant [{} x %mapal_site_reg] [{}]\n",
            rows.len(),
            rows.join(", ")
        ),
    )
}

/// `load-dispatch`'s registry type and entry (`mapal_choice.h`'s
/// `struct mapal_site_reg`, `mapal_gpu.c`'s `mapal_gpu_sites`).
pub(crate) const SITES_DECL: &str = "%mapal_site_reg = type { i64, ptr, i32, i32 }\n\
     declare void @mapal_gpu_sites(ptr, i32)\n";

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
        || host.contains("call void @mapal_gpu_zero_for(")
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
pub(crate) fn site_rungs(
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
pub(crate) const S54_DECLS: [(&str, &str); 9] = [
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
    // gaps-disasm C1 `zero-hoist`: a hoisted zero, skipped where its site's
    // pick does not add into the output.
    (
        "@mapal_gpu_zero_for(",
        "declare void @mapal_gpu_zero_for(ptr, i64, i64)\n",
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

mod cand_plan;
mod cell_kernels;
mod combine_plan;
mod device_text;
mod emit_site;
mod epi_rowfold;
mod k_bulk;
mod k_core;
mod k_mma;
mod k_mma16;
mod k_naive;
mod k_split;
mod k_streamk;
mod k_tiled;
mod k_wgmma_ws;
mod kernel_geom;
mod launch_lines;
mod launch_rules;
mod mma_plan;
mod placement;
mod site_table;
mod split_plan;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod walk_arms;

pub(crate) use self::cand_plan::{launch_for, plan_cands, plan_shape};
use self::cand_plan::{plan_pick, resized};
pub(crate) use self::cell_kernels::{
    CapArg, CellGeo, CellKernel, cell_grid, cell_kernel_text, cell_variant, lanes_variant,
    launched_kernels, post_variant, slots_variant, trap_variant, uniform_from_i64, uniform_to_i64,
};
use self::cell_kernels::{DEVICE_TRAP, kernel_symbol, site_fold};
pub(crate) use self::combine_plan::{
    BULK_BAR_BYTES, bulk_stage, cell_blocks, naive_cells_admit, vec_fit, ws_tensors,
};
use self::combine_plan::{
    combine, kvec_b, kvec_swizzle, panel_regs, prefetch_depth, staged_fits, streaming, vec_takes,
    vector_a, vector_b,
};
pub(crate) use self::device_text::device_module;
use self::device_text::{Arith, kernel, kernel_text, reduce_kernel_text};
pub(crate) use self::epi_rowfold::{
    EpiArg, EpiCall, EpiKernel, EpiSite, GenKernel, cell_id, gen_id, kernel_name, rowfold_grid,
    rowfold_id, rowfold_lanes, warp_lanes,
};
use self::epi_rowfold::{gen_kernel_text, rowfold_entry};
use self::k_streamk::Segment;
use self::kernel_geom::{
    Frag, Kernel, LaneMap, affine_delta, frag_layout, kmajor_pad, lane_maps, panel_sa,
    raster_block, thread_coords,
};
pub(crate) use self::kernel_geom::{
    fma_constraints, fma_layout, fma_streams, fnv_text, kernel_module, lowered_text, lsu_per_panel,
    mma_widths, panel_excess,
};
use self::launch_lines::{KERNEL_RUNGS, k_fill, plan_line};
pub(crate) use self::launch_lines::{fill_grid, launch_in, launch_in_rules};
pub(crate) use self::launch_rules::{
    Arithmetic, Combine, Launch, hybrid_tag, launch, launch_split, zero_bytes,
};
use self::mma_plan::mma_combine;
pub(crate) use self::mma_plan::mma_walk;
use self::placement::{GEN_THREADS, b_k_major, radix_whole};
pub(crate) use self::placement::{
    NAIVE_THREADS, device_made, gpu_site, placement, row_group, tile_plan, uniform_site,
};
pub(crate) use self::site_table::{
    SITE_DECL, SiteCand, SiteTable, concat_plan, epi_order, epi_walk, fnv, group_order, site_cands,
};
use self::site_table::{SITE_RERUN, apart, cand_ids, cand_kernels, concat_kernel_text, recipe};
pub(crate) use self::split_plan::{
    BF16_RN_OVERFLOW, TF32_RNA_OVERFLOW, col_store, elem_bytes_of, kernel_id, kernel_id_col,
    kernel_id_split, kernel_id_with, site_reduced, split_flush, split_name, split_parts,
};
use self::split_plan::{
    half_operands, mma16_layout, narrow_streams, split_cost, split_issue, split_operands,
    split_pays,
};
#[cfg(test)]
pub(crate) use self::walk_arms::WALK_ARMS;

#[cfg(test)]
use self::combine_plan::streamk_priced;
#[cfg(test)]
use self::device_text::max_index;
#[cfg(test)]
use self::kernel_geom::Num;
#[cfg(test)]
use self::mma_plan::{MMA_CUT_PM, MMA_WALK_PM, MmaCut, MmaWalk};
#[cfg(test)]
use self::site_table::launch_key;
