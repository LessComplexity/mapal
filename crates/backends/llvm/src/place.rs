//! **ADR-0050 C2: the GPU's translation into the machine model, and the
//! present emitters' lowering declarations.**
//!
//! - [`machine_of`]: a GPU [`TargetProfile`] as a [`mapal_place::Machine`] —
//!   the profile's physical properties as facts, each with its provenance
//!   ([`crate::profile::PlaceFacts`]); a missing fact takes a marked derived
//!   or nominal interval; no value is borrowed from another part as if
//!   measured and no table by version is read.
//! - The lowerings of the present emitters — the one-thread-per-cell kernel,
//!   the FMA tile, the warp `mma.sync` tile (TF32, 16-bit, split forms), the
//!   warpgroup `wgmma` tile and its warp-specialized form — each declaring
//!   its constraints, its footprint, its instruction counts per class per
//!   trip and its ring's in-flight discipline ([`mapal_place::Lowering`]),
//!   and the coordinates of its candidate space (D5 steps 2–3) as domains
//!   over the machine's facts.
//! - [`schedule_of`] / [`launch_of`]: the rules' [`Launch`] as a schedule
//!   and back. The rungs a launch carries beyond the site's table
//!   ([`crate::func::gpu::site_rungs`]) are the D6 steps a lowering keeps
//!   (`params` `rung+` / `rung-`), listed by the coverage test.

#![cfg_attr(not(test), allow(dead_code))]

mod emitter;
mod facts;
mod launch;
mod lowering;
mod machine;
mod search;
mod site;
#[cfg(test)]
mod tests;

pub use self::emitter::Emitter;
pub use self::facts::{DRAM, HOST, L2, OP_FFMA, OP_MULADD, QUERIED, REGS, SMEM, TMEM};
use self::facts::{
    E_DEVICE, E_GROUP, E_LANE, E_SUB, E_UNIT, eff_points, fact, fmt_of, geo, home_places, latency,
    measures, op_name, rate_field, row_cite, src,
};
pub(crate) use self::launch::table_rungs;
pub use self::launch::{FMA, MMA, NAIVE, NAIVE_REGS, WGMMA, WGMMA_WS, launch_of, schedule_of};
use self::launch::{SPLITS, param, smem_home};
#[allow(unused_imports)]
pub use self::machine::{FITS, machine_of};
use self::search::{LAYOUT, LSU, SEARCHING, memo_rungs};
#[allow(unused_imports)]
pub use self::search::{PLACE_BAND_PM, PLACE_NULL, lowering_of, place_opts};
#[allow(unused_imports)]
pub(crate) use self::search::{
    Placed, Probe, Seen, forced, place_line, placed, placed_for, probe, rules,
};
use self::site::ORDER;
use self::site::stores_col;
#[allow(unused_imports)]
pub use self::site::{ColScope, OrderScope, WarmScope, site_of, warm_of, warm_sites};

use crate::func::gpu::{
    self, Arithmetic, Combine, Launch, NAIVE_THREADS, cell_blocks, elem_bytes_of, fill_grid,
    site_rungs,
};
use crate::profile::{
    DeviceCost, Face, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, Home as MHome,
    MatrixOp, MmaFmt, NAIVE_UNROLL, NO_PLAN, TargetProfile, WARPGROUP_WARPS, WGMMA_N_MAX,
    WGMMA_N_STEP, fma_held_regs,
};
use mapal_ir::{TileSite, Ty};
use mapal_place::{
    Atomics, Banking, Bind, Cache, Class, Comb, Declared, Dispatch, EffPoint, Exec, ExecKind, Fact,
    Fmt, Fold, Home, InstanceMap, Keep, LaunchCost, LaunchMode, Layout, Level, Lowering, Machine,
    Major, Mgmt, Mover, MoverKind, NSet, Op, Pipe, Place, Policy, Prim, PrimOf, Reach, Refusal,
    Res, Resource, Schedule, Site, Src, Start, TravKind, Traversal,
};
