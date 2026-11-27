//! **kernel-merging `horizontal`: independent launches side by side in one
//! grid.** Two device launches of one sequence (a loop trip's advance cone,
//! the straight-line walk with each loop one step) that the dependency fact
//! (`access_plan`, lifted with the backend's byte sharing: residency aliases,
//! carries, arena offsets, a zero tail's outputs) leaves unordered — no RAW /
//! WAR / WAW between them, and none between the earlier one and any step the
//! walk emits between them — need not be two launches: one grid whose blocks
//! are the concatenation of their grids runs both, a block dispatching on its
//! index to its launch's body (each kernel a device function of its own block
//! coordinates). The group launches at its last member; every earlier
//! member's launch is deferred there (its host half — allocations, pointer
//! arrays — stays at its own position), and the frees of the steps between
//! move after it.
//!
//! The rule, per sequence, greedily in walk order: a launch joins the open
//! group when it is independent of every member and the merged grid fits —
//! its arguments fit one launch (`LAUNCHV_MAX`), its block holds every
//! member's threads (the largest; a smaller member's extra warps exit and its
//! barriers count its own threads), its shared memory every member's (the
//! largest), and at the merged footprint (the largest block and shared memory,
//! each member's own residency — its registers — re-expressed in blocks of the
//! merged size) no member takes more waves than alone (`occupancy`) and the
//! members' last waves fit one wave together (`waves`: the S54 wave rule,
//! [`super::resident::waves_merge`], over `k` grids). Under both the merged
//! grid runs every full wave a member ran alone and one shared last wave —
//! never longer than the members back to back — and issues one launch instead
//! of `k`. A step that touches a member's bytes closes the group before it.
//!
//! Members: cell kernels (`devmap`; not a range cell, whose grid is the
//! trip's), units (`flow`), and tile sites whose rules' candidate is unsplit
//! — a table's unsplit candidates make the merged table's tuples, in order of
//! their ranks, at most `--cands` — including a retained site's row pass (one
//! arm of its trip's branch: the group merges when that arm ran, by a flag the
//! arm sets; otherwise every member launches alone at the group's point).
//! Host intent `; mapal.hz.host launches= groups= members= branchy=
//! refused=`, one `; mapal.hz kernel= block= smem= members=` line per merged
//! kernel (the device unit composes it from these). OFF:
//! `--gpu-off=horizontal`.

mod device;
mod emit;
mod plan;
#[cfg(test)]
mod tests;

pub(super) use self::device::Head;
#[cfg(test)]
pub(super) use self::device::parse_head;
pub(crate) use self::device::{HzSpec, compose, hz_members, hz_specs};
pub(crate) use self::emit::HzEmit;

use super::resident::ResidentPlan;
use super::zero::Step;
use super::*;
use crate::profile::Gpu;
use mapal_ir::AccessPlan;

/// One merged launch: its members (launching morphisms, in walk order; the
/// group launches at the last), whether each is a retained site's row pass
/// (`branchy`), and whether every member without a table may run twice
/// (`rerun`: it writes no byte it reads).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HzGroup {
    pub members: Vec<MorphismId>,
    pub branchy: Vec<bool>,
    pub rerun: bool,
    /// kernel-merging `calls`: each member's level in a persistent group
    /// (all 0: a `horizontal` group, one grid side by side).
    pub level: Vec<u32>,
    /// `calls`: a member whose atomic candidates the persistent kernel may
    /// take (its zero pass runs one level before it).
    pub zero_ok: Vec<bool>,
}

impl HzGroup {
    /// A `calls` group: one persistent kernel, grid barriers between levels.
    pub(crate) fn persistent(&self) -> bool {
        self.level.iter().any(|&l| l > 0)
    }
}

/// A launch the rule may place in a merged grid: its rules' geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub blocks: u64,
    pub block: u64,
    pub smem: u64,
    /// Blocks of this launch one SM holds alone (threads, shared memory,
    /// registers).
    pub per_sm: u64,
    /// Its kernel arguments.
    pub args: usize,
}

/// Why the rule declined a launch the open group could have taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refuse {
    Args,
    Block,
    Smem,
    Occupancy,
    Waves,
}

impl Refuse {
    const ALL: [Refuse; 5] = [
        Refuse::Args,
        Refuse::Block,
        Refuse::Smem,
        Refuse::Occupancy,
        Refuse::Waves,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Refuse::Args => "args",
            Refuse::Block => "block",
            Refuse::Smem => "smem",
            Refuse::Occupancy => "occupancy",
            Refuse::Waves => "waves",
        }
    }
}

/// Blocks of a merged grid one SM holds: the thread, block and shared-memory
/// limits at the largest block and shared memory, and each member's own
/// residency re-expressed in blocks of the merged size (its registers: a
/// member resident `per_sm` blocks of `block` threads alone keeps its
/// per-thread registers in the merged kernel, which holds the largest).
pub(crate) fn merged_per_sm(shapes: &[Shape], gpu: &Gpu) -> u64 {
    let b = shapes.iter().map(|s| s.block).max().unwrap_or(1).max(1);
    let smem = shapes.iter().map(|s| s.smem).max().unwrap_or(0);
    let threads = (gpu.max_threads_per_sm / b).min(gpu.max_blocks_per_sm);
    let shared = match smem {
        0 => u64::MAX,
        s => gpu.smem_per_sm / (s + gpu.smem_reserved_per_block),
    };
    // At least one: the merged kernel's `maxntid` caps its registers at what
    // one block of the merged size holds.
    let own = shapes
        .iter()
        .map(|s| (s.block * s.per_sm.max(1) / b).max(1))
        .min()
        .unwrap_or(u64::MAX);
    threads.min(shared).min(own)
}

/// **The rule** (module doc): the merged grid's wave (`sms × blocks/SM`), or
/// why it does not fit.
pub(crate) fn side_by_side(shapes: &[Shape], gpu: &Gpu) -> Result<u64, Refuse> {
    if shapes.iter().map(|s| s.args).sum::<usize>() > gpu::LAUNCHV_MAX {
        return Err(Refuse::Args);
    }
    let b = shapes.iter().map(|s| s.block).max().unwrap_or(1);
    if b > gpu.max_threads_per_block || shapes.iter().any(|s| s.block % gpu.warp != 0) {
        return Err(Refuse::Block);
    }
    if shapes.iter().map(|s| s.smem).max().unwrap_or(0) > gpu.smem_per_block {
        return Err(Refuse::Smem);
    }
    let per = merged_per_sm(shapes, gpu);
    if per == 0 {
        return Err(Refuse::Occupancy);
    }
    let w = gpu.sms * per;
    let more = shapes
        .iter()
        .any(|s| s.blocks.div_ceil(w) > s.blocks.div_ceil(gpu.sms * s.per_sm.max(1)));
    if more {
        return Err(Refuse::Occupancy);
    }
    let last = |n: u64| n - w * (n.div_ceil(w).max(1) - 1);
    if shapes.iter().map(|s| last(s.blocks)).sum::<u64>() > w {
        return Err(Refuse::Waves);
    }
    Ok(w)
}
