//! **kernel-merging `calls`: a chain of dependent launches as one persistent
//! kernel.** Every launch of a sequence (a loop trip's advance cone, the
//! straight-line walk with each loop one step) ends where the next dependent
//! one starts: the device drains the grid, the host issues the next launch
//! (`issue_ns`), the device ramps it up. A chain of launches whose bytes only
//! the chain touches needs none of that: one co-resident grid runs every
//! launch's body as a device function over that launch's blocks (block `i` of
//! the launch on persistent block `i mod G`), and a grid barrier stands only
//! where the dependency fact orders two launches.
//!
//! **Levels.** The units are `horizontal`'s output in walk order — a single
//! launch, or a merged group (its members side by side). A unit's level is one
//! above the highest level of a chain member it conflicts with (`access_plan`:
//! RAW / WAR / WAW, lifted with residency aliases and carries); none: level 0.
//! Members of one level run side by side (their block ranges concatenated);
//! consecutive levels are separated by one grid barrier. A step between the
//! chain's first and last member that touches a member's bytes (a copy-out, an
//! upload, a launch that stays its own) closes the chain before it: the chain
//! launches at its last member, every earlier member's launch deferred there.
//! A retained site's row pass (a branch arm) stays at level 0 — its whole-site
//! arm launches in place, which only a member with no earlier producer in the
//! chain may. An atomic site (its candidates add into a zeroed output) joins
//! at level ≥ 1: its zero pass runs inside the kernel one level before it, as
//! grid-stride stores, where no member of that level touches the bytes.
//!
//! **Co-residency.** A grid barrier needs every block resident. The planned
//! grid is the persistent footprint's wave (`sms ×` blocks per SM at the
//! largest block, shared memory and each member's own register residency,
//! [`super::horizontal::merged_per_sm`]), capped at the largest level. The
//! runtime launches it with `cuLaunchCooperativeKernel` at the loaded
//! kernel's own occupancy (`cuOccupancyMaxActiveBlocksPerMultiprocessor`: the
//! ptxas register count), which the driver refuses rather than oversubscribe;
//! the bodies stride over their blocks, so any grid is correct.
//!
//! **The rule, from measured properties** (`DeviceCost::gsync_ns`,
//! `gsync_block_ps`, `dep_launch_ns`, `dep_launch_block_ps`, `pdl_launch_ns`,
//! `coop_launch_ns`; `benches/gpu/calib.cu gsync`; `coop_first_ns`; the
//! bodies' price [`Gpu::coop_body_ns`], `calib.cu body`): a chain of `k`
//! units over `L` levels forms when one cooperative launch of the persistent
//! grid plus `L − 1` grid barriers, the member bodies' change inside one
//! kernel and, for a chain outside every loop, the process's first
//! cooperative launch (`coop_first_ns`) cost the device no more than the `k`
//! dependent launches it replaces (each its own grid, programmatic where the
//! runtime launches them so), and no launch takes more rounds of the
//! persistent grid than it took waves alone (`occupancy`), and the members of
//! a level fit one round together (`waves`). The host then issues one launch
//! instead of `k`. Every part prices the bodies: its own measured cycles
//! (`coop_body_cycles`), else the rtx-5080's at its own clock (derived); the
//! boundaries alone under-priced every forced chain of the rtx-5080 that lost
//! by more than they say and re-formed the H100's smlp / firew_rowsum chains
//! that lost 1.55 / 2.3x (`review-fixes.md`, `cross-arch/README.md`). A part
//! without the measured barrier and launch costs forms no chain.
//! OFF: `--gpu-off=calls`.

use super::horizontal::{HzGroup, Refuse, Shape, merged_per_sm};
use super::resident::ResidentPlan;
use super::zero::Step;
use super::*;
use crate::profile::Gpu;
use mapal_ir::AccessPlan;

mod kernel;
#[cfg(test)]
mod tests;
mod walk;

pub(crate) use self::kernel::GSYNC;
pub(super) use self::kernel::{missing_decls, persistent_kernel};

/// The most pointers one persistent launch passes (`mapal_gpu.c`'s
/// cooperative entry and site tables).
pub(crate) const COOP_ARGS_MAX: usize = 32;

/// `mapal_gpu_site` flags bit 2: every candidate of the table is a persistent
/// kernel, launched cooperatively at most its row's `gx` blocks.
pub(crate) const SITE_COOP: u32 = 4;

/// **The persistent grid of `levels`** (each level's launches side by side):
/// its blocks, or why the launches do not fit one co-resident grid.
pub(crate) fn persistent_fit(levels: &[Vec<Shape>], gpu: &Gpu) -> Result<u64, Refuse> {
    let all: Vec<Shape> = levels.iter().flatten().copied().collect();
    if all.iter().map(|s| s.args).sum::<usize>() > COOP_ARGS_MAX {
        return Err(Refuse::Args);
    }
    let b = all.iter().map(|s| s.block).max().unwrap_or(1);
    if b > gpu.max_threads_per_block || all.iter().any(|s| s.block % gpu.warp != 0) {
        return Err(Refuse::Block);
    }
    if all.iter().map(|s| s.smem).max().unwrap_or(0) > gpu.smem_per_block {
        return Err(Refuse::Smem);
    }
    let per = merged_per_sm(&all, gpu);
    if per == 0 {
        return Err(Refuse::Occupancy);
    }
    let w = gpu.sms * per;
    if all
        .iter()
        .any(|s| s.blocks.div_ceil(w) > s.blocks.div_ceil(gpu.sms * s.per_sm.max(1)))
    {
        return Err(Refuse::Occupancy);
    }
    let last = |n: u64| n - w * (n.div_ceil(w).max(1) - 1);
    if levels
        .iter()
        .any(|l| l.len() > 1 && l.iter().map(|s| last(s.blocks)).sum::<u64>() > w)
    {
        return Err(Refuse::Waves);
    }
    let most = levels
        .iter()
        .map(|l| l.iter().map(|s| s.blocks).sum::<u64>())
        .max()
        .unwrap_or(1);
    Ok(most.clamp(1, w))
}

/// **The rule's prices** (module doc), ns of device time: `(persistent,
/// separate)` — one cooperative dependent launch of `grid` blocks
/// (`coop_launch_ns`), `barriers` grid barriers, the bodies' change inside
/// one kernel ([`Gpu::coop_body_ns`]: measured, or derived from the
/// rtx-5080's cycles at this part's clock) and its share of the process's
/// first cooperative launch (`coop_first_ns` over `launches`, the chain's
/// launches a call: 1 outside every loop, a counted loop's trips; 0 for a
/// loop no fact counts — no term), against one dependent launch per unit of `units` blocks
/// each, as the runtime issues them (`pdl`: programmatic dependent launches,
/// where the part has them). `None` on a part without the measured barrier
/// and launch costs.
///
/// open-rules: the bodies' change is a **per-round** fact. `rounds` is the
/// chain's widest level in rounds of the persistent grid (`⌈blocks / grid⌉`
/// of a level's members side by side). A chain every level of which fits
/// one round runs each body over its own blocks, as its launch did: `calib.cu
/// body`'s one-round chains measured −0.37 … +0.07 µs on the RTX 3090, A100,
/// RTX 4090 and H100 (60f0f8ce) and +0.14 … +0.64 on the RTX 5080 — no
/// term. The grid-stride form (more than one round: fewer blocks in flight
/// than the members' own waves) pays [`Gpu::coop_body_ns`] (its 4-round
/// measurement). The boundaries are the part's measured ones, else derived
/// from its clock ([`Gpu::chain_cost`]): every part is priced.
///
/// open-rules: and **the phases' length** — `work_ns`, the members' work
/// at the part's peaks (each member's flops over the FFMA rate or its bytes
/// over the DRAM rate, whichever is longer: a lower bound of its phase).
/// A member body inside the persistent kernel runs without its launch's
/// read-only path and at the merged kernel's registers; that change grows
/// with the phase, not with the boundaries: [`CALLS_PHASE_PM`] of it.
///
/// calls-issue (plan-cost D2's `op_ns`): **the host's issue time** — the
/// separate form issues one launch per unit from the host, unit `k` at
/// `k · issue`, so a unit starts when the one before it is done and its
/// launch gap passed (`s_{k−1} + d_{k−1} + launch_at(k)`) or when it is
/// issued, whichever is later: the host runs ahead behind a long unit, and
/// the device idles only where the issues outrun the work (`d`: `unit_ns`,
/// its members' work at the part's peaks — [`CALLS_PHASE_PM`]'s `work_ns`
/// per unit). A per-boundary `max(launch, issue − d_prev)` (D2's `op_ns`
/// with no queue) over-formed: it charged the RTX 5080's causal chain
/// (units 7.1 µs, 1.1 µs, then three tiny maps) 1.7 µs a tiny boundary
/// while the host was 7 µs ahead, and the chain forced there measured
/// 1.106× (`open-rules.md`). The chain is one issue. `issue_ns` is the
/// host's measured property
/// ([`crate::profile::DeviceCost::issue_ns`]); `0` (a part without it, or
/// `--gpu-off=calls-issue`) is the price before it exactly, and the price is
/// non-decreasing in `issue_ns`: a slower host forms every chain a faster
/// one forms.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chain_ns(
    units: &[u64],
    grid: u64,
    barriers: u64,
    gpu: &Gpu,
    pdl: bool,
    launches: u64,
    rounds: u64,
    work_ns: f64,
    unit_ns: &[f64],
    issue_ns: f64,
) -> Option<(f64, f64)> {
    let (d, _) = gpu.chain_cost();
    let gaps = units
        .iter()
        .map(|&b| d.launch_at(b, gpu.sms, pdl))
        .collect::<Option<Vec<f64>>>()?;
    let mut sep: f64 = gaps.iter().sum();
    if issue_ns > 0.0 && unit_ns.len() == units.len() && !units.is_empty() {
        // The host issues unit `k` at `k · issue`; the device starts it when
        // the unit before it is done and its launch gap passed, or when it
        // is issued, whichever is later — the host runs ahead behind a long
        // unit and the device idles only where the issues outrun the work.
        let mut s = 0.0f64;
        for k in 1..units.len() {
            s = (s + unit_ns[k - 1] + gaps[k]).max(k as f64 * issue_ns);
        }
        let work: f64 = unit_ns[..units.len() - 1].iter().sum();
        sep = sep.max(gaps[0] + s - work);
    }
    // The process's first cooperative launch, shared by the chain's launches
    // in one call (0: a loop no fact counts — shared, no term).
    let first = match launches {
        0 => 0,
        n => d.coop_first_ns / n,
    };
    let body = if rounds > 1 { gpu.coop_body_ns() } else { 0.0 }
        + work_ns * CALLS_PHASE_PM as f64 / 1000.0;
    let per = d.coop_launch_at(grid, gpu.sms)?
        + barriers as f64 * d.gsync_at(grid, gpu.sms)?
        + body
        + first as f64;
    Some((per, sep))
}

/// open-rules `calls`: what a member's phase costs inside one persistent
/// kernel beyond its own launch, per mille of the phase — the smallest
/// proportional change a merged phase measured: the RTX 5080's rt_mlp2 GEMM
/// pairs, 166.5 / 165.3 µs against 161.0 / 163.7 µs as four kernels (+2.2 %,
/// 127–128 registers against 95–96; `review-fixes.md` §3). The others are
/// larger: the H100's smlp / firew_rowsum phases +59 / +84 % (7a2b67bf), the
/// A100's causal family +10 % (41.3 against 37.5 µs a trip), the 5080's
/// decode rows +19 %. On long phases (the H100's GEMM pairs, the 4090's
/// diamond) it outweighs the ~0.7 µs a boundary saves; on the causal and
/// decode chains' short ones it does not.
pub const CALLS_PHASE_PM: u64 = 22;

/// The rounds of the persistent grid of `grid` blocks the widest level of
/// `levels` takes (its members side by side).
pub(crate) fn chain_rounds(levels: &[Vec<Shape>], grid: u64) -> u64 {
    levels
        .iter()
        .map(|l| {
            l.iter()
                .map(|s| s.blocks)
                .sum::<u64>()
                .div_ceil(grid.max(1))
        })
        .max()
        .unwrap_or(1)
}

/// The chains the rule considered in one function, for the intent line.
#[derive(Clone, Debug, Default)]
pub(super) struct CallsStats {
    pub chains: usize,
    pub formed: usize,
    pub units: usize,
    pub barriers: usize,
    pub zeros: usize,
    /// Why a unit did not join the open chain, or a chain did not form.
    pub refused: std::collections::BTreeMap<&'static str, usize>,
    /// `(persistent, separate)` ns of the formed chains, summed.
    pub per_ns: f64,
    pub sep_ns: f64,
    /// The widest level of any priced chain, in rounds of its persistent
    /// grid ([`chain_rounds`]).
    pub rounds: u64,
    /// calls-issue: the host issue time the prices read (0: off), and each
    /// priced chain's `persistent:separate:unit ns,…` (evidence).
    pub issue_ns: f64,
    pub priced: Vec<String>,
}

/// A placed member: `(step, level, zero_ok, out)` — its step in the sequence,
/// its level, whether its zero pass runs in the kernel, its output.
type Placed = (usize, u32, bool, Option<ObjectId>);

/// One chain under construction: its units (step indices of their members),
/// each member's level and whether its zero pass may run in the kernel.
#[derive(Default)]
struct Chain {
    units: Vec<Vec<usize>>,
    /// `(step, level, zero_ok, out)` per member, in walk order.
    members: Vec<Placed>,
    /// `horizontal` groups the chain absorbs.
    absorbed: Vec<usize>,
}

impl Chain {
    fn levels(&self) -> u32 {
        self.members.iter().map(|m| m.1 + 1).max().unwrap_or(0)
    }
}

// ---- device ---------------------------------------------------------------------------------------
