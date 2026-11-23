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
/// rtx-5080's cycles at this part's clock) and, outside every loop (`looped`
/// false: the chain runs once), the process's first cooperative launch
/// (`coop_first_ns`), against one dependent launch per unit of `units` blocks
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn chain_ns(
    units: &[u64],
    grid: u64,
    barriers: u64,
    gpu: &Gpu,
    pdl: bool,
    looped: bool,
    rounds: u64,
    work_ns: f64,
) -> Option<(f64, f64)> {
    let (d, _) = gpu.chain_cost();
    let sep = units
        .iter()
        .map(|&b| d.launch_at(b, gpu.sms, pdl))
        .sum::<Option<f64>>()?;
    let first = if looped { 0 } else { d.coop_first_ns };
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

impl FnEmit<'_> {
    /// **`calls`** over one sequence (module doc): every chain the rule
    /// forms becomes one persistent group of `plan.hz` (the `horizontal`
    /// groups it absorbs marked in `dead`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn calls_walk(
        &self,
        plan: &mut ResidentPlan,
        seq: &[Step],
        acc: &AccessPlan,
        access: &[(Vec<ObjectId>, Vec<ObjectId>)],
        gated: &[(Vec<ObjectId>, Vec<ObjectId>)],
        dead: &mut Vec<bool>,
        stats: &mut CallsStats,
    ) {
        let Some(gpu) = self.profile.gpu() else {
            return;
        };
        let shapes: Vec<Option<Shape>> = seq
            .iter()
            .enumerate()
            .map(|(j, s)| match *s {
                Step::Op(m) => self.hz_shape(plan, m, true).filter(|_| {
                    !gated
                        .iter()
                        .any(|g| self.hz_conflict(plan, acc, g, &access[j]))
                }),
                Step::Loop(_) => None,
            })
            .collect();
        let op = |j: usize| match seq[j] {
            Step::Op(m) => Some(m),
            Step::Loop(_) => None,
        };
        let structural = |j: usize| {
            op(j).is_some_and(|m| {
                matches!(
                    self.ir.morphism(m).expect("m").op,
                    Operation::Pair { .. } | Operation::Proj { .. }
                )
            })
        };
        // A step that is a `horizontal` member: its group, whether it is the
        // group's last (where the group's launch stands), its members' steps.
        let group_of: Vec<Option<(usize, bool, Vec<usize>)>> = (0..seq.len())
            .map(|j| {
                let m = op(j)?;
                let &(g, k) = plan.hz_of.get(m)?;
                if g >= dead.len() || dead[g] {
                    return None;
                }
                let ms = &plan.hz[g].members;
                let steps = (0..seq.len())
                    .filter(|&i| op(i).is_some_and(|x| ms.contains(&x)))
                    .collect();
                Some((g, k + 1 == ms.len(), steps))
            })
            .collect();
        let mut chain = Chain::default();
        for j in 0..seq.len() {
            let unit: Option<(Vec<usize>, Option<usize>)> = match &group_of[j] {
                Some((_, false, _)) => continue,
                Some((g, true, steps)) => (steps.len() == plan.hz[*g].members.len()
                    && steps.iter().all(|&i| shapes[i].is_some()))
                .then(|| (steps.clone(), Some(*g))),
                None => shapes[j].map(|_| (vec![j], None)),
            };
            let Some((steps, g)) = unit else {
                // A step that stays where it is: it closes the chain when it
                // touches a member's bytes (the members run after it).
                if !structural(j)
                    && chain
                        .members
                        .iter()
                        .any(|&(i, ..)| self.hz_conflict(plan, acc, &access[i], &access[j]))
                {
                    self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
                }
                continue;
            };
            let ms: Vec<MorphismId> = steps.iter().filter_map(|&i| op(i)).collect();
            // A launch whose output the host copies out (no `keep`) issues
            // that copy right after it: it ends a chain.
            let copied = ms.iter().any(|&m| self.hz_copied(plan, m));
            // A merged unit whose member before its last copies out: the copy
            // issues between the members, where no persistent launch stands
            // (the group launches at its last member). It stays its own.
            if ms[..ms.len() - 1].iter().any(|&m| self.hz_copied(plan, m)) {
                if !chain.units.is_empty() {
                    *stats.refused.entry("copied").or_default() += 1;
                }
                self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
                continue;
            }
            match self.calls_place(plan, acc, access, &shapes, &chain, &steps, &ms, gpu) {
                Ok(placed) => {
                    chain.units.push(steps);
                    chain.members.extend(placed);
                    chain.absorbed.extend(g);
                    if copied {
                        self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
                    }
                    continue;
                }
                // Not in this chain: it closes it and starts the next.
                Err(why) if !chain.units.is_empty() => {
                    *stats.refused.entry(why).or_default() += 1;
                }
                Err(_) => {}
            }
            self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
            if let Ok(placed) =
                self.calls_place(plan, acc, access, &shapes, &chain, &steps, &ms, gpu)
            {
                chain.units.push(steps);
                chain.members.extend(placed);
                chain.absorbed.extend(g);
                if copied {
                    self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
                }
            }
        }
        self.calls_close(plan, seq, acc, access, &shapes, &mut chain, dead, stats);
    }

    /// The members of a unit (`steps`, launching `ms`) placed in `chain`:
    /// each one's level, zero placement and output; why not, when the unit
    /// does not join it.
    #[allow(clippy::too_many_arguments)]
    fn calls_place(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        access: &[(Vec<ObjectId>, Vec<ObjectId>)],
        shapes: &[Option<Shape>],
        chain: &Chain,
        steps: &[usize],
        ms: &[MorphismId],
        gpu: &Gpu,
    ) -> Result<Vec<Placed>, &'static str> {
        let conflicts = |i: usize| {
            steps
                .iter()
                .any(|&k| self.hz_conflict(plan, acc, &access[i], &access[k]))
        };
        let mut level = chain
            .members
            .iter()
            .filter(|&&(i, ..)| conflicts(i))
            .map(|&(_, l, ..)| l + 1)
            .max()
            .unwrap_or(0);
        let branchy = ms.iter().any(|&m| self.hz_row(plan, m).is_some());
        if branchy && level > 0 {
            return Err("branchy");
        }
        // An atomic site: its zero pass one level before it, where no member
        // of that level touches its output.
        let outs: Vec<Option<ObjectId>> = ms
            .iter()
            .map(|&m| {
                let site = self.tile_plan.as_ref()?.sites.get(m)?;
                let (arms, order) = self.site_choice(m);
                let any_atomic = self.hz_row(plan, m).is_none()
                    && gpu::site_cands(site, self.profile, self.contract, arms, order, false)
                        .iter()
                        .any(|c| matches!(c.launch.combine, gpu::Combine::Atomic { .. }));
                any_atomic.then(|| self.ir.morphism(m).expect("m").target)
            })
            .collect();
        let c0_atomic = ms
            .iter()
            .any(|&m| self.hz_row(plan, m).is_none() && self.calls_c0_atomic(m));
        let touches_at = |l: u32, out: ObjectId| {
            chain.members.iter().any(|&(i, li, ..)| {
                li == l
                    && access[i]
                        .0
                        .iter()
                        .chain(&access[i].1)
                        .any(|&x| self.zero_shares(plan, acc, x, out))
            })
        };
        if c0_atomic {
            level = level.max(1);
            if outs.iter().flatten().any(|&o| touches_at(level - 1, o)) {
                level += 1;
            }
        }
        if branchy && level > 0 {
            return Err("branchy");
        }
        let placed: Vec<Placed> = steps
            .iter()
            .zip(&outs)
            .map(|(&i, out)| {
                let ok = out.is_some_and(|o| level >= 1 && !touches_at(level - 1, o));
                (i, level, ok, *out)
            })
            .collect();
        // The atomic rule candidate needs its zero in the kernel.
        if c0_atomic && placed.iter().any(|p| p.3.is_some() && !p.2) {
            return Err("zero");
        }
        let mut levels: Vec<Vec<Shape>> =
            vec![Vec::new(); (chain.levels().max(level + 1)) as usize];
        for &(i, l, ..) in chain.members.iter().chain(&placed) {
            levels[l as usize].push(shapes[i].expect("a member's shape"));
        }
        persistent_fit(&levels, gpu).map_err(|r| r.name())?;
        Ok(placed)
    }

    /// A member's phase at the part's peaks, in ns: its sum of products'
    /// flops (a tile site: `2 · rows · c · k`) over the FFMA rate, or the
    /// bytes it reads and writes over the DRAM rate, whichever is longer.
    fn calls_work_ns(
        &self,
        access: &(Vec<ObjectId>, Vec<ObjectId>),
        m: MorphismId,
        gpu: &Gpu,
    ) -> f64 {
        let flops = self
            .tile_plan
            .as_ref()
            .and_then(|t| t.sites.get(m))
            .map_or(0, |s| 2 * s.rows * s.c * s.k);
        let mut objs: Vec<ObjectId> = access.0.iter().chain(&access.1).copied().collect();
        objs.sort();
        objs.dedup();
        let bytes: u64 = objs
            .iter()
            .map(|&o| self.array_bytes(&self.obj_ty(o)))
            .sum();
        let at = |n: u64, rate: u64| {
            if rate == 0 {
                0.0
            } else {
                n as f64 / rate as f64
            }
        };
        at(flops, gpu.ffma_flops_per_ns()).max(at(bytes, gpu.dram_peak_bytes_per_ns()))
    }

    /// Is `m`'s rule candidate (the one the plan's shape describes) atomic?
    fn calls_c0_atomic(&self, m: MorphismId) -> bool {
        let Some(site) = self.tile_plan.as_ref().and_then(|t| t.sites.get(m)) else {
            return false;
        };
        let (arms, order) = self.site_choice(m);
        let cands = gpu::site_cands(site, self.profile, self.contract, arms, order, false);
        matches!(cands[0].launch.combine, gpu::Combine::Atomic { .. })
    }

    /// The open chain as one persistent group, when it spans two levels or
    /// more and the rule prices it no slower on the device.
    #[allow(clippy::too_many_arguments)]
    fn calls_close(
        &self,
        plan: &mut ResidentPlan,
        seq: &[Step],
        acc: &AccessPlan,
        access: &[(Vec<ObjectId>, Vec<ObjectId>)],
        shapes: &[Option<Shape>],
        chain: &mut Chain,
        dead: &mut Vec<bool>,
        stats: &mut CallsStats,
    ) {
        let c = std::mem::take(chain);
        let levels = c.levels();
        if c.units.len() < 2 || levels < 2 {
            return;
        }
        stats.chains += 1;
        let Some(gpu) = self.profile.gpu() else {
            return;
        };
        let mut lv: Vec<Vec<Shape>> = vec![Vec::new(); levels as usize];
        for &(i, l, ..) in &c.members {
            lv[l as usize].push(shapes[i].expect("a member's shape"));
        }
        let grid = match persistent_fit(&lv, gpu) {
            Ok(g) => g,
            Err(r) => {
                *stats.refused.entry(r.name()).or_default() += 1;
                return;
            }
        };
        let unit_blocks: Vec<u64> = c
            .units
            .iter()
            .map(|u| u.iter().map(|&i| shapes[i].expect("shape").blocks).sum())
            .collect();
        let barriers = u64::from(levels - 1);
        let pdl = gpu::pdl(self.ir, self.profile, self.contract, self.gpu_arms);
        // A chain inside a loop shares the process's first cooperative
        // launch with its other trips.
        let region = self.regions();
        let looped = c.members.iter().any(|&(i, ..)| match seq[i] {
            Step::Op(m) => region.get(m).is_some_and(|r| r.0.is_some()),
            Step::Loop(_) => true,
        });
        let rounds = chain_rounds(&lv, grid);
        stats.rounds = stats.rounds.max(rounds);
        let work: f64 = c
            .members
            .iter()
            .map(|&(i, ..)| match seq[i] {
                Step::Op(m) => self.calls_work_ns(&access[i], m, gpu),
                Step::Loop(_) => 0.0,
            })
            .sum();
        let Some((per, sep)) =
            chain_ns(&unit_blocks, grid, barriers, gpu, pdl, looped, rounds, work)
        else {
            return;
        };
        // `--calls=force` (measurement: the suite's forced chains against the
        // shipped build): the cost test aside.
        if !self.gpu_arms.calls_force && per > sep {
            *stats.refused.entry("cost").or_default() += 1;
            return;
        }
        stats.formed += 1;
        stats.units += c.units.len();
        stats.barriers += barriers as usize;
        stats.zeros += c.members.iter().filter(|m| m.2).count();
        stats.per_ns += per;
        stats.sep_ns += sep;
        let op = |i: usize| match seq[i] {
            Step::Op(m) => m,
            Step::Loop(_) => unreachable!("a member is a launch"),
        };
        let mut members = c.members.clone();
        members.sort_by_key(|m| m.0);
        // Re-run safety: no member reads a byte it or a later member writes.
        let rerun = members.iter().enumerate().all(|(k, &(i, ..))| {
            !access[i].0.iter().any(|&x| {
                members[k..].iter().any(|&(l, ..)| {
                    access[l]
                        .1
                        .iter()
                        .any(|&y| self.zero_shares(plan, acc, x, y))
                })
            })
        });
        for &g in &c.absorbed {
            dead[g] = true;
        }
        let ms: Vec<MorphismId> = members.iter().map(|m| op(m.0)).collect();
        let g = plan.hz.len();
        dead.push(false);
        for (k, &m) in ms.iter().enumerate() {
            plan.hz_of.insert(m, (g, k));
        }
        let last = *ms.last().expect("two units");
        let (first, end) = (members[0].0, members.last().expect("members").0);
        for step in &seq[first..end] {
            let Step::Op(m) = *step else { continue };
            if m == last {
                continue;
            }
            if let Some(objs) = plan.free_after.remove(m) {
                match plan.free_after.get_mut(last) {
                    Some(v) => v.extend(objs),
                    None => {
                        plan.free_after.insert(last, objs);
                    }
                }
            }
        }
        plan.hz.push(HzGroup {
            branchy: ms.iter().map(|&m| self.hz_row(plan, m).is_some()).collect(),
            members: ms,
            rerun,
            level: members.iter().map(|m| m.1).collect(),
            zero_ok: members.iter().map(|m| m.2).collect(),
        });
    }
}

// ---- device ---------------------------------------------------------------------------------------

/// **The grid barrier** between two levels of a persistent kernel (`calib.cu
/// gsync`'s measured form): the block meets, its thread 0 adds one arrival
/// to the kernel's counter with release semantics at GPU scope — block 0
/// adds `0x80000000 − (n − 1)`, every other block 1, so the counter's top bit
/// flips exactly when all `n` blocks arrived (its low 31 bits start at zero
/// and every complete barrier restores them) — and spins on an acquire load
/// until the bit differs from what it read; then the block meets again. The
/// acquire at GPU scope and the block meet make every write of the level
/// before visible to every thread of the level after (the PTX memory
/// model's release / acquire pattern, cumulative through `bar.sync`).
pub(crate) const GSYNC: &str = "define internal void @mapal_gsync(ptr addrspace(1) %c) alwaysinline {
entry:
  call void @llvm.nvvm.barrier0()
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lead = icmp eq i32 %t, 0
  br i1 %lead, label %arrive, label %done
arrive:
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %n = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %first = icmp eq i32 %b, 0
  %rest = sub i32 -2147483647, %n
  %inc = select i1 %first, i32 %rest, i32 1
  %old = call i32 asm sideeffect \"atom.add.release.gpu.global.u32 $0, [$1], $2;\", \"=r,l,r,~{memory}\"(ptr addrspace(1) %c, i32 %inc)
  br label %spin
spin:
  %cur = call i32 asm sideeffect \"ld.acquire.gpu.global.u32 $0, [$1];\", \"=r,l,~{memory}\"(ptr addrspace(1) %c)
  %x = xor i32 %old, %cur
  %flip = icmp slt i32 %x, 0
  br i1 %flip, label %done, label %spin
done:
  call void @llvm.nvvm.barrier0()
  ret void
}
";

/// The intrinsics a persistent kernel calls, declared when the module has
/// none of its own.
const DECLS: [&str; 5] = [
    "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
    "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
    "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()",
    "declare void @llvm.nvvm.barrier0()",
    "declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)",
];

/// Each of [`DECLS`] the module `text` lacks.
pub(super) fn missing_decls(text: &str) -> String {
    DECLS
        .iter()
        .filter(|d| !text.contains(*d))
        .map(|d| format!("{d}\n"))
        .collect()
}

/// **The persistent kernel of `spec`** (module doc): level by level, each
/// zero pass due before the next level as grid-stride stores, then the
/// level's members over their concatenated block ranges — block `i` on
/// persistent block `i mod G`, each member a device function of its own
/// block coordinates (a member of fewer threads than the block on its own
/// warps, its barriers counted), a block barrier (id 1, apart from the
/// members' id 0) between two blocks' bodies: the next body reuses the
/// shared memory, and a padded member's idle warps wait there — then the
/// grid barrier. Parameters lose `noalias` / `readonly`: a later level reads,
/// through another parameter, what an earlier one wrote (never through the
/// non-coherent read-only path).
pub(super) fn persistent_kernel(
    spec: &super::horizontal::HzSpec,
    heads: &[&super::horizontal::Head],
) -> String {
    let b = spec.block;
    let mut params = Vec::new();
    for (i, h) in heads.iter().enumerate() {
        for (_, full, name) in &h.params {
            let stem = name.trim_start_matches('%');
            let at = full.len() - name.len();
            let ty = full[..at]
                .replace("noalias ", "")
                .replace("readonly ", "")
                .replace("writeonly ", "");
            params.push(format!("{ty}%h{i}.{stem}"));
        }
    }
    let minct = spec
        .members
        .iter()
        .zip(heads)
        .filter_map(|(m, h)| h.minctasm.map(|c| (c * m.3).div_ceil(b)))
        .max();
    let mut attrs = format!("\"nvvm.maxntid\"=\"{b}\"");
    if let Some(c) = minct {
        attrs += &format!(" \"nvvm.minctasm\"=\"{c}\"");
    }
    let levels = spec.level.iter().copied().max().unwrap_or(0) + 1;
    let members: Vec<String> = spec
        .members
        .iter()
        .zip(&spec.level)
        .map(|(m, l)| format!("{}:{}x{}x{}:l{l}", m.0, m.1, m.2, m.3))
        .collect();
    let name = &spec.name;
    let mut s = format!(
        "; mapal.plan kernel={name} kind=calls block={b} levels={levels} members={}\n\
         @{name}.gsync = internal addrspace(1) global i32 0, align 4\n\
         define ptx_kernel void @{name}({}) {attrs} {{\n\
         entry:\n\
         \x20 %cl.b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %cl.t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %cl.g = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n\
         \x20 %cl.gb = mul i32 %cl.g, {b}\n\
         \x20 %cl.b0 = mul i32 %cl.b, {b}\n\
         \x20 %cl.gt = add i32 %cl.b0, %cl.t\n\
         \x20 br label %cl.l0\n",
        members.join(","),
        params.join(", ")
    );
    for l in 0..levels {
        s += &format!("cl.l{l}:\n");
        let mut cur = format!("cl.l{l}");
        // The zero passes of the next level's atomic members.
        for (k, (i, (bytes, p))) in spec
            .zero
            .iter()
            .enumerate()
            .filter(|&(i, _)| spec.level[i] == l + 1)
            .filter_map(|(i, z)| z.map(|z| (i, z)))
            .enumerate()
        {
            let stem = heads[i].params[p].2.trim_start_matches('%');
            let (w, store) = match bytes {
                x if x % 16 == 0 => (
                    16,
                    "store <4 x i32> zeroinitializer, ptr addrspace(1) %P, align 16",
                ),
                x if x % 4 == 0 => (4, "store i32 0, ptr addrspace(1) %P, align 4"),
                _ => (1, "store i8 0, ptr addrspace(1) %P, align 1"),
            };
            let n = bytes / w;
            let t = format!("{l}_{k}");
            s += &format!(
                "  br label %cl.zh{t}\n\
                 cl.zh{t}:\n\
                 \x20 %cl.zi{t} = phi i32 [ %cl.gt, %{cur} ], [ %cl.zn{t}, %cl.zs{t} ]\n\
                 \x20 %cl.zc{t} = icmp ult i32 %cl.zi{t}, {n}\n\
                 \x20 br i1 %cl.zc{t}, label %cl.zs{t}, label %cl.ze{t}\n\
                 cl.zs{t}:\n\
                 \x20 %cl.zo{t} = zext i32 %cl.zi{t} to i64\n\
                 \x20 %cl.zb{t} = mul i64 %cl.zo{t}, {w}\n\
                 \x20 %cl.zp{t} = getelementptr i8, ptr addrspace(1) %h{i}.{stem}, i64 %cl.zb{t}\n\
                 \x20 {}\n\
                 \x20 %cl.zn{t} = add i32 %cl.zi{t}, %cl.gb\n\
                 \x20 br label %cl.zh{t}\n\
                 cl.ze{t}:\n",
                store.replace("%P", &format!("%cl.zp{t}"))
            );
            cur = format!("cl.ze{t}");
        }
        let lm: Vec<usize> = (0..spec.members.len())
            .filter(|&i| spec.level[i] == l)
            .collect();
        if !lm.is_empty() {
            let total: u64 = lm
                .iter()
                .map(|&i| spec.members[i].1 * spec.members[i].2)
                .sum();
            // The member selection for block index `idx` (labels and values
            // tagged `tag`), every path on to `next`.
            let dispatch = |tag: &str, idx: &str, next: &str| -> String {
                let mut d = String::new();
                let mut start = 0u64;
                for (k, &i) in lm.iter().enumerate() {
                    let (mname, gx, gy, bi) = &spec.members[i];
                    let end = start + gx * gy;
                    d += &format!("cl.m{tag}_{k}:\n");
                    if k + 1 < lm.len() {
                        d += &format!(
                            "  %cl.s{tag}_{k} = icmp ult i32 {idx}, {end}\n  br i1 %cl.s{tag}_{k}, label %cl.r{tag}_{k}, label %cl.m{tag}_{}\n",
                            k + 1
                        );
                    } else {
                        d += &format!("  br label %cl.r{tag}_{k}\n");
                    }
                    d += &format!("cl.r{tag}_{k}:\n");
                    if *bi < b {
                        d += &format!(
                            "  %cl.p{tag}_{k} = icmp ult i32 %cl.t, {bi}\n  br i1 %cl.p{tag}_{k}, label %cl.q{tag}_{k}, label %{next}\ncl.q{tag}_{k}:\n"
                        );
                    }
                    let args: Vec<String> = heads[i]
                        .params
                        .iter()
                        .map(|(ty, _, pn)| format!("{ty} %h{i}.{}", pn.trim_start_matches('%')))
                        .chain([
                            format!("i32 %cl.ux{tag}_{k}"),
                            format!("i32 %cl.uy{tag}_{k}"),
                        ])
                        .collect();
                    d += &format!(
                        "  %cl.o{tag}_{k} = sub i32 {idx}, {start}\n\
                         \x20 %cl.ux{tag}_{k} = urem i32 %cl.o{tag}_{k}, {gx}\n\
                         \x20 %cl.uy{tag}_{k} = udiv i32 %cl.o{tag}_{k}, {gx}\n\
                         \x20 call void @hz.{mname}({})\n\
                         \x20 br label %{next}\n",
                        args.join(", ")
                    );
                    start = end;
                }
                d
            };
            // The first block of the level straight-line (the common case:
            // a level of at most `G` blocks runs one each) — a body inside a
            // loop of unknown trips compiles to fewer loads in flight
            // (decode's rows: 10.8 us against 4.2 alone, `kernel-merging.md`
            // Calls) — then the loop over the rest, when the grid is smaller.
            s += &format!(
                "  %cl.c{l} = icmp ult i32 %cl.b, {total}\n\
                 \x20 br i1 %cl.c{l}, label %cl.mf{l}_0, label %cl.e{l}\n"
            );
            s += &dispatch(&format!("f{l}"), "%cl.b", &format!("cl.fx{l}"));
            s += &format!(
                "cl.fx{l}:\n\
                 \x20 call void @llvm.nvvm.barrier.sync.cnt(i32 1, i32 {b})\n\
                 \x20 %cl.r{l} = add i32 %cl.b, %cl.g\n\
                 \x20 br label %cl.h{l}\n\
                 cl.h{l}:\n\
                 \x20 %cl.i{l} = phi i32 [ %cl.r{l}, %cl.fx{l} ], [ %cl.n{l}, %cl.x{l} ]\n\
                 \x20 %cl.k{l} = icmp ult i32 %cl.i{l}, {total}\n\
                 \x20 br i1 %cl.k{l}, label %cl.m{l}_0, label %cl.e{l}\n"
            );
            s += &dispatch(&l.to_string(), &format!("%cl.i{l}"), &format!("cl.x{l}"));
            s += &format!(
                "cl.x{l}:\n\
                 \x20 call void @llvm.nvvm.barrier.sync.cnt(i32 1, i32 {b})\n\
                 \x20 %cl.n{l} = add i32 %cl.i{l}, %cl.g\n\
                 \x20 br label %cl.h{l}\n\
                 cl.e{l}:\n"
            );
        }
        if l + 1 < levels {
            s += &format!(
                "  call void @mapal_gsync(ptr addrspace(1) @{name}.gsync)\n  br label %cl.l{}\n",
                l + 1
            );
        } else {
            s += "  ret void\n";
        }
    }
    s + "}\n"
}

#[cfg(test)]
mod tests {
    use super::super::horizontal::{HzSpec, parse_head};
    use super::*;
    use crate::profile::COOP_BODY_CYCLES_RTX_5080;

    fn gpu() -> &'static Gpu {
        crate::profile::TargetProfile::named("rtx-5080")
            .and_then(|p| p.gpu())
            .expect("rtx-5080 has a GPU")
    }

    fn shape(blocks: u64, block: u64, smem: u64, per_sm: u64, args: usize) -> Shape {
        Shape {
            blocks,
            block,
            smem,
            per_sm,
            args,
        }
    }

    #[test]
    fn the_persistent_grid_fits_or_says_why() {
        let g = gpu();
        // decode's trip: rows and query (1 block each), then two 128-block sites.
        let lv = vec![
            vec![
                shape(1, 256, 0, 6, 3),
                shape(1, 256, 0, 6, 3),
                shape(1, 256, 0, 6, 2),
            ],
            vec![shape(128, 256, 0, 6, 3)],
            vec![shape(128, 256, 0, 6, 3)],
        ];
        assert_eq!(persistent_fit(&lv, g), Ok(128));
        // The grid is capped at one wave of the merged footprint.
        let big = vec![
            vec![shape(4096, 256, 0, 6, 3)],
            vec![shape(4096, 256, 0, 6, 3)],
        ];
        assert_eq!(persistent_fit(&big, g), Ok(g.sms * 6));
        // More pointers than one cooperative launch passes.
        let many = vec![vec![shape(1, 256, 0, 6, 17)], vec![shape(1, 256, 0, 6, 16)]];
        assert_eq!(persistent_fit(&many, g), Err(Refuse::Args));
        // A large member's residency falls at the merged shared memory.
        let occ = vec![
            vec![shape(4 * 504, 256, 0, 6, 3)],
            vec![shape(1, 256, 90 * 1024, 1, 3)],
        ];
        assert_eq!(persistent_fit(&occ, g), Err(Refuse::Occupancy));
        // Two full waves side by side in one level: no round saved.
        let w = g.sms * 6;
        let full = vec![
            vec![shape(w, 256, 0, 6, 3), shape(w, 256, 0, 6, 3)],
            vec![shape(1, 256, 0, 6, 3)],
        ];
        assert_eq!(persistent_fit(&full, g), Err(Refuse::Waves));
    }

    #[test]
    fn the_rule_prices_barriers_against_launches() {
        let g = gpu();
        let d = g.cost.as_ref().expect("measured");
        assert_eq!(g.chain_cost(), (*d, true), "the rtx-5080 measured them");
        // Decode's chain in its loop, one round of its 128-block grid: a
        // cooperative launch (2044 ns) and two barriers (553 + 44 · 1.89 ns
        // at 128 blocks), no body term (one round) and its phases' 22 ‰,
        // against three plain dependent launches (650 ns): the boundaries
        // alone keep it separate here.
        let body = g.coop_body_ns();
        assert!((body - 4390.0 / 2.45).abs() < 0.5, "{body}");
        let at128 = 553.0 + 44.0 * 1.89;
        let (per, sep) = chain_ns(&[3, 128, 128], 128, 2, g, false, true, 1, 0.0).expect("priced");
        assert!(
            (per - (2044.0 + 2.0 * at128)).abs() < 0.5 && (sep - 1950.0).abs() < 0.5,
            "{per} {sep}"
        );
        assert!(per > sep);
        // The phases: 22 ‰ of their work at the part's peaks.
        let (long, _) =
            chain_ns(&[3, 128, 128], 128, 2, g, false, true, 1, 100_000.0).expect("priced");
        assert!((long - per - 2200.0).abs() < 0.5, "{long} {per}");
        // The grid-stride form (more than one round) pays the bodies'
        // measured change (`calib.cu body`, 4 rounds: 1792 ns).
        let (strided, _) =
            chain_ns(&[3, 512, 512], 128, 2, g, false, true, 4, 0.0).expect("priced");
        assert!((strided - per - body).abs() < 0.5, "{strided} {per}");
        // Outside every loop it also pays the process's first cooperative
        // launch (38 us).
        let (once, _) = chain_ns(&[3, 128, 128], 128, 2, g, false, false, 1, 0.0).expect("priced");
        assert!((once - per - 38_000.0).abs() < 0.5, "{once} {per}");
        // Programmatic dependent launches (486 ns each): further apart.
        let (_, sep) = chain_ns(&[3, 128, 128], 128, 2, g, true, true, 1, 0.0).expect("priced");
        assert!((sep - 1458.0).abs() < 0.5, "{sep}");
        // A long chain of one-block launches: twelve barriers of one block
        // (553 ns priced) against twelve more plain launches — still not
        // enough to pay the cooperative launch back here.
        let (per, sep) = chain_ns(&[1; 13], 1, 12, g, false, true, 1, 0.0).expect("priced");
        assert!(per > sep, "{per} {sep}");
        // Without the cooperative launch's price, the barriers alone win a
        // one-round chain; the grid-stride form's bodies keep it separate.
        let mut cheap = *d;
        cheap.coop_launch_ns = 0;
        let g2 = Gpu {
            cost: Some(cheap),
            ..*g
        };
        let (per, sep) =
            chain_ns(&[3, 128, 128], 128, 2, &g2, false, true, 1, 0.0).expect("priced");
        assert!(per <= sep, "{per} {sep}");
        let (per, sep) =
            chain_ns(&[3, 512, 512], 128, 2, &g2, false, true, 4, 0.0).expect("priced");
        assert!(per > sep, "{per} {sep}");
        assert!((d.gsync_at(504, g.sms).unwrap() - (553.0 + 420.0 * 1.89)).abs() < 0.5);
        // open-rules: a part without the measured barrier and launch costs
        // (the 5090) prices chains on the rtx-5080's cycles at its own clock
        // (2.41 GHz: 562 / 660 / 494 / 2078 ns) — derived, as its bodies'.
        let p5090 = crate::profile::TargetProfile::named("rtx-5090")
            .and_then(|p| p.gpu())
            .unwrap();
        let (dv, measured) = p5090.chain_cost();
        assert!(!measured);
        assert_eq!(
            (
                dv.gsync_ns,
                dv.dep_launch_ns,
                dv.pdl_launch_ns,
                dv.coop_launch_ns
            ),
            (562, 660, 494, 2078)
        );
        let (per, sep) = chain_ns(&[3, 128, 128], 128, 2, p5090, true, true, 1, 0.0).unwrap();
        assert!(
            per > sep,
            "the 5090 prices decode as the 5080 does: {per} {sep}"
        );
        assert_eq!(p5090.coop_body_cycles(), (COOP_BODY_CYCLES_RTX_5080, false));
    }

    /// **open-rules `calls`, replayed** (cross-arch 60f0f8ce and 7a2b67bf,
    /// the RTX 5080): the bodies' change is a per-round fact and the
    /// phases' length a fact of the chain. One-round chains (every level
    /// within its persistent grid) price no body term: decode's chain forms
    /// on the 4090 / 3090 (656 / 522 ns saved; it won 1.20 / 1.48 formed in
    /// 7a2b67bf and lost refused in 60f0f8ce) as does causal's (won 1.16 /
    /// 1.36). The grid-stride form pays the derived 4-round body: the a100's
    /// causal chain at 4 rounds refuses (2021 ns saved against 3113). Long
    /// phases pay their 22 ‰: the H100's smlp chain (two GEMM phases of at
    /// least 70 us at its FFMA peak, 1.47x slower formed) refuses, 1540 ns
    /// against 695 saved. Open: the a100's one-round causal family forms
    /// (its forced chain measured 1.3-1.7 % slower: +10 % phases inside).
    #[test]
    fn the_rounds_and_phases_decide_where_the_boundaries_alone_did_not() {
        let part = |n: &str| {
            *crate::profile::TargetProfile::named(n)
                .and_then(|p| p.gpu())
                .unwrap()
        };
        for (n, dec_save) in [("rtx-4090", 656.0), ("rtx-3090", 522.0)] {
            let g = part(n);
            let (per, sep) = chain_ns(&[3, 128, 128], 128, 2, &g, false, true, 1, 0.0).unwrap();
            assert!(
                (sep - per - dec_save).abs() < 1.0,
                "{n} decode: {per} {sep}"
            );
            let (per, sep) =
                chain_ns(&[3, 128, 1, 4, 1, 32], 128, 5, &g, false, true, 1, 0.0).unwrap();
            assert!(per < sep, "{n} causal: {per} {sep}");
        }
        let a100 = part("a100");
        let causal = [3, 128, 1, 4, 1, 32];
        let (per, sep) = chain_ns(&causal, 128, 5, &a100, false, true, 1, 0.0).unwrap();
        assert!((sep - per - 2021.0).abs() < 1.0, "{per} {sep}");
        let (per, sep) = chain_ns(&causal, 128, 5, &a100, false, true, 4, 0.0).unwrap();
        assert!(
            per > sep && (a100.coop_body_ns() - 3113.0).abs() < 1.0,
            "{per} {sep}"
        );
        let h100 = part("h100");
        let flops = 2.0 * (4096.0 * 256.0 * 1024.0) * 2.0;
        let work = flops / h100.ffma_flops_per_ns() as f64;
        assert!(work > 70_000.0, "{work}");
        let (per, sep) = chain_ns(&[256, 256], 256, 1, &h100, false, true, 1, work).unwrap();
        assert!(
            per > sep && sep - (per - work * 0.022) > 600.0,
            "{per} {sep}"
        );
    }

    #[test]
    fn a_persistent_kernel_strides_levels_and_zeroes_before_its_atomic_member() {
        let k = |n: &str, b: u64| {
            format!(
                "define ptx_kernel void @{n}(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias align 4 %o) \"nvvm.maxntid\"=\"{b}\" {{\nentry:\n  call void @llvm.nvvm.barrier0()\n  ret void\n}}\n"
            )
        };
        let (a, b) = ("mapal_tile_000000000000000a", "mapal_tile_000000000000000b");
        let ha = parse_head(k(a, 32).lines().next().unwrap());
        let hb = parse_head(k(b, 256).lines().next().unwrap());
        let spec = HzSpec {
            name: "mapal_tile_00000000000000cc".into(),
            block: 256,
            members: vec![(a.into(), 2, 1, 32), (b.into(), 4, 32, 256)],
            level: vec![0, 1],
            zero: vec![None, Some((4096, 1))],
        };
        let t = persistent_kernel(&spec, &[&ha, &hb]);
        // No `noalias` / `readonly`: level 1 reads what level 0 wrote.
        let head = t.lines().find(|l| l.starts_with("define")).unwrap();
        assert!(
            !head.contains("noalias") && !head.contains("readonly"),
            "{head}"
        );
        assert!(head.contains("ptr addrspace(1) align 16 %h0.a") && head.contains("align 4 %h1.o"));
        // One grid barrier between the two levels, on the kernel's counter.
        assert_eq!(t.matches("call void @mapal_gsync(").count(), 1);
        assert!(
            t.contains("@mapal_tile_00000000000000cc.gsync = internal addrspace(1) global i32 0")
        );
        // `b`'s 4096-byte zero runs at level 0 (before the barrier) as 256
        // grid-stride 16-byte stores into its output parameter.
        let (l0, l1) = t.split_once("call void @mapal_gsync(").unwrap();
        assert!(
            l0.contains("icmp ult i32 %cl.zi0_0, 256")
                && l0.contains("getelementptr i8, ptr addrspace(1) %h1.o")
        );
        assert!(l0.contains("store <4 x i32> zeroinitializer") && !l1.contains("zeroinitializer"));
        // Each level's first block straight-line, then a loop striding over
        // the rest (2 and 128 blocks) by the grid.
        assert!(l0.contains("icmp ult i32 %cl.b, 2") && l0.contains("icmp ult i32 %cl.i0, 2"));
        assert!(l1.contains("icmp ult i32 %cl.b, 128") && l1.contains("icmp ult i32 %cl.i1, 128"));
        assert!(
            t.contains("%cl.r0 = add i32 %cl.b, %cl.g")
                && t.contains("%cl.n0 = add i32 %cl.i0, %cl.g")
        );
        // A 32-thread member runs on its own warps; the tile boundary is
        // barrier 1, apart from the members' barrier 0.
        assert_eq!(l0.matches("icmp ult i32 %cl.t, 32").count(), 2);
        assert!(t.matches("barrier.sync.cnt(i32 1, i32 256)").count() == 4);
        assert!(t.contains("urem i32 %cl.o1_0, 4") && t.contains("udiv i32 %cl.o1_0, 4"));
    }

    #[test]
    fn the_grid_barrier_flips_one_bit_with_release_and_acquire() {
        assert!(GSYNC.contains("atom.add.release.gpu.global.u32"));
        assert!(GSYNC.contains("ld.acquire.gpu.global.u32"));
        // 0x80000000 − (n − 1) = 0x80000001 − n.
        assert!(GSYNC.contains("sub i32 -2147483647, %n"));
        assert_eq!(GSYNC.matches("call void @llvm.nvvm.barrier0()").count(), 2);
        // The counter's top bit flips once all `n` blocks added: simulate.
        for n in [1u32, 2, 3, 84, 504] {
            let mut ctr = 0u32;
            for _ in 0..3 {
                let start = ctr;
                let mut olds = Vec::new();
                for b in 0..n {
                    let inc = if b == 0 {
                        0x8000_0001u32.wrapping_sub(n)
                    } else {
                        1
                    };
                    olds.push(ctr);
                    ctr = ctr.wrapping_add(inc);
                    // No block sees the flip before the last arrival.
                    if b + 1 < n {
                        assert!(
                            olds.iter().all(|&o| (o ^ ctr) & 0x8000_0000 == 0) || b == 0 && n == 1
                        );
                    }
                }
                assert!(olds.iter().all(|&o| (o ^ ctr) & 0x8000_0000 != 0), "n {n}");
                assert_eq!(ctr & 0x7fff_ffff, start & 0x7fff_ffff);
            }
        }
    }
}
