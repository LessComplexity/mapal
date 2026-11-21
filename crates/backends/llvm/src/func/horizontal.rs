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

impl FnEmit<'_> {
    /// **`horizontal`** (module doc): fills `plan.hz` / `plan.hz_of`, moves
    /// the frees inside each group's window after its launch.
    pub(super) fn horizontal(&self, plan: &mut ResidentPlan) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if !(rungs.horizontal || rungs.calls) || !plan.chain.is_empty() {
            return;
        }
        let Some(gpu) = self.profile.gpu() else {
            return;
        };
        let acc = self.ir.access_plan(self.f);
        let gated: Vec<(Vec<ObjectId>, Vec<ObjectId>)> = self
            .gated
            .keys()
            .map(|m| self.hz_access(plan, &acc, m))
            .collect();
        let mut refused = [0usize; Refuse::ALL.len()];
        let mut launches = 0;
        let mut dead: Vec<bool> = Vec::new();
        let mut calls = super::calls::CallsStats::default();
        for seq in self.zero_sequences() {
            let access: Vec<(Vec<ObjectId>, Vec<ObjectId>)> = seq
                .iter()
                .map(|s| match *s {
                    Step::Op(m) => self.hz_access(plan, &acc, m),
                    Step::Loop(merge) => self.loop_access(plan, &acc, merge),
                })
                .collect();
            let shapes: Vec<Option<Shape>> = seq
                .iter()
                .enumerate()
                .map(|(j, s)| match *s {
                    Step::Op(m) => self.hz_shape(plan, m, false).filter(|_| {
                        !gated
                            .iter()
                            .any(|g| self.hz_conflict(plan, &acc, g, &access[j]))
                    }),
                    Step::Loop(_) => None,
                })
                .collect();
            launches += shapes.iter().flatten().count();
            // A `Pair` or `Proj` assembles a product of pointers and values on
            // the host: it touches no device byte a deferred launch reads or
            // writes (a member's output is read on the host only after its
            // launch's copy-out, which issues the launch in place).
            let structural = |j: usize| match seq[j] {
                Step::Op(m) => matches!(
                    self.ir.morphism(m).expect("m").op,
                    Operation::Pair { .. } | Operation::Proj { .. }
                ),
                Step::Loop(_) => false,
            };
            let mut open: Vec<usize> = Vec::new();
            for j in 0..seq.len() {
                if !rungs.horizontal {
                    break;
                }
                // A step touching a member's bytes: the group launches before it.
                if !structural(j)
                    && open
                        .iter()
                        .any(|&i| self.hz_conflict(plan, &acc, &access[i], &access[j]))
                {
                    self.hz_close(plan, &seq, &acc, &access, &mut open);
                }
                let Some(s) = shapes[j] else {
                    continue;
                };
                if !open.is_empty() {
                    let mut all: Vec<Shape> =
                        open.iter().map(|&i| shapes[i].expect("a launch")).collect();
                    all.push(s);
                    let fit = match side_by_side(&all, gpu) {
                        Err(Refuse::Occupancy | Refuse::Waves) if self.gpu_arms.hz_force => Ok(0),
                        r => r,
                    };
                    match fit {
                        Ok(_) => {
                            open.push(j);
                            continue;
                        }
                        Err(r) => {
                            refused[Refuse::ALL.iter().position(|&x| x == r).expect("listed")] += 1;
                            self.hz_close(plan, &seq, &acc, &access, &mut open);
                        }
                    }
                }
                open.push(j);
            }
            self.hz_close(plan, &seq, &acc, &access, &mut open);
            // kernel-merging `calls`: chains of the sequence's units, where the
            // part's barrier and dependent-launch costs were measured.
            let priced = gpu
                .cost
                .as_ref()
                .is_some_and(|d| d.gsync_ns > 0 && d.dep_launch_ns > 0);
            if rungs.calls && priced {
                dead.resize(plan.hz.len(), false);
                self.calls_walk(plan, &seq, &acc, &access, &gated, &mut dead, &mut calls);
            }
        }
        // The groups `calls` absorbed are gone; the rest keep their order.
        if dead.iter().any(|&d| d) {
            let old = std::mem::take(&mut plan.hz);
            let mut at = vec![usize::MAX; old.len()];
            for (g, grp) in old.into_iter().enumerate() {
                if !dead.get(g).copied().unwrap_or(false) {
                    at[g] = plan.hz.len();
                    plan.hz.push(grp);
                }
            }
            for (g, grp) in plan.hz.iter().enumerate() {
                for (k, &m) in grp.members.iter().enumerate() {
                    plan.hz_of.insert(m, (g, k));
                }
            }
            debug_assert!(at.iter().filter(|&&a| a != usize::MAX).count() == plan.hz.len());
        }
        if calls.chains > 0 || !calls.refused.is_empty() {
            let why: Vec<String> = calls
                .refused
                .iter()
                .map(|(r, n)| format!("{r}:{n}"))
                .collect();
            plan.calls_line = format!(
                "; mapal.calls.host chains={} formed={} units={} barriers={} zeros={} per_ns={:.0} sep_ns={:.0} refused={}\n",
                calls.chains,
                calls.formed,
                calls.units,
                calls.barriers,
                calls.zeros,
                calls.per_ns,
                calls.sep_ns,
                if why.is_empty() {
                    "0".to_string()
                } else {
                    why.join(",")
                }
            );
        }
        let refusals: usize = refused.iter().sum();
        let flat: Vec<&HzGroup> = plan.hz.iter().filter(|g| !g.persistent()).collect();
        if flat.is_empty() && refusals == 0 {
            return;
        }
        let why: Vec<String> = Refuse::ALL
            .iter()
            .zip(refused)
            .filter(|&(_, n)| n > 0)
            .map(|(r, n)| format!("{}:{n}", r.name()))
            .collect();
        let all: Vec<MorphismId> = flat.iter().flat_map(|g| g.members.clone()).collect();
        let cells = all.iter().filter(|&&m| plan.cells.contains_key(m)).count();
        let units = all.iter().filter(|&&m| plan.units.contains_key(m)).count();
        let branchy: usize = flat
            .iter()
            .map(|g| g.branchy.iter().filter(|&&b| b).count())
            .sum();
        plan.hz_line = format!(
            "; mapal.hz.host launches={launches} groups={} members={} cells={cells} units={units} tiles={} branchy={branchy} refused={}\n",
            flat.len(),
            all.len(),
            all.len() - cells - units,
            if why.is_empty() {
                "0".to_string()
            } else {
                why.join(",")
            }
        );
    }

    /// The open group (step indices into `seq`) as a merged launch when it
    /// holds two or more: its members, the frees of every step from its first
    /// member to its last moved after the last.
    fn hz_close(
        &self,
        plan: &mut ResidentPlan,
        seq: &[Step],
        acc: &AccessPlan,
        access: &[(Vec<ObjectId>, Vec<ObjectId>)],
        open: &mut Vec<usize>,
    ) {
        let take = std::mem::take(open);
        if take.len() < 2 {
            return;
        }
        let op = |i: usize| match seq[i] {
            Step::Op(m) => m,
            Step::Loop(_) => unreachable!("a member is a launch"),
        };
        let members: Vec<MorphismId> = take.iter().map(|&i| op(i)).collect();
        let branchy = members
            .iter()
            .map(|&m| self.hz_row(plan, m).is_some())
            .collect();
        // A member without a table runs twice only when it writes no byte it
        // reads (a table's own flag says so for it at the launch).
        let rerun = take.iter().all(|&i| {
            let (r, w) = &access[i];
            !r.iter()
                .any(|&x| w.iter().any(|&y| self.zero_shares(plan, acc, x, y)))
        });
        let g = plan.hz.len();
        for (k, &m) in members.iter().enumerate() {
            plan.hz_of.insert(m, (g, k));
        }
        let last = *members.last().expect("two members");
        for i in take[0]..*take.last().expect("two") {
            let Step::Op(m) = seq[i] else { continue };
            if let Some(objs) = plan.free_after.remove(m) {
                match plan.free_after.get_mut(last) {
                    Some(v) => v.extend(objs),
                    None => {
                        plan.free_after.insert(last, objs);
                    }
                }
            }
        }
        let n = members.len();
        plan.hz.push(HzGroup {
            members,
            branchy,
            rerun,
            level: vec![0; n],
            zero_ok: vec![false; n],
        });
    }

    /// `(reads, writes)` of the device work emitted at `m`
    /// ([`Self::step_access`]). The rule runs before the arena and before
    /// `zero-fold`: no arena offset joins two objects here (the arena keeps a
    /// deferred launch's operands until its group launches), and a zero tail
    /// joins a group only at its last member, checked there.
    pub(super) fn hz_access(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        m: MorphismId,
    ) -> (Vec<ObjectId>, Vec<ObjectId>) {
        self.step_access(plan, acc, m)
    }

    /// Does a later access conflict with an earlier one (either order: a
    /// deferred launch moves past the step)?
    pub(super) fn hz_conflict(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        a: &(Vec<ObjectId>, Vec<ObjectId>),
        b: &(Vec<ObjectId>, Vec<ObjectId>),
    ) -> bool {
        acc.hazard((&a.0, &a.1), (&b.0, &b.1), &|x, y| {
            self.zero_shares(plan, acc, x, y)
        })
        .any()
    }

    /// A retained site's row pass, when `m` launches one under this plan
    /// (the emission's own test, `emit_retained`).
    pub(super) fn hz_row(&self, plan: &ResidentPlan, m: MorphismId) -> Option<&TileSite> {
        let target = self.ir.morphism(m)?.target;
        let r = self.retain.get(target)?;
        let kept = !plan.is_carry_next(target) && plan.alias(target) == target;
        (kept && self.on_device(target) && gpu::gpu_site(&r.row, self.profile, true))
            .then_some(&r.row)
    }

    /// The launch emitted at `m`, when the rule may merge it: its rules'
    /// geometry. Cells (not a range cell or a batched write), units, and
    /// unsplit tile sites (not a linear sum's table, a column-major store, an
    /// L2-windowed launch or a row whose split launches without a table);
    /// with `atomic` (`calls`, whose kernel zeroes the output one level
    /// before) a site whose rule candidate adds into a zeroed output too.
    pub(super) fn hz_shape(
        &self,
        plan: &ResidentPlan,
        m: MorphismId,
        atomic: bool,
    ) -> Option<Shape> {
        let gpu = self.profile.gpu()?;
        let zjobs = plan.zero_by.get(m).map_or(0, Vec::len);
        let zargs = if zjobs > 0 { 1 + 2 * zjobs } else { 0 };
        let alone = |b: u64, smem: u64| {
            let t = (gpu.max_threads_per_sm / b.max(1)).min(gpu.max_blocks_per_sm);
            match smem {
                0 => t,
                s => t.min(gpu.smem_per_sm / (s + gpu.smem_reserved_per_block)),
            }
            .max(1)
        };
        if plan.skip.contains_key(m) {
            return None;
        }
        if let Some(c) = plan.cells.get(m) {
            let batched = plan.batch.iter().any(|b| b.writes.iter().any(|w| w.0 == m));
            if c.range.is_some() || batched {
                return None;
            }
            let (gx, block) = gpu::CellGeo {
                range: None,
                lanes: c.lanes,
                zero: 0,
            }
            .grid(c.n);
            return Some(Shape {
                blocks: gx,
                block,
                smem: 0,
                per_sm: alone(block, 0),
                args: 1 + c.arg_count() + zargs,
            });
        }
        if let Some(u) = plan.units.get(m) {
            // long-rows `row-slices`: a sliced unit strides over its items
            // and may meet its grid: a launch of its own.
            if u.slices.is_some() {
                return None;
            }
            let g = self.geo(u);
            let stored = u.members.iter().filter(|mb| mb.store).count();
            return Some(Shape {
                blocks: g.gx,
                block: g.block,
                smem: g.smem,
                per_sm: alone(g.block, g.smem),
                args: stored + u.gmem.len() + zargs,
            });
        }
        let site = self.tile_plan.as_ref()?.sites.get(m)?;
        if !self.device.contains_key(m) {
            return None;
        }
        let col = plan.rowfold.values().any(|r| r.colmajor == Some(m))
            || plan.units.values().flat_map(|u| &u.members).any(|mb| {
                matches!(mb.kind, super::flow::UKind::Fold { colmajor: Some(c), .. } if c == m)
            });
        let epi = self.groups.get(m);
        if col || epi.is_some_and(|g| !g.terms.is_empty()) {
            return None;
        }
        let row = self.hz_row(plan, m);
        let one = row.unwrap_or(site);
        let order = match epi {
            Some(_) => crate::Order::Sequential,
            None => self.order,
        };
        let cands = gpu::site_cands(
            one,
            self.profile,
            self.contract,
            self.gpu_arms,
            order,
            row.is_some(),
        );
        let c0 = cands[0].launch;
        // A one-candidate atomic site launches directly behind its own zero
        // op (no table to take): never a member.
        let atomic_ok = atomic
            && row.is_none()
            && cands.len() > 1
            && matches!(c0.combine, gpu::Combine::Atomic { .. });
        if c0.combine != gpu::Combine::None && !atomic_ok {
            return None;
        }
        if row.is_none()
            && epi.is_none()
            && gpu::host_rungs(self.profile, self.contract, self.gpu_arms).l2_window
            && self.l2_window(m, site, &c0).is_some()
        {
            return None;
        }
        // A row with one candidate launches `launch_in`'s split where `row-split` cuts it.
        if row.is_some() && cands.len() < 2 {
            let lc = gpu::launch_in(one, self.profile, self.contract, self.gpu_arms, self.order);
            if lc.rungs.row_split && lc.combine != gpu::Combine::None {
                return None;
            }
        }
        let args = 3 + epi.map_or(0, |g| {
            g.caps().len() + g.chain.iter().filter(|mb| mb.store).count()
        });
        // An atomic split's partitions are blocks along `y` (the recipe's `gy · s`).
        let split = match c0.combine {
            gpu::Combine::Atomic { s } => s,
            _ => 1,
        };
        Some(Shape {
            blocks: c0.gx * c0.gy * split,
            block: c0.block,
            smem: c0.smem,
            per_sm: c0.wave(gpu) / gpu.sms,
            args,
        })
    }
}

// ---- emission -------------------------------------------------------------------------------------

/// The emission's state of every group: what each member's launch left.
#[derive(Clone, Debug, Default)]
pub(crate) struct HzEmit {
    /// The member being emitted: `(group, member)`.
    pub cur: Option<(usize, usize)>,
    /// A retained site's whole-site arm: launched in place, never captured.
    pub standalone: bool,
    groups: Vec<HzState>,
}

#[derive(Clone, Debug, Default)]
struct HzState {
    caps: Vec<Option<Cap>>,
    /// A retained member's flag (`alloca i1`): its row arm ran this trip.
    flags: Vec<Option<String>>,
    /// A member's emission held a launch the merge cannot take: its own
    /// launch ran in place, the others launch alone at the group's point.
    aborted: bool,
}

/// One member's deferred launch: its pointer array, and its table or kernel.
#[derive(Clone, Debug)]
struct Cap {
    arr: String,
    nptr: usize,
    kind: CapKind,
    /// The launch as the member would issue it (re-issued when not merged).
    call: String,
}

#[derive(Clone, Debug)]
enum CapKind {
    Table {
        global: String,
        rows: Vec<Row>,
        flags: u32,
        out: usize,
    },
    Fixed(Row),
}

/// A launch's kernel and geometry (a `%mapal_cand` row's fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    id: u64,
    zero: u64,
    ws: u64,
    gx: u64,
    gy: u64,
    block: u64,
    smem: u64,
}

/// A runtime launch line's parts: a table (`mapal_gpu_site`), a pointer
/// array (`launchv`), or three pointers by value (`launch`).
enum Parsed {
    Site {
        global: String,
        flags: u32,
        nptr: usize,
        arr: String,
        out: usize,
    },
    Launchv {
        row: Row,
        nptr: usize,
        arr: String,
    },
    Launch {
        row: Row,
        args: [String; 3],
    },
}

fn int(s: &str, ty: &str) -> Option<i64> {
    s.trim().strip_prefix(ty)?.trim().parse().ok()
}

/// The fields of `call void @mapal_gpu_<entry>(…)` (a ` ; unit` tag allowed,
/// any other tag — a split's — not).
fn call_fields<'s>(line: &'s str, entry: &str) -> Option<Vec<&'s str>> {
    let line = line.trim();
    let line = line.strip_suffix(" ; unit").unwrap_or(line);
    let rest = line.strip_prefix(&format!("call void @mapal_gpu_{entry}("))?;
    let inner = rest.strip_suffix(')')?;
    (!inner.contains(';')).then(|| inner.split(", ").collect())
}

fn parse_launch(line: &str) -> Option<Parsed> {
    if let Some(f) = call_fields(line, "site") {
        let [_, flags, stream, _, global, nptr, arr, out, _] = f[..] else {
            return None;
        };
        if int(stream, "i32")? != 0 {
            return None;
        }
        return Some(Parsed::Site {
            global: global.strip_prefix("ptr ")?.to_string(),
            flags: int(flags, "i32")? as u32,
            nptr: int(nptr, "i32")? as usize,
            arr: arr.strip_prefix("ptr ")?.to_string(),
            out: int(out, "i32")? as usize,
        });
    }
    let geom = |f: &[&str]| -> Option<[u64; 4]> {
        let v: Vec<u64> = f
            .iter()
            .map(|x| int(x, "i32").map(|v| v as u64))
            .collect::<Option<_>>()?;
        v.try_into().ok()
    };
    if let Some(f) = call_fields(line, "launchv") {
        let [id, stream, nptr, arr, ..] = f[..] else {
            return None;
        };
        let [gx, gy, block, smem] = geom(f.get(4..8)?)?;
        if f.len() != 8 || int(stream, "i32")? != 0 {
            return None;
        }
        return Some(Parsed::Launchv {
            row: Row {
                id: int(id, "i64")? as u64,
                zero: 0,
                ws: 0,
                gx,
                gy,
                block,
                smem,
            },
            nptr: int(nptr, "i32")? as usize,
            arr: arr.strip_prefix("ptr ")?.to_string(),
        });
    }
    if let Some(f) = call_fields(line, "launch") {
        if f.len() != 8 {
            return None;
        }
        let [gx, gy, block, smem] = geom(&f[4..8])?;
        let arg = |x: &str| x.strip_prefix("i64 ").map(str::to_string);
        return Some(Parsed::Launch {
            row: Row {
                id: int(f[0], "i64")? as u64,
                zero: 0,
                ws: 0,
                gx,
                gy,
                block,
                smem,
            },
            args: [arg(f[1])?, arg(f[2])?, arg(f[3])?],
        });
    }
    None
}

/// The rows of table `global` in `tables` (`%mapal_cand { i64 id, i64 rid,
/// i64 zero, i64 ws, i32 gx, i32 gy, i32 block, i32 smem, … }`).
fn table_rows(tables: &str, global: &str) -> Option<Vec<Row>> {
    let line = tables
        .lines()
        .find(|l| l.starts_with(&format!("{global} = ")))?;
    line.split("%mapal_cand { ")
        .skip(1)
        .map(|r| {
            let f: Vec<&str> = r.split(" }").next()?.split(", ").collect();
            let i = |k: usize, ty: &str| int(f.get(k)?, ty);
            Some(Row {
                id: i(0, "i64")? as u64,
                zero: i(2, "i64")? as u64,
                ws: i(3, "i64")? as u64,
                gx: i(4, "i32")? as u64,
                gy: i(5, "i32")? as u64,
                block: i(6, "i32")? as u64,
                smem: i(7, "i32")? as u64,
            })
        })
        .collect()
}

/// A merged kernel's id: its members' kernels and grids.
fn hz_id(rows: &[Row]) -> u64 {
    let key: Vec<String> = rows
        .iter()
        .map(|r| format!("{:016x}:{}x{}x{}", r.id, r.gx, r.gy, r.block))
        .collect();
    gpu::fnv(&format!("hz|{}", key.join("|")))
}

impl FnEmit<'_> {
    /// `line`'s hook while member `(g, i)` is emitted: a launch the merge
    /// takes is captured (`true`: not emitted here); any other runtime call
    /// aborts the group — the member's captured launch is issued in place,
    /// before it (`false`: the caller emits the line).
    pub(super) fn hz_take(&mut self, s: &str) -> bool {
        let Some((g, i)) = self.hz.cur else {
            return false;
        };
        if self.hz.standalone || !s.contains("call void @mapal_gpu_") {
            return false;
        }
        if self.hz.groups.len() <= g {
            self.hz.groups.resize(g + 1, HzState::default());
        }
        let k = self.resident.as_ref().expect("plan").hz[g].members.len();
        let st = &mut self.hz.groups[g];
        if st.caps.len() < k {
            st.caps.resize(k, None);
            st.flags.resize(k, None);
        }
        let parsed = (!s.contains('\n'))
            .then(|| parse_launch(s))
            .flatten()
            .filter(|_| self.hz.groups[g].caps[i].is_none() && !self.hz.groups[g].aborted);
        let Some(p) = parsed else {
            // Not a launch the merge takes: this member's launch runs here.
            self.hz_abort(g, i);
            return false;
        };
        let cap = match p {
            Parsed::Site {
                global,
                flags,
                nptr,
                arr,
                out,
            } => {
                let Some(rows) = table_rows(&self.cand_tables, &global) else {
                    self.hz_abort(g, i);
                    return false;
                };
                Cap {
                    arr,
                    nptr,
                    kind: CapKind::Table {
                        global,
                        rows,
                        flags,
                        out,
                    },
                    call: s.trim().to_string(),
                }
            }
            Parsed::Launchv { row, nptr, arr } => Cap {
                arr,
                nptr,
                kind: CapKind::Fixed(row),
                call: s.trim().to_string(),
            },
            Parsed::Launch { row, args } => {
                // Three pointers by value: kept in an array, so a launch in
                // a branch arm reaches the group's point.
                let arr = format!("%hzp{}", self.fresh());
                self.allocas
                    .push_str(&format!("  {arr} = alloca [3 x i64]\n"));
                for (k, a) in args.iter().enumerate() {
                    let p = self.tmp();
                    self.body.push_str(&format!(
                        "  {p} = getelementptr [3 x i64], ptr {arr}, i64 0, i64 {k}\n  store i64 {a}, ptr {p}\n"
                    ));
                }
                Cap {
                    arr: arr.clone(),
                    nptr: 3,
                    kind: CapKind::Fixed(row),
                    call: format!(
                        "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 3, ptr {arr}, i32 {}, i32 {}, i32 {}, i32 {})",
                        row.id as i64, row.gx, row.gy, row.block, row.smem
                    ),
                }
            }
        };
        if self.resident.as_ref().expect("plan").hz[g].branchy[i] {
            let flag = match self.hz.groups[g].flags[i].clone() {
                Some(f) => f,
                None => {
                    let f = format!("%hzf{}", self.fresh());
                    self.allocas
                        .push_str(&format!("  {f} = alloca i1\n  store i1 false, ptr {f}\n"));
                    self.hz.groups[g].flags[i] = Some(f.clone());
                    f
                }
            };
            self.body
                .push_str(&format!("  store i1 true, ptr {flag}\n"));
        }
        self.hz.groups[g].caps[i] = Some(cap);
        true
    }

    /// Member `i` of group `g` launches in place: its captured launch is
    /// issued here, and in a `calls` group every earlier member's before it
    /// (they produce what it reads); the rest launch alone.
    fn hz_abort(&mut self, g: usize, i: usize) {
        self.hz.cur = None;
        let group = &self.resident.as_ref().expect("plan").hz[g];
        let persistent = group.persistent();
        // kernel-merging `calls`: the last member's own launch was taken and
        // what follows it (its output's copy-out) reads what the group
        // writes: the group launches here, before it.
        if persistent
            && i + 1 == group.members.len()
            && !self.hz.groups[g].aborted
            && self.hz.groups[g].caps[i].is_some()
        {
            self.hz_launch(g);
            return;
        }
        self.hz.groups[g].aborted = true;
        let from = if persistent { 0 } else { i };
        for k in from..=i {
            if let Some(c) = self.hz.groups[g].caps[k].take() {
                let flag = self.hz.groups[g].flags[k].clone();
                self.hz_issue(&c, flag.as_deref());
            }
        }
    }

    /// Issue a captured launch on its own (under its flag when it ran in a
    /// branch arm).
    fn hz_issue(&mut self, c: &Cap, flag: Option<&str>) {
        let Some(f) = flag else {
            self.line(&c.call);
            return;
        };
        let (on, next) = (self.label(), self.label());
        let v = self.tmp();
        self.line(format!("{v} = load i1, ptr {f}"));
        self.line(format!("br i1 {v}, label %{on}, label %{next}"));
        self.label_line(&on);
        self.line(&c.call);
        self.line(format!("br label %{next}"));
        self.label_line(&next);
    }

    /// The merged tuples of the captured launches: each member's unsplit
    /// candidates (a fixed launch its own), every tuple whose block and shared
    /// memory a launch takes, in order of their ranks' sum, at most `--cands`.
    fn hz_tuples(&self, group: &HzGroup, caps: &[&Cap]) -> Vec<Vec<Row>> {
        let gpu = self.profile.gpu().expect("a GPU profile");
        // kernel-merging `calls`: a member whose zero pass the persistent
        // kernel runs one level before it takes its atomic candidates too.
        let opts: Vec<Vec<(usize, Row)>> = caps
            .iter()
            .enumerate()
            .map(|(i, c)| match &c.kind {
                CapKind::Table { rows, .. } => rows
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.ws == 0 && (r.zero == 0 || group.zero_ok.get(i).copied().unwrap_or(false))
                    })
                    .collect(),
                CapKind::Fixed(r) => vec![(0, *r)],
            })
            .collect();
        if opts.iter().any(Vec::is_empty) {
            return Vec::new();
        }
        let mut all: Vec<(Vec<usize>, Vec<Row>)> = vec![(Vec::new(), Vec::new())];
        for o in &opts {
            all = all
                .into_iter()
                .flat_map(|(ks, rs)| {
                    o.iter().map(move |&(k, r)| {
                        let (mut ks, mut rs) = (ks.clone(), rs.clone());
                        ks.push(k);
                        rs.push(r);
                        (ks, rs)
                    })
                })
                .collect();
        }
        all.retain(|(_, rs)| {
            let b = rs.iter().map(|r| r.block).max().unwrap_or(0);
            b <= gpu.max_threads_per_block
                && rs.iter().all(|r| r.block % gpu.warp == 0)
                && rs.iter().map(|r| r.smem).max().unwrap_or(0) <= gpu.smem_per_block
        });
        all.sort_by_key(|(ks, _)| (ks.iter().sum::<usize>(), ks.clone()));
        let cap = usize::try_from(self.gpu_arms.site_cands)
            .unwrap_or(usize::MAX)
            .max(1);
        all.into_iter().take(cap).map(|(_, rs)| rs).collect()
    }

    /// At group `g`'s last member: the merged launch over every member's
    /// captured pointers (when every member's launch was captured and a tuple
    /// fits; under the retained members' flags), each member's own launch
    /// otherwise.
    pub(super) fn hz_launch(&mut self, g: usize) {
        let group = self.resident.as_ref().expect("plan").hz[g].clone();
        let k = group.members.len();
        let st = match self.hz.groups.get_mut(g) {
            Some(s) => std::mem::take(s),
            None => HzState::default(),
        };
        let caps: Vec<Option<Cap>> = (0..k).map(|i| st.caps.get(i).cloned().flatten()).collect();
        let flags: Vec<Option<String>> =
            (0..k).map(|i| st.flags.get(i).cloned().flatten()).collect();
        let complete: Option<Vec<&Cap>> = (!st.aborted)
            .then(|| caps.iter().map(Option::as_ref).collect::<Option<Vec<_>>>())
            .flatten();
        let tuples = complete
            .as_ref()
            .map(|c| self.hz_tuples(&group, c))
            .unwrap_or_default();
        let fallback = |e: &mut Self| {
            for (c, f) in caps.iter().zip(&flags) {
                if let Some(c) = c {
                    e.hz_issue(c, f.as_deref());
                }
            }
        };
        if tuples.is_empty() {
            fallback(self);
        } else {
            let all_flags: Vec<&String> = flags.iter().flatten().collect();
            let arms = (!all_flags.is_empty()).then(|| {
                let mut acc = String::new();
                for f in &all_flags {
                    let v = self.tmp();
                    self.line(format!("{v} = load i1, ptr {f}"));
                    acc = if acc.is_empty() {
                        v
                    } else {
                        let a = self.tmp();
                        self.line(format!("{a} = and i1 {acc}, {v}"));
                        a
                    };
                }
                let (m, f, end) = (self.label(), self.label(), self.label());
                self.line(format!("br i1 {acc}, label %{m}, label %{f}"));
                self.label_line(&m);
                (f, end)
            });
            let caps: Vec<&Cap> = complete.expect("complete");
            self.hz_merged(&group, &caps, &tuples);
            if let Some((f, end)) = arms {
                self.line(format!("br label %{end}"));
                self.label_line(&f);
                fallback(self);
                self.line(format!("br label %{end}"));
                self.label_line(&end);
            }
            // A table no other launch issues is not the module's.
            if flags.iter().all(Option::is_none) {
                for c in &caps {
                    if let CapKind::Table { global, .. } = &c.kind {
                        self.cand_tables = self
                            .cand_tables
                            .lines()
                            .filter(|l| !l.starts_with(&format!("{global} = ")))
                            .map(|l| format!("{l}\n"))
                            .collect();
                    }
                }
            }
        }
        for f in flags.iter().flatten() {
            self.line(format!("store i1 false, ptr {f}"));
        }
    }

    /// The merged launch: the members' pointers in one array, then the
    /// tuples' table (or the one kernel).
    fn hz_merged(&mut self, group: &HzGroup, caps: &[&Cap], tuples: &[Vec<Row>]) {
        let np: usize = caps.iter().map(|c| c.nptr).sum();
        let arr = format!("%hzv{}", self.fresh());
        self.allocas
            .push_str(&format!("  {arr} = alloca [{np} x i64]\n"));
        let mut off = 0;
        let mut out = None;
        let mut rerun = group.rerun;
        for c in caps {
            for p in 0..c.nptr {
                let (s, v, d) = (self.tmp(), self.tmp(), self.tmp());
                self.line(format!(
                    "{s} = getelementptr [{n} x i64], ptr {a}, i64 0, i64 {p}",
                    n = c.nptr,
                    a = c.arr
                ));
                self.line(format!("{v} = load i64, ptr {s}"));
                self.line(format!(
                    "{d} = getelementptr [{np} x i64], ptr {arr}, i64 0, i64 {}",
                    off + p
                ));
                self.line(format!("store i64 {v}, ptr {d}"));
            }
            if let CapKind::Table { flags, out: o, .. } = &c.kind {
                rerun &= flags & 1 != 0;
                out.get_or_insert(off + o);
            }
            off += c.nptr;
        }
        let persistent = group.persistent();
        let mut rows = Vec::new();
        for t in tuples {
            let block = t.iter().map(|r| r.block).max().expect("members");
            let smem = t.iter().map(|r| r.smem).max().expect("members");
            if persistent {
                // kernel-merging `calls`: one persistent kernel, at most the
                // largest level's blocks (the runtime caps it at the loaded
                // kernel's co-resident grid).
                let members: Vec<String> = t
                    .iter()
                    .zip(caps)
                    .zip(&group.level)
                    .map(|((r, c), l)| {
                        let z = match (&c.kind, r.zero) {
                            (CapKind::Table { out, .. }, z) if z > 0 => format!(":z{z}@{out}"),
                            _ => String::new(),
                        };
                        format!(
                            "{}:{}:{}:{}:l{l}{z}",
                            gpu::kernel_name(r.id),
                            r.gx,
                            r.gy,
                            r.block
                        )
                    })
                    .collect();
                let id = gpu::fnv(&format!("calls|{}", members.join("|")));
                let mut per_level: std::collections::BTreeMap<u32, u64> = Default::default();
                for (r, &l) in t.iter().zip(&group.level) {
                    *per_level.entry(l).or_default() += r.gx * r.gy;
                }
                let gx = per_level.values().copied().max().unwrap_or(1);
                self.line(format!(
                    "; mapal.hz kernel={} block={block} smem={smem} grid={gx} members={}",
                    gpu::kernel_name(id),
                    members.join(",")
                ));
                rows.push((id, gx, block, smem));
                continue;
            }
            let id = hz_id(t);
            let gx: u64 = t.iter().map(|r| r.gx * r.gy).sum();
            let members: Vec<String> = t
                .iter()
                .map(|r| format!("{}:{}:{}:{}", gpu::kernel_name(r.id), r.gx, r.gy, r.block))
                .collect();
            self.line(format!(
                "; mapal.hz kernel={} block={block} smem={smem} members={}",
                gpu::kernel_name(id),
                members.join(",")
            ));
            rows.push((id, gx, block, smem));
        }
        if rows.len() == 1 {
            let (id, gx, block, smem) = rows[0];
            let entry = if persistent { "launch_coop" } else { "launchv" };
            self.line(format!(
                "call void @mapal_gpu_{entry}(i64 {}, i32 0, i32 {np}, ptr {arr}, i32 {gx}, i32 1, i32 {block}, i32 {smem})",
                id as i64
            ));
            return;
        }
        let global = format!("@{}.hz{}", self.fnames[self.f], self.fresh());
        let consts: Vec<String> = rows
            .iter()
            .map(|&(id, gx, block, smem)| {
                format!(
                    "%mapal_cand {{ i64 {}, i64 0, i64 0, i64 0, i32 {gx}, i32 1, i32 {block}, i32 {smem}, i32 0, i32 0 }}",
                    id as i64
                )
            })
            .collect();
        self.cand_tables.push_str(&format!(
            "{global} = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
            consts.len(),
            consts.join(", ")
        ));
        let coop = if persistent {
            super::calls::SITE_COOP
        } else {
            0
        };
        self.line(format!(
            "call void @mapal_gpu_site(i64 {}, i32 {}, i32 0, i32 {}, ptr {global}, i32 {np}, ptr {arr}, i32 {}, ptr null)",
            rows[0].0 as i64,
            u32::from(rerun) | coop,
            rows.len(),
            out.unwrap_or(0)
        ));
    }
}

// ---- device ---------------------------------------------------------------------------------------

/// A merged kernel as the host's `; mapal.hz` line names it: its block, and
/// each member's kernel, grid and block, in block-range order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HzSpec {
    pub name: String,
    pub block: u64,
    pub members: Vec<(String, u64, u64, u64)>,
    /// kernel-merging `calls`: each member's level (all 0: side by side).
    pub level: Vec<u32>,
    /// `calls`: a member's zero pass — its output's bytes and parameter.
    pub zero: Vec<Option<(u64, usize)>>,
}

impl HzSpec {
    pub(crate) fn persistent(&self) -> bool {
        self.level.iter().any(|&l| l > 0)
    }
}

/// Every merged kernel the host text names (once each).
pub(crate) fn hz_specs(host: &str) -> Vec<HzSpec> {
    let mut out: Vec<HzSpec> = Vec::new();
    for line in host.lines() {
        let Some(rest) = line.trim().strip_prefix("; mapal.hz kernel=") else {
            continue;
        };
        let mut f = rest.split_whitespace();
        let name = f.next().unwrap_or_default().to_string();
        let mut block = 0;
        let mut members = Vec::new();
        let (mut level, mut zero) = (Vec::new(), Vec::new());
        for kv in f {
            if let Some(b) = kv.strip_prefix("block=") {
                block = b.parse().unwrap_or(0);
            } else if let Some(ms) = kv.strip_prefix("members=") {
                for m in ms.split(',') {
                    let p: Vec<&str> = m.split(':').collect();
                    if let [n, gx, gy, b, ref rest @ ..] = p[..] {
                        members.push((
                            n.to_string(),
                            gx.parse().unwrap_or(0),
                            gy.parse().unwrap_or(0),
                            b.parse().unwrap_or(0),
                        ));
                        // `calls`: `l<level>`, then `z<bytes>@<param>`.
                        let l = rest
                            .iter()
                            .find_map(|x| x.strip_prefix('l'))
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        let z = rest.iter().find_map(|x| {
                            let (b, p) = x.strip_prefix('z')?.split_once('@')?;
                            Some((b.parse().ok()?, p.parse().ok()?))
                        });
                        level.push(l);
                        zero.push(z);
                    }
                }
            }
        }
        if !out.iter().any(|s| s.name == name) {
            out.push(HzSpec {
                name,
                block,
                members,
                level,
                zero,
            });
        }
    }
    out
}

/// The kernels the merged kernels call (they must be in the module to be
/// composed).
pub(crate) fn hz_members(specs: &[HzSpec]) -> std::collections::BTreeSet<String> {
    specs
        .iter()
        .flat_map(|s| s.members.iter().map(|m| m.0.clone()))
        .collect()
}

/// A kernel's parameters `(type, full text, name)` and its `nvvm` bounds.
pub(super) struct Head {
    pub params: Vec<(String, String, String)>,
    pub minctasm: Option<u64>,
}

pub(super) fn parse_head(define: &str) -> Head {
    let open = define.find('(').expect("a define's parameters");
    let close = define.rfind(") ").expect("a define's attributes");
    let ps = &define[open + 1..close];
    let params = ps
        .split(", ")
        .filter(|p| !p.is_empty())
        .map(|p| {
            let name = p.rsplit(' ').next().expect("a named parameter").to_string();
            let ty = match p.strip_prefix("ptr addrspace(") {
                Some(r) => format!("ptr addrspace({}", &r[..=r.find(')').expect("addrspace")]),
                None => p.split(' ').next().expect("a type").to_string(),
            };
            (ty, p.to_string(), name)
        })
        .collect();
    let attr = |k: &str| {
        define[close..]
            .split(&format!("\"nvvm.{k}\"=\""))
            .nth(1)
            .and_then(|r| r.split('"').next())
            .and_then(|v| v.parse().ok())
    };
    Head {
        params,
        minctasm: attr("minctasm"),
    }
}

/// Kernel `name`'s text as a device function of its block coordinates:
/// `@hz.<name>(<params>, i32 %hz.bx, i32 %hz.by)` (not a `mapal_tile_` symbol:
/// no launch loads it), inlined into each merged
/// kernel; its `ctaid` reads are the arguments, and (`pad`: a merged block
/// larger than its own `block`) its barriers count its own threads.
pub(super) fn device_fn(text: &str, name: &str, pad: Option<u64>) -> String {
    let (define, body) = text.split_once('\n').expect("a define line");
    let open = define.find('(').expect("parameters");
    let close = define.rfind(") ").expect("attributes");
    let ps = &define[open + 1..close];
    let sep = if ps.is_empty() { "" } else { ", " };
    let mut out = format!(
        "define internal void @hz.{name}({ps}{sep}i32 %hz.bx, i32 %hz.by) alwaysinline {{\n"
    );
    let mut b = body
        .replace(
            "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
            "add i32 %hz.bx, 0",
        )
        .replace(
            "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()",
            "add i32 %hz.by, 0",
        );
    if let Some(n) = pad {
        b = b.replace(
            "call void @llvm.nvvm.barrier0()",
            &format!("call void @llvm.nvvm.barrier.sync.cnt(i32 0, i32 {n})"),
        );
    }
    out.push_str(&b);
    out
}

/// The merged kernel of `spec`: block `b` of the grid runs the member whose
/// block range holds it, at its own coordinates (`x` fastest, the order the
/// hardware issues a grid's blocks in); a member of fewer threads than the
/// block runs on its own warps, the others exit.
fn merged_kernel(spec: &HzSpec, heads: &[&Head]) -> String {
    let b = spec.block;
    let mut params = Vec::new();
    for (i, h) in heads.iter().enumerate() {
        for (_, full, name) in &h.params {
            let stem = name.trim_start_matches('%');
            let at = full.len() - name.len();
            params.push(format!("{}%h{i}.{stem}", &full[..at]));
        }
    }
    // Each member keeps its own register cap: `minctasm·block` threads a
    // member asked of an SM, in blocks of the merged size.
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
    let members: Vec<String> = spec
        .members
        .iter()
        .map(|m| format!("{}:{}x{}x{}", m.0, m.1, m.2, m.3))
        .collect();
    let mut s = format!(
        "; mapal.plan kernel={} kind=hz block={b} members={}\n\
         define ptx_kernel void @{}({}) {attrs} {{\n\
         entry:\n\
         \x20 %hz.b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %hz.t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 br label %hz.m0\n",
        spec.name,
        members.join(","),
        spec.name,
        params.join(", ")
    );
    let n = spec.members.len();
    let mut start = 0u64;
    for (i, ((name, gx, gy, bi), h)) in spec.members.iter().zip(heads).enumerate() {
        let end = start + gx * gy;
        s += &format!("hz.m{i}:\n");
        if i + 1 < n {
            s += &format!(
                "  %hz.c{i} = icmp ult i32 %hz.b, {end}\n  br i1 %hz.c{i}, label %hz.r{i}, label %hz.m{}\n",
                i + 1
            );
        } else {
            s += &format!("  br label %hz.r{i}\n");
        }
        s += &format!("hz.r{i}:\n");
        if *bi < b {
            s += &format!(
                "  %hz.p{i} = icmp ult i32 %hz.t, {bi}\n  br i1 %hz.p{i}, label %hz.g{i}, label %hz.end\nhz.g{i}:\n"
            );
        }
        let args: Vec<String> = h
            .params
            .iter()
            .map(|(ty, _, name)| format!("{ty} %h{i}.{}", name.trim_start_matches('%')))
            .chain([format!("i32 %hz.x{i}"), format!("i32 %hz.y{i}")])
            .collect();
        s += &format!(
            "  %hz.l{i} = sub i32 %hz.b, {start}\n\
             \x20 %hz.x{i} = urem i32 %hz.l{i}, {gx}\n\
             \x20 %hz.y{i} = udiv i32 %hz.l{i}, {gx}\n\
             \x20 call void @hz.{name}({})\n\
             \x20 br label %hz.end\n",
            args.join(", ")
        );
        start = end;
    }
    s + "hz.end:\n  ret void\n}\n"
}

/// The device module with every merged kernel of `specs`: each member kernel
/// once as a device function (its standalone kernel kept only when a host
/// launch still names it, `live`), then the merged kernels.
pub(crate) fn compose(
    module: String,
    specs: &[HzSpec],
    live: &std::collections::BTreeSet<String>,
) -> String {
    if specs.is_empty() {
        return module;
    }
    let mut out = module;
    let mut heads: std::collections::BTreeMap<String, Head> = std::collections::BTreeMap::new();
    let mut fns = String::new();
    let mut padded = false;
    for name in hz_members(specs) {
        let head = format!("define ptx_kernel void @{name}(");
        let start = out
            .find(&head)
            .unwrap_or_else(|| panic!("merged member {name} is in the device module"));
        let end = start + out[start..].find("\n}\n").expect("a kernel's end") + 3;
        let text = out[start..end].to_string();
        let own = specs
            .iter()
            .flat_map(|s| s.members.iter().map(move |m| (s.block, m)))
            .filter(|(_, m)| m.0 == name)
            .map(|(b, m)| (b, m.3))
            .collect::<Vec<_>>();
        let pad = own.iter().any(|&(b, bi)| b > bi) && text.contains("@llvm.nvvm.barrier0()");
        padded |= pad;
        heads.insert(
            name.clone(),
            parse_head(text.lines().next().expect("define")),
        );
        fns.push('\n');
        fns.push_str(&device_fn(&text, &name, pad.then(|| own[0].1)));
        if !live.contains(&name) {
            // Its facts lines go with it (the merged kernel's line names it).
            let mut from = start;
            for tag in [format!("; mapal.plan kernel={name}"), "; cell".to_string()] {
                let prev = out[..from.saturating_sub(1)]
                    .rfind('\n')
                    .map_or(0, |i| i + 1);
                let line = &out[prev..from.saturating_sub(1)];
                if from > 0 && (line == tag || line.starts_with(&format!("{tag} "))) {
                    from = prev;
                }
            }
            out.replace_range(from..end, "");
        }
    }
    if padded && !out.contains("declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)") {
        out.push_str("\ndeclare void @llvm.nvvm.barrier.sync.cnt(i32, i32)\n");
    }
    out.push_str(&fns);
    if specs.iter().any(HzSpec::persistent) {
        let decls = super::calls::missing_decls(&out);
        out.push('\n');
        out.push_str(&decls);
        out.push_str(super::calls::GSYNC);
    }
    for spec in specs {
        let hs: Vec<&Head> = spec.members.iter().map(|m| &heads[&m.0]).collect();
        out.push('\n');
        if spec.persistent() {
            out.push_str(&super::calls::persistent_kernel(spec, &hs));
        } else {
            out.push_str(&merged_kernel(spec, &hs));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> &'static Gpu {
        crate::profile::TargetProfile::named("rtx-5080")
            .and_then(|p| p.gpu())
            .expect("rtx-5080 has a GPU")
    }

    fn shape(blocks: u64, block: u64, smem: u64, per_sm: u64) -> Shape {
        Shape {
            blocks,
            block,
            smem,
            per_sm,
            args: 3,
        }
    }

    #[test]
    fn small_grids_merge_and_full_last_waves_do_not() {
        let g = gpu();
        // Three one-block launches: one wave together.
        let tiny = [
            shape(1, 256, 0, 6),
            shape(2, 32, 66048, 1),
            shape(1, 256, 0, 6),
        ];
        assert!(side_by_side(&tiny, g).is_ok());
        // Two cells of 4096 blocks: their last waves (64 each) fit one.
        let w = g.sms * 6;
        let big = [shape(4096, 256, 0, 6), shape(4096, 256, 0, 6)];
        assert_eq!(side_by_side(&big, g), Ok(w));
        // Two grids whose last waves are full: no wave saved.
        let full = [shape(w, 256, 0, 6), shape(w, 256, 0, 6)];
        assert_eq!(side_by_side(&full, g), Err(Refuse::Waves));
        // A member's residency falls at the merged footprint: refused.
        let occ = [shape(4 * w, 256, 0, 6), shape(1, 256, 90 * 1024, 1)];
        assert_eq!(side_by_side(&occ, g), Err(Refuse::Occupancy));
        // Arguments over one launch's.
        let many = [Shape { args: 9, ..tiny[0] }, Shape { args: 8, ..tiny[2] }];
        assert_eq!(side_by_side(&many, g), Err(Refuse::Args));
    }

    #[test]
    fn merged_residency_keeps_each_members_registers() {
        let g = gpu();
        // A 128-thread tile at 4 blocks / SM (512 threads' registers) in a
        // 256-thread grid: 2 blocks / SM.
        let s = [shape(10, 128, 0, 4), shape(10, 256, 0, 6)];
        assert_eq!(merged_per_sm(&s, g), 2);
    }

    #[test]
    fn a_kernel_becomes_a_device_function_of_its_block() {
        let k = "define ptx_kernel void @mapal_tile_00000000000000aa(ptr addrspace(1) noalias readonly align 16 %a, i64 %c0) \"nvvm.maxntid\"=\"32\" \"nvvm.minctasm\"=\"2\" {\n\
                 entry:\n  %v1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n  call void @llvm.nvvm.barrier0()\n  ret void\n}\n";
        let f = device_fn(k, "mapal_tile_00000000000000aa", Some(32));
        assert!(f.starts_with("define internal void @hz.mapal_tile_00000000000000aa(ptr addrspace(1) noalias readonly align 16 %a, i64 %c0, i32 %hz.bx, i32 %hz.by) alwaysinline {"));
        assert!(f.contains("%v1 = add i32 %hz.bx, 0") && f.contains("%v2 = add i32 %hz.by, 0"));
        assert!(f.contains("barrier.sync.cnt(i32 0, i32 32)") && !f.contains("barrier0"));
        let h = parse_head(k.lines().next().unwrap());
        assert_eq!(h.minctasm, Some(2));
        assert_eq!(h.params[0].0, "ptr addrspace(1)");
        assert_eq!(h.params[1].0, "i64");
        let spec = HzSpec {
            name: "mapal_tile_00000000000000bb".into(),
            block: 256,
            members: vec![
                ("mapal_tile_00000000000000aa".into(), 2, 1, 32),
                ("mapal_tile_00000000000000aa".into(), 1, 3, 32),
            ],
            level: vec![0, 0],
            zero: vec![None, None],
        };
        let m = merged_kernel(&spec, &[&h, &h]);
        assert!(m.contains("@mapal_tile_00000000000000bb(ptr addrspace(1) noalias readonly align 16 %h0.a, i64 %h0.c0, ptr addrspace(1) noalias readonly align 16 %h1.a, i64 %h1.c0)"));
        // minctasm 2 at 32 threads: 64 threads, under one 256-thread block.
        assert!(m.contains("\"nvvm.maxntid\"=\"256\" \"nvvm.minctasm\"=\"1\""));
        assert!(m.contains("icmp ult i32 %hz.b, 2") && m.contains("sub i32 %hz.b, 2"));
        assert!(m.contains("icmp ult i32 %hz.t, 32"));
    }

    #[test]
    fn a_member_launched_nowhere_else_leaves_only_its_device_function() {
        let k = |n: &str| {
            format!(
                "; cell\n; mapal.plan kernel={n} kind=cell uniform=0\ndefine ptx_kernel void @{n}(ptr addrspace(1) %o) \"nvvm.maxntid\"=\"256\" {{\nentry:\n  ret void\n}}\n"
            )
        };
        let (a, b) = ("mapal_tile_000000000000000a", "mapal_tile_000000000000000b");
        let module = format!("; head\n{}\n{}", k(a), k(b));
        let spec = HzSpec {
            name: "mapal_tile_000000000000000c".into(),
            block: 256,
            members: vec![(a.into(), 4, 1, 256), (b.into(), 2, 1, 256)],
            level: vec![0, 0],
            zero: vec![None, None],
        };
        let live: std::collections::BTreeSet<String> = [b.to_string()].into();
        let out = compose(module, &[spec], &live);
        assert!(
            !out.contains(&format!("void @{a}("))
                && !out.contains(&format!("kernel={a} kind=cell"))
        );
        assert!(out.contains(&format!("define ptx_kernel void @{b}(")));
        assert!(out.contains(&format!("define internal void @hz.{a}(")));
        assert!(out.contains(&format!("define internal void @hz.{b}(")));
        assert!(out.contains("define ptx_kernel void @mapal_tile_000000000000000c(ptr addrspace(1) %h0.o, ptr addrspace(1) %h1.o)"));
        assert_eq!(out.matches("; cell\n").count(), 1);
    }

    #[test]
    fn launch_lines_parse() {
        let Some(Parsed::Site {
            global, nptr, out, ..
        }) = parse_launch(
            "call void @mapal_gpu_site(i64 -5, i32 1, i32 0, i32 4, ptr @mapal_main.cands152, i32 3, ptr %p153, i32 2, ptr null)",
        )
        else {
            panic!("site")
        };
        assert_eq!((global.as_str(), nptr, out), ("@mapal_main.cands152", 3, 2));
        assert!(matches!(
            parse_launch(
                "call void @mapal_gpu_launchv(i64 7, i32 0, i32 6, ptr %dv1, i32 4, i32 1, i32 256, i32 0) ; unit"
            ),
            Some(Parsed::Launchv { nptr: 6, .. })
        ));
        // A dynamic grid, a split's tag, a stream: not taken.
        assert!(parse_launch("call void @mapal_gpu_launchv(i64 7, i32 0, i32 6, ptr %dv1, i32 %t9, i32 1, i32 256, i32 0)").is_none());
        assert!(parse_launch("call void @mapal_gpu_launch(i64 7, i64 %a, i64 %b, i64 %o, i32 4, i32 2, i32 256, i32 0) ; splitk=2 atomic").is_none());
        assert!(parse_launch("call void @mapal_gpu_launchv(i64 7, i32 1, i32 3, ptr %p, i32 4, i32 1, i32 256, i32 0)").is_none());
        let rows = table_rows(
            "@g = private unnamed_addr constant [2 x %mapal_cand] [%mapal_cand { i64 -3, i64 0, i64 0, i64 0, i32 1, i32 1, i32 256, i32 0, i32 0, i32 0 }, %mapal_cand { i64 4, i64 0, i64 1024, i64 0, i32 2, i32 2, i32 32, i32 66048, i32 0, i32 0 }]\n",
            "@g",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[1].zero, rows[1].gx, rows[1].smem), (1024, 2, 66048));
    }
}
