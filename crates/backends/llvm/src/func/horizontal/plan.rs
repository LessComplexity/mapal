//! The `horizontal` rule over a sequence: which launches are independent and merge into one grid.
use super::*;

impl FnEmit<'_> {
    /// **`horizontal`** (module doc): fills `plan.hz` / `plan.hz_of`, moves
    /// the frees inside each group's window after its launch.
    pub(crate) fn horizontal(&self, plan: &mut ResidentPlan) {
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
            // A launch whose stored value the host copies out issues that
            // copy right after it: it is the last member of its group (the
            // merged launch stands at the last member).
            let copied = |j: usize| match seq[j] {
                Step::Op(m) => self.hz_copied(plan, m),
                Step::Loop(_) => false,
            };
            let copies: Vec<bool> = (0..seq.len()).map(copied).collect();
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
                            if copies[j] {
                                self.hz_close(plan, &seq, &acc, &access, &mut open);
                            }
                            continue;
                        }
                        Err(r) => {
                            refused[Refuse::ALL.iter().position(|&x| x == r).expect("listed")] += 1;
                            self.hz_close(plan, &seq, &acc, &access, &mut open);
                        }
                    }
                }
                open.push(j);
                if copies[j] {
                    self.hz_close(plan, &seq, &acc, &access, &mut open);
                }
            }
            self.hz_close(plan, &seq, &acc, &access, &mut open);
            // kernel-merging `calls`: chains of the sequence's units, where the
            // part's barrier and dependent-launch costs were measured.
            // Every part is priced: its measured boundaries, else derived
            // from its clock (`Gpu::chain_cost`).
            if rungs.calls {
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
                "; mapal.calls.host chains={} formed={} units={} barriers={} zeros={} per_ns={:.0} sep_ns={:.0} refused={} rounds={}\n",
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
                },
                calls.rounds
            );
            // calls-issue: the host issue time and each priced chain.
            if calls.issue_ns > 0.0 {
                plan.calls_line = format!(
                    "{} issue_ns={:.0} priced={}\n",
                    plan.calls_line.trim_end(),
                    calls.issue_ns,
                    calls.priced.join(";")
                );
            }
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
    pub(super) fn hz_close(
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
    pub(crate) fn hz_access(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        m: MorphismId,
    ) -> (Vec<ObjectId>, Vec<ObjectId>) {
        self.step_access(plan, acc, m)
    }

    /// Does a later access conflict with an earlier one (either order: a
    /// deferred launch moves past the step)?
    pub(crate) fn hz_conflict(
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
    pub(crate) fn hz_row(&self, plan: &ResidentPlan, m: MorphismId) -> Option<&TileSite> {
        let target = self.ir.morphism(m)?.target;
        let r = self.retain.get(target)?;
        let kept = !plan.is_carry_next(target) && plan.alias(target) == target;
        (kept && self.on_device(target) && gpu::gpu_site(&r.row, self.profile, true))
            .then_some(&r.row)
    }
    /// Does launch `m` copy a value out to the host right after it? Every
    /// array it stores — its target, a flow unit's stored members, an
    /// epilogue group's stored results — under the emission's own test (the
    /// plan keeps it on the device, or it is a carry's next value), and the
    /// residency alias it shares kept too. The copy is a runtime call after
    /// the launch: nothing merges across it.
    pub(crate) fn hz_copied(&self, plan: &ResidentPlan, m: MorphismId) -> bool {
        let target = self.ir.morphism(m).expect("m").target;
        let unit = plan
            .units
            .get(m)
            .into_iter()
            .flat_map(|u| u.members.iter().filter(|mb| mb.store).map(|mb| mb.out));
        let epi = self
            .groups
            .get(m)
            .into_iter()
            .flat_map(|g| g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out));
        let kept = |o: ObjectId| {
            (plan.keep.contains_key(o) || plan.is_carry_next(o))
                && plan.keep.contains_key(plan.alias(o))
        };
        std::iter::once(target)
            .chain(unit)
            .chain(epi)
            .any(|o| !kept(o))
    }

    /// The arms and order site `m`'s candidates are drawn under where it
    /// launches on its own or as a member: its epilogue group's
    /// ([`gpu::epi_order`]: a `tile-post` group splits only into a
    /// workspace; any other group is ordered), else the function's.
    pub(crate) fn site_choice(&self, m: MorphismId) -> (crate::GpuArms, crate::Order) {
        let walk = self
            .tile_plan
            .as_ref()
            .and_then(|t| t.sites.get(m))
            .is_none_or(|s| gpu::epi_walk(s, self.profile, self.gpu_arms));
        gpu::epi_order(
            self.groups.get(m).map(|g| g.post.is_some()),
            self.gpu_arms,
            self.order,
            walk,
        )
    }

    /// The launch emitted at `m`, when the rule may merge it: its rules'
    /// geometry. Cells (not a range cell or a batched write), units, and
    /// unsplit tile sites (not a linear sum's table, a column-major store, an
    /// L2-windowed launch or a row whose split launches without a table);
    /// with `atomic` (`calls`, whose kernel zeroes the output one level
    /// before) a site whose rule candidate adds into a zeroed output too.
    pub(crate) fn hz_shape(
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
            // bf16-class `device-trap`: a trap-checking cell launches alone
            // (its launch arms the checkpoint's read).
            if c.range.is_some() || c.retain.is_some() || batched || c.trap {
                return None;
            }
            let (gx, block) = gpu::CellGeo {
                range: None,
                piece: 0,
                lanes: c.lanes,
                zero: 0,
                clip: false,
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
        // long-prefill `uniform-base`: its launch moves the operands'
        // pointers by captured numbers — a launch of its own.
        if !self.device.contains_key(m) || gpu::uniform_site(site) {
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
        // The arms and order the emission draws this launch's table under
        // (a retained row's: the function's; a site's: its epilogue
        // group's, `gpu::epi_order`) — the launch priced is the one taken.
        let (arms, order) = match row {
            Some(_) => (self.gpu_arms, self.order),
            None => self.site_choice(m),
        };
        let cands = gpu::site_cands(one, self.profile, self.contract, arms, order, row.is_some());
        let c0 = cands[0].launch;
        // wgmma-16bit: a warp-specialized `wgmma` launch (tensor maps by
        // value, a producer warpgroup's `setmaxnreg` share of its block)
        // launches alone — merged, every such member faulted on the H100
        // (`illegal memory access`, 22 of 22 generated programs; alone, 26
        // of 26 within the bound). Open: what the merged kernel breaks.
        if c0.wgmma.is_some_and(|w| w.ws) {
            return None;
        }
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
            && gpu::host_rungs(self.profile, self.contract, arms).l2_window
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
            g.caps().len() + g.chain.iter().filter(|mb| mb.store).count() + g.uniforms().len()
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
