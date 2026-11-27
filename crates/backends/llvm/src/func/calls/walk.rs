//! the `calls` chain walk on `FnEmit`: place, work estimate, atomic level and close

use super::*;

impl FnEmit<'_> {
    /// **`calls`** over one sequence (module doc): every chain the rule
    /// forms becomes one persistent group of `plan.hz` (the `horizontal`
    /// groups it absorbs marked in `dead`).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::func) fn calls_walk(
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
    pub(super) fn calls_place(
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
    pub(super) fn calls_work_ns(
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
    pub(super) fn calls_c0_atomic(&self, m: MorphismId) -> bool {
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
    pub(super) fn calls_close(
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
        // launch with its other trips: review-fixes — a counted loop's trips
        // (`trip_split`: its constant `[lo, hi)`), each paying its share
        // (the rtx-5080's 2-trip `px_rt_mlp2_128` / `px_rg_mlp2_128`: the
        // first launch 57 against 14 µs, the chains formed with no share
        // measured 1.77× / 1.48× their separate launches); a loop no fact
        // counts shares it as before (0: no term); outside every loop the
        // chain pays it whole (1).
        let region = self.regions();
        let mut launches = 1u64;
        for &(i, ..) in &c.members {
            let n = match seq[i] {
                Step::Op(m) => self.launches_in(region.get(m).and_then(|r| r.0)),
                Step::Loop(_) => 0,
            };
            launches = match (launches, n) {
                (_, 0) | (0, _) => 0,
                (a, b) => a.max(b),
            };
        }
        let rounds = chain_rounds(&lv, grid);
        stats.rounds = stats.rounds.max(rounds);
        let member_ns = |i: usize| match seq[i] {
            Step::Op(m) => self.calls_work_ns(&access[i], m, gpu),
            Step::Loop(_) => 0.0,
        };
        let work: f64 = c.members.iter().map(|&(i, ..)| member_ns(i)).sum();
        // calls-issue: each unit's device time (its members' phases) and the
        // host's issue time, where the rung is on.
        let unit_ns: Vec<f64> = c
            .units
            .iter()
            .map(|u| u.iter().map(|&i| member_ns(i)).sum())
            .collect();
        let issue = match gpu::host_rungs(self.profile, self.contract, self.gpu_arms).calls_issue {
            true => gpu.cost.map_or(0, |d| d.issue_ns) as f64,
            false => 0.0,
        };
        let Some((per, sep)) = chain_ns(
            &unit_blocks,
            grid,
            barriers,
            gpu,
            pdl,
            launches,
            rounds,
            work,
            &unit_ns,
            issue,
        ) else {
            return;
        };
        // The intent line names a chain only where the host term moved its
        // decision (a module whose decisions it did not move keeps its text).
        let (_, sep0) = chain_ns(
            &unit_blocks,
            grid,
            barriers,
            gpu,
            pdl,
            launches,
            rounds,
            work,
            &[],
            0.0,
        )
        .expect("priced above");
        if (per > sep) != (per > sep0) {
            stats.issue_ns = issue;
            stats.priced.push(format!(
                "{per:.0}:{sep0:.0}>{sep:.0}:{}",
                unit_ns
                    .iter()
                    .map(|u| format!("{u:.0}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
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
