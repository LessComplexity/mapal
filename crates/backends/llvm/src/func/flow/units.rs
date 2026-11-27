//! `flow_units` and `unit_of`: which morphisms form one row-block unit and how its members lower

use super::*;

impl<'a> FnEmit<'a> {
    /// **The rule (`flow`)**: group the placed folds and cells
    /// of this plan into unit kernels, greedily in position order — a group
    /// starts at a unit no group holds and takes, one at a time, the earliest
    /// reader of a member that [`Self::unit_of`] admits. Refusals are
    /// recorded with their reason for the host intent line.
    pub(in crate::func) fn flow_units(&self, plan: &mut ResidentPlan) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let mut refused: Vec<(MorphismId, MorphismId, &'static str)> = Vec::new();
        let mut absorbed = 0;
        // compile-time-graph A: each row-block group's refusal (its launch).
        let mut block_refused: Vec<(MorphismId, String)> = Vec::new();
        // flash-attention `stream`: each stream group's refusal.
        let mut stream_refused: Vec<(MorphismId, String)> = Vec::new();
        if rungs.flow && plan.chain.is_empty() {
            let mut cand: SecondaryMap<MorphismId, Cand> = SecondaryMap::new();
            for (m, r) in plan.rowfold.iter() {
                cand.insert(m, Cand::Fold(r.clone()));
            }
            // ponytail: a unit kernel takes no scalar argument; a cell that
            // reads a uniform runs alone. Pass uniforms to units when one pays.
            for (m, c) in plan
                .cells
                .iter()
                .filter(|(_, c)| c.uniform.is_empty() && c.nested.is_empty() && c.retain.is_none())
            {
                cand.insert(m, Cand::Cell(c.clone()));
            }
            let fp = self.ir.footprint_plan(self.f);
            let region = self.regions();
            let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
            let target = |m: MorphismId| self.ir.morphism(m).expect("unit").target;
            let mut order: Vec<MorphismId> = cand.keys().collect();
            order.sort_by_key(|&m| pos(m));
            let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            let mut groups: Vec<Vec<MorphismId>> = Vec::new();
            // compile-time-graph A: the row-block groups `unit_plan` proves
            // closed — tile sites and the folds and cells reading their rows —
            // each one unit kernel where the part holds its rows on chip.
            let mut blocks: Vec<Unit> = Vec::new();
            // The part's rung table decides where the rule runs (a forced
            // `b` measures it anywhere).
            if self.gpu_arms.row_block && (rungs.row_block || self.gpu_arms.row_block_b > 0) {
                let streams = self.ir.stream_plan(self.f);
                for (gi, grp) in self.ir.unit_plan(self.f).groups.into_iter().enumerate() {
                    // sweep-attention: an extended group's stream covers its
                    // members up to its last consumer; the ones after it
                    // (reading only its value) launch on their own after it.
                    let n = match streams.of(gi) {
                        Some(sg) if grp.extended && rungs.stream => sg.roles.len(),
                        _ => grp.members.len(),
                    };
                    let ms: Vec<MorphismId> = grp.members[..n].iter().map(|mb| mb.m).collect();
                    let mut bc = cand.clone();
                    let placed = grp.members[..n].iter().all(|mb| match &mb.kind {
                        UnitKind::Tile(t) => {
                            // The site as the device plan holds it, outside
                            // an epilogue group and a retained row.
                            let site = self
                                .tile_plan
                                .as_ref()
                                .and_then(|p| p.sites.get(mb.m))
                                .cloned()
                                .unwrap_or_else(|| t.clone());
                            // flash-attention F2: an extended group's
                            // epilogue site is its own group, its post alone.
                            let epi = match grp.extended {
                                true => self
                                    .groups
                                    .get(mb.m)
                                    .is_none_or(|e| e.chain.is_empty() && e.terms.is_empty()),
                                false => !self.groups.contains_key(mb.m),
                            };
                            // An extended group's tile may run as a device
                            // cell kernel today (a narrow `P·V`): the unit
                            // takes it as its site.
                            let on_device = self.device.contains_key(mb.m)
                                || (grp.extended && plan.cells.contains_key(mb.m));
                            // sweep-attention `unit-uniform`: an extended
                            // group's launch moves a captured number's
                            // operand (`GMove`); a row-block unit's does not.
                            let ok = on_device
                                && epi
                                && !self.retain.contains_key(mb.out)
                                && (grp.extended || !gpu::uniform_site(&site));
                            if ok {
                                bc.insert(mb.m, Cand::Tile(site));
                            }
                            ok
                        }
                        _ => cand.contains_key(mb.m) && !taken.contains_key(mb.m),
                    });
                    if !placed {
                        block_refused.push((ms[0], "placed".into()));
                        continue;
                    }
                    let res = self.unit_of(&ms, &bc, &fp, &region, plan, Some(&grp));
                    if let Some(why) = self.stream_why.take() {
                        stream_refused.push((
                            ms[0],
                            match (why, self.stream_priced.take()) {
                                ("cost", Some((s, p))) => format!("cost:{s}/{p}"),
                                _ => why.into(),
                            },
                        ));
                    }
                    match res {
                        Ok(u) => {
                            for &m in &ms {
                                taken.insert(m, ());
                            }
                            blocks.push(u);
                        }
                        // A cost refusal names the model's cheapest `b`:
                        // `cost:b<b>:<fused>/<split>` ns.
                        Err(why) => block_refused.push((
                            ms[0],
                            match (why, self.block_priced.take()) {
                                ("cost", Some((b, f, s))) => format!("cost:b{b}:{f}/{s}"),
                                _ => why.into(),
                            },
                        )),
                    }
                }
            }
            for &seed in &order {
                if taken.contains_key(seed) {
                    continue;
                }
                let mut g = vec![seed];
                loop {
                    let mut next: Vec<(MorphismId, MorphismId)> = g
                        .iter()
                        .flat_map(|&x| {
                            self.cell_readers(target(x))
                                .into_iter()
                                .map(move |r| (x, r))
                        })
                        .filter(|&(_, r)| {
                            cand.contains_key(r) && !taken.contains_key(r) && !g.contains(&r)
                        })
                        .collect();
                    next.sort_by_key(|&(_, r)| pos(r));
                    next.dedup_by_key(|(_, r)| *r);
                    let mut admitted = false;
                    for (p, r) in next {
                        let mut trial = g.clone();
                        trial.push(r);
                        match self.unit_of(&trial, &cand, &fp, &region, plan, None) {
                            Ok(_) => {
                                refused.retain(|&(_, c, _)| c != r);
                                g = trial;
                                admitted = true;
                                break;
                            }
                            Err(why) => {
                                refused.retain(|&(_, c, _)| c != r);
                                refused.push((p, r, why));
                            }
                        }
                    }
                    if !admitted {
                        break;
                    }
                }
                // A lone fold is a unit of one where its lanes may be a tree:
                // the row's steps across a block, not one thread's walk. Not
                // over a column-major operand (`rowfold-layout`'s sole-reader
                // store): there a row's cells are `rows` apart, so the lanes
                // would not coalesce and one thread's walk down a column does
                // (cpu-gpu-boundary item 5: rtx-4090 rowsum_4096 1.056).
                let tree = |m: MorphismId| {
                    self.order == crate::Order::Free
                        && matches!(cand.get(m), Some(Cand::Fold(r))
                            if tree_step(&r.site.op) && r.site.k > 1 && (r.colmajor.is_none() || !rungs.tree_gate))
                };
                if g.len() >= 2
                    || (tree(seed) && self.unit_of(&g, &cand, &fp, &region, plan, None).is_ok())
                {
                    for &m in &g {
                        taken.insert(m, ());
                    }
                    groups.push(g);
                }
            }
            for g in groups {
                let u = self
                    .unit_of(&g, &cand, &fp, &region, plan, None)
                    .expect("an admitted group is a unit");
                for (k, mb) in u.members.iter().enumerate() {
                    plan.cells.remove(mb.m);
                    plan.rowfold.remove(mb.m);
                    if k > 0 {
                        plan.skip.insert(mb.m, ());
                    }
                    if !mb.store {
                        for v in plan.free_after.values_mut() {
                            v.retain(|&o| o != mb.out);
                        }
                    }
                }
                refused.retain(|&(_, c, _)| !g.contains(&c));
                plan.units.insert(g[0], u);
            }
            for u in blocks {
                for (k, mb) in u.members.iter().enumerate() {
                    plan.cells.remove(mb.m);
                    plan.rowfold.remove(mb.m);
                    if k > 0 {
                        plan.skip.insert(mb.m, ());
                    }
                    if !mb.store {
                        for v in plan.free_after.values_mut() {
                            v.retain(|&o| o != mb.out);
                        }
                    }
                }
                // A tile member absorbs a device launch only where the plan
                // held it as one: an extended group's tile placed as a cell
                // kernel (a narrow `P·V`) left `plan.cells` above.
                absorbed += u
                    .members
                    .iter()
                    .filter(|mb| {
                        matches!(mb.kind, UKind::Tile { .. }) && self.device.contains_key(mb.m)
                    })
                    .count();
                let g: Vec<MorphismId> = u.members.iter().map(|mb| mb.m).collect();
                refused.retain(|&(_, c, _)| !g.contains(&c));
                plan.units.insert(g[0], u);
            }
        }
        // The host intent: every group, every launch, every refused edge.
        let launches = self.device.len() - absorbed
            + plan.rowfold.len()
            + plan.cells.len()
            + plan.units.len()
            + plan.devgen.len();
        let name = |m: MorphismId| self.object_label(self.ir.morphism(m).expect("m").target);
        let refused: Vec<String> = refused
            .iter()
            .map(|&(p, c, why)| format!("{}->{}:{why}", name(p), name(c)))
            .collect();
        // Only where the rule decided something: a function it leaves alone
        // keeps its text.
        let block: String = block_refused
            .iter()
            .map(|(m, why)| format!(" row_block={}:{why}", name(*m)))
            .chain(
                stream_refused
                    .iter()
                    .map(|(m, why)| format!(" stream={}:{why}", name(*m))),
            )
            .collect();
        if !plan.units.is_empty() || !refused.is_empty() || !block.is_empty() {
            plan.flow_line = format!(
                "; mapal.flow.host groups={} units={} launches={launches} refused={}{block}\n",
                self.groups.len() + plan.units.len(),
                plan.units.len(),
                refused.join(";")
            );
        }
    }

    /// An object's name in an intent line: its source name, else `o<seq>`.
    pub(in crate::func) fn object_label(&self, o: ObjectId) -> String {
        let obj = self.ir.object(o).expect("object resolves");
        match &obj.name {
            Some(n) => n.clone(),
            None => format!("{o:?}")
                .trim_start_matches("ObjectId(")
                .trim_end_matches(')')
                .replace('v', "_")
                .chars()
                .fold(String::from("o"), |mut s, c| {
                    s.push(c);
                    s
                }),
        }
    }

    /// The unit of the members `ms` (position order), or why not: `phase`
    /// (another region, or an operand not complete at the launch), `gated`,
    /// `footprint`, `pointers`, `budget`, `cost`.
    pub(super) fn unit_of(
        &self,
        ms: &[MorphismId],
        cand: &SecondaryMap<MorphismId, Cand>,
        fp: &mapal_ir::FootprintPlan,
        region: &SecondaryMap<MorphismId, (Option<ObjectId>, bool)>,
        plan: &ResidentPlan,
        grp: Option<&UnitGroup>,
    ) -> Result<Unit, &'static str> {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let launch = ms[0];
        let rs = *region.get(launch).ok_or("phase")?;
        if ms.iter().any(|&m| region.get(m) != Some(&rs)) {
            return Err("phase");
        }
        if ms.iter().any(|&m| self.gated.contains_key(m)) {
            return Err("gated");
        }
        let target = |m: MorphismId| self.ir.morphism(m).expect("member").target;
        let extent = |m: MorphismId| array_len(&self.obj_ty(target(m)));
        let fold_rows: Vec<u64> = ms
            .iter()
            .filter_map(|&m| match &cand[m] {
                Cand::Fold(r) => Some(r.site.rows),
                Cand::Cell(_) | Cand::Tile(_) => None,
            })
            .collect();
        let rows = match grp {
            // A row-block group's axis is `unit_plan`'s.
            Some(g) => g.rows,
            None => fold_rows.first().copied().unwrap_or_else(|| extent(ms[0])),
        };
        let unaligned = match grp {
            Some(_) => ms.iter().any(|&m| extent(m) % rows != 0),
            None => {
                fold_rows.iter().any(|&r| r != rows)
                    || ms.iter().any(|&m| extent(m) % rows != 0)
                    || (fold_rows.is_empty() && ms.iter().any(|&m| extent(m) != rows))
            }
        };
        if rows == 0 || unaligned {
            return Err("footprint");
        }
        // Complete in `Gmem` when the group launches, where it runs: made by
        // a morphism before the launch in this region or straight-line, or a
        // view of this loop's state.
        let complete = |o: ObjectId| match self.ir.in_edges(o) {
            [] => true,
            [p] => {
                let made = pos(*p) < pos(launch)
                    && region.get(*p).is_some_and(|&r| r == rs || r.0.is_none());
                let view = rs.0.is_some_and(|mg| {
                    let mo = self.ir.morphism(*p).expect("morphism resolves");
                    matches!(mo.op, Operation::Proj { .. }) && mo.source == mg
                });
                made || view
            }
            _ => false,
        };
        let reach_of = |m: MorphismId, slot: u32| {
            fp.of(m)
                .iter()
                .find(|e| e.slot == slot && e.law.is_none())
                .map(|e| e.reach.clone())
        };
        let mut members: Vec<UMember> = Vec::new();
        let mut gmem: Vec<ObjectId> = Vec::new();
        let mut gmove: Vec<Option<GMove>> = Vec::new();
        // An operand entry a pointer: one per array and move.
        let push_moved = |o: ObjectId,
                          mv: Option<GMove>,
                          gmem: &mut Vec<ObjectId>,
                          gmove: &mut Vec<Option<GMove>>| {
            match (0..gmem.len()).find(|&g| gmem[g] == o && gmove[g] == mv) {
                Some(g) => g,
                None => {
                    gmem.push(o);
                    gmove.push(mv);
                    gmem.len() - 1
                }
            }
        };
        let push = |o: ObjectId, gmem: &mut Vec<ObjectId>, gmove: &mut Vec<Option<GMove>>| {
            push_moved(o, None, gmem, gmove)
        };
        // May a reader with `cw` cells per row read member `p` at `reach`?
        let readable = |p: &UMember, reach: &Reach, cw: u64| -> Result<bool, &'static str> {
            let free = p.srcs.iter().all(|s| !matches!(s, USrc::Member { .. }));
            // Every read of the slot at one affine address (a body that
            // reads it at two takes the range, and no one-cell buffer).
            if !matches!(reach, Reach::Affine { .. }) {
                return Err("footprint");
            }
            match p.kind {
                UKind::Fold { .. } if rowwise_single(reach, cw, rows) => Ok(true),
                UKind::Fold { .. } => Err("footprint"),
                // A tile's rows are on chip: read in its own row, never a halo.
                UKind::Tile { .. } if rowwise(reach, cw, p.w, rows) => Ok(true),
                UKind::Tile { .. } => Err("footprint"),
                UKind::Cell { .. } if rowwise(reach, cw, p.w, rows) => Ok(true),
                UKind::Cell { .. } if free => Ok(false),
                UKind::Cell { .. } => Err("footprint"),
            }
        };
        for &m in ms {
            let morph = self.ir.morphism(m).expect("member");
            let w = extent(m) / rows;
            let member =
                |o: ObjectId, members: &[UMember]| members.iter().position(|mb| mb.out == o);
            let mb = match &cand[m] {
                Cand::Fold(r) => {
                    let a = self
                        .pair_source(morph.source, r.site.a.slot)
                        .ok_or("footprint")?;
                    let src = match member(a, &members) {
                        Some(k) => {
                            let reach = reach_of(m, r.site.a.slot).ok_or("footprint")?;
                            // A fold's step reads `k` cells of its row: the
                            // producer is a cell (recomputed at each), never
                            // another fold.
                            if !matches!(members[k].kind, UKind::Cell { .. } | UKind::Tile { .. }) {
                                return Err("footprint");
                            }
                            let row = readable(&members[k], &reach, 1)?;
                            USrc::Member { k, reach, row }
                        }
                        None if complete(a) => USrc::Gmem(push(a, &mut gmem, &mut gmove)),
                        None => return Err("phase"),
                    };
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Fold {
                            site: r.site.clone(),
                            colmajor: r.colmajor,
                        },
                        w,
                        store: false,
                        srcs: vec![src],
                    }
                }
                Cand::Tile(site) => {
                    let a = self
                        .pair_source(morph.source, site.a.slot)
                        .ok_or("footprint")?;
                    let b = self
                        .pair_source(morph.source, site.b.slot)
                        .ok_or("footprint")?;
                    // sweep-attention `unit-uniform`: an operand a captured
                    // number moves is its own pointer, moved at the launch.
                    let moved = |a: bool| {
                        let r = site.radix.as_deref()?;
                        let (ds, read) = if a {
                            (&r.ua, &site.a)
                        } else {
                            (&r.ub, &site.b)
                        };
                        (!ds.is_empty()).then(|| GMove {
                            source: morph.source,
                            terms: ds.clone(),
                            elem: read.elem_bytes(&site.elem),
                        })
                    };
                    // `a` is the unit's own rows (a row-wide member read in
                    // its row, row-aligned) or complete in `Gmem`; `b` is read
                    // by every row: never a member.
                    // flash-attention F1: a row operand `unit_plan` proved
                    // a whole row under a digit permutation of the rows.
                    // sweep-attention: an extended group's row operand the
                    // fact proved row-closed where `Reach` cannot (a one-head
                    // trip's head digit; `unit_plan`'s own-read check) —
                    // the stream reads it window by window, never by reach.
                    let digit = grp.is_some_and(|g| {
                        g.members.iter().find(|x| x.m == m).is_some_and(|x| {
                            x.row_of.is_some()
                                || (g.extended && x.reads.iter().any(|&(s, _)| s == site.a.slot))
                        })
                    });
                    let a_src = match member(a, &members) {
                        Some(k) if digit => USrc::Member {
                            k,
                            reach: SAME,
                            row: true,
                        },
                        Some(k) => {
                            let p = &members[k];
                            if p.per_row()
                                || (site.a.ci != p.w && rows > 1)
                                || site.a.base + site.a.ck * (site.k - 1) >= p.w
                            {
                                return Err("footprint");
                            }
                            let reach = reach_of(m, site.a.slot).ok_or("footprint")?;
                            if !readable(p, &reach, site.c)? {
                                return Err("footprint");
                            }
                            USrc::Member {
                                k,
                                reach,
                                row: true,
                            }
                        }
                        None if complete(a) => {
                            USrc::Gmem(push_moved(a, moved(true), &mut gmem, &mut gmove))
                        }
                        None => return Err("phase"),
                    };
                    // A member row operand is the unit's own: no pointer moves.
                    if matches!(a_src, USrc::Member { .. }) && moved(true).is_some() {
                        return Err("footprint");
                    }
                    let b_src = match member(b, &members) {
                        Some(_) => return Err("footprint"),
                        None if complete(b) => {
                            USrc::Gmem(push_moved(b, moved(false), &mut gmem, &mut gmove))
                        }
                        None => return Err("phase"),
                    };
                    let mut srcs = vec![a_src, b_src];
                    // `tile-post` (an extended group's): each array its
                    // epilogue reads, after `a` and `b` in its slots' order —
                    // memory, or a per-row member at the row it ends in.
                    for &slot in site.post.iter().flat_map(|p| &p.reads) {
                        let arr = self.pair_source(morph.source, slot).ok_or("footprint")?;
                        srcs.push(match member(arr, &members) {
                            Some(k) if members[k].per_row() => USrc::Member {
                                k,
                                reach: SAME,
                                row: true,
                            },
                            Some(_) => return Err("footprint"),
                            None if complete(arr) => USrc::Gmem(push(arr, &mut gmem, &mut gmove)),
                            None => return Err("phase"),
                        });
                    }
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Tile { site: site.clone() },
                        w,
                        store: false,
                        srcs,
                    }
                }
                Cand::Cell(c) => {
                    let Operation::Map { body, captures } = morph.op else {
                        return Err("footprint");
                    };
                    let mut srcs = Vec::new();
                    for slot in 0..=captures {
                        let arr = if captures == 0 {
                            Some(morph.source)
                        } else {
                            self.pair_source(morph.source, slot)
                        }
                        .ok_or("footprint")?;
                        let law = c.laws.iter().find(|&&(s, _)| s == slot).map(|&(_, f)| f);
                        let src = if slot == captures && c.mapped.is_none() {
                            USrc::Index
                        } else if let Some(f) = law {
                            USrc::Law(f)
                        } else if let Some(k) = member(arr, &members) {
                            let reach = if slot == captures {
                                SAME
                            } else {
                                reach_of(m, slot).ok_or("footprint")?
                            };
                            let row = readable(&members[k], &reach, w)?;
                            USrc::Member { k, reach, row }
                        } else if complete(arr) {
                            USrc::Gmem(push(arr, &mut gmem, &mut gmove))
                        } else {
                            return Err("phase");
                        };
                        srcs.push(src);
                    }
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Cell { body, captures },
                        w,
                        store: false,
                        srcs,
                    }
                }
            };
            members.push(mb);
        }
        for mb in &mut members {
            mb.store = self.lup.escapes(mb.out)
                || plan.is_carry_next(mb.out)
                || self.read_outside(mb.out, ms);
        }
        // Budget: one pointer per operand and stored value; per-row values
        // held in registers beside the address state.
        let gpu = self.profile.gpu().expect("a unit has a GPU profile");
        let stored = members.iter().filter(|mb| mb.store).count();
        if gmem.len() + stored > gpu::LAUNCHV_MAX {
            return Err("pointers");
        }
        if let Some(grp) = grp {
            // flash-attention `stream`: the group as one pass over windows
            // of its axis, where the fact proves it and the rule prices it.
            self.stream_why.set(None);
            match self.stream_choice(&members, rows, &gmem, grp) {
                Ok(Some(st)) => {
                    let key = format!(
                        "unit|{members:?}|{rows}|{}|stream{}",
                        self.profile.name,
                        st.label()
                    );
                    let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
                        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
                    });
                    return Ok(Unit {
                        id,
                        rows,
                        members,
                        gmem,
                        gmove,
                        block: None,
                        slices: None,
                        stream: Some(st),
                    });
                }
                Ok(None) => {}
                Err(why) => self.stream_why.set(Some(why)),
            }
            // The row-block units realize plain groups only.
            if grp.extended {
                return Err("extended");
            }
            let blk = self.block_choice(&members, rows, &gmem, grp)?;
            let key = format!(
                "unit|{members:?}|{rows}|{}|block{}",
                self.profile.name, blk.b
            );
            let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            });
            return Ok(Unit {
                id,
                rows,
                members,
                gmem,
                gmove,
                block: Some(blk),
                slices: None,
                stream: None,
            });
        }
        let elem = |o: ObjectId| {
            crate::profile::elem_bytes(super::super::resident::array_elem(&self.obj_ty(o)))
        };
        let held: u64 = members
            .iter()
            .filter(|mb| mb.per_row())
            .map(|mb| elem(mb.out).div_ceil(4))
            .sum::<u64>()
            + crate::profile::FMA_ADDR_REGS;
        if held > gpu.regs_per_thread {
            return Err("budget");
        }
        if let Some(d) = gpu.cost.filter(|d| d.flops_per_ns > 0) {
            let (fused, unfused) = self.unit_roofline(&members, rows, &gmem, elem);
            let ns = |(f, b): (u64, u64)| {
                crate::profile::roofline_ns(f, b, d.flops_per_ns, d.bytes_per_ns)
            };
            // plan-cost (D1, D2): a launch costs its in-stream idle after the
            // op before it, not a host round trip; `launch_ns` where the part
            // has no plan rates (D8).
            let launch = |prev: u64| {
                self.profile
                    .op_ns(prev as f64, 0)
                    .map_or(d.launch_ns, |x| x.round() as u64)
            };
            let mut prev = u64::MAX;
            let unfused: u64 = unfused
                .into_iter()
                .map(|x| {
                    let t = launch(prev) + ns(x);
                    prev = ns(x);
                    t
                })
                .sum();
            if launch(u64::MAX) + ns(fused) > unfused {
                return Err("cost");
            }
        }
        // long-rows `row-slices`: few long rows over several blocks each.
        let slices = self.row_slices(&members, rows, &gmem, self.launches_in(rs.0));
        let key = match &slices {
            None => format!("unit|{members:?}|{rows}|{}", self.profile.name),
            Some(sl) => format!(
                "unit|{members:?}|{rows}|{}|slices{}",
                self.profile.name, sl.s
            ),
        };
        let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        Ok(Unit {
            id,
            rows,
            members,
            gmem,
            gmove,
            block: None,
            slices,
            stream: None,
        })
    }
}
