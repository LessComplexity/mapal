//! `resident_plan`: device residency decisions, chains and the bytes entering the device

use super::*;

impl<'a> FnEmit<'a> {
    /// Build the plan. `None` when the function has no device site or cell.
    pub(crate) fn resident_plan(&self, rungs: crate::GpuRungs) -> Option<ResidentPlan> {
        let (devgen, arena) = (rungs.devgen, rungs.arena);
        let mut sites = self.device_sites();
        let folds = self.fold_candidates(&mut sites);
        let (mut cells, carries) = self.cell_sites(&mut sites, &folds, devgen);
        if sites.is_empty() && cells.is_empty() {
            return None;
        }
        // `scalar-fold`: a uniform a device whole fold produces is read in
        // the cell from its buffer.
        for c in cells.values_mut() {
            c.dev = c
                .uniform
                .iter()
                .copied()
                .filter(|&s| {
                    self.ir.in_edges(c.caps[s as usize]).first().is_some_and(|&p| {
                        matches!(sites.get(p), Some(DevSite::RowFold(r)) if r.site.rows == 1
                            && matches!(self.ir.morphism(p).map(|x| x.op), Some(Operation::Fold { .. })))
                    })
                })
                .collect();
        }
        let mut plan = ResidentPlan {
            carries,
            ..ResidentPlan::default()
        };
        // full-model `once-range`: each straight-line range cell's output is
        // its source's buffer (`range_in_place` admitted it over a dead one).
        let line = self.straight_line();
        for (m, c) in cells.iter() {
            let Some(ou) = c.range else { continue };
            if line.contains_key(m) {
                let t = self.ir.morphism(m).expect("cell").target;
                plan.inplace.insert(t, c.caps[ou.slot as usize]);
            }
        }
        for (m, site) in sites.iter() {
            if let DevSite::RowFold(r) = site {
                plan.rowfold.insert(m, r.clone());
            }
        }
        let mut device = Vec::new(); // every object given a device copy here

        // Device outputs — sites', cell kernels' (`devmap`), an epilogue's
        // stored results (S54): kept on the device unless the host reads one whole.
        let outs: Vec<ObjectId> = sites
            .keys()
            .chain(cells.keys())
            .map(|m| self.ir.morphism(m).expect("site").target)
            .chain(
                self.groups
                    .values()
                    .flat_map(|g| g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out)),
            )
            .collect();
        // caching-memory (`carry-take`): a device output that is a device
        // carry's init and dies at the loop's entry is the carry's first
        // state — the entry takes its buffer, no host reader is left, so it
        // is kept (no copy out). Its only use is the init product's pair.
        let take = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).carry_take;
        let taken: Vec<(ObjectId, ObjectId)> = plan
            .carries
            .iter()
            .filter(|_| take)
            .filter_map(|c| {
                let lp = self.ir.loop_plan(self.f, c.merge)?;
                let a = self.pair_source(lp.init, c.k)?;
                let enter = self
                    .ir
                    .in_edges(c.merge)
                    .iter()
                    .copied()
                    .find(|&m| {
                        self.ir
                            .morphism(m)
                            .is_some_and(|x| x.op == Operation::LoopEnter)
                    })
                    .and_then(|m| self.lup.position(m))?;
                (self.is_view(lp.init, &[c.k])
                    && ((self.ir.out_edges(a).len() == 1
                        && self.ir.try_owner(a) == Some(self.f)
                        && !self.lup.escapes(a)
                        && self.lup.death(a).is_some_and(|d| d <= enter))
                        || self.bytes_enter(&plan, a, lp.init, enter)))
                .then_some((a, lp.init))
            })
            .collect();
        for c in outs {
            device.push(c);
            let uses = if plan.is_carry_next(c) {
                vec![] // checked device-only when the carry was admitted
            } else if taken.iter().any(|&(a, _)| a == c) {
                // The taking loop's init pair is the carry's way in (its
                // buffer); any other reader — a host fold or element before
                // the entry (`bytes_enter` admits one), another loop entered
                // with it earlier — still needs the bytes out. The state's
                // way out (`last_use_plan` counts it as the array's escape)
                // is the carry's: the array's out-edges alone.
                let into: Vec<MorphismId> = self
                    .ir
                    .out_edges(c)
                    .iter()
                    .copied()
                    .filter(|&e| {
                        self.ir.morphism(e).is_some_and(|x| {
                            matches!(x.op, Operation::Pair { .. })
                                && taken.iter().any(|&(a, init)| a == c && init == x.target)
                        })
                    })
                    .collect();
                self.edge_uses(&sites, &cells, c, &into)
            } else {
                self.uses(&sites, &cells, c, &[])
            };
            if uses.iter().all(|u| !matches!(u, Use::Host)) {
                plan.keep.insert(c, ());
            }
        }
        for carry in &plan.carries {
            if let Some(x) = carry.exit
                && self
                    .uses(&sites, &cells, x, &[])
                    .iter()
                    .all(|u| !matches!(u, Use::Host))
            {
                plan.keep.insert(x, ());
            }
        }
        // One-element copy-outs, for every kept object (and the carry views).
        let kept: Vec<ObjectId> = plan
            .keep
            .keys()
            .chain(plan.carries.iter().flat_map(|c| c.views.iter().copied()))
            .collect();
        for o in kept {
            for u in self.uses(&sites, &cells, o, &[]) {
                if let Use::Elem(m) = u {
                    plan.elem.insert(m, ());
                }
            }
        }

        // The operands device sites read (their feeders).
        let mut read: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for (m, site) in sites.iter() {
            let src = self.ir.morphism(m).expect("site").source;
            for slot in site.reads() {
                if let Some(f) = self.pair_source(src, slot) {
                    read.insert(f, ());
                }
            }
        }
        // An epilogue's operands are read in the store: there, complete.
        for g in self.groups.values() {
            for o in g.caps() {
                read.insert(o, ());
            }
            for mb in &g.chain {
                plan.skip.insert(mb.m, ());
            }
            for &t in &g.terms {
                plan.skip.insert(t, ());
            }
        }
        for c in cells.values() {
            for (o, _) in c.operands() {
                read.insert(o, ());
            }
        }
        // S65 rule R: a device carry's init array that dies at the loop entry
        // is observed by the device only (the init product forwards a
        // pointer), so a closed-form one is built there — the entry then
        // hands its buffer to the carry (`resident_loop_enter`).
        let mut carried_init: Vec<ObjectId> = Vec::new();
        for c in &plan.carries {
            let Some(lp) = self.ir.loop_plan(self.f, c.merge) else {
                continue;
            };
            let Some(a) = self.pair_source(lp.init, c.k) else {
                continue;
            };
            let kind = self.ir.object(a).expect("object resolves").kind;
            let enter = self.ir.out_edges(lp.init).first().copied();
            if self.is_view(lp.init, &[c.k])
                && self.ir.out_edges(a).len() == 1
                && kind != ObjectKind::Constant
                && kind != ObjectKind::Parameter
                && self.ir.try_owner(a) == Some(self.f)
                && enter
                    .and_then(|e| self.lup.position(e))
                    .is_some_and(|p| self.lup.dead_after(a, p))
            {
                read.insert(a, ());
                carried_init.push(a);
            }
        }
        let carried = |o: ObjectId| plan.carries.iter().any(|c| c.views.contains(&o));
        for (o, _) in read.iter() {
            if carried(o) || self.produced_on_device(&sites, &cells, o) {
                continue;
            }
            let Some(&p) = self.ir.in_edges(o).first() else {
                continue; // a parameter: copied in at the site
            };
            let init = carried_init.contains(&o);
            if devgen
                && let Some(body) = self.gen_law(p)
                && (init
                    || self
                        .uses(&sites, &cells, o, &[])
                        .iter()
                        .all(|u| matches!(u, Use::Dev | Use::Elem(_) | Use::None)))
            {
                let n = array_len(&self.obj_ty(o));
                plan.devgen.insert(
                    p,
                    Gen {
                        id: gpu::gen_id(&self.fnames[body], n, self.profile),
                        body,
                        n,
                    },
                );
                if self
                    .uses(&sites, &cells, o, &[])
                    .iter()
                    .any(|u| matches!(u, Use::Elem(_)))
                {
                    plan.keep.insert(o, ());
                    for u in self.uses(&sites, &cells, o, &[]) {
                        if let Use::Elem(m) = u {
                            plan.elem.insert(m, ());
                        }
                    }
                }
            } else if init {
                continue; // host-built: the loop entry copies it in
            } else {
                plan.upload.insert(o, ());
            }
            device.push(o);
        }
        if rungs.delta_h2d {
            plan.delta = self.delta_uploads(&plan.upload, rungs.retain_same);
        }
        // `trip-batch`, once every operand's device copy is placed: a loop
        // invariant's is made before the loop (uploaded, devgen'd or a
        // device product), so the batch at the entry reads it there.
        if rungs.trip_batch {
            let at_entry = |o: ObjectId| {
                plan.upload.contains_key(o)
                    || self
                        .ir
                        .in_edges(o)
                        .first()
                        .is_some_and(|&p| plan.devgen.contains_key(p))
                    || self.produced_on_device(&sites, &cells, o)
            };
            plan.batch = self.batches(&cells, &plan.carries, &at_entry);
        }

        // `cell-law`: a closed-form producer whose every reader evaluates it
        // in a cell (an unloaded slot) is built nowhere.
        if cells.values().any(|c| !c.laws.is_empty()) {
            for (m, _) in self.ir.morphisms() {
                let t = self.ir.morphism(m).expect("morphism resolves").target;
                if self.ir.try_owner(t) == Some(self.f)
                    && !self.lup.escapes(t)
                    && self.gen_law(m).is_some()
                    && !self.ir.out_edges(t).is_empty()
                    && self
                        .uses(&sites, &cells, t, &[])
                        .iter()
                        .all(|u| *u == Use::None)
                {
                    plan.skip.insert(m, ());
                }
            }
        }

        // Inputs no reader is left for: an `Iota`/`Fill` (trap-free by
        // definition) read only by device-built producers and by device-site
        // slots the kernel never loads.
        for (m, _) in self.ir.morphisms() {
            let morph = self.ir.morphism(m).expect("morphism resolves");
            if self.ir.try_owner(morph.target) != Some(self.f)
                || !matches!(morph.op, Operation::Iota | Operation::Fill)
                || self.lup.escapes(morph.target)
            {
                continue;
            }
            let t = morph.target;
            let unread = self.ir.out_edges(t).iter().all(|&c| {
                let cm = self.ir.morphism(c).expect("morphism resolves");
                match cm.op {
                    Operation::Map { captures: 0, .. } => {
                        plan.devgen.contains_key(c) || plan.skip.contains_key(c)
                    }
                    Operation::Pair { slot, .. } => {
                        self.pointer_only_array_component(cm.target, slot)
                            && self
                                .ir
                                .out_edges(cm.target)
                                .iter()
                                .all(|&m2| self.slot_use(&sites, &cells, m2, slot) == Use::None)
                    }
                    _ => false,
                }
            });
            if unread && !self.ir.out_edges(t).is_empty() {
                plan.skip.insert(m, ());
            }
        }

        // Free at death where both ends run once; everything else at exit.
        let line = self.straight_line();
        let by_pos: Vec<(u32, MorphismId)> = line
            .keys()
            .filter_map(|m| Some((self.lup.position(m)?, m)))
            .collect();
        for o in device {
            if plan.is_carry_next(o) || plan.alias(o) != o {
                continue;
            }
            let produced_once = self.ir.in_edges(o).iter().all(|&p| line.contains_key(p));
            let Some(death) = self.buffer_death(&plan, o) else {
                continue;
            };
            if let Some(&(_, d)) = by_pos.iter().find(|&&(p, _)| p == death)
                && produced_once
            {
                match plan.free_after.get_mut(d) {
                    Some(v) => v.push(o),
                    None => {
                        plan.free_after.insert(d, vec![o]);
                    }
                }
            }
        }
        plan.cells = cells;
        self.chains(&sites, &mut plan);
        // `flow`: the placed folds and cells, grouped.
        self.flow_units(&mut plan);
        // kernel-merging `horizontal`: independent launches in one grid —
        // before the arena, which keeps every byte a deferred launch reads
        // until its group launches.
        self.horizontal(&mut plan);
        // ponytail: the arena reuses bytes by topo position, which concurrent
        // streams do not follow; a function that opens streams keeps its own
        // buffers. Color per stream when a part ships both on one program.
        if arena && plan.chain.is_empty() {
            // caching-memory `arena-fit`: the members also placed largest
            // first; that arena where it is smaller and folds as many zero
            // passes (`zero-fold` reads the arena's byte sharing: a tighter
            // packing may put a member between a site's zero and the site).
            let birth = self.arena(&plan, false);
            let fit = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).arena_fit;
            let large = fit
                .then(|| self.arena(&plan, true))
                .filter(|a| a.bytes < birth.bytes);
            plan.arena = Some(match large {
                Some(a) if self.zero_folds(&plan, &a) >= self.zero_folds(&plan, &birth) => a,
                _ => birth,
            });
        }
        // kernel-merging `zero-fold`: zero passes into earlier kernels.
        self.zero_fold(&mut plan);
        let free = self.carry_host_free(&plan);
        if !free.is_empty() {
            let a = plan.arena.get_or_insert_with(Arena::default);
            for o in free {
                a.host_free.insert(o, ());
            }
        }
        Some(plan)
    }

    /// **S54 R7: device sites on streams.** Straight-line device sites (a site
    /// in a loop or guard arm stays on the legacy stream) are covered by
    /// chains in topo order: a site continues the chain whose last site is one
    /// of its `PathPlan` ancestors, else opens one. Two chains are opened as
    /// streams only when
    /// - every one of these sites' outputs stays on the device (`keep`: read
    ///   by device sites and one-element copies only). A whole copy-out puts
    ///   the host between the chains: it waits for the transfer (at 512², five
    ///   to ten times one kernel) before the next launch. In the first two
    ///   sessions, diamond and mlp2 with a host join ran 0.60–0.96× on streams on
    ///   every part, and indep flipped sign between sessions
    ///   (`streams-<part>.txt`, `-confirm.txt`); and
    /// - the grids can actually share the part — the **wave rule**
    ///   ([`waves_merge`]): two independent grids' last waves fit one wave,
    ///   `t_a / w_a + t_b / w_b ≤ 1` (a site's `w` is `sms × blocks_per_sm` of
    ///   its launch geometry). Grids that fill their last waves have nothing to
    ///   overlap.
    ///
    /// Otherwise the text stays the one-stream text. With streams open, the
    /// frees between the first and last chained launch (`cuMemFree` waits for
    /// the whole device) move after the last.
    pub(super) fn chains(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        plan: &mut ResidentPlan,
    ) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let (Some(pp), true) = (self.path_plan, rungs.streams) else {
            return;
        };
        let gpu = self.profile.gpu().expect("a device site has a GPU profile");
        let line = self.straight_line();
        let task_of = |m: MorphismId| {
            pp.tasks
                .iter()
                .position(|t| matches!(t.kind, TaskKind::Split { site, .. } if site == m))
        };
        let tile = |m: MorphismId| match &sites[m] {
            DevSite::Tile(t) => Some(t),
            DevSite::RowFold(_) => None,
        };
        let mut order: Vec<(u32, MorphismId, usize)> = sites
            .keys()
            .filter(|&m| line.contains_key(m) && tile(m).is_some())
            .filter_map(|m| Some((self.lup.position(m)?, m, task_of(m)?)))
            .collect();
        order.sort();
        let device_only = order.iter().all(|&(_, m, _)| {
            plan.keep
                .contains_key(self.ir.morphism(m).expect("site").target)
        });
        if order.len() < 2 || !device_only {
            return;
        }
        // S54 integration: a launch the chains do not order — a device site off
        // the chains (a fold, a loop's site) or a cell kernel (`devmap`) — must
        // not read a chained output; they run on the legacy stream, which the
        // chains' non-blocking streams do not wait for.
        let chained: Vec<ObjectId> = order
            .iter()
            .map(|&(_, m, _)| self.ir.morphism(m).expect("site").target)
            .collect();
        let on_chain = |o: ObjectId| {
            let o = plan.alias(o);
            let writer = grouped_writer(&self.groups, o)
                .map_or(o, |m| self.ir.morphism(m).expect("group site").target);
            chained.contains(&writer)
        };
        let reads_chained = |src: ObjectId, slots: &[u32]| {
            slots
                .iter()
                .any(|&k| self.pair_source(src, k).is_some_and(on_chain))
        };
        let unordered = sites.iter().any(|(m, s)| {
            !order.iter().any(|&(_, o, _)| o == m)
                && reads_chained(self.ir.morphism(m).expect("site").source, &s.reads())
        }) || plan
            .cells
            .values()
            .any(|c| c.operands().iter().any(|&(o, _)| on_chain(o)));
        if unordered {
            return;
        }
        // Every task's ancestors (the deps, transitively).
        let mut anc: Vec<Option<std::collections::BTreeSet<usize>>> = vec![None; pp.tasks.len()];
        fn ancestors(
            pp: &PathPlan,
            t: usize,
            memo: &mut Vec<Option<std::collections::BTreeSet<usize>>>,
        ) -> std::collections::BTreeSet<usize> {
            if let Some(s) = &memo[t] {
                return s.clone();
            }
            let mut s = std::collections::BTreeSet::new();
            for &d in &pp.tasks[t].deps {
                s.insert(d);
                s.extend(ancestors(pp, d, memo));
            }
            memo[t] = Some(s.clone());
            s
        }
        let anc: Vec<_> = order
            .iter()
            .map(|&(_, _, t)| ancestors(pp, t, &mut anc))
            .collect();
        let before = |i: usize, j: usize| anc[j].contains(&order[i].2);
        let mut chain = Vec::with_capacity(order.len());
        let mut tail: Vec<usize> = Vec::new();
        for i in 0..order.len() {
            match (0..i).rev().find(|&j| before(j, i) && tail[chain[j]] == j) {
                Some(j) => {
                    chain.push(chain[j]);
                    tail[chain[j]] = i;
                }
                None => {
                    chain.push(tail.len());
                    tail.push(i);
                }
            }
        }
        if tail.len() < 2 {
            return;
        }
        let grid = |i: usize| {
            let l = gpu::launch(
                tile(order[i].1).expect("a tile site"),
                self.profile,
                self.contract,
                self.gpu_arms,
            );
            (l.gx * l.gy, l.wave(gpu))
        };
        let merge = (0..order.len()).any(|i| {
            (i + 1..order.len()).any(|j| {
                chain[i] != chain[j]
                    && !before(i, j)
                    && !before(j, i)
                    && waves_merge(grid(i), grid(j))
            })
        });
        if rungs.stream_gate && !merge {
            return;
        }
        for (i, &(_, m, _)) in order.iter().enumerate() {
            plan.chain.insert(m, chain[i] as u32 + 1);
        }
        // `chain-h2d`: an upload read only by one chain's sites (through the
        // pointer-only products that carry it) crosses on that chain's
        // stream, so a chain the `PathPlan` proves independent never waits
        // for it — the copy overlaps the other chains' kernels.
        if rungs.chain_h2d {
            for o in plan.upload.keys().collect::<Vec<_>>() {
                let mut readers = Vec::new();
                for &m in self.ir.out_edges(o) {
                    match self.ir.morphism(m).expect("morphism resolves").op {
                        Operation::Pair { slot, .. } => {
                            let t = self.ir.morphism(m).expect("m").target;
                            if !self.pointer_only_array_component(t, slot) {
                                readers.push(None);
                                continue;
                            }
                            readers.extend(
                                self.ir
                                    .out_edges(t)
                                    .iter()
                                    .map(|&r| plan.chain.get(r).copied()),
                            );
                        }
                        _ => readers.push(plan.chain.get(m).copied()),
                    }
                }
                if let Some(&Some(k)) = readers.first()
                    && readers.iter().all(|r| *r == Some(k))
                {
                    plan.upload_chain.insert(o, k);
                }
            }
        }
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let (lo, last) = (order[0].0, order[order.len() - 1].1);
        let moved: Vec<MorphismId> = plan
            .free_after
            .keys()
            .filter(|&d| d != last && (lo..pos(last)).contains(&pos(d)))
            .collect();
        for d in moved {
            let objs = plan.free_after.remove(d).expect("listed");
            match plan.free_after.get_mut(last) {
                Some(v) => v.extend(objs),
                None => {
                    plan.free_after.insert(last, objs);
                }
            }
        }
    }

    /// S54 `arena` (R6). Device side: the objects `free_after` frees — produced
    /// once and dead at a morphism that runs once — colored by their
    /// `[producer, death]` positions; anything used from a guard arm stays out
    /// (its arm runs at the `Phi`, past its position). Host side: an array no
    /// host morphism writes or reads whole — kept, device-built or never built
    /// — has no host slot. Carry objects keep both: the loop state holds the
    /// array inline.
    /// full-model `carry-take` (`once-range`): the bytes of `a` — a loop's
    /// init array, the state product `init`'s slot — are read past the loop's
    /// entry (at `enter`) only through the loop's state: `init` feeds the
    /// `LoopEnter` alone, every other use of `a` sits at or before `enter`
    /// and keeps no alias past it, and every array sharing `a`'s device
    /// buffer (an in-place chain) is dead by then. `last_use_plan` counts the
    /// state's way to the function's output as `a`'s escape (a product's
    /// every field escapes through its projections); the carry owns that way.
    pub(super) fn bytes_enter(
        &self,
        plan: &ResidentPlan,
        a: ObjectId,
        init: ObjectId,
        enter: u32,
    ) -> bool {
        let fresh = self
            .ir
            .object(a)
            .is_some_and(|o| o.kind == ObjectKind::Temporary)
            && self.ir.try_owner(a) == Some(self.f)
            && self.lup.carried_by(a).is_none();
        let into = self.ir.out_edges(init).iter().all(|&e| {
            self.ir
                .morphism(e)
                .is_some_and(|x| x.op == Operation::LoopEnter)
        });
        let uses = self.ir.out_edges(a).iter().all(|&e| {
            let Some(x) = self.ir.morphism(e) else {
                return false;
            };
            if x.target == init {
                return true;
            }
            let aliasing = matches!(
                x.op,
                Operation::Pair { .. }
                    | Operation::Proj { .. }
                    | Operation::Index
                    | Operation::Phi
                    | Operation::Call(_)
                    | Operation::Output
                    | Operation::LoopEnter
                    | Operation::LoopExit
            );
            self.lup.position(e).is_some_and(|p| p <= enter)
                && (!aliasing || self.lup.death(x.target).is_some_and(|d| d <= enter))
        });
        let root = plan.alias(a);
        let shared = std::iter::once(root)
            .chain(plan.inplace.keys().filter(|&t| plan.alias(t) == root))
            .filter(|&x| x != a)
            .all(|x| self.death_of(x).is_some_and(|d| d <= enter));
        fresh && into && uses && shared
    }
}
