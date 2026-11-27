//! Entry points of the site plans: emission, tile / move / row-fold plans and the trip split.

use super::*;

impl CategoryIr {
    /// The objects owned by function `f`, in deterministic order.
    pub(crate) fn func_objects(&self, f: FuncId) -> Vec<ObjectId> {
        self.fact(f, |s| &s.func_objects, || self.func_objects_compute(f))
    }

    pub(super) fn func_objects_compute(&self, f: FuncId) -> Vec<ObjectId> {
        self.objects
            .iter()
            .filter(|(id, _)| self.owner.get(*id) == Some(&f))
            .map(|(id, _)| id)
            .collect()
    }

    /// The per-function minimal-emission query. Classification is total over
    /// owned, non-constant, non-token objects and is a pure function of the
    /// sealed graph.
    ///
    /// Products are dissolved first. Consumer counts are then taken through
    /// those transparent products, so a field used by two consumers is named
    /// once rather than duplicated when its product wrapper disappears.
    pub fn emission_plan(&self, f: FuncId) -> EmissionPlan {
        self.fact(f, |s| &s.emission_plan, || self.emission_plan_compute(f))
    }

    pub(super) fn emission_plan_compute(&self, f: FuncId) -> EmissionPlan {
        let bounds = self.bounds_proof(f);
        let mut boundary: SecondaryMap<ObjectId, ()> = SecondaryMap::new();

        if let Some(fd) = self.func(f) {
            boundary.insert(fd.output, ());
        }
        for &o in &self.func_objects(f) {
            let obj = &self.objects[o];
            if obj.kind == ObjectKind::LoopMerge || matches!(obj.ty, Ty::Array { .. }) {
                boundary.insert(o, ());
            }
        }

        for (_, morph) in self.morphisms() {
            if self.try_owner(morph.source) != Some(f) {
                continue;
            }
            if self.emission_guarded(morph.id, &bounds) {
                boundary.insert(morph.target, ());
            }
            match morph.op {
                Operation::Output => {
                    boundary.insert(morph.source, ());
                    boundary.insert(morph.target, ());
                }
                Operation::Call(_) => {
                    boundary.insert(morph.source, ());
                }
                Operation::Map { .. }
                | Operation::Fold { .. }
                | Operation::Zip
                | Operation::Enumerate
                | Operation::Update
                | Operation::Iota
                | Operation::Fill
                    if self.objects[morph.source].ty.product_arity().is_some() =>
                {
                    boundary.insert(morph.source, ());
                }
                Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {
                    boundary.insert(morph.source, ());
                    boundary.insert(morph.target, ());
                }
                _ => {}
            }
        }

        // Every object incident to a canonical loop's decide/advance cone is a
        // statement point. Non-canonical loop endpoints were covered above.
        for scc in self.loop_structure(f) {
            for merge in scc.merges {
                if let Some(plan) = self.loop_plan(f, merge) {
                    for &m in plan.decide_order.iter().chain(&plan.advance_order) {
                        let morph = &self.morphisms[m];
                        boundary.insert(morph.source, ());
                        boundary.insert(morph.target, ());
                    }
                }
            }
        }

        let mut dissolved: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &o in &self.func_objects(f) {
            let obj = &self.objects[o];
            let Some(arity) = obj.ty.product_arity_u32() else {
                continue;
            };
            if obj.kind == ObjectKind::Constant
                || ty_contains_token(&obj.ty)
                || boundary.contains_key(o)
            {
                continue;
            }
            // Dissolution requires the product be FULLY Pair-built: the
            // redistribution below resolves fields via `pair_slot_source`,
            // which is `None` for a Proj-produced tuple — dissolving one
            // would silently drop its consumers from the counts, letting a
            // shared field chain classify Inline and duplicate textually
            // (an R-NODUP break). Proj-produced products stay countable.
            if (0..arity).all(|i| self.pair_slot_source(o, i).is_some())
                && self.out_edges(o).iter().all(|&m| {
                    matches!(self.morphisms[m].op, Operation::Proj { .. })
                        || is_pair_primitive(self.morphisms[m].op)
                })
            {
                dissolved.insert(o, ());
            }
        }

        // Ordinary edges count directly, except Pair edges into transparent
        // products. Those are replaced below by the product's actual field
        // uses: all fields for a primitive, one field for Proj.
        let mut consumers: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        for &o in &self.func_objects(f) {
            let obj = &self.objects[o];
            if obj.kind == ObjectKind::Constant
                || ty_contains_token(&obj.ty)
                || dissolved.contains_key(o)
            {
                continue;
            }
            let count = self
                .out_edges(o)
                .iter()
                .filter(|&&m| {
                    !matches!(self.morphisms[m].op, Operation::Pair { .. })
                        || !dissolved.contains_key(self.morphisms[m].target)
                })
                .count() as u32;
            consumers.insert(o, count);
        }
        for (product, _) in dissolved.iter() {
            for &m in self.out_edges(product) {
                match self.morphisms[m].op {
                    Operation::Proj { index } => {
                        if let Some(source) = self.pair_slot_source(product, index) {
                            increment(&mut consumers, source);
                        }
                    }
                    op if is_pair_primitive(op) => {
                        for &pair in self.in_edges(product) {
                            if matches!(self.morphisms[pair].op, Operation::Pair { .. }) {
                                increment(&mut consumers, self.morphisms[pair].source);
                            }
                        }
                    }
                    _ => unreachable!("dissolved products have only transparent consumers"),
                }
            }
        }

        let mut class = SecondaryMap::new();
        for &o in &self.func_objects(f) {
            let obj = &self.objects[o];
            if obj.kind == ObjectKind::Constant || ty_contains_token(&obj.ty) {
                continue;
            }
            let c = if dissolved.contains_key(o) {
                EmissionClass::Dissolved
            } else if boundary.contains_key(o) || consumers.get(o).copied().unwrap_or(0) != 1 {
                EmissionClass::Named
            } else {
                EmissionClass::Inline
            };
            class.insert(o, c);
        }
        EmissionPlan { class }
    }

    /// Whether the producer needs statement-form guards.
    pub(super) fn emission_guarded(&self, m: MorphismId, bounds: &BoundsProof) -> bool {
        let morph = &self.morphisms[m];
        match morph.op {
            Operation::Div | Operation::Mod => {
                if matches!(self.objects[morph.target].ty, Ty::Float { .. }) {
                    return false;
                }
                self.pair_slot_source(morph.source, 1)
                    .is_none_or(|o| !safe_integer_divisor(&self.objects[o]))
            }
            Operation::Index | Operation::Update => !bounds.proven(m),
            _ => false,
        }
    }

    /// Recognize affine map/fold sites whose cell chains may be interleaved
    /// without changing their operation or operand order.
    pub fn tile_plan(&self, f: FuncId) -> TilePlan {
        self.fact(f, |s| &s.tile_plan, || self.tile_plan_compute(f))
    }

    pub(super) fn tile_plan_compute(&self, f: FuncId) -> TilePlan {
        self.tile_plan_with(f, false)
    }

    /// [`Self::tile_plan`] plus the one-row sites whose lane operand is read
    /// at a lane stride other than 1: `out[j] = fold_k a[k]·b[j·s + k]`, a
    /// product against a transposed operand (`q·Kᵀ`). Geometry only — the
    /// stride is the program's. `tile_plan` keeps refusing them: the CPU tile
    /// ladder walks `b` at unit lane stride; a backend whose realization
    /// addresses any affine read (the GPU's one-thread-per-cell kernel) asks
    /// for this plan instead.
    pub fn tile_plan_strided(&self, f: FuncId) -> TilePlan {
        self.fact(
            f,
            |s| &s.tile_plan_strided,
            || self.tile_plan_strided_compute(f),
        )
    }

    pub(super) fn tile_plan_strided_compute(&self, f: FuncId) -> TilePlan {
        self.tile_plan_with(f, true)
    }

    pub(super) fn tile_plan_with(&self, f: FuncId, strided: bool) -> TilePlan {
        let mut sites = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return TilePlan { sites };
        };
        let form = TileForm {
            strided,
            narrow: false,
        };
        for &m in &fd.morphisms {
            if let Some(site) = self.tile_site(m, form).filter(|s| s.post.is_none()) {
                sites.insert(m, site);
            }
        }
        TilePlan { sites }
    }

    /// **Tile sites whose map body goes on past the fold** (device-geometry
    /// `tile-post`): the sum of products is a [`TileSite`] (strided lanes
    /// admitted, as [`Self::tile_plan_strided`]) and the cell's other terms
    /// its [`TilePost`] epilogue. Disjoint from [`Self::tile_plan`] and
    /// [`Self::tile_plan_strided`], which keep only sites whose body IS the
    /// fold, so every consumer of those reads what it always read; a backend
    /// that evaluates the epilogue in the site's store asks for these.
    pub fn tile_plan_post(&self, f: FuncId) -> TilePlan {
        self.fact(
            f,
            |s| &s.tile_plan_post,
            || {
                let mut sites = SecondaryMap::new();
                let form = TileForm {
                    strided: true,
                    narrow: false,
                };
                for &m in self.func(f).map(|d| d.morphisms.as_slice()).unwrap_or(&[]) {
                    if let Some(site) = self.tile_site(m, form).filter(|s| s.post.is_some()) {
                        sites.insert(m, site);
                    }
                }
                TilePlan { sites }
            },
        )
    }

    /// **Tile sites over narrow storage** (ADR-0043): a sum of products whose
    /// operand (one or both) is the exact widening of an element of a `bf16`
    /// / `f16` array — `acc + x[i·k + k'] · (w[k'·n + j] -> widen<f32>)` —
    /// with the strided lanes and epilogues of [`Self::tile_plan_strided`]
    /// and [`Self::tile_plan_post`] (`post` tells them apart). Each such read
    /// carries its format ([`TileRead::narrow`]); the product is the widened
    /// values' (`elem`), exact operands, so the site's bound is the f32
    /// site's. Disjoint from every other tile plan (they need the product to
    /// read the array's own elements), so their consumers read what they
    /// always read; a realization that loads the narrow format asks for
    /// these.
    pub fn tile_plan_narrow(&self, f: FuncId) -> TilePlan {
        self.fact(
            f,
            |s| &s.tile_plan_narrow,
            || {
                let mut sites = SecondaryMap::new();
                let form = TileForm {
                    strided: true,
                    narrow: true,
                };
                for &m in self.func(f).map(|d| d.morphisms.as_slice()).unwrap_or(&[]) {
                    if let Some(site) = self
                        .tile_site(m, form)
                        .filter(|s| s.a.narrow.is_some() || s.b.narrow.is_some())
                    {
                        sites.insert(m, site);
                    }
                }
                TilePlan { sites }
            },
        )
    }

    /// Recognize fold-less [`MoveSite`]s — maps that only move data, whose read
    /// address is affine in `(t÷C, t%C)`. The peer of [`Self::tile_plan`], and
    /// disjoint from it by construction: this arm requires the body to hold **no**
    /// fold, that one requires exactly one.
    pub fn move_plan(&self, f: FuncId) -> MovePlan {
        self.fact(f, |s| &s.move_plan, || self.move_plan_compute(f))
    }

    pub(super) fn move_plan_compute(&self, f: FuncId) -> MovePlan {
        let mut sites = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return MovePlan { sites };
        };
        for &m in &fd.morphisms {
            if let Some(site) = self.move_site(m) {
                sites.insert(m, site);
            }
        }
        MovePlan { sites }
    }

    /// Recognize [`RowFoldSite`]s — maps whose body is one in-order fold of one
    /// affine read. Disjoint from [`Self::tile_plan`] (its step needs a product
    /// of two reads, this one the read itself) and from [`Self::move_plan`] (no
    /// fold there).
    pub fn row_fold(&self, f: FuncId) -> SecondaryMap<MorphismId, RowFoldSite> {
        self.fact(f, |s| &s.row_fold, || self.row_fold_compute(f))
    }

    /// **The trip split** (recurrence-precision item 1): per canonical loop
    /// with a counter, its guard-cut trip intervals and each interval's
    /// recurrence ([`TripSplit`]). Built from `loop_plan`, `guard_plan` (of
    /// `f` and of each map body in the trip) and `map_law`.
    pub fn trip_split(&self, f: FuncId) -> Vec<TripSplit> {
        self.fact(
            f,
            |s| &s.trip_split,
            || {
                self.loop_structure(f)
                    .iter()
                    .filter_map(|scc| match scc.merges.as_slice() {
                        &[merge] => self.trip_split_of(f, &self.loop_plan(f, merge)?),
                        _ => None,
                    })
                    .collect()
            },
        )
    }

    pub(super) fn trip_split_of(&self, f: FuncId, plan: &LoopPlan) -> Option<TripSplit> {
        let merge = plan.merge;
        let Ty::Tuple(parts) = &self.objects.get(merge)?.ty else {
            return None;
        };
        let next = self.pair_slot_source(plan.back_route, 0)?;
        let cond = self.pair_slot_source(plan.exit_route, 1)?;
        let views = |k: u32| -> Vec<ObjectId> {
            self.out_edges(merge)
                .iter()
                .filter_map(|&m| {
                    let mm = &self.morphisms[m];
                    (mm.op == Operation::Proj { index: k }).then_some(mm.target)
                })
                .collect()
        };
        let int_const = |o: ObjectId| -> Option<i64> {
            let ob = self.objects.get(o)?;
            match (ob.kind, ob.value.as_ref()?) {
                (ObjectKind::Constant, Value::I32(v)) => Some(i64::from(*v)),
                (ObjectKind::Constant, Value::I64(v)) => Some(*v),
                _ => None,
            }
        };
        // `x op C` as `(op, C)` with `x` the operand `is_x` accepts (a flipped
        // compare is normalized).
        let compare = |o: ObjectId, is_x: &dyn Fn(ObjectId) -> bool| -> Option<(Operation, i64)> {
            let m = self.morphisms.get(*self.in_edges(o).first()?)?;
            if !matches!(
                m.op,
                Operation::Lt | Operation::Le | Operation::Gt | Operation::Ge
            ) {
                return None;
            }
            let (l, r) = (
                self.pair_slot_source(m.source, 0)?,
                self.pair_slot_source(m.source, 1)?,
            );
            let flip = |op| match op {
                Operation::Lt => Operation::Gt,
                Operation::Gt => Operation::Lt,
                Operation::Le => Operation::Ge,
                _ => Operation::Le,
            };
            if is_x(l) {
                Some((m.op, int_const(r)?))
            } else if is_x(r) {
                Some((flip(m.op), int_const(l)?))
            } else {
                None
            }
        };
        // The counter: an int slot from a constant, `t + 1`, continued while `t < hi`.
        let (counter, lo, hi) = (0..parts.len() as u32).find_map(|k| {
            if !matches!(parts[k as usize], Ty::Int { .. }) {
                return None;
            }
            let vk = views(k);
            let lo = int_const(self.pair_slot_source(plan.init, k)?)?;
            let nk = self.pair_slot_source(next, k)?;
            let add = self.morphisms.get(*self.in_edges(nk).first()?)?;
            let (a, b) = (
                self.pair_slot_source(add.source, 0)?,
                self.pair_slot_source(add.source, 1)?,
            );
            let step_one = add.op == Operation::Add
                && ((vk.contains(&a) && int_const(b) == Some(1))
                    || (vk.contains(&b) && int_const(a) == Some(1)));
            let hi = match compare(cond, &|o| vk.contains(&o))? {
                (Operation::Lt, c) => c,
                (Operation::Le, c) => c.checked_add(1)?,
                _ => return None,
            };
            (step_one && lo < hi).then_some((k, lo, hi))
        })?;
        let vc = views(counter);
        let body: SecondaryMap<MorphismId, ()> = plan
            .decide_order
            .iter()
            .chain(&plan.advance_order)
            .map(|&m| (m, ()))
            .collect();
        let truth = |op: Operation, c: i64, t: i64| match op {
            Operation::Lt => t < c,
            Operation::Le => t <= c,
            Operation::Gt => t > c,
            _ => t >= c,
        };
        let cut = |op: Operation, c: i64| match op {
            Operation::Lt | Operation::Ge => c,
            _ => c.saturating_add(1),
        };
        // Guards on the counter: `f`'s own (in the trip) and each trip map's
        // body's, whose test reads a capture the map takes from `t`.
        let mut guards: Vec<(GuardSite, Operation, i64)> = Vec::new();
        let mut all: Vec<GuardSite> = self.guard_plan(f);
        for site in self.guard_plan(f) {
            if body.contains_key(site.phi)
                && let Some((op, c)) = compare(site.cond, &|o| vc.contains(&o))
            {
                guards.push((site, op, c));
            }
        }
        for &m in plan.decide_order.iter().chain(&plan.advance_order) {
            let Operation::Map { body: b, captures } = self.morphisms[m].op else {
                continue;
            };
            let input = self.func(b)?.input;
            let src = self.morphisms[m].source;
            let from_t = |o: ObjectId| {
                self.tile_input_proj_index(o, input).is_some_and(|s| {
                    s < captures
                        && self
                            .pair_slot_source(src, s)
                            .is_some_and(|x| vc.contains(&x))
                })
            };
            let sites = self.guard_plan(b);
            for site in &sites {
                if let Some((op, c)) = compare(site.cond, &from_t) {
                    guards.push((site.clone(), op, c));
                }
            }
            all.extend(sites);
        }
        let mut cuts: Vec<i64> = guards
            .iter()
            .map(|g| cut(g.1, g.2))
            .filter(|&c| lo < c && c < hi)
            .collect();
        cuts.push(lo);
        cuts.push(hi);
        cuts.sort_unstable();
        cuts.dedup();
        let effect = body.keys().any(|m| {
            let mm = &self.morphisms[m];
            [mm.source, mm.target]
                .iter()
                .any(|&o| crate::ty::ty_contains_token(&self.objects[o].ty))
        });
        let slots: Vec<u32> = (0..parts.len() as u32)
            .filter(|&k| k != counter && !views(k).is_empty())
            .collect();
        let bounds = self.bounds_proof(f);
        let mut phases = Vec::new();
        for (n, w) in cuts.windows(2).enumerate() {
            let (plo, phi) = (w[0], w[1]);
            // The arms this interval does not take, with every site nested in them.
            let mut dead: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            for (site, op, c) in &guards {
                let off = if truth(*op, *c, plo) {
                    &site.on_false
                } else {
                    &site.on_true
                };
                dead.extend(off.own.iter().map(|&m| (m, ())));
            }
            loop {
                let before = dead.len();
                for s in &all {
                    if dead.contains_key(s.phi) {
                        dead.extend(
                            s.on_true
                                .own
                                .iter()
                                .chain(&s.on_false.own)
                                .map(|&m| (m, ())),
                        );
                    }
                }
                if dead.len() == before {
                    break;
                }
            }
            // A trip map's capture slot every read of which is in a dropped arm.
            let slot_dead = |m: MorphismId, s: u32| -> bool {
                let Operation::Map { body: b, captures } = self.morphisms[m].op else {
                    return false;
                };
                let Some(fd) = self.func(b) else { return false };
                s < captures
                    && self.out_edges(fd.input).iter().all(|&p| {
                        self.morphisms[p].op != (Operation::Proj { index: s })
                            || dead.contains_key(p)
                            || self
                                .out_edges(self.morphisms[p].target)
                                .iter()
                                .all(|&q| dead.contains_key(q))
                    })
            };
            let reach = |seeds: &[ObjectId]| -> SecondaryMap<ObjectId, ()> {
                let mut seen: SecondaryMap<ObjectId, ()> = seeds.iter().map(|&o| (o, ())).collect();
                let mut work = seeds.to_vec();
                while let Some(o) = work.pop() {
                    for &m in self.out_edges(o) {
                        let mm = &self.morphisms[m];
                        if !body.contains_key(m) || dead.contains_key(m) {
                            continue;
                        }
                        if let Operation::Pair { slot, .. } = mm.op {
                            let readers = self.out_edges(mm.target);
                            if !readers.is_empty() && readers.iter().all(|&r| slot_dead(r, slot)) {
                                continue;
                            }
                        }
                        if seen.insert(mm.target, ()).is_none() {
                            work.push(mm.target);
                        }
                    }
                }
                seen
            };
            let nexts: Vec<Option<ObjectId>> = slots
                .iter()
                .map(|&k| self.pair_slot_source(next, k))
                .collect();
            let batched: Vec<(u32, MorphismId)> = slots
                .iter()
                .zip(&nexts)
                .filter_map(|(&k, &nk)| {
                    let m = *self.in_edges(nk?).first()?;
                    let Operation::Map { captures, .. } = self.morphisms[m].op else {
                        return None;
                    };
                    let Some(ElemSrc::Apply {
                        outside: Some(ou),
                        source,
                        ..
                    }) = self.map_law(m)
                    else {
                        return None;
                    };
                    let invariant = |o: ObjectId| vc.contains(&o) || !plan.scc_objects.contains(&o);
                    let rows = ou.a == 1
                        && ou.step == 1
                        && ou.len <= ou.d
                        && ou.e.is_some_and(|e| vc.contains(&e));
                    let own = self
                        .pair_slot_source(source, ou.slot)
                        .is_some_and(|x| views(k).contains(&x));
                    let rest = (0..=captures)
                        .filter(|&s| s != ou.slot && !slot_dead(m, s))
                        .all(|s| self.pair_slot_source(source, s).is_some_and(invariant));
                    (rows && own && rest).then_some((k, m))
                })
                .collect();
            let is_batched = |k: u32| batched.iter().any(|b| b.0 == k);
            let reached: Vec<SecondaryMap<ObjectId, ()>> =
                slots.iter().map(|&k| reach(&views(k))).collect();
            // Slot edges `i → j`: `j`'s next reads `i`'s view.
            let n_s = slots.len();
            let mut closure = vec![vec![false; n_s]; n_s];
            for (i, row) in closure.iter_mut().enumerate() {
                for (j, c) in row.iter_mut().enumerate() {
                    *c = !(i == j && is_batched(slots[j]))
                        && nexts[j].is_some_and(|nj| reached[i].contains_key(nj));
                }
            }
            for k in 0..n_s {
                for i in 0..n_s {
                    if closure[i][k] {
                        for j in 0..n_s {
                            closure[i][j] |= closure[k][j];
                        }
                    }
                }
            }
            let recurrent: Vec<u32> = (0..n_s)
                .filter(|&i| closure[i][i])
                .map(|i| slots[i])
                .collect();
            let rec_reach = reach(&recurrent.iter().flat_map(|&k| views(k)).collect::<Vec<_>>());
            let outside: Vec<MorphismId> = plan
                .decide_order
                .iter()
                .chain(&plan.advance_order)
                .copied()
                .filter(|&m| {
                    !dead.contains_key(m) && !rec_reach.contains_key(self.morphisms[m].target)
                })
                .collect();
            let unbatched: Vec<ObjectId> = slots
                .iter()
                .filter(|&&k| !is_batched(k))
                .flat_map(|&k| views(k))
                .collect();
            let from_unbatched = reach(&unbatched);
            // A skipped trip's work must be unobservable: trap-free, so the
            // interpreter's trap at an early trip is not dropped.
            let skip = n == 0
                && phi - plo >= 2
                && !effect
                && body
                    .keys()
                    .all(|m| dead.contains_key(m) || self.morph_trap_free(m, &bounds, None, 0))
                && !reach(&slots.iter().flat_map(|&k| views(k)).collect::<Vec<_>>())
                    .contains_key(cond)
                && slots.iter().zip(&nexts).all(|(&k, &nk)| {
                    is_batched(k)
                        || nk.is_some_and(|nk| {
                            views(k).contains(&nk) || !from_unbatched.contains_key(nk)
                        })
                });
            phases.push(TripPhase {
                lo: plo,
                hi: phi,
                recurrent,
                outside,
                batched,
                skip,
            });
        }
        Some(TripSplit {
            merge,
            counter,
            lo,
            hi,
            phases,
        })
    }
}
