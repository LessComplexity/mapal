//! the resident plan's use classification: slot uses, carries, device leaves, cell readers and regions

use super::*;

impl<'a> FnEmit<'a> {
    /// The device sites of this function, by morphism.
    pub(super) fn device_sites(&self) -> SecondaryMap<MorphismId, DevSite> {
        let mut out = SecondaryMap::new();
        if let Some(plan) = &self.tile_plan {
            for (m, site) in plan.sites.iter() {
                if self.device.contains_key(m) {
                    out.insert(m, DevSite::Tile(site.clone()));
                }
            }
            let folds = super::super::rowfold::rowfold_sites(
                self.ir,
                self.f,
                self.profile,
                self.contract,
                self.gpu_arms,
            );
            for (m, r) in folds {
                out.insert(m, DevSite::RowFold(r));
            }
        }
        out
    }

    /// How consumer `m2` reads slot `slot` of the pointer-only product it
    /// consumes.
    pub(super) fn slot_use(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
        m2: MorphismId,
        slot: u32,
    ) -> Use {
        if let Some(site) = sites.get(m2) {
            return if site.reads().contains(&slot) {
                Use::Dev
            } else {
                Use::None
            };
        }
        // An epilogue member (S54): read in a site's store, or not at all.
        if let Some(src) = self.member_src(m2, slot) {
            return if src == EpiSrc::Index {
                Use::None
            } else {
                Use::Dev
            };
        }
        // A cell kernel loads every capture it does not evaluate, and the
        // mapped array unless its element is the index.
        if let Some(c) = cells.get(m2) {
            let cap = (slot as usize) < c.caps.len();
            return if (cap && !c.law(slot)) || (!cap && c.mapped.is_some()) {
                Use::Dev
            } else {
                Use::None
            };
        }
        let op = self.ir.morphism(m2).expect("morphism resolves").op;
        if op == Operation::Index && slot == 0 && self.bp.proven(m2) {
            return Use::Elem(m2);
        }
        Use::Host
    }

    /// Every use of array `o`, through the pointer-only products that carry it.
    /// A by-value `Pair` is `Fwd` when the carry owns it (`fwd`), else `Host`.
    ///
    /// A value made here that outlives the call (`last_use_plan`'s `escapes`:
    /// it reaches the function's `Return` / `Output`) is read by the caller,
    /// on the host — a function's result is a host value — so it is a `Host`
    /// use even with no reader in the function: a loop exit or a site output
    /// returned by a callee is copied out, never kept or built device-only.
    /// (A `Parameter` escapes as a borrow, not a read: it is not counted.)
    pub(super) fn uses(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
        o: ObjectId,
        fwd: &[MorphismId],
    ) -> Vec<Use> {
        let mut out = Vec::new();
        if self.lup.escapes(o)
            && self.ir.try_owner(o) == Some(self.f)
            && self
                .ir
                .object(o)
                .is_some_and(|x| x.kind != ObjectKind::Parameter)
        {
            out.push(Use::Host);
        }
        out.extend(self.edge_uses(sites, cells, o, fwd));
        out
    }

    /// [`Self::uses`] by `o`'s out-edges alone (its escape not counted).
    pub(super) fn edge_uses(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
        o: ObjectId,
        fwd: &[MorphismId],
    ) -> Vec<Use> {
        let mut out = Vec::new();
        for &m in self.ir.out_edges(o) {
            let morph = self.ir.morphism(m).expect("morphism resolves");
            match morph.op {
                Operation::Pair { .. } if fwd.contains(&m) => out.push(Use::Fwd),
                Operation::Pair { slot, .. }
                    if self.pointer_only_array_component(morph.target, slot) =>
                {
                    for &m2 in self.ir.out_edges(morph.target) {
                        out.push(self.slot_use(sites, cells, m2, slot));
                    }
                }
                // `scalar-fold`: a number every reader of which is a cell
                // loading it from its device buffer.
                Operation::Pair { slot, .. }
                    if !self.ir.out_edges(morph.target).is_empty()
                        && self
                            .ir
                            .out_edges(morph.target)
                            .iter()
                            .all(|&m2| cells.get(m2).is_some_and(|c| c.dev.contains(&slot))) =>
                {
                    out.push(Use::Dev)
                }
                Operation::Map { captures: 0, .. } if self.member_src(m, 0).is_some() => {
                    out.push(Use::Dev)
                }
                // A capture-free cell maps `o` itself.
                _ if cells.contains_key(m) => out.push(self.slot_use(sites, cells, m, 0)),
                _ => out.push(Use::Host),
            }
        }
        out
    }

    /// The morphisms the straight-line walk emits itself — run exactly once
    /// per call (outside every loop cone and guard arm).
    pub(in crate::func) fn straight_line(&self) -> SecondaryMap<MorphismId, ()> {
        let mut in_scc: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for scc in self.ir.loop_structure(self.f) {
            for &mg in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, mg) {
                    for &mo in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                        owned.insert(mo, ());
                    }
                }
            }
            for o in scc.objects {
                in_scc.insert(o, ());
            }
        }
        let mut out = SecondaryMap::new();
        for m in self.ir.topo_order(self.f) {
            let morph = self.ir.morphism(m).expect("morphism resolves");
            let loop_op = matches!(
                morph.op,
                Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
            );
            if !loop_op
                && !owned.contains_key(m)
                && !in_scc.contains_key(morph.source)
                && !in_scc.contains_key(morph.target)
                && !self.gated.contains_key(m)
            {
                out.insert(m, ());
            }
        }
        out
    }

    /// The loop-carry candidates (F3), each checked against its uses.
    pub(super) fn carries(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
    ) -> Vec<Carry> {
        let site_out = |o: ObjectId| self.produced_on_device(sites, cells, o);
        let mut out: Vec<Carry> = Vec::new();
        // `range-cell`: a range cell fed by another state's carry (decode's
        // `x` row reads `o`) rides only once that carry is found — repeat
        // until no carry is added (each round only adds, so it ends).
        loop {
            let found = out.len();
            for scc in self.ir.loop_structure(self.f) {
                for &merge in &scc.merges {
                    let Some(plan) = self.ir.loop_plan(self.f, merge) else {
                        continue;
                    };
                    let Ty::Tuple(parts) = self.obj_ty(merge) else {
                        // ponytail: a bare-array state (no counter) has no
                        // program yet; it is the same rule without the Proj.
                        continue;
                    };
                    let Some(state) = self.pair_source(plan.back_route, 0) else {
                        continue;
                    };
                    let back_pair: Vec<MorphismId> = self.ir.out_edges(state).to_vec();
                    for (k, ty) in parts.iter().enumerate() {
                        let k = k as u32;
                        if !matches!(ty, Ty::Array { .. })
                            || out.iter().any(|c| c.merge == merge && c.k == k)
                        {
                            continue;
                        }
                        let Some(next) = self.pair_source(state, k) else {
                            continue;
                        };
                        if !site_out(next) || self.pair_source(plan.init, k).is_none() {
                            continue;
                        }
                        // The back edge swaps the carry's buffer with
                        // `next`'s: right only when every trip writes `next`
                        // again. A next state made outside the trip (one that
                        // reads neither the state nor the counter, hoisted)
                        // is written once — the second swap would hand the
                        // state its first buffer back. Such a state is the
                        // one object after any trip: its pointer forwards.
                        let made = self.ir.in_edges(next).iter().any(|m| {
                            plan.decide_order.contains(m) || plan.advance_order.contains(m)
                        });
                        if !made {
                            continue;
                        }
                        let views: Vec<ObjectId> = self
                            .ir
                            .out_edges(merge)
                            .iter()
                            .map(|&m| self.ir.morphism(m).expect("morphism resolves"))
                            .filter(|mo| mo.op == Operation::Proj { index: k })
                            .map(|mo| mo.target)
                            .collect();
                        // The exit payload, when it is a view: its Pair is the
                        // carry's, and the exit object shares the slot.
                        let exit_pair =
                            self.ir
                                .in_edges(plan.exit_route)
                                .iter()
                                .copied()
                                .find(|&m| {
                                    let mo = self.ir.morphism(m).expect("morphism resolves");
                                    matches!(mo.op, Operation::Pair { slot: 0, .. })
                                        && views.contains(&mo.source)
                                });
                        let exit = exit_pair.and_then(|_| {
                            plan.exits
                                .first()
                                .map(|&x| self.ir.morphism(x).expect("exit").target)
                        });
                        // `state` must feed only the back route, and `next` must
                        // reach it only through `state`'s slot `k`.
                        let next_pair: Vec<MorphismId> = self
                            .ir
                            .out_edges(next)
                            .iter()
                            .copied()
                            .filter(|&m| self.ir.morphism(m).expect("m").target == state)
                            .collect();
                        if back_pair.len() != 1 || next_pair.len() != 1 {
                            continue;
                        }
                        let fwd: Vec<MorphismId> =
                            next_pair.iter().chain(exit_pair.iter()).copied().collect();
                        let device_only = |o: ObjectId| {
                            self.uses(sites, cells, o, &fwd)
                                .iter()
                                .all(|u| !matches!(u, Use::Host))
                        };
                        let read = views
                            .iter()
                            .any(|&v| self.uses(sites, cells, v, &fwd).contains(&Use::Dev));
                        // S65 `by-ref`: a state no morphism of the trip reads
                        // (only forwarded to the exit) stays where it is produced
                        // — the loop's products forward its pointer.
                        let unread = self.is_view(merge, &[k])
                            && views.iter().all(|&v| {
                                self.uses(sites, cells, v, &fwd)
                                    .iter()
                                    .all(|u| *u == Use::Fwd)
                            });
                        if (read || unread)
                            && !views.is_empty()
                            && device_only(next)
                            && views.iter().all(|&v| device_only(v))
                        {
                            // `range-cell`: `next` is a range cell over this
                            // state's own view (so its buffer holds the cells off
                            // the interval).
                            // placement `chain`: or the end of a chain of range
                            // cells, each written over the one before, the first
                            // over the view.
                            let (range, chain) = match self.range_chain(next, &views, cells) {
                                Some((cs, chain)) => (Some(cs), chain),
                                None => (None, Vec::new()),
                            };
                            // The state rides the device for the range cell alone
                            // only when something else holds it there: a device
                            // site reading the state, or an operand of the cell
                            // made on the device (else the host keeps it).
                            if let Some(cs) = &range {
                                // A site reading the state (a view, or the written
                                // state, which shares the buffer) — a cell reader
                                // is no support: it is on the device only because
                                // the state is (an island the host would keep).
                                let read_by_site = |o: ObjectId| {
                                    sites.iter().any(|(m, site)| {
                                        let src = self.ir.morphism(m).expect("site").source;
                                        site.reads()
                                            .iter()
                                            .any(|&k| self.pair_source(src, k) == Some(o))
                                    })
                                };
                                let other = views.iter().any(|&v| read_by_site(v))
                                    || read_by_site(next)
                                    || chain.iter().any(|&o| read_by_site(o));
                                let fed = cs.iter().flat_map(|c| c.operands()).any(|(o, _)| {
                                    !views.contains(&o)
                                        && !chain.contains(&o)
                                        && (site_out(o) || out.iter().any(|x| x.views.contains(&o)))
                                });
                                if !other && !fed {
                                    continue;
                                }
                            }
                            let inplace = range.is_some();
                            out.push(Carry {
                                merge,
                                k,
                                next,
                                views,
                                exit,
                                inplace,
                                chain,
                            });
                        }
                    }
                }
            }
            if out.len() == found {
                break;
            }
        }
        out
    }

    /// placement `chain`: `next` as the end of a chain of range cells over
    /// one of `views` — each cell's written array the output of the one
    /// before, the first's a view — with the chain's intermediate outputs.
    /// A one-cell chain is the S65 `range-cell` carry.
    pub(super) fn range_chain<'c>(
        &self,
        next: ObjectId,
        views: &[ObjectId],
        cells: &'c SecondaryMap<MorphismId, Cell>,
    ) -> Option<(Vec<&'c Cell>, Vec<ObjectId>)> {
        let (mut o, mut cs, mut chain) = (next, Vec::new(), Vec::new());
        for _ in 0..256 {
            let c = cells.get(*self.ir.in_edges(o).first()?)?;
            let w = c.caps[c.range?.slot as usize];
            cs.push(c);
            if views.contains(&w) {
                return Some((cs, chain));
            }
            chain.push(w);
            o = w;
        }
        None
    }

    /// The `devgen` law gate: `out[i] = body(i)` with a clean body that calls
    /// nothing and takes the `i32` index — exactly what a one-thread-per-element
    /// kernel can evaluate with the host's own body text.
    pub(in crate::func) fn gen_law(&self, m: MorphismId) -> Option<FuncId> {
        let morph = self.ir.morphism(m)?;
        let Operation::Map { body, captures: 0 } = morph.op else {
            return None;
        };
        let Some(ElemSrc::Apply {
            body: lb,
            captures: 0,
            inner,
            ..
        }) = self.elem.src(morph.target)
        else {
            return None;
        };
        let fd = self.ir.func(body)?;
        (*lb == body
            && **inner == ElemSrc::Index
            && self.obj_ty(fd.input) == Ty::i32()
            && float_cell(array_elem(&self.obj_ty(morph.target)))
            && self.device_leaf(body))
        .then_some(body)
    }

    /// A body the device runs with the host's own text: clean (no trap call,
    /// no effect), a leaf (no `Call`/`Map`/`Fold`), and every op lowering to
    /// the same bits on the device as on the host. A float `%` does not: the
    /// host calls `fmodf`, NVPTX expands `frem` to `fma(-y, trunc(x / y), x)`
    /// (S53 review: 2051 of 4096 elements differ).
    ///
    /// A nested `Map`/`Fold` is one more body of the same kind (generic-fusion
    /// natural forms): the device unit carries its text too
    /// ([`nested_bodies`]). Depth-guarded (J1).
    pub(in crate::func) fn device_leaf(&self, body: FuncId) -> bool {
        self.device_leaf_at(body, 0)
    }

    /// [`Self::device_leaf`], or (bf16-class `device-trap`, `traps`) the
    /// same leaf whose own unproven index reads are its only unclean ops
    /// (`trap_cell_plan` proved that much; each nested body a clean leaf).
    pub(super) fn trap_leaf(&self, body: FuncId, traps: bool) -> bool {
        if !traps {
            return self.device_leaf(body);
        }
        let Some(fd) = self.ir.func(body) else {
            return false;
        };
        fd.morphisms.iter().all(|&g| {
            let Some(x) = self.ir.morphism(g) else {
                return false;
            };
            let float_mod = x.op == Operation::Mod
                && self
                    .ir
                    .object(x.source)
                    .and_then(|o| o.ty.component_ty(0).cloned())
                    .is_some_and(|t| is_float(&t));
            !float_mod
                && match x.op {
                    Operation::Call(_) => false,
                    Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                        self.device_leaf_at(body, 1)
                    }
                    _ => true,
                }
        })
    }

    pub(super) fn device_leaf_at(&self, body: FuncId, depth: u32) -> bool {
        let Some(fd) = self.ir.func(body) else {
            return false;
        };
        let leaf = fd.morphisms.iter().all(|&g| {
            let Some(x) = self.ir.morphism(g) else {
                return false;
            };
            let float_mod = x.op == Operation::Mod
                && self
                    .ir
                    .object(x.source)
                    .and_then(|o| o.ty.component_ty(0).cloned())
                    .is_some_and(|t| is_float(&t));
            !float_mod
                && match x.op {
                    Operation::Call(_) => false,
                    Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                        depth < 8 && self.device_leaf_at(body, depth + 1)
                    }
                    _ => true,
                }
        });
        self.attrs.clean(body) && leaf
    }

    /// The source of slot `slot` of epilogue member `m`, `None` when `m` is
    /// in no group.
    pub(in crate::func) fn member_src(&self, m: MorphismId, slot: u32) -> Option<EpiSrc> {
        self.groups
            .values()
            .flat_map(|g| &g.chain)
            .find(|mb| mb.m == m)
            .and_then(|mb| mb.srcs.get(slot as usize).copied())
    }

    /// Emit this map body as its epilogue / unit device variant (S54): the
    /// captured arrays in `slots` arrive as one-cell buffers holding cell `t`,
    /// under its slots' own name ([`gpu::slots_variant`]).
    pub(crate) fn set_cell_slots(&mut self, slots: Vec<u32>) {
        self.cell_suffix = gpu::slots_variant("", &slots);
        self.cell_slots = slots;
    }

    /// `tile-post`: emit this map body as a tile site's epilogue variant —
    /// fold `fold`'s value arrives as `%fv` (`<name>_post`).
    pub(crate) fn set_post_fold(&mut self, fold: Option<MorphismId>) {
        self.post_fold = fold;
    }

    /// Does the `Index` over product `source` read a one-cell buffer?
    pub(in crate::func) fn cell_read(&self, source: ObjectId) -> bool {
        if self.cell_slots.is_empty() {
            return false;
        }
        let input = self.ir.func(self.f).expect("func resolves").input;
        let Some(arr) = self.pair_source(source, 0) else {
            return false;
        };
        matches!(self.ir.in_edges(arr), [m] if {
            let mo = self.ir.morphism(*m).expect("morphism resolves");
            mo.source == input
                && matches!(mo.op, Operation::Proj { index } if self.cell_slots.contains(&index))
        })
    }

    /// The `Map`s reading `v`: directly (capture-free) or through a product.
    pub(in crate::func) fn cell_readers(&self, v: ObjectId) -> Vec<MorphismId> {
        let is_map = |m: MorphismId| {
            matches!(
                self.ir.morphism(m).expect("morphism resolves").op,
                Operation::Map { .. }
            )
        };
        let mut out = Vec::new();
        for &m in self.ir.out_edges(v) {
            let mo = self.ir.morphism(m).expect("morphism resolves");
            match mo.op {
                Operation::Map { captures: 0, .. } => out.push(m),
                Operation::Pair { .. } => out.extend(
                    self.ir
                        .out_edges(mo.target)
                        .iter()
                        .copied()
                        .filter(|&m2| is_map(m2)),
                ),
                _ => {}
            }
        }
        out
    }

    /// Does anything but the morphisms `ms` read `o` (through a product or
    /// directly)? A value made here that outlives the call (`last_use_plan`'s
    /// `escapes`: paired into the function's result) is read by the caller,
    /// as [`Self::uses`] counts it — a product with no reader in the function
    /// is the return tuple, not an unread value.
    pub(in crate::func) fn read_outside(&self, o: ObjectId, ms: &[MorphismId]) -> bool {
        let returned = self.lup.escapes(o)
            && self.ir.try_owner(o) == Some(self.f)
            && self
                .ir
                .object(o)
                .is_some_and(|x| x.kind != ObjectKind::Parameter);
        returned
            || self.ir.out_edges(o).iter().any(|&m| {
                let mo = self.ir.morphism(m).expect("morphism resolves");
                match mo.op {
                    Operation::Pair { .. } => self
                        .ir
                        .out_edges(mo.target)
                        .iter()
                        .any(|m2| !ms.contains(m2)),
                    _ => !ms.contains(&m),
                }
            })
    }

    /// Where each morphism runs: straight-line (`(None, _)`), or one phase of
    /// one loop (`(Some(merge), advance)`). A morphism in neither (a guard
    /// arm, loop machinery) is absent.
    /// review-fixes-2: **how many times a launch in region `merge` runs a
    /// call** — once outside every loop (`None`); a counted loop's trips
    /// (`trip_split`: its constant `[lo, hi)`); 0 for a loop no fact counts.
    /// What shares the process's first cooperative launch (`coop_first_ns`:
    /// each launch its `1 / n`, a loop of unknown trips no term).
    pub(in crate::func) fn launches_in(&self, merge: Option<ObjectId>) -> u64 {
        let Some(mg) = merge else {
            return 1;
        };
        self.ir
            .trip_split(self.f)
            .iter()
            .find(|t| t.merge == mg)
            .map_or(0, |t| (t.hi - t.lo).max(1) as u64)
    }

    pub(in crate::func) fn regions(&self) -> SecondaryMap<MorphismId, (Option<ObjectId>, bool)> {
        let mut region = SecondaryMap::new();
        for m in self.straight_line().keys() {
            region.insert(m, (None, false));
        }
        for scc in self.ir.loop_structure(self.f) {
            for &mg in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, mg) {
                    for &m in &plan.decide_order {
                        region.insert(m, (Some(mg), false));
                    }
                    for &m in &plan.advance_order {
                        region.insert(m, (Some(mg), true));
                    }
                }
            }
        }
        region
    }
}
