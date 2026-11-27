//! per-cell placement: `cell_of`, batches, in-place range writes and cell lanes

use super::*;

impl<'a> FnEmit<'a> {
    /// The `devmap` gate on a [`CellSite`] (`cell_plan`: a classifiable body
    /// over array captures, number captures (`uniform`) and nested folds over
    /// captures (`nested`), every read classified): outside every guard
    /// arm, a body whose text names no runtime symbol (clean) and lowers to
    /// the host's bits on the device ([`Self::device_leaf`]), number elements, an
    /// input product with no erased component, and no more pointers than one
    /// launch passes. A closed-form producer is `devgen`'s, not a cell.
    pub(super) fn cell_of(&self, m: MorphismId, site: &CellSite) -> Option<Cell> {
        let morph = self.ir.morphism(m)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        // ADR-0043: a narrow element is stored like a number (no arithmetic
        // reaches it but a widening or a rounding).
        let number = |t: &Ty| matches!(t, Ty::Float { .. } | Ty::Int { .. } | Ty::Narrow(_));
        let Ty::Array { elem, .. } = self.obj_ty(morph.target) else {
            return None;
        };
        // `uniform`: an in-place range map over a uniform's cells (`map_law`
        // piecewise, `Outside`) is no whole-extent cell (it would rewrite
        // every cell per trip); under `range-cell` it is a cell over its
        // interval, written in its state's buffer (`range_in_place`).
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let range = if rungs.range_cell {
            self.range_in_place(m)
        } else {
            None
        };
        let piecewise = || {
            range.is_none()
                && matches!(
                    self.ir.map_law(m),
                    Some(ElemSrc::Apply {
                        outside: Some(_),
                        ..
                    })
                )
        };
        // `nest-cell`: a body folding over captured arrays, unless the map is
        // a tile or row-fold site (their own kernels).
        let nest = !site.nested.is_empty();
        let own = || {
            self.ir.tile_plan_strided(self.f).sites.contains_key(m)
                || self.ir.row_fold(self.f).contains_key(m)
        };
        // full-model `once-range`: a piecewise range map outside every loop
        // (a prompt's rows written into a cache, no carry to write in place)
        // runs once as a whole-extent device cell — every cell its arm's,
        // one pass over the array — rather than a host loop whose output
        // the device then reads whole.
        let once = || rungs.once_range && self.straight_line().contains_key(m);
        if self.gated.contains_key(m)
            || (nest && (!rungs.nest_cell || own()))
            || (!site.uniform.is_empty() && (!rungs.uniform || (piecewise() && !once())))
            || self.gen_law(m).is_some()
            || (site.traps.is_empty() && !self.attrs.clean(body))
            || !self.trap_leaf(body, !site.traps.is_empty())
            || !float_cell(&elem)
        {
            return None;
        }
        let caps = (0..captures)
            .map(|k| self.pair_source(morph.source, k))
            .collect::<Option<Vec<_>>>()?;
        let mapped = if captures == 0 {
            morph.source
        } else {
            self.pair_source(morph.source, captures)?
        };
        let input = self.obj_ty(self.ir.func(body)?.input);
        let whole = captures == 0 || residual_arity(&input) == captures as usize + 1;
        // Over an `iota` the element is the index; any other mapped array is read.
        let loaded = site.reads.iter().any(|r| r.slot == captures);
        if !whole || !number(array_elem(&self.obj_ty(mapped))) {
            return None;
        }
        // onto-device: the kernel forms an unloaded element from its thread
        // index as `i32`; an extent past i32 (`i64` elements) stays on the
        // host (open: an `i64` cell index).
        if !loaded && *array_elem(&self.obj_ty(mapped)) != Ty::i32() {
            return None;
        }
        let mut laws = if !loaded && rungs.cell_law {
            self.laws(body, site, &caps)
        } else {
            Vec::new()
        };
        // A range cell writes its interval over its source's buffer: that
        // array is the output, materialized — never evaluated in the cell
        // (`once-range`: a closed-form start a prompt writes over).
        laws.retain(|&(s, _)| range.is_none_or(|ou| ou.slot != s));
        let lanes = self.cell_lanes(site);
        let tag = match (range.is_some(), lanes) {
            (false, 1) => self.fnames[body].clone(),
            (r, l) => format!("{}|range={r}|lanes={l}", self.fnames[body]),
        };
        // structure: two sites of one body differ by the captures each
        // evaluates from its law (`cell-law`) — their kernels take different
        // parameters, so the name carries the law slots (two inlined copies
        // of one fn: one over an `iota` law, one over a loaded array).
        let tag = if laws.is_empty() {
            tag
        } else {
            let slots: Vec<String> = laws
                .iter()
                .map(|(s, g)| format!("{s}:{}", self.fnames[*g]))
                .collect();
            format!("{tag}|laws={}", slots.join(","))
        };
        // caching-memory `retain-cell`: a loop cell whose output changes in
        // a row delta only — not an in-place range cell (its buffer is the
        // state's), not a guarded one (an arm may skip a trip).
        let retain_on = range.is_none() && !self.gated.contains_key(m);
        // caching-memory `retain-same`: one whose output is the same on
        // every trip of its loop launches on the first trip only.
        let same = (retain_on && rungs.retain_same)
            .then(|| {
                let up = self.ir.unchanged_plan(self.f);
                let merge = *up.loop_of.get(morph.target)?;
                up.same_every_trip(morph.target).then_some(RetainCell {
                    merge,
                    delta: mapal_ir::Delta::Top,
                    id: 0,
                    lanes: 1,
                    same: true,
                })
            })
            .flatten();
        let retain = (same.is_none()
            && retain_on
            && rungs.retain_cell
            && (rungs.retain_delta || (rungs.resident && rungs.retain_resident)))
            .then(|| {
                let up = self.ir.unchanged_plan(self.f);
                let d = up.delta(morph.target)?;
                let merge = *up.loop_of.get(morph.target)?;
                let start = match d {
                    mapal_ir::Delta::Rows { start, .. } | mapal_ir::Delta::Pieces { start, .. } => {
                        start
                    }
                    mapal_ir::Delta::Top => return None,
                };
                let view = self.ir.morphism(*self.ir.in_edges(start.var).first()?)?;
                (view.source == merge || start.var == merge).then_some(())?;
                // The delta's cells per trip (at most): its range, or its pieces.
                let cells = match d {
                    mapal_ir::Delta::Pieces { w, step, .. } => site.extent.div_ceil(step) * w,
                    mapal_ir::Delta::Rows { stride, .. } => stride,
                    mapal_ir::Delta::Top => site.extent,
                }
                .min(site.extent);
                let lanes = self.cell_lanes(&CellSite {
                    extent: cells,
                    ..site.clone()
                });
                Some(RetainCell {
                    merge,
                    delta: d,
                    same: false,
                    id: gpu::cell_id(
                        &format!("{tag}|retain|lanes={lanes}"),
                        site.extent,
                        self.profile,
                    ),
                    lanes,
                })
            })
            .flatten()
            .or(same);
        let cell = Cell {
            id: gpu::cell_id(&tag, site.extent, self.profile),
            body,
            n: site.extent,
            mapped: loaded.then_some(mapped),
            caps,
            laws,
            uniform: site.uniform.clone(),
            nested: site.nested.clone(),
            range,
            dev: Vec::new(),
            lanes: if site.traps.is_empty() { lanes } else { 1 },
            retain: if site.traps.is_empty() { retain } else { None },
            trap: !site.traps.is_empty(),
        };
        // The output, then every operand and uniform (and an interval's start
        // and count): one 8-byte argument each.
        let extra = if range.is_some() { 2 } else { 0 };
        (cell.operands().len() + cell.uniform.len() + extra < gpu::LAUNCHV_MAX).then_some(cell)
    }

    /// `trip-batch` (from [`CategoryIr::trip_split`]): each loop whose first
    /// interval is skippable and every batched write of which is a range
    /// cell of whole rows (`len = d`) over its own carry, its one uniform the
    /// counter and every other operand a carry view of the loop (so at entry
    /// each is its carry's buffer) or a loop invariant made before the loop
    /// whose device copy is placed there (`at_entry`).
    pub(super) fn batches(
        &self,
        cells: &SecondaryMap<MorphismId, Cell>,
        carries: &[Carry],
        at_entry: &dyn Fn(ObjectId) -> bool,
    ) -> Vec<Batch> {
        let mut out = Vec::new();
        'loops: for ts in self.ir.trip_split(self.f) {
            let Some(p) = ts
                .phases
                .first()
                .filter(|p| p.skip && !p.batched.is_empty())
            else {
                continue;
            };
            let Some(lp) = self.ir.loop_plan(self.f, ts.merge) else {
                continue;
            };
            let in_body = |o: ObjectId| {
                self.ir
                    .in_edges(o)
                    .iter()
                    .any(|m| lp.decide_order.contains(m) || lp.advance_order.contains(m))
            };
            let carried = |o: ObjectId| {
                carries
                    .iter()
                    .any(|c| c.merge == ts.merge && c.views.contains(&o))
                    || (!lp.scc_objects.contains(&o) && !in_body(o) && at_entry(o))
            };
            let mut writes = Vec::new();
            for &(_, m) in &p.batched {
                let Some(c) = cells.get(m) else {
                    continue 'loops;
                };
                let Some(ou) = c.range else { continue 'loops };
                let Some(e) = ou.e else { continue 'loops };
                let [u] = c.uniform[..] else { continue 'loops };
                let start = (p.lo + ou.b).checked_mul(ou.d as i64);
                let count = (p.hi - 1 - p.lo).checked_mul(ou.d as i64);
                let ok = ou.len == ou.d
                    && c.caps[u as usize] == e
                    && c.mapped.is_none()
                    && c.laws.is_empty()
                    && c.lanes == 1
                    && (0..c.caps.len() as u32).all(|s| s == u || carried(c.caps[s as usize]))
                    && start
                        .zip(count)
                        .is_some_and(|(s, n)| s >= 0 && n > 0 && (s + n) as u64 <= c.n);
                if !ok {
                    continue 'loops;
                }
                let id = gpu::cell_id(&format!("{}|batch", self.fnames[c.body]), c.n, self.profile);
                let (s, n) = (start.expect("start") as u64, count.expect("count") as u64);
                writes.push((m, id, s, n, u, ou.d, ou.b));
            }
            out.push(Batch {
                merge: ts.merge,
                counter: ts.counter,
                enter: p.hi - 1,
                writes,
            });
        }
        out
    }

    /// `trip-batch` at the loop entry (after the carries hold their init):
    /// the batched writes, then the counter set to the interval's last trip.
    pub(crate) fn trip_batch(&mut self, merge: ObjectId) {
        let Some(b) = self
            .resident
            .as_ref()
            .and_then(|p| p.batch.iter().find(|b| b.merge == merge))
            .cloned()
        else {
            return;
        };
        for &(m, id, start, count, _, _, _) in &b.writes {
            let c = self.resident.as_ref().expect("plan").cells[m].clone();
            let target = self.ir.morphism(m).expect("cell").target;
            let mut ptrs = vec![self.dload(target)];
            for (s, &o) in c.caps.iter().enumerate() {
                if !c.uniform.contains(&(s as u32)) {
                    ptrs.push(self.dload(o));
                }
            }
            ptrs.push(start.to_string());
            ptrs.push(count.to_string());
            let args = format!("%dv{}", self.fresh());
            let np = ptrs.len();
            self.allocas
                .push_str(&format!("  {args} = alloca [{np} x i64]\n"));
            for (i, p) in ptrs.iter().enumerate() {
                let g = self.tmp();
                self.line(format!(
                    "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"
                ));
                self.line(format!("store i64 {p}, ptr {g}"));
            }
            let (gx, block) = gpu::cell_grid(count);
            self.line(format!(
                "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 0)",
                id as i64
            ));
        }
        let (llt, ptr) = self.component_ptr(merge, b.counter).expect("counter slot");
        self.line(format!(
            "; mapal.batch enter={} writes={}",
            b.enter,
            b.writes.len()
        ));
        self.line(format!("store {llt} {}, ptr {ptr}", b.enter));
    }

    /// `range-cell`'s legality: `m` is a piecewise map ([`mapal_ir::Outside`])
    /// over a loop state's view whose output is that state's next value, and
    /// no view of the state is read after `m` — the trip writes its state in
    /// place, so its cells off the interval are already the state's.
    /// placement `chain`: or a link of a chain of such maps from the view to
    /// the next value, each writing over the one before, each written array
    /// dead past its writer (a trip's appends to one cache). As
    /// `state_death` (frame.rs) has it, a whole-state use or a guard-arm read
    /// (run at its `Phi`, past its own position) reads any time.
    ///
    /// ponytail: not `in_place_source` itself — this runs in the residency
    /// plan, before `prepare_storage` indexes the roots `state_death` walks.
    pub(super) fn range_in_place(&self, m: MorphismId) -> Option<mapal_ir::Outside> {
        let (ou, arr) = self.range_write(m)?;
        // full-model `once-range`: a range map run once (outside every loop
        // and guard arm) writes its interval into its source's own buffer
        // when nothing reads the source after it (`last_use_plan`); else it
        // is no range cell (a whole-extent cell, [`Self::cell_of`]).
        if self.straight_line().contains_key(m) {
            let once = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).once_range;
            return (once && self.writes_over_dead(m, arr)).then_some(ou);
        }
        let target = self.ir.morphism(m)?.target;
        let chain = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).range_chain;
        // placement `chain`: the written array is a state view, or the output
        // of a range map written over one (each dead past its writer).
        let mut w = arr;
        let mut steps = 0;
        let (view, k, merge) = loop {
            steps += 1;
            if steps > 256 {
                return None;
            }
            let p = self.ir.morphism(*self.ir.in_edges(w).first()?)?;
            if let Operation::Proj { index } = p.op
                && self.ir.object(p.source)?.kind == ObjectKind::LoopMerge
            {
                break (w, index, p.source);
            }
            if !chain {
                return None;
            }
            let (_, before) = self.range_write(*self.ir.in_edges(w).first()?)?;
            w = before;
        };
        let plan = self.ir.loop_plan(self.f, merge)?;
        let state = self.pair_source(plan.back_route, 0)?;
        let next = self.pair_source(state, k)?;
        let views: Vec<ObjectId> = self
            .ir
            .out_edges(merge)
            .iter()
            .filter_map(|&e| self.ir.morphism(e))
            .filter_map(|x| match x.op {
                Operation::Proj { index } if index == k => Some(Some(x.target)),
                Operation::Proj { .. } => None,
                _ => Some(None), // a whole-state use
            })
            .collect::<Option<_>>()?;
        // ... and the target is the next state, or written over by a range
        // map whose output is (each dead past its writer).
        let mut t = target;
        for _ in 0..256 {
            if t == next {
                let pos = self.lup.position(m)?;
                let dead = if arr == view {
                    self.lup.view_dead_after(arr, pos)
                } else {
                    self.lup.dead_after(arr, pos)
                };
                return (!views.iter().any(|&v| self.guard_read(v, Some(m))) && dead).then_some(ou);
            }
            if !chain {
                return None;
            }
            let after = self.ir.out_edges(t).iter().find_map(|&e| {
                let pm = self.ir.morphism(e)?;
                if !matches!(pm.op, Operation::Pair { .. }) {
                    return None;
                }
                self.ir.out_edges(pm.target).iter().find_map(|&u| {
                    let (_, over) = self.range_write(u)?;
                    (over == t).then_some(u)
                })
            })?;
            t = self.ir.morphism(after)?.target;
        }
        None
    }

    /// A piecewise in-place range map (`map_law`'s `Outside`): its interval
    /// and the array its cells off the interval are (the one it writes over).
    pub(super) fn range_write(&self, m: MorphismId) -> Option<(mapal_ir::Outside, ObjectId)> {
        let Some(ElemSrc::Apply {
            outside: Some(ou),
            source,
            ..
        }) = self.ir.map_law(m)
        else {
            return None;
        };
        Some((ou, self.pair_source(source, ou.slot)?))
    }

    /// full-model `once-range`: `arr` — the array the straight-line range
    /// map `m` writes over — owns its bytes (made here, once, by a morphism
    /// that is no view: not a product's slot, a projection, an element view,
    /// a `Phi`, a call's result or a loop's state) and no morphism reads it
    /// or a view of it after `m` (`last_use_plan.view_dead_after`): `m`'s
    /// output may be those bytes.
    pub(super) fn writes_over_dead(&self, m: MorphismId, arr: ObjectId) -> bool {
        let Some(pos) = self.lup.position(m) else {
            return false;
        };
        let Some(&p) = self.ir.in_edges(arr).first() else {
            return false;
        };
        let owns = self.ir.morphism(p).is_some_and(|x| {
            !matches!(
                x.op,
                Operation::Pair { .. }
                    | Operation::Proj { .. }
                    | Operation::Index
                    | Operation::Phi
                    | Operation::Call(_)
                    | Operation::LoopEnter
                    | Operation::LoopExit
            )
        });
        owns && self.ir.in_edges(arr).len() == 1
            && self
                .ir
                .object(arr)
                .is_some_and(|o| o.kind == ObjectKind::Temporary)
            && self.ir.try_owner(arr) == Some(self.f)
            && self.straight_line().contains_key(p)
            && self.lup.view_dead_after(arr, pos)
    }

    /// `cell-lanes`: threads a nest cell takes — a warp, when the order is
    /// free (the default face, ADR-0040), the body folds once (one nested
    /// fold, so one tree), and its cells leave the part's resident threads
    /// idle. 1 otherwise.
    pub(super) fn cell_lanes(&self, site: &CellSite) -> u64 {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        match site.tree {
            Some((fold, _, k)) if rungs.cell_lanes => {
                let fm = self.ir.morphism(fold).expect("fold");
                let elem = self.obj_ty(fm.target);
                // placement `lanes`: a fold that reads an array along its
                // own axis at unit stride (`step_algebra`'s regroup reads: a
                // row of `w` for a 1-row projection) is read coalesced by a
                // warp a cell and `k` elements apart by a thread a cell —
                // there a warp a cell whether or not the cells fill the part.
                let unit = rungs.stride_lanes
                    && match fm.op {
                        Operation::Fold { body, .. } => self
                            .ir
                            .step_algebra(body)
                            .regroup
                            .is_some_and(|r| r.reads.contains(&Some(1))),
                        _ => false,
                    };
                let units = if unit { 1 } else { site.extent };
                gpu::warp_lanes(self.profile, self.order, &elem, k, units)
            }
            _ => 1,
        }
    }

    /// `cell-law`'s fact: the captures of a cell over an `iota` that are
    /// closed-form (`devgen`'s law, `x[i] = f(i)`) and read only at the cell's
    /// own index (every read `Same`, and the capture used for nothing but
    /// those reads). Such a read is `f(i)`, evaluated in the cell: nothing
    /// stores `x`, nothing loads it.
    pub(super) fn laws(
        &self,
        body: FuncId,
        site: &CellSite,
        caps: &[ObjectId],
    ) -> Vec<(u32, FuncId)> {
        let Some(fd) = self.ir.func(body) else {
            return Vec::new();
        };
        let only_read = |s: u32| {
            self.ir.out_edges(fd.input).iter().all(|&p| {
                let x = self.ir.morphism(p).expect("morphism resolves");
                x.op != Operation::Proj { index: s }
                    || self.ir.out_edges(x.target).iter().all(|&q| {
                        let y = self.ir.morphism(q).expect("morphism resolves");
                        matches!(y.op, Operation::Pair { slot: 0, .. })
                            && self.ir.out_edges(y.target).iter().all(|&r| {
                                self.ir.morphism(r).expect("morphism resolves").op
                                    == Operation::Index
                            })
                    })
            })
        };
        caps.iter()
            .enumerate()
            .filter_map(|(s, &c)| {
                let s = s as u32;
                let mut reads = site.reads.iter().filter(|r| r.slot == s).peekable();
                let same = reads.peek().is_some() && reads.all(|r| r.foot == Footprint::Same);
                let law = self.gen_law(*self.ir.in_edges(c).first()?)?;
                (same && only_read(s)).then_some((s, law))
            })
            .collect()
    }
}
