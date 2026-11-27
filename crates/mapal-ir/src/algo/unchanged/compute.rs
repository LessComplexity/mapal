//! The unchanged-plan computation: `unchanged_plan`, `call_keep` and the row-read images they walk.

use super::*;

impl CategoryIr {
    /// The unchanged-since-last-time query (see the module doc): per loop
    /// body array, the cells that may differ from the previous trip; per
    /// value, the input slots it depends on. Stored on the graph.
    pub fn unchanged_plan(&self, f: FuncId) -> UnchangedPlan {
        self.fact(f, |s| &s.unchanged_plan, || self.unchanged_plan_compute(f))
    }

    /// **`call_keep`** ([`CallKeep`]): every `Call` site of `f` with the
    /// callee slots its loops keep and the callee values kept with them.
    /// Stored on the graph; built from `loop_structure`, `unchanged_plan(f)`
    /// (`same`) and `unchanged_plan(callee)` (`inputs`), so nested sites
    /// compose: a callee's own call site carries its arguments' masks.
    pub fn call_keep(&self, f: FuncId) -> SecondaryMap<MorphismId, CallKeep> {
        self.fact(f, |s| &s.call_keep, || self.call_keep_compute(f))
    }

    pub(super) fn call_keep_compute(&self, f: FuncId) -> SecondaryMap<MorphismId, CallKeep> {
        let mut out = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return out;
        };
        let loops = self.loop_structure(f);
        let up = self.unchanged_plan(f);
        for &m in &fd.morphisms {
            let Some(mm) = self.morphism(m) else { continue };
            let Operation::Call(g) = mm.op else { continue };
            let inl: Vec<&super::super::LoopScc> = loops
                .iter()
                .filter(|l| l.objects.contains(&mm.target) || l.objects.contains(&mm.source))
                .collect();
            let gin = self.func(g).map(|gd| gd.input);
            let arity = match gin.and_then(|i| self.object(i)).map(|o| &o.ty) {
                Some(Ty::Tuple(ts)) => Some(ts.len() as u32),
                Some(_) => None,
                None => continue,
            };
            let same_at = |a: ObjectId| {
                up.same.contains_key(a) || inl.iter().all(|l| !l.objects.contains(&a))
            };
            let mut kept = 0u64;
            if !inl.is_empty() {
                match arity {
                    None => {
                        if same_at(mm.source) {
                            kept = 1;
                        }
                    }
                    Some(n) => {
                        for k in 0..n.min(63) {
                            let args = self.component_sources(mm.source, k).unwrap_or_default();
                            if !args.is_empty() && args.iter().all(|&a| same_at(a)) {
                                kept |= 1 << k;
                            }
                        }
                    }
                }
            }
            let gup = self.unchanged_plan(g);
            let mut values: Vec<ObjectId> = gup
                .inputs
                .iter()
                .filter(|&(_, &mask)| kept != 0 && mask != 0 && mask & !kept == 0)
                .map(|(o, _)| o)
                .collect();
            values.sort();
            out.insert(
                m,
                CallKeep {
                    callee: Some(g),
                    kept,
                    values,
                    loops: inl.iter().flat_map(|l| l.merges.iter().copied()).collect(),
                },
            );
        }
        out
    }

    /// The reads of map `m`'s cell over rows of `q` cells (see [`Lin`]):
    /// `None` when the body is opaque (a call) or `m` is no map.
    pub(crate) fn row_reads(&self, m: MorphismId, q: u64) -> Option<Vec<RowRead>> {
        let map = self.morphism(m)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        let Ty::Array { size: n, .. } = self.object(map.target)?.ty else {
            return None;
        };
        if q == 0 || n == 0 {
            return None;
        }
        let bd = self.func(body)?;
        let mapped = if captures == 0 {
            map.source
        } else {
            self.pair_slot_source(map.source, captures)?
        };
        let t = Lin {
            c: i128::from(q),
            lo: 0,
            hi: i128::from(q) - 1,
        };
        let mut w = Walk {
            ir: self,
            rmax: i128::from((n - 1) / q),
            memo: HashMap::new(),
            reads: Vec::new(),
            opaque: false,
        };
        // The element: the index itself over an iota of the map's extent,
        // else the mapped array's cell `t`.
        let elem = if self.tile_iota_size(mapped) == Some(n) {
            Some(t)
        } else {
            w.reads.push(RowRead {
                slot: captures,
                at: Some(t),
            });
            None
        };
        let mut slots: Vec<Option<Lin>> = Vec::new();
        let mut arrays: Vec<Option<u32>> = Vec::new();
        for k in 0..captures {
            let cap = self.pair_slot_source(map.source, k);
            let is_arr = cap
                .and_then(|o| self.object(o))
                .is_some_and(|o| matches!(o.ty, Ty::Array { .. }));
            slots.push(None);
            arrays.push(is_arr.then_some(k));
        }
        let sc = if captures == 0 {
            Scope {
                input: bd.input,
                slots: Vec::new(),
                whole: elem,
                arrays: Vec::new(),
                whole_array: None,
            }
        } else {
            slots.push(elem);
            arrays.push(None);
            Scope {
                input: bd.input,
                slots,
                whole: None,
                arrays,
                whole_array: None,
            }
        };
        w.walk(body, &sc, 0);
        (!w.opaque).then_some(w.reads)
    }

    /// The candidate row widths of map `m`: 1 and every literal divisor or
    /// modulus of its body, ascending.
    pub(super) fn row_widths(&self, m: MorphismId) -> Vec<u64> {
        let mut qs = vec![1u64];
        if let Some(Operation::Map { body, .. }) = self.morphism(m).map(|x| x.op)
            && let Some(bd) = self.func(body)
        {
            for &g in &bd.morphisms {
                let Some(x) = self.morphism(g) else { continue };
                if matches!(x.op, Operation::Div | Operation::Mod)
                    && let Some(d) = self
                        .pair_slot_source(x.source, 1)
                        .and_then(|o| const_int(self, self.through_pairs(o)))
                        .and_then(|d| u64::try_from(d).ok())
                    && d > 1
                {
                    qs.push(d);
                }
            }
        }
        qs.sort_unstable();
        qs.dedup();
        qs
    }

    /// `(a·var + b − sub)·num / den` as an affine of `var` — exact, and only
    /// where neither form leaves `var`'s type over the counter's range.
    pub(super) fn scale_affine(
        &self,
        f: FuncId,
        a: Affine,
        sub: i128,
        num: i128,
        den: i128,
    ) -> Option<Affine> {
        let (na, nb) = (
            i128::from(a.a).checked_mul(num)?,
            i128::from(a.b).checked_sub(sub)?.checked_mul(num)?,
        );
        if den <= 0 || na % den != 0 || nb % den != 0 {
            return None;
        }
        let (ra, rb) = (na / den, nb / den);
        let ty = int_range(&self.object(a.var)?.ty)?;
        let (tlo, thi) = self.bounds_proof(f).view_range(a.var)?;
        let ok = |x: i128, y: i128| {
            let (p, q) = (x.checked_mul(tlo), x.checked_mul(thi));
            matches!((p, q), (Some(p), Some(q))
                if p.min(q).checked_add(y).is_some_and(|v| v >= ty.0)
                    && p.max(q).checked_add(y).is_some_and(|v| v <= ty.1))
        };
        (ok(a.a.into(), a.b.into()) && ok(ra, rb)).then_some(Affine {
            var: a.var,
            a: i64::try_from(ra).ok()?,
            b: i64::try_from(rb).ok()?,
        })
    }

    /// The image of input delta `d` through a read `s·R + [lo, hi]` over
    /// rows of `q` output cells: the output cells that may differ.
    pub(super) fn read_image(&self, f: FuncId, d: Delta, at: Lin, q: u64, out: ObjectId) -> Delta {
        let (s, q) = (at.c, i128::from(q));
        if s <= 0 {
            return Delta::Top;
        }
        // Output rows R with [R·s + lo, R·s + hi] meeting input row ρ:
        // R ∈ [ρ − ⌊hi/s⌋, ρ + ⌊(s − 1 − lo)/s⌋].
        let (fh, fl) = (fdiv(at.hi, s), fdiv(s - 1 - at.lo, s));
        let rows = |cells: u64| -> Option<i128> {
            let c = i128::from(cells);
            (c % s == 0).then(|| c / s)
        };
        let image = || -> Option<Delta> {
            match d {
                Delta::Rows { stride, start, .. } => {
                    let m = rows(stride)?;
                    let start = self.scale_affine(f, start, fh.checked_mul(s)?, q, s)?;
                    let width = m.checked_add(fl)?.checked_add(fh)?.checked_mul(q)?;
                    Some(Delta::Rows {
                        stride: u64::try_from(width).ok().filter(|&w| w > 0)?,
                        start,
                        next: out,
                    })
                }
                Delta::Pieces { w, step, start, .. } => {
                    let m = rows(w)?;
                    let st = rows(step)?;
                    let start = self.scale_affine(f, start, fh.checked_mul(s)?, q, s)?;
                    let width = m.checked_add(fl)?.checked_add(fh)?.checked_mul(q)?;
                    let step = st.checked_mul(q)?;
                    (width > 0 && width < step).then_some(())?;
                    Some(Delta::Pieces {
                        w: u64::try_from(width).ok()?,
                        step: u64::try_from(step).ok()?,
                        start,
                        next: out,
                    })
                }
                Delta::Top => None,
            }
        };
        image().unwrap_or(Delta::Top)
    }

    /// The trip delta of map `m`'s output given each array capture's state
    /// (`state(slot)`: `None` unchanged), trying row widths finest first.
    pub(super) fn map_image(&self, f: FuncId, m: MorphismId, state: &dyn Fn(u32) -> Chg) -> Chg {
        let Some(map) = self.morphism(m) else {
            return Chg::Arr(Delta::Top);
        };
        let out = map.target;
        for q in self.row_widths(m) {
            let Some(reads) = self.row_reads(m, q) else {
                return Chg::Arr(Delta::Top);
            };
            let mut acc = Chg::Same;
            let mut ok = true;
            for r in &reads {
                let d = match state(r.slot) {
                    Chg::Same => continue,
                    Chg::Num => Delta::Top,
                    Chg::Arr(d) => d,
                };
                let img = match r.at {
                    Some(at) => self.read_image(f, d, at, q, out),
                    None => Delta::Top,
                };
                if img == Delta::Top {
                    ok = false;
                    break;
                }
                acc = join(acc, Chg::Arr(img));
                if acc == Chg::Arr(Delta::Top) {
                    ok = false;
                    break;
                }
            }
            if ok {
                return acc;
            }
        }
        Chg::Arr(Delta::Top)
    }

    pub(super) fn unchanged_plan_compute(&self, f: FuncId) -> UnchangedPlan {
        let mut out = UnchangedPlan {
            inputs: self.input_masks(f),
            ..UnchangedPlan::default()
        };
        let wd = self.write_delta(f);
        let on = self.fact_arms().unchanged;
        let gated: Vec<MorphismId> = self
            .guard_plan(f)
            .into_iter()
            .filter(GuardSite::gated)
            .flat_map(|s| s.on_true.own.into_iter().chain(s.on_false.own))
            .collect();
        let topo = self.topo_order(f);
        for scc in self.loop_structure(f) {
            let &[merge] = scc.merges.as_slice() else {
                continue;
            };
            let Some(plan) = self.loop_plan(f, merge) else {
                continue;
            };
            let Some(next) = self.pair_slot_source(plan.back_route, 0) else {
                continue;
            };
            // write_delta's claims stand as they are (its keys: carried views,
            // back-edge values, their `Same` images).
            // write_delta's claims (its keys: carried views, back-edge values'
            // `Same` images). A view's is the cells this trip's write will
            // change; against its own previous trip it differs in the cells
            // the previous trip wrote — the same delta at the counter's
            // previous value.
            let is_view = |o: ObjectId| {
                o == merge
                    || self.in_edges(o).iter().any(|&e| {
                        self.morphism(e).is_some_and(|m| {
                            m.source == merge && matches!(m.op, Operation::Proj { .. })
                        })
                    })
            };
            let as_view = |d: Delta| self.previous_trip(f, merge, plan.back_route, d);
            for (o, d) in wd.iter() {
                if self.loop_of_view(o, merge, &plan) {
                    let d = if is_view(o) { as_view(*d) } else { *d };
                    out.trip.insert(o, d);
                    out.loop_of.insert(o, merge);
                }
            }
            // A carried write's back-edge value differs from its own last
            // trip only where this trip writes (write_delta's `next`).
            for (o, d) in wd.iter() {
                if let Delta::Rows { next, .. } | Delta::Pieces { next, .. } = *d
                    && is_view(o)
                    && self.loop_of_view(o, merge, &plan)
                {
                    out.trip.insert(next, *d);
                    out.loop_of.insert(next, merge);
                }
            }
            if !on {
                continue;
            }
            let body: Vec<MorphismId> = topo
                .iter()
                .copied()
                .filter(|m| plan.decide_order.contains(m) || plan.advance_order.contains(m))
                .collect();
            let mut st: SecondaryMap<ObjectId, Chg> = SecondaryMap::new();
            // Seeds: each state component's views — write_delta's delta, or
            // unchanged when the back edge carries the view itself.
            let merge_ty = &self.object(merge).map(|o| o.ty.clone());
            let comps: Vec<(Option<u32>, Option<ObjectId>)> = match merge_ty {
                Some(Ty::Tuple(ts)) => (0..ts.len() as u32)
                    .map(|k| (Some(k), self.pair_slot_source(next, k)))
                    .collect(),
                _ => vec![(None, Some(next))],
            };
            let mut all_same = true;
            for (k, nk) in comps {
                let views: Vec<ObjectId> = match k {
                    None => vec![merge],
                    Some(k) => self
                        .out_edges(merge)
                        .iter()
                        .filter_map(|&e| self.morphism(e))
                        .filter(|e| e.op == Operation::Proj { index: k })
                        .map(|e| e.target)
                        .collect(),
                };
                let kept = nk.is_some_and(|n| views.contains(&self.through_pairs(n)));
                for v in views {
                    let is_arr = matches!(self.object(v).map(|o| &o.ty), Some(Ty::Array { .. }));
                    let c = match (kept, is_arr) {
                        (true, _) => Chg::Same,
                        (false, true) => {
                            Chg::Arr(as_view(wd.get(v).copied().unwrap_or(Delta::Top)))
                        }
                        (false, false) => Chg::Num,
                    };
                    all_same &= c == Chg::Same;
                    st.insert(v, c);
                    if c == Chg::Same {
                        out.same.insert(v, ());
                        out.loop_of.insert(v, merge);
                    }
                }
            }
            if !all_same && k_tuple(merge_ty) {
                st.insert(merge, Chg::Num);
            }
            // A carried write's back-edge value differs from its own last
            // trip only where the view does (write_delta's `next`).
            for (o, d) in wd.iter() {
                if let Delta::Rows { next, .. } | Delta::Pieces { next, .. } = *d
                    && st.contains_key(o)
                {
                    st.insert(next, Chg::Arr(*d));
                }
            }
            let state_of = |st: &SecondaryMap<ObjectId, Chg>, o: ObjectId| -> Chg {
                st.get(o).copied().unwrap_or(Chg::Same)
            };
            // Forward, to a fixpoint (an inner loop's merge joins its back edge).
            for _ in 0..8 {
                let mut moved = false;
                for &m in &body {
                    let Some(mm) = self.morphism(m) else { continue };
                    let t = mm.target;
                    let is_arr = matches!(self.object(t).map(|o| &o.ty), Some(Ty::Array { .. }));
                    let comps = self.components(mm.source);
                    let any = comps.iter().any(|&c| state_of(&st, c) != Chg::Same)
                        || state_of(&st, mm.source) != Chg::Same;
                    // A view of the merge, and a carried write's back-edge
                    // value, keep their seeds.
                    let seeded = mm.source == merge
                        || wd.values().any(|d| {
                            matches!(*d, Delta::Rows { next, .. } | Delta::Pieces { next, .. } if next == t)
                        });
                    // (A back-edge value whose view is no seed — `write_delta`
                    // keyed it on an image — takes its own delta: it changes
                    // where this trip writes, never "unchanged".)
                    let own = || {
                        wd.values()
                            .find_map(|d| match *d {
                                Delta::Rows { next, .. } | Delta::Pieces { next, .. }
                                    if next == t =>
                                {
                                    Some(Chg::Arr(*d))
                                }
                                _ => None,
                            })
                            .unwrap_or(if is_arr {
                                Chg::Arr(Delta::Top)
                            } else {
                                Chg::Num
                            })
                    };
                    let c = if seeded {
                        st.get(t).copied().unwrap_or_else(own)
                    } else if !any {
                        Chg::Same
                    } else if gated.contains(&m) {
                        if is_arr {
                            Chg::Arr(Delta::Top)
                        } else {
                            Chg::Num
                        }
                    } else {
                        match mm.op {
                            Operation::Map { captures, .. } if is_arr => {
                                let slot_state = |k: u32| -> Chg {
                                    let o = if captures == 0 {
                                        Some(mm.source)
                                    } else {
                                        self.pair_slot_source(mm.source, k)
                                    };
                                    o.map_or(Chg::Arr(Delta::Top), |o| state_of(&st, o))
                                };
                                // A number the cells read alike that changed: every cell.
                                let num = (0..captures).any(|k| {
                                    self.pair_slot_source(mm.source, k)
                                        .is_some_and(|o| state_of(&st, o) == Chg::Num)
                                });
                                match wd.get(t) {
                                    Some(&d) => Chg::Arr(d),
                                    None if num => Chg::Arr(Delta::Top),
                                    None => self.map_image(f, m, &slot_state),
                                }
                            }
                            // ADR-0045 D7: a read in the body yields the
                            // same bytes every trip.
                            Operation::Proj { .. } if self.read_of(t).is_some() => Chg::Same,
                            Operation::Proj { index } => {
                                match self.pair_slot_source(mm.source, index) {
                                    Some(s) => state_of(&st, s),
                                    None if is_arr => Chg::Arr(Delta::Top),
                                    None => Chg::Num,
                                }
                            }
                            _ if is_arr => Chg::Arr(wd.get(t).copied().unwrap_or(Delta::Top)),
                            _ => Chg::Num,
                        }
                    };
                    let old = state_of(&st, t);
                    let new = join(old, c);
                    if new != old || !st.contains_key(t) {
                        moved |= new != old;
                        st.insert(t, new);
                    }
                }
                if !moved {
                    break;
                }
            }
            for &m in &body {
                let Some(mm) = self.morphism(m) else { continue };
                let t = mm.target;
                let is_arr = matches!(self.object(t).map(|o| &o.ty), Some(Ty::Array { .. }));
                match state_of(&st, t) {
                    Chg::Same => {
                        out.same.insert(t, ());
                        out.loop_of.insert(t, merge);
                    }
                    Chg::Arr(d) if is_arr && !out.trip.contains_key(t) => {
                        if matches!(mm.op, Operation::Map { .. })
                            && matches!(d, Delta::Rows { .. } | Delta::Pieces { .. })
                        {
                            out.derived += 1;
                        }
                        out.trip.insert(t, d);
                        out.loop_of.insert(t, merge);
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Delta `d` of this trip's write, as the cells a merge view differs in
    /// from its value on the previous trip: the previous trip's write — the
    /// counter one step back (`var − c`, the back edge carrying `var + c`).
    /// `Top` when the step is unknown or the shifted start can wrap.
    pub(super) fn previous_trip(
        &self,
        f: FuncId,
        merge: ObjectId,
        back_route: ObjectId,
        d: Delta,
    ) -> Delta {
        let start = match d {
            Delta::Rows { start, .. } | Delta::Pieces { start, .. } => start,
            Delta::Top => return Delta::Top,
        };
        let step = || -> Option<i64> {
            let next = self.pair_slot_source(back_route, 0)?;
            let k = match self
                .in_edges(start.var)
                .first()
                .and_then(|&e| self.morphism(e))
            {
                Some(m) if m.source == merge => match m.op {
                    Operation::Proj { index } => Some(index),
                    _ => None,
                },
                _ if start.var == merge => None,
                _ => return None,
            };
            let nk = match k {
                Some(k) => self.pair_slot_source(next, k)?,
                None => next,
            };
            let mut var = None;
            let (a, c) = self.affine_in(nk, merge, &mut var, 0)?;
            (a == 1 && var == Some(start.var))
                .then(|| i64::try_from(c).ok())
                .flatten()
        };
        let Some(c) = step() else {
            return Delta::Top;
        };
        let Some(shift) = i128::from(start.a).checked_mul(c.into()) else {
            return Delta::Top;
        };
        let Some(start) = self.scale_affine(f, start, shift, 1, 1) else {
            return Delta::Top;
        };
        match d {
            Delta::Rows { stride, next, .. } => Delta::Rows {
                stride,
                start,
                next,
            },
            Delta::Pieces { w, step, next, .. } => Delta::Pieces {
                w,
                step,
                start,
                next,
            },
            Delta::Top => Delta::Top,
        }
    }

    /// `o` belongs to loop `merge`: one of its views, or computed in its body.
    pub(super) fn loop_of_view(
        &self,
        o: ObjectId,
        merge: ObjectId,
        plan: &super::super::LoopPlan,
    ) -> bool {
        if o == merge {
            return true;
        }
        self.in_edges(o).iter().any(|&e| {
            self.morphism(e).is_some_and(|m| {
                m.source == merge
                    || plan.decide_order.contains(&e)
                    || plan.advance_order.contains(&e)
            })
        })
    }

    /// The `Pair`-assembled components of a product object (empty for a
    /// value that is no assembled product).
    pub(super) fn components(&self, o: ObjectId) -> Vec<ObjectId> {
        self.in_edges(o)
            .iter()
            .filter_map(|&e| self.morphism(e))
            .filter(|m| matches!(m.op, Operation::Pair { .. }))
            .map(|m| m.source)
            .collect()
    }

    /// The objects that can be component `k` of product `o`: its `Pair{k}`
    /// feeder, or — a loop merge — component `k` of its init and of its
    /// back-edge state. `None` when one is not known.
    pub(super) fn component_sources(&self, o: ObjectId, k: u32) -> Option<Vec<ObjectId>> {
        if let Some(s) = self.pair_slot_source(o, k) {
            return Some(vec![s]);
        }
        let ins = self.in_edges(o);
        if ins.is_empty() {
            return None;
        }
        let mut out = Vec::new();
        for &e in ins {
            let m = self.morphism(e)?;
            let state = match m.op {
                Operation::LoopEnter => m.source,
                Operation::LoopBack => self.pair_slot_source(m.source, 0)?,
                _ => return None,
            };
            out.push(self.pair_slot_source(state, k)?);
        }
        Some(out)
    }

    /// Across calls: the input slots of `f` each of its values depends on
    /// (a fixpoint over every edge; a clock read depends on everything).
    pub(super) fn input_masks(&self, f: FuncId) -> SecondaryMap<ObjectId, u64> {
        let mut mask: SecondaryMap<ObjectId, u64> = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return mask;
        };
        let objs = self.func_objects(f);
        for &o in &objs {
            mask.insert(o, 0);
        }
        let input = fd.input;
        let tuple = matches!(self.object(input).map(|o| &o.ty), Some(Ty::Tuple(_)));
        // A whole-input read (not a slot's `Proj`) depends on every slot.
        mask.insert(input, if tuple { u64::MAX } else { 1 });
        for &m in &fd.morphisms {
            if let Some(mm) = self.morphism(m)
                && mm.source == input
                && let Operation::Proj { index } = mm.op
            {
                mask.insert(mm.target, 1u64 << index.min(63));
            }
        }
        for _ in 0..objs.len().max(1) {
            let mut moved = false;
            for &m in &fd.morphisms {
                let Some(mm) = self.morphism(m) else { continue };
                if mm.source == input && matches!(mm.op, Operation::Proj { .. }) {
                    continue;
                }
                let add = match mm.op {
                    Operation::TimeMs => u64::MAX,
                    // ADR-0045 D7: a read's array is the file's bytes — the
                    // same at every call and every trip (a literal path, the
                    // recorded directory, a file the run does not change).
                    Operation::Proj { .. } if self.read_of(mm.target).is_some() => 0,
                    Operation::Proj { index } => match self.component_sources(mm.source, index) {
                        Some(cs) => cs
                            .iter()
                            .map(|&c| mask.get(c).copied().unwrap_or(0))
                            .fold(0, |a, b| a | b),
                        None => mask.get(mm.source).copied().unwrap_or(0),
                    },
                    _ => mask.get(mm.source).copied().unwrap_or(0),
                };
                let old = mask.get(mm.target).copied().unwrap_or(0);
                if old | add != old {
                    mask.insert(mm.target, old | add);
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        mask
    }
}
