//! `plan_views`, roots, sharing, death and par-safety of the frame planning

use super::*;

impl<'a> FnEmit<'a> {
    /// **S65 rule R, host half: forwarding is not observing.** `Pair`, `Proj`,
    /// `Phi` and a loop's enter, back edge and exit re-expose an array, so they
    /// move its address: a product's array position (component path) is a
    /// pointer unless a consumer reads the product whole (`Output`, a by-value
    /// body input, …), and a bare array a forwarding morphism makes is a view
    /// (`ptr_resident`). Positions one whole-value copy joins (a loop's
    /// products, nested products, `Phi` arms) are decided together. A pure
    /// function of the IR and the rung: the residency plan and every emitter of
    /// the function (the host and its tasks) lay products out alike.
    pub(in crate::func) fn plan_views(&mut self) {
        if std::mem::replace(&mut self.views_done, true) {
            return;
        }
        self.by_ref = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).by_ref;
        if !self.by_ref {
            return;
        }
        self.init_abi();
        for scc in self.ir.loop_structure(self.f) {
            for plan in scc
                .merges
                .iter()
                .filter_map(|&mg| self.ir.loop_plan(self.f, mg))
            {
                let body = plan
                    .decide_order
                    .iter()
                    .chain(&plan.advance_order)
                    .copied()
                    .collect();
                self.loops.push(Loop { plan, body });
            }
        }
        let fd = self.ir.func(self.f).expect("func resolves");
        let mut cand: std::collections::HashSet<Pos> = Default::default();
        for (o, kind, ty) in self.owned_objects() {
            let op = self
                .ir
                .in_edges(o)
                .first()
                .map(|&m| self.ir.morphism(m).expect("m").op);
            let fwd = matches!(
                op,
                Some(
                    Operation::Pair { .. }
                        | Operation::Proj { .. }
                        | Operation::Phi
                        | Operation::LoopExit
                )
            );
            let kept = kind == ObjectKind::LoopMerge || fwd && kind == ObjectKind::Temporary;
            if !kept || o == fd.input || o == fd.output {
                continue;
            }
            // (A bare array `Pair`s build is a literal: its own bytes.)
            let literal = matches!(op, Some(Operation::Pair { .. }));
            cand.extend(
                array_paths(&ty)
                    .into_iter()
                    .filter(|p| !(literal && p.is_empty()))
                    .map(|p| (o, p)),
            );
        }
        // structure: the by-ref input's nested pointer positions are the
        // ABI's — pinned. A consumer that needs bytes there converts at its
        // edge (`copy_layout`); the input is never anything but pointers.
        let pinned: std::collections::HashSet<Pos> = match &self.byref {
            Some((input, _, _)) => self
                .byref_paths
                .iter()
                .filter(|p| !p.is_empty())
                .map(|p| (*input, p.clone()))
                .collect(),
            None => Default::default(),
        };
        cand.extend(pinned.iter().cloned());
        // structure: an `Index` of an array of arrays or of products is a
        // view of the element (its address inside the array's bytes), unless
        // a consumer needs the bytes.
        for (o, _, ty) in self.owned_objects() {
            if self
                .ir
                .in_edges(o)
                .first()
                .is_some_and(|&m| self.ir.morphism(m).expect("m").op == Operation::Index)
            {
                cand.extend(array_paths(&ty).into_iter().map(|p| (o, p)));
            }
        }
        // `eq`: one whole-value copy moves both, so both are pointers or
        // neither; `need`: a view of a field needs the field to be a pointer.
        let (mut eq, mut need): (Vec<(Pos, Pos)>, Vec<(Pos, Pos)>) = Default::default();
        for (_, mo) in self
            .ir
            .morphisms()
            .filter(|(_, mo)| self.ir.try_owner(mo.source) == Some(self.f))
        {
            let (s, t) = (mo.source, mo.target);
            let nested = |o| {
                array_paths(&self.obj_ty(o))
                    .into_iter()
                    .filter(|q| !q.is_empty())
            };
            let pres: &[&[u32]] = match mo.op {
                Operation::Pair { slot, .. } => {
                    eq.extend(nested(s).map(|q| ((t, ext(&[slot], &q)), (s, q))));
                    &[]
                }
                Operation::Proj { index } => {
                    for q in array_paths(&self.obj_ty(t)) {
                        let pair = ((t, q.clone()), (s, ext(&[index], &q)));
                        if q.is_empty() {
                            need.push(pair)
                        } else {
                            eq.push(pair)
                        }
                    }
                    &[]
                }
                Operation::LoopEnter => &[&[]],
                Operation::LoopBack | Operation::LoopExit => &[&[0]],
                Operation::Phi => &[&[0], &[1]],
                _ => &[],
            };
            for q in nested(t) {
                eq.extend(pres.iter().map(|pre| ((t, q.clone()), (s, ext(pre, &q)))));
            }
            // A consumer reading a product position as anything but an
            // address (or a forward) needs the bytes there.
            for p in nested(s) {
                let (k, one) = (p[0], p.len() == 1);
                let address = match mo.op {
                    Operation::Index | Operation::Update => one && k == 0,
                    Operation::Zip => one && k <= 1,
                    // structure: a capture's arrays at any depth and a call
                    // argument's travel by address (`capture_paths`,
                    // `lower_named_input_ty`).
                    Operation::Map { captures, .. } => one && k <= captures || k < captures,
                    Operation::Fold { captures, .. } => one && k == captures + 1 || k < captures,
                    Operation::Call(_) => true,
                    op => {
                        !pres.is_empty()
                            || matches!(op, Operation::Pair { .. } | Operation::Proj { .. })
                    }
                };
                if !address && !pinned.contains(&(s, p.clone())) {
                    cand.remove(&(s, p));
                }
            }
        }
        loop {
            let mut changed = false;
            for (a, b) in &eq {
                if cand.contains(a) != cand.contains(b) {
                    changed |= (!pinned.contains(a) && cand.remove(a))
                        | (!pinned.contains(b) && cand.remove(b));
                }
            }
            for (v, n) in &need {
                changed |= !cand.contains(n) && !pinned.contains(v) && cand.remove(v);
            }
            if !changed {
                break;
            }
        }
        let mut cand: Vec<Pos> = cand.into_iter().filter(|x| !pinned.contains(x)).collect();
        cand.sort();
        for (o, p) in cand {
            if p.is_empty() {
                self.ptr_resident.insert(o, ());
            } else {
                self.views.entry(o).expect("live key").or_default().push(p);
            }
        }
    }

    /// Position `p` of `o` holds a pointer (`[]`: `o` is one).
    pub(in crate::func) fn is_view(&self, o: ObjectId, p: &[u32]) -> bool {
        if self.byref.as_ref().is_some_and(|(input, _, _)| *input == o) {
            return self.byref_paths.iter().any(|q| q == p);
        }
        if p.is_empty() {
            self.ptr_resident.contains_key(o)
        } else {
            self.views.get(o).is_some_and(|v| v.iter().any(|q| q == p))
        }
    }

    /// The buffers the bytes at position `p` of `o` may be: each view followed
    /// back to the bytes it exposes (both arms of a `Phi`), and, with `alias`,
    /// each in-place `Update` to the array it writes.
    pub(in crate::func) fn roots(&self, o: ObjectId, p: &[u32], alias: bool) -> Vec<Root> {
        let (mut out, mut stack, mut steps) = (Vec::new(), vec![(o, p.to_vec())], 0);
        while let Some((o, p)) = stack.pop() {
            steps += 1;
            let up = |k: u32| ext(&[k], &p);
            let r = match self
                .ir
                .in_edges(o)
                .first()
                .map(|&m| self.ir.morphism(m).expect("m"))
            {
                _ if steps > 4096 => Root::Other,
                _ if alias && p.is_empty() && self.update_aliases.contains_key(o) => {
                    stack.push((self.update_aliases[o], p));
                    continue;
                }
                _ if !self.is_view(o, &p) => {
                    if p.is_empty() {
                        Root::Obj(o)
                    } else {
                        Root::Other
                    }
                }
                _ if self.ir.object(o).expect("object").kind == ObjectKind::LoopMerge => {
                    Root::Merge(o, p)
                }
                Some(mo) => match mo.op {
                    Operation::Pair { .. } => match self.pair_source(o, p[0]) {
                        Some(s) => {
                            stack.push((s, p[1..].to_vec()));
                            continue;
                        }
                        None => Root::Other,
                    },
                    Operation::Proj { index } => {
                        stack.push((mo.source, up(index)));
                        continue;
                    }
                    Operation::LoopExit | Operation::Phi => {
                        stack.push((mo.source, up(0)));
                        if mo.op == Operation::Phi {
                            stack.push((mo.source, up(1)));
                        }
                        continue;
                    }
                    // A view written in place, seen as its own bytes: an
                    // `Update`, or a piecewise map run in place over its set
                    // (`in_place_source`). Else the state looks unowned and
                    // the entry copy of a still-read init is skipped.
                    Operation::Update => Root::Obj(o),
                    Operation::Map { .. } if self.update_aliases.contains_key(o) => Root::Obj(o),
                    _ => Root::Other,
                },
                None => Root::Other,
            };
            if !out.contains(&r) {
                out.push(r);
            }
        }
        out
    }

    /// `o` is written in the trip.
    pub(super) fn made_in(&self, lp: &Loop, o: ObjectId) -> bool {
        self.ir.in_edges(o).iter().any(|m| lp.body.contains(m))
    }

    /// Where `o` is complete: its last producing edge (a product's last `Pair`).
    pub(super) fn born(&self, o: ObjectId) -> u32 {
        self.ir
            .in_edges(o)
            .iter()
            .filter_map(|&m| self.lup.position(m))
            .max()
            .unwrap_or(0)
    }

    /// The state position is carried on the device (no host bytes).
    pub(super) fn device_carried(&self, merge: ObjectId, p: &[u32]) -> bool {
        p.len() == 1
            && self
                .resident
                .as_ref()
                .is_some_and(|r| r.carries.iter().any(|c| c.merge == merge && c.k == p[0]))
    }

    /// Every object by the bytes its views lead to (`roots` without the
    /// in-place hops), built once the views are.
    pub(super) fn index_roots(&mut self) {
        let mut idx: std::collections::HashMap<Root, Vec<ObjectId>> = Default::default();
        for (o, _, ty) in self.owned_objects() {
            let own = !self.ptr_resident.contains_key(o) && matches!(ty, Ty::Array { .. });
            let mut rs: Vec<Root> = own.then_some(Root::Obj(o)).into_iter().collect();
            let paths = match self.ptr_resident.contains_key(o) {
                true => vec![Vec::new()],
                false => self.views.get(o).cloned().unwrap_or_default(),
            };
            for r in paths.iter().flat_map(|q| self.roots(o, q, false)) {
                if !rs.contains(&r) {
                    rs.push(r);
                }
            }
            for r in rs {
                idx.entry(r).or_default().push(o);
            }
        }
        self.root_index = idx;
    }

    /// The objects whose bytes may be `root`'s: its views, and the `Update`s
    /// written in place into it with theirs.
    pub(super) fn sharing(&self, root: &Root) -> Vec<ObjectId> {
        let mut fam: Vec<ObjectId> = self.root_index.get(root).cloned().unwrap_or_default();
        let mut set: std::collections::HashSet<ObjectId> = fam.iter().copied().collect();
        let next = |set: &std::collections::HashSet<ObjectId>| {
            self.update_aliases
                .iter()
                .find(|&(t, s)| set.contains(s) && !set.contains(&t))
                .map(|(t, _)| t)
        };
        while let Some(t) = next(&set) {
            for &o in
                std::iter::once(&t).chain(self.root_index.get(&Root::Obj(t)).into_iter().flatten())
            {
                if set.insert(o) {
                    fam.push(o);
                }
            }
        }
        fam
    }

    /// `o` is read from a guard arm (which runs at its `Phi`, past its own
    /// position) — the `Update` `upd` reading it aside.
    pub(in crate::func) fn guard_read(&self, o: ObjectId, upd: Option<MorphismId>) -> bool {
        let src = upd.map(|u| self.ir.morphism(u).expect("m").source);
        self.ir.out_edges(o).iter().any(|&e| {
            self.gated.contains_key(e)
                && Some(e) != upd
                && Some(self.ir.morphism(e).expect("m").target) != src
        })
    }

    /// The last position of the trip at which a buffer the state position `p`
    /// may be is read: over every object of the body whose bytes it may be,
    /// views included (`LastUsePlan::view_last_use`). For the `Update` `upd`:
    /// only the objects made before it. A whole-state or a guard-arm read
    /// reads any time.
    pub(super) fn state_death(&self, lp: &Loop, p: &[u32], upd: Option<(MorphismId, u32)>) -> u32 {
        let merge = lp.plan.merge;
        let whole = |m| !matches!(self.ir.morphism(m).expect("m").op, Operation::Proj { .. });
        if self.ir.out_edges(merge).iter().any(|&m| whole(m)) {
            return u32::MAX;
        }
        let mut d = 0;
        for o in self.sharing(&Root::Merge(merge, p.to_vec())) {
            if !self.made_in(lp, o) || upd.is_some_and(|(_, at)| self.born(o) >= at) {
                continue;
            }
            if self.guard_read(o, upd.map(|(u, _)| u)) {
                return u32::MAX;
            }
            d = d.max(self.lup.view_last_use(o).unwrap_or(0));
        }
        d
    }

    /// The init's array `a` at position `p` may become the loop's: this
    /// function's own bytes, whose one use is the init's products (each the
    /// next one's only feeder; the init's only use the `LoopEnter`).
    pub(super) fn takeable(&self, lp: &Loop, p: &[u32], a: ObjectId) -> bool {
        let mut o = lp.plan.init;
        let mut chain = vec![o];
        for &k in p {
            match self.pair_source(o, k) {
                Some(s) => o = s,
                None => return false,
            }
            chain.push(o);
        }
        o == a
            && self.ir.object(a).expect("object").kind == ObjectKind::Temporary
            && !self.ptr_resident.contains_key(a)
            && !self.host_free(a)
            && chain.iter().all(|&c| self.ir.out_edges(c).len() == 1)
    }

    /// The loop owns state position `p`'s buffer for an in-place write: every
    /// buffer its next state may be is the state's own or written in the
    /// trip, for this position alone. `Some(true)`: the init's buffer is the
    /// loop's; `Some(false)`: the entry copies the init into one.
    pub(super) fn state_owner(&self, lp: &Loop, p: &[u32]) -> Option<bool> {
        let (merge, next) = (lp.plan.merge, |q: &[u32]| {
            self.roots(lp.plan.back_route, &ext(&[0], q), false)
        });
        let others: Vec<Root> = array_paths(&self.obj_ty(merge))
            .iter()
            .filter(|q| q.as_slice() != p)
            .flat_map(|q| next(q))
            .collect();
        let owned = next(p).iter().all(|r| match r {
            Root::Merge(m, q) => *m == merge && q == p,
            Root::Obj(o) => self.made_in(lp, *o) && !others.contains(r),
            Root::Other => false,
        });
        (owned && !self.device_carried(merge, p))
            .then(|| matches!(self.roots(lp.plan.init, p, false)[..], [Root::Obj(a)] if self.takeable(lp, p, a)))
    }

    /// In the parallel entry, every read of `arr`'s bytes the positions put
    /// before the write `m` (every read of the old bytes, by `dead_after`)
    /// runs in `m`'s task or one the plan orders before it. (S64 wrote a loop's
    /// exit in place while another task still read it.)
    pub(in crate::func) fn par_safe(&self, arr: ObjectId, m: MorphismId) -> bool {
        let Some(par) = &self.par else { return true };
        let Some(&tw) = par.task.get(m) else {
            return false;
        };
        let src = self.ir.morphism(m).expect("m").source;
        let (mut seen, mut stack): (SecondaryMap<ObjectId, ()>, _) =
            (SecondaryMap::new(), vec![arr]);
        while let Some(o) = stack.pop() {
            if seen.insert(o, ()).is_some() {
                continue;
            }
            for &e in self.ir.out_edges(o) {
                let mo = self.ir.morphism(e).expect("m");
                if e == m || mo.target == src {
                    continue;
                }
                // A loop's own edges run where its body does; a loop's exit
                // object reads its final state, after the loop.
                let loop_op = matches!(
                    mo.op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                );
                let ordered = par
                    .task
                    .get(e)
                    .map_or(loop_op, |&tr| tr == tw || par.before[tr][tw]);
                if self.lup.position(e) < self.lup.position(m) && !ordered {
                    return false;
                }
                if matches!(
                    mo.op,
                    Operation::Pair { .. } | Operation::Proj { .. } | Operation::Phi
                ) || mo.op == Operation::LoopEnter
                {
                    stack.push(mo.target);
                }
            }
        }
        true
    }
}
