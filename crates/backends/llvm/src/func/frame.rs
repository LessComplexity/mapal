//! storage preparation, elided arrays, local slots, the `%Frame` layout
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

impl<'a> FnEmit<'a> {
    /// Configure by-ref inputs and Update slot aliases. Returns the lowered
    /// incoming argument type.
    pub(super) fn prepare_storage(&mut self) -> Option<String> {
        let fd = self.ir.func(self.f).expect("func resolves");
        let in_ty = self.obj_ty(fd.input);
        let (bk, btext) = match fd.kind {
            FuncKind::MapBody | FuncKind::FoldBody => {
                let k = self.body_captures();
                (k, lower_body_input_ty(&in_ty, k))
            }
            FuncKind::Named => (u32::MAX, lower_named_input_ty(&in_ty)),
        };
        if let Some(text) = btext {
            self.byref = Some((fd.input, bk, text));
            if bk == u32::MAX && matches!(&in_ty, Ty::Array { .. }) {
                self.ptr_resident.insert(fd.input, ());
            }
            let owned_ids: Vec<ObjectId> = self
                .ir
                .objects()
                .filter(|(id, _)| self.ir.try_owner(*id) == Some(self.f))
                .map(|(id, _)| id)
                .collect();
            for id in owned_ids {
                for &m in self.ir.in_edges(id) {
                    let morph = self.ir.morphism(m).expect("morphism resolves");
                    if let Operation::Proj { index } = morph.op
                        && morph.source == fd.input
                        && index < bk
                        && matches!(in_ty.component_ty(index), Some(Ty::Array { .. }))
                    {
                        self.ptr_resident.insert(id, ());
                    }
                }
            }
        }
        let in_llt = match &self.byref {
            Some((_, _, text)) => Some(text.clone()),
            None => lower_ty(&in_ty),
        };

        // S65 `by-ref`: forwarding moves pointers (views); a device carry's
        // exit with host bytes holds its own (the host state has none to point
        // at). Then the in-place `Update`s, to a fixed point (a chain's next
        // link writes in place once its source does), and the loop buffers.
        self.plan_views();
        let exits: Vec<ObjectId> =
            self.resident.iter().flat_map(|p| p.carries.iter().filter_map(|c| c.exit)).collect();
        for x in exits {
            if !self.host_free(x) {
                self.ptr_resident.remove(x);
            }
        }
        if self.by_ref {
            self.index_roots();
        }
        let ms = fd.morphisms.clone();
        let mut grew = true;
        while std::mem::take(&mut grew) {
            for &m in &ms {
                let t = self.ir.morphism(m).expect("morphism resolves").target;
                if self.update_aliases.contains_key(t) {
                    continue;
                }
                if let Some(s) = self.in_place_source(m) {
                    self.elided_updates.insert(t, ());
                    self.update_aliases.insert(t, s);
                    if self.ptr_resident.contains_key(s) {
                        self.ptr_resident.insert(t, ());
                    }
                    grew = true;
                }
            }
        }
        self.plan_loop_buffers();
        self.mark_elided_arrays(&ms);
        // S54 `arena`: a device-only array has no host bytes to hold.
        if let Some(a) = self.resident.as_ref().and_then(|p| p.arena.as_ref()) {
            for o in a.host_free.keys() {
                self.elided_arrays.insert(o, ());
            }
        }
        in_llt
    }

    /// Step 3b: find arrays no one will ever load, because every consumer
    /// rebuilds the element from its law (`elem_plan`).
    ///
    /// Deliberately narrow. An array qualifies only when **every** out-edge is a
    /// `Map`/`Fold` reading it directly as the mapped/folded array — the
    /// capture-free shape. A captured consumer reaches its array through a
    /// `Pair` product, and following that chain is more analysis than the win
    /// justifies today; those arrays keep their buffer. Conservative in the safe
    /// direction: a missed elision costs a store pass, a wrong one dereferences
    /// a field that does not exist.
    pub(super) fn mark_elided_arrays(&mut self, morphisms: &[MorphismId]) {
        for &m in morphisms {
            let Some(morph) = self.ir.morphism(m) else {
                continue;
            };
            if !matches!(
                morph.op,
                Operation::Iota | Operation::Fill | Operation::Zip | Operation::Enumerate
            ) {
                continue;
            }
            let arr = morph.target;
            // The law must be one a consumer can actually build. `Apply` is
            // excluded because THIS backend declines it (see `APPLY_INLINE`)
            // and a declined `Apply` degrades to loading exactly this array.
            match self.elem.src(arr) {
                Some(ElemSrc::Index | ElemSrc::Broadcast { .. } | ElemSrc::Pair(..)) => {}
                _ => continue,
            }
            if self.ir.object(arr).map(|o| o.kind) != Some(ObjectKind::Temporary) {
                continue;
            }
            let consumers = self.ir.out_edges(arr);
            if consumers.is_empty() {
                continue;
            }
            let all_inline = consumers.iter().all(|&c| {
                let Some(cm) = self.ir.morphism(c) else {
                    return false;
                };
                match cm.op {
                    Operation::Map { captures: 0, .. } => cm.source == arr,
                    Operation::Fold { .. } | Operation::Map { .. } => false,
                    _ => false,
                }
            });
            if all_inline {
                self.elided_arrays.insert(arr, ());
            }
        }
    }

    pub(super) fn owned_objects(&self) -> Vec<(ObjectId, ObjectKind, Ty)> {
        self.ir
            .objects()
            .filter(|(id, _)| self.ir.try_owner(*id) == Some(self.f))
            .map(|(id, obj)| (id, obj.kind, obj.ty.clone()))
            .collect()
    }

    pub(super) fn slot_type(
        &self,
        id: ObjectId,
        ty: &Ty,
        in_llt: &Option<String>,
    ) -> Option<String> {
        if self.ptr_resident.contains_key(id) {
            Some("ptr".into())
        } else if Some(id) == self.byref.as_ref().map(|(input, _, _)| *input) {
            in_llt.clone()
        } else {
            self.lower_slot_ty(id, ty)
        }
    }

    pub(super) fn allocate_local_slots(&mut self, in_llt: &Option<String>) {
        let mut ord = 0u32;
        for (id, kind, ty) in self.owned_objects() {
            if kind == ObjectKind::Constant {
                continue;
            }
            if self.elided_updates.contains_key(id) || self.elided_arrays.contains_key(id) {
                ord += 1;
                continue;
            }
            if let Some(llt) = self.slot_type(id, &ty, in_llt) {
                let name = format!("%o{ord}");
                self.slots.insert(id, name.clone());
                self.entry_alloc(&name, &llt, None);
            }
            ord += 1;
        }
    }

    pub(super) fn build_frame_layout(
        &self,
        in_llt: &Option<String>,
        path_plan: &PathPlan,
    ) -> FrameLayout {
        let mut fields = SecondaryMap::new();
        let mut order = Vec::new();
        let mut ord = 0u32;
        for (id, kind, ty) in self.owned_objects() {
            if kind == ObjectKind::Constant {
                continue;
            }
            // step 3b: an array nobody loads needs no storage. Dropping the
            // FIELD is the part DCE cannot do for us — `%Frame` is one object
            // shared across every task, so an unread member still costs its
            // bytes in the allocation.
            if self.elided_updates.contains_key(id) || self.elided_arrays.contains_key(id) {
                ord += 1;
                continue;
            }
            if let Some(llt) = self.slot_type(id, &ty, in_llt) {
                let index = order.len() as u32;
                fields.insert(
                    id,
                    FrameField {
                        owner: id,
                        index,
                        ordinal: ord,
                        llt,
                    },
                );
                order.push(id);
            }
            ord += 1;
        }
        for (target, _) in self.update_aliases.iter() {
            let mut source = self.update_aliases[target];
            while let Some(&next) = self.update_aliases.get(source) {
                source = next;
            }
            let field = fields
                .get(source)
                .expect("elided Update source has a frame field")
                .clone();
            fields.insert(target, field);
        }
        let mut packed = SecondaryMap::new();
        if self.packing
            && let Some(tile_plan) = &self.tile_plan
        {
            for task in &path_plan.tasks {
                if let TaskKind::Split { site: m, .. } = &task.kind
                    && let Some(site) = tile_plan.sites.get(*m)
                    && packing_site(site, self.profile, self.contract)
                {
                    packed.insert(
                        *m,
                        PackedField {
                            index: (order.len() + packed.len()) as u32,
                            ordinal: packed.len() as u32,
                        },
                    );
                }
            }
        }
        FrameLayout {
            fields,
            order,
            packed,
        }
    }

    pub(super) fn materialize_frame_slots(&mut self) {
        let order = self.frame.as_ref().expect("frame layout").order.clone();
        for o in order {
            self.slot(o).expect("frame field resolves");
        }
    }
}

// ---- S65 rule R on the host: views, and the buffers of a loop's state ------------------------------

/// A loop state position's buffers ([`FnEmit::plan_loop_buffers`]). `Own`:
/// the trip writes the state in place and the init's bytes are read
/// elsewhere, so the entry copies them into the loop's own buffer. `Colors`:
/// the next state's bytes `root` are written while the state they replace is
/// read; each trip they take the first of `root`'s slot and one spare per
/// state position `at` they become (the first spare the init's, `take`, when
/// it dies at the entry) that no such position points at.
#[derive(Clone, Debug)]
pub(crate) enum LoopBuf {
    Own { merge: ObjectId, path: Vec<u32> },
    Colors { merge: ObjectId, root: ObjectId, at: Vec<Vec<u32>>, take: Option<ObjectId> },
}

/// A canonical loop of the function and its body (decide and advance cones).
#[derive(Clone, Debug)]
pub(crate) struct Loop {
    pub plan: mapal_ir::LoopPlan,
    pub body: std::collections::HashSet<MorphismId>,
}

/// Where the bytes behind a view are: an object's own, a loop state
/// position's (the init's, then each trip's next state's), or bytes this
/// function does not hold (a borrowed input's, an inline field's).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Root {
    Obj(ObjectId),
    Merge(ObjectId, Vec<u32>),
    Other,
}

/// The parallel entry's task of each morphism and the plan's order of tasks
/// (transitive): positions order one task, these order tasks.
#[derive(Debug)]
pub(crate) struct ParOrder {
    task: SecondaryMap<MorphismId, usize>,
    before: Vec<Vec<bool>>,
}

impl ParOrder {
    pub(crate) fn new(plan: &PathPlan) -> Self {
        let n = plan.tasks.len();
        let (mut task, mut before) = (SecondaryMap::new(), vec![vec![false; n]; n]);
        for (t, tk) in plan.tasks.iter().enumerate() {
            let ms = match &tk.kind {
                TaskKind::Split { site, .. } => std::slice::from_ref(site),
                TaskKind::Seq { morphisms } => morphisms.as_slice(),
            };
            for &m in ms {
                task.insert(m, t);
            }
            for &d in &tk.deps {
                before[d][t] = true;
            }
        }
        for k in 0..n {
            let row = before[k].clone();
            for i in 0..n {
                if before[i][k] {
                    before[i].iter_mut().zip(&row).for_each(|(b, &r)| *b |= r);
                }
            }
        }
        ParOrder { task, before }
    }
}

fn ext(pre: &[u32], q: &[u32]) -> Vec<u32> {
    pre.iter().chain(q).copied().collect()
}

type Pos = (ObjectId, Vec<u32>);

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
    pub(super) fn plan_views(&mut self) {
        if std::mem::replace(&mut self.views_done, true) {
            return;
        }
        self.by_ref = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).by_ref;
        if !self.by_ref {
            return;
        }
        for scc in self.ir.loop_structure(self.f) {
            for plan in scc.merges.iter().filter_map(|&mg| self.ir.loop_plan(self.f, mg)) {
                let body = plan.decide_order.iter().chain(&plan.advance_order).copied().collect();
                self.loops.push(Loop { plan, body });
            }
        }
        let fd = self.ir.func(self.f).expect("func resolves");
        let mut cand: std::collections::HashSet<Pos> = Default::default();
        for (o, kind, ty) in self.owned_objects() {
            let op = self.ir.in_edges(o).first().map(|&m| self.ir.morphism(m).expect("m").op);
            let fwd = matches!(
                op,
                Some(Operation::Pair { .. } | Operation::Proj { .. } | Operation::Phi | Operation::LoopExit)
            );
            let kept = kind == ObjectKind::LoopMerge || fwd && kind == ObjectKind::Temporary;
            if !kept || o == fd.input || o == fd.output {
                continue;
            }
            // (A bare array `Pair`s build is a literal: its own bytes.)
            let literal = matches!(op, Some(Operation::Pair { .. }));
            cand.extend(array_paths(&ty).into_iter().filter(|p| !(literal && p.is_empty())).map(|p| (o, p)));
        }
        // `eq`: one whole-value copy moves both, so both are pointers or
        // neither; `need`: a view of a field needs the field to be a pointer.
        let (mut eq, mut need): (Vec<(Pos, Pos)>, Vec<(Pos, Pos)>) = Default::default();
        for (_, mo) in self.ir.morphisms().filter(|(_, mo)| self.ir.try_owner(mo.source) == Some(self.f)) {
            let (s, t) = (mo.source, mo.target);
            let nested = |o| array_paths(&self.obj_ty(o)).into_iter().filter(|q| !q.is_empty());
            let pres: &[&[u32]] = match mo.op {
                Operation::Pair { slot, .. } => {
                    eq.extend(nested(s).map(|q| ((t, ext(&[slot], &q)), (s, q))));
                    &[]
                }
                Operation::Proj { index } => {
                    for q in array_paths(&self.obj_ty(t)) {
                        let pair = ((t, q.clone()), (s, ext(&[index], &q)));
                        if q.is_empty() { need.push(pair) } else { eq.push(pair) }
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
                    Operation::Map { captures, .. } => one && k <= captures,
                    Operation::Fold { captures, .. } => one && (k < captures || k == captures + 1),
                    Operation::Call(_) => one,
                    op => !pres.is_empty() || matches!(op, Operation::Pair { .. } | Operation::Proj { .. }),
                };
                if !address {
                    cand.remove(&(s, p));
                }
            }
        }
        loop {
            let mut changed = false;
            for (a, b) in &eq {
                if cand.contains(a) != cand.contains(b) {
                    changed |= cand.remove(a) | cand.remove(b);
                }
            }
            for (v, n) in &need {
                changed |= !cand.contains(n) && cand.remove(v);
            }
            if !changed {
                break;
            }
        }
        let mut cand: Vec<Pos> = cand.into_iter().collect();
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
    pub(super) fn is_view(&self, o: ObjectId, p: &[u32]) -> bool {
        if p.is_empty() {
            self.ptr_resident.contains_key(o)
        } else {
            self.views.get(o).is_some_and(|v| v.iter().any(|q| q == p))
        }
    }

    /// The buffers the bytes at position `p` of `o` may be: each view followed
    /// back to the bytes it exposes (both arms of a `Phi`), and, with `alias`,
    /// each in-place `Update` to the array it writes.
    pub(super) fn roots(&self, o: ObjectId, p: &[u32], alias: bool) -> Vec<Root> {
        let (mut out, mut stack, mut steps) = (Vec::new(), vec![(o, p.to_vec())], 0);
        while let Some((o, p)) = stack.pop() {
            steps += 1;
            let up = |k: u32| ext(&[k], &p);
            let r = match self.ir.in_edges(o).first().map(|&m| self.ir.morphism(m).expect("m")) {
                _ if steps > 4096 => Root::Other,
                _ if alias && p.is_empty() && self.update_aliases.contains_key(o) => {
                    stack.push((self.update_aliases[o], p));
                    continue;
                }
                _ if !self.is_view(o, &p) => if p.is_empty() { Root::Obj(o) } else { Root::Other },
                _ if self.ir.object(o).expect("object").kind == ObjectKind::LoopMerge => Root::Merge(o, p),
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
    fn made_in(&self, lp: &Loop, o: ObjectId) -> bool {
        self.ir.in_edges(o).iter().any(|m| lp.body.contains(m))
    }

    /// Where `o` is complete: its last producing edge (a product's last `Pair`).
    fn born(&self, o: ObjectId) -> u32 {
        self.ir.in_edges(o).iter().filter_map(|&m| self.lup.position(m)).max().unwrap_or(0)
    }

    /// The state position is carried on the device (no host bytes).
    fn device_carried(&self, merge: ObjectId, p: &[u32]) -> bool {
        p.len() == 1
            && self.resident.as_ref().is_some_and(|r| r.carries.iter().any(|c| c.merge == merge && c.k == p[0]))
    }

    /// Every object by the bytes its views lead to (`roots` without the
    /// in-place hops), built once the views are.
    fn index_roots(&mut self) {
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
    fn sharing(&self, root: &Root) -> Vec<ObjectId> {
        let mut fam: Vec<ObjectId> = self.root_index.get(root).cloned().unwrap_or_default();
        let mut set: std::collections::HashSet<ObjectId> = fam.iter().copied().collect();
        let next = |set: &std::collections::HashSet<ObjectId>| {
            self.update_aliases.iter().find(|&(t, s)| set.contains(s) && !set.contains(&t)).map(|(t, _)| t)
        };
        while let Some(t) = next(&set) {
            for &o in std::iter::once(&t).chain(self.root_index.get(&Root::Obj(t)).into_iter().flatten()) {
                if set.insert(o) {
                    fam.push(o);
                }
            }
        }
        fam
    }

    /// `o` is read from a guard arm (which runs at its `Phi`, past its own
    /// position) — the `Update` `upd` reading it aside.
    pub(super) fn guard_read(&self, o: ObjectId, upd: Option<MorphismId>) -> bool {
        let src = upd.map(|u| self.ir.morphism(u).expect("m").source);
        self.ir.out_edges(o).iter().any(|&e| {
            self.gated.contains_key(e) && Some(e) != upd && Some(self.ir.morphism(e).expect("m").target) != src
        })
    }

    /// The last position of the trip at which a buffer the state position `p`
    /// may be is read: over every object of the body whose bytes it may be,
    /// views included (`LastUsePlan::view_last_use`). For the `Update` `upd`:
    /// only the objects made before it. A whole-state or a guard-arm read
    /// reads any time.
    fn state_death(&self, lp: &Loop, p: &[u32], upd: Option<(MorphismId, u32)>) -> u32 {
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
    fn takeable(&self, lp: &Loop, p: &[u32], a: ObjectId) -> bool {
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
    fn state_owner(&self, lp: &Loop, p: &[u32]) -> Option<bool> {
        let (merge, next) = (lp.plan.merge, |q: &[u32]| self.roots(lp.plan.back_route, &ext(&[0], q), false));
        let others: Vec<Root> =
            array_paths(&self.obj_ty(merge)).iter().filter(|q| q.as_slice() != p).flat_map(|q| next(q)).collect();
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
    pub(super) fn par_safe(&self, arr: ObjectId, m: MorphismId) -> bool {
        let Some(par) = &self.par else { return true };
        let Some(&tw) = par.task.get(m) else { return false };
        let src = self.ir.morphism(m).expect("m").source;
        let (mut seen, mut stack): (SecondaryMap<ObjectId, ()>, _) = (SecondaryMap::new(), vec![arr]);
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
                let loop_op = matches!(mo.op, Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit);
                let ordered = par.task.get(e).map_or(loop_op, |&tr| tr == tw || par.before[tr][tw]);
                if self.lup.position(e) < self.lup.position(m) && !ordered {
                    return false;
                }
                if matches!(mo.op, Operation::Pair { .. } | Operation::Proj { .. } | Operation::Phi)
                    || mo.op == Operation::LoopEnter
                {
                    stack.push(mo.target);
                }
            }
        }
        true
    }

    /// The `Update` `m` (at position `at`) may write its array `arr` in place:
    /// the bytes are one buffer this function owns — and, inside a loop, the
    /// trip makes (else its next trip reads them) — and no view of them is
    /// read after `at` (`LastUsePlan::view_dead_after`); or a loop state's
    /// buffer the loop owns, no other view of which the trip reads again.
    pub(super) fn writes_in_place(&self, m: MorphismId, arr: ObjectId, at: u32) -> bool {
        let loops: Vec<&Loop> = self.loops.iter().filter(|lp| lp.body.contains(&m)).collect();
        match &self.roots(arr, &[], true)[..] {
            [Root::Obj(_)] => {
                let [Root::Obj(c)] = self.roots(arr, &[], false)[..] else { return false };
                self.ir.object(c).expect("object").kind == ObjectKind::Temporary
                    && !self.host_free(c)
                    && loops.iter().all(|lp| self.made_in(lp, c))
                    && !self
                        .sharing(&Root::Obj(c))
                        .into_iter()
                        .any(|o| self.born(o) < at && self.guard_read(o, Some(m)))
                    && self.lup.view_dead_after(c, at)
            }
            [Root::Merge(merge, p)] => loops.iter().find(|lp| lp.plan.merge == *merge).is_some_and(|lp| {
                self.state_owner(lp, p).is_some() && self.state_death(lp, p, Some((m, at))) <= at
            }),
            _ => false,
        }
    }

    /// **S65 rule R's buffer assignment for a loop's state**, after the
    /// in-place `Update`s: the `LastUsePlan` interval coloring across the back
    /// edge. The next state's bytes `r`, first written at position `W` of the
    /// trip, replace the state the trip reads until `D` ([`Self::state_death`]):
    /// `D < W` — one buffer; else two (ping-pong; one more per extra state
    /// position the bytes become). A state the trip writes in place lives in
    /// the loop's own buffer: the init's when it dies at the entry, else one
    /// copy made there (the one copy the facts require).
    fn plan_loop_buffers(&mut self) {
        let mut out = Vec::new();
        for lp in &self.loops {
            let merge = lp.plan.merge;
            let paths: Vec<Vec<u32>> = array_paths(&self.obj_ty(merge))
                .into_iter()
                .filter(|p| self.is_view(merge, p) && !self.device_carried(merge, p))
                .collect();
            let next = |q: &[u32]| self.roots(lp.plan.back_route, &ext(&[0], q), true);
            let written = |p: &[u32]| {
                self.sharing(&Root::Merge(merge, p.to_vec())).iter().any(|&t| self.update_aliases.contains_key(t))
            };
            for p in paths.iter().filter(|p| written(p) && self.state_owner(lp, p) == Some(false)) {
                out.push(LoopBuf::Own { merge, path: p.clone() });
            }
            let mut done: Vec<ObjectId> = Vec::new();
            for r in paths.iter().flat_map(|p| next(p)) {
                let Root::Obj(r) = r else { continue };
                if done.contains(&r) || !self.made_in(lp, r) || self.host_free(r) {
                    continue;
                }
                done.push(r);
                let at: Vec<Vec<u32>> = paths.iter().filter(|q| next(q).contains(&Root::Obj(r))).cloned().collect();
                let unit = self.resident.as_ref().and_then(|p| p.unit_writer(r));
                let w = self.ir.in_edges(r).iter().copied().chain(resident::grouped_writer(&self.groups, r)).chain(unit);
                let w = w.filter_map(|m| self.lup.position(m)).min().unwrap_or(0);
                if at.iter().map(|q| self.state_death(lp, q, None)).max().unwrap_or(0) < w {
                    continue; // one buffer: the state is dead before r is written
                }
                let take = match &at[..] {
                    [q] if !written(q) => match self.roots(lp.plan.init, q, false)[..] {
                        [Root::Obj(a)]
                            if self.takeable(lp, q, a) && lower_ty(&self.obj_ty(a)) == lower_ty(&self.obj_ty(r)) =>
                        {
                            Some(a)
                        }
                        _ => None,
                    },
                    _ => None,
                };
                out.push(LoopBuf::Colors { merge, root: r, at, take });
            }
        }
        self.bufs = out;
    }

    /// The address of the pointer cell of state position `p` of `merge`.
    fn state_cell(&mut self, merge: ObjectId, p: &[u32]) -> String {
        let Some((&k0, rest)) = p.split_first() else {
            return self.slot(merge).expect("merge slot");
        };
        let (_, mut cell) = self.component_ptr(merge, k0).expect("state field");
        let mut ty = self.obj_ty(merge).component_ty(k0).cloned().expect("state ty");
        let mut paths = sub_paths(self.views.get(merge).map_or(&[][..], |v| v), k0);
        for &k in rest {
            let llt = lower_ptr_paths(&ty, &paths).expect("state lowers");
            cell = self.field_ptr(&cell, &ty, &llt, k).expect("state field");
            (paths, ty) = (sub_paths(&paths, k), ty.component_ty(k).cloned().expect("state ty"));
        }
        cell
    }

    fn state_ty(&self, merge: ObjectId, p: &[u32]) -> Ty {
        p.iter().fold(self.obj_ty(merge), |t, &k| t.component_ty(k).cloned().expect("state ty"))
    }

    /// A buffer a loop's state may point at: a task's frame dies when it
    /// returns while the state's views (in `%Frame`) are read after, so there
    /// an arena block (the entry releases it with the rest).
    fn state_alloc(&mut self, llt: &str) -> String {
        let name = format!("%ls{}", self.fresh());
        if matches!(self.guard_flavor, GuardFlavor::Task) {
            let (bytes, align) = (llt_bytes(llt), llt_align(llt));
            self.allocas.push_str(&format!("  {name} = call ptr @mapal_rt_alloc(i64 {bytes}, i64 {align})\n"));
        } else {
            self.entry_alloc(&name, llt, None);
        }
        name
    }

    /// Loop entry, after the init's pointers became the state's: a state the
    /// trip writes in place gets the loop's own copy of the init.
    pub(crate) fn loop_enter_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs.clone() {
            let LoopBuf::Own { merge: m, path } = b else { continue };
            if m != merge {
                continue;
            }
            let llt = lower_ty(&self.state_ty(merge, &path)).expect("state lowers");
            let (cell, src) = (self.state_cell(merge, &path), self.tmp());
            self.line(format!("{src} = load ptr, ptr {cell}"));
            let own = self.state_alloc(&llt);
            self.emit_memcpy(&own, &src, &llt);
            self.line(format!("store ptr {own}, ptr {cell}"));
        }
    }

    /// Each trip's first instructions: a colored next state's buffer for this
    /// trip — the first of its slot and its spares no state position points at.
    pub(crate) fn loop_trip_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs.clone() {
            let LoopBuf::Colors { merge: m, root, at, take } = b else { continue };
            if m != merge {
                continue;
            }
            let llt = lower_ty(&self.obj_ty(root)).expect("root lowers");
            let mut bufs = vec![self.slot(root).expect("root slot")];
            for i in 0..at.len() {
                let spare = match take.filter(|_| i == 0) {
                    Some(a) => self.slot(a).expect("taken slot"),
                    None => self.state_alloc(&llt),
                };
                bufs.push(spare);
            }
            let mut curs = Vec::new();
            for q in &at {
                let (cell, c) = (self.state_cell(merge, q), self.tmp());
                self.line(format!("{c} = load ptr, ptr {cell}"));
                curs.push(c);
            }
            let mut w = bufs.last().expect("a spare").clone();
            for buf in bufs[..bufs.len() - 1].iter().rev() {
                let mut busy = "false".to_string();
                for c in &curs {
                    let (e, o) = (self.tmp(), self.tmp());
                    self.line(format!("{e} = icmp eq ptr {c}, {buf}"));
                    self.line(format!("{o} = or i1 {busy}, {e}"));
                    busy = o;
                }
                let s = self.tmp();
                self.line(format!("{s} = select i1 {busy}, ptr {w}, ptr {buf}"));
                w = s;
            }
            for (o, _) in self.ir.objects() {
                if self.alias_root(o) == root {
                    self.slots.insert(o, w.clone());
                }
            }
        }
    }

    /// The object whose storage `o` shares through in-place `Update`s.
    fn alias_root(&self, mut o: ObjectId) -> ObjectId {
        while let Some(&s) = self.update_aliases.get(o) {
            o = s;
        }
        o
    }

    /// The `carried=` field of a loop's `; mapal.move` line: per array position
    /// of the state, its bytes and how the trip carries it — `value` (inline in
    /// the products: the OFF arm), `device`, `inplace` (the state's buffer, the
    /// trip writing it), `two` (colored) or `one` (the next state's).
    pub(crate) fn carried_desc(&self, merge: ObjectId) -> String {
        let desc: Vec<String> = array_paths(&self.obj_ty(merge))
            .into_iter()
            .map(|p| {
                let bytes = lower_ty(&self.state_ty(merge, &p)).map_or(0, |l| llt_bytes(&l));
                let colored = self.bufs.iter().any(
                    |b| matches!(b, LoopBuf::Colors { merge: m, at, .. } if *m == merge && at.contains(&p)),
                );
                let state = Root::Merge(merge, p.clone());
                let written = self.sharing(&state).iter().any(|&t| self.update_aliases.contains_key(t));
                let mode = match () {
                    _ if self.device_carried(merge, &p) => "device",
                    _ if !self.is_view(merge, &p) => "value",
                    _ if colored => "two",
                    _ if written => "inplace",
                    _ => "one",
                };
                let at: Vec<String> = p.iter().map(u32::to_string).collect();
                format!("{}:{bytes}:{mode}", if at.is_empty() { "-".into() } else { at.join(".") })
            })
            .collect();
        if desc.is_empty() { "-".into() } else { desc.join(",") }
    }
}
