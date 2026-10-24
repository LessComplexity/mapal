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

        // S65 `by-ref`: forwarding moves pointers (views), so an `Update` may
        // write a view's bytes in place, and the loop state's buffers follow.
        self.plan_views();
        if let Some(p) = &self.resident {
            // A device carry's exit with host bytes holds its own (the host's
            // state has none to point at): the exit copies it out.
            let own: Vec<ObjectId> = p.carries.iter().filter_map(|c| c.exit).collect();
            for x in own {
                if !self.host_free(x) {
                    self.ptr_resident.remove(x);
                }
            }
        }
        if self.by_ref {
            self.index_roots();
        }
        // In-place `Update`s, to a fixed point: a chain's next link writes in
        // place once its source does (its bytes are then the root's).
        let ms = fd.morphisms.clone();
        loop {
            let mut grew = false;
            for &m in &ms {
                let target = self.ir.morphism(m).expect("morphism resolves").target;
                if self.update_aliases.contains_key(target) {
                    continue;
                }
                if let Some(source) = self.update_in_place_source(m) {
                    self.elided_updates.insert(target, ());
                    self.update_aliases.insert(target, source);
                    if self.ptr_resident.contains_key(source) {
                        self.ptr_resident.insert(target, ());
                    }
                    grew = true;
                }
            }
            if !grew {
                break;
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

/// One loop state position's host buffers ([`FnEmit::plan_loop_buffers`]).
#[derive(Clone, Debug)]
pub(crate) struct LoopBuf {
    pub merge: ObjectId,
    /// The array's position in the merge (`[]`: the merge is the array).
    pub path: Vec<u32>,
    pub kind: BufKind,
}

#[derive(Clone, Debug)]
pub(crate) enum BufKind {
    /// The trip writes the state in place, and the init's bytes are read
    /// elsewhere or are not this function's: copied once, at the entry, into
    /// the loop's own buffer.
    Own,
    /// `root`, the next state's bytes, is written while the state it replaces
    /// is still read: each trip it takes the first of its slot and the spares
    /// (one per state position `at` its bytes become) that no such position
    /// points at. `take`: the init's buffer, dead at the entry, is the first
    /// spare.
    Colors {
        root: ObjectId,
        at: Vec<Vec<u32>>,
        take: Option<ObjectId>,
    },
}

/// A canonical loop of the function and its body's morphisms.
#[derive(Clone, Debug)]
pub(crate) struct Loop {
    pub plan: mapal_ir::LoopPlan,
    pub body: std::collections::HashSet<MorphismId>,
}

/// Where the bytes behind a view live ([`FnEmit::roots`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Root {
    /// An object's own bytes.
    Obj(ObjectId),
    /// A loop state position: the init's bytes, then each trip's next state's.
    Merge(ObjectId, Vec<u32>),
    /// Bytes this function does not hold (a borrowed input's).
    Other,
}

/// The parallel entry's task order: which task runs each morphism, and which
/// tasks the plan's dependencies order before which (transitively). An
/// in-place write must follow every read of the bytes it overwrites —
/// positions order one task, these order tasks.
#[derive(Debug)]
pub(crate) struct ParOrder {
    task: SecondaryMap<MorphismId, usize>,
    before: Vec<Vec<bool>>,
}

impl ParOrder {
    pub(crate) fn new(plan: &PathPlan) -> Self {
        let n = plan.tasks.len();
        let mut task = SecondaryMap::new();
        let mut before = vec![vec![false; n]; n];
        for (t, tk) in plan.tasks.iter().enumerate() {
            match &tk.kind {
                TaskKind::Split { site, .. } => {
                    task.insert(*site, t);
                }
                TaskKind::Seq { morphisms } => {
                    for &m in morphisms {
                        task.insert(m, t);
                    }
                }
            }
            for &d in &tk.deps {
                before[d][t] = true;
            }
        }
        for k in 0..n {
            for i in 0..n {
                if before[i][k] {
                    for j in 0..n {
                        if before[k][j] {
                            before[i][j] = true;
                        }
                    }
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
    /// move its address: a product's array position (component path) holds a
    /// pointer unless a consumer reads the product's value whole (`Output`, a
    /// by-value body input, …), and a bare array a forwarding morphism makes is
    /// a view (`ptr_resident`). Positions one whole-value copy joins (a loop's
    /// products, nested products, `Phi` arms) are decided together. A pure
    /// function of the IR and the rung — the residency plan and every emitter
    /// of the function (the host and its tasks) lay the products out alike.
    pub(super) fn plan_views(&mut self) {
        if self.views_done {
            return;
        }
        self.views_done = true;
        self.by_ref = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).by_ref;
        if !self.by_ref {
            return;
        }
        for scc in self.ir.loop_structure(self.f) {
            for &merge in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, merge) {
                    let body = plan.decide_order.iter().chain(&plan.advance_order).copied().collect();
                    self.loops.push(Loop { plan, body });
                }
            }
        }
        let fd = self.ir.func(self.f).expect("func resolves");
        let (input, output) = (fd.input, fd.output);
        let mut cand: std::collections::HashSet<Pos> = std::collections::HashSet::new();
        for (o, kind, ty) in self.owned_objects() {
            if matches!(kind, ObjectKind::Constant | ObjectKind::Parameter | ObjectKind::Return)
                || o == input
                || o == output
            {
                continue;
            }
            let op = self.ir.in_edges(o).first().map(|&m| self.ir.morphism(m).expect("m").op);
            let fwd = kind == ObjectKind::LoopMerge
                || matches!(
                    op,
                    Some(Operation::Pair { .. } | Operation::Proj { .. } | Operation::Phi | Operation::LoopExit)
                );
            for p in array_paths(&ty).into_iter().filter(|_| fwd) {
                // A bare array `Pair`s build is a literal: its own bytes.
                if !(p.is_empty() && matches!(op, Some(Operation::Pair { .. }))) {
                    cand.insert((o, p));
                }
            }
        }
        // `eq`: one whole-value copy moves both, so both are pointers or
        // neither. `need`: a view of a field needs the field to be a pointer.
        let (mut eq, mut need): (Vec<(Pos, Pos)>, Vec<(Pos, Pos)>) = (Vec::new(), Vec::new());
        for (_, mo) in self.ir.morphisms() {
            if self.ir.try_owner(mo.source) != Some(self.f) {
                continue;
            }
            let (s, t) = (mo.source, mo.target);
            let pres: &[&[u32]] = match mo.op {
                Operation::Pair { slot, .. } => {
                    for q in array_paths(&self.obj_ty(s)).into_iter().filter(|q| !q.is_empty()) {
                        eq.push(((t, ext(&[slot], &q)), (s, q)));
                    }
                    &[]
                }
                Operation::Proj { index } => {
                    for q in array_paths(&self.obj_ty(t)) {
                        let src = (s, ext(&[index], &q));
                        if q.is_empty() {
                            need.push(((t, q), src));
                        } else {
                            eq.push(((t, q), src));
                        }
                    }
                    &[]
                }
                Operation::LoopEnter => &[&[]],
                Operation::LoopBack | Operation::LoopExit => &[&[0]],
                Operation::Phi => &[&[0], &[1]],
                _ => &[],
            };
            for q in array_paths(&self.obj_ty(t)).into_iter().filter(|q| !q.is_empty()) {
                for pre in pres {
                    eq.push(((t, q.clone()), (s, ext(pre, &q))));
                }
            }
            // A consumer reading a product position as anything but an address
            // (or forwarding it) needs the bytes there.
            for p in array_paths(&self.obj_ty(s)).into_iter().filter(|p| !p.is_empty()) {
                let (k, one) = (p[0], p.len() == 1);
                let address = match mo.op {
                    Operation::Pair { .. }
                    | Operation::Proj { .. }
                    | Operation::Phi
                    | Operation::LoopEnter
                    | Operation::LoopBack
                    | Operation::LoopExit => true,
                    Operation::Index | Operation::Update => one && k == 0,
                    Operation::Zip => one && k <= 1,
                    Operation::Map { captures, .. } => one && k <= captures,
                    Operation::Fold { captures, .. } => one && (k < captures || k == captures + 1),
                    Operation::Call(_) => one,
                    _ => false,
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
                if cand.contains(v) && !cand.contains(n) {
                    changed |= cand.remove(v);
                }
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
            } else if let Some(v) = self.views.get_mut(o) {
                v.push(p);
            } else {
                self.views.insert(o, vec![p]);
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
        let mut out: Vec<Root> = Vec::new();
        let mut stack = vec![(o, p.to_vec())];
        let mut steps = 0;
        while let Some((o, p)) = stack.pop() {
            steps += 1;
            let r = if steps > 4096 {
                Some(Root::Other)
            } else if alias
                && p.is_empty()
                && let Some(&s) = self.update_aliases.get(o)
            {
                stack.push((s, p));
                None
            } else if !self.is_view(o, &p) {
                Some(if p.is_empty() { Root::Obj(o) } else { Root::Other })
            } else if self.ir.object(o).expect("object resolves").kind == ObjectKind::LoopMerge {
                Some(Root::Merge(o, p))
            } else {
                match self.ir.in_edges(o).first().map(|&m| self.ir.morphism(m).expect("m")) {
                    Some(mo) => match mo.op {
                        Operation::Pair { .. } => match self.pair_source(o, p[0]) {
                            Some(s) => {
                                stack.push((s, p[1..].to_vec()));
                                None
                            }
                            None => Some(Root::Other),
                        },
                        Operation::Proj { index } => {
                            stack.push((mo.source, ext(&[index], &p)));
                            None
                        }
                        Operation::LoopExit => {
                            stack.push((mo.source, ext(&[0], &p)));
                            None
                        }
                        Operation::Phi => {
                            stack.push((mo.source, ext(&[0], &p)));
                            stack.push((mo.source, ext(&[1], &p)));
                            None
                        }
                        // A view written in place, seen as its own bytes.
                        Operation::Update => Some(Root::Obj(o)),
                        _ => Some(Root::Other),
                    },
                    None => Some(Root::Other),
                }
            };
            if let Some(r) = r
                && !out.contains(&r)
            {
                out.push(r);
            }
        }
        out
    }

    /// `o` is written in the trip (its producer is in the loop body).
    fn made_in(&self, lp: &Loop, o: ObjectId) -> bool {
        self.ir.in_edges(o).iter().any(|m| lp.body.contains(m))
    }

    /// Where `o` is complete: its last producing edge (a product's last `Pair`).
    fn born(&self, o: ObjectId) -> u32 {
        self.ir.in_edges(o).iter().filter_map(|&m| self.lup.position(m)).max().unwrap_or(0)
    }

    /// The loop state position is carried on the device (no host bytes).
    fn device_carried(&self, merge: ObjectId, p: &[u32]) -> bool {
        p.len() == 1
            && self
                .resident
                .as_ref()
                .is_some_and(|r| r.carries.iter().any(|c| c.merge == merge && c.k == p[0]))
    }

    /// Every object, by the bytes its views lead to (`roots` without the
    /// in-place hops) — built once the views are, before any `Update` is.
    fn index_roots(&mut self) {
        let mut idx: std::collections::HashMap<Root, Vec<ObjectId>> = Default::default();
        for (o, _, ty) in self.owned_objects() {
            let mut rs: Vec<Root> = Vec::new();
            if !self.ptr_resident.contains_key(o) && matches!(ty, Ty::Array { .. }) {
                rs.push(Root::Obj(o));
            }
            let paths = if self.ptr_resident.contains_key(o) {
                vec![Vec::new()]
            } else {
                self.views.get(o).cloned().unwrap_or_default()
            };
            for q in paths {
                for r in self.roots(o, &q, false) {
                    if !rs.contains(&r) {
                        rs.push(r);
                    }
                }
            }
            for r in rs {
                idx.entry(r).or_default().push(o);
            }
        }
        self.root_index = idx;
    }

    /// The objects whose bytes may be `root`'s, the `Update`s writing in
    /// place into it and their views included.
    fn sharing(&self, root: &Root) -> Vec<ObjectId> {
        let mut fam: Vec<ObjectId> = self.root_index.get(root).cloned().unwrap_or_default();
        let mut set: std::collections::HashSet<ObjectId> = fam.iter().copied().collect();
        loop {
            let mut grew = false;
            for (t, &s) in self.update_aliases.iter() {
                if set.contains(&s) && !set.contains(&t) {
                    set.insert(t);
                    fam.push(t);
                    for &o in self.root_index.get(&Root::Obj(t)).into_iter().flatten() {
                        if set.insert(o) {
                            fam.push(o);
                        }
                    }
                    grew = true;
                }
            }
            if !grew {
                return fam;
            }
        }
    }

    /// `o` is read from a guard arm (which runs at its `Phi`, past its own
    /// position) — the `Update` `upd` reading it aside.
    fn guard_read(&self, o: ObjectId, upd: Option<MorphismId>) -> bool {
        let src = upd.map(|u| self.ir.morphism(u).expect("m").source);
        self.ir.out_edges(o).iter().any(|&e| {
            self.gated.contains_key(e)
                && Some(e) != upd
                && Some(self.ir.morphism(e).expect("m").target) != src
        })
    }

    /// The last position of the trip at which a buffer the state position
    /// `(merge, p)` may be is read: over every object of the body whose bytes
    /// it may be, with every view of them (`LastUsePlan::view_last_use`).
    /// `upd`: for that `Update`, only the objects made before it. A
    /// whole-merge use or a guard-arm read reads any time: `u32::MAX`.
    fn state_death(&self, lp: &Loop, p: &[u32], upd: Option<(MorphismId, u32)>) -> u32 {
        let merge = lp.plan.merge;
        let whole = self.ir.out_edges(merge).iter().any(|&m| {
            !matches!(self.ir.morphism(m).expect("m").op, Operation::Proj { .. })
        });
        if whole {
            return u32::MAX;
        }
        let mut d = 0;
        for o in self.sharing(&Root::Merge(merge, p.to_vec())) {
            if !self.made_in(lp, o) || upd.is_some_and(|(_, b)| self.born(o) >= b) {
                continue;
            }
            if self.guard_read(o, upd.map(|(u, _)| u)) {
                return u32::MAX;
            }
            d = d.max(self.lup.view_last_use(o).unwrap_or(0));
        }
        d
    }

    /// The init's array `a` (at position `p` of the init) may become the
    /// loop's buffer: this function's own bytes, read by nothing but the loop
    /// once the loop starts, every view of them included.
    fn takeable(&self, lp: &Loop, p: &[u32], a: ObjectId) -> bool {
        let plan = &lp.plan;
        let obj = self.ir.object(a).expect("object resolves");
        let Some(&enter) = self.ir.in_edges(plan.merge).iter().find(|&&m| {
            self.ir.morphism(m).expect("m").op == Operation::LoopEnter
        }) else {
            return false;
        };
        let at = self.lup.position(enter).unwrap_or(0);
        if obj.kind != ObjectKind::Temporary
            || self.ir.try_owner(a) != Some(self.f)
            || self.ptr_resident.contains_key(a)
            || self.host_free(a)
            || self.elided_arrays.contains_key(a)
        {
            return false;
        }
        // The init, its products down to `a`, and `a`.
        let mut path = vec![plan.init];
        for &k in p {
            match self.pair_source(*path.last().expect("init"), k) {
                Some(s) => path.push(s),
                None => return false,
            }
        }
        if *path.last().expect("a") != a {
            return false;
        }
        for (i, &o) in path.iter().enumerate() {
            for &e in self.ir.out_edges(o) {
                let mo = self.ir.morphism(e).expect("m");
                let into = if i == 0 {
                    mo.op == Operation::LoopEnter
                } else {
                    matches!(mo.op, Operation::Pair { .. }) && mo.target == path[i - 1]
                };
                if into {
                    continue;
                }
                let viewed = matches!(mo.op, Operation::Pair { .. } | Operation::Phi | Operation::Proj { .. })
                    && (self.lup.escapes(mo.target)
                        || self.lup.view_death(mo.target).is_some_and(|d| d > at));
                if i + 1 < path.len() || self.lup.position(e).is_none_or(|q| q > at) || viewed {
                    return false;
                }
            }
        }
        true
    }

    /// The loop owns state position `p`'s buffer for an in-place write in the
    /// trip: every buffer the next state may be is the state's own or written
    /// in the trip (and no other position's). `Some(true)`: the init's buffer
    /// is the loop's (it dies at the entry); `Some(false)`: the entry copies
    /// the init into one.
    fn state_owner(&self, lp: &Loop, p: &[u32]) -> Option<bool> {
        let plan = &lp.plan;
        if self.device_carried(plan.merge, p) {
            return None;
        }
        let next = |q: &[u32]| self.roots(plan.back_route, &ext(&[0], q), false);
        let others: Vec<Root> = array_paths(&self.obj_ty(plan.merge))
            .iter()
            .filter(|q| q.as_slice() != p)
            .flat_map(|q| next(q))
            .collect();
        for r in next(p) {
            let mine = match &r {
                Root::Merge(m, q) => *m == plan.merge && q == p,
                Root::Obj(o) => self.made_in(lp, *o) && !others.contains(&r),
                Root::Other => false,
            };
            if !mine {
                return None;
            }
        }
        Some(matches!(self.roots(plan.init, p, false)[..], [Root::Obj(a)] if self.takeable(lp, p, a)))
    }

    /// In the parallel entry, every read of `arr`'s bytes before the write
    /// `m` (through the morphisms forwarding them) runs in `m`'s task or one
    /// the plan orders before it — positions order one task only (the S64
    /// arm wrote `out` in place while another task still read it).
    pub(super) fn par_safe(&self, arr: ObjectId, m: MorphismId) -> bool {
        let Some(par) = &self.par else {
            return true;
        };
        let Some(&tw) = par.task.get(m) else {
            return false;
        };
        let src = self.ir.morphism(m).expect("m").source;
        let mut seen: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut stack = vec![arr];
        while let Some(o) = stack.pop() {
            if seen.insert(o, ()).is_some() {
                continue;
            }
            for &e in self.ir.out_edges(o) {
                let mo = self.ir.morphism(e).expect("m");
                if e == m || mo.target == src {
                    continue;
                }
                // A loop's own edges run where its body does.
                let loop_op = matches!(
                    mo.op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                ) && par.task.get(e).is_none();
                // A read the positions put before the write (every read of
                // the old bytes — `dead_after`) must run before it.
                let old = self.lup.position(e) < self.lup.position(m);
                if old
                    && !loop_op
                    && !par.task.get(e).is_some_and(|&tr| tr == tw || par.before[tr][tw])
                {
                    return false;
                }
                // (A loop's exit object reads its final state, after the loop.)
                if matches!(
                    mo.op,
                    Operation::Pair { .. } | Operation::Proj { .. } | Operation::Phi | Operation::LoopEnter
                ) {
                    stack.push(mo.target);
                }
            }
        }
        true
    }

    /// The `Update` `m` (at position `at`) may write its array `arr` in place:
    /// the bytes are one buffer this function owns, and no view of them is
    /// read after `at` (`LastUsePlan::view_dead_after`) — nor, for a buffer
    /// made outside a loop running `m`, by its next trip. A loop state's
    /// buffer: the loop owns it and the trip reads no other view of it.
    pub(super) fn writes_in_place(&self, m: MorphismId, arr: ObjectId, at: u32) -> bool {
        let loops: Vec<&Loop> = self.loops.iter().filter(|lp| lp.body.contains(&m)).collect();
        match &self.roots(arr, &[], true)[..] {
            [Root::Obj(_)] => {
                let [Root::Obj(c)] = self.roots(arr, &[], false)[..] else {
                    return false;
                };
                let obj = self.ir.object(c).expect("object resolves");
                let gated_read = self
                    .sharing(&Root::Obj(c))
                    .into_iter()
                    .any(|o| self.born(o) < at && self.guard_read(o, Some(m)));
                obj.kind == ObjectKind::Temporary
                    && self.ir.try_owner(c) == Some(self.f)
                    && !self.host_free(c)
                    && !self.elided_arrays.contains_key(c)
                    && !gated_read
                    && loops.iter().all(|lp| self.made_in(lp, c))
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
    /// trip, replace the state the trip reads until `D`
    /// ([`Self::state_death`]): `D < W` — one buffer; else two, `r` taking
    /// whichever the state does not point at (ping-pong; one more per extra
    /// state position its bytes become). A state the trip writes in place
    /// lives in the loop's own buffer: the init's when it dies at the entry,
    /// else a copy made there (the one copy the facts require).
    fn plan_loop_buffers(&mut self) {
        if !self.by_ref {
            return;
        }
        let mut out = Vec::new();
        for lp in &self.loops {
            let (plan, merge) = (&lp.plan, lp.plan.merge);
            let paths: Vec<Vec<u32>> = array_paths(&self.obj_ty(merge))
                .into_iter()
                .filter(|p| self.is_view(merge, p) && !self.device_carried(merge, p))
                .collect();
            let next = |q: &[u32]| self.roots(plan.back_route, &ext(&[0], q), true);
            let mut written: Vec<Vec<u32>> = Vec::new();
            for p in &paths {
                let state = Root::Merge(merge, p.clone());
                if self.sharing(&state).iter().any(|t| self.update_aliases.contains_key(*t)) {
                    written.push(p.clone());
                    if self.state_owner(lp, p) == Some(false) {
                        out.push(LoopBuf {
                            merge,
                            path: p.clone(),
                            kind: BufKind::Own,
                        });
                    }
                }
            }
            let mut done: Vec<ObjectId> = Vec::new();
            for p in &paths {
                for r in next(p) {
                    let Root::Obj(r) = r else { continue };
                    if done.contains(&r) || !self.made_in(lp, r) || self.host_free(r) {
                        continue;
                    }
                    done.push(r);
                    let at: Vec<Vec<u32>> = paths
                        .iter()
                        .filter(|q| next(q).contains(&Root::Obj(r)))
                        .cloned()
                        .collect();
                    let w = self
                        .ir
                        .in_edges(r)
                        .iter()
                        .copied()
                        .chain(resident::grouped_writer(&self.groups, r))
                        .filter_map(|m| self.lup.position(m))
                        .min()
                        .unwrap_or(0);
                    let d = at.iter().map(|q| self.state_death(lp, q, None)).max().unwrap_or(0);
                    if d < w {
                        continue; // one buffer: the state is dead before r is written
                    }
                    let take = match (&at[..], written.contains(&at[0])) {
                        ([q], false) => match self.roots(plan.init, q, false)[..] {
                            [Root::Obj(a)]
                                if self.takeable(lp, q, a)
                                    && lower_ty(&self.obj_ty(a)) == lower_ty(&self.obj_ty(r)) =>
                            {
                                Some(a)
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    out.push(LoopBuf {
                        merge,
                        path: at[0].clone(),
                        kind: BufKind::Colors { root: r, at, take },
                    });
                }
            }
        }
        self.bufs = out;
    }

    /// The address of the pointer cell of state position `p` of `merge`.
    fn state_cell(&mut self, merge: ObjectId, p: &[u32]) -> String {
        if p.is_empty() {
            return self.slot(merge).expect("merge slot");
        }
        let (_, mut cell) = self.component_ptr(merge, p[0]).expect("state field");
        let mut ty = self.obj_ty(merge).component_ty(p[0]).cloned().expect("state ty");
        let mut paths = sub_paths(self.views.get(merge).map_or(&[][..], |v| v), p[0]);
        for &k in &p[1..] {
            let llt = lower_ptr_paths(&ty, &paths).expect("state lowers");
            cell = self.field_ptr(&cell, &ty, &llt, k).expect("state field");
            paths = sub_paths(&paths, k);
            ty = ty.component_ty(k).cloned().expect("state ty");
        }
        cell
    }

    fn bufs_of(&self, merge: ObjectId) -> Vec<LoopBuf> {
        self.bufs.iter().filter(|b| b.merge == merge).cloned().collect()
    }

    fn state_ty(&self, merge: ObjectId, p: &[u32]) -> Ty {
        let mut ty = self.obj_ty(merge);
        for &k in p {
            ty = ty.component_ty(k).cloned().expect("state ty");
        }
        ty
    }

    /// A buffer a loop's state may point at. A task's frame dies when it
    /// returns, while the state's views live in `%Frame` and are read after:
    /// an arena block there (the entry releases it with the rest).
    fn state_alloc(&mut self, name: &str, llt: &str) {
        if matches!(self.guard_flavor, GuardFlavor::Task) {
            let (bytes, align) = (llt_bytes(llt), llt_align(llt));
            self.allocas.push_str(&format!(
                "  {name} = call ptr @mapal_rt_alloc(i64 {bytes}, i64 {align})\n"
            ));
        } else {
            self.entry_alloc(name, llt, None);
        }
    }

    /// Loop entry, after the init's pointers are the state's: a state the
    /// trip writes in place gets the loop's own copy of the init.
    pub(crate) fn loop_enter_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs_of(merge) {
            if !matches!(b.kind, BufKind::Own) {
                continue;
            }
            let llt = lower_ty(&self.state_ty(merge, &b.path)).expect("state array lowers");
            let cell = self.state_cell(merge, &b.path);
            let src = self.tmp();
            self.line(format!("{src} = load ptr, ptr {cell}"));
            let own = format!("%lb{}", self.fresh());
            self.state_alloc(&own, &llt);
            self.emit_memcpy(&own, &src, &llt);
            self.line(format!("store ptr {own}, ptr {cell}"));
        }
    }

    /// Each trip's first instructions: a two-colored next state's buffer for
    /// this trip — the first of its slot and its spares no state position
    /// points at.
    pub(crate) fn loop_trip_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs_of(merge) {
            let BufKind::Colors { root, at, take } = b.kind else {
                continue;
            };
            let llt = lower_ty(&self.obj_ty(root)).expect("root lowers");
            let mut bufs = vec![self.slot(root).expect("root slot")];
            for i in 0..at.len() {
                match take.filter(|_| i == 0) {
                    Some(a) => bufs.push(self.slot(a).expect("taken slot")),
                    None => {
                        let s = format!("%ls{}", self.fresh());
                        self.state_alloc(&s, &llt);
                        bufs.push(s);
                    }
                }
            }
            let mut curs = Vec::new();
            for q in &at {
                let cell = self.state_cell(merge, q);
                let c = self.tmp();
                self.line(format!("{c} = load ptr, ptr {cell}"));
                curs.push(c);
            }
            let mut w = bufs.last().expect("a spare").clone();
            for buf in bufs[..bufs.len() - 1].iter().rev() {
                let mut busy = String::new();
                for c in &curs {
                    let e = self.tmp();
                    self.line(format!("{e} = icmp eq ptr {c}, {buf}"));
                    if busy.is_empty() {
                        busy = e;
                    } else {
                        let o = self.tmp();
                        self.line(format!("{o} = or i1 {busy}, {e}"));
                        busy = o;
                    }
                }
                let s = self.tmp();
                self.line(format!("{s} = select i1 {busy}, ptr {w}, ptr {buf}"));
                w = s;
            }
            let sharing: Vec<ObjectId> = self
                .ir
                .objects()
                .map(|(o, _)| o)
                .filter(|&o| self.alias_root(o) == root)
                .collect();
            for o in sharing {
                self.slots.insert(o, w.clone());
            }
        }
    }

    /// The object whose storage `o` shares through in-place `Update`s.
    pub(super) fn alias_root(&self, mut o: ObjectId) -> ObjectId {
        while let Some(&s) = self.update_aliases.get(o) {
            o = s;
        }
        o
    }

    /// The `carried=` field of a loop's `; mapal.move` line: each array
    /// position of the state, its bytes and how the trip carries it —
    /// `value` (inline in the products: the OFF arm), `device`, `inplace`
    /// (one buffer the trip writes), `one` or `two` (buffers per next state).
    pub(crate) fn carried_desc(&self, merge: ObjectId) -> String {
        let mut out = Vec::new();
        for p in array_paths(&self.obj_ty(merge)) {
            let bytes = lower_ty(&self.state_ty(merge, &p)).map_or(0, |l| llt_bytes(&l));
            let state = Root::Merge(merge, p.clone());
            let mode = if self.device_carried(merge, &p) {
                "device"
            } else if !self.is_view(merge, &p) {
                "value"
            } else if self.bufs.iter().any(|b| {
                b.merge == merge && matches!(&b.kind, BufKind::Colors { at, .. } if at.contains(&p))
            }) {
                "two"
            } else if self.sharing(&state).iter().any(|t| self.update_aliases.contains_key(*t)) {
                "inplace"
            } else {
                "one"
            };
            let at: Vec<String> = p.iter().map(u32::to_string).collect();
            let at = if at.is_empty() { "-".to_string() } else { at.join(".") };
            out.push(format!("{at}:{bytes}:{mode}"));
        }
        if out.is_empty() { "-".into() } else { out.join(",") }
    }
}
