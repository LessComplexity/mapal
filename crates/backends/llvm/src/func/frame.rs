//! storage preparation, elided arrays, local slots, the `%Frame` layout
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

mod loop_buffers;
mod views;

impl<'a> FnEmit<'a> {
    /// Configure by-ref inputs and Update slot aliases. Returns the lowered
    /// incoming argument type.
    /// The call ABI of this fn's input and output (by-ref input positions,
    /// `ptr %out`): a function of its signature alone, set before anything
    /// lays out products (`plan_views` pins the input's pointers).
    pub(super) fn init_abi(&mut self) {
        let fd = self.ir.func(self.f).expect("func resolves");
        let in_ty = self.obj_ty(fd.input);
        let (bk, btext, bpaths) = match fd.kind {
            FuncKind::MapBody | FuncKind::FoldBody => {
                let k = self.body_captures();
                (k, lower_body_input_ty(&in_ty, k), capture_paths(&in_ty, k))
            }
            FuncKind::Named => (u32::MAX, lower_named_input_ty(&in_ty), array_paths(&in_ty)),
        };
        self.out_ptr = fd.kind == FuncKind::Named
            && self.f != self.ir.entry()
            && out_by_ptr(&self.obj_ty(fd.output));
        if let Some(text) = btext {
            self.byref_paths = bpaths;
            self.byref = Some((fd.input, bk, text));
        }
    }

    pub(super) fn prepare_storage(&mut self) -> Option<String> {
        self.init_abi();
        let fd = self.ir.func(self.f).expect("func resolves");
        let in_ty = self.obj_ty(fd.input);
        if let Some((_, bk, _)) = self.byref.clone() {
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
        let exits: Vec<ObjectId> = self
            .resident
            .iter()
            .flat_map(|p| p.carries.iter().filter_map(|c| c.exit))
            .collect();
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
        // ADR-0045: a read's `(token, array)` pair holds no bytes of its own —
        // the file lands in the array it yields (`emit_read`).
        for &m in &ms {
            let mo = self.ir.morphism(m).expect("morphism resolves");
            if mo.op == Operation::Read {
                self.elided_arrays.insert(mo.target, ());
            }
        }
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
        let output = self.ir.func(self.f).expect("func resolves").output;
        for (id, kind, ty) in self.owned_objects() {
            if kind == ObjectKind::Constant {
                continue;
            }
            if self.out_ptr && id == output {
                // structure: the caller's storage (`ptr %out`).
                self.slots.insert(id, "%out".to_string());
                ord += 1;
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
    Own {
        merge: ObjectId,
        path: Vec<u32>,
    },
    Colors {
        merge: ObjectId,
        root: ObjectId,
        at: Vec<Vec<u32>>,
        take: Option<ObjectId>,
    },
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
