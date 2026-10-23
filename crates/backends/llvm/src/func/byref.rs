//! S65 rule R for loop state (`by-ref`): an object's bytes exist where a
//! morphism observes them, and move only when an observer sits on the other
//! side.
//!
//! A loop forwards its state through products — the init, the merge, the
//! next state, the back route, the exit route — and reads it through `Proj`
//! views. None of those observes an array's bytes: they re-expose them. So a
//! carried array component is a `ptr` in every one of those products, its
//! views and its exit object hold the pointer, and the bytes stay in the one
//! buffer their producer wrote. Which buffer that is follows from the
//! `LastUsePlan` positions of the trip, never from the workload:
//!
//! | mode | fact | storage |
//! |---|---|---|
//! | `Device` | the component is a device carry (F3, `resident.rs`) | none on the host: the views, next and a kept exit are host-free |
//! | `Ptr` | every view dies before next's buffer is first written in the trip | next's own buffer; the back edge stores its address |
//! | `Owned` | next is an in-place `Update` chain over the view | one loop-owned buffer, written in place |
//! | `PingPong` | next's writer reads the view (`x = f(x)`) | next's buffer and a spare, swapped at the back edge |
//!
//! The loop owns a buffer by taking the init feeder's when the feeder dies at
//! the `LoopEnter` (no copy), else by one copy into a fresh one at entry (a
//! small array only: a fresh block in a worker task would be stack).
//! Anything else keeps the by-value products (always correct).

use super::*;

/// One array component of a loop state, carried by reference.
#[derive(Clone, Debug)]
pub(crate) struct Family {
    pub merge: ObjectId,
    pub k: u32,
    pub init: ObjectId,
    /// The init's array (`Pair` feeder of `init` at `k`).
    pub feeder: ObjectId,
    pub enter: MorphismId,
    /// `Proj{k}` views of the merge.
    pub views: Vec<ObjectId>,
    pub state: ObjectId,
    pub back_route: ObjectId,
    /// The next state's array (`Pair` feeder of `state` at `k`).
    pub next: ObjectId,
    /// The exit route and exit object, when the exit payload is a view.
    pub exit_route: Option<ObjectId>,
    pub exit: Option<ObjectId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Device,
    Ptr,
    Owned { take: bool },
    PingPong { take: bool },
}

impl Mode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Device => "device",
            Mode::Ptr => "ptr",
            Mode::Owned { .. } => "owned",
            Mode::PingPong { .. } => "pingpong",
        }
    }
}

/// A decided component and its emission names.
#[derive(Clone, Debug)]
pub(crate) struct CarryRef {
    pub fam: Family,
    pub mode: Mode,
    /// `PingPong`: the cells holding the buffer next is written to this trip,
    /// and the other one.
    pub cells: Option<(String, String)>,
}

enum Retry {
    /// No in-place write into this view (the `Update` copies instead).
    NoInplace(ObjectId),
    /// Keep this component by value.
    Drop,
}

impl<'a> FnEmit<'a> {
    /// The rung, as the host decides it (any profile).
    fn by_ref_on(&self) -> bool {
        gpu::host_rungs(self.profile, self.contract, self.gpu_arms).by_ref
    }

    /// Every loop-state array component whose products only forward it — a
    /// pure function of the IR and the rung, so every emitter of the function
    /// (the parallel host and its tasks) lays the products out alike.
    pub(crate) fn families(&self) -> Vec<Family> {
        let mut out = Vec::new();
        if !self.by_ref_on() {
            return out;
        }
        let only = |o: ObjectId, op: fn(Operation) -> bool| {
            self.ir
                .out_edges(o)
                .iter()
                .all(|&m| op(self.ir.morphism(m).expect("morphism resolves").op))
        };
        for scc in self.ir.loop_structure(self.f) {
            for &merge in &scc.merges {
                let Some(plan) = self.ir.loop_plan(self.f, merge) else {
                    continue;
                };
                let Ty::Tuple(parts) = self.obj_ty(merge) else {
                    continue;
                };
                let Some(&enter) = self.ir.out_edges(plan.init).first() else {
                    continue;
                };
                let Some(state) = self.pair_source(plan.back_route, 0) else {
                    continue;
                };
                if !only(plan.init, |op| op == Operation::LoopEnter)
                    || !only(merge, |op| matches!(op, Operation::Proj { .. }))
                    || !only(plan.back_route, |op| op == Operation::LoopBack)
                    || self.ir.out_edges(state).len() != 1
                    || self.ir.in_edges(merge).len() != 2
                {
                    continue;
                }
                let body: Vec<ObjectId> = plan
                    .decide_order
                    .iter()
                    .chain(plan.advance_order.iter())
                    .map(|&m| self.ir.morphism(m).expect("morphism resolves").target)
                    .collect();
                for (k, ty) in parts.iter().enumerate() {
                    let k = k as u32;
                    if !matches!(ty, Ty::Array { .. }) {
                        continue;
                    }
                    let (Some(feeder), Some(next)) =
                        (self.pair_source(plan.init, k), self.pair_source(state, k))
                    else {
                        continue;
                    };
                    let views: Vec<ObjectId> = self
                        .ir
                        .out_edges(merge)
                        .iter()
                        .map(|&m| self.ir.morphism(m).expect("morphism resolves"))
                        .filter(|mo| mo.op == Operation::Proj { index: k })
                        .map(|mo| mo.target)
                        .collect();
                    let into_state = self
                        .ir
                        .out_edges(next)
                        .iter()
                        .filter(|&&m| self.ir.morphism(m).expect("m").target == state)
                        .count();
                    // next is written in the trip (or is an in-place chain
                    // over a view, which `carry_mode` reads off the aliases).
                    if into_state != 1 || views.contains(&next) || !body.contains(&next) {
                        continue;
                    }
                    let exit_route = self
                        .pair_source(plan.exit_route, 0)
                        .filter(|p| views.contains(p))
                        .filter(|_| {
                            plan.exits.len() == 1
                                && only(plan.exit_route, |op| op == Operation::LoopExit)
                        })
                        .map(|_| plan.exit_route);
                    let exit = exit_route
                        .map(|_| self.ir.morphism(plan.exits[0]).expect("exit").target);
                    out.push(Family {
                        merge,
                        k,
                        init: plan.init,
                        feeder,
                        enter,
                        views,
                        state,
                        back_route: plan.back_route,
                        next,
                        exit_route,
                        exit,
                    });
                }
            }
        }
        out
    }

    /// The object whose storage `o` shares through in-place `Update`s.
    pub(super) fn alias_root(&self, mut o: ObjectId) -> ObjectId {
        while let Some(&s) = self.update_aliases.get(o) {
            o = s;
        }
        o
    }

    /// The component is carried on the device (F3, `resident.rs`).
    fn device_carry(&self, f: &Family) -> bool {
        self.resident
            .as_ref()
            .is_some_and(|p| p.carries.iter().any(|c| c.merge == f.merge && c.k == f.k))
    }

    /// The loop may take the init feeder's buffer: nothing reads it after the
    /// `LoopEnter`, and it is this function's own.
    fn can_take(&self, f: &Family) -> bool {
        let a = f.feeder;
        let kind = self.ir.object(a).expect("object resolves").kind;
        kind != ObjectKind::Constant
            && kind != ObjectKind::Parameter
            && self.ir.try_owner(a) == Some(self.f)
            && !self.ptr_resident.contains_key(a)
            && !self.host_free(a)
            && self
                .lup
                .position(f.enter)
                .is_some_and(|p| self.lup.dead_after(a, p))
    }

    /// A fresh loop-owned buffer fits the stack.
    fn small(&self, f: &Family) -> bool {
        let ty = self.obj_ty(f.merge).component_ty(f.k).cloned().expect("carry ty");
        lower_ty(&ty).is_some_and(|l| llt_bytes(&l) < self.profile.heap_min_bytes)
    }

    fn carry_mode(&self, f: &Family) -> Result<Mode, Retry> {
        if self.device_carry(f) {
            return Ok(Mode::Device);
        }
        let own = |take| (take || self.small(f)).then_some(take);
        let root = self.alias_root(f.next);
        if f.views.contains(&root) {
            return own(self.can_take(f))
                .map(|take| Mode::Owned { take })
                .ok_or(Retry::NoInplace(root));
        }
        // An in-place write into the state that does not end at next would
        // land in a buffer a later trip reads.
        if let Some((t, _)) = self
            .update_aliases
            .iter()
            .find(|&(t, _)| f.views.contains(&self.alias_root(t)))
        {
            return Err(Retry::NoInplace(self.alias_root(t)));
        }
        if self.ptr_resident.contains_key(root) {
            return Err(Retry::Drop);
        }
        let write = self
            .ir
            .in_edges(root)
            .first()
            .and_then(|&p| self.lup.position(p))
            .ok_or(Retry::Drop)?;
        if f
            .views
            .iter()
            .all(|&v| self.lup.death(v).is_some_and(|d| d < write))
        {
            return Ok(Mode::Ptr);
        }
        own(self.can_take(f))
            .map(|take| Mode::PingPong { take })
            .ok_or(Retry::Drop)
    }

    /// Decide every carried component (`prepare_storage`), with the in-place
    /// `Update` admission that depends on it; then the product layouts.
    pub(super) fn plan_carry_refs(&mut self, ms: &[MorphismId]) {
        let base = self.ptr_resident.clone();
        let mut fams = self.families();
        let mut no_inplace: Vec<ObjectId> = Vec::new();
        let decided = loop {
            self.ptr_resident = base.clone();
            self.loop_owned = SecondaryMap::new();
            self.elided_updates = SecondaryMap::new();
            self.update_aliases = SecondaryMap::new();
            let output = self.ir.func(self.f).expect("func resolves").output;
            for f in &fams {
                for &v in &f.views {
                    self.ptr_resident.insert(v, ());
                    if f.views.len() == 1 && !no_inplace.contains(&v) && !self.device_carry(f) {
                        self.loop_owned.insert(v, ());
                    }
                }
                // A host-carried exit object takes the state's pointer (an
                // `Update` of it then copies: its bytes are the loop's).
                if let Some(x) = f.exit.filter(|&x| x != output && !self.device_carry(f)) {
                    self.ptr_resident.insert(x, ());
                }
            }
            // In-place admission, to a fixpoint: a chain's next link is a
            // pointer slot only once its source is.
            loop {
                let mut grew = false;
                for &m in ms {
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
                        if self.loop_owned.contains_key(source) {
                            self.loop_owned.insert(target, ());
                        }
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            let mut out = Vec::new();
            let mut retry = false;
            fams.retain(|f| match self.carry_mode(f) {
                Ok(mode) => {
                    out.push((f.clone(), mode));
                    true
                }
                Err(Retry::NoInplace(v)) => {
                    no_inplace.push(v);
                    retry = true;
                    true
                }
                Err(Retry::Drop) => {
                    retry = true;
                    false
                }
            });
            if !retry {
                break out;
            }
        };
        for (f, mode) in decided {
            // `Ptr` over an init that dies at the entry: next is written into
            // the init's buffer (every view of trip 1 has died before it is),
            // so the state has one buffer.
            if mode == Mode::Ptr && self.can_take(&f) {
                let root = self.alias_root(f.next);
                self.elided_updates.insert(root, ());
                self.update_aliases.insert(root, f.feeder);
            }
            let mut add = |o: ObjectId, path: Vec<u32>| match self.ref_paths.get_mut(o) {
                Some(v) => v.push(path),
                None => {
                    self.ref_paths.insert(o, vec![path]);
                }
            };
            add(f.init, vec![f.k]);
            add(f.merge, vec![f.k]);
            add(f.state, vec![f.k]);
            add(f.back_route, vec![0, f.k]);
            if let Some(e) = f.exit_route {
                add(e, vec![0]);
            }
            if let Some(x) = f.exit.filter(|_| mode == Mode::Device) {
                self.exit_by_device.insert(x, ());
            }
            let cells = matches!(mode, Mode::PingPong { .. }).then(|| {
                let (c, s) = (format!("%rc{}", self.fresh()), format!("%rs{}", self.fresh()));
                self.allocas
                    .push_str(&format!("  {c} = alloca ptr\n  {s} = alloca ptr\n"));
                (c, s)
            });
            self.carry_refs.push(CarryRef {
                fam: f,
                mode,
                cells,
            });
        }
    }

    fn refs_of(&self, merge: ObjectId) -> Vec<CarryRef> {
        self.carry_refs
            .iter()
            .filter(|c| c.fam.merge == merge)
            .cloned()
            .collect()
    }

    /// Loop entry, after the init product is stored in the merge: the loop's
    /// own buffers.
    pub(crate) fn carry_enter(&mut self, merge: ObjectId) {
        for c in self.refs_of(merge) {
            let f = &c.fam;
            let llt = lower_ty(&self.obj_ty(f.merge).component_ty(f.k).cloned().expect("ty"))
                .expect("carry lowers");
            let fresh = |fe: &mut Self| {
                let b = format!("%rb{}", fe.fresh());
                fe.entry_alloc(&b, &llt, None);
                b
            };
            match c.mode {
                Mode::Owned { take: false } => {
                    let b = fresh(self);
                    let a = self.array_operand_ptr(f.feeder, None).expect("init array");
                    self.emit_memcpy(&b, &a, &llt);
                    let (_, fp) = self.component_ptr(f.merge, f.k).expect("merge field");
                    self.line(format!("store ptr {b}, ptr {fp}"));
                }
                Mode::PingPong { take } => {
                    let spare = if take {
                        self.array_operand_ptr(f.feeder, None).expect("init array")
                    } else {
                        fresh(self)
                    };
                    let root = self.alias_root(f.next);
                    let own = self.slot(root).expect("next buffer");
                    let (cur, other) = c.cells.clone().expect("cells");
                    self.line(format!("store ptr {own}, ptr {cur}"));
                    self.line(format!("store ptr {spare}, ptr {other}"));
                }
                _ => {}
            }
        }
    }

    /// Each trip's first instructions: next's buffer for this trip.
    pub(crate) fn carry_header(&mut self, merge: ObjectId) {
        for c in self.refs_of(merge) {
            let Some((cur, _)) = c.cells else {
                continue;
            };
            let p = self.tmp();
            self.line(format!("{p} = load ptr, ptr {cur}"));
            let root = self.alias_root(c.fam.next);
            let sharing: Vec<ObjectId> = self
                .ir
                .objects()
                .map(|(o, _)| o)
                .filter(|&o| o == root || self.alias_root(o) == root)
                .collect();
            for o in sharing {
                self.slots.insert(o, p.clone());
            }
        }
    }

    /// The back edge, after the merge took the next state's pointer: the
    /// buffer it points at is read next trip, so the other one is written.
    pub(crate) fn carry_back(&mut self, merge: ObjectId) {
        for c in self.refs_of(merge) {
            let Some((cur, other)) = c.cells else {
                continue;
            };
            let (a, b) = (self.tmp(), self.tmp());
            self.line(format!("{a} = load ptr, ptr {cur}"));
            self.line(format!("{b} = load ptr, ptr {other}"));
            self.line(format!("store ptr {b}, ptr {cur}"));
            self.line(format!("store ptr {a}, ptr {other}"));
        }
    }

    /// The `carried=` field of a loop's `; mapal.move` line: each array
    /// component, its bytes and how it is carried (`value`: inline in the
    /// products, the OFF arm).
    pub(crate) fn carried_desc(&self, merge: ObjectId) -> String {
        let Ty::Tuple(parts) = self.obj_ty(merge) else {
            return "-".into();
        };
        let refs = self.refs_of(merge);
        let mut out = Vec::new();
        for (k, ty) in parts.iter().enumerate() {
            if !matches!(ty, Ty::Array { .. }) {
                continue;
            }
            let bytes = lower_ty(ty).map_or(0, |l| llt_bytes(&l));
            let mode = refs
                .iter()
                .find(|c| c.fam.k == k as u32)
                .map_or("value", |c| c.mode.name());
            out.push(format!("{k}:{bytes}:{mode}"));
        }
        if out.is_empty() {
            "-".into()
        } else {
            out.join(",")
        }
    }


    /// S65 movement intent, emitted as the loop header's first line: what the
    /// facts require to move per trip — the residency plan's own derivation
    /// (uploads of host-produced arrays a device site reads, `delta-h2d` rows,
    /// copy-outs of device outputs a host morphism reads, one-element reads),
    /// and how each carried array travels. `benches/gpu/evidence.py` counts
    /// what the host IR realizes against it. A loop with no array state off a
    /// GPU build moves nothing and gets no line.
    pub(crate) fn move_line(&self, plan: &mapal_ir::LoopPlan, header: &str) -> String {
        let arrays = matches!(self.obj_ty(plan.merge), Ty::Tuple(ref p)
            if p.iter().any(|t| matches!(t, Ty::Array { .. })));
        if !arrays && self.resident.is_none() {
            return String::new();
        }
        let bytes = |o: ObjectId| {
            lower_ty(&self.obj_ty(o)).map_or(0, |l| llt_bytes(&l))
        };
        let (mut h2d, mut d2h, mut retained) = (0u64, 0u64, 0usize);
        let body: Vec<MorphismId> = plan
            .decide_order
            .iter()
            .chain(plan.advance_order.iter())
            .copied()
            .filter(|&m| !self.gated.contains_key(m))
            .collect();
        if let Some(p) = &self.resident {
            for &m in &body {
                let morph = self.ir.morphism(m).expect("morphism resolves");
                let t = morph.target;
                if p.upload.contains_key(t) && !p.devgen.contains_key(m) {
                    h2d += match p.delta.get(t) {
                        Some(du) => {
                            let elem = self.obj_ty(t).component_ty(0).cloned().expect("elem");
                            du.stride * crate::profile::elem_bytes(&elem)
                        }
                        None => bytes(t),
                    };
                }
                if p.elem.contains_key(m) {
                    d2h += lower_ty(&self.obj_ty(t)).map_or(0, |l| llt_bytes(&l));
                }
                // The values a launch leaves in device memory, copied out
                // whole unless kept (`emit_site_resident`).
                let mut outs = Vec::new();
                if self.device.contains_key(m) {
                    match self.groups.get(m) {
                        Some(g) => {
                            if g.store_out {
                                outs.push(t);
                            }
                            outs.extend(g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out));
                        }
                        None => outs.push(t),
                    }
                } else if p.rowfold.contains_key(m) || p.cells.contains_key(m) {
                    outs.push(t);
                }
                for o in outs {
                    if !p.keep.contains_key(o) && !p.is_carry_next(o) {
                        d2h += bytes(o);
                    }
                }
                retained += self.retain.contains_key(t) as usize;
            }
        }
        format!(
            "; mapal.move loop={header} carried={} h2d={h2d} d2h={d2h} host_copy=0 alloc=0 retained={retained}\n",
            self.carried_desc(plan.merge)
        )
    }
}
