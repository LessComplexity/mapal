//! Retention at whole-object scope on a GPU profile (S53, plan-s53 §2.3):
//! an object's device copy is kept while the graph says a later reader is a
//! device site, so data crosses the bus once per object rather than once per
//! site — and not at all when the device can build it.
//!
//! **The rule.** Objects are immutable and every morphism writes a fresh one
//! (an in-place `Update` needs a dead source), so a device copy can never go
//! stale. What is left is placement, read off facts the IR already records:
//! - **F1 / F2** — an array some device site reads is copied in right after
//!   its producer, once: a second reader and a loop-invariant operand (whose
//!   producer sits outside the loop) find it already there.
//! - **F3** — a loop's state component that a device site reads and a device
//!   site produces is *carried* on the device (`Carry`): the back edge swaps
//!   two device buffers and the host never sees the intermediate states.
//! - **host reads** — a device output is copied out only if the host reads it:
//!   whole, right after the launch, when any host reader needs the array; one
//!   element at a time when every host reader is a proven `Index`.
//! - **death** — a buffer is freed after the morphism at `last_use_plan`'s
//!   death position when both ends run once (straight-line code), else at
//!   function exit (the buffer is reused by every iteration meanwhile).
//! - **`devgen`** — a producer whose element law is closed-form over the index
//!   (`Apply{captures: 0, inner: Index}` with a clean leaf body) and whose
//!   every reader is on the device is computed there (`mapal_gen_<id>`): no
//!   host loop, no copy. Inputs left with no reader at all are not produced.
//!
//! Every rule names an object relation, never a workload: on a decode loop the
//! same rules keep the weights resident and carry the state — what a hand
//! written program calls a cache is what falls out.
//!
//! Anything outside the recognized shapes keeps the host copy valid, which is
//! always correct: a missed retention costs a copy, never a value.

use super::*;

/// One use of an array object, seen through the products that carry it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// The product slot is never read (a device site's mapped index space).
    None,
    /// A device site reads it as an operand.
    Dev,
    /// A host `Index` with a proven bound reads one element.
    Elem(MorphismId),
    /// A `Pair` the carry owns (the next state, the exit payload).
    Fwd,
    /// Anything else: the host needs the whole array.
    Host,
}

/// A loop state component kept on the device (F3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Carry {
    /// The `LoopMerge` (a product state); its device slot is the carry's.
    pub merge: ObjectId,
    /// The state component.
    pub k: u32,
    /// The device site's output that becomes the next state.
    pub next: ObjectId,
    /// `Proj{k}` views of the merge: the current state inside the body.
    pub views: Vec<ObjectId>,
    /// The loop's exit object, when the exit payload is a view.
    pub exit: Option<ObjectId>,
}

/// A producer computed on the device (`devgen`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Gen {
    pub id: u64,
    pub body: FuncId,
    pub n: u64,
}

/// The placement decisions for one function — a pure function of the IR, the
/// profile and the rungs, computed before the walk.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResidentPlan {
    /// Host-produced arrays a device site reads: copied in after the producer.
    pub upload: SecondaryMap<ObjectId, ()>,
    /// Device-produced arrays the host never reads whole (no copy-out).
    pub keep: SecondaryMap<ObjectId, ()>,
    /// Host `Index` reads served by a one-element copy-out.
    pub elem: SecondaryMap<MorphismId, ()>,
    /// Buffers freed right after this morphism (their death).
    pub free_after: SecondaryMap<MorphismId, Vec<ObjectId>>,
    /// Producers computed on the device.
    pub devgen: SecondaryMap<MorphismId, Gen>,
    /// Producers no reader is left for (every reader is device-built).
    pub skip: SecondaryMap<MorphismId, ()>,
    pub carries: Vec<Carry>,
}

impl ResidentPlan {
    pub(crate) fn is_carry_next(&self, o: ObjectId) -> bool {
        self.carries.iter().any(|c| c.next == o)
    }

    /// The object whose device slot `o` shares: a carry's views and exit share
    /// the merge's slot.
    pub(crate) fn alias(&self, o: ObjectId) -> ObjectId {
        self.carries
            .iter()
            .find(|c| c.views.contains(&o) || c.exit == Some(o))
            .map_or(o, |c| c.merge)
    }
}

impl<'a> FnEmit<'a> {
    /// The device sites of this function, by morphism.
    fn device_sites(&self) -> SecondaryMap<MorphismId, TileSite> {
        let mut out = SecondaryMap::new();
        if let Some(plan) = &self.tile_plan {
            for (m, site) in plan.sites.iter() {
                if gpu::gpu_site(site, self.profile) {
                    out.insert(m, site.clone());
                }
            }
        }
        out
    }

    /// How consumer `m2` reads slot `slot` of the pointer-only product it
    /// consumes.
    fn slot_use(
        &self,
        sites: &SecondaryMap<MorphismId, TileSite>,
        m2: MorphismId,
        slot: u32,
    ) -> Use {
        if let Some(site) = sites.get(m2) {
            return if site.a.slot == slot || site.b.slot == slot {
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
    fn uses(
        &self,
        sites: &SecondaryMap<MorphismId, TileSite>,
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
                        out.push(self.slot_use(sites, m2, slot));
                    }
                }
                _ => out.push(Use::Host),
            }
        }
        out
    }

    /// The morphisms the straight-line walk emits itself — run exactly once
    /// per call (outside every loop cone and guard arm).
    fn straight_line(&self) -> SecondaryMap<MorphismId, ()> {
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
    fn carries(&self, sites: &SecondaryMap<MorphismId, TileSite>) -> Vec<Carry> {
        let site_out = |o: ObjectId| self.ir.in_edges(o).iter().any(|&m| sites.contains_key(m));
        let mut out = Vec::new();
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
                    if !matches!(ty, Ty::Array { .. }) {
                        continue;
                    }
                    let Some(next) = self.pair_source(state, k) else {
                        continue;
                    };
                    if !site_out(next) || self.pair_source(plan.init, k).is_none() {
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
                    let exit_pair = self
                        .ir
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
                        self.uses(sites, o, &fwd)
                            .iter()
                            .all(|u| !matches!(u, Use::Host))
                    };
                    let read = views
                        .iter()
                        .any(|&v| self.uses(sites, v, &fwd).contains(&Use::Dev));
                    if read && device_only(next) && views.iter().all(|&v| device_only(v)) {
                        out.push(Carry {
                            merge,
                            k,
                            next,
                            views,
                            exit,
                        });
                    }
                }
            }
        }
        out
    }

    /// The `devgen` law gate: `out[i] = body(i)` with a clean body that calls
    /// nothing and takes the `i32` index — exactly what a one-thread-per-element
    /// kernel can evaluate with the host's own body text.
    pub(super) fn gen_law(&self, m: MorphismId) -> Option<FuncId> {
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
        let leaf = fd.morphisms.iter().all(|&g| {
            !matches!(
                self.ir.morphism(g).map(|x| x.op),
                Some(Operation::Call(_) | Operation::Map { .. } | Operation::Fold { .. })
            )
        });
        (*lb == body
            && **inner == ElemSrc::Index
            && self.obj_ty(fd.input) == Ty::i32()
            && is_float(array_elem(&self.obj_ty(morph.target)))
            && self.attrs.clean(body)
            && leaf)
            .then_some(body)
    }

    /// Build the plan. `None` when the function has no device site.
    pub(crate) fn resident_plan(&self, devgen: bool) -> Option<ResidentPlan> {
        let sites = self.device_sites();
        if sites.is_empty() {
            return None;
        }
        let mut plan = ResidentPlan {
            carries: self.carries(&sites),
            ..ResidentPlan::default()
        };
        let mut device = Vec::new(); // every object given a device copy here

        // Device outputs: kept on the device unless the host reads one whole.
        for (m, _) in sites.iter() {
            let c = self.ir.morphism(m).expect("site").target;
            device.push(c);
            let uses = if plan.is_carry_next(c) {
                vec![] // checked device-only when the carry was admitted
            } else {
                self.uses(&sites, c, &[])
            };
            if uses.iter().all(|u| !matches!(u, Use::Host)) {
                plan.keep.insert(c, ());
            }
        }
        for carry in &plan.carries {
            if let Some(x) = carry.exit
                && self
                    .uses(&sites, x, &[])
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
            for u in self.uses(&sites, o, &[]) {
                if let Use::Elem(m) = u {
                    plan.elem.insert(m, ());
                }
            }
        }

        // The operands device sites read (their feeders).
        let mut read: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for (m, site) in sites.iter() {
            let src = self.ir.morphism(m).expect("site").source;
            for slot in [site.a.slot, site.b.slot] {
                if let Some(f) = self.pair_source(src, slot) {
                    read.insert(f, ());
                }
            }
        }
        let carried = |o: ObjectId| plan.carries.iter().any(|c| c.views.contains(&o));
        for (o, _) in read.iter() {
            if carried(o) || self.produced_on_device(&sites, o) {
                continue;
            }
            let Some(&p) = self.ir.in_edges(o).first() else {
                continue; // a parameter: copied in at the site
            };
            if devgen
                && let Some(body) = self.gen_law(p)
                && self
                    .uses(&sites, o, &[])
                    .iter()
                    .all(|u| matches!(u, Use::Dev | Use::Elem(_) | Use::None))
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
                    .uses(&sites, o, &[])
                    .iter()
                    .any(|u| matches!(u, Use::Elem(_)))
                {
                    plan.keep.insert(o, ());
                    for u in self.uses(&sites, o, &[]) {
                        if let Use::Elem(m) = u {
                            plan.elem.insert(m, ());
                        }
                    }
                }
            } else {
                plan.upload.insert(o, ());
            }
            device.push(o);
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
                    Operation::Map { captures: 0, .. } => plan.devgen.contains_key(c),
                    Operation::Pair { slot, .. } => {
                        self.pointer_only_array_component(cm.target, slot)
                            && self
                                .ir
                                .out_edges(cm.target)
                                .iter()
                                .all(|&m2| self.slot_use(&sites, m2, slot) == Use::None)
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
            let Some(death) = self.lup.death(o) else {
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
        Some(plan)
    }

    fn produced_on_device(&self, sites: &SecondaryMap<MorphismId, TileSite>, o: ObjectId) -> bool {
        self.ir.in_edges(o).iter().any(|&m| sites.contains_key(m))
    }

    // --- emission ---------------------------------------------------------

    /// `o`'s device slot (`i64`, 0 = no buffer), minted zeroed in the entry
    /// block on first ask.
    pub(super) fn dslot(&mut self, o: ObjectId) -> String {
        let o = self.resident.as_ref().map_or(o, |p| p.alias(o));
        if let Some(s) = self.dslots.get(o) {
            return s.clone();
        }
        let s = format!("%d{}", self.fresh());
        self.allocas
            .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
        self.dslots.insert(o, s.clone());
        s
    }

    /// `o`'s device buffer, allocated on first use (a loop reuses it).
    fn dalloc(&mut self, o: ObjectId, bytes: u64) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        self.line(format!(
            "{d} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {bytes})"
        ));
        d
    }

    fn dload(&mut self, o: ObjectId) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        self.line(format!("{d} = load i64, ptr {slot}"));
        d
    }

    fn array_bytes(&self, ty: &Ty) -> u64 {
        array_len(ty) * crate::profile::elem_bytes(array_elem(ty))
    }

    /// Replaces the host realization of `m` (a device-built or unread
    /// producer). `true` when `m` was handled here.
    pub(super) fn resident_instead(&mut self, m: MorphismId) -> bool {
        let Some(plan) = &self.resident else {
            return false;
        };
        if plan.skip.contains_key(m) {
            return true;
        }
        let Some(g) = plan.devgen.get(m).cloned() else {
            return false;
        };
        let t = self.ir.morphism(m).expect("morphism resolves").target;
        let bytes = self.array_bytes(&self.obj_ty(t));
        let d = self.dalloc(t, bytes);
        self.line(format!(
            "call void @mapal_gpu_gen(i64 {}, i64 {d}, i64 {})",
            g.id as i64, g.n
        ));
        self.dvalid.insert(t, ());
        true
    }

    /// After `m`: copy its output in if a device site reads it, and free every
    /// buffer whose death is `m`.
    pub(super) fn resident_after(&mut self, m: MorphismId) {
        let Some(plan) = &self.resident else {
            return;
        };
        let t = self.ir.morphism(m).expect("morphism resolves").target;
        let upload = plan.upload.contains_key(t) && !plan.devgen.contains_key(m);
        let frees = plan.free_after.get(m).cloned().unwrap_or_default();
        if upload {
            let ty = self.obj_ty(t);
            let bytes = self.array_bytes(&ty);
            let h = self.array_operand_ptr(t, None).expect("uploaded array ptr");
            let d = self.dalloc(t, bytes);
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
            ));
            self.dvalid.insert(t, ());
        }
        for o in frees {
            let slot = self.dslot(o);
            self.line(format!("call void @mapal_gpu_free(ptr {slot})"));
            self.dvalid.remove(o);
        }
    }

    /// The device half of a site under `resident`: operands from their device
    /// copies (copied in here only when no producer did), output kept on the
    /// device, copied out whole only when the host reads it whole.
    pub(super) fn emit_site_resident(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        call: impl Fn(&str, &str, &str) -> String,
    ) {
        let src_ty = self.obj_ty(source);
        let mut ops = Vec::new();
        let mut temps = Vec::new();
        for slot in [site.a.slot, site.b.slot] {
            let bytes = self.array_bytes(src_ty.component_ty(slot).expect("tile operand"));
            let feeder = self.pair_source(source, slot);
            if let Some(f) = feeder
                && self
                    .dvalid
                    .contains_key(self.resident.as_ref().map_or(f, |p| p.alias(f)))
            {
                ops.push(self.dload(f));
                continue;
            }
            // No producer copied it in (a parameter, an exit object): copy
            // it in here, every time the site runs.
            let h = self
                .array_operand_ptr(source, Some(slot))
                .expect("tile operand ptr");
            let slot_name = format!("%d{}", self.fresh());
            self.allocas.push_str(&format!(
                "  {slot_name} = alloca i64\n  store i64 0, ptr {slot_name}\n"
            ));
            let d = self.tmp();
            self.line(format!(
                "{d} = call i64 @mapal_gpu_alloc(ptr {slot_name}, i64 {bytes})"
            ));
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
            ));
            temps.push(slot_name);
            ops.push(d);
        }
        let o_bytes = self.array_bytes(&self.obj_ty(target));
        let dout = self.dalloc(target, o_bytes);
        self.line(call(&ops[0], &ops[1], &dout));
        self.dvalid.insert(target, ());
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        let keep = self
            .resident
            .as_ref()
            .is_some_and(|p| p.keep.contains_key(target) || p.is_carry_next(target));
        if !keep {
            let h = self.slot(target).expect("tile output slot");
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h}, i64 {dout}, i64 0, i64 {o_bytes})"
            ));
        }
    }

    /// A proven `Index` on a device-only array: copy the one element out
    /// right before the host loads it.
    pub(super) fn resident_elem(&mut self, m: MorphismId, source: ObjectId, ep: &str, idx: &str) {
        if !self
            .resident
            .as_ref()
            .is_some_and(|p| p.elem.contains_key(m))
        {
            return;
        }
        let arr = self.pair_source(source, 0).expect("elem read array");
        let eb = crate::profile::elem_bytes(array_elem(&self.obj_ty(arr)));
        let d = self.dload(arr);
        let off = self.tmp();
        self.line(format!("{off} = mul i64 {idx}, {eb}"));
        self.line(format!(
            "call void @mapal_gpu_d2h(ptr {ep}, i64 {d}, i64 {off}, i64 {eb})"
        ));
    }

    fn carries_of(&self, merge: ObjectId) -> Vec<Carry> {
        self.resident.as_ref().map_or_else(Vec::new, |p| {
            p.carries
                .iter()
                .filter(|c| c.merge == merge)
                .cloned()
                .collect()
        })
    }

    /// Loop entry: the carry's buffer gets the init state, copied in once.
    pub(crate) fn resident_loop_enter(&mut self, merge: ObjectId, init: ObjectId) {
        for c in self.carries_of(merge) {
            let ty = self
                .obj_ty(merge)
                .component_ty(c.k)
                .cloned()
                .expect("carry ty");
            let bytes = self.array_bytes(&ty);
            let h = self.array_operand_ptr(init, Some(c.k)).expect("init ptr");
            let d = self.dalloc(merge, bytes);
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
            ));
            self.dvalid.insert(merge, ());
        }
    }

    /// The back edge: the next state's buffer becomes the carry's, and the old
    /// state's buffer (dead past the back edge) is the next launch's output.
    pub(crate) fn resident_loop_back(&mut self, merge: ObjectId) {
        for c in self.carries_of(merge) {
            let (cs, ns) = (self.dslot(merge), self.dslot(c.next));
            let (old, new) = (self.tmp(), self.tmp());
            self.line(format!("{old} = load i64, ptr {cs}"));
            self.line(format!("{new} = load i64, ptr {ns}"));
            self.line(format!("store i64 {new}, ptr {cs}"));
            self.line(format!("store i64 {old}, ptr {ns}"));
        }
    }

    /// The exit: the exit object is the carry's buffer; copied out whole when
    /// the host reads it whole.
    pub(crate) fn resident_loop_exit(&mut self, merge: ObjectId, exit: ObjectId) {
        for c in self.carries_of(merge) {
            if c.exit != Some(exit) {
                continue;
            }
            let bytes = self.array_bytes(&self.obj_ty(exit));
            self.dvalid.insert(merge, ());
            if !self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(exit))
            {
                let d = self.dload(exit);
                let h = self.slot(exit).expect("exit slot");
                self.line(format!(
                    "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 0, i64 {bytes})"
                ));
            }
        }
    }

    /// Function exit: free every buffer still held (a no-op on freed slots).
    pub(super) fn resident_teardown(&mut self) {
        let slots: Vec<String> = self.dslots.values().cloned().collect();
        let mut seen = std::collections::BTreeSet::new();
        for s in slots {
            if seen.insert(s.clone()) {
                self.line(format!("call void @mapal_gpu_free(ptr {s})"));
            }
        }
    }
}

fn array_len(ty: &Ty) -> u64 {
    match ty {
        Ty::Array { size, .. } => *size,
        _ => 0,
    }
}

fn array_elem(ty: &Ty) -> &Ty {
    match ty {
        Ty::Array { elem, .. } => elem,
        other => other,
    }
}
