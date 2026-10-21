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
use mapal_ir::Footprint;

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

/// Where one input slot of an epilogue member's body comes from in the site's
/// kernel (S54 R1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpiSrc {
    /// Group value `k` at this cell: 0 is the site's accumulator, `j + 1`
    /// member `j`'s result. A captured slot reads it only at `t` (`Same`).
    Value(usize),
    /// An array complete in `Gmem` before the launch, read at a cell-only
    /// footprint through its device pointer.
    Gmem(ObjectId),
    /// The cell index `t` (the mapped `iota`).
    Index,
}

/// One per-cell morphism evaluated in a site's store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Member {
    pub m: MorphismId,
    pub body: FuncId,
    pub out: ObjectId,
    /// A reader outside the group exists: the result is stored.
    pub store: bool,
    /// One per body input slot (`0..=captures`; a capture-free map has one).
    pub srcs: Vec<EpiSrc>,
}

/// A device site's epilogue group (R1): its members in chain order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Epi {
    /// A reader outside the group exists: the site's own output is stored.
    pub store_out: bool,
    pub chain: Vec<Member>,
}

impl Epi {
    /// The distinct `Gmem` operands, in first-use order (the kernel's
    /// pointer parameters after `a, b, out`).
    pub(crate) fn caps(&self) -> Vec<ObjectId> {
        let mut out = Vec::new();
        for s in self.chain.iter().flat_map(|mb| &mb.srcs) {
            if let EpiSrc::Gmem(o) = *s
                && !out.contains(&o)
            {
                out.push(o);
            }
        }
        out
    }
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
        // An epilogue member (S54): read in a site's store, or not at all.
        if let Some(src) = self.member_src(m2, slot) {
            return if src == EpiSrc::Index {
                Use::None
            } else {
                Use::Dev
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
                Operation::Map { captures: 0, .. } if self.member_src(m, 0).is_some() => {
                    out.push(Use::Dev)
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
        let site_out = |o: ObjectId| self.produced_on_device(sites, o);
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
        (*lb == body
            && **inner == ElemSrc::Index
            && self.obj_ty(fd.input) == Ty::i32()
            && is_float(array_elem(&self.obj_ty(morph.target)))
            && self.device_leaf(body))
        .then_some(body)
    }

    /// A body the device runs with the host's own text: clean (no trap call,
    /// no effect), a leaf (no `Call`/`Map`/`Fold`), and every op lowering to
    /// the same bits on the device as on the host. A float `%` does not: the
    /// host calls `fmodf`, NVPTX expands `frem` to `fma(-y, trunc(x / y), x)`
    /// (S53 review: 2051 of 4096 elements differ).
    fn device_leaf(&self, body: FuncId) -> bool {
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
                && !matches!(
                    x.op,
                    Operation::Call(_) | Operation::Map { .. } | Operation::Fold { .. }
                )
        });
        self.attrs.clean(body) && leaf
    }

    /// The source of slot `slot` of epilogue member `m`, `None` when `m` is
    /// in no group.
    pub(super) fn member_src(&self, m: MorphismId, slot: u32) -> Option<EpiSrc> {
        self.groups
            .values()
            .flat_map(|g| &g.chain)
            .find(|mb| mb.m == m)
            .and_then(|mb| mb.srcs.get(slot as usize).copied())
    }

    /// Emit this map body as its epilogue device variant (S54): the captured
    /// arrays in `slots` arrive as one-cell buffers holding cell `t`.
    pub(crate) fn set_cell_slots(&mut self, slots: Vec<u32>) {
        self.cell_slots = slots;
    }

    /// Does the `Index` over product `source` read a one-cell buffer?
    pub(super) fn cell_read(&self, source: ObjectId) -> bool {
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
    fn cell_readers(&self, v: ObjectId) -> Vec<MorphismId> {
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
    /// directly)?
    fn read_outside(&self, o: ObjectId, ms: &[MorphismId]) -> bool {
        self.ir.out_edges(o).iter().any(|&m| {
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

    /// **S54 R1: the per-cell morphisms each device site's store evaluates.**
    ///
    /// Sites in execution order; a `Map` joins site `s`'s group when its
    /// [`CellSite`](mapal_ir::CellSite) reads a group value (the site's output,
    /// or a member's) only at `t`, and every other operand is complete in
    /// `Gmem` when `s` launches — produced by an earlier launch, or on the host
    /// before `s` (then copied in or built on the device, as a site operand
    /// is) — at a footprint that is a function of `t` alone (`Same`, `Row`,
    /// `Col`, `Const`). Law 1 of the placement model: an operand is read only
    /// where it is complete. Members run where the site runs (the same loop
    /// phase, never a guard arm), their bodies are device leaves, and their
    /// cells are the site's (`extent = rows · c`). A chain composes: a member
    /// of a member joins. Each group value is stored only when a reader
    /// outside the group exists — the use count; the rest live in a register.
    pub(super) fn epilogues(&self) -> SecondaryMap<MorphismId, Epi> {
        let mut out: SecondaryMap<MorphismId, Epi> = SecondaryMap::new();
        let sites = self.device_sites();
        let cells = self.ir.cell_plan(self.f);
        if sites.is_empty() || cells.is_empty() {
            return out;
        }
        // Where a morphism runs: straight-line, or one phase of one loop.
        let mut region: SecondaryMap<MorphismId, (Option<ObjectId>, bool)> = SecondaryMap::new();
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
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let target = |m: MorphismId| self.ir.morphism(m).expect("morphism resolves").target;
        let mut order: Vec<MorphismId> = sites.keys().collect();
        order.sort_by_key(|&m| pos(m));
        // The launch after which a device-made object is complete in `Gmem`.
        let mut made: SecondaryMap<ObjectId, MorphismId> = SecondaryMap::new();
        for &s in &order {
            made.insert(target(s), s);
        }
        let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for &s in &order {
            let l = gpu::launch(
                &sites[s],
                self.profile,
                self.contract,
                self.gpu_arms,
                self.reduced,
            );
            let Some(&rs) = region.get(s) else {
                continue;
            };
            if !l.rungs.epilogue {
                continue;
            }
            let c = target(s);
            let n = array_len(&self.obj_ty(c));
            let before = |p: MorphismId| {
                pos(p) < pos(s) && region.get(p).is_some_and(|&r| r == rs || r.0.is_none())
            };
            let mut values = vec![c];
            let mut chain: Vec<Member> = Vec::new();
            loop {
                // Complete before `s` launches, where `s` runs.
                let complete = |o: ObjectId| match made.get(o) {
                    Some(&p) => p != s && before(p),
                    None => matches!(self.ir.in_edges(o), [p] if before(*p)),
                };
                let mut cands: Vec<MorphismId> = values
                    .iter()
                    .flat_map(|&v| self.cell_readers(v))
                    .filter(|&m| !taken.contains_key(m) && cells.contains_key(m))
                    .collect();
                cands.sort_by_key(|&m| pos(m));
                cands.dedup();
                let admit = |mc: MorphismId| -> Option<Member> {
                    let morph = self.ir.morphism(mc)?;
                    let Operation::Map { body, captures } = morph.op else {
                        return None;
                    };
                    let cell = &cells[mc];
                    let in_ty = self.obj_ty(self.ir.func(body)?.input);
                    if region.get(mc) != Some(&rs)
                        || cell.extent != n
                        || !is_float(array_elem(&self.obj_ty(morph.target)))
                        || !self.device_leaf(body)
                    {
                        return None;
                    }
                    let mut srcs = Vec::new();
                    for slot in 0..=captures {
                        let arr = if captures == 0 {
                            morph.source
                        } else {
                            self.pair_source(morph.source, slot)?
                        };
                        let feet: Vec<Footprint> = cell
                            .reads
                            .iter()
                            .filter(|r| r.slot == slot)
                            .map(|r| r.foot)
                            .collect();
                        let src = match values.iter().position(|&v| v == arr) {
                            // A group value read off its own cell needs every
                            // block's store first: a kernel boundary (R3).
                            Some(k) if feet.iter().all(|&f| f == Footprint::Same) => {
                                EpiSrc::Value(k)
                            }
                            Some(_) => return None,
                            None if slot == captures && feet.is_empty() => EpiSrc::Index,
                            // ponytail: a mapped array outside the group is one
                            // more load per cell; admit it when a program has one.
                            None if slot == captures => return None,
                            None if feet.contains(&Footprint::Other) || !complete(arr) => {
                                return None;
                            }
                            None => EpiSrc::Gmem(arr),
                        };
                        if captures > 0 && crate::ty::erased_index(&in_ty, slot) != Some(slot) {
                            return None;
                        }
                        srcs.push(src);
                    }
                    srcs.iter()
                        .any(|s| matches!(s, EpiSrc::Value(_)))
                        .then_some(Member {
                            m: mc,
                            body,
                            out: morph.target,
                            store: false,
                            srcs,
                        })
                };
                let Some(mb) = cands.into_iter().find_map(admit) else {
                    break;
                };
                // `a, b, out`, every operand and every member's output.
                let mut trial = Epi {
                    store_out: true,
                    chain: chain.clone(),
                };
                trial.chain.push(mb.clone());
                if 3 + trial.caps().len() + trial.chain.len() > gpu::LAUNCHV_MAX {
                    break;
                }
                taken.insert(mb.m, ());
                values.push(mb.out);
                made.insert(mb.out, s);
                chain.push(mb);
            }
            if !chain.is_empty() {
                out.insert(
                    s,
                    Epi {
                        store_out: true,
                        chain,
                    },
                );
            }
        }
        // Stores from the use count, once every group is known (a later
        // group's operand is an outside reader of an earlier group's value).
        for (s, g) in out.iter_mut() {
            let ms: Vec<MorphismId> = g.chain.iter().map(|mb| mb.m).collect();
            g.store_out = self.read_outside(target(s), &ms);
            for mb in &mut g.chain {
                mb.store = self.read_outside(mb.out, &ms);
            }
        }
        out
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

        // Device outputs, an epilogue's stored results among them (S54): kept
        // on the device unless the host reads one whole.
        let outs: Vec<ObjectId> = sites
            .keys()
            .map(|m| self.ir.morphism(m).expect("site").target)
            .chain(
                self.groups
                    .values()
                    .flat_map(|g| g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out)),
            )
            .collect();
        for c in outs {
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
        // An epilogue's operands are read in the store: there, complete.
        for g in self.groups.values() {
            for o in g.caps() {
                read.insert(o, ());
            }
            for mb in &g.chain {
                plan.skip.insert(mb.m, ());
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
        self.ir
            .in_edges(o)
            .iter()
            .any(|&m| sites.contains_key(m) || self.member_src(m, 0).is_some())
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
        id: i64,
        geom: &str,
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
        let m = self.ir.in_edges(target)[0];
        let group = self.groups.get(m).cloned();
        // The values this launch leaves in `Gmem`: the site's output unless
        // its group holds every reader (S54), and each stored member result.
        let mut stored = Vec::new();
        let dout = if group.as_ref().is_none_or(|g| g.store_out) {
            let d = self.dalloc(target, o_bytes);
            stored.push((target, d.clone()));
            d
        } else {
            "0".to_string()
        };
        match &group {
            None => self.line(format!(
                "call void @mapal_gpu_launch(i64 {id}, i64 {}, i64 {}, i64 {dout}, {geom})",
                ops[0], ops[1]
            )),
            Some(g) => {
                let mut ptrs = vec![ops[0].clone(), ops[1].clone(), dout];
                for o in g.caps() {
                    debug_assert!(self.dvalid.contains_key(o), "epilogue operand in Gmem");
                    ptrs.push(self.dload(o));
                }
                for mb in g.chain.iter().filter(|mb| mb.store) {
                    let bytes = self.array_bytes(&self.obj_ty(mb.out));
                    let d = self.dalloc(mb.out, bytes);
                    ptrs.push(d.clone());
                    stored.push((mb.out, d));
                }
                let n = ptrs.len();
                let arr = format!("%p{}", self.fresh());
                self.allocas
                    .push_str(&format!("  {arr} = alloca [{n} x i64]\n"));
                for (i, p) in ptrs.iter().enumerate() {
                    let g = self.tmp();
                    self.line(format!(
                        "{g} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 {i}"
                    ));
                    self.line(format!("store i64 {p}, ptr {g}"));
                }
                self.line(format!(
                    "call void @mapal_gpu_launchv(i64 {id}, i32 0, i32 {n}, ptr {arr}, {geom})"
                ));
            }
        }
        for (o, _) in &stored {
            self.dvalid.insert(*o, ());
        }
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            let keep = self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(o) || p.is_carry_next(o));
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("device output slot");
                self.line(format!(
                    "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 0, i64 {bytes})"
                ));
            }
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
