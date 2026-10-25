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
//! - **`arena`** (S54 R6) — every buffer freed at a straight-line death sits at
//!   a compile-time offset of one per-function allocation: interval coloring
//!   over `LastUsePlan` positions, so objects whose lives are disjoint share
//!   bytes. A device-only object (kept, device-built, or never built) gets no
//!   host slot: the host neither allocates nor pages it.
//!
//! Every rule names an object relation, never a workload: on a decode loop the
//! same rules keep the weights resident and carry the state — what a hand
//! written program calls a cache is what falls out.
//!
//! Anything outside the recognized shapes keeps the host copy valid, which is
//! always correct: a missed retention costs a copy, never a value.

use super::*;
use mapal_ir::{Affine, CellSite, Delta, Footprint};

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

impl Carry {
    /// The object whose device slot is this component's buffer: its first
    /// view (one per component — a state may carry several arrays).
    pub(crate) fn key(&self) -> ObjectId {
        self.views[0]
    }
}

/// The buffer of a resident launch an L2 access-policy window covers (S54
/// `l2-window`, [`FnEmit::l2_window`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Window {
    A,
    B,
    Out,
}

/// A producer computed on the device (`devgen`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Gen {
    pub id: u64,
    pub body: FuncId,
    pub n: u64,
}

/// A host-written array uploaded as the rows its iteration wrote (S54
/// `delta-h2d`): its device copy persists across iterations and differs from
/// the host's only in `Δ`, so after one whole upload per loop entry only
/// `[start, start + stride)` crosses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeltaUp {
    /// The loop whose entry resets the first-pass flag.
    pub merge: ObjectId,
    /// The written row's first cell: `a·var + b`.
    pub start: Affine,
    pub stride: u64,
}

/// A device site: a tile site (R0), or a row fold over a device output (R0
/// for folds, S54 `rowfold`).
#[derive(Clone, Debug)]
pub(crate) enum DevSite {
    Tile(TileSite),
    RowFold(super::rowfold::RowFold),
}

impl DevSite {
    /// The source slots the kernel loads.
    fn reads(&self) -> Vec<u32> {
        match self {
            DevSite::Tile(s) => vec![s.a.slot, s.b.slot],
            DevSite::RowFold(r) => vec![r.site.a.slot],
        }
    }
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

/// The `Map`/`Fold` bodies nested in `f`, transitively, in first-met order:
/// the functions a device body calls, which the device unit carries too.
pub(crate) fn nested_bodies(ir: &CategoryIr, f: FuncId) -> Vec<FuncId> {
    let mut out: Vec<FuncId> = Vec::new();
    let mut stack = vec![f];
    while let Some(g) = stack.pop() {
        for &m in ir.func(g).map_or(&[][..], |d| &d.morphisms[..]) {
            if let Some(Operation::Map { body, .. } | Operation::Fold { body, .. }) =
                ir.morphism(m).map(|x| x.op)
                && !out.contains(&body)
            {
                out.push(body);
                stack.push(body);
            }
        }
    }
    out
}

/// The base launch that writes an epilogue member's physical bytes.
pub(super) fn grouped_writer(groups: &SecondaryMap<MorphismId, Epi>, o: ObjectId) -> Option<MorphismId> {
    groups
        .iter()
        .find(|(_, g)| g.chain.iter().any(|mb| mb.out == o))
        .map(|(m, _)| m)
}

/// A per-cell map placed on the device (S54 `devmap`, ARCHITECTURE R2): its
/// own one-thread-per-cell kernel, `out[i] = body(caps…, x[i])` with the
/// host's body text, launched on the device copies of its operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Cell {
    pub id: u64,
    pub body: FuncId,
    /// The extent: one thread per output cell.
    pub n: u64,
    /// The mapped array, loaded per cell; `None` when its element is the
    /// index itself (an `iota`'s law), formed in the kernel.
    pub mapped: Option<ObjectId>,
    /// The captured arrays in slot order, passed by reference as the host
    /// passes them.
    pub caps: Vec<ObjectId>,
    /// `cell-law`: the capture slots evaluated in the cell instead of loaded,
    /// each with its producer's closed-form body (`devgen`'s law).
    pub laws: Vec<(u32, FuncId)>,
    /// The number capture slots (`CellSite::uniform`): kernel arguments by
    /// value, the host's scalar at launch.
    pub uniform: Vec<u32>,
    /// The capture slots a nested fold reads (`CellSite::nested`).
    pub nested: Vec<u32>,
}

impl Cell {
    /// The arrays the kernel reads, with the product slot each sits in
    /// (`None`: the bare mapped source of a capture-free map).
    fn operands(&self) -> Vec<(ObjectId, Option<u32>)> {
        self.args().filter(|a| !a.2).map(|(o, s, _)| (o, s)).collect()
    }

    /// The kernel's arguments after its output, in [`gpu::cell_kernel_text`]'s
    /// parameter order: the mapped array, then each capture slot not
    /// evaluated in the cell — `(object, slot, uniform)`, a uniform passed by
    /// value, the rest by reference.
    fn args(&self) -> impl Iterator<Item = (ObjectId, Option<u32>, bool)> + '_ {
        let k = self.caps.len() as u32;
        let mapped = self.mapped.map(|o| (o, (k > 0).then_some(k)));
        let caps = (0..k).filter(|&s| !self.law(s)).map(|s| (self.caps[s as usize], Some(s)));
        mapped
            .into_iter()
            .chain(caps)
            .map(|(o, s)| (o, s, s.is_some_and(|s| self.uniform.contains(&s))))
    }

    fn law(&self, slot: u32) -> bool {
        self.laws.iter().any(|&(s, _)| s == slot)
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
    /// S54 `arena`: the device arena and the host-free objects.
    pub arena: Option<Arena>,
    /// `upload`s that cross as the rows their iteration wrote (`delta-h2d`).
    pub delta: SecondaryMap<ObjectId, DeltaUp>,
    /// Row folds run on the device (S54 `rowfold`).
    pub rowfold: SecondaryMap<MorphismId, super::rowfold::RowFold>,
    /// Per-cell maps run as their own device kernels (`devmap`).
    pub cells: SecondaryMap<MorphismId, Cell>,
    /// S54 `streams` (R7): each straight-line device site's chain (its stream,
    /// from 1), when [`FnEmit::chains`] opened streams; empty = one stream.
    pub chain: SecondaryMap<MorphismId, u32>,
    /// cpu-gpu-boundary `chain-h2d`: an upload every device reader of which
    /// is a site of one chain, copied in on that chain's stream.
    pub upload_chain: SecondaryMap<ObjectId, u32>,
    /// `flow`: the unit groups, by their launch (first member).
    pub units: SecondaryMap<MorphismId, super::flow::Unit>,
    /// generic-fusion: the host intent line (`; mapal.flow.host …`).
    pub flow_line: String,
}

/// Every arena offset is a multiple of this: `cuMemAlloc`'s own alignment
/// guarantee (CUDA C Programming Guide: an address returned by the driver's
/// allocators is aligned to at least 256 bytes), so an arena member is
/// addressed exactly as its own allocation would be.
pub(crate) const ARENA_ALIGN: u64 = 256;

/// One function's device arena (S54 R6), a pure function of the plan.
#[derive(Clone, Debug, Default)]
pub(crate) struct Arena {
    /// The allocation, a multiple of [`ARENA_ALIGN`]; 0 = no member.
    pub bytes: u64,
    /// Each member's byte offset: the objects freed at a straight-line death.
    pub offset: SecondaryMap<ObjectId, u64>,
    /// Device-only arrays: no host slot (no `%Frame` field, no paging).
    pub host_free: SecondaryMap<ObjectId, ()>,
}

/// First-fit interval coloring: `live[i] = (birth, death, bytes)` occupies
/// `[birth, death]` (inclusive: a site reading one object while writing
/// another holds both); in birth order each takes the lowest
/// [`ARENA_ALIGN`]-aligned offset no time-overlapping object holds. Returns
/// the offsets and the arena size.
pub(crate) fn color(live: &[(u32, u32, u64)]) -> (Vec<u64>, u64) {
    let size = |i: usize| live[i].2.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
    let mut order: Vec<usize> = (0..live.len()).collect();
    order.sort_by_key(|&i| (live[i].0, i));
    let mut off = vec![0; live.len()];
    let (mut placed, mut total): (Vec<usize>, u64) = (Vec::new(), 0);
    for i in order {
        let (b, d, _) = live[i];
        let mut busy: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&j| live[j].0 <= d && b <= live[j].1)
            .map(|&j| (off[j], off[j] + size(j)))
            .collect();
        busy.sort_unstable();
        let mut at = 0;
        for (s, e) in busy {
            if at + size(i) <= s {
                break;
            }
            at = at.max(e);
        }
        off[i] = at;
        total = total.max(at + size(i));
        placed.push(i);
    }
    (off, total)
}

impl ResidentPlan {
    pub(crate) fn is_carry_next(&self, o: ObjectId) -> bool {
        self.carries.iter().any(|c| c.next == o)
    }

    /// The unit launch that writes `o`'s device bytes, when a unit group
    /// stores it (generic-fusion): before `o`'s own morphism's position.
    pub(crate) fn unit_writer(&self, o: ObjectId) -> Option<MorphismId> {
        self.units
            .iter()
            .find(|(_, u)| u.members.iter().any(|mb| mb.out == o && mb.store))
            .map(|(m, _)| m)
    }

    /// The chain stream `o` is produced on (0: the legacy stream).
    pub(crate) fn stream_of(
        &self,
        ir: &CategoryIr,
        groups: &SecondaryMap<MorphismId, Epi>,
        o: ObjectId,
    ) -> u32 {
        // A carry's views and exit are its state's: the merge's writers.
        let state = self.carries.iter().find(|c| c.views.contains(&o) || c.exit == Some(o));
        let o = state.map_or(o, |c| c.merge);
        if let Some(m) = grouped_writer(groups, o) {
            return self.chain.get(m).copied().unwrap_or(0);
        }
        ir.in_edges(o)
            .iter()
            .find_map(|&m| self.chain.get(m).copied())
            .unwrap_or(0)
    }

    /// The object whose device slot `o` shares: a carry's views and exit share
    /// the component's slot (its key).
    pub(crate) fn alias(&self, o: ObjectId) -> ObjectId {
        self.carries
            .iter()
            .find(|c| c.views.contains(&o) || c.exit == Some(o))
            .map_or(o, Carry::key)
    }
}

impl<'a> FnEmit<'a> {
    /// The device sites of this function, by morphism.
    fn device_sites(&self) -> SecondaryMap<MorphismId, DevSite> {
        let mut out = SecondaryMap::new();
        if let Some(plan) = &self.tile_plan {
            for (m, site) in plan.sites.iter() {
                if self.device.contains_key(m) {
                    out.insert(m, DevSite::Tile(site.clone()));
                }
            }
            let folds = super::rowfold::rowfold_sites(
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
    fn slot_use(
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
    fn uses(
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
    fn carries(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
    ) -> Vec<Carry> {
        let site_out = |o: ObjectId| self.produced_on_device(sites, cells, o);
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
                            self.uses(sites, cells, v, &fwd).iter().all(|u| *u == Use::Fwd)
                        });
                    if (read || unread)
                        && !views.is_empty()
                        && device_only(next)
                        && views.iter().all(|&v| device_only(v))
                    {
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
    ///
    /// A nested `Map`/`Fold` is one more body of the same kind (generic-fusion
    /// natural forms): the device unit carries its text too
    /// ([`nested_bodies`]). Depth-guarded (J1).
    pub(super) fn device_leaf(&self, body: FuncId) -> bool {
        self.device_leaf_at(body, 0)
    }

    fn device_leaf_at(&self, body: FuncId, depth: u32) -> bool {
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
    pub(super) fn cell_readers(&self, v: ObjectId) -> Vec<MorphismId> {
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
    pub(super) fn read_outside(&self, o: ObjectId, ms: &[MorphismId]) -> bool {
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

    /// Where each morphism runs: straight-line (`(None, _)`), or one phase of
    /// one loop (`(Some(merge), advance)`). A morphism in neither (a guard
    /// arm, loop machinery) is absent.
    pub(super) fn regions(&self) -> SecondaryMap<MorphismId, (Option<ObjectId>, bool)> {
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
        let region = self.regions();
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
            // A row fold's store is a scalar per row: no group (S54 `rowfold`).
            let DevSite::Tile(site) = &sites[s] else {
                continue;
            };
            let l = gpu::launch(
                site,
                self.profile,
                self.contract,
                self.gpu_arms,
            );
            let Some(&rs) = region.get(s) else {
                continue;
            };
            // A mixed-radix site stores digit by digit: no epilogue (`radix-tile`).
            if !l.rungs.epilogue || site.radix.is_some() {
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
                    // ponytail: the site's store takes no scalar argument and
                    // walks no nested fold; such a member stays its own cell.
                    if !cell.uniform.is_empty()
                        || !cell.nested.is_empty()
                        || region.get(mc) != Some(&rs)
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

    /// The `upload`s a loop's iteration writes only rows of (`delta-h2d`): the
    /// value `write_delta` says crosses the back edge, or a `Same` image of it.
    /// Its producer runs every iteration, inside the loop, so its device copy
    /// is never freed there and holds last iteration's value.
    fn delta_uploads(
        &self,
        upload: &SecondaryMap<ObjectId, ()>,
    ) -> SecondaryMap<ObjectId, DeltaUp> {
        let mut out = SecondaryMap::new();
        for d in self.ir.write_delta(self.f).values() {
            let Delta::Rows {
                stride,
                start,
                next,
            } = *d
            else {
                continue;
            };
            // The counter is a `Proj` of the loop's merge.
            let merge = self.ir.in_edges(start.var).first();
            // (A range wider than the array is the whole upload.)
            if let Some(merge) = merge.and_then(|&m| self.ir.morphism(m)).map(|m| m.source)
                && upload.contains_key(next)
                && stride <= array_len(&self.obj_ty(next))
            {
                out.insert(
                    next,
                    DeltaUp {
                        merge,
                        start,
                        stride,
                    },
                );
            }
        }
        out
    }

    /// The `devmap` gate on a [`CellSite`] (`cell_plan`: a classifiable body
    /// over array captures, number captures (`uniform`) and nested folds over
    /// captures (`nested`), every read classified): outside every guard
    /// arm, a body whose text names no runtime symbol (clean) and lowers to
    /// the host's bits on the device ([`Self::device_leaf`]), number elements, an
    /// input product with no erased component, and no more pointers than one
    /// launch passes. A closed-form producer is `devgen`'s, not a cell.
    fn cell_of(&self, m: MorphismId, site: &CellSite) -> Option<Cell> {
        let morph = self.ir.morphism(m)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        let number = |t: &Ty| matches!(t, Ty::Float { .. } | Ty::Int { .. });
        let Ty::Array { elem, .. } = self.obj_ty(morph.target) else {
            return None;
        };
        // `uniform`. ponytail: an in-place range map over a uniform's cells
        // (`map_law` piecewise, `Outside`) stays the host's range write; a
        // whole-extent cell would rewrite every cell per trip. A device range
        // cell is item 1c.
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let piecewise = || matches!(self.ir.map_law(m), Some(ElemSrc::Apply { outside: Some(_), .. }));
        // `nest-cell`: a body folding over captured arrays, unless the map is
        // a tile or row-fold site (their own kernels).
        let nest = !site.nested.is_empty();
        let own = || {
            self.ir.tile_plan_strided(self.f).sites.contains_key(m)
                || self.ir.row_fold(self.f).contains_key(m)
        };
        if self.gated.contains_key(m)
            || (nest && (!rungs.nest_cell || own()))
            || (!site.uniform.is_empty() && (!rungs.uniform || piecewise()))
            || self.gen_law(m).is_some()
            || !self.attrs.clean(body)
            || !self.device_leaf(body)
            || !is_float(&elem)
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
        let laws =
            if !loaded && rungs.cell_law {
                self.laws(body, site, &caps)
            } else {
                Vec::new()
            };
        let cell = Cell {
            id: gpu::cell_id(&self.fnames[body], site.extent, self.profile),
            body,
            n: site.extent,
            mapped: loaded.then_some(mapped),
            caps,
            laws,
            uniform: site.uniform.clone(),
            nested: site.nested.clone(),
        };
        // The output, then every operand and uniform: one 8-byte argument each.
        (cell.operands().len() + cell.uniform.len() < gpu::LAUNCHV_MAX).then_some(cell)
    }

    /// `cell-law`'s fact: the captures of a cell over an `iota` that are
    /// closed-form (`devgen`'s law, `x[i] = f(i)`) and read only at the cell's
    /// own index (every read `Same`, and the capture used for nothing but
    /// those reads). Such a read is `f(i)`, evaluated in the cell: nothing
    /// stores `x`, nothing loads it.
    fn laws(&self, body: FuncId, site: &CellSite, caps: &[ObjectId]) -> Vec<(u32, FuncId)> {
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

    /// R2 (ARCHITECTURE §5): the per-cell maps that run on the device. A cell
    /// runs there when every operand is already in device memory — a device
    /// site's or an earlier cell's output, a carried state, a tile operand
    /// (copied in for its site anyway), or closed-form (`devgen` builds it) —
    /// and either some operand lives only there or no host reader needs the
    /// output whole: a whole-host reader forces a copy-out, which costs no
    /// more than copying the operands out only when one of them is device-only.
    ///
    /// Cells and carries decide each other (a carried view is an operand; a
    /// cell's output can be the next state), so both are the greatest fixed
    /// point from "every candidate": each round drops the cells whose operands
    /// or readers the round before no longer supports. Nothing is added, so it
    /// ends; with `devmap` off there is no candidate and the carries are S53's.
    /// S65 rule R for row folds: a fold over an array complete on the device —
    /// any device output (a cell kernel's, another fold's), not only a tile
    /// site's (`rowfold::rowfold_sites`) — runs there. Every such fold is a
    /// candidate site; [`Self::cell_sites`]' fixed point drops the ones whose
    /// operand is not device-produced (the byte rule: a fold over a host array
    /// stays on the host).
    fn fold_candidates(&self, sites: &mut SecondaryMap<MorphismId, DevSite>) -> Vec<MorphismId> {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if !rungs.rowfold
            || !rungs.by_ref
            || sites.is_empty()
            || (!rungs.retained_fold && !retain::retained_sites(self.ir, self.f, self.profile, rungs).is_empty())
        {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (m, site) in self.ir.row_fold(self.f).iter() {
            let window = site.w > 1 || site.a.ksplit.is_some();
            if sites.contains_key(m) || !is_float(&site.elem) || site.k == 0 || (window && !rungs.window) {
                continue;
            }
            sites.insert(
                m,
                DevSite::RowFold(super::rowfold::RowFold {
                    site: site.clone(),
                    colmajor: None,
                }),
            );
            out.push(m);
        }
        out
    }

    fn cell_sites(
        &self,
        sites_mut: &mut SecondaryMap<MorphismId, DevSite>,
        folds: &[MorphismId],
        devgen: bool,
    ) -> (SecondaryMap<MorphismId, Cell>, Vec<Carry>) {
        let mut cells = SecondaryMap::new();
        // A Map / Fold body runs once per element of its site: a launch there
        // would be one per element.
        let body = self.ir.morphisms().any(|(_, x)| {
            matches!(x.op, Operation::Map { body, .. } | Operation::Fold { body, .. } if body == self.f)
        });
        if !body && gpu::host_rungs(self.profile, self.contract, self.gpu_arms).devmap {
            for (m, site) in self.ir.cell_plan(self.f).iter() {
                // An epilogue member (R1) runs in its site's store already.
                if !sites_mut.contains_key(m)
                    && self.member_src(m, 0).is_none()
                    && self.cell_pays(m, site)
                    && let Some(c) = self.cell_of(m, site)
                {
                    cells.insert(m, c);
                }
            }
        }
        loop {
            let (drop, unfed, carries) = self.cell_round(sites_mut, folds, &cells, devgen);
            if drop.is_empty() && unfed.is_empty() {
                return (cells, carries);
            }
            for m in drop {
                cells.remove(m);
            }
            for m in unfed {
                sites_mut.remove(m);
            }
        }
    }

    /// One round of [`Self::cell_sites`]' fixed point: the cells and the
    /// candidate folds the round before no longer supports.
    fn cell_round(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        folds: &[MorphismId],
        cells: &SecondaryMap<MorphismId, Cell>,
        devgen: bool,
    ) -> (Vec<MorphismId>, Vec<MorphismId>, Vec<Carry>) {
        {
            let mut feeders: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for (m, site) in sites.iter() {
                let src = self.ir.morphism(m).expect("site").source;
                for slot in site.reads() {
                    if let Some(f) = self.pair_source(src, slot) {
                        feeders.insert(f, ());
                    }
                }
            }
            let carries = self.carries(sites, cells);
            let carried = |o: ObjectId| {
                carries
                    .iter()
                    .any(|c| c.views.contains(&o) || c.exit == Some(o))
            };
            let device_only = |o| carried(o) || self.produced_on_device(sites, &cells, o);
            // ponytail: an `iota` operand of a nest cell (non-empty `nested`,
            // so only under `nest-cell`; plain cells keep the rule above) is
            // uploaded once at its producer (`n·4` bytes, outside the loop
            // where the rewriter hoists it); a device iota kernel if one ever
            // sits in a loop cone. Not narrowed to `nested` slots: a fold in a
            // guard arm walks its range through the arm's input, which
            // `nest_reads` does not attribute (causal's `s` over `krange`).
            let iota = |o: ObjectId| {
                self.ir.in_edges(o).iter().any(|&p| {
                    self.ir.morphism(p).is_some_and(|x| x.op == Operation::Iota)
                })
            };
            let there = |o| {
                device_only(o)
                    || feeders.contains_key(o)
                    || (devgen
                        && self
                            .ir
                            .in_edges(o)
                            .iter()
                            .any(|&p| self.gen_law(p).is_some()))
            };
            let drop: Vec<MorphismId> = cells
                .iter()
                .filter(|(m, c)| {
                    let ops = c.operands();
                    let t = self.ir.morphism(*m).expect("cell").target;
                    let host = || self.uses(sites, cells, t, &[]).contains(&Use::Host);
                    let walked = |o| !c.nested.is_empty() && iota(o);
                    !ops.iter().all(|&(o, _)| there(o) || walked(o))
                        || !(ops.iter().any(|&(o, _)| device_only(o)) || !host())
                })
                .map(|(m, _)| m)
                .collect();
            let unfed: Vec<MorphismId> = folds
                .iter()
                .copied()
                .filter(|&m| sites.contains_key(m))
                .filter(|&m| {
                    let src = self.ir.morphism(m).expect("fold").source;
                    let a = sites[m].reads()[0];
                    !self.pair_source(src, a).is_some_and(device_only)
                })
                .collect();
            (drop, unfed, carries)
        }
    }

    /// One thread per cell: `work` flops each (a nested fold's `k`), every
    /// read and the store.
    fn cell_roofline(&self, m: MorphismId, site: &CellSite) -> (u64, u64) {
        let t = self.ir.morphism(m).expect("cell").target;
        let w = crate::profile::elem_bytes(array_elem(&self.obj_ty(t)));
        let bytes = (site.reads.len() as u64 + 1)
            .saturating_mul(site.extent)
            .saturating_mul(w);
        (site.extent.saturating_mul(site.work), bytes)
    }

    /// S62 `place` for a per-cell map: on the device when an operand is
    /// device-produced, or when one cell per thread (`extent` flops, every
    /// read and the store) finishes there first — [`gpu::placement`]'s rule.
    ///
    /// A cell that reads no array is a closed form of its index and uniforms:
    /// `devgen`'s rule decides it (built where every reader is), so it is a
    /// candidate and [`Self::cell_round`] keeps it when no host reads it whole.
    fn cell_pays(&self, m: MorphismId, site: &CellSite) -> bool {
        let t = self.ir.morphism(m).expect("cell").target;
        let (flops, bytes) = self.cell_roofline(m, site);
        (site.reads.is_empty() && !site.uniform.is_empty())
            || gpu::device_made(self.ir, &self.device, t)
            || self.profile.device_pays(flops, bytes)
    }

    /// Build the plan. `None` when the function has no device site or cell.
    pub(crate) fn resident_plan(&self, rungs: crate::GpuRungs) -> Option<ResidentPlan> {
        let (devgen, arena) = (rungs.devgen, rungs.arena);
        let mut sites = self.device_sites();
        let folds = self.fold_candidates(&mut sites);
        let (cells, carries) = self.cell_sites(&mut sites, &folds, devgen);
        if sites.is_empty() && cells.is_empty() {
            return None;
        }
        let mut plan = ResidentPlan {
            carries,
            ..ResidentPlan::default()
        };
        for (m, site) in sites.iter() {
            if let DevSite::RowFold(r) = site {
                plan.rowfold.insert(m, r.clone());
            }
        }
        let mut device = Vec::new(); // every object given a device copy here

        // Device outputs — sites', cell kernels' (`devmap`), an epilogue's
        // stored results (S54): kept on the device unless the host reads one whole.
        let outs: Vec<ObjectId> = sites
            .keys()
            .chain(cells.keys())
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
                self.uses(&sites, &cells, c, &[])
            };
            if uses.iter().all(|u| !matches!(u, Use::Host)) {
                plan.keep.insert(c, ());
            }
        }
        for carry in &plan.carries {
            if let Some(x) = carry.exit
                && self
                    .uses(&sites, &cells, x, &[])
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
            for u in self.uses(&sites, &cells, o, &[]) {
                if let Use::Elem(m) = u {
                    plan.elem.insert(m, ());
                }
            }
        }

        // The operands device sites read (their feeders).
        let mut read: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for (m, site) in sites.iter() {
            let src = self.ir.morphism(m).expect("site").source;
            for slot in site.reads() {
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
        for c in cells.values() {
            for (o, _) in c.operands() {
                read.insert(o, ());
            }
        }
        // S65 rule R: a device carry's init array that dies at the loop entry
        // is observed by the device only (the init product forwards a
        // pointer), so a closed-form one is built there — the entry then
        // hands its buffer to the carry (`resident_loop_enter`).
        let mut carried_init: Vec<ObjectId> = Vec::new();
        for c in &plan.carries {
            let Some(lp) = self.ir.loop_plan(self.f, c.merge) else {
                continue;
            };
            let Some(a) = self.pair_source(lp.init, c.k) else {
                continue;
            };
            let kind = self.ir.object(a).expect("object resolves").kind;
            let enter = self.ir.out_edges(lp.init).first().copied();
            if self.is_view(lp.init, &[c.k])
                && self.ir.out_edges(a).len() == 1
                && kind != ObjectKind::Constant
                && kind != ObjectKind::Parameter
                && self.ir.try_owner(a) == Some(self.f)
                && enter
                    .and_then(|e| self.lup.position(e))
                    .is_some_and(|p| self.lup.dead_after(a, p))
            {
                read.insert(a, ());
                carried_init.push(a);
            }
        }
        let carried = |o: ObjectId| plan.carries.iter().any(|c| c.views.contains(&o));
        for (o, _) in read.iter() {
            if carried(o) || self.produced_on_device(&sites, &cells, o) {
                continue;
            }
            let Some(&p) = self.ir.in_edges(o).first() else {
                continue; // a parameter: copied in at the site
            };
            let init = carried_init.contains(&o);
            if devgen
                && let Some(body) = self.gen_law(p)
                && (init
                    || self
                        .uses(&sites, &cells, o, &[])
                        .iter()
                        .all(|u| matches!(u, Use::Dev | Use::Elem(_) | Use::None)))
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
                    .uses(&sites, &cells, o, &[])
                    .iter()
                    .any(|u| matches!(u, Use::Elem(_)))
                {
                    plan.keep.insert(o, ());
                    for u in self.uses(&sites, &cells, o, &[]) {
                        if let Use::Elem(m) = u {
                            plan.elem.insert(m, ());
                        }
                    }
                }
            } else if init {
                continue; // host-built: the loop entry copies it in
            } else {
                plan.upload.insert(o, ());
            }
            device.push(o);
        }
        if rungs.delta_h2d {
            plan.delta = self.delta_uploads(&plan.upload);
        }

        // `cell-law`: a closed-form producer whose every reader evaluates it
        // in a cell (an unloaded slot) is built nowhere.
        if cells.values().any(|c| !c.laws.is_empty()) {
            for (m, _) in self.ir.morphisms() {
                let t = self.ir.morphism(m).expect("morphism resolves").target;
                if self.ir.try_owner(t) == Some(self.f)
                    && !self.lup.escapes(t)
                    && self.gen_law(m).is_some()
                    && !self.ir.out_edges(t).is_empty()
                    && self
                        .uses(&sites, &cells, t, &[])
                        .iter()
                        .all(|u| *u == Use::None)
                {
                    plan.skip.insert(m, ());
                }
            }
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
                    Operation::Map { captures: 0, .. } => {
                        plan.devgen.contains_key(c) || plan.skip.contains_key(c)
                    }
                    Operation::Pair { slot, .. } => {
                        self.pointer_only_array_component(cm.target, slot)
                            && self
                                .ir
                                .out_edges(cm.target)
                                .iter()
                                .all(|&m2| self.slot_use(&sites, &cells, m2, slot) == Use::None)
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
        plan.cells = cells;
        self.chains(&sites, &mut plan);
        // `flow`: the placed folds and cells, grouped.
        self.flow_units(&mut plan);
        // ponytail: the arena reuses bytes by topo position, which concurrent
        // streams do not follow; a function that opens streams keeps its own
        // buffers. Color per stream when a part ships both on one program.
        if arena && plan.chain.is_empty() {
            plan.arena = Some(self.arena(&plan));
        }
        let free = self.carry_host_free(&plan);
        if !free.is_empty() {
            let a = plan.arena.get_or_insert_with(Arena::default);
            for o in free {
                a.host_free.insert(o, ());
            }
        }
        Some(plan)
    }

    /// **S54 R7: device sites on streams.** Straight-line device sites (a site
    /// in a loop or guard arm stays on the legacy stream) are covered by
    /// chains in topo order: a site continues the chain whose last site is one
    /// of its `PathPlan` ancestors, else opens one. Two chains are opened as
    /// streams only when
    /// - every one of these sites' outputs stays on the device (`keep`: read
    ///   by device sites and one-element copies only). A whole copy-out puts
    ///   the host between the chains: it waits for the transfer (at 512², five
    ///   to ten times one kernel) before the next launch. In the first two
    ///   sessions, diamond and mlp2 with a host join ran 0.60–0.96× on streams on
    ///   every part, and indep flipped sign between sessions
    ///   (`streams-<part>.txt`, `-confirm.txt`); and
    /// - the grids can actually share the part — the **wave rule**
    ///   ([`waves_merge`]): two independent grids' last waves fit one wave,
    ///   `t_a / w_a + t_b / w_b ≤ 1` (a site's `w` is `sms × blocks_per_sm` of
    ///   its launch geometry). Grids that fill their last waves have nothing to
    ///   overlap.
    ///
    /// Otherwise the text stays the one-stream text. With streams open, the
    /// frees between the first and last chained launch (`cuMemFree` waits for
    /// the whole device) move after the last.
    fn chains(&self, sites: &SecondaryMap<MorphismId, DevSite>, plan: &mut ResidentPlan) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let (Some(pp), true) = (self.path_plan, rungs.streams) else {
            return;
        };
        let gpu = self.profile.gpu().expect("a device site has a GPU profile");
        let line = self.straight_line();
        let task_of = |m: MorphismId| {
            pp.tasks
                .iter()
                .position(|t| matches!(t.kind, TaskKind::Split { site, .. } if site == m))
        };
        let tile = |m: MorphismId| match &sites[m] {
            DevSite::Tile(t) => Some(t),
            DevSite::RowFold(_) => None,
        };
        let mut order: Vec<(u32, MorphismId, usize)> = sites
            .keys()
            .filter(|&m| line.contains_key(m) && tile(m).is_some())
            .filter_map(|m| Some((self.lup.position(m)?, m, task_of(m)?)))
            .collect();
        order.sort();
        let device_only = order.iter().all(|&(_, m, _)| {
            plan.keep
                .contains_key(self.ir.morphism(m).expect("site").target)
        });
        if order.len() < 2 || !device_only {
            return;
        }
        // S54 integration: a launch the chains do not order — a device site off
        // the chains (a fold, a loop's site) or a cell kernel (`devmap`) — must
        // not read a chained output; they run on the legacy stream, which the
        // chains' non-blocking streams do not wait for.
        let chained: Vec<ObjectId> = order
            .iter()
            .map(|&(_, m, _)| self.ir.morphism(m).expect("site").target)
            .collect();
        let on_chain = |o: ObjectId| {
            let o = plan.alias(o);
            let writer = grouped_writer(&self.groups, o)
                .map_or(o, |m| self.ir.morphism(m).expect("group site").target);
            chained.contains(&writer)
        };
        let reads_chained = |src: ObjectId, slots: &[u32]| {
            slots
                .iter()
                .any(|&k| self.pair_source(src, k).is_some_and(on_chain))
        };
        let unordered = sites.iter().any(|(m, s)| {
            !order.iter().any(|&(_, o, _)| o == m)
                && reads_chained(self.ir.morphism(m).expect("site").source, &s.reads())
        }) || plan
            .cells
            .values()
            .any(|c| c.operands().iter().any(|&(o, _)| on_chain(o)));
        if unordered {
            return;
        }
        // Every task's ancestors (the deps, transitively).
        let mut anc: Vec<Option<std::collections::BTreeSet<usize>>> = vec![None; pp.tasks.len()];
        fn ancestors(
            pp: &PathPlan,
            t: usize,
            memo: &mut Vec<Option<std::collections::BTreeSet<usize>>>,
        ) -> std::collections::BTreeSet<usize> {
            if let Some(s) = &memo[t] {
                return s.clone();
            }
            let mut s = std::collections::BTreeSet::new();
            for &d in &pp.tasks[t].deps {
                s.insert(d);
                s.extend(ancestors(pp, d, memo));
            }
            memo[t] = Some(s.clone());
            s
        }
        let anc: Vec<_> = order
            .iter()
            .map(|&(_, _, t)| ancestors(pp, t, &mut anc))
            .collect();
        let before = |i: usize, j: usize| anc[j].contains(&order[i].2);
        let mut chain = Vec::with_capacity(order.len());
        let mut tail: Vec<usize> = Vec::new();
        for i in 0..order.len() {
            match (0..i).rev().find(|&j| before(j, i) && tail[chain[j]] == j) {
                Some(j) => {
                    chain.push(chain[j]);
                    tail[chain[j]] = i;
                }
                None => {
                    chain.push(tail.len());
                    tail.push(i);
                }
            }
        }
        if tail.len() < 2 {
            return;
        }
        let grid = |i: usize| {
            let l = gpu::launch(
                tile(order[i].1).expect("a tile site"),
                self.profile,
                self.contract,
                self.gpu_arms,
            );
            (l.gx * l.gy, l.wave(gpu))
        };
        let merge = (0..order.len()).any(|i| {
            (i + 1..order.len()).any(|j| {
                chain[i] != chain[j]
                    && !before(i, j)
                    && !before(j, i)
                    && waves_merge(grid(i), grid(j))
            })
        });
        if rungs.stream_gate && !merge {
            return;
        }
        for (i, &(_, m, _)) in order.iter().enumerate() {
            plan.chain.insert(m, chain[i] as u32 + 1);
        }
        // `chain-h2d`: an upload read only by one chain's sites (through the
        // pointer-only products that carry it) crosses on that chain's
        // stream, so a chain the `PathPlan` proves independent never waits
        // for it — the copy overlaps the other chains' kernels.
        if rungs.chain_h2d {
            for o in plan.upload.keys().collect::<Vec<_>>() {
                let mut readers = Vec::new();
                for &m in self.ir.out_edges(o) {
                    match self.ir.morphism(m).expect("morphism resolves").op {
                        Operation::Pair { slot, .. } => {
                            let t = self.ir.morphism(m).expect("m").target;
                            if !self.pointer_only_array_component(t, slot) {
                                readers.push(None);
                                continue;
                            }
                            readers.extend(self.ir.out_edges(t).iter().map(|&r| plan.chain.get(r).copied()));
                        }
                        _ => readers.push(plan.chain.get(m).copied()),
                    }
                }
                if let Some(&Some(k)) = readers.first()
                    && readers.iter().all(|r| *r == Some(k))
                {
                    plan.upload_chain.insert(o, k);
                }
            }
        }
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let (lo, last) = (order[0].0, order[order.len() - 1].1);
        let moved: Vec<MorphismId> = plan
            .free_after
            .keys()
            .filter(|&d| d != last && (lo..pos(last)).contains(&pos(d)))
            .collect();
        for d in moved {
            let objs = plan.free_after.remove(d).expect("listed");
            match plan.free_after.get_mut(last) {
                Some(v) => v.extend(objs),
                None => {
                    plan.free_after.insert(last, objs);
                }
            }
        }
    }

    /// S54 `arena` (R6). Device side: the objects `free_after` frees — produced
    /// once and dead at a morphism that runs once — colored by their
    /// `[producer, death]` positions; anything used from a guard arm stays out
    /// (its arm runs at the `Phi`, past its position). Host side: an array no
    /// host morphism writes or reads whole — kept, device-built or never built
    /// — has no host slot. Carry objects keep both: the loop state holds the
    /// array inline.
    fn arena(&self, plan: &ResidentPlan) -> Arena {
        let gated_use = |o: ObjectId| {
            self.ir.out_edges(o).iter().any(|&m| {
                self.gated.contains_key(m)
                    || self
                        .ir
                        .out_edges(self.ir.morphism(m).expect("m").target)
                        .iter()
                        .any(|&m2| self.gated.contains_key(m2))
            })
        };
        let members: Vec<ObjectId> = plan
            .free_after
            .values()
            .flatten()
            .copied()
            .filter(|&o| !gated_use(o))
            .collect();
        // An epilogue member's output (S54 R1) is written by its site's launch,
        // which reads the site's operands: it is born at the site, not at its
        // own (later) map — else it could take the bytes of an operand that dies
        // at the site while the kernel still reads it (S54 integration:
        // px_bias_100's `d` took `a`'s offset and the checksum raced).
        let live: Vec<(u32, u32, u64)> = members
            .iter()
            .map(|&o| {
                let birth = self
                    .ir
                    .in_edges(o)
                    .iter()
                    .copied()
                    .chain(grouped_writer(&self.groups, o))
                    .chain(plan.unit_writer(o))
                    .filter_map(|p| self.lup.position(p))
                    .min();
                let death = self.lup.death(o);
                let bytes = self.array_bytes(&self.obj_ty(o));
                (
                    birth.expect("produced"),
                    death.expect("freed at death"),
                    bytes,
                )
            })
            .collect();
        let (offs, bytes) = color(&live);
        let mut a = Arena {
            bytes,
            ..Arena::default()
        };
        for (o, off) in members.into_iter().zip(offs) {
            a.offset.insert(o, off);
        }
        let built = plan.devgen.keys().chain(plan.skip.keys());
        let built: Vec<ObjectId> = built
            .map(|m| self.ir.morphism(m).expect("m").target)
            .collect();
        // An epilogue member (S54 R1) is skipped on the host too, but a stored
        // one not kept is copied out to a host reader: it keeps its slot.
        let copied_out = |o: ObjectId| {
            !plan.keep.contains_key(o)
                && (self
                    .groups
                    .values()
                    .any(|g| g.chain.iter().any(|mb| mb.out == o && mb.store))
                    || plan.unit_writer(o).is_some())
        };
        for o in plan.keep.keys().chain(built) {
            if !plan.is_carry_next(o)
                && plan.alias(o) == o
                && !self.lup.escapes(o)
                && self.ir.try_owner(o) == Some(self.f)
                && !copied_out(o)
            {
                a.host_free.insert(o, ());
            }
        }
        a
    }

    /// S65 `by-ref`: a device carry whose loop products forward a pointer has
    /// no host bytes — its views, its next state, and its exit object unless
    /// the host reads that whole (then copied out into its own). With or
    /// without the arena: the host state holds no bytes to point at.
    fn carry_host_free(&self, plan: &ResidentPlan) -> Vec<ObjectId> {
        let mut out = Vec::new();
        for c in &plan.carries {
            if !self.is_view(c.merge, &[c.k]) {
                continue;
            }
            let exit = c.exit.filter(|x| plan.keep.contains_key(*x));
            for o in c.views.iter().copied().chain([c.next]).chain(exit) {
                if !self.lup.escapes(o) && self.ir.try_owner(o) == Some(self.f) {
                    out.push(o);
                }
            }
        }
        out
    }

    /// `o` has no host slot (S54 `arena`).
    pub(super) fn host_free(&self, o: ObjectId) -> bool {
        self.resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .is_some_and(|a| a.host_free.contains_key(o))
    }

    fn arena_offset(&self, o: ObjectId) -> Option<(u64, u64)> {
        let a = self.resident.as_ref()?.arena.as_ref()?;
        Some((a.bytes, *a.offset.get(o)?))
    }

    /// **S54 `l2-window`: the buffer of site `m`'s launch the graph proves is
    /// read again soonest**, and its bytes. The output, when the next device
    /// site by `LastUsePlan` position reads it (a chain's intermediate: reuse
    /// distance one launch — its stores persist for the reader); else the
    /// operand whose panels the most blocks read (`a` row-panel: `gx` blocks,
    /// `b` column-panel: `gy`; on a tie `b`, whose readers row-major order puts
    /// `gx` launches apart). Only a buffer whose bytes fit the part's `l2_bytes`:
    /// a window larger than the cache it reserves persists nothing whole.
    pub(super) fn l2_window(
        &self,
        m: MorphismId,
        site: &TileSite,
        l: &gpu::Launch,
    ) -> Option<(Window, u64)> {
        let l2 = self.profile.gpu()?.l2_bytes;
        let morph = self.ir.morphism(m).expect("site");
        let out = self.array_bytes(&self.obj_ty(morph.target));
        let pos = |m| self.lup.position(m);
        let next = self
            .device_sites()
            .iter()
            .filter(|&(m2, _)| pos(m2) > pos(m))
            .min_by_key(|&(m2, _)| pos(m2))
            .map(|(m2, s2)| {
                let src = self.ir.morphism(m2).expect("site").source;
                s2.reads()
                    .into_iter()
                    .any(|k| self.pair_source(src, k) == Some(morph.target))
            });
        if next == Some(true) && out <= l2 {
            return Some((Window::Out, out));
        }
        let src_ty = self.obj_ty(morph.source);
        let bytes = |slot| self.array_bytes(src_ty.component_ty(slot).expect("tile operand"));
        [
            (Window::B, l.gy, bytes(site.b.slot)),
            (Window::A, l.gx, bytes(site.a.slot)),
        ]
        .into_iter()
        .filter(|&(_, _, b)| b <= l2)
        .max_by_key(|&(w, reuse, _)| (reuse, w == Window::B))
        .map(|(w, _, b)| (w, b))
    }

    /// A device site's, a cell kernel's (`devmap`) or an epilogue member's
    /// output (S54).
    fn produced_on_device(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
        o: ObjectId,
    ) -> bool {
        self.ir.in_edges(o).iter().any(|&m| {
            sites.contains_key(m) || cells.contains_key(m) || self.member_src(m, 0).is_some()
        })
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
    pub(super) fn dalloc(&mut self, o: ObjectId, bytes: u64) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        // An arena member: base + offset, the arena made on the first ask
        // (lazily, so `eager-init` still overlaps the host work before it) —
        // or at entry (`entry-arena`), where the slot already holds its base.
        if let Some((total, off)) = self.arena_offset(o) {
            let base = self.tmp();
            if gpu::host_rungs(self.profile, self.contract, self.gpu_arms).entry_arena {
                self.line(format!("{base} = load i64, ptr %darena"));
            } else {
                self.line(format!(
                    "{base} = call i64 @mapal_gpu_arena(ptr %darena, i64 {total})"
                ));
            }
            self.line(format!("{d} = add i64 {base}, {off}"));
            self.line(format!("store i64 {d}, ptr {slot}"));
            return d;
        }
        self.line(format!(
            "{d} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {bytes})"
        ));
        self.dallocs.push((slot, bytes));
        d
    }

    pub(super) fn dload(&mut self, o: ObjectId) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        self.line(format!("{d} = load i64, ptr {slot}"));
        d
    }

    pub(super) fn array_bytes(&self, ty: &Ty) -> u64 {
        array_len(ty) * crate::profile::elem_bytes(array_elem(ty))
    }

    /// Replaces the host realization of `m` (a device-built or unread
    /// producer). `true` when `m` was handled here.
    pub(super) fn resident_instead(&mut self, m: MorphismId) -> bool {
        let Some(plan) = &self.resident else {
            return false;
        };
        if let Some(u) = plan.units.get(m).cloned() {
            self.emit_unit(&u);
            return true;
        }
        if plan.skip.contains_key(m) {
            return true;
        }
        if let Some(r) = plan.rowfold.get(m).cloned() {
            self.emit_rowfold(m, &r);
            return true;
        }
        if let Some(c) = plan.cells.get(m).cloned() {
            self.emit_cell(m, &c);
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

    /// A cell's launch (`devmap`): its operands' device copies (copied in
    /// here only when nothing placed one, as a site does), one thread per
    /// cell, the output kept on the device unless the host reads it whole.
    fn emit_cell(&mut self, m: MorphismId, c: &Cell) {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let (source, target) = (morph.source, morph.target);
        let bytes = self.array_bytes(&self.obj_ty(target));
        let mut ptrs = vec![self.dalloc(target, bytes)];
        let mut temps = Vec::new();
        for (o, slot, uniform) in c.args() {
            if let (true, Some(u)) = (uniform, slot) {
                let (t, v) = self.load_component(source, u).expect("uniform loads");
                let w = self.tmp();
                self.line(gpu::uniform_to_i64(&w, &v, &t));
                ptrs.push(w);
                continue;
            }
            let at = self.resident.as_ref().map_or(o, |p| p.alias(o));
            if self.dvalid.contains_key(at) {
                ptrs.push(self.dload(o));
                continue;
            }
            let ob = self.array_bytes(&self.obj_ty(o));
            let h = self
                .array_operand_ptr(source, slot)
                .expect("cell operand ptr");
            let s = format!("%d{}", self.fresh());
            self.allocas
                .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
            let d = self.tmp();
            self.line(format!(
                "{d} = call i64 @mapal_gpu_alloc(ptr {s}, i64 {ob})"
            ));
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {ob})"
            ));
            temps.push(s);
            ptrs.push(d);
        }
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
        let (gx, block) = gpu::cell_grid(c.n);
        self.line(format!(
            "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 0)",
            c.id as i64
        ));
        self.dvalid.insert(target, ());
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        let keep = self
            .resident
            .as_ref()
            .is_some_and(|p| p.keep.contains_key(target) || p.is_carry_next(target));
        if !keep {
            let h = self.slot(target).expect("cell output slot");
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h}, i64 {}, i64 0, i64 {bytes})",
                ptrs[0]
            ));
        }
    }

    /// The device kernels of this function's cells, each with the functions
    /// it calls (name, text): the body — `cell-law`'s variant of it when a
    /// capture is evaluated — and each evaluated capture's law.
    /// Every row fold this function runs on the device (the device unit's
    /// kernels for the ones `rowfold::rowfold_sites` does not see).
    pub(crate) fn rowfold_kernels(&self) -> Vec<mapal_ir::RowFoldSite> {
        self.resident
            .as_ref()
            .map_or_else(Vec::new, |p| p.rowfold.values().map(|r| r.site.clone()).collect())
    }

    pub(crate) fn cell_kernels(&self) -> Vec<gpu::CellKernel> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.cells
            .iter()
            .map(|(m, c)| {
                let morph = self.ir.morphism(m).expect("cell morphism");
                let (out, _) = array_parts(&self.obj_ty(morph.target));
                let input = self.obj_ty(self.ir.func(c.body).expect("cell body").input);
                let arg =
                    lower_body_input_ty(&input, c.caps.len() as u32).expect("cell input lowers");
                let elem = c.mapped.map(|o| array_parts(&self.obj_ty(o)).0);
                let vals: Vec<(u32, String)> = c
                    .laws
                    .iter()
                    .map(|&(s, _)| (s, array_parts(&self.obj_ty(c.caps[s as usize])).0))
                    .collect();
                let name = &self.fnames[c.body];
                let mut fns = vec![(
                    if vals.is_empty() {
                        name.clone()
                    } else {
                        gpu::cell_variant(name)
                    },
                    self.body_text(c.body, vals.clone()),
                )];
                let caps: Vec<gpu::CapArg> = (0..c.caps.len() as u32)
                    .map(|s| {
                        if c.uniform.contains(&s) {
                            let t = lower_ty(&self.obj_ty(c.caps[s as usize]))
                                .expect("a number uniform lowers");
                            return gpu::CapArg::Uniform(t);
                        }
                        let Some(&(_, law)) = c.laws.iter().find(|&&(x, _)| x == s) else {
                            return gpu::CapArg::Ptr;
                        };
                        fns.push((self.fnames[law].clone(), self.body_text(law, Vec::new())));
                        let t = vals.iter().find(|&&(x, _)| x == s).expect("law value").1.clone();
                        gpu::CapArg::Law(self.fnames[law].clone(), t)
                    })
                    .collect();
                let laws = c.laws.iter().map(|&(_, l)| l);
                for nb in std::iter::once(c.body).chain(laws).flat_map(|b| nested_bodies(self.ir, b)) {
                    fns.push((self.fnames[nb].clone(), self.body_text(nb, Vec::new())));
                }
                // The facts the kernel was built from, for its `; mapal.plan`.
                let work = self.ir.cell_plan(self.f).get(m).map_or(1, |x| x.work);
                let kernel = format!(
                    "; mapal.cell nested={} work={work}\n{}",
                    c.nested.len(),
                    gpu::cell_kernel_text(c.id, &fns[0].0, c.n, &out, &arg, elem.as_deref(), &caps)
                );
                (fns, kernel)
            })
            .collect()
    }

    /// `f`'s text as the module emits it (a leaf body, so the text is a pure
    /// function of the IR), or `cell-law`'s device variant of it.
    fn body_text(&self, f: FuncId, vals: Vec<(u32, String)>) -> String {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        fe.cell_vals = vals;
        fe.emit()
    }

    /// After `m`: copy its output in if a device site reads it, and free every
    /// buffer whose death is `m`.
    pub(super) fn resident_after(&mut self, m: MorphismId) {
        let Some(plan) = &self.resident else {
            return;
        };
        let t = self.ir.morphism(m).expect("morphism resolves").target;
        let upload = plan.upload.contains_key(t) && !plan.devgen.contains_key(m);
        let delta = plan.delta.get(t).cloned();
        let frees = plan.free_after.get(m).cloned().unwrap_or_default();
        let chain = plan.upload_chain.get(t).copied();
        if upload {
            let ty = self.obj_ty(t);
            let bytes = self.array_bytes(&ty);
            let h = self.array_operand_ptr(t, None).expect("uploaded array ptr");
            let d = self.dalloc(t, bytes);
            // `pinned` (S54): a copy that runs more than once is page-locked.
            self.pin(m, &h, bytes, true);
            match (delta, chain) {
                (Some(du), _) => self.emit_delta_h2d(t, &du, &d, &h, bytes),
                (None, Some(k)) => self.line(format!(
                    "call void @mapal_gpu_h2d_on(i32 {k}, i64 {d}, ptr {h}, i64 {bytes})"
                )),
                (None, None) => self.line(format!(
                    "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
                )),
            }
            self.dvalid.insert(t, ());
        }
        for o in frees {
            if self.arena_offset(o).is_some() {
                continue; // its bytes are the arena's, released at exit
            }
            let slot = self.dslot(o);
            self.line(format!("call void @mapal_gpu_free(ptr {slot})"));
            self.dvalid.remove(o);
        }
    }

    /// `delta-h2d`: after a loop entry's first (whole) upload, only the rows
    /// the iteration wrote cross — `stride` cells from `start`, at the same
    /// byte offset on both sides. Every executed `Update` was in bounds, so
    /// the range is too.
    fn emit_delta_h2d(&mut self, t: ObjectId, du: &DeltaUp, d: &str, h: &str, bytes: u64) {
        let eb = crate::profile::elem_bytes(array_elem(&self.obj_ty(t)));
        let flag = self.retain_flag(t);
        let whole = format!("call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})");
        let Some((wllt, tv)) = self.load_whole(du.start.var) else {
            self.line(whole);
            return;
        };
        let valid = self.tmp();
        self.line(format!("{valid} = load i1, ptr {flag}"));
        let start = self.affine_cell(&wllt, &tv, &du.start);
        // `Δ` is a claim about cells: rows past the array (a trip that writes
        // nothing) cross whole. Unsigned: a start below zero fails too. (A
        // delta upload is never wider than its array: `delta_uploads`.)
        let fits = self.tmp();
        let last = array_len(&self.obj_ty(t)) - du.stride;
        self.line(format!("{fits} = icmp ule i64 {start}, {last}"));
        let go = self.tmp();
        self.line(format!("{go} = and i1 {valid}, {fits}"));
        let (rows, first, done) = (self.label(), self.label(), self.label());
        self.line(format!("br i1 {go}, label %{rows}, label %{first}"));
        self.label_line(&first);
        self.line(whole);
        self.line(format!("store i1 true, ptr {flag}"));
        self.line(format!("br label %{done}"));
        self.label_line(&rows);
        let off = self.tmp();
        self.line(format!("{off} = mul i64 {start}, {eb}"));
        let dd = self.tmp();
        self.line(format!("{dd} = add i64 {d}, {off}"));
        let hp = self.tmp();
        self.line(format!("{hp} = getelementptr i8, ptr {h}, i64 {off}"));
        self.line(format!(
            "call void @mapal_gpu_h2d(i64 {dd}, ptr {hp}, i64 {})",
            du.stride * eb
        ));
        self.line(format!("br label %{done}"));
        self.label_line(&done);
    }

    /// A retained site's restricted pass under `resident` (S54
    /// `retain-resident`): the one-row site `one` reads row `row`'s `a` span
    /// in place in `a`'s device copy and stores into row `row` of the output's
    /// resident buffer. A host reader of the output gets that row only: its
    /// host copy is retained the same way. An operand with no device copy
    /// here runs the whole site instead (always correct).
    pub(super) fn emit_row_resident(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        row: &str,
        call: impl Fn(&str, &str, &str) -> String,
    ) -> bool {
        let alias = |p: &Option<ResidentPlan>, o: ObjectId| p.as_ref().map_or(o, |p| p.alias(o));
        let feeders: Vec<ObjectId> = [site.a.slot, site.b.slot]
            .iter()
            .filter_map(|&s| self.pair_source(source, s))
            .filter(|&f| self.dvalid.contains_key(alias(&self.resident, f)))
            .collect();
        let &[fa, fb] = feeders.as_slice() else {
            return false;
        };
        let eb = crate::profile::elem_bytes(&site.elem);
        let (a, b) = (self.dload(fa), self.dload(fb));
        let a_cells = self.tmp();
        self.line(format!("{a_cells} = mul i64 {row}, {}", site.a.ci));
        let a_cell = self.tmp();
        self.line(format!("{a_cell} = add i64 {a_cells}, {}", site.a.base));
        let a_off = self.tmp();
        self.line(format!("{a_off} = mul i64 {a_cell}, {eb}"));
        let a_row = self.tmp();
        self.line(format!("{a_row} = add i64 {a}, {a_off}"));
        let o = self.dload(target);
        let o_off = self.tmp();
        self.line(format!("{o_off} = mul i64 {row}, {}", site.c * eb));
        let o_row = self.tmp();
        self.line(format!("{o_row} = add i64 {o}, {o_off}"));
        self.line(call(&a_row, &b, &o_row));
        let keep = self
            .resident
            .as_ref()
            .is_some_and(|p| p.keep.contains_key(target));
        if !keep {
            let h = self.slot(target).expect("tile output slot");
            let h_row = self.tmp();
            self.line(format!("{h_row} = getelementptr i8, ptr {h}, i64 {o_off}"));
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h_row}, i64 {o}, i64 {o_off}, i64 {})",
                site.c * eb
            ));
        }
        true
    }

    /// The device half of a site under `resident`: operands from their device
    /// copies (copied in here only when no producer did), output kept on the
    /// device, copied out whole only when the host reads it whole.
    pub(super) fn emit_site_resident(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        slots: &[u32],
        // `(kernel id, geometry)` of a tile site: the `launchv` it takes when
        // its store evaluates an epilogue group (S54 R1) or it runs on a
        // chain's stream (S54 R7) instead of `call`'s launch.
        grouped: Option<(i64, String)>,
        call: impl Fn(&[String], &str) -> String,
    ) {
        let src_ty = self.obj_ty(source);
        let ms = self.ir.in_edges(target)[0];
        let mut ops = Vec::new();
        let mut temps = Vec::new();
        for &slot in slots {
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
            self.pin(ms, &h, bytes, true);
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
            ));
            temps.push(slot_name);
            ops.push(d);
        }
        let plan = self.resident.clone().expect("resident site");
        let s = plan.stream_of(self.ir, &self.groups, target);
        if s != 0 && self.first_chained(&plan, target) {
            self.chain_prologue(&plan);
        }
        let o_bytes = self.array_bytes(&self.obj_ty(target));
        let m = self.ir.in_edges(target)[0];
        let group = grouped
            .clone()
            .and_then(|ig| Some((ig, self.groups.get(m).cloned()?)));
        // The values this launch leaves in `Gmem`: the site's output unless
        // its group holds every reader (S54), and each stored member result.
        let mut stored = Vec::new();
        let dout = if group.as_ref().is_none_or(|(_, g)| g.store_out) {
            let d = self.dalloc(target, o_bytes);
            stored.push((target, d.clone()));
            d
        } else {
            "0".to_string()
        };
        // A cross-chain edge (S54 R7): an operand on another chain's stream — a
        // site operand, or an epilogue's complete operand — is waited for.
        if !plan.chain.is_empty() {
            let mut feeds: Vec<ObjectId> = slots
                .iter()
                .filter_map(|&k| self.pair_source(source, k))
                .collect();
            if let Some((_, g)) = &group {
                feeds.extend(g.caps());
            }
            let mut waits: Vec<u32> = feeds
                .iter()
                .map(|&f| plan.stream_of(self.ir, &self.groups, f))
                .filter(|&p| p != 0 && p != s)
                .collect();
            waits.sort_unstable();
            waits.dedup();
            for p in waits {
                self.line(format!("call void @mapal_gpu_event(i32 {p}, i32 {s})"));
            }
        }
        // The pointers of a `launchv`: an epilogue group's, or a chained site's
        // own three.
        let ptrs = match (&group, &grouped) {
            (Some((_, g)), _) => {
                let mut ptrs = vec![ops[0].clone(), ops[1].clone(), dout.clone()];
                for o in g.caps() {
                    debug_assert!(self.dvalid.contains_key(plan.alias(o)), "epilogue operand in Gmem");
                    ptrs.push(self.dload(o));
                }
                for mb in g.chain.iter().filter(|mb| mb.store) {
                    let bytes = self.array_bytes(&self.obj_ty(mb.out));
                    let d = self.dalloc(mb.out, bytes);
                    ptrs.push(d.clone());
                    stored.push((mb.out, d));
                }
                Some(ptrs)
            }
            (None, Some(_)) if s != 0 => Some(vec![ops[0].clone(), ops[1].clone(), dout.clone()]),
            _ => None,
        };
        match (ptrs, grouped) {
            (Some(ptrs), Some((id, geom))) => {
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
                    "call void @mapal_gpu_launchv(i64 {id}, i32 {s}, i32 {n}, ptr {arr}, {geom})"
                ));
            }
            _ => self.line(call(&ops, &dout)),
        }
        for (o, _) in &stored {
            self.dvalid.insert(*o, ());
        }
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            let keep = plan.keep.contains_key(o) || plan.is_carry_next(o);
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("device output slot");
                self.pin(ms, &h, bytes, false);
                // A device fold's output lands in what was a host loop's slot: say
                // the copy keeps no pointer to it, as the loop did not, or the slot
                // counts as escaped and a later host loop that reads it loses its
                // vector form (S54: softmax_rows' `exp` pass ran scalar, +17 ms).
                // ponytail: every `d2h` is just as non-capturing; marked here only,
                // so no other program's text moves.
                let fold = self
                    .ir
                    .in_edges(o)
                    .first()
                    .is_some_and(|m| plan.rowfold.contains_key(*m));
                let h = if fold { format!("nocapture {h}") } else { h };
                self.d2h(s, &h, &d, "0", bytes);
            }
        }
    }

    /// A copy-out on stream `s`: the legacy entry on stream 0 (the one-stream
    /// text), else queued behind the chain's kernels and waited for.
    pub(super) fn d2h(&mut self, s: u32, h: &str, d: &str, off: &str, bytes: impl std::fmt::Display) {
        if s == 0 {
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 {off}, i64 {bytes})"
            ));
        } else {
            self.line(format!(
                "call void @mapal_gpu_d2h_on(i32 {s}, ptr {h}, i64 {d}, i64 {off}, i64 {bytes})"
            ));
        }
    }

    /// `pinned`: page-lock the host array a copy at `m` moves, when that copy
    /// runs more than once — `m` inside a loop (or guard arm), so one
    /// registration is paid for every crossing after it (the runtime pins an
    /// array once). A copy that runs once is not pinned: registering costs
    /// more than one pinned copy saves on every box measured
    /// (`streams-<part>.txt`, S54 trace + `copy_rate`: 0.06–0.09 ms per MB
    /// registered at 64 MB, 0.3–0.5 at 1 MB, against 0.002 (RTX 4090 box) to
    /// 0.08 ms per MB (RTX 5090 box) saved per crossing). An upload (`up`)
    /// under `async-h2d` is not pinned: the runtime would have to wait for a
    /// page-locked source it does not stage, and staging makes the pin moot.
    fn pin(&mut self, m: MorphismId, h: &str, bytes: u64, up: bool) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if rungs.pinned && !(up && rungs.async_h2d) && !self.straight_line().contains_key(m) {
            self.line(format!("call void @mapal_gpu_pin(ptr {h}, i64 {bytes})"));
        }
    }

    /// Is `target`'s site the first chained launch of the function?
    fn first_chained(&self, plan: &ResidentPlan, target: ObjectId) -> bool {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let first = plan.chain.keys().min_by_key(|&m| pos(m));
        first.is_some_and(|m| self.ir.morphism(m).expect("site").target == target)
    }

    /// Before the first chained launch: every stored chained output's buffer, so no
    /// `cuMemAlloc` (which may wait for the device) lands between two chains'
    /// launches.
    fn chain_prologue(&mut self, plan: &ResidentPlan) {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let mut sites: Vec<MorphismId> = plan.chain.keys().collect();
        sites.sort_by_key(|&m| pos(m));
        for m in sites {
            if self.groups.get(m).is_some_and(|g| !g.store_out) {
                continue;
            }
            let c = self.ir.morphism(m).expect("site").target;
            let bytes = self.array_bytes(&self.obj_ty(c));
            self.dalloc(c, bytes);
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
        let s = self
            .resident
            .as_ref()
            .expect("plan")
            .stream_of(self.ir, &self.groups, arr);
        let d = self.dload(arr);
        let off = self.tmp();
        self.line(format!("{off} = mul i64 {idx}, {eb}"));
        self.d2h(s, ep, &d, &off, eb);
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

    /// S65 movement intent, the loop header's first line: what the facts
    /// require to cross per trip — this plan's own derivation (uploads of
    /// host-produced arrays a device site reads, `delta-h2d` rows, copy-outs
    /// of device outputs a host morphism reads, one-element reads) — and how
    /// each state array travels (`frame.rs` `carried_desc`).
    /// `benches/gpu/evidence.py` counts what the host IR realizes against it.
    /// A loop with no array state off a GPU build moves nothing: no line.
    pub(crate) fn move_line(&self, plan: &mapal_ir::LoopPlan, header: &str) -> String {
        if array_paths(&self.obj_ty(plan.merge)).is_empty() && self.resident.is_none() {
            return String::new();
        }
        let bytes = |o: ObjectId| lower_ty(&self.obj_ty(o)).map_or(0, |l| llt_bytes(&l));
        let (mut h2d, mut d2h, mut retained) = (0u64, 0u64, 0usize);
        // An `Update` whose source is still read after it (or is not this
        // function's to write) copies it: the one host copy the facts require.
        let copy: u64 = plan
            .decide_order
            .iter()
            .chain(&plan.advance_order)
            .map(|&m| self.ir.morphism(m).expect("morphism resolves"))
            .filter(|mo| mo.op == Operation::Update && !self.update_aliases.contains_key(mo.target))
            .map(|mo| bytes(mo.target))
            .sum();
        if let Some(p) = &self.resident {
            // (A guard arm's work counts: the arm may run on every trip.)
            for &m in plan.decide_order.iter().chain(&plan.advance_order) {
                let t = self.ir.morphism(m).expect("morphism resolves").target;
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
                    d2h += bytes(t);
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
                } else if let Some(u) = p.units.get(m) {
                    outs.extend(u.members.iter().filter(|mb| mb.store).map(|mb| mb.out));
                }
                for o in outs {
                    if !p.keep.contains_key(o) && !p.is_carry_next(o) {
                        // A retained output (`retain.rs`) rebuilds whole once
                        // per loop entry; a trip copies out its one new row.
                        d2h += match self.retain.get(o) {
                            Some(r) => {
                                r.stride * crate::profile::elem_bytes(array_elem(&self.obj_ty(o)))
                            }
                            None => bytes(o),
                        };
                    }
                }
                retained += self.retain.contains_key(t) as usize;
            }
        }
        format!(
            "; mapal.move loop={header} carried={} h2d={h2d} d2h={d2h} host_copy={copy} alloc=0 retained={retained}\n",
            self.carried_desc(plan.merge)
        )
    }

    /// Loop entry: the carry's buffer gets the init state, copied in once.
    pub(crate) fn resident_loop_enter(&mut self, merge: ObjectId, init: ObjectId) {
        for c in self.carries_of(merge) {
            // S65: a device-built init array is the carry's first state — its
            // buffer moves to the carry (its slot is emptied, so one free).
            if let Some(a) = self.pair_source(init, c.k)
                && self.host_free(a)
                && self.dvalid.contains_key(a)
            {
                let (sa, sm) = (self.dslot(a), self.dslot(c.key()));
                let d = self.tmp();
                self.line(format!("{d} = load i64, ptr {sa}"));
                self.line(format!("store i64 {d}, ptr {sm}"));
                self.line(format!("store i64 0, ptr {sa}"));
                self.dvalid.remove(a);
                self.dvalid.insert(c.key(), ());
                continue;
            }
            let ty = self
                .obj_ty(merge)
                .component_ty(c.k)
                .cloned()
                .expect("carry ty");
            let bytes = self.array_bytes(&ty);
            let h = self.array_operand_ptr(init, Some(c.k)).expect("init ptr");
            let d = self.dalloc(c.key(), bytes);
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
            ));
            self.dvalid.insert(c.key(), ());
        }
    }

    /// The back edge: the next state's buffer becomes the carry's, and the old
    /// state's buffer (dead past the back edge) is the next launch's output.
    pub(crate) fn resident_loop_back(&mut self, merge: ObjectId) {
        for c in self.carries_of(merge) {
            let (cs, ns) = (self.dslot(c.key()), self.dslot(c.next));
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
            self.dvalid.insert(c.key(), ());
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
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let entry = rungs.arena && rungs.entry_arena && self.resident.is_some() && self.entry_arena();
        // `async-h2d`: every upload is enqueued without a host wait (the
        // runtime stages the source), so no upload is a sync point: the trip
        // waits only at a copy-out, which exists only where a host morphism
        // reads a device value (`upload`/`keep`/`elem` in the plan).
        if rungs.async_h2d && self.resident.is_some() {
            self.body = self
                .body
                .replace("call void @mapal_gpu_h2d(", "call void @mapal_gpu_h2d_async(");
        }
        // Pins end with the frame they pinned.
        if self.body.contains("@mapal_gpu_pin(") {
            self.line("call void @mapal_gpu_unpin()");
        }
        let slots: Vec<String> = self
            .dslots
            .iter()
            .filter(|&(o, _)| self.arena_offset(o).is_none())
            .map(|(_, s)| s.clone())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for s in slots.into_iter().chain(self.split_slots.clone()) {
            if seen.insert(s.clone()) {
                self.line(format!("call void @mapal_gpu_free(ptr {s})"));
            }
        }
        if self.arena_slot() || entry {
            self.line("call void @mapal_gpu_free(ptr %darena)");
        }
    }

    /// **`entry-arena`: the arena at function entry, loop-cone buffers in
    /// it.** Every slot the walk allocated with `mapal_gpu_alloc` (`dallocs`)
    /// that no `free_after` death frees (a loop-cone buffer, a carry's next, a
    /// split workspace) is placed after the straight-line members, each live
    /// over the whole function — `[0, ∞)` in the same coloring, so it shares
    /// no byte with anything: a back edge may swap it into another slot at any
    /// trip. The entry block makes the arena (static size) and seeds those
    /// slots, so their first-use allocation finds its buffer. A slot freed
    /// before exit keeps its own allocation. The runtime releases a member
    /// with its arena, whichever slot holds it at exit (`mapal_gpu_free`).
    /// Returns whether the function has an arena.
    ///
    /// ponytail: whole-function intervals, so the cones of two loops in one
    /// function never share bytes; per-loop spans (`loop_plan` through the
    /// death of each carry's exit alias) when that footprint matters.
    fn entry_arena(&mut self) -> bool {
        let freed: std::collections::BTreeSet<&String> = self
            .resident
            .iter()
            .flat_map(|p| p.free_after.values().flatten())
            .filter(|&&o| self.arena_offset(o).is_none())
            .filter_map(|&o| self.dslots.get(o))
            .collect();
        let mut cone: Vec<(String, u64)> = Vec::new();
        for (slot, bytes) in &self.dallocs {
            if freed.contains(slot) {
                continue;
            }
            match cone.iter_mut().find(|(s, _)| s == slot) {
                Some(c) => c.1 = c.1.max(*bytes),
                None => cone.push((slot.clone(), *bytes)),
            }
        }
        let line = self
            .resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .map_or(0, |a| a.bytes);
        let live: Vec<(u32, u32, u64)> = cone.iter().map(|&(_, b)| (0, u32::MAX, b)).collect();
        let (offs, cone_bytes) = color(&live);
        let total = line + cone_bytes;
        if total == 0 {
            return false;
        }
        let mut text = String::new();
        if line == 0 {
            text.push_str("  %darena = alloca i64\n  store i64 0, ptr %darena\n");
        }
        let base = self.tmp();
        text.push_str(&format!(
            "  {base} = call i64 @mapal_gpu_arena(ptr %darena, i64 {total})\n"
        ));
        for ((slot, _), off) in cone.iter().zip(offs) {
            let d = self.tmp();
            text.push_str(&format!(
                "  {d} = add i64 {base}, {}\n  store i64 {d}, ptr {slot}\n",
                line + off
            ));
        }
        self.allocas.push_str(&text);
        true
    }

    /// The function has arena members, so `%darena` exists.
    pub(crate) fn arena_slot(&self) -> bool {
        self.resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .is_some_and(|a| a.bytes > 0)
    }
}

/// The wave rule (S54 R7): grids of `(blocks, wave)` on separate streams take
/// one wave fewer than back to back exactly when their last waves — `b − w·(⌈b
/// / w⌉ − 1)` blocks each — fit in one: `t_a / w_a + t_b / w_b ≤ 1`. Two
/// grids under one wave each are the case `b = t`. Measured on device-only
/// sites (`twin`): the A100 at 2048, where two grids of ~1.2 waves run in 3
/// waves instead of 4, 21–27% faster in two sessions
/// (`streams-a100-confirm.txt`, `-final.txt`); where the rule says no (the
/// RTX 4090 and 5090 at 1024–2048) streams measured 3–6% slower. The first
/// build's 8–37% losses with this rule at 2048–4096 were programs with a host
/// join, which no longer open streams at all.
pub(crate) fn waves_merge((ba, wa): (u64, u64), (bb, wb): (u64, u64)) -> bool {
    let last = |b: u64, w: u64| b - w * (b.div_ceil(w) - 1);
    last(ba, wa) * wb + last(bb, wb) * wa <= wa * wb
}

pub(super) fn array_len(ty: &Ty) -> u64 {
    match ty {
        Ty::Array { size, .. } => *size,
        _ => 0,
    }
}

pub(super) fn array_elem(ty: &Ty) -> &Ty {
    match ty {
        Ty::Array { elem, .. } => elem,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{ARENA_ALIGN, color};

    #[test]
    fn grouped_stream_lookup_uses_aliases_and_keeps_unknowns_legacy() {
        use super::{Carry, Epi, Member, ResidentPlan};
        use mapal_ir::{FuncKind, IrBuilder, SourceLoc, Ty, Value};
        use slotmap::SecondaryMap;

        let loc = SourceLoc::empty_at(0);
        let mut b = IrBuilder::new();
        let f = b
            .declare(FuncKind::Named, "lookup", Ty::Unit, Ty::i32(), loc)
            .unwrap();
        let mut fb = b.build_fn(f).unwrap();
        let host = fb.input();
        let mut object = || fb.constant(Value::I32(1), loc).unwrap();
        let (merge, view, exit, next, raw) = (object(), object(), object(), object(), object());
        fb.output(raw, None, loc).unwrap();
        fb.finish().unwrap();
        let ir = b.seal(f).unwrap();
        let output = ir.func(f).unwrap().output;
        let site = ir.in_edges(output)[0];
        let mut groups = SecondaryMap::new();
        groups.insert(
            site,
            Epi {
                store_out: false,
                chain: vec![Member {
                    m: site,
                    body: f,
                    out: merge,
                    store: true,
                    srcs: vec![],
                }],
            },
        );
        let mut plan = ResidentPlan::default();
        plan.chain.insert(site, 2);
        plan.carries.push(Carry {
            merge,
            k: 0,
            next,
            views: vec![view],
            exit: Some(exit),
        });
        for o in [merge, view, exit, output] {
            assert_eq!(plan.stream_of(&ir, &groups, o), 2);
        }
        // The next iteration owns a separate slot; host/unknown writers stay legacy.
        for o in [next, host, raw] {
            assert_eq!(plan.stream_of(&ir, &groups, o), 0);
        }
        plan.chain.clear();
        assert_eq!(plan.stream_of(&ir, &groups, view), 0);
    }

    /// Disjoint lives share bytes; overlapping ones (an inclusive end: the
    /// site that reads one object writes the next) never do.
    #[test]
    fn color_shares_bytes_exactly_across_disjoint_lives() {
        let mb = 64 << 20;
        // The e2e chain: a, b, e built first; c = a·b at 3; d = c·e at 4; d[0] at 5.
        let (off, total) = color(&[(0, 3, mb), (1, 3, mb), (2, 4, mb), (3, 4, mb), (4, 5, mb)]);
        assert_eq!(
            off,
            [0, mb, 2 * mb, 3 * mb, 0],
            "d takes a's bytes, not c's or e's"
        );
        assert_eq!(total, 4 * mb);
        // Born where the other dies: both live at that morphism.
        let (off, _) = color(&[(0, 2, 100), (2, 3, 100)]);
        assert_ne!(off[0], off[1]);
        // Strictly after: shared.
        let (off, total) = color(&[(0, 2, 100), (3, 4, 100)]);
        assert_eq!((off[0], off[1], total), (0, 0, ARENA_ALIGN));
    }

    /// Every offset is `cuMemAlloc`'s alignment, and a hole is reused first-fit.
    #[test]
    fn color_aligns_and_fills_holes() {
        // x dies early; y lives long; z (born after x) takes x's hole below y.
        let (off, total) = color(&[(0, 1, 10), (0, 9, 300), (2, 9, 5)]);
        assert!(off.iter().all(|o| o % ARENA_ALIGN == 0), "{off:?}");
        assert_eq!(off, [0, ARENA_ALIGN, 0]);
        assert_eq!(total, 3 * ARENA_ALIGN);
        assert_eq!(color(&[]), (vec![], 0));
    }

    use super::waves_merge;

    #[test]
    fn the_wave_rule_merges_exactly_the_last_waves_that_fit_one() {
        // Two quarter waves share one wave; two full waves never do.
        assert!(waves_merge((64, 256), (64, 256)));
        assert!(!waves_merge((256, 256), (256, 256)));
        // Under one wave each but not together: two waves either way.
        assert!(!waves_merge((128, 164), (128, 164)));
        // Many waves with short tails (the 3090 at 4096, 82 SMs × 2): 40 + 40
        // of 164 — one wave saved of 14.
        assert!(waves_merge((1024, 164), (1024, 164)));
        // ~1.2 waves each (the A100 at 2048): 4 waves back to back, 3 together.
        assert!(waves_merge((256, 216), (256, 216)));
        // A tail of exactly half each fits; one block more does not.
        assert!(waves_merge((50, 100), (50, 100)));
        assert!(!waves_merge((51, 100), (50, 100)));
        // Different geometries weigh each tail by its own wave.
        assert!(waves_merge((128, 256), (32, 64)));
        assert!(!waves_merge((128, 256), (33, 64)));
    }
}
