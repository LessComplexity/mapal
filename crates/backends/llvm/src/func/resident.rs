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

mod arena;
mod cell_of;
mod cell_sites;
mod emit_cell;
mod emit_site;
mod epilogues;
mod loops;
mod plan;
#[cfg(test)]
mod tests;
mod uses;

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
    /// `range-cell`: `next` is written in place over the state (a range
    /// cell), so it shares the carry's buffer.
    pub inplace: bool,
    /// placement `chain`: the range cells' outputs before `next`, each
    /// written in place over the one before (the first over a view): they
    /// share the carry's buffer too (a trip's appends to one cache).
    pub chain: Vec<ObjectId>,
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
    /// caching-memory `retain-same`: the value is the same on every trip
    /// (`unchanged_plan`'s `same`): the first trip uploads it whole, no
    /// later one uploads anything (`start` / `stride` unused, `stride` 0).
    pub same: bool,
}

/// A device site: a tile site (R0), or a row fold over a device output (R0
/// for folds, S54 `rowfold`).
#[derive(Clone, Debug)]
pub(crate) enum DevSite {
    Tile(TileSite),
    RowFold(super::rowfold::RowFold),
}

impl DevSite {
    /// The source slots the kernel loads (a `tile-post` epilogue's reads too).
    fn reads(&self) -> Vec<u32> {
        match self {
            DevSite::Tile(s) => {
                let mut r = vec![s.a.slot, s.b.slot];
                r.extend(s.post.iter().flat_map(|p| p.reads.iter().copied()));
                r
            }
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
    /// `tile-post`: a number capture (a loop's position), by value — the
    /// `j`-th of the group's uniforms ([`Epi::uniforms`]).
    Uniform(u32),
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

/// device-geometry `tile-post`: the site's own map body after its fold,
/// evaluated on the accumulator before any member — the site's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Post {
    pub body: FuncId,
    pub fold: MorphismId,
    /// One per body input slot: `Gmem` for an array the epilogue reads,
    /// `Index` for the element, `None` for an array only the fold reads (a
    /// null pointer: the variant never touches it).
    pub srcs: Vec<Option<EpiSrc>>,
}

/// A device site's epilogue group (R1): its members in chain order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Epi {
    /// `tile-post`: the site's own body after its fold (value 0 is its result).
    pub post: Option<Post>,
    /// A reader outside the group exists: the site's own output is stored.
    pub store_out: bool,
    pub chain: Vec<Member>,
    /// plan-cost `linear` (§4a concat): the other terms of a site table this
    /// site launches (their sites emit nothing; `chain` is the sum, stored),
    /// and each term's partitions. Empty on an epilogue group.
    pub terms: Vec<MorphismId>,
    pub split: u64,
}

impl Epi {
    /// The distinct `Gmem` operands, in first-use order (the kernel's
    /// pointer parameters after `a, b, out`).
    /// `tile-post`: the number captures the epilogue reads, by capture slot,
    /// in order (`EpiSrc::Uniform(j)` is the `j`-th): one 8-byte argument
    /// each, after the stored members' pointers.
    pub(crate) fn uniforms(&self) -> Vec<u32> {
        self.post
            .iter()
            .flat_map(|p| p.srcs.iter().enumerate())
            .filter_map(|(slot, s)| matches!(s, Some(EpiSrc::Uniform(_))).then_some(slot as u32))
            .collect()
    }

    pub(crate) fn caps(&self) -> Vec<ObjectId> {
        let mut out = Vec::new();
        let post = self.post.iter().flat_map(|p| p.srcs.iter().flatten());
        for s in post.chain(self.chain.iter().flat_map(|mb| &mb.srcs)) {
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
pub(super) fn grouped_writer(
    groups: &SecondaryMap<MorphismId, Epi>,
    o: ObjectId,
) -> Option<MorphismId> {
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
    /// `range-cell`: the interval a piecewise map writes in place (its
    /// state's carry buffer); one thread a written cell.
    pub range: Option<mapal_ir::Outside>,
    /// `scalar-fold`: the uniform slots a device whole fold produces, loaded
    /// from its one-element buffer.
    pub dev: Vec<u32>,
    /// `cell-lanes`: threads a cell.
    pub lanes: u64,
    /// caching-memory `retain-cell`: the output differs from its previous
    /// trip's only in a row delta (`unchanged_plan`).
    pub retain: Option<RetainCell>,
    /// bf16-class `device-trap`: the body checks an unproven index
    /// (`CellSite::traps`): its device variant records the trap, and the
    /// launch arms the next checkpoint's read of it.
    pub trap: bool,
}

/// caching-memory `retain-cell`: a loop's cell whose output differs from its
/// previous trip's value only in `delta` (`mapal_ir::UnchangedPlan::trip`):
/// the first trip after the loop's entry launches the whole map, every
/// later one only the delta's cells (kernel `id`: the range geometry of the
/// delta, cells past the extent clipped) into the same persisting buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RetainCell {
    pub merge: ObjectId,
    pub delta: mapal_ir::Delta,
    pub id: u64,
    /// `cell-lanes` for the delta's cells (its own extent: a few rows can
    /// take a warp each where the whole map's cannot).
    pub lanes: u64,
    /// caching-memory `retain-same`: the output is the same on every trip
    /// (`unchanged_plan`'s `same`) — the first trip launches the whole map,
    /// no later one launches anything (`delta` unused, no delta kernel).
    pub same: bool,
}

impl RetainCell {
    /// The delta kernel's geometry: `(step, piece width)`.
    pub(crate) fn geo(&self) -> (u64, u64) {
        match self.delta {
            mapal_ir::Delta::Pieces { w, step, .. } => (step, w),
            _ => (1, 0),
        }
    }
}

impl Cell {
    /// The arrays the kernel reads, with the product slot each sits in
    /// (`None`: the bare mapped source of a capture-free map).
    fn operands(&self) -> Vec<(ObjectId, Option<u32>)> {
        self.args()
            .filter(|a| !a.2)
            .map(|(o, s, _)| (o, s))
            .collect()
    }

    /// The kernel's arguments after its output, in [`gpu::cell_kernel_text`]'s
    /// parameter order: the mapped array, then each capture slot not
    /// evaluated in the cell — `(object, slot, uniform)`, a uniform passed by
    /// value, the rest by reference.
    /// The kernel's arguments after its output ([`Self::args`]).
    pub(crate) fn arg_count(&self) -> usize {
        self.args().count()
    }

    fn args(&self) -> impl Iterator<Item = (ObjectId, Option<u32>, bool)> + '_ {
        let k = self.caps.len() as u32;
        let mapped = self.mapped.map(|o| (o, (k > 0).then_some(k)));
        let caps = (0..k)
            .filter(|&s| !self.law(s))
            .map(|s| (self.caps[s as usize], Some(s)));
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
    /// full-model `once-range`: a straight-line range cell's output → the
    /// array it writes in place over (its source, dead after it): one
    /// device buffer, the chain's first ([`Self::alias`]).
    pub inplace: SecondaryMap<ObjectId, ObjectId>,
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
    /// kernel-merging `zero-fold`: a kernel (a cell or a unit launch) that
    /// zeroes the outputs of later atomic sites, each `(site, bytes)`.
    pub zero_by: SecondaryMap<MorphismId, Vec<(MorphismId, u64)>>,
    /// The sites whose zero pass a `zero_by` kernel runs — or (gaps-disasm
    /// C1 `zero-hoist`) a zero op issued after the `zero_after` step.
    pub zeroed: SecondaryMap<MorphismId, ()>,
    /// gaps-disasm C1 `zero-hoist`: a step after which the zero ops of later
    /// atomic sites no kernel took are issued, each `(site, bytes)` — the
    /// earliest point the walk allows, off the site's own host path.
    pub zero_after: SecondaryMap<MorphismId, Vec<(MorphismId, u64)>>,
    /// Each hoisted site's slot `j`: the global `@<fn>.zh<j>` holding its
    /// candidate table (null without one), so the hoisted op zeroes only
    /// where the site's pick adds into the output (`mapal_gpu_zero_for`).
    pub zero_hoist_ix: SecondaryMap<MorphismId, usize>,
    /// The host intent line (`; mapal.zero.host …`): the atomic sites, the
    /// folded ones, and why each other kept its zero op.
    pub zero_line: String,
    /// `trip-batch`: the loops entered at the last trip of a skippable first
    /// interval, each batched write made first by one cell launch.
    pub batch: Vec<Batch>,
    /// kernel-merging `horizontal`: the merged launches, and each member's
    /// `(group, index)`.
    pub hz: Vec<super::horizontal::HzGroup>,
    pub hz_of: SecondaryMap<MorphismId, (usize, usize)>,
    /// The host intent line (`; mapal.hz.host …`).
    pub hz_line: String,
    /// kernel-merging `calls`: the host intent line (`; mapal.calls.host …`).
    pub calls_line: String,
}

/// recurrence-precision `trip-batch`: a loop whose first trip interval
/// [`mapal_ir::TripPhase::skip`]s. At entry its batched range cells run once
/// over every cell the interval's trips but the last write (`[start, start +
/// count)`, cell `c` from trip `c ÷ d − b`), and the counter starts at
/// `enter` — the interval's last trip, which runs as an ordinary trip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Batch {
    pub merge: ObjectId,
    pub counter: u32,
    pub enter: i64,
    /// `(range cell, kernel id, start, count, counter slot, d, b)` per batched write.
    pub writes: Vec<(MorphismId, u64, u64, u64, u32, u64, i64)>,
}

/// Every arena offset is a multiple of this: `cuMemAlloc`'s own alignment
/// guarantee (CUDA C Programming Guide: an address returned by the driver's
/// allocators is aligned to at least 256 bytes), so an arena member is
/// addressed exactly as its own allocation would be.
pub(crate) const ARENA_ALIGN: u64 = 256;

/// `mapal_gpu_site` flags bit 3 (tune-context, `mapal_choice.h`
/// `MAPAL_SITE_COLD`): an operand of the site lives in DRAM in the program.
pub(crate) const SITE_COLD: u32 = 8;

/// One function's device arena (S54 R6), a pure function of the plan.
#[derive(Clone, Debug, Default)]
pub(crate) struct Arena {
    /// The allocation, a multiple of [`ARENA_ALIGN`]; 0 = no member.
    pub bytes: u64,
    /// Each member's byte offset: the objects freed at a straight-line death.
    pub offset: SecondaryMap<ObjectId, u64>,
    /// Device-only arrays: no host slot (no `%Frame` field, no paging).
    pub host_free: SecondaryMap<ObjectId, ()>,
    /// caching-memory `cone-span`: each member with its `(birth, death,
    /// offset, aligned bytes)`, which the entry arena's later buffers avoid.
    pub spans: Vec<(ObjectId, u32, u32, u64, u64)>,
}

/// First-fit interval coloring: `live[i] = (birth, death, bytes)` occupies
/// `[birth, death]` (inclusive: a site reading one object while writing
/// another holds both); in birth order each takes the lowest
/// [`ARENA_ALIGN`]-aligned offset no time-overlapping object holds. Returns
/// the offsets and the arena size.
pub(crate) fn color(live: &[(u32, u32, u64)]) -> (Vec<u64>, u64) {
    color_by(live, false)
}

/// [`color`], and with `by_size` (caching-memory `arena-fit`) the objects
/// placed largest first (then in birth order): a large late object no
/// longer lands above the small ones that filled the low offsets before it
/// (first-fit by decreasing size, the interval-packing order that keeps
/// the peak near the live bytes' maximum).
pub(crate) fn color_by(live: &[(u32, u32, u64)], by_size: bool) -> (Vec<u64>, u64) {
    let size = |i: usize| live[i].2.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
    let mut order: Vec<usize> = (0..live.len()).collect();
    if by_size {
        order.sort_by_key(|&i| (std::cmp::Reverse(size(i)), live[i].0, i));
    } else {
        order.sort_by_key(|&i| (live[i].0, i));
    }
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

/// caching-memory `cone-span`: offsets for buffers live from `from` to the
/// function's exit (`bufs[i] = (from, bytes)`), placed in birth order at the
/// lowest [`ARENA_ALIGN`]-aligned offset that no member live at or after
/// `from` (`spans`: `(birth, death, offset, aligned bytes)`, fixed) and no
/// buffer placed before holds. Returns the absolute offsets and the arena's
/// size (at least `line`, the members' own).
pub(crate) fn cone_fit(
    spans: &[(u32, u32, u64, u64)],
    line: u64,
    bufs: &[(u32, u64)],
) -> (Vec<u64>, u64) {
    let size = |b: u64| b.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
    let mut order: Vec<usize> = (0..bufs.len()).collect();
    order.sort_by_key(|&i| (bufs[i].0, i));
    let mut off = vec![0; bufs.len()];
    let mut placed: Vec<(u64, u64)> = Vec::new();
    let mut total = line;
    for i in order {
        let (from, bytes) = bufs[i];
        let mut busy: Vec<(u64, u64)> = spans
            .iter()
            .filter(|&&(_, d, _, _)| d >= from)
            .map(|&(_, _, o, b)| (o, o + b))
            .chain(placed.iter().copied())
            .collect();
        busy.sort_unstable();
        let mut at = 0;
        for (s, e) in busy {
            if at + size(bytes) <= s {
                break;
            }
            at = at.max(e);
        }
        off[i] = at;
        placed.push((at, at + size(bytes)));
        total = total.max(at + size(bytes));
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
        let state = self
            .carries
            .iter()
            .find(|c| c.views.contains(&o) || c.exit == Some(o));
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
    /// the component's slot (its key), and so does a next state written in
    /// place (`range-cell`), and a straight-line range cell's output written
    /// over its dead source (`once-range`).
    pub(crate) fn alias(&self, o: ObjectId) -> ObjectId {
        // full-model `once-range`: a straight-line in-place chain's buffer
        // is its first array's.
        let mut o = o;
        for _ in 0..4096 {
            match self.inplace.get(o) {
                Some(&w) => o = w,
                None => break,
            }
        }
        self.carries
            .iter()
            .find(|c| {
                c.views.contains(&o)
                    || c.exit == Some(o)
                    || (c.inplace && (c.next == o || c.chain.contains(&o)))
            })
            .map_or(o, Carry::key)
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

/// A cell's or a generated producer's element: a float, or (ADR-0043) a
/// narrow float the body rounds into — stored by the kernel's own `store`.
fn float_cell(ty: &Ty) -> bool {
    is_float(ty) || matches!(ty, Ty::Narrow(_))
}
