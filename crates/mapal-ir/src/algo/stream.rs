//! **`stream_plan` (flash-attention F4): one pass over windows of an axis.**
//!
//! A row-block group ([`CategoryIr::unit_plan`]) whose tile site folds over
//! the cells of an in-group row (`k` = the row's width `w`, its row operand
//! `a` that row) is a chain *tile → cells and row folds over its lanes →
//! tile folding over the same axis*: attention's `QKᵀ → row max → exp → row
//! sum → P·V`, softmax-weighted pooling, a logsumexp's row. Every member of
//! the axis evaluates **window by window** — a window being any contiguous
//! run of the axis — when each fold over the axis combines its state across
//! windows by its own `⊕` and each per-row value its terms read is taken by
//! `fold_combine`'s law ([`DepLaw`]): a factor or shift applied once after
//! the last window, a rescale whose shift is the running extreme of the
//! same `x` (ADR-0042 D7: `exp(a − m₁) = exp(a − m) · exp(m − m₁)`). Then
//! the axis-wide values (the scores, the probabilities) are never stored and
//! each operand window is read once a row block.
//!
//! Roles ([`StreamRole`]): `Produce` (a tile whose lanes are the axis),
//! `Cell` (a per-cell map over the axis reading axis-wide members at its own
//! cell; each read of an axis fold's value is substituted by its law's
//! stand-in), `Fold` (a row fold over the axis), `Consume` (a tile folding
//! over the axis), `Row` (per-row work outside the window loop). A value a
//! term needs in its final form (a [`DepLaw::Phase`]) or an axis-wide value
//! read outside the group refuses the group (`passes > 1`: the producer would
//! be recomputed). Geometry and algebra only: whether the pass pays, its
//! window, rows and split are the backend's (`TargetProfile`).

use crate::algo::{
    AxisDigit, DepLaw, Digit, FoldCombinePlan, FootprintPlan, RadixForm, Reach, RescaleArg,
    RowFoldOp,
};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, Value};
use crate::unit::{UnitGroup, UnitKind};

/// A value constant along the axis that a fold's or consumer's term reads,
/// and how ([`crate::RowDep`] restricted to the group's members).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamLaw {
    /// The member whose per-row value it is (an axis fold of the group).
    pub value: usize,
    pub law: DepLaw,
    /// [`DepLaw::Rescale`]'s exponent: its shift is `value`'s running state.
    pub rescale: Option<RescaleArg>,
}

/// What one member of a [`StreamGroup`] is to the pass.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamRole {
    /// A tile site whose lanes are the axis: each window's cells, read from
    /// memory operands only.
    Produce,
    /// A per-cell map over the axis: `(slot, value member, law)` — each read
    /// of an axis fold's per-row value, taken as the law's stand-in (a
    /// factor's `1`, a shift's `∓0`, a rescale's running shift).
    Cell { subst: Vec<(u32, usize, DepLaw)> },
    /// A row fold over the axis: its state carried window to window.
    Fold { laws: Vec<StreamLaw> },
    /// A tile site folding over the axis (`k` = the axis, `a` an axis-wide
    /// member): `c` accumulators a row carried window to window.
    Consume { laws: Vec<StreamLaw> },
    /// Per-row work outside the window loop: before it (reading memory or
    /// earlier rows) or `after` it (reading a fold's or consumer's final
    /// value).
    Row { after: bool },
}

/// One group the pass realizes ([`CategoryIr::stream_plan`]).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamGroup {
    /// Index into [`crate::UnitPlan::groups`].
    pub group: usize,
    /// The axis: cells per row of the axis-wide members, `k` of the folds
    /// and consumers.
    pub axis: u64,
    /// Per member of the group, in its order — the first `roles.len()`
    /// members (sweep-attention: the members after the last consumer that
    /// read only its value, a carried sum of a head loop's outputs, run after
    /// the stream as their own launches, reading its stored rows).
    pub roles: Vec<StreamRole>,
    /// Axis-wide members never stored, and their bytes over every row.
    pub dissolved: Vec<usize>,
    pub dissolved_bytes: u64,
    /// The state a row carries across windows: each fold's element, each
    /// consumer's `c` elements.
    pub state_bytes: u64,
    /// `1`: one pass ([`StreamGroup`] exists only then).
    pub passes: u32,
    /// flash-attention F5/F6: the producer's mask over a block of its rows
    /// and a window of its lanes, and whether a wholly masked window may be
    /// skipped (`None`: no guarded producer).
    pub guard: Option<StreamGuard>,
}

/// One side of a guard's comparison over a producer cell `e = row·w +
/// lane`: `base + Σ row digits + Σ lane digits`, each `coef·((x ÷ stride) %
/// radix)` (`radix == 0`: the quotient itself) — plus, mask-frontend, the
/// terms a compile-time box cannot see: digits of numbers the producer map
/// captures (`uniform`: one value a launch, a decode step's position) and
/// one per-row value read from memory (`load`: a runtime position vector's
/// limit). A side with either is `open`: a box over it is mixed until the
/// kernel, holding the values, classifies it.
#[derive(Clone, PartialEq, Eq)]
pub struct GuardSide {
    pub base: u64,
    pub row: Vec<AxisDigit>,
    pub lane: Vec<AxisDigit>,
    pub uniform: Vec<Digit>,
    pub load: Option<RowLoad>,
}

/// The derived `Debug`'s text, `uniform` / `load` printed only where a term
/// is (the `RadixTile` precedent: text keyed on it stays as it was).
impl std::fmt::Debug for GuardSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("GuardSide");
        d.field("base", &self.base)
            .field("row", &self.row)
            .field("lane", &self.lane);
        if !self.uniform.is_empty() {
            d.field("uniform", &self.uniform);
        }
        if let Some(l) = &self.load {
            d.field("load", l);
        }
        d.finish()
    }
}

/// mask-frontend: **a guard side's per-row value from memory** — capture
/// `slot` of the producer map (an integer array the epilogue reads) at
/// `base + Σ index digits of the row`, times `coef`; the read proven in
/// bounds (`bounds_proof`). A block's range of it is its rows' least and
/// greatest value, read once a row block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowLoad {
    pub slot: u32,
    pub base: u64,
    pub index: Vec<AxisDigit>,
    pub coef: u64,
}

/// The values an open [`GuardSide`] reads: a captured number's (`uniform`)
/// and a row load's array element (`load`), `None` where unknown.
pub trait GuardEnv {
    fn number(&self, slot: u32) -> Option<u64>;
    fn element(&self, slot: u32, index: u64) -> Option<i64>;
}

/// No values: every open side is unknown (a compile-time box).
pub struct NoValues;

impl GuardEnv for NoValues {
    fn number(&self, _: u32) -> Option<u64> {
        None
    }
    fn element(&self, _: u32, _: u64) -> Option<i64> {
        None
    }
}

/// mask-frontend: one side of a guard before it is split into row and lane
/// digits — a digit form of the body's inputs and at most one per-row load
/// `(slot, its index's form, coef)`.
struct SideTerm {
    form: RadixForm,
    load: Option<(u32, RadixForm, u64)>,
}

/// What a guard holds over a box of producer cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxClass {
    /// Every cell computes the fold.
    Visible,
    /// No cell does: each is the masked constant.
    Masked,
    Mixed,
}

/// flash-attention F5: **a guarded producer's mask** — the tile site's
/// epilogue is `(lhs cmp rhs) ? body(fold) : masked` with both sides digit
/// sums of the cell index (`TilePost::guarded`): over a box of rows and
/// lanes each side's range bounds the comparison, so a box is wholly
/// visible, wholly masked, or mixed — exact for every input (a box it
/// cannot decide is mixed, and a mixed box computes every cell).
#[derive(Clone, Debug, PartialEq)]
pub struct GuardBox {
    pub cmp: Operation,
    pub lhs: GuardSide,
    pub rhs: GuardSide,
    /// The fold is computed where the comparison is this.
    pub fold_when: bool,
    /// A masked cell's value.
    pub masked: f64,
}

/// flash-attention F5/F6 of one stream group.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamGuard {
    /// The guarded producer (the group's only one).
    pub producer: usize,
    pub form: GuardBox,
    /// F6: a wholly masked window may be skipped — no axis-wide value is
    /// stored, every axis cell reads its index only to address the members
    /// it reads (its value at a masked cell is one function of the masked
    /// constant and its row's values), every fold over the axis is a `+` or
    /// a running extreme, and every consumer's column operand is proven
    /// finite (`finite_plan`: `0·y = 0`). The kernel still checks, after the
    /// visible windows, that each fold's term at the masked constant is its
    /// identity at the final values; where it is not the skipped windows run.
    pub skip: bool,
    /// Why not, when `skip` is false.
    pub why: &'static str,
}

fn digit_range(d: &AxisDigit, x0: u64, x1: u64) -> (u64, u64) {
    let (q0, q1) = (x0 / d.stride, x1 / d.stride);
    match d.radix {
        0 => (d.coef * q0, d.coef * q1),
        r if q0 / r == q1 / r => (d.coef * (q0 % r), d.coef * (q1 % r)),
        r => (0, d.coef * (r - 1)),
    }
}

impl GuardSide {
    /// A side of the row and lane alone: no per-launch number, no load.
    pub fn closed(&self) -> bool {
        self.uniform.is_empty() && self.load.is_none()
    }

    /// The side's range over rows `[r0, r1]` and lanes `[l0, l1]`, the open
    /// terms' values from `env` (`None`: one unknown; a load over at most
    /// 2¹⁶ rows).
    pub fn range_with(
        &self,
        (r0, r1): (u64, u64),
        (l0, l1): (u64, u64),
        env: &dyn GuardEnv,
    ) -> Option<(i128, i128)> {
        let mut lo = i128::from(self.base);
        let mut hi = lo;
        for (ds, a, b) in [(&self.row, r0, r1), (&self.lane, l0, l1)] {
            for d in ds {
                let (x, y) = digit_range(d, a, b);
                lo += i128::from(x);
                hi += i128::from(y);
            }
        }
        for d in &self.uniform {
            let v = env.number(d.slot)?;
            let q = v / d.stride;
            let x = i128::from(d.coef) * i128::from(if d.radix == 0 { q } else { q % d.radix });
            lo += x;
            hi += x;
        }
        if let Some(l) = &self.load {
            if r1 < r0 || r1 - r0 >= 1 << 16 {
                return None;
            }
            let (mut a, mut b) = (i128::MAX, i128::MIN);
            for r in r0..=r1 {
                let at = l.base + l.index.iter().map(|d| digit_range(d, r, r).0).sum::<u64>();
                let v = i128::from(env.element(l.slot, at)?) * i128::from(l.coef);
                a = a.min(v);
                b = b.max(v);
            }
            lo += a;
            hi += b;
        }
        Some((lo, hi))
    }

    /// The side's row part — its base, row digits and open terms — over the
    /// rows `rows` (any set: a block walked in [`crate::UnitGroup::order`]),
    /// `None` with an unknown value or no row.
    pub fn rows_range(&self, rows: &[u64], env: &dyn GuardEnv) -> Option<(i128, i128)> {
        let (mut lo, mut hi) = (i128::MAX, i128::MIN);
        for &r in rows {
            let (a, b) = self.range_with((r, r), (0, 0), env)?;
            let lane = self.lanes_range((0, 0));
            lo = lo.min(a - lane.0);
            hi = hi.max(b - lane.1);
        }
        (lo <= hi).then_some((lo, hi))
    }

    /// The side's lane digits' range over lanes `[l0, l1]`.
    pub fn lanes_range(&self, (l0, l1): (u64, u64)) -> (i128, i128) {
        let mut lo = 0i128;
        let mut hi = 0i128;
        for d in &self.lane {
            let (x, y) = digit_range(d, l0, l1);
            lo += i128::from(x);
            hi += i128::from(y);
        }
        (lo, hi)
    }

    /// The closed side's range over rows `[r0, r1]` and lanes `[l0, l1]`.
    pub fn range(&self, rows: (u64, u64), lanes: (u64, u64)) -> (u64, u64) {
        let (lo, hi) = self
            .range_with(rows, lanes, &NoValues)
            .expect("a closed side");
        (lo as u64, hi as u64)
    }

    /// The closed side at one cell.
    pub fn at(&self, row: u64, lane: u64) -> u64 {
        self.range((row, row), (lane, lane)).0
    }
}

impl GuardBox {
    /// Both sides closed: every box classifies at compile time.
    pub fn closed(&self) -> bool {
        self.lhs.closed() && self.rhs.closed()
    }

    /// The comparison over rows `[r0, r1]` × lanes `[l0, l1]`: always true
    /// (`Some(true)`), always false, or either (an unknown value: either).
    fn holds(&self, rows: (u64, u64), lanes: (u64, u64), env: &dyn GuardEnv) -> Option<bool> {
        let a = self.lhs.range_with(rows, lanes, env)?;
        let b = self.rhs.range_with(rows, lanes, env)?;
        self.decide(a, b)
    }

    /// The class of a box whose sides range over `a` and `b`.
    pub fn class_of(&self, a: (i128, i128), b: (i128, i128)) -> BoxClass {
        match self.decide(a, b) {
            Some(c) if c == self.fold_when => BoxClass::Visible,
            Some(_) => BoxClass::Masked,
            None => BoxClass::Mixed,
        }
    }

    fn decide(&self, a: (i128, i128), b: (i128, i128)) -> Option<bool> {
        let (t, f) = match self.cmp {
            Operation::Le => (a.1 <= b.0, a.0 > b.1),
            Operation::Lt => (a.1 < b.0, a.0 >= b.1),
            Operation::Ge => (a.0 >= b.1, a.1 < b.0),
            Operation::Gt => (a.0 > b.1, a.1 <= b.0),
            Operation::Eq => (
                a.0 == a.1 && b.0 == b.1 && a.0 == b.0,
                a.1 < b.0 || b.1 < a.0,
            ),
            Operation::Neq => (
                a.1 < b.0 || b.1 < a.0,
                a.0 == a.1 && b.0 == b.1 && a.0 == b.0,
            ),
            _ => (false, false),
        };
        match (t, f) {
            (true, _) => Some(true),
            (_, true) => Some(false),
            _ => None,
        }
    }

    /// The box's class (module doc of [`GuardBox`]); an open side's box is
    /// mixed.
    pub fn classify(&self, rows: (u64, u64), lanes: (u64, u64)) -> BoxClass {
        self.classify_with(rows, lanes, &NoValues)
    }

    /// The box's class with the open terms' values.
    pub fn classify_with(
        &self,
        rows: (u64, u64),
        lanes: (u64, u64),
        env: &dyn GuardEnv,
    ) -> BoxClass {
        match self.holds(rows, lanes, env) {
            Some(c) if c == self.fold_when => BoxClass::Visible,
            Some(_) => BoxClass::Masked,
            None => BoxClass::Mixed,
        }
    }

    /// One cell of a closed guard: is its fold computed?
    pub fn visible(&self, row: u64, lane: u64) -> bool {
        self.visible_with(row, lane, &NoValues)
            .expect("a closed guard")
    }

    /// One cell with the open terms' values (`None`: one unknown).
    pub fn visible_with(&self, row: u64, lane: u64, env: &dyn GuardEnv) -> Option<bool> {
        let a = self.lhs.range_with((row, row), (lane, lane), env)?.0;
        let b = self.rhs.range_with((row, row), (lane, lane), env)?.0;
        let c = match self.cmp {
            Operation::Le => a <= b,
            Operation::Lt => a < b,
            Operation::Ge => a >= b,
            Operation::Gt => a > b,
            Operation::Eq => a == b,
            _ => a != b,
        };
        Some(c == self.fold_when)
    }
}

/// Every stream group of one function, and the groups refused with why.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamPlan {
    pub groups: Vec<StreamGroup>,
    /// `(unit group, reason)`: `no-axis`, `producer`, `read`, `escape`,
    /// `phase`, `shift`, `member`.
    pub refused: Vec<(usize, &'static str)>,
}

impl StreamPlan {
    /// The stream group of unit group `g`.
    pub fn of(&self, g: usize) -> Option<&StreamGroup> {
        self.groups.iter().find(|s| s.group == g)
    }
}

impl CategoryIr {
    /// **The stream groups of `f`** (module doc): built from `unit_plan`,
    /// `fold_combine` and `footprint_plan`. Stored on the graph; empty under
    /// `FactArms::stream` off.
    pub fn stream_plan(&self, f: FuncId) -> StreamPlan {
        self.fact(f, |s| &s.stream_plan, || self.stream_plan_compute(f))
    }

    fn stream_plan_compute(&self, f: FuncId) -> StreamPlan {
        let mut plan = StreamPlan::default();
        if !self.fact_arms().stream || self.func(f).is_none() {
            return plan;
        }
        let units = self.unit_plan(f);
        let fc = self.fold_combine(f);
        let fp = self.footprint_plan(f);
        for (gi, g) in units.groups.iter().enumerate() {
            match self.stream_group(gi, g, &fc, &fp) {
                Ok(s) => plan.groups.push(s),
                Err(why) => plan.refused.push((gi, why)),
            }
        }
        plan
    }

    /// **The rescale's shift** (ADR-0042 D7): may the per-row value made by
    /// row fold `value` be the shift of a rescale law `arg` that fold `fold`
    /// takes? Its step keeps the extreme bounding every term by `1` (a
    /// running max where the exponent rises with `x`, a min where it falls),
    /// it folds the exponent's own `x` (the array whose same-cell read `x`
    /// is), and both walk the same cells of a row (one `k`, one read). The
    /// one test every consumer of the law reads (`row-slices`, the stream).
    pub fn rescale_shift_ok(
        &self,
        f: FuncId,
        fold: MorphismId,
        value: MorphismId,
        arg: &RescaleArg,
    ) -> bool {
        let folds = self.row_fold(f);
        let (Some(sk), Some(sv)) = (folds.get(fold), folds.get(value)) else {
            return false;
        };
        let fc = self.fold_combine(f);
        extreme_up(sv.op) == Some(arg.up)
            && arg.against.is_some()
            && fc.sites.get(value).map(|c| c.term) == arg.against
            && sv.k == sk.k
            && (sv.a.ci, sv.a.ck, sv.a.base) == (sk.a.ci, sk.a.ck, sk.a.base)
    }

    fn stream_group(
        &self,
        gi: usize,
        g: &UnitGroup,
        fc: &FoldCombinePlan,
        fp: &FootprintPlan,
    ) -> Result<StreamGroup, &'static str> {
        let ms = &g.members;
        // The axis: a tile folding over an in-group row of its own `k`.
        let axis = ms
            .iter()
            .find_map(|mb| match &mb.kind {
                UnitKind::Tile(t) if t.k > 1 => mb
                    .reads
                    .iter()
                    .find(|&&(s, p)| s == t.a.slot && ms[p].w == t.k)
                    .map(|_| t.k),
                _ => None,
            })
            .ok_or("no-axis")?;
        // sweep-attention: the members after the last consumer (a tile
        // folding the axis over an in-group axis-wide row) that read only it
        // or each other are no part of the pass: they follow it, reading its
        // stored rows (a head loop's carried sum `y + attg`, its width the
        // axis's where `d = T`).
        let consumer = ms.iter().rposition(|mb| match &mb.kind {
            UnitKind::Tile(t) => {
                t.k == axis
                    && mb
                        .reads
                        .iter()
                        .any(|&(s, p)| s == t.a.slot && ms[p].w == axis)
            }
            _ => false,
        });
        let ms = match consumer {
            Some(c) if (c + 1..ms.len()).all(|k| ms[k].reads.iter().all(|&(_, p)| p >= c)) => {
                &ms[..=c]
            }
            _ => &ms[..],
        };
        let n = ms.len();
        let wide = |k: usize| ms[k].w == axis;
        // Does `mb` read its source `p`'s row `i` at cell `j` for step `j`?
        let along = |ci: u64, ck: u64, base: u64, clane: u64| {
            ck == 1 && base == 0 && ci == axis && clane == 0
        };
        let mut roles: Vec<StreamRole> = Vec::with_capacity(n);
        for (k, mb) in ms.iter().enumerate() {
            let reads_final = mb.reads.iter().any(|&(_, p)| {
                matches!(
                    roles[p],
                    StreamRole::Fold { .. }
                        | StreamRole::Consume { .. }
                        | StreamRole::Row { after: true }
                )
            });
            // A tile folding the axis over an in-group row is its consumer
            // even where its own width is the axis's too (`d = T`).
            let consumes = |t: &crate::TileSite| {
                t.k == axis && mb.reads.iter().any(|&(s, p)| s == t.a.slot && wide(p))
            };
            let role = match &mb.kind {
                UnitKind::Tile(t) if mb.w == axis && t.c == axis && !consumes(t) => {
                    // Its operands are memory (complete before the launch).
                    if !mb.reads.is_empty() {
                        return Err("producer");
                    }
                    StreamRole::Produce
                }
                UnitKind::Tile(t) if t.k == axis => {
                    // Its row operand an axis-wide row: its own row (the plain
                    // form) or one under a digit permutation of the rows
                    // (`unit_plan` checked it reads the row whole).
                    let a = mb.reads.iter().find(|&&(s, _)| s == t.a.slot);
                    let whole = mb.row_of.is_some() || along(t.a.ci, t.a.ck, t.a.base, t.a.clane);
                    match a {
                        Some(&(_, p)) if wide(p) && whole => {}
                        _ => return Err("read"),
                    }
                    // Any other read is its epilogue's, of a per-row value
                    // complete after the last window.
                    let post = t.post.as_ref();
                    for &(s, p) in &mb.reads {
                        if s == t.a.slot {
                            continue;
                        }
                        let ok = post.is_some_and(|x| x.reads.contains(&s))
                            && ms[p].w == 1
                            && matches!(
                                roles[p],
                                StreamRole::Fold { .. } | StreamRole::Row { after: false }
                            );
                        if !ok {
                            return Err("read");
                        }
                    }
                    StreamRole::Consume { laws: Vec::new() }
                }
                UnitKind::Tile(_) => return Err("member"),
                UnitKind::Fold(s) if s.k == axis && mb.reads.iter().any(|&(_, p)| wide(p)) => {
                    if s.w != 1
                        || !along(s.a.ci, s.a.ck, s.a.base, s.a.clane)
                        || s.a.ksplit.is_some()
                    {
                        return Err("read");
                    }
                    StreamRole::Fold { laws: Vec::new() }
                }
                UnitKind::Fold(_) => StreamRole::Row { after: reads_final },
                UnitKind::Cell if wide(k) => {
                    for &(slot, p) in &mb.reads {
                        let same = fp
                            .of(mb.m)
                            .iter()
                            .filter(|e| e.slot == slot && e.producer == ms[p].out)
                            .all(|e| e.reach.is_same());
                        let ok = match &roles[p] {
                            StreamRole::Produce | StreamRole::Cell { .. } => same,
                            StreamRole::Fold { .. } | StreamRole::Row { after: false } => {
                                ms[p].w == 1
                            }
                            _ => false,
                        };
                        if !ok {
                            return Err("read");
                        }
                    }
                    if mb.stored {
                        return Err("escape");
                    }
                    StreamRole::Cell { subst: Vec::new() }
                }
                UnitKind::Cell => {
                    if mb.reads.iter().any(|&(_, p)| wide(p)) {
                        return Err("read");
                    }
                    StreamRole::Row { after: reads_final }
                }
            };
            roles.push(role);
        }
        if !roles.iter().any(|r| matches!(r, StreamRole::Produce)) {
            return Err("producer");
        }
        // A stored axis-wide value other than a producer's needs the final
        // per-row values: a second pass.
        for (k, mb) in ms.iter().enumerate() {
            if matches!(roles[k], StreamRole::Cell { .. }) && mb.stored {
                return Err("escape");
            }
        }
        // The axis folds and consumers each axis-wide member feeds at its own
        // cell (through cells of the axis).
        let mut feeds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for k in (0..n).rev() {
            let combines = matches!(
                roles[k],
                StreamRole::Fold { .. } | StreamRole::Consume { .. }
            );
            for &(_, p) in &ms[k].reads {
                if !wide(p) {
                    continue;
                }
                let add: Vec<usize> = match combines {
                    true => vec![k],
                    false => feeds[k].clone(),
                };
                for x in add {
                    if !feeds[p].contains(&x) {
                        feeds[p].push(x);
                    }
                }
            }
        }
        let folds: Vec<bool> = roles
            .iter()
            .map(|r| matches!(r, StreamRole::Fold { .. }))
            .collect();
        // Each combining member's laws over the group's axis folds.
        let mut laws: Vec<Vec<StreamLaw>> = vec![Vec::new(); n];
        for k in 0..n {
            if !matches!(
                roles[k],
                StreamRole::Fold { .. } | StreamRole::Consume { .. }
            ) {
                continue;
            }
            let Some(site) = fc.sites.get(ms[k].m) else {
                return Err("phase");
            };
            for d in &site.deps {
                let Some(v) = ms.iter().position(|x| x.out == d.value) else {
                    continue;
                };
                match &roles[v] {
                    StreamRole::Fold { .. } => {}
                    // Complete before the loop: read as it is.
                    StreamRole::Row { after: false } => continue,
                    _ => return Err("phase"),
                }
                if d.law == DepLaw::Phase {
                    return Err("phase");
                }
                if d.law == DepLaw::Rescale {
                    let ok = d.rescale.as_ref().is_some_and(|arg| {
                        let extreme = match &ms[v].kind {
                            UnitKind::Fold(s) => extreme_up(s.op),
                            _ => None,
                        };
                        extreme == Some(arg.up)
                            && arg.against.is_some()
                            && fc.sites.get(ms[v].m).map(|c| c.term) == arg.against
                            && v < k
                    });
                    if !ok {
                        return Err("shift");
                    }
                }
                if !laws[k].iter().any(|l| l.value == v && l.law == d.law) {
                    laws[k].push(StreamLaw {
                        value: v,
                        law: d.law,
                        rescale: d.rescale.clone(),
                    });
                }
            }
        }
        // Each cell's reads of an axis fold: the law every combining member
        // it feeds takes them by (one law a read).
        for c in 0..n {
            if !matches!(roles[c], StreamRole::Cell { .. }) {
                continue;
            }
            let mut subst = Vec::new();
            for &(slot, v) in &ms[c].reads {
                if !folds[v] {
                    continue;
                }
                if feeds[c].is_empty() {
                    return Err("phase");
                }
                let mut law: Option<DepLaw> = None;
                for &x in &feeds[c] {
                    let Some(site) = fc.sites.get(ms[x].m) else {
                        return Err("phase");
                    };
                    let found = site
                        .deps
                        .iter()
                        .find(|d| d.via == ms[c].m && d.slot == slot && d.value == ms[v].out)
                        .map(|d| d.law);
                    match (found, law) {
                        (None | Some(DepLaw::Phase), _) => return Err("phase"),
                        (Some(l), None) => law = Some(l),
                        (Some(l), Some(m)) if l == m => {}
                        _ => return Err("phase"),
                    }
                }
                let law = law.expect("a fed cell has a law");
                // A rescale's shift is `v`'s running state: made before.
                if law == DepLaw::Rescale && v > c {
                    return Err("shift");
                }
                subst.push((slot, v, law));
            }
            roles[c] = StreamRole::Cell { subst };
        }
        for k in 0..n {
            match &mut roles[k] {
                StreamRole::Fold { laws: l } | StreamRole::Consume { laws: l } => {
                    *l = std::mem::take(&mut laws[k]);
                }
                _ => {}
            }
        }
        let dissolved: Vec<usize> = (0..n)
            .filter(|&k| wide(k) && !ms[k].stored)
            .filter(|&k| matches!(roles[k], StreamRole::Produce | StreamRole::Cell { .. }))
            .collect();
        let dissolved_bytes = dissolved.iter().map(|&k| g.rows * axis * ms[k].elem).sum();
        let state_bytes = (0..n)
            .map(|k| match (&roles[k], &ms[k].kind) {
                (StreamRole::Fold { .. }, _) => ms[k].elem,
                (StreamRole::Consume { .. }, UnitKind::Tile(t)) => t.c * ms[k].elem,
                _ => 0,
            })
            .sum();
        let guard = self.stream_guard(g, &roles, axis);
        Ok(StreamGroup {
            group: gi,
            axis,
            roles,
            dissolved,
            dissolved_bytes,
            state_bytes,
            passes: 1,
            guard,
        })
    }
}

impl CategoryIr {
    /// flash-attention F5/F6: the group's guarded producer, its mask's form
    /// and whether a wholly masked window may be skipped ([`StreamGuard`]).
    /// Off under `FactArms::guard_box`.
    fn stream_guard(&self, g: &UnitGroup, roles: &[StreamRole], axis: u64) -> Option<StreamGuard> {
        if !self.fact_arms().guard_box {
            return None;
        }
        // The members the pass covers (`StreamGroup::roles`).
        let ms = &g.members[..roles.len()];
        let producers: Vec<usize> = (0..ms.len())
            .filter(|&k| roles[k] == StreamRole::Produce)
            .collect();
        let (k, form) = producers.iter().find_map(|&k| {
            let UnitKind::Tile(t) = &ms[k].kind else {
                return None;
            };
            self.guard_box(ms[k].m, t, axis).map(|f| (k, f))
        })?;
        let f = self.try_owner(ms[k].out)?;
        let fin = self.finite_plan(f);
        let why = if producers.len() != 1 {
            "producers"
        } else if (0..ms.len()).any(|j| ms[j].w == axis && ms[j].stored) {
            "stored"
        } else if (0..ms.len()).any(|j| match (&roles[j], &ms[j].kind) {
            (StreamRole::Fold { .. }, UnitKind::Fold(s)) => !matches!(
                s.op,
                RowFoldOp::Select { .. }
                    | RowFoldOp::Arith {
                        op: Operation::Add,
                        ..
                    }
            ),
            (StreamRole::Cell { .. }, _) => !self.cell_index_addresses(ms[j].m),
            _ => false,
        }) {
            "term"
        } else if (0..ms.len()).any(|j| match (&roles[j], &ms[j].kind) {
            (StreamRole::Consume { .. }, UnitKind::Tile(t)) => self
                .pair_slot_source(self.morphisms[ms[j].m].source, t.b.slot)
                .is_none_or(|b| !fin.finite(b)),
            _ => false,
        }) {
            "finite"
        } else {
            ""
        };
        Some(StreamGuard {
            producer: k,
            form,
            skip: why.is_empty(),
            why,
        })
    }

    /// F5: tile site `m`'s epilogue mask as a [`GuardBox`] over its cells
    /// `e = row·w + lane` (`w` its lanes): the guard whose arm holds the
    /// fold makes the body's output, its other arm a float constant, its
    /// condition a comparison of two digit sums of the map index alone,
    /// each digit a digit of the row or of the lane.
    fn guard_box(&self, m: MorphismId, t: &crate::TileSite, w: u64) -> Option<GuardBox> {
        let post = t.post.as_ref()?;
        // mask-frontend: the sides are over the output's own rows — the
        // group's ([`crate::unit::out_rows`]: a row-wise mixed-radix output).
        if !post.guarded || crate::unit::out_rows(t, t.rows).is_err() || t.c != w {
            return None;
        }
        let Operation::Map { body, captures } = self.morphisms.get(m)?.op else {
            return None;
        };
        let bd = self.func(body)?;
        let gs = self
            .guard_structure(body)
            .into_iter()
            .find(|g| g.on_true.own.contains(&post.fold) || g.on_false.own.contains(&post.fold))?;
        if self.morphisms.get(gs.phi)?.target != bd.output {
            return None;
        }
        let fold_when = gs.on_true.own.contains(&post.fold);
        let other = if fold_when { &gs.on_false } else { &gs.on_true };
        let ov = self.objects.get(self.through_pairs(other.value))?;
        let masked = match (&ov.kind, &ov.value) {
            (ObjectKind::Constant, Some(Value::F32(x))) => f64::from(*x),
            (ObjectKind::Constant, Some(Value::F64(x))) => *x,
            _ => return None,
        };
        let [cm] = self.in_edges(gs.cond) else {
            return None;
        };
        let cmo = self.morphisms.get(*cm)?;
        if !matches!(
            cmo.op,
            Operation::Lt
                | Operation::Le
                | Operation::Gt
                | Operation::Ge
                | Operation::Eq
                | Operation::Neq
        ) {
            return None;
        }
        let source = self.morphisms.get(m)?.source;
        let forms_plain = self.radix_plan(body);
        // mask-frontend: a digit form as row / lane digits of the cell, a
        // captured number's digits folded into the base where it is a
        // literal, a per-launch term where it is not.
        let split =
            |form: &RadixForm| -> Option<(u64, Vec<AxisDigit>, Vec<AxisDigit>, Vec<Digit>)> {
                let (mut base, mut row, mut lane, mut uniform) =
                    (form.base, Vec::new(), Vec::new(), Vec::new());
                for d in &form.digits {
                    if d.stride == 0 {
                        return None;
                    }
                    if d.slot != captures {
                        let o = self.pair_slot_source(source, d.slot)?;
                        match self.tile_literal_u64(o) {
                            Some(v) => {
                                let q = v / d.stride;
                                let q = if d.radix == 0 { q } else { q % d.radix };
                                base = base.checked_add(d.coef.checked_mul(q)?)?;
                            }
                            None if matches!(self.objects.get(o)?.ty, Ty::Int { .. }) => {
                                uniform.push(*d)
                            }
                            None => return None,
                        }
                        continue;
                    }
                    let (s, r, c) = (d.stride, d.radix, d.coef);
                    if s.is_multiple_of(w) {
                        row.push(AxisDigit {
                            stride: s / w,
                            radix: r,
                            coef: c,
                        });
                    } else if r != 0 && w.is_multiple_of(s * r) {
                        lane.push(AxisDigit {
                            stride: s,
                            radix: r,
                            coef: c,
                        });
                    } else if r == 0 && w.is_multiple_of(s) {
                        row.push(AxisDigit {
                            stride: 1,
                            radix: 0,
                            coef: c * (w / s),
                        });
                        lane.push(AxisDigit {
                            stride: s,
                            radix: 0,
                            coef: c,
                        });
                    } else {
                        return None;
                    }
                }
                Some((base, row, lane, uniform))
            };
        let terms = self.fact_arms().guard_terms;
        let side = |o| -> Option<GuardSide> {
            let term = match terms {
                true => self.guard_term(source, body, captures, o, 0)?,
                false => SideTerm {
                    form: forms_plain.get(o)?.clone(),
                    load: None,
                },
            };
            let (base, row, lane, uniform) = split(&term.form)?;
            if !terms && !uniform.is_empty() {
                return None;
            }
            let load = match term.load {
                None => None,
                Some((slot, at, coef)) => {
                    // One value a row: the index of the row's digits alone.
                    let (ib, ir, il, iu) = split(&at)?;
                    if !il.is_empty() || !iu.is_empty() {
                        return None;
                    }
                    Some(RowLoad {
                        slot,
                        base: ib,
                        index: ir,
                        coef,
                    })
                }
            };
            Some(GuardSide {
                base,
                row,
                lane,
                uniform,
                load,
            })
        };
        let _ = bd;
        Some(GuardBox {
            cmp: cmo.op,
            lhs: side(self.pair_slot_source(cmo.source, 0)?)?,
            rhs: side(self.pair_slot_source(cmo.source, 1)?)?,
            fold_when,
            masked,
        })
    }

    /// mask-frontend: **a guard side's term** — value `o` of map body `body`
    /// (its source product `source`, `captures` captures) as a digit form of
    /// the body's inputs ([`Self::radix_plan`]) through `+`, `· literal` and
    /// reads of a captured integer array proven in bounds: an array whose
    /// element law is the index's affine image (`x = map(iota n) (t → a·t +
    /// c)`, `c` of literals: `elem_plan`'s law through the capture) reads as
    /// that image of the index's form; any other as the read itself — a
    /// per-row load (one a side).
    fn guard_term(
        &self,
        source: ObjectId,
        body: FuncId,
        captures: u32,
        o: ObjectId,
        depth: u32,
    ) -> Option<SideTerm> {
        if depth > 8 {
            return None;
        }
        let forms = self.radix_plan(body);
        if let Some(f) = forms.get(o) {
            return Some(SideTerm {
                form: f.clone(),
                load: None,
            });
        }
        if let Some(v) = self.tile_literal_u64(o) {
            return Some(SideTerm {
                form: RadixForm {
                    base: v,
                    digits: Vec::new(),
                },
                load: None,
            });
        }
        let [d] = self.in_edges(o) else {
            return None;
        };
        let dm = self.morphisms.get(*d)?;
        let arg = |k| self.pair_slot_source(dm.source, k);
        match dm.op {
            Operation::Add => {
                let x = self.guard_term(source, body, captures, arg(0)?, depth + 1)?;
                let y = self.guard_term(source, body, captures, arg(1)?, depth + 1)?;
                if x.load.is_some() && y.load.is_some() {
                    return None;
                }
                Some(SideTerm {
                    form: x.form.add(&y.form)?,
                    load: x.load.or(y.load),
                })
            }
            Operation::Mul => {
                let (c, x) = match (
                    self.tile_literal_u64(arg(0)?),
                    self.tile_literal_u64(arg(1)?),
                ) {
                    (Some(c), _) => (c, arg(1)?),
                    (_, Some(c)) => (c, arg(0)?),
                    _ => return None,
                };
                let t = self.guard_term(source, body, captures, x, depth + 1)?;
                let load = match t.load {
                    Some((s, at, k)) => Some((s, at, k.checked_mul(c)?)),
                    None => None,
                };
                Some(SideTerm {
                    form: t.form.scale(c)?,
                    load,
                })
            }
            Operation::Index => {
                let input = self.func(body)?.input;
                let slot = self
                    .tile_input_proj_index(arg(0)?, input)
                    .filter(|&s| s < captures)?;
                let at = forms.get(arg(1)?)?.clone();
                if !self.bounds_proof(body).proven(*d) {
                    return None;
                }
                let x = self.pair_slot_source(source, slot)?;
                let Ty::Array { elem, size } = &self.objects.get(x)?.ty else {
                    return None;
                };
                if !matches!(**elem, Ty::Int { .. }) {
                    return None;
                }
                if let Some(form) = self.affine_law_at(x, *size, &at) {
                    return Some(SideTerm { form, load: None });
                }
                Some(SideTerm {
                    form: RadixForm::default(),
                    load: Some((slot, at, 1)),
                })
            }
            _ => None,
        }
    }

    /// mask-frontend: `x[at]` where `x`'s element law is affine in its
    /// index — `x = map(iota n) (t → Σ coef·t + literals)` (its one
    /// definer a classifiable map over an `iota` of its length, the body's
    /// output a digit form whose every digit is the element itself or a
    /// literal capture's) — as that law at `at`'s form.
    fn affine_law_at(&self, x: ObjectId, n: u64, at: &RadixForm) -> Option<RadixForm> {
        let [xm] = self.in_edges(x) else {
            return None;
        };
        let morph = self.morphisms.get(*xm)?;
        let Operation::Map {
            body: g,
            captures: gc,
        } = morph.op
        else {
            return None;
        };
        self.map_law(*xm)?;
        let mapped = match gc {
            0 => morph.source,
            _ => self.pair_slot_source(morph.source, gc)?,
        };
        if self.tile_iota_size(mapped) != Some(n) {
            return None;
        }
        let law = self.radix_plan(g).get(self.func(g)?.output)?.clone();
        let mut out = RadixForm {
            base: law.base,
            digits: Vec::new(),
        };
        for d in &law.digits {
            if d.slot == gc {
                if d.stride != 1 || d.radix != 0 {
                    return None;
                }
                out = out.add(&at.scale(d.coef)?)?;
            } else {
                let v = self.tile_literal_u64(self.pair_slot_source(morph.source, d.slot)?)?;
                let q = v / d.stride;
                let q = if d.radix == 0 { q } else { q % d.radix };
                out.base = out.base.checked_add(d.coef.checked_mul(q)?)?;
            }
        }
        Some(out)
    }

    /// F6: cell map `m`'s body reads its index only to address the arrays it
    /// captures (its value at a cell is a function of the values it reads
    /// there), and reads no array but the group's (no bias at the cell).
    fn cell_index_addresses(&self, m: MorphismId) -> bool {
        let Some(Operation::Map { body, captures }) = self.morphisms.get(m).map(|x| x.op) else {
            return false;
        };
        let Some(bd) = self.func(body) else {
            return false;
        };
        // The index's uses through integer arithmetic must end as an
        // `Index`'s address (slot 1 of its pair).
        let mut stack: Vec<crate::graph::ObjectId> = self
            .out_edges(bd.input)
            .iter()
            .filter(|&&p| self.morphisms[p].op == (Operation::Proj { index: captures }))
            .map(|&p| self.morphisms[p].target)
            .collect();
        let mut seen = Vec::new();
        while let Some(o) = stack.pop() {
            if seen.contains(&o) {
                continue;
            }
            seen.push(o);
            if o == bd.output {
                return false;
            }
            for &u in self.out_edges(o) {
                let um = &self.morphisms[u];
                match um.op {
                    Operation::Pair { slot: 1, arity: 2 }
                        if self
                            .out_edges(um.target)
                            .iter()
                            .all(|&x| self.morphisms[x].op == Operation::Index) => {}
                    Operation::Add
                    | Operation::Sub
                    | Operation::Mul
                    | Operation::Div
                    | Operation::Mod
                    | Operation::Pair { .. } => stack.push(um.target),
                    _ => return false,
                }
            }
        }
        true
    }
}

/// A running max (`Some(true)`) or min (`Some(false)`) step; `None` for an
/// arithmetic one. `e` wins when it compares greater (`e > acc`, `acc < e`)
/// and the select picks `e`: a max.
pub fn extreme_up(op: RowFoldOp) -> Option<bool> {
    match op {
        RowFoldOp::Select {
            cmp,
            acc_lhs,
            acc_if_true,
        } => {
            let gt = matches!(cmp, Operation::Gt | Operation::Ge);
            Some(gt == (acc_lhs == acc_if_true))
        }
        RowFoldOp::Arith { .. } => None,
    }
}

impl Reach {
    /// A same-row read of a one-wide producer (`x[t ÷ w]`).
    pub fn is_row_of(&self, w: u64) -> bool {
        matches!(self, Reach::Affine { w: rw, base: 0, cq: 1, cr: 0, taps } if *rw == w && taps.is_empty())
    }
}
