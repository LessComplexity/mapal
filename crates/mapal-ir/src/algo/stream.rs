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
    AxisDigit, DepLaw, FoldCombinePlan, FootprintPlan, Reach, RescaleArg, RowFoldOp,
};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectKind, Operation};
use crate::ty::Value;
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
/// radix)` (`radix == 0`: the quotient itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardSide {
    pub base: u64,
    pub row: Vec<AxisDigit>,
    pub lane: Vec<AxisDigit>,
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
    /// The side's range over rows `[r0, r1]` and lanes `[l0, l1]`.
    pub fn range(&self, (r0, r1): (u64, u64), (l0, l1): (u64, u64)) -> (u64, u64) {
        let mut lo = self.base;
        let mut hi = self.base;
        for (ds, a, b) in [(&self.row, r0, r1), (&self.lane, l0, l1)] {
            for d in ds {
                let (x, y) = digit_range(d, a, b);
                lo += x;
                hi += y;
            }
        }
        (lo, hi)
    }

    /// The side at one cell.
    pub fn at(&self, row: u64, lane: u64) -> u64 {
        self.range((row, row), (lane, lane)).0
    }
}

impl GuardBox {
    /// The comparison over rows `[r0, r1]` × lanes `[l0, l1]`: always true
    /// (`Some(true)`), always false, or either.
    fn holds(&self, rows: (u64, u64), lanes: (u64, u64)) -> Option<bool> {
        let (a, b) = (self.lhs.range(rows, lanes), self.rhs.range(rows, lanes));
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

    /// The box's class (module doc of [`GuardBox`]).
    pub fn classify(&self, rows: (u64, u64), lanes: (u64, u64)) -> BoxClass {
        match self.holds(rows, lanes) {
            Some(c) if c == self.fold_when => BoxClass::Visible,
            Some(_) => BoxClass::Masked,
            None => BoxClass::Mixed,
        }
    }

    /// One cell: is its fold computed?
    pub fn visible(&self, row: u64, lane: u64) -> bool {
        let (a, b) = (self.lhs.at(row, lane), self.rhs.at(row, lane));
        let c = match self.cmp {
            Operation::Le => a <= b,
            Operation::Lt => a < b,
            Operation::Ge => a >= b,
            Operation::Gt => a > b,
            Operation::Eq => a == b,
            _ => a != b,
        };
        c == self.fold_when
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
        if !post.guarded || t.radix.as_ref().is_some_and(|r| r.out.is_some()) || t.c != w {
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
        let forms = self.radix_plan(body);
        let side = |o| -> Option<GuardSide> {
            let form = forms.get(o)?;
            let mut out = GuardSide {
                base: form.base,
                row: Vec::new(),
                lane: Vec::new(),
            };
            for d in &form.digits {
                if d.slot != captures || d.stride == 0 {
                    return None;
                }
                let (s, r, c) = (d.stride, d.radix, d.coef);
                if s.is_multiple_of(w) {
                    out.row.push(AxisDigit {
                        stride: s / w,
                        radix: r,
                        coef: c,
                    });
                } else if r != 0 && w.is_multiple_of(s * r) {
                    out.lane.push(AxisDigit {
                        stride: s,
                        radix: r,
                        coef: c,
                    });
                } else if r == 0 && w.is_multiple_of(s) {
                    out.row.push(AxisDigit {
                        stride: 1,
                        radix: 0,
                        coef: c * (w / s),
                    });
                    out.lane.push(AxisDigit {
                        stride: s,
                        radix: 0,
                        coef: c,
                    });
                } else {
                    return None;
                }
            }
            Some(out)
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
