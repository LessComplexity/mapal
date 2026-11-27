//! **The bounds fact** ([`CategoryIr::bounds_proof`]): which `Index` reads are
//! provably inside their array, from integer ranges over the object graph.
//!
//! Ranges come from constants, `iota` elements, enumerate indices, literal
//! ramps, map/fold quantification (a body's element param), captures (ranged
//! in their site's context, recursing out through enclosing bodies),
//! loop-carried counters (from [`CategoryIr::loop_plan`]: the init, an affine
//! step `t + c` and the continue condition bound the merge view), and the
//! conditions that hold where a value is used:
//!
//! - a guard arm's condition holds on the arm's exclusive work
//!   ([`CategoryIr::guard_structure`]), nested arms accumulating;
//! - a loop's continue condition holds on its advance cone, its exit
//!   condition on the exit payload (a loop's exit value read later);
//! - a `Phi`'s value is its true arm's under the condition or its false
//!   arm's under the negation (a select / piecewise clamp).
//!
//! A condition `a < b` (`≤ > ≥ = ≠`, `∧` on the true pole, `∨` on the false,
//! `¬` flipping) narrows both operands, and the narrowing runs back through
//! their definitions (`+ − ·k ÷k %k`, widen, projections), so `c % 128 < 64`
//! narrows `c` too and `a[c + 64]` is in bounds. Ranges are signed (`i128`
//! holds every `i64` and their sums); an operation whose result can leave its
//! type's range (wrap) is unranged. Conservative throughout: anything unknown
//! is not proven, and the consumers keep their trap guards there.
//!
//! An `Index` proven only under a guard arm's condition is safe only while
//! that arm runs alone: the proof records the arm ([`BoundsProof::arm_used`])
//! and [`CategoryIr::guard_plan`] gates it ([`crate::GuardArm::cond_reads`]).

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, Value};

mod compute;

/// A value range: a closed signed interval, or an enumerate element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rng {
    /// A value in `[lo, hi]`.
    Int(i128, i128),
    /// A `(i32, X)` enumerate element over `n`: its `.0` ∈ `[0, n)`.
    EnumIdx(u64),
}

/// The per-fn bounds proof (the deduced query `bounds_proof : IR × FuncId →
/// BoundsProof`, stored on the graph). See the module doc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundsProof {
    /// The `Operation::Index` morphisms proven in bounds.
    proven: SecondaryMap<MorphismId, ()>,
    /// Guard arms (their boundary `Pair` edge) whose condition a proof used.
    arms: SecondaryMap<MorphismId, ()>,
    /// Every `Index`'s index range where it runs (proven or not).
    index: SecondaryMap<MorphismId, (i128, i128)>,
    /// Each map/fold site's capture ranges where the site runs — what its
    /// body's capture params are seeded with.
    sites: SecondaryMap<MorphismId, Vec<Option<(i128, i128)>>>,
    /// caching-memory: each integer view of a canonical loop's merge (the
    /// counter as the body reads it) — its range over every trip.
    views: SecondaryMap<ObjectId, (i128, i128)>,
    /// ADR-0051 D4: each `Widen` of an integer — its source's range where it
    /// runs (the exactness of `widen<f32>` of an `i32`).
    widen: SecondaryMap<MorphismId, (i128, i128)>,
}

impl BoundsProof {
    /// Whether `m` (an `Operation::Index` morphism) is provably in bounds
    /// (where it runs — see [`Self::arm_used`]).
    pub fn proven(&self, m: MorphismId) -> bool {
        self.proven.contains_key(m)
    }

    /// Whether a proof holds only under the condition of the guard arm whose
    /// boundary `Pair` edge is `edge` (the arm must run gated).
    pub fn arm_used(&self, edge: MorphismId) -> bool {
        self.arms.contains_key(edge)
    }

    /// The range of `m`'s index where `m` runs (`m` an `Index`), if known.
    pub fn index_range(&self, m: MorphismId) -> Option<(i128, i128)> {
        self.index.get(m).copied()
    }

    /// The range of map/fold site `m`'s capture `slot` (an integer) where
    /// `m` runs, if known — what its body's capture param is seeded with.
    pub fn capture_range(&self, m: MorphismId, slot: u32) -> Option<(i128, i128)> {
        *self.sites.get(m)?.get(slot as usize)?
    }

    /// The range of loop merge view `v` (an integer state component, as the
    /// loop body reads it) over every trip, if known.
    pub fn view_range(&self, v: ObjectId) -> Option<(i128, i128)> {
        self.views.get(v).copied()
    }

    /// ADR-0051 D4: the range of the integer `m` widens (`m` a `Widen` of an
    /// integer) where `m` runs, if known — under the same conditions an
    /// index's range is (guard arms, loop conditions, captured ranges).
    pub fn widen_range(&self, m: MorphismId) -> Option<(i128, i128)> {
        self.widen.get(m).copied()
    }
}

/// A condition object and the pole it holds at.
type Cond = (ObjectId, bool);

/// One link of a morphism's context: a condition, and the guard arm edge it
/// comes from (`None`: a loop's continue condition — no gating needed).
type CtxLink = (ObjectId, bool, Option<MorphismId>);

type RngMap = SecondaryMap<ObjectId, Rng>;
type Pins = SecondaryMap<ObjectId, (i128, i128)>;

/// The `[min, max]` of an int type (`None`: not an int).
fn type_range(ty: &Ty) -> Option<(i128, i128)> {
    match ty {
        Ty::Int {
            bits: 8,
            signed: false,
        } => Some((0, 255)),
        Ty::Int { bits: 32, .. } => Some((i32::MIN as i128, i32::MAX as i128)),
        Ty::Int { bits: 64, .. } => Some((i64::MIN as i128, i64::MAX as i128)),
        _ => None,
    }
}

fn int(r: Option<Rng>) -> Option<(i128, i128)> {
    match r {
        Some(Rng::Int(lo, hi)) => Some((lo, hi)),
        _ => None,
    }
}

fn meet(a: (i128, i128), b: (i128, i128)) -> Option<(i128, i128)> {
    let r = (a.0.max(b.0), a.1.min(b.1));
    (r.0 <= r.1).then_some(r)
}

fn hull(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    (a.0.min(b.0), a.1.max(b.1))
}

/// Interval arithmetic in the target's int type: `None` when the result can
/// leave the type's range (wrap), a divisor range holds zero, or an operand
/// is unknown.
fn arith(
    op: Operation,
    a: (i128, i128),
    b: (i128, i128),
    ty: (i128, i128),
) -> Option<(i128, i128)> {
    let (al, ah) = a;
    let (bl, bh) = b;
    let r = match op {
        Operation::Add => (al + bl, ah + bh),
        Operation::Sub => (al - bh, ah - bl),
        Operation::Mul => {
            let c = [al * bl, al * bh, ah * bl, ah * bh];
            (*c.iter().min()?, *c.iter().max()?)
        }
        Operation::Div => {
            if bl <= 0 && bh >= 0 {
                return None;
            }
            let c = [al / bl, al / bh, ah / bl, ah / bh];
            (*c.iter().min()?, *c.iter().max()?)
        }
        Operation::Mod => {
            if bl <= 0 && bh >= 0 {
                return None;
            }
            let small = bl.abs().min(bh.abs());
            let m = bl.abs().max(bh.abs()) - 1;
            if al >= 0 && ah < small {
                (al, ah)
            } else if al >= 0 {
                (0, ah.min(m))
            } else if ah <= 0 {
                (al.max(-m), 0)
            } else {
                (al.max(-m), ah.min(m))
            }
        }
        _ => return None,
    };
    (r.0 >= ty.0 && r.1 <= ty.1).then_some(r)
}

/// The integer value of a constant object, if it is one.
fn const_int(ir: &CategoryIr, o: ObjectId) -> Option<i128> {
    let obj = ir.object(o)?;
    if obj.kind != ObjectKind::Constant {
        return None;
    }
    match obj.value {
        Some(Value::I32(x)) => Some(x.into()),
        Some(Value::I64(x)) => Some(x.into()),
        Some(Value::U8(x)) => Some(x.into()),
        _ => None,
    }
}

/// One function's range evaluation.
struct Eval<'a> {
    ir: &'a CategoryIr,
    f: FuncId,
    topo: Vec<MorphismId>,
    /// Topo position of each of `f`'s morphisms (cone re-evaluation order).
    pos: SecondaryMap<MorphismId, usize>,
    /// Ranges known without evaluation: constants, body params, captures.
    seeds: RngMap,
    /// Per loop merge, each carried slot's range over every iteration.
    merges: SecondaryMap<ObjectId, Vec<Option<(i128, i128)>>>,
    /// Per `LoopExit` morphism, its exit condition (the payload's context).
    exit_cond: SecondaryMap<MorphismId, ObjectId>,
    /// `cond-bounds` (`FactArms`): `Phi` arms ranged under their condition.
    cond: bool,
}

impl Eval<'_> {
    fn ty_range(&self, o: ObjectId) -> Option<(i128, i128)> {
        type_range(&self.ir.objects[o].ty)
    }

    /// The single defining morphism of `o` inside `f` (none for seeds,
    /// multi-producer objects or foreign objects).
    fn def(&self, o: ObjectId) -> Option<MorphismId> {
        match self.ir.in_edges(o) {
            [m] if self.pos.contains_key(*m) => Some(*m),
            _ => None,
        }
    }

    fn slot(&self, product: ObjectId, k: u32) -> Option<ObjectId> {
        self.ir.pair_slot_source(product, k)
    }

    /// One morphism's target range from the current map (`depth > 0`: a cone
    /// re-evaluation — `Phi` and exits keep the map's value).
    fn step(&self, m: MorphismId, rng: &RngMap, depth: u32) -> Option<Rng> {
        let ir = self.ir;
        let morph = &ir.morphisms[m];
        let t = morph.target;
        let get = |o: Option<ObjectId>| o.and_then(|o| int(rng.get(o).copied()));
        match morph.op {
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                let x = self.slot(morph.source, 0);
                // A remainder's range needs no range of its dividend: any
                // value of the type (a wrapped product included) leaves
                // `x % k` inside `(−|k|, |k|)` (ADR-0051: `(t * 11 + 57) % 103`
                // past `t * 11`'s `i32` range).
                let a = match (get(x), morph.op) {
                    (Some(a), _) => a,
                    (None, Operation::Mod) => self.ty_range(x?)?,
                    (None, _) => return None,
                };
                let b = get(self.slot(morph.source, 1))?;
                arith(morph.op, a, b, self.ty_range(t)?).map(|(l, h)| Rng::Int(l, h))
            }
            Operation::Neg => {
                let (l, h) = get(Some(morph.source))?;
                let ty = self.ty_range(t)?;
                (-h >= ty.0 && -l <= ty.1).then_some(Rng::Int(-h, -l))
            }
            Operation::Widen => {
                self.ty_range(t)?;
                let (l, h) = get(Some(morph.source))?;
                Some(Rng::Int(l, h))
            }
            Operation::Proj { index } => {
                let src = morph.source;
                if let Some(slots) = self.merges.get(src) {
                    return slots
                        .get(index as usize)
                        .copied()
                        .flatten()
                        .map(|(l, h)| Rng::Int(l, h));
                }
                if let Some(Rng::EnumIdx(n)) = rng.get(src).copied()
                    && index == 0
                {
                    return Some(Rng::Int(0, n as i128 - 1));
                }
                if depth == 0
                    && let Some(&x) = ir.in_edges(src).first()
                    && let Some(&c) = self.exit_cond.get(x)
                {
                    // A component of a loop's exit value: the payload's
                    // component, under the exit condition.
                    let payload = self.slot(ir.morphisms[x].source, 0)?;
                    let comp = self.slot(payload, index)?;
                    return self.under(rng, comp, &[(c, true)]);
                }
                // A view of a product no `Pair` builds (a body's input): its
                // own range — its seed, met with any pin a sibling view of
                // the same slot passed on (`narrow`); a component of a
                // selecting fold's result: its `select_range`.
                match self.slot(src, index) {
                    Some(s) => rng.get(s).copied(),
                    None => rng
                        .get(t)
                        .copied()
                        .or_else(|| self.select_range(rng, src, index)),
                }
            }
            Operation::Index => {
                let arr = self.slot(morph.source, 0)?;
                ir.element_range(arr)
            }
            Operation::LoopEnter => self
                .merges
                .get(t)?
                .first()
                .copied()
                .flatten()
                .and_then(|r| {
                    // A one-value loop: the merge object itself is the view.
                    type_range(&ir.objects[t].ty).map(|_| Rng::Int(r.0, r.1))
                }),
            Operation::LoopExit if depth == 0 => {
                let c = *self.exit_cond.get(m)?;
                let payload = self.slot(morph.source, 0)?;
                type_range(&ir.objects[payload].ty)?;
                self.under(rng, payload, &[(c, true)])
            }
            Operation::Phi if depth == 0 && self.cond => {
                let tv = self.slot(morph.source, 0)?;
                let fv = self.slot(morph.source, 1)?;
                let c = self.slot(morph.source, 2)?;
                self.ty_range(t)?;
                let a = int(self.under(rng, tv, &[(c, true)]))?;
                let b = int(self.under(rng, fv, &[(c, false)]))?;
                let (l, h) = hull(a, b);
                Some(Rng::Int(l, h))
            }
            _ => None,
        }
    }

    /// Component `k` of a fold's result when the fold **selects** it: every
    /// step's new component `k` is the accumulator's own, the step's element
    /// or a constant (an argmax / argmin / first-match index). Its range is
    /// the hull of the seed's component, the items' element range and those
    /// constants — the values it can ever hold.
    fn select_range(&self, rng: &RngMap, folded: ObjectId, k: u32) -> Option<Rng> {
        let ir = self.ir;
        let m = self.def(folded)?;
        let Operation::Fold { body, captures } = ir.morphisms[m].op else {
            return None;
        };
        let src = ir.morphisms[m].source;
        let seed = self.slot(src, captures)?;
        let items = self.slot(src, captures + 1)?;
        let mut range = int(rng.get(self.slot(seed, k)?).copied())?;
        let bd = ir.func(body)?;
        let input = bd.input;
        // The body param a view `Proj{slot}` of the body's input stands for.
        let param = |o: ObjectId| -> Option<u32> {
            let &e = ir.in_edges(o).first()?;
            match ir.morphisms[e].op {
                Operation::Proj { index } if ir.morphisms[e].source == input => Some(index),
                _ => None,
            }
        };
        let mut todo = vec![bd.output];
        let mut seen = 0;
        while let Some(o) = todo.pop() {
            seen += 1;
            if seen > 64 {
                return None;
            }
            // A whole step result: the accumulator itself, a `Phi` of step
            // results, or a pack whose component `k` is an allowed value.
            if param(o) == Some(captures) {
                continue;
            }
            let o = ir.through_pairs(o);
            if param(o) == Some(captures) {
                continue;
            }
            let &e = ir.in_edges(o).first()?;
            match ir.morphisms[e].op {
                Operation::Phi => {
                    let ps = ir.morphisms[e].source;
                    todo.push(self.slot(ps, 0)?);
                    todo.push(self.slot(ps, 1)?);
                }
                Operation::Pair { .. } => {
                    let c = ir.through_pairs(self.slot(o, k)?);
                    if let Some(v) = const_int(ir, c) {
                        range = hull(range, (v, v));
                        continue;
                    }
                    if param(c) == Some(captures + 1) {
                        range = hull(range, int(ir.element_range(items))?);
                        continue;
                    }
                    // Else only the accumulator's own component `k`.
                    let &ce = ir.in_edges(c).first()?;
                    let cm = &ir.morphisms[ce];
                    if cm.op != (Operation::Proj { index: k }) || param(cm.source) != Some(captures)
                    {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        Some(Rng::Int(range.0, range.1))
    }

    /// The full forward pass: seeds, then every morphism in topo order, each
    /// result met with its pin.
    fn forward(&self, pins: &Pins) -> RngMap {
        let mut rng = self.seeds.clone();
        for (o, &p) in pins.iter() {
            let r = match int(rng.get(o).copied()) {
                Some(cur) => meet(cur, p).unwrap_or(cur),
                None => p,
            };
            rng.insert(o, Rng::Int(r.0, r.1));
        }
        for &m in &self.topo {
            let t = self.ir.morphisms[m].target;
            if let Some(r) = self.step(m, &rng, 0) {
                rng.insert(t, Self::pinned(r, pins.get(t).copied()));
            }
        }
        rng
    }

    fn pinned(r: Rng, pin: Option<(i128, i128)>) -> Rng {
        match (r, pin) {
            (Rng::Int(l, h), Some(p)) => {
                let (l, h) = meet((l, h), p).unwrap_or((l, h));
                Rng::Int(l, h)
            }
            _ => r,
        }
    }

    /// `rng` re-evaluated under `conds`: pins from the conditions (two rounds,
    /// the second against the first's ranges).
    fn context(&self, rng: &RngMap, conds: &[Cond]) -> RngMap {
        if conds.is_empty() {
            return rng.clone();
        }
        let mut cur = rng.clone();
        let mut pins = Pins::new();
        for _ in 0..2 {
            for &(c, pole) in conds {
                self.assume(&cur, &mut pins, c, pole, 0);
            }
            cur = self.forward(&pins);
        }
        cur
    }

    /// The range of `v` under `conds`: pins, then a re-evaluation of `v`'s
    /// backward cone only (`Phi`s and exits inside keep their range).
    fn under(&self, rng: &RngMap, v: ObjectId, conds: &[Cond]) -> Option<Rng> {
        let pins = self.constrain(rng, conds);
        let mut cone: Vec<MorphismId> = Vec::new();
        let mut seen: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut stack = vec![v];
        while let Some(o) = stack.pop() {
            if seen.insert(o, ()).is_some() || cone.len() > 512 {
                continue;
            }
            if let Some(m) = self.def(o) {
                cone.push(m);
                let src = self.ir.morphisms[m].source;
                match self.ir.morphisms[m].op {
                    Operation::Phi | Operation::LoopExit | Operation::LoopEnter => {}
                    _ => {
                        stack.push(src);
                        for &e in self.ir.in_edges(src) {
                            if matches!(self.ir.morphisms[e].op, Operation::Pair { .. }) {
                                stack.push(self.ir.morphisms[e].source);
                            }
                        }
                    }
                }
            }
        }
        cone.sort_by_key(|m| self.pos[*m]);
        let mut local = rng.clone();
        for (o, &p) in pins.iter() {
            if self.def(o).is_none() || !cone.contains(&self.def(o).unwrap()) {
                let r = match int(local.get(o).copied()) {
                    Some(cur) => meet(cur, p).unwrap_or(cur),
                    None => p,
                };
                local.insert(o, Rng::Int(r.0, r.1));
            }
        }
        for m in cone {
            let t = self.ir.morphisms[m].target;
            if let Some(r) = self.step(m, &local, 1) {
                local.insert(t, Self::pinned(r, pins.get(t).copied()));
            }
        }
        local.get(v).copied()
    }

    /// Pins implied by `conds` holding, against the ranges in `rng`.
    fn constrain(&self, rng: &RngMap, conds: &[Cond]) -> Pins {
        let mut pins = Pins::new();
        for &(c, pole) in conds {
            self.assume(rng, &mut pins, c, pole, 0);
        }
        pins
    }

    /// The current range of `o`: its pin met with its range, else either,
    /// else its type's whole range.
    fn cur(&self, rng: &RngMap, pins: &Pins, o: ObjectId) -> Option<(i128, i128)> {
        let r = int(rng.get(o).copied());
        match (r, pins.get(o).copied()) {
            (Some(r), Some(p)) => Some(meet(r, p).unwrap_or(r)),
            (Some(r), None) => Some(r),
            (None, Some(p)) => Some(p),
            (None, None) => self.ty_range(o),
        }
    }

    fn assume(&self, rng: &RngMap, pins: &mut Pins, c: ObjectId, pole: bool, depth: u32) {
        if depth > 16 {
            return;
        }
        let Some(m) = self.def(c) else { return };
        let morph = &self.ir.morphisms[m];
        let (a, b) = (self.slot(morph.source, 0), self.slot(morph.source, 1));
        match morph.op {
            Operation::Not => self.assume(rng, pins, morph.source, !pole, depth + 1),
            Operation::And if pole => {
                for x in [a, b].into_iter().flatten() {
                    self.assume(rng, pins, x, true, depth + 1);
                }
            }
            Operation::Or if !pole => {
                for x in [a, b].into_iter().flatten() {
                    self.assume(rng, pins, x, false, depth + 1);
                }
            }
            Operation::Proj { index } => {
                if let Some(x) = self.slot(morph.source, index) {
                    self.assume(rng, pins, x, pole, depth + 1);
                }
            }
            Operation::Lt
            | Operation::Le
            | Operation::Gt
            | Operation::Ge
            | Operation::Eq
            | Operation::Neq => {
                let (Some(a), Some(b)) = (a, b) else { return };
                let (Some(ta), Some(tb)) = (self.ty_range(a), self.ty_range(b)) else {
                    return;
                };
                // Normalize to `a rel b` holding.
                let rel = match (morph.op, pole) {
                    (Operation::Lt, true) | (Operation::Ge, false) => Operation::Lt,
                    (Operation::Le, true) | (Operation::Gt, false) => Operation::Le,
                    (Operation::Gt, true) | (Operation::Le, false) => Operation::Gt,
                    (Operation::Ge, true) | (Operation::Lt, false) => Operation::Ge,
                    (Operation::Eq, true) | (Operation::Neq, false) => Operation::Eq,
                    _ => Operation::Neq,
                };
                let (Some(ra), Some(rb)) = (self.cur(rng, pins, a), self.cur(rng, pins, b)) else {
                    return;
                };
                let (na, nb) = match rel {
                    Operation::Lt => ((ta.0, rb.1 - 1), (ra.0 + 1, tb.1)),
                    Operation::Le => ((ta.0, rb.1), (ra.0, tb.1)),
                    Operation::Gt => ((rb.0 + 1, ta.1), (tb.0, ra.1 - 1)),
                    Operation::Ge => ((rb.0, ta.1), (tb.0, ra.1)),
                    Operation::Eq => (rb, ra),
                    _ => {
                        // `a ≠ b` with `b` one value at an end of `a`.
                        let na = if rb.0 == rb.1 && rb.0 == ra.0 {
                            (ra.0 + 1, ra.1)
                        } else if rb.0 == rb.1 && rb.0 == ra.1 {
                            (ra.0, ra.1 - 1)
                        } else {
                            ra
                        };
                        (na, rb)
                    }
                };
                self.narrow(rng, pins, a, na, 0);
                self.narrow(rng, pins, b, nb, 0);
            }
            _ => {}
        }
    }

    /// Pin `o` to `r` (met with its current range) and run the narrowing back
    /// through `o`'s definition.
    fn narrow(&self, rng: &RngMap, pins: &mut Pins, o: ObjectId, r: (i128, i128), depth: u32) {
        if depth > 16 {
            return;
        }
        let Some(cur) = self.cur(rng, pins, o) else {
            return;
        };
        // An empty meet is an infeasible context: keep the range (sound).
        let Some(new) = meet(cur, r) else { return };
        if pins.get(o) == Some(&new) {
            return;
        }
        pins.insert(o, new);
        let Some(m) = self.def(o) else { return };
        let morph = &self.ir.morphisms[m];
        let (a, b) = (self.slot(morph.source, 0), self.slot(morph.source, 1));
        // The forward range must exist (no wrap) for the inverse to be exact.
        let exact = int(rng.get(o).copied()).is_some();
        let cur_of = |x: Option<ObjectId>| x.and_then(|x| self.cur(rng, pins, x));
        match morph.op {
            Operation::Add | Operation::Sub if exact => {
                let (Some(xa), Some(xb), Some(ra), Some(rb)) = (a, b, cur_of(a), cur_of(b)) else {
                    return;
                };
                let (na, nb) = if morph.op == Operation::Add {
                    ((new.0 - rb.1, new.1 - rb.0), (new.0 - ra.1, new.1 - ra.0))
                } else {
                    ((new.0 + rb.0, new.1 + rb.1), (ra.0 - new.1, ra.1 - new.0))
                };
                self.narrow(rng, pins, xa, na, depth + 1);
                self.narrow(rng, pins, xb, nb, depth + 1);
            }
            Operation::Mul if exact => {
                let (Some(xa), Some(xb)) = (a, b) else { return };
                for (x, k) in [(xa, const_int(self.ir, xb)), (xb, const_int(self.ir, xa))] {
                    if let Some(k) = k
                        && k > 0
                    {
                        let lo = new.0.div_euclid(k) + i128::from(new.0.rem_euclid(k) != 0);
                        let hi = new.1.div_euclid(k);
                        self.narrow(rng, pins, x, (lo, hi), depth + 1);
                    }
                }
            }
            Operation::Div => {
                let (Some(xa), Some(k), Some(ra)) =
                    (a, b.and_then(|b| const_int(self.ir, b)), cur_of(a))
                else {
                    return;
                };
                if k > 0 && ra.0 >= 0 && new.0 >= 0 {
                    self.narrow(rng, pins, xa, (new.0 * k, new.1 * k + k - 1), depth + 1);
                }
            }
            Operation::Mod => {
                let (Some(xa), Some(k), Some(ra)) =
                    (a, b.and_then(|b| const_int(self.ir, b)), cur_of(a))
                else {
                    return;
                };
                if k <= 0 || ra.0 < 0 {
                    return;
                }
                let (rl, rh) = (new.0.max(0), new.1.min(k - 1));
                if rl > rh {
                    return;
                }
                // The least `x ≥ ra.0` and the greatest `x ≤ ra.1` with
                // `x % k ∈ [rl, rh]`.
                let (q0, r0) = (ra.0 / k, ra.0 % k);
                let lo = if r0 <= rh {
                    q0 * k + r0.max(rl)
                } else {
                    (q0 + 1) * k + rl
                };
                let (q1, r1) = (ra.1 / k, ra.1 % k);
                let hi = if r1 >= rl {
                    q1 * k + r1.min(rh)
                } else {
                    (q1 - 1) * k + rh
                };
                if lo <= hi {
                    self.narrow(rng, pins, xa, (lo, hi), depth + 1);
                }
            }
            Operation::Widen => {
                if self.ty_range(morph.source).is_some() {
                    self.narrow(rng, pins, morph.source, new, depth + 1);
                }
            }
            Operation::Proj { index } => match self.slot(morph.source, index) {
                Some(x) => self.narrow(rng, pins, x, new, depth + 1),
                // Views of one slot of a product no `Pair` builds (a body's
                // input: `i` read twice is two `Proj`s) are one value: each
                // sibling view takes the pin too.
                None => {
                    let ir = self.ir;
                    for &e in ir.out_edges(morph.source) {
                        let sib = ir.morphisms[e].target;
                        if sib != o
                            && ir.morphisms[e].op == (Operation::Proj { index })
                            && let Some(c) = self.cur(rng, pins, sib)
                            && let Some(r) = meet(c, new)
                        {
                            pins.insert(sib, r);
                        }
                    }
                }
            },
            _ => {}
        }
    }

    /// Each canonical loop's carried int slots over every iteration: init
    /// `I`, next `t + c` (`c` constant), and the continue condition's bound on
    /// `t` (`t ≤ U` for `c > 0`, `t ≥ L` for `c < 0`) give `[I.lo, max(I.hi,
    /// U + c)]` (resp. `[min(I.lo, L + c), I.hi]`), unranged when `U + c`
    /// leaves the type; a constant next `b` gives `I ∪ {b}`.
    fn loop_ranges(&mut self, rng: &RngMap) {
        let ir = self.ir;
        for scc in ir.loop_structure(self.f) {
            for &merge in &scc.merges {
                let Some(plan) = ir.loop_plan(self.f, merge) else {
                    continue;
                };
                let (Some(next), Some(cond)) =
                    (self.slot(plan.back_route, 0), self.slot(plan.back_route, 1))
                else {
                    continue;
                };
                if let Some(cx) = self.slot(plan.exit_route, 1) {
                    for &x in &plan.exits {
                        self.exit_cond.insert(x, cx);
                    }
                }
                let n = match &ir.objects[merge].ty {
                    Ty::Tuple(tys) => tys.len() as u32,
                    _ => 1,
                };
                let tuple = n > 1 || matches!(ir.objects[merge].ty, Ty::Tuple(_));
                let mut slots: Vec<Option<(i128, i128)>> = Vec::new();
                for k in 0..n {
                    let comp = |o: ObjectId| if tuple { self.slot(o, k) } else { Some(o) };
                    let views: Vec<ObjectId> = if tuple {
                        ir.out_edges(merge)
                            .iter()
                            .filter(|&&e| ir.morphisms[e].op == (Operation::Proj { index: k }))
                            .map(|&e| ir.morphisms[e].target)
                            .collect()
                    } else {
                        vec![merge]
                    };
                    slots.push(self.slot_range(
                        rng,
                        merge,
                        comp(plan.init),
                        comp(next),
                        &views,
                        cond,
                    ));
                }
                self.merges.insert(merge, slots);
            }
        }
    }

    /// `o` as `a·t + b` over the slot's views `t` (exact; `+ − ·const`,
    /// `Proj`-of-`Pair` hops; any other view of `merge` bails).
    fn affine(
        &self,
        o: ObjectId,
        merge: ObjectId,
        views: &[ObjectId],
        depth: u32,
    ) -> Option<(i128, i128)> {
        if depth > 64 {
            return None;
        }
        if views.contains(&o) {
            return Some((1, 0));
        }
        if let Some(v) = const_int(self.ir, o) {
            return Some((0, v));
        }
        let m = self.def(o)?;
        let morph = &self.ir.morphisms[m];
        match morph.op {
            Operation::Proj { index } if morph.source != merge => {
                self.affine(self.slot(morph.source, index)?, merge, views, depth + 1)
            }
            Operation::Add | Operation::Sub | Operation::Mul => {
                let (xa, xb) = self.affine(self.slot(morph.source, 0)?, merge, views, depth + 1)?;
                let (ya, yb) = self.affine(self.slot(morph.source, 1)?, merge, views, depth + 1)?;
                match morph.op {
                    Operation::Add => Some((xa + ya, xb + yb)),
                    Operation::Sub => Some((xa - ya, xb - yb)),
                    _ if xa == 0 => Some((ya * xb, yb * xb)),
                    _ if ya == 0 => Some((xa * yb, xb * yb)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn slot_range(
        &self,
        rng: &RngMap,
        merge: ObjectId,
        init: Option<ObjectId>,
        next: Option<ObjectId>,
        views: &[ObjectId],
        cond: ObjectId,
    ) -> Option<(i128, i128)> {
        let ty = type_range(&self.ir.objects[*views.first()?].ty)?;
        let i = int(rng.get(init?).copied())?;
        let (a, c) = self.affine(next?, merge, views, 0)?;
        if a == 0 {
            return (c >= ty.0 && c <= ty.1).then_some(hull(i, (c, c)));
        }
        if a != 1 {
            return None;
        }
        if c == 0 {
            return Some(i);
        }
        // The continue condition's bound on the view (whole type range in).
        let mut open = rng.clone();
        for &v in views {
            open.insert(v, Rng::Int(ty.0, ty.1));
        }
        let pins = self.constrain(&open, &[(cond, true)]);
        let mut b = ty;
        for &v in views {
            if let Some(p) = pins.get(v) {
                b = meet(b, *p)?;
            }
        }
        if c > 0 {
            let top = b.1.checked_add(c)?;
            (b.1 < ty.1 && top <= ty.1).then_some((i.0, i.1.max(top)))
        } else {
            let bot = b.0.checked_add(c)?;
            (b.0 > ty.0 && bot >= ty.0).then_some((i.0.min(bot), i.1))
        }
    }
}
