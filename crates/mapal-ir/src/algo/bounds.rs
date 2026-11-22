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
                let a = get(self.slot(morph.source, 0))?;
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
                self.slot(src, index).and_then(|s| rng.get(s).copied())
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
            Operation::Proj { index } => {
                if let Some(x) = self.slot(morph.source, index) {
                    self.narrow(rng, pins, x, new, depth + 1);
                }
            }
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

impl CategoryIr {
    /// The bounds-proof query (see the module doc): which of `f`'s `Index`
    /// morphisms are provably in bounds where they run, each index's range,
    /// and the guard arms a proof needs gated. Stored on the graph.
    pub fn bounds_proof(&self, f: FuncId) -> BoundsProof {
        self.fact(f, |s| &s.bounds_proof, || self.bounds_proof_compute(f))
    }

    fn bounds_proof_compute(&self, f: FuncId) -> BoundsProof {
        let mut out = BoundsProof::default();
        let Some(fd) = self.func(f) else {
            return out;
        };
        let topo = self.topo_order(f);
        let mut pos = SecondaryMap::new();
        for (i, &m) in topo.iter().enumerate() {
            pos.insert(m, i);
        }
        let mut ev = Eval {
            ir: self,
            f,
            topo,
            pos,
            seeds: RngMap::new(),
            merges: SecondaryMap::new(),
            exit_cond: SecondaryMap::new(),
            cond: true,
        };
        // --- seeds: constants, then the body params ---------------------------
        for o in self.func_objects(f) {
            if let Some(v) = const_int(self, o) {
                ev.seeds.insert(o, Rng::Int(v, v));
            }
        }
        self.seed_body(f, fd.input, &mut ev.seeds);

        // `cond-bounds` off (`FactArms`): plain intervals — no loop ranges, no
        // conditions (a `Phi` and an exit are unranged), the proof before it.
        let cond = self.fact_arms().cond_bounds;
        ev.cond = cond;

        // --- loops: carried slot ranges from the unlooped pass ----------------
        let flat = ev.forward(&Pins::new());
        if cond {
            ev.loop_ranges(&flat);
        }
        let base = ev.forward(&Pins::new());

        // --- contexts: the conditions holding where each morphism runs --------
        let links = if cond {
            self.bounds_links(f)
        } else {
            SecondaryMap::new()
        };
        let mut memo: SecondaryMap<MorphismId, Vec<CtxLink>> = SecondaryMap::new();
        let mut cache: Vec<(Vec<Cond>, RngMap)> = Vec::new();
        let mut at = |conds: Vec<Cond>| -> RngMap {
            if conds.is_empty() {
                return base.clone();
            }
            if let Some((_, r)) = cache.iter().find(|(c, _)| *c == conds) {
                return r.clone();
            }
            let r = ev.context(&base, &conds);
            cache.push((conds, r.clone()));
            r
        };
        for &m in &fd.morphisms {
            let morph = &self.morphisms[m];
            match morph.op {
                Operation::Index => {
                    let (Some(arr), Some(idx)) = (
                        self.pair_slot_source(morph.source, 0),
                        self.pair_slot_source(morph.source, 1),
                    ) else {
                        continue;
                    };
                    let Ty::Array { size, .. } = self.objects[arr].ty else {
                        continue;
                    };
                    let inside = |r: Option<(i128, i128)>| {
                        r.is_some_and(|(lo, hi)| lo >= 0 && hi < i128::from(size))
                    };
                    let chain = chain_of(&links, &mut memo, m, 0);
                    let all: Vec<Cond> = chain.iter().map(|&(c, p, _)| (c, p)).collect();
                    let r = int(at(all).get(idx).copied());
                    if let Some(r) = r {
                        out.index.insert(m, r);
                    }
                    if !inside(r) {
                        continue;
                    }
                    out.proven.insert(m, ());
                    if chain.iter().any(|l| l.2.is_some()) {
                        // Needed the arms? Re-check under the loop conditions only.
                        let loops: Vec<Cond> = chain
                            .iter()
                            .filter(|l| l.2.is_none())
                            .map(|&(c, p, _)| (c, p))
                            .collect();
                        if !inside(int(at(loops).get(idx).copied())) {
                            for &(_, _, e) in &chain {
                                if let Some(e) = e {
                                    out.arms.insert(e, ());
                                }
                            }
                        }
                    }
                }
                Operation::Map { captures, .. } | Operation::Fold { captures, .. }
                    if captures > 0 =>
                {
                    let chain = chain_of(&links, &mut memo, m, 0);
                    let all: Vec<Cond> = chain.iter().map(|&(c, p, _)| (c, p)).collect();
                    let r = at(all);
                    let caps = (0..captures)
                        .map(|j| {
                            self.pair_slot_source(morph.source, j)
                                .and_then(|c| int(r.get(c).copied()))
                        })
                        .collect();
                    out.sites.insert(m, caps);
                }
                _ => {}
            }
        }
        out
    }

    /// Seed a body fn's params: the element param rides its site source's
    /// element range, each capture param the site owner's range of the
    /// captured value where the site runs. Every site of the body must agree
    /// (the hull; one unknown site leaves the param unranged).
    fn seed_body(&self, f: FuncId, input: ObjectId, seeds: &mut RngMap) {
        let mut elem: Option<Option<Rng>> = None;
        let mut caps: Option<Vec<Option<(i128, i128)>>> = None;
        let mut shape: Option<(u32, bool)> = None;
        for (m, morph) in self.morphisms() {
            let (k, is_fold) = match morph.op {
                Operation::Map { body, captures } if body == f => (captures, false),
                Operation::Fold { body, captures } if body == f => (captures, true),
                _ => continue,
            };
            if shape.is_some_and(|s| s != (k, is_fold)) {
                return;
            }
            shape = Some((k, is_fold));
            let arr = if is_fold {
                self.pair_slot_source(morph.source, k + 1)
            } else if k == 0 {
                Some(morph.source)
            } else {
                self.pair_slot_source(morph.source, k)
            };
            let er = arr.and_then(|a| self.element_range(a));
            elem = Some(match elem {
                None => er,
                Some(prev) => match (prev, er) {
                    (Some(Rng::Int(a, b)), Some(Rng::Int(c, d))) => {
                        let (l, h) = hull((a, b), (c, d));
                        Some(Rng::Int(l, h))
                    }
                    (Some(x), Some(y)) if x == y => Some(x),
                    _ => None,
                },
            });
            let site: Vec<Option<(i128, i128)>> = match self.try_owner(morph.source) {
                Some(of) if of != f => self
                    .bounds_proof(of)
                    .sites
                    .get(m)
                    .cloned()
                    .unwrap_or_else(|| vec![None; k as usize]),
                _ => vec![None; k as usize],
            };
            caps = Some(match caps {
                None => site,
                Some(prev) => prev
                    .iter()
                    .zip(&site)
                    .map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some(hull(*a, *b)),
                        _ => None,
                    })
                    .collect(),
            });
        }
        let Some((k, is_fold)) = shape else { return };
        if let Some(Some(er)) = elem {
            if !is_fold && k == 0 {
                seeds.insert(input, er);
            } else {
                let p = if is_fold { k + 1 } else { k };
                for t in self.proj_targets(input, p) {
                    seeds.insert(t, er);
                }
            }
        }
        for (j, r) in caps.unwrap_or_default().into_iter().enumerate() {
            if let Some((l, h)) = r {
                for t in self.proj_targets(input, j as u32) {
                    seeds.insert(t, Rng::Int(l, h));
                }
            }
        }
    }

    fn proj_targets(&self, o: ObjectId, k: u32) -> Vec<ObjectId> {
        self.out_edges(o)
            .iter()
            .filter(|&&m| self.morphisms[m].op == (Operation::Proj { index: k }))
            .map(|&m| self.morphisms[m].target)
            .collect()
    }

    /// Each of `f`'s morphisms' direct context: the morphisms whose context it
    /// inherits (its guard site's `Phi`, its loop's `LoopEnter`) and its own
    /// conditions (the arm's, the continue condition on an advance member).
    fn bounds_links(&self, f: FuncId) -> SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)> {
        let mut links: SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)> =
            SecondaryMap::new();
        let mut add = |m: MorphismId, up: Option<MorphismId>, c: Option<CtxLink>| {
            if links.get(m).is_none() {
                links.insert(m, (Vec::new(), Vec::new()));
            }
            let e = links.get_mut(m).expect("inserted");
            if let Some(u) = up
                && u != m
                && !e.0.contains(&u)
            {
                e.0.push(u);
            }
            if let Some(c) = c
                && !e.1.contains(&c)
            {
                e.1.push(c);
            }
        };
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                let Some(plan) = self.loop_plan(f, merge) else {
                    continue;
                };
                let enter = self
                    .in_edges(merge)
                    .iter()
                    .copied()
                    .find(|&e| self.morphisms[e].op == Operation::LoopEnter);
                let cond = self.pair_slot_source(plan.back_route, 1);
                for &m in plan.decide_order.iter().chain(&plan.exits) {
                    add(m, enter, None);
                }
                for &m in &plan.advance_order {
                    add(m, enter, cond.map(|c| (c, true, None)));
                }
            }
        }
        for site in self.guard_structure(f) {
            for (arm, pole) in [(&site.on_true, true), (&site.on_false, false)] {
                for &m in &arm.own {
                    add(m, Some(site.phi), Some((site.cond, pole, Some(arm.edge))));
                }
            }
        }
        links
    }

    /// The range of an array's elements: iota ⇒ `[0, n)`; enumerate ⇒ `(i32,
    /// X)` elements over `n`; a literal array of int constants ⇒ `[min, max]`.
    /// `Proj`-of-pair indirection and body-input capture indirection (the
    /// array rides in as a captured value) are resolved first.
    pub(crate) fn element_range(&self, arr: ObjectId) -> Option<Rng> {
        let mut arr = arr;
        for _ in 0..16 {
            if let Some(next) = self.proj_of_pair(arr) {
                arr = next;
                continue;
            }
            if let Some(next) = self.capture_proj_of_input(arr) {
                arr = next;
                continue;
            }
            break;
        }
        let definer = self.in_edges(arr).first().copied()?;
        let dm = &self.morphisms[definer];
        let size = match &self.objects[arr].ty {
            Ty::Array { size, .. } => *size,
            _ => return None,
        };
        match dm.op {
            Operation::Iota => Some(Rng::Int(0, i128::from(size) - 1)),
            Operation::Enumerate => Some(Rng::EnumIdx(size)),
            Operation::Pair { .. } => {
                // A pack_array literal: every in-edge a Pair with a Constant
                // int source — the element range is [min, max] of the values.
                let mut lo = i128::MAX;
                let mut hi = i128::MIN;
                for &m in self.in_edges(arr) {
                    let pm = &self.morphisms[m];
                    if !matches!(pm.op, Operation::Pair { .. }) {
                        return None;
                    }
                    let v = const_int(self, pm.source)?;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                Some(Rng::Int(lo, hi))
            }
            _ => None,
        }
    }
}

/// A morphism's full context: its inherited contexts then its own links,
/// deduplicated (memoized; depth-guarded against a malformed cycle).
fn chain_of(
    links: &SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)>,
    memo: &mut SecondaryMap<MorphismId, Vec<CtxLink>>,
    m: MorphismId,
    depth: u32,
) -> Vec<CtxLink> {
    if let Some(c) = memo.get(m) {
        return c.clone();
    }
    let Some((up, own)) = links.get(m) else {
        return Vec::new();
    };
    let mut out: Vec<CtxLink> = Vec::new();
    if depth < 32 {
        for &u in up {
            for l in chain_of(links, memo, u, depth + 1) {
                if !out.contains(&l) {
                    out.push(l);
                }
            }
        }
    }
    for &l in own {
        if !out.contains(&l) {
            out.push(l);
        }
    }
    memo.insert(m, out.clone());
    out
}
