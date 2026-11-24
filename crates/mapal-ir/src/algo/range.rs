//! **`finite_plan` (flash-attention F7): values proven finite.**
//!
//! Interval transfer over the dataflow: each float value's range `[lo, hi]`
//! where every element of it is finite — constants, `iota` elements and
//! their widenings, `+ − × ÷` (a divisor bounded away from zero), `exp` of
//! an argument bounded above, a guard's arms (their union), a map's body
//! over its captured ranges, a fold iterated over its trip count (at most
//! [`FOLD_TRIPS`]; a range that stops moving stops the walk), a narrowing
//! (its rounding inside the range's reach). Anything else — a function
//! input, a loop's carried state, a range past the format's largest finite
//! value — is not proven. Each step's bounds are padded outward
//! ([`PAD`]): rounding never takes a value past them.
//!
//! The consumer: a masked window whose probabilities are exactly `+0`
//! (`stream_plan`'s absorb) adds `0 · y` to a product — `+0` exactly when
//! `y` is finite. A second consumer is ADR-0043's `f16` member admission
//! (a value range fact).

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, Value};

/// A fold's trips walked at most: past them, a range that still moves is
/// not proven.
pub const FOLD_TRIPS: u64 = 1 << 16;

/// The outward padding of every step's bounds (relative and absolute).
const PAD: f64 = 1.0 / (1u64 << 20) as f64;

/// An abstract value: a number's range (`None`: not proven), an array of
/// such elements, a product.
#[derive(Clone, Debug, PartialEq)]
enum Abs {
    Num(Option<(f64, f64)>),
    Arr(Box<Abs>),
    Prod(Vec<Abs>),
}

const TOP: Abs = Abs::Num(None);

impl Abs {
    fn num(&self) -> Option<(f64, f64)> {
        match self {
            Abs::Num(x) => *x,
            _ => None,
        }
    }

    fn join(&self, o: &Abs) -> Abs {
        match (self, o) {
            (Abs::Num(Some(a)), Abs::Num(Some(b))) => Abs::Num(Some((a.0.min(b.0), a.1.max(b.1)))),
            (Abs::Arr(a), Abs::Arr(b)) => Abs::Arr(Box::new(a.join(b))),
            (Abs::Prod(a), Abs::Prod(b)) if a.len() == b.len() => {
                Abs::Prod(a.iter().zip(b).map(|(x, y)| x.join(y)).collect())
            }
            _ => TOP,
        }
    }
}

/// Every proven-finite value of one function: a number's range, an array's
/// elements' range.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FinitePlan {
    pub ranges: SecondaryMap<ObjectId, (f64, f64)>,
}

impl FinitePlan {
    /// Is every element of `o` proven finite?
    pub fn finite(&self, o: ObjectId) -> bool {
        self.ranges.contains_key(o)
    }
}

/// The largest finite value of a float type (an integer has no limit here).
fn limit(ty: &Ty) -> f64 {
    match ty {
        Ty::Float { bits: 32 } => f32::MAX as f64,
        Ty::Float { .. } => f64::MAX / 2.0,
        Ty::Narrow(_) => 65504.0_f64.max(f32::MAX as f64),
        _ => f64::MAX / 2.0,
    }
}

fn elem_ty(ty: &Ty) -> &Ty {
    match ty {
        Ty::Array { elem, .. } => elem_ty(elem),
        t => t,
    }
}

/// `[lo, hi]` padded outward, kept only when finite under `lim`.
fn padded(lo: f64, hi: f64, lim: f64) -> Option<(f64, f64)> {
    let lo = lo - lo.abs() * PAD - f64::MIN_POSITIVE;
    let hi = hi + hi.abs() * PAD + f64::MIN_POSITIVE;
    (lo.is_finite() && hi.is_finite() && lo >= -lim && hi <= lim && lo <= hi).then_some((lo, hi))
}

impl CategoryIr {
    /// **Values proven finite** (module doc) of `f`. Stored on the graph;
    /// empty under `FactArms::finite` off.
    pub fn finite_plan(&self, f: FuncId) -> FinitePlan {
        self.fact(f, |s| &s.finite_plan, || self.finite_plan_compute(f))
    }

    fn finite_plan_compute(&self, f: FuncId) -> FinitePlan {
        let mut plan = FinitePlan::default();
        if !self.fact_arms().finite {
            return plan;
        }
        let Some(fd) = self.func(f) else {
            return plan;
        };
        let mut memo: SecondaryMap<ObjectId, Abs> = SecondaryMap::new();
        for o in self.func_objects(f) {
            let v = self.abs(o, fd.input, &TOP, &mut memo, 0);
            let r = match &v {
                Abs::Num(x) => *x,
                Abs::Arr(e) => e.num(),
                Abs::Prod(_) => None,
            };
            if let Some(r) = r
                && matches!(
                    elem_ty(&self.objects[o].ty),
                    Ty::Float { .. } | Ty::Narrow(_)
                )
            {
                plan.ranges.insert(o, r);
            }
        }
        plan
    }

    /// Object `o`'s abstract value inside the function whose input is
    /// `input` (bound to `arg`).
    fn abs(
        &self,
        o: ObjectId,
        input: ObjectId,
        arg: &Abs,
        memo: &mut SecondaryMap<ObjectId, Abs>,
        depth: u32,
    ) -> Abs {
        if let Some(v) = memo.get(o) {
            return v.clone();
        }
        // A cycle (a loop's carried state) or a deep chain: not proven.
        memo.insert(o, TOP);
        let v = self.abs_compute(o, input, arg, memo, depth);
        memo.insert(o, v.clone());
        v
    }

    fn abs_compute(
        &self,
        o: ObjectId,
        input: ObjectId,
        arg: &Abs,
        memo: &mut SecondaryMap<ObjectId, Abs>,
        depth: u32,
    ) -> Abs {
        if depth > 64 {
            return TOP;
        }
        let Some(obj) = self.objects.get(o) else {
            return TOP;
        };
        if o == input {
            return arg.clone();
        }
        if obj.kind == ObjectKind::Constant
            && let Some(v) = &obj.value
        {
            let x = match v {
                Value::F32(x) => *x as f64,
                Value::F64(x) => *x,
                Value::I32(x) => *x as f64,
                Value::I64(x) => *x as f64,
                Value::U8(x) => *x as f64,
                _ => return TOP,
            };
            return match x.is_finite() {
                true => Abs::Num(Some((x, x))),
                false => TOP,
            };
        }
        let lim = limit(elem_ty(&obj.ty));
        let ins = self.in_edges(o);
        // A product: its slots.
        if ins
            .iter()
            .all(|&m| matches!(self.morphisms[m].op, Operation::Pair { .. }))
            && !ins.is_empty()
        {
            let arity = match self.morphisms[ins[0]].op {
                Operation::Pair { arity, .. } => arity,
                _ => 0,
            };
            let mut slots = vec![TOP; arity as usize];
            for &m in ins {
                if let Operation::Pair { slot, .. } = self.morphisms[m].op
                    && let Some(x) = slots.get_mut(slot as usize)
                {
                    *x = self.abs(self.morphisms[m].source, input, arg, memo, depth + 1);
                }
            }
            return Abs::Prod(slots);
        }
        let [m] = ins else {
            return TOP;
        };
        let mo = &self.morphisms[*m];
        let src = |me: &Self, memo: &mut SecondaryMap<ObjectId, Abs>| {
            me.abs(mo.source, input, arg, memo, depth + 1)
        };
        let two = |me: &Self, memo: &mut SecondaryMap<ObjectId, Abs>| match src(me, memo) {
            Abs::Prod(p) if p.len() == 2 => (p[0].num(), p[1].num()),
            _ => (None, None),
        };
        let float = matches!(elem_ty(&obj.ty), Ty::Float { .. } | Ty::Narrow(_));
        match mo.op {
            Operation::Proj { index } => match src(self, memo) {
                Abs::Prod(p) => p.get(index as usize).cloned().unwrap_or(TOP),
                _ => TOP,
            },
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                let (Some(a), Some(b)) = two(self, memo) else {
                    return TOP;
                };
                let r = match mo.op {
                    Operation::Add => Some((a.0 + b.0, a.1 + b.1)),
                    Operation::Sub => Some((a.0 - b.1, a.1 - b.0)),
                    // A square is never negative.
                    Operation::Mul
                        if self.pair_slot_source(mo.source, 0).is_some()
                            && self.pair_slot_source(mo.source, 0)
                                == self.pair_slot_source(mo.source, 1) =>
                    {
                        let m = (a.0 * a.0).max(a.1 * a.1);
                        match a.0 <= 0.0 && a.1 >= 0.0 {
                            true => Some((0.0, m)),
                            false => Some(((a.0 * a.0).min(a.1 * a.1), m)),
                        }
                    }
                    Operation::Mul => {
                        let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
                        Some((
                            p.iter().copied().fold(f64::INFINITY, f64::min),
                            p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        ))
                    }
                    Operation::Div if b.0 > 0.0 || b.1 < 0.0 => {
                        let p = [a.0 / b.0, a.0 / b.1, a.1 / b.0, a.1 / b.1];
                        let (lo, hi) = (
                            p.iter().copied().fold(f64::INFINITY, f64::min),
                            p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        );
                        // An integer quotient truncates toward zero.
                        match float {
                            true => Some((lo, hi)),
                            false => {
                                Some((lo.min(0.0).trunc().min(lo.trunc()), hi.trunc().max(0.0)))
                            }
                        }
                    }
                    Operation::Mod if !float && a.0 >= 0.0 && b.0 > 0.0 => {
                        Some((0.0, a.1.min(b.1 - 1.0)))
                    }
                    _ => None,
                };
                Abs::Num(r.and_then(|(lo, hi)| padded(lo, hi, lim)))
            }
            Operation::Neg => match src(self, memo).num() {
                Some((lo, hi)) => Abs::Num(Some((-hi, -lo))),
                None => TOP,
            },
            Operation::Widen | Operation::Narrow => match src(self, memo).num() {
                Some((lo, hi)) => Abs::Num(padded(lo, hi, lim)),
                None => TOP,
            },
            Operation::Exp => match src(self, memo).num() {
                Some((lo, hi)) if hi < lim.ln() - 1.0 => Abs::Num(padded(lo.exp(), hi.exp(), lim)),
                _ => TOP,
            },
            // ADR-0046: `sqrt` of a non-negative range; `sin` / `cos` in
            // `[−1, 1]`.
            Operation::Math(crate::math::MathFn::Sqrt) => match src(self, memo).num() {
                Some((lo, hi)) if lo >= 0.0 => Abs::Num(padded(lo.sqrt(), hi.sqrt(), lim)),
                _ => TOP,
            },
            Operation::Math(crate::math::MathFn::Sin | crate::math::MathFn::Cos) => {
                Abs::Num(padded(-1.0, 1.0, lim))
            }
            Operation::Phi => match src(self, memo) {
                Abs::Prod(p) if p.len() == 3 => p[0].join(&p[1]),
                _ => TOP,
            },
            Operation::Index => match src(self, memo) {
                Abs::Prod(p) if p.len() == 2 => match &p[0] {
                    Abs::Arr(e) => (**e).clone(),
                    _ => TOP,
                },
                _ => TOP,
            },
            Operation::Iota => match &obj.ty {
                Ty::Array { size, .. } => Abs::Arr(Box::new(Abs::Num(Some((
                    0.0,
                    size.saturating_sub(1) as f64,
                ))))),
                _ => TOP,
            },
            Operation::Fill => Abs::Arr(Box::new(src(self, memo))),
            Operation::Update => match src(self, memo) {
                Abs::Prod(p) if p.len() == 3 => match &p[0] {
                    Abs::Arr(e) => Abs::Arr(Box::new(e.join(&p[2]))),
                    _ => TOP,
                },
                _ => TOP,
            },
            Operation::Call(g) => {
                let a = src(self, memo);
                self.abs_fn(g, &a, depth + 1)
            }
            Operation::Map { body, captures } => {
                let s = src(self, memo);
                let input = match (captures, s) {
                    (0, Abs::Arr(e)) => (*e).clone(),
                    (c, Abs::Prod(p)) if p.len() == c as usize + 1 => {
                        let mut slots = p[..c as usize].to_vec();
                        slots.push(match &p[c as usize] {
                            Abs::Arr(e) => (**e).clone(),
                            _ => TOP,
                        });
                        Abs::Prod(slots)
                    }
                    _ => return TOP,
                };
                Abs::Arr(Box::new(self.abs_fn(body, &input, depth + 1)))
            }
            Operation::Fold { body, captures } => {
                let Abs::Prod(p) = src(self, memo) else {
                    return TOP;
                };
                let c = captures as usize;
                if p.len() != c + 2 {
                    return TOP;
                }
                let n = match self
                    .objects
                    .get(self.pair_slot_source(mo.source, captures + 1).unwrap_or(o))
                {
                    Some(x) => match &x.ty {
                        Ty::Array { size, .. } => *size,
                        _ => return TOP,
                    },
                    None => return TOP,
                };
                let elem = match &p[c + 1] {
                    Abs::Arr(e) => (**e).clone(),
                    _ => return TOP,
                };
                let mut acc = p[c].clone();
                for trip in 0..n {
                    if trip >= FOLD_TRIPS {
                        return TOP;
                    }
                    let mut slots = p[..c].to_vec();
                    slots.push(acc.clone());
                    slots.push(elem.clone());
                    let next = self.abs_fn(body, &Abs::Prod(slots), depth + 1);
                    if next == acc {
                        break;
                    }
                    if next.num().is_none() && matches!(next, Abs::Num(_)) {
                        return TOP;
                    }
                    acc = next;
                }
                acc
            }
            _ => TOP,
        }
    }

    /// Function `g`'s output on input `arg`.
    fn abs_fn(&self, g: FuncId, arg: &Abs, depth: u32) -> Abs {
        let Some(gd) = self.func(g) else {
            return TOP;
        };
        let mut memo: SecondaryMap<ObjectId, Abs> = SecondaryMap::new();
        self.abs(gd.output, gd.input, arg, &mut memo, depth)
    }
}
