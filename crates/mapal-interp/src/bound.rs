//! The composed bound of the default face (ADR-0041 D5 `bound(o)`, the
//! output's bound; ADR-0049 D1): how far the default face may print from the
//! exact face, per printed value, composed through the program.
//!
//! Each operation's own deviation is the one its ADR admits, read from the IR
//! facts:
//!
//! - **member** — a product admitting a reduced member (`precision`, ADR-0041
//!   D3 / ADR-0043 D4) may round each multiplicand to the member's grid. The
//!   rounding is round-to-nearest of the operand's value, so its error
//!   `δ = member(x) − x` is known, with its sign; a realization rounds an
//!   operand slot of a product at every execution or at none (a stored
//!   array, a tensor tile's fragments, the weakest oracle). Each
//!   `(product, slot)` changes the output by `Σ_exec ∂out/∂y · δ·other`. The
//!   two roundings' cross term `|δ_a·δ_b|` counts per execution.
//! - **step** — a rounding (a member's, or a narrow one: ADR-0043 D3) of an
//!   operand that carries a deviation gives the grid value of the exact
//!   operand unless the deviation reaches a tie: short of the near tie it
//!   does (a narrow rounding absorbs the deviation), short of the far tie it
//!   may step once to the near tie's side (a known sign), past it a step of
//!   either side with the deviation passing (ADR-0043 D3's `e + ulp`). One
//!   value steps once at all its reads: per `(value, product, slot)`. The
//!   operand's deviation bound is its own sweep, every rounding before it
//!   decided first.
//! - **order** — a fold whose step may regroup (`step_algebra.regroup`) sums
//!   in any order: `2·γ_{K+1}(u_acc)·M` (ADR-0040 D2 on both faces; `M` the
//!   chain's magnitudes, `u_acc = 2⁻²³` where its products admit a member — a
//!   tensor-core accumulator, D3).
//! - **fma** — an fma pair may fuse: `2·u·(|a·b| + |y|)`.
//! - **math** — a builtin meets its contract on each face (ADR-0042 /
//!   ADR-0046): `2·(rel·|y| + abs)`.
//!
//! Every other operation is one IEEE rounding the realization performs as the
//! oracle does: equal operands, equal bits. The run records every float
//! operation of the exact face on a tape — operands, partial derivatives, its
//! own deviation — and each printed value's bound is the first-order
//! composition through the program's derivatives (one reverse sweep per
//! printed value): the signed changes a realization takes or not (members,
//! known-side steps) at the larger of their positive and negative sums, every
//! other deviation by magnitude. The derivatives keep their signs until a
//! deviation's magnitude is taken, so a contraction (a Newton step, a
//! cancellation) is the program's own rather than a worst case per operation
//! (per-operation running error analysis counts 16 Newton steps 16 times).
//! First order: a float comparison whose operands lie within their bound of
//! each other may flip its arm; the bound assumes it does not.

use std::cell::RefCell;

use mapal_ir::math::MathFn;
use mapal_ir::{CategoryIr, FuncId, MorphismId, NarrowFloat, Operation, RowFoldOp, Value};
use slotmap::SecondaryMap;

use crate::value::RValue;

/// No operand.
const NONE: u32 = u32::MAX;

/// A printed value's composed bound and its parts by source.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bound {
    /// A print that is not a float: its own text (a string, exact on every
    /// face), or `0e0` (an integer or a boolean: no deviation); `None` for
    /// a float, whose bound is `total`.
    pub text: Option<String>,
    /// The print ends its line.
    pub newline: bool,
    /// `|default − exact| ≤ total` to first order.
    pub total: f64,
    /// Multiplicands rounded to an admitted member (magnitudes).
    pub member: f64,
    /// Regrouped folds.
    pub order: f64,
    /// Fused product–add pairs.
    pub fma: f64,
    /// Grid steps of roundings of an operand carrying a deviation
    /// (magnitudes).
    pub flip: f64,
    /// Math builtins' contracts.
    pub math: f64,
}

/// Where a node's own deviation comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Src {
    None,
    Member,
    Order,
    Fma,
    Math,
}

/// One float operation of the exact face.
#[derive(Clone, Copy)]
struct Node {
    a: u32,
    da: f64,
    b: u32,
    db: f64,
    /// Its own deviation's magnitude.
    nu: f64,
    src: Src,
    /// `M`: the magnitudes a chain of additions adds (ADR-0040 D2),
    /// `|value|` past any other operation.
    m: f64,
    /// Some deviation reaches it (its own or an operand's).
    noisy: bool,
}

/// A multiplicand rounded at one execution: the product's node, its
/// `(product, slot)` and the product's first-order change `δ·other`.
#[derive(Clone, Copy)]
struct Event {
    node: u32,
    key: u32,
    d: f64,
}

/// A rounding of a tracked operand that carries a deviation: where the
/// operand's bound reaches `dist` (its distance to the nearest tie), the
/// rounding may land `ulp` away from the exact face's.
#[derive(Clone, Copy)]
struct Read {
    /// The operand's node.
    x: u32,
    /// The rounding's node (a `Narrow`, or a product rounding a multiplicand).
    at: u32,
    tie: Tie,
    /// A product's multiplicand: its `(product, slot)` key and the other
    /// multiplicand; `None` for a `Narrow`.
    other: Option<(u32, f64)>,
}

/// What a function's operations need from its facts, read once.
#[derive(Clone, Default)]
struct FnInfo {
    /// The regroupable float step of a fold body (`step_algebra.regroup`).
    regroup: Option<(MorphismId, RowFoldOp)>,
    /// Each fma pair's add / sub: the slot of its product.
    fma: SecondaryMap<MorphismId, u32>,
    /// A product of the body admits a reduced member (a tensor core's
    /// accumulator may sum it, `u_a = 2⁻²³`).
    member: bool,
}

#[derive(Default)]
struct Tape {
    nodes: Vec<Node>,
    events: Vec<Event>,
    reads: Vec<Read>,
    /// A possible grid step at a deviating operand's rounding, keyed per
    /// `(operand, product, slot)` (`resolve`; any node order); `far`: its
    /// side unknown (the deviation may reach the far tie).
    steps: Vec<Event>,
    far: Vec<bool>,
    /// `(product, slot)` → its event key.
    keys: SecondaryMap<MorphismId, [u32; 2]>,
    nkeys: u32,
    /// Each print: its float's node (`None`: exact on every face), its text
    /// where it is not a float, and whether it ends a line.
    printed: Vec<(Option<u32>, Option<String>, bool)>,
    info: SecondaryMap<FuncId, FnInfo>,
}

thread_local! {
    static TAPE: RefCell<Option<Tape>> = const { RefCell::new(None) };
}

/// Records while `f` runs; returns its result and each printed scalar's
/// composed bound, in print order.
pub(crate) fn record<R>(f: impl FnOnce() -> R) -> (R, Vec<Bound>) {
    TAPE.with(|t| *t.borrow_mut() = Some(Tape::default()));
    let r = f();
    let mut tape = TAPE
        .with(|t| t.borrow_mut().take())
        .expect("the tape this run started");
    resolve(&mut tape);
    let mut buf = Buf::default();
    let bounds = tape
        .printed
        .iter()
        .map(|(p, text, newline)| Bound {
            text: text.clone(),
            newline: *newline,
            ..p.map_or_else(Bound::default, |r| sweep(&tape, r, &mut buf))
        })
        .collect();
    (r, bounds)
}

/// Decides each recorded rounding of a deviating operand `x`, in `x`'s tape
/// order (its sweep sees every earlier decision), against `e`, `x`'s bound:
/// short of the near tie the rounding gives the exact face's grid value — a
/// `Narrow` absorbs the deviation (`∂ = 0`); short of the far tie it may
/// step once to the near tie's side (a signed step); past it, a step of
/// either side with the deviation passing (ADR-0043 D3's `e + ulp`). Steps
/// at one value are one step: keyed per `(x, product, slot)`.
fn resolve(tape: &mut Tape) {
    let mut reads = std::mem::take(&mut tape.reads);
    reads.sort_by_key(|r| r.x);
    let mut buf = Buf::default();
    let mut i = 0;
    let mut slots: Vec<(u32, u32)> = Vec::new();
    while i < reads.len() {
        let x = reads[i].x;
        let e = sweep(tape, x, &mut buf).total;
        slots.clear();
        while i < reads.len() && reads[i].x == x {
            let r = reads[i];
            i += 1;
            let t = r.tie;
            // Past the far tie the step's side is unknown and the operand's
            // deviation passes (ADR-0043 D3's `e + ulp`); short of it, one
            // known step or none.
            let far = e >= t.far;
            if r.other.is_none() && !far {
                // A narrow rounding gives the exact face's bits or one step.
                tape.nodes[r.at as usize].da = 0.0;
            }
            if e < t.near {
                continue;
            }
            let (k, o) = r.other.unwrap_or((u32::MAX - r.at, 1.0));
            let key = match slots.iter().find(|s| s.0 == k) {
                Some(s) => s.1,
                None => {
                    slots.push((k, tape.far.len() as u32));
                    tape.far.push(far);
                    tape.far.len() as u32 - 1
                }
            };
            tape.steps.push(Event {
                node: r.at,
                key,
                d: if far { t.ulp } else { t.step } * o,
            });
        }
    }
}

/// A sweep's reusable buffers.
#[derive(Default)]
struct Buf {
    adj: Vec<f64>,
    per: Vec<f64>,
}

/// Node `r`'s bound: one reverse sweep for `∂r/∂x_i`, then each node's own
/// deviation and each `(product, slot)`'s signed sum.
fn sweep(tape: &Tape, r: u32, buf: &mut Buf) -> Bound {
    let adj = &mut buf.adj;
    adj.clear();
    adj.resize(r as usize + 1, 0.0);
    adj[r as usize] = 1.0;
    let mut b = Bound::default();
    for i in (0..=r as usize).rev() {
        let g = adj[i];
        if g == 0.0 {
            continue;
        }
        let n = tape.nodes[i];
        let own = g.abs() * n.nu;
        match n.src {
            Src::None => {}
            Src::Member => b.member += own,
            Src::Order => b.order += own,
            Src::Fma => b.fma += own,
            Src::Math => b.math += own,
        }
        if n.a != NONE {
            adj[n.a as usize] += g * n.da;
        }
        if n.b != NONE {
            adj[n.b as usize] += g * n.db;
        }
    }
    // The events are recorded in node order: those at or before `r`.
    // Each key's signed sum: a realization takes it or not, so the signed
    // parts' worst subset is the larger of their positive and negative sums.
    let unsigned = b.member + b.order + b.fma + b.flip + b.math;
    let (mut pos, mut neg) = (0.0, 0.0);
    let upto = tape.events.partition_point(|e| e.node <= r);
    let per = &mut buf.per;
    per.clear();
    per.resize(tape.nkeys as usize, 0.0);
    for e in &tape.events[..upto] {
        per[e.key as usize] += adj[e.node as usize] * e.d;
    }
    for &x in per.iter() {
        b.member += x.abs();
        if x > 0.0 { pos += x } else { neg -= x }
    }
    per.clear();
    per.resize(tape.far.len(), 0.0);
    for e in tape.steps.iter().filter(|e| e.node <= r) {
        per[e.key as usize] += adj[e.node as usize] * e.d;
    }
    let mut far = 0.0;
    for (&x, &f) in per.iter().zip(&tape.far) {
        b.flip += x.abs();
        if f {
            far += x.abs();
        } else if x > 0.0 {
            pos += x
        } else {
            neg -= x
        }
    }
    b.total = unsigned + far + f64::max(pos, neg);
    b
}

pub(crate) fn on() -> bool {
    TAPE.with(|t| t.borrow().is_some())
}

/// A float scalar's value as `f64`.
fn num(v: &Value) -> Option<f64> {
    match v {
        Value::F32(x) => Some(f64::from(*x)),
        Value::F64(x) => Some(*x),
        Value::Narrow(n, b) => Some(f64::from(mapal_ir::narrow::widen(*n, *b))),
        _ => None,
    }
}

/// `x`'s node, its `M` and whether a deviation reaches it.
fn leaf(tape: &Tape, x: &RValue) -> (u32, f64, bool) {
    match x {
        RValue::Tracked(_, i) => {
            let n = tape.nodes[*i as usize];
            (*i, n.m, n.noisy)
        }
        RValue::Scalar(v) => (NONE, num(v).map_or(0.0, f64::abs), false),
        _ => (NONE, 0.0, false),
    }
}

const UNIT: RValue = RValue::Unit;

/// A node's own deviation.
struct Own {
    nu: f64,
    src: Src,
    /// Its multiplicands' roundings deviate it (events follow).
    rounds: bool,
}

const EXACT: Own = Own {
    nu: 0.0,
    src: Src::None,
    rounds: false,
};

/// Appends a node (operands `(x, ∂)`), or the plain value where the
/// operation is exact on every face and its operands are.
fn push(tape: &mut Tape, y: Value, ops: [(&RValue, f64); 2], own: Own, m: f64) -> RValue {
    let (a, _, na) = leaf(tape, ops[0].0);
    let (b, _, nb) = leaf(tape, ops[1].0);
    if a == NONE && b == NONE && own.nu == 0.0 && !own.rounds {
        return RValue::Scalar(y);
    }
    let i = tape.nodes.len() as u32;
    tape.nodes.push(Node {
        a,
        da: ops[0].1,
        b,
        db: ops[1].1,
        nu: own.nu,
        src: own.src,
        m,
        noisy: own.nu > 0.0 || own.rounds || na || nb,
    });
    RValue::Tracked(y, i)
}

/// `f`'s [`FnInfo`], read from its facts the first time.
fn info<'t>(tape: &'t mut Tape, ir: &CategoryIr, f: FuncId) -> &'t FnInfo {
    if !tape.info.contains_key(f) {
        let alg = ir.step_algebra(f);
        let mut i = FnInfo {
            regroup: alg.regroup.filter(|r| !r.exact).map(|r| (r.step, r.op)),
            member: ir.func(f).is_some_and(|fd| {
                fd.morphisms.iter().any(|&m| {
                    ir.morphism(m).is_some_and(|x| x.op == Operation::Mul)
                        && ir.precision(m).reduced()
                })
            }),
            ..FnInfo::default()
        };
        for (add, pair) in &alg.fma {
            i.fma.insert(add, pair.slot);
        }
        tape.info.insert(f, i);
    }
    &tape.info[f]
}

/// The unit roundoff of a float width.
fn unit_of(v: &Value) -> f64 {
    match v {
        Value::F64(_) => 2f64.powi(-53),
        _ => 2f64.powi(-24),
    }
}

/// `member(x) − x` for the member a product admits (0 on its grid: a
/// widened narrow value, an exact constant; or where it admits none).
fn member_delta(p: mapal_ir::Precision, x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let r = match (p.half(), p.admits_tf32()) {
        (Some(n), _) => f64::from(mapal_ir::narrow::widen(n, mapal_ir::narrow::round(n, x))),
        (None, true) => f64::from(crate::eval::round_tf32(x as f32)),
        (None, false) => x,
    };
    r - x
}

/// `x`'s distance to the nearest tie of a grid of `p` significand bits
/// (least normal exponent `emin`) and the grid's step there.
/// `x` against the grid it rounds on (to `r`): the near tie's distance and
/// the rounded value's signed step across it, the far tie's distance, and
/// the larger step.
#[derive(Clone, Copy)]
struct Tie {
    near: f64,
    step: f64,
    far: f64,
    ulp: f64,
}

/// [`Tie`] on a grid of `p` significand bits, least normal exponent `emin`.
fn tie(x: f64, p: i32, emin: i32, r: f64) -> Tie {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let (x, r) = (x.abs(), r.abs());
    let e = if r > 0.0 {
        (r.log2().floor() as i32).max(emin)
    } else {
        emin
    };
    let up = 2f64.powi(e - p + 1);
    let down = if r == 2f64.powi(e) && e > emin {
        up / 2.0
    } else {
        up
    };
    let (hi, lo) = (
        ((r + up / 2.0) - x).max(0.0),
        (x - (r - down / 2.0)).max(0.0),
    );
    if hi <= lo {
        Tie {
            near: hi,
            step: sign * up,
            far: lo,
            ulp: up,
        }
    } else {
        Tie {
            near: lo,
            step: -sign * down,
            far: hi,
            ulp: up,
        }
    }
}

/// `x` on the member grid a product rounds to.
fn member_tie(p: mapal_ir::Precision, x: f64) -> Option<Tie> {
    if !x.is_finite() {
        return None;
    }
    let r = x + member_delta(p, x);
    match (p.half(), p.admits_tf32()) {
        (Some(n), _) => {
            let f = mapal_ir::narrow::NarrowForm::of(n);
            Some(tie(x, f.precision as i32, f.emin, r))
        }
        (None, true) => Some(tie(x, 11, -126, r)),
        (None, false) => None,
    }
}

/// `a op b = y` of morphism `m` in `f` (`Add` / `Sub` / `Mul` / `Div` /
/// `Mod` on floats).
pub(crate) fn arith(
    ir: &CategoryIr,
    f: FuncId,
    m: MorphismId,
    op: Operation,
    a: &RValue,
    b: &RValue,
    y: Value,
) -> RValue {
    let (Some(x0), Some(x1), Some(yv)) = (
        num(crate::eval::scalar(a)),
        num(crate::eval::scalar(b)),
        num(&y),
    ) else {
        return RValue::Scalar(y);
    };
    let u = unit_of(&y);
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let tape = t.as_mut().expect("recording");
        let fi = info(tape, ir, f);
        let step = fi.regroup.is_some_and(|(st, _)| st == m);
        let fma = fi.fma.get(m).copied();
        let (_, ma, _) = leaf(tape, a);
        let (_, mb, _) = leaf(tape, b);
        match op {
            Operation::Add | Operation::Sub => {
                let s = if op == Operation::Sub { -1.0 } else { 1.0 };
                // A regroup step's roundings are its fold's (`fold_end`); an
                // fma pair's add may skip its product's rounding.
                let own = match fma {
                    Some(slot) if !step => Own {
                        nu: 2.0 * u * ((if slot == 0 { x0 } else { x1 }).abs() + yv.abs()),
                        src: Src::Fma,
                        rounds: false,
                    },
                    _ => EXACT,
                };
                push(tape, y, [(a, 1.0), (b, s)], own, ma + mb)
            }
            Operation::Mul => {
                let p = ir.precision(m);
                let (d0, d1) = (member_delta(p, x0), member_delta(p, x1));
                let own = Own {
                    nu: (d0 * d1).abs(),
                    src: Src::Member,
                    rounds: d0 != 0.0 || d1 != 0.0,
                };
                let r = push(tape, y, [(a, x1), (b, x0)], own, yv.abs());
                if let RValue::Tracked(_, i) = r
                    && p.reduced()
                {
                    let keys = match tape.keys.get(m) {
                        Some(k) => *k,
                        None => {
                            let k = [tape.nkeys, tape.nkeys + 1];
                            tape.nkeys += 2;
                            tape.keys.insert(m, k);
                            k
                        }
                    };
                    for (slot, x, xv, other, d) in [(0, a, x0, x1, d0), (1, b, x1, x0, d1)] {
                        if d != 0.0 {
                            tape.events.push(Event {
                                node: i,
                                key: keys[slot],
                                d: d * other,
                            });
                        }
                        let (xn, _, noisy) = leaf(tape, x);
                        if noisy && let Some(tie) = member_tie(p, xv) {
                            tape.reads.push(Read {
                                x: xn,
                                at: i,
                                tie,
                                other: Some((keys[slot], other)),
                            });
                        }
                    }
                }
                r
            }
            Operation::Div => push(tape, y, [(a, 1.0 / x1), (b, -yv / x1)], EXACT, yv.abs()),
            _ => push(
                tape,
                y,
                [(a, 1.0), (b, -(x0 / x1).trunc())],
                EXACT,
                yv.abs(),
            ),
        }
    })
}

/// `-x`.
pub(crate) fn neg(x: &RValue, y: Value) -> RValue {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let tape = t.as_mut().expect("recording");
        let m = num(&y).map_or(0.0, f64::abs);
        push(tape, y, [(x, -1.0), (&UNIT, 0.0)], EXACT, m)
    })
}

/// An exact widening: the operand's node, the wider value.
pub(crate) fn widen(x: &RValue, y: Value) -> RValue {
    match x {
        RValue::Tracked(_, i) if num(&y).is_some() => RValue::Tracked(y, *i),
        _ => RValue::Scalar(y),
    }
}

/// `x -> to_<n>` (ADR-0043 D3): exact where `x` is; else the operand's
/// deviation passes, and a tie may flip — `2·(u·|x| + abs)`.
pub(crate) fn narrow(x: &RValue, n: NarrowFloat, y: Value) -> RValue {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let tape = t.as_mut().expect("recording");
        let (xn, _, noisy) = leaf(tape, x);
        let xs = num(crate::eval::scalar(x)).unwrap_or(0.0);
        let r = push(tape, y.clone(), [(x, 1.0), (&UNIT, 0.0)], EXACT, xs.abs());
        if noisy
            && xs.is_finite()
            && let (RValue::Tracked(_, at), Some(yv)) = (&r, num(&y))
        {
            let f = mapal_ir::narrow::NarrowForm::of(n);
            tape.reads.push(Read {
                x: xn,
                at: *at,
                tie: tie(xs, f.precision as i32, f.emin, yv),
                other: None,
            });
        }
        r
    })
}

/// A math builtin `y = f(x)` at width `bits`: its contract on each face.
pub(crate) fn math(fun: MathFn, bits: u32, x: &RValue, y: Value) -> RValue {
    let (Some(xv), Some(yv)) = (num(crate::eval::scalar(x)), num(&y)) else {
        return RValue::Scalar(y);
    };
    let Some(c) = mapal_ir::math::contract(fun, bits) else {
        return RValue::Scalar(y);
    };
    let d = match fun {
        MathFn::Exp => yv,
        MathFn::Sqrt => 0.5 / yv,
        MathFn::Sin => xv.cos(),
        MathFn::Cos => -xv.sin(),
    };
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let tape = t.as_mut().expect("recording");
        let own = Own {
            nu: 2.0 * (c.rel * yv.abs() + c.abs),
            src: Src::Math,
            rounds: false,
        };
        push(tape, y, [(x, d), (&UNIT, 0.0)], own, yv.abs())
    })
}

/// A fold of body `body` over `k` elements produced `total`: where its step
/// regroups, any order (ADR-0040 D2: `γ_{K+1}(u_acc)·M` per face, `M` the
/// chain's magnitudes — a product fold's `|total|`; a running max / min
/// regroups exactly).
pub(crate) fn fold_end(ir: &CategoryIr, body: FuncId, k: usize, total: RValue) -> RValue {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        let tape = t.as_mut().expect("recording");
        let fi = info(tape, ir, body);
        let (regroup, member) = (fi.regroup, fi.member);
        let Some((_, op)) = regroup else {
            return total;
        };
        let y = crate::eval::scalar(&total).clone();
        let Some(v) = num(&y).map(f64::abs) else {
            return total;
        };
        let u = match (&y, member) {
            (Value::F64(_), _) => 2f64.powi(-53),
            (_, true) => 2f64.powi(-23),
            _ => 2f64.powi(-24),
        };
        let (_, m, _) = leaf(tape, &total);
        let mag = match op {
            RowFoldOp::Arith {
                op: Operation::Add | Operation::Sub,
                ..
            } => m,
            RowFoldOp::Arith { .. } => v,
            RowFoldOp::Select { .. } => return total,
        };
        let own = Own {
            nu: 2.0 * mapal_ir::gamma(k as u64 + 1, u) * mag,
            src: Src::Order,
            rounds: false,
        };
        push(tape, y, [(&total, 1.0), (&UNIT, 0.0)], own, m)
    })
}

/// A printed scalar (`newline`: the print ends its line).
pub(crate) fn printed(x: &RValue, newline: bool) {
    TAPE.with(|t| {
        if let Some(tape) = t.borrow_mut().as_mut() {
            let v = crate::eval::scalar(x);
            let text = match v {
                Value::F32(_) | Value::F64(_) | Value::Narrow(..) => None,
                Value::Str(s) => Some(s.clone()),
                _ => Some("0e0".to_string()),
            };
            let node = match x {
                RValue::Tracked(_, i) => Some(*i),
                _ => None,
            };
            tape.printed.push((node, text, newline));
        }
    });
}
