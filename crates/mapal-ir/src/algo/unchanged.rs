//! **The unchanged-since-last-time fact** (caching-memory,
//! [`CategoryIr::unchanged_plan`]): which part of a value is the same as the
//! last time it was computed — across the trips of a loop, and across the
//! calls of a function.
//!
//! **Across trips.** [`CategoryIr::write_delta`] says which cells of a
//! loop-carried array one trip writes. A value the body computes from it
//! differs from its previous trip's value only in the cells whose inputs
//! differ: a map's cell `t` reads, for every captured array, the cells its
//! body addresses — written `s·(t ÷ q) + [lo, hi]` (one row of `s` cells per
//! row of `q` output cells, a window around it) when the address is that
//! form ([`Lin`], derived through `+ − ·k ÷k %k`, guard selects and nested
//! map / fold bodies, whose own indices range over their extents). A written
//! row of the input is then a written row (or two, for a window reaching
//! into the neighbour) of the output: [`Delta::Rows`] maps to `Rows`,
//! [`Delta::Pieces`] (one row of every sequence) to `Pieces`. A row fold
//! (`out[b] = fold_k x[b·D + k]`), a per-row scale (`x[e] · r[e ÷ D]`), a
//! projection (`fold_k x[i·D + k] · w[k·N + j]`) and a rotation reading its
//! row's partner cell all keep the delta; a cell reading a whole captured
//! array, or a number that changes every trip, is [`Delta::Top`]. A value
//! whose every input is unchanged is unchanged ([`UnchangedPlan::same`]):
//! a transform of a carried array the loop never writes.
//!
//! On a decode loop that carries the history of a layer's inputs and
//! recomputes K and V for every position, this is what says only the new
//! position's rows of K and V differ — the KV cache, deduced.
//!
//! **Across calls.** Every value of a function is a function of some of the
//! function's input slots ([`UnchangedPlan::inputs`], a bit mask); one of
//! none (constants only, no clock read) is the same at every call.
//!
//! Sound by construction: a claim is only ever narrower than "everything
//! changed" where each step is exact (a delta's start divided by a row width
//! only where the counter's range keeps the start from wrapping).

use std::collections::HashMap;

use slotmap::SecondaryMap;

use super::{Affine, Delta, GuardSite};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, Value};

mod compute;

/// The per-function fact (see the module doc). Stored on the graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnchangedPlan {
    /// Across trips: every array a canonical loop's body computes on every
    /// trip (and every carried array's merge view), the cells that may differ
    /// from its value on the previous trip — [`Delta::Rows`] /
    /// [`Delta::Pieces`] (only those; `next` is the object itself unless it
    /// is [`CategoryIr::write_delta`]'s) or [`Delta::Top`]. An array of the
    /// body that is unchanged is in [`Self::same`] instead.
    pub trip: SecondaryMap<ObjectId, Delta>,
    /// The loop (its `LoopMerge`) of each [`Self::trip`] / [`Self::same`]
    /// entry.
    pub loop_of: SecondaryMap<ObjectId, ObjectId>,
    /// Values (arrays or numbers) a loop's body computes on every trip that
    /// are the same on every trip: every input is unchanged.
    pub same: SecondaryMap<ObjectId, ()>,
    /// Across calls: every value of the function, the input slots of the
    /// function it is a function of (bit `k`: slot `k`; slots past 63 share
    /// bit 63; a clock read: every bit). `0`: the same at every call.
    pub inputs: SecondaryMap<ObjectId, u64>,
    /// How many map sites the trip walk derived a row delta for (coverage).
    pub derived: u32,
}

impl UnchangedPlan {
    /// The cells of `o` that may differ from its previous trip's value;
    /// `None` when `o` is no loop-body array (or is unchanged, [`Self::same`]).
    pub fn delta(&self, o: ObjectId) -> Option<Delta> {
        self.trip.get(o).copied()
    }

    /// `o` is computed on every trip of a loop and equal on every trip.
    pub fn same_every_trip(&self, o: ObjectId) -> bool {
        self.same.contains_key(o)
    }

    /// `o` is the same at every call of its function (constants only).
    pub fn same_every_call(&self, o: ObjectId) -> bool {
        self.inputs.get(o) == Some(&0)
    }
}

/// `c·R + v`, `v ∈ [lo, hi]`, `R = t ÷ q` the map cell's row: an integer of
/// a map body (or of a body nested in it) as a multiple of the row plus a
/// bounded rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lin {
    pub c: i128,
    pub lo: i128,
    pub hi: i128,
}

impl Lin {
    fn konst(v: i128) -> Self {
        Lin { c: 0, lo: v, hi: v }
    }
}

/// One array read of a map cell: the map's capture slot (the mapped array:
/// slot `captures`) and its address (`None`: anywhere in the array).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowRead {
    pub slot: u32,
    pub at: Option<Lin>,
}

/// One body's evaluation scope: its input param, the [`Lin`] of each input
/// slot (`whole`: the input itself when it is no tuple), and the outer map's
/// capture slot each array input slot is.
struct Scope {
    input: ObjectId,
    slots: Vec<Option<Lin>>,
    whole: Option<Lin>,
    arrays: Vec<Option<u32>>,
    whole_array: Option<u32>,
}

/// The row-address walk of one map at one row width `q`.
struct Walk<'a> {
    ir: &'a CategoryIr,
    /// The largest row: `(n − 1) ÷ q`.
    rmax: i128,
    memo: HashMap<ObjectId, Option<Lin>>,
    reads: Vec<RowRead>,
    /// A body the walk cannot see through (a call): no claim.
    opaque: bool,
}

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

fn int_range(ty: &Ty) -> Option<(i128, i128)> {
    match *ty {
        Ty::Int { bits, signed: true } if (1..=64).contains(&bits) => {
            let m = 1i128 << (bits - 1);
            Some((-m, m - 1))
        }
        Ty::Int {
            bits,
            signed: false,
        } if (1..=64).contains(&bits) => Some((0, (1i128 << bits) - 1)),
        _ => None,
    }
}

impl Walk<'_> {
    /// The value's range over every row `R ∈ [0, rmax]` fits `ty` (the
    /// program computes it without wrapping).
    fn fits(&self, l: Lin, ty: &Ty) -> bool {
        let Some((tlo, thi)) = int_range(ty) else {
            return false;
        };
        let (a, b) = (0, l.c.checked_mul(self.rmax));
        let Some(b) = b else { return false };
        let (mn, mx) = (a.min(b), a.max(b));
        matches!((mn.checked_add(l.lo), mx.checked_add(l.hi)), (Some(x), Some(y)) if x >= tlo && y <= thi)
    }

    fn lin(&mut self, o: ObjectId, sc: &Scope, depth: u32) -> Option<Lin> {
        if depth > 64 {
            return None;
        }
        if let Some(&v) = self.memo.get(&o) {
            return v;
        }
        let r = self.lin_of(o, sc, depth);
        self.memo.insert(o, r);
        r
    }

    fn lin_of(&mut self, o: ObjectId, sc: &Scope, depth: u32) -> Option<Lin> {
        let ir = self.ir;
        if let Some(v) = const_int(ir, o) {
            return Some(Lin::konst(v));
        }
        if o == sc.input {
            return sc.whole;
        }
        let &[m] = ir.in_edges(o) else {
            return None;
        };
        let mm = ir.morphism(m)?;
        let ty = &ir.object(o)?.ty;
        let arg = |k| ir.pair_slot_source(mm.source, k);
        let r = match mm.op {
            Operation::Proj { index } if mm.source == sc.input => {
                return sc.slots.get(index as usize).copied().flatten();
            }
            Operation::Proj { index } => {
                let s = ir.pair_slot_source(mm.source, index)?;
                return self.lin(s, sc, depth + 1);
            }
            Operation::Widen => {
                let s = mm.source;
                if !matches!(ir.object(s)?.ty, Ty::Int { .. }) {
                    return None;
                }
                self.lin(s, sc, depth + 1)?
            }
            Operation::Neg => {
                let x = self.lin(mm.source, sc, depth + 1)?;
                Lin {
                    c: x.c.checked_neg()?,
                    lo: x.hi.checked_neg()?,
                    hi: x.lo.checked_neg()?,
                }
            }
            Operation::Add | Operation::Sub => {
                let x = self.lin(arg(0)?, sc, depth + 1)?;
                let y = self.lin(arg(1)?, sc, depth + 1)?;
                if mm.op == Operation::Add {
                    Lin {
                        c: x.c.checked_add(y.c)?,
                        lo: x.lo.checked_add(y.lo)?,
                        hi: x.hi.checked_add(y.hi)?,
                    }
                } else {
                    Lin {
                        c: x.c.checked_sub(y.c)?,
                        lo: x.lo.checked_sub(y.hi)?,
                        hi: x.hi.checked_sub(y.lo)?,
                    }
                }
            }
            Operation::Mul => {
                let x = self.lin(arg(0)?, sc, depth + 1)?;
                let y = self.lin(arg(1)?, sc, depth + 1)?;
                let (k, v) = match (x, y) {
                    (Lin { c: 0, lo, hi }, v) if lo == hi => (lo, v),
                    (v, Lin { c: 0, lo, hi }) if lo == hi => (lo, v),
                    _ => return None,
                };
                let (a, b) = (v.lo.checked_mul(k)?, v.hi.checked_mul(k)?);
                Lin {
                    c: v.c.checked_mul(k)?,
                    lo: a.min(b),
                    hi: a.max(b),
                }
            }
            Operation::Div | Operation::Mod => {
                let d = const_int(ir, ir.through_pairs(arg(1)?)).filter(|&d| d > 0)?;
                let x = self.lin(arg(0)?, sc, depth + 1)?;
                if x.c != 0 && (x.c < 0 || x.lo < 0 || x.c % d != 0) {
                    return None;
                }
                if mm.op == Operation::Div {
                    // (c·R + v) ÷ d = (c/d)·R + v ÷ d for v ≥ 0, d | c;
                    // `÷` is monotone, so an R-free interval maps by its ends.
                    Lin {
                        c: x.c / d,
                        lo: x.lo / d,
                        hi: x.hi / d,
                    }
                } else if x.lo >= 0 {
                    // (c·R + v) % d = v % d for v ≥ 0, d | c.
                    if x.lo / d == x.hi / d {
                        Lin {
                            c: 0,
                            lo: x.lo % d,
                            hi: x.hi % d,
                        }
                    } else {
                        Lin {
                            c: 0,
                            lo: 0,
                            hi: d - 1,
                        }
                    }
                } else {
                    return None;
                }
            }
            // A select: either arm's value.
            Operation::Phi => {
                let x = self.lin(arg(0)?, sc, depth + 1)?;
                let y = self.lin(arg(1)?, sc, depth + 1)?;
                if x.c != y.c {
                    return None;
                }
                Lin {
                    c: x.c,
                    lo: x.lo.min(y.lo),
                    hi: x.hi.max(y.hi),
                }
            }
            _ => return None,
        };
        self.fits(r, ty).then_some(r)
    }

    /// The array object `arr` of body scope `sc` as the outer map's capture
    /// slot, if it is one (through guard-routing `Proj`-of-`Pair` hops).
    fn outer_slot(&self, arr: ObjectId, sc: &Scope) -> Option<u32> {
        let arr = self.ir.through_pairs(arr);
        if arr == sc.input {
            return sc.whole_array;
        }
        let k = proj_index(self.ir, arr, sc.input)?;
        sc.arrays.get(k as usize).copied().flatten()
    }

    /// Every array read of body `body` evaluated in `sc` (nested bodies too).
    fn walk(&mut self, body: FuncId, sc: &Scope, depth: u32) {
        let ir = self.ir;
        if depth > 8 {
            self.opaque = true;
            return;
        }
        let Some(fd) = ir.func(body) else {
            self.opaque = true;
            return;
        };
        for &g in &fd.morphisms {
            let Some(x) = ir.morphism(g) else { continue };
            match x.op {
                Operation::Call(_) => self.opaque = true,
                Operation::Index => {
                    let (Some(arr), Some(addr)) = (
                        ir.pair_slot_source(x.source, 0),
                        ir.pair_slot_source(x.source, 1),
                    ) else {
                        self.opaque = true;
                        continue;
                    };
                    if let Some(slot) = self.outer_slot(arr, sc) {
                        let at = self.lin(addr, sc, 0);
                        self.reads.push(RowRead { slot, at });
                    }
                }
                Operation::Map {
                    body: nb,
                    captures: nc,
                }
                | Operation::Fold {
                    body: nb,
                    captures: nc,
                } => {
                    let fold = matches!(x.op, Operation::Fold { .. });
                    let Some(nd) = ir.func(nb) else {
                        self.opaque = true;
                        continue;
                    };
                    let walked = match (fold, nc) {
                        (false, 0) => Some(x.source),
                        (false, _) => ir.pair_slot_source(x.source, nc),
                        (true, _) => ir.pair_slot_source(x.source, nc + 1),
                    };
                    let Some(walked) = walked else {
                        self.opaque = true;
                        continue;
                    };
                    // The element: an index over an iota, else a read of the
                    // walked array (all of it, when it is a capture).
                    let elem = match ir.tile_iota_size(walked) {
                        Some(k) if k > 0 => Some(Lin {
                            c: 0,
                            lo: 0,
                            hi: i128::from(k) - 1,
                        }),
                        _ => {
                            if let Some(slot) = self.outer_slot(walked, sc) {
                                self.reads.push(RowRead { slot, at: None });
                            }
                            None
                        }
                    };
                    let mut slots = Vec::new();
                    let mut arrays = Vec::new();
                    for k in 0..nc {
                        let o = ir.pair_slot_source(x.source, k);
                        slots.push(o.and_then(|o| self.lin(o, sc, 0)));
                        arrays.push(o.and_then(|o| self.outer_slot(o, sc)));
                    }
                    let (whole, whole_array) = if nc == 0 && !fold {
                        (elem, None)
                    } else {
                        if fold {
                            slots.push(None); // the accumulator
                            arrays.push(None);
                        }
                        slots.push(elem);
                        arrays.push(None);
                        (None, None)
                    };
                    let inner = Scope {
                        input: nd.input,
                        slots,
                        whole,
                        arrays,
                        whole_array,
                    };
                    self.walk(nb, &inner, depth + 1);
                }
                _ => {}
            }
        }
    }
}

/// `o` is `Proj{k}` of `input`: `k`.
fn proj_index(ir: &CategoryIr, o: ObjectId, input: ObjectId) -> Option<u32> {
    let &[m] = ir.in_edges(o) else {
        return None;
    };
    let mm = ir.morphism(m)?;
    match mm.op {
        Operation::Proj { index } if mm.source == input => Some(index),
        _ => None,
    }
}

/// Floor division.
fn fdiv(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

/// A changed value's state in the trip walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chg {
    Same,
    /// An array: the cells that may differ.
    Arr(Delta),
    /// A number or product that may differ.
    Num,
}

/// **Across-call retention at one call site** (`retain-call`,
/// [`CategoryIr::call_keep`]): a `Call(g)` site of `f` inside a loop of `f`
/// runs once per trip; the callee input slots whose argument is the same at
/// every execution of the site — an object made outside every loop that
/// contains the site, or one [`UnchangedPlan::same`] says is equal on every
/// trip (objects are immutable: the same object is the same value) — and
/// with [`UnchangedPlan::inputs`] of the callee, the callee values all of
/// whose input slots are kept: on every execution after the site's first
/// they equal the first's (the consumer computes them once and retains
/// them). A site outside every loop runs once per call of `f`: nothing
/// kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallKeep {
    pub callee: Option<FuncId>,
    /// Bit `k`: the argument of the callee's input slot `k` is the same at
    /// every execution of the site (slots past 63 share bit 63, never kept).
    pub kept: u64,
    /// The callee's values whose input slots are all kept (at least one:
    /// a constant-only value, mask 0, is the same at every site already).
    pub values: Vec<ObjectId>,
    /// The loops (their `LoopMerge`) containing the site.
    pub loops: Vec<ObjectId>,
}

fn k_tuple(ty: &Option<Ty>) -> bool {
    matches!(ty, Some(Ty::Tuple(_)))
}

/// The join of two trip states (the cells either may differ in).
fn join(a: Chg, b: Chg) -> Chg {
    match (a, b) {
        (Chg::Same, x) | (x, Chg::Same) => x,
        (Chg::Num, _) | (_, Chg::Num) => Chg::Num,
        (Chg::Arr(x), Chg::Arr(y)) => Chg::Arr(delta_union(x, y)),
    }
}

/// The union of two deltas of one array: one range, or one piece set, when
/// they overlap or touch; [`Delta::Top`] otherwise.
fn delta_union(x: Delta, y: Delta) -> Delta {
    match (x, y) {
        _ if x == y => x,
        (
            Delta::Rows {
                stride: s1,
                start: a1,
                next,
            },
            Delta::Rows {
                stride: s2,
                start: a2,
                ..
            },
        ) if a1.var == a2.var && a1.a == a2.a => {
            let (b1, b2) = (i128::from(a1.b), i128::from(a2.b));
            let (e1, e2) = (b1 + i128::from(s1), b2 + i128::from(s2));
            let (lo, hi) = (b1.min(b2), e1.max(e2));
            if b1.max(b2) > e1.min(e2) {
                return Delta::Top;
            }
            match (i64::try_from(lo), u64::try_from(hi - lo)) {
                (Ok(b), Ok(stride)) => Delta::Rows {
                    stride,
                    start: Affine { b, ..a1 },
                    next,
                },
                _ => Delta::Top,
            }
        }
        (
            Delta::Pieces {
                w: w1,
                step,
                start: a1,
                next,
            },
            Delta::Pieces {
                w: w2,
                step: st2,
                start: a2,
                ..
            },
        ) if step == st2 && a1.var == a2.var && a1.a == a2.a => {
            let (b1, b2) = (i128::from(a1.b), i128::from(a2.b));
            let (e1, e2) = (b1 + i128::from(w1), b2 + i128::from(w2));
            let (lo, hi) = (b1.min(b2), e1.max(e2));
            if b1.max(b2) > e1.min(e2) || hi - lo >= i128::from(step) {
                return Delta::Top;
            }
            match (i64::try_from(lo), u64::try_from(hi - lo)) {
                (Ok(b), Ok(w)) => Delta::Pieces {
                    w,
                    step,
                    start: Affine { b, ..a1 },
                    next,
                },
                _ => Delta::Top,
            }
        }
        _ => Delta::Top,
    }
}
