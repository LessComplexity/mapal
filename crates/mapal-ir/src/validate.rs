//! `validate()` — the independent oracle (DESIGN §11).
//!
//! A from-scratch pass over a sealed [`CategoryIr`] that re-derives the
//! invariants **without sharing code with the builder's checks**. Its purpose is
//! (a) the property test's oracle — *every graph the public API produces
//! validates clean* — and (b) a debug-assert hook for future passes.
//!
//! **Independence is scoped honestly** (DESIGN §11): each invariant splits into a
//! *graph-shape clause* (validate-checkable from the sealed graph alone) and,
//! for some, an *API-discipline / provenance clause* (builder-only). validate()
//! certifies exactly the graph-shape clauses. The provenance clauses it cannot
//! check are listed below.
//!
//! Clauses validate() **cannot** independently certify (provenance only):
//! - I2: that each edge arose from the builder's per-call typing dispatch — but
//!   the *resulting* §5.1 shape is re-derived here from tys.
//! - I7: that `constant()` is the only setter of `value` — but the
//!   `value.is_some() ⇔ kind==Constant ∧ value.ty()==ty` shape *is* checked.
//! - I8: the "via the ret-write API" provenance — reduced here to the
//!   graph-shape clause "`op == Output` ⇒ `target.kind == Return` ∧ source ty =
//!   target ty".
//! - The I4 loop-fork *origin* (a `LoopHandle`) — re-expressed structurally via
//!   SCC membership.

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::{self, Ty, Value};

mod flow_checks;

use self::flow_checks::*;

/// A well-formedness violation found by [`validate`] (DESIGN §11). Mirrors
/// `IrError` but carries ids instead of build context — no `Display` (C3).
#[derive(Clone, Debug, PartialEq)]
pub enum IrViolation {
    /// An object's in-edge set does not match any I3 shape.
    BadInEdges(ObjectId),
    /// An edge violates the §5.1 typing table.
    BadEdgeType(MorphismId),
    /// A `Constant` with `value` shape wrong, or a non-Constant with a value.
    BadConstant(ObjectId),
    /// A Parameter/Constant with a non-empty in-edge set.
    SourceHasInEdges(ObjectId),
    /// A Return object with out-edges (must be a sink).
    ReturnHasOutEdges(ObjectId),
    /// `op == Output` whose target is not a Return, or ty mismatch.
    BadOutput(MorphismId),
    /// I4: a token-bearing object with >1 token consumer outside the loop fork.
    TokenNotLinear(ObjectId),
    /// I4b: a token-bearing non-Return object with no token consumer.
    TokenDropped(ObjectId),
    /// I4: a `Phi` selecting a token-bearing ty.
    TokenInPhi(MorphismId),
    /// I4: a token inside a Map/Fold body.
    TokenInBody(FuncId),
    /// §8: a loop-carried token with a token-free exit payload.
    TokenNotEscaping(ObjectId),
    /// I9s: `Str` in a product/array outside the print pair.
    StrOutsidePrint(ObjectId),
    /// Two `Struct` tys sharing a name with different fields.
    StructNameConflict(String),
    /// A cross-function morphism (source/target owners differ).
    CrossFunctionEdge(MorphismId),
    /// An object/morphism with no owner.
    Unowned(ObjectId),
    /// I5: a `LoopMerge` whose SCC placement is malformed.
    BadLoop(ObjectId),
    /// I6: a cycle in the function reference graph.
    RecursiveReference(Vec<FuncId>),
    /// I-RET: mixed full/slot writers, missing slot, etc.
    BadReturn(ObjectId),
    /// I10: a ty deeper than `MAX_TY_DEPTH`.
    TyTooDeep(ObjectId),
    /// I9: a non-Core ty on some object.
    NonCoreType(ObjectId),
    /// ADR-0018: an `Enumerate` edge over an array of size `> i32::MAX` (the
    /// index component `i32` cannot name every element). Twin of
    /// [`crate::IrError::EnumerateIndexOverflow`].
    EnumerateIndexOverflow(MorphismId),
    /// ADR-0029: an `Iota` edge whose source is not a `Constant` integer equal
    /// to the target array's static size (the static-n rule; builder-minted,
    /// this is defense-in-depth). Twin of the builder's static-n rejection.
    IotaCountMismatch(MorphismId),
    /// ADR-0029: a `Widen` edge outside the allowed numeric lattice. Twin of
    /// [`crate::IrError::InvalidWiden`].
    InvalidWiden(MorphismId),
    /// ADR-0027: a `Map`/`Fold` capture component with no runtime
    /// representation (`Unit`, an erased-element array, or an all-erased
    /// product) — no body-input slot on the backends. Twin of
    /// [`crate::IrError::ErasedCapture`].
    ErasedCapture(MorphismId),
}

/// Re-derive every graph-shape invariant on a sealed graph (DESIGN §11). An
/// empty vec means well-formed.
pub fn validate(ir: &CategoryIr) -> Vec<IrViolation> {
    let mut v = Vec::new();
    check_ownership(ir, &mut v);
    check_objects(ir, &mut v);
    check_edges(ir, &mut v);
    check_in_edge_shapes(ir, &mut v);
    check_tokens(ir, &mut v);
    check_str(ir, &mut v);
    check_struct_names(ir, &mut v);
    check_returns(ir, &mut v);
    check_loops(ir, &mut v);
    check_references(ir, &mut v);
    v
}

/// I6 cross-function + every object/morphism owned.
fn check_ownership(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (id, _) in ir.objects() {
        if ir.object(id).is_some() && ir.try_owner(id).is_none() {
            v.push(IrViolation::Unowned(id));
        }
    }
    for (mid, m) in ir.morphisms() {
        let so = ir.try_owner(m.source);
        let to = ir.try_owner(m.target);
        if so.is_none() || to.is_none() || so != to {
            v.push(IrViolation::CrossFunctionEdge(mid));
        }
    }
}

/// I7 constant shape, I9/I10 ty intake re-derivation, Return-is-sink,
/// Parameter/Constant in-edge-free.
fn check_objects(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (id, obj) in ir.objects() {
        if !ty::ty_depth_ok(&obj.ty) {
            v.push(IrViolation::TyTooDeep(id));
            continue;
        }
        if !ty_is_core(&obj.ty) {
            v.push(IrViolation::NonCoreType(id));
        }
        // I7: value.is_some() ⇔ kind==Constant, and value.ty()==ty.
        match (&obj.value, obj.kind) {
            (Some(val), ObjectKind::Constant) => {
                if val.ty() != obj.ty {
                    v.push(IrViolation::BadConstant(id));
                }
            }
            (Some(_), _) => v.push(IrViolation::BadConstant(id)),
            (None, ObjectKind::Constant) => v.push(IrViolation::BadConstant(id)),
            (None, _) => {}
        }
        // Parameter/Constant must have no in-edges.
        if matches!(obj.kind, ObjectKind::Parameter | ObjectKind::Constant)
            && !ir.in_edges(id).is_empty()
        {
            v.push(IrViolation::SourceHasInEdges(id));
        }
        // Return is a sink.
        if obj.kind == ObjectKind::Return && !ir.out_edges(id).is_empty() {
            v.push(IrViolation::ReturnHasOutEdges(id));
        }
    }
}

/// I2 (graph-shape clause) + I8: re-derive the §5.1 typing table from tys.
fn check_edges(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (mid, m) in ir.morphisms() {
        let (sty, tty) = match (ir.object(m.source), ir.object(m.target)) {
            (Some(s), Some(t)) => (&s.ty, &t.ty),
            _ => {
                v.push(IrViolation::BadEdgeType(mid));
                continue;
            }
        };
        if !edge_type_ok(ir, m, sty, tty) {
            if m.op == Operation::Widen {
                v.push(IrViolation::InvalidWiden(mid));
            } else {
                v.push(IrViolation::BadEdgeType(mid));
            }
        }
        // I8 graph-shape clause: Output ⇒ target is Return ∧ source ty == target ty.
        if m.op == Operation::Output {
            let tgt_kind = ir.object(m.target).map(|o| o.kind);
            if tgt_kind != Some(ObjectKind::Return) || sty != tty {
                v.push(IrViolation::BadOutput(mid));
            }
        }
        // ADR-0018 extra condition (independent of the typing shape above):
        // an `Enumerate` array must be indexable by `i32` — `n ≤ i32::MAX`.
        if m.op == Operation::Enumerate
            && let Ty::Array { size, .. } = sty
            && *size > i32::MAX as u64
        {
            v.push(IrViolation::EnumerateIndexOverflow(mid));
        }
        // ADR-0029 extra condition (independent of the typing shape above):
        // an `Iota` source — resp. a `Fill` internal pair's slot-1 feeder —
        // must be a `Constant` integer whose value equals the target array's
        // static size (the static-n rule).
        if matches!(m.op, Operation::Iota | Operation::Fill) {
            let count_obj = match m.op {
                Operation::Iota => ir.object(m.source),
                Operation::Fill => ir.in_edges(m.source).iter().find_map(|&pm| {
                    let pmorph = ir.morphism(pm).expect("morphism resolves");
                    if matches!(pmorph.op, Operation::Pair { slot: 1, .. }) {
                        ir.object(pmorph.source)
                    } else {
                        None
                    }
                }),
                _ => None,
            };
            let ok = count_obj
                .map(|src| {
                    src.kind == ObjectKind::Constant
                        && match (&src.value, ir.object(m.target).map(|t| &t.ty)) {
                            (Some(Value::I32(x)), Some(Ty::Array { size, .. })) => {
                                *x >= 0 && *x as u64 == *size
                            }
                            (Some(Value::I64(x)), Some(Ty::Array { size, .. })) => {
                                *x >= 0 && *x as u64 == *size
                            }
                            (Some(Value::U8(x)), Some(Ty::Array { size, .. })) => {
                                *x as u64 == *size
                            }
                            _ => false,
                        }
                })
                .unwrap_or(false);
            if !ok {
                v.push(IrViolation::IotaCountMismatch(mid));
            }
        }
        // ADR-0027 extra condition (defense in depth; the builder rejects
        // these as `ErasedCapture`): a capture component of a capturing
        // Map/Fold must have a runtime representation. `Str` components are
        // covered by `StrOutsidePrint`, so they are not re-diagnosed here.
        if let Operation::Map { captures, .. } | Operation::Fold { captures, .. } = m.op {
            let k = captures as usize;
            if k > 0
                && let Ty::Tuple(ts) = sty
                && ts.len() >= k
                && ts[..k]
                    .iter()
                    .any(|c| ty::ty_residual_empty(c) && !ty::ty_contains_str(c))
            {
                v.push(IrViolation::ErasedCapture(mid));
            }
        }
    }
}

/// I2: whether a single edge satisfies its §5.1 row, derived only from tys.
fn edge_type_ok(ir: &CategoryIr, m: &crate::graph::Morphism, sty: &Ty, tty: &Ty) -> bool {
    match m.op {
        Operation::Pair { slot, arity } => {
            // target is a product of the given arity; source matches slot ty.
            // `product_arity()` is `u64` (F4/SND-3); the `Pair` field is `u32`,
            // so a > u32::MAX-arity array can never have a matching `Pair` edge.
            match tty.product_arity() {
                Some(a) if a == arity as u64 && slot < arity => {
                    tty.component_ty(slot).map(|c| c == sty).unwrap_or(false)
                }
                _ => false,
            }
        }
        Operation::Proj { index } => {
            matches!(sty, Ty::Tuple(_) | Ty::Struct { .. })
                && sty.component_ty(index).map(|c| c == tty).unwrap_or(false)
        }
        Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
            two_tuple(sty)
                .map(|(a, b)| a == b && a.is_numeric() && tty == a)
                .unwrap_or(false)
        }
        Operation::Neg => sty.is_numeric() && sty == tty,
        // ADR-0042 / ADR-0046: a math builtin keeps its float width.
        Operation::Math(_) => matches!(sty, Ty::Float { bits: 32 | 64 }) && sty == tty,
        Operation::Eq | Operation::Neq => two_tuple(sty)
            .map(|(a, b)| a == b && (a.is_numeric() || *a == Ty::Bool) && *tty == Ty::Bool)
            .unwrap_or(false),
        Operation::Lt | Operation::Gt | Operation::Le | Operation::Ge => two_tuple(sty)
            .map(|(a, b)| a == b && a.is_numeric() && *tty == Ty::Bool)
            .unwrap_or(false),
        Operation::And | Operation::Or => two_tuple(sty)
            .map(|(a, b)| *a == Ty::Bool && *b == Ty::Bool && *tty == Ty::Bool)
            .unwrap_or(false),
        Operation::Not => *sty == Ty::Bool && *tty == Ty::Bool,
        Operation::Widen => matches!(
            (sty, tty),
            (
                Ty::Int {
                    bits: 32,
                    signed: true
                },
                Ty::Int {
                    bits: 64,
                    signed: true
                } | Ty::Float { bits: 32 | 64 }
            ) | (Ty::Float { bits: 32 }, Ty::Float { bits: 64 })
                | (Ty::Narrow(_), Ty::Float { bits: 32 | 64 })
        ),
        // ADR-0043: one rounding from a float into a narrow format.
        Operation::Narrow => *sty == Ty::f32() && matches!(tty, Ty::Narrow(_)),
        Operation::Phi => match sty {
            Ty::Tuple(ts) if ts.len() == 3 => {
                ts[0] == ts[1]
                    && ts[2] == Ty::Bool
                    && &ts[0] == tty
                    && !ty::ty_contains_token(&ts[0])
            }
            _ => false,
        },
        Operation::Call(g) => match ir.func(g) {
            Some(def) => {
                ir.object(def.input).map(|o| &o.ty) == Some(sty)
                    && ir.object(def.output).map(|o| &o.ty) == Some(tty)
            }
            None => false,
        },
        Operation::Map { body, captures } => {
            let k = captures as usize;
            if k == 0 {
                // ADR-0009/LC-2 shape: bare array source, body input = element.
                match (sty, tty, ir.func(body)) {
                    (
                        Ty::Array { elem: se, size: sn },
                        Ty::Array { elem: te, size: tn },
                        Some(def),
                    ) => {
                        sn == tn
                            && ir.object(def.input).map(|o| &o.ty) == Some(&**se)
                            && ir.object(def.output).map(|o| &o.ty) == Some(&**te)
                    }
                    _ => false,
                }
            } else {
                // ADR-0027: source (c₁…cₖ, [T; n]) → [U; n]; body input (c₁…cₖ, T).
                match (sty, tty, ir.func(body)) {
                    (Ty::Tuple(src), Ty::Array { elem: te, size: tn }, Some(def))
                        if src.len() == k + 1 =>
                    {
                        if let Ty::Array { elem: se, size: sn } = &src[k] {
                            if sn != tn {
                                return false;
                            }
                            let mut want: Vec<Ty> = src[..k].to_vec();
                            want.push(se.as_ref().clone());
                            ir.object(def.input).map(|o| &o.ty) == Some(&Ty::Tuple(want))
                                && ir.object(def.output).map(|o| &o.ty) == Some(te.as_ref())
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }
        }
        Operation::Fold { body, captures } => {
            let k = captures as usize;
            if k == 0 {
                match (sty, ir.func(body)) {
                    (Ty::Tuple(ts), Some(def)) if ts.len() == 2 => {
                        if let Ty::Array { elem, .. } = &ts[1] {
                            let body_in = ir.object(def.input).map(|o| o.ty.clone());
                            let body_out = ir.object(def.output).map(|o| o.ty.clone());
                            let want_in = Ty::Tuple(vec![ts[0].clone(), (**elem).clone()]);
                            body_in == Some(want_in)
                                && body_out.as_ref() == Some(&ts[0])
                                && tty == &ts[0]
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            } else {
                // ADR-0027: source (c₁…cₖ, Acc, [T; n]) → Acc; body input (c₁…cₖ, Acc, T).
                match (sty, ir.func(body)) {
                    (Ty::Tuple(ts), Some(def)) if ts.len() == k + 2 => {
                        if let Ty::Array { elem, .. } = &ts[k + 1] {
                            let acc_ty = &ts[k];
                            let mut want: Vec<Ty> = ts[..k].to_vec();
                            want.push(acc_ty.clone());
                            want.push(elem.as_ref().clone());
                            let body_in = ir.object(def.input).map(|o| o.ty.clone());
                            let body_out = ir.object(def.output).map(|o| o.ty.clone());
                            body_in == Some(Ty::Tuple(want))
                                && body_out.as_ref() == Some(acc_ty)
                                && tty == acc_ty
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }
        }
        Operation::Index => match sty {
            Ty::Tuple(ts) if ts.len() == 2 => match &ts[0] {
                Ty::Array { elem, .. } => ts[1].is_integer() && &**elem == tty,
                _ => false,
            },
            _ => false,
        },
        // ADR-0018. Pure typing shape only; the `n ≤ i32::MAX` extra condition is
        // re-derived separately in `check_edges` (an "extra condition", not a
        // typing judgment — matching this module's golden scope).
        Operation::Zip => match (sty, tty) {
            (Ty::Tuple(ts), Ty::Array { elem: te, size: tn }) if ts.len() == 2 => {
                match (&ts[0], &ts[1], &**te) {
                    (
                        Ty::Array { elem: ae, size: an },
                        Ty::Array { elem: be, size: bn },
                        Ty::Tuple(pair),
                    ) if pair.len() == 2 => {
                        an == bn && an == tn && pair[0] == **ae && pair[1] == **be
                    }
                    _ => false,
                }
            }
            _ => false,
        },
        Operation::Enumerate => match (sty, tty) {
            (Ty::Array { elem: se, size: sn }, Ty::Array { elem: te, size: tn }) => {
                sn == tn
                    && match &**te {
                        Ty::Tuple(pair) if pair.len() == 2 => {
                            pair[0] == Ty::i32() && pair[1] == **se
                        }
                        _ => false,
                    }
            }
            _ => false,
        },
        // ADR-0029. Pure typing shapes only; the `Iota` count-constant value
        // tie is the extra condition in `check_edges` (the Enumerate split).
        Operation::Iota => match tty {
            Ty::Array { elem, size } => sty.is_integer() && **elem == Ty::index_ty(*size),
            _ => false,
        },
        // `(T × I) → [T; n]` — the internal 2-tuple (value, count), as Zip.
        Operation::Fill => match (sty, tty) {
            (Ty::Tuple(ts), Ty::Array { elem, .. }) if ts.len() == 2 => {
                ts[1].is_integer() && **elem == ts[0]
            }
            _ => false,
        },
        // ADR-0021: (Array{T,n}, I, T) -> Array{T,n}. I any Core integer scalar;
        // val ty = elem ty; target = the source array ty (fresh, same shape).
        Operation::Update => match sty {
            Ty::Tuple(ts) if ts.len() == 3 => match &ts[0] {
                Ty::Array { elem, .. } => ts[1].is_integer() && **elem == ts[2] && ts[0] == *tty,
                _ => false,
            },
            _ => false,
        },
        Operation::Print { .. } => match sty {
            Ty::Tuple(ts) if ts.len() == 2 => {
                ts[0] == Ty::IoToken && ts[1].is_printable() && *tty == Ty::IoToken
            }
            _ => false,
        },
        // ADR-0045: (IoToken, Str) -> (IoToken, [T; n]), T readable; the path
        // is the literal (a `Str` Constant — I9s keeps every Str one).
        Operation::Read => match (sty, tty) {
            (Ty::Tuple(ss), Ty::Tuple(ts)) if ss.len() == 2 && ts.len() == 2 => {
                ss[0] == Ty::IoToken
                    && ss[1] == Ty::Str
                    && ts[0] == Ty::IoToken
                    && crate::read::readable(&ts[1])
            }
            _ => false,
        },
        // plan-time-builtin: IoToken -> (IoToken, f64).
        Operation::TimeMs => match tty {
            Ty::Tuple(ts) if ts.len() == 2 => {
                *sty == Ty::IoToken && ts[0] == Ty::IoToken && ts[1] == Ty::f64()
            }
            _ => false,
        },
        Operation::LoopEnter => {
            sty == tty && ir.object(m.target).map(|o| o.kind) == Some(ObjectKind::LoopMerge)
        }
        Operation::LoopBack => match sty {
            Ty::Tuple(ts) if ts.len() == 2 => {
                ts[1] == Ty::Bool
                    && &ts[0] == tty
                    && ir.object(m.target).map(|o| o.kind) == Some(ObjectKind::LoopMerge)
            }
            _ => false,
        },
        Operation::LoopExit => match sty {
            Ty::Tuple(ts) if ts.len() == 2 => ts[1] == Ty::Bool && &ts[0] == tty,
            _ => false,
        },
        Operation::Output => sty == tty,
    }
}

/// I3: re-derive the one-definition rule from in-edge sets.
fn check_in_edge_shapes(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (id, obj) in ir.objects() {
        let ins = ir.in_edges(id);
        let ok = match obj.kind {
            ObjectKind::Parameter | ObjectKind::Constant => ins.is_empty(),
            ObjectKind::LoopMerge => {
                // 1 LoopEnter + ≥1 LoopBack, nothing else.
                let mut enters = 0;
                let mut backs = 0;
                let mut other = 0;
                for &mid in ins {
                    match ir.morphism(mid).unwrap().op {
                        Operation::LoopEnter => enters += 1,
                        Operation::LoopBack => backs += 1,
                        _ => other += 1,
                    }
                }
                enters == 1 && backs >= 1 && other == 0
            }
            ObjectKind::Return => true, // I-RET handled separately.
            ObjectKind::Temporary => {
                // (b) one value-producing definer, OR (c) exactly `arity` Pair
                // edges with distinct slots (product object).
                let all_pair = !ins.is_empty()
                    && ins
                        .iter()
                        .all(|&mid| matches!(ir.morphism(mid).unwrap().op, Operation::Pair { .. }));
                if all_pair {
                    product_pair_shape_ok(ir, &obj.ty, ins)
                } else {
                    ins.len() == 1
                }
            }
        };
        if !ok {
            v.push(IrViolation::BadInEdges(id));
        }
    }
}

/// Whether a product object's `Pair` in-edges cover exactly its arity with
/// distinct slots.
fn product_pair_shape_ok(ir: &CategoryIr, ty: &Ty, ins: &[MorphismId]) -> bool {
    // `product_arity()` is `u64` (F4/SND-3); `Pair` slot/arity fields are `u32`.
    // A product whose arity exceeds the in-edge count (incl. any > u32::MAX-arity
    // array) cannot be exactly covered, so the count check rejects it.
    let arity = match ty.product_arity() {
        Some(a) => a,
        None => return false,
    };
    if ins.len() as u64 != arity {
        return false;
    }
    let mut seen: Vec<u32> = Vec::with_capacity(ins.len());
    for &mid in ins {
        if let Operation::Pair { slot, arity: a } = ir.morphism(mid).unwrap().op {
            if a as u64 != arity || (slot as u64) >= arity || seen.contains(&slot) {
                return false;
            }
            seen.push(slot);
        } else {
            return false;
        }
    }
    true
}

/// I9s: `Str` may appear only as a Constant ty or the second component of a
/// `(IoToken, Str)` print pair. Re-derived from product membership.
fn check_str(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (id, obj) in ir.objects() {
        match &obj.ty {
            // A bare Str object is fine only if it is a Constant (the literal).
            Ty::Str => {
                if obj.kind != ObjectKind::Constant {
                    // A Str-typed temporary is only legitimate as the value
                    // feeding a print pair — but as a standalone object that is
                    // still just a Str scalar (the Constant). A non-constant Str
                    // scalar would be a lowering bug; flag it.
                    v.push(IrViolation::StrOutsidePrint(id));
                }
            }
            // The print pair: (IoToken, Str) is allowed; any other product
            // containing Str is not.
            Ty::Tuple(ts) => {
                if ts.iter().any(ty::ty_contains_str) {
                    let is_print_pair = ts.len() == 2 && ts[0] == Ty::IoToken && ts[1] == Ty::Str;
                    if !is_print_pair {
                        v.push(IrViolation::StrOutsidePrint(id));
                    }
                }
            }
            other => {
                if ty::ty_contains_str(other) {
                    v.push(IrViolation::StrOutsidePrint(id));
                }
            }
        }
    }
}

/// StructNameConflict: all `Struct`s sharing a name must have identical fields.
fn check_struct_names(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    let mut seen: Vec<(String, Vec<(String, Ty)>)> = Vec::new();
    for (_, obj) in ir.objects() {
        let mut stack: Vec<&Ty> = vec![&obj.ty];
        while let Some(t) = stack.pop() {
            match t {
                Ty::Struct { name, fields } => {
                    if let Some((_, prev)) = seen.iter().find(|(n, _)| n == name) {
                        if prev != fields {
                            v.push(IrViolation::StructNameConflict(name.clone()));
                        }
                    } else {
                        seen.push((name.clone(), fields.clone()));
                    }
                    for (_, inner) in fields {
                        stack.push(inner);
                    }
                }
                Ty::Tuple(ts) => stack.extend(ts.iter()),
                Ty::Array { elem, .. } => stack.push(elem),
                _ => {}
            }
        }
    }
}

/// I-RET: Return in-edge completeness, re-derived per function.
fn check_returns(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (_, def) in ir.funcs() {
        let ret = def.output;
        let out_ty = match ir.object(ret) {
            Some(o) => o.ty.clone(),
            None => continue,
        };
        let mut full = 0usize;
        let mut slots: Vec<u32> = Vec::new();
        for &mid in ir.in_edges(ret) {
            match ir.morphism(mid).unwrap().op {
                Operation::Pair { slot, .. } => slots.push(slot),
                _ => full += 1,
            }
        }
        if full > 0 && !slots.is_empty() {
            v.push(IrViolation::BadReturn(ret));
            continue;
        }
        if !slots.is_empty() {
            match out_ty.product_arity() {
                None => v.push(IrViolation::BadReturn(ret)),
                Some(arity) => {
                    slots.sort_unstable();
                    let dup = slots.windows(2).any(|w| w[0] == w[1]);
                    // Completeness against the true `u64` arity without
                    // truncation: the deduplicated `u32` slots must be exactly
                    // `0..arity`. A > u32::MAX-arity product (not slot-
                    // addressable) is correctly flagged missing (F4/SND-3).
                    let complete = !dup
                        && (slots.len() as u64) == arity
                        && slots.iter().enumerate().all(|(k, &s)| s as u64 == k as u64);
                    if dup || !complete {
                        v.push(IrViolation::BadReturn(ret));
                    }
                }
            }
        } else if full == 0 && out_ty != Ty::Unit {
            v.push(IrViolation::BadReturn(ret));
        }
    }
}

#[cfg(test)]
mod enumerate_bound_twin;
#[cfg(test)]
mod erased_capture_twin;
#[cfg(test)]
mod iota_count_twin;
#[cfg(test)]
mod typing_table_golden;
#[cfg(test)]
mod widen_twin;
