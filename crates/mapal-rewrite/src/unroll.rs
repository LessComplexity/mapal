//! R-UN — unroll a counted loop whose trips are known before the program runs.
//!
//! A canonical counted loop (`trip_split`: a counter `t`, `t + 1 → t`, a
//! constant first trip, continued while `t < hi`, `hi` a constant) is replayed
//! as its trips one after another: trip `k`'s state is trip `k − 1`'s next
//! state, its counter the constant `lo + k` (constant folding takes it from
//! there), and the exit value is computed from the last state — when **a trip
//! takes a record of an array made outside the loop at the counter**
//! (`model.layers[layer]`, a struct that holds arrays): each trip's view is
//! another part of one invariant value, and unrolled each view is at a
//! constant index — made once, outside every loop, placed once. The function
//! is then loop-free, so a layer loop in a function called inside another loop
//! (a decode step's) inlines there instead of nesting a loop — which neither
//! lowering nor the backends realize — or staying a call whose arrays cross
//! the boundary on each execution. A row of an array of arrays (a sequence's
//! step, a matrix's row) is data a loop walks: it stays a loop.
//!
//! The body must do array work (a map, fold or call besides the view), the
//! state must carry no IO token (an effect's order is the loop's), the trips
//! at most [`UNROLL_MAX_TRIPS`] and the unrolled body at most
//! [`UNROLL_MAX_MORPHISMS`] — recorded bounds on compile time and code size,
//! never semantics-bearing.

use slotmap::SecondaryMap;

use mapal_ir::{CategoryIr, ObjectId, Operation, Ty, ty_contains_token};

use crate::plan::RewritePlan;

/// The most trips a loop is unrolled to (a layer stack's depth, not a sweep).
pub const UNROLL_MAX_TRIPS: i64 = 128;

/// The most morphisms the unrolled trips may add (trips × body).
pub const UNROLL_MAX_MORPHISMS: usize = 1 << 16;

/// Analyze every counted loop for R-UN.
pub fn analyze_unroll(ir: &CategoryIr) -> RewritePlan {
    let mut plan = RewritePlan::new();
    for (f, _) in ir.funcs() {
        for ts in ir.trip_split(f) {
            let trips = ts.hi - ts.lo;
            if !(1..=UNROLL_MAX_TRIPS).contains(&trips) {
                continue;
            }
            let Some(lp) = ir.loop_plan(f, ts.merge) else {
                continue;
            };
            let Some(merge) = ir.object(ts.merge) else {
                continue;
            };
            if ty_contains_token(&merge.ty) {
                continue;
            }
            let scc: SecondaryMap<ObjectId, ()> = lp.scc_objects.iter().map(|&o| (o, ())).collect();
            let body: Vec<_> = lp
                .decide_order
                .iter()
                .chain(&lp.advance_order)
                .copied()
                .collect();
            let mut heavy = false;
            let mut view = false;
            for &m in &body {
                let Some(morph) = ir.morphism(m) else {
                    continue;
                };
                match morph.op {
                    Operation::Map { captures, .. } | Operation::Fold { captures, .. } => {
                        heavy = true;
                        view |= captured_records(ir, &scc, morph.source, captures);
                    }
                    Operation::Call(_) => heavy = true,
                    Operation::Index => view |= per_trip_view(ir, &scc, morph.source),
                    _ => {}
                }
            }
            if !heavy || !view {
                continue;
            }
            if (trips as usize).saturating_mul(body.len()) > UNROLL_MAX_MORPHISMS {
                continue;
            }
            plan.unroll.insert(ts.merge, trips as u32);
        }
    }
    plan
}

/// `Index`'s source `(array, index)`: the array made outside the loop, the
/// index in it, and the element a record (struct or tuple) holding arrays —
/// a layer's weights. A row of an array of arrays (a sequence's step, a
/// matrix's row) is data the loop walks, and stays a loop.
fn per_trip_view(ir: &CategoryIr, scc: &SecondaryMap<ObjectId, ()>, source: ObjectId) -> bool {
    let feeders = pair_feeders(ir, source);
    let [Some(array), Some(index)] = feeders[..] else {
        return false;
    };
    let Some(Ty::Array { elem, .. }) = ir.object(array).map(|o| &o.ty) else {
        return false;
    };
    let record = matches!(**elem, Ty::Struct { .. } | Ty::Tuple(_));
    !scc.contains_key(array) && scc.contains_key(index) && record && holds_array(elem)
}

/// A map / fold of the trip capturing an array of records made outside the
/// loop beside an integer of the trip (the counter): its body takes the
/// trip's record there (`stack[layer].weight[i]` inside the map).
fn captured_records(
    ir: &CategoryIr,
    scc: &SecondaryMap<ObjectId, ()>,
    source: ObjectId,
    captures: u32,
) -> bool {
    let caps: Vec<ObjectId> = ir
        .in_edges(source)
        .iter()
        .filter_map(|&m| {
            let morph = ir.morphism(m)?;
            matches!(morph.op, Operation::Pair { slot, .. } if slot < captures)
                .then_some(morph.source)
        })
        .collect();
    let records = caps.iter().any(|&c| {
        !scc.contains_key(c)
            && matches!(ir.object(c).map(|o| &o.ty), Some(Ty::Array { elem, .. })
                if matches!(**elem, Ty::Struct { .. } | Ty::Tuple(_)) && holds_array(elem))
    });
    let trip = caps.iter().any(|&c| {
        scc.contains_key(c) && matches!(ir.object(c).map(|o| &o.ty), Some(Ty::Int { .. }))
    });
    records && trip
}

fn holds_array(ty: &Ty) -> bool {
    match ty {
        Ty::Array { .. } => true,
        Ty::Tuple(ts) => ts.iter().any(holds_array),
        Ty::Struct { fields, .. } => fields.iter().any(|(_, t)| holds_array(t)),
        _ => false,
    }
}

/// The two slot feeders of a `Pair`-built pair, by slot.
fn pair_feeders(ir: &CategoryIr, product: ObjectId) -> [Option<ObjectId>; 2] {
    let mut out = [None, None];
    for &m in ir.in_edges(product) {
        if let Some(morph) = ir.morphism(m)
            && let Operation::Pair { slot, .. } = morph.op
            && (slot as usize) < 2
        {
            out[slot as usize] = Some(morph.source);
        }
    }
    out
}
