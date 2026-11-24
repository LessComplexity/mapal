//! ADR-0048 D3 — compile-time evaluation. Every function's
//! [`mapal_ir::CategoryIr::static_plan`] roots that its rule realizes as
//! values ([`mapal_ir::Realize::Embed`], or [`mapal_ir::Realize::Attempt`]
//! when the oracle comes out within the budget) are computed by the
//! interpreter — the oracle, so the value is exact on every face — and
//! replace their cones: a scalar becomes a constant (`constify`), a product
//! a pack of constants (`embed`). A root whose evaluation traps or runs out of
//! budget stays computed at run time (a trap is the program's meaning).
//! Dead cones fall to the default pipeline's DCE.

use mapal_interp::RValue;
use mapal_ir::{CategoryIr, EVAL_WORK, ObjectId, Realize};

use crate::plan::RewritePlan;
use crate::replay::replay;

/// What compile-time evaluation did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StaticReport {
    /// Roots replaced by their value.
    pub embedded: u64,
    /// Roots the rule leaves computed at run time.
    pub computed: u64,
    /// Roots evaluated that trapped or ran out of budget (left at run time).
    pub failed: u64,
}

/// Replace every root `static_plan` realizes as a value by its value (ADR-0048
/// D3). The identity (no rebuild) when there is none.
pub fn fold_static(ir: CategoryIr) -> (CategoryIr, StaticReport) {
    let mut report = StaticReport::default();
    let mut plan = RewritePlan::new();
    for (f, _) in ir.funcs() {
        let sp = ir.static_plan(f);
        report.computed += sp
            .roots
            .iter()
            .filter(|r| r.realize == Realize::Compute)
            .count() as u64;
        let roots: Vec<ObjectId> = sp.evaluated().map(|r| r.value).collect();
        if roots.is_empty() {
            continue;
        }
        let values = mapal_interp::eval_values(&ir, f, &roots, EVAL_WORK);
        let mut done: Vec<ObjectId> = Vec::new();
        for (o, v) in roots.into_iter().zip(values) {
            match v {
                Some(RValue::Scalar(x)) => {
                    plan.constify.insert(o, x);
                    report.embedded += 1;
                    done.push(o);
                }
                Some(v @ (RValue::Array(_) | RValue::Tuple(_) | RValue::Struct { .. })) => {
                    plan.embed.insert(o, v);
                    report.embedded += 1;
                    done.push(o);
                }
                _ => report.failed += 1,
            }
        }
        drop_cones(&ir, f, &done, &mut plan);
    }
    if plan.is_empty() {
        return (ir, report);
    }
    match replay(&ir, &plan) {
        Ok(out) => (out, report),
        // A non-canonical loop shape: every rewrite is the identity (R6).
        Err(_) => (ir, StaticReport::default()),
    }
}

/// The cones of the embedded roots `done`, evaluated without a trap, that
/// nothing else reads: dropped (their may-trap work — an index, a division —
/// is proven not to trap, which DCE alone could not assume). Loop members
/// stay (replay rebuilds a loop as a unit; DCE drops a dead one).
fn drop_cones(ir: &CategoryIr, f: mapal_ir::FuncId, done: &[ObjectId], plan: &mut RewritePlan) {
    use slotmap::SecondaryMap;
    let mut in_loop: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    for scc in ir.loop_structure(f) {
        for o in scc.objects {
            in_loop.insert(o, ());
        }
    }
    let output = ir.func(f).map(|d| d.output);
    // Candidates: the cones' temporaries.
    let mut cone: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    let mut stack: Vec<ObjectId> = Vec::new();
    for &r in done {
        for &m in ir.in_edges(r) {
            if let Some(mm) = ir.morphism(m) {
                stack.push(mm.source);
            }
        }
    }
    while let Some(o) = stack.pop() {
        let Some(obj) = ir.object(o) else { continue };
        if obj.kind != mapal_ir::ObjectKind::Temporary
            || in_loop.contains_key(o)
            || Some(o) == output
            || cone.insert(o, ()).is_some()
        {
            continue;
        }
        for &m in ir.in_edges(o) {
            if let Some(mm) = ir.morphism(m) {
                stack.push(mm.source);
            }
        }
    }
    // Keep a candidate any other value still reads (and, through it, its
    // sources): to a fixpoint.
    let dead = |o: ObjectId, cone: &SecondaryMap<ObjectId, ()>| {
        ir.out_edges(o).iter().all(|&m| {
            ir.morphism(m).is_some_and(|mm| {
                cone.contains_key(mm.target)
                    || plan.constify.contains_key(mm.target)
                    || plan.embed.contains_key(mm.target)
            })
        })
    };
    loop {
        let live: Vec<ObjectId> = cone.keys().filter(|&o| !dead(o, &cone)).collect();
        if live.is_empty() {
            break;
        }
        for o in live {
            cone.remove(o);
        }
    }
    for (o, ()) in cone {
        if !plan.constify.contains_key(o) && !plan.embed.contains_key(o) {
            plan.drop.insert(o, ());
        }
    }
}
