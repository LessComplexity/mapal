//! The shared, plan-driven replayer (DESIGN §1.1). One graph constructor for
//! every pass: it walks each function's `topo_order`, classifies every object by
//! its *recipe*, and rebuilds an equivalent object through the public
//! [`IrBuilder`]. Well-formedness of the output is by construction; `validate()`
//! is a redundant independent check (R2).
//!
//! Identity replay — `replay(ir, &RewritePlan::new())` — is the WP1 soundness
//! anchor: it must reproduce an interp-equal, validate-clean graph on the 10
//! in-Core examples.
//!
//! **Recipe classification** (DESIGN §1.1 table): composite builder primitives
//! (`binop`, `phi`, `index`, `zip`, `print`, loop routes) mint their internal
//! `Pair` products atomically, so those products are *not* re-materialized — the
//! primitive call rebuilds them from the slot feeders. A product with any
//! explicit consumer (or ≥2 consumers) *is* materialized via `pack`.

use slotmap::SecondaryMap;

use mapal_ir::{
    CategoryIr, Dest, FuncId, FuncKind, IrBuilder, MorphismId, ObjectId, ObjectKind, Operation, Ty,
    Value,
};

use crate::plan::{FusionSpec, LiftKind, LiftSpec, RewritePlan};

mod emit;
mod inline;
mod loops;
mod support;
mod synth;

use self::emit::*;
use self::inline::*;
use self::loops::*;
pub use self::support::is_canonical;
use self::support::*;
use self::synth::*;

/// Why replay declined to rebuild (DESIGN §1.1 / R6). The only user-relevant
/// classification failure; the driver turns it into the whole-graph identity
/// path (`skipped_non_canonical`). Builder rejections during a well-formed
/// rebuild are internal bugs and panic instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// A function contains a loop shape outside the canonical quartet
    /// (multi-merge SCC, multiple backs, or multiple exits per merge).
    NonCanonicalLoop,
}

/// Where a replayed body's Return value lands (DESIGN §1.1 + the `inline`
/// pass). Normal fn replay writes the replayed fn's own Return ([`RetDest::Own`]);
/// an inlined call site redirects the callee's Return writers to the call
/// target's fresh object.
enum RetDest {
    /// Return writers target the enclosing fn's own Return object.
    Own,
    /// An inlined call whose target is a `Temporary`: the callee's Return
    /// value defines one fresh object under this name, handed back to the
    /// caller for the `remap` insertion.
    Fresh(Option<String>),
}

/// Rebuild `ir` under `plan` (DESIGN §1.1). Returns a fresh sealed graph, or
/// [`ReplayError::NonCanonicalLoop`] if any function's loop shape is outside the
/// interp-M1 canonical quartet (R6 — the driver takes the whole-graph identity
/// path).
pub fn replay(ir: &CategoryIr, plan: &RewritePlan) -> Result<CategoryIr, ReplayError> {
    if !is_canonical(ir) {
        return Err(ReplayError::NonCanonicalLoop);
    }
    debug_assert!(
        plan.is_consistent(ir),
        "replay: plan violates §1.2 P1/P2 consistency"
    );

    let mut b = IrBuilder::new();

    // Function-level DCE (§1.1/§3.1): only functions reachable from the entry
    // over the *post-plan* reference graph survive. A fused pair drops the two
    // original bodies (inlined into a synthesized `h`) when nothing else refers
    // to them; uncalled Named fns fall out too (RW7). Sound — the oracle never
    // evaluates unreferenced functions.
    let live = live_functions(ir, plan);

    // Declare every live function first (declare-before-reference: Call/Map/Fold
    // may target a fn defined later). Preserve kind/name/tys/loc.
    let mut fmap: SecondaryMap<FuncId, FuncId> = SecondaryMap::new();
    for (fid, def) in ir.funcs() {
        if !live.contains_key(fid) {
            continue;
        }
        let in_ty = ir.object(def.input).expect("fn input").ty.clone();
        let out_ty = ir.object(def.output).expect("fn output").ty.clone();
        let nf = b
            .declare(def.kind, &def.name, in_ty, out_ty, def.loc)
            .expect("replay: declare rejected a well-formed function");
        fmap.insert(fid, nf);
    }

    // Synthesize one fused `MapBody` per live fusion (DESIGN §4), built while no
    // `FnBuilder` is active. Keyed by the g-edge so replay swaps `Map{g}` for
    // `Map{h}` on the original array.
    let mut fused: SecondaryMap<MorphismId, FuncId> = SecondaryMap::new();
    let mut fused_n = 0u32;
    for (m_g, spec) in plan.fuse.iter() {
        let container = ir.owner(ir.morphism(m_g).expect("fused g-edge").source);
        if !live.contains_key(container) {
            continue;
        }
        let h = synthesize_fused(&mut b, ir, &fmap, m_g, spec, &mut fused_n);
        fused.insert(m_g, h);
    }

    // Synthesize one collection body per lifted loop before any enclosing
    // function is built; the replacement Map/Fold then references it exactly
    // like a source-authored body.
    let mut lifted: SecondaryMap<ObjectId, FuncId> = SecondaryMap::new();
    let mut lifted_n = 0u32;
    for (merge, spec) in plan.lift.iter() {
        let container = ir.owner(merge);
        if !live.contains_key(container) {
            continue;
        }
        let body = synthesize_lifted_body(&mut b, ir, &fmap, merge, spec, &mut lifted_n);
        lifted.insert(merge, body);
    }

    // A single global old→new object map: object ids are unique graph-wide and
    // every edge stays within one function (I6), so one map is unambiguous.
    let mut remap: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();

    for (fid, _) in ir.funcs() {
        if !live.contains_key(fid) {
            continue;
        }
        replay_fn(&mut b, ir, plan, fid, &fmap, &fused, &lifted, &mut remap);
    }

    let entry = fmap[ir.entry()];
    Ok(b.seal(entry)
        .expect("replay: seal rejected a builder-built graph")
        .with_data_dir(ir.data_dir().map(str::to_string)))
}

/// Rebuild one function's body (DESIGN §1.1). Phase A reconstructs body objects
/// in topo order; Phase B wires the Return writers.
#[allow(clippy::too_many_arguments)]
fn replay_fn(
    b: &mut IrBuilder,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fid: FuncId,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
) {
    let def = ir.func(fid).expect("fn def");
    let ret_obj = def.output;
    let obj_order = object_topo_order(ir, fid);

    let mut fb = b
        .build_fn(fmap[fid])
        .expect("replay: build_fn rejected a declared function");

    // The input Parameter maps to the builder's own input object.
    remap.insert(def.input, fb.input());

    let mut done: SecondaryMap<ObjectId, bool> = SecondaryMap::new();

    // Phase A — body objects (constants, materialized products, op temporaries,
    // loop quartets). The Return object and its direct-writer primitives are
    // deferred to Phase B.
    for &o in &obj_order {
        if done.get(o).copied().unwrap_or(false) {
            continue;
        }
        reconstruct_a(
            &mut fb,
            ir,
            plan,
            fmap,
            fused,
            lifted,
            remap,
            &mut done,
            &obj_order,
            o,
            ret_obj,
            &RetDest::Own,
        );
    }

    // Phase B — Return writers (DESIGN §1.1: `Dest::Ret` canonical form).
    for &m in ir.in_edges(ret_obj) {
        reconstruct_return_writer(&mut fb, ir, plan, fmap, fused, lifted, remap, m);
    }

    fb.finish()
        .expect("replay: finish rejected a rebuilt function");
}

/// Phase A dispatch for one object.
#[allow(clippy::too_many_arguments)]
fn reconstruct_a(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    done: &mut SecondaryMap<ObjectId, bool>,
    obj_order: &[ObjectId],
    o: ObjectId,
    ret_obj: ObjectId,
    ret_dest: &RetDest,
) {
    if o == ret_obj {
        done.insert(o, true);
        return;
    }
    let obj = ir.object(o).expect("object");
    match obj.kind {
        ObjectKind::Parameter => {
            // Pre-mapped to fb.input().
            done.insert(o, true);
        }
        ObjectKind::Constant => {
            let v = obj.value.clone().expect("constant value");
            let new = fb.constant(v, obj.loc).expect("replay: constant");
            remap.insert(o, new);
            done.insert(o, true);
        }
        ObjectKind::LoopMerge if plan.drop_loops.contains_key(o) => {
            // ADR-0048 D3: a loop compile-time evaluation ran to its end that
            // nothing reads: not replayed (its read exits are constants).
            let lp = ir.loop_plan(ir.owner(o), o).expect("canonical loop");
            done.insert(o, true);
            for &x in &lp.scc_objects {
                done.insert(x, true);
            }
            for &m in lp
                .decide_order
                .iter()
                .chain(&lp.advance_order)
                .chain(&lp.exits)
            {
                let t = ir.morphism(m).expect("loop edge").target;
                if !plan.constify.contains_key(t) && !plan.embed.contains_key(t) {
                    done.insert(t, true);
                }
            }
        }
        ObjectKind::LoopMerge => {
            reconstruct_loop(
                fb, ir, plan, fmap, fused, lifted, remap, done, obj_order, o, ret_obj, ret_dest,
            );
        }
        ObjectKind::Return => {
            // Only reachable for a foreign Return; the owning one is ret_obj.
            done.insert(o, true);
        }
        ObjectKind::Temporary => {
            if plan.drop.contains_key(o) || plan.alias.contains_key(o) {
                done.insert(o, true);
                return;
            }
            if let Some(v) = plan.constify.get(o) {
                let new = fb.constant(v.clone(), obj.loc).expect("replay: constify");
                remap.insert(o, new);
                done.insert(o, true);
                return;
            }
            if let Some(v) = plan.embed.get(o) {
                let new = materialize(fb, &obj.ty, v, obj.loc).expect("replay: embed");
                remap.insert(o, new);
                done.insert(o, true);
                return;
            }
            if is_internal_pack(ir, o) || flat_skip(ir, o) {
                // Rebuilt internally by its consuming primitive (a flattened
                // loop's state: `flat_pack`); not materialized.
                done.insert(o, true);
                return;
            }
            reconstruct_temp(fb, ir, plan, fmap, fused, lifted, remap, o);
            done.insert(o, true);
        }
    }
}

/// ADR-0048 D3: a value of type `ty` as constants packed slot by slot.
fn materialize(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ty: &mapal_ir::Ty,
    v: &mapal_interp::RValue,
    loc: mapal_ir::SourceLoc,
) -> Result<ObjectId, mapal_ir::IrError> {
    use mapal_interp::RValue;
    use mapal_ir::{Dest, Ty};
    match (ty, v) {
        (_, RValue::Scalar(x)) => fb.constant(x.clone(), loc),
        (Ty::Array { elem, .. }, RValue::Array(es)) => {
            let comps = es
                .iter()
                .map(|e| materialize(fb, elem, e, loc))
                .collect::<Result<Vec<_>, _>>()?;
            fb.pack_array(&comps, Dest::Fresh(None), loc)
        }
        (Ty::Tuple(ts), RValue::Tuple(es)) => {
            let comps = ts
                .iter()
                .zip(es)
                .map(|(t, e)| materialize(fb, t, e, loc))
                .collect::<Result<Vec<_>, _>>()?;
            fb.pack(&comps, Dest::Fresh(None), loc)
        }
        (Ty::Struct { fields, .. }, RValue::Struct { fields: vs, .. }) => {
            let comps = fields
                .iter()
                .zip(vs)
                .map(|((_, t), (_, e))| materialize(fb, t, e, loc))
                .collect::<Result<Vec<_>, _>>()?;
            fb.pack_struct(ty.clone(), &comps, Dest::Fresh(None), loc)
        }
        _ => Err(mapal_ir::IrError::EmptyProduct),
    }
}

#[cfg(test)]
mod tests;
