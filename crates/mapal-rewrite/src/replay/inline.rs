//! Inlining a call site: the callee body and its Return redirect.

use super::*;

/// Inline one planned `Call` edge: replay the callee's body verbatim into the
/// caller's builder with the callee's Parameter mapped to the call's (already
/// replayed) source object and the callee's Return writers redirected per
/// `ret_dest`. The callee's objects get fresh ids in builder emission order
/// (L2: same graph → same inlined graph); the callee's own planned Calls
/// recurse through [`reconstruct_temp`]. Returns the object holding the
/// callee's Return value (meaningful only for [`RetDest::Fresh`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn inline_call(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    m: MorphismId,
    ret_dest: RetDest,
) -> ObjectId {
    let morph = ir.morphism(m).expect("inline call edge");
    let Operation::Call(g) = morph.op else {
        unreachable!("inline: plan channel keys a non-Call edge")
    };
    let def = ir.func(g).expect("inline callee");
    let ret_obj = def.output;
    let obj_order = object_topo_order(ir, g);

    // A callee-local id map (the caller's map is read only for the call
    // source): object ids are unique graph-wide and each inlined copy is fully
    // wired before returning, so one local map per site is unambiguous — a
    // diamond-shared callee simply gets one fresh copy per site (the
    // documented duplication policy).
    let mut local: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();
    let arg = feed(plan, remap, morph.source);
    local.insert(def.input, arg);
    let mut done: SecondaryMap<ObjectId, bool> = SecondaryMap::new();

    // Phase A — body objects, exactly as `replay_fn` (loop quartets included).
    for &o in &obj_order {
        if done.get(o).copied().unwrap_or(false) {
            continue;
        }
        reconstruct_a(
            fb, ir, plan, fmap, fused, lifted, &mut local, &mut done, &obj_order, o, ret_obj,
            &ret_dest,
        );
    }

    // Phase B — the callee's Return writers, redirected.
    inline_return(
        fb, ir, plan, fmap, fused, lifted, &local, m, ret_obj, ret_dest, arg,
    )
}

/// Phase B of an inlined call: wire the callee's Return writers per `ret_dest`
/// ([`RetDest::Own`]: they replay as the enclosing fn's own Return writes;
/// [`RetDest::Fresh`]: the Return value is captured in one fresh object).
/// Returns the object holding the Return value (`Fresh`), or `arg` as a dummy
/// (`Own` — the writes already landed on the enclosing fn's Return).
#[allow(clippy::too_many_arguments)]
pub(super) fn inline_return(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    local: &SecondaryMap<ObjectId, ObjectId>,
    call: MorphismId,
    ret_obj: ObjectId,
    ret_dest: RetDest,
    arg: ObjectId,
) -> ObjectId {
    let writers = ir.in_edges(ret_obj);
    let call_loc = ir.morphism(call).expect("call").loc;

    // A slot-wise return (a product written by `Pair` slot writes).
    if !writers.is_empty()
        && writers
            .iter()
            .all(|&w| matches!(ir.morphism(w).unwrap().op, Operation::Pair { .. }))
    {
        let comps: Vec<ObjectId> = slot_feeders(ir, ret_obj)
            .into_iter()
            .map(|s| feed(plan, local, s))
            .collect();
        return match ret_dest {
            RetDest::Fresh(name) => {
                let ty = &ir.object(ret_obj).expect("ret").ty;
                match ty {
                    mapal_ir::Ty::Tuple(_) => fb.pack(&comps, Dest::Fresh(name), call_loc),
                    mapal_ir::Ty::Struct { .. } => {
                        fb.pack_struct(ty.clone(), &comps, Dest::Fresh(name), call_loc)
                    }
                    mapal_ir::Ty::Array { .. } => {
                        fb.pack_array(&comps, Dest::Fresh(name), call_loc)
                    }
                    _ => unreachable!("inline: slot-written Return with non-product ty"),
                }
                .expect("inline: pack")
            }
            RetDest::Own => {
                for &w in writers {
                    let mo = ir.morphism(w).expect("ret writer");
                    let Operation::Pair { slot, .. } = mo.op else {
                        unreachable!("inline: mixed Return writers")
                    };
                    fb.output(feed(plan, local, mo.source), Some(slot), mo.loc)
                        .expect("inline: output slot");
                }
                arg
            }
        };
    }

    // The canonical single full-value writer (Output / primitive / LoopExit).
    assert!(
        writers.len() == 1,
        "inline: callee Return has a non-canonical writer set"
    );
    let mo = ir.morphism(writers[0]).expect("ret writer");
    match mo.op {
        // Emitted by the loop quartet with the redirect dest; the Fresh case
        // recorded the exit object as the Return value (`reconstruct_loop`).
        Operation::LoopExit => match ret_dest {
            RetDest::Fresh(_) => local[ret_obj],
            RetDest::Own => arg,
        },
        Operation::Output => match ret_dest {
            RetDest::Fresh(_) => feed(plan, local, mo.source),
            RetDest::Own => {
                fb.output(feed(plan, local, mo.source), None, mo.loc)
                    .expect("inline: output");
                arg
            }
        },
        // A callee whose Return is written by a call that is itself inlined
        // (`fn f(x) { x -> g -> ret; }`): substitute that callee too, with
        // the same destination — as `replay_fn` does for its own Return
        // writers. Emitting it as a `Call` would name a function the plan
        // dropped.
        Operation::Call(_) if plan.inline.contains_key(writers[0]) => inline_call(
            fb, ir, plan, fmap, fused, lifted, local, writers[0], ret_dest,
        ),
        op => {
            let dest = match ret_dest {
                RetDest::Fresh(name) => Dest::Fresh(name),
                RetDest::Own => Dest::Ret { slot: None },
            };
            // A Return written by a fused `Map{g}` (DESIGN §4): `Map{h}` on the
            // original array.
            if let Some(&h) = fused.get(writers[0]) {
                return emit_fused_map(fb, ir, plan, local, h, writers[0], dest, mo.loc);
            }
            emit_op(
                fb, ir, plan, fmap, local, op, mo.source, mo.target, dest, mo.loc,
            )
        }
    }
}

// --- recipe classification helpers ----------------------------------------
