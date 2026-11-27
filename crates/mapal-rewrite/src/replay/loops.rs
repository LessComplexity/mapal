//! Loop reconstruction: canonical, unrolled, lifted and flat-state loops.

use super::*;

/// Reconstruct one loop as a unit (DESIGN §1.1 quartet). Canonical shapes only —
/// [`is_canonical`] has already gated `replay`, so this loop's SCC has exactly
/// one merge, one back, one exit.
#[allow(clippy::too_many_arguments)]
pub(super) fn reconstruct_loop(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    done: &mut SecondaryMap<ObjectId, bool>,
    obj_order: &[ObjectId],
    merge: ObjectId,
    ret_obj: ObjectId,
    ret_dest: &RetDest,
) {
    if let Some(&trips) = plan.unroll.get(merge) {
        reconstruct_unrolled_loop(
            fb, ir, plan, fmap, fused, lifted, remap, done, obj_order, merge, trips, ret_obj,
            ret_dest,
        );
        return;
    }
    if let Some(spec) = plan.lift.get(merge) {
        reconstruct_lifted_loop(
            fb,
            ir,
            plan,
            lifted[merge],
            spec,
            remap,
            done,
            merge,
            ret_obj,
            ret_dest,
        );
        return;
    }

    let merge_obj = ir.object(merge).expect("merge");

    // One source of truth for the loop's attribution (DESIGN §3, BL7 —
    // mapal-ir's loop_plan). `replay` is gated by `is_canonical`, so `Some` here.
    let lp = ir
        .loop_plan(ir.owner(merge), merge)
        .expect("canonical loop (is_canonical gated)");

    // Route payloads can leave the SCC (for example a token/value exit pack).
    // Rebuild the complete canonical driver region before wiring its routes.
    let body: SecondaryMap<ObjectId, bool> = {
        let mut members = SecondaryMap::new();
        for &o in &lp.scc_objects {
            members.insert(o, true);
        }
        for &m in lp.decide_order.iter().chain(&lp.advance_order) {
            members.insert(ir.morphism(m).expect("loop body edge").target, true);
        }
        members
    };

    // init → begin_loop. The init is loop-invariant (topo defers LoopEnter), so
    // it is already mapped.
    let init = lp.init;
    // The state's leaves when a slot is itself a product (`mut kv: KvCache`):
    // the loop carries the leaves (scalar replacement of the state), each
    // product-typed view rebuilt as a pack of leaf views (`flat_views`).
    let leaves = flat_state(ir, merge);
    let init_new = match &leaves {
        Some(ls) => flat_pack(fb, ir, plan, remap, init, ls, merge_obj.loc),
        None => feed(plan, remap, init),
    };
    let lh = fb
        .begin_loop(init_new, merge_obj.loc)
        .expect("replay: begin_loop");
    let merge_new = fb.merge_of(&lh);
    match &leaves {
        Some(ls) => flat_views(fb, ir, merge, merge_new, ls, remap, done),
        None => {
            remap.insert(merge, merge_new);
        }
    }
    done.insert(merge, true);

    // Body objects, in the global topo suborder of the canonical driver region.
    for &o in obj_order {
        if o == merge || done.get(o).copied().unwrap_or(false) {
            continue;
        }
        if !body.contains_key(o) {
            continue;
        }
        reconstruct_a(
            fb, ir, plan, fmap, fused, lifted, remap, done, obj_order, o, ret_obj, ret_dest,
        );
    }

    // Back edge: the LoopBack route packs (next_state, cond).
    let back_route = lp.back_route;
    let bf = slot_feeders(ir, back_route);
    let back_loc = back_morph_loc(ir, merge);
    let next = match &leaves {
        Some(ls) => flat_pack(fb, ir, plan, remap, bf[0], ls, back_loc),
        None => feed(plan, remap, bf[0]),
    };
    fb.loop_back(&lh, next, feed(plan, remap, bf[1]), back_loc)
        .expect("replay: loop_back");

    // Exit edge: the single attributed LoopExit, leaving the SCC.
    let exit_route = lp.exit_route;
    let exit_morph = ir.morphism(lp.exits[0]).expect("loop exit");
    let (exit_tgt, exit_loc) = (exit_morph.target, exit_morph.loc);
    let ef = slot_feeders(ir, exit_route);
    let tgt_obj = ir.object(exit_tgt).expect("exit target");
    let dest = if exit_tgt == ret_obj {
        match ret_dest {
            RetDest::Own => Dest::Ret { slot: None },
            // Inlined: the exit mints the call target's object instead of
            // writing the callee's (nonexistent) Return.
            RetDest::Fresh(name) => Dest::Fresh(name.clone()),
        }
    } else {
        Dest::Fresh(tgt_obj.name.clone())
    };
    let exit_new = fb
        .loop_exit(
            &lh,
            feed(plan, remap, ef[0]),
            feed(plan, remap, ef[1]),
            dest,
            exit_loc,
        )
        .expect("replay: loop_exit");
    if exit_tgt == ret_obj {
        // Inlined: record the exit object as the callee's Return value —
        // `inline_return` picks it up.
        if matches!(ret_dest, RetDest::Fresh(_)) {
            remap.insert(ret_obj, exit_new);
        }
    } else {
        remap.insert(exit_tgt, exit_new);
        done.insert(exit_tgt, true);
    }

    fb.end_loop(lh).expect("replay: end_loop");
}

/// R-UN ([`crate::unroll`]): a counted loop replayed as its `trips` trips, one
/// after another. Trip `k` replays the decide and advance phases with the
/// merge standing for trip `k`'s state (the init for the first); its back
/// route's next state is trip `k + 1`'s. After the last trip the decide phase
/// runs once more on the final state and its exit route's value is the
/// loop's exit object — what the loop leaves with. No loop is built: every
/// trip's counter is a constant once constant folding runs.
#[allow(clippy::too_many_arguments)]
pub(super) fn reconstruct_unrolled_loop(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    done: &mut SecondaryMap<ObjectId, bool>,
    obj_order: &[ObjectId],
    merge: ObjectId,
    trips: u32,
    ret_obj: ObjectId,
    ret_dest: &RetDest,
) {
    let lp = ir
        .loop_plan(ir.owner(merge), merge)
        .expect("canonical loop (is_canonical gated)");
    let targets = |order: &[MorphismId]| -> SecondaryMap<ObjectId, ()> {
        order
            .iter()
            .map(|&m| (ir.morphism(m).expect("loop body edge").target, ()))
            .collect()
    };
    let scc: SecondaryMap<ObjectId, ()> = lp.scc_objects.iter().map(|&o| (o, ())).collect();
    let mut every: SecondaryMap<ObjectId, ()> = targets(&lp.decide_order);
    for (o, _) in targets(&lp.advance_order).iter() {
        every.insert(o, ());
    }
    for (o, _) in scc.iter() {
        every.insert(o, ());
    }
    let decide = targets(&lp.decide_order);
    // One pass over the body in the global topo suborder, the merge standing
    // for `state`; `only` limits it to the decide phase (the final pass).
    let pass = |fb: &mut mapal_ir::FnBuilder<'_>,
                remap: &mut SecondaryMap<ObjectId, ObjectId>,
                done: &mut SecondaryMap<ObjectId, bool>,
                state: ObjectId,
                only: Option<&SecondaryMap<ObjectId, ()>>| {
        for (o, _) in every.iter() {
            done.remove(o);
        }
        remap.insert(merge, state);
        done.insert(merge, true);
        for &o in obj_order {
            if o == merge || !every.contains_key(o) || only.is_some_and(|d| !d.contains_key(o)) {
                continue;
            }
            if done.get(o).copied().unwrap_or(false) {
                continue;
            }
            reconstruct_a(
                fb, ir, plan, fmap, fused, lifted, remap, done, obj_order, o, ret_obj, ret_dest,
            );
        }
    };
    let mut state = feed(plan, remap, lp.init);
    let back = slot_feeders(ir, lp.back_route);
    for _ in 0..trips {
        pass(fb, remap, done, state, None);
        state = feed(plan, remap, back[0]);
    }
    pass(fb, remap, done, state, Some(&decide));
    let exit = slot_feeders(ir, lp.exit_route);
    let value = feed(plan, remap, exit[0]);
    for (o, _) in every.iter() {
        done.insert(o, true);
    }
    let exit_morph = ir.morphism(lp.exits[0]).expect("loop exit");
    let exit_tgt = exit_morph.target;
    if exit_tgt == ret_obj {
        match ret_dest {
            RetDest::Own => fb
                .output(value, None, exit_morph.loc)
                .expect("replay: unrolled loop's exit value is the function's output"),
            RetDest::Fresh(_) => {
                remap.insert(ret_obj, value);
            }
        }
    } else {
        remap.insert(exit_tgt, value);
        done.insert(exit_tgt, true);
    }
}

/// Scalar replacement of a loop state: the leaf paths of a tuple state one of
/// whose slots is itself a product (a struct of arrays), when every use of the
/// merge is a slot view. `None`: the state is carried as it is.
pub(super) fn flat_state(ir: &CategoryIr, merge: ObjectId) -> Option<Vec<Vec<u32>>> {
    let ty = &ir.object(merge)?.ty;
    let mapal_ir::Ty::Tuple(parts) = ty else {
        return None;
    };
    let product =
        |t: &mapal_ir::Ty| matches!(t, mapal_ir::Ty::Tuple(_) | mapal_ir::Ty::Struct { .. });
    if !parts.iter().any(product) {
        return None;
    }
    let views = ir
        .out_edges(merge)
        .iter()
        .all(|&m| matches!(ir.morphism(m).map(|x| x.op), Some(Operation::Proj { .. })));
    if !views {
        return None;
    }
    fn walk(t: &mapal_ir::Ty, at: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        match t {
            mapal_ir::Ty::Tuple(_) | mapal_ir::Ty::Struct { .. } => {
                for k in 0..t.product_arity_u32().unwrap_or(0) {
                    at.push(k);
                    walk(t.component_ty(k).expect("component"), at, out);
                    at.pop();
                }
            }
            _ => out.push(at.clone()),
        }
    }
    let mut out = Vec::new();
    walk(ty, &mut Vec::new(), &mut out);
    Some(out)
}

/// `o` (a state value: the init, the next state) as the flat tuple of its
/// leaves: each leaf through the packs that build `o` in the input graph
/// (their slot feeders — such a pack is not replayed, [`flat_skip`]), then
/// `Proj`s of the first object no pack builds, one per path prefix (a linear
/// token leaf is taken once).
pub(super) fn flat_pack(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    o: ObjectId,
    leaves: &[Vec<u32>],
    loc: mapal_ir::SourceLoc,
) -> ObjectId {
    let mut made: Vec<(ObjectId, Vec<u32>, ObjectId)> = Vec::new();
    let mut comps = Vec::new();
    for p in leaves {
        let mut x = plan.resolve_alias(o);
        let mut rest: &[u32] = p;
        while let Some((&k, tail)) = rest.split_first() {
            if !is_pack_built(ir, x) {
                break;
            }
            x = plan.resolve_alias(slot_feeders(ir, x)[k as usize]);
            rest = tail;
        }
        let mut y = feed(plan, remap, x);
        for i in 0..rest.len() {
            let prefix = rest[..=i].to_vec();
            y = match made.iter().find(|(b, q, _)| *b == x && *q == prefix) {
                Some(&(_, _, z)) => z,
                None => {
                    let z = fb
                        .proj(y, rest[i], Dest::Fresh(None), loc)
                        .expect("replay: state leaf proj");
                    made.push((x, prefix, z));
                    z
                }
            };
        }
        comps.push(y);
    }
    fb.pack(&comps, Dest::Fresh(None), loc)
        .expect("replay: flat state pack")
}

/// Every in-edge of `o` is a `Pair`: a pack.
pub(super) fn is_pack_built(ir: &CategoryIr, o: ObjectId) -> bool {
    let ins = ir.in_edges(o);
    !ins.is_empty()
        && ins
            .iter()
            .all(|&m| matches!(ir.morphism(m).unwrap().op, Operation::Pair { .. }))
}

/// A pack whose one reader is a flattened loop's entry (its init) or back
/// route (its next state): [`flat_pack`] reads its feeders, so it is not
/// replayed (a token it holds is taken once).
pub(super) fn flat_skip(ir: &CategoryIr, o: ObjectId) -> bool {
    if !is_pack_built(ir, o) {
        return false;
    }
    let [m] = ir.out_edges(o) else {
        return false;
    };
    let mo = ir.morphism(*m).unwrap();
    match mo.op {
        Operation::LoopEnter => flat_state(ir, mo.target).is_some(),
        Operation::Pair { slot: 0, .. } => ir.out_edges(mo.target).iter().any(|&b| {
            let bm = ir.morphism(b).unwrap();
            bm.op == Operation::LoopBack && flat_state(ir, bm.target).is_some()
        }),
        _ => false,
    }
}

/// Each slot view of the flattened `merge`: a leaf slot's view is the new
/// merge's leaf view; a product slot's is the pack (struct or tuple) of its
/// leaves' views, which `proj∘pack` then forwards to the leaves.
pub(super) fn flat_views(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    merge: ObjectId,
    merge_new: ObjectId,
    leaves: &[Vec<u32>],
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    done: &mut SecondaryMap<ObjectId, bool>,
) {
    fn build(
        fb: &mut mapal_ir::FnBuilder<'_>,
        merge_new: ObjectId,
        leaves: &[Vec<u32>],
        at: &mut Vec<u32>,
        ty: &mapal_ir::Ty,
        dest: Dest,
        loc: mapal_ir::SourceLoc,
    ) -> ObjectId {
        if let Some(i) = leaves.iter().position(|p| p == at) {
            return fb
                .proj(merge_new, i as u32, dest, loc)
                .expect("replay: flat leaf view");
        }
        let mut comps = Vec::new();
        for k in 0..ty.product_arity_u32().unwrap_or(0) {
            at.push(k);
            let t = ty.component_ty(k).expect("component").clone();
            comps.push(build(fb, merge_new, leaves, at, &t, Dest::Fresh(None), loc));
            at.pop();
        }
        match ty {
            mapal_ir::Ty::Struct { .. } => fb.pack_struct(ty.clone(), &comps, dest, loc),
            _ => fb.pack(&comps, dest, loc),
        }
        .expect("replay: flat product view")
    }
    for &m in ir.out_edges(merge) {
        let mo = ir.morphism(m).expect("merge view");
        let Operation::Proj { index } = mo.op else {
            continue;
        };
        let v = ir.object(mo.target).expect("view");
        let new = build(
            fb,
            merge_new,
            leaves,
            &mut vec![index],
            &v.ty,
            Dest::Fresh(v.name.clone()),
            v.loc,
        );
        remap.insert(mo.target, new);
        done.insert(mo.target, true);
    }
}

/// Replace one planned loop SCC with `const K → Iota(K) → Map/Fold`, wiring the
/// collection result to the old `LoopExit` target.
#[allow(clippy::too_many_arguments)]
pub(super) fn reconstruct_lifted_loop(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    body: FuncId,
    spec: &LiftSpec,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    done: &mut SecondaryMap<ObjectId, bool>,
    merge: ObjectId,
    ret_obj: ObjectId,
    ret_dest: &RetDest,
) {
    let lp = ir
        .loop_plan(ir.owner(merge), merge)
        .expect("lift plan keys a canonical loop");
    let exit = ir.morphism(lp.exits[0]).expect("lifted loop exit");
    let target = exit.target;
    let target_obj = ir.object(target).expect("lifted loop exit target");
    let dest = if target == ret_obj {
        match ret_dest {
            RetDest::Own => Dest::Ret { slot: None },
            RetDest::Fresh(name) => Dest::Fresh(name.clone()),
        }
    } else {
        Dest::Fresh(target_obj.name.clone())
    };

    let count = fb
        .constant(Value::I32(spec.count), exit.loc)
        .expect("lift: count constant");
    let items = fb
        .iota(count, Dest::Fresh(None), exit.loc)
        .expect("lift: iota");
    let captures: Vec<_> = spec
        .captures
        .iter()
        .map(|&o| feed(plan, remap, o))
        .collect();
    let result = match spec.kind {
        LiftKind::Fold { seed, .. } => fb
            .fold_captured(
                body,
                &captures,
                feed(plan, remap, seed),
                items,
                dest,
                exit.loc,
            )
            .expect("lift: fold_captured"),
        LiftKind::Map => fb
            .map_captured(body, &captures, items, dest, exit.loc)
            .expect("lift: map_captured"),
    };

    for &o in &lp.scc_objects {
        done.insert(o, true);
    }
    done.insert(lp.back_route, true);
    done.insert(lp.exit_route, true);
    done.insert(merge, true);
    if target == ret_obj {
        if matches!(ret_dest, RetDest::Fresh(_)) {
            remap.insert(ret_obj, result);
        }
    } else {
        remap.insert(target, result);
        done.insert(target, true);
    }
}
