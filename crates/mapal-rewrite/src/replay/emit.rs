//! Per-object and per-operation emission: temporaries, `emit_op`, Return writers.

use super::*;

/// Reconstruct a materialized-product or op-defined `Temporary` as a fresh
/// object (DESIGN §1.1). Names and locs preserved; `Dest::Fresh` throughout —
/// Return writes are handled separately (Phase B).
pub(super) fn reconstruct_temp(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &mut SecondaryMap<ObjectId, ObjectId>,
    o: ObjectId,
) {
    let obj = ir.object(o).expect("temp");
    let name = obj.name.clone();
    let dest = Dest::Fresh(name);
    let ins = ir.in_edges(o);

    // A product object: all in-edges are `Pair`. Rebuild via the matching packer
    // from its slot feeders (declaration order = slot order).
    if !ins.is_empty()
        && ins
            .iter()
            .all(|&m| matches!(ir.morphism(m).unwrap().op, Operation::Pair { .. }))
    {
        let comps: Vec<ObjectId> = slot_feeders(ir, o)
            .into_iter()
            .map(|s| feed(plan, remap, s))
            .collect();
        let new = match &obj.ty {
            mapal_ir::Ty::Tuple(_) => fb.pack(&comps, dest, obj.loc),
            mapal_ir::Ty::Struct { .. } => fb.pack_struct(obj.ty.clone(), &comps, dest, obj.loc),
            mapal_ir::Ty::Array { .. } => fb.pack_array(&comps, dest, obj.loc),
            _ => unreachable!("product object with non-product ty"),
        }
        .expect("replay: pack");
        remap.insert(o, new);
        return;
    }

    // A fused `Map{g}` edge (DESIGN §4): emit `Map{h}` on the ORIGINAL array.
    if let Some(&h) = fused.get(ins[0]) {
        let new = emit_fused_map(fb, ir, plan, remap, h, ins[0], dest, obj.loc);
        remap.insert(o, new);
        return;
    }

    // An inlined `Call` edge (region-emission plan Move 1): substitute the
    // callee's body; its Return value defines this object.
    if plan.inline.contains_key(ins[0]) {
        let new = inline_call(
            fb,
            ir,
            plan,
            fmap,
            fused,
            lifted,
            remap,
            ins[0],
            RetDest::Fresh(obj.name.clone()),
        );
        remap.insert(o, new);
        return;
    }

    // A single op edge defines this object.
    let m = ir.morphism(ins[0]).expect("op edge");
    let new = emit_op(
        fb, ir, plan, fmap, remap, m.op, m.source, m.target, dest, m.loc,
    );
    remap.insert(o, new);
}

/// Emit one op primitive with feeders resolved through the internal pack (for
/// packed-source ops) or directly (DESIGN §1.1). Returns the result object.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_op(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    op: Operation,
    src: ObjectId,
    target: ObjectId,
    dest: Dest,
    loc: mapal_ir::SourceLoc,
) -> ObjectId {
    use Operation::*;
    match op {
        Add | Sub | Mul | Div | Mod | Eq | Neq | Lt | Gt | Le | Ge | And | Or => {
            let f = slot_feeders(ir, src);
            let new = fb
                .binop(
                    op,
                    feed(plan, remap, f[0]),
                    feed(plan, remap, f[1]),
                    dest,
                    loc,
                )
                .expect("replay: binop");
            // ADR-0041 D1: a product keeps its admitted set wherever it lands.
            if let Some(&m) = ir.in_edges(target).first() {
                fb.admit(new, ir.precision(m));
            }
            new
        }
        Neg | Not => fb
            .unop(op, feed(plan, remap, src), dest, loc)
            .expect("replay: unop"),
        Math(f) => fb
            .math(f, feed(plan, remap, src), dest, loc)
            .expect("replay: math"),
        Widen => {
            let out_ty = ir.object(target).expect("replay: Widen target").ty.clone();
            fb.widen(feed(plan, remap, src), out_ty, dest, loc)
                .expect("replay: widen")
        }
        Narrow => {
            let mapal_ir::Ty::Narrow(n) = ir.object(target).expect("replay: Narrow target").ty
            else {
                unreachable!("replay: Narrow into a non-narrow ty")
            };
            fb.narrow(feed(plan, remap, src), n, dest, loc)
                .expect("replay: narrow")
        }
        Proj { index } => fb
            .proj(feed(plan, remap, src), index, dest, loc)
            .expect("replay: proj"),
        Index => {
            let f = slot_feeders(ir, src);
            fb.index(feed(plan, remap, f[0]), feed(plan, remap, f[1]), dest, loc)
                .expect("replay: index")
        }
        Phi => {
            let f = slot_feeders(ir, src);
            fb.phi(
                feed(plan, remap, f[0]),
                feed(plan, remap, f[1]),
                feed(plan, remap, f[2]),
                dest,
                loc,
            )
            .expect("replay: phi")
        }
        Zip => {
            let f = slot_feeders(ir, src);
            fb.zip(feed(plan, remap, f[0]), feed(plan, remap, f[1]), dest, loc)
                .expect("replay: zip")
        }
        Enumerate => fb
            .enumerate(feed(plan, remap, src), dest, loc)
            .expect("replay: enumerate"),
        Iota => fb
            .iota(feed(plan, remap, src), dest, loc)
            .expect("replay: iota"),
        Fill => fb
            // The EXISTING (already-replayed) internal tuple, never the sugar
            // `fill(x, n)` — the sugar mints a fresh tuple, which resurrects
            // every CSE'd duplicate and breaks the fixpoint (S21).
            .fill_from(feed(plan, remap, src), dest, loc)
            .expect("replay: fill"),
        Update => {
            let f = slot_feeders(ir, src);
            fb.update(
                feed(plan, remap, f[0]),
                feed(plan, remap, f[1]),
                feed(plan, remap, f[2]),
                dest,
                loc,
            )
            .expect("replay: update")
        }
        Call(g) => fb
            .call(fmap[g], feed(plan, remap, src), dest, loc)
            .expect("replay: call"),
        Map { body, captures: 0 } => fb
            .map(fmap[body], feed(plan, remap, src), dest, loc)
            .expect("replay: map"),
        Map { body, captures } => {
            // ADR-0027: the source is the product (c₁…cₖ, arr) — re-thread the
            // leading capture components, never silently rebuild as k=0.
            let f = slot_feeders(ir, src);
            let (caps, data) = split_capture_source(&f, captures as usize, 1, "Map");
            let caps: Vec<ObjectId> = caps.iter().map(|&c| feed(plan, remap, c)).collect();
            fb.map_captured(fmap[body], &caps, feed(plan, remap, data[0]), dest, loc)
                .expect("replay: map_captured")
        }
        Fold { body, captures: 0 } => fb
            .fold(fmap[body], feed(plan, remap, src), dest, loc)
            .expect("replay: fold"),
        Fold { body, captures } => {
            // ADR-0027: the source is (c₁…cₖ, acc, arr) — captures first, then
            // the historical (acc, arr) tail.
            let f = slot_feeders(ir, src);
            let (caps, data) = split_capture_source(&f, captures as usize, 2, "Fold");
            let caps: Vec<ObjectId> = caps.iter().map(|&c| feed(plan, remap, c)).collect();
            fb.fold_captured(
                fmap[body],
                &caps,
                feed(plan, remap, data[0]),
                feed(plan, remap, data[1]),
                dest,
                loc,
            )
            .expect("replay: fold_captured")
        }
        Print { newline } => {
            // `print`/`println` take (token, value) and always mint a fresh
            // IoToken result (no Dest); `dest` is unused here (Print never
            // targets Return directly).
            let f = slot_feeders(ir, src);
            let (tok, val) = (feed(plan, remap, f[0]), feed(plan, remap, f[1]));
            if newline {
                fb.println(tok, val, loc).expect("replay: println")
            } else {
                fb.print(tok, val, loc).expect("replay: print")
            }
        }
        TimeMs => {
            // `time` (plan-time-builtin) takes the bare token (no packed
            // source) and mints the fresh `(IoToken, f64)` pair; `dest` is
            // unused, exactly as for Print. No rule rewrites it — it is
            // effectful, so replay only has to rebuild it faithfully.
            fb.time_ms(feed(plan, remap, src), loc)
                .expect("replay: time_ms")
        }
        Read => {
            // ADR-0045: (token, path) → the fresh `(IoToken, array)` pair,
            // the array type the original target's. Effectful: rebuilt as is.
            let f = slot_feeders(ir, src);
            let (tok, path) = (feed(plan, remap, f[0]), feed(plan, remap, f[1]));
            let array = ir
                .object(target)
                .and_then(|o| o.ty.component_ty(1))
                .cloned()
                .expect("replay: a read's target is (token, array)");
            fb.read(tok, path, &array, loc).expect("replay: read")
        }
        Pair { .. } | Output | LoopEnter | LoopBack | LoopExit => {
            unreachable!("emit_op: {op:?} is not an object-defining primitive")
        }
    }
}

/// Phase B — wire one Return in-edge (DESIGN §1.1). `Output`/`Pair` are existing
/// objects fed via `output()`; a primitive edge targets Return directly and is
/// rebuilt with `Dest::Ret{None}`. `LoopExit` was already emitted by the loop
/// quartet (skipped here).
#[allow(clippy::too_many_arguments)]
pub(super) fn reconstruct_return_writer(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    fmap: &SecondaryMap<FuncId, FuncId>,
    fused: &SecondaryMap<MorphismId, FuncId>,
    lifted: &SecondaryMap<ObjectId, FuncId>,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    m: MorphismId,
) {
    let morph = ir.morphism(m).expect("ret in-edge");
    // A fused `Map{g}` writing straight to Return (DESIGN §4).
    if let Some(&h) = fused.get(m) {
        emit_fused_map(
            fb,
            ir,
            plan,
            remap,
            h,
            m,
            Dest::Ret { slot: None },
            morph.loc,
        );
        return;
    }
    match morph.op {
        Operation::Output => {
            fb.output(feed(plan, remap, morph.source), None, morph.loc)
                .expect("replay: output full");
        }
        Operation::Pair { slot, .. } => {
            fb.output(feed(plan, remap, morph.source), Some(slot), morph.loc)
                .expect("replay: output slot");
        }
        // Emitted by the loop quartet with Dest::Ret{None}; not a Phase-B writer.
        Operation::LoopExit => {}
        // A value-producing primitive targeting Return directly (Dest::Ret{None}).
        op => {
            // An inlined `Call` writing Return directly (region-emission plan
            // Move 1): the callee's Return writers replay as this fn's own.
            if plan.inline.contains_key(m) {
                inline_call(fb, ir, plan, fmap, fused, lifted, remap, m, RetDest::Own);
                return;
            }
            emit_op(
                fb,
                ir,
                plan,
                fmap,
                remap,
                op,
                morph.source,
                morph.target,
                Dest::Ret { slot: None },
                morph.loc,
            );
        }
    }
}

// --- call inlining (region-emission plan Move 1) ----------------------------
