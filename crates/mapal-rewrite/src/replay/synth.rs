//! Synthesized bodies: lifted and fused functions and their maps.

use super::*;

/// Build the captured Map/Fold body requested by one [`LiftSpec`]. This is the
/// fused-body machinery pattern applied to a loop cone: map special leaves to
/// body parameters, replay the selected original objects through the same
/// primitive emitter, then write the cone root to the synthesized Return.
pub(super) fn synthesize_lifted_body(
    b: &mut IrBuilder,
    ir: &CategoryIr,
    fmap: &SecondaryMap<FuncId, FuncId>,
    merge: ObjectId,
    spec: &LiftSpec,
    n: &mut u32,
) -> FuncId {
    let loc = ir.object(merge).expect("lift merge").loc;
    let mut inputs: Vec<Ty> = spec
        .captures
        .iter()
        .map(|&o| ir.object(o).expect("lift capture").ty.clone())
        .collect();
    let (kind, prefix) = match spec.kind {
        LiftKind::Fold { accumulator, .. } => {
            inputs.push(ir.object(accumulator).expect("lift accumulator").ty.clone());
            (FuncKind::FoldBody, "lift_fold")
        }
        LiftKind::Map => (FuncKind::MapBody, "lift_map"),
    };
    inputs.push(Ty::i32());
    let input_ty = if kind == FuncKind::MapBody && inputs.len() == 1 {
        Ty::i32()
    } else {
        Ty::Tuple(inputs)
    };
    let output_ty = ir
        .object(spec.body_result)
        .expect("lift body result")
        .ty
        .clone();
    let name = format!("{prefix}${n}");
    *n += 1;
    let body = b
        .declare(kind, &name, input_ty, output_ty, loc)
        .expect("lift: declare body");

    let mut fb = b.build_fn(body).expect("lift: build body");
    let input = fb.input();
    let mut local: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();
    if kind == FuncKind::MapBody && spec.captures.is_empty() {
        local.insert(spec.counter, input);
    } else {
        let mut slot = 0u32;
        for &capture in &spec.captures {
            let p = fb
                .proj(input, slot, Dest::Fresh(None), loc)
                .expect("lift: capture projection");
            local.insert(capture, p);
            slot += 1;
        }
        if let LiftKind::Fold { accumulator, .. } = spec.kind {
            let p = fb
                .proj(input, slot, Dest::Fresh(None), loc)
                .expect("lift: accumulator projection");
            local.insert(accumulator, p);
            slot += 1;
        }
        let item = fb
            .proj(input, slot, Dest::Fresh(None), loc)
            .expect("lift: item projection");
        local.insert(spec.counter, item);
    }

    let mut members: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    for &o in &spec.body_objects {
        members.insert(o, ());
        let obj = ir.object(o).expect("lift body object");
        if obj.kind == ObjectKind::Constant {
            let c = fb
                .constant(obj.value.clone().expect("constant"), obj.loc)
                .expect("lift: body constant");
            local.insert(o, c);
        }
    }

    let empty = RewritePlan::new();
    let no_fuse: SecondaryMap<MorphismId, FuncId> = SecondaryMap::new();
    let no_lift: SecondaryMap<ObjectId, FuncId> = SecondaryMap::new();
    for o in object_topo_order(ir, ir.owner(merge)) {
        if o == spec.body_result || !members.contains_key(o) || local.contains_key(o) {
            continue;
        }
        let obj = ir.object(o).expect("lift body object");
        match obj.kind {
            ObjectKind::Temporary if is_internal_pack(ir, o) => {}
            ObjectKind::Temporary => {
                reconstruct_temp(&mut fb, ir, &empty, fmap, &no_fuse, &no_lift, &mut local, o)
            }
            _ => unreachable!("lift analysis admitted an unmapped body leaf"),
        }
    }
    emit_lifted_return(&mut fb, ir, fmap, spec.body_result, &local);
    fb.finish().expect("lift: finish body");
    body
}

/// Emit the lifted cone root directly to the synthesized Return when it has a
/// defining primitive. This preserves the lower-canonical body shape consumed
/// by `tile_plan` (final Add/Fold targets Return, rather than `tmp → Output`).
pub(super) fn emit_lifted_return(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    fmap: &SecondaryMap<FuncId, FuncId>,
    root: ObjectId,
    local: &SecondaryMap<ObjectId, ObjectId>,
) {
    if let Some(&value) = local.get(root) {
        fb.output(value, None, ir.object(root).expect("lift root").loc)
            .expect("lift: body output");
        return;
    }
    let obj = ir.object(root).expect("lift root");
    let ins = ir.in_edges(root);
    if !ins.is_empty()
        && ins
            .iter()
            .all(|&m| matches!(ir.morphism(m).unwrap().op, Operation::Pair { .. }))
    {
        let comps: Vec<_> = slot_feeders(ir, root)
            .into_iter()
            .map(|o| local[o])
            .collect();
        match &obj.ty {
            Ty::Tuple(_) => fb.pack(&comps, Dest::Ret { slot: None }, obj.loc),
            Ty::Struct { .. } => {
                fb.pack_struct(obj.ty.clone(), &comps, Dest::Ret { slot: None }, obj.loc)
            }
            Ty::Array { .. } => fb.pack_array(&comps, Dest::Ret { slot: None }, obj.loc),
            _ => unreachable!("lift: Pair-defined non-product root"),
        }
        .expect("lift: body return pack");
        return;
    }
    let [m] = ins else {
        unreachable!("lift: unmapped body root without one definition")
    };
    let morph = ir.morphism(*m).expect("lift root definer");
    emit_op(
        fb,
        ir,
        &RewritePlan::new(),
        fmap,
        local,
        morph.op,
        morph.source,
        morph.target,
        Dest::Ret { slot: None },
        morph.loc,
    );
}

// --- map fusion synthesis (DESIGN §4) -------------------------------------

/// How a fused body writes its Return: `Fresh` captures the value in a new object
/// (`f`, whose result feeds `g`); `Return` writes it into `h`'s own Return (`g`).
pub(super) enum Redirect {
    Fresh,
    Return,
}

/// Synthesize one fused `MapBody` `h = g ∘ f` (DESIGN §4): declare it, inline
/// `f`'s body (param ↦ `h.input`, Return captured as `r₁`), then `g`'s body
/// (param ↦ `r₁`, Return ↦ `h`'s Return). Returns the new body's `FuncId`.
/// Preconditions (single full-value Return writer, loop-free) are the analysis's
/// contract; a violation here is an internal bug and panics.
///
/// ADR-0027: with `captures = k > 0` both bodies take `(c₁…cₖ, x)` — `h`'s
/// input carries the shared captures, so `g`'s parameter is re-packed as
/// `(π₀ h.in … πₖ₋₁ h.in, r₁)` rather than bare `r₁`.
pub(super) fn synthesize_fused(
    b: &mut IrBuilder,
    ir: &CategoryIr,
    fmap: &SecondaryMap<FuncId, FuncId>,
    m_g: MorphismId,
    spec: &FusionSpec,
    n: &mut u32,
) -> FuncId {
    let loc = ir.morphism(m_g).expect("g-edge").loc;
    let f_def = ir.func(spec.f).expect("f body");
    let g_def = ir.func(spec.g).expect("g body");
    let in_ty = ir.object(f_def.input).expect("f in").ty.clone();
    let out_ty = ir.object(g_def.output).expect("g out").ty.clone();
    let name = format!("fused${n}");
    *n += 1;
    let h = b
        .declare(FuncKind::MapBody, &name, in_ty, out_ty, loc)
        .expect("replay: declare fused body");

    let mut hb = b.build_fn(h).expect("replay: build fused body");
    let h_in = hb.input();
    let r1 = inline_body(&mut hb, ir, fmap, spec.f, h_in, Redirect::Fresh);
    let g_param = if spec.captures == 0 {
        r1
    } else {
        let mut comps: Vec<ObjectId> = Vec::with_capacity(spec.captures as usize + 1);
        for i in 0..spec.captures {
            let p = hb
                .proj(h_in, i, Dest::Fresh(None), loc)
                .expect("replay: fused capture proj");
            comps.push(p);
        }
        comps.push(r1);
        hb.pack(&comps, Dest::Fresh(None), loc)
            .expect("replay: fused g-param pack")
    };
    inline_body(&mut hb, ir, fmap, spec.g, g_param, Redirect::Return);
    hb.finish().expect("replay: finish fused body");
    h
}

/// Inline one loop-free, single-full-value-Return body into the fused builder
/// (DESIGN §4). `param_new` is the object its Parameter maps to; `redirect`
/// decides how its Return value lands. Returns the object holding the Return
/// value (meaningful only for `Redirect::Fresh`). The copy is verbatim — an empty
/// plan, no nested fusion (a later fixpoint round reaches nested opportunities).
pub(super) fn inline_body(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    fmap: &SecondaryMap<FuncId, FuncId>,
    body: FuncId,
    param_new: ObjectId,
    redirect: Redirect,
) -> ObjectId {
    let plan = RewritePlan::new();
    let no_fuse: SecondaryMap<MorphismId, FuncId> = SecondaryMap::new();
    let no_lift: SecondaryMap<ObjectId, FuncId> = SecondaryMap::new();

    let def = ir.func(body).expect("inline body");
    let ret_obj = def.output;
    let obj_order = object_topo_order(ir, body);

    let mut local: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();
    local.insert(def.input, param_new);
    let mut done: SecondaryMap<ObjectId, bool> = SecondaryMap::new();

    for &o in &obj_order {
        if done.get(o).copied().unwrap_or(false) {
            continue;
        }
        reconstruct_a(
            fb,
            ir,
            &plan,
            fmap,
            &no_fuse,
            &no_lift,
            &mut local,
            &mut done,
            &obj_order,
            o,
            ret_obj,
            &RetDest::Own,
        );
    }

    // The single full-value Return writer (precondition): either an `Output` from
    // an existing body object, or a value-producing primitive into Return.
    let m = ir.in_edges(ret_obj)[0];
    let morph = ir.morphism(m).expect("ret writer");
    match redirect {
        Redirect::Fresh => match morph.op {
            Operation::Output => feed(&plan, &local, morph.source),
            op => emit_op(
                fb,
                ir,
                &plan,
                fmap,
                &local,
                op,
                morph.source,
                morph.target,
                Dest::Fresh(None),
                morph.loc,
            ),
        },
        Redirect::Return => {
            match morph.op {
                Operation::Output => {
                    fb.output(feed(&plan, &local, morph.source), None, morph.loc)
                        .expect("replay: fused output");
                }
                op => {
                    emit_op(
                        fb,
                        ir,
                        &plan,
                        fmap,
                        &local,
                        op,
                        morph.source,
                        morph.target,
                        Dest::Ret { slot: None },
                        morph.loc,
                    );
                }
            }
            param_new
        }
    }
}

/// Emit the fused `Map{h}` on the ORIGINAL array (DESIGN §4). `m_g` is the old
/// `Map{g}` edge; its mapped array `mid` was produced by a `Map{f}` — `f`'s
/// mapped array (already replayed) is the fused map's array input.
///
/// ADR-0027: with `captures = k > 0` the analysis has guaranteed both maps read
/// the identical capture objects, so the fused map re-threads `f`'s leading
/// source components with [`mapal_ir::FnBuilder::map_captured`]; the fused body's
/// capture count is `k`.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fused_map(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ir: &CategoryIr,
    plan: &RewritePlan,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    h: FuncId,
    m_g: MorphismId,
    dest: Dest,
    loc: mapal_ir::SourceLoc,
) -> ObjectId {
    let g_morph = ir.morphism(m_g).expect("g-edge");
    let Operation::Map { captures, .. } = g_morph.op else {
        unreachable!("emit_fused_map: fused key is not a Map edge")
    };
    let k = captures as usize;
    // g's mapped array (= f's result): the bare source for k=0, else the source
    // product's last slot.
    let mid = if k == 0 {
        g_morph.source
    } else {
        let feeders = slot_feeders(ir, g_morph.source);
        let (_, data) = split_capture_source(&feeders, k, 1, "Map");
        data[0]
    };
    let m_f = ir.in_edges(mid)[0];
    let f_src = ir.morphism(m_f).expect("f-edge").source;
    if k == 0 {
        fb.map(h, feed(plan, remap, f_src), dest, loc)
            .expect("replay: fused map")
    } else {
        let f = slot_feeders(ir, f_src);
        let (caps, data) = split_capture_source(&f, k, 1, "Map");
        let caps: Vec<ObjectId> = caps.iter().map(|&c| feed(plan, remap, c)).collect();
        fb.map_captured(h, &caps, feed(plan, remap, data[0]), dest, loc)
            .expect("replay: fused map_captured")
    }
}
