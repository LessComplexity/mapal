//! The seal-time global checks `IrBuilder::seal` runs: token sinks and linearity, SCC placement, struct names, acyclic references, returns and loops.

use super::*;

// --- the seal-time global checks ------------------------------------------

/// Run the global graph-shape checks `seal` owns (DESIGN §9 ledger): I4b token
/// sinks, I5 per-edge SCC placement, I6 acyclicity + StructNameConflict, I-RET
/// completeness, and loop completeness.
pub(super) fn check_sealed(ir: &CategoryIr) -> Result<(), IrError> {
    check_struct_name_conflict(ir)?;
    check_str(ir)?;
    check_reference_acyclic(ir)?;
    check_i_ret(ir)?;
    check_token_linearity(ir)?;
    check_token_sinks(ir)?;
    check_loops(ir)?;
    Ok(())
}

/// I9s graph-shape clause (DESIGN §9): `Str` appears only as a `Constant`
/// object's ty, or the second component of the `(IoToken, Str)` pair that
/// `print()` builds internally. The per-call packers already reject `Str`
/// components, but a `Str`-bearing *declared* input/output ty (a bare `Str`
/// param/return, or `Str` inside a tuple/struct/array) reaches the graph without
/// going through a packer — so `seal` must re-derive this graph-shape clause to
/// keep the builder's rejection set identical to `validate`'s (the headline
/// "seal Ok ⇒ validate empty" property; DESIGN §16 item 3). Mirrors
/// `validate::check_str` exactly.
pub(super) fn check_str(ir: &CategoryIr) -> Result<(), IrError> {
    for (_, obj) in ir.objects() {
        match &obj.ty {
            // A bare Str object is legitimate only as a Constant (the literal).
            Ty::Str => {
                if obj.kind != ObjectKind::Constant {
                    return Err(IrError::StrOutsidePrint);
                }
            }
            // The print pair (IoToken, Str) is allowed; no other product may
            // contain Str.
            Ty::Tuple(ts) => {
                if ts.iter().any(ty::ty_contains_str) {
                    let is_print_pair = ts.len() == 2 && ts[0] == Ty::IoToken && ts[1] == Ty::Str;
                    if !is_print_pair {
                        return Err(IrError::StrOutsidePrint);
                    }
                }
            }
            other => {
                if ty::ty_contains_str(other) {
                    return Err(IrError::StrOutsidePrint);
                }
            }
        }
    }
    Ok(())
}

/// I4 (token linearity, DESIGN §8): a token-bearing object has at most one
/// token-bearing consumer, except the structural loop fork (exactly two `Pair`
/// consumers, one route into a `LoopBack` to a merge in the object's SCC, the
/// other into a `LoopExit` whose source is in that SCC). SCC-derived, no
/// `LoopHandle` knowledge.
pub(super) fn check_token_linearity(ir: &CategoryIr) -> Result<(), IrError> {
    // Per-object SCC membership across all functions (each function's SCCs are
    // disjoint by ownership).
    let mut scc_of: SecondaryMap<ObjectId, usize> = SecondaryMap::new();
    let mut base = 0usize;
    for (f, _) in ir.funcs() {
        let comps = ir.sccs(f);
        for (i, comp) in comps.iter().enumerate() {
            for &o in comp {
                scc_of.insert(o, base + i);
            }
        }
        base += comps.len();
    }

    for (id, obj) in ir.objects() {
        if !ty::ty_contains_token(&obj.ty) {
            continue;
        }
        let consumers: Vec<MorphismId> = ir
            .out_edges(id)
            .iter()
            .copied()
            .filter(|&m| {
                let tgt = ir.morphism(m).unwrap().target;
                ir.object(tgt)
                    .map(|o| ty::ty_contains_token(&o.ty))
                    .unwrap_or(false)
            })
            .collect();
        if consumers.len() > 1 && !is_loop_fork(ir, &scc_of, id, &consumers) {
            return Err(IrError::TokenNotLinear);
        }
    }
    Ok(())
}

/// The structural loop fork (DESIGN §8/I4): a token-bearing object inside a loop
/// SCC with exactly two token consumers (both `Pair` edges), one whose forward
/// cone reaches a `LoopBack` returning into the object's SCC, the other whose
/// cone reaches a `LoopExit` leaving the SCC. The two fire mutually exclusively
/// on the shared `Bool`.
///
/// Generalized beyond a single hop: when `U` is a product, the forking token is
/// packed into the loop-state object first, so the `LoopBack`/`LoopExit` is
/// several edges downstream. We classify each consumer by what its forward cone
/// reaches.
pub(super) fn is_loop_fork(
    ir: &CategoryIr,
    scc_of: &SecondaryMap<ObjectId, usize>,
    obj: ObjectId,
    consumers: &[MorphismId],
) -> bool {
    if consumers.len() != 2 {
        return false;
    }
    let obj_scc = match scc_of.get(obj) {
        Some(&s) => s,
        None => return false,
    };
    let mut has_back = false;
    let mut has_exit = false;
    for &mid in consumers {
        let m = ir.morphism(mid).unwrap();
        // A consumer threads the token on: a `Pair` into an effect's
        // source, or `TimeMs`, which takes the bare token.
        if !matches!(m.op, Operation::Pair { .. } | Operation::TimeMs) {
            return false;
        }
        match cone_reaches(ir, scc_of, m.target, obj_scc) {
            ConeKind::Back => has_back = true,
            ConeKind::Exit => has_exit = true,
            ConeKind::Neither => {}
        }
    }
    has_back && has_exit
}

/// What a forward cone from `start` reaches w.r.t. the loop SCC `scc`.
pub(super) enum ConeKind {
    /// Reaches a `LoopBack` returning into `scc`.
    Back,
    /// Reaches a `LoopExit` leaving `scc`.
    Exit,
    Neither,
}

/// Classify the forward cone from `start` (iterative BFS; J1). `Back` if it hits
/// a `LoopBack` into `scc`; `Exit` if it hits a `LoopExit` leaving `scc`; the
/// back classification wins if both are somehow reachable (the back route is the
/// in-SCC path).
pub(super) fn cone_reaches(
    ir: &CategoryIr,
    scc_of: &SecondaryMap<ObjectId, usize>,
    start: ObjectId,
    scc: usize,
) -> ConeKind {
    let mut seen: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
    let mut queue: Vec<ObjectId> = vec![start];
    seen.insert(start, true);
    let mut found_exit = false;
    while let Some(o) = queue.pop() {
        for &m in ir.out_edges(o) {
            let mm = ir.morphism(m).unwrap();
            match mm.op {
                Operation::LoopBack if scc_of.get(mm.target) == Some(&scc) => {
                    return ConeKind::Back;
                }
                Operation::LoopExit if scc_of.get(mm.target) != Some(&scc) => {
                    found_exit = true;
                }
                _ => {}
            }
            let t = mm.target;
            // Don't traverse past a LoopBack into the SCC (handled above) or out
            // through a LoopExit; just continue the forward walk for packs.
            if !seen.get(t).copied().unwrap_or(false) {
                seen.insert(t, true);
                queue.push(t);
            }
        }
    }
    if found_exit {
        ConeKind::Exit
    } else {
        ConeKind::Neither
    }
}

/// I6 / StructNameConflict: all `Struct` tys sharing a name must have identical
/// fields across the whole graph.
pub(super) fn check_struct_name_conflict(ir: &CategoryIr) -> Result<(), IrError> {
    // Insertion-ordered (name → first fields) registry; no HashMap (I12).
    let mut seen: Vec<(String, Vec<(String, Ty)>)> = Vec::new();
    for (_, obj) in ir.objects() {
        collect_struct_defs(&obj.ty, &mut seen)?;
    }
    Ok(())
}

/// Walk a ty (iterative; J1), recording each `Struct`'s fields and flagging a
/// name reused with different fields.
pub(super) fn collect_struct_defs(
    ty: &Ty,
    seen: &mut Vec<(String, Vec<(String, Ty)>)>,
) -> Result<(), IrError> {
    let mut stack: Vec<&Ty> = vec![ty];
    while let Some(t) = stack.pop() {
        match t {
            Ty::Struct { name, fields } => {
                if let Some((_, prev)) = seen.iter().find(|(n, _)| n == name) {
                    if prev != fields {
                        return Err(IrError::StructNameConflict);
                    }
                } else {
                    seen.push((name.clone(), fields.clone()));
                }
                for (_, inner) in fields {
                    stack.push(inner);
                }
            }
            Ty::Tuple(ts) => {
                for inner in ts {
                    stack.push(inner);
                }
            }
            Ty::Array { elem, .. } => stack.push(elem),
            _ => {}
        }
    }
    Ok(())
}

/// I6: the reference graph (Call edges + Map/Fold body refs) is acyclic.
/// Iterative DFS with an explicit stack (J1); reports the cycle on detection.
pub(super) fn check_reference_acyclic(ir: &CategoryIr) -> Result<(), IrError> {
    // Build a reference adjacency: f → [referenced funcs], in deterministic
    // order (over the function's morphisms).
    let func_ids: Vec<FuncId> = ir.funcs().map(|(id, _)| id).collect();

    // 0 = unvisited, 1 = on stack (gray), 2 = done (black).
    let mut color: SecondaryMap<FuncId, u8> = SecondaryMap::new();
    for &f in &func_ids {
        color.insert(f, 0);
    }

    fn refs_of(ir: &CategoryIr, f: FuncId) -> Vec<FuncId> {
        let mut out = Vec::new();
        if let Some(def) = ir.func(f) {
            for &m in &def.morphisms {
                if let Some(morph) = ir.morphism(m) {
                    match morph.op {
                        Operation::Call(g)
                        | Operation::Map { body: g, .. }
                        | Operation::Fold { body: g, .. } => out.push(g),
                        _ => {}
                    }
                }
            }
        }
        out
    }

    for &root in &func_ids {
        if color[root] != 0 {
            continue;
        }
        // path holds the gray stack for cycle reconstruction.
        let mut path: Vec<FuncId> = Vec::new();
        struct Frame {
            node: FuncId,
            refs: Vec<FuncId>,
            cursor: usize,
        }
        let mut stack: Vec<Frame> = vec![Frame {
            node: root,
            refs: refs_of(ir, root),
            cursor: 0,
        }];
        color.insert(root, 1);
        path.push(root);

        while let Some(frame) = stack.last_mut() {
            if frame.cursor < frame.refs.len() {
                let w = frame.refs[frame.cursor];
                frame.cursor += 1;
                match color.get(w).copied().unwrap_or(2) {
                    0 => {
                        color.insert(w, 1);
                        path.push(w);
                        let refs = refs_of(ir, w);
                        stack.push(Frame {
                            node: w,
                            refs,
                            cursor: 0,
                        });
                    }
                    1 => {
                        // Back edge → cycle. Slice path from w to the end.
                        let start = path.iter().position(|&x| x == w).unwrap_or(0);
                        let cycle = path[start..].to_vec();
                        return Err(IrError::RecursiveCall { cycle });
                    }
                    _ => {}
                }
            } else {
                color.insert(frame.node, 2);
                path.pop();
                stack.pop();
            }
        }
    }
    Ok(())
}

/// I-RET completeness (DESIGN §9): every function's Return object has a complete
/// writer set — either ≥1 full-value writer, or all `arity` distinct slots, and
/// never mixed. Unit returns may have zero writers.
pub(super) fn check_i_ret(ir: &CategoryIr) -> Result<(), IrError> {
    for (_, def) in ir.funcs() {
        let ret = def.output;
        let out_ty = ir.object(ret).map(|o| &o.ty).cloned().unwrap_or(Ty::Unit);
        let mut full = 0usize;
        let mut slots: Vec<u32> = Vec::new();
        for &m in ir.in_edges(ret) {
            let morph = ir.morphism(m).expect("ret in-edge");
            match morph.op {
                Operation::Pair { slot, .. } => slots.push(slot),
                _ => full += 1,
            }
        }
        if full > 0 && !slots.is_empty() {
            return Err(IrError::RetMixedWriters);
        }
        if !slots.is_empty() {
            let arity = match out_ty.product_arity() {
                Some(a) => a,
                None => return Err(IrError::RetNotProduct),
            };
            slots.sort_unstable();
            for w in slots.windows(2) {
                if w[0] == w[1] {
                    return Err(IrError::RetSlotConflict);
                }
            }
            // Completeness against the true `u64` arity without truncation: the
            // deduplicated `u32` slots must be exactly `0..arity` (F4/SND-3).
            if (slots.len() as u64) != arity {
                return Err(IrError::RetSlotMissing);
            }
            for (k, &slot) in slots.iter().enumerate() {
                if slot as u64 != k as u64 {
                    return Err(IrError::RetSlotMissing);
                }
            }
        } else if full == 0 && out_ty != Ty::Unit {
            // No writers and a non-Unit output: a missing return.
            return Err(IrError::RetSlotMissing);
        }
    }
    Ok(())
}

/// I4b (token sink): every token-bearing object with zero token-bearing
/// out-edges must be its function's Return object.
pub(super) fn check_token_sinks(ir: &CategoryIr) -> Result<(), IrError> {
    for (id, obj) in ir.objects() {
        if !ty::ty_contains_token(&obj.ty) {
            continue;
        }
        let has_token_consumer = ir.out_edges(id).iter().any(|&m| {
            let tgt = ir.morphism(m).unwrap().target;
            ir.object(tgt)
                .map(|o| ty::ty_contains_token(&o.ty))
                .unwrap_or(false)
        });
        if has_token_consumer {
            continue;
        }
        // No token-bearing out-edge: must be the owning function's Return.
        let owner = ir.owner(id);
        let is_return = ir.func(owner).map(|d| d.output) == Some(id);
        if !is_return {
            return Err(IrError::TokenDropped);
        }
    }
    Ok(())
}

/// I5 (per-edge SCC placement) + loop completeness + the loop-carried-token
/// escape rule (DESIGN §7/§8). Runs per LoopMerge using the function's SCCs.
pub(super) fn check_loops(ir: &CategoryIr) -> Result<(), IrError> {
    for (f, _) in ir.funcs() {
        // Membership lookup: object → its SCC index (for this function).
        let comps = ir.sccs(f);
        let mut scc_of: SecondaryMap<ObjectId, usize> = SecondaryMap::new();
        for (i, comp) in comps.iter().enumerate() {
            for &o in comp {
                scc_of.insert(o, i);
            }
        }
        let nontrivial = |i: usize| -> bool {
            comps[i].len() > 1
                || comps[i]
                    .first()
                    .map(|&o| {
                        ir.out_edges(o)
                            .iter()
                            .any(|&m| ir.morphism(m).unwrap().target == o)
                    })
                    .unwrap_or(false)
        };

        for (id, obj) in ir.objects() {
            if ir.owner(id) != f || obj.kind != ObjectKind::LoopMerge {
                continue;
            }
            let merge_scc = scc_of[id];
            if !nontrivial(merge_scc) {
                // A LoopMerge whose SCC is trivial means no back edge closes the
                // cycle: treat as a missing back edge.
                return Err(IrError::LoopWithoutBack);
            }

            let mut backs = 0usize;
            let mut exits = 0usize;
            // Token-escape bookkeeping: does U carry a token?
            let u_has_token = ty::ty_contains_token(&obj.ty);

            // LoopEnter source must be OUTSIDE the SCC; LoopBack sources INSIDE.
            for &m in ir.in_edges(id) {
                let morph = ir.morphism(m).unwrap();
                match morph.op {
                    // Enter from inside the cycle — malformed.
                    Operation::LoopEnter if scc_of.get(morph.source) == Some(&merge_scc) => {
                        return Err(IrError::LoopBackOutsideScc);
                    }
                    Operation::LoopBack => {
                        backs += 1;
                        // I5 is a per-edge check on the carried STATE, not on the
                        // route object. The route `(next_state, cond)` is always
                        // pulled into the SCC by its `cond` slot edge (the guard
                        // is genuinely in the cycle), so testing the route's own
                        // SCC membership would pass even when `next_state` came
                        // from outside the loop (e.g. a constant). Test the
                        // route's slot-0 `Pair` source — the next_state object —
                        // instead (DESIGN §5.1 LoopBack row; §9 I5).
                        let state = loopback_state_source(ir, morph.source);
                        match state {
                            Some(s) if scc_of.get(s) == Some(&merge_scc) => {}
                            _ => return Err(IrError::LoopBackOutsideScc),
                        }
                    }
                    _ => {}
                }
            }
            if backs == 0 {
                return Err(IrError::LoopWithoutBack);
            }

            // LoopExit edges belonging to this merge: the exit's source is a
            // route object `(B, Bool)` packed from a value derived *inside* the
            // SCC (the route itself is a downward leaf, not in the cycle). A
            // LoopExit belongs to this loop iff its route has a `Pair` in-edge
            // whose source is in this SCC, and the exit target is outside.
            for (_, morph) in ir.morphisms() {
                if morph.op != Operation::LoopExit {
                    continue;
                }
                if !exit_reachable_from_merge(ir, id, morph.source) {
                    continue;
                }
                if scc_of.get(morph.target) == Some(&merge_scc) {
                    // Exit target inside the cycle — not a real exit.
                    continue;
                }
                exits += 1;
                if u_has_token {
                    // token-in ⇒ token-out: the exit payload B must carry it.
                    let b_ty = &ir.object(morph.target).unwrap().ty;
                    if !ty::ty_contains_token(b_ty) {
                        return Err(IrError::TokenNotEscaping);
                    }
                }
            }
            if exits == 0 {
                return Err(IrError::LoopWithoutExit);
            }
        }
    }
    Ok(())
}
