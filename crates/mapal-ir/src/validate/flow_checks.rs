//! Token, loop and reference checks of `validate`, and the graph helpers they share.

use super::*;

/// I4 + I4b: token linearity (with the structural loop-fork exception) and the
/// token sink rule, all SCC-derived.
pub(super) fn check_tokens(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    // Precompute per-function SCC membership for the loop-fork exception.
    let scc_of = build_scc_membership(ir);

    for (id, obj) in ir.objects() {
        if !ty::ty_contains_token(&obj.ty) {
            continue;
        }
        // Token-bearing consumers: out-edges whose target ty contains a token.
        let mut token_consumers: Vec<MorphismId> = Vec::new();
        for &mid in ir.out_edges(id) {
            let tgt = ir.morphism(mid).unwrap().target;
            if ir
                .object(tgt)
                .map(|o| ty::ty_contains_token(&o.ty))
                .unwrap_or(false)
            {
                token_consumers.push(mid);
            }
        }

        // I4b token sink: zero token consumers ⇒ must be the owning Return.
        if token_consumers.is_empty() {
            let owner = ir.owner(id);
            let is_return = ir.func(owner).map(|d| d.output) == Some(id);
            if !is_return {
                v.push(IrViolation::TokenDropped(id));
            }
            continue;
        }

        // I4 linearity: at most one token consumer, except the loop fork.
        if token_consumers.len() > 1 && !is_loop_fork(ir, &scc_of, id, &token_consumers) {
            v.push(IrViolation::TokenNotLinear(id));
        }
    }

    // Phi may not select a token-bearing ty.
    for (mid, m) in ir.morphisms() {
        if m.op == Operation::Phi
            && let Some(o) = ir.object(m.target)
            && ty::ty_contains_token(&o.ty)
        {
            v.push(IrViolation::TokenInPhi(mid));
        }
    }

    // Map/Fold bodies must be token-free.
    for (fid, def) in ir.funcs() {
        if matches!(
            def.kind,
            crate::graph::FuncKind::MapBody | crate::graph::FuncKind::FoldBody
        ) {
            let in_tok = ir
                .object(def.input)
                .map(|o| ty::ty_contains_token(&o.ty))
                .unwrap_or(false);
            let out_tok = ir
                .object(def.output)
                .map(|o| ty::ty_contains_token(&o.ty))
                .unwrap_or(false);
            let any_tok = def.morphisms.iter().any(|&m| {
                let mm = ir.morphism(m).unwrap();
                ir.object(mm.source)
                    .map(|o| ty::ty_contains_token(&o.ty))
                    .unwrap_or(false)
            });
            if in_tok || out_tok || any_tok {
                v.push(IrViolation::TokenInBody(fid));
            }
        }
    }
}

/// The structural loop fork (DESIGN §8/I4): an object inside a loop SCC with
/// exactly two token consumers, both `Pair` edges into route objects, one route
/// consumed by a `LoopBack` into a merge whose SCC contains the object, the
/// other by a `LoopExit` whose source is in that SCC.
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
    // Both consumers must be Pair edges; classify each consumer's forward cone
    // (the back route reaches a LoopBack into the SCC; the exit route reaches a
    // LoopExit leaving it). When U is a product, the fork is several edges
    // upstream of the route, so we trace cones rather than one hop.
    let mut has_back = false;
    let mut has_exit = false;
    for &mid in consumers {
        let m = ir.morphism(mid).unwrap();
        // A consumer threads the token on: a `Pair` into an effect's
        // source, or `TimeMs`, which takes the bare token.
        if !matches!(m.op, Operation::Pair { .. } | Operation::TimeMs) {
            return false;
        }
        match cone_classify(ir, scc_of, m.target, obj_scc) {
            Some(true) => has_back = true,
            Some(false) => has_exit = true,
            None => {}
        }
    }
    has_back && has_exit
}

/// Classify a forward cone from `start`: `Some(true)` if it reaches a `LoopBack`
/// into `scc`, `Some(false)` if it reaches a `LoopExit` leaving `scc`, else
/// `None` (iterative BFS; J1). Independent local copy (no builder code shared).
pub(super) fn cone_classify(
    ir: &CategoryIr,
    scc_of: &SecondaryMap<ObjectId, usize>,
    start: ObjectId,
    scc: usize,
) -> Option<bool> {
    let mut seen: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
    let mut queue: Vec<ObjectId> = vec![start];
    seen.insert(start, true);
    let mut found_exit = false;
    while let Some(o) = queue.pop() {
        for &m in ir.out_edges(o) {
            let mm = ir.morphism(m).unwrap();
            match mm.op {
                Operation::LoopBack if scc_of.get(mm.target) == Some(&scc) => return Some(true),
                Operation::LoopExit if scc_of.get(mm.target) != Some(&scc) => found_exit = true,
                _ => {}
            }
            let t = mm.target;
            if !seen.get(t).copied().unwrap_or(false) {
                seen.insert(t, true);
                queue.push(t);
            }
        }
    }
    if found_exit { Some(false) } else { None }
}

/// I5 + loop-carried-token escape, SCC-derived per function.
pub(super) fn check_loops(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    for (f, _) in ir.funcs() {
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
            let merge_scc = match scc_of.get(id) {
                Some(&s) => s,
                None => {
                    v.push(IrViolation::BadLoop(id));
                    continue;
                }
            };
            if !nontrivial(merge_scc) {
                v.push(IrViolation::BadLoop(id));
                continue;
            }
            let mut backs = 0;
            let mut bad = false;
            for &mid in ir.in_edges(id) {
                let m = ir.morphism(mid).unwrap();
                match m.op {
                    Operation::LoopEnter if scc_of.get(m.source) == Some(&merge_scc) => {
                        bad = true;
                    }
                    Operation::LoopBack => {
                        backs += 1;
                        // Per-edge I5 on the carried STATE, not the route object:
                        // the route `(next_state, cond)` is always inside the SCC
                        // via its `cond` slot edge, so test the route's slot-0
                        // `Pair` source (the next_state). DESIGN §9 I5.
                        match loopback_state_source(ir, m.source) {
                            Some(s) if scc_of.get(s) == Some(&merge_scc) => {}
                            _ => bad = true,
                        }
                    }
                    _ => {}
                }
            }
            if backs == 0 {
                bad = true;
            }
            // ≥1 LoopExit belonging to this loop, target outside; token escape.
            // The exit route is a downward leaf (not in the SCC), so "belongs"
            // means its route has a Pair in-edge from an object in the SCC.
            let u_has_token = ty::ty_contains_token(&obj.ty);
            let mut exits = 0;
            for (_, m) in ir.morphisms() {
                if m.op != Operation::LoopExit {
                    continue;
                }
                if !reachable(ir, id, m.source) {
                    continue;
                }
                if scc_of.get(m.target) == Some(&merge_scc) {
                    continue;
                }
                exits += 1;
                if u_has_token {
                    let b_ty = &ir.object(m.target).unwrap().ty;
                    if !ty::ty_contains_token(b_ty) {
                        v.push(IrViolation::TokenNotEscaping(id));
                    }
                }
            }
            if exits == 0 {
                bad = true;
            }
            if bad {
                v.push(IrViolation::BadLoop(id));
            }
        }
    }
}

/// I6: function reference graph acyclic (iterative DFS; J1).
pub(super) fn check_references(ir: &CategoryIr, v: &mut Vec<IrViolation>) {
    let func_ids: Vec<FuncId> = ir.funcs().map(|(id, _)| id).collect();
    let mut color: SecondaryMap<FuncId, u8> = SecondaryMap::new();
    for &f in &func_ids {
        color.insert(f, 0);
    }
    let refs_of = |f: FuncId| -> Vec<FuncId> {
        let mut out = Vec::new();
        if let Some(def) = ir.func(f) {
            for &m in &def.morphisms {
                if let Some(mm) = ir.morphism(m) {
                    match mm.op {
                        Operation::Call(g)
                        | Operation::Map { body: g, .. }
                        | Operation::Fold { body: g, .. } => out.push(g),
                        _ => {}
                    }
                }
            }
        }
        out
    };
    for &root in &func_ids {
        if color[root] != 0 {
            continue;
        }
        let mut path: Vec<FuncId> = Vec::new();
        struct Frame {
            refs: Vec<FuncId>,
            cursor: usize,
            node: FuncId,
        }
        let mut stack: Vec<Frame> = vec![Frame {
            refs: refs_of(root),
            cursor: 0,
            node: root,
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
                        stack.push(Frame {
                            refs: refs_of(w),
                            cursor: 0,
                            node: w,
                        });
                    }
                    1 => {
                        let start = path.iter().position(|&x| x == w).unwrap_or(0);
                        v.push(IrViolation::RecursiveReference(path[start..].to_vec()));
                        return;
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
}

// --- small local helpers (no builder code shared) -------------------------

/// Per-object SCC index across all functions (each function's SCCs are disjoint
/// by ownership, so a single map is unambiguous).
pub(super) fn build_scc_membership(ir: &CategoryIr) -> SecondaryMap<ObjectId, usize> {
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
    scc_of
}

/// Whether `to` is forward-reachable from `from` (iterative BFS; J1). Used to
/// attribute a `LoopExit`'s route to its merge.
pub(super) fn reachable(ir: &CategoryIr, from: ObjectId, to: ObjectId) -> bool {
    let mut seen: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
    let mut queue: Vec<ObjectId> = vec![from];
    seen.insert(from, true);
    while let Some(o) = queue.pop() {
        if o == to {
            return true;
        }
        for &m in ir.out_edges(o) {
            let t = ir.morphism(m).unwrap().target;
            if !seen.get(t).copied().unwrap_or(false) {
                seen.insert(t, true);
                queue.push(t);
            }
        }
    }
    false
}

/// The carried-state object of a `LoopBack` route (DESIGN §7): the source of the
/// route's slot-0 `Pair` in-edge (the `next_state`; slot 1 is the `cond`).
/// `route` is the `LoopBack` morphism's source. `None` if the route is not the
/// canonical `(next_state, cond)` shape. Re-derived independently of the builder
/// (DESIGN §11).
pub(super) fn loopback_state_source(ir: &CategoryIr, route: ObjectId) -> Option<ObjectId> {
    for &mid in ir.in_edges(route) {
        let m = ir.morphism(mid)?;
        if let Operation::Pair { slot: 0, .. } = m.op {
            return Some(m.source);
        }
    }
    None
}

/// The two component tys of a 2-tuple, else `None`.
pub(super) fn two_tuple(ty: &Ty) -> Option<(&Ty, &Ty)> {
    match ty {
        Ty::Tuple(ts) if ts.len() == 2 => Some((&ts[0], &ts[1])),
        _ => None,
    }
}

/// Iterative Core ty check (mirrors the builder's intake, encoded once here as
/// the §5.1-adjacent data rule — no shared helper, per §11 independence).
pub(super) fn ty_is_core(ty: &Ty) -> bool {
    let mut stack: Vec<&Ty> = vec![ty];
    while let Some(t) = stack.pop() {
        match t {
            Ty::Int { bits, signed } => {
                if !matches!((bits, signed), (32, true) | (64, true) | (8, false)) {
                    return false;
                }
            }
            Ty::Float { bits } => {
                if *bits != 32 && *bits != 64 {
                    return false;
                }
            }
            Ty::Bool | Ty::Unit | Ty::Str | Ty::IoToken | Ty::Narrow(_) => {}
            Ty::Tuple(ts) => {
                if ts.len() < 2 {
                    return false;
                }
                stack.extend(ts.iter());
            }
            Ty::Struct { fields, .. } => stack.extend(fields.iter().map(|(_, t)| t)),
            Ty::Array { elem, size } => {
                if *size < 1 {
                    return false;
                }
                stack.push(elem);
            }
        }
    }
    true
}
