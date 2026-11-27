//! Replay support: slot feeders, topo order, liveness of functions, canonical-loop test.

use super::*;

/// Whether `op` reads its source as an **internally-packed** tuple (the builder
/// mints that product atomically). These sources are never materialized.
///
/// ADR-0027: a `Map`/`Fold` with `captures > 0` qualifies — `map_captured` /
/// `fold_captured` mint the `(c₁…cₖ, …)` source product atomically. The k=0
/// forms keep the historical shapes (Map's bare array; Fold's materialized
/// `(acc, arr)` product). Keep in sync with `graph_rewrites`' copy — CSE
/// representatives must never be packs replay treats as internal.
pub(super) fn reads_packed_source(op: Operation) -> bool {
    use Operation::*;
    match op {
        Add
        | Sub
        | Mul
        | Div
        | Mod
        | Eq
        | Neq
        | Lt
        | Gt
        | Le
        | Ge
        | And
        | Or
        | Phi
        | Index
        | Zip
        | Update
        | Print { .. }
        | Read
        | LoopBack
        | LoopExit => true,
        Map { captures, .. } | Fold { captures, .. } => captures > 0,
        _ => false,
    }
}

/// Whether `o` is an internal pack (a product consumed **only** as one
/// packed-source primitive's source) — not materialized (DESIGN §1.1). A product
/// with an explicit consumer or ≥2 consumers falls through to `false` and is
/// materialized.
pub(super) fn is_internal_pack(ir: &CategoryIr, o: ObjectId) -> bool {
    if ir.object(o).map(|x| x.kind) != Some(ObjectKind::Temporary) {
        return false;
    }
    let outs = ir.out_edges(o);
    if outs.len() != 1 {
        return false;
    }
    let m = ir.morphism(outs[0]).unwrap();
    m.source == o && reads_packed_source(m.op)
}

/// The slot feeders of a product object, in slot order (declaration = slot
/// order, load-bearing for non-commutative ops; DESIGN §1.1).
pub(super) fn slot_feeders(ir: &CategoryIr, product: ObjectId) -> Vec<ObjectId> {
    let mut v: Vec<(u32, ObjectId)> = ir
        .in_edges(product)
        .iter()
        .filter_map(|&m| {
            let mo = ir.morphism(m).unwrap();
            if let Operation::Pair { slot, .. } = mo.op {
                Some((slot, mo.source))
            } else {
                None
            }
        })
        .collect();
    v.sort_by_key(|(s, _)| *s);
    v.into_iter().map(|(_, s)| s).collect()
}

/// Split a captured `Map`/`Fold` source product's slot feeders into the
/// leading `k` capture components and the trailing data inputs (`data` = 1
/// for `Map`: the array; `data` = 2 for `Fold`: acc + array). The builder
/// pins the source arity at construction (R2), so a mismatch here means a
/// malformed graph reached replay — an internal bug, named loudly. Never a
/// `debug_assert`: over-arity would silently truncate in release, and a
/// non-Pair-fed source (a short feeder list, since [`slot_feeders`] filters
/// to `Pair` edges) would panic as an unmarked index-OOB.
pub(super) fn split_capture_source<'a, T>(
    feeders: &'a [T],
    k: usize,
    data: usize,
    op: &str,
) -> (&'a [T], &'a [T]) {
    assert!(
        feeders.len() == k + data,
        "replay: {op}{{k={k}}} source product arity expected {}, found {}",
        k + data,
        feeders.len()
    );
    feeders.split_at(k)
}

/// Feeder lookup: resolve `alias` transitively, then map to the new id (DESIGN
/// §1.1 id remap). Present by the time it is read (alias points earlier in topo,
/// feeders precede consumers).
pub(super) fn feed(
    plan: &RewritePlan,
    remap: &SecondaryMap<ObjectId, ObjectId>,
    o: ObjectId,
) -> ObjectId {
    let r = plan.resolve_alias(o);
    remap
        .get(r)
        .copied()
        .unwrap_or_else(|| panic!("replay: feeder {r:?} (from {o:?}) is not mapped"))
}

// --- loop structure helpers -----------------------------------------------

/// The loc of `merge`'s `LoopBack` morphism.
pub(super) fn back_morph_loc(ir: &CategoryIr, merge: ObjectId) -> mapal_ir::SourceLoc {
    ir.in_edges(merge)
        .iter()
        .find_map(|&m| {
            let mo = ir.morphism(m).unwrap();
            (mo.op == Operation::LoopBack).then_some(mo.loc)
        })
        .expect("loop back morphism")
}

// --- canonicity gate (R6) -------------------------------------------------

/// Whether every loop in `ir` is the canonical quartet: one merge per SCC, one
/// back and one exit per merge (DESIGN §5, R6). A non-canonical shape makes
/// `replay` return [`ReplayError::NonCanonicalLoop`].
///
/// Delegates the per-merge attribution to [`mapal_ir::CategoryIr::loop_plan`]
/// (DESIGN §3, BL7 — one source of truth): a merge is canonical iff its SCC has
/// exactly one merge and `loop_plan` yields `Some` (which itself encodes the
/// one-back / one-attributed-exit conditions).
pub fn is_canonical(ir: &CategoryIr) -> bool {
    ir.funcs().all(|(f, _)| {
        ir.loop_structure(f)
            .iter()
            .all(|lscc| lscc.merges.len() == 1 && ir.loop_plan(f, lscc.merges[0]).is_some())
    })
}

// --- object topo order ----------------------------------------------------

/// A topological object order for function `f` — feeders before consumers, loop
/// back-edges non-gating, `LoopMerge` complete on its `LoopEnter` (DESIGN §1.1).
/// Derived from `ir.topo_order`, which already defers `LoopEnter` so every
/// loop-invariant feeder precedes the header.
pub(super) fn object_topo_order(ir: &CategoryIr, f: FuncId) -> Vec<ObjectId> {
    let objs: Vec<ObjectId> = ir
        .objects()
        .filter(|(id, _)| ir.try_owner(*id) == Some(f))
        .map(|(id, _)| id)
        .collect();

    // Gating in-edge count per object (mirrors the builder's completion rule).
    let mut remaining: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
    for &o in &objs {
        let kind = ir.object(o).unwrap().kind;
        let count = match kind {
            ObjectKind::Parameter | ObjectKind::Constant => 0,
            ObjectKind::LoopMerge => ir
                .in_edges(o)
                .iter()
                .filter(|&&m| ir.morphism(m).unwrap().op == Operation::LoopEnter)
                .count() as u32,
            _ => ir.in_edges(o).len() as u32,
        };
        remaining.insert(o, count);
    }

    let mut order: Vec<ObjectId> = Vec::new();
    for &o in &objs {
        if remaining[o] == 0 {
            order.push(o);
        }
    }

    // Drive completion off the morphism topo order; LoopBack never gates.
    for m in ir.topo_order(f) {
        let morph = ir.morphism(m).unwrap();
        if morph.op == Operation::LoopBack {
            continue;
        }
        let tgt = morph.target;
        if ir.try_owner(tgt) != Some(f) {
            continue;
        }
        let r = remaining[tgt];
        if r > 0 {
            let nr = r - 1;
            remaining.insert(tgt, nr);
            if nr == 0 {
                order.push(tgt);
            }
        }
    }

    order
}

// --- function-level DCE (DESIGN §1.1 / §3.1) ------------------------------

/// The functions reachable from the entry over the *post-plan* reference graph.
/// A fused `Map{g}` contributes the two bodies' own refs (they are inlined into a
/// synthesized `h`) rather than `f`/`g`; a dropped or aliased Call/Map/Fold
/// contributes nothing. Uncalled fns fall out (RW7).
pub(super) fn live_functions(ir: &CategoryIr, plan: &RewritePlan) -> SecondaryMap<FuncId, ()> {
    let mut live: SecondaryMap<FuncId, ()> = SecondaryMap::new();
    let mut stack = vec![ir.entry()];
    while let Some(f) = stack.pop() {
        if live.contains_key(f) {
            continue;
        }
        live.insert(f, ());
        push_refs(ir, plan, f, &mut stack);
    }
    live
}

/// Push the functions a live function `f` references under the plan.
pub(super) fn push_refs(ir: &CategoryIr, plan: &RewritePlan, f: FuncId, stack: &mut Vec<FuncId>) {
    let def = ir.func(f).expect("live fn");
    for &m in &def.morphisms {
        let morph = ir.morphism(m).expect("morph");
        let (target, body) = match morph.op {
            Operation::Call(g) => (morph.target, g),
            Operation::Map { body, .. } | Operation::Fold { body, .. } => (morph.target, body),
            _ => continue,
        };
        // A dropped/aliased defining edge is not replayed ⇒ no reference.
        if plan.drop.contains_key(target) || plan.alias.contains_key(target) {
            continue;
        }
        if let Some(spec) = plan.fuse.get(m) {
            // Fused: `h` inlines both bodies verbatim, referencing their own
            // Call/Map/Fold targets via fmap. `f`/`g` themselves are not kept.
            push_raw_refs(ir, spec.f, stack);
            push_raw_refs(ir, spec.g, stack);
        } else if plan.inline.contains_key(m) {
            // Inlined (Move 1): the callee is not referenced — but the inline
            // copy's own Call/Map/Fold targets are, so walk the callee's refs
            // plan-aware. Inline edges form a DAG (the pass's cycle guard), so
            // the descent terminates.
            push_refs(ir, plan, body, stack);
        } else {
            stack.push(body);
        }
    }
}

/// Push a body's raw Call/Map/Fold targets — used when the body is inlined into a
/// fused `h` (the inline copy references the originals through fmap).
pub(super) fn push_raw_refs(ir: &CategoryIr, fid: FuncId, stack: &mut Vec<FuncId>) {
    let def = ir.func(fid).expect("body fn");
    for &m in &def.morphisms {
        match ir.morphism(m).expect("morph").op {
            Operation::Call(g)
            | Operation::Map { body: g, .. }
            | Operation::Fold { body: g, .. } => stack.push(g),
            _ => {}
        }
    }
}

// --- loop lifting synthesis ------------------------------------------------
