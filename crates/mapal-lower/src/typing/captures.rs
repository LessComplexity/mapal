//! Map/fold body captures: reads, mutation captures and effect spans.

use super::*;

/// The span of the first effectful stage (a direct `print`, or a call to an
/// effectful fn) inside a map/fold body (L1605), if any.
pub(super) fn body_effect_span(
    source: &str,
    block: &Block,
    fn_sigs: &BTreeMap<String, FnSig>,
) -> Option<syn::SourceLoc> {
    let mut found: Option<syn::SourceLoc> = None;
    walk_block_stages(source, block, &mut |stage| {
        if found.is_some() {
            return;
        }
        if let StageKind::Expr(e) = &stage.kind
            && let ExprKind::Var(n) = &e.kind
        {
            let text = name_text(source, *n);
            let effectful = fn_sigs.get(text).map(|s| s.effectful).unwrap_or(false);
            if crate::is_print_builtin(text)
                || crate::is_time_builtin(text)
                || crate::is_read_builtin(text)
                || effectful
            {
                found = Some(stage.span);
            }
        }
    });
    found
}

/// The free variables of a map/fold body (ADR-0027): **read captures** —
/// names resolving to neither a block param, a body-local binding, nor a
/// global fn/builtin — collected in first-use order. A **write** to an
/// enclosing name (an indexed bind `c[i] <- v` targeting it — ADR-0021's
/// "indexed bind is a rebind, never a fresh shadow") is a mutation capture,
/// rejected by the caller with L1108.
#[derive(Default)]
pub(super) struct BodyCaptures {
    /// Read captures in first-use order (deduped).
    pub(super) reads: Vec<(String, syn::SourceLoc)>,
    seen: std::collections::BTreeSet<String>,
    /// The first write-to-enclosing-name found (name, span), if any.
    pub(super) write: Option<(String, syn::SourceLoc)>,
    /// True while descending into a NESTED map/fold body: reads still collect
    /// (transitive captures, ADR-0027 Q3) but writes are NOT recorded here —
    /// the inner body's own `body_captures` (run by its own typing pass)
    /// owns the L1108, and recording it twice would duplicate the diagnostic.
    nested: bool,
}

impl BodyCaptures {
    fn read(&mut self, name: &str, span: syn::SourceLoc) {
        if self.seen.insert(name.to_string()) {
            self.reads.push((name.to_string(), span));
        }
    }

    /// Record a write-to-enclosing-name (kept: the first one) — unless walking
    /// a nested body (see `nested`).
    fn record_write(&mut self, name: &str, span: syn::SourceLoc) {
        if !self.nested && self.write.is_none() {
            self.write = Some((name.to_string(), span));
        }
    }
}

/// Collect the body's free variables (ADR-0027). L1108's old role — "the
/// body may not reference an enclosing local (blocks are not closures)" — is
/// replaced: reads become captures; writes remain rejected.
/// The enclosing names `block` reads, in first-use order (the free variables
/// a loop exit arm needs from its loop: [`crate::emit`]'s deferred exit).
pub(crate) fn block_reads(
    source: &str,
    block: &Block,
    fn_sigs: &BTreeMap<String, FnSig>,
) -> Vec<String> {
    body_captures(source, block, &Default::default(), fn_sigs)
        .reads
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

pub(super) fn body_captures(
    source: &str,
    block: &Block,
    params: &std::collections::BTreeSet<&str>,
    fn_sigs: &BTreeMap<String, FnSig>,
) -> BodyCaptures {
    let mut local: std::collections::BTreeSet<String> =
        params.iter().map(|s| s.to_string()).collect();
    let mut acc = BodyCaptures::default();
    capture_block(source, block, &mut local, fn_sigs, &mut acc);
    acc
}

pub(super) fn capture_block(
    source: &str,
    block: &Block,
    local: &mut std::collections::BTreeSet<String>,
    fn_sigs: &BTreeMap<String, FnSig>,
    acc: &mut BodyCaptures,
) {
    for item in &block.items {
        if let BlockItem::Stmt(s) = item {
            capture_stmt(source, s, local, fn_sigs, acc);
        }
    }
    if let Some(t) = &block.tail {
        capture_chain(source, t, local, fn_sigs, acc);
    }
}

pub(super) fn capture_stmt(
    source: &str,
    stmt: &Stmt,
    local: &mut std::collections::BTreeSet<String>,
    fn_sigs: &BTreeMap<String, FnSig>,
    acc: &mut BodyCaptures,
) {
    match &stmt.kind {
        StmtKind::Chain(c) => capture_chain(source, c, local, fn_sigs, acc),
        // An *indexed* bind `c[i] <- v` (ADR-0021) is a rebind, not a fresh
        // shadow: targeting an enclosing name is a WRITE to a captured
        // variable — the one remaining L1108 case (reads are legal captures;
        // writes are not). The index expr + value are ordinary reads.
        StmtKind::Bind(b) if b.index.is_some() => {
            let name = name_text(source, b.name);
            if !local.contains(name) && fn_sigs.get(name).is_none() {
                acc.record_write(name, b.name.span);
            }
            if let Some(idx) = &b.index {
                capture_expr(source, idx, local, fn_sigs, acc);
            }
            capture_expr(source, &b.value, local, fn_sigs, acc);
        }
        StmtKind::Bind(b) => {
            capture_expr(source, &b.value, local, fn_sigs, acc);
            local.insert(name_text(source, b.name).to_string());
        }
        // Loops inside map/fold bodies are Core-legal (bodies are ordinary
        // token-free fns; ATK-14). Descend into the loop body with the set
        // saved/restored — the emitter pushes a scope per loop
        // (emit.rs:2509/2549), so a loop-local binding must not leak outward
        // (a later read of the enclosing name is a capture, never the
        // shadow). The exit arm's Bind names DO register afterward (§8.5: the
        // exit value binds in the loop's enclosing scope, so a read after the
        // loop resolves to it — never a capture).
        StmtKind::Loop(l) => {
            let saved = local.clone();
            capture_block(source, &l.body, local, fn_sigs, acc);
            *local = saved;
            for name in loop_exit_bind_names(source, &l.body) {
                local.insert(name);
            }
        }
        StmtKind::Error => {}
    }
}

/// The names a loop's routing-guard exit arm binds into the loop's ENCLOSING
/// scope (§8.5's `bind_new_enclosing`): the terminal `-> name` of the exit
/// arm's final chain. Everything else bound inside the loop body lives in the
/// popped loop-body frame, so the capture walk registers only these.
pub(super) fn loop_exit_bind_names(source: &str, body: &Block) -> Vec<String> {
    // The routing guard is the body's tail, else its last chain statement.
    let final_chain: Option<&Chain> = body.tail.as_ref().or_else(|| {
        body.items.iter().rev().find_map(|it| match it {
            BlockItem::Stmt(s) => match &s.kind {
                StmtKind::Chain(c) => Some(c),
                _ => None,
            },
            _ => None,
        })
    });
    let Some(chain) = final_chain else {
        return Vec::new();
    };
    let Some(Stage {
        kind: StageKind::Guard(arms),
        ..
    }) = chain.stages.last()
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for arm in arms {
        if arm_is_routing(&arm.payload) {
            continue; // the jump arm carries no exit binding
        }
        // The arm's final chain (tail wins over the last chain statement,
        // mirroring `arm_is_routing`).
        let c: &Chain = match &arm.payload {
            ArmPayload::Chain(c) => c,
            ArmPayload::Block(b) => {
                let last_chain = b.items.iter().rev().find_map(|it| match it {
                    BlockItem::Stmt(s) => match &s.kind {
                        StmtKind::Chain(c) => Some(c),
                        _ => None,
                    },
                    _ => None,
                });
                match b.tail.as_ref().or(last_chain) {
                    Some(c) => c,
                    None => continue,
                }
            }
        };
        if let Some(stage) = c.stages.last()
            && let StageKind::Bind { name, .. } = &stage.kind
        {
            out.push(name_text(source, *name).to_string());
        }
    }
    out
}

pub(super) fn capture_chain(
    source: &str,
    chain: &Chain,
    local: &mut std::collections::BTreeSet<String>,
    fn_sigs: &BTreeMap<String, FnSig>,
    acc: &mut BodyCaptures,
) {
    if let Some(h) = &chain.head {
        capture_expr(source, h, local, fn_sigs, acc);
    }
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::Expr(e) => {
                // a bare-name stage that binds a fresh local registers it —
                // but targeting an already-captured (read) name is a rebind of
                // a read-only capture (D2b: a captured name is never a rebind
                // target): record the WRITE so the caller reports the teaching
                // L1108 (naming the variable) instead of falling through to
                // emit's generic L1104.
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(source, *n);
                    if !local.contains(text)
                        && fn_sigs.get(text).is_none()
                        && !crate::is_print_builtin(text)
                        && !crate::is_time_builtin(text)
                        && !crate::is_read_builtin(text)
                    {
                        if acc.seen.contains(text) {
                            acc.record_write(text, e.span);
                        } else {
                            local.insert(text.to_string());
                        }
                    }
                } else {
                    capture_expr(source, e, local, fn_sigs, acc);
                }
            }
            StageKind::OpShorthand { expr } => {
                capture_expr(source, expr, local, fn_sigs, acc);
            }
            StageKind::Bind { name, .. } => {
                local.insert(name_text(source, *name).to_string());
            }
            StageKind::Guard(arms) => {
                // Each arm lowers in its own child scope (emit.rs:2045-2047),
                // so a bind/shadow inside one arm must not leak into a sibling
                // arm or the enclosing body — save/restore per arm.
                for arm in arms {
                    let saved = local.clone();
                    match &arm.payload {
                        ArmPayload::Chain(c) => capture_chain(source, c, local, fn_sigs, acc),
                        ArmPayload::Block(b) => capture_block(source, b, local, fn_sigs, acc),
                    }
                    *local = saved;
                }
            }
            // Fanout branches and seq bodies lower in the enclosing scope (pin b /
            // LD8), so a capture of an enclosing local inside one is still a
            // capture (L1108), and a bind inside one registers a body-local.
            // Descend into both, matching `effect_chain` (ADR-0019 §8.10);
            // otherwise a real capture surfaces later as a misleading L1101.
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    capture_chain(source, b, local, fn_sigs, acc);
                }
            }
            StageKind::SeqBlock(body) => {
                capture_block(source, body, local, fn_sigs, acc);
            }
            // A nested map/fold body (ADR-0027 Q3 — transitive captures): its
            // params and locals are scoped to IT, not to the enclosing body —
            // descend with the set saved/restored, collecting free names that
            // only the enclosing scope can provide (they become THIS body's
            // captures too, resolved from the enclosing scope's bindings).
            // WRITE recording is suppressed during the descent: the inner
            // body's own typing pass owns the L1108 (recording here too would
            // report the same write twice).
            StageKind::MapFold { params, body, .. } => {
                let saved = local.clone();
                for p in params {
                    local.insert(name_text(source, *p).to_string());
                }
                let nested = acc.nested;
                acc.nested = true;
                capture_block(source, body, local, fn_sigs, acc);
                acc.nested = nested;
                *local = saved;
            }
            _ => {}
        }
    }
}

pub(super) fn capture_expr(
    source: &str,
    expr: &Expr,
    local: &std::collections::BTreeSet<String>,
    fn_sigs: &BTreeMap<String, FnSig>,
    acc: &mut BodyCaptures,
) {
    // LIFO stack with children pushed in REVERSE source order, so leaves pop
    // (and record) in source order of first use (ADR-0027 §Determinism).
    enum Work<'a> {
        E(&'a Expr),
        /// A struct-pun field (`{ r }` reads `r`): the name + its span.
        Pun(&'a str, syn::SourceLoc),
    }
    let mut stack: Vec<Work> = vec![Work::E(expr)];
    while let Some(w) = stack.pop() {
        let e = match w {
            Work::E(e) => e,
            Work::Pun(fname, span) => {
                if !local.contains(fname) && fn_sigs.get(fname).is_none() {
                    acc.read(fname, span);
                }
                continue;
            }
        };
        match &e.kind {
            ExprKind::Var(n) => {
                let text = name_text(source, *n);
                if !local.contains(text) && fn_sigs.get(text).is_none() {
                    acc.read(text, e.span);
                }
            }
            ExprKind::Unary { operand, .. } => stack.push(Work::E(operand)),
            ExprKind::Binary { lhs, rhs, .. } => {
                stack.push(Work::E(rhs));
                stack.push(Work::E(lhs));
            }
            ExprKind::Member { base, .. } => stack.push(Work::E(base)),
            ExprKind::Index { base, index } => {
                stack.push(Work::E(index));
                stack.push(Work::E(base));
            }
            ExprKind::Tuple(es) | ExprKind::Array(es) => {
                stack.extend(es.iter().rev().map(Work::E));
            }
            ExprKind::Struct { fields, .. } => {
                for fi in fields.iter().rev() {
                    match &fi.value {
                        // pun shorthand `{ r }` reads `r` — a capture if not local.
                        Some(v) => stack.push(Work::E(v)),
                        None => stack.push(Work::Pun(name_text(source, fi.name), fi.span)),
                    }
                }
            }
            _ => {}
        }
    }
}

/// Walk every stage in a block (statements, tails, nested guards/fanouts),
/// invoking `f`. Does NOT descend into nested map/fold bodies.
pub(super) fn walk_block_stages(source: &str, block: &Block, f: &mut impl FnMut(&Stage)) {
    for item in &block.items {
        if let BlockItem::Stmt(s) = item {
            walk_stmt_stages(source, s, f);
        }
    }
    if let Some(t) = &block.tail {
        walk_chain_stages(source, t, f);
    }
}

pub(super) fn walk_stmt_stages(source: &str, stmt: &Stmt, f: &mut impl FnMut(&Stage)) {
    match &stmt.kind {
        StmtKind::Chain(c) => walk_chain_stages(source, c, f),
        StmtKind::Loop(l) => walk_block_stages(source, &l.body, f),
        _ => {}
    }
}

pub(super) fn walk_chain_stages(source: &str, chain: &Chain, f: &mut impl FnMut(&Stage)) {
    for stage in &chain.stages {
        f(stage);
        match &stage.kind {
            StageKind::Guard(arms) => {
                for arm in arms {
                    match &arm.payload {
                        ArmPayload::Chain(c) => walk_chain_stages(source, c, f),
                        ArmPayload::Block(b) => walk_block_stages(source, b, f),
                    }
                }
            }
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    walk_chain_stages(source, b, f);
                }
            }
            StageKind::SeqBlock(body) => walk_block_stages(source, body, f),
            _ => {}
        }
    }
}
