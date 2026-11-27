//! Syntactic scans over blocks and chains: Phi-arm contents, loop assigns, effects, routing guards.

use super::*;

/// What a Phi-arm scan found (for the L1404/L1405/L1408 checks).
pub(super) struct PhiArmScan {
    pub(super) ret_span: Option<syn::SourceLoc>,
    pub(super) effect_span: Option<syn::SourceLoc>,
    pub(super) loopjump_span: Option<syn::SourceLoc>,
    pub(super) assigns: Vec<(String, syn::SourceLoc)>,
}

pub(super) fn scan_phi_arm(
    source: &str,
    fn_sigs: &BTreeMap<String, FnSig>,
    payload: &ArmPayload,
) -> PhiArmScan {
    let mut scan = PhiArmScan {
        ret_span: None,
        effect_span: None,
        loopjump_span: None,
        assigns: Vec::new(),
    };
    match payload {
        ArmPayload::Chain(c) => scan_chain(source, fn_sigs, c, &mut scan),
        ArmPayload::Block(b) => scan_block(source, fn_sigs, b, &mut scan),
    }
    scan
}

pub(super) fn scan_block(
    source: &str,
    fn_sigs: &BTreeMap<String, FnSig>,
    block: &Block,
    scan: &mut PhiArmScan,
) {
    for item in &block.items {
        if let BlockItem::Stmt(s) = item {
            scan_stmt(source, fn_sigs, s, scan);
        }
    }
    if let Some(t) = &block.tail {
        scan_chain(source, fn_sigs, t, scan);
    }
}

pub(super) fn scan_stmt(
    source: &str,
    fn_sigs: &BTreeMap<String, FnSig>,
    stmt: &Stmt,
    scan: &mut PhiArmScan,
) {
    match &stmt.kind {
        StmtKind::Chain(c) => scan_chain(source, fn_sigs, c, scan),
        // An *indexed* bind `c[i] <- v` (ADR-0021 §2 wiring c) rebinds `c`
        // unconditionally; inside a Phi-position arm that is an enclosing-mut
        // assignment → L1408 (the check_phi_arm loop filters to enclosing muts).
        StmtKind::Bind(b) if b.index.is_some() => {
            scan.assigns
                .push((name_text(source, b.name).to_string(), stmt.span));
        }
        StmtKind::Bind(_) => {}
        StmtKind::Loop(_) => {}
        StmtKind::Error => {}
    }
}

pub(super) fn scan_chain(
    source: &str,
    fn_sigs: &BTreeMap<String, FnSig>,
    chain: &Chain,
    scan: &mut PhiArmScan,
) {
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::Ret { .. } if scan.ret_span.is_none() => {
                scan.ret_span = Some(stage.span);
            }
            StageKind::LoopJump if scan.loopjump_span.is_none() => {
                scan.loopjump_span = Some(stage.span);
            }
            StageKind::Expr(e) => {
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(source, *n);
                    // A bare-name stage that resolves to `print`/`time` or to
                    // a user-defined effectful fn is an effect in a Phi arm
                    // (L1404; tokens cannot pass a Phi). Mirror typing.rs's
                    // L1605 check.
                    let effectful = crate::is_print_builtin(text)
                        || crate::is_time_builtin(text)
                        || crate::is_read_builtin(text)
                        || fn_sigs.get(text).map(|s| s.effectful).unwrap_or(false);
                    if effectful {
                        if scan.effect_span.is_none() {
                            scan.effect_span = Some(stage.span);
                        }
                    } else {
                        // a bare-name stage may be an enclosing-mut rebind.
                        scan.assigns.push((text.to_string(), stage.span));
                    }
                }
            }
            StageKind::Guard(arms) => {
                for arm in arms {
                    match &arm.payload {
                        ArmPayload::Chain(c) => scan_chain(source, fn_sigs, c, scan),
                        ArmPayload::Block(b) => scan_block(source, fn_sigs, b, scan),
                    }
                }
            }
            // Fanout branches and seq bodies lower in the enclosing scope (pin b /
            // LD8), so an effect / `-> ret` / enclosing-mut rebind inside one is
            // still an effect / ret / rebind in the Phi arm — it would be emitted
            // unconditionally, exactly the hazard L1404/L1405/L1408 exist to catch.
            // Descend into both, matching `effect_chain` (ADR-0019 §8.10).
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    scan_chain(source, fn_sigs, b, scan);
                }
            }
            StageKind::SeqBlock(body) => scan_block(source, fn_sigs, body, scan),
            _ => {}
        }
    }
}

/// How an exit arm terminates.
pub(super) enum ExitTerminal {
    Ret,
    Bind(String, bool),
    BareValue,
}

/// Collect the names assigned (rebound in stage position) anywhere in a loop
/// body, in source order. An assignment is a stage that is a bare `Var` name
/// resolving to an existing mut (we collect all bare-name stages; the caller
/// filters to enclosing-scope muts).
pub(super) fn collect_loop_assigns(source: &str, block: &Block, out: &mut Vec<String>) {
    for item in &block.items {
        match item {
            BlockItem::Stmt(s) => collect_assigns_stmt(source, s, out),
            BlockItem::Arm(a) => collect_assigns_payload(source, &a.payload, out),
        }
    }
    if let Some(t) = &block.tail {
        collect_assigns_chain(source, t, out);
    }
}

pub(super) fn collect_assigns_stmt(source: &str, stmt: &Stmt, out: &mut Vec<String>) {
    match &stmt.kind {
        StmtKind::Chain(c) => collect_assigns_chain(source, c, out),
        // An *indexed* bind `c[i] <- v` (ADR-0021 §2 wiring b) rebinds `c` — a
        // mut array written in a loop body joins the carried set (else it is
        // dropped from the merge and the loop miscompiles). A plain bind is a
        // fresh local, not an assignment to an enclosing name.
        StmtKind::Bind(b) if b.index.is_some() => {
            out.push(name_text(source, b.name).to_string());
        }
        StmtKind::Bind(_) => {}
        StmtKind::Loop(_) => {
            // Nested-loop interiors cannot assign outer-carried muts (L1504); we
            // do NOT descend (their assignments belong to the inner loop).
        }
        StmtKind::Error => {}
    }
}

pub(super) fn collect_assigns_payload(source: &str, p: &ArmPayload, out: &mut Vec<String>) {
    match p {
        ArmPayload::Chain(c) => collect_assigns_chain(source, c, out),
        ArmPayload::Block(b) => collect_loop_assigns(source, b, out),
    }
}

pub(super) fn collect_assigns_chain(source: &str, chain: &Chain, out: &mut Vec<String>) {
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::Expr(e) => {
                if let ExprKind::Var(n) = &e.kind {
                    out.push(name_text(source, *n).to_string());
                }
            }
            StageKind::Guard(arms) => {
                for arm in arms {
                    collect_assigns_payload(source, &arm.payload, out);
                }
            }
            // Fanout branches and seq bodies lower in the enclosing scope (pin b /
            // LD8), so a rebind of an enclosing loop `mut` inside one is a
            // loop-carried assignment — it MUST reach the carried-set discovery,
            // else the state is dropped and the loop miscompiles. Descend into
            // both, matching `effect_chain` (ADR-0019 §8.10).
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    collect_assigns_chain(source, b, out);
                }
            }
            StageKind::SeqBlock(body) => collect_loop_assigns(source, body, out),
            _ => {}
        }
    }
}

/// Whether a loop body contains a direct `print` (the loop then carries a token).
/// Whether a loop body carries an effect (ATK-02): a direct `print`, **or** a
/// bare-name stage resolving (past locals) to an effectful fn. Uses the same
/// predicate as typing's `body_effect_span` (§6.1) so the loop-body effect
/// detector and the L1605 detector agree — otherwise an effectful call hoists
/// out of the loop cycle (token absent from `U`).
pub(super) fn loop_body_has_effect(
    source: &str,
    block: &Block,
    fn_sigs: &BTreeMap<String, FnSig>,
) -> bool {
    let mut found = false;
    effect_block(source, block, fn_sigs, &mut found);
    found
}

pub(super) fn effect_block(
    source: &str,
    block: &Block,
    fn_sigs: &BTreeMap<String, FnSig>,
    found: &mut bool,
) {
    for item in &block.items {
        match item {
            BlockItem::Stmt(s) => effect_stmt(source, s, fn_sigs, found),
            BlockItem::Arm(a) => effect_payload(source, &a.payload, fn_sigs, found),
        }
    }
    if let Some(t) = &block.tail {
        effect_chain(source, t, fn_sigs, found);
    }
}

pub(super) fn effect_stmt(
    source: &str,
    stmt: &Stmt,
    fn_sigs: &BTreeMap<String, FnSig>,
    found: &mut bool,
) {
    match &stmt.kind {
        StmtKind::Chain(c) => effect_chain(source, c, fn_sigs, found),
        StmtKind::Loop(l) => effect_block(source, &l.body, fn_sigs, found),
        _ => {}
    }
}

pub(super) fn effect_payload(
    source: &str,
    p: &ArmPayload,
    fn_sigs: &BTreeMap<String, FnSig>,
    found: &mut bool,
) {
    match p {
        ArmPayload::Chain(c) => effect_chain(source, c, fn_sigs, found),
        ArmPayload::Block(b) => effect_block(source, b, fn_sigs, found),
    }
}

pub(super) fn effect_chain(
    source: &str,
    chain: &Chain,
    fn_sigs: &BTreeMap<String, FnSig>,
    found: &mut bool,
) {
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::Expr(e) => {
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(source, *n);
                    let effectful = fn_sigs.get(text).map(|s| s.effectful).unwrap_or(false);
                    if crate::is_print_builtin(text)
                        || crate::is_time_builtin(text)
                        || crate::is_read_builtin(text)
                        || effectful
                    {
                        *found = true;
                    }
                }
            }
            StageKind::Guard(arms) => {
                for arm in arms {
                    effect_payload(source, &arm.payload, fn_sigs, found);
                }
            }
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    effect_chain(source, b, fn_sigs, found);
                }
            }
            StageKind::SeqBlock(body) => effect_block(source, body, fn_sigs, found),
            _ => {}
        }
    }
}

/// If a chain's final stage is a Guard with a routing arm, return its arms.
pub(super) fn chain_routing_guard(chain: &Chain) -> Option<&[GuardArm]> {
    if let Some(stage) = chain.stages.last()
        && let StageKind::Guard(arms) = &stage.kind
        && arms
            .iter()
            .any(|a| crate::typing::arm_is_routing(&a.payload))
    {
        return Some(arms);
    }
    None
}

/// Classify a routing guard's arms into (jump_arm, jump_pole, exit_arm,
/// exit_pole). Exactly one jump pole + one exit pole over {true,false} (L1409);
/// `-_->` may stand for one pole.
pub(super) fn classify_routing(
    arms: &[GuardArm],
    sp: syn::SourceLoc,
) -> ER<(&GuardArm, bool, &GuardArm, bool)> {
    // Routing arm = the one whose final chain ends in LoopJump.
    let mut jump: Option<(&GuardArm, Option<bool>)> = None;
    let mut exit: Option<(&GuardArm, Option<bool>)> = None;
    for a in arms {
        let pole = match a.discr {
            GuardDiscr::True => Some(true),
            GuardDiscr::False => Some(false),
            GuardDiscr::Default => None,
            GuardDiscr::Int(_) => {
                return Err(diag(
                    LCode::RoutingGuardArms,
                    a.discr_span,
                    "integer-discriminant routing arm is out of Core",
                ));
            }
            GuardDiscr::OutOfCore => {
                return Err(diag(LCode::UncleanTree, a.discr_span, "out-of-core arm"));
            }
        };
        if crate::typing::arm_is_routing(&a.payload) {
            if jump.is_some() {
                return Err(diag(
                    LCode::RoutingGuardArms,
                    a.discr_span,
                    "two jump arms in one routing guard",
                ));
            }
            jump = Some((a, pole));
        } else {
            if exit.is_some() {
                return Err(diag(
                    LCode::RoutingGuardArms,
                    a.discr_span,
                    "two exit arms in one routing guard",
                ));
            }
            exit = Some((a, pole));
        }
    }
    let (jump_arm, jpole) =
        jump.ok_or_else(|| diag(LCode::LoopNoExit, sp, "routing guard has no jump arm"))?;
    let (exit_arm, epole) =
        exit.ok_or_else(|| diag(LCode::LoopNoExit, sp, "routing guard has no exit arm"))?;
    if arms.len() != 2 {
        return Err(diag(
            LCode::RoutingGuardArms,
            sp,
            "routing guard must have exactly two arms",
        ));
    }
    // Resolve poles: each must be a distinct member of {true,false}; a Default
    // stands for whichever pole the other arm does not occupy.
    let (jp, ep) = match (jpole, epole) {
        (Some(j), Some(e)) => {
            if j == e {
                return Err(diag(
                    LCode::RoutingGuardArms,
                    sp,
                    "both routing arms share a pole",
                ));
            }
            (j, e)
        }
        (Some(j), None) => (j, !j),
        (None, Some(e)) => (!e, e),
        (None, None) => {
            return Err(diag(
                LCode::RoutingGuardArms,
                sp,
                "routing guard with no concrete poles",
            ));
        }
    };
    Ok((jump_arm, jp, exit_arm, ep))
}

/// Whether an expression produces a fresh value via a primitive that can take a
/// `Dest` (so `<expr> -> ret` can write Return directly; pin 2). A bare
/// `Var`/literal is a pre-existing wire (uses `output()` for `-> ret`); a
/// negative-literal fold is a fresh Constant and still uses `output()` (it has
/// no producing primitive to redirect), so it is NOT value-producing here.
pub(super) fn expr_is_value_producing(kind: &ExprKind) -> bool {
    match kind {
        ExprKind::Binary { .. }
        | ExprKind::Member { .. }
        | ExprKind::Index { .. }
        | ExprKind::Tuple(_)
        | ExprKind::Array(_)
        | ExprKind::Struct { .. } => true,
        ExprKind::Unary { op, operand, .. } => {
            // A folded negative literal is a Constant (pre-existing); a real
            // unop is value-producing.
            !matches!(
                (op, &operand.kind),
                (UnOp::Neg, ExprKind::Int(_)) | (UnOp::Neg, ExprKind::Float)
            )
        }
        _ => false,
    }
}

/// Whether a chain's final stage is a `-> ret` marker.
pub(super) fn chain_ends_in_ret(chain: &Chain) -> bool {
    matches!(
        chain.stages.last().map(|s| &s.kind),
        Some(StageKind::Ret { .. })
    )
}

/// Whether an exit-arm payload terminates in `-> ret` (for the L1504 nested
/// inner-exit-via-ret check). For a `;`-terminated Block (`tail: None`) the
/// terminal is the LAST chain item (LD21: statement-vs-tail identical; ATK-11).
pub(super) fn exit_payload_is_ret(payload: &ArmPayload) -> bool {
    let chain: Option<&Chain> = match payload {
        ArmPayload::Chain(c) => Some(c),
        ArmPayload::Block(b) => match b.tail.as_ref() {
            Some(t) => Some(t),
            None => b.items.iter().rev().find_map(|it| match it {
                BlockItem::Stmt(s) => match &s.kind {
                    StmtKind::Chain(c) => Some(c),
                    _ => None,
                },
                _ => None,
            }),
        },
    };
    match chain {
        Some(c) => matches!(
            c.stages.last().map(|s| &s.kind),
            Some(StageKind::Ret { .. })
        ),
        None => false,
    }
}

/// The first `ret.k` slot a loop exit arm block writes as a statement, if any
/// ([`Emitter::slot_exit_value`]).
/// The slot a `… -> ret.k` chain writes.
pub(super) fn ret_slot_of(c: &Chain) -> Option<u64> {
    match c.stages.last().map(|st| &st.kind) {
        Some(StageKind::Ret { proj: Some((k, _)) }) => Some(*k),
        _ => None,
    }
}

pub(super) fn slot_exit(b: &Block) -> Option<u64> {
    b.items.iter().find_map(|item| match item {
        BlockItem::Stmt(s) => match &s.kind {
            StmtKind::Chain(c) => match c.stages.last().map(|st| &st.kind) {
                Some(StageKind::Ret { proj: Some((k, _)) }) => Some(*k),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    })
}
