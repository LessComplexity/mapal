//! Return-completeness verdict (L1306) and arm/return-site helpers.

use super::*;

/// Return-completeness verdict for a function (L1306). Reports into `diags`.
/// Requires, for a non-Unit declared output: ≥1 full return site (a `-> ret`
/// write, a ret-terminal loop exit, or a non-Unit fn-body tail) OR exactly-once
/// coverage of every `ret.k` slot; mixed bare/slot, duplicate slots, or
/// value-less `-> ret` fail. Unit-output fns need no return.
pub fn check_return_completeness(
    source: &str,
    fd: &syn::FnDecl,
    has_declared_output: bool,
    out_arity: Option<u64>,
    diags: &mut Vec<syn::Diagnostic>,
) {
    if !has_declared_output {
        return; // Unit output: no return required.
    }
    let mut full_sites = 0usize;
    let mut value_less_full = 0usize;
    let mut slots: Vec<u64> = Vec::new();
    let mut body_tail_returns = false;
    collect_ret_sites(
        source,
        &fd.body,
        true,
        &mut full_sites,
        &mut value_less_full,
        &mut slots,
    );
    // A non-Unit fn-body tail (a dangling head-only or value chain, not a `-> ret`)
    // returns the value (LD21/OQ8).
    if let Some(tail) = &fd.body.tail
        && tail_is_value(tail)
    {
        body_tail_returns = true;
    }

    let full = full_sites + value_less_full + usize::from(body_tail_returns);
    if !slots.is_empty() && (full_sites > 0 || body_tail_returns) {
        diags.push(diag(
            LCode::IncompleteReturn,
            fd.name.span,
            "function mixes whole-value and slot (`ret.k`) returns".to_string(),
        ));
        return;
    }
    if value_less_full > 0 {
        diags.push(diag(
            LCode::IncompleteReturn,
            fd.name.span,
            "value-less `-> ret` in a function with a non-Unit output".to_string(),
        ));
        return;
    }
    if !slots.is_empty() {
        // Slot coverage must be exactly 0..arity, each once.
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != slots.len() {
            diags.push(diag(
                LCode::IncompleteReturn,
                fd.name.span,
                "duplicate `ret.k` slot write".to_string(),
            ));
            return;
        }
        if let Some(arity) = out_arity
            && (slots.len() as u64) != arity
        {
            diags.push(diag(
                LCode::IncompleteReturn,
                fd.name.span,
                "incomplete `ret.k` slot coverage".to_string(),
            ));
        }
        return;
    }
    if full == 0 {
        diags.push(diag(
            LCode::IncompleteReturn,
            fd.name.span,
            "function with a non-Unit output has no return".to_string(),
        ));
    }
}

/// Whether a fn-body tail chain is a dangling value (returns via the virtual
/// ret continuation) — i.e. it does NOT end in an explicit `-> ret`, `-> loop`,
/// `print`, or a bare bind.
pub(super) fn tail_is_value(chain: &Chain) -> bool {
    match chain.stages.last() {
        None => chain.head.is_some(),
        Some(s) => {
            !matches!(
                s.kind,
                StageKind::Ret { .. } | StageKind::LoopJump | StageKind::Bind { .. }
            ) && !matches!(&s.kind, StageKind::Expr(e) if is_print_or_sink(e))
        }
    }
}

pub(super) fn is_print_or_sink(_e: &Expr) -> bool {
    false
}

/// Collect return sites in a function body: `-> ret` full/slot writes (incl. in
/// loop exits and guard arms — though L1405 keeps `-> ret` out of Phi arms, a
/// routing-guard exit arm's `-> ret` counts). `full_sites` counts value-bearing
/// full returns; `value_less_full` counts bare value-less `-> ret`; `slots`
/// collects each `ret.k` index.
pub(super) fn collect_ret_sites(
    source: &str,
    block: &Block,
    _top: bool,
    full_sites: &mut usize,
    value_less_full: &mut usize,
    slots: &mut Vec<u64>,
) {
    for item in &block.items {
        match item {
            BlockItem::Stmt(s) => collect_ret_stmt(source, s, full_sites, value_less_full, slots),
            BlockItem::Arm(_) => {}
        }
    }
    if let Some(t) = &block.tail {
        collect_ret_chain(source, t, full_sites, value_less_full, slots);
    }
}

pub(super) fn collect_ret_stmt(
    source: &str,
    stmt: &Stmt,
    full_sites: &mut usize,
    value_less_full: &mut usize,
    slots: &mut Vec<u64>,
) {
    match &stmt.kind {
        StmtKind::Chain(c) => collect_ret_chain(source, c, full_sites, value_less_full, slots),
        StmtKind::Loop(l) => {
            collect_ret_sites(source, &l.body, false, full_sites, value_less_full, slots)
        }
        _ => {}
    }
}

pub(super) fn collect_ret_chain(
    source: &str,
    chain: &Chain,
    full_sites: &mut usize,
    value_less_full: &mut usize,
    slots: &mut Vec<u64>,
) {
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::Ret { proj } => match proj {
                // Keep the full u64 so a slot > u32::MAX cannot spuriously
                // satisfy (or collide in) the L1306 coverage check; emit.rs is
                // the authority and rejects such a slot with L1205.
                Some((k, _)) => slots.push(*k),
                None => {
                    // value-less iff the chain is just the bare `-> ret` marker.
                    if chain.head.is_none() && chain.stages.len() == 1 {
                        *value_less_full += 1;
                    } else {
                        *full_sites += 1;
                    }
                }
            },
            StageKind::Guard(arms) => {
                for arm in arms {
                    match &arm.payload {
                        ArmPayload::Chain(c) => {
                            collect_ret_chain(source, c, full_sites, value_less_full, slots)
                        }
                        ArmPayload::Block(b) => {
                            collect_ret_sites(source, b, false, full_sites, value_less_full, slots)
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The name an exit arm's terminal binds (`… -> name`): a `Bind` stage, or a
/// final bare name (a fresh binding when it resolves to nothing, LD1 — the
/// caller keeps it only if the arm walk bound it).
pub(super) fn arm_exit_bind(payload: &ArmPayload) -> Option<syn::Name> {
    match arm_final_chain(payload)?.stages.last().map(|s| &s.kind) {
        Some(StageKind::Bind { name, .. }) => Some(*name),
        Some(StageKind::Expr(e)) => match &e.kind {
            ExprKind::Var(n) => Some(*n),
            _ => None,
        },
        _ => None,
    }
}

/// The final chain of an arm payload: a block's tail, else its last chain
/// statement.
pub(super) fn arm_final_chain(payload: &ArmPayload) -> Option<&Chain> {
    match payload {
        ArmPayload::Chain(c) => Some(c),
        ArmPayload::Block(b) => match b.items.last() {
            Some(BlockItem::Stmt(s)) => {
                if let StmtKind::Chain(c) = &s.kind {
                    if b.tail.is_some() {
                        b.tail.as_ref()
                    } else {
                        Some(c)
                    }
                } else {
                    b.tail.as_ref()
                }
            }
            _ => b.tail.as_ref(),
        },
    }
}

/// Whether an arm payload is a routing arm (headless chain ending in LoopJump).
pub fn arm_is_routing(payload: &ArmPayload) -> bool {
    let final_chain: Option<&Chain> = match payload {
        ArmPayload::Chain(c) => Some(c),
        ArmPayload::Block(b) => match b.items.last() {
            Some(BlockItem::Stmt(s)) => {
                if let StmtKind::Chain(c) = &s.kind {
                    // last item is a chain; but block tail wins if present
                    if b.tail.is_some() {
                        b.tail.as_ref()
                    } else {
                        Some(c)
                    }
                } else {
                    b.tail.as_ref()
                }
            }
            _ => b.tail.as_ref(),
        },
    };
    match final_chain {
        Some(c) => {
            c.head.is_none()
                && matches!(c.stages.last().map(|s| &s.kind), Some(StageKind::LoopJump))
        }
        None => false,
    }
}
