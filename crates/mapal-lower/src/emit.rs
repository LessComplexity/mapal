//! Pass C + D2/D3 + E — declaration, the §8 emission recipes, and seal.
//!
//! Emission walks statements in source order maintaining the scope stack, the
//! wire (`cur: Option<ObjectId>` within a chain), and the current token
//! (effectful fns). Every builder call passes the `SourceLoc` of the surface
//! node driving it (§10). The builder is the second line of type checking
//! (LD12): a `TypeMismatch`/`NotAProduct`/etc. that corresponds to a user
//! condition is mapped to its L-code; anything truly unexpected is L1901.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use mapal_ir::{
    CategoryIr, Dest, FnBuilder, FuncId, FuncKind, IrBuilder, IrError, ObjectId, Operation, Ty,
    Value,
};
use mapal_syntax::{
    self as syn, ArmPayload, BinOp, Block, BlockItem, Chain, CollOp, Expr, ExprKind, GuardArm,
    GuardDiscr, MemberField, Program, Stage, StageKind, Stmt, StmtKind, UnOp,
};

use crate::diag::{LCode, diag};
use crate::effects::Effects;
use crate::formats::{Arg, Format, Realize, Verb};
use crate::scope::ScopeStack;
use crate::sizes::{Generics, Instance, SiteKey, SizeEnv};
use crate::typing::{FnSig, TypeInfo};
use crate::tys::{TypeTable, ir_loc, name_text, src_slice};

mod builders;
mod chains;
mod collections;
mod decl;
mod exprs;
mod guards;
mod loop_exits;
mod loops;
mod scan;
mod stmts;

use self::scan::*;

/// ADR-0047 D5: the instances a round declares, by generic function (sorted
/// size tuples with their names), and the log emission keeps of its calls.
pub struct Mono<'a> {
    pub generics: &'a Generics<'a>,
    pub instances: &'a BTreeMap<String, Vec<(Vec<Arg>, String)>>,
    pub log: &'a RefCell<CallLog>,
    /// The typing walk erred where a generic call was untyped: emit every
    /// unit as far as it goes, to log the calls (the result is discarded).
    pub probe: bool,
    /// ADR-0048 D3: when set, the object each name `main` binds (the
    /// compile-time evaluation of top-level bindings reads them).
    pub bound: Option<&'a RefCell<BTreeMap<String, ObjectId>>>,
}

/// What emission found at the calls to generic functions: each site's binding
/// (from the builder's exact argument type), the instances called, and the
/// ones called but not declared this round.
#[derive(Default)]
pub struct CallLog {
    pub sites: BTreeMap<SiteKey, Vec<Arg>>,
    pub called: BTreeSet<Instance>,
    pub requested: BTreeSet<Instance>,
    /// ADR-0051 D4: the `widen` conversions exact only on a value range,
    /// proven by the facts once the IR is sealed ([`crate::convert`]).
    pub widens: Vec<crate::convert::WidenCheck>,
}

/// A surface-bound symbol (DESIGN §3).
#[derive(Clone)]
pub struct Binding {
    pub obj: ObjectId,
    pub ty: Ty,
    pub mutable: bool,
    /// Declaration sequence number (params first, then bindings in source
    /// order). Carried-state order is declaration order (LD4); a rebind keeps
    /// the original `decl_seq`.
    pub decl_seq: u32,
}

/// A short alias for the per-function diagnostic-or-id result. Emission is
/// fail-fast (LD12): the first error aborts via `?`.
type ER<T> = Result<T, syn::Diagnostic>;

/// The per-function emission context.
struct Emitter<'a> {
    source: &'a str,
    types: &'a TypeTable,
    effects: &'a Effects,
    fn_ids: &'a BTreeMap<String, FuncId>,
    fn_sigs: &'a BTreeMap<String, FnSig>,
    info: &'a TypeInfo,
    /// stage span → declared+built body FuncId (D2).
    body_ids: &'a BTreeMap<(u32, u32), FuncId>,
    owner_name: String,
    /// current token register (effectful fns); None for pure fns.
    token: Option<ObjectId>,
    /// the scope stack of surface-bound symbols (DESIGN §3).
    scope: ScopeStack<Binding>,
    /// object → its ty (the builder does not expose object tys, so we track them
    /// for branch decisions; the builder remains the authority on correctness).
    obj_ty: BTreeMap<ObjectId, Ty>,
    /// loop nesting depth (for the inner-exit-via-ret restriction, §8.5/LD20).
    loop_depth: u32,
    /// monotonic declaration sequence counter (carried-state order = declaration
    /// order, LD4).
    decl_seq: u32,
    /// objects derived from the current loop's merge view (for L1503): seeded at
    /// the carried-name merge projs + token, propagated through primitives, and
    /// killed by a merge-independent rebind (§7.3).
    derived: std::collections::BTreeSet<ObjectId>,
    /// whether any enclosing loop carries a token (a nested loop inside one is
    /// L1504 — the token cannot get a third route consumer).
    enclosing_token_loop: bool,
    /// names carried by an enclosing loop (an inner loop assigning one is L1504).
    outer_carried: std::collections::BTreeSet<String>,
    /// The current map/fold body's read-capture names (ADR-0027; empty outside
    /// a body). A `-> name` targeting one is a fresh body-local shadow, never
    /// a rebind of the shared capture (D2b — a rebind after a read is already
    /// the typing walk's teaching L1108).
    captures: std::collections::BTreeSet<String>,
    /// mask-frontend: each struct this emitter packed and its components in
    /// field order — a nested body's field capture through one takes the
    /// component itself, not a projection of the pack (a guard arm's
    /// projection is not forwarded before the arm resolves).
    packed: BTreeMap<ObjectId, Vec<ObjectId>>,
    /// ADR-0041 D2: the precisions this scope's products admit (the innermost
    /// `: compute` around it; `{f32}` without one).
    prec: mapal_ir::Precision,
    /// ADR-0047: the generic functions, the round's instances and call log.
    mono: &'a Mono<'a>,
    /// ADR-0047 D4: the instance's size values (empty in a fixed-size fn).
    sizes: &'a SizeEnv,
}

/// A body's `: compute` as the IR's set; `outer` where it has none.
fn ir_prec(p: Option<syn::Precision>, outer: mapal_ir::Precision) -> mapal_ir::Precision {
    p.map_or(outer, |p| mapal_ir::Precision {
        f32: p.f32,
        tf32: p.tf32,
        bf16: p.bf16,
        f16: p.f16,
    })
}

/// Map a builder `IrError` to an L-code diagnostic at `span`. User-diagnosable
/// conditions get their specific code; anything else is L1901 (a lower bug).
fn ir_err(e: IrError, span: syn::SourceLoc) -> syn::Diagnostic {
    let (code, msg): (LCode, String) = match &e {
        IrError::TypeMismatch {
            expected, found, ..
        } => (
            LCode::TypeMismatch,
            format!(
                "type mismatch: expected `{}`, found `{}`",
                ty_name(expected),
                ty_name(found)
            ),
        ),
        IrError::NotAProduct => (
            LCode::NotAProduct,
            "operation on a non-product value".to_string(),
        ),
        IrError::SlotOutOfRange => (LCode::SlotOutOfRange, "slot index out of range".to_string()),
        IrError::StrOutsidePrint => (
            LCode::StrOutsidePrint,
            "string value used outside `print`".to_string(),
        ),
        IrError::EmptyProduct => (LCode::EmptyArray, "empty product/array".to_string()),
        IrError::TyTooDeep => (LCode::TypeTooDeep, "type nests too deeply".to_string()),
        IrError::RetTypeMismatch => (
            LCode::TypeMismatch,
            "return value type does not match the declared output".to_string(),
        ),
        // A token inside a map/fold body (e.g. a token-typed capture — the
        // surface can't name tokens, so this is IrBuilder-driven): the same
        // user-facing family as L1605 (bodies are effect-free), never the
        // internal catch-all.
        IrError::TokenInBody => (
            LCode::BodyEffectful,
            "effect token inside a map/fold body: bodies are pure (effects belong in the enclosing function)".to_string(),
        ),
        IrError::InvalidWiden => (
            LCode::InvalidWiden,
            "illegal widening pair; legal lattice: i32→i64, i32→f32, i32→f64, f32→f64"
                .to_string(),
        ),
        IrError::InvalidExp => (
            LCode::InvalidExp,
            "a math builtin (`exp`, `sqrt`, `sin`, `cos`) takes an `f32` or `f64`".to_string(),
        ),
        IrError::InvalidRead => (
            LCode::ReadAnnotation,
            "`read` yields a one-dimensional array of i32, i64, u8, f32, f64, bf16 or f16"
                .to_string(),
        ),
        IrError::InvalidNarrow => (
            LCode::InvalidNarrow,
            "`to<bf16>` / `to<f16>` take an `f32`".to_string(),
        ),
        other => (
            LCode::Internal,
            format!("internal lowering error: {other:?}"),
        ),
    };
    diag(code, span, msg)
}

/// Render a ty for diagnostics (delegates to mapal-ir's label fn).
pub fn ty_name(ty: &Ty) -> String {
    mapal_ir::ty_label(ty)
}

/// Lower a parse-clean program to a sealed IR (the crate entry; DESIGN §1).
pub fn lower_program(
    source: &str,
    program: &Program,
    types: &TypeTable,
    effects: &Effects,
    fn_sigs: &BTreeMap<String, FnSig>,
    typing: &BTreeMap<String, TypeInfo>,
    mono: &Mono,
) -> Result<CategoryIr, Vec<syn::Diagnostic>> {
    let mut b = IrBuilder::new();

    // The units to declare and build, in source order: a fixed-size fn, or
    // each instance of a generic one at its position, by size tuple (ADR-0047
    // D6), with its size environment.
    let mut units: Vec<(&syn::FnDecl, String, SizeEnv)> = Vec::new();
    for item in &program.items {
        if let syn::Item::Fn(fd) = item {
            let name = name_text(source, fd.name).to_string();
            if fd.generics.is_empty() {
                units.push((fd, name, types.base_env().clone()));
            } else {
                for (vs, iname) in mono.instances.get(&name).into_iter().flatten() {
                    units.push((
                        fd,
                        iname.clone(),
                        crate::sizes::env_of(source, &fd.generics, vs, types.base_env()),
                    ));
                }
            }
        }
    }

    // Pass C: declare every surface fn in source order with its IR tys (§6.2).
    let mut fn_ids: BTreeMap<String, FuncId> = BTreeMap::new();
    let mut main_id: Option<FuncId> = None;
    for (fd, name, _) in &units {
        let sig = fn_sigs.get(name).expect("fn sig");
        let (in_ty, out_ty) = ir_signature(sig);
        let fid = b
            .declare(FuncKind::Named, name, in_ty, out_ty, ir_loc(fd.span))
            .map_err(|e| vec![ir_err(e, fd.span)])?;
        if name == "main" {
            main_id = Some(fid);
        }
        fn_ids.insert(name.clone(), fid);
    }
    let main_id = main_id.ok_or_else(|| {
        vec![diag(
            LCode::NoMain,
            program.span,
            "no `fn main`".to_string(),
        )]
    })?;

    // Pass D: per fn in source order — D2 (bodies) then D3 (outer). A probe
    // (ADR-0047 D5) keeps going past a unit's error and returns the first.
    let mut first_err: Option<Vec<syn::Diagnostic>> = None;
    for (fd, name, env) in &units {
        let fd = *fd;
        let info = typing.get(name).expect("typing info");
        // An error inside an instance names it (ADR-0047 D5).
        let note = |mut d: syn::Diagnostic| {
            if !fd.generics.is_empty() {
                d.message = format!("{} (in `{name}`)", d.message);
            }
            vec![d]
        };

        // D2: declare + build every map/fold body, innermost-first.
        let mut body_ids: BTreeMap<(u32, u32), FuncId> = BTreeMap::new();
        let prec = ir_prec(fd.body.compute, mapal_ir::Precision::F32);
        // ADR-0048 D3: the bodies of the top-level chains this function
        // reads (emitted at its outermost scope), at the default precision.
        let tops = types.globals().used_by(source, fd);
        if !tops.is_empty() {
            let block = types.globals().chains_block(&tops);
            let built = build_bodies(
                &Shared {
                    source,
                    types,
                    effects,
                    fn_ids: &fn_ids,
                    fn_sigs,
                    info,
                    owner: name,
                    mono,
                    sizes: env,
                },
                &block,
                &mut b,
                &mut body_ids,
                mapal_ir::Precision::F32,
            )
            .map_err(note);
            if let Err(e) = built {
                if !mono.probe {
                    return Err(e);
                }
                first_err.get_or_insert(e);
            }
        }
        let built = build_bodies(
            &Shared {
                source,
                types,
                effects,
                fn_ids: &fn_ids,
                fn_sigs,
                info,
                owner: name,
                mono,
                sizes: env,
            },
            &fd.body,
            &mut b,
            &mut body_ids,
            prec,
        )
        .map_err(note);
        if let Err(e) = built {
            if !mono.probe {
                return Err(e);
            }
            first_err.get_or_insert(e);
        }

        // D3: build the outer fn body.
        let fid = fn_ids[name];
        let mut em = Emitter {
            source,
            types,
            effects,
            fn_ids: &fn_ids,
            fn_sigs,
            info,
            body_ids: &body_ids,
            owner_name: name.clone(),
            token: None,
            scope: ScopeStack::new(),
            obj_ty: BTreeMap::new(),
            loop_depth: 0,
            decl_seq: 0,
            derived: std::collections::BTreeSet::new(),
            enclosing_token_loop: false,
            outer_carried: std::collections::BTreeSet::new(),
            captures: std::collections::BTreeSet::new(),
            packed: BTreeMap::new(),
            prec,
            mono,
            sizes: env,
        };
        let fb = b.build_fn(fid).map_err(|e| vec![ir_err(e, fd.span)])?;
        if let Err(e) = em.emit_fn(fd, fb).map_err(note) {
            if !mono.probe {
                return Err(e);
            }
            first_err.get_or_insert(e);
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }

    // Pass E: seal.
    b.seal(main_id).map_err(|e| {
        vec![diag(
            LCode::Internal,
            program.span,
            format!("seal failed: {e:?}"),
        )]
    })
}

/// The IR signature (input, output tys) for a surface fn per §6.2.
fn ir_signature(sig: &FnSig) -> (Ty, Ty) {
    let a: Option<Ty> = if sig.params.is_empty() {
        None
    } else if sig.params.len() == 1 {
        Some(sig.params[0].clone())
    } else {
        Some(Ty::Tuple(sig.params.clone()))
    };
    let b = sig.ret.clone();
    if !sig.effectful {
        let in_ty = a.unwrap_or(Ty::Unit);
        let out_ty = b.unwrap_or(Ty::Unit);
        (in_ty, out_ty)
    } else {
        let in_ty = match a {
            Some(a) => Ty::Tuple(vec![Ty::IoToken, a]),
            None => Ty::IoToken,
        };
        let out_ty = match b {
            Some(b) => Ty::Tuple(vec![Ty::IoToken, b]),
            None => Ty::IoToken,
        };
        (in_ty, out_ty)
    }
}

/// ADR-0048 D5: what an error message says about `_` at a value of type
/// `before` that was read as `after` — `_` read through, or (`before ==
/// after`) a struct with a `_` none of whose chain fits.
fn default_note(before: &Ty, after: &Ty) -> Option<String> {
    let label = |t: &Ty| mapal_ir::ty_label(t);
    if before != after {
        return Some(format!(
            "`_` was applied: the `{}` value was read as its `_`, `{}`",
            label(before),
            label(after)
        ));
    }
    let inner = TypeTable::default_of(before)?;
    Some(format!(
        "`{}` has a default field `_: {}`, but neither fits here",
        label(before),
        label(inner)
    ))
}

/// What every body built for one function (or instance) shares.
struct Shared<'a> {
    source: &'a str,
    types: &'a TypeTable,
    effects: &'a Effects,
    fn_ids: &'a BTreeMap<String, FuncId>,
    fn_sigs: &'a BTreeMap<String, FnSig>,
    info: &'a TypeInfo,
    owner: &'a str,
    mono: &'a Mono<'a>,
    sizes: &'a SizeEnv,
}

/// D2: declare and build every map/fold body in `block`, innermost-first.
fn build_bodies(
    sh: &Shared,
    block: &Block,
    b: &mut IrBuilder,
    out: &mut BTreeMap<(u32, u32), FuncId>,
    outer: mapal_ir::Precision,
) -> ER<()> {
    let Shared {
        source,
        types,
        effects,
        fn_ids,
        fn_sigs,
        info,
        owner,
        mono,
        sizes,
    } = *sh;
    // Find every MapFold stage by walking the tree; for each, recurse into its
    // body first (innermost-first), then declare+build it.
    let stages = collect_mapfold_stages(block);
    for (stage, body, params, op, around) in stages {
        // ADR-0041 D2: innermost wins — the step's own, a block's around it, then `outer`.
        let prec = ir_prec(body.compute, ir_prec(around, outer));
        // Recurse into nested map/fold inside this body first.
        build_bodies(sh, body, b, out, prec)?;

        let sig = info
            .block(stage.span)
            .cloned()
            .ok_or_else(|| internal(stage.span, "missing block signature"))?;
        let body_out_ty = sig.body_out.clone();
        let body_in_ty = sig.body_in.clone();
        let kind = match op {
            CollOp::Map => FuncKind::MapBody,
            CollOp::Fold => FuncKind::FoldBody,
        };
        let bid = b
            .declare(
                kind,
                &sig.name,
                body_in_ty.clone(),
                body_out_ty.clone(),
                ir_loc(stage.span),
            )
            .map_err(|e| ir_err(e, stage.span))?;
        out.insert((stage.span.start, stage.span.end), bid);

        // Build the body fn.
        let mut em = Emitter {
            source,
            types,
            effects,
            fn_ids,
            fn_sigs,
            info,
            body_ids: out,
            owner_name: owner.to_string(),
            token: None,
            scope: ScopeStack::new(),
            obj_ty: BTreeMap::new(),
            loop_depth: 0,
            decl_seq: 0,
            derived: std::collections::BTreeSet::new(),
            enclosing_token_loop: false,
            outer_carried: std::collections::BTreeSet::new(),
            captures: std::collections::BTreeSet::new(),
            packed: BTreeMap::new(),
            prec,
            mono,
            sizes,
        };
        let fb = b.build_fn(bid).map_err(|e| ir_err(e, stage.span))?;
        em.emit_body(
            op,
            params,
            body,
            &body_in_ty,
            &body_out_ty,
            &sig.captures,
            &sig.records,
            fb,
        )?;
    }
    Ok(())
}

/// Collect every map/fold stage with its body/params/op, in source order,
/// recursing through statements/guards/fanouts but NOT into nested map/fold
/// bodies (the recursion in `build_bodies` handles nesting).
fn collect_mapfold_stages(
    block: &Block,
) -> Vec<(&Stage, &Block, &[syn::Name], CollOp, Option<syn::Precision>)> {
    let mut out = Vec::new();
    collect_block(block, &mut out);
    out
}

fn collect_block<'a>(
    block: &'a Block,
    out: &mut Vec<(
        &'a Stage,
        &'a Block,
        &'a [syn::Name],
        CollOp,
        Option<syn::Precision>,
    )>,
) {
    let start = out.len();
    for item in &block.items {
        match item {
            BlockItem::Stmt(s) => collect_stmt(s, out),
            BlockItem::Arm(a) => collect_arm(&a.payload, out),
        }
    }
    if let Some(t) = &block.tail {
        collect_chain(t, out);
    }
    // ADR-0041 D2: a `loop: c { … }` / `seq: c { … }` covers the steps inside
    // it that no inner block already claimed (innermost wins).
    for e in &mut out[start..] {
        e.4 = e.4.or(block.compute);
    }
}

fn collect_stmt<'a>(
    stmt: &'a Stmt,
    out: &mut Vec<(
        &'a Stage,
        &'a Block,
        &'a [syn::Name],
        CollOp,
        Option<syn::Precision>,
    )>,
) {
    match &stmt.kind {
        StmtKind::Chain(c) => collect_chain(c, out),
        StmtKind::Bind(_) => {}
        StmtKind::Loop(l) => collect_block(&l.body, out),
        StmtKind::Error => {}
    }
}

fn collect_arm<'a>(
    p: &'a ArmPayload,
    out: &mut Vec<(
        &'a Stage,
        &'a Block,
        &'a [syn::Name],
        CollOp,
        Option<syn::Precision>,
    )>,
) {
    match p {
        ArmPayload::Chain(c) => collect_chain(c, out),
        ArmPayload::Block(b) => collect_block(b, out),
    }
}

fn collect_chain<'a>(
    chain: &'a Chain,
    out: &mut Vec<(
        &'a Stage,
        &'a Block,
        &'a [syn::Name],
        CollOp,
        Option<syn::Precision>,
    )>,
) {
    for stage in &chain.stages {
        match &stage.kind {
            StageKind::MapFold { op, params, body } => {
                out.push((stage, body, params.as_slice(), *op, None));
            }
            StageKind::Guard(arms) => {
                for arm in arms {
                    collect_arm(&arm.payload, out);
                }
            }
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    collect_chain(b, out);
                }
            }
            StageKind::SeqBlock(body) => collect_block(body, out),
            _ => {}
        }
    }
}

fn internal(span: syn::SourceLoc, msg: &str) -> syn::Diagnostic {
    diag(LCode::Internal, span, msg.to_string())
}

/// Context for how a block's tail is treated.
enum BodyCtx {
    FnBody {
        ret: Option<Ty>,
        effectful: bool,
    },
    MapFoldBody {
        out_ty: Ty,
    },
    #[allow(dead_code)]
    Plain,
}

/// Context for how a chain's final wire is treated.
#[derive(Clone)]
pub enum ChainCtx {
    Statement,
    FnBodyReturn {
        effectful: bool,
    },
    BodyReturn,
    HeadlessSeed(ObjectId),
    /// A return-position tail whose value the caller consumes directly (the
    /// effectful-B-present path packs it with the token — §8.1). Behaves like
    /// `Statement` (the value is handed back, not written to Ret), but the chain
    /// is in **return position**: a tail-less `seq` here demands a value it has
    /// none of, so it draws L1611 rather than being a legal no-op (ADR-0019 pin c).
    RetValue,
}
/// Map a surface `BinOp` to the IR `Operation`.
fn map_binop(op: BinOp) -> Operation {
    match op {
        BinOp::Add => Operation::Add,
        BinOp::Sub => Operation::Sub,
        BinOp::Mul => Operation::Mul,
        BinOp::Div => Operation::Div,
        BinOp::Mod => Operation::Mod,
        BinOp::Eq => Operation::Eq,
        BinOp::Ne => Operation::Neq,
        BinOp::Lt => Operation::Lt,
        BinOp::Gt => Operation::Gt,
        BinOp::Le => Operation::Le,
        BinOp::Ge => Operation::Ge,
        BinOp::And => Operation::And,
        BinOp::Or => Operation::Or,
    }
}

#[cfg(test)]
mod tests;
