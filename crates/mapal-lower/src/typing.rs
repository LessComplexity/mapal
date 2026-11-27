//! Pass D1 — the typing walk (DESIGN §7). A bottom-up ty synthesis over each
//! function body whose only inference device is literal-width unification
//! (LD3). It computes exactly what emission needs and nothing more:
//!
//! - `lit_ty`: the resolved ty for every unsized `Int`/`Float` literal, keyed by
//!   span (spans are unique per node, J2).
//! - map/fold block signatures (elem/acc tys, body result ty), keyed by stage span.
//! - the return-completeness verdict (L1306).
//!
//! Literal-width unification is a union-find over per-literal type variables.
//! Unification points are the §7.2 list; unresolved IntVars default `i32`,
//! FloatVars `f64`. Emission consults `lit_ty` to build each `constant()` with
//! the right `Value` variant; the builder is the second line of type checking
//! (LD12), so D1 does not re-derive every operand ty — it resolves literals and
//! the structural facts emission cannot compute locally.

use std::collections::BTreeMap;

use mapal_ir::Ty;
use mapal_syntax::{
    self as syn, ArmPayload, Block, BlockItem, Chain, CollOp, Expr, ExprKind, GuardDiscr, Program,
    Stage, StageKind, Stmt, StmtKind, UnOp,
};

use crate::diag::{LCode, diag};
use crate::formats::{Arg, Format, Verb};
use crate::scope::ScopeStack;
use crate::sizes::{Generics, Instance, SiteKey, SizeEnv};
use crate::tys::{TypeTable, name_text};

mod captures;
mod exprs;
mod returns;
mod sigs;
mod walk;
mod walk_core;

pub(crate) use self::captures::block_reads;
pub use self::returns::{arm_is_routing, check_return_completeness};
pub use self::sigs::{build_fn_sigs, no_generic_args, removed_conversion};
use self::{captures::*, returns::*, sigs::*};

/// A literal's resolved scalar ty (the union-find leaf state).
#[derive(Clone, Debug, PartialEq)]
enum LitState {
    /// An integer literal not yet pinned to a width.
    IntVar,
    /// A float literal not yet pinned to a width.
    FloatVar,
    /// Pinned to a concrete scalar ty.
    Fixed(Ty),
}

/// The per-function typing result emission consults.
#[derive(Default)]
pub struct TypeInfo {
    /// span(start,end) → resolved scalar ty for each literal.
    pub lit_ty: BTreeMap<(u32, u32), Ty>,
    /// stage span → map/fold signature.
    pub blocks: BTreeMap<(u32, u32), BlockSig>,
}

/// The synthesized signature of a map/fold operator block (LD11).
#[derive(Clone, Debug)]
pub struct BlockSig {
    /// `T` for map, `(Acc, T)` for fold — the body fn's declared input,
    /// prepended with `captures` when non-empty (ADR-0027).
    pub body_in: Ty,
    /// The body fn's declared output (the body's tail value ty).
    pub body_out: Ty,
    /// ADR-0027: read captures `(name, ty)` in first-use order — the leading
    /// hidden components of `body_in` (`captures.len()` of them). A struct
    /// capture is its leaf fields (mask-frontend: [`flatten_captures`]), each
    /// named by its path `name.field…`.
    pub captures: Vec<(String, Ty)>,
    /// mask-frontend: the struct captures `(name, ty)` the body rebuilds
    /// from their fields' captures.
    pub records: Vec<(String, Ty)>,
    /// `{owner}::{map|fold}@{i}` synthesized name.
    pub name: String,
}

/// mask-frontend: **a struct capture narrowed to its fields** — each
/// struct-typed capture as its leaf fields (nested structs flattened), named
/// `name.field…` (`.` is never in an identifier), so a body's captures are
/// arrays and numbers — what a device site reads (`cell_plan`, the tile
/// plans) — and the body rebuilds the struct from them: a field read is then
/// a projection of a pack, which the rewriter forwards to the field's own
/// capture. Returns the flattened captures and the struct ones.
pub fn flatten_captures(caps: &[(String, Ty)]) -> (Vec<(String, Ty)>, Vec<(String, Ty)>) {
    fn go(name: String, t: &Ty, out: &mut Vec<(String, Ty)>) {
        match t {
            Ty::Struct { fields, .. } => {
                for (f, ft) in fields {
                    go(format!("{name}.{f}"), ft, out);
                }
            }
            _ => out.push((name, t.clone())),
        }
    }
    let mut flat = Vec::new();
    let mut records = Vec::new();
    for (name, t) in caps {
        if matches!(t, Ty::Struct { .. }) {
            records.push((name.clone(), t.clone()));
        }
        go(name.clone(), t, &mut flat);
    }
    (flat, records)
}

impl TypeInfo {
    /// The resolved ty for a literal at `span`, if recorded.
    pub fn lit(&self, span: syn::SourceLoc) -> Option<&Ty> {
        self.lit_ty.get(&(span.start, span.end))
    }
    /// The block signature for the map/fold stage at `span`.
    pub fn block(&self, span: syn::SourceLoc) -> Option<&BlockSig> {
        self.blocks.get(&(span.start, span.end))
    }
}

/// A type or an unresolved literal variable, used during the walk.
#[derive(Clone, Debug)]
enum WTy {
    /// A fully-known ty.
    Known(Ty),
    /// A literal variable (union-find key).
    Var(usize),
    /// A tuple whose components may carry vars (so annotation unification can
    /// reach them — e.g. an array-of-unsized-floats bound to `[f32; n]`).
    Tuple(Vec<WTy>),
    /// An array whose element may carry a var.
    Array(Box<WTy>, u64),
    /// Unknown — an error already reported, or an unresolvable expr.
    Unknown,
}

/// ADR-0047: what the typing walk of one function (or instance) needs about
/// size generics — its own sizes, the generic functions, the call sites'
/// bindings emission found in an earlier round.
pub struct SizeCtx<'a> {
    /// The function's (instance) name: `f` or `f<4, 8>`.
    pub owner: String,
    /// The instance's size values (empty for a fixed-size function).
    pub env: SizeEnv,
    pub generics: &'a Generics<'a>,
    /// Emission's binding of each call site, from earlier rounds (D5).
    pub sites: &'a BTreeMap<SiteKey, Vec<Arg>>,
}

/// What the typing walk found at its calls to generic functions: the binding
/// it used at each site, and the instances they name.
#[derive(Default)]
pub struct SizeUse {
    pub sites: BTreeMap<SiteKey, Vec<Arg>>,
    pub instances: Vec<Instance>,
    /// Generic calls the walk could not type (its own types did not reach
    /// and no earlier round's emission bound the site): its diagnostics
    /// downstream of one may be artifacts of the missing result type.
    pub unresolved: usize,
}

/// The typing walk state for one function.
struct Walk<'a> {
    source: &'a str,
    types: &'a TypeTable,
    fn_sigs: &'a BTreeMap<String, FnSig>,
    owner_name: String,
    /// ADR-0047: the size context and what the walk found.
    sizes: &'a SizeCtx<'a>,
    size_use: SizeUse,
    effects: &'a crate::effects::Effects,
    /// the owner's declared (surface) return ty, for ret-write unification.
    owner_ret: Option<Ty>,
    /// The function body's tail chain (its span): a return position.
    fn_tail: Option<(u32, u32)>,
    /// Walking the last stage of [`Self::fn_tail`] (ADR-0051 D3: its result
    /// flows into the owner's output).
    at_fn_tail: bool,
    /// union-find parent links over literal vars.
    uf: Vec<usize>,
    /// per-var state.
    state: Vec<LitState>,
    /// var index → literal span (for lit_ty output).
    var_span: Vec<(u32, u32)>,
    diags: Vec<syn::Diagnostic>,
    info: TypeInfo,
    /// per-owner map/fold counter (deterministic block names).
    block_counter: usize,
    /// ADR-0048: walking a top-level chain (its bindings are the top-level
    /// names themselves, not locals reusing them).
    top_chain: bool,
    /// map/fold signatures whose literal widths resolve with the function's
    /// (a body's float literal takes its width from a later use or
    /// annotation): `(stage span, body input components, body output,
    /// captures, name)`, each component a `(WTy, fallback)`.
    pending_blocks: Vec<PendingBlock>,
}

/// A map/fold signature awaiting [`Walk::resolve_all`].
struct PendingBlock {
    span: (u32, u32),
    /// `Some(components)`: the body input is this tuple; `None`: `body_in`.
    comps: Option<Vec<(WTy, Ty)>>,
    body_in: (WTy, Ty),
    body_out: (WTy, Ty),
    captures: Vec<((String, Ty), WTy)>,
    records: Vec<(String, Ty)>,
    name: String,
}

/// A surface fn's declared (surface) signature, for call typing in D1.
#[derive(Clone)]
pub struct FnSig {
    /// surface param tys (resolved), in order.
    pub params: Vec<Ty>,
    /// surface return ty (resolved), if declared.
    pub ret: Option<Ty>,
    pub effectful: bool,
    /// ADR-0047: a size-generic function's placeholder (`params` / `ret`
    /// empty: each instance has its own resolved signature).
    pub generic: bool,
}

/// Run D1 over one function (or one instance of a size-generic function,
/// `sizes.owner` naming it). Returns the typing info, the diagnostics and the
/// generic calls it typed (ADR-0047 D5).
pub fn analyze_fn(
    source: &str,
    types: &TypeTable,
    effects: &crate::effects::Effects,
    fn_sigs: &BTreeMap<String, FnSig>,
    fd: &syn::FnDecl,
    sizes: &SizeCtx,
) -> (TypeInfo, Vec<syn::Diagnostic>, SizeUse) {
    let owner_name = sizes.owner.clone();
    let owner_ret = fn_sigs.get(&owner_name).and_then(|s| s.ret.clone());
    let mut walk = Walk {
        source,
        types,
        fn_sigs,
        owner_name,
        sizes,
        size_use: SizeUse::default(),
        effects,
        owner_ret,
        fn_tail: fd.body.tail.as_ref().map(|t| (t.span.start, t.span.end)),
        at_fn_tail: false,
        uf: Vec::new(),
        state: Vec::new(),
        var_span: Vec::new(),
        diags: Vec::new(),
        info: TypeInfo::default(),
        block_counter: 0,
        top_chain: false,
        pending_blocks: Vec::new(),
    };
    let mut scope: ScopeStack<WTy> = ScopeStack::new();
    // Bind params to their resolved tys.
    let sig = walk.fn_sigs.get(&walk.owner_name).cloned();
    if let Some(sig) = &sig {
        for (p, pty) in fd.params.iter().zip(sig.params.iter()) {
            walk.check_not_size(p.name);
            scope.bind(name_text(source, p.name), WTy::Known(pty.clone()));
        }
    }
    // ADR-0048 D3: the top-level values this function reads that are not
    // literals are their chains, walked at the outermost scope.
    let globals = types.globals();
    for i in globals.used_by(source, fd) {
        walk.top_chain = true;
        walk.chain(&globals.items[i].chain, &mut scope);
        walk.top_chain = false;
    }
    walk.block(&fd.body, &mut scope);
    walk.resolve_all();
    (walk.info, walk.diags, walk.size_use)
}

fn array_elem(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Array { elem, .. } => Some((**elem).clone()),
        _ => None,
    }
}

/// The element WTy of an array binding (for `c[i] <- v` element-ty unification,
/// ADR-0021). Handles both a fully-known array and a var-carrying element.
fn array_elem_wty(w: &WTy) -> Option<WTy> {
    match w {
        WTy::Array(e, _) => Some((**e).clone()),
        WTy::Known(Ty::Array { elem, .. }) => Some(WTy::Known((**elem).clone())),
        _ => None,
    }
}
