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

impl Walk<'_> {
    /// ADR-0048 D3: a top-level scalar computed before the program runs — an
    /// unannotated number is a literal (its width from context), an
    /// annotated one a constant of its type.
    fn global_const(&mut self, text: &str, span: syn::SourceLoc) -> WTy {
        use crate::globals::GlobalConst;
        match self.types.globals().consts[text].clone() {
            GlobalConst::Int {
                pinned: Some(t), ..
            }
            | GlobalConst::Float {
                pinned: Some(t), ..
            } => WTy::Known(t),
            GlobalConst::Int {
                value,
                pinned: None,
            } => {
                let var = self.fresh_var(span, false);
                if value > i128::from(i32::MAX) || value < i128::from(i32::MIN) {
                    self.pin(var, &Ty::i64(), span);
                }
                WTy::Var(var)
            }
            GlobalConst::Float { pinned: None, .. } => WTy::Var(self.fresh_var(span, true)),
            GlobalConst::Bool(_) => WTy::Known(Ty::Bool),
        }
    }

    /// ADR-0048 D5: a known struct read through its `_` field(s) until `fits`.
    fn through_default(&mut self, w: &WTy, fits: impl Fn(&Ty) -> bool) -> WTy {
        if let Some(t) = self.wty_to_ty(w)
            && matches!(t, Ty::Struct { .. })
            && let Some(k) = TypeTable::default_steps(&t, &fits)
        {
            let mut cur = t;
            for _ in 0..k {
                cur = TypeTable::default_of(&cur).cloned().unwrap_or(Ty::Unit);
            }
            return WTy::Known(cur);
        }
        w.clone()
    }

    // --- size parameters (ADR-0047 D4) --------------------------------------

    /// The value of a size parameter named `text` not shadowed in `scope`.
    fn size_value(&self, text: &str, scope: &ScopeStack<WTy>) -> Option<u64> {
        if scope.contains(text) {
            return None;
        }
        self.sizes.env.get(text).copied()
    }

    /// An integer literal's value, or a size expression's (D4: a size is an
    /// integer literal in every respect).
    fn int_const(&self, e: &Expr, scope: &ScopeStack<WTy>) -> Option<u64> {
        match &e.kind {
            ExprKind::Int(n) => Some(*n),
            _ => self.size_fold(e, scope).and_then(|v| u64::try_from(v).ok()),
        }
    }

    /// D4: the value of an integer expression over size parameters and
    /// literals (`rows * columns`, `n - 1`), folded to one literal.
    fn size_fold(&self, e: &Expr, scope: &ScopeStack<WTy>) -> Option<i128> {
        if self.sizes.env.is_empty() {
            return None;
        }
        crate::sizes::fold_value(self.source, e, &|t| self.size_value(t, scope))
    }

    /// L1703: a binding may not reuse a size parameter's name.
    fn check_not_size(&mut self, name: syn::Name) {
        let text = name_text(self.source, name);
        if self.types.globals().is_top(text) {
            if !self.top_chain {
                self.diags.push(diag(
                    LCode::TopNameReused,
                    name.span,
                    format!(
                        "`{text}` is a top-level name; a local cannot reuse it — pick another name \
                         (ADR-0048 D1)"
                    ),
                ));
            }
        } else if self.sizes.env.contains_key(text) {
            self.diags.push(diag(
                LCode::SizeRebound,
                name.span,
                format!("`{text}` is a size parameter; it cannot be rebound — pick another name"),
            ));
        } else if self.sizes.env.format(text).is_some() {
            self.diags.push(diag(
                LCode::SizeRebound,
                name.span,
                format!(
                    "`{text}` is a format parameter (ADR-0051); it cannot be bound as a value — \
                     pick another name"
                ),
            ));
        }
    }

    /// D3 at a call to a generic function (typing side): emission's binding
    /// of this site when an earlier round found one, else this walk's own
    /// inference from `cur` (and the next stage's annotation), after the
    /// stage's explicit arguments (ADR-0051 D2). The result is the instance's
    /// output type; `Unknown` when neither reaches.
    fn generic_call(
        &mut self,
        name: &str,
        cur: &WTy,
        next: Option<&Stage>,
        span: syn::SourceLoc,
        args: Option<&[syn::SizeExpr]>,
    ) -> WTy {
        let Some(fd) = self.sizes.generics.get(name).copied() else {
            return WTy::Unknown;
        };
        let explicit = match args {
            Some(a) => match crate::sizes::explicit_args(
                self.source,
                a,
                &self.sizes.env,
                self.source,
                name,
                &fd.generics,
                span,
            ) {
                Ok(v) => v,
                Err(d) => {
                    self.diags.push(d);
                    return WTy::Unknown;
                }
            },
            None => Vec::new(),
        };
        let key: SiteKey = (self.owner_name.clone(), span.start, span.end);
        let values = match self.sizes.sites.get(&key) {
            Some(v) => Some(v.clone()),
            None => self.wty_to_ty(cur).and_then(|arg| {
                let expected = self.expected_flow(next);
                crate::sizes::infer(
                    self.source,
                    self.types,
                    fd,
                    &arg,
                    expected.as_ref(),
                    &explicit,
                    span,
                )
                .ok()
            }),
        };
        let Some(values) = values else {
            self.size_use.unresolved += 1;
            return WTy::Unknown;
        };
        let env = crate::sizes::env_of(self.source, &fd.generics, &values, self.types.base_env());
        let effectful = self.effects.is_effectful(name);
        let Some(sig) = crate::sizes::instance_sig(self.source, self.types, fd, &env, effectful)
        else {
            return WTy::Unknown;
        };
        self.size_use.sites.insert(key, values.clone());
        self.size_use.instances.push((name.to_string(), values));
        self.type_call(&sig, cur, span);
        match sig.ret {
            Some(r) => WTy::Known(r),
            None => WTy::Unknown,
        }
    }

    /// ADR-0051 D4: `x -> to<…>` / `x -> widen<…>` (typing side): the target
    /// from the explicit argument, else the expected type (the next stage's
    /// annotation, the owner's output at `-> ret`, a fixed function's
    /// parameter the value flows into). Emission owns the admission (L1614 /
    /// L1616) and the undetermined target (L1708); a literal flowing into a
    /// narrow target is an `f32` (the one source), into an explicit
    /// `source` that format.
    fn conversion(
        &mut self,
        verb: Verb,
        args: Option<&[syn::SizeExpr]>,
        cur: &WTy,
        next: Option<&Stage>,
        span: syn::SourceLoc,
    ) -> WTy {
        let decl = verb.decl();
        let explicit = match args {
            Some(a) => match crate::sizes::explicit_args(
                self.source,
                a,
                &self.sizes.env,
                crate::formats::PRELUDE,
                verb.name(),
                &decl.generics,
                span,
            ) {
                Ok(v) => v,
                Err(d) => {
                    self.diags.push(d);
                    return WTy::Unknown;
                }
            },
            None => Vec::new(),
        };
        let format_at = |i: usize| match explicit.get(i) {
            Some(Arg::Format(f)) => Some(*f),
            _ => None,
        };
        if let (Some(src), WTy::Var(_)) = (format_at(1), cur) {
            self.unify_known(cur, &src.ty(), span);
        }
        let target = format_at(0).or_else(|| {
            let want = self.expected_flow(next);
            want.as_ref().and_then(Format::of)
        });
        match target {
            Some(t) => {
                // A float literal flowing in takes its width from the
                // conversion: an `f32` into a narrow target (ADR-0043's one
                // source), the target itself into `f32` / `f64` (the identity).
                // An integer literal keeps its own width (`3 -> to<f32>`
                // converts an `i32`).
                if format_at(1).is_none()
                    && let WTy::Var(v) = cur
                    && t.is_float()
                {
                    let r = self.find(*v);
                    if matches!(self.state[r], LitState::FloatVar) {
                        let width = if t.narrow().is_some() {
                            Ty::f32()
                        } else {
                            t.ty()
                        };
                        self.unify_known(cur, &width, span);
                    }
                }
                WTy::Known(t.ty())
            }
            None => WTy::Unknown,
        }
    }

    /// ADR-0051 D3: the type a call's or a conversion's result flows into —
    /// the next stage's annotation, the owner's output at `-> ret` (`ret.k`:
    /// its component) or at the function body's tail, or the one parameter of
    /// a fixed function called next. Emission reads the same sources
    /// (`emit::Emitter::expected_flow`).
    fn expected_flow(&mut self, next: Option<&Stage>) -> Option<Ty> {
        if let Some(t) = self.expected_result(next) {
            return Some(t);
        }
        match next.map(|s| &s.kind) {
            None if self.at_fn_tail => self.owner_ret.clone(),
            Some(StageKind::Ret { proj: Some((k, _)) }) => self
                .owner_ret
                .as_ref()
                .and_then(|r| u32::try_from(*k).ok().and_then(|k| r.component_ty(k)))
                .cloned(),
            _ => self.expected_param(next),
        }
    }

    /// ADR-0051 D3: the type a value flows into as the next stage's input —
    /// the one parameter of a fixed (non-generic) function called there.
    fn expected_param(&self, next: Option<&Stage>) -> Option<Ty> {
        let Some(StageKind::Expr(e)) = next.map(|s| &s.kind) else {
            return None;
        };
        let ExprKind::Var(n) = &e.kind else {
            return None;
        };
        match self.fn_sigs.get(name_text(self.source, *n)) {
            Some(sig) if !sig.generic => match sig.params.as_slice() {
                [one] => Some(one.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    /// D3.4: the type a call's result is bound with — the next stage's
    /// annotation (`-> y: [f32; 8]`), or the owner's output at a `-> ret`.
    fn expected_result(&mut self, next: Option<&Stage>) -> Option<Ty> {
        match next.map(|s| &s.kind) {
            Some(StageKind::Bind { ty: Some(t), .. }) => {
                let mut sink = Vec::new();
                self.types
                    .resolve_sized(self.source, t, &self.sizes.env, &mut sink)
            }
            Some(StageKind::Ret { proj: None }) => self.owner_ret.clone(),
            _ => None,
        }
    }

    // --- union-find ---------------------------------------------------------

    fn fresh_var(&mut self, span: syn::SourceLoc, float: bool) -> usize {
        let v = self.uf.len();
        self.uf.push(v);
        self.state.push(if float {
            LitState::FloatVar
        } else {
            LitState::IntVar
        });
        self.var_span.push((span.start, span.end));
        v
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.uf[x] != x {
            self.uf[x] = self.uf[self.uf[x]];
            x = self.uf[x];
        }
        x
    }

    /// Unify a literal var with a concrete ty (pin it). Reports L1203 on a
    /// var/scalar-kind clash (IntVar↔Float etc.).
    fn pin(&mut self, var: usize, ty: &Ty, span: syn::SourceLoc) {
        let r = self.find(var);
        match self.state[r].clone() {
            LitState::IntVar => {
                if matches!(ty, Ty::Int { .. }) {
                    self.state[r] = LitState::Fixed(ty.clone());
                } else {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "integer literal cannot have type `{}`",
                            crate::emit::ty_name(ty)
                        ),
                    ));
                    self.state[r] = LitState::Fixed(Ty::i32());
                }
            }
            LitState::FloatVar => {
                if matches!(ty, Ty::Float { .. }) {
                    self.state[r] = LitState::Fixed(ty.clone());
                } else {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "float literal cannot have type `{}`",
                            crate::emit::ty_name(ty)
                        ),
                    ));
                    self.state[r] = LitState::Fixed(Ty::f64());
                }
            }
            LitState::Fixed(prev) => {
                if &prev != ty {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "literal type conflict: `{}` vs `{}`",
                            crate::emit::ty_name(&prev),
                            crate::emit::ty_name(ty)
                        ),
                    ));
                }
            }
        }
    }

    /// Unify two literal vars (must be same kind).
    fn union(&mut self, a: usize, b: usize, span: syn::SourceLoc) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match (self.state[ra].clone(), self.state[rb].clone()) {
            (LitState::Fixed(t), _) => {
                self.uf[rb] = ra;
                self.pin(ra, &t.clone(), span);
                // re-pin rb's prior fixed value if any handled by pin merge below
                if let LitState::Fixed(tb) = self.state[rb].clone() {
                    self.pin(ra, &tb, span);
                }
            }
            (_, LitState::Fixed(t)) => {
                self.uf[ra] = rb;
                self.pin(rb, &t, span);
            }
            (LitState::IntVar, LitState::IntVar) | (LitState::FloatVar, LitState::FloatVar) => {
                self.uf[rb] = ra;
            }
            _ => {
                // IntVar vs FloatVar — clash.
                self.diags.push(diag(
                    LCode::LiteralTypeConflict,
                    span,
                    "integer/float literal width unification clash".to_string(),
                ));
                self.uf[rb] = ra;
            }
        }
    }

    /// Unify a WTy with a concrete ty (the common case at annotation points),
    /// recursing into composites so an array/tuple of unsized literals takes the
    /// annotation's element/component widths.
    fn unify_known(&mut self, w: &WTy, ty: &Ty, span: syn::SourceLoc) {
        match (w, ty) {
            (WTy::Var(v), _) => self.pin(*v, ty, span),
            (WTy::Tuple(ws), Ty::Tuple(ts)) if ws.len() == ts.len() => {
                for (cw, ct) in ws.iter().zip(ts.iter()) {
                    self.unify_known(cw, ct, span);
                }
            }
            (WTy::Array(ew, _), Ty::Array { elem, .. }) => {
                self.unify_known(ew, elem, span);
            }
            _ => {}
        }
    }

    /// Unify two WTys (binop operands, phi arms, etc.).
    fn unify(&mut self, a: &WTy, b: &WTy, span: syn::SourceLoc) {
        match (a, b) {
            (WTy::Var(x), WTy::Var(y)) => self.union(*x, *y, span),
            (WTy::Var(x), WTy::Known(t)) | (WTy::Known(t), WTy::Var(x)) => self.pin(*x, t, span),
            (WTy::Tuple(xs), WTy::Tuple(ys)) if xs.len() == ys.len() => {
                for (cx, cy) in xs.iter().zip(ys.iter()) {
                    self.unify(cx, cy, span);
                }
            }
            (WTy::Array(ex, _), WTy::Array(ey, _)) => self.unify(ex, ey, span),
            (WTy::Var(x), other) | (other, WTy::Var(x)) => {
                if let Some(t) = self.materialize(other) {
                    self.pin(*x, &t, span);
                }
            }
            _ => {}
        }
    }

    /// Materialize a WTy to a concrete ty if all leaves are currently known
    /// (vars resolve to their current fixed/default state).
    fn materialize(&mut self, w: &WTy) -> Option<Ty> {
        self.wty_to_ty(w)
    }

    // --- resolution ---------------------------------------------------------

    /// Resolve every literal var to a concrete ty (defaults i32/f64) and write
    /// `info.lit_ty`. Range-checking is done at emission against the resolved ty.
    fn resolve_all(&mut self) {
        for v in 0..self.uf.len() {
            let r = self.find(v);
            let ty = match self.state[r].clone() {
                LitState::Fixed(t) => t,
                LitState::IntVar => Ty::i32(),
                LitState::FloatVar => Ty::f64(),
            };
            let span = self.var_span[v];
            self.info.lit_ty.insert(span, ty);
        }
        for b in std::mem::take(&mut self.pending_blocks) {
            let mut ty_of =
                |(w, fallback): &(WTy, Ty)| self.wty_to_ty(w).unwrap_or(fallback.clone());
            let body_in = match &b.comps {
                Some(comps) => Ty::Tuple(comps.iter().map(&mut ty_of).collect()),
                None => ty_of(&b.body_in),
            };
            let body_out = ty_of(&b.body_out);
            let captures = b
                .captures
                .into_iter()
                .map(|((name, t), w)| (name, ty_of(&(w, t))))
                .collect();
            self.info.blocks.insert(
                b.span,
                BlockSig {
                    body_in,
                    body_out,
                    captures,
                    records: b.records,
                    name: b.name,
                },
            );
        }
    }

    // --- the walk -----------------------------------------------------------

    fn block(&mut self, block: &Block, scope: &mut ScopeStack<WTy>) {
        for item in &block.items {
            match item {
                BlockItem::Stmt(s) => self.stmt(s, scope),
                BlockItem::Arm(_) => {
                    // bare arm at block level — guard arms only appear inside a
                    // Guard stage; a top-level arm is structurally rejected by
                    // the parser. Defensive no-op.
                }
            }
        }
        if let Some(tail) = &block.tail {
            self.chain(tail, scope);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, scope: &mut ScopeStack<WTy>) {
        match &stmt.kind {
            StmtKind::Chain(c) => {
                self.chain(c, scope);
            }
            StmtKind::Bind(b) => {
                self.check_not_size(b.name);
                let v = self.expr(&b.value, scope);
                // `c[i] <- v` (ADR-0021): an *indexed* bind is an element write,
                // not a fresh binding. Type the index, unify `v` with `c`'s
                // element ty (so a bare literal resolves to the element width, not
                // a spurious i32 clash — this is how width-unification covers the
                // element-ty mismatch, L1201/L1203), and leave `c`'s binding
                // intact (its array ty is unchanged).
                if let Some(idx) = &b.index {
                    let _ = self.expr(idx, scope);
                    if let Some((bt, _)) = scope.resolve(name_text(self.source, b.name))
                        && let Some(elem) = array_elem_wty(&bt.clone())
                    {
                        self.unify(&v, &elem, b.span);
                    }
                    return;
                }
                if let Some(annot) = &b.ty
                    && let Some(t) = self.types.resolve_sized(
                        self.source,
                        annot,
                        &self.sizes.env,
                        &mut self.diags,
                    )
                {
                    self.unify_known(&v, &t, b.span);
                    scope.bind(name_text(self.source, b.name), WTy::Known(t));
                    return;
                }
                scope.bind(name_text(self.source, b.name), v);
            }
            StmtKind::Loop(l) => {
                scope.push_loop();
                self.block(&l.body, scope);
                scope.pop();
            }
            StmtKind::Error => {}
        }
    }

    fn chain(&mut self, chain: &Chain, scope: &mut ScopeStack<WTy>) {
        self.chain_seeded(chain, WTy::Unknown, scope);
    }

    /// Like [`chain`], but a **headless** chain (no head expr) starts from `seed`
    /// instead of `Unknown` — the fanout wire threaded into each branch so a
    /// tuple-producing op (`enumerate`/`zip`) inside a branch types its downstream
    /// `map` body correctly (emit already threads the real wire; this keeps the
    /// D1 BlockSig in step).
    fn chain_seeded(&mut self, chain: &Chain, seed: WTy, scope: &mut ScopeStack<WTy>) {
        // `() -> g`: the Unit witness a zero-parameter fn takes (anywhere
        // else `()` is not a value — emission's L1301).
        let zero_call = matches!(chain.stages.first().map(|s| &s.kind),
            Some(StageKind::Expr(e)) if matches!(&e.kind, ExprKind::Var(n)
                if self.fn_sigs.get(name_text(self.source, *n)).is_some_and(|g| g.params.is_empty())));
        let mut cur: WTy = match &chain.head {
            Some(h) if zero_call && matches!(h.kind, ExprKind::Unit) => WTy::Known(Ty::Unit),
            Some(h) => self.expr(h, scope),
            None => seed,
        };
        let mut prev: Option<&Expr> = chain.head.as_ref();
        let is_fn_tail = self.fn_tail == Some((chain.span.start, chain.span.end));
        for (i, stage) in chain.stages.iter().enumerate() {
            let next = chain.stages.get(i + 1);
            let outer = self.at_fn_tail;
            self.at_fn_tail = is_fn_tail && next.is_none();
            cur = self.stage(&stage.kind, stage.span, cur, prev, next, scope);
            self.at_fn_tail = outer;
            prev = match &stage.kind {
                StageKind::Expr(e) => Some(e),
                _ => None,
            };
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn stage(
        &mut self,
        kind: &StageKind,
        stage_span: syn::SourceLoc,
        cur: WTy,
        prev: Option<&Expr>,
        next: Option<&Stage>,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        match kind {
            StageKind::Expr(e) => {
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(self.source, *n);
                    if scope.contains(text) {
                        // rebind: cur unifies into the existing binding's ty.
                        if let Some((b, _)) = scope.resolve(text) {
                            let bty = b.clone();
                            self.unify(&cur, &bty, e.span);
                        }
                        cur
                    } else if self.fn_sigs.get(text).is_some_and(|s| s.generic) {
                        // ADR-0047: a call to a size-generic function.
                        self.generic_call(text, &cur, next, e.span, None)
                    } else if let Some(sig) = self.fn_sigs.get(text).cloned() {
                        // call: cur (arg) unifies with declared input; result ty.
                        self.type_call(&sig, &cur, e.span);
                        match &sig.ret {
                            Some(r) => WTy::Known(r.clone()),
                            None => WTy::Unknown,
                        }
                    } else if crate::is_print_builtin(text) {
                        // print sink; result is "no value".
                        WTy::Unknown
                    } else if crate::is_read_builtin(text) {
                        // ADR-0045: the binding's annotation types the array
                        // (emit owns L1617 / L1618); nothing to unify here.
                        WTy::Unknown
                    } else if crate::is_time_builtin(text) {
                        // `() -> time` (plan-time-builtin): no source, f64
                        // milliseconds out. `cur` is whatever the wire-less
                        // head left (Unknown) and is deliberately not unified.
                        WTy::Known(Ty::f64())
                    } else if text == "zip" {
                        // zip: cur is a 2-tuple `([A;n], [B;n])`; result `[(A,B);n]`.
                        // Synthesize when resolvable so a downstream map sees the
                        // tuple element ty (emit re-derives + owns the L-codes).
                        self.zip_result(&cur)
                    } else if text == "enumerate" {
                        // enumerate: cur is `[A;n]`; result `[(i32, A);n]`.
                        self.enumerate_result(&cur)
                    } else if text == "iota" {
                        // ADR-0031: `n -> iota` — the count flows in. Size
                        // synthesized from the visible literal stage; emit
                        // re-derives and the builder owns static-n (L1612).
                        // A size parameter is a literal (ADR-0047 D4).
                        match prev.and_then(|p| self.int_const(p, scope)) {
                            Some(n) if n > 0 && n <= i32::MAX as u64 => {
                                WTy::Array(Box::new(WTy::Known(Ty::i32())), n)
                            }
                            // onto-device: an extent past i32 — the count is an
                            // `i64` literal and the elements its index type.
                            Some(n) if n <= i64::MAX as u64 && n > 0 => {
                                self.unify_known(&cur, &Ty::i64(), e.span);
                                WTy::Array(Box::new(WTy::Known(Ty::index_ty(n))), n)
                            }
                            _ => WTy::Unknown,
                        }
                    } else if text == "fill" {
                        // ADR-0031: `(x, n) -> fill` — elem from the tuple's
                        // first component, size from the visible literal
                        // (emit re-derives; L1613).
                        let elem = match &cur {
                            WTy::Tuple(ws) if ws.len() == 2 => ws[0].clone(),
                            _ => WTy::Unknown,
                        };
                        match prev.map(|p| &p.kind) {
                            Some(ExprKind::Tuple(xs)) if xs.len() == 2 => {
                                match self.int_const(&xs[1], scope) {
                                    Some(n) if n > 0 && n <= i32::MAX as u64 => {
                                        WTy::Array(Box::new(elem), n)
                                    }
                                    Some(n) if n > 0 && n <= i64::MAX as u64 => {
                                        if let WTy::Tuple(ws) = &cur {
                                            self.unify_known(&ws[1], &Ty::i64(), e.span);
                                        }
                                        WTy::Array(Box::new(elem), n)
                                    }
                                    _ => WTy::Unknown,
                                }
                            }
                            _ => WTy::Unknown,
                        }
                    } else if let Some(verb) = Verb::from_name(text) {
                        // ADR-0051 D4: `to` / `widen`, the target inferred
                        // from the context (emit owns L1614 / L1616 / L1708).
                        self.conversion(verb, None, &cur, next, e.span)
                    } else if let Some((verb, f)) = crate::formats::removed(text) {
                        self.diags.push(removed_conversion(text, verb, f, e.span));
                        WTy::Unknown
                    } else if mapal_ir::MathFn::from_name(text).is_some() {
                        // ADR-0042 / ADR-0046: `x -> exp` (`sqrt`, `sin`,
                        // `cos`) keeps the float width — a literal's variable
                        // flows through, so the width is whatever unifies
                        // with it (emit owns L1615).
                        match &cur {
                            WTy::Known(Ty::Float { .. }) | WTy::Var(_) => cur.clone(),
                            _ => WTy::Unknown,
                        }
                    } else {
                        // bind cur to a fresh name.
                        self.check_not_size(*n);
                        scope.bind(text, cur.clone());
                        cur
                    }
                } else if let ExprKind::Generic { name, args } = &e.kind {
                    // ADR-0051 D2: a stage with explicit generic arguments.
                    let text = name_text(self.source, *name);
                    if self.fn_sigs.get(text).is_some_and(|s| s.generic) && !scope.contains(text) {
                        self.generic_call(text, &cur, next, e.span, Some(args))
                    } else if let (Some(verb), false) =
                        (Verb::from_name(text), scope.contains(text))
                    {
                        self.conversion(verb, Some(args), &cur, next, e.span)
                    } else {
                        self.diags.push(no_generic_args(
                            self.source,
                            text,
                            scope.contains(text),
                            self.fn_sigs.contains_key(text),
                            e.span,
                        ));
                        WTy::Unknown
                    }
                } else {
                    // general expr stage — emission rejects (L1302). Walk for
                    // literals anyway so they resolve.
                    self.expr(e, scope);
                    WTy::Unknown
                }
            }
            StageKind::OpShorthand { expr } => {
                // The hole is `cur`; lower the expr with Hole↦cur.
                self.expr_with_hole(expr, &cur, scope)
            }
            StageKind::Bind { name, ty, .. } => {
                self.check_not_size(*name);
                if let Some(annot) = ty
                    && let Some(t) = self.types.resolve_sized(
                        self.source,
                        annot,
                        &self.sizes.env,
                        &mut self.diags,
                    )
                {
                    self.unify_known(&cur, &t, stage_span);
                    scope.bind(name_text(self.source, *name), WTy::Known(t.clone()));
                    return WTy::Known(t);
                }
                scope.bind(name_text(self.source, *name), cur.clone());
                cur
            }
            StageKind::Ret { proj } => {
                // Unify the returned value with the declared output (full) or
                // component (`ret.k`) ty so literals take the return width
                // (§7.2). Reports L1203 on an int/float clash via `unify_known`.
                if let Some(out) = self.owner_ret.clone() {
                    match proj {
                        None => self.unify_known(&cur, &out, stage_span),
                        // Skip the widening for a slot > u32::MAX (it would
                        // truncate and unify against the wrong component);
                        // emit.rs rejects the slot with L1205 (LOWER-RETK-TRUNC).
                        Some((k, _)) if *k <= u32::MAX as u64 => {
                            if let Some(c) = out.component_ty(*k as u32) {
                                self.unify_known(&cur, &c.clone(), stage_span);
                            }
                        }
                        Some(_) => {}
                    }
                }
                WTy::Unknown
            }
            StageKind::LoopJump => WTy::Unknown,
            StageKind::Guard(arms) => self.guard(arms, &cur, scope),
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    self.chain_seeded(b, cur.clone(), scope);
                }
                WTy::Unknown
            }
            StageKind::SeqBlock(body) => {
                // Type-walk the seq body in the enclosing scope (ADR-0019 pin b):
                // chain statements + tail seed from the wire `cur` (pin a), so a
                // headless chain resolves its literals against the seq input.
                // Bind/Loop are ordinary statements. Value ty is left Unknown
                // (like a fanout join): emission re-synthesizes the real ty.
                for item in &body.items {
                    if let BlockItem::Stmt(s) = item {
                        match &s.kind {
                            StmtKind::Chain(c) => self.chain_seeded(c, cur.clone(), scope),
                            _ => self.stmt(s, scope),
                        }
                    }
                }
                if let Some(tail) = &body.tail {
                    self.chain_seeded(tail, cur.clone(), scope);
                }
                WTy::Unknown
            }
            StageKind::MapFold { op, params, body } => {
                self.map_fold(*op, params, body, stage_span, &cur, scope)
            }
            StageKind::StmtBlock(_) | StageKind::Error(_) => WTy::Unknown,
        }
    }

    fn guard(&mut self, arms: &[syn::GuardArm], cur: &WTy, scope: &mut ScopeStack<WTy>) -> WTy {
        // For value-match guards, each integer discriminant unifies with the
        // scrutinee (so a bare-int scrutinee literal resolves). Arm payloads
        // produce values that unify into a common ty.
        let mut arm_tys: Vec<WTy> = Vec::new();
        let mut is_routing = false;
        let routing_guard = arms.iter().any(|a| arm_is_routing(&a.payload));
        for arm in arms {
            if let GuardDiscr::Int(_) = arm.discr {
                // unify cur with i (cur is integer); we cannot construct a lit
                // var for the discr (it lives in the guard token), but we pin
                // cur to integer-ish by leaving it; emission range-checks.
            }
            // Detect routing (headless chain ending in LoopJump).
            if arm_is_routing(&arm.payload) {
                is_routing = true;
            }
            // A routing guard's exit arm binds its exit value in the loop's
            // enclosing scope (DESIGN §8.5, as emission does), so a later
            // map/fold body sees it as a capture. A bare final name binds only
            // when it was unbound before the arm (else it is a rebind or call).
            let exit_name = (routing_guard && !arm_is_routing(&arm.payload))
                .then(|| arm_exit_bind(&arm.payload))
                .flatten()
                .map(|n| name_text(self.source, n))
                .filter(|t| !scope.contains(t) && !self.fn_sigs.contains_key(*t));
            scope.push();
            let aty = self.arm_value(&arm.payload, cur, scope);
            let exit_bind = exit_name.and_then(|t| scope.resolve(t).map(|(w, _)| (t, w.clone())));
            scope.pop();
            if let Some((text, w)) = exit_bind {
                scope.bind_loop_exit(text, w);
            }
            arm_tys.push(aty);
        }
        if is_routing {
            return WTy::Unknown;
        }
        // Phi guard: unify arm tys pairwise; result is the common ty.
        for i in 1..arm_tys.len() {
            let (a, b) = (arm_tys[0].clone(), arm_tys[i].clone());
            self.unify(&a, &b, syn::SourceLoc::new(0, 0));
        }
        arm_tys.into_iter().next().unwrap_or(WTy::Unknown)
    }

    fn arm_value(&mut self, payload: &ArmPayload, cur: &WTy, scope: &mut ScopeStack<WTy>) -> WTy {
        match payload {
            ArmPayload::Chain(c) => self.headless_chain_value(c, cur, scope),
            ArmPayload::Block(b) => {
                for item in &b.items {
                    if let BlockItem::Stmt(s) = item {
                        self.stmt(s, scope);
                    }
                }
                match &b.tail {
                    Some(t) => self.headless_chain_value(t, cur, scope),
                    None => WTy::Unknown,
                }
            }
        }
    }

    /// A headless chain in an arm position seeds cur := scrutinee unless it is a
    /// bare marker (-> ret / -> loop).
    fn headless_chain_value(
        &mut self,
        chain: &Chain,
        scrutinee: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        let mut cur = match &chain.head {
            Some(h) => self.expr(h, scope),
            None => scrutinee.clone(),
        };
        let mut prev: Option<&Expr> = chain.head.as_ref();
        for (i, stage) in chain.stages.iter().enumerate() {
            let next = chain.stages.get(i + 1);
            cur = self.stage(&stage.kind, stage.span, cur, prev, next, scope);
            prev = match &stage.kind {
                StageKind::Expr(e) => Some(e),
                _ => None,
            };
        }
        cur
    }

    fn map_fold(
        &mut self,
        op: CollOp,
        params: &[syn::Name],
        body: &Block,
        stage_span: syn::SourceLoc,
        cur: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        // Determine element / acc tys from cur (ADR-0048 D5: a struct reads
        // through its `_` array).
        let cur = &match (op, cur) {
            (CollOp::Map, _) => self.through_default(cur, |t| matches!(t, Ty::Array { .. })),
            (CollOp::Fold, WTy::Tuple(ws)) if ws.len() == 2 => WTy::Tuple(vec![
                ws[0].clone(),
                self.through_default(&ws[1], |t| matches!(t, Ty::Array { .. })),
            ]),
            (CollOp::Fold, WTy::Known(Ty::Tuple(ts))) if ts.len() == 2 => WTy::Tuple(vec![
                WTy::Known(ts[0].clone()),
                self.through_default(&WTy::Known(ts[1].clone()), |t| {
                    matches!(t, Ty::Array { .. })
                }),
            ]),
            _ => cur.clone(),
        };
        let cur_ty = self.wty_to_ty(cur);
        let idx = self.block_counter;
        self.block_counter += 1;
        let name = format!(
            "{}::{}@{}",
            self.owner_name,
            match op {
                CollOp::Map => "map",
                CollOp::Fold => "fold",
            },
            idx
        );
        // L1601: arity (map = 1, fold = 2 params).
        let expected_arity = match op {
            CollOp::Map => 1,
            CollOp::Fold => 2,
        };
        if params.len() != expected_arity {
            self.diags.push(diag(
                LCode::BlockArity,
                stage_span,
                format!(
                    "{} block takes {} parameter(s), found {}",
                    match op {
                        CollOp::Map => "map",
                        CollOp::Fold => "fold",
                    },
                    expected_arity,
                    params.len()
                ),
            ));
        }
        // L1604: the body must have a tail value.
        if body.tail.is_none() {
            self.diags.push(diag(
                LCode::BodyNoValue,
                body.span,
                "map/fold body has no tail value".to_string(),
            ));
        }
        // L1605: no effects (print / effectful call) inside the body.
        if let Some(sp) = body_effect_span(self.source, body, self.fn_sigs) {
            self.diags.push(diag(
                LCode::BodyEffectful,
                sp,
                "effectful operation inside a map/fold body".to_string(),
            ));
        }
        // ADR-0027: bodies may READ enclosing bindings (pure read captures —
        // broadcast edges, hidden leading body-input components). The one
        // remaining L1108 case is a WRITE to an enclosing name (an indexed
        // bind targeting it, ADR-0021's "rebind, never a fresh shadow").
        for p in params {
            self.check_not_size(*p);
        }
        // ADR-0047 D4: a size parameter is a constant in the body, never a
        // capture (seeded as a body-local name).
        let param_names: std::collections::BTreeSet<&str> = params
            .iter()
            .map(|p| name_text(self.source, *p))
            .chain(self.sizes.env.keys().map(String::as_str))
            .chain(self.types.globals().consts.keys().map(String::as_str))
            .collect();
        let caps = body_captures(self.source, body, &param_names, self.fn_sigs);
        if let Some((cap, sp)) = &caps.write {
            self.diags.push(diag(
                LCode::CaptureInBody,
                *sp,
                format!(
                    "map/fold body writes to the enclosing `{cap}` — a capture is read-only \
                     (body instances run per-element, in parallel, in an order you don't control; \
                     write your own local and pass the result out instead)"
                ),
            ));
        }
        // Resolve each read capture's ty from the enclosing scope (unknown
        // names error later in the body walk as L1101 — not duplicated here).
        // A capture keeps its literal vars (`[1.0, 2.0] -> xs` read in a
        // body as `f32`): the body's uses and the signature share them.
        let mut cap_tys: Vec<(String, Ty)> = Vec::new();
        let mut cap_ws: Vec<WTy> = Vec::new();
        for (cap, _sp) in &caps.reads {
            if let Some((w, _)) = scope.resolve(cap) {
                let w = w.clone();
                if let Some(t) = self.wty_to_ty(&w) {
                    cap_tys.push((cap.clone(), t));
                    cap_ws.push(w);
                }
            }
        }
        // Seed the body scope with the captures (leading components), so the
        // body walk types them and nested bodies can capture transitively.

        let mut body_scope: ScopeStack<WTy> = ScopeStack::new();
        for ((cap, _), w) in cap_tys.iter().zip(&cap_ws) {
            body_scope.bind(cap, w.clone());
        }
        // The fold seed and element WTys are extracted from `cur` *without*
        // resolving (so the seed's var threads into the body and unifies).
        let (seed_w, elem_w): (WTy, WTy) = match cur {
            WTy::Tuple(ws) if ws.len() == 2 => {
                let arr_elem = match &ws[1] {
                    WTy::Array(e, _) => (**e).clone(),
                    WTy::Known(Ty::Array { elem, .. }) => WTy::Known((**elem).clone()),
                    _ => WTy::Unknown,
                };
                (ws[0].clone(), arr_elem)
            }
            _ => (WTy::Unknown, WTy::Unknown),
        };
        let body_in: Ty = match op {
            CollOp::Map => {
                let elem = cur_ty.as_ref().and_then(array_elem).unwrap_or(Ty::i32());
                if let Some(p0) = params.first() {
                    body_scope.bind(name_text(self.source, *p0), WTy::Known(elem.clone()));
                }
                elem
            }
            CollOp::Fold => {
                // Bind acc to the seed WTy, t to the element WTy; walking the body
                // unifies them (acc + px.r ⇒ seed resolves to f32).
                if params.len() == 2 {
                    body_scope.bind(name_text(self.source, params[0]), seed_w.clone());
                    body_scope.bind(name_text(self.source, params[1]), elem_w.clone());
                }
                Ty::Unit // filled in after the body walk resolves acc.
            }
        };
        // Walk the body for literals + tail value ty.
        for item in &body.items {
            if let BlockItem::Stmt(s) = item {
                self.stmt(s, &mut body_scope);
            }
        }
        let tail_ty = match &body.tail {
            Some(t) => self.headless_chain_value(t, &WTy::Unknown, &mut body_scope),
            None => WTy::Unknown,
        };
        // For fold, the body result IS the acc ty; unify the tail value with the
        // seed so both resolve together.
        if matches!(op, CollOp::Fold) {
            self.unify(&seed_w, &tail_ty, stage_span);
        }
        // The body's output keeps its literal vars: a float literal in a map
        // body or a fold's seed takes its width from a later use or
        // annotation (`map { i -> 0.01 } -> start: [f32; 64]`); the block's
        // signature is read once the function's literals resolve.
        let out_fallback = match op {
            CollOp::Map => array_elem(&cur_ty.clone().unwrap_or(Ty::i32())).unwrap_or(Ty::i32()),
            CollOp::Fold => self.wty_to_ty(&seed_w).unwrap_or(Ty::f32()),
        };
        let body_out_w = match (self.wty_to_ty(&tail_ty), op) {
            (Some(_), _) => tail_ty.clone(),
            (None, CollOp::Fold) if self.wty_to_ty(&seed_w).is_some() => seed_w.clone(),
            (None, _) => WTy::Known(out_fallback.clone()),
        };
        // mask-frontend: a struct capture as its fields' captures (the body
        // scope above still types the struct by its name).
        let (cap_tys, cap_ws, records) =
            match cap_tys.iter().any(|(_, t)| matches!(t, Ty::Struct { .. })) {
                false => (cap_tys, cap_ws, Vec::new()),
                true => {
                    let mut tys = Vec::new();
                    let mut ws = Vec::new();
                    for (c, w) in cap_tys.iter().zip(&cap_ws) {
                        match &c.1 {
                            Ty::Struct { .. } => {
                                let (flat, _) = flatten_captures(std::slice::from_ref(c));
                                for f in flat {
                                    ws.push(WTy::Known(f.1.clone()));
                                    tys.push(f);
                                }
                            }
                            _ => {
                                tys.push(c.clone());
                                ws.push(w.clone());
                            }
                        }
                    }
                    let (_, records) = flatten_captures(&cap_tys);
                    (tys, ws, records)
                }
            };
        let caps: Vec<(WTy, Ty)> = cap_tys
            .iter()
            .zip(&cap_ws)
            .map(|((_, t), w)| (w.clone(), t.clone()))
            .collect();
        let comps = match op {
            CollOp::Fold => {
                let mut comps = caps;
                comps.push((seed_w.clone(), Ty::f32()));
                comps.push((elem_w.clone(), Ty::f32()));
                Some(comps)
            }
            CollOp::Map if !caps.is_empty() => {
                let mut comps = caps;
                comps.push((WTy::Known(body_in.clone()), body_in.clone()));
                Some(comps)
            }
            CollOp::Map => None,
        };
        self.pending_blocks.push(PendingBlock {
            span: (stage_span.start, stage_span.end),
            comps,
            body_in: (WTy::Known(body_in.clone()), body_in),
            body_out: (body_out_w.clone(), out_fallback),
            captures: cap_tys.into_iter().zip(cap_ws).collect(),
            records,
            name,
        });
        match op {
            CollOp::Map => WTy::Array(
                Box::new(body_out_w),
                cur_ty.as_ref().and_then(array_size).unwrap_or(1),
            ),
            CollOp::Fold => body_out_w,
        }
    }

    fn type_call(&mut self, sig: &FnSig, arg: &WTy, span: syn::SourceLoc) {
        // A zero-param fn is uncallable in Core (DESIGN §6.3): a `v -> g` call
        // stage always supplies an argument, so calling one is an arg-vs-declared
        // -input mismatch (L1201), NOT a literal-width conflict. Do not unify the
        // arg against `Unit` (that would pin a literal var and misreport L1203 —
        // ATK-18).
        // `() -> g` calls a zero-parameter fn with the Unit witness.
        if sig.params.is_empty() && matches!(arg, WTy::Known(Ty::Unit)) {
            return;
        }
        if sig.params.is_empty() {
            self.diags.push(diag(
                LCode::TypeMismatch,
                span,
                "function takes no input; the call supplies an argument".to_string(),
            ));
            return;
        }
        // unify arg with the product of param tys (single → that ty).
        let in_ty = if sig.params.len() == 1 {
            sig.params[0].clone()
        } else {
            Ty::Tuple(sig.params.clone())
        };
        self.unify_known(arg, &in_ty, span);
    }

    /// D1 result ty for a `zip` stage: `([A;n],[B;n]) → [(A,B);n]` (ADR-0018).
    /// Unknown unless `cur` resolves to a 2-tuple of equal-size arrays — emission
    /// owns the misuse L-codes (L1606/L1607/L1608); D1 only threads the ty so a
    /// downstream `map` sees the tuple element.
    fn zip_result(&mut self, cur: &WTy) -> WTy {
        let Some(Ty::Tuple(cs)) = self.wty_to_ty(cur) else {
            return WTy::Unknown;
        };
        if let [
            Ty::Array { elem: ae, size: an },
            Ty::Array { elem: be, size: bn },
        ] = cs.as_slice()
            && an == bn
        {
            return WTy::Known(Ty::Array {
                elem: Box::new(Ty::Tuple(vec![(**ae).clone(), (**be).clone()])),
                size: *an,
            });
        }
        WTy::Unknown
    }

    /// D1 result ty for an `enumerate` stage: `[A;n] → [(i32,A);n]` (ADR-0018).
    fn enumerate_result(&mut self, cur: &WTy) -> WTy {
        match self.wty_to_ty(cur) {
            Some(Ty::Array { elem, size }) => WTy::Known(Ty::Array {
                elem: Box::new(Ty::Tuple(vec![Ty::i32(), (*elem).clone()])),
                size,
            }),
            _ => WTy::Unknown,
        }
    }

    // --- expressions --------------------------------------------------------

    fn expr(&mut self, expr: &Expr, scope: &mut ScopeStack<WTy>) -> WTy {
        // ADR-0047 D4: a size expression in value position is one integer
        // literal (unpinned `Ty::index_ty`).
        if matches!(expr.kind, ExprKind::Binary { .. })
            && let Some(v) = self.size_fold(expr, scope)
        {
            let var = self.fresh_var(expr.span, false);
            if v > i128::from(i32::MAX) || v < i128::from(i32::MIN) {
                self.pin(var, &Ty::i64(), expr.span);
            }
            return WTy::Var(var);
        }
        match &expr.kind {
            ExprKind::Int(_) => WTy::Var(self.fresh_var(expr.span, false)),
            ExprKind::Float => WTy::Var(self.fresh_var(expr.span, true)),
            ExprKind::Bool(_) => WTy::Known(Ty::Bool),
            ExprKind::Str => WTy::Known(Ty::Str),
            ExprKind::Var(n) => {
                let text = name_text(self.source, *n);
                match scope.resolve(text) {
                    Some((b, _)) => b.clone(),
                    None if self.types.globals().consts.contains_key(text) => {
                        self.global_const(text, expr.span)
                    }
                    None => match self.sizes.env.get(text).copied() {
                        // ADR-0047 D4: an integer literal of the size's value;
                        // unpinned it is `Ty::index_ty(n)`.
                        Some(v) => {
                            let var = self.fresh_var(expr.span, false);
                            if v > i32::MAX as u64 {
                                self.pin(var, &Ty::i64(), expr.span);
                            }
                            WTy::Var(var)
                        }
                        None => WTy::Unknown,
                    },
                }
            }
            ExprKind::Hole => WTy::Unknown,
            ExprKind::Unary { op, operand, .. } => {
                let inner = self.expr(operand, scope);
                match op {
                    UnOp::Neg => inner,
                    UnOp::Not => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                let l = self.expr(lhs, scope);
                let r = self.expr(rhs, scope);
                self.unify(&l, &r, *op_span);
                use syn::BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Mod => l,
                    Eq | Ne | Lt | Gt | Le | Ge | And | Or => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Member { base, field } => {
                let bty = self.expr(base, scope);
                match field {
                    syn::MemberField::Named(fname) => {
                        let fn_ = name_text(self.source, *fname);
                        let bty = self.through_default(&bty, |t| match t {
                            Ty::Struct { fields, .. } => fields.iter().any(|(n, _)| n == fn_),
                            _ => false,
                        });
                        let bt = self.wty_to_ty(&bty);
                        if let Some(Ty::Struct { fields, .. }) = bt.as_ref() {
                            let fn_ = name_text(self.source, *fname);
                            if let Some((_, t)) = fields.iter().find(|(n, _)| n == fn_) {
                                return WTy::Known(t.clone());
                            }
                        }
                        WTy::Unknown
                    }
                    syn::MemberField::Index { value, .. } => {
                        // A tuple of literals keeps its vars: `acc.1` of a
                        // `(1.0, 1.0)` seed takes the width its use gives.
                        if let WTy::Tuple(ws) = &bty
                            && let Some(w) = ws.get(*value as usize)
                        {
                            return w.clone();
                        }
                        let bt = self.wty_to_ty(&bty);
                        // Skip the widening for an index > u32::MAX (it would
                        // truncate and unify against the wrong component);
                        // emit.rs rejects it with L1205 (LOWER-MEMBERIDX-TYPING-TRUNC).
                        if *value <= u32::MAX as u64
                            && let Some(t) = bt.as_ref()
                            && let Some(c) = t.component_ty(*value as u32)
                        {
                            return WTy::Known(c.clone());
                        }
                        WTy::Unknown
                    }
                }
            }
            ExprKind::Index { base, index } => {
                let bty = self.expr(base, scope);
                let bty = self.through_default(&bty, |t| matches!(t, Ty::Array { .. }));
                let iv = self.expr(index, scope);
                // index is an integer; if a lit var, leave it (defaults i32) —
                // unless the literal itself is past i32 (an extent past 2³¹
                // indexes in `i64`, its index type).
                if let Some(n) = self.int_const(index, scope)
                    && n > i32::MAX as u64
                    && n <= i64::MAX as u64
                {
                    self.unify_known(&iv, &Ty::i64(), index.span);
                }
                // An array of literals keeps its element var (`[1.0, 2.0]`
                // read as `f32`).
                if let WTy::Array(e, _) = &bty {
                    return (**e).clone();
                }
                let bt = self.wty_to_ty(&bty);
                match bt.as_ref() {
                    Some(Ty::Array { elem, .. }) => WTy::Known((**elem).clone()),
                    _ => WTy::Unknown,
                }
            }
            ExprKind::Tuple(es) => {
                // Keep components as WTys so a later annotation can reach their
                // vars (composite literal-width inference).
                let ws: Vec<WTy> = es.iter().map(|e| self.expr(e, scope)).collect();
                if ws.len() >= 2 {
                    WTy::Tuple(ws)
                } else {
                    WTy::Unknown
                }
            }
            ExprKind::Array(es) => {
                let mut ws: Vec<WTy> = Vec::with_capacity(es.len());
                for e in es {
                    let w = self.expr(e, scope);
                    if let Some(prev) = ws.first() {
                        self.unify(prev, &w, e.span);
                    }
                    ws.push(w);
                }
                match ws.into_iter().next() {
                    Some(first) => WTy::Array(Box::new(first), es.len() as u64),
                    None => WTy::Unknown,
                }
            }
            ExprKind::Struct { name, fields } => {
                let tname = name_text(self.source, *name);
                // ADR-0048 D4: a size-generic type's literal, its sizes from
                // its fields' types (where they are known).
                if let Some(td) = self
                    .types
                    .decl(tname)
                    .filter(|td| !td.generics.is_empty())
                    .cloned()
                {
                    let mut tys = Vec::new();
                    let mut ws = Vec::new();
                    for f in &td.fields {
                        let fname = name_text(self.source, f.name);
                        let w = fields
                            .iter()
                            .find(|fi| name_text(self.source, fi.name) == fname)
                            .map(|fi| match &fi.value {
                                Some(v) => self.expr(v, scope),
                                None => scope
                                    .resolve(fname)
                                    .map(|(b, _)| b.clone())
                                    .unwrap_or(WTy::Unknown),
                            })
                            .unwrap_or(WTy::Unknown);
                        // A literal field (no type yet) carries no size.
                        tys.push(match &w {
                            WTy::Var(_) => Some(Ty::i32()),
                            _ => self.wty_to_ty(&w),
                        });
                        ws.push(w);
                    }
                    let Some(tys) = tys.into_iter().collect::<Option<Vec<Ty>>>() else {
                        return WTy::Unknown;
                    };
                    let Some(ty) =
                        crate::sizes::infer_type(self.source, self.types, &td, &tys, expr.span)
                            .ok()
                            .and_then(|vs| self.types.instance(self.source, tname, &vs, expr.span))
                    else {
                        return WTy::Unknown;
                    };
                    // The fields' literals take the instance's field types.
                    if let Ty::Struct { fields: decl, .. } = &ty {
                        for (w, (_, ft)) in ws.iter().zip(decl) {
                            self.unify_known(w, ft, expr.span);
                        }
                    }
                    return WTy::Known(ty);
                }
                if let Some(sty) = self.types.get(tname).cloned() {
                    if let Ty::Struct { fields: decl, .. } = &sty {
                        for fi in fields {
                            let fname = name_text(self.source, fi.name);
                            let declty = decl
                                .iter()
                                .find(|(n, _)| n == fname)
                                .map(|(_, t)| t.clone());
                            let w = match &fi.value {
                                Some(v) => self.expr(v, scope),
                                None => scope
                                    .resolve(fname)
                                    .map(|(b, _)| b.clone())
                                    .unwrap_or(WTy::Unknown),
                            };
                            if let Some(dt) = declty {
                                self.unify_known(&w, &dt, fi.span);
                            }
                        }
                    }
                    WTy::Known(sty)
                } else {
                    WTy::Unknown
                }
            }
            // ADR-0031: `ExprKind::Call` is always P0108-rejected upstream
            // (the ADR-0029 carve is gone); no clean tree contains one.
            _ => WTy::Unknown,
        }
    }

    /// Lower an OpShorthand expr where the single `Hole` leaf is `cur`.
    fn expr_with_hole(&mut self, expr: &Expr, hole: &WTy, scope: &mut ScopeStack<WTy>) -> WTy {
        match &expr.kind {
            ExprKind::Hole => hole.clone(),
            ExprKind::Unary { op, operand, .. } => {
                let inner = self.expr_with_hole(operand, hole, scope);
                match op {
                    UnOp::Neg => inner,
                    UnOp::Not => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                let l = self.expr_with_hole(lhs, hole, scope);
                let r = self.expr_with_hole(rhs, hole, scope);
                self.unify(&l, &r, *op_span);
                use syn::BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Mod => l,
                    _ => WTy::Known(Ty::Bool),
                }
            }
            _ => self.expr(expr, scope),
        }
    }

    /// Resolve a WTy to a concrete ty if currently known (vars resolve to their
    /// current fixed state, else `None` until `resolve_all`).
    fn wty_to_ty(&mut self, w: &WTy) -> Option<Ty> {
        match w {
            WTy::Known(t) => Some(t.clone()),
            WTy::Var(v) => {
                let r = self.find(*v);
                match self.state[r].clone() {
                    LitState::Fixed(t) => Some(t),
                    LitState::IntVar => Some(Ty::i32()),
                    LitState::FloatVar => Some(Ty::f64()),
                }
            }
            WTy::Tuple(ws) => {
                let mut ts = Vec::with_capacity(ws.len());
                for cw in ws.clone() {
                    ts.push(self.wty_to_ty(&cw)?);
                }
                Some(Ty::Tuple(ts))
            }
            WTy::Array(ew, size) => {
                let e = self.wty_to_ty(ew)?;
                Some(Ty::Array {
                    elem: Box::new(e),
                    size: *size,
                })
            }
            WTy::Unknown => None,
        }
    }
}

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
fn tail_is_value(chain: &Chain) -> bool {
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

fn is_print_or_sink(_e: &Expr) -> bool {
    false
}

/// Collect return sites in a function body: `-> ret` full/slot writes (incl. in
/// loop exits and guard arms — though L1405 keeps `-> ret` out of Phi arms, a
/// routing-guard exit arm's `-> ret` counts). `full_sites` counts value-bearing
/// full returns; `value_less_full` counts bare value-less `-> ret`; `slots`
/// collects each `ret.k` index.
fn collect_ret_sites(
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

fn collect_ret_stmt(
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

fn collect_ret_chain(
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
fn arm_exit_bind(payload: &ArmPayload) -> Option<syn::Name> {
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
fn arm_final_chain(payload: &ArmPayload) -> Option<&Chain> {
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

/// The span of the first effectful stage (a direct `print`, or a call to an
/// effectful fn) inside a map/fold body (L1605), if any.
fn body_effect_span(
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
struct BodyCaptures {
    /// Read captures in first-use order (deduped).
    reads: Vec<(String, syn::SourceLoc)>,
    seen: std::collections::BTreeSet<String>,
    /// The first write-to-enclosing-name found (name, span), if any.
    write: Option<(String, syn::SourceLoc)>,
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

fn body_captures(
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

fn capture_block(
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

fn capture_stmt(
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
fn loop_exit_bind_names(source: &str, body: &Block) -> Vec<String> {
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

fn capture_chain(
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

fn capture_expr(
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
fn walk_block_stages(source: &str, block: &Block, f: &mut impl FnMut(&Stage)) {
    for item in &block.items {
        if let BlockItem::Stmt(s) = item {
            walk_stmt_stages(source, s, f);
        }
    }
    if let Some(t) = &block.tail {
        walk_chain_stages(source, t, f);
    }
}

fn walk_stmt_stages(source: &str, stmt: &Stmt, f: &mut impl FnMut(&Stage)) {
    match &stmt.kind {
        StmtKind::Chain(c) => walk_chain_stages(source, c, f),
        StmtKind::Loop(l) => walk_block_stages(source, &l.body, f),
        _ => {}
    }
}

fn walk_chain_stages(source: &str, chain: &Chain, f: &mut impl FnMut(&Stage)) {
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

fn array_size(ty: &Ty) -> Option<u64> {
    match ty {
        Ty::Array { size, .. } => Some(*size),
        _ => None,
    }
}

/// ADR-0051 D1: a generic signature's type names, checked at probe values
/// (every size 2²⁰, every format parameter the first format of its kind):
/// only the errors no instance could avoid — an unknown type, a format
/// outside a kind — are reported (a size's range depends on the instance).
fn probe_generic_sig(
    source: &str,
    types: &TypeTable,
    fd: &syn::FnDecl,
    diags: &mut Vec<syn::Diagnostic>,
) {
    let mut env = types.base_env().clone();
    for p in &fd.generics {
        let text = name_text(source, p.name).to_string();
        match crate::formats::kind_of(source, p) {
            None => {
                env.insert(text, 1 << 20);
            }
            Some(k) => {
                let kind = k.unwrap_or(crate::formats::Kind::Number);
                let first = crate::formats::Format::ALL
                    .into_iter()
                    .find(|f| kind.admits(*f))
                    .expect("every kind has a member");
                env.formats.insert(text, first);
            }
        }
    }
    let mut sink = Vec::new();
    for t in fd.params.iter().map(|p| &p.ty).chain(fd.ret_ty.as_ref()) {
        let _ = types.resolve_sized(source, t, &env, &mut sink);
    }
    for d in sink {
        let keep = d.code.0 == LCode::UnknownType.code() || d.code.0 == LCode::FormatKind.code();
        if keep && !diags.iter().any(|x| x.span == d.span && x.code == d.code) {
            diags.push(d);
        }
    }
}

/// Build the surface fn signature table (resolved param/return tys) used by D1
/// call typing.
pub fn build_fn_sigs(
    source: &str,
    types: &TypeTable,
    effects: &crate::effects::Effects,
    program: &Program,
    diags: &mut Vec<syn::Diagnostic>,
) -> BTreeMap<String, FnSig> {
    let mut out = BTreeMap::new();
    for item in &program.items {
        if let syn::Item::Fn(fd) = item {
            let name = name_text(source, fd.name).to_string();
            if !fd.generics.is_empty() {
                // ADR-0047: each instance resolves its own signature. Its
                // types are checked once here, at probe values (ADR-0051 D1:
                // a name in a type position is a format, a declared type or
                // a declared format parameter).
                probe_generic_sig(source, types, fd, diags);
                out.insert(
                    name.clone(),
                    FnSig {
                        params: Vec::new(),
                        ret: None,
                        effectful: effects.is_effectful(&name),
                        generic: true,
                    },
                );
                continue;
            }
            let mut params = Vec::new();
            for p in &fd.params {
                if let Some(t) = types.resolve(source, &p.ty, diags) {
                    params.push(t);
                } else {
                    params.push(Ty::i32());
                }
            }
            let ret = fd
                .ret_ty
                .as_ref()
                .and_then(|t| types.resolve(source, t, diags));
            out.insert(
                name.clone(),
                FnSig {
                    params,
                    ret,
                    effectful: effects.is_effectful(&name),
                    generic: false,
                },
            );
        }
    }
    out
}

/// ADR-0051 D6 (L1713): a removed format-named conversion, naming its
/// replacement.
pub fn removed_conversion(
    text: &str,
    verb: Verb,
    f: Format,
    span: syn::SourceLoc,
) -> syn::Diagnostic {
    let alt = match (verb, f) {
        (Verb::Widen, Format::F32) => {
            " — or `to<f32>` where the value may not be exact (an `i32` past ±2²⁴)".to_string()
        }
        (Verb::Widen, _) => " — or `to<…>` where the conversion may round".to_string(),
        _ => String::new(),
    };
    diag(
        LCode::RemovedConversion,
        span,
        format!(
            "`{text}` was removed: write `{}<{f}>`{alt} (ADR-0051 D6: the format is the \
             conversion's argument)",
            verb.name()
        ),
    )
}

/// ADR-0051 D2 (L1712): explicit arguments `name<…>` on a stage that takes
/// none — a value, a fixed function, a builtin without generic parameters,
/// or no function at all.
pub fn no_generic_args(
    source: &str,
    text: &str,
    local: bool,
    function: bool,
    span: syn::SourceLoc,
) -> syn::Diagnostic {
    let _ = source;
    let what = if local {
        format!("`{text}` is a value")
    } else if function {
        format!("`{text}` declares no generic parameters")
    } else if crate::is_reserved(text) {
        format!("the builtin `{text}` takes no generic arguments")
    } else {
        format!("`{text}` is no generic function")
    };
    diag(
        LCode::GenericArgs,
        span,
        format!(
            "{what}: explicit arguments `<…>` apply to a generic function (`fn f<n, t: float>`) \
             or to `to` / `widen` (ADR-0051 D2)"
        ),
    )
}
