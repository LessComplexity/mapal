//! ADR-0048 — top-level bindings and compile-time evaluation.
//!
//! A top-level chain (`2048 -> hidden_size;`) binds program-wide immutable
//! values. They are order-independent (a dependency graph over the bindings,
//! through the functions and types they use; a cycle is L1807), never `mut`
//! (L1801), never effectful (L1806).
//!
//! **Compile-time evaluation.** A top-level value's dataflow cone has no input
//! and no effect by construction, so it is computed before the program runs —
//! by the interpreter, the oracle, on a synthetic program whose `main` holds
//! the bindings' chains (in dependency order) and whatever functions and
//! types they reach. A binding that a size names must come out that way (an
//! integer ≥ 1); one whose cone needs a size not known yet waits for a later
//! round. Scalars that evaluate become literals ([`GlobalConst`]): an
//! unannotated integer or float is a literal in every respect (its width from
//! context, ADR-0047 D4), an annotated one a constant of its type. Every
//! other top-level value — an array, a tuple, a struct, a scalar whose
//! evaluation traps, diverges or would take longer than the budget — is
//! realized where a function reads it: its chain is emitted at the function's
//! outermost scope (captured into map / fold bodies like any enclosing value),
//! and the graph-level fact `static_plan` decides by cost whether its value
//! is embedded or computed at run time (mapal-rewrite `fold_static`).

use std::collections::{BTreeMap, BTreeSet};

use mapal_ir::{Ty, Value};
use mapal_syntax::{
    self as syn, ArmPayload, Block, BlockItem, Chain, Expr, ExprKind, Program, SizeExpr, SizeKind,
    StageKind, StmtKind, TyKind,
};

use crate::diag::{LCode, diag};
use crate::sizes::SizeEnv;
use crate::tys::name_text;

/// The interpreter steps one evaluation round may spend on a top-level value
/// (the oracle runs at roughly 10⁷–10⁸ steps a second: well under a second).
pub const EVAL_BUDGET: u64 = 1 << 24;

/// A top-level scalar computed before the program runs (ADR-0048 D3).
#[derive(Clone, Debug, PartialEq)]
pub enum GlobalConst {
    /// An integer. `pinned`: the binding's annotation (`-> n: i64`); `None`
    /// is a literal whose width comes from context.
    Int {
        value: i128,
        pinned: Option<Ty>,
    },
    /// A float. `pinned` as for `Int`; `literal`: the binding is one float
    /// literal (`0.00001 -> epsilon;`), whose own text then types at each use
    /// exactly as the literal would.
    Float {
        value: f64,
        pinned: Option<Ty>,
        literal: Option<syn::SourceLoc>,
    },
    Bool(bool),
}

/// One top-level binding statement and the names it binds.
#[derive(Clone, Debug)]
pub struct TopItem {
    pub chain: Chain,
    pub span: syn::SourceLoc,
    /// The names its binding stages bind, in order.
    pub names: Vec<String>,
}

/// The program's top-level values as lowering sees them.
#[derive(Clone, Debug, Default)]
pub struct Globals {
    /// Every top-level binding statement, in source order.
    pub items: Vec<TopItem>,
    /// Every top-level name → the index of its item.
    pub owner: BTreeMap<String, usize>,
    /// The scalars computed before the program runs.
    pub consts: BTreeMap<String, GlobalConst>,
    /// The positive integers among them: usable as sizes (ADR-0047 D2) and,
    /// unpinned, as integer literals (D4).
    pub sizes: SizeEnv,
    /// The items realized by their chain where a function reads them, in
    /// dependency order.
    pub computed: Vec<usize>,
    /// The items each item reads (directly or through functions and types).
    pub deps: Vec<BTreeSet<usize>>,
}

impl Globals {
    /// Whether `name` is a top-level name.
    pub fn is_top(&self, name: &str) -> bool {
        self.owner.contains_key(name)
    }

    /// The computed items function `fd` reads ([`Self::computed_used`]).
    pub fn used_by(&self, source: &str, fd: &syn::FnDecl) -> Vec<usize> {
        if self.computed.is_empty() {
            return Vec::new();
        }
        let mut n = Names::default();
        n.block(source, &fd.body);
        self.computed_used(source, &n.values)
    }

    /// The chains of `items` as one block (their map / fold bodies are built
    /// with the reading function's).
    pub fn chains_block(&self, items: &[usize]) -> Block {
        Block {
            items: items
                .iter()
                .map(|&i| {
                    BlockItem::Stmt(syn::Stmt {
                        kind: StmtKind::Chain(self.items[i].chain.clone()),
                        span: self.items[i].span,
                    })
                })
                .collect(),
            tail: None,
            span: self
                .items
                .first()
                .map_or(syn::SourceLoc::new(0, 0), |i| i.span),
            compute: None,
        }
    }

    /// The computed items a block of a function reads, transitively, in
    /// dependency order (each emitted once at the function's outermost scope).
    pub fn computed_used(&self, source: &str, names: &BTreeSet<String>) -> Vec<usize> {
        let mut need: BTreeSet<usize> = BTreeSet::new();
        let mut stack: Vec<usize> = names
            .iter()
            .filter_map(|n| self.owner.get(n).copied())
            .collect();
        while let Some(i) = stack.pop() {
            if !self.computed.contains(&i) || !need.insert(i) {
                continue;
            }
            let mut inner = Names::default();
            inner.chain(source, &self.items[i].chain);
            stack.extend(
                inner
                    .values
                    .iter()
                    .filter_map(|n| self.owner.get(n).copied()),
            );
        }
        self.computed
            .iter()
            .copied()
            .filter(|i| need.contains(i))
            .collect()
    }
}

/// The names a piece of syntax mentions: value names (variables, called
/// functions, struct puns), type names and size names. An over-approximation
/// of what it reads (a local is never a top-level name: L1802).
#[derive(Default)]
pub struct Names {
    pub values: BTreeSet<String>,
    pub types: BTreeSet<String>,
    pub sizes: BTreeSet<String>,
}

impl Names {
    pub fn fn_decl(&mut self, source: &str, fd: &syn::FnDecl) {
        for p in &fd.params {
            self.ty(source, &p.ty);
        }
        if let Some(t) = &fd.ret_ty {
            self.ty(source, t);
        }
        self.block(source, &fd.body);
    }

    pub fn type_decl(&mut self, source: &str, td: &syn::TypeDecl) {
        for f in &td.fields {
            self.ty(source, &f.ty);
        }
    }

    pub fn ty(&mut self, source: &str, t: &syn::Ty) {
        match &t.kind {
            TyKind::Named(n) => {
                self.types.insert(name_text(source, *n).to_string());
            }
            TyKind::Generic { name, args } => {
                self.types.insert(name_text(source, *name).to_string());
                for a in args {
                    self.size(source, a);
                }
            }
            TyKind::Tuple(ts) => ts.iter().for_each(|t| self.ty(source, t)),
            TyKind::Array { elem, size, .. } => {
                if let Some(e) = size {
                    self.size(source, e);
                }
                self.ty(source, elem);
            }
            TyKind::Dynamic(e) => self.ty(source, e),
            TyKind::Error => {}
        }
    }

    fn size(&mut self, source: &str, e: &SizeExpr) {
        match &e.kind {
            SizeKind::Int(_) => {}
            SizeKind::Name(n) => {
                self.sizes.insert(name_text(source, *n).to_string());
            }
            SizeKind::Binary { lhs, rhs, .. } => {
                self.size(source, lhs);
                self.size(source, rhs);
            }
        }
    }

    pub fn block(&mut self, source: &str, b: &Block) {
        for it in &b.items {
            match it {
                BlockItem::Stmt(s) => match &s.kind {
                    StmtKind::Chain(c) => self.chain(source, c),
                    StmtKind::Bind(bs) => {
                        self.values.insert(name_text(source, bs.name).to_string());
                        if let Some(i) = &bs.index {
                            self.expr(source, i);
                        }
                        if let Some(t) = &bs.ty {
                            self.ty(source, t);
                        }
                        self.expr(source, &bs.value);
                    }
                    StmtKind::Loop(l) => self.block(source, &l.body),
                    StmtKind::Error => {}
                },
                BlockItem::Arm(a) => self.payload(source, &a.payload),
            }
        }
        if let Some(t) = &b.tail {
            self.chain(source, t);
        }
    }

    fn payload(&mut self, source: &str, p: &ArmPayload) {
        match p {
            ArmPayload::Chain(c) => self.chain(source, c),
            ArmPayload::Block(b) => self.block(source, b),
        }
    }

    pub fn chain(&mut self, source: &str, c: &Chain) {
        if let Some(h) = &c.head {
            self.expr(source, h);
        }
        for st in &c.stages {
            match &st.kind {
                StageKind::Expr(e) | StageKind::OpShorthand { expr: e } => self.expr(source, e),
                StageKind::Bind { ty, .. } => {
                    if let Some(t) = ty {
                        self.ty(source, t);
                    }
                }
                StageKind::Guard(arms) => {
                    for a in arms {
                        self.payload(source, &a.payload);
                    }
                }
                StageKind::Fanout { branches, .. } => {
                    for b in branches {
                        self.chain(source, b);
                    }
                }
                StageKind::MapFold { body, .. } => self.block(source, body),
                StageKind::SeqBlock(b) | StageKind::StmtBlock(b) => self.block(source, b),
                StageKind::Ret { .. } | StageKind::LoopJump | StageKind::Error(_) => {}
            }
        }
    }

    fn expr(&mut self, source: &str, e: &Expr) {
        match &e.kind {
            ExprKind::Var(n) => {
                self.values.insert(name_text(source, *n).to_string());
            }
            ExprKind::Unary { operand, .. } => self.expr(source, operand),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(source, lhs);
                self.expr(source, rhs);
            }
            ExprKind::Member { base, .. } => self.expr(source, base),
            ExprKind::Index { base, index } => {
                self.expr(source, base);
                self.expr(source, index);
            }
            ExprKind::Tuple(es) | ExprKind::Array(es) => {
                es.iter().for_each(|x| self.expr(source, x))
            }
            ExprKind::Struct { name, fields } => {
                self.types.insert(name_text(source, *name).to_string());
                for f in fields {
                    match &f.value {
                        Some(v) => self.expr(source, v),
                        None => {
                            self.values.insert(name_text(source, f.name).to_string());
                        }
                    }
                }
            }
            ExprKind::Call { callee, args } => {
                self.expr(source, callee);
                args.iter().for_each(|x| self.expr(source, x));
            }
            ExprKind::Question(x) => self.expr(source, x),
            ExprKind::Int(_)
            | ExprKind::Float
            | ExprKind::Str
            | ExprKind::Bool(_)
            | ExprKind::Unit
            | ExprKind::Hole
            | ExprKind::Error => {}
        }
    }
}

/// D1: collect the top-level binding statements and check their shape — a
/// binding stage at least (L1805), never `mut` (L1801), no `ret` / `loop`
/// stage (L1803), each name once and not a function's or a reserved name
/// (L1804 / L1009).
pub fn collect(
    source: &str,
    program: &Program,
    fn_names: &BTreeSet<String>,
    diags: &mut Vec<syn::Diagnostic>,
) -> Globals {
    let mut g = Globals::default();
    for item in &program.items {
        let syn::Item::Bind(tb) = item else { continue };
        let index = g.items.len();
        let mut names = Vec::new();
        for st in &tb.chain.stages {
            let bound = match &st.kind {
                StageKind::Bind { mut_span, name, .. } => Some((*mut_span, *name)),
                // A bare name that is no function or builtin binds (`-> n`).
                StageKind::Expr(Expr {
                    kind: ExprKind::Var(n),
                    ..
                }) if binds_bare(source, *n, fn_names) => Some((None, *n)),
                _ => None,
            };
            match (&st.kind, bound) {
                (_, Some((mut_span, name))) => {
                    let text = name_text(source, name).to_string();
                    if let Some(m) = mut_span {
                        diags.push(diag(
                            LCode::TopMutable,
                            m,
                            format!(
                                "a top-level binding is immutable: `{text}` cannot be `mut` \
                                 (ADR-0048 D1; mutable state lives in a function)"
                            ),
                        ));
                    }
                    if crate::is_reserved(&text) {
                        diags.push(diag(
                            LCode::ReservedName,
                            name.span,
                            format!("`{text}` is a reserved name"),
                        ));
                    } else if fn_names.contains(&text) {
                        diags.push(diag(
                            LCode::TopDuplicate,
                            name.span,
                            format!("`{text}` is already the name of a function"),
                        ));
                    } else if g.owner.contains_key(&text) {
                        diags.push(diag(
                            LCode::TopDuplicate,
                            name.span,
                            format!("the top-level name `{text}` is bound twice"),
                        ));
                    } else {
                        g.owner.insert(text.clone(), index);
                        names.push(text);
                    }
                }
                (StageKind::Ret { .. } | StageKind::LoopJump, _) => {
                    diags.push(diag(
                        LCode::TopStage,
                        st.span,
                        "`ret` and `loop` have no meaning at the top level: a top-level chain \
                         ends in a binding (`2048 -> hidden_size;`)",
                    ));
                }
                _ => {}
            }
        }
        if names.is_empty()
            && !tb
                .chain
                .stages
                .iter()
                .any(|s| stage_binds(source, s, fn_names))
        {
            diags.push(diag(
                LCode::TopNoBinding,
                tb.span,
                "a top-level chain must bind a name (`2048 -> hidden_size;`): \
                 there is nothing else it could do",
            ));
        }
        g.items.push(TopItem {
            chain: tb.chain.clone(),
            span: tb.span,
            names,
        });
    }
    g
}

/// Whether a bare-name stage `-> n` of a top-level chain binds `n` (it is no
/// function and no builtin).
fn binds_bare(source: &str, n: syn::Name, fn_names: &BTreeSet<String>) -> bool {
    let text = name_text(source, n);
    !fn_names.contains(text) && !crate::is_reserved(text)
}

fn stage_binds(source: &str, st: &syn::Stage, fn_names: &BTreeSet<String>) -> bool {
    match &st.kind {
        StageKind::Bind { .. } => true,
        StageKind::Expr(Expr {
            kind: ExprKind::Var(n),
            ..
        }) => binds_bare(source, *n, fn_names),
        _ => false,
    }
}

/// D2: the item graph — item `i` reads item `j` when its chain, or a function
/// or type it reaches, names one of `j`'s names. A cycle is L1807.
pub fn dependencies(
    source: &str,
    program: &Program,
    g: &mut Globals,
    diags: &mut Vec<syn::Diagnostic>,
) {
    // Per function / type: the names it mentions.
    let mut fn_names: BTreeMap<String, Names> = BTreeMap::new();
    let mut ty_names: BTreeMap<String, Names> = BTreeMap::new();
    for item in &program.items {
        match item {
            syn::Item::Fn(fd) => {
                let mut n = Names::default();
                n.fn_decl(source, fd);
                fn_names.insert(name_text(source, fd.name).to_string(), n);
            }
            syn::Item::Type(td) => {
                let mut n = Names::default();
                n.type_decl(source, td);
                ty_names.insert(name_text(source, td.name).to_string(), n);
            }
            _ => {}
        }
    }
    // The top-level names a function or type reaches, transitively.
    let reach = |start: &Names| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut seen_fn: BTreeSet<String> = BTreeSet::new();
        let mut seen_ty: BTreeSet<String> = BTreeSet::new();
        let mut stack: Vec<&Names> = vec![start];
        while let Some(n) = stack.pop() {
            for v in n.values.iter().chain(&n.sizes) {
                if g.owner.contains_key(v) {
                    out.insert(v.clone());
                } else if let Some(f) = fn_names.get(v)
                    && seen_fn.insert(v.clone())
                {
                    stack.push(f);
                }
            }
            for t in &n.types {
                if let Some(td) = ty_names.get(t)
                    && seen_ty.insert(t.clone())
                {
                    stack.push(td);
                }
            }
        }
        out
    };
    g.deps = g
        .items
        .iter()
        .map(|it| {
            // The chain's own names are its bindings, not its reads; read
            // back through a function or a type they are a cycle.
            let mut n = Names::default();
            n.chain(source, &it.chain);
            for own in &it.names {
                n.values.remove(own);
            }
            reach(&n)
                .iter()
                .filter_map(|v| g.owner.get(v).copied())
                .collect()
        })
        .collect();
    // L1807: a cycle (iterative three-colour DFS; one report per cycle).
    let n = g.items.len();
    let mut color = vec![0u8; n];
    for root in 0..n {
        if color[root] != 0 {
            continue;
        }
        let mut path: Vec<usize> = vec![root];
        let mut stack: Vec<(usize, Vec<usize>)> =
            vec![(root, g.deps[root].iter().copied().collect())];
        color[root] = 1;
        while let Some((node, rest)) = stack.last_mut() {
            let node = *node;
            match rest.pop() {
                Some(next) => match color[next] {
                    0 => {
                        color[next] = 1;
                        path.push(next);
                        stack.push((next, g.deps[next].iter().copied().collect()));
                    }
                    1 => {
                        let at = path.iter().position(|&p| p == next).unwrap_or(0);
                        let cycle: Vec<String> = path[at..]
                            .iter()
                            .chain(std::iter::once(&next))
                            .map(|&k| g.items[k].names.first().cloned().unwrap_or_default())
                            .collect();
                        diags.push(diag(
                            LCode::TopCycle,
                            g.items[next].span,
                            format!(
                                "top-level bindings depend on each other in a cycle: {} \
                                 (ADR-0048 D2: each is computed from the ones before it)",
                                cycle.join(" -> ")
                            ),
                        ));
                    }
                    _ => {}
                },
                None => {
                    color[node] = 2;
                    path.pop();
                    stack.pop();
                }
            }
        }
    }
}

/// The items in dependency order (sources first; ties by source order).
pub fn order(g: &Globals) -> Vec<usize> {
    let n = g.items.len();
    let mut done = vec![false; n];
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let before = out.len();
        for i in 0..n {
            if !done[i] && g.deps[i].iter().all(|&j| done[j]) {
                done[i] = true;
                out.push(i);
            }
        }
        if out.len() == before {
            // A cycle (already reported): the rest in source order.
            out.extend((0..n).filter(|&i| !done[i]));
            break;
        }
    }
    out
}

/// Turn an evaluated top-level scalar into its literal form.
pub fn as_const(
    v: &mapal_interp::RValue,
    annotated: bool,
    literal: Option<syn::SourceLoc>,
) -> Option<GlobalConst> {
    let mapal_interp::RValue::Scalar(s) = v else {
        return None;
    };
    let pinned = |t: Ty| annotated.then_some(t);
    Some(match s {
        Value::I32(x) => GlobalConst::Int {
            value: i128::from(*x),
            pinned: pinned(Ty::i32()),
        },
        Value::I64(x) => GlobalConst::Int {
            value: i128::from(*x),
            pinned: pinned(Ty::i64()),
        },
        Value::F32(x) => GlobalConst::Float {
            value: f64::from(*x),
            pinned: pinned(Ty::f32()),
            literal: if annotated { None } else { literal },
        },
        Value::F64(x) => GlobalConst::Float {
            value: *x,
            pinned: pinned(Ty::f64()),
            literal: if annotated { None } else { literal },
        },
        Value::Bool(b) => GlobalConst::Bool(*b),
        // u8, narrow floats, strings: realized by their chain.
        _ => return None,
    })
}

/// The binding stage that binds `name` in `chain` (its annotation, if any).
pub fn annotation<'c>(source: &str, chain: &'c Chain, name: &str) -> Option<Option<&'c syn::Ty>> {
    chain.stages.iter().find_map(|st| match &st.kind {
        StageKind::Bind { name: n, ty, .. } if name_text(source, *n) == name => Some(ty.as_ref()),
        StageKind::Expr(Expr {
            kind: ExprKind::Var(n),
            ..
        }) if name_text(source, *n) == name => Some(None),
        _ => None,
    })
}

/// A chain that is one float literal bound once (`0.00001 -> epsilon;`): the
/// literal's span.
pub fn float_literal(chain: &Chain) -> Option<syn::SourceLoc> {
    match (&chain.head, chain.stages.as_slice()) {
        (Some(h), [st])
            if matches!(
                st.kind,
                StageKind::Bind { ty: None, .. }
                    | StageKind::Expr(Expr {
                        kind: ExprKind::Var(_),
                        ..
                    })
            ) =>
        {
            match &h.kind {
                ExprKind::Float => Some(h.span),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A chain that is one integer literal (or its negation) bound once without an
/// annotation (`2048 -> hidden_size;`): its value — a literal at every use,
/// whatever its width (`3000000000 -> big;` is an `i64` where one is needed).
pub fn int_literal(chain: &Chain) -> Option<i128> {
    let (Some(h), [st]) = (&chain.head, chain.stages.as_slice()) else {
        return None;
    };
    if !matches!(
        st.kind,
        StageKind::Bind {
            ty: None,
            mut_span: None,
            ..
        } | StageKind::Expr(Expr {
            kind: ExprKind::Var(_),
            ..
        })
    ) {
        return None;
    }
    match &h.kind {
        ExprKind::Int(v) => Some(i128::from(*v)),
        ExprKind::Unary {
            op: syn::UnOp::Neg,
            operand,
            ..
        } => match operand.kind {
            ExprKind::Int(v) => Some(-i128::from(v)),
            _ => None,
        },
        _ => None,
    }
}
