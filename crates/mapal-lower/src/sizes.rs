//! ADR-0047 — size-generic functions: size expressions (D2), inference of a
//! call's sizes from its argument's type (D3), instance names (D6) and the
//! declaration checks (D1).
//!
//! Everything here is exact integer arithmetic over `i128` with checked
//! products; a size lies in `1 ..= 2⁶³ − 1`. Monomorphization itself (the
//! rounds, D5) lives in `lib.rs`; typing and emission call [`infer`] at a call
//! to a generic function, typing with its own types, emission with the
//! builder's.

use std::collections::{BTreeMap, BTreeSet};

use mapal_ir::Ty;
use mapal_syntax::{self as syn, SizeExpr, SizeKind, SizeOp, TyKind};

use crate::diag::{LCode, diag};
use crate::typing::FnSig;
use crate::tys::{TypeTable, name_text, src_slice};

/// The size parameters in scope and their values (an instance's binding; empty
/// in a fixed-size function).
pub type SizeEnv = BTreeMap<String, u64>;

/// The largest size: an extent is an `i64` (`Ty::index_ty`).
const MAX_SIZE: i128 = i64::MAX as i128;

/// A call site: (owner — a function or instance name, the callee stage's span).
pub type SiteKey = (String, u32, u32);

/// One instance: the generic function's name and its size values, in the
/// declared order.
pub type Instance = (String, Vec<u64>);

/// The size-generic functions of a program, by name.
pub type Generics<'a> = BTreeMap<String, &'a syn::FnDecl>;

/// D6: `name<v₁, v₂>` — not an identifier, so no source function collides.
pub fn instance_name(generic: &str, values: &[u64]) -> String {
    let vs: Vec<String> = values.iter().map(u64::to_string).collect();
    format!("{generic}<{}>", vs.join(", "))
}

/// The size environment of an instance of `decl`: the top-level integers
/// (`base`, ADR-0048 D3) and the instance's sizes.
pub fn env_of(source: &str, decl: &syn::FnDecl, values: &[u64], base: &SizeEnv) -> SizeEnv {
    let mut env = base.clone();
    for (n, v) in decl.sizes.iter().zip(values) {
        env.insert(name_text(source, *n).to_string(), *v);
    }
    env
}

fn in_range(v: i128) -> bool {
    (1..=MAX_SIZE).contains(&v)
}

/// Evaluate a size expression in `i128` (unchecked range; `None` on an
/// unbound name or an overflow).
fn value(source: &str, e: &SizeExpr, env: &SizeEnv) -> Result<Option<i128>, ()> {
    Ok(match &e.kind {
        SizeKind::Int(n) => Some(i128::from(*n)),
        SizeKind::Name(n) => env.get(name_text(source, *n)).map(|v| i128::from(*v)),
        SizeKind::Binary { op, lhs, rhs } => {
            let (Some(a), Some(b)) = (value(source, lhs, env)?, value(source, rhs, env)?) else {
                return Ok(None);
            };
            Some(
                match op {
                    SizeOp::Add => a.checked_add(b),
                    SizeOp::Sub => a.checked_sub(b),
                    SizeOp::Mul => a.checked_mul(b),
                }
                .ok_or(())?,
            )
        }
    })
}

/// D2: a size expression's value under `env`. An undeclared name is L1701, a
/// value outside `1 ..= 2⁶³ − 1` L1707.
pub fn eval(source: &str, e: &SizeExpr, env: &SizeEnv) -> Result<u64, syn::Diagnostic> {
    if let Some(n) = first_unknown(source, e, env) {
        let text = name_text(source, n);
        return Err(diag(
            LCode::UnknownSize,
            n.span,
            format!(
                "`{text}` is not a size here: a size is an integer literal, a size parameter \
                 declared after the function or type name (`fn f<{text}>(x: [f32; {text}])`, \
                 ADR-0047) or a top-level integer (`64 -> {text};`, ADR-0048)"
            ),
        ));
    }
    let text = src_slice(source, e.span);
    match value(source, e, env) {
        Ok(Some(v)) if in_range(v) => Ok(v as u64),
        Ok(Some(v)) => Err(diag(
            LCode::SizeOutOfRange,
            e.span,
            format!("the size `{text}` is {v} here; an array size is at least 1"),
        )),
        _ => Err(diag(
            LCode::SizeOutOfRange,
            e.span,
            format!("the size `{text}` overflows: an array size is at most 2⁶³ − 1"),
        )),
    }
}

fn first_unknown(source: &str, e: &SizeExpr, env: &SizeEnv) -> Option<syn::Name> {
    match &e.kind {
        SizeKind::Int(_) => None,
        SizeKind::Name(n) => (!env.contains_key(name_text(source, *n))).then_some(*n),
        SizeKind::Binary { lhs, rhs, .. } => {
            first_unknown(source, lhs, env).or_else(|| first_unknown(source, rhs, env))
        }
    }
}

/// Every name a size expression reads, with its occurrence count.
fn names(source: &str, e: &SizeExpr, out: &mut BTreeMap<String, usize>) {
    match &e.kind {
        SizeKind::Int(_) => {}
        SizeKind::Name(n) => *out.entry(name_text(source, *n).to_string()).or_default() += 1,
        SizeKind::Binary { lhs, rhs, .. } => {
            names(source, lhs, out);
            names(source, rhs, out);
        }
    }
}

/// Every size expression in a surface type, outermost dimension first (a
/// size-generic type's arguments included, ADR-0048 D4).
fn size_exprs<'t>(ty: &'t syn::Ty, out: &mut Vec<&'t SizeExpr>) {
    match &ty.kind {
        TyKind::Array { elem, size, .. } => {
            if let Some(e) = size {
                out.push(e);
            }
            size_exprs(elem, out);
        }
        TyKind::Generic { args, .. } => out.extend(args.iter()),
        TyKind::Tuple(ts) => ts.iter().for_each(|t| size_exprs(t, out)),
        TyKind::Named(_) | TyKind::Dynamic(_) | TyKind::Error => {}
    }
}

/// ADR-0048 D4: `e` with every size parameter of a generic type replaced by
/// its argument at a use (`rows * columns` of `Matrix<n, 64>` is `n * 64`).
fn subst_size(source: &str, e: &SizeExpr, by: &BTreeMap<String, SizeExpr>) -> SizeExpr {
    match &e.kind {
        SizeKind::Name(n) => match by.get(name_text(source, *n)) {
            Some(a) => a.clone(),
            None => e.clone(),
        },
        SizeKind::Int(_) => e.clone(),
        SizeKind::Binary { op, lhs, rhs } => SizeExpr {
            kind: SizeKind::Binary {
                op: *op,
                lhs: Box::new(subst_size(source, lhs, by)),
                rhs: Box::new(subst_size(source, rhs, by)),
            },
            span: e.span,
        },
    }
}

/// [`subst_size`] over a surface type.
fn subst_ty(source: &str, t: &syn::Ty, by: &BTreeMap<String, SizeExpr>) -> syn::Ty {
    let kind = match &t.kind {
        TyKind::Array {
            elem,
            len,
            len_span,
            size,
        } => TyKind::Array {
            elem: Box::new(subst_ty(source, elem, by)),
            len: *len,
            len_span: *len_span,
            size: size.as_ref().map(|e| subst_size(source, e, by)),
        },
        TyKind::Generic { name, args } => TyKind::Generic {
            name: *name,
            args: args.iter().map(|a| subst_size(source, a, by)).collect(),
        },
        TyKind::Tuple(ts) => TyKind::Tuple(ts.iter().map(|x| subst_ty(source, x, by)).collect()),
        k => k.clone(),
    };
    syn::Ty { kind, span: t.span }
}

/// D1: the declaration checks of a size-generic function — duplicate or
/// clashing size names (L1702), undeclared names in its signature (L1701),
/// size parameters its signature never mentions (L1706).
pub fn check_decl(
    source: &str,
    fd: &syn::FnDecl,
    fn_names: &BTreeSet<String>,
    base: &SizeEnv,
    tops: &crate::globals::Globals,
    diags: &mut Vec<syn::Diagnostic>,
) {
    let fname = name_text(source, fd.name);
    let params: BTreeSet<&str> = fd
        .params
        .iter()
        .map(|p| name_text(source, p.name))
        .collect();
    let mut declared: BTreeSet<&str> = BTreeSet::new();
    for n in &fd.sizes {
        let text = name_text(source, *n);
        if crate::is_reserved(text) {
            diags.push(diag(
                LCode::ReservedName,
                n.span,
                format!("`{text}` is a reserved name"),
            ));
        } else if tops.is_top(text) {
            diags.push(diag(
                LCode::TopNameReused,
                n.span,
                format!(
                    "`{text}` is a top-level name; a size parameter of `{fname}` cannot reuse it \
                     (ADR-0048 D1)"
                ),
            ));
        } else if !declared.insert(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                n.span,
                format!("size parameter `{text}` is declared twice"),
            ));
        } else if params.contains(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                n.span,
                format!("size parameter `{text}` has the name of a parameter of `{fname}`"),
            ));
        } else if fn_names.contains(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                n.span,
                format!("size parameter `{text}` has the name of a function"),
            ));
        }
    }
    let mut exprs = Vec::new();
    for p in &fd.params {
        size_exprs(&p.ty, &mut exprs);
    }
    if let Some(r) = &fd.ret_ty {
        size_exprs(r, &mut exprs);
    }
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    for e in &exprs {
        names(source, e, &mut used);
        let mut env: SizeEnv = base.clone();
        env.extend(declared.iter().map(|d| (d.to_string(), 1)));
        if let Some(n) = first_unknown(source, e, &env) {
            let text = name_text(source, n);
            diags.push(diag(
                LCode::UnknownSize,
                n.span,
                format!(
                    "`{text}` is not a size parameter of `{fname}`: declare it — \
                     `fn {fname}<…, {text}>(…)` (ADR-0047: no implicit size parameters)"
                ),
            ));
        }
    }
    for n in &fd.sizes {
        let text = name_text(source, *n);
        if !used.contains_key(text) {
            diags.push(diag(
                LCode::SizeUnbound,
                n.span,
                format!(
                    "size parameter `{text}` appears in no parameter or output type of \
                     `{fname}`, so no call can determine it"
                ),
            ));
        }
    }
}

/// One equation `e = n` of a call: a declared dimension against the actual
/// length.
struct Equation {
    expr: SizeExpr,
    n: u64,
    /// Where it came from: `x: [f32; n]` (a parameter) or the output type.
    from: String,
}

/// Match a declared type against an actual one, collecting the equations of
/// every size-expression dimension (D3.1). Literal dimensions and scalars are
/// the builder's to check (a mismatch there is its L1201). ADR-0048: a
/// size-generic type's fields are matched with its sizes substituted (D4),
/// and an actual struct with a `_` field that does not fit is read as its
/// `_` (D5).
fn collect(
    source: &str,
    types: &TypeTable,
    pattern: &syn::Ty,
    actual: &Ty,
    from: &str,
    out: &mut Vec<Equation>,
) -> Result<(), ()> {
    let r = collect_exact(source, types, pattern, actual, from, out);
    if r.is_err()
        && let Some(inner) = TypeTable::default_of(actual)
    {
        return collect(source, types, pattern, inner, from, out);
    }
    r
}

fn collect_exact(
    source: &str,
    types: &TypeTable,
    pattern: &syn::Ty,
    actual: &Ty,
    from: &str,
    out: &mut Vec<Equation>,
) -> Result<(), ()> {
    match (&pattern.kind, actual) {
        (TyKind::Array { elem, size, .. }, Ty::Array { elem: ae, size: n }) => {
            if let Some(e) = size {
                out.push(Equation {
                    expr: e.clone(),
                    n: *n,
                    from: from.to_string(),
                });
            }
            collect(source, types, elem, ae, from, out)
        }
        (TyKind::Array { .. }, _) => Err(()),
        (TyKind::Tuple(ps), Ty::Tuple(ts)) if ps.len() == ts.len() => {
            for (p, t) in ps.iter().zip(ts) {
                collect(source, types, p, t, from, out)?;
            }
            Ok(())
        }
        (TyKind::Tuple(_), _) => Err(()),
        (TyKind::Generic { name, args }, _) => {
            let Some(td) = types.decl(name_text(source, *name)) else {
                return Ok(());
            };
            if td.sizes.len() != args.len() {
                return Ok(());
            }
            let by: BTreeMap<String, SizeExpr> = td
                .sizes
                .iter()
                .zip(args)
                .map(|(n, a)| (name_text(source, *n).to_string(), a.clone()))
                .collect();
            match (td.fields.as_slice(), actual) {
                ([only], _) if name_text(source, only.name) == "_" => collect(
                    source,
                    types,
                    &subst_ty(source, &only.ty, &by),
                    actual,
                    from,
                    out,
                ),
                (fs, Ty::Struct { fields, .. }) if fs.len() == fields.len() => {
                    for (f, (_, at)) in fs.iter().zip(fields) {
                        collect(source, types, &subst_ty(source, &f.ty, &by), at, from, out)?;
                    }
                    Ok(())
                }
                _ => Err(()),
            }
        }
        _ => Ok(()),
    }
}

/// Invert `e = n` for its one unknown `p` (D3.3), along the path to it.
fn invert(source: &str, e: &SizeExpr, n: i128, p: &str, env: &SizeEnv) -> Result<i128, Inversion> {
    match &e.kind {
        SizeKind::Name(_) | SizeKind::Int(_) => Ok(n),
        SizeKind::Binary { op, lhs, rhs } => {
            let mut in_lhs = BTreeMap::new();
            names(source, lhs, &mut in_lhs);
            let (known, unknown, left) = if in_lhs.contains_key(p) {
                (rhs, lhs, true)
            } else {
                (lhs, rhs, false)
            };
            let k = match value(source, known, env) {
                Ok(Some(k)) => k,
                _ => return Err(Inversion::Overflow),
            };
            let target = match (op, left) {
                (SizeOp::Add, _) => n.checked_sub(k),
                (SizeOp::Sub, true) => n.checked_add(k),
                (SizeOp::Sub, false) => k.checked_sub(n),
                (SizeOp::Mul, _) => {
                    if k == 0 {
                        return Err(Inversion::Zero);
                    }
                    if n % k != 0 {
                        return Err(Inversion::Fraction { n, k });
                    }
                    Some(n / k)
                }
            }
            .ok_or(Inversion::Overflow)?;
            invert(source, unknown, target, p, env)
        }
    }
}

enum Inversion {
    Overflow,
    Zero,
    Fraction { n: i128, k: i128 },
}

/// ADR-0048 D4: the sizes of a literal of size-generic type `td` whose fields
/// (in declared order) have types `fields` — inferred as a call's would be,
/// the fields standing for parameters.
pub fn infer_type(
    source: &str,
    types: &TypeTable,
    td: &syn::TypeDecl,
    fields: &[Ty],
    span: syn::SourceLoc,
) -> Result<Vec<u64>, syn::Diagnostic> {
    let fd = syn::FnDecl {
        name: td.name,
        sizes: td.sizes.clone(),
        params: td
            .fields
            .iter()
            .map(|f| syn::Param {
                mut_span: None,
                name: f.name,
                ty: f.ty.clone(),
                span: f.span,
            })
            .collect(),
        ret_ty: None,
        body: syn::Block {
            items: Vec::new(),
            tail: None,
            span: td.span,
            compute: None,
        },
        span: td.span,
    };
    let arg = match fields {
        [one] => one.clone(),
        many => Ty::Tuple(many.to_vec()),
    };
    infer(source, types, &fd, &arg, None, span)
}

/// D3: the sizes of a call to `fd` whose argument has type `arg` — matched
/// against the parameter types, solved, then (where parameters stay unbound)
/// against the output type with `expected`, the call's annotated binding or
/// the owner's output. Values in the declared order; `span` places errors.
pub fn infer(
    source: &str,
    types: &TypeTable,
    fd: &syn::FnDecl,
    arg: &Ty,
    expected: Option<&Ty>,
    span: syn::SourceLoc,
) -> Result<Vec<u64>, syn::Diagnostic> {
    let fname = name_text(source, fd.name);
    let mismatch = |what: &Ty, want: &str| {
        diag(
            LCode::TypeMismatch,
            span,
            format!(
                "`{fname}` takes {want}; the argument is `{}`",
                mapal_ir::ty_label(what)
            ),
        )
    };
    // D3.1: one equation per size-expression dimension.
    let mut eqs: Vec<Equation> = Vec::new();
    let from = |p: &syn::Param| {
        format!(
            "`{}: {}`",
            name_text(source, p.name),
            src_slice(source, p.ty.span)
        )
    };
    match fd.params.as_slice() {
        [] => return Err(mismatch(arg, "no input")),
        [p] => collect(source, types, &p.ty, arg, &from(p), &mut eqs)
            .map_err(|()| mismatch(arg, &format!("`{}`", src_slice(source, p.ty.span))))?,
        ps => {
            let want = format!("{} parameters", ps.len());
            let Ty::Tuple(ts) = arg else {
                return Err(mismatch(arg, &want));
            };
            if ts.len() != ps.len() {
                return Err(mismatch(arg, &want));
            }
            for (p, t) in ps.iter().zip(ts) {
                collect(source, types, &p.ty, t, &from(p), &mut eqs)
                    .map_err(|()| mismatch(t, &format!("`{}`", src_slice(source, p.ty.span))))?;
            }
        }
    }
    let mut env: SizeEnv = types.base_env().clone();
    let mut origin: BTreeMap<String, usize> = BTreeMap::new();
    solve(source, fname, &eqs, &mut env, &mut origin, span)?;
    // D3.4: the output type against the annotation, where still needed.
    let declared: Vec<String> = fd
        .sizes
        .iter()
        .map(|n| name_text(source, *n).to_string())
        .collect();
    let unbound = |env: &SizeEnv| declared.iter().find(|d| !env.contains_key(*d)).cloned();
    if unbound(&env).is_some()
        && let (Some(r), Some(want)) = (&fd.ret_ty, expected)
    {
        let mut out_eqs = Vec::new();
        let from = format!("the result `{}`", mapal_ir::ty_label(want));
        if collect(source, types, r, want, &from, &mut out_eqs).is_ok() {
            eqs.extend(out_eqs);
            solve(source, fname, &eqs, &mut env, &mut origin, span)?;
        }
    }
    // D3.5: every parameter bound, every equation holds, the output well formed.
    if let Some(p) = unbound(&env) {
        return Err(diag(
            LCode::SizeUnbound,
            span,
            format!(
                "cannot infer size `{p}` of `{fname}` from this call: no argument dimension \
                 determines it — bind the result with its type (`-> y: [f32; …]`)"
            ),
        ));
    }
    for eq in &eqs {
        let got = eval(source, &eq.expr, &env).map_err(|mut d| {
            d.span = span;
            d
        })?;
        if got != eq.n {
            let text = src_slice(source, eq.expr.span);
            let mut used = BTreeMap::new();
            names(source, &eq.expr, &mut used);
            let bound: Vec<String> = declared
                .iter()
                .filter(|d| used.contains_key(*d))
                .map(|d| {
                    let at = origin
                        .get(d)
                        .map_or(String::new(), |&i| format!(" (from {})", eqs[i].from));
                    format!("`{d}` = {}{at}", env[d])
                })
                .collect();
            let have = if matches!(eq.expr.kind, SizeKind::Name(_)) {
                String::new()
            } else {
                format!(", so `{text}` = {got}")
            };
            return Err(diag(
                LCode::SizeConflict,
                span,
                format!(
                    "conflicting sizes in this call to `{fname}`: {}{have}, but {} has {} \
                     for `{text}`",
                    bound.join(", "),
                    eq.from,
                    eq.n
                ),
            ));
        }
    }
    if let Some(r) = &fd.ret_ty {
        let mut exprs = Vec::new();
        size_exprs(r, &mut exprs);
        for e in exprs {
            eval(source, e, &env).map_err(|mut d| {
                d.message = format!("in the output of `{fname}`: {}", d.message);
                d.span = span;
                d
            })?;
        }
    }
    Ok(declared.iter().map(|d| env[d]).collect())
}

/// D3.2–3: bind and invert to a fixpoint.
fn solve(
    source: &str,
    fname: &str,
    eqs: &[Equation],
    env: &mut SizeEnv,
    origin: &mut BTreeMap<String, usize>,
    span: syn::SourceLoc,
) -> Result<(), syn::Diagnostic> {
    loop {
        let mut progress = false;
        for (i, eq) in eqs.iter().enumerate() {
            let mut used = BTreeMap::new();
            names(source, &eq.expr, &mut used);
            let unknown: Vec<(&String, &usize)> =
                used.iter().filter(|(n, _)| !env.contains_key(*n)).collect();
            let [(p, count)] = unknown.as_slice() else {
                continue;
            };
            let text = src_slice(source, eq.expr.span);
            let inverted = if **count == 1 {
                invert(source, &eq.expr, i128::from(eq.n), p, env)
            } else if monotone(source, &eq.expr, p, env) {
                // D3.3: a repeated unknown (`n * n`) in a non-decreasing
                // expression: the one whole solution, by bisection.
                match search(source, &eq.expr, i128::from(eq.n), p, env) {
                    Search::Found(v) => Ok(v),
                    Search::None => {
                        return Err(diag(
                            LCode::SizeNotIntegral,
                            span,
                            format!(
                                "size `{p}` of `{fname}` is not a whole number: {} has {} for \
                                 `{text}`, and no whole `{p}` gives {}",
                                eq.from, eq.n, eq.n
                            ),
                        ));
                    }
                    Search::Many => continue,
                }
            } else {
                continue;
            };
            let v = match inverted {
                Ok(v) => v,
                Err(Inversion::Fraction { n, k }) => {
                    return Err(diag(
                        LCode::SizeNotIntegral,
                        span,
                        format!(
                            "size `{p}` of `{fname}` is not a whole number: {} has {} for \
                             `{text}`, which needs `{p}` = {n} / {k}",
                            eq.from, eq.n
                        ),
                    ));
                }
                Err(Inversion::Zero) => {
                    return Err(diag(
                        LCode::SizeConflict,
                        span,
                        format!(
                            "{} has {} for `{text}` in this call to `{fname}`, but `{text}` is 0 \
                             for every `{p}`",
                            eq.from, eq.n
                        ),
                    ));
                }
                Err(Inversion::Overflow) => {
                    return Err(diag(
                        LCode::SizeOutOfRange,
                        span,
                        format!("solving `{text}` = {} for `{p}` overflows", eq.n),
                    ));
                }
            };
            if !in_range(v) {
                return Err(diag(
                    LCode::SizeOutOfRange,
                    span,
                    format!(
                        "size `{p}` of `{fname}` would be {v} ({} has {} for `{text}`); a size \
                         is at least 1",
                        eq.from, eq.n
                    ),
                ));
            }
            env.insert((*p).clone(), v as u64);
            origin.insert((*p).clone(), i);
            progress = true;
        }
        if !progress {
            return Ok(());
        }
    }
}

fn contains(source: &str, e: &SizeExpr, p: &str) -> bool {
    let mut used = BTreeMap::new();
    names(source, e, &mut used);
    used.contains_key(p)
}

/// `e` with `p` at `v` (the other names at their values in `env`).
fn value_at(source: &str, e: &SizeExpr, p: &str, v: i128, env: &SizeEnv) -> Option<i128> {
    let mut env = env.clone();
    env.insert(p.to_string(), u64::try_from(v).ok()?);
    value(source, e, &env).ok().flatten()
}

/// Whether `e` is non-decreasing in `p` over `p ≥ 1`: every subtrahend free
/// of `p`, every factor non-negative at `p = 1` (hence everywhere after).
fn monotone(source: &str, e: &SizeExpr, p: &str, env: &SizeEnv) -> bool {
    match &e.kind {
        SizeKind::Int(_) | SizeKind::Name(_) => true,
        SizeKind::Binary { op, lhs, rhs } => match op {
            SizeOp::Add => monotone(source, lhs, p, env) && monotone(source, rhs, p, env),
            SizeOp::Sub => monotone(source, lhs, p, env) && !contains(source, rhs, p),
            SizeOp::Mul => {
                let at_one = |x: &SizeExpr| value_at(source, x, p, 1, env).is_some_and(|v| v >= 0);
                monotone(source, lhs, p, env)
                    && monotone(source, rhs, p, env)
                    && at_one(lhs)
                    && at_one(rhs)
            }
        },
    }
}

enum Search {
    Found(i128),
    None,
    /// More than one value (the expression is flat in `p` there).
    Many,
}

/// The `p ≥ 1` with `e(p) = n` for a non-decreasing `e` (an overflow counts as
/// past every `n`).
fn search(source: &str, e: &SizeExpr, n: i128, p: &str, env: &SizeEnv) -> Search {
    let at = |v: i128| value_at(source, e, p, v, env);
    let (mut lo, mut hi) = (1i128, MAX_SIZE);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if at(mid).is_none_or(|x| x >= n) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if at(lo) != Some(n) {
        Search::None
    } else if lo < MAX_SIZE && at(lo + 1) == Some(n) {
        Search::Many
    } else {
        Search::Found(lo)
    }
}

/// D4: an integer expression over size parameters and integer literals with
/// `+`, `-`, `*` — a size expression in value position — folds to one
/// literal. `size(text)` gives a size parameter's value (`None`: not one).
/// `None` unless at least one leaf is a size parameter (literal arithmetic
/// stays arithmetic) and the value fits an `i64`.
pub fn fold_value(
    source: &str,
    e: &syn::Expr,
    size: &impl Fn(&str) -> Option<u64>,
) -> Option<i128> {
    fn go(
        source: &str,
        e: &syn::Expr,
        size: &impl Fn(&str) -> Option<u64>,
        sized: &mut bool,
    ) -> Option<i128> {
        use syn::{BinOp, ExprKind};
        match &e.kind {
            ExprKind::Int(n) => Some(i128::from(*n)),
            ExprKind::Var(n) => {
                let v = size(name_text(source, *n))?;
                *sized = true;
                Some(i128::from(v))
            }
            ExprKind::Binary { op, lhs, rhs, .. } => {
                let (a, b) = (go(source, lhs, size, sized)?, go(source, rhs, size, sized)?);
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    let mut sized = false;
    let v = go(source, e, size, &mut sized)?;
    (sized && (i128::from(i64::MIN)..=MAX_SIZE).contains(&v)).then_some(v)
}

/// An instance's resolved signature (the generic's types under its sizes).
pub fn instance_sig(
    source: &str,
    types: &TypeTable,
    fd: &syn::FnDecl,
    env: &SizeEnv,
    effectful: bool,
) -> Option<FnSig> {
    let mut sink = Vec::new();
    let mut params = Vec::with_capacity(fd.params.len());
    for p in &fd.params {
        params.push(types.resolve_sized(source, &p.ty, env, &mut sink)?);
    }
    let ret = match &fd.ret_ty {
        Some(t) => Some(types.resolve_sized(source, t, env, &mut sink)?),
        None => None,
    };
    sink.is_empty().then_some(FnSig {
        params,
        ret,
        effectful,
        generic: false,
    })
}
