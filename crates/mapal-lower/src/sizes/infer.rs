//! Size inference from argument types (ADR-0047 D3): equations, inversion, solving, instance signatures.

use super::*;

/// One equation `e = n` of a call: a declared dimension against the actual
/// length.
pub(super) struct Equation {
    expr: SizeExpr,
    n: u64,
    /// Where it came from: `x: [f32; n]` (a parameter) or the output type.
    from: String,
}

/// ADR-0051 D3: a format parameter in a type position against the actual
/// scalar format there.
pub(super) struct FormatEq {
    param: String,
    fmt: Format,
    from: String,
}

/// The equations of one call: sizes and formats.
#[derive(Default)]
pub(super) struct Equations {
    sizes: Vec<Equation>,
    formats: Vec<FormatEq>,
}

/// Match a declared type against an actual one, collecting the equations of
/// every size-expression dimension (D3.1) and (ADR-0051 D3) of every format
/// parameter in a type position (`fparams`, the declaration's). Literal
/// dimensions and fixed scalars are the builder's to check (a mismatch there
/// is its L1201). ADR-0048: a size-generic type's fields are matched with
/// its arguments substituted (D4), and an actual struct with a `_` field that
/// does not fit is read as its `_` (D5).
pub(super) fn collect(
    source: &str,
    types: &TypeTable,
    fparams: &BTreeMap<String, Kind>,
    pattern: &syn::Ty,
    actual: &Ty,
    from: &str,
    out: &mut Equations,
) -> Result<(), ()> {
    let r = collect_exact(source, types, fparams, pattern, actual, from, out);
    if r.is_err()
        && let Some(inner) = TypeTable::default_of(actual)
    {
        return collect(source, types, fparams, pattern, inner, from, out);
    }
    r
}

pub(super) fn collect_exact(
    source: &str,
    types: &TypeTable,
    fparams: &BTreeMap<String, Kind>,
    pattern: &syn::Ty,
    actual: &Ty,
    from: &str,
    out: &mut Equations,
) -> Result<(), ()> {
    match (&pattern.kind, actual) {
        (TyKind::Named(n), _) if fparams.contains_key(name_text(source, *n)) => {
            let fmt = Format::of(actual).ok_or(())?;
            out.formats.push(FormatEq {
                param: name_text(source, *n).to_string(),
                fmt,
                from: from.to_string(),
            });
            Ok(())
        }
        (TyKind::Array { elem, size, .. }, Ty::Array { elem: ae, size: n }) => {
            if let Some(e) = size {
                out.sizes.push(Equation {
                    expr: e.clone(),
                    n: *n,
                    from: from.to_string(),
                });
            }
            collect(source, types, fparams, elem, ae, from, out)
        }
        (TyKind::Array { .. }, _) => Err(()),
        (TyKind::Tuple(ps), Ty::Tuple(ts)) if ps.len() == ts.len() => {
            for (p, t) in ps.iter().zip(ts) {
                collect(source, types, fparams, p, t, from, out)?;
            }
            Ok(())
        }
        (TyKind::Tuple(_), _) => Err(()),
        (TyKind::Generic { name, args }, _) => {
            let Some(td) = types.decl(name_text(source, *name)) else {
                return Ok(());
            };
            if td.generics.len() != args.len() {
                return Ok(());
            }
            let by: BTreeMap<String, SizeExpr> = td
                .generics
                .iter()
                .zip(args)
                .map(|(p, a)| (name_text(source, p.name).to_string(), a.clone()))
                .collect();
            match (td.fields.as_slice(), actual) {
                ([only], _) if name_text(source, only.name) == "_" => collect(
                    source,
                    types,
                    fparams,
                    &subst_ty(source, &only.ty, &by),
                    actual,
                    from,
                    out,
                ),
                (fs, Ty::Struct { fields, .. }) if fs.len() == fields.len() => {
                    for (f, (_, at)) in fs.iter().zip(fields) {
                        collect(
                            source,
                            types,
                            fparams,
                            &subst_ty(source, &f.ty, &by),
                            at,
                            from,
                            out,
                        )?;
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
pub(super) fn invert(
    source: &str,
    e: &SizeExpr,
    n: i128,
    p: &str,
    env: &SizeEnv,
) -> Result<i128, Inversion> {
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

pub(super) enum Inversion {
    Overflow,
    Zero,
    Fraction { n: i128, k: i128 },
}

/// ADR-0048 D4: the arguments of a literal of generic type `td` whose
/// fields (in declared order) have types `fields` — inferred as a call's
/// would be, the fields standing for parameters (ADR-0051: a format
/// parameter from the field's scalar format).
pub fn infer_type(
    source: &str,
    types: &TypeTable,
    td: &syn::TypeDecl,
    fields: &[Ty],
    span: syn::SourceLoc,
) -> Result<Vec<Arg>, syn::Diagnostic> {
    let fd = syn::FnDecl {
        name: td.name,
        generics: td.generics.clone(),
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
    infer(source, types, &fd, &arg, None, &[], span)
}

/// D3: the arguments of a call to `fd` (written in `source`) whose argument
/// has type `arg` — ADR-0051: the `explicit` prefix first, then matched
/// against the parameter types (sizes solved, formats bound), then (where
/// parameters stay unbound) against the output type with `expected`, the
/// call's annotated binding or the owner's output. Values in the declared
/// order; `span` places errors.
pub fn infer(
    source: &str,
    types: &TypeTable,
    fd: &syn::FnDecl,
    arg: &Ty,
    expected: Option<&Ty>,
    explicit: &[Arg],
    span: syn::SourceLoc,
) -> Result<Vec<Arg>, syn::Diagnostic> {
    let fname = name_text(source, fd.name);
    let fparams = format_params(source, &fd.generics);
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
    // ADR-0051 D2: the explicit prefix binds first.
    let mut env: SizeEnv = types.base_env().clone();
    let mut forigin: BTreeMap<String, String> = BTreeMap::new();
    for (p, a) in fd.generics.iter().zip(explicit) {
        let text = name_text(source, p.name).to_string();
        match a {
            Arg::Size(n) => {
                env.insert(text, *n);
            }
            Arg::Format(f) => {
                env.formats.insert(text.clone(), *f);
                forigin.insert(text, "the stage's explicit argument".into());
            }
        }
    }
    // D3.1: one equation per size-expression dimension and per format
    // parameter in a type position.
    let mut eqs = Equations::default();
    let from = |p: &syn::Param| {
        format!(
            "`{}: {}`",
            name_text(source, p.name),
            src_slice(source, p.ty.span)
        )
    };
    match fd.params.as_slice() {
        [] => return Err(mismatch(arg, "no input")),
        [p] => collect(source, types, &fparams, &p.ty, arg, &from(p), &mut eqs)
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
                collect(source, types, &fparams, &p.ty, t, &from(p), &mut eqs)
                    .map_err(|()| mismatch(t, &format!("`{}`", src_slice(source, p.ty.span))))?;
            }
        }
    }
    bind_formats(fname, &eqs.formats, &mut env, &mut forigin, span)?;
    let mut origin: BTreeMap<String, usize> = BTreeMap::new();
    solve(source, fname, &eqs.sizes, &mut env, &mut origin, span)?;
    // D3.4: the output type against the annotation, where still needed.
    let declared: Vec<String> = size_names(source, &fd.generics)
        .into_iter()
        .map(str::to_string)
        .collect();
    let unbound = |env: &SizeEnv| declared.iter().find(|d| !env.contains_key(*d)).cloned();
    let unbound_format = |env: &SizeEnv| fparams.keys().find(|d| env.format(d).is_none()).cloned();
    // ADR-0051 D3: the expected type is a determination like the argument's
    // — it completes the formats still unbound, and a format it fixes
    // differently from the explicit prefix or the argument is a conflict
    // (L1709), never silently the argument's.
    if let (Some(r), Some(want)) = (&fd.ret_ty, expected) {
        let mut out_eqs = Equations::default();
        let from = format!("the result `{}`", mapal_ir::ty_label(want));
        let needed = unbound(&env).is_some() || unbound_format(&env).is_some();
        if collect(source, types, &fparams, r, want, &from, &mut out_eqs).is_ok() {
            bind_formats(fname, &out_eqs.formats, &mut env, &mut forigin, span)?;
            if needed {
                eqs.sizes.extend(out_eqs.sizes);
                solve(source, fname, &eqs.sizes, &mut env, &mut origin, span)?;
            }
        }
    }
    // ADR-0051 D3: every format parameter determined, each within its kind.
    if let Some(p) = unbound_format(&env) {
        return Err(diag(
            LCode::FormatUndetermined,
            span,
            format!(
                "cannot infer format `{p}` of `{fname}` from this call: no argument or expected \
                 type determines it — give it at the stage (`-> {fname}<…>`) or bind the result \
                 with its type (`-> y: …`, ADR-0051 D3)"
            ),
        ));
    }
    for (p, kind) in &fparams {
        let f = env.format(p).expect("bound above");
        if !kind.admits(f) {
            let from = forigin.get(p).cloned().unwrap_or_default();
            return Err(out_of_kind(p, fname, *kind, f, span, &from));
        }
    }
    // D3.3': sizes that only appear in products of two or more unknowns
    // (`rows * columns`, `tokens * columns`, `tokens * rows`): every whole
    // assignment that satisfies all the equations, found by trying the
    // divisors of one product's value — taken only when it is the one.
    let eqs = eqs.sizes;
    if unbound(&env).is_some() {
        let mut found = Vec::new();
        joint(source, fname, &eqs, &env, span, &mut found);
        if let [one] = found.as_slice() {
            env = one.clone();
        } else if found.len() > 1 {
            let show = |e: &SizeEnv| {
                declared
                    .iter()
                    .map(|d| format!("`{d}` = {}", e.get(d).copied().unwrap_or(0)))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(diag(
                LCode::SizeUnbound,
                span,
                format!(
                    "the sizes of `{fname}` are ambiguous in this call: {} and {} both fit — \
                     bind the result with its type (`-> y: [f32; …]`)",
                    show(&found[0]),
                    show(&found[1])
                ),
            ));
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
        for e in exprs
            .into_iter()
            .filter(|e| !is_format_arg(source, e, &env))
        {
            eval(source, e, &env).map_err(|mut d| {
                d.message = format!("in the output of `{fname}`: {}", d.message);
                d.span = span;
                d
            })?;
        }
    }
    Ok(fd
        .generics
        .iter()
        .map(|p| {
            let text = name_text(source, p.name);
            match env.format(text) {
                Some(f) if !p.is_size() => Arg::Format(f),
                _ => Arg::Size(env[text]),
            }
        })
        .collect())
}

/// ADR-0051 D3: bind each format equation's parameter; a second, different
/// determination is L1709.
pub(super) fn bind_formats(
    fname: &str,
    feqs: &[FormatEq],
    env: &mut SizeEnv,
    origin: &mut BTreeMap<String, String>,
    span: syn::SourceLoc,
) -> Result<(), syn::Diagnostic> {
    for e in feqs {
        match env.format(&e.param) {
            Some(f) if f != e.fmt => {
                let was = origin.get(&e.param).cloned().unwrap_or_default();
                return Err(diag(
                    LCode::FormatConflict,
                    span,
                    format!(
                        "conflicting formats in this call to `{fname}`: `{}` = `{f}` (from {was}), \
                         but {} has `{}`",
                        e.param, e.from, e.fmt
                    ),
                ));
            }
            Some(_) => {}
            None => {
                env.formats.insert(e.param.clone(), e.fmt);
                origin.insert(e.param.clone(), e.from.clone());
            }
        }
    }
    Ok(())
}

/// D3.2–3: bind and invert to a fixpoint.
pub(super) fn solve(
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

/// D3.3': the whole solutions of `eqs` past what [`solve`] binds alone, at
/// most two (enough to tell one from many). One unknown of an equation that
/// is a product of names and literals takes each divisor of the equation's
/// value in turn; [`solve`] propagates it; an assignment that still leaves
/// unknowns recurses. Equations with `+` / `-` are not searched.
pub(super) fn joint(
    source: &str,
    fname: &str,
    eqs: &[Equation],
    env: &SizeEnv,
    span: syn::SourceLoc,
    found: &mut Vec<SizeEnv>,
) {
    let unknowns = |eq: &Equation| -> Vec<String> {
        let mut used = BTreeMap::new();
        names(source, &eq.expr, &mut used);
        used.into_keys().filter(|n| !env.contains_key(n)).collect()
    };
    let pick = eqs
        .iter()
        .filter(|eq| product_only(&eq.expr))
        .map(|eq| (unknowns(eq), eq.n))
        .filter(|(u, _)| !u.is_empty())
        .min_by_key(|(u, _)| u.len());
    let Some((names_left, n)) = pick else {
        // Nothing left to search: every equation must hold as bound.
        let holds = eqs
            .iter()
            .all(|eq| matches!(value(source, &eq.expr, env), Ok(Some(v)) if v == i128::from(eq.n)));
        if holds && !found.contains(env) {
            found.push(env.clone());
        }
        return;
    };
    let p = &names_left[0];
    for v in divisors(n) {
        if found.len() > 1 {
            return;
        }
        let mut trial = env.clone();
        trial.insert(p.clone(), v);
        let mut origin = BTreeMap::new();
        if solve(source, fname, eqs, &mut trial, &mut origin, span).is_err() {
            continue;
        }
        joint(source, fname, eqs, &trial, span, found);
    }
}

/// Whether `e` is built from names and literals by `*` alone.
pub(super) fn product_only(e: &SizeExpr) -> bool {
    match &e.kind {
        SizeKind::Name(_) | SizeKind::Int(_) => true,
        SizeKind::Binary { op, lhs, rhs } => {
            matches!(op, SizeOp::Mul) && product_only(lhs) && product_only(rhs)
        }
    }
}

/// The divisors of `n`, ascending.
pub(super) fn divisors(n: u64) -> Vec<u64> {
    let (mut small, mut large) = (Vec::new(), Vec::new());
    let mut d = 1u64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            small.push(d);
            if d != n / d {
                large.push(n / d);
            }
        }
        d += 1;
    }
    small.extend(large.into_iter().rev());
    small
}

pub(super) fn contains(source: &str, e: &SizeExpr, p: &str) -> bool {
    let mut used = BTreeMap::new();
    names(source, e, &mut used);
    used.contains_key(p)
}

/// `e` with `p` at `v` (the other names at their values in `env`).
pub(super) fn value_at(
    source: &str,
    e: &SizeExpr,
    p: &str,
    v: i128,
    env: &SizeEnv,
) -> Option<i128> {
    let mut env = env.clone();
    env.insert(p.to_string(), u64::try_from(v).ok()?);
    value(source, e, &env).ok().flatten()
}

/// Whether `e` is non-decreasing in `p` over `p ≥ 1`: every subtrahend free
/// of `p`, every factor non-negative at `p = 1` (hence everywhere after).
pub(super) fn monotone(source: &str, e: &SizeExpr, p: &str, env: &SizeEnv) -> bool {
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

pub(super) enum Search {
    Found(i128),
    None,
    /// More than one value (the expression is flat in `p` there).
    Many,
}

/// The `p ≥ 1` with `e(p) = n` for a non-decreasing `e` (an overflow counts as
/// past every `n`).
pub(super) fn search(source: &str, e: &SizeExpr, n: i128, p: &str, env: &SizeEnv) -> Search {
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
