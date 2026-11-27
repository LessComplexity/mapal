//! ADR-0047 — size-generic functions: size expressions (D2), inference of a
//! call's sizes from its argument's type (D3), instance names (D6) and the
//! declaration checks (D1). ADR-0051 extends every step to format
//! parameters: a declared list holds sizes and formats (`<n, cache: float>`),
//! a call may give a prefix explicitly (`-> to<bf16>`), a format parameter in
//! a type position binds to the argument's scalar format there, then to the
//! expected result's.
//!
//! Everything here is exact integer arithmetic over `i128` with checked
//! products; a size lies in `1 ..= 2⁶³ − 1`. Monomorphization itself (the
//! rounds, D5) lives in `lib.rs`; typing and emission call [`infer`] at a call
//! to a generic function, typing with its own types, emission with the
//! builder's.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};

use mapal_ir::Ty;
use mapal_syntax::{self as syn, SizeExpr, SizeKind, SizeOp, TyKind};

use crate::diag::{LCode, diag};
use crate::formats::{Arg, Format, Kind, kind_of};
use crate::typing::FnSig;
use crate::tys::{TypeTable, name_text, src_slice};

mod infer;

pub use self::infer::{fold_value, infer, infer_type, instance_sig};

/// The generic parameters in scope and their values (an instance's binding;
/// the top-level integers alone in a fixed function): the sizes — the map
/// this derefs to — and the format parameters (ADR-0051).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SizeEnv {
    sizes: BTreeMap<String, u64>,
    /// ADR-0051: the format parameters' values.
    pub formats: BTreeMap<String, Format>,
}

impl Deref for SizeEnv {
    type Target = BTreeMap<String, u64>;
    fn deref(&self) -> &Self::Target {
        &self.sizes
    }
}

impl DerefMut for SizeEnv {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sizes
    }
}

impl SizeEnv {
    /// The value of format parameter `name` in this instance.
    pub fn format(&self, name: &str) -> Option<Format> {
        self.formats.get(name).copied()
    }
}

/// The largest size: an extent is an `i64` (`Ty::index_ty`).
const MAX_SIZE: i128 = i64::MAX as i128;

/// A call site: (owner — a function or instance name, the callee stage's span).
pub type SiteKey = (String, u32, u32);

/// One instance: the generic function's name and its arguments (sizes and
/// formats), in the declared order.
pub type Instance = (String, Vec<Arg>);

/// The generic functions of a program, by name.
pub type Generics<'a> = BTreeMap<String, &'a syn::FnDecl>;

/// D6: `name<v₁, v₂>` — not an identifier, so no source function collides;
/// formats by their names (ADR-0051 D5: `to<bf16, f32>`).
pub fn instance_name(generic: &str, values: &[Arg]) -> String {
    let vs: Vec<String> = values.iter().map(Arg::to_string).collect();
    format!("{generic}<{}>", vs.join(", "))
}

/// The environment of an instance of a declaration with generic list
/// `generics`: the top-level integers (`base`, ADR-0048 D3) and the
/// instance's sizes and formats.
pub fn env_of(
    source: &str,
    generics: &[syn::GenericParam],
    values: &[Arg],
    base: &SizeEnv,
) -> SizeEnv {
    let mut env = base.clone();
    for (p, v) in generics.iter().zip(values) {
        let text = name_text(source, p.name).to_string();
        match v {
            Arg::Size(n) => {
                env.insert(text, *n);
            }
            Arg::Format(f) => {
                env.formats.insert(text, *f);
            }
        }
    }
    env
}

/// The names of a declared list's size parameters.
pub fn size_names<'s>(source: &'s str, generics: &[syn::GenericParam]) -> Vec<&'s str> {
    generics
        .iter()
        .filter(|p| p.is_size())
        .map(|p| name_text(source, p.name))
        .collect()
}

/// The format parameters of a declared list with their kinds (an unknown
/// kind — L1711 at the declaration — reads as `number`).
pub fn format_params(source: &str, generics: &[syn::GenericParam]) -> BTreeMap<String, Kind> {
    generics
        .iter()
        .filter_map(|p| {
            kind_of(source, p).map(|k| {
                (
                    name_text(source, p.name).to_string(),
                    k.unwrap_or(Kind::Number),
                )
            })
        })
        .collect()
}

/// ADR-0051 D2: a stage's explicit arguments `<a₁, …>` resolved against the
/// callee's declared list `generics` (written in `decl_source`): a size
/// parameter takes a size expression under the caller's `env`; a format
/// parameter a format name or one of the caller's format parameters, within
/// its kind. More arguments than parameters is L1712.
#[allow(clippy::too_many_arguments)]
pub fn explicit_args(
    source: &str,
    args: &[SizeExpr],
    env: &SizeEnv,
    decl_source: &str,
    callee: &str,
    generics: &[syn::GenericParam],
    span: syn::SourceLoc,
) -> Result<Vec<Arg>, syn::Diagnostic> {
    if args.len() > generics.len() {
        let want: Vec<String> = generics
            .iter()
            .map(|p| src_slice(decl_source, p.span).to_string())
            .collect();
        return Err(diag(
            LCode::GenericArgs,
            span,
            format!(
                "`{callee}` takes {} generic argument(s), `{callee}<{}>`; this stage gives {}",
                generics.len(),
                want.join(", "),
                args.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(args.len());
    for (p, a) in generics.iter().zip(args) {
        let pname = name_text(decl_source, p.name);
        let atext = src_slice(source, a.span);
        match kind_of(decl_source, p) {
            None => {
                let named_format = match &a.kind {
                    SizeKind::Name(n) => {
                        let t = name_text(source, *n);
                        Format::from_name(t).is_some() || env.format(t).is_some()
                    }
                    _ => false,
                };
                if named_format {
                    return Err(diag(
                        LCode::FormatKind,
                        a.span,
                        format!(
                            "`{pname}` of `{callee}` is a size; `{atext}` is a format (ADR-0051 D2: the arguments follow the declared list)"
                        ),
                    ));
                }
                out.push(Arg::Size(eval(source, a, env)?));
            }
            Some(kind) => {
                let kind = kind.unwrap_or(Kind::Number);
                let fmt = match &a.kind {
                    SizeKind::Name(n) => {
                        let t = name_text(source, *n);
                        Format::from_name(t).or_else(|| env.format(t))
                    }
                    _ => None,
                };
                let Some(fmt) = fmt else {
                    return Err(diag(
                        LCode::FormatKind,
                        a.span,
                        format!(
                            "`{pname}` of `{callee}` is a format (`{}`); `{atext}` is not a format — write one of {} or a format parameter of the caller (ADR-0051 D2)",
                            kind.name(),
                            kind.members()
                        ),
                    ));
                };
                if !kind.admits(fmt) {
                    return Err(out_of_kind(pname, callee, kind, fmt, a.span, "given"));
                }
                out.push(Arg::Format(fmt));
            }
        }
    }
    Ok(out)
}

/// L1710: a format outside its parameter's kind.
fn out_of_kind(
    pname: &str,
    callee: &str,
    kind: Kind,
    fmt: Format,
    span: syn::SourceLoc,
    from: &str,
) -> syn::Diagnostic {
    diag(
        LCode::FormatKind,
        span,
        format!(
            "`{pname}` of `{callee}` is a `{}` ({}); `{fmt}` ({from}) is not one (ADR-0051 D1)",
            kind.name(),
            kind.members()
        ),
    )
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

/// ADR-0051: a generic type's argument that is a format — a format name or
/// a format parameter in `env` (`Matrix<r, c, bf16>`, `Matrix<r, c, cache>`).
pub fn is_format_arg(source: &str, e: &SizeExpr, env: &SizeEnv) -> bool {
    match &e.kind {
        SizeKind::Name(n) => {
            let t = name_text(source, *n);
            Format::from_name(t).is_some() || env.format(t).is_some()
        }
        _ => false,
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

/// ADR-0051 D1 (L1710): a format — a declared format parameter or a format's
/// name — read as an array length, where only a size stands. A generic type's
/// argument may be either (the type's declaration decides, checked where the
/// type resolves), so only array lengths are checked here.
pub fn check_format_lengths(
    source: &str,
    ty: &syn::Ty,
    formats: &BTreeMap<String, Kind>,
    owner: &str,
    diags: &mut Vec<syn::Diagnostic>,
) {
    fn length_names(e: &SizeExpr, out: &mut Vec<syn::Name>) {
        match &e.kind {
            SizeKind::Int(_) => {}
            SizeKind::Name(n) => out.push(*n),
            SizeKind::Binary { lhs, rhs, .. } => {
                length_names(lhs, out);
                length_names(rhs, out);
            }
        }
    }
    match &ty.kind {
        TyKind::Array { elem, size, .. } => {
            if let Some(e) = size {
                let mut found = Vec::new();
                length_names(e, &mut found);
                for n in found {
                    let text = name_text(source, n);
                    let what = match (formats.get(text), Format::from_name(text)) {
                        (Some(k), _) => format!(
                            "`{text}` is a format parameter of `{owner}` (kind `{}`)",
                            k.name()
                        ),
                        (None, Some(_)) => format!("`{text}` is a format"),
                        (None, None) => continue,
                    };
                    diags.push(diag(
                        LCode::FormatKind,
                        n.span,
                        format!(
                            "{what}, but an array's length is a size (ADR-0051 D1): \
                             declare a size parameter — `<…, {text}>` without a kind"
                        ),
                    ));
                }
            }
            check_format_lengths(source, elem, formats, owner, diags);
        }
        TyKind::Tuple(ts) => ts
            .iter()
            .for_each(|t| check_format_lengths(source, t, formats, owner, diags)),
        TyKind::Dynamic(elem) => check_format_lengths(source, elem, formats, owner, diags),
        TyKind::Generic { .. } | TyKind::Named(_) | TyKind::Error => {}
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
        // ADR-0051: a format parameter of the type, replaced by its argument
        // (a format name or the enclosing function's format parameter).
        TyKind::Named(n) => match by.get(name_text(source, *n)).map(|a| &a.kind) {
            Some(SizeKind::Name(m)) => TyKind::Named(*m),
            _ => TyKind::Named(*n),
        },
        k => k.clone(),
    };
    syn::Ty { kind, span: t.span }
}

/// D1: the declaration checks of a generic function — duplicate or clashing
/// parameter names (L1702), undeclared names in its signature's sizes
/// (L1701), size parameters its signature never mentions (L1706), and
/// (ADR-0051 D1) each format parameter's kind (L1711). A format parameter
/// need not appear in the signature: a call may give it explicitly.
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
    let declared = check_generic_names(source, &fd.generics, fname, tops, diags);
    for p in &fd.generics {
        let text = name_text(source, p.name);
        if params.contains(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                p.name.span,
                format!("generic parameter `{text}` has the name of a parameter of `{fname}`"),
            ));
        } else if fn_names.contains(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                p.name.span,
                format!("generic parameter `{text}` has the name of a function"),
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
    let formats = format_params(source, &fd.generics);
    for t in fd.params.iter().map(|p| &p.ty).chain(fd.ret_ty.as_ref()) {
        check_format_lengths(source, t, &formats, fname, diags);
    }
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    for e in &exprs {
        names(source, e, &mut used);
        let mut env: SizeEnv = base.clone();
        env.extend(
            declared
                .iter()
                .filter(|d| !formats.contains_key(**d))
                .map(|d| (d.to_string(), 1)),
        );
        if let Some(n) = first_unknown(source, e, &env) {
            let text = name_text(source, n);
            // A format argument of a generic type (`Matrix<r, c, cache>`) is
            // a format, not a size.
            if formats.contains_key(text) || Format::from_name(text).is_some() {
                continue;
            }
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
    for p in fd.generics.iter().filter(|p| p.is_size()) {
        let text = name_text(source, p.name);
        if !used.contains_key(text) {
            diags.push(diag(
                LCode::SizeUnbound,
                p.name.span,
                format!(
                    "size parameter `{text}` appears in no parameter or output type of \
                     `{fname}`, so no call can determine it"
                ),
            ));
        }
    }
}

/// D1 (ADR-0047 / ADR-0048 D4 / ADR-0051 D1): each name of a declared
/// generic list is fresh — not reserved (L1009), not a top-level name
/// (L1802), declared once (L1702) — and a format parameter's kind is one of
/// `float`, `narrow`, `int`, `number` (L1711). Returns the declared names.
pub fn check_generic_names<'s>(
    source: &'s str,
    generics: &[syn::GenericParam],
    owner: &str,
    tops: &crate::globals::Globals,
    diags: &mut Vec<syn::Diagnostic>,
) -> BTreeSet<&'s str> {
    let mut declared: BTreeSet<&str> = BTreeSet::new();
    for p in generics {
        let text = name_text(source, p.name);
        let what = if p.is_size() {
            "size parameter"
        } else {
            "format parameter"
        };
        if crate::is_reserved(text) {
            diags.push(diag(
                LCode::ReservedName,
                p.name.span,
                format!("`{text}` is a reserved name"),
            ));
        } else if tops.is_top(text) {
            diags.push(diag(
                LCode::TopNameReused,
                p.name.span,
                format!(
                    "`{text}` is a top-level name; a {what} of `{owner}` cannot reuse it \
                     (ADR-0048 D1)"
                ),
            ));
        } else if !declared.insert(text) {
            diags.push(diag(
                LCode::DuplicateSize,
                p.name.span,
                format!("{what} `{text}` is declared twice"),
            ));
        }
        if let Some(Err(k)) = kind_of(source, p) {
            diags.push(diag(
                LCode::UnknownKind,
                p.kind.map_or(p.span, |k| k.span),
                format!(
                    "`{k}` is not a format kind: a format parameter's kind is `float`, \
                     `narrow`, `int` or `number` (`{text}: float`, ADR-0051 D1)"
                ),
            ));
        }
    }
    declared
}
