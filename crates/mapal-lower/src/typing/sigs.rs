//! Function signature collection and generic-call checks.

use super::*;

pub(super) fn array_size(ty: &Ty) -> Option<u64> {
    match ty {
        Ty::Array { size, .. } => Some(*size),
        _ => None,
    }
}

/// ADR-0051 D1: a generic signature's type names, checked at probe values
/// (every size 2²⁰, every format parameter the first format of its kind):
/// only the errors no instance could avoid — an unknown type, a format
/// outside a kind — are reported (a size's range depends on the instance).
pub(super) fn probe_generic_sig(
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
