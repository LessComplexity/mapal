//! Pass A — the type table (DESIGN §5). Collects `type` declarations into
//! `name → Ty::Struct`, resolves field tys (`TyKind → mapal_ir::Ty`), and
//! rejects duplicates, recursive type definitions, empty types/arrays, and
//! over-deep tys. `mapal_ir::Ty::Struct` carries fully-inlined nested fields, so
//! resolution fully expands struct references.

use std::collections::BTreeMap;

use mapal_ir::{MAX_TY_DEPTH, Ty};
use mapal_syntax::{self as syn, Program, Ty as SynTy, TyKind};

use crate::diag::{LCode, diag};
use crate::formats::Arg;
use crate::globals::Globals;
use crate::sizes::SizeEnv;

/// Convert a `mapal_syntax::SourceLoc` to a `mapal_ir::SourceLoc` (field-identical;
/// DESIGN §1 / D8).
pub fn ir_loc(s: syn::SourceLoc) -> mapal_ir::SourceLoc {
    mapal_ir::SourceLoc::new(s.start, s.end)
}

/// The resolved type table: surface type name → its fully-expanded
/// `Ty::Struct` — or, for an ADR-0048 alias (a type whose only field is `_`),
/// the `_` field's type itself. Size-generic declarations (ADR-0048 D4) are
/// kept as declarations and instantiated at each use's sizes. Keyed
/// deterministically (BTreeMap; I12/§11).
pub struct TypeTable {
    map: BTreeMap<String, Ty>,
    /// Every well-formed declaration (generic ones included), for
    /// instantiation at a use's sizes.
    decls: BTreeMap<String, syn::TypeDecl>,
    cyclic: std::collections::BTreeSet<String>,
    /// ADR-0048: the program's top-level values.
    globals: Globals,
}

impl TypeTable {
    /// Look up a resolved (fixed-size) type by surface name.
    pub fn get(&self, name: &str) -> Option<&Ty> {
        self.map.get(name)
    }

    /// ADR-0048: the program's top-level values.
    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    /// The size environment outside every size-generic function: the
    /// top-level positive integers (ADR-0048 D3).
    pub fn base_env(&self) -> &SizeEnv {
        &self.globals.sizes
    }

    /// ADR-0048 D5: the type a value of type `ty` reads as through its
    /// default field — `ty._` when `ty` is a struct with a `_` field.
    pub fn default_of(ty: &Ty) -> Option<&Ty> {
        match ty {
            Ty::Struct { fields, .. } => fields.iter().find(|(n, _)| n == "_").map(|(_, t)| t),
            _ => None,
        }
    }

    /// D5: the `_` steps (0, 1, 2, …) that make `ty` read as a type `fits`
    /// accepts — the value as is first, then `ty._`, `ty._._`, … (a chain is
    /// followed deterministically, the first fit taken).
    pub fn default_steps(ty: &Ty, fits: impl Fn(&Ty) -> bool) -> Option<usize> {
        let mut cur = ty;
        for k in 0..=MAX_DEFAULT_CHAIN {
            if fits(cur) {
                return Some(k);
            }
            cur = Self::default_of(cur)?;
        }
        None
    }
}

/// The longest `_` chain a value is read through (types nest at most
/// `MAX_TY_DEPTH` deep, so a chain is shorter).
pub const MAX_DEFAULT_CHAIN: usize = MAX_TY_DEPTH;

/// Build the type table for `program` (DESIGN §5), collecting every diagnostic.
/// Returns the table regardless (errored entries are simply absent), so later
/// passes can still resolve the well-formed ones and report their own errors.
/// `globals` gives the sizes a field may name (ADR-0048 D3).
pub fn build(
    source: &str,
    program: &Program,
    globals: &Globals,
) -> (TypeTable, Vec<syn::Diagnostic>) {
    let mut diags = Vec::new();

    // 1. Index declarations by name in source order; report duplicates (L1004).
    //    `decls` keeps the first declaration per name (deterministic).
    let mut decl_order: Vec<&syn::TypeDecl> = Vec::new();
    let mut seen_names: BTreeMap<String, ()> = BTreeMap::new();
    for item in &program.items {
        if let syn::Item::Type(td) = item {
            let name = name_text(source, td.name).to_string();
            if seen_names.contains_key(&name) {
                diags.push(diag(
                    LCode::DuplicateType,
                    td.name.span,
                    format!("duplicate type declaration `{name}`"),
                ));
                continue;
            }
            seen_names.insert(name, ());
            decl_order.push(td);
        }
    }

    // Map name → decl for reference resolution.
    let mut by_name: BTreeMap<&str, &syn::TypeDecl> = BTreeMap::new();
    for td in &decl_order {
        by_name.insert(name_text(source, td.name), *td);
    }

    // 2. Reject empty types and duplicate fields up front (L1010 / L1005);
    //    ADR-0048 D4: a size list's names (L1702 / L1706 / L1802).
    for td in &decl_order {
        if td.fields.is_empty() {
            diags.push(diag(
                LCode::EmptyType,
                td.span,
                format!(
                    "type `{}` has no fields; a zero-field type cannot be constructed",
                    name_text(source, td.name)
                ),
            ));
        }
        let mut field_seen: BTreeMap<&str, ()> = BTreeMap::new();
        for f in &td.fields {
            let fname = name_text(source, f.name);
            if field_seen.contains_key(fname) {
                diags.push(diag(
                    LCode::DuplicateField,
                    f.name.span,
                    format!("duplicate field `{fname}`"),
                ));
            } else {
                field_seen.insert(fname, ());
            }
        }
        check_type_sizes(source, td, globals, &mut diags);
    }

    // 3. Detect struct-reference cycles (L1007) via iterative DFS over the decl
    //    reference graph (a struct references another struct by Named tyref).
    let cyclic = detect_cycles(source, &decl_order, &by_name, &mut diags);

    let mut table = TypeTable {
        map: BTreeMap::new(),
        decls: decl_order
            .iter()
            .map(|td| (name_text(source, td.name).to_string(), (*td).clone()))
            .collect(),
        cyclic,
        globals: globals.clone(),
    };

    // 4. Resolve each non-cyclic, fixed-size declaration (a struct, or an
    //    alias's `_` type). Resolution recurses through referenced types.
    let mut map: BTreeMap<String, Ty> = BTreeMap::new();
    for td in &decl_order {
        let name = name_text(source, td.name).to_string();
        if table.cyclic.contains(&name) || !td.generics.is_empty() {
            continue;
        }
        if let Some(ty) = table.instantiate_at(source, td, &[], 0, &mut diags) {
            map.insert(name, ty);
        }
    }
    table.map = map;
    (table, diags)
}

/// ADR-0048 D4 / ADR-0051 D1: a generic type's declaration checks — each
/// name declared once (L1702), not a top-level name (L1802), a format
/// parameter's kind one of D1's (L1711), a size parameter used by a field
/// (L1706); a field's size names are the type's sizes or top-level integers
/// (else L1701, reported at resolution).
fn check_type_sizes(
    source: &str,
    td: &syn::TypeDecl,
    globals: &Globals,
    diags: &mut Vec<syn::Diagnostic>,
) {
    let tname = name_text(source, td.name);
    crate::sizes::check_generic_names(source, &td.generics, tname, globals, diags);
    let mut used = crate::globals::Names::default();
    used.type_decl(source, td);
    for p in td.generics.iter().filter(|p| p.is_size()) {
        let text = name_text(source, p.name);
        if !used.sizes.contains(text) {
            diags.push(diag(
                LCode::SizeUnbound,
                p.name.span,
                format!("size parameter `{text}` of `{tname}` appears in no field type"),
            ));
        }
    }
}

/// The text of a `Name` (a bare span into `source`).
pub fn name_text(source: &str, name: syn::Name) -> &str {
    &source[name.span.start as usize..name.span.end as usize]
}

/// The text of an arbitrary span into `source` (literal lexemes, etc.).
pub fn src_slice(source: &str, span: syn::SourceLoc) -> &str {
    &source[span.start as usize..span.end as usize]
}

/// Iterative DFS over the struct reference graph; returns the set of type names
/// that participate in a cycle (and reports one L1007 per cycle root reached).
fn detect_cycles(
    source: &str,
    decls: &[&syn::TypeDecl],
    by_name: &BTreeMap<&str, &syn::TypeDecl>,
    diags: &mut Vec<syn::Diagnostic>,
) -> std::collections::BTreeSet<String> {
    use std::collections::BTreeSet;
    let mut cyclic: BTreeSet<String> = BTreeSet::new();
    // 0 = white, 1 = gray (on path), 2 = black.
    let mut color: BTreeMap<String, u8> = BTreeMap::new();
    for td in decls {
        color.insert(name_text(source, td.name).to_string(), 0);
    }

    for td in decls {
        let root = name_text(source, td.name).to_string();
        if color.get(&root).copied().unwrap_or(2) != 0 {
            continue;
        }
        // Explicit stack: (node, referenced names, cursor).
        let mut path: Vec<String> = Vec::new();
        struct Frame {
            node: String,
            refs: Vec<(String, syn::SourceLoc)>,
            cursor: usize,
        }
        let mut stack: Vec<Frame> = vec![Frame {
            node: root.clone(),
            refs: struct_refs(source, by_name.get(root.as_str()).unwrap()),
            cursor: 0,
        }];
        color.insert(root.clone(), 1);
        path.push(root);

        while let Some(frame) = stack.last_mut() {
            if frame.cursor < frame.refs.len() {
                let (w, wspan) = frame.refs[frame.cursor].clone();
                frame.cursor += 1;
                // Only recurse into names that are actual struct decls.
                if !by_name.contains_key(w.as_str()) {
                    continue;
                }
                match color.get(&w).copied().unwrap_or(2) {
                    0 => {
                        color.insert(w.clone(), 1);
                        path.push(w.clone());
                        let refs = struct_refs(source, by_name.get(w.as_str()).unwrap());
                        stack.push(Frame {
                            node: w,
                            refs,
                            cursor: 0,
                        });
                    }
                    1 => {
                        // Back edge: every node from `w` onward in the path is cyclic.
                        let start = path.iter().position(|x| x == &w).unwrap_or(0);
                        for n in &path[start..] {
                            cyclic.insert(n.clone());
                        }
                        diags.push(diag(
                            LCode::RecursiveType,
                            wspan,
                            format!("recursive type definition involving `{w}`"),
                        ));
                    }
                    _ => {}
                }
            } else {
                color.insert(frame.node.clone(), 2);
                path.pop();
                stack.pop();
            }
        }
    }
    cyclic
}

/// The struct-type names referenced (transitively at the syntactic surface) by
/// a declaration's field tys, with the span of each reference.
fn struct_refs(source: &str, td: &syn::TypeDecl) -> Vec<(String, syn::SourceLoc)> {
    let mut out = Vec::new();
    for f in &td.fields {
        collect_named_refs(source, &f.ty, &mut out);
    }
    out
}

/// Collect `Named` type references in a ty (iterative; covers tuple/array nesting).
fn collect_named_refs(source: &str, ty: &SynTy, out: &mut Vec<(String, syn::SourceLoc)>) {
    let mut stack: Vec<&SynTy> = vec![ty];
    while let Some(t) = stack.pop() {
        match &t.kind {
            TyKind::Named(n) => out.push((name_text(source, *n).to_string(), n.span)),
            TyKind::Generic { name, .. } => {
                out.push((name_text(source, *name).to_string(), name.span))
            }
            TyKind::Tuple(ts) => {
                for inner in ts {
                    stack.push(inner);
                }
            }
            TyKind::Array { elem, .. } => stack.push(elem),
            TyKind::Dynamic(inner) => stack.push(inner),
            TyKind::Error => {}
        }
    }
}

/// The `TyKind ⇀ Ty` resolution skeleton (DESIGN §5): scalar / `Tuple` /
/// `Array`+L1208 / `Dynamic`|`Error`+L1000 arms and the trailing depth/L1209
/// check. A `Named` reference and a size-generic `Generic` one resolve through
/// the table's declarations ([`TypeTable::named`]).
fn resolve_tykind(
    table: &TypeTable,
    source: &str,
    ty: &SynTy,
    env: &SizeEnv,
    depth: usize,
    diags: &mut Vec<syn::Diagnostic>,
) -> Option<Ty> {
    if depth > MAX_TY_DEPTH {
        diags.push(diag(
            LCode::TypeTooDeep,
            ty.span,
            format!("type nests deeper than the {MAX_TY_DEPTH}-level limit"),
        ));
        return None;
    }
    let resolved = match &ty.kind {
        TyKind::Named(n) => {
            let text = name_text(source, *n);
            match text {
                "i32" => Ty::i32(),
                "i64" => Ty::i64(),
                "u8" => Ty::u8(),
                "f32" => Ty::f32(),
                "f64" => Ty::f64(),
                // ADR-0043: narrow float storage types.
                "bf16" => Ty::bf16(),
                "f16" => Ty::f16(),
                "f8e4m3" => Ty::f8e4m3(),
                "f8e5m2" => Ty::f8e5m2(),
                "bool" => Ty::Bool,
                // ADR-0051 D1: a format parameter at the instance's format.
                _ => match env.format(text) {
                    Some(f) => f.ty(),
                    None => table.named(source, text, n.span, &[], depth, diags)?,
                },
            }
        }
        TyKind::Generic { name, args } => {
            let text = name_text(source, *name);
            let values = table.generic_args(source, text, name.span, args, env, diags)?;
            table.named(source, text, name.span, &values, depth, diags)?
        }
        TyKind::Tuple(ts) => {
            let mut out = Vec::with_capacity(ts.len());
            for inner in ts {
                out.push(resolve_tykind(table, source, inner, env, depth + 1, diags)?);
            }
            Ty::Tuple(out)
        }
        TyKind::Array {
            elem,
            len,
            len_span,
            size,
        } => {
            // ADR-0047 D2: a size expression evaluates under the instance's
            // sizes (and the top-level integers, ADR-0048 D3).
            let n = match size {
                Some(e) => match crate::sizes::eval(source, e, env) {
                    Ok(n) => n,
                    Err(d) => {
                        diags.push(d);
                        return None;
                    }
                },
                None if *len == 0 => {
                    diags.push(diag(
                        LCode::EmptyArray,
                        *len_span,
                        "array type has size 0".to_string(),
                    ));
                    return None;
                }
                None => *len,
            };
            let e = resolve_tykind(table, source, elem, env, depth + 1, diags)?;
            Ty::Array {
                elem: Box::new(e),
                size: n,
            }
        }
        TyKind::Dynamic(_) | TyKind::Error => {
            diags.push(diag(
                LCode::UncleanTree,
                ty.span,
                "unclean type node reached lowering".to_string(),
            ));
            return None;
        }
    };
    if !depth_ok(&resolved) {
        diags.push(diag(
            LCode::TypeTooDeep,
            ty.span,
            format!("type nests deeper than the {MAX_TY_DEPTH}-level limit"),
        ));
        return None;
    }
    Some(resolved)
}

impl TypeTable {
    /// ADR-0048 D4 / ADR-0051 D2: a generic type's arguments at a use,
    /// resolved against its declared list — a size parameter's size
    /// expression evaluated under `env`, a format parameter's format name (or
    /// the enclosing function's format parameter) within its kind. An
    /// unknown type or a wrong count resolves the arguments as sizes, for
    /// [`TypeTable::named`] to report.
    fn generic_args(
        &self,
        source: &str,
        text: &str,
        span: syn::SourceLoc,
        args: &[syn::SizeExpr],
        env: &SizeEnv,
        diags: &mut Vec<syn::Diagnostic>,
    ) -> Option<Vec<Arg>> {
        let decl = self
            .decls
            .get(text)
            .filter(|td| td.generics.len() == args.len());
        let Some(td) = decl else {
            let mut values = Vec::with_capacity(args.len());
            for a in args {
                if crate::sizes::is_format_arg(source, a, env) {
                    // A wrong count is the error to report, not the format.
                    values.push(Arg::Size(1));
                    continue;
                }
                match crate::sizes::eval(source, a, env) {
                    Ok(v) => values.push(Arg::Size(v)),
                    Err(d) => {
                        diags.push(d);
                        return None;
                    }
                }
            }
            return Some(values);
        };
        match crate::sizes::explicit_args(source, args, env, source, text, &td.generics, span) {
            Ok(v) => Some(v),
            Err(d) => {
                diags.push(d);
                None
            }
        }
    }

    /// A declared type at `values` (its arguments in declared order; empty
    /// for a fixed type): L1809 when the count is wrong.
    fn named(
        &self,
        source: &str,
        text: &str,
        span: syn::SourceLoc,
        values: &[Arg],
        depth: usize,
        diags: &mut Vec<syn::Diagnostic>,
    ) -> Option<Ty> {
        if self.cyclic.contains(text) {
            // Part of a reported cycle; do not re-resolve.
            return None;
        }
        let Some(td) = self.decls.get(text) else {
            let hint = if text.starts_with(|c: char| c.is_ascii_lowercase()) {
                format!(
                    ": a type is a number format, `bool`, a declared type, or a format parameter \
                     declared after the function or type name (`fn f<…, {text}: float>`, \
                     ADR-0051)"
                )
            } else {
                String::new()
            };
            diags.push(diag(
                LCode::UnknownType,
                span,
                format!("unknown type `{text}`{hint}"),
            ));
            return None;
        };
        if td.generics.len() != values.len() {
            let want = td
                .generics
                .iter()
                .map(|p| src_slice(source, p.span))
                .collect::<Vec<_>>()
                .join(", ");
            diags.push(diag(
                LCode::TypeSizes,
                span,
                if td.generics.is_empty() {
                    format!("type `{text}` takes no sizes")
                } else {
                    format!(
                        "type `{text}` takes {} argument(s), `{text}<{want}>`; this use gives {}",
                        td.generics.len(),
                        values.len()
                    )
                },
            ));
            return None;
        }
        if values.is_empty()
            && let Some(t) = self.map.get(text)
        {
            return Some(t.clone());
        }
        self.instantiate_at(source, td, values, depth, diags)
    }

    /// Resolve declaration `td` at `values` (ADR-0048 D4; a fixed-size
    /// declaration at none): an alias is its `_` field's type, a struct is
    /// named `T<v₁, v₂>` at its sizes.
    fn instantiate_at(
        &self,
        source: &str,
        td: &syn::TypeDecl,
        values: &[Arg],
        depth: usize,
        diags: &mut Vec<syn::Diagnostic>,
    ) -> Option<Ty> {
        let env = crate::sizes::env_of(source, &td.generics, values, &self.globals.sizes);
        let tname = name_text(source, td.name);
        // ADR-0048 D5: a type whose only field is `_` is that field's type
        // (a transparent alias, both ways).
        if let [only] = td.fields.as_slice()
            && name_text(source, only.name) == "_"
        {
            return resolve_tykind(self, source, &only.ty, &env, depth + 1, diags);
        }
        let mut fields = Vec::with_capacity(td.fields.len());
        let mut field_seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for f in &td.fields {
            let fname = name_text(source, f.name);
            if !field_seen.insert(fname) {
                // Duplicate already reported; skip the dup so the struct ty is sane.
                continue;
            }
            let fty = resolve_tykind(self, source, &f.ty, &env, depth + 1, diags)?;
            fields.push((fname.to_string(), fty));
        }
        if fields.is_empty() {
            // Empty type already reported (L1010); do not register it.
            return None;
        }
        let name = if values.is_empty() {
            tname.to_string()
        } else {
            crate::sizes::instance_name(tname, values)
        };
        let ty = Ty::Struct { name, fields };
        if !depth_ok(&ty) {
            diags.push(diag(
                LCode::TypeTooDeep,
                td.span,
                format!("type `{tname}` nests deeper than the {MAX_TY_DEPTH}-level limit"),
            ));
            return None;
        }
        Some(ty)
    }
}

/// Whether `ty` is within `MAX_TY_DEPTH` (mirrors mapal-ir's intake guard; lower
/// must reject deep tys before they reach a builder call, per LD12).
pub fn depth_ok(ty: &Ty) -> bool {
    let mut stack: Vec<(&Ty, usize)> = vec![(ty, 1)];
    while let Some((t, depth)) = stack.pop() {
        if depth > MAX_TY_DEPTH {
            return false;
        }
        match t {
            Ty::Tuple(ts) => {
                for inner in ts {
                    stack.push((inner, depth + 1));
                }
            }
            Ty::Struct { fields, .. } => {
                for (_, inner) in fields {
                    stack.push((inner, depth + 1));
                }
            }
            Ty::Array { elem, .. } => stack.push((elem, depth + 1)),
            _ => {}
        }
    }
    true
}

impl TypeTable {
    /// Resolve a surface annotation ty to a `mapal_ir::Ty` using the table.
    /// Struct references resolve to the table's fully-expanded entry; scalars
    /// and composites resolve structurally. Reports L-codes into `diags`.
    pub fn resolve(
        &self,
        source: &str,
        ty: &SynTy,
        diags: &mut Vec<syn::Diagnostic>,
    ) -> Option<Ty> {
        let env = self.globals.sizes.clone();
        self.resolve_sized(source, ty, &env, diags)
    }

    /// [`TypeTable::resolve`] inside an instance of a size-generic function:
    /// a size expression evaluates under `env` (ADR-0047 D2).
    pub fn resolve_sized(
        &self,
        source: &str,
        ty: &SynTy,
        env: &SizeEnv,
        diags: &mut Vec<syn::Diagnostic>,
    ) -> Option<Ty> {
        resolve_tykind(self, source, ty, env, 0, diags)
    }

    /// ADR-0048 D4: type `name` at sizes `values` (`None` with no report
    /// when it does not resolve).
    pub fn instance(
        &self,
        source: &str,
        name: &str,
        values: &[Arg],
        span: syn::SourceLoc,
    ) -> Option<Ty> {
        self.named(source, name, span, values, 0, &mut Vec::new())
    }

    /// A type declaration by name (ADR-0048 D4: a generic type's sizes are
    /// substituted into its fields by [`crate::sizes`] inference).
    pub fn decl(&self, name: &str) -> Option<&syn::TypeDecl> {
        self.decls.get(name)
    }
}
