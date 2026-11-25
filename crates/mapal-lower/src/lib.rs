//! `mapal-lower`: parse tree → sealed IR (DESIGN `docs/components/lower/DESIGN.md`).
//!
//! [`lower`] turns a **parse-clean** [`mapal_syntax::Program`] into a **sealed**
//! [`mapal_ir::CategoryIr`], or a list of lower-stage diagnostics (L-codes). It
//! runs five passes (DESIGN §2):
//!
//! - **A** type table (`tys`), **B** effects & call graph (`effects`),
//!   **C** declare + **D** per-fn typing (`typing`) and emission (`emit`),
//!   **E** seal.
//!
//! Totality (DESIGN §13): a clean tree + bounded recursion + fail-fast emission
//! ⇒ `lower` never panics or hangs. Determinism is sacred (DESIGN §11): no
//! `HashMap` iteration affects emission order — `BTreeMap`/`Vec` throughout.

mod diag;
mod effects;
mod emit;
mod globals;
mod scope;
mod sizes;
mod typing;
mod tys;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use mapal_syntax::{Diagnostic, Program};

use crate::diag::{LCode, diag};
use crate::globals::Globals;
use crate::sizes::{Generics, Instance, SiteKey};
use crate::tys::name_text;

/// ADR-0047 D5: the most lowering rounds monomorphization takes before it
/// keeps the last complete result (a program whose typing walk types every
/// generic call settles in one).
const MAX_ROUNDS: usize = 32;

/// Lower a parse-clean program to a sealed IR.
///
/// Precondition: `program` came from a `parse()` whose diagnostics were empty
/// (syntax J3: no Error nodes, no rejected-but-kept forms). Encountering such a
/// form anyway yields L1000, never a panic. Total: never panics, never hangs
/// (recursion is bounded by the parser's depth guard P0011; DESIGN §13).
///
/// ADR-0048: a program with top-level bindings first computes the ones it can
/// before it runs ([`evaluate_globals`], the interpreter on a synthetic
/// program), then lowers with them.
pub fn lower(source: &str, program: &Program) -> Result<mapal_ir::CategoryIr, Vec<Diagnostic>> {
    if !program
        .items
        .iter()
        .any(|i| matches!(i, mapal_syntax::Item::Bind(_)))
    {
        return lower_with(source, program, &Globals::default(), None);
    }
    let fn_names: BTreeSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            mapal_syntax::Item::Fn(fd) => Some(name_text(source, fd.name).to_string()),
            _ => None,
        })
        .collect();
    let mut diags: Vec<Diagnostic> = Vec::new();
    let mut g = globals::collect(source, program, &fn_names, &mut diags);
    globals::dependencies(source, program, &mut g, &mut diags);
    if let Ok(effects) = effects::analyze(source, program, &fn_names) {
        for it in &g.items {
            if effects::chain_effectful(source, &it.chain, &fn_names, &effects) {
                diags.push(diag(
                    LCode::TopEffect,
                    it.span,
                    "a top-level binding may not have an effect: `print`, `println`, `time`, `read` and \
                     effectful functions belong in a function (ADR-0048 D1)",
                ));
            }
        }
    }
    if !diags.is_empty() {
        return Err(diags);
    }
    evaluate_globals(source, program, &mut g, &mut diags);
    if !diags.is_empty() {
        return Err(diags);
    }
    lower_with(source, program, &g, None).map_err(|ds| {
        ds.into_iter()
            .map(|mut d| {
                // L1808: a size that names a top-level value which is no
                // integer ≥ 1 computed before the program runs.
                let text = tys::src_slice(source, d.span);
                if d.code.0 == LCode::UnknownSize.code() && g.is_top(text) {
                    d.code = mapal_syntax::DiagCode(LCode::NotComputable.code());
                    d.message = format!(
                        "`{text}` is a top-level value, but not an integer ≥ 1 computed before \
                         the program runs, so it cannot be a size (ADR-0048 D3: a size's value \
                         must come out of compile-time evaluation)"
                    );
                }
                d
            })
            .collect()
    })
}

/// ADR-0048 D3: compute the top-level scalars before the program runs. Each
/// round lowers a synthetic program — the types and functions whose sizes are
/// known so far, and a `main` holding the bindings whose cones they reach, in
/// dependency order — and runs each scalar binding's cone on the interpreter
/// (under [`globals::EVAL_BUDGET`]). A scalar that comes out is a literal from
/// then on (a size, if a positive integer); the rest are realized by their
/// chains where functions read them ([`Globals::computed`]).
fn evaluate_globals(source: &str, program: &Program, g: &mut Globals, diags: &mut Vec<Diagnostic>) {
    use globals::Names;
    use mapal_syntax as syn;
    let order = globals::order(g);
    let n = g.items.len();
    let mut resolved = vec![false; n];
    let mut fns: BTreeMap<String, (Names, &syn::FnDecl)> = BTreeMap::new();
    let mut tys: BTreeMap<String, (Names, &syn::TypeDecl)> = BTreeMap::new();
    for item in &program.items {
        match item {
            syn::Item::Fn(fd) => {
                let mut nm = Names::default();
                nm.fn_decl(source, fd);
                fns.insert(name_text(source, fd.name).to_string(), (nm, fd));
            }
            syn::Item::Type(td) => {
                let mut nm = Names::default();
                nm.type_decl(source, td);
                tys.insert(name_text(source, td.name).to_string(), (nm, td));
            }
            _ => {}
        }
    }
    // The synthetic `main`'s name: a span past the source (never reported).
    let ext = format!("{source}\nmain");
    let main_name = syn::Name {
        span: syn::SourceLoc::new(source.len() as u32 + 1, source.len() as u32 + 5),
    };
    let own = |d: &[syn::Name]| -> BTreeSet<String> {
        d.iter()
            .map(|x| name_text(source, *x).to_string())
            .collect()
    };
    let mut all_const = vec![false; n];
    // A bare integer literal needs no evaluation (and no width of its own).
    for i in 0..n {
        if let (Some(v), [name]) = (
            globals::int_literal(&g.items[i].chain),
            g.items[i].names.as_slice(),
        ) {
            resolved[i] = true;
            all_const[i] = true;
            if v >= 1 && v <= i128::from(i64::MAX) {
                g.sizes.insert(name.clone(), v as u64);
            }
            g.consts.insert(
                name.clone(),
                globals::GlobalConst::Int {
                    value: v,
                    pinned: None,
                },
            );
        }
    }
    loop {
        // Types whose sizes are known (their own size parameters aside).
        let mut ty_ok: BTreeSet<String> = tys.keys().cloned().collect();
        loop {
            let before = ty_ok.len();
            let keep: BTreeSet<String> = ty_ok
                .iter()
                .filter(|t| {
                    let (nm, td) = &tys[*t];
                    let params = own(&td.sizes);
                    nm.sizes
                        .iter()
                        .all(|s| g.sizes.contains_key(s) || params.contains(s))
                        && nm
                            .types
                            .iter()
                            .all(|r| !tys.contains_key(r) || ty_ok.contains(r))
                })
                .cloned()
                .collect();
            ty_ok = keep;
            if ty_ok.len() == before {
                break;
            }
        }
        // Functions whose sizes, types, callees and top-level reads are known.
        let mut fn_ok: BTreeSet<String> = fns.keys().filter(|f| *f != "main").cloned().collect();
        loop {
            let before = fn_ok.len();
            let keep: BTreeSet<String> = fn_ok
                .iter()
                .filter(|f| {
                    let (nm, fd) = &fns[*f];
                    let params = own(&fd.sizes);
                    nm.sizes
                        .iter()
                        .all(|s| g.sizes.contains_key(s) || params.contains(s))
                        && nm
                            .types
                            .iter()
                            .all(|t| !tys.contains_key(t) || ty_ok.contains(t))
                        && nm.values.iter().all(|v| {
                            (!fns.contains_key(v) || fn_ok.contains(v))
                                && (!g.is_top(v) || g.consts.contains_key(v))
                        })
                })
                .cloned()
                .collect();
            fn_ok = keep;
            if fn_ok.len() == before {
                break;
            }
        }
        // The bindings this round can lower: their sizes known, their reads
        // constants or other bindings of the round.
        let mut cand: BTreeSet<usize> = (0..n).filter(|&i| !resolved[i]).collect();
        let chain_names: Vec<Names> = g
            .items
            .iter()
            .map(|it| {
                let mut nm = Names::default();
                nm.chain(source, &it.chain);
                nm
            })
            .collect();
        loop {
            let before = cand.len();
            let names_in: BTreeSet<String> = cand
                .iter()
                .flat_map(|&i| g.items[i].names.iter().cloned())
                .collect();
            cand.retain(|&i| {
                let nm = &chain_names[i];
                nm.sizes.iter().all(|s| g.sizes.contains_key(s))
                    && nm
                        .types
                        .iter()
                        .all(|t| !tys.contains_key(t) || ty_ok.contains(t))
                    && nm.values.iter().all(|v| {
                        (!fns.contains_key(v) || fn_ok.contains(v))
                            && (!g.is_top(v) || g.consts.contains_key(v) || names_in.contains(v))
                    })
            });
            if cand.len() == before {
                break;
            }
        }
        if cand.is_empty() {
            break;
        }
        let ordered: Vec<usize> = order.iter().copied().filter(|i| cand.contains(i)).collect();
        let mut items: Vec<syn::Item> = Vec::new();
        for (t, (_, td)) in &tys {
            if ty_ok.contains(t) {
                items.push(syn::Item::Type((*td).clone()));
            }
        }
        for (f, (_, fd)) in &fns {
            if fn_ok.contains(f) {
                items.push(syn::Item::Fn((*fd).clone()));
            }
        }
        let body = syn::Block {
            items: ordered
                .iter()
                .map(|&i| {
                    syn::BlockItem::Stmt(syn::Stmt {
                        kind: syn::StmtKind::Chain(g.items[i].chain.clone()),
                        span: g.items[i].span,
                    })
                })
                .collect(),
            tail: None,
            span: program.span,
            compute: None,
        };
        items.push(syn::Item::Fn(syn::FnDecl {
            name: main_name,
            sizes: Vec::new(),
            params: Vec::new(),
            ret_ty: None,
            body,
            span: program.span,
        }));
        let synth = Program {
            items,
            span: program.span,
        };
        let ctx = Globals {
            consts: g.consts.clone(),
            sizes: g.sizes.clone(),
            ..Globals::default()
        };
        let bound = RefCell::new(BTreeMap::new());
        match lower_with(&ext, &synth, &ctx, Some(&bound)) {
            Ok(ir) => {
                let bound = bound.into_inner();
                let mut roots: Vec<(usize, String, mapal_ir::ObjectId)> = Vec::new();
                for &i in &ordered {
                    for name in &g.items[i].names {
                        if let Some(&o) = bound.get(name)
                            && ir
                                .object(o)
                                .is_some_and(|x| x.ty.is_numeric() || x.ty == mapal_ir::Ty::Bool)
                        {
                            roots.push((i, name.clone(), o));
                        }
                    }
                }
                let objs: Vec<mapal_ir::ObjectId> = roots.iter().map(|r| r.2).collect();
                let vals = mapal_interp::eval_values(&ir, ir.entry(), &objs, globals::EVAL_BUDGET);
                let mut got: BTreeMap<String, globals::GlobalConst> = BTreeMap::new();
                for ((i, name, _), v) in roots.iter().zip(vals) {
                    let chain = &g.items[*i].chain;
                    let annotated =
                        matches!(globals::annotation(source, chain, name), Some(Some(_)));
                    if let Some(c) = v.as_ref().and_then(|v| {
                        globals::as_const(v, annotated, globals::float_literal(chain))
                    }) {
                        got.insert(name.clone(), c);
                    }
                }
                for &i in &ordered {
                    resolved[i] = true;
                    if g.items[i].names.iter().all(|x| got.contains_key(x)) {
                        all_const[i] = true;
                        for x in &g.items[i].names {
                            let c = got[x].clone();
                            if let globals::GlobalConst::Int { value, .. } = c
                                && value >= 1
                                && value <= i128::from(i64::MAX)
                            {
                                g.sizes.insert(x.clone(), value as u64);
                            }
                            g.consts.insert(x.clone(), c);
                        }
                    }
                }
            }
            Err(ds) => {
                let inside: Vec<Diagnostic> = ds
                    .into_iter()
                    .filter(|d| {
                        ordered.iter().any(|&i| {
                            let s = g.items[i].span;
                            d.span.start >= s.start && d.span.end <= s.end
                        })
                    })
                    .collect();
                if !inside.is_empty() {
                    diags.extend(inside);
                    return;
                }
                // The errors are the functions' (lowering proper reports
                // them): these bindings are realized by their chains.
                for &i in &ordered {
                    resolved[i] = true;
                }
            }
        }
    }
    g.computed = order.into_iter().filter(|&i| !all_const[i]).collect();
}

/// Lower with the top-level values `globals` (ADR-0048). `bound`, when given,
/// records the object each name `main` binds (compile-time evaluation reads
/// the top-level bindings' values off them).
fn lower_with(
    source: &str,
    program: &Program,
    globals: &Globals,
    bound: Option<&RefCell<BTreeMap<String, mapal_ir::ObjectId>>>,
) -> Result<mapal_ir::CategoryIr, Vec<Diagnostic>> {
    let mut diags: Vec<Diagnostic> = Vec::new();

    // Defensive: any Error item is an unclean tree (L1000).
    for item in &program.items {
        if let mapal_syntax::Item::Error(sp) = item {
            diags.push(diag(LCode::UncleanTree, *sp, "error item reached lowering"));
        }
    }

    // --- structural fn checks (NoMain / MainShape / DuplicateFn / Reserved /
    //     DuplicateParam), collected.
    let mut fn_names: BTreeSet<String> = BTreeSet::new();
    let mut has_main = false;
    {
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        for item in &program.items {
            if let mapal_syntax::Item::Fn(fd) = item {
                let name = name_text(source, fd.name).to_string();
                // Reserved name: `fn print`, or a builtin scalar type name.
                if is_reserved(&name) {
                    diags.push(diag(
                        LCode::ReservedName,
                        fd.name.span,
                        format!("`{name}` is a reserved name"),
                    ));
                }
                if seen.contains_key(&name) {
                    diags.push(diag(
                        LCode::DuplicateFn,
                        fd.name.span,
                        format!("duplicate function `{name}`"),
                    ));
                } else {
                    seen.insert(name.clone(), ());
                    fn_names.insert(name.clone());
                }
                if name == "main" {
                    has_main = true;
                    if !fd.sizes.is_empty() {
                        diags.push(diag(
                            LCode::MainShape,
                            fd.name.span,
                            "`main` takes no size parameters",
                        ));
                    }
                    if !fd.params.is_empty() || fd.ret_ty.is_some() {
                        diags.push(diag(
                            LCode::MainShape,
                            fd.name.span,
                            "`main` must have no parameters and no declared return",
                        ));
                    }
                }
                // Duplicate parameter names (L1006).
                let mut pseen: BTreeSet<String> = BTreeSet::new();
                for p in &fd.params {
                    let pn = name_text(source, p.name).to_string();
                    if !pseen.insert(pn.clone()) {
                        diags.push(diag(
                            LCode::DuplicateParam,
                            p.name.span,
                            format!("duplicate parameter `{pn}`"),
                        ));
                    }
                }
            }
        }
    }
    if !has_main {
        diags.push(diag(LCode::NoMain, program.span, "no `fn main`"));
    }
    // ADR-0047 D1: the size-generic functions' declarations.
    let mut generics: Generics = BTreeMap::new();
    for item in &program.items {
        if let mapal_syntax::Item::Fn(fd) = item
            && !fd.sizes.is_empty()
        {
            sizes::check_decl(source, fd, &fn_names, &globals.sizes, globals, &mut diags);
            generics
                .entry(name_text(source, fd.name).to_string())
                .or_insert(fd);
        }
    }

    // --- Pass A: type table.
    let (types, ty_diags) = tys::build(source, program, globals);
    diags.extend(ty_diags);

    // --- Pass B: effects & call graph (L1008 aborts).
    let effects = match effects::analyze(source, program, &fn_names) {
        Ok(e) => e,
        Err(mut e) => {
            diags.append(&mut e);
            return Err(diags);
        }
    };

    // Resolve surface fn signatures (param/return tys) — feeds D1 and C.
    let fn_sigs = typing::build_fn_sigs(source, &types, &effects, program, &mut diags);
    if !diags.is_empty() && !generics.is_empty() {
        // A generic declaration's errors make its instances meaningless.
        return Err(diags);
    }

    // --- ADR-0047 D5: monomorphization rounds. Each round types every
    // fixed-size function and every instance its calls reach (the typing
    // walk infers where its types reach, else takes emission's binding of the
    // site from an earlier round), then emits. Emission resolves every call to
    // a generic function from the builder's exact types; the round is final
    // when it agrees with the typing walk at every site and every instance it
    // calls was declared. A program without generic functions takes one round
    // (the original pipeline).
    let mut seeds: BTreeSet<Instance> = BTreeSet::new();
    let mut sites: BTreeMap<SiteKey, Vec<u64>> = BTreeMap::new();
    let mut last_ok: Option<mapal_ir::CategoryIr> = None;
    for _ in 0..MAX_ROUNDS {
        let mut round_diags = diags.clone();
        let mut sigs = fn_sigs.clone();
        let (typing_info, instances, used, unresolved) = type_round(
            source,
            program,
            &types,
            &effects,
            &generics,
            &sites,
            &seeds,
            &mut sigs,
            &mut round_diags,
        );
        // If the typing/structural passes found errors, stop before emission
        // (emission is fail-fast and assumes a well-typed-enough tree; DESIGN
        // §1 LD12 — the typing pass is the multi-error surface) — unless a
        // generic call the walk could not type left its result unknown: the
        // errors downstream of it may be artifacts, so emit as a probe to
        // learn the site's binding (ADR-0047 D5) and report them only if that
        // teaches nothing.
        let probe = !round_diags.is_empty();
        if probe && unresolved == 0 {
            return Err(round_diags);
        }

        // --- Pass C + D2/D3 + E: declare, emit, seal (fail-fast).
        let log = RefCell::new(emit::CallLog::default());
        let result = emit::lower_program(
            source,
            program,
            &types,
            &effects,
            &sigs,
            &typing_info,
            &emit::Mono {
                generics: &generics,
                instances: &instances,
                log: &log,
                probe,
                bound,
            },
        );
        if generics.is_empty() {
            return result;
        }
        let log = log.into_inner();
        let declared: BTreeSet<Instance> = instances
            .iter()
            .flat_map(|(g, vs)| vs.iter().map(move |(v, _)| (g.clone(), v.clone())))
            .collect();
        let disagree = log.sites.iter().any(|(k, v)| used.get(k) != Some(v));
        let missing = !log.requested.is_empty();
        if disagree || missing {
            sites.extend(log.sites.clone());
            seeds.extend(log.requested.iter().cloned());
            seeds.extend(log.called.iter().cloned());
            continue;
        }
        if probe {
            return Err(round_diags);
        }
        match result {
            Err(d) => return Err(d),
            Ok(ir) if log.called == declared => return Ok(ir),
            Ok(ir) => {
                // An instance no call reaches (the typing walk's inference
                // disagreed with emission's somewhere upstream): drop it.
                last_ok = Some(ir);
                sites.extend(log.sites.clone());
                seeds = log.called.clone();
            }
        }
    }
    last_ok.ok_or_else(|| {
        vec![diag(
            LCode::Internal,
            program.span,
            "size inference did not settle (ADR-0047 D5)",
        )]
    })
}

/// One typing round (D1 over every fixed-size function, then every instance
/// the typed calls reach — a worklist). Returns the typing info by function
/// (or instance) name, the instances by generic function (sorted size
/// tuples, each with its name), and the binding the walk used at each site.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn type_round(
    source: &str,
    program: &Program,
    types: &tys::TypeTable,
    effects: &effects::Effects,
    generics: &Generics,
    sites: &BTreeMap<SiteKey, Vec<u64>>,
    seeds: &BTreeSet<Instance>,
    sigs: &mut BTreeMap<String, typing::FnSig>,
    diags: &mut Vec<Diagnostic>,
) -> (
    BTreeMap<String, typing::TypeInfo>,
    BTreeMap<String, Vec<(Vec<u64>, String)>>,
    BTreeMap<SiteKey, Vec<u64>>,
    usize,
) {
    enum Unit<'a> {
        Fixed(&'a mapal_syntax::FnDecl),
        Inst(Instance),
    }
    let mut queue: VecDeque<Unit> = VecDeque::new();
    for item in &program.items {
        if let mapal_syntax::Item::Fn(fd) = item
            && fd.sizes.is_empty()
        {
            queue.push_back(Unit::Fixed(fd));
        }
    }
    let mut known: BTreeSet<Instance> = BTreeSet::new();
    for s in seeds {
        if known.insert(s.clone()) {
            queue.push_back(Unit::Inst(s.clone()));
        }
    }
    let mut typing_info: BTreeMap<String, typing::TypeInfo> = BTreeMap::new();
    let mut used: BTreeMap<SiteKey, Vec<u64>> = BTreeMap::new();
    let mut unresolved = 0;
    let mut instance_diags: Vec<Diagnostic> = Vec::new();
    while let Some(unit) = queue.pop_front() {
        let (fd, name, env, inst) = match &unit {
            Unit::Fixed(fd) => (
                *fd,
                name_text(source, fd.name).to_string(),
                types.base_env().clone(),
                false,
            ),
            Unit::Inst((g, vs)) => {
                let Some(fd) = generics.get(g).copied() else {
                    continue;
                };
                let env = sizes::env_of(source, fd, vs, types.base_env());
                let Some(sig) =
                    sizes::instance_sig(source, types, fd, &env, effects.is_effectful(g))
                else {
                    continue;
                };
                let name = sizes::instance_name(g, vs);
                sigs.insert(name.clone(), sig);
                (fd, name, env, true)
            }
        };
        let ctx = typing::SizeCtx {
            owner: name.clone(),
            env,
            generics,
            sites,
        };
        let (info, mut d, found) = typing::analyze_fn(source, types, effects, sigs, fd, &ctx);
        // L1306: return completeness (D1; syntactic presence is sound for
        // Core — L1405 keeps `-> ret` out of Phi arms).
        let sig = sigs.get(&name);
        let has_output = sig.map(|s| s.ret.is_some()).unwrap_or(false);
        let out_arity = sig
            .and_then(|s| s.ret.as_ref())
            .and_then(|t| t.product_arity());
        typing::check_return_completeness(source, fd, has_output, out_arity, &mut d);
        if inst {
            for mut x in d {
                // One report per span and code, naming the first instance.
                if !instance_diags
                    .iter()
                    .any(|y| y.code == x.code && y.span == x.span)
                {
                    x.message = format!("{} (in `{name}`)", x.message);
                    instance_diags.push(x);
                }
            }
        } else {
            diags.extend(d);
        }
        typing_info.insert(name, info);
        used.extend(found.sites);
        unresolved += found.unresolved;
        for i in found.instances {
            if known.insert(i.clone()) {
                queue.push_back(Unit::Inst(i));
            }
        }
    }
    diags.extend(instance_diags);
    let mut instances: BTreeMap<String, Vec<(Vec<u64>, String)>> = BTreeMap::new();
    for (g, vs) in known {
        let name = sizes::instance_name(&g, &vs);
        if typing_info.contains_key(&name) {
            instances.entry(g).or_default().push((vs, name));
        }
    }
    (typing_info, instances, used, unresolved)
}

/// The builtin print family (ADR-0015): `print` (raw) and `println` (newline).
/// One predicate so the many effect / typing / emit sites that special-case the
/// builtin cannot drift — adding `println` otherwise silently breaks every missed
/// call site (the FRAMEWORK §5 "one source of truth for a shared check" rule; this
/// helper was introduced precisely because `println` regressed an un-updated site).
pub(crate) fn is_print_builtin(name: &str) -> bool {
    matches!(name, "print" | "println")
}

/// Pure bare-stage builtins (ADR-0018/ADR-0029). Resolved by name like
/// `print`/`println`, but token-free and legal in parallel fanout.
pub(crate) fn is_pure_builtin(name: &str) -> bool {
    matches!(name, "zip" | "enumerate" | "iota" | "fill")
        || mapal_ir::MathFn::from_name(name).is_some()
        || widen_target(name).is_some()
        || narrow_target(name).is_some()
}

/// ADR-0043: the narrow format a rounding builtin (`to_bf16`, `to_f16`)
/// produces.
pub(crate) fn narrow_target(name: &str) -> Option<mapal_ir::NarrowFloat> {
    match name {
        "to_bf16" => Some(mapal_ir::NarrowFloat::Bf16),
        "to_f16" => Some(mapal_ir::NarrowFloat::F16),
        "to_f8e4m3" => Some(mapal_ir::NarrowFloat::F8e4m3),
        "to_f8e5m2" => Some(mapal_ir::NarrowFloat::F8e5m2),
        _ => None,
    }
}

/// The `time` stage builtin (plan-time-builtin): effectful like `print` (it
/// threads the IO token, so it is never reordered, folded, or dropped), but
/// value-producing and wire-LESS — `() -> time` is its only shape. Same
/// one-predicate discipline as [`is_print_builtin`].
pub(crate) fn is_time_builtin(name: &str) -> bool {
    name == "time"
}

/// The `read` stage builtin (ADR-0045): effectful like `time` (it threads the
/// IO token), value-producing, fed by a string literal path, typed by the
/// binding's annotation — `"w.bin" -> read -> w: [f32; n]` is its only shape.
pub(crate) fn is_read_builtin(name: &str) -> bool {
    name == "read"
}

/// Target type selected by an ADR-0029 widening builtin name.
pub(crate) fn widen_target(name: &str) -> Option<mapal_ir::Ty> {
    match name {
        "widen_i64" => Some(mapal_ir::Ty::i64()),
        "widen_f32" => Some(mapal_ir::Ty::f32()),
        "widen_f64" => Some(mapal_ir::Ty::f64()),
        _ => None,
    }
}

/// Whether a name is reserved (L1009): stage builtins or a scalar type name.
pub(crate) fn is_reserved(name: &str) -> bool {
    is_print_builtin(name)
        || is_pure_builtin(name)
        || is_time_builtin(name)
        || is_read_builtin(name)
        || matches!(
            name,
            "i32" | "i64" | "u8" | "f32" | "f64" | "bf16" | "f16" | "f8e4m3" | "f8e5m2" | "bool"
        )
}
