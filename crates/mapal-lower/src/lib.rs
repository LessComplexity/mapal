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
mod scope;
mod sizes;
mod typing;
mod tys;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use mapal_syntax::{Diagnostic, Program};

use crate::diag::{LCode, diag};
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
pub fn lower(source: &str, program: &Program) -> Result<mapal_ir::CategoryIr, Vec<Diagnostic>> {
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
            sizes::check_decl(source, fd, &fn_names, &mut diags);
            generics
                .entry(name_text(source, fd.name).to_string())
                .or_insert(fd);
        }
    }

    // --- Pass A: type table.
    let (types, ty_diags) = tys::build(source, program);
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
                sizes::SizeEnv::new(),
                false,
            ),
            Unit::Inst((g, vs)) => {
                let Some(fd) = generics.get(g).copied() else {
                    continue;
                };
                let env = sizes::env_of(source, fd, vs);
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
    matches!(name, "zip" | "enumerate" | "iota" | "fill" | "exp")
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
        || matches!(
            name,
            "i32" | "i64" | "u8" | "f32" | "f64" | "bf16" | "f16" | "f8e4m3" | "f8e5m2" | "bool"
        )
}
