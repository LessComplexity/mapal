//! `mapal-interp`: the Mapal compiler's **fueled reference interpreter** — THE
//! ORACLE (interp DESIGN; ADR-0002 E1; ADR-0013 traps; ADR-0015 print/println;
//! ADR-0016 guard-first loops).
//!
//! It consumes a sealed, `validate()`-clean `mapal_ir::CategoryIr` (borrowed, never
//! mutated) and totally evaluates it into an [`Outcome`]: every run halts with
//! `Done`, `Diverged` (budget exhausted, E1), or `Trapped` (ADR-0013) — never a
//! panic, never a hang.
//!
//! Public API (interp DESIGN §8):
//! - [`run`] — seed `main`, walk the graph, collect program output.
//! - [`eval_call`] — call one function directly (used by value-contract tests).
//! - [`RValue`] / [`Outcome`] / [`TrapKind`] — the value domain (§1).
//! - [`RunResult`] — `{ outcome, output }`.
//!
//! No `Display` impls (C3); a plain [`render`] owns value→string. Determinism
//! (E2) is structural: `SecondaryMap` + `Vec` only, no `HashMap` anywhere.

mod eval;
mod loops;
mod value;

use mapal_ir::{CategoryIr, FuncId, MorphismId, ObjectId, Ty};
use slotmap::SecondaryMap;

pub use value::{Outcome, RValue, TrapKind, render};

/// The result of running a program (interp DESIGN §7): the [`Outcome`] plus the
/// observable stdout (`output = ""` unless `Done` with a token).
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub outcome: Outcome,
    pub output: String,
}

/// Run a program: seed the entry function, walk the graph, collect output
/// (interp DESIGN §7). `budget` is the global step counter (E1).
pub fn run(ir: &CategoryIr, budget: u64) -> RunResult {
    run_ordered(ir, budget, &[])
}

/// [`run`] with the compiled program's split folds replayed (ADR-0040 D3):
/// each `(func, fold, S)` folds in `S` contiguous partitions — partition 0
/// from its seed, the others from 0 — added in partition order. The `fixed`
/// face's oracle; `orders` comes from the emitter (`Emission::orders`), keyed
/// on the IR it compiled.
pub fn run_ordered(
    ir: &CategoryIr,
    budget: u64,
    orders: &[(FuncId, MorphismId, u64)],
) -> RunResult {
    let entry = ir.entry();
    let fd = ir.func(entry).expect("sealed graph: entry resolves");
    let input_ty = &ir
        .object(fd.input)
        .expect("sealed graph: input resolves")
        .ty;

    // Seed per the entry protocol (§7): IoToken ⇒ Token(""), else Unit.
    let (arg, is_token) = match input_ty {
        Ty::IoToken => (RValue::Token(String::new()), true),
        // The six examples are all effectful (IoToken → IoToken); a pure
        // `main : Unit → Unit` (or any other entry shape) seeds Unit, output "".
        _ => (RValue::Unit, false),
    };

    let mut fuel = budget;
    let outcome = match eval::eval_fn(ir, entry, arg, &mut fuel, orders) {
        Ok(v) => Outcome::Done(v),
        Err(abort) => abort.into_outcome(),
    };
    let output = match (&outcome, is_token) {
        (Outcome::Done(RValue::Token(log)), true) => log.clone(),
        _ => String::new(),
    };
    RunResult { outcome, output }
}

/// [`run_ordered`] with every product evaluated by its weakest admitted
/// member (ADR-0041, `mapal test --weakest`): a product whose set admits
/// `tf32` rounds its f32 operands to TF32 (round-to-nearest-even), then
/// multiplies in f32. The other side of D3's bound from [`run`]'s f32.
pub fn run_weakest(
    ir: &CategoryIr,
    budget: u64,
    orders: &[(FuncId, MorphismId, u64)],
) -> RunResult {
    eval::WEAKEST.with(|w| w.set(true));
    let r = run_ordered(ir, budget, orders);
    eval::WEAKEST.with(|w| w.set(false));
    r
}

/// ADR-0048 (compile-time evaluation): the values of `roots` — objects of
/// `f` whose dataflow cone holds no input and no effect — computed from their
/// cones alone, the rest of `f` never run. All roots are evaluated together
/// under `budget`; when that aborts (a trap, the budget) each root is retried
/// alone under the same budget so one root's trap does not hide another's
/// value. `None` for a root whose cone reaches the input or an IO token (not
/// computable before the program runs), or whose evaluation aborts.
pub fn eval_values(
    ir: &CategoryIr,
    f: FuncId,
    roots: &[ObjectId],
    budget: u64,
) -> Vec<Option<RValue>> {
    let cones: Vec<Option<SecondaryMap<ObjectId, ()>>> =
        roots.iter().map(|&r| static_cone(ir, f, r)).collect();
    let mut all: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    for c in cones.iter().flatten() {
        for (o, ()) in c {
            all.insert(o, ());
        }
    }
    let mut fuel = budget;
    if let Ok(env) = eval::eval_cone(ir, f, &all, &mut fuel) {
        return roots
            .iter()
            .zip(&cones)
            .map(|(&r, c)| c.as_ref().and_then(|_| env.get(r).cloned()))
            .collect();
    }
    roots
        .iter()
        .zip(cones)
        .map(|(&r, c)| {
            let c = c?;
            let mut fuel = budget;
            eval::eval_cone(ir, f, &c, &mut fuel)
                .ok()
                .and_then(|env| env.get(r).cloned())
        })
        .collect()
}

/// The backward cone of `root` in `f`, closed under sources (a loop merge
/// brings its whole loop: decide / advance / exit work), or `None` when it
/// reaches the function's input or an IO-token value.
fn static_cone(ir: &CategoryIr, f: FuncId, root: ObjectId) -> Option<SecondaryMap<ObjectId, ()>> {
    let input = ir.func(f)?.input;
    let mut cone: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    let mut stack = vec![root];
    while let Some(o) = stack.pop() {
        if cone.contains_key(o) {
            continue;
        }
        let obj = ir.object(o)?;
        if o == input || mapal_ir::ty_contains_token(&obj.ty) {
            return None;
        }
        cone.insert(o, ());
        for &m in ir.in_edges(o) {
            stack.push(ir.morphism(m)?.source);
        }
        if obj.kind == mapal_ir::ObjectKind::LoopMerge
            && let Some(plan) = ir.loop_plan(f, o)
        {
            for &m in plan
                .decide_order
                .iter()
                .chain(&plan.advance_order)
                .chain(&plan.exits)
            {
                let mm = ir.morphism(m)?;
                stack.push(mm.target);
                stack.push(mm.source);
            }
        }
    }
    Some(cone)
}

/// Call one function directly with an argument (interp DESIGN §8). The public
/// lift of the internal `Result<RValue, Abort>` at the entry boundary.
pub fn eval_call(ir: &CategoryIr, f: FuncId, arg: RValue, budget: u64) -> Outcome {
    let mut budget = budget;
    match eval::eval_fn(ir, f, arg, &mut budget, &[]) {
        Ok(v) => Outcome::Done(v),
        Err(abort) => abort.into_outcome(),
    }
}

/// The observable a compiled program must reproduce (L1 / BL8 `@main`
/// protocol) — a public contract: the backend differential suite and
/// `mapal test` both judge the native arm against it. `None` for a Diverged
/// run (no verdict). `Some((None, 101))` for a trap (stdout ignored).
/// `Some((Some(stdout), 0))` for a Done program — the entry protocol decides
/// the stdout: an `IoToken` entry's is the token log; a closed `Unit → scalar`
/// entry's is the wrapper-printed return.
pub fn native_observable(ir: &CategoryIr, rr: &RunResult) -> Option<(Option<String>, i32)> {
    match &rr.outcome {
        Outcome::Diverged => None,
        Outcome::Trapped(_) => Some((None, 101)),
        Outcome::Done(v) => {
            let entry = ir.entry();
            let in_ty = &ir.object(ir.func(entry).unwrap().input).unwrap().ty;
            let stdout = if *in_ty == Ty::IoToken {
                rr.output.clone()
            } else if let RValue::Scalar(sv) = v {
                // The @main wrapper prints the scalar return with a newline.
                format!("{}\n", render(sv))
            } else {
                String::new() // non-printable return: wrapper prints nothing
            };
            Some((Some(stdout), 0))
        }
    }
}
