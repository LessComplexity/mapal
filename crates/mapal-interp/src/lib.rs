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

use mapal_ir::{CategoryIr, FuncId, Ty};

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

    let outcome = eval_call(ir, entry, arg, budget);
    let output = match (&outcome, is_token) {
        (Outcome::Done(RValue::Token(log)), true) => log.clone(),
        _ => String::new(),
    };
    RunResult { outcome, output }
}

/// Call one function directly with an argument (interp DESIGN §8). The public
/// lift of the internal `Result<RValue, Abort>` at the entry boundary.
pub fn eval_call(ir: &CategoryIr, f: FuncId, arg: RValue, budget: u64) -> Outcome {
    let mut budget = budget;
    match eval::eval_fn(ir, f, arg, &mut budget) {
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
