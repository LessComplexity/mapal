//! Golden oracle for DESIGN §5.1 (the Operation typing table).
//!
//! The rows below are transcribed **by hand** from DESIGN.md §5.1 and
//! asserted against `edge_type_ok` op-by-op. This is a **test-only** oracle:
//! it is deliberately NOT shared into the builder or `validate` production
//! paths, because the independence of those two realizations is itself the
//! load-bearing property (DESIGN §11 / FRAMEWORK §7.2) — a shared table
//! would collapse `seal Ok ⇒ validate empty` into a tautology.
//!
//! Scope: `edge_type_ok` certifies only the **typing** judgment (source and
//! target ty). §5.1 "extra conditions" that are graph-shape/provenance
//! clauses live in other passes and are intentionally NOT retested here —
//! `Call`/`Map`/`Fold` FuncKind (I6, `check_references`/builder), `Output`
//! target-is-Return (I8, `check_edges`), `LoopExit` target-outside-SCC (I5,
//! `check_loops`), and Map/Fold body token-freedom (`check_tokens`).
use super::*;
use crate::graph::{FuncDef, FuncKind, Morphism, Object};
use crate::loc::SourceLoc;
use slotmap::SlotMap;

mod fixture;
mod rows_core;
mod rows_io;

use self::fixture::*;
use self::rows_core::{Case, core_rows};
use self::rows_io::io_rows;

#[test]
fn edge_type_ok_matches_design_5_1() {
    let fx = fixture();

    // (label, op, source ty, target ty, target obj, expected §5.1 judgment),
    // in the table's order: `rows_core` then `rows_io`.
    let mut cases: Vec<Case> = core_rows(&fx);
    cases.extend(io_rows(&fx));

    let mut failures = Vec::new();
    for (label, op, sty, tty, target, expected) in cases {
        let got = check(&fx, op, sty, tty, target);
        if got != expected {
            failures.push(format!(
                "  {label}: §5.1 says {expected}, edge_type_ok says {got}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "DESIGN §5.1 typing table disagrees with edge_type_ok:\n{}",
        failures.join("\n")
    );
}
