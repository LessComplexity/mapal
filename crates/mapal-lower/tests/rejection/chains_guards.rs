//! Rejection tests: chains, returns and guards (L1301-L1409).

use super::*;

// --- L1301–L1307: chains / returns ------------------------------------------

#[test]
fn l1301_headless_chain() {
    // a statement-level headless op chain that is not `-> ret;`.
    assert_rejects("fn main() { -> print; }\n", "L1301");
}

#[test]
fn l1302_expr_stage() {
    // a general expression stage that does not consume the wire.
    assert_rejects(
        "fn f(x: i32, y: i32) -> i32 { x -> y + 1 -> ret; }\nfn main() {}\n",
        "L1302",
    );
}

#[test]
fn l1301_unit_as_value() {
    // plan-time-builtin: `()` produces no object — its only use is the
    // wire-LESS head of `() -> time`. In a value position (an operand, a
    // block tail) it is L1301, and the message teaches the one legal use.
    assert_rejects("fn main() { () + 1 -> println; }\n", "L1301");
    let src = "fn f() -> i32 { () }\nfn main() {}\n";
    assert_rejects(src, "L1301");
    let po = mapal_syntax::parse(src);
    let ds = mapal_lower::lower(src, &po.program).expect_err("`()` tail is not a value");
    let d = ds
        .iter()
        .find(|d| d.code.0 == "L1301")
        .unwrap_or_else(|| panic!("expected L1301, got {ds:?}"));
    assert!(
        d.message.contains("() -> time"),
        "the Unit arm's message names the one legal use: {d:?}"
    );
}

#[test]
fn l1302_time_with_a_wire() {
    // plan-time-builtin: `time` is the one stage that takes no wire; feeding it
    // one is L1302 ("`time` takes no value: write `() -> time`").
    assert_rejects("fn main() { 5 -> time -> t; t -> println; }\n", "L1302");
}

#[test]
fn l1303_ret_mid_chain() {
    assert_rejects(
        "fn f(x: i32) -> i32 { x -> ret -> ret; }\nfn main() {}\n",
        "L1303",
    );
}

#[test]
fn l1304_jump_misplaced() {
    // `-> loop` outside a loop (Unit-output fn, so L1306 does not fire first).
    assert_rejects("fn f(x: i32) { x -> loop; }\nfn main() {}\n", "L1304");
}

#[test]
fn l1305_fanout_no_value() {
    // a fanout whose branch produces no value, then the chain continues.
    let src = r#"fn g(x: i32) { x -> print; }
fn sq(x: i32) -> i32 { x * x -> ret; }
fn main() {
    6 -> {
        -> sq -> a;
        -> g;
    } -> r;
    r -> print;
}
"#;
    assert_rejects(src, "L1305");
}

#[test]
fn l1306_incomplete_return() {
    // non-Unit output with no return write.
    assert_rejects("fn f(x: i32) -> i32 { x -> y; }\nfn main() {}\n", "L1306");
}

#[test]
fn l1306_mixed_bare_and_slot() {
    assert_rejects(
        "fn f() -> (i32, i32) { 1 -> ret; 2 -> ret.1; }\nfn main() {}\n",
        "L1306",
    );
}

#[test]
fn l1306_duplicate_slot() {
    assert_rejects(
        "fn f() -> (i32, i32) { 1 -> ret.0; 2 -> ret.0; }\nfn main() {}\n",
        "L1306",
    );
}

#[test]
fn l1307_effectful_return_shape() {
    // a `ret.k` write in an effectful fn.
    assert_rejects(
        "fn f(x: i32) -> (i32, i32) { x -> print; x -> ret.0; x -> ret.1; }\nfn main() {}\n",
        "L1307",
    );
}

#[test]
fn an_effectful_fns_slot_exit_runs_after_its_loop() {
    // A loop exit's `ret.k` writes in an effectful fn: the arm runs after the
    // loop (its print once), the slots packed into the one whole return — no
    // print feeds the loop's exit route (decided every trip).
    let src = "fn f(n: i32) -> (i32, i32) {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> { \"done\" -> print; i -> ret.0; i * 2 -> ret.1; }
        }
    }
}
fn main() { 3 -> f -> r; r.1 -> print; }
";
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
    for (f, _) in ir.funcs() {
        for scc in ir.loop_structure(f) {
            for &m in &scc.merges {
                let lp = ir.loop_plan(f, m).expect("canonical loop");
                assert!(
                    lp.decide_order.iter().all(|&d| !matches!(
                        ir.morphism(d).unwrap().op,
                        mapal_ir::Operation::Print { .. }
                    )),
                    "a print in a loop's decide phase"
                );
            }
        }
    }
}

#[test]
fn l1404_an_effect_in_an_exit_arm_that_cannot_follow_its_loop() {
    // A headless exit binding reads the guard's condition, so its arm cannot
    // run after the loop; its print would run every trip — rejected.
    assert_rejects(
        "fn main() {
    mut i: i32 <- 0;
    loop {
        (i < 3) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> { \"done\" -> print; -> b; }
        }
    }
    b -> print;
}
",
        "L1404",
    );
}

// --- L1401–L1409: guards -----------------------------------------------------

#[test]
fn l1401_guard_arm_missing() {
    // a bool guard with only one pole, no default.
    assert_rejects(
        "fn f(b: bool, x: i32) -> i32 { b -> { -true-> x; } -> ret; }\nfn main() {}\n",
        "L1401",
    );
}

#[test]
fn l1402_guard_arm_duplicate() {
    assert_rejects(
        "fn f(k: i32) -> i32 { k -> { -0-> 1; -0-> 2; -_-> 3; } -> ret; }\nfn main() {}\n",
        "L1402",
    );
}

#[test]
fn l1403_guard_arm_mixed() {
    assert_rejects(
        "fn f(b: bool) -> i32 { b -> { -true-> 1; -0-> 2; } -> ret; }\nfn main() {}\n",
        "L1403",
    );
}

#[test]
fn l1404_guard_arm_effectful() {
    // print inside a Phi-position arm.
    assert_rejects(
        "fn f(b: bool, x: i32) -> i32 { b -> { -true-> { x -> print; x } -false-> x; } -> ret; }\nfn main() {}\n",
        "L1404",
    );
}

#[test]
fn l1405_guard_ret_in_phi_arm() {
    assert_rejects(
        "fn f(b: bool, x: i32) -> i32 { b -> { -true-> { x -> ret; x } -false-> x; } -> ret; }\nfn main() {}\n",
        "L1405",
    );
}

#[test]
fn l1406_guard_scrutinee_type() {
    // bool poles on a non-bool scrutinee.
    assert_rejects(
        "fn f(x: i32) -> i32 { x -> { -true-> 1; -false-> 2; } -> ret; }\nfn main() {}\n",
        "L1406",
    );
}

#[test]
fn l1000_mixed_guard_block_is_loud_not_filtered() {
    // B2 end-to-end (2026-08-01 review): a guard block with a tail used to
    // reach lower as a clean two-arm Guard with `999` silently deleted.
    // The parser now retains the complete block as a StmtBlock (P0006+P0115
    // at parse), and lower rejects a StmtBlock stage with L1000 — loud, no
    // source loss. Not parse-clean by construction, so not via assert_rejects.
    let src =
        "fn f(b: bool) -> i32 { b -> { -true-> 1; -false-> 2; 999 } -> ret; }\nfn main() {}\n";
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.iter().any(|d| d.code.0 == "P0006"),
        "parse names the tail mixing: {:?}",
        po.diagnostics
    );
    let Err(ds) = mapal_lower::lower(src, &po.program) else {
        panic!("a mixed guard block must not lower");
    };
    assert!(
        ds.iter().any(|d| d.code.0 == "L1000"),
        "lower rejects the retained block loudly: {:?}",
        ds.iter().map(|d| &d.code.0).collect::<Vec<_>>()
    );
}

#[test]
fn l1407_routing_guard_shape() {
    // a routing guard that is not the loop body's final item.
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> i -> ret;
        }
        i -> print;
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1407");
}

#[test]
fn l1408_assign_in_phi_arm() {
    // assignment to an enclosing mut inside a Phi-position arm.
    let src = r#"fn f(b: bool) -> i32 {
    mut x: i32 <- 0;
    b -> { -true-> { 5 -> x; x } -false-> x; } -> ret;
}
fn main() {}
"#;
    assert_rejects(src, "L1408");
}

#[test]
fn l1409_routing_guard_arms() {
    // a three-arm routing guard (integer discriminants in a routing guard).
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        i -> {
            -0-> { i + 1 -> i; -> loop; }
            -1-> { i + 2 -> i; -> loop; }
            -_-> i -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1409");
}
