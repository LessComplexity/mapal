//! Rejection tests: `seq` blocks and the named review-finding regressions.

use super::*;

// --- L1611: seq statement block (ADR-0019 / WP2) ----------------------------

#[test]
fn l1611_seq_continues_no_tail() {
    // A seq whose chain continues (a following `-> g` stage) but that has no
    // tail value → L1611 (no more silent pack-of-tails; ADR-0019 pin c).
    assert_rejects(
        "fn g(x: i32) -> i32 { x -> ret; }\nfn main() { 5 -> seq { 3 -> a; } -> g -> r; r -> println; }\n",
        "L1611",
    );
}

#[test]
fn l1611_seq_return_position_no_tail() {
    // A seq in return position with no tail value → L1611 (the return demands a
    // value the tail-less seq cannot supply).
    assert_rejects(
        "fn f(x: i32) -> i32 { x -> seq { x -> a; } }\nfn main() {}\n",
        "L1611",
    );
}

#[test]
fn l1611_effectful_seq_return_position_no_tail() {
    // WP2 fixer regression (finding F5/F6): an EFFECTFUL fn whose body-tail chain
    // ends in a tail-less `seq` in return position must draw L1611 — not fall
    // through to the pre-existing L1306. The effectful-B-present tail lowers under
    // `ChainCtx::RetValue` so `emit_seq_block` sees return position (ADR-0019
    // pin c; DESIGN §8.10). The pure analogue is `l1611_seq_return_position_no_tail`.
    assert_rejects(
        "fn f(x: i32) -> i32 { \"hi\" -> println; x -> seq { 3 -> a; } }\nfn main() { 5 -> f -> r; r -> println; }\n",
        "L1611",
    );
    // A tail-less seq that is itself the whole (effectful) body-tail: same code.
    assert_rejects(
        "fn f(x: i32) -> i32 { x -> seq { \"hi\" -> println; } }\nfn main() { 5 -> f -> r; r -> println; }\n",
        "L1611",
    );
}

#[test]
fn seq_return_position_valued_effectful_lowers_clean() {
    // WP2 fixer regression: the RetValue fix must NOT break a VALUED seq in an
    // effectful fn's return position — the tail value is still handed back and
    // packed with the token (prints "hi" then returns x*2).
    let _ = lower_ok(
        "fn f(x: i32) -> i32 { \"hi\" -> println; x -> seq { x * 2 -> a; a } }\nfn main() { 5 -> f -> r; r -> println; }\n",
    );
}

#[test]
fn l1404_effectful_seq_in_phi_arm() {
    // WP2 fixer regression (finding F1/F2): an effectful stage inside a `seq`
    // inside a Phi-position guard arm must draw L1404 — the phi-arm scan descends
    // into the seq body (ADR-0019 §8.10). Without the descent the effect lowered
    // UNCONDITIONALLY (hoisted out of the Phi): a validate-clean miscompile.
    assert_rejects(
        "fn f(b: bool, x: i32) -> i32 { b -> { -true-> { x -> seq { x -> println; x -> a; a } } -false-> x; } -> ret; }\nfn main() {}\n",
        "L1404",
    );
}

#[test]
fn l1404_effectful_fanout_in_phi_arm() {
    // Sibling of the above (finding F1's Fanout analog): the phi-arm scan descends
    // into fanout branches too — an effect in a fanout branch inside a Phi arm is
    // the same unconditional-effect hazard, L1404.
    assert_rejects(
        "fn f(b: bool, x: i32) -> i32 { b -> { -true-> { x -> { -> println; -> a; } a } -false-> x; } -> ret; }\nfn main() {}\n",
        "L1404",
    );
}

// (The pre-ADR-0027 `l1108_capture_in_seq_in_map_body` row moved to
// tests/captures.rs as the positive pin `capture_in_seq_in_map_body_lowers`:
// the seq-descent read of enclosing `k` is a legal capture now — D1.)

#[test]
fn l1108_indexed_bind_captures_enclosing_local_in_map_body() {
    // Sibling of the (now positive) seq-in-body case — see
    // `capture_in_seq_in_map_body_lowers` in tests/captures.rs — for the
    // ADR-0021 sugar: an
    // indexed bind `c[i] <- v` whose target `c` is an enclosing local (not a
    // body-local) captures it — must draw L1108, not the misleading L1101
    // "unresolved name". The map/fold-body capture check (typing.rs
    // `capture_stmt`) must NOT treat the indexed-bind target as a fresh local.
    assert_rejects(
        "fn main() { mut c: [i32; 3] <- [0,0,0]; [1,2,3] -> map { e -> e -> seq { c[0] <- e; e } } -> r; r[0] -> println; }\n",
        "L1108",
    );
    // The index expression is capture-checked too — but a READ of enclosing
    // `k` in the index is a legal capture now (ADR-0027 D1): the L1108 here
    // comes from the WRITE `c[k] <- e` targeting enclosing `c` (an indexed
    // bind is a rebind, never a fresh shadow — ADR-0021).
    assert_rejects(
        "fn main() { 2 -> k; mut c: [i32; 3] <- [0,0,0]; [1,2,3] -> map { e -> e -> seq { c[k] <- e; e } } -> r; r[0] -> println; }\n",
        "L1108",
    );
}

#[test]
fn indexed_update_of_body_local_in_map_body_lowers_clean() {
    // Inverse control for the fix above: an indexed bind whose target is a
    // *body-local* (declared inside the map body) is not a capture — it lowers
    // clean (ADR-0021 §1 "legal in map/fold bodies").
    let _ = lower_ok(
        "fn main() { [1,2,3] -> map { e -> e -> seq { mut d: [i32; 2] <- [0,0]; d[0] <- e; d[0] } } -> r; r[0] -> println; }\n",
    );
}

#[test]
fn effectful_seq_in_fanout_join_rejected() {
    // ADR-0019 pin e / plan WP2 item 5: an effectful `seq` branch inside a
    // *Plain* (parallel) fanout that joins is rejected by lower's existing
    // L1305 — exactly as a bare effectful branch is (the effectful seq produces
    // no value for the join). Parity with `print_branch_join` below: the seq
    // opens no effect escape hatch.
    assert_rejects(
        "fn main() { 5 -> { -> seq { -> println }; -> println; } -> x; x -> println; }\n",
        "L1305",
    );
    // The bare-print branch it must match, unchanged (plan matrix).
    assert_rejects(
        "fn main() { 5 -> { -> println; -> println; } -> x; x -> println; }\n",
        "L1305",
    );
}

#[test]
fn empty_seq_lowers_clean() {
    // `seq { }` in statement position: no value, no continuation — lowers clean.
    let _ = lower_ok("fn main() { 5 -> seq { } }\n");
}

#[test]
fn seq_bindings_escape_to_enclosing_scope() {
    // ADR-0019 pin b: a binding made inside `seq` lives in the enclosing scope,
    // so `a` is visible after the seq (else `a -> println` would be L1101). The
    // `fanout.mapal` idiom, now for seq.
    let _ = lower_ok("fn main() { 5 -> seq { 7 -> a; } a -> println; }\n");
}

#[test]
fn seq_headless_statements_seed_from_input() {
    // ADR-0019 pin a / compat pin 3: the old bare-chain branch form — headless
    // chain *statements* — seeds each from the seq input and lowers clean (each
    // `-> println` prints the seq input `42`, ordered by the token thread).
    let _ = lower_ok("fn main() { 42 -> seq { -> println; -> println; } }\n");
}

// --- named regression: the `0.0`-seed fold must lower CLEAN ------------------

#[test]
fn fold_zero_seed_lowers_clean() {
    // The §7.2 unification regression: the f32 seed `0.0` must resolve to f32
    // through `acc + px.r`, so the program lowers without error.
    let src = r#"type Pixel { r: f32, g: f32, b: f32 }
fn main() {
    [Pixel { r: 1.0, g: 2.0, b: 3.0 }] -> img: [Pixel; 1];
    (0.0, img) -> fold { acc, px -> acc + px.r } -> total;
    total -> print;
}
"#;
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).expect("0.0-seed fold must lower clean");
    assert!(mapal_ir::validate(&ir).is_empty());
}

// --- named regression: single-branch fanout passes through unpacked ---------

#[test]
fn single_branch_fanout_passes_through() {
    // A single-branch fanout's join is the branch's tail value (no pack; FAN-1).
    let src = r#"fn sq(x: i32) -> i32 { x * x -> ret; }
fn main() {
    6 -> { -> sq -> a; } -> r;
    r -> print;
}
"#;
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).expect("single-branch fanout lowers");
    assert!(mapal_ir::validate(&ir).is_empty());
}

// ===========================================================================
//  Review-finding regressions (named after the finding id)
// ===========================================================================

#[test]
fn lower_1_retk_above_u32_max_rejected() {
    // LOWER-1 / LOWER-RETK-TRUNC: a `ret.k` slot > u32::MAX must be rejected with
    // L1205 (the Session-04 truncation lesson), not silently truncated to slot 0.
    assert_rejects(
        "fn f() -> (i32, i32) { 1 -> ret.1; 2 -> ret.4294967296; }\nfn main() {}\n",
        "L1205",
    );
}

#[test]
fn lower_retk_trunc_noncolliding_rejected() {
    // LOWER-RETK-TRUNC: the worst case — ret.0 + ret.(2^32+1) — used to lower
    // entirely clean (truncated to slot 1); it must now be L1205.
    assert_rejects(
        "fn f() -> (i32, i32) { 10 -> ret.0; 20 -> ret.4294967297; }\nfn main() {}\n",
        "L1205",
    );
}

#[test]
fn lower_memberidx_typing_trunc_rejected() {
    // LOWER-MEMBERIDX-TYPING-TRUNC: a member index > u32::MAX is rejected with
    // L1205 by emission; the typing pass no longer transiently widens against the
    // truncated component.
    assert_rejects(
        "fn f(x: (i32, i32)) -> i32 { x.4294967296 -> ret; }\nfn main() {}\n",
        "L1205",
    );
}

#[test]
fn lower_2_value_producing_effectful_ret_lowers_clean() {
    // LOWER-2: the §6.2 `yes | present | present` row — an effectful fn with a
    // declared return B and a value-producing `<expr> -> ret`. Lowers clean as a
    // `pack(tok, value) -> Dest::Ret { slot: None }` full-tuple writer (LD18).
    let src = "fn f(x: i32) -> i32 { x -> print; x + 1 -> ret; }\nfn main() { 3 -> f -> r; r -> print; }\n";
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn lower_2_value_producing_effectful_fn_body_tail_lowers_clean() {
    // LOWER-2 case 2: the fn-body virtual-ret tail in an effectful B-present fn.
    let src = "fn f(x: i32) -> i32 { x -> print; x + 1 }\nfn main() { 3 -> f -> r; r -> print; }\n";
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_03_effectful_call_in_phi_arm_rejected() {
    // ATK-03: an effectful user-defined call in a Phi-position arm must be L1404
    // (not silently accepted, firing the effect unconditionally on both paths).
    let src = r#"fn log(x: i32) { x -> print; }
fn main() {
    (1 > 0) -> {
        -true-> { 5 -> log; 9 }
        -false-> 7;
    } -> v;
    "done" -> print;
}
"#;
    assert_rejects(src, "L1404");
}

#[test]
fn atk_04_bool_guard_result_ty_recorded() {
    // ATK-04: a bool-guard Phi result must have its ty recorded (and merge tag
    // propagated). Printing it (was L1207), using it as a routing-guard cond (was
    // L1406), and a jump next-state through it (was L1503) must all lower clean.
    // (a) print the bool-guard result.
    let a = "fn main() { (1 > 0) -> { -true-> 1; -false-> 0; } -> y; y -> print; }\n";
    let ir = lower_ok(a);
    assert!(mapal_ir::validate(&ir).is_empty());
    // (b) routing-guard cond through a Phi.
    let b = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        (i < n) -> { -true-> true; -false-> false; } -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> i -> ret;
        }
    }
}
fn main() { 5 -> f -> r; r -> print; }
"#;
    let irb = lower_ok(b);
    assert!(mapal_ir::validate(&irb).is_empty());
}

#[test]
fn atk_08_effect_after_consumed_token_is_l1307() {
    // ATK-08: a print after a ret-write in an effectful fn (token consumed) must
    // be L1307, not the internal L1901 "print in pure fn".
    let src = "fn f(x: i32) -> i32 { x -> print; x -> ret; 5 -> print; }\nfn main() { 3 -> f -> r; r -> print; }\n";
    assert_rejects(src, "L1307");
    // And the ret-terminal loop-exit variant.
    let loop_src = r#"fn main() {
    mut i: i32 <- 3;
    loop {
        i -> print;
        (i > 0) -> {
            -true-> { i - 1 -> i; -> loop; }
            -false-> -> ret;
        }
    }
    5 -> print;
}
"#;
    assert_rejects(loop_src, "L1307");
}

#[test]
fn atk_15_bool_guard_arm_bookkeeping() {
    // ATK-15(a): a default-only bool guard — both poles missing → L1401.
    assert_rejects(
        "fn f(x: bool) -> i32 { x -> { -_-> 5; } -> ret; }\nfn main() {}\n",
        "L1401",
    );
    // ATK-15(b): duplicate `-_->` default arms → L1402.
    assert_rejects(
        "fn f(x: bool) -> i32 { x -> { -true-> 1; -_-> 2; -_-> 3; } -> ret; }\nfn main() {}\n",
        "L1402",
    );
    // ATK-15(c): a default coexisting with both concrete poles (unreachable) → L1402.
    assert_rejects(
        "fn f(x: bool) -> i32 { x -> { -true-> 1; -false-> 2; -_-> 3; } -> ret; }\nfn main() {}\n",
        "L1402",
    );
}

#[test]
fn atk_16_second_effectful_ret_is_l1307() {
    // ATK-16: a second surface ret-write in an effectful fn must be L1307 (each
    // write consumes the final token), not L1201.
    let src = "fn f(x: i32) -> i32 { x -> print; x -> ret; x -> ret; }\nfn main() { 3 -> f -> r; r -> print; }\n";
    let codes = lower_err_codes(src);
    assert!(codes.iter().any(|c| c == "L1307"), "got {codes:?}");
    assert!(!codes.iter().any(|c| c == "L1201"), "got {codes:?}");
}

#[test]
fn atk_17_unit_effectful_fanout_tail_lowers_clean() {
    // ATK-17: a fanout as the fn-body tail of a Unit-output effectful fn must
    // lower clean (the tail is dead code as a value; LD21) — like the `;`-form.
    let tail = "fn main() { 1 -> { -> print; -> print; } }\n";
    let ir = lower_ok(tail);
    assert!(mapal_ir::validate(&ir).is_empty());
    let stmt = "fn main() { 1 -> { -> print; -> print; }; }\n";
    let ir2 = lower_ok(stmt);
    assert!(mapal_ir::validate(&ir2).is_empty());
}

#[test]
fn atk_18_zero_param_call_is_l1201() {
    // ATK-18: calling a zero-param fn with an argument must be an L1201-family
    // arg-vs-input mismatch, not the nonsensical L1203 "int literal cannot have
    // type `()`".
    let src = "fn g() -> i32 { 5 -> ret; }\nfn main() { 7 -> g -> x; x -> print; }\n";
    let codes = lower_err_codes(src);
    assert!(codes.iter().any(|c| c == "L1201"), "got {codes:?}");
    assert!(!codes.iter().any(|c| c == "L1203"), "got {codes:?}");
}

#[test]
fn atk_19_headless_bind_exit_arm_seeds_scrutinee() {
    // ATK-19: a headless `-> name` binding as a routing-guard exit arm must seed
    // `cur := scrutinee` (§8.3), not misreport (was L1305/L1301). The exit arm
    // `-> c` binds the guard's incoming bool cond, which is then returned.
    let src = r#"fn f(n: i32) -> bool {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> -> c;
        }
    }
    c -> ret;
}
fn main() { 5 -> f -> r; r -> print; }
"#;
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_05_loop_exit_binding_visible_after_loop() {
    // ATK-05: an exit-arm Bind terminal (`-false-> i -> total`) must bind in the
    // loop's ENCLOSING scope (§8.5 step 4), so the post-loop `total -> ret` reads
    // it (was a false L1101 when bound into the popped loop-body frame).
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> i -> total;
        }
    }
    total -> ret;
}
fn main() { 5 -> f -> r; r -> print; }
"#;
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_07_pure_var_tail_returns() {
    // ATK-07(a): a pure fn-body tail that is a bare VARIABLE (`{ x }`) must route
    // through `output()`/`emit_ret_existing` (LD21 + §8.1: bare pre-existing →
    // Output), not pass `Dest::Ret` to the `Var` arm (which ignored it → seal
    // `RetSlotMissing` / L1901).
    let a = "fn f(x: i32) -> i32 { x }\nfn main() { 1 -> f -> r; r -> print; }\n";
    let ir = lower_ok(a);
    assert!(mapal_ir::validate(&ir).is_empty());
    // A bare literal tail returns the same way (also a pre-existing wire → Output).
    let lit = "fn g(x: i32) -> i32 { 5 }\nfn main() { 1 -> g -> r; r -> print; }\n";
    let irl = lower_ok(lit);
    assert!(mapal_ir::validate(&irl).is_empty());
}

#[test]
fn atk_07_effectful_b_tail_returns() {
    // ATK-07(b): an effectful-with-B fn-body tail (`{ x -> print; x + 1 }`) must
    // lower clean (the `pack(tok, value) -> Dest::Ret` full-tuple writer; LD18).
    let src = "fn f(x: i32) -> i32 { x -> print; x + 1 }\nfn main() { 3 -> f -> r; r -> print; }\n";
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_09_valueless_ret_exit_in_unit_fn_lowers_clean() {
    // ATK-09: a value-less `-> ret` loop-exit in a pure Unit-output fn must lower
    // clean (was a false L1306). §8.5/LP-1: the payload is the merge-state view
    // (the merge object), exiting to `Dest::Fresh(None)` with zero Return writers.
    let src = r#"fn f(n: i32) {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> -> ret;
        }
    }
}
fn main() { 3 -> f; "done" -> print; }
"#;
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_10_statement_level_bare_ret_is_permitted() {
    // ATK-10: a statement-level bare `-> ret;` is in the catalogue (§8.3), not a
    // false L1301. (a) pure Unit fn: a no-op.
    let a = "fn f(x: i32) { -> ret; }\nfn main() {}\n";
    let ir = lower_ok(a);
    assert!(mapal_ir::validate(&ir).is_empty());
    // (b) effectful B-absent fn: writes the current token to Return.
    let b = "fn main() { \"a\" -> print; -> ret; }\n";
    let irb = lower_ok(b);
    assert!(mapal_ir::validate(&irb).is_empty());
}

#[test]
fn atk_10_bare_ret_then_effect_is_l1307() {
    // ATK-10 corollary: after a statement-level `-> ret;` consumes the token in an
    // effectful fn, a later effect finds the register consumed → L1307.
    assert_rejects(
        "fn main() { \"a\" -> print; -> ret; \"b\" -> print; }\n",
        "L1307",
    );
}

#[test]
fn atk_11_semicolon_terminated_exit_block_lowers_clean() {
    // ATK-11: an exit arm that is a Block with `tail: None` (a `;`-terminated
    // terminal) must be handled like the same arm without `;` (LD21:
    // statement-vs-tail identical) — the last chain item is peeled as the
    // terminal. (a) `-false-> { -> ret; }`.
    let a = r#"fn f(n: i32) {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> { -> ret; }
        }
    }
}
fn main() { 3 -> f; "done" -> print; }
"#;
    let ir = lower_ok(a);
    assert!(mapal_ir::validate(&ir).is_empty());
    // (b) `-false-> { 99 -> print; -> ret; }` (preceding statement + ret terminal).
    let b = r#"fn f(n: i32) {
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> { 99 -> print; -> ret; }
        }
    }
}
fn main() { 3 -> f; "done" -> print; }
"#;
    let irb = lower_ok(b);
    assert!(mapal_ir::validate(&irb).is_empty());
}

#[test]
fn atk_12_pure_call_result_derives_from_merge() {
    // ATK-12: a pure-call result must carry the derives-from-merge tag (§7.3: any
    // derived operand ⇒ derived result), so a guard cond flowing through a pure
    // call passes the L1503 derivation test (was a false L1503).
    let src = r#"fn isneg(x: i32) -> bool { (x < 0) -> ret; }
fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        i - n -> isneg -> c;
        c -> {
            -true-> { i + 1 -> i; -> loop; }
            -false-> i -> ret;
        }
    }
}
fn main() { 5 -> f -> r; r -> print; }
"#;
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}

#[test]
fn atk_13_headless_routing_guard_is_l1301() {
    // ATK-13: a headless routing guard (`-> { -true-> -> loop; -false-> … }`) has
    // no scrutinee — a user error reported as L1301 HeadlessChain, not the
    // internal L1901 (which must stay unreachable from parse-clean input).
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    loop {
        i + 1 -> i;
        -> {
            -true-> -> loop;
            -false-> i -> ret;
        };
    }
}
fn main() { 5 -> f -> r; r -> print; }
"#;
    let codes = lower_err_codes(src);
    assert!(codes.iter().any(|c| c == "L1301"), "got {codes:?}");
    assert!(!codes.iter().any(|c| c == "L1901"), "got {codes:?}");
}

#[test]
fn atk_14_loop_in_map_body_lowers_clean() {
    // ATK-14: the map/fold-body capture checker must descend into loop bodies,
    // registering body-local loop binds AND exit-arm Bind names as body-locals
    // (was a false L1108). The loop inside the map body is Core-legal (bodies are
    // ordinary token-free fns) and lowers clean.
    let src = r#"fn main() {
    [1, 2, 3] -> map {
        x ->
        mut i: i32 <- 0;
        loop {
            (i < x) -> {
                -true-> { i + 1 -> i; -> loop; }
                -false-> i -> v;
            }
        }
        v
    } -> brr: [i32; 3];
    brr[0] -> print;
}
"#;
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
}
