//! Differential tests over the examples, sequential loops, trap exits and u8 indexing.

use super::*;

// --- tests ----------------------------------------------------------------

/// The M2 line: the 10 examples, raw and rewritten, compile-and-run oracle-equal
/// at `-O0` and `-O2`.
#[test]
fn differential_examples_raw_and_rewritten() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_examples: clang not found");
        return;
    };
    for name in EXAMPLES {
        let ir = build_example(name);
        let rr = run(&ir, BUDGET);
        assert_parity(&clang, &ir, &rr, &format!("{name}/raw"));

        // Rebuild (CategoryIr is not Clone) and rewrite.
        let res = rewrite(build_example(name));
        let rr2 = run(&res.ir, BUDGET);
        assert_parity(&clang, &res.ir, &rr2, &format!("{name}/rewritten"));
    }
}

/// Two sequential loops in one fn (S12 P0 shape) compile-and-run to "20\n".
#[test]
fn differential_two_sequential_loops() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut a: i32 <- 0;
    loop { (i < n) -> { -true-> { a + 2 -> a; i + 1 -> i; -> loop; } -false-> a -> aa; } }
    mut j: i32 <- 0;
    mut b: i32 <- 0;
    loop { (j < n) -> { -true-> { b + 3 -> b; j + 1 -> j; -> loop; } -false-> b -> bb; } }
    aa + bb -> ret;
}
fn main() { 4 -> f -> r; r -> println; }
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "20\n");
    assert_parity(&clang, &ir, &rr, "two_loops");
}

/// Trap parity (exit 101): div-by-zero and an out-of-bounds `Update`.
#[test]
fn differential_traps_exit_101() {
    let Some(clang) = clang() else {
        return;
    };
    // Div by zero.
    let div0 = r#"
fn f(a: i32, b: i32) -> i32 { a / b -> ret; }
fn main() { (1, 0) -> f -> r; r -> println; }
"#;
    let ir = lower_src(div0);
    let rr = run(&ir, BUDGET);
    assert!(matches!(rr.outcome, Outcome::Trapped(_)));
    assert_parity(&clang, &ir, &rr, "div0");

    // Out-of-bounds Update (index 9 into [i32; 4]) — built via IrBuilder.
    use mapal_ir::{Dest, FuncKind, IrBuilder, SourceLoc, Value};
    const L: SourceLoc = SourceLoc { start: 0, end: 0 };
    let mut b = IrBuilder::new();
    let arr_ty = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    // Unit -> i32 closed pure: build [0;4], update OOB, index [0] → print via wrapper.
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let arr = fb
            .pack_array(&[zero, zero, zero, zero], Dest::Fresh(None), L)
            .unwrap();
        let oob = fb.constant(Value::I32(9), L).unwrap();
        let v = fb.constant(Value::I32(7), L).unwrap();
        let up = fb.update(arr, oob, v, Dest::Fresh(None), L).unwrap();
        let i0 = fb.constant(Value::I32(0), L).unwrap();
        fb.index(up, i0, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let _ = arr_ty;
    let ir = b.seal(f).unwrap();
    let rr = run(&ir, BUDGET);
    assert!(
        matches!(rr.outcome, Outcome::Trapped(_)),
        "expected OOB trap"
    );
    assert_parity(&clang, &ir, &rr, "update_oob");
}

/// plan-s38: a trap must not swallow output that precedes it in SOURCE order.
///
/// **This is the one thing `assert_parity` cannot see.** `expect_native` maps
/// `Outcome::Trapped(_)` to `(None, 101)`, and the stdout comparison is guarded
/// by `if let Some(want)`, so every trapping case in the 1,280-run sweep checks
/// the exit code and *discards* stdout — at both `-O0` and `-O2`. The pre-trap
/// output prefix therefore had zero coverage in this suite.
///
/// It is also exactly what the source-position tie-break changed. The prints and
/// the `f` call are **independent** — a `Call` carries no IoToken, so the graph
/// imposes no order between them and `topo_order`'s tie-break decides. Under the
/// old insertion-order tie-break the emitter could hoist the call (and, in the
/// parallel spine, its synchronous `mapal_par_run_pinned`) ahead of both prints,
/// so the trap killed the process before either line was written. Source order
/// says both lines come first.
///
/// The prefix is deterministic: `mapal_trap` flushes stdout before `exit(101)`.
///
/// **The expectation is a literal, not the oracle's `rr.output`, and it has to
/// be.** `interp::run` derives `output` from the IoToken's accumulated log —
/// `(Outcome::Done(RValue::Token(log)), true) => log.clone(), _ => String::new()`
/// (crates/mapal-interp/src/lib.rs:55). Interpreted output is a *value* carried
/// by the token; on a trap the token never reaches the Return, so the log dies
/// with the aborted computation and `rr.output` is `""`. Compiled output is a
/// real side effect and survives. The two I/O models therefore diverge exactly
/// on the trap path, which is why `expect_native` returns `None` for stdout on a
/// trapping run — that is forced, not lazy. Any test in this class must pin the
/// expected prefix from the source, as this one does.
///
/// Verified as a real negative control, not just a regression pin: the same
/// program emitted by the compiler at `main@d3ca82c` hoists
/// `mapal_par_run_pinned` ahead of ALL THREE prints and outputs nothing before
/// dying; the fixed compiler emits print/print/run_pinned/print. Both exit 101,
/// which is exactly why the 1,280-run sweep could not see the difference.
#[test]
fn differential_trap_preserves_preceding_output() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn f(a: i32, b: i32) -> i32 { a / b -> ret; }
fn main() {
    111 -> println;
    222 -> println;
    (1, 0) -> f -> r;
    r -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert!(
        matches!(rr.outcome, Outcome::Trapped(_)),
        "expected a div-zero trap, got {:?}",
        rr.outcome
    );

    // ADR-0038 D4: value legs pin `generic` for uniformity — the ONE ambient/
    // native leg in this file is differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .expect("emit");
    for opt in ["-O0", "-O2"] {
        let (out, code) =
            compile_run(&clang, &ll, "trap_prefix", opt).expect("native run timed out");
        assert_eq!(code, 101, "{opt}: exit code");
        // The assertion the suite was missing: stdout, on a trapping run.
        assert_eq!(
            String::from_utf8_lossy(&out),
            "111\n222\n",
            "{opt}: a trap swallowed output written before it in source order"
        );
    }
}

/// The u8 value path (DESIGN §1): the `mapal_print_u8` `i8 zeroext` ABI and the
/// u8 `Index` `zext`+guard. DESIGN §1 says a dropped `zeroext` prints garbage for
/// u8 > 127 on arm64 and *only the differential* can catch it (the mapal-rt unit
/// table can't). No example or testgen program uses u8, so this is the sole
/// compile-and-run cover for that class. Built via `IrBuilder` (no surface u8).
#[test]
fn differential_u8_index_and_print() {
    let Some(clang) = clang() else {
        return;
    };
    use mapal_ir::{Dest, FuncKind, IrBuilder, SourceLoc, Value};
    const L: SourceLoc = SourceLoc { start: 0, end: 0 };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let tok = fb.input();
        // [u8; 4] with high-bit-set values (200, 255 > 127).
        let e0 = fb.constant(Value::U8(200), L).unwrap();
        let e1 = fb.constant(Value::U8(50), L).unwrap();
        let e2 = fb.constant(Value::U8(255), L).unwrap();
        let e3 = fb.constant(Value::U8(10), L).unwrap();
        let arr = fb
            .pack_array(&[e0, e1, e2, e3], Dest::Fresh(None), L)
            .unwrap();
        // u8-typed index ⇒ exercises the `zext`+guard path; result 255 > 127
        // ⇒ exercises the print `i8 zeroext` ABI.
        let idx = fb.constant(Value::U8(2), L).unwrap();
        let got = fb.index(arr, idx, Dest::Fresh(None), L).unwrap();
        let tok = fb.println(tok, got, L).unwrap();
        fb.output(tok, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "255\n");
    assert_parity(&clang, &ir, &rr, "u8_index_print");
}
