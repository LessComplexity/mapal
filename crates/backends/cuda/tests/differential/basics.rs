//! Differential tests over the examples, loops, traps, float printing and loop-driven updates.

use super::*;

// --- tests ----------------------------------------------------------------

/// The §6.8 line, part 1: the 10 examples (`vector.mapal` excluded — the
/// aspirational exhibit, not Core), raw and rewritten, compile-and-run
/// oracle-equal.
#[test]
fn differential_examples_raw_and_rewritten() {
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_examples_raw_and_rewritten: {SKIP_REASON}");
        return;
    };
    for name in EXAMPLES {
        let ir = build_example(name);
        let rr = run(&ir, BUDGET);
        assert_parity(&nvcc, &ir, &rr, &format!("{name}/raw"));

        // Rebuild (CategoryIr is not Clone) and rewrite.
        let res = rewrite(build_example(name));
        let rr2 = run(&res.ir, BUDGET);
        assert_parity(&nvcc, &res.ir, &rr2, &format!("{name}/rewritten"));
    }
}

#[test]
fn differential_iota_fill() {
    let src = r#"
fn main() {
    4 -> iota -> a;
    (7, 4) -> fill -> b;
    (a, b) -> zip -> map { p -> p.0 + p.1 } -> c;
    c[3] -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "10\n", "oracle contract");
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_iota_fill (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "iota_fill/raw");

    // Rebuild (CategoryIr is not Clone) and rewrite — the replay Iota/Fill
    // arms are on this path.
    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, "10\n", "rewritten oracle contract");
    assert_parity(&nvcc, &res.ir, &rr2, "iota_fill/rewritten");
}

#[test]
fn differential_widen() {
    let src = r#"
fn main() {
    2 -> iota -> a;
    a[1] - 2 -> x;
    x -> widen<i64> -> i;
    a -> map { y -> y - 2 -> widen<f32> } -> fs;
    a -> map { y -> y - 2 -> widen<f64> } -> ds;
    fs -> map { y -> y -> widen<f64> } -> fds;
    i -> println;
    fs[1] -> println;
    ds[1] -> println;
    fds[1] -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "-1\n-1\n-1\n-1\n", "oracle contract");

    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, rr.output, "rewritten oracle contract");

    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_widen (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "widen/raw");
    assert_parity(&nvcc, &res.ir, &rr2, "widen/rewritten");
}

/// Two sequential loops in one fn (S12 P0 shape) compile-and-run to "20\n" —
/// raw and rewritten (§6.8).
#[test]
fn differential_two_sequential_loops() {
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
    assert_eq!(rr.output, "20\n", "oracle contract");
    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, "20\n", "rewritten oracle contract");
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_two_sequential_loops (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "two_loops/raw");
    assert_parity(&nvcc, &res.ir, &rr2, "two_loops/rewritten");
}

/// The llvm `exit_only_payload_emitted_once` pin (`golden_ll.rs:198`),
/// ported: an exit-only computed payload (`acc + 12345`, consumed only by
/// the exit route) belongs to the decide cone but lives outside the loop
/// SCC. The driver-ownership skip must emit it exactly once — re-emission
/// after the loop is dead recompute for values and a DOUBLE side effect for
/// an exit-arm Print (the llvm miscompile class). TEXT-ONLY like llvm's, on
/// raw AND rewritten `.cu` (§6.8): the interp oracle PANICS on this shape
/// today (a decide-cone read-before-write in mapal-interp's loop driver —
/// the reason llvm pins it textually), so no oracle expectation exists to
/// compile-run against; and a pure recompute would be runtime-unobservable
/// anyway — the occurrence count IS the detector. Runs wherever loop
/// emission exists (WP4 landed).
#[test]
fn differential_exit_only_payload_emitted_once() {
    let src = r#"
fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut acc: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { acc + 2 -> acc; i + 1 -> i; -> loop; }
            -false-> acc + 12345 -> ret;
        }
    }
}
fn main() { 4 -> f -> r; r -> println; }
"#;
    // Raw and rewritten (§6.8): the payload constant appears exactly once.
    for tag in ["raw", "rewritten"] {
        let ir = if tag == "raw" {
            lower_src(src)
        } else {
            rewrite(lower_src(src)).ir
        };
        let cu = match mapal_backend_cuda::emit(&ir) {
            Ok(cu) => cu,
            Err(e) => panic!("exit_only_payload/{tag}: emit failed: {e:?}"),
        };
        assert_eq!(
            cu.matches("12345").count(),
            1,
            "exit_only_payload/{tag}: exit-only payload must be emitted exactly once \
             (driver-owned):\n{cu}"
        );
    }
}

/// Trap parity (exit 101): div-by-zero and an out-of-bounds `Update` — raw
/// and rewritten, the rewritten oracle re-asserted trapped (R1: rewrite
/// preserves the trap class).
#[test]
fn differential_traps_exit_101() {
    // Div by zero (surface).
    let div0 = r#"
fn f(a: i32, b: i32) -> i32 { a / b -> ret; }
fn main() { (1, 0) -> f -> r; r -> println; }
"#;
    let ir = lower_src(div0);
    let rr = run(&ir, BUDGET);
    assert!(
        matches!(rr.outcome, Outcome::Trapped(_)),
        "oracle must trap"
    );
    // Out-of-bounds Update (index 9 into [i32; 4]) — IrBuilder (llvm port).
    let oob = build_update_oob();
    let rr_oob = run(&oob, BUDGET);
    assert!(
        matches!(rr_oob.outcome, Outcome::Trapped(_)),
        "expected OOB trap"
    );
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_traps_exit_101 (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "div0/raw");
    let res = rewrite(lower_src(div0));
    let rr2 = run(&res.ir, BUDGET);
    assert!(
        matches!(rr2.outcome, Outcome::Trapped(_)),
        "rewritten oracle must trap"
    );
    assert_parity(&nvcc, &res.ir, &rr2, "div0/rewritten");
    assert_parity(&nvcc, &oob, &rr_oob, "update_oob/raw");
    let res2 = rewrite(build_update_oob());
    let rr2_oob = run(&res2.ir, BUDGET);
    assert!(
        matches!(rr2_oob.outcome, Outcome::Trapped(_)),
        "rewritten OOB oracle must trap"
    );
    assert_parity(&nvcc, &res2.ir, &rr2_oob, "update_oob/rewritten");
}

/// `Unit → i32` closed pure: build [0;4], `Update` at 9 (OOB), `Index` [0]
/// into the result — the wrapper would print the return; the trap fires
/// first (llvm harness construction, ported).
pub(crate) fn build_update_oob() -> CategoryIr {
    use mapal_ir::{Dest, FuncKind, IrBuilder, Value};
    let mut b = IrBuilder::new();
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
    b.seal(f).unwrap()
}

/// Float-print parity (§6.8 pin): f64 shortest round-trip Display
/// (`0.1 + 0.2`), an f32 division, IEEE float ÷0 (±inf/NaN — the ADR-0013
/// S13 amendment: NO trap), and float `Mod` (Rust `%` ≡ C `fmod` — the
/// DESIGN §4/§10 open question this case pins). Every print flows through
/// the same mapal-rt on the host, so what the pin checks is VALUE parity
/// (`-fmad=false`, BC9) rendered byte-equal.
#[test]
fn differential_float_print_parity() {
    let src = r#"
fn main() {
    0.1 + 0.2 -> s;
    s -> println;
    1.0 / 3.0 -> t: f32;
    t -> println;
    1.0 / 0.0 -> pinf;
    pinf -> println;
    0.0 - 1.0 -> neg1;
    neg1 / 0.0 -> ninf;
    ninf -> println;
    0.0 / 0.0 -> nan;
    nan -> println;
    7.5 % 2.0 -> m;
    m -> println;
    0.0 - 7.5 -> neg75;
    neg75 % 2.0 -> nm;
    nm -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(
        rr.output, "0.30000000000000004\n0.33333334\ninf\n-inf\nNaN\n1.5\n-1.5\n",
        "oracle contract"
    );
    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, rr.output, "rewritten oracle contract");
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_float_print_parity (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "float_print/raw");
    assert_parity(&nvcc, &res.ir, &rr2, "float_print/rewritten");
}

/// ADR-0021's motivating program, end-to-end through nvcc: matmul4 as one
/// flattened loop building the result via a loop-carried `mut c` array and
/// `c[t] <- v` (the U4 contract) — oracle-equal ("8\n136\n"), raw and
/// rewritten. Covers the combined loop-carried + `Update` shape that
/// testgen's disjoint Update/loop steps never compose (llvm harness port).
#[test]
fn differential_matmul_loop_driven_update() {
    let src = r#"
fn cell(a: [f32; 16], b: [f32; 16], i: i32, j: i32) -> f32 {
    mut k: i32   <- 0;
    mut acc: f32 <- 0.0;
    loop {
        (k < 4) -> {
            -true-> {
                acc + a[i * 4 + k] * b[k * 4 + j] -> acc;
                k + 1 -> k;
                -> loop;
            }
            -false-> acc -> ret;
        }
    }
}

fn matmul4(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    mut c: [f32; 16] <- b;
    mut t: i32       <- 0;
    loop {
        (t < 16) -> {
            -true-> {
                t / 4 -> i;
                t % 4 -> j;
                (a, b, i, j) -> cell -> v;
                c[t] <- v;
                t + 1 -> t;
                -> loop;
            }
            -false-> c -> ret;
        }
    }
}

fn main() {
    [ 1.0,  2.0,  3.0,  4.0,
      5.0,  6.0,  7.0,  8.0,
      9.0, 10.0, 11.0, 12.0,
     13.0, 14.0, 15.0, 16.0] -> a: [f32; 16];

    [1.0, 0.0, 0.0, 0.0,
     0.0, 1.0, 0.0, 0.0,
     0.0, 0.0, 1.0, 0.0,
     0.0, 0.0, 0.0, 1.0] -> b: [f32; 16];

    (a, b) -> matmul4 -> c;

    c[7] -> println;
    (0.0, c) -> fold { acc, x -> acc + x } -> sum;
    sum -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "8\n136\n", "oracle contract");
    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, "8\n136\n", "rewritten oracle contract");
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_matmul_loop_driven_update (native leg): {SKIP_REASON}");
        return;
    };
    assert_parity(&nvcc, &ir, &rr, "matmul_loop_driven/raw");
    assert_parity(&nvcc, &res.ir, &rr2, "matmul_loop_driven/rewritten");
}
