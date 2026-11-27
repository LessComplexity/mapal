//! Differentials for loop-driven updates, exit payloads, by-ref calls, parallel runs, time bracket and profiles.

use super::*;

/// loop building the result via a loop-carried `mut c` array and `c[t] <- v`
/// (the U4 contract) — compiled and run through clang, oracle-equal ("8\n136\n").
/// Covers the combined loop-carried + `Update` shape that testgen's disjoint
/// Update/loop steps never compose (S13 orchestrator ruling on review finding
/// `loop-carried-update-never-clanged`).
#[test]
fn differential_matmul_loop_driven_update() {
    let Some(clang) = clang() else {
        return;
    };
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
    assert_parity(&clang, &ir, &rr, "matmul_loop_driven/raw");

    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_parity(&clang, &res.ir, &rr2, "matmul_loop_driven/rewritten");
}

/// ADR-0027 review blocker #1, end-to-end NATIVE: a computed exit payload and
/// an exit-arm captured map — chains that feed only the loop's exit route, so
/// they leave the SCC — compile-and-run oracle-equal (raw and rewritten).
/// Pre-fix, llvm emitted the payload chain AFTER the loop (silent miscompile:
/// the exit route read a never-written alloca).
#[test]
fn differential_computed_exit_payloads() {
    let Some(clang) = clang() else {
        return;
    };
    let cases: &[(&str, &str, &str)] = &[
        (
            "computed_exit_payload",
            r#"
fn f(n: i32) -> i32 {
    mut t: i32 <- 1;
    mut k: i32 <- 0;
    loop {
        (k < n) -> {
            -true-> {
                t + 1 -> t;
                k + 1 -> k;
                -> loop;
            }
            -false-> t * 2 -> ret;
        }
    }
}
fn main() { 3 -> f -> r; r -> println; }
"#,
            "8\n",
        ),
        (
            "exit_arm_captured_map",
            r#"
fn f(xs: [i32; 3], n: i32) -> [i32; 3] {
    mut t: i32 <- 10;
    mut k: i32 <- 0;
    loop {
        (k < n) -> {
            -true-> {
                t + 1 -> t;
                k + 1 -> k;
                -> loop;
            }
            -false-> xs -> map { e -> e + t } -> r;
        }
    }
    r
}
fn main() { ([1, 2, 3], 2) -> f -> r; r[0] -> println; r[2] -> println; }
"#,
            "13\n15\n",
        ),
    ];
    for (tag, src, expected) in cases {
        let ir = lower_src(src);
        let rr = run(&ir, BUDGET);
        assert_eq!(rr.output, *expected, "{tag}: oracle contract");
        assert_parity(&clang, &ir, &rr, &format!("{tag}/raw"));

        let res = rewrite(lower_src(src));
        let rr2 = run(&res.ir, BUDGET);
        assert_parity(&clang, &res.ir, &rr2, &format!("{tag}/rewritten"));
    }
}

/// BL5 amendment (suggestions #8), end-to-end NATIVE: the by-ref call-arg
/// shapes the examples don't isolate — a Named fn returning its whole by-ref
/// input product (the `load_whole` escaping-use assembly) and a bulk op reading
/// an array straight off the by-ref input product (`zip` on the input itself)
/// — compile-and-run oracle-equal (raw and rewritten, `-O0` and `-O2`).
#[test]
fn differential_byref_call_args() {
    let Some(clang) = clang() else {
        return;
    };
    let cases: &[(&str, &str, &str)] = &[
        (
            "whole_input_return",
            r#"
fn id(p: ([i32; 2], i32)) -> ([i32; 2], i32) {
    p -> ret;
}
fn main() { ([7, 9], 5) -> id -> q; q.0[1] -> println; q.1 -> println; }
"#,
            "9\n5\n",
        ),
        (
            "zip_on_input_product",
            r#"
fn dot(ab: ([i32; 3], [i32; 3])) -> i32 {
    ab -> zip -> zs;
    (0, zs) -> fold { acc, p -> acc + p.0 * p.1 } -> r;
    r -> ret;
}
fn main() { ([1, 2, 3], [4, 5, 6]) -> dot -> r; r -> println; }
"#,
            "32\n",
        ),
    ];
    for (tag, src, expected) in cases {
        let ir = lower_src(src);
        let rr = run(&ir, BUDGET);
        assert_eq!(rr.output, *expected, "{tag}: oracle contract");
        assert_parity(&clang, &ir, &rr, &format!("{tag}/raw"));

        let res = rewrite(lower_src(src));
        let rr2 = run(&res.ir, BUDGET);
        assert_parity(&clang, &res.ir, &rr2, &format!("{tag}/rewritten"));
    }
}

/// ADR-0029 stage 1: `iota`/`fill` — the loop-skeleton `.ll` compiled and run.
/// The only end-to-end cover until the lower stage lands (built via `IrBuilder`;
/// no surface syntax yet).
#[test]
fn differential_iota_fill() {
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
        // iota(5)[3] = 3 → println; fill(7, 4)[2] = 7 → println.
        let five = fb.constant(Value::I32(5), L).unwrap();
        let seq = fb.iota(five, Dest::Fresh(None), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let x = fb.index(seq, three, Dest::Fresh(None), L).unwrap();
        let seven = fb.constant(Value::I32(7), L).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let filled = fb.fill(seven, four, Dest::Fresh(None), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let y = fb.index(filled, two, Dest::Fresh(None), L).unwrap();
        let tok = fb.println(tok, x, L).unwrap();
        let tok = fb.println(tok, y, L).unwrap();
        fb.output(tok, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let rr = run(&ir, BUDGET);
    assert_parity(&clang, &ir, &rr, "iota_fill");
}

#[test]
fn differential_widen() {
    let src = r#"
fn main() {
    2 -> iota -> a;
    a[1] - 2 -> x;
    x -> widen<i64> -> i;
    x -> widen<f32> -> f;
    x -> widen<f64> -> d;
    f -> widen<f64> -> fd;
    i -> println;
    f -> println;
    d -> println;
    fd -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "-1\n-1\n-1\n-1\n", "oracle contract");

    let res = rewrite(lower_src(src));
    let rr2 = run(&res.ir, BUDGET);
    assert_eq!(rr2.output, rr.output, "rewritten oracle contract");

    let Some(clang) = clang() else {
        eprintln!("SKIP differential_widen: clang not found");
        return;
    };
    assert_parity(&clang, &ir, &rr, "widen/raw");
    assert_parity(&clang, &res.ir, &rr2, "widen/rewritten");
}

#[test]
fn differential_parallel_bign() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_parallel_bign: clang not found");
        return;
    };
    let src = r#"
fn main() {
    65536 -> iota -> cells;
    4 -> iota -> ks;
    cells -> map { cell ->
        cell / 256 -> row;
        cell % 256 -> col;
        (0, ks) -> fold { acc, k -> acc + (row + k) * (col + k) } -> value;
        value
    } -> out;
    out[65535] -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    // ADR-0038 D4: the split-task structure this asserts is profile shape
    // (grain sizes are profile fields) — pin the profile by name.
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert!(
        ll.contains("i32 1, ptr @task") && ll.contains("i64 65536"),
        "the large map must use a split task:\n{ll}"
    );
    assert_parity(&clang, &ir, &rr, "parallel_bign");
}

/// plan-s29 emission item 4 (heap lowering): an entry frame at or above the
/// 256 KB threshold moves from `alloca` to the `mapal_rt_alloc` arena — the
/// change that lets 2048² f32 run at all on macOS (64 MB hard stack ceiling).
/// The observable must not move: byte-equal to the interp oracle at -O0 AND
/// -O2. The fold is load-bearing — it reads EVERY cell, so an arena block
/// short by even one field corrupts or crashes instead of passing quietly.
/// The n=64 twin below the threshold is the negative control: same program,
/// still on the stack, same oracle discipline.
#[test]
fn differential_heap_lowered_arrays() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_heap_lowered_arrays: clang not found");
        return;
    };
    for (n, heap) in [(100_000u32, true), (64, false)] {
        let src = format!(
            "fn main() {{\n    {n} -> iota -> t;\n    \
             t -> map {{ x -> (x * 7 + 13) % 101 - 50 }} -> a;\n    \
             (0, a) -> fold {{ acc, x -> acc + x }} -> s;\n    \
             s -> println;\n    a[{}] -> println;\n}}\n",
            n - 1
        );
        let ir = lower_src(&src);
        let rr = run(&ir, BUDGET);
        // ADR-0038 D4: the arena threshold (`heap_min_bytes`) is a profile
        // field, so placement structure is profile shape — pin the name.
        let ll = mapal_backend_llvm::emit_with_opts(
            &ir,
            &mapal_backend_llvm::EmitOpts {
                target: "generic",
                ..mapal_backend_llvm::EmitOpts::exact()
            },
        )
        .unwrap();
        assert_eq!(
            ll.contains("%frame = call ptr @mapal_rt_alloc(i64 "),
            heap,
            "n={n}: the entry frame is arena-placed iff it crosses the threshold:\n{ll}"
        );
        assert_eq!(
            ll.contains("call void @mapal_rt_free_all()"),
            heap,
            "n={n}: the teardown pairs with the arena:\n{ll}"
        );
        assert_parity(&clang, &ir, &rr, &format!("heap_lowered_n{n}"));
    }
}

#[test]
fn differential_parallel_trap_order() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_parallel_trap_order: clang not found");
        return;
    };
    let src = r#"
fn main() {
    32 -> iota -> xs;
    xs[0] -> first;
    first -> println;
    xs -> map { x ->
        x - first -> shifted;
        shifted - 7 -> divisor;
        100 / divisor
    } -> doomed;
    "after" -> println;
}
"#;
    let ir = lower_src(src);
    let rr = run(&ir, BUDGET);
    assert!(matches!(rr.outcome, Outcome::Trapped(_)));
    let prefix = run(
        &lower_src("fn main() { 32 -> iota -> xs; xs[0] -> first; first -> println; }"),
        BUDGET,
    )
    .output;
    assert_eq!(prefix, "0\n");
    // ADR-0038 D4: pin the profile by name (the parallel-plan shape is
    // profile-derived; the value legs pin alongside for uniformity).
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert!(ll.contains("call void @mapal_par_trap(i64 "));
    for opt in ["-O0", "-O2"] {
        let (_dir, exe) = compile_exe(&clang, &ll, "parallel_trap_order", opt);
        let (out, code) = run_exe(&exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("parallel_trap_order/{opt}: timed out"));
        assert_eq!(code, 101, "parallel_trap_order/{opt}: exit code");
        assert_eq!(
            String::from_utf8_lossy(&out),
            prefix,
            "parallel_trap_order/{opt}: stdout prefix"
        );
    }
}

pub(crate) const PARALLEL_STABLE_SRC: &str = r#"
fn main() {
    8192 -> iota -> xs;
    3 -> scale;
    xs -> map { x -> x * scale } -> ys;
    (0, ys) -> fold { acc, x -> acc + x } -> total;
    total -> println;
}
"#;

#[test]
fn differential_parallel_env_matrix() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_parallel_env_matrix: clang not found");
        return;
    };
    let ir = lower_src(PARALLEL_STABLE_SRC);
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("oracle completes") else {
        panic!("parallel env fixture must complete");
    };
    // ADR-0038 D4: value legs pin `generic` for uniformity — the ONE ambient/
    // native leg in this file is differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    for opt in ["-O0", "-O2"] {
        let (_dir, exe) = compile_exe(&clang, &ll, "parallel_env_matrix", opt);
        for mapal_par in [Some("1"), Some("8"), None] {
            let (out, code) = run_exe(&exe, TIMEOUT_SECS, mapal_par)
                .unwrap_or_else(|| panic!("parallel_env_matrix/{opt}/{mapal_par:?}: timed out"));
            assert_eq!(code, 0, "parallel_env_matrix/{opt}/{mapal_par:?}");
            assert_eq!(
                String::from_utf8_lossy(&out),
                want,
                "parallel_env_matrix/{opt}/{mapal_par:?}"
            );
        }
    }
}

/// plan-time-builtin: a self-bracketing program. The bracketed work is
/// byte-identical to its untimed twin (the clock changes no answer), and the
/// printed `elapsed` is a finite f64 ≥ 0 at `-O0`/`-O2` under both the default
/// pool and `MAPAL_PAR=1`. The MAPAL_PAR legs are the S29 regression pin: before
/// `path_plan` fenced on `TimeMs` the second read fired while the bracketed
/// tasks were still in flight, and before the result's consumer cone was held
/// on the host spine the subtraction raced the host's write of the clock value
/// — both surfaced as a NEGATIVE elapsed. No upper bound is ever asserted (a
/// wall-clock bound is a flake, not a test).
#[test]
fn differential_time_bracket() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_time_bracket: clang not found");
        return;
    };
    let src = r#"
fn main() {
    8192 -> iota -> xs;
    () -> time -> t0;
    xs -> map { x -> (x * 7) % 13 } -> ys;
    (0, ys) -> fold { acc, y -> acc + y } -> total;
    () -> time -> t1;
    total -> println;
    t1 - t0 -> elapsed;
    elapsed -> println;
}
"#;
    // The same program with the bracket removed: its whole stdout is the
    // checksum line the timed run must reproduce verbatim.
    let untimed = run(
        &lower_src(
            r#"
fn main() {
    8192 -> iota -> xs;
    xs -> map { x -> (x * 7) % 13 } -> ys;
    (0, ys) -> fold { acc, y -> acc + y } -> total;
    total -> println;
}
"#,
        ),
        BUDGET,
    )
    .output;
    assert_eq!(untimed, "49147\n");
    // ADR-0038 D4: value legs pin `generic` for uniformity — the ONE ambient/
    // native leg in this file is differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        &lower_src(src),
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    for opt in ["-O0", "-O2"] {
        let (_dir, exe) = compile_exe(&clang, &ll, "time_bracket", opt);
        for mapal_par in [None, Some("1")] {
            let tag = format!(
                "time_bracket/{opt}/MAPAL_PAR={}",
                mapal_par.unwrap_or("default")
            );
            let (out, code) = run_exe(&exe, TIMEOUT_SECS, mapal_par)
                .unwrap_or_else(|| panic!("{tag}: timed out"));
            assert_eq!(code, 0, "{tag}: exit code");
            let out = String::from_utf8_lossy(&out);
            let (checksum, elapsed) = out.split_at(untimed.len());
            assert_eq!(checksum, untimed, "{tag}: the bracket changed the answer");
            // `elapsed` is printed by Rust's `Display` — a plain decimal, never
            // scientific notation.
            let ms: f64 = elapsed
                .trim_end()
                .parse()
                .unwrap_or_else(|e| panic!("{tag}: elapsed {elapsed:?} is not an f64: {e}"));
            assert!(
                ms.is_finite() && ms >= 0.0,
                "{tag}: elapsed must be finite and non-negative, got {ms}"
            );
        }
    }
}

#[test]
fn differential_parallel_run_twice() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_parallel_run_twice: clang not found");
        return;
    };
    let ir = lower_src(PARALLEL_STABLE_SRC);
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("oracle completes") else {
        panic!("parallel repeat fixture must complete");
    };
    // ADR-0038 D4: value legs pin `generic` for uniformity — the ONE ambient/
    // native leg in this file is differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    for opt in ["-O0", "-O2"] {
        let (_dir, exe) = compile_exe(&clang, &ll, "parallel_run_twice", opt);
        let first = run_exe(&exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("parallel_run_twice/{opt}/first: timed out"));
        let second = run_exe(&exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("parallel_run_twice/{opt}/second: timed out"));
        assert_eq!(first, second, "parallel_run_twice/{opt}: schedule drift");
        assert_eq!(first.1, 0, "parallel_run_twice/{opt}: exit code");
        assert_eq!(
            String::from_utf8_lossy(&first.0),
            want,
            "parallel_run_twice/{opt}: oracle"
        );
    }
}

/// plan-s31-target-profiles composition rule 2, discharged: **every profile
/// field is value-invariant**. `zen3` is the profile that actually changes
/// emission — 32 B vectors and 16 registers give TJ 16 → 32 and TI 4 → 2, so
/// the whole tile geometry moves — and this shape exercises a main tile, a
/// runtime remainder and an i-tail under BOTH profiles (C=40: generic runs two
/// TJ=16 tiles + tj=8, zen3 one TJ=32 tile + tj=8; rows=5 leaves a tail either
/// way, 5 % 4 == 1 and 5 % 2 == 1).
///
/// The claim under test is not "zen3 is fast" — it is unmeasured on hardware
/// and marked so. It is that choosing it cannot change an answer, which is what
/// keeps the differential suite a valid gate under every profile (ADR-0032 D1).
#[test]
fn differential_zen3_profile_is_value_invariant() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn matmul(a: [f32; 35], b: [f32; 280]) -> [f32; 200] {
    200 -> iota -> cells;
    7 -> iota -> ks;
    cells -> map { cell ->
        cell / 40 -> i;
        cell % 40 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 7 + k] * b[k * 40 + j] }
    } -> c;
    c -> ret;
}
fn main() {
    35 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    280 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    (a, b) -> matmul -> c;
    c[0] -> println;
    c[199] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("zen3 shape oracle completes") else {
        panic!("zen3 shape oracle must complete");
    };

    // ADR-0038 D4: `emit`'s default target is `native` (S46), so on a
    // zen3-like runner the ambient emission IS ≈zen3 and the assert_ne below
    // would compare a profile against itself. Pin the reference profile by
    // name — that is what makes the comparison machine-independent.
    let generic = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    let zen3 = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "zen3",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert_ne!(
        generic, zen3,
        "zen3 must actually re-tile (TJ 16 -> 32, TI 4 -> 2), else this proves nothing"
    );

    let untiled = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            tiling: false,
            // ADR-0038 D4: the comparison's other leg pins `generic` by name —
            // keep both legs inside the one profile.
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert_tiled_parity(&clang, &generic, &untiled, &want, "profile_generic");
    assert_tiled_parity(&clang, &zen3, &untiled, &want, "profile_zen3");
}

/// ADR-0038 D4: the ONE differential leg that intentionally floats on the host
/// profile — every other test pins a profile by name. `emit`'s default target
/// is `native` (S46), so this asserts that whatever this machine detects, the
/// emitted code computes the interp oracle's values. If this fails on a host,
/// detection or emission is value-divergent there — a real bug, not a shape
/// mismatch.
#[test]
fn differential_native_profile_is_value_correct() {
    let Some(clang) = clang() else {
        return;
    };
    // The same 4×4 map+fold matmul shape as differential_tiled_matmul: small
    // enough for the interp budget, matmul-shaped so the tile nest fires under
    // whatever geometry the host profile derives. ZERO shape assertions — the
    // shape pins live in the generic-pinned twin above.
    let src = r#"
fn main() {
    [ -37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0] -> a: [f32; 16];
    [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0] -> b: [f32; 16];
    16 -> iota -> cells;
    4 -> iota -> ks;
    cells -> map { cell ->
        cell / 4 -> i;
        cell % 4 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] }
    } -> c;
    c[0] -> println;
    c[15] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("native oracle completes") else {
        panic!("native oracle must complete");
    };
    // The only ambient profile left in this file — the host-detected default
    // is the whole point of this leg (on the exact face: the oracle's bits).
    let ll =
        mapal_backend_llvm::emit_with_opts(&ir, &mapal_backend_llvm::EmitOpts::exact()).unwrap();
    for opt in ["-O0", "-O2"] {
        let (out, code) = compile_run(&clang, &ll, &format!("native_profile/{opt}"), opt)
            .unwrap_or_else(|| panic!("native_profile/{opt}: run timed out"));
        assert_eq!(code, 0, "native_profile/{opt}: exit code");
        assert_eq!(
            String::from_utf8_lossy(&out),
            want,
            "native_profile/{opt}: oracle stdout"
        );
    }
}
