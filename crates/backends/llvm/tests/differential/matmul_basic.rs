//! Tiled matmul differentials: plain, via helper fn, and the default rewritten matmul4.

use super::*;

/// ADR-0027, end-to-end NATIVE: capturing map/fold bodies compile-and-run
/// oracle-equal — the capture components reach the body call as leading
/// arguments (the goldens pin the emission shape; this proves the emitted
/// `.ll` is valid and computes the oracle's values). The third program is the
/// ADR-0021 motivating matmul in its natural one-kernel map+fold form, pinned
/// to the S16 reference values (`-275` / `3748`).
#[test]
fn differential_captures() {
    let Some(clang) = clang() else {
        return;
    };
    let cases: &[(&str, &str)] = &[
        (
            "capture_map",
            r#"
fn main() {
    3 -> scale;
    [1, 2, 3] -> a;
    a -> map { x -> x * scale } -> b;
    b[1] -> println;
}
"#,
        ),
        (
            "capture_fold",
            r#"
fn main() {
    3 -> scale;
    [1, 2, 3] -> a;
    (0, a) -> fold { acc, x -> acc + x * scale } -> total;
    total -> println;
}
"#,
        ),
        (
            "capture_one_kernel_matmul",
            r#"
fn matmul(a: [f32; 16], b: [f32; 16], ks: [i32; 4]) -> [f32; 16] {
    a -> enumerate -> map { p ->
        p.0 / 4 -> i;
        p.0 % 4 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] } -> cell;
        cell
    } -> c;
    c -> ret;
}
fn main() {
    [ -37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0] -> a: [f32; 16];
    [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0] -> b: [f32; 16];
    [0, 1, 2, 3] -> ks: [i32; 4];
    (a, b, ks) -> matmul -> c;
    c[0] -> println;
    c[15] -> println;
}
"#,
        ),
    ];
    let want: &[&str] = &["6\n", "18\n", "-275\n3748\n"];
    for ((tag, src), expected) in cases.iter().zip(want) {
        let ir = lower_src(src);
        let rr = run(&ir, BUDGET);
        assert_eq!(rr.output, *expected, "{tag}: oracle contract");
        assert_parity(&clang, &ir, &rr, &format!("{tag}/raw"));

        let res = rewrite(lower_src(src));
        let rr2 = run(&res.ir, BUDGET);
        assert_parity(&clang, &res.ir, &rr2, &format!("{tag}/rewritten"));
    }
}

#[test]
fn differential_tiled_matmul() {
    let Some(clang) = clang() else {
        return;
    };
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
    let (Some(want), 0) = expect_native(&ir, &rr).expect("matmul oracle completes") else {
        panic!("matmul oracle must complete");
    };
    // ADR-0038 D4: shape constants are generic-derived — pin the profile by
    // name, never the ambient default (native since S46).
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    // S26 nest: flat [TILE_I * TILE_J] acc scratch, the head/interior/tail row
    // split (biased-`lo` + direct-`hi` udivs bound the interior full-window
    // rows), and a constant-TILE_J lane bound on the main j body. SSA tmp names
    // renumber with any emitter edit, so the lane bound pins the line shape
    // (`icmp uge i64 .., 16`), not a name.
    assert!(
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = udiv i64 %lo, 4")
            && tiled.contains(" = add i64 %hi, 3")
            && tiled.contains(" = add i64 %lo, 3")
            && tiled.contains(" = udiv i64 %hi, 4")
            && tiled
                .lines()
                .any(|l| l.contains(" = icmp uge i64 ") && l.ends_with(", 16")),
        "split task must contain the TI-blocked tiled row-clipping nest:\n{tiled}"
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
    assert!(
        untiled.contains(" = call float @fn"),
        "untiled map must retain its body call:\n{untiled}"
    );

    for opt in ["-O0", "-O2"] {
        let tiled_run = compile_run(&clang, &tiled, &format!("tiled_matmul/{opt}"), opt)
            .unwrap_or_else(|| panic!("tiled_matmul/{opt}: tiled run timed out"));
        let untiled_run = compile_run(&clang, &untiled, &format!("untiled_matmul/{opt}"), opt)
            .unwrap_or_else(|| panic!("tiled_matmul/{opt}: untiled run timed out"));
        assert_eq!(tiled_run.1, 0, "tiled_matmul/{opt}: tiled exit code");
        assert_eq!(untiled_run.1, 0, "tiled_matmul/{opt}: untiled exit code");
        assert_eq!(
            String::from_utf8_lossy(&tiled_run.0),
            want,
            "tiled_matmul/{opt}: oracle stdout"
        );
        assert_eq!(
            tiled_run.0, untiled_run.0,
            "tiled_matmul/{opt}: tiled/untiled stdout"
        );
    }
}

#[test]
fn differential_tiled_matmul_via_helper_fn() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn dot(a: [f32; 16], b: [f32; 16], kr: [i32; 4], cell: i32) -> f32 {
    cell / 4 -> i;
    cell % 4 -> j;
    (0.0, kr) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] } -> ret;
}
fn main() {
    [ -37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0] -> a: [f32; 16];
    [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0] -> b: [f32; 16];
    16 -> iota -> cells;
    4 -> iota -> kr;
    cells -> map { cell -> (a, b, kr, cell) -> dot } -> c;
    c[0] -> println;
    c[15] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    assert!(
        ir.morphisms()
            .all(|(_, m)| !matches!(m.op, Operation::Call(_))),
        "default rewrite must strip the helper Call inside the MapBody"
    );
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("helper matmul oracle completes") else {
        panic!("helper matmul oracle must complete");
    };
    // ADR-0038 D4: shape constants are generic-derived — pin the profile by
    // name, never the ambient default (native since S46).
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert!(
        tiled
            .lines()
            .any(|line| line.ends_with(" = alloca [64 x float], align 64"))
            && tiled
                .lines()
                .any(|line| line.ends_with(" = alloca [64 x float]")),
        "helper-free MapBody must expose the packed tiled nest and accumulator:\n{tiled}"
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

    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_via_helper");
}

#[test]
fn differential_default_rewritten_matmul4_lifts_and_tiles() {
    let Some(clang) = clang() else {
        return;
    };
    let ir = rewrite(build_example("matmul4_loop")).ir;
    assert!(
        ir.morphisms()
            .all(|(_, m)| !matches!(m.op, Operation::Call(_))),
        "the lifted cell/matmul callees must inline"
    );
    assert!(
        ir.funcs().all(|(f, _)| ir.loop_structure(f).is_empty()),
        "the default-rewritten matmul4 must be loop-SCC-free"
    );
    assert!(
        ir.morphisms().any(|(_, m)| {
            let Operation::Map { body, .. } = m.op else {
                return false;
            };
            ir.func(body).expect("MapBody").morphisms.iter().any(|&bm| {
                matches!(
                    ir.morphism(bm).expect("body morphism").op,
                    Operation::Fold { .. }
                )
            })
        }),
        "the lifted graph must contain a Map with a Fold body"
    );

    let rr = run(&ir, BUDGET);
    assert_eq!(rr.output, "-275\n3748\n", "interp oracle contract");
    // ADR-0038 D4: shape constants are generic-derived — pin the profile by
    // name, never the ambient default (native since S46). The value legs below
    // (default pool and MAPAL_PAR=1) stay on this same pinned emission.
    let ll = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    assert!(
        ll.lines()
            .any(|line| line.ends_with(" = alloca [64 x float], align 64")),
        "default-rewritten matmul4 must tile with a packed align-64 panel:\n{ll}"
    );

    for opt in ["-O0", "-O2"] {
        let (_dir, exe) = compile_exe(&clang, &ll, "default_rewritten_matmul4", opt);
        let default = run_exe(&exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("default_rewritten_matmul4/{opt}: timed out"));
        let one = run_exe(&exe, TIMEOUT_SECS, Some("1"))
            .unwrap_or_else(|| panic!("default_rewritten_matmul4/{opt}/MAPAL_PAR=1: timed out"));
        assert_eq!(default.1, 0, "default_rewritten_matmul4/{opt}: exit");
        assert_eq!(
            String::from_utf8_lossy(&default.0),
            rr.output,
            "default_rewritten_matmul4/{opt}: oracle stdout"
        );
        assert_eq!(
            default, one,
            "default_rewritten_matmul4/{opt}: MAPAL_PAR=1 parity"
        );
    }
}
