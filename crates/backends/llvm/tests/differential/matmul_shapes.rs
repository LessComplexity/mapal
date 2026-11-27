//! Tiled matmul differentials over assorted shapes, f64 and the loop-carried pack.

use super::*;

/// S26 coverage gap (a): a j **remainder after a full main tile** plus a
/// `rows % TILE_I` i-tail. rows=5, C=20, K=7: each row runs one constant-16
/// main j-tile then the runtime `tj = 4` remainder; `rows % 4 == 1` sends row
/// 4 through the TI=1 tail path (the 4x4 case above has C < TILE_J, so its
/// main body never executes, and rows=4 leaves no tail).
#[test]
fn differential_tiled_matmul_r5_c20_k7() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,
      10.0, 15.0, 20.0, 25.0, 30.0, -32.0, -27.0, -22.0,
     -17.0, -12.0, -7.0, -2.0, 3.0, 8.0, 13.0, 18.0,
      23.0, 28.0, 33.0, -29.0, -24.0, -19.0, -14.0, -9.0,
      -4.0, 1.0, 6.0] -> a: [f32; 35];
    [-37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0,
     -26.0, -19.0, -12.0, -5.0, 2.0, 9.0, 16.0, 23.0,
      30.0, 37.0, 44.0, -50.0, -43.0, -36.0, -29.0, -22.0,
     -15.0, -8.0, -1.0, 6.0, 13.0, 20.0, 27.0, 34.0,
      41.0, 48.0, -46.0, -39.0, -32.0, -25.0, -18.0, -11.0,
      -4.0, 3.0, 10.0, 17.0, 24.0, 31.0, 38.0, 45.0,
     -49.0, -42.0, -35.0, -28.0, -21.0, -14.0, -7.0, 0.0,
      7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0,
      18.0, 25.0, 32.0, 39.0, 46.0, -48.0, -41.0, -34.0,
     -27.0, -20.0, -13.0, -6.0, 1.0, 8.0, 15.0, 22.0,
      29.0, 36.0, 43.0, 50.0, -44.0, -37.0, -30.0, -23.0,
     -16.0, -9.0, -2.0, 5.0, 12.0, 19.0, 26.0, 33.0,
      40.0, 47.0, -47.0, -40.0, -33.0, -26.0, -19.0, -12.0,
      -5.0, 2.0, 9.0, 16.0, 23.0, 30.0, 37.0, 44.0,
     -50.0, -43.0, -36.0, -29.0, -22.0, -15.0, -8.0, -1.0,
      6.0, 13.0, 20.0, 27.0] -> b: [f32; 140];
    100 -> iota -> cells;
    7 -> iota -> ks;
    cells -> map { cell ->
        cell / 20 -> i;
        cell % 20 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 7 + k] * b[k * 20 + j] }
    } -> c;
    c[0] -> println;
    c[80] -> println;
    c[99] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("matmul r5_c20_k7 oracle completes")
    else {
        panic!("matmul r5_c20_k7 oracle must complete");
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
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = udiv i64 %lo, 20")
            && tiled.contains(" = udiv i64 %hi, 20"),
        "split task must contain the TI-blocked tiled nest:\n{tiled}"
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

    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_r5_c20_k7");
}

/// S26 coverage gap (b): **two full main j-tiles, no remainder** plus a
/// two-row i-tail. rows=6, C=32, K=5: j runs the constant-16 main body twice
/// and the remainder check falls through (`32 % 16 == 0`); `rows % 4 == 2`
/// sends rows 4-5 through the TI=1 tail path.
#[test]
fn differential_tiled_matmul_r6_c32_k5() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,
      10.0, 15.0, 20.0, 25.0, 30.0, -32.0, -27.0, -22.0,
     -17.0, -12.0, -7.0, -2.0, 3.0, 8.0, 13.0, 18.0,
      23.0, 28.0, 33.0, -29.0, -24.0, -19.0] -> a: [f32; 30];
    [-37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0,
     -26.0, -19.0, -12.0, -5.0, 2.0, 9.0, 16.0, 23.0,
      30.0, 37.0, 44.0, -50.0, -43.0, -36.0, -29.0, -22.0,
     -15.0, -8.0, -1.0, 6.0, 13.0, 20.0, 27.0, 34.0,
      41.0, 48.0, -46.0, -39.0, -32.0, -25.0, -18.0, -11.0,
      -4.0, 3.0, 10.0, 17.0, 24.0, 31.0, 38.0, 45.0,
     -49.0, -42.0, -35.0, -28.0, -21.0, -14.0, -7.0, 0.0,
      7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0,
      18.0, 25.0, 32.0, 39.0, 46.0, -48.0, -41.0, -34.0,
     -27.0, -20.0, -13.0, -6.0, 1.0, 8.0, 15.0, 22.0,
      29.0, 36.0, 43.0, 50.0, -44.0, -37.0, -30.0, -23.0,
     -16.0, -9.0, -2.0, 5.0, 12.0, 19.0, 26.0, 33.0,
      40.0, 47.0, -47.0, -40.0, -33.0, -26.0, -19.0, -12.0,
      -5.0, 2.0, 9.0, 16.0, 23.0, 30.0, 37.0, 44.0,
     -50.0, -43.0, -36.0, -29.0, -22.0, -15.0, -8.0, -1.0,
      6.0, 13.0, 20.0, 27.0, 34.0, 41.0, 48.0, -46.0,
     -39.0, -32.0, -25.0, -18.0, -11.0, -4.0, 3.0, 10.0,
      17.0, 24.0, 31.0, 38.0, 45.0, -49.0, -42.0, -35.0] -> b: [f32; 160];
    192 -> iota -> cells;
    5 -> iota -> ks;
    cells -> map { cell ->
        cell / 32 -> i;
        cell % 32 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 5 + k] * b[k * 32 + j] }
    } -> c;
    c[0] -> println;
    c[128] -> println;
    c[191] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("matmul r6_c32_k5 oracle completes")
    else {
        panic!("matmul r6_c32_k5 oracle must complete");
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
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = udiv i64 %lo, 32")
            && tiled.contains(" = udiv i64 %hi, 32"),
        "split task must contain the TI-blocked tiled nest:\n{tiled}"
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

    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_r6_c32_k5");
}

/// S27 f64 width pin: TJ=8 gives two full panels plus a four-lane remainder;
/// K=5 exercises the unrolled pair body and its trailing single-k step, while
/// rows=6 retains the two-row TI tail.
#[test]
fn differential_tiled_matmul_r6_c20_k5_f64() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    30 -> iota -> ais;
    ais -> map { x -> (x * 5 + 7) % 67 - 33 -> widen<f64> } -> a;
    100 -> iota -> bis;
    bis -> map { x -> (x * 7 + 11) % 101 - 50 -> widen<f64> } -> b;
    120 -> iota -> cells;
    5 -> iota -> ks;
    cells -> map { cell ->
        cell / 20 -> i;
        cell % 20 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 5 + k] * b[k * 20 + j] }
    } -> c;
    c[0] -> println;
    c[80] -> println;
    c[119] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("matmul r6_c20_k5_f64 oracle completes")
    else {
        panic!("matmul r6_c20_k5_f64 oracle must complete");
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
        tiled.contains(" = alloca [32 x double]")
            && tiled.contains(" = alloca [120 x double], align 64")
            && tiled.contains("call void @llvm.prefetch.p0("),
        "f64 tiled nest must use TJ=8, packed panels, and prefetch:\n{tiled}"
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
    let nopack = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            packing: false,
            // ADR-0038 D4: the comparison's other leg pins `generic` by name —
            // keep both legs inside the one profile.
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();

    assert_tiled_parity(
        &clang,
        &tiled,
        &untiled,
        &want,
        "tiled_matmul_r6_c20_k5_f64",
    );
    for opt in ["-O0", "-O2"] {
        let packed_run = compile_run(
            &clang,
            &tiled,
            &format!("tiled_matmul_r6_c20_k5_f64/{opt}/packed"),
            opt,
        )
        .unwrap_or_else(|| panic!("tiled_matmul_r6_c20_k5_f64/{opt}/packed: timed out"));
        let nopack_run = compile_run(
            &clang,
            &nopack,
            &format!("tiled_matmul_r6_c20_k5_f64/{opt}/nopack"),
            opt,
        )
        .unwrap_or_else(|| panic!("tiled_matmul_r6_c20_k5_f64/{opt}/nopack: timed out"));
        assert_eq!(
            packed_run, nopack_run,
            "tiled_matmul_r6_c20_k5_f64/{opt}: --no-pack parity"
        );
    }
}

#[test]
fn differential_tiled_matmul_loop_carried_pack() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    [2.0, 0.0, 0.0, 0.0,
     0.0, 2.0, 0.0, 0.0,
     0.0, 0.0, 2.0, 0.0,
     0.0, 0.0, 0.0, 2.0] -> a: [f32; 16];
    [1.0, 2.0, 3.0, 4.0,
     5.0, 6.0, 7.0, 8.0,
     9.0, 10.0, 11.0, 12.0,
     13.0, 14.0, 15.0, 16.0] -> b0: [f32; 16];
    4 -> iota -> kr;
    16 -> iota -> cells;
    mut b: [f32; 16] <- b0;
    mut iteration: i32 <- 0;
    loop {
        (iteration < 2) -> {
            -true-> {
                cells -> map { cell ->
                    cell / 4 -> i;
                    cell % 4 -> j;
                    (0.0, kr) -> fold { acc, k ->
                        acc + a[i * 4 + k] * b[k * 4 + j]
                    }
                } -> c;
                c -> b;
                iteration + 1 -> iteration;
                -> loop;
            }
            -false-> b -> out;
        }
    }
    8 -> iota -> xs;
    xs -> map { x -> x * 3 + 1 } -> extra;
    out[0] -> println;
    out[15] -> println;
    extra[7] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("loop-carried pack oracle completes")
    else {
        panic!("loop-carried pack oracle must complete");
    };
    // ADR-0038 D4: pack-frame and parallel task structure are generic-derived
    // shape — pin the profile by name, never the ambient default.
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    let entry = tiled
        .split("define internal void @mapal_main(")
        .nth(1)
        .and_then(|s| s.split("\n}\n").next())
        .expect("parallel mapal_main");
    assert!(
        entry.contains("call ptr @mapal_par_begin(")
            && entry.contains("call void @mapal_par_launch("),
        "entry must retain a multi-path parallel plan:\n{tiled}"
    );
    let loop_task = tiled
        .split("define internal void @task")
        .skip(1)
        .filter_map(|s| s.split("\n}\n").next())
        .find(|task| task.contains("alloca [64 x float], align 64"))
        .unwrap_or_else(|| panic!("loop task must allocate and pack b inline:\n{tiled}"));
    assert!(
        loop_task.contains("getelementptr [64 x float], ptr") && !tiled.contains("%pack_field"),
        "Seq-loop packing must stay in its task, not the frame:\n{tiled}"
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
    for opt in ["-O0", "-O2"] {
        let (_tiled_dir, tiled_exe) =
            compile_exe(&clang, &tiled, "tiled_matmul_loop_carried_pack", opt);
        let (_untiled_dir, untiled_exe) =
            compile_exe(&clang, &untiled, "untiled_matmul_loop_carried_pack", opt);
        let default = run_exe(&tiled_exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("loop-carried pack/{opt}/default: timed out"));
        let one = run_exe(&tiled_exe, TIMEOUT_SECS, Some("1"))
            .unwrap_or_else(|| panic!("loop-carried pack/{opt}/MAPAL_PAR=1: timed out"));
        let untiled = run_exe(&untiled_exe, TIMEOUT_SECS, None)
            .unwrap_or_else(|| panic!("loop-carried pack/{opt}/untiled: timed out"));
        assert_eq!(default.1, 0, "loop-carried pack/{opt}: exit code");
        assert_eq!(
            String::from_utf8_lossy(&default.0),
            want,
            "loop-carried pack/{opt}: oracle stdout"
        );
        assert_eq!(default, one, "loop-carried pack/{opt}: MAPAL_PAR=1 parity");
        assert_eq!(default, untiled, "loop-carried pack/{opt}: tiled parity");
    }
}
