//! Tiled FIR differentials: plain, remainder and split.

use super::*;

#[test]
fn differential_tiled_fir() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    71 -> iota -> tx;
    tx -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> x;
    8 -> iota -> kr;
    kr -> map { t -> (t * 5 + 3) % 31 - 15 -> widen<f32> } -> w;
    64 -> iota -> ts;
    ts -> map { t ->
        (0.0, kr) -> fold { acc, k -> acc + w[k] * x[t + k] }
    } -> y;
    y[0] -> println;
    y[63] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("FIR oracle completes") else {
        panic!("FIR oracle must complete");
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
            && tiled
                .lines()
                .any(|line| line.contains(" = add i64 ") && line.trim_end().ends_with(", 64"))
            && !tiled.contains(" = udiv i64 %lo, 64"),
        "split task must contain the S28 window nest: TI blocks stepping TI·TJ over the lane axis:\n{tiled}"
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
        let tiled_run = compile_run(&clang, &tiled, &format!("tiled_fir/{opt}"), opt)
            .unwrap_or_else(|| panic!("tiled_fir/{opt}: tiled run timed out"));
        let untiled_run = compile_run(&clang, &untiled, &format!("untiled_fir/{opt}"), opt)
            .unwrap_or_else(|| panic!("tiled_fir/{opt}: untiled run timed out"));
        assert_eq!(tiled_run.1, 0, "tiled_fir/{opt}: tiled exit code");
        assert_eq!(untiled_run.1, 0, "tiled_fir/{opt}: untiled exit code");
        assert_eq!(
            String::from_utf8_lossy(&tiled_run.0),
            want,
            "tiled_fir/{opt}: oracle stdout"
        );
        assert_eq!(
            tiled_run.0, untiled_run.0,
            "tiled_fir/{opt}: tiled/untiled stdout"
        );
    }
}

/// S28 window-rung remainder coverage: N=86 = TI·TJ + TJ + 6 — one full TI
/// block, then the TI=1 remainder region runs one constant-TJ main tile plus
/// the runtime `tj = 6` tile. K=7 (odd) exercises the block's plain single-k
/// loop (the ×2 unroll gate is `K % 2 == 0`).
#[test]
fn differential_tiled_fir_remainder() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    92 -> iota -> tx;
    tx -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> x;
    7 -> iota -> kr;
    kr -> map { t -> (t * 5 + 3) % 31 - 15 -> widen<f32> } -> w;
    86 -> iota -> ts;
    ts -> map { t ->
        (0.0, kr) -> fold { acc, k -> acc + w[k] * x[t + k] }
    } -> y;
    y[0] -> println;
    y[64] -> println;
    y[85] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("FIR remainder oracle completes") else {
        panic!("FIR remainder oracle must complete");
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
            && tiled
                .lines()
                .any(|line| line.contains(" = add i64 ") && line.trim_end().ends_with(", 64")),
        "remainder case must contain the window nest's TI·TJ block step:\n{tiled}"
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
    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_fir_remainder");
}

/// S28 window-rung split coverage: N=10000 forces >1 split slice at the
/// default GRAIN=4096 (the harness has no grain knob; MAPAL_PAR pins the pool
/// size, and `slice_ranges` divides the range evenly across slices — not at
/// GRAIN boundaries). MAPAL_PAR=2 gives [0,5000)+[5000,10000): 5000 % 64 = 8,
/// a mid-block boundary — the second slice's window enters the block loop at
/// jb=lo and exits through the TI=1 remainder. MAPAL_PAR=1 is the single full
/// range: 156 full blocks plus a constant-TJ remainder tile.
#[test]
fn differential_tiled_fir_split() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    10007 -> iota -> tx;
    tx -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> x;
    8 -> iota -> kr;
    kr -> map { t -> (t * 5 + 3) % 31 - 15 -> widen<f32> } -> w;
    10000 -> iota -> ts;
    ts -> map { t ->
        (0.0, kr) -> fold { acc, k -> acc + w[k] * x[t + k] }
    } -> y;
    y[0] -> println;
    y[5000] -> println;
    y[9999] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("FIR split oracle completes") else {
        panic!("FIR split oracle must complete");
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
        tiled.contains(" = alloca [64 x float]"),
        "split case must contain the window nest:\n{tiled}"
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
        let tag_o = format!("tiled_fir_split/{opt}");
        let (_tiled_dir, tiled_exe) = compile_exe(&clang, &tiled, &format!("{tag_o}/tiled"), opt);
        let (_untiled_dir, untiled_exe) =
            compile_exe(&clang, &untiled, &format!("{tag_o}/untiled"), opt);
        for par in [None, Some("1"), Some("2")] {
            let tag_p = format!("{tag_o}/MAPAL_PAR={}", par.unwrap_or("default"));
            let tiled_run = run_exe(&tiled_exe, TIMEOUT_SECS, par)
                .unwrap_or_else(|| panic!("{tag_p}: tiled run timed out"));
            let untiled_run = run_exe(&untiled_exe, TIMEOUT_SECS, par)
                .unwrap_or_else(|| panic!("{tag_p}: untiled run timed out"));
            assert_eq!(tiled_run.1, 0, "{tag_p}: tiled exit code");
            assert_eq!(untiled_run.1, 0, "{tag_p}: untiled exit code");
            assert_eq!(
                String::from_utf8_lossy(&tiled_run.0),
                want,
                "{tag_p}: oracle stdout"
            );
            assert_eq!(tiled_run.0, untiled_run.0, "{tag_p}: tiled/untiled stdout");
        }
    }
}
