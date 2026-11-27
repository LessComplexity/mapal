//! Tiled conv2d differentials and the tiled-versus-untiled parity helpers.

use super::*;

/// The conv2d 3×3 shape at output side `side` (img side+2): y[i,j] =
/// Σₖ w[k]·img[(i + k÷3)·(side+2) + j + k%3] — the fold body's `k/3`, `k%3`
/// is the k-split record the S28 conv rung cashes.
pub(crate) fn conv2d_src(side: u64) -> String {
    let img = (side + 2) * (side + 2);
    let n = side * side;
    let stride = side + 2;
    let mid = n / 2 - 1;
    let last = n - 1;
    format!(
        r#"
fn main() {{
    {img} -> iota -> ti;
    ti -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> }} -> img;
    9 -> iota -> kr;
    kr -> map {{ t -> (t * 5 + 3) % 31 - 15 -> widen<f32> }} -> w;
    {n} -> iota -> ts;
    ts -> map {{ t ->
        t / {side} -> i;
        t % {side} -> j;
        (0.0, kr) -> fold {{ acc, k -> acc + w[k] * img[(i + k / 3) * {stride} + j + k % 3] }}
    }} -> y;
    y[0] -> println;
    y[{mid}] -> println;
    y[{last}] -> println;
}}
"#
    )
}

/// The KC-nest twin of [`emit_tiled_and_untiled`]: the k-panel nest is a
/// default-OFF performance tailor (`EmitOpts::kc_nest` — a measured 3x loss
/// locally at 1024 f32, S29), so its tests opt in explicitly. The untiled side
/// is unchanged, which is the point: the nest must be bit-exact against it.
pub(crate) fn emit_kc_and_untiled(ir: &CategoryIr) -> (String, String) {
    let tiled = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            kc_nest: true,
            // S46: the default target now DETECTS the machine, and a detected
            // L2 changes the derived `tile_kc` that decides whether this nest
            // fires at all. These tests assert the nest's SHAPE, so they pin
            // the reference profile by name.
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    let untiled = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            tiling: false,
            // ADR-0038 D4: the comparison's other leg pins `generic` by name —
            // keep both legs inside the one profile.
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    (tiled, untiled)
}

pub(crate) fn emit_tiled_and_untiled(ir: &CategoryIr) -> (String, String) {
    // ADR-0038 D4: shape constants are generic-derived — pin the profile by
    // name, never the ambient default (native since S46).
    let tiled = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    let untiled = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            tiling: false,
            // ADR-0038 D4: the comparison's other leg pins `generic` by name —
            // keep both legs inside the one profile.
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap();
    (tiled, untiled)
}

/// S28 conv rung coverage: side 16 has C % TJ == 0 — one constant-TJ main
/// tile per row, no remainder — the unrolled 9-tap micro-kernel, byte-equal
/// vs the untiled emission and the interp oracle at -O0 and -O2.
#[test]
fn differential_tiled_conv2d() {
    let Some(clang) = clang() else {
        return;
    };
    let ir = rewrite(lower_src(&conv2d_src(16))).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("conv2d oracle completes") else {
        panic!("conv2d oracle must complete");
    };
    let (tiled, untiled) = emit_tiled_and_untiled(&ir);
    assert!(
        tiled.contains(" = alloca [16 x float]"),
        "conv2d must tile into the unrolled tap nest:\n{tiled}"
    );
    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_conv2d");
}

/// side 20: C % TJ != 0 — each row runs one constant-TJ main tile plus the
/// runtime `tj = 4` remainder tile.
#[test]
fn differential_tiled_conv2d_remainder() {
    let Some(clang) = clang() else {
        return;
    };
    let ir = rewrite(lower_src(&conv2d_src(20))).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("conv2d remainder oracle completes")
    else {
        panic!("conv2d remainder oracle must complete");
    };
    let (tiled, untiled) = emit_tiled_and_untiled(&ir);
    assert!(
        tiled.contains(" = alloca [16 x float]"),
        "conv2d remainder case must tile:\n{tiled}"
    );
    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_conv2d_remainder");
}

/// side 92: n = 8464 > 2·GRAIN, so the default pool cuts 3 slices, and
/// `slice_ranges`' even division lands the boundaries mid-row at j=62 and
/// j=31 (both % TJ ≠ 0 — mid-tile). MAPAL_PAR=1 is the single full range;
/// MAPAL_PAR=2 cuts two slices at a row-aligned boundary.
#[test]
fn differential_tiled_conv2d_split() {
    let Some(clang) = clang() else {
        return;
    };
    let ir = rewrite(lower_src(&conv2d_src(92))).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("conv2d split oracle completes") else {
        panic!("conv2d split oracle must complete");
    };
    let (tiled, untiled) = emit_tiled_and_untiled(&ir);
    assert!(
        tiled.contains(" = alloca [16 x float]"),
        "conv2d split case must tile:\n{tiled}"
    );

    for opt in ["-O0", "-O2"] {
        let tag_o = format!("tiled_conv2d_split/{opt}");
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

/// The tiled differential run loop: compile tiled + untiled at -O0 and -O2,
/// run each — exit 0, tiled stdout byte-equal to the interp oracle, and
/// tiled == untiled byte-equal (per-cell bit-exactness is the tile nest's
/// hard invariant).
pub(crate) fn assert_tiled_parity(clang: &str, tiled: &str, untiled: &str, want: &str, tag: &str) {
    for opt in ["-O0", "-O2"] {
        let tag_o = format!("{tag}/{opt}");
        let tiled_run = compile_run(clang, tiled, &format!("{tag_o}/tiled"), opt)
            .unwrap_or_else(|| panic!("{tag_o}/tiled: run timed out"));
        let untiled_run = compile_run(clang, untiled, &format!("{tag_o}/untiled"), opt)
            .unwrap_or_else(|| panic!("{tag_o}/untiled: run timed out"));
        assert_eq!(tiled_run.1, 0, "{tag_o}/tiled: exit code");
        assert_eq!(untiled_run.1, 0, "{tag_o}/untiled: exit code");
        assert_eq!(
            String::from_utf8_lossy(&tiled_run.0),
            want,
            "{tag_o}/tiled: oracle stdout"
        );
        assert_eq!(tiled_run.0, untiled_run.0, "{tag_o}: tiled/untiled stdout");
    }
}
