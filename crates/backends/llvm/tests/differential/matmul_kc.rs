//! Tiled matmul differentials on the kc (k-panel) paths.

use super::*;

/// S29 KC rung: K=200 crosses TILE_KC=128 — the packed nest splits into the
/// peeled kc==0 panel (seed) plus one runtime-short loop panel [128, 200)
/// (K % KC = 72 ≠ 0). 32×32 keeps one runtime-short jb block (C=32 < NC=512,
/// C % TJ = 0 so no j remainder). Byte-equal vs untiled + oracle at -O0/-O2.
#[test]
fn differential_tiled_matmul_kc_32x32x200() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    6400 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    6400 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    200 -> iota -> ks;
    1024 -> iota -> cells;
    cells -> map { cell ->
        cell / 32 -> i;
        cell % 32 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 200 + k] * b[k * 32 + j] }
    } -> c;
    c[0] -> println;
    c[1023] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("kc 32x32x200 oracle completes") else {
        panic!("kc 32x32x200 oracle must complete");
    };
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
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
    assert!(
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = alloca [512 x float], align 64"),
        "K > TILE_KC must take the KC nest: TI×TJ acc (partials park in `out`) \
         + TI×KC a-panel pack:\n{tiled}"
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

    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_kc_32x32x200");
}

/// S29 KC rung, multi-panel: K=300 gives the peeled kc==0 panel, one FULL
/// middle panel [128, 256) through the kc loop, and the runtime-short last
/// panel [256, 300) (K % KC = 44) — the reload/spill acc parking across
/// three panels. Byte-equal vs untiled + oracle at -O0/-O2.
#[test]
fn differential_tiled_matmul_kc_middle_panel() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    2400 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    9600 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    300 -> iota -> ks;
    256 -> iota -> cells;
    cells -> map { cell ->
        cell / 32 -> i;
        cell % 32 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 300 + k] * b[k * 32 + j] }
    } -> c;
    c[0] -> println;
    c[255] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("kc middle-panel oracle completes") else {
        panic!("kc middle-panel oracle must complete");
    };
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
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
    assert!(
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = alloca [512 x float], align 64"),
        "K=300 must take the KC nest:\n{tiled}"
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

    assert_tiled_parity(
        &clang,
        &tiled,
        &untiled,
        &want,
        "tiled_matmul_kc_middle_panel",
    );
}

/// S29 KC rung, C % NC ≠ 0 (no split): rows=4, C=540, K=136 (K % KC = 8 —
/// one peeled panel + one short loop panel). C = 512 + 28: jb block 0 is the
/// full 512 lanes; block 1 runs one constant-TJ main tile [512, 528) plus the
/// runtime `tj = 12` remainder — both tile kinds inside the runtime-short
/// block. rows=4 = one TI interior block, single slice (n < GRAIN).
#[test]
fn differential_tiled_matmul_kc_c540() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    544 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    73440 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    136 -> iota -> ks;
    2160 -> iota -> cells;
    cells -> map { cell ->
        cell / 540 -> i;
        cell % 540 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 136 + k] * b[k * 540 + j] }
    } -> c;
    c[0] -> println;
    c[1023] -> println;
    c[2159] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("kc c540 oracle completes") else {
        panic!("kc c540 oracle must complete");
    };
    let (tiled, untiled) = emit_kc_and_untiled(&ir);
    assert!(
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = alloca [512 x float], align 64"),
        "C % NC case must contain the KC nest:\n{tiled}"
    );
    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_kc_c540");
}

/// S29 KC rung, split coverage: rows=31, C=136, K=136 (K % KC = 8 ≠ 0 — one
/// peeled panel + one short loop panel). n = 4216 > GRAIN ⇒ 2 slices at the
/// default pool and MAPAL_PAR=2; the boundary 2108 lands mid-row at j=68 —
/// mid-jb-block (the single block spans [0, 136)), clipping a partial first
/// tile (panel_lane0 = 4). rows % 4 = 3 keeps the TI=1 tail region live;
/// MAPAL_PAR=1 is the single full range. The gate is tiled == untiled native
/// byte-parity (the tile_ab discipline): a 2-slice case needs n > GRAIN, so
/// with K > TILE_KC the fold-step count necessarily exceeds the interp's
/// 10M-step budget — the oracle leg is carried by the four smaller KC cases.
#[test]
fn differential_tiled_matmul_kc_split() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    4216 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    18496 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    136 -> iota -> ks;
    4216 -> iota -> cells;
    cells -> map { cell ->
        cell / 136 -> i;
        cell % 136 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 136 + k] * b[k * 136 + j] }
    } -> c;
    c[0] -> println;
    c[2107] -> println;
    c[2108] -> println;
    c[4215] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let (tiled, untiled) = emit_kc_and_untiled(&ir);
    assert!(
        tiled.contains(" = alloca [64 x float]")
            && tiled.contains(" = alloca [512 x float], align 64"),
        "split case must contain the KC nest:\n{tiled}"
    );
    assert!(
        untiled.contains(" = call float @fn"),
        "untiled map must retain its body call:\n{untiled}"
    );

    for opt in ["-O0", "-O2"] {
        let tag_o = format!("tiled_matmul_kc_split/{opt}");
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
            assert_eq!(tiled_run.0, untiled_run.0, "{tag_p}: tiled/untiled stdout");
        }
    }
}

/// S29 KC rung, f64 width: TJ=8 ⇒ NC=256; C=20 keeps one short jb block (two
/// constant-TJ main tiles + the runtime `tj = 4` remainder), K=200 crosses
/// TILE_KC (K % KC = 72), rows=6 keeps the two-row TI tail. Packed vs
/// --no-pack byte parity (the R1 attribution control) plus the untiled +
/// oracle legs at -O0/-O2.
#[test]
fn differential_tiled_matmul_kc_f64() {
    let Some(clang) = clang() else {
        return;
    };
    let src = r#"
fn main() {
    1200 -> iota -> ta;
    ta -> map { t -> (t * 5 + 7) % 67 - 33 -> widen<f64> } -> a;
    4000 -> iota -> tb;
    tb -> map { t -> (t * 7 + 11) % 101 - 50 -> widen<f64> } -> b;
    200 -> iota -> ks;
    120 -> iota -> cells;
    cells -> map { cell ->
        cell / 20 -> i;
        cell % 20 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 200 + k] * b[k * 20 + j] }
    } -> c;
    c[0] -> println;
    c[80] -> println;
    c[119] -> println;
}
"#;
    let ir = rewrite(lower_src(src)).ir;
    let rr = run(&ir, BUDGET);
    let (Some(want), 0) = expect_native(&ir, &rr).expect("kc f64 oracle completes") else {
        panic!("kc f64 oracle must complete");
    };
    let tiled = mapal_backend_llvm::emit_with_opts(
        &ir,
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
    assert!(
        tiled.contains(" = alloca [32 x double]")
            && tiled.contains(" = alloca [512 x double], align 64"),
        "f64 KC nest must use TI×TJ=32 acc + TI×KC=512 apack:\n{tiled}"
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

    assert_tiled_parity(&clang, &tiled, &untiled, &want, "tiled_matmul_kc_f64");
    for opt in ["-O0", "-O2"] {
        let packed_run = compile_run(
            &clang,
            &tiled,
            &format!("tiled_matmul_kc_f64/{opt}/packed"),
            opt,
        )
        .unwrap_or_else(|| panic!("tiled_matmul_kc_f64/{opt}/packed: timed out"));
        let nopack_run = compile_run(
            &clang,
            &nopack,
            &format!("tiled_matmul_kc_f64/{opt}/nopack"),
            opt,
        )
        .unwrap_or_else(|| panic!("tiled_matmul_kc_f64/{opt}/nopack: timed out"));
        assert_eq!(
            packed_run, nopack_run,
            "tiled_matmul_kc_f64/{opt}: --no-pack parity"
        );
    }
}
