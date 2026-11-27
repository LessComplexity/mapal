//! Golden IR for the tiled map shapes: matmul, kc, 1d and the contract flags.

use super::*;

#[test]
fn golden_tile_map_shapes() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_SRC)).ir;
    // ADR-0038 D4: shape assertions bind to a named profile, never the ambient
    // default — every constant below (TJ=16, TI=4, the [64 x float] scratch) is
    // `generic`'s NEON geometry, and a detected AVX2/AVX-512 profile tiles
    // differently.
    let tiled = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let tiled_fn = function_containing(&tiled, " = alloca [64 x float]");
    // Rows run in TI=4 register blocks, so the accumulator is one flat
    // entry-block scratch of TI×TJ elems (subrow r at r*16 + lane).
    assert!(
        tiled_fn
            .lines()
            .any(|line| line.trim_start().starts_with("%s")
                && line.contains(" = alloca [64 x float]")),
        "tile accumulator must be one flat [TI=4 x TJ=16] entry-block scratch alloca:\n{tiled_fn}"
    );
    assert!(
        !tiled_fn.contains(" = call float @fn"),
        "tiled map must not call its per-cell body:\n{tiled_fn}"
    );
    // Row bounds: the row lo plus the interior full-window row end — the i
    // loop splits into boundary head rows / TI-blocked full-window interior /
    // tail rows.
    assert!(
        tiled_fn.contains(" = udiv i64 %lo, 4")
            && tiled_fn.contains(" = add i64 %hi, 3")
            && tiled_fn.contains(" = add i64 %lo, 3")
            && tiled_fn.contains(" = udiv i64 %hi, 4"),
        "tile nest must contain the row bounds, incl. the interior full-window end:\n{tiled_fn}"
    );
    // plan-s30: the jt-outer main body is the vector-accumulator path — its
    // TI=4 seed/lane/store loops are replaced by `<TJ x float>` phis, so NO
    // lane loop is bounded by the constant TJ=16 any more. The loops that
    // remain (remainder tile, boundary rows) are all runtime bounded.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = icmp uge i64 %t") && line.ends_with(", 16")
            })
            .count(),
        0,
        "the main tile must have no constant-TJ lane loop left:\n{tiled_fn}"
    );
    // The main tile's accumulator state lives only in SSA: 4 header phis + 4
    // exit phis, one contiguous b vector load per k step (×2 unrolled + the
    // odd tail) and one vector store per subrow — no accumulator load, store
    // or GEP anywhere in the k loop.
    assert_eq!(
        (
            tiled_fn.matches("phi <16 x float>").count(),
            tiled_fn.matches("load <16 x float>").count(),
            tiled_fn.matches("store <16 x float>").count(),
        ),
        (8, 3, 4),
        "main tile: 4 header + 4 exit phis, 3 b loads, 4 subrow stores:\n{tiled_fn}"
    );
    // Composition rule 3: vector memory ops carry the ELEMENT alignment, never
    // the `<16 x float>` ABI's 64 — `j0` offsets are arbitrary.
    assert!(
        tiled_fn
            .lines()
            .filter(|line| line.contains("<16 x float>") && line.contains(", ptr "))
            .all(|line| line.trim_end().ends_with(", align 4")),
        "vector accumulator memory must be element-aligned:\n{tiled_fn}"
    );
    // The acc scratch is now addressed only by the tiles that kept the memory
    // form (remainder + boundary rows); it was 44 GEPs before S30.
    assert_eq!(
        tiled_fn
            .matches("getelementptr [64 x float], ptr %s0")
            .count(),
        24,
        "only the non-vector tiles may address the acc scratch:\n{tiled_fn}"
    );
    // The outer main and remainder bodies each compute their packed-panel
    // base once, then reuse it across head/interior/tail row groups.
    assert_eq!(
        (
            tiled_fn
                .lines()
                .filter(|line| line.contains(" = udiv i64 ") && line.trim_end().ends_with(", 16"))
                .count(),
            tiled_fn
                .lines()
                .filter(|line| line.contains(" = mul i64 ") && line.trim_end().ends_with(", 64"))
                .count(),
        ),
        (2, 2),
        "jt-outer kernel must compute one panel base in each main/remainder body:\n{tiled_fn}"
    );
    // One runtime `tj = min(remaining, TJ)` select belongs to the outer
    // remainder tile. Boundary rows retain signed jw clipping in both outer
    // bodies.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = select i1 ") && line.ends_with(", i64 16")
            })
            .count(),
        1,
        "jt-outer remainder must clip once to tj = min(remaining, TJ):\n{tiled_fn}"
    );
    assert_eq!(
        (
            tiled_fn.matches(" = icmp slt i64 ").count(),
            tiled_fn.matches(" = icmp sgt i64 ").count(),
        ),
        (4, 4),
        "head/tail rows must retain signed jw clipping in both jt bodies:\n{tiled_fn}"
    );
    assert!(
        tiled.contains(" = alloca [64 x float], align 64")
            && tiled.contains("store float zeroinitializer")
            && tiled.contains("getelementptr [64 x float], ptr %packed"),
        "module must zero-pad and address the packed b buffer:\n{tiled}"
    );
    assert!(
        tiled_fn.contains("call void @llvm.prefetch.p0(")
            && tiled_fn
                .lines()
                .any(|line| line.contains(" = add i64 ") && line.ends_with(", 2")),
        "tile nest must prefetch and unroll k:\n{tiled_fn}"
    );
    insta::assert_snapshot!("tile_nest_shape", tiled_fn);

    let untiled = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            tiling: false,
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let untiled_fn = function_containing(&untiled, " = call float @fn");
    assert!(
        !untiled_fn
            .lines()
            .any(|line| line.trim_start().starts_with("%s")
                && line.contains(" = alloca [16 x float]")),
        "untiled map must not allocate the tile accumulator:\n{untiled_fn}"
    );
    assert!(
        untiled_fn.contains(" = call float @fn"),
        "untiled map must retain its per-cell body call:\n{untiled_fn}"
    );
    insta::assert_snapshot!("untiled_map_shape", untiled_fn);

    let f64_ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_F64_SRC)).ir;
    let f64 = emit_with_opts(
        &f64_ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let f64_fn = function_containing(&f64, " = alloca [32 x double]");
    assert!(
        f64_fn.contains(" = alloca [32 x double]")
            && f64.contains(" = alloca [120 x double], align 64")
            && f64.contains("store double zeroinitializer")
            && f64.contains("getelementptr [120 x double], ptr %packed")
            && f64_fn
                .lines()
                .any(|line| line.contains(" = add i64 ") && line.ends_with(", 8"))
            && f64_fn.contains("call void @llvm.prefetch.p0("),
        "f64 tile nest must use TJ=8 throughout packing and the kernel:\n{f64_fn}"
    );
    assert_eq!(
        (
            f64_fn
                .lines()
                .filter(|line| line.contains(" = udiv i64 ") && line.trim_end().ends_with(", 8"))
                .count(),
            f64_fn
                .lines()
                .filter(|line| line.contains(" = mul i64 ") && line.trim_end().ends_with(", 40"))
                .count(),
        ),
        (2, 2),
        "f64 jt-outer kernel must compute one panel base in each main/remainder body:\n{f64_fn}"
    );
    insta::assert_snapshot!("tile_nest_shape_f64", f64_fn);
}

/// S29 KC rung golden: a packed site with K=300 > TILE_KC=128 takes the KC
/// nest — jb blocks of NC=512 lanes outer, k-panels of 128 next (the peeled
/// kc==0 panel, the [128, K) loop with the runtime-short `k_hi = min(kc+128,
/// K)` last panel), the existing head/interior/tail i regions innermost, and
/// the a-panel pack per (i-block, kc). The acc discipline pins: seed splat
/// only in the kc==0 sweep (12 subrow seed loops: interior main+remainder
/// trios ×4 subrows + head/tail boundary trios ×1), an `out` spill at EVERY
/// panel end (2 sweeps × 12 stores) and a reload only in the post-kc0 sweep
/// (12 loads). Sites with K ≤ TILE_KC keep the jt-outer nest byte-for-byte
/// (the tile_nest_shape / tile_nest_shape_f64 goldens above are unmoved).
#[test]
fn golden_tile_map_shape_kc() {
    let src = KC_SRC;
    let ir = mapal_rewrite::rewrite(lower_src(src)).ir;
    // The KC nest is a default-OFF performance tailor (EmitOpts::kc_nest —
    // measured a 3x loss locally at 1024 f32, S29); opt in to pin its shape.
    let tiled = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            kc_nest: true,
            // S46: the default target now DETECTS the machine, and a detected L2
            // changes the derived `tile_kc` that decides whether this nest fires.
            // This golden pins the nest's SHAPE, so it pins the reference profile.
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let tiled_fn = function_containing(&tiled, " = alloca [512 x float], align 64");
    // The KC accumulator: one [TI=4 x TJ=16] entry-block scratch, the same
    // width as the jt-outer nest's — partial sums park in `out` at every panel
    // end (the (jc, kc, ic) order runs other i-blocks in between), so only the
    // j-tile being computed is ever live.
    assert!(
        tiled_fn
            .lines()
            .any(|line| line.trim_start().starts_with("%s")
                && line.contains(" = alloca [64 x float]")),
        "KC acc must be one [TI=4 x TJ=16] entry-block scratch alloca:\n{tiled_fn}"
    );
    // The a-panel pack scratch: [TI=4 x TILE_KC=128], 64-aligned.
    assert!(
        tiled_fn
            .lines()
            .any(|line| line.trim_start().starts_with("%s")
                && line.contains(" = alloca [512 x float], align 64")),
        "KC a-panel pack must be an align-64 [TI=4 x KC=128] scratch alloca:\n{tiled_fn}"
    );
    assert!(
        !tiled_fn.contains(" = call float @fn"),
        "tiled map must not call its per-cell body:\n{tiled_fn}"
    );
    // The jb loop (NC=512 lane blocks): the block-end add and the loop step.
    assert!(
        tiled_fn
            .lines()
            .filter(|line| line.contains(" = add i64 ") && line.trim_end().ends_with(", 512"))
            .count()
            >= 2,
        "the jb block must step NC=512 (block end + loop step):\n{tiled_fn}"
    );
    // The jb block end clips jb_end = min(jb0 + NC, C=32): 1 select, plus
    // the four boundary-row jw_hi clips (head/tail × the two kc sweeps)
    // sharing the C literal.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = select i1 ") && line.contains(", i64 32, i64 ")
            })
            .count(),
        1 + 4,
        "the jb block end must clip jb_end = min(jb0 + NC, C):\n{tiled_fn}"
    );
    // The kc loop init with the literal TILE_KC=128 and the runtime-short
    // last panel's k_hi = min(kc + 128, K=300) select.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| line.trim_start().starts_with("store i64 128, ptr"))
            .count(),
        1,
        "the kc loop must init at TILE_KC=128:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = select i1 ") && line.contains(", i64 300, i64 ")
            })
            .count(),
        1,
        "the last kc panel must clip k_hi = min(kc + TILE_KC, K):\n{tiled_fn}"
    );
    // The acc discipline: seed splat only in the peeled kc==0 sweep — 8 scalar
    // subrow seed loops (remainder + boundary trios) now that the two main
    // trios splat into a vector instead; an out spill at every panel end and a
    // reload only in the post-kc0 sweep — 32 out-array GEPs total (the main
    // trios' 12 scalar lane-loop GEPs collapse to 4 vector ones per sweep).
    assert_eq!(
        tiled_fn.matches("store float 0x0000000000000000").count(),
        8,
        "seed splat belongs to the kc==0 sweep only:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches("getelementptr [256 x float]").count(),
        32,
        "2 spills + 1 reload per (i-block, j-tile) across the two sweeps:\n{tiled_fn}"
    );
    // plan-s30, the point of the rung: the KC k loop carries its accumulators
    // in phis, so nothing of the acc tile round-trips memory between panels.
    // Both main trios (the peeled kc==0 sweep and the [128, K) sweep) get 4
    // header + 4 exit phis; the 10 vector loads are 3 packed-b loads per trio
    // plus the post-kc0 sweep's 4 partial-sum reloads; the 8 vector stores are
    // the two sweeps' panel-end parks.
    assert_eq!(
        (
            tiled_fn.matches("phi <16 x float>").count(),
            tiled_fn.matches("load <16 x float>").count(),
            tiled_fn.matches("store <16 x float>").count(),
        ),
        (16, 10, 8),
        "KC main tiles must carry vector accumulators across the k loop:\n{tiled_fn}"
    );
    assert!(
        tiled_fn
            .lines()
            .filter(|line| line.contains("<16 x float>") && line.contains(", ptr "))
            .all(|line| line.trim_end().ends_with(", align 4")),
        "vector accumulator memory must be element-aligned:\n{tiled_fn}"
    );
    // The acc scratch is addressed only by the tiles that kept the memory form
    // (remainder + boundary rows); it was 88 GEPs before S30.
    assert_eq!(
        tiled_fn
            .matches("getelementptr [64 x float], ptr %s0")
            .count(),
        48,
        "only the non-vector tiles may address the acc scratch:\n{tiled_fn}"
    );
    // The kernel keeps the packed-b panel addressing, the ×2 k unroll, and
    // the next-k-line prefetch.
    assert!(
        tiled_fn.contains("call void @llvm.prefetch.p0(")
            && tiled_fn
                .lines()
                .any(|line| line.contains(" = add i64 ") && line.ends_with(", 2"))
            && tiled_fn.contains("getelementptr [9600 x float], ptr %packed"),
        "KC kernel must keep the packed panel, k unroll, and prefetch:\n{tiled_fn}"
    );
    insta::assert_snapshot!("tile_nest_shape_kc", tiled_fn);
}

#[test]
fn tile_contract_flags_are_opt_in() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_SRC)).ir;
    let plain = emit_with_opts(&ir, &EmitOpts::exact()).unwrap();
    let contracted = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            contract: true,
            ..EmitOpts::exact()
        },
    )
    .unwrap();

    assert!(!plain.contains("fmul contract"));
    assert!(!plain.contains("fadd contract"));
    assert!(contracted.contains("fmul contract float"));
    assert!(contracted.contains("fadd contract float"));
    assert!(
        !contracted.contains("reassoc"),
        "a sequential order never reassociates"
    );
}

/// ADR-0040 D1/D4: the default face's tiled steps carry `reassoc contract`
/// (the fold's order is free, a step fuses) and nothing else — no `fast`,
/// `nsz`, `afn`, `arcp`, `ninf`, `nnan`.
#[test]
fn default_face_flags_the_tiled_steps() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_SRC)).ir;
    let opts = EmitOpts {
        target: "generic",
        ..EmitOpts::default()
    };
    let ll = emit_with_opts(&ir, &opts).unwrap();
    assert!(ll.contains("fmul reassoc contract float"));
    assert!(ll.contains("fadd reassoc contract float"));
    for f in [" fast ", " nsz ", " afn ", " arcp ", " ninf ", " nnan "] {
        assert!(!ll.contains(f), "{f}");
    }
    let free = EmitOpts {
        contract: false,
        ..opts
    };
    let ll = emit_with_opts(&ir, &free).unwrap();
    assert!(ll.contains("fmul reassoc float") && !ll.contains("contract"));
}

#[test]
fn golden_tile_map_shape_1d() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_FIR_SRC)).ir;
    // ADR-0038 D4: the TJ=16 lane bounds below are `generic`'s geometry.
    let tiled = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let tiled_fn = function_containing(&tiled, " = alloca [64 x float]");
    // The S28 window rung: TI=4 register blocks over the lane axis, so the
    // accumulator is one flat entry-block scratch of TI×TJ elems (subrow r at
    // r*16 + lane) — the rung-2 placement, one row deep.
    assert!(
        tiled_fn
            .lines()
            .any(|line| line.trim_start().starts_with("%s")
                && line.contains(" = alloca [64 x float]")),
        "FIR tile accumulator must be one flat [TI=4 x TJ=16] entry-block scratch alloca:\n{tiled_fn}"
    );
    assert!(
        !tiled_fn.contains(" = call float @fn"),
        "tiled FIR map must not call its per-sample body:\n{tiled_fn}"
    );
    // Full blocks step TI·TJ = 64 lanes (`jb + 64 <= hi`), entered from the
    // task window lo — no row loop, no [0, C) clip (rows == 1 makes the task
    // range the whole window).
    assert!(
        !tiled_fn.contains(" = udiv i64 %lo, 16") && !tiled_fn.contains(" = add i64 %hi, 15"),
        "window nest must drop the collapsed row loop:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| line.contains(" = add i64 ") && line.trim_end().ends_with(", 64"))
            .count(),
        1,
        "full blocks must step TI·TJ = 64:\n{tiled_fn}"
    );
    // The block body: per-subrow seed splat (4), the ×2-unrolled k body (K=4
    // even: two steps of 4 subrow lane loops), the single-k tail step (4),
    // and per-subrow stores (4) — every lane loop constant-bounded by TJ.
    // The TI=1 remainder adds one constant-TJ main trio (seed + k-lane +
    // store = 3).
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = icmp uge i64 %t") && line.ends_with(", 16")
            })
            .count(),
        4 + 2 * 4 + 4 + 4 + 3,
        "block + remainder-main lane loops must be bounded by the constant TJ=16:\n{tiled_fn}"
    );
    // ONE runtime `tj = min(remaining, TJ)` select in the whole nest — the
    // TI=1 remainder tile. Full blocks never select.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = select i1 ") && line.ends_with(", i64 16")
            })
            .count(),
        1,
        "only the TI=1 remainder tile may clip to tj = min(remaining, TJ):\n{tiled_fn}"
    );
    // The shared read: ONE scalar w[k] load per emitted k step (three in the
    // block path — the ×2 pair plus the tail step — two in the TI=1
    // remainder), never one per (k, subrow): 5 w-load GEPs feed 14 FMA lane
    // loops (3 block steps × TI=4 subrows + 2 remainder k loops).
    assert_eq!(
        tiled_fn.matches("getelementptr [4 x float]").count(),
        5,
        "one scalar a load per k step, shared across the TI subrows:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches(" = fmul float ").count(),
        14,
        "3 block k-steps x TI=4 subrows + 2 remainder k-loops of FMAs:\n{tiled_fn}"
    );
    // The block k loop is ×2-unrolled (K=4 even) in the trio's shape: one
    // `kk + 2` step. The TI=1 remainder keeps the plain single-k loop (two
    // `k >= 4` heads — main tile and remainder tile).
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| line.contains(" = add i64 ") && line.ends_with(", 2"))
            .count(),
        1,
        "the block k loop must unroll x2:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = icmp uge i64 %t") && line.ends_with(", 4")
            })
            .count(),
        2,
        "the TI=1 remainder must keep the plain single-k loop:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches(" = icmp ule i64 ").count(),
        2,
        "the full-block guard plus the TI=1 main-tile guard:\n{tiled_fn}"
    );
    insta::assert_snapshot!("tile_nest_shape_1d", tiled_fn);
}
