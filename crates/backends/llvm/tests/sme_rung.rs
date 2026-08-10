//! The ARM SME rung's gate, at the text level (plan-s41 step 3).
//!
//! The realization is one file (`src/func/sme.rs`) selected at one point
//! (`src/func/tile.rs`), so what needs pinning is not its internals but its
//! **gate**: it fires exactly on a profile with a streaming matrix unit, on the
//! contract face, on a matmul-shaped f32 site whose panels are whole — and
//! nowhere else. Every negative here is a byte-identity claim in disguise: the
//! same source under any other profile, or under the exact face, must emit what
//! it emitted before the rung existed.
//!
//! The positive assertions are the three facts that cost a SIGILL each to
//! learn on the hardware (`benches/sme/README.md`), so they are asserted as
//! text rather than trusted: `aarch64_pstate_sm_body` (never `_sm_enabled`),
//! literal `splat (i1 true)` predicates, and the `mopa` intrinsic itself.

use mapal_backend_llvm::{EmitOpts, emit_with_opts};
use mapal_ir::CategoryIr;

/// The attention shape at 32×32 — two CHAINED matmul-shaped sites.
/// `benches/shapes/attn_16.mapal` is the same shape at the oracle size; this
/// one is exactly ONE panel of the 2×2 tile block (32 = 2 × 16 in both axes),
/// which is the smallest shape the rung still takes.
const ATTN_32: &str = r#"
fn main() {
    1024 -> iota -> tq;
    tq -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> q;
    tq -> map { t -> (t * 5 + 29) % 101 - 50 -> widen_f32 } -> kt;
    tq -> map { t -> (t * 3 + 41) % 101 - 50 -> widen_f32 } -> v;
    32 -> iota -> kr;
    tq -> map { t ->
        t / 32 -> i;
        t % 32 -> j;
        (0.0, kr) -> fold { acc, k -> acc + q[i * 32 + k] * kt[k * 32 + j] }
    } -> s;
    tq -> map { t ->
        t / 32 -> i;
        t % 32 -> j;
        (0.0, kr) -> fold { acc, k -> acc + s[i * 32 + k] * v[k * 32 + j] }
    } -> o;
    o[0] -> println;
    o[1023] -> println;
}
"#;

/// The same shape at 24 — a recognized matmul site whose panels are NOT whole
/// at a 16-wide tile. The remainder is a real shape and this rung does not
/// build it, so the site must fall back, silently and byte-identically.
const ATTN_24: &str = r#"
fn main() {
    576 -> iota -> tq;
    tq -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> q;
    tq -> map { t -> (t * 5 + 29) % 101 - 50 -> widen_f32 } -> kt;
    tq -> map { t -> (t * 3 + 41) % 101 - 50 -> widen_f32 } -> v;
    24 -> iota -> kr;
    tq -> map { t ->
        t / 24 -> i;
        t % 24 -> j;
        (0.0, kr) -> fold { acc, k -> acc + q[i * 24 + k] * kt[k * 24 + j] }
    } -> s;
    tq -> map { t ->
        t / 24 -> i;
        t % 24 -> j;
        (0.0, kr) -> fold { acc, k -> acc + s[i * 24 + k] * v[k * 24 + j] }
    } -> o;
    o[0] -> println;
    o[575] -> println;
}
"#;

/// The same shape at 16 — one ZA tile, but only a QUARTER of the 2×2 panel.
/// It fired before the tile block was derived and it does not now: the price of
/// spending all four tiles is that the whole-panel clause quantises to `ti·t`,
/// so 16 joins 24 on the fallback path. Pinned so the trade is recorded rather
/// than discovered.
const ATTN_16: &str = r#"
fn main() {
    256 -> iota -> tq;
    tq -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> q;
    tq -> map { t -> (t * 5 + 29) % 101 - 50 -> widen_f32 } -> kt;
    tq -> map { t -> (t * 3 + 41) % 101 - 50 -> widen_f32 } -> v;
    16 -> iota -> kr;
    tq -> map { t ->
        t / 16 -> i;
        t % 16 -> j;
        (0.0, kr) -> fold { acc, k -> acc + q[i * 16 + k] * kt[k * 16 + j] }
    } -> s;
    tq -> map { t ->
        t / 16 -> i;
        t % 16 -> j;
        (0.0, kr) -> fold { acc, k -> acc + s[i * 16 + k] * v[k * 16 + j] }
    } -> o;
    o[0] -> println;
    o[255] -> println;
}
"#;

/// The f64 twin of [`ATTN_32`]: same recognized shape, wrong element width.
/// The rung is f32-only by deliberate scope — `benches/sme/run16.c` is what
/// verified it, and it is f32.
const ATTN_32_F64: &str = r#"
fn main() {
    1024 -> iota -> tq;
    tq -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f64 } -> q;
    tq -> map { t -> (t * 5 + 29) % 101 - 50 -> widen_f64 } -> kt;
    tq -> map { t -> (t * 3 + 41) % 101 - 50 -> widen_f64 } -> v;
    32 -> iota -> kr;
    tq -> map { t ->
        t / 32 -> i;
        t % 32 -> j;
        (0.0, kr) -> fold { acc, k -> acc + q[i * 32 + k] * kt[k * 32 + j] }
    } -> s;
    tq -> map { t ->
        t / 32 -> i;
        t % 32 -> j;
        (0.0, kr) -> fold { acc, k -> acc + s[i * 32 + k] * v[k * 32 + j] }
    } -> o;
    o[0] -> println;
    o[1023] -> println;
}
"#;

/// Every text the SME rung — and only the SME rung — can put in a module.
const SME_MARKERS: &[&str] = &[
    "aarch64_new_za",
    "aarch64_pstate_sm_body",
    "llvm.aarch64.sme.zero",
    "llvm.aarch64.sme.mopa.nxv4f32",
    "llvm.aarch64.sme.read.horiz.nxv4f32",
    "splat (i1 true)",
    "@mapal_sme_panel",
    "vscale x 4 x float",
];

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

/// Emission of `src` under `target`, at the face `contract` selects. The
/// rewrite is what the shipped harnesses run (`benches/shapes/shapes_ab.sh`),
/// and it is what makes a matmul site recognizable in the first place.
fn emit_at(src: &str, target: &'static str, contract: bool) -> String {
    let ir = mapal_rewrite::rewrite(lower_src(src)).ir;
    emit_with_opts(
        &ir,
        &EmitOpts {
            contract,
            target,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn assert_no_sme(ll: &str, why: &str) {
    for marker in SME_MARKERS {
        assert!(
            !ll.contains(marker),
            "{why}: found `{marker}` in the module"
        );
    }
}

/// The positive: on `apple-m4-sme`, on the contract face, a whole-panel f32
/// matmul site becomes streaming panel calls against the verified kernel.
#[test]
fn sme_rung_emits_the_verified_kernel() {
    let ll = emit_at(ATTN_32, "apple-m4-sme", true);
    for marker in SME_MARKERS {
        assert!(ll.contains(marker), "missing `{marker}`:\n{ll}");
    }
    // The attribute that pushes the streaming-mode transition onto callers.
    // With it the program SIGILLs; `_body` is what makes the kernel
    // self-contained, and nothing else in the module knows streaming mode
    // exists.
    assert!(
        !ll.contains("aarch64_pstate_sm_enabled"),
        "sm_enabled pushes the mode transition onto callers and SIGILLs"
    );
    // This part has SME without full SVE — the emitted feature set must not
    // ask for SVE, or the prologue faults before `smstart`.
    assert!(ll.contains("\"target-features\"=\"+sme,+sme2,+neon,+fp-armv8,+v8a\""));
    // Two chained sites, one kernel: the geometry that varies between them
    // (`bn`, `cn`, `K`) is passed, not baked.
    assert_eq!(
        ll.matches("define internal void @mapal_sme_panel").count(),
        1,
        "one panel kernel per module"
    );
    assert_eq!(
        ll.matches("call void @mapal_sme_panel").count(),
        2,
        "one call per recognized site"
    );
    // The A panel staged for one i-panel: `ti · t × k` = 32 × 32.
    assert!(
        ll.contains("alloca [1024 x float], align 64"),
        "the A-panel pack scratch is ti*t*k elements:\n{ll}"
    );
    // The declarations follow the emitted call, exactly like the arena block.
    assert!(ll.contains("declare void @llvm.aarch64.sme.zero(i32 immarg)"));
}

/// The defect this test exists to keep fixed: `profile.rs` records **4** f32 ZA
/// tiles and the emitter must spend all of them. `TargetProfile::sme_block`
/// derives the 2×2 arrangement; the kernel then issues four `fmopa`s into four
/// distinct tiles from `ti + tj = 4` loads, and reads all four back.
///
/// Measured payoff, `benches/sme/mm4.c` (f32, 1 thread, min-of-7, values
/// identical): 423 → 777 GFLOP/s at 1024², 237 → 619 at 2048².
#[test]
fn sme_kernel_spends_every_za_tile_the_profile_records() {
    let ll = emit_at(ATTN_32, "apple-m4-sme", true);
    let kernel = ll
        .split_once("define internal void @mapal_sme_panel")
        .expect("the kernel is emitted")
        .1;
    for tile in 0..4 {
        assert!(
            kernel.contains(&format!("mopa.nxv4f32(i32 {tile},")),
            "ZA tile {tile} never accumulated — the profile records 4:\n{kernel}"
        );
        assert!(
            kernel.contains(&format!("i32 {tile}, i32 %r32)")),
            "ZA tile {tile} never read back:\n{kernel}"
        );
    }
    assert_eq!(
        kernel.matches("mopa.nxv4f32").count(),
        4,
        "ti*tj outer products per k"
    );
    // 4 operand loads feeding 4 MACs — the square-most split's whole point.
    // 1x4 would be 5 loads for the same 4 MACs.
    assert_eq!(
        kernel.matches("load <vscale x 4 x float>").count(),
        4,
        "ti + tj operand loads per k"
    );
}

/// The negative that carries the byte-identity claim: no profile that existed
/// before the rung can select it, whatever the face.
#[test]
fn sme_rung_is_unreachable_from_every_other_profile() {
    for target in ["generic", "apple-m", "zen3", "cuda-ada"] {
        assert_no_sme(&emit_at(ATTN_32, target, true), target);
        assert_no_sme(&emit_at(ATTN_32, target, false), target);
    }
}

/// ADR-0032 D1/D3: `fmopa` fuses (measured — 92/256 cells differ against
/// separate mul+add, 0/256 against `fmaf`), so the rung is a contract-face
/// realization. Under the default exact face the NEON rung must run.
#[test]
fn sme_rung_is_a_contract_face_realization() {
    assert_no_sme(&emit_at(ATTN_32, "apple-m4-sme", false), "exact face");
}

/// `apple-m4-sme` is `apple-m` plus TWO declared capabilities — the SME unit,
/// and (plan-s50 P3) the measured dispatch economics. Wherever the SME rung does
/// not fire, the only thing that may differ is the dispatch threshold's verdict,
/// which is the `kind` argument of a task registration and nothing else.
///
/// So the assertion normalises exactly that one field and demands byte identity
/// everywhere else. A change to the SME rung that leaked into a declining shape
/// still fails; so does a threshold that moved anything other than a `kind`.
/// Four ways of not firing: the exact face, two partial panels, and f64.
#[test]
fn sme_profile_differs_from_apple_m_only_in_dispatch_kind_where_the_rung_declines() {
    // A task registration is
    // `mapal_par_task(ptr %h, i32 IDX, i32 KIND, ptr @task…`, so this touches
    // the kind and can touch nothing else.
    fn normalize_kind(ll: &str) -> String {
        ll.replace(", i32 1, ptr @task", ", i32 0, ptr @task")
    }
    for (src, contract, why) in [
        (ATTN_32, false, "exact face"),
        (ATTN_24, true, "24 is not a whole 32-wide panel"),
        (ATTN_16, true, "16 is one tile, not the 2x2 block"),
        (ATTN_32_F64, true, "f64 is out of scope"),
    ] {
        let sme = emit_at(src, "apple-m4-sme", contract);
        assert_no_sme(&sme, why);
        assert_eq!(
            normalize_kind(&sme),
            normalize_kind(&emit_at(src, "apple-m", contract)),
            "{why}"
        );
    }
}

/// The other half of the claim above, stated positively: on these shapes the
/// difference is REAL and it is the threshold. Every one of them is a 16-32
/// wide attention — far under the 30 us a dispatch costs on the measured part —
/// so `apple-m4-sme` must inline what `apple-m`, which has no measured
/// economics, still hands to the pool.
#[test]
fn the_dispatch_threshold_is_what_separates_the_two_profiles_here() {
    let sme = emit_at(ATTN_32, "apple-m4-sme", true);
    let plain = emit_at(ATTN_32, "apple-m", true);
    let kinds = |ll: &str| {
        ll.lines()
            .filter(|l| l.contains("mapal_par_task(ptr %h"))
            .filter(|l| l.contains(", ptr @task"))
            .filter(|l| l.contains(", i32 1, ptr @task"))
            .count()
    };
    assert!(
        kinds(&plain) > kinds(&sme),
        "a part with measured dispatch costs must dispatch strictly fewer of \
         these tiny tasks: apple-m has {} split registrations, apple-m4-sme {}",
        kinds(&plain),
        kinds(&sme)
    );
}

/// A matmul-shaped SME site whose `k` is `K`, at 32×32 outputs — so the A pack
/// buffer is `32 · K · 4` bytes and `K` alone decides whether it fits L1D.
fn pack_shape(k: u64) -> String {
    format!(
        r#"
fn main() {{
    1024 -> iota -> tout;
    {ab} -> iota -> tab;
    tab -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    tab -> map {{ t -> (t * 5 + 29) % 101 - 50 -> widen_f32 }} -> b;
    {k} -> iota -> kr;
    tout -> map {{ t ->
        t / 32 -> i;
        t % 32 -> j;
        (0.0, kr) -> fold {{ acc, k -> acc + a[i * {k} + k] * b[k * 32 + j] }}
    }} -> o;
    o[0] -> println;
    o[1023] -> println;
}}
"#,
        ab = 32 * k,
        k = k
    )
}

/// plan-s51 H2 — the A pack is blocked exactly when its buffer stops fitting L1D.
///
/// **The pack's waste is on the STORE side.** Since S42 the pack is row-outer /
/// k-inner, so reads walk a row contiguously; the stores go to
/// `ap[pk·32 + pi]`, which puts consecutive `pk` one 128 B line apart and writes
/// 4 bytes of each. The next row returns for the next 4. That is free while `ap`
/// is still resident and costs 32× the line traffic once it is not, so the whole
/// decision is `32 · k · sizeof > l1d_bytes` — 1024 is the last `k` that fits.
///
/// Measured on the loops the emitter actually emits
/// (`benches/sme/packconflict.c`): flat 1.000–1.051× at `ap` ≤ 64 KB, **2.752×
/// at 256 KB**, 1.771× at 512 KB. Integrated on `nn4_2048`, **1.092× at one
/// thread, disjoint**.
///
/// Both halves are asserted, because only the pair is a gate: a change that
/// blocked unconditionally would still pass the positive assertion alone, and
/// blocking below the boundary is a measured LOSS of nothing plus real code.
#[test]
fn the_a_pack_is_blocked_exactly_when_it_stops_fitting_l1d() {
    // k = 1024 -> ap = 131072 = L1D exactly, so it still fits and must not block.
    let fits = emit_at(&pack_shape(1024), "apple-m4-sme", true);
    // k = 1056 -> ap = 135168 > L1D.
    let over = emit_at(&pack_shape(1056), "apple-m4-sme", true);

    // Both must actually be taking the SME rung, or the test proves nothing
    // about the pack. (The rung's own markers are asserted elsewhere; here it is
    // enough that the kernel is present.)
    for (ll, why) in [(&fits, "k=1024"), (&over, "k=1056")] {
        assert!(
            ll.contains("@mapal_sme_panel"),
            "{why}: the SME rung did not fire, so this test is vacuous"
        );
    }

    // The block loop's fingerprint is its clamp: the final block is ragged, so
    // the emitted bound is `select over, kc, pk0 + block` with `kc` literal.
    assert!(
        over.contains(", i64 1056, i64 %"),
        "k=1056: ap is 132 KB against a 128 KB L1D and the pack must be blocked \
         — no clamp against the k bound found"
    );
    assert!(
        !fits.contains(", i64 1024, i64 %"),
        "k=1024: ap is exactly L1D, so the pack still fits and must emit the \
         single-pass copy — found a block clamp"
    );
}
