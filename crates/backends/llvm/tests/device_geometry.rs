//! device-geometry: the model-scale layer's device gaps, as facts and rules.
//!
//! - `tile-post` (`CategoryIr::tile_plan_post`): a sum of products whose map
//!   body goes on past its fold (a residual, a scale, a mask, a per-row
//!   divisor, a number capture) is a tile site whose store evaluates the body
//!   after the fold; a workspace split's reduce launch does where the fold is
//!   cut — OFF arm `--gpu-off=tile-post`;
//! - `head-group` (a fact arm): a mixed-radix site's batch digits (a KV head's
//!   query heads, a sequence) are row digits of `a` and block-uniform row
//!   digits of `b`, so a block's `b` panel serves the whole group — OFF arm
//!   `--gpu-off=head-group`;
//! - `piece-delta` (a fact arm): a per-sequence row append to a carried
//!   `[B][S][W]` state, `((t ÷ W) % S == p) ? new : x[t]`, is piecewise
//!   (`Outside::w`, `Delta::Pieces`) and runs over its pieces in the state's
//!   buffer — OFF arm `--gpu-off=piece-delta`;
//! - `row-tree` (a fact arm): a row fold beside folds that load nothing (a
//!   norm's root) splits over a warp's lanes — OFF arm `--gpu-off=row-tree`.
//!
//! Facts and emitted plans on hand-written programs; values through the GPU
//! emulator against the interpreter (free threads, serialized in thread order
//! and reversed) on generated programs (`testgen::layer_strategy`), with
//! coverage counters.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, Delta, ElemSrc, FactArms, Operation};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// The GPU emission on the rtx-5080 with the rungs `off` off; `place` off so
/// every capable site is on the device, and the emulator's rungs off.
fn device_emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// The `; mapal.plan` lines of the device unit.
fn plans(em: &Emission) -> Vec<&str> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.plan"))
        .collect()
}

/// The entry's maps, in function order.
fn maps(ir: &CategoryIr) -> Vec<mapal_ir::MorphismId> {
    let f = ir.entry();
    ir.func(f)
        .unwrap()
        .morphisms
        .iter()
        .copied()
        .filter(|&m| matches!(ir.morphism(m).unwrap().op, Operation::Map { .. }))
        .collect()
}

const GEMM_RES: &str = "fn main() {
    4096 -> iota -> ra;
    ra -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> a;
    ra -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> b;
    4096 -> iota -> o;
    64 -> iota -> rk;
    o -> map {
        e -> e / 64 -> i;
        e % 64 -> j;
        (0.0, rk) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] } -> y;
        a[e] + y
    } -> c;
    c[0] -> println;
    c[4095] -> println;
    c[2049] -> println;
}
";

/// Finding 6: `fold -> y; a[e] + y` is a tile site with an epilogue — the
/// fact (`tile_plan_post`: the residual's slot read after the fold, the
/// strict plans unchanged), the plan (a staged tile whose store calls the
/// body's `_post` variant), and its OFF arm (the per-cell kernel).
#[test]
fn tile_post_takes_a_residual_after_the_fold() {
    let ir = lower(GEMM_RES);
    let f = ir.entry();
    let post = ir.tile_plan_post(f);
    assert_eq!(post.sites.len(), 1, "one post site");
    let (m, site) = post.sites.iter().next().unwrap();
    let p = site.post.as_ref().expect("a post");
    assert_eq!(p.reads, vec![site.a.slot], "the residual is `a`");
    assert!(p.uniform.is_empty() && !p.guarded);
    assert!(!ir.tile_plan(f).sites.contains_key(m));
    assert!(!ir.tile_plan_strided(f).sites.contains_key(m));
    let on = device_emission(&ir, &[]);
    assert!(
        on.text.contains("device_sites=1"),
        "{}",
        on.text.lines().last().unwrap()
    );
    let dev = on.device.as_deref().unwrap();
    assert!(plans(&on).iter().any(|l| l.contains("kind=tile")));
    assert!(
        dev.contains("_post({"),
        "the store calls the epilogue variant"
    );
    let off = device_emission(&ir, &["tile-post"]);
    assert!(off.text.contains("device_sites=0"));
    assert!(!plans(&off).iter().any(|l| l.contains("kind=tile")));
    if let Some(clang) = emu::clang() {
        let want = interp(&ir);
        for em in [&on, &off] {
            let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
            for mode in MODES {
                assert_eq!(ex.run(&mode).0, want, "{mode:?}");
            }
        }
    }
}

/// A GEMV whose body adds the input after the fold: its candidates split
/// the fold into a workspace whose reduce launch evaluates the epilogue
/// (`kind=reduce … epi=1`) — the one combine that sees whole sums.
#[test]
fn tile_post_gemv_reduces_through_the_epilogue() {
    let src = "fn main() {
    2048 -> iota -> rx;
    rx -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> x;
    4194304 -> iota -> rw;
    rw -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> w;
    2048 -> iota -> o;
    2048 -> iota -> rk;
    o -> map {
        j -> (0.0, rk) -> fold { acc, k -> acc + x[k] * w[k * 2048 + j] } -> y;
        x[j] * 0.5 + y
    } -> c;
    c[0] -> println;
    c[2047] -> println;
    c[1234] -> println;
}
";
    let ir = lower(src);
    let em = device_emission(&ir, &[]);
    let ps = plans(&em);
    assert!(
        ps.iter()
            .any(|l| l.contains("kind=reduce") && l.contains("epi=1")),
        "{ps:#?}"
    );
    // The atomic and stream-K splits never take an epilogue.
    assert!(
        !ps.iter().any(|l| l.contains("epi=1")
            && (l.contains("combine=atomic") || l.contains("combine=streamk")))
    );
}

/// Finding 9: RMSNorm's row — a sum of squares, then a Heron root over a
/// constant range — has the sum as its tree fold (`CellSite::tree`), so its
/// cell takes a warp's lanes; the fact arm off keeps one thread a row.
#[test]
fn row_tree_takes_the_loading_fold_beside_a_root() {
    let src = "fn main() {
    512 -> iota -> rx;
    rx -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> x;
    4 -> iota -> rr;
    128 -> iota -> rw;
    8 -> iota -> rh;
    rr -> map {
        b -> (0.0, rw) -> fold { a1, i -> x[b * 128 + i] -> v; a1 + v * v } -> ss;
        ss / 128.0 + 1.0 -> ra;
        (0.5 + 0.5 * ra, rh) -> fold { a3, k -> 0.5 * (a3 + ra / a3) } -> sd;
        1.0 / sd
    } -> rinv;
    rinv[0] -> println;
    rinv[3] -> println;
}
";
    let ir = lower(src);
    let m = maps(&ir)[1];
    let cell = ir.cell_plan(ir.entry()).get(m).cloned().expect("a cell");
    assert!(cell.tree.is_some(), "the sum of squares is the tree");
    let off = ir.refacted(FactArms {
        row_tree: false,
        ..FactArms::default()
    });
    let moff = maps(&off)[1];
    assert!(off.cell_plan(off.entry()).get(moff).unwrap().tree.is_none());
    let on = device_emission(&ir, &[]);
    assert!(plans(&on).iter().any(|l| l.contains("lanes=32")));
    let offe = device_emission(&ir, &["row-tree"]);
    assert!(!plans(&offe).iter().any(|l| l.contains("lanes=")));
}

/// A post site with one row a `b` panel over a k-major `b` (a decode step's
/// masked scores, `k0[r * 256 + k]`) stays the per-cell kernel — a warp a
/// cell reads `b` along `k` coalesced and skips the masked cells — while the
/// same row over a lane-major `b` (`v0[r * 256 + j]`, P·V) is a post tile;
/// a number a whole fold makes, read by value, keeps the cell too.
#[test]
fn tile_post_leaves_a_lone_k_major_row_to_the_cell() {
    let src = "fn main() {
    262144 -> iota -> ik;
    ik -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> k0;
    ik -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> v0;
    256 -> iota -> rk;
    rk -> map { t -> (t * 7 + 2) % 9 - 4 -> widen<f32> } -> q;
    1024 -> iota -> rr;
    0 -> z0: i32;
    (z0, rk) -> fold { acc, v -> acc + v % 3 } -> pp;
    pp % 1024 -> pos;
    rr -> map {
        r -> (r <= pos) -> {
            -true-> {
                (0.0, rk) -> fold { acc, k -> acc + q[k] * k0[r * 256 + k] } -> dt;
                dt * 0.5
            }
            -false-> -1000.0;
        }
    } -> s;
    rk -> map {
        j -> (0.0, rr) -> fold { acc, r -> acc + s[r] * v0[r * 256 + j] } -> a;
        a * 0.25 + q[j]
    } -> o;
    s[0] -> println;
    o[0] -> println;
    o[255] -> println;
}
";
    let ir = lower(src);
    assert_eq!(
        ir.tile_plan_post(ir.entry()).sites.len(),
        2,
        "both are post facts"
    );
    let on = device_emission(&ir, &[]);
    let ps = plans(&on);
    let post: Vec<_> = ps.iter().filter(|l| l.contains(" post=1")).collect();
    assert!(
        post.iter().all(|l| l.contains(" c=256 ")),
        "only P·V: {post:#?}"
    );
    assert!(!post.is_empty(), "{ps:#?}");
    assert!(
        ps.iter()
            .any(|l| l.contains("kind=cell") && l.contains("lanes=")),
        "the scores a lane cell: {ps:#?}"
    );
    if let Some(clang) = emu::clang() {
        let want = interp(&ir);
        let ex = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap_or(""));
        for mode in MODES {
            assert_eq!(ex.run(&mode).0, want, "{mode:?}");
        }
    }
}

const APPEND: &str = "fn main() {
    1536 -> iota -> i0;
    i0 -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> kc0;
    48 -> iota -> rbw;
    rbw -> map { t -> (t * 5 + 2) % 7 - 3 -> widen<f32> } -> nw;
    1536 -> iota -> rc;
    mut kc: [f32; 1536] <- kc0;
    mut acc: [f32; 48] <- nw;
    mut t: i32 <- 0;
    loop {
        (t < 3) -> {
            -true-> {
                2 + t -> p;
                rbw -> map { e -> acc[e] + 1.0 } -> acc;
                rc -> map {
                    c -> c / 512 -> bb;
                    c / 16 % 32 -> r;
                    (r == p) -> {
                        -true-> acc[bb * 16 + c % 16] * 2.0;
                        -false-> kc[c];
                    }
                } -> kc;
                t + 1 -> t;
                -> loop;
            }
            -false-> kc -> out;
        }
    }
    out[0] -> println;
    out[1535] -> println;
    out[33] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold { a2, v -> a2 + v } -> total;
    total -> println;
}
";

/// Finding 8: a row append to every sequence of a carried `[3][32][16]`
/// state is pieces of 16 cells every 512 (`Outside::w`), its carried write
/// `Delta::Pieces`, and on the device a range cell over the 3 pieces in the
/// state's buffer; the arm off rewrites the whole state.
#[test]
fn piece_delta_sees_a_per_sequence_append() {
    let ir = lower(APPEND);
    let f = ir.entry();
    let pieces: Vec<_> = ir
        .write_delta(f)
        .values()
        .filter_map(|d| match *d {
            Delta::Pieces { w, step, start, .. } => Some((w, step, start.a, start.b)),
            _ => None,
        })
        .collect();
    assert_eq!(pieces, vec![(16, 512, 16, 32)], "rows 2 + t of 16 cells");
    let outside = ir.funcs().flat_map(|(g, d)| {
        d.morphisms
            .iter()
            .filter_map(|&m| match ir.map_law(m) {
                Some(ElemSrc::Apply {
                    outside: Some(ou), ..
                }) => Some(ou),
                _ => None,
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(move |o| (g, o))
    });
    assert!(outside.into_iter().any(|(_, o)| o.w == 16 && o.step == 512));
    let off = ir.refacted(FactArms {
        piece_delta: false,
        ..FactArms::default()
    });
    assert!(
        off.write_delta(off.entry())
            .values()
            .all(|d| !matches!(d, Delta::Pieces { .. }))
    );
    let on = device_emission(&ir, &[]);
    assert!(
        plans(&on).iter().any(|l| l.contains("piece=16")),
        "{:#?}",
        plans(&on)
    );
    let offe = device_emission(&ir, &["piece-delta"]);
    assert!(!plans(&offe).iter().any(|l| l.contains("piece=")));
    if let Some(clang) = emu::clang() {
        let want = interp(&ir);
        for em in [&on, &offe] {
            let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
            for mode in MODES {
                assert_eq!(ex.run(&mode).0, want, "{mode:?}");
            }
        }
    }
}

const GQA: &str = "fn main() {
    2048 -> iota -> iq;
    iq -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> q;
    65536 -> iota -> ik;
    ik -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> kc;
    32 -> iota -> rd;
    0 -> z0: i32;
    (z0, rd) -> fold { acc, v -> acc + v % 3 } -> pp;
    pp % 128 -> pos;
    8192 -> iota -> o;
    o -> map {
        e -> e / 4096 -> bb;
        e / 128 % 32 -> hh;
        e % 128 -> r;
        (r <= pos) -> {
            -true-> {
                (0.0, rd) -> fold {
                    acc, u -> acc + q[(bb * 32 + hh) * 32 + u] * kc[((bb * 128 + r) * 8 + hh / 4) * 32 + u]
                } -> dt;
                dt * 0.5
            }
            -false-> -1000.0;
        }
    } -> s;
    s[0] -> println;
    s[8191] -> println;
    s[4097] -> println;
}
";

/// Finding 7: QKᵀ over grouped query heads (4 per KV head, a masked
/// position the program computes) is a mixed-radix tile site whose batch
/// digits (the sequence, the KV head) are block-uniform row digits of `K`:
/// one block per group of 4 query heads (`tm=4 ty=1`), the mask an epilogue
/// with a uniform; the arm off leaves the per-cell kernel.
#[test]
fn head_group_tiles_grouped_query_heads() {
    let ir = lower(GQA);
    let f = ir.entry();
    let radix = ir.tile_plan_radix(f);
    let (_, site) = radix
        .iter()
        .find(|(_, s)| s.post.is_some())
        .expect("a radix post site");
    let rb = site.radix.as_ref().unwrap().b.as_ref().expect("b digits");
    assert_eq!(
        rb.i.iter().map(|d| d.stride).min(),
        Some(4),
        "4 query heads a group"
    );
    let p = site.post.as_ref().unwrap();
    assert!(p.guarded && p.uniform.len() == 1, "{p:?}");
    let off = ir.refacted(FactArms {
        head_group: false,
        ..FactArms::default()
    });
    assert!(
        off.tile_plan_radix(off.entry())
            .values()
            .all(|s| s.post.is_none())
    );
    let on = device_emission(&ir, &[]);
    let ps = plans(&on);
    assert!(
        ps.iter().any(|l| l.contains("kind=tile")
            && l.contains("radix=")
            && l.contains(" tm=4 ")
            && l.contains(" ty=1 ")),
        "{ps:#?}"
    );
    let dev = on.device.as_deref().unwrap();
    assert!(dev.contains(", i64 %u0)"), "the mask position by value");
    let offe = device_emission(&ir, &["head-group"]);
    assert!(!plans(&offe).iter().any(|l| l.contains("radix=")));
    if let Some(clang) = emu::clang() {
        let want = interp(&ir);
        for em in [&on, &offe] {
            let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
            for mode in MODES {
                assert_eq!(ex.run(&mode).0, want, "{mode:?}");
            }
        }
    }
}

/// One generated program: the facts' and plans' coverage, and its values on
/// the device (every mode) and the host against the interpreter; its kind's
/// OFF arm's values on a third of them.
fn check_geom(clang: &str, i: usize, p: &testgen::LayerProg) -> Vec<(&'static str, u64)> {
    let src = testgen::layer_prog_src(p);
    let ir = lower(&src);
    let want = interp(&ir);
    let f = ir.entry();
    let mut c = vec![("programs", 1)];
    for (_, s) in ir.tile_plan_post(f).sites.iter() {
        c.push(("post-facts", 1));
        let post = s.post.as_ref().unwrap();
        if post.guarded {
            c.push(("post-guarded-facts", 1));
        }
    }
    for (_, s) in ir
        .tile_plan_radix(f)
        .iter()
        .filter(|(_, s)| s.post.is_some())
    {
        if s.radix
            .as_ref()
            .and_then(|r| r.b.as_ref())
            .is_some_and(|b| !b.i.is_empty())
        {
            c.push(("group-facts", 1));
        }
    }
    if ir
        .write_delta(f)
        .values()
        .any(|d| matches!(d, Delta::Pieces { .. }))
    {
        c.push(("piece-facts", 1));
    }
    if ir.cell_plan(f).values().any(|s| s.tree.is_some()) {
        c.push(("tree-facts", 1));
    }
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let ps = plans(&em);
    if dev.contains("_post({") {
        c.push(("post-kernels", 1));
    }
    if ps
        .iter()
        .any(|l| l.contains("kind=reduce") && l.contains("epi=1"))
    {
        c.push(("post-reduce", 1));
    }
    if dev.contains(", i64 %u0)") {
        c.push(("post-uniform", 1));
    }
    if ps
        .iter()
        .any(|l| l.contains("radix=") && l.contains(" ty=1 "))
    {
        c.push(("group-tiles", 1));
    }
    // A digit-affine output (`radix=…out`: a query token above the heads),
    // and its fold cut into a workspace (the reduce stores at the address).
    let outs: Vec<&&str> = ps
        .iter()
        .filter(|l| l.contains(" post=1") && l.contains("radix=") && l.contains("out"))
        .collect();
    if !outs.is_empty() {
        c.push(("group-out-tiles", 1));
    }
    if outs.iter().any(|l| l.contains("combine=splitk")) {
        c.push(("group-out-splits", 1));
    }
    if ps.iter().any(|l| l.contains("piece=")) {
        c.push(("piece-cells", 1));
    }
    if ps
        .iter()
        .any(|l| l.contains("kind=cell") && l.contains("lanes="))
    {
        c.push(("lane-rows", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    // Every site-table candidate in turn (`MAPAL_GPU_PICK`, clamped per
    // site): a split's reduce, a group tile's staged variants — each one a
    // tuner may pick.
    let cands = ps
        .iter()
        .filter(|l| l.contains("kind=tile") || l.contains("kind=naive"))
        .count();
    let picks: Vec<String> = (1..cands.min(9)).map(|k| k.to_string()).collect();
    let envs: Vec<[(&str, &str); 1]> = picks
        .iter()
        .map(|k| [("MAPAL_GPU_PICK", k.as_str())])
        .collect();
    for (pick, (got, _)) in picks.iter().zip(ex.run_each(&envs)) {
        assert_eq!(got, want, "pick {pick}\n{src}");
        c.push(("candidate-runs", 1));
    }
    let cpu = emu::run_emulated(
        clang,
        &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
        "",
    );
    assert_eq!(cpu, want, "cpu\n{src}");
    if i % 3 == 0 {
        let arm = match p.kind {
            0 => "tile-post",
            1 => "head-group",
            2 => "piece-delta",
            _ => "row-tree",
        };
        let off = device_emission(&ir, &[arm]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "{arm} off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// Generated programs (`testgen::layer_strategy`): epilogue GEMMs (a
/// residual, a column bias, a mask around the fold, a per-row divisor, a
/// number capture, a select, a mixed form; GEMVs among them), grouped-head
/// QKᵀ and P·V (1–2 sequences and KV heads, 1, 2 or 4 query heads a group,
/// masks at a uniform position or the head), per-sequence row appends in a
/// loop, and row norms with one or two local folds beside the row's sum.
/// Every reach counter is non-zero.
#[test]
fn device_geometry_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::LayerProg> = std::iter::repeat_with(|| {
        testgen::layer_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(96)
    .collect();
    let results = testgen::par_check(&progs, |i, p| check_geom(&clang, i, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("device-geometry generated: {c:?}");
    for key in [
        "programs",
        "post-facts",
        "post-guarded-facts",
        "post-kernels",
        "post-reduce",
        "post-uniform",
        "group-facts",
        "group-tiles",
        "piece-facts",
        "piece-cells",
        "tree-facts",
        "lane-rows",
        "off-arm-runs",
        "candidate-runs",
        "group-out-tiles",
        "group-out-splits",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// long-prefill `uniform-base`: the 8192-token prefill's loop over KV-head
/// groups, small. The counter `g` slices the group's query heads and KV head
/// out of `q` / `kr` and its weight panel out of `wo`: each map is a
/// mixed-radix tile site whose operands' pointers the launch moves by `g`
/// (`ua` / `ub`), in bounds at `g`'s top where the site runs (`g < 2`).
const SLICED: &str = "fn main() {
    16384 -> iota -> i_q;
    i_q -> map { t -> (t * 5 + 3) % 7 - 3 -> widen<f32> } -> q;
    4096 -> iota -> i_k;
    i_k -> map { t -> (t * 3 + 1) % 5 - 2 -> widen<f32> } -> kr;
    16384 -> iota -> i_w;
    i_w -> map { t -> (t * 3 + 2) % 5 - 2 -> widen<f32> } -> wo;
    2048 -> iota -> y0;
    y0 -> map { t -> t % 3 -> widen<f32> } -> yi;
    32 -> iota -> rd;
    16384 -> iota -> rs;
    256 -> iota -> rgd;
    2048 -> iota -> ry;
    mut y: [f32; 2048] <- yi;
    mut g: i32 <- 0;
    loop {
        (g < 2) -> {
            -true-> {
                rs -> map {
                    e -> e / 4096 -> hh;
                    e / 64 % 64 -> i;
                    e % 64 -> r;
                    (0.0, rd) -> fold {
                        acc, u -> acc + q[i * 256 + (g * 4 + hh) * 32 + u] * kr[r * 64 + g * 32 + u]
                    } -> dt;
                    dt * 0.5
                } -> s;
                ry -> map {
                    e -> e / 32 -> i;
                    e % 32 -> j;
                    (0.0, rgd) -> fold {
                        acc, k -> acc + s[(k / 64 * 64 + i) * 64 + k % 64] * wo[(g * 256 + k) * 32 + j]
                    } -> yo;
                    y[e] + yo
                } -> y;
                g + 1 -> g;
                -> loop;
            }
            -false-> y -> out;
        }
    }
    out[0] -> println;
    out[2047] -> println;
}
";

#[test]
fn uniform_base_moves_the_operands_by_the_loop_counter() {
    let ir = lower(SLICED);
    let f = ir.entry();
    let sites: Vec<mapal_ir::TileSite> = ir.tile_plan_radix(f).values().cloned().collect();
    let moved: Vec<(Vec<mapal_ir::Digit>, Vec<mapal_ir::Digit>)> = sites
        .iter()
        .filter_map(|s| s.radix.as_deref())
        .filter(|r| r.uniform())
        .map(|r| (r.ua.clone(), r.ub.clone()))
        .collect();
    assert_eq!(moved.len(), 2, "{sites:#?}");
    // QKᵀ: `q` by `g·4·32` (four query heads a group), `kr` by `g·32`
    // (its KV head); the projection: `wo` by `g·256·32` (its panel).
    let coefs: Vec<(Vec<u64>, Vec<u64>)> = moved
        .iter()
        .map(|(a, b)| {
            let c = |ds: &[mapal_ir::Digit]| ds.iter().map(|d| d.coef).collect::<Vec<_>>();
            (c(a), c(b))
        })
        .collect();
    assert!(coefs.contains(&(vec![128], vec![32])), "{coefs:?}");
    assert!(coefs.contains(&(vec![], vec![8192])), "{coefs:?}");
    let got = |ds: &[mapal_ir::Digit], g: u64| mapal_ir::uniform_at(ds, |_| g);
    let qk = moved.iter().find(|m| !m.0.is_empty()).unwrap();
    assert_eq!(got(&qk.0, 1) + got(&qk.1, 1), 160);
    // OFF arm: no form, a per-cell kernel (the facts before).
    let off = ir.refacted(FactArms {
        uniform_base: false,
        ..FactArms::default()
    });
    assert!(
        off.tile_plan_radix(off.entry())
            .values()
            .all(|s| !s.radix.as_deref().is_some_and(|r| r.uniform()))
    );
    let on = device_emission(&ir, &[]);
    let ps = plans(&on);
    assert!(
        ps.iter().filter(|l| l.contains(" ubase=")).count() >= 2,
        "{ps:#?}"
    );
    let want = interp(&ir);
    if let Some(clang) = emu::clang() {
        let ex = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap_or(""));
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            assert_eq!(got, want, "{mode:?}");
        }
        let offe = device_emission(&ir, &["uniform-base"]);
        assert!(plans(&offe).iter().all(|l| !l.contains(" ubase=")));
        let got = emu::run_emulated(&clang, &offe.text, offe.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "uniform-base off");
    }
}

/// A counter whose range where the site runs (`g < 3`) puts the weight panel
/// past `wo` at its top has no uniform form (nor do wrapped reads).
#[test]
fn uniform_base_refuses_a_slice_past_the_array() {
    let src = SLICED
        .replace("(g < 2) -> {", "(g < 3) -> {")
        .replace(
            "acc, u -> acc + q[i * 256 + (g * 4 + hh) * 32 + u] * kr[r * 64 + g * 32 + u]",
            "acc, u -> acc + q[(i * 256 + (g * 4 + hh) * 32 + u) % 16384] * kr[(r * 64 + g * 32 + u) % 4096]",
        );
    let ir = lower(&src);
    let f = ir.entry();
    for s in ir.tile_plan_radix(f).values() {
        // The projection's panel runs past `wo` at `g = 2`: refused; the
        // wrapped reads are no digit form at all.
        assert!(!s.radix.as_deref().is_some_and(|r| r.uniform()), "{s:#?}");
    }
}

/// Does a plan line move a narrow operand's pointer (`narrow=b:bf16 …
/// ubase=a,b`)?
fn narrow_uniform(ps: &[&str]) -> bool {
    let field = |l: &str, k: &str| -> String {
        l.split(' ')
            .find_map(|w| w.strip_prefix(k))
            .unwrap_or("")
            .to_string()
    };
    ps.iter().any(|l| {
        let ub = field(l, "ubase=");
        field(l, "narrow=")
            .split('+')
            .filter_map(|x| x.split(':').next())
            .any(|o| !o.is_empty() && ub.split(',').any(|u| u == o))
    })
}

/// One generated sliced program ([`testgen::SliceProg`]): the facts' and
/// plans' coverage, and its values on the device (every mode, every site
/// candidate) and the host against the interpreter; the OFF arm's on a third.
fn check_slice(clang: &str, i: usize, p: &testgen::SliceProg) -> Vec<(&'static str, u64)> {
    let src = testgen::slice_prog_src(p);
    let ir = lower(&src);
    let want = interp(&ir);
    let f = ir.entry();
    let mut c = vec![("programs", 1)];
    let mut facts = 0;
    for s in ir.tile_plan_radix(f).values() {
        if s.radix.as_deref().is_some_and(|r| r.uniform()) {
            facts += 1;
            c.push(("uniform-facts", 1));
            c.push(match s.post {
                Some(_) => ("uniform-post-facts", 1),
                None => ("uniform-plain-facts", 1),
            });
        }
    }
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let ps = plans(&em);
    let tiles: Vec<&&str> = ps.iter().filter(|l| l.contains(" ubase=")).collect();
    if !tiles.is_empty() {
        c.push(("uniform-tiles", 1));
    }
    if tiles.iter().any(|l| l.contains(" post=1")) {
        c.push(("uniform-post-tiles", 1));
    }
    if tiles.iter().any(|l| l.contains("combine=splitk")) {
        c.push(("uniform-splits", 1));
    }
    if facts > 0 && tiles.is_empty() {
        c.push(("uniform-refused", 1));
    }
    // ADR-0043 under `uniform-base`: a narrow (`bf16`) operand whose pointer
    // the counter moves — by its own 2-byte elements.
    let narrow_moved = narrow_uniform(&ps);
    if narrow_moved {
        c.push(("narrow-uniform-tiles", 1));
    }
    if em.text.contains("; mapal.take") {
        c.push(("carry-takes", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    let cands = ps
        .iter()
        .filter(|l| l.contains("kind=tile") || l.contains("kind=naive"))
        .count();
    let picks: Vec<String> = (1..cands.min(9)).map(|k| k.to_string()).collect();
    let envs: Vec<[(&str, &str); 1]> = picks
        .iter()
        .map(|k| [("MAPAL_GPU_PICK", k.as_str())])
        .collect();
    for (pick, (got, _)) in picks.iter().zip(ex.run_each(&envs)) {
        assert_eq!(got, want, "pick {pick}\n{src}");
        c.push(("candidate-runs", 1));
    }
    let cpu = emu::run_emulated(
        clang,
        &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
        "",
    );
    assert_eq!(cpu, want, "cpu\n{src}");
    if i % 3 == 0 {
        let off = device_emission(&ir, &["uniform-base", "carry-take"]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "uniform-base, carry-take off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    // The tile launch path (`resident` off): each operand copied from where
    // its terms move it, its bytes its own storage's.
    if i % 3 == 1 || narrow_moved {
        let off = device_emission(&ir, &["resident"]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "resident off\n{src}");
        let offp = plans(&off);
        if offp.iter().any(|l| l.contains(" ubase=")) {
            c.push(("launch-path-uniform", 1));
        }
        if narrow_uniform(&offp) {
            c.push(("launch-path-narrow", 1));
        }
    }
    c
}

/// Generated sliced programs (`testgen::slice_strategy`): 1–4 KV groups a
/// loop's trips, 1, 2 or 4 query heads a group, QKᵀ or P·V (plain, masked
/// where the counter moves the mask, scaled, per-row divisors), K/V shared or
/// per sequence, stored `f32` or `bf16` (the narrow operand's pointer moved in
/// its own 2-byte elements, on the resident and the tile launch path), head
/// widths 3–32, the group's projection into a carried `y`. Every reach counter
/// is non-zero.
#[test]
fn uniform_base_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::SliceProg> = std::iter::repeat_with(|| {
        testgen::slice_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(64)
    .collect();
    let results = testgen::par_check(&progs, |i, p| check_slice(&clang, i, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("uniform-base generated: {c:?}");
    for key in [
        "programs",
        "uniform-facts",
        "uniform-post-facts",
        "uniform-plain-facts",
        "uniform-tiles",
        "uniform-post-tiles",
        "uniform-splits",
        "uniform-refused",
        "narrow-uniform-tiles",
        "launch-path-uniform",
        "launch-path-narrow",
        "carry-takes",
        "candidate-runs",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// long-prefill `carry-take`: the carried `y`'s init `yi` is host-built and
/// already on the device (a row fold reads it before the loop); its device
/// copy dies at the loop's entry, so the carry starts in that buffer — one
/// upload fewer than the OFF arm, the values the interpreter's either way.
#[test]
fn carry_take_starts_the_carry_in_the_inits_device_copy() {
    let src = SLICED.replace(
        "    2048 -> iota -> ry;\n",
        "    2048 -> iota -> ry;\n    512 -> iota -> rr;\n    rr -> map { b -> (0.0, rd) -> fold { acc, i -> acc + yi[b * 4 + i % 4] } } -> y2;\n    y2[5] -> println;\n",
    );
    let ir = lower(&src);
    let h2d = |em: &Emission| em.text.matches("call void @mapal_gpu_h2d(").count();
    let on = device_emission(&ir, &[]);
    let off = device_emission(&ir, &["carry-take"]);
    assert!(
        on.text.contains("; mapal.take carry=0 bytes=8192"),
        "{}",
        on.text
    );
    assert!(!off.text.contains("; mapal.take"));
    assert_eq!(h2d(&on) + 1, h2d(&off));
    let want = interp(&ir);
    if let Some(clang) = emu::clang() {
        for em in [&on, &off] {
            let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
            for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
                assert_eq!(got, want, "{mode:?}");
            }
        }
    }
}
