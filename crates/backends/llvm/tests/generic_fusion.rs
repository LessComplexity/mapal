//! `flow` (`func/flow.rs`): the footprint rule's unit groups,
//! through the GPU emulator — values against the interpreter with the rung on
//! and off, and the launches and stored intermediates the facts require.

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, MorphismId, ObjectId, Operation};
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

/// The exact-face emission on `target` with `off` rungs off (and the ones the
/// emulator cannot run: `async`, `pdl`, `streams`; `place` so every capable
/// site is on the device). `flow` is forced on: it ships OFF on every measured
/// part (`GPU_RUNGS_OFF`, the fusion-natural-forms sweep), and these tests are
/// its proof of values, launches and stores.
fn emission(ir: &CategoryIr, target: &'static str, off: &[&str]) -> mapal_backend_llvm::Emission {
    emission_on(ir, target, off, EmitOpts::exact())
}

/// [`emission`] on the face `base` gives.
fn emission_on(
    ir: &CategoryIr,
    target: &'static str,
    off: &[&str],
    base: EmitOpts,
) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    for r in off
        .iter()
        .chain(&["place", "async", "pdl", "streams", "pinned"])
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// `(launches, units)` of the host text: every kernel launch (the `devgen`
/// builds of the closed-form operands aside: the same on both arms), the
/// unit ones.
fn launches(host: &str) -> (usize, usize) {
    let l = host
        .lines()
        .filter(|l| l.contains("call void @mapal_gpu_launch"))
        .count();
    let u = host.lines().filter(|l| l.ends_with("; unit")).count();
    (l, u)
}

/// A fused program: equal to the interpreter ON and OFF, and ON launches no
/// more than OFF. Returns the ON emission and both launch counts.
fn check(src: &str, target: &'static str) -> (mapal_backend_llvm::Emission, usize, usize) {
    let ir = lower(src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let on = emission(&ir, target, &[]);
    let off = emission(&ir, target, &["flow"]);
    let clang = emu::clang().expect("clang");
    for (arm, em) in [("on", &on), ("off", &off)] {
        let dev = em.device.as_ref().expect("a device unit");
        let got = emu::run_emulated(&clang, &em.text, dev);
        assert_eq!(got, rr.output, "flow {arm}\n{src}");
    }
    let (l_on, _) = launches(&on.text);
    let (l_off, u_off) = launches(&off.text);
    assert_eq!(u_off, 0, "the OFF arm forms no unit");
    assert!(l_on <= l_off, "ON {l_on} launches, OFF {l_off}");
    (on, l_on, l_off)
}

const N: u64 = 32;

fn head(n: u64) -> String {
    format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
",
        nn = n * n
    )
}

fn tail(o: &str, len: u64) -> String {
    format!(
        "    0.0 -> zero: f32;\n    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> chk;\n    chk -> println;\n    {o}[0] -> println;\n    {o}[{}] -> println;\n}}\n",
        len - 1
    )
}

fn row_fold(o: &str, src: &str, n: u64, seed: &str, step: &str) -> String {
    format!(
        "    krange -> map {{ i -> ({seed}, krange) -> fold {{ acc, j -> {src}[i * {n} + j] -> e; {step} }} }} -> {o};\n"
    )
}

/// exp(x) for x <= 0 as `gen_tiles.py`'s `expo`, in its natural form: the
/// clamped base squared once per step of an 8-step range (a fold nested in
/// the cell, whose body the unit kernel carries).
const EXPO: &str = "x / 256.0 + 1.0 -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> v0; 8 -> iota -> sq; (v0, sq) -> fold { acc, k -> acc * acc }";

/// softmax over the rows of a tile site's output (the `softmax_rows` probe's
/// form): a row max, a per-cell exp of the difference, a row sum of that, the
/// quotient. OFF: the tile, two folds, two cells (5 launches); ON: the tile and
/// one unit holding the four (2 launches), only `y` stored.
fn softmax(n: u64) -> String {
    let expo = EXPO;
    head(n)
        + &row_fold(
            "mx",
            "c",
            n,
            "-1000000000.0",
            "(e > acc) -> { -true-> e; -false-> acc; }",
        )
        + &format!("    ta -> map {{ t -> (c[t] - mx[t / {n}]) / {n}.0 -> x; {expo} }} -> ex;\n")
        + &row_fold("den", "ex", n, "0.0", "acc + e")
        + &format!("    ta -> map {{ t -> ex[t] / den[t / {n}] }} -> y;\n")
        + &tail("y", n * n)
}

#[test]
fn softmax_rows_is_one_unit_below_the_tile() {
    if emu::clang().is_none() {
        return;
    }
    for target in ["rtx-4090", "h100-pcie"] {
        let (on, l_on, l_off) = check(&softmax(N), target);
        assert_eq!((l_on, l_off), (2, 5), "{target}");
        let dev = on.device.as_ref().unwrap();
        let line = dev
            .lines()
            .find(|l| l.starts_with("; mapal.flow kernel="))
            .expect("a flow intent");
        assert!(
            line.contains("store=y ") && line.contains("keep=mx:reg:4;ex:reg:0;den:reg:4"),
            "{line}"
        );
    }
}

/// flow on the default face: softmax's and attention's rows are blocks of
/// lanes — `mx` and `den` trees, `ex` (read by `den` and the output) computed
/// once into shared memory — and the values answer to the interpreter within
/// the bound; on the exact face the folds walk in order (no tree).
#[test]
fn default_face_rows_are_lanes_with_tree_folds() {
    let Some(clang) = emu::clang() else {
        return;
    };
    for (src, target) in [(softmax(N), "rtx-4090"), (attention(64, 16), "h100-pcie")] {
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        let on = emission_on(&ir, target, &[], EmitOpts::default());
        let dev = on.device.as_ref().expect("a device unit");
        let line = dev
            .lines()
            .find(|l| l.starts_with("; mapal.flow kernel="))
            .expect("a flow intent");
        assert!(field(line, "lanes").parse::<u64>().unwrap() >= 32, "{line}");
        assert_eq!((field(line, "tree"), field(line, "onchip")), ("2", "1"), "{line}");
        assert!(dev.contains("@llvm.nvvm.barrier0()"), "{target}");
        let got = emu::run_emulated(&clang, &on.text, dev);
        assert!(within(&got, &rr.output), "{target}: {got:?} vs {:?}", rr.output);
        let exact = emission(&ir, target, &[]);
        let line = exact.device.as_ref().unwrap().lines().find(|l| l.starts_with("; mapal.flow kernel="));
        assert_eq!(line.map(|l| field(l, "tree")), Some("0"), "{target}");
    }
}

/// cpu-gpu-boundary item 5 `tree-gate`: a lone row fold over a tile site's
/// output is a tree unit on the default face, unless `rowfold-layout` stored
/// that output column-major for it (the rtx-4090 contract face): there a
/// row's cells are `rows` apart and the fold keeps its one-thread-a-row
/// kernel. Off: the unit whatever the layout. Values within the bound.
#[test]
fn a_lone_fold_over_a_column_major_store_is_no_tree_unit() {
    let src = head(N) + &row_fold("r", "c", N, "0.0", "acc + e") + &tail("r", N);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let units = |em: &mapal_backend_llvm::Emission| launches(&em.text).1;
    let on = emission_on(&ir, "rtx-4090", &[], EmitOpts::default());
    let off = emission_on(&ir, "rtx-4090", &["tree-gate"], EmitOpts::default());
    let row_major = emission_on(&ir, "rtx-4090", &["rowfold-layout"], EmitOpts::default());
    assert_eq!((units(&on), units(&off), units(&row_major)), (0, 1, 1));
    let Some(clang) = emu::clang() else {
        return;
    };
    for em in [&on, &off] {
        let got = emu::run_emulated(&clang, &em.text, em.device.as_ref().expect("a device unit"));
        assert!(within(&got, &rr.output), "{got:?} vs {:?}", rr.output);
    }
}

/// An element-wise chain over a tile site's output and an operand.
#[test]
fn ew_chain_is_one_unit() {
    if emu::clang().is_none() {
        return;
    }
    let src = head(N)
        + "    c -> map { v -> v * 3.0 } -> x;\n    x -> map { v -> v + 2.0 } -> y0;\n    ta -> map { t -> y0[t] * b[t] } -> z;\n"
        + "    z[0] -> println;\n    z[100] -> println;\n    z[1023] -> println;\n}\n";
    let (on, l_on, l_off) = check(&src, "rtx-4090");
    // The tile with `x`, `y0` and `z` in its store: one launch either way.
    assert_eq!((l_on, l_off), (1, 1), "{}", on.text);
}

/// A row normalization: a fold, then a per-cell quotient by its row value.
#[test]
fn rowsum_normalize() {
    if emu::clang().is_none() {
        return;
    }
    let n = N;
    let src = head(n)
        + &row_fold("r", "c", n, "0.0", "acc + e")
        + &format!("    ta -> map {{ t -> c[t] / (r[t / {n}] + 100.0) }} -> q;\n")
        + &tail("q", n * n);
    let (_, l_on, l_off) = check(&src, "rtx-4090");
    assert_eq!((l_on, l_off), (2, 3));
}

/// A halo: `q` reads the free cell `x` at `t / n` — outside its own cell, so
/// `x` is recomputed there (its only read is the tile output, in `Gmem`).
#[test]
fn halo_of_a_free_cell_is_recomputed() {
    if emu::clang().is_none() {
        return;
    }
    let n = N;
    let src = head(n)
        + &row_fold("r", "c", n, "0.0", "acc + e")
        + &format!("    ta -> map {{ t -> c[t / {n} * {n}] * 0.5 }} -> x;\n")
        + &format!("    ta -> map {{ t -> c[t] / (r[t / {n}] + 100.0) + x[t / {n}] }} -> q;\n")
        + &tail("q", n * n);
    let (on, l_on, l_off) = check(&src, "h100-pcie");
    assert_eq!((l_on, l_off), (3, 4));
    let dev = on.device.unwrap();
    let line = dev
        .lines()
        .find(|l| l.starts_with("; mapal.flow kernel="))
        .expect("a unit");
    assert!(
        line.contains("q:cell:1x1+halo") && line.contains("keep=x:reg:4"),
        "{line}"
    );
}

/// `window`: a 2-D convolution (`k÷3`, `k%3` taps) runs on the device; OFF it
/// is the host's conv rung. Its per-cell consumer is the tile root's epilogue.
#[test]
fn window_conv_runs_on_the_device() {
    if emu::clang().is_none() {
        return;
    }
    let n = 16u64;
    let w = n + 2;
    let src = format!(
        "fn main() {{
    {ii} -> iota -> ti;
    ti -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> img;
    9 -> iota -> k9;
    k9 -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 11.0 }} -> wk;
    {nn} -> iota -> to;
    to -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, k9) -> fold {{ acc, k -> acc + wk[k] * img[(i + k / 3) * {w} + j + k % 3] }}
    }} -> y;
    y -> map {{ v -> v * 0.5 + 1.0 }} -> z;
",
        ii = w * w,
        nn = n * n
    ) + &tail("z", n * n);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let clang = emu::clang().unwrap();
    let on = emission(&ir, "rtx-4090", &[]);
    let dev = on.device.as_ref().expect("the conv is a device site");
    assert!(dev.contains("kind=naive") && dev.contains("epi=1"), "{dev}");
    assert_eq!(emu::run_emulated(&clang, &on.text, dev), rr.output);
    let off = emission(&ir, "rtx-4090", &["window"]);
    assert!(
        off.device.is_none(),
        "window off: the conv stays on the host"
    );
}

/// `window`: a weightless 2-D window (a 2×2 box filter) over the conv's device
/// output is a row fold with a split map index and split taps — a device
/// fold, in order per cell — and a per-cell consumer of it joins its unit.
#[test]
fn window_box_filter_is_a_device_row_fold() {
    if emu::clang().is_none() {
        return;
    }
    let n = 16u64;
    let (w, m) = (n + 2, n - 1);
    let src = format!(
        "fn main() {{
    {ii} -> iota -> ti;
    ti -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> img;
    9 -> iota -> k9;
    k9 -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 11.0 }} -> wk;
    {nn} -> iota -> to;
    to -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, k9) -> fold {{ acc, k -> acc + wk[k] * img[(i + k / 3) * {w} + j + k % 3] }}
    }} -> y;
    4 -> iota -> k4;
    {mm} -> iota -> tb;
    tb -> map {{ t ->
        t / {m} -> i;
        t % {m} -> j;
        (0.0, k4) -> fold {{ acc, k -> acc + y[(i + k / 2) * {n} + j + k % 2] }}
    }} -> bx;
    tb -> map {{ t -> bx[t] * 0.25 }} -> z;
",
        ii = w * w,
        nn = n * n,
        mm = m * m
    ) + &tail("z", m * m);
    let ir = lower(&src);
    let folds = ir.row_fold(ir.entry());
    let s = folds.values().next().expect("the box filter is a row fold");
    assert_eq!((s.rows, s.k, s.w, s.a.ci, s.a.clane), (m * m, 4, m, n, 1));
    assert_eq!(
        s.a.ksplit,
        Some(mapal_ir::TileKSplit {
            div: 2,
            cq: n,
            cr: 1
        })
    );
    let (on, l_on, l_off) = check(&src, "rtx-4090");
    // The conv; the box fold and `z` (one unit ON, two launches OFF).
    assert_eq!((l_on, l_off), (2, 3), "{}", on.text);
}

/// `q·Kᵀ` over a 2-D grid (`k` read at the lane stride `d`) is a device site
/// (`tile_plan_strided`'s split branch), then the row softmax and `·V`.
fn attention(t: u64, d: u64) -> String {
    let expo = EXPO;
    format!(
        "fn main() {{
    {td} -> iota -> tq;
    tq -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> q;
    tq -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 64.0 }} -> kk;
    tq -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 16.0 }} -> vv;
    {d} -> iota -> kd;
    {t} -> iota -> kt;
    {tt} -> iota -> ts;
    ts -> map {{ e ->
        e / {t} -> i;
        e % {t} -> j;
        (0.0, kd) -> fold {{ acc, k -> acc + q[i * {d} + k] * kk[j * {d} + k] }}
    }} -> s;
    kt -> map {{ i -> (-1000000000.0, kt) -> fold {{ acc, j -> s[i * {t} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    ts -> map {{ e -> s[e] - mx[e / {t}] -> x; {expo} }} -> ex;
    kt -> map {{ i -> (0.0, kt) -> fold {{ acc, j -> acc + ex[i * {t} + j] }} }} -> den;
    ts -> map {{ e -> ex[e] / den[e / {t}] }} -> p;
    tq -> map {{ e ->
        e / {d} -> i;
        e % {d} -> j;
        (0.0, kt) -> fold {{ acc, u -> acc + p[i * {t} + u] * vv[u * {d} + j] }}
    }} -> o;
",
        td = t * d,
        tt = t * t
    ) + &tail("o", t * d)
}

#[test]
fn attention_is_three_launches() {
    if emu::clang().is_none() {
        return;
    }
    // q·Kᵀ, one unit for the row softmax, ·V.
    let (_, l_on, l_off) = check(&attention(32, 8), "rtx-4090");
    assert_eq!((l_on, l_off), (3, 6));
}

/// `FUSE_SRC=<file.mapal>`: the program's values on and off against the
/// interpreter, both test profiles; prints the launch counts (the suite's
/// `px_*` chain differentials).
#[test]
#[ignore]
fn check_program_file() {
    let src = std::fs::read_to_string(std::env::var("FUSE_SRC").expect("FUSE_SRC")).unwrap();
    for target in ["rtx-4090", "h100-pcie"] {
        let (_, l_on, l_off) = check(&src, target);
        eprintln!("{target}: launches on {l_on} off {l_off}");
    }
}

/// The intent lines of a program's ON emission (debugging aid).
#[test]
#[ignore]
fn dump_intents() {
    let src = std::env::var("FUSE_SRC")
        .map_or_else(|_| softmax(N), |p| std::fs::read_to_string(p).unwrap());
    let ir = lower(&src);
    let on = emission(&ir, "rtx-4090", &[]);
    for l in on
        .text
        .lines()
        .filter(|l| l.contains("mapal.flow") || l.contains("launch"))
    {
        eprintln!("{l}");
    }
    for l in on
        .device
        .unwrap()
        .lines()
        .filter(|l| l.contains("mapal.flow") || l.contains("mapal.plan"))
    {
        eprintln!("{l}");
    }
}

/// One `key=value` field of an intent line (`""` when absent).
fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// The labels of a `;`-separated intent field, each up to its first `:`.
fn labels(v: &str) -> Vec<String> {
    v.split(';')
        .filter(|x| !x.is_empty())
        .map(|x| x.split(':').next().unwrap().to_string())
        .collect()
}

/// The object a unit's intent line names (`object_label`: the source name).
fn named(ir: &CategoryIr, label: &str) -> Option<ObjectId> {
    ir.objects()
        .find(|(_, o)| o.name.as_deref() == Some(label))
        .map(|(id, _)| id)
}

/// The morphisms reading `o`, through the products that carry it.
fn readers(ir: &CategoryIr, o: ObjectId) -> Vec<MorphismId> {
    let mut out = Vec::new();
    for &m in ir.out_edges(o) {
        let mo = ir.morphism(m).unwrap();
        match mo.op {
            Operation::Pair { .. } => out.extend(readers(ir, mo.target)),
            _ => out.push(m),
        }
    }
    out
}

/// Per `; mapal.move` loop line: the planned `(h2d, d2h)` per trip.
fn planned_moves(host: &str) -> Vec<(u64, u64)> {
    host.lines()
        .filter_map(|l| l.trim().strip_prefix("; mapal.move loop="))
        .map(|l| {
            let n = |k: &str| field(l, k).parse::<u64>().unwrap_or(0);
            (n("h2d"), n("d2h"))
        })
        .collect()
}

/// **The rule on unseen programs.** ≥ 256 generated chains of mixed
/// algorithms (testgen `chain_strategy`: laws, capture-free maps, cells at
/// every footprint, tile sites, windows, row folds, a fold's value per cell,
/// whole folds, loops), rtx-4090 and h100-pcie alternating, emulated:
/// 1. values: the exact face equals the interpreter, `flow` on and off;
/// 2. groups: on keeps every tile root off had (`epi=1` kernels), and adds
///    unit groups only;
/// 3. launches: on launches exactly off's less one per unit member after the
///    first — every launch is a group or an ungrouped device unit;
/// 4. stores: a unit stores a member iff the IR has a reader of it outside the
///    unit or it escapes (recomputed here from the graph); a unit kernel's
///    pointers are its stored values and operands, nothing kept;
/// 5. movement: per loop trip, on moves no byte more than off.
#[test]
fn generic_fusion_chains_values_launches_and_stores() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP generic_fusion_chains: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let n: usize = std::env::var("FUSE_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::chain_strategy();
    let mut findings = Vec::new();
    let (mut device, mut units, mut members, mut kept, mut halos, mut saved) = (0, 0, 0, 0, 0, 0);
    let (mut defaults, mut trees, mut onchip) = (0, 0, 0);
    let lines_of = |d: &str| -> Vec<String> {
        d.lines()
            .filter(|l| l.starts_with("; mapal.flow kernel="))
            .map(String::from)
            .collect()
    };
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
    for z in 0..n {
        let ch = strat.new_tree(&mut runner).unwrap().current();
        let src = ch.source();
        let ir = lower(&src);
        let rr = run(&ir, 400_000_000);
        assert!(
            matches!(rr.outcome, Outcome::Done(_)),
            "chain#{z} does not finish\n{src}"
        );
        let target = if z % 2 == 0 { "rtx-4090" } else { "h100-pcie" };
        let ems = std::panic::catch_unwind(|| {
            (emission(&ir, target, &[]), emission(&ir, target, &["flow"]))
        });
        let Ok((on, off)) = ems else {
            findings.push(format!("chain#{z} {target}: the emitter panicked\n{src}"));
            continue;
        };
        let (Some(don), Some(doff)) = (&on.device, &off.device) else {
            if on.device.is_some() != off.device.is_some() {
                findings.push(format!("chain#{z}: a device unit on one arm only\n{src}"));
            }
            continue;
        };
        device += 1;
        for (arm, em, dev) in [("on", &on, don), ("off", &off, doff)] {
            let got = emu::run_emulated(&clang, &em.text, dev);
            if got != rr.output {
                findings.push(format!(
                    "chain#{z} {target} flow {arm}: {got:?} != {:?}\n{src}",
                    rr.output
                ));
            }
        }
        // 1b. The default face (fma steps, order free): a unit's folds are
        // trees across the row's lanes, so on answers to the interpreter
        // within the bound, not bit for bit (every 4th program).
        if z % 4 == 0 && !lines_of(don).is_empty() {
            let base = EmitOpts::default();
            let (a, b) = (
                emission_on(&ir, target, &[], base),
                emission_on(&ir, target, &["flow"], base),
            );
            if let (Some(da), Some(db)) = (&a.device, &b.device) {
                defaults += 1;
                let (ga, gb) = (
                    emu::run_emulated(&clang, &a.text, da),
                    emu::run_emulated(&clang, &b.text, db),
                );
                if !within(&ga, &rr.output) || !within(&gb, &rr.output) {
                    findings.push(format!(
                        "chain#{z} {target} default face: on {ga:?} off {gb:?} vs {:?}\n{src}",
                        rr.output
                    ));
                }
                // The geometry: a row of more than one cell or fold step is a
                // block of lanes, and every associative fold of a unit is a tree.
                for l in lines_of(da) {
                    let w: u64 = field(&l, "unit").trim_start_matches("1x").parse().unwrap_or(1);
                    let lanes: u64 = field(&l, "lanes").parse().unwrap_or(0);
                    let folds = labels(field(&l, "members"))
                        .iter()
                        .filter(|m| m.contains(":rowfold:"))
                        .count();
                    if (w > 1 || folds > 0) && lanes < 32 {
                        findings.push(format!("chain#{z}: a unit row of {w} on {lanes} lanes: {l}\n{src}"));
                    }
                    trees += field(&l, "tree").parse::<usize>().unwrap_or(0);
                    onchip += field(&l, "onchip").parse::<usize>().unwrap_or(0);
                }
            }
        }
        // 2. Tile roots unchanged.
        let epi = |d: &str| {
            d.lines()
                .filter(|l| l.starts_with("; mapal.plan") && l.contains(" epi=1 "))
                .count()
        };
        if epi(don) != epi(doff) {
            findings.push(format!(
                "chain#{z}: tile roots {} on, {} off\n{src}",
                epi(don),
                epi(doff)
            ));
        }
        // 3. Launches.
        let lines: Vec<&str> = don
            .lines()
            .filter(|l| l.starts_with("; mapal.flow kernel="))
            .collect();
        let unit_members: usize = lines
            .iter()
            .map(|l| labels(field(l, "members")).len())
            .sum();
        let (l_on, u_on) = launches(&on.text);
        let (l_off, _) = launches(&off.text);
        if u_on != lines.len() || l_on + unit_members != l_off + lines.len() {
            findings.push(format!(
                "chain#{z}: {l_on} launches on ({u_on} units of {unit_members} members), {l_off} off\n{src}"
            ));
        }
        units += lines.len();
        members += unit_members;
        saved += l_off.saturating_sub(l_on);
        // 4. Stores, recomputed from the graph.
        let lup = ir.last_use_plan(ir.entry());
        for l in &lines {
            let store = labels(field(l, "store"));
            let keep = labels(field(l, "keep"));
            kept += keep.len();
            halos += usize::from(field(l, "members").contains("+halo"));
            let objs: Vec<(String, Option<ObjectId>)> = store
                .iter()
                .chain(&keep)
                .map(|x| (x.clone(), named(&ir, x)))
                .collect();
            let ms: Vec<MorphismId> = objs
                .iter()
                .filter_map(|(_, o)| o.and_then(|o| ir.in_edges(o).first().copied()))
                .collect();
            for (label, o) in &objs {
                let Some(o) = *o else {
                    findings.push(format!(
                        "chain#{z}: unit member {label} names no object\n{l}"
                    ));
                    continue;
                };
                let outside = lup.escapes(o) || readers(&ir, o).iter().any(|r| !ms.contains(r));
                if outside != store.contains(label) {
                    findings.push(format!(
                        "chain#{z}: {label} stored={} but outside reader={outside}\n{l}\n{src}",
                        store.contains(label)
                    ));
                }
            }
            let name = field(l, "kernel");
            let def = don
                .lines()
                .find(|d| d.contains(&format!("void @{name}(")))
                .unwrap_or("");
            let params = def.matches("ptr addrspace(1)").count();
            if params != store.len() + labels(field(l, "gmem")).len() {
                findings.push(format!(
                    "chain#{z}: {name} has {params} pointers\n{l}\n{def}"
                ));
            }
        }
        for l in on
            .text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("; mapal.flow.host "))
        {
            for r in field(l, "refused").split(';').filter(|x| !x.is_empty()) {
                *reasons
                    .entry(r.rsplit(':').next().unwrap().to_string())
                    .or_default() += 1;
            }
        }
        // 5. Movement per trip.
        let (pon, poff) = (planned_moves(&on.text), planned_moves(&off.text));
        if pon.len() != poff.len() || pon.iter().zip(&poff).any(|(a, b)| a.0 > b.0 || a.1 > b.1) {
            findings.push(format!(
                "chain#{z}: per-trip moves on {pon:?} off {poff:?}\n{src}"
            ));
        }
    }
    assert!(
        findings.is_empty(),
        "{} findings:\n{}",
        findings.len(),
        findings.join("\n")
    );
    eprintln!(
        "generic_fusion_chains: {n} programs, {device} with a device unit; {units} units of {members} members \
         ({kept} kept, {halos} with a halo); {saved} launches removed; refusals {reasons:?}; \
         {defaults} default-face programs within the bound ({trees} tree folds, {onchip} rows on chip)"
    );
    assert!(
        n < 256 || (device >= 128 && units >= 32 && kept > 0),
        "too few fused programs"
    );
}

/// The symbols `define … void @mapal_(tile|gen)_<id>(` in a device unit.
fn defined_kernels(dev: &str) -> std::collections::BTreeSet<String> {
    dev.lines()
        .filter(|l| l.starts_with("define "))
        .filter_map(|l| l.split_once(" void @").map(|(_, r)| r))
        .filter_map(|r| r.split_once('(').map(|(n, _)| n.to_owned()))
        .filter(|n| n.starts_with("mapal_tile_") || n.starts_with("mapal_gen_"))
        .collect()
}

/// The kernels the host's launch sites load (`mapal_gpu.c`'s naming).
fn launched_kernels(host: &str) -> std::collections::BTreeSet<String> {
    let mut live = std::collections::BTreeSet::new();
    for l in host.lines() {
        let Some((_, r)) = l.split_once("call void @mapal_gpu_") else {
            continue;
        };
        let Some((entry, args)) = r.split_once("(i64 ") else {
            continue;
        };
        let prefix = match entry {
            "tile" | "launch" | "launch_l2" | "launchv" => "mapal_tile",
            "gen" => "mapal_gen",
            _ => continue,
        };
        let lit: String = args.chars().take_while(|c| *c == '-' || c.is_ascii_digit()).collect();
        let id = lit
            .parse::<u64>()
            .unwrap_or_else(|_| lit.parse::<i64>().expect("launch id") as u64);
        live.insert(format!("{prefix}_{id:016x}"));
    }
    live
}

/// Dead kernels (launch-site use counts): over unseen programs from the
/// generator (device loops, chains) on every GPU profile, default face and the
/// exact all-on-device face, the device unit defines exactly the kernels some
/// host launch site loads — none unlaunched, none missing.
#[test]
fn every_device_kernel_has_a_launch_site() {
    let mut runner = TestRunner::deterministic();
    let (dev_s, chain_s) = (testgen::device_prog_strategy(), testgen::chain_strategy());
    let (mut units, mut kernels) = (0usize, 0usize);
    for z in 0..48 {
        let src = if z % 2 == 0 {
            testgen::device_src(&dev_s.new_tree(&mut runner).unwrap().current())
        } else {
            chain_s.new_tree(&mut runner).unwrap().current().source()
        };
        let ir = lower(&src);
        for target in ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"] {
            let default = EmitOpts {
                host: "generic",
                target,
                ..EmitOpts::default()
            };
            for em in [
                emit_detailed(&ir, &default).expect("emits"),
                emission(&ir, target, &[]),
            ] {
                let Some(dev) = em.device.as_deref() else {
                    continue;
                };
                let (def, live) = (defined_kernels(dev), launched_kernels(&em.text));
                assert_eq!(def, live, "#{z} {target}: defined vs launched\n{src}");
                units += 1;
                kernels += def.len();
            }
        }
    }
    assert!(
        units >= 64 && kernels >= units,
        "too few device units: {units} units, {kernels} kernels"
    );
}

/// The default face's values against the interpreter's: every numeric token
/// within a relative 1e-4 of the larger magnitude (or 1e-4 absolute below 1),
/// the rest equal.
fn within(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.split_whitespace().collect(), want.split_whitespace().collect());
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
            _ => a == b,
        })
}
