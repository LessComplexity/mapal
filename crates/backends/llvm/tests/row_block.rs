//! compile-time-graph A: `unit_plan`'s row-block groups and the `flow` rule's
//! row-block unit kernels — the fact on known shapes, values through the GPU
//! emulator against the interpreter with the arm on and off, and the launches
//! the fact predicts, on hand-written and generated programs.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, FuncId, UnitKind};
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

fn main_of(ir: &CategoryIr) -> FuncId {
    ir.funcs().find(|(_, d)| d.name == "main").expect("main").0
}

/// The emission on `target` and `base`'s face: the emulator's rungs off
/// (`async`, `pdl`, `streams`, `pinned`; `place` so every capable site is on
/// the device), `flow` on, the row-block arm `on` (its rung forced on: the
/// shipped table keeps it off), its rows forced to `b` (0 = derive).
fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts, on: bool, b: u64) -> Emission {
    emission_off(ir, target, base, on, b, &[])
}

/// [`emission`] with the rungs `off` off too.
fn emission_off(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    on: bool,
    b: u64,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.row_block = on;
    gpu.on.row_block = on;
    gpu.row_block_b = b;
    for r in ["place", "async", "pdl", "streams", "pinned"]
        .iter()
        .chain(off)
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

/// Kernel launches of a host text (site tables, launches, units).
fn launches(host: &str) -> usize {
    host.lines()
        .filter(|l| {
            l.contains("call void @mapal_gpu_launch") || l.contains("call void @mapal_gpu_site")
        })
        .count()
}

/// The row-block unit intent lines of a device unit.
fn block_lines(em: &Emission) -> Vec<String> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.flow kernel=") && l.contains(" rows_per_unit="))
        .map(str::to_string)
        .collect()
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

fn within(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

/// exp(x) for x <= 0 in its natural form (`gen_tiles.py`'s `expo`).
const EXPO: &str = "x / 256.0 + 1.0 -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> v0; 8 -> iota -> sq; (v0, sq) -> fold { acc, k -> acc * acc }";

/// Single-head attention over `t` rows of width `d`, with `extra` lines after
/// `o` (and `out` printed).
fn attention(t: u64, d: u64, extra: &str, out: &str, len: u64) -> String {
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
    ts -> map {{ e -> s[e] - mx[e / {t}] -> x; {EXPO} }} -> ex;
    kt -> map {{ i -> (0.0, kt) -> fold {{ acc, j -> acc + ex[i * {t} + j] }} }} -> den;
    ts -> map {{ e -> ex[e] / den[e / {t}] }} -> p;
    tq -> map {{ e ->
        e / {d} -> i;
        e % {d} -> j;
        (0.0, kt) -> fold {{ acc, u -> acc + p[i * {t} + u] * vv[u * {d} + j] }}
    }} -> o;
{extra}    0.0 -> zero: f32;
    (zero, {out}) -> fold {{ acc, v -> acc + v }} -> chk;
    chk -> println;
    {out}[0] -> println;
    {out}[{last}] -> println;
}}
",
        td = t * d,
        tt = t * t,
        last = len - 1
    )
}

fn kinds(ir: &CategoryIr) -> Vec<Vec<(&'static str, u64, bool, Option<usize>)>> {
    ir.unit_plan(main_of(ir))
        .groups
        .iter()
        .map(|g| {
            g.members
                .iter()
                .map(|mb| {
                    let k = match mb.kind {
                        UnitKind::Tile(_) => "tile",
                        UnitKind::Fold(_) => "fold",
                        UnitKind::Cell => "cell",
                    };
                    (k, mb.w, mb.stored, mb.color)
                })
                .collect()
        })
        .collect()
}

/// The fact on attention: one group from `q·Kᵀ` to `·V`; only `o` stored;
/// `s`, `ex` and `p` share one row buffer (each the last reader of the one
/// before, at its own cell), `mx` and `den` another.
#[test]
fn unit_plan_attention_is_one_group() {
    let ir = lower(&attention(32, 8, "", "o", 256));
    let plan = ir.unit_plan(main_of(&ir));
    assert_eq!(plan.groups.len(), 1);
    let g = &plan.groups[0];
    assert_eq!(g.rows, 32);
    assert_eq!(
        kinds(&ir)[0],
        vec![
            ("tile", 32, false, Some(0)),
            ("fold", 1, false, Some(1)),
            ("cell", 32, false, Some(0)),
            ("fold", 1, false, Some(1)),
            ("cell", 32, false, Some(0)),
            ("tile", 8, true, None),
        ]
    );
    assert_eq!(g.row_bytes, 32 * 4 + 4);
    // The column operands, `kk` and `vv`, read whole by every unit.
    assert_eq!(g.shared_bytes, 2 * 256 * 4);
    // Stored once: a second query is the stored fact.
    assert_eq!(ir.unit_plan(main_of(&ir)), plan);
}

/// A member read outside the group is stored; a per-row value read after a
/// tile is still a member (`o / den` closes in its row).
#[test]
fn unit_plan_outside_reader_stores() {
    let extra = "    tq -> map { e -> o[e] / den[e / 8] } -> o2;\n    den[3] -> println;\n";
    let ir = lower(&attention(32, 8, extra, "o2", 256));
    let k = kinds(&ir);
    assert_eq!(k.len(), 1);
    let g = &k[0];
    assert_eq!(g.len(), 7, "{g:?}");
    assert_eq!(g[3], ("fold", 1, true, Some(1)), "den is printed: stored");
    // A tile never writes over its row operand (other threads still read it).
    assert_eq!(g[5], ("tile", 8, false, Some(2)), "o is read by o2 only");
    assert_eq!(g[6], ("cell", 8, true, None));
}

/// Not closed: a consumer reading the tile's output by column, and a tile
/// whose column operand `b` is a member (every row reads all of it).
#[test]
fn unit_plan_refuses_unclosed_reads() {
    let n = 16;
    let head = format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> kr;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
",
        nn = n * n
    );
    // A column sum of `c`: row `i` of the fold reads column `i`.
    let col = head.clone()
        + &format!(
            "    kr -> map {{ i -> (0.0, kr) -> fold {{ acc, j -> acc + c[j * {n} + i] }} }} -> cs;\n    cs[0] -> println;\n    cs[{}] -> println;\n}}\n",
            n - 1
        );
    assert!(
        lower(&col)
            .unit_plan(main_of(&lower(&col)))
            .groups
            .is_empty()
    );
    // `c` as the next tile's `b`.
    let bop = head
        + &format!(
            "    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{ acc, k -> acc + a[i * {n} + k] * c[k * {n} + j] }}
    }} -> d;
    d[0] -> println;
    d[{}] -> println;
}}
",
            n * n - 1
        );
    let ir = lower(&bop);
    assert!(ir.unit_plan(main_of(&ir)).groups.is_empty());
}

/// Values: attention ON (every `b` that fits, both faces) against the
/// interpreter, and the OFF arm; ON is one launch where OFF is three or more.
#[test]
fn attention_row_block_values_and_launches() {
    let Some(clang) = emu::clang() else {
        return;
    };
    // `o2` reads a per-row value (`den`) and a row (`ex`) across the second
    // tile (`ex` keeps `o` from being an epilogue root: it is no operand
    // complete before `o` launches).
    for extra in [
        "",
        "    tq -> map { e -> o[e] / den[e / 8] + ex[(e / 8) * 32] } -> o2;\n",
    ] {
        let out = if extra.is_empty() { "o" } else { "o2" };
        let src = attention(32, 8, extra, out, 256);
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        let off = emission(&ir, "rtx-5080", EmitOpts::exact(), false, 0);
        assert!(block_lines(&off).is_empty());
        let l_off = launches(&off.text);
        let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap());
        assert_eq!(got, rr.output, "OFF");
        for b in [1, 2, 4, 8] {
            for (face, base) in [
                ("exact", EmitOpts::exact()),
                ("default", EmitOpts::default()),
            ] {
                let on = emission(&ir, "rtx-5080", base, true, b);
                let lines = block_lines(&on);
                assert_eq!(lines.len(), 1, "b={b} {face}: {}", on.text);
                assert_eq!(field(&lines[0], "rows_per_unit"), b.to_string());
                assert_eq!(field(&lines[0], "tiles"), "2");
                assert_eq!(launches(&on.text), 1, "b={b} {face}: one launch");
                assert!(l_off >= 3, "OFF: {l_off}");
                // Free-running threads, then serialized between barriers in
                // thread order and in reverse (`MAPAL_EMU_SERIAL`).
                let ex = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap());
                for mode in ["0", "1", "2"] {
                    let (got, _) = ex.run(&[("MAPAL_EMU_SERIAL", mode)]);
                    match face {
                        "exact" => assert_eq!(got, rr.output, "b={b} exact serial={mode}"),
                        _ => assert!(
                            within(&got, &rr.output),
                            "b={b} serial={mode}: {got:?} vs {:?}",
                            rr.output
                        ),
                    }
                }
            }
        }
    }
}

fn bench(name: &str) -> CategoryIr {
    let src = std::fs::read_to_string(format!(
        "{}/../../../benches/algorithms/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    lower(&src)
}

/// The shipped arms on `target` (the emulator's rung offs change the split's
/// plans), the row-block rung forced on (`rung`), its rows forced to `b`.
fn shipped(ir: &CategoryIr, target: &'static str, rung: bool, b: u64) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.row_block = rung;
    gpu.row_block_b = b;
    let opts = EmitOpts {
        target,
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The rule on the real shape, `pt_attn_1024_64` on the rtx-5080: the group
/// fits (b = 16, 84 KB) but the plan model keeps the boundary — the unit reads
/// Kᵀ straight from L2 at half a sector a lane, which the split's tile site
/// stages (measured: the fused kernel 33.3 us against 31.7 us for the split
/// kernels, `compile-time-graph.md`). Forced, it is one launch storing `o`
/// alone. Shipped, the rung is off (the rtx-5090 row of its architecture).
#[test]
fn attn_1024_on_rtx_5080_keeps_the_boundary_by_cost() {
    let ir = bench("pt_attn_1024_64");
    let off = shipped(&ir, "rtx-5080", false, 0);
    assert!(block_lines(&off).is_empty());
    assert!(!off.text.contains(" row_block="), "{}", off.text);
    let on = shipped(&ir, "rtx-5080", true, 0);
    assert!(block_lines(&on).is_empty());
    assert!(on.text.contains(" row_block=s:cost"), "{}", on.text);
    let forced = shipped(&ir, "rtx-5080", false, 16);
    let lines = block_lines(&forced);
    assert_eq!(lines.len(), 1, "{}", forced.text);
    assert_eq!(launches(&forced.text), 1);
    assert_eq!(field(&lines[0], "store"), "o");
    assert_eq!(field(&lines[0], "smem"), "83968");
}

/// Every GPU part's shipped default forms no row-block unit (the rung's
/// `GPU_RUNGS_OFF` rows: no realized unit was measured to pay); with the rung
/// on, a part without plan rates refuses each group `unpriced` (nothing
/// weighs the unit against the launches it replaces) and a part with them
/// decides by the model.
#[test]
fn row_block_ships_off_and_refuses_unpriced() {
    let parts = [
        ("rtx-5080", true),
        ("rtx-5090", true),
        ("rtx-4090", true),
        ("h100-pcie", false),
        ("h100", false),
        ("a100", false),
        ("rtx-3090", false),
    ];
    for name in ["pt_attn_1024_64", "px_attn_64_16"] {
        let ir = bench(name);
        for (part, priced) in parts {
            let def = shipped(&ir, part, false, 0);
            assert!(block_lines(&def).is_empty(), "{name} {part}: shipped");
            assert!(!def.text.contains(" row_block="), "{name} {part}");
            let on = shipped(&ir, part, true, 0);
            let refused: Vec<&str> = on
                .text
                .lines()
                .flat_map(|l| l.split_whitespace())
                .filter_map(|f| f.strip_prefix("row_block="))
                .collect();
            if !priced {
                assert!(block_lines(&on).is_empty(), "{name} {part}: unpriced");
                assert!(
                    !refused.is_empty() && refused.iter().all(|r| r.ends_with(":unpriced")),
                    "{name} {part}: {refused:?}"
                );
            }
            for l in block_lines(&on) {
                let (fused, split): (u64, u64) = (
                    field(&l, "ns_fused").parse().unwrap(),
                    field(&l, "ns_split").parse().unwrap(),
                );
                assert!(fused > 0 && fused <= split, "{name} {part}: {l}");
            }
        }
    }
}

/// kernel-merging `residence_plan`: the plan model prices each split
/// segment's bytes where they live (the working set from producer to last
/// reader against the part's L2), so attention is one rule on the three
/// parts with plan rates. The split softmax's rows (12.6 MB at 1024 × 64)
/// stay in L2 on the 64 / 72 / 96 MB caches: every large attention keeps
/// its boundary by cost (the rtx-5080's measured order: fused 33.3 us > split
/// 31.7); before, the 4090 / 5090 priced those rows at the DRAM placement
/// rate `bytes_per_ns` and formed the unit. The launch-bound
/// `px_attn_64_16` forms on all three.
#[test]
fn attention_pricing_is_one_rule_across_parts() {
    for part in ["rtx-4090", "rtx-5090", "rtx-5080"] {
        for name in ["pt_attn_1024_64", "pt_attn_1024_128", "pt_attn_2048_64"] {
            let on = shipped(&bench(name), part, true, 0);
            assert!(block_lines(&on).is_empty(), "{name} {part}");
            let why: Vec<&str> = on
                .text
                .lines()
                .flat_map(|l| l.split_whitespace())
                .filter_map(|f| f.strip_prefix("row_block=s:cost:"))
                .collect();
            assert_eq!(why.len(), 1, "{name} {part}: {}", on.text);
            let (f, s) = why[0].split_once(':').unwrap().1.split_once('/').unwrap();
            let (f, s): (u64, u64) = (f.parse().unwrap(), s.parse().unwrap());
            assert!(f > s && s > 0, "{name} {part}: fused {f} split {s}");
        }
        let small = shipped(&bench("px_attn_64_16"), part, true, 0);
        assert_eq!(block_lines(&small).len(), 1, "px_attn_64_16 {part}");
    }
}

/// One generated program's row-block checks ([`row_block_on_generated_chains`]).
#[derive(Default)]
struct ChainCheck {
    groups: usize,
    formed: usize,
    why: std::collections::BTreeMap<String, usize>,
    /// Formed units whose row group spans several warps, that fold as a tree,
    /// split a tile's `k` over a row's spare lanes, take FMA steps, and write
    /// a row buffer an earlier member of their segment died out of.
    wide: usize,
    tree: usize,
    split: usize,
    fma: usize,
    reuse: usize,
    findings: Vec<String>,
}

/// The device text of each row-block unit kernel of `em`.
fn unit_texts(em: &Emission) -> Vec<String> {
    let dev = em.device.as_deref().unwrap_or("");
    let lines: Vec<&str> = dev.lines().collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l.starts_with("; mapal.flow kernel=") && l.contains(" rows_per_unit=") {
            let end = lines[i..]
                .iter()
                .position(|x| *x == "}")
                .map_or(lines.len(), |e| i + e);
            out.push(lines[i..end].join("\n"));
        }
    }
    out
}

/// A row group's lanes in a unit kernel's text (`%ll<s> = urem i64 %l, <l>`).
fn row_group_lanes(text: &str) -> Vec<u64> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim_start();
            if !l.starts_with("%ll") {
                return None;
            }
            l.split_once(" = urem i64 %l, ")?.1.trim().parse().ok()
        })
        .collect()
}

fn check_chain(clang: &str, z: usize, src: &str) -> ChainCheck {
    let mut ck = ChainCheck::default();
    let ir = lower(src);
    let plan = ir.unit_plan(main_of(&ir));
    if plan.groups.is_empty() {
        return ck;
    }
    ck.groups = plan.groups.len();
    // A row member whose buffer's last row occupant before it is a member it
    // does not read: `unit_plan` freed the buffer at that member's death.
    let reused = plan.groups.iter().any(|g| {
        g.members.iter().enumerate().any(|(c, mb)| {
            let prev = (0..c)
                .rev()
                .find(|&p| g.members[p].w > 1 && g.members[p].color == mb.color);
            mb.w > 1
                && mb.color.is_some()
                && prev.is_some_and(|p| !mb.reads.iter().any(|&(_, r)| r == p))
        })
    });
    let bmax = plan
        .groups
        .iter()
        .map(|g| 1u64 << g.rows.trailing_zeros().min(3))
        .min()
        .unwrap();
    let mut bs = vec![1, bmax];
    bs.dedup();
    let faces = [
        ("exact", EmitOpts::exact()),
        ("default", EmitOpts::default()),
    ];
    // The epilogue rung off: a tile with per-cell readers is the group's.
    let off_rungs = ["epilogue"];
    let mut runs: Vec<(String, Emission, bool)> = Vec::new();
    for (face, base) in &faces {
        let off = emission_off(&ir, "rtx-5080", *base, false, 0, &off_rungs);
        let l_off = launches(&off.text);
        for (n, &b) in bs.iter().enumerate() {
            let on = emission_off(&ir, "rtx-5080", *base, true, b, &off_rungs);
            let lines = block_lines(&on);
            if *face == "exact" && n == 0 {
                for l in on.text.lines().filter(|l| l.contains("mapal.flow.host")) {
                    for f in l
                        .split_whitespace()
                        .filter_map(|f| f.strip_prefix("row_block="))
                    {
                        *ck.why
                            .entry(f.split(':').nth(1).unwrap_or("").to_string())
                            .or_insert(0) += 1;
                    }
                }
                ck.formed = lines.len();
            }
            if lines.is_empty() {
                continue;
            }
            // Each formed group is one launch holding exactly the fact's
            // members (its tiles included); ON launches no more than OFF.
            let sizes: Vec<usize> = plan.groups.iter().map(|g| g.members.len()).collect();
            for l in &lines {
                let n = field(l, "members").split(';').count();
                let tiles: usize = field(l, "tiles").parse().unwrap();
                if !sizes.contains(&n) || tiles == 0 {
                    ck.findings.push(format!(
                        "chain#{z} {face} b={b}: unit of {n} members ({tiles} tiles) vs groups {sizes:?}\n{l}"
                    ));
                }
                if *face == "default" && field(l, "tree") != "0" {
                    ck.tree += 1;
                }
            }
            let (l_on, l_off) = (launches(&on.text), l_off);
            if l_on > l_off {
                ck.findings.push(format!(
                    "chain#{z} {face} b={b}: launches ON {l_on} > OFF {l_off}\n{src}"
                ));
            }
            for t in unit_texts(&on) {
                ck.wide += usize::from(row_group_lanes(&t).iter().any(|&l| l > 32));
                ck.split += usize::from(t.lines().any(|l| l.starts_with('T') && l.ends_with("z:")));
                ck.fma += usize::from(t.contains("@llvm.fma."));
            }
            ck.reuse = usize::from(reused);
            runs.push((format!("{face} b={b}"), on, *face == "exact"));
        }
        if !runs.is_empty() {
            runs.push((format!("{face} off"), off, *face == "exact"));
        }
    }
    if runs.is_empty() {
        return ck;
    }
    let rr = run(&ir, 400_000_000);
    if !matches!(rr.outcome, Outcome::Done(_)) {
        return ck;
    }
    // The default face's reference beside the interpreter: its OFF arm, the
    // same realization of everything outside the units (a whole fold that
    // cancels may leave the interpreter's 1e-4 on either arm).
    let mut off_default: Option<String> = None;
    runs.sort_by_key(|(arm, _, _)| !arm.ends_with(" off"));
    for (arm, em, exact) in &runs {
        let ex = emu::build_emulated(
            clang,
            &em.text,
            em.device.as_deref().expect("a device unit"),
        );
        // Block threads serialized between barriers in thread order and in
        // reverse: a write and a read no barrier orders meet in one order.
        for mode in ["1", "2"] {
            let (got, _) = ex.run(&[("MAPAL_EMU_SERIAL", mode)]);
            let ok = match (exact, arm.ends_with(" off")) {
                (true, _) => got == rr.output,
                (false, true) => {
                    off_default.get_or_insert(got.clone());
                    true
                }
                (false, false) => {
                    within(&got, &rr.output)
                        || off_default.as_deref().is_some_and(|o| within(&got, o))
                }
            };
            if !ok {
                ck.findings.push(format!(
                    "chain#{z} {arm} serial={mode}: {got:?} vs {:?}\n{src}",
                    rr.output
                ));
            }
        }
    }
    ck
}

/// Generated chains (`testgen::chain_strategy`, and `wide_chain_strategy`'s
/// rows of up to 128 cells; neither designed on this rule), the epilogue rung
/// off so a tile with per-cell readers is the group's to take: every
/// row-block unit formed at one row a block and at the largest `b` (≤ 8) its
/// rows allow, on the exact face (= the interpreter) and the default face
/// (within 1e-4: tree folds, k-slice partials, FMA steps), each emulated with
/// its block's threads serialized between barriers in thread order and in
/// reverse (`MAPAL_EMU_SERIAL`: a missing barrier is a wrong value on every
/// run, not a race); each unit holds exactly the fact's members, and ON
/// launches no more than OFF. The paths the units take are counted and
/// required, not assumed.
#[test]
fn row_block_on_generated_chains() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let (narrow, wide) = (testgen::chain_strategy(), testgen::wide_chain_strategy());
    let mut srcs: Vec<String> = (0..400)
        .map(|_| narrow.new_tree(&mut runner).unwrap().current().source())
        .collect();
    srcs.extend((0..600).map(|_| wide.new_tree(&mut runner).unwrap().current().source()));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let checks: std::sync::Mutex<Vec<ChainCheck>> = Default::default();
    std::thread::scope(|s| {
        for _ in 0..emu::job_slots().max(1) {
            s.spawn(|| {
                loop {
                    let z = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(src) = srcs.get(z) else {
                        break;
                    };
                    let ck = check_chain(&clang, z, src);
                    checks.lock().unwrap().push(ck);
                }
            });
        }
    });
    let checks = checks.into_inner().unwrap();
    let sum = |f: fn(&ChainCheck) -> usize| checks.iter().map(f).sum::<usize>();
    let (groups, formed) = (sum(|c| c.groups), sum(|c| c.formed));
    let (wide, tree, split, fma, reuse) = (
        sum(|c| c.wide),
        sum(|c| c.tree),
        sum(|c| c.split),
        sum(|c| c.fma),
        sum(|c| c.reuse),
    );
    let mut why: std::collections::BTreeMap<String, usize> = Default::default();
    for c in &checks {
        for (k, v) in &c.why {
            *why.entry(k.clone()).or_insert(0) += v;
        }
    }
    let findings: Vec<&String> = checks.iter().flat_map(|c| &c.findings).collect();
    eprintln!(
        "row_block_on_generated_chains: {} programs, {groups} groups, {formed} formed, refused {why:?}; \
         units over several warps {wide}, tree folds {tree}, split tiles {split}, FMA {fma}; programs reusing a freed row buffer {reuse}",
        srcs.len()
    );
    assert!(
        findings.is_empty(),
        "{}",
        findings
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    for (n, what) in [
        (formed, "formed"),
        (wide, "a row group over several warps"),
        (tree, "a tree fold"),
        (split, "a split tile"),
        (fma, "an FMA step"),
        (reuse, "a freed row buffer reused"),
    ] {
        assert!(n > 0, "no generated unit took {what}");
    }
}

/// `ROWBLOCK_SRC=<file.mapal>` (`ROWBLOCK_B`): the program's values ON
/// (forced rows) and OFF against the interpreter, printed.
#[test]
#[ignore]
fn check_program_file() {
    let src =
        std::fs::read_to_string(std::env::var("ROWBLOCK_SRC").expect("ROWBLOCK_SRC")).unwrap();
    let b = std::env::var("ROWBLOCK_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let clang = emu::clang().expect("clang");
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let on = emission(&ir, "rtx-5080", EmitOpts::exact(), true, b);
    let off = emission(&ir, "rtx-5080", EmitOpts::exact(), false, 0);
    eprintln!("blocks ON: {:?}", block_lines(&on));
    eprintln!("interp: {:?}", rr.output);
    for (arm, em) in [("on", &on), ("off", &off)] {
        let got = emu::run_emulated(&clang, &em.text, em.device.as_deref().unwrap());
        eprintln!("{arm}: {got:?}");
    }
}

/// `ROWBLOCK_SRC=<file.mapal>`: print `unit_plan`'s groups.
#[test]
#[ignore]
fn print_plan_file() {
    let src =
        std::fs::read_to_string(std::env::var("ROWBLOCK_SRC").expect("ROWBLOCK_SRC")).unwrap();
    let ir = lower(&src);
    for g in &ir.unit_plan(main_of(&ir)).groups {
        eprintln!(
            "group rows={} row_bytes={} shared={}",
            g.rows, g.row_bytes, g.shared_bytes
        );
        for mb in &g.members {
            let k = match &mb.kind {
                UnitKind::Tile(t) => format!("tile c={} k={} a={:?} b={:?}", t.c, t.k, t.a, t.b),
                UnitKind::Fold(f) => format!("fold k={}", f.k),
                UnitKind::Cell => "cell".into(),
            };
            eprintln!(
                "  {:?} {k} w={} reads={:?} stored={} color={:?}",
                mb.out, mb.w, mb.reads, mb.stored, mb.color
            );
        }
    }
}
