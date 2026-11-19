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
/// the device), `flow` on, the row-block arm `on`, its rows forced to `b`
/// (0 = derive).
fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts, on: bool, b: u64) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.row_block = on;
    gpu.row_block_b = b;
    for r in ["place", "async", "pdl", "streams", "pinned"] {
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
        && g.iter()
            .zip(&w)
            .all(|(a, b)| match (a.parse::<f64>(), b.parse::<f64>()) {
                (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
                _ => a == b,
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
                let got = emu::run_emulated(&clang, &on.text, on.device.as_deref().unwrap());
                match face {
                    "exact" => assert_eq!(got, rr.output, "b={b} exact"),
                    _ => assert!(
                        within(&got, &rr.output),
                        "b={b}: {got:?} vs {:?}",
                        rr.output
                    ),
                }
            }
        }
    }
}

/// The rule on the real shape, `pt_attn_1024_64` on the rtx-5080: the group
/// fits (b = 16, 84 KB) but the plan model keeps the boundary — the unit reads
/// Kᵀ straight from L2 at half a sector a lane, which the split's tile site
/// stages (measured: the fused kernel 33.3 us against 29.8 us for the three,
/// `compile-time-graph.md`). Forced, it is one launch storing `o` alone.
#[test]
fn attn_1024_on_rtx_5080_keeps_the_boundary_by_cost() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../benches/algorithms/pt_attn_1024_64.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    // The shipped arms (the emulator's rung offs change the split's plans).
    let shipped = |b: u64| {
        let mut gpu = GpuArms::default();
        gpu.row_block_b = b;
        let opts = EmitOpts {
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        };
        emit_detailed(&ir, &opts).expect("emits")
    };
    let on = shipped(0);
    assert!(block_lines(&on).is_empty());
    assert!(on.text.contains(" row_block=s:cost"), "{}", on.text);
    let forced = shipped(16);
    let lines = block_lines(&forced);
    assert_eq!(lines.len(), 1, "{}", forced.text);
    assert_eq!(launches(&forced.text), 1);
    assert_eq!(field(&lines[0], "store"), "o");
    assert_eq!(field(&lines[0], "smem"), "83968");
}

/// Generated chains (`testgen::chain_strategy`, not designed on this rule):
/// every row-block group `unit_plan` finds, forced to the largest `b` its rows
/// allow, gives the interpreter's values on the exact face, and each formed
/// group is one launch holding its tiles (the site calls drop by its tiles).
#[test]
fn row_block_on_generated_chains() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let strat = testgen::chain_strategy();
    let (mut groups, mut formed, mut findings) = (0, 0, Vec::new());
    let mut why: std::collections::BTreeMap<String, usize> = Default::default();
    for z in 0..400 {
        let src = strat.new_tree(&mut runner).unwrap().current().source();
        let ir = lower(&src);
        let plan = ir.unit_plan(main_of(&ir));
        if plan.groups.is_empty() {
            continue;
        }
        groups += plan.groups.len();
        let b = plan
            .groups
            .iter()
            .map(|g| 1u64 << g.rows.trailing_zeros().min(3))
            .min()
            .unwrap();
        let on = emission(&ir, "rtx-5080", EmitOpts::exact(), true, b);
        let off = emission(&ir, "rtx-5080", EmitOpts::exact(), false, 0);
        let lines = block_lines(&on);
        for l in on.text.lines().filter(|l| l.contains("mapal.flow.host")) {
            for f in l
                .split_whitespace()
                .filter_map(|f| f.strip_prefix("row_block="))
            {
                *why.entry(f.split(':').nth(1).unwrap_or("").to_string())
                    .or_insert(0) += 1;
            }
        }
        if lines.is_empty() {
            continue;
        }
        formed += lines.len();
        // Each formed group is one launch holding exactly the fact's members
        // (its tiles included), and ON launches no more than OFF (whose flow
        // units may hold a group member beside other cells).
        let sizes: Vec<usize> = plan.groups.iter().map(|g| g.members.len()).collect();
        for l in &lines {
            let n = field(l, "members").split(';').count();
            let tiles: usize = field(l, "tiles").parse().unwrap();
            if !sizes.contains(&n) || tiles == 0 {
                findings.push(format!(
                    "chain#{z}: unit of {n} members ({tiles} tiles) vs groups {sizes:?}\n{l}"
                ));
            }
        }
        let (l_on, l_off) = (launches(&on.text), launches(&off.text));
        if l_on > l_off {
            findings.push(format!(
                "chain#{z}: launches ON {l_on} > OFF {l_off}\n{src}"
            ));
        }
        let rr = run(&ir, 400_000_000);
        if !matches!(rr.outcome, Outcome::Done(_)) {
            continue;
        }
        for (arm, em) in [("on", &on), ("off", &off)] {
            let got = emu::run_emulated(
                &clang,
                &em.text,
                em.device.as_deref().expect("a device unit"),
            );
            if got != rr.output {
                findings.push(format!(
                    "chain#{z} {arm}: {got:?} vs {:?}\n{src}",
                    rr.output
                ));
            }
        }
    }
    assert!(findings.is_empty(), "{}", findings.join("\n"));
    eprintln!("row_block_on_generated_chains: {groups} groups, {formed} formed, refused {why:?}");
    assert!(formed > 0, "no generated chain formed a row-block unit");
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
