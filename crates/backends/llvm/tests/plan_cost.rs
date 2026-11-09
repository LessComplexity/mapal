//! plan-cost (design §5): the cost model's picks on programs from the
//! generator (`testgen::chain_strategy`, `geom_prog_strategy`), which it was
//! not designed on — parts without plan rates emit the rules' text byte for
//! byte (D8), and on the calibrated parts every emitted program answers the
//! interpreter within ADR-0040's bound through the GPU emulator.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// `base` on `target` with `off` rungs off, and the ones the emulator cannot
/// run (`pdl`, `streams`, `pinned`; `cp.async` it queues, tile-efficiency
/// T2); `place` off puts every capable site on the device.
fn emission(ir: &CategoryIr, target: &'static str, off: &[&str], base: EmitOpts) -> Emission {
    let mut gpu = GpuArms::default();
    for r in off.iter().chain(&["place", "pdl", "streams", "pinned"]) {
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

/// Generated chains (the `linear_plan` sum node among them), tile-geometry
/// programs and the `_512` chains.
fn programs(chains: usize, geoms: usize) -> Vec<String> {
    let mut runner = TestRunner::deterministic();
    let (cs, gs) = (testgen::chain_strategy(), testgen::geom_prog_strategy());
    let mut out: Vec<String> = (0..chains)
        .map(|_| cs.new_tree(&mut runner).unwrap().current().source())
        .collect();
    out.extend(
        (0..geoms).map(|_| testgen::geom_src(&gs.new_tree(&mut runner).unwrap().current()).0),
    );
    out
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

/// D8: a part without plan rates keeps today's picks — the `plan-cost` ON
/// arm's emission equals its OFF arm's, host and device, both faces.
#[test]
fn uncalibrated_parts_emit_the_rules_byte_for_byte() {
    let mut srcs = programs(48, 24);
    for p in ["pt_indep_512", "pt_mlp_512", "pt_diamond_512"] {
        srcs.push(
            std::fs::read_to_string(format!("../../../benches/algorithms/{p}.mapal"))
                .expect("bench"),
        );
    }
    let mut device = 0;
    for src in &srcs {
        let ir = lower(src);
        for target in ["h100-pcie", "rtx-3090", "a100"] {
            for base in [EmitOpts::default(), EmitOpts::exact()] {
                let on = emission(&ir, target, &[], base);
                let off = emission(&ir, target, &["plan-cost"], base);
                assert_eq!(on.text, off.text, "{target} host\n{src}");
                assert_eq!(on.device, off.device, "{target} device\n{src}");
                assert!(
                    !on.device.as_deref().unwrap_or("").contains(" pick="),
                    "{target}"
                );
                device += usize::from(on.device.is_some());
            }
        }
    }
    assert!(device >= 100, "programs with a device unit: {device}");
}

/// On rtx-4090 and rtx-5090 every tile kernel's plan line carries the pick,
/// and the default face's values answer the interpreter within the bound,
/// the cost model's picks and the rules' alike.
#[test]
fn calibrated_parts_values_within_the_bound() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP plan_cost values: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let (mut checked, mut moved, mut picks) = (0, 0, 0);
    for (z, src) in programs(96, 0).iter().enumerate() {
        let ir = lower(src);
        let rr = run(&ir, 400_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        let target = if z % 2 == 0 { "rtx-4090" } else { "rtx-5090" };
        let on = emission(&ir, target, &[], EmitOpts::default());
        let off = emission(&ir, target, &["plan-cost"], EmitOpts::default());
        let Some(dev) = &on.device else {
            continue;
        };
        picks += dev
            .lines()
            .filter(|l| l.starts_with("; mapal.plan") && l.contains(" pick="))
            .count();
        moved += usize::from(on.text != off.text || on.device != off.device);
        let got = emu::run_emulated(&clang, &on.text, dev);
        assert!(
            within(&got, &rr.output),
            "#{z} {target}: {got:?} != {:?}\n{src}",
            rr.output
        );
        checked += 1;
    }
    eprintln!(
        "plan-cost values: {checked} programs, {moved} emit differently from the rules, {picks} picks"
    );
    assert!(
        checked >= 32 && picks >= 32,
        "{checked} programs, {picks} picks"
    );
}

/// A sum of `t` folds `x_i · y_i` over one `k` range of an `m × n` grid;
/// `share` 1 / 2: every term reads one `a` / one `b` (diamond's shape).
fn sum_src(m: u64, n: u64, k: u64, t: usize, share: u8) -> String {
    let mut s = format!(
        "fn main() {{\n    {} -> iota -> ta;\n    {k} -> iota -> kr;\n",
        m * n
    );
    s += &format!(
        "    {} -> iota -> tak;\n    {} -> iota -> tkb;\n",
        m * k,
        k * n
    );
    for i in 0..t {
        let (p, q) = (7 + 2 * i, 11 + 2 * i);
        s += &format!(
            "    tak -> map {{ t -> (t * {p} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a{i};\n"
        );
        s += &format!(
            "    tkb -> map {{ t -> (t * {q} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b{i};\n"
        );
    }
    for i in 0..t {
        let a = if share == 1 { 0 } else { i };
        let b = if share == 2 { 0 } else { i };
        s += &format!(
            "    ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> acc + a{a}[i * {k} + k] * b{b}[k * {n} + j] }} }} -> c{i};\n"
        );
    }
    let sum: Vec<String> = (0..t).map(|i| format!("c{i}[t]")).collect();
    s += &format!("    ta -> map {{ t -> {} }} -> g;\n", sum.join(" + "));
    // Cells, not a checksum: a whole fold's own reassociation is not this
    // site's bound.
    for cell in [0, m * n / 3, m * n / 2 + 1, m * n - 1] {
        s += &format!("    g[{cell}] -> println;\n");
    }
    s + "}\n"
}

/// plan-cost `linear` (§4a concat): where the cost model takes the site
/// table, one zero and one launch replace the terms' launches, and the sum
/// answers the interpreter within the bound; the rung's OFF arm launches the
/// terms. Sizes drawn per program (skinny, long `k`: the terms split).
#[test]
fn site_tables_launch_once_and_keep_the_values() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site tables: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let dims = (1u64..17, 0usize..3, 0usize..3, 2usize..4, 0u8..3);
    let launches = |e: &Emission| {
        e.text
            .lines()
            .filter(|l| {
                l.contains("call void @mapal_gpu_launch") || l.contains("call void @mapal_gpu_zero")
            })
            .count()
    };
    let (mut tables, mut runs) = (0, 0);
    for z in 0..24 {
        let (m, ni, ki, t, share) = dims.new_tree(&mut runner).unwrap().current();
        let (n, k) = ([32, 64, 128][ni], [256, 512, 1024][ki]);
        let src = sum_src(m, n, k, t, share);
        let ir = lower(&src);
        assert_eq!(ir.linear_plan(ir.entry()).len(), 1, "{src}");
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        let target = if z % 2 == 0 { "rtx-4090" } else { "rtx-5090" };
        let on = emission(&ir, target, &[], EmitOpts::default());
        let off = emission(&ir, target, &["linear"], EmitOpts::default());
        let dev = on.device.as_deref().expect("a device unit");
        let table = dev.contains(&format!(" terms={t}\n"));
        if table {
            tables += 1;
            assert_eq!(launches(&on), 2, "a site table: zero + one launch\n{src}");
            assert!(launches(&off) >= t, "OFF launches every term\n{src}");
        }
        assert!(!off.device.as_deref().unwrap_or("").contains(" terms="));
        for (arm, e) in [("on", &on), ("off", &off)] {
            let got = emu::run_emulated(&clang, &e.text, e.device.as_ref().expect("device"));
            assert!(
                within(&got, &rr.output),
                "{arm} {target}: {got:?} != {:?}\n{src}",
                rr.output
            );
        }
        runs += 1;
    }
    eprintln!("site tables: {tables} of {runs} sums");
    assert!(tables >= 8, "site tables taken: {tables} of {runs}");
}

/// The plan-cost blind set (`MAPAL_BLIND=<dir> cargo test -- --ignored
/// write_blind_set`): 40 `testgen::tile_prog_strategy` programs from a seed no
/// fit used, each product 0.01–5 ms at a 4090's ~40 TFLOP/s and ~0.9 TB/s, as
/// `<dir>/b<i>.mapal`, the same program at interpreter sizes as
/// `<dir>/small/b<i>.mapal`, and each product's `m×n×k` in `<dir>/sites.tsv`
/// (`mapal fmt` the programs after).
#[test]
#[ignore]
fn write_blind_set() {
    use proptest::test_runner::{Config, RngAlgorithm, TestRng};
    let dir = std::env::var("MAPAL_BLIND").expect("MAPAL_BLIND=<dir>");
    std::fs::create_dir_all(format!("{dir}/small")).unwrap();
    let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &[0xb1; 32]);
    let mut runner = TestRunner::new_with_rng(Config::default(), rng);
    let ms = |(m, n, k): (u64, u64, u64)| {
        let flops = 2.0 * (m * n * k) as f64 / 40e12;
        flops.max(4.0 * (m * k + k * n + m * n) as f64 / 0.9e12) * 1e3
    };
    let (strat, mut tsv) = (
        testgen::tile_prog_strategy(),
        String::from("prog\tkind\tsites\n"),
    );
    let mut i = 0;
    while i < 40 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let (src, sites) = testgen::tile_src(&p, false);
        if !sites.iter().all(|&s| (0.01..=5.0).contains(&ms(s))) {
            continue;
        }
        std::fs::write(format!("{dir}/b{i:02}.mapal"), src).unwrap();
        std::fs::write(
            format!("{dir}/small/b{i:02}.mapal"),
            testgen::tile_src(&p, true).0,
        )
        .unwrap();
        let s: Vec<String> = sites
            .iter()
            .map(|(m, n, k)| format!("{m}x{n}x{k}"))
            .collect();
        tsv += &format!("b{i:02}\t{}\t{}\n", p.kind, s.join(","));
        i += 1;
    }
    std::fs::write(format!("{dir}/sites.tsv"), tsv).unwrap();
}
