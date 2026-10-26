//! site-geometry: one fill decision for every tile site — `fill-wave` (a
//! grid with fewer warps than the fill rule's target cuts its free fold into
//! the fewest partitions that fill it, counted at the tile's own residency)
//! and `row-split` (a retained site's one-row pass takes that chooser) — on
//! programs from the generator (`testgen::geom_prog_strategy`, which the
//! rules were not designed on): counted from the emission against the OFF
//! arm, and the values through the GPU emulator, ON = OFF byte for byte
//! (integer-valued operands: every cut of the fold is exact).

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed, resolve_profile};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The default face (free order: the splits), the rungs the emulator cannot
/// run off (`async`, `pdl`, `streams`), `place` off (every capable site on
/// the device); `on` false = the site-geometry rungs' OFF arm.
fn emission(ir: &CategoryIr, target: &'static str, on: bool) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams", "place"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in ["fill-wave", "row-split"] {
        *(if on { gpu.on.rung(r) } else { gpu.off.rung(r) }).expect("rung") = true;
    }
    let opts = EmitOpts { host: "generic", target, gpu, ..EmitOpts::default() };
    emit_detailed(ir, &opts).expect("emits")
}

/// `(rows, blocks × S, warps per block, S)` of every split tile kernel's
/// `; mapal.plan` line.
fn splits(dev: Option<&str>) -> Vec<(u64, u64, u64, u64)> {
    let field = |l: &str, k: &str| -> u64 {
        l.split(' ').find_map(|f| f.strip_prefix(k)).and_then(|v| v.parse().ok()).unwrap_or(0)
    };
    dev.unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.plan") && (l.contains("combine=atomic") || l.contains("combine=splitk")))
        .map(|l| {
            let s = field(l, "S=");
            (field(l, "rows="), field(l, "gx=") * field(l, "gy=") * s, field(l, "block=").div_ceil(32), s)
        })
        .collect()
}

#[test]
fn fill_wave_and_row_split_counted_and_valued_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::geom_prog_strategy();
    let clang = emu::clang();
    let targets = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];
    let (mut filled, mut short, mut rows, mut same, mut valued) = (0, 0, 0, 0, 0);
    for z in 0..40 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let (src, (m, ..)) = testgen::geom_src(&p);
        let ir = lower(&src);
        let target = targets[z % targets.len()];
        let fill = resolve_profile(target).and_then(|t| t.gpu()).expect("a GPU part").fill_warps();
        let (on, off) = (emission(&ir, target, true), emission(&ir, target, false));
        let (s_on, s_off) = (splits(on.device.as_deref()), splits(off.device.as_deref()));
        // OFF never cuts a retained row: its row kernel is the ordered one.
        assert!(!p.retained || s_off.iter().all(|s| s.0 > 1 || m == 1), "#{z} {target}\n{src}");
        for &(r, grid, warps, s) in &s_on {
            assert!(s > 1, "#{z}");
            if p.retained && r == 1 {
                rows += 1;
            }
            // An under-filled grid: the fewest partitions that fill the part
            // (half as many would not), or, a fold too short, as many as fit.
            if (grid / s) * warps >= fill {
                continue;
            }
            if grid * warps >= fill {
                filled += 1;
                assert!(s % 2 == 1 || (grid / 2) * warps < fill, "#{z} {target} S={s}\n{src}");
            } else {
                short += 1;
            }
        }
        if s_on == s_off && on.text == off.text {
            same += 1;
        }
        if let Some(cc) = &clang {
            let dev = |e: &mapal_backend_llvm::Emission| e.device.clone().unwrap_or_default();
            let (x, y) = (emu::run_emulated(cc, &on.text, &dev(&on)), emu::run_emulated(cc, &off.text, &dev(&off)));
            assert_eq!(x.lines().count(), 3, "#{z}: the program printed\n{x}\n{src}");
            assert_eq!(x, y, "#{z} {target} values\n{src}");
            valued += 1;
        }
    }
    eprintln!("site-geometry: {filled} filled splits, {short} short, {rows} row splits, {same} unchanged, {valued} valued");
    assert!(filled >= 8, "the generator reaches fill-wave ({filled})");
    assert!(rows >= 4, "the generator reaches row-split ({rows})");
}

/// `(kernel, rows, blocks, block, tm, tn, ty)` of every FMA tile kernel.
fn tiles(dev: Option<&str>) -> Vec<(String, u64, u64, u64, u64, u64, u64)> {
    let field = |l: &str, k: &str| -> u64 {
        l.split(' ').find_map(|f| f.strip_prefix(k)).and_then(|v| v.parse().ok()).unwrap_or(0)
    };
    dev.unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.plan") && l.contains(" kind=tile "))
        .map(|l| {
            let name = l.split(' ').find_map(|f| f.strip_prefix("kernel=")).unwrap_or("").to_string();
            let f = |k| field(l, k);
            (name, f("rows="), f("gx=") * f("gy="), f("block="), f("tm="), f("tn="), f("ty="))
        })
        .collect()
}

/// site-geometry `fit-first` (R1) on convolutions and products from the
/// generator: where it moves a tile, the thread grid fits the rows, every SM
/// has a block, the register tile and the threads over the grid are OFF's —
/// only fewer, larger blocks — and the values are OFF's (emulated).
#[test]
fn fit_first_counted_and_valued_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let (radix, geom) = (testgen::radix_prog_strategy(), testgen::geom_prog_strategy());
    let clang = emu::clang();
    let targets = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];
    let (mut moved, mut same, mut valued) = (0, 0, 0);
    for z in 0..40 {
        let src = match z % 2 {
            0 => testgen::radix_src(&radix.new_tree(&mut runner).unwrap().current()).0,
            // (A few rows over many lanes: the shape the rule moves.)
            _ => {
                let p = geom.new_tree(&mut runner).unwrap().current();
                testgen::geom_src(&testgen::GeomProg { wide: true, retained: false, ..p }).0
            }
        };
        let ir = lower(&src);
        let target = targets[z / 2 % targets.len()];
        let sms = resolve_profile(target).and_then(|t| t.gpu()).expect("a GPU part").sms;
        let at = |on: bool| {
            let mut gpu = GpuArms::default();
            for r in ["async", "pdl", "streams", "place"] {
                *gpu.off.rung(r).expect("rung") = true;
            }
            *(if on { gpu.on.rung("fit-first") } else { gpu.off.rung("fit-first") }).expect("rung") = true;
            emit_detailed(&ir, &EmitOpts { host: "generic", target, gpu, ..EmitOpts::exact() }).expect("emits")
        };
        let (on, off) = (at(true), at(false));
        if on.device == off.device {
            same += 1;
            continue;
        }
        moved += 1;
        let offs = tiles(off.device.as_deref());
        for (name, rows, grid, block, tm, tn, ty) in tiles(on.device.as_deref()) {
            let Some(o) = offs.iter().find(|o| o.0 == name) else { continue };
            if (grid, block, ty) == (o.2, o.3, o.6) {
                continue;
            }
            assert!(ty * tm <= rows && grid >= sms, "#{z} {target} {name}\n{src}");
            // OFF's halving lost the register tile (`cell-fill`'s 1 × 1), or
            // kept it on fewer threads a block.
            if (o.4, o.5) != (1, 1) {
                assert_eq!((tm, tn, grid * block), (o.4, o.5, o.2 * o.3), "#{z} {target} {name}\n{src}");
                assert!(block > o.3, "#{z} {target} {name}: larger blocks\n{src}");
            }
        }
        if let Some(cc) = &clang {
            let dev = |e: &mapal_backend_llvm::Emission| e.device.clone().unwrap_or_default();
            let (x, y) = (emu::run_emulated(cc, &on.text, &dev(&on)), emu::run_emulated(cc, &off.text, &dev(&off)));
            assert_eq!(x, y, "#{z} {target} values\n{src}");
            valued += 1;
        }
    }
    eprintln!("fit-first: {moved} moved, {same} unchanged, {valued} valued");
    assert!(moved >= 4, "the generator reaches fit-first ({moved})");
}
