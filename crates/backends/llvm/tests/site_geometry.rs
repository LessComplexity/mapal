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
        launches_give_planned_smem(&on, &format!("#{z} {target}\n{src}"));
        // The re-sized split kernel still walks each partition's `k / S` in
        // whole panels (`fill-wave` keeps the unsplit launch otherwise).
        for l in on.device.as_deref().unwrap_or("").lines().filter(|l| l.starts_with("; mapal.plan") && l.contains(" kind=tile ")) {
            let f = |k: &str| -> u64 { l.split(' ').find_map(|x| x.strip_prefix(k)).and_then(|v| v.parse().ok()).unwrap_or(0) };
            if f("S=") > 1 {
                assert!((f("k=") / f("S=")).is_multiple_of(f("bk=")), "#{z} {target}: bk | k/S\n{l}\n{src}");
            }
        }
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

/// Every tile launch (`launch`, `launchv`) gives its kernel the dynamic
/// shared memory the kernel's `; mapal.plan` line was sized for (the smoke
/// box: an epilogue group's ordered kernel launched with a split tile's).
fn launches_give_planned_smem(e: &mapal_backend_llvm::Emission, ctx: &str) {
    let dev = e.device.as_deref().unwrap_or("");
    let planned = |id: i64| {
        let name = format!("kernel=mapal_tile_{:016x} ", id as u64);
        dev.lines().find(|l| l.starts_with("; mapal.plan") && l.contains(&name)).and_then(|l| {
            l.split(' ').find_map(|f| f.strip_prefix("smem=")).and_then(|v| v.parse::<u64>().ok())
        })
    };
    for l in e.text.lines().filter(|l| l.contains("call void @mapal_gpu_launch")) {
        let args = l.split('(').nth(1).unwrap_or("").split(')').next().unwrap_or("");
        let id = args.split(", ").next().and_then(|a| a.strip_prefix("i64 ")).and_then(|v| v.parse::<i64>().ok());
        let smem = args.rsplit(", ").next().and_then(|a| a.strip_prefix("i32 ")).and_then(|v| v.parse::<u64>().ok());
        if let (Some(smem), Some(want)) = (smem, id.and_then(planned)) {
            assert_eq!(smem, want, "launch smem != plan smem\n{l}\n{ctx}");
        }
    }
}

/// The smoke box's failure, on its shape: a resident two-layer chain whose
/// middle product feeds a bias + relu epilogue group (an ordered launch)
/// while `fill-wave` cuts the fold of a sibling site. Every launch carries the
/// shared memory its kernel was sized for, on every part.
#[test]
fn epilogue_launch_keeps_its_own_tile_beside_a_split() {
    let src = "fn main() {
    32768 -> iota -> tx;
    65536 -> iota -> two;
    131072 -> iota -> tup;
    131072 -> iota -> tdn;
    65536 -> iota -> th;
    256 -> iota -> dr;
    512 -> iota -> hr;
    tx -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> x;
    two -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 256.0 } -> wo;
    tup -> map { t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 256.0 } -> wup;
    tdn -> map { t -> (t * 3 + 1) % 89 - 44 -> widen_f32 -> v; v / 512.0 } -> wdn;
    hr -> map { t -> t % 13 - 6 -> widen_f32 -> v; v / 5.0 } -> b1;
    tx -> map { t -> t / 256 -> i; t % 256 -> j; (0.0, dr) -> fold { acc, k -> acc + x[i * 256 + k] * wo[k * 256 + j] } } -> p;
    th -> map { t -> t / 512 -> i; t % 512 -> j; (0.0, dr) -> fold { acc, k -> acc + p[i * 256 + k] * wup[k * 512 + j] } } -> hraw;
    th -> map { t -> hraw[t] + b1[t % 512] -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> r; r } -> hh;
    tx -> map { t -> t / 256 -> i; t % 256 -> j; (0.0, hr) -> fold { acc, k -> acc + hh[i * 512 + k] * wdn[k * 256 + j] } } -> m;
    m[0] -> println;
    m[32767] -> println;
}
";
    let ir = lower(src);
    let mut split = 0;
    for target in ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"] {
        let e = emit_detailed(&ir, &EmitOpts { host: "generic", target, ..EmitOpts::default() }).expect("emits");
        split += e.device.as_deref().unwrap_or("").matches("combine=atomic").count();
        launches_give_planned_smem(&e, target);
    }
    assert!(split > 0, "a sibling site's fold is cut somewhere");
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

/// The candidates `benches/gpu/knobs.json` lists for `knob` on `arch`.
fn candidates(arch: &str, knob: &str) -> Vec<u64> {
    let json = include_str!("../../../../benches/gpu/knobs.json");
    let row = json.split(&format!("\"{arch}\":")).nth(1).and_then(|r| r.split('}').next()).expect("an arch row");
    let list = row.split(&format!("\"{knob}\": [")).nth(1).and_then(|r| r.split(']').next()).expect("a knob");
    list.split(',').map(|v| v.trim().parse().expect("a count")).collect()
}

/// ADR-0034 D4 (site-geometry B4): every knob candidate the offline search
/// times is value-invariant. On products from the generator, one candidate
/// per program (`--geom=<knob>=<value>`, the search's own flag) prints the
/// table row's values in the emulator; counted: candidates that moved a
/// kernel. (A `cp.async` kernel — forced stages — is counted, not emulated.)
#[test]
fn knob_candidates_value_invariant_on_unseen_programs() {
    const KNOBS: [&str; 9] = ["resident", "split", "tm", "tn", "warps", "bk", "stages", "wg_n", "wg_groups"];
    let mut runner = TestRunner::deterministic();
    let strat = testgen::geom_prog_strategy();
    let clang = emu::clang();
    let targets = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];
    let (mut moved, mut valued) = (0, 0);
    for z in 0..48usize {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::geom_src(&testgen::GeomProg { wide: false, ..p.clone() }).0;
        let ir = lower(&src);
        let target = targets[z % targets.len()];
        let knob = KNOBS[z / targets.len() % KNOBS.len()];
        let cands = candidates(target, knob);
        let Some(&v) = cands.get(1 + usize::from(p.law[0]) % cands.len().saturating_sub(1).max(1)) else {
            continue;
        };
        let at = |forced: bool| {
            let mut opts = EmitOpts { host: "generic", target, ..EmitOpts::default() };
            for r in ["async", "pdl", "streams", "place"] {
                *opts.gpu.off.rung(r).expect("rung") = true;
            }
            if forced {
                assert!(mapal_backend_llvm::flags::apply(&mut opts, &format!("--geom={knob}={v}")).expect("a knob"));
            }
            emit_detailed(&ir, &opts).expect("emits")
        };
        let (on, off) = (at(true), at(false));
        if on.device == off.device {
            continue;
        }
        moved += 1;
        let dev = |e: &mapal_backend_llvm::Emission| e.device.clone().unwrap_or_default();
        if let (Some(cc), false) = (&clang, dev(&on).contains("call void @llvm.nvvm.cp.async")) {
            let (x, y) = (emu::run_emulated(cc, &on.text, &dev(&on)), emu::run_emulated(cc, &off.text, &dev(&off)));
            assert_eq!(x, y, "#{z} {target} --geom={knob}={v}\n{src}");
            valued += 1;
        }
    }
    eprintln!("knobs: {moved} candidates moved a kernel, {valued} valued");
    assert!(moved >= 8, "the candidates reach the kernels ({moved})");
}

/// site-geometry `tf32-radix` (R2) on convolutions from the generator whose
/// fold admits TF32 (`: tf32 | f32` on the step): counted from the emission —
/// ON a radix site takes the `mma.sync` tile (never `wgmma`), its digit-affine
/// operands copied by 4-byte `cp.async`; OFF the FFMA radix tile; ON = OFF on
/// a program without a radix site — and every ON module assembles to PTX
/// (`llc`, when found). The tensor core's values are the box's value gate
/// (the emulator runs no `mma.sync`).
#[test]
fn tf32_radix_counted_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::radix_prog_strategy();
    let llc = std::process::Command::new("llc").arg("--version").output().is_ok();
    let targets = [("rtx-4090", "sm_89"), ("h100-pcie", "sm_90a"), ("rtx-5090", "sm_120"), ("rtx-3090", "sm_86")];
    let (mut tensor, mut ffma, mut assembled) = (0, 0, 0);
    // (Few generated convolutions fit a tensor tile: `k = cin·taps` a whole
    // number of `mma` steps, `cout` whole m16 tiles.)
    for z in 0..128 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::radix_src(&p).0.replacen("(0.0, k_y) -> fold {", "(0.0, k_y) -> fold: tf32 | f32 {", 1);
        let ir = lower(&src);
        let (target, sm) = targets[z % targets.len()];
        let at = |on: bool| {
            let mut gpu = GpuArms::default();
            *gpu.off.rung("place").expect("rung") = true;
            *(if on { gpu.on.rung("tf32-radix") } else { gpu.off.rung("tf32-radix") }).expect("rung") = true;
            emit_detailed(&ir, &EmitOpts { host: "generic", target, gpu, ..EmitOpts::default() })
                .expect("emits")
                .device
                .unwrap_or_default()
        };
        let (on, off) = (at(true), at(false));
        let radix = |d: &str, kind: &str| {
            d.lines().filter(|l| l.starts_with("; mapal.plan") && l.contains(" radix=") && l.contains(kind)).count()
        };
        assert_eq!(radix(&off, " kind=mma ") + radix(&off, " kind=wgmma "), 0, "#{z} {target} OFF\n{src}");
        assert_eq!(radix(&on, " kind=wgmma "), 0, "#{z} {target}: no wgmma radix tile\n{src}");
        if radix(&on, " kind=mma ") > 0 {
            tensor += 1;
            assert!(on.contains("call void @llvm.nvvm.cp.async.ca.shared.global.4("), "#{z} {target}\n{src}");
            if llc {
                let dir = tempfile::tempdir().unwrap();
                let ll = dir.path().join("d.ll");
                std::fs::write(&ll, &on).unwrap();
                let st = std::process::Command::new("llc")
                    .args(["-mtriple=nvptx64-nvidia-cuda", &format!("-mcpu={sm}"), "-mattr=+ptx87", "-o", "/dev/null"])
                    .arg(&ll)
                    .status()
                    .unwrap();
                assert!(st.success(), "#{z} {target}: llc\n{src}");
                assembled += 1;
            }
        } else {
            ffma += 1;
            if radix(&off, " kind=tile ") == 0 {
                assert_eq!(on, off, "#{z} {target}: no radix tile, no change\n{src}");
            }
        }
    }
    eprintln!("tf32-radix: {tensor} tensor-core, {ffma} not, {assembled} assembled");
    assert!(tensor >= 8, "the generator reaches the radix mma tile ({tensor})");
}
