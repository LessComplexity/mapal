//! recurrence-precision `tf32-tile` (ADR-0041 D6): a tile site whose product
//! admits `tf32` (`precision_plan.set`, from the program's `: compute`) takes
//! the tensor-core kernel — `mma.sync` sm_80+, `wgmma` sm_90a. On programs
//! from the generator (`testgen::tf32_prog_strategy`, which the rung was not
//! designed on): counted from the emission against its OFF arm and against
//! the same program without the type; values from the interpreter's two
//! sides (f32 and `--weakest`) within D3's bound against each cell's `M`.

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, bound_coef};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The default face on `target` (`place` off: every capable site on the
/// device), `tf32-tile` on or off.
fn device(ir: &CategoryIr, target: &'static str, on: bool) -> String {
    let mut gpu = GpuArms::default();
    *gpu.off.rung("place").expect("rung") = true;
    *(if on { gpu.on.rung("tf32-tile") } else { gpu.off.rung("tf32-tile") }).expect("rung") = true;
    let opts = EmitOpts { host: "generic", target, gpu, ..EmitOpts::default() };
    emit_detailed(ir, &opts).expect("emits").device.unwrap_or_default()
}

/// The tensor-core kernels a module launches.
fn tensor_kernels(dev: &str) -> usize {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan") && (l.contains(" kind=mma ") || l.contains(" kind=wgmma ")))
        .count()
}

/// The module's plan lines without kernel names or the site's product set
/// (`set=`): what runs.
fn shape(dev: &str) -> Vec<String> {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan"))
        .map(|l| {
            let keep = |w: &&str| !w.starts_with("kernel=") && !w.starts_with("set=");
            l.split(' ').filter(keep).collect::<Vec<_>>().join(" ")
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace().map(|x| x.parse().expect("a float")).collect()
}

/// `precision_plan.bound` (D3 / ADR-0040 D2): the product's tile site carries
/// the coefficient of its set's weakest member at its fold length.
#[test]
fn precision_plan_bound_per_site() {
    for (form, tf) in [(0u8, true), (1, true), (2, false), (3, false)] {
        let p = testgen::Tf32Prog { dims: [1, 1, 2], bt: false, form, law: [0, 0] };
        let (src, (_, k)) = testgen::tf32_src(&p);
        let ir = lower(&src);
        let bounds: Vec<f64> = ir.funcs().flat_map(|(f, _)| ir.precision_plan(f).bound.values().copied().collect::<Vec<_>>()).collect();
        assert!(bounds.contains(&bound_coef(k, tf)), "form {form}: {bounds:?}\n{src}");
        let set: usize = ir.funcs().map(|(f, _)| ir.precision_plan(f).set.len()).sum();
        assert_eq!(set > 0, tf, "form {form}: set holds the tf32 products only");
    }
}

#[test]
fn tf32_tile_counted_and_bounded_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::tf32_prog_strategy();
    let targets = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];
    let (mut tensor, mut ffma, mut off_forms, mut valued) = (0, 0, 0, 0);
    for z in 0..40 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let (src, (work, k)) = testgen::tf32_src(&p);
        let ir = lower(&src);
        let target = targets[z % targets.len()];
        let (on, off) = (device(&ir, target, true), device(&ir, target, false));
        let n_on = tensor_kernels(&on);
        // OFF: no tensor core, the same kernels as the program without a type.
        assert_eq!(tensor_kernels(&off), 0, "#{z} {target} OFF\n{src}");
        let plain = testgen::Tf32Prog { form: 3, ..p.clone() };
        let base = device(&lower(&testgen::tf32_src(&plain).0), target, true);
        assert_eq!(shape(&off), shape(&base), "#{z} {target}: OFF is the f32 tile\n{src}");
        if !testgen::tf32_admits(&p) {
            off_forms += 1;
            assert_eq!(n_on, 0, "#{z}: no `tf32` in the set, no tensor core\n{src}");
            assert_eq!(on, base, "#{z}: an f32-only program is unchanged\n{src}");
        } else if n_on > 0 {
            tensor += 1;
            let mut tensor_lines = on.lines().filter(|l| l.contains(" kind=mma ") || l.contains(" kind=wgmma "));
            assert!(tensor_lines.all(|l| l.contains(" set=tf32|f32 ")), "#{z} {target}: the plan names the set\n{src}");
            for (kind, op) in [(" kind=mma ", "mma.sync"), (" kind=wgmma ", "wgmma.mma_async")] {
                if on.contains(kind) {
                    assert!(on.contains(op), "#{z} {target}: {op}\n{src}");
                }
            }
            if target != "h100-pcie" {
                assert!(!on.contains(" kind=wgmma "), "#{z} {target}: wgmma is sm_90a's");
            }
        } else {
            ffma += 1;
        }
        // D3 from both sides: f32 and weakest each within their bound of the
        // exact value, so within the sum of the two against each cell's M.
        // (The debug interpreter: products up to 2^17 multiply-adds.)
        if work > 1 << 17 {
            continue;
        }
        valued += 1;
        let f = floats(&mapal_interp::run(&ir, 1 << 34).output);
        let w = floats(&mapal_interp::run_weakest(&ir, 1 << 34, &[]).output);
        let tf = testgen::tf32_admits(&p);
        let coef = 2.0 * (bound_coef(k, tf) + bound_coef(k, false));
        for c in 0..2 {
            let (yf, yw, mag) = (f[2 * c], w[2 * c], f[2 * c + 1]);
            assert!((yf - yw).abs() <= coef * mag, "#{z} cell {c}: {yw} vs {yf}, M {mag}\n{src}");
            if !tf {
                assert_eq!(yf, yw, "#{z}: an f32-only product has one value");
            }
        }
    }
    eprintln!("tf32-tile: {tensor} tensor-core, {ffma} f32 tile, {off_forms} without tf32, {valued} valued");
    assert!(tensor >= 8, "the generator reaches the tensor cores ({tensor})");
}
