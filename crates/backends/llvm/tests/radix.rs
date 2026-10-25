//! recurrence-precision `radix-tile`: the mixed-radix index fact
//! (`CategoryIr::radix_plan`) and the tile sites it admits
//! (`CategoryIr::tile_plan_radix`) — a convolution's implicit GEMM. The fact is
//! checked address by address against the program's own index arithmetic; the
//! rung on programs from the generator (`testgen::radix_prog_strategy`, which
//! it was not designed on): counted from the emission against its OFF arm,
//! and the values through the GPU emulator, ON = OFF byte for byte (exact face).

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, TileSite};
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

fn radix_sites(ir: &CategoryIr) -> Vec<TileSite> {
    ir.funcs().flat_map(|(f, _)| ir.tile_plan_radix(f).into_iter().map(|(_, s)| s)).collect()
}

fn bench(name: &str) -> String {
    let p = format!("{}/../../../benches/algorithms/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(p).expect("bench program")
}

#[test]
fn convmc_is_one_implicit_gemm_site() {
    let ir = lower(&bench("pt_convmc_8_64_56_3.mapal"));
    let [s] = &radix_sites(&ir)[..] else { panic!("one radix site") };
    // Lanes are the output's fast digits (b, i, j: B·H·W), rows the output
    // channel, k the taps (Cin·9) — weights plain, the image and output digit-affine.
    assert_eq!((s.rows, s.c, s.k), (64, 25088, 576));
    assert_eq!((s.a.ci, s.a.ck, s.a.clane, s.a.base), (576, 1, 0, 0));
    let r = s.radix.as_deref().expect("radix");
    assert!(r.a.is_none() && r.b.is_some() && r.out.is_some());
    // No CPU site: the host's tile ladder is untouched.
    assert!(ir.funcs().all(|(f, _)| ir.tile_plan(f).sites.is_empty()));
}

/// Every address the fact names equals the program's index arithmetic, for
/// every `(t, k)` of the small convolution.
#[test]
fn radix_addresses_match_the_program_everywhere() {
    let ir = lower(&bench("px_convmc_2_4_6_3.mapal"));
    let [s] = &radix_sites(&ir)[..] else { panic!("one radix site") };
    let r = s.radix.as_deref().expect("radix");
    let (b_, o) = (r.b.as_ref().expect("img radix"), r.out.as_ref().expect("out radix"));
    // b = img (lanes: batch, row, column), rows = output channel.
    assert_eq!((s.rows, s.c, s.k), (4, 72, 36));
    let mut seen = std::collections::BTreeSet::new();
    for co in 0..4 {
        for lane in 0..72 {
            let (b, i, j) = (lane / 36, lane / 6 % 6, lane % 6);
            let t = b * 144 + co * 36 + i * 6 + j;
            assert_eq!(o.at(co, lane, 0), t);
            seen.insert(t);
            for k in 0..36 {
                let (ci, q) = (k / 9, k % 9);
                assert_eq!(b_.at(co, lane, k), ((b * 4 + ci) * 8 + i + q / 3) * 8 + j + q % 3);
                assert_eq!(s.a.ci * co + s.a.ck * k, co * 36 + k);
            }
        }
    }
    assert_eq!(seen.len(), 288, "the output map is a bijection");
}

/// The exact face on `target`, `radix-tile` on or off (and the rungs the
/// emulator cannot run: `async`, `pdl`, `streams`; `place` off: every capable
/// site on the device, whatever its size).
fn emission(ir: &CategoryIr, target: &'static str, on: bool) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams", "place"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    *(if on { gpu.on.rung("radix-tile") } else { gpu.off.rung("radix-tile") }).expect("rung") = true;
    let opts = EmitOpts { host: "generic", target, gpu, ..EmitOpts::exact() };
    emit_detailed(ir, &opts).expect("emits")
}

fn radix_kernels(dev: Option<&str>) -> usize {
    dev.map_or(0, |d| d.lines().filter(|l| l.starts_with("; mapal.plan") && l.contains(" radix=")).count())
}

#[test]
fn radix_tile_counted_and_valued_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::radix_prog_strategy();
    let clang = emu::clang();
    let targets = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];
    let (mut tiled, mut refused, mut valued) = (0, 0, 0);
    for z in 0..48 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let (src, _) = testgen::radix_src(&p);
        let ir = lower(&src);
        let admitted = !radix_sites(&ir).is_empty();
        assert_eq!(admitted, !p.mix, "#{z}: the fact admits exactly the unmixed programs\n{src}");
        let target = targets[z % targets.len()];
        let (on, off) = (emission(&ir, target, true), emission(&ir, target, false));
        let n_on = radix_kernels(on.device.as_deref());
        assert_eq!(radix_kernels(off.device.as_deref()), 0, "#{z} OFF has no radix kernel");
        if n_on > 0 {
            tiled += 1;
            // ON takes the map from wherever OFF ran it (a nest cell or the host).
            let cells = |d: &Option<String>| d.as_deref().map_or(0, |d| d.matches("kind=cell").count());
            assert!(cells(&on.device) <= cells(&off.device), "#{z} {target}\n{src}");
        } else {
            refused += 1;
            assert_eq!(on.device, off.device, "#{z}: no site, no change\n{src}");
        }
        // (An OFF arm with no device unit runs the host alone.)
        if let (Some(cc), true) = (&clang, n_on > 0) {
            let (a, b) = (on.device.as_deref().expect("dev"), off.device.as_deref().unwrap_or(""));
            let (x, y) = (emu::run_emulated(cc, &on.text, a), emu::run_emulated(cc, &off.text, b));
            assert!(x.lines().count() == 3, "#{z}: the program printed\n{x}");
            assert_eq!(x, y, "#{z} {target} values\n{src}");
            valued += 1;
        }
    }
    eprintln!("radix-tile: {tiled} tiled, {refused} not, {valued} valued in the emulator");
    assert!(tiled >= 8, "the generator reaches the tile ({tiled})");
}

/// A register tile (`tm·tn > 1`, vector stores through the digit-affine
/// output) on every measured part: ON = OFF in the emulator, exact face.
#[test]
fn radix_register_tile_values() {
    let src = "fn main() {
    4096 -> iota -> i_w;
    i_w -> map { t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 11.0 } -> wt;
    2592 -> iota -> i_img;
    i_img -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> img;
    8192 -> iota -> o_y;
    64 -> iota -> k_y;
    o_y -> map { t ->
        t / 4096 -> b; t / 64 % 64 -> co; t / 8 % 8 -> i; t % 8 -> j;
        (0.0, k_y) -> fold { acc, k -> k / 4 -> ci; k % 4 -> q; acc + wt[co * 64 + k] * img[((b * 16 + ci) * 9 + i + q / 2) * 9 + j + q % 2] }
    } -> y;
    y[0] -> println;
    y[8191] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold { acc, v -> acc + v } -> total;
    total -> println;
}
";
    let ir = lower(src);
    let Some(cc) = emu::clang() else { return };
    for target in ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"] {
        let (on, off) = (emission(&ir, target, true), emission(&ir, target, false));
        let dev = on.device.as_deref().expect("dev");
        let plan = dev.lines().find(|l| l.contains(" radix=")).expect("a radix kernel");
        assert!(!plan.contains(" tm=1 tn=1 "), "{target}: a register tile\n{plan}");
        let x = emu::run_emulated(&cc, &on.text, dev);
        assert_eq!(x, emu::run_emulated(&cc, &off.text, off.device.as_deref().unwrap_or("")), "{target}");
    }
}
