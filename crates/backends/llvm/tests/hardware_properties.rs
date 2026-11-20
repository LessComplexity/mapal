//! hardware-properties: the rung rules read a part's physical properties —
//! SMs, L2 bytes, threads and registers per SM, FP32 lanes, clocks, the DRAM
//! interface — never its name. On programs from the generator
//! (`testgen::geom_prog_strategy`, which the rules were not designed on), each
//! compiled for a part drawn at random from those properties (`--cost=`'s
//! property names over a named base): the emission is counted against the
//! every-rung arm (`--gpu-off=arch-rungs`, S52) on both sides of each
//! threshold the rules read, and the values run through the GPU emulator —
//! free-running, serialized and with the blocks in reverse — equal the
//! every-rung arm byte for byte (integer-valued operands: every cut of the
//! fold is exact).

use mapal_backend_llvm::{CostOver, DeviceCost, EmitOpts, GpuArms, emit_detailed, resolve_profile};
use mapal_ir::CategoryIr;
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

/// A part drawn from physical properties over a named base.
struct Part {
    base: &'static str,
    over: Vec<(&'static str, u64)>,
}

impl Part {
    fn draw(z: usize, pick: [u8; 6]) -> Self {
        let bases = [
            "rtx-4090",
            "rtx-5090",
            "h100-pcie",
            "rtx-3090",
            "a100",
            "rtx-5080",
        ];
        let at = |i: usize, v: &[u64]| v[usize::from(pick[i]) % v.len()];
        let over = vec![
            ("sms", at(0, &[24, 46, 84, 114, 170])),
            ("l2_bytes", at(1, &[4 << 20, 6 << 20, 40 << 20, 96 << 20])),
            ("fp32_lanes", at(2, &[64, 128])),
            ("mem_bus_bits", at(3, &[128, 256, 384, 5120])),
            ("max_threads_per_sm", at(4, &[1536, 2048])),
            ("clock_khz", at(5, &[1_400_000, 2_500_000])),
        ];
        Self {
            base: bases[z % bases.len()],
            over,
        }
    }

    fn cost(&self) -> CostOver {
        let mut c: CostOver = [None; DeviceCost::NAMES.len()];
        for &(n, v) in &self.over {
            c[DeviceCost::NAMES.iter().position(|x| *x == n).expect(n)] = Some(v);
        }
        c
    }

    /// The property sides the rules read, as `rules.rs` computes them.
    fn sides(&self) -> [bool; 4] {
        let g = *resolve_profile(self.base)
            .and_then(|p| p.gpu())
            .expect("gpu");
        let g = g.with_cost(&self.cost());
        [
            g.ffma_per_dram_byte_pm().is_some_and(|b| b >= 70_000),
            g.regs_per_resident_thread() <= 32,
            g.l2_bytes_per_sm() < 128 * 1024,
            g.dram_bytes_per_us_per_sm() >= 14_600,
        ]
    }
}

/// The default face, `place` off (every capable site on the device), the
/// rungs the emulator cannot run off (`async`, `pdl`, `streams`, `pinned`);
/// `rules` false = every rung (the S52 arm the rules are measured against).
fn emission(ir: &CategoryIr, part: &Part, rules: bool) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    gpu.off.arch = !rules;
    emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target: part.base,
            gpu,
            cost: part.cost(),
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

#[test]
fn rung_rules_counted_and_valued_on_unseen_parts_and_programs() {
    let mut runner = TestRunner::deterministic();
    let strat = (
        testgen::geom_prog_strategy(),
        proptest::array::uniform6(proptest::prelude::any::<u8>()),
    );
    let clang = emu::clang();
    // Per property the rules read: programs on each side of its threshold.
    let mut sides = [[0u32; 2]; 4];
    let (mut moved, mut valued, mut modes) = (0, 0, 0);
    for z in 0..36 {
        let (p, pick) = strat.new_tree(&mut runner).unwrap().current();
        let (src, _) = testgen::geom_src(&p);
        let ir = lower(&src);
        let part = Part::draw(z, pick);
        for (i, s) in part.sides().into_iter().enumerate() {
            sides[i][usize::from(s)] += 1;
        }
        let (on, every) = (emission(&ir, &part, true), emission(&ir, &part, false));
        let dev = |e: &mapal_backend_llvm::Emission| e.device.clone().unwrap_or_default();
        if dev(&on) != dev(&every) {
            moved += 1;
        }
        let Some(cc) = &clang else { continue };
        let want = emu::run_emulated(cc, &every.text, &dev(&every));
        assert_eq!(
            want.lines().count(),
            3,
            "#{z} {} {:?}: printed\n{want}\n{src}",
            part.base,
            part.over
        );
        let e = emu::build_emulated(cc, &on.text, &dev(&on));
        for env in [
            &[][..],
            &[("MAPAL_EMU_SERIAL", "1")][..],
            &[("MAPAL_EMU_SERIAL", "2")][..],
            &[("MAPAL_EMU_REVERSE", "1")][..],
        ] {
            let (got, _) = e.run(env);
            assert_eq!(
                got, want,
                "#{z} {} {:?} {env:?}: the rules' values\n{src}",
                part.base, part.over
            );
            modes += 1;
        }
        valued += 1;
    }
    eprintln!(
        "hardware-properties: sides (lo/hi) balance {:?} regs {:?} l2 {:?} dram {:?}; \
         {moved} emissions moved by the rules, {valued} valued ({modes} emulator runs)",
        sides[0], sides[1], sides[2], sides[3]
    );
    for (i, s) in sides.iter().enumerate() {
        assert!(
            s[0] > 0 && s[1] > 0,
            "property {i}: both sides reached {s:?}"
        );
    }
    assert!(moved >= 6, "the rules change the emission ({moved})");
}
