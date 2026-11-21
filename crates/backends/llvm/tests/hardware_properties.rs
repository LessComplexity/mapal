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
//!
//! cross-arch: the operand-stream rungs read DRAM bytes per µs per SM (the
//! drawn parts reach both sides of `DRAM_PER_SM_LOW`), `pdl` rides the
//! module's loops (`launches_repeat`: generated programs with and without a
//! loop of device sites), and a retained row's split is priced against the
//! launches it adds (`row_split_pays`: the same programs on the measured H100
//! and with its launch cost unmeasured, values equal).

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
            ("mem_clock_khz", at(4, &[1_215_000, 10_501_000, 15_001_000])),
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
            g.dram_bytes_per_us_per_sm() < 9_200,
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
        "hardware-properties: sides (lo/hi) dram-scarce {:?} regs {:?} l2 {:?} dram {:?}; \
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

/// Printed values equal within the default face's bound (relative 1e-4).
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

/// `--cost=` overrides by name.
fn over(cost: &[(&str, u64)]) -> CostOver {
    let mut c: CostOver = [None; DeviceCost::NAMES.len()];
    for &(n, v) in cost {
        c[DeviceCost::NAMES.iter().position(|x| *x == n).expect(n)] = Some(v);
    }
    c
}

/// One generated program's emission on `target` with `cost` overrides and
/// `on` rungs forced (measurement arms).
fn emit_on(
    ir: &CategoryIr,
    target: &'static str,
    cost: &[(&str, u64)],
    on: &[&str],
) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
    let mut c: CostOver = [None; DeviceCost::NAMES.len()];
    for &(n, v) in cost {
        c[DeviceCost::NAMES.iter().position(|x| *x == n).expect(n)] = Some(v);
    }
    emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target,
            gpu,
            cost: c,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

/// `mapal_gpu_init`'s flag word in a host module (0 without one).
fn init_flags(text: &str) -> u32 {
    text.split("call void @mapal_gpu_init(i32 ")
        .nth(1)
        .and_then(|t| t.split(')').next())
        .map_or(0, |v| v.parse().expect("flags"))
}

/// **cross-arch `pdl` and `row-split` on unseen programs** (`calls_prog_strategy`:
/// chains of cells, sites and row sums, in a loop or straight-line, with and
/// without the retained rows' pattern). On the H100 SXM (programmatic launch
/// 734 < 1722 ns measured): the module launches programmatically exactly when
/// its device sites repeat in a loop, its kernels wait only then, and no tile
/// kernel triggers its dependent at entry unless `pdl-early` is forced. Its
/// retained rows (≤ 256 KB, 1.7 µs a pass alone) are not split against
/// 3.4 µs of added launches; with the one-SM rate unmeasured the property rule
/// splits them — and the values of both, run through the emulator
/// (free-running, serialized, reversed), agree within the default face's
/// bound (the split reorders the fold).
#[test]
fn pdl_rides_loops_and_row_split_is_priced_on_unseen_programs() {
    const PDL: u32 = 4;
    let mut runner = TestRunner::deterministic();
    let clang = emu::clang();
    let (mut pdl_sides, mut split_sides, mut modes) = ([0u32; 2], [0u32; 2], 0);
    let mut early_triggers = 0;
    for z in 0..24 {
        let p = testgen::calls_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let src = testgen::hz_src(&p);
        let ir = lower(&src);
        let e = emit_on(&ir, "h100", &[], &[]);
        let dev = e.device.clone().unwrap_or_default();
        let flag = init_flags(&e.text) & PDL != 0;
        if dev.contains("define ptx_kernel") {
            assert_eq!(flag, p.looped, "#{z}: pdl where the launches repeat\n{src}");
            assert_eq!(dev.contains("griddepcontrol.wait"), flag, "#{z}\n{src}");
            assert!(
                !dev.contains("launch_dependents"),
                "#{z}: exit trigger\n{src}"
            );
            pdl_sides[usize::from(flag)] += 1;
            if flag {
                // `pdl-early` adds the tile kernels' entry trigger and nothing
                // else.
                let early = emit_on(&ir, "h100", &[], &["pdl-early"]);
                let d = early.device.unwrap_or_default();
                let less: String = d
                    .lines()
                    .filter(|l| !l.contains("launch_dependents"))
                    .map(|l| format!("{l}\n"))
                    .collect();
                assert_eq!(less.trim_end(), dev.trim_end(), "#{z}: pdl-early\n{src}");
                early_triggers += d.matches("launch_dependents").count();
            }
        }
        // The retained rows' split: priced on the measured part, unpriced
        // with its launch cost removed.
        // Candidate 0 (`--cands=1`: a table's other candidates force the
        // opposite split, `row-split` being a choice axis): the unsplit row
        // passes (one block a row, no combine) it keeps.
        let c0 = |cost: &[(&str, u64)]| {
            let mut gpu = GpuArms::default();
            gpu.site_cands = 1;
            emit_detailed(
                &ir,
                &EmitOpts {
                    host: "generic",
                    target: "h100",
                    gpu,
                    cost: over(cost),
                    ..EmitOpts::default()
                },
            )
            .expect("emits")
            .device
            .unwrap_or_default()
        };
        let rows = |t: &str| {
            t.lines()
                .filter(|l| {
                    l.contains("kind=naive") && l.contains(" rows=1 ") && l.contains("combine=none")
                })
                .count()
        };
        let (a, b) = (rows(&c0(&[])), rows(&c0(&[("sm_bytes_per_ns", 0)])));
        assert!(
            a >= b,
            "#{z}: pricing only keeps rows unsplit ({a} < {b})\n{src}"
        );
        if p.kv {
            split_sides[usize::from(a > b)] += 1;
            // The same rows on a part whose one SM streams 1 B/ns: the pass
            // alone outweighs the launches, the split pays and is kept.
            let slow = rows(&c0(&[("sm_bytes_per_ns", 1)]));
            assert_eq!(slow, b, "#{z}: a paying split is kept\n{src}");
            split_sides[0] += u32::from(a > b);
        }
        let Some(cc) = &clang else { continue };
        // Values: the priced emission against the unpriced (split) one, the
        // emulator's three schedules; `pdl` is not emulated (forced off).
        let off = |t: &'static str, cost: &[(&str, u64)]| {
            let mut gpu = GpuArms::default();
            *gpu.off.rung("pdl").expect("rung") = true;
            let mut c: CostOver = [None; DeviceCost::NAMES.len()];
            for &(n, v) in cost {
                c[DeviceCost::NAMES.iter().position(|x| *x == n).expect(n)] = Some(v);
            }
            emit_detailed(
                &ir,
                &EmitOpts {
                    host: "generic",
                    target: t,
                    gpu,
                    cost: c,
                    ..EmitOpts::default()
                },
            )
            .expect("emits")
        };
        if a == b {
            continue;
        }
        let (pr, un) = (off("h100", &[]), off("h100", &[("sm_bytes_per_ns", 0)]));
        let want = emu::run_emulated(cc, &un.text, un.device.as_deref().unwrap());
        let built = emu::build_emulated(cc, &pr.text, pr.device.as_deref().unwrap());
        for env in [
            &[][..],
            &[("MAPAL_EMU_SERIAL", "1")][..],
            &[("MAPAL_EMU_SERIAL", "2")][..],
            &[("MAPAL_EMU_REVERSE", "1")][..],
        ] {
            // The default face's bound: the split reorders a fold (long-rows'
            // tolerance, relative 1e-4).
            let got = built.run(env).0;
            assert!(within(&got, &want), "#{z} {env:?}: {got} vs {want}\n{src}");
            modes += 1;
        }
    }
    eprintln!(
        "cross-arch: pdl (plain / programmatic) {pdl_sides:?}, pdl-early entry triggers {early_triggers}; \
         retained-row split (kept / priced off) {split_sides:?}; {modes} emulator runs"
    );
    assert!(early_triggers > 0, "pdl-early reached a tile kernel");
    assert!(
        pdl_sides[0] > 0 && pdl_sides[1] > 0,
        "both pdl sides {pdl_sides:?}"
    );
    assert!(
        split_sides[0] > 0 && split_sides[1] > 0,
        "a split kept and one priced off {split_sides:?}"
    );
}
