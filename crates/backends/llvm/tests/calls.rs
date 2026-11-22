//! kernel-merging `calls`: a chain of dependent launches of one sequence runs
//! as one persistent kernel — each launch's body a device function over its
//! blocks, a grid barrier between the levels the dependency fact orders, an
//! atomic site's zero pass one level before it. The rule's decision on the
//! decode step and the OFF arm; values through the GPU emulator (cooperative
//! launches: every block's threads at once, one block holding the baton
//! between grid barriers) against the interpreter and the OFF arm under every
//! candidate pick, at several persistent grids, free-running, serialized and
//! with the blocks in reverse, on a hand-written decode and on generated
//! programs (`testgen::calls_prog_strategy`).

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
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

fn bench(name: &str) -> CategoryIr {
    lower(
        &std::fs::read_to_string(format!(
            "{}/../../../benches/algorithms/{name}.mapal",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
}

/// The emulator's emission: its rungs off (`place`, so every capable site is
/// on the device), the default face, chains formed whatever their prices
/// (`--calls=force`: the realized path under test; the rtx-5080's prices
/// refuse them).
fn emulated(ir: &CategoryIr, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.calls_force = true;
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).unwrap() = true;
    }
    *gpu.off.rung("calls").unwrap() = !on;
    emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// The host intent line `; mapal.calls.host …`.
fn host_line(em: &Emission) -> String {
    em.text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("; mapal.calls.host "))
        .unwrap_or("")
        .to_string()
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
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-3 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

fn launches(err: &str) -> u64 {
    let l = err
        .lines()
        .find_map(|l| l.strip_prefix("emu_total "))
        .unwrap_or("");
    field(l, "launches").parse().unwrap_or(0)
}

/// The persistent kernels' spec lines (`; mapal.hz kernel=… grid=…`).
fn specs(em: &Emission) -> Vec<String> {
    em.text
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("; mapal.hz kernel=") && l.contains(" grid="))
        .map(str::to_string)
        .collect()
}

/// `pt_decode_1024_256` on the rtx-5080: the trip's rows and query (one
/// `horizontal` group, level 0), `s` (reads `kk` and `q`: level 1, its zero
/// pass at level 0) and `o` (reads `s` and `vv`: level 2, zero at level 1)
/// make one chain of 128 blocks. Shipped, the rule refuses it by cost: a
/// cooperative launch (2044 ns) and two grid barriers (553 + 44 · 1.89 ns
/// each at 128 blocks) against three dependent launches (650 ns each; 486 as
/// programmatic launches, `pdl` forced on — refused too). Forced
/// (`--calls=force`, the measurement arm), it is one persistent kernel: the
/// tables keep the rows' unsplit candidates and `s` / `o`'s atomic ones;
/// every tuple a persistent kernel with two grid barriers, launched
/// cooperatively. `x` (a range cell) launches alone. OFF: no chain, the text
/// before the rung.
#[test]
fn decode_runs_its_trip_as_one_persistent_kernel() {
    let ir = bench("pt_decode_1024_256");
    let emit = |pdl: bool, force: bool, on: bool| {
        let mut gpu = GpuArms::default();
        gpu.on.pdl = pdl;
        gpu.calls_force = force;
        *gpu.off.rung("calls").unwrap() = !on;
        emit_detailed(
            &ir,
            &EmitOpts {
                target: "rtx-5080",
                gpu,
                ..EmitOpts::default()
            },
        )
        .expect("emits")
    };
    for pdl in [false, true] {
        let def = emit(pdl, false, true);
        let h = host_line(&def);
        assert_eq!(
            (
                field(&h, "chains"),
                field(&h, "formed"),
                field(&h, "refused")
            ),
            ("1", "0", "cost:1"),
            "pdl {pdl}: {h}"
        );
        assert!(specs(&def).is_empty());
    }
    let on = emit(false, true, true);
    let h = host_line(&on);
    assert_eq!(
        (
            field(&h, "formed"),
            field(&h, "units"),
            field(&h, "barriers"),
            field(&h, "zeros"),
            field(&h, "refused")
        ),
        ("1", "3", "2", "2", "0"),
        "{h}"
    );
    let sp = specs(&on);
    assert!(!sp.is_empty() && sp.len() <= 8, "{sp:?}");
    assert!(
        sp[0].contains("grid=128")
            && sp[0].contains(":l1:z4096@2")
            && sp[0].contains(":l2:z1024@2"),
        "{}",
        sp[0]
    );
    // The table's flags: re-run safe, persistent (`MAPAL_SITE_COOP`).
    assert!(
        on.text.contains("call void @mapal_gpu_site(i64 ")
            && on.text.contains(", i32 5, i32 0, i32 ")
    );
    let dev = on.device.as_deref().unwrap();
    for s in &sp {
        let name = field(s, "kernel");
        let at = dev
            .find(&format!("define ptx_kernel void @{name}("))
            .unwrap_or_else(|| panic!("{name}"));
        let text = &dev[at..at + dev[at..].find("\n}\n").unwrap()];
        assert_eq!(text.matches("call void @mapal_gsync(").count(), 2, "{name}");
        assert!(!text.lines().next().unwrap().contains("noalias"), "{name}");
        assert!(text.contains("barrier.sync.cnt(i32 1, i32 256)"), "{name}");
    }
    assert!(dev.contains("define internal void @mapal_gsync(ptr addrspace(1) %c)"));
    assert!(dev.contains("atom.add.release.gpu.global.u32"));
    // The atomic sites' zero passes are the kernel's: `zero-fold` takes none.
    assert!(!on.text.contains("; mapal.zero sites="), "zero tails left");
    let off = emit(false, true, false);
    assert!(host_line(&off).is_empty() && specs(&off).is_empty());
    assert!(!off.device.as_deref().unwrap().contains("@mapal_gsync"));
    // A part without the measured barrier and launch costs forms no chain
    // (the 5090: the cross-arch parts carry theirs).
    let mut gpu = GpuArms::default();
    gpu.calls_force = true;
    let p5090 = emit_detailed(
        &ir,
        &EmitOpts {
            target: "rtx-5090",
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    assert!(host_line(&p5090).is_empty() && specs(&p5090).is_empty());
}

/// **Every part prices a chain's bodies** (`calib.cu body`): the rtx-5080
/// measured them (4390 cycles at its 2.45 GHz); every other GPU profile
/// derives them from those cycles at its own clock — never `unpriced`. On the
/// suite's chains: the rtx-5080 refuses every one by cost (a cooperative
/// launch of 2044 ns against launches of 486 / 650); decode's chain is refused
/// on every part (656 / 522 ns saved on the 4090 / 3090 against 1742 / 2812 of
/// bodies); the H100's smlp / firew_rowsum chains — formed on the boundaries
/// alone in the cross-arch 7a2b67bf round, 1.55 / 2.3x slower — are refused
/// (695 ns saved against 2429); causal's six-launch chain forms on the 4090
/// and the 3090, not on the a100 or the H100. The derivation follows the
/// clock (`--cost=clock_khz=`): the a100 at 2.3 GHz forms causal's chain, the
/// 3090 at 1.0 GHz refuses it.
#[test]
fn every_part_prices_chains() {
    use mapal_backend_llvm::TargetProfile;
    let gpu = |n: &str| TargetProfile::named(n).and_then(|p| p.gpu()).expect(n);
    let five = gpu("rtx-5080").coop_body_cycles();
    assert!(five.1 && five.0 > 0, "the rtx-5080 measured it: {five:?}");
    for n in [
        "cuda-ada",
        "rtx-4090",
        "h100",
        "h100-pcie",
        "rtx-5090",
        "rtx-5080",
        "rtx-3090",
        "a100",
    ] {
        let g = gpu(n);
        assert_eq!(g.coop_body_cycles(), (five.0, n == "rtx-5080"), "{n}");
        let ns = five.0 as f64 * 1e6 / g.clock_khz as f64;
        assert!(
            (g.coop_body_ns() - ns).abs() < 1e-6,
            "{n}: {}",
            g.coop_body_ns()
        );
    }
    let emit = |ir: &CategoryIr, target: &'static str, clock: Option<u64>| {
        let mut opts = EmitOpts {
            target,
            ..EmitOpts::default()
        };
        if let Some(c) = clock {
            let f = format!("--cost=clock_khz={c}");
            assert!(mapal_backend_llvm::flags::apply(&mut opts, &f).expect("a flag"));
        }
        host_line(&emit_detailed(ir, &opts).expect("emits"))
    };
    // `(program, the parts whose chain forms)`, over the parts with the
    // measured barrier and launch costs.
    let parts = ["rtx-5080", "rtx-4090", "rtx-3090", "a100", "h100"];
    let want: [(&str, &[&str]); 5] = [
        ("pt_decode_1024_256", &[]),
        ("pt_causal_1024_256", &["rtx-4090", "rtx-3090"]),
        ("pt_smlp_4096_256_1024", &[]),
        ("pt_firew_rowsum_2048x2048_16", &[]),
        ("pt_rt_mlp2_1024", &[]),
    ];
    for (p, forms) in want {
        let ir = bench(p);
        for part in parts {
            let h = emit(&ir, part, None);
            assert!(!h.contains("unpriced"), "{part} {p}: {h}");
            let formed = !h.is_empty() && field(&h, "formed") != "0";
            assert_eq!(formed, forms.contains(&part), "{part} {p}: {h}");
            if !h.is_empty() && !formed && field(&h, "chains") != "0" {
                assert!(field(&h, "refused").contains("cost"), "{part} {p}: {h}");
            }
        }
        if p == "pt_causal_1024_256" {
            assert_ne!(field(&emit(&ir, "a100", Some(2_300_000)), "formed"), "0");
            assert_eq!(
                field(&emit(&ir, "rtx-3090", Some(1_000_000)), "formed"),
                "0"
            );
        }
    }
}

/// A decode step small enough to emulate (16 rows of 256, 3 trips): values
/// equal the interpreter and the OFF arm under four picks, at persistent grids
/// of 1, 2 and 3 blocks, free-running, serialized and in reverse; the first
/// trip (the rows' whole sites in place) launches each member alone, later
/// trips one cooperative launch for three.
#[test]
fn decode_values_through_the_persistent_kernel() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let (trips, rows, d) = (3u64, 16u64, 256u64);
    let src = format!(
        "fn main() {{
    {dd} -> iota -> tw;
    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 64.0 }} -> wk;
    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 64.0 }} -> wv;
    {rd} -> iota -> tc;
    {d} -> iota -> krange;
    {rows} -> iota -> rrange;
    (0.0, {rd}) -> fill -> x0;
    (0.0, {d}) -> fill -> o0;
    mut x: [f32; {rd}] <- x0;
    mut o: [f32; {d}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {trips}) -> {{
            -true-> {{
                tc -> map {{ c -> (c / {d} == t) -> {{ -true-> {{ (c * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 16.0 + o[c % {d}] }} -false-> x[c]; }} }} -> x;
                tc -> map {{ c -> c / {d} -> i; c % {d} -> j; (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wk[k * {d} + j] }} }} -> kk;
                tc -> map {{ c -> c / {d} -> i; c % {d} -> j; (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wv[k * {d} + j] }} }} -> vv;
                krange -> map {{ k -> ((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / 64.0 }} -> q;
                rrange -> map {{ r -> (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;
                krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        dd = d * d,
        rd = rows * d,
        last = d - 1
    );
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let on = emulated(&ir, true);
    let off = emulated(&ir, false);
    let h = host_line(&on);
    assert_eq!(field(&h, "formed"), "1", "{h}\n{}", on.text);
    let (eon, eoff) = (
        emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap()),
        emu::build_emulated(&clang, &off.text, off.device.as_deref().unwrap()),
    );
    for pick in ["0", "1", "3", "7"] {
        let (b, eb) = eoff.run(&[("MAPAL_GPU_PICK", pick)]);
        assert!(within(&b, &rr.output), "pick {pick} OFF: {b:?}");
        for grid in ["1", "2", "3"] {
            let env = [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_GRID", grid)];
            let (a, ea) = eon.run(&env);
            assert!(
                within(&a, &rr.output),
                "pick {pick} grid {grid} ON: {a:?} vs {:?}",
                rr.output
            );
            assert!(
                within(&a, &b),
                "pick {pick} grid {grid}: ON {a:?} OFF {b:?}"
            );
            // One cooperative launch for the three of every trip after the
            // first (pick 0: each member one launch alone; a workspace
            // candidate's reduce is a launch of its own OFF).
            if pick == "0" {
                assert_eq!(
                    launches(&ea) + 2 * (trips - 1),
                    launches(&eb),
                    "pick {pick} grid {grid}\n{ea}\n{eb}"
                );
            }
            assert!(launches(&ea) < launches(&eb), "pick {pick} grid {grid}");
            for mode in ["1", "2"] {
                let (s, _) = eon.run(&[env[0], env[1], ("MAPAL_EMU_SERIAL", mode)]);
                assert!(
                    within(&s, &rr.output),
                    "pick {pick} grid {grid} serial {mode}"
                );
            }
            let (r, _) = eon.run(&[env[0], env[1], ("MAPAL_EMU_REVERSE", "1")]);
            assert!(within(&r, &rr.output), "pick {pick} grid {grid} reverse");
        }
    }
}

/// Coverage of [`calls_on_generated_programs`].
#[derive(Default, Debug)]
struct Cover {
    programs: usize,
    chains: usize,
    formed: usize,
    units: usize,
    barriers: usize,
    zeros: usize,
    refused: usize,
    cost: usize,
    three_levels: usize,
    tables: usize,
    padded: usize,
    branchy: usize,
    loops: usize,
    straight: usize,
    fewer: usize,
    runs: usize,
}

/// Over generated programs (`calls_prog_strategy`, in parallel): the OFF arm
/// has no persistent kernel; ON, picks 0 and 3, at persistent grids of 1 and
/// 3 blocks, and at 3 serialized in thread order and reverse and with the
/// blocks in reverse, equals the interpreter (within the default face's bound) and
/// the OFF arm, with fewer launches where a chain formed. Coverage asserted:
/// chains formed in loop trips and straight-line code, chains of three levels
/// or more, zero passes in the kernel, merged tables, members of fewer
/// threads than the block, retained rows, chains the rule refused.
#[test]
fn calls_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::calls_prog_strategy();
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let progs: Vec<testgen::HzProg> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let cv = std::sync::Mutex::new(Cover::default());
    let next = std::sync::atomic::AtomicUsize::new(0);
    // Programs in parallel (each emulated run holds one of the harness's
    // job slots): one worker a slot.
    std::thread::scope(|sc| {
        for _ in 0..emu::job_slots().max(1) {
            sc.spawn(|| {
                loop {
                    let z = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(p) = progs.get(z) else { break };
                    // `MAPAL_TESTGEN_ONLY=<z>`: one program (a failure's repro).
                    if std::env::var("MAPAL_TESTGEN_ONLY")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .is_some_and(|o| o != z)
                    {
                        continue;
                    }
                    eprintln!("calls gen #{z} start");
                    let d = one_program(&clang, z, p);
                    eprintln!("calls gen #{z} done");
                    let mut c = cv.lock().unwrap();
                    c.add(&d);
                }
            });
        }
    });
    let cv = cv.into_inner().unwrap();
    eprintln!("calls_on_generated_programs: {cv:?}");
    assert!(
        cv.formed > 0
            && cv.loops > 0
            && cv.straight > 0
            && cv.three_levels > 0
            && cv.zeros > 0
            && cv.tables > 0
            && cv.padded > 0
            && cv.branchy > 0
            && cv.refused > 0
            && cv.fewer > 0,
        "{cv:?}"
    );
}

impl Cover {
    fn add(&mut self, d: &Cover) {
        self.programs += d.programs;
        self.chains += d.chains;
        self.formed += d.formed;
        self.units += d.units;
        self.barriers += d.barriers;
        self.zeros += d.zeros;
        self.refused += d.refused;
        self.cost += d.cost;
        self.three_levels += d.three_levels;
        self.tables += d.tables;
        self.padded += d.padded;
        self.branchy += d.branchy;
        self.loops += d.loops;
        self.straight += d.straight;
        self.fewer += d.fewer;
        self.runs += d.runs;
    }
}

/// One generated program's checks ([`calls_on_generated_programs`]).
fn one_program(clang: &str, z: usize, p: &testgen::HzProg) -> Cover {
    let mut cv = Cover::default();
    let src = testgen::hz_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "#{z}\n{src}");
    let on = emulated(&ir, true);
    let off = emulated(&ir, false);
    assert!(host_line(&off).is_empty() && specs(&off).is_empty(), "#{z}");
    cv.programs += 1;
    let h = host_line(&on);
    let n = |k: &str| field(&h, k).parse::<usize>().unwrap_or(0);
    cv.chains += n("chains");
    cv.refused += usize::from(!h.is_empty() && field(&h, "refused") != "0");
    cv.cost += usize::from(field(&h, "refused").contains("cost"));
    let formed = n("formed");
    if formed == 0 {
        assert!(specs(&on).is_empty(), "#{z}");
        return cv;
    }
    cv.formed += formed;
    cv.units += n("units");
    cv.barriers += n("barriers");
    cv.zeros += n("zeros");
    let sp = specs(&on);
    cv.three_levels += usize::from(sp.iter().any(|s| s.contains(":l2")));
    cv.tables += usize::from(sp.len() > formed);
    cv.padded += usize::from(sp.iter().any(|s| {
        let b = field(s, "block");
        field(s, "members")
            .split(',')
            .any(|m| m.split(':').nth(3) != Some(b))
    }));
    cv.branchy += usize::from(p.kv && sp.iter().any(|s| s.contains(":l0")));
    if p.looped {
        cv.loops += 1;
    } else {
        cv.straight += 1;
    }
    let (eon, eoff) = (
        emu::build_emulated(clang, &on.text, on.device.as_deref().unwrap()),
        emu::build_emulated(clang, &off.text, off.device.as_deref().unwrap()),
    );
    for pick in ["0", "3"] {
        let (b, eb) = eoff.run(&[("MAPAL_GPU_PICK", pick)]);
        for grid in ["1", "3"] {
            let env = [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_GRID", grid)];
            let (a, ea) = eon.run(&env);
            assert!(
                within(&a, &rr.output),
                "#{z} pick {pick} grid {grid}: {a:?} vs {:?}\n{src}",
                rr.output
            );
            assert!(
                within(&a, &b),
                "#{z} pick {pick} grid {grid}: ON {a:?} OFF {b:?}\n{src}"
            );
            // Fewer launches, but for the retained pattern's first trip: its
            // whole sites launch in place, so each member launches alone (the
            // `horizontal` groups the chain took are gone that trip).
            assert!(
                p.kv || launches(&ea) <= launches(&eb),
                "#{z} pick {pick} grid {grid}: launches ON {ea} OFF {eb}\n{src}\n{h}"
            );
            cv.fewer += usize::from(launches(&ea) < launches(&eb));
            cv.runs += 1;
        }
        // Serialized in thread order and reverse, and the blocks in reverse,
        // at three persistent blocks (each body strides over several).
        let env = [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_GRID", "3")];
        for (k, v) in [
            ("MAPAL_EMU_SERIAL", "1"),
            ("MAPAL_EMU_SERIAL", "2"),
            ("MAPAL_EMU_REVERSE", "1"),
        ] {
            let (s, _) = eon.run(&[env[0], env[1], (k, v)]);
            assert!(within(&s, &rr.output), "#{z} pick {pick} {k}={v}\n{src}");
            cv.runs += 1;
        }
    }
    cv
}
