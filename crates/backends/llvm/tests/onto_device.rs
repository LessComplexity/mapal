//! onto-device: idiomatic programs compile to a device form.
//!
//! - a loop's exit value is readable inside a later map / fold body (the
//!   typing pass binds a routing guard's exit in the loop's enclosing scope);
//! - `bounds_proof` ranges an index by the conditions that hold where it runs
//!   (a guard arm's, a loop's continue and exit conditions, a `Phi`'s per arm)
//!   and ranges loop-carried counters from `loop_plan` — OFF arm
//!   `--gpu-off=cond-bounds` (`FactArms`);
//! - `carry-made`: a loop view is device-made when its next state is, so a
//!   step's small maps over a device-carried state stay on the device — OFF
//!   arm `--gpu-off=carry-made`.
//!
//! Facts on hand-written repros; values through the GPU emulator against the
//! interpreter (free threads, serialized in thread order and reversed) on
//! generated programs (`testgen::onto_prog_strategy`), with coverage counters.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, FactArms, FuncId, MorphismId, Operation};
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

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// Every `Index` of the graph with its function.
fn indexes(ir: &CategoryIr) -> Vec<(FuncId, MorphismId)> {
    ir.funcs()
        .flat_map(|(f, fd)| {
            fd.morphisms
                .iter()
                .filter(|&&m| ir.morphism(m).is_some_and(|x| x.op == Operation::Index))
                .map(move |&m| (f, m))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `(indexes, proven, proven only under a guard arm)`.
fn proof_counts(ir: &CategoryIr) -> (usize, usize, usize) {
    let all = indexes(ir);
    let proven = all
        .iter()
        .filter(|&&(f, m)| ir.bounds_proof(f).proven(m))
        .count();
    let armed = ir
        .funcs()
        .flat_map(|(f, _)| ir.guard_plan(f))
        .flat_map(|s| [s.on_true.cond_reads, s.on_false.cond_reads])
        .filter(|&c| c)
        .count();
    (all.len(), proven, armed)
}

fn traps(text: &str) -> usize {
    text.matches("call void @mapal_trap(").count()
        + text.matches("call void @mapal_par_trap(").count()
}

/// The GPU emission on the rtx-5080 with the rungs `off` off; `place` off so
/// every capable site is on the device, and the emulator's rungs off.
fn device_emission(ir: &CategoryIr, base: EmitOpts, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The shipped GPU build (rules as shipped) with the rungs `off` off.
fn shipped(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in off {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "x86-64-v3",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// Finding 1: a loop's exit value read inside a later map body (and through
/// an alias, and in a fold body) — was `L1101 unresolved name`.
#[test]
fn loop_exit_value_reads_in_later_bodies() {
    let src = "fn main() {
    (1.0, 4) -> fill -> x0;
    2 -> iota -> r2;
    mut x: [f32; 4] <- x0;
    mut t: i32 <- 0;
    loop {
        (t < 3) -> {
            -true-> {
                x -> map { v -> v * 2.0 } -> x;
                t + 1 -> t;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    r2 -> map { j -> out[2 + j] } -> last;
    last[1] -> println;
    out -> res;
    r2 -> map { j -> res[j] + 1.0 } -> l2;
    l2[0] -> println;
    (0.0, r2) -> fold { a, j -> a + out[j] } -> s;
    s -> println;
}
";
    assert_eq!(interp(&lower(src)), "8\n9\n16\n");
}

/// Finding 2: Llama's rotate_half partner read `a[c ± 64]` under `c % 128 <
/// 64` — in bounds under the guard (the narrowing runs back through `%` to
/// `c`); the arms are gated (`cond_reads`), no trap is emitted. OFF arm: the
/// interval proof keeps both traps.
#[test]
fn guard_condition_proves_partner_read() {
    let src = "fn main() {
    256 -> iota -> r;
    r -> map { t -> t % 101 -> widen_f32 } -> a;
    r -> map {
        c -> c % 128 -> i;
        (i < 64) -> {
            -true-> a[c + 64];
            -false-> a[c - 64];
        }
    } -> b;
    b[0] -> println;
    b[200] -> println;
}
";
    let ir = lower(src);
    let (n, proven, armed) = proof_counts(&ir);
    assert_eq!(
        (n, proven, armed),
        (4, 4, 2),
        "both partner reads (and the two prints), both arms gated"
    );
    let off = ir.refacted(FactArms {
        cond_bounds: false,
        ..FactArms::default()
    });
    assert_eq!(proof_counts(&off).1, 2, "only the prints' constant indexes");
    // The CPU build under the OFF arm (`--gpu-off=cond-bounds` rebuilds the
    // facts under `FactArms`): both partner reads checked.
    let cpu = |ir: &CategoryIr, off: bool| {
        let mut opts = EmitOpts::default();
        opts.gpu.off.cond_bounds = off;
        emit_detailed(ir, &opts).expect("emits").text
    };
    assert_eq!(traps(&cpu(&ir, false)), 0);
    assert!(traps(&cpu(&ir, true)) >= 2, "{}", cpu(&ir, true));
    assert_eq!(interp(&ir), "64\n35\n");
}

/// Finding 3: a decode step's position `p = 28 + t` (`t < 4`) indexes the
/// table: `t` ranges `[0, 3]` on the advancing trips (`loop_plan`: init 0,
/// step 1, continue `t < 4`), `p * 4 + i ≤ 127`. A counter that can wrap
/// (`t * 65536` past i32) stays unranged.
#[test]
fn loop_counter_ranges_index() {
    let src = "fn main() {
    128 -> iota -> r;
    r -> map { t -> t % 101 -> widen_f32 } -> tab;
    4 -> iota -> r4;
    (0.0, 4) -> fill -> x0;
    mut x: [f32; 4] <- x0;
    mut t: i32 <- 0;
    loop {
        (t < 4) -> {
            -true-> {
                28 + t -> p;
                r4 -> map { i -> x[i] + tab[p * 4 + i] } -> x;
                t + 1 -> t;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    out[0] -> println;
}
";
    let ir = lower(src);
    let (n, proven, _) = proof_counts(&ir);
    assert_eq!(n, proven, "every read proven");
    assert!(
        proof_counts(&ir.refacted(FactArms {
            cond_bounds: false,
            ..FactArms::default()
        }))
        .1 < n
    );
    assert_eq!(interp(&ir), "68\n");
    // Wrap: `t * 65536` leaves i32 on the way to the exit — no range.
    let wrap = src.replace("28 + t -> p;", "t * 1073741824 + 28 -> p;");
    let ir = lower(&wrap);
    let (n, proven, _) = proof_counts(&ir);
    assert!(proven < n, "a wrapping index stays checked");
}

/// A loop's exit counter (`u` at exit is `K`) read in a later body: in
/// bounds by the exit condition; a `Phi`-clamped index (`j < 0 ? 0 : j`) by
/// each arm's condition.
#[test]
fn exit_condition_and_phi_arms_range() {
    let src = "fn main() {
    16 -> iota -> r;
    r -> map { t -> t % 7 -> widen_f32 } -> a;
    mut u: i32 <- 0;
    loop {
        (u < 3) -> {
            -true-> {
                u + 1 -> u;
                -> loop;
            }
            -false-> u -> un;
        }
    }
    13 -> iota -> rk;
    rk -> map { j -> a[un + j] } -> y;
    r -> map { c -> c - 3 -> j; (j < 0) -> { -true-> 0; -false-> j; } -> k; a[k] } -> z;
    y[12] -> println;
    z[0] -> println;
}
";
    let ir = lower(src);
    let (n, proven, armed) = proof_counts(&ir);
    assert_eq!(
        (n, proven, armed),
        (4, 4, 0),
        "no arm needed: the Phi's value is ranged"
    );
    assert_eq!(interp(&ir), "1\n0\n");
}

/// Finding 5 (`carry-made`): the decode step's RMSNorm and O projection read
/// the carried `x`; the O projection is device-made (its operands are), so
/// `x` is device-made through the back edge and the whole step stays on the
/// device — `x` device-carried, no per-trip copies. OFF arm: `x` crosses to
/// the host each trip.
#[test]
fn decode_state_stays_on_device() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../benches/algorithms/pt_llm_dec_b1_l2048.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let moves = |em: &Emission| -> String {
        em.text
            .lines()
            .filter(|l| l.starts_with("; mapal.move"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let on = shipped(&ir, &[]);
    assert_eq!(traps(&on.text), 0);
    let m = moves(&on);
    assert!(
        m.contains("2:16384:device") && m.contains("h2d=0 d2h=0"),
        "{m}"
    );
    // (`tile-post` off too: with the O projection a post tile site, this
    // OFF arm also runs the norm's scaling map on the host and uploads its
    // 16 KB — device-geometry's open placement item.)
    let off = shipped(&ir, &["carry-made", "tile-post"]);
    let m = moves(&off);
    assert!(m.contains("2:16384:two") && m.contains("h2d=16384"), "{m}");
    // `cond-bounds` off: RoPE keeps its traps, the caches cross too.
    let nb = shipped(&ir, &["cond-bounds"]);
    assert!(traps(&nb.text) > 0);
}

/// The small decode twin (natural RoPE, the position from the loop counter)
/// through the emulator, every capable site on the device, exact face: the
/// interpreter's values bit for bit in every mode.
#[test]
fn natural_decode_twin_on_device() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../benches/algorithms/px_llm_dec_b2_l16.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let want = interp(&ir);
    let em = device_emission(&ir, EmitOpts::exact(), &[]);
    assert_eq!(traps(&em.text), 0);
    let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap());
    for mode in MODES {
        let (got, _) = ex.run(&mode);
        assert_eq!(got, want, "{mode:?}");
    }
}

/// One generated program: the facts' coverage and the values on the device
/// (every mode) and on the host against the interpreter; the OFF arm's values
/// on a quarter of them.
fn check_onto(clang: &str, i: usize, p: &testgen::OntoProg) -> Vec<(&'static str, u64)> {
    let src = testgen::onto_src(p);
    let ir = lower(&src);
    let want = interp(&ir);
    let mut c = vec![("programs", 1)];
    let (n, proven, armed) = proof_counts(&ir);
    let off_ir = ir.refacted(FactArms {
        cond_bounds: false,
        ..FactArms::default()
    });
    let (_, proven_off, _) = proof_counts(&off_ir);
    c.push(("indexes", n as u64));
    c.push(("proven", proven as u64));
    c.push((
        "proven-by-conditions",
        (proven - proven_off.min(proven)) as u64,
    ));
    c.push(("guard-gated-arms", armed as u64));
    if proven == n {
        c.push(("trap-free", 1));
    } else {
        c.push(("checked-kept", 1));
    }
    let em = device_emission(&ir, EmitOpts::default(), &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let kernels = dev.matches("define ptx_kernel").count();
    c.push(("device-kernels", kernels as u64));
    if em.text.lines().any(|l| {
        l.starts_with("; mapal.move") && l.contains(":device") && l.contains("h2d=0 d2h=0")
    }) {
        c.push(("device-carried", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for mode in MODES {
        let (got, _) = ex.run(&mode);
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    let cpu = emu::run_emulated(
        clang,
        &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
        "",
    );
    assert_eq!(cpu, want, "cpu\n{src}");
    if i % 4 == 0 {
        let off = device_emission(&ir, EmitOpts::default(), &["cond-bounds"]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "cond-bounds off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// Generated programs (`testgen::onto_prog_strategy`): guarded partner
/// reads in seven forms, counters of four continue forms up and down as
/// table positions (some under a guard too), the exit state and an exit
/// counter read in later bodies. Every reach counter is non-zero.
#[test]
fn onto_device_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::OntoProg> = std::iter::repeat_with(|| {
        testgen::onto_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(96)
    .collect();
    let results: Vec<Vec<(&'static str, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = progs
            .chunks(progs.len().div_ceil(emu::job_slots()))
            .enumerate()
            .map(|(ci, chunk)| {
                let clang = clang.clone();
                let base = ci * progs.len().div_ceil(emu::job_slots());
                sc.spawn(move || {
                    chunk
                        .iter()
                        .enumerate()
                        .flat_map(|(i, p)| check_onto(&clang, base + i, p))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("onto-device generated: {c:?}");
    for key in [
        "programs",
        "proven-by-conditions",
        "guard-gated-arms",
        "trap-free",
        "checked-kept",
        "device-kernels",
        "device-carried",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// A native build of `ir` (the host's CPU build) run once: its exit code and
/// stdout.
fn native(clang: &str, ir: &CategoryIr) -> (i32, String) {
    let em = emit_detailed(ir, &EmitOpts::default()).expect("emits");
    let dir = tempfile::tempdir().unwrap();
    let ll = dir.path().join("p.ll");
    let exe = dir.path().join("p");
    std::fs::write(&ll, &em.text).unwrap();
    let _slot = emu::slot();
    let ok = std::process::Command::new(clang)
        .args(["-O1", "-Wno-override-module"])
        .arg(&ll)
        .arg(emu::rt_lib())
        .args(["-lm", "-lpthread", "-ldl", "-o"])
        .arg(&exe)
        .status()
        .unwrap()
        .success();
    assert!(ok, "link");
    let out = std::process::Command::new(&exe)
        .env("MAPAL_PAR", "2")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// `bounds_proof` is sound on programs whose offsets, thresholds,
/// comparisons, steps and limits are drawn freely
/// (`testgen::bounds_fuzz_strategy`): the native build, which drops every
/// proven read's check, traps (exit 101) exactly where the interpreter traps
/// and prints its values where it finishes — a proven read that left its
/// array would read past it unchecked instead.
#[test]
fn bounds_proof_sound_on_fuzzed_offsets() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::BoundsFuzz> = std::iter::repeat_with(|| {
        testgen::bounds_fuzz_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(240)
    .collect();
    let results: Vec<Vec<(&'static str, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = progs
            .chunks(progs.len().div_ceil(emu::job_slots()))
            .map(|chunk| {
                let clang = clang.clone();
                sc.spawn(move || {
                    chunk
                        .iter()
                        .flat_map(|p| {
                            let src = testgen::bounds_fuzz_src(p);
                            let ir = lower(&src);
                            let rr = run(&ir, 2_000_000_000);
                            let (n, proven, armed) = proof_counts(&ir);
                            let (code, out) = native(&clang, &ir);
                            let mut c = vec![
                                ("programs", 1),
                                ("indexes", n as u64),
                                ("proven", proven as u64),
                                ("guard-gated-arms", armed as u64),
                            ];
                            match rr.outcome {
                                Outcome::Done(_) => {
                                    assert_eq!((code, &out), (0, &rr.output), "{src}");
                                    c.push(("done", 1));
                                }
                                Outcome::Trapped(_) => {
                                    assert_eq!(code, 101, "{src}\n{out}");
                                    c.push(("trapped", 1));
                                    if proven > 0 {
                                        c.push(("trapped-with-proofs", 1));
                                    }
                                }
                                ref o => panic!("{o:?}\n{src}"),
                            }
                            c
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("bounds fuzz: {c:?}");
    for key in [
        "done",
        "trapped",
        "trapped-with-proofs",
        "proven",
        "guard-gated-arms",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// `host-par`: a GPU build's host maps — `h` reads `g`, which a trap-capable
/// gather of the device's `x` keeps on the host, and `perm` beside it — run
/// as split tasks on the pool (slice functions over `[lo, hi)`), the CPU
/// build's rule; the values are the interpreter's (default face: `x` within
/// its bound, so compared on the exact face) with 1, 2 and 4 pool lanes.
/// OFF arm: inline, one core.
#[test]
fn host_maps_of_a_gpu_build_split_on_the_pool() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = "fn main() {
    262144 -> iota -> r;
    32768 -> iota -> rq;
    64 -> iota -> kd;
    rq -> map { e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; v / 16.0 } -> q;
    rq -> map { e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 } -> w;
    r -> map { e -> e / 512 -> i; e % 512 -> j; (0.0, kd) -> fold { acc, d -> acc + q[i * 64 + d] * w[d * 512 + j] } } -> x;
    r -> map { t -> (t * 7 + 3) % 262144 } -> perm;
    r -> map { c -> x[perm[c]] } -> g;
    r -> map { c -> g[c] * 2.0 + g[(c + 1) % 262144] } -> h;
    h[0] -> println;
    h[262143] -> println;
    0.0 -> zero: f32;
    (zero, h) -> fold { acc, v -> acc + v } -> total;
    total -> println;
}
";
    let ir = lower(src);
    let want = interp(&ir);
    let shipped = |off: &[&str]| {
        let mut gpu = GpuArms::default();
        for r in off.iter().chain(&["async", "pdl", "streams", "pinned"]) {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "x86-64-v3",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::exact()
        };
        emit_detailed(&ir, &opts).expect("emits")
    };
    let on = shipped(&[]);
    assert!(on.text.contains("@mapal_main_hpar"), "{}", on.text);
    assert!(on.text.contains("call ptr @mapal_par_begin(i32 1)"));
    let ex = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap_or(""));
    for lanes in ["1", "2", "4"] {
        let (got, _) = ex.run(&[("MAPAL_PAR", lanes)]);
        assert_eq!(got, want, "MAPAL_PAR={lanes}");
    }
    let off = shipped(&["host-par"]);
    assert!(!off.text.contains("@mapal_main_hpar"));
    let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap_or(""));
    assert_eq!(got, want);
}

/// `host-par` on generated programs (`testgen::host_par_strategy`): every
/// host map chain's values through the emulator (exact face) equal the
/// interpreter's at 1 and 3 pool lanes, and the OFF arm's; the split maps
/// are counted (non-zero).
#[test]
fn host_par_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::HostParProg> = std::iter::repeat_with(|| {
        testgen::host_par_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(12)
    .collect();
    let results: Vec<Vec<(&'static str, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = progs
            .chunks(progs.len().div_ceil(emu::job_slots()))
            .map(|chunk| {
                let clang = clang.clone();
                sc.spawn(move || {
                    chunk
                        .iter()
                        .flat_map(|p| {
                            let src = testgen::host_par_src(p);
                            let ir = lower(&src);
                            let want = interp(&ir);
                            let build = |off: &[&str]| {
                                let mut gpu = GpuArms::default();
                                for r in off.iter().chain(&["async", "pdl", "streams", "pinned"]) {
                                    *gpu.off.rung(r).expect("rung") = true;
                                }
                                let opts = EmitOpts {
                                    host: "x86-64-v3",
                                    target: "rtx-5080",
                                    gpu,
                                    ..EmitOpts::exact()
                                };
                                emit_detailed(&ir, &opts).expect("emits")
                            };
                            let on = build(&[]);
                            let split = on
                                .text
                                .lines()
                                .filter(|l| l.starts_with("define internal void @mapal_main_hpar"))
                                .count();
                            let ex = emu::build_emulated(
                                &clang,
                                &on.text,
                                on.device.as_deref().unwrap_or(""),
                            );
                            for lanes in ["1", "3"] {
                                let (got, _) = ex.run(&[("MAPAL_PAR", lanes)]);
                                assert_eq!(got, want, "MAPAL_PAR={lanes}\n{src}");
                            }
                            let off = build(&["host-par"]);
                            assert!(!off.text.contains("@mapal_main_hpar"));
                            let got = emu::run_emulated(
                                &clang,
                                &off.text,
                                off.device.as_deref().unwrap_or(""),
                            );
                            assert_eq!(got, want, "host-par off\n{src}");
                            vec![("programs", 1), ("split-maps", split as u64)]
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("host-par generated: {c:?}");
    assert!(c.get("split-maps").copied().unwrap_or(0) > 0, "{c:?}");
}
