//! gapfix `iota-law` (gaps-disasm C8): an array a body fn captures takes
//! the element law of what its sites pass (`elem_plan` through captures) —
//! a fold over a captured `iota` computes its index and loads nothing (the
//! decode norm's `(0.0, rD) -> fold` over `rD = D -> iota` outside the loop).
//! A capture with no closed law (a mapped array) is loaded as before: its
//! emission is the OFF arm's, byte for byte.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
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

fn emission(ir: &CategoryIr, base: EmitOpts, target: &'static str, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "streams", "pinned"].iter().chain(off) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    gpu.site_cands = 1;
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// A norm loop: `rows` rows of `d`, `trips` steps, each step's per-row
/// root-mean-square (a fold over the captured `rD`, then a Newton fold over
/// the captured `r4`) scaling the rows. `law`: 0 — `rD` an `iota` outside
/// the loop; 1 — `rD` an `iota` bound inside the loop body; 2 — `rD` a mapped
/// permutation (no closed law: the control).
#[derive(Clone, Debug)]
struct NormProg {
    rows: u64,
    d: u64,
    trips: i64,
    law: u8,
    mul: u64,
}

fn norm_strategy() -> impl Strategy<Value = NormProg> {
    (
        prop::sample::select(vec![1u64, 3, 8, 32, 40]),
        prop::sample::select(vec![32u64, 64, 96, 256]),
        1i64..4,
        0u8..3,
        prop::sample::select(vec![3u64, 5, 7]),
    )
        .prop_map(|(rows, d, trips, law, mul)| NormProg {
            rows,
            d,
            trips,
            law,
            mul,
        })
}

fn norm_src(p: &NormProg) -> String {
    let (r, d, n) = (p.rows, p.d, p.rows * p.d);
    let rd = match p.law {
        0 => format!("    {d} -> iota -> rD;\n"),
        1 => String::new(),
        _ => format!(
            "    {d} -> iota -> iD;\n    iD -> map {{\n        i -> (i * {m}) % {d}\n    }} -> rD;\n",
            m = if p.d % p.mul == 0 { 1 } else { p.mul }
        ),
    };
    let inner = match p.law {
        1 => format!("                {d} -> iota -> rD;\n"),
        _ => String::new(),
    };
    // The Newton fold's own captured range: an `iota`, or (the control) a
    // mapped permutation, so the control captures no closed law at all.
    let r4 = match p.law {
        2 => "    4 -> iota -> i4;\n    i4 -> map {\n        i -> (i * 3) % 4\n    } -> r4;\n",
        _ => "    4 -> iota -> r4;\n",
    };
    format!(
        "fn main() {{
    {n} -> iota -> ix;
    ix -> map {{
        t -> (t * {m} + 5) % 9 - 4 -> widen_f32
    }} -> x0;
    {r} -> iota -> rR;
    {n} -> iota -> rRD;
{r4}{rd}    mut x: [f32; {n}] <- x0;
    mut t: i32 <- 0;
    loop {{
        (t < {trips}) -> {{
            -true-> {{
{inner}                rR -> map {{
                    b -> (0.0, rD) -> fold {{
                        acc, i -> x[b * {d} + i] -> v;
                        acc + v * v
                    }} -> ss;
                    ss / {d}.0 + 1.0 -> ra;
                    (ra, r4) -> fold {{
                        acc, k -> 0.5 * (acc + ra / acc)
                    }} -> sd;
                    1.0 / sd
                }} -> rinv;
                rRD -> map {{
                    e -> x[e] * rinv[e / {d}] + 0.25
                }} -> x;
                t + 1 -> t;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
    out[{mid}] -> println;
}}
",
        m = p.mul,
        trips = p.trips,
        last = n - 1,
        mid = n / 2 + 1,
    )
}

fn close(a: &str, b: &str, tol: f64) -> bool {
    a == b
        || match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(y)) => (x - y).abs() <= tol * x.abs().max(y.abs()).max(1e-30),
            _ => false,
        }
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

#[derive(Debug, Default, Clone, Copy)]
struct Cover {
    programs: u64,
    fired_host: u64,
    fired_device: u64,
    control_equal: u64,
}

fn check(clang: &str, p: &NormProg) -> Cover {
    let src = norm_src(p);
    let formatted = mapal_syntax::format_source(&src).expect("formats");
    assert_eq!(formatted, src, "the generator writes formatted source");
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    let mut c = Cover {
        programs: 1,
        ..Cover::default()
    };
    // The host (a CPU profile): the law's index in place of the load.
    let (h_on, h_off) = (
        emission(&ir, EmitOpts::exact(), "generic", &[]),
        emission(&ir, EmitOpts::exact(), "generic", &["iota-law"]),
    );
    // The device (RTX 5080), the exact face and the default face (the
    // warp-lane cells).
    let mut dev = Vec::new();
    for (face, base) in [
        ("exact", EmitOpts::exact()),
        ("default", EmitOpts::default()),
    ] {
        let on = emission(&ir, base.clone(), "rtx-5080", &[]);
        let off = emission(&ir, base, "rtx-5080", &["iota-law"]);
        dev.push((face, on, off));
    }
    let moved = |a: &Emission, b: &Emission| a.text != b.text || a.device != b.device;
    match p.law {
        // A mapped `rD` has no closed law: nothing moves.
        2 => {
            assert!(!moved(&h_on, &h_off), "control moved (host)\n{src}");
            for (face, on, off) in &dev {
                assert!(!moved(on, off), "control moved ({face})\n{src}");
            }
            c.control_equal = 1;
        }
        // An `iota` outside the loop: the captured law is read.
        0 => {
            assert!(moved(&h_on, &h_off), "no law read (host)\n{src}");
            c.fired_host = 1;
            c.fired_device = u64::from(dev.iter().any(|(_, on, off)| moved(on, off)));
        }
        _ => {
            c.fired_host = u64::from(moved(&h_on, &h_off));
            c.fired_device = u64::from(dev.iter().any(|(_, on, off)| moved(on, off)));
        }
    }
    // Values: the device arms in every emulator mode — the exact face the
    // interpreter's bytes, the default face within a relative 2⁻¹⁶ (the rows'
    // sums of squares regroup; the Newton root and quotient are the
    // interpreter's).
    for (face, on, off) in &dev {
        for e in [on, off] {
            let d = e.device.as_deref().unwrap_or("").to_string();
            let built = emu::build_emulated(clang, &e.text, &d);
            for mode in MODES {
                let (got, _) = built.run(&mode);
                if *face == "exact" {
                    assert_eq!(got, want, "{face} {mode:?}\n{src}");
                    continue;
                }
                for (a, b) in got.lines().zip(want.lines()) {
                    assert!(
                        close(a, b, 2f64.powi(-16)),
                        "{face} {mode:?}: {a} vs {b}\n{src}"
                    );
                }
            }
        }
    }
    c
}

/// 24 generated norm loops: a fold over a captured `iota` reads its law (the
/// emission moves, host and device), a mapped capture does not (the OFF arm's
/// text, byte for byte); values in every emulator mode on both arms.
#[test]
fn iota_law_on_generated_loops() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<NormProg> = (0..24)
        .map(|_| norm_strategy().new_tree(&mut runner).unwrap().current())
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c.programs += k.programs;
        c.fired_host += k.fired_host;
        c.fired_device += k.fired_device;
        c.control_equal += k.control_equal;
    }
    eprintln!("iota-law generated: {c:?}");
    let all = [c.programs, c.fired_host, c.fired_device, c.control_equal];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
