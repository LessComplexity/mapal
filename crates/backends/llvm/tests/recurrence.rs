//! recurrence-precision item 1: prefill and decode emerge from the loop's
//! recurrence. The facts (`trip_split`, `whole_fold`) on the bench programs
//! and on unseen ones; the rungs read off the emitted host IR against their
//! OFF arms; values on the emulated device against the interpreter.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

/// Item 1's rungs.
const RUNGS: [&str; 3] = ["range-cell", "scalar-fold", "trip-batch"];

/// `rtx-4090`, the exact face unless `free` (the default face), the rungs
/// `off` forced off (and those the emulator cannot run: `async`, `pdl`,
/// `streams`).
fn arm(ir: &CategoryIr, off: &[&str], free: bool) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams"].iter().chain(off) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    // The 4090 ships `fold-lanes` OFF (a `GPU_RUNGS_OFF` row); these tests
    // check the rule against its OFF arm, not the part's table.
    gpu.on.fold_lanes = true;
    let base = if free {
        EmitOpts::default()
    } else {
        EmitOpts::exact()
    };
    emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-4090",
            gpu,
            ..base
        },
    )
    .expect("emits")
}

/// Per program loop (`; mapal.move loop=…` to its latch, in its function):
/// the trip's `[uploads, copy-outs and device waits]` calls.
fn trip_calls(ll: &str) -> Vec<[usize; 2]> {
    let mut out = Vec::new();
    for f in ll.split("\ndefine ").skip(1) {
        let lines: Vec<&str> = f.lines().collect();
        for (h, l) in lines.iter().enumerate() {
            let Some(rest) = l.strip_prefix("; mapal.move loop=") else {
                continue;
            };
            let back = format!(
                "br label %{}",
                rest.split_whitespace().next().expect("header")
            );
            let latch = (h..lines.len())
                .rev()
                .find(|&i| lines[i].trim() == back)
                .expect("latch");
            let n = |k: &str| lines[h..latch].iter().filter(|x| x.contains(k)).count();
            out.push([
                n("@mapal_gpu_h2d"),
                n("@mapal_gpu_d2h") + n("@mapal_gpu_sync("),
            ]);
        }
    }
    out
}

/// The cell kernels' `; mapal.plan` lines carrying `key` in the device module.
fn cells_with(dev: &Option<String>, key: &str) -> usize {
    dev.iter()
        .flat_map(|d| d.lines())
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" kind=cell ") && l.contains(key))
        .count()
}

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn bench(name: &str) -> String {
    let p = format!(
        "{}/../../../benches/algorithms/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// The trip split of the four attention loops: decode and causal are one
/// recurrent interval (`x` row t reads `o` of trip t − 1, `o` reads `x`);
/// prefill's loop is the same from t = 8; the natural one-loop generator is
/// cut at its prompt guard — the prompt interval has no recurrence, `x` is
/// batched there and its trips but the last are skippable.
#[test]
fn trip_split_of_the_attention_loops() {
    for (name, lo, phases) in [
        ("decode_16_8", 0, 1),
        ("causal_16_8", 0, 1),
        ("prefill_16_8_8", 8, 1),
        ("gen_16_8_8", 0, 2),
    ] {
        let ir = lower_src(&bench(name));
        let ts = ir.trip_split(ir.entry());
        assert_eq!(ts.len(), 1, "{name}: {ts:?}");
        let s = &ts[0];
        assert_eq!(
            (s.lo, s.hi, s.phases.len()),
            (lo, 16, phases),
            "{name}: {s:?}"
        );
        let last = s.phases.last().expect("a phase");
        assert_eq!(last.recurrent.len(), 2, "{name}: x and o recur: {last:?}");
        assert!(last.batched.is_empty() && !last.skip, "{name}: {last:?}");
        assert!(
            !last.outside.is_empty(),
            "{name}: q is outside the recurrence: {last:?}"
        );
        if phases == 2 {
            let p = &s.phases[0];
            assert_eq!((p.lo, p.hi), (0, 8), "{name}: {p:?}");
            assert!(
                p.recurrent.is_empty() && p.batched.len() == 1 && p.skip,
                "{name}: {p:?}"
            );
        }
    }
}

/// Item 1b on the attention loops: with `range-cell` and `scalar-fold` every
/// recurrent node is on the device — the steady trip uploads and copies out
/// nothing and never waits on the device — where their OFF arm moves per trip.
#[test]
fn the_attention_trips_never_sync() {
    for name in ["decode_16_8", "causal_16_8", "prefill_16_8_8", "gen_16_8_8"] {
        let ir = lower_src(&bench(name));
        let (on, off) = (arm(&ir, &[], true), arm(&ir, &RUNGS, true));
        let (con, coff) = (trip_calls(&on.text), trip_calls(&off.text));
        assert_eq!(con, [[0, 0]], "{name}: per trip ON");
        assert!(
            coff.iter().all(|c| c[0] + c[1] > 0),
            "{name}: per trip OFF {coff:?}"
        );
    }
}

/// `whole_fold`: causal's `mx` (a running max) and `den` (a sum) are whole
/// folds of one row; decode has none.
#[test]
fn whole_folds_of_a_trip() {
    let ir = lower_src(&bench("causal_16_8"));
    let w = ir.whole_fold(ir.entry());
    let mut ops: Vec<_> = w
        .values()
        .map(|s| {
            (
                s.rows,
                s.k,
                s.a.ck,
                matches!(s.op, mapal_ir::RowFoldOp::Select { .. }),
            )
        })
        .collect();
    ops.sort_by_key(|o| o.3);
    assert_eq!(ops, [(1, 16, 1, false), (1, 16, 1, true)], "{w:?}");
    let ir = lower_src(&bench("decode_16_8"));
    assert!(ir.whole_fold(ir.entry()).is_empty());
}

/// **Item 1's rungs on unseen programs** (testgen `recur_prog_strategy`:
/// row-recurrent loops — a natural row write fed by the previous trip's row,
/// a prompt guard on the counter, sites, guarded scores, whole folds read back
/// per cell, vector·matrix cells): against the arm with `range-cell`,
/// `scalar-fold` and `trip-batch` off, each rule fires (counted from the
/// emission: a range cell, a cell reading a device fold, a batched entry) and
/// never in the OFF arm; per trip no more uploads or copy-outs than OFF, and
/// fewer over all programs; the emulated device prints the interpreter's
/// values on both arms and moves no more bytes per run.
#[test]
fn recurrence_rungs_on_unseen_programs() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP recurrence_rungs_on_unseen_programs: clang not found");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::recur_prog_strategy();
    let (mut fired, mut trips_on, mut trips_off, mut traps, mut inv_batch) =
        ([0usize; 3], 0usize, 0usize, 0usize, 0usize);
    // Valued programs with a guard in the written cell, a half row, two row-written states.
    let (mut shapes, mut shape_batch) = ([0usize; 3], [0usize; 3]);
    for z in 0..40 {
        let prog = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::recur_src(&prog);
        let ir = lower_src(&src);
        let rr = mapal_interp::run(&ir, 400_000_000);
        let trapped = matches!(rr.outcome, mapal_interp::Outcome::Trapped(_));
        let want = rr.output;
        let (on, off) = (arm(&ir, &[], false), arm(&ir, &RUNGS, false));
        let marks = |e: &Emission| {
            [
                cells_with(&e.device, " range="),
                cells_with(&e.device, " dev="),
                e.text.matches("; mapal.batch ").count(),
            ]
        };
        assert_eq!(marks(&off), [0; 3], "#{z} OFF\n{src}");
        for (f, m) in fired.iter_mut().zip(marks(&on)) {
            *f += m;
        }
        // The prompt row reading the loop invariant `w` batches too.
        if prog.var & 1 == 1 {
            inv_batch += marks(&on)[2];
        }
        let (con, coff) = (trip_calls(&on.text), trip_calls(&off.text));
        assert_eq!(con.len(), coff.len(), "#{z}\n{src}");
        for (a, b) in con.iter().zip(&coff) {
            assert!(
                a[0] <= b[0] && a[1] <= b[1],
                "#{z}: per trip {a:?} > OFF {b:?}\n{src}"
            );
            trips_on += a[0] + a[1];
            trips_off += b[0] + b[1];
        }
        if let (Some(a), Some(b)) = (on.device.as_deref(), off.device.as_deref()) {
            // A trap at an early (skippable) trip is the interpreter's on both
            // arms: `trip-batch` never drops it.
            if trapped {
                traps += 1;
                for (h, d, tag) in [(&on.text, a, "ON"), (&off.text, b, "OFF")] {
                    let r = std::panic::catch_unwind(|| emu::run_emulated_totals(&clang, h, d));
                    assert!(
                        r.is_err(),
                        "#{z} {tag}: the interpreter traps, the emulated run did not\n{src}"
                    );
                }
                continue;
            }
            let run = |h: &str, d: &str, tag: &str| {
                std::panic::catch_unwind(|| emu::run_emulated_totals(&clang, h, d))
                    .unwrap_or_else(|_| panic!("#{z} {tag}: the emulated run failed\n{src}"))
            };
            let (ra, ta) = run(&on.text, a, "ON");
            let (rb, tb) = run(&off.text, b, "OFF");
            assert_eq!(ra, want, "#{z} ON\n{src}");
            assert_eq!(rb, want, "#{z} OFF\n{src}");
            for (n, c) in shapes.iter_mut().enumerate() {
                *c += usize::from(prog.var & (16 << n) != 0);
                shape_batch[n] += usize::from(prog.var & (16 << n) != 0) * marks(&on)[2];
            }
            assert!(
                ta[3] + ta[4] <= tb[3] + tb[4],
                "#{z}: moved {ta:?} > OFF {tb:?}\n{src}"
            );
        }
    }
    eprintln!(
        "recurrence_rungs_on_unseen_programs: fired {fired:?} (range, dev fold, batch); per-trip calls {trips_on} vs OFF {trips_off}; {traps} trapping; {inv_batch} batches over an invariant; valued {shapes:?}, batches {shape_batch:?} (cell guard, half row, two states)"
    );
    assert!(
        shapes.iter().all(|&c| c > 0),
        "a row-write shape was never valued: {shapes:?}"
    );
    assert!(inv_batch > 0, "no batch read a loop invariant");
    assert!(
        fired.iter().all(|&f| f > 0),
        "a rule fired on no generated program: {fired:?}"
    );
    assert!(
        trips_on < trips_off,
        "no per-trip call saved: {trips_on} vs {trips_off}"
    );
}

/// `trip-batch` over the row-write shapes it was not built on (the generator's
/// `var` bits 4..6: a guard in the written cell, a half row, two row-written
/// states): each batches — two slots in one batch — and the emulated device
/// prints the interpreter's values on both arms.
#[test]
fn trip_batch_over_row_write_shapes() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP trip_batch_over_row_write_shapes: clang not found");
        return;
    };
    // bits 2..3 ≠ 0: no trap; 119 sets every shape bit plus `w` and row `it + 1`.
    for (var, slots) in [(20u8, 1usize), (36, 1), (68, 2), (119, 2)] {
        let prog = testgen::RecurProg {
            big: false,
            trips: 6,
            prompt: 3,
            ops: vec![[0, 0], [4, 0]],
            whole: true,
            var,
        };
        let src = testgen::recur_src(&prog);
        let ir = lower_src(&src);
        let want = mapal_interp::run(&ir, 400_000_000).output;
        let (on, off) = (arm(&ir, &[], false), arm(&ir, &RUNGS, false));
        assert!(
            on.text
                .contains(&format!("; mapal.batch enter=2 writes={slots}\n")),
            "var {var}: no batch\n{src}"
        );
        for (e, tag) in [(&on, "ON"), (&off, "OFF")] {
            let got = emu::run_emulated(
                &clang,
                &e.text,
                e.device.as_deref().expect("a device module"),
            );
            assert_eq!(got, want, "var {var} {tag}\n{src}");
        }
    }
}

/// **The natural generator is prefill's twin**: `gen_16_8_8` (one loop from
/// t = 0, a guard `t < 8` picking the prompt row) against `prefill_16_8_8`
/// (the prompt's attention written out before a loop from t = 8), on the
/// emulated device: the same outputs, and the natural form moves no more
/// (uploads, copy-outs and their bytes per run).
#[test]
fn the_natural_generator_is_prefills_twin() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP the_natural_generator_is_prefills_twin: clang not found");
        return;
    };
    let run = |name: &str| {
        let e = arm(&lower_src(&bench(name)), &[], false);
        emu::run_emulated_totals(
            &clang,
            &e.text,
            e.device.as_deref().expect("a device module"),
        )
    };
    let ((go, gt), (po, pt)) = (run("gen_16_8_8"), run("prefill_16_8_8"));
    let lines = |o: &str| o.lines().take(3).map(str::to_string).collect::<Vec<_>>();
    assert_eq!(lines(&go), lines(&po), "the twins' outputs");
    eprintln!("gen {gt:?} vs prefill {pt:?} ([h2d, d2h, launches, bytes in, bytes out] per run)");
    for i in [0, 1, 3, 4] {
        assert!(gt[i] <= pt[i], "gen moves more: {gt:?} vs prefill {pt:?}");
    }
}

/// The kernels of `kind` whose `; mapal.plan` line carries `key`.
fn plans_with(dev: &Option<String>, kind: &str, key: &str) -> usize {
    dev.iter()
        .flat_map(|d| d.lines())
        .filter(|l| {
            l.starts_with("; mapal.plan ")
                && l.contains(&format!(" kind={kind} "))
                && l.contains(key)
        })
        .count()
}

/// **Item 2's `cell-lanes` and `fold-lanes` on unseen programs** (testgen
/// `recur_prog_strategy`, the default face: order free): each against its own
/// OFF arm, a nest cell whose fold regroups (`CellSite::tree`) is a warp a
/// cell, and a device row fold (a whole fold included; `flow` off, whose
/// tree gate leaves such folds at scale) a warp a row —
/// counted from the emission, never in the OFF arm, the module combining by
/// ≥ 5 butterfly shuffles a lane kernel — and the emulated device prints the
/// interpreter's values within the regrouping's rounding on both arms.
#[test]
fn lanes_on_unseen_programs() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP lanes_on_unseen_programs: clang not found");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::recur_prog_strategy();
    let nums = |o: &str| {
        o.lines()
            .map(|l| l.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
    };
    let mut fired = [0usize; 2];
    for z in 0..24 {
        let src = testgen::recur_src(&strat.new_tree(&mut runner).unwrap().current());
        let ir = lower_src(&src);
        let rr = mapal_interp::run(&ir, 400_000_000);
        if matches!(rr.outcome, mapal_interp::Outcome::Trapped(_)) {
            continue;
        }
        let want = nums(&rr.output);
        // `fold-lanes` takes the row folds `flow` leaves (its tree gate):
        // with `flow` off every generated row fold is one.
        for (n, (rung, kind, base)) in [
            ("cell-lanes", "cell", &[][..]),
            ("fold-lanes", "rowfold", &["flow"][..]),
        ]
        .into_iter()
        .enumerate()
        {
            let on = arm(&ir, base, true);
            let off = arm(&ir, &[base, &[rung]].concat(), true);
            assert_eq!(
                plans_with(&off.device, kind, " lanes="),
                0,
                "#{z} {rung} OFF\n{src}"
            );
            let lanes = plans_with(&on.device, kind, " lanes=32");
            fired[n] += lanes;
            if lanes == 0 {
                continue;
            }
            let dev = on.device.as_deref().expect("a device module");
            let shfl = dev
                .matches("call float @llvm.nvvm.shfl.sync.bfly.f32(")
                .count();
            assert!(
                shfl >= 5 * lanes,
                "#{z} {rung}: {shfl} shuffles for {lanes} lane kernels\n{src}"
            );
            let a = emu::run_emulated(&clang, &on.text, dev);
            let b = emu::run_emulated(&clang, &off.text, off.device.as_deref().expect("dev"));
            // Both arms against the interpreter (the oracle). ponytail: relative
            // tolerance — the composed programs print no per-cell M (ADR-0040's
            // |seed| + Σ|a·b|); the per-cell bound is gated where M is printed
            // (splitk, tf32_tile).
            for (out, tag) in [(&a, "ON"), (&b, "OFF")] {
                let x = nums(out);
                assert_eq!(
                    x.len(),
                    want.len(),
                    "#{z} {rung} {tag}\n{out}\n{}\n{src}",
                    rr.output
                );
                for (u, v) in x.iter().zip(&want) {
                    let same = u == v || (u.is_nan() && v.is_nan());
                    assert!(
                        same || (u - v).abs() <= 1e-3 * v.abs().max(1.0),
                        "#{z} {rung} {tag}: {out:?} vs {:?}\n{src}",
                        rr.output
                    );
                }
            }
        }
    }
    eprintln!(
        "lanes_on_unseen_programs: {} lane cells, {} lane row folds",
        fired[0], fired[1]
    );
    assert!(
        fired.iter().all(|&f| f > 0),
        "a lane rule fired on no generated program: {fired:?}"
    );
}

/// `fold_tree_step` (`CellSite::tree`): causal's `s` and `o` nest cells each
/// fold once by `+` (a dot product), so both may run as lanes.
#[test]
fn nest_cells_whose_fold_regroups() {
    let ir = lower_src(&bench("causal_16_8"));
    let trees: Vec<_> = ir
        .cell_plan(ir.entry())
        .values()
        .filter_map(|c| c.tree)
        .map(|(_, op, k)| (op, k))
        .collect();
    let add = |op| {
        matches!(
            op,
            mapal_ir::RowFoldOp::Arith {
                op: mapal_ir::Operation::Add,
                ..
            }
        )
    };
    assert!(
        trees.len() >= 2 && trees.iter().all(|&(op, k)| add(op) && k >= 8),
        "{trees:?}"
    );
}
