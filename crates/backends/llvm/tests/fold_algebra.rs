//! fold-algebra: `step_algebra` — the dataflow fact FMA contraction and fold
//! regrouping are read from (`mapal_ir::StepAlgebra`), its realization on the
//! host (`FnEmit::emit_arith` flags) and its device consequence (every tile
//! site's step is an FMA pair, so `Arithmetic::scalar` fuses on `contract`
//! whatever the lane stride).
//!
//! Generated programs the change was not designed on (`testgen`): the fact's
//! invariants, tile sites ⊂ FMA pairs, and values against the interpreter —
//! bit for bit on the exact face, within ADR-0040 D2's bound on the default
//! face, also with both OFF arms.

use std::process::Command;

use mapal_backend_llvm::{EmitOpts, emit_with_opts};
use mapal_interp::run;
use mapal_ir::{CategoryIr, FuncId, Operation, RowFoldOp};
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

fn opts(base: EmitOpts, off: &[&str]) -> EmitOpts {
    let mut o = EmitOpts {
        host: "generic",
        target: "generic",
        ..base
    };
    for a in off {
        assert!(mapal_backend_llvm::flags::apply(&mut o, a).unwrap(), "{a}");
    }
    o
}

const DOT: &str = "fn main() {
    4096 -> iota -> ix;
    ix -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> x;
    ix -> map { t -> (t * 5 + 3) % 31 - 15 -> widen_f32 } -> y;
    (x, y) -> zip -> xy;
    (0.0, xy) -> fold { acc, p -> acc + p.0 * p.1 } -> d;
    d -> println;
    (0.0, xy) -> fold { acc, p -> acc - p.0 * p.1 } -> e;
    e -> println;
    (0.0, x) -> fold { acc, v -> acc + v } -> s;
    s -> println;
    (-1000.0, x) -> fold { acc, v -> (v > acc) -> { -true-> v; -false-> acc; } } -> mx;
    mx -> println;
    (0, ix) -> fold { acc, v -> acc + v } -> si;
    si -> println;
    xy -> map { p -> 2.5 * p.0 + p.1 } -> z;
    z[7] -> println;
}
";

/// Every fold of the program: `(body, op of its regroup, exact, k, reads)`
/// and the FMA pairs of each function.
fn folds(ir: &CategoryIr) -> Vec<(FuncId, Option<mapal_ir::Regroup>)> {
    ir.morphisms()
        .filter_map(|(_, m)| match m.op {
            Operation::Fold { body, .. } => Some((body, ir.step_algebra(body).regroup)),
            _ => None,
        })
        .collect()
}

#[test]
fn the_fact_on_known_steps() {
    let ir = lower(DOT);
    let fs = folds(&ir);
    assert_eq!(fs.len(), 5, "{fs:?}");
    let (mut dot, mut sub, mut sum, mut max, mut int) = (0, 0, 0, 0, 0);
    for (body, g) in &fs {
        let alg = ir.step_algebra(*body);
        match (alg.fma.values().next(), g) {
            // `acc + p.0·p.1`: a pair and a float regroup; the factors are
            // components of the folded element — unit stride along `k`.
            (Some(p), Some(g)) if !p.sub => {
                assert!(!g.exact && g.k == 4096 && alg.fused_mul.contains_key(p.mul));
                assert_eq!(g.reads, [Some(1), Some(1)], "{g:?}");
                dot += 1;
            }
            // `acc − p.0·p.1`: a (sub) pair; `−` does not regroup.
            (Some(p), None) => {
                assert!(p.sub);
                sub += 1;
            }
            (None, Some(g)) => match g.op {
                RowFoldOp::Arith { .. } if g.exact => int += 1,
                RowFoldOp::Arith { .. } => {
                    assert_eq!(g.reads, [Some(1)], "{g:?}");
                    sum += 1;
                }
                RowFoldOp::Select { .. } => {
                    assert!(!g.exact);
                    max += 1;
                }
            },
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!((dot, sub, sum, max, int), (1, 1, 1, 1, 1));
    // The map's `2.5·p.0 + p.1` is a pair of the map body.
    let map_pairs: usize = ir
        .morphisms()
        .filter_map(|(_, m)| match m.op {
            Operation::Map { body, .. } => Some(ir.step_algebra(body).fma.len()),
            _ => None,
        })
        .sum();
    assert!(map_pairs >= 1);
}

/// The realization: flags per face, and what LLVM makes of them on this host.
#[test]
fn flags_and_realized_intents() {
    let ir = lower(DOT);
    let ll = |base: EmitOpts, off: &[&str]| emit_with_opts(&ir, &opts(base, off)).unwrap();
    let count = |t: &str, pat: &str| t.lines().filter(|l| l.contains(pat)).count();
    let def = ll(EmitOpts::default(), &[]);
    // Free order: the dot and the float sum regroup; the pairs contract.
    assert_eq!(count(&def, "fadd reassoc contract float"), 1, "{def}");
    assert_eq!(count(&def, "fadd reassoc float"), 1);
    assert!(count(&def, "fmul contract float") >= 3);
    assert!(count(&def, "fsub contract float") >= 1);
    // Sequential contract face: a bit-for-bit oracle compares host and
    // device, so each pair is one explicit `llvm.fma` (the device tile's
    // `OrderedFma`), never a flag a target without FMA would not fuse; nothing
    // regroups.
    let seq = ll(EmitOpts::default(), &["--order=sequential"]);
    assert_eq!(count(&seq, " reassoc"), 0);
    assert!(count(&seq, "call float @llvm.fma.f32(") >= 3, "{seq}");
    assert!(seq.contains("declare float @llvm.fma.f32("));
    assert!(count(&seq, " = fneg float ") >= 1, "the sub pair");
    // The exact face and both OFF arms.
    let exact = ll(EmitOpts::exact(), &[]);
    assert_eq!(count(&exact, " reassoc") + count(&exact, " contract"), 0);
    let nopairs = ll(EmitOpts::default(), &["--no-fma-pairs"]);
    assert_eq!(count(&nopairs, " contract"), 0);
    assert_eq!(count(&nopairs, " reassoc"), 2);
    let noreg = ll(EmitOpts::default(), &["--no-fold-regroup"]);
    assert_eq!(count(&noreg, " reassoc"), 0);

    // Realized (x86-64 with FMA): the regrouped float folds vectorize into
    // lane partials; their OFF arm stays a scalar chain.
    let Some(clang) = emu::clang() else {
        return;
    };
    if !cfg!(target_arch = "x86_64") {
        return;
    }
    let asm = |ll: &str| {
        let dir = tempfile::tempdir().unwrap();
        let (src, out) = (dir.path().join("p.ll"), dir.path().join("p.s"));
        std::fs::write(&src, ll).unwrap();
        let ok = Command::new(&clang)
            .args(["-O2", "-march=x86-64-v3", "-S", "-Wno-override-module"])
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "clang -S failed");
        std::fs::read_to_string(out).unwrap()
    };
    let (on, off) = (asm(&def), asm(&noreg));
    let vec_fma = |s: &str| count(s, "vfmadd") - count(s, "ss ");
    assert!(
        count(&on, "vaddps") > 0 && vec_fma(&on) > 0,
        "no vector partials"
    );
    assert_eq!(count(&off, "vaddps"), 0, "OFF arm vectorized a fold");
}

/// The program's stdout, or `None` when it exited non-zero (a trap).
fn run_native(clang: &str, ll: &str, tag: &str) -> Option<String> {
    // One of the binary's child slots (programs are checked in parallel).
    let _slot = emu::slot();
    let dir = tempfile::tempdir().unwrap();
    let (src, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&src, ll).unwrap();
    let ok = Command::new(clang)
        .args(["-O2", "-march=native", "-Wno-override-module"])
        .arg(&src)
        .arg(emu::rt_lib())
        .args(["-lpthread", "-ldl", "-lm", "-o"])
        .arg(&exe)
        .status()
        .is_ok_and(|s| s.success());
    assert!(ok, "{tag}: clang failed");
    let out = Command::new(&exe).output().unwrap();
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The default face's values against the interpreter's: every numeric token
/// within a relative 1e-4 of the larger magnitude (1e-4 absolute below 1);
/// identical tokens (an infinity, a NaN) match.
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

/// The fact's invariants on one graph; returns `(pairs, regroups, tile sites)`.
fn invariants(ir: &CategoryIr, tag: &str) -> (usize, usize, usize) {
    let (mut pairs, mut regroups, mut tiles) = (0, 0, 0);
    for (f, fd) in ir.funcs() {
        let alg = ir.step_algebra(f);
        for (s, p) in alg.fma.iter() {
            let (sm, mm) = (ir.morphism(s).unwrap(), ir.morphism(p.mul).unwrap());
            assert!(matches!(sm.op, Operation::Add | Operation::Sub), "{tag}");
            assert_eq!(mm.op, Operation::Mul, "{tag}");
            assert_eq!(ir.out_edges(mm.target).len(), 1, "{tag}: shared product");
            assert!(
                fd.morphisms.contains(&p.mul),
                "{tag}: pair across functions"
            );
            assert_eq!(alg.fused_mul.get(p.mul), Some(&s), "{tag}");
            pairs += 1;
        }
        if let Some(g) = &alg.regroup {
            assert_eq!(
                ir.in_edges(fd.output),
                [g.step],
                "{tag}: step is the output"
            );
            assert_eq!(ir.fold_tree_step(g.fold), Some(g.op), "{tag}");
            regroups += 1;
        }
    }
    // Every tile site's step is an FMA pair of its fold body: the device's
    // fused step is the algebra's.
    for f in ir.funcs().map(|(f, _)| f) {
        for (m, _) in ir.tile_plan_strided(f).sites.iter() {
            let Operation::Map { body, .. } = ir.morphism(m).unwrap().op else {
                continue;
            };
            let fold_body = ir.func(body).unwrap().morphisms.iter().find_map(|&x| {
                match ir.morphism(x).unwrap().op {
                    Operation::Fold { body, .. } => Some(body),
                    _ => None,
                }
            });
            let Some(fb) = fold_body else { continue };
            let out = ir.func(fb).unwrap().output;
            let step = ir.in_edges(out)[0];
            assert!(
                ir.step_algebra(fb).fma.contains_key(step),
                "{tag}: tile site {m:?} is not an FMA pair"
            );
            tiles += 1;
        }
    }
    (pairs, regroups, tiles)
}

#[test]
fn generated_programs_keep_values_and_invariants() {
    let clang = emu::clang();
    let mut runner = TestRunner::deterministic();
    let chains = testgen::chain_strategy();
    let recurs = testgen::recur_prog_strategy();
    let radixes = testgen::radix_prog_strategy();
    let geoms = testgen::geom_prog_strategy();
    // Chains (maps, cells, tile sites, row / whole folds, loops),
    // recurrences (attention-like trips: max / sum row folds, products),
    // radix (implicit-GEMM digit reads) and strided geometry sites.
    let srcs: Vec<String> = (0..64)
        .map(|z| match z % 4 {
            0 => chains.new_tree(&mut runner).unwrap().current().source(),
            1 => testgen::recur_src(&recurs.new_tree(&mut runner).unwrap().current()),
            2 => testgen::radix_src(&radixes.new_tree(&mut runner).unwrap().current()).0,
            _ => testgen::geom_src(&geoms.new_tree(&mut runner).unwrap().current()).0,
        })
        .collect();
    // Per program: pairs, regroups, tiles, contracted, reassoc, valued, trapped, diverged.
    let counts = testgen::par_check(&srcs, |z, src| {
        let mut c = [0usize; 8];
        let tag = format!("gen#{z}");
        let ir = lower(&src);
        let (p, g, t) = invariants(&ir, &tag);
        (c[0], c[1], c[2]) = (p, g, t);
        let exact = emit_with_opts(&ir, &opts(EmitOpts::exact(), &[])).unwrap();
        assert!(
            !exact.contains(" reassoc") && !exact.contains(" contract"),
            "{tag}"
        );
        let def = emit_with_opts(&ir, &opts(EmitOpts::default(), &[])).unwrap();
        c[3] = def.matches("fmul contract").count();
        c[4] = def.matches(" reassoc").count();
        let Some(clang) = &clang else { return c };
        let rr = run(&ir, 400_000_000);
        // A program the interpreter traps on traps compiled too; one past the
        // interpreter's step budget has no oracle here.
        match rr.outcome {
            mapal_interp::Outcome::Done(_) => {}
            mapal_interp::Outcome::Trapped(_) => {
                assert!(run_native(clang, &exact, &tag).is_none(), "{tag}: no trap");
                c[6] = 1;
                return c;
            }
            mapal_interp::Outcome::Diverged => {
                c[7] = 1;
                return c;
            }
        }
        let out =
            |ll: &str| run_native(clang, ll, &tag).unwrap_or_else(|| panic!("{tag} trap\n{src}"));
        assert_eq!(out(&exact), rr.output, "{tag} exact\n{src}");
        let got = out(&def);
        assert!(
            within(&got, &rr.output),
            "{tag} default\n{got}\n{}\n{src}",
            rr.output
        );
        for off in ["--no-fma-pairs", "--no-fold-regroup"] {
            let ll = emit_with_opts(&ir, &opts(EmitOpts::default(), &[off])).unwrap();
            assert!(within(&out(&ll), &rr.output), "{tag} {off}\n{src}");
        }
        c[5] = 1;
        c
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (pairs, regroups, tiles, contracted, reassoc) = (sum(0), sum(1), sum(2), sum(3), sum(4));
    let (valued, trapped, diverged) = (sum(5), sum(6), sum(7));
    eprintln!(
        "fold_algebra: {pairs} pairs, {regroups} regroups, {tiles} tile sites; default face: \
         {contracted} contracted muls, {reassoc} reassoc steps; {valued} valued, {trapped} trapped, {diverged} past the budget"
    );
    assert!(pairs > 0 && regroups > 0 && tiles > 0 && contracted > 0 && reassoc > 0);
    assert!(valued >= 32, "{valued} valued programs");
}
