//! bound-blkdec: the default face of a program whose products admit `bf16`
//! answers to the composed bound (ADR-0041 D5 `bound(o)`, ADR-0049 D1:
//! `mapal_interp::run_bound`), not to one site's coefficient. On generated
//! decode blocks (`testgen::BlockProg`, every fold `: bf16 | f32`: arrays
//! stored bf16 by `round-store`, the attention output's epilogue ending in
//! the stored rounding by `round-post`, a bf16 cache written with values that
//! carry the default face's deviation): the emulated device's values against
//! the exact oracle within each printed value's composed bound, in every
//! emulator mode and under the `round-store` OFF arm, and the weakest-member
//! oracle (another admitted realization) within it too. Coverage non-zero:
//! programs with stored arrays and a stored epilogue rounding; device values
//! outside the one-site coefficient the probe gate used before
//! (`value_gate.py --bf16`, the five-GPU round's `px_llm_blkdec_b2_l16_bf16`)
//! — this test fails on that bound; and device values outside the f32 twin's
//! composed bound (the same program without `: bf16 | f32`: the roundings,
//! not admitted there, are caught).

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Bound, Outcome, run_bound, run_weakest};
use mapal_ir::{CategoryIr, NarrowFloat, bound_coef_unit};
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

/// The default face on the rtx-5080, the rungs `off` off; the emulator's
/// rungs off (`pdl`, `streams`, `pinned`), `place` off (every capable site on
/// the device), `horizontal` / `calls` off.
fn emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The emulator modes: free threads, each block's threads serialized between
/// barriers, a block at a time, blocks in reverse.
const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|x| x.parse().expect("a float"))
        .collect()
}

/// The longest fold of `ir` (`value_gate.py`'s `K` at its least).
fn longest_fold(ir: &CategoryIr) -> u64 {
    ir.morphisms()
        .filter_map(|(_, m)| match m.op {
            mapal_ir::Operation::Fold { captures, .. } => {
                match ir.object(m.source)?.ty.component_ty(captures + 1)? {
                    mapal_ir::Ty::Array { size, .. } => Some(*size),
                    _ => None,
                }
            }
            _ => None,
        })
        .max()
        .unwrap_or(1)
}

/// The oracle's exact values and each one's composed bound.
fn oracle(ir: &CategoryIr, src: &str) -> (Vec<f64>, Vec<Bound>) {
    let (rr, bounds) = run_bound(ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = floats(&rr.output);
    assert_eq!(want.len(), bounds.len(), "one bound a printed value\n{src}");
    (want, bounds)
}

/// One program's checks; coverage `[programs, stored, round-post, outside
/// one-site, outside twin, runs]`.
fn check(clang: &str, p: &testgen::BlockProg) -> [u64; 6] {
    let src = testgen::block_src(p, true);
    let ir = lower(&src);
    let (want, bounds) = oracle(&ir, &src);
    // The f32 twin: the same exact face, no permission to round.
    let twin = testgen::block_src(p, false);
    let (want2, twin_bounds) = oracle(&lower(&twin), &twin);
    assert_eq!(want, want2, "the twin's exact face\n{src}");
    let mut cover = [1, 0, 0, 0, 0, 0];
    if let Some(g) = ir.narrow_stored() {
        cover[1] = 1;
        let rounded = g.funcs().any(|(f, _)| {
            let post = g.tile_plan_post(f).sites;
            let radix = g.tile_plan_radix(f);
            let narrow = g.tile_plan_narrow(f).sites;
            post.into_iter().chain(radix).chain(narrow).any(|(_, s)| {
                s.post
                    .as_ref()
                    .is_some_and(|x| x.round == Some(NarrowFloat::Bf16))
            })
        });
        cover[2] = u64::from(rounded);
    }
    // value_gate.py `--bf16` before: relative `2·((1 + u_t)²·(1 + γ_{K+1}) − 1)`.
    let one_site = 2.0 * bound_coef_unit(longest_fold(&ir), Some(2f64.powi(-8)));
    let mut judge = |got: &[f64], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}\n{src}");
        for ((g, w), (b, t)) in got.iter().zip(&want).zip(bounds.iter().zip(&twin_bounds)) {
            let d = (g - w).abs();
            assert!(
                d <= b.total,
                "{what}: {g} vs {w}: |d| {d:e} > composed {:e} ({b:?})\n{src}",
                b.total
            );
            cover[3] |= u64::from(d > one_site * w.abs());
            cover[4] |= u64::from(d > t.total);
        }
    };
    judge(
        &floats(&run_weakest(&ir, 4_000_000_000, &[]).output),
        "weakest",
    );
    let mut runs = 0;
    for off in [&[][..], &["round-store"][..]] {
        let em = emission(&ir, off);
        let built = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        let modes: &[[(&str, &str); 2]] = if off.is_empty() { &MODES } else { &MODES[..1] };
        for (mode, (got, _)) in modes.iter().zip(built.run_each(modes)) {
            judge(&floats(&got), &format!("{off:?} {mode:?}"));
            runs += 1;
        }
    }
    cover[5] = runs;
    cover
}

/// CLAUDE.md "format generated source too": both forms of every generated
/// decode block are the formatter's layout, as `mapal fmt --check` wants.
#[test]
fn block_sources_are_formatted() {
    let mut runner = TestRunner::deterministic();
    for i in 0..64 {
        let p = testgen::block_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        for member in [true, false] {
            let src = testgen::block_src(&p, member);
            testgen::assert_formatted(&src, &format!("block #{i} member={member}"));
        }
    }
}

#[test]
fn composed_bound_on_generated_decode_blocks() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::BlockProg> = std::iter::repeat_with(|| {
        testgen::block_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(32)
    .collect();
    let covers = testgen::par_check(&progs, |_, p| check(&clang, p));
    let mut c = [0u64; 6];
    for k in covers {
        for (a, b) in c.iter_mut().zip(k) {
            *a += b;
        }
    }
    eprintln!(
        "composed bound: programs, stored, round-post, outside one-site, outside twin, runs: {c:?}"
    );
    assert!(c.iter().all(|&x| x > 0), "{c:?}");
}
