//! flash-attention: the facts one pass over windows reads — `stream_plan`
//! (F4) over the row-block groups of every tile plan (`unit_plan` F2, digit
//! rows F1), `fold_combine` over the epilogue and mixed-radix sites (F3), a
//! guarded producer's mask (F5, its absorb F6) and `finite_plan` (F7) —
//! each against the program it is read off, by brute force where it is a
//! claim about values.

mod testgen;

use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, DepLaw, StreamRole};

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

fn bench(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../../benches/algorithms/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn named(ir: &CategoryIr, n: &str) -> mapal_ir::ObjectId {
    ir.objects()
        .find(|(_, o)| o.name.as_deref() == Some(n))
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no {n}"))
}

/// The model-scale prefill (`px_llm_pre_t12`, the 2048-token program's
/// twin): one extended group — the masked scores produced over the keys, the
/// row max, `exp` shifted by it, the sum, and `P·V` written token-major (its
/// rows a digit permutation of the scores': F1) with `÷ den` in its
/// epilogue; its laws the rescale on the max (F3 over the mixed-radix
/// epilogue site); a KV head's query heads one panel batch; the causal mask
/// a guard over (row, key) digits whose wholly masked windows are skipped
/// (F5), the value cache proven finite (F7).
#[test]
fn model_scale_prefill_is_one_stream_group() {
    let ir = lower(&bench("px_llm_pre_t12"));
    let f = ir.entry();
    let sp = ir.stream_plan(f);
    assert_eq!(sp.groups.len(), 1, "{sp:?}");
    let g = &sp.groups[0];
    let unit = &ir.unit_plan(f).groups[g.group];
    assert!(unit.extended);
    let name = |k: usize| {
        ir.object(unit.members[k].out)
            .and_then(|o| o.name.clone())
            .unwrap()
    };
    let names: Vec<String> = (0..unit.members.len()).map(name).collect();
    assert_eq!(names, ["s", "mx", "pe", "den", "attf"]);
    assert_eq!(g.axis, 12);
    assert_eq!(g.roles[0], StreamRole::Produce);
    assert!(
        matches!(&g.roles[2], StreamRole::Cell { subst } if subst[..] == [(1, 1, DepLaw::Rescale)])
    );
    match &g.roles[4] {
        StreamRole::Consume { laws } => {
            assert_eq!(laws.len(), 1);
            assert_eq!((laws[0].value, laws[0].law), (1, DepLaw::Rescale));
        }
        r => panic!("{r:?}"),
    }
    // F1: P·V's rows `(token, head)` read the scores' rows `head·T + token`.
    let rm = unit.members[4]
        .row_of
        .as_ref()
        .expect("a digit permutation");
    let (t, h) = (12u64, unit.rows / 12);
    for r in 0..unit.rows {
        assert_eq!(rm.member_row(rm.group_row(r)), r);
    }
    // F2: the KV head's query heads share one value panel: its rows (two
    // heads of 12 tokens) the most a power of two divides, 8.
    assert_eq!(unit.panel, 8);
    let _ = (t, h);
    // F3: the epilogue site's fold laws.
    let fc = ir.fold_combine(f);
    let attf = fc
        .sites
        .get(ir.in_edges(named(&ir, "attf"))[0])
        .expect("F3: a combine");
    assert_eq!(
        attf.deps.iter().map(|d| d.law).collect::<Vec<_>>(),
        [DepLaw::Rescale]
    );
    // F5/F6/F7: the causal mask, skipped; `qkv` finite.
    let gd = g.guard.as_ref().expect("a guarded producer");
    assert!(gd.skip, "{}", gd.why);
    assert!(gd.form.masked < -1e8);
    assert!(ir.finite_plan(f).finite(named(&ir, "qkv")));
    // The fact arms off.
    for arms in [
        mapal_ir::FactArms {
            guard_box: false,
            ..ir.fact_arms()
        },
        mapal_ir::FactArms {
            finite: false,
            ..ir.fact_arms()
        },
        mapal_ir::FactArms {
            stream: false,
            ..ir.fact_arms()
        },
    ] {
        let off = ir.refacted(arms);
        let sp = off.stream_plan(off.entry());
        match (arms.stream, arms.guard_box, arms.finite) {
            (false, _, _) => assert!(sp.groups.is_empty()),
            (_, false, _) => assert!(sp.groups[0].guard.is_none()),
            _ => {
                let gd = sp.groups[0].guard.as_ref().unwrap();
                assert!(!gd.skip && gd.why == "finite");
            }
        }
    }
}

/// F7 on hand-written chains: a sum of squares then a Heron root (a
/// norm's `1 / sqrt`) is finite and its bound holds every value the
/// interpreter computes; `exp` of an unbounded score, a quotient by a value
/// that may be zero, and a loop's state are not proven.
#[test]
fn finite_ranges_hold_and_refuse() {
    let src = "fn main() {
    64 -> iota -> ix;
    ix -> map {
        t -> (t * 7 + 3) % 13 - 6 -> widen_f32 -> v;
        v / 4.0
    } -> x;
    8 -> iota -> rr;
    8 -> iota -> rk;
    16 -> iota -> r16;
    rr -> map {
        b -> (0.0, rk) -> fold {
            acc, i -> x[b * 8 + i] -> v;
            acc + v * v
        } -> ss;
        ss / 8.0 + 0.00001 -> ra;
        0.5 + 0.5 * ra -> s0;
        (s0, r16) -> fold {
            acc, k -> 0.5 * (acc + ra / acc)
        } -> sd;
        1.0 / sd
    } -> rinv;
    ix -> map {
        t -> x[t] / x[(t + 1) % 64]
    } -> q;
    rr -> map {
        b -> rinv[b] * 100.0 -> exp
    } -> e;
    rinv[0] -> println;
    rinv[1] -> println;
    rinv[2] -> println;
    rinv[3] -> println;
    rinv[4] -> println;
    rinv[5] -> println;
    rinv[6] -> println;
    rinv[7] -> println;
    q[3] -> println;
    e[0] -> println;
}
";
    let ir = lower(src);
    let f = ir.entry();
    let fp = ir.finite_plan(f);
    let (lo, hi) = *fp
        .ranges
        .get(named(&ir, "rinv"))
        .expect("the norm's scale is finite");
    assert!(fp.finite(named(&ir, "x")));
    assert!(
        !fp.finite(named(&ir, "q")),
        "a quotient by a value that may be zero"
    );
    let rr = run(&ir, 100_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("interp")
    };
    for v in rr.output.lines().take(8) {
        let v: f64 = v.parse().unwrap();
        assert!(lo <= v && v <= hi, "{v} outside [{lo}, {hi}]");
    }
    // `exp` of `100 / sd` is not bounded below `f32`'s largest exponent.
    assert!(!fp.finite(named(&ir, "e")));
}

/// mask-frontend: the run's values of a generated program's open guard
/// sides (`testgen::flash_tm_src`: `first` is `0`, `pos[i] = i`).
pub struct RunEnv;

impl mapal_ir::GuardEnv for RunEnv {
    fn number(&self, _: u32) -> Option<u64> {
        Some(0)
    }
    fn element(&self, _: u32, index: u64) -> Option<i64> {
        Some(index as i64)
    }
}

/// F1–F7 over generated attention programs (`testgen::flash_strategy`, the
/// model-scale form): every proven-finite array's printed values inside its
/// range; every digit permutation a bijection whose inverse inverts it; the
/// mask's classes against every cell. mask-frontend: the token-major forms
/// (`tm`) — the mask through a positions array whose law is `first + x`
/// with `first` a literal is the closed `r ≤ i`, with a run-time `first` a
/// per-row load, `first + i` a per-launch number; the scores' tile order a
/// walk order of the group whose blocks share a KV head's panel.
#[test]
fn stream_facts_on_generated_attention() {
    use proptest::strategy::{Strategy, ValueTree};
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for _ in 0..96 {
        let p = testgen::flash_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        if p.heads == 0 || p.heads * p.n * p.n > 16384 {
            continue;
        }
        let src = testgen::flash_src(&p);
        let ir = lower(&src);
        let f = ir.entry();
        *c.entry("programs").or_default() += 1;
        let fp = ir.finite_plan(f);
        for n in ["q", "kr", "vv"] {
            assert!(fp.finite(named(&ir, n)), "{n}\n{src}");
        }
        let sp = ir.stream_plan(f);
        for g in &sp.groups {
            let unit = &ir.unit_plan(f).groups[g.group];
            for m in unit.members.iter().filter_map(|m| m.row_of.as_ref()) {
                let mut seen = vec![false; unit.rows as usize];
                for r in 0..unit.rows {
                    let gr = m.group_row(r);
                    assert!(!seen[gr as usize]);
                    seen[gr as usize] = true;
                    assert_eq!(m.member_row(gr), r);
                }
                *c.entry("permutations").or_default() += 1;
            }
            if let Some(o) = &unit.order {
                // A bijection of the rows whose blocks of `panel` positions
                // share every tile member's column panel.
                let mut seen = vec![false; unit.rows as usize];
                for q in 0..unit.rows {
                    let gr = o.group_row(q);
                    assert!(!seen[gr as usize], "{src}");
                    seen[gr as usize] = true;
                }
                assert!(unit.panel > 1, "{src}");
                *c.entry("walk orders").or_default() += 1;
            }
            if let Some(gd) = &g.guard {
                let env = RunEnv;
                for r in 0..unit.rows {
                    for l in 0..g.axis {
                        let v = gd.form.visible_with(r, l, &env).expect("the run's values");
                        // The run's mask: `r ≤ i` at the cell's token.
                        let token = r / p.heads.max(1);
                        if p.tm > 0 {
                            assert_eq!(v, l <= token, "{src}");
                        }
                        let cls = gd.form.classify_with((r, r), (l, l), &env);
                        assert_eq!(cls == mapal_ir::BoxClass::Visible, v);
                        assert_eq!(cls == mapal_ir::BoxClass::Masked, !v);
                        if gd.form.closed() {
                            assert_eq!(gd.form.visible(r, l), v);
                        } else {
                            assert_eq!(gd.form.classify((r, r), (l, l)), mapal_ir::BoxClass::Mixed);
                        }
                    }
                }
                *c.entry("masks").or_default() += 1;
                let side = &gd.form.rhs;
                match p.tm {
                    1 | 2 => {
                        assert!(gd.form.closed(), "{:?}\n{src}", gd.form);
                        *c.entry(["", "tm digit masks", "tm law masks"][p.tm as usize])
                            .or_default() += 1;
                    }
                    3 => {
                        assert!(side.load.is_some(), "{:?}\n{src}", gd.form);
                        *c.entry("tm load masks").or_default() += 1;
                    }
                    4 => {
                        assert!(!side.uniform.is_empty(), "{:?}\n{src}", gd.form);
                        *c.entry("tm uniform masks").or_default() += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    eprintln!("stream facts generated: {c:?}");
    for k in [
        "programs",
        "permutations",
        "masks",
        "walk orders",
        "tm digit masks",
        "tm law masks",
        "tm load masks",
        "tm uniform masks",
    ] {
        assert!(c.get(k).copied().unwrap_or(0) > 0, "{k}: {c:?}");
    }
}
