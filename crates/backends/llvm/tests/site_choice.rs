//! per-site-choice B4: every site's candidate table on programs from the
//! generator (`testgen` geom / order / mid / tile / chain strategies, which
//! the candidates were not designed on), on the four GPU parts:
//!
//! - every candidate's values answer the interpreter — within ADR-0040's
//!   bound on the default face, bit for bit on the exact face — through the
//!   GPU emulator, one run per candidate index (`MAPAL_GPU_PICK`);
//! - running every candidate in turn (`MAPAL_GPU_PICK=all`) leaves the last
//!   one's bits: a site flagged `RERUN` really is;
//! - candidate 0 is the rules' plan: `--cands=1` emits no table, and each
//!   table's first recipe and kernel are the `--cands=1` build's launch and
//!   kernel text;
//! - the decision (`mapal_choice.h`, shared with the device runtime) under a
//!   fake clock: the faster candidate wins and is written, the next run reads
//!   it, a candidate within 3 % keeps candidate 0, a corrupt wisdom file is
//!   ignored, a changed table re-tunes;
//! - the candidate set: cap, distinct launches, one `Combine` on the fixed
//!   and exact faces, no table for the excluded site classes.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, Order, emit_detailed};
use mapal_ir::{CategoryIr, bound_coef};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const PARTS: [&str; 4] = ["rtx-4090", "rtx-5090", "rtx-3090", "h100-pcie"];

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// `base` on `target` with at most `cands` candidates per table, `on` / `off`
/// rung arms, `place` off (every capable site on the device) and the rungs
/// the emulator's one thread does not run (`pdl`, `streams`, `pinned`) off.
fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    cands: u64,
    on: &[&str],
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms {
        site_cands: cands,
        ..GpuArms::default()
    };
    for r in off.iter().chain(&["place", "pdl", "streams", "pinned"]) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
        *gpu.off.rung(r).expect("rung") = false;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// One `mapal_gpu_site` call of a host text: candidate 0's id, the flags and
/// each candidate's recipe (`id rid zero ws gx gy block smem rgx rblock`).
#[derive(Debug)]
struct Table {
    site: u64,
    flags: u32,
    rows: Vec<[i64; 10]>,
}

fn tables(host: &str) -> Vec<Table> {
    host.lines()
        .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_site("))
        .map(|args| {
            let a: Vec<&str> = args.trim_end_matches(')').split(", ").collect();
            let v = |i: usize| a[i].split(' ').nth(1).expect("typed argument");
            let def = format!("{} = ", v(4));
            let global = host
                .lines()
                .find(|d| d.starts_with(&def))
                .expect("the table constant");
            let rows: Vec<[i64; 10]> = global
                .split("%mapal_cand { ")
                .skip(1)
                .map(|r| {
                    let fields: Vec<i64> = r
                        .split(" }")
                        .next()
                        .expect("a row")
                        .split(", ")
                        .map(|f| f.split(' ').nth(1).expect("typed").parse().expect("int"))
                        .collect();
                    fields.try_into().expect("ten fields")
                })
                .collect();
            assert_eq!(rows.len(), v(3).parse::<usize>().unwrap(), "{global}");
            Table {
                site: v(0).parse::<i64>().unwrap() as u64,
                flags: v(1).parse().unwrap(),
                rows,
            }
        })
        .collect()
}

/// The `; mapal.plan` line of kernel `id` in a device unit.
fn plan(dev: &str, id: i64) -> Option<&str> {
    let name = format!("kernel=mapal_tile_{:016x} ", id as u64);
    dev.lines()
        .find(|l| l.starts_with("; mapal.plan") && l.contains(&name))
}

/// Kernel `id`'s text (its plan line through its closing brace), its loop
/// metadata unnumbered (the module numbers it by kernel position).
fn kernel_text(dev: &str, id: i64) -> Option<String> {
    let name = format!("@mapal_tile_{:016x}(", id as u64);
    let at = dev.find(&format!(
        "; mapal.plan kernel=mapal_tile_{:016x} ",
        id as u64
    ))?;
    let body = &dev[at..];
    assert!(body.contains(&name));
    let end = body.find("\n}\n")? + 3;
    let mut out = String::new();
    let mut digits = false;
    for c in body[..end].chars() {
        if digits && c.is_ascii_digit() {
            continue;
        }
        digits = c == '!';
        out.push(c);
    }
    Some(out)
}

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split(' ')
        .find_map(|f| f.strip_prefix(key))
        .unwrap_or("")
}

/// The printed floats of a run, without the clock (`iter ms=`).
fn floats(out: &str) -> Vec<f64> {
    out.lines()
        .filter(|l| !l.starts_with("iter ms="))
        .map(|l| l.trim().parse().unwrap_or(f64::NAN))
        .collect()
}

/// A run's output without the clock's line (`iter ms=<value>`).
fn untimed(out: &str) -> String {
    out.lines()
        .filter(|l| !l.starts_with("iter ms="))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// A program to check and how: `exact` bit for bit, else each printed value
/// within `tol[i]` (`None`: relative 1e-4, the chains' and the blind set's
/// check in `plan_cost.rs`).
struct Case {
    src: String,
    base: EmitOpts,
    tol: Option<Vec<f64>>,
}

/// ADR-0040 D2's a-priori bound of a geometry program's three printed values
/// (`y[0]`, `y[last]`, the whole sum): `γ_k · M` per cell with `M ≤ k·|a|·|b|`
/// at the operand law's largest magnitudes, and the sum's own `γ_{mn}`.
fn geom_tol((m, n, k): (u64, u64, u64), frac: bool) -> Vec<f64> {
    let ab = if frac {
        (50.0 / 7.0) * (51.0 / 9.0)
    } else {
        50.0 * 51.0
    };
    let mag = k as f64 * ab;
    // `+ 4`: an epilogue's bias (|bias| <= 3) and relu add one rounding.
    let cell = bound_coef(k, false) * mag + (mag + 4.0) * 2f64.powi(-23);
    let cells = (m * n) as f64;
    let total = cells * cell + bound_coef(m * n, false) * cells * (mag + 4.0);
    vec![cell, cell, total]
}

fn check(case: &Case, got: &str, want: &str) -> bool {
    if case.tol.is_none() && case.base.order == Order::Sequential {
        return got == want;
    }
    let (g, w) = (floats(got), floats(want));
    g.len() == w.len()
        && g.iter().zip(&w).enumerate().all(|(i, (a, b))| {
            let tol = match &case.tol {
                Some(t) => t[i.min(t.len() - 1)],
                None => 1e-4 * a.abs().max(b.abs()).max(1.0),
            };
            (a - b).abs() <= tol
        })
}

/// The generator's programs, rotating over the strategies and faces: geom
/// (integer operands, the default face), order (non-integer, `f64`, `a` along
/// `i`, `k = 8·odd`: the exact face bit for bit, and the default face within
/// the bound), mid (tile-efficiency extents), tile (the blind set's shapes at
/// interpreter sizes) and chains.
fn cases(count: usize) -> Vec<Case> {
    let mut runner = TestRunner::deterministic();
    let (geom, order) = (
        testgen::geom_prog_strategy(),
        testgen::order_prog_strategy(),
    );
    let (chain, tile) = (testgen::chain_strategy(), testgen::tile_prog_strategy());
    let def = EmitOpts::default;
    (0..count)
        .map(|z| match z % 6 {
            0 | 1 => {
                let (src, dims) = testgen::geom_src(&geom.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, false)),
                }
            }
            2 => {
                let (src, _) = testgen::geom_src(&order.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: EmitOpts::exact(),
                    tol: None,
                }
            }
            3 => {
                let (src, dims) =
                    testgen::geom_src(&order.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, true)),
                }
            }
            4 => {
                // The emulator and the interpreter run every cell: mid
                // extents up to 2^25 products.
                let (src, dims) = loop {
                    let (src, (m, n, k)) =
                        testgen::mid_src(&geom.new_tree(&mut runner).unwrap().current());
                    if m * n * k <= 1 << 25 {
                        break (src, (m, n, k));
                    }
                };
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, false)),
                }
            }
            _ if z % 12 == 5 => Case {
                src: testgen::tile_src(&tile.new_tree(&mut runner).unwrap().current(), true).0,
                base: def(),
                tol: None,
            },
            _ => Case {
                src: chain.new_tree(&mut runner).unwrap().current().source(),
                base: def(),
                tol: None,
            },
        })
        .collect()
}

/// B4 values and re-run safety: for every program with a table, the emulated
/// output of every candidate index below the program's largest table answers
/// the interpreter; `MAPAL_GPU_PICK=all` equals the last index bit for bit
/// where every table is flagged `RERUN`. per-site-choice D4: the part does not
/// follow the strategy — every strategy meets every part — and each part
/// reaches tables, exact-face runs and `f64` runs.
#[test]
fn every_candidate_answers_the_interpreter() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice values: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let (mut programs, mut tabled, mut runs, mut reruns, mut sites) = (0, 0, 0, 0, 0);
    let mut exact = 0;
    let mut by_n = [0usize; 9];
    // Per part: [tabled programs, exact-face runs, f64 runs].
    let mut per = [[0usize; 3]; 4];
    for (z, case) in cases(48).iter().enumerate() {
        let ir = lower(&case.src);
        // Each block of 12 runs every strategy, rotated over the parts by
        // block (a Latin square: every strategy meets every part); the
        // rotation is the one under which the deterministic draws give every
        // part tables, exact-face and f64 runs.
        let part = (z % 12 + [3, 2, 0, 1][z / 12 % 4]) % PARTS.len();
        let target = PARTS[part];
        let e = emission(&ir, target, case.base, 8, &[], &[]);
        programs += 1;
        let ts = tables(&e.text);
        let Some(max) = ts.iter().map(|t| t.rows.len()).max() else {
            continue;
        };
        tabled += 1;
        per[part][0] += 1;
        sites += ts.len();
        for t in &ts {
            by_n[t.rows.len().min(8)] += 1;
        }
        let rr = mapal_interp::run(&ir, 4_000_000_000);
        assert!(
            matches!(rr.outcome, mapal_interp::Outcome::Done(_)),
            "#{z}: {:?}\n{}",
            rr.outcome,
            case.src
        );
        let dev = e.device.as_deref().expect("a device unit");
        let built = emu::build_emulated(&clang, &e.text, dev);
        let picks: Vec<String> = (0..max)
            .map(|i| i.to_string())
            .chain(["all".to_string()])
            .collect();
        let outs: Vec<String> = std::thread::scope(|s| {
            let b = &built;
            let hs: Vec<_> = picks
                .iter()
                .map(|p| s.spawn(move || b.run(&[("MAPAL_GPU_PICK", p.as_str())]).0))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, got) in outs[..max].iter().enumerate() {
            assert!(
                check(case, got, &rr.output),
                "#{z} {target} candidate {i}: {got:?} != {:?}\n{}",
                rr.output,
                case.src
            );
            runs += 1;
            exact += usize::from(case.base.order == Order::Sequential);
            per[part][1] += usize::from(case.base.order == Order::Sequential);
            per[part][2] += usize::from(case.src.contains("f64"));
        }
        if ts.iter().all(|t| t.flags & 1 == 1) {
            assert_eq!(
                untimed(&outs[max]),
                untimed(&outs[max - 1]),
                "#{z} {target}: every candidate in turn vs the last\n{}",
                case.src
            );
            reruns += 1;
        }
    }
    eprintln!(
        "site-choice values: {programs} programs, {tabled} with tables ({sites} sites, n histogram \
         {by_n:?}), {runs} candidate runs ({exact} on the exact face), {reruns} re-run checks; \
         per part [tabled, exact runs, f64 runs] {per:?}"
    );
    assert!(
        tabled >= 20 && runs >= 64 && exact >= 10 && reruns >= 16,
        "the generator reaches site tables"
    );
    for (p, [t, x, f]) in PARTS.iter().zip(per) {
        assert!(
            t >= 6 && x >= 2 && f >= 2,
            "{p}: {t} tabled, {x} exact runs, {f} f64 runs"
        );
    }
}

/// per-site-choice D1: candidate 0 is the pre-sprint emission — `--cands=1`
/// at HEAD hashes (host + device IR less the `; mapal.` evidence lines) as
/// 089f1a6 emitted each program of a fixed bench list on the four parts and
/// three faces (`tests/golden/c0-089f1a6.tsv`): `bank-plan` OFF with the
/// `kmajor-a` rows is the pre-sprint layout, byte for byte.
#[test]
fn candidate_zero_is_the_pre_sprint_emission() {
    let golden = include_str!("golden/c0-089f1a6.tsv");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../benches/algorithms");
    let norm = |t: &str| -> String {
        t.split_inclusive('\n')
            .filter(|l| !l.starts_with("; mapal."))
            .collect()
    };
    let fnv = |b: &[u8]| {
        b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
            (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    let mut rows = 0;
    let mut bad = Vec::new();
    for l in golden.lines().filter(|l| !l.starts_with('#')) {
        let [prog, part, face, want]: [&str; 4] = l
            .split('\t')
            .collect::<Vec<_>>()
            .try_into()
            .expect("program part face hash");
        let src = std::fs::read_to_string(format!("{root}/{prog}.mapal")).expect("a bench program");
        let mut opts = EmitOpts::default();
        let face_flags: &[&str] = match face {
            "contract" => &["--order=sequential"],
            "exact" => &["--exact"],
            _ => &[],
        };
        let target = format!("--target={part}");
        for f in [target.as_str(), "--host=x86-64-v3", "--cands=1"]
            .iter()
            .chain(face_flags)
        {
            assert!(mapal_backend_llvm::flags::apply(&mut opts, f).expect("a flag"));
        }
        let e = emit_detailed(&lower(&src), &opts).expect("emits");
        let body = format!(
            "{}\0{}",
            norm(&e.text),
            norm(e.device.as_deref().unwrap_or(""))
        );
        let got = format!("{:016x}", fnv(body.as_bytes()));
        if got != want {
            bad.push(format!("{prog} {part} {face}: {got} != {want}"));
        }
        rows += 1;
    }
    assert!(rows >= 200, "{rows} golden rows");
    assert!(bad.is_empty(), "differs from 089f1a6:\n{}", bad.join("\n"));
}

/// B4 c0 identity: `--cands=1` emits no table and no site line; where the
/// default emits none either, the two emissions are equal byte for byte, and
/// every table's candidate 0 is the `--cands=1` build's launch (kernel,
/// grid, block, shared memory, zero, workspace, reduce) and kernel text.
#[test]
fn candidate_zero_is_the_rules_plan() {
    let (mut same, mut tables_checked, mut kernels) = (0, 0, 0);
    for (z, case) in cases(96).iter().enumerate() {
        let ir = lower(&case.src);
        for target in PARTS {
            let one = emission(&ir, target, case.base, 1, &[], &[]);
            assert!(
                !one.text.contains("@mapal_gpu_site(")
                    && !one.device.as_deref().unwrap_or("").contains("; mapal.site"),
                "#{z} {target}: --cands=1 emits a table"
            );
            let all = emission(&ir, target, case.base, 8, &[], &[]);
            let ts = tables(&all.text);
            if ts.is_empty() {
                assert_eq!(all.text, one.text, "#{z} {target} host");
                assert_eq!(all.device, one.device, "#{z} {target} device");
                same += 1;
                continue;
            }
            let (dev8, dev1) = (
                all.device.as_deref().expect("device"),
                one.device.as_deref().expect("device"),
            );
            for t in &ts {
                let c0 = t.rows[0];
                assert_eq!(c0[0] as u64, t.site, "#{z} {target}: site = candidate 0");
                // The `--cands=1` launch of that kernel: `launch(id, a, b, o,
                // gx, gy, block, smem)` or `launchv(id, s, n, p, gx, gy, …)`.
                let call = one
                    .text
                    .lines()
                    .find(|l| {
                        l.contains(&format!("@mapal_gpu_launch(i64 {}, ", c0[0]))
                            || l.contains(&format!("@mapal_gpu_launchv(i64 {}, ", c0[0]))
                    })
                    .unwrap_or_else(|| panic!("#{z} {target}: no launch of {}", c0[0]));
                let args: Vec<i64> = call
                    .split('(')
                    .nth(1)
                    .unwrap()
                    .split(')')
                    .next()
                    .unwrap()
                    .split(", ")
                    .filter_map(|a| a.strip_prefix("i32 ").and_then(|v| v.parse().ok()))
                    .collect();
                let grid = &args[args.len() - 4..];
                assert_eq!(grid, &c0[4..8], "#{z} {target}: grid\n{call}");
                // An atomic / stream-K recipe zeroes the output as the
                // rules' launch is preceded by its zero.
                let zero = one.text.lines().any(|l| {
                    l.contains("@mapal_gpu_zero(") && l.ends_with(&format!(", i64 {})", c0[2]))
                });
                assert!(c0[2] == 0 || zero, "#{z} {target}: zero {}", c0[2]);
                if c0[1] != 0 {
                    // The workspace split: its reduce launch, its workspace.
                    assert!(
                        one.text
                            .contains(&format!("@mapal_gpu_launch(i64 {}, ", c0[1])),
                        "#{z} {target}: reduce"
                    );
                    assert!(
                        one.text.lines().any(|l| l.contains("@mapal_gpu_alloc(")
                            && l.ends_with(&format!(", i64 {})", c0[3]))),
                        "#{z} {target}: workspace bytes"
                    );
                }
                for id in [c0[0], c0[1]].into_iter().filter(|&i| i != 0) {
                    assert_eq!(
                        kernel_text(dev8, id),
                        kernel_text(dev1, id),
                        "#{z} {target}: kernel {id:x} text"
                    );
                    kernels += 1;
                }
                tables_checked += 1;
            }
        }
    }
    eprintln!(
        "site-choice c0: {same} emissions without a table equal, {tables_checked} tables, \
         {kernels} candidate-0 kernels equal"
    );
    assert!(tables_checked >= 32 && same >= 16);
}

/// B4 candidate set: at most `--cands` candidates, the table's length on its
/// `; mapal.site` line, distinct recipes and kernels; on the fixed and exact
/// faces every candidate folds as candidate 0 does; each candidate kernel
/// carries its plan line.
#[test]
fn candidate_sets_cap_dedupe_and_keep_the_order() {
    let (mut capped, mut ordered, mut lines) = (0, 0, 0);
    let (mut flips, mut conflict_free, mut ruled_bank) = ([0usize; 2], 0, 0);
    let fixed = EmitOpts {
        contract: false,
        order: Order::Fixed,
        ..EmitOpts::default()
    };
    for (z, case) in cases(60).iter().enumerate() {
        let ir = lower(&case.src);
        let target = PARTS[z % PARTS.len()];
        for (base, cap) in [
            (case.base, 8),
            (case.base, 3),
            (case.base, 2),
            (fixed, 8),
            (EmitOpts::exact(), 8),
        ] {
            let e = emission(&ir, target, base, cap, &[], &[]);
            let dev = e.device.as_deref().unwrap_or("");
            for t in tables(&e.text) {
                let n = t.rows.len();
                assert!(
                    (2..=cap as usize).contains(&n),
                    "#{z} {target} cap {cap}: n={n}"
                );
                capped += usize::from(n == cap as usize);
                let mut ids: Vec<i64> = t.rows.iter().map(|r| r[0]).collect();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), n, "#{z} {target}: distinct kernels");
                let site = dev
                    .lines()
                    .find(|l| l.starts_with(&format!("; mapal.site site={:016x} ", t.site)))
                    .unwrap_or_else(|| panic!("#{z} {target}: no site line"));
                assert_eq!(field(site, "n="), n.to_string(), "#{z}: {site}");
                lines += 1;
                // per-site-choice A4 x B1: the `kmajor-a` axis flips what the
                // register rule decided — the forced k-major panel beside a
                // declined (transposed) candidate 0, the transposed one beside
                // a taken one.
                let kmaj = |id: i64| plan(dev, id).is_some_and(|p| p.contains(" kmajor=1"));
                for (r, ax) in t.rows.iter().zip(field(site, "axes=").split(',')) {
                    // D1 / D5.3: a k-major panel the register rule takes under
                    // `bank-plan` (not forced) is conflict-free.
                    if ax == "+bank-plan" && kmaj(r[0]) {
                        let p = plan(dev, r[0]).unwrap_or("");
                        assert!(p.contains(" adeg=1 "), "#{z} {target}: {p}");
                        ruled_bank += 1;
                    }
                    let on = match ax {
                        "+kmajor-a" => true,
                        "-kmajor-a" => false,
                        _ => continue,
                    };
                    let c0 = kmaj(t.rows[0][0]);
                    assert_eq!((c0, kmaj(r[0])), (!on, on), "#{z} {target}: {site}");
                    flips[usize::from(on)] += 1;
                    let p = plan(dev, if on { r[0] } else { t.rows[0][0] }).unwrap_or("");
                    conflict_free += usize::from(p.contains(" adeg=1 "));
                }
                let combine = |id: i64| {
                    let p = plan(dev, id).unwrap_or_else(|| panic!("#{z}: no plan line {id:x}"));
                    (field(p, "combine=").to_string(), field(p, "S=").to_string())
                };
                let c0 = combine(t.rows[0][0]);
                if base.order != Order::Free {
                    for r in &t.rows {
                        assert_eq!(combine(r[0]), c0, "#{z} {target} {:?}", base.order);
                    }
                    ordered += 1;
                }
            }
        }
    }
    eprintln!(
        "site-choice sets: {lines} tables, {capped} at their cap, {ordered} fixed/exact; \
         kmajor-a flips -{} +{}, {conflict_free} k-major kernels adeg=1; {ruled_bank} `+bank-plan` \
         k-major kernels (all adeg=1)",
        flips[0], flips[1]
    );
    assert!(capped >= 8 && ordered >= 8 && lines >= 40);
    assert!(flips[0] + flips[1] > 0, "the kmajor-a axis reaches a table");
}

/// B4 excluded classes: no table without `resident` (the round trip), in a
/// function on chain streams, on a column-major store (`rowfold-layout`), on
/// a site table (`linear`) or on a launch with an L2 window.
#[test]
fn excluded_site_classes_keep_one_launch() {
    let (mut col, mut window, mut chained) = (0, 0, 0);
    for (z, case) in cases(48).iter().enumerate() {
        let ir = lower(&case.src);
        let target = PARTS[z % PARTS.len()];
        let round = emission(&ir, target, case.base, 8, &[], &["resident"]);
        assert!(!round.text.contains("@mapal_gpu_site("), "#{z} round trip");
        let e = emission(
            &ir,
            target,
            case.base,
            8,
            &["l2-window", "rowfold-layout"],
            &[],
        );
        let dev = e.device.as_deref().unwrap_or("");
        for t in tables(&e.text) {
            let p = plan(dev, t.rows[0][0]).expect("plan line");
            assert_eq!(field(p, "col="), "0", "#{z} {target}: a column-major table");
            assert!(!p.contains(" terms="), "#{z}: a linear table");
        }
        for l in e
            .text
            .lines()
            .filter(|l| l.contains("@mapal_gpu_launch_l2(i64 "))
        {
            let id = l
                .split("(i64 ")
                .nth(1)
                .and_then(|r| r.split(',').next())
                .unwrap();
            assert!(
                !tables(&e.text)
                    .iter()
                    .any(|t| t.site == id.parse::<i64>().unwrap() as u64),
                "#{z}: an L2-window launch in a table"
            );
            window += 1;
        }
        col += dev
            .lines()
            .filter(|l| l.starts_with("; mapal.plan") && l.contains(" col=1 "))
            .count();
        // Chain streams on (the emulator is not involved: emission only).
        let mut gpu = GpuArms::default();
        for r in ["streams", "stream-gate"] {
            *gpu.on.rung(r).expect("rung") = true;
        }
        *gpu.off.rung("place").expect("rung") = true;
        let opts = EmitOpts {
            host: "generic",
            target,
            gpu,
            ..case.base
        };
        let s = emit_detailed(&ir, &opts).expect("emits");
        for f in s.text.split("\ndefine ").skip(1) {
            let on_streams = f.contains("@mapal_gpu_event(")
                || f.split("@mapal_gpu_launchv(i64 ")
                    .skip(1)
                    .any(|t| t.split(", ").nth(1).is_some_and(|x| x != "i32 0"));
            if on_streams {
                assert!(!f.contains("@mapal_gpu_site("), "#{z}: a table on streams");
                chained += 1;
            }
        }
    }
    // A row fold over a site's output (`rowfold.rs`'s row sum): its
    // column-major store keeps one launch.
    let n = 64;
    let rowsum = format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 }} -> b;
    ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }} }} -> c;
    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> c[i * {n} + j] -> e; acc + e }} }} -> r;
    r[0] -> println;
    r[{last}] -> println;
}}
",
        nn = n * n,
        last = n - 1
    );
    let ir = lower(&rowsum);
    for target in PARTS {
        for base in [EmitOpts::default(), EmitOpts::exact()] {
            let e = emission(
                &ir,
                target,
                base,
                8,
                &["rowfold", "rowfold-layout"],
                &["fill-rows", "epilogue", "devmap"],
            );
            let dev = e.device.as_deref().unwrap_or("");
            let cols: Vec<&str> = dev
                .lines()
                .filter(|l| l.starts_with("; mapal.plan") && l.contains(" col=1 "))
                .collect();
            for t in tables(&e.text) {
                let p = plan(dev, t.rows[0][0]).expect("plan line");
                assert_eq!(field(p, "col="), "0", "{target}: a column-major table");
            }
            col += cols.len();
        }
    }
    eprintln!(
        "site-choice excluded: {col} column-major kernels, {window} L2 windows, {chained} chained functions"
    );
    assert!(window >= 1, "the generator reaches an L2 window");
    assert!(col >= 4, "the row sum stores column-major");
}

/// The first generated program (geom strategy, on `target`) whose one table
/// holds at least `n` candidates and whose every table is `RERUN`.
fn tabled_program(target: &'static str, n: usize) -> (String, Emission) {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::geom_prog_strategy();
    for _ in 0..200 {
        let (src, _) = testgen::geom_src(&strat.new_tree(&mut runner).unwrap().current());
        let ir = lower(&src);
        let e = emission(&ir, target, EmitOpts::default(), 8, &[], &[]);
        let ts = tables(&e.text);
        if ts.len() == 1 && ts[0].rows.len() >= n && ts[0].flags & 1 == 1 {
            return (src, e);
        }
    }
    panic!("no generated program with a {n}-candidate table on {target}");
}

/// `(src, pick)` of the trace's `op=site` lines.
fn decisions(err: &str) -> Vec<(String, u32)> {
    err.lines()
        .filter(|l| l.starts_with("mapal_gpu_trace op=site "))
        .map(|l| {
            (
                field(l, "src=").to_string(),
                field(l, "pick=").parse().unwrap(),
            )
        })
        .collect()
}

/// B4 decision under the fake clock (`MAPAL_EMU_NS`), through the shipped
/// decision code (`mapal_choice.h`).
#[test]
fn the_measured_pick_is_stored_and_reused() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice decision: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let (src, e) = tabled_program("rtx-4090", 3);
    let n = tables(&e.text)[0].rows.len();
    let built = emu::build_emulated(&clang, &e.text, e.device.as_deref().unwrap());
    let want = built.run(&[("MAPAL_GPU_PICK", "0")]).0;
    let dir = tempfile::tempdir().unwrap();
    let wdir = dir.path().join("w");
    let wis = wdir.to_str().unwrap();
    let run = |ns: &str, wis: &str, extra: &[(&str, &str)]| {
        let mut env = vec![
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", wis),
            ("MAPAL_GPU_TRACE", "1"),
        ];
        env.extend_from_slice(extra);
        let (out, err) = built.run(&env);
        (out, err.clone(), decisions(&err), emu::totals(&err)[2])
    };
    let file = || std::fs::read_to_string(wdir.join("wisdom-emu-sm0.tsv")).unwrap_or_default();
    // Candidate 1 is 20 % faster than candidate 0 (1 ms: five reps).
    let ns = "1000000,800000,1200000";
    let (out, err, d, trial_launches) = run(ns, wis, &[]);
    assert_eq!(d, vec![("trial".into(), 1)], "{err}\n{src}");
    assert_eq!(out, want, "the winner's output");
    let trials = err.lines().filter(|l| l.contains("op=trial ")).count();
    // per-site-choice D3: a non-zero winner is confirmed head to head (3
    // alternating pairs with candidate 0).
    assert_eq!(
        trials,
        5 * n + 6,
        "five rotated reps of every candidate, three pairs\n{err}"
    );
    let site = err
        .lines()
        .find(|l| l.starts_with("mapal_gpu_trace op=site "))
        .unwrap();
    assert!(
        field(site, "reps=") == "5"
            && !field(site, "ns0=").is_empty()
            && !field(site, "nsw=").is_empty(),
        "{site}"
    );
    let w = file();
    assert_eq!(w.lines().count(), 1, "{w}");
    let cols: Vec<&str> = w.trim_end().split('\t').collect();
    assert_eq!(
        (cols[2], cols[3], cols[4]),
        (n.to_string().as_str(), "1", "5"),
        "{w}"
    );
    assert_eq!(
        cols.len(),
        5 + 2 * n,
        "key site n pick reps ns.. names..\n{w}"
    );
    // The next run reads the stored winner and launches it alone.
    let (out, err, d, launches) = run(ns, wis, &[]);
    assert_eq!(d, vec![("wisdom".into(), 1)], "{err}");
    assert_eq!(out, want);
    let solo = built.run(&[("MAPAL_GPU_PICK", "1")]).1;
    assert_eq!(launches, emu::totals(&solo)[2], "only the winner launches");
    assert!(trial_launches > launches);
    assert_eq!(file().lines().count(), 1, "nothing appended");
    // A candidate within 3 % keeps candidate 0.
    let close = dir.path().join("close");
    let (_, err, d, _) = run("1000000,980000,990000", close.to_str().unwrap(), &[]);
    assert_eq!(d, vec![("trial".into(), 0)], "{err}");
    // A corrupt wisdom file (garbage, a pick past the table) is ignored.
    let bad = dir.path().join("bad");
    std::fs::create_dir_all(&bad).unwrap();
    let key = cols[0];
    std::fs::write(
        bad.join("wisdom-emu-sm0.tsv"),
        format!("not a line\n\u{1}\u{2}\n{key}\t{}\t{n}\t{n}\t5\n", cols[1]),
    )
    .unwrap();
    let (_, err, d, _) = run(ns, bad.to_str().unwrap(), &[]);
    assert_eq!(d, vec![("trial".into(), 1)], "{err}");
    // Forced and off: no trial, nothing written.
    let (_, _, d, _) = run(ns, close.to_str().unwrap(), &[("MAPAL_GPU_PICK", "rules")]);
    assert_eq!(d, vec![("forced".into(), 0)]);
    // D3: `auto` is the default path; `tune` ignores the stored entry, trials
    // again and appends; anything else is an error, never candidate 0.
    let (_, _, d, _) = run(ns, wis, &[("MAPAL_GPU_PICK", "auto")]);
    assert_eq!(d, vec![("wisdom".into(), 1)]);
    let (_, _, d, _) = run(ns, wis, &[("MAPAL_GPU_PICK", "tune")]);
    assert_eq!(d, vec![("trial".into(), 1)]);
    assert_eq!(file().lines().count(), 2, "the re-tune appended");
    let typo = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        built.run(&[("MAPAL_EMU_NS", ns), ("MAPAL_GPU_PICK", "rule")])
    }));
    assert!(typo.is_err(), "MAPAL_GPU_PICK=rule is an error");
    let (_, _, d, _) = run(ns, "off", &[]);
    assert_eq!(d, vec![("rules".into(), 0)]);
    // A changed table (two candidates: other bytes) re-tunes against the
    // same wisdom file.
    let ir = lower(&src);
    let e2 = emission(&ir, "rtx-4090", EmitOpts::default(), 2, &[], &[]);
    let built2 = emu::build_emulated(&clang, &e2.text, e2.device.as_deref().unwrap());
    let err = built2
        .run(&[
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", wis),
            ("MAPAL_GPU_TRACE", "1"),
        ])
        .1;
    assert_eq!(decisions(&err), vec![("trial".into(), 1)], "{err}");
    assert_eq!(
        file().lines().count(),
        3,
        "the new table's decision appended"
    );
    // D4: a table without `RERUN` (the flag an aliased output clears,
    // `apart`) is never trialled.
    let host = e.text.replacen(
        &format!(
            "@mapal_gpu_site(i64 {}, i32 1,",
            tables(&e.text)[0].site as i64
        ),
        &format!(
            "@mapal_gpu_site(i64 {}, i32 0,",
            tables(&e.text)[0].site as i64
        ),
        1,
    );
    assert_ne!(host, e.text);
    let built0 = emu::build_emulated(&clang, &host, e.device.as_deref().unwrap());
    let fresh = dir.path().join("norerun");
    let err = built0
        .run(&[
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", fresh.to_str().unwrap()),
            ("MAPAL_GPU_TRACE", "1"),
        ])
        .1;
    assert_eq!(decisions(&err), vec![("rules".into(), 0)], "{err}");
    assert!(!err.contains("op=trial "), "{err}");
    assert!(!fresh.join("wisdom-emu-sm0.tsv").exists());
}

/// per-site-choice D4: on a program whose candidates differ in bits
/// (non-integer operands), the output after a trial is the winner's and the
/// site's last launch is the winner's kernel — whether the winner ran last in
/// the trial or not.
#[test]
fn the_winner_runs_last() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice winner: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let strat = testgen::order_prog_strategy();
    for z in 0..60 {
        let (src, _) = testgen::geom_src(&strat.new_tree(&mut runner).unwrap().current());
        let target = PARTS[z % PARTS.len()];
        let e = emission(&lower(&src), target, EmitOpts::default(), 2, &[], &[]);
        let ts = tables(&e.text);
        if ts.len() != 1 || ts[0].flags & 1 == 0 {
            continue;
        }
        let built = emu::build_emulated(&clang, &e.text, e.device.as_deref().unwrap());
        let outs: Vec<String> = ["0", "1"]
            .iter()
            .map(|p| untimed(&built.run(&[("MAPAL_GPU_PICK", p)]).0))
            .collect();
        if outs[0] == outs[1] {
            continue;
        }
        let ids: Vec<String> = ts[0]
            .rows
            .iter()
            .map(|r| format!("kernel=mapal_tile_{:016x}", r[0] as u64))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        for (ns, pick) in [("1000000,800000", 1), ("800000,1000000", 0)] {
            let w = dir.path().join(format!("w{pick}"));
            let (out, err) = built.run(&[
                ("MAPAL_EMU_NS", ns),
                ("MAPAL_WISDOM", w.to_str().unwrap()),
                ("MAPAL_GPU_TRACE", "1"),
            ]);
            assert_eq!(decisions(&err), vec![("trial".into(), pick)], "{err}");
            assert_eq!(
                untimed(&out),
                outs[pick as usize],
                "#{z} {target}: the winner's bits\n{src}"
            );
            let last = err
                .lines()
                .filter(|l| l.starts_with("mapal_gpu_trace op=launch "))
                .filter_map(|l| ids.iter().position(|k| l.split(' ').any(|f| f == k)))
                .last();
            assert_eq!(
                last,
                Some(pick as usize),
                "#{z} {target}: the site's last launch\n{err}"
            );
        }
        return;
    }
    panic!("no generated program whose two candidates differ in bits");
}
