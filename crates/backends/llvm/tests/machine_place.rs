//! ADR-0050 C4 / C5: **`machine-place`** — the machine model's search as a
//! tile site's launch (`--gpu-on=machine-place`, default off), on programs
//! from the generator (`testgen::GemmProg`: GEMMs it was not designed on):
//!
//! - **R3, every GPU profile.** Every generated GEMM (f32, the TF32 set,
//!   `fold: bf16 | f32`) emits with the rung on, each tile kernel carries
//!   its `; mapal.place` line (schedule, price, terms, open flips), the OFF
//!   arm's text has none and equals the default's; where `llc` and `ptxas`
//!   are present the device module assembles for the part's architecture
//!   with no spill (the post-lowering gate on: `--place-gate=ptxas`).
//! - **The picks' values within the bound** (float operands: no order is
//!   exact): the hybrid's walked and whole tiles (the whole ones bit for bit
//!   the ordered kernel's), and the search's own picks on every profile.
//! - **C5, the hybrid's values.** The rules' tile under the hybrid
//!   traversal (`--place-trav=hybrid --place-tile`: whole waves ordered,
//!   the tail walked — one launch of `whole + ctas` blocks) and the best
//!   hybrid of the FMA tile (`--place-trav=hybrid`): integer operands (exact
//!   in every order), plain and as an epilogue group (the tail's tickets),
//!   the emulated device equal to the interpreter in every emulator mode —
//!   byte for byte, as every order is exact on these operands (a whole
//!   tile's block is the ordered kernel's fold).

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const PARTS: [&str; 8] = [
    "cuda-ada",
    "rtx-4090",
    "h100",
    "h100-pcie",
    "rtx-5090",
    "rtx-5080",
    "rtx-3090",
    "a100",
];

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

/// `src` under `fold: <set>` (a generated GEMM's fold).
fn face(src: String, set: &str) -> String {
    assert!(src.contains("(0.0, kr) -> fold {"), "{src}");
    match set {
        "" => src,
        s => src.replace("(0.0, kr) -> fold {", &format!("(0.0, kr) -> fold: {s} {{")),
    }
}

fn emission(ir: &CategoryIr, target: &'static str, on: &[&str], extra: &[&str]) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    for r in ["pdl", "streams", "pinned", "horizontal", "calls"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
    let mut opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::default()
    };
    for f in extra {
        assert!(mapal_backend_llvm::flags::apply(&mut opts, f).expect("a flag"));
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// Each tile kernel's plan fields (`; mapal.plan`) and its `; mapal.place`
/// line, if any.
fn plans(e: &Emission) -> Vec<(HashMap<String, String>, Option<String>)> {
    let dev = e.device.as_deref().unwrap_or_default();
    let lines: Vec<&str> = dev.lines().collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !l.starts_with("; mapal.plan ") || l.contains(" kind=cell") || l.contains(" kind=reduce")
        {
            continue;
        }
        let f: HashMap<String, String> = l
            .split(' ')
            .filter_map(|f| f.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let place = lines[i + 1..]
            .iter()
            .take_while(|x| x.starts_with("; "))
            .find(|x| x.starts_with("; mapal.place "))
            .map(|x| x.to_string());
        out.push((f, place));
    }
    out
}

/// `llc` + `ptxas -v` on a device module for `arch` (`None` where either
/// tool is missing): each kernel's registers and spill bytes.
fn ptxas(dev: &str, arch: &str) -> Option<Vec<(String, u64, u64)>> {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("llc") || !have("ptxas") {
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    std::fs::write(p("d.ll"), dev).unwrap();
    let mut llc = Command::new("llc");
    llc.args(["-O3", &format!("-mcpu={arch}"), "-nvptx-fma-level=0"]);
    if let Some(x) = mapal_backend_llvm::Ptx::declared(dev) {
        llc.arg(format!("-mattr={}", x.mattr()));
    }
    let ok = llc
        .arg(p("d.ll"))
        .arg("-o")
        .arg(p("d.ptx"))
        .status()
        .unwrap()
        .success();
    assert!(ok, "llc -mcpu={arch} refused the module");
    let out = Command::new("ptxas")
        .args([&format!("-arch={arch}"), "-v"])
        .arg(p("d.ptx"))
        .arg("-o")
        .arg(p("d.cubin"))
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "ptxas -arch={arch}: {log}");
    let mut ks = Vec::new();
    for f in log.split("Compiling entry function '").skip(1) {
        let name = f.split('\'').next().unwrap_or("").to_string();
        let regs = f
            .split("Used ")
            .nth(1)
            .and_then(|r| r.split(' ').next())
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let spill = f
            .split(" bytes spill stores")
            .next()
            .and_then(|r| r.rsplit(' ').next())
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        ks.push((name, regs, spill));
    }
    Some(ks)
}

/// The part's architecture (`-mcpu`), by its profile.
fn arch(part: &str) -> &'static str {
    mapal_backend_llvm::resolve_profile(part)
        .and_then(|p| p.gpu())
        .map(|g| g.arch)
        .expect("a GPU part")
}

/// Generated GEMMs at plan extents (`n` a multiple of 128 so the tensor
/// tiles divide; scales 0–2), under each face.
fn programs(n: usize) -> Vec<(String, String)> {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let mut out = Vec::new();
    let mut i = 0;
    while out.len() < n {
        let p = strat.new_tree(&mut runner).unwrap().current();
        if p.scale > 2 {
            continue;
        }
        let (m, nn, k) = testgen::gemm_dims(&p);
        let set = ["", "tf32 | f32", "bf16 | f32"][i % 3];
        i += 1;
        out.push((
            format!("#{i} {m}x{nn}x{k} {set}"),
            face(testgen::gemm_src_at(&p, m, nn, k), set),
        ));
    }
    out
}

#[test]
fn r3_generated_gemms_place_lower_and_assemble_on_every_part() {
    let n = if cfg!(debug_assertions) { 3 } else { 12 };
    let progs = programs(n);
    let checks: Vec<(String, &'static str)> = progs
        .iter()
        .flat_map(|(tag, _)| PARTS.iter().map(move |p| (tag.clone(), *p)))
        .collect();
    let results = testgen::par_check(&checks, |_, (tag, part)| {
        let src = &progs.iter().find(|(t, _)| t == tag).unwrap().1;
        let ir = lower(src);
        let on = emission(&ir, part, &["machine-place"], &["--place-gate=ptxas"]);
        let off = emission(&ir, part, &[], &[]);
        let ps = plans(&on);
        assert!(!ps.is_empty(), "{tag} {part}: a tile kernel");
        // Every tile kernel the pick's realization launches says where it
        // was placed; the OFF arm never does.
        let placed = ps.iter().filter(|(_, l)| l.is_some()).count();
        assert!(placed > 0, "{tag} {part}: no `; mapal.place` line");
        let dev_off = off.device.clone().unwrap_or_default();
        assert!(
            !dev_off.contains("; mapal.place "),
            "{tag} {part}: OFF placed"
        );
        let kept = ps
            .iter()
            .filter_map(|(_, l)| l.as_ref())
            .filter(|l| l.contains(" kept=1 "))
            .count();
        let open = ps
            .iter()
            .filter_map(|(_, l)| l.as_ref())
            .filter(|l| !l.contains(" open=[]"))
            .count();
        let changed =
            on.device.as_deref().map(strip_place) != off.device.as_deref().map(strip_place);
        // A kernel the search placed (not the rules' kept) assembles with no
        // spill: the gate refused the ones that do. A kept rules' kernel's
        // spill is the rules' own (counted, not the search's).
        let (mut clean, mut rules_spill) = (0, 0);
        if let Some(ks) = ptxas(on.device.as_deref().unwrap_or_default(), arch(part)) {
            for (k, _, s) in &ks {
                let placed_here = ps.iter().any(|(f, l)| {
                    f.get("kernel") == Some(k) && l.as_ref().is_some_and(|l| l.contains(" kept=0 "))
                });
                match (placed_here, *s) {
                    (true, s) => {
                        assert_eq!(s, 0, "{tag} {part}: the placed {k} spills");
                        clean += 1;
                    }
                    (false, 0) => clean += 1,
                    (false, _) => rules_spill += 1,
                }
            }
        }
        (placed, kept, open, u8::from(changed), clean, rules_spill)
    });
    let sum = |f: fn(&(usize, usize, usize, u8, usize, usize)) -> usize| {
        results.iter().map(f).sum::<usize>()
    };
    eprintln!(
        "machine-place R3: {} emissions, {} placed kernels ({} kept the rules' launch, {} open), \
         {} changed, {} kernels assembled with no spill, {} rules' kernels spill",
        results.len(),
        sum(|r| r.0),
        sum(|r| r.1),
        sum(|r| r.2),
        sum(|r| r.3 as usize),
        sum(|r| r.4),
        sum(|r| r.5)
    );
}

/// A device module without its `; mapal.place` lines.
fn strip_place(dev: &str) -> String {
    dev.lines()
        .filter(|l| !l.starts_with("; mapal.place "))
        .collect::<Vec<_>>()
        .join("\n")
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

fn num(p: &HashMap<String, String>, k: &str) -> u64 {
    p.get(k).and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// **C5: the hybrid's values through the emulator.** 2 generated GEMMs
/// at runnable extents: `fold: bf16 | f32` at 512 × 6144 × 64 (the rules'
/// 16-bit warp tile, 192 tiles on 168 slots: one whole wave, the first 24
/// tiles walked) plain and as an epilogue group, and f32 at 256 × 2048 ×
/// 128 (the best register-staged FMA tile's hybrid) — each kernel a hybrid
/// (`whole=` on its plan line), the emulated device equal to the
/// interpreter in every mode.
#[test]
fn c5_hybrid_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..2)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let mut hybrids = 0usize;
        let cases: [(&str, (u64, u64, u64), bool, &[&str]); 3] = [
            (
                "bf16 | f32",
                (512, 6144, 64),
                false,
                &["--place-trav=hybrid", "--place-tile"],
            ),
            (
                "bf16 | f32",
                (512, 6144, 64),
                true,
                &["--place-trav=hybrid", "--place-tile"],
            ),
            ("", (256, 2048, 128), false, &["--place-trav=hybrid"]),
        ];
        for (set, (m, n, k), epi, flags) in cases {
            let src = face(
                match epi {
                    false => testgen::gemm_src_at(p, m, n, k),
                    true => testgen::gemm_epi_src_at(p, m, n, k, z as u8),
                },
                set,
            );
            let ctx = format!("#{z} {set} {m}x{n}x{k} epi={epi}");
            let ir = lower(&src);
            let rr = run(&ir, 20_000_000_000);
            assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
            let want = rr.output;
            let on = emission(&ir, "rtx-5080", &["machine-place"], flags);
            let ps = plans(&on);
            let h = ps.iter().filter(|(f, _)| num(f, "whole") > 0).count();
            assert!(h > 0, "{ctx}: the forced hybrid\n{:?}", ps);
            hybrids += h;
            let dev = on.device.clone().unwrap_or_default();
            let built = emu::build_emulated(&cc, &on.text, &dev);
            for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
                assert_eq!(got, want, "{ctx} {env:?}\n{src}");
            }
        }
        hybrids
    });
    let n: usize = counts.iter().sum();
    eprintln!("hybrid values: {n} hybrid kernels, each equal to the interpreter in 3 modes");
    assert!(n >= 4);
}

/// The printed floats of an output, in order.
fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|w| w.parse::<f64>().unwrap_or(f64::NAN))
        .collect()
}

/// The printed cells of `got` against their exact sums `want` (`(exact,
/// M)`): each within `c·M + 2⁻²²·|w|` — the set's a-priori coefficient `c`
/// of the exact sum (ADR-0049: correct is the flow and the bound, not
/// bit-identity).
fn within(got: &str, want: &[(f64, f64)], c: f64, what: &str) {
    let g = floats(got);
    assert_eq!(g.len(), want.len(), "{what}");
    for (i, (&y, &(w, m))) in g.iter().zip(want).enumerate() {
        let tol = c * m + 2f64.powi(-22) * w.abs();
        assert!(
            (y - w).abs() <= tol,
            "{what} cell {i}: {y} vs {w} (M {m}, tol {tol})"
        );
    }
}

/// The a-priori coefficient of a `fold: <set>` GEMM's product at `k`
/// against the exact sum: the set's weakest member's (its operands' unit),
/// and the f32 sum's.
fn coef(set: &str, k: u64) -> f64 {
    let unit = match set {
        "bf16 | f32" => Some(2f64.powi(-8)),
        "tf32 | f32" => Some(2f64.powi(-11)),
        _ => None,
    };
    mapal_ir::bound_coef_unit(k, unit) + mapal_ir::bound_coef_unit(k, None)
}

/// **ADR-0050 review: the search's picks' values within the a-priori
/// bound, with float data.** Integer operands are exact in any order, so
/// they cannot show a rounding or fold-order defect; these operands are not
/// (`v / 7`, `v / 9`), and a cell of every 32 × 64 block (every output tile
/// sampled) is checked against its exact sum and `M` (`testgen::
/// gemm_bound_ref`, f64 over the program's f32 operands):
///
/// - **the hybrid** (the 16-bit warp tile at 512 × 6144 × 64: 168 whole
///   tiles, the raster order's first 24 walked, the zero pass their tile
///   row): every sampled cell within the bound in every emulator mode, the
///   walked tiles' cells among them (counted); a whole tile's cells **bit
///   for bit** the ordered kernel's (one piece, the same fold);
/// - **the search's own picks** (the rung alone: the default band and
///   gate) on every GPU profile — a generated GEMM each, the three sets in
///   turn: the emulated device within the bound.
#[test]
fn picks_within_the_bound_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..PARTS.len())
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let dev = |e: &Emission| e.device.clone().unwrap_or_default();
    // Case 0: the hybrid against the ordered kernel of the same tile; case
    // `1 + z`: the search's own pick on part `z` (one parallel pass: the
    // interpreter's and the emulator's runs overlap).
    let cases: Vec<usize> = (0..=PARTS.len()).collect();
    let res = testgen::par_check(&cases, |_, &case| {
        if case > 0 {
            let z = case - 1;
            let part = PARTS[z];
            let p = &progs[z];
            let set = ["", "tf32 | f32", "bf16 | f32"][z % 3];
            let (m, n, k) = [(128u64, 1024u64, 256u64), (256, 512, 128), (64, 2048, 128)][z % 3];
            let (src, cells) = testgen::gemm_bound_src_at(p, m, n, k, (32, 64));
            let want = testgen::gemm_bound_ref(p, (m, n, k), (32, 64), &cells);
            let src = face(src, set);
            let ir = lower(&src);
            let on = emission(&ir, part, &["machine-place"], &[]);
            let replaced = plans(&on)
                .iter()
                .filter_map(|(_, l)| l.as_ref())
                .filter(|l| l.contains(" kept=0 "))
                .count();
            let got = emu::run_emulated(&cc, &on.text, &dev(&on));
            within(
                &got,
                &want,
                coef(set, k),
                &format!("{part} {m}x{n}x{k} {set}"),
            );
            return (cells.len(), replaced, 0, 0);
        }
        let p = &progs[0];
        let (m, n, k) = (512, 6144, 64);
        let (src, cells) = testgen::gemm_bound_src_at(p, m, n, k, (32, 64));
        let want = testgen::gemm_bound_ref(p, (m, n, k), (32, 64), &cells);
        let src = face(src, "bf16 | f32");
        let ir = lower(&src);
        let hyb = emission(
            &ir,
            "rtx-5080",
            &["machine-place"],
            &["--place-trav=hybrid", "--place-tile"],
        );
        let ord = emission(
            &ir,
            "rtx-5080",
            &["machine-place"],
            &["--place-trav=ordered", "--place-tile"],
        );
        let ps = plans(&hyb);
        let (t, _) = ps
            .iter()
            .find(|(f, _)| num(f, "whole") > 0)
            .unwrap_or_else(|| panic!("the forced hybrid\n{ps:?}"));
        let (gx, gy, h) = (num(t, "gx"), num(t, "gy"), num(t, "raster").max(1));
        assert!(gx > 0 && gy > 0 && n % gx == 0 && m % gy == 0, "{t:?}");
        let (bm, bn) = (m / gy, n / gx);
        let walked_tiles = gx * gy - num(t, "whole");
        // The raster order's tile `l` (groups of `h` block-rows, column by
        // column): the walk takes the first `walked_tiles`.
        let tile_of = |l: u64| {
            let (grp, rem) = (l / (gx * h), l % (gx * h));
            let rows = h.min(gy - grp * h);
            (grp * h + rem % rows, rem / rows)
        };
        let walked: std::collections::HashSet<(u64, u64)> =
            (0..walked_tiles).map(tile_of).collect();
        let built = emu::build_emulated(&cc, &hyb.text, &dev(&hyb));
        let c = coef("bf16 | f32", k);
        let mut got_h = String::new();
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            within(&got, &want, c, &format!("hybrid {env:?}"));
            got_h = got;
        }
        let got_o = emu::run_emulated(&cc, &ord.text, &dev(&ord));
        within(&got_o, &want, c, "ordered");
        let (gh, go) = (floats(&got_h), floats(&got_o));
        let (mut in_walk, mut whole_equal) = (0, 0);
        for (x, &(i, j)) in cells.iter().enumerate() {
            match walked.contains(&(i / bm, j / bn)) {
                true => in_walk += 1,
                false => {
                    assert_eq!(
                        gh[x].to_bits(),
                        go[x].to_bits(),
                        "a whole tile's cell ({i}, {j}) is the ordered kernel's"
                    );
                    whole_equal += 1;
                }
            }
        }
        assert!(
            in_walk > 0 && whole_equal > 0,
            "{in_walk} walked, {whole_equal} whole"
        );
        (cells.len(), 0, in_walk, whole_equal)
    });
    let (h, res) = (res[0], &res[1..]);
    eprintln!(
        "hybrid 512x6144x64: {} cells within the bound in 3 modes, {} in walked tiles; {} \
         whole-tile cells bit-equal to the ordered kernel",
        h.0, h.2, h.3
    );
    let checked: usize = res.iter().map(|r| r.0).sum();
    let replaced: usize = res.iter().map(|r| r.1).sum();
    eprintln!(
        "machine-place picks: {} parts, {checked} cells within the bound, {replaced} placed \
         kernels replaced the rules' launch",
        res.len()
    );
}
