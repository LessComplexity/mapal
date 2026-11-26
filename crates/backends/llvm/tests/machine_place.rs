//! ADR-0050 C4: **`machine-place`** — the machine model's search as a
//! tile site's launch (`--gpu-on=machine-place`, default off), on programs
//! from the generator (`testgen::GemmProg`: GEMMs it was not designed on):
//!
//! - **R3, every GPU profile.** Every generated GEMM (f32, the TF32 set,
//!   `fold: bf16 | f32`) emits with the rung on, each tile kernel carries
//!   its `; mapal.place` line (schedule, price, terms, open flips), the OFF
//!   arm's text has none and equals the default's; where `llc` and `ptxas`
//!   are present the device module assembles for the part's architecture
//!   with no spill (the post-lowering gate on: `--place-gate=ptxas`).

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

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
