//! `mapal build|run|dump-ir|test` — the driver over
//! parse → lower → check → rewrite → emit → (opt∘llc) → link → exec, with the
//! interpreter as the oracle (docs/components/cli/DESIGN.md).
//!
//! It makes no codegen decision: every optimization in its output is derived
//! inside `mapal-backend-llvm` from the IR artifacts against the named
//! `TargetProfile`. The CLI routes `EmitOpts` (the shared emitter flags, so
//! every rung's OFF arm is reachable here) and runs the toolchain; the one
//! toolchain parameter that depends on the machine, the device unit's
//! `-mcpu`, is read from the profile's GPU facts.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use mapal_backend_llvm::{EmitError, EmitOpts, Order, emit_detailed, flags, resolve_profile};
use mapal_interp::{Outcome, TrapKind, native_observable};
use mapal_ir::CategoryIr;
use mapal_syntax::{Diagnostic, LineIndex, Severity};

const USAGE: &str = "usage: mapal <build|run|dump-ir|test> <file.mapal> [options]
  build    [-o <exe>] [--emit-only]       emit <exe>.ll (+ .dev.ll/.ptx on a GPU profile) and link
  run      [--interp [--weakest]] [--budget N]  compile and execute (--interp: the oracle)
  dump-ir  [--stage=lowered|rewritten|llvm|device]
  test     [--budget N] [--timeout SECS]  the fixed face (or --exact) vs the oracle, byte for byte
           [--weakest]                    vs the weakest-member oracle, within ADR-0041 D3's bound
common: --target=<profile> --no-rewrite -v, and the emitter flags
  --perf --no-tile --no-pack --kc --move-panel=off|W:B
  --exact --no-contract --contract --order=sequential|fixed|free
  --gpu-off=<rung,..> --gpu-on=<rung,..> --splitk=<S>
  (default face: fma steps, order free, bounded — ADR-0040)
env: CC (clang), MAPAL_RT (libmapal_rt.a), LLVM (opt/llc dir), MAPAL_CUDA, MAPAL_LD";

/// `Err` carries the exit code; the message is already on stderr.
type Res<T> = Result<T, u8>;

#[derive(Clone)]
struct Args {
    cmd: String,
    file: String,
    out: Option<PathBuf>,
    emit_only: bool,
    interp: bool,
    /// ADR-0041: the oracle evaluates each product with its weakest member.
    weakest: bool,
    budget: u64,
    timeout: u64,
    stage: String,
    rewrite: bool,
    verbose: bool,
    opts: EmitOpts,
    /// The face flags as given (`test` judges only bit-replayable faces).
    face_flags: Vec<String>,
}

fn main() -> ExitCode {
    let code = match parse_args(std::env::args().skip(1)) {
        Ok(a) => match a.cmd.as_str() {
            "build" => build_cmd(&a),
            "run" => run_cmd(&a),
            "dump-ir" => dump_ir(&a),
            "test" => test_cmd(&a),
            _ => unreachable!("checked in parse_args"),
        },
        Err(c) => Err(c),
    };
    ExitCode::from(code.unwrap_or_else(|c| c))
}

fn usage(msg: &str) -> u8 {
    eprintln!("mapal: {msg}\n{USAGE}");
    2
}

fn parse_args(mut it: impl Iterator<Item = String>) -> Res<Args> {
    let cmd = it.next().unwrap_or_default();
    if !matches!(cmd.as_str(), "build" | "run" | "dump-ir" | "test") {
        return Err(usage(&format!("unknown subcommand `{cmd}`")));
    }
    let mut a = Args {
        cmd,
        file: String::new(),
        out: None,
        emit_only: false,
        interp: false,
        weakest: false,
        budget: 100_000_000,
        timeout: 60,
        stage: "llvm".into(),
        rewrite: true,
        verbose: false,
        opts: EmitOpts::default(),
        face_flags: Vec::new(),
    };
    let num = |v: Option<String>, f: &str| -> Res<u64> {
        v.and_then(|v| v.parse().ok())
            .ok_or_else(|| usage(&format!("{f} expects a non-negative integer")))
    };
    while let Some(s) = it.next() {
        if matches!(s.as_str(), "--contract" | "--no-contract" | "--exact") || s.starts_with("--order=") {
            a.face_flags.push(s.clone());
        }
        match flags::apply(&mut a.opts, &s) {
            Ok(true) => continue,
            Err(e) => return Err(usage(&e)),
            Ok(false) => {}
        }
        match s.as_str() {
            "-o" => a.out = Some(it.next().ok_or_else(|| usage("-o expects a path"))?.into()),
            "--emit-only" => a.emit_only = true,
            "--interp" => a.interp = true,
            "--weakest" => a.weakest = true,
            "--budget" => a.budget = num(it.next(), "--budget")?,
            "--timeout" => a.timeout = num(it.next(), "--timeout")?,
            "--no-rewrite" => a.rewrite = false,
            "-v" => a.verbose = true,
            _ if s.starts_with("--stage=") => a.stage = s["--stage=".len()..].into(),
            _ if !s.starts_with('-') && a.file.is_empty() => a.file = s,
            _ => return Err(usage(&format!("unknown argument `{s}`"))),
        }
    }
    if a.file.is_empty() {
        return Err(usage("no input file"));
    }
    Ok(a)
}

// --- front end ------------------------------------------------------------

/// `file:line:col`, the one location format every stage's report uses.
fn at(file: &str, lines: &LineIndex<'_>, offset: u32) -> String {
    let lc = lines.line_col(offset);
    format!("{file}:{}:{}", lc.line, lc.col)
}

/// parse → lower → check: the checked, lowered IR (the oracle's input).
fn frontend(a: &Args) -> Res<CategoryIr> {
    let src = std::fs::read_to_string(&a.file).map_err(|e| {
        eprintln!("mapal: cannot read {}: {e}", a.file);
        2
    })?;
    let lines = LineIndex::new(&src);
    let report = |stage: &str, ds: &[Diagnostic]| {
        for d in ds {
            let sev = match d.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
            };
            let loc = at(&a.file, &lines, d.span.start);
            eprintln!("{loc} {sev}[{}] ({stage}) {}", d.code.0, d.message);
        }
    };
    let po = mapal_syntax::parse(&src);
    report("parse", &po.diagnostics);
    if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(1);
    }
    let ir = mapal_lower::lower(&src, &po.program).map_err(|ds| {
        report("lower", &ds);
        1
    })?;
    let ds = mapal_check::check(&src, &po.program, &ir);
    if !ds.is_empty() {
        report("check", &ds);
        return Err(1);
    }
    Ok(ir)
}

/// The compiled arm's IR: rewritten unless `--no-rewrite` (the shipped
/// pipeline is `emit --rewrite`; the differential suite proves both arms).
fn rewritten(a: &Args, ir: CategoryIr) -> CategoryIr {
    if !a.rewrite {
        return ir;
    }
    let r = mapal_rewrite::rewrite(ir);
    if a.verbose {
        eprintln!("rewrite: {:?}", r.report);
    }
    r.ir
}

fn emit(a: &Args, ir: &CategoryIr) -> Res<mapal_backend_llvm::Emission> {
    emit_detailed(ir, &a.opts).map_err(|e| {
        match e {
            EmitError::Unsupported { feature, loc } => {
                let src = std::fs::read_to_string(&a.file).unwrap_or_default();
                let loc = at(&a.file, &LineIndex::new(&src), loc.start);
                eprintln!("{loc} error[B0001] (emit) unsupported: {feature}");
            }
            EmitError::Internal(m) => eprintln!("mapal: emit: {m}"),
        }
        1
    })
}

// --- toolchain ------------------------------------------------------------

/// `<path><ext>` — artifacts sit next to the executable, named after it.
fn side(exe: &Path, ext: &str) -> PathBuf {
    let mut s = exe.as_os_str().to_owned();
    s.push(ext);
    s.into()
}

/// S64 evidence pass (`MAPAL_EVIDENCE=1`): the step's LLVM optimization
/// remarks beside `exe` — `<exe>.host.remarks.yaml` (clang),
/// `<exe>.dev.{opt,llc}.remarks.yaml` — for `benches/gpu/evidence.py`.
fn remarks(cmd: &mut Command, exe: &Path, tool: &str) {
    if env("MAPAL_EVIDENCE").as_deref() != Some("1") {
        return;
    }
    if tool == "clang" {
        let f = side(exe, ".host.remarks.yaml");
        cmd.arg("-fsave-optimization-record=yaml")
            .arg(format!("-foptimization-record-file={}", f.display()))
            .arg("-foptimization-record-passes=loop-vectorize|loop-unroll|slp-vectorizer|inline|sroa|licm");
    } else {
        let f = side(exe, &format!(".dev.{tool}.remarks.yaml"));
        cmd.arg(format!("-pass-remarks-output={}", f.display()));
        if tool == "opt" {
            cmd.arg("-pass-remarks-filter=loop-unroll|sroa|licm|gvn|instcombine|loop-vectorize");
        }
    }
}

fn write(p: &Path, text: &str) -> Res<()> {
    std::fs::write(p, text).map_err(|e| {
        eprintln!("mapal: write {}: {e}", p.display());
        1
    })
}

/// Run a toolchain step: a tool that cannot be spawned is exit 3 (toolchain
/// missing), one that fails is exit 1.
fn sh(a: &Args, mut cmd: Command, hint: &str) -> Res<()> {
    if a.verbose {
        eprintln!("+ {cmd:?}");
    }
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => {
            eprintln!("mapal: {:?} failed ({s})", cmd.get_program());
            Err(1)
        }
        Err(e) => {
            eprintln!("mapal: cannot run {:?}: {e} ({hint})", cmd.get_program());
            Err(3)
        }
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn cc() -> Command {
    Command::new(env("CC").unwrap_or_else(|| "clang".into()))
}

/// `$LLVM/<name>` (bundle.sh's convention), else `<name>` on PATH.
fn llvm_tool(name: &str) -> Command {
    Command::new(env("LLVM").map_or(name.into(), |d| Path::new(&d).join(name)))
}

/// `$MAPAL_RT`, else `libmapal_rt.a` next to this binary — cargo uplifts a
/// staticlib only when it is built as a root (`cargo build -p mapal-rt`).
fn rt_lib() -> Res<PathBuf> {
    let p = env("MAPAL_RT").map(PathBuf::from).or_else(|| {
        let exe = std::env::current_exe().ok()?;
        Some(exe.parent()?.join("libmapal_rt.a"))
    });
    match p {
        Some(p) if p.exists() => Ok(p),
        p => {
            eprintln!(
                "mapal: runtime library not found{} (run `cargo build -p mapal-rt`, or set MAPAL_RT)",
                p.map(|p| format!(" at {}", p.display()))
                    .unwrap_or_default()
            );
            Err(3)
        }
    }
}

/// Emit `ir` and produce `exe` (or, with `--emit-only`, its translation
/// units). Returns the resolved profile and the PTX a GPU build loads.
fn build(a: &Args, ir: &CategoryIr, exe: &Path) -> Res<(&'static str, Option<PathBuf>)> {
    let em = emit(a, ir)?;
    let host = side(exe, ".ll");
    write(&host, &em.text)?;
    let mut wrote = vec![host.clone()];
    // A device unit exists only on a GPU profile with an offloadable site;
    // `-mcpu` is that profile's hardware fact, `-nvptx-fma-level=0` keeps llc
    // from contracting what the emitter chose per face (fmul/fadd on Exact,
    // llvm.fma on Contract). opt -O3 / llc -O3 are bundle.sh's measured flags.
    let ptx = match &em.device {
        None => None,
        Some(dev) => {
            // S62 `hostopt` names a GPU build `<gpu>+<host>`: the device
            // half is the GPU profile's.
            let arch = resolve_profile(em.profile.split('+').next().unwrap_or(em.profile))
                .and_then(|p| p.gpu())
                .expect("a device unit comes from a GPU profile")
                .arch;
            let (dll, oll, ptx) = (
                side(exe, ".dev.ll"),
                side(exe, ".dev.opt.ll"),
                side(exe, ".ptx"),
            );
            write(&dll, dev)?;
            let mut opt = llvm_tool("opt");
            opt.arg("-O3").arg(&dll).arg("-S").arg("-o").arg(&oll);
            remarks(&mut opt, exe, "opt");
            sh(a, opt, "set LLVM=<dir with opt/llc>")?;
            let mut llc = llvm_tool("llc");
            llc.args(["-O3", &format!("-mcpu={arch}"), "-nvptx-fma-level=0"])
                .arg(&oll)
                .arg("-o")
                .arg(&ptx);
            remarks(&mut llc, exe, "llc");
            sh(a, llc, "set LLVM=<dir with opt/llc>")?;
            wrote.extend([dll, ptx.clone()]);
            Some(ptx)
        }
    };
    let list = |w: &[PathBuf]| {
        w.iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    if a.emit_only {
        eprintln!("wrote {} [target={}]", list(&wrote), em.profile);
        return Ok((em.profile, ptx));
    }
    let rt = rt_lib()?;
    let mut link = cc();
    // The emitted module names no triple (it is the host's), which clang
    // reports as an override; bundle.sh silences the same note.
    let quiet = "-Wno-override-module";
    if ptx.is_some() {
        // The GPU half of the runtime links against libcuda, so it is built
        // here, next to the host object — box_run.sh's link line exactly.
        let cuda = PathBuf::from(env("MAPAL_CUDA").unwrap_or_else(|| "/usr/local/cuda".into()));
        if !cuda.join("include/cuda.h").exists() {
            eprintln!(
                "wrote {}; linking needs CUDA at {} (set MAPAL_CUDA)",
                list(&wrote),
                cuda.display()
            );
            return Err(3);
        }
        let (obj, gpu_c) = (side(exe, ".o"), side(exe, ".mapal_gpu.c"));
        let mut c = cc();
        c.args(["-O2", "-c", quiet]).arg(&host).arg("-o").arg(&obj);
        remarks(&mut c, exe, "clang");
        sh(a, c, "set CC=<clang>")?;
        write(&gpu_c, include_str!("../../mapal-rt/gpu/mapal_gpu.c"))?;
        link.arg(&obj).arg(&gpu_c).arg(&rt);
        link.arg(format!("-I{}", cuda.join("include").display()));
        link.arg(format!("-L{}", cuda.join("lib64/stubs").display()));
        link.args(["-lcuda", "-lpthread", "-ldl", "-lm"]);
    } else {
        link.args(["-O2", quiet]).arg(&host).arg(&rt);
        remarks(&mut link, exe, "clang");
    }
    if let Some(ld) = env("MAPAL_LD") {
        link.arg(format!("-fuse-ld={ld}"));
    }
    link.arg("-o").arg(exe);
    sh(a, link, "set CC=<clang>")?;
    wrote.push(exe.to_owned());
    if a.cmd == "build" || a.verbose {
        eprintln!("wrote {} [target={}]", list(&wrote), em.profile);
    }
    Ok((em.profile, ptx))
}

/// Build into a fresh temp dir, hand the executable to `f`, then clean up.
fn in_temp<T>(
    a: &Args,
    ir: &CategoryIr,
    f: impl FnOnce(Command) -> Res<T>,
) -> Res<(&'static str, T)> {
    let dir = std::env::temp_dir().join(format!("mapal-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let stem = Path::new(&a.file).file_stem().unwrap_or_default();
    let exe = dir.join(stem);
    let r = build(a, ir, &exe).and_then(|(profile, ptx)| {
        let mut cmd = Command::new(&exe);
        if let Some(p) = ptx {
            cmd.env("MAPAL_PTX", p);
        }
        Ok((profile, f(cmd)?))
    });
    let _ = std::fs::remove_dir_all(&dir);
    r
}

// --- subcommands ----------------------------------------------------------

fn build_cmd(a: &Args) -> Res<u8> {
    let ir = rewritten(a, frontend(a)?);
    let exe = a
        .out
        .clone()
        .unwrap_or_else(|| Path::new(&a.file).with_extension(""));
    if exe == Path::new(&a.file) {
        return Err(usage(
            "the executable would overwrite the source; pass -o <exe>",
        ));
    }
    build(a, &ir, &exe)?;
    Ok(0)
}

fn run_cmd(a: &Args) -> Res<u8> {
    let ir = frontend(a)?;
    if !a.interp {
        let ir = rewritten(a, ir);
        let (_, code) = in_temp(a, &ir, |mut cmd| {
            let st = cmd.status().map_err(|e| {
                eprintln!("mapal: cannot run the built program: {e}");
                1
            })?;
            Ok(st.code().unwrap_or(1))
        })?;
        return Ok(code as u8);
    }
    // The oracle prints the observable the compiled program must (L1/BL8),
    // and exits with its code: 0, or 101 on a trap.
    let rr = if a.weakest {
        mapal_interp::run_weakest(&ir, a.budget, &[])
    } else {
        mapal_interp::run(&ir, a.budget)
    };
    let Some((out, code)) = native_observable(&ir, &rr) else {
        eprintln!(
            "mapal: diverged — step budget ({}) exhausted (E1). Raise it with --budget N.",
            a.budget
        );
        return Err(1);
    };
    print!("{}", out.unwrap_or_default());
    if let Outcome::Trapped(k) = rr.outcome {
        let what = match k {
            TrapKind::DivZero => "division/modulo by zero",
            TrapKind::IndexOob => "array index out of bounds",
        };
        eprintln!("mapal: trapped — {what} (ADR-0013).");
    }
    Ok(code as u8)
}

fn dump_ir(a: &Args) -> Res<u8> {
    let ir = frontend(a)?;
    match a.stage.as_str() {
        "lowered" => println!("{ir:#?}"),
        "rewritten" => println!("{:#?}", rewritten(a, ir)),
        "llvm" => print!("{}", emit(a, &rewritten(a, ir))?.text),
        "device" => match emit(a, &rewritten(a, ir))?.device {
            Some(dev) => print!("{dev}"),
            None => {
                eprintln!(
                    "mapal: target `{}` has no device unit for this program",
                    a.opts.target
                );
                return Err(1);
            }
        },
        s => return Err(usage(&format!("unknown stage `{s}`"))),
    }
    Ok(0)
}

/// The equalizer check: `exec∘link∘emit∘rewrite ≡ interp` on the observable
/// (exit code, and stdout byte for byte when the oracle defines it), on the
/// `fixed` face (ADR-0040 D3: separate steps, the compiled partition order,
/// which the oracle replays) — or `--exact` (sequential).
fn test_cmd(a: &Args) -> Res<u8> {
    let given = |f: &str| a.face_flags.iter().any(|x| x == f);
    let separate = given("--no-contract") || given("--exact");
    if given("--contract")
        || given("--order=free")
        || (given("--order=sequential") && !separate)
    {
        eprintln!(
            "mapal: test judges the fixed face (or --exact) — the default, contract and \
             reduced faces are bound-checked, not bit-equal to the interpreter \
             (ADR-0040 D2; benches/gpu/value_gate.py)"
        );
        return Err(2);
    }
    let mut a = a.clone();
    a.opts.contract = false;
    if a.opts.order == Order::Free {
        a.opts.order = Order::Fixed;
    }
    let ir = rewritten(&a, frontend(&a)?);
    // ADR-0041: a `: tf32` product may take the tensor-core kernel, which is
    // not the f32 oracle's bytes — judged against the weakest member instead.
    if ir.funcs().any(|(f, _)| !ir.precision_plan(f).set.is_empty()) {
        a.weakest = true;
    }
    let a = &a;
    // The orders are keyed on the IR the compiler saw: a split program's
    // oracle runs on it too (`--no-rewrite`: the lowered IR is that IR); an
    // unsplit one on the lowered IR, as before.
    let orders: Vec<_> = emit(a, &ir)?
        .orders
        .iter()
        .map(|o| (o.func, o.fold, o.s))
        .collect();
    let raw;
    let oracle = if orders.is_empty() {
        raw = frontend(a)?;
        &raw
    } else {
        &ir
    };
    let rr = if a.weakest {
        mapal_interp::run_weakest(oracle, a.budget, &orders)
    } else {
        mapal_interp::run_ordered(oracle, a.budget, &orders)
    };
    let Some((want_out, want_code)) = native_observable(oracle, &rr) else {
        println!("SKIP {}: oracle diverged at budget {}", a.file, a.budget);
        return Err(1);
    };
    let (profile, got) = in_temp(a, &ir, |mut cmd| {
        cmd.stdout(Stdio::piped());
        wait_boxed(cmd, a.timeout)
    })?;
    let tag = format!("{} [target={profile}, -O2]", a.file);
    let Some((got_out, got_code)) = got else {
        println!("FAIL {tag}: timeout ({}s)", a.timeout);
        return Ok(1);
    };
    let got_out = String::from_utf8_lossy(&got_out);
    if got_code != want_code {
        println!("FAIL {tag}: exit want {want_code}, got {got_code}");
        return Ok(1);
    }
    // ADR-0041 D3 from the other side: the compiled f32 realization against
    // the weakest-member oracle, each float within both faces' bounds of the
    // program's longest fold — relative, value_gate.py's fallback (a value
    // under cancellation needs the magnitude oracle there).
    if a.weakest
        && let Some(want) = &want_out
    {
        let rel = weakest_rel(longest_fold(oracle));
        let (w, g): (Vec<_>, Vec<_>) = (want.split_whitespace().collect(), got_out.split_whitespace().collect());
        let bad = (w.len() != g.len()).then(|| format!("{} tokens, want {}", g.len(), w.len())).or_else(|| {
            w.iter().zip(&g).find_map(|(x, y)| match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(p), Ok(q)) if (p - q).abs() <= rel * p.abs().max(q.abs()) => None,
                (Ok(_), Ok(_)) => Some(format!("{y} vs {x} beyond rel {rel:.3e}")),
                _ => (x != y).then(|| format!("{y} != {x}")),
            })
        });
        if let Some(b) = bad {
            println!("FAIL {tag} (--weakest): {b}");
            return Ok(1);
        }
        println!("PASS {tag} (--weakest, rel {rel:.3e})");
        return Ok(0);
    }
    if let Some(want) = want_out
        && want != got_out
    {
        let (w, g): (Vec<_>, Vec<_>) = (want.lines().collect(), got_out.lines().collect());
        match (0..w.len().max(g.len())).find(|&i| w.get(i) != g.get(i)) {
            Some(i) => println!(
                "FAIL {tag}: stdout differs at line {}: want `{}`, got `{}`",
                i + 1,
                w.get(i).unwrap_or(&"<eof>"),
                g.get(i).unwrap_or(&"<eof>")
            ),
            None => println!("FAIL {tag}: stdout differs in the trailing newline"),
        }
        return Ok(1);
    }
    println!("PASS {tag}");
    Ok(0)
}

/// The longest fold's trip count in `ir` (ADR-0040's `K`).
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

/// `--weakest`'s relative tolerance at fold length `k`: twice (value_gate's
/// fallback) the sum of ADR-0041 D3's TF32 bound and ADR-0040 D2's f32 one
/// (`mapal_ir::bound_coef`, the IR's own) — the two sides of the exact value.
fn weakest_rel(k: u64) -> f64 {
    2.0 * (mapal_ir::bound_coef(k, true) + mapal_ir::bound_coef(k, false))
}

/// Spawn and wait, time-boxed; `Ok(None)` on timeout. Stdout is drained on a
/// thread so a chatty program cannot block on a full pipe.
fn wait_boxed(mut cmd: Command, secs: u64) -> Res<Option<(Vec<u8>, i32)>> {
    let mut child = cmd.spawn().map_err(|e| {
        eprintln!("mapal: cannot run the built program: {e}");
        1
    })?;
    let mut so = child.stdout.take().expect("stdout is piped");
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = so.read_to_end(&mut buf);
        buf
    });
    let start = Instant::now();
    loop {
        if let Ok(Some(st)) = child.try_wait() {
            return Ok(Some((
                reader.join().unwrap_or_default(),
                st.code().unwrap_or(-1),
            )));
        }
        if start.elapsed() > Duration::from_secs(secs) {
            let _ = child.kill();
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
