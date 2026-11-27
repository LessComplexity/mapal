//! The nvcc toolchain, GPU slot gate, compile/run plumbing and oracle-parity assertion.

use super::*;

/// The one skip reason (HANDOFF §5.5), printed by every skip path.
pub(crate) const SKIP_REASON: &str = "nvcc not found ($NVCC unset, $CUDA_HOME/bin/nvcc missing, \
     not on PATH) — the CUDA differential runs on the remote vast.ai 4090 box \
     (DESIGN §6); skipping, never a faked pass (HANDOFF §5.5)";

// --- toolchain ------------------------------------------------------------

/// The pure discovery-order core (§6.5), unit-tested without touching the
/// process env or PATH: `$NVCC` (respected verbatim, llvm's `$CC` rule) →
/// `$CUDA_HOME/bin/nvcc` → `which nvcc`. Candidates arrive pre-validated;
/// the first present wins. `None` ⇒ skip-with-reason.
pub(crate) fn pick_nvcc(
    env_nvcc: Option<String>,
    cuda_home_nvcc: Option<String>,
    which_nvcc: Option<String>,
) -> Option<String> {
    env_nvcc.or(cuda_home_nvcc).or(which_nvcc)
}

/// The `$CUDA_HOME` candidate path (§6.5's middle leg).
pub(crate) fn cuda_home_nvcc(home: &str) -> String {
    format!("{home}/bin/nvcc")
}

/// Discover `nvcc` (§6.5). `$NVCC` is used verbatim (non-empty); a stale
/// `$CUDA_HOME` (candidate not a file) falls through to PATH.
pub(crate) fn nvcc() -> Option<String> {
    if let Ok(v) = std::env::var("NVCC")
        && !v.is_empty()
    {
        return Some(v);
    }
    if let Ok(h) = std::env::var("CUDA_HOME") {
        let p = cuda_home_nvcc(&h);
        if Path::new(&p).is_file() {
            return Some(p);
        }
    }
    let out = Command::new("which").arg("nvcc").output().ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

pub(crate) fn rt_lib() -> PathBuf {
    static BUILD: Once = Once::new();
    // Serialize the mapal-rt staticlib build so parallel test binaries don't race.
    BUILD.call_once(|| {
        let ok = Command::new("cargo")
            .args(["build", "-p", "mapal-rt"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "cargo build -p mapal-rt failed");
    });
    PathBuf::from(format!(
        "{}/../../../target/debug/libmapal_rt.a",
        env!("CARGO_MANIFEST_DIR")
    ))
}

/// The §4/§6.6 compile recipe argv (unit-tested): flags first, then the
/// translation unit, the mapal-rt staticlib, `-o exe`, and the pinned Linux
/// link tail LAST. `--use_fast_math` and host `-march=native`/`-mfma` are
/// forbidden (§4) — `local::compile_recipe_argv_pinned` asserts their absence.
pub(crate) fn nvcc_argv(cu: &Path, rt: &Path, exe: &Path) -> Vec<String> {
    let s = |p: &Path| p.to_str().expect("temp paths are UTF-8").to_string();
    // `-fmad=false` here is the CONFORMANCE pin (byte-parity vs the oracle),
    // deliberately NOT the product default: Sapir flipped the product/bench
    // recipe to `-fmad=true` at S24b close (DESIGN §4 amendment). This gate
    // measures semantics, not speed — contraction stays off.
    vec![
        "-std=c++17".into(),
        "-fmad=false".into(),
        "-arch=sm_89".into(),
        s(cu),
        s(rt),
        "-o".into(),
        s(exe),
        "-lpthread".into(),
        "-ldl".into(),
        "-lm".into(),
    ]
}

/// One time-boxed subprocess run, stdout/stderr captured to files in the
/// tempdir (no pipe-buffer deadlock; stderr feeds the failure reports).
pub(crate) struct ProcOut {
    /// `None` = killed at the timeout.
    pub(crate) code: Option<i32>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// Spawn `cmd`, time-boxed at `secs` (5 ms poll, the llvm cadence). On
/// timeout the child is killed AND reaped; partial output is still read back.
pub(crate) fn run_timeboxed(cmd: &mut Command, dir: &Path, stem: &str, secs: u64) -> ProcOut {
    let out_path = dir.join(format!("{stem}.stdout"));
    let err_path = dir.join(format!("{stem}.stderr"));
    let mut child = cmd
        .stdout(Stdio::from(std::fs::File::create(&out_path).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&err_path).unwrap()))
        .spawn()
        .unwrap_or_else(|e| {
            // A bogus $NVCC (respected verbatim, llvm's CC rule) surfaces
            // here — name the tool, not just the OS error.
            panic!(
                "spawn `{}` failed: {e}",
                cmd.get_program().to_string_lossy()
            )
        });
    let start = Instant::now();
    let code = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status.code().unwrap_or(-1)); // signal death → -1
        }
        if start.elapsed() > Duration::from_secs(secs) {
            let _ = child.kill();
            let _ = child.wait(); // reap
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    ProcOut {
        code,
        stdout: std::fs::read(&out_path).unwrap(),
        stderr: std::fs::read(&err_path).unwrap(),
    }
}

/// What the native run did (the classifier's input).
#[derive(Debug)]
pub(crate) enum RunOut {
    /// The process exited on its own: exit code (-1 on signal death),
    /// captured stdout bytes, captured stderr bytes.
    Exited {
        code: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// Killed at the 15 s run timeout.
    TimedOut,
}

/// The L1/§3 verdict for one compile+run (§6.7).
#[derive(Debug)]
pub(crate) enum Verdict {
    /// exit 0 ∧ stdout byte-equal, or exit 101 matching an oracle trap.
    Pass,
    /// A parity failure — an R1 data point against the backend (wrong exit
    /// class, mismatched stdout, timeout, or any other exit).
    Parity(String),
    /// Exit 102 — a harness-visible infra failure (box/toolchain/emitter
    /// bug), NEVER an R1 data point (DESIGN §3). Fails the test, reported
    /// apart from parity so it can masquerade as neither parity nor trap.
    Infra(String),
}

/// Classify one native run against the oracle expectation (§6.7 rules).
/// `want` is `(None, 101)` for an oracle trap (stdout ignored) or
/// `(Some(stdout), 0)` for a Done program. Exit 102 is checked FIRST — it is
/// infra regardless of the expected class.
pub(crate) fn classify(
    tag: &str,
    want_out: &Option<String>,
    want_code: i32,
    got: &RunOut,
) -> Verdict {
    let RunOut::Exited {
        code,
        stdout,
        stderr,
    } = got
    else {
        return Verdict::Parity(format!("{tag}: native run timed out ({TIMEOUT_SECS}s)"));
    };
    if *code == 102 {
        return Verdict::Infra(format!(
            "{tag}: exit 102 — CUDA infra failure (box/toolchain/emitter bug, \
             never an R1 data point); stderr: {}",
            String::from_utf8_lossy(stderr)
        ));
    }
    if *code != want_code {
        return Verdict::Parity(format!(
            "{tag}: exit {code} != oracle {want_code} (trap/done classes never cross); \
             stderr: {}",
            String::from_utf8_lossy(stderr)
        ));
    }
    if let Some(want) = want_out
        && stdout.as_slice() != want.as_bytes()
    {
        let got_text = String::from_utf8_lossy(stdout);
        return Verdict::Parity(format!("{tag}: stdout {got_text:?} != oracle {want:?}"));
    }
    Verdict::Pass
}

/// Compile `.cu` + libmapal_rt.a per the §4/§6.6 recipe and run, time-boxed
/// (120 s compile, 15 s run). Panics — naming the program, with stderr and
/// the full source — on nvcc failure or compile timeout (an emitter bug or a
/// broken box; llvm's clang-failure panic carried over).
/// Bytes one test program's CUDA context and buffers are budgeted on the GPU.
pub(crate) const GPU_BYTES_PER_RUN: u64 = 512 << 20;

/// How many compiled programs may run on the GPU at once: `MAPAL_GPU_JOBS`
/// if set, else the GPU's free memory (queried once) over
/// [`GPU_BYTES_PER_RUN`], at least 1. Fanning the runs out to every CPU
/// (64 on a Threadripper) exhausts a 16 GB card at context creation —
/// `cudaMalloc(d_trap): out of memory` on every program, an infra failure,
/// never a parity one.
pub(crate) fn gpu_jobs() -> usize {
    static JOBS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *JOBS.get_or_init(|| {
        if let Some(n) = std::env::var("MAPAL_GPU_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            return std::cmp::max(n, 1);
        }
        let free_mib = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().next().and_then(|l| l.trim().parse::<u64>().ok()));
        match free_mib {
            Some(m) => std::cmp::max(((m << 20) / GPU_BYTES_PER_RUN) as usize, 1),
            None => 4,
        }
    })
}

/// A held GPU run slot; released on drop.
pub(crate) struct GpuSlot;

pub(crate) static GPU_SLOTS: (Mutex<usize>, std::sync::Condvar) =
    (Mutex::new(0), std::sync::Condvar::new());

pub(crate) fn gpu_slot() -> GpuSlot {
    let (lock, cv) = &GPU_SLOTS;
    let mut busy = lock.lock().unwrap();
    while *busy >= gpu_jobs() {
        busy = cv.wait(busy).unwrap();
    }
    *busy += 1;
    GpuSlot
}

impl Drop for GpuSlot {
    fn drop(&mut self) {
        let (lock, cv) = &GPU_SLOTS;
        *lock.lock().unwrap() -= 1;
        cv.notify_one();
    }
}

pub(crate) fn compile_run(nvcc: &str, cu: &str, tag: &str) -> RunOut {
    let dir = tempfile::tempdir().unwrap();
    let cup = dir.path().join("p.cu");
    let exe = dir.path().join("p");
    std::fs::write(&cup, cu).unwrap();
    let argv = nvcc_argv(&cup, &rt_lib(), &exe);
    let cc = run_timeboxed(
        Command::new(nvcc).args(&argv),
        dir.path(),
        "nvcc",
        COMPILE_TIMEOUT_SECS,
    );
    match cc.code {
        Some(0) => {}
        Some(c) => panic!(
            "{tag}: nvcc exit {c}:\n{}\n---\n{cu}",
            String::from_utf8_lossy(&cc.stderr)
        ),
        None => panic!("{tag}: nvcc timed out ({COMPILE_TIMEOUT_SECS}s)"),
    }
    // Compiling is CPU work and fans out to every core; running holds a CUDA
    // context on the one GPU, so the runs at once are bounded by its memory.
    let _slot = gpu_slot();
    match run_timeboxed(&mut Command::new(&exe), dir.path(), "run", TIMEOUT_SECS) {
        ProcOut {
            code: Some(code),
            stdout,
            stderr,
        } => RunOut::Exited {
            code,
            stdout,
            stderr,
        },
        ProcOut { code: None, .. } => RunOut::TimedOut,
    }
}

// --- oracle expectation ---------------------------------------------------

/// The expected native observable from the interp oracle (L1 / BL8, ported
/// verbatim). Returns `None` for a Diverged run (skip). `Some((None, 101))`
/// for a trap (stdout ignored). `Some((Some(stdout), 0))` for a Done program —
/// the entry protocol decides the observable: an `IoToken` entry's stdout is
/// the token log; a closed `Unit → scalar` entry's is the wrapper-printed
/// return.
pub(crate) fn expect_native(ir: &CategoryIr, rr: &RunResult) -> Option<(Option<String>, i32)> {
    match &rr.outcome {
        Outcome::Diverged => None,
        Outcome::Trapped(_) => Some((None, 101)),
        Outcome::ReadFailed(_) => Some((None, mapal_ir::read::EXIT_CODE)),
        Outcome::Done(v) => {
            let entry = ir.entry();
            let in_ty = &ir.object(ir.func(entry).unwrap().input).unwrap().ty;
            let stdout = if *in_ty == Ty::IoToken {
                rr.output.clone()
            } else if let RValue::Scalar(sv) = v {
                // The main wrapper prints the scalar return with a newline.
                format!("{}\n", render(sv))
            } else {
                String::new() // non-printable return: wrapper prints nothing
            };
            Some((Some(stdout), 0))
        }
    }
}

/// Emit `ir`, nvcc-compile+run, and assert L1/§3 parity against `rr`
/// (Diverged ⇒ skip). Parity and infra verdicts both fail the test naming
/// the program; the verdict text says which class fired.
pub(crate) fn assert_parity(nvcc: &str, ir: &CategoryIr, rr: &RunResult, tag: &str) {
    let Some((want_out, want_code)) = expect_native(ir, rr) else {
        return; // diverged — skip
    };
    let cu = mapal_backend_cuda::emit(ir).unwrap_or_else(|e| panic!("{tag}: emit failed: {e:?}"));
    match classify(tag, &want_out, want_code, &compile_run(nvcc, &cu, tag)) {
        Verdict::Pass => {}
        Verdict::Parity(m) | Verdict::Infra(m) => panic!("{m}"),
    }
}

pub(crate) fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

pub(crate) fn build_example(name: &str) -> CategoryIr {
    let path = format!(
        "{}/../../../examples/{}.mapal",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let src = std::fs::read_to_string(&path).unwrap();
    lower_src(&src)
}
