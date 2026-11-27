//! The clang toolchain, compile/run plumbing and the oracle-parity assertion every differential test shares.

use super::*;

// --- toolchain ------------------------------------------------------------

pub(crate) fn clang() -> Option<String> {
    // Respect CC, else find clang on PATH.
    if let Ok(cc) = std::env::var("CC") {
        return Some(cc);
    }
    let found = Command::new("which")
        .arg("clang")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    // `MAPAL_REQUIRE_CLANG` turns skip-with-reason into a hard failure. Every one of
    // this file's skip sites goes through here, so the guard belongs here and
    // nowhere else. CI sets it: a suite that skips itself proves nothing, and the
    // workflow used to prove otherwise by running the whole 1,280-case differential a
    // SECOND time with `--nocapture` and grepping for "skip" — ~25 min on a two-core
    // Linux runner, which hit the 60-minute job timeout on the first run that got far
    // enough to reach it. Failing here makes the FIRST run its own proof.
    assert!(
        !(found.is_none() && std::env::var("MAPAL_REQUIRE_CLANG").is_ok_and(|v| v != "0")),
        "MAPAL_REQUIRE_CLANG is set but clang was not found (CC unset, `which clang` \
         failed) — this suite would have skipped and reported success without \
         compiling anything"
    );
    found
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

/// Compile `.ll` + libmapal_rt.a at optimization level `opt` and run, time-boxed.
/// `None` on a timeout (the harness fails loudly rather than hanging). Panics if
/// clang errors.
pub(crate) fn compile_run(clang: &str, ll: &str, tag: &str, opt: &str) -> Option<(Vec<u8>, i32)> {
    let (_dir, exe) = compile_exe(clang, ll, tag, opt);
    run_exe(&exe, TIMEOUT_SECS, None)
}

pub(crate) fn compile_exe(
    clang: &str,
    ll: &str,
    tag: &str,
    opt: &str,
) -> (tempfile::TempDir, PathBuf) {
    try_compile_exe(clang, ll, opt)
        .unwrap_or_else(|e| panic!("{tag}: clang failed:\n{e}\n---\n{ll}"))
}

/// [`compile_exe`] without the panic — `Err(stderr)` when clang rejects the
/// module. The batched path uses this so a batch that fails to build can be
/// retried case-by-case and blamed on the one module responsible.
pub(crate) fn try_compile_exe(
    clang: &str,
    ll: &str,
    opt: &str,
) -> Result<(tempfile::TempDir, PathBuf), String> {
    let dir = tempfile::tempdir().unwrap();
    let llp = dir.path().join("p.ll");
    let exe = dir.path().join("p");
    std::fs::write(&llp, ll).unwrap();
    let mut cmd = Command::new(clang);
    cmd.arg(opt).arg(&llp).arg(rt_lib()).arg("-o").arg(&exe);
    // Linking dominates a case, and this suite does ~1,280 of them per run.
    // Measured on one representative module (min of 3): compile+link 0.08 s,
    // compile alone 0.02 s, link alone 0.05 s — so ~60% of the cost is the
    // linker, and Linux's default GNU ld is the slower one. CI sets MAPAL_LD=lld
    // there. Unset (the local default) ⇒ platform default linker, so nobody has
    // to install anything to run the suite.
    if let Ok(ld) = std::env::var("MAPAL_LD")
        && !ld.is_empty()
    {
        cmd.arg(format!("-fuse-ld={ld}"));
    }
    let out = {
        let _slot = slot();
        cmd.output().unwrap()
    };
    if out.status.success() {
        Ok((dir, exe))
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

pub(crate) fn run_exe(exe: &Path, secs: u64, mapal_par: Option<&str>) -> Option<(Vec<u8>, i32)> {
    let mut command = Command::new(exe);
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .env_remove("MAPAL_PAR")
        // Values only: no host spin while other children share the cores.
        .env("MAPAL_SPIN", "0");
    if let Some(value) = mapal_par {
        command.env("MAPAL_PAR", value);
    }
    wait_boxed(command, secs)
}

/// Run case `slot` out of a batched executable (`exe <slot>`). Same
/// one-process-per-case isolation as [`run_exe`]: a trapping case still exits
/// 101, a printing case still owns the whole stdout.
pub(crate) fn run_case(exe: &Path, secs: u64, slot: usize) -> Option<(Vec<u8>, i32)> {
    let mut command = Command::new(exe);
    command
        .arg(slot.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .env("MAPAL_PAR", SWEEP_LANES)
        .env("MAPAL_SPIN", "0");
    wait_boxed(command, secs)
}

/// Spawn and wait, time-boxed. `None` on timeout (the caller fails loudly
/// rather than hanging).
pub(crate) fn wait_boxed(mut command: Command, secs: u64) -> Option<(Vec<u8>, i32)> {
    let _slot = slot();
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            let mut buf = Vec::new();
            child.stdout.take().unwrap().read_to_end(&mut buf).unwrap();
            // Killed by a signal: the shell's `128 + signal`.
            let code = status.code().unwrap_or_else(|| {
                use std::os::unix::process::ExitStatusExt;
                128 + status.signal().unwrap_or(0)
            });
            return Some((buf, code));
        }
        if start.elapsed() > Duration::from_secs(secs) {
            let _ = child.kill();
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

// --- oracle expectation ---------------------------------------------------

/// Emit `ir`, compile+run at **both `-O0` and `-O2`**, and assert L1 parity
/// against `rr` at each level — the same oracle expectations (DESIGN §8 `-O2`
/// row: optimization is where accidentally-relied-on LLVM-level UB surfaces).
pub(crate) fn assert_parity(clang: &str, ir: &CategoryIr, rr: &RunResult, tag: &str) {
    let Some((want_out, want_code)) = expect_native(ir, rr) else {
        return; // diverged — skip
    };
    // ADR-0038 D4: value legs pin `generic` by name too, for uniformity — the
    // ONE ambient/native leg in this file is
    // differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap_or_else(|e| panic!("{tag}: emit failed: {e:?}"));
    for opt in ["-O0", "-O2"] {
        let tag_o = format!("{tag}/{opt}");
        let (got_out, got_code) = compile_run(clang, &ll, &tag_o, opt)
            .unwrap_or_else(|| panic!("{tag_o}: native run timed out ({TIMEOUT_SECS}s)"));
        assert_eq!(got_code, want_code, "{tag_o}: exit code");
        if let Some(want) = &want_out {
            assert_eq!(
                String::from_utf8_lossy(&got_out),
                *want,
                "{tag_o}: stdout mismatch"
            );
        }
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
