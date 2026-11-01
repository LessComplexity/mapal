//! The GPU emulator harness the integration tests share (S54 `retain_delta`,
//! S62 `splitk`): the device unit compiled for the host and run thread by
//! thread through `crates/mapal-rt/gpu/mapal_gpu_emu.c`.

use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Child processes (clang, test binaries) this test binary runs at once, across
/// all of its tests: half the cores left after the two CLAUDE.md keeps free
/// (a child runs a 2-lane pool), or `MAPAL_TEST_JOBS`.
///
/// Unbounded, each sweep ran `available_parallelism` workers, each starting a
/// binary whose runtime pool was itself `available_parallelism` wide: 14 × 14
/// runnable threads on 14 cores — the runtime's own oversubscription cliff
/// (`mapal-rt` `spin_iters`, 1500x at 14 executors) — so programs ran into the
/// 15 s timeout with every core pegged and the machine unusable.
pub fn job_slots() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MAPAL_TEST_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
                (cores.saturating_sub(2) / 2).max(1)
            })
    })
}

/// Children in flight, bounded by [`job_slots`].
static IN_FLIGHT: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());

/// Hold one of [`job_slots`] while a child process runs.
pub struct Slot;

pub fn slot() -> Slot {
    let (m, cv) = &IN_FLIGHT;
    let guard = m.lock().unwrap_or_else(PoisonError::into_inner);
    let mut used = cv
        .wait_while(guard, |n| *n >= job_slots())
        .unwrap_or_else(PoisonError::into_inner);
    *used += 1;
    Slot
}

impl Drop for Slot {
    fn drop(&mut self) {
        let (m, cv) = &IN_FLIGHT;
        *m.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        cv.notify_one();
    }
}

/// The device unit compiled for the host: special registers become the
/// emulator's `emu_*` (`mapal_gpu_emu.c`), global and shared pointers generic
/// (shared memory is `emu_smem`, `barrier0` is `emu_barrier`), the PTX-only
/// inline asm (`griddepcontrol`, the raster's opaque `mov`) dropped or an
/// identity, and an atomic's device scope the host's. `cp.async` is not
/// emulated: tests that reach a staged kernel turn `async` off.
pub fn host_device(dev: &str) -> String {
    let mut out = String::new();
    for line in dev.lines() {
        let nvvm_decl = line.starts_with("declare")
            && (line.contains("@llvm.nvvm.") || line.contains("cp.async"));
        if line.starts_with("target ")
            || line.starts_with("@mapal_smem")
            || nvvm_decl
            || line.contains("asm sideeffect \"griddepcontrol")
        {
            continue;
        }
        let mut l = line
            .replace("ptx_kernel ", "")
            .replace("addrspace(1) ", "")
            .replace("addrspace(3) ", "")
            .replace("@mapal_smem", "@emu_smem")
            .replace(
                "call void @llvm.nvvm.barrier0()",
                "call void @emu_barrier()",
            )
            .replace(" syncscope(\"device\")", "");
        for (ty, bits, c) in [("i32", 32, "r"), ("i64", 64, "l")] {
            let asm = format!("call {ty} asm \"mov.b{bits} $0, $1;\", \"={c},{c}\"({ty} ");
            if let Some(i) = l.find(&asm) {
                let v = l[i + asm.len()..].trim_end_matches(')').to_string();
                l = format!("{}add {ty} {v}, 0", &l[..i]);
            }
        }
        // A generic-to-global cast is an identity once both are generic.
        if let Some((lhs, rest)) = l.split_once(" = addrspacecast ptr ")
            && let Some(v) = rest.strip_suffix(" to ptr")
        {
            l = format!("{lhs} = getelementptr i8, ptr {v}, i64 0");
        }
        // `cell-lanes`: a warp's butterfly shuffle meets its 31 lanes in
        // the emulator (`emu_shfl_bfly`; the kernel runs threaded).
        if l.contains("@llvm.nvvm.shfl.sync.bfly.f32(i32 -1, ") {
            l = l
                .replace("@llvm.nvvm.shfl.sync.bfly.f32(i32 -1, ", "@emu_shfl_bfly(")
                .replace(", i32 31)", ")");
        }
        for (r, e) in [
            ("laneid", "laneid"),
            ("tid.x", "tid_x"),
            ("ctaid.x", "ctaid_x"),
            ("ctaid.y", "ctaid_y"),
        ] {
            l = l.replace(
                &format!("@llvm.nvvm.read.ptx.sreg.{r}()"),
                &format!("@emu_{e}()"),
            );
        }
        out.push_str(&l);
        out.push('\n');
    }
    // A module with shuffles runs its cell launches threaded.
    let lanes = if out.contains("@emu_shfl_bfly(") {
        "@emu_threaded = global i32 1\n"
    } else {
        ""
    };
    out + lanes
        + "declare i32 @emu_tid_x()\ndeclare i32 @emu_ctaid_x()\ndeclare i32 @emu_ctaid_y()\n\
           declare i32 @emu_laneid()\ndeclare float @emu_shfl_bfly(float, i32)\n\
           declare void @emu_barrier()\n@emu_smem = external global [0 x i8], align 16\n"
}

/// Emulation runs every device thread on the host, so it gets 8x the native timeout.
const EMU_TIMEOUT_SECS: u64 = 120;

/// Build and run `host` + `dev` against the emulated device; its stdout.
#[allow(dead_code)]
pub fn run_emulated(clang: &str, host: &str, dev: &str) -> String {
    run_emulated_totals(clang, host, dev).0
}

/// [`run_emulated`], and the run's device traffic (`emu_total`: uploads,
/// copy-outs, launches and their bytes) as `[h2d, d2h, launches, bytes_in,
/// bytes_out]`.
#[allow(dead_code)]
pub fn run_emulated_totals(clang: &str, host: &str, dev: &str) -> (String, [u64; 5]) {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    std::fs::write(p("h.ll"), host).unwrap();
    std::fs::write(p("d.ll"), host_device(dev)).unwrap();
    let emu = format!(
        "{}/../../mapal-rt/gpu/mapal_gpu_emu.c",
        env!("CARGO_MANIFEST_DIR")
    );
    let _slot = slot();
    let ok = Command::new(clang)
        .args(["-O1", "-Wno-override-module", "-c"])
        .arg(p("d.ll"))
        .arg("-o")
        .arg(p("d.o"))
        .status()
        .unwrap()
        .success();
    assert!(ok, "device unit does not compile for the host");
    let mut link = Command::new(clang);
    link.args(["-O1", "-Wno-override-module"])
        .arg(p("h.ll"))
        .arg(p("d.o"))
        .arg(&emu)
        .arg(rt_lib())
        .arg("-o")
        .arg(p("x"));
    if cfg!(target_os = "linux") {
        // The emulator finds kernels by name with `dlsym`.
        link.args(["-rdynamic", "-ldl", "-lm"]);
    }
    assert!(link.status().unwrap().success(), "link failed");
    // Time-boxed: an emulated program that never finishes fails the test
    // instead of pinning a core until someone kills it.
    let mut child = Command::new(p("x"))
        .env("MAPAL_PAR", "2")
        .env("MAPAL_SPIN", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(EMU_TIMEOUT_SECS) {
            let _ = child.kill();
            panic!("emulated run timed out after {EMU_TIMEOUT_SECS} s");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    let mut totals = [0u64; 5];
    if let Some(l) = err.lines().find_map(|l| l.strip_prefix("emu_total ")) {
        for (t, kv) in totals.iter_mut().zip(l.split_whitespace()) {
            *t = kv
                .split_once('=')
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (String::from_utf8_lossy(&out.stdout).into_owned(), totals)
}

pub fn clang() -> Option<String> {
    if let Ok(cc) = std::env::var("CC") {
        return Some(cc);
    }
    let found = Command::new("which")
        .arg("clang")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    assert!(
        !(found.is_none() && std::env::var("MAPAL_REQUIRE_CLANG").is_ok_and(|v| v != "0")),
        "MAPAL_REQUIRE_CLANG is set but clang was not found"
    );
    found
}

pub fn rt_lib() -> std::path::PathBuf {
    static BUILD: std::sync::Once = std::sync::Once::new();
    BUILD.call_once(|| {
        let ok = Command::new("cargo")
            .args(["build", "-p", "mapal-rt"])
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "cargo build -p mapal-rt failed");
    });
    format!(
        "{}/../../../target/debug/libmapal_rt.a",
        env!("CARGO_MANIFEST_DIR")
    )
    .into()
}
