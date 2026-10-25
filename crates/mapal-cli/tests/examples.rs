//! The `mapal` binary end to end. Every test fails on the pre-S62 stub, which
//! exited 1 for every invocation. Targets are pinned by name (ADR-0038 D4).
//! Skip policy as `crates/backends/llvm/tests/differential.rs`: no clang ⇒
//! skip with a reason, and `MAPAL_REQUIRE_CLANG` turns that into a failure.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Once;

/// The examples whose oracle finishes and which emit. Absent: `matmul128`
/// and `matmul4096` (the interpreter does not finish them inside the default
/// budget in a debug build) and `vector` (P0001 — not Mapal-Core).
const EXAMPLES: &[&str] = &[
    "abs",
    "calc",
    "fanout",
    "fib",
    "fir",
    "matmul4_loop",
    "pipeline",
    "sepia",
    "seq_demo",
    "sum_to_n",
    "vector_add",
    "zip_demo",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn clang_present() -> bool {
    let found = std::env::var("CC").is_ok()
        || Command::new("which")
            .arg("clang")
            .output()
            .is_ok_and(|o| o.status.success());
    assert!(
        found || !std::env::var("MAPAL_REQUIRE_CLANG").is_ok_and(|v| v != "0"),
        "MAPAL_REQUIRE_CLANG is set but clang was not found — this test would skip"
    );
    if !found {
        eprintln!("SKIP: clang not found (set CC or put clang on PATH)");
    }
    found
}

/// `libmapal_rt.a`, built once (cargo uplifts a staticlib only as a root).
fn rt_lib() -> PathBuf {
    static BUILD: Once = Once::new();
    BUILD.call_once(|| {
        let ok = Command::new(env!("CARGO"))
            .args(["build", "-p", "mapal-rt"])
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "cargo build -p mapal-rt failed");
    });
    root().join("target/debug/libmapal_rt.a")
}

fn mapal(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(args)
        .current_dir(root())
        .env("MAPAL_RT", rt_lib())
        .output()
        .expect("spawn mapal")
}

fn temp_src(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mapal-cli-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.mapal"));
    std::fs::write(&p, src).unwrap();
    p
}

#[test]
fn examples_compiled_match_interp() {
    if !clang_present() {
        return;
    }
    for n in EXAMPLES {
        let f = format!("examples/{n}.mapal");
        let o = mapal(&["test", &f, "--target=generic"]);
        let out = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            o.status.code(),
            Some(0),
            "{f}: {out}{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(out.starts_with("PASS "), "{f}: {out}");
    }
}

#[test]
fn trap_exit_code_matches_both_ways() {
    if !clang_present() {
        return;
    }
    let p = temp_src(
        "trap",
        "fn f(a: i32, b: i32) -> i32 { a / b -> ret; }\nfn main() { (1, 0) -> f -> r; r -> println; }\n",
    );
    let p = p.to_str().unwrap();
    assert_eq!(mapal(&["run", "--interp", p]).status.code(), Some(101));
    assert_eq!(
        mapal(&["run", p, "--target=generic"]).status.code(),
        Some(101)
    );
    assert_eq!(
        mapal(&["test", p, "--target=generic"]).status.code(),
        Some(0)
    );
}

#[test]
fn dump_ir_emits_translation_unit() {
    let o = mapal(&["dump-ir", "examples/abs.mapal", "--target=generic"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stdout).contains("define i32 @main()"));
}

/// One offloadable tile site (64×64×64, f32) — `backends/llvm/tests/gpu_host.rs`.
const SITE: &str = r#"fn main() {
    4096 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 7.0 } -> a;
    ta -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 9.0 } -> b;
    64 -> iota -> krange;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c;
    c[4095] -> println;
}
"#;

/// A GPU profile writes the device unit next to the host unit, and the PTX's
/// `.target` is the profile's `arch` fact (rtx-4090 → sm_89). No CUDA, no run.
#[test]
fn gpu_profile_build_writes_device_unit() {
    let src = temp_src("site", SITE);
    let o = mapal(&[
        "build",
        src.to_str().unwrap(),
        "--target=rtx-4090",
        // S62 `place` keeps a 64³ site on the host (the 4090's measured
        // costs: the host finishes it sooner); its OFF arm launches every
        // capable site, which is the device unit this test reads.
        "--gpu-off=place",
        "--emit-only",
    ]);
    let stem = src.with_extension("");
    let read = |ext: &str| std::fs::read_to_string(format!("{}{ext}", stem.display()));
    let dev = read(".dev.ll").expect("device unit written");
    assert!(
        dev.contains("mapal_tile_"),
        "a tile kernel in the device unit"
    );
    assert!(read(".ll").unwrap().contains("@mapal_gpu_init"));
    let llc = std::env::var("LLVM")
        .map(|d| Path::new(&d).join("llc").exists())
        .unwrap_or(false)
        || Command::new("which")
            .arg("llc")
            .output()
            .is_ok_and(|o| o.status.success());
    if llc {
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(read(".ptx").unwrap().contains(".target sm_89"));
    } else {
        eprintln!("SKIP ptx: llc not found (set LLVM)");
        assert_eq!(
            o.status.code(),
            Some(3),
            "a missing llc is a toolchain exit"
        );
    }
}

/// ADR-0040: `mapal test` judges the bit-replayable faces only — the default
/// (forced to the fixed face), `--exact`, `--no-contract --order=fixed`; the
/// default face's own flags (`--contract`, `--order=free`), the contract face
/// (`--order=sequential` with fma steps) and TF32 are bound-checked
/// elsewhere, so they are refused (exit 2) before anything runs.
#[test]
fn test_judges_the_replayable_faces() {
    if !clang_present() {
        return;
    }
    let f = "examples/fir.mapal";
    for ok in [
        &[][..],
        &["--exact"],
        &["--no-contract", "--order=fixed"],
        &["--no-contract", "--order=sequential"],
    ] {
        let mut a = vec!["test", f, "--target=generic"];
        a.extend_from_slice(ok);
        let o = mapal(&a);
        assert_eq!(o.status.code(), Some(0), "{ok:?}: {}", String::from_utf8_lossy(&o.stdout));
    }
    for refused in [
        &["--contract"][..],
        &["--order=free"],
        &["--order=sequential"],
    ] {
        let mut a = vec!["test", f, "--target=generic"];
        a.extend_from_slice(refused);
        let o = mapal(&a);
        assert_eq!(o.status.code(), Some(2), "{refused:?}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("fixed face"));
    }
    // ADR-0041 D7: the flag is gone; the program carries `: tf32 | f32`.
    let o = mapal(&["test", f, "--precision=tf32"]);
    assert_eq!(o.status.code(), Some(2), "--precision=tf32 is refused");
    assert!(String::from_utf8_lossy(&o.stderr).contains("ADR-0041"));
    let o = mapal(&["run", f, "--order=sideways"]);
    assert_eq!(o.status.code(), Some(2), "a malformed order is an error");
}
