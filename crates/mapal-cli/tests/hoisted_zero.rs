//! full-model: the hoisted zero × merged launch regression
//! (`crates/backends/llvm/tests/horizontal.rs` `a_hoisted_zero_waits_for_the_merged_launch` checks it in every
//! emulator mode) on the device itself, through `mapal run`. Opt-in, `MAPAL_GPU_LIVE=1` (a GPU and CUDA,
//! `MAPAL_CUDA`): the program's values on the device equal the oracle's. Without the fix the rotated query and
//! everything computed from it printed 0 on the RTX 5080.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Once;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
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
        .env("MAPAL_RT", rt_lib())
        .output()
        .expect("spawn mapal")
}

fn close(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter()
            .zip(&w)
            .all(|(a, b)| match (a.parse::<f64>(), b.parse::<f64>()) {
                (Ok(x), Ok(y)) => (x - y).abs() <= 1e-3 * x.abs().max(y.abs()).max(1.0),
                _ => a == b,
            })
}

#[test]
fn live_gpu_hoisted_zero_waits_for_the_merged_launch() {
    if std::env::var("MAPAL_GPU_LIVE").map_or(true, |v| v != "1") {
        eprintln!("SKIP: set MAPAL_GPU_LIVE=1 for the device leg");
        return;
    }
    let src = root().join("crates/backends/llvm/tests/fixtures/hoisted_zero_merged.mapal");
    let p = src.to_str().unwrap();
    let oracle = mapal(&["run", "--interp", "--budget", "20000000000", p]);
    assert_eq!(oracle.status.code(), Some(0));
    let want = String::from_utf8_lossy(&oracle.stdout).into_owned();
    for target in ["--target=native-gpu", "--target=rtx-5080"] {
        let got = mapal(&["run", target, "--splitk=2", p]);
        let out = String::from_utf8_lossy(&got.stdout).into_owned();
        assert!(
            got.status.code() == Some(0) && close(&out, &want),
            "{target}\n{out}\nwant\n{want}\n{}",
            String::from_utf8_lossy(&got.stderr)
        );
    }
}
