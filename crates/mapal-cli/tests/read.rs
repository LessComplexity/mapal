//! ADR-0045 through the `mapal` binary: a relative `read` path resolves
//! against the program's source directory, recorded at compile time — so a
//! built program run from any working directory finds its data, and
//! `$MAPAL_DATA_DIR` redirects it; failures exit 103 with one message on the
//! oracle and the compiled program.
//!
//! Two legs need the GPU and are opt-in (`benches/local/read_gpu.sh` runs
//! them under the bench lock): `MAPAL_GPU_LIVE=1` runs the golden and
//! failing programs on the device, `MAPAL_READ_GIB=1` reads a 1 GiB file and
//! checks the bytes it moves (one upload, none in the loop) with the
//! evidence tooling (`benches/gpu/evidence.py` `move_facts`) and the runtime
//! trace.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Once;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> PathBuf {
    root().join("crates/backends/llvm/tests/fixtures/read")
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

/// `mapal <args>` run in `cwd`.
fn mapal_in(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(args)
        .current_dir(cwd)
        .env("MAPAL_RT", rt_lib())
        .env_remove("MAPAL_DATA_DIR")
        .output()
        .expect("spawn mapal")
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// The golden program and its files, copied into a fresh directory.
fn golden_copy() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(fixtures()).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, dir.path().join(p.file_name().unwrap())).unwrap();
    }
    dir
}

fn golden_out() -> String {
    std::fs::read_to_string(fixtures().join("golden.out")).unwrap()
}

/// `mapal run --interp` from another working directory resolves the reads
/// against the source file's directory; `mapal test` holds the compiled
/// program to the oracle's bytes.
#[test]
fn oracle_and_test_resolve_against_the_source_directory() {
    let golden = fixtures().join("golden.mapal");
    let g = golden.to_str().unwrap();
    let o = mapal_in(Path::new("/"), &["run", "--interp", g]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), golden_out());
    if !clang_present() {
        return;
    }
    let o = mapal_in(Path::new("/"), &["test", g, "--target=generic"]);
    assert!(
        text(&o.stdout).starts_with("PASS "),
        "{}{}",
        text(&o.stdout),
        text(&o.stderr)
    );
}

/// A built program run from another directory finds its data (the source
/// directory is baked in); `$MAPAL_DATA_DIR` moves it; with neither the data
/// nor the override, the read fails with exit 103 naming the file.
#[test]
fn built_program_runs_from_another_directory() {
    if !clang_present() {
        return;
    }
    let src = golden_copy();
    let out = tempfile::tempdir().unwrap();
    let exe = out.path().join("golden");
    let prog = src.path().join("golden.mapal");
    let o = mapal_in(
        out.path(),
        &[
            "build",
            prog.to_str().unwrap(),
            "-o",
            exe.to_str().unwrap(),
            "--target=generic",
        ],
    );
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let run = |cwd: &Path, env: Option<&Path>| {
        let mut c = Command::new(&exe);
        c.current_dir(cwd).env_remove("MAPAL_DATA_DIR");
        if let Some(d) = env {
            c.env("MAPAL_DATA_DIR", d);
        }
        c.output().unwrap()
    };
    let o = run(Path::new("/"), None);
    assert_eq!((o.status.code(), text(&o.stdout)), (Some(0), golden_out()));
    // Move the data elsewhere: the override finds it, the baked path fails.
    let alt = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(src.path()).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "bin") {
            std::fs::rename(&p, alt.path().join(p.file_name().unwrap())).unwrap();
        }
    }
    let o = run(out.path(), Some(alt.path()));
    assert_eq!((o.status.code(), text(&o.stdout)), (Some(0), golden_out()));
    let o = run(alt.path(), None);
    assert_eq!(o.status.code(), Some(103));
    let canon = std::fs::canonicalize(src.path()).unwrap();
    assert_eq!(
        text(&o.stderr).trim_end(),
        format!(
            "mapal: read failed: {}: No such file or directory (os error 2)",
            canon.join("i32.bin").display()
        )
    );
}

/// A failing read through the CLI: the oracle and the compiled program exit
/// 103 with the same message; `mapal test` agrees.
#[test]
fn failure_both_ways() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("short.bin"), [1u8; 6]).unwrap();
    let p = dir.path().join("short.mapal");
    std::fs::write(
        &p,
        "fn main() {\n    \"short.bin\" -> read -> values: [i64; 1];\n    values[0] -> println;\n}\n",
    )
    .unwrap();
    let p = p.to_str().unwrap();
    let canon = std::fs::canonicalize(dir.path()).unwrap();
    let msg = format!(
        "mapal: read failed: {}: expected 8 bytes, found 6",
        canon.join("short.bin").display()
    );
    let o = mapal_in(Path::new("/"), &["run", "--interp", p]);
    assert_eq!(
        (o.status.code(), text(&o.stderr).trim_end()),
        (Some(103), msg.as_str())
    );
    if !clang_present() {
        return;
    }
    let o = mapal_in(Path::new("/"), &["run", p, "--target=generic"]);
    assert_eq!(
        (o.status.code(), text(&o.stderr).trim_end()),
        (Some(103), msg.as_str())
    );
    let o = mapal_in(Path::new("/"), &["test", p, "--target=generic"]);
    assert!(text(&o.stdout).starts_with("PASS "), "{}", text(&o.stdout));
}

/// `MAPAL_GPU_LIVE=1`: the golden program and a failing read on the GPU.
#[test]
fn live_gpu_golden_and_failure() {
    if std::env::var("MAPAL_GPU_LIVE").map_or(true, |v| v != "1") || !clang_present() {
        eprintln!("SKIP: set MAPAL_GPU_LIVE=1 (benches/local/read_gpu.sh) for the device legs");
        return;
    }
    let target = std::env::var("MAPAL_READ_TARGET").unwrap_or_else(|_| "rtx-5080".into());
    let t = format!("--target={target}");
    let src = golden_copy();
    let g = src.path().join("golden.mapal");
    let o = mapal_in(Path::new("/"), &["run", g.to_str().unwrap(), &t]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), golden_out());
    std::fs::write(src.path().join("f32.bin"), [0u8; 12]).unwrap();
    let o = mapal_in(Path::new("/"), &["run", g.to_str().unwrap(), &t]);
    assert_eq!(o.status.code(), Some(103));
    let canon = std::fs::canonicalize(src.path()).unwrap();
    assert!(
        text(&o.stderr).contains(&format!(
            "mapal: read failed: {}: expected 32 bytes, found 12",
            canon.join("f32.bin").display()
        )),
        "{}",
        text(&o.stderr)
    );
}

const GIB_N: u64 = 16384;

/// `MAPAL_READ_GIB=1`: a 1 GiB weight file (16384 × 16384 f32) read once and
/// used by every trip of a bracketed loop. The runtime moves the file's
/// bytes once into the host array (`mapal_read_trace`), the device's one
/// upload carries them (`mapal_gpu_trace_total`: two uploads — the weights
/// and the loop's start vector — at 1 and at 3 trips), and the host IR has
/// no upload in the loop (`evidence.py` `move_facts`: `h2d_n` per trip 0,
/// the run's `h2d_b` the file plus the start vector). Writes the measured
/// numbers to `$MAPAL_READ_GIB_OUT` when set.
#[test]
fn gib_read_moves_once_on_the_gpu() {
    if std::env::var("MAPAL_READ_GIB").map_or(true, |v| v != "1") || !clang_present() {
        eprintln!("SKIP: set MAPAL_READ_GIB=1 (benches/local/read_gpu.sh) for the 1 GiB leg");
        return;
    }
    let target = std::env::var("MAPAL_READ_TARGET").unwrap_or_else(|_| "rtx-5080".into());
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let n = GIB_N;
    let w = |r: u64, k: u64| ((r * 7 + k * 3) % 13) as f32 / 64.0 - 6.0 / 64.0;
    let x0 = |k: u64| ((k * 5 + 1) % 11) as f32 / 8.0 - 5.0 / 8.0;
    {
        use std::io::Write as _;
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(dir.path().join("weights.bin")).unwrap());
        for r in 0..n {
            let row: Vec<u8> = (0..n).flat_map(|k| w(r, k).to_le_bytes()).collect();
            f.write_all(&row).unwrap();
        }
        let x: Vec<u8> = (0..n).flat_map(|k| x0(k).to_le_bytes()).collect();
        std::fs::write(dir.path().join("start.bin"), x).unwrap();
    }
    let mut report = String::new();
    let mut moved = Vec::new();
    for trips in [1u32, 3] {
        let src = format!(
            "fn main() {{
    \"weights.bin\" -> read -> weights: [f32; {nn}];
    \"start.bin\" -> read -> start: [f32; {n}];
    {n} -> iota -> rows;
    mut x: [f32; {n}] <- start;
    mut step: i32 <- 0;
    () -> time -> t0;
    loop {{
        (step < {trips}) -> {{
            -true-> {{
                rows -> map {{
                    r -> (0.0, rows) -> fold {{
                        acc, k -> acc + weights[r * {n} + k] * x[k]
                    }}
                }} -> y;
                y -> map {{
                    v -> v / 1024.0
                }} -> x;
                step + 1 -> step;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    () -> time -> t1;
    out[0] -> println;
    out[{last}] -> println;
    t1 - t0 -> println;
}}
",
            nn = n * n,
            last = n - 1
        );
        let p = dir.path().join(format!("gib{trips}.mapal"));
        std::fs::write(&p, src).unwrap();
        let exe = dir.path().join(format!("gib{trips}"));
        let o = mapal_in(
            dir.path(),
            &[
                "build",
                p.to_str().unwrap(),
                "-o",
                exe.to_str().unwrap(),
                &format!("--target={target}"),
            ],
        );
        assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
        let o = Command::new(&exe)
            .current_dir("/")
            .env("MAPAL_PTX", exe.with_extension("ptx"))
            .env("MAPAL_GPU_TRACE", "1")
            .output()
            .unwrap();
        let err = text(&o.stderr);
        assert_eq!(o.status.code(), Some(0), "{err}");
        // The values: f64 on the host, within the f32 fold bound.
        let mut x: Vec<f64> = (0..n).map(|k| f64::from(x0(k))).collect();
        for _ in 0..trips {
            x = (0..n)
                .map(|r| {
                    (0..n)
                        .map(|k| f64::from(w(r, k)) * x[k as usize])
                        .sum::<f64>()
                        / 1024.0
                })
                .collect();
        }
        let got: Vec<f64> = text(&o.stdout)
            .lines()
            .map(|l| l.trim().parse().unwrap())
            .collect();
        for (g, want) in got.iter().zip([x[0], x[(n - 1) as usize]]) {
            assert!(
                (g - want).abs() <= 1e-3 * want.abs().max(1e-6),
                "{g} vs {want}"
            );
        }
        let field = |line: &str, key: &str| -> f64 {
            line.split_whitespace()
                .find_map(|f| f.strip_prefix(&format!("{key}=")))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{key} in {line}"))
        };
        let reads: Vec<&str> = err
            .lines()
            .filter(|l| l.starts_with("mapal_read_trace "))
            .collect();
        assert_eq!(reads.len(), 2, "{err}");
        assert_eq!(field(reads[0], "bytes") as u64, n * n * 4);
        let total = err
            .lines()
            .find(|l| l.starts_with("mapal_gpu_trace_total "))
            .expect("trace total");
        let (h2d, bytes_in) = (field(total, "h2d") as u64, field(total, "bytes_in") as u64);
        assert_eq!(bytes_in, n * n * 4 + n * 4, "{total}");
        assert_eq!(h2d, 2, "{total}");
        // The host IR, through the evidence tooling.
        let facts = Command::new("python3")
            .arg("-c")
            .arg(
                "import json,sys; sys.path.insert(0, sys.argv[1]); from evidence import move_facts; \
                 print(json.dumps(move_facts(open(sys.argv[2]).read(), [int(sys.argv[3])])))",
            )
            .arg(root().join("benches/gpu"))
            .arg(exe.with_extension("ll"))
            .arg(trips.to_string())
            .output()
            .unwrap();
        let facts = text(&facts.stdout);
        let num = |key: &str| -> u64 {
            facts
                .split(&format!("\"{key}\": "))
                .nth(1)
                .and_then(|r| r.split([',', '}']).next())
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("{key} in {facts}"))
        };
        assert_eq!(num("h2d_n"), 0, "an upload inside the loop: {facts}");
        assert_eq!(num("runtime_h2d_b"), n * n * 4 + n * 4, "{facts}");
        let ms = |key: &str| field(total, key);
        let read_ms = field(reads[0], "ms");
        let loop_ms: f64 = got[2];
        moved.push((h2d, bytes_in));
        report += &format!(
            "trips={trips} read_ms={read_ms:.1} h2d_ms={:.1} h2d={h2d} bytes_in={bytes_in} loop_ms={loop_ms:.3} ir_h2d_per_trip={} ir_runtime_h2d_b={}\n",
            ms("h2d_ms"),
            num("h2d_n"),
            num("runtime_h2d_b"),
        );
    }
    assert_eq!(moved[0], moved[1], "uploads grow with the trips");
    eprint!("{report}");
    if let Ok(out) = std::env::var("MAPAL_READ_GIB_OUT") {
        std::fs::write(out, report).unwrap();
    }
}
