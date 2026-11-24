//! ADR-0047 end to end through the `mapal` binary: a size-generic program
//! (the user's `project` at two shapes) against the interpreter on the native
//! CPU (`mapal test`, byte for byte), and on the GPU (`--target=native-gpu`,
//! ignored by default: it needs an NVIDIA GPU and CUDA at `MAPAL_CUDA`) —
//! with its hand-specialized twin printing the same. Skip policy as
//! `examples.rs`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Once;

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
        .env("MAPAL_PAR", "2")
        .output()
        .expect("spawn mapal")
}

fn temp_src(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mapal-cli-generics-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.mapal"));
    std::fs::write(&p, src).unwrap();
    p
}

fn stdout(o: &Output, what: &str) -> String {
    assert_eq!(
        o.status.code(),
        Some(0),
        "{what}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

const GENERIC: &str =
    "fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> column_indices;
    rows -> iota -> map {
        row -> (0.0, column_indices) -> fold {
            sum, column -> weight[row * columns + column] -> widen_f32 -> w;
            sum + w * x[column]
        }
    } -> ret;
}

fn main() {
    131072 -> iota -> map {
        i -> i % 7 - 3 -> widen_f32 -> * 0.25 -> to_bf16
    } -> w;
    512 -> iota -> map {
        i -> i % 5 -> widen_f32 -> + 1.0
    } -> x;
    256 -> iota -> map {
        i -> i % 3 -> widen_f32
    } -> x2;
    (w, x) -> project -> y;
    (w, x2) -> project -> z;
    y[0] -> println;
    y[255] -> println;
    z[0] -> println;
    z[511] -> println;
}
";

/// [`GENERIC`] with each instance written out (`project<256, 512>`,
/// `project<512, 256>`).
fn hand() -> String {
    let body = |rows: u64, columns: u64| {
        format!(
            "fn project_{rows}_{columns}(weight: [bf16; {}], x: [f32; {columns}]) -> [f32; {rows}] {{
    {columns} -> iota -> column_indices;
    {rows} -> iota -> map {{
        row -> (0.0, column_indices) -> fold {{
            sum, column -> weight[row * {columns} + column] -> widen_f32 -> w;
            sum + w * x[column]
        }}
    }} -> ret;
}}
",
            rows * columns
        )
    };
    let main = &GENERIC[GENERIC.find("fn main").unwrap()..]
        .replacen("(w, x) -> project", "(w, x) -> project_256_512", 1)
        .replacen("(w, x2) -> project", "(w, x2) -> project_512_256", 1);
    format!("{}\n{}\n{main}", body(256, 512), body(512, 256))
}

#[test]
fn generic_program_matches_the_oracle_and_its_twin_natively() {
    if !clang_present() {
        return;
    }
    let (g, h) = (temp_src("generic", GENERIC), temp_src("hand", &hand()));
    let (g, h) = (g.to_str().unwrap(), h.to_str().unwrap());
    let want = stdout(&mapal(&["run", "--interp", h]), "twin interp");
    assert_eq!(stdout(&mapal(&["run", "--interp", g]), "interp"), want);
    let t = stdout(&mapal(&["test", g, "--target=generic"]), "mapal test");
    assert!(t.starts_with("PASS "), "{t}");
    assert_eq!(
        stdout(&mapal(&["run", g, "--target=generic"]), "native"),
        want
    );
    // The formatter keeps the declared size list as written.
    let f = stdout(&mapal(&["fmt", "--check", g]), "fmt --check");
    assert!(f.is_empty(), "{f}");
}

#[test]
#[ignore = "needs an NVIDIA GPU and CUDA (`MAPAL_CUDA`, default /usr/local/cuda)"]
fn generic_program_on_the_gpu() {
    let (g, h) = (
        temp_src("generic_gpu", GENERIC),
        temp_src("hand_gpu", &hand()),
    );
    let (g, h) = (g.to_str().unwrap(), h.to_str().unwrap());
    let want = stdout(&mapal(&["run", "--interp", g]), "interp");
    for p in [g, h] {
        assert_eq!(
            stdout(&mapal(&["run", p, "--target=native-gpu"]), "native-gpu"),
            want,
            "{p}"
        );
    }
}
