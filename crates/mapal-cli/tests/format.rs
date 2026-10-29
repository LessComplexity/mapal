//! Formatting is a source operation: no LLVM, runtime, or device is needed.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

const COMPACT: &str = "fn main(){1->println;}";
const FORMATTED: &str = "fn main() {\n    1 -> println;\n}\n";

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mapal-formatter-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cli(args: &[&str], source: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(args)
        .env("CC", "/nonexistent-mapal-test-compiler")
        .env("LLVM", "/nonexistent-mapal-test-llvm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    if let Some(source) = source {
        stdin.write_all(source.as_bytes()).unwrap();
    }
    drop(stdin);
    child.wait_with_output().unwrap()
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn stdin_formats_without_toolchain_and_check_detects_drift() {
    let output = cli(&["fmt", "-"], Some(COMPACT));
    success(&output);
    assert_eq!(output.stdout, FORMATTED.as_bytes());
    assert!(output.stderr.is_empty());

    let drift = cli(&["fmt", "--check", "-"], Some(COMPACT));
    assert_eq!(drift.status.code(), Some(1));
    assert!(drift.stdout.is_empty());
    assert!(String::from_utf8_lossy(&drift.stderr).contains("needs formatting"));
    success(&cli(&["fmt", "--check", "-"], Some(FORMATTED)));
}

#[test]
fn preview_check_and_in_place_share_output_and_preserve_files_when_read_only() {
    let dir = Directory::new();
    let source = dir.file("file with spaces.mapal", COMPACT);
    let preview = cli(&["fmt", "--stdout", path(&source)], None);
    success(&preview);
    assert_eq!(preview.stdout, FORMATTED.as_bytes());
    assert_eq!(fs::read_to_string(&source).unwrap(), COMPACT);
    let drift = cli(&["fmt", "--check", path(&source)], None);
    assert_eq!(drift.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&source).unwrap(), COMPACT);

    let written = cli(&["fmt", path(&source)], None);
    success(&written);
    assert!(written.stdout.is_empty());
    assert_eq!(fs::read_to_string(&source).unwrap(), FORMATTED);
    let modified = fs::metadata(&source).unwrap().modified().unwrap();
    success(&cli(&["fmt", path(&source)], None));
    assert_eq!(fs::metadata(&source).unwrap().modified().unwrap(), modified);
    success(&cli(&["fmt", "--check", path(&source)], None));
}

#[test]
fn invalid_or_unreadable_batch_input_never_writes_valid_siblings() {
    let dir = Directory::new();
    let valid = dir.file("valid.mapal", COMPACT);
    let invalid = dir.file("invalid.mapal", "fn main(){");
    for failing in [&invalid, &dir.0.join("missing.mapal")] {
        let output = cli(&["fmt", path(&valid), path(failing)], None);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read_to_string(&valid).unwrap(), COMPACT);
    }
    assert_eq!(fs::read_to_string(invalid).unwrap(), "fn main(){");
}

#[test]
fn invalid_stdin_has_diagnostics_but_no_partial_output() {
    let output = cli(&["fmt", "-"], Some("fn main(){"));
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostic.contains("-:1:"), "{diagnostic}");
    assert!(diagnostic.contains("error[P"), "{diagnostic}");
}

#[test]
fn explicit_batch_and_end_of_options_work() {
    let dir = Directory::new();
    let a = dir.file("a.mapal", COMPACT);
    let b = dir.file("b.mapal", COMPACT);
    success(&cli(&["fmt", "--", path(&a), path(&b), path(&a)], None));
    assert_eq!(fs::read_to_string(a).unwrap(), FORMATTED);
    assert_eq!(fs::read_to_string(b).unwrap(), FORMATTED);
}

#[test]
fn help_and_usage_are_separate_from_compile_flags() {
    let help = cli(&["fmt", "--help"], None);
    success(&help);
    assert!(String::from_utf8_lossy(&help.stdout).contains("--stdout"));
    for args in [
        vec!["fmt"],
        vec!["fmt", "--check", "--stdout", "-"],
        vec!["fmt", "--stdout", "a.mapal", "b.mapal"],
        vec!["fmt", "-", "a.mapal"],
        vec!["fmt", "--target=generic", "-"],
    ] {
        let output = cli(&args, None);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn original_interpreter_dispatch_still_works() {
    let dir = Directory::new();
    let source = dir.file("run.mapal", FORMATTED);
    let output = cli(&["run", "--interp", path(&source)], None);
    success(&output);
    assert_eq!(output.stdout, b"1\n");
}

#[cfg(unix)]
#[test]
fn atomic_replacement_preserves_permissions_and_symlinks() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let dir = Directory::new();
    let source = dir.file("source.mapal", COMPACT);
    fs::set_permissions(&source, fs::Permissions::from_mode(0o640)).unwrap();
    let link = dir.0.join("link.mapal");
    symlink(&source, &link).unwrap();
    success(&cli(&["fmt", path(&link), path(&source)], None));
    assert!(fs::symlink_metadata(link).unwrap().is_symlink());
    assert_eq!(fs::read_to_string(&source).unwrap(), FORMATTED);
    assert_eq!(
        fs::metadata(source).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
}
