//! `mapal repl` through the binary: a script piped on stdin (no prompts, no
//! banner), results on stdout, diagnostics on stderr. The interpreter is the
//! only engine, so no LLVM toolchain is needed.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// (stdout, stderr) of a piped session, run from the repository root.
fn session(script: &str) -> (String, String) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .arg("repl")
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "repl exited with {:?}", out.status);
    (
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[test]
fn bare_expression_prints_its_value() {
    let (out, err) = session("1 + 2\n");
    assert_eq!(out, "3\n");
    assert_eq!(err, "");
}

#[test]
fn binding_is_echoed_and_used_later() {
    let (out, _) = session("3 -> a\na * 2\n");
    assert_eq!(out, "a = 3\n6\n");
    // The semicolon form binds the same way, and an expression is not kept.
    let (out, _) = session("3 -> a;\n10 + 1\na\n");
    assert_eq!(out, "a = 3\n11\n3\n");
}

#[test]
fn statements_print_only_what_is_new() {
    let (out, _) = session("1 -> println;\n2 -> println;\n3 -> n\n");
    assert_eq!(out, "1\n2\nn = 3\n");
}

#[test]
fn multi_line_function_then_call_then_redefinition() {
    let script = "fn double(x: i32) -> i32 {\n    x * 2 -> ret;\n}\n5 -> double\n\
                  fn double(x: i32) -> i32 {\n    x * 3 -> ret;\n}\n5 -> double\n";
    let (out, err) = session(script);
    assert_eq!(out, "defined double\n10\ndefined double\n15\n");
    assert_eq!(err, "");
}

#[test]
fn a_bad_line_leaves_nothing_behind() {
    // Bad syntax, then an out-of-bounds index, then a name defined by the
    // failed lines: the session still holds only what succeeded.
    let (out, err) = session("1 -> keep\n5 ->\n[1, 2, 3][7] -> lost\nlost\nkeep + 1\n");
    assert_eq!(out, "keep = 1\n2\n");
    // Diagnostics point at what was typed (line 1 of the input, the caret under the gap).
    assert!(err.contains("input:1:5 error[P"), "{err}");
    assert!(err.contains("  | 5 ->\n  |     ^"), "{err}");
    assert!(err.contains("array index out of bounds"), "{err}");
    assert!(err.contains("unresolved name `lost`"), "{err}");
}

#[test]
fn a_type_error_points_into_the_input() {
    let (out, err) = session("fn f(x: i32) -> i32 {\n    x + true -> ret;\n}\n2 -> println;\n");
    assert_eq!(out, "2\n");
    assert!(err.contains("input:2:"), "{err}");
    assert!(err.contains("  |     x + true -> ret;"), "{err}");
}

#[test]
fn reset_forgets_everything() {
    let (out, err) = session("1 -> a\n:reset\na\n2 -> a\n");
    assert_eq!(out, "a = 1\nsession cleared\na = 2\n");
    assert!(err.contains("unresolved name `a`"), "{err}");
}

#[test]
fn load_then_call_a_loaded_function() {
    let (out, err) = session(":load examples/softmax.mapal\n[1.0, 2.0, 3.0, 4.0] -> softmax\n");
    assert_eq!(
        out,
        "loaded 1 item(s) from examples/softmax.mapal: softmax\n\
         [0.032058604, 0.08714432, 0.23688284, 0.6439143]\n"
    );
    assert_eq!(err, "");
}

#[test]
fn list_is_a_formatted_program() {
    let (out, _) =
        session("fn  inc(x:i32)->i32{x+1->ret;}\n4 -> inc -> five\n five -> println\n:list\n");
    let listing = out
        .split_once("defined inc\nfive = 5\n5\n")
        .expect("the session's own output comes first")
        .1
        .to_string();
    assert!(listing.contains("fn main() {"), "{listing}");
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("session.mapal");
    std::fs::write(&file, &listing).unwrap();
    let check = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(["fmt", "--check"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}\n{listing}",
        String::from_utf8_lossy(&check.stderr)
    );
    // And it is a program: it runs on the interpreter and prints what the session did.
    let run = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(["run", "--interp"])
        .arg(&file)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout), "5\n");
}

#[test]
fn save_writes_the_session() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("saved.mapal");
    let (out, _) = session(&format!("7 -> seven\n:save {}\n", file.display()));
    assert!(out.starts_with("seven = 7\nsaved "), "{out}");
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "fn main() {\n    7 -> seven;\n}\n"
    );
}

#[test]
fn bound_toggles_error_bounds() {
    let (out, _) = session(":bound\n:bound\n0.5 -> half\n");
    assert_eq!(out, "bound printing on\nbound printing off\nhalf = 0.5\n");
    let (out, _) = session("[1.0, 2.0] -> x\n:bound\nx[0]\n");
    assert!(out.contains("bound printing on\n"), "{out}");
    assert!(
        out.trim_end().ends_with('e') || out.contains("e-") || out.contains("e0"),
        "{out}"
    );
}

#[test]
fn mermaid_draws_only_the_expression() {
    let (out, err) = session("[1.0, 2.0] -> xs\n:mermaid (0.0, xs) -> fold { s, v -> s + v }\n");
    assert!(out.contains("flowchart"), "{out}{err}");
    // The session's name is an input, the value is named `result`; no print
    // node, no internal name, no empty frame of `main`.
    assert!(out.contains("xs: [f64; 2]"), "{out}");
    assert!(out.contains("result: f64"), "{out}");
    assert!(
        !out.contains("println") && !out.contains("shown_value"),
        "{out}"
    );
    assert!(
        !out.contains('\u{2205}') && !out.contains("fn main"),
        "{out}"
    );
}

#[test]
fn rebinding_a_name_reflows_what_reads_it() {
    let script = "fn sq(x: i32) -> i32 {\n    x * x -> ret;\n}\n\
                  5 -> a\na -> sq -> b\n7 -> a\nb\n";
    let (out, err) = session(script);
    assert_eq!(out, "defined sq\na = 5\nb = 25\na = 7\nb = 49\n49\n");
    assert_eq!(err, "");
}

#[test]
fn a_rebinding_that_does_not_fit_changes_nothing() {
    // `a` was bound before `later`, so its new value cannot read `later`.
    let (out, err) = session("5 -> a\n6 -> later\nlater -> a\na\n");
    assert_eq!(out, "a = 5\nlater = 6\n5\n");
    assert!(err.contains("unresolved name `later`"), "{err}");
    assert!(
        err.contains("hint: `a` is rebound where it was first bound"),
        "{err}"
    );
}

#[test]
fn arrays_print_as_lists() {
    let (out, _) = session("[1, 2, 3] -> xs\n[[1, 2], [3, 4]]\n");
    assert_eq!(out, "xs = [1, 2, 3]\n[[1, 2], [3, 4]]\n");
}

#[test]
fn commands_help_type_and_quit() {
    let (out, err) = session(":help\n:type 1\n:quit\n1 + 1\n");
    assert!(out.contains(":load <file>"), "{out}");
    assert!(!out.contains("\n2\n"), "{out}");
    assert!(err.contains(":type is not available"), "{err}");
}
