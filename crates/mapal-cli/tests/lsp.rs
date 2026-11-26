//! `mapal lsp`: the language server over stdio, as editors start it.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};

fn frame(body: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

fn read_message(out: &mut impl BufRead) -> String {
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        let n = out.read_line(&mut line).unwrap();
        assert!(n > 0, "the server closed its output (it exited or aborted)");
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(n) = line.strip_prefix("Content-Length: ") {
            len = n.parse().unwrap();
        }
    }
    let mut body = vec![0; len];
    out.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

#[test]
fn lsp_initializes_and_exits_cleanly() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(["lsp", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    stdin
        .write_all(&frame(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#,
        ))
        .unwrap();
    let init = read_message(&mut stdout);
    assert!(init.contains("\"definitionProvider\":true"), "{init}");
    assert!(init.contains("mapal-lsp"), "{init}");
    stdin
        .write_all(&frame(
            r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#,
        ))
        .unwrap();
    stdin
        .write_all(&frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#))
        .unwrap();
    assert!(read_message(&mut stdout).contains("\"id\":2"));
    stdin
        .write_all(&frame(r#"{"jsonrpc":"2.0","method":"exit"}"#))
        .unwrap();
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

/// The server outlives what the CLI reports on: a file nested past the
/// parser's limit (P0011) and a long flat sum, which the front end walks
/// recursively (more than a spawned thread's default 2 MiB stack in a debug
/// build, within the 8 MiB main thread `mapal build` uses). Each gets its
/// diagnostics published; the server then shuts down cleanly.
#[test]
fn lsp_survives_deep_programs() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(["lsp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    stdin
        .write_all(&frame(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#,
        ))
        .unwrap();
    read_message(&mut stdout);
    stdin
        .write_all(&frame(
            r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#,
        ))
        .unwrap();
    let n = 40_000;
    let nested = format!(
        "fn main() {{\n    {}1{} -> println;\n}}\n",
        "(".repeat(n),
        ")".repeat(n)
    );
    let terms = vec!["1"; 200].join(" + ");
    let sum = format!("fn main() {{\n    {terms} -> println;\n}}\n");
    for (uri, text, code) in [
        ("file:///tmp/nested.mapal", nested, Some("P0011")),
        ("file:///tmp/sum.mapal", sum, None),
    ] {
        let text = text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        let open = format!(
            r#"{{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{{"textDocument":{{"uri":"{uri}","languageId":"mapal","version":1,"text":"{text}"}}}}}}"#
        );
        stdin.write_all(&frame(&open)).unwrap();
        let published = loop {
            let m = read_message(&mut stdout);
            if m.contains("publishDiagnostics") && m.contains(uri) {
                break m;
            }
        };
        match code {
            Some(c) => assert!(published.contains(c), "{published}"),
            None => assert!(published.contains("\"diagnostics\":[]"), "{published}"),
        }
    }
    stdin
        .write_all(&frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#))
        .unwrap();
    assert!(read_message(&mut stdout).contains("\"id\":2"));
    stdin
        .write_all(&frame(r#"{"jsonrpc":"2.0","method":"exit"}"#))
        .unwrap();
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

#[test]
fn lsp_rejects_other_transports() {
    let out = Command::new(env!("CARGO_BIN_EXE_mapal"))
        .args(["lsp", "--port", "9000"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("stdin/stdout"));
}
