//! `mapal repl` — an interactive session on the interpreter (the oracle).
//!
//! The session is a list of items (`fn`, `type`) and the body of an implicit
//! `fn main()` (bindings and statements, in order). Every input is judged by
//! building the whole program and running it on the interpreter; because the
//! interpreter is deterministic, only the output beyond what the session's
//! previous successful run printed is shown. A failing input prints its
//! diagnostics against what was typed and leaves the session as it was.

use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;

use mapal_interp::{Outcome, TrapKind, native_observable};
use mapal_syntax::{Diagnostic, Item, LineIndex, Severity, format_source};

use crate::Res;

const HELP: &str = "usage: mapal repl [--budget N]
  an interactive session on the interpreter. Type a definition, a binding, a
  statement or an expression; a multi-line input continues while ( [ { are open.";

const COMMANDS: &str = "\
Input
  fn f<n>(x: [f32; n]) -> [f32; n] { ... }   define (or redefine) a function; `type` likewise
  expr -> name;                              bind a value, shown as `name = value`
  expr -> println;                           a statement: kept, run, its new output shown
  expr                                       evaluate and show the value (kept nowhere)
Commands
  :help              this text
  :quit, :q, exit    leave (Ctrl-D too)
  :reset             forget every definition and binding
  :list              the session as one formatted source file
  :load <file>       add the file's fn / type items (its `main` is ignored, not run)
  :save <file>       write the session as a formatted program
  :bound             toggle printing floats as their default-face error bound
  :mermaid <expr>    the condensed string diagram of an expression (alone if it stands
                     alone, else with the session's bindings; static folding off)
  :type <expr>       not available: the checker does not expose a value's type on its own";

/// One piece of the synthesized program: an item or a body statement.
#[derive(Clone)]
struct Entry {
    /// The key a redefinition replaces (items); empty for statements.
    name: String,
    /// The text placed in the program (may carry a synthesized `;`).
    text: String,
    /// The prefix of `text` that the user typed.
    visible: usize,
    /// Where it came from, for diagnostics outside the current input.
    label: String,
    /// Lines before this entry in the file it was loaded from.
    line_base: u32,
}

impl Entry {
    fn typed(name: &str, text: &str, label: &str) -> Entry {
        Entry {
            name: name.to_string(),
            text: text.to_string(),
            visible: text.len(),
            label: label.to_string(),
            line_base: 0,
        }
    }
}

struct Seg {
    start: usize,
    end: usize,
    entry: Entry,
}

struct Built {
    src: String,
    segs: Vec<Seg>,
}

#[derive(Default, Clone)]
struct Session {
    items: Vec<Entry>,
    body: Vec<Entry>,
}

impl Session {
    /// Entries kept: later diagnostics name them as earlier input.
    fn retire(&mut self) {
        for e in self.items.iter_mut().chain(self.body.iter_mut()) {
            if e.label == "input" {
                e.label = format!("earlier input `{}`", first_line(&e.text));
            }
        }
    }
}

struct Repl {
    s: Session,
    /// What the last successful run of the session printed.
    baseline: String,
    bound: bool,
    budget: u64,
}

/// Output of a run: the text, or already-reported failure.
type Ran = Result<String, Fail>;

/// A failed run: the lines to show, and the first diagnostic's code and message.
#[derive(Default)]
struct Fail {
    lines: Vec<String>,
    code: String,
    msg: String,
}

impl Fail {
    fn text(msg: &str) -> Fail {
        Fail {
            lines: vec![format!("error: {msg}")],
            ..Fail::default()
        }
    }

    fn print(&self) {
        for l in &self.lines {
            eprintln!("{l}");
        }
    }

    /// The type of "cannot print `T`".
    fn printed_type(&self) -> Option<String> {
        let rest = self.msg.strip_prefix("cannot print `")?;
        Some(rest.split('`').next()?.to_string())
    }
}

/// How many array elements `:`-less display shows before cutting off.
const SHOW_LEAVES: usize = 64;

/// An array type `[[f32; 3]; 2]` as nested extents.
enum Shape {
    Scalar,
    Array(usize, Box<Shape>),
}

impl Shape {
    fn parse(t: &str) -> Option<Shape> {
        let t = t.trim();
        let Some(inner) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']')) else {
            return (!t.is_empty() && t.chars().all(|c| c.is_alphanumeric() || c == '_'))
                .then_some(Shape::Scalar);
        };
        let (elem, n) = inner.rsplit_once(';')?;
        Some(Shape::Array(
            n.trim().parse().ok()?,
            Box::new(Shape::parse(elem)?),
        ))
    }

    /// Index paths of the first `cap` scalars, row-major.
    fn leaves(&self, cap: usize) -> Vec<Vec<usize>> {
        fn go(s: &Shape, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>, cap: usize) {
            if out.len() >= cap {
                return;
            }
            match s {
                Shape::Scalar => out.push(path.clone()),
                Shape::Array(n, e) => {
                    for i in 0..*n {
                        path.push(i);
                        go(e, path, out, cap);
                        path.pop();
                    }
                }
            }
        }
        let mut out = Vec::new();
        go(self, &mut Vec::new(), &mut out, cap);
        out
    }

    /// `[a, b, c]` from the printed leaves, cut with `...` past `cap`.
    fn render<'a>(&self, lines: &mut impl Iterator<Item = &'a str>, cap: usize) -> String {
        fn go<'a>(
            s: &Shape,
            lines: &mut impl Iterator<Item = &'a str>,
            left: &mut usize,
            out: &mut String,
        ) {
            match s {
                Shape::Scalar => {
                    out.push_str(lines.next().unwrap_or("?"));
                    *left -= 1;
                }
                Shape::Array(n, e) => {
                    out.push('[');
                    for i in 0..*n {
                        if *left == 0 {
                            out.push_str("...");
                            break;
                        }
                        if i > 0 {
                            out.push_str(", ");
                        }
                        go(e, lines, left, out);
                    }
                    out.push(']');
                }
            }
        }
        let mut out = String::new();
        go(self, lines, &mut { cap }, &mut out);
        out
    }
}

pub(super) fn run(args: impl Iterator<Item = String>) -> Res<u8> {
    let mut budget = 100_000_000u64;
    let mut it = args;
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(0);
            }
            "--budget" => match it.next().and_then(|v| v.parse().ok()) {
                Some(n) => budget = n,
                None => {
                    eprintln!("mapal repl: --budget expects a non-negative integer\n{HELP}");
                    return Err(2);
                }
            },
            _ => {
                eprintln!("mapal repl: unknown argument `{a}`\n{HELP}");
                return Err(2);
            }
        }
    }
    let mut repl = Repl {
        s: Session::default(),
        baseline: String::new(),
        bound: false,
        budget,
    };
    if std::io::stdin().is_terminal() {
        interactive(&mut repl);
    } else {
        piped(&mut repl);
    }
    Ok(0)
}

fn piped(repl: &mut Repl) {
    let mut pending = String::new();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if !feed(repl, &mut pending, &line) {
            return;
        }
    }
    if !pending.trim().is_empty() {
        // End of input inside an open bracket: report it as typed.
        let text = std::mem::take(&mut pending);
        let _ = repl.input(&text);
    }
}

fn interactive(repl: &mut Repl) {
    use rustyline::error::ReadlineError;
    let Ok(mut rl) = rustyline::DefaultEditor::new() else {
        piped(repl);
        return;
    };
    let history = history_file();
    if let Some(h) = &history {
        let _ = rl.load_history(h);
    }
    println!(
        "Mapal {} — the interpreter. :help for commands, exit or Ctrl-D to quit.",
        env!("CARGO_PKG_VERSION")
    );
    let mut pending = String::new();
    let mut entry = String::new();
    loop {
        let prompt = if pending.is_empty() {
            "mapal> "
        } else {
            "   ...> "
        };
        match rl.readline(prompt) {
            Ok(line) => {
                // One history entry per complete input, however many lines.
                if !entry.is_empty() {
                    entry.push('\n');
                }
                entry.push_str(&line);
                let quit = !feed(repl, &mut pending, &line);
                if pending.is_empty() {
                    if !entry.trim().is_empty() {
                        let _ = rl.add_history_entry(entry.as_str());
                    }
                    entry.clear();
                }
                if quit {
                    break;
                }
            }
            Err(ReadlineError::Interrupted) => {
                pending.clear();
                entry.clear();
            }
            Err(_) => break,
        }
    }
    if let Some(h) = &history {
        if let Some(dir) = h.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = rl.save_history(h);
    }
}

fn history_file() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("mapal").join("repl_history"))
}

/// Add a line to the pending input; evaluate it when it is complete.
/// `false` when the session should end.
fn feed(repl: &mut Repl, pending: &mut String, line: &str) -> bool {
    if pending.is_empty() && line.trim_start().starts_with(':') {
        return repl.command(line.trim());
    }
    // `exit` / `quit` leave too, the words a newcomer types first.
    if pending.is_empty() && matches!(line.trim(), "exit" | "quit" | "exit()" | "quit()") {
        return false;
    }
    if !pending.is_empty() {
        pending.push('\n');
    }
    pending.push_str(line);
    if scan(pending).open {
        return true;
    }
    let text = std::mem::take(pending);
    if !text.trim().is_empty() {
        repl.input(&text);
    }
    true
}

mod commands;
mod render;
mod scan;
mod session;

use self::render::*;
use self::scan::*;
