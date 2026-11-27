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

// --- scanning -------------------------------------------------------------

struct Scan {
    /// A bracket or string is still open.
    open: bool,
    /// The text with comments blanked (same length).
    code: String,
    /// Byte offsets of `->` at bracket depth 0.
    arrows: Vec<usize>,
    /// How many `;` sit at bracket depth 0.
    semis: usize,
}

fn scan(s: &str) -> Scan {
    let b = s.as_bytes();
    let mut code = b.to_vec();
    let (mut depth, mut i) = (0i32, 0usize);
    let mut arrows = Vec::new();
    let mut semis = 0usize;
    let mut in_str = false;
    while i < b.len() {
        let c = b[i];
        if in_str {
            if c == b'\\' {
                i += 1;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                code[i] = b' ';
                i += 1;
            }
            continue;
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut j = i + 2;
            while j < b.len() && !(b[j] == b'*' && b.get(j + 1) == Some(&b'/')) {
                j += 1;
            }
            let end = (j + 2).min(b.len());
            for (k, x) in code.iter_mut().enumerate().take(end).skip(i) {
                if *x != b'\n' {
                    *x = b' ';
                    let _ = k;
                }
            }
            if j >= b.len() {
                depth += 1; // an unterminated block comment keeps the input open
            }
            i = end;
            continue;
        } else if matches!(c, b'(' | b'[' | b'{') {
            depth += 1;
        } else if matches!(c, b')' | b']' | b'}') {
            depth -= 1;
        } else if c == b'-' && b.get(i + 1) == Some(&b'>') && depth == 0 {
            arrows.push(i);
        } else if c == b';' && depth == 0 {
            semis += 1;
        }
        i += 1;
    }
    Scan {
        open: depth > 0 || in_str,
        code: String::from_utf8_lossy(&code).into_owned(),
        arrows,
        semis,
    }
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_alphabetic() || f == '_')
        && c.all(|x| x.is_alphanumeric() || x == '_')
}

/// The first word of `code`, when it is exactly an item keyword.
fn item_name(code: &str) -> Option<String> {
    let t = code.trim_start();
    let rest = t
        .strip_prefix("fn")
        .or_else(|| t.strip_prefix("type"))
        .filter(|r| r.starts_with(char::is_whitespace))?;
    let name: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    Some(name)
}

enum Kind {
    Item(String),
    /// A statement kept in the body; `Some(name)` when it ends by binding one.
    Stmt {
        bound: Option<String>,
    },
    Expr,
}

impl Repl {
    fn classify(&self, text: &str) -> Kind {
        let sc = scan(text);
        let code = sc.code.trim();
        if let Some(name) = item_name(code) {
            return Kind::Item(name);
        }
        let first: String = code
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if first == "mut" || first == "loop" {
            return Kind::Stmt { bound: None };
        }
        let semi = code.ends_with(';');
        let core = code.trim_end_matches(';').trim_end();
        let bound = sc
            .arrows
            .iter()
            .rev()
            .find(|&&a| a + 2 <= core.len())
            .and_then(|&a| {
                let tail = core[a + 2..].trim();
                let name = tail.split(':').next().unwrap_or("").trim();
                let known = mapal_lower::builtin_names().contains(&name)
                    || matches!(name, "ret" | "loop")
                    || self.s.items.iter().any(|e| e.name == name);
                (is_ident(name) && !known).then(|| name.to_string())
            });
        // `expr -> println` without its `;` is the statement, not a value to show.
        let prints = sc
            .arrows
            .last()
            .is_some_and(|&a| matches!(core[a + 2..].trim(), "println" | "print"));
        match (semi || prints, bound) {
            (_, Some(n)) => Kind::Stmt { bound: Some(n) },
            (true, None) => Kind::Stmt { bound: None },
            (false, None) => Kind::Expr,
        }
    }

    /// Evaluate one complete input.
    fn input(&mut self, text: &str) -> bool {
        let text = text.trim_end();
        match self.classify(text) {
            Kind::Item(name) => self.define(&name, text),
            Kind::Stmt { bound } => self.statement(text, bound),
            Kind::Expr => self.expression(text),
        }
        true
    }

    fn define(&mut self, name: &str, text: &str) {
        if name == "main" {
            eprintln!("error: `main` is the session's own body; define other functions");
            return;
        }
        self.add_items(vec![Entry::typed(name, text, "input")], !name.is_empty());
    }

    /// Add or replace items, validate by running the program, commit on success.
    fn add_items(&mut self, new: Vec<Entry>, announce: bool) {
        let mut s = self.s.clone();
        for e in &new {
            match s
                .items
                .iter_mut()
                .find(|x| !e.name.is_empty() && x.name == e.name)
            {
                Some(slot) => *slot = e.clone(),
                None => s.items.push(e.clone()),
            }
        }
        let out = match self.run_session(&s, &[]) {
            Ok(out) => out,
            Err(f) => return f.print(),
        };
        s.retire();
        self.s = s;
        self.baseline = out;
        if announce {
            for e in new.iter().filter(|e| !e.name.is_empty()) {
                println!("defined {}", e.name);
            }
        }
    }

    fn statement(&mut self, text: &str, bound: Option<String>) {
        let mut stmt = Entry::typed("", text, "input");
        if !text.trim_end().ends_with(';') && !text.trim_end().ends_with('}') {
            stmt.text.push(';');
        }
        // A single binding may take the place of an earlier binding of its name.
        if scan(text).semis <= 1 {
            stmt.name = bound.clone().unwrap_or_default();
        }
        let mut s = self.s.clone();
        let at = (!stmt.name.is_empty())
            .then(|| s.body.iter().position(|e| e.name == stmt.name))
            .flatten();
        match at {
            Some(i) => s.body[i] = stmt,
            None => s.body.push(stmt),
        }
        let out = match self.run_session(&s, &[]) {
            Ok(out) => out,
            Err(f) => {
                f.print();
                if let Some(i) = at {
                    eprintln!(
                        "hint: `{}` is rebound where it was first bound (line {} of the session body): \
                         the new value cannot read a name bound after it, and the statements after it \
                         must still accept its type",
                        s.body[i].name,
                        i + 1
                    );
                }
                return;
            }
        };
        let mut lines = Vec::new();
        if let Some(name) = &bound {
            let shown = match self.value_of(&s, &out, name, name.len()) {
                Ok(v) => v,
                Err(_) => "(not printable)".to_string(),
            };
            lines.push(format!("{name} = {shown}"));
        }
        if let Some(i) = at {
            // The bindings after it flow again from the new value: show those that changed.
            for e in s.body[i + 1..].iter().filter(|e| !e.name.is_empty()) {
                let new = self.value_of(&s, &out, &e.name, e.name.len());
                let old = self.value_of(&self.s, &self.baseline, &e.name, e.name.len());
                if let Ok(new) = new
                    && old.ok().as_deref() != Some(new.as_str())
                {
                    lines.push(format!("{} = {new}", e.name));
                }
            }
            if out != self.baseline {
                print!("output now:\n{out}");
            }
        } else {
            print!("{}", delta(&self.baseline, &out));
        }
        for l in lines {
            println!("{l}");
        }
        s.retire();
        self.s = s;
        self.baseline = out;
    }

    fn expression(&mut self, text: &str) {
        let expr = text.trim_end().trim_end_matches(';');
        match self.value_of(&self.s.clone(), &self.baseline, expr, expr.len()) {
            Ok(v) => println!("{v}"),
            Err(f) => f.print(),
        }
    }

    /// The text `expr` prints as, on top of session `s`, nothing kept. An
    /// array (which `println` does not take) is shown element by element.
    fn value_of(
        &self,
        s: &Session,
        before: &str,
        expr: &str,
        visible: usize,
    ) -> Result<String, Fail> {
        let probe = Entry {
            visible,
            ..Entry::typed("", &format!("{expr} -> println;"), "input")
        };
        match self.run_session(s, &[probe]) {
            Ok(out) => Ok(delta(before, &out).trim_end().to_string()),
            Err(f) if f.code == "L1207" => {
                let Some(shape) = f.printed_type().and_then(|t| Shape::parse(&t)) else {
                    return Err(f);
                };
                let mut extra = vec![Entry {
                    visible,
                    ..Entry::typed("", &format!("{expr} -> shown_value;"), "input")
                }];
                let leaves = shape.leaves(SHOW_LEAVES);
                for idx in &leaves {
                    let at: String = idx.iter().map(|i| format!("[{i}]")).collect();
                    extra.push(Entry {
                        visible: 0,
                        ..Entry::typed("", &format!("shown_value{at} -> println;"), "echo")
                    });
                }
                let out = self.run_session(s, &extra).map_err(|_| f)?;
                let mut lines = delta(before, &out).lines();
                Ok(shape.render(&mut lines, SHOW_LEAVES))
            }
            Err(f) => Err(f),
        }
    }

    /// Run `s` (plus unkept `extra` statements); the output on success.
    fn run_session(&self, s: &Session, extra: &[Entry]) -> Ran {
        let built = assemble(s, extra);
        self.execute(&built)
    }

    fn execute(&self, built: &Built) -> Ran {
        let src = &built.src;
        let fail = |stage: &str, ds: &[Diagnostic]| {
            let mut f = Fail::default();
            for d in ds.iter().filter(|d| d.severity == Severity::Error) {
                render(built, stage, d, &mut f);
            }
            f
        };
        let po = mapal_syntax::parse(src);
        if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
            return Err(fail("parse", &po.diagnostics));
        }
        let ir = match mapal_lower::lower(src, &po.program) {
            Ok(ir) => ir.with_data_dir(crate::source_dir("x")),
            Err(ds) => return Err(fail("lower", &ds)),
        };
        let ds = mapal_check::check(src, &po.program, &ir);
        if ds.iter().any(|d| d.severity == Severity::Error) {
            return Err(fail("check", &ds));
        }
        self.interpret(ir)
    }

    /// The interpreter on `ir`; a panic inside it ends the input, not the session.
    fn interpret(&self, ir: mapal_ir::CategoryIr) -> Ran {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.interpret_ir(ir)))
            .unwrap_or_else(|_| Err(Fail::text("the interpreter failed on this program")))
    }

    fn interpret_ir(&self, ir: mapal_ir::CategoryIr) -> Ran {
        // A program that prints nothing has no bounds to show (and the
        // interpreter's bound run needs the print token).
        let prints = ir
            .func(ir.entry())
            .and_then(|f| ir.object(f.input))
            .is_some_and(|o| o.ty == mapal_ir::Ty::IoToken);
        if self.bound && prints {
            let ir = mapal_rewrite::rewrite(ir).ir;
            let (rr, bounds) = mapal_interp::run_bound(&ir, self.budget);
            if !matches!(rr.outcome, Outcome::Done(_)) {
                return Err(Fail::text("the composed bound needs a run that completes"));
            }
            return Ok(with_bounds(&rr.output, &bounds));
        }
        let rr = mapal_interp::run(&ir, self.budget);
        let Some((out, _)) = native_observable(&ir, &rr) else {
            return Err(Fail::text(&format!(
                "diverged — step budget ({}) exhausted (--budget N)",
                self.budget
            )));
        };
        match &rr.outcome {
            Outcome::Trapped(k) => {
                let what = match k {
                    TrapKind::DivZero => "division/modulo by zero",
                    TrapKind::IndexOob => "array index out of bounds",
                };
                Err(Fail::text(&format!("trapped — {what}")))
            }
            Outcome::ReadFailed(msg) => Err(Fail::text(msg)),
            _ => Ok(out.unwrap_or_default()),
        }
    }

    // --- commands ---------------------------------------------------------

    /// `false` to leave.
    fn command(&mut self, line: &str) -> bool {
        let (cmd, arg) = match line.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (line, ""),
        };
        match cmd {
            ":quit" | ":q" | ":exit" => return false,
            ":help" | ":h" | ":?" => println!("{COMMANDS}"),
            ":reset" => {
                self.s = Session::default();
                self.baseline.clear();
                println!("session cleared");
            }
            ":list" => print!("{}", self.listing()),
            ":save" => self.save(arg),
            ":load" => self.load(arg),
            ":bound" => {
                self.bound = !self.bound;
                // The baseline must come from the same printing as what follows.
                match self.run_session(&self.s.clone(), &[]) {
                    Ok(out) => self.baseline = out,
                    Err(_) => self.baseline.clear(),
                }
                println!("bound printing {}", if self.bound { "on" } else { "off" });
            }
            ":mermaid" => self.mermaid(arg),
            ":type" => eprintln!(
                "error: :type is not available — the checker does not expose a value's type on its own"
            ),
            _ => eprintln!("error: unknown command `{cmd}` (:help lists them)"),
        }
        true
    }

    fn program(&self) -> String {
        let mut p = String::new();
        for e in &self.s.items {
            p.push_str(&e.text);
            p.push_str("\n\n");
        }
        p.push_str("fn main() {\n");
        for e in &self.s.body {
            p.push_str(&e.text);
            p.push('\n');
        }
        p.push_str("}\n");
        p
    }

    fn listing(&self) -> String {
        let raw = self.program();
        format_source(&raw).unwrap_or(raw)
    }

    fn save(&self, path: &str) {
        if path.is_empty() {
            eprintln!("error: :save needs a file name");
            return;
        }
        match std::fs::write(path, self.listing()) {
            Ok(()) => println!("saved {path}"),
            Err(e) => eprintln!("error: cannot write {path}: {e}"),
        }
    }

    fn load(&mut self, path: &str) {
        if path.is_empty() {
            eprintln!("error: :load needs a file name");
            return;
        }
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read {path}: {e}");
                return;
            }
        };
        let po = mapal_syntax::parse(&src);
        let lines = LineIndex::new(&src);
        if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
            for d in po
                .diagnostics
                .iter()
                .filter(|d| d.severity == Severity::Error)
            {
                let lc = lines.line_col(d.span.start);
                eprintln!(
                    "{path}:{}:{} error[{}] (parse) {}",
                    lc.line, lc.col, d.code.0, d.message
                );
            }
            return;
        }
        let mut new = Vec::new();
        for item in &po.program.items {
            let (name, span) = match item {
                Item::Fn(f) => (slice(&src, f.name.span).to_string(), f.span),
                Item::Type(t) => (slice(&src, t.name.span).to_string(), t.span),
                Item::Bind(b) => (String::new(), b.span),
                Item::Error(_) => continue,
            };
            if name == "main" {
                continue;
            }
            new.push(Entry {
                name,
                text: slice(&src, span).to_string(),
                visible: span.end as usize - span.start as usize,
                label: path.to_string(),
                line_base: lines.line_col(span.start).line - 1,
            });
        }
        let n = new.len();
        let names: Vec<String> = new.iter().map(|e| e.name.clone()).collect();
        self.add_items(new, false);
        if names
            .iter()
            .all(|nm| nm.is_empty() || self.s.items.iter().any(|e| &e.name == nm))
        {
            println!("loaded {n} item(s) from {path}: {}", names.join(", "));
        }
    }

    fn mermaid(&self, input: &str) {
        if input.trim().is_empty() {
            eprintln!("error: :mermaid needs an expression");
            return;
        }
        let text = input.trim_end().trim_end_matches(';');
        // A binding is drawn under its own name; an expression under `result`.
        let (stmt, name) = match self.classify(text) {
            Kind::Stmt { bound: Some(n) } => (format!("{text};"), n),
            _ => (format!("{text} -> result;"), "result".to_string()),
        };
        let probe = Entry {
            name: name.clone(),
            visible: text.len(),
            ..Entry::typed("", &stmt, "input")
        };
        // The session's bindings as inputs (top-level values, no effects), the
        // expression alone in `main`; else the whole session before it.
        let mut inputs = Session {
            items: self.s.items.clone(),
            body: Vec::new(),
        };
        for e in &self.s.body {
            if !e.name.is_empty() && e.name != name && !e.text.contains("print") {
                inputs.items.push(e.clone());
            }
        }
        // The rewritten graph (what the compiler works on); where it folds
        // the expression away entirely (constants), the lowered one.
        for rewrite in [true, false] {
            for s in [&inputs, &self.s] {
                let built = assemble(s, std::slice::from_ref(&probe));
                if let Some(text) = diagram(&built, rewrite) {
                    let text = unwrapped(&text);
                    if text.lines().count() > 1 {
                        print!("{text}");
                        return;
                    }
                }
            }
        }
        let built = assemble(&self.s, std::slice::from_ref(&probe));
        if let Err(f) = self.execute(&built) {
            f.print();
        }
    }
}

/// The diagram without the frame of `main`: no box for `main` itself or for
/// the session's functions the expression does not use, no unused `ret` node,
/// and the `∅` that ends an unconsumed result drawn as `output`.
fn unwrapped(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut drop = vec![false; lines.len()];
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        if let Some((id, rest)) = t.split_once("((\"ret\"))")
            && rest.is_empty()
            && !id.is_empty()
            && !lines.iter().any(|l| l.trim() != t && l.contains(id))
        {
            drop[i] = true;
        }
    }
    let mut open: Vec<usize> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        if t.starts_with("subgraph ") {
            open.push(i);
        } else if t == "end"
            && let Some(start) = open.pop()
        {
            let empty = drop[start + 1..i].iter().all(|&d| d);
            if empty || lines[start].contains("\"fn main\"") {
                drop[start] = true;
                drop[i] = true;
            }
        }
    }
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if !drop[i] {
            out.push_str(&line.replace("((\"∅\"))", "((\"output\"))"));
            out.push('\n');
        }
    }
    out
}

/// The condensed string diagram of a program, `None` when it does not
/// parse, lower and check (silently).
fn diagram(built: &Built, rewrite: bool) -> Option<String> {
    let po = mapal_syntax::parse(&built.src);
    if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return None;
    }
    let ir = mapal_lower::lower(&built.src, &po.program).ok()?;
    if mapal_check::check(&built.src, &po.program, &ir)
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        return None;
    }
    let ir = if rewrite {
        mapal_rewrite::rewrite(ir).ir
    } else {
        ir
    };
    Some(ir.to_mermaid_string_condensed_src(&built.src))
}

/// Each printed line with its default-face bound (the format of `run
/// --interp --bound`): `0.5  ± 1.1e-16`. A string prints as itself.
fn with_bounds(output: &str, bounds: &[mapal_interp::Bound]) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() != bounds.len() || bounds.iter().any(|b| !b.newline) {
        return crate::bound_shape(bounds);
    }
    let mut out = String::new();
    for (line, b) in lines.iter().zip(bounds) {
        match &b.text {
            Some(t) if t == line => out.push_str(line),
            Some(t) => out.push_str(&format!("{line}  ± {t}")),
            None => out.push_str(&format!("{line}  ± {:e}", b.total)),
        }
        out.push('\n');
    }
    out
}

fn slice(src: &str, span: mapal_syntax::SourceLoc) -> &str {
    &src[span.start as usize..span.end as usize]
}

/// The output beyond what `prev` already printed.
fn delta<'a>(prev: &str, new: &'a str) -> &'a str {
    new.strip_prefix(prev).unwrap_or(new)
}

fn assemble(s: &Session, extra: &[Entry]) -> Built {
    let mut src = String::new();
    let mut segs = Vec::new();
    let mut put = |src: &mut String, e: &Entry| {
        let start = src.len();
        src.push_str(&e.text);
        segs.push(Seg {
            start,
            end: src.len(),
            entry: e.clone(),
        });
        src.push('\n');
    };
    for e in &s.items {
        put(&mut src, e);
        src.push('\n');
    }
    src.push_str("fn main() {\n");
    for e in &s.body {
        put(&mut src, e);
    }
    for e in extra {
        put(&mut src, e);
    }
    src.push_str("}\n");
    Built { src, segs }
}

/// Render one diagnostic against the text the user typed.
fn render(built: &Built, stage: &str, d: &Diagnostic, f: &mut Fail) {
    if f.code.is_empty() {
        f.code = d.code.0.to_string();
        f.msg = d.message.clone();
    }
    let off = d.span.start as usize;
    let Some(seg) = built.segs.iter().find(|g| off >= g.start && off <= g.end) else {
        // In the synthesized frame (`fn main`): nothing typed to point at.
        f.lines
            .push(format!("error[{}] ({stage}) {}", d.code.0, d.message));
        return;
    };
    let e = &seg.entry;
    let rel = (off - seg.start).min(e.visible);
    let shown = &e.text[..e.visible];
    let lc = LineIndex::new(shown).line_col(rel as u32);
    let place = if e.label == "echo" {
        "the shown value".to_string()
    } else {
        e.label.clone()
    };
    f.lines.push(format!(
        "{place}:{}:{} error[{}] ({stage}) {}",
        lc.line + e.line_base,
        lc.col,
        d.code.0,
        d.message
    ));
    if let Some(line) = shown.lines().nth(lc.line as usize - 1) {
        f.lines.push(format!("  | {line}"));
        f.lines
            .push(format!("  | {}^", " ".repeat(lc.col as usize - 1)));
    }
}

fn first_line(t: &str) -> String {
    let l = t.lines().next().unwrap_or("");
    if l.len() < t.len() {
        format!("{l} ...")
    } else {
        l.to_string()
    }
}
