//! `impl Repl`: the `:` commands (program, listing, save, load, mermaid).

use super::*;

impl Repl {
    // --- commands ---------------------------------------------------------

    /// `false` to leave.
    pub(super) fn command(&mut self, line: &str) -> bool {
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

    pub(super) fn program(&self) -> String {
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

    pub(super) fn listing(&self) -> String {
        let raw = self.program();
        format_source(&raw).unwrap_or(raw)
    }

    pub(super) fn save(&self, path: &str) {
        if path.is_empty() {
            eprintln!("error: :save needs a file name");
            return;
        }
        match std::fs::write(path, self.listing()) {
            Ok(()) => println!("saved {path}"),
            Err(e) => eprintln!("error: cannot write {path}: {e}"),
        }
    }

    pub(super) fn load(&mut self, path: &str) {
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

    pub(super) fn mermaid(&self, input: &str) {
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
