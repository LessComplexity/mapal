//! `impl Repl`: classifying an input, the session's items and body, and running it.

use super::*;

impl Repl {
    pub(super) fn classify(&self, text: &str) -> Kind {
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
    pub(super) fn input(&mut self, text: &str) -> bool {
        let text = text.trim_end();
        match self.classify(text) {
            Kind::Item(name) => self.define(&name, text),
            Kind::Stmt { bound } => self.statement(text, bound),
            Kind::Expr => self.expression(text),
        }
        true
    }

    pub(super) fn define(&mut self, name: &str, text: &str) {
        if name == "main" {
            eprintln!("error: `main` is the session's own body; define other functions");
            return;
        }
        self.add_items(vec![Entry::typed(name, text, "input")], !name.is_empty());
    }

    /// Add or replace items, validate by running the program, commit on success.
    pub(super) fn add_items(&mut self, new: Vec<Entry>, announce: bool) {
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

    pub(super) fn statement(&mut self, text: &str, bound: Option<String>) {
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

    pub(super) fn expression(&mut self, text: &str) {
        let expr = text.trim_end().trim_end_matches(';');
        match self.value_of(&self.s.clone(), &self.baseline, expr, expr.len()) {
            Ok(v) => println!("{v}"),
            Err(f) => f.print(),
        }
    }

    /// The text `expr` prints as, on top of session `s`, nothing kept. An
    /// array (which `println` does not take) is shown element by element.
    pub(super) fn value_of(
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
    pub(super) fn run_session(&self, s: &Session, extra: &[Entry]) -> Ran {
        let built = assemble(s, extra);
        self.execute(&built)
    }

    pub(super) fn execute(&self, built: &Built) -> Ran {
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
    pub(super) fn interpret(&self, ir: mapal_ir::CategoryIr) -> Ran {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.interpret_ir(ir)))
            .unwrap_or_else(|_| Err(Fail::text("the interpreter failed on this program")))
    }

    pub(super) fn interpret_ir(&self, ir: mapal_ir::CategoryIr) -> Ran {
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
}
