//! The session's program text, diagrams, bounds and diagnostics rendering.

use super::*;

/// The diagram without the frame of `main`: no box for `main` itself or for
/// the session's functions the expression does not use, no unused `ret` node,
/// and the `∅` that ends an unconsumed result drawn as `output`.
pub(super) fn unwrapped(text: &str) -> String {
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
pub(super) fn diagram(built: &Built, rewrite: bool) -> Option<String> {
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
pub(super) fn with_bounds(output: &str, bounds: &[mapal_interp::Bound]) -> String {
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

pub(super) fn slice(src: &str, span: mapal_syntax::SourceLoc) -> &str {
    &src[span.start as usize..span.end as usize]
}

/// The output beyond what `prev` already printed.
pub(super) fn delta<'a>(prev: &str, new: &'a str) -> &'a str {
    new.strip_prefix(prev).unwrap_or(new)
}

pub(super) fn assemble(s: &Session, extra: &[Entry]) -> Built {
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
pub(super) fn render(built: &Built, stage: &str, d: &Diagnostic, f: &mut Fail) {
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

pub(super) fn first_line(t: &str) -> String {
    let l = t.lines().next().unwrap_or("");
    if l.len() < t.len() {
        format!("{l} ...")
    } else {
        l.to_string()
    }
}
