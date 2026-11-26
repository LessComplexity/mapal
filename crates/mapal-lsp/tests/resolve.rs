//! Name resolution against the compiler, over every program in the repo
//! (`examples/*.mapal`, `benches/algorithms/**/*.mapal`, the editor fixture)
//! and the scoping cases the repo's programs do not happen to contain
//! (`tests/programs/*.mapal`).
//!
//! Two oracles, neither of which is the analysis itself:
//!
//! - **Every name resolves.** In a program the compiler accepts, each
//!   identifier is a definition, a use of one, or a builtin / primitive name;
//!   a use names its definition's name, lies where the definition is visible,
//!   and (for a local) comes after it.
//! - **Rename is meaning-preserving.** Renaming a definition and every
//!   occurrence the analysis attributes to it, to a fresh name of the same
//!   length, must lower to the same IR (names mapped back): a missed use is an
//!   unknown name, a use attributed to the wrong definition changes the graph.
//!   This is the compiler saying what each name means.

use std::path::{Path, PathBuf};

use mapal_lsp::Analysis;
use mapal_lsp::analysis::{DefKind, Role};
use mapal_syntax::{Severity, TokenKind};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn programs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/programs")
}

fn mapal_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            mapal_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "mapal") {
            out.push(p);
        }
    }
}

fn corpus() -> Vec<PathBuf> {
    let mut out = Vec::new();
    mapal_files(&root().join("examples"), &mut out);
    mapal_files(&root().join("benches/algorithms"), &mut out);
    mapal_files(&root().join("editors/test"), &mut out);
    mapal_files(&programs(), &mut out);
    assert!(
        out.len() > 100,
        "corpus not found under {}",
        root().display()
    );
    out
}

/// The lowered IR's debug text, or `None` when the compiler rejects the text.
fn lowered(text: &str) -> Option<String> {
    let po = mapal_syntax::parse(text);
    if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return None;
    }
    let ir = mapal_lower::lower(text, &po.program).ok()?;
    Some(format!("{ir:#?}"))
}

fn rel(p: &Path) -> String {
    p.strip_prefix(root()).unwrap_or(p).display().to_string()
}

#[test]
fn every_name_resolves() {
    let mut checked = 0usize;
    let mut names = 0usize;
    let mut failures = Vec::new();
    for file in corpus() {
        let text = std::fs::read_to_string(&file).unwrap();
        if lowered(&text).is_none() {
            continue;
        }
        checked += 1;
        let a = Analysis::new(&text);
        let lines = mapal_syntax::LineIndex::new(&text);
        for tok in mapal_syntax::lex(&text).tokens {
            if tok.kind != TokenKind::Ident {
                continue;
            }
            names += 1;
            let word = &text[tok.span.start as usize..tok.span.end as usize];
            let at = lines.line_col(tok.span.start);
            let here = format!("{}:{}:{} `{word}`", rel(&file), at.line, at.col);
            if a.lexical.iter().any(|(s, _)| *s == tok.span) {
                continue;
            }
            let Some(o) = a.occs.iter().find(|o| o.span == tok.span) else {
                failures.push(format!("{here}: unresolved"));
                continue;
            };
            let d = &a.defs[o.def];
            if d.name != word {
                failures.push(format!("{here}: resolves to `{}`", d.name));
            }
            if o.role == Role::Decl && d.name_span != tok.span && d.kind != DefKind::Global {
                failures.push(format!(
                    "{here}: a declaration that is not its definition's name"
                ));
            }
            if o.role != Role::Decl
                && matches!(
                    d.kind,
                    DefKind::Local | DefKind::Param | DefKind::BodyParam | DefKind::SizeParam
                )
            {
                if d.name_span.start >= tok.span.start {
                    failures.push(format!("{here}: used before its definition"));
                }
                if !(d.visible.start <= tok.span.start && tok.span.end <= d.visible.end) {
                    failures.push(format!("{here}: used outside its definition's scope"));
                }
            }
        }
    }
    assert!(checked > 100, "only {checked} programs lowered");
    assert!(
        failures.is_empty(),
        "{} of {names} names in {checked} programs:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("every_name_resolves: {names} names in {checked} programs");
}

/// A fresh name of `len` characters that no quoted IR name starts with and
/// the source never spells, so mapping it back in the IR text is exact.
fn fresh(len: usize, text: &str, ir: &str, taken: &mut Vec<String>) -> Option<String> {
    let ok = |c: &str, taken: &Vec<String>| {
        !ir.contains(&format!("\"{c}"))
            && !taken.iter().any(|t| t == c)
            && mapal_lsp::server::valid_name(c).is_ok()
            && !mapal_syntax::lex(text).tokens.iter().any(|t| {
                t.kind == TokenKind::Ident && &text[t.span.start as usize..t.span.end as usize] == c
            })
            && (len == 1 || !text.contains(c))
    };
    let alphabet: Vec<char> = ('a'..='z').chain('0'..='9').collect();
    for n in 0..200_000usize {
        let mut s = String::new();
        let mut k = n;
        s.push(['z', 'q', 'j', 'k', 'v', 'w', 'x', 'y'][k % 8]);
        k /= 8;
        while s.len() < len {
            s.push(alphabet[k % alphabet.len()]);
            k /= alphabet.len();
        }
        if s.len() == len && ok(&s, taken) {
            taken.push(s.clone());
            return Some(s);
        }
        if len == 1 && n >= 8 {
            break;
        }
    }
    None
}

fn apply(text: &str, mut edits: Vec<(mapal_syntax::SourceLoc, String)>) -> String {
    edits.sort_by_key(|(s, _)| std::cmp::Reverse(s.start));
    let mut out = text.to_string();
    for (s, t) in edits {
        out.replace_range(s.start as usize..s.end as usize, &t);
    }
    out
}

/// Check renaming each chosen definition of `file`; returns (renamed, failures).
fn rename_file(file: &Path, every: usize) -> (usize, Vec<String>) {
    let text = std::fs::read_to_string(file).unwrap();
    let Some(ir) = lowered(&text) else {
        return (0, Vec::new());
    };
    let a = Analysis::new(&text);
    let lines = mapal_syntax::LineIndex::new(&text);
    let mut taken = Vec::new();
    let mut failures = Vec::new();
    let mut renamed = 0;
    for (i, d) in a.defs.iter().enumerate() {
        if d.name == "main" || d.name == "_" || i % every != 0 {
            continue;
        }
        let Some(new) = fresh(d.name.len(), &text, &ir, &mut taken) else {
            continue;
        };
        let renamed_text = apply(&text, mapal_lsp::server::rename_edits(&a, i, &new));
        let at = lines.line_col(d.name_span.start);
        let here = format!(
            "{}:{}:{} {:?} `{}` → `{new}`",
            rel(file),
            at.line,
            at.col,
            d.kind,
            d.name
        );
        renamed += 1;
        match lowered(&renamed_text) {
            None => failures.push(format!("{here}: the renamed program no longer compiles")),
            Some(after) => {
                let back = after.replace(&format!("\"{new}"), &format!("\"{}", d.name));
                if back != ir {
                    failures.push(format!("{here}: the renamed program means something else"));
                }
            }
        }
    }
    (renamed, failures)
}

#[test]
fn renaming_any_definition_keeps_the_program() {
    let files = corpus();
    let jobs = std::env::var("MAPAL_TESTGEN_JOBS")
        .ok()
        .and_then(|j| j.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    // Examples and the editor fixture: every definition. The benchmark
    // programs repeat a few shapes many times: every third definition.
    let work: Vec<(PathBuf, usize)> = files
        .into_iter()
        .map(|f| {
            let every = if rel(&f).starts_with("benches") { 3 } else { 1 };
            (f, every)
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new((0usize, Vec::<String>::new()));
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((f, every)) = work.get(i) else {
                        break;
                    };
                    let (n, fs) = rename_file(f, *every);
                    let mut r = results.lock().unwrap();
                    r.0 += n;
                    r.1.extend(fs);
                }
            });
        }
    });
    let (renamed, failures) = results.into_inner().unwrap();
    assert!(renamed > 500, "only {renamed} renames checked");
    assert!(
        failures.is_empty(),
        "{} of {renamed} renames:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("renaming_any_definition_keeps_the_program: {renamed} renames");
}

/// A struct-literal pun is a field and a value at once: renaming either
/// spells the pun out, and the program keeps its meaning.
#[test]
fn renaming_through_a_pun_keeps_the_program() {
    let text = "type P {\n    x: i32,\n    y: i32,\n}\n\nfn main() {\n    3 -> x;\n    4 -> y;\n    P {\n        x,\n        y\n    } -> p;\n    p.x + p.y -> println;\n}\n";
    let before = lowered(text).expect("the pun program compiles");
    let a = Analysis::new(text);
    let pun = text.find("        x,").unwrap() as u32 + 8;
    let local = a.occ_at(pun).unwrap().def;
    assert_eq!(
        a.defs[local].kind,
        DefKind::Local,
        "definition at a pun is the value"
    );
    let field = a
        .defs
        .iter()
        .position(|d| d.kind == DefKind::Field && d.name == "x")
        .unwrap();
    assert!(
        a.occurrences(field).any(|o| o.span.start == pun),
        "the pun is a use of the field"
    );
    for (def, new, want) in [(local, "v", "x: v"), (field, "f", "f: x")] {
        let edits = mapal_lsp::server::rename_edits(&a, def, new);
        assert!(
            edits.iter().any(|(s, t)| s.start == pun && t == want),
            "{edits:?}"
        );
        let renamed = apply(text, edits);
        let after =
            lowered(&renamed).unwrap_or_else(|| panic!("renamed program compiles:\n{renamed}"));
        // Spans move (the pun grew), so compare what the program prints.
        assert_eq!(
            before.matches("Print").count(),
            after.matches("Print").count(),
            "{renamed}"
        );
    }
}

fn program(name: &str) -> String {
    std::fs::read_to_string(programs().join(name)).unwrap()
}

/// The definition the name at the `nth` `needle` (`into` bytes in) resolves to.
fn def_at<'a>(
    a: &'a Analysis,
    text: &str,
    needle: &str,
    nth: usize,
    into: usize,
) -> Option<&'a mapal_lsp::analysis::Def> {
    let at = text.match_indices(needle).nth(nth)?.0 + into;
    a.def_at(at as u32).map(|d| &a.defs[d])
}

/// ADR-0019 pin b: a `seq` block lowers in the enclosing scope, so a name it
/// binds is read after it (and renaming it keeps the program, through the
/// corpus oracle above); pin a: a headless statement starts from the seq's
/// input, so its binding has the input's type.
#[test]
fn seq_bindings_escape_the_block() {
    let text = program("seq_scope.mapal");
    assert!(lowered(&text).is_some(), "the program compiles");
    let a = Analysis::new(&text);
    let escaped =
        def_at(&a, &text, "escaped + next", 0, 0).expect("`escaped` resolves after the seq");
    assert_eq!(escaped.kind, DefKind::Local);
    assert_eq!(
        escaped.name_span.start as usize,
        text.find("-> escaped;").unwrap() + 3
    );
    assert!(
        def_at(&a, &text, "escaped + next", 0, 10).is_some(),
        "`next` resolves after the seq"
    );
    let sample = def_at(&a, &text, "sample.value", 0, 0).expect("`sample` resolves");
    assert_eq!(
        sample.ty.render(),
        "Reading",
        "a headless statement starts from the seq's input"
    );
    let value = def_at(&a, &text, "sample.value", 0, 7).expect("the field resolves");
    assert_eq!((value.kind, value.name.as_str()), (DefKind::Field, "value"));
}

/// ADR-0048 D2: functions and top-level values are program-wide whatever the
/// order. A call to a function declared below its caller, a top-level value
/// bound from a call, and a top-level value read above its binding all have
/// their type, so their fields resolve (renaming a field keeps the program,
/// through the corpus oracle above).
#[test]
fn types_flow_from_later_declarations() {
    let text = program("declared_later.mapal");
    assert!(lowered(&text).is_some(), "the program compiles");
    let a = Analysis::new(&text);
    for (needle, nth, into, field) in [
        ("made.bias", 0, 5, "bias"),
        ("made.scale", 0, 5, "scale"),
        ("pick.scale", 0, 5, "scale"),
        ("cfg.scale + cfg.bias", 0, 4, "scale"),
        ("cfg.scale + cfg.bias", 0, 16, "bias"),
        ("cfg.scale * 2.0", 0, 4, "scale"),
    ] {
        let d = def_at(&a, &text, needle, nth, into)
            .unwrap_or_else(|| panic!("`{field}` in `{needle}` resolves"));
        assert_eq!(
            (d.kind, d.name.as_str()),
            (DefKind::Field, field),
            "{needle}"
        );
    }
    let made = def_at(&a, &text, "made.bias", 0, 0).unwrap();
    assert_eq!(made.ty.render(), "Config");
    let cfg = def_at(&a, &text, "cfg.scale + cfg.bias", 0, 0).unwrap();
    assert_eq!(
        (cfg.kind, cfg.ty.render()),
        (DefKind::Global, "Config".to_string())
    );
    let make = def_at(&a, &text, "-> make -> made", 0, 3).unwrap();
    assert_eq!(
        (make.kind, make.ty.render()),
        (DefKind::Function, "Config".to_string())
    );
    // The pre-pass leaves no trace: one declaration per definition.
    for (i, d) in a.defs.iter().enumerate() {
        let decls = a
            .occurrences(i)
            .filter(|o| o.role == Role::Decl && o.span == d.name_span)
            .count();
        assert!(decls <= 1, "`{}` declared {decls} times", d.name);
    }
}
