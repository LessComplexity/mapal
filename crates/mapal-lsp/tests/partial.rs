//! Files mid-edit: the analysis never panics on a broken file, and what the
//! intact part of the file says still resolves the same way.

use std::path::Path;

use mapal_lsp::Analysis;

fn read(rel: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel),
    )
    .unwrap()
}

/// `(use, its definition)` for every occurrence ending before `cut` whose
/// definition is also before it, by span.
fn resolved_before(a: &Analysis, cut: u32) -> Vec<(u32, u32)> {
    a.occs
        .iter()
        .filter(|o| o.span.end <= cut && a.defs[o.def].name_span.end <= cut)
        .map(|o| (o.span.start, a.defs[o.def].name_span.start))
        .collect()
}

#[test]
fn truncated_files_resolve_their_intact_part() {
    for rel in [
        "examples/llama.mapal",
        "examples/matmul4_loop.mapal",
        "examples/fanout.mapal",
    ] {
        let text = read(rel);
        let full = Analysis::new(&text);
        // Cut at each top-level item boundary: everything before it is intact.
        let mut boundaries: Vec<u32> = full
            .program
            .items
            .iter()
            .map(|i| match i {
                mapal_syntax::Item::Fn(f) => f.span.end,
                mapal_syntax::Item::Type(t) => t.span.end,
                mapal_syntax::Item::Bind(b) => b.span.end,
                mapal_syntax::Item::Error(s) => s.end,
            })
            .collect();
        boundaries.dedup();
        for &end in &boundaries {
            // The next item cut in half: `fn f(` with the rest of the file gone.
            let mut cut_text = text[..end as usize].to_string();
            cut_text.push_str("\nfn half(x: i32) -> i32 {\n    x -> ");
            let cut = Analysis::new(&cut_text);
            let want = resolved_before(&full, end);
            let got = resolved_before(&cut, end);
            // Uses of names defined later in the file cannot resolve; the rest must.
            for w in &want {
                assert!(
                    got.contains(w),
                    "{rel}: cut at {end}: use at {} lost its definition",
                    w.0
                );
            }
        }
        // Any cut at all: no panic.
        for cut in (0..text.len()).step_by(41) {
            if text.is_char_boundary(cut) {
                let _ = Analysis::new(&text[..cut]);
            }
        }
        // Each line deleted in turn: no panic, and functions still resolve.
        let lines: Vec<&str> = text.lines().collect();
        for skip in 0..lines.len() {
            let edited: String = lines
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != skip)
                .map(|(_, l)| format!("{l}\n"))
                .collect();
            let a = Analysis::new(&edited);
            assert!(
                a.defs
                    .iter()
                    .any(|d| d.kind == mapal_lsp::analysis::DefKind::Function)
            );
        }
    }
}

#[test]
fn aspirational_sketch_still_resolves() {
    // `vector.mapal` is post-Core syntax the parser rejects; its functions and
    // their parameters still resolve.
    let text = read("examples/vector.mapal");
    let a = Analysis::new(&text);
    assert!(!a.parse_diagnostics.is_empty());
    let fns: Vec<&str> = a
        .defs
        .iter()
        .filter(|d| d.kind == mapal_lsp::analysis::DefKind::Function)
        .map(|d| d.name.as_str())
        .collect();
    assert!(fns.contains(&"main"), "{fns:?}");
}
