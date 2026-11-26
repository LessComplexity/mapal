//! Print each name of a file with its highlight class and definition —
//! `cargo run -p mapal-lsp --example classify -- file.mapal` (a debugging aid
//! for reviewing `editors/test/fixture.tokens.tsv`).

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: classify <file.mapal>");
    let text = std::fs::read_to_string(&path).expect("readable file");
    let a = mapal_lsp::Analysis::new(&text);
    let lines = mapal_syntax::LineIndex::new(&text);
    for (span, class) in a.classified() {
        let lc = lines.line_col(span.start);
        let def = a
            .occ_at(span.start)
            .filter(|o| o.span == span)
            .map(|o| {
                let d = &a.defs[o.def];
                let at = lines.line_col(d.name_span.start);
                format!("-> {}:{} {:?} {}", at.line, at.col, d.kind, d.ty.render())
            })
            .unwrap_or_default();
        println!(
            "{}:{}\t{}\t{}\t{}",
            lc.line,
            lc.col,
            a.slice(span),
            class.name(),
            def
        );
    }
}
