//! `CategoryIr::to_mermaid_flow`: the condensed flow view of the compiler's own
//! graph. Golden output for two programs, the lint on every example and
//! exposition program (both stages), and the soundness property: every edge of
//! the flow view is a dependence path in the full graph.
//!
//! Regenerate the goldens with `MAPAL_BLESS=1 cargo test -p mapal-cli --test flow_mermaid`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mapal_ir::{CategoryIr, FlowAnchor, ObjectId, Operation, lint_mermaid};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The IR `mapal dump-ir` draws: lowered, or rewritten with the CLI's default
/// pipeline (static evaluation, then the rewriter).
fn ir_of(file: &Path, rewritten: bool) -> CategoryIr {
    let src = std::fs::read_to_string(file).unwrap();
    let po = mapal_syntax::parse(&src);
    assert!(po.diagnostics.is_empty(), "{}: {:?}", file.display(), po.diagnostics);
    let dir = std::fs::canonicalize(file.parent().unwrap()).unwrap();
    let ir = mapal_lower::lower(&src, &po.program)
        .unwrap_or_else(|d| panic!("{}: lower {d:?}", file.display()))
        .with_data_dir(Some(dir.display().to_string()));
    assert!(mapal_check::check(&src, &po.program, &ir).is_empty());
    if !rewritten {
        return ir;
    }
    let (ir, _) = mapal_rewrite::fold_static(ir);
    mapal_rewrite::rewrite(ir).ir
}

fn programs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for d in ["examples", "docs/exposition"] {
        for e in std::fs::read_dir(root().join(d)).unwrap() {
            let p = e.unwrap().path();
            // vector.mapal is outside Mapal-Core (P0001): it does not lower.
            if p.extension().is_some_and(|x| x == "mapal") && p.file_name().unwrap() != "vector.mapal" {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

fn golden(name: &str, file: &str) {
    let src = std::fs::read_to_string(root().join(file)).unwrap();
    let got = ir_of(&root().join(file), true).to_mermaid_flow_src(&src);
    assert!(lint_mermaid(&got).is_empty(), "{name}: {:?}", lint_mermaid(&got));
    let path = root().join("crates/mapal-cli/tests/golden").join(format!("{name}.flow.mmd"));
    if std::env::var("MAPAL_BLESS").is_ok() {
        std::fs::write(&path, &got).unwrap();
    }
    let want = std::fs::read_to_string(&path).expect("golden missing (MAPAL_BLESS=1)");
    assert_eq!(got, want, "{name}: flow view changed (MAPAL_BLESS=1 to accept)");
}

#[test]
fn golden_softmax_flow() {
    golden("softmax", "examples/softmax.mapal");
}

#[test]
fn golden_dense_flow() {
    golden("dense", "docs/exposition/dense.mapal");
}

#[test]
fn every_program_lints_clean_in_both_forms_and_stages() {
    for p in programs() {
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            for (form, text) in [("full", ir.to_mermaid()), ("flow", ir.to_mermaid_flow())] {
                let lints = lint_mermaid(&text);
                assert!(lints.is_empty(), "{} {form} rewritten={rewritten}: {lints:?}", p.display());
            }
            assert_eq!(ir.to_mermaid_flow(), ir.to_mermaid_flow(), "deterministic");
        }
    }
}

/// Reachability in the full graph, with a `Map` / `Fold` site's body joined to it:
/// the site's operand reaches the body's input, the body's result reaches the
/// site's result.
fn reach(ir: &CategoryIr, from: ObjectId) -> HashSet<ObjectId> {
    let mut adj: HashMap<ObjectId, Vec<ObjectId>> = HashMap::new();
    for (_, m) in ir.morphisms() {
        adj.entry(m.source).or_default().push(m.target);
        if let Operation::Map { body, .. } | Operation::Fold { body, .. } = m.op {
            let d = ir.func(body).unwrap();
            adj.entry(m.source).or_default().push(d.input);
            adj.entry(d.output).or_default().push(m.target);
        }
    }
    let mut seen = HashSet::new();
    let mut stack = vec![from];
    while let Some(o) = stack.pop() {
        if seen.insert(o) {
            stack.extend(adj.get(&o).into_iter().flatten().copied());
        }
    }
    seen
}

#[test]
fn every_flow_edge_is_a_dependence_path_in_the_full_graph() {
    let mut edges = 0;
    for p in programs() {
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            let g = ir.flow_graph();
            for &(a, b, _) in &g.edges {
                let from = match g.nodes[a] {
                    FlowAnchor::Object(o) => o,
                    FlowAnchor::Print(m) => ir.morphism(m).unwrap().target,
                };
                let to = match g.nodes[b] {
                    FlowAnchor::Object(o) => o,
                    FlowAnchor::Print(m) => ir.morphism(m).unwrap().source,
                };
                assert!(
                    reach(&ir, from).contains(&to),
                    "{} rewritten={rewritten}: flow edge {a}->{b} has no path in the graph",
                    p.display()
                );
                edges += 1;
            }
        }
    }
    assert!(edges > 100, "the property saw only {edges} edges");
}

/// Every arrow of the flow view is labelled; a label from source text is a
/// substring of the program (whitespace collapsed, long text ellipsized).
#[test]
fn every_flow_edge_is_labelled_and_source_labels_come_from_the_source() {
    let (mut edges, mut from_src) = (0, 0);
    for p in programs() {
        let src = std::fs::read_to_string(&p).unwrap();
        let norm = src.split_whitespace().collect::<Vec<_>>().join(" ");
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            let g = ir.flow_graph();
            for k in 0..g.edges.len() {
                let words = g.edge_label(&ir, k, None);
                assert!(!words.trim().is_empty(), "{}: edge {k} unlabelled", p.display());
                let l = g.edge_label(&ir, k, Some(&src));
                assert!(!l.trim().is_empty(), "{}: edge {k} unlabelled", p.display());
                let text = l.strip_suffix('…').unwrap_or(&l);
                if l != words {
                    from_src += 1;
                    assert!(norm.contains(text), "{}: label `{l}` is not source text", p.display());
                }
                edges += 1;
            }
            for line in ir.to_mermaid_flow_src(&src).lines().filter(|l| l.contains("-->")) {
                assert!(line.contains(" -- \""), "{}: unlabelled arrow `{line}`", p.display());
            }
        }
    }
    assert!(edges > 100 && from_src > 20, "saw {edges} edges, {from_src} from source");
}
