//! The two categorical Mermaid views of the compiler's graph:
//! `--mermaid=category` (commutative diagram: objects are nodes, morphisms are
//! arrows, a product is one pairing `⟨a, b⟩` from the function's context `Γ`)
//! and `--mermaid=string` (string diagram: steps are boxes, values are wires,
//! a product is several wires entering one box, a copy is a `Δ` node), each in
//! a full and a `condensed` form. Goldens for three programs in all four forms,
//! the lint on every example and exposition program, and the structural laws.
//!
//! Regenerate the goldens with `MAPAL_BLESS=1 cargo test -p mapal-cli --test category_mermaid`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mapal_ir::{CategoryIr, Diagram, EdgeKind, NodeKind, ObjectKind, Operation, lint_mermaid};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ir_of(file: &Path, rewritten: bool) -> CategoryIr {
    let src = std::fs::read_to_string(file).unwrap();
    let po = mapal_syntax::parse(&src);
    assert!(
        po.diagnostics.is_empty(),
        "{}: {:?}",
        file.display(),
        po.diagnostics
    );
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
            if p.extension().is_some_and(|x| x == "mapal")
                && p.file_name().unwrap() != "vector.mapal"
            {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

/// The four forms, as `mapal dump-ir --mermaid=<form>` prints them.
fn forms(ir: &CategoryIr, src: &str) -> [(&'static str, String); 4] {
    [
        ("category", ir.to_mermaid_category()),
        (
            "category-condensed",
            ir.to_mermaid_category_condensed_src(src),
        ),
        ("string", ir.to_mermaid_string()),
        ("string-condensed", ir.to_mermaid_string_condensed_src(src)),
    ]
}

fn golden(name: &str, file: &str) {
    let src = std::fs::read_to_string(root().join(file)).unwrap();
    let ir = ir_of(&root().join(file), true);
    for (form, got) in forms(&ir, &src) {
        assert!(
            lint_mermaid(&got).is_empty(),
            "{name} {form}: {:?}",
            lint_mermaid(&got)
        );
        let path = root()
            .join("crates/mapal-cli/tests/golden")
            .join(format!("{name}.{form}.mmd"));
        if std::env::var("MAPAL_BLESS").is_ok() {
            std::fs::write(&path, &got).unwrap();
        }
        let want = std::fs::read_to_string(&path).expect("golden missing (MAPAL_BLESS=1)");
        assert_eq!(got, want, "{name} {form} changed (MAPAL_BLESS=1 to accept)");
    }
}

#[test]
fn golden_pipeline_category_and_string() {
    golden("pipeline", "examples/pipeline.mapal");
}

#[test]
fn golden_softmax_category_and_string() {
    golden("softmax", "examples/softmax.mapal");
}

#[test]
fn golden_dense_category_and_string() {
    golden("dense", "docs/exposition/dense.mapal");
}

#[test]
fn every_program_lints_clean_in_all_four_forms_and_both_stages() {
    for p in programs() {
        let src = std::fs::read_to_string(&p).unwrap();
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            for (form, text) in forms(&ir, &src) {
                let lints = lint_mermaid(&text);
                assert!(
                    lints.is_empty(),
                    "{} {form} rewritten={rewritten}: {lints:?}",
                    p.display()
                );
                let again = forms(&ir, &src);
                assert!(
                    again.iter().any(|(f, t)| *f == form && *t == text),
                    "{form} is deterministic"
                );
            }
        }
    }
}

fn ends(d: &Diagram) -> (HashSet<&str>, HashSet<&str>) {
    let nodes: HashSet<&str> = d.nodes.iter().map(|n| n.id.as_str()).collect();
    let subs: HashSet<&str> = d.subs.iter().skip(1).map(|s| s.id.as_str()).collect();
    (nodes, subs)
}

/// Mode 1: every arrow joins two drawn nodes; no arrow ends in a product object
/// except the one pairing from the function's `Γ`; every product has its pairing
/// and every other morphism is exactly one arrow.
#[test]
fn category_arrows_have_one_source_and_one_target_and_products_come_from_gamma() {
    let mut pairings = 0;
    for p in programs() {
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            let d = ir.category_diagram();
            let (nodes, _) = ends(&d);
            let by_id: HashMap<&str, &mapal_ir::DNode> =
                d.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
            for e in &d.edges {
                let ctx = format!(
                    "{} rewritten={rewritten}: {} -> {}",
                    p.display(),
                    e.from,
                    e.to
                );
                assert!(
                    nodes.contains(e.from.as_str()) && nodes.contains(e.to.as_str()),
                    "{ctx}"
                );
                let to = by_id[e.to.as_str()];
                if to.kind == NodeKind::Product {
                    assert_eq!(e.kind, EdgeKind::Pairing, "{ctx}: an arrow into a product");
                    let from = by_id[e.from.as_str()];
                    assert_eq!(from.kind, NodeKind::Context, "{ctx}: a pairing not from Γ");
                    assert_eq!(from.sub, to.sub, "{ctx}: Γ of another function");
                    pairings += 1;
                } else {
                    assert_ne!(e.kind, EdgeKind::Pairing, "{ctx}");
                }
                assert!(!e.label.contains("Pair "), "{ctx}: a Pair i/n pseudo-arrow");
            }
            // one pairing per product object, one arrow per other morphism
            let products: HashSet<_> = ir
                .morphisms()
                .filter(|(_, m)| matches!(m.op, Operation::Pair { .. }))
                .map(|(_, m)| m.target)
                .collect();
            let n_pair = d
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Pairing)
                .count();
            assert_eq!(n_pair, products.len(), "{}: pairings", p.display());
            let n_other = ir
                .morphisms()
                .filter(|(_, m)| !matches!(m.op, Operation::Pair { .. }))
                .count();
            assert_eq!(
                d.edges.len() - n_pair,
                n_other,
                "{}: one arrow per morphism",
                p.display()
            );
            assert_eq!(
                d.nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Context)
                    .count(),
                ir.funcs().count()
            );
        }
    }
    assert!(pairings > 100, "saw {pairings} pairings");
}

/// Mode 1, condensed: the same laws for the arrows that stay (a product is drawn
/// only where several named values feed one step, and then it comes from `Γ`).
#[test]
fn condensed_category_keeps_the_laws() {
    for p in programs() {
        let src = std::fs::read_to_string(&p).unwrap();
        let ir = ir_of(&p, true);
        let d = ir.category_condensed_diagram(Some(&src));
        let (nodes, _) = ends(&d);
        let by_id: HashMap<&str, &mapal_ir::DNode> =
            d.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        for e in &d.edges {
            assert!(
                nodes.contains(e.from.as_str()) && nodes.contains(e.to.as_str()),
                "{}: {e:?}",
                p.display()
            );
            if by_id[e.to.as_str()].kind == NodeKind::Product {
                assert_eq!(e.kind, EdgeKind::Pairing, "{}: {e:?}", p.display());
                assert_eq!(
                    by_id[e.from.as_str()].kind,
                    NodeKind::Context,
                    "{}: {e:?}",
                    p.display()
                );
            }
        }
    }
}

/// Mode 2: every morphism but the pairing and the identity is exactly one box;
/// every object but a product (wires into the box) and the identity's output
/// (the source's wire continues) is exactly one wire, plus a `Δ` copy exactly
/// when it has two or more readers; nothing draws a `Pair`.
#[test]
fn string_every_morphism_is_one_box_and_every_object_one_wire() {
    let (mut boxes, mut wires, mut splits) = (0, 0, 0);
    for p in programs() {
        for rewritten in [false, true] {
            let ir = ir_of(&p, rewritten);
            let d = ir.string_diagram();
            let (nodes, subs) = ends(&d);
            let at = format!("{} rewritten={rewritten}", p.display());
            for e in &d.edges {
                assert!(
                    (nodes.contains(e.from.as_str()) || subs.contains(e.from.as_str()))
                        && (nodes.contains(e.to.as_str()) || subs.contains(e.to.as_str())),
                    "{at}: {e:?}"
                );
                assert!(
                    matches!(e.kind, EdgeKind::Wire | EdgeKind::Branch),
                    "{at}: {e:?}"
                );
                assert!(!e.label.contains("Pair "), "{at}");
            }
            for (m, mo) in ir.morphisms() {
                let n = d.nodes.iter().filter(|x| x.morph == Some(m)).count()
                    + d.subs.iter().filter(|x| x.morph == Some(m)).count();
                let want =
                    usize::from(!matches!(mo.op, Operation::Pair { .. } | Operation::Output));
                assert_eq!(n, want, "{at}: morphism {:?}", mo.op);
                boxes += want;
            }
            let products: HashSet<_> = ir
                .morphisms()
                .filter(|(_, m)| matches!(m.op, Operation::Pair { .. }))
                .map(|(_, m)| m.target)
                .collect();
            let identity: HashSet<_> = ir
                .morphisms()
                .filter(|(_, m)| m.op == Operation::Output)
                .map(|(_, m)| m.target)
                .collect();
            for (o, obj) in ir.objects() {
                if ir.try_owner(o).is_none() {
                    continue;
                }
                let main = d
                    .edges
                    .iter()
                    .filter(|e| e.kind == EdgeKind::Wire && e.obj == Some(o))
                    .count();
                let branches = d
                    .edges
                    .iter()
                    .filter(|e| e.kind == EdgeKind::Branch && e.obj == Some(o))
                    .count();
                let split = d
                    .nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Split && n.obj == Some(o))
                    .count();
                if products.contains(&o) || identity.contains(&o) {
                    assert_eq!(main + split, 0, "{at}: product / identity {o:?} has a wire");
                    continue;
                }
                assert_eq!(
                    main, 1,
                    "{at}: object {:?} ({:?}) has {main} wires",
                    obj.name, obj.kind
                );
                // branches: the copy's, plus a merge's feeds (a loop state has two writers)
                let merge = d
                    .nodes
                    .iter()
                    .filter(|n| n.kind == NodeKind::Merge && n.obj == Some(o))
                    .count();
                if split == 1 {
                    splits += 1;
                    assert!(
                        branches >= 2 + if merge == 1 { 2 } else { 0 },
                        "{at}: copy of one reader"
                    );
                } else {
                    assert_eq!(split, 0);
                    assert_eq!(
                        branches,
                        if merge == 1 { ir.in_edges(o).len() } else { 0 },
                        "{at}: stray branches"
                    );
                }
                if obj.kind == ObjectKind::Constant {
                    assert!(
                        d.nodes
                            .iter()
                            .any(|n| n.kind == NodeKind::Terminal && n.obj == Some(o)),
                        "{at}: constant end"
                    );
                }
                wires += 1;
            }
            // a product is wires into a box: every Δ and wire label is slot-ordered, never `Pair`
            assert!(d.nodes.iter().all(|n| !n.label.contains("Pair")), "{at}");
        }
    }
    assert!(
        boxes > 500 && wires > 500 && splits > 50,
        "saw {boxes} boxes, {wires} wires, {splits} copies"
    );
}
