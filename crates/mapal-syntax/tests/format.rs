use mapal_syntax::{Severity, TokenKind, format_source, lex, parse};

fn tokens(source: &str) -> Vec<(TokenKind, &str)> {
    lex(source)
        .tokens
        .into_iter()
        .map(|t| (t.kind, &source[t.span.start as usize..t.span.end as usize]))
        .collect()
}

fn assert_stable(source: &str) -> String {
    let formatted = format_source(source).unwrap_or_else(|errors| panic!("{errors:?}\n{source}"));
    assert_eq!(tokens(source), tokens(&formatted));
    assert_eq!(format_source(&formatted).unwrap(), formatted, "{formatted}");
    assert!(parse(&formatted).diagnostics.is_empty());
    formatted
}

#[test]
fn expands_bodies_and_indents_statements() {
    assert_eq!(
        assert_stable("fn main(){1->println;}"),
        "fn main() {\n    1 -> println;\n}\n"
    );
    assert_eq!(assert_stable("fn empty(){}"), "fn empty() {\n}\n");
    assert_eq!(
        assert_stable("fn main(){xs->map{x->x*2}->ys;(0,ys)->fold{a,b->a+b}->ret;}"),
        "fn main() {\n    xs -> map {\n        x -> x * 2\n    } -> ys;\n    (0, ys) -> fold {\n        a, b -> a + b\n    } -> ret;\n}\n"
    );
}

#[test]
fn retains_comments_strings_and_literal_spelling() {
    let source = "// heading Ω\r\nfn main(){// body\r\n\t\"{ -> ; // \\\"}\"->println;// tail\r\n\t// explanation\r\n\t0007->x;1.2500->y;}// end";
    assert_eq!(
        assert_stable(source),
        "// heading Ω\nfn main() { // body\n    \"{ -> ; // \\\"}\" -> println; // tail\n    // explanation\n    0007 -> x;\n    1.2500 -> y;\n} // end\n"
    );
    assert_eq!(assert_stable("// only a comment"), "// only a comment\n");
    assert_eq!(assert_stable(" \t\r\n"), "");
}

#[test]
fn formats_records_arrays_and_nested_projections() {
    assert_eq!(
        assert_stable(
            "type Pair{x:i32,y:[i32;2]}fn main(){Pair{x:1,y:[2,3]}->p;p.y[0]->println;p.0 .1->ret;}"
        ),
        "type Pair {\n    x: i32,\n    y: [i32; 2]\n}\n\nfn main() {\n    Pair {\n        x: 1,\n        y: [2, 3]\n    } -> p;\n    p.y[0] -> println;\n    p.0 .1 -> ret;\n}\n"
    );
}

#[test]
fn formats_guard_arms_and_unary_operators() {
    assert_eq!(
        assert_stable("fn main(){(a < - 1)->{-true->{- 7 ->ret;}-false->! false;}}"),
        "fn main() {\n    (a < -1) -> {\n        -true-> {\n            -7 -> ret;\n        }\n        -false-> !false;\n    }\n}\n"
    );
    assert_stable("fn main(){a- -b->x;--b->y; - 7 ->ret;}");
}

#[test]
fn precision_separators_use_parser_validity() {
    assert_eq!(
        assert_stable("fn f()->f32:tf32|f32{xs->map:tf32|f32{x->x+1.0}->ret;}"),
        "fn f() -> f32: tf32 | f32 {\n    xs -> map: tf32 | f32 {\n        x -> x + 1.0\n    } -> ret;\n}\n"
    );
}

#[test]
fn normalizes_existing_continuations_and_paragraphs() {
    assert_eq!(
        assert_stable("fn main(){\n\t1\n->inc\n\t\t->x;\n\n\n\t(x,\ny)\n->add->ret;\n}"),
        "fn main() {\n    1\n        -> inc\n        -> x;\n\n    (x,\n        y)\n        -> add -> ret;\n}\n"
    );
    assert_stable("fn main(){xs->map{x->\n\t\t x+1\n}->ret;}");
}

#[test]
fn ordinary_identifier_chains_are_not_map_fold_parameter_lists() {
    assert_eq!(
        assert_stable("fn main(){x->\ny;}"),
        "fn main() {\n    x ->\n        y;\n}\n"
    );
    assert_eq!(
        assert_stable("fn main():tf32|f32{x->\ny;}"),
        "fn main(): tf32 | f32 {\n    x ->\n        y;\n}\n"
    );
    assert_eq!(
        assert_stable("fn main(){loop{x->\ny;}}"),
        "fn main() {\n    loop {\n        x ->\n            y;\n    }\n}\n"
    );
    assert_eq!(
        assert_stable("fn main(){b->{-true->{x->\ny;}-false->0;}}"),
        "fn main() {\n    b -> {\n        -true-> {\n            x ->\n                y;\n        }\n        -false-> 0;\n    }\n}\n"
    );
}

#[test]
fn map_fold_parameter_arrows_begin_the_body_without_continuation_indent() {
    assert_eq!(
        assert_stable("fn main(){xs->map{x->\nx}->ret;}"),
        "fn main() {\n    xs -> map {\n        x ->\n        x\n    } -> ret;\n}\n"
    );
    assert_eq!(
        assert_stable("fn main(){xs->map:tf32|f32{x->\nx}->ret;}"),
        "fn main() {\n    xs -> map: tf32 | f32 {\n        x ->\n        x\n    } -> ret;\n}\n"
    );
    assert_eq!(
        assert_stable("fn main(){(0.0,xs)->fold:tf32|f32{acc,v->\nacc+v}->ret;}"),
        "fn main() {\n    (0.0, xs) -> fold: tf32 | f32 {\n        acc, v ->\n        acc + v\n    } -> ret;\n}\n"
    );
}

#[test]
fn rejects_invalid_input_and_depth_exhaustion() {
    for source in [
        "fn main(){",
        "fn main(){/* unsupported */1->ret;}",
        "fn main(){\"unterminated\n}",
        "fn main(){1 + ->ret;}",
        "fn main(){a = 1;}",
    ] {
        let errors = format_source(source).unwrap_err();
        assert!(errors.iter().any(|d| d.severity == Severity::Error));
    }
    let deep = format!("fn main(){{{}1{}->ret;}}", "(".repeat(150), ")".repeat(150));
    assert!(format_source(&deep).is_err());
}

#[test]
fn refuses_out_of_core_labels_without_rewriting_them() {
    let source = "fn main(){:again{1->x;->:again;}}";
    let diagnostics = format_source(source).unwrap_err();
    assert_eq!(
        diagnostics.iter().filter(|d| d.code.0 == "P0110").count(),
        2,
        "both the labelled block and jump retain the parser's rejection"
    );
}

#[test]
fn bounded_generated_whitespace_preserves_program_tokens() {
    let programs = [
        "fn f(a:i32)->i32{a->map{x->x+1}->ret;}",
        "type P{x:i32,y:[i32;2]}fn f(){P{x:1,y:[2,3]}->p;p.y[1]->ret;}",
        "fn f(){loop{a-> {-true->{a+1->a;->loop;}-false->a->ret;}}}",
        "fn f():tf32|f32{xs->map{x->(0,x)->fold{acc,v->acc+v}}->ret;}",
    ];
    let gaps = [" ", "\t", "\n", "\n\n", "\r\n\t"];
    for (p, program) in programs.iter().enumerate() {
        let original = tokens(program);
        for seed in 0..48 {
            let mut generated = String::new();
            let mut state = (seed + p * 48 + 1) as u64;
            for (kind, text) in &original {
                if *kind != TokenKind::Eof {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    generated.push_str(text);
                    generated.push_str(gaps[state as usize % gaps.len()]);
                }
            }
            assert_stable(&generated);
        }
    }
}

#[test]
fn shipped_source_corpus_preserves_tokens_and_is_idempotent() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut checked = 0;
    let mut skipped = 0;
    // Deliberately bounded, nonrecursive reads. Never walk archived worktrees,
    // generated bundles or provider caches.
    for directory in ["examples", "benches/algorithms"] {
        for entry in std::fs::read_dir(root.join(directory)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("mapal") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let diagnostics = parse(&source).diagnostics;
            if diagnostics.iter().any(|d| d.severity == Severity::Error) {
                assert_eq!(
                    path,
                    root.join("examples/vector.mapal"),
                    "unexpected invalid corpus source: {diagnostics:?}"
                );
                skipped += 1;
                continue;
            }
            let formatted = format_source(&source)
                .unwrap_or_else(|errors| panic!("{}: {errors:?}", path.display()));
            assert_eq!(tokens(&source), tokens(&formatted), "{}", path.display());
            assert_eq!(
                format_source(&formatted).unwrap(),
                formatted,
                "{}",
                path.display()
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "corpus unexpectedly small: {checked}");
    eprintln!("formatter corpus: {checked} valid files checked, {skipped} invalid files skipped");
}
