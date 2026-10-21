//! S53 `resident` / `devgen` (plan-s53 §2.3): where each array's device copy
//! lives between GPU sites, pinned on the emitted host text. The placement is
//! the claim, so every test reads it off the `mapal_gpu_*` calls in order; the
//! values are the box differential's job (16/16 on three parts, both faces).

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_with_opts};
use mapal_ir::CategoryIr;

const GPU: &str = "h100-pcie";

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit(src: &str, gpu: GpuArms) -> String {
    emit_with_opts(
        &lower_src(src),
        &EmitOpts {
            target: GPU,
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn on() -> GpuArms {
    GpuArms::default()
}

/// The measurement arm with `rung` forced off.
fn off(rung: &str) -> GpuArms {
    let mut a = GpuArms::default();
    *a.off.rung(rung).expect("a rung") = true;
    a
}

fn no_devgen() -> GpuArms {
    off("devgen")
}

fn mapal_main(ll: &str) -> &str {
    let start = ll
        .find("define internal void @mapal_main(")
        .expect("mapal_main");
    let end = start + ll[start..].find("\n}\n").expect("mapal_main end") + 3;
    &ll[start..end]
}

/// The runtime calls of `mapal_main`, in emission order, as `op args` with the
/// SSA names dropped — the location map's trace through the function.
fn transfers(ll: &str) -> Vec<String> {
    mapal_main(ll)
        .lines()
        .filter_map(|l| {
            let at = l.find("@mapal_gpu_")?;
            let op = &l[at + "@mapal_gpu_".len()..];
            let name = &op[..op.find('(')?];
            let bytes = op
                .rsplit("i64 ")
                .next()
                .map(|s| s.trim_end_matches(')').to_owned())
                .unwrap_or_default();
            Some(match name {
                "h2d" | "alloc" => format!("{name} {bytes}"),
                "d2h" => format!("d2h {bytes}"),
                other => other.to_owned(),
            })
        })
        .collect()
}

fn site(out: &str, a: &str, b: &str, n: u64) -> String {
    format!(
        "    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + {a}[i * {n} + k] * {b}[k * {n} + j] }}
    }} -> {out};\n"
    )
}

fn operand(name: &str, mul: u32) -> String {
    format!("    ta -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen_f32 }} -> {name};\n")
}

/// The e2e shape: two closed-form operands, one site, one element printed.
fn e2e_src() -> String {
    format!(
        "fn main() {{\n    4096 -> iota -> ta;\n    64 -> iota -> krange;\n{}{}{}    c[0] -> println;\n}}\n",
        operand("a", 7),
        operand("b", 11),
        site("c", "a", "b", 64)
    )
}

fn chain_src() -> String {
    format!(
        "fn main() {{\n    4096 -> iota -> ta;\n    64 -> iota -> krange;\n{}{}{}{}{}    d[0] -> println;\n}}\n",
        operand("a", 7),
        operand("b", 11),
        operand("e", 5),
        site("c", "a", "b", 64),
        site("d", "c", "e", 64)
    )
}

fn loop_src() -> String {
    format!(
        "fn main() {{
    4096 -> iota -> ta;
    64 -> iota -> krange;
{}{}    mut x: [f32; 4096] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < 8) -> {{
            -true-> {{
{}                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
}}
",
        operand("x0", 7),
        operand("w", 11),
        site("x", "x", "w", 64)
    )
}

#[test]
fn e2e_copies_in_after_the_producers_and_out_one_element() {
    let ll = emit(&e2e_src(), no_devgen());
    let main = mapal_main(&ll);
    // h2d sits right after each producer's fill loop, before the site.
    let a_fill = main.find("call float @fn1(").expect("a's producer");
    let b_fill = main.find("call float @fn2(").expect("b's producer");
    let h2d: Vec<usize> = main
        .match_indices("@mapal_gpu_h2d(")
        .map(|(i, _)| i)
        .collect();
    let launch = main.find("@mapal_gpu_launch(").expect("launch");
    assert_eq!(h2d.len(), 2, "one copy-in per operand");
    assert!(a_fill < h2d[0] && h2d[0] < b_fill && b_fill < h2d[1] && h2d[1] < launch);
    assert_eq!(
        transfers(&ll),
        [
            "alloc 16384",
            "h2d 16384",
            "alloc 16384",
            "h2d 16384",
            "alloc 16384",
            "launch",
            // a and b die at the site: freed right after it.
            "free",
            "free",
            // c is read by `c[0]` only: four bytes, then c dies.
            "d2h 4",
            "free",
            // function exit: every slot once more (a no-op on freed slots).
            "free",
            "free",
            "free",
        ]
    );
    assert!(
        !ll.contains("@mapal_gpu_tile("),
        "the S52 round trip is the OFF arm"
    );
}

#[test]
fn chain_copies_each_input_once_and_never_moves_c() {
    let ll = emit(&chain_src(), no_devgen());
    let t = transfers(&ll);
    assert_eq!(
        t.iter().filter(|x| x.starts_with("h2d")).count(),
        3,
        "{t:?}"
    );
    assert_eq!(t.iter().filter(|x| *x == "launch").count(), 2);
    // The only copy-out is `d[0]`; `c` lives and dies on the device.
    assert_eq!(
        t.iter()
            .filter(|x| x.starts_with("d2h"))
            .collect::<Vec<_>>(),
        ["d2h 4"]
    );
}

/// `mapal_main`'s basic blocks, in order: `(label, text)`.
fn blocks(main: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for l in main.lines().skip(1) {
        if !l.starts_with(' ') && l.ends_with(':') {
            out.push((l.trim_end_matches(':').to_owned(), String::new()));
        } else if let Some(b) = out.last_mut() {
            b.1.push_str(l);
            b.1.push('\n');
        }
    }
    out
}

#[test]
fn loop_copies_w_in_outside_the_loop_and_carries_x_on_the_device() {
    let ll = emit(&loop_src(), no_devgen());
    let bs = blocks(mapal_main(&ll));
    let advance = bs
        .iter()
        .position(|(_, t)| t.contains("@mapal_gpu_launch("))
        .expect("the site's block");
    let label = &bs[advance].0;
    let header = bs
        .iter()
        .position(|(_, t)| t.contains(&format!("label %{label},")))
        .expect("the loop header branches to the body");
    for i in [header, advance] {
        assert!(
            !bs[i].1.contains("@mapal_gpu_h2d(") && !bs[i].1.contains("@mapal_gpu_d2h("),
            "no transfer inside the loop:\n{}",
            bs[i].1
        );
    }
    // Before the loop: `w` after its producer, the init state at loop entry.
    let before: String = bs[..header].iter().map(|(_, t)| t.as_str()).collect();
    assert_eq!(
        before.matches("@mapal_gpu_h2d(").count(),
        2,
        "w and x0, once each"
    );
    // The back edge swaps two device slots: the next launch writes into the
    // buffer the state just left.
    assert_eq!(
        bs[advance].1.matches("store i64").count(),
        2,
        "the carry swap"
    );
    // After the loop: only `out[0]`, four bytes.
    let after: String = bs[advance + 1..].iter().map(|(_, t)| t.as_str()).collect();
    assert_eq!(after.matches("@mapal_gpu_d2h(").count(), 1);
    assert!(after.contains("i64 4)"), "out[0] is one element");
}

#[test]
fn devgen_builds_closed_form_operands_on_the_device() {
    let ll = emit(&e2e_src(), on());
    let t = transfers(&ll);
    assert!(
        !t.iter().any(|x| x.starts_with("h2d")),
        "bytes_in = 0: {t:?}"
    );
    assert_eq!(t.iter().filter(|x| *x == "gen").count(), 2);
    // Neither the operand loops nor the index spaces are built on the host:
    // `mapal_main` has no loop left at all.
    let main = mapal_main(&ll);
    assert!(!main.contains("call float @fn1("), "a is device-built");
    assert!(!main.contains("icmp uge i64"), "no host loop:\n{main}");
}

#[test]
fn devgen_refuses_an_unclean_body() {
    // An integer division by a non-literal can trap: the body is not clean,
    // so recomputing it elsewhere could lose the trap. `a` stays a host
    // producer and is copied in; `b` (clean) is still device-built.
    let src = e2e_src().replace("(t * 7 + 13) % 101 - 50", "(t * 7 + 13) / (t % 5 + 1) - 50");
    let ll = emit(&src, on());
    let t = transfers(&ll);
    assert_eq!(
        t.iter().filter(|x| x.starts_with("h2d")).count(),
        1,
        "{t:?}"
    );
    assert_eq!(t.iter().filter(|x| *x == "gen").count(), 1, "{t:?}");
}

#[test]
fn off_arm_is_the_s52_round_trip() {
    let ll = emit(&e2e_src(), off("resident"));
    assert!(ll.contains("@mapal_gpu_tile("));
    assert!(!ll.contains("@mapal_gpu_alloc"));
    // A CPU profile never sees either.
    let cpu = emit_with_opts(
        &lower_src(&e2e_src()),
        &EmitOpts {
            target: "generic",
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    assert!(!cpu.contains("@mapal_gpu"));
}

#[test]
fn golden_resident_host_modules() {
    insta::assert_snapshot!("resident_e2e", emit(&e2e_src(), on()));
    insta::assert_snapshot!("resident_e2e_no_devgen", emit(&e2e_src(), no_devgen()));
    insta::assert_snapshot!("resident_chain", emit(&chain_src(), on()));
    insta::assert_snapshot!("resident_loop", emit(&loop_src(), no_devgen()));
    let unclean = e2e_src().replace("(t * 7 + 13) % 101 - 50", "(t * 7 + 13) / (t % 5 + 1) - 50");
    insta::assert_snapshot!("resident_unclean", emit(&unclean, on()));
}
