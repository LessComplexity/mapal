//! S53 `resident` / `devgen` (plan-s53 §2.3): where each array's device copy
//! lives between GPU sites, pinned on the emitted host text. The placement is
//! the claim, so every test reads it off the `mapal_gpu_*` calls in order; the
//! values are the box differential's job (16/16 on three parts, both faces).
//! The S53 pins take the S54 `arena` rung's OFF arm (per-object buffers and
//! host slots, byte for byte); the arena's own pins follow them.

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
            host: "generic",
            target: GPU,
            gpu,
            ..EmitOpts::exact()
        },
    )
    .expect("emits")
}

/// The S53 path: the shipped rungs with `arena` forced off.
fn on() -> GpuArms {
    off("arena")
}

/// The shipped emission, `arena` included; S54's R7 host rungs (`streams`,
/// `pinned`, `pdl`) held off — they have their own tests (`tests/streams.rs`).
fn shipped() -> GpuArms {
    offs(&[])
}

/// The measurement arm with each of `rungs` forced off (and R7 held off).
fn offs(rungs: &[&str]) -> GpuArms {
    let mut a = GpuArms::default();
    for r in rungs.iter().chain(&["streams", "pinned", "pdl"]) {
        *a.off.rung(r).expect("a rung") = true;
    }
    a
}

fn off(rung: &str) -> GpuArms {
    offs(&[rung])
}

fn no_devgen() -> GpuArms {
    offs(&["devgen", "arena"])
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
                "h2d" | "alloc" | "arena" => format!("{name} {bytes}"),
                "d2h" => format!("d2h {bytes}"),
                // S54 `l2-window`: a cache policy, not a transfer.
                "launch_l2" => "launch".to_owned(),
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
    let launch = main.find("@mapal_gpu_launch").expect("launch");
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
        .position(|(_, t)| t.contains("@mapal_gpu_launch"))
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
fn devgen_refuses_a_float_remainder() {
    // A float `%` lowers to `fmodf` on the host and to `fma(-y, trunc(x / y),
    // x)` on NVPTX: different bits, so a body with one is never device-built.
    // `a` is copied in; `b` (no float `%`) is still device-built.
    let src = e2e_src().replace(
        "(t * 7 + 13) % 101 - 50 -> widen_f32 }",
        "(t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v % 0.7 }",
    );
    assert_ne!(src, e2e_src(), "the operand was rewritten");
    let t = transfers(&emit(&src, on()));
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
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
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

// --- S54 `arena` (R6) ------------------------------------------------------

/// The `add i64 %base, OFF` of every arena member, in emission order.
fn offsets(ll: &str) -> Vec<u64> {
    let lines: Vec<&str> = mapal_main(ll).lines().collect();
    lines
        .windows(2)
        .filter(|w| w[0].contains("@mapal_gpu_arena("))
        .map(|w| {
            w[1].rsplit(", ")
                .next()
                .expect("offset")
                .parse()
                .expect("a literal")
        })
        .collect()
}

/// The `(a, b, o, window)` SSA names of every launch in `mapal_main`.
fn launch_args(ll: &str) -> Vec<(String, String, String, Option<String>)> {
    mapal_main(ll)
        .lines()
        .filter(|l| l.contains("@mapal_gpu_launch"))
        .map(|l| {
            let ptrs: Vec<String> = l
                .split("i64 ")
                .skip(2)
                .map(|s| s.split([',', ')']).next().unwrap_or("").to_owned())
                .collect();
            let win = l.contains("_l2(").then(|| ptrs[3].clone());
            (ptrs[0].clone(), ptrs[1].clone(), ptrs[2].clone(), win)
        })
        .collect()
}

#[test]
fn arena_lays_disjoint_lives_on_shared_bytes() {
    let ll = emit(&chain_src(), shipped());
    // a, b, e (device-built), c = a·b, d = c·e; 4096 floats each. d is born
    // after a and b die (at c's site): it takes a's bytes. c's site reads a
    // and b while writing c, and d's reads c and e: those never share.
    let n = 4096 * 4;
    assert_eq!(offsets(&ll), [0, n, 2 * n, 3 * n, 0]);
    let t = transfers(&ll);
    assert!(!t.iter().any(|x| x.starts_with("alloc")), "{t:?}");
    assert!(
        t.iter()
            .filter(|x| x.starts_with("arena"))
            .all(|x| *x == format!("arena {}", 4 * n)),
        "{t:?}"
    );
    // No free at a death: the arena goes once, at exit.
    let main = mapal_main(&ll);
    assert_eq!(main.matches("@mapal_gpu_free(").count(), 1, "{main}");
    assert!(main.contains("call void @mapal_gpu_free(ptr %darena)"));
    // The OFF arm is S53's: one buffer per object, freed at its death.
    let off = transfers(&emit(&chain_src(), on()));
    assert_eq!(off.iter().filter(|x| x.starts_with("alloc")).count(), 5);
    assert!(!off.iter().any(|x| x.starts_with("arena")));
}

#[test]
fn arena_device_only_objects_have_no_host_slot() {
    // e2e: `ta`, `krange` (never built), a, b (device-built), c (kept; one
    // element read). None has host bytes; `c[0]` lands in a scratch cell.
    let ll = emit(&e2e_src(), shipped());
    let main = mapal_main(&ll);
    assert!(
        !main.contains("alloca [4096 x") && !main.contains("alloca [64 x"),
        "{main}"
    );
    assert!(!main.contains("@mapal_rt_alloc("), "{main}");
    assert!(main.contains("call void @mapal_gpu_d2h(ptr %s"), "{main}");
    let ll = emit(&e2e_src(), on());
    let main = mapal_main(&ll);
    assert!(main.contains("alloca [4096 x float]") && main.contains("alloca [4096 x i32]"));
    // A copied-in operand has host bytes: devgen off, a and b keep their slots
    // (and `ta`, which their host loops read); c and `krange` still have none.
    let ll = emit(&e2e_src(), off("devgen"));
    let main = mapal_main(&ll);
    assert_eq!(main.matches("alloca [4096 x float]").count(), 2, "{main}");
    assert_eq!(main.matches("alloca [4096 x i32]").count(), 1, "{main}");
    assert!(!main.contains("alloca [64 x"), "{main}");
}

#[test]
fn arena_keeps_loop_scoped_and_escaping_objects_in_their_own_buffers() {
    // `w` is read inside the loop (dead at a loop position); `x0` is the
    // loop's borrowed init (escapes); `x` is the carry. None is in the arena.
    let on_ = transfers(&emit(&loop_src(), off("devgen")));
    let off_ = transfers(&emit(&loop_src(), no_devgen()));
    assert!(!on_.iter().any(|x| x.starts_with("arena")), "{on_:?}");
    assert_eq!(on_, off_);
}

#[test]
fn golden_arena_host_modules() {
    insta::assert_snapshot!("arena_e2e", emit(&e2e_src(), shipped()));
    insta::assert_snapshot!("arena_chain", emit(&chain_src(), shipped()));
}

/// S54 `l2-window`: in a chain, the first site's output is what the next
/// launch reads (reuse distance one launch), so its window covers the output;
/// the last site's output goes to the host, so its window is the operand the
/// most blocks read: `a` (each row-panel read by the grid's 4 columns) over
/// `b` (2 rows) on this 64×64 site's 4×2 grid. Off: no window, the S53 text.
/// Forced on: S56's H100 PCIe table leaves it off on both faces. S62
/// `fill-rows` held off: it is what keeps this site's grid 4×2.
#[test]
fn l2_window_covers_the_chain_intermediate_then_the_most_read_operand() {
    let mut forced = on();
    *forced.on.rung("l2-window").expect("a rung") = true;
    forced.off.fill_rows = true;
    let ll = emit(&chain_src(), forced);
    for contract in [false, true] {
        let shipped = emit_with_opts(
            &lower_src(&chain_src()),
            &EmitOpts {
                host: "generic",
                target: GPU,
                contract,
                gpu: on(),
                ..EmitOpts::exact()
            },
        )
        .expect("emits");
        assert!(!shipped.contains("launch_l2"), "contract={contract}: OFF");
    }
    let l = launch_args(&ll);
    assert_eq!(l.len(), 2, "{ll}");
    assert!(
        mapal_main(&ll).contains("i32 4, i32 2, "),
        "a 4×2 grid:\n{ll}"
    );
    assert_eq!(
        l[0].3.as_ref(),
        Some(&l[0].2),
        "c, read by the next launch:\n{ll}"
    );
    assert_eq!(l[1].3.as_ref(), Some(&l[1].0), "the last site's a:\n{ll}");
    assert!(ll.contains("declare void @mapal_gpu_launch_l2("));
    let off = emit(&chain_src(), off("l2-window"));
    assert!(launch_args(&off).iter().all(|x| x.3.is_none()), "{off}");
    assert!(!off.contains("launch_l2"), "{off}");
}

// --- S65 rule R: the loop state moves pointers, never bytes -------------------

/// `; mapal.move` loops of `ll`: (the `carried=` field, `llvm.memcpy` calls
/// from the header to the latch — what every trip copies).
fn trip_moves(ll: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for f in ll.split("\ndefine ").skip(1) {
        let lines: Vec<&str> = f.lines().collect();
        for (h, l) in lines.iter().enumerate() {
            let Some(rest) = l.strip_prefix("; mapal.move loop=") else {
                continue;
            };
            let header = rest.split_whitespace().next().expect("header");
            let carried = rest
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix("carried="))
                .expect("carried")
                .to_string();
            let back = format!("br label %{header}");
            let latch = (h..lines.len())
                .rev()
                .find(|&i| lines[i].trim() == back)
                .expect("latch");
            let copies = lines[h..=latch]
                .iter()
                .filter(|x| x.contains("@llvm.memcpy"))
                .count();
            out.push((carried, copies));
        }
    }
    out
}

/// One loop over the state `x` (from `x0`) and the loop-invariant `w`.
fn state_loop(body: &str) -> String {
    format!(
        "fn main() {{
    4096 -> iota -> ta;
    64 -> iota -> krange;
{}{}    mut x: [f32; 4096] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < 8) -> {{
            -true-> {{
{body}                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
    out[4095] -> println;
}}
",
        operand("x0", 7),
        operand("w", 11),
    )
}

fn cpu(src: &str, gpu: GpuArms) -> String {
    emit_with_opts(
        &lower_src(src),
        &EmitOpts {
            target: "generic",
            gpu,
            ..EmitOpts::exact()
        },
    )
    .expect("emits")
}

/// The CPU build of a loop whose site reads the state it replaces
/// (`x = site(x, w)`): the next state is written while the old one is read,
/// so it takes the buffer the state does not point at (two colors) — the
/// init's, dead at the entry, is the second. No byte crosses the back edge;
/// the S64 arm copies the state three times a trip.
#[test]
fn state_read_by_its_writer_ping_pongs_without_copies() {
    let src = state_loop(&site("x", "x", "w", 64).replacen("    ", "                ", 1));
    let on = cpu(&src, GpuArms::default());
    assert_eq!(trip_moves(&on), [("0:16384:two".to_string(), 0)], "{on}");
    assert!(!on.contains("%ls"), "the second buffer is the init's:\n{on}");
    let s64 = cpu(&src, off("by-ref"));
    assert_eq!(trip_moves(&s64), [("0:16384:value".to_string(), 3)]);
}

/// `y = site(x, w); x' = relu(y)`: the state dies at the site, before its
/// next value is written — one buffer, the next state's own.
#[test]
fn state_dead_before_its_next_is_written_keeps_one_buffer() {
    let body = format!(
        "{}                y -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> x;\n",
        site("y", "x", "w", 64).replacen("    ", "                ", 1)
    );
    let on = cpu(&state_loop(&body), GpuArms::default());
    assert_eq!(trip_moves(&on), [("0:16384:one".to_string(), 0)], "{on}");
}

/// A loop writing one row of its state per trip (`x[t*d + j] <- v`, the
/// decode shape): the update chain writes the state's buffer in place (the
/// init dies at the entry, so the buffer is the init's), and the one-row
/// `o`, dead before its next is computed, keeps one buffer. Nothing is copied
/// per trip; S64 copied the state, its views and its row update every trip.
#[test]
fn row_writes_into_the_state_land_in_place() {
    let on = cpu(DECODE_16_8, GpuArms::default());
    assert_eq!(
        trip_moves(&on),
        [("0:512:inplace,1:32:one".to_string(), 0)],
        "{on}"
    );
    let s64 = cpu(DECODE_16_8, off("by-ref"));
    let (carried, copies) = &trip_moves(&s64)[0];
    assert_eq!(carried, "0:512:value,1:32:value");
    assert!(*copies >= 5, "{copies}");
}

/// On the device build a state the device carries has no host bytes: its
/// views, its next state and its kept exit are host-free (no array alloca),
/// the products forward a null pointer, and the trip copies nothing.
#[test]
fn device_carried_state_has_no_host_bytes() {
    let ll = emit(&loop_src(), shipped());
    assert_eq!(
        trip_moves(&ll),
        [("0:16384:device".to_string(), 0)],
        "{ll}"
    );
    let main = mapal_main(&ll);
    assert_eq!(main.matches("alloca [4096 x float]").count(), 0, "{main}");
    assert!(main.contains("store ptr null"), "{main}");
}

const DECODE_16_8: &str = r#"fn main() {
    64 -> iota -> tw;
    tw -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 128.0 } -> wk;
    tw -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 128.0 } -> wv;
    128 -> iota -> tc;
    8 -> iota -> krange;
    16 -> iota -> rrange;
    (0.0, 128) -> fill -> x0;
    (0.0, 8) -> fill -> o0;
    mut x: [f32; 128] <- x0;
    mut o: [f32; 8] <- o0;
    mut t: i32 <- 0;
    loop {
        (t < 16) -> {
            -true-> {
                ((t * 8 + 0) * 7 + 13) % 101 - 50 -> widen_f32 -> f0;
                f0 / 64.0 + o[0] -> v0;
                x[t * 8 + 0] <- v0;
                ((t * 8 + 1) * 7 + 13) % 101 - 50 -> widen_f32 -> f1;
                f1 / 64.0 + o[1] -> v1;
                x[t * 8 + 1] <- v1;
                tc -> map { c ->
                    c / 8 -> i;
                    c % 8 -> j;
                    (0.0, krange) -> fold { acc, k -> acc + x[i * 8 + k] * wk[k * 8 + j] }
                } -> kk;
                tc -> map { c ->
                    c / 8 -> i;
                    c % 8 -> j;
                    (0.0, krange) -> fold { acc, k -> acc + x[i * 8 + k] * wv[k * 8 + j] }
                } -> vv;
                krange -> map { k -> ((t * 8 + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / 128.0 } -> q;
                rrange -> map { r -> (0.0, krange) -> fold { acc, k -> acc + q[k] * kk[r * 8 + k] } } -> s;
                krange -> map { j -> (0.0, rrange) -> fold { acc, r -> acc + s[r] * vv[r * 8 + j] } } -> o;
                t + 1 -> t;
                -> loop;
            }
            -false-> o -> out;
        }
    }
    out[0] -> println;
    out[7] -> println;
}
"#;
