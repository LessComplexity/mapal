//! S54 `devmap` / `cell-law` (ARCHITECTURE §5 R2): a per-cell map runs as its
//! own device kernel exactly when its operands are already in device memory,
//! and a closed-form capture read at the cell's own index is evaluated in the
//! cell. Placement is the claim, so every test reads it off the emitted
//! `mapal_gpu_*` calls; values are the box differential's job (`px_*`, both
//! faces, every arm).

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;

const GPU: &str = "rtx-4090";

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit_arms(src: &str, mut gpu: GpuArms, contract: bool) -> Emission {
    // R2 alone (S54 integration): R1 (`epilogue`) takes these programs'
    // per-cell steps into the sites' stores first, and `arena` lays out their
    // allocations; each item's own tests pin those.
    for r in ["epilogue", "arena", "streams", "pinned", "pdl"] {
        *gpu.off.rung(r).expect("a rung") = true;
    }
    emit_detailed(
        &lower_src(src),
        &EmitOpts {
            target: GPU,
            gpu,
            contract,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn emit(src: &str, off: &[&str]) -> Emission {
    let mut a = GpuArms::default();
    for r in off {
        *a.off.rung(r).expect("a rung") = true;
    }
    emit_arms(src, a, false)
}

fn mapal_main(ll: &str) -> &str {
    let start = ll
        .find("define internal void @mapal_main(")
        .expect("mapal_main");
    let end = start + ll[start..].find("\n}\n").expect("mapal_main end") + 3;
    &ll[start..end]
}

/// `mapal_main`'s runtime calls in order: the op, and the byte count of a copy.
fn calls(ll: &str) -> Vec<String> {
    mapal_main(ll)
        .lines()
        .filter_map(|l| {
            let at = l.find("@mapal_gpu_")?;
            let op = &l[at + "@mapal_gpu_".len()..];
            let name = &op[..op.find('(')?];
            let last = op.rsplit("i64 ").next().unwrap_or_default();
            Some(match name {
                "h2d" | "d2h" => format!("{name} {}", last.trim_end_matches(')')),
                other => other.to_owned(),
            })
        })
        .collect()
}

fn count(c: &[String], op: &str) -> usize {
    c.iter().filter(|x| x.as_str() == op).count()
}

/// Host loops of at least `n` trips left in `mapal_main` (per-cell work).
fn host_loops(ll: &str, n: u64) -> usize {
    mapal_main(ll)
        .lines()
        .filter_map(|l| {
            l.split("icmp uge i64 ")
                .nth(1)?
                .split(", ")
                .nth(1)?
                .parse::<u64>()
                .ok()
        })
        .filter(|&t| t >= n)
        .count()
}

const N: u64 = 64;

fn head() -> String {
    format!(
        "fn main() {{\n    {} -> iota -> ta;\n    {N} -> iota -> krange;\n",
        N * N
    )
}

fn operand(name: &str, mul: u32) -> String {
    format!(
        "    ta -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> {name};\n"
    )
}

fn site(out: &str, a: &str, b: &str) -> String {
    format!(
        "    ta -> map {{ t ->
        t / {N} -> i;
        t % {N} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + {a}[i * {N} + k] * {b}[k * {N} + j] }}
    }} -> {out};\n"
    )
}

fn corners(o: &str) -> String {
    format!(
        "    {o}[0] -> println;\n    {o}[{}] -> println;\n}}\n",
        N * N - 1
    )
}

fn whole(o: &str) -> String {
    format!(
        "    0.0 -> zero: f32;\n    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> s;\n    s -> println;\n}}\n"
    )
}

/// `ew`: three per-cell maps, both inputs closed-form, no tile site.
fn ew(tail: &str) -> String {
    head()
        + &operand("a", 7)
        + &operand("b", 11)
        + "    a -> map { v -> v * 3.0 } -> x;\n"
        + "    x -> map { v -> v + 2.0 } -> y;\n"
        + "    ta -> map { t -> y[t] * b[t] } -> z;\n"
        + tail
}

#[test]
fn ew_read_at_corners_is_one_cell_kernel_with_its_operands_evaluated() {
    let em = emit(&ew(&corners("z")), &[]);
    let c = calls(&em.text);
    assert_eq!(count(&c, "launchv"), 1, "{c:?}");
    // `cell-law`: y (the rewritten x∘a chain) and b are closed-form and read
    // at the cell's own index: evaluated in the cell, built nowhere.
    assert_eq!(count(&c, "gen"), 0, "{c:?}");
    assert!(!c.iter().any(|x| x.starts_with("h2d")), "{c:?}");
    assert_eq!(
        c.iter()
            .filter(|x| x.starts_with("d2h"))
            .collect::<Vec<_>>(),
        ["d2h 4", "d2h 4"],
        "the corners only"
    );
    assert_eq!(host_loops(&em.text, N * N), 0, "no per-cell host loop");
    let dev = em.device.expect("a device unit");
    assert_eq!(dev.matches("; cell\n").count(), 1);
    assert!(
        dev.contains("_cell({ ptr, ptr, i32 } %a, float %v0, float %v1)"),
        "{dev}"
    );
    // The kernel takes the output only: nothing is loaded.
    assert!(
        dev.contains("(ptr addrspace(1) %o) \"nvvm.maxntid\"=\"256\""),
        "{dev}"
    );
}

#[test]
fn ew_without_cell_law_loads_the_devgen_buffers() {
    let em = emit(&ew(&corners("z")), &["cell-law"]);
    let c = calls(&em.text);
    assert_eq!((count(&c, "gen"), count(&c, "launchv")), (2, 1), "{c:?}");
    assert!(!c.iter().any(|x| x.starts_with("h2d")), "{c:?}");
    assert_eq!(host_loops(&em.text, N * N), 0);
    let dev = em.device.expect("a device unit");
    assert!(!dev.contains("_cell("), "the host body, unchanged");
    assert!(
        dev.contains("ptr addrspace(1) %c0, ptr addrspace(1) %c1)"),
        "{dev}"
    );
}

#[test]
fn ew_read_whole_stays_on_the_host_byte_for_byte() {
    // R2's negative control: every operand is closed-form (none lives only on
    // the device) and the host reads z whole, so a device z costs a whole
    // copy-out that the host loop does not. The text is the OFF arm's.
    let src = ew(&whole("z"));
    let on = emit(&src, &[]);
    let off = emit(&src, &["devmap"]);
    assert_eq!(on.text, off.text);
    assert_eq!(on.device, off.device);
    assert!(!on.text.contains("@mapal_gpu_launchv"));
}

#[test]
fn off_arm_has_no_cell() {
    for src in [ew(&corners("z")), mlp(&corners("d"))] {
        let em = emit(&src, &["devmap"]);
        assert!(!em.text.contains("@mapal_gpu_launchv"));
        assert!(!em.text.contains("declare void @mapal_gpu_launchv"));
        assert!(em.device.is_none_or(|d| !d.contains("; cell\n")));
    }
}

/// `mlp`: a site, a per-cell step, a site.
fn mlp(tail: &str) -> String {
    head()
        + &operand("a", 7)
        + &operand("b", 11)
        + &operand("e", 5)
        + &site("c", "a", "b")
        + "    c -> map { v -> v * 0.5 } -> h;\n"
        + &site("d", "h", "e")
        + tail
}

#[test]
fn mlp_keeps_the_chain_on_the_device() {
    let on = calls(&emit(&mlp(&corners("d")), &[]).text);
    let kernels: Vec<&String> = on.iter().filter(|x| x.starts_with("launch")).collect();
    assert_eq!(kernels, ["launch", "launchv", "launch"], "{on:?}");
    assert!(!on.iter().any(|x| x.starts_with("h2d")), "{on:?}");
    assert_eq!(
        on.iter().filter(|x| x.starts_with("d2h")).count(),
        2,
        "the corners: {on:?}"
    );
    // OFF: c copied out whole, h built on the host and copied in whole.
    let off = calls(&emit(&mlp(&corners("d")), &["devmap"]).text);
    let bytes = (N * N * 4).to_string();
    assert!(off.contains(&format!("d2h {bytes}")), "{off:?}");
    assert!(off.contains(&format!("h2d {bytes}")), "{off:?}");
}

#[test]
fn a_whole_host_reader_of_a_device_operand_still_takes_the_cell() {
    // The host folds h whole, so h is copied out either way; its operand c
    // lives only on the device, so the cell saves c's copy-out and the loop.
    let src = head()
        + &operand("a", 7)
        + &operand("b", 11)
        + &site("c", "a", "b")
        + "    c -> map { v -> v * 0.5 + 1.0 } -> h;\n"
        + &whole("h");
    let em = emit(&src, &[]);
    let c = calls(&em.text);
    let bytes = (N * N * 4).to_string();
    assert_eq!(count(&c, "launchv"), 1, "{c:?}");
    assert_eq!(
        c.iter()
            .filter(|x| x.starts_with("d2h"))
            .collect::<Vec<_>>(),
        [&format!("d2h {bytes}")],
        "h, once; never c"
    );
}

#[test]
fn an_operand_only_the_host_holds_keeps_the_map_on_the_host() {
    // `u`'s body divides by a non-literal: unclean, so neither devgen nor a
    // cell may build it, and copying it in for one map is the host loop's
    // cost moved, not removed.
    let src = head()
        + "    ta -> map { t -> (t * 7 + 13) / (t % 5 + 1) -> widen_f32 } -> u;\n"
        + "    u -> map { v -> v * 2.0 } -> w;\n"
        + &corners("w");
    let on = emit(&src, &[]);
    assert_eq!(on.text, emit(&src, &["devmap"]).text);
    assert!(!on.text.contains("@mapal_gpu_"));
}

#[test]
fn a_float_remainder_body_is_never_a_cell() {
    let src = head()
        + &operand("a", 7)
        + &operand("b", 11)
        + &site("c", "a", "b")
        + "    c -> map { v -> v % 0.7 } -> h;\n"
        + &corners("h");
    let c = calls(&emit(&src, &[]).text);
    assert_eq!(count(&c, "launchv"), 0, "{c:?}");
    assert!(
        c.iter().any(|x| x == &format!("d2h {}", N * N * 4)),
        "c copied out for the host map"
    );
}

/// A loop whose next state is a per-cell map of a site's output (`pt_loop`).
fn loop_relu() -> String {
    head()
        + &operand("x0", 7)
        + &operand("w", 11)
        + &format!(
            "    mut x: [f32; {}] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < 4) -> {{
            -true-> {{
{}                y -> map {{ v -> (v > 0.0) -> {{ -true-> v; -false-> 0.0; }} }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
",
            N * N,
            site("y", "x", "w").replace("    ", "                ")
        )
        + &corners("out")
}

/// A loop whose carried state feeds a per-cell map before the site.
fn loop_pre() -> String {
    head()
        + &operand("x0", 7)
        + &operand("w", 11)
        + &format!(
            "    mut x: [f32; {}] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < 4) -> {{
            -true-> {{
                x -> map {{ v -> v * 0.5 }} -> xs;
{}                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
",
            N * N,
            site("x", "xs", "w").replace("    ", "                ")
        )
        + &corners("out")
}

/// The calls inside the loop body: between the loop's first and last launch.
fn in_loop(c: &[String]) -> &[String] {
    let first = c
        .iter()
        .position(|x| x.starts_with("launch"))
        .expect("a launch");
    let last = c
        .iter()
        .rposition(|x| x.starts_with("launch"))
        .expect("a launch");
    &c[first..=last]
}

#[test]
fn a_loop_whose_next_state_is_a_cell_carries_it_on_the_device() {
    let c = calls(&emit(&loop_relu(), &[]).text);
    let body = in_loop(&c);
    assert_eq!(
        body.iter().filter(|x| x.starts_with("launch")).count(),
        2,
        "{c:?}"
    );
    assert!(
        !body
            .iter()
            .any(|x| x.starts_with("h2d") || x.starts_with("d2h")),
        "no copy inside the loop: {c:?}"
    );
    // OFF: the relu runs on the host, so y comes out whole every trip.
    let off = calls(&emit(&loop_relu(), &["devmap"]).text);
    assert!(off.contains(&format!("d2h {}", N * N * 4)), "{off:?}");
}

#[test]
fn a_map_over_a_carry_view_before_a_site_is_a_cell() {
    // The view is an operand of the cell and the cell's output feeds the
    // site: carries and cells decide each other (cell_sites' fixed point).
    let c = calls(&emit(&loop_pre(), &[]).text);
    let body = in_loop(&c);
    assert_eq!(
        body.iter()
            .filter(|x| x.starts_with("launch"))
            .cloned()
            .collect::<Vec<_>>(),
        ["launchv", "launch"],
        "{c:?}"
    );
    assert!(
        !body
            .iter()
            .any(|x| x.starts_with("h2d") || x.starts_with("d2h")),
        "{c:?}"
    );
}

#[test]
fn the_contract_face_places_the_same_cells() {
    for src in [ew(&corners("z")), mlp(&corners("d")), loop_relu()] {
        let exact = calls(&emit_arms(&src, GpuArms::default(), false).text);
        let contract = calls(&emit_arms(&src, GpuArms::default(), true).text);
        assert_eq!(exact, contract);
    }
}

#[test]
fn golden_devmap_device_units() {
    let ew = emit(&ew(&corners("z")), &[]);
    insta::assert_snapshot!("devmap_ew_host", ew.text);
    insta::assert_snapshot!("devmap_ew_device", ew.device.expect("device"));
    let mlp = emit(&mlp(&corners("d")), &[]);
    insta::assert_snapshot!("devmap_mlp_host", mapal_main(&mlp.text));
}
