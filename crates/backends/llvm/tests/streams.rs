//! S54 `streams` / `pinned` / `pdl` (R7, plan-s54 §4.3): independent device
//! sites on their own streams, page-locked crossings, dependent launches on
//! sm_90+. Each test reads the placement off the emitted `mapal_gpu_*` calls;
//! the values are the box differential's job.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit(src: &str, target: &'static str, gpu: GpuArms) -> Emission {
    emit_detailed(
        &lower_src(src),
        &EmitOpts {
            target,
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn host(src: &str, target: &'static str, gpu: GpuArms) -> String {
    emit(src, target, gpu).text
}

fn off(rungs: &[&str]) -> GpuArms {
    let mut a = GpuArms::default();
    for r in rungs {
        *a.off.rung(r).expect("a rung") = true;
    }
    a
}

fn on() -> GpuArms {
    GpuArms::default()
}

fn mapal_main(ll: &str) -> &str {
    let start = ll
        .find("define internal void @mapal_main(")
        .expect("mapal_main");
    let end = start + ll[start..].find("\n}\n").expect("mapal_main end") + 3;
    &ll[start..end]
}

/// `mapal_main`'s runtime calls in order, `op` plus the stream for the S54
/// entries (`launchv 1`, `event 1>2`, `d2h_on 2`).
fn calls(ll: &str) -> Vec<String> {
    mapal_main(ll)
        .lines()
        .filter(|l| l.contains("call ") && l.contains("@mapal_gpu_"))
        .map(|l| {
            let op = &l[l.find("@mapal_gpu_").expect("call") + "@mapal_gpu_".len()..];
            let name = &op[..op.find('(').expect("args")];
            let i32s: Vec<&str> = op
                .split("i32 ")
                .skip(1)
                .map(|x| x.split([',', ')']).next().expect("arg"))
                .collect();
            match name {
                "launchv" | "d2h_on" => format!("{name} {}", i32s[0]),
                "event" => format!("event {}>{}", i32s[0], i32s[1]),
                other => other.to_owned(),
            }
        })
        .collect()
}

fn init_flags(ll: &str) -> u32 {
    let at = ll
        .find("call void @mapal_gpu_init(i32 ")
        .expect("init call");
    let rest = &ll[at + "call void @mapal_gpu_init(i32 ".len()..];
    rest[..rest.find(')').expect(")")].parse().expect("flags")
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
    format!(
        "    ta -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> {name};\n"
    )
}

fn program(n: u64, ops: &[(&str, u32)], body: &str, outs: &[&str]) -> String {
    let mut s = format!(
        "fn main() {{\n    {} -> iota -> ta;\n    {n} -> iota -> krange;\n",
        n * n
    );
    for (o, m) in ops {
        s += &operand(o, *m);
    }
    s += body;
    for o in outs {
        s += &format!(
            "    {o}[0] -> println;\n    {o}[{}] -> println;\n",
            n * n - 1
        );
    }
    s + "}\n"
}

/// Two independent sites joined per cell on the host (gen_tiles `indep`).
fn indep(n: u64) -> String {
    program(
        n,
        &[("a", 7), ("b", 11), ("e", 5), ("f", 3)],
        &(site("c", "a", "b", n)
            + &site("d", "e", "f", n)
            + "    ta -> map { t -> c[t] + d[t] } -> g;\n"),
        &["g"],
    )
}

/// One output, two independent readers, joined per cell (`diamond`).
fn diamond(n: u64) -> String {
    program(
        n,
        &[("a", 7), ("b", 11), ("e", 5), ("g", 3)],
        &(site("c", "a", "b", n)
            + &site("d1", "c", "e", n)
            + &site("d2", "c", "g", n)
            + "    ta -> map { t -> d1[t] + d2[t] } -> h;\n"),
        &["h"],
    )
}

/// Two independent site → host map → site chains (`mlp2` without the bias).
fn mlp2(n: u64) -> String {
    program(
        n,
        &[
            ("a", 7),
            ("b", 11),
            ("e", 5),
            ("f", 3),
            ("g", 13),
            ("p", 17),
        ],
        &(site("c1", "a", "b", n)
            + "    c1 -> map { v -> v * 0.5 } -> h1;\n"
            + &site("d1", "h1", "e", n)
            + &site("c2", "f", "g", n)
            + "    c2 -> map { v -> v * 0.5 } -> h2;\n"
            + &site("d2", "h2", "p", n)),
        &["d1", "d2"],
    )
}

/// One output read by a host map and by a second site (`fanout`).
fn fanout(n: u64) -> String {
    program(
        n,
        &[("a", 7), ("b", 11), ("e", 5)],
        &(site("c", "a", "b", n)
            + "    c -> map { v -> v * 0.5 + 1.0 } -> h;\n"
            + &site("d", "c", "e", n)),
        &["c", "h", "d"],
    )
}

const H100: &str = "h100-pcie";

#[test]
fn independent_sites_take_two_streams_and_copy_out_at_the_join() {
    let ll = host(&indep(64), H100, on());
    let t = calls(&ll);
    let pos = |x: &str| {
        t.iter()
            .position(|c| c == x)
            .unwrap_or_else(|| panic!("{x} in {t:?}"))
    };
    // Both launches before either copy-out: the second chain launches before
    // the host waits for the first.
    assert!(pos("launchv 1") < pos("launchv 2"), "{t:?}");
    assert!(
        pos("launchv 2") < pos("d2h_on 1") && pos("d2h_on 1") < pos("d2h_on 2"),
        "{t:?}"
    );
    // The operands die at the sites; their frees wait for the last launch.
    let frees_before = t[..pos("launchv 2")]
        .iter()
        .filter(|c| *c == "free")
        .count();
    assert_eq!(frees_before, 0, "{t:?}");
    assert!(
        !t.contains(&"launch".to_owned()),
        "every site on a chain: {t:?}"
    );
    assert!(
        !t.iter().any(|c| c.starts_with("event")),
        "no edge between the chains: {t:?}"
    );
    // The copy-outs sit right before the host join reads them.
    let main = mapal_main(&ll);
    let last_d2h = main.rfind("@mapal_gpu_d2h_on(").expect("d2h");
    // The join's body takes `c` and `d` by pointer.
    let join = main
        .find("call float @fn")
        .expect("the host join's body call");
    assert!(last_d2h < join);
    assert_eq!(init_flags(&ll) & 2, 2, "the runtime creates the streams");
}

#[test]
fn diamond_waits_on_its_join_edge_only() {
    let t = calls(&host(&diamond(64), H100, on()));
    let launches: Vec<&String> = t.iter().filter(|c| c.starts_with("launchv")).collect();
    // c and d1 are one chain; d2 reads c from the other.
    assert_eq!(launches, ["launchv 1", "launchv 1", "launchv 2"], "{t:?}");
    let ev: Vec<&String> = t.iter().filter(|c| c.starts_with("event")).collect();
    assert_eq!(ev, ["event 1>2"], "{t:?}");
    let e = t.iter().position(|c| c == "event 1>2").expect("event");
    assert_eq!(
        t[e + 1],
        "launchv 2",
        "the wait sits right before d2: {t:?}"
    );
}

#[test]
fn two_mlps_are_two_chains_with_no_edge() {
    let t = calls(&host(&mlp2(64), H100, on()));
    let mut launches: Vec<&String> = t.iter().filter(|c| c.starts_with("launchv")).collect();
    launches.sort();
    assert_eq!(
        launches,
        ["launchv 1", "launchv 1", "launchv 2", "launchv 2"],
        "{t:?}"
    );
    assert!(!t.iter().any(|c| c.starts_with("event")), "{t:?}");
}

#[test]
fn a_single_chain_keeps_one_stream() {
    let ll = host(&fanout(64), H100, on());
    assert!(!ll.contains("@mapal_gpu_launchv("), "fanout is one chain");
    assert_eq!(init_flags(&ll) & 2, 0);
}

#[test]
fn a_filled_part_keeps_one_stream_unless_the_gate_is_off() {
    // 1024² on the H100 PCIe: each grid's last wave is too full to share one.
    let gated = host(&indep(1024), H100, on());
    assert!(!gated.contains("@mapal_gpu_launchv("));
    let forced = host(&indep(1024), H100, off(&["stream-gate"]));
    assert!(forced.contains("@mapal_gpu_launchv("));
}

#[test]
fn every_off_arm_is_the_one_stream_pageable_text() {
    let all_off = off(&["streams", "pinned", "pdl"]);
    for src in [indep(64), diamond(64), mlp2(64), fanout(64)] {
        let ll = host(&src, H100, all_off);
        for entry in ["launchv", "event", "d2h_on", "pin", "unpin"] {
            assert!(!ll.contains(&format!("@mapal_gpu_{entry}(")), "{entry}");
        }
        assert_eq!(init_flags(&ll), 1, "eager-init only");
        let dev = emit(&src, H100, all_off).device.expect("a device unit");
        assert!(!dev.contains("griddepcontrol"));
    }
    // Streams off alone: one stream, pins still placed.
    let ll = host(&indep(64), H100, off(&["streams"]));
    assert!(!ll.contains("@mapal_gpu_launchv(") && ll.contains("@mapal_gpu_pin("));
}

#[test]
fn pins_are_the_arrays_that_cross_the_bus() {
    // devgen off: every operand is uploaded; `g` is read whole on the host
    // (copied out), the corners of `g` are host reads of a host array.
    let ll = host(&indep(64), H100, off(&["devgen", "streams"]));
    let main = mapal_main(&ll);
    let lines: Vec<&str> = main.lines().collect();
    let mut pinned = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let Some(at) = l.find("@mapal_gpu_pin(ptr ") else {
            continue;
        };
        let p = l[at + "@mapal_gpu_pin(ptr ".len()..]
            .split(',')
            .next()
            .expect("ptr");
        pinned.push(p.to_owned());
        // The next transfer moves exactly the pinned array.
        let next = lines[i + 1..]
            .iter()
            .find(|x| x.contains("@mapal_gpu_h2d(") || x.contains("@mapal_gpu_d2h("))
            .expect("a copy after the pin");
        assert!(next.contains(&format!("ptr {p},")), "{l} then {next}");
    }
    // a, b, e, f up; c, d down.
    assert_eq!(pinned.len(), 6, "{pinned:?}");
    assert_eq!(main.matches("@mapal_gpu_h2d(").count(), 4);
    assert!(main.contains("call void @mapal_gpu_unpin()"));
    // devgen on: nothing goes up, the two outputs come down.
    let ll = host(&indep(64), H100, off(&["streams"]));
    assert_eq!(mapal_main(&ll).matches("@mapal_gpu_pin(").count(), 2);
}

#[test]
fn a_one_element_read_is_not_pinned() {
    // `c` is only read at two corners: one-element copies, nothing pinned.
    let src = program(64, &[("a", 7), ("b", 11)], &site("c", "a", "b", 64), &["c"]);
    let ll = host(&src, H100, on());
    assert!(ll.contains("@mapal_gpu_d2h("));
    assert!(!ll.contains("@mapal_gpu_pin("));
}

#[test]
fn pdl_is_hopper_and_later_only() {
    let src = diamond(64);
    let h = emit(&src, H100, on());
    let dev = h.device.expect("device unit");
    let kernels = dev.matches("define ptx_kernel void @mapal_tile_").count();
    assert!(kernels > 0);
    assert_eq!(dev.matches("griddepcontrol.wait;").count(), kernels);
    assert_eq!(init_flags(&h.text) & 4, 4);
    for part in ["rtx-4090", "rtx-3090", "a100"] {
        let e = emit(&src, part, on());
        assert!(
            !e.device.expect("device").contains("griddepcontrol"),
            "{part}"
        );
        assert_eq!(init_flags(&e.text) & 4, 0, "{part}");
    }
    let e = emit(&src, "rtx-5090", on());
    assert!(e.device.expect("device").contains("griddepcontrol.wait;"));
    let off_h = emit(&src, H100, off(&["pdl"]));
    assert!(!off_h.device.expect("device").contains("griddepcontrol"));
    assert_eq!(init_flags(&off_h.text) & 4, 0);
}
