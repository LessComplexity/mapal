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

fn emit(src: &str, target: &'static str, mut gpu: GpuArms) -> Emission {
    // R7 alone (S54 integration): R1 (`epilogue`) and R2 (`devmap`) place these
    // programs' per-cell joins on the device (and launch them with `launchv` on
    // the legacy stream), `arena` lays out their buffers; their own tests pin those.
    for r in ["epilogue", "devmap", "arena"] {
        *gpu.off.rung(r).expect("a rung") = true;
    }
    emit_detailed(
        &lower_src(src),
        &EmitOpts {
            host: "generic",
            target,
            gpu,
            ..EmitOpts::exact()
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

/// The measurement arm with `rungs` forced on — the mechanism, whatever the
/// part's measured table ships (`pinned` and `pdl` ship OFF on every part).
fn forced(rungs: &[&str]) -> GpuArms {
    let mut a = GpuArms::default();
    for r in rungs {
        *a.on.rung(r).expect("a rung") = true;
    }
    a
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

/// `indep` without its join: both outputs read at the corners only, so both
/// stay on the device (gen_tiles `twin`).
fn twin(n: u64) -> String {
    program(
        n,
        &[("a", 7), ("b", 11), ("e", 5), ("f", 3)],
        &(site("c", "a", "b", n) + &site("d", "e", "f", n)),
        &["c", "d"],
    )
}

/// One output, two independent readers; device-only (corners).
fn diamond(n: u64) -> String {
    program(
        n,
        &[("a", 7), ("b", 11), ("e", 5), ("g", 3)],
        &(site("c", "a", "b", n) + &site("d1", "c", "e", n) + &site("d2", "c", "g", n)),
        &["d1", "d2"],
    )
}

/// Two independent two-site chains, device-only (corners).
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
            + &site("d1", "c1", "e", n)
            + &site("c2", "f", "g", n)
            + &site("d2", "c2", "p", n)),
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
/// A part whose measured table ships `streams` ON (the A100).
const ON_PART: &str = "a100";

#[test]
fn independent_device_only_sites_take_two_streams() {
    let ll = host(&twin(64), ON_PART, on());
    let t = calls(&ll);
    let pos = |x: &str| {
        t.iter()
            .position(|c| c == x)
            .unwrap_or_else(|| panic!("{x} in {t:?}"))
    };
    // Both launches before any copy-out, each corner on its producer's stream.
    assert!(pos("launchv 1") < pos("launchv 2"), "{t:?}");
    assert!(pos("launchv 2") < pos("d2h_on 1"), "{t:?}");
    assert_eq!(t.iter().filter(|c| *c == "d2h_on 1").count(), 2, "{t:?}");
    assert_eq!(t.iter().filter(|c| *c == "d2h_on 2").count(), 2, "{t:?}");
    // The operands die at the sites; their frees wait for the last launch.
    let frees_before = t[..pos("launchv 2")]
        .iter()
        .filter(|c| *c == "free")
        .count();
    assert_eq!(frees_before, 0, "{t:?}");
    assert!(!t.contains(&"launch".to_owned()), "{t:?}");
    assert!(!t.iter().any(|c| c.starts_with("event")), "{t:?}");
    assert_eq!(init_flags(&ll) & 2, 2, "the runtime creates the streams");
}

#[test]
fn a_whole_copy_out_keeps_one_stream() {
    // indep's join reads c and d whole on the host: the host is between the
    // chains, so there is nothing to overlap (measured slower on five parts).
    let ll = host(&indep(64), ON_PART, on());
    assert!(!ll.contains("@mapal_gpu_launchv("));
    assert!(!ll.contains("@mapal_gpu_event("));
    assert_eq!(init_flags(&ll) & 2, 0);
}

#[test]
fn diamond_waits_on_its_join_edge_only() {
    let t = calls(&host(&diamond(64), ON_PART, on()));
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
fn two_chains_with_no_edge() {
    let t = calls(&host(&mlp2(64), ON_PART, on()));
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
    let ll = host(&fanout(64), ON_PART, on());
    assert!(!ll.contains("@mapal_gpu_launchv("), "fanout is one chain");
    assert_eq!(init_flags(&ll) & 2, 0);
}

#[test]
fn a_filled_part_keeps_one_stream_unless_the_gate_is_off() {
    // 1024² on the ON_PART PCIe: the two grids do not fit one wave together.
    let gated = host(&twin(1024), ON_PART, on());
    assert!(!gated.contains("@mapal_gpu_launchv("));
    let forced = host(&twin(1024), ON_PART, off(&["stream-gate"]));
    assert!(forced.contains("@mapal_gpu_launchv("));
}

#[test]
fn a_part_whose_table_turns_streams_off_keeps_one_stream() {
    for part in ["rtx-3090", "rtx-4090", "rtx-5090", H100, "h100"] {
        assert!(
            !host(&twin(64), part, on()).contains("@mapal_gpu_launchv("),
            "{part}"
        );
    }
    assert!(host(&twin(64), "rtx-5090", forced(&["streams"])).contains("@mapal_gpu_launchv("));
}

#[test]
fn every_off_arm_is_the_one_stream_pageable_text() {
    let all_off = off(&["streams", "pinned", "pdl"]);
    for src in [indep(64), twin(64), diamond(64), mlp2(64), fanout(64)] {
        let ll = host(&src, H100, all_off);
        for entry in ["launchv", "event", "d2h_on", "pin", "unpin"] {
            assert!(!ll.contains(&format!("@mapal_gpu_{entry}(")), "{entry}");
        }
        assert_eq!(init_flags(&ll), 1, "eager-init only");
        let dev = emit(&src, H100, all_off).device.expect("a device unit");
        assert!(!dev.contains("griddepcontrol"));
    }
    // Streams off alone: one stream; pins are a separate rung.
    let ll = host(&twin(64), H100, off(&["streams"]));
    assert!(!ll.contains("@mapal_gpu_launchv("));
    let mut a = forced(&["pinned"]);
    *a.off.rung("streams").expect("a rung") = true;
    let ll = host(&relu_loop(64), H100, a);
    assert!(ll.contains("@mapal_gpu_pin("));
}

/// `x ← relu(x · w)` three times, the relu on the host (gen_tiles `loop`):
/// the site's output comes down and the next state goes up every iteration.
fn relu_loop(n: u64) -> String {
    format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> krange;
{a}{w}    mut x: [f32; {nn}] <- a;
    mut it: i32 <- 0;
    loop {{
        (it < 3) -> {{
            -true-> {{
{s}                y -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
}}
",
        nn = n * n,
        last = n * n - 1,
        a = operand("a", 7),
        w = operand("w", 11),
        s = site("y", "x", "w", n),
    )
}

#[test]
fn pins_are_the_arrays_that_cross_the_bus_repeatedly() {
    // Straight-line copies run once: never pinned (devgen off, so a, b, e, f
    // go up and c, d come down — six single crossings).
    let ll = host(&indep(64), H100, off(&["devgen", "streams"]));
    assert_eq!(mapal_main(&ll).matches("@mapal_gpu_h2d(").count(), 4);
    assert!(!ll.contains("@mapal_gpu_pin("));
    // Shipped: no part's table pins (the sign followed the host's copy path).
    assert!(!host(&relu_loop(64), H100, on()).contains("@mapal_gpu_pin("));
    // In the loop, each pin sits right before the copy of the array it pins.
    let ll = host(&relu_loop(64), H100, forced(&["pinned"]));
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
        let next = lines[i + 1..]
            .iter()
            .find(|x| x.contains("@mapal_gpu_h2d(") || x.contains("@mapal_gpu_d2h("))
            .expect("a copy after the pin");
        assert!(next.contains(&format!("ptr {p},")), "{l} then {next}");
    }
    // y down and x up, every iteration; the loop's entry and exit run once.
    assert_eq!(pinned.len(), 2, "{pinned:?}\n{main}");
    assert!(main.contains("call void @mapal_gpu_unpin()"));
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
    // Shipped: OFF on both sm_90+ parts measured (no stable sign).
    for part in [H100, "rtx-5090"] {
        let e = emit(&src, part, on());
        assert!(
            !e.device.expect("device").contains("griddepcontrol"),
            "{part}"
        );
        assert_eq!(init_flags(&e.text) & 4, 0, "{part}");
    }
    let h = emit(&src, H100, forced(&["pdl"]));
    let dev = h.device.expect("device unit");
    let kernels = dev.matches("define ptx_kernel void @mapal_tile_").count();
    assert!(kernels > 0);
    assert_eq!(dev.matches("griddepcontrol.wait;").count(), kernels);
    assert_eq!(init_flags(&h.text) & 4, 4);
    for part in ["rtx-4090", "rtx-3090", "a100"] {
        let e = emit(&src, part, forced(&["pdl"]));
        assert!(
            !e.device.expect("device").contains("griddepcontrol"),
            "{part}"
        );
        assert_eq!(init_flags(&e.text) & 4, 0, "{part}");
    }
    let e = emit(&src, "rtx-5090", forced(&["pdl"]));
    assert!(e.device.expect("device").contains("griddepcontrol.wait;"));
}

// Combined streams + epilogue regressions: the standalone R7 helper disables fusion.
fn grouped(src: &str, contract: bool) -> Emission {
    let mut gpu = forced(&["streams"]);
    *gpu.off.rung("stream-gate").expect("rung") = true;
    emit_detailed(
        &lower_src(src),
        &EmitOpts {
            host: "generic",
            target: "rtx-5090",
            contract,
            gpu,
            ..EmitOpts::exact()
        },
    )
    .expect("combined placement emits")
}

const GROUPED_MEMBER: &str = r#"fn main() {
    4096 -> iota -> ta;
    64 -> iota -> krange;
    ta -> map { t -> (t % 7 + 1) -> widen_f32 } -> a;
    ta -> map { t -> (t % 5 + 2) -> widen_f32 } -> b;
    ta -> map { t -> (t % 11 + 1) -> widen_f32 } -> e;
    ta -> map { t -> (t % 13 + 1) -> widen_f32 } -> f;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c1;
    c1 -> map { v -> v * 2.0 } -> h1;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + e[i * 64 + k] * f[k * 64 + j] }
    } -> c2;
    c2 -> map { v -> v * 2.0 } -> h2;
    h2[0] -> println;
    h2[4095] -> println;
    h1[0] -> println;
    h1[4095] -> println;
    a[0] -> println;
    b[0] -> println;
    e[0] -> println;
    f[0] -> println;
    c1[0] -> println;
    c1[4095] -> println;
    c2[0] -> println;
    c2[4095] -> println;
}
"#;

const GROUPED_FORK: &str = r#"fn main() {
    4096 -> iota -> ta;
    64 -> iota -> krange;
    ta -> map { t -> (t % 7 + 1) -> widen_f32 } -> a;
    ta -> map { t -> (t % 5 + 2) -> widen_f32 } -> b;
    ta -> map { t -> (t % 11 + 1) -> widen_f32 } -> e;
    ta -> map { t -> (t % 13 + 1) -> widen_f32 } -> f;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c1;
    c1 -> map { v -> v * 2.0 } -> h1;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + h1[i * 64 + k] * e[k * 64 + j] }
    } -> left;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + h1[i * 64 + k] * f[k * 64 + j] }
    } -> right;
    right[0] -> println;
    right[4095] -> println;
    left[0] -> println;
    left[4095] -> println;
    h1[0] -> println;
    h1[4095] -> println;
    a[0] -> println;
    b[0] -> println;
    e[0] -> println;
    f[0] -> println;
    c1[0] -> println;
    c1[4095] -> println;
}
"#;

const GROUPED_ROW_FOLD: &str = r#"fn main() {
    4096 -> iota -> ta;
    64 -> iota -> krange;
    ta -> map { t -> (t % 7 + 1) -> widen_f32 } -> a;
    ta -> map { t -> (t % 5 + 2) -> widen_f32 } -> b;
    ta -> map { t -> (t % 11 + 1) -> widen_f32 } -> e;
    ta -> map { t -> (t % 13 + 1) -> widen_f32 } -> f;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c1;
    c1 -> map { v -> v * 2.0 } -> h1;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + e[i * 64 + k] * f[k * 64 + j] }
    } -> c2;
    c2 -> map { v -> v * 2.0 } -> h2;
    krange -> map { i -> (0.0, krange) -> fold { acc, j -> acc + h1[i * 64 + j] } } -> rows;
    rows[0] -> println;
    rows[63] -> println;
    h2[0] -> println;
    h2[4095] -> println;
    h1[0] -> println;
    h1[4095] -> println;
    a[0] -> println;
    b[0] -> println;
    e[0] -> println;
    f[0] -> println;
    c1[0] -> println;
    c1[4095] -> println;
    c2[0] -> println;
    c2[4095] -> println;
}
"#;

#[test]
fn grouped_member_corners_wait_on_the_physical_writer() {
    for contract in [false, true] {
        let t = calls(&grouped(GROUPED_MEMBER, contract).text);
        let launches: Vec<_> = t.iter().filter(|x| x.starts_with("launchv")).collect();
        assert_eq!(launches, ["launchv 1", "launchv 2"], "{t:?}");
        let reads: Vec<_> = t.iter().filter(|x| x.starts_with("d2h")).collect();
        assert_eq!(
            reads,
            [
                "d2h_on 2", "d2h_on 2", "d2h_on 1", "d2h_on 1", "d2h", "d2h", "d2h", "d2h",
                "d2h_on 1", "d2h_on 1", "d2h_on 2", "d2h_on 2"
            ],
            "{t:?}"
        );
        assert!(!t.iter().any(|x| x.starts_with("event")), "{t:?}");
    }
}

#[test]
fn grouped_member_fork_waits_only_for_the_other_chain() {
    for contract in [false, true] {
        let t = calls(&grouped(GROUPED_FORK, contract).text);
        let launches: Vec<_> = t.iter().filter(|x| x.starts_with("launchv")).collect();
        assert_eq!(launches, ["launchv 1", "launchv 1", "launchv 2"], "{t:?}");
        let events: Vec<_> = t.iter().filter(|x| x.starts_with("event")).collect();
        assert_eq!(events, ["event 1>2"], "{t:?}");
        let wait = t.iter().position(|x| x == "event 1>2").expect("wait");
        assert_eq!(t[wait + 1], "launchv 2", "{t:?}");
    }
}

#[test]
fn grouped_member_capture_uses_one_producer_wait() {
    // h1 is an external Gmem capture of c2's group, not c2's tile operand.
    let src = GROUPED_MEMBER.replace(
        "    c2 -> map { v -> v * 2.0 } -> h2;",
        "    ta -> map { t -> c2[t] + h1[t] + h1[t] } -> h2;",
    );
    for contract in [false, true] {
        let e = grouped(&src, contract);
        let t = calls(&e.text);
        let launches: Vec<_> = t.iter().filter(|x| x.starts_with("launchv")).collect();
        assert_eq!(launches, ["launchv 1", "launchv 2"], "{t:?}");
        // a,b,raw result,h1 capture,stored h2: the capture is in Epi.caps().
        assert!(
            mapal_main(&e.text).contains("i32 2, i32 5, ptr"),
            "{}",
            e.text
        );
        let events: Vec<_> = t.iter().filter(|x| x.starts_with("event")).collect();
        assert_eq!(events, ["event 1>2"], "{t:?}");
        let wait = t.iter().position(|x| x == "event 1>2").expect("wait");
        // A stored epilogue output is allocated after the event; the next launch
        // must be the consuming group. Do not hoist that live allocation.
        let next_launch = t[wait + 1..].iter().find(|x| x.starts_with("launch"));
        assert_eq!(next_launch.map(String::as_str), Some("launchv 2"), "{t:?}");
    }
}

#[test]
fn grouped_member_legacy_cell_keeps_the_conservative_fallback() {
    let src = GROUPED_MEMBER.replace("    h2[0] -> println;",
        "    ta -> map { t -> h1[4095 - t] } -> rev;\n    rev[0] -> println;\n    rev[4095] -> println;\n    h2[0] -> println;");
    for contract in [false, true] {
        for src in [src.clone(), src.replace("h1[4095 - t]", "c1[4095 - t]")] {
            let e = grouped(&src, contract);
            assert!(e.device.expect("device").contains("; cell"));
            assert_eq!(init_flags(&e.text) & 2, 0, "unordered legacy consumer");
            let t = calls(&e.text);
            assert!(
                !t.iter()
                    .any(|x| x == "launchv 1" || x == "launchv 2" || x.starts_with("event")),
                "{t:?}"
            );
        }
    }
}

#[test]
fn grouped_member_row_fold_keeps_its_ordered_whole_copy() {
    // S64's arm: the fold over the member's output runs on the host, behind
    // an ordered whole copy (S65 `by-ref` off).
    let s64 = |src: &str, contract| {
        let mut gpu = forced(&["streams"]);
        for r in ["stream-gate", "by-ref"] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        emit_detailed(
            &lower_src(src),
            &EmitOpts {
                host: "generic",
                target: "rtx-5090",
                contract,
                gpu,
                ..EmitOpts::exact()
            },
        )
        .expect("combined placement emits")
    };
    for contract in [false, true] {
        let e = s64(GROUPED_ROW_FOLD, contract);
        let t = calls(&e.text);
        let first = t.iter().position(|x| x == "launchv 1").expect("producer");
        assert_eq!(t[first + 1], "d2h_on 1", "{t:?}");
        assert!(
            mapal_main(&e.text)
                .lines()
                .any(|l| l.contains("@mapal_gpu_d2h_on(i32 1,") && l.contains("i64 16384)"))
        );
        // The existing base-output row-fold veto also remains conservative.
        let raw = GROUPED_ROW_FOLD.replace("acc + h1[i * 64 + j]", "acc + c1[i * 64 + j]");
        let e = s64(&raw, contract);
        assert_eq!(init_flags(&e.text) & 2, 0);
        // S65 rule R: the fold's operand is complete on the device, so the
        // fold runs there — one more launch, and the member's output never
        // crosses whole (every copy-out left is one element).
        let e = grouped(GROUPED_ROW_FOLD, contract);
        let t = calls(&e.text);
        assert_eq!(t.iter().filter(|x| x.starts_with("launch")).count(), 3, "{t:?}");
        assert!(
            mapal_main(&e.text)
                .lines()
                .filter(|l| l.contains("@mapal_gpu_d2h"))
                .all(|l| l.ends_with("i64 4)")),
            "{}",
            e.text
        );
    }
}

#[test]
fn grouped_loop_carry_views_stay_on_the_legacy_stream() {
    for contract in [false, true] {
        let e = grouped(&relu_loop(64), contract);
        assert_eq!(init_flags(&e.text) & 2, 0);
        let t = calls(&e.text);
        assert!(
            !t.iter()
                .any(|x| x == "launchv 1" || x == "launchv 2" || x.starts_with("event")),
            "{t:?}"
        );
    }
}

/// Allocated slots that neither supply a pointer load nor a used alloc result.
/// These are real storage, even though the selected group passes raw pointer 0.
fn unread_allocations(ll: &str) -> Vec<String> {
    let main = mapal_main(ll);
    main.lines()
        .filter(|l| l.contains(" = call i64 @mapal_gpu_alloc(ptr "))
        .filter_map(|l| {
            let (result, args) = l.trim().split_once(" = call i64 @mapal_gpu_alloc(ptr ")?;
            let slot = args.split(',').next().expect("slot");
            let loaded = main.contains(&format!("load i64, ptr {slot}\n"));
            let used = main.lines().any(|line| {
                line != l
                    && line
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '%')
                        .any(|token| token == result)
            });
            (!loaded && !used).then(|| l.trim().to_owned())
        })
        .collect()
}

#[test]
fn grouped_chains_allocate_only_materialized_raw_outputs() {
    let n = 64;
    let diamond_join = program(
        n,
        &[("a", 7), ("b", 11), ("e", 5), ("g", 3)],
        &(site("c", "a", "b", n)
            + &site("d1", "c", "e", n)
            + &site("d2", "c", "g", n)
            + "    ta -> map { t -> d1[t] + d2[t] } -> h;\n"),
        &["h"],
    );
    let fused_mlp2 = program(
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
    );
    let mut unread = Vec::new();
    for (name, src) in [
        ("independent", indep(n)),
        ("diamond", diamond_join),
        ("mlp2", fused_mlp2),
    ] {
        for contract in [false, true] {
            let e = grouped(&src, contract);
            let t = calls(&e.text);
            assert!(t.iter().any(|x| x == "launchv 1"), "{name}: {t:?}");
            assert!(t.iter().any(|x| x == "launchv 2"), "{name}: {t:?}");
            for allocation in unread_allocations(&e.text) {
                unread.push(format!("{name} contract={contract}: {allocation}"));
            }
        }
    }
    assert!(unread.is_empty(), "{unread:#?}");
}

#[test]
fn raw_storage_controls_keep_their_observable_pointers() {
    for src in [
        twin(64),
        fanout(64),
        GROUPED_MEMBER.to_owned(),
        GROUPED_FORK.to_owned(),
        relu_loop(64),
    ] {
        for contract in [false, true] {
            let e = grouped(&src, contract);
            assert!(
                unread_allocations(&e.text).is_empty(),
                "{:?}",
                unread_allocations(&e.text)
            );
        }
    }
}
