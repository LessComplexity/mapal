//! S62 `place`, the mechanism: one predicate decides every reader of a
//! site's placement. Synthetic costs (not a part's), so the verdicts are
//! fixed whatever the calibration tables say.
use super::*;
use crate::emit::emit_module;
use crate::profile::{DeviceCost, DispatchCost, Machine, RTX_4090, X86_64_V3, with_host};

/// rtx-4090's device and x86-64-v3's host, with round costs: a launch is
/// 20 µs, the device 20 Tflop/s; one host lane 50 Gflop/s.
fn synth(place: bool) -> &'static TargetProfile {
    let mut p = with_host(&RTX_4090, &X86_64_V3);
    p.dispatch = Some(DispatchCost {
        flops_per_ns: 50,
        bytes_per_ns: 10,
        dispatch_ns: 10_000,
        lanes: 0,
    });
    if let Machine::Gpu(g) = &mut p.machine {
        g.cost = place.then_some(DeviceCost {
            flops_per_ns: 20_000,
            bytes_per_ns: 500,
            launch_ns: 20_000,
            ..crate::profile::NO_PLAN
        });
    }
    Box::leak(Box::new(p))
}

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    mapal_rewrite::rewrite(mapal_lower::lower(src, &po.program).expect("lowers")).ir
}

fn law(name: &str, len: u64, mul: u64) -> String {
    format!(
        "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> {name};\n"
    )
}

fn site(out: &str, a: &str, b: &str, m: u64, n: u64, k: u64) -> String {
    format!(
        "    {mn} -> iota -> o_{out};\n    {k} -> iota -> k_{out};\n    o_{out} -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, k_{out}) -> fold {{ acc, k -> acc + {a}[i * {k} + k] * {b}[k * {n} + j] }} }} -> {out};\n",
        mn = m * n
    )
}

/// An 8³ chain (three sites) written first, then a 256³ chain, then a
/// small site (8×8, k = 256) over the large chain's output.
fn mixed() -> CategoryIr {
    let mut s = String::from("fn main() {\n");
    for (x, mul) in [("sa", 7), ("sb", 11), ("se", 5), ("sf", 3)] {
        s += &law(x, 64, mul);
    }
    for (x, mul) in [("a", 7), ("b", 11), ("e", 5)] {
        s += &law(x, 65536, mul);
    }
    s += &law("w", 2048, 13);
    s += &site("c1", "sa", "sb", 8, 8, 8);
    s += &site("c2", "c1", "se", 8, 8, 8);
    s += &site("c3", "c2", "sf", 8, 8, 8);
    s += &site("c", "a", "b", 256, 256, 256);
    s += "    c -> map { v -> v * 0.5 } -> h;\n";
    s += &site("d", "h", "e", 256, 256, 256);
    s += &site("g", "d", "w", 8, 8, 256);
    s += "    c3[63] -> println;\n    d[65535] -> println;\n    g[63] -> println;\n}\n";
    lower(&s)
}

/// The rules' plan (`--cands=1`): placement is what this checks (the
/// row-block arm off: its chain of 8×8 sites is `row_block.rs`'s; the
/// `horizontal` arm off: the three independent 8³ sites would launch as
/// one, `horizontal.rs`'s).
fn emit_on(ir: &CategoryIr, place: bool) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        row_block: false,
        ..GpuArms::default()
    };
    *gpu.off.rung("horizontal").expect("a rung") = true;
    *gpu.off.rung("calls").expect("a rung") = true;
    let opts = EmitOpts {
        gpu,
        ..EmitOpts::exact()
    };
    emit_module(ir, &opts, synth(place)).expect("emits")
}

fn count(text: &str, pat: &str) -> usize {
    text.matches(pat).count()
}

/// The three 8³ sites move to the host — no launch, no kernel, no device
/// generation of their operands — while the 256³ chain and the small site
/// it feeds stay on the device.
#[test]
fn place_is_one_predicate() {
    let ir = mixed();
    let (on, off) = (emit_on(&ir, true), emit_on(&ir, false));
    let (don, doff) = (
        on.device.expect("a device chain"),
        off.device.expect("device"),
    );
    let kernels = |d: &str| count(d, "define ptx_kernel void @mapal_tile_");
    // The host walk, the residency plan and the device unit agree: the
    // three small sites (one kernel: the same shape) lost their kernel
    // AND their launches together.
    assert_eq!(kernels(&doff) - kernels(&don), 1, "{don}");
    let launches = |t: &str| count(t, "call void @mapal_gpu_launch");
    assert_eq!(launches(&off.text) - launches(&on.text), 3);
    // Left on the device: `c` (with `h` in its store), `d`, and `g` — an
    // 8×8 site its roofline puts on the host, kept on the device because
    // its operand `d` is device-produced (`fed`): `d` never crosses whole.
    assert_eq!(launches(&on.text), 3, "{}", on.text);
    // Their closed-form operands are no longer built on the device.
    let gens = |d: &str| count(d, "define ptx_kernel void @mapal_gen_");
    assert_eq!(gens(&doff) - gens(&don), 4);
}

/// Without measured costs (`place` off, or a part without a `cost` row)
/// the placement is S52–S54's: every capable site.
#[test]
fn without_costs_every_capable_site_is_placed() {
    let ir = mixed();
    let mut gpu = GpuArms {
        site_cands: 1,
        row_block: false,
        ..GpuArms::default()
    };
    *gpu.off.rung("horizontal").expect("a rung") = true;
    *gpu.off.rung("calls").expect("a rung") = true;
    let opts = EmitOpts {
        gpu,
        ..EmitOpts::exact()
    };
    let today = emit_module(&ir, &opts, &RTX_4090).expect("emits");
    // gemv-stream: the 4090's measured in-flight knee (a cost row the
    // synthetic part lacks) is printed on a streaming site's plan line;
    // these sites' kernels and launches are the same without it.
    let knee = |d: Option<String>| {
        d.map(|d| {
            d.lines()
                .map(|l| match l.find(" knee=") {
                    Some(i) => {
                        let rest = &l[i + 6..];
                        let end = rest.find(' ').unwrap_or(rest.len());
                        format!("{}{}", &l[..i], &rest[end..])
                    }
                    None => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
    };
    assert_eq!(
        knee(emit_on(&ir, false).device),
        knee(today.device),
        "the device unit is S54's"
    );
}
