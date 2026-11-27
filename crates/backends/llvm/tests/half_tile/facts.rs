//! Member facts, bound and realization per arm of the half-precision tile.

use super::*;

/// D4 facts: `: bf16 | f32` is the product's set (`Precision::half`), its
/// tile site carries the member, and `precision_plan.bound` is D3's with
/// per-operand units (ADR-0043 slice 3, F5): the f32 operand's `u_t = 2⁻⁸`,
/// the `bf16`-stored weight's 0 (it embeds in the member), `u_a = 2⁻²³`; the
/// weakest member's interpreter rounds the f32 operand (another value), the
/// f32 side does not.
#[test]
fn member_facts_and_bound() {
    let ir = lower(GEMV);
    let f = ir.entry();
    let sites = ir.tile_plan_narrow(f);
    let (m, s) = sites.sites.iter().next().expect("a narrow site");
    assert_eq!(s.half, Some(NarrowFloat::Bf16));
    assert_eq!(s.unit(), Some(2f64.powi(-8)));
    let u = 2f64.powi(-8);
    let (ua, ub) = s.units().expect("a member");
    assert!((ua, ub) == (u, 0.0) || (ua, ub) == (0.0, u), "{ua} {ub}");
    let pp = ir.precision_plan(f);
    assert_eq!(
        pp.bound.get(m).copied(),
        Some(mapal_ir::bound_coef_units(64, ua, ub, 2f64.powi(-23)))
    );
    assert!(pp.bound.get(m).copied() < Some(bound_coef_unit(64, Some(u))));
    let sets: Vec<_> = ir
        .funcs()
        .flat_map(|(g, _)| ir.precision_plan(g).set.into_iter().map(|(_, p)| p))
        .collect();
    let mul = sets.first().expect("the product's set");
    assert!(mul.bf16 && mul.f32 && !mul.tf32 && !mul.f16);
    let f32 = run(&ir, 1 << 30).output;
    let weak = run_weakest(&ir, 1 << 30, &[]).output;
    assert_ne!(f32, weak, "the member rounds `a`");
}

/// D6 / the realization per arm on the RTX 5080 (`half-fill` off: the
/// small site takes the tile wherever it fits): `mma.sync.m16n8k16.bf16` —
/// `cp.async` panels (the f32 `a` 4 bytes, the bf16 `b` 2), `ldmatrix.trans`
/// of `b`, `cvt.rn.bf16x2.f32` of `a`; OFF (`half-tile`) and the exact face:
/// no 16-bit `mma`. The plan names the member's set and the instruction.
/// `half-fill` on: this skinny site (8 flop / B against the 5080's 54.9,
/// 16 one-warp blocks) takes the FMA tile; a compute-bound one (256 × 512 ×
/// 512, 85 flop / B) the 16-bit tile.
#[test]
fn realization_per_arm() {
    let ir = lower(GEMV);
    let shipped = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    assert!(half_lines(shipped.device.as_deref().unwrap_or("")).is_empty());
    let big = GEMV
        .replace("2048 -> iota -> ta", "131072 -> iota -> ta")
        .replace("8192 -> iota -> tw", "262144 -> iota -> tw")
        .replace("4096 -> iota -> t;", "131072 -> iota -> t;")
        .replace("64 -> iota -> kk", "512 -> iota -> kk")
        .replace("e / 128", "e / 512")
        .replace("e % 128", "e % 512")
        .replace("q * 128 + j", "q * 512 + j")
        .replace("i * 64 + q", "i * 512 + q")
        .replace("y[4095]", "y[131071]");
    let e = emission(&lower(&big), "rtx-5080", EmitOpts::default(), &[]);
    assert_eq!(
        half_lines(e.device.as_deref().unwrap_or("")).len(),
        1,
        "{big}"
    );
    // sweep-gemm `round-store`: `a` (f32, read only by the `bf16` product)
    // is stored rounded by its producer — a 2-byte panel, no conversion.
    let stored = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let sl = half_lines(stored.device.as_deref().expect("device"));
    assert_eq!(sl.len(), 1);
    assert!(
        sl[0].contains(" ea=2 eb=2") && sl[0].contains(" narrow=a:bf16+b:bf16 "),
        "{}",
        sl[0]
    );
    // Its OFF arm: the f32 operand rounded as each fragment is read.
    let on = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-fill", "round-store"],
    );
    let dev = on.device.as_deref().expect("device");
    let lines = half_lines(dev);
    assert_eq!(lines.len(), 1, "{dev}");
    let l = &lines[0];
    for f in [
        " kind=mma ",
        " op=m16n8k16.bf16 ",
        " ea=4 eb=2",
        " set=bf16|f32 ",
        " narrow=b:bf16 ",
    ] {
        assert!(l.contains(f), "{f}: {l}");
    }
    for i in [
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "ldmatrix.sync.aligned.m8n8.x",
        ".trans.shared.b16",
        "cvt.rn.bf16x2.f32",
        "@llvm.nvvm.cp.async.cg.shared.global.16",
    ] {
        assert!(dev.contains(i), "{i}");
    }
    let off = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-tile", "half-fill"],
    );
    let dev_off = off.device.as_deref().unwrap_or("");
    assert!(half_lines(dev_off).is_empty());
    assert!(!dev_off.contains("m16n8k16"));
    let exact = emission(&ir, "rtx-5080", EmitOpts::exact(), &["half-fill"]);
    assert!(
        half_lines(exact.device.as_deref().unwrap_or("")).is_empty(),
        "the exact face keeps the oracle's f32 products in its order"
    );
    // Every part listing the warp bf16 product takes it; ids differ from the
    // f32 program's (the site keys its member).
    for t in ["rtx-3090", "a100", "rtx-4090", "rtx-5090", "h100-pcie"] {
        let e = emission(&ir, t, EmitOpts::default(), &["half-fill"]);
        assert_eq!(
            half_lines(e.device.as_deref().unwrap_or("")).len(),
            1,
            "{t}"
        );
    }
}

/// `{f32}` over two operands of one narrow format: exact products, so the
/// member's realization (the products chained on the truncating
/// accumulator) is not admitted — every profiled part truncates
/// (`mma_calib.cu`): no `half-tile` kernel under the default face's set.
/// f32-tensor-cores: the split form `narrow1` takes it instead (each k-step
/// from a zeroed accumulator, one round-to-nearest add: D2 at the site's
/// `K`, `split_tile.rs`), its own rung `split-tile`.
#[test]
fn exact_products_need_the_round_to_nearest_accumulator() {
    let src = GEMV
        .replace("fold: bf16 | f32", "fold")
        .replace("v / 29.0 }", "v / 29.0 -> to<bf16> }")
        .replace(
            "acc + a[i * 64 + q] * b",
            "a[i * 64 + q] -> widen<f32> -> x; acc + x * b",
        );
    let ir = lower(&src);
    let s = ir.tile_plan_narrow(ir.entry());
    let site = s.sites.values().next().expect("site");
    assert!(site.a.narrow.is_some() && site.b.narrow.is_some() && site.half.is_none());
    let e = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = e.device.as_deref().unwrap_or("");
    assert!(half_lines(dev).is_empty());
    // This GEMV is DRAM-bound and its grid small: `split_pays` keeps the FMA
    // tile; the ON arm (the bound alone) takes narrow1.
    let mut arms = GpuArms::default();
    *arms.on.rung("split-tile").expect("rung") = true;
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ] {
        *arms.off.rung(r).expect("rung") = true;
    }
    let forced = emit_detailed(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu: arms,
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    assert!(
        forced
            .device
            .as_deref()
            .unwrap_or("")
            .lines()
            .any(|l| l.starts_with("; mapal.plan ") && l.contains(" split=narrow1")),
        "the split form narrow1"
    );
    let e = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-fill", "split-tile"],
    );
    assert!(!e.device.as_deref().unwrap_or("").contains("mma.sync"));
    // Under the program's `: bf16 | f32` the same operands take it.
    let ir = lower(&src.replace("-> fold {", "-> fold: bf16 | f32 {"));
    let e = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let lines = half_lines(e.device.as_deref().unwrap_or(""));
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains(" ea=2 eb=2"), "{}", lines[0]);
}
