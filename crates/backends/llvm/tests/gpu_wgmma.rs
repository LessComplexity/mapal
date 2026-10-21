//! The Hopper `wgmma` rung's gate and its rounding, at the device-text level
//! (S53 `wgmma-hopper`). The kernel itself is checked on the part (the TF32
//! bound gate in `benches/gpu/tile_bench.cu`); what is pinned here is where it
//! may appear, that every operand it reads is rounded to nearest-even (ADR-0039
//! `to_tf32`; `wgmma` truncates what it reads), and that nothing else moved.

use mapal_backend_llvm::{EmitOpts, GpuRungs, Reduced, emit_detailed};

/// `benches/gpu/gen_tiles.py`'s bench site at 1024.
const MM_1024: &str = r#"
fn main() {
    1048576 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> a;
    ta -> map { t -> (t * 7 + 57) % 101 - 50 -> widen_f32 } -> b;
    1024 -> iota -> krange;
    ta -> map { t ->
        t / 1024 -> i;
        t % 1024 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 1024 + k] * b[k * 1024 + j] }
    } -> c;
    c[0] -> println;
}
"#;

fn device(target: &'static str, contract: bool, reduced: Option<Reduced>, gpu: GpuRungs) -> String {
    let po = mapal_syntax::parse(MM_1024);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(MM_1024, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    let ir = mapal_rewrite::rewrite(ir).ir;
    let opts = EmitOpts {
        target,
        contract,
        reduced,
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(&ir, &opts)
        .expect("emits")
        .device
        .expect("a GPU profile offloads the site")
}

const TF32: Option<Reduced> = Some(Reduced::Tf32);

#[test]
fn wgmma_kernel_on_sm_90a_rounds_every_operand_to_nearest_even() {
    let k = device("h100-pcie", true, TF32, GpuRungs::default());
    assert!(k.contains("wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32"));
    assert!(!k.contains("mma.sync"));
    // RNE, never `rna` (ties away) — and nothing reaches `wgmma` unrounded:
    // per panel, 4 per staged `a` chunk this thread copied (128·32/4/128 = 8)
    // and 4 per k8 step of `bᵀ` fragments (32/8 = 4).
    assert!(!k.contains("cvt.rna"));
    assert_eq!(k.matches("cvt.rn.tf32.f32").count(), 8 * 4 + 4 * 4);
    // The rounded panel is fenced to the async proxy before the barrier.
    assert!(k.contains("fence.proxy.async.shared::cta"));
    // Descriptor fields for bk = 32: LBO 128 B, SBO 1024 B, no swizzle.
    assert!(k.contains(", 274878431232\n"));
}

/// The OFF arm is the S52 `mma.sync` kernel, unchanged (still `cvt.rna`).
#[test]
fn wgmma_off_is_the_mma_sync_kernel() {
    let off = GpuRungs {
        wgmma: false,
        ..GpuRungs::default()
    };
    let k = device("h100-pcie", true, TF32, off);
    assert!(!k.contains("wgmma"));
    assert!(k.contains("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"));
    assert!(k.contains("cvt.rna.tf32.f32"));
}

/// Only `sm_90a` has `wgmma`; only the TF32 face admits tensor cores; the
/// exact and contract faces on Hopper take the FMA kernel untouched.
#[test]
fn wgmma_is_unreachable_off_hopper_and_off_the_tf32_face() {
    for target in ["rtx-4090", "rtx-5090", "cuda-ada"] {
        assert!(
            !device(target, true, TF32, GpuRungs::default()).contains("wgmma"),
            "{target}"
        );
    }
    for contract in [false, true] {
        let k = device("h100-pcie", contract, None, GpuRungs::default());
        assert!(
            !k.contains("wgmma") && !k.contains("tf32"),
            "contract={contract}"
        );
    }
}
