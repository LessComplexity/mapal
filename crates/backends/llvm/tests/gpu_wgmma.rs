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
    // 4 per `a` chunk this thread stages (128·32/4/128 = 8 per panel; the
    // two prologue panels and the one stored per iteration) and 4 per k8
    // step of `bᵀ` fragments (32/8 = 4).
    assert!(!k.contains("cvt.rna"));
    assert_eq!(k.matches("cvt.rn.tf32.f32").count(), 3 * 8 * 4 + 4 * 4);
    // `a` arrives through registers (`ld.global` → round → `st.shared`):
    // never read back from shared memory to be rounded in place.
    assert_eq!(
        k.matches("load <4 x float>, ptr addrspace(1)").count(),
        3 * 8
    );
    assert!(!k.contains("load <4 x float>, ptr addrspace(3)"));
    // The stores are fenced to the async proxy before the barrier.
    assert!(k.contains("fence.proxy.async.shared::cta"));
    // Descriptor: 128-byte swizzle, SBO one 1024-byte atom; the atoms need
    // the dynamic shared memory 1024-aligned.
    assert!(k.contains(&format!(", {}\n", 0x4000_0040_0001_0000u64)));
    assert!(k.contains("@mapal_smem = external addrspace(3) global [0 x i8], align 1024"));
}

/// `wgmma-reg-stage` off: `a` lands by `cp.async` and each thread rounds the
/// chunks it copied in place, once per panel (the first S53 kernel).
#[test]
fn wgmma_reg_stage_off_rounds_the_landed_panel_in_place() {
    let off = GpuRungs {
        wgmma_reg_stage: false,
        ..GpuRungs::default()
    };
    let k = device("h100-pcie", true, TF32, off);
    assert!(k.contains("wgmma.mma_async"));
    assert_eq!(k.matches("cvt.rn.tf32.f32").count(), 8 * 4 + 4 * 4);
    assert_eq!(k.matches("load <4 x float>, ptr addrspace(3)").count(), 8);
    assert!(!k.contains("load <4 x float>, ptr addrspace(1)"));
    assert!(k.contains("fence.proxy.async.shared::cta"));
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
