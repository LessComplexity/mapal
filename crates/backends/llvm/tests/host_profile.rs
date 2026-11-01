//! S62 `hostopt`: a GPU build emits everything outside a device site against
//! the host's CPU profile, and a GPU build whose every site stays on the host
//! IS that host's CPU build — before S62 it was `generic`'s sequential text.

use mapal_backend_llvm::{EmitError, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;

/// Two independent integer tile sites: no device-capable site (the device
/// realization is float-only), two paths for the host's parallel entry.
const INT_SITES: &str = r#"fn main() {
    4096 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 } -> a;
    ta -> map { t -> (t * 11 + 57) % 103 - 51 } -> b;
    64 -> iota -> krange;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0, krange) -> fold { acc, k -> acc + b[i * 64 + k] * a[k * 64 + j] }
    } -> d;
    c[4095] -> println;
    d[4095] -> println;
}
"#;

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit(ir: &CategoryIr, target: &'static str, host: &'static str, gpu: GpuArms) -> String {
    let opts = EmitOpts {
        target,
        host,
        gpu,
        ..EmitOpts::exact()
    };
    let em = emit_detailed(ir, &opts).expect("emits");
    assert!(em.device.is_none(), "{target}: no device-capable site");
    em.text
}

/// The host code of a GPU build is the host profile's CPU build: `zen3`'s
/// widths and its parallel entry, not `generic`'s sequential walk.
#[test]
fn gpu_build_host_code_is_the_host_profiles_cpu_build() {
    let ir = lower_src(INT_SITES);
    let cpu = emit(&ir, "zen3", "native", GpuArms::default());
    assert!(
        cpu.contains("@mapal_par_begin"),
        "two paths: the parallel entry"
    );
    for target in ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"] {
        assert_eq!(
            emit(&ir, target, "zen3", GpuArms::default()),
            cpu,
            "{target}"
        );
    }
    // The default host is the machine emitting, like the default target.
    assert_eq!(
        emit(&ir, "rtx-4090", "native", GpuArms::default()),
        emit(&ir, "native", "native", GpuArms::default())
    );
}

/// The OFF arm is the S52–S54 GPU build: `generic`'s facts, sequential.
#[test]
fn hostopt_off_is_the_generic_sequential_host() {
    let ir = lower_src(INT_SITES);
    let mut off = GpuArms::default();
    off.off.hostopt = true;
    let text = emit(&ir, "rtx-4090", "zen3", off);
    assert!(!text.contains("@mapal_par_begin"));
    assert_ne!(text, emit(&ir, "zen3", "native", GpuArms::default()));
}

/// One `n³` f32 site over closed-form operands.
fn site(n: u64) -> String {
    format!(
        "fn main() {{
    {nn} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 9.0 }} -> b;
    {n} -> iota -> krange;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
    c[{last}] -> println;
}}
",
        nn = n * n,
        last = n * n - 1
    )
}

/// S62 `place` on the measured parts (`x86-64-v3`'s lane rate, each part's
/// device cost): a 4³ site (128 flops — nanoseconds on one host lane against a
/// launch of microseconds) stays on the host, so the build is the CPU build
/// and starts no GPU runtime; a 512³ site still launches. Before S62 the 4³
/// site launched on every GPU target.
#[test]
fn small_site_stays_on_the_host() {
    let (small, large) = (lower_src(&site(4)), lower_src(&site(512)));
    for target in ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"] {
        let opts = EmitOpts {
            target,
            host: "x86-64-v3",
            ..EmitOpts::exact()
        };
        let em = emit_detailed(&small, &opts).expect("emits");
        assert!(em.device.is_none(), "{target}");
        assert!(!em.text.contains("@mapal_gpu_"), "{target}: no GPU runtime");
        let em = emit_detailed(&large, &opts).expect("emits");
        assert!(em.device.is_some(), "{target}");
        assert!(em.text.contains("call void @mapal_gpu_launch"), "{target}");
        // `place` off: the small site launches again (the S52–S54 placement).
        let mut off = opts;
        off.gpu.off.place = true;
        let em = emit_detailed(&small, &off).expect("emits");
        assert!(em.text.contains("call void @mapal_gpu_launch"), "{target}");
    }
}

/// A host profile names the host of a GPU target: a GPU profile as the host,
/// or an unknown name, is an error rather than a silent default.
#[test]
fn host_must_be_a_cpu_profile_of_a_gpu_target() {
    let ir = lower_src(INT_SITES);
    for (target, host) in [("rtx-4090", "rtx-5090"), ("rtx-4090", "zen_3")] {
        let opts = EmitOpts {
            target,
            host,
            ..EmitOpts::exact()
        };
        assert!(
            matches!(emit_detailed(&ir, &opts), Err(EmitError::Internal(_))),
            "{target} + {host}"
        );
    }
}
