//! S53 `host-runtime`: the GPU runtime's init is `main`'s first call exactly
//! when the module has a GPU site, and it carries the host rung `eager-init` as
//! a flag bit. Nothing else in the module moves.

use mapal_backend_llvm::{EmitOpts, GpuArms, GpuRungs, emit_detailed};
use mapal_ir::CategoryIr;

/// One offloadable tile site (64×64×64, f32).
const SITE: &str = r#"fn main() {
    4096 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 7.0 } -> a;
    ta -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 9.0 } -> b;
    64 -> iota -> krange;
    ta -> map { t ->
        t / 64 -> i;
        t % 64 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }
    } -> c;
    c[4095] -> println;
}
"#;

/// No tile site at all.
const NO_SITE: &str = r#"fn main() {
    16 -> iota -> t;
    t -> map { x -> x * 3 } -> u;
    u[15] -> println;
}
"#;

/// Lowered and rewritten, the bench pipeline (`emit --rewrite`).
fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn host(src: &str, target: &'static str, gpu: GpuArms) -> String {
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::exact()
    };
    emit_detailed(&lower_src(src), &opts).unwrap().text
}

/// The first instruction of `@main`.
fn main_first(ll: &str) -> &str {
    let head = "define i32 @main() {\nentry:\n";
    let body = &ll[ll.find(head).expect("@main") + head.len()..];
    body.lines().next().unwrap().trim()
}

#[test]
fn a_gpu_site_makes_init_mains_first_call() {
    let ll = host(SITE, "rtx-4090", GpuArms::default());
    assert!(
        ll.contains("call void @mapal_gpu_launch"),
        "the site offloads (`resident`'s split entries)"
    );
    // The shipped rungs: eager-init (1), preload (8); stage (16) rides on
    // `async-h2d`, which ships OFF on the measured parts.
    assert_eq!(main_first(&ll), "call void @mapal_gpu_init(i32 9)");
    assert!(ll.contains("declare void @mapal_gpu_init(i32)\n"));
}

/// The OFF arm of `eager-init`. S54's `pdl` (sm_90+) is another flag bit and
/// changes the kernels, so both arms here hold it off: one rung at a time.
fn off_eager() -> GpuArms {
    GpuArms {
        off: GpuRungs {
            eager_init: true,
            pdl: true,
            ..GpuRungs::default()
        },
        ..GpuArms::default()
    }
}

fn no_pdl() -> GpuArms {
    GpuArms {
        off: GpuRungs {
            pdl: true,
            ..GpuRungs::default()
        },
        ..GpuArms::default()
    }
}

#[test]
fn the_rung_is_one_flag_bit() {
    let ll = host(SITE, "h100-pcie", off_eager());
    assert_eq!(main_first(&ll), "call void @mapal_gpu_init(i32 8)");
}

#[test]
fn the_host_rung_moves_nothing_but_the_flag() {
    let on = host(SITE, "rtx-5090", no_pdl());
    let off = host(SITE, "rtx-5090", off_eager());
    assert_eq!(
        on.replace("@mapal_gpu_init(i32 9)", "@mapal_gpu_init(i32 8)"),
        off
    );
    let dev = |gpu| {
        let opts = EmitOpts {
            host: "generic",
            target: "rtx-5090",
            gpu,
            ..EmitOpts::exact()
        };
        emit_detailed(&lower_src(SITE), &opts).unwrap().device
    };
    assert!(dev(no_pdl()).is_some());
    assert_eq!(
        dev(no_pdl()),
        dev(off_eager()),
        "the device unit is unchanged"
    );
}

#[test]
fn no_init_without_a_gpu_site() {
    for target in ["generic", "apple-m", "zen3"] {
        let ll = host(SITE, target, GpuArms::default());
        assert!(
            !ll.contains("mapal_gpu"),
            "{target}: a CPU profile has no GPU runtime"
        );
    }
    let ll = host(NO_SITE, "rtx-4090", GpuArms::default());
    assert!(
        !ll.contains("mapal_gpu"),
        "a GPU profile without a site has no GPU runtime"
    );
    assert_eq!(main_first(&ll), "call void @mapal_main()");
}

#[test]
fn golden_gpu_host_module() {
    insta::assert_snapshot!(
        "gpu_host_rtx_4090",
        host(SITE, "rtx-4090", GpuArms::default())
    );
}
