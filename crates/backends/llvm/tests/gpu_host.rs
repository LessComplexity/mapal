//! S53 `host-runtime`: the GPU runtime's init is `main`'s first call exactly
//! when the module has a GPU site, and it carries the two host rungs
//! (`eager-init`, `pool`) as flag bits. Nothing else in the module moves.

use mapal_backend_llvm::{EmitOpts, GpuRungs, emit_detailed};
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

fn host(src: &str, target: &'static str, gpu: GpuRungs) -> String {
    let opts = EmitOpts {
        target,
        gpu,
        ..EmitOpts::default()
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
    let ll = host(SITE, "rtx-4090", GpuRungs::default());
    assert!(
        ll.contains("call void @mapal_gpu_tile("),
        "the site offloads"
    );
    // The shipped rungs: eager-init ON, pool OFF (measured, `GpuRungs::pool`).
    assert_eq!(main_first(&ll), "call void @mapal_gpu_init(i32 1)");
    assert!(ll.contains("declare void @mapal_gpu_init(i32)\n"));
}

#[test]
fn each_rung_is_one_flag_bit() {
    for (eager_init, pool, flags) in [
        (true, true, 3),
        (false, true, 2),
        (true, false, 1),
        (false, false, 0),
    ] {
        let rungs = GpuRungs {
            eager_init,
            pool,
            ..GpuRungs::default()
        };
        let ll = host(SITE, "h100-pcie", rungs);
        assert_eq!(
            main_first(&ll),
            format!("call void @mapal_gpu_init(i32 {flags})")
        );
    }
}

#[test]
fn the_host_rungs_move_nothing_but_the_flags() {
    let both = GpuRungs {
        eager_init: true,
        pool: true,
        ..GpuRungs::default()
    };
    let on = host(SITE, "rtx-5090", both);
    let off = host(
        SITE,
        "rtx-5090",
        GpuRungs {
            eager_init: false,
            pool: false,
            ..GpuRungs::default()
        },
    );
    assert_eq!(
        on.replace("@mapal_gpu_init(i32 3)", "@mapal_gpu_init(i32 0)"),
        off
    );
    let dev = |gpu| {
        let opts = EmitOpts {
            target: "rtx-5090",
            gpu,
            ..EmitOpts::default()
        };
        emit_detailed(&lower_src(SITE), &opts).unwrap().device
    };
    let off_rungs = GpuRungs {
        eager_init: false,
        pool: false,
        ..GpuRungs::default()
    };
    assert!(dev(GpuRungs::default()).is_some());
    assert_eq!(
        dev(GpuRungs::default()),
        dev(off_rungs),
        "the device unit is unchanged"
    );
}

#[test]
fn no_init_without_a_gpu_site() {
    for target in ["generic", "apple-m", "zen3"] {
        let ll = host(SITE, target, GpuRungs::default());
        assert!(
            !ll.contains("mapal_gpu"),
            "{target}: a CPU profile has no GPU runtime"
        );
    }
    let ll = host(NO_SITE, "rtx-4090", GpuRungs::default());
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
        host(SITE, "rtx-4090", GpuRungs::default())
    );
}
