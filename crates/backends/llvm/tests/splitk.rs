//! S62 `splitk` (ADR-0040): the reassociation face's deterministic split-K.
//! ON: the program prints the reference that folds in ADR-0040's partitioned
//! order (partition count from `TargetProfile::gpu_splitk` over the profile's
//! own facts) — which differs from the interpreter's ordered fold, so the test
//! sees the reassociation. OFF (`--gpu-off=splitk`): the interpreter's bits and
//! the Exact kernel's text. Run through the GPU emulator; fails on a compiler
//! without `Reduced::Reassoc`.

use mapal_backend_llvm::{EmitOpts, Reduced, TargetProfile, emit_detailed};
use mapal_ir::CategoryIr;

mod emu;
use emu::{clang, run_emulated};

/// `m × n × k` with `a[i·k + kk] = 2²⁴` at `kk = 0` and 1 elsewhere, `b = 1`.
/// In order every 1 after `2²⁴` rounds away (ties to even): each cell is
/// `2²⁴`. Partitions after the first sum their 1s exactly and add them to
/// `2²⁴` once each, so the partitioned order prints a different value.
fn src(m: u64, n: u64, k: u64) -> String {
    format!(
        "fn main() {{
    {mk} -> iota -> ta;
    ta -> map {{ t -> t % {k} -> kk; 1 + 16777215 * (({k1} - kk) / {k1}) -> widen_f32 }} -> a;
    {kn} -> iota -> tb;
    tb -> map {{ t -> t * 0 + 1 -> widen_f32 }} -> b;
    {k} -> iota -> krange;
    {mn} -> iota -> tc;
    tc -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, kx -> acc + a[i * {k} + kx] * b[kx * {n} + j] }}
    }} -> c;
    (0.0, c) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
    c[0] -> println;
    c[{last}] -> println;
}}
",
        mk = m * k,
        k1 = k - 1,
        kn = k * n,
        mn = m * n,
        last = m * n - 1,
    )
}

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit(ir: &CategoryIr, contract: bool, reduced: Option<Reduced>, off: bool) -> (String, String) {
    let mut opts = EmitOpts {
        target: "rtx-5090",
        contract,
        reduced,
        ..EmitOpts::default()
    };
    opts.gpu.off.splitk = off;
    // S62 integration: the split launches pinned here are derived on the
    // one-thread-per-cell grid, the tile the M16 cell had before the geometry
    // and cell-fill rungs (which re-tile it; gpu.rs
    // `splitk_derivation_on_the_measured_cells` covers that composition).
    for r in [
        "row-fit",
        "wave-gate",
        "grid-share",
        "panel-regs",
        "fill-rows",
        "cell-fill",
    ] {
        *opts.gpu.off.rung(r).expect("a rung") = true;
    }
    let em = emit_detailed(ir, &opts).expect("emits");
    (em.text, em.device.expect("a device unit"))
}

/// The `mapal_gpu_launch` calls of `host`: `(kernel name, i32 arguments)`.
fn launches(host: &str) -> Vec<(String, Vec<u64>)> {
    host.lines()
        .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_launch(i64 "))
        .map(|l| {
            let id: i64 = l.split(',').next().unwrap().parse().unwrap();
            let geom = l
                .split("i32 ")
                .skip(1)
                .map(|x| x.split([',', ')']).next().unwrap().parse().unwrap())
                .collect();
            (format!("mapal_tile_{:016x}", id as u64), geom)
        })
        .collect()
}

/// Kernel `name`'s text in the device unit, its name blanked.
fn kernel(dev: &str, name: &str) -> String {
    let start = dev
        .find(&format!("define ptx_kernel void @{name}("))
        .unwrap_or_else(|| panic!("no kernel {name}"));
    let end = start + dev[start..].find("\n}\n").expect("kernel end");
    dev[start..end].replace(name, "K")
}

/// The partition count the profile derives for the one-thread-per-cell grid
/// of an `m × n × k` f32 site: its blocks against every resident block slot.
fn derived(m: u64, n: u64, k: u64) -> u64 {
    let p = TargetProfile::named("rtx-5090").expect("a profile");
    let g = p.gpu().expect("a GPU profile");
    let wave = g.sms * (g.max_threads_per_sm / 256).min(g.max_blocks_per_sm);
    p.gpu_splitk((m * n).div_ceil(256), wave, k, 1, m * n, 4)
}

/// ADR-0040 §2: partition `z` folds `kk ∈ [z·kc, (z+1)·kc)` from 0 (partition
/// 0 from the seed, 0.0) with the face's step; partitions add in order. Then
/// the program's own ordered `total`.
fn reference(m: u64, n: u64, k: u64, s: u64, contract: bool) -> [f32; 3] {
    let a = |kk: u64| if kk == 0 { 16_777_216.0f32 } else { 1.0 };
    let kc = k / s;
    let part = |z: u64| {
        (z * kc..(z + 1) * kc).fold(0.0f32, |acc, kk| {
            if contract {
                a(kk).mul_add(1.0, acc)
            } else {
                acc + a(kk) * 1.0
            }
        })
    };
    let cell = (1..s).fold(part(0), |acc, z| acc + part(z));
    let total = (0..m * n).fold(0.0f32, |acc, _| acc + cell);
    [total, cell, cell]
}

fn printed(out: &str) -> Vec<f32> {
    out.lines()
        .map(|l| l.trim().parse().expect("a float line"))
        .collect()
}

/// The measured M16 cell: the split launch and its kernels as derived.
#[test]
fn m16_cell_splits_as_derived_and_off_is_the_exact_kernel() {
    let (m, n, k) = (16, 2048, 7168);
    let ir = lower(&src(m, n, k));
    let s = derived(m, n, k);
    assert_eq!(s, 8, "128 blocks against 170 SMs × 6 slots");
    let (host, dev) = emit(&ir, true, Some(Reduced::Reassoc), false);
    let l = launches(&host);
    assert_eq!(l.len(), 2, "split + reduce: {l:?}");
    assert!(host.contains(&format!("; splitk={s}\n")));
    assert_eq!(l[0].1, [128, s, 256, 0], "the split grid: z in y");
    assert_eq!(l[1].1, [128, 1, 256, 0], "the reduce grid");
    let split = kernel(&dev, &l[0].0);
    assert!(split.contains("ctaid.y") && split.contains(&format!(", {}\n", k / s)));
    assert_eq!(
        kernel(&dev, &l[1].0).matches(" = fadd ").count() as u64,
        s - 1
    );
    assert!(!dev.contains(" fast ") && !dev.contains(" reassoc "));
    // OFF: the Contract face's kernel and grid, under the reassoc face's name.
    let (host, dev) = emit(&ir, true, Some(Reduced::Reassoc), true);
    let (chost, cdev) = emit(&ir, true, None, false);
    let (l, cl) = (launches(&host), launches(&chost));
    assert_eq!(l.len(), 1);
    assert_eq!(kernel(&dev, &l[0].0), kernel(&cdev, &cl[0].0));
    assert_eq!(l[0].1, cl[0].1);
}

/// A site small enough to interpret: one block, so `S` doubles until it
/// stops cutting `k` evenly — `k = 12`: `S = 4`, three steps each.
#[test]
fn reassoc_bits_are_the_partitioned_fold_and_off_bits_are_the_oracle() {
    let (m, n, k) = (2, 64, 12);
    let ir = lower(&src(m, n, k));
    let rr = mapal_interp::run(&ir, 100_000_000);
    assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
    let ordered = reference(m, n, k, 1, false);
    assert_eq!(printed(&rr.output), ordered);
    let s = derived(m, n, k);
    assert_eq!(s, 4);
    assert_ne!(
        reference(m, n, k, s, false),
        ordered,
        "the inputs see the order"
    );
    let Some(cc) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    for contract in [false, true] {
        let (host, dev) = emit(&ir, contract, Some(Reduced::Reassoc), false);
        assert!(host.contains(&format!("; splitk={s}\n")));
        let out = run_emulated(&cc, &host, &dev);
        assert_eq!(
            printed(&out),
            reference(m, n, k, s, contract),
            "contract={contract}"
        );
        // Deterministic: the same bits on a second run.
        assert_eq!(run_emulated(&cc, &host, &dev), out);
    }
    let (host, dev) = emit(&ir, false, Some(Reduced::Reassoc), true);
    assert!(!host.contains("; splitk="));
    assert_eq!(run_emulated(&cc, &host, &dev), rr.output);
}
