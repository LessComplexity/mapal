//! ADR-0040: the fold orders of a tile site, through the GPU emulator.
//!
//! - `fixed` (S62's deterministic split-K, D3): the program prints the fold in
//!   the compiled partition order — which differs from the interpreter's
//!   ordered fold, so the test sees the reassociation — and
//!   `mapal_interp::run_ordered` replays exactly that order from
//!   `Emission::orders`, bit for bit. OFF (`--gpu-off=splitk`): the
//!   interpreter's bits.
//! - the default face (`free` order, `fma` steps; S63 `splitk-atomic` here):
//!   every printed value within D2's a-priori bound of the exact sum.

use mapal_backend_llvm::{EmitOpts, Order, TargetProfile, emit_detailed};
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

/// The emission on the 5090 for one face, `off` the listed rungs.
fn emit_face(ir: &CategoryIr, base: EmitOpts, off: &[&str]) -> mapal_backend_llvm::Emission {
    let mut opts = EmitOpts {
        target: "rtx-5090",
        ..base
    };
    if ir_k(ir) == 12 {
        // The interpretable site: one block, so the derived count would cut
        // `k` into single steps (no order to see); force S = 4 (the
        // measurement override, `--splitk=4`): three steps per partition.
        opts.gpu.splitk = 4;
    }
    // S62 integration: the launches pinned here are derived on the
    // one-thread-per-cell grid, the tile the M16 cell had before the geometry
    // and cell-fill rungs (which re-tile it; gpu.rs
    // `the_chooser_on_the_measured_cells` covers that composition). The
    // emulator runs only that kernel (no barriers). `place` off: the small
    // interpretable site stays on the device (its roofline says host).
    // site-geometry `fill-wave` / `fit-first` off: the S62 count and tile
    // (their arms: site_geometry.rs).
    for r in [
        "place",
        "fill-wave",
        "fit-first",
        "row-fit",
        "wave-gate",
        "grid-share",
        "panel-regs",
        "fill-rows",
        "cell-fill",
    ]
    .iter()
    .chain(off)
    {
        *opts.gpu.off.rung(r).expect("a rung") = true;
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// The `k` of the program's tile site (its fold's array length).
fn ir_k(ir: &CategoryIr) -> u64 {
    ir.funcs()
        .flat_map(|(f, _)| {
            ir.tile_plan(f)
                .sites
                .values()
                .map(|s| s.k)
                .collect::<Vec<_>>()
        })
        .next()
        .unwrap_or(0)
}

/// The fixed face: separate steps, the compiled partition order (D3).
fn fixed(contract: bool) -> EmitOpts {
    EmitOpts {
        contract,
        order: Order::Fixed,
        ..EmitOpts::default()
    }
}

fn emit(ir: &CategoryIr, contract: bool, off: bool) -> (String, String) {
    let em = emit_face(ir, fixed(contract), if off { &["splitk"] } else { &[] });
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
/// of an `m × n × k` f32 site: its blocks against every resident block slot,
/// inside one wave (ADR-0040 D4, M1).
fn derived(m: u64, n: u64, k: u64, workspace: bool) -> u64 {
    let p = TargetProfile::named("rtx-5090").expect("a profile");
    let g = p.gpu().expect("a GPU profile");
    let wave = g.sms * (g.max_threads_per_sm / 256).min(g.max_blocks_per_sm);
    p.split_count((m * n).div_ceil(256), wave, k, m * n, 4, workspace)
}

/// ADR-0040 D3: partition `z` folds `kk ∈ [z·kc, (z+1)·kc)` from 0 (partition
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
fn m16_cell_splits_as_derived_and_off_is_the_ordered_kernel() {
    let (m, n, k) = (16, 2048, 7168);
    let ir = lower(&src(m, n, k));
    let s = derived(m, n, k, true);
    assert_eq!(
        s, 7,
        "128 blocks against 170 SMs × 6 slots, one wave, s | 7168"
    );
    let (host, dev) = emit(&ir, true, false);
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
    // OFF: the sequential face's kernel and grid.
    let (host, dev) = emit(&ir, true, true);
    let seq = emit_face(
        &ir,
        EmitOpts {
            order: Order::Sequential,
            ..EmitOpts::default()
        },
        &[],
    );
    let (chost, cdev) = (seq.text, seq.device.expect("device"));
    let (l, cl) = (launches(&host), launches(&chost));
    assert_eq!(l.len(), 1);
    assert_eq!(kernel(&dev, &l[0].0), kernel(&cdev, &cl[0].0));
    assert_eq!(l[0].1, cl[0].1);
}

/// A site small enough to interpret, split in four (`--splitk=4`).
#[test]
fn fixed_bits_are_the_partitioned_fold_the_oracle_replays() {
    let (m, n, k) = (2, 64, 12);
    let ir = lower(&src(m, n, k));
    let rr = mapal_interp::run(&ir, 100_000_000);
    assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
    let ordered = reference(m, n, k, 1, false);
    assert_eq!(printed(&rr.output), ordered);
    assert_eq!(
        derived(m, n, k, true),
        12,
        "one block: every step its own partition"
    );
    let s = 4;
    assert_ne!(
        reference(m, n, k, s, false),
        ordered,
        "the inputs see the order"
    );
    // The emitter publishes the order it compiled; the oracle replays it.
    let em = emit_face(&ir, fixed(false), &[]);
    assert_eq!(em.orders.len(), 1, "one split fold: {:?}", em.orders);
    assert_eq!(em.orders[0].s, s);
    let orders: Vec<_> = em.orders.iter().map(|o| (o.func, o.fold, o.s)).collect();
    let replay = mapal_interp::run_ordered(&ir, 100_000_000, &orders);
    assert_eq!(printed(&replay.output), reference(m, n, k, s, false));
    // No order published off the fixed face.
    assert!(emit_face(&ir, EmitOpts::default(), &[]).orders.is_empty());
    assert!(emit_face(&ir, EmitOpts::exact(), &[]).orders.is_empty());
    let Some(cc) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    for contract in [false, true] {
        let (host, dev) = emit(&ir, contract, false);
        assert!(host.contains(&format!("; splitk={s}\n")));
        let out = run_emulated(&cc, &host, &dev);
        assert_eq!(
            printed(&out),
            reference(m, n, k, s, contract),
            "contract={contract}"
        );
        // Deterministic: the same bits on a second run.
        assert_eq!(run_emulated(&cc, &host, &dev), out);
        if !contract {
            // ADR-0040 D3: the compiled program is the oracle's replay, bit
            // for bit.
            assert_eq!(out, replay.output);
        }
    }
    let (host, dev) = emit(&ir, false, true);
    assert!(!host.contains("; splitk="));
    assert_eq!(run_emulated(&cc, &host, &dev), rr.output);
}

/// The default face (ADR-0040 D1): `fma` steps, order free — here the atomic
/// split (`splitk-atomic`: the output zeroed, every partition added into it).
/// Every printed cell is within D2's bound `γ_{K+1}·M` of the exact sum; the
/// rung's OFF arm is the fixed face's workspace split.
#[test]
fn default_face_is_within_the_a_priori_bound() {
    let (m, n, k) = (2, 64, 12);
    let ir = lower(&src(m, n, k));
    let em = emit_face(&ir, EmitOpts::default(), &[]);
    let (host, dev) = (em.text, em.device.expect("a device unit"));
    assert!(host.contains("; splitk=4 atomic\n"), "{host}");
    assert!(host.contains("call void @mapal_gpu_zero(i64 "));
    assert!(host.contains("declare void @mapal_gpu_zero(i64, i64)"));
    assert!(dev.contains("atomicrmw fadd ptr addrspace(1)"));
    assert!(!dev.contains(" fast ") && !dev.contains(" reassoc "));
    let off = emit_face(&ir, EmitOpts::default(), &["splitk-atomic"]).text;
    assert!(off.contains("; splitk=4\n"));
    assert!(!off.contains("@mapal_gpu_zero"));
    let Some(cc) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let out = printed(&run_emulated(&cc, &host, &dev));
    let u = (2.0f64).powi(-24);
    let gamma = |n: f64| n * u / (1.0 - n * u);
    // M = |seed| + Σ|a·b| = 2²⁴ + (k − 1) per cell; exact = M.
    let cell_m = 16_777_216.0 + (k - 1) as f64;
    for &got in &out[1..] {
        let err = (f64::from(got) - cell_m).abs();
        assert!(
            err <= gamma((k + 1) as f64) * cell_m,
            "got {got}, |err| {err}"
        );
    }
}

/// An `m × n × k` product of integer-valued f32 operands (`gen_tiles.py`'s
/// law): every partial sum is exact, so every order — and every correct
/// realization — prints the same bits; a lost or doubled piece does not.
fn mm_src(m: u64, n: u64, k: u64) -> String {
    format!(
        "fn main() {{
    {mk} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    {kn} -> iota -> tb;
    tb -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
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
    c[{mid}] -> println;
    c[{last}] -> println;
}}
",
        mk = m * k,
        kn = k * n,
        mn = m * n,
        mid = m * n / 2 + n / 3,
        last = m * n - 1,
    )
}

/// [`mm_src`]'s printed values, computed exactly.
fn mm_exact(m: u64, n: u64, k: u64) -> Vec<f32> {
    let a = |t: u64| ((t * 7 + 13) % 101) as f64 - 50.0;
    let b = |t: u64| ((t * 7 + 57) % 101) as f64 - 50.0;
    let cell = |t: u64| {
        let (i, j) = (t / n, t % n);
        (0..k).map(|kk| a(i * k + kk) * b(kk * n + j)).sum::<f64>() as f32
    };
    let total = (0..m * n).fold(0.0f32, |acc, t| acc + cell(t));
    vec![total, cell(0), cell(m * n / 2 + n / 3), cell(m * n - 1)]
}

/// S63 the combines of the staged register tile, run by the threaded
/// emulator (barriers, shared memory): the atomic split (RTX 5090, 32×64×256,
/// S = 4) and stream-K (H100 PCIe, 64×512×1024: 32 tiles × 32 panels on 114
/// blocks of 9 units since S64's `warp-fill` class split; S63: 16 tiles on 103
/// blocks of 5) print the exact values; the fixed face's workspace
/// split prints the oracle's replay of its compiled order, bit for bit.
#[test]
fn tiled_combines_emulated() {
    let Some(cc) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    for (target, (m, n, k), tag, replay) in [
        ("rtx-5090", (32, 64, 256), "; splitk=4 atomic\n", true),
        ("h100-pcie", (64, 512, 1024), "; streamk=114x9\n", false),
    ] {
        let ir = lower(&mm_src(m, n, k));
        let at = |base: EmitOpts| {
            let mut opts = EmitOpts { target, ..base };
            opts.gpu.off.place = true;
            // The S63 chooser (`fill-wave`'s arm: site_geometry.rs).
            opts.gpu.off.fill_wave = true;
            opts.gpu.off.fit_first = true;
            let em = emit_detailed(&ir, &opts).expect("emits");
            let dev = em.device.clone().expect("a device unit");
            (em, dev)
        };
        let (em, dev) = at(EmitOpts::default());
        assert!(em.text.contains(tag), "{target}: {}", em.text);
        let out = run_emulated(&cc, &em.text, &dev);
        assert_eq!(printed(&out), mm_exact(m, n, k), "{target} {tag}");
        if !replay {
            continue;
        }
        let (em, dev) = at(fixed(false));
        assert!(em.text.contains("; splitk=4\n"), "{}", em.text);
        let orders: Vec<_> = em.orders.iter().map(|o| (o.func, o.fold, o.s)).collect();
        assert_eq!(orders.len(), 1);
        let want = mapal_interp::run_ordered(&ir, 1_000_000_000, &orders);
        assert_eq!(run_emulated(&cc, &em.text, &dev), want.output);
    }
}
