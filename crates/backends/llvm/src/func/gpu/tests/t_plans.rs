//! Tests: cell tiles, matrix walks and cuts, the chooser and plan costs.

use super::*;

/// S62 integration: `cell-fill`'s one-output-per-thread tile under
/// `splitk` (reached with `fill-rows` off: the 5090 session's LOO arm at
/// M16 × 2048 × 7168) stores partition `z`'s partials at `z·rows·c` in the
/// workspace. Its scalar store dropped that offset, so every partition
/// raced on slab 0 (the box gate: not bit-exact, a different hash per
/// process — `benches/results/gpu/s62/integrated/rtx-5090`). S63: the
/// atomic variant of the same kernel adds into `%o` itself, no slab.
#[test]
fn cell_tile_split_stores_into_its_slab() {
    let s = site(16, 2048, 7168, true, true, Ty::f32());
    let mut arms = crate::GpuArms::default();
    arms.off.fill_rows = true;
    for contract in [false, true] {
        let l = launch_in(&s, &RTX_5090, contract, arms, None, Order::Fixed);
        let g = l.tiled.expect("tiled");
        let Combine::Workspace { s: parts } = l.combine else {
            panic!("the fixed order splits the cell tile: {:?}", l.combine)
        };
        assert_eq!((g.tm, g.tn), (1, 1), "the cell tile");
        for (c, slab_offset) in [
            (Combine::Workspace { s: parts }, true),
            (Combine::Atomic { s: parts }, false),
        ] {
            let text = kernel_text("k", &s, &RTX_5090, contract, arms, None, false, None, c);
            let z = text
                .lines()
                .find_map(|x| x.trim().split_once(" = udiv ").map(|(z, _)| z.to_string()))
                .expect("z = ctaid.y / gy");
            let slab = format!("{z}, {}", 16 * 2048);
            assert_eq!(
                text.lines()
                    .any(|x| x.contains(" = mul ") && x.contains(&slab)),
                slab_offset,
                "{c:?}: the workspace store offsets by z·rows·c, the atomic one not"
            );
            assert_eq!(
                text.contains("atomicrmw fadd ptr addrspace(1)"),
                !slab_offset,
                "{c:?}"
            );
            assert!(!text.contains(" fast ") && !text.contains(" reassoc "));
        }
    }
}

/// S63 the chooser (ADR-0040 D4) on the measured cells and parts
/// (`--nocapture` prints the table): the sequential order never cuts;
/// the fixed order cuts only into a workspace; the free order picks
/// stream-K, the atomic split, the workspace split or nothing, and each
/// rung's OFF arm is the next realization down. Every split stays inside
/// one wave and cuts whole panels (M1/G6).
#[test]
fn mma_pieces_on_the_measured_cells() {
    // f32-tensor-cores `mma-combine` on the rtx-5080's TF32 cells (ABBA,
    // `f32-tensor-cores.md`): the measured best cut where one won, none
    // where every cut lost; the OFF arm never cuts. bugs-properties
    // `cut-price`: 2048 × 4096 × 4096 takes the measured S = 4 (1.3346
    // against 1.4179 ms, `rung_replay.tsv`), priced with its partials;
    // the 2048³ MLP GEMM keeps its measured S = 8 (its adds in the L2).
    for (m, n, k, best) in [
        (128, 6144, 4096, 8),
        (128, 4096, 14336, 8),
        (128, 4096, 4096, 1),
        (128, 28672, 4096, 1),
        (2048, 4096, 4096, 4),
        (2048, 4096, 14336, 8),
        (2048, 6144, 4096, 1),
        (2048, 2048, 2048, 8),
    ] {
        let s = site(m, n, k, true, true, Ty::f32());
        let at = |arms| {
            launch_in(
                &s,
                &crate::profile::RTX_5080,
                false,
                arms,
                Some(Reduced::Tf32),
                Order::Free,
            )
            .combine
        };
        let on = at(crate::GpuArms::default());
        assert_eq!(on.splits(), best, "{m}x{n}x{k}: {on:?}");
        assert!(matches!(on, Combine::Atomic { .. } | Combine::None));
        let off = at(off(&["mma-combine"]));
        assert_eq!(off, Combine::None, "{m}x{n}x{k}");
    }
}

/// bf16 tensor-tile `mma-walk`: **every measured walk, replayed**
/// (`bf16-tensor-tile.md`: the RTX 5080, bf16 `fold: bf16 | f32`, the
/// model's projections, CUPTI medians of the ordered kernel against the
/// walk — `--gpu-on=mma-walk` where the price refuses it; hopper
/// `disasm-rented.md`: the RTX 3090's suite row and the A100's down
/// projection, same box ON / `--gpu-off=mma-walk`, and the 3090's PTX
/// JIT on the RTX 5080 at the 3090's grid; hopper-wgmma `walk-reuse`:
/// three RTX 5080 projections past its L2 keep, ABBA ON / OFF). The price
/// ([`MmaWalk::pays`]) must walk where the walk won and keep the ordered
/// grid (or its cut) where it lost. The Ampere walks lost on their reads:
/// the walk's rotation puts an operand slice's readers 8–16 units apart
/// where the 3090's 6 MiB / the A100's 20 MiB keep hold 2–3 units of the
/// walk's stream, so the walk re-read ~1 GB of 50 / 67 MB of operands
/// (1275 + 39 µs against 1082; 788 against 761) — the loop itself is
/// the ordered kernel's (64 HMMA, 16 LDSM, 8 LDGSTS a trip in both; the
/// ordered kernel's static 128 HMMA is LLVM's peel of its last trip).
/// The 5080's 64 MiB keep holds the same grid's readers (628.5 + 21.2
/// against 666.6), but not 3072 × 4096 × 8192's `a` readers (33–38
/// units apart, 25 held): DRAM-bound, 1829 + 40 against 1836. Open:
/// 512 × 2048 × 2048 and 2048 × 512 × 2048 (the ordered 64 × 128 tile
/// of 32 × 32 warp tiles runs at 85 % of the 128 × 128 tile's pipe rate,
/// which the price reads as equal; the walk won 0.876 / 0.854 where the
/// price, at the pipe for both, sees 1.02): the warp tile's realized
/// efficiency is the missing property; and the 3090's grid on the 5080
/// (a 2.5 % win inside the price's 5 % margin).
#[test]
fn mma_walk_replay() {
    let mut open = Vec::new();
    for a in &WALK_ARMS {
        let gpu = a.part.gpu().expect("a GPU");
        let (price, rate) = a.price(gpu);
        let (o, c, w) = price.ns(gpu, rate);
        let [m, n, k] = a.mnk;
        eprintln!(
            "walk {} {m}x{n}x{k}: priced {:.1} / {:.1} / {:.1} us ({:.3}; reads {:.0} MB), measured {} / {} ({:.3})",
            a.part.name,
            o / 1000.0,
            c / 1000.0,
            w / 1000.0,
            w / o.min(c),
            price.walk_bytes(gpu.l2_keep_bytes()) as f64 / 1e6,
            a.ord,
            a.walk,
            a.walk / a.ord
        );
        if price.pays(gpu, rate) != a.won() {
            open.push((a.part.name, a.mnk));
        }
    }
    const OPEN: [(&str, [u64; 3]); 3] = [
        ("rtx-5080", [512, 2048, 2048]),
        ("rtx-5080", [2048, 512, 2048]),
        // The 3090's grid on the 5080: the walk won 2.5 %, inside the
        // 5 % the price asks.
        ("rtx-5080", [2048, 4096, 4096]),
    ];
    assert_eq!(
        open,
        OPEN.to_vec(),
        "measured walks the price does not pick"
    );
}

/// hopper-wgmma `walk-reuse`: **a walk's operand reads, against the
/// keep** — over generated grids (tiles, panels, the walk's blocks and
/// iters as [`mma_walk`] derives them), each slice is read at least
/// once and at most once a reader; a keep holding the whole walk's
/// stream reads every slice once; the reads never grow with the keep.
#[test]
fn walk_reuse_bounds() {
    let mut runner =
        proptest::test_runner::TestRunner::new(proptest::test_runner::Config::with_cases(64));
    let shape = (1u64..=24, 1u64..=48, 2u64..=96, 40u64..=200);
    runner
        .run(&shape, |(gy, gx, panels, slots)| {
            let tiles = gx * gy;
            let iters = (tiles * panels).div_ceil(slots);
            let ctas = (tiles * panels).div_ceil(iters);
            let (sa, sb) = (128 * 32 * 2, 128 * 32 * 2);
            let w = MmaWalk {
                ordered: (tiles, 1),
                cut: 1,
                tiles,
                panels,
                panel_flops: 1,
                blocks_per_sm: 2,
                ctas,
                iters,
                tile_cells: 128 * 128,
                cells: tiles * 128 * 128,
                bytes: (gy * sa + gx * sb) * panels,
                grid_x: gx,
                slices: (sa, sb),
            };
            let once = (gy * sa + gx * sb) * panels;
            let every = tiles * (sa + sb) * panels;
            let mut last = u64::MAX;
            for keep in [0, 1 << 20, 6 << 20, 64 << 20, u64::MAX / 4] {
                let r = w.operand_bytes(keep);
                prop_assert!(once <= r && r <= every, "{r} outside [{once}, {every}]");
                prop_assert!(r <= last, "more reads with a larger keep");
                last = r;
            }
            prop_assert_eq!(w.operand_bytes(u64::MAX / 4), once);
            prop_assert_eq!(w.walk_bytes(u64::MAX / 4), w.bytes);
            Ok(())
        })
        .expect("walk reads within their bounds");
}

/// bugs-properties `cut-price`: **every measured `mma-combine` arm of
/// cross-arch 5fcb401b, replayed** (`analysis/arms.md`, same-box OFF /
/// ON, the arm's plan line: its tile, grid and cut `S`) — the priced cut
/// ([`MmaCut::ns`]) must keep the measured side: cut where ON won, whole
/// where it lost (a null either way). The wave count alone cut all 20
/// (9 lost). Open: the 4090's 128 × 6144 × 4096 (bf16 / TF32, priced
/// 1.18 / 1.14 and cut as before: its `b` stream at the DRAM rate bounds
/// both arms, 55 / 105 µs against 44 / 83 µs of waves; a DRAM floor in
/// the price misprices the A100's under-filled 128 × 4096 × 14336, whose
/// uncut kernel runs at 2.8× its peak estimate — the tensor tile's
/// efficiency at its occupancy is the missing fact) — `OPEN` below.
#[test]
fn mma_cut_replay() {
    use crate::profile::{A100, RTX_3090, RTX_5080};
    // part, m, n, k, tf32 (else bf16), mt, nt, warps_x, warps_y, bk,
    // stages, blocks an SM, S, OFF / ON.
    let arms: [(&TargetProfile, u64, u64, u64, bool, [u64; 8], f64); 21] = [
        (
            &RTX_3090,
            2048,
            2048,
            2048,
            true,
            [4, 8, 2, 2, 16, 2, 2, 8],
            0.946,
        ),
        (
            &RTX_3090,
            2048,
            2048,
            2048,
            true,
            [4, 8, 2, 2, 16, 2, 2, 8],
            0.943,
        ),
        (
            &RTX_3090,
            128,
            4096,
            14336,
            false,
            [2, 2, 4, 2, 64, 2, 2, 8],
            0.928,
        ),
        (
            &RTX_3090,
            128,
            4096,
            14336,
            true,
            [2, 2, 4, 2, 32, 2, 2, 8],
            1.074,
        ),
        (
            &RTX_3090,
            128,
            6144,
            4096,
            false,
            [4, 4, 2, 2, 16, 2, 4, 8],
            1.273,
        ),
        (
            &RTX_3090,
            128,
            6144,
            4096,
            true,
            [4, 4, 2, 2, 8, 2, 4, 8],
            1.494,
        ),
        (
            &A100,
            2048,
            2048,
            2048,
            true,
            [4, 8, 2, 2, 32, 2, 2, 4],
            0.836,
        ),
        (
            &A100,
            2048,
            2048,
            2048,
            true,
            [4, 8, 2, 2, 32, 2, 2, 4],
            0.914,
        ),
        (
            &A100,
            128,
            4096,
            14336,
            false,
            [2, 2, 4, 2, 128, 2, 2, 8],
            1.106,
        ),
        (
            &A100,
            128,
            4096,
            14336,
            true,
            [2, 2, 4, 2, 64, 2, 2, 8],
            1.342,
        ),
        (
            &A100,
            2048,
            4096,
            14336,
            false,
            [4, 8, 2, 2, 64, 2, 2, 2],
            1.090,
        ),
        (
            &A100,
            2048,
            4096,
            14336,
            true,
            [4, 8, 2, 2, 32, 2, 2, 8],
            0.956,
        ),
        (
            &RTX_4090,
            128,
            4096,
            14336,
            false,
            [2, 2, 4, 2, 64, 2, 2, 2],
            1.087,
        ),
        (
            &RTX_4090,
            128,
            4096,
            14336,
            true,
            [2, 2, 4, 2, 32, 2, 2, 2],
            1.020,
        ),
        (
            &RTX_4090,
            128,
            6144,
            4096,
            false,
            [2, 2, 4, 2, 64, 2, 2, 4],
            0.864,
        ),
        (
            &RTX_4090,
            128,
            6144,
            4096,
            true,
            [2, 2, 4, 2, 32, 2, 2, 4],
            0.922,
        ),
        (
            &H100,
            128,
            6144,
            4096,
            false,
            [2, 2, 4, 2, 128, 3, 2, 4],
            0.899,
        ),
        // The rtx-5080's (f32-tensor-cores.md, ABBA; ON / OFF inverted).
        (
            &RTX_5080,
            2048,
            2048,
            2048,
            true,
            [4, 8, 2, 2, 16, 2, 2, 8],
            1.0 / 0.945,
        ),
        (
            &RTX_5080,
            128,
            6144,
            4096,
            true,
            [2, 4, 4, 2, 16, 3, 2, 8],
            0.2389 / 0.1815,
        ),
        (
            &RTX_5080,
            128,
            4096,
            14336,
            true,
            [2, 2, 4, 2, 32, 2, 2, 8],
            0.4541 / 0.4163,
        ),
        (
            &RTX_5080,
            2048,
            4096,
            14336,
            true,
            [4, 8, 2, 2, 16, 2, 2, 8],
            4.90 / 4.43,
        ),
    ];
    const OPEN: [(&str, u64, u64, bool); 2] = [
        ("rtx-4090", 6144, 4096, true),
        ("rtx-4090", 6144, 4096, false),
    ];
    let (mut kept, mut open) = (0, 0);
    for (p, m, n, k, tf32, [mt, nt, wx, wy, bk, stages, bps, s], meas) in arms {
        let g = p.gpu().expect("gpu");
        let fmt = if tf32 { MmaFmt::Tf32 } else { MmaFmt::Bf16 };
        let op = g.caps.matrix_op(fmt, Home::Regs).expect("a warp product");
        let (bm, bn) = (MMA_M * mt * wy, MMA_N * nt * wx);
        let cut = MmaCut {
            blocks: (m / bm) * (n / bn),
            slots: g.sms * bps,
            panels: k / bk,
            stages,
            panel_flops: 2 * bm * bn * bk,
            blocks_per_sm: bps,
            cells: m * n,
            bytes: m * k * 4 + k * n * if tf32 { 4 } else { 2 } + m * n * 4,
        };
        let rate = g.matrix_flops_per_ns(&op);
        let priced = cut.ns(g, rate, 1) / cut.ns(g, rate, s);
        let cuts = priced * MMA_CUT_PM as f64 >= 1000.0;
        eprintln!(
            "{} {m}x{n}x{k} {} S={s}: priced {priced:.2}, measured {meas:.3}",
            p.name,
            if tf32 { "tf32" } else { "bf16" }
        );
        if (meas - 1.0).abs() <= 0.03 {
            continue;
        }
        if OPEN.contains(&(p.name, n, k, tf32)) && m == 128 {
            assert_ne!(cuts, meas > 1.0, "{} {m}x{n}x{k}: no longer open", p.name);
            open += 1;
            continue;
        }
        assert_eq!(
            cuts,
            meas > 1.0,
            "{} {m}x{n}x{k}: priced {priced:.2}",
            p.name
        );
        kept += 1;
    }
    assert_eq!((kept, open), (18, 2));
}

#[test]
fn the_chooser_on_the_measured_cells() {
    let cells = [
        (16, 2048, 7168),
        (16, 1024, 4096),
        (16, 18432, 7168),
        (1, 2048, 7168),
        (1, 1024, 4096),
        (128, 2048, 7168),
        (128, 1024, 4096),
        (4096, 4096, 4096),
    ];
    let parts = [&RTX_5090, &RTX_4090, &H100_PCIE, &crate::profile::RTX_3090];
    // The rules (plan-cost's OFF arm; `plan_cost_picks_the_argmin` is ON's).
    let mut arms = crate::GpuArms::default();
    arms.off.plan_cost = true;
    let off = |names: &[&str]| {
        let mut a = arms;
        for n in names {
            *a.off.rung(n).expect("a rung") = true;
        }
        a
    };
    for p in parts {
        let g = p.gpu().unwrap();
        for (m, n, k) in cells {
            let s = site(m, n, k, true, true, Ty::f32());
            for contract in [false, true] {
                let at = |a, o| launch_in(&s, p, contract, a, None, o);
                let l = at(arms, Order::Free);
                let unit = l.tiled.map_or(1, |t| t.bk);
                let (blocks, wave) = (l.gx * l.gy, l.wave(g));
                // The fill class the rung rules read (`launch`'s `wps`).
                let base = p.gpu_tile(m, n, k, &s.elem, false, false);
                let (fb, ft) = base.map_or(((m * n).div_ceil(NAIVE_THREADS), NAIVE_THREADS), |t| {
                    ((m / t.bm()) * (n / t.bn()), t.threads())
                });
                let fill = if g.warps_per_scheduler(fb, ft) < GPU_RESIDENT_BLOCKS {
                    "under"
                } else {
                    "filled"
                };
                println!(
                    "{} {m}x{n}x{k} contract={contract} fill={fill} kernel={} grid={}x{} \
                         block={} blocks={blocks} wave={wave} free={:?} fixed={:?}",
                    p.name,
                    if l.tiled.is_some() { "tiled" } else { "naive" },
                    l.gx,
                    l.gy,
                    l.block,
                    l.combine,
                    at(arms, Order::Fixed).combine,
                );
                assert_eq!(at(arms, Order::Sequential).combine, Combine::None);
                for o in [Order::Free, Order::Fixed] {
                    // (A `fill-wave` split's wave is its own launch's.)
                    let lo = at(arms, o);
                    let c = lo.combine;
                    let parts_ = c.splits();
                    assert!(parts_ == 1 || blocks * parts_ <= lo.wave(g), "{c:?}");
                    assert!(k.is_multiple_of(parts_ * unit), "{c:?}");
                    if let Combine::StreamK { ctas, iters, .. } = c {
                        assert!(ctas <= wave && ctas * iters >= blocks * (k / unit));
                    }
                }
                assert!(matches!(
                    at(arms, Order::Fixed).combine,
                    Combine::None | Combine::Workspace { .. }
                ));
                // Each OFF arm is the next realization down.
                let no_sk = at(off(&["streamk"]), Order::Free).combine;
                assert!(!matches!(no_sk, Combine::StreamK { .. }));
                let no_at = at(off(&["streamk", "splitk-atomic"]), Order::Free).combine;
                assert_eq!(no_at, at(arms, Order::Fixed).combine);
                let none = at(off(&["streamk", "splitk-atomic", "splitk"]), Order::Free);
                assert_eq!(none.combine, Combine::None);
                let base = launch(&s, p, contract, arms, None);
                let geom = |l: &Launch| (l.tiled, l.gx, l.gy, l.block, l.smem, l.raster);
                assert_eq!(geom(&none), geom(&base), "the order never moves the tile");
            }
        }
    }
    // The one-wave clamp (G6): the 3090 at 16 × 18432 × 7168 holds 984
    // blocks of the 1152-block grid; no split fits one wave. Stream-K
    // would take the tail; its measured row (Filled) ships the grid —
    // where the grid fills the part. large-gemm `streamk-fill`: its 144
    // one-warp tiles give the part fewer than its 656 fill warps, so the
    // `fill-wave` split fills it (S = 7, 1008 blocks); the arm off, the
    // one-wave stream-K.
    let r3090 = &crate::profile::RTX_3090;
    let s = site(16, 18432, 7168, true, true, Ty::f32());
    let l = launch_in(&s, r3090, true, arms, None, Order::Free);
    assert_eq!(l.combine, Combine::Atomic { s: 7 }, "{:?}", l.combine);
    let l = launch_in(&s, r3090, true, off(&["streamk-fill"]), None, Order::Free);
    assert_eq!(l.combine.splits(), 1, "{:?}", l.combine);
    let s16 = site(16, 2048, 7168, true, true, Ty::f32());
    // S63 the shipped default per part (`GPU_RUNGS_OFF`, measured,
    // `benches/results/gpu/s63/default-float/`): each is a session arm
    // (site-geometry `fill-wave` off: the S63 chooser; its own arm is
    // `site_geometry.rs`'s; sweep-gemm-2 `streamk-price` postdates it, as
    // f32-tensor-cores `split-tile`: the H100's measured `mma.sync` rate
    // takes the 4096³ cell's tf32x3 tile, tensor-cores-2).
    let s63 = off(&[
        "fill-wave",
        "k-fill",
        "streamk-fill",
        "streamk-price",
        "split-tile",
    ]);
    let shipped = |p, (m, n, k)| {
        let c = launch_in(
            &site(m, n, k, true, true, Ty::f32()),
            p,
            true,
            s63,
            None,
            Order::Free,
        )
        .combine;
        match c {
            Combine::None => "none",
            Combine::Workspace { .. } => "workspace",
            Combine::Atomic { .. } => "atomic",
            Combine::StreamK { .. } => "streamk",
        }
    };
    // hardware-properties: the rules' picks (stream-K filled, `splitk`
    // off on the exact face below 128 KiB of L2 a SM, `splitk-atomic` on
    // every part) — the measured cells' winners are replayed in
    // `rung_replay.tsv`.
    let h100 = &H100_PCIE;
    for (p, cell, want) in [
        (&RTX_5090, (16, 2048, 7168), "none"),
        (&RTX_5090, (16, 18432, 7168), "streamk"),
        (&RTX_5090, (4096, 4096, 4096), "streamk"),
        (&RTX_4090, (16, 18432, 7168), "streamk"),
        (h100, (16, 1024, 4096), "atomic"),
        (h100, (16, 2048, 7168), "atomic"),
        (h100, (128, 2048, 7168), "none"),
        (h100, (4096, 4096, 4096), "streamk"),
        (r3090, (16, 1024, 4096), "atomic"),
        (r3090, (16, 2048, 7168), "streamk"),
        (r3090, (128, 2048, 7168), "atomic"),
        (r3090, (4096, 4096, 4096), "streamk"),
    ] {
        assert_eq!(shipped(p, cell), want, "{} {cell:?}", p.name);
    }
    // The measured table: the 3090 ships no workspace split at (filled,
    // exact) — its one split cell there lost (S62) — on the fixed face.
    let pre = off(&[
        "row-fit",
        "wave-gate",
        "grid-share",
        "panel-regs",
        "fill-rows",
        "cell-fill",
    ]);
    let l = launch_in(&s16, r3090, false, pre, None, Order::Fixed);
    assert_eq!(l.combine, Combine::None);
    let l = launch_in(&s16, r3090, true, pre, None, Order::Fixed);
    assert!(matches!(l.combine, Combine::Workspace { .. }));
}

/// plan-cost: on a part with plan rates, a free-order site's fold is the
/// cheapest candidate the rungs admit — never dearer than the rules' pick
/// (a candidate itself) — every candidate cuts `k` into whole panels, and
/// each rung's OFF arm removes its candidates; on a part without the
/// rates, and on the other orders, the plan is the rules' byte for byte
/// (D8: the rules' launch, every cell, part, face and order).
#[test]
fn plan_cost_picks_the_argmin() {
    let cells = [
        (16, 2048, 7168),
        (16, 1024, 4096),
        (128, 2048, 7168),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];
    // The rung forced on: the 4090/5090 rows ship it off (plan-cost stopgap).
    let mut on = crate::GpuArms::default();
    on.on.plan_cost = true;
    let mut rules = on;
    rules.on.plan_cost = false;
    rules.off.plan_cost = true;
    let mut no_atomic = on;
    no_atomic.off.splitk_atomic = true;
    no_atomic.off.streamk = true;
    let mut costed = 0;
    // The launch but for the rung set the arms differ in.
    let same = |a: Launch, b: Launch| {
        Launch {
            rungs: b.rungs,
            ..a
        } == b
    };
    for p in crate::profile::GPU_PARTS {
        for (m, n, k) in cells {
            let s = site(m, n, k, true, true, Ty::f32());
            for contract in [false, true] {
                let l = launch(&s, p, contract, on, None);
                let cands = plan_cands(&l, &s, p, contract, on, Order::Free);
                for o in [Order::Sequential, Order::Fixed] {
                    assert!(plan_cands(&l, &s, p, contract, on, o).is_empty());
                    let at = |a| launch_in(&s, p, contract, a, None, o);
                    assert!(same(at(on), at(rules)), "{} {o:?}", p.name);
                }
                let at = |a| launch_in(&s, p, contract, a, None, Order::Free);
                if p.plan_rates().is_none() {
                    assert!(cands.is_empty(), "{}", p.name);
                    assert!(same(at(on), at(rules)), "D8: {} {m}x{n}x{k}", p.name);
                    continue;
                }
                costed += 1;
                let pick = cands.first().expect("the ordered kernel is a candidate");
                assert_eq!(at(on).combine, pick.combine);
                let rule = at(rules).combine;
                let rule_ns = cands.iter().find(|c| c.combine == rule).expect("rules'");
                assert!(pick.cost.ns <= rule_ns.cost.ns);
                assert!(cands.windows(2).all(|w| w[0].cost.ns <= w[1].cost.ns));
                for c in &cands {
                    let r = resized(&s, p, contract, on, l, c.combine);
                    let bk = r.tiled.map_or(1, |t| t.bk);
                    assert!(k.is_multiple_of(c.combine.splits() * bk), "{c:?}");
                }
                let l = launch(&s, p, contract, no_atomic, None);
                let few = plan_cands(&l, &s, p, contract, no_atomic, Order::Free);
                assert!(
                    few.iter()
                        .all(|c| matches!(c.combine, Combine::None | Combine::Workspace { .. }))
                );
            }
        }
    }
    assert!(
        costed >= 20,
        "the calibrated parts reach the chooser ({costed})"
    );
}

/// A forced count (`--splitk=S`, the part's `Geom::split`) stays the plan
/// under plan-cost: the chooser ranks only that count's splits, so the
/// ordered kernel or the rules' pick never replaces it.
#[test]
fn a_forced_split_is_the_only_plan() {
    let mut forced = crate::GpuArms::default();
    forced.splitk = 4;
    forced.on.plan_cost = true;
    let mut ranked = 0;
    for p in [&RTX_4090, &RTX_5090] {
        for (m, n, k) in [(16, 2048, 7168), (512, 512, 512), (4096, 4096, 4096)] {
            let s = site(m, n, k, true, true, Ty::f32());
            let l = launch(&s, p, true, forced, None);
            let cands = plan_cands(&l, &s, p, true, forced, Order::Free);
            let got = launch_in(&s, p, true, forced, None, Order::Free).combine;
            assert!(cands.iter().all(|c| c.combine.splits() == 4), "{cands:?}");
            if !cands.is_empty() {
                ranked += 1;
                assert_eq!(got.splits(), 4, "{} {m}x{n}x{k}", p.name);
            }
        }
    }
    assert!(ranked > 0, "the forced count reaches the chooser");
}

/// plan-cost `linear` (D6): a site table's `S` keeps each product's depth
/// — `k/S` roundings in its partial, `S·T − 1` adds into the zeroed sum —
/// within the source's `k + T − 1` (`S·T ≤ k` for `S > 1`: `k = 4, T = 3`
/// takes no `S = 2`), over skinny and short-`k` sites.
#[test]
fn site_table_split_stays_within_the_source_bound() {
    // `linear` forced on: the rules ship it off (no measured gain).
    let mut on = crate::GpuArms::default();
    on.on.linear = true;
    let (mut tables, mut split) = (0, 0);
    for p in [&RTX_4090, &RTX_5090] {
        for (m, n, k) in [1u64, 2, 16]
            .iter()
            .flat_map(|&m| [32u64, 128, 1024].map(move |n| (m, n)))
            .flat_map(|(m, n)| [4u64, 6, 8, 12, 16, 24, 64, 1024].map(move |k| (m, n, k)))
        {
            let s = site(m, n, k, true, true, Ty::f32());
            for t in 2..=7 {
                let bound = mapal_ir::bound_coef(k + t - 1, false);
                let Some((sp, _)) = concat_plan(&s, p, true, on, t, bound) else {
                    continue;
                };
                tables += 1;
                split += u64::from(sp > 1);
                assert!(
                    k / sp + sp * t - 1 <= k + t - 1,
                    "{} {m}x{n}x{k} T={t}: S={sp}",
                    p.name
                );
            }
        }
    }
    assert!(tables > 0 && split > 0, "{tables} tables, {split} split");
}

/// plan-cost §5 pick stability, over sites of the generator's sizes
/// (`testgen::geom_prog_strategy`: rows 1…1024, c and k 64…4096): ±10 %
/// on any one rate moves every candidate's time by at most the factor the
/// rate does, so it changes a pick only where the runner-up the plan line
/// reports is within that band (`alt.ns ≤ 1.1/0.9 · pick.ns`).
#[test]
fn plan_pick_is_stable_under_rate_noise() {
    let on = crate::GpuArms::default();
    let (mut flips, mut checked) = (0, 0);
    for p in [&RTX_4090, &RTX_5090] {
        let g = *p.gpu().unwrap();
        for (m, n, k) in [1u64, 16, 64, 128, 256, 512, 1024]
            .iter()
            .flat_map(|&m| [64u64, 256, 1024, 4096].map(move |n| (m, n)))
            .flat_map(|(m, n)| [64u64, 512, 4096].map(move |k| (m, n, k)))
        {
            let s = site(m, n, k, true, true, Ty::f32());
            let l = launch(&s, p, true, on, None);
            let cands = plan_cands(&l, &s, p, true, on, Order::Free);
            let (Some(pick), Some(alt)) = (cands.first(), cands.get(1)) else {
                continue;
            };
            // The rates (the physical-property names after `clock_khz`
            // reshape the part, not its rates).
            for (i, name) in crate::profile::DeviceCost::NAMES
                .iter()
                .enumerate()
                .take(24)
            {
                let v = g.rate(name);
                for f in [0.9, 1.1] {
                    let mut over = [None; crate::profile::DeviceCost::NAMES.len()];
                    over[i] = Some(((v as f64) * f).round() as u64);
                    let q = TargetProfile {
                        machine: crate::profile::Machine::Gpu(g.with_cost(&over)),
                        ..*p
                    };
                    let lq = launch(&s, &q, true, on, None);
                    let moved = plan_cands(&lq, &s, &q, true, on, Order::Free);
                    checked += 1;
                    if moved.first().map(|c| c.combine) != Some(pick.combine) {
                        flips += 1;
                        assert!(
                            alt.cost.ns <= pick.cost.ns * 1.1 / 0.9 + 1.0,
                            "{} {m}x{n}x{k} {name}×{f}: {} → {:?}, runner-up {} at {} vs {}",
                            p.name,
                            pick.label(),
                            moved.first().map(|c| c.label()),
                            alt.label(),
                            alt.cost.ns,
                            pick.cost.ns
                        );
                    }
                }
            }
        }
    }
    eprintln!("plan-cost stability: {flips} flips in {checked} perturbed picks");
    assert!(checked > 1000);
}
