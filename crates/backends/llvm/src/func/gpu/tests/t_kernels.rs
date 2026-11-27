//! Tests: plan lines, kernel text properties, banks and swizzle, the off arms.

use super::*;

/// S64 evidence pass: each kernel's `; mapal.plan` line reads back as the
/// launch it was emitted from (the chooser's cells and parts, both step
/// faces, every realization) — the intent `benches/gpu/evidence.py` checks.
#[test]
fn plan_line_reads_back_as_its_launch() {
    let cells = [
        (16, 2048, 7168),
        (16, 18432, 7168),
        (128, 2048, 7168),
        (4096, 4096, 4096),
    ];
    let parts = [&RTX_5090, &RTX_4090, &H100_PCIE, &crate::profile::RTX_3090];
    // The FMA tile's launches: `split-tile` (tensor-cores-2: the H100's
    // measured TF32 rate takes 4096³) has its own read-back in
    // `tests/split_tile.rs`.
    let mut arms = crate::GpuArms::default();
    *arms.off.rung("split-tile").expect("a rung") = true;
    for p in parts {
        for (m, n, k) in cells {
            let s = site(m, n, k, true, true, Ty::f32());
            for (contract, order) in [
                (false, Order::Sequential),
                (true, Order::Free),
                (false, Order::Fixed),
            ] {
                let l = launch_in(&s, p, contract, arms, None, order);
                let text = kernel_text("k", &s, p, contract, arms, None, false, None, l.combine);
                let line = text.lines().next().expect("a plan line");
                let f: std::collections::HashMap<&str, &str> = line
                    .strip_prefix("; mapal.plan ")
                    .expect("the first line")
                    .split(' ')
                    .filter_map(|kv| kv.split_once('='))
                    .collect();
                let n = |key: &str| f[key].parse::<u64>().expect(key);
                assert_eq!(
                    (n("gx"), n("gy"), n("block"), n("smem"), n("raster")),
                    (l.gx, l.gy, l.block, l.smem, l.raster),
                    "{line}"
                );
                let g = l.tiled.expect("the FMA tile");
                assert_eq!(
                    (
                        n("tm"),
                        n("tn"),
                        n("tx"),
                        n("ty"),
                        n("bk"),
                        n("stages"),
                        n("blocks_per_sm")
                    ),
                    (g.tm, g.tn, g.tx, g.ty, g.bk, g.stages, g.blocks_per_sm)
                );
                assert_eq!(n("S"), l.combine.splits());
                let on: Vec<&str> = f["rungs_on"].split(',').collect();
                for r in KERNEL_RUNGS {
                    assert_eq!(on.contains(&r), l.rungs.get(r), "{r} in {line}");
                }
                assert_eq!(f["face"], format!("{:?}", l.arithmetic));
            }
        }
    }
}

/// S63 `streamk`: the kernel walks its share of the `tiles × panels`
/// space piece by piece — one panel loop per piece entered from `seg`,
/// the seed only on a tile's first piece, the store an atomic add, the
/// latch back to `seg` — and keeps the register-staged tile's panel loop.
#[test]
fn streamk_kernel_walks_pieces() {
    let s = site(128, 2048, 7168, true, true, Ty::f32());
    let arms = crate::GpuArms::default();
    let l = launch(&s, &RTX_5090, true, arms, None);
    let g = l.tiled.expect("tiled");
    if g.stages >= 2 {
        return;
    }
    let panels = 7168 / g.bk;
    let blocks = l.gx * l.gy;
    let iters = (blocks * panels).div_ceil(l.wave(RTX_5090.gpu().unwrap()));
    let c = Combine::StreamK {
        ctas: (blocks * panels).div_ceil(iters),
        iters,
        whole: 0,
    };
    let text = kernel_text("k", &s, &RTX_5090, true, arms, None, false, None, c);
    assert!(text.contains("seg:\n") && text.contains("segl:\n") && text.contains("exit:\n"));
    assert!(text.contains("label %seg, label %exit"));
    assert!(text.contains(&format!(", {iters}\n")), "start = cta·iters");
    assert!(
        text.contains(", %seg], ["),
        "the panel loop is entered per piece"
    );
    assert!(text.contains("atomicrmw fadd ptr addrspace(1)"));
    assert!(
        !text
            .lines()
            .any(|x| x.trim_start().starts_with("store") && x.contains("addrspace(1)")),
        "no plain output store"
    );
    assert!(text.contains("select i1"), "the seed rides p0 = 0");
    assert!(!text.contains("@SEG_NEXT@") && !text.contains("@NEXT"));
}

/// The block map is a bijection on the grid for every height it is
/// derived for (a power of two dividing `gy`), evaluated by the same code
/// that emits it.
#[test]
fn raster_block_is_a_bijection_on_the_grid() {
    for (gx, gy) in [(2, 2), (12, 8), (64, 64), (32, 16), (3, 32), (1, 4)] {
        for h in (0..=u64::ilog2(gy)).map(|p| 1u64 << p) {
            let mut seen = vec![false; (gx * gy) as usize];
            for by in 0..gy {
                for bx in 0..gx {
                    let (c, r) = raster_block(&mut Num, &bx, &by, gx, h);
                    assert!(c < gx && r < gy, "{gx}x{gy} h={h}");
                    assert!(!std::mem::replace(&mut seen[(r * gx + c) as usize], true));
                }
            }
            // Within a group, consecutive launches walk down a column.
            let (c0, r0) = raster_block(&mut Num, &0, &0, gx, h);
            let (c1, r1) = raster_block(&mut Num, &1, &0, gx, h);
            if h > 1 && gx > 1 {
                assert_eq!((c0, r0, c1, r1), (0, 0, 0, 1));
            }
        }
    }
}

/// `devmap`: one thread per cell, the block the naive kernel's, the grid
/// the fewest blocks that cover the extent; a cell's id never names a gen
/// kernel of the same body (different key, different prefix anyway).
#[test]
fn cell_grid_covers_the_extent_one_thread_per_cell() {
    for n in [1, 255, 256, 257, 128 * 128, 4096 * 4096] {
        let (gx, block) = cell_grid(n);
        assert_eq!(block, NAIVE_THREADS);
        assert!(gx * block >= n && (gx - 1) * block < n, "n={n}");
    }
    let p = crate::profile::resolve("rtx-4090").expect("profile");
    assert_ne!(cell_id("fn2", 16384, p), gen_id("fn2", 16384, p));
    let caps = [
        CapArg::Ptr,
        CapArg::Law("fn3".to_string(), "float".to_string()),
    ];
    let k = cell_kernel_text(
        1,
        "fn2_cell",
        300,
        "float",
        "{ ptr, ptr, i32 }",
        None,
        &caps,
        CellGeo::ONE,
    );
    assert!(
        k.contains("(ptr addrspace(1) %o, ptr addrspace(1) %c0)"),
        "{k}"
    );
    assert!(k.contains("%v1 = call float @fn3(i32 %e)"), "{k}");
    assert!(
        k.contains("@fn2_cell({ ptr, ptr, i32 } %a, float %v1)"),
        "{k}"
    );
    assert!(k.contains("icmp ult i64 %i, 300"), "{k}");
    // A uniform rides the launch's 8-byte slot and is narrowed in the cell.
    let caps = [CapArg::Uniform("i32".into()), CapArg::Ptr];
    let k = cell_kernel_text(
        2,
        "fn4",
        8,
        "float",
        "{ i32, ptr, i32 }",
        None,
        &caps,
        CellGeo::ONE,
    );
    assert!(
        k.contains("(ptr addrspace(1) %o, i64 %c0, ptr addrspace(1) %c1)"),
        "{k}"
    );
    assert!(k.contains("%g0 = trunc i64 %c0 to i32"), "{k}");
    assert!(
        k.contains("%a0 = insertvalue { i32, ptr, i32 } poison, i32 %g0, 0"),
        "{k}"
    );
}

/// `idx32`: every S52 site indexes in `i32`; an operand past 2³¹ elements
/// keeps `i64`.
#[test]
fn idx32_takes_i32_exactly_when_every_index_fits() {
    for p in PARTS {
        for s in s52_sites() {
            let l = launch(&s, p, false, force(GpuRungs::ALL), None);
            assert!(
                max_index(&s, &l, Combine::None) < 1 << 31,
                "{} {s:?}",
                p.name
            );
            let k = kernel_text(
                "k",
                &s,
                p,
                false,
                force(GpuRungs::ALL),
                None,
                false,
                None,
                Combine::None,
            );
            assert!(k.contains("add nuw nsw i32"), "{} {s:?}", p.name);
        }
    }
    // 2¹⁶ rows of 2¹⁶: `a`'s last element is 2³² − 1.
    let big = site(1 << 16, 1 << 16, 1 << 16, true, true, Ty::f32());
    let l = launch(&big, &RTX_4090, false, force(GpuRungs::ALL), None);
    assert!(max_index(&big, &l, Combine::None) >= 1 << 31);
    let k = kernel_text(
        "k",
        &big,
        &RTX_4090,
        false,
        force(GpuRungs::ALL),
        None,
        false,
        None,
        Combine::None,
    );
    assert!(!k.contains("nuw nsw i32") && k.contains("add i64"));
}

/// `affine`: for every S52 site, part and arm, every copy `p` of both
/// panels is copy 0 plus a constant — the lane map evaluated at
/// `tid + p·threads` equals the `udiv`/`urem` chain at `tid` plus the delta,
/// for every `tid` ([`affine_delta`] checks exactly that).
#[test]
fn affine_decomposition_equals_udiv_urem_for_every_copy_and_thread() {
    let mut copies = 0;
    for p in PARTS {
        for r in arms() {
            for s in s52_sites() {
                let Some(g) = launch(&s, p, false, force(r), None).tiled else {
                    continue;
                };
                for (map, n) in lane_maps(&s, &g, g.kmajor, false) {
                    for c in 0..n {
                        let d = affine_delta(map, g.threads(), c);
                        assert!(d.is_some(), "{} {s:?} {map:?} copy {c}", p.name);
                        copies += 1;
                    }
                }
            }
        }
    }
    assert!(
        copies > 1000,
        "the sweep reached the tiled kernels: {copies}"
    );
}

proptest! {
    /// Power-of-two extents make every lane map periodic in `threads`
    /// (`Q = threads / l` divides `rows` or `rows` divides `Q`).
    #[test]
    fn power_of_two_lane_maps_are_affine(
        t in 5u32..11, l in 0u32..4, rows in 0u32..9, unit in 0u32..3, p in 0u64..64,
    ) {
        let (threads, l, rows, unit) = (1u64 << t, 1u64 << l, 1u64 << rows, 1u64 << unit);
        for map in [
            LaneMap::RowMajor { per_row: l, unit },
            LaneMap::ColMajor { rows },
            LaneMap::Split { l, rows, unit },
        ] {
            prop_assert!(affine_delta(map, threads, p).is_some(), "{map:?}");
        }
    }
}

/// `pad`: on register staging each of a warp's `lanes` transposed-A store
/// instructions hits 32 distinct banks; the S52 map (all `bk/lanes` lanes
/// of a row in one warp, no pad) is a `bk/lanes`-way conflict — 4-way at
/// the 4096² geometry the SASS review read.
#[test]
fn sync_a_store_is_conflict_free_under_pad() {
    let worst_way = |s: &TileSite, g: &GpuTile| {
        let [(map, n), _] = lane_maps(s, g, g.kmajor, false);
        let lanes = GPU_VEC_BYTES / 4;
        let mut worst = 0;
        for warp in 0..g.threads() / 32 {
            for c in 0..n {
                for j in 0..lanes {
                    let mut hits = [0; 32];
                    for lane in 0..32 {
                        let li = warp * 32 + lane + c * g.threads();
                        let (r, kc) = map.coords(&mut Num, &li);
                        hits[(((kc + j) * g.sa() + r) % 32) as usize] += 1;
                    }
                    worst = worst.max(*hits.iter().max().unwrap());
                }
            }
        }
        worst
    };
    for p in PARTS {
        for wide_fma in [true, false] {
            // tile-efficiency T2's k-major panel has no transposed store.
            let on = GpuRungs {
                cp_async: false,
                async_stage: false,
                kmajor_a: false,
                wide_fma,
                ..GpuRungs::ALL
            };
            for n in [1024, 2048, 4096, 8192] {
                let s = site(n, n, n, true, true, Ty::f32());
                let g = launch(&s, p, false, force(on), None).tiled.expect("tiled");
                assert_eq!((g.a_row_lanes, g.pad_a), (2, 4), "{} {n}", p.name);
                assert_eq!(worst_way(&s, &g), 1, "{} {n} pad on", p.name);
                // `bk / lanes` lanes per row ⇒ that many ways.
                let off = launch(&s, p, false, force(GpuRungs { pad: false, ..on }), None);
                let g = off.tiled.expect("tiled");
                let ways = g.bk / (GPU_VEC_BYTES / 4);
                assert_eq!(worst_way(&s, &g), ways, "{} {n} S52 map", p.name);
                assert!(ways > 1);
            }
        }
    }
}

/// A tile of the bank reference (`.claude/sprints/per-site-choice/
/// scratch/bank/bank.py`): `(tm, tn, tx, ty, bk, stages)`, warp and
/// interleave rungs, k-major or transposed.
pub(super) fn bank_tile(
    t: [u64; 6],
    warp: bool,
    interleave: bool,
    kmajor: bool,
) -> (GpuTile, GpuRungs) {
    let [tm, tn, tx, ty, bk, stages] = t;
    let g = GpuTile {
        tm,
        tn,
        tx,
        ty,
        bk,
        blocks_per_sm: 1,
        stages,
        pad_a: 0,
        a_row_lanes: 0,
        held: 0,
        kmajor,
    };
    let r = GpuRungs {
        warp,
        interleave,
        ..GpuRungs::ALL
    };
    (g, r)
}

/// per-site-choice D4: a hand-built alias — the output shares a carry's
/// slot with an operand — is not `RERUN`; distinct slots are.
#[test]
fn an_aliased_output_is_not_rerun_safe() {
    let key = |o: u32| if o == 7 { 3 } else { o };
    assert!(apart(&[1, 2], &[5], key));
    assert!(!apart(&[1, 3], &[7], key), "7 shares 3's slot");
    assert!(!apart(&[4], &[4], |o| o));
}

/// gapfix `kvec-swizzle`: the k-major `b` copy's scattered stores on the
/// bank model — every register-staged kvb geometry with `bn` whole bank
/// rows: conflict-free under the swizzle where a chunk spans a vector of
/// rows (`bk ≤ 32`), no worse than half the unswizzled degree past it,
/// and the fragment reads' degree unchanged (a permutation of groups).
#[test]
fn kvec_swizzle_spreads_the_scattered_stores() {
    let s = site(1024, 8192, 128, true, false, Ty::f32());
    let mut seen = 0;
    for (tm, tn, tx, ty, bk) in [
        (4, 4, 32, 1, 32),
        (4, 4, 32, 1, 16),
        (4, 4, 32, 1, 64),
        (4, 4, 16, 2, 32),
        (8, 4, 8, 4, 32),
        (4, 8, 16, 2, 16),
    ] {
        let (g, r) = bank_tile([tm, tn, tx, ty, bk, 1], true, true, false);
        if !kvec_b(&s, &g) {
            continue;
        }
        let off = GpuRungs {
            kvec_swizzle: false,
            ..r
        };
        let (on, base) = (
            panel_excess(&s, &g, &r, false),
            panel_excess(&s, &g, &off, false),
        );
        assert!(kvec_swizzle(&s, &g, &r).is_some(), "{g:?}");
        assert_eq!(on[1], base[1], "reads unchanged {g:?}");
        match bk <= 32 {
            true => assert_eq!(on[3], 1, "conflict-free {g:?} (off {})", base[3]),
            false => assert!(on[3] * 2 <= base[3], "{g:?}: {} vs {}", on[3], base[3]),
        }
        assert!(base[3] > 1, "the unswizzled copy conflicts {g:?}");
        seen += 1;
    }
    assert!(seen >= 4);
}

/// per-site-choice A3: [`bank_degree`] / [`panel_excess`] reproduce the
/// reference calculator's self-test (bank.py `selftest`).
#[test]
fn bank_model_matches_the_reference() {
    let s = site(4096, 4096, 4096, true, true, Ty::f32());
    // S52: register-staged transposed RowMajor copy, bm = 128 unpadded: 4-way.
    let (g, r) = bank_tile([8, 8, 16, 16, 16, 1], true, true, false);
    assert_eq!(panel_excess(&s, &g, &r, false)[2], 4);
    // S53 `pad`: 2 lanes a row, pad 4: conflict-free.
    let p = GpuTile {
        pad_a: 4,
        a_row_lanes: 2,
        ..g
    };
    assert_eq!(panel_excess(&s, &p, &r, false), [1; 4]);
    // async transposed element copy at its `conflict_free_pad` (8).
    assert_eq!(
        crate::profile::conflict_free_pad(128, |l| (l % 4, l / 4)),
        Some(8)
    );
    let a = GpuTile {
        stages: 3,
        pad_a: 8,
        ..g
    };
    assert_eq!(panel_excess(&s, &a, &r, false)[2], 1);
    // One 16-byte read of 4 rows 4 apart at a stride ≡ 0 mod 32 words:
    // 16 words on 4 banks ⇒ degree 4, ideal 1; a 16-byte copy of 32
    // consecutive chunks: 128 words, 4 a bank ⇒ degree 4 = ideal 4.
    assert_eq!(bank_degree((0..32).map(|l| ((l / 8) * 4 * 32, 4))), (4, 1));
    assert_eq!(bank_degree((0..32).map(|l| (4 * l, 4))), (4, 4));
    assert_eq!(bank_degree((0..32).map(|l| (l, 1))), (1, 1));
    assert_eq!(bank_degree((0..32).map(|_| (7, 1))), (1, 1), "a broadcast");
}

/// per-site-choice A1/A2: the 30 distinct k-major geometries the
/// tile-efficiency sweep emitted (bank.md §3): with the shipped row map
/// (`trow·aw + …`, unpadded) the fragment read is 2- or 4-way in 27;
/// with rows `trow + i·ty` and [`kmajor_pad`] every `a` class is
/// conflict-free. Before = [`bank_degree`] on the shipped map's
/// addresses; after = [`panel_excess`] on the emitter's maps.
#[test]
fn kmajor_panel_is_conflict_free_on_the_swept_geometries() {
    #[rustfmt::skip]
        let geoms: [([u64; 6], bool, bool); 30] = [
            ([4, 4, 8, 8, 32, 3], true, true), ([8, 8, 16, 16, 16, 1], false, true),
            ([4, 4, 8, 16, 64, 2], true, true), ([2, 4, 32, 8, 64, 1], false, true),
            ([1, 4, 8, 8, 32, 1], false, true), ([8, 16, 8, 16, 8, 1], false, true),
            ([8, 4, 16, 16, 32, 2], true, true), ([8, 16, 8, 16, 8, 1], true, true),
            ([8, 4, 32, 8, 32, 2], true, true), ([8, 4, 16, 16, 64, 2], true, true),
            ([4, 4, 8, 16, 128, 2], true, true), ([4, 4, 8, 8, 128, 3], true, true),
            ([8, 4, 16, 16, 16, 1], true, true), ([4, 4, 8, 8, 8, 1], true, true),
            ([8, 8, 16, 2, 16, 1], true, true), ([8, 8, 16, 16, 16, 1], true, true),
            ([4, 4, 16, 16, 16, 1], true, true), ([8, 8, 16, 16, 16, 3], true, true),
            ([8, 4, 8, 8, 8, 1], true, true), ([4, 4, 8, 8, 128, 1], true, true),
            ([1, 1, 16, 16, 64, 1], false, false), ([2, 4, 8, 8, 16, 1], true, true),
            ([1, 4, 8, 8, 32, 1], true, true), ([8, 4, 16, 16, 16, 1], false, false),
            ([8, 8, 8, 8, 8, 1], true, true), ([8, 8, 8, 16, 8, 1], false, false),
            ([4, 4, 8, 8, 128, 1], false, true), ([4, 4, 8, 16, 16, 1], false, true),
            ([8, 8, 8, 8, 8, 1], false, true), ([8, 8, 8, 16, 8, 1], false, true),
        ];
    let s = site(4096, 4096, 4096, true, true, Ty::f32());
    let mut table =
        String::from("tm tn tx ty bk stages warp il | before lds st | pad after lds st\n");
    let mut conflicted = 0;
    for (t, warp, il) in geoms {
        let (g, r) = bank_tile(t, warp, il, true);
        // The shipped map: rows `trow·a_step + (i / aw)·a_chunk + i % aw`.
        let f = frag_layout(&s, &g, false, il);
        let threads = g.threads();
        let before = (0..g.tm)
            .flat_map(|i| (0..threads / WARP).map(move |w| (i, w)))
            .map(|(i, w)| {
                let rows = (w * WARP..(w + 1) * WARP).map(|tid| {
                    let trow = thread_coords(&mut Num, &tid, &g, warp, false, WARP).0;
                    let row = trow * f.a_step + (i / f.aw) * f.a_chunk + i % f.aw;
                    (row * g.bk, 4)
                });
                let (d, id) = bank_degree(rows);
                d.div_ceil(id)
            })
            .max()
            .unwrap();
        let [lds0, _, st0, _] = panel_excess(&s, &g, &r, false);
        assert_eq!(st0, 1, "{t:?}: the 16-byte copy is conflict-free unpadded");
        let pad = kmajor_pad(&s, &g, &r);
        let [lds, _, st, _] = panel_excess(&s, &GpuTile { pad_a: pad, ..g }, &r, false);
        assert!(pad.is_multiple_of(4) && pad < 32, "{t:?}: pad {pad}");
        assert_eq!(
            (lds, st),
            (1, 1),
            "{t:?} warp {warp} interleave {il}: pad {pad}"
        );
        conflicted += usize::from(before > 1);
        table += &format!(
            "{t:?} {warp} {il} | {before} {st0} | {pad} {lds} {st} (new map unpadded: {lds0})\n"
        );
    }
    eprintln!("{table}");
    assert!(
        conflicted >= 25,
        "the shipped map conflicts: {conflicted} of 30"
    );
}

/// Arms with `names` forced off, the profile's table otherwise.
pub(super) fn off(names: &[&str]) -> crate::GpuArms {
    let mut off = GpuRungs::default();
    for n in names {
        *off.rung(n).expect("a listed name") = true;
    }
    crate::GpuArms {
        on: GpuRungs::default(),
        off,
        splitk: 0,
        cands: false,
        free: false,
        ..crate::GpuArms::default()
    }
}

/// S62: the M16 projection (16 × 2048 × 7168) was a one-thread-per-cell
/// launch on every part (`gpu_tile` halves its thread grid past 16 rows).
/// Now one output per thread in a 16 × 16 block of the same 128 × 256
/// grid (`cell-fill`, shipped) — and, under `asyncstage` (an arm: its
/// table rows ship it OFF), its panels streamed by `cp.async` through the
/// stages the share left by ≤ 2 blocks per SM holds, each bound pinned.
/// S62 integration: the geometry rungs (`fill-rows`, `row-fit`) give this
/// site a register tile first; `cell-fill` is what it takes without them.
#[test]
fn starved_m16_site_takes_the_cell_tile_and_streams() {
    use crate::profile::RTX_3090;
    let s = site(16, 2048, 7168, true, true, Ty::f32());
    let base = off(&["fill-rows", "row-fit"]);
    for p in [&RTX_5090, &RTX_4090, &H100_PCIE, &RTX_3090] {
        let gpu = p.gpu().unwrap();
        for contract in [false, true] {
            let shipped = launch(&s, p, contract, base, None);
            let g = shipped
                .tiled
                .unwrap_or_else(|| panic!("{} cell tile", p.name));
            assert_eq!(
                (g.tm, g.tn, g.threads(), g.stages),
                (1, 1, 256, 1),
                "{}",
                p.name
            );
            assert_eq!(shipped.gx * shipped.gy, 128);
            let mut arms = base;
            arms.on.async_stage = true;
            let l = launch(&s, p, contract, arms, None);
            let g = l.tiled.unwrap_or_else(|| panic!("{} tiled", p.name));
            assert_eq!(
                (g.tm, g.tn, g.threads(), l.gx * l.gy),
                (1, 1, 256, 128),
                "{}",
                p.name
            );
            assert!(g.stages >= 2, "{} streams", p.name);
            let w = 128u64.div_ceil(gpu.sms);
            assert_eq!(g.blocks_per_sm, w, "{}", p.name);
            let share = (gpu.smem_per_sm / w - gpu.smem_reserved_per_block).min(gpu.smem_per_block);
            let stage = g.stage_elems() * 4;
            assert!(g.stages * stage <= share, "{} fits", p.name);
            assert!(
                g.stages == s.k / g.bk || (g.stages + 1) * stage > share,
                "{} fills",
                p.name
            );
            assert!(g.stages <= s.k / g.bk);
            assert_eq!(l.smem, g.stages * stage);
            let text = kernel_text(
                "m16",
                &s,
                p,
                contract,
                arms,
                None,
                false,
                None,
                Combine::None,
            );
            assert!(
                text.contains("@llvm.nvvm.cp.async.cg.shared.global.16"),
                "{}",
                p.name
            );
            let wait = format!("@llvm.nvvm.cp.async.wait.group(i32 {})", g.stages - 2);
            assert!(text.contains(&wait), "{}", p.name);
            // The fold stays one in-order chain per cell.
            assert_eq!(text.contains("@llvm.fma.f32"), contract);
        }
    }
}

/// Every S62 tile rung off (these two and the geometry rungs) reproduces
/// the launches emitted before S62 (the selection table's literals): one
/// thread per cell at M16, the register-staged 4 × 4 tile at M128 on the
/// RTX 5090.
#[test]
fn the_off_arms_reproduce_todays_launch() {
    // ADR-0050 review: a candidate's kernel id keys its launch text
    // (`launch_key`); the fields C4 / C5 added (`machine_place`, a
    // walk's `whole: 0`) must not enter it, or every candidate kernel
    // and reduce id moves with the rung off (`gpu_changed.sh`,
    // `BUNDLE_CACHE` and wisdom keys see the suite as changed).
    let s = site(2048, 2048, 2048, true, true, Ty::f32());
    for p in [&RTX_4090, &RTX_5090, &H100] {
        let l = launch(&s, p, false, crate::GpuArms::default(), None);
        for combine in [
            Combine::None,
            Combine::Atomic { s: 4 },
            Combine::Workspace { s: 2 },
            Combine::StreamK {
                ctas: 160,
                iters: 9,
                whole: 0,
            },
        ] {
            let mut on = Launch { combine, ..l };
            let off = launch_key(&on);
            assert!(
                !off.contains("machine_place") && !off.contains("whole"),
                "{off}"
            );
            on.rungs.machine_place = true;
            assert_eq!(launch_key(&on), off, "the rung keys no kernel");
        }
        let hybrid = Launch {
            combine: Combine::StreamK {
                ctas: 160,
                iters: 9,
                whole: 7,
            },
            ..l
        };
        assert!(
            launch_key(&hybrid).contains("whole: 7"),
            "a hybrid keys its whole blocks"
        );
    }
    let arms = off(&[
        "cell-fill",
        "asyncstage",
        "row-fit",
        "wave-gate",
        "grid-share",
        "panel-regs",
        "fill-rows",
        "kmajor-a",
        "full-tile",
    ]);
    let s = site(16, 2048, 7168, true, true, Ty::f32());
    let l = launch(&s, &RTX_5090, false, arms, None);
    assert_eq!(
        (l.tiled, l.gx, l.gy, l.block, l.smem),
        (None, 128, 1, 256, 0)
    );
    // hardware-properties: `pad` ships under-filled on both faces (the
    // H100 exact face measured it winning, the 5090's was null), so the
    // exact face's transposed `a` store is padded as the contract one.
    let s = site(128, 2048, 7168, true, true, Ty::f32());
    for (contract, smem) in [(false, 2176), (true, 2176)] {
        let l = launch(&s, &RTX_5090, contract, arms, None);
        let g = l.tiled.expect("tiled");
        assert_eq!(
            (g.stages, l.gx, l.gy, l.block, l.smem),
            (1, 64, 4, 64, smem)
        );
    }
}

/// A grid whose wave fills every block slot its tile was budgeted for
/// keeps the text it had: `asyncstage` derives nothing there.
#[test]
fn filled_grids_are_untouched() {
    use crate::profile::GPU_PARTS;
    for p in GPU_PARTS {
        for (m, n, k) in [(4096, 4096, 4096), (16, 18432, 7168)] {
            let s = site(m, n, k, true, true, Ty::f32());
            for contract in [false, true] {
                let text = |arms| {
                    kernel_text("f", &s, p, contract, arms, None, false, None, Combine::None)
                };
                assert_eq!(
                    text(crate::GpuArms::default()),
                    text(off(&["asyncstage"])),
                    "{} {m}x{n}x{k}",
                    p.name
                );
            }
        }
    }
}

/// One-row sites keep the one-thread-per-cell kernel: a 1 × 256 block
/// copies whole vectors only at `bk ≥ 1024`, which no share holds.
#[test]
fn m1_keeps_the_naive_kernel() {
    use crate::profile::GPU_PARTS;
    let s = site(1, 2048, 7168, true, true, Ty::f32());
    for p in GPU_PARTS {
        for contract in [false, true] {
            let l = launch(&s, p, contract, crate::GpuArms::default(), None);
            assert_eq!(l.tiled, None, "{}", p.name);
        }
    }
}
