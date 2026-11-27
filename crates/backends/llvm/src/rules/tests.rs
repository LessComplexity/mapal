//! Tests of the rung rules: replay of the cited measurements.

use super::*;
use crate::profile::{
    DeviceCost, GPU_PARTS, Home, Machine, MmaFmt, RTX_4090, RTX_5080, RTX_5090, TargetProfile,
    resolve, wave_full,
};
use mapal_ir::Ty;

/// The fill of a square `n`-cell's base FMA grid on `p` (the class the
/// arch-select sweeps keyed a measured size by).
fn wps_square(p: &TargetProfile, n: u64) -> u64 {
    let g = p
        .gpu_tile(n, n, n, &Ty::f32(), false, false)
        .expect("divisible square site");
    let blocks = (n / g.bm()) * (n / g.bn());
    p.gpu()
        .expect("gpu")
        .warps_per_scheduler(blocks, g.threads())
}

/// One measurement the rules pick against (`rung_replay.tsv`).
struct Entry<'a> {
    rung: &'a str,
    part: &'a str,
    winner: &'a str,
    cond: &'a str,
}

/// The measurements whose winner no property the rules read separates —
/// the open "missing fact" items, each named in
/// `local-rtx-5080/hardware-properties.md`. The replay must find exactly
/// these: a rule change that fixes one moves it out, one that breaks a
/// measurement fails here.
const OPEN: [(&str, &str, &str, &str); 22] = [
    // The launch does not read whether its operands are in L2 (flushed
    // L2 wins, warm L2 and the composed MLP lose): `residence_plan`.
    ("asyncstage", "h100-pcie", "on", "l2-flushed"),
    ("asyncstage", "rtx-5090", "on", "l2-flushed"),
    // Site-level: the 64-row convolution's fitted grid wins, the
    // 16-row projection's loses on the same part.
    ("fit-first", "rtx-4090", "on", "conv-site"),
    // At a filled site the fitted grid is kept (the split that follows
    // fills the part): the 144-block grids' tail waves (0.56 / 0.63 of
    // the SM slots) are not refused — the gate needs the split count,
    // which is chosen after the tile (open).
    ("row-fit", "h100-pcie", "off", "blocks=144"),
    ("row-fit", "rtx-4090", "off", "blocks=144"),
    // The fold's strided-read cost vs the tile's column store (1-2 %
    // of device time, one face each).
    ("rowfold-layout", "a100", "on", "-"),
    ("rowfold-layout", "rtx-4090", "on", "-"),
    // Site-level wave quantization: stream-K wins some cells of a
    // class and loses others on the same part.
    ("streamk", "rtx-3090", "off", "-"),
    ("streamk", "rtx-3090", "on", "-"),
    ("streamk", "rtx-4090", "off", "-"),
    ("streamk", "rtx-5090", "on", "-"),
    // The filled tile lowers the split count on the free order: the
    // order axis is not a rule input.
    ("warp-fill", "rtx-3090", "off", "free-order"),
    // Measured against one thread a cell before cell-fill; the gate now
    // yields the 8-block grid to the cell tile, which the row-fit line
    // measured 4x faster: re-measure the gate against the cell tile.
    ("wave-gate", "h100-pcie", "off", "blocks=8"),
    // open-rules: the S5x register-staged tile on the H100 PCIe lost
    // the wide tile above one wave (2048^2, 4096^2); the SXM part's
    // current tile won 4096^3 at 3.88 waves (7a2b67bf). Re-measure the
    // PCIe part on the current tile.
    ("wide-fma", "h100-pcie", "off", "blocks=256,wave=228"),
    ("wide-fma", "h100-pcie", "off", "blocks=1024,wave=228"),
    // cross-arch: `min-blocks` won filled on the H100 SXM and the A100
    // (32 registers a resident thread, where the H100 PCIe's cap spilled
    // 8-16 B and lost) — the derived residency's register cap against
    // the tile's own registers, per site, is the open fact (`rules.md`).
    ("min-blocks", "h100", "on", "-"),
    ("min-blocks", "h100", "on", "-"),
    ("min-blocks", "a100", "on", "-"),
    // regressions `vec-cells`: the H100's 4096 x 14336 projection won
    // the vector grid (224 blocks on 132 SMs, spread 1.18 against the
    // scalar grid's 1.03) by 3.5 %, where 2048 x 7168 and 28672 x 4096
    // at the same grids were null / lost: the spread does not separate
    // it (its 32-trip partitions against 8 and 64).
    ("vec-cells", "h100", "on", "blocks=224,scalar=896"),
    // review-fixes (replay lines for sweep-gemm's and hopper's rungs):
    // `round-store`'s one loss, the 5080's 128 x 28672 x 4096 bf16
    // projection — the stored 2-byte panel fits `bk` 32 x 2 where the
    // f32 panel left 16 x 3 over a 1.33-wave grid streaming its `b` from
    // DRAM: the tail wave's residency x the operand's residence against
    // the stage depth (sweep-gemm §2, `mma-depth`'s split) is the open
    // fact, not the storage.
    ("round-store", "rtx-5080", "off", "blocks=224"),
    // `wgmma-ws` on the H100's 128 x 4096 x 4096: its 256 CTAs at one an
    // SM run 2 waves that each re-read the 64 MB `b` against the 50 MB
    // L2 (S53's 2-3 blocks an SM hold them in one) — the operand's
    // re-read across waves (`raster` answers 1 here) is the open fact
    // (hopper.md open 5; the persistent grid in raster order or R6's
    // split would read `b` once). hopper-launch: still 0.0876 / 0.0941 ms
    // with the launch path fixed; the raster order removes it (the
    // `l2-gate` line below: 0.0822 ms).
    ("wgmma-ws", "h100", "off", "blocks=256"),
    // hopper-launch: the gate keeps row-major where one block's panels
    // fit the L2, but a row-major wave of 132 one-CTA-an-SM tiles reads
    // every `b` panel of the 128-row projections (64-235 MB) and the next
    // wave reads them again; raster wins 5-12 % there and is null on the
    // 2048-row sites. The wave's own panel bytes against the L2 is the
    // open fact (it changes the gate for every tile kernel on every part).
    // gapfix tensor-tile (C11): the reuse-distance gate now groups these
    // sites itself (the `l2-gate h100 on` line); this line keeps the old
    // gate's arms, so it stays open as their record.
    ("l2-gate", "h100", "off", "-"),
    // tensor-cores-2's GeForce bf16x2n line (pipe-bound at issue 0.38,
    // ~90 % of its bf16 pipe) left here: precision `split-rate` weighs
    // the split against the FFMA tile's achieved rate
    // (`Gpu::ffma_tile_flops_per_ns`, the rtx-5080's 32200 flops a ns:
    // 0.61 of its peak), not the FFMA peak — 512 / 2 against 1.6 x 156.
];

/// **hardware-properties: every cited measurement, replayed.** For each
/// line of `rung_replay.tsv` with a winner, the rules over that part's
/// physical properties (and the line's host condition) must pick it at
/// every fill and face the line covers — except the [`OPEN`] ones.
#[test]
fn rung_rules_replay() {
    let mut picked = 0;
    let mut null = 0;
    let mut mixed = 0;
    let mut open = Vec::new();
    // Per rung: (picked, null, mixed, open) — the evidence file's table.
    let mut per: std::collections::BTreeMap<&str, [u32; 4]> = Default::default();
    for line in include_str!("../rung_replay.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        let [rung, part, fill, face, winner, cond, _source] = f[..] else {
            panic!("7 columns: {line}")
        };
        let e = Entry {
            rung,
            part,
            winner,
            cond,
        };
        assert!(crate::GpuRungs::NAMES.contains(&rung), "{line}");
        let p = GPU_PARTS
            .iter()
            .find(|p| p.name == part)
            .unwrap_or_else(|| panic!("{line}: not a replayed part"));
        let mut g = *p.gpu().expect("gpu");
        // A condition the part's profile does not carry: a measured
        // property of the box (`name=value`, a `--cost=` name: the host's
        // pageable rate, a capped clock), or a site's geometry
        // (`row_bytes=`, `repeats=`), comma-separated.
        let mut row_bytes = None;
        let mut repeats = None;
        let mut blocks = None;
        let mut wave = None;
        let mut scalar = None;
        let mut split = None;
        let mut operands = None;
        let mut issue = None;
        let mut read_once = None;
        // hopper-launch `ws-persist`: the site's tiles, stage depth and
        // tile bytes (the CTAs are the part's SMs at one a tile).
        let mut tiles = None;
        let mut stages = None;
        let mut tile_bytes = None;
        // hopper-wgmma `walk-reuse`: a walked bf16 site `MxNxK`, priced
        // as `func::gpu::WALK_ARMS` records it (`mma_walk_replay`).
        let mut walk = None;
        for kv in cond.split(',') {
            let Some((k, v)) = kv.split_once('=') else {
                continue;
            };
            match k {
                "row_bytes" => row_bytes = Some(v.parse::<u64>().expect("row_bytes")),
                "repeats" => repeats = Some(v == "1"),
                "blocks" => blocks = Some(v.parse::<u64>().expect("blocks")),
                "wave" => wave = Some(v.parse::<u64>().expect("wave")),
                "scalar" => scalar = Some(v.parse::<u64>().expect("scalar")),
                "split" => split = Some(v),
                // bugs-properties: an epilogue group's operand bytes
                // (`a` and `b`).
                "operands" => operands = Some(v.parse::<u64>().expect("operands")),
                "issue" => issue = Some(v.parse::<u64>().expect("issue")),
                "tiles" => tiles = Some(v.parse::<u64>().expect("tiles")),
                "stages" => stages = Some(v.parse::<u64>().expect("stages")),
                "tile_bytes" => tile_bytes = Some(v.parse::<u64>().expect("tile_bytes")),
                "walk" => {
                    let d: Vec<u64> = v.split('x').map(|x| x.parse().expect(line)).collect();
                    walk = Some([d[0], d[1], d[2]]);
                }
                // gapfix `stream-ld`: the bytes of an operand the grid
                // reads once.
                "read_once" => read_once = Some(v.parse::<u64>().expect("read_once")),
                // wgmma-16bit: `a`'s home (read below).
                "a" => {}
                _ => {
                    let n: u64 = v.parse().expect(line);
                    let mut over = [None; DeviceCost::NAMES.len()];
                    over[DeviceCost::NAMES
                        .iter()
                        .position(|x| *x == k)
                        .unwrap_or_else(|| panic!("{line}: {k}"))] = Some(n);
                    g = g.with_cost(&over);
                }
            }
        }
        let fills: Vec<u64> = match fill {
            "U0" => vec![0],
            "U1" => vec![1],
            "U" => vec![0, 1],
            "F" => vec![2, 4, 64],
            "A" => vec![0, 1, 2, 4, 64],
            n => vec![wps_square(
                p,
                n.trim_start_matches('n').parse().expect(line),
            )],
        };
        let faces: Vec<Face> = face
            .chars()
            .map(|c| match c {
                'E' => Face::Exact,
                'C' => Face::Contract,
                'T' => Face::Tf32,
                _ => panic!("{line}: face"),
            })
            .collect();
        let count = per.entry(rung).or_default();
        let want = match winner {
            "on" => true,
            "off" => false,
            "null" => {
                null += 1;
                count[1] += 1;
                continue;
            }
            "mixed" => {
                mixed += 1;
                count[2] += 1;
                continue;
            }
            w => panic!("{line}: winner {w}"),
        };
        let all = fills.iter().all(|&wps| {
            faces.iter().all(|&fc| {
                // `fold-lanes` over a column-major operand: the layout
                // guard in `rowfold_lanes` declines whatever the rung.
                let guard = !(e.rung == "fold-lanes" && e.cond == "colmajor");
                let r = rungs(&g, wps, fc);
                let mut v = r.get(e.rung) && guard;
                // The site's side of the priced rules.
                if let (Some(b), "row-split") = (row_bytes, e.rung) {
                    v = v && row_split_pays(&g, b);
                }
                if let (Some(rep), "pdl") = (repeats, e.rung) {
                    v = pdl_pays(&g, rep);
                }
                // f32-tensor-cores `split-tile`: the form's passes
                // against the part's matrix and FFMA rates.
                if let (Some(f), "split-tile") = (split, e.rung) {
                    let (kind, fmt) = match f {
                        "tf32x3" => (mapal_ir::split::SplitKind::Tf32x3, MmaFmt::Tf32),
                        "narrow1" => (mapal_ir::split::SplitKind::Narrow1, MmaFmt::Bf16),
                        "bf16x3n" => (mapal_ir::split::SplitKind::Bf16x3n, MmaFmt::Bf16),
                        "bf16x2n" => (mapal_ir::split::SplitKind::Bf16x2n, MmaFmt::Bf16),
                        _ => panic!("{line}: split form {f}"),
                    };
                    let op = g.caps.matrix_op(fmt, Home::Regs).expect(line);
                    let p = mapal_ir::split::SplitForm::of(kind).products();
                    // tensor-cores-2: the realization's issue ratio
                    // (`issue=`, per mille; none: the rate alone).
                    v = v && split_rate_pays(&g, &op, p, issue.unwrap_or(1000));
                }
                // gapfix `stream-ld`: the read-once operand against the
                // L2's keep.
                // wgmma-16bit `wgmma16`: the part's warpgroup 16-bit
                // product (`a=regs`: `a` from registers, else from a
                // descriptor) against its warp `mma` rate.
                if e.rung == "wgmma16" {
                    let a = match e.cond.contains("a=regs") {
                        true => Home::Regs,
                        false => Home::Smem,
                    };
                    v = v && wgmma16_pays(&g, MmaFmt::Bf16, a).is_some();
                }
                // hopper-wgmma `walk-reuse`: the site's walk price
                // (its reads against the part's L2 keep).
                if let (Some(site), "mma-walk") = (walk, e.rung) {
                    let a = crate::func::gpu::WALK_ARMS
                        .iter()
                        .find(|a| a.part.name == e.part && a.mnk == site)
                        .unwrap_or_else(|| panic!("{line}: no measured walk"));
                    let (price, rate) = a.price(&g);
                    v = v && price.pays(&g, rate);
                }
                if let (Some(b), "stream-ld") = (read_once, e.rung) {
                    v = v && stream_ld_pays(&g, b);
                }
                // bugs-properties `streamk-epi`: the group's operands
                // against the part's L2.
                if let (Some(b), "streamk-epi") = (operands, e.rung) {
                    v = v && streamk_epi_pays(&g, b);
                }
                // The persistent walk's saved fills and epilogues.
                if let (Some(t), Some(s), Some(b), "ws-persist") =
                    (tiles, stages, tile_bytes, e.rung)
                {
                    v = v && ws_persist_pays(&g, t, g.sms, s, b);
                }
                // The wide tile's grid against its wave.
                if let (Some(b), Some(w), "wide-fma") = (blocks, wave, e.rung) {
                    v = v && wide_fma_pays(&g, b, w);
                }
                // The vector cells' grid against the scalar grid's
                // spread at the same split.
                if let (Some(b), Some(sc), "vec-cells") = (blocks, scalar, e.rung) {
                    v = v && vec_cells_pay(&g, b, sc);
                }
                // A row-fitted grid's blocks: where `wave-gate` gates,
                // its fact decides whether the fitted grid runs
                // (`row-fit`) or yields (`wave-gate`).
                if let (Some(b), None) = (blocks, wave) {
                    let full = wave_full(b, g.sms);
                    v = match e.rung {
                        "row-fit" => v && (!r.wave_gate || full),
                        "wave-gate" => v && !full,
                        _ => v,
                    };
                }
                v == want
            })
        });
        match all {
            true => {
                picked += 1;
                count[0] += 1;
            }
            false => {
                open.push((e.rung, e.part, e.winner, e.cond));
                count[3] += 1;
            }
        }
    }
    for (rung, [p, n, m, o]) in &per {
        eprintln!("replay {rung}\t{p}\t{n}\t{m}\t{o}");
    }
    assert_eq!(open, OPEN.to_vec(), "measurements the rules do not pick");
    // The counts the evidence file reports.
    assert_eq!((picked, null, mixed, open.len()), (181, 83, 18, 22));
}

/// precision `split-rate`: the split is weighed against the FFMA tile's
/// achieved rate — the rtx-5080's measured 32200 flops a ns admits
/// bf16x2n (256 against 1.6 × 156 flops an SM a clock) and still refuses
/// tf32x3; the same part without a measured rate (the FFMA peak, the rule
/// before) refuses bf16x2n; a part with none measured (the A100) keeps
/// its peak.
#[test]
fn split_rate_weighs_the_ffma_tiles_achieved_rate() {
    let g = *crate::profile::RTX_5080.gpu().expect("gpu");
    assert_eq!(g.ffma_tile_flops_per_ns(), 32200);
    let bf = g.caps.matrix_op(MmaFmt::Bf16, Home::Regs).expect("bf16");
    let tf = g.caps.matrix_op(MmaFmt::Tf32, Home::Regs).expect("tf32");
    assert!(split_rate_pays(&g, &bf, 2, 375));
    assert!(!split_rate_pays(&g, &tf, 3, 160));
    let peak = Gpu { cost: None, ..g };
    assert_eq!(peak.ffma_tile_flops_per_ns(), peak.ffma_flops_per_ns());
    assert!(!split_rate_pays(&peak, &bf, 2, 375));
    let a = *crate::profile::A100.gpu().expect("gpu");
    assert_eq!(a.ffma_tile_flops_per_ns(), a.ffma_flops_per_ns());
}

/// hopper-launch `ws-persist`, hand-worked on the H100 SXM: the 2048 ×
/// 4096 × 4096 site's 256 tiles of 256 × 128 on 132 CTAs walk two rounds
/// — one later tile a CTA, whose fill (3 copies at `tma_copy_ns` 159) and
/// epilogue (131 072 B at the SM's DRAM share) the walk overlaps; 132
/// tiles are one round (nothing saved, a CTA a tile); a part with twice
/// the SMs takes the 256 in one round too.
#[test]
fn ws_persist_prices_the_walk() {
    let h = *crate::profile::H100.gpu().expect("gpu");
    let tile = 256 * 128 * 4;
    let fill = 3 * h.tma_copy_ns();
    assert_eq!(fill, 477);
    let epi = tile * 1000 / h.dram_bytes_per_us_per_sm();
    assert_eq!(ws_persist_saves_ns(256, h.sms, fill, epi), fill + epi);
    assert_eq!(
        ws_persist_saves_ns(1792, h.sms, fill, epi),
        13 * (fill + epi)
    );
    assert_eq!(ws_persist_saves_ns(132, h.sms, fill, epi), 0);
    assert!(ws_persist_pays(&h, 256, h.sms, 4, tile));
    assert!(!ws_persist_pays(&h, 132, h.sms, 4, tile));
    let wide = crate::profile::Gpu {
        sms: 2 * h.sms,
        ..h
    };
    assert!(!ws_persist_pays(&wide, 256, wide.sms, 4, tile));
}

/// No rule reads a name, an architecture string or a version: a part
/// renamed and re-targeted takes the same rungs, and the hand profiles of
/// one SM with different properties are decided by those properties.
#[test]
fn rungs_read_properties_never_names() {
    for p in GPU_PARTS {
        let g = *p.gpu().expect("gpu");
        let mut other = *p;
        other.name = "renamed";
        other.machine = Machine::Gpu(crate::profile::Gpu {
            arch: "sm_999",
            ..g
        });
        for wps in [0, 1, 2, 4, 64] {
            for face in [Face::Exact, Face::Contract, Face::Tf32] {
                assert_eq!(p.gpu_rungs_on(wps, face), other.gpu_rungs_on(wps, face));
            }
        }
    }
    // The 5080 and 5090 share an SM; the rules see their properties.
    let b = RTX_5090.gpu().expect("gpu");
    let f = RTX_5080.gpu().expect("gpu");
    assert!(b.dram_bytes_per_us_per_sm() >= DRAM_PER_SM_LOW);
    assert!(f.dram_bytes_per_us_per_sm() >= DRAM_PER_SM_LOW);
    assert!(RTX_4090.gpu().unwrap().dram_bytes_per_us_per_sm() < DRAM_PER_SM_LOW);
    // cross-arch: the 4090 box capped to a sustained 1.96 GHz takes the
    // same rungs as the part at its 2.52 GHz rating — no rule reads the
    // clock alone.
    let capped = crate::profile::Gpu {
        clock_khz: 1_955_131,
        ..*RTX_4090.gpu().unwrap()
    };
    for wps in [0, 1, 2, 4, 64] {
        for face in [Face::Exact, Face::Contract, Face::Tf32] {
            let (a, c) = (
                rungs(RTX_4090.gpu().unwrap(), wps, face),
                rungs(&capped, wps, face),
            );
            assert_eq!(
                (a.wide_fma, a.cp_async, a.kmajor_a),
                (c.wide_fma, c.cp_async, c.kmajor_a)
            );
        }
    }
    // pdl reads the measured launches: the 5080's programmatic launch
    // (486 ns) is cheaper than its plain one (650 ns); the 5090 has
    // none measured.
    assert!(RTX_5080.gpu_rungs(0, true).pdl);
    assert!(!RTX_5090.gpu_rungs(0, true).pdl);
    // `pinned` reads the host's measured pageable rate.
    assert!(RTX_5080.gpu_rungs(0, true).pinned);
    assert!(!RTX_5090.gpu_rungs(0, true).pinned);
    // Every rung the rules keep off on every part is off on any part.
    let generic = resolve("cuda-ada").expect("profile");
    for n in [
        "async-h2d",
        "asyncstage",
        "l2-window",
        "retain-delta",
        "streams",
    ] {
        assert!(!generic.gpu_rungs(2, true).get(n), "{n}");
    }
}
