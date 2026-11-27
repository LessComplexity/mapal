//! Schedules and launches: [`schedule_of`] and [`launch_of`], the rules' launch both ways.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

// ---- schedules and launches ---------------------------------------------------

/// The one-thread-per-cell kernel's registers at up to eight cells a
/// thread where the full-occupancy budget allows them (the `ptxas -v`
/// reports' largest, sm_80 – sm_120).
pub const NAIVE_REGS: u64 = 40;

/// The lowering ids.
pub const NAIVE: &str = "naive";
pub const FMA: &str = "fma";
pub const MMA: &str = "mma";
pub const WGMMA: &str = "wgmma";
pub const WGMMA_WS: &str = "wgmma-ws";

/// The matrix operation row of format `fmt` issued by a warp or a warpgroup
/// (`group`), with `a` from `a` where the part lists that home, else the
/// row of that format (the schedule's level then names `a`'s home: the S53
/// TF32 kernel's `bᵀ` fragments from registers).
pub(super) fn op_index(g: &Gpu, fmt: MmaFmt, a: MHome, group: bool) -> Option<usize> {
    let rows = || {
        g.caps
            .matrix
            .iter()
            .enumerate()
            .filter(move |(_, o)| o.fmt == fmt && (o.shape.threads > 32) == group)
    };
    rows()
        .find(|(_, o)| o.a == a)
        .or_else(|| rows().next())
        .map(|(i, _)| i + 2)
}

/// The split forms a `{f32}` product may take on a matrix instruction, in
/// one order (the schedule's `split` parameter: 0 none, else 1 + the index).
pub(super) const SPLITS: [mapal_ir::split::SplitKind; 5] = {
    use mapal_ir::split::SplitKind as K;
    [K::Tf32x3, K::Bf16x6, K::Bf16x3n, K::Bf16x2n, K::Narrow1]
};

pub(super) fn split_code(s: Option<mapal_ir::split::SplitKind>) -> u64 {
    match s {
        None => 0,
        Some(k) => 1 + SPLITS.iter().position(|x| *x == k).unwrap_or(0) as u64,
    }
}

pub(super) fn split_of(c: u64) -> Option<mapal_ir::split::SplitKind> {
    match c {
        0 => None,
        c => Some(SPLITS[(c - 1) as usize]),
    }
}

/// The traversal a launch's combine and raster realize.
pub(super) fn trav_of(l: &Launch) -> Traversal {
    let h = l.raster.max(1);
    if let Some(w) = l.wgmma.filter(|w| w.ctas > 0) {
        return Traversal::Persistent { h, ctas: w.ctas };
    }
    match l.combine {
        Combine::None => Traversal::Ordered { h },
        Combine::Atomic { s } => Traversal::Split {
            s,
            comb: Comb::Atomic,
            h,
        },
        Combine::Workspace { s } => Traversal::Split {
            s,
            comb: Comb::Workspace,
            h,
        },
        Combine::StreamK {
            ctas,
            iters,
            whole: 0,
        } => Traversal::Walk { h, ctas, iters },
        Combine::StreamK { ctas, iters, .. } => Traversal::Hybrid { h, ctas, iters },
    }
}

/// The combine traversal `t` lowers to; a hybrid's whole waves are the
/// `tiles` of the grid over `slots` (the footprint's).
pub(super) fn combine_of(t: Traversal, tiles: u64, slots: u64) -> Combine {
    match t {
        Traversal::Ordered { .. } | Traversal::Persistent { .. } => Combine::None,
        Traversal::Hybrid { ctas, iters, .. } => Combine::StreamK {
            ctas,
            iters,
            whole: tiles / slots.max(1) * slots.max(1),
        },
        Traversal::Split {
            s,
            comb: Comb::Atomic,
            ..
        } => Combine::Atomic { s },
        Traversal::Split { s, .. } => Combine::Workspace { s },
        Traversal::Walk { ctas, iters, .. } => Combine::StreamK {
            ctas,
            iters,
            whole: 0,
        },
    }
}

pub(super) fn smem_home(depth: u64, major: Major, pad: u64, swizzle: u64) -> Option<Home> {
    Some(Home {
        place: SMEM,
        layout: Layout {
            major,
            pad,
            swizzle,
        },
        depth,
    })
}

/// The face a launch's arithmetic is emitted on.
pub(super) fn face_of(a: Arithmetic) -> Face {
    a.face()
}

/// The site's table rungs at the launch's face (the D6 steps the profile's
/// rules decide for the site's fill).
pub(crate) fn table_rungs(
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    a: Arithmetic,
) -> crate::GpuRungs {
    let gpu = profile.gpu().expect("a GPU profile");
    let (blocks, threads) = fill_grid(site, profile);
    let wps = gpu.warps_per_scheduler(blocks, threads);
    site_rungs(profile, wps, face_of(a), arms, site)
}

/// The rungs `r` carries beyond `base`, as schedule parameters.
pub(super) fn rung_params(base: &crate::GpuRungs, r: &crate::GpuRungs) -> Vec<(&'static str, u64)> {
    let (mut b, mut x) = (*base, *r);
    let mut out = Vec::new();
    for (i, name) in crate::GpuRungs::NAMES.iter().enumerate() {
        let (vb, vx) = (
            *b.rung(name).expect("a rung"),
            *x.rung(name).expect("a rung"),
        );
        if vb != vx {
            out.push((if vx { "rung+" } else { "rung-" }, i as u64));
        }
    }
    out
}

/// **The rules' launch as a schedule**: the lowering, the operation and its
/// roles, the levels (a lane's or group's tile, the slot's tile and panel
/// with each operand's home, layout and depth), the slot's composition, the
/// traversal, the occupancy, and the lowering's encoding parameters.
pub fn schedule_of(
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    l: &Launch,
) -> Schedule {
    let gpu = profile.gpu().expect("a GPU profile");
    let base = table_rungs(site, profile, arms, l.arithmetic);
    let mut params = rung_params(&base, &l.rungs);
    let slot = |ext: [u64; 3], homes: [Option<Home>; 2]| Level {
        ext,
        bind: Bind::Slot,
        homes,
    };
    let grid = Level {
        ext: [site.rows, site.c, site.k],
        bind: Bind::Exec(E_DEVICE),
        homes: [None, None],
    };
    let base_sched = |lowering, op, swap, levels, groups: [u64; 2], per_unit| Schedule {
        lowering,
        op,
        swap,
        levels,
        groups,
        producers: 0,
        regs: [0, 0],
        cluster: [1, 1],
        multicast: [false, false],
        trav: trav_of(l),
        launch: LaunchMode::Plain,
        policy: [Policy::Default; 2],
        per_unit,
        params: Vec::new(),
    };
    let scalar_op = match l.arithmetic {
        Arithmetic::SeparateMulAdd => OP_MULADD,
        _ => OP_FFMA,
    };
    let mut x = if let Some(w) = l.wgmma {
        let op = op_index(gpu, w.fmt, w.a_home, true).unwrap_or(usize::MAX);
        params.extend([
            ("ea", w.ea),
            ("eb", w.eb),
            ("pad_b", w.pad_b),
            ("b_kmajor", u64::from(w.b_kmajor)),
            ("n", w.n),
        ]);
        let mut x = base_sched(
            if w.ws { WGMMA_WS } else { WGMMA },
            op,
            !w.direct,
            vec![
                Level {
                    ext: [64, w.n, w.mma_k()],
                    bind: Bind::Exec(E_GROUP),
                    homes: [
                        Some(Home {
                            place: match w.a_home {
                                MHome::Regs => REGS,
                                MHome::Smem => SMEM,
                                MHome::Tmem => TMEM,
                            },
                            layout: Layout::K,
                            depth: 1,
                        }),
                        None,
                    ],
                },
                slot(
                    [w.bm(), w.bn(), w.bk],
                    [
                        smem_home(w.stages, Major::K, 0, 128),
                        smem_home(
                            w.stages,
                            if w.b_kmajor { Major::K } else { Major::Mn },
                            w.pad_b,
                            if w.ws { 128 } else { 0 },
                        ),
                    ],
                ),
                grid,
            ],
            [w.groups, 1],
            w.blocks_per_sm,
        );
        x.producers = u64::from(w.ws);
        x.regs = w.realloc.map_or([0, 0], |(p, c)| [p, c]);
        x
    } else if let Some(m) = l.mma {
        let op = op_index(gpu, m.fmt, MHome::Regs, false).unwrap_or(usize::MAX);
        params.extend([
            ("ea", m.ea),
            ("eb", m.eb),
            ("split", split_code(m.split)),
            ("mt", m.mt),
            ("nt", m.nt),
        ]);
        base_sched(
            MMA,
            op,
            false,
            vec![
                Level {
                    ext: [m.wm(), m.wn(), m.mma_k()],
                    bind: Bind::Exec(E_GROUP),
                    homes: [None, None],
                },
                slot(
                    [m.bm(), m.bn(), m.bk],
                    [
                        smem_home(m.stages, Major::K, m.pad_a, 0),
                        smem_home(
                            m.stages,
                            if m.bt { Major::K } else { Major::Mn },
                            m.pad_b,
                            0,
                        ),
                    ],
                ),
                grid,
            ],
            [m.warps_y, m.warps_x],
            m.blocks_per_sm,
        )
    } else if let Some(t) = l.tiled {
        params.extend([("a_row_lanes", t.a_row_lanes), ("held", t.held)]);
        base_sched(
            FMA,
            scalar_op,
            false,
            vec![
                Level {
                    ext: [t.tm, t.tn, 1],
                    bind: Bind::Exec(E_LANE),
                    homes: [None, None],
                },
                Level {
                    ext: [t.ty, t.tx, 1],
                    bind: Bind::Exec(E_GROUP),
                    homes: [None, None],
                },
                slot(
                    [t.bm(), t.bn(), t.bk],
                    [
                        smem_home(
                            t.stages,
                            if t.kmajor { Major::K } else { Major::Mn },
                            t.pad_a,
                            0,
                        ),
                        smem_home(t.stages, Major::Mn, 0, 0),
                    ],
                ),
                grid,
            ],
            [t.threads().div_ceil(gpu.warp), 1],
            t.blocks_per_sm,
        )
    } else {
        params.push(("cells", l.cells));
        let per = (gpu.max_threads_per_sm / l.block.max(1)).min(gpu.max_blocks_per_sm);
        let bn = (NAIVE_THREADS * l.cells).min(site.c);
        let bm = ((NAIVE_THREADS * l.cells) / site.c.max(1)).clamp(1, site.rows.max(1));
        base_sched(
            NAIVE,
            scalar_op,
            false,
            vec![
                Level {
                    ext: [1, l.cells, 1],
                    bind: Bind::Exec(E_LANE),
                    homes: [None, None],
                },
                slot([bm, bn, NAIVE_UNROLL], [None, None]),
                grid,
            ],
            [NAIVE_THREADS / gpu.warp.max(1), 1],
            per,
        )
    };
    if l.arithmetic == Arithmetic::SplitMma {
        params.push(("split_arith", 1));
    }
    x.params = params;
    x
}

pub(super) fn param(x: &Schedule, name: &str) -> u64 {
    x.param(name).unwrap_or(0)
}

pub(super) fn major(h: &Option<Home>) -> Major {
    h.map_or(Major::K, |h| h.layout.major)
}

/// **The launch a schedule lowers to**: the kernel geometry its lowering
/// emits, the grid, block and shared memory its footprint declares, the
/// raster and combine of its traversal, and the site's table rungs with the
/// schedule's own D6 steps applied.
pub fn launch_of(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    x: &Schedule,
) -> Launch {
    let gpu = profile.gpu().expect("a GPU profile");
    let e = elem_bytes_of(&site.elem);
    let s = x.slot();
    let [ha, hb] = s.homes;
    let depth = x.depth();
    let mut tiled = None;
    let mut mma = None;
    let mut wgmma = None;
    let (gx, gy, block, smem, cells);
    let arithmetic = match x.lowering {
        NAIVE | FMA => match x.op {
            OP_MULADD => Arithmetic::SeparateMulAdd,
            _ => Arithmetic::OrderedFma,
        },
        _ => {
            let o = &gpu.caps.matrix[x.op - 2];
            match (o.shape.threads > 32, o.fmt) {
                (true, MmaFmt::Tf32) => Arithmetic::Tf32Wgmma,
                (true, _) => Arithmetic::HalfWgmma,
                (false, _) if param(x, "split_arith") == 1 => Arithmetic::SplitMma,
                (false, MmaFmt::Tf32) => Arithmetic::Tf32Mma,
                (false, _) => Arithmetic::HalfMma,
            }
        }
    };
    let _ = contract;
    let mut rungs = memo_rungs(site, profile, arms, arithmetic);
    for (n, v) in &x.params {
        let on = match *n {
            "rung+" => true,
            "rung-" => false,
            _ => continue,
        };
        *rungs
            .rung(crate::GpuRungs::NAMES[*v as usize])
            .expect("a rung") = on;
    }
    match x.lowering {
        NAIVE => {
            cells = param(x, "cells").max(1);
            gx = cell_blocks(site, cells);
            gy = 1;
            block = NAIVE_THREADS;
            smem = 0;
        }
        FMA => {
            let [tm, tn, _] = x.levels[0].ext;
            let [ty, tx, _] = x.levels[1].ext;
            let t = GpuTile {
                tm,
                tn,
                tx,
                ty,
                bk: x.bk(),
                blocks_per_sm: x.per_unit,
                stages: depth,
                pad_a: ha.map_or(0, |h| h.layout.pad),
                a_row_lanes: param(x, "a_row_lanes"),
                held: param(x, "held"),
                kmajor: major(&ha) == Major::K,
            };
            cells = 1;
            gx = site.c / t.bn();
            gy = site.rows / t.bm();
            block = t.threads();
            smem = t.smem_bytes(e);
            tiled = Some(t);
        }
        MMA => {
            let o = &gpu.caps.matrix[x.op - 2];
            let m = GpuMmaTile {
                mt: param(x, "mt"),
                nt: param(x, "nt"),
                warps_x: x.groups[1],
                warps_y: x.groups[0],
                bk: x.bk(),
                stages: depth,
                pad_a: ha.map_or(0, |h| h.layout.pad),
                pad_b: hb.map_or(0, |h| h.layout.pad),
                blocks_per_sm: x.per_unit,
                fmt: o.fmt,
                ea: param(x, "ea"),
                eb: param(x, "eb"),
                bt: major(&hb) == Major::K,
                split: split_of(param(x, "split")),
            };
            cells = 1;
            gx = site.c / m.bn();
            gy = site.rows / m.bm();
            block = m.threads(gpu.warp);
            smem = m.smem_bytes()
                + match gpu::bulk_stage(site, &m, gpu, &rungs) {
                    true => gpu::BULK_BAR_BYTES * m.stages,
                    false => 0,
                };
            mma = Some(m);
        }
        _ => {
            let o = &gpu.caps.matrix[x.op - 2];
            let ctas = match x.trav {
                Traversal::Persistent { ctas, .. } => ctas,
                _ => 0,
            };
            let w = GpuWgmmaTile {
                n: param(x, "n"),
                groups: x.groups[0],
                bk: x.bk(),
                stages: depth,
                pad_b: param(x, "pad_b"),
                blocks_per_sm: x.per_unit,
                ws: x.lowering == WGMMA_WS,
                realloc: (x.regs != [0, 0]).then_some((x.regs[0], x.regs[1])),
                ctas,
                fmt: o.fmt,
                direct: !x.swap,
                ea: param(x, "ea"),
                eb: param(x, "eb"),
                a_home: match x.levels[0].homes[0].map(|h| h.place) {
                    Some(REGS) => MHome::Regs,
                    Some(TMEM) => MHome::Tmem,
                    Some(SMEM) => MHome::Smem,
                    _ => o.a,
                },
                b_kmajor: param(x, "b_kmajor") == 1,
            };
            cells = 1;
            (gx, gy) = match ctas {
                0 => (site.c / w.bn(), site.rows / w.bm()),
                c => (c, 1),
            };
            block = w.threads(gpu.warp);
            smem = w.smem_bytes();
            wgmma = Some(w);
        }
    }
    Launch {
        arithmetic,
        tiled,
        mma,
        wgmma,
        gx,
        gy,
        block,
        smem,
        raster: x.trav.raster().max(1),
        rungs,
        combine: combine_of(
            x.trav,
            (site.rows / x.bm().max(1)) * (site.c / x.bn().max(1)),
            gpu.sms * x.per_unit.max(1),
        ),
        cells,
    }
}
