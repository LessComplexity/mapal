//! [`Emitter`]: a present emitter as a lowering over one site (its declarations and search domains).
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

// ---- the lowerings' declarations ----------------------------------------------

/// A present emitter as a [`Lowering`], over one site of one profile.
pub struct Emitter<'a> {
    pub id: &'static str,
    pub site: &'a TileSite,
    pub profile: &'a TargetProfile,
    pub contract: bool,
    pub arms: crate::GpuArms,
}

impl Emitter<'_> {
    pub(super) fn gpu(&self) -> &Gpu {
        self.profile.gpu().expect("a GPU profile")
    }

    /// The emitters of a site (the search's lowerings, ADR-0050 C4).
    #[allow(dead_code)]
    pub fn all<'a>(
        site: &'a TileSite,
        profile: &'a TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
    ) -> Vec<Emitter<'a>> {
        [NAIVE, FMA, MMA, WGMMA, WGMMA_WS]
            .into_iter()
            .map(|id| Emitter {
                id,
                site,
                profile,
                contract,
                arms,
            })
            .collect()
    }

    /// **The coordinates of the lowering's slot space** (D5 steps 2–3), in
    /// order: each coordinate's domain given the ones before it — from the
    /// machine's facts (register file, shared memory, threads, the
    /// operation's shape and homes, the site's extents), never a table. The
    /// enumeration ([`Lowering::slots`]) and the membership check
    /// ([`Self::contains`]) walk the same domains.
    pub fn coords(&self) -> &'static [&'static str] {
        match self.id {
            NAIVE => &["op", "cells"],
            FMA => &[
                "op",
                "tm",
                "tn",
                "tx",
                "ty",
                "bk",
                "stages",
                "kmajor",
                "pad_a",
                "a_row_lanes",
                "held",
            ],
            MMA => &[
                "op", "mt", "nt", "warps_y", "warps_x", "bk", "stages", "bt", "pad_a", "pad_b",
                "split", "ea", "eb",
            ],
            _ => &[
                "op", "n", "groups", "bk", "stages", "b_kmajor", "pad_b", "swap", "ea", "eb",
                "regs_p", "regs_c",
            ],
        }
    }

    /// The domain of coordinate `i` given `pre` (the coordinates before it).
    pub fn domain(&self, i: usize, pre: &[u64]) -> Vec<u64> {
        let g = self.gpu();
        let s = self.site;
        let e = elem_bytes_of(&s.elem);
        let pow2 = |hi: u64| -> Vec<u64> {
            (0..20)
                .map(|p| 1u64 << p)
                .take_while(|&v| v <= hi)
                .collect()
        };
        let ops = |group: Option<bool>| -> Vec<u64> {
            let mut v = Vec::new();
            match group {
                None => v.extend([OP_FFMA as u64, OP_MULADD as u64]),
                Some(grp) => {
                    for (j, o) in g.caps.matrix.iter().enumerate() {
                        let fits = (o.shape.threads > 32) == grp && o.shape.threads > 1;
                        let admitted = match o.fmt {
                            MmaFmt::Tf32 => s.tf32 || !grp,
                            f => s.half.is_some_and(|h| MmaFmt::of_narrow(h) == f) || !grp,
                        };
                        if fits && admitted {
                            v.push(j as u64 + 2);
                        }
                    }
                }
            }
            v
        };
        let stages =
            |bk: u64, hi: u64| -> Vec<u64> { (1..=hi.min(s.k / bk.max(1)).max(1)).collect() };
        match (self.id, self.coords()[i]) {
            (NAIVE, "op") | (FMA, "op") => ops(None),
            (MMA, "op") => ops(Some(false)),
            (_, "op") => ops(Some(true)),
            (NAIVE, "cells") => pow2(GPU_VEC_BYTES / e.min(GPU_VEC_BYTES).max(1) * 4),
            // A lane's accumulators: whose held registers fit the lane's
            // file; a row group's rows (a radix head group) too.
            (FMA, "tm") | (FMA, "tn") => {
                let mut v: Vec<u64> = (1..=32)
                    .filter(|&t| fma_held_regs(t, 1, e) <= g.regs_per_thread)
                    .collect();
                v.retain(|t| t.is_power_of_two() || gpu::row_group(s).is_multiple_of(*t));
                v
            }
            (FMA, "tx") | (FMA, "ty") => pow2(g.max_threads_per_block),
            (FMA, "bk") => (1..=s.k).filter(|b| s.k.is_multiple_of(*b)).collect(),
            // A ring of up to 8 panels (more stages than panels: a slot's
            // double buffer); the shared memory it takes is declared.
            (FMA, "stages") => (1..=8).collect(),
            (_, "kmajor") | (_, "bt") | (_, "b_kmajor") | (_, "swap") => vec![0, 1],
            (FMA, "pad_a") | (MMA, "pad_a") | (MMA, "pad_b") | (_, "pad_b") => (0..=32).collect(),
            (FMA, "a_row_lanes") => {
                let mut v = vec![0];
                v.extend(pow2(pre[5].max(1) * 32));
                v
            }
            // The registers a whole-file tile holds: at least its counted
            // footprint, at most the lane's file (0: the half-file rule).
            (FMA, "held") => {
                let mut v = vec![0];
                v.extend(fma_held_regs(pre[1], pre[2], e)..=g.regs_per_thread);
                v
            }
            // The warp tile in instruction shapes, warps a slot.
            (MMA, "mt") | (MMA, "nt") => pow2(16),
            (MMA, "warps_y") | (MMA, "warps_x") => pow2(g.max_threads_per_block / g.warp),
            (MMA, "bk") | (_, "bk") => {
                let o = g.caps.matrix.get((pre[0] as usize).wrapping_sub(2));
                let mk = o.map_or(8, |o| o.shape.k);
                (1..=s.k / mk.max(1))
                    .map(|t| t * mk)
                    .filter(|b| s.k.is_multiple_of(*b))
                    .collect()
            }
            (MMA, "stages") => stages(pre[5], 16),
            (MMA, "split") => (0..=SPLITS.len() as u64).collect(),
            (_, "ea") | (_, "eb") => vec![1, 2, 4],
            (_, "n") => (1..=WGMMA_N_MAX / WGMMA_N_STEP)
                .map(|t| t * WGMMA_N_STEP)
                .collect(),
            (_, "groups") => pow2(g.max_threads_per_block / (WARPGROUP_WARPS * g.warp)),
            (_, "stages") => stages(pre[3], 16),
            (_, "regs_p") | (_, "regs_c") => {
                let mut v = vec![0];
                if let Some(r) = g.caps.reg_realloc {
                    v.extend((r.min..=r.max).step_by(r.step as usize));
                }
                v
            }
            _ => vec![],
        }
    }

    /// A schedule's coordinates in [`Self::coords`] order.
    pub fn coords_of(&self, x: &Schedule) -> Vec<u64> {
        let s = x.slot();
        let [ha, hb] = s.homes;
        let pad = |h: Option<Home>| h.map_or(0, |h| h.layout.pad);
        let maj = |h: Option<Home>| u64::from(h.is_some_and(|h| h.layout.major == Major::K));
        match self.id {
            NAIVE => vec![x.op as u64, param(x, "cells")],
            FMA => vec![
                x.op as u64,
                x.levels[0].ext[0],
                x.levels[0].ext[1],
                x.levels[1].ext[1],
                x.levels[1].ext[0],
                x.bk(),
                x.depth(),
                maj(ha),
                pad(ha),
                param(x, "a_row_lanes"),
                param(x, "held"),
            ],
            MMA => vec![
                x.op as u64,
                param(x, "mt"),
                param(x, "nt"),
                x.groups[0],
                x.groups[1],
                x.bk(),
                x.depth(),
                maj(hb),
                pad(ha),
                pad(hb),
                param(x, "split"),
                param(x, "ea"),
                param(x, "eb"),
            ],
            _ => vec![
                x.op as u64,
                param(x, "n"),
                x.groups[0],
                x.bk(),
                x.depth(),
                param(x, "b_kmajor"),
                param(x, "pad_b"),
                u64::from(x.swap),
                param(x, "ea"),
                param(x, "eb"),
                x.regs[0],
                x.regs[1],
            ],
        }
    }

    /// **Membership**: `x` is a point of this lowering's slot space — every
    /// coordinate inside its domain, the declared constraints met, the
    /// occupancy at most the footprint's. `Err`: the first coordinate or
    /// constraint outside.
    pub fn contains(&self, m: &Machine, x: &Schedule) -> Result<Declared, String> {
        if x.lowering != self.id {
            return Err(format!("lowering {} not {}", x.lowering, self.id));
        }
        let c = self.coords_of(x);
        for i in 0..c.len() {
            let dom = self.domain(i, &c[..i]);
            if !dom.contains(&c[i]) {
                return Err(format!(
                    "{}={} outside its domain ({} values{})",
                    self.coords()[i],
                    c[i],
                    dom.len(),
                    if dom.len() <= 8 {
                        format!(": {dom:?}")
                    } else {
                        String::new()
                    }
                ));
            }
        }
        let s = site_of(self.site, self.contract, crate::Order::Free);
        let d = self.declare(m, &s, x).map_err(|r| r.0)?;
        let bytes = [(SMEM, d.smem)];
        let occ = m.occupancy(d.threads, d.regs, &bytes);
        // The occupancy is the lowering's only where it emits it
        // (`.minnctapersm`, `min-blocks`); elsewhere `per_unit` describes
        // what the rules assumed, and the evaluator takes the footprint's.
        let back = launch_of(self.site, self.profile, self.contract, self.arms, x);
        // Only a tile kernel emits it (the cell kernel never does).
        let emitted = back.rungs.min_blocks
            && (back.tiled.is_some() || back.mma.is_some() || back.wgmma.is_some());
        if emitted && x.per_unit > occ.max(1) && x.per_unit > 1 {
            return Err(format!(
                "per_unit {} above the footprint's occupancy {occ} ({} lanes x {} regs, {} B)",
                x.per_unit, d.threads, d.regs, d.smem
            ));
        }
        Ok(d)
    }
}

impl Emitter<'_> {
    /// **The search's domain of coordinate `i`** given `pre` (ADR-0050 C4):
    /// [`Self::domain`] narrowed by the site's facts — an operand's stored
    /// layout (`bt`, `b_kmajor`) and staged widths (`ea`, `eb`) are the
    /// site's, tiles divide its extents, a slot's lanes fit a block, a
    /// lane's accumulators its registers — and by dominance: a layout's pads
    /// and lanes are the emitter's own conflict-free choice by the bank
    /// model (no evaluator term reads them), a whole-file budget (`held`)
    /// and a register reallocation (`regs_p`, `regs_c`) the counted
    /// footprint, the split forms the rules' (an `{f32}` product's split
    /// realization stays the incumbent's: open, ADR-0050 C6).
    pub(super) fn narrowed(&self, i: usize, pre: &[u64]) -> Vec<u64> {
        let g = self.gpu();
        let site = self.site;
        let e = elem_bytes_of(&site.elem);
        let threads_max = g.max_threads_per_block;
        let rows = gpu::row_group(site);
        let mut dom = self.domain(i, pre);
        let name = self.coords()[i];
        let divides = |n: u64, d: u64| d > 0 && n.is_multiple_of(d);
        match (self.id, name) {
            (FMA, "tn") => dom.retain(|&tn| fma_held_regs(pre[1], tn, e) <= g.regs_per_thread),
            (FMA, "tx") => dom.retain(|&tx| divides(site.c, tx * pre[2])),
            // A slot holds whole groups (the dispatch issues a group's lanes
            // together; a partial group's lanes idle).
            (FMA, "ty") => dom.retain(|&ty| {
                ty * pre[3] <= threads_max
                    && (ty * pre[3]).is_multiple_of(g.warp)
                    && divides(rows, ty * pre[1])
            }),
            // Every thread copies an equal share of each panel.
            (FMA, "bk") => {
                let (bm, bn, threads) = (pre[4] * pre[1], pre[3] * pre[2], pre[3] * pre[4]);
                dom.retain(|&bk| {
                    (bm * bk).is_multiple_of(threads) && (bk * bn).is_multiple_of(threads)
                });
            }
            // A streamed (`cp.async`) panel where the emitter streams it,
            // and the stages the slot's shared memory holds.
            (FMA, "stages") => {
                let r = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
                let streams = gpu::fma_streams(site, &r);
                dom.retain(|&st| {
                    let t = GpuTile {
                        stages: st,
                        ..self.fma_tile(pre)
                    };
                    st <= (site.k / pre[5].max(1)).max(1)
                        && (st == 1 || streams)
                        && t.smem_bytes(e) <= g.smem_per_block
                })
            }
            (FMA, "kmajor") => {
                if !(site.a.ck == 1 && pre[3] >= gpu::GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES) {
                    dom = vec![0];
                }
            }
            (FMA, "pad_a") | (FMA, "a_row_lanes") => {
                // The layout of the prefix's tile (one derivation for both
                // coordinates and every stage count past the first).
                let t = self.fma_tile(pre);
                let key = [
                    t.tm,
                    t.tn,
                    t.tx,
                    t.ty,
                    t.bk,
                    t.stages.min(2),
                    u64::from(t.kmajor),
                ];
                let l = LAYOUT
                    .with(|c| c.borrow().get(&key).copied())
                    .unwrap_or_else(|| {
                        let r = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
                        let l = gpu::fma_layout(site, self.profile, t, &r);
                        let l = (l.pad_a, l.a_row_lanes);
                        LAYOUT.with(|c| c.borrow_mut().insert(key, l));
                        l
                    });
                dom = vec![match name {
                    "pad_a" => l.0,
                    _ => l.1,
                }];
            }
            (FMA, "held") => dom = vec![0],
            (MMA, "nt") => dom
                .retain(|&nt| pre[1] * nt * 4 + crate::profile::MMA_ADDR_REGS <= g.regs_per_thread),
            (MMA, "warps_y") => dom.retain(|&wy| divides(rows, wy * pre[1] * 16)),
            (MMA, "warps_x") => dom.retain(|&wx| {
                wx * pre[3] * g.warp <= threads_max && divides(site.c, wx * pre[2] * 8)
            }),
            // Whole 16-byte copies a thread of each panel.
            (MMA, "bk") => {
                let (ea, eb) = self.widths(pre[0]).unwrap_or((4, 4));
                let (bm, bn) = (pre[3] * pre[1] * 16, pre[4] * pre[2] * 8);
                let threads = pre[3] * pre[4] * g.warp;
                let (la, lb) = (GPU_VEC_BYTES / ea, GPU_VEC_BYTES / eb);
                dom.retain(|&bk| {
                    (bm * bk / la).is_multiple_of(threads) && (bk * bn / lb).is_multiple_of(threads)
                });
            }
            // A `cp.async` ring of two stages at least (the emitter's ring
            // waits on `stages − 2` groups: a single-panel fold of one stage
            // stores no product), within the slot's shared memory.
            (MMA, "stages") => dom.retain(|&st| {
                st >= 2
                    && self
                        .mma_tile(pre, st)
                        .is_some_and(|t| t.smem_bytes() <= g.smem_per_block)
            }),
            (MMA, "bt") => dom = vec![u64::from(site.b.clane != 1)],
            (MMA, "pad_a") | (MMA, "pad_b") => {
                let pads = self.mma_pads(pre);
                dom = match (name, pads) {
                    (_, None) => vec![],
                    ("pad_a", Some((a, _))) => vec![a],
                    (_, Some((_, b))) => vec![b],
                };
            }
            (MMA, "split") => dom = vec![0],
            (MMA, "ea") | (MMA, "eb") | (_, "ea") | (_, "eb") => {
                let (ea, eb) = self.widths(pre[0]).unwrap_or((0, 0));
                dom = vec![if name == "ea" { ea } else { eb }];
            }
            (WGMMA | WGMMA_WS, "n") => dom.retain(|&n| n <= 256),
            (WGMMA | WGMMA_WS, "bk") => {
                let (_, eb) = self.widths(pre[0]).unwrap_or((4, 4));
                dom.retain(|&bk| (bk * eb).is_multiple_of(128));
            }
            (WGMMA, "stages") => dom.retain(|&st| st >= crate::profile::WGMMA_MIN_STAGES),
            // The warp-specialized ring: two stages at least (one only where
            // one panel is the fold: the direct form).
            (WGMMA_WS, "stages") => dom.retain(|&st| st >= 2 || site.k / pre[3].max(1) == 1),
            (WGMMA | WGMMA_WS, "b_kmajor") => dom = vec![u64::from(site.b.clane != 1)],
            (WGMMA | WGMMA_WS, "pad_b") => dom = vec![0],
            (WGMMA | WGMMA_WS, "swap") => {
                let (n, rows_g) = (pre[1], pre[2] * 64);
                dom.retain(|&sw| match sw {
                    1 => divides(site.rows, n) && divides(site.c, rows_g),
                    _ => divides(site.rows, rows_g) && divides(site.c, n),
                });
            }
            (WGMMA | WGMMA_WS, "regs_p") | (WGMMA | WGMMA_WS, "regs_c") => dom = vec![0],
            _ => {}
        }
        // The largest tiles and panels first: the threshold the bound cuts
        // at tightens early (the pick is the same in any order).
        if matches!(
            name,
            "cells"
                | "tm"
                | "tn"
                | "tx"
                | "ty"
                | "mt"
                | "nt"
                | "warps_y"
                | "warps_x"
                | "n"
                | "groups"
                | "bk"
        ) {
            dom.reverse();
        }
        dom
    }

    /// The warp tile's panel pads by the bank model (the emitter's rule,
    /// `gpu_mma_panel_cands`): a TF32 panel's conflict-free pad, an f32
    /// panel of a 16-bit product's pair pad, a 16-bit panel's `ldmatrix`
    /// rows; `None` where no pad is conflict-free.
    pub(super) fn mma_pads(&self, pre: &[u64]) -> Option<(u64, u64)> {
        use crate::profile::{conflict_free_pad, ldmatrix_pad, pair_pad};
        let fmt = self.fmt(pre[0])?;
        let (ea, eb) = self.widths(pre[0])?;
        let (bk, bn) = (pre[5], pre[4] * pre[2] * 8);
        let bt = self.site.b.clane != 1;
        let pad_a = match (fmt, ea) {
            (MmaFmt::Tf32, _) => conflict_free_pad(bk, |l| (l / 4, l % 4))?,
            (_, 4) => pair_pad(bk)?,
            _ => ldmatrix_pad(bk),
        };
        let pad_b = match (bt, fmt, eb) {
            (false, MmaFmt::Tf32, _) => conflict_free_pad(bn, |l| (l % 4, l / 4))?,
            (false, _, 4) => conflict_free_pad(bn, |l| (2 * (l % 4), l / 4))?,
            (false, _, _) => ldmatrix_pad(bn),
            (true, _, 4) => pair_pad(bk)?,
            (true, _, _) => ldmatrix_pad(bk),
        };
        Some((pad_a, pad_b))
    }

    /// The warp tile of an MMA prefix (through `bk`) at `stages`.
    pub(super) fn mma_tile(&self, pre: &[u64], stages: u64) -> Option<GpuMmaTile> {
        let (ea, eb) = self.widths(pre[0])?;
        let (pad_a, pad_b) = self.mma_pads(pre)?;
        Some(GpuMmaTile {
            mt: pre[1],
            nt: pre[2],
            warps_y: pre[3],
            warps_x: pre[4],
            bk: pre[5],
            stages,
            pad_a,
            pad_b,
            blocks_per_sm: 0,
            fmt: self.fmt(pre[0])?,
            ea,
            eb,
            bt: self.site.b.clane != 1,
            split: None,
        })
    }

    /// The matrix row's format of operation id `op` (`None`: a scalar op).
    pub(super) fn fmt(&self, op: u64) -> Option<MmaFmt> {
        self.gpu()
            .caps
            .matrix
            .get((op as usize).wrapping_sub(2))
            .map(|o| o.fmt)
    }

    /// The panel widths the site's operands stage at on operation `op`: a
    /// 16-bit format's ([`gpu::mma_widths`]), TF32's four bytes.
    pub(super) fn widths(&self, op: u64) -> Option<(u64, u64)> {
        match self.fmt(op)? {
            MmaFmt::Tf32 => Some((4, 4)),
            f => gpu::mma_widths(self.site, f),
        }
    }

    /// The FMA tile of an FMA prefix (through `kmajor`), its layout unset.
    pub(super) fn fma_tile(&self, c: &[u64]) -> GpuTile {
        GpuTile {
            tm: c[1],
            tn: c[2],
            tx: c[3],
            ty: c[4],
            bk: c[5],
            blocks_per_sm: 0,
            stages: c.get(6).copied().unwrap_or(1),
            pad_a: 0,
            a_row_lanes: 0,
            held: 0,
            kmajor: c.get(7) == Some(&1),
        }
    }

    /// A schedule from coordinates (the enumeration's point), or `None`
    /// where the coordinates name no tile of this lowering.
    pub(super) fn build(&self, c: &[u64], swap: bool) -> Option<Schedule> {
        let g = self.gpu();
        let s = self.site;
        let grid = Level {
            ext: [s.rows, s.c, s.k],
            bind: Bind::Exec(E_DEVICE),
            homes: [None, None],
        };
        let mk = |lowering, op: u64, levels, groups| Schedule {
            lowering,
            op: op as usize,
            swap,
            levels,
            groups,
            producers: 0,
            regs: [0, 0],
            cluster: [1, 1],
            multicast: [false, false],
            trav: Traversal::Ordered { h: 1 },
            launch: LaunchMode::Plain,
            policy: [Policy::Default; 2],
            per_unit: 0,
            params: Vec::new(),
        };
        let maj = |b: u64| if b == 1 { Major::K } else { Major::Mn };
        Some(match self.id {
            NAIVE => {
                let cells = c[1];
                let bn = (NAIVE_THREADS * cells).min(s.c);
                let bm = ((NAIVE_THREADS * cells) / s.c.max(1)).clamp(1, s.rows.max(1));
                let mut x = mk(
                    NAIVE,
                    c[0],
                    vec![
                        Level {
                            ext: [1, cells, 1],
                            bind: Bind::Exec(E_LANE),
                            homes: [None, None],
                        },
                        Level {
                            ext: [bm, bn, NAIVE_UNROLL],
                            bind: Bind::Slot,
                            homes: [None, None],
                        },
                        grid,
                    ],
                    [NAIVE_THREADS / g.warp.max(1), 1],
                );
                x.params = vec![("cells", cells)];
                x
            }
            FMA => {
                let (tm, tn, tx, ty, bk, st) = (c[1], c[2], c[3], c[4], c[5], c[6]);
                let mut x = mk(
                    FMA,
                    c[0],
                    vec![
                        Level {
                            ext: [tm, tn, 1],
                            bind: Bind::Exec(E_LANE),
                            homes: [None, None],
                        },
                        Level {
                            ext: [ty, tx, 1],
                            bind: Bind::Exec(E_GROUP),
                            homes: [None, None],
                        },
                        Level {
                            ext: [ty * tm, tx * tn, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, maj(c[7]), c[8], 0),
                                smem_home(st, Major::Mn, 0, 0),
                            ],
                        },
                        grid,
                    ],
                    [(tx * ty).div_ceil(g.warp), 1],
                );
                x.params = vec![("a_row_lanes", c[9]), ("held", c[10])];
                x
            }
            MMA => {
                let o = g.caps.matrix.get((c[0] as usize).wrapping_sub(2))?;
                let (mt, nt, wy, wx, bk, st) = (c[1], c[2], c[3], c[4], c[5], c[6]);
                let mut x = mk(
                    MMA,
                    c[0],
                    vec![
                        Level {
                            ext: [16 * mt, 8 * nt, o.shape.k],
                            bind: Bind::Exec(E_GROUP),
                            homes: [None, None],
                        },
                        Level {
                            ext: [wy * 16 * mt, wx * 8 * nt, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, Major::K, c[8], 0),
                                smem_home(st, maj(c[7]), c[9], 0),
                            ],
                        },
                        grid,
                    ],
                    [wy, wx],
                );
                x.params = vec![
                    ("ea", c[11]),
                    ("eb", c[12]),
                    ("split", c[10]),
                    ("mt", mt),
                    ("nt", nt),
                ];
                if c[10] > 0 {
                    x.params.push(("split_arith", 1));
                }
                x
            }
            _ => {
                let (n, groups, bk, st) = (c[1], c[2], c[3], c[4]);
                let swap = c[7] == 1;
                let (bm, bn) = match swap {
                    false => (groups * 64, n),
                    true => (n, groups * 64),
                };
                let a_home = match g
                    .caps
                    .matrix
                    .get((c[0] as usize).wrapping_sub(2))
                    .map(|o| o.a)
                {
                    Some(MHome::Regs) => REGS,
                    _ => SMEM,
                };
                let mut x = mk(
                    self.id,
                    c[0],
                    vec![
                        Level {
                            ext: [64, n, 8],
                            bind: Bind::Exec(E_GROUP),
                            homes: [
                                Some(Home {
                                    place: a_home,
                                    layout: Layout::K,
                                    depth: 1,
                                }),
                                None,
                            ],
                        },
                        Level {
                            ext: [bm, bn, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, Major::K, 0, 128),
                                smem_home(
                                    st,
                                    maj(c[5]),
                                    c[6],
                                    if self.id == WGMMA_WS { 128 } else { 0 },
                                ),
                            ],
                        },
                        grid,
                    ],
                    [groups, 1],
                );
                x.swap = swap;
                x.producers = u64::from(self.id == WGMMA_WS);
                x.regs = [c[10], c[11]];
                x.params = vec![
                    ("ea", c[8]),
                    ("eb", c[9]),
                    ("pad_b", c[6]),
                    ("b_kmajor", c[5]),
                    ("n", n),
                ];
                x
            }
        })
    }
}
