//! The matrix tile's combine, `k` cut and walk plans (`MmaCut`, `MmaWalk`).

use super::*;

/// **The chooser (ADR-0040 D4, gap map M1/G2/G6)**: how a site's `k` fold is
/// realized, from IR facts (`TilePlan`'s independent cells, the site's
/// extents and step) and the part's (`Launch::wave`, `sm`, L2).
///
/// - `sequential`: in order.
/// - `fixed`: the deterministic workspace split (`splitk`), when the grid
///   leaves resident slots empty.
/// - `free`: stream-K when its makespan — `⌈blocks·P / wave⌉` panel steps per
///   block plus one fixup — beats the best split's
///   `⌈blocks·S / wave⌉ · P/S` (wave quantization, G6); else the atomic split
///   (`splitk-atomic`); else the workspace split (deterministic, legal under
///   `free`); else in order. Each step down is the rung above's OFF arm.
///
/// Only the scalar kernels have a fold to cut; `raster` and a split are
/// exclusive by derivation (a grid is grouped only above one wave, split only
/// below), stream-K walks tiles in the raster order instead. A forced
/// `--splitk=S` replaces the derived count where `S` divides the panels.
/// **f32-tensor-cores R-T1 `mma-combine`: a tensor-core tile's `k` cut into
/// pieces** (ADR-0040 D4 on the `mma.sync` kernels, [`Kernel::mma`] /
/// [`Kernel::mma16`]): the partials of `S` pieces add into the zeroed output
/// (`splitk-atomic`) or a workspace (`splitk`). A forced count (`--splitk=S`,
/// the part's `Geom::split`) where it cuts the panels whole; else the rule
/// [`mma_pieces`]. Free order only (the tensor tiles exist only there); not
/// a mixed-radix site (its digit-affine copies take no piece offset) nor a
/// bulk-staged one (its box copies none either).
pub(super) fn mma_combine(
    l: &Launch,
    m: &GpuMmaTile,
    site: &TileSite,
    profile: &TargetProfile,
    order: Order,
    arms: crate::GpuArms,
) -> Combine {
    let Some(gpu) = profile.gpu() else {
        return Combine::None;
    };
    if let Some((c, _)) = mma_walk(l, site, profile, arms, order) {
        return c;
    }
    let rungs = l.rungs;
    if !rungs.mma_combine
        || order != Order::Free
        || site.radix.is_some()
        || l.raster != 1
        || bulk_stage(site, m, gpu, &rungs)
    {
        return Combine::None;
    }
    let panels = site.k / m.bk;
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    let s = match forced {
        0 => mma_pieces(l, m, site, gpu),
        s if s > 1 && panels.is_multiple_of(s) => s,
        _ => 1,
    };
    // A split form's bound counts the pieces' combines (`S − 1` adds).
    let admitted = |s: u64| match (m.split, gpu.caps.matrix_op(m.fmt, Home::Regs)) {
        (Some(kind), Some(op)) => SplitForm::of(kind)
            .admits(site.k, s, op.shape.k, op.acc_model(), 1)
            .is_some(),
        _ => true,
    };
    let s = if admitted(s) { s } else { 1 };
    let cells = site.rows * site.c;
    match s {
        1 => Combine::None,
        s if rungs.splitk_atomic => Combine::Atomic { s },
        s if rungs.splitk && s * cells * 4 <= gpu.l2_bytes => Combine::Workspace { s },
        _ => Combine::None,
    }
}

/// f32-tensor-cores `mma-combine`: the fewest panels one piece of a split
/// tensor tile folds (the measured pieces: 16–112; none shorter was timed).
pub(crate) const MMA_PIECE_PANELS: u64 = 8;

/// f32-tensor-cores `mma-combine`: the most pieces a tensor tile is cut into
/// (16 lost the m128 cell, 0.1488 against 0.1373 ms).
pub(crate) const MMA_PIECES_MAX: u64 = 8;

/// f32-tensor-cores `mma-combine`: the predicted time a cut must save, per
/// mille — beyond the rtx-5080's null (2048 × 6144 × 4096: predicted 0.97,
/// measured 1.009; 128 × 4096 × 4096 predicted 0.97, measured 0.98 / 1.08 in
/// two sessions) and inside every measured gain (predicted ≤ 0.90).
pub(crate) const MMA_CUT_PM: u64 = 950;

/// **[`mma_combine`]'s rule: the pieces a tensor tile's grid takes** (1:
/// none). A grid of `blocks` tiles on the `slots` the tile holds (`sms ×` its
/// blocks an SM) runs `⌈blocks·S / slots⌉` waves of pieces of `P/S` panels
/// each, and every piece pays a fixed cost: its pipeline's prologue (`stages −
/// 1` panels landing before its first product) and its fixup (the tile's
/// partials added into the output, about one panel) — `stages` panels. The
/// cut `S` (dividing the panels, each piece at least [`MMA_PIECE_PANELS`], at
/// most [`MMA_PIECES_MAX`]) minimizing `⌈blocks·S / slots⌉·(P/S + stages)`
/// panel times plus the cut's own partials and zero pass, in ns at the
/// part's rates ([`MmaCut::ns`], bugs-properties `cut-price`), is taken
/// where it saves more than [`MMA_CUT_PM`] of the uncut time. A
/// grid of at least one wave over a site whose flops per operand byte stay
/// below twice the part's matrix balance (its matrix rate over its DRAM
/// rate: every SM already streams the operands from DRAM) is not cut: the
/// cut only adds its partials' traffic.
///
/// rtx-5080, TF32, ABBA (`f32-tensor-cores.md`): winners 128 × 6144 × 4096
/// (96 blocks on 168 slots, `S` = 8: 0.181 against 0.239 ms), 128 × 4096 ×
/// 14336 (0.411 / 0.458), 2048 × 4096 × 4096 (1.335 / 1.416), 2048 × 4096 ×
/// 14336 (4.43 / 4.90), mlp_2048 / mlp2_2048 (0.945 / 0.940); refused: 128 ×
/// 28672 × 4096 (224 blocks, 61.8 flop / B against 59.2: every cut lost,
/// 0.744–0.764 against 0.696 ms), 2048 × 6144 × 4096 (null), 128 × 4096 ×
/// 4096 (16 panels a piece: mixed).
pub(super) fn mma_pieces(l: &Launch, m: &GpuMmaTile, site: &TileSite, gpu: &Gpu) -> u64 {
    let blocks = l.gx * l.gy;
    let slots = gpu.sms * m.blocks_per_sm.max(1);
    let panels = site.k / m.bk;
    let rate = gpu
        .caps
        .matrix_op(m.fmt, Home::Regs)
        .map_or(0, |op| gpu.matrix_flops_per_ns(&op));
    // integration (tensor-cores-2 × bugs-properties `cut-price`): a split
    // tile's panel runs its form's `P` products at the tile's issue ratio
    // ([`split_issue`]) — the rate its panels realize, the one its waves and
    // its cut's fixed work are weighed at. At the matrix pipe's rate the
    // A100's tf32x3 panels were priced 3× fast, so the cut's partials
    // outweighed the waves it saves: `pj_m128_n4096_k4096` / `_k14336` went
    // uncut where tensor-cores-2 measured their `S` = 8 cuts winning (1.105 /
    // 1.055 over the FFMA tile); priced here, both cut again. The 2048-row
    // tiles stay whole: tensor-cores-2's run timed the 2048³ tf32x3 tile both
    // ways side by side (`mlp_2048` / `mlp2_2048`: whole 0.891–0.901 ms, `S`
    // = 4 0.915–0.919, the same hot loop — the cut's `red` adds and zero pass
    // cost more than the wave tail it saves). `pj_m2048_n4096_k4096` /
    // `_k14336` (measured at `S` = 2, now whole) have no whole-tile timing:
    // open until the next A100 round times them.
    let rate = match m.split {
        Some(kind) => {
            let issue = split_issue(site, gpu, m, kind, l.rungs.split_issue).max(1000);
            rate * 1000 / (SplitForm::of(kind).products().max(1) * issue)
        }
        None => rate,
    };
    let dram = gpu
        .cost
        .map_or(0, |d| d.dram_bytes_per_ns)
        .max(gpu.dram_peak_bytes_per_ns());
    let bytes = site.rows * site.k * m.ea + site.k * site.c * m.eb + site.rows * site.c * 4;
    let flops = 2 * site.rows * site.c * site.k;
    // flops / bytes ≥ 2 · rate / dram, in integers.
    let compute = dram > 0 && rate > 0 && flops * dram >= 2 * rate * bytes;
    if blocks >= slots && !compute {
        return 1;
    }
    let cut = MmaCut {
        blocks,
        slots,
        panels,
        stages: m.stages,
        panel_flops: 2 * m.bm() * m.bn() * m.bk,
        blocks_per_sm: m.blocks_per_sm.max(1),
        cells: site.rows * site.c,
        bytes,
    };
    let time = |s: u64| cut.ns(gpu, rate, s);
    let mut best = (1, time(1));
    let mut s = 2;
    while s <= MMA_PIECES_MAX {
        if panels.is_multiple_of(s) && panels / s >= MMA_PIECE_PANELS && time(s) < best.1 {
            best = (s, time(s));
        }
        s *= 2;
    }
    match best.1 * 1000.0 <= MMA_CUT_PM as f64 * time(1) {
        true => best.0,
        false => 1,
    }
}

/// bugs-properties `cut-price`: **a tensor tile's grid and the price of
/// cutting its `k`** ([`mma_pieces`]) — `blocks` tiles on `slots` resident
/// places, `panels` panels of `panel_flops` each, a pipeline of `stages`,
/// `blocks_per_sm` blocks sharing an SM, `cells` outputs, `bytes` of
/// operands and output (its working set).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MmaCut {
    pub blocks: u64,
    pub slots: u64,
    pub panels: u64,
    pub stages: u64,
    pub panel_flops: u64,
    pub blocks_per_sm: u64,
    pub cells: u64,
    pub bytes: u64,
}

impl MmaCut {
    /// **The cut into `s` pieces, ns**: `⌈blocks·s / slots⌉` waves of
    /// pieces of `panels / s + stages` panels (the prologue and the fixup
    /// about `stages` panels), a panel at one block's share of the SM's
    /// matrix rate (`rate`, the part's flops per ns), and — where it cuts —
    /// the cut's own work: the output's zero pass (`zero_ns` and its bytes at
    /// the fill rate) and `s · cells` partials added by `red` at the part's
    /// `atomic_ops_per_ns` ([`crate::profile::Gpu::plan_cost`]'s rates, as
    /// `streamk_priced` prices a stream-K fixup) — where the cut's working
    /// set spills the L2 the part keeps ([`Gpu::l2_keep_bytes`]); inside it
    /// the adds resolve in the L2 beside the other pieces' products (`red`
    /// is issued and left), overlapped with the waves: the rtx-5080's 2048³
    /// MLP GEMM (48 MiB in its 66 MiB keep) won its cut 1.06× (local ABBA:
    /// 1.568 against 1.667 ms) where the serial price of its 33.5 M adds
    /// (84 µs) refused it. The panel count alone (the
    /// rule before) took every cut whose waves shrank: cross-arch 5fcb401b
    /// measured 9 of 20 such cuts losing (`mma-combine`, finding 5) — the
    /// A100's and 3090's 2048³ MLP GEMMs (`S` 4 / 8: 16.8–33.5 M adds at 248
    /// / 217 an ns, 68–155 µs against 62–68 µs of waves saved), the 4090's
    /// and H100's 128 × 6144 × 4096 (9.6–10.7 µs of partials against 2–10
    /// µs); priced, 18 of the 20 arms pick their measured side.
    pub fn ns(&self, gpu: &Gpu, rate: u64, s: u64) -> f64 {
        let waves = (self.blocks * s).div_ceil(self.slots) as f64;
        let per_sm = rate as f64 / gpu.sms.max(1) as f64;
        let panel = match per_sm > 0.0 {
            true => self.panel_flops as f64 * self.blocks_per_sm as f64 / per_sm,
            false => 1.0,
        };
        let run = waves * ((self.panels / s) as f64 + self.stages as f64) * panel;
        let d = gpu.plan_cost();
        let fill = match (d.fill_bytes_per_ns, d.l2_bytes_per_ns) {
            (0, l2) => l2,
            (f, _) => f,
        };
        let spills = self.bytes > gpu.l2_keep_bytes();
        let fixed = match (s, d.atomic_ops_per_ns, per_sm > 0.0) {
            (1, _, _) | (_, 0, _) | (_, _, false) => 0.0,
            (s, a, true) => {
                let adds = match spills {
                    true => (s * self.cells) as f64 / a as f64,
                    false => 0.0,
                };
                adds + d.zero_ns as f64 + (self.cells * 4) as f64 / fill.max(1) as f64
            }
        };
        run + fixed
    }
}

/// bf16 tensor-tile `mma-walk`: the price must save this much of the
/// ordered grid's (or its `k` cut's), per mille — [`MMA_CUT_PM`]'s margin.
pub(crate) const MMA_WALK_PM: u64 = 950;

/// **bf16 tensor-tile `mma-walk`: a 16-bit tensor tile's grid walked by
/// stream-K over every resident slot**, and the launch that runs it — `None`
/// where the price ([`MmaWalk`]) keeps the rules' grid `l`.
///
/// A tensor tile is bound by its SM's matrix pipe, which the SM's resident
/// blocks share: a grid of `tiles` tiles costs `⌈tiles / sms⌉` tiles of
/// products on its busiest SM, whatever the blocks an SM. Measured (RTX 5080,
/// bf16 `fold: bf16 | f32`, CUPTI; `bf16-tensor-tile.md`): 512 × 2048 × 8192
/// on 128 tiles of 64 × 128 (2 an SM, 216 µs = two tiles at the pipe's
/// rate, 44 SMs idle half the time), 512 × 8192 × 2048 and 2048 × 2048 × 2048
/// on 256 tiles of 128 × 128 (4 on the busiest SMs where 3.05 would do:
/// 192.7 / 193.1 µs, four tiles at 48.3), 2048 × 8192 × 2048 on 1024 (13
/// where 12.19 would do: 628.0 µs). The walk keeps the whole warp tile (the
/// `fill` rung's halving and the grid's warp halving are for a grid that
/// must fill the part by its tiles: [`TargetProfile::gpu_mma16_walk_tile`])
/// and spreads the `tiles × panels` products evenly over `sms ×` its blocks
/// an SM; its costs are the pieces' partials added by `red` into the zeroed
/// output (the pieces of shared tiles, [`MmaWalk::ns`]) and that zero pass
/// — on an epilogue group the tiles' workspace and tickets
/// ([`Kernel::ticket`]), priced at the larger of the two. Free order only (the
/// partials add in no fixed order); no split form (its bound counts a fixed
/// number of combines), radix or window site, forced cut, or bulk-staged
/// tile.
pub(crate) fn mma_walk(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    order: Order,
) -> Option<(Combine, Launch)> {
    let m = l.mma?;
    let gpu = profile.gpu()?;
    let r = l.rungs;
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some();
    if !r.mma_walk
        || !r.mma_combine
        || order != Order::Free
        || m.fmt == MmaFmt::Tf32
        || m.split.is_some()
        || site.radix.is_some()
        || window
        || l.wgmma.is_some()
        || site.elem != Ty::f32()
        || arms.off.get("streamk")
        || arms.splitk > 0
        || gpu.geom.split > 0
        || bulk_stage(site, &m, gpu, &r)
    {
        return None;
    }
    let e = elem_bytes_of(&site.elem);
    let rows = row_group(site);
    let w = profile.gpu_mma16_walk_tile(
        rows,
        site.c,
        site.k,
        r.wide,
        (m.fmt, m.ea, m.eb, m.bt),
        Depth::of(&r, e),
    )?;
    let (bm, bn) = (w.bm(), w.bn());
    if !site.rows.is_multiple_of(bm) || !site.c.is_multiple_of(bn) || !site.k.is_multiple_of(w.bk) {
        return None;
    }
    let rate = gpu
        .caps
        .matrix_op(m.fmt, Home::Regs)
        .map_or(0, |op| gpu.matrix_flops_per_ns(&op));
    let tiles = (site.rows / bm) * (site.c / bn);
    let panels = site.k / w.bk;
    let slots = gpu.sms * w.blocks_per_sm.max(1);
    if rate == 0 || panels < 2 || tiles * panels <= slots {
        return None;
    }
    // The walk visits its tiles row-major: where gapfix tensor-tile's C11
    // gate groups this grid's walk ([`Gpu::raster`]: a block's panels leave
    // the L2, or row-major's `b` re-reads leave the site DRAM-bound — 256 ×
    // 28672 × 4096 bf16, its 235 MB `b` read once a block-row), the ordered
    // grid keeps the grouped walk.
    let grouped = r.raster
        && gpu.raster(
            bm, bn, slots, site.rows, site.c, site.k, e, r.l2_gate, w.eb, rate,
        ) > 1;
    if grouped {
        return None;
    }
    let iters = (tiles * panels).div_ceil(slots);
    let ctas = (tiles * panels).div_ceil(iters);
    let price = MmaWalk {
        ordered: (l.gx * l.gy, 2 * m.bm() * m.bn() * site.k),
        cut: mma_pieces(l, &m, site, gpu),
        tiles,
        panels,
        panel_flops: 2 * bm * bn * w.bk,
        blocks_per_sm: w.blocks_per_sm.max(1),
        ctas,
        iters,
        tile_cells: bm * bn,
        cells: site.rows * site.c,
        bytes: site.rows * site.k * m.ea + site.k * site.c * m.eb + site.rows * site.c * 4,
        grid_x: site.c / bn,
        slices: (bm * w.bk * w.ea, bn * w.bk * w.eb),
    };
    // `--gpu-on=mma-walk` walks every admitted site (the measurement and
    // emulator arm, as `streamk-epi`'s).
    if !arms.on.get("mma-walk") && !price.pays(gpu, rate) {
        return None;
    }
    let c = Combine::StreamK {
        ctas,
        iters,
        whole: 0,
    };
    let walk = Launch {
        mma: Some(w),
        gx: site.c / bn,
        gy: site.rows / bm,
        block: w.threads(gpu.warp),
        smem: w.smem_bytes(),
        raster: 1,
        combine: c,
        ..*l
    };
    Some((c, walk))
}

/// bf16 tensor-tile `mma-walk`: **a tensor tile grid's time on the SMs'
/// matrix pipes, ordered and walked** ([`mma_walk`]). `ordered`: the rules'
/// grid (`tiles`, each tile's flops), `cut` the `k` cut [`mma_pieces`] gives
/// it; the walk: `tiles × panels` of `panel_flops` over `ctas` blocks of
/// `iters` panels, `blocks_per_sm` sharing an SM; `tile_cells` a tile's
/// outputs, `cells` the site's, `bytes` its operands and output; `grid_x`
/// the walk's tiles a block-row (tile `t` at row `t / grid_x`, column
/// `t % grid_x`: the order [`Kernel::segment`] walks), `slices` the bytes of
/// one panel of a tile's `a` (`bm·bk·ea`) and `b` (`bn·bk·eb`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MmaWalk {
    pub ordered: (u64, u64),
    pub cut: u64,
    pub tiles: u64,
    pub panels: u64,
    pub panel_flops: u64,
    pub blocks_per_sm: u64,
    pub ctas: u64,
    pub iters: u64,
    pub tile_cells: u64,
    pub cells: u64,
    pub bytes: u64,
    pub grid_x: u64,
    pub slices: (u64, u64),
}

/// hopper-wgmma `walk-reuse`: the panel offsets [`MmaWalk::operand_bytes`]
/// samples along `k` (the walk's reader times shift with the offset where
/// a reader crosses a block's boundary).
pub(super) const WALK_REUSE_SAMPLES: u64 = 8;

impl MmaWalk {
    /// hopper-wgmma `walk-reuse`: **the operand bytes the walk reads from
    /// DRAM** — each operand panel slice once per run of its readers that
    /// the L2 still holds between them. Block `c` runs units `[c·iters,
    /// (c+1)·iters)` back to back, so tile `t`'s panel `κ` is read at unit
    /// time `(t·panels + κ) mod iters`; the tiles of a block-row read the
    /// same `a` slice, those of a column the same `b` slice — at times the
    /// walk's rotation (`panels` against `iters`) scatters over the run. In
    /// one unit the walk streams `ctas · (a + b slice)` bytes; a reader within
    /// `keep / that` units of the previous one finds the slice in the L2
    /// (`keep`: [`Gpu::l2_keep_bytes`], the largest cyclic working set the
    /// part's L2 keeps — the measured property), a later one reads it again.
    /// The ordered grid runs a wave's tiles at one panel offset, so its
    /// readers share every slice (row-major's `b` distance is
    /// [`Gpu::b_rereads`], C11); the walk's offsets differ block to block.
    /// Measured (hopper `disasm-rented`): the RTX 3090's 2048 × 4096 × 4096
    /// walk (6 MiB keep: readers 16 units apart, 2.7 MB a unit) read ~1 GB
    /// for 50 MB of operands — 1275 µs against the ordered grid's 1082; the
    /// RTX 5080's 64 MiB keep holds the very grid's (628 against 667).
    pub fn operand_bytes(&self, keep: u64) -> u64 {
        let gx = self.grid_x.max(1);
        let gy = self.tiles / gx;
        let (sa, sb) = self.slices;
        let iters = self.iters.max(1);
        let near = keep / (self.ctas * (sa + sb)).max(1);
        let samples = self.panels.clamp(1, WALK_REUSE_SAMPLES);
        let mut times: Vec<u64> = Vec::with_capacity(gx.max(gy) as usize);
        // The runs of readers whose gap the L2 holds: one read each.
        let runs = |times: &mut Vec<u64>| -> u64 {
            times.sort_unstable();
            1 + times.windows(2).filter(|w| w[1] - w[0] > near).count() as u64
        };
        let (mut reads_a, mut reads_b) = (0, 0);
        for s in 0..samples {
            let kappa = s * self.panels / samples;
            let at = |t: u64| (t * self.panels + kappa) % iters;
            for r in 0..gy {
                times.clear();
                times.extend((0..gx).map(|j| at(r * gx + j)));
                reads_a += runs(&mut times);
            }
            for j in 0..gx {
                times.clear();
                times.extend((0..gy).map(|r| at(r * gx + j)));
                reads_b += runs(&mut times);
            }
        }
        (reads_a * sa + reads_b * sb) * self.panels / samples
    }

    /// The walk's DRAM bytes: [`Self::bytes`] with its operands read as
    /// [`Self::operand_bytes`] reads them (each slice at least once).
    pub fn walk_bytes(&self, keep: u64) -> u64 {
        let gx = self.grid_x.max(1);
        let (sa, sb) = self.slices;
        let once = (self.tiles / gx * sa + gx * sb) * self.panels;
        self.bytes + self.operand_bytes(keep).saturating_sub(once)
    }

    /// The pieces of shared tiles: a tile whose `panels` units span blocks
    /// `c₀ … c₁` of the walk (`iters` units a block) has `c₁ − c₀ + 1` pieces
    /// when it is shared, none (a whole store) otherwise.
    pub fn shared_pieces(&self) -> u64 {
        (0..self.tiles)
            .map(|t| {
                let c0 = t * self.panels / self.iters;
                let c1 = ((t + 1) * self.panels - 1) / self.iters;
                if c1 > c0 { c1 - c0 + 1 } else { 0 }
            })
            .sum()
    }

    /// **The three times, ns: `(ordered, cut, walk)`.** The busiest SM's
    /// products at its matrix pipe's rate (`rate / sms` flops a ns): the
    /// ordered grid `⌈tiles / sms⌉` tiles, the cut `⌈tiles·S / sms⌉` pieces
    /// of `1/S` of a tile, the walk `blocks_per_sm · iters` panels plus a
    /// pipeline refill (the part's loaded latency `lat_ns`) for each piece
    /// its blocks start, shared by the SM's blocks. A cut or a walk adds
    /// the output's zero pass (`zero_ns` and its bytes at the fill rate; the
    /// walk at the larger of the output and its tiles' workspace, an
    /// epilogue group's) and its partials at the part's `red` rate
    /// (`atomic_ops_per_ns`: measured, the 512 × 2048 × 8192 bf16 cuts paid
    /// 8 / 16 µs for their 4 M / 8 M adds, the rate's 10 / 20). None is
    /// below its reads at the DRAM rate: the ordered grid and its cut read
    /// the operands once (a wave's tiles share each panel slice), the walk
    /// as [`Self::walk_bytes`] reads them (hopper-wgmma `walk-reuse`).
    pub fn ns(&self, gpu: &Gpu, rate: u64) -> (f64, f64, f64) {
        let sms = gpu.sms.max(1);
        let per_sm = rate as f64 / sms as f64;
        let d = gpu.plan_cost();
        let fill = match (d.fill_bytes_per_ns, d.l2_bytes_per_ns) {
            (0, l2) => l2.max(1),
            (f, _) => f,
        } as f64;
        let dram = gpu
            .cost
            .map_or(0, |c| c.dram_bytes_per_ns)
            .max(gpu.dram_peak_bytes_per_ns())
            .max(1) as f64;
        let floor = self.bytes as f64 / dram;
        let adds = |n: u64| match d.atomic_ops_per_ns {
            0 => 0.0,
            a => n as f64 / a as f64,
        };
        let zero = |bytes: u64| d.zero_ns as f64 + bytes as f64 / fill;
        let (tiles, flops) = self.ordered;
        let ordered = (tiles.div_ceil(sms) * flops) as f64 / per_sm;
        let s = self.cut.max(1);
        let cut = match s {
            1 => ordered,
            s => {
                (tiles * s).div_ceil(sms) as f64 * flops as f64 / s as f64 / per_sm
                    + zero(self.cells * 4)
                    + adds(s * self.cells)
            }
        };
        let pieces = self.ctas + self.tiles;
        let refill = pieces.div_ceil(sms) as f64 * d.lat_ns as f64 / self.blocks_per_sm as f64;
        let ws = self.ctas * self.tile_cells * 4;
        let walk = (self.blocks_per_sm * self.iters * self.panel_flops) as f64 / per_sm
            + refill
            + zero(ws.max(self.cells * 4))
            + adds(self.shared_pieces() * self.tile_cells);
        let reads = self.walk_bytes(gpu.l2_keep_bytes()) as f64 / dram;
        (ordered.max(floor), cut.max(floor), walk.max(reads))
    }

    /// The walk saves [`MMA_WALK_PM`] of the better of the ordered grid
    /// and its cut — and its re-reads ([`Self::walk_bytes`]) do not make
    /// a site DRAM-bound that reading its operands once is not
    /// ([`Gpu::dram_bound`]: past [`crate::profile::DRAM_BOUND_PM`] of the products the
    /// load latency is no longer hidden, so the pipe time the walk evens
    /// out is not what bounds it — C11's line for row-major's `b` re-reads,
    /// hopper-wgmma `walk-reuse`: the RTX 3090's and A100's walks, the RTX
    /// 5080's 3072 × 4096 × 8192).
    pub fn pays(&self, gpu: &Gpu, rate: u64) -> bool {
        let (ordered, cut, walk) = self.ns(gpu, rate);
        let flops = self.tiles * self.panels * self.panel_flops;
        let rereads_bind = gpu.dram_bound(self.walk_bytes(gpu.l2_keep_bytes()), flops, rate)
            && !gpu.dram_bound(self.bytes, flops, rate);
        !rereads_bind && walk * 1000.0 <= MMA_WALK_PM as f64 * ordered.min(cut)
    }
}
