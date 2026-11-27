//! `TargetProfile`: the warpgroup (`wgmma`) tile family.

use super::*;

impl TargetProfile {
    /// **The Hopper `wgmma` geometry for a `rows × c × k` f32 site on the TF32
    /// face** ([`GpuWgmmaTile`]). `None` off `sm_90a`. For each panel depth
    /// `bk` (a power of two ≥ the instruction's k, dividing `k`):
    ///
    /// 1. **`n`** — the largest multiple of 8 ≤ 256 dividing `rows` whose
    ///    `n/2` accumulators, the panel's `4·bk/8` A fragments and
    ///    [`WGMMA_ADDR_REGS`] fit `regs_per_thread` (the whole-thread rule).
    /// 2. **Warpgroups per block** (a power of two) from `regs_per_sm` over
    ///    that count, all in **one** resident block, not split across
    ///    [`GPU_RESIDENT_BLOCKS`]: that policy keeps a second block computing
    ///    while one waits at the panel barrier, and here the MMA group in
    ///    flight computes through it. Measured (S53 run 1, H100 PCIe): the
    ///    split geometry ran 0.61–0.64× of this one at 2048–8192, and with
    ///    128-byte swizzle rows (`bk` ≥ 32) it has no three-stage tile at all.
    /// 3. **Fill** — while the grid has fewer blocks than SMs, halve the
    ///    warpgroups, then `n`.
    /// 4. **Smem** — `b` rows padded by [`conflict_free_pad`] for the A
    ///    fragment pattern; the panel must fit [`WGMMA_MIN_STAGES`] stages
    ///    in a block's share, copy in whole 16-byte chunks per thread, and
    ///    fill whole [`WGMMA_SWIZZLE_BYTES`] rows.
    ///
    /// The deepest valid `bk` wins; then as many stages as the share holds.
    ///
    /// ADR-0034 D1 (site-geometry B4): a forced `n` / warpgroup count
    /// ([`Geom::wg_n`], `wg_groups`) replaces steps 1–3; one the site cannot
    /// take falls back to the derived tile, never to another kernel class.
    pub fn gpu_wgmma_tile(&self, rows: u64, c: u64, k: u64) -> Option<GpuWgmmaTile> {
        self.gpu_wgmma_tile_s53(rows, c, k)
    }

    /// **hopper-wgmma-tma R2 (`wgmma-ws`): the warp-specialized `wgmma`
    /// geometry** for a `rows × c × k` f32 site on the TF32 face, fed by
    /// tensor boxes. `None` without the warpgroup product, the bulk-copy
    /// engine, or a geometry below. `realloc` (R3, `ws-regs`): the part's
    /// [`Caps::reg_realloc`] may move registers between the producer and the
    /// consumers.
    ///
    /// - **Depth.** `bk` is the TF32 swizzle span, 32 words: one `aᵀ` box row
    ///   is one 128-byte swizzle row, the descriptor's layout.
    /// - **Block.** One producer warpgroup and `g` ∈ {2, 1} consumer
    ///   warpgroups, one block per SM by construction (the shared memory
    ///   below is the block's own, not a register-derived share — why the
    ///   S53 derivation refused the 2048² MLP site).
    /// - **Registers.** A consumer holds `n/2` accumulators, `4·bk/8`
    ///   fragments and [`WGMMA_WS_ADDR_REGS`]: under `realloc` within
    ///   `(regs_per_sm − 128·`[`WGMMA_WS_PRODUCER_REGS`]`)/(128·g)` (down to
    ///   the step, at most `max`), else within the launch's even share
    ///   `regs_per_sm / threads`.
    /// - **Shared memory.** `stages` × (the `aᵀ` box + the `b` boxes, whole
    ///   1024-byte patterns) + 24 bytes of barriers a stage within
    ///   `smem_per_block`; as many stages as fit, at most `k/bk`, at least 2.
    /// - **Fill.** Among the `(g, n)` that fit (`n` a multiple of 8 dividing
    ///   `rows`, `bn = 64·g` dividing `c`), the largest flops per staged byte
    ///   (`bn·n/(bn + n)`) whose grid covers the SMs; none covers them: the
    ///   largest grid (the split that would fill it is open).
    pub fn gpu_wgmma_ws_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        realloc: bool,
        digits: bool,
    ) -> Option<GpuWgmmaTile> {
        let g = self.gpu()?;
        if !g.has_wgmma() {
            return None;
        }
        let op = g.caps.matrix_op(MmaFmt::Tf32, Home::Smem)?;
        let form = WsForm {
            op,
            ea: 4,
            eb: 4,
            b_kmajor: false,
        };
        self.gpu_wgmma_ws_tile_of(rows, c, k, realloc, digits, form)
    }

    /// wgmma-16bit: **[`Self::gpu_wgmma_ws_tile`] over any warpgroup
    /// product** — the one tile family, the instruction and the operands'
    /// staged widths a parameter ([`WsForm`]). The form's [`MatrixOp`]
    /// decides the orientation (`mn_major`: `c = a·b` with consumer
    /// warpgroups along `c`'s rows, `bm = 64·g`, `n` its columns; else the
    /// TF32 product transposed, `bn = 64·g`, `n` rows), `a`'s home (a
    /// register operand holds `4·bk/k` fragment registers; a descriptor
    /// none) and the depth unit — one 128-byte swizzle row of the format
    /// (32 TF32 words, 64 16-bit elements); an f32 `a` staged for a 16-bit
    /// product spans two digit columns a row. A lanes-major `b` lands in
    /// 128-byte lane slabs (`n` whole slabs); a k-major one in `n` rows.
    pub fn gpu_wgmma_ws_tile_of(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        realloc: bool,
        digits: bool,
        form: WsForm,
    ) -> Option<GpuWgmmaTile> {
        let g = self.gpu()?;
        let op = form.op;
        if g.caps.bulk.is_none() || op.b != Home::Smem || op.acc != Home::Regs {
            return None;
        }
        let direct = op.mn_major;
        let a_home = if direct { op.a } else { Home::Regs };
        let row = WGMMA_SWIZZLE_BYTES / op.fmt.bytes();
        if !k.is_multiple_of(row) {
            return None;
        }
        // R1's price: a stage's 2 copies (`tma-digits`) issue within the time
        // its tensor cores take to consume it — the shallowest such depth (a
        // whole number of swizzle rows, at most 4: the `a` box's k digits);
        // none: the deepest. Without digits a stage is one swizzle row deep.
        let depths: Vec<u64> = match digits {
            true => (1..=4)
                .map(|d| d * row)
                .filter(|bk| k.is_multiple_of(*bk))
                .collect(),
            false => vec![row],
        };
        let wg = WARPGROUP_WARPS * g.warp;
        let rr = g.caps.reg_realloc.filter(|_| realloc);
        let mut cands = Vec::new();
        for &bk in &depths {
            let a_regs = match a_home {
                Home::Regs => 4 * (bk / op.shape.k),
                _ => 0,
            };
            let frag = a_regs + WGMMA_WS_ADDR_REGS;
            for groups in [2u64, 1] {
                let threads = (groups + 1) * wg;
                if threads > g.max_threads_per_block {
                    continue;
                }
                let (cap, regs) = match rr {
                    Some(r) => {
                        let p = WGMMA_WS_PRODUCER_REGS.max(r.min);
                        let cap = (g.regs_per_sm.saturating_sub(wg * p) / (wg * groups)) / r.step
                            * r.step;
                        let cap = cap.min(r.max);
                        (cap, Some((p, cap)))
                    }
                    None => (
                        (g.regs_per_sm / threads).min(g.regs_per_thread) / 8 * 8,
                        None,
                    ),
                };
                if cap <= frag {
                    continue;
                }
                // The warpgroups' side, and the extent `n` runs along.
                let (side, along) = match direct {
                    true => (rows, c),
                    false => (c, rows),
                };
                if !side.is_multiple_of(groups * WGMMA_M) {
                    continue;
                }
                // A lanes-major `b` (the direct form's `n` lanes) in whole
                // 128-byte slabs.
                let step = match direct && !form.b_kmajor {
                    true => WGMMA_SWIZZLE_BYTES / form.eb,
                    false => WGMMA_N_STEP,
                };
                let n_max = (2 * (cap - frag)).min(WGMMA_N_MAX);
                for n in (1..=n_max / step).rev().map(|i| i * step) {
                    if !along.is_multiple_of(n) {
                        continue;
                    }
                    let geom = GpuWgmmaTile {
                        n,
                        groups,
                        bk,
                        stages: 2,
                        pad_b: 0,
                        blocks_per_sm: 1,
                        ws: true,
                        realloc: regs,
                        ctas: 0,
                        fmt: op.fmt,
                        direct,
                        ea: form.ea,
                        eb: form.eb,
                        a_home,
                        b_kmajor: form.b_kmajor,
                    };
                    let per = geom.ws_stage_bytes() + 24;
                    let stages = (g.smem_per_block / per).min(k / bk);
                    // At least a two-stage ring; wgmma-16bit: a fold one stage
                    // deep (`k = bk`, attention's head width 64) is one stage —
                    // nothing to overlap within a tile.
                    let least = match direct {
                        true => 2.min(k / bk),
                        false => 2,
                    };
                    if stages < least.max(1) {
                        continue;
                    }
                    cands.push(GpuWgmmaTile { stages, ..geom });
                }
            }
        }
        let tiles = |w: &GpuWgmmaTile| (rows / w.bm()) * (c / w.bn());
        let copies = if digits { 2 } else { 0 };
        let priced = |w: &GpuWgmmaTile| {
            let flops = 2 * w.bm() * w.bn() * w.bk;
            copies * g.tma_copy_ns() <= g.stage_consume_ns(flops, 1, &op)
        };
        // Per (groups, n): the shallowest priced depth, else the deepest.
        let pick = |w: &GpuWgmmaTile| {
            let same: Vec<&GpuWgmmaTile> = cands
                .iter()
                .filter(|x| x.n == w.n && x.groups == w.groups)
                .collect();
            let best = same
                .iter()
                .filter(|x| priced(x) && x.stages >= 3)
                .min_by_key(|x| x.bk)
                .or_else(|| same.iter().filter(|x| priced(x)).min_by_key(|x| x.bk))
                .or_else(|| same.iter().max_by_key(|x| x.bk))
                .copied();
            best.is_some_and(|b| b.bk == w.bk)
        };
        let cands: Vec<GpuWgmmaTile> = cands.iter().filter(|w| pick(w)).copied().collect();
        let intensity = |w: &GpuWgmmaTile| w.bn() * w.bm() * 1000 / (w.bn() + w.bm());
        cands
            .iter()
            .filter(|w| tiles(w) >= g.sms)
            .max_by_key(|w| (intensity(w), w.groups))
            .or_else(|| cands.iter().max_by_key(|w| (tiles(w), intensity(w))))
            .copied()
    }

    /// The S53 `wgmma` geometry ([`Self::gpu_wgmma_tile`]'s derivation).
    fn gpu_wgmma_tile_s53(&self, rows: u64, c: u64, k: u64) -> Option<GpuWgmmaTile> {
        let q = self.gpu()?.geom;
        self.wgmma_tile_at(rows, c, k, q.wg_n, q.wg_groups)
            .or_else(|| {
                (q.wg_n + q.wg_groups > 0)
                    .then(|| self.wgmma_tile_at(rows, c, k, 0, 0))
                    .flatten()
            })
    }

    fn wgmma_tile_at(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wg_n: u64,
        wg_groups: u64,
    ) -> Option<GpuWgmmaTile> {
        let g = self.gpu()?;
        if !g.has_wgmma() {
            return None;
        }
        let wg_threads = WARPGROUP_WARPS * g.warp;
        // The largest legal `n` ≤ `cap` dividing `rows`.
        let fit_n = |cap: u64| {
            (1..=cap.min(WGMMA_N_MAX) / WGMMA_N_STEP)
                .rev()
                .map(|i| i * WGMMA_N_STEP)
                .find(|n| rows.is_multiple_of(*n))
        };
        let regs = |n: u64, bk: u64| n / 2 + 4 * (bk / WGMMA_K) + WGMMA_ADDR_REGS;
        let mut best = None;
        let mut bk = WGMMA_K;
        while bk <= k {
            let frag_regs = regs(0, bk);
            let geom = (|| {
                if !k.is_multiple_of(bk) || frag_regs >= g.regs_per_thread {
                    return None;
                }
                let cap = 2 * (g.regs_per_thread - frag_regs);
                // site-geometry B4: a forced `n` that divides the rows and fits.
                let forced_n = Some(wg_n).filter(|&n| {
                    n > 0
                        && n % WGMMA_N_STEP == 0
                        && n <= cap.min(WGMMA_N_MAX)
                        && rows.is_multiple_of(n)
                });
                let mut n = forced_n.or_else(|| fit_n(cap))?;
                let resident = g.regs_per_sm / regs(n, bk);
                let per_block = (resident / wg_threads)
                    .min(g.max_threads_per_block / wg_threads)
                    .max(1);
                let forced_g = Some(wg_groups).filter(|&x| x.is_power_of_two() && x <= per_block);
                let mut groups = forced_g.unwrap_or(1 << per_block.ilog2());
                while forced_n.is_none()
                    && forced_g.is_none()
                    && (rows / n) * (c / (groups * WGMMA_M)) < g.sms
                {
                    if groups > 1 {
                        groups /= 2;
                    } else if let Some(m) = fit_n(n / 2) {
                        n = m;
                    } else {
                        break;
                    }
                }
                let (bn, threads) = (groups * WGMMA_M, groups * wg_threads);
                if !c.is_multiple_of(bn) {
                    return None;
                }
                let blocks_per_sm = residency(g, regs(n, bk), threads).max(1);
                let share = (g.smem_per_sm / blocks_per_sm)
                    .saturating_sub(g.smem_reserved_per_block)
                    .min(g.smem_per_block);
                // A fragment pattern: lane = 4·group + t reads row k = t,
                // column n = group of the `b` panel.
                let pad_b = conflict_free_pad(bn, |l| (l % 4, l / 4))?;
                let geom = GpuWgmmaTile {
                    n,
                    groups,
                    bk,
                    stages: WGMMA_MIN_STAGES,
                    pad_b,
                    blocks_per_sm,
                    ws: false,
                    realloc: None,
                    ctas: 0,
                    ..GpuWgmmaTile::TF32
                };
                // Whole 16-byte chunks per thread, and whole swizzle rows.
                let whole = (n * bk / 4).is_multiple_of(threads)
                    && (bk * bn / 4).is_multiple_of(threads)
                    && (bk * 4).is_multiple_of(WGMMA_SWIZZLE_BYTES);
                (whole && geom.smem_bytes() <= share).then(|| {
                    let stages = (share / (geom.stage_words() * 4))
                        .min(k / bk)
                        .max(WGMMA_MIN_STAGES);
                    GpuWgmmaTile { stages, ..geom }
                })
            })();
            best = geom.or(best);
            bk *= 2;
        }
        best
    }
}
