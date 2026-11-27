//! `TargetProfile`: the FFMA tile family (cp.async stages, group and cell tiles, k-major padding).

use super::*;

impl TargetProfile {
    /// **The `cp.async` panel pipeline for an FMA tile (sm_80+).** Panels stream
    /// into `stages` shared-memory buffers while earlier ones are consumed, so
    /// the prefetch holds no registers. `a` is copied element by element into
    /// its transposed panel — `GPU_VEC_BYTES / elem` lanes along `k` per row,
    /// so four lanes read one 16-byte global run — and its rows are padded by
    /// [`conflict_free_pad`] for exactly that copy pattern. `bk` is the largest
    /// power of two for which two stages fit a block's share of shared memory
    /// and every thread copies whole units; then as many stages as fit, up to
    /// the number of panels. `None` on a part without `cp.async`.
    pub fn gpu_tile_async(&self, g: GpuTile, k: u64, elem: &Ty) -> Option<GpuTile> {
        let gpu = self.gpu()?;
        if !gpu.has_mma_sync() {
            return None;
        }
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let (bm, bn, threads) = (g.bm(), g.bn(), g.threads());
        // tile-efficiency T2: a k-major panel's 16-byte copies write
        // consecutive chunks of contiguous rows — conflict-free unpadded.
        let pad_a = match g.kmajor {
            true => 0,
            false => conflict_free_pad(bm, |l| (l % lanes, l / lanes))?,
        };
        let a_units = if g.kmajor { lanes } else { 1 };
        let share = (gpu.smem_per_sm / g.blocks_per_sm.max(1))
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let mut best = None;
        let mut bk = lanes;
        while bk <= k {
            let geom = GpuTile {
                bk,
                stages: 2,
                pad_a,
                ..g
            };
            if geom.smem_bytes(e) > share {
                break;
            }
            let whole = (bm * bk / a_units).is_multiple_of(threads)
                && (bk * bn / lanes).is_multiple_of(threads);
            // ADR-0034 D1: a forced panel depth is the only one taken.
            if k.is_multiple_of(bk) && whole && (gpu.geom.bk == 0 || bk == gpu.geom.bk) {
                best = Some(geom);
            }
            bk *= 2;
        }
        let geom = best?;
        let stages = (share / (geom.stage_elems() * e)).min(k / geom.bk).max(2);
        // A forced stage count that fits the share replaces the derived one.
        let stages = Some(gpu.geom.stages)
            .filter(|&s| s >= 2 && GpuTile { stages: s, ..geom }.smem_bytes(e) <= share)
            .unwrap_or(stages);
        Some(GpuTile { stages, ..geom })
    }

    /// **device-geometry `head-group`: the FMA tile of a row group** — a
    /// block that holds all `group` rows of one batch (a KV head's query
    /// heads) as one thread's `tm = group` register rows, and a warp (or the
    /// widest power of two up to one whose vectors tile `c`) over the lanes,
    /// `tn` = one vector each: its `b` panel (the KV head's rows)
    /// is staged once for the group. The accumulators and fragments
    /// ([`fma_held_regs`]) fit half of `regs_per_thread`, as
    /// [`Self::gpu_tile`]'s step 1. The panel depth: the deepest `bk` up to a
    /// warp's steps dividing `k` whose panels every thread loads whole and
    /// the block's share holds. `None` where the group is no power of two,
    /// its rows outgrow the registers, or `c` is no whole number of vectors.
    pub fn gpu_group_tile(&self, group: u64, c: u64, k: u64, elem: &Ty) -> Option<GpuTile> {
        let g = self.gpu()?;
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        if !group.is_power_of_two()
            || fma_held_regs(group, lanes, e) > g.regs_per_thread / 2
            || !c.is_multiple_of(lanes)
        {
            return None;
        }
        let (tm, tn) = (group, lanes);
        // One warp over the lanes — the widest power of two up to a warp
        // whose vectors tile `c` (a 64-wide head: half a warp).
        let vecs = c / tn;
        let tx = 1u64 << vecs.trailing_zeros().min(g.warp.ilog2());
        let held = fma_held_regs(tm, tn, e);
        let (bm, bn) = (tm, tx * tn);
        let loads_whole =
            |bk: u64| (bm * bk / lanes).is_multiple_of(tx) && (bk * bn / lanes).is_multiple_of(tx);
        // The deepest panel up to one warp's steps — or the shallowest whose
        // `a` panel every thread loads whole, for a group of fewer rows than
        // a warp's vectors: the group's rows are few, so the panel is mostly
        // `b`'s, and its residency follows from it.
        let fits = |bk: u64| {
            bk >= lanes
                && k.is_multiple_of(bk)
                && (bm + bn) * bk * e + g.smem_reserved_per_block <= g.smem_per_block
                && loads_whole(bk)
        };
        let depths = || {
            (0..=k.max(1).ilog2())
                .map(|p| 1u64 << p)
                .filter(|&bk| fits(bk))
        };
        let cap = depths().min()?.max(g.warp.max(lanes));
        let bk = depths().filter(|&bk| bk <= cap).max()?;
        let panel = (bm + bn) * bk * e + g.smem_reserved_per_block;
        let blocks_per_sm = residency(g, held, tx).min(g.smem_per_sm / panel).max(1);
        Some(GpuTile {
            tm,
            tn,
            tx,
            ty: 1,
            bk,
            blocks_per_sm,
            stages: 1,
            pad_a: 0,
            a_row_lanes: 0,
            held: 0,
            kmajor: false,
        })
    }

    /// **One output per thread as a staged tile (S62 rung `cell-fill`)**, for
    /// a shaped `rows × c × k` site [`Self::gpu_tile`] does not divide (its
    /// thread grid, halved to fill the SMs, outgrows `rows`). The fold of each
    /// cell stays one register's in-order chain (`tm = tn = 1`); what changes
    /// is who reads a panel element: the block's `bm` rows share one `b`
    /// lane panel and its `bn` lanes one `a` row panel, where the
    /// one-thread-per-cell block (one row, `threads` lanes) reads `b` once per
    /// output row.
    ///
    /// - Threads: [`Self::gpu_tile`] step 2's count — the one-thread-per-cell
    ///   grid's block size on every profiled part.
    /// - Block: square-most, `bm` the largest power of two dividing `rows`
    ///   with `bm² ≤ threads`, `bn = threads / bm` (fewest panel bytes,
    ///   `(bm + bn)·bk`, per `bm·bn` cells); `c % bn == 0`.
    /// - Blocks per SM from what such a thread holds ([`fma_held_regs`] at
    ///   1 × 1), the thread and block-slot limits.
    /// - `bk`: [`Self::gpu_tile`] step 5 at that share — every thread copies
    ///   whole vectors, so a one-row block (`bm = 1`) needs `bk ≥ lanes·threads`
    ///   and is refused by the share: one-row sites keep `naive`.
    pub fn gpu_cell_tile(&self, rows: u64, c: u64, k: u64, elem: &Ty) -> Option<GpuTile> {
        let g = self.gpu()?;
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let threads = block_threads(g, square_register_tile(g, e).1);
        let bm = (0..=rows.max(1).ilog2())
            .map(|p| 1u64 << p)
            .filter(|&b| rows.is_multiple_of(b) && b * b <= threads)
            .max()?;
        let bn = threads / bm;
        if !c.is_multiple_of(bn) {
            return None;
        }
        let blocks_per_sm = residency(g, fma_held_regs(1, 1, e), threads).max(1);
        let share = (g.smem_per_sm / blocks_per_sm)
            .saturating_sub(g.smem_reserved_per_block)
            .min(g.smem_per_block);
        let loads_whole = |bk: u64| {
            (bm * bk / lanes).is_multiple_of(threads) && (bk * bn / lanes).is_multiple_of(threads)
        };
        let bk = (0..=k.max(1).ilog2())
            .map(|p| 1u64 << p)
            .filter(|&bk| {
                bk >= lanes
                    && k.is_multiple_of(bk)
                    && (bm + bn) * bk * e <= share
                    && loads_whole(bk)
            })
            .max()?;
        Some(GpuTile {
            tm: 1,
            tn: 1,
            tx: bn,
            ty: bm,
            bk,
            blocks_per_sm,
            stages: 1,
            pad_a: 0,
            a_row_lanes: 0,
            held: 0,
            kmajor: false,
        })
    }

    /// **Panel stages from the wave the grid actually puts on an SM (S62
    /// rung `asyncstage`).** A tile's shared memory is budgeted for
    /// `blocks_per_sm` resident blocks; a grid of `blocks` puts
    /// `w = min(blocks_per_sm, ⌈blocks / sms⌉)` on the fullest SM. When
    /// `w < blocks_per_sm` the share a block may take is larger than the tile
    /// was sized for and nothing else will use it, so the panels stream with
    /// `cp.async` through as many stages as `w` blocks' shares hold, up to the
    /// number of panels. Fewer resident blocks is also fewer warps to issue
    /// while one waits at the panel barrier: the stages in flight are what
    /// covers the load latency instead.
    ///
    /// The panel depth is the tile's own when the element/vector copies divide
    /// among the threads (the register-staged arm's depth: only the pipeline
    /// changes), else [`Self::gpu_tile_async`]'s at the larger share.
    /// `blocks_per_sm` becomes `w` (`.minnctapersm`, the wave, the raster agree
    /// with the shared memory allocated). `None` when the wave fills every slot
    /// (`w = blocks_per_sm`: the share is the one [`Self::gpu_tile_async`]
    /// already derives — that rung's decision stands), on a part without
    /// `cp.async`, or when two stages do not fit.
    pub fn gpu_tile_stages(&self, g: GpuTile, blocks: u64, k: u64, elem: &Ty) -> Option<GpuTile> {
        let gpu = self.gpu()?;
        let w = blocks.div_ceil(gpu.sms).clamp(1, g.blocks_per_sm.max(1));
        if w >= g.blocks_per_sm || !gpu.has_mma_sync() {
            return None;
        }
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let (bm, bn, threads) = (g.bm(), g.bn(), g.threads());
        let share = (gpu.smem_per_sm / w)
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let wave = GpuTile {
            blocks_per_sm: w,
            ..g
        };
        let a_units = if g.kmajor { lanes } else { 1 };
        let whole = (bm * g.bk / a_units).is_multiple_of(threads)
            && (g.bk * bn / lanes).is_multiple_of(threads)
            && k.is_multiple_of(g.bk);
        let held = whole
            .then(|| match g.kmajor {
                true => Some(0),
                false => conflict_free_pad(bm, |l| (l % lanes, l / lanes)),
            })
            .flatten()
            .map(|pad_a| GpuTile {
                stages: 2,
                pad_a,
                ..wave
            })
            .filter(|h| h.smem_bytes(e) <= share);
        let geom = match held {
            Some(h) => h,
            None => self.gpu_tile_async(wave, k, elem)?,
        };
        let stages = (share / (geom.stage_elems() * e)).min(k / geom.bk).max(2);
        Some(GpuTile { stages, ..geom })
    }

    /// **The register-staged transposed-A store (S53 rung `pad`).** A thread
    /// loads `lanes` consecutive `k` of one `a` row and stores them as `lanes`
    /// scalar stores, one `k` row of `As` apart. With all `bk / lanes` lanes of a
    /// row in one warp, the lanes sharing a panel row sit `lanes` `k` rows
    /// apart, and no 16-byte-aligned pad separates more than two of them: the
    /// S52 map is a 4-way bank conflict (SASS review). So: the most lanes per
    /// row `L` (a power of two, ≤ `bk / lanes`) for which [`conflict_free_pad`]
    /// finds a pad for lane `l` → (`k` = `lanes·(l mod L)`, row `l / L`), and
    /// the panel still fits the block's share of shared memory. f32 on a
    /// 128-row panel: `L` = 2 (one 32-byte sector per row), pad 4. 4-byte
    /// elements only: the bank model counts one word per element.
    pub fn gpu_tile_sync_a(&self, g: GpuTile, elem: &Ty) -> GpuTile {
        let (Some(gpu), 4) = (self.gpu(), elem_bytes(elem)) else {
            return g;
        };
        let lanes = GPU_VEC_BYTES / 4;
        let share = (gpu.smem_per_sm / g.blocks_per_sm.max(1))
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let kv = g.bk / lanes;
        (0..=kv.max(1).ilog2())
            .rev()
            .map(|p| 1u64 << p)
            .find_map(|l| {
                let pad_a = conflict_free_pad(g.bm(), |x| (lanes * (x % l), x / l))?;
                let geom = GpuTile {
                    pad_a,
                    a_row_lanes: l,
                    ..g
                };
                (geom.smem_bytes(4) <= share).then_some(geom)
            })
            .unwrap_or(g)
    }

    /// **The k-major `a` panel padded (per-site-choice A2)** by `pad` — the
    /// bank model's least conflict-free pad at a panel depth — at the deepest
    /// `bk ≤ g.bk` where the padded panel fits the block's share: a
    /// `cp.async` panel with as many stages as fit, up to the panels (at
    /// least two; a forced count exactly), a register-staged one its one
    /// buffer. A shallower panel still has every thread copy whole 16-byte
    /// units and `bk | k` ([`Self::gpu_tile_async`]). Blocks/SM never change; `g` when nothing
    /// fits. A shallower panel is the price of the pad: the register rule
    /// (`launch`) then weighs it against the transposed panel.
    pub fn gpu_tile_kmajor_pad(
        &self,
        g: GpuTile,
        k: u64,
        elem: &Ty,
        pad: impl Fn(&GpuTile) -> u64,
    ) -> GpuTile {
        let Some(gpu) = self.gpu() else {
            return g;
        };
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let share = (gpu.smem_per_sm / g.blocks_per_sm.max(1))
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let mut bk = g.bk;
        // `g`'s own depth stands as derived; a shallower one must be a legal panel.
        while bk > 0
            && (bk == g.bk
                || (bk >= lanes
                    && k.is_multiple_of(bk)
                    && (g.bm() * bk / lanes).is_multiple_of(g.threads())
                    && (bk * g.bn() / lanes).is_multiple_of(g.threads())))
        {
            let at = GpuTile { bk, ..g };
            let p = GpuTile {
                pad_a: pad(&at),
                ..at
            };
            let stages = match (g.stages, gpu.geom.stages) {
                (s, 0) if s >= 2 => (share / (p.stage_elems() * e)).min(k / bk),
                (s, _) => s,
            };
            let p = GpuTile { stages, ..p };
            if p.smem_bytes(e) <= share && (stages >= 2 || g.stages < 2) {
                return p;
            }
            bk /= 2;
        }
        g
    }
}
