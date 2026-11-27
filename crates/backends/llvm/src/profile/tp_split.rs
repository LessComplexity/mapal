//! `TargetProfile`: tile geometry selection, split-K and stream-K pricing.

use super::*;

impl TargetProfile {
    /// **The GPU tile geometry for a `rows × c × k` site — the same three
    /// budgets the CPU rungs read, at their GPU locations.** `None` when the
    /// site does not divide into any geometry this derives; the emitter then
    /// takes the one-thread-per-cell kernel, which is slower and never wrong.
    ///
    /// | CPU rule | GPU reading here |
    /// | --- | --- |
    /// | [`Self::tile_i`]: half the vector file holds accumulators | half of `regs_per_thread` holds the `tm × tn` register tile |
    /// | [`Self::tile_kc`]: half of L2 holds one packed panel | a block's share of `smem_per_sm` holds the `bk`-deep A and B panels |
    /// | the parallel driver's slice count | blocks: at least one per SM |
    ///
    /// Steps, each a fact or a stated policy:
    /// 1. **Register tile.** The largest square power-of-two `t`, at least one
    ///    vector wide, with `t² · regs(elem) ≤ regs_per_thread / 2` (the CPU's
    ///    half-the-file headroom). f32 on 255 registers: `8 × 8`.
    /// 2. **Threads per block.** Registers per thread are estimated at twice
    ///    the accumulators; `regs_per_sm` over that is the resident-thread
    ///    budget, split across [`GPU_RESIDENT_BLOCKS`] blocks so one computes
    ///    while another waits at the barrier. 65 536 / 128 / 2 = 256 = 16 × 16.
    /// 3. **Fill the schedulers** (`fill`, S53 rung `warp-fill`). While the
    ///    grid has fewer warps than [`Gpu::fill_warps`], halve the register
    ///    tile's longer side (`tn` on a tie), down to one vector. Grid warps
    ///    are `rows·c / (tm·tn·warp)`, which no thread-grid split changes;
    ///    threads per block stay those of the unshrunk tile (step 2).
    /// 4. **Fill the machine.** While the grid has fewer blocks than SMs, halve
    ///    the thread grid (x then y), down to one warp — a block left idle is a
    ///    whole SM idle, which no tile shape wins back.
    /// 5. **Panel depth.** Blocks per SM = the tightest of registers, threads
    ///    and the block-slot limit; `bk` is the largest power of two whose two
    ///    panels fit one block's share of shared memory, dividing `k`, at least
    ///    one vector deep, and loadable in whole vectors by every thread.
    ///
    /// `wide` (S53 rung `wide-fma`, the TF32 `wide` rule on the FMA tile):
    /// step 1 counts what a thread holds ([`fma_held_regs`]) against the whole
    /// file — the largest `tm × tn`, `tn ∈ {tm, 2·tm}`, square first: 255 ⇒
    /// 8 × 16 (128 + 48 + 32 = 208). Step 2: threads = `regs_per_sm` over that
    /// count over [`GPU_RESIDENT_BLOCKS`], rounded down to a power of two
    /// (65 536 / 208 / 2 ⇒ 128), split square-most on the *block* tile
    /// (`ty·tm ≈ tx·tn` ⇒ 16 × 8 ⇒ 128 × 128). Steps 3 and 4 never shrink it:
    /// a site with fewer such blocks than SMs — or, under `fill`, fewer warps
    /// than [`Gpu::fill_warps`] — takes the S52 geometry instead, which they
    /// then fill.
    ///
    /// The S62 geometry rungs ([`Self::gpu_tile_with`]) are all off here: this
    /// is the geometry the rung table's fill key reads (`gpu.rs:launch`).
    pub fn gpu_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        elem: &Ty,
        wide: bool,
        fill: bool,
    ) -> Option<GpuTile> {
        // tile-efficiency T2: no site layout here, so no k-major fragments.
        let r = crate::GpuRungs {
            fill,
            kmajor_a: false,
            ..crate::GpuRungs::default()
        };
        // The rung table's rows were measured on grids this geometry keys at
        // the seed row (`resident` 2, nothing forced): a `--geom=` arm or a
        // searched table row moves the launch, never which rows apply.
        self.with_geom(GEOM)
            .gpu_tile_with(rows, c, k, elem, wide, r)
    }

    /// This profile with its GPU part's geometry row replaced by `geom`.
    pub(super) fn with_geom(&self, geom: Geom) -> TargetProfile {
        let mut p = *self;
        if let Machine::Gpu(g) = &mut p.machine {
            g.geom = geom;
        }
        p
    }

    /// [`Self::gpu_tile`] under the rungs `r` reads: `fill` (`warp-fill`) and
    /// the S62 ordered-geometry rungs. Each re-partitions the independent
    /// output domain (`rows × c`, every cell's `k` fold kept whole and in
    /// order on one thread, `TilePlan`) or moves a panel boundary — never the
    /// order a cell folds:
    ///
    /// - `fill-rows` (step 3): the fill rule may shrink `tm` to 1, not one
    ///   vector — the vector floor was the emitter's `<lanes x t>` fragment
    ///   load, not a fact. `tn` keeps it (the output's vector store). Only
    ///   where the vector-floor tile has no geometry on these rows: a scalar
    ///   `a` row costs a shared load per FMA row, which pays against the
    ///   one-thread-per-cell kernel, not against a vector tile (S62 v1 on the
    ///   5090, 4090, H100 PCIe and 3090: 1.15–5.7× over the former in every
    ///   cell; against the latter 0.53–0.96× in 12 cells, ±2% in 5, 1.18–1.40× in 2).
    /// - `row-fit` (after step 4): a thread grid taller than the site
    ///   (`ty·tm > rows`) halves `ty` and doubles `tx`, thread count kept.
    /// - `wave-gate`: a row-fitted grid with fewer blocks than SMs is `None` —
    ///   an SM without a block is idle for the whole launch (S59: 64 and 32
    ///   blocks on 170 SMs measured 1.37–1.47× the one-thread-per-cell time).
    /// - `grid-share` (step 5): one SM holds at most `⌈blocks / sms⌉` of this
    ///   grid's blocks, so the panel share is sized against that residency.
    pub fn gpu_tile_with(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        elem: &Ty,
        wide: bool,
        r: crate::GpuRungs,
    ) -> Option<GpuTile> {
        self.gpu_tile_split(rows, c, k, elem, wide, r, 1)
    }

    /// [`Self::gpu_tile_with`] for a grid whose fold is cut in `split`
    /// partitions (site-geometry `fill-wave`): `grid-share` counts the
    /// `blocks · split` the launch puts on the part. 1 = the unsplit grid.
    ///
    /// ADR-0034 D1 (site-geometry B4): the part's forced tile knobs
    /// ([`Geom::tm`], `tn`, `warps`, `bk`) replace the step they name — the
    /// register tile (steps 1 and 3), the threads (steps 2 and 4), the panel
    /// depth (step 5) — and every other step derives as before; a forced tile
    /// the site cannot take falls back to the derived one.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_tile_split(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        elem: &Ty,
        wide: bool,
        r: crate::GpuRungs,
        split: u64,
    ) -> Option<GpuTile> {
        let q = self.gpu()?.geom;
        self.tile_split_at(rows, c, k, elem, wide, r, split)
            .or_else(|| {
                (q.tm + q.tn + q.warps + q.bk > 0)
                    .then(|| {
                        let derived = Geom {
                            tm: 0,
                            tn: 0,
                            warps: 0,
                            bk: 0,
                            ..q
                        };
                        self.with_geom(derived)
                            .tile_split_at(rows, c, k, elem, wide, r, split)
                    })
                    .flatten()
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn tile_split_at(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        elem: &Ty,
        wide: bool,
        r: crate::GpuRungs,
        split: u64,
    ) -> Option<GpuTile> {
        let fill = r.fill;
        if r.fill_rows && fill && !wide {
            let floor = crate::GpuRungs {
                fill_rows: false,
                ..r
            };
            if let Some(t) = self.gpu_tile_split(rows, c, k, elem, wide, floor, split) {
                return Some(t);
            }
        }
        let g = self.gpu()?;
        let q = g.geom;
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);

        // tile-efficiency T2: a k-major `a` fragment holds `lanes` steps of
        // its `tm` rows (`lanes − 1` more registers a row).
        let held_at = |m: u64, n: u64| {
            fma_held_regs(m, n, e) + u64::from(r.kmajor_a) * (lanes - 1) * m * e.div_ceil(4)
        };
        let (mut tm, mut tn, mut regs_est) = if wide {
            let (mut tm, mut tn) = (lanes, lanes);
            loop {
                let (m, n) = if tn == tm { (tm, 2 * tm) } else { (2 * tm, tn) };
                if held_at(m, n) > g.regs_per_thread {
                    break;
                }
                (tm, tn) = (m, n);
            }
            (tm, tn, held_at(tm, tn))
        } else {
            let (t, regs_est) = square_register_tile(g, e);
            (t, t, regs_est)
        };
        let mut held = if wide { regs_est } else { 0 };
        // A forced register tile: a thread holds its accumulators twice over
        // (the derived budget), or what it holds when that outgrows the file.
        let forced_tile = q.tm + q.tn > 0;
        if forced_tile {
            tm = if q.tm > 0 { q.tm } else { tm };
            tn = if q.tn > 0 { q.tn } else { tn };
            if !(tm.is_power_of_two() && tn.is_power_of_two()) || !tn.is_multiple_of(lanes) {
                return None;
            }
            let twice = 2 * tm * tn * e.div_ceil(4);
            held = if wide || twice > g.regs_per_thread {
                held_at(tm, tn)
            } else {
                0
            };
            regs_est = if held > 0 { held } else { twice };
        }
        // tile-efficiency T1: a split grid that keeps the full tile fills the
        // machine along `k` — no shrink of the tile or its threads, and the
        // grid's fill counts its `split` partitions.
        let fill_k = r.full_tile && split > 1;
        let parts = if fill_k { split } else { 1 };
        let under_warps =
            |tm: u64, tn: u64| fill && !fill_k && (rows / tm) * (c / tn) / g.warp < g.fill_warps();
        let tm_floor = if r.fill_rows { 1 } else { lanes };
        while !wide && !forced_tile && under_warps(tm, tn) {
            if tn >= tm && tn > lanes {
                tn /= 2;
            } else if tm > tm_floor {
                tm /= 2;
            } else {
                break;
            }
        }

        let threads = match q.warps {
            0 => block_threads(g, regs_est),
            w if (w * g.warp).is_power_of_two() && w * g.warp <= g.max_threads_per_block => {
                w * g.warp
            }
            _ => return None,
        };
        // The block tile square-most (`ty·tm ≈ tx·tn`) for the wide or a forced
        // tile, else the thread grid.
        let (mut tx, mut ty) = if wide || forced_tile || q.warps > 0 {
            let threads = 1 << threads.max(1).ilog2();
            let aspect = |ty: u64| {
                let (h, w) = (ty * tm, threads / ty * tn);
                h.max(w) / h.min(w)
            };
            let ty = (0..=threads.ilog2())
                .map(|p| 1u64 << p)
                .min_by_key(|&ty| aspect(ty))
                .unwrap_or(1);
            (threads / ty, ty)
        } else {
            // Square-most power-of-two split of the thread grid, x the wider.
            let mut tx = 1;
            while tx * tx < threads {
                tx *= 2;
            }
            (tx, (threads / tx).max(1))
        };

        // The wide tile buys reuse with threads; a site that cannot give every
        // SM a block at its full grid keeps the S52 tile, which step 3 then
        // fills (S53: halving the wide grid at 1024² measured 0.70× on Ada and
        // Blackwell — half the warps per block over the same tiles).
        // large-gemm `streamk-fill`: nor one whose grid, at the tile's own
        // threads, gives the part fewer warps than its fill target — half
        // the S52 tile's warps a block, it neither streams nor splits there
        // (256 × 6144 × 8192: 96 whole-file blocks of 4 warps, 0.57 of a
        // wave; the S52 tile's 96 blocks of 8 warps stream-K over the part).
        let wide_short = r.streamk_fill
            && !fill_k
            && (rows / (ty * tm)) * (c / (tx * tn)) * threads.div_ceil(g.warp) < g.fill_warps();
        if wide
            && ((rows / (ty * tm)) * (c / (tx * tn)) * parts < g.sms
                || under_warps(tm, tn)
                || wide_short)
        {
            return self.gpu_tile_split(rows, c, k, elem, false, r, split);
        }
        // site-geometry `fit-first` (R1): a grid short only because its thread
        // grid outgrows the rows — reshaped (`row-fit`'s move, threads kept),
        // it gives every SM a block — reshapes instead of losing threads.
        if r.fit_first && rows > 0 && ty * tm > rows {
            let (mut fy, mut fx) = (ty, tx);
            while fy > 1 && fy * tm > rows {
                fy /= 2;
                fx *= 2;
            }
            let divides = rows.is_multiple_of(fy * tm) && c.is_multiple_of(fx * tn);
            if divides && (rows / (fy * tm)) * (c / (fx * tn)) * parts >= g.sms {
                (ty, tx) = (fy, fx);
            }
        }
        let mut halve_x = q.warps == 0;
        while !fill_k
            && q.warps == 0
            && (rows / (ty * tm)) * (c / (tx * tn)) < g.sms
            && tx * ty > g.warp
        {
            if halve_x && tx > 1 {
                tx /= 2;
            } else if ty > 1 {
                ty /= 2;
            }
            halve_x = !halve_x;
        }
        // S62 `row-fit`: a successful geometry already has `bm ≤ rows`, so
        // this moves only a grid that would otherwise be rejected below.
        let fit = r.row_fit && rows > 0 && ty > 1 && ty * tm > rows;
        while fit && ty > 1 && ty * tm > rows {
            ty /= 2;
            tx *= 2;
        }
        let blocks = (rows / (ty * tm)) * (c / (tx * tn));
        // hardware-properties `wave-gate`: a row-fitted grid whose waves leave
        // SMs idle ([`wave_full`]) yields to the cell tile / one thread a cell.
        if fit && r.wave_gate && !wave_full(blocks * parts, g.sms) {
            return None;
        }

        let geom = |bk| GpuTile {
            tm,
            tn,
            tx,
            ty,
            bk,
            blocks_per_sm: 0,
            stages: 1,
            pad_a: 0,
            a_row_lanes: 0,
            held,
            kmajor: false,
        };
        let (bm, bn, threads) = (geom(0).bm(), geom(0).bn(), tx * ty);
        if rows % bm != 0 || c % bn != 0 {
            return None;
        }
        let grid = if r.grid_share {
            (blocks * split.max(1)).div_ceil(g.sms)
        } else {
            u64::MAX
        };
        // per-site-choice A4: a k-major thread holds `lanes − 1` more fragment
        // registers a row; where that outgrows the budgeted tile's estimate,
        // residency reads what it holds (a held tile has them in `held_at`).
        // Under `bank-plan` (D1); off, the pre-sprint estimate.
        let regs_res = match (r.kmajor_a && r.bank_plan, held) {
            (true, 0) => regs_est.max(held_at(tm, tn)),
            _ => regs_est,
        };
        let blocks_per_sm = residency(g, regs_res, threads).min(grid).max(1);
        let share = (g.smem_per_sm / blocks_per_sm)
            .saturating_sub(g.smem_reserved_per_block)
            .min(g.smem_per_block);

        let loads_whole = |bk: u64| {
            (bm * bk / lanes).is_multiple_of(threads) && (bk * bn / lanes).is_multiple_of(threads)
        };
        let mut bk = lanes;
        let mut best = None;
        while bk <= k && (bm + bn) * bk * e <= share {
            if k.is_multiple_of(bk) && loads_whole(bk) && (q.bk == 0 || bk == q.bk) {
                best = Some(bk);
            }
            bk *= 2;
        }
        Some(GpuTile {
            blocks_per_sm,
            ..geom(best?)
        })
    }

    /// large-gemm `k-fill`: the side of the S52 square register tile
    /// ([`Self::gpu_tile`] step 1) at `elem` — the full tile an under-filled
    /// site keeps; `None` off a GPU.
    pub fn fma_tile_side(&self, elem: &Ty) -> Option<u64> {
        Some(square_register_tile(self.gpu()?, elem_bytes(elem)).0)
    }

    /// **The blocks of tile `g` one SM can hold** at its registers and
    /// threads — [`Self::gpu_tile_with`] step 5 without `grid-share`'s cap at
    /// the unsplit grid (site-geometry `fill-wave`). The panel share follows
    /// the residency, so shared memory does not bound it here.
    pub fn tile_capacity(&self, g: &GpuTile, elem: &Ty) -> u64 {
        let Some(gpu) = self.gpu() else {
            return g.blocks_per_sm;
        };
        let regs = if g.held > 0 {
            g.held
        } else {
            square_register_tile(gpu, elem_bytes(elem)).1
        };
        residency(gpu, regs, g.threads().max(1))
            .max(g.blocks_per_sm)
            .max(1)
    }

    /// **S62/S63 split-K sizing (ADR-0040 D4, gap map M1): how many contiguous
    /// `k` partitions a tile site's fold is cut into**; 1 = the ordered kernel.
    /// `blocks` is the site's grid, `wave` the blocks the part holds at that
    /// geometry at once (`Launch::wave`), `panels` the fold's whole panels
    /// (`k / GpuTile::bk`; `k` for one thread per cell), `cells` the output
    /// extent, `elem` its bytes; `workspace` when the partials are materialized
    /// (`splitk`) rather than added into the output (`splitk-atomic`).
    ///
    /// Every output cell is independent (`TilePlan`), so the grid already is
    /// every block the output domain gives; when it leaves resident slots
    /// empty, the fold is the only parallel axis left. The largest `S` that
    /// divides the panels (each partition's affine reads stay inside the
    /// bounds the site proved, no remainder path), keeps the split grid inside
    /// one wave (`blocks·S ≤ wave`: no tail wave, G6) and, for a workspace,
    /// keeps the partials the reduce reads back in L2 (`S·cells·elem ≤ l2`,
    /// the S62 derivation's invariant). Only device facts and site extents
    /// enter.
    pub fn split_count(
        &self,
        blocks: u64,
        wave: u64,
        panels: u64,
        cells: u64,
        elem: u64,
        workspace: bool,
    ) -> u64 {
        let Some(g) = self.gpu() else {
            return 1;
        };
        (1..=panels.min(wave / blocks.max(1)))
            .rev()
            .find(|&d| panels.is_multiple_of(d) && (!workspace || d * cells * elem <= g.l2_bytes))
            .unwrap_or(1)
    }

    /// **site-geometry `fill-wave`: the fewest partitions that fill the
    /// part** — the smallest `S` dividing the panels for which the split
    /// grid's `blocks · S · warps` reaches [`Gpu::fill_warps`], inside `cap`
    /// blocks (the wave the part holds at the tile's own registers and
    /// threads, [`Self::tile_capacity`]) and, for a workspace, the L2 (as
    /// [`Self::split_count`]). A fold too short to fill takes the largest such
    /// `S` — [`Self::split_count`] at `cap`.
    #[allow(clippy::too_many_arguments)]
    pub fn split_fill(
        &self,
        blocks: u64,
        warps: u64,
        cap: u64,
        panels: u64,
        cells: u64,
        elem: u64,
        workspace: bool,
    ) -> u64 {
        let Some(g) = self.gpu() else {
            return 1;
        };
        let most = self.split_count(blocks, cap, panels, cells, elem, workspace);
        (1..=most)
            .find(|&d| panels.is_multiple_of(d) && blocks * d * warps >= g.fill_warps())
            .unwrap_or(most)
    }

    /// **gaps-disasm C5 `split-price`: the atomic split of a DRAM-streaming
    /// under-filled site, priced** — over the counts dividing `panels` inside
    /// `cap` blocks, the cheapest of
    ///
    /// `bytes / dram_rate · spread(blocks · S) + S · cells · split_cell_ps`
    ///
    /// (the stream at the busiest SM's share — [`crate::rules::grid_spread_pm`]
    /// — and every partition's partial cells at the part's measured
    /// [`DeviceCost::split_cell_ps`]); the fewer partitions on a tie. The fill
    /// rule ([`Self::split_fill`]) takes the fewest partitions that give the
    /// part `fill_warps` and prices neither: on the rtx-5080 its S = 56 for
    /// the 16 × 2048 × 7168 cell ran 88.1 µs against the priced S = 14's 79.9
    /// (cuBLAS 80.9; `gaps-disasm.md` C5). `None` where the part measured no
    /// `split_cell_ps` or no DRAM rate: the fill rule stands.
    pub fn split_priced(
        &self,
        blocks: u64,
        cap: u64,
        panels: u64,
        cells: u64,
        bytes: u64,
    ) -> Option<u64> {
        let g = self.gpu()?;
        let d = g.plan_cost();
        if d.split_cell_ps == 0 || d.dram_bytes_per_ns == 0 {
            return None;
        }
        let stream = bytes as f64 / d.dram_bytes_per_ns as f64;
        (1..=panels.min(cap / blocks.max(1)).max(1))
            .filter(|s| panels.is_multiple_of(*s))
            .map(|s| {
                let spread = crate::rules::grid_spread_pm(blocks * s, g.sms) as f64 / 1000.0;
                (
                    s,
                    stream * spread + (s * cells) as f64 * d.split_cell_ps as f64 / 1000.0,
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(s, _)| s)
    }

    /// **stream-fly: the fewest `k` partitions that keep a streaming site's
    /// DRAM busy** — the smallest `S` dividing `panels` at which the split
    /// grid (`blocks · S` blocks, each keeping `fly` bytes in flight) puts a
    /// block on every SM and keeps at least [`Gpu::stream_fly_per_sm`] bytes
    /// in flight on each, inside `cap` blocks (the wave the kernel holds),
    /// and, for a workspace, inside the L2 ([`Self::split_count`]). A grid
    /// that cannot reach it inside the wave takes the largest such `S`. Only
    /// the part's SMs, its measured in-flight knee and the site's geometry
    /// enter: a part whose DRAM needs more bytes in flight an SM (the H100
    /// SXM's 32 KiB against the RTX 5080's 8) cuts the fold finer.
    #[allow(clippy::too_many_arguments)]
    pub fn split_stream(
        &self,
        blocks: u64,
        fly: u64,
        cap: u64,
        panels: u64,
        cells: u64,
        elem: u64,
        workspace: bool,
    ) -> u64 {
        let Some(g) = self.gpu() else {
            return 1;
        };
        let most = self.split_count(blocks, cap, panels, cells, elem, workspace);
        let need = g.stream_fly_per_sm() * g.sms;
        (1..=most)
            .find(|&d| panels.is_multiple_of(d) && blocks * d >= g.sms && blocks * d * fly >= need)
            .unwrap_or(most)
    }
}
