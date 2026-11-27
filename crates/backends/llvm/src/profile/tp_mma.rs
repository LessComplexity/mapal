//! `TargetProfile`: the `mma.sync` tensor tile family.

use super::*;

impl TargetProfile {
    /// **The tensor-core geometry for a `rows × c × k` f32 site on the TF32
    /// face** — [`Self::gpu_tile`]'s rules with an `m16n8` tile as the unit.
    ///
    /// 1. **Warp tile.** Each m16n8 tile holds 4 f32 accumulators per thread;
    ///    the most tiles (a power of two) whose accumulators fit half of
    ///    `regs_per_thread`, square-most: 255 registers ⇒ 16 tiles ⇒ 4 × 4,
    ///    a 64 × 32 warp tile.
    /// 2. **Warps per block** from `regs_per_sm` over twice the accumulators,
    ///    split across [`GPU_RESIDENT_BLOCKS`]; halved while the grid has fewer
    ///    blocks than SMs. Before that, `fill` (S53 rung `warp-fill`) halves the
    ///    warp tile's longer side (`wn` on a tie), down to one m16n8 tile, while
    ///    the grid has fewer warps than [`Gpu::fill_warps`] — overriding `wide`
    ///    on a small site. Warps per block stay those of the unshrunk tile.
    /// 3. **Padding** from the fragment access patterns ([`conflict_free_pad`]).
    /// 4. **Panel depth and stages.** `bk` is the largest power of two (≥ the
    ///    mma's k) for which two stages fit a block's share of shared memory
    ///    and every thread copies whole 16-byte chunks; then as many stages as
    ///    the share holds, up to the number of panels.
    ///
    /// `wide` (S52 rung, the SASS review's rule): instead of "accumulators ≤
    /// half the file", count what a thread actually holds — accumulators, both
    /// buffers of its A and B fragments (`2·(4·mt + 2·nt)`), and
    /// [`MMA_ADDR_REGS`] of addresses — against the whole file. 255 registers
    /// ⇒ 32 tiles, 4 × 8 = a 64 × 64 warp tile (CUTLASS's), 224 registers.
    pub fn gpu_mma_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        if !g.has_mma_sync() {
            return None;
        }
        self.gpu_mma_geom(
            rows,
            c,
            k,
            wide,
            fill,
            MmaFmt::Tf32,
            4,
            4,
            false,
            Depth::Panel,
        )
    }

    /// **f32-tensor-cores `split-tile` (tf32x3): the TF32 tile of a split
    /// `{f32}` site** — [`Self::gpu_mma_tile`]'s rules with each fragment
    /// held twice (its `hi` and `lo` parts) in the register count: the
    /// whole-file tile at 255 registers is `4 × 4`, not `4 × 8`.
    pub fn gpu_mma_split_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
        depth: Depth,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        if !g.has_mma_sync() {
            return None;
        }
        self.gpu_mma_geom_parts(
            rows,
            c,
            k,
            wide,
            fill,
            (MmaFmt::Tf32, 4, 4, false),
            depth,
            Some((2, 2)),
        )
    }

    /// [`Self::gpu_mma_tile`] with the panel depth of f32-tensor-cores
    /// `mma-depth` where `depth` ([`mma_depth_pick`]).
    pub fn gpu_mma_tile_at(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
        depth: Depth,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        if !g.has_mma_sync() {
            return None;
        }
        self.gpu_mma_geom(rows, c, k, wide, fill, MmaFmt::Tf32, 4, 4, false, depth)
    }

    /// **The 16-bit tensor-core geometry** (tensor-cores, ADR-0043 D6): the
    /// rules of [`Self::gpu_mma_tile`] over the part's warp product for
    /// `fmt` (`mma.sync.m16n8k16`: the same `m16n8` tile and fragment
    /// registers, `k` 16), with panels at their storage widths `ea` / `eb`
    /// (2: a narrow operand copied as is by 16-byte `cp.async`, read by
    /// `ldmatrix`; 4: an f32 operand rounded to `fmt` as its fragment is
    /// read). The pads put an `ldmatrix` row (16 bytes) of every lane in its
    /// own 16-byte bank group and the f32 fragment pairs (`LDS.64`) of each
    /// half warp in distinct banks. `None` where the part lists no warp
    /// product of `fmt` with `ldmatrix` and `cp.async`. `bt`: the `b` panel
    /// is `n × k` ([`GpuMmaTile::bt`]). `split`: a split form's parts of
    /// the `a` and `b` fragments a thread holds (narrow1 `(1, 1)`, bf16x3n
    /// `(3, 1)` / `(1, 3)`: the f32 operand's three bf16 parts).
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_mma16_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
        fmt: MmaFmt,
        ea: u64,
        eb: u64,
        bt: bool,
        depth: Depth,
        split: Option<(u64, u64)>,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let op = g.caps.matrix_op(fmt, Home::Regs)?;
        if !(g.caps.ldmatrix && g.caps.cp_async && op.shape.m == MMA_M && op.shape.n == MMA_N)
            || op.shape.k != 32 / fmt.bytes()
        {
            return None;
        }
        self.gpu_mma_geom_parts(rows, c, k, wide, fill, (fmt, ea, eb, bt), depth, split)
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_mma_geom(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
        fmt: MmaFmt,
        ea: u64,
        eb: u64,
        bt: bool,
        depth: Depth,
    ) -> Option<GpuMmaTile> {
        self.gpu_mma_geom_parts(rows, c, k, wide, fill, (fmt, ea, eb, bt), depth, None)
    }

    /// [`Self::gpu_mma_geom`] under a split form (`split`: the parts of
    /// each `a` / `b` fragment a thread holds, f32-tensor-cores) with each
    /// tile's chain result beside its accumulator: ptxas
    /// issues a k-step's products before their adds, so a thread holds a
    /// temporary accumulator a tile (the rtx-5080's narrow1 `4 × 8` tile at
    /// 254 registers spilled 64 B of `cp.async` addresses in the loop).
    #[allow(clippy::too_many_arguments)]
    fn gpu_mma_geom_parts(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
        depth: Depth,
        parts: Option<(u64, u64)>,
    ) -> Option<GpuMmaTile> {
        self.gpu_mma_geom_walk(
            rows,
            c,
            k,
            (wide, fill, false),
            (fmt, ea, eb, bt),
            depth,
            parts,
        )
    }

    /// bf16 tensor-tile `mma-walk`: **the tile a stream-K walk runs** — the
    /// 16-bit tile of [`Self::gpu_mma16_tile`] at its whole warp tile and
    /// warps (neither the `fill` rung's warp-tile halving nor the halving of
    /// a block's warps until the grid has a tile an SM: the walk gives every
    /// SM its blocks whatever the tile count). `None` as there.
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_mma16_walk_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
        depth: Depth,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let op = g.caps.matrix_op(fmt, Home::Regs)?;
        if !(g.caps.ldmatrix && g.caps.cp_async && op.shape.m == MMA_M && op.shape.n == MMA_N)
            || op.shape.k != 32 / fmt.bytes()
        {
            return None;
        }
        self.gpu_mma_geom_walk(
            rows,
            c,
            k,
            (wide, false, true),
            (fmt, ea, eb, bt),
            depth,
            None,
        )
    }

    /// [`Self::gpu_mma_geom_parts`] with `walk`: the block keeps its warps
    /// however few tiles its grid has ([`Self::gpu_mma16_walk_tile`]).
    #[allow(clippy::too_many_arguments)]
    fn gpu_mma_geom_walk(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        (wide, fill, walk): (bool, bool, bool),
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
        depth: Depth,
        parts: Option<(u64, u64)>,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let acc_per_tile = MMA_M * MMA_N / g.warp; // 4
        let split = parts.is_some();
        let (pa, pb) = parts.unwrap_or((1, 1));
        let (mt, nt, regs_est) = if wide {
            // nt = 2·mt keeps the warp tile square (16·mt = 8·nt).
            let held = |mt: u64| {
                let nt = 2 * mt;
                acc_per_tile * mt * nt * (1 + u64::from(split))
                    + 2 * (4 * mt * pa + 2 * nt * pb)
                    + MMA_ADDR_REGS
            };
            let mut mt = 1;
            while held(2 * mt) <= g.regs_per_thread {
                mt *= 2;
            }
            (mt, 2 * mt, held(mt))
        } else {
            let budget = g.regs_per_thread / 2;
            let mut tiles = 1;
            while 2 * tiles * acc_per_tile <= budget {
                tiles *= 2;
            }
            let mut mt = 1;
            while mt * mt < tiles {
                mt *= 2;
            }
            (mt, tiles / mt, 2 * tiles * acc_per_tile)
        };
        let (mut mt, mut nt) = (mt, nt);
        // tensor-cores: a 16-bit tile's warp rows never pass the site's (a
        // skinny site: decode's 32 rows under a 64-row warp tile); TF32
        // keeps S52's rule (no tile).
        while fmt != MmaFmt::Tf32 && mt > 1 && !rows.is_multiple_of(mt * MMA_M) {
            mt /= 2;
        }
        while fmt != MmaFmt::Tf32 && nt > 1 && !c.is_multiple_of(nt * MMA_N) {
            nt /= 2;
        }
        while fill && (rows / (mt * MMA_M)) * (c / (nt * MMA_N)) < g.fill_warps() {
            if nt * MMA_N >= mt * MMA_M && nt > 1 {
                nt /= 2;
            } else if mt > 1 {
                mt /= 2;
            } else {
                break;
            }
        }
        let resident_threads = g.regs_per_sm / regs_est;
        let warps = (resident_threads / g.geom.resident / g.warp).max(1);
        // f32-tensor-cores: a split tile's block takes a power of two of
        // warps (its temporaries' estimate, 160 registers, leaves 6: the
        // block would not tile the site's power-of-two extents).
        let warps = if split { 1 << warps.ilog2() } else { warps };
        // Square-most block tile: warps_y·wm ≈ warps_x·wn.
        let (wm, wn) = (mt * MMA_M, nt * MMA_N);
        let aspect = |wy: u64| {
            let (h, w) = (wy * wm, warps / wy * wn);
            h.max(w) / h.min(w)
        };
        let mut warps_y = (0..=warps.ilog2())
            .map(|p| 1u64 << p)
            .min_by_key(|&wy| aspect(wy))
            .unwrap_or(1);
        let mut warps_x = warps / warps_y;
        let mut halve_x = true;
        while !walk
            && (rows / (warps_y * wm)) * (c / (warps_x * wn)) < g.sms
            && warps_x * warps_y > 1
        {
            if halve_x && warps_x > 1 {
                warps_x /= 2;
            } else if warps_y > 1 {
                warps_y /= 2;
            }
            halve_x = !halve_x;
        }
        self.gpu_mma_block(
            (rows, c, k),
            (mt, nt, warps_x, warps_y),
            regs_est,
            (fmt, ea, eb, bt),
            depth,
            split,
        )
    }

    /// [`Self::gpu_mma_geom_parts`]' tile for one block of `warps_x ×
    /// warps_y` warps: its register residency, then the panels — gapfix
    /// tensor-tile's `mma-fly` ring on a DRAM-bound site, else the deepest
    /// panel at the most blocks an SM whose share fits two stages.
    fn gpu_mma_block(
        &self,
        (rows, c, k): (u64, u64, u64),
        (mt, nt, warps_x, warps_y): (u64, u64, u64, u64),
        regs_est: u64,
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
        depth: Depth,
        split: bool,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let (wm, wn) = (mt * MMA_M, nt * MMA_N);
        let (bm, bn) = (warps_y * wm, warps_x * wn);
        if rows % bm != 0 || c % bn != 0 {
            return None;
        }
        let threads = warps_x * warps_y * g.warp;
        let resident = residency(g, regs_est, threads).max(1);
        // tensor-cores: a 16-bit tile whose register residency leaves each
        // block less shared memory than two panels (a one-warp block of a
        // skinny site: 32 blocks an SM, 2 KB each) holds fewer blocks — the
        // shared memory bounds the residency; TF32 keeps S52's rule (no
        // tile where the register residency's share is too small).
        let floor = if fmt == MmaFmt::Tf32 { resident } else { 1 };
        let panel = (floor..=resident).rev().find_map(|blocks_per_sm| {
            self.gpu_mma_panels(
                k,
                (mt, nt, warps_x, warps_y),
                blocks_per_sm,
                threads,
                (fmt, ea, eb, bt),
                depth,
            )
        })?;
        // gapfix tensor-tile (C3) `mma-fly`: a DRAM-bound site's ring keeps
        // the part's in-flight bytes past its wait ([`Self::gpu_mma_fly`]),
        // over the residencies the panel rule offers (the same tile).
        match (depth, split) {
            (Depth::Fly { walk }, false) => Some(
                self.gpu_mma_fly(
                    (rows, c, k),
                    (mt, nt, warps_x, warps_y),
                    (floor, resident),
                    threads,
                    (fmt, ea, eb, bt),
                    walk,
                    &panel,
                )
                .unwrap_or(panel),
            ),
            _ => Some(panel),
        }
    }

    /// gapfix tensor-tile (C3) `mma-fly`: **the ring of a tensor tile whose
    /// panel loads outlast the products that cover them** — `None` (the S52
    /// panel `base`) unless
    ///
    /// 1. the tile reads its fragments as stored (`ldmatrix` of the
    ///    instruction's own format: both panels at the 16-bit format's
    ///    width — a TF32 tile, or an f32 panel of a 16-bit product, converts
    ///    every element as it reads it),
    /// 2. an operand's bytes pass what the L2 keeps ([`Gpu::l2_keep_bytes`]:
    ///    it streams from DRAM on every call), and
    /// 3. the S52 panel's load is exposed but the SM is not DRAM-bound: its
    ///    blocks' DRAM bytes a panel (`blocks_per_sm · bk ·` the streamed
    ///    widths, over the blocks of a wave that read the panel together —
    ///    under the walk the site takes, `walk`: row-major, or grouped by
    ///    [`Gpu::raster`]'s `h` block-rows, each `b` panel read by the `h`
    ///    blocks of its column and each `a` panel by the `wave / h` of its
    ///    row) at the SM's share of the DRAM rate take `T` ≤ the panel's
    ///    products at the SM's matrix rate, `t`, and `t` < `T` + the loaded
    ///    latency (`lat_ns`) — the next panel, issued as this one's products
    ///    start, has not landed when they end (`wait_group 0`).
    ///
    /// Then every residency the panel rule offers (`floor..=resident` blocks
    /// an SM, each block's share of shared memory) offers its panels
    /// ([`Self::gpu_mma_panel_cands`]) whose products last at least the
    /// loaded latency a panel, and [`mma_fly_pick`] takes the ring.
    /// Measured (`gapfix-tensor-tile.md`): the RTX 5080's 128 × 28672 × 4096
    /// bf16 projection (`T` 1546 ≤ `t` 1672 < 2119 ns; the ring's `bk` 16
    /// panel 836 ns of products) won the ring 0.400 → 0.344 ms; its 128 ×
    /// 4096 × 14336 one (a 64 × 64 tile: the ring's `bk` 32 panel 418 ns,
    /// under the 573 ns latency) lost it (+4.8 %); the RTX 4090's 128 × 28672
    /// (`T` 2139 > `t` 1626: DRAM-bound) lost it (+5.8 %), the A100's (2378 >
    /// 1470) was null; every TF32 ring lost (+4–6 % on the 5080, the 4090,
    /// the A100). The walk (gap-fixes merge): the 5080's 2048 × 28672 × 4096
    /// bf16 projection, grouped by C11's gate (`h` 16) but priced row-major
    /// (`T` 1442 ≤ `t` 1672 < `T` + 573), took the ring and lost it — the
    /// S52 `bk` 32 × 2 3.99 ms, the ring `bk` 16 × 4 4.52 —: its grouped wave
    /// reads each `b` / `a` panel with 16 / 10 blocks (`T` 233 ns), no load
    /// exposed.
    #[allow(clippy::too_many_arguments)]
    fn gpu_mma_fly(
        &self,
        (rows, c, k): (u64, u64, u64),
        tile: (u64, u64, u64, u64),
        (floor, resident): (u64, u64),
        threads: u64,
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
        walk: Option<(bool, u64)>,
        base: &GpuMmaTile,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let (mt, nt, warps_x, warps_y) = tile;
        let (bm, bn) = (warps_y * mt * MMA_M, warps_x * nt * MMA_N);
        let narrow = fmt != MmaFmt::Tf32 && ea == fmt.bytes() && eb == fmt.bytes();
        let keep = g.l2_keep_bytes();
        let (a_dram, b_dram) = (rows * k * ea > keep, k * c * eb > keep);
        if !narrow || !(a_dram || b_dram) || bm == 0 || bn == 0 {
            return None;
        }
        let rate = g
            .caps
            .matrix_op(fmt, Home::Regs)
            .map_or(0, |op| g.matrix_flops_per_ns(&op));
        let d = g.plan_cost();
        let bps = base.blocks_per_sm.max(1);
        // A streamed panel is read from DRAM once for the blocks of a wave
        // that read it together. Row-major: a `b` panel by the wave's rows
        // of the grid (`min(tiles, wave) / gx`, at least one), an `a` panel by its row's
        // blocks in the wave (`min(gx, wave)`). The grouped walk the site
        // takes (`walk`: `l2-gate` and the site's element bytes, decided by
        // [`Gpu::raster`] from this tile's block and residency): `b` by its
        // `h` block-rows, `a` by the wave's `wave / h` columns — bytes ·1000
        // a block-panel.
        let (gx, tiles, wave) = (c / bn, (rows / bm) * (c / bn), g.sms * bps);
        let h = walk.map_or(1, |(gate, e)| {
            g.raster(bm, bn, wave, rows, c, k, e, gate, eb, rate)
        });
        let (b_num, b_den, a_den) = match h > 1 {
            true => (1, h, (wave / h).min(gx)),
            false => (gx, tiles.min(wave).max(gx), gx.min(wave)),
        };
        let b_part = u64::from(b_dram) * bn * eb * 1000 * b_num / b_den.max(1);
        let a_part = u64::from(a_dram) * bm * ea * 1000 / a_den.max(1);
        // ns · 1000 in integers: the panel's products at the SM's rate, its
        // DRAM bytes at the SM's share.
        let t = bps * 2 * bm * bn * base.bk * g.sms * 1000 / rate.max(1);
        let tt = bps * base.bk * (a_part + b_part) * g.sms / d.dram_bytes_per_ns.max(1);
        if rate == 0 || tt > t || t >= tt + d.lat_ns * 1000 {
            return None;
        }
        let cands: Vec<GpuMmaTile> = (floor..=resident)
            .flat_map(|bps| {
                self.gpu_mma_panel_cands(k, tile, bps, threads, (fmt, ea, eb, bt))
                    .unwrap_or_default()
            })
            .filter(|t| g.geom.bk == 0 || t.bk == g.geom.bk)
            // Each panel's products (its blocks sharing the SM's pipe) last at
            // least the loaded latency: the ring's one wait and barrier a panel
            // stay inside a latency of products.
            .filter(|t| {
                t.blocks_per_sm * 2 * bm * bn * t.bk * g.sms * 1000 / rate.max(1) >= d.lat_ns * 1000
            })
            .map(|t| match g.geom.stages {
                0 => t,
                s => GpuMmaTile {
                    stages: s.min(t.stages).max(2),
                    ..t
                },
            })
            .collect();
        let dram = (a_dram.then_some(ea), b_dram.then_some(eb));
        mma_fly_pick(&cands, dram, g.stream_fly_per_sm()).copied()
    }

    /// [`Self::gpu_mma_geom`]'s panel depth, pads and stages for a block
    /// tile at `blocks_per_sm`: `None` where no whole panel's two stages fit
    /// the block's share of shared memory.
    fn gpu_mma_panels(
        &self,
        k: u64,
        tile: (u64, u64, u64, u64),
        blocks_per_sm: u64,
        threads: u64,
        fmt: (MmaFmt, u64, u64, bool),
        depth: Depth,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        let cands = self.gpu_mma_panel_cands(k, tile, blocks_per_sm, threads, fmt)?;
        // The panel depth: a forced `--geom=bk=` where the site takes it;
        // f32-tensor-cores `mma-depth` ([`mma_depth_pick`]); else the deepest
        // panel two stages fit (S52), as many stages as the share holds.
        let forced = cands.iter().find(|c| g.geom.bk > 0 && c.bk == g.geom.bk);
        let geom = *match (forced, depth) {
            (Some(c), _) => c,
            (None, Depth::InFlight) => mma_depth_pick(&cands)?,
            (None, _) => cands.last()?,
        };
        let stages = match g.geom.stages {
            0 => geom.stages,
            s => s.min(geom.stages).max(2),
        };
        Some(GpuMmaTile { stages, ..geom })
    }

    /// [`Self::gpu_mma_panels`]' candidates: each panel depth `bk` (a power
    /// of two from the instruction's `k`, dividing `k`) whose two stages fit
    /// the block's share at `blocks_per_sm` and copy in whole 16-byte
    /// chunks, with as many stages as the share holds; `None` where a pad
    /// has no conflict-free form.
    fn gpu_mma_panel_cands(
        &self,
        k: u64,
        (mt, nt, warps_x, warps_y): (u64, u64, u64, u64),
        blocks_per_sm: u64,
        threads: u64,
        (fmt, ea, eb, bt): (MmaFmt, u64, u64, bool),
    ) -> Option<Vec<GpuMmaTile>> {
        let g = self.gpu()?;
        let mma_k = 32 / fmt.bytes();
        let (bm, bn) = (warps_y * mt * MMA_M, warps_x * nt * MMA_N);
        let share = (g.smem_per_sm / blocks_per_sm)
            .saturating_sub(g.smem_reserved_per_block)
            .min(g.smem_per_block);

        // Fragment patterns of m16n8k8 (PTX ISA): lane = 4·group + t.
        // A (row-major): row = group, col = t.  B (k × n): row = t, col = group.
        // 16-bit panels: `ldmatrix` rows of 16 bytes ([`ldmatrix_pad`]); an
        // f32 panel of a 16-bit product: `a` pairs (row g, k 2t..2t+1) as one
        // `LDS.64`, `b` words (k 2t + j, n g).
        // An `n × k` panel (`bt`) is read like `a`'s.
        let pad_kn = match (fmt, eb) {
            (MmaFmt::Tf32, _) => conflict_free_pad(bn, |l| (l % 4, l / 4)),
            (_, 4) => conflict_free_pad(bn, |l| (2 * (l % 4), l / 4)),
            _ => Some(ldmatrix_pad(bn)),
        };
        let mut cands: Vec<GpuMmaTile> = Vec::new();
        let mut bk = mma_k;
        while bk <= k {
            let pad_a = match (fmt, ea) {
                (MmaFmt::Tf32, _) => conflict_free_pad(bk, |l| (l / 4, l % 4))?,
                (_, 4) => pair_pad(bk)?,
                _ => ldmatrix_pad(bk),
            };
            let pad_b = match (bt, eb) {
                (false, _) => pad_kn?,
                (true, 4) => pair_pad(bk)?,
                (true, _) => ldmatrix_pad(bk),
            };
            let geom = GpuMmaTile {
                mt,
                nt,
                warps_x,
                warps_y,
                bk,
                stages: 2,
                pad_a,
                pad_b,
                blocks_per_sm,
                fmt,
                ea,
                eb,
                bt,
                split: None,
            };
            // Whole 16-byte copies per thread, each panel row 16-byte aligned.
            let (la, lb) = (GPU_VEC_BYTES / ea, GPU_VEC_BYTES / eb);
            let whole = (bm * bk / la).is_multiple_of(threads)
                && (bk * bn / lb).is_multiple_of(threads)
                && (geom.lda() * ea).is_multiple_of(GPU_VEC_BYTES)
                && (geom.ldb() * eb).is_multiple_of(GPU_VEC_BYTES);
            if geom.smem_bytes() > share {
                break;
            }
            if k.is_multiple_of(bk) && whole {
                let stages = (share / geom.stage_bytes()).min(k / bk).max(2);
                cands.push(GpuMmaTile { stages, ..geom });
            }
            bk *= 2;
        }
        Some(cands)
    }
}
