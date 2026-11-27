//! The launch geometry: `Arithmetic`, `Launch`, `Combine` and the split-launch rules.

use super::*;

/// The arithmetic the selected kernel actually emits, distinct from the
/// module's permission to contract or use TF32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arithmetic {
    SeparateMulAdd,
    OrderedFma,
    Tf32Mma,
    Tf32Wgmma,
    /// tensor-cores: `mma.sync.m16n8k16` over a 16-bit member (ADR-0043 D4).
    HalfMma,
    /// wgmma-16bit: `wgmma.m64nNk16` over a 16-bit member — the same
    /// products and per-instruction sums as [`Self::HalfMma`]'s `k16` steps.
    HalfWgmma,
    /// f32-tensor-cores `split-tile`: a `{f32}` product on a matrix
    /// instruction by a split form (`mapal_ir::split`), the default face's
    /// bound (ADR-0040 D2).
    SplitMma,
}

impl Arithmetic {
    pub(crate) fn scalar(site: &TileSite, contract: bool, free: bool, pairs: bool) -> Self {
        // fold-algebra: a tile site's step is `acc + a·b` with the product read
        // once — a `step_algebra` FMA pair by construction — and the host fuses
        // every such pair under `contract` (`FnEmit::emit_arith`, the same
        // emitter the device cells use), so the device fuses whatever B's lane
        // stride or radix form: admission is the algebra, not the geometry.
        // OFF arm (`--no-fma-pairs`): the host's generic steps stay separate,
        // so a strided-lane site does too; a radix site fuses where the order
        // is free (recurrence-precision `radix-tile`: its `_c` oracle replays
        // separate steps).
        let fused = pairs
            || match site.radix {
                Some(_) => free,
                None => site.b.clane == 1,
            };
        if contract && fused {
            Self::OrderedFma
        } else {
            Self::SeparateMulAdd
        }
    }

    pub(crate) fn face(self) -> Face {
        match self {
            Self::SeparateMulAdd => Face::Exact,
            Self::OrderedFma => Face::Contract,
            Self::Tf32Mma | Self::Tf32Wgmma | Self::HalfMma | Self::HalfWgmma | Self::SplitMma => {
                Face::Tf32
            }
        }
    }
}

/// Which kernel a GPU site gets, and how it is launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    pub arithmetic: Arithmetic,
    pub tiled: Option<GpuTile>,
    /// The tensor-core kernel, on a reduced face where the part has one.
    pub mma: Option<GpuMmaTile>,
    /// Hopper's asynchronous warpgroup tensor-core kernel (TF32 face, sm_90a).
    pub wgmma: Option<GpuWgmmaTile>,
    pub gx: u64,
    pub gy: u64,
    pub block: u64,
    pub smem: u64,
    /// The block-row group height the kernel walks its grid in (S54
    /// `raster`, [`Gpu::raster`]); 1 = row-major, no remap emitted.
    pub raster: u64,
    /// The rungs this site's kernel is emitted with: the profile's table at
    /// the site's fill, under the measurement arms.
    pub rungs: crate::GpuRungs,
    /// How the site's `k` fold is realized (ADR-0040 D4): in order, or cut
    /// and combined. [`launch`] gives `None`; [`launch_in`] chooses under
    /// the emission's order.
    pub combine: Combine,
    /// gemv-stream `vec-cells`: the adjacent cells (lanes of one row) a
    /// thread of the one-thread-per-cell kernel folds, one 16-byte load of
    /// `b` a `k` step ([`vec_cells`]); 1 everywhere else.
    pub cells: u64,
}

/// The realization of a tile site's `k` fold (ADR-0040 D3/D4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Combine {
    /// One chain per cell, increasing `k`: the ordered kernel.
    None,
    /// S62 `splitk`: `s` partial folds (`z` in `ctaid.y` above the unsplit
    /// `gy`) into a workspace, then one reduce launch in partition order.
    Workspace { s: u64 },
    /// S63 `splitk-atomic`: the same `s` partial folds, each added into the
    /// zeroed output by `atomicrmw fadd` (combine order unspecified).
    Atomic { s: u64 },
    /// S63 `streamk`: `ctas` blocks, each walking `iters` consecutive units
    /// of the `tiles × panels` iteration space; a tile's pieces add into the
    /// zeroed output by `atomicrmw fadd`. ADR-0050 C5 hybrid: `whole` blocks
    /// before them each fold one whole tile of the raster order (the whole
    /// waves, tiles `0..whole`), the `ctas` walking the tail's units after
    /// them — one launch of `whole + ctas` blocks (0: the plain walk).
    StreamK { ctas: u64, iters: u64, whole: u64 },
}

/// ADR-0050 C5: **the bytes the zero pass of launch `l` clears** — the
/// whole output, or a hybrid's prefix where its whole pieces store (the
/// 16-bit warp tile): its walked tiles are the raster order's first
/// `tiles − whole` ([`Kernel::segment`]), whose rows span the first
/// `h·⌈tail / (gx·h)⌉` tile rows (the whole tiles' blocks store over
/// whatever lies there).
pub(crate) fn zero_bytes(site: &TileSite, l: &Launch) -> u64 {
    let all = site.rows * site.c * elem_bytes_of(&site.elem);
    let Combine::StreamK { whole, .. } = l.combine else {
        return all;
    };
    // The 16-bit warp tile's whole pieces store (`mma_walk_store`); the FMA
    // tile's every piece adds (`store_o`), so its whole output is zeroed.
    let bm = match l.mma {
        Some(m) if m.fmt != MmaFmt::Tf32 => m.bm(),
        _ => return all,
    };
    if whole == 0 || bm == 0 || l.gx == 0 {
        return all;
    }
    let tiles = l.gx * (site.rows / bm);
    let h = l.raster.max(1);
    let rows = (bm * h * (tiles - whole).div_ceil(l.gx * h)).min(site.rows);
    rows * site.c * elem_bytes_of(&site.elem)
}

/// ADR-0050 C5: a hybrid stream-K launch's host-line tag (`+<whole>`: the
/// whole-tile blocks ahead of the walk; empty for the plain walk).
pub(crate) fn hybrid_tag(whole: u64) -> String {
    match whole {
        0 => String::new(),
        w => format!("+{w}"),
    }
}

impl Combine {
    /// The partitions a split kernel folds (`z` extent); 1 otherwise.
    pub(crate) fn splits(self) -> u64 {
        match self {
            Combine::Workspace { s } | Combine::Atomic { s } => s,
            _ => 1,
        }
    }
}

impl Launch {
    /// Blocks one wave holds: every SM times the blocks resident at this
    /// geometry (the derived tile's own count; for the one-thread-per-cell
    /// kernel, the thread and block limits of one SM).
    pub(crate) fn wave(&self, gpu: &Gpu) -> u64 {
        let per_sm = self
            .wgmma
            .map(|w| w.blocks_per_sm)
            .or(self.mma.map(|m| m.blocks_per_sm))
            .or(self.tiled.map(|t| t.blocks_per_sm))
            .unwrap_or_else(|| (gpu.max_threads_per_sm / self.block).min(gpu.max_blocks_per_sm));
        gpu.sms * per_sm.max(1)
    }
}

/// The staged kernel needs the separable shape the CPU's register-blocked rung
/// cashes (`b.ci == 0`): `a` ignores the lane and `b` ignores the row, so a
/// block's row panel of `a` and lane panel of `b` are each read once per `k`
/// step and shared by every cell of the tile. Strides are free — each operand
/// is staged with vector loads when its fast axis is unit-stride and aligned
/// ([`vector_a`], [`vector_b`]) and element by element otherwise. Anything else
/// takes the one-thread-per-cell kernel.
///
/// The rungs are decided here, once per site (S53): the site's fill is its
/// grid at the base FMA geometry (or the one-thread-per-cell grid), in warps
/// per scheduler; the profile's measured table at that fill and face, then the
/// arms.
pub(crate) fn launch(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> Launch {
    let l = launch_split(site, profile, contract, arms, 1, false);
    // ADR-0050 C4 `machine-place`: the ordered kernel every host use of
    // this launch names is the placed one.
    match l.rungs.machine_place {
        true => crate::place::placed_for(site, profile, contract, arms, Combine::None, l),
        false => l,
    }
}

/// [`launch`] for the kernel that folds `split` partitions (site-geometry
/// `fill-wave`): the FMA tile's panels and stages are sized for the
/// `blocks · split` grid ([`TargetProfile::gpu_tile_split`]). 1 = [`launch`].
/// `full` (tile-efficiency T1): the split grid keeps the full register tile.
pub(crate) fn launch_split(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    split: u64,
    full: bool,
) -> Launch {
    // ADR-0050 C2: the coverage probe sees every launch the rules make (a
    // test switch; off, the identity).
    let l = launch_split_rules(site, profile, contract, arms, split, full);
    crate::place::probe(site, profile, contract, arms, l)
}

pub(super) fn launch_split_rules(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    split: u64,
    full: bool,
) -> Launch {
    let e = elem_bytes_of(&site.elem);
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    // ADR-0043: a narrow operand is widened as its panel is copied through
    // registers (the register-staged tile, the one-thread-per-cell kernel);
    // a `cp.async` copy moves bytes unconverted and the tensor-core kernels
    // stage `elem`-wide fragments, so those refuse it (open: 16-bit panels
    // and `mma.m16n8k16`, ADR-0043 slice 2).
    let narrow = site.a.narrow.is_some() || site.b.narrow.is_some();
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    // `head-group`: a block's rows within one batch (its `b` panel's rows):
    // the tile is chosen over one batch's rows.
    let grows = row_group(site);
    let (blocks, threads) = fill_grid(site, profile);
    let wps = gpu.warps_per_scheduler(blocks, threads);
    // gapfix tensor-tile (C11): the tile's `b` element as staged and its
    // pipe's rate price row-major's `b` re-reads ([`Gpu::raster`]).
    let raster =
        |rungs: crate::GpuRungs, bm: u64, bn: u64, per_sm: u64, eb: u64, rate: u64| match rungs
            .raster
        {
            true => gpu.raster(
                bm,
                bn,
                gpu.sms * per_sm,
                site.rows,
                site.c,
                site.k,
                e,
                rungs.l2_gate,
                eb,
                rate,
            ),
            false => 1,
        };
    let matrix_rate = |fmt: MmaFmt, home: Home| {
        gpu.caps
            .matrix_op(fmt, home)
            .map_or(0, |op| gpu.matrix_flops_per_ns(&op))
    };
    // Attempt tensor geometry under its own policy. Permission alone is not
    // an emitted face: rejection must fall through to the scalar policy.
    // recurrence-precision `tf32-tile`: the permission is the site's product
    // set (ADR-0041 D6); off, the f32 FFMA tile it also admits.
    // A mixed-radix site stores digit by digit: site-geometry `tf32-radix`
    // gives the `mma.sync` tile element copies and stores for its
    // digit-affine addresses (`wgmma`'s swizzled 16-byte panels have none).
    let rx = site.radix.as_deref();
    let radix_ok = |rungs: crate::GpuRungs| rx.is_none() || rungs.tf32_radix;
    // tensor-cores (ADR-0043 D4 / D6): a product of 16-bit multiplicands on
    // the part's warp instruction for that format ([`half_operands`]): the
    // `m16n8` tile of the TF32 kernel at `k` 16, its panels at their storage
    // widths. Off (`half-tile`), or where no tile divides the site: the TF32
    // kernel the set may also admit, else the FMA tile.
    if let Some((fmt, ea, eb)) = half_operands(site, gpu)
        && shaped
        && rx.is_none()
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        let bt = site.b.clane != 1;
        // wgmma-16bit (`wgmma16`): the warpgroup product where the part
        // lists one for the format faster than its warp `mma`
        // ([`crate::rules::wgmma16_pays`]) — `b` its descriptor operand,
        // stored in the format; `a` from a K-major descriptor where stored in
        // it, else from registers (rounded in the fragment load) — on the
        // warp-specialized tile family's geometry and tensor boxes.
        let group = (rungs.mma && rungs.half_tile && rungs.wgmma16 && eb == fmt.bytes())
            .then(|| {
                let a = if ea == fmt.bytes() {
                    Home::Smem
                } else {
                    Home::Regs
                };
                crate::rules::wgmma16_pays(gpu, fmt, a)
            })
            .flatten();
        let ws16 = group
            .and_then(|op| {
                let form = WsForm {
                    op,
                    ea,
                    eb,
                    b_kmajor: bt,
                };
                profile.gpu_wgmma_ws_tile_of(
                    site.rows,
                    site.c,
                    site.k,
                    rungs.ws_regs,
                    rungs.tma_digits,
                    form,
                )
            })
            .filter(|w| ws_tensors(site, w, gpu, rungs.tma_digits).is_some());
        if let (Some(w), Some(op)) = (ws16, group) {
            let tiles = (site.rows / w.bm()) * (site.c / w.bn());
            let ctas = gpu.sms * w.blocks_per_sm.max(1);
            let tile_bytes = w.bm() * w.bn() * 4;
            let w = match rungs.ws_persist
                && crate::rules::ws_persist_pays(gpu, tiles, ctas, w.stages, tile_bytes)
            {
                true => GpuWgmmaTile { ctas, ..w },
                false => w,
            };
            let (gx, gy) = match w.ctas {
                0 => (site.c / w.bn(), site.rows / w.bm()),
                c => (c, 1),
            };
            return Launch {
                arithmetic: Arithmetic::HalfWgmma,
                tiled: None,
                mma: None,
                wgmma: Some(w),
                gx,
                gy,
                block: w.threads(gpu.warp),
                smem: w.smem_bytes(),
                raster: raster(
                    rungs,
                    w.bm(),
                    w.bn(),
                    w.blocks_per_sm,
                    w.eb,
                    gpu.matrix_flops_per_ns(&op),
                ),
                rungs,
                combine: Combine::None,
                cells: 1,
            };
        }
        let tile = (rungs.mma && rungs.half_tile)
            .then(|| {
                profile.gpu_mma16_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    fmt,
                    ea,
                    eb,
                    bt,
                    Depth::of(&rungs, e),
                    None,
                )
            })
            .flatten();
        // tensor-cores: the 16-bit tile pays where the FMA tile would be
        // bound by its FFMA issue — the site's flops per operand byte at or
        // above the part's balance ([`Gpu::ffma_per_dram_byte_pm`]) — or
        // where its own grid gives every warp scheduler its warps
        // ([`Gpu::fill_warps`]). Below the balance with an under-filled grid
        // the bytes in flight decide, and the FMA tile splits `k` to fill
        // the part (RTX 5080, 16 × 6144 × 4096: 15.9 flop / B against 54.9,
        // 96 one-warp blocks: 50.5 us against the FMA tile's 36.9; 128 rows:
        // 116 flop / B, 768 warps: 110.8 against 174.0 us). Open: the split
        // 16-bit tile.
        let bytes = site.rows * site.k * ea + site.k * site.c * eb + site.rows * site.c * 4;
        let balanced = gpu
            .ffma_per_dram_byte_pm()
            .is_none_or(|b| 2 * site.rows * site.c * site.k * 1000 >= b * bytes);
        let filled = |m: &GpuMmaTile| {
            !rungs.half_fill
                || balanced
                || (site.rows / m.bm()) * (site.c / m.bn()) * m.warps_x * m.warps_y
                    >= gpu.fill_warps()
        };
        if let Some(m) = tile.filter(filled) {
            return Launch {
                arithmetic: Arithmetic::HalfMma,
                tiled: None,
                mma: Some(m),
                wgmma: None,
                gx: site.c / m.bn(),
                gy: site.rows / m.bm(),
                block: m.threads(gpu.warp),
                smem: m.smem_bytes(),
                raster: raster(
                    rungs,
                    m.bm(),
                    m.bn(),
                    m.blocks_per_sm,
                    m.eb,
                    matrix_rate(fmt, Home::Regs),
                ),
                rungs,
                combine: Combine::None,
                cells: 1,
            };
        }
    }
    if site.tf32
        && site.elem == Ty::f32()
        && shaped
        && !narrow
        && (vector_a(site) || rx.is_some_and(|x| x.a.is_some()))
        && (vector_b(site) || rx.is_some_and(|x| x.b.is_some()))
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        if rungs.mma && rungs.tf32_tile && radix_ok(rungs) {
            // hopper-wgmma-tma R2: the warp-specialized kernel where both
            // operands are tensor boxes of its panels, else the S53 kernel.
            let wgmma = (rungs.wgmma && rx.is_none())
                .then(|| {
                    rungs
                        .wgmma_ws
                        .then(|| {
                            profile.gpu_wgmma_ws_tile(
                                site.rows,
                                site.c,
                                site.k,
                                rungs.ws_regs,
                                rungs.tma_digits,
                            )
                        })
                        .flatten()
                        .filter(|w| ws_tensors(site, w, gpu, rungs.tma_digits).is_some())
                        .or_else(|| profile.gpu_wgmma_tile(site.rows, site.c, site.k))
                })
                .flatten();
            if let Some(w) = wgmma {
                // hopper-launch `ws-persist`: the warp-specialized kernel on
                // `sms × blocks_per_sm` CTAs walking its tiles, where the walk
                // saves the later rounds' fills and epilogues.
                let tiles = (site.rows / w.bm()) * (site.c / w.bn());
                let ctas = gpu.sms * w.blocks_per_sm.max(1);
                let tile_bytes = w.bm() * w.bn() * 4;
                let w = match w.ws
                    && rungs.ws_persist
                    && crate::rules::ws_persist_pays(gpu, tiles, ctas, w.stages, tile_bytes)
                {
                    true => GpuWgmmaTile { ctas, ..w },
                    false => w,
                };
                let (gx, gy) = match w.ctas {
                    0 => (site.c / w.bn(), site.rows / w.bm()),
                    c => (c, 1),
                };
                return Launch {
                    arithmetic: Arithmetic::Tf32Wgmma,
                    tiled: None,
                    mma: None,
                    wgmma: Some(w),
                    gx,
                    gy,
                    block: w.threads(gpu.warp),
                    smem: w.smem_bytes(),
                    raster: raster(
                        rungs,
                        w.bm(),
                        w.bn(),
                        w.blocks_per_sm,
                        4,
                        matrix_rate(MmaFmt::Tf32, Home::Smem),
                    ),
                    rungs,
                    combine: Combine::None,
                    cells: 1,
                };
            }
            let mma = profile.gpu_mma_tile_at(
                grows,
                site.c,
                site.k,
                rungs.wide,
                rungs.fill,
                Depth::of(&rungs, e),
            );
            if let Some(m) = mma {
                return Launch {
                    arithmetic: Arithmetic::Tf32Mma,
                    tiled: None,
                    mma: Some(m),
                    wgmma: None,
                    gx: site.c / m.bn(),
                    gy: site.rows / m.bm(),
                    block: m.threads(gpu.warp),
                    // TMA `bulk-stage`: a transaction barrier a stage.
                    smem: m.smem_bytes()
                        + if bulk_stage(site, &m, gpu, &rungs) {
                            BULK_BAR_BYTES * m.stages
                        } else {
                            0
                        },
                    raster: raster(
                        rungs,
                        m.bm(),
                        m.bn(),
                        m.blocks_per_sm,
                        4,
                        matrix_rate(MmaFmt::Tf32, Home::Regs),
                    ),
                    rungs,
                    combine: Combine::None,
                    cells: 1,
                };
            }
        }
    }
    // f32-tensor-cores R-F `split-tile`: a `{f32}` product on the part's
    // matrix instruction by a split form, where the bound admits it and the
    // rate rule prices it ([`split_operands`], [`split_pays`]).
    let least = site_rungs(profile, wps, Face::Tf32, arms, site).split_least;
    if let Some((kind, fmt, ea, eb)) = split_operands(site, gpu, least)
        && shaped
        && rx.is_none()
    {
        let rungs = site_rungs(profile, wps, Face::Tf32, arms, site);
        let bt = site.b.clane != 1;
        let tile = (rungs.mma && rungs.split_tile)
            .then(|| match kind {
                SplitKind::Tf32x3 => profile.gpu_mma_split_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    Depth::of(&rungs, e),
                ),
                _ => profile.gpu_mma16_tile(
                    grows,
                    site.c,
                    site.k,
                    rungs.wide,
                    rungs.fill,
                    fmt,
                    ea,
                    eb,
                    bt,
                    Depth::of(&rungs, e),
                    Some(split_parts(kind, ea, eb)),
                ),
            })
            .flatten()
            .filter(|m| !bulk_stage(site, m, gpu, &rungs))
            .map(|m| GpuMmaTile {
                split: Some(kind),
                ..m
            })
            .filter(|m| split_pays(site, gpu, kind, m, arms.on.split_tile, rungs.split_issue));
        if let Some(m) = tile {
            return Launch {
                arithmetic: Arithmetic::SplitMma,
                tiled: None,
                mma: Some(m),
                wgmma: None,
                gx: site.c / m.bn(),
                gy: site.rows / m.bm(),
                block: m.threads(gpu.warp),
                smem: m.smem_bytes(),
                raster: raster(
                    rungs,
                    m.bm(),
                    m.bn(),
                    m.blocks_per_sm,
                    m.eb,
                    matrix_rate(m.fmt, Home::Regs) / SplitForm::of(kind).products().max(1),
                ),
                rungs,
                combine: Combine::None,
                cells: 1,
            };
        }
    }
    let arithmetic = Arithmetic::scalar(site, contract, arms.free, arms.pairs);
    let rungs = site_rungs(profile, wps, arithmetic.face(), arms, site);
    // open-rules `wide-fma`, the site's side ([`crate::rules::wide_fma_pays`]):
    // the wide tile's own grid (its split partitions included) against the
    // blocks one wave of it holds. The ON arm takes it anyway.
    let rungs = match rungs.wide_fma && !arms.on.wide_fma && shaped {
        true => {
            let wide =
                profile.gpu_tile_split(grows, site.c, site.k, &site.elem, true, rungs, split);
            let pays = wide.is_none_or(|w| {
                let blocks = (site.rows / w.bm()) * (site.c / w.bn()) * split;
                crate::rules::wide_fma_pays(gpu, blocks, gpu.sms * w.blocks_per_sm.max(1))
            });
            crate::GpuRungs {
                wide_fma: pays,
                ..rungs
            }
        }
        false => rungs,
    };
    // tile-efficiency T2: `a`'s panel rows run along `k` (`vector_a`), so it
    // can stay untransposed — where one quarter-warp's fragment read is one
    // panel row (`tx` ≥ the lanes a wavefront serves, [`Kernel::tiled`]).
    let kmajor = rungs.kmajor_a && vector_a(site) && rx.is_none_or(|x| x.a.is_none());
    let tile_rungs = crate::GpuRungs {
        kmajor_a: kmajor,
        full_tile: full,
        ..rungs
    };
    // per-site-choice A2: the k-major panel's bank pad, where it fits — on
    // each pipeline's panel before they are weighed, and on the final stages.
    let kpad = |g: GpuTile| match g.kmajor && rungs.bank_plan {
        true => {
            profile.gpu_tile_kmajor_pad(g, site.k, &site.elem, |at| kmajor_pad(site, at, &rungs))
        }
        false => g,
    };
    let tiled = shaped
        .then(|| {
            profile.gpu_tile_split(
                grows,
                site.c,
                site.k,
                &site.elem,
                rungs.wide_fma,
                tile_rungs,
                split,
            )
        })
        .flatten()
        // device-geometry `head-group`: a row group below every derived
        // tile's rows — one block a group, `tm` its rows.
        .or_else(|| {
            (shaped && grows < site.rows)
                .then(|| profile.gpu_group_tile(grows, site.c, site.k, &site.elem))
                .flatten()
        })
        // S62 `cell-fill`: no register tile divides the site — one output per
        // thread, still staged ([`TargetProfile::gpu_cell_tile`]).
        .or_else(|| {
            (shaped && rungs.cell_fill)
                .then(|| profile.gpu_cell_tile(grows, site.c, site.k, &site.elem))
                .flatten()
        })
        .map(|g| GpuTile {
            kmajor: kmajor && g.tx >= GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES,
            ..g
        })
        .and_then(|g| {
            // ADR-0034 D1: a forced stage count picks the pipeline (1 =
            // register staging, >= 2 = `cp.async`) over the rung.
            let stream = match gpu.geom.stages {
                0 => rungs.cp_async,
                s => s >= 2,
            } && (!narrow || narrow_streams(site, &rungs));
            let streamed = stream
                .then(|| profile.gpu_tile_async(g, site.k, &site.elem))
                .flatten()
                .map(kpad);
            // tile-efficiency T3 (`async-depth`): a thread does `tm·tn·bk ≥
            // 1024` FFMA per barrier and wait. A `cp.async` panel shallower
            // than that yields to register staging only where staging's
            // panel is deeper (`k = 8·odd` offers neither a deeper one); a
            // forced ADR-0034 depth stands.
            let floor = 1024 / (g.tm * g.tn).max(1);
            // per-site-choice D2: a forced `async` arm takes the panel anyway.
            let shallow = |a: &GpuTile| {
                rungs.async_depth && gpu.geom.bk == 0 && a.bk < floor && !arms.on.cp_async
            };
            if let Some(a) = streamed.filter(|a| !shallow(a)) {
                return Some(a);
            }
            // Register staging: the prefetch panel takes what a whole-file tile
            // leaves of the file (`wide-fma`), else (S62 `panel-regs`) what the
            // derived residency leaves after the tile, else one fragment pair.
            // `panel-regs` only on a grid that leaves a warp scheduler without a
            // warp: there the panel's independent loads are the only latency
            // hiding; with a warp per scheduler, warps hide it and the deeper
            // panel costs registers (S62 v1, the 5090, 4090, H100 PCIe and 3090:
            // 35 of 38 measured cells on this side of the line).
            let grid = (site.rows / g.bm()) * (site.c / g.bn()) * split;
            let g = match rungs.prefetch {
                true if g.held > 0 => {
                    prefetch_depth(g, e, gpu.regs_per_thread.saturating_sub(g.held))
                }
                true if rungs.panel_regs && gpu.warps_per_scheduler(grid, g.threads()) == 0 => {
                    prefetch_depth(g, e, panel_regs(gpu, &g, e))
                }
                true => prefetch_depth(g, e, g.tm + g.tn),
                false => g,
            };
            let staged = if rungs.pad && vector_a(site) && !g.kmajor {
                profile.gpu_tile_sync_a(g, &site.elem)
            } else {
                kpad(g)
            };
            match streamed {
                Some(a) if a.bk >= staged.bk => Some(a),
                // tensor-cores `staged-regs`: a register-staged panel the
                // thread cannot hold beside its tile — every copy's value
                // live from its load to its store, `(bm + bn)·bk / threads`
                // registers — spills (the block decode's narrow 1 × 4096 ×
                // 14336 site: one-row `a` panels whole at `bk` 128 over 32
                // threads, 516 registers, 216–244 B spilled on every rented
                // part). No such tile: the stream where the rules give one,
                // else the next kernel.
                _ if rungs.staged_regs && !staged_fits(gpu, &staged, e) => streamed,
                _ => Some(staged),
            }
        })
        // S62 `asyncstage`: stages from the blocks this grid puts on an SM.
        // Legal for the unpredicated copies above: `rows % bm`, `c % bn`,
        // `k % bk` are 0 (`TileSite` extents) and `shaped` makes each panel
        // one read per block; the panel order `0..k/bk` and each cell's step
        // order are unchanged ([`Kernel::step`]).
        .map(|g| {
            let blocks = (site.rows / g.bm()) * (site.c / g.bn()) * split;
            // per-site-choice D2: `async` forced off is register staging.
            (rungs.async_stage
                && gpu.geom.stages == 0
                && !arms.off.cp_async
                && (!narrow || narrow_streams(site, &rungs)))
            .then(|| profile.gpu_tile_stages(g, blocks, site.k, &site.elem))
            .flatten()
            .unwrap_or(g)
        })
        .map(kpad);
    // per-site-choice A4 (register rule): the k-major panel is taken where its
    // geometry is no worse than the transposed panel's — `bk` and blocks/SM,
    // which carry its `lanes − 1` extra fragment registers a row
    // ([`TargetProfile::gpu_tile_split`]) — and its `a` panel conflict-free
    // (`adeg` 1: no pad fit, [`TargetProfile::gpu_tile_kmajor_pad`]).
    // Elsewhere the transposed tile, the kernel's rungs unchanged. An arm
    // that forces `kmajor-a` on takes the k-major tile wherever it applies:
    // the rule's alternative (per-site-choice Part B) and its ON arm. A tile
    // too narrow for the k-major read (`tx`) is the transposed one either way.
    if let Some(k) = tiled.filter(|_| kmajor && rungs.bank_plan) {
        let [a, _, st, _] = panel_excess(site, &k, &rungs, false);
        let mut flat = arms;
        *flat.on.rung("kmajor-a").expect("a rung") = false;
        *flat.off.rung("kmajor-a").expect("a rung") = true;
        let t = launch_split(site, profile, contract, flat, split, full);
        let ruled = |t: &GpuTile| {
            arms.on.kmajor_a
                || (a.max(st) == 1 && k.bk >= t.bk && k.blocks_per_sm >= t.blocks_per_sm)
        };
        if t.tiled.is_some_and(|t| !(k.kmajor && ruled(&t))) {
            return Launch { rungs, ..t };
        }
    }
    match tiled {
        Some(g) => Launch {
            arithmetic,
            tiled: Some(g),
            mma: None,
            wgmma: None,
            gx: site.c / g.bn(),
            gy: site.rows / g.bm(),
            block: g.threads(),
            smem: g.smem_bytes(e),
            raster: raster(
                rungs,
                g.bm(),
                g.bn(),
                g.blocks_per_sm,
                site.b.narrow.map_or(e, |n| n.bytes()),
                gpu.plan_cost().flops_per_ns,
            ),
            rungs,
            combine: Combine::None,
            cells: 1,
        },
        None => {
            // gemv-stream `vec-cells`: the ordered kernel takes the vector
            // cells where their grid alone puts a block on every SM; a split
            // kernel takes them where the split grid does ([`resized`]).
            let v = vec_fit(site, gpu, &rungs);
            let cells = match vec_takes(site, gpu, &rungs, v, 1) {
                true => v,
                false => 1,
            };
            Launch {
                arithmetic,
                tiled: None,
                mma: None,
                wgmma: None,
                // One thread per cell: its own grid, not the fill key's (which
                // keys a tile the launch may not take — a `--geom=` arm).
                gx: cell_blocks(site, cells),
                gy: 1,
                block: NAIVE_THREADS,
                smem: 0,
                raster: 1,
                rungs,
                combine: Combine::None,
                cells,
            }
        }
    }
}
