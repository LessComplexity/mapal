//! The fold combine choice, stream-K pricing and the register-staged tile rules.

use super::*;

pub(super) fn combine(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    order: Order,
    arms: crate::GpuArms,
) -> Combine {
    // A window's taps (`window`) are walked in order: its `k` is not cut.
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || radix_whole(site);
    if order == Order::Sequential || l.wgmma.is_some() || window {
        return Combine::None;
    }
    if let Some(m) = l.mma {
        return mma_combine(l, &m, site, profile, order, arms);
    }
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let rungs = l.rungs;
    let e = elem_bytes_of(&site.elem);
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let blocks = l.gx * l.gy;
    let wave = l.wave(gpu);
    // site-geometry `fill-wave`: a grid with fewer warps than the fill rule's
    // target is cut into the fewest partitions that reach it, inside the wave
    // the part holds at the tile's own registers and threads.
    let warps = l.block.div_ceil(gpu.warp);
    let under = rungs.fill_wave && blocks * warps < gpu.fill_warps();
    let cap = match l.tiled {
        Some(g) if under => gpu.sms * profile.tile_capacity(&g, &site.elem),
        _ => wave,
    };
    // `--splitk=` (arms) over the part's table (`Geom::split`).
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    // gemv-stream `stream-fly`: a streaming one-thread-per-cell site is cut
    // until its grid — at the vector cells its split takes ([`vec_fit`]) —
    // puts a block on every SM with the part's in-flight knee in flight on
    // each ([`TargetProfile::split_stream`]): a thread keeps `NAIVE_UNROLL`
    // loads of `cells · e_b` bytes of `b` in flight.
    let naive = l.tiled.is_none();
    let stream = naive && rungs.stream_fly && streaming(site, gpu);
    // gaps-disasm C5 `split-price`: an under-filled tile whose operands stream
    // from DRAM (its flops per byte below the part's FFMA balance) takes the
    // atomic split its price gives ([`TargetProfile::split_priced`]: the
    // stream at the grid's spread against every partition's partial cells).
    let bytes = (site.rows * site.k + site.k * site.c + site.rows * site.c) * e;
    let streams = gpu
        .ffma_per_dram_byte_pm()
        .is_some_and(|b| 2 * site.rows * site.c * site.k * 1000 < b * bytes);
    let priced = |workspace: bool| {
        (under && !workspace && !naive && streams && rungs.split_price)
            .then(|| profile.split_priced(blocks, cap, panels, site.rows * site.c, bytes))
            .flatten()
    };
    let count = |workspace| match forced {
        s if s > 0 && panels.is_multiple_of(s) => s,
        _ => {
            let rule = match (under, priced(workspace)) {
                (true, Some(s)) => s,
                (true, None) => {
                    profile.split_fill(blocks, warps, cap, panels, site.rows * site.c, e, workspace)
                }
                (false, _) => {
                    profile.split_count(blocks, wave, panels, site.rows * site.c, e, workspace)
                }
            };
            match stream {
                true => rule.max(stream_split(
                    site, profile, gpu, &rungs, wave, panels, e, workspace,
                )),
                false => rule,
            }
        }
    };
    let s_ws = if rungs.splitk && l.raster == 1 {
        count(true)
    } else {
        1
    };
    if order == Order::Fixed {
        return match s_ws {
            1 => Combine::None,
            s => Combine::Workspace { s },
        };
    }
    // large-gemm `k-fill`: an under-filled site's full tile, stream-K over
    // the slots it holds ([`k_fill`]); [`resized`] gives its kernel that tile.
    // (Its own rung: the table's `streamk` is the filled class's; an arm
    // forcing stream-K off turns it off too.)
    if !arms.off.get("streamk")
        && let Some((c, _)) = k_fill(l, site, profile, contract, arms)
    {
        return c;
    }
    // `red.global.add.f32` exists from sm_20, `.f64` from sm_60.
    let atomic = e == 4 || gpu.caps.red_f64;
    let s_at = count(false);
    // `fill-wave` counts the split, not the ladder: stream-K is weighed
    // against the S63 count, as before.
    let s_cmp = match under && forced == 0 {
        true => profile.split_count(blocks, wave, panels, site.rows * site.c, e, false),
        false => s_at,
    };
    // ponytail: stream-K runs the register-staged tile (its panel loop takes
    // a runtime trip count); the cp.async pipeline's prologue is unrolled over
    // a constant one — add a predicated prologue when a streamed tile needs it.
    // large-gemm `streamk-fill`: stream-K spreads the tiles over the slots
    // the launched grid holds (`wave`); a grid that does not give the part
    // `fill_warps` at its own geometry holds one block an SM there (its
    // `grid-share` residency) — 84 one-warp blocks for 16 × 2048 × 7168,
    // 1.41× cuBLAS — so it takes the `fill-wave` split instead.
    // (With `fill-wave` off there is no fill split to take: stream-K as before.)
    let filled_grid =
        !(rungs.streamk_fill && rungs.fill_wave) || blocks * warps >= gpu.fill_warps();
    // sweep-gemm-2 `streamk-price`: where the part prices a stream-K fixup
    // ([`streamk_priced`]: its FMA tile rate, `red` rate and zero pass), an
    // under-filled site (its class ships `streamk` off) streams too when the
    // price says so.
    // (An arm that turns `streamk` off — a group's order, a measurement —
    // stays off.)
    let priced = rungs.streamk_price
        && !arms.off.get("streamk")
        && l.tiled.is_some_and(|g| {
            streamk_priced(profile, site, &g, wave, blocks, wave, 1, 1, 1).is_some()
        });
    let streamk = (rungs.streamk || priced)
        && filled_grid
        && atomic
        && l.tiled.is_some_and(|g| g.stages < 2)
        && panels >= 2
        && blocks * panels > wave;
    if streamk {
        let iters = (blocks * panels).div_ceil(wave);
        let split = (blocks * s_cmp).div_ceil(wave) * (panels / s_cmp);
        let ctas = (blocks * panels).div_ceil(iters);
        let g = l.tiled.expect("a streamed site is a tile");
        // A class that ships `streamk` keeps its panel count (its measured
        // wins, `rung_replay.tsv`); an under-filled site streams by its price.
        let pays = match rungs.streamk {
            true => iters + 1 < split,
            false => {
                streamk_priced(profile, site, &g, wave, blocks, ctas, iters, s_cmp, split)
                    == Some(true)
            }
        };
        if pays {
            return Combine::StreamK {
                ctas,
                iters,
                whole: 0,
            };
        }
    }
    if rungs.splitk_atomic && atomic && l.raster == 1 && s_at > 1 {
        return Combine::Atomic { s: s_at };
    }
    match s_ws {
        1 => Combine::None,
        s => Combine::Workspace { s },
    }
}

/// **sweep-gemm-2 `streamk-price`: does stream-K beat the S63 count at this
/// site, its fixed costs priced by the part's rates?** The FMA tile's
/// `flops_per_ns`, `red.global.add.f32`'s `atomic_ops_per_ns` and the zero
/// pass's `zero_ns` where the part measured them, else derived from its
/// properties ([`Gpu::plan_cost`]: review-fixes — the 3090 / A100 / H100
/// profiles measured none of the three, and the panel-count rule this price
/// replaced, measured losing, stood there). `None` only where the element is
/// not `f32` (the rates are f32's) or the part gives no factor to derive
/// them from (no clock): the panel-count rule `iters + 1 < split` there.
///
/// One block's panel takes `t = 2·bm·bn·bk / (flops_per_ns / wave)` — the
/// tile's FMAs at its share of the part's FMA tile rate. Stream-K walks
/// `iters` panels a block, then every piece (`ctas + tiles` of them: a block's
/// walk crosses a tile edge once per tile it touches) adds its `bm·bn`
/// partials (`red`, at `atomic_ops_per_ns` over the device), after the
/// output's zero pass (`zero_ns` + its bytes at the fill rate). The S63 count
/// `s` runs `split` panels a block; when it cuts (`s > 1`) it pays `s·tiles`
/// tiles of partials and the same zero pass. A count without those costs
/// (`iters + 1 < split`, one panel of fixup) lost where pieces are many
/// against panels (rtx-5080: the 2048 × 2048 × 128 scores, 8 panels a tile,
/// 1.41×; the 512³ cells, 1.15–1.24×) and won where a tile holds 64 panels
/// (loopmm's 1024³, 0.95×).
#[allow(clippy::too_many_arguments)]
pub(super) fn streamk_priced(
    profile: &TargetProfile,
    site: &TileSite,
    g: &GpuTile,
    wave: u64,
    tiles: u64,
    ctas: u64,
    iters: u64,
    s: u64,
    split: u64,
) -> Option<bool> {
    let d = profile.gpu()?.plan_cost();
    let e = elem_bytes_of(&site.elem);
    if e != 4 || d.flops_per_ns == 0 || d.atomic_ops_per_ns == 0 || d.zero_ns == 0 {
        return None;
    }
    let fill = match (d.fill_bytes_per_ns, d.l2_bytes_per_ns) {
        (0, 0) => return None,
        (0, l2) => l2,
        (f, _) => f,
    };
    let tile = (g.bm() * g.bn()) as f64;
    let panel = 2.0 * tile * g.bk as f64 * wave as f64 / d.flops_per_ns as f64;
    let partials = |pieces: u64| pieces as f64 * tile / d.atomic_ops_per_ns as f64;
    let zero = d.zero_ns as f64 + (site.rows * site.c * e) as f64 / fill as f64;
    let streamk = iters as f64 * panel + partials(ctas + tiles) + zero;
    let cut = split as f64 * panel
        + match s {
            1 => 0.0,
            s => partials(s * tiles) + zero,
        };
    Some(streamk < cut)
}

/// The prefetch rung holds the next panel in registers while the current one
/// is computed, so its depth is bounded by registers, not shared memory: one
/// panel's elements per thread, `(bm + bn)·bk / threads`, may take at most
/// `budget` registers — one fragment pair's worth (`tm + tn`, S52), or what
/// the register tile leaves of the file (`wide-fma`, S53). Halve `bk` until it
/// fits; a halved power of two still divides `k` and still loads in whole
/// vectors per thread.
/// tensor-cores `staged-regs`: a register-staged tile's panel — each copy's
/// value held from its load to its store, `(bm + bn)·bk` elements over the
/// block's threads — fits the registers a thread may hold. The panel alone:
/// ptxas shares the tile's fragment registers with the copies it has stored
/// (the B8 projection's 8 × 8 tile, 132 panel + 128 tile registers, compiles
/// without a spill), while a panel past the file spills whatever the tile
/// needs (the block decode's one-row sites, 516: 216–244 B; the 5090's 1 ×
/// 18432 × 7168 GEMV, 516: 2512 B). A cp.async tile holds no panel in
/// registers.
pub(super) fn staged_fits(gpu: &Gpu, g: &GpuTile, elem_bytes: u64) -> bool {
    g.stages >= 2
        || (g.bm() + g.bn()) * g.bk * elem_bytes.div_ceil(4) / g.threads() <= gpu.regs_per_thread
}

pub(super) fn prefetch_depth(g: GpuTile, elem_bytes: u64, budget: u64) -> GpuTile {
    let lanes = GPU_VEC_BYTES / elem_bytes;
    let regs = |bk: u64| (g.bm() + g.bn()) * bk * elem_bytes.div_ceil(4) / g.threads();
    let whole = |bk: u64| {
        (g.bm() * bk / lanes).is_multiple_of(g.threads())
            && (bk * g.bn() / lanes).is_multiple_of(g.threads())
    };
    let mut bk = g.bk;
    while regs(bk) > budget && bk / 2 >= lanes && whole(bk / 2) {
        bk /= 2;
    }
    GpuTile { bk, ..g }
}

/// S62 `panel-regs`: the registers a register-staged thread has free for its
/// prefetch panel — the file the derived residency gives it (`regs_per_sm`
/// over `threads × blocks_per_sm`, at most `regs_per_thread`; the bound
/// `minctasm` holds ptxas to) less what its tile holds ([`fma_held_regs`]),
/// and never less than S52's one fragment pair. The S52 8×8 tile at 256
/// threads, two per SM, holds its whole 128-register share: the pair.
pub(super) fn panel_regs(gpu: &Gpu, g: &GpuTile, elem_bytes: u64) -> u64 {
    let file = (gpu.regs_per_sm / (g.threads() * g.blocks_per_sm.max(1))).min(gpu.regs_per_thread);
    file.saturating_sub(fma_held_regs(g.tm, g.tn, elem_bytes))
        .max(g.tm + g.tn)
}

pub(super) fn aligned(site: &TileSite, x: u64) -> bool {
    x.is_multiple_of(GPU_VEC_BYTES / elem_bytes_of(&site.elem))
}

/// `a`'s panel rows are contiguous along `k` and vector-aligned.
pub(super) fn vector_a(site: &TileSite) -> bool {
    site.a.ck == 1 && aligned(site, site.a.ci) && aligned(site, site.a.base)
}

/// `b`'s panel rows are contiguous along the lane and vector-aligned.
pub(super) fn vector_b(site: &TileSite) -> bool {
    site.b.clane == 1 && aligned(site, site.b.ck) && aligned(site, site.b.base)
}

/// **sweep-attention-2 `kmajor-b`: a register-staged tile's `b` panel copied
/// along its k-major rows** — a vector of `k` a copy (16 bytes of f32, 8 of
/// a 16-bit format), each key's row read coalesced, scattered into the
/// `[k][n]` panel the fragments read — where `b` is unit-stride along `k`
/// and not along its lanes ([`b_k_major`]), every other term of its address
/// a whole vector (a mixed-radix `b`: one unit `k` digit whose radix the
/// vector divides, its row and lane digits whole vectors), and `bk` whole
/// vectors. Otherwise ([`vector_b`] aside) the copies walk a column one
/// element each: every thread of a warp a different row, a request a row.
pub(super) fn kvec_b(s: &TileSite, g: &GpuTile) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    if g.stages >= 2 || vector_b(s) || !g.bk.is_multiple_of(lanes) || !b_k_major(s) {
        return false;
    }
    match s.radix.as_deref().and_then(|r| r.b.as_ref()) {
        Some(b) => {
            matches!(b.k[..], [d] if d.stride == 1 && d.coef == 1 && d.radix.is_multiple_of(lanes))
                && aligned(s, b.base)
                && b.lane.iter().chain(&b.i).all(|d| aligned(s, d.coef))
        }
        None => {
            s.b.ck == 1 && aligned(s, s.b.clane) && aligned(s, s.b.base) && s.b.ksplit.is_none()
        }
    }
}

/// gapfix `kvec-swizzle` (gaps-disasm C12): **the k-major `b` panel's word
/// groups XOR-swizzled by the bank geometry** — the copy scatters a vector
/// along `k` to `lanes` panel rows `bn` words apart, so a warp's stores of one
/// element land `bn` apart: at `bn` a multiple of the banks' 32 words, every
/// thread with the same key row in one bank (the decode scores' 8-way
/// `STS`). A warp's copies cover `kv = bk ÷ lanes` key chunks of `banks ÷ kv`
/// consecutive key rows each; panel word `k·bn + n` is stored at
/// `k·bn + (n ⊕ σ(k))`, `σ(k) = ((k ÷ lanes) mod G)·S` with `S` the rows a
/// chunk spans (at least a fragment vector, `lanes` words) and `G = banks ÷
/// S`: each chunk's rows on their own banks — conflict-free where `S` is the
/// rows (`bk ≤ 32`), `S ÷ rows`-way past it. The fragment reads apply the
/// same `σ` (a permutation of whole `lanes` groups inside the row: their
/// vectors and their degree unchanged). `Some((G, S))` where the panel is
/// f32 words, `kmajor-b` copies and `bn` is whole bank rows; `None`
/// otherwise or with the rung off.
pub(super) fn kvec_swizzle(
    s: &TileSite,
    g: &GpuTile,
    rungs: &crate::GpuRungs,
) -> Option<(u64, u64)> {
    let e = elem_bytes_of(&s.elem);
    let lanes = GPU_VEC_BYTES / e;
    let banks = GPU_WAVEFRONT_BYTES / 4;
    let kv = (g.bk / lanes).max(1);
    let step = (banks / kv).max(lanes);
    let groups = banks / step;
    (rungs.kvec_swizzle
        && rungs.kmajor_b
        && kvec_b(s, g)
        && e == 4
        && groups > 1
        && g.bn().is_multiple_of(banks))
    .then_some((groups, step))
}

/// gemv-stream: the site's flops per operand byte fall below the part's
/// FFMA balance ([`Gpu::ffma_per_dram_byte_pm`]): its time is the bytes it
/// streams, not its FMAs (a matrix-vector product: `2·c·k` flops over
/// `c·k·e` bytes). Unknown balance: not streaming.
pub(super) fn streaming(site: &TileSite, gpu: &Gpu) -> bool {
    let bytes = site.rows * site.k * site.a.elem_bytes(&site.elem)
        + site.k * site.c * site.b.elem_bytes(&site.elem)
        + site.rows * site.c * elem_bytes_of(&site.elem);
    gpu.ffma_per_dram_byte_pm()
        .is_some_and(|b| 2 * site.rows * site.c * site.k * 1000 < b * bytes)
}

/// **gemv-stream `vec-cells`: the adjacent cells one 16-byte load of `b`
/// covers** — `16 / e_b` lanes of one row, where the layout facts prove the
/// load: `b` unit-stride along the lane with its base, row and `k` strides
/// whole 16-byte granules of its storage, the row's lanes a whole number of
/// such vectors (so a thread's cells never cross a row and its output cells
/// are one aligned vector too), `a` the same for every lane (one `a` load a
/// `k` step serves every cell), plain affine reads, a streaming site. 1 (one
/// cell a thread) otherwise or with the rung off.
/// **The vector-cells kernel's structural constraint** (`naive_cells`): a
/// thread's `cells` consecutive lanes of one row read `b` as one vector at an
/// affine offset a `k` step (no mixed radix, no `k` split; `a` ignores the
/// lane, `b`'s lanes unit-stride), whole vectors at every row and step, and a
/// row holds whole groups of `cells`. The rules' [`vec_fit`] prices within it;
/// the placement search's lowering declares it.
pub(crate) fn naive_cells_admit(site: &TileSite, cells: u64) -> bool {
    let eb = site.b.elem_bytes(&site.elem);
    let whole = |x: u64| (x * eb).is_multiple_of((cells * eb).min(GPU_VEC_BYTES));
    site.radix.is_none()
        && site.a.ksplit.is_none()
        && site.b.ksplit.is_none()
        && site.a.clane == 0
        && site.b.clane == 1
        && whole(site.b.base)
        && whole(site.b.ci)
        && whole(site.b.ck)
        && site.c.is_multiple_of(cells)
}

pub(crate) fn vec_fit(site: &TileSite, gpu: &Gpu, rungs: &crate::GpuRungs) -> u64 {
    let eb = site.b.elem_bytes(&site.elem);
    let v = GPU_VEC_BYTES / eb.max(1);
    let whole = |x: u64| (x * eb).is_multiple_of(GPU_VEC_BYTES);
    let fits = rungs.vec_cells
        && v > 1
        && site.radix.is_none()
        && site.a.ksplit.is_none()
        && site.b.ksplit.is_none()
        && site.a.clane == 0
        && site.b.clane == 1
        && whole(site.b.base)
        && whole(site.b.ci)
        && whole(site.b.ck)
        && site.c.is_multiple_of(v)
        && streaming(site, gpu);
    if fits { v } else { 1 }
}

/// Bytes of one `mbarrier` (a stage's transaction barrier).
pub(crate) const BULK_BAR_BYTES: u64 = 8;

/// **TMA `bulk-stage`: the staging rule** — the warp TF32 `mma` tile's
/// panels are copied by the part's bulk-copy engine
/// ([`crate::profile::Caps::bulk`]) where its geometry covers every panel
/// row ([`crate::profile::BulkCopy::box_fits`]: an `a` row of
/// `bk` elements, a `b` row of `bn`, each row's global address and its
/// shared row (`lda`, `ldb` words apart) whole 16-byte granules), a stage's
/// bytes fit one barrier phase, the tile streams (`stages ≥ 2`) and its
/// barriers fit beside the stages. Plain affine reads only (a mixed-radix
/// operand copies element by element).
pub(crate) fn bulk_stage(
    site: &TileSite,
    g: &GpuMmaTile,
    gpu: &Gpu,
    rungs: &crate::GpuRungs,
) -> bool {
    let Some(bc) = gpu.caps.bulk.filter(|_| rungs.bulk_stage) else {
        return false;
    };
    let e = 4;
    let whole = |x: u64| (x * e).is_multiple_of(bc.granule);
    g.fmt == MmaFmt::Tf32
        && !g.bt
        && g.stages >= 2
        && site.radix.is_none()
        && site.a.ck == 1
        && site.b.clane == 1
        && bc.box_fits(1, g.bk, site.a.ci, site.a.base, e, None)
        && bc.box_fits(1, g.bn(), site.b.ck, site.b.base, e, None)
        && whole(g.lda())
        && whole(g.ldb())
        && whole(g.stage_words())
        && (g.bm() * g.bk + g.bk * g.bn()) * e <= bc.tx_bytes
        && g.smem_bytes() + BULK_BAR_BYTES * g.stages <= gpu.smem_per_block
}

/// hopper-wgmma-tma R1 for the warp-specialized `wgmma` kernel: **its two
/// operands as tensor boxes** — `a` as the dense `k × rows` tensor its
/// [`mapal_ir::TileRead::tensor_form`] proves (the `aᵀ` box: `bk` words of
/// `n` rows, the 128-byte swizzle the descriptor reads), `b` as the dense
/// `c × k` tensor (`bk` rows of one 32-word swizzle span, `bn / 32` boxes a
/// stage; under `digits` one 3-d box: the lane axis split into its digits
/// `32·q + r`, `q` outermost at 32 words, which lands those slabs in the same
/// bytes) — every box within the engine's geometry
/// ([`crate::profile::BulkCopy::tensor_box_fits`]). The recipe the runtime
/// encodes each map from (`mapal_gpu.c` `tmap_args`): `[arg, elem bytes,
/// rank, swizzle, base bytes, dims[5], byte strides[4], box[5], 0]` per map.
/// `None`: plain f32 reads only (a narrow, strided-lane, digit or window
/// operand keeps the S53 kernel).
pub(crate) fn ws_tensors(
    site: &TileSite,
    w: &GpuWgmmaTile,
    gpu: &Gpu,
    digits: bool,
) -> Option<[[u64; 20]; 2]> {
    use mapal_ir::TensorAxis as X;
    let bc = gpu.caps.bulk?;
    if site.radix.is_some() {
        return None;
    }
    // The TF32 form reads f32 storage only; wgmma-16bit's (`direct`) boxes
    // carry each operand at its staged width ([`GpuWgmmaTile::ea`] / `eb`).
    if !w.direct && (site.a.narrow.is_some() || site.b.narrow.is_some()) {
        return None;
    }
    let (ea, eb) = (w.ea, w.eb);
    let span = WGMMA_SWIZZLE_BYTES;
    let a = site.a.tensor_form(site.rows, site.c, site.k)?;
    let b = site.b.tensor_form(site.rows, site.c, site.k)?;
    let b_axes = match w.b_kmajor {
        true => [X::K, X::Lane],
        false => [X::Lane, X::K],
    };
    if a.axes() != [X::K, X::I] || b.axes() != b_axes {
        return None;
    }
    // A k-major box `bk` deep over `rows` rows: deeper than one swizzle row,
    // its k axis as the digits `k = r·q + p` (`q` outermost, `r` elements
    // apart), which lands the descriptor's atom columns `rows·128` bytes
    // apart.
    let kbox = |f: mapal_ir::TensorForm, other: X, rows: u64, e: u64| {
        let r = span / e;
        if w.bk > r {
            let (kd, id) = (f.dims[0], f.dims[1]);
            let split = mapal_ir::TensorForm {
                base: f.base,
                dims: vec![
                    mapal_ir::TensorDim { extent: r, ..kd },
                    id,
                    mapal_ir::TensorDim {
                        extent: kd.extent / r,
                        stride: r,
                        ..kd
                    },
                ],
            };
            let exact = kd.extent.is_multiple_of(w.bk) && id.extent.is_multiple_of(rows);
            (split, vec![r, rows, w.bk / r], exact)
        } else {
            let (sides, exact) = f.box_of(|x| if x == other { rows } else { w.bk });
            (f, sides, exact)
        }
    };
    let (a, asides, aexact) = kbox(a, X::I, w.bm(), ea);
    let dims = |f: &mapal_ir::TensorForm| {
        f.dims
            .iter()
            .map(|d| (d.extent, d.stride))
            .collect::<Vec<_>>()
    };
    let rb = span / eb;
    let (b, bsides, bexact) = match w.b_kmajor {
        true => kbox(b, X::Lane, w.bn(), eb),
        false => {
            let (sides, exact) = b.box_of(|x| if x == X::Lane { rb } else { w.bk });
            (b, sides, exact)
        }
    };
    let ok = aexact
        && bexact
        && bc.tensor_box_fits(&dims(&a), &asides, a.base, ea, Some(span))
        && bc.tensor_box_fits(&dims(&b), &bsides, b.base, eb, Some(span))
        && (w.b_kmajor || w.bn().is_multiple_of(rb));
    if !ok {
        return None;
    }
    // `tma-digits`: a lanes-major `b`'s lane axis `c = r·(c/r)` as its two
    // digits, the outer one last (`span` bytes apart): one box of all the
    // stage's slabs.
    let b = match digits && !w.b_kmajor {
        true => {
            let (lane, kd) = (b.dims[0], b.dims[1]);
            let split = mapal_ir::TensorForm {
                base: b.base,
                dims: vec![
                    mapal_ir::TensorDim { extent: rb, ..lane },
                    kd,
                    mapal_ir::TensorDim {
                        extent: lane.extent / rb,
                        stride: rb,
                        ..lane
                    },
                ],
            };
            let sides = [rb, w.bk, w.bn() / rb];
            if !bc.tensor_box_fits(&dims(&split), &sides, split.base, eb, Some(span)) {
                return None;
            }
            (split, sides.to_vec())
        }
        false => (b, bsides),
    };
    let (b, bsides) = b;
    let rec = |arg: u64, f: &mapal_ir::TensorForm, sides: &[u64], e: u64| {
        let mut r = [0u64; 20];
        r[0] = arg;
        r[1] = e;
        r[2] = f.dims.len() as u64;
        r[3] = span;
        r[4] = f.base * e;
        for (i, d) in f.dims.iter().enumerate() {
            r[5 + i] = d.extent;
            if i > 0 {
                r[9 + i] = d.stride * e;
            }
            r[14 + i] = sides[i];
        }
        r
    };
    Some([rec(0, &a, &asides, ea), rec(1, &b, &bsides, eb)])
}

/// The one-thread-per-cell grid at `cells` cells a thread.
pub(crate) fn cell_blocks(site: &TileSite, cells: u64) -> u64 {
    (site.rows * site.c).div_ceil(NAIVE_THREADS * cells)
}

/// gemv-stream `vec-cells`: the trips of `NAIVE_UNROLL` loads a thread's
/// partition must keep for the vector cells — between the RTX 5080's loser
/// (1 × 1024 × 4096: the 128 partitions its one vector block needs leave 4
/// trips, 0.87× the scalar cells at 32) and its winners (16–128 trips:
/// 1.03–1.42×); at 8 trips null (1 × 4096 × 4096 bf16, 1.03×). A partition
/// that short spends its time in its ramp and its adds, not its stream.
pub(crate) const VEC_MIN_TRIPS: u64 = 8;

/// Whether a one-thread-per-cell kernel folding `s` partitions takes `v`
/// cells a thread: its vector grid puts a block on every SM at `s`, each
/// partition keeps [`VEC_MIN_TRIPS`] trips, and (`vec-spread`) the grid
/// spreads over the SMs as the part's DRAM share asks
/// ([`crate::rules::vec_cells_pay`] against one cell a thread at the same
/// `s`).
pub(super) fn vec_takes(
    site: &TileSite,
    gpu: &Gpu,
    rungs: &crate::GpuRungs,
    v: u64,
    s: u64,
) -> bool {
    let s = s.max(1);
    v > 1
        && cell_blocks(site, v) * s >= gpu.sms
        && site.k / s >= VEC_MIN_TRIPS * NAIVE_UNROLL
        && (!rungs.vec_spread
            || crate::rules::vec_cells_pay(gpu, cell_blocks(site, v) * s, cell_blocks(site, 1) * s))
}

/// gemv-stream `stream-fly`: the split of a streaming one-thread-per-cell
/// site — [`TargetProfile::split_stream`] on the vector grid where its cells
/// are taken at that split ([`vec_takes`]), else on one cell a thread (a
/// thread keeps `NAIVE_UNROLL` loads of `cells · e_b` bytes of `b` in flight).
#[allow(clippy::too_many_arguments)]
pub(super) fn stream_split(
    site: &TileSite,
    profile: &TargetProfile,
    gpu: &Gpu,
    rungs: &crate::GpuRungs,
    wave: u64,
    panels: u64,
    e: u64,
    workspace: bool,
) -> u64 {
    let eb = site.b.elem_bytes(&site.elem);
    let at = |v: u64| {
        let fly = NAIVE_THREADS * NAIVE_UNROLL * v * eb;
        profile.split_stream(
            cell_blocks(site, v),
            fly,
            wave,
            panels,
            site.rows * site.c,
            e,
            workspace,
        )
    };
    let v = vec_fit(site, gpu, rungs);
    match v > 1 {
        true => Some(at(v)).filter(|&s| vec_takes(site, gpu, rungs, v, s)),
        false => None,
    }
    .unwrap_or_else(|| at(1))
}
