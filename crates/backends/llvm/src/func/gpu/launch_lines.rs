//! The launch's plan line, the kernel rung table, `launch_in` and the k-fill rule.

use super::*;

/// The grid a site's fill is keyed by (`launch`'s `wps`): the S52 geometry
/// (no S53 geometry rung) — the table's rows were measured on grids
/// classified by it — or the one-thread-per-cell grid.
pub(crate) fn fill_grid(site: &TileSite, profile: &TargetProfile) -> (u64, u64) {
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    let base = shaped
        .then(|| profile.gpu_tile(site.rows, site.c, site.k, &site.elem, false, false))
        .flatten();
    match base {
        Some(g) => ((site.rows / g.bm()) * (site.c / g.bn()), g.threads()),
        None => ((site.rows * site.c).div_ceil(NAIVE_THREADS), NAIVE_THREADS),
    }
}

/// S64 evidence pass: one `; mapal.plan` comment line per kernel — what the
/// emitter derived for it (geometry, fill key, face, index width, the
/// kernel's rungs), the intent `benches/gpu/evidence.py` checks the PTX and
/// SASS against. A comment: `llc` drops it, the kernel is unchanged. Only the
/// kernel rungs are listed, so a host rung's arm leaves the device text alone.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_line(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    l: &Launch,
    idx: &str,
    combine: Combine,
    col: bool,
    epi: bool,
    walk: Option<&str>,
) -> String {
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let (blocks, threads) = fill_grid(site, profile);
    let kind = match (l.wgmma, l.mma, l.tiled) {
        (Some(_), _, _) => "wgmma",
        (None, Some(_), _) => "mma",
        (None, None, Some(_)) => "tile",
        _ => "naive",
    };
    let mut f = format!(
        "; mapal.plan kernel={name} kind={kind} face={:?} idx={idx} rows={} c={} k={} wps={} \
         gx={} gy={} block={} smem={} raster={}",
        l.arithmetic,
        site.rows,
        site.c,
        site.k,
        gpu.warps_per_scheduler(blocks, threads),
        l.gx,
        l.gy,
        l.block,
        l.smem,
        l.raster
    );
    // half-precision (ADR-0043): the operands loaded in a narrow format (intent.py `narrow-tile`).
    let narrow: String = [("a", site.a.narrow), ("b", site.b.narrow)]
        .iter()
        .filter_map(|(o, n)| n.map(|n| format!("{o}:{}", n.name())))
        .collect::<Vec<_>>()
        .join("+");
    if !narrow.is_empty() {
        f += &format!(" narrow={narrow}");
    }
    if let Some(g) = l.tiled {
        f += &format!(
            " tm={} tn={} tx={} ty={} bk={} stages={} pad_a={} held={} blocks_per_sm={} lsu={} \
             lsu_issue={} kmajor={} adeg={}",
            g.tm,
            g.tn,
            g.tx,
            g.ty,
            g.bk,
            g.stages,
            g.pad_a,
            g.held,
            g.blocks_per_sm,
            lsu_per_panel(site, &g, &l.rungs, col),
            // per-site-choice D4: the instructions alone, what SASS shows
            // (intent.py `lsu`).
            lsu_per_panel(
                site,
                &g,
                &crate::GpuRungs {
                    bank_plan: false,
                    ..l.rungs
                },
                col
            ),
            u8::from(g.kmajor && !col),
            // per-site-choice A5: the `a` panel's worst bank excess, fragment
            // read and copy write (intent.py `kmajor-a`: 1).
            {
                let [a, _, st, _] = panel_excess(site, &g, &l.rungs, col);
                a.max(st)
            }
        );
        // sweep-attention-2 `kmajor-b` (intent.py): `b`'s copies along its
        // k-major rows.
        if l.rungs.kmajor_b && kvec_b(site, &g) {
            f += " kvb=1";
            // gapfix `kvec-swizzle` (intent.py): the copies' bank groups.
            if let Some((gs, st)) = kvec_swizzle(site, &g, &l.rungs) {
                f += &format!(" swz={gs}x{st}");
            }
        }
    }
    if let Some(m) = l.mma {
        f += &format!(
            " mt={} nt={} warps_x={} warps_y={} bk={} stages={} blocks_per_sm={}",
            m.mt, m.nt, m.warps_x, m.warps_y, m.bk, m.stages, m.blocks_per_sm
        );
        // tensor-cores: the instruction and each panel's storage bytes
        // (intent.py `half-tile`); the TF32 line unchanged.
        if m.fmt != MmaFmt::Tf32 {
            f += &format!(
                " op=m16n8k{}.{} ea={} eb={}",
                m.mma_k(),
                m.fmt.ptx(),
                m.ea,
                m.eb
            );
            if m.bt {
                f += " bt=1";
            }
        }
        // TMA `bulk-stage` (intent.py): the panels the bulk-copy engine stages.
        if bulk_stage(site, &m, gpu, &l.rungs) {
            f += " stage=bulk";
        }
        // f32-tensor-cores `split-tile` (intent.py): the split form.
        if let Some(k) = m.split {
            f += &format!(" split={}", split_name(k));
            // tensor-cores-2 `split-issue` (intent.py): the k-steps a zeroed
            // chain carries before its flush, and the issue ratio priced.
            let flush = match l.rungs.split_issue {
                true => split_flush(&m, site.k, combine.splits(), gpu),
                false => 1,
            };
            let (issue_pm, slots_pm) = split_cost(site, gpu, &m, k, l.rungs.split_issue);
            f += &format!(" flush={flush} issue_pm={issue_pm} slots_pm={slots_pm}");
        }
    }
    if let Some(w) = l.wgmma {
        f += &format!(
            " n={} groups={} bk={} stages={} blocks_per_sm={}",
            w.n, w.groups, w.bk, w.stages, w.blocks_per_sm
        );
        // hopper-wgmma-tma (intent.py `wgmma-ws`): the producer warpgroup,
        // the tensor boxes a stage, the registers each side keeps.
        if w.ws {
            let b = if l.rungs.tma_digits {
                1
            } else {
                w.ws_b_boxes()
            };
            f += &format!(" ws=1 stage=tma boxes={}", 1 + b);
            // wgmma-16bit (intent.py `wgmma16`): the instruction, each
            // operand's staged bytes, `a`'s home, `b`'s major.
            if w.direct {
                f += &format!(
                    " op=m64n{}k{}.{} ea={} eb={} a_home={} bt={}",
                    w.n,
                    w.mma_k(),
                    w.fmt.ptx(),
                    w.ea,
                    w.eb,
                    match w.a_home {
                        Home::Smem => "smem",
                        _ => "regs",
                    },
                    u8::from(w.b_kmajor)
                );
            }
            if let Some((p, c)) = w.realloc {
                f += &format!(" regs={p}/{c}");
            }
            // hopper-launch (intent.py `ws-persist`): the walk's CTAs and
            // its tiles.
            if w.ctas > 0 {
                let tiles = (site.rows / w.bm()) * (site.c / w.bn());
                f += &format!(" persist={} tiles={tiles}", w.ctas);
            }
        }
    }
    // generic-fusion `window`: a window site's tap grid, `(k÷div) × div`
    // (intent.py `window`); no other kernel's line changes.
    if let Some(ks) = site.a.ksplit.as_ref().or(site.b.ksplit.as_ref()) {
        f += &format!(" taps={}x{}", site.k / ks.div, ks.div);
    }
    // recurrence-precision `radix-tile`: which addresses are mixed-radix
    // (intent.py `radix`); no other kernel's line changes.
    if let Some(r) = site.radix.as_deref() {
        let parts = [("a", &r.a), ("b", &r.b), ("out", &r.out)];
        let on: Vec<&str> = parts
            .iter()
            .filter(|(_, x)| x.is_some())
            .map(|(n, _)| *n)
            .collect();
        f += &format!(" radix={}", on.join(","));
        // long-prefill `uniform-base`: the operands whose pointer the launch
        // moves by a captured number (intent.py `uniform-base`).
        let moved: Vec<&str> = [("a", &r.ua), ("b", &r.ub)]
            .iter()
            .filter(|(_, ds)| !ds.is_empty())
            .map(|(n, _)| *n)
            .collect();
        if !moved.is_empty() {
            f += &format!(" ubase={}", moved.join(","));
        }
    }
    // device-geometry `head-group`: the rows one `b` panel serves (a KV
    // head's query heads), where fewer than the site's (intent.py
    // `head-group`); `tile-post`: the store evaluates the body after the fold.
    if row_group(site) < site.rows {
        f += &format!(" group={}", row_group(site));
    }
    if site.post.is_some() {
        f += " post=1";
    }
    // recurrence-precision `tf32-tile`: the site's product set when it admits
    // TF32 (ADR-0041 D5 `set`, intent.py `tf32-tile`); f32-only lines unchanged.
    if let Some(n) = site.half {
        f += &format!(
            " set={}{}|f32",
            n.name(),
            if site.tf32 { "|tf32" } else { "" }
        );
    } else if site.tf32 {
        f += " set=tf32|f32";
    }
    let (c, s, ctas, iters) = match combine {
        Combine::None => ("none", 1, 0, 0),
        Combine::Workspace { s } => ("splitk", s, 0, 0),
        Combine::Atomic { s } => ("atomic", s, 0, 0),
        Combine::StreamK { ctas, iters, .. } => ("streamk", 1, ctas, iters),
    };
    // ADR-0050 C5: a hybrid's whole-tile blocks ahead of its walk.
    if let Combine::StreamK { whole, .. } = combine
        && whole > 0
    {
        f += &format!(" whole={whole}");
    }
    // sweep-gemm-2: which rule walks a stream-K site (intent.py) — the class
    // that ships `streamk`, `k-fill`'s full tile, or `streamk-price`.
    if let Some(w) = walk {
        f += &format!(" walk={w}");
    }
    // site-geometry `fill-wave`: the target its split grid reaches (intent.py).
    if l.rungs.fill_wave && s > 1 {
        f += &format!(" fill_warps={}", gpu.fill_warps());
    }
    // gemv-stream (intent.py `vec-cells`, `stream-fly`): a streaming
    // one-thread-per-cell kernel's cells a thread, the bytes of `b` its
    // launched grid keeps in flight per SM it uses (`NAIVE_UNROLL` loads of
    // `cells · e_b` a thread), and the part's knee.
    if kind == "naive" && streaming(site, gpu) {
        let grid = l.gx * l.gy * s;
        let fly = grid * NAIVE_THREADS * NAIVE_UNROLL * l.cells * site.b.elem_bytes(&site.elem);
        f += &format!(
            " cells={} fly_sm={} knee={}",
            l.cells,
            fly / grid.min(gpu.sms).max(1),
            gpu.stream_fly_per_sm()
        );
    }
    let on: Vec<&str> = KERNEL_RUNGS
        .iter()
        .copied()
        .filter(|n| l.rungs.get(n))
        .collect();
    format!(
        "{f} combine={c} S={s} ctas={ctas} iters={iters} col={} epi={} rungs_on={}\n",
        u8::from(col),
        u8::from(epi),
        on.join(",")
    )
}

/// The rungs a kernel's text reads ([`plan_line`]); the rest are host rungs.
pub(super) const KERNEL_RUNGS: [&str; 49] = [
    "arch-rungs",
    "interleave",
    "unroll",
    "min-blocks",
    "warp",
    "prefetch",
    "mma",
    "async",
    "wide",
    "ldmatrix",
    "warp-fill",
    "idx32",
    "affine",
    "pad",
    "wide-fma",
    "wgmma-hopper",
    "raster",
    "l2-gate",
    "pdl",
    "row-fit",
    "wave-gate",
    "grid-share",
    "panel-regs",
    "fill-rows",
    "cell-fill",
    "asyncstage",
    "splitk",
    "splitk-atomic",
    "streamk",
    "row-split",
    "fill-wave",
    "fit-first",
    "tf32-radix",
    "kmajor-a",
    "full-tile",
    "async-depth",
    "bank-plan",
    "streamk-fill",
    "k-fill",
    "vec-cells",
    "stream-fly",
    "bulk-stage",
    "wgmma-ws",
    "ws-regs",
    "tma-digits",
    "kmajor-b",
    "stream-ld",
    "kvec-swizzle",
    "wgmma16",
];

/// [`launch`] with the fold's realization chosen under `order` (ADR-0040 D4).
/// ADR-0050 C4 `machine-place`: the machine model's pick
/// ([`crate::place::placed`]), the rules' launch its incumbent.
pub(crate) fn launch_in(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> Launch {
    let l = crate::place::rules(|| launch_in_rules(site, profile, contract, arms, order));
    match l.rungs.machine_place {
        true => crate::place::placed(site, profile, contract, arms, order, l),
        false => l,
    }
}

/// [`launch_in`]'s rule chain (the OFF arm of `machine-place`, its
/// incumbent).
pub(crate) fn launch_in_rules(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> Launch {
    let l = launch(site, profile, contract, arms);
    let cands = match l.rungs.plan_cost {
        true => plan_cands(&l, site, profile, contract, arms, order),
        false => Vec::new(),
    };
    let c = match cands.first() {
        Some(pick) => pick.combine,
        None => combine(&l, site, profile, contract, order, arms),
    };
    let mut l = resized(site, profile, contract, arms, l, c);
    l.combine = c;
    l
}

/// **large-gemm `k-fill`: an under-filled site keeps the full register tile
/// and fills the part along `k` with stream-K over the slots the tile
/// holds.** Where the rules' launch `l` shrank the tile or its threads to
/// fill the part (`warp-fill`, the fill rule's halving: fewer accumulators
/// a thread or fewer threads than the full S52 tile,
/// [`TargetProfile::gpu_tile_split`] under `full-tile`), the full tile's
/// grid is walked by stream-K at `sms × residency` CTAs — the residency at
/// the tile's own registers and threads ([`TargetProfile::tile_capacity`]),
/// not `grid-share`'s one block an SM, which leaves an SM's second slot
/// empty (128 × 6144 × 4096: 84 CTAs 0.231 ms, 167 CTAs 0.205) — where each
/// CTA folds at least [`K_FILL_PANELS`] panels. A shrunk tile issues more
/// shared loads per FFMA (`4 × 4`: 2 `LDS.128` per 16 FFMA, against 4 per
/// 64 at `8 × 8`), which bounds it at the part's shared-load rate; a CTA's
/// stream-K fixup (the atomic adds of at most two tile pieces) is the cost
/// on the other side, paid back over a long enough run (rtx-5080: 128 × 4096
/// × 4096 0.148 ms against 0.219 shrunk, 128 × 6144 × 4096 0.205 against
/// 0.324; 1024³ at 24 panels a CTA 0.098 against 0.088 shrunk — the shrunk
/// tile stays). Free order only (the fixup adds). The combine and the launch
/// its kernel runs ([`combine`] takes the first, [`resized`] the second).
pub(super) fn k_fill(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> Option<(Combine, Launch)> {
    let r = l.rungs;
    let gpu = profile.gpu()?;
    let g = l.tiled?;
    let e = elem_bytes_of(&site.elem);
    let atomic = e == 4 || gpu.caps.red_f64;
    // (A whole-file tile is no shrunk one: it is taken only where its own
    // grid fills the part, `streamk-fill` in [`TargetProfile::gpu_tile_split`].)
    if !r.k_fill
        || !r.streamk_fill
        || !atomic
        || g.held > 0
        || arms.splitk > 0
        || gpu.geom.split > 0
    {
        return None;
    }
    // The full S52 tile (its `bm`, `bn`, threads), not the whole-file
    // `wide-fma` tile, which halves a block's warps where the grid is short
    // of them (128 × 4096 × 4096: `8 × 16` at `S = 8` 0.174 ms against
    // `8 × 8` at `S = 4` 0.162).
    let mut a = arms;
    *a.off.rung("wide-fma")? = true;
    let fg = launch_split(site, profile, contract, a, 2, true).tiled?;
    let side = profile.fma_tile_side(&site.elem)?;
    let full_side = fg.tm == side && fg.tn == side;
    if !full_side || fg.tm * fg.tn <= g.tm * g.tn && fg.threads() <= g.threads() || fg.stages >= 2 {
        return None;
    }
    let tiles = (site.rows / fg.bm()) * (site.c / fg.bn());
    let cap = gpu.sms * profile.tile_capacity(&fg, &site.elem);
    if tiles >= cap {
        return None;
    }
    // Sized for `cap` blocks: the panel share of the residency it runs at.
    let full = launch_split(site, profile, contract, a, cap.div_ceil(tiles), true);
    let t = full.tiled?;
    let panels = site.k / t.bk;
    let wave = gpu.sms * t.blocks_per_sm;
    let iters = (tiles * panels).div_ceil(wave);
    let floor = match gpu.geom.kfill {
        0 => K_FILL_PANELS,
        f => f,
    };
    if t.tm * t.tn != fg.tm * fg.tn || t.stages >= 2 || iters < floor || panels < 2 {
        return None;
    }
    let c = Combine::StreamK {
        ctas: (tiles * panels).div_ceil(iters),
        iters,
        whole: 0,
    };
    // The launch fills the part: the filled class's index width (`idx32`,
    // `contract && filled`) — the walk over a tile count that is no power of
    // two divides in `i32` (128 × 6144 × 4096 spilled 24 B at `i64`).
    let idx32 = rungs_at(profile, GPU_RESIDENT_BLOCKS, Face::of(contract, None), arms).idx32;
    let rungs = crate::GpuRungs {
        idx32,
        ..full.rungs
    };
    Some((
        c,
        Launch {
            combine: c,
            rungs,
            ..full
        },
    ))
}

/// large-gemm `k-fill`: the fewest panels a stream-K CTA of the full tile
/// folds. Between the rtx-5080's measured loser (1024³: 24.4 panels a CTA,
/// 0.098 ms against the shrunk tile's 0.088) and winners (128 × 4096 ×
/// 4096: 48.8, 0.148 against 0.219; 128 × 6144 × 4096: 73.6, 0.205
/// against 0.324): a CTA's fixup — the atomic adds of its two tile pieces,
/// `2 · bm · bn` at the part's per-SM atomic rate — costs about two and a
/// half panels of its FFMA time there, so the run must be long against it.
pub(crate) const K_FILL_PANELS: u64 = 32;
