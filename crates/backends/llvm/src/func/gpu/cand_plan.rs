//! The chooser's candidates: `Cand`, `plan_cands`, `plan_shape`, `plan_pick` and resized launches.

use super::*;

/// One candidate plan of a site's chooser: its fold realization, the launch
/// it emits and the cost model's reading of both (plan-cost §2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cand {
    pub combine: Combine,
    pub shape: PlanShape,
    pub cost: PlanCost,
}

impl Cand {
    /// The candidate's name on the plan line.
    pub(crate) fn label(&self) -> String {
        match self.combine {
            Combine::None => "none".into(),
            Combine::Atomic { s } => format!("atomic{s}"),
            Combine::Workspace { s } => format!("splitk{s}"),
            Combine::StreamK { ctas, iters, .. } => format!("streamk{ctas}x{iters}"),
        }
    }
}

/// **plan-cost: the chooser.** On a part with calibrated plan rates
/// ([`TargetProfile::plan_rates`]), a free-order scalar
/// site's candidates — the ordered kernel, the atomic and the workspace split
/// at every `S` dividing the panels, stream-K, each at the geometry its split
/// grid takes ([`resized`]), each admitted by its own rung as in [`combine`],
/// and the rules' own pick — ranked by [`TargetProfile::site_ns`], cheapest
/// first (a tie keeps the earlier: the ordered kernel, then the rules').
/// A forced count leaves only its own splits. Empty where the rules decide:
/// another order, a part without the rates, the tensor-core kernels, a window
/// (whose `k` is never cut) and a forced count no split admits. [`launch_in`]
/// takes the first under rung `plan-cost`; its OFF arm keeps the rules' plan
/// and still prints the ranking (the fit harness reads every arm's).
pub(crate) fn plan_cands(
    l: &Launch,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> Vec<Cand> {
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || radix_whole(site);
    let rungs = l.rungs;
    if order != Order::Free
        || profile.plan_rates().is_none()
        || l.mma.is_some()
        || l.wgmma.is_some()
        || window
    {
        return Vec::new();
    }
    let gpu = profile.gpu().expect("a GPU site has a GPU profile");
    let e = elem_bytes_of(&site.elem);
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let blocks = l.gx * l.gy;
    let wave = l.wave(gpu);
    let atomic = e == 4 || gpu.caps.red_f64;
    let forced = if arms.splitk > 0 {
        arms.splitk
    } else {
        gpu.geom.split
    };
    // A forced count (`--splitk=S`, the part's `Geom::split`) is the only
    // plan: the ordered kernel and the rules' pick are not candidates.
    let mut combos = match forced {
        0 => vec![
            Combine::None,
            combine(l, site, profile, contract, order, arms),
        ],
        _ => Vec::new(),
    };
    for s in (2..=panels.min(64)).filter(|&s| panels.is_multiple_of(s)) {
        if forced > 0 && s != forced {
            continue;
        }
        if rungs.splitk_atomic && atomic && l.raster == 1 {
            combos.push(Combine::Atomic { s });
        }
        if rungs.splitk && l.raster == 1 && s * site.rows * site.c * e <= gpu.l2_bytes {
            combos.push(Combine::Workspace { s });
        }
    }
    // Stream-K at the rules' one wave (see [`combine`]).
    if forced == 0
        && rungs.streamk
        && atomic
        && l.tiled.is_some_and(|g| g.stages < 2)
        && panels >= 2
        && blocks * panels > wave
    {
        let iters = (blocks * panels).div_ceil(wave);
        combos.push(Combine::StreamK {
            ctas: (blocks * panels).div_ceil(iters),
            iters,
            whole: 0,
        });
    }
    let mut seen = Vec::new();
    let mut cands: Vec<Cand> = combos
        .into_iter()
        .filter(|c| {
            let new = !seen.contains(c);
            seen.push(*c);
            new
        })
        .filter_map(|c| {
            let r = resized(site, profile, contract, arms, *l, c);
            let shape = plan_shape(site, &r, c, gpu);
            let cost = profile.site_ns(&shape)?;
            Some(Cand {
                combine: c,
                shape,
                cost,
            })
        })
        .collect();
    cands.sort_by(|a, b| a.cost.ns.total_cmp(&b.cost.ns));
    cands
}

/// The cost model's reading of launch `l` folding the site as `c`.
pub(crate) fn plan_shape(site: &TileSite, l: &Launch, c: Combine, gpu: &Gpu) -> PlanShape {
    // A cp.async pipeline keeps `stages − 1` panels in flight; the
    // register-staged tile its one prefetch panel; the naive kernel one
    // unrolled trip.
    let (bm, bn, acc, per_sm, bk, step, fly, lsu) = match l.tiled {
        Some(g) => (
            g.bm(),
            g.bn(),
            g.tm * g.tn,
            g.blocks_per_sm,
            g.bk,
            g.bk,
            g.stages.saturating_sub(1).max(1),
            lsu_per_panel(site, &g, &l.rungs, false),
        ),
        None => (
            1,
            l.block * l.cells,
            l.cells,
            l.wave(gpu) / gpu.sms,
            1,
            NAIVE_UNROLL,
            1,
            0,
        ),
    };
    let tiles = l.gx * l.gy;
    let (blocks, kb, s, comb) = match c {
        Combine::None => (tiles, site.k, 1, PlanComb::None),
        Combine::Atomic { s } => (tiles * s, site.k / s, s, PlanComb::Atomic),
        Combine::Workspace { s } => (tiles * s, site.k / s, s, PlanComb::Workspace),
        Combine::StreamK { ctas, iters, .. } => (ctas, iters * bk, 1, PlanComb::StreamK),
    };
    PlanShape {
        rows: site.rows,
        c: site.c,
        k: site.k,
        e: elem_bytes_of(&site.elem),
        ea: site.a.elem_bytes(&site.elem),
        eb: site.b.elem_bytes(&site.elem),
        bm,
        bn,
        threads: l.block,
        per_sm,
        acc,
        tiles,
        blocks,
        kb,
        s,
        comb,
        bk: step,
        fly,
        lsu,
    }
}

/// plan-cost: what a site's plan line adds — ` pick= ns= alt=<cand>:<ns>
/// cross=x<scale>` (the runner-up and the scale where it takes over,
/// [`TargetProfile::crossover`]) — and, under `--plan-cands` on a row-major
/// kernel of the site (`cands_here`), one `; mapal.cand` line per candidate,
/// keyed by the site's ordered [`kernel_id`].
/// Empty where the rules decide.
pub(super) fn plan_pick(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    cands_here: bool,
) -> String {
    let order = if arms.free {
        Order::Free
    } else {
        Order::Sequential
    };
    let l = launch(site, profile, contract, arms);
    let cands = plan_cands(&l, site, profile, contract, arms, order);
    let Some(pick) = cands.first() else {
        return String::new();
    };
    let id = kernel_id(site, profile, contract);
    let mut f = format!(
        " site={id:016x} pick={} ns={:.1}",
        pick.label(),
        pick.cost.ns
    );
    match cands.get(1) {
        Some(alt) => {
            let cross = profile
                .crossover(&pick.shape, &alt.shape)
                .map_or("none".into(), |x| format!("x{x}"));
            f += &format!(" alt={}:{:.1} cross={cross}", alt.label(), alt.cost.ns);
        }
        None => f += " alt=none cross=none",
    }
    if arms.cands && cands_here {
        // The rates the candidates were costed at (the harness's mirror).
        let g = profile.gpu().expect("a GPU site has a GPU profile");
        f += "\n; mapal.rates";
        for name in crate::profile::DeviceCost::NAMES {
            f += &format!(" {name}={}", g.rate(name));
        }
        f += &format!(
            " sms={} warp={} schedulers={}\n",
            g.sms,
            g.warp,
            g.schedulers()
        );
        for (i, c) in cands.iter().enumerate() {
            let (p, t) = (c.shape, c.cost);
            f += &format!(
                "; mapal.cand site={:016x} cand={} pick={} ns={:.1} rows={} c={} k={} e={} \
                 bm={} bn={} threads={} per_sm={} acc={} tiles={} blocks={} kb={} s={} \
                 bk={} fly={} lsu={} ffma_lsu={:.2} waves={} tail={} wps={:.3} f={:.1} l={:.1} \
                 d={:.1} chain={:.1} u={:.1} kernel={:.1} combine={:.1} fixed={:.1} launches={} \
                 zeros={} atomics={}",
                id,
                c.label(),
                u8::from(i == 0),
                t.ns,
                p.rows,
                p.c,
                p.k,
                p.e,
                p.bm,
                p.bn,
                p.threads,
                p.per_sm,
                p.acc,
                p.tiles,
                p.blocks,
                p.kb,
                p.s,
                p.bk,
                p.fly,
                p.lsu,
                (p.acc * p.bk) as f64 / p.lsu.max(1) as f64,
                t.waves,
                t.tail,
                t.wps,
                t.f_ns,
                t.l_ns,
                t.d_ns,
                t.c_ns,
                t.u_ns,
                t.kernel_ns,
                t.combine_ns,
                t.fixed_ns,
                t.launches,
                t.zeros,
                t.atomics
            );
            if i + 1 < cands.len() {
                f.push('\n');
            }
        }
    }
    f
}

/// `l` re-sized for the split grid of `combine` under `fill-wave`
/// ([`launch_split`]) — kept only when the re-sized panel still cuts each
/// partition's `k / S` whole (the split kernel walks `kext / bk` panels).
pub(super) fn resized(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    l: Launch,
    combine: Combine,
) -> Launch {
    let r = resized_rules(site, profile, contract, arms, l, combine);
    // ADR-0050 C4 `machine-place`: the kernel of realization `combine` runs
    // the machine model's best schedule of it (the pick, or the best
    // ordered one for the ordered kernel beside it).
    let r = match r.rungs.machine_place {
        true => crate::place::placed_for(site, profile, contract, arms, combine, r),
        false => r,
    };
    crate::place::probe(site, profile, contract, arms, r)
}

pub(super) fn resized_rules(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    l: Launch,
    combine: Combine,
) -> Launch {
    let s = combine.splits();
    let whole = |r: &Launch| r.tiled.is_some_and(|g| (site.k / s).is_multiple_of(g.bk));
    // large-gemm `k-fill`: the stream-K [`combine`] took over the full tile
    // runs that tile.
    if let Combine::StreamK { .. } = combine
        && let Some((c, full)) = k_fill(&l, site, profile, contract, arms)
        && c == combine
    {
        return full;
    }
    // bf16 tensor-tile `mma-walk`: the walk [`mma_combine`] took runs its
    // whole warp tile.
    if let Combine::StreamK { .. } = combine
        && let Some((c, walk)) = mma_walk(&l, site, profile, arms, Order::Free)
        && c == combine
    {
        return walk;
    }
    let rule = match l.rungs.fill_wave && s > 1 {
        true => Some(launch_split(site, profile, contract, arms, s, false))
            .filter(whole)
            .unwrap_or(l),
        false => l,
    };
    // tile-efficiency T1: where the plan is the cost model's (`plan-cost`, a
    // part with its rates) and the order free, a split kernel may fill the
    // machine along `k` with the full register tile instead — a candidate
    // beside the rules' (shrunk) tile at the same `S`, taken where
    // [`TargetProfile::site_ns`] says it is cheaper (never a smaller tile).
    let full = s > 1 && arms.free && l.rungs.full_tile && l.rungs.plan_cost && l.tiled.is_some();
    let Some(gpu) = profile.gpu().filter(|_| full) else {
        return split_cells(site, profile, rule, s);
    };
    let r = launch_split(site, profile, contract, arms, s, true);
    let acc = |x: &Launch| x.tiled.map_or(0, |g| g.tm * g.tn);
    let ns = |x: &Launch| {
        profile
            .site_ns(&plan_shape(site, x, combine, gpu))
            .map(|c| c.ns)
    };
    // per-site-choice: `arms.full` (a site candidate) takes it wherever it
    // divides — the cost model then ranks the full-tile plans.
    match (ns(&r), ns(&rule)) {
        (Some(a), Some(b)) if whole(&r) && acc(&r) > acc(&rule) && (a < b || arms.full) => r,
        _ => rule,
    }
}

/// gemv-stream `vec-cells`: a one-thread-per-cell kernel folding `s`
/// partitions takes the vector cells ([`vec_fit`]) where its split grid
/// puts a block on every SM (the ordered kernel's rule, [`launch_split`],
/// at `s` times the blocks); every other launch is `l`.
pub(super) fn split_cells(site: &TileSite, profile: &TargetProfile, l: Launch, s: u64) -> Launch {
    let naive = l.tiled.is_none() && l.mma.is_none() && l.wgmma.is_none();
    let Some(gpu) = profile.gpu().filter(|_| naive && l.cells == 1 && s > 1) else {
        return l;
    };
    let v = vec_fit(site, gpu, &l.rungs);
    match vec_takes(site, gpu, &l.rungs, v, s) {
        true => Launch {
            cells: v,
            gx: cell_blocks(site, v),
            ..l
        },
        false => l,
    }
}

/// The launch a kernel of realization `combine` is emitted with: under
/// `fill-wave` a split kernel's tile is the split grid's ([`launch_split`]).
pub(crate) fn launch_for(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    combine: Combine,
) -> Launch {
    resized(
        site,
        profile,
        contract,
        arms,
        launch(site, profile, contract, arms),
        combine,
    )
}
