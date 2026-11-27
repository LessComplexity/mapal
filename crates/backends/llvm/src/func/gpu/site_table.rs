//! The site table: candidate kernels, ids, recipes and the concatenated plan.

use super::*;

/// per-site-choice B2: a site's candidate table (`mapal_choice.h`'s
/// `struct mapal_cand`) and the one call that runs its pick. Declared only
/// beside a call, so a module without a table keeps its text.
pub(crate) const SITE_DECL: &str = "%mapal_cand = type { i64, i64, i64, i64, i32, i32, i32, i32, i32, i32 }\n\
     declare void @mapal_gpu_site(i64, i32, i32, i32, ptr, i32, ptr, i32, ptr)\n";

/// `mapal_gpu_site`'s flag: running the site again is value-safe.
pub(super) const SITE_RERUN: u32 = 1;

/// **per-site-choice B1: one candidate of a site's table** — the launch it
/// realizes, the arms its kernels are emitted under, and the axes flipped
/// against candidate 0 (`rules` for candidate 0 itself).
#[derive(Clone, Debug)]
pub(crate) struct SiteCand {
    pub arms: crate::GpuArms,
    pub launch: Launch,
    pub axes: String,
}

/// per-site-choice B2: a site's table on its way to the launch point
/// (`emit_site_resident`, `emit_row_resident`), which emits it through
/// [`FnEmit::site_call`] in place of the single launch.
#[derive(Clone, Debug)]
pub(crate) struct SiteTable {
    /// Candidate 0's main kernel id.
    pub site: u64,
    pub flags: u32,
    /// One `%mapal_cand` constant per candidate.
    pub rows: Vec<String>,
    /// The largest workspace a candidate takes (0: none).
    pub ws: u64,
}

/// The launch a site realizes under `arms`: [`launch_in`]'s. A retained
/// row's one-row pass (`row`) cuts its fold only under `row-split` — a
/// workspace only on a free order (a fixed face replays one count per fold) —
/// and launches unsplit otherwise.
pub(super) fn realized(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
) -> Launch {
    let l = launch_in(site, profile, contract, arms, order);
    let split = match l.combine {
        _ if !row => true,
        Combine::None => false,
        Combine::Workspace { .. } => l.rungs.row_split && order == Order::Free,
        _ => l.rungs.row_split,
    };
    match split {
        true => l,
        false => launch(site, profile, contract, arms),
    }
}

/// `arms` with axis `ax` applied against what launch `from` realized, and its
/// label (`+async`: forced on). per-site-choice D2: the alternative is the
/// opposite of `from`'s realized state ([`crate::profile::ChoiceAxis::realized`]),
/// forced — a forced arm bypasses the rule that declined it (`kmajor-a`'s
/// register rule, `async-depth`). `None` where a measurement arm pins the rung:
/// an A/B arm stays what it forces, in every candidate.
pub(super) fn axis_arms(
    arms: crate::GpuArms,
    ax: &crate::profile::ChoiceAxis,
    from: &Launch,
) -> Option<(crate::GpuArms, String)> {
    let mut a = arms;
    if ax.full {
        if arms.off.get("plan-cost") || arms.off.get("full-tile") {
            return None;
        }
        *a.on.rung("plan-cost")? = true;
        *a.on.rung("full-tile")? = true;
        a.full = true;
        return Some((a, "+full-tile".into()));
    }
    if arms.on.get(ax.rung) || arms.off.get(ax.rung) {
        return None;
    }
    let on = !(ax.realized)(from);
    *(if on {
        a.on.rung(ax.rung)
    } else {
        a.off.rung(ax.rung)
    })? = true;
    Some((a, format!("{}{}", if on { '+' } else { '-' }, ax.rung)))
}

/// **per-site-choice B1: a site's candidates.** Candidate 0 is exactly the
/// rules' plan ([`realized`] under `arms`); then each
/// [`crate::profile::CHOICE_AXES`] entry flipped against its rungs, then the
/// pairs of two flips that each changed the launch (async × kmajor-a). A
/// candidate is kept when its launch — kernel and grid; a flipped rung shows
/// only through them — differs from every one before it, and on an order
/// other than free only with candidate 0's `Combine` (a fixed face replays one
/// partition count, an exact one has none). At most `arms.site_cands`.
/// `row`: the retained one-row pass of [`realized`].
pub(crate) fn site_cands(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
) -> Vec<SiteCand> {
    let at = |a: crate::GpuArms| realized(site, profile, contract, a, order, row);
    let base = at(arms);
    let mut out = vec![SiteCand {
        arms,
        launch: base,
        axes: "rules".into(),
    }];
    let cap = usize::try_from(arms.site_cands).unwrap_or(usize::MAX);
    // ADR-0050 C4: a placed site's table is its pick (a run-time
    // confirmation among the search's top is C6's).
    if cap <= 1 || base.rungs.machine_place {
        return out;
    }
    // `bank-plan` is the one axis rung the kernel text reads beyond the
    // launch: a k-major panel's row map ([`frag_layout`]).
    let map = |l: &Launch| l.rungs.bank_plan && l.tiled.is_some_and(|t| t.kmajor);
    let same = |a: &Launch, b: &Launch| {
        Launch {
            rungs: b.rungs,
            ..*a
        } == *b
            && map(a) == map(b)
    };
    // D2: no candidate where the axis's realized state did not change.
    let flip = |ax: &crate::profile::ChoiceAxis, fa: crate::GpuArms, from: &Launch| {
        let (a, axes) = axis_arms(fa, ax, from)?;
        let launch = at(a);
        (ax.full || (ax.realized)(&launch) != (ax.realized)(from)).then_some(SiteCand {
            arms: a,
            launch,
            axes,
        })
    };
    let keep = |out: &mut Vec<SiteCand>, c: SiteCand| {
        let order_ok = order == Order::Free || c.launch.combine == base.combine;
        if out.len() < cap && order_ok && !out.iter().any(|o| same(&o.launch, &c.launch)) {
            out.push(c);
        }
    };
    let singles: Vec<(&crate::profile::ChoiceAxis, SiteCand)> = crate::profile::CHOICE_AXES
        .iter()
        .filter_map(|ax| Some((ax, flip(ax, arms, &base)?)))
        .collect();
    for (_, c) in &singles {
        keep(&mut out, c.clone());
    }
    let moved: Vec<&(&crate::profile::ChoiceAxis, SiteCand)> = singles
        .iter()
        .filter(|(_, c)| !same(&c.launch, &base))
        .collect();
    for (i, (_, x)) in moved.iter().enumerate() {
        for (ay, _) in &moved[i + 1..] {
            if out.len() >= cap {
                return out;
            }
            if let Some(c) = flip(ay, x.arms, &x.launch) {
                let axes = format!("{}{}", x.axes, c.axes);
                keep(&mut out, SiteCand { axes, ..c });
            }
        }
    }
    out
}

/// The arms and order an epilogue group's candidates are realized under. A
/// `tile-post` group's fold may split into a workspace, whose reduce launch
/// evaluates the epilogue once every cell's sum is whole (the atomic and
/// stream-K splits are off). Any other group (sweep-runtime-2
/// `streamk-epi`) may take the stream-K walk the rules give its site, whose
/// tile-completing CTA evaluates the members on the whole sums
/// ([`Kernel::ticket`]); its atomic and workspace splits stay off — one
/// launch either way. `--gpu-off=streamk-epi`: the ordered kernel it
/// measured with (S54); so does a group whose walk would lose its panels'
/// L2 reuse ([`epi_walk`], bugs-properties).
pub(crate) fn group_order(
    epi: Option<&EpiKernel>,
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    order: Order,
) -> (crate::GpuArms, Order) {
    epi_order(
        epi.map(|e| e.post.is_some()),
        arms,
        order,
        epi_walk(site, profile, arms),
    )
}

/// bugs-properties `streamk-epi`: whether an epilogue group at `site` may
/// take the stream-K walk — its operands' L2 reuse on the part
/// ([`crate::rules::streamk_epi_pays`]); `--gpu-on=streamk-epi` forces it
/// (the measurement arm).
pub(crate) fn epi_walk(site: &TileSite, profile: &TargetProfile, arms: crate::GpuArms) -> bool {
    let operands = site.rows * site.k * site.a.elem_bytes(&site.elem)
        + site.k * site.c * site.b.elem_bytes(&site.elem);
    arms.on.get("streamk-epi")
        || profile
            .gpu()
            .is_none_or(|g| crate::rules::streamk_epi_pays(g, operands))
}

/// [`group_order`] by the group's shape alone: `post` — `None` without an
/// epilogue group, else whether it carries a `tile-post` body. The planning
/// passes (`horizontal`, `calls`) draw a site's candidates under this too,
/// so the launch they price is the one the emission's table holds.
pub(crate) fn epi_order(
    post: Option<bool>,
    arms: crate::GpuArms,
    order: Order,
    walk: bool,
) -> (crate::GpuArms, Order) {
    match post {
        Some(true) => {
            let mut a = arms;
            *a.off.rung("splitk-atomic").expect("a rung") = true;
            *a.off.rung("streamk").expect("a rung") = true;
            (a, order)
        }
        Some(_) if !arms.off.get("streamk-epi") && walk => {
            let mut a = arms;
            *a.off.rung("splitk-atomic").expect("a rung") = true;
            *a.off.rung("splitk").expect("a rung") = true;
            (a, order)
        }
        Some(_) => (arms, Order::Sequential),
        None => (arms, order),
    }
}

/// per-site-choice B2: no buffer a launch writes is one it reads, under the
/// residency plan's `alias` (the slot an object shares): `RERUN`.
pub(super) fn apart<T: Copy + PartialEq>(
    reads: &[T],
    writes: &[T],
    alias: impl Fn(T) -> T,
) -> bool {
    reads
        .iter()
        .all(|&r| writes.iter().all(|&w| alias(r) != alias(w)))
}

/// FNV-1a of `key`: the kernel ids' hash.
pub(crate) fn fnv(key: &str) -> u64 {
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// What a candidate kernel's id keys beyond its kind: the launch with only
/// its kernel rungs (the ones its plan line lists).
pub(super) fn launch_key(l: &Launch) -> String {
    let mut r = crate::GpuRungs::default();
    for n in KERNEL_RUNGS.iter().filter(|n| l.rungs.get(n)) {
        *r.rung(n).expect("a kernel rung") = true;
    }
    // The ids predate the host-only `row-block`, `zero-fold`, `row-stage`, `horizontal`, `calls` and
    // `narrow-tile` rungs (off here: no kernel rung): their fields are not part of a kernel's key, so
    // tuned picks and wisdom keep their ids.
    format!("{:?}", Launch { rungs: r, ..*l })
        .replace(", row_block: false", "")
        .replace(", zero_fold: false", "")
        .replace(", row_stage: false", "")
        .replace(", horizontal: false", "")
        .replace(", calls: false", "")
        .replace(", calls_issue: false", "")
        // integration: `post-table` decides the host's launch path only.
        .replace(", post_table: false", "")
        // half-precision: `narrow-tile` decides which sites exist (no kernel text).
        .replace(", narrow_tile: false", "")
        // tensor-cores: `half-tile` / `narrow-async` choose the kernel (no
        // kernel rung), and a TF32 tile keys as it did before its format.
        .replace(", half_tile: false", "")
        .replace(", narrow_async: false", "")
        .replace(", staged_regs: false", "")
        .replace(", half_fill: false", "")
        .replace(", narrow_radix: false", "")
        .replace(", fmt: Tf32, ea: 4, eb: 4, bt: false", "")
        // gemv-stream: the rungs decide `cells` and the split (both keyed);
        // a launch of one cell a thread keys as it did before them.
        .replace(", vec_cells: true", "")
        .replace(", vec_cells: false", "")
        .replace(", stream_fly: true", "")
        .replace(", stream_fly: false", "")
        // regressions: `vec-spread` decides `cells` (keyed).
        .replace(", vec_spread: true", "")
        .replace(", vec_spread: false", "")
        // TMA: `bulk-stage` keys the mma tile's kernel through `smem` (its
        // stages' barriers); a launch it does not stage keys as before.
        .replace(", bulk_stage: true", "")
        .replace(", bulk_stage: false", "")
        // hopper-wgmma-tma: the warp-specialized kernel keys through its
        // tile (`ws`, `realloc`); the S53 kernel and every other launch key
        // as before.
        .replace(", wgmma_ws: true", "")
        .replace(", wgmma_ws: false", "")
        .replace(", ws_regs: true", "")
        .replace(", ws_regs: false", "")
        .replace(", tma_digits: true", "")
        .replace(", tma_digits: false", "")
        .replace(", ws: false, realloc: None", "")
        // wgmma-16bit: the TF32 form's operand fields, and the rung (the
        // tile's fields key a 16-bit kernel).
        .replace(
            ", fmt: Tf32, direct: false, ea: 4, eb: 4, a_home: Regs, b_kmajor: false",
            "",
        )
        .replace(", wgmma16: true", "")
        .replace(", wgmma16: false", "")
        // sweep-attention: `unit-uniform` decides which units exist (no
        // kernel text of a tile site's).
        .replace(", unit_uniform: false", "")
        // mask-frontend: a stream unit's mask terms, walk order and
        // epilogue numbers (keyed by the unit; no tile site's text).
        .replace(", guard_terms: false", "")
        .replace(", walk_order: false", "")
        .replace(", stream_numbers: false", "")
        // bf16-class: `finite-check` decides a stream unit's skip (no tile
        // site's text).
        .replace(", finite_check: false", "")
        .replace(", narrow_window: false", "")
        .replace(", device_trap: false", "")
        // sweep-attention-2: `stream-mma` decides a stream unit's kernel (no
        // tile site's text); `kmajor-b` changes how a candidate copies `b`,
        // not which candidate it is: its id is the arms' one (as candidate
        // 0's is), every launch keys as before.
        .replace(", stream_mma: false", "")
        .replace(", kmajor_b: true", "")
        .replace(", kmajor_b: false", "")
        // sweep-runtime-2: `streamk-epi` decides an epilogue group's
        // combine (keyed through its ticket kernel's id).
        .replace(", streamk_epi: false", "")
        // bf16 tensor-tile: `mma-walk` decides a tensor site's combine
        // (keyed through its stream-K kernel's id).
        .replace(", mma_walk: false", "")
        // model-ready placement: host-only rungs (where cells run, which
        // range cells write in place, a cell's lanes — keyed by the cell).
        .replace(", trip_cells: false", "")
        .replace(", bytes_place: false", "")
        .replace(", once_bytes: false", "")
        .replace(", cand_made: false", "")
        .replace(", once_range: false", "")
        .replace(", range_chain: false", "")
        .replace(", stride_lanes: false", "")
        .replace(", carry_fwd: false", "")
        // model-ready integration: `lanes-unroll` (a cell body's loop
        // metadata) and ADR-0046's `math-hw` (a math builtin's unit) are
        // never a tile kernel's text — tuned picks keep their ids.
        .replace(", math_hw: false", "")
        .replace(", lanes_unroll: false", "")
        .replace(", fly_unroll: false", "")
        // ADR-0050 C4 / C5: `machine-place` chooses the launch (no kernel
        // rung, so always off here), and a walk without whole-tile blocks
        // (`whole: 0`) is the stream-K launch it was — ids as before them.
        .replace(", machine_place: false", "")
        .replace(", whole: 0 }", " }")
        .replace(", cells: 1 }", " }")
}

/// A candidate's kernel ids `(main, reduce)`, `reduce` 0 without a
/// workspace. Candidate 0 (`first`) keeps the ids the site had before
/// candidates; another keys its launch onto them.
pub(super) fn cand_ids(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    epi: Option<&EpiKernel>,
    l: &Launch,
    first: bool,
) -> (u64, u64) {
    let split = |part| kernel_id_split(site, profile, contract, part);
    let (main, rid) = match l.combine {
        Combine::None => (kernel_id_with(site, profile, contract, epi), 0),
        Combine::Atomic { .. } => (split("atomic"), 0),
        // sweep-runtime-2 `streamk-epi`: a group's stream-K kernel evaluates
        // its members at each tile's ticket (its id names the group).
        Combine::StreamK { .. } => match epi {
            Some(e) => (fnv(&format!("{:016x}|ticket|{e:?}", split("streamk"))), 0),
            None => (split("streamk"), 0),
        },
        // `tile-post`: the reduce launch evaluates the epilogue (its id
        // names the group, as an ordered group kernel's does).
        Combine::Workspace { .. } => match epi {
            Some(e) => (
                split("splitk"),
                fnv(&format!("{:016x}|{e:?}", split("reduce"))),
            ),
            None => (split("splitk"), split("reduce")),
        },
    };
    if first {
        return (main, rid);
    }
    let key = launch_key(l);
    let id = |x: u64| fnv(&format!("{x:016x}|{key}"));
    (id(main), if rid == 0 { 0 } else { id(rid) })
}

/// A candidate's launch recipe (B2) as the host's `%mapal_cand` constant,
/// and its workspace bytes.
pub(super) fn recipe(
    site: &TileSite,
    l: &Launch,
    (id, rid): (u64, u64),
    epi: bool,
) -> (String, u64) {
    let cells = site.rows * site.c;
    let bytes_o = cells * elem_bytes_of(&site.elem);
    let (cgx, cblock) = cell_grid(cells);
    let (zero, ws, gx, gy, rgx, rblock) = match l.combine {
        Combine::None => (0, 0, l.gx, l.gy, 0, 0),
        Combine::Atomic { s } => (bytes_o, 0, l.gx, l.gy * s, 0, 0),
        // sweep-runtime-2 `streamk-epi`: the sums and the tiles' tickets in
        // a workspace the runtime zeroes (`ws` without a reduce kernel).
        Combine::StreamK { ctas, whole, .. } if epi => {
            (0, ticket_bytes(site, l), whole + ctas, 1, 0, 0)
        }
        Combine::StreamK { ctas, whole, .. } => (zero_bytes(site, l), 0, whole + ctas, 1, 0, 0),
        Combine::Workspace { s } => (0, s * bytes_o, l.gx, l.gy * s, cgx, cblock),
    };
    let text = format!(
        "%mapal_cand {{ i64 {}, i64 {}, i64 {zero}, i64 {ws}, i32 {gx}, i32 {gy}, i32 {}, i32 {}, \
         i32 {rgx}, i32 {rblock} }}",
        id as i64, rid as i64, l.block, l.smem
    );
    (text, ws)
}

/// sweep-runtime-2 `streamk-epi`: a group's stream-K workspace — one tile
/// of sums a block (a shared tile's slot is its first block,
/// [`Kernel::ticket_split`]; 16-byte aligned), then one `i32` ticket a slot.
pub(crate) fn ticket_bytes(site: &TileSite, l: &Launch) -> u64 {
    // ADR-0050 C5 hybrid: a slot a block of the launch (a whole tile's
    // block never takes its slot).
    let Combine::StreamK { ctas, whole, .. } = l.combine else {
        unreachable!("a stream-K launch")
    };
    let ctas = whole + ctas;
    // bf16 tensor-tile `mma-walk`: or the tensor tile.
    let (bm, bn) = match (l.tiled, l.mma) {
        (Some(g), _) => (g.bm(), g.bn()),
        (None, Some(m)) => (m.bm(), m.bn()),
        _ => unreachable!("stream-K walks the FMA or the tensor tile"),
    };
    let sums = ctas * bm * bn * elem_bytes_of(&site.elem);
    sums.div_ceil(GPU_VEC_BYTES) * GPU_VEC_BYTES + 4 * ctas
}

/// A table's evidence line (B1): candidate 0's id, the count, every
/// candidate's main kernel and the axes it flipped.
pub(super) fn site_line(ids: &[(u64, u64)], cands: &[SiteCand]) -> String {
    let ks: Vec<String> = ids.iter().map(|(i, _)| format!("{i:016x}")).collect();
    let axes: Vec<&str> = cands.iter().map(|c| c.axes.as_str()).collect();
    format!(
        "; mapal.site site={:016x} n={} cands={} axes={}\n",
        ids[0].0,
        ids.len(),
        ks.join(","),
        axes.join(",")
    )
}

/// The device half of a site's table: every candidate's kernels but candidate
/// 0's (the site's own, emitted as before), and the site's `; mapal.site`
/// line — only where the host's table names them (`live`).
#[allow(clippy::too_many_arguments)]
pub(super) fn cand_kernels(
    kernels: &mut std::collections::BTreeMap<String, String>,
    lines: &mut std::collections::BTreeSet<String>,
    live: &std::collections::BTreeSet<String>,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    row: bool,
    epi: Option<&EpiKernel>,
) {
    let cands = site_cands(site, profile, contract, arms, order, row);
    let ids: Vec<(u64, u64)> = cands
        .iter()
        .enumerate()
        .map(|(i, c)| cand_ids(site, profile, contract, epi, &c.launch, i == 0))
        .collect();
    // `tile-post`: candidate 0 of a post group may be a split, whose kernels
    // the plain group emission does not make: every candidate here — and
    // (`streamk-epi`) any group's stream-K candidate 0.
    let first = usize::from(
        !epi.is_some_and(|e| e.post.is_some() || cands[0].launch.combine != Combine::None),
    );
    if !ids[first..]
        .iter()
        .any(|(i, _)| live.contains(&kernel_name(*i)))
    {
        return;
    }
    for (c, &(id, rid)) in cands.iter().zip(&ids).skip(first) {
        let name = kernel_name(id);
        let comb = c.launch.combine;
        // A workspace split's partials are plain sums: the epilogue runs in
        // its reduce launch.
        let main_epi = match comb {
            Combine::Workspace { .. } => None,
            _ => epi,
        };
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(
                &name, site, profile, contract, c.arms, false, main_epi, comb,
            )
        });
        if let Combine::Workspace { s } = comb {
            let red = kernel_name(rid);
            kernels.entry(red.clone()).or_insert_with(|| {
                format!(
                    "; mapal.plan kernel={red} kind=reduce rows={} c={} S={s}{}\n",
                    site.rows,
                    site.c,
                    if epi.is_some() { " epi=1" } else { "" }
                ) + &reduce_kernel_text(&red, site, profile, contract, c.arms, s, epi)
            });
        }
    }
    lines.insert(site_line(&ids, &cands));
}

/// **plan-cost `linear` (§4a concat): the site table of `terms` sites of
/// `site`'s shape** — one launch, `z = term·S + part`, every partial added
/// into the zeroed sum — at the cheapest `S` by [`TargetProfile::site_ns`]:
/// `T×` the blocks and operand bytes, `S·T` adders per cell. A product goes
/// through `k/S` roundings in its partial and `S·T − 1` adds into the zeroed
/// sum: `S` is admitted only where `γ` of that depth stays within `bound`,
/// `linear_plan`'s source bound `γ_{k+T−1}` (D6; for `S > 1`, `S·T ≤ k`).
/// `None` where the scalar kernels do not run, the part has no plan rates, or
/// a launch's pointer table (≤ 16) is too short.
pub(crate) fn concat_plan(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    terms: u64,
    bound: f64,
) -> Option<(u64, PlanCost)> {
    let l = launch(site, profile, contract, arms);
    let gpu = profile.gpu()?;
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some() || site.radix.is_some();
    let atomic = elem_bytes_of(&site.elem) == 4 || gpu.caps.red_f64;
    if l.mma.is_some() || l.wgmma.is_some() || window || !atomic || l.raster != 1 {
        return None;
    }
    if !l.rungs.linear || 2 * terms + 1 > 16 || profile.plan_rates().is_none() {
        return None;
    }
    let panels = site.k / l.tiled.map_or(1, |g| g.bk);
    let mut best: Option<(u64, PlanCost)> = None;
    for s in (1..=panels.min(64)).filter(|&s| panels.is_multiple_of(s)) {
        let c = Combine::Atomic { s };
        let r = resized(site, profile, contract, arms, l, c);
        if mapal_ir::bound_coef_unit(site.k / s + s * terms - 1, site.unit()) > bound
            || !(site.k / s).is_multiple_of(r.tiled.map_or(1, |g| g.bk))
        {
            continue;
        }
        let mut shape = plan_shape(site, &r, c, gpu);
        shape.blocks *= terms;
        shape.k *= terms;
        shape.s *= terms;
        let cost = profile.site_ns(&shape)?;
        if best.is_none_or(|(_, b)| cost.ns < b.ns) {
            best = Some((s, cost));
        }
    }
    best
}

/// The device text of [`concat_plan`]'s kernel: the atomic split kernel of
/// `S` partitions over a table of `terms` sites.
pub(super) fn concat_kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    terms: u64,
    s: u64,
) -> String {
    let c = Combine::Atomic { s };
    let mut k = kernel(site, profile, contract, arms, false, None, c);
    k.terms = terms;
    let l = launch_for(site, profile, contract, arms, c);
    k.out = plan_line(name, site, profile, &l, k.idx, c, false, false, None);
    k.out.pop();
    k.out += &format!(" terms={terms}\n");
    match l.tiled {
        Some(g) => k.tiled(name, g),
        None if l.cells > 1 => k.naive_cells(name, l.block, l.cells),
        None => k.naive(name, l.block),
    }
    k.stream_field();
    k.out
}
