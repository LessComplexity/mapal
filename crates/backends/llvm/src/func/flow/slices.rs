//! a unit's row slices, their phases and the block choice

use super::*;

impl<'a> FnEmit<'a> {
    /// **long-rows `row-slices`: how many blocks a row of a unit takes.**
    /// Where the order is free and every fold of the unit regroups (a tree
    /// step over `f32`), a row's folds and stored cells split into `s`
    /// slices of their axis, one block each; [`Self::slice_phases`] orders
    /// the members by `fold_combine`'s laws. The rule prices one block a row
    /// against `s` blocks a row with the part's measured properties: a
    /// block's walk of `n` elements a lane at `row_iter_ns` each (rounds of
    /// the co-resident blocks), floored by the unit's bytes at the rate of
    /// where they live (`residence_plan`), plus the launch — programmatic
    /// for one phase, cooperative and `phases − 1` grid barriers for more,
    /// and its share of the process's first cooperative launch
    /// (`coop_first_ns`, the driver's one-time setup, over `launches`: 1
    /// outside every loop, a counted loop's trips, 0 for unknown trips).
    /// The cheapest `s` wins where it beats one block a row; a part without
    /// `row_iter_ns`, `gsync_ns` or `coop_launch_ns` slices nothing.
    pub(crate) fn row_slices(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        launches: u64,
    ) -> Option<Slices> {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let force = self.gpu_arms.row_slices_s;
        if !rungs.row_slices || self.order != crate::Order::Free || force == 1 {
            return None;
        }
        let gpu = self.profile.gpu()?;
        let f32_tree = |site: &RowFoldSite| {
            tree_step(&site.op) && site.elem == mapal_ir::Ty::f32() && site.a.ksplit.is_none()
        };
        // The extents a row's work splits along: each fold's `k`, each
        // stored row cell's `w`.
        let mut exts = Vec::new();
        for mb in members {
            match &mb.kind {
                UKind::Fold { site, colmajor } if f32_tree(site) && colmajor.is_none() => {
                    exts.push(site.k)
                }
                UKind::Fold { .. } | UKind::Tile { .. } => return None,
                UKind::Cell { .. } if !mb.per_row() && mb.store => exts.push(mb.w),
                UKind::Cell { .. } => {}
            }
        }
        if exts.is_empty() {
            return None;
        }
        let (phase, avail, phases, laws, args) = self.slice_phases(members);
        // A sliced block's threads ([`slice_geo`]); a row's partials combine
        // one a lane, so `s` stays within them.
        let lanes_of = |s: u64| {
            exts.iter()
                .map(|&e| (e / s).next_power_of_two().clamp(gpu.warp, 256))
                .max()
                .unwrap_or(gpu.warp)
        };
        let fits = |s: u64| exts.iter().all(|&e| e % s == 0) && s <= lanes_of(s);
        let slices = |s: u64, ns: (u64, u64), launches: bool| Slices {
            s,
            phase: phase.clone(),
            avail: avail.clone(),
            phases,
            laws: laws.clone(),
            args: args.clone(),
            ns,
            launches,
        };
        if force > 1 {
            // A forced `--gpu-on=slice-launches` arm: the phases as launches.
            let launches = self.gpu_arms.on.slice_launches && phases > 1;
            return fits(force).then(|| slices(force, (0, 0), launches));
        }
        // open-rules: every part is priced — its measurements, else derived
        // from its clock (`Gpu::slice_cost`).
        let d = gpu.slice_cost();
        if d.row_iter_ns == 0 || d.gsync_ns == 0 || d.coop_launch_ns == 0 {
            return None;
        }
        // The unit's bytes: each fold's walk reads its row's operand cells
        // (a recomputed cell reads its own), each stored cell is written.
        let elem = |o: ObjectId| {
            crate::profile::elem_bytes(super::super::resident::array_elem(&self.obj_ty(o)))
        };
        let reads = |mb: &UMember| {
            mb.srcs
                .iter()
                .filter(|s| matches!(s, USrc::Gmem(_) | USrc::Member { row: false, .. }))
                .count() as u64
        };
        let bytes: u64 = members
            .iter()
            .map(|mb| match &mb.kind {
                UKind::Fold { site, .. } => rows * site.k * 4 * reads(mb).max(1),
                UKind::Cell { .. } if mb.store => rows * mb.w * elem(mb.out) * (reads(mb) + 1),
                _ => 0,
            })
            .sum();
        let res = self.ir.residence_plan(self.f);
        let dram = gmem
            .iter()
            .any(|&o| res.at(o, gpu.l2_bytes) == mapal_ir::Residence::Dram);
        let rate = match dram {
            true => d.dram_bytes_per_ns,
            false => d.l2_bytes_per_ns,
        }
        .max(1) as f64;
        // One block a row keeps a row two members read on chip (`unit_geo`):
        // a walk that reads only kept rows and per-row values reads no
        // memory. A sliced block keeps none.
        let kept = unit_geo(
            &Unit {
                id: 0,
                rows,
                members: members.to_vec(),
                gmem: gmem.to_vec(),
                gmove: Vec::new(),
                block: None,
                slices: None,
                stream: None,
            },
            self.profile,
            true,
            |mb| match &mb.kind {
                UKind::Fold { site, .. } => crate::profile::elem_bytes(&site.elem),
                _ => elem(mb.out),
            },
        )
        .kept;
        fn from_memory(members: &[UMember], kept: &[Option<u64>], k: usize) -> bool {
            members[k].srcs.iter().any(|s| match s {
                USrc::Gmem(_) => true,
                USrc::Member { k: p, .. } if !members[*p].per_row() => {
                    kept[*p].is_none() && from_memory(members, kept, *p)
                }
                _ => false,
            })
        }
        // Fold combines a sliced block reads: one partial a lane, one load
        // latency each.
        let restores: u64 = (0..phases)
            .map(|p| {
                members
                    .iter()
                    .enumerate()
                    .filter(|&(k, mb)| {
                        matches!(mb.kind, UKind::Fold { .. }) && avail[k] <= p && phase[k] < p
                    })
                    .count() as u64
            })
            .sum();
        // The walks: each fold and each stored or kept row cell, `n ÷ s`
        // elements over `lanes` threads, in rounds of the resident blocks.
        let time = |s: u64| {
            let lanes = lanes_of(s);
            let per_sm = (gpu.max_threads_per_sm / lanes).clamp(1, gpu.max_blocks_per_sm);
            let items = rows * s;
            let grid = items.min(gpu.sms * per_sm);
            let rounds = items.div_ceil(grid) as f64;
            let none = vec![None; members.len()];
            let kept = if s > 1 { &none } else { &kept };
            let walk: u64 = (0..members.len())
                .filter(|&k| from_memory(members, kept, k))
                .map(|k| match &members[k].kind {
                    UKind::Fold { site, .. } => (site.k / s).div_ceil(lanes),
                    UKind::Cell { .. }
                        if !members[k].per_row() && (members[k].store || kept[k].is_some()) =>
                    {
                        (members[k].w / s).div_ceil(lanes)
                    }
                    _ => 0,
                })
                .sum();
            let restore = if s > 1 { restores * d.lat_ns } else { 0 };
            let work = (rounds * (walk * d.row_iter_ns + restore) as f64).max(bytes as f64 / rate);
            // A unit outside every loop launches once: its cooperative launch
            // may be the process's first (`coop_first_ns`); a loop's trips
            // share that cost. long-prefill `slice-launches`: or one launch a
            // phase, each a kernel boundary in place of a grid barrier.
            // review-fixes-2: a counted loop's trips each pay their share
            // (`launches_in`); a loop of unknown trips no term.
            let first = match launches {
                0 => 0,
                n => d.coop_first_ns / n,
            };
            let one_launch = d.launch_at(grid, gpu.sms, rungs.pdl)?;
            let (launch, launches) = match s > 1 && phases > 1 {
                true => {
                    let coop = d.coop_launch_at(grid, gpu.sms)?
                        + f64::from(phases - 1) * d.gsync_at(grid, gpu.sms)?
                        + first as f64;
                    let each = f64::from(phases) * one_launch;
                    match rungs.slice_launches && each < coop {
                        true => (each, true),
                        false => (coop, false),
                    }
                }
                false => (one_launch, false),
            };
            Some((launch + work, launches))
        };
        let one = time(1)?.0;
        let best = (1..=6)
            .map(|e| 1u64 << e)
            // A slice of at least a warp of cells.
            .filter(|&s| fits(s) && exts.iter().all(|&e| e / s >= gpu.warp))
            .filter_map(|s| Some((s, time(s)?)))
            .min_by(|a, b| a.1.0.total_cmp(&b.1.0))?;
        let (s, (ns, launches)) = best;
        (ns < one).then(|| slices(s, (ns.round() as u64, one.round() as u64), launches))
    }

    /// **The phases of a sliced unit** (`fold_combine`): a fold's partials
    /// run in the phase its term's sources are complete in — a per-row value
    /// its term reads by a factor or shift law excepted: that read takes the
    /// law's identity and the value is applied after the combine. A fold is
    /// complete one phase after its partials (and once its laws' values
    /// are); a cell in the phase its sources are complete in. Returns per
    /// member the phase and the completion phase, the phase count, and each
    /// fold's laws.
    #[allow(clippy::type_complexity)]
    pub(super) fn slice_phases(
        &self,
        members: &[UMember],
    ) -> (
        Vec<u32>,
        Vec<u32>,
        u32,
        Vec<Vec<(usize, u32, DepLaw)>>,
        Vec<Vec<Option<mapal_ir::RescaleArg>>>,
    ) {
        let fc = self.ir.fold_combine(self.f);
        let n = members.len();
        let (mut ph, mut avail) = (vec![0u32; n], vec![0u32; n]);
        let mut laws: Vec<Vec<(usize, u32, DepLaw)>> = vec![Vec::new(); n];
        let mut args: Vec<Vec<Option<mapal_ir::RescaleArg>>> = vec![Vec::new(); n];
        // ADR-0042 D7: a rescale law is taken where the value is an earlier
        // fold of the unit over the exponent's `x` (the same array, the same
        // slices of the same row) whose step keeps the extreme that bounds
        // every term by 1 — its partial over a slice is that slice's shift.
        let rescale_ok = |k: usize, v: usize, arg: &mapal_ir::RescaleArg| {
            v < k
                && matches!(members[k].kind, UKind::Fold { .. })
                && matches!(members[v].kind, UKind::Fold { .. })
                && self
                    .ir
                    .rescale_shift_ok(self.f, members[k].m, members[v].m, arg)
        };
        let src_phase = |srcs: &[USrc], skip: &[usize], avail: &[u32]| {
            srcs.iter()
                .filter_map(|s| match s {
                    USrc::Member { k, .. } if !skip.contains(k) => Some(avail[*k]),
                    _ => None,
                })
                .max()
                .unwrap_or(0)
        };
        for k in 0..n {
            let mb = &members[k];
            match &mb.kind {
                UKind::Fold { .. } => {
                    let mut skip = Vec::new();
                    if let Some(&USrc::Member { k: c, .. }) = mb.srcs.first() {
                        for d in fc.sites.get(mb.m).map_or(&[][..], |x| &x.deps[..]) {
                            if d.law == DepLaw::Phase || d.via != members[c].m {
                                continue;
                            }
                            let Some(&USrc::Member {
                                k: v, row: true, ..
                            }) = members[c].srcs.get(d.slot as usize)
                            else {
                                continue;
                            };
                            if members[v].out == d.value && members[v].per_row() {
                                if d.law == DepLaw::Rescale
                                    && !d.rescale.as_ref().is_some_and(|a| rescale_ok(k, v, a))
                                {
                                    continue;
                                }
                                skip.push(v);
                                laws[k].push((v, d.slot, d.law));
                                args[k].push(d.rescale.clone());
                            }
                        }
                        ph[k] = src_phase(&members[c].srcs, &skip, &avail);
                        // A rescale's shift is its value's partial over the
                        // same slice: computed in the same item, before.
                        for (&(v, _, law), _) in laws[k].iter().zip(&args[k]) {
                            if law == DepLaw::Rescale {
                                ph[k] = ph[k].max(ph[v]);
                            }
                        }
                    }
                    avail[k] = laws[k]
                        .iter()
                        .map(|&(v, _, _)| avail[v])
                        .fold(ph[k] + 1, u32::max);
                }
                UKind::Cell { .. } => {
                    ph[k] = src_phase(&mb.srcs, &[], &avail);
                    avail[k] = ph[k];
                }
                UKind::Tile { .. } => unreachable!("a sliced unit holds no tile"),
            }
        }
        let phases = (0..n)
            .map(|k| match members[k].kind {
                UKind::Fold { .. } => avail[k] + 1,
                _ => ph[k] + 1,
            })
            .max()
            .unwrap_or(1);
        (ph, avail, phases, laws, args)
    }

    /// **compile-time-graph A: how many rows a row-block unit holds.** The
    /// powers of two `b` dividing `rows` whose layout ([`block_geo`]) fits the
    /// part's opt-in shared memory and whose tile accumulators fit a thread's
    /// registers; the cheapest of them by the plan model
    /// ([`Self::block_cost`]), kept only when it beats the split launches. A
    /// part without plan rates, or a shape the model cannot price, keeps the
    /// boundary: nothing else weighs the unit against the launches it
    /// replaces. `Err` names the last test every `b` failed: `budget`,
    /// `unpriced`, `cost`.
    pub(super) fn block_choice(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        grp: &UnitGroup,
    ) -> Result<Block, &'static str> {
        let gpu = self.profile.gpu().expect("a unit has a GPU profile");
        let free = self.order == crate::Order::Free;
        let color: Vec<Option<usize>> = members
            .iter()
            .map(|mb| {
                grp.members
                    .iter()
                    .find(|g| g.m == mb.m)
                    .and_then(|g| g.color)
            })
            .collect();
        let lanes = block_lanes(members, gpu.warp);
        for mb in members {
            if let UKind::Tile { site } = &mb.kind
                && !(site.c % lanes == 0 || lanes % site.c == 0)
            {
                return Err("footprint");
            }
        }
        let mut why = "budget";
        let mut best: Option<(Block, u64)> = None;
        let forced = self.gpu_arms.row_block_b;
        self.block_priced.set(None);
        // kernel-merging `row-stage`: each tile's panels staged through
        // shared memory at every candidate depth × buffers that fits (the
        // plan model picks); a `b` no staging fits reads them straight.
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        // `cp.async` where the part has it (sm_80+); `--gpu-off=async` copies
        // through registers.
        let cp = gpu.has_mma_sync() && !self.gpu_arms.off.cp_async;
        for sh in (0..=6).rev() {
            let b = 1u64 << sh;
            if !rows.is_multiple_of(b) || (forced > 0 && b != forced) {
                continue;
            }
            // Each tile's staged options; the combinations of them (each tile
            // its own, the panel region their largest), then — when no
            // combination fits — every tile read straight from memory.
            let opts: Vec<Vec<Option<Stage>>> = members
                .iter()
                .map(|mb| match &mb.kind {
                    UKind::Tile { site } if rungs.row_stage => {
                        let mut o: Vec<Option<Stage>> = Vec::new();
                        let cands = match self.gpu_arms.row_stage_force {
                            Some(f) => vec![f],
                            None => stage_cands(cp),
                        };
                        for (kc, st, p) in cands {
                            let x = stage_of(
                                site,
                                lanes,
                                free,
                                b,
                                matches!(mb.srcs[0], USrc::Gmem(_)),
                                kc,
                                st,
                                p,
                                cp,
                            );
                            if x.is_some() && !o.contains(&x) {
                                o.push(x);
                            }
                        }
                        if o.is_empty() {
                            o.push(None);
                        }
                        o
                    }
                    _ => vec![None],
                })
                .collect();
            let mut combos: Vec<Vec<Option<Stage>>> = vec![Vec::new()];
            for o in &opts {
                combos = combos
                    .into_iter()
                    .flat_map(|c| {
                        o.iter().map(move |x| {
                            let mut c = c.clone();
                            c.push(*x);
                            c
                        })
                    })
                    .take(STAGE_COMBOS)
                    .collect();
            }
            let direct = vec![None; members.len()];
            if !combos.contains(&direct) {
                combos.push(direct);
            }
            let mut staged_fit = false;
            let mut first: Option<Block> = None;
            for stage in combos {
                let any = stage.iter().any(Option::is_some);
                if !any && staged_fit {
                    continue;
                }
                let blk = Block {
                    b,
                    color: color.clone(),
                    ns: (0, 0),
                    stage,
                };
                let probe = Unit {
                    id: 0,
                    rows,
                    members: members.to_vec(),
                    gmem: gmem.to_vec(),
                    gmove: Vec::new(),
                    block: Some(blk.clone()),
                    slices: None,
                    stream: None,
                };
                let geo = self.geo(&probe);
                let acc = members
                    .iter()
                    .filter_map(|mb| match &mb.kind {
                        UKind::Tile { site } => Some(b * tile_split(site, lanes, free).0),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                if geo.smem > gpu.smem_per_block
                    || acc + crate::profile::FMA_ADDR_REGS > gpu.regs_per_thread / 2
                {
                    continue;
                }
                staged_fit |= any;
                first.get_or_insert(blk.clone());
                // The model's waves charge an unfilled grid its busiest SM.
                match self.block_cost(&probe, &geo) {
                    Some((fused, split)) => {
                        why = "cost";
                        if self.block_priced.get().is_none_or(|x| fused < x.1) {
                            self.block_priced.set(Some((b, fused, split)));
                        }
                        if (fused <= split || forced > 0)
                            && best.as_ref().is_none_or(|x| fused < x.1)
                        {
                            best = Some((
                                Block {
                                    ns: (fused, split),
                                    ..blk
                                },
                                fused,
                            ));
                        }
                    }
                    None if why == "budget" => why = "unpriced",
                    None => {}
                }
            }
            // A forced `b` takes its cheapest priced realization, else its
            // first that fits.
            if forced > 0 {
                if best.is_none() {
                    best = first.map(|blk| (blk, 0));
                }
                break;
            }
        }
        best.map(|(blk, _)| blk).ok_or(why)
    }
}
