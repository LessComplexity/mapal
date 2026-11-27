//! The `stream` rule (R1-R2): [`FnEmit::stream_choice`], its cost model and the pricing constants.
use super::*;

impl<'a> FnEmit<'a> {
    /// **The `stream` rule (R1–R2).** Where `stream_plan` proves the row-block
    /// group `grp` one pass, the order is free (the folds regroup into
    /// windows) and the rung is on: every geometry — `threads` of 128 or 256,
    /// `b` rows dividing the group's, windows of 16–64 keys (at least 16: the
    /// rescales stay inside the default face's bound), `rt` rows a thread —
    /// whose shared memory fits the part's opt-in limit and whose registers
    /// fit a thread's, priced by the plan model against the split
    /// realization's launches; the cheapest when it beats them.
    /// `Ok(None)`: not a stream group (or the rung is off); `Err` names why
    /// a stream group stays split.
    pub(crate) fn stream_choice(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        grp: &mapal_ir::UnitGroup,
    ) -> Result<Option<Stream>, &'static str> {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if !rungs.stream || self.order != crate::Order::Free {
            return Ok(None);
        }
        let units = self.ir.unit_plan(self.f);
        let Some(gi) = units.groups.iter().position(|g| g == grp) else {
            return Ok(None);
        };
        let sp = self.ir.stream_plan(self.f);
        // A group the fact refuses is no stream group (its reason is the
        // fact's: `stream_plan().refused`).
        let Some(sg) = sp.of(gi) else {
            return Ok(None);
        };
        // The members the pass covers (`StreamGroup::roles`): a unit of the
        // whole group where trailing members follow is no stream unit.
        if members.len() != sg.roles.len() {
            return Ok(None);
        }
        let gpu = self.profile.gpu().ok_or("unpriced")?;
        // review-fixes: priced on every part — its measured plan rates, else
        // the ones its properties derive (`TargetProfile::priced`): a part
        // without a calibration no longer keeps the split realization
        // unweighed.
        let priced = self.profile.priced();
        // What this realization emits: f32 values, plain tile sites whose
        // operands are memory, no per-row work outside the windows.
        let f32t = mapal_ir::Ty::f32();
        for (k, mb) in members.iter().enumerate() {
            let elem =
                crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)));
            // bf16-class `round-post`: a consumer whose epilogue rounds its
            // cell as stored writes the narrow format (its call's result).
            let rounded = matches!(
                (&sg.roles[k], &mb.kind),
                (StreamRole::Consume { .. }, UKind::Tile { site })
                    if site.post.as_ref().is_some_and(|p| p.round.is_some())
            );
            if elem != 4 && !(elem == 2 && rounded) {
                return Err("elem");
            }
            match (&sg.roles[k], &mb.kind) {
                (StreamRole::Row { .. }, _) => return Err("row"),
                (StreamRole::Produce, UKind::Tile { site }) => {
                    // Plain, epilogue or mixed-radix addresses over memory
                    // operands — a captured number moving one (`unit-uniform`:
                    // the launch moves its pointer) where the epilogue never
                    // reads that number; no narrow element yet.
                    // mask-frontend: the epilogue's arrays memory (a mask's
                    // per-row limit), its numbers launch arguments
                    // (`stream-numbers`; off: none read after the fold, no
                    // array read).
                    if self.gpu_arms.off.stream_numbers
                        && site
                            .post
                            .as_ref()
                            .is_some_and(|p| !p.after.is_empty() || !p.reads.is_empty())
                    {
                        return Err("site");
                    }
                    // bf16-class `narrow-window`: a bf16 operand (a bf16
                    // KV cache, a query stored bf16) where the pass folds on
                    // the matrix product (checked below).
                    let ok = site
                        .a
                        .narrow
                        .is_none_or(|n| n == mapal_ir::NarrowFloat::Bf16)
                        && site
                            .b
                            .narrow
                            .is_none_or(|n| n == mapal_ir::NarrowFloat::Bf16)
                        && site.a.ksplit.is_none()
                        && site.b.ksplit.is_none()
                        && site.elem == f32t;
                    if !ok {
                        return Err("site");
                    }
                    if !mb.srcs.iter().all(|x| matches!(x, USrc::Gmem(_))) {
                        return Err("site-src");
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    // Its row operand the group's own rows: only its column
                    // operand may be moved.
                    if site.radix.as_ref().is_some_and(|r| !r.ua.is_empty()) {
                        return Err("consume-ua");
                    }
                    if self.gpu_arms.off.stream_numbers
                        && site.post.as_ref().is_some_and(|p| !p.after.is_empty())
                    {
                        return Err("site");
                    }
                    let ok = site
                        .b
                        .narrow
                        .is_none_or(|n| n == mapal_ir::NarrowFloat::Bf16)
                        && site.b.ksplit.is_none()
                        && site.elem == f32t;
                    if !ok {
                        return Err("consume");
                    }
                    if !matches!(mb.srcs[..2], [USrc::Member { .. }, USrc::Gmem(_)]) {
                        return Err("consume-src");
                    }
                }
                (StreamRole::Fold { .. }, UKind::Fold { site, colmajor }) => {
                    if site.elem != f32t || colmajor.is_some() {
                        return Err("elem");
                    }
                }
                (StreamRole::Cell { .. }, UKind::Cell { .. }) => {}
                _ => return Err("member"),
            }
        }
        // mask-frontend: an open guard's per-launch numbers are the
        // producer's launch arguments (`stream-numbers`: its epilogue's
        // numbers) and its row loads the producer's memory operands — else
        // no window is skipped.
        let mut sg = sg.clone();
        if let Some(gd) = sg.guard.as_mut()
            && gd.skip
        {
            let post = match &members[gd.producer].kind {
                UKind::Tile { site } => site.post.clone(),
                _ => None,
            };
            let ok = post.is_some_and(|p| {
                [&gd.form.lhs, &gd.form.rhs].iter().all(|sd| {
                    sd.uniform.iter().all(|d| p.after.contains(&d.slot))
                        && sd.load.as_ref().is_none_or(|l| p.reads.contains(&l.slot))
                })
            });
            if !ok {
                gd.skip = false;
                gd.why = "open";
            }
        }
        // bf16-class `finite-check` off: a column operand finite only at run
        // time skips no window (as before).
        if let Some(gd) = sg.guard.as_mut()
            && gd.skip
            && !gd.check.is_empty()
            && self.gpu_arms.off.finite_check
        {
            gd.skip = false;
            gd.why = "finite";
        }
        let sg = &sg;
        let cp = gpu.has_mma_sync() && !self.gpu_arms.off.cp_async;
        let forced = self.gpu_arms.stream_force;
        let axis = sg.axis;
        // sweep-attention-2 `stream-mma`: the warp matrix product every
        // product of the pass admits (`None`: FFMA only).
        let mma = match rungs.stream_mma {
            true => mma_fmt(members, &sg.roles, gpu),
            false => None,
        };
        // bf16-class `narrow-window`: a narrow column operand's windows hold
        // its own bytes, which only the matrix product's fragments take.
        let narrow = members.iter().any(|mb| {
            matches!(&mb.kind, UKind::Tile { site } if site.a.narrow.is_some() || site.b.narrow.is_some())
        });
        if narrow && (mma.is_none() || self.gpu_arms.off.narrow_window) {
            return Err("narrow");
        }
        let mut best: Option<(Stream, f64)> = None;
        // A forced geometry takes the matrix product where it admits one
        // (the measurement arm of the instruction; `stream-mma` off: FFMA).
        let mut forced_mma: Option<Stream> = None;
        let mut why = "budget";
        for threads in [256u64, 128, 64, 32] {
            for b in [128u64, 64, 32, 16, 8, 4, 2, 1] {
                // A block's rows inside one panel batch (`unit_plan`'s).
                if !rows.is_multiple_of(b) || !grp.panel.is_multiple_of(b) {
                    continue;
                }
                for kb in [64u64, 32, 16] {
                    // A window of at least 16 keys keeps the rescales inside
                    // the bound; a row shorter than that is one window.
                    let kb = if axis < 16 {
                        axis.next_power_of_two().max(4)
                    } else {
                        kb
                    };
                    if kb > axis.next_power_of_two().max(4) {
                        continue;
                    }
                    for rt in [8u64, 4, 2, 1] {
                        if !b.is_multiple_of(rt) {
                            continue;
                        }
                        let groups = b / rt;
                        if !threads.is_multiple_of(groups) {
                            continue;
                        }
                        let g = threads / groups;
                        if !g.is_power_of_two() || g > gpu.warp || g > kb || g < 4 {
                            continue;
                        }
                        if let Some((fb, fk, fr, _, ft)) = forced
                            && (fb != b
                                || fk != kb
                                || (fr > 0 && fr != rt)
                                || (ft > 0 && ft != threads))
                        {
                            continue;
                        }
                        for sp in [1u64, 2, 4, 8, 16] {
                            // R3: a split of the windows where no mask skips them
                            // (`split-kv` off: one block a row block).
                            let nbs = axis.div_ceil(kb);
                            if sp > 1
                                && (sg.guard.as_ref().is_some_and(|g| g.skip)
                                    || !rungs.split_kv
                                    || !nbs.is_multiple_of(sp))
                            {
                                continue;
                            }
                            if let Some((_, _, _, fs, _)) = forced
                                && fs > 0
                                && fs != sp
                            {
                                continue;
                            }
                            // `stream-mma`: the instruction's layout — two
                            // rows and a quad's columns a thread, a warp's
                            // 16 rows, whole 16-key `k`-steps a window.
                            let frag = rt == 2
                                && g == 4
                                && b.is_multiple_of(MMA_M)
                                && kb.is_multiple_of(MMA_K);
                            let both = [mma, None];
                            let fmts: &[Option<crate::profile::MmaFmt>] = match (mma, frag) {
                                (Some(_), true) if narrow => &both[..1],
                                (Some(_), true) => &both[..],
                                _ if narrow => &[][..],
                                _ => &[None][..],
                            };
                            for &fmt in fmts {
                                let st = Stream {
                                    group: sg.clone(),
                                    b,
                                    kb,
                                    rt,
                                    g,
                                    threads,
                                    cp,
                                    ns: (0, 0),
                                    s: sp,
                                    rows_of: grp.members.iter().map(|m| m.row_of.clone()).collect(),
                                    order: grp.order.clone(),
                                    mma: fmt,
                                };
                                let lay = st.layout(members);
                                if lay.total > gpu.smem_per_block
                                    || st.regs(members) > gpu.regs_per_thread.min(232)
                                {
                                    continue;
                                }
                                let ns = match forced {
                                    Some(_) => 0.0,
                                    None => {
                                        match self
                                            .stream_ns(&priced, &st, members, rows, gmem, lay.total)
                                        {
                                            Some(x) => x,
                                            None => {
                                                why = "unpriced";
                                                continue;
                                            }
                                        }
                                    }
                                };
                                why = "cost";
                                if forced.is_some() && fmt.is_some() && forced_mma.is_none() {
                                    forced_mma = Some(st.clone());
                                }
                                if best.as_ref().is_none_or(|x| ns < x.1) {
                                    best = Some((st, ns));
                                }
                            }
                        }
                    }
                }
            }
        }
        let (mut st, ns) = best.ok_or(why)?;
        if forced.is_some() {
            return Ok(Some(forced_mma.unwrap_or(st)));
        }
        let split = self
            .split_ns(&priced, members, rows, gmem)
            .ok_or("unpriced")?;
        st.ns = (ns.round() as u64, split.round() as u64);
        self.stream_priced.set(Some((st.ns.0, st.ns.1)));
        if ns >= split {
            return Err("cost");
        }
        Ok(Some(st))
    }

    /// **The streamed unit's time (R1)** on the part's plan rates: one
    /// launch; the products (`2·b·kb·(d + c)` flops a window and row block,
    /// each axis cell's `exp` path as [`STREAM_CELL_FLOPS`]) at the shipped
    /// FFMA tile's rate, the grid's waves charging a partial last wave its
    /// busiest SM; each block's window loads from L2 (`K`/`V` re-read by every
    /// row block) and the unique operand and output bytes at the rate of
    /// where they live (`residence_plan`); and each window's barriers, a
    /// block's dependent chain the other resident blocks hide only in part:
    /// `nb · waves · 4 · bar_ns`.
    pub(super) fn stream_ns(
        &self,
        priced: &crate::profile::TargetProfile,
        st: &Stream,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        smem: u64,
    ) -> Option<f64> {
        let (gpu, d) = priced.plan_rates()?;
        if d.flops_per_ns == 0 {
            return None;
        }
        let w = st.group.axis;
        // R3: each block walks its slice's windows.
        let nb = st.nb() / st.s;
        let blocks = rows / st.b * st.s;
        let per_sm = st.per_sm(gpu, members, smem);
        // Latency hiding: an SM whose resident warps leave a scheduler fewer
        // than two to switch between stalls on every load and barrier — such
        // a geometry is not priced (2 warps an SM ran the 2048-token prefill
        // at 2.9× its 8-warp geometry, `flash-attention.md`).
        if per_sm * st.threads / gpu.warp.max(1) < 2 * gpu.schedulers.max(1) {
            return None;
        }
        let slots = gpu.sms * per_sm;
        let waves = blocks.div_ceil(slots);
        let (mut fma, mut win_bytes, mut cells) = (0u64, 0u64, 0u64);
        for (k, mb) in members.iter().enumerate() {
            match (&st.group.roles[k], &mb.kind) {
                (StreamRole::Produce, UKind::Tile { site }) => {
                    fma += site.k;
                    let ea = if site.a.narrow.is_some() { 2 } else { 4 };
                    win_bytes += st.b * site.k * ea + nb * st.kb * site.k * self::win_bytes(site);
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    fma += site.c;
                    win_bytes += nb * st.kb * site.c * self::win_bytes(site);
                }
                (StreamRole::Cell { .. }, _) => cells += 1,
                _ => {}
            }
        }
        // F5: the windows the blocks visit (each block its band of windows
        // its rows see, `visible_band`'s rule on the host).
        let seen = st.visited(rows);
        let flops =
            ((2 * rows * w * fma + rows * w * cells * STREAM_CELL_FLOPS) as f64 * seen) as u64;
        // bf16-class `finite-check`: a checked consumer's column bytes in
        // the windows each block skips, read for the check (no products).
        let check: u64 = st.skip().map_or(0, |gd| {
            gd.check
                .iter()
                .filter_map(|&k| match &members.get(k)?.kind {
                    UKind::Tile { site } => Some(nb * st.kb * site.c * self::win_bytes(site)),
                    _ => None,
                })
                .sum()
        });
        let win_bytes = (win_bytes as f64 * seen + check as f64 * (1.0 - seen)) as u64;
        // attention-chains: the busiest SM — its `⌈blocks / sms⌉` blocks
        // carry their share of the products at the SM's rate (a partial wave
        // of `per_sm`-block slots no slower: one block alone on an SM issues
        // at the SM's rate), unless its resident warps leave a scheduler
        // fewer than two (a grid smaller than the SMs' slots: its blocks
        // stall on every load and barrier, `starve` ×). The A100's 64-block
        // grids of 256 threads ran 0.60× the slot-filled price (attn 2048 ×
        // 128 `b64xkb32:s2`: 0.445 ms against 0.740 priced).
        let busiest = blocks.div_ceil(gpu.sms);
        let warps_res = busiest.min(per_sm) * st.threads / gpu.warp.max(1);
        let starve = ((2 * gpu.schedulers.max(1)) as f64 / warps_res.max(1) as f64).max(1.0);
        let fill = (busiest * gpu.sms) as f64 / blocks as f64 * starve;
        // `stream-mma`: the products at the matrix product's peak, at the
        // shipped FFMA tile's fraction of its own peak.
        let rate = match st
            .mma
            .and_then(|f| gpu.caps.matrix_op(f, crate::profile::Home::Regs))
        {
            Some(op) => {
                let frac = d.flops_per_ns as f64 / gpu.ffma_flops_per_ns().max(1) as f64;
                (gpu.matrix_flops_per_ns(&op) as f64 * frac).max(d.flops_per_ns as f64)
            }
            None => d.flops_per_ns as f64,
        };
        let fma_ns = flops as f64 / rate * fill;
        // The products' issue on one SM: each window's warp FFMA at the
        // part's FP32 lanes (at the shipped tile's fraction of peak) against
        // its shared-memory wavefronts (one 128-byte wavefront a clock, the
        // ISA's 32 banks × 4 B): a lane group's row operand a broadcast, its
        // `g` lanes' distinct column rows `g · 16` bytes a 16-byte load.
        let warps = st.threads / gpu.warp.max(1);
        let wf = |lanes: u64| (lanes * 16).div_ceil(gpu::GPU_WAVEFRONT_BYTES);
        // And their shared loads' issue: a lane group's row operand and its
        // window rows, a 16-byte load each (4 elements of `k`), at the part's
        // measured LSU issue rate (`lsu_ipc_pm`): the 2048-token prefill's
        // geometries rank by it (`flash-attention.md`).
        let (mut ffma, mut lds, mut lsu) = (0u64, 0u64, 0u64);
        // sweep-attention-2 `stream-mma`: the products' flops a window on the
        // matrix product, and the register packs (`cvt`) of their operands.
        let mma_op = st
            .mma
            .and_then(|f| gpu.caps.matrix_op(f, crate::profile::Home::Regs));
        let (mut mma_flops, mut packs) = (0u64, 0u64);
        for (k, mb) in members.iter().enumerate() {
            match (&st.group.roles[k], &mb.kind) {
                // A warp's `b` fragments a window: two 8-byte loads an `n`
                // tile and `k`-step (two wavefronts each), packed in pairs;
                // its `a` fragments are the registers loaded once.
                (StreamRole::Produce, UKind::Tile { site }) if mma_op.is_some() => {
                    mma_flops += 2 * st.b * st.kb * site.k;
                    let loads = warps * (site.k / MMA_K) * (st.kb / MMA_N) * 2;
                    lds += 2 * loads;
                    lsu += loads;
                    packs += loads * gpu.warp;
                }
                // Its `b` fragments: four 4-byte loads an `n` tile and
                // `k`-step (one wavefront each); its `a` fragments are the
                // producer's cells, packed.
                (StreamRole::Consume { .. }, UKind::Tile { site }) if mma_op.is_some() => {
                    mma_flops += 2 * st.b * st.kb * site.c;
                    let loads = warps * (st.kb / MMA_K) * (site.c / MMA_N) * 4;
                    lds += loads;
                    lsu += loads;
                    packs += (loads / 2 + warps * (st.kb / MMA_K) * 4) * gpu.warp;
                }
                (StreamRole::Produce, UKind::Tile { site }) => {
                    ffma += st.b * st.kb * site.k / gpu.warp.max(1);
                    lds += warps * site.k.div_ceil(4) * (st.rt + st.ct() * wf(st.g));
                    let v = if site.k.is_multiple_of(4) { 1 } else { 4 };
                    lsu += warps * site.k.div_ceil(4) * (st.rt + st.ct()) * v;
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    ffma += st.b * st.kb * site.c / gpu.warp.max(1);
                    let (oc, contig) = st.oc(site.c);
                    lds += warps * (st.kb / 4) * (st.rt + 4 * oc.div_ceil(4) * wf(st.g));
                    let v = if contig { oc.div_ceil(4) } else { oc };
                    lsu += warps * (st.kb / 4) * (st.rt + 4 * v);
                }
                _ => {}
            }
        }
        let lsu_ipc = (d.lsu_ipc_pm.max(1) as f64 / 1000.0).min(4.0);
        // The shipped tile's fraction of FFMA peak: the stalls an issue-bound
        // loop of products and loads pays beside its issue.
        let peak = gpu.ffma_flops_per_ns().max(1) as f64;
        let eff = (d.flops_per_ns as f64 / peak).clamp(0.05, 1.0);
        // Every thread instruction a window issues beside the products: each
        // axis cell's path ([`STREAM_CELL_INSTR`], its `exp`), and each row's
        // window work a thread repeats — the butterfly of its extreme
        // (`log2 g` shuffle steps), each rescale's `exp` ([`STREAM_ROW_INSTR`])
        // — at the part's FP32 lanes an SM issues a clock.
        let (mut cells_w, mut rescales) = (0u64, 0u64);
        for r in &st.group.roles {
            match r {
                StreamRole::Cell { .. } => cells_w += 1,
                StreamRole::Fold { laws } | StreamRole::Consume { laws } => {
                    rescales += laws.iter().filter(|l| l.law == DepLaw::Rescale).count() as u64
                }
                _ => {}
            }
        }
        let steps = u64::from(st.g.max(2).ilog2());
        let thr = ffma * gpu.warp
            + packs
            + st.b * st.kb * cells_w * STREAM_CELL_INSTR
            + st.threads
                * (st.rt * (4 * steps + STREAM_ROW_INSTR * rescales.min(1)) + STREAM_ROW_INSTR);
        let issue = thr as f64 / gpu.fp32_lanes.max(1) as f64;
        // A window's four barriers stall its block; the other resident
        // blocks' issue hides them in part.
        let bar_cycles = 4.0 * gpu.bar_ns() as f64 * gpu.clock_khz as f64 / 1e6 / per_sm as f64;
        // `stream-mma`: the products' issue on the SM's matrix pipe at the
        // part's rate for the instruction (`calib.cu matrix`).
        let mma_cycles = match &mma_op {
            Some(op) => mma_flops as f64 / gpu.matrix_flops_per_clk(op).max(1) as f64,
            None => 0.0,
        };
        // attention-chains: the FFMA window's products and their shared
        // loads largely in series — a `k`-step's products wait on its loads,
        // and at the stream's residency (two warps a scheduler) little other
        // issue covers them: the window's cycles are the longer of its thread
        // instructions at the FP32 lanes and its 16-byte warp loads at the
        // part's broadcast-load rate (`lds_bcast_pm`, `calib.cu lds bcast`:
        // 0.24–0.25 a clock on the RTX 5080 and the A100 alike), plus
        // [`STREAM_LOAD_SERIAL`] of the shorter. The larger of the two at the
        // FFMA tile's efficiency priced the A100's one-row-a-thread
        // geometries 1.5–1.9× fast — its 64 FP32 lanes double the issue
        // term, which hid the loads (attn 2048 × 128 `b8xkb32:rt1g8t64`:
        // 0.479 ms against 0.256 priced; the four-row `b32xkb32:rt4g16t128`
        // 0.316 against 0.314). `stream-mma`: the matrix pipe's issue as
        // before.
        let cycles = match mma_op {
            Some(_) => {
                issue
                    .max(lds as f64)
                    .max(lsu as f64 / lsu_ipc)
                    .max(mma_cycles)
                    / eff
                    + bar_cycles
            }
            None => {
                let loads = lsu as f64 / d.lds_bcast(gpu.warp);
                (issue.max(loads) + issue.min(loads) * STREAM_LOAD_SERIAL).max(lds as f64)
                    + bar_cycles
            }
        };
        // Every resident block of an SM shares its pipes: a wave of `per_sm`.
        let windows = (blocks as f64 / gpu.sms as f64).ceil() * nb as f64 * seen;
        let pipe_ns = cycles * windows * starve * 1e6 / gpu.clock_khz.max(1) as f64;
        let fma_ns = fma_ns.max(pipe_ns);
        let l2_ns = (blocks * win_bytes) as f64 / d.l2_bytes_per_ns.max(1) as f64;
        let res = self.ir.residence_plan(self.f);
        let rate = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let bytes = |o: ObjectId| self.array_bytes(&self.obj_ty(o)) as f64;
        let mem_ns: f64 = gmem.iter().map(|&o| bytes(o) / rate(o)).sum::<f64>()
            + members
                .iter()
                .filter(|mb| mb.store)
                .map(|mb| bytes(mb.out) / rate(mb.out))
                .sum::<f64>();
        // The barriers are each window's (in its cycles above).
        let chain = 0.0;
        // R3: the row block's last block reads the slices' partials back
        // (from L2, the latency of the ticket's fence and atomic first).
        let combine = match st.s {
            1 => 0.0,
            sp => {
                let per_row: u64 = members
                    .iter()
                    .enumerate()
                    .map(|(k, mb)| match (&st.group.roles[k], &mb.kind) {
                        (StreamRole::Consume { .. }, UKind::Tile { site }) => site.c,
                        (StreamRole::Fold { .. }, _) => 1,
                        _ => 0,
                    })
                    .sum();
                let sm_rate = (d.l2_bytes_per_ns as f64 / gpu.sms.max(1) as f64).max(1.0);
                2.0 * d.lat_ns as f64 + (sp * st.b * per_row * 4) as f64 / sm_rate
            }
        };
        let launch = self
            .profile
            .op_ns(f64::MAX, 0)
            .unwrap_or(d.launch_ns as f64);
        // Each wave's blocks wait their first windows' loads (the prologue
        // has nothing to overlap) and, split, the ticket's fence and atomic:
        // one loaded latency each (`lat_ns`).
        let fixed = (waves * (1 + u64::from(st.s > 1))) as f64 * d.lat_ns as f64;
        Some(launch + fma_ns.max(l2_ns).max(mem_ns) + chain + combine + fixed)
    }

    /// The split realization's time: each tile site's best candidate plan,
    /// the members between tiles as one launch a segment reading and writing
    /// the axis-wide rows at the rate of where they live — `block_cost`'s
    /// split side, independent of any block.
    pub(super) fn split_ns(
        &self,
        priced: &crate::profile::TargetProfile,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
    ) -> Option<f64> {
        let (gpu, d) = priced.plan_rates()?;
        let free = self.order == crate::Order::Free;
        let res = self.ir.residence_plan(self.f);
        let at = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let roof = |f: u64, by_ns: f64| (f as f64 / d.flops_per_ns.max(1) as f64).max(by_ns);
        let launch = |prev: f64| {
            priced
                .op_ns(prev, 0)
                .map_or(d.launch_ns as f64, |x| x.round())
        };
        let seg = segments(members);
        let e = 4u64;
        let mut split = 0f64;
        let mut seg_cost: Vec<(u64, f64)> = vec![(0, 0.0); seg.last().map_or(0, |s| s + 1)];
        for (k, mb) in members.iter().enumerate() {
            match &mb.kind {
                UKind::Tile { site } => {
                    let l = gpu::launch(site, self.profile, self.contract, self.gpu_arms);
                    let order = if free {
                        crate::Order::Free
                    } else {
                        crate::Order::Sequential
                    };
                    let own = gpu::plan_cands(
                        &l,
                        site,
                        self.profile,
                        self.contract,
                        self.gpu_arms,
                        order,
                    )
                    .first()
                    .map(|c| c.cost.ns)
                    .or_else(|| {
                        let shape = gpu::plan_shape(site, &l, gpu::Combine::None, gpu);
                        priced.site_ns(&shape).map(|c| c.ns)
                    })?;
                    // Its stored rows cross memory for the next segment.
                    // Its own launch's first loads wait a loaded latency,
                    // as each wave of the streamed unit's (`stream_ns`).
                    split += own + (rows * mb.w * e) as f64 / at(mb.out) + d.lat_ns as f64;
                }
                _ => {
                    let work = match &mb.kind {
                        UKind::Fold { site, .. } => rows * site.k,
                        _ => rows * mb.w,
                    };
                    let gm: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Gmem(g) => Some((work * e) as f64 / at(gmem[*g])),
                            _ => None,
                        })
                        .sum();
                    let crossed: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Member { k: p, .. } if seg[*p] != seg[k] => {
                                let pm = &members[*p];
                                Some((rows * pm.w * e) as f64 / at(pm.out))
                            }
                            _ => None,
                        })
                        .sum();
                    let read_later = members.iter().enumerate().any(|(j, r)| {
                        seg[j] != seg[k]
                            && r.srcs
                                .iter()
                                .any(|s| matches!(s, USrc::Member { k: p, .. } if *p == k))
                    });
                    let written = match read_later {
                        true => (rows * mb.w * e) as f64 / at(mb.out),
                        false => 0.0,
                    };
                    let x = &mut seg_cost[seg[k]];
                    x.0 += work;
                    x.1 += gm + crossed + written;
                }
            }
        }
        let mut prev = f64::MAX;
        for &(f, by) in seg_cost.iter().filter(|x| x.0 > 0) {
            split += launch(prev) + roof(f, by) + d.lat_ns as f64;
            prev = roof(f, by);
        }
        Some(split)
    }
}

/// attention-chains: the share of the shorter of a stream window's two
/// issue streams — its products' thread instructions at the FP32 lanes, its
/// 16-byte shared loads at the broadcast-load rate — that adds to the
/// longer (0: they overlap; 1: in series). ½: the geometry sweeps of
/// `attn_2048_128` / `attn_2048_64` / `attn_1024_64` / `llm_pre_t8192` on two
/// A100 SXM4 boxes and of `attn_2048_128` on the RTX 5080 fit within an rms
/// of 1.13 / 1.07 of their geomean at ½ (in series 1.14 / 1.05, overlapped
/// 1.18 / 1.11; `attention-chains.md`).
pub(crate) const STREAM_LOAD_SERIAL: f64 = 0.5;

/// The flops one axis cell's evaluation is priced at against the products:
/// the default face's `exp` (22 SASS on the RTX 5080, `exp.md`) and its
/// operands' few.
pub(crate) const STREAM_CELL_FLOPS: u64 = 48;

/// The thread instructions one axis cell's path issues beside the products
/// (its `exp`'s 22 SASS on the RTX 5080, `exp.md`, its fold steps and its
/// row operand's store).
pub(crate) const STREAM_CELL_INSTR: u64 = 24;

/// The thread instructions one row's rescale (an `exp` and its select), or a
/// window's loop and waits, issue.
pub(crate) const STREAM_ROW_INSTR: u64 = 30;
