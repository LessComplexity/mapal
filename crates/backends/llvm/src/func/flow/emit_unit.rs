//! a unit's cost and roofline, `emit_unit` and the kernel body drivers

use super::*;

impl<'a> FnEmit<'a> {
    /// The plan model's reading of a row-block unit against the launches it
    /// replaces, `(fused, split)` ns, `None` without plan rates. Fused: each
    /// tile as a `b × c` tile of `rows ÷ b` blocks folding all of `k`
    /// ([`TargetProfile::site_ns`]: its FFMA, the column operand every block
    /// re-reads from L2, the latency chain), the members between tiles on
    /// chip, the stored outputs, one launch. Split: each tile site's best
    /// candidate plan, the members between tiles as one launch per segment
    /// reading and writing the intermediates in memory. Every byte stream
    /// moves at the rate of where its value lives (`residence_plan` against
    /// the part's L2: `l2_bytes_per_ns` or `dram_bytes_per_ns`), the same
    /// rule on every part.
    pub(super) fn block_cost(&self, u: &Unit, geo: &Geo) -> Option<(u64, u64)> {
        let (gpu, d) = self.profile.plan_rates()?;
        if d.flops_per_ns == 0 {
            return None;
        }
        let blk = u.block.as_ref()?;
        let free = self.order == crate::Order::Free;
        let (b, lanes, rows) = (blk.b, geo.lanes, u.rows);
        let blocks = rows / b;
        let regs = 2 * crate::profile::FMA_ADDR_REGS;
        let per_sm = [
            gpu.max_blocks_per_sm,
            gpu.smem_per_sm / (geo.smem + gpu.smem_reserved_per_block).max(1),
            gpu.max_threads_per_sm / lanes,
            gpu.regs_per_sm / (lanes * regs),
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
        .max(1);
        // kernel-merging: where each value's bytes live decides their rate.
        let res = self.ir.residence_plan(self.f);
        let at = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let roof = |f: u64, by_ns: f64| (f as f64 / d.flops_per_ns as f64).max(by_ns);
        let launch = |prev: f64| {
            self.profile
                .op_ns(prev, 0)
                .map_or(d.launch_ns as f64, |x| x.round())
        };
        let seg = segments(&u.members);
        let elem = |mb: &UMember| {
            crate::profile::elem_bytes(super::super::resident::array_elem(&self.obj_ty(mb.out)))
        };
        let (mut fused, mut split) = (launch(f64::MAX), 0f64);
        let mut seg_cost: Vec<(u64, f64)> = vec![(0, 0.0); seg.last().map_or(0, |s| s + 1)];
        for (k, mb) in u.members.iter().enumerate() {
            let e = elem(mb);
            let stored = if mb.store { rows * mb.w * e } else { 0 };
            fused += stored as f64 / at(mb.out);
            match &mb.kind {
                UKind::Tile { site } => {
                    let (nj, slices) = tile_split(site, lanes, free);
                    let kper = site.k / slices;
                    let step = [16, 8, 4, 2, 1]
                        .into_iter()
                        .find(|s| kper % s == 0)
                        .unwrap_or(1);
                    // A block's k-slices are columns of a `b × c·slices`
                    // tile folding `k ÷ slices` (the same FFMA, each thread's
                    // own trips); 4-wide loads where `k` is contiguous.
                    let vec = |ck: u64| if ck == 1 && step % 4 == 0 { 4 } else { 1 };
                    let mut shape = crate::profile::PlanShape {
                        rows,
                        c: site.c,
                        k: site.k,
                        e,
                        ea: site.a.elem_bytes(&site.elem),
                        eb: site.b.elem_bytes(&site.elem),
                        bm: b,
                        bn: site.c * slices,
                        threads: lanes,
                        per_sm,
                        acc: b * nj,
                        tiles: blocks,
                        blocks,
                        kb: kper,
                        s: 1,
                        comb: crate::profile::PlanComb::None,
                        bk: step,
                        fly: 1,
                        lsu: b * step / vec(site.a.ck) + nj * step / vec(site.b.ck),
                    };
                    let col = (blocks * site.k * site.c * e) as f64;
                    match blk.stage.get(k).copied().flatten() {
                        // kernel-merging `row-stage`: the separate tile
                        // kernel's terms — a panel trip of `kc`, `stages − 1`
                        // panels in flight, its shared-memory reads (16-byte
                        // chunks along `k`; one element a column across `k`)
                        // and each thread's copies (`LDGSTS`, or `LDG` + `STS`).
                        // A buffer copies, waits and folds in turn: its
                        // latency chain adds to the trip instead of hiding.
                        Some(st) => {
                            let v = 16 / e;
                            let (ca, cb) = st.copies(b, site.c, slices, e, lanes);
                            let a_rd = match &mb.srcs[0] {
                                USrc::Gmem(_) => b * st.kc / v,
                                _ => b * st.kc / vec(site.a.ck),
                            };
                            let njp = nj / st.passes;
                            let b_rd = match st.b_kmaj {
                                true => njp * st.kc / v,
                                false => njp * st.kc,
                            };
                            let copies = (ca + cb) * if st.cp { 1 } else { 2 };
                            // Each column pass a `b × c / passes` tile of its
                            // own (the row operand re-read a pass).
                            shape.c = site.c / st.passes;
                            shape.bn = site.c * slices / st.passes;
                            shape.acc = b * njp;
                            shape.bk = st.kc;
                            shape.fly = (st.stages - 1).max(1);
                            shape.lsu = a_rd + b_rd + copies;
                            let pc = self.profile.site_ns(&shape)?;
                            fused += st.passes as f64
                                * match st.stages {
                                    1 => (pc.f_ns.max(pc.l_ns) + pc.c_ns).max(pc.d_ns),
                                    _ => pc.kernel_ns,
                                };
                            // Each column's run of a trip is `kc` elements:
                            // under a 32-byte sector the rest crosses unused.
                            let run = match (st.b_kmaj, st.b_vec || site.b.clane == 1) {
                                (true, _) => st.kc * e,
                                (false, true) => SECTOR_BYTES,
                                (false, false) => e,
                            };
                            let eff = (run as f64 / SECTOR_BYTES as f64).min(1.0);
                            fused += (1.0 / eff - 1.0) * col / d.l2_bytes_per_ns.max(1) as f64;
                        }
                        None => {
                            fused += self.profile.site_ns(&shape)?.kernel_ns;
                            // The column operand is read straight from `Gmem`
                            // (no shared panel): a warp's lanes are `clane`
                            // elements apart, so unless they are adjacent each
                            // lane's run along `k` (a 4-wide load, else one
                            // element) fills part of each 32-byte sector it
                            // touches. The unused bytes still cross from L2
                            // (rtx-5080 attn_1024_64: the q·Kᵀ phase alone
                            // 20.2 us in the unit against 9.4 us as its tile
                            // site, CUPTI).
                            let run = vec(site.b.ck) * e;
                            let eff = match site.b.clane {
                                1 => 1.0,
                                _ => (run as f64 / SECTOR_BYTES as f64).min(1.0),
                            };
                            fused += (1.0 / eff - 1.0) * col / d.l2_bytes_per_ns.max(1) as f64;
                        }
                    }
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
                        self.profile.site_ns(&shape).map(|c| c.ns)
                    })?;
                    split += own;
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
                            USrc::Gmem(g) => Some((work * e) as f64 / at(u.gmem[*g])),
                            _ => None,
                        })
                        .sum();
                    // Split: every member row it reads crosses memory, and a
                    // row a later segment reads is written there.
                    let crossed: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Member { k: p, .. } if seg[*p] != seg[k] => {
                                let pm = &u.members[*p];
                                Some((rows * pm.w * e) as f64 / at(pm.out))
                            }
                            _ => None,
                        })
                        .sum();
                    let read_later = u.members.iter().enumerate().any(|(j, r)| {
                        seg[j] != seg[k]
                            && r.srcs
                                .iter()
                                .any(|s| matches!(s, USrc::Member { k: p, .. } if *p == k))
                    });
                    let written = if read_later {
                        (rows * mb.w * e) as f64 / at(mb.out)
                    } else {
                        0.0
                    };
                    let x = &mut seg_cost[seg[k]];
                    x.0 += work;
                    x.1 += gm + crossed + written;
                    fused += roof(work, gm);
                }
            }
        }
        let mut prev = f64::MAX;
        for &(f, by) in seg_cost.iter().filter(|x| x.0 > 0) {
            split += launch(prev) + roof(f, by);
            prev = roof(f, by);
        }
        Some((fused.round() as u64, split.round() as u64))
    }

    /// `(flops, bytes)` of the fused launch — every member evaluation, the
    /// recomputed halo and in-row re-reads included — and of each member's
    /// own launch.
    pub(super) fn unit_roofline(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        elem: impl Fn(ObjectId) -> u64,
    ) -> ((u64, u64), Vec<(u64, u64)>) {
        let steps = |mb: &UMember| match &mb.kind {
            UKind::Fold { site, .. } => rows * site.k,
            UKind::Cell { .. } => rows * mb.w,
            UKind::Tile { site } => rows * site.c * site.k,
        };
        // Evaluations: a fold walks its row once; a per-row cell runs once a
        // row; a cell of `w > 1` is swept when stored and evaluated at every
        // read a register does not serve.
        let mut evals: Vec<u64> = members
            .iter()
            .map(|mb| match mb.kind {
                UKind::Fold { .. } | UKind::Tile { .. } => steps(mb),
                UKind::Cell { .. } if mb.per_row() || mb.store => steps(mb),
                UKind::Cell { .. } => 0,
            })
            .collect();
        for i in (0..members.len()).rev() {
            for s in &members[i].srcs {
                if let USrc::Member { k, row, .. } = s
                    && !(members[*k].per_row() && *row)
                {
                    evals[*k] += evals[i];
                }
            }
        }
        let gmem_reads = |mb: &UMember| {
            mb.srcs
                .iter()
                .filter(|s| matches!(s, USrc::Gmem(_)))
                .count() as u64
        };
        let e = gmem.first().map_or(4, |&o| elem(o));
        let flops: u64 = evals.iter().sum();
        let bytes: u64 = members
            .iter()
            .zip(&evals)
            .map(|(mb, &n)| n * gmem_reads(mb) * e + if mb.store { rows * mb.w * e } else { 0 })
            .sum();
        let own = members
            .iter()
            .map(|mb| {
                let reads = mb
                    .srcs
                    .iter()
                    .filter(|s| matches!(s, USrc::Gmem(_) | USrc::Member { .. }))
                    .count() as u64;
                match &mb.kind {
                    UKind::Fold { site, .. } => (rows * site.k, (rows * site.k + rows) * e),
                    UKind::Cell { .. } | UKind::Tile { .. } => {
                        (steps(mb), (reads + 1) * steps(mb) * e)
                    }
                }
            })
            .collect();
        ((flops, bytes), own)
    }

    /// The host half of a unit: its stored values' buffers, its operands'
    /// device copies (copied in here only when nothing placed one), one
    /// launch, and the copy-out of a stored value the host reads whole.
    pub(in crate::func) fn emit_unit(&mut self, u: &Unit) {
        let mut ptrs = Vec::new();
        let mut stored = Vec::new();
        for mb in u.members.iter().filter(|mb| mb.store) {
            let bytes = self.array_bytes(&self.obj_ty(mb.out));
            let d = self.dalloc(mb.out, bytes);
            ptrs.push(d.clone());
            stored.push((mb.out, d));
        }
        let mut temps = Vec::new();
        let Geo {
            gx, block, smem, ..
        } = self.geo(u);
        for (gi, &o) in u.gmem.iter().enumerate() {
            let at = self.resident.as_ref().map_or(o, |p| p.alias(o));
            let d = if self.dvalid.contains_key(at) {
                self.dload(o)
            } else {
                let ob = self.array_bytes(&self.obj_ty(o));
                let h = self.array_operand_ptr(o, None).expect("unit operand ptr");
                let s = format!("%d{}", self.fresh());
                self.allocas
                    .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
                let d = self.tmp();
                self.line(format!(
                    "{d} = call i64 @mapal_gpu_alloc(ptr {s}, i64 {ob})"
                ));
                self.line(format!(
                    "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {ob})"
                ));
                temps.push(s);
                d
            };
            // sweep-attention `unit-uniform`: the pointer moved by its
            // captured numbers' terms (in bounds: the fact's capture range).
            let mv = u.gmove.get(gi).cloned().flatten();
            match mv.and_then(|mv| self.uniform_bytes(mv.source, &mv.terms, mv.elem)) {
                Some(off) => {
                    let q = self.tmp();
                    self.line(format!("{q} = add i64 {d}, {off}"));
                    ptrs.push(q);
                }
                None => ptrs.push(d),
            }
        }
        // kernel-merging `zero-fold`: later sites' zero passes as its tail.
        self.zero_args(u.members[0].m, &gx.to_string(), block, &mut ptrs);
        // mask-frontend `stream-numbers`: the epilogues' numbers, as the
        // host holds them this launch.
        for (k, s) in stream_numbers(u) {
            let source = self.ir.morphism(u.members[k].m).expect("member").source;
            let (t, v) = self
                .load_component(source, s)
                .expect("a captured number loads");
            let w = self.tmp();
            self.line(gpu::uniform_to_i64(&w, &v, &t));
            ptrs.push(w);
        }
        // long-prefill `slice-launches`: one launch a phase, its number last.
        let phased = u
            .slices
            .as_ref()
            .filter(|sl| sl.launches)
            .map(|sl| sl.phases);
        if phased.is_some() {
            ptrs.push("0".to_string());
        }
        let args = format!("%dv{}", self.fresh());
        let np = ptrs.len();
        self.allocas
            .push_str(&format!("  {args} = alloca [{np} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        // long-rows `row-slices`: phases meet at grid barriers — every block
        // co-resident (the runtime caps the grid at its occupancy; the items
        // loop strides).
        let entry = match u.slices.as_ref().is_some_and(Slices::coop) {
            true => "launch_coop",
            false => "launchv",
        };
        for p in 0..phased.unwrap_or(1) {
            if p > 0 {
                let g = self.tmp();
                self.line(format!(
                    "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {}",
                    np - 1
                ));
                self.line(format!("store i64 {p}, ptr {g}"));
            }
            self.line(format!(
                "call void @mapal_gpu_{entry}(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 {smem}) ; unit",
                u.id as i64
            ));
        }
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            self.dvalid.insert(o, ());
            let keep = self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(o) || p.is_carry_next(o));
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("unit output slot");
                self.d2h(0, &format!("nocapture {h}"), &d, "0", bytes);
            }
        }
    }

    /// The unit's geometry on this face ([`unit_geo`]).
    pub(crate) fn geo(&self, u: &Unit) -> Geo {
        unit_geo(
            u,
            self.profile,
            self.order == crate::Order::Free,
            |mb| match &mb.kind {
                UKind::Fold { site, .. } => crate::profile::elem_bytes(&site.elem),
                UKind::Tile { site } => crate::profile::elem_bytes(&site.elem),
                UKind::Cell { .. } => crate::profile::elem_bytes(
                    super::super::resident::array_elem(&self.obj_ty(mb.out)),
                ),
            },
        )
    }

    /// This function's unit kernels for the device module.
    pub(crate) fn unit_kernels(&self) -> Vec<UnitKernel> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.units
            .values()
            .map(|u| {
                let (fns, text) = self.unit_kernel(u);
                UnitKernel {
                    f: self.f,
                    members: u.members.iter().map(|mb| mb.m).collect(),
                    fns,
                    text,
                }
            })
            .collect()
    }

    /// A body's text as a unit calls it: the host's, or the variant whose
    /// capture `slots` arrive as one-cell buffers (`_u<slots>`).
    pub(super) fn unit_body(&self, f: FuncId, slots: &[u32]) -> (String, String) {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        if slots.is_empty() {
            return (self.fnames[f].clone(), fe.emit());
        }
        fe.set_cell_slots(slots.to_vec());
        (gpu::slots_variant(&self.fnames[f], slots), fe.emit())
    }

    /// `tile-post`: map body `f`'s epilogue variant (its fold `fold`'s value
    /// an argument), the capture `slots` as one-cell buffers
    /// (`<name>[_u<slots>]_post`).
    pub(super) fn unit_post_body(
        &self,
        f: FuncId,
        fold: MorphismId,
        slots: &[u32],
    ) -> (String, String) {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        fe.set_post_fold(Some(fold));
        if !slots.is_empty() {
            fe.set_cell_slots(slots.to_vec());
        }
        let name = gpu::slots_variant(&self.fnames[f], slots);
        (gpu::post_variant(&name), fe.emit())
    }
}
