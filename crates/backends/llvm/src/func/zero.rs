//! **kernel-merging `zero-fold`: a zero pass is work of an earlier kernel.**
//! An atomic site (split-K `Atomic`, stream-K) adds into an output that must
//! read zero when it starts; the zero pass that makes it so has been its own
//! op (`cuMemsetD8Async`, a host issue and a device gap a site). It needs no
//! op of its own: any kernel the same stream runs between the last earlier
//! access to the output's bytes and the site can store the zeros as extra
//! work. The order comes from the dependency fact (`access_plan`: the bytes
//! each morphism reads and writes, storage classes) lifted onto this
//! function's kernels (a unit's members, a site's epilogue chain) with the
//! backend's own byte sharing (residency aliases, carries, arena offsets).
//!
//! The rule, per sequence (a loop trip's advance cone; the straight-line walk
//! with each loop as one step): for a site `L` whose candidates include an
//! atomic combine, walk back from `L`; the first step that touches `L`'s
//! output bytes ends the walk; the nearest step before it that is a cell or
//! unit kernel `E` of this plan takes `L`'s zero as a grid-stride tail, when
//! the plan rates price the tail (`bytes` at the rate `E`'s blocks store,
//! capped at the part's measured tail rate) below the zero op (`zero_ns` +
//! `bytes` at the measured fill rate + the idle it leaves): a small zero
//! always (its op is fixed cost), a large one never where the tail stores
//! slower than the fill (rtx-5080: 1740 vs 2320 B/ns, crossover ≈ 7 MiB). `L` then launches with [`SITE_ZEROED`]: the
//! runtime skips a candidate's zero once the site's pick is settled (a trial
//! still zeroes before each candidate it times), and the direct atomic launch
//! drops its zero. Not folded: a guarded (gated) step touching the bytes, a
//! retained row (its range is the trip's), an epilogue group, a function on
//! streams (its order is the events'), a part without plan rates or the
//! measured fill and tail rates (`unpriced`), a `horizontal` member (its
//! merged tuples are unsplit). A tail rides a group member only at its last
//! (the group's launch), when no member touches the bytes and the merged
//! launch's arguments take the tail's. OFF: `--gpu-off=zero-fold`.

use super::resident::ResidentPlan;
use super::*;
use mapal_ir::AccessPlan;

/// `mapal_gpu_site` flags bit 1: an earlier kernel zeroed the output.
pub(crate) const SITE_ZEROED: u32 = 2;

/// Bytes one block of a zero tail stores a clock: 256 threads storing a 16-B
/// vector an iteration of about eight instructions, four schedulers issuing
/// (64 warp-instructions in 16 clocks for 4 KB) — a model bound on a grid of
/// few blocks, checked against the kernel's SASS and CUPTI time in the
/// evidence (`kernel-merging.md`). A wide grid stores at the part's measured
/// tail rate (`DeviceCost::tail_bytes_per_ns`).
const ZERO_BYTES_PER_BLOCK_CLOCK: f64 = 256.0;

/// One step of a sequence the rule walks.
pub(super) enum Step {
    /// A morphism emitted at this point.
    Op(MorphismId),
    /// A whole loop (the straight-line walk emits it at its `LoopEnter`).
    Loop(ObjectId),
}

impl<'a> FnEmit<'a> {
    /// **`zero-fold`** (module doc): fills `plan.zero_by` / `plan.zeroed` and
    /// gives each folding kernel its own id.
    pub(super) fn zero_fold(&self, plan: &mut ResidentPlan) {
        if !gpu::host_rungs(self.profile, self.contract, self.gpu_arms).zero_fold
            || !plan.chain.is_empty()
        {
            return;
        }
        // Priced only where the tail's and the fill's rates were measured.
        let Some((g, d)) = self
            .profile
            .plan_rates()
            .filter(|(_, d)| d.tail_bytes_per_ns > 0 && d.fill_bytes_per_ns > 0)
        else {
            return;
        };
        let acc = self.ir.access_plan(self.f);
        let sites = self.zero_sites(plan);
        if sites.is_empty() {
            return;
        }
        // Gated steps run where their guard's arm is emitted: a site whose
        // bytes one touches keeps its zero.
        let gated: Vec<(Vec<ObjectId>, Vec<ObjectId>)> = self
            .gated
            .keys()
            .map(|m| self.step_access(plan, &acc, m))
            .collect();
        // Why each atomic site keeps its zero op: a gated step touches its
        // bytes, the walk met an access first (`stopped`), no kernel took it
        // (`none`: no eligible kernel, its pointers full, or priced out).
        let (mut gated_n, mut stopped, mut none) = (0, 0, 0);
        for seq in self.zero_sequences() {
            let access: Vec<(Vec<ObjectId>, Vec<ObjectId>)> = seq
                .iter()
                .map(|s| match *s {
                    Step::Op(m) => self.step_access(plan, &acc, m),
                    Step::Loop(merge) => self.loop_access(plan, &acc, merge),
                })
                .collect();
            for (j, step) in seq.iter().enumerate() {
                let Step::Op(l) = *step else {
                    continue;
                };
                let Some(&bytes) = sites.get(l) else {
                    continue;
                };
                let out = self.ir.morphism(l).expect("site").target;
                let touches = |a: &(Vec<ObjectId>, Vec<ObjectId>)| {
                    a.0.iter()
                        .chain(&a.1)
                        .any(|&x| self.zero_shares(plan, &acc, x, out))
                };
                if gated.iter().any(touches) {
                    gated_n += 1;
                    continue;
                }
                let mut decided = false;
                for i in (0..j).rev() {
                    if touches(&access[i]) {
                        stopped += 1;
                        decided = true;
                        break;
                    }
                    let Step::Op(e) = seq[i] else {
                        continue;
                    };
                    let Some((blocks, base)) = self.zero_kernel(plan, e) else {
                        continue;
                    };
                    let jobs = plan.zero_by.get(e).map_or(0, Vec::len);
                    // kernel-merging `horizontal`: a member's tail runs in
                    // its group's launch, at the last member — only there,
                    // when no member touches the bytes and the merged
                    // launch's arguments take the tail's.
                    let others = match plan.hz_of.get(e) {
                        None => 0,
                        Some(&(g, k)) => {
                            let ms = &plan.hz[g].members;
                            // A `calls` group zeroes its own atomic outputs;
                            // its members take no tail.
                            if plan.hz[g].persistent()
                                || k + 1 != ms.len()
                                || ms.iter().any(|&x| touches(&self.hz_access(plan, &acc, x)))
                            {
                                continue;
                            }
                            ms[..k]
                                .iter()
                                .map(|&x| {
                                    self.hz_shape(plan, x, false)
                                        .map_or(gpu::LAUNCHV_MAX, |s| s.args)
                                })
                                .sum()
                        }
                    };
                    if others + base + 1 + 2 * (jobs + 1) > gpu::LAUNCHV_MAX {
                        continue;
                    }
                    // The plan rates: the tail at the rate `E`'s blocks
                    // store (the part's tail rate, few blocks their own
                    // issue) against the zero op it replaces (its fixed
                    // cost, its fill, the idle it leaves on the stream).
                    let own = d.zero_ns as f64 + bytes as f64 / d.fill_bytes_per_ns as f64;
                    let own = own + (d.gap_ns as f64).max(d.issue_ns as f64 - own);
                    let rate = (d.tail_bytes_per_ns as f64).min(
                        blocks.min(g.sms) as f64 * ZERO_BYTES_PER_BLOCK_CLOCK * g.clock_khz as f64
                            / 1e6,
                    );
                    if bytes as f64 / rate.max(1e-9) >= own {
                        continue;
                    }
                    match plan.zero_by.get_mut(e) {
                        Some(v) => v.push((l, bytes)),
                        None => {
                            plan.zero_by.insert(e, vec![(l, bytes)]);
                        }
                    }
                    plan.zeroed.insert(l, ());
                    decided = true;
                    break;
                }
                none += usize::from(!decided);
            }
        }
        plan.zero_line = format!(
            "; mapal.zero.host sites={} folded={} kernels={} gated={gated_n} stopped={stopped} none={none}\n",
            sites.len(),
            plan.zeroed.len(),
            plan.zero_by.len()
        );
        // A folding kernel is another kernel: its own id.
        let rekey = |id: u64, n: usize| {
            format!("{id:016x}|zero{n}")
                .bytes()
                .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };
        let folds: Vec<(MorphismId, usize)> =
            plan.zero_by.iter().map(|(e, v)| (e, v.len())).collect();
        for (e, n) in folds {
            if let Some(c) = plan.cells.get_mut(e) {
                c.id = rekey(c.id, n);
            } else if let Some(u) = plan.units.get_mut(e) {
                u.id = rekey(u.id, n);
            }
        }
    }

    /// The atomic sites the rule may fold, with their output bytes: device
    /// tile sites of the plain launch path (no epilogue group, no retained
    /// row, not a unit member) one of whose candidates adds into a zeroed
    /// output.
    fn zero_sites(&self, plan: &ResidentPlan) -> SecondaryMap<MorphismId, u64> {
        let mut out = SecondaryMap::new();
        let Some(tp) = &self.tile_plan else {
            return out;
        };
        for (m, site) in tp.sites.iter() {
            let target = self.ir.morphism(m).expect("site").target;
            // A `horizontal` member launches in its group's merged table,
            // whose tuples hold no atomic candidate: nothing to zero.
            if !self.device.contains_key(m)
                || plan.skip.contains_key(m)
                || plan.hz_of.contains_key(m)
                || plan.units.contains_key(m)
                || self.groups.contains_key(m)
                || self.retain.contains_key(target)
            {
                continue;
            }
            let atomic = gpu::site_cands(
                site,
                self.profile,
                self.contract,
                self.gpu_arms,
                self.order,
                false,
            )
            .iter()
            .map(|c| c.launch.combine)
            .chain([
                gpu::launch_in(site, self.profile, self.contract, self.gpu_arms, self.order)
                    .combine,
            ])
            .any(|c| {
                matches!(
                    c,
                    gpu::Combine::Atomic { .. } | gpu::Combine::StreamK { .. }
                )
            });
            if atomic {
                out.insert(m, self.array_bytes(&self.obj_ty(target)));
            }
        }
        out
    }

    /// A kernel that may take a zero tail: `(blocks, pointer arguments)` of
    /// a cell (not a batched write) or a unit launch.
    fn zero_kernel(&self, plan: &ResidentPlan, e: MorphismId) -> Option<(u64, usize)> {
        if let Some(c) = plan.cells.get(e) {
            if plan.batch.iter().any(|b| b.writes.iter().any(|w| w.0 == e)) {
                return None;
            }
            let cells = match c.range {
                Some(ou) => ou.len.min(c.n),
                None => c.n,
            };
            let blocks = (cells * c.lanes).div_ceil(gpu::NAIVE_THREADS).max(1);
            let base = 1 + c.arg_count() + if c.range.is_some() { 2 } else { 0 };
            return Some((blocks, base));
        }
        let u = plan.units.get(e)?;
        let base = u.members.iter().filter(|mb| mb.store).count() + u.gmem.len();
        Some((self.geo(u).gx, base))
    }

    /// The sequences the rule walks: each canonical loop's advance cone,
    /// then the straight-line walk (a loop one step).
    pub(super) fn zero_sequences(&self) -> Vec<Vec<Step>> {
        let mut out = Vec::new();
        let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut in_scc: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for scc in self.ir.loop_structure(self.f) {
            for &merge in &scc.merges {
                if let Some(lp) = self.ir.loop_plan(self.f, merge) {
                    for &m in lp.decide_order.iter().chain(&lp.advance_order) {
                        owned.insert(m, ());
                    }
                    out.push(
                        lp.advance_order
                            .iter()
                            .filter(|&&m| !self.gated.contains_key(m))
                            .map(|&m| match self.ir.morphism(m).expect("m").op {
                                Operation::LoopEnter => {
                                    Step::Loop(self.ir.morphism(m).expect("m").target)
                                }
                                _ => Step::Op(m),
                            })
                            .collect(),
                    );
                }
            }
            for o in scc.objects {
                in_scc.insert(o, ());
            }
        }
        let mut top = Vec::new();
        for m in self.ir.topo_order(self.f) {
            let mo = self.ir.morphism(m).expect("m");
            match mo.op {
                Operation::LoopEnter if !self.gated.contains_key(m) => {
                    top.push(Step::Loop(mo.target))
                }
                Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {}
                _ if owned.contains_key(m)
                    || in_scc.contains_key(mo.source)
                    || in_scc.contains_key(mo.target)
                    || self.gated.contains_key(m) => {}
                _ => top.push(Step::Op(m)),
            }
        }
        out.push(top);
        out
    }

    /// `(reads, writes)` of the device work emitted at `m`: a unit's members
    /// (inputs it does not make, stored values), a site with its epilogue
    /// chain; nothing for a member that runs inside another's launch; else
    /// the morphism's own.
    pub(super) fn step_access(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        m: MorphismId,
    ) -> (Vec<ObjectId>, Vec<ObjectId>) {
        if let Some(u) = plan.units.get(m) {
            let outs: Vec<ObjectId> = u.members.iter().map(|mb| mb.out).collect();
            let reads = u
                .members
                .iter()
                .flat_map(|mb| acc.reads(mb.m).iter().copied())
                .filter(|o| !outs.contains(o))
                .collect();
            let writes = u
                .members
                .iter()
                .filter(|mb| mb.store)
                .map(|mb| mb.out)
                .collect();
            return (reads, writes);
        }
        if plan.skip.contains_key(m) {
            return (Vec::new(), Vec::new());
        }
        let mut reads = acc.reads(m).to_vec();
        let mut writes = acc.writes(m).to_vec();
        if let Some(g) = self.groups.get(m) {
            for mb in &g.chain {
                reads.extend(acc.reads(mb.m));
                writes.push(mb.out);
            }
        }
        (reads, writes)
    }

    /// Every access of the loop at `merge`, as one step.
    pub(super) fn loop_access(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        merge: ObjectId,
    ) -> (Vec<ObjectId>, Vec<ObjectId>) {
        let Some(lp) = self.ir.loop_plan(self.f, merge) else {
            // Not canonical: it touches everything.
            let all: Vec<ObjectId> = self.ir.objects().map(|(o, _)| o).collect();
            return (all.clone(), all);
        };
        let (mut r, mut w) = (Vec::new(), Vec::new());
        for &m in lp.decide_order.iter().chain(&lp.advance_order) {
            let (a, b) = self.step_access(plan, acc, m);
            r.extend(a);
            w.extend(b);
        }
        (r, w)
    }

    /// May `x` and `y` share device bytes: one storage class
    /// (`access_plan`), one residency alias or carry buffer, or overlapping
    /// arena offsets?
    pub(super) fn zero_shares(
        &self,
        plan: &ResidentPlan,
        acc: &AccessPlan,
        x: ObjectId,
        y: ObjectId,
    ) -> bool {
        if acc.shares(x, y) || plan.alias(x) == plan.alias(y) {
            return true;
        }
        let carry = |o: ObjectId| {
            plan.carries.iter().position(|c| {
                c.views.contains(&o) || c.exit == Some(o) || c.next == o || c.merge == o
            })
        };
        if carry(x).is_some() && carry(x) == carry(y) {
            return true;
        }
        let Some(a) = &plan.arena else {
            return false;
        };
        match (a.offset.get(x), a.offset.get(y)) {
            (Some(&ox), Some(&oy)) => {
                let (bx, by) = (
                    self.array_bytes(&self.obj_ty(x)),
                    self.array_bytes(&self.obj_ty(y)),
                );
                ox < oy + by && oy < ox + bx
            }
            _ => false,
        }
    }

    /// At kernel `e`'s launch (`gx` blocks of `block` threads): its zero
    /// tail's arguments — the grid's threads, then each folded site's output
    /// (allocated here, as the site's own launch would) and bytes.
    pub(super) fn zero_args(
        &mut self,
        e: MorphismId,
        gx: &str,
        block: u64,
        ptrs: &mut Vec<String>,
    ) {
        let Some(jobs) = self
            .resident
            .as_ref()
            .and_then(|p| p.zero_by.get(e))
            .cloned()
        else {
            return;
        };
        let (w, t) = (self.tmp(), self.tmp());
        self.line(format!("{w} = zext i32 {gx} to i64"));
        self.line(format!("{t} = mul i64 {w}, {block}"));
        ptrs.push(t);
        let mut total = 0;
        for &(l, bytes) in &jobs {
            let target = self.ir.morphism(l).expect("site").target;
            let d = self.dalloc(target, bytes);
            ptrs.push(d);
            ptrs.push(bytes.to_string());
            total += bytes;
        }
        self.line(format!("; mapal.zero sites={} bytes={total}", jobs.len()));
    }
}

/// A zero tail's kernel parameters: the grid's threads, then each job's
/// output and bytes.
pub(crate) fn zero_params(jobs: usize) -> Vec<String> {
    std::iter::once("i64 %zt".to_string())
        .chain((0..jobs).flat_map(|j| [format!("ptr addrspace(1) %zp{j}"), format!("i64 %zn{j}")]))
        .collect()
}

/// The zero tail (`zero-fold`), placed at a kernel's exit block before its
/// `ret`: thread `g` of the grid's `%zt` (blocks of `threads`) stores a zero
/// 16-byte vector at `g`, `g + %zt`, … of each job's `%zn ÷ 16` vectors (every
/// device buffer starts 16-byte aligned: `cuMemAlloc`, the arena's
/// `ARENA_ALIGN`), then thread `g < %zn mod 16` the byte past them. Every
/// thread of the kernel reaches it.
pub(crate) fn zero_tail(jobs: usize, threads: u64) -> String {
    if jobs == 0 {
        return String::new();
    }
    let mut s = format!(
        "  %zf.t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %zf.b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %zf.t64 = zext i32 %zf.t to i64\n\
         \x20 %zf.b64 = zext i32 %zf.b to i64\n\
         \x20 %zf.base = mul i64 %zf.b64, {threads}\n\
         \x20 %zf.g = add i64 %zf.base, %zf.t64\n\
         \x20 br label %zf0\n"
    );
    for j in 0..jobs {
        s += &format!(
            "zf{j}:\n\
             \x20 %zf{j}.w = lshr i64 %zn{j}, 4\n\
             \x20 br label %zf{j}.h\n\
             zf{j}.h:\n\
             \x20 %zf{j}.i = phi i64 [ %zf.g, %zf{j} ], [ %zf{j}.nx, %zf{j}.s ]\n\
             \x20 %zf{j}.c = icmp ult i64 %zf{j}.i, %zf{j}.w\n\
             \x20 br i1 %zf{j}.c, label %zf{j}.s, label %zf{j}.r\n\
             zf{j}.s:\n\
             \x20 %zf{j}.p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %zp{j}, i64 %zf{j}.i\n\
             \x20 store <4 x i32> zeroinitializer, ptr addrspace(1) %zf{j}.p, align 16\n\
             \x20 %zf{j}.nx = add i64 %zf{j}.i, %zt\n\
             \x20 br label %zf{j}.h\n\
             zf{j}.r:\n\
             \x20 %zf{j}.rem = and i64 %zn{j}, 15\n\
             \x20 %zf{j}.rc = icmp ult i64 %zf.g, %zf{j}.rem\n\
             \x20 br i1 %zf{j}.rc, label %zf{j}.rb, label %zf{n}\n\
             zf{j}.rb:\n\
             \x20 %zf{j}.ro = shl i64 %zf{j}.w, 4\n\
             \x20 %zf{j}.ri = add i64 %zf{j}.ro, %zf.g\n\
             \x20 %zf{j}.rp = getelementptr inbounds i8, ptr addrspace(1) %zp{j}, i64 %zf{j}.ri\n\
             \x20 store i8 0, ptr addrspace(1) %zf{j}.rp, align 1\n\
             \x20 br label %zf{n}\n",
            n = j + 1
        );
    }
    s + &format!("zf{jobs}:\n")
}
