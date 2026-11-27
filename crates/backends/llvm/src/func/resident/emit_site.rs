//! resident emission after a launch: delta uploads, row residency and site launches

use super::*;

impl<'a> FnEmit<'a> {
    /// After `m`: copy its output in if a device site reads it, and free every
    /// buffer whose death is `m`.
    pub(in crate::func) fn resident_after(&mut self, m: MorphismId) {
        let Some(plan) = &self.resident else {
            return;
        };
        let t = self.ir.morphism(m).expect("morphism resolves").target;
        let upload = plan.upload.contains_key(t) && !plan.devgen.contains_key(m);
        let delta = plan.delta.get(t).cloned();
        let frees = plan.free_after.get(m).cloned().unwrap_or_default();
        let chain = plan.upload_chain.get(t).copied();
        if upload {
            let ty = self.obj_ty(t);
            let bytes = self.array_bytes(&ty);
            let h = self.array_operand_ptr(t, None).expect("uploaded array ptr");
            let d = self.dalloc(t, bytes);
            // A `+0.0` fill is zeroed on the device: no host bytes cross (the
            // carry's init rule, at any producer).
            let zero = delta.is_none() && chain.is_none() && self.zero_fill(t);
            // `pinned` (S54): a copy that runs more than once is page-locked.
            if !zero {
                self.pin(m, &h, bytes, true);
            }
            match (delta, chain) {
                (None, None) if zero => {
                    self.line(format!("call void @mapal_gpu_zero(i64 {d}, i64 {bytes})"))
                }
                (Some(du), _) => self.emit_delta_h2d(t, &du, &d, &h, bytes),
                (None, Some(k)) => self.line(format!(
                    "call void @mapal_gpu_h2d_on(i32 {k}, i64 {d}, ptr {h}, i64 {bytes})"
                )),
                (None, None) => self.line(format!(
                    "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
                )),
            }
            self.dvalid.insert(t, ());
        }
        for o in frees {
            if self.arena_offset(o).is_some() {
                continue; // its bytes are the arena's, released at exit
            }
            let slot = self.dslot(o);
            self.line(format!("call void @mapal_gpu_free(ptr {slot})"));
            self.dvalid.remove(o);
        }
        // gaps-disasm C1 `zero-hoist`: later atomic sites' zero ops, hoisted
        // here (`zero_fold`).
        let hoisted = self
            .resident
            .as_ref()
            .and_then(|p| p.zero_after.get(m))
            .cloned()
            .unwrap_or_default();
        for (l, bytes) in hoisted {
            let target = self.ir.morphism(l).expect("site").target;
            let d = self.dalloc(target, bytes);
            let j = self
                .resident
                .as_ref()
                .and_then(|p| p.zero_hoist_ix.get(l).copied())
                .expect("a hoisted site has its slot");
            // The site's table (null without one): the runtime skips the
            // zero where the table's pick does not add into the output.
            let t = self.tmp();
            self.line(format!(
                "{t} = load ptr, ptr @{}.zh{j}",
                self.fnames[self.f]
            ));
            self.line(format!(
                "call void @mapal_gpu_zero_for(ptr {t}, i64 {d}, i64 {bytes})"
            ));
        }
    }

    /// `delta-h2d`: after a loop entry's first (whole) upload, only the rows
    /// the iteration wrote cross — `stride` cells from `start`, at the same
    /// byte offset on both sides. Every executed `Update` was in bounds, so
    /// the range is too.
    pub(super) fn emit_delta_h2d(
        &mut self,
        t: ObjectId,
        du: &DeltaUp,
        d: &str,
        h: &str,
        bytes: u64,
    ) {
        let eb = crate::profile::elem_bytes(array_elem(&self.obj_ty(t)));
        let flag = self.retain_flag(t);
        let whole = format!("call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})");
        // `retain-same`: the loop's first trip uploads it, no later one.
        if du.same {
            let valid = self.tmp();
            self.line(format!("{valid} = load i1, ptr {flag}"));
            let (first, done) = (self.label(), self.label());
            self.line(format!("br i1 {valid}, label %{done}, label %{first}"));
            self.label_line(&first);
            self.line(whole);
            self.line(format!("store i1 true, ptr {flag}"));
            self.line(format!("br label %{done}"));
            self.label_line(&done);
            return;
        }
        let Some((wllt, tv)) = self.load_whole(du.start.var) else {
            self.line(whole);
            return;
        };
        let valid = self.tmp();
        self.line(format!("{valid} = load i1, ptr {flag}"));
        let start = self.affine_cell(&wllt, &tv, &du.start);
        // `Δ` is a claim about cells: rows past the array (a trip that writes
        // nothing) cross whole. Unsigned: a start below zero fails too. (A
        // delta upload is never wider than its array: `delta_uploads`.)
        let fits = self.tmp();
        let last = array_len(&self.obj_ty(t)) - du.stride;
        self.line(format!("{fits} = icmp ule i64 {start}, {last}"));
        let go = self.tmp();
        self.line(format!("{go} = and i1 {valid}, {fits}"));
        let (rows, first, done) = (self.label(), self.label(), self.label());
        self.line(format!("br i1 {go}, label %{rows}, label %{first}"));
        self.label_line(&first);
        self.line(whole);
        self.line(format!("store i1 true, ptr {flag}"));
        self.line(format!("br label %{done}"));
        self.label_line(&rows);
        let off = self.tmp();
        self.line(format!("{off} = mul i64 {start}, {eb}"));
        let dd = self.tmp();
        self.line(format!("{dd} = add i64 {d}, {off}"));
        let hp = self.tmp();
        self.line(format!("{hp} = getelementptr i8, ptr {h}, i64 {off}"));
        self.line(format!(
            "call void @mapal_gpu_h2d(i64 {dd}, ptr {hp}, i64 {})",
            du.stride * eb
        ));
        self.line(format!("br label %{done}"));
        self.label_line(&done);
    }

    /// A retained site's restricted pass under `resident` (S54
    /// `retain-resident`): the one-row site `one` reads row `row`'s `a` span
    /// in place in `a`'s device copy and stores into row `row` of the output's
    /// resident buffer. A host reader of the output gets that row only: its
    /// host copy is retained the same way. An operand with no device copy
    /// here runs the whole site instead (always correct).
    pub(in crate::func) fn emit_row_resident(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        row: &str,
        call: impl Fn(&str, &str, &str) -> String,
    ) -> bool {
        let alias = |p: &Option<ResidentPlan>, o: ObjectId| p.as_ref().map_or(o, |p| p.alias(o));
        let feeders: Vec<ObjectId> = [site.a.slot, site.b.slot]
            .iter()
            .filter_map(|&s| self.pair_source(source, s))
            .filter(|&f| self.dvalid.contains_key(alias(&self.resident, f)))
            .collect();
        let &[fa, fb] = feeders.as_slice() else {
            return false;
        };
        let eb = crate::profile::elem_bytes(&site.elem);
        let (a, b) = (self.dload(fa), self.dload(fb));
        let a_cells = self.tmp();
        self.line(format!("{a_cells} = mul i64 {row}, {}", site.a.ci));
        let a_cell = self.tmp();
        self.line(format!("{a_cell} = add i64 {a_cells}, {}", site.a.base));
        let a_off = self.tmp();
        self.line(format!("{a_off} = mul i64 {a_cell}, {eb}"));
        let a_row = self.tmp();
        self.line(format!("{a_row} = add i64 {a}, {a_off}"));
        // `retain-pieces`: the gathered rows land in their compact buffer
        // (the caller places them).
        if let Some(o_row) = self.row_out.clone() {
            match self.site_table.take() {
                Some(t) => {
                    self.site_call(t, &[a_row, b, o_row]);
                }
                None => self.line(call(&a_row, &b, &o_row)),
            }
            self.row_done = true;
            return true;
        }
        let o = self.dload(target);
        let o_off = self.tmp();
        self.line(format!("{o_off} = mul i64 {row}, {}", site.c * eb));
        let o_row = self.tmp();
        self.line(format!("{o_row} = add i64 {o}, {o_off}"));
        // per-site-choice: a row with candidates launches its table.
        match self.site_table.take() {
            Some(t) => {
                self.site_call(t, &[a_row, b, o_row]);
            }
            None => self.line(call(&a_row, &b, &o_row)),
        }
        let keep = self
            .resident
            .as_ref()
            .is_some_and(|p| p.keep.contains_key(target));
        if !keep {
            let h = self.slot(target).expect("tile output slot");
            let h_row = self.tmp();
            self.line(format!("{h_row} = getelementptr i8, ptr {h}, i64 {o_off}"));
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h_row}, i64 {o}, i64 {o_off}, i64 {})",
                site.c * eb
            ));
        }
        true
    }

    /// Operand `slot` of `source` on the device: its producer's copy, or
    /// (a parameter, an exit object) copied in here, every time the site
    /// runs, into a temporary `temps` frees after the launch.
    pub(super) fn dev_operand(
        &mut self,
        source: ObjectId,
        slot: u32,
        ms: MorphismId,
        temps: &mut Vec<String>,
    ) -> String {
        let src_ty = self.obj_ty(source);
        let bytes = self.array_bytes(src_ty.component_ty(slot).expect("tile operand"));
        let feeder = self.pair_source(source, slot);
        if let Some(f) = feeder
            && self
                .dvalid
                .contains_key(self.resident.as_ref().map_or(f, |p| p.alias(f)))
        {
            return self.dload(f);
        }
        let h = self
            .array_operand_ptr(source, Some(slot))
            .expect("tile operand ptr");
        let slot_name = format!("%d{}", self.fresh());
        self.allocas.push_str(&format!(
            "  {slot_name} = alloca i64\n  store i64 0, ptr {slot_name}\n"
        ));
        let d = self.tmp();
        self.line(format!(
            "{d} = call i64 @mapal_gpu_alloc(ptr {slot_name}, i64 {bytes})"
        ));
        self.pin(ms, &h, bytes, true);
        self.line(format!(
            "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
        ));
        temps.push(slot_name);
        d
    }

    /// The device half of a site under `resident`: operands from their device
    /// copies (copied in here only when no producer did), output kept on the
    /// device, copied out whole only when the host reads it whole.
    pub(in crate::func) fn emit_site_resident(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        slots: &[u32],
        // `(kernel id, geometry)` of a tile site: the `launchv` it takes when
        // its store evaluates an epilogue group (S54 R1) or it runs on a
        // chain's stream (S54 R7) instead of `call`'s launch.
        grouped: Option<(i64, String)>,
        call: impl Fn(&[String], &str) -> String,
    ) {
        let ms = self.ir.in_edges(target)[0];
        let mut ops = Vec::new();
        let mut temps = Vec::new();
        for &slot in slots {
            let d = self.dev_operand(source, slot, ms, &mut temps);
            ops.push(d);
        }
        // long-prefill `uniform-base`: a tile site's operand pointer moved by
        // its captured numbers' terms (a head group, a weight panel); the
        // kernel addresses from there as from the array's start.
        let site = self
            .tile_plan
            .as_ref()
            .and_then(|p| p.sites.get(ms))
            .cloned();
        if let Some(site) = site.filter(|s| slots == [s.a.slot, s.b.slot])
            && let Some(r) = site.radix.as_deref()
        {
            // A term moves the operand by whole elements of its own storage:
            // a narrow operand's (ADR-0043, a `bf16` KV cache) are 2 bytes,
            // not the product's `elem`.
            for (i, read, ds) in [(0, &site.a, &r.ua), (1, &site.b, &r.ub)] {
                let e = read.elem_bytes(&site.elem);
                if let Some(off) = self.uniform_bytes(source, ds, e) {
                    let p = self.tmp();
                    self.line(format!("{p} = add i64 {}, {off}", ops[i]));
                    ops[i] = p;
                }
            }
        }
        let plan = self.resident.clone().expect("resident site");
        let s = plan.stream_of(self.ir, &self.groups, target);
        if s != 0 && self.first_chained(&plan, target) {
            self.chain_prologue(&plan);
        }
        let o_bytes = self.array_bytes(&self.obj_ty(target));
        let m = self.ir.in_edges(target)[0];
        let group = grouped
            .clone()
            .and_then(|ig| Some((ig, self.groups.get(m).cloned()?)));
        // The values this launch leaves in `Gmem`: the site's output unless
        // its group holds every reader (S54), and each stored member result.
        let mut stored = Vec::new();
        let dout = if group.as_ref().is_none_or(|(_, g)| g.store_out) {
            let d = self.dalloc(target, o_bytes);
            stored.push((target, d.clone()));
            d
        } else {
            "0".to_string()
        };
        // A cross-chain edge (S54 R7): an operand on another chain's stream — a
        // site operand, or an epilogue's complete operand — is waited for.
        if !plan.chain.is_empty() {
            let mut feeds: Vec<ObjectId> = slots
                .iter()
                .filter_map(|&k| self.pair_source(source, k))
                .collect();
            if let Some((_, g)) = &group {
                feeds.extend(g.caps());
            }
            let mut waits: Vec<u32> = feeds
                .iter()
                .map(|&f| plan.stream_of(self.ir, &self.groups, f))
                .filter(|&p| p != 0 && p != s)
                .collect();
            waits.sort_unstable();
            waits.dedup();
            for p in waits {
                self.line(format!("call void @mapal_gpu_event(i32 {p}, i32 {s})"));
            }
        }
        // The pointers of a `launchv`: an epilogue group's, or a chained site's
        // own three.
        let ptrs = match (&group, &grouped) {
            (Some((_, g)), _) if !g.terms.is_empty() => {
                let sum = g.chain[0].out;
                let bytes = self.array_bytes(&self.obj_ty(sum));
                let d = self.dalloc(sum, bytes);
                self.line(format!("call void @mapal_gpu_zero(i64 {d}, i64 {bytes})"));
                stored.push((sum, d.clone()));
                let mut ptrs = vec![ops[0].clone(), ops[1].clone(), d];
                for &t in &g.terms {
                    let src = self.ir.morphism(t).expect("term").source;
                    let x = match self.tile_plan.as_ref().and_then(|p| p.sites.get(t)) {
                        Some(x) => (x.a.slot, x.b.slot),
                        None => unreachable!("a term is a tile site"),
                    };
                    for slot in [x.0, x.1] {
                        ptrs.push(self.dev_operand(src, slot, ms, &mut temps));
                    }
                }
                Some(ptrs)
            }
            (Some((_, g)), _) => {
                let mut ptrs = vec![ops[0].clone(), ops[1].clone(), dout.clone()];
                for o in g.caps() {
                    // `tile-post`: the site's own operand, as `a` and `b` are.
                    let own = g.post.as_ref().and_then(|p| {
                        p.srcs
                            .iter()
                            .position(|x| *x == Some(EpiSrc::Gmem(o)))
                            .map(|k| k as u32)
                    });
                    if let Some(slot) = own {
                        ptrs.push(self.dev_operand(source, slot, ms, &mut temps));
                        continue;
                    }
                    debug_assert!(
                        self.dvalid.contains_key(plan.alias(o)),
                        "epilogue operand in Gmem"
                    );
                    ptrs.push(self.dload(o));
                }
                for mb in g.chain.iter().filter(|mb| mb.store) {
                    let bytes = self.array_bytes(&self.obj_ty(mb.out));
                    let d = self.dalloc(mb.out, bytes);
                    ptrs.push(d.clone());
                    stored.push((mb.out, d));
                }
                // `tile-post`: each number the epilogue reads, by value.
                for slot in g.uniforms() {
                    let (t, v) = self.load_component(source, slot).expect("uniform loads");
                    let w = self.tmp();
                    self.line(gpu::uniform_to_i64(&w, &v, &t));
                    ptrs.push(w);
                }
                Some(ptrs)
            }
            (None, Some(_)) if s != 0 => Some(vec![ops[0].clone(), ops[1].clone(), dout.clone()]),
            _ => None,
        };
        match (ptrs, grouped) {
            // per-site-choice: a site with candidates launches its table over
            // the same pointers.
            (ptrs, _) if self.site_table.is_some() => {
                let mut t = self.site_table.take().expect("a table");
                // kernel-merging `zero-fold`: an earlier kernel zeroed it.
                if plan.zeroed.contains_key(m) {
                    t.flags |= super::super::zero::SITE_ZEROED;
                }
                // tune-context: an operand that lives in DRAM in the program
                // (`residence_plan` against the working set the part's L2
                // keeps, `Gpu::l2_keep_bytes`) — its trial runs cold
                // (`MAPAL_SITE_COLD`, `mapal_gpu.c` `site_timed`).
                if let Some(gpu) = self.profile.gpu() {
                    let res = self.ir.residence_plan(self.f);
                    let keep = gpu.l2_keep_bytes();
                    let cold = slots.iter().any(|&s| {
                        self.pair_source(source, s)
                            .is_some_and(|o| res.at(o, keep) == mapal_ir::Residence::Dram)
                    });
                    if cold {
                        t.flags |= SITE_COLD;
                    }
                }
                let ptrs =
                    ptrs.unwrap_or_else(|| vec![ops[0].clone(), ops[1].clone(), dout.clone()]);
                // gaps-disasm C1 `zero-hoist`: the hoisted zero's slot holds
                // this table (`mapal_gpu_zero_for`).
                let slot = plan.zero_hoist_ix.get(m).copied();
                let global = self.site_call(t, &ptrs);
                if let Some(j) = slot {
                    self.zero_slot(j, &global);
                }
            }
            (Some(ptrs), Some((id, geom))) => {
                let n = ptrs.len();
                let arr = format!("%p{}", self.fresh());
                self.allocas
                    .push_str(&format!("  {arr} = alloca [{n} x i64]\n"));
                for (i, p) in ptrs.iter().enumerate() {
                    let g = self.tmp();
                    self.line(format!(
                        "{g} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 {i}"
                    ));
                    self.line(format!("store i64 {p}, ptr {g}"));
                }
                self.line(format!(
                    "call void @mapal_gpu_launchv(i64 {id}, i32 {s}, i32 {n}, ptr {arr}, {geom})"
                ));
            }
            _ => self.line(call(&ops, &dout)),
        }
        for (o, _) in &stored {
            self.dvalid.insert(*o, ());
        }
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            let keep = plan.keep.contains_key(o) || plan.is_carry_next(o);
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("device output slot");
                self.pin(ms, &h, bytes, false);
                // A device fold's output lands in what was a host loop's slot: say
                // the copy keeps no pointer to it, as the loop did not, or the slot
                // counts as escaped and a later host loop that reads it loses its
                // vector form (S54: softmax_rows' `exp` pass ran scalar, +17 ms).
                // ponytail: every `d2h` is just as non-capturing; marked here only,
                // so no other program's text moves.
                let fold = self
                    .ir
                    .in_edges(o)
                    .first()
                    .is_some_and(|m| plan.rowfold.contains_key(*m));
                let h = if fold { format!("nocapture {h}") } else { h };
                self.d2h(s, &h, &d, "0", bytes);
            }
        }
    }

    /// A copy-out on stream `s`: the legacy entry on stream 0 (the one-stream
    /// text), else queued behind the chain's kernels and waited for.
    pub(in crate::func) fn d2h(
        &mut self,
        s: u32,
        h: &str,
        d: &str,
        off: &str,
        bytes: impl std::fmt::Display,
    ) {
        if s == 0 {
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 {off}, i64 {bytes})"
            ));
        } else {
            self.line(format!(
                "call void @mapal_gpu_d2h_on(i32 {s}, ptr {h}, i64 {d}, i64 {off}, i64 {bytes})"
            ));
        }
    }

    /// `pinned`: page-lock the host array a copy at `m` moves, when that copy
    /// runs more than once — `m` inside a loop (or guard arm), so one
    /// registration is paid for every crossing after it (the runtime pins an
    /// array once). A copy that runs once is not pinned: registering costs
    /// more than one pinned copy saves on every box measured
    /// (`streams-<part>.txt`, S54 trace + `copy_rate`: 0.06–0.09 ms per MB
    /// registered at 64 MB, 0.3–0.5 at 1 MB, against 0.002 (RTX 4090 box) to
    /// 0.08 ms per MB (RTX 5090 box) saved per crossing). An upload (`up`)
    /// under `async-h2d` is not pinned: the runtime would have to wait for a
    /// page-locked source it does not stage, and staging makes the pin moot.
    pub(super) fn pin(&mut self, m: MorphismId, h: &str, bytes: u64, up: bool) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if rungs.pinned && !(up && rungs.async_h2d) && !self.straight_line().contains_key(m) {
            self.line(format!("call void @mapal_gpu_pin(ptr {h}, i64 {bytes})"));
        }
    }

    /// Is `target`'s site the first chained launch of the function?
    pub(super) fn first_chained(&self, plan: &ResidentPlan, target: ObjectId) -> bool {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let first = plan.chain.keys().min_by_key(|&m| pos(m));
        first.is_some_and(|m| self.ir.morphism(m).expect("site").target == target)
    }

    /// Before the first chained launch: every stored chained output's buffer, so no
    /// `cuMemAlloc` (which may wait for the device) lands between two chains'
    /// launches.
    pub(super) fn chain_prologue(&mut self, plan: &ResidentPlan) {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let mut sites: Vec<MorphismId> = plan.chain.keys().collect();
        sites.sort_by_key(|&m| pos(m));
        for m in sites {
            if self.groups.get(m).is_some_and(|g| !g.store_out) {
                continue;
            }
            let c = self.ir.morphism(m).expect("site").target;
            let bytes = self.array_bytes(&self.obj_ty(c));
            self.dalloc(c, bytes);
        }
    }

    /// A proven `Index` on a device-only array: copy the one element out
    /// right before the host loads it.
    pub(in crate::func) fn resident_elem(
        &mut self,
        m: MorphismId,
        source: ObjectId,
        ep: &str,
        idx: &str,
    ) {
        if !self
            .resident
            .as_ref()
            .is_some_and(|p| p.elem.contains_key(m))
        {
            return;
        }
        let arr = self.pair_source(source, 0).expect("elem read array");
        let eb = crate::profile::elem_bytes(array_elem(&self.obj_ty(arr)));
        let s = self
            .resident
            .as_ref()
            .expect("plan")
            .stream_of(self.ir, &self.groups, arr);
        let d = self.dload(arr);
        let off = self.tmp();
        self.line(format!("{off} = mul i64 {idx}, {eb}"));
        self.d2h(s, ep, &d, &off, eb);
    }

    pub(super) fn carries_of(&self, merge: ObjectId) -> Vec<Carry> {
        self.resident.as_ref().map_or_else(Vec::new, |p| {
            p.carries
                .iter()
                .filter(|c| c.merge == merge)
                .cloned()
                .collect()
        })
    }

    /// S65 movement intent, the loop header's first line: what the facts
    /// require to cross per trip — this plan's own derivation (uploads of
    /// host-produced arrays a device site reads, `delta-h2d` rows, copy-outs
    /// of device outputs a host morphism reads, one-element reads) — and how
    /// each state array travels (`frame.rs` `carried_desc`).
    /// `benches/gpu/evidence.py` counts what the host IR realizes against it.
    /// A loop with no array state off a GPU build moves nothing: no line.
    pub(crate) fn move_line(&self, plan: &mapal_ir::LoopPlan, header: &str) -> String {
        if array_paths(&self.obj_ty(plan.merge)).is_empty() && self.resident.is_none() {
            return String::new();
        }
        let (copy, h2d, d2h, retained) = self.trip_moves(plan);
        // caching-memory `retain-same`: the cells launched and the uploads
        // copied on the loop's first trip only.
        let same = self.resident.as_ref().map_or(0, |p| {
            plan.decide_order
                .iter()
                .chain(&plan.advance_order)
                .filter(|&&m| {
                    let t = self.ir.morphism(m).expect("morphism resolves").target;
                    p.cells
                        .get(m)
                        .is_some_and(|c| c.retain.as_ref().is_some_and(|r| r.same))
                        || p.delta.get(t).is_some_and(|d| d.same)
                })
                .count()
        });
        format!(
            "; mapal.move loop={header} carried={} h2d={h2d} d2h={d2h} host_copy={copy} alloc=0 retained={retained}{}\n",
            self.carried_desc(plan.merge),
            match same {
                0 => String::new(),
                n => format!(" same={n}"),
            }
        )
    }
}
