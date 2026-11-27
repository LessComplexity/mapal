//! `emit_task`, body captures, the filtered walks and checkpoints

use super::*;

impl<'a> FnEmit<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::func) fn emit_task(
        ir: &'a CategoryIr,
        f: FuncId,
        fnames: &'a SecondaryMap<FuncId, String>,
        strings: &'a SecondaryMap<ObjectId, StrGlobal>,
        attrs: &'a FnAttrs,
        frame: &FrameLayout,
        task_id: usize,
        task: &mapal_ir::Task,
        tiling: bool,
        packing: bool,
        contract: bool,
        order: crate::Order,
        kc_nest: bool,
        move_panel: MovePanel,
        profile: &'static TargetProfile,
        outer_pack: bool,
        arms: crate::GpuArms,
        par: &std::rc::Rc<super::super::frame::ParOrder>,
    ) -> String {
        if let TaskKind::Split { site: m, n } = &task.kind
            && let Some(site) = tiling
                .then(|| ir.tile_plan(f))
                .and_then(|plan| plan.sites.get(*m).cloned())
            && packing
            && packing_site(&site, profile, contract)
        {
            let mut slice = FnEmit::new(
                ir, f, fnames, strings, attrs, tiling, packing, contract, order, kc_nest,
                move_panel, profile,
            );
            slice.gpu_arms = arms;
            slice.par = Some(par.clone());
            slice.guard_flavor = GuardFlavor::Task;
            slice.split_range = true;
            slice.prepare_storage();
            slice.frame = Some(frame.clone());
            let mut members = SecondaryMap::new();
            members.insert(*m, ());
            slice.walk_filtered(&members, true);
            slice.line("ret void");
            let slice_fn = format!(
                "define internal void @task{task_id}_slice(i64 %lo, i64 %hi, ptr %frame) {{\nentry:\n{}{}{}}}\n",
                slice.allocas, slice.frame_geps, slice.body
            );

            // plan-s43-parallel-bpack: the pack is its own range-parameterized
            // function over j-tiles, so the wrapper can dispatch it across the
            // pool instead of running it inline on one lane. S43 §4c measured
            // the inline version at 16.349 ms of a 54.164 ms threaded N=4096
            // wall — 30.2%, thread-count-independent, pure Amdahl.
            let source = ir.morphism(*m).expect("tile map resolves").source;
            let mut pack = FnEmit::new(
                ir, f, fnames, strings, attrs, tiling, packing, contract, order, kc_nest,
                move_panel, profile,
            );
            pack.gpu_arms = arms;
            pack.par = Some(par.clone());
            pack.guard_flavor = GuardFlavor::Task;
            pack.split_range = true;
            pack.prepare_storage();
            pack.frame = Some(frame.clone());
            let packed = pack.packed_buffer(*m, &site);
            pack.emit_pack_copy(source, &site, &packed);
            pack.line("ret void");
            let pack_fn = format!(
                "define internal void @task{task_id}_pack(i64 %lo, i64 %hi, ptr %frame) {{\nentry:\n{}{}{}}}\n",
                pack.allocas, pack.frame_geps, pack.body
            );

            // plan-s50 P0: when the driver registered this site as two OUTER
            // tasks, there is no wrapper — the pack and the matmul are dispatched
            // by the outer run and joined by a `mapal_par_dep` edge. The two
            // nested runs below cost 36.8 us each at 14 lanes, per site, per
            // invocation, inside the timed region.
            if outer_pack {
                return format!("{slice_fn}\n{pack_fn}");
            }

            let mut emit = FnEmit::new(
                ir, f, fnames, strings, attrs, tiling, packing, contract, order, kc_nest,
                move_panel, profile,
            );
            emit.gpu_arms = arms;
            emit.par = Some(par.clone());
            emit.guard_flavor = GuardFlavor::Task;
            emit.prepare_storage();
            emit.frame = Some(frame.clone());
            // RUN-ONCE IS PRESERVED AND IS NOT THE SAME PROPERTY AS RUN-SERIAL.
            // The outer registration is still `kind = 0` (drive.rs:266-275), so
            // this wrapper body executes exactly once — S27's run-once wrapper,
            // "help-first finish ⇒ width-1 sound". What changes is only the
            // WIDTH of the pack inside it.
            //
            // Two sequential nested runs, not one handle with a dep edge: a
            // dep-unlocked task is scheduled `Placement::Local(lane)`
            // (mapal-rt `complete_slice`), which would put every matmul slice on
            // one deque and skip the rank-sorted `Placement::Seed` the matmul
            // gets today. `mapal_par_finish` is `help_until(remaining == 0)`, so
            // the first run's return is the happens-before edge from every pack
            // store to every matmul load, and the second dispatch stays exactly
            // what it was. The extra handle costs one Box and a wake, once per
            // program run.
            let tile_j = profile.tile_j(&site.elem);
            let tiles = site.c.div_ceil(tile_j);
            let pack_handle = emit.tmp();
            emit.line(format!("{pack_handle} = call ptr @mapal_par_begin(i32 1)"));
            // The pack's quantum is ONE j-tile, not the matmul's `ti·t·c` — a
            // different axis in different units, so `slice_sizing` is not reused.
            // `slice_elems = 0` would be the silent-nothing bug: `slice_ranges`
            // falls to `min(T, ceil(n/4096))`, which at 256 tiles is ONE slice
            // and the pack stays serial with nothing failing.
            // local-rtx-5080: both nested runs register their width.
            let (pack_kind, pack_width) = profile.dispatch.map_or((1, 0), |cost| {
                let bytes = 2u64
                    .saturating_mul(site.k)
                    .saturating_mul(site.c)
                    .saturating_mul(crate::profile::elem_bytes(&site.elem));
                registration(profile, bytes / cost.bytes_per_ns.max(1))
            });
            emit.line(format!(
                "call void @mapal_par_task(ptr {pack_handle}, i32 0, i32 {pack_kind}, ptr @task{task_id}_pack, i64 {tiles}, i32 {}, i64 1, i32 4, i32 {pack_width})",
                task.rank
            ));
            emit.line(format!(
                "call void @mapal_par_launch(ptr {pack_handle}, ptr %frame)"
            ));
            emit.line(format!("call void @mapal_par_finish(ptr {pack_handle})"));
            let handle = emit.tmp();
            emit.line(format!("{handle} = call ptr @mapal_par_begin(i32 1)"));
            // The packed flavor slices HERE, in the nested dispatch — the outer
            // task is the pack wrapper and is never split — so this is the call
            // the region's sizing belongs on.
            let (min_slice, oversub) = emit.slice_sizing(&site);
            let (slice_kind, slice_width) = profile.dispatch.map_or((1, 0), |cost| {
                let flops = 2u64.saturating_mul(*n).saturating_mul(site.k);
                let bytes = crate::profile::tile_bytes(site.rows, site.c, site.k, &site.elem);
                registration(
                    profile,
                    crate::profile::roofline_ns(flops, bytes, cost.flops_per_ns, cost.bytes_per_ns),
                )
            });
            emit.line(format!(
                "call void @mapal_par_task(ptr {handle}, i32 0, i32 {slice_kind}, ptr @task{task_id}_slice, i64 {n}, i32 {}, i64 {min_slice}, i32 {oversub}, i32 {slice_width})",
                task.rank
            ));
            emit.line(format!(
                "call void @mapal_par_launch(ptr {handle}, ptr %frame)"
            ));
            emit.line(format!("call void @mapal_par_finish(ptr {handle})"));
            emit.line("ret void");
            let wrapper_fn = format!(
                "define internal void @task{task_id}(i64 %lo, i64 %hi, ptr %frame) {{\nentry:\n{}{}{}}}\n",
                emit.allocas, emit.frame_geps, emit.body
            );
            return format!("{slice_fn}\n{pack_fn}\n{wrapper_fn}");
        }

        let mut emit = FnEmit::new(
            ir, f, fnames, strings, attrs, tiling, packing, contract, order, kc_nest, move_panel,
            profile,
        );
        emit.gpu_arms = arms;
        emit.par = Some(par.clone());
        emit.guard_flavor = GuardFlavor::Task;
        emit.split_range = matches!(task.kind, TaskKind::Split { .. });
        // A sequential task runs once per program (its registration is
        // `kind = 0`), so its big blocks may live in the arena the entry
        // releases after `mapal_par_finish` — not on a worker's stack (S65:
        // `pt_loop_4096`'s in-loop 64 MB pack overflowed it, rc 139). No
        // teardown here: the entry's frees everything.
        emit.heap_ok = matches!(task.kind, TaskKind::Seq { .. });
        emit.watermark = match &task.kind {
            TaskKind::Split { .. } => false,
            TaskKind::Seq { morphisms } => !morphisms.iter().any(|&m| {
                matches!(
                    ir.morphism(m).expect("morphism resolves").op,
                    Operation::Fold { .. } | Operation::LoopEnter
                )
            }),
        };
        emit.prepare_storage();
        emit.frame = Some(frame.clone());

        let mut members = SecondaryMap::new();
        match &task.kind {
            TaskKind::Split { site, .. } => {
                members.insert(*site, ());
            }
            TaskKind::Seq { morphisms } => {
                for &m in morphisms {
                    members.insert(m, ());
                }
            }
        }
        emit.walk_filtered(&members, true);
        emit.line("ret void");
        format!(
            "define internal void @task{task_id}(i64 %lo, i64 %hi, ptr %frame) {{\nentry:\n{}{}{}}}\n",
            emit.allocas, emit.frame_geps, emit.body
        )
    }

    /// The capture count of the Map/Fold site whose body is `self.f` (unique —
    /// lower mints one fresh body fn per lambda site, ADR-0027). `0` when no
    /// site references the fn (a dead body), which is the identity lowering.
    pub(in crate::func) fn body_captures(&self) -> u32 {
        for (_, m) in self.ir.morphisms() {
            match m.op {
                Operation::Map { body, captures } | Operation::Fold { body, captures }
                    if body == self.f =>
                {
                    return captures;
                }
                _ => {}
            }
        }
        0
    }

    pub(in crate::func) fn walk(&mut self) {
        // Driver-owned morphisms: everything in a loop plan's decide/advance
        // cones, plus anything incident to an SCC object. Plan membership is the
        // precise rule — an exit-only payload chain (e.g. a computed exit value,
        // or an exit-arm Print) leaves the SCC but still belongs to the decide
        // cone; skipping by SCC incidence alone would re-emit it after the loop
        // (dead recompute for values, a DOUBLE side effect for Print).
        let mut in_scc: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for scc in self.ir.loop_structure(self.f) {
            for &mg in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, mg) {
                    for &mo in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                        owned.insert(mo, ());
                    }
                }
            }
            for o in scc.objects {
                in_scc.insert(o, ());
            }
        }

        for m in self.ir.topo_order(self.f) {
            let morph = self.ir.morphism(m).expect("morphism resolves");
            match morph.op {
                Operation::LoopEnter => {
                    if self.gated.contains_key(m) {
                        // plan-s40: arm-owned loop — its Phi drives it.
                        continue;
                    }
                    crate::loops::emit_loop(self, morph.target)
                }
                Operation::LoopBack | Operation::LoopExit => {}
                _ => {
                    if owned.contains_key(m)
                        || in_scc.contains_key(morph.source)
                        || in_scc.contains_key(morph.target)
                    {
                        continue; // driver-owned
                    }
                    if self.gated.contains_key(m) {
                        continue; // plan-s39: fired only from its Phi's branch
                    }
                    self.emit_morphism(m);
                }
            }
        }
    }

    pub(in crate::func) fn walk_filtered(
        &mut self,
        members: &SecondaryMap<MorphismId, ()>,
        include_members: bool,
    ) {
        let mut in_scc: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for scc in self.ir.loop_structure(self.f) {
            for &merge in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, merge) {
                    for &m in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                        owned.insert(m, ());
                    }
                }
            }
            for o in scc.objects {
                in_scc.insert(o, ());
            }
        }

        for m in self.ir.topo_order(self.f) {
            if let Some(pin) = self
                .host
                .as_ref()
                .and_then(|host| host.pinned.get(m))
                .cloned()
            {
                self.line(format!(
                    "call void @mapal_par_wait(ptr %h, ptr @pin{}_entries, i32 {})",
                    pin.task, pin.len
                ));
                self.line(format!(
                    "call void @mapal_par_check(ptr %h, i64 {})",
                    pin.topo
                ));
                self.line(format!(
                    "call void @mapal_par_run_pinned(ptr %h, i32 {})",
                    pin.task
                ));
            }

            if members.contains_key(m) != include_members {
                continue;
            }
            let morph = self.ir.morphism(m).expect("morphism resolves");
            match morph.op {
                Operation::LoopEnter => {
                    if self.gated.contains_key(m) {
                        // plan-s40: arm-owned loop — its Phi drives it, and
                        // path_plan folded it into the Phi's Seq task, so no
                        // pre-loop checkpoint belongs here either.
                        continue;
                    }
                    if let Some(list) = self
                        .host
                        .as_ref()
                        .and_then(|host| host.pre_loop.get(m))
                        .cloned()
                    {
                        for c in &list {
                            self.line(format!(
                                "call void @mapal_par_wait(ptr %h, ptr @ckpt{}_entries, i32 {})",
                                c.ordinal, c.len
                            ));
                            self.line(format!(
                                "call void @mapal_par_check(ptr %h, i64 {})",
                                c.topo
                            ));
                        }
                    }
                    crate::loops::emit_loop(self, morph.target)
                }
                Operation::LoopBack | Operation::LoopExit => {}
                _ => {
                    if owned.contains_key(m)
                        || in_scc.contains_key(morph.source)
                        || in_scc.contains_key(morph.target)
                    {
                        continue;
                    }
                    if self.gated.contains_key(m) {
                        continue; // plan-s39: fired only from its Phi's branch
                    }
                    self.emit_morphism(m);
                }
            }
        }
    }

    pub(in crate::func) fn emit_checkpoint(&mut self, m: MorphismId) {
        let Some(list) = self
            .host
            .as_ref()
            .and_then(|host| host.checkpoints.get(m))
            .cloned()
        else {
            return;
        };
        for checkpoint in list {
            self.line(format!(
                "call void @mapal_par_wait(ptr %h, ptr @ckpt{}_entries, i32 {})",
                checkpoint.ordinal, checkpoint.len
            ));
            self.line(format!(
                "call void @mapal_par_check(ptr %h, i64 {})",
                checkpoint.topo
            ));
        }
    }
}
