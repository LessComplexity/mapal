//! `emit_parallel`: the parallel-path driver

use super::*;

impl<'a> FnEmit<'a> {
    pub(crate) fn emit_parallel(
        ir: &'a CategoryIr,
        f: FuncId,
        fnames: &'a SecondaryMap<FuncId, String>,
        strings: &'a SecondaryMap<ObjectId, StrGlobal>,
        attrs: &'a FnAttrs,
        plan: &PathPlan,
        perf_timing: bool,
        tiling: bool,
        packing: bool,
        contract: bool,
        order: crate::Order,
        kc_nest: bool,
        move_panel: MovePanel,
        profile: &'static TargetProfile,
        arms: crate::GpuArms,
    ) -> String {
        let mut host = FnEmit::new(
            ir, f, fnames, strings, attrs, tiling, packing, contract, order, kc_nest, move_panel,
            profile,
        );
        // The rung arms (S65 `by-ref` decides the frame's layout): the host and
        // every task read the same ones.
        host.gpu_arms = arms;
        let par = std::rc::Rc::new(super::super::frame::ParOrder::new(plan));
        host.par = Some(par.clone());
        host.perf_timing = perf_timing;
        // The host runs once per program, so its blocks may go to the arena.
        host.heap_ok = true;
        let fd = ir.func(f).expect("func resolves");
        let in_llt = host.prepare_storage();
        let frame = host.build_frame_layout(&in_llt, plan);
        host.frame = Some(frame.clone());
        // The parallel flavor packs every array into ONE `%Frame`, so THAT is
        // the block that blows the stack — heap-lower it as a unit. Tasks take
        // `ptr %frame` either way, and every field access is a `getelementptr
        // %Frame, ptr %frame, …`, so nothing below this line changes.
        let frame_bytes = llt_bytes(&frame.struct_llt());
        if host.heap_block(frame_bytes) {
            let align = llt_align(&frame.struct_llt());
            host.allocas.push_str(&format!(
                "  %frame = call ptr @mapal_rt_alloc(i64 {frame_bytes}, i64 {align})\n"
            ));
        } else {
            host.allocas.push_str("  %frame = alloca %Frame\n");
        }
        host.materialize_frame_slots();
        host.allocate_frame_packs();

        let topo = ir.topo_order(f);
        let mut assigned = SecondaryMap::new();
        let mut pinned = SecondaryMap::new();
        for (task_id, task) in plan.tasks.iter().enumerate() {
            let members: &[MorphismId] = match &task.kind {
                TaskKind::Split { site, .. } => std::slice::from_ref(site),
                TaskKind::Seq { morphisms } => morphisms,
            };
            for &m in members {
                assigned.insert(m, ());
            }
            if task.pinned {
                let first = *members.first().expect("task has a member");
                let first_topo = topo
                    .iter()
                    .position(|&candidate| candidate == first)
                    .expect("task member is in topo") as u32;
                pinned.insert(
                    first,
                    PinnedEmit {
                        task: task_id,
                        topo: first_topo,
                        len: task.deps.len(),
                    },
                );
            }
        }
        let mut checkpoints: SecondaryMap<MorphismId, Vec<CheckpointEmit>> = SecondaryMap::new();
        for (ordinal, checkpoint) in plan
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.topo != u32::MAX)
            .enumerate()
        {
            let emit = CheckpointEmit {
                ordinal,
                topo: checkpoint.topo,
                len: checkpoint.wait.len(),
            };
            let at = topo[checkpoint.at as usize];
            if let Some(list) = checkpoints.get_mut(at) {
                list.push(emit);
            } else {
                checkpoints.insert(at, vec![emit]);
            }
        }
        let mut pre_loop: SecondaryMap<MorphismId, Vec<CheckpointEmit>> = SecondaryMap::new();
        for scc in ir.loop_structure(f) {
            let mut objects: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for &o in &scc.objects {
                objects.insert(o, ());
            }
            let mut members: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            for &m in &topo {
                let morph = ir.morphism(m).expect("morphism resolves");
                if objects.contains_key(morph.source) || objects.contains_key(morph.target) {
                    members.insert(m, ());
                }
            }
            for &merge in &scc.merges {
                if let Some(plan) = ir.loop_plan(f, merge) {
                    for &m in plan
                        .decide_order
                        .iter()
                        .chain(&plan.advance_order)
                        .chain(&plan.exits)
                    {
                        members.insert(m, ());
                    }
                }
            }
            // A checkpoint site inside this loop ⟹ the loop is effectful (a
            // print is token-bearing), hence host-emitted; hoist its wait+check
            // to the loop's first LoopEnter.
            let first_enter = topo.iter().copied().find(|&m| {
                members.contains_key(m)
                    && matches!(
                        ir.morphism(m).expect("morphism resolves").op,
                        Operation::LoopEnter
                    )
            });
            let Some(first_enter) = first_enter else {
                continue;
            };
            for (ordinal, checkpoint) in plan
                .checkpoints
                .iter()
                .filter(|checkpoint| checkpoint.topo != u32::MAX)
                .enumerate()
            {
                if members.contains_key(topo[checkpoint.topo as usize]) {
                    let emit = CheckpointEmit {
                        ordinal,
                        topo: checkpoint.topo,
                        len: checkpoint.wait.len(),
                    };
                    if let Some(list) = pre_loop.get_mut(first_enter) {
                        list.push(emit);
                    } else {
                        pre_loop.insert(first_enter, vec![emit]);
                    }
                }
            }
        }
        host.host = Some(HostEmit {
            checkpoints,
            pinned,
            pre_loop,
        });

        if let Some(t) = &in_llt
            && let Some(slot) = host.slot(fd.input)
        {
            host.line(format!("store {t} %arg, ptr {slot}"));
        }
        // plan-s50 P0: a packed tile site is TWO TASKS IN THIS RUN — the B pack
        // and the tile matmul, joined by a dep edge — not a `kind = 0` wrapper
        // that opens two nested runs of its own.
        //
        // The nested form cost 36.8 us per inner run at 14 lanes
        // (`mapal-rt/examples/dispatch_cost.rs`), paid per site per invocation
        // INSIDE the timed region: `nn2` opened four of them and spent ~95% of
        // its d=128 parallel wall in the runtime. Two outer tasks pay the outer
        // run's dispatch, which is already being paid.
        //
        // S43 §7 rejected exactly this shape, and the reason has now been
        // removed rather than ignored: a dep-unlocked task was scheduled
        // `Placement::Local(lane)`, which would have put every matmul slice on
        // one deque instead of spreading it. `mapal-rt`'s `schedule` now keeps
        // `Local` only for a SINGLE-piece task and spreads a multi-piece one
        // from its producer's lane, so the matmul's placement survives the move.
        //
        // The pack inherits the matmul's dependencies because it reads the same
        // operand; `mapal_par_dep(pack, matmul)` is the happens-before edge that
        // `mapal_par_finish` used to provide by returning.
        let packed_sites: Vec<(usize, u64, u64)> = plan
            .tasks
            .iter()
            .enumerate()
            .filter_map(|(task_id, task)| {
                let TaskKind::Split { site, .. } = &task.kind else {
                    return None;
                };
                let elided = host
                    .ir
                    .morphism(*site)
                    .is_some_and(|m| host.elided_arrays.contains_key(m.target));
                if elided || !host.packing {
                    return None;
                }
                let record = host.tile_plan.as_ref()?.sites.get(*site)?;
                if !packing_site(record, profile, host.contract) {
                    return None;
                }
                // The pack's range is j-tiles, a different axis in different
                // units from the matmul's `ti·t·c` — see `emit_task`.
                let tiles = record.c.div_ceil(profile.tile_j(&record.elem));
                // plan-s50 P3, the pack half: a pure copy, so its cost is bytes
                // over the streaming rate with no compute term at all. At d=128
                // that is a 128 KB copy, ~1 us against a 30 us wave, and handing
                // it to the pool cannot pay. At N=4096 it is 33 MB and S43
                // measured it at 30.2% of the threaded wall, so it must still
                // dispatch.
                let pack_bytes = 2u64
                    .saturating_mul(record.k)
                    .saturating_mul(record.c)
                    .saturating_mul(crate::profile::elem_bytes(&record.elem));
                Some((task_id, tiles, pack_bytes))
            })
            .collect();
        let pack_task_of: std::collections::HashMap<usize, usize> = packed_sites
            .iter()
            .enumerate()
            .map(|(k, (task_id, ..))| (*task_id, plan.tasks.len() + k))
            .collect();

        host.line(format!(
            "%h = call ptr @mapal_par_begin(i32 {})",
            plan.tasks.len() + packed_sites.len()
        ));
        for (task_id, task) in plan.tasks.iter().enumerate() {
            let (kind, n) = match &task.kind {
                // step 3b fallout: `path_plan` cut this task from the GRAPH,
                // where the producer is a real morphism making a real array.
                // The backend then decided nothing reads that array and emitted
                // no body. Registering it `Split` anyway would have the pool
                // slice a million-element range across every core to run a
                // function with zero instructions — free at MAPAL_PAR=1, one
                // dispatch per core plus a join at width. Register it as a
                // single unsplittable unit instead; the dep edges are untouched,
                // so dependents still wait on it and nothing renumbers.
                TaskKind::Split { site, .. }
                    if host
                        .ir
                        .morphism(*site)
                        .is_some_and(|m| host.elided_arrays.contains_key(m.target)) =>
                {
                    (0, 1)
                }
                TaskKind::Split { n, .. } => (1, *n),
                TaskKind::Seq { morphisms } => (0, morphisms.len().max(1) as u64),
            };
            // plan-s50 P3 — DO NOT DISPATCH WHAT CANNOT PAY FOR ITS OWN WAVE.
            //
            // A dispatch buys one lane's share of the task and costs a whole
            // wave (`DispatchCost::dispatch_ns`). Below that the pool is a net
            // loss, and it is measurably so: `nn4` at d=128 runs 0.0790 ms on
            // one thread and 0.1448 ms on fourteen — parallel is 1.8x SLOWER —
            // and `conv2d_512` is 0.0476 against 0.0569. BLAS has had this
            // threshold forever (`GEMM_MULTITHREAD_THRESHOLD`); it is why numpy
            // wins these sizes with no scheduler at all.
            //
            // Registering the task `kind = 0` is the whole mechanism: a kind-0
            // task is one slice, and `mapal-rt` already runs a one-slice task on
            // the calling thread instead of queueing it. So the decision lives
            // here, in the backend, per backend, and `mapal-ir` never learns it.
            //
            // A PURE OPERATION COUNT CANNOT MAKE THIS CALL, which is why the
            // estimate is a roofline. `conv2d_1024` has HALF the flops of an
            // `nn4_256` layer (1.9e7 vs 3.4e7) and takes 3.7x LONGER (250 us vs
            // 68 us) because one is memory-bound and the other compute-bound;
            // they must go opposite ways, so no threshold on flops exists.
            // `max(flops/rate, bytes/rate)` orders all six measured cells.
            let kind = if kind == 1 && host.under_dispatch_cost(task) {
                0
            } else {
                kind
            };
            // plan-s32 step 2: the region's slice sizing, decided here because
            // both inputs are compile-time facts — the recorded reuse structure
            // and the profile's tile factors. The lane count is NOT an input;
            // the runtime multiplies `oversub` by however many lanes it has.
            let (min_slice, oversub) = match &task.kind {
                TaskKind::Split { site, .. } => host
                    .tile_plan
                    .as_ref()
                    .and_then(|plan| plan.sites.get(*site))
                    .map_or((0, 0), |site| host.slice_sizing(site)),
                TaskKind::Seq { .. } => (0, 0),
            };
            // plan-s50 P0: a packed site's own body is now `@task{id}_slice`;
            // `@task{id}` no longer exists, because the wrapper it named is gone.
            let body = if pack_task_of.contains_key(&task_id) {
                format!("@task{task_id}_slice")
            } else {
                format!("@task{task_id}")
            };
            // local-rtx-5080: the lanes this task wakes (`0` = the whole pool).
            let width = if kind == 1 {
                host.task_width(task).filter(|&w| w > 1).unwrap_or(0)
            } else {
                0
            };
            host.line(format!(
                "call void @mapal_par_task(ptr %h, i32 {task_id}, i32 {kind}, ptr {body}, i64 {n}, i32 {}, i64 {min_slice}, i32 {oversub}, i32 {width})",
                task.rank
            ));
        }
        for (task_id, tiles, pack_bytes) in &packed_sites {
            let pack_id = pack_task_of[task_id];
            let pack_ns =
                |cost: crate::profile::DispatchCost| pack_bytes / cost.bytes_per_ns.max(1);
            let (pack_kind, pack_width) = match profile.dispatch {
                Some(cost) if cost.lanes >= 2 => registration(profile, pack_ns(cost)),
                Some(cost) => (u64::from(pack_ns(cost) >= cost.dispatch_ns), 0),
                None => (1, 0),
            };
            // `slice_elems = 0` would be the silent-nothing bug: `slice_ranges`
            // would fall back to `min(T, ceil(n/4096))`, which at 256 tiles is
            // ONE slice — the pack stays serial and nothing fails. The quantum
            // is one j-tile; `oversub = 4` reproduces the nested form exactly.
            host.line(format!(
                "call void @mapal_par_task(ptr %h, i32 {pack_id}, i32 {pack_kind}, ptr @task{task_id}_pack, i64 {tiles}, i32 {}, i64 1, i32 4, i32 {pack_width})",
                plan.tasks[*task_id].rank
            ));
        }
        for (task_id, task) in plan.tasks.iter().enumerate() {
            if task.pinned {
                host.line(format!("call void @mapal_par_pin(ptr %h, i32 {task_id})"));
            }
        }
        for (after, task) in plan.tasks.iter().enumerate() {
            for &before in &task.deps {
                host.line(format!(
                    "call void @mapal_par_dep(ptr %h, i32 {before}, i32 {after})"
                ));
            }
        }
        for (task_id, ..) in &packed_sites {
            let pack_id = pack_task_of[task_id];
            // The pack reads the same operand the matmul does, so it waits on
            // exactly what the matmul waited on — and the matmul now waits on it.
            for &before in &plan.tasks[*task_id].deps {
                host.line(format!(
                    "call void @mapal_par_dep(ptr %h, i32 {before}, i32 {pack_id})"
                ));
            }
            host.line(format!(
                "call void @mapal_par_dep(ptr %h, i32 {pack_id}, i32 {task_id})"
            ));
        }
        host.line("call void @mapal_par_launch(ptr %h, ptr %frame)");
        host.walk_filtered(&assigned, false);
        host.line("call void @mapal_par_finish(ptr %h)");

        let ret_ty = host.obj_ty(fd.output);
        let sig_ret = match lower_ty(&ret_ty) {
            Some(t) => {
                let slot = host.slot(fd.output).expect("non-void return has a slot");
                let value = host.tmp();
                host.line(format!("{value} = load {t}, ptr {slot}"));
                if host.perf_timing {
                    host.line("call void @mapal_perf_end()");
                }
                host.heap_teardown();
                host.line(format!("ret {t} {value}"));
                t
            }
            None => {
                if host.perf_timing {
                    host.line("call void @mapal_perf_end()");
                }
                host.heap_teardown();
                host.line("ret void");
                "void".into()
            }
        };
        let param = in_llt
            .as_ref()
            .map(|t| format!("{t} %arg"))
            .unwrap_or_default();

        let mut out = frame.definition();
        out.push('\n');
        for (ordinal, checkpoint) in plan
            .checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.topo != u32::MAX)
            .enumerate()
        {
            out.push_str(&wait_global(
                &format!("ckpt{ordinal}_entries"),
                &checkpoint.wait,
            ));
        }
        for (task_id, task) in plan.tasks.iter().enumerate() {
            if task.pinned {
                let wait = task
                    .deps
                    .iter()
                    .map(|&task| WaitEntry {
                        task,
                        threshold: None,
                    })
                    .collect::<Vec<_>>();
                out.push_str(&wait_global(&format!("pin{task_id}_entries"), &wait));
            }
        }
        if plan
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.topo != u32::MAX)
            || plan.tasks.iter().any(|task| task.pinned)
        {
            out.push('\n');
        }

        for (task_id, task) in plan.tasks.iter().enumerate() {
            out.push_str(&Self::emit_task(
                ir,
                f,
                fnames,
                strings,
                attrs,
                &frame,
                task_id,
                task,
                tiling,
                packing,
                contract,
                order,
                kc_nest,
                move_panel,
                profile,
                pack_task_of.contains_key(&task_id),
                arms,
                &par,
            ));
            out.push('\n');
        }
        let perf_begin = if host.perf_timing {
            "  call void @mapal_perf_begin()\n"
        } else {
            ""
        };
        out.push_str(&format!(
            "define internal {sig_ret} @{}({param}) {{\nentry:\n{}{}{}{}{}\n",
            fnames[f], perf_begin, host.allocas, host.frame_geps, host.body, "}"
        ));
        out
    }
}
