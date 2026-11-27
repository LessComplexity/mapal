//! `path_plan`: independent parallel paths, tasks and host checkpoints.

use super::*;

impl CategoryIr {
    /// The per-function parallel path query. Tasks are emitted in first topo
    /// occurrence order, so their vector indices are deterministic task ids.
    /// Token-bearing morphisms and every morphism in an effectful loop region
    /// remain on the host spine; Print/TimeMs morphisms still contribute
    /// checkpoints.
    /// Trap capability follows Call/Map/Fold function closures and is attributed
    /// to each reference site's topo position. A task containing a pure Call to
    /// a trap-capable named function is pinned; Map/Fold sites never pin.
    /// Checkpoint trap guards use the last earlier site as their watermark
    /// threshold; consumed-value producers require completion.
    pub fn path_plan(&self, f: FuncId) -> PathPlan {
        self.fact(f, |s| &s.path_plan, || self.path_plan_compute(f))
    }

    pub(super) fn path_plan_compute(&self, f: FuncId) -> PathPlan {
        let topo = self.topo_order(f);
        let bounds = self.bounds_proof(f);
        let trap_capable = self.fn_trap_capabilities();
        let is_token = |m: MorphismId| {
            let morph = &self.morphisms[m];
            ty_contains_token(&self.objects[morph.source].ty)
                || ty_contains_token(&self.objects[morph.target].ty)
        };

        // Recover each loop's complete sequential region. SCC incidence covers
        // the cycle and loop control; loop_plan adds computed route cones that
        // execute inside a canonical loop despite sitting outside the SCC.
        let loops = self.loop_structure(f);
        let mut loop_members = Vec::with_capacity(loops.len());
        for scc in &loops {
            let mut objects: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for &o in &scc.objects {
                objects.insert(o, ());
            }
            let mut members: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            for &m in &topo {
                let morph = &self.morphisms[m];
                if objects.contains_key(morph.source) || objects.contains_key(morph.target) {
                    members.insert(m, ());
                }
            }
            for &merge in &scc.merges {
                if let Some(plan) = self.loop_plan(f, merge) {
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
            loop_members.push(
                topo.iter()
                    .copied()
                    .filter(|m| members.contains_key(*m))
                    .collect::<Vec<_>>(),
            );
        }

        let mut host_loop_member: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for members in &loop_members {
            if members.iter().copied().any(&is_token) {
                for &m in members {
                    host_loop_member.insert(m, ());
                }
            }
        }
        let mut loop_of: SecondaryMap<MorphismId, usize> = SecondaryMap::new();
        for (loop_id, members) in loop_members.iter().enumerate() {
            if !members.iter().copied().any(&is_token) {
                for &m in members {
                    if !host_loop_member.contains_key(m) && !loop_of.contains_key(m) {
                        loop_of.insert(m, loop_id);
                    }
                }
            }
        }

        // plan-s39: guard-arm-owned morphisms fold into their Phi's Seq task —
        // a Split site cannot be gated (it is dispatched at launch, before the
        // condition's value exists), so a gated bulk op runs sequentially
        // inside its arm's branch instead.
        // ponytail: sequential guarded maps; per-task enable predicates in
        // mapal-rt if a real program ever guards a big map.
        let gated: SecondaryMap<MorphismId, ()> = {
            let mut g = SecondaryMap::new();
            for site in self.guard_plan(f).into_iter().filter(GuardSite::gated) {
                for &m in site.on_true.own.iter().chain(site.on_false.own.iter()) {
                    g.insert(m, ());
                }
            }
            g
        };
        let is_bulk = |m: MorphismId| {
            matches!(
                self.morphisms[m].op,
                Operation::Map { .. }
                    | Operation::Fold { .. }
                    | Operation::Zip
                    | Operation::Enumerate
                    | Operation::Iota
                    | Operation::Fill
            )
        };

        // plan-time-builtin, generalized (structure: host-value cone): a
        // value the host spine writes — a clock read, a token operation's
        // product (an effectful call's result, the payload a printing loop
        // exits with), a host loop's state — exists only once the host
        // reaches its producer, while tasks are dispatched at launch. A task
        // reading one races the write (§4.5 Law 1: no transformation reads
        // data not present at its location; the symptoms: a NEGATIVE elapsed,
        // a stale index trapping out of bounds, a null view). So the consumer
        // cone follows the host: a scalar consumer joins the spine (scalar
        // arithmetic in practice, so this costs nothing), a loop reading one
        // runs as a host loop, and an ungated bulk op reading one stays a task
        // pinned to the spine at its own topo position, after the producer —
        // sequential, which is correct before it is fast.
        let is_clock = |m: MorphismId| matches!(self.morphisms[m].op, Operation::TimeMs);
        // ADR-0045 D8: a file read is a DAG node too, for the opposite reason
        // to the clock: its value is an array the program's bulk work reads,
        // and a task reading a spine-written value races the write (the
        // clock's cone could stay on the spine; a model's weights cannot).
        // As a task — the read and the `Proj` of its array out of the
        // `(token, array)` pair — its readers depend on it by ordinary
        // dataflow edges, and it starts as soon as the run launches (a file
        // loads beside the work that does not read it). A failed read is a
        // trap site: it records its failure at its topo position, like a
        // task's trap, and the first checkpoint past it delivers it — after
        // every output and trap the oracle gives before it.
        let is_read = |m: MorphismId| {
            matches!(self.morphisms[m].op, Operation::Read)
                || self.read_of(self.morphisms[m].target).is_some()
        };
        // The function's input and what a `Proj` takes out of it exist at entry
        // (a token product's array field included): not host values.
        let mut at_entry: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        if let Some(fd) = self.func(f) {
            at_entry.insert(fd.input, ());
        }
        for &m in &topo {
            let morph = &self.morphisms[m];
            if matches!(morph.op, Operation::Proj { .. }) && at_entry.contains_key(morph.source) {
                at_entry.insert(morph.target, ());
            }
        }
        let mut host_value: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut host_cone: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut host_read_bulk: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        loop {
            let mut grew = false;
            for &m in &topo {
                let morph = &self.morphisms[m];
                // A file read (and the `Proj` of its array) is a task its
                // readers depend on by dataflow edges (ADR-0045 D8): the token
                // it takes orders nothing it computes, and its array is no
                // host value.
                if is_read(m) {
                    continue;
                }
                let on_host = (is_token(m) && !is_clock(m) && !is_read(m))
                    || is_clock(m)
                    || host_loop_member.contains_key(m)
                    || host_cone.contains_key(m);
                if on_host {
                    if !at_entry.contains_key(morph.target) {
                        grew |= host_value.insert(morph.target, ()).is_none();
                    }
                    continue;
                }
                if !host_value.contains_key(morph.source) || host_read_bulk.contains_key(m) {
                    continue;
                }
                grew = true;
                if let Some(&id) = loop_of.get(m) {
                    for &x in &loop_members[id] {
                        host_loop_member.insert(x, ());
                    }
                } else if is_bulk(m) && !gated.contains_key(m) {
                    host_read_bulk.insert(m, ());
                } else {
                    host_cone.insert(m, ());
                }
            }
            if !grew {
                break;
            }
        }

        // plan-s33b: a clock read is a DAG node, not a spine instruction. It is
        // token-bearing, so it would be host by `is_token` — but then the only
        // ordering it has is `before` (the source-order fence below), and the
        // tasks written AFTER it are free to start the moment their data is
        // ready, which is before the host reaches the read. Making it a task
        // gives it `after` too, as ordinary dependency edges (§3: one mechanism,
        // not two). A clock inside an effectful loop region stays host —
        // `host_loop_member` still claims it — and keeps its checkpoint.
        let is_host = |m: MorphismId| {
            (is_token(m) && !is_clock(m) && !is_read(m))
                || host_loop_member.contains_key(m)
                || host_cone.contains_key(m)
        };
        // plan-s40: an arm-owned loop (its LoopEnter handle is gated) must not
        // mint its own launch-dispatched Seq task — like a gated bulk op, its
        // whole region folds into the Phi's sequential component, and the Phi
        // drives it only when the condition picks that arm.
        let gated_loop: Vec<bool> = loop_members
            .iter()
            .map(|members| {
                members.iter().any(|&m| {
                    matches!(self.morphisms[m].op, Operation::LoopEnter) && gated.contains_key(m)
                })
            })
            .collect();
        let in_gated_loop = |m: MorphismId| loop_of.get(m).is_some_and(|&id| gated_loop[id]);
        let is_scalar = |m: MorphismId| {
            !is_host(m)
                && !is_clock(m)
                && !is_read(m)
                // A gated loop's whole region is scalar — bulk ops inside it
                // run sequentially under the driver, under the gate.
                && (in_gated_loop(m)
                    || (!loop_of.contains_key(m)
                        && (gated.contains_key(m)
                            || !matches!(
                                self.morphisms[m].op,
                                Operation::Map { .. }
                                    | Operation::Fold { .. }
                                    | Operation::Zip
                                    | Operation::Enumerate
                                    | Operation::Iota
                                    | Operation::Fill
                            ))))
        };

        let mut tasks = Vec::new();
        let mut weights = Vec::new();
        let mut task_of: SecondaryMap<MorphismId, TaskId> = SecondaryMap::new();
        let mut scalar_seen: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut loop_seen = vec![false; loops.len()];

        // First topo occurrence fixes task order. Scalar components are
        // undirected components of the scalar-only object/morphism subgraph.
        for &m in &topo {
            if is_host(m) || task_of.contains_key(m) {
                continue;
            }

            let (kind, weight) =
                if let Some(loop_id) = loop_of.get(m).copied().filter(|&id| !gated_loop[id]) {
                    if loop_seen[loop_id] {
                        continue;
                    }
                    loop_seen[loop_id] = true;
                    let morphisms: Vec<_> = topo
                        .iter()
                        .copied()
                        .filter(|&candidate| loop_of.get(candidate).copied() == Some(loop_id))
                        .collect();
                    (TaskKind::Seq { morphisms }, 1)
                } else {
                    match self.morphisms[m].op {
                        // plan-s39: a gated bulk op falls through to the scalar
                        // component (`_` arm) — it joins its Phi's Seq task, since
                        // a Split site is dispatched at launch, before the guard
                        // condition's value exists.
                        Operation::Map { .. }
                        | Operation::Zip
                        | Operation::Enumerate
                        | Operation::Iota
                        | Operation::Fill
                            if !gated.contains_key(m) && loop_of.get(m).is_none() =>
                        {
                            let n = self.bulk_element_count(m);
                            (
                                TaskKind::Split { site: m, n },
                                n.min(u32::MAX as u64) as u32,
                            )
                        }
                        Operation::Fold { .. }
                            if !gated.contains_key(m) && loop_of.get(m).is_none() =>
                        {
                            let n = self.bulk_element_count(m);
                            (
                                TaskKind::Seq { morphisms: vec![m] },
                                n.min(u32::MAX as u64) as u32,
                            )
                        }
                        // plan-s33b: the read alone, never grouped with the scalar
                        // arithmetic around it — the whole point is that its edges
                        // fence work, and a shared task would drag that work inside.
                        Operation::TimeMs => (TaskKind::Seq { morphisms: vec![m] }, 1),
                        // ADR-0045 D8: the read and its array's `Proj`, alone.
                        Operation::Read => {
                            let mut morphisms = vec![m];
                            morphisms.extend(
                                self.out_edges(self.morphisms[m].target)
                                    .iter()
                                    .filter(|&&p| self.read_of(self.morphisms[p].target).is_some()),
                            );
                            (TaskKind::Seq { morphisms }, 1)
                        }
                        _ => {
                            if scalar_seen.contains_key(m) {
                                continue;
                            }
                            let mut component: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
                            let mut stack = vec![m];
                            scalar_seen.insert(m, ());
                            while let Some(current) = stack.pop() {
                                component.insert(current, ());
                                let morph = &self.morphisms[current];
                                // plan-s49: adjacency is DATAFLOW plus CO-WRITING, and
                                // deliberately NOT "shares an object in any role".
                                //   in_edges(source)  — who produced my input   (chain)
                                //   out_edges(target) — who reads my output     (chain)
                                //   in_edges(target)  — who else writes my output:
                                //                       product slot writers ARE one
                                //                       producer and must share a task
                                // What is dropped is `out_edges(source)`: two morphisms
                                // that merely READ one object are siblings, not a chain.
                                // Merging siblings merges work with no order between it,
                                // and once one sibling's cone consumes a bulk output the
                                // other sibling feeds, that single task is both before
                                // and after the bulk task — a CYCLE. A cyclic plan is
                                // silently demoted to fully sequential emission by the
                                // backend (`path_plan_is_acyclic`), so the symptom is
                                // lost parallelism, not a crash. Two chained matmuls
                                // sharing one `k` range is the whole reproducer: every
                                // multi-layer net emitted single-threaded.
                                for &neighbor in self
                                    .in_edges(morph.source)
                                    .iter()
                                    .chain(self.in_edges(morph.target))
                                    .chain(self.out_edges(morph.target))
                                {
                                    if is_scalar(neighbor) && !scalar_seen.contains_key(neighbor) {
                                        scalar_seen.insert(neighbor, ());
                                        stack.push(neighbor);
                                    }
                                }
                            }
                            let morphisms = topo
                                .iter()
                                .copied()
                                .filter(|candidate| component.contains_key(*candidate))
                                .collect();
                            (TaskKind::Seq { morphisms }, 1)
                        }
                    }
                };

            let task_id = tasks.len();
            let members: &[MorphismId] = match &kind {
                TaskKind::Split { site, .. } => std::slice::from_ref(site),
                TaskKind::Seq { morphisms } => morphisms,
            };
            for &member in members {
                task_of.insert(member, task_id);
            }
            tasks.push(Task {
                kind,
                deps: Vec::new(),
                rank: 0,
                trap_min: None,
                pinned: false,
            });
            weights.push(weight);
        }

        let mut topo_pos: SecondaryMap<MorphismId, u32> = SecondaryMap::new();
        for (i, &m) in topo.iter().enumerate() {
            topo_pos.insert(m, i.min(u32::MAX as usize) as u32);
        }
        let mut trap_sites = vec![Vec::new(); tasks.len()];
        // The last SOURCE position each task occupies. `TimeMs` fences on it
        // (plan-time-builtin; see the checkpoint loop). Source position, not
        // topo position: pure work has no ordering relation to a clock read,
        // so topo order is free to schedule the whole program before or after
        // it — and does. What the programmer means by bracketing is the work
        // WRITTEN between the two reads, which is exactly this key.
        let task_max_loc: Vec<u32> = tasks
            .iter()
            .map(|task| {
                let members: &[MorphismId] = match &task.kind {
                    TaskKind::Split { site, .. } => std::slice::from_ref(site),
                    TaskKind::Seq { morphisms } => morphisms,
                };
                members
                    .iter()
                    .map(|&m| self.morphisms[m].loc.start)
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        // The mirror key, for the `after` half of a clock read's fence: the
        // first SOURCE position a task occupies, for the same reason.
        let task_min_loc: Vec<u32> = tasks
            .iter()
            .map(|task| {
                let members: &[MorphismId] = match &task.kind {
                    TaskKind::Split { site, .. } => std::slice::from_ref(site),
                    TaskKind::Seq { morphisms } => morphisms,
                };
                members
                    .iter()
                    .map(|&m| self.morphisms[m].loc.start)
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        // The clock-read tasks, with the source position they fence on.
        let clock_tasks: Vec<(TaskId, u32)> = tasks
            .iter()
            .enumerate()
            .filter_map(|(task_id, task)| match &task.kind {
                TaskKind::Seq { morphisms } => morphisms
                    .first()
                    .filter(|&&m| morphisms.len() == 1 && is_clock(m))
                    .map(|&m| (task_id, self.morphisms[m].loc.start)),
                TaskKind::Split { .. } => None,
            })
            .collect();

        // A task produces every target written by one of its morphisms. Product
        // slot writers are in the same scalar component, hence the same task.
        let mut producer: SecondaryMap<ObjectId, TaskId> = SecondaryMap::new();
        for (task_id, task) in tasks.iter().enumerate() {
            let members: &[MorphismId] = match &task.kind {
                TaskKind::Split { site, .. } => std::slice::from_ref(site),
                TaskKind::Seq { morphisms } => morphisms,
            };
            for &member in members {
                let target = self.morphisms[member].target;
                let old = producer.insert(target, task_id);
                debug_assert!(old.is_none_or(|id| id == task_id));
            }
        }

        // Direct object dataflow is the complete dependency relation.
        for (task_id, task) in tasks.iter_mut().enumerate() {
            let members: &[MorphismId] = match &task.kind {
                TaskKind::Split { site, .. } => std::slice::from_ref(site),
                TaskKind::Seq { morphisms } => morphisms,
            };
            for &member in members {
                let morph = &self.morphisms[member];
                if let Some(dep) = producer.get(morph.source).copied()
                    && dep != task_id
                    && !task.deps.contains(&dep)
                {
                    task.deps.push(dep);
                }
                if self.path_trap_capable(member, &bounds, &trap_capable) {
                    let pos = topo_pos[member];
                    task.trap_min = Some(task.trap_min.map_or(pos, |old| old.min(pos)));
                    trap_sites[task_id].push(pos);
                }
                if let Operation::Call(g) = morph.op
                    && !is_token(member)
                    && trap_capable.get(g).copied().unwrap_or(true)
                {
                    task.pinned = true;
                }
                // The read runs on the host spine at its own topo position, the
                // same as a trap-capable call: a worker running it would put the
                // clock wherever the pool happened to be, which is the ordering
                // this task exists to remove. A bulk op reading a host value
                // runs there too (the host-value cone above).
                if matches!(morph.op, Operation::TimeMs) || host_read_bulk.contains_key(member) {
                    task.pinned = true;
                }
            }
            task.deps.sort_unstable();
        }

        // plan-s33b: the clock read's edges, both ways. `before` is the S29
        // fence restated as dependencies — every task written entirely earlier
        // must be Done. `after` is the half that was missing: every task written
        // entirely later cannot be dispatched until the read has happened, so
        // the interval `t1 - t0` is one the bracketed work cannot leave early.
        // Source order is the key for both halves, for the reason recorded at
        // `task_max_loc`: pure work has no dataflow relation to a clock read.
        // The two sets are disjoint (`max < start` vs `min > start`) and a
        // clock task is in neither of its own, so no edge can close a cycle.
        for &(clock, start) in &clock_tasks {
            for id in 0..tasks.len() {
                if id == clock {
                    continue;
                }
                if task_max_loc[id] < start {
                    if !tasks[clock].deps.contains(&id) {
                        tasks[clock].deps.push(id);
                    }
                } else if task_min_loc[id] > start && !tasks[id].deps.contains(&clock) {
                    tasks[id].deps.push(clock);
                }
            }
        }
        if !clock_tasks.is_empty() {
            for task in &mut tasks {
                task.deps.sort_unstable();
            }
        }

        // Critical-path weight to a sink. A scalar component may first appear
        // before all of its product slots are produced, so derive a task-level
        // topo order rather than assuming task ids are already topological.
        let mut dependents = vec![Vec::new(); tasks.len()];
        let mut remaining: Vec<usize> = tasks.iter().map(|task| task.deps.len()).collect();
        for (task_id, task) in tasks.iter().enumerate() {
            for &dep in &task.deps {
                dependents[dep].push(task_id);
            }
        }
        let mut task_topo: Vec<TaskId> = remaining
            .iter()
            .enumerate()
            .filter_map(|(id, &deps)| (deps == 0).then_some(id))
            .collect();
        let mut cursor = 0;
        while cursor < task_topo.len() {
            let task_id = task_topo[cursor];
            cursor += 1;
            for &dependent in &dependents[task_id] {
                remaining[dependent] -= 1;
                if remaining[dependent] == 0 {
                    task_topo.push(dependent);
                }
            }
        }
        debug_assert_eq!(task_topo.len(), tasks.len());
        for &task_id in task_topo.iter().rev() {
            let tail = dependents[task_id]
                .iter()
                .map(|&dependent| tasks[dependent].rank)
                .max()
                .unwrap_or(0);
            tasks[task_id].rank = weights[task_id].saturating_add(tail);
        }

        let mut checkpoints = Vec::new();
        // The tasks an ungated earlier checkpoint already waited to complete:
        // a later host read of their objects needs no wait of its own.
        let mut waited: Vec<bool> = vec![false; tasks.len()];
        let is_checkpoint = |m: MorphismId| {
            let op = self.morphisms[m].op;
            matches!(op, Operation::Print { .. })
                || matches!(op, Operation::TimeMs | Operation::Read) && !task_of.contains_key(m)
                || matches!(op, Operation::Call(_)) && is_token(m)
        };
        // The progress a checkpoint at topo `point` waits for.
        let checkpoint_wait = |m: MorphismId, point: u32| -> Vec<WaitEntry> {
            let morph = &self.morphisms[m];
            let mut wait: Vec<WaitEntry> = trap_sites
                .iter()
                .enumerate()
                .filter_map(|(task, sites)| {
                    sites
                        .iter()
                        .copied()
                        .filter(|&site| site < point)
                        .max()
                        .map(|threshold| WaitEntry {
                            task,
                            threshold: Some(threshold),
                        })
                })
                .collect();

            // Token products are host-built, so walk through their unassigned
            // structural edges until reaching each consumed value's producer
            // task. That producer's own deps have already completed.
            let mut seen: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            let mut stack = vec![morph.source];
            while let Some(o) = stack.pop() {
                if seen.contains_key(o) {
                    continue;
                }
                seen.insert(o, ());
                if let Some(task_id) = producer.get(o).copied() {
                    if let Some(entry) = wait.iter_mut().find(|entry| entry.task == task_id) {
                        entry.threshold = None;
                    } else {
                        wait.push(WaitEntry {
                            task: task_id,
                            threshold: None,
                        });
                    }
                    continue;
                }
                for &incoming in self.in_edges(o) {
                    stack.push(self.morphisms[incoming].source);
                }
            }
            // plan-time-builtin composition rule 1: a clock read must not be
            // reordered across the work it brackets. Under the parallel
            // orchestrator that reordering is the default — a `TimeMs` on the
            // host spine consumes only the token, so it has NO value producer
            // to wait for and would read the clock while the tasks it is meant
            // to measure are still in flight. So `TimeMs` FENCES: every task
            // written entirely BEFORE it in the source must have completed.
            // Source order is the right key precisely because the dataflow
            // graph gives none — pure work is unordered against a clock read,
            // so topo order legally puts all of it on either side. `t1 - t0`
            // therefore measures the work written between the two reads, which
            // is what a bracket means in any normal language, and what lets one
            // opened after the data generation exclude it (the S28 finding).
            if matches!(morph.op, Operation::TimeMs) {
                for (task, &last) in task_max_loc.iter().enumerate() {
                    if last >= morph.loc.start {
                        continue;
                    }
                    match wait.iter_mut().find(|entry| entry.task == task) {
                        Some(entry) => entry.threshold = None,
                        None => wait.push(WaitEntry {
                            task,
                            threshold: None,
                        }),
                    }
                }
            }
            wait.sort_unstable_by_key(|entry| entry.task);
            wait
        };
        // Where the host spine runs each pinned task (`mapal_par_run_pinned`):
        // its first member's topo position.
        let run_at: Vec<Option<u32>> = tasks
            .iter()
            .map(|task| {
                let first = match &task.kind {
                    TaskKind::Split { site, .. } => *site,
                    TaskKind::Seq { morphisms } => morphisms[0],
                };
                task.pinned.then(|| topo_pos[first])
            })
            .collect();
        // The latest pinned run a wait needs behind it: a wait on a task that
        // is — or depends on — a pinned task the host runs later never returns
        // (the host is the one waiting).
        let pinned_floor = |wait: &[WaitEntry]| -> u32 {
            let mut seen = vec![false; tasks.len()];
            let mut stack: Vec<TaskId> = wait.iter().map(|entry| entry.task).collect();
            let mut floor = 0;
            while let Some(t) = stack.pop() {
                if std::mem::replace(&mut seen[t], true) {
                    continue;
                }
                if let Some(at) = run_at[t] {
                    floor = floor.max(at);
                }
                stack.extend(tasks[t].deps.iter().copied());
            }
            floor
        };
        // Where each checkpoint's wait goes (`Checkpoint::at`) and the
        // host-built edges it already precedes. Its value is assembled through
        // `Pair` / `Proj` edges; the wait goes before the earliest of them that
        // reads a task's object — not a token alone (it reads no bytes), and
        // none before a pinned run the wait needs (`pinned_floor`). Covered:
        // the walked edges at or after the wait, except — for a checkpoint
        // inside a loop, whose wait is hoisted to the loop's entry — the edges
        // outside that loop, which run before the entry. An edge left
        // uncovered takes its own host-read wait below.
        let mut member_loop: SecondaryMap<MorphismId, usize> = SecondaryMap::new();
        for (id, members) in loop_members.iter().enumerate() {
            for &m in members {
                member_loop.entry(m).expect("live key").or_insert(id);
            }
        }
        let mut covered: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut planned: SecondaryMap<MorphismId, (u32, Vec<WaitEntry>)> = SecondaryMap::new();
        for &m in &topo {
            if !is_checkpoint(m) {
                continue;
            }
            let point = topo_pos[m];
            let wait = checkpoint_wait(m, point);
            let floor = pinned_floor(&wait);
            let home = member_loop.get(m).copied();
            let mut at = point;
            let mut walked: Vec<MorphismId> = Vec::new();
            let mut seen: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            let mut stack = vec![self.morphisms[m].source];
            while let Some(o) = stack.pop() {
                if seen.insert(o, ()).is_some() || producer.contains_key(o) {
                    continue;
                }
                for &incoming in self.in_edges(o) {
                    let edge = &self.morphisms[incoming];
                    if !matches!(edge.op, Operation::Pair { .. } | Operation::Proj { .. }) {
                        continue;
                    }
                    if producer.contains_key(edge.source)
                        && self.objects[edge.target].ty != Ty::IoToken
                        && topo_pos[incoming] >= floor
                    {
                        at = at.min(topo_pos[incoming]);
                    }
                    walked.push(incoming);
                    stack.push(edge.source);
                }
            }
            for edge in walked {
                if topo_pos[edge] >= at
                    && (home.is_none() || member_loop.get(edge).copied() == home)
                {
                    covered.insert(edge, ());
                }
            }
            planned.insert(m, (at, wait));
        }
        for (i, &m) in topo.iter().enumerate() {
            let morph = &self.morphisms[m];
            // plan-s33b: a clock read that became a task carries its fence in
            // its own dependency edges, and the emitter's pinned sequence still
            // emits the `mapal_par_check` at this topo position — a second
            // wait list here would be the same ordering stated twice (§3).
            // A clock left on the spine (inside an effectful loop) keeps its
            // checkpoint, which is the only fence it has.
            // ADR-0045 D8: a read left on the spine (in an effectful loop)
            // can fail there, so it observes like a print: every earlier
            // trap is decided and delivered before it runs.
            let Some((at, wait)) = planned.remove(m) else {
                // structure (host reads): any other host-spine morphism that
                // reads a task's object — a `Pair` building a printing loop's
                // entry state from a struct a task filled, a `Proj` of it, the
                // scalar host cone — waits for that task first. A checkpoint's
                // wait list covers what it reaches through host-built edges; this
                // covers the rest, once per task. A token alone (a read's
                // `Proj` of it) reads no bytes: the read stays asynchronous,
                // its next checkpoint waits for it.
                if is_host(m)
                    && !covered.contains_key(m)
                    && self.objects[morph.target].ty != Ty::IoToken
                    && let Some(t) = producer.get(morph.source).copied()
                    && !waited[t]
                {
                    if !gated.contains_key(m) {
                        waited[t] = true;
                    }
                    let point = i.min(u32::MAX as usize) as u32;
                    checkpoints.push(Checkpoint {
                        topo: point,
                        at: point,
                        wait: vec![WaitEntry {
                            task: t,
                            threshold: None,
                        }],
                    });
                }
                continue;
            };
            if !gated.contains_key(m) {
                for entry in wait.iter().filter(|e| e.threshold.is_none()) {
                    waited[entry.task] = true;
                }
            }
            checkpoints.push(Checkpoint {
                topo: i.min(u32::MAX as usize) as u32,
                at,
                wait,
            });
        }
        checkpoints.push(Checkpoint {
            topo: u32::MAX,
            at: u32::MAX,
            wait: (0..tasks.len())
                .map(|task| WaitEntry {
                    task,
                    threshold: None,
                })
                .collect(),
        });

        PathPlan { tasks, checkpoints }
    }
}
