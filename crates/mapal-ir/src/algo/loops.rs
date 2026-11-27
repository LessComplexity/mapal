//! Loop facts: Tarjan SCCs, the topological order, `loop_structure` and `loop_plan`.

use super::*;

impl CategoryIr {
    /// Strongly-connected components of function `f`'s object subgraph, via
    /// **iterative** Tarjan (DESIGN §13). Edges are `f`'s morphisms; every SCC
    /// is returned, trivial (single-object, no self-loop) ones included.
    ///
    /// Order: Tarjan emits SCCs in reverse-topological order; within an SCC,
    /// objects are in the order Tarjan pops them. Deterministic given the
    /// insertion-ordered adjacency.
    pub fn sccs(&self, f: FuncId) -> Vec<Vec<ObjectId>> {
        self.fact(f, |s| &s.sccs, || self.sccs_compute(f))
    }

    pub(super) fn sccs_compute(&self, f: FuncId) -> Vec<Vec<ObjectId>> {
        let nodes = self.func_objects(f);

        // Tarjan bookkeeping in SecondaryMaps (no HashMap; I12).
        let mut index: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        let mut lowlink: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        let mut on_stack: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
        let mut visited: SecondaryMap<ObjectId, bool> = SecondaryMap::new();

        let mut next_index: u32 = 0;
        let mut tarjan_stack: Vec<ObjectId> = Vec::new();
        let mut sccs: Vec<Vec<ObjectId>> = Vec::new();

        // Explicit DFS frame: the node, and a cursor into its out-edges.
        struct Frame {
            node: ObjectId,
            edge_cursor: usize,
        }

        for &root in &nodes {
            if visited.get(root).copied().unwrap_or(false) {
                continue;
            }
            let mut call_stack: Vec<Frame> = vec![Frame {
                node: root,
                edge_cursor: 0,
            }];

            // Initialize root on first push.
            index.insert(root, next_index);
            lowlink.insert(root, next_index);
            next_index += 1;
            visited.insert(root, true);
            on_stack.insert(root, true);
            tarjan_stack.push(root);

            while let Some(frame) = call_stack.last_mut() {
                let v = frame.node;
                let out = self.out_edges(v);
                let mut recursed = false;

                while frame.edge_cursor < out.len() {
                    let m = out[frame.edge_cursor];
                    frame.edge_cursor += 1;
                    let w = self.morphisms[m].target;
                    // Only follow edges that stay within f's object set.
                    if self.owner.get(w) != Some(&f) {
                        continue;
                    }
                    if !visited.get(w).copied().unwrap_or(false) {
                        // "Recurse" into w.
                        index.insert(w, next_index);
                        lowlink.insert(w, next_index);
                        next_index += 1;
                        visited.insert(w, true);
                        on_stack.insert(w, true);
                        tarjan_stack.push(w);
                        call_stack.push(Frame {
                            node: w,
                            edge_cursor: 0,
                        });
                        recursed = true;
                        break;
                    } else if on_stack.get(w).copied().unwrap_or(false) {
                        let iw = index[w];
                        let lv = lowlink[v];
                        if iw < lv {
                            lowlink.insert(v, iw);
                        }
                    }
                }

                if recursed {
                    continue;
                }

                // Done with v's edges: if it is an SCC root, pop the component.
                if lowlink[v] == index[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = tarjan_stack.pop().expect("tarjan stack underflow");
                        on_stack.insert(w, false);
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(comp);
                }

                // Pop v's frame and propagate lowlink to the parent.
                call_stack.pop();
                if let Some(parent) = call_stack.last() {
                    let p = parent.node;
                    let lv = lowlink[v];
                    let lp = lowlink[p];
                    if lv < lp {
                        lowlink.insert(p, lv);
                    }
                }
            }
        }

        sccs
    }

    /// A topological order of `f`'s morphisms (DESIGN §13), Kahn's algorithm
    /// over the morphisms with **`LoopBack` edges excluded as gating** (but
    /// still emitted — interp needs them).
    ///
    /// Completion rule: Parameters/Constants start complete; a product object
    /// completes when all `arity` slot edges are emitted; a `LoopMerge`
    /// completes on its `LoopEnter` alone (header-first); every other object
    /// completes on its one definer. `LoopExit` edges are ordinary gating edges.
    /// `LoopBack` is appended after its source completes, never gating.
    ///
    /// **Ties break on source position** — `(loc.start, loc.end, insertion
    /// index)` — never on insertion order alone (plan-s38). The dataflow graph
    /// does not order two independent trapping operations, yet which of them
    /// traps first is observable, so the tie-break must be a function of the
    /// *program*. Insertion order is a function of the compiler: rewriting
    /// creates and destroys objects, so `Inline` could turn `Trapped(IndexOob)`
    /// into `Trapped(DivZero)` and break `eval ∘ rewrite = eval`. Source
    /// position is the only intrinsic key available, and it is the same
    /// reasoning as S29's clock-read fence at wider scope. The insertion index
    /// keeps the key total, so the walk stays deterministic when a desugaring
    /// emits several morphisms from one span.
    ///
    /// Note: exact `loc` ties (a Parameter and Return sharing the function
    /// span, loop objects sharing the loop span) fall back to insertion order.
    /// None of those objects trap on their own, so the ordering they receive is
    /// unobservable.
    ///
    /// **`LoopEnter` edges are deferred**: a ready `LoopEnter` is released only
    /// when no other morphism is ready. This guarantees every morphism not
    /// (transitively) gated by a loop merge — i.e. every loop-invariant
    /// computation, however many hops from its sources — precedes the loop
    /// header in the order. The interp loop driver and straight-line backends
    /// rely on this: they read loop-invariant operands when the header fires
    /// (S12 fix; previously a multi-hop invariant like `x * 2` inside a loop
    /// body was ordered after its `LoopEnter` and read-before-write).
    ///
    /// **Clock reads cut the order** ([`Self::fence_clocks`], plan-s53): the
    /// work written before a `time` precedes it and the work written after it
    /// follows. A function without one keeps the order above byte for byte.
    pub fn topo_order(&self, f: FuncId) -> Vec<MorphismId> {
        self.fact(f, |s| &s.topo_order, || self.topo_order_compute(f))
    }

    pub(super) fn topo_order_compute(&self, f: FuncId) -> Vec<MorphismId> {
        let nodes = self.func_objects(f);

        // For each object, how many in-edges must fire before it "completes".
        // (b)/(c)/(d)/(e) of I3: product objects need all arity Pair edges;
        // LoopMerge needs only the LoopEnter; others need their single definer;
        // Parameters/Constants need zero.
        let mut remaining: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        for &o in &nodes {
            let obj = &self.objects[o];
            let count = match obj.kind {
                ObjectKind::Parameter | ObjectKind::Constant => 0,
                ObjectKind::LoopMerge => {
                    // Completes on the single LoopEnter edge alone.
                    self.in_edges(o)
                        .iter()
                        .filter(|&&m| self.morphisms[m].op == Operation::LoopEnter)
                        .count() as u32
                }
                _ => {
                    // Temporary/Return: count the gating in-edges. LoopBack edges
                    // never target these (they target LoopMerge), so every in-edge
                    // here gates.
                    self.in_edges(o).len() as u32
                }
            };
            remaining.insert(o, count);
        }

        // The tie-break key per object: source position, made total by the
        // object's insertion index. `nodes` is in insertion order, so the index
        // is also how we get back from a popped key to its object.
        let mut pos: SecondaryMap<ObjectId, (u32, u32, u32)> = SecondaryMap::new();
        for (i, &o) in nodes.iter().enumerate() {
            let loc = self.objects[o].loc;
            pos.insert(o, (loc.start, loc.end, i as u32));
        }

        // Objects that are complete and ready to release their out-edges,
        // popped in source-position order.
        let mut ready: BinaryHeap<Reverse<(u32, u32, u32)>> = BinaryHeap::new();
        for &o in &nodes {
            if remaining[o] == 0 {
                ready.push(Reverse(pos[o]));
            }
        }

        let mut order: Vec<MorphismId> = Vec::new();
        let mut released: SecondaryMap<ObjectId, bool> = SecondaryMap::new();

        // LoopEnter edges whose source is complete, awaiting a drained worklist
        // (the deferral rule above). FIFO in source-release order (deterministic).
        let mut deferred_enters: Vec<MorphismId> = Vec::new();
        let mut dcursor = 0;

        // Process ready objects source-position-first. When the worklist drains,
        // release the next deferred LoopEnter and continue.
        loop {
            while let Some(Reverse(key)) = ready.pop() {
                let o = nodes[key.2 as usize];
                if released.get(o).copied().unwrap_or(false) {
                    continue;
                }
                released.insert(o, true);

                for &m in self.out_edges(o) {
                    let morph = &self.morphisms[m];
                    if morph.op == Operation::LoopBack {
                        // Emitted but never gates: append once its source is complete.
                        order.push(m);
                        continue;
                    }
                    if morph.op == Operation::LoopEnter {
                        // Deferred: released only when nothing else is ready.
                        deferred_enters.push(m);
                        continue;
                    }
                    // Ordinary gating edge: emit it, then decrement the target.
                    order.push(m);
                    let tgt = morph.target;
                    if self.owner.get(tgt) != Some(&f) {
                        continue;
                    }
                    let r = remaining[tgt];
                    if r > 0 {
                        let nr = r - 1;
                        remaining.insert(tgt, nr);
                        if nr == 0 {
                            ready.push(Reverse(pos[tgt]));
                        }
                    }
                }
            }

            // Worklist drained: release the next deferred LoopEnter, if any.
            if dcursor < deferred_enters.len() {
                let m = deferred_enters[dcursor];
                dcursor += 1;
                order.push(m);
                let tgt = self.morphisms[m].target;
                if self.owner.get(tgt) == Some(&f) {
                    let r = remaining[tgt];
                    if r > 0 {
                        let nr = r - 1;
                        remaining.insert(tgt, nr);
                        if nr == 0 {
                            ready.push(Reverse(pos[tgt]));
                        }
                    }
                }
                continue;
            }
            break;
        }

        self.fence_clocks(f, order)
    }

    /// plan-s53 `clock-fence`: a clock read sits after the work written before
    /// it and before the work written after it. Kahn alone cannot see that — a
    /// `TimeMs` consumes only the token, so it is released the moment the token
    /// is, and on the GPU e2e program (S52) `t0` landed before the input
    /// generation and `t1` before the site: `t1 - t0` timed the generation.
    /// `path_plan` fences the parallel walk by source position (S29/S33b); this
    /// is the same key for every walk of `topo_order` — the sequential emitter,
    /// the interpreter, and the plans that take positions from it.
    ///
    /// `own(m)` = 2 × (clocks written before `m`), + 1 if `m` is itself a clock;
    /// `seg(m)` = max(`own(m)`, `seg` of every producer of `m.source`), to a
    /// fixpoint, and every morphism incident to a loop SCC takes the SCC's
    /// maximum — the loop runs as one unit at its header, so the header must
    /// follow everything the loop reads. `seg` never decreases along an edge,
    /// so the stable sort by it is still a topological order, and equal
    /// segments keep Kahn's order. A clock inside a loop is not a cut: it reads
    /// once per iteration, at its place in the cycle.
    pub(super) fn fence_clocks(&self, f: FuncId, mut order: Vec<MorphismId>) -> Vec<MorphismId> {
        if !order
            .iter()
            .any(|&m| self.morphisms[m].op == Operation::TimeMs)
        {
            return order;
        }
        let loops = self.loop_structure(f);
        let mut loop_of: SecondaryMap<ObjectId, usize> = SecondaryMap::new();
        for (i, scc) in loops.iter().enumerate() {
            for &o in &scc.objects {
                loop_of.insert(o, i);
            }
        }
        let is_cut = |m: MorphismId| {
            let morph = &self.morphisms[m];
            morph.op == Operation::TimeMs && !loop_of.contains_key(morph.source)
        };
        let mut cuts: Vec<u32> = order
            .iter()
            .filter(|&&m| is_cut(m))
            .map(|&m| self.morphisms[m].loc.start)
            .collect();
        if cuts.is_empty() {
            return order;
        }
        cuts.sort_unstable();
        let own = |m: MorphismId| {
            let before = cuts.partition_point(|&s| s < self.morphisms[m].loc.start) as u32;
            2 * before + is_cut(m) as u32
        };
        let incident = |m: MorphismId| {
            let morph = &self.morphisms[m];
            loop_of
                .get(morph.source)
                .or_else(|| loop_of.get(morph.target))
                .copied()
        };

        let mut seg: SecondaryMap<MorphismId, u32> = order.iter().map(|&m| (m, own(m))).collect();
        loop {
            let mut changed = false;
            for &m in &order {
                let from = self
                    .in_edges(self.morphisms[m].source)
                    .iter()
                    .filter_map(|&p| seg.get(p).copied())
                    .max()
                    .unwrap_or(0);
                if from > seg[m] {
                    seg[m] = from;
                    changed = true;
                }
            }
            let mut top = vec![0; loops.len()];
            for &m in &order {
                if let Some(i) = incident(m) {
                    top[i] = top[i].max(seg[m]);
                }
            }
            for &m in &order {
                if let Some(i) = incident(m)
                    && seg[m] < top[i]
                {
                    seg[m] = top[i];
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        debug_assert!(
            order.iter().all(|&m| !is_cut(m) || seg[m] == own(m)),
            "a clock read was pushed past its own segment"
        );
        order.sort_by_key(|&m| seg[m]);
        order
    }

    /// Loop regions recovered by SCC (DESIGN §7/§13): one [`LoopScc`] per
    /// non-trivial SCC, each tagged with its `LoopMerge` objects.
    ///
    /// A trivial SCC (one object, no self-loop) is not a loop and is excluded.
    pub fn loop_structure(&self, f: FuncId) -> Vec<LoopScc> {
        self.fact(f, |s| &s.loop_structure, || self.loop_structure_compute(f))
    }

    pub(super) fn loop_structure_compute(&self, f: FuncId) -> Vec<LoopScc> {
        let mut out = Vec::new();
        for comp in self.sccs(f) {
            let nontrivial = comp.len() > 1 || self.has_self_loop(comp[0]);
            if !nontrivial {
                continue;
            }
            let mut merges: Vec<ObjectId> = comp
                .iter()
                .copied()
                .filter(|&o| self.objects[o].kind == ObjectKind::LoopMerge)
                .collect();
            // Deterministic order for merges (by insertion within the comp).
            merges.sort_by_key(|&o| self.object_seq(o));
            let mut objects = comp;
            objects.sort_by_key(|&o| self.object_seq(o));
            out.push(LoopScc { objects, merges });
        }
        out
    }

    /// The per-merge loop attribution for the loop headed by `merge` in function
    /// `f` (DESIGN §3, BL7). Returns `None` for any non-canonical shape: `merge`
    /// not in a nontrivial SCC, a multi-merge SCC, ≠1 `LoopEnter`/`LoopBack` into
    /// `merge`, or ≠1 `LoopExit` attributed to this merge's SCC.
    ///
    /// **Exit attribution** is by *route-feeder membership* (the S12 rule): a
    /// `LoopExit` belongs to this merge iff its route object is `Pair`-fed by an
    /// object of THIS merge's SCC — never per-function union or reachability,
    /// which mis-attribute a downstream loop's exit to an upstream merge in a
    /// two-sequential-loop function.
    ///
    /// **Decide/advance split** (ADR-0016): the decide cone is every body
    /// morphism whose target is backward-reachable from `exit_route` (cond +
    /// exit-route feeders, incl. exit-feeding effects); the advance cone is the
    /// rest (next-state), unreachable on the exit step.
    pub fn loop_plan(&self, f: FuncId, merge: ObjectId) -> Option<LoopPlan> {
        let plans = |s: &crate::facts::FuncFacts| -> Option<LoopPlan> {
            s.loop_plans
                .get_or_init(|| {
                    self.loop_structure(f)
                        .iter()
                        .flat_map(|scc| scc.merges.iter())
                        .map(|&m| (m, self.loop_plan_compute(f, m)))
                        .collect()
                })
                .iter()
                .find(|(m, _)| *m == merge)
                .and_then(|(_, p)| p.clone())
        };
        match self.facts_of(f) {
            Some(s) => plans(s),
            None => self.loop_plan_compute(f, merge),
        }
    }

    pub(super) fn loop_plan_compute(&self, f: FuncId, merge: ObjectId) -> Option<LoopPlan> {
        // Single-merge SCC (M1 canonical shape) — THIS merge's SCC only, never
        // the per-function union.
        let scc = self
            .loop_structure(f)
            .into_iter()
            .find(|s| s.merges.contains(&merge))?;
        if scc.merges.len() != 1 {
            return None;
        }
        let in_scc: SecondaryMap<ObjectId, ()> = {
            let mut m = SecondaryMap::new();
            for &o in &scc.objects {
                m.insert(o, ());
            }
            m
        };

        // init / back_route from the merge's in-edges.
        let mut init = None;
        let mut back_routes: Vec<ObjectId> = Vec::new();
        for &m in self.in_edges(merge) {
            let morph = &self.morphisms[m];
            match morph.op {
                Operation::LoopEnter => {
                    if init.is_some() {
                        return None;
                    }
                    init = Some(morph.source);
                }
                Operation::LoopBack => back_routes.push(morph.source),
                _ => {}
            }
        }
        let init = init?;
        if back_routes.len() != 1 {
            return None;
        }
        let back_route = back_routes[0];

        // exits: LoopExit morphisms whose route object is fed (slot-Pair source)
        // by an in-SCC value — attribute by route-feeder membership.
        let mut exits: Vec<MorphismId> = Vec::new();
        for (mid, morph) in self.morphisms() {
            if self.try_owner(morph.source) != Some(f) {
                continue;
            }
            if morph.op != Operation::LoopExit {
                continue;
            }
            let route = morph.source;
            let fed_by_scc = self.in_edges(route).iter().any(|&pm| {
                let pmo = &self.morphisms[pm];
                matches!(pmo.op, Operation::Pair { .. }) && in_scc.contains_key(pmo.source)
            });
            if fed_by_scc {
                exits.push(mid);
            }
        }
        if exits.len() != 1 {
            return None;
        }
        let exit_route = self.morphisms[exits[0]].source;

        // The loop-variant cone of the route feeders: every object the route
        // Pair edges transitively depend on that is also forward-reachable
        // from the merge, over non-loop morphisms of `f`. A computed payload
        // (`t * 2 -> ret`) or an exit-arm fanout leaves the SCC — its chain
        // feeds only a route, never cycles back to the merge — so
        // SCC-incidence plus direct route Pairs never schedules it (ADR-0027
        // review: interp read-before-write; llvm emitted it after the loop).
        // Loop ops are never cone edges, so neither walk crosses into a
        // neighboring loop (gated by its LoopExit) nor into the merge's own
        // LoopEnter/LoopBack.
        //
        // The merge-reachability bound matters: a loop-INVARIANT producer
        // (unreachable from the merge — e.g. a param proj also read by the
        // init assembly) must stay walk-owned so it evaluates exactly once,
        // before the driver fires (the S12 rule). Only its *uses* inside the
        // cone re-fire per iteration; the invariant object itself is read
        // from the pre-loop env.
        let mut cone: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        cone.insert(back_route, ());
        cone.insert(exit_route, ());
        let mut variant: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        variant.insert(merge, ());
        loop {
            let mut changed = false;
            for (_, morph) in self.morphisms() {
                if self.try_owner(morph.source) != Some(f) {
                    continue;
                }
                if matches!(
                    morph.op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                ) {
                    continue;
                }
                if cone.contains_key(morph.target) && !cone.contains_key(morph.source) {
                    cone.insert(morph.source, ());
                    changed = true;
                }
                if variant.contains_key(morph.source) && !variant.contains_key(morph.target) {
                    variant.insert(morph.target, ());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // body_order: topo_order filtered to morphisms incident to SCC(merge) OR
        // targeting a loop-variant object, excluding loop ops. Route objects
        // sit outside the SCC, but every morphism assembling them — direct
        // Pair edges and the whole computed-payload chain behind them — must
        // re-fire each iteration, so membership-by-SCC alone would drop them.
        // So must a trip's dead work (a value that feeds no route: it runs,
        // and may trap, every trip — generic-fusion's twin sweep: a dead site
        // over the state ran once outside the loop, the interpreter reading a
        // half-staged capture, the parallel host racing the loop for its
        // view).
        let body_order: Vec<MorphismId> = self
            .topo_order(f)
            .into_iter()
            .filter(|&m| {
                let morph = &self.morphisms[m];
                if matches!(
                    morph.op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                ) {
                    return false;
                }
                in_scc.contains_key(morph.source)
                    || in_scc.contains_key(morph.target)
                    || variant.contains_key(morph.target)
            })
            .collect();

        // Decide set (ADR-0016): objects backward-reachable within body_order's
        // edges from exit_route. Fixpoint over the (small) body.
        let mut in_d: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        in_d.insert(exit_route, ());
        loop {
            let mut changed = false;
            for &m in &body_order {
                let morph = &self.morphisms[m];
                if in_d.contains_key(morph.target) && !in_d.contains_key(morph.source) {
                    in_d.insert(morph.source, ());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut decide_order = Vec::new();
        let mut advance_order = Vec::new();
        for &m in &body_order {
            let morph = &self.morphisms[m];
            if in_d.contains_key(morph.target) {
                decide_order.push(m);
            } else {
                advance_order.push(m);
            }
        }

        // Product targets assembled by Pair edges across the whole body.
        let mut product_targets: Vec<ObjectId> = Vec::new();
        for &m in &body_order {
            let morph = &self.morphisms[m];
            if matches!(morph.op, Operation::Pair { .. })
                && !product_targets.contains(&morph.target)
            {
                product_targets.push(morph.target);
            }
        }

        Some(LoopPlan {
            merge,
            init,
            back_route,
            exits,
            exit_route,
            scc_objects: scc.objects,
            decide_order,
            advance_order,
            product_targets,
        })
    }

    /// Whether `o` has an edge to itself (the only way a 1-object SCC is a loop).
    pub(super) fn has_self_loop(&self, o: ObjectId) -> bool {
        self.out_edges(o)
            .iter()
            .any(|&m| self.morphisms[m].target == o)
    }

    /// A stable sequence number for an object (its position in insertion order),
    /// used to make `loop_structure` output order independent of Tarjan's
    /// pop order.
    pub(super) fn object_seq(&self, o: ObjectId) -> usize {
        self.objects
            .iter()
            .position(|(id, _)| id == o)
            .unwrap_or(usize::MAX)
    }
}
