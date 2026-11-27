//! `LastUsePlan` and the `last_use_plan` query (death points, escapes, loop-carried values).

use super::*;

/// The per-fn last-use plan (docs/components/ir/plans/plan-last-use.md §2 —
/// the deduced query `last_use : IR × FuncId → LastUsePlan`, the BL7 pattern
/// alongside [`loop_plan`]): per-object death positions plus the escape and
/// loop-carried classifications the backend consumers (in-place `Update`,
/// back-edge freeing, arena coloring) read through [`LastUsePlan::dead_after`]
/// and the accessors. Total and deterministic on any sealed fn (rules 5-6):
/// non-canonical loops contribute no carried set and no position adjustment,
/// so consumers fall back to today's behavior on `None`-shaped answers.

#[derive(Clone, Debug, PartialEq)]
pub struct LastUsePlan {
    /// Rule 1's oracle-order position per morphism: `topo_order`, with every
    /// canonical loop's morphisms re-ranked **decide < `LoopExit` < advance <
    /// `LoopBack`** within the topo slots they already occupied (the
    /// guard-first quartet's execution order, ADR-0016; a global permutation —
    /// non-loop morphisms keep their topo positions).
    pub(super) position: SecondaryMap<MorphismId, u32>,
    /// The greatest adjusted position of any use of the object, counting a
    /// `Pair`/`Phi` edge as a use at the product's own last-use position —
    /// rule 2's retention "through Pair fields and Phi arms" applied to
    /// liveness: a packed handle lives as long as the product holding it.
    /// Absent for objects with no uses.
    pub(super) use_pos: SecondaryMap<ObjectId, u32>,
    /// Rule 2's escape set (conservative): every `Parameter`, plus anything
    /// reaching the fn's output — but for the loop's own carried state (the
    /// merge, its `Proj` views, the back-route `Pair` cone) the path through
    /// that loop's own `LoopExit` does NOT count (the per-iteration release
    /// valve: the escaping final instance is protected via the exit OBJECT,
    /// which is not exempt). Every other escape path (a different loop's
    /// exit excepted only when it, too, is the carrier's own) counts.
    pub(super) escapes: SecondaryMap<ObjectId, bool>,
    /// `carried_by` (rule 3): object → the `LoopMerge` its value crosses a
    /// `LoopBack` into — the back-route `Pair` cone, loop-body objects only
    /// (a loop-invariant route feeder is not carried).
    pub(super) carried: SecondaryMap<ObjectId, ObjectId>,
    /// `use_pos` when forwarding re-exposes the bytes instead of copying them
    /// (a view, S65): a `Proj` of an array-holding component and a
    /// `LoopEnter` pin too — the projection lives on the product's bytes, the
    /// merge's first state and the loop's exit objects (which may be that
    /// state) on the init's. The pins [`escape_reach`] already counts, as
    /// liveness. `LoopExit` is no pin: a loop's own buffers are not written
    /// once it has exited, and the init's ride the `LoopEnter` pin.
    pub(super) view_pos: SecondaryMap<ObjectId, u32>,
}

impl LastUsePlan {
    /// The morphism's adjusted position (rule 1's oracle-order ranking).
    /// `Some` for every morphism of the fn.
    pub fn position(&self, m: MorphismId) -> Option<u32> {
        self.position.get(m).copied()
    }

    /// `death` (plan §2): the greatest adjusted position of any use of `o`.
    /// `None` (⊥) for escaping objects (rule 2 — never dead inside the fn),
    /// and also `None` for objects with no uses (no use position exists; such
    /// an object is dead everywhere after its definition — [`Self::dead_after`]
    /// answers `true` for it).
    pub fn death(&self, o: ObjectId) -> Option<u32> {
        if self.escapes(o) {
            return None;
        }
        self.use_pos.get(o).copied()
    }

    /// `escapes` (rule 2): may this object's value outlive the fn (reachable
    /// into `Output`/`Return` incl. through `Pair` fields and `Phi` arms, a
    /// borrowed `Parameter`, or an outer fn's capture source — the last rides
    /// the body fn's own `Parameter` seed, per-fn)? Escaping objects are
    /// never freed and never written in place.
    pub fn escapes(&self, o: ObjectId) -> bool {
        self.escapes.get(o).copied().unwrap_or(false)
    }

    /// `carried_by` (rule 3): the `LoopMerge` this object's value crosses a
    /// `LoopBack` into — it lives "into the next iteration" (two-iteration
    /// liveness). `None` for everything else, the merge itself included.
    pub fn carried_by(&self, o: ObjectId) -> Option<ObjectId> {
        self.carried.get(o).copied()
    }

    /// `dead_after` (rule 4's consumer predicate): all uses of `o` sit at or
    /// before `idx` (rule 1's ranking, retention pins included) AND `o` does
    /// not escape AND `o` is not loop-carried. `idx` is typically
    /// [`Self::position`] of the writing morphism (e.g. the `Update` whose
    /// source is `o`). Rule 4's remaining half is the consumer's: a loop's
    /// borrowed init (`escapes(init)` — rule 2) is never written in place.
    pub fn dead_after(&self, o: ObjectId, idx: u32) -> bool {
        !self.escapes(o)
            && self.carried.get(o).is_none()
            && self.use_pos.get(o).copied().is_none_or(|p| p <= idx)
    }

    /// `view_death` (S65): [`Self::death`] of `o`'s **bytes** when forwarding
    /// does not copy them — the greatest position at which `o` or any view of
    /// it (a product holding it, a projection of that, a `Phi` choosing it, a
    /// loop state starting from it, an exit object that may be that state) is
    /// used. `None` for escaping objects.
    pub fn view_death(&self, o: ObjectId) -> Option<u32> {
        if self.escapes(o) {
            return None;
        }
        self.view_pos.get(o).copied()
    }

    /// The greatest position at which `o` or a view of its bytes is used,
    /// escape aside: within one trip of a loop, what its own exit forwards out
    /// is not read again (the per-iteration release valve of `escapes`).
    pub fn view_last_use(&self, o: ObjectId) -> Option<u32> {
        self.view_pos.get(o).copied()
    }

    /// [`Self::dead_after`] over [`Self::view_death`]: no view of `o`'s bytes
    /// is used after `idx`, `o` does not escape and is not carried.
    pub fn view_dead_after(&self, o: ObjectId, idx: u32) -> bool {
        !self.escapes(o)
            && self.carried.get(o).is_none()
            && self.view_pos.get(o).copied().is_none_or(|p| p <= idx)
    }
}

/// Whether a value of `ty` holds array bytes (an array, or a product with one).
pub(super) fn holds_array(ty: &Ty) -> bool {
    match ty {
        Ty::Array { .. } => true,
        Ty::Tuple(ts) => ts.iter().any(holds_array),
        Ty::Struct { fields, .. } => fields.iter().any(|(_, t)| holds_array(t)),
        _ => false,
    }
}

impl CategoryIr {
    /// The per-fn last-use plan (docs/components/ir/plans/plan-last-use.md §2,
    /// rules 1-6): death positions + escape/carried classification, composed
    /// from [`topo_order`](CategoryIr::topo_order) and
    /// [`loop_plan`](CategoryIr::loop_plan) — never re-derived. `O(V + E)`,
    /// recursion-free (J1), total on any sealed fn: non-canonical loops
    /// contribute no carried set and no re-ranking, so consumers fall back.
    pub fn last_use_plan(&self, f: FuncId) -> LastUsePlan {
        self.fact(f, |s| &s.last_use_plan, || self.last_use_plan_compute(f))
    }

    pub(super) fn last_use_plan_compute(&self, f: FuncId) -> LastUsePlan {
        let topo = self.topo_order(f);
        let mut position: SecondaryMap<MorphismId, u32> = SecondaryMap::new();
        for (i, &m) in topo.iter().enumerate() {
            position.insert(m, i as u32);
        }

        // Canonical loops with their plans (non-canonical shapes contribute
        // nothing — rule 6's graceful degradation).
        let mut plans: Vec<LoopPlan> = Vec::new();
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                if let Some(plan) = self.loop_plan(f, merge) {
                    plans.push(plan);
                }
            }
        }

        // Rule 1's re-ranking: within each canonical loop, the guard-first
        // quartet's execution order — decide cone, then the LoopExit, then
        // the advance cone, then the LoopBack — spliced into the topo slots
        // those morphisms already occupy (a global permutation: every other
        // morphism keeps its topo position; ties inside a phase keep topo
        // order). The LoopExit rank between the cones is what lets the
        // exit-route retention pin land correctly: an exit-route Pair use of
        // the merge precedes an advance-cone write (matmul4 — legal) but
        // follows a decide-cone write into the same buffer (illegal).
        let mut claimed: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for plan in &plans {
            let mut members: Vec<(MorphismId, u8, u32)> = Vec::new();
            for &m in &plan.decide_order {
                members.push((m, 0, position[m]));
            }
            for &m in &plan.exits {
                members.push((m, 1, position[m]));
            }
            for &m in &plan.advance_order {
                members.push((m, 2, position[m]));
            }
            for &m in self.in_edges(plan.merge) {
                if self.morphisms[m].op == Operation::LoopBack {
                    members.push((m, 3, position[m]));
                }
            }
            // Nested canonical loops overlap (the inner's body is part of the
            // outer's): an already-claimed slot degrades this loop's ranking
            // (rule 6 — consumers fall back; carried/escape are unaffected).
            if members.iter().any(|&(m, _, _)| claimed.contains_key(m)) {
                continue;
            }
            let mut slots: Vec<u32> = members.iter().map(|&(_, _, p)| p).collect();
            slots.sort_unstable();
            members.sort_by_key(|&(_, phase, p)| (phase, p));
            for (slot, (m, _, _)) in slots.iter().zip(members.iter()) {
                position.insert(*m, *slot);
                claimed.insert(*m, ());
            }
        }

        // carried_by (rule 3) + the exempt set (the merge, its Proj views —
        // the current-state aliases — and the back-route Pair cone) per
        // canonical loop. Body objects only: a loop-INVARIANT route feeder
        // (not merge-reachable) is not carried — its single buffer would be
        // re-read after a back-edge free.
        let mut carried: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();
        let mut exempt: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for plan in &plans {
            let mut body: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for &o in &plan.scc_objects {
                body.insert(o, ());
            }
            for &m in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                body.insert(self.morphisms[m].target, ());
            }
            // The merge and its Proj-view closure.
            let mut stack = vec![plan.merge];
            while let Some(v) = stack.pop() {
                if exempt.contains_key(v) {
                    continue;
                }
                exempt.insert(v, ());
                for &m in self.out_edges(v) {
                    if matches!(self.morphisms[m].op, Operation::Proj { .. }) {
                        stack.push(self.morphisms[m].target);
                    }
                }
            }
            // The back-route Pair cone (the next-state chain crossing the
            // LoopBack), restricted to loop-body objects. Entry is through
            // the route's STATE field (slot 0 — the `(U × Bool) → U`
            // contract; the cond at slot 1 is consumed, not carried); from
            // there, every Pair feeder is state structure (a product state
            // packs its components at all slots).
            carried.insert(plan.back_route, plan.merge);
            exempt.insert(plan.back_route, ());
            let mut stack: Vec<ObjectId> = self
                .in_edges(plan.back_route)
                .iter()
                .filter(|&&m| matches!(self.morphisms[m].op, Operation::Pair { slot: 0, .. }))
                .map(|&m| self.morphisms[m].source)
                .collect();
            while let Some(v) = stack.pop() {
                if carried.contains_key(v) {
                    continue;
                }
                carried.insert(v, plan.merge);
                exempt.insert(v, ());
                for &m in self.in_edges(v) {
                    let morph = &self.morphisms[m];
                    if matches!(morph.op, Operation::Pair { .. }) && body.contains_key(morph.source)
                    {
                        stack.push(morph.source);
                    }
                }
            }
        }

        // Rule 2's escape sets, by reverse reachability from the fn's output:
        // full (any path) and no-LoopExit (the exempt set's variant — carried
        // state escapes only via a path that crosses NO loop exit, e.g. a
        // direct post-loop consumer; its own loop's exit is the per-iteration
        // release valve, and the escaping final instance rides the exit
        // object, which is NOT exempt). Parameters always escape (borrowed —
        // an outer fn's capture source rides the body fn's own Parameter).
        let fd = self.func(f).expect("func resolves");
        let esc_full = self.escape_reach(f, fd.output, true);
        let esc_noexit = self.escape_reach(f, fd.output, false);
        let mut escapes: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
        for (o, obj) in self.objects() {
            if self.try_owner(o) != Some(f) {
                continue;
            }
            let esc = obj.kind == ObjectKind::Parameter
                || if exempt.contains_key(o) {
                    esc_noexit.get(o).copied().unwrap_or(false)
                } else {
                    esc_full.get(o).copied().unwrap_or(false)
                };
            if esc {
                escapes.insert(o, true);
            }
        }

        // use_pos: the greatest adjusted use position, with Pair/Phi
        // retention pins (a packed handle lives as long as the product
        // holding it). Swept in descending definer-position order — a
        // product's definer always outranks its components' (def-before-use
        // survives the phase permutation), so its use_pos is settled first.
        let mut definer: Vec<(ObjectId, Option<u32>)> = Vec::new();
        for (o, obj) in self.objects() {
            if self.try_owner(o) != Some(f) {
                continue;
            }
            let d = match obj.kind {
                ObjectKind::Parameter | ObjectKind::Constant => None,
                ObjectKind::LoopMerge => self
                    .in_edges(o)
                    .iter()
                    .filter(|&&m| self.morphisms[m].op == Operation::LoopEnter)
                    .map(|&m| position[m])
                    .max(),
                _ => self.in_edges(o).iter().map(|&m| position[m]).max(),
            };
            definer.push((o, d));
        }
        // Descending definer position; no-definer objects (Parameters,
        // Constants) strictly last. The sort is stable, so ties keep
        // insertion order (deterministic, and no per-object position scans —
        // object_seq would make this O(V² log V) on the §16 deep graph).
        definer.sort_by_key(|&(_, d)| (d.is_none(), d.map_or(0, |p| u32::MAX - p)));
        let mut use_pos: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        for &(o, _) in &definer {
            let mut best: Option<u32> = None;
            for &m in self.out_edges(o) {
                let p = position[m];
                best = Some(best.map_or(p, |b: u32| b.max(p)));
                if matches!(
                    self.morphisms[m].op,
                    Operation::Pair { .. } | Operation::Phi
                ) && let Some(&up) = use_pos.get(self.morphisms[m].target)
                {
                    best = Some(best.map_or(up, |b: u32| b.max(up)));
                }
            }
            if let Some(b) = best {
                use_pos.insert(o, b);
            }
        }

        // view_pos: the same sweep, every forwarding morphism a pin. A pin's
        // target is defined after its source (a `LoopEnter`'s merge and the
        // loop's `LoopExit` targets after the init), so it is settled first.
        let exits_of = |merge: ObjectId| -> Vec<ObjectId> {
            plans
                .iter()
                .filter(|p| p.merge == merge)
                .flat_map(|p| p.exits.iter().map(|&x| self.morphisms[x].target))
                .collect()
        };
        let mut view_pos: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        for &(o, _) in &definer {
            let mut best: Option<u32> = None;
            for &m in self.out_edges(o) {
                let morph = &self.morphisms[m];
                let pins: Vec<ObjectId> = match morph.op {
                    Operation::Pair { .. } | Operation::Phi => vec![morph.target],
                    Operation::Proj { index }
                        if self.objects[o]
                            .ty
                            .component_ty(index)
                            .is_some_and(holds_array) =>
                    {
                        vec![morph.target]
                    }
                    Operation::LoopEnter => {
                        let mut v = exits_of(morph.target);
                        v.push(morph.target);
                        v
                    }
                    _ => Vec::new(),
                };
                let p = pins
                    .iter()
                    .filter_map(|&t| view_pos.get(t).copied())
                    .fold(position[m], u32::max);
                best = Some(best.map_or(p, |b: u32| b.max(p)));
            }
            if let Some(b) = best {
                view_pos.insert(o, b);
            }
        }

        LastUsePlan {
            position,
            use_pos,
            escapes,
            carried,
            view_pos,
        }
    }

    /// Reverse reachability from `root` over `f`'s in-edges — rule 2's escape
    /// walk — traversing only the value-RETAINING/-aliasing morphisms: `Pair`
    /// (the product holds the component), `Phi` (the result aliases either
    /// arm — plan §7's note), `Output` (identity), `LoopExit` (payload
    /// forwarding; dropped when `allow_loop_exit` is false — the
    /// carried-state exemption's walk), `LoopEnter` (the init aliases the
    /// merge's first instance), `Proj` (the projection aliases the product's
    /// field), `Call` (the call boundary may return an alias of its borrowed
    /// argument), and an array-typed `Index` (the sub-buffer alias).
    /// Everything else — arithmetic, comparisons, `Update`/`Map`/`Fold`/
    /// `Zip`/`Enumerate` (fresh buffers/results), `Print`, `TimeMs`, `Neg`,
    /// `Not` — consumes and produces anew: no escape flows through it.
    /// Worklist BFS, deterministic (in-edge insertion order), no HashMap (L2).
    pub(super) fn escape_reach(
        &self,
        f: FuncId,
        root: ObjectId,
        allow_loop_exit: bool,
    ) -> SecondaryMap<ObjectId, bool> {
        let mut seen: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
        let mut stack = vec![root];
        while let Some(o) = stack.pop() {
            if seen.get(o).copied().unwrap_or(false) {
                continue;
            }
            seen.insert(o, true);
            for &m in self.in_edges(o) {
                let morph = &self.morphisms[m];
                let retention = match morph.op {
                    Operation::Pair { .. }
                    | Operation::Phi
                    | Operation::Output
                    | Operation::LoopEnter
                    | Operation::Proj { .. }
                    | Operation::Call(_) => true,
                    Operation::LoopExit => allow_loop_exit,
                    // The sub-buffer alias: an array element shares the
                    // parent's flat buffer region; a scalar element shares
                    // nothing.
                    Operation::Index => {
                        matches!(self.objects[morph.target].ty, crate::ty::Ty::Array { .. })
                    }
                    _ => false,
                };
                if !retention {
                    continue;
                }
                let s = morph.source;
                if self.try_owner(s) == Some(f) {
                    stack.push(s);
                }
            }
        }
        seen
    }
}
