//! Guard facts (`guard_plan`, `guard_structure`) and the trap capability of paths and functions.

use super::*;

impl CategoryIr {
    /// Static element count for a Map/Zip/Enumerate/Iota/Fill/Fold site.
    pub(super) fn bulk_element_count(&self, m: MorphismId) -> u64 {
        let morph = &self.morphisms[m];
        let array = match morph.op {
            Operation::Fold { captures, .. } => self.pair_slot_source(morph.source, captures + 1),
            _ => Some(morph.target),
        }
        .expect("sealed bulk operation has its array operand");
        match self.objects[array].ty {
            Ty::Array { size, .. } => size,
            _ => unreachable!("sealed bulk operation has an array extent"),
        }
    }

    /// Direct trap capability within one function. Bounds proofs exempt Index
    /// only; Update remains trap-capable, as do integer Div/Mod.
    pub(super) fn path_local_trap_capable(&self, m: MorphismId, bounds: &BoundsProof) -> bool {
        let morph = &self.morphisms[m];
        match morph.op {
            Operation::Div | Operation::Mod => {
                matches!(self.objects[morph.target].ty, Ty::Int { .. })
            }
            Operation::Index => !bounds.proven(m),
            Operation::Update => true,
            _ => false,
        }
    }

    /// Per-function trap-capability fixpoint. Trap-free leaves flow backward
    /// through Call/Map/Fold references; unresolved cycles remain capable.
    pub(super) fn fn_trap_capabilities(&self) -> SecondaryMap<FuncId, bool> {
        self.facts
            .trap_capabilities
            .get_or_init(|| self.fn_trap_capabilities_compute())
            .clone()
    }

    pub(super) fn fn_trap_capabilities_compute(&self) -> SecondaryMap<FuncId, bool> {
        let mut capable = SecondaryMap::new();
        let mut remaining = SecondaryMap::new();
        let mut dependents: SecondaryMap<FuncId, Vec<FuncId>> = SecondaryMap::new();
        for (f, _) in self.funcs() {
            capable.insert(f, true);
            dependents.insert(f, Vec::new());
        }

        for (f, def) in self.funcs() {
            let bounds = self.bounds_proof(f);
            let local = def
                .morphisms
                .iter()
                .copied()
                .any(|m| self.path_local_trap_capable(m, &bounds));
            let mut refs = Vec::new();
            for &m in &def.morphisms {
                match self.morphisms[m].op {
                    Operation::Call(g)
                    | Operation::Map { body: g, .. }
                    | Operation::Fold { body: g, .. } => refs.push(g),
                    _ => {}
                }
            }
            for &g in &refs {
                if let Some(callers) = dependents.get_mut(g) {
                    callers.push(f);
                }
            }
            if !local {
                remaining.insert(f, refs.len());
            }
        }

        let mut worklist = Vec::new();
        for (f, _) in self.funcs() {
            if remaining.get(f) == Some(&0) {
                capable.insert(f, false);
                worklist.push(f);
            }
        }
        let mut cursor = 0;
        while cursor < worklist.len() {
            let trap_free = worklist[cursor];
            cursor += 1;
            for &caller in dependents
                .get(trap_free)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let ready = if let Some(left) = remaining.get_mut(caller) {
                    *left -= 1;
                    *left == 0
                } else {
                    false
                };
                if ready {
                    capable.insert(caller, false);
                    worklist.push(caller);
                }
            }
        }
        capable
    }

    /// Trap capability at a path-plan site, including referenced closures.
    pub(super) fn path_trap_capable(
        &self,
        m: MorphismId,
        bounds: &BoundsProof,
        capable: &SecondaryMap<FuncId, bool>,
    ) -> bool {
        if self.path_local_trap_capable(m, bounds) {
            return true;
        }
        match self.morphisms[m].op {
            Operation::Call(g)
            | Operation::Map { body: g, .. }
            | Operation::Fold { body: g, .. } => capable.get(g).copied().unwrap_or(true),
            // ADR-0045 D4: a read fails where its file is missing or of
            // another length.
            Operation::Read => true,
            _ => false,
        }
    }

    /// Per-Phi guard sites (plan-s39): the condition object and each arm's
    /// exclusive work. Sites are returned in topo order of their `Phi`.
    ///
    /// An arm's RAW exclusive work is what liveness loses when the arm's
    /// `Pair` edge is removed: objects backward-reachable from the roots
    /// (function output, token-bearing objects, loop machinery) through the
    /// full graph but not without that edge, plus the edge itself. Nested
    /// sites' work is then subtracted so own-lists partition (doc on
    /// [`GuardArm`]). A Phi whose triple is not the builder's three-`Pair`
    /// shape, or whose arm reaches loop machinery, is skipped — consumers keep
    /// strict semantics for it.
    pub fn guard_plan(&self, f: FuncId) -> Vec<GuardSite> {
        self.fact(f, |s| &s.guard_plan, || self.guard_plan_compute(f))
    }

    pub(super) fn guard_plan_compute(&self, f: FuncId) -> Vec<GuardSite> {
        let topo = self.topo_order(f);

        // No Phi ⇒ no sites. Skip the unit construction, bounds proof and
        // trap-capability fixpoint entirely — most functions are Phi-free,
        // and this query runs per pass per fixpoint round (S40 compile-time
        // A/B: the sweep regressed +16% before this gate).
        if !topo.iter().any(|&m| self.morphisms[m].op == Operation::Phi) {
            return Vec::new();
        }

        let mut sites = self.guard_structure(f);
        if sites.is_empty() {
            return sites;
        }
        let (mut units, unit_of_machinery, _) = self.guard_units(f, &topo);
        let bounds = self.bounds_proof(f);
        let trap_capable = self.fn_trap_capabilities();

        // Unit trap flags: any member trap-capable (transitively, via bodies
        // and calls). Needed because unit internals never enter an own-list —
        // the LoopEnter handle must carry the whole unit's flag.
        for unit in &mut units {
            unit.can_trap = unit
                .members
                .iter()
                .any(|(m, ())| self.path_trap_capable(m, &bounds, &trap_capable));
        }
        for site in &mut sites {
            for arm in [&mut site.on_true, &mut site.on_false] {
                // Flags describe the FINAL list, not the raw walk. A LoopEnter
                // handle carries its whole unit's trap flag, and a loop is
                // heavy by definition (unbounded trip count).
                arm.can_trap = arm.own.iter().any(|&m| {
                    self.path_trap_capable(m, &bounds, &trap_capable)
                        || unit_of_machinery.get(m).is_some_and(|&u| units[u].can_trap)
                });
                arm.heavy = arm.own.iter().any(|&m| {
                    matches!(
                        self.morphisms[m].op,
                        Operation::Map { .. }
                            | Operation::Fold { .. }
                            | Operation::Zip
                            | Operation::Enumerate
                            | Operation::Iota
                            | Operation::Fill
                            | Operation::Call(_)
                            | Operation::LoopEnter
                    )
                });
                // An `Index` the proof holds in bounds only under this arm's
                // condition is safe only while the arm runs alone: gate it.
                arm.cond_reads = bounds.arm_used(arm.edge);
            }
        }

        // TRANSITIVE flags. Subtraction moved each nested site's work out of
        // the enclosing arm's list, so direct flags under-report: `calc`'s
        // 5-arm match right-folds to nested Phis with `a % b` innermost, and
        // the outer arms looked trap-free because the `Div`/`Mod` had been
        // attributed to the innermost site. Ungated, the outer arms then ran
        // the whole nested chain and `calc(0, 20, 0)` still trapped. An arm
        // that gates a nested site inherits that site's flags. Sites are in
        // topo order of their `Phi`, so nested sites are already final when an
        // enclosing one is reached — one forward pass suffices.
        for i in 0..sites.len() {
            let inherited: Vec<(bool, bool, bool)> = (0..i)
                .map(|j| {
                    (
                        sites[i].on_true.own.contains(&sites[j].phi),
                        sites[j].on_true.can_trap || sites[j].on_false.can_trap,
                        sites[j].on_true.heavy || sites[j].on_false.heavy,
                    )
                })
                .collect();
            let inherited_f: Vec<(bool, bool, bool)> = (0..i)
                .map(|j| {
                    (
                        sites[i].on_false.own.contains(&sites[j].phi),
                        sites[j].on_true.can_trap || sites[j].on_false.can_trap,
                        sites[j].on_true.heavy || sites[j].on_false.heavy,
                    )
                })
                .collect();
            let site = &mut sites[i];
            for (arm, src) in [
                (&mut site.on_true, &inherited),
                (&mut site.on_false, &inherited_f),
            ] {
                for &(nested_here, trap, heavy) in src {
                    if nested_here {
                        arm.can_trap |= trap;
                        arm.heavy |= heavy;
                    }
                }
            }
        }
        sites
    }

    /// The structural half of [`Self::guard_plan`]: each Phi-position guard
    /// site with its arms' exclusive work (own-lists after hierarchy
    /// subtraction and re-close), every flag `false`. A stored fact of its
    /// own because [`Self::bounds_proof`] reads it (a guard's condition holds
    /// on its arm's own work) while `guard_plan`'s flags read the proof.
    pub(crate) fn guard_structure(&self, f: FuncId) -> Vec<GuardSite> {
        self.fact(
            f,
            |s| &s.guard_structure,
            || self.guard_structure_compute(f),
        )
    }

    pub(super) fn guard_structure_compute(&self, f: FuncId) -> Vec<GuardSite> {
        let topo = self.topo_order(f);
        if !topo.iter().any(|&m| self.morphisms[m].op == Operation::Phi) {
            return Vec::new();
        }
        let (units, unit_of_machinery, unit_of_member) = self.guard_units(f, &topo);

        // Raw sites first; hierarchy subtraction below.
        let mut sites: Vec<GuardSite> = Vec::new();
        for &m in &topo {
            if self.morphisms[m].op != Operation::Phi {
                continue;
            }
            let triple = self.morphisms[m].source;
            // The builder's phi() mints a fresh triple with exactly three Pair
            // edges, slots 0/1/2 (builder.rs). Anything else is hand-built —
            // skip, keeping strict semantics there.
            let edges = self.in_edges(triple);
            if edges.len() != 3 {
                continue;
            }
            let mut feeder: [Option<MorphismId>; 3] = [None; 3];
            for &e in edges {
                if let Operation::Pair { slot, arity: 3 } = self.morphisms[e].op
                    && slot < 3
                {
                    feeder[slot as usize] = Some(e);
                }
            }
            let (Some(te), Some(fe), Some(ce)) = (feeder[0], feeder[1], feeder[2]) else {
                continue;
            };
            let [on_true, on_false] =
                [te, fe].map(|e| self.guard_arm(&topo, &units, &unit_of_machinery, e));
            sites.push(GuardSite {
                phi: m,
                cond: self.morphisms[ce].source,
                on_true,
                on_false,
            });
        }

        // Direct ownership: strip from each arm every morphism owned by a
        // nested site (one whose phi is in this arm's raw list). The nested
        // phi itself stays — it is the gate that fires the nested work.
        let raw: Vec<(SecondaryMap<MorphismId, ()>, MorphismId)> = sites
            .iter()
            .map(|s| {
                let mut set = SecondaryMap::new();
                for &m in s.on_true.own.iter().chain(s.on_false.own.iter()) {
                    set.insert(m, ());
                }
                (set, s.phi)
            })
            .collect();
        for i in 0..sites.len() {
            let site = &mut sites[i];
            for arm in [&mut site.on_true, &mut site.on_false] {
                let arm_has: SecondaryMap<MorphismId, ()> = {
                    let mut s = SecondaryMap::new();
                    for &m in &arm.own {
                        s.insert(m, ());
                    }
                    s
                };
                // Review find [2]: a nested site whose Phi sits INSIDE an
                // owned unit (an in-body guard) is gated by this arm through
                // the unit's handle, not through its Phi appearing in `own` —
                // but its exclusive work can be loop-INVARIANT (fed by
                // constants), hence not a unit member, and without this it
                // stayed in the enclosing arm's list too: owned twice, fired
                // twice.
                let in_owned_unit = |phi: MorphismId| {
                    unit_of_member
                        .get(phi)
                        .is_some_and(|&u| units[u].enters.iter().any(|h| arm_has.contains_key(*h)))
                };
                let mut nested: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
                for (j, (set, phi)) in raw.iter().enumerate() {
                    if j != i && (arm_has.contains_key(*phi) || in_owned_unit(*phi)) {
                        for (m, ()) in set.iter() {
                            nested.insert(m, ());
                        }
                    }
                }
                arm.own.retain(|m| !nested.contains_key(*m));

                // RE-CLOSE. Subtraction can break consumer closure: a morphism
                // may have joined this arm only because a sibling guard's edge
                // was also owned at walk time, and that edge has just been
                // removed. Left in, its target would be gated while a consumer
                // outside the arm still reads it. Drop to a fixpoint; anything
                // dropped simply runs unconditionally, which is always safe.
                // plan-s40: a consumer inside a loop unit counts as covered
                // when the unit's LoopEnter handle is in the arm — the driver
                // fires internals, so the handle stands for all of them.
                loop {
                    let have: SecondaryMap<MorphismId, ()> = {
                        let mut s = SecondaryMap::new();
                        for &m in &arm.own {
                            s.insert(m, ());
                        }
                        s
                    };
                    let covered = |c: MorphismId| {
                        have.contains_key(c)
                            || unit_of_member.get(c).is_some_and(|&u| {
                                units[u].enters.iter().any(|h| have.contains_key(*h))
                            })
                    };
                    let before = arm.own.len();
                    let edge = arm.edge;
                    arm.own.retain(|&m| {
                        if m == edge {
                            return true;
                        }
                        if let Some(&u) = unit_of_machinery.get(m) {
                            // Review find [0]: the handle stands for the whole
                            // unit, so it is re-tested with the JOIN predicate —
                            // the unit's outputs must still be consumed inside
                            // the arm. Testing only out_edges(merge) is vacuous
                            // (every consumer is a member), which let a gated
                            // loop survive while the subtraction cascade
                            // dropped its payload consumers: the un-gated
                            // survivors then read an object the gated loop
                            // never wrote. Sink members refuse here too.
                            return units[u].members.iter().all(|(mm, ())| {
                                let outs = self.out_edges(self.morphisms[mm].target);
                                !outs.is_empty()
                                    && outs
                                        .iter()
                                        .all(|&c| units[u].members.contains_key(c) || covered(c))
                            });
                        }
                        self.out_edges(self.morphisms[m].target)
                            .iter()
                            .all(|&c| covered(c))
                    });
                    if arm.own.len() == before {
                        break;
                    }
                }
            }
        }
        sites
    }

    /// Loop UNITS (plan-s40) of `f` for guard ownership: per SCC, the whole
    /// sequential region the flat walk hands the driver, by member and by
    /// machinery morphism. `can_trap` is left `false` (flags are
    /// [`Self::guard_plan`]'s).
    #[allow(clippy::type_complexity)]
    pub(super) fn guard_units(
        &self,
        f: FuncId,
        topo: &[MorphismId],
    ) -> (
        Vec<LoopUnit>,
        SecondaryMap<MorphismId, usize>,
        SecondaryMap<MorphismId, usize>,
    ) {
        // plan-s40: loop UNITS — per SCC, the whole sequential region the flat
        // walk hands the driver (SCC-incident morphisms, machinery, and the
        // loop_plan cones; the same construction as path_plan's loop_members).
        // An arm owns a loop as a unit or not at all: per-morphism consumer
        // closure provably cannot complete a cycle, so the only machinery it
        // can reach alone is the exit boundary, and gating that fragment
        // starves the driver ("route object built before read", S39 §4a).
        let mut units: Vec<LoopUnit> = Vec::new();
        let mut unit_of_machinery: SecondaryMap<MorphismId, usize> = SecondaryMap::new();
        let mut unit_of_member: SecondaryMap<MorphismId, usize> = SecondaryMap::new();
        for scc in self.loop_structure(f) {
            let mut objects: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for &o in &scc.objects {
                objects.insert(o, ());
            }
            let mut members: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            for &m in topo {
                let morph = &self.morphisms[m];
                if objects.contains_key(morph.source) || objects.contains_key(morph.target) {
                    members.insert(m, ());
                }
            }
            let mut canonical = scc.merges.len() == 1 && self.loop_plan(f, scc.merges[0]).is_some();
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
            let idx = units.len();
            let mut enters = Vec::new();
            for (m, ()) in members.iter() {
                if matches!(
                    self.morphisms[m].op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                ) {
                    if let Some(&prev) = unit_of_machinery.get(m) {
                        // Machinery shared between two SCC regions: neither
                        // unit is independently drivable — never join either.
                        units[prev].canonical = false;
                        canonical = false;
                    }
                    unit_of_machinery.insert(m, idx);
                    if matches!(self.morphisms[m].op, Operation::LoopEnter) {
                        enters.push(m);
                    }
                }
                unit_of_member.insert(m, idx);
            }
            units.push(LoopUnit {
                members,
                enters,
                canonical,
                can_trap: false,
            });
        }

        (units, unit_of_machinery, unit_of_member)
    }

    /// One arm's exclusive work.
    ///
    /// **Consumer closure, not liveness.** A morphism is arm-owned iff EVERY
    /// consumer of its target is arm-owned; the boundary `Pair` edge seeds the
    /// set. Liveness is the wrong test here and was a real defect: nothing in
    /// this pipeline deletes dead code before execution — the interpreter walks
    /// every morphism in topo order — so a *dead* consumer still reads its
    /// operand. Under the liveness rule a `Proj` feeding both the arm and a
    /// dead `Neg` looked exclusive (removing the arm edge left only dead uses),
    /// got gated, and the dead `Neg` then read an object the unchosen arm never
    /// wrote. Requiring every consumer to be owned is what makes gating an
    /// object safe, whether or not the other consumers matter.
    ///
    /// **Loops join as units (plan-s40).** Machinery is never a per-morphism
    /// candidate; reaching a `LoopEnter`/`LoopBack`/`LoopExit` offers its whole
    /// [`LoopUnit`], which joins iff every consumer outside the unit of every
    /// member's target is already owned. In the own-list the unit is its
    /// `LoopEnter` handle(s) alone — internals are the driver's, exactly the
    /// set the flat walk already skips. This replaces the v1 refusal, which
    /// keyed the site's SEMANTICS on graph shape (SCC presence) and therefore
    /// could not commute with `LiftLoops` (S39 §4a).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn guard_arm(
        &self,
        topo: &[MorphismId],
        units: &[LoopUnit],
        unit_of_machinery: &SecondaryMap<MorphismId, usize>,
        edge: MorphismId,
    ) -> GuardArm {
        let value = self.morphisms[edge].source;

        let mut owned_set: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut owned_via_unit: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        let mut joined = vec![false; units.len()];
        owned_set.insert(edge, ());
        // Backward worklist: after owning `m`, its source's producers become
        // candidates. A candidate joins iff all of its target's consumers are
        // already owned (an object with no consumer is a sink — never owned:
        // a Return/Output-class sink is observable, and a dead `Temporary`
        // sink must NOT be gated — a dead cone can read BOTH arms' values,
        // and gating it into one arm would read the other's un-fired value.
        // DCE preserves such cones instead; see graph_rewrites.rs).
        let mut queue: Vec<MorphismId> = self.in_edges(value).to_vec();
        while let Some(cand) = queue.pop() {
            if owned_set.contains_key(cand) {
                continue;
            }
            if let Some(&u) = unit_of_machinery.get(cand) {
                if joined[u] || !units[u].canonical {
                    continue;
                }
                let unit = &units[u];
                // A site INSIDE the loop (its boundary edge is a unit member)
                // can never own its own enclosing unit — from inside, every
                // "external" consumer is vacuously satisfied. The driver
                // already runs that site; the arm gates only body work.
                if unit.members.contains_key(edge) {
                    continue;
                }
                // Sink targets refuse the join exactly as they refuse
                // per-morphism ownership (review find [1]): `all()` over an
                // empty consumer list is vacuously true, and a unit member
                // writing a consumerless object — the function's Return, an
                // Output, or dead code DCE must preserve — is work the arm
                // must not gate.
                let joinable = unit.members.iter().all(|(m, ())| {
                    let outs = self.out_edges(self.morphisms[m].target);
                    !outs.is_empty()
                        && outs
                            .iter()
                            .all(|c| unit.members.contains_key(*c) || owned_set.contains_key(*c))
                });
                if !joinable {
                    continue;
                }
                joined[u] = true;
                for (m, ()) in unit.members.iter() {
                    owned_set.insert(m, ());
                    owned_via_unit.insert(m, ());
                }
                for (m, ()) in unit.members.iter() {
                    // The unit's external inputs join the walk; re-offering
                    // producers of member targets can complete other
                    // candidates' consumer sets.
                    queue.extend_from_slice(self.in_edges(self.morphisms[m].source));
                    queue.extend_from_slice(self.in_edges(self.morphisms[m].target));
                }
                continue;
            }
            let t = self.morphisms[cand].target;
            let outs = self.out_edges(t);
            if outs.is_empty() || !outs.iter().all(|o| owned_set.contains_key(*o)) {
                continue;
            }
            owned_set.insert(cand, ());
            queue.extend_from_slice(self.in_edges(self.morphisms[cand].source));
            // Owning `cand` can complete another candidate's consumer set, so
            // re-offer this object's other producers too.
            queue.extend_from_slice(self.in_edges(t));
        }

        let mut own: Vec<MorphismId> = Vec::new();
        for &m in topo {
            let morph = &self.morphisms[m];
            if !owned_set.contains_key(m) {
                continue;
            }
            if owned_via_unit.contains_key(m) {
                // The LoopEnter handle stands for the unit in `own`; internals
                // stay out, so subtraction, re-close and the consumers' gated
                // skip-sets never see them — the driver fires them.
                if matches!(morph.op, Operation::LoopEnter) {
                    own.push(m);
                }
                continue;
            }
            own.push(m);
        }
        GuardArm {
            value,
            edge,
            own,
            can_trap: false,
            heavy: false,
            cond_reads: false,
        }
    }
}
