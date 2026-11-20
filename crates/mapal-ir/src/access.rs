//! **`access_plan`: which bytes each morphism reads and writes, and the order
//! that follows.** The compile-time dependency fact device work is placed
//! from: a morphism's reads are the objects whose bytes its evaluation loads —
//! its source, or, through every `Pair` that assembles it (and every `Proj` of
//! such a product), each slot's source; its writes are its target, except for
//! the structural morphisms that make no bytes (`Pair`, `Proj`, the loop
//! routing). Objects that may share one buffer form a storage class: a loop
//! state slot's merge views, the next value crossing the back edge into it
//! and the exit object that is that view (a carried buffer may be written in
//! place), and an `Update`'s target with the array it updates. Two accesses
//! conflict where a later one reads bytes an earlier one writes (RAW), writes
//! bytes an earlier one reads (WAR) or writes bytes an earlier one writes
//! (WAW). Each canonical loop's trip — its advance cone, in order — is stored
//! with its edges; [`AccessPlan::edges`] lifts the fact onto any sequence of
//! grouped accesses (a backend's kernels) with an extra byte-sharing relation
//! (a backend's arena).

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation};
use crate::ty::Ty;

/// The conflicts between an earlier and a later access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hazard {
    /// The later reads bytes the earlier writes.
    pub raw: bool,
    /// The later writes bytes the earlier reads.
    pub war: bool,
    /// Both write the same bytes.
    pub waw: bool,
}

impl Hazard {
    /// Any conflict: the later must run after the earlier.
    pub fn any(self) -> bool {
        self.raw || self.war || self.waw
    }
}

/// One function's access fact ([`CategoryIr::access_plan`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AccessPlan {
    reads: SecondaryMap<MorphismId, Vec<ObjectId>>,
    writes: SecondaryMap<MorphismId, Vec<ObjectId>>,
    /// Storage class representative; absent = the object alone.
    class: SecondaryMap<ObjectId, ObjectId>,
    /// Per canonical loop: `(merge, advance order, edges)`, `edges[j]` the
    /// earlier positions `i < j` the `j`-th morphism conflicts with.
    trips: Vec<(ObjectId, Vec<MorphismId>, Vec<Vec<(usize, Hazard)>>)>,
}

impl AccessPlan {
    /// The objects whose bytes `m` reads.
    pub fn reads(&self, m: MorphismId) -> &[ObjectId] {
        self.reads.get(m).map_or(&[], Vec::as_slice)
    }

    /// The objects whose bytes `m` writes (none for a structural morphism).
    pub fn writes(&self, m: MorphismId) -> &[ObjectId] {
        self.writes.get(m).map_or(&[], Vec::as_slice)
    }

    /// The storage class of `o`: one representative per set of objects that
    /// may share a buffer.
    pub fn class(&self, o: ObjectId) -> ObjectId {
        self.class.get(o).copied().unwrap_or(o)
    }

    /// May `x` and `y` share bytes?
    pub fn shares(&self, x: ObjectId, y: ObjectId) -> bool {
        self.class(x) == self.class(y)
    }

    /// The conflicts of a later access `(reads, writes)` with an earlier
    /// one, bytes shared by class or by `extra` (a backend's placement).
    pub fn hazard(
        &self,
        before: (&[ObjectId], &[ObjectId]),
        after: (&[ObjectId], &[ObjectId]),
        extra: &dyn Fn(ObjectId, ObjectId) -> bool,
    ) -> Hazard {
        let hit = |xs: &[ObjectId], ys: &[ObjectId]| {
            xs.iter()
                .any(|&x| ys.iter().any(|&y| self.shares(x, y) || extra(x, y)))
        };
        Hazard {
            raw: hit(before.1, after.0),
            war: hit(before.0, after.1),
            waw: hit(before.1, after.1),
        }
    }

    /// Edges of a sequence of grouped accesses (each `(reads, writes)`, in
    /// issue order): `out[j]` = every earlier `i` the `j`-th conflicts with.
    pub fn edges(
        &self,
        ops: &[(Vec<ObjectId>, Vec<ObjectId>)],
        extra: &dyn Fn(ObjectId, ObjectId) -> bool,
    ) -> Vec<Vec<(usize, Hazard)>> {
        (0..ops.len())
            .map(|j| {
                (0..j)
                    .filter_map(|i| {
                        let h = self.hazard((&ops[i].0, &ops[i].1), (&ops[j].0, &ops[j].1), extra);
                        h.any().then_some((i, h))
                    })
                    .collect()
            })
            .collect()
    }

    /// The trip of the loop at `merge`: its advance order and each
    /// morphism's edges to the earlier ones (by class alone).
    pub fn trip(&self, merge: ObjectId) -> Option<(&[MorphismId], &[Vec<(usize, Hazard)>])> {
        self.trips
            .iter()
            .find(|t| t.0 == merge)
            .map(|t| (t.1.as_slice(), t.2.as_slice()))
    }
}

impl CategoryIr {
    /// **The access fact of `f`** (module doc): stored on the graph, built
    /// from the morphisms' sources and targets and `loop_plan`.
    pub fn access_plan(&self, f: FuncId) -> AccessPlan {
        self.fact(f, |s| &s.access_plan, || self.access_plan_compute(f))
    }

    /// The objects whose bytes a read of `o` loads: through the `Pair`s that
    /// assemble a product and the `Proj`s of one, each slot's source.
    fn byte_sources(&self, o: ObjectId, out: &mut Vec<ObjectId>, seen: &mut Vec<ObjectId>) {
        if seen.contains(&o) {
            return;
        }
        seen.push(o);
        let ins = self.in_edges(o);
        let pairs: Vec<MorphismId> = ins
            .iter()
            .copied()
            .filter(|&m| matches!(self.morphisms[m].op, Operation::Pair { .. }))
            .collect();
        if !pairs.is_empty() && pairs.len() == ins.len() {
            for m in pairs {
                self.byte_sources(self.morphisms[m].source, out, seen);
            }
            return;
        }
        if let [m] = ins
            && let Operation::Proj { index } = self.morphisms[*m].op
        {
            let p = self.morphisms[*m].source;
            if let Some(s) = self.pair_slot(p, index) {
                return self.byte_sources(s, out, seen);
            }
        }
        if !out.contains(&o) {
            out.push(o);
        }
    }

    /// The source of slot `k` of the product `p` (its `Pair` edge).
    fn pair_slot(&self, p: ObjectId, k: u32) -> Option<ObjectId> {
        self.in_edges(p).iter().find_map(|&m| {
            let mo = &self.morphisms[m];
            matches!(mo.op, Operation::Pair { slot, .. } if slot == k).then_some(mo.source)
        })
    }

    fn access_plan_compute(&self, f: FuncId) -> AccessPlan {
        let mut plan = AccessPlan::default();
        if self.func(f).is_none() {
            return plan;
        }
        let objects = self.func_objects(f);
        for &o in &objects {
            for &m in self.in_edges(o) {
                let mo = &self.morphisms[m];
                let structural = matches!(
                    mo.op,
                    Operation::Pair { .. }
                        | Operation::Proj { .. }
                        | Operation::LoopEnter
                        | Operation::LoopBack
                        | Operation::LoopExit
                );
                let mut reads = Vec::new();
                self.byte_sources(mo.source, &mut reads, &mut Vec::new());
                plan.reads.insert(m, reads);
                plan.writes
                    .insert(m, if structural { Vec::new() } else { vec![o] });
            }
        }
        // Storage classes: union-find over the objects.
        let mut parent: SecondaryMap<ObjectId, ObjectId> = SecondaryMap::new();
        fn find(p: &SecondaryMap<ObjectId, ObjectId>, mut x: ObjectId) -> ObjectId {
            while let Some(&y) = p.get(x) {
                if y == x {
                    break;
                }
                x = y;
            }
            x
        }
        fn union(p: &mut SecondaryMap<ObjectId, ObjectId>, a: ObjectId, b: ObjectId) {
            let (ra, rb) = (find(p, a), find(p, b));
            if ra != rb {
                p.insert(rb, ra);
                if !p.contains_key(ra) {
                    p.insert(ra, ra);
                }
            }
        }
        for &o in &objects {
            for &m in self.in_edges(o) {
                let mo = &self.morphisms[m];
                if mo.op == Operation::Update
                    && let Some(a) = self.pair_slot(mo.source, 0)
                {
                    union(&mut parent, o, a);
                }
            }
        }
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                let Some(lp) = self.loop_plan(f, merge) else {
                    continue;
                };
                let Some(state) = self.pair_slot(lp.back_route, 0) else {
                    continue;
                };
                let exit = lp.exits.first().map(|&x| self.morphisms[x].target);
                let payload = self.pair_slot(lp.exit_route, 0);
                let slots: Vec<Option<u32>> = match self.object(merge).map(|x| &x.ty) {
                    Some(Ty::Tuple(parts)) => (0..parts.len() as u32).map(Some).collect(),
                    _ => vec![None],
                };
                for k in slots {
                    let (views, next): (Vec<ObjectId>, Option<ObjectId>) = match k {
                        Some(k) => (
                            self.out_edges(merge)
                                .iter()
                                .filter(|&&m| self.morphisms[m].op == Operation::Proj { index: k })
                                .map(|&m| self.morphisms[m].target)
                                .collect(),
                            self.pair_slot(state, k),
                        ),
                        None => (vec![merge], Some(state)),
                    };
                    let Some(&first) = views.first() else {
                        continue;
                    };
                    for &v in &views[1..] {
                        union(&mut parent, first, v);
                    }
                    if let Some(n) = next {
                        union(&mut parent, first, n);
                    }
                    if let (Some(x), Some(p)) = (exit, payload)
                        && views.contains(&p)
                    {
                        union(&mut parent, first, x);
                    }
                }
            }
        }
        for &o in &objects {
            let r = find(&parent, o);
            if r != o {
                plan.class.insert(o, r);
            }
        }
        // Each canonical loop's trip, in advance order.
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                let Some(lp) = self.loop_plan(f, merge) else {
                    continue;
                };
                let ops: Vec<(Vec<ObjectId>, Vec<ObjectId>)> = lp
                    .advance_order
                    .iter()
                    .map(|&m| (plan.reads(m).to_vec(), plan.writes(m).to_vec()))
                    .collect();
                let edges = plan.edges(&ops, &|_, _| false);
                plan.trips.push((merge, lp.advance_order.clone(), edges));
            }
        }
        plan
    }
}
