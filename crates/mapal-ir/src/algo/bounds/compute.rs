//! The bounds-proof computation: the `bounds_proof` query, its seeds and links.

use super::*;

impl CategoryIr {
    /// The bounds-proof query (see the module doc): which of `f`'s `Index`
    /// morphisms are provably in bounds where they run, each index's range,
    /// and the guard arms a proof needs gated. Stored on the graph.
    pub fn bounds_proof(&self, f: FuncId) -> BoundsProof {
        self.fact(f, |s| &s.bounds_proof, || self.bounds_proof_compute(f))
    }

    pub(super) fn bounds_proof_compute(&self, f: FuncId) -> BoundsProof {
        let mut out = BoundsProof::default();
        let Some(fd) = self.func(f) else {
            return out;
        };
        let topo = self.topo_order(f);
        let mut pos = SecondaryMap::new();
        for (i, &m) in topo.iter().enumerate() {
            pos.insert(m, i);
        }
        let mut ev = Eval {
            ir: self,
            f,
            topo,
            pos,
            seeds: RngMap::new(),
            merges: SecondaryMap::new(),
            exit_cond: SecondaryMap::new(),
            cond: true,
        };
        // --- seeds: constants, then the body params ---------------------------
        for o in self.func_objects(f) {
            if let Some(v) = const_int(self, o) {
                ev.seeds.insert(o, Rng::Int(v, v));
            }
        }
        self.seed_body(f, fd.input, &mut ev.seeds);

        // `cond-bounds` off (`FactArms`): plain intervals — no loop ranges, no
        // conditions (a `Phi` and an exit are unranged), the proof before it.
        let cond = self.fact_arms().cond_bounds;
        ev.cond = cond;

        // --- loops: carried slot ranges from the unlooped pass ----------------
        let flat = ev.forward(&Pins::new());
        if cond {
            ev.loop_ranges(&flat);
        }
        let base = ev.forward(&Pins::new());
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                let views = std::iter::once(merge).chain(
                    self.out_edges(merge)
                        .iter()
                        .filter(|&&e| matches!(self.morphisms[e].op, Operation::Proj { .. }))
                        .map(|&e| self.morphisms[e].target),
                );
                for v in views {
                    if let Some(r) = int(base.get(v).copied()) {
                        out.views.insert(v, r);
                    }
                }
            }
        }

        // --- contexts: the conditions holding where each morphism runs --------
        let links = if cond {
            self.bounds_links(f)
        } else {
            SecondaryMap::new()
        };
        let mut memo: SecondaryMap<MorphismId, Vec<CtxLink>> = SecondaryMap::new();
        let mut cache: Vec<(Vec<Cond>, RngMap)> = Vec::new();
        let mut at = |conds: Vec<Cond>| -> RngMap {
            if conds.is_empty() {
                return base.clone();
            }
            if let Some((_, r)) = cache.iter().find(|(c, _)| *c == conds) {
                return r.clone();
            }
            let r = ev.context(&base, &conds);
            cache.push((conds, r.clone()));
            r
        };
        for &m in &fd.morphisms {
            let morph = &self.morphisms[m];
            match morph.op {
                Operation::Index => {
                    let (Some(arr), Some(idx)) = (
                        self.pair_slot_source(morph.source, 0),
                        self.pair_slot_source(morph.source, 1),
                    ) else {
                        continue;
                    };
                    let Ty::Array { size, .. } = self.objects[arr].ty else {
                        continue;
                    };
                    let inside = |r: Option<(i128, i128)>| {
                        r.is_some_and(|(lo, hi)| lo >= 0 && hi < i128::from(size))
                    };
                    let chain = chain_of(&links, &mut memo, m, 0);
                    let all: Vec<Cond> = chain.iter().map(|&(c, p, _)| (c, p)).collect();
                    let r = int(at(all).get(idx).copied());
                    if let Some(r) = r {
                        out.index.insert(m, r);
                    }
                    if !inside(r) {
                        continue;
                    }
                    out.proven.insert(m, ());
                    if chain.iter().any(|l| l.2.is_some()) {
                        // Needed the arms? Re-check under the loop conditions only.
                        let loops: Vec<Cond> = chain
                            .iter()
                            .filter(|l| l.2.is_none())
                            .map(|&(c, p, _)| (c, p))
                            .collect();
                        if !inside(int(at(loops).get(idx).copied())) {
                            for &(_, _, e) in &chain {
                                if let Some(e) = e {
                                    out.arms.insert(e, ());
                                }
                            }
                        }
                    }
                }
                Operation::Widen if type_range(&self.objects[morph.source].ty).is_some() => {
                    let chain = chain_of(&links, &mut memo, m, 0);
                    let all: Vec<Cond> = chain.iter().map(|&(c, p, _)| (c, p)).collect();
                    if let Some(r) = int(at(all).get(morph.source).copied()) {
                        out.widen.insert(m, r);
                    }
                }
                Operation::Map { captures, .. } | Operation::Fold { captures, .. }
                    if captures > 0 =>
                {
                    let chain = chain_of(&links, &mut memo, m, 0);
                    let all: Vec<Cond> = chain.iter().map(|&(c, p, _)| (c, p)).collect();
                    let r = at(all);
                    let caps = (0..captures)
                        .map(|j| {
                            self.pair_slot_source(morph.source, j)
                                .and_then(|c| int(r.get(c).copied()))
                        })
                        .collect();
                    out.sites.insert(m, caps);
                }
                _ => {}
            }
        }
        out
    }

    /// Seed a body fn's params: the element param rides its site source's
    /// element range, each capture param the site owner's range of the
    /// captured value where the site runs. Every site of the body must agree
    /// (the hull; one unknown site leaves the param unranged).
    pub(super) fn seed_body(&self, f: FuncId, input: ObjectId, seeds: &mut RngMap) {
        let mut elem: Option<Option<Rng>> = None;
        let mut caps: Option<Vec<Option<(i128, i128)>>> = None;
        let mut shape: Option<(u32, bool)> = None;
        for (m, morph) in self.morphisms() {
            let (k, is_fold) = match morph.op {
                Operation::Map { body, captures } if body == f => (captures, false),
                Operation::Fold { body, captures } if body == f => (captures, true),
                _ => continue,
            };
            if shape.is_some_and(|s| s != (k, is_fold)) {
                return;
            }
            shape = Some((k, is_fold));
            let arr = if is_fold {
                self.pair_slot_source(morph.source, k + 1)
            } else if k == 0 {
                Some(morph.source)
            } else {
                self.pair_slot_source(morph.source, k)
            };
            let er = arr.and_then(|a| self.element_range(a));
            elem = Some(match elem {
                None => er,
                Some(prev) => match (prev, er) {
                    (Some(Rng::Int(a, b)), Some(Rng::Int(c, d))) => {
                        let (l, h) = hull((a, b), (c, d));
                        Some(Rng::Int(l, h))
                    }
                    (Some(x), Some(y)) if x == y => Some(x),
                    _ => None,
                },
            });
            let site: Vec<Option<(i128, i128)>> = match self.try_owner(morph.source) {
                Some(of) if of != f => self
                    .bounds_proof(of)
                    .sites
                    .get(m)
                    .cloned()
                    .unwrap_or_else(|| vec![None; k as usize]),
                _ => vec![None; k as usize],
            };
            caps = Some(match caps {
                None => site,
                Some(prev) => prev
                    .iter()
                    .zip(&site)
                    .map(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Some(hull(*a, *b)),
                        _ => None,
                    })
                    .collect(),
            });
        }
        let Some((k, is_fold)) = shape else { return };
        if let Some(Some(er)) = elem {
            if !is_fold && k == 0 {
                seeds.insert(input, er);
            } else {
                let p = if is_fold { k + 1 } else { k };
                for t in self.proj_targets(input, p) {
                    seeds.insert(t, er);
                }
            }
        }
        for (j, r) in caps.unwrap_or_default().into_iter().enumerate() {
            if let Some((l, h)) = r {
                for t in self.proj_targets(input, j as u32) {
                    seeds.insert(t, Rng::Int(l, h));
                }
            }
        }
    }

    pub(super) fn proj_targets(&self, o: ObjectId, k: u32) -> Vec<ObjectId> {
        self.out_edges(o)
            .iter()
            .filter(|&&m| self.morphisms[m].op == (Operation::Proj { index: k }))
            .map(|&m| self.morphisms[m].target)
            .collect()
    }

    /// Each of `f`'s morphisms' direct context: the morphisms whose context it
    /// inherits (its guard site's `Phi`, its loop's `LoopEnter`) and its own
    /// conditions (the arm's, the continue condition on an advance member).
    pub(super) fn bounds_links(
        &self,
        f: FuncId,
    ) -> SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)> {
        let mut links: SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)> =
            SecondaryMap::new();
        let mut add = |m: MorphismId, up: Option<MorphismId>, c: Option<CtxLink>| {
            if links.get(m).is_none() {
                links.insert(m, (Vec::new(), Vec::new()));
            }
            let e = links.get_mut(m).expect("inserted");
            if let Some(u) = up
                && u != m
                && !e.0.contains(&u)
            {
                e.0.push(u);
            }
            if let Some(c) = c
                && !e.1.contains(&c)
            {
                e.1.push(c);
            }
        };
        for scc in self.loop_structure(f) {
            for &merge in &scc.merges {
                let Some(plan) = self.loop_plan(f, merge) else {
                    continue;
                };
                let enter = self
                    .in_edges(merge)
                    .iter()
                    .copied()
                    .find(|&e| self.morphisms[e].op == Operation::LoopEnter);
                let cond = self.pair_slot_source(plan.back_route, 1);
                for &m in plan.decide_order.iter().chain(&plan.exits) {
                    add(m, enter, None);
                }
                for &m in &plan.advance_order {
                    add(m, enter, cond.map(|c| (c, true, None)));
                }
            }
        }
        for site in self.guard_structure(f) {
            for (arm, pole) in [(&site.on_true, true), (&site.on_false, false)] {
                for &m in &arm.own {
                    add(m, Some(site.phi), Some((site.cond, pole, Some(arm.edge))));
                }
            }
        }
        links
    }

    /// The range of an array's elements: iota ⇒ `[0, n)`; enumerate ⇒ `(i32,
    /// X)` elements over `n`; a literal array of int constants ⇒ `[min, max]`.
    /// `Proj`-of-pair indirection and body-input capture indirection (the
    /// array rides in as a captured value) are resolved first.
    pub(crate) fn element_range(&self, arr: ObjectId) -> Option<Rng> {
        let mut arr = arr;
        for _ in 0..16 {
            if let Some(next) = self.proj_of_pair(arr) {
                arr = next;
                continue;
            }
            if let Some(next) = self.capture_proj_of_input(arr) {
                arr = next;
                continue;
            }
            break;
        }
        let definer = self.in_edges(arr).first().copied()?;
        let dm = &self.morphisms[definer];
        let size = match &self.objects[arr].ty {
            Ty::Array { size, .. } => *size,
            _ => return None,
        };
        match dm.op {
            Operation::Iota => Some(Rng::Int(0, i128::from(size) - 1)),
            Operation::Enumerate => Some(Rng::EnumIdx(size)),
            Operation::Pair { .. } => {
                // A pack_array literal: every in-edge a Pair with a Constant
                // int source — the element range is [min, max] of the values.
                let mut lo = i128::MAX;
                let mut hi = i128::MIN;
                for &m in self.in_edges(arr) {
                    let pm = &self.morphisms[m];
                    if !matches!(pm.op, Operation::Pair { .. }) {
                        return None;
                    }
                    let v = const_int(self, pm.source)?;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                Some(Rng::Int(lo, hi))
            }
            _ => None,
        }
    }
}

/// A morphism's full context: its inherited contexts then its own links,
/// deduplicated (memoized; depth-guarded against a malformed cycle).
pub(super) fn chain_of(
    links: &SecondaryMap<MorphismId, (Vec<MorphismId>, Vec<CtxLink>)>,
    memo: &mut SecondaryMap<MorphismId, Vec<CtxLink>>,
    m: MorphismId,
    depth: u32,
) -> Vec<CtxLink> {
    if let Some(c) = memo.get(m) {
        return c.clone();
    }
    let Some((up, own)) = links.get(m) else {
        return Vec::new();
    };
    let mut out: Vec<CtxLink> = Vec::new();
    if depth < 32 {
        for &u in up {
            for l in chain_of(links, memo, u, depth + 1) {
                if !out.contains(&l) {
                    out.push(l);
                }
            }
        }
    }
    for &l in own {
        if !out.contains(&l) {
            out.push(l);
        }
    }
    memo.insert(m, out.clone());
    out
}
