//! `write_delta`: the range of a carried array each loop trip writes, and the pair / projection helpers it reads.

use super::*;

impl CategoryIr {
    /// The write delta of every loop-carried array in `f` (S53 `retain-delta`
    /// — the fact [`bounds_proof`](Self::bounds_proof) never had: it ranges
    /// `Index`, and `Update` stays trap-capable). Keyed by the carried array's
    /// merge view: the `LoopMerge` itself when the loop carries one value, its
    /// `Proj{k}` otherwise.
    ///
    /// [`Delta::Rows`] when the value crossing the back edge is reached from
    /// that view through `Update`s and piecewise maps only (`Proj`-of-`Pair`
    /// hops are transparent; a map whose [`Outside`] slot is the walked array
    /// writes its interval) whose cells are all `a·t + b_j` (+ the interval)
    /// — one int merge view `t`, one `a`, in `t`'s wrapping arithmetic — and
    /// together cover one contiguous range. [`Delta::Top`] for every other
    /// carried array. Canonical loops only.
    ///
    /// A map's interval holds every cell it writes in `[0, n)` (the body
    /// computes nothing else), and the trip's cells there are its interval's
    /// — so the union is sound for maps too; where the interval misses the
    /// array the claim is vacuous (see [`Delta::Rows`]).
    ///
    /// Why wrapping is enough: every index is `X + b_j` for the same wrapped
    /// `X = a·t`, and an executed `Update` traps unless its index is in
    /// `[0, n)` — so when the iteration completes, its cells are exactly
    /// `[start, start + stride)` with `start = X + b_min ≥ 0`.
    ///
    /// **Same consumers (S54).** A captures-0 `Map` over a `Rows` value `v`,
    /// `y[c] = m(v[c])`, run every iteration (in the loop body, not in a guard
    /// arm), differs from its own previous-iteration value only where `v`
    /// does: `y` gets the same `Δ`, keyed by `y` itself (`next: y`; it is no
    /// view), and so on down a chain of such maps. Never the other way: a
    /// back-edge value that is a map's output is rewritten whole (`Top`).
    pub fn write_delta(&self, f: FuncId) -> SecondaryMap<ObjectId, Delta> {
        self.fact(f, |s| &s.write_delta, || self.write_delta_compute(f))
    }

    pub(super) fn write_delta_compute(&self, f: FuncId) -> SecondaryMap<ObjectId, Delta> {
        let mut out = SecondaryMap::new();
        let gated: Vec<MorphismId> = self
            .guard_plan(f)
            .into_iter()
            .filter(GuardSite::gated)
            .flat_map(|s| s.on_true.own.into_iter().chain(s.on_false.own))
            .collect();
        for scc in self.loop_structure(f) {
            let &[merge] = scc.merges.as_slice() else {
                continue;
            };
            let Some(plan) = self.loop_plan(f, merge) else {
                continue;
            };
            let Some(next) = self.pair_slot_source(plan.back_route, 0) else {
                continue;
            };
            let merge_ty = &self.objects[merge].ty;
            let slots: Vec<(Option<u32>, ObjectId)> = match merge_ty {
                Ty::Tuple(ts) => (0..ts.len() as u32)
                    .filter_map(|k| Some((Some(k), self.pair_slot_source(next, k)?)))
                    .collect(),
                _ => vec![(None, next)],
            };
            for (k, nk) in slots {
                let comp = k.map_or(Some(merge_ty), |k| merge_ty.component_ty(k));
                if !matches!(comp, Some(Ty::Array { .. })) {
                    continue;
                }
                let is_view = |o: ObjectId| match k {
                    None => o == merge,
                    Some(k) => self.in_edges(o).iter().any(|&m| {
                        let mm = &self.morphisms[m];
                        mm.source == merge && mm.op == Operation::Proj { index: k }
                    }),
                };
                // Walk back from the back-route value to the view.
                let (mut o, mut idxs, mut view) = (nk, Vec::new(), None);
                for _ in 0..self.objects.len() {
                    if is_view(o) {
                        view = Some(o);
                        break;
                    }
                    let Some(&m) = self.in_edges(o).first() else {
                        break;
                    };
                    let mm = &self.morphisms[m];
                    let step = match mm.op {
                        Operation::Update => self.pair_slot_source(mm.source, 1).and_then(|i| {
                            idxs.push(Outside {
                                slot: 0,
                                d: 1,
                                a: 1,
                                e: Some(i),
                                b: 0,
                                step: 1,
                                len: 1,
                                w: 1,
                            });
                            self.pair_slot_source(mm.source, 0)
                        }),
                        Operation::Proj { .. } => self.proj_of_pair(o),
                        // A piecewise map: only its interval is not the walked array's.
                        Operation::Map { .. } => match self.map_law(m) {
                            Some(ElemSrc::Apply {
                                outside: Some(ou),
                                source,
                                ..
                            }) => {
                                idxs.push(ou);
                                self.pair_slot_source(source, ou.slot)
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some(prev) = step else { break };
                    o = prev;
                }
                let first_view = || match k {
                    None => Some(merge),
                    Some(k) => self.out_edges(merge).iter().find_map(|&m| {
                        let mm = &self.morphisms[m];
                        (mm.op == Operation::Proj { index: k }).then_some(mm.target)
                    }),
                };
                let Some(key) = view.or_else(first_view) else {
                    continue;
                };
                let delta = view
                    .and_then(|_| {
                        self.rows_of(&idxs, merge, nk)
                            .or_else(|| self.pieces_of(&idxs, merge, nk))
                    })
                    .unwrap_or(Delta::Top);
                out.insert(key, delta);
                let Delta::Rows { stride, start, .. } = delta else {
                    continue;
                };
                let mut frontier = vec![nk];
                while let Some(v) = frontier.pop() {
                    for &m in self.out_edges(v) {
                        let mm = &self.morphisms[m];
                        let in_body =
                            plan.advance_order.contains(&m) || plan.decide_order.contains(&m);
                        if matches!(mm.op, Operation::Map { captures: 0, .. })
                            && mm.source == v
                            && in_body
                            && !gated.contains(&m)
                            && !out.contains_key(mm.target)
                        {
                            let next = mm.target;
                            out.insert(
                                next,
                                Delta::Rows {
                                    stride,
                                    start,
                                    next,
                                },
                            );
                            frontier.push(next);
                        }
                    }
                }
            }
        }
        out
    }

    /// `piece-delta`: the [`Delta::Pieces`] of one iteration's writes, or
    /// `None` — the walk met exactly one write, a piecewise map whose pieces
    /// are `w > 1` cells (`(t ÷ d) % s == r`), `r` affine in the counter.
    pub(super) fn pieces_of(
        &self,
        idxs: &[Outside],
        merge: ObjectId,
        next: ObjectId,
    ) -> Option<Delta> {
        let [p] = idxs else { return None };
        if p.w <= 1 {
            return None;
        }
        let mut var = None;
        let (ae, be) = match p.e {
            Some(e) => self.affine_in(e, merge, &mut var, 0)?,
            None => return None,
        };
        let var = var?;
        let d = i128::from(p.d);
        let a = d.checked_mul(i128::from(p.a).checked_mul(ae)?)?;
        let b = d.checked_mul(i128::from(p.a).checked_mul(be)?.checked_add(p.b.into())?)?;
        Some(Delta::Pieces {
            w: p.w,
            step: p.step,
            start: Affine {
                var,
                a: i64::try_from(a).ok()?,
                b: i64::try_from(b).ok()?,
            },
            next,
        })
    }

    /// The [`Delta::Rows`] of one iteration's writes, or `None`: each the
    /// cells `[d·(a·e + b), … + len)` of an [`Outside`] piece (an `Update` at
    /// `i` is `d = a = len = 1, b = 0, e = i`), `e` affine in the counter.
    pub(super) fn rows_of(
        &self,
        idxs: &[Outside],
        merge: ObjectId,
        next: ObjectId,
    ) -> Option<Delta> {
        let (mut var, mut a, mut bs) = (None, None, Vec::new());
        for p in idxs {
            // (Every `step`-th cell is no range.)
            if p.step != 1 && p.len > 1 {
                return None;
            }
            let (ae, be) = match p.e {
                Some(e) => self.affine_in(e, merge, &mut var, 0)?,
                None => (0, 0),
            };
            let d = i128::from(p.d);
            let ai = d.checked_mul(i128::from(p.a).checked_mul(ae)?)?;
            let bi = d.checked_mul(i128::from(p.a).checked_mul(be)?.checked_add(p.b.into())?)?;
            if *a.get_or_insert(ai) != ai {
                return None;
            }
            bs.push((bi, i128::from(p.len)));
        }
        let var = var?;
        let bits = match self.objects[var].ty {
            Ty::Int { bits, .. } => u32::from(bits),
            _ => return None,
        };
        // Wrap into the counter's width: the program computes there.
        let wrap = |x: i128| -> i64 {
            let m = 1i128 << bits;
            let r = x.rem_euclid(m);
            (if r >= m / 2 { r - m } else { r }) as i64
        };
        let mut bs: Vec<(i128, i128)> = bs
            .into_iter()
            .map(|(b, l)| (i128::from(wrap(b)), l))
            .collect();
        bs.sort_unstable();
        // One contiguous union.
        let lo = bs.first()?.0;
        let mut end = lo;
        for &(b, l) in &bs {
            if b > end {
                return None;
            }
            end = end.max(b + l);
        }
        let lo = i64::try_from(lo).ok()?;
        let stride = u64::try_from(end - i128::from(lo)).ok()?;
        Some(Delta::Rows {
            stride,
            start: Affine {
                var,
                a: wrap(a?),
                b: lo,
            },
            next,
        })
    }

    /// `o` as `(a, b)` with `o = a·var + b`, `var` an int `Proj` of `merge`
    /// (the first one met; a second is `None`). Exact over `i128`; `Add`,
    /// `Sub`, and `Mul` by a constant only. Depth-guarded (J1).
    pub(super) fn affine_in(
        &self,
        o: ObjectId,
        merge: ObjectId,
        var: &mut Option<ObjectId>,
        depth: u32,
    ) -> Option<(i128, i128)> {
        if depth > 64 {
            return None;
        }
        let obj = &self.objects[o];
        if obj.kind == ObjectKind::Constant {
            return match obj.value {
                Some(Value::I32(x)) => Some((0, x.into())),
                Some(Value::I64(x)) => Some((0, x.into())),
                _ => None,
            };
        }
        let mm = &self.morphisms[*self.in_edges(o).first()?];
        match mm.op {
            Operation::Proj { .. } if mm.source == merge && matches!(obj.ty, Ty::Int { .. }) => {
                (*var.get_or_insert(o) == o).then_some((1, 0))
            }
            Operation::Proj { .. } => self.affine_in(self.proj_of_pair(o)?, merge, var, depth + 1),
            Operation::Add | Operation::Sub | Operation::Mul => {
                let x = self.pair_slot_source(mm.source, 0)?;
                let y = self.pair_slot_source(mm.source, 1)?;
                let (xa, xb) = self.affine_in(x, merge, var, depth + 1)?;
                let (ya, yb) = self.affine_in(y, merge, var, depth + 1)?;
                match mm.op {
                    Operation::Add => Some((xa.checked_add(ya)?, xb.checked_add(yb)?)),
                    Operation::Sub => Some((xa.checked_sub(ya)?, xb.checked_sub(yb)?)),
                    _ if xa == 0 => Some((ya.checked_mul(xb)?, yb.checked_mul(xb)?)),
                    _ if ya == 0 => Some((xa.checked_mul(yb)?, xb.checked_mul(yb)?)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// If `o` is a `Proj{k}` of a pair-assembled tuple, the component object
    /// (the `Pair{k}` feeder's source) — one indirection level.
    /// Every use of `o` assembles `into` — a `Pair` into it, or into a
    /// product whose every use is a `Proj` whose value does the same (a guard
    /// arm's routed captures). Depth-guarded (J1).
    pub(super) fn feeds_only(&self, o: ObjectId, into: ObjectId, depth: u32) -> bool {
        depth < 16
            && self.out_edges(o).iter().all(|&u| {
                let um = &self.morphisms[u];
                if !matches!(um.op, Operation::Pair { .. }) {
                    return false;
                }
                um.target == into
                    || self.out_edges(um.target).iter().all(|&v| {
                        let vm = &self.morphisms[v];
                        match vm.op {
                            Operation::Proj { .. } => self.feeds_only(vm.target, into, depth + 1),
                            Operation::Pair { .. } => vm.target == into,
                            _ => false,
                        }
                    })
            })
    }

    /// `o` through its `Proj`-of-`Pair` hops (a guard arm's routed values):
    /// the object the value was made as. Depth-guarded (J1).
    pub(crate) fn through_pairs(&self, mut o: ObjectId) -> ObjectId {
        for _ in 0..16 {
            match self.proj_of_pair(o) {
                Some(x) if x != o => o = x,
                _ => break,
            }
        }
        o
    }

    pub(super) fn proj_of_pair(&self, o: ObjectId) -> Option<ObjectId> {
        let m = self.in_edges(o).first().copied()?;
        let pm = &self.morphisms[m];
        let Operation::Proj { index } = pm.op else {
            return None;
        };
        self.pair_slot_source(pm.source, index)
    }

    /// If `o` is a `Proj{k}` of a body fn's input `Parameter` at a capture
    /// position, the matching feeder of THAT fn's site source (the value the
    /// enclosing fn actually passed) — one capture indirection level.
    pub(super) fn capture_proj_of_input(&self, o: ObjectId) -> Option<ObjectId> {
        let m = self.in_edges(o).first().copied()?;
        let pm = &self.morphisms[m];
        let Operation::Proj { index } = pm.op else {
            return None;
        };
        if self.objects[pm.source].kind != ObjectKind::Parameter {
            return None;
        }
        let of = self.try_owner(pm.source)?;
        for (_, morph) in self.morphisms() {
            let k = match morph.op {
                Operation::Map { body, captures } if body == of => captures,
                Operation::Fold { body, captures } if body == of => captures,
                _ => continue,
            };
            // Capture positions are 0..k (map: element at k; fold: acc at k,
            // element at k+1) — only capture Projs resolve here.
            if index < k {
                return self.pair_slot_source(morph.source, index);
            }
            return None;
        }
        None
    }

    /// The source object feeding slot `slot` of a product object (the
    /// `Pair{slot}` in-edge's source), if present.
    pub(crate) fn pair_slot_source(&self, product: ObjectId, slot: u32) -> Option<ObjectId> {
        self.in_edges(product).iter().find_map(|&m| {
            let pm = &self.morphisms[m];
            if matches!(pm.op, Operation::Pair { slot: s, .. } if s == slot) {
                Some(pm.source)
            } else {
                None
            }
        })
    }
}
