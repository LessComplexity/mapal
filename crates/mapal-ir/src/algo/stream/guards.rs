//! Stream guards: a group's guarded producer, mask form and guard boxes.

use super::*;

impl CategoryIr {
    /// flash-attention F5/F6: the group's guarded producer, its mask's form
    /// and whether a wholly masked window may be skipped ([`StreamGuard`]).
    /// Off under `FactArms::guard_box`.
    pub(super) fn stream_guard(
        &self,
        g: &UnitGroup,
        roles: &[StreamRole],
        axis: u64,
    ) -> Option<StreamGuard> {
        if !self.fact_arms().guard_box {
            return None;
        }
        // The members the pass covers (`StreamGroup::roles`).
        let ms = &g.members[..roles.len()];
        let producers: Vec<usize> = (0..ms.len())
            .filter(|&k| roles[k] == StreamRole::Produce)
            .collect();
        let (k, form) = producers.iter().find_map(|&k| {
            let UnitKind::Tile(t) = &ms[k].kind else {
                return None;
            };
            self.guard_box(ms[k].m, t, axis).map(|f| (k, f))
        })?;
        let f = self.try_owner(ms[k].out)?;
        let fin = self.finite_plan(f);
        let why = if producers.len() != 1 {
            "producers"
        } else if (0..ms.len()).any(|j| ms[j].w == axis && ms[j].stored) {
            "stored"
        } else if (0..ms.len()).any(|j| match (&roles[j], &ms[j].kind) {
            (StreamRole::Fold { .. }, UnitKind::Fold(s)) => !matches!(
                s.op,
                RowFoldOp::Select { .. }
                    | RowFoldOp::Arith {
                        op: Operation::Add,
                        ..
                    }
            ),
            (StreamRole::Cell { .. }, _) => !self.cell_index_addresses(ms[j].m),
            _ => false,
        }) {
            "term"
        } else {
            ""
        };
        // A consumer's column operand not proven finite: checked at run time.
        let check: Vec<usize> = (0..ms.len())
            .filter(|&j| match (&roles[j], &ms[j].kind) {
                (StreamRole::Consume { .. }, UnitKind::Tile(t)) => self
                    .pair_slot_source(self.morphisms[ms[j].m].source, t.b.slot)
                    .is_none_or(|b| !fin.finite(b)),
                _ => false,
            })
            .collect();
        Some(StreamGuard {
            producer: k,
            form,
            skip: why.is_empty(),
            why,
            check,
        })
    }

    /// F5: tile site `m`'s epilogue mask as a [`GuardBox`] over its cells
    /// `e = row·w + lane` (`w` its lanes): the guard whose arm holds the
    /// fold makes the body's output, its other arm a float constant, its
    /// condition a comparison of two digit sums of the map index alone,
    /// each digit a digit of the row or of the lane.
    pub(super) fn guard_box(&self, m: MorphismId, t: &crate::TileSite, w: u64) -> Option<GuardBox> {
        let post = t.post.as_ref()?;
        // mask-frontend: the sides are over the output's own rows — the
        // group's ([`crate::unit::out_rows`]: a row-wise mixed-radix output).
        if !post.guarded || crate::unit::out_rows(t, t.rows).is_err() || t.c != w {
            return None;
        }
        let Operation::Map { body, captures } = self.morphisms.get(m)?.op else {
            return None;
        };
        let bd = self.func(body)?;
        let gs = self
            .guard_structure(body)
            .into_iter()
            .find(|g| g.on_true.own.contains(&post.fold) || g.on_false.own.contains(&post.fold))?;
        if self.morphisms.get(gs.phi)?.target != bd.output {
            return None;
        }
        let fold_when = gs.on_true.own.contains(&post.fold);
        let other = if fold_when { &gs.on_false } else { &gs.on_true };
        let ov = self.objects.get(self.through_pairs(other.value))?;
        let masked = match (&ov.kind, &ov.value) {
            (ObjectKind::Constant, Some(Value::F32(x))) => f64::from(*x),
            (ObjectKind::Constant, Some(Value::F64(x))) => *x,
            _ => return None,
        };
        let [cm] = self.in_edges(gs.cond) else {
            return None;
        };
        let cmo = self.morphisms.get(*cm)?;
        if !matches!(
            cmo.op,
            Operation::Lt
                | Operation::Le
                | Operation::Gt
                | Operation::Ge
                | Operation::Eq
                | Operation::Neq
        ) {
            return None;
        }
        let source = self.morphisms.get(m)?.source;
        let forms_plain = self.radix_plan(body);
        // mask-frontend: a digit form as row / lane digits of the cell, a
        // captured number's digits folded into the base where it is a
        // literal, a per-launch term where it is not.
        let split =
            |form: &RadixForm| -> Option<(u64, Vec<AxisDigit>, Vec<AxisDigit>, Vec<Digit>)> {
                let (mut base, mut row, mut lane, mut uniform) =
                    (form.base, Vec::new(), Vec::new(), Vec::new());
                for d in &form.digits {
                    if d.stride == 0 {
                        return None;
                    }
                    if d.slot != captures {
                        let o = self.pair_slot_source(source, d.slot)?;
                        match self.tile_literal_u64(o) {
                            Some(v) => {
                                let q = v / d.stride;
                                let q = if d.radix == 0 { q } else { q % d.radix };
                                base = base.checked_add(d.coef.checked_mul(q)?)?;
                            }
                            None if matches!(self.objects.get(o)?.ty, Ty::Int { .. }) => {
                                uniform.push(*d)
                            }
                            None => return None,
                        }
                        continue;
                    }
                    let (s, r, c) = (d.stride, d.radix, d.coef);
                    if s.is_multiple_of(w) {
                        row.push(AxisDigit {
                            stride: s / w,
                            radix: r,
                            coef: c,
                        });
                    } else if r != 0 && w.is_multiple_of(s * r) {
                        lane.push(AxisDigit {
                            stride: s,
                            radix: r,
                            coef: c,
                        });
                    } else if r == 0 && w.is_multiple_of(s) {
                        row.push(AxisDigit {
                            stride: 1,
                            radix: 0,
                            coef: c * (w / s),
                        });
                        lane.push(AxisDigit {
                            stride: s,
                            radix: 0,
                            coef: c,
                        });
                    } else {
                        return None;
                    }
                }
                Some((base, row, lane, uniform))
            };
        let terms = self.fact_arms().guard_terms;
        let side = |o| -> Option<GuardSide> {
            let term = match terms {
                true => self.guard_term(source, body, captures, o, 0)?,
                false => SideTerm {
                    form: forms_plain.get(o)?.clone(),
                    load: None,
                },
            };
            let (base, row, lane, uniform) = split(&term.form)?;
            if !terms && !uniform.is_empty() {
                return None;
            }
            let load = match term.load {
                None => None,
                Some((slot, at, coef)) => {
                    // One value a row: the index of the row's digits alone.
                    let (ib, ir, il, iu) = split(&at)?;
                    if !il.is_empty() || !iu.is_empty() {
                        return None;
                    }
                    Some(RowLoad {
                        slot,
                        base: ib,
                        index: ir,
                        coef,
                    })
                }
            };
            Some(GuardSide {
                base,
                row,
                lane,
                uniform,
                load,
            })
        };
        let _ = bd;
        Some(GuardBox {
            cmp: cmo.op,
            lhs: side(self.pair_slot_source(cmo.source, 0)?)?,
            rhs: side(self.pair_slot_source(cmo.source, 1)?)?,
            fold_when,
            masked,
        })
    }

    /// mask-frontend: **a guard side's term** — value `o` of map body `body`
    /// (its source product `source`, `captures` captures) as a digit form of
    /// the body's inputs ([`Self::radix_plan`]) through `+`, `· literal` and
    /// reads of a captured integer array proven in bounds: an array whose
    /// element law is the index's affine image (`x = map(iota n) (t → a·t +
    /// c)`, `c` of literals: `elem_plan`'s law through the capture) reads as
    /// that image of the index's form; any other as the read itself — a
    /// per-row load (one a side).
    pub(super) fn guard_term(
        &self,
        source: ObjectId,
        body: FuncId,
        captures: u32,
        o: ObjectId,
        depth: u32,
    ) -> Option<SideTerm> {
        if depth > 8 {
            return None;
        }
        let forms = self.radix_plan(body);
        if let Some(f) = forms.get(o) {
            return Some(SideTerm {
                form: f.clone(),
                load: None,
            });
        }
        if let Some(v) = self.tile_literal_u64(o) {
            return Some(SideTerm {
                form: RadixForm {
                    base: v,
                    digits: Vec::new(),
                },
                load: None,
            });
        }
        let [d] = self.in_edges(o) else {
            return None;
        };
        let dm = self.morphisms.get(*d)?;
        let arg = |k| self.pair_slot_source(dm.source, k);
        match dm.op {
            Operation::Add => {
                let x = self.guard_term(source, body, captures, arg(0)?, depth + 1)?;
                let y = self.guard_term(source, body, captures, arg(1)?, depth + 1)?;
                if x.load.is_some() && y.load.is_some() {
                    return None;
                }
                Some(SideTerm {
                    form: x.form.add(&y.form)?,
                    load: x.load.or(y.load),
                })
            }
            Operation::Mul => {
                let (c, x) = match (
                    self.tile_literal_u64(arg(0)?),
                    self.tile_literal_u64(arg(1)?),
                ) {
                    (Some(c), _) => (c, arg(1)?),
                    (_, Some(c)) => (c, arg(0)?),
                    _ => return None,
                };
                let t = self.guard_term(source, body, captures, x, depth + 1)?;
                let load = match t.load {
                    Some((s, at, k)) => Some((s, at, k.checked_mul(c)?)),
                    None => None,
                };
                Some(SideTerm {
                    form: t.form.scale(c)?,
                    load,
                })
            }
            Operation::Index => {
                let input = self.func(body)?.input;
                let slot = self
                    .tile_input_proj_index(arg(0)?, input)
                    .filter(|&s| s < captures)?;
                let at = forms.get(arg(1)?)?.clone();
                if !self.bounds_proof(body).proven(*d) {
                    return None;
                }
                let x = self.pair_slot_source(source, slot)?;
                let Ty::Array { elem, size } = &self.objects.get(x)?.ty else {
                    return None;
                };
                if !matches!(**elem, Ty::Int { .. }) {
                    return None;
                }
                if let Some(form) = self.affine_law_at(x, *size, &at) {
                    return Some(SideTerm { form, load: None });
                }
                Some(SideTerm {
                    form: RadixForm::default(),
                    load: Some((slot, at, 1)),
                })
            }
            _ => None,
        }
    }

    /// mask-frontend: `x[at]` where `x`'s element law is affine in its
    /// index — `x = map(iota n) (t → Σ coef·t + literals)` (its one
    /// definer a classifiable map over an `iota` of its length, the body's
    /// output a digit form whose every digit is the element itself or a
    /// literal capture's) — as that law at `at`'s form.
    pub(super) fn affine_law_at(&self, x: ObjectId, n: u64, at: &RadixForm) -> Option<RadixForm> {
        let [xm] = self.in_edges(x) else {
            return None;
        };
        let morph = self.morphisms.get(*xm)?;
        let Operation::Map {
            body: g,
            captures: gc,
        } = morph.op
        else {
            return None;
        };
        self.map_law(*xm)?;
        let mapped = match gc {
            0 => morph.source,
            _ => self.pair_slot_source(morph.source, gc)?,
        };
        if self.tile_iota_size(mapped) != Some(n) {
            return None;
        }
        let law = self.radix_plan(g).get(self.func(g)?.output)?.clone();
        let mut out = RadixForm {
            base: law.base,
            digits: Vec::new(),
        };
        for d in &law.digits {
            if d.slot == gc {
                if d.stride != 1 || d.radix != 0 {
                    return None;
                }
                out = out.add(&at.scale(d.coef)?)?;
            } else {
                let v = self.tile_literal_u64(self.pair_slot_source(morph.source, d.slot)?)?;
                let q = v / d.stride;
                let q = if d.radix == 0 { q } else { q % d.radix };
                out.base = out.base.checked_add(d.coef.checked_mul(q)?)?;
            }
        }
        Some(out)
    }

    /// F6: cell map `m`'s body reads its index only to address the arrays it
    /// captures (its value at a cell is a function of the values it reads
    /// there), and reads no array but the group's (no bias at the cell).
    pub(super) fn cell_index_addresses(&self, m: MorphismId) -> bool {
        let Some(Operation::Map { body, captures }) = self.morphisms.get(m).map(|x| x.op) else {
            return false;
        };
        let Some(bd) = self.func(body) else {
            return false;
        };
        // The index's uses through integer arithmetic must end as an
        // `Index`'s address (slot 1 of its pair).
        let mut stack: Vec<crate::graph::ObjectId> = self
            .out_edges(bd.input)
            .iter()
            .filter(|&&p| self.morphisms[p].op == (Operation::Proj { index: captures }))
            .map(|&p| self.morphisms[p].target)
            .collect();
        let mut seen = Vec::new();
        while let Some(o) = stack.pop() {
            if seen.contains(&o) {
                continue;
            }
            seen.push(o);
            if o == bd.output {
                return false;
            }
            for &u in self.out_edges(o) {
                let um = &self.morphisms[u];
                match um.op {
                    Operation::Pair { slot: 1, arity: 2 }
                        if self
                            .out_edges(um.target)
                            .iter()
                            .all(|&x| self.morphisms[x].op == Operation::Index) => {}
                    Operation::Add
                    | Operation::Sub
                    | Operation::Mul
                    | Operation::Div
                    | Operation::Mod
                    | Operation::Pair { .. } => stack.push(um.target),
                    _ => return false,
                }
            }
        }
        true
    }
}
