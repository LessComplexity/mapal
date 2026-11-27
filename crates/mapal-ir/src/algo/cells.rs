//! Per-cell maps (`cell_plan`, `trap_cell_plan`) and the footprint plan.

use super::*;

impl CategoryIr {
    /// **The per-cell morphisms of `f` and what each cell reads** (S54,
    /// [`CellSite`]). A `Map` is one when its body is classifiable (trap-,
    /// loop- and effect-free: [`Self::body_is_classifiable`]), calls no
    /// function, every nested `Map`/`Fold` in it walks local values or reads
    /// captured arrays only (`nested`, `work`), every `Index` in it is proven
    /// and reads a captured array, and every capture is an array or a number
    /// (`uniform`). Each `Index` address is classified over
    /// the map index `t` — which exists only when the mapped array is an
    /// `iota`; over any other mapped array the element is a value and an
    /// address built from it is [`Footprint::Other`].
    pub fn cell_plan(&self, f: FuncId) -> SecondaryMap<MorphismId, CellSite> {
        self.fact(f, |s| &s.cell_plan, || self.cell_plan_compute(f))
    }

    pub(super) fn cell_plan_compute(&self, f: FuncId) -> SecondaryMap<MorphismId, CellSite> {
        let mut out = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return out;
        };
        for &m in &fd.morphisms {
            if let Some(site) = self.cell_site(m, false) {
                out.insert(m, site);
            }
        }
        out
    }

    /// bf16-class `device-trap`: **the maps [`Self::cell_plan`] refuses only
    /// for their unproven reads** — a body classifiable but for `Index`es
    /// `bounds_proof` cannot prove (an embedding's row read through a
    /// run-time token id), every other condition of a [`CellSite`] met, the
    /// unproven reads in [`CellSite::traps`]. Where such a cell runs, each
    /// of those reads checks its index: out of bounds, the program's trap is
    /// recorded (and element 0 read) and raised at the next host
    /// checkpoint — the oracle's outcome; in bounds, the oracle's element.
    /// Disjoint from `cell_plan`. Stored on the graph.
    pub fn trap_cell_plan(&self, f: FuncId) -> SecondaryMap<MorphismId, CellSite> {
        self.fact(
            f,
            |s| &s.trap_cell_plan,
            || {
                let mut out = SecondaryMap::new();
                let Some(fd) = self.func(f) else {
                    return out;
                };
                let plain = self.cell_plan(f);
                for &m in &fd.morphisms {
                    if plain.contains_key(m) {
                        continue;
                    }
                    if let Some(site) = self.cell_site(m, true).filter(|s| !s.traps.is_empty()) {
                        out.insert(m, site);
                    }
                }
                out
            },
        )
    }

    /// [`Self::body_is_classifiable`] but for the body's own unproven
    /// `Index` reads (nested bodies classifiable): those reads, or `None`.
    pub(super) fn classifiable_but_index(&self, body: FuncId) -> Option<Vec<MorphismId>> {
        let def = self.func(body)?;
        let effectful = def.morphisms.iter().any(|m| {
            matches!(
                self.morphisms.get(*m).map(|x| x.op),
                Some(Operation::Print { .. } | Operation::TimeMs | Operation::Read)
            )
        });
        if effectful || !self.loop_structure(body).is_empty() {
            return None;
        }
        let bounds = self.bounds_proof(body);
        let mut traps = Vec::new();
        for &m in &def.morphisms {
            if self.morph_trap_free(m, &bounds, None, 0) {
                continue;
            }
            if self.morphisms.get(m)?.op != Operation::Index {
                return None;
            }
            traps.push(m);
        }
        Some(traps)
    }

    pub(super) fn cell_site(&self, m: MorphismId, trapping: bool) -> Option<CellSite> {
        let map = self.morphisms.get(m)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        let mapped = if captures == 0 {
            map.source
        } else {
            self.pair_slot_source(map.source, captures)?
        };
        let Ty::Array { size: extent, .. } = self.objects.get(map.target)?.ty else {
            return None;
        };
        let bd = self.func(body)?;
        let mut nested = Vec::new();
        let leaf = bd
            .morphisms
            .iter()
            .all(|&g| match self.morphisms.get(g).map(|x| x.op) {
                Some(Operation::Map { .. } | Operation::Fold { .. }) => {
                    self.local_nest(g, bd.input, 0)
                        || self
                            .nest_reads(g, bd.input, captures, 0)
                            .map(|r| nested.extend(r))
                            .is_some()
                }
                Some(Operation::Call(_)) | None => false,
                Some(_) => true,
            });
        if !leaf {
            return None;
        }
        let traps = match trapping {
            false if self.body_is_classifiable(body) => Vec::new(),
            false => return None,
            true => self.classifiable_but_index(body)?,
        };
        nested.sort_unstable();
        nested.dedup();
        let mut uniform = Vec::new();
        for s in 0..captures {
            match self
                .pair_slot_source(map.source, s)
                .and_then(|o| self.objects.get(o))
                .map(|o| &o.ty)
            {
                Some(Ty::Array { .. }) => {}
                Some(Ty::Int { .. } | Ty::Float { .. }) => uniform.push(s),
                _ => return None,
            }
        }
        // The mapped element: the index itself over an `iota`, else a read.
        let index = self.tile_iota_size(mapped) == Some(extent);
        let mut reads = Vec::new();
        if !index {
            reads.push(CellRead {
                slot: captures,
                foot: Footprint::Same,
            });
        }
        for &g in &bd.morphisms {
            let x = self.morphisms.get(g)?;
            if x.op != Operation::Index {
                continue;
            }
            let arr = self.pair_slot_source(x.source, 0)?;
            let slot = self
                .tile_input_proj_index(arr, bd.input)
                .filter(|&s| s < captures)?;
            let addr = self.pair_slot_source(x.source, 1)?;
            let foot = if index {
                self.cell_footprint(addr, bd.input, captures)
            } else {
                Footprint::Other
            };
            reads.push(CellRead { slot, foot });
        }
        reads.extend(nested.iter().map(|&slot| CellRead {
            slot,
            foot: Footprint::Other,
        }));
        let work = bd
            .morphisms
            .iter()
            .fold(1u64, |w, &g| w.saturating_add(self.nest_steps(g, 0)));
        let bulk: Vec<MorphismId> = bd
            .morphisms
            .iter()
            .copied()
            .filter(|&g| {
                matches!(
                    self.morphisms[g].op,
                    Operation::Map { .. } | Operation::Fold { .. }
                )
            })
            .collect();
        // device-geometry `row-tree`: the one nested `Map`/`Fold` that loads
        // array elements — the others load none (a Heron root over a constant
        // range, an exponent's squarings: a few steps) and every lane runs
        // them alike.
        let reading: Vec<MorphismId> = match self.fact_arms().row_tree {
            true => bulk
                .iter()
                .copied()
                .filter(|&g| self.nest_reads_memory(g, 0))
                .collect(),
            false => bulk.clone(),
        };
        let tree = match reading[..] {
            [g] if !nested.is_empty() => self.fold_tree_step(g).and_then(|op| {
                let Operation::Fold { captures: fc, .. } = self.morphisms[g].op else {
                    return None;
                };
                let k =
                    self.tile_iota_size(self.pair_slot_source(self.morphisms[g].source, fc + 1)?)?;
                Some((g, op, k))
            }),
            _ => None,
        };
        Some(CellSite {
            traps,
            extent,
            reads,
            body,
            uniform,
            nested,
            work,
            tree,
        })
    }

    /// Element steps of one evaluation of `g` when it is a `Map`/`Fold`: the
    /// extent it walks times (1 + its body's nested steps); 0 otherwise.
    pub(super) fn nest_steps(&self, g: MorphismId, depth: u32) -> u64 {
        let Some(x) = self.morphisms.get(g) else {
            return 0;
        };
        let (Operation::Map { body, captures: nc } | Operation::Fold { body, captures: nc }) = x.op
        else {
            return 0;
        };
        let over = match x.op {
            Operation::Map { .. } if nc == 0 => Some(x.source),
            Operation::Map { .. } => self.pair_slot_source(x.source, nc),
            _ => self.pair_slot_source(x.source, nc + 1),
        };
        let n = match over.and_then(|o| self.objects.get(o)).map(|o| &o.ty) {
            Some(&Ty::Array { size, .. }) => size,
            _ => 0,
        };
        let inner = match self.func(body) {
            Some(b) if depth < 8 => b
                .morphisms
                .iter()
                .map(|&h| self.nest_steps(h, depth + 1))
                .sum(),
            _ => 0,
        };
        n.saturating_mul(inner.saturating_add(1))
    }

    /// The capture slots (`< captures`, of the body whose input is `input`)
    /// that nested `Map`/`Fold` `g` reads: every `Index` in its body (and in
    /// theirs) addresses one of its own captured arrays, each a capture of
    /// the enclosing body; an array it maps or folds over is either local or
    /// such a capture. `None` on anything else (a `Call`, an array of
    /// unknown origin). Depth-guarded (J1).
    pub(super) fn nest_reads(
        &self,
        g: MorphismId,
        input: ObjectId,
        captures: u32,
        depth: u32,
    ) -> Option<Vec<u32>> {
        let x = self.morphisms.get(g)?;
        let (Operation::Map { body, captures: nc } | Operation::Fold { body, captures: nc }) = x.op
        else {
            return None;
        };
        let nb = self.func(body)?;
        if depth >= 8 {
            return None;
        }
        // A component of `g`'s source → the enclosing body's capture slot.
        let up = |o: ObjectId| {
            self.tile_input_proj_index(o, input)
                .filter(|&s| s < captures)
                .filter(|_| matches!(self.objects.get(o).map(|o| &o.ty), Some(Ty::Array { .. })))
        };
        let mut out = Vec::new();
        // The array `g` maps or folds over: an iota or local value, or a capture.
        let over = match x.op {
            Operation::Map { .. } if nc == 0 => Some(x.source),
            Operation::Map { .. } => self.pair_slot_source(x.source, nc),
            _ => self.pair_slot_source(x.source, nc + 1),
        }?;
        if let Some(s) = up(over) {
            out.push(s);
        } else if self.tile_input_proj_index(over, input).is_some() {
            return None;
        }
        for &h in &nb.morphisms {
            let y = self.morphisms.get(h)?;
            let inner = match y.op {
                Operation::Call(_) => return None,
                Operation::Index => {
                    let arr = self.pair_slot_source(y.source, 0)?;
                    vec![
                        self.tile_input_proj_index(arr, nb.input)
                            .filter(|&s| s < nc)?,
                    ]
                }
                Operation::Map { .. } | Operation::Fold { .. } => {
                    self.nest_reads(h, nb.input, nc, depth + 1)?
                }
                _ => continue,
            };
            for ns in inner {
                out.push(up(self.pair_slot_source(x.source, ns)?)?);
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// A nested `Map`/`Fold` `g` of a cell body with input `input` that reads
    /// nothing of the cell's inputs but scalars: no array operand is one of
    /// `input`'s components, and its body (and theirs) holds no `Index` — so
    /// the cell's reads are exactly its own `Index`es (generic-fusion natural
    /// forms: a fold over a range the body makes). Depth-guarded (J1).
    /// Nested `Map`/`Fold` `g` loads array elements: it walks an array that
    /// is no `iota` (whose element is its index), or its body (any depth)
    /// indexes one. A Heron root folded over a constant range loads nothing.
    pub(super) fn nest_reads_memory(&self, g: MorphismId, depth: u32) -> bool {
        let Some(x) = self.morphisms.get(g) else {
            return true;
        };
        let (walked, body) = match x.op {
            Operation::Fold { body, captures } => {
                (self.pair_slot_source(x.source, captures + 1), body)
            }
            Operation::Map { body, captures: 0 } => (Some(x.source), body),
            Operation::Map { body, captures } => (self.pair_slot_source(x.source, captures), body),
            _ => return false,
        };
        if depth > 8 || walked.is_none_or(|a| self.tile_iota_size(a).is_none()) {
            return true;
        }
        self.func(body).is_none_or(|fd| {
            fd.morphisms
                .iter()
                .any(|&h| match self.morphisms.get(h).map(|y| y.op) {
                    Some(Operation::Index | Operation::Call(_)) | None => true,
                    Some(Operation::Map { .. } | Operation::Fold { .. }) => {
                        self.nest_reads_memory(h, depth + 1)
                    }
                    _ => false,
                })
        })
    }

    pub(super) fn local_nest(&self, g: MorphismId, input: ObjectId, depth: u32) -> bool {
        let Some(x) = self.morphisms.get(g) else {
            return false;
        };
        let (Operation::Map { body, .. } | Operation::Fold { body, .. }) = x.op else {
            return false;
        };
        let Some(src) = self.objects.get(x.source) else {
            return false;
        };
        let parts: Vec<ObjectId> = match &src.ty {
            Ty::Tuple(ts) => (0..ts.len() as u32)
                .filter_map(|k| self.pair_slot_source(x.source, k))
                .collect(),
            _ => vec![x.source],
        };
        let captured = parts.iter().any(|&o| {
            matches!(self.objects.get(o).map(|o| &o.ty), Some(Ty::Array { .. }))
                && self.tile_input_proj_index(o, input).is_some()
        });
        let Some(nb) = self.func(body) else {
            return false;
        };
        depth < 8
            && !captured
            && nb
                .morphisms
                .iter()
                .all(|&h| match self.morphisms.get(h).map(|y| y.op) {
                    Some(Operation::Index | Operation::Call(_)) | None => false,
                    Some(Operation::Map { .. } | Operation::Fold { .. }) => {
                        self.local_nest(h, nb.input, depth + 1)
                    }
                    Some(_) => true,
                })
    }

    /// `addr` over the map index `t` (`Proj{captures}` of the body input):
    /// `t`, `t ÷ c`, `t % c` (literal `c`), a literal, or anything else.
    pub(super) fn cell_footprint(
        &self,
        addr: ObjectId,
        input: ObjectId,
        captures: u32,
    ) -> Footprint {
        match self.cell_affine(addr, input, captures) {
            Some(a) if a.cq == 0 && a.cr == 0 => Footprint::Const(a.base),
            // `t` itself: `1·(t÷1)`, or `w·(t÷w) + t%w`.
            Some(a) if a.base == 0 && (a.cq, a.cr) == (a.w, u64::from(a.w > 1)) => Footprint::Same,
            Some(a) if a.base == 0 && a.w > 1 && (a.cq, a.cr) == (1, 0) => {
                Footprint::Row { c: a.w }
            }
            Some(a) if a.base == 0 && a.w > 1 && (a.cq, a.cr) == (0, 1) => {
                Footprint::Col { c: a.w }
            }
            _ => Footprint::Other,
        }
    }

    /// `addr` over the map index `t` (`Proj{captures}` of the body input) as
    /// `base + cq·(t÷w) + cr·(t%w)`: the [`Self::move_affine`] walk (Add,
    /// Mul by a literal, a literal) over the leaves `t`, `t÷c`, `t%c` with
    /// one literal divisor `c` (`w = c`; `w = 1` without one, `t = t÷1`).
    pub(super) fn cell_affine(
        &self,
        addr: ObjectId,
        input: ObjectId,
        captures: u32,
    ) -> Option<CellAff> {
        let raw = self.cell_affine_walk(addr, input, captures, 0)?;
        let w = raw.w.unwrap_or(1);
        Some(CellAff {
            w,
            base: raw.base,
            cq: raw.cq.checked_add(raw.ct.checked_mul(w)?)?,
            cr: raw.cr.checked_add(if w > 1 { raw.ct } else { 0 })?,
        })
    }

    pub(super) fn cell_affine_walk(
        &self,
        o: ObjectId,
        input: ObjectId,
        captures: u32,
        depth: u32,
    ) -> Option<CellAffRaw> {
        if depth >= 16 {
            return None;
        }
        if self.tile_input_proj_index(o, input) == Some(captures) {
            return Some(CellAffRaw {
                ct: 1,
                ..CellAffRaw::default()
            });
        }
        if let Some(base) = self.tile_literal_u64(o) {
            return Some(CellAffRaw {
                base,
                ..CellAffRaw::default()
            });
        }
        let [m] = self.in_edges(o) else {
            return None;
        };
        for (op, quot) in [(Operation::Div, true), (Operation::Mod, false)] {
            if let Some((_, c)) = self.tile_split(*m, input, captures, op) {
                return Some(CellAffRaw {
                    w: Some(c),
                    cq: u64::from(quot),
                    cr: u64::from(!quot),
                    ..CellAffRaw::default()
                });
            }
        }
        let morph = self.morphisms.get(*m)?;
        let lhs = self.pair_slot_source(morph.source, 0)?;
        let rhs = self.pair_slot_source(morph.source, 1)?;
        match morph.op {
            Operation::Add => self
                .cell_affine_walk(lhs, input, captures, depth + 1)?
                .add(self.cell_affine_walk(rhs, input, captures, depth + 1)?),
            Operation::Mul => {
                let (scale, value) = match self.tile_literal_u64(lhs) {
                    Some(s) => (s, rhs),
                    None => (self.tile_literal_u64(rhs)?, lhs),
                };
                self.cell_affine_walk(value, input, captures, depth + 1)?
                    .scale(scale)
            }
            _ => None,
        }
    }

    /// **`footprint_plan` (generic-fusion): every consumer's read edges** —
    /// the one query a fusion rule asks "which producer cells does this
    /// consumer's cell `t` read". A translation of what the recognizers
    /// already found ([`Self::tile_plan_strided`], [`Self::row_fold`],
    /// [`Self::move_plan`], [`Self::cell_plan`], [`Self::bounds_proof`],
    /// [`Self::elem_plan`]); a consumer none of them recognizes reads its
    /// captured arrays [`Reach::Whole`]. Only proven reads are affine or a
    /// range. A producer whose element law is rooted at the index is read as
    /// [`Reach::Law`]; a capture-free classifiable map over a loaded array
    /// adds a second edge onto that array with the same reach (`law` set).
    pub fn footprint_plan(&self, f: FuncId) -> FootprintPlan {
        self.fact(f, |s| &s.footprint_plan, || self.footprint_plan_compute(f))
    }

    pub(super) fn footprint_plan_compute(&self, f: FuncId) -> FootprintPlan {
        let mut plan = FootprintPlan::default();
        let Some(fd) = self.func(f) else {
            return plan;
        };
        let tiles = self.tile_plan_strided(f);
        let folds = self.row_fold(f);
        let moves = self.move_plan(f);
        let cells = self.cell_plan(f);
        let elem = self.elem_plan(f);
        let bounds = self.bounds_proof(f);
        // A closed form of the index alone: capture-free applications down
        // to `Index` (a captured array is bytes the law reads).
        let rooted = |o: ObjectId| {
            let mut s = elem.src(o);
            while let Some(ElemSrc::Apply {
                inner, captures: 0, ..
            }) = s
            {
                s = Some(inner);
            }
            matches!(s, Some(ElemSrc::Index))
        };
        let len = |o: ObjectId| match self.objects.get(o).map(|x| &x.ty) {
            Some(Ty::Array { size, .. }) => *size,
            _ => 0,
        };
        let read = |src: ObjectId, slot: Option<u32>| match slot {
            Some(s) => self.pair_slot_source(src, s),
            None => Some(src),
        };
        let tile_reach = |r: &TileRead, c: u64, k: u64| {
            let taps = match r.ksplit {
                Some(ks) => vec![
                    Tap {
                        coef: ks.cq,
                        ext: k / ks.div,
                    },
                    Tap {
                        coef: ks.cr,
                        ext: ks.div,
                    },
                ],
                None => vec![Tap { coef: r.ck, ext: k }],
            };
            Reach::Affine {
                w: c,
                base: r.base,
                cq: r.ci,
                cr: r.clane,
                taps,
            }
        };
        for &m in &fd.morphisms {
            let Some(morph) = self.morphisms.get(m) else {
                continue;
            };
            let mut edges: Vec<(u32, Option<u32>, Reach)> = Vec::new();
            match morph.op {
                Operation::Map { captures, .. } => {
                    if let Some(s) = tiles.sites.get(m) {
                        for r in [&s.a, &s.b] {
                            edges.push((r.slot, Some(r.slot), tile_reach(r, s.c, s.k)));
                        }
                    } else if let Some(s) = folds.get(m) {
                        // A row fold's cells are its map's: `w = s.w`.
                        edges.push((s.a.slot, Some(s.a.slot), tile_reach(&s.a, s.w, s.k)));
                    } else if let Some(s) = moves.sites.get(m) {
                        let slot = self.move_read_slot(m);
                        if let Some((slot, base)) = slot {
                            edges.push((
                                slot,
                                Some(slot),
                                Reach::Affine {
                                    w: s.width,
                                    base,
                                    cq: s.cq,
                                    cr: s.cr,
                                    taps: Vec::new(),
                                },
                            ));
                        }
                    } else if let Some(s) = cells.get(m) {
                        for r in &s.reads {
                            let slot = (captures > 0).then_some(r.slot);
                            let reach = if s.nested.contains(&r.slot) {
                                read(morph.source, slot)
                                    .map_or(Reach::Whole, |a| Reach::Range { lo: 0, hi: len(a) })
                            } else {
                                self.cell_read_reach(m, r, captures, &bounds, len)
                            };
                            edges.push((r.slot, slot, reach));
                        }
                        for &u in &s.uniform {
                            edges.push((u, Some(u), Reach::Whole));
                        }
                    } else {
                        for s in 0..captures {
                            edges.push((s, Some(s), Reach::Whole));
                        }
                        let mapped = (captures > 0).then_some(captures);
                        let same = Reach::Affine {
                            w: 1,
                            base: 0,
                            cq: 1,
                            cr: 0,
                            taps: Vec::new(),
                        };
                        if let Some(a) = read(morph.source, mapped)
                            && self.tile_iota_size(a).is_none()
                        {
                            edges.push((captures, mapped, same));
                        }
                    }
                }
                Operation::Fold { captures, .. } => {
                    let slot = captures + 1;
                    if let Some(a) = self.pair_slot_source(morph.source, slot) {
                        edges.push((
                            slot,
                            Some(slot),
                            Reach::Affine {
                                w: 1,
                                base: 0,
                                cq: 0,
                                cr: 0,
                                taps: vec![Tap {
                                    coef: 1,
                                    ext: len(a),
                                }],
                            },
                        ));
                    }
                    for s in 0..captures {
                        edges.push((s, Some(s), Reach::Whole));
                    }
                }
                Operation::Index => {
                    let a = self.pair_slot_source(morph.source, 0);
                    let addr = self.pair_slot_source(morph.source, 1);
                    let reach = match (a, addr.and_then(|x| self.tile_literal_u64(x))) {
                        (Some(a), Some(k)) if k < len(a) => Reach::Affine {
                            w: 1,
                            base: k,
                            cq: 0,
                            cr: 0,
                            taps: Vec::new(),
                        },
                        (Some(a), _) if bounds.proven(m) => Reach::Range { lo: 0, hi: len(a) },
                        _ => Reach::Whole,
                    };
                    edges.push((0, Some(0), reach));
                }
                _ => {}
            }
            let mut out = Vec::new();
            for (slot, at, reach) in edges {
                let Some(producer) = read(morph.source, at) else {
                    continue;
                };
                let reach = if rooted(producer) { Reach::Law } else { reach };
                // See through a capture-free classifiable map over a loaded
                // array: `producer[a]` is `body(x[a])`.
                if reach != Reach::Law
                    && let Some(ElemSrc::Apply {
                        captures: 0, inner, ..
                    }) = elem.src(producer)
                    && let ElemSrc::Load { source, slot: None } = **inner
                {
                    out.push(Edge {
                        consumer: m,
                        slot,
                        producer: source,
                        reach: reach.clone(),
                        law: Some(producer),
                    });
                }
                out.push(Edge {
                    consumer: m,
                    slot,
                    producer,
                    reach,
                    law: None,
                });
            }
            if !out.is_empty() {
                plan.edges.insert(m, out);
            }
        }
        plan
    }

    /// A [`CellRead`]'s reach: the affine walk of its address over the map
    /// index (every cell read is proven, `cell_plan`), else the proven range.
    pub(super) fn cell_read_reach(
        &self,
        m: MorphismId,
        r: &CellRead,
        captures: u32,
        bounds: &BoundsProof,
        len: impl Fn(ObjectId) -> u64,
    ) -> Reach {
        let affine = |a: CellAff| Reach::Affine {
            w: a.w,
            base: a.base,
            cq: a.cq,
            cr: a.cr,
            taps: Vec::new(),
        };
        let morph = &self.morphisms[m];
        let arr = if captures == 0 {
            Some(morph.source)
        } else {
            self.pair_slot_source(morph.source, r.slot)
        };
        let whole = arr.map_or(Reach::Whole, |a| Reach::Range { lo: 0, hi: len(a) });
        if r.slot == captures {
            // The mapped element, read at `t`.
            return affine(CellAff {
                w: 1,
                base: 0,
                cq: 1,
                cr: 0,
            });
        }
        let Operation::Map { body, .. } = morph.op else {
            return whole;
        };
        let Some(bd) = self.func(body) else {
            return whole;
        };
        // Every Index of this slot: one reach when they agree, else the range.
        let mut found: Option<Reach> = None;
        for &g in &bd.morphisms {
            let Some(x) = self.morphisms.get(g) else {
                continue;
            };
            if x.op != Operation::Index
                || self
                    .pair_slot_source(x.source, 0)
                    .and_then(|a| self.tile_input_proj_index(a, bd.input))
                    != Some(r.slot)
            {
                continue;
            }
            let reach = self
                .pair_slot_source(x.source, 1)
                .and_then(|a| self.cell_affine(a, bd.input, captures))
                .map_or_else(|| whole.clone(), affine);
            match &found {
                None => found = Some(reach),
                Some(f) if *f == reach => {}
                Some(_) => return whole,
            }
        }
        let _ = bounds;
        found.unwrap_or(whole)
    }

    /// A move site's read slot and base (the [`MoveSite`] record keeps the
    /// coefficients only).
    pub(super) fn move_read_slot(&self, site: MorphismId) -> Option<(u32, u64)> {
        let map = self.morphisms.get(site)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        let bd = self.func(body)?;
        let (q, _) = bd
            .morphisms
            .iter()
            .find_map(|&m| self.tile_split(m, bd.input, captures, Operation::Div))?;
        let (r, _) = bd
            .morphisms
            .iter()
            .find_map(|&m| self.tile_split(m, bd.input, captures, Operation::Mod))?;
        let index = self.tile_definer(bd.output, Operation::Index)?;
        let slot = self.tile_input_proj_index(self.pair_slot_source(index.source, 0)?, bd.input)?;
        let aff = self.move_affine(self.pair_slot_source(index.source, 1)?, q, r, 0)?;
        Some((slot, aff.base))
    }
}
