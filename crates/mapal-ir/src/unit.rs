//! **`unit_plan` (compile-time-graph A): row-block units.** Which map and
//! fold sites one kernel can hold with every intermediate on chip, read off
//! the facts the recognizers and the liveness pass already stored — no new
//! geometry is derived here.
//!
//! A group is a set of tile sites ([`CategoryIr::tile_plan_strided`]: `out[i,
//! j] = fold_k a[i, k]·b[k, j]`, `a` row-only and `b` column-only), row folds
//! ([`CategoryIr::row_fold`], one value per row) and cells
//! ([`CategoryIr::cell_plan`], no nested fold, no number capture) whose cells
//! all factor through one leading index `i < rows` — `w` cells per row — such
//! that every in-group edge `p → c` is row-closed ([`Reach::within_rows`]:
//! `c`'s row `i` reads only `p`'s row `i`). A tile member reads an in-group
//! producer only as its row operand `a`: its `b` operand is read by every row,
//! so a producer there is a grid-wide dependency. Every other operand is made
//! before the group's first member. A member no reader outside the group needs
//! and that does not escape ([`LastUsePlan::escapes`]) is never stored; one
//! with an in-group reader keeps its rows on chip, in the buffer first-fit
//! coloring by death inside the group gives it (a cell that is the last reader
//! of a same-width row it reads only at its own cell takes that row's buffer
//! in place). Geometry only: how many rows a unit holds, and whether the
//! group pays at all, is the backend's question (`TargetProfile`).

use slotmap::SecondaryMap;

use crate::algo::{Reach, RowFoldSite, TileSite};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation};
use crate::ty::Ty;

/// What a [`UnitMember`] is.
#[derive(Clone, Debug, PartialEq)]
pub enum UnitKind {
    /// A tile site: `w = c` cells per row, each a fold over `k` of its row
    /// operand `a` against the column operand `b`.
    Tile(TileSite),
    /// A row fold: one value per row.
    Fold(RowFoldSite),
    /// A per-cell map.
    Cell,
}

/// One member of a [`UnitGroup`].
#[derive(Clone, Debug, PartialEq)]
pub struct UnitMember {
    pub m: MorphismId,
    pub out: ObjectId,
    pub kind: UnitKind,
    /// Cells per unit row.
    pub w: u64,
    /// The output's element bytes.
    pub elem: u64,
    /// `(slot, member)`: each input slot read from an earlier member.
    pub reads: Vec<(u32, usize)>,
    /// The value escapes or a reader outside the group exists: stored.
    pub stored: bool,
    /// Its rows stay on chip for a later member: the buffer it takes.
    pub color: Option<usize>,
}

/// One row-block group.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitGroup {
    /// The unit axis: every member's cell `t` is `i·w + j`, `i < rows`.
    pub rows: u64,
    /// In position order (`last_use_plan`); the first is the launch.
    pub members: Vec<UnitMember>,
    /// On-chip bytes one unit row takes: each color's widest row.
    pub row_bytes: u64,
    /// Operand bytes a unit reads whole whatever its rows: the tile members'
    /// column operands.
    pub shared_bytes: u64,
}

/// Every row-block group of one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnitPlan {
    pub groups: Vec<UnitGroup>,
}

impl Reach {
    /// Does this reach, read by a consumer with `cw` cells per unit row, stay
    /// inside the producer's row `i` (`pw` cells per row) for every `i <
    /// rows`? Exact: when the consumer's row is whole reach rows (`cw % w ==
    /// 0`) both ends of the hull are affine in `i`, so the end rows decide;
    /// otherwise every row is checked, up to 65536 of them (an unaligned read
    /// past that is refused).
    pub fn within_rows(&self, cw: u64, pw: u64, rows: u64) -> bool {
        let Reach::Affine { w, .. } = self else {
            return false;
        };
        if rows == 0 {
            return false;
        }
        let inside = |i: u64| {
            self.span(i * cw..(i + 1) * cw)
                .is_some_and(|s| s.start >= i * pw && s.end <= (i + 1) * pw)
        };
        if *w > 0 && cw % w == 0 {
            return [0, 1.min(rows - 1), rows - 1].into_iter().all(inside);
        }
        rows <= 1 << 16 && (0..rows).all(inside)
    }

    /// The producer address hull the consumer cells `cells` read, for an
    /// affine reach (over a one-wide producer, [`Reach::rect`]).
    pub fn span(&self, cells: std::ops::Range<u64>) -> Option<std::ops::Range<u64>> {
        let Reach::Affine { w, .. } = self else {
            return None;
        };
        let (lo, hi) = (cells.start, cells.end);
        if hi <= lo || *w == 0 {
            return None;
        }
        let (q0, q1) = (lo / w, (hi - 1) / w);
        let cols = if q0 == q1 {
            lo % w..(hi - 1) % w + 1
        } else {
            0..*w
        };
        let [rows, _] = self.rect(1, q0..q1 + 1, cols)?;
        Some(rows)
    }

    /// The consumer's own cell: `t` reads `t`.
    pub fn is_same(&self) -> bool {
        matches!(self, Reach::Affine { w: 1, base: 0, cq: 1, cr: 0, taps } if taps.is_empty())
    }
}

/// A candidate member before grouping.
#[derive(Clone)]
enum Cand {
    Tile(TileSite),
    Fold(RowFoldSite),
    Cell(u64),
}

impl CategoryIr {
    /// **The row-block groups of `f`** (module doc): stored on the graph,
    /// built from `tile_plan_strided`, `row_fold`, `cell_plan`,
    /// `footprint_plan` and `last_use_plan`.
    pub fn unit_plan(&self, f: FuncId) -> UnitPlan {
        self.fact(f, |s| &s.unit_plan, || self.unit_plan_compute(f))
    }

    fn unit_plan_compute(&self, f: FuncId) -> UnitPlan {
        let mut plan = UnitPlan::default();
        if self.func(f).is_none() {
            return plan;
        }
        let tiles = self.tile_plan_strided(f);
        let folds = self.row_fold(f);
        let cells = self.cell_plan(f);
        let fp = self.footprint_plan(f);
        let lup = self.last_use_plan(f);
        let mut cand: SecondaryMap<MorphismId, Cand> = SecondaryMap::new();
        for (m, s) in tiles.sites.iter() {
            let plain = s.radix.is_none()
                && s.a.ksplit.is_none()
                && s.b.ksplit.is_none()
                && s.a.clane == 0
                && s.b.ci == 0
                && s.rows > 0
                && s.c > 0
                && s.k > 0;
            if plain {
                cand.insert(m, Cand::Tile(s.clone()));
            }
        }
        for (m, s) in folds.iter() {
            if s.w == 1 && s.a.ksplit.is_none() && s.rows > 0 && !cand.contains_key(m) {
                cand.insert(m, Cand::Fold(s.clone()));
            }
        }
        for (m, c) in cells.iter() {
            if c.nested.is_empty() && c.uniform.is_empty() && c.extent > 0 && !cand.contains_key(m)
            {
                cand.insert(m, Cand::Cell(c.extent));
            }
        }
        let pos = |m: MorphismId| lup.position(m).unwrap_or(u32::MAX);
        let target = |m: MorphismId| self.morphisms[m].target;
        let elem = |o: ObjectId| match self.object(o).map(|x| &x.ty) {
            Some(Ty::Array { elem, .. }) => match **elem {
                Ty::Float { bits } | Ty::Int { bits, .. } => u64::from(bits) / 8,
                _ => 0,
            },
            _ => 0,
        };
        let len = |o: ObjectId| match self.object(o).map(|x| &x.ty) {
            Some(Ty::Array { size, .. }) => *size,
            _ => 0,
        };
        // The non-`Pair` morphisms reading `o`, through the products it rides.
        let readers = |o: ObjectId| {
            let mut out = Vec::new();
            let mut stack = vec![o];
            while let Some(x) = stack.pop() {
                for &m in self.out_edges(x) {
                    match self.morphisms[m].op {
                        Operation::Pair { .. } => stack.push(self.morphisms[m].target),
                        _ => out.push(m),
                    }
                }
            }
            out
        };
        let mut order: Vec<MorphismId> = cand.keys().collect();
        order.sort_by_key(|&m| (pos(m), format!("{m:?}")));
        let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for &seed in &order {
            let Cand::Tile(s) = &cand[seed] else {
                continue;
            };
            if taken.contains_key(seed) {
                continue;
            }
            let rows = s.rows;
            let mut g: Vec<(MorphismId, u64, Vec<(u32, usize)>)> = vec![(seed, s.c, Vec::new())];
            // The seed's own operands: made before it, read whole by the
            // tile (neither may be a member: there is none yet).
            loop {
                let outs: Vec<ObjectId> = g.iter().map(|&(m, _, _)| target(m)).collect();
                let launch = pos(g[0].0);
                let mut next: Vec<MorphismId> = order
                    .iter()
                    .copied()
                    .filter(|&r| !taken.contains_key(r) && !g.iter().any(|x| x.0 == r))
                    .filter(|&r| fp.of(r).iter().any(|e| outs.contains(&e.producer)))
                    .collect();
                next.sort_by_key(|&r| pos(r));
                let mut admitted = None;
                for r in next {
                    let w = match &cand[r] {
                        Cand::Tile(t) if t.rows == rows => t.c,
                        Cand::Fold(f) if f.rows == rows => 1,
                        Cand::Cell(ext) if ext % rows == 0 => ext / rows,
                        _ => continue,
                    };
                    let mut reads = Vec::new();
                    let closed = fp.of(r).iter().all(|e| {
                        let member = outs.iter().position(|&o| o == e.producer);
                        match member {
                            Some(k) => {
                                let pw = g[k].1;
                                let a_slot = match &cand[r] {
                                    Cand::Tile(t) => Some(t.a.slot),
                                    _ => None,
                                };
                                let row_operand = a_slot.is_none_or(|a| a == e.slot);
                                let ok = e.law.is_none()
                                    && row_operand
                                    && e.reach.within_rows(w, pw, rows);
                                if ok && !reads.contains(&(e.slot, k)) {
                                    reads.push((e.slot, k));
                                }
                                ok
                            }
                            // Bytes from outside: made before the launch.
                            None => {
                                e.reach == Reach::Law
                                    || self.in_edges(e.producer).iter().all(|&p| pos(p) < launch)
                            }
                        }
                    });
                    if closed && !reads.is_empty() {
                        admitted = Some((r, w, reads));
                        break;
                    }
                }
                match admitted {
                    Some(x) => g.push(x),
                    None => break,
                }
            }
            if g.len() < 2 {
                continue;
            }
            for &(m, _, _) in &g {
                taken.insert(m, ());
            }
            plan.groups
                .push(self.unit_group(rows, &g, &cand, &lup, elem, len, readers));
        }
        plan
    }

    #[allow(clippy::too_many_arguments)]
    fn unit_group(
        &self,
        rows: u64,
        g: &[(MorphismId, u64, Vec<(u32, usize)>)],
        cand: &SecondaryMap<MorphismId, Cand>,
        lup: &crate::algo::LastUsePlan,
        elem: impl Fn(ObjectId) -> u64,
        len: impl Fn(ObjectId) -> u64,
        readers: impl Fn(ObjectId) -> Vec<MorphismId>,
    ) -> UnitGroup {
        let fp = self.footprint_plan(lup_owner(self, g[0].0));
        let mut members: Vec<UnitMember> = g
            .iter()
            .map(|(m, w, reads)| {
                let out = self.morphisms[*m].target;
                let kind = match &cand[*m] {
                    Cand::Tile(t) => UnitKind::Tile(t.clone()),
                    Cand::Fold(f) => UnitKind::Fold(f.clone()),
                    Cand::Cell(_) => UnitKind::Cell,
                };
                let outside = readers(out)
                    .into_iter()
                    .any(|r| !g.iter().any(|x| x.0 == r));
                UnitMember {
                    m: *m,
                    out,
                    kind,
                    w: *w,
                    elem: elem(out),
                    reads: reads.clone(),
                    stored: lup.escapes(out) || outside,
                    color: None,
                }
            })
            .collect();
        // Death inside the group: the last member reading each one.
        let n = members.len();
        let mut death: Vec<Option<usize>> = vec![None; n];
        for (c, mb) in members.iter().enumerate() {
            for &(_, p) in &mb.reads {
                death[p] = Some(death[p].map_or(c, |d: usize| d.max(c)));
            }
        }
        // First-fit coloring by death: a member with an in-group reader takes
        // the lowest buffer free at its birth; a cell that is the last reader
        // of a same-width row it reads only at its own cell takes that row's
        // buffer in place.
        let mut occupant: Vec<(usize, u64)> = Vec::new(); // (member, bytes) per color
        for c in 0..n {
            if death[c].is_none() {
                continue;
            }
            let mb = &members[c];
            let same_only = |p: usize| {
                fp.of(mb.m)
                    .iter()
                    .filter(|e| e.producer == members[p].out)
                    .all(|e| e.reach.is_same())
            };
            let free = |j: usize| match death[j] {
                Some(d) if d < c => true,
                Some(d) if d == c => {
                    matches!(mb.kind, UnitKind::Cell)
                        && members[j].w == mb.w
                        && members[j].elem == mb.elem
                        && same_only(j)
                }
                _ => false,
            };
            let bytes = mb.w * mb.elem;
            let col = occupant.iter().position(|&(j, _)| free(j));
            let col = match col {
                Some(k) => {
                    occupant[k] = (c, occupant[k].1.max(bytes));
                    k
                }
                None => {
                    occupant.push((c, bytes));
                    occupant.len() - 1
                }
            };
            members[c].color = Some(col);
        }
        let row_bytes = occupant.iter().map(|&(_, b)| b).sum();
        let shared_bytes = members
            .iter()
            .filter_map(|mb| match &mb.kind {
                UnitKind::Tile(t) => {
                    let b = self.pair_slot_source(self.morphisms[mb.m].source, t.b.slot)?;
                    Some(len(b) * elem(b))
                }
                _ => None,
            })
            .sum();
        UnitGroup {
            rows,
            members,
            row_bytes,
            shared_bytes,
        }
    }
}

/// The function owning morphism `m`.
fn lup_owner(ir: &CategoryIr, m: MorphismId) -> FuncId {
    ir.try_owner(ir.morphisms[m].target)
        .expect("a member's output has an owner")
}
