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

use crate::algo::{AxisDigit, Reach, RowFoldSite, TileSite};
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
    /// flash-attention F1: its rows a digit permutation of the group's
    /// (`None`: the group's own rows).
    pub row_of: Option<RowMap>,
}

/// flash-attention F1: **a member's rows as a digit permutation of the
/// group's** — attention's `P·V` row `(token, head)` reading the scores' row
/// `head·T + token`. Both directions are digit sums, checked against each
/// other on every row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMap {
    /// The group row of member row `r`: `Σ coef·((r ÷ stride) % radix)`.
    pub to_group: Vec<AxisDigit>,
    /// The member row of group row `g` (the inverse).
    pub from_group: Vec<AxisDigit>,
}

fn digit_sum(ds: &[AxisDigit], x: u64) -> u64 {
    ds.iter().map(|d| d.coef * ((x / d.stride) % d.radix)).sum()
}

impl RowMap {
    pub fn group_row(&self, r: u64) -> u64 {
        digit_sum(&self.to_group, r)
    }

    pub fn member_row(&self, g: u64) -> u64 {
        digit_sum(&self.from_group, g)
    }

    /// The map of a row operand reading whole rows under the digit
    /// permutation `to_group` ([`crate::whole_rows`]), when it is one of
    /// `rows` rows: its inverse is the digits with stride and coefficient
    /// exchanged, checked on every row (at most 2²² of them).
    pub fn of(to_group: Vec<AxisDigit>, rows: u64) -> Option<RowMap> {
        if rows == 0 || rows > 1 << 22 || to_group.iter().any(|d| d.radix == 0 || d.stride == 0) {
            return None;
        }
        let from_group: Vec<AxisDigit> = to_group
            .iter()
            .map(|d| AxisDigit {
                stride: d.coef,
                radix: d.radix,
                coef: d.stride,
            })
            .collect();
        if from_group.iter().any(|d| d.stride == 0) {
            return None;
        }
        let m = RowMap {
            to_group,
            from_group,
        };
        let ok = (0..rows).all(|r| {
            let g = m.group_row(r);
            g < rows && m.member_row(g) == r
        });
        ok.then_some(m)
    }
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
    /// flash-attention F2: the group holds a member the row-block units do
    /// not (a mixed-radix or epilogue tile site, or rows under a digit
    /// permutation) — formed after every plain group, from what they left.
    pub extended: bool,
    /// flash-attention F2: the panel batch — the most group rows (a power of
    /// two dividing `rows`) over which every tile member's column operand
    /// address does not move: a block of that many rows reads one `b` panel
    /// (a KV head's rows for its query heads).
    pub panel: u64,
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

impl Cand {
    /// A tile the row-block units realize (`tile_plan_strided`'s plain
    /// form): every other tile is an extended group's.
    fn plain(&self) -> bool {
        match self {
            Cand::Tile(s) => {
                s.radix.is_none()
                    && s.post.is_none()
                    && s.a.narrow.is_none()
                    && s.b.narrow.is_none()
                    && s.a.ksplit.is_none()
                    && s.b.ksplit.is_none()
                    && s.a.clane == 0
                    && s.b.ci == 0
            }
            _ => true,
        }
    }
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
        let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        self.unit_pass(f, &cand, &mut taken, false, &mut plan);
        // flash-attention F2: the groups of every other tile plan — epilogue
        // sites (`tile_plan_post`), mixed-radix ones (`tile_plan_radix`) —
        // from the members the plain groups left, each seeded by such a tile.
        if self.fact_arms().stream {
            let mut ext = cand.clone();
            let post = self.tile_plan_post(f);
            let radix = self.tile_plan_radix(f);
            for (m, s) in post.sites.iter().chain(radix.iter()) {
                let ok = s.rows > 0
                    && s.c > 0
                    && s.k > 0
                    && s.a.ksplit.is_none()
                    && s.b.ksplit.is_none()
                    && (self.fact_arms().unit_uniform
                        || s.radix.as_ref().is_none_or(|r| !r.uniform()));
                if ok && !ext.contains_key(m) {
                    ext.insert(m, Cand::Tile(s.clone()));
                }
            }
            self.unit_pass(f, &ext, &mut taken, true, &mut plan);
        }
        plan
    }

    /// One grouping pass over the candidates `cand` (module doc): each group
    /// seeded by a tile no group holds — in the `extended` pass only a tile
    /// the row-block units do not realize, whose reads may then be a whole
    /// row under a digit permutation ([`RowMap`]).
    fn unit_pass(
        &self,
        f: FuncId,
        cand: &SecondaryMap<MorphismId, Cand>,
        taken: &mut SecondaryMap<MorphismId, ()>,
        extended: bool,
        plan: &mut UnitPlan,
    ) {
        let fp = self.footprint_plan(f);
        let lup = self.last_use_plan(f);
        let pos = |m: MorphismId| lup.position(m).unwrap_or(u32::MAX);
        let target = |m: MorphismId| self.morphisms[m].target;
        let elem = |o: ObjectId| match self.object(o).map(|x| &x.ty) {
            Some(Ty::Array { elem, .. }) => match **elem {
                Ty::Float { bits } | Ty::Int { bits, .. } => u64::from(bits) / 8,
                Ty::Narrow(n) => n.bytes(),
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
        for &seed in &order {
            let Cand::Tile(s) = &cand[seed] else {
                continue;
            };
            if taken.contains_key(seed) || (extended && cand[seed].plain()) {
                continue;
            }
            let rows = s.rows;
            // (member, cells a row, reads, its rows' map)
            let mut g: Vec<(MorphismId, u64, Vec<(u32, usize)>, Option<RowMap>)> =
                vec![(seed, s.c, Vec::new(), None)];
            // The seed's own operands: made before it, read whole by the
            // tile (neither may be a member: there is none yet).
            loop {
                let outs: Vec<ObjectId> = g.iter().map(|x| target(x.0)).collect();
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
                    // flash-attention F1: a tile reading a whole in-group row
                    // through a digit permutation of its rows.
                    let digit = match (&cand[r], extended) {
                        (Cand::Tile(t), true) => self.digit_rows(r, t, &g, &outs, rows),
                        _ => None,
                    };
                    // sweep-attention: an epilogue's reads of a per-row
                    // member at the tile's own rows, by their digit forms
                    // where `Reach` cannot say it (a one-head trip's head
                    // digit `c ÷ d`, always `0`).
                    let post_own = match (&cand[r], extended, &digit) {
                        (Cand::Tile(t), true, None) => self.post_rows(r, t, &g, &outs, rows, |i| i),
                        _ => Vec::new(),
                    };
                    let mut reads = Vec::new();
                    let closed = fp.of(r).iter().all(|e| {
                        let member = outs.iter().position(|&o| o == e.producer);
                        match member {
                            Some(k) => {
                                // A member read under a permutation of the
                                // rows: the digit form's own check.
                                if let Some((map, slots)) = &digit
                                    && slots.contains(&e.slot)
                                {
                                    let _ = map;
                                    if !reads.contains(&(e.slot, k)) {
                                        reads.push((e.slot, k));
                                    }
                                    return true;
                                }
                                // A member read by a member whose rows are a
                                // permutation: only in the permuted rows.
                                if g[k].3.is_some() {
                                    return false;
                                }
                                let pw = g[k].1;
                                let a_slot = match &cand[r] {
                                    Cand::Tile(t) => Some(t.a.slot),
                                    _ => None,
                                };
                                let row_operand = a_slot.is_none_or(|a| a == e.slot);
                                // An epilogue's read of a per-row member at
                                // its own row (a `÷ den` after `P·V`).
                                let post_row = match &cand[r] {
                                    Cand::Tile(t) => {
                                        t.post.as_ref().is_some_and(|p| p.reads.contains(&e.slot))
                                            && pw == 1
                                    }
                                    _ => false,
                                };
                                // sweep-attention: a row operand the tile's own
                                // plain read proves row `i` of the member
                                // (`a[base + ci·i + ck·k]`, `ci` its row width,
                                // every `k` inside the row) where `Reach`
                                // cannot (a one-head trip's head digit).
                                let tile_row = match &cand[r] {
                                    Cand::Tile(t) if extended && row_operand => {
                                        t.radix
                                            .as_ref()
                                            .is_none_or(|x| x.a.is_none() && x.ua.is_empty())
                                            && t.a.ksplit.is_none()
                                            && t.a.clane == 0
                                            && t.a.ci == pw
                                            && t.a.ck >= 1
                                            && t.a.base + t.a.ck * (t.k - 1) < pw
                                    }
                                    _ => false,
                                };
                                let ok = e.law.is_none()
                                    && (row_operand || post_row)
                                    && (e.reach.within_rows(w, pw, rows)
                                        || tile_row
                                        || (post_row && post_own.contains(&e.slot)));
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
                        admitted = Some((r, w, reads, digit.map(|d| d.0)));
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
            for x in &g {
                taken.insert(x.0, ());
            }
            let mut grp = self.unit_group(rows, &g, cand, &lup, elem, len, readers);
            grp.extended = extended;
            plan.groups.push(grp);
        }
    }

    /// flash-attention F1: tile `r`'s reads of the group `g`'s members under
    /// a digit permutation of its rows — its row operand a whole row of an
    /// in-group member read through its mixed-radix rows
    /// ([`crate::whole_rows`]), each epilogue read of a per-row member at the
    /// same permuted row (checked on every row and both end lanes) — as the
    /// rows' map and the slots it covers.
    fn digit_rows(
        &self,
        r: MorphismId,
        t: &TileSite,
        g: &[(MorphismId, u64, Vec<(u32, usize)>, Option<RowMap>)],
        outs: &[ObjectId],
        rows: u64,
    ) -> Option<(RowMap, Vec<u32>)> {
        let ra = t.radix.as_ref()?.a.as_ref()?;
        let map = self.morphisms.get(r)?;
        let a = self.pair_slot_source(map.source, t.a.slot)?;
        let k = outs.iter().position(|&o| o == a)?;
        if g[k].1 != t.k || g[k].3.is_some() {
            return None;
        }
        let (base, digits) = crate::algo::whole_rows(ra, t.k)?;
        if base != 0 {
            return None;
        }
        let rm = RowMap::of(digits, rows)?;
        let mut slots = vec![t.a.slot];
        // The epilogue's reads of the group's per-row members, each at the
        // permuted row.
        if let Some(post) = &t.post {
            let at = self.post_rows(r, t, g, outs, rows, |i| rm.group_row(i));
            for &slot in &post.reads {
                let src = self.pair_slot_source(map.source, slot)?;
                if outs.contains(&src) && !at.contains(&slot) {
                    return None;
                }
            }
            slots.extend(at);
        }
        Some((rm, slots))
    }

    /// flash-attention F1 / sweep-attention: the epilogue slots of tile `r`
    /// (`t.post`) that read a per-row member of the group `g` only at group
    /// row `row(i)` for the tile's row `i` — every read of the slot in the
    /// body an index whose digit form ([`CategoryIr::radix_plan`]) gives
    /// that row at both end lanes of every row. The check where `Reach`
    /// cannot say it: a head digit `c ÷ d` that is `0` on every cell of a
    /// one-head trip, a digit permutation of the rows.
    fn post_rows(
        &self,
        r: MorphismId,
        t: &TileSite,
        g: &[(MorphismId, u64, Vec<(u32, usize)>, Option<RowMap>)],
        outs: &[ObjectId],
        rows: u64,
        row: impl Fn(u64) -> u64,
    ) -> Vec<u32> {
        let mut slots = Vec::new();
        let Some(post) = &t.post else {
            return slots;
        };
        let Some(map) = self.morphisms.get(r) else {
            return slots;
        };
        let Operation::Map { body, captures } = map.op else {
            return slots;
        };
        let Some(bd) = self.func(body) else {
            return slots;
        };
        let forms = self.radix_plan(body);
        let input = bd.input;
        let out = t.radix.as_ref().and_then(|x| x.out.clone());
        let cell = |i: u64, lane: u64| match &out {
            Some(o) => o.at(i, lane, 0),
            None => i * t.c + lane,
        };
        for &slot in &post.reads {
            let Some(src) = self.pair_slot_source(map.source, slot) else {
                continue;
            };
            let Some(p) = outs.iter().position(|&o| o == src) else {
                continue;
            };
            if g[p].1 != 1 || g[p].3.is_some() {
                continue;
            }
            // Every read of the slot in the body: its index's digits.
            let mut any = false;
            let mut all = true;
            for &m in &bd.morphisms {
                let mo = &self.morphisms[m];
                if mo.op != Operation::Index {
                    continue;
                }
                let Some(arr) = self.pair_slot_source(mo.source, 0) else {
                    all = false;
                    break;
                };
                if self.tile_input_proj_index(arr, input) != Some(slot) {
                    continue;
                }
                let form = self
                    .pair_slot_source(mo.source, 1)
                    .and_then(|at| forms.get(at))
                    .filter(|f| f.digits.iter().all(|d| d.slot == captures));
                let Some(form) = form else {
                    all = false;
                    break;
                };
                let index = |e: u64| -> u64 {
                    form.base
                        + form
                            .digits
                            .iter()
                            .map(|d| {
                                let q = e / d.stride;
                                d.coef * if d.radix == 0 { q } else { q % d.radix }
                            })
                            .sum::<u64>()
                };
                let ok = (0..rows).all(|i| {
                    let want = row(i);
                    index(cell(i, 0)) == want && index(cell(i, t.c - 1)) == want
                });
                if !ok {
                    all = false;
                    break;
                }
                any = true;
            }
            if any && all {
                slots.push(slot);
            }
        }
        slots
    }

    #[allow(clippy::too_many_arguments)]
    fn unit_group(
        &self,
        rows: u64,
        g: &[(MorphismId, u64, Vec<(u32, usize)>, Option<RowMap>)],
        cand: &SecondaryMap<MorphismId, Cand>,
        lup: &crate::algo::LastUsePlan,
        elem: impl Fn(ObjectId) -> u64,
        len: impl Fn(ObjectId) -> u64,
        readers: impl Fn(ObjectId) -> Vec<MorphismId>,
    ) -> UnitGroup {
        let fp = self.footprint_plan(lup_owner(self, g[0].0));
        let mut members: Vec<UnitMember> = g
            .iter()
            .map(|(m, w, reads, row_of)| {
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
                    row_of: row_of.clone(),
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
        let panel = self.panel_rows(rows, &members);
        UnitGroup {
            rows,
            members,
            row_bytes,
            shared_bytes,
            extended: false,
            panel,
        }
    }
}

impl CategoryIr {
    /// flash-attention F2: the most group rows — a power of two dividing
    /// `rows` — over which no tile member's column operand address moves
    /// (its mixed-radix row digits, at each member's own rows; a plain
    /// column operand reads no row).
    fn panel_rows(&self, rows: u64, members: &[UnitMember]) -> u64 {
        let mut step = 0u64;
        for mb in members {
            let UnitKind::Tile(t) = &mb.kind else {
                continue;
            };
            let Some(b) = t.radix.as_ref().and_then(|r| r.b.as_ref()) else {
                continue;
            };
            if b.i.is_empty() {
                continue;
            }
            let at = |gr: u64| {
                let r = mb.row_of.as_ref().map_or(gr, |m| m.member_row(gr));
                digit_sum(&b.i, r)
            };
            let mut prev = at(0);
            for gr in 1..rows.min(1 << 22) {
                let x = at(gr);
                if x != prev {
                    step = gcd(step, gr);
                    prev = x;
                }
            }
        }
        match step {
            0 => rows.max(1),
            s => 1u64 << s.trailing_zeros().min(rows.trailing_zeros()),
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// The function owning morphism `m`.
fn lup_owner(ir: &CategoryIr, m: MorphismId) -> FuncId {
    ir.try_owner(ir.morphisms[m].target)
        .expect("a member's output has an owner")
}
