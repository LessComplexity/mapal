//! **`linear_plan`** (plan-cost §4a, D6): a map whose cell is a sum of tile
//! sites' cells at that cell — `out[t] = Σᵢ xᵢ[t]`, each `xᵢ` a fold over the
//! same `(rows, c, k)` from seed 0 that nothing else reads and that does not
//! escape — is one fold over the `T` terms' concatenated `k` ranges. Legality
//! and the source's bound only; the realization (each term's chain ≤ `K`, the
//! term boundary a partition boundary) is the backend's.

use slotmap::SecondaryMap;

use super::{CellSite, Footprint, LastUsePlan, TilePlan, bound_coef};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation};
use crate::ty::Value;

/// One sum of folds (keyed by the summing map in [`CategoryIr::linear_plan`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSite {
    /// The terms' tile sites ([`CategoryIr::tile_plan`]), in the body's
    /// left-to-right leaf order.
    pub terms: Vec<MorphismId>,
    /// Each term's operand producers `(a, b)`: two terms reading one object at
    /// one [`super::TileRead`] share that operand.
    pub operands: Vec<(ObjectId, ObjectId)>,
    /// ADR-0040's coefficient of `M = Σ Mᵢ` for the source: each term's
    /// `k`-step fold, then `T − 1` adds — `γ_{K+T}` (D6). A realization that
    /// merges the terms into one `T·K` chain exceeds it; one that keeps each
    /// chain ≤ `K` and adds ≤ `T` partials per cell does not.
    pub bound: f64,
}

impl CategoryIr {
    /// The deduced query `linear : IR × FuncId → {map ↦ LinearSite}`. Stored
    /// on the graph.
    pub fn linear_plan(&self, f: FuncId) -> SecondaryMap<MorphismId, LinearSite> {
        self.fact(f, |s| &s.linear_plan, || self.linear_plan_compute(f))
    }

    fn linear_plan_compute(&self, f: FuncId) -> SecondaryMap<MorphismId, LinearSite> {
        let mut out = SecondaryMap::new();
        let cells = self.cell_plan(f);
        if cells.is_empty() {
            return out;
        }
        let (tiles, lup) = (self.tile_plan(f), self.last_use_plan(f));
        for (m, cell) in cells.iter() {
            if let Some(s) = self.linear_site(m, cell, &tiles, &lup) {
                out.insert(m, s);
            }
        }
        out
    }

    fn linear_site(
        &self,
        m: MorphismId,
        cell: &CellSite,
        tiles: &TilePlan,
        lup: &LastUsePlan,
    ) -> Option<LinearSite> {
        let map = self.morphisms.get(m)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        // Every read a captured array at the cell (the mapped array is the
        // index), no number capture, no nested fold.
        let at_cell = cell
            .reads
            .iter()
            .all(|r| r.foot == Footprint::Same && r.slot < captures);
        if !at_cell || !cell.uniform.is_empty() || !cell.nested.is_empty() {
            return None;
        }
        // The body's value: a tree of `+` over those reads.
        let bd = self.func(body)?;
        let mut slots = Vec::new();
        let mut stack = vec![bd.output];
        while let Some(v) = stack.pop() {
            if let Some(add) = self.tile_definer(v, Operation::Add) {
                stack.push(self.pair_slot_source(add.source, 1)?);
                stack.push(self.pair_slot_source(add.source, 0)?);
                continue;
            }
            let ix = self.tile_definer(v, Operation::Index)?;
            let arr = self.pair_slot_source(ix.source, 0)?;
            let slot = self
                .tile_input_proj_index(arr, bd.input)
                .filter(|&s| s < captures)?;
            let addr = self.pair_slot_source(ix.source, 1)?;
            if self.cell_footprint(addr, bd.input, captures) != Footprint::Same {
                return None;
            }
            slots.push(slot);
        }
        if slots.len() < 2 {
            return None;
        }
        let (mut terms, mut operands) = (Vec::new(), Vec::new());
        let mut first: Option<&super::TileSite> = None;
        let mut tf32 = false;
        for slot in slots {
            let arr = self.pair_slot_source(map.source, slot)?;
            let [p] = *self.in_edges(arr) else {
                return None;
            };
            let site = tiles.sites.get(p)?;
            let zero = matches!(site.seed, Value::F32(x) if x == 0.0)
                || matches!(site.seed, Value::F64(x) if x == 0.0);
            // Read only through this map's captures, and not escaping.
            let only = self
                .out_edges(arr)
                .iter()
                .all(|&e| self.morphisms[e].target == map.source)
                && self.out_edges(map.source) == [m];
            let same = first.is_none_or(|f0| {
                (f0.rows, f0.c, f0.k, &f0.elem) == (site.rows, site.c, site.k, &site.elem)
            });
            if !zero || !only || !same || lup.escapes(arr) || terms.contains(&p) {
                return None;
            }
            if site.rows * site.c != cell.extent {
                return None;
            }
            let src = self.morphisms[p].source;
            operands.push((
                self.pair_slot_source(src, site.a.slot)?,
                self.pair_slot_source(src, site.b.slot)?,
            ));
            terms.push(p);
            tf32 |= site.tf32;
            first.get_or_insert(site);
        }
        let k = first?.k;
        Some(LinearSite {
            bound: bound_coef(k + terms.len() as u64 - 1, tf32),
            terms,
            operands,
        })
    }
}
