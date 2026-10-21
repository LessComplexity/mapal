//! Retention at part-of-object scope (S53 `retain-delta`): inside a loop, a
//! tile site `y = F(x, w)` over a carried `x` recomputes only the rows the
//! iteration wrote.
//!
//! **The rule.** If (i) the iteration writes only cells `Δ(t)` of `x`
//! ([`mapal_ir::Delta::Rows`]: the value crossing the back edge is the merge
//! value updated there), (ii) `w` is loop-invariant, and (iii) output row `r`
//! reads `x` only inside row `r` (`a.clane == 0`, `a.ci == stride`, the row's
//! `k` span within one stride), then every output row outside `Δ(t)` is
//! bit-identical to last iteration's — same inputs, same fold, same order. So
//! `y`'s own slot, which persists across iterations, is the retained result:
//! the first pass runs the whole site into it, every later pass runs the site
//! restricted to the written row. Nothing here names a workload; on a decode
//! loop the retained `K = x·Wk`, `V = x·Wv` are what is called a KV cache.
//!
//! The restricted pass is the same realization at a smaller extent: on the GPU
//! a `rows = 1` site whose operand and output pointers are offset to the row
//! (its kernel is in the device unit, [`retained_sites`] is shared with
//! [`super::gpu::device_module`]); on the host the site's own nest under a
//! `[r·c, r·c + c)` bound (`bulk_bounds`). Anything the row test cannot place
//! at run time (first pass, a write straddling two rows, a row past the site)
//! runs the whole site — the OFF arm, which is always correct.

use super::*;
use mapal_ir::{Affine, Delta, TileRead};

/// One site the rule fires on.
#[derive(Clone, Debug)]
pub(crate) struct Retained {
    /// The loop whose entry resets the first-pass flag.
    pub merge: ObjectId,
    /// The written row's first cell: `a·var + b`.
    pub start: Affine,
    pub stride: u64,
    /// The same site, one row, `a` read from base 0 (the host offsets it).
    pub row: TileSite,
}

/// Every tile site of `f` the rule fires on, keyed by the site's output.
/// GPU profiles only: the only parts the rung was measured on.
pub(crate) fn retained_sites(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    rungs: crate::GpuRungs,
) -> SecondaryMap<ObjectId, Retained> {
    let mut out = SecondaryMap::new();
    let on = rungs.retain_delta || (rungs.resident && rungs.retain_resident);
    if !on || profile.gpu().is_none() {
        return out;
    }
    let deltas = ir.write_delta(f);
    if !deltas.values().any(|d| matches!(d, Delta::Rows { .. })) {
        return out;
    }
    let pair_src = |agg: ObjectId, k: u32| pair_source_ir(ir, agg, k);
    // `y`'s slot is the retained result, so nothing may write it in place.
    let updated: Vec<ObjectId> = ir
        .func(f)
        .expect("func resolves")
        .morphisms
        .iter()
        .map(|&m| ir.morphism(m).expect("morphism resolves"))
        .filter(|m| m.op == Operation::Update)
        .filter_map(|m| pair_src(m.source, 0))
        .collect();
    // A site in a guard arm may skip an iteration, and its rows go stale.
    let gated: Vec<MorphismId> = ir
        .guard_plan(f)
        .into_iter()
        .filter(GuardSite::gated)
        .flat_map(|s| s.on_true.own.into_iter().chain(s.on_false.own))
        .collect();
    let plan = ir.tile_plan(f);
    for lscc in ir.loop_structure(f) {
        let &[merge] = lscc.merges.as_slice() else {
            continue;
        };
        let Some(lp) = ir.loop_plan(f, merge) else {
            continue;
        };
        let in_loop = |m: MorphismId| lp.decide_order.contains(&m) || lp.advance_order.contains(&m);
        for (m, site) in plan.sites.iter() {
            if !in_loop(m) || gated.contains(&m) {
                continue;
            }
            let morph = ir.morphism(m).expect("morphism resolves");
            let (Some(a), Some(b)) = (
                pair_src(morph.source, site.a.slot),
                pair_src(morph.source, site.b.slot),
            ) else {
                continue;
            };
            // This loop's `Δ` of `a`: the carried array's own, or (S54) a
            // `Same` image of it; the counter view names the loop.
            let Some((stride, start)) = deltas.values().find_map(|d| match *d {
                Delta::Rows {
                    stride,
                    start,
                    next,
                } if next == a && proj_source(ir, start.var) == Some(merge) => {
                    Some((stride, start))
                }
                _ => None,
            }) else {
                continue;
            };
            let row_local = site.a.ksplit.is_none()
                && site.b.ksplit.is_none()
                && site.a.clane == 0
                && site.a.ci == stride
                && site.k.saturating_sub(1) * site.a.ck < stride
                && site.b.ci == 0;
            // `w` is computed before the loop: no in-edge is a loop morphism.
            let invariant = ir.in_edges(b).iter().all(|&e| !in_loop(e));
            if !row_local || !invariant || updated.contains(&morph.target) {
                continue;
            }
            let row = TileSite {
                rows: 1,
                a: TileRead {
                    base: 0,
                    ..site.a.clone()
                },
                ..site.clone()
            };
            out.insert(
                morph.target,
                Retained {
                    merge,
                    start,
                    stride,
                    row,
                },
            );
        }
    }
    out
}

fn proj_source(ir: &CategoryIr, o: ObjectId) -> Option<ObjectId> {
    let m = ir.morphism(*ir.in_edges(o).first()?)?;
    matches!(m.op, Operation::Proj { .. }).then_some(m.source)
}

impl<'a> FnEmit<'a> {
    /// The first-pass flag of the retained site writing `y`, or of the delta
    /// upload of `y` (minted once).
    pub(super) fn retain_flag(&mut self, y: ObjectId) -> String {
        if let Some(f) = self.retain_flags.get(y) {
            return f.clone();
        }
        let f = self.scratch("i1");
        self.retain_flags.insert(y, f.clone());
        f
    }

    /// At loop entry: every site this loop retains starts with a whole pass,
    /// and every array it delta-uploads (`delta-h2d`) with a whole upload.
    pub(crate) fn retain_reset(&mut self, merge: ObjectId) {
        let ups = self.resident.as_ref().map_or_else(Vec::new, |p| {
            p.delta
                .iter()
                .filter(|(_, u)| u.merge == merge)
                .map(|(o, _)| o)
                .collect()
        });
        let ys: Vec<ObjectId> = self
            .retain
            .iter()
            .filter(|(_, r)| r.merge == merge)
            .map(|(y, _)| y)
            .chain(ups)
            .collect();
        for y in ys {
            let f = self.retain_flag(y);
            self.line(format!("store i1 false, ptr {f}"));
        }
    }

    /// `a·var + b` as an `i64`, computed in the counter's own wrapping width
    /// (`var` loaded as `wllt` value `t`) and sign-extended.
    pub(super) fn affine_cell(&mut self, wllt: &str, t: &str, a: &Affine) -> String {
        let at = self.tmp();
        self.line(format!("{at} = mul {wllt} {t}, {}", a.a));
        let start = self.tmp();
        self.line(format!("{start} = add {wllt} {at}, {}", a.b));
        if wllt == "i64" {
            return start;
        }
        let s = self.tmp();
        self.line(format!("{s} = sext {wllt} {start} to i64"));
        s
    }

    /// The retained site: the row the iteration wrote, or the whole site.
    pub(super) fn emit_retained(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        packed: Option<PackedBuffer>,
        r: &Retained,
    ) {
        let flag = self.retain_flag(target);
        let Some((wllt, t)) = self.load_whole(r.start.var) else {
            self.emit_tiled_map_realize(source, target, site, packed);
            return;
        };
        let valid = self.tmp();
        self.line(format!("{valid} = load i1, ptr {flag}"));
        // The cell the program's first `Update` wrote, then the row it is in.
        let start64 = self.affine_cell(&wllt, &t, &r.start);
        let rel = self.tmp();
        self.line(format!("{rel} = sub i64 {start64}, {}", site.a.base));
        let row = self.tmp();
        self.line(format!("{row} = udiv i64 {rel}, {}", r.stride));
        let rem = self.tmp();
        self.line(format!("{rem} = urem i64 {rel}, {}", r.stride));
        let aligned = self.tmp();
        self.line(format!("{aligned} = icmp eq i64 {rem}, 0"));
        // Unsigned: a `rel` below zero reads as huge and fails this too.
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult i64 {row}, {}", site.rows));
        let placed = self.tmp();
        self.line(format!("{placed} = and i1 {aligned}, {inside}"));
        let part = self.tmp();
        self.line(format!("{part} = and i1 {placed}, {valid}"));
        let (one, whole, done) = (self.label(), self.label(), self.label());
        self.line(format!("br i1 {part}, label %{one}, label %{whole}"));

        self.label_line(&whole);
        self.emit_tiled_map_realize(source, target, site, packed.clone());
        self.line(format!("store i1 true, ptr {flag}"));
        self.line(format!("br label %{done}"));

        self.label_line(&one);
        if gpu::gpu_site(&r.row, self.profile) {
            self.emit_tiled_map_gpu_row(source, target, site, &r.row, &row);
        } else {
            let lo = self.tmp();
            self.line(format!("{lo} = mul i64 {row}, {}", site.c));
            let hi = self.tmp();
            self.line(format!("{hi} = add i64 {lo}, {}", site.c));
            self.row_bounds = Some((lo, hi));
            self.emit_tiled_map_realize(source, target, site, packed);
            self.row_bounds = None;
        }
        self.line(format!("br label %{done}"));
        self.label_line(&done);
    }
}
