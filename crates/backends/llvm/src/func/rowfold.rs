//! R0 for folds (S54 `rowfold`, plan-s54 §4.4): a row-local in-order fold
//! ([`mapal_ir::RowFoldSite`]) runs on the device when the array it reads is
//! already there — a device tile site's output — one thread per row, `k` in
//! order. Each row is the oracle's chain (ADR-0028 D4), so the bits are the
//! host fold's; what moves is the transmission: the output no longer crosses
//! the bus whole for the host to fold it.
//!
//! **The byte rule.** A fold over a host array stays on the host: the device
//! would first need the whole array copied in, which is the transmission this
//! rule removes. The fold is placed where its operand already is.
//!
//! **The reader-chosen store order (`rowfold-layout`).** When the fold is the
//! output's *only* reader — its exact use count, a whole-program fact — the
//! order the output is read in is known: row `i` walks `k` fastest and the
//! rows are the fold's threads. So the producing site stores the output
//! column-major, `o[j·rows + i]`, and the fold reads `c[i + k·rows]`:
//! consecutive threads read consecutive words. Nothing else can observe the
//! layout, because nothing else reads the array.

use super::*;
use mapal_ir::{RowFoldSite, TileRead};

/// A row fold placed on the device.
#[derive(Clone, Debug)]
pub(crate) struct RowFold {
    /// The site; its read re-addressed when the producer stores column-major.
    pub site: RowFoldSite,
    /// The producing tile site, when it stores column-major for this fold.
    pub colmajor: Option<MorphismId>,
}

/// Every row fold of `f` that runs on the device, by morphism. Shared with
/// [`super::gpu::device_module`], so the two units agree on every kernel name
/// (the [`super::retain::retained_sites`] precedent).
pub(crate) fn rowfold_sites(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> SecondaryMap<MorphismId, RowFold> {
    let mut out = SecondaryMap::new();
    let rungs = gpu::host_rungs(profile, contract, arms);
    // `resident` is what keeps the operand on the device; a function with a
    // retained site keeps the S52 round trips (`FnEmit::set_gpu_arms`).
    if !rungs.rowfold
        || !rungs.resident
        || profile.gpu().is_none()
        || !retain::retained_sites(ir, f, profile, rungs).is_empty()
    {
        return out;
    }
    let tiles = ir.tile_plan(f);
    let dev = gpu::placement(
        ir,
        f,
        &gpu::tile_plan(ir, f, profile, contract, arms),
        profile,
        rungs.window,
    );
    for (m, site) in ir.row_fold(f).iter() {
        let window = site.w > 1 || site.a.ksplit.is_some();
        if !is_float(&site.elem) || site.k == 0 || (window && !rungs.window) {
            continue;
        }
        let morph = ir.morphism(m).expect("morphism resolves");
        let Some(c) = pair_source_ir(ir, morph.source, site.a.slot) else {
            continue;
        };
        // ponytail: the operand a device site produced; an operand that is on
        // the device for another reason (uploaded for a site, carried) is the
        // same rule — add it when a program folds one.
        let Some((p, tile)) = ir
            .in_edges(c)
            .first()
            .and_then(|&p| Some((p, tiles.sites.get(p)?)))
            .filter(|(p, _)| dev.contains_key(*p))
        else {
            continue;
        };
        let mut fold = site.clone();
        let mut colmajor = None;
        // The fold reads cell (i, k) of the producer's `rows × c` grid, row
        // by row, and is its only reader.
        let in_order = site.rows == tile.rows
            && site.w == 1
            && site.k == tile.c
            && site.a
                == TileRead {
                    base: 0,
                    ci: tile.c,
                    ck: 1,
                    ..site.a.clone()
                };
        let launch = gpu::launch(tile, profile, contract, arms);
        if rungs.rowfold_layout
            && in_order
            && only_reader(ir, c, m, site.a.slot)
            && gpu::col_store(&launch, tile)
        {
            fold.a = TileRead {
                ci: 1,
                ck: tile.rows,
                ..fold.a
            };
            colmajor = Some(p);
        }
        out.insert(
            m,
            RowFold {
                site: fold,
                colmajor,
            },
        );
    }
    out
}

/// `reader` is the only morphism that reads `o`: every edge out of `o` is the
/// `Pair` into `reader`'s source at `slot`, and that product feeds nothing
/// else.
fn only_reader(ir: &CategoryIr, o: ObjectId, reader: MorphismId, slot: u32) -> bool {
    let edges = ir.out_edges(o);
    !edges.is_empty()
        && edges.iter().all(|&e| {
            let pm = ir.morphism(e).expect("morphism resolves");
            matches!(pm.op, Operation::Pair { slot: s, .. } if s == slot)
                && ir.out_edges(pm.target) == [reader]
        })
}

impl<'a> FnEmit<'a> {
    /// The device half of a placed row fold: the operand from its device copy,
    /// one launch, the output kept on the device unless the host reads it
    /// whole (`emit_site_resident`).
    pub(super) fn emit_rowfold(&mut self, m: MorphismId, r: &RowFold) {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let (source, target) = (morph.source, morph.target);
        let lanes = gpu::rowfold_lanes(
            &r.site,
            self.profile,
            self.contract,
            self.gpu_arms,
            self.order,
        );
        let id = gpu::rowfold_id(&r.site, self.profile, lanes) as i64;
        let (blocks, threads) = gpu::rowfold_grid(&r.site, self.profile, lanes);
        self.emit_site_resident(source, target, &[r.site.a.slot], None, |ops, o| {
            let a = &ops[0];
            format!(
                "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {a}, i64 {o}, \
                 i32 {blocks}, i32 1, i32 {threads}, i32 0)"
            )
        });
    }
}
