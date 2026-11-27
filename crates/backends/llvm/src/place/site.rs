//! The site as the model sees it ([`site_of`]) and the scopes that warm, column or order it.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

// ---- the site ---------------------------------------------------------------

/// The site as placement reads it: extents, operand bytes as stored, the
/// fold's freedom, whether `k` may be cut (no window or mixed-radix
/// operand), the formats its precision set admits.
pub fn site_of(site: &TileSite, contract: bool, order: crate::Order) -> Site {
    let e = elem_bytes_of(&site.elem);
    let mut fmts = vec![if e == 8 { Fmt::F64 } else { Fmt::F32 }];
    if site.tf32 {
        fmts.push(Fmt::Tf32);
    }
    match site.half {
        Some(mapal_ir::NarrowFloat::Bf16) => fmts.push(Fmt::Bf16),
        Some(mapal_ir::NarrowFloat::F16) => fmts.push(Fmt::F16),
        _ => {}
    }
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some();
    let _ = contract;
    let warm = warm_of(site);
    Site {
        m: site.rows,
        n: site.c,
        k: site.k,
        batch: 1,
        ea: site.a.elem_bytes(&site.elem),
        eb: site.b.elem_bytes(&site.elem),
        eo: e,
        epi: 0,
        fold: match order {
            crate::Order::Free => Fold::Free,
            crate::Order::Fixed => Fold::Fixed,
            crate::Order::Sequential => Fold::Sequential,
        },
        kcut: site.radix.is_none() && !window,
        fmts,
        warm,
    }
}

thread_local! {
    /// The emission's tile sites' warm operands ([`WarmScope`]), by the
    /// site's text; `None` outside an emission.
    pub(super) static WARM: std::cell::RefCell<Option<std::collections::HashMap<String, [bool; 2]>>> =
        const { std::cell::RefCell::new(None) };
}

/// **A site's warm operands** (ADR-0050 D3): `[a, b]` resident in the
/// device cache when the site reads them — `residence_plan` at the part's
/// keep: the working set from the operand's producer (a program input:
/// the call's start, so the previous call's bytes) to its last reader fits
/// ([`warm_sites`]; the fact `MAPAL_SITE_COLD` reads, its complement).
/// Read from the emission's [`WarmScope`]; none outside one.
pub fn warm_of(site: &TileSite) -> [bool; 2] {
    WARM.with(|w| {
        w.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{site:?}")).copied())
            .unwrap_or([false, false])
    })
}

/// Every tile site of `ir` (the IR's tile plans: affine, strided, narrow)
/// and its warm operands at a cache keeping `keep` bytes: an operand is warm
/// where `residence_plan` places its bytes on chip or in the cache.
pub fn warm_sites(ir: &mapal_ir::CategoryIr, keep: u64) -> Vec<(TileSite, [bool; 2])> {
    let mut out = Vec::new();
    for (f, _) in ir.funcs() {
        let res = ir.residence_plan(f);
        for plan in [
            ir.tile_plan(f),
            ir.tile_plan_strided(f),
            ir.tile_plan_narrow(f),
        ] {
            for (m, site) in plan.sites.iter() {
                let source = ir.morphism(m).expect("a site's morphism").source;
                let warm = |slot: u32| {
                    crate::func::pair_source_ir(ir, source, slot)
                        .is_some_and(|o| res.at(o, keep) != mapal_ir::Residence::Dram)
                };
                out.push((site.clone(), [warm(site.a.slot), warm(site.b.slot)]));
            }
        }
    }
    out
}

/// **The warm operands of an emission's sites, while it runs**: every
/// launch of the emission (its plans, its kernels, its host) reads one map,
/// so a site is placed alike wherever its launch is derived. Sites of one
/// text in several places take the conjunction (warm only where every one
/// is). Only under `machine-place` (the map is the search's input alone).
pub struct WarmScope {
    prev: Option<std::collections::HashMap<String, [bool; 2]>>,
}

impl WarmScope {
    pub fn new(ir: &mapal_ir::CategoryIr, p: &TargetProfile, arms: crate::GpuArms) -> Self {
        let map = match (p.gpu(), arms.on.machine_place) {
            (Some(g), true) => {
                let mut m: std::collections::HashMap<String, [bool; 2]> = Default::default();
                for (site, w) in warm_sites(ir, g.l2_keep_bytes()) {
                    let e = m.entry(format!("{site:?}")).or_insert(w);
                    e[0] &= w[0];
                    e[1] &= w[1];
                }
                Some(m)
            }
            _ => None,
        };
        WarmScope {
            prev: WARM.with(|w| w.replace(map)),
        }
    }
}

thread_local! {
    /// The tile sites whose output the emission stores column-major
    /// ([`ColScope`]); `None` outside an emission under `machine-place`.
    pub(super) static COLMAJOR: std::cell::RefCell<Option<std::collections::HashSet<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// **The sites an emission stores column-major, while it runs** (the row
/// folds' `colmajor`: a producing tile a fold reads in order, the rules'
/// base launch able to store so): a pick that cannot store column-major is
/// dropped for such a site alone — the layout is the site's, decided before
/// any placement, never the rules' launch's capability (R3: a 4096 x 8192 x
/// 8192 TF32 site stores row-major, yet its rules' FMA tile could store
/// column-major, so the filter on the capability dropped the tensor pick and
/// no kernel carried a place line). Only under `machine-place`.
pub struct ColScope {
    prev: Option<std::collections::HashSet<String>>,
}

impl ColScope {
    pub fn new(
        ir: &mapal_ir::CategoryIr,
        p: &TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
    ) -> Self {
        let set = match (p.gpu(), arms.on.machine_place) {
            (Some(_), true) => {
                let mut set = std::collections::HashSet::new();
                for (f, _) in ir.funcs() {
                    let tiles = ir.tile_plan(f);
                    for (_, fold) in crate::func::rowfold::rowfold_sites(ir, f, p, contract, arms) {
                        if let Some(t) = fold.colmajor.and_then(|m| tiles.sites.get(m)) {
                            set.insert(format!("{t:?}"));
                        }
                    }
                }
                Some(set)
            }
            _ => None,
        };
        ColScope {
            prev: COLMAJOR.with(|c| c.replace(set)),
        }
    }
}

impl Drop for ColScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        COLMAJOR.with(|c| *c.borrow_mut() = prev);
    }
}

/// Whether `site`'s output is stored column-major: the emission's
/// [`ColScope`], else (no scope: a launch derived outside an emission) the
/// rules' launch `rules` able to store so — the conservative reading.
pub(super) fn stores_col(site: &TileSite, rules: &Launch) -> bool {
    COLMAJOR.with(|c| match &*c.borrow() {
        Some(set) => set.contains(&format!("{site:?}")),
        None => gpu::col_store(rules, site),
    })
}

thread_local! {
    /// The emission's order ([`OrderScope`]): the order its host launches
    /// place each site under.
    pub(super) static ORDER: std::cell::Cell<Option<crate::Order>> = const { std::cell::Cell::new(None) };
}

/// The emission's order for [`placed_for`]: the kernel of a realization is
/// the pick of the order the host placed the site under, whichever of the
/// two is emitted first (cost-rental: the H100's fixed-order 16 x 2048 x
/// 7168 cell launched the pick's split grid over the rules' kernel —
/// wrong values — where the kernel text came before the host's search).
pub struct OrderScope {
    prev: Option<crate::Order>,
}

impl OrderScope {
    pub fn new(order: crate::Order) -> Self {
        OrderScope {
            prev: ORDER.with(|o| o.replace(Some(order))),
        }
    }
}

impl Drop for OrderScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ORDER.with(|o| o.set(prev));
    }
}

impl Drop for WarmScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        WARM.with(|w| *w.borrow_mut() = prev);
    }
}
