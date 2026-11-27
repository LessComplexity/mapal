//! Tests of the GPU tile realization.

use super::*;
use crate::GpuRungs;
use crate::profile::{H100, H100_PCIE, RTX_4090, RTX_5090};
use mapal_ir::TileRead;
use proptest::prelude::*;

/// The tests' call shape from before ADR-0041 D6: `reduced` is now the
/// site's own product set ([`TileSite::tf32`]), so it rides on the site.
pub(super) fn with_set(site: &TileSite, reduced: Option<Reduced>) -> TileSite {
    TileSite {
        tf32: reduced.is_some(),
        ..site.clone()
    }
}

pub(super) fn launch(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
) -> Launch {
    super::launch(&with_set(site, reduced), p, contract, arms)
}

pub(super) fn launch_in(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
    order: Order,
) -> Launch {
    super::launch_in(&with_set(site, reduced), p, contract, arms, order)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kernel_text(
    name: &str,
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    reduced: Option<Reduced>,
    col: bool,
    epi: Option<&EpiKernel>,
    combine: Combine,
) -> String {
    super::kernel_text(
        name,
        &with_set(site, reduced),
        p,
        contract,
        arms,
        col,
        epi,
        combine,
    )
}

/// A `m × n × k` site as `benches/gpu/gen_tiles.py` writes it: `a[i·K + k]`
/// (or column-major `a[k·M + i]`), `b[k·N + j]` (or `b[j·K + k]`).
pub(super) fn site(m: u64, n: u64, k: u64, a_rm: bool, b_rm: bool, elem: Ty) -> TileSite {
    let read = |slot, ci, ck, clane| TileRead {
        slot,
        base: 0,
        ci,
        ck,
        clane,
        ksplit: None,
        narrow: None,
    };
    let a = if a_rm {
        read(0, k, 1, 0)
    } else {
        read(0, 1, m, 0)
    };
    let b = if b_rm {
        read(1, 0, n, 1)
    } else {
        read(1, 0, 1, k)
    };
    let seed = if elem == Ty::f32() {
        Value::F32(0.0)
    } else {
        Value::F64(0.0)
    };
    TileSite {
        rows: m,
        c: n,
        k,
        a,
        b,
        seed,
        elem,
        mul_a_first: true,
        add_acc_first: true,
        radix: None,
        tf32: false,
        half: None,
        post: None,
    }
}

/// Every S52 box site: the four timed squares and the six differential
/// programs.
pub(super) fn s52_sites() -> Vec<TileSite> {
    let f32 = Ty::f32;
    let mut v: Vec<TileSite> = [1024, 2048, 4096, 8192]
        .map(|n| site(n, n, n, true, true, f32()))
        .into();
    v.extend([
        site(256, 256, 256, true, true, f32()),
        site(128, 384, 96, true, true, f32()),
        site(256, 256, 128, false, true, f32()),
        site(256, 256, 128, true, false, f32()),
        site(100, 100, 100, true, true, f32()),
        site(128, 128, 128, true, true, Ty::f64()),
    ]);
    v
}

/// Arms that emit exactly the rung set `r`, whatever the profile's table.
pub(super) fn force(r: GpuRungs) -> crate::GpuArms {
    let mut off = GpuRungs::ALL;
    for n in GpuRungs::NAMES {
        *off.rung(n).expect("a listed name") = !r.get(n);
    }
    off.arch = true;
    crate::GpuArms {
        on: GpuRungs::default(),
        off,
        splitk: 0,
        cands: false,
        free: false,
        ..crate::GpuArms::default()
    }
}

pub(super) const PARTS: [&TargetProfile; 4] = [&RTX_4090, &H100_PCIE, &H100, &RTX_5090];

mod t_arith;
mod t_kernels;
mod t_plans;

use self::t_arith::arms;
use self::t_kernels::off;
