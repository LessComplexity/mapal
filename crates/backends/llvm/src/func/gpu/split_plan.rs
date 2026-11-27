//! Split realizations of an `{f32}` product, the 16-bit layouts and the kernel ids.

use super::*;

pub(crate) fn elem_bytes_of(elem: &Ty) -> u64 {
    match elem {
        Ty::Float { bits } => u64::from(*bits) / 8,
        _ => 4,
    }
}

/// The site as its kernel ids key it: its `Debug` text, a plain site's
/// without the `radix` field it lacks (ids from before `radix-tile` stand).
pub(super) fn site_key(site: &TileSite) -> String {
    format!("{site:?}")
        .replace(", radix: None", "")
        .replace(", tf32: false", "")
        .replace(", half: None", "")
        .replace(", post: None", "")
}

/// ADR-0041 D6: the reduced realization a site's products admit — `tf32` in
/// `precision_plan.set` of its product ([`TileSite::tf32`]), or a 16-bit
/// member ([`TileSite::half`], ADR-0043 D4, the weaker) — keyed from the
/// program, never a flag (D7).
pub(crate) fn site_reduced(site: &TileSite) -> Option<Reduced> {
    site.half
        .map(Reduced::Half)
        .or(site.tf32.then_some(Reduced::Tf32))
}

/// tensor-cores `narrow-async`: the FMA tile streams a narrow `b` panel in
/// its own format — an 8-byte `cp.async` per 4-element vector (a 2-byte copy
/// has no `cp.async`), widened as each fragment is read from shared memory —
/// where `b` is the one narrow operand, its rows unit-stride and vector
/// aligned, and the rules stream the panel (`async`). A narrow `a` (the
/// transposed panel's scatter) stays register-staged.
pub(super) fn narrow_streams(site: &TileSite, rungs: &crate::GpuRungs) -> bool {
    rungs.narrow_async
        && site.a.narrow.is_none()
        && site.b.narrow.is_some()
        && vector_b(site)
        && site.radix.is_none()
}

/// **The 16-bit matrix product a site's operands admit** (tensor-cores,
/// ADR-0043 D4): the format and each panel's storage width in shared
/// memory, or `None`.
///
/// - A site whose product admits a 16-bit member ([`TileSite::half`]) takes
///   it where the part lists a warp product of that format whose
///   accumulator meets the set's `u_a = 2⁻²³` ([`profile::MatrixOp::acc_meets`]):
///   a narrow operand of that format is copied as is (2 bytes), an f32
///   operand is rounded once to `bf16` (4 bytes, rounded as its fragment is
///   read). An `f16` member rounds no f32 operand: D4 admits it only where a
///   value-range fact proves the operand below 65504, and none is stated yet.
/// - A `{f32}` product of two operands of one narrow format is exact on the
///   tensor core; it takes the instruction only where its accumulator meets
///   the default face's `u_a = 2⁻²⁴` (every measured part truncates:
///   `mma_calib.cu`, so none does).
///
/// Each panel's rows must be 16-byte copies: unit stride along `k` (`a`;
/// `b` stored `n × k`, [`GpuMmaTile::bt`]) or the lane (`b` stored `k × n`),
/// stride and base whole 16-byte chunks of its storage.
/// f32-tensor-cores `split-tile`: the split form a `{f32}` tile site's
/// operands take, the instruction's format and each panel's storage bytes —
/// `Narrow1` where both are stored in one narrow format (their exact
/// widenings: one exact product a k-step), on that format's warp product
/// with [`half_operands`]' panel layouts; `Tf32x3` where both are f32, on
/// the TF32 warp product with the TF32 tile's 16-byte panels; `Bf16x3n`
/// where one operand is stored `bf16` and the other f32 (tensor-cores-2:
/// bf16 weights against f32 activations) — the f32 one cut into three exact
/// bf16 parts as its fragment is read, the narrow one its exact widening,
/// on the bf16 warp product (its panels at their storage widths, 4 and 2).
/// `None`: a reduced product set (its member's realization), an f32 × `f16`
/// pair (no three-part `f16` cut covers f32's range), or no such
/// instruction.
pub(super) fn split_operands(
    site: &TileSite,
    gpu: &Gpu,
    least: bool,
) -> Option<(SplitKind, MmaFmt, u64, u64)> {
    if site.elem != Ty::f32() || site.tf32 || site.half.is_some() {
        return None;
    }
    match (site.a.narrow, site.b.narrow) {
        (Some(mapal_ir::NarrowFloat::Bf16), None) | (None, Some(mapal_ir::NarrowFloat::Bf16)) => {
            let op = gpu.caps.matrix_op(MmaFmt::Bf16, Home::Regs)?;
            if op.shape != MMA_SYNC_16 {
                return None;
            }
            let (ea, eb) = mma16_layout(site, MmaFmt::Bf16)?;
            // `split-least`: two parts where the bound covers their split
            // error at this `K`, else the exact three.
            let two = least
                && SplitForm::of(SplitKind::Bf16x2n)
                    .admits(site.k, 1, op.shape.k, op.acc_model(), 1)
                    .is_some();
            let kind = if two {
                SplitKind::Bf16x2n
            } else {
                SplitKind::Bf16x3n
            };
            Some((kind, MmaFmt::Bf16, ea, eb))
        }
        (Some(x), Some(y)) if x == y => {
            let fmt = MmaFmt::of_narrow(x);
            // The split tile issues the 16-bit warp product's geometry
            // (m16n8k16, two-byte fragments): a format whose warp product
            // has another (fp8's m16n8k32) has no split kernel.
            let op = gpu.caps.matrix_op(fmt, Home::Regs)?;
            if op.shape != MMA_SYNC_16 {
                return None;
            }
            let (ea, eb) = mma16_layout(site, fmt)?;
            Some((SplitKind::Narrow1, fmt, ea, eb))
        }
        (None, None) => {
            gpu.caps.matrix_op(MmaFmt::Tf32, Home::Regs)?;
            (vector_a(site) && vector_b(site)).then_some((SplitKind::Tf32x3, MmaFmt::Tf32, 4, 4))
        }
        _ => None,
    }
}

/// tensor-cores-2: **a split tile's flush depth** — the k-steps each
/// tile's zeroed chain carries before its round-to-nearest add into the f32
/// accumulator: the deepest the form's bound admits at the site's `K` and
/// pieces ([`SplitForm::admits`], a power of two), at most one panel's
/// k-steps (the panel loop is unrolled, so a flush falls inside it). 1 off
/// a split form. Each flush is `4·mt·nt` FADDs a thread: at one a k-step
/// they were as many issue slots as the three products' (sm_80 SASS).
pub(crate) fn split_flush(g: &GpuMmaTile, k: u64, pieces: u64, gpu: &Gpu) -> u64 {
    let (Some(kind), Some(op)) = (g.split, gpu.caps.matrix_op(g.fmt, Home::Regs)) else {
        return 1;
    };
    SplitForm::of(kind)
        .admits(k, pieces, op.shape.k, op.acc_model(), g.bk / g.mma_k())
        .unwrap_or(1)
}

/// The least `|x|` bits whose TF32 `rna` rounds to infinity (`2¹²⁸ − 2¹⁰³`).
pub(crate) const TF32_RNA_OVERFLOW: u32 = 0x7f7f_f000;
/// The least `|x|` bits whose bf16 `rn` (ties to even) rounds to infinity.
pub(crate) const BF16_RN_OVERFLOW: u32 = 0x7f7f_8000;

/// The parts of the `a` and `b` fragments a split tile's thread holds: an
/// f32 panel (4-byte elements) is cut into the form's parts, a narrow one is
/// its own exact widening.
pub(crate) fn split_parts(kind: SplitKind, ea: u64, eb: u64) -> (u64, u64) {
    let f = SplitForm::of(kind);
    let p = u64::from(f.parts_a.max(f.parts_b));
    let parts = |e: u64| if e == 4 { p } else { 1 };
    (parts(ea), parts(eb))
}

/// The plan line's name of a split form.
pub(crate) fn split_name(k: SplitKind) -> &'static str {
    match k {
        SplitKind::Tf32x3 => "tf32x3",
        SplitKind::Bf16x6 => "bf16x6",
        SplitKind::Bf16x3n => "bf16x3n",
        SplitKind::Bf16x2n => "bf16x2n",
        SplitKind::Narrow1 => "narrow1",
    }
}

/// **f32-tensor-cores R-F: whether a split tile `m` realizes the site.**
/// 1. *bound* — always: the form's coefficient at the site's `K` meets D2
///    ([`SplitForm::admits`], the accumulator's measured guard bits);
/// 2. *part* — the matrix rate the tile realizes at its instruction issue
///    ([`split_issue`]: its cuts, flushes and products a k-step against the
///    matrix pipe) pays for the form's `P` passes against the FFMA rate
///    ([`crate::rules::split_rate_pays`]);
/// 3. *site* — the flops per operand byte reach the part's FFMA balance
///    (a DRAM-bound site keeps the FFMA tile, which splits `k` to fill the
///    part) or the tile's grid fills the warp schedulers (`half-fill`'s
///    rule).
///
/// `forced` (`--gpu-on=split-tile`): the bound alone (the measurement arm).
pub(super) fn split_pays(
    site: &TileSite,
    gpu: &Gpu,
    kind: SplitKind,
    m: &GpuMmaTile,
    forced: bool,
    issue: bool,
) -> bool {
    let Some(op) = gpu.caps.matrix_op(m.fmt, Home::Regs) else {
        return false;
    };
    let form = SplitForm::of(kind);
    if form
        .admits(site.k, 1, op.shape.k, op.acc_model(), 1)
        .is_none()
    {
        return false;
    }
    if forced {
        return true;
    }
    if !crate::rules::split_rate_pays(
        gpu,
        &op,
        form.products(),
        split_issue(site, gpu, m, kind, issue),
    ) {
        return false;
    }
    let bytes = site.rows * site.k * m.ea + site.k * site.c * m.eb + site.rows * site.c * 4;
    let balanced = gpu
        .ffma_per_dram_byte_pm()
        .is_none_or(|b| 2 * site.rows * site.c * site.k * 1000 >= b * bytes);
    balanced || (site.rows / m.bm()) * (site.c / m.bn()) * m.warps_x * m.warps_y >= gpu.fill_warps()
}

/// tensor-cores-2 `split-issue`: the split tile `m`'s issue ratio against
/// its matrix pipe ([`crate::rules::split_issue_pm`]) — its thread's cut
/// elements a k-step (an f32 `a` fragment's `k/2` of `m16 × k` a row tile,
/// an f32 `b` fragment's `k/4` a column tile), at the cut's measured slots
/// and the flush the bound admits; `issue` off: the 5fcb401b realization
/// (the `cvt.rna` cut, a flush every k-step).
pub(super) fn split_issue(
    site: &TileSite,
    gpu: &Gpu,
    m: &GpuMmaTile,
    kind: SplitKind,
    issue: bool,
) -> u64 {
    split_cost(site, gpu, m, kind, issue).0
}

/// [`split_issue`] and the slots a matrix instruction it prices
/// ([`crate::rules::split_slots_pm`]), both per mille.
pub(super) fn split_cost(
    site: &TileSite,
    gpu: &Gpu,
    m: &GpuMmaTile,
    kind: SplitKind,
    issue: bool,
) -> (u64, u64) {
    let Some(op) = gpu.caps.matrix_op(m.fmt, Home::Regs) else {
        return (u64::MAX, u64::MAX);
    };
    let k = op.shape.k;
    let cut = u64::from(m.ea == 4) * m.mt * k / 2 + u64::from(m.eb == 4) * m.nt * k / 4;
    let per_cut = match (kind, issue) {
        (SplitKind::Tf32x3, true) => crate::rules::SPLIT_CUT_TF32,
        (SplitKind::Tf32x3, false) => crate::rules::SPLIT_CUT_TF32_CVT,
        (SplitKind::Bf16x2n, _) => crate::rules::SPLIT_CUT_BF16X2,
        _ => crate::rules::SPLIT_CUT_BF16X3,
    };
    let flush = match issue {
        true => split_flush(m, site.k, 1, gpu),
        false => 1,
    };
    let products = SplitForm::of(kind).products();
    (
        crate::rules::split_issue_pm(gpu, &op, products, (m.mt, m.nt), cut, per_cut, flush),
        crate::rules::split_slots_pm(products, (m.mt, m.nt), cut, per_cut, flush),
    )
}

/// The 16-bit tile's panel widths for operands of `fmt` ([`half_operands`]'
/// layout rule): each panel's rows whole 16-byte copies.
pub(super) fn mma16_layout(site: &TileSite, fmt: MmaFmt) -> Option<(u64, u64)> {
    let width = |r: &mapal_ir::TileRead| match r.narrow {
        Some(n) if MmaFmt::of_narrow(n) == fmt => Some(2),
        Some(_) => None,
        None if fmt == MmaFmt::Bf16 => Some(4),
        None => None,
    };
    let (ea, eb) = (width(&site.a)?, width(&site.b)?);
    let lanes = |e: u64| GPU_VEC_BYTES / e;
    let rows_a = site.a.ck == 1
        && site.a.ci.is_multiple_of(lanes(ea))
        && site.a.base.is_multiple_of(lanes(ea));
    let rows_b = site.b.base.is_multiple_of(lanes(eb))
        && match (site.b.clane, site.b.ck) {
            (1, ck) => ck.is_multiple_of(lanes(eb)),
            (cl, 1) => cl.is_multiple_of(lanes(eb)),
            _ => false,
        };
    (rows_a && rows_b).then_some((ea, eb))
}

pub(super) fn half_operands(site: &TileSite, gpu: &Gpu) -> Option<(MmaFmt, u64, u64)> {
    if site.elem != Ty::f32() {
        return None;
    }
    let (fmt, u_a) = match site.half {
        Some(n) => (MmaFmt::of_narrow(n), -23),
        None => match (site.a.narrow, site.b.narrow) {
            (Some(x), Some(y)) if x == y => (MmaFmt::of_narrow(x), -24),
            _ => return None,
        },
    };
    let op = gpu.caps.matrix_op(fmt, Home::Regs)?;
    if !op.acc_meets(u_a) {
        return None;
    }
    let width = |r: &mapal_ir::TileRead| match r.narrow {
        Some(n) if MmaFmt::of_narrow(n) == fmt => Some(2),
        Some(_) => None,
        None if fmt == MmaFmt::Bf16 => Some(4),
        None => None,
    };
    let (ea, eb) = (width(&site.a)?, width(&site.b)?);
    let lanes = |e: u64| GPU_VEC_BYTES / e;
    let rows_a = site.a.ck == 1
        && site.a.ci.is_multiple_of(lanes(ea))
        && site.a.base.is_multiple_of(lanes(ea));
    let rows_b = site.b.base.is_multiple_of(lanes(eb))
        && match (site.b.clane, site.b.ck) {
            (1, ck) => ck.is_multiple_of(lanes(eb)),
            (cl, 1) => cl.is_multiple_of(lanes(eb)),
            _ => false,
        };
    (rows_a && rows_b).then_some((fmt, ea, eb))
}

/// A stable id for a site's kernel: FNV-1a over everything the kernel text
/// depends on. The host passes it to the runtime, which looks the kernel up as
/// `mapal_tile_<id>` — so the two translation units agree without sharing state.
pub(crate) fn kernel_id(site: &TileSite, profile: &TargetProfile, contract: bool) -> u64 {
    kernel_id_with(site, profile, contract, None)
}

/// [`kernel_id`] of a site whose store evaluates `epi` (S54): the key also
/// names the group — its bodies, how each slot is fed, what is stored.
pub(crate) fn kernel_id_with(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    epi: Option<&EpiKernel>,
) -> u64 {
    let mut key = format!(
        "{}|{}|{contract}|{:?}",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
    if let Some(e) = epi {
        key.push_str(&format!("|{e:?}"));
    }
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// [`kernel_id`] of the site's kernel storing its output column-major
/// (`rowfold-layout`, [`col_store`]); the row-major id unchanged.
pub(crate) fn kernel_id_col(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    col: bool,
) -> u64 {
    if !col {
        return kernel_id(site, profile, contract);
    }
    let key = format!(
        "{}|{}|{contract}|{:?}|col",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// S62/S63: the ids of a site's combine kernels — `part` is `splitk` (the
/// partial folds into the workspace), `reduce` (its reduce), `atomic` (the
/// partial folds added into the output) or `streamk` — the `col` precedent.
/// The ordered kernel keeps [`kernel_id`], so a function that does not split
/// the site (non-resident, an epilogue, a stream) launches that.
pub(crate) fn kernel_id_split(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    part: &str,
) -> u64 {
    let key = format!(
        "{}|{}|{contract}|{:?}|{part}",
        site_key(site),
        profile.name,
        site_reduced(site)
    );
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Can this launch store its output column-major (`o[j·rows + i]`)? The FMA
/// tile writes `lanes` consecutive rows of one column per vector, which needs
/// whole vectors of rows (`tm % lanes == 0`); the one-thread-per-cell kernel
/// and the `cell-fill` tile (`tn < lanes`, the kernel's `narrow` store) one
/// word. A `fill-rows` tile (`tm < lanes ≤ tn`) stores row vectors: row-major.
/// ponytail: the tensor-core kernels keep row-major — no program reads a TF32
/// site through a fold; add their store when one does.
pub(crate) fn col_store(l: &Launch, site: &TileSite) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&site.elem);
    l.mma.is_none()
        && l.wgmma.is_none()
        && site.radix.is_none()
        && l.tiled
            .is_none_or(|g| g.tm.is_multiple_of(lanes) || g.tn < lanes)
        && site.rows.is_multiple_of(lanes)
}
