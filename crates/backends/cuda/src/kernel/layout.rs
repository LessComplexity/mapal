//! Local-array budgets, ABI sizes, buffer bytes and the index/grid expression helpers.

use super::*;

/// The pinned per-thread local-array budget (DESIGN §5's documented cell:
/// "if it exceeds the per-thread local-memory budget the honest cell is a
/// documented `Unsupported`, recorded here and in STATUS — never
/// improvised"). A per-thread local array — a twin's produced local
/// (`{T} oK[{flat}]`, §2 item 8) or the fold kernel's array-acc copy
/// (`{T} acc[{m}];`, BC4) — whose nominal flat byte size exceeds this is
/// rejected, not emitted. 16 KiB sits far above the corpus (sepia's
/// `[Pixel; 16]` is 192 B) and far below the local-memory wall.
pub(crate) const MAX_LOCAL_ARRAY_BYTES: u64 = 16384;

/// The nominal (unpadded) byte size of a flat element: the scalar width;
/// a product sums its surviving components (an array component rides as a
/// handle — pointer width). Erased ⇒ `None` (no storage to budget). This
/// is a deterministic budget measure, not the ABI `sizeof` (padding is the
/// target compiler's business).
pub(super) fn nominal_sizeof(ty: &Ty) -> Option<u64> {
    match ty {
        Ty::Int { bits, .. } | Ty::Float { bits } => Some(*bits as u64 / 8),
        Ty::Narrow(n) => Some(n.bytes()),
        Ty::Bool => Some(1),
        Ty::Unit | Ty::Str | Ty::IoToken => None,
        Ty::Array { .. } => Some(8), // a handle field
        Ty::Tuple(_) | Ty::Struct { .. } => {
            let mut sum = 0u64;
            let mut k = 0u32;
            while let Some(c) = ty.component_ty(k) {
                if lower_ty(c).is_some() {
                    sum += nominal_sizeof(c)?;
                }
                k += 1;
            }
            Some(sum)
        }
    }
}

/// The nominal flat byte size of `ty` as a per-thread local array —
/// `flat_count(ty) * nominal_sizeof(base)` — or `None` if the flat base is
/// erased (no local exists to budget).
pub(crate) fn local_array_bytes(ty: &Ty) -> Option<u64> {
    Some(flat_count(ty) * nominal_sizeof(flat_base(ty))?)
}

/// `x` rounded up to a multiple of `a` (a power of two).
pub(crate) fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

/// The C-layout byte size of `ty`'s lowered form — the arena measure
/// (plan-smart-arenas §3: offsets must be ABI-exact numerics, and
/// [`nominal_sizeof`] explicitly undercounts padded products). Scalars at
/// width; a product lays out its surviving components by C struct rules —
/// each component at its own alignment, the struct tail-padded to its widest
/// component's alignment — except residual-1, which lowers to the **bare**
/// surviving component (no wrapper, no padding). Erased ⇒ `None`. An array
/// ty contributes a handle field (pointer width) — reachable only as a
/// product component; buffer capacities go through [`buffer_bytes_of`] (flat
/// base × flat count). The emitted `static_assert`s (module.rs) pin these
/// sizes against nvcc's `sizeof` on the box leg.
pub(crate) fn abi_sizeof(ty: &Ty) -> Option<u64> {
    match ty {
        Ty::Int { bits, .. } | Ty::Float { bits } => Some(*bits as u64 / 8),
        Ty::Narrow(n) => Some(n.bytes()),
        Ty::Bool => Some(1),
        Ty::Unit | Ty::Str | Ty::IoToken => None,
        Ty::Array { .. } => Some(8), // a handle field
        Ty::Tuple(_) | Ty::Struct { .. } => {
            let mut comps: Vec<&Ty> = Vec::new();
            let mut k = 0u32;
            while let Some(c) = ty.component_ty(k) {
                if lower_ty(c).is_some() {
                    comps.push(c);
                }
                k += 1;
            }
            match comps.as_slice() {
                [] => None,
                [only] => abi_sizeof(only), // residual-1: the bare component
                _ => {
                    let mut off = 0u64;
                    let mut max_align = 1u64;
                    for c in comps {
                        let a = abi_alignof(c)?;
                        off = align_up(off, a) + abi_sizeof(c)?;
                        max_align = max_align.max(a);
                    }
                    Some(align_up(off, max_align)) // tail padding
                }
            }
        }
    }
}

/// The C-layout alignment of `ty`'s lowered form: scalars at width, a handle
/// at pointer width, a product at its widest surviving component's alignment
/// (residual-1: the bare component's). Erased ⇒ `None`.
pub(super) fn abi_alignof(ty: &Ty) -> Option<u64> {
    match ty {
        Ty::Int { bits, .. } | Ty::Float { bits } => Some(*bits as u64 / 8),
        Ty::Narrow(n) => Some(n.bytes()),
        Ty::Bool => Some(1),
        Ty::Unit | Ty::Str | Ty::IoToken => None,
        Ty::Array { .. } => Some(8),
        Ty::Tuple(_) | Ty::Struct { .. } => {
            let mut max_align: Option<u64> = None;
            let mut k = 0u32;
            while let Some(c) = ty.component_ty(k) {
                if lower_ty(c).is_some() {
                    let a = abi_alignof(c)?;
                    max_align = Some(max_align.map_or(a, |m: u64| m.max(a)));
                }
                k += 1;
            }
            max_align
        }
    }
}

/// The F7 budget check: over-budget ⇒ the documented `Unsupported` cell
/// with the measured size in the feature string.
pub(crate) fn check_local_array_budget(ty: &Ty, loc: SourceLoc) -> Result<(), EmitError> {
    if let Some(bytes) = local_array_bytes(ty)
        && bytes > MAX_LOCAL_ARRAY_BYTES
    {
        return Err(EmitError::Unsupported {
            feature: format!(
                "per-thread local array over {MAX_LOCAL_ARRAY_BYTES} bytes ({bytes} bytes)"
            ),
            loc,
        });
    }
    Ok(())
}

/// The launch-form side of the F7 budget: a launch-form `Fold` with an
/// array-typed acc copies it into a per-thread local in the single-thread
/// kernel (`{ct} acc[{m}];`, BC4) — check every fold site. (The inline
/// form's array acc lives in the twin's produced local — checked there.)
pub(crate) fn check_fold_acc_budgets(ir: &CategoryIr) -> Result<(), EmitError> {
    for (f, _) in ir.funcs() {
        for site in collect_sites(ir, f) {
            let morph = ir.morphism(site.m).expect("morphism resolves");
            let Operation::Fold { captures, .. } = morph.op else {
                continue;
            };
            let src_ty = &ir.object(morph.source).expect("source resolves").ty;
            // ADR-0027: the acc is the source product's component k.
            let acc_ty = src_ty.component_ty(captures).expect("fold acc");
            if matches!(acc_ty, Ty::Array { .. }) {
                check_local_array_budget(acc_ty, morph.loc)?;
            }
        }
    }
    Ok(())
}

/// The lowered C++ type of `ty`'s flat base element.
pub(crate) fn flat_base_ct(ty: &Ty) -> Option<String> {
    lower_ty(flat_base(ty))
}

/// `(elem_ty, size)` of an `Array` ty (cloned — the llvm `array_parts`).
pub(crate) fn array_parts(ty: &Ty) -> (Ty, u64) {
    match ty {
        Ty::Array { elem, size } => ((**elem).clone(), *size),
        _ => unreachable!("expected an array ty"),
    }
}

/// The cudaMalloc byte-count text for a device buffer holding `ty` (an array
/// ty): `sizeof({base}) * {flat}ULL`.
pub(crate) fn buffer_bytes(ty: &Ty) -> String {
    format!(
        "sizeof({}) * {}ULL",
        flat_base_ct(ty).expect("array base lowers"),
        flat_count(ty)
    )
}

/// The arena-capacity measure of a device buffer holding `ty` (an array ty):
/// the ABI-exact flat base size × the flat count — the numeric twin of
/// [`buffer_bytes`]'s text (`sizeof(base) * flat`, pinned equal by the
/// emitted `static_assert`s on the box leg).
pub(crate) fn buffer_bytes_of(ty: &Ty) -> Option<u64> {
    Some(abi_sizeof(flat_base(ty))? * flat_count(ty))
}

/// The source object of the `Pair{slot==k}` edge feeding aggregate `agg`.
/// Whether `e` needs no parentheses when nested into a larger expression:
/// an identifier / literal / member path (`in.f3`, `o7`), or one already
/// wrapped by a single balanced outer `(...)` pair (WP-B/WP-C inlining).
pub(crate) fn is_atomic_expr(e: &str) -> bool {
    if !e.is_empty()
        && e.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return true;
    }
    if e.starts_with('(') && e.ends_with(')') {
        let mut depth = 0i32;
        for (i, c) in e.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return i == e.len() - 1;
                    }
                }
                _ => {}
            }
        }
    }
    false
}

pub(crate) fn pair_source(ir: &CategoryIr, agg: ObjectId, k: u32) -> Option<ObjectId> {
    for &m in ir.in_edges(agg) {
        let morph = ir.morphism(m).expect("morphism resolves");
        if let Operation::Pair { slot, .. } = morph.op
            && slot == k
        {
            return Some(morph.source);
        }
    }
    None
}

/// The element-load expression for index `i` of a flat buffer `base` whose
/// element ty is `elem`: `base[i]` for scalars/products (AoS); the sub-array
/// pointer `base + (i * m)` for array elements (the flat-nested seam, §5).
pub(super) fn elem_expr(base: &str, i: &str, elem: &Ty) -> String {
    if matches!(elem, Ty::Array { .. }) {
        format!("({base} + ({i} * {}ULL))", flat_count(elem))
    } else {
        format!("{base}[{i}]")
    }
}

/// The width-rule extension (§3): `(int64_t){expr}` — value-preserving for
/// both signed and unsigned operands, realizing llvm's zext/sext split.
pub(crate) fn extend_index(expr: &str) -> String {
    format!("(int64_t){expr}")
}

/// The signed two-sided bounds guard (§3): `idx < 0 || idx >= (int64_t)n`.
pub(super) fn bounds_cond(idx64: &str, n: u64) -> String {
    format!("{idx64} < 0 || {idx64} >= (int64_t){n}")
}

/// The grid expression for an `n`-element elementwise kernel (BC3, 64-bit).
pub(crate) fn grid_expr(n: u64) -> String {
    format!("(unsigned int)(({n}ULL + 255ULL) / 256ULL)")
}

/// The number of `Pair` in-edges per array-typed object of `f` — the
/// pack_array literal sites. The construction is emitted at the object's
/// **last** Pair edge in `topo_order`: every element's producer precedes its
/// Pair edge, and every use of the literal follows all of them (topo
/// guarantee), so the last edge is the one point where all elements are
/// materialized and no use has happened.
pub(crate) fn literal_pair_counts(ir: &CategoryIr, f: FuncId) -> SecondaryMap<ObjectId, usize> {
    let mut out: SecondaryMap<ObjectId, usize> = SecondaryMap::new();
    let fd = ir.func(f).expect("func resolves");
    for &m in &fd.morphisms {
        let morph = ir.morphism(m).expect("morphism resolves");
        if let Operation::Pair { .. } = morph.op {
            let tty = &ir.object(morph.target).expect("target resolves").ty;
            if matches!(tty, Ty::Array { .. }) {
                let cur = out.get(morph.target).copied().unwrap_or(0);
                out.insert(morph.target, cur + 1);
            }
        }
    }
    out
}
