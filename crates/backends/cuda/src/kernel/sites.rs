//! Bulk-op sites: element types, the device-product-array check, site collection and host reachability.

use super::*;

// ---------------------------------------------------------------------------
// F3's recorded Unsupported cell: arrays embedded in products on device
// ---------------------------------------------------------------------------

/// The element tys a launch-form bulk-op site's kernel loads from / stores
/// into global memory (the F3 cell's domain): a product-with-array element
/// would put a local-memory or buffer-interior pointer into a global buffer.
pub(super) fn site_element_tys(ir: &CategoryIr, morph: &Morphism) -> Vec<Ty> {
    let ty_of = |o: ObjectId| ir.object(o).expect("object resolves").ty.clone();
    let elem_of = |t: &Ty| match t {
        Ty::Array { elem, .. } => Some((**elem).clone()),
        _ => None,
    };
    let src = ty_of(morph.source);
    let tgt = ty_of(morph.target);
    let mut out: Vec<Ty> = Vec::new();
    match morph.op {
        Operation::Map { captures, .. } => {
            // ADR-0027: k>0 ⇒ the mapped array is the source product's last
            // component; the leading captures ride as kernel parameters
            // (ordinary dataflow — never an escaping element type).
            let arr = if captures == 0 {
                Some(&src)
            } else {
                src.component_ty(captures)
            };
            if let Some(at) = arr {
                out.extend(elem_of(at));
            }
            out.extend(elem_of(&tgt));
        }
        Operation::Enumerate => {
            out.extend(elem_of(&src));
            out.extend(elem_of(&tgt));
        }
        Operation::Iota | Operation::Fill => {
            out.extend(elem_of(&tgt));
        }
        Operation::Zip => {
            // The source is the (A, B) pair of input arrays; the target's
            // element is the (EA, EB) product the kernel stores.
            for k in 0..2 {
                if let Some(at) = src.component_ty(k) {
                    out.extend(elem_of(at));
                }
            }
            out.extend(elem_of(&tgt));
        }
        Operation::Update => {
            // The array's element and the replacement value.
            if let Some(at) = src.component_ty(0) {
                out.extend(elem_of(at));
            }
            if let Some(vt) = src.component_ty(2) {
                out.push(vt.clone());
            }
        }
        Operation::Index => {
            // The array's element IS the result (device buffer or readback).
            if let Some(at) = src.component_ty(0) {
                out.extend(elem_of(at));
            }
        }
        Operation::Fold { captures, .. } => {
            // The accumulator and the array's element (ADR-0027: the
            // captures shift them to components k and k+1).
            if let Some(acc) = src.component_ty(captures) {
                out.push(acc.clone());
            }
            if let Some(at) = src.component_ty(captures + 1) {
                out.extend(elem_of(at));
            }
        }
        _ => {}
    }
    out
}

/// The F3 cell (DESIGN §5's recorded `Unsupported`, "never a silent
/// miscompile"): reject a product whose residual contains an array in any
/// DEVICE value context that escapes per-thread storage —
///
/// - **a kernel's element type** (every launch-form bulk-op site's global-
///   memory loads/stores — incl. arrays OF such products and a consuming
///   `Index` across launches), and
/// - **a Twin/`__host__ __device__` fn's return type** (a pointer-field
///   struct returned by value on the device pass, or written through the
///   array out-param into the caller's buffer),
///
/// plus **device-side locals** of the product type itself — minus the
/// shapes that provably stay in per-thread storage: the input parameter
/// (arrives by value — the fold body's `(acc, e)` pair) and the transient
/// operand aggregates of bulk ops / calls (their fields are gathered
/// field-wise by the op; the struct itself is never stored to global
/// memory). ADR-0027 adds the same shape once more: a product used **only
/// by `Proj` edges whose surviving array components are all projected
/// out** (the captured map/fold source's feeder wire — the lower projects
/// the seed and the captured handles into the op's source product) is
/// destructured field-wise exactly like a bulk-op source: the struct never
/// escapes per-thread storage and its array content leaves only as
/// ordinary handles. Host-side products holding handles are NOT the cell
/// (their `T*` fields are host copies of device pointers — the F2 shape),
/// and neither are arrays OF scalar products, products of scalar products,
/// or scalar products.
pub(crate) fn check_device_product_arrays(
    ir: &CategoryIr,
    quals: &Qualifiers,
) -> Result<(), EmitError> {
    const CELL: &str = "arrays embedded in products on device";
    let cell = |loc: SourceLoc| EmitError::Unsupported {
        feature: CELL.into(),
        loc,
    };
    for (f, fd) in ir.funcs() {
        for site in collect_sites(ir, f) {
            let morph = ir.morphism(site.m).expect("morphism resolves");
            if site_element_tys(ir, morph)
                .iter()
                .any(tree_contains_product_with_array)
            {
                return Err(cell(morph.loc));
            }
        }
        if quals.get(f) == FnQual::HostOnly {
            continue; // host-only fns hold handles, never device values
        }
        let out_obj = ir.object(fd.output).expect("output resolves");
        if tree_contains_product_with_array(&out_obj.ty) {
            return Err(cell(out_obj.loc));
        }
        // The transient operand aggregates of this fn's bulk ops / calls:
        // consumed field-wise by the op, never stored (the in-twin Index
        // pair, the in-twin fold's (seed, array) pair, a device call's
        // by-value argument struct).
        let mut transient: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &m in &fd.morphisms {
            let morph = ir.morphism(m).expect("morphism resolves");
            if is_bulk_op(morph.op) || matches!(morph.op, Operation::Call(_)) {
                transient.insert(morph.source, ());
            }
        }
        for (o, obj) in ir.objects() {
            if ir.try_owner(o) != Some(f)
                || o == fd.input
                || o == fd.output
                || obj.kind == ObjectKind::Constant
                || transient.contains_key(o)
            {
                continue;
            }
            if residual_contains_array(&obj.ty) && !proj_destructured(ir, o, &obj.ty) {
                return Err(cell(obj.loc));
            }
        }
    }
    Ok(())
}

/// The ADR-0027 transient shape: is object `o` (typed `ty`, a product whose
/// residual contains an array) used **only** by `Proj` edges, with every
/// surviving array-typed component among the projected indices? Then the
/// struct is destructured field-wise and never escapes per-thread storage
/// (its array content leaves only as `{T}*` handle copies), exactly like a
/// bulk-op source aggregate. A product with no out-edges, a non-Proj use, or
/// an unprojected array component keeps the recorded cell.
pub(super) fn proj_destructured(ir: &CategoryIr, o: ObjectId, ty: &Ty) -> bool {
    let edges = ir.out_edges(o);
    if edges.is_empty() {
        return false;
    }
    let mut projected: Vec<u32> = Vec::new();
    for &m in edges {
        let morph = ir.morphism(m).expect("morphism resolves");
        match morph.op {
            Operation::Proj { index } => projected.push(index),
            _ => return false,
        }
    }
    let mut k = 0u32;
    while let Some(comp) = ty.component_ty(k) {
        if lower_ty(comp).is_some() && matches!(comp, Ty::Array { .. }) && !projected.contains(&k) {
            return false;
        }
        k += 1;
    }
    true
}

// ---------------------------------------------------------------------------
// Launch-form op sites
// ---------------------------------------------------------------------------

/// One launch-form array-bulk op site: the morphism and its kernel's name.
pub(crate) struct Site {
    pub m: MorphismId,
    pub kname: String,
}

/// Is `op` a launch-form array-bulk op (one kernel per site)?
pub(super) fn is_bulk_op(op: Operation) -> bool {
    matches!(
        op,
        Operation::Map { .. }
            | Operation::Zip
            | Operation::Enumerate
            | Operation::Iota
            | Operation::Fill
            | Operation::Fold { .. }
            | Operation::Index
            | Operation::Update
    )
}

/// The fn's launch-form bulk-op sites in `topo_order`, named
/// `k{f_ord}_{site_ord}`. Sites inside a canonical loop's decide/advance
/// cones are included (WP4): the loop driver emits their host launches
/// through the same op table, once per iteration at runtime.
pub(crate) fn collect_sites(ir: &CategoryIr, f: FuncId) -> Vec<Site> {
    let f_ord = ir.funcs().position(|(id, _)| id == f).expect("fn resolves");
    let mut out = Vec::new();
    for m in ir.topo_order(f) {
        let morph = ir.morphism(m).expect("morphism resolves");
        if is_bulk_op(morph.op) {
            let kname = format!("k{f_ord}_{}", out.len());
            out.push(Site { m, kname });
        }
    }
    out
}

/// The fns whose HOST definition can execute: the entry plus everything
/// reachable from it through `Call` edges (fixpoint; robust to recursion).
/// Map/fold bodies are invoked by kernels, not host calls, so a fn outside
/// this set has no host caller at runtime.
pub(crate) fn host_reachable(ir: &CategoryIr) -> SecondaryMap<FuncId, bool> {
    let mut live: SecondaryMap<FuncId, bool> = SecondaryMap::new();
    live.insert(ir.entry(), true);
    loop {
        let mut changed = false;
        for (f, fd) in ir.funcs() {
            if !live.get(f).copied().unwrap_or(false) {
                continue;
            }
            for &m in &fd.morphisms {
                let morph = ir.morphism(m).expect("morphism resolves");
                if let Operation::Call(g) = morph.op
                    && !live.get(g).copied().unwrap_or(false)
                {
                    live.insert(g, true);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    live
}

/// The dead-host-twin predicate (suggestions.md #12): a [`FnQual::Twin`] fn
/// with no host caller is invoked only from device code (kernels and twins
/// name its `d_` twin), so its host definition — and the launch-form kernels
/// only that definition launches — are dead text. Skipping them changes no
/// launch and no behavior; the twin (device section) is unaffected.
pub(crate) fn dead_host_twin(
    quals: &Qualifiers,
    live: &SecondaryMap<FuncId, bool>,
    f: FuncId,
) -> bool {
    quals.get(f) == FnQual::Twin && !live.get(f).copied().unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Shared shape helpers
// ---------------------------------------------------------------------------

/// The innermost element ty of a (possibly nested) array (DESIGN §5 flat
/// aggregate). Identity for non-arrays.
pub(super) fn flat_base(ty: &Ty) -> &Ty {
    let mut t = ty;
    while let Ty::Array { elem, .. } = t {
        t = elem;
    }
    t
}

/// The flat element count of `ty`: the product of array sizes along the
/// nesting (1 for non-arrays).
pub(crate) fn flat_count(ty: &Ty) -> u64 {
    let mut n = 1u64;
    let mut t = ty;
    while let Ty::Array { elem, size } = t {
        n *= size;
        t = elem;
    }
    n
}
