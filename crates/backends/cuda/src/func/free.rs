//! Function signatures, in-place update facts and C literal helpers of the host emitter.

use super::*;

/// The C++ function signature text (no body): `static {ret} {name}({param})`.
/// Erased input ⇒ no parameter, erased output ⇒ `void` (the llvm rule). Used
/// for both the file-top prototypes and the definitions — C++ needs a
/// declaration before any call, so all fns are prototyped first. A
/// [`FnQual::HostDevice`] fn gets the single two-site definition:
/// `static __host__ __device__ {ret} {name}({param}, unsigned int* d_trap)`.
/// The trap parameter is named `d_trap` **deliberately shadowing the host
/// global**: inside the fn, `d_trap` then resolves to the parameter on the
/// device pass and to the threaded-through global on the host pass, so call
/// sites (`callee(arg, d_trap)`) and the Div/Mod guard's `*d_trap = 1u;`
/// are source-identical on both sites (§3's kind+1 flag encoding). #14: the
/// parameter rides only when the fn can trap (`caps`) — a trap-free fn's
/// callers pass nothing (func.rs `emit_call`, kernel.rs's device calls).
pub(crate) fn fn_signature(
    ir: &CategoryIr,
    f: FuncId,
    fnames: &SecondaryMap<FuncId, String>,
    qual: FnQual,
    caps: &kernel::TrapCaps,
) -> String {
    let fd = ir.func(f).expect("func resolves");
    let in_ty = &ir.object(fd.input).expect("input resolves").ty;
    let ret_ty = &ir.object(fd.output).expect("output resolves").ty;
    let ret = lower_ty(ret_ty).unwrap_or_else(|| "void".into());
    let param = match lower_ty(in_ty) {
        Some(t) => format!("{t} in"),
        None => String::new(),
    };
    if qual == FnQual::HostDevice {
        let params = match (param.is_empty(), caps.get(f)) {
            (true, true) => "unsigned int* d_trap".to_string(),
            (false, true) => format!("{param}, unsigned int* d_trap"),
            (true, false) => String::new(),
            (false, false) => param,
        };
        return format!("static __host__ __device__ {ret} {}({params})", fnames[f]);
    }
    format!("static {ret} {}({param})", fnames[f])
}

// --- free helpers ---------------------------------------------------------

/// In-place `Update` legality (plan-last-use §2 rule 4 + the consumer
/// composition pinned by mapal-ir's `last_use_borrowed_init_is_never_dead`):
/// `Update(s, …)` may write into the source's buffer iff
///
/// 1. `dead_after(s, position(update))` — the plan's predicate: no use of
///    `s` at/after the update (rule 1's ranking puts the decide cone's reads
///    before the advance cone, so a guard read of the carried array still
///    converts), `s` does not escape, `s` is not carried. This alone decides
///    DIRECT sources: a `Parameter`/borrowed or escape-reaching source fails
///    it (the plan's `escapes`), a `Phi`/`Proj`/`Call` result may pass it —
///    so the remaining half is the consumer's, per the source's alias root
///    (the emission-level pointer copies):
///
/// 2. a **`LoopMerge` root** (the carried case — the source rides the merge's
///    current-state buffer) requires [`merge_family_dead`] (every same-field
///    current-state alias dies at the update — a later read of the same
///    buffer vetoes) AND [`owned_loop_init`] (the loop's init is an owned,
///    read-nowhere-else buffer — a borrowed `Parameter` init or a
///    ptr-resident/extra-used init vetoes: its buffer is the caller's, or is
///    still read — the full copy stays);
///
/// 3. **any other root** requires the source itself to be a fresh fn-owned
///    buffer ([`fresh_owned_buffer`]: a literal or bulk-op target — never a
///    `Proj`/`Phi`/`Call` result, whose storage aliases another object's or
///    may be borrowed).
///
/// Where the plan can't prove, the sites fall back to the full-copy Update
/// (today's behavior — the conservative default).
pub(crate) fn in_place_update(
    ir: &CategoryIr,
    plan: &LastUsePlan,
    f: FuncId,
    m: MorphismId,
    src: ObjectId,
) -> bool {
    let Some(pos) = plan.position(m) else {
        return false;
    };
    if !plan.dead_after(src, pos) {
        return false;
    }
    // The alias root: walk `Proj` in-edges source-ward (a `Proj` result is a
    // handle copy of the product's field — the same buffer), recording the
    // field path (innermost first).
    let mut root = src;
    let mut field_path: Vec<u32> = Vec::new();
    loop {
        let proj = ir.in_edges(root).iter().copied().find(|&e| {
            matches!(
                ir.morphism(e).expect("morphism resolves").op,
                Operation::Proj { .. }
            )
        });
        let Some(e) = proj else { break };
        let morph = ir.morphism(e).expect("morphism resolves");
        let Operation::Proj { index } = morph.op else {
            unreachable!()
        };
        field_path.push(index);
        root = morph.source;
    }
    if ir.object(root).expect("object resolves").kind == ObjectKind::LoopMerge {
        merge_family_dead(ir, plan, root, &field_path, pos) && owned_loop_init(ir, f, root)
    } else {
        field_path.is_empty() && fresh_owned_buffer(ir, src)
    }
}

/// The merge-family half of [`in_place_update`]'s rule 4: every use of the
/// source field's buffer — through the merge itself (whole-aggregate uses:
/// the exit assembly's handle copies, direct whole-merge operands) or
/// through same-field `Proj` views (the current-state aliases) — ranks
/// at/before the update. Other fields' projections name other buffers and
/// are irrelevant (a product state's scalar counter never vetoes the array
/// field). Positions and dead-ness are the plan's; the field selection is
/// the emitter's buffer structure.
pub(super) fn merge_family_dead(
    ir: &CategoryIr,
    plan: &LastUsePlan,
    agg: ObjectId,
    field_path: &[u32],
    pos: u32,
) -> bool {
    for &e in ir.out_edges(agg) {
        let morph = ir.morphism(e).expect("morphism resolves");
        if let Operation::Proj { index } = morph.op {
            if !field_path.is_empty() && index != field_path[0] {
                continue; // a different field: another buffer.
            }
            // A current-state alias of the source's buffer: it and its
            // matching descendants must die at the update too (a read of
            // the old value after the write would observe the mutation).
            if !plan.dead_after(morph.target, pos)
                || !merge_family_dead(ir, plan, morph.target, &field_path[1..], pos)
            {
                return false;
            }
            continue;
        }
        // Whole-aggregate uses (Pair/LoopExit handle copies included) must
        // rank at/before the update — the plan's positions, not liveness.
        if plan.position(e).is_none_or(|p| p > pos) {
            return false;
        }
    }
    true
}

/// The borrowed/extra-used init veto ([`in_place_update`]'s consumer half —
/// mapal-ir's pin: "a loop's borrowed init is never written in place"). Walks
/// the init's assembly cone (the init object plus everything it packs,
/// source-ward through `Pair` edges): every buffer in it must be fn-owned
/// and read nowhere but the ONE entry copy — a `Parameter` (borrowed), a
/// ptr-resident provenance (`Phi`/`Proj`/`Call` — the buffer aliases or may
/// be borrowed), an extra use (the value is read elsewhere, and iteration 1
/// writes the merge's buffer = the init's), or a second `LoopEnter` (two
/// loops sharing the init) all veto.
pub(super) fn owned_loop_init(ir: &CategoryIr, f: FuncId, merge: ObjectId) -> bool {
    let Some(lp) = ir.loop_plan(f, merge) else {
        return false;
    };
    let mut cone: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    let mut stack = vec![lp.init];
    while let Some(o) = stack.pop() {
        if cone.contains_key(o) {
            continue;
        }
        cone.insert(o, ());
        for &e in ir.in_edges(o) {
            let morph = ir.morphism(e).expect("morphism resolves");
            if matches!(morph.op, Operation::Pair { .. }) {
                stack.push(morph.source);
            }
        }
    }
    let mut entries = 0;
    for (o, _) in cone.iter() {
        let obj = ir.object(o).expect("object resolves");
        if obj.kind == ObjectKind::Parameter {
            return false; // the borrowed handle (mapal-ir's pin)
        }
        let buffer_bearing = matches!(obj.ty, Ty::Array { .. } | Ty::Tuple(_) | Ty::Struct { .. });
        if matches!(obj.ty, Ty::Array { .. }) && !fresh_owned_buffer(ir, o) {
            return false; // ptr-resident or borrowed provenance
        }
        if !buffer_bearing {
            continue; // scalars have no buffer — their other uses are reads of a copy
        }
        for &e in ir.out_edges(o) {
            let morph = ir.morphism(e).expect("morphism resolves");
            match morph.op {
                Operation::LoopEnter => entries += 1,
                Operation::Pair { .. } if cone.contains_key(morph.target) => {}
                _ => return false, // read/packed elsewhere, or a second consumer
            }
        }
    }
    entries == 1
}

/// A fresh fn-owned buffer: defined only by ops whose targets this backend
/// allocates itself (an array literal's `Pair` cone, or a bulk-op target) —
/// never a `Phi`/`Proj`/`Call` result (pointer-resident: aliases another
/// object's buffer, or may be the callee's borrowed argument).
pub(super) fn fresh_owned_buffer(ir: &CategoryIr, o: ObjectId) -> bool {
    let obj = ir.object(o).expect("object resolves");
    if obj.kind != ObjectKind::Temporary {
        return false;
    }
    let in_edges = ir.in_edges(o);
    !in_edges.is_empty()
        && in_edges.iter().all(|&e| {
            matches!(
                ir.morphism(e).expect("morphism resolves").op,
                Operation::Pair { .. }
                    | Operation::Map { .. }
                    | Operation::Zip
                    | Operation::Enumerate
                    | Operation::Iota
                    | Operation::Fill
                    | Operation::Update
                    | Operation::Fold { .. }
                    | Operation::Index
            )
        })
}

/// The lvalues of every array-typed component of the value `base` (typed
/// `ty`) — the escape set for the fn-epilogue free guard (§2, F2). A bare
/// array contributes `base` itself; a product contributes each surviving
/// array-typed component's field lvalue (`base.f{erased_index}` — the ty.rs
/// residual remap), recursing into nested products; scalars and erased
/// components contribute nothing. Pointer equality against this set
/// subsumes the Phi/alias escape shapes (whichever buffer the return value
/// holds is the escaping one).
pub(super) fn escape_lvalues(ty: &Ty, base: &str) -> Vec<String> {
    match ty {
        Ty::Array { .. } => vec![base.to_string()],
        Ty::Tuple(_) | Ty::Struct { .. } => {
            // A residual-1 product is bare: the local IS the single
            // surviving component (no `.f` field — the component_expr rule).
            let bare = residual_arity(ty) == 1;
            let mut out = Vec::new();
            let mut k = 0u32;
            while let Some(comp) = ty.component_ty(k) {
                if lower_ty(comp).is_some() {
                    let lbase = if bare {
                        base.to_string()
                    } else {
                        let eidx = erased_index(ty, k).expect("surviving component");
                        format!("{base}.f{eidx}")
                    };
                    out.extend(escape_lvalues(comp, &lbase));
                }
                k += 1;
            }
            out
        }
        _ => Vec::new(),
    }
}

pub(crate) fn is_float(ty: &Ty) -> bool {
    matches!(ty, Ty::Float { .. })
}

/// The unsigned int C++ type of the same width — BC2's cast target.
pub(crate) fn unsigned_twin(ty: &Ty) -> String {
    match ty {
        Ty::Int { bits, .. } => format!("uint{bits}_t"),
        _ => unreachable!("unsigned twin of non-int"),
    }
}

/// The `<cstdint>` MIN macro for a signed int C++ type (the MIN/-1 guard's
/// bound and the Div overflow value).
pub(super) fn int_min(ct: &str) -> &'static str {
    match ct {
        "int8_t" => "INT8_MIN",
        "int16_t" => "INT16_MIN",
        "int32_t" => "INT32_MIN",
        "int64_t" => "INT64_MIN",
        _ => unreachable!("non-Core int width in Div/Mod"),
    }
}

/// The C++ constant text for a scalar `Value`. Ints render decimal except
/// `MIN` (the bare decimal `-…8` literal doesn't fit the C++ literal's type —
/// the `<cstdint>` macros instead). Floats render as round-tripping
/// scientific literals: Rust `{:e}` prints the shortest digits that
/// round-trip (the same value mapal-rt's `Display` parity prints), and the
/// exponent keeps huge values like 1e300 floating literals (plain `Display`
/// would print 300 digits with no point — an *integer* literal, ill-formed).
/// f32 gets the `f` suffix (no double-rounding); NaN/±inf use the `<cmath>`
/// macros.
pub(crate) fn const_literal(v: &Value) -> String {
    match v {
        Value::I32(n) if *n == i32::MIN => "INT32_MIN".into(),
        Value::I32(n) => n.to_string(),
        Value::I64(n) if *n == i64::MIN => "INT64_MIN".into(),
        Value::I64(n) => n.to_string(),
        Value::U8(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        // ADR-0043: the format's bit pattern.
        Value::Narrow(mapal_ir::NarrowFloat::Bf16, b) => {
            format!("__ushort_as_bfloat16((unsigned short){b:#06x})")
        }
        Value::Narrow(mapal_ir::NarrowFloat::F16, b) => {
            format!("__ushort_as_half((unsigned short){b:#06x})")
        }
        // ADR-0043 slice 3: the pattern in the storage byte.
        Value::Narrow(n @ (mapal_ir::NarrowFloat::F8e4m3 | mapal_ir::NarrowFloat::F8e5m2), b) => {
            let t = crate::ty::lower_ty(&mapal_ir::Ty::Narrow(*n)).expect("fp8 lowers");
            format!("[] {{ {t} v; v.__x = (__nv_fp8_storage_t){b:#04x}; return v; }}()")
        }
        Value::F32(x) => float_literal(
            x.is_nan(),
            x.is_sign_negative(),
            x.is_infinite(),
            format!("{x:e}"),
            "f",
        ),
        Value::F64(x) => float_literal(
            x.is_nan(),
            x.is_sign_negative(),
            x.is_infinite(),
            format!("{x:e}"),
            "",
        ),
        Value::Str(_) | Value::Unit => unreachable!("Str / Unit is not a scalar operand"),
    }
}

pub(super) fn float_literal(
    is_nan: bool,
    neg: bool,
    is_inf: bool,
    digits: String,
    suffix: &str,
) -> String {
    if is_nan {
        return if neg { "(-NAN)".into() } else { "NAN".into() };
    }
    if is_inf {
        let h = if suffix == "f" {
            "HUGE_VALF"
        } else {
            "HUGE_VAL"
        };
        return if neg { format!("-{h}") } else { h.into() };
    }
    format!("{digits}{suffix}")
}
