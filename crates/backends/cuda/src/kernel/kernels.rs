//! The __global__ kernel emitters: per-site kernel text and the kernel set.

use super::*;

// ---------------------------------------------------------------------------
// Kernel emission
// ---------------------------------------------------------------------------

/// The body-input argument assembly for a map/fold body call (ADR-0027 —
/// shared by the launch form's kernels and the inline form's per-thread
/// loops): the `(c₁…cₖ, original…)` product under the residual-erasure
/// remap, the captures as the leading fields. `parts` is one optional
/// expression per body-input component (`None` = an erased component, which
/// has no field to store). Returns the declaration/store lines (the caller
/// emits them at its indent) plus the argument expression: the assembled
/// struct for residual ≥ 2, the bare surviving component for residual-1,
/// `None` for residual-0. Assembly is split into pre-loop invariant lines
/// and per-iteration varying lines.
pub(super) fn assemble_body_arg(
    pair_ty: &Ty,
    parts: &[Option<String>],
    varying: &[bool],
) -> (Vec<String>, Vec<String>, Option<String>) {
    assert_eq!(parts.len(), varying.len(), "body argument variance");
    if residual_arity(pair_ty) >= 2 {
        let pct = lower_ty(pair_ty).expect("body input lowers");
        let mut pre = vec![format!("{pct} pair;")];
        let mut inloop = Vec::new();
        for (slot, (part, &is_varying)) in parts.iter().zip(varying).enumerate() {
            if let (Some(eidx), Some(expr)) = (erased_index(pair_ty, slot as u32), part) {
                let line = format!("pair.f{eidx} = {expr};");
                if is_varying {
                    inloop.push(line);
                } else {
                    pre.push(line);
                }
            }
        }
        (pre, inloop, Some("pair".to_string()))
    } else {
        // Residual ≤ 1: the bare survivor (if any) is the whole argument.
        (
            Vec::new(),
            Vec::new(),
            parts.iter().flatten().next().cloned(),
        )
    }
}

/// One `__global__` definition for `site` (deterministic text, no host slot
/// names — parameters are formal; the host launch in func.rs maps actuals
/// positionally onto the same list). A trap-FREE site (#14, `caps.site`)
/// drops the `unsigned int* trap` parameter — the launch then passes no
/// `d_trap` and skips the readback (func.rs's `launch_and_check`).
pub(crate) fn emit_kernel(
    ir: &CategoryIr,
    site: &Site,
    fnames: &SecondaryMap<FuncId, String>,
    quals: &Qualifiers,
    caps: &TrapCaps,
) -> String {
    let morph = ir.morphism(site.m).expect("morphism resolves");
    let k = Kernel {
        ir,
        name: &site.kname,
        fnames,
        quals,
        caps,
        capable: caps.site(ir, site.m),
        out: String::new(),
    };
    match morph.op {
        Operation::Map { body, captures } => {
            k.map_kernel(morph.source, morph.target, body, captures)
        }
        Operation::Zip => k.zip_kernel(morph.source, morph.target),
        Operation::Enumerate => k.enumerate_kernel(morph.source, morph.target),
        Operation::Iota => k.iota_kernel(morph.target),
        Operation::Fill => k.fill_kernel(morph.target),
        Operation::Update => k.update_kernel(morph.source, morph.target),
        Operation::Index => k.index_kernel(morph.source, morph.target),
        Operation::Fold { body, captures } => {
            k.fold_kernel(morph.source, morph.target, body, captures)
        }
        _ => unreachable!("not a bulk-op site"),
    }
}

/// The module's deduplicated kernel table (suggestions.md #17 — "kernel
/// shape dedup"): each structurally unique kernel is emitted ONCE and
/// launched per site. The shape key is the kernel's emitted text minus its
/// name — the op kind, parameter/element types, `n`/flat counts, and the
/// body fn's device name (where applicable) all ride the text, so text
/// equality IS structural equality. Deterministic (L2): fns iterate in
/// `ir.funcs()` order, sites in `collect_sites` order; the first site of a
/// shape lends its `k{f_ord}_{site_ord}` name to the surviving definition,
/// and the dedup map is only ever looked up, never iterated.
pub(crate) struct KernelSet {
    /// The deduplicated `__global__` definitions, in first-appearance order.
    pub text: String,
    /// fn → (site morphism → the launch's kernel name — the shape's
    /// survivor, not necessarily the site's own name).
    pub names: SecondaryMap<FuncId, HashMap<MorphismId, String>>,
}

/// Emit every fn's launch-form bulk-op sites as one deduplicated kernel
/// table (#17). [`FnEmit`](crate::func::FnEmit) consumes `names` so each
/// site's launch names the surviving definition. Sites of a dead host twin
/// (#12) contribute nothing: the only launcher of those kernels is the
/// fn's skipped host definition.
pub(crate) fn emit_kernel_set(
    ir: &CategoryIr,
    fnames: &SecondaryMap<FuncId, String>,
    quals: &Qualifiers,
    caps: &TrapCaps,
    live: &SecondaryMap<FuncId, bool>,
) -> KernelSet {
    let mut text = String::new();
    let mut names: SecondaryMap<FuncId, HashMap<MorphismId, String>> = SecondaryMap::new();
    // Shape key (definition text minus the name) → the surviving kernel name.
    let mut survivors: HashMap<String, String> = HashMap::new();
    for (f, _) in ir.funcs() {
        if dead_host_twin(quals, live, f) {
            continue; // #12: no host definition ⇒ no launches ⇒ no kernels
        }
        for site in collect_sites(ir, f) {
            let emitted = emit_kernel(ir, &site, fnames, quals, caps);
            // The name appears only in the `__global__ void {name}(` header —
            // kernel bodies name only body fns and `trap`, never the kernel.
            let key = &emitted[emitted.find('(').expect("kernel header")..];
            let kname = match survivors.get(key) {
                Some(survivor) => survivor.clone(),
                None => {
                    survivors.insert(key.to_string(), site.kname.clone());
                    text.push_str(&emitted);
                    site.kname.clone()
                }
            };
            let mut per_fn = names.get(f).cloned().unwrap_or_default();
            per_fn.insert(site.m, kname);
            names.insert(f, per_fn);
        }
    }
    KernelSet { text, names }
}

pub(super) struct Kernel<'a> {
    ir: &'a CategoryIr,
    name: &'a str,
    fnames: &'a SecondaryMap<FuncId, String>,
    quals: &'a Qualifiers,
    caps: &'a TrapCaps,
    /// The site's own trap capability (#14): a trap-free kernel carries no
    /// `unsigned int* trap` parameter.
    capable: bool,
    out: String,
}

impl<'a> Kernel<'a> {
    fn obj_ty(&self, o: ObjectId) -> Ty {
        self.ir.object(o).expect("object resolves").ty.clone()
    }

    fn line(&mut self, indent: usize, s: impl AsRef<str>) {
        for _ in 0..indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    /// `__global__ void k…(params) {` + the 64-bit thread-index prologue
    /// where `threaded` (elementwise kernels), else just the header.
    fn header(&mut self, params: &[String], threaded: bool) {
        self.out.push_str(&format!(
            "__global__ void {}({}) {{\n",
            self.name,
            params.join(", ")
        ));
        if threaded {
            self.line(
                1,
                "unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;",
            );
        }
    }

    /// The device call to body fn `g`: name + `trap` threading — the trap
    /// argument rides only when the callee can trap (#14); a trap-free
    /// callee takes no trap parameter (twin_signature/fn_signature).
    fn body_call(&self, g: FuncId, arg: Option<String>, dest: Option<String>) -> String {
        let name = self.quals.device_name(self.fnames, g);
        let mut args: Vec<String> = Vec::new();
        if let Some(a) = arg {
            args.push(a);
        }
        if let Some(d) = dest {
            args.push(d);
        }
        if self.caps.get(g) {
            args.push("trap".into());
        }
        format!("{name}({})", args.join(", "))
    }

    /// The trailing trap parameter, pushed iff this site is trap-capable
    /// (#14 — a trap-free kernel's launch passes no `d_trap`).
    fn push_trap_param(&self, params: &mut Vec<String>) {
        if self.capable {
            params.push("unsigned int* trap".into());
        }
    }

    /// The elementwise prologue shared by Map/Zip/Enumerate: kernel params
    /// `(out?, in…, trap)` — erased operands are omitted (erased-element
    /// arrays have no representation).
    fn elementwise_header(&mut self, params: &[String], n: u64) {
        self.header(params, true);
        self.line(1, format!("if (i < {n}ULL) {{"));
    }

    fn map_kernel(
        mut self,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) -> String {
        let src_ty = self.obj_ty(source);
        // ADR-0027: k>0 ⇒ the source is the product `(c₁…cₖ, [T; n])` — the
        // mapped array is its last component; k=0 ⇒ the bare array (every
        // pre-existing program).
        let arr_ty = if captures == 0 {
            src_ty.clone()
        } else {
            src_ty
                .component_ty(captures)
                .cloned()
                .expect("map array component")
        };
        let (elem, n) = array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);
        let (uelem, _) = array_parts(&tgt_ty);
        let arr_ct = lower_ty(&arr_ty);
        let tgt_ct = lower_ty(&tgt_ty);

        let mut params: Vec<String> = Vec::new();
        if let Some(ct) = &tgt_ct {
            params.push(format!("{ct} out"));
        }
        if let Some(ct) = &arr_ct {
            params.push(format!("{ct} in"));
        }
        // ADR-0027: the captured buffers/scalars ride as extra kernel
        // parameters, positionally after the existing operands, before the
        // trap pointer. An array capture is a plain handle (`{base}* cap{j}`);
        // a scalar capture is a by-value parameter — both are per-thread
        // reads of the same device storage.
        let mut cap_args: Vec<Option<String>> = Vec::new();
        for j in 0..captures {
            let cap_ty = src_ty.component_ty(j).cloned().expect("map capture");
            if let Some(ct) = lower_ty(&cap_ty) {
                params.push(format!("{ct} cap{j}"));
                cap_args.push(Some(format!("cap{j}")));
            } else {
                cap_args.push(None);
            }
        }
        self.push_trap_param(&mut params);
        self.elementwise_header(&params, n);

        // The body argument: k=0 ⇒ the bare element value (or nothing, if
        // the element is erased); k>0 ⇒ the `(c₁…cₖ, elem)` body-input
        // product assembled with the captures as leading fields. The store:
        // the out element, or nothing. Array-typed body results ride the
        // twin out-param convention (the body fn is necessarily a Twin then
        // — array return).
        let elem_arg = if lower_ty(&elem).is_some() {
            Some(elem_expr("in", "i", &elem))
        } else {
            None
        };
        let arg = if captures == 0 {
            elem_arg
        } else {
            let pair_ty = self.obj_ty(self.ir.func(body).expect("body resolves").input);
            let mut parts = cap_args.clone();
            parts.push(elem_arg);
            let varying = vec![false; parts.len()];
            let (pre, inloop, arg) = assemble_body_arg(&pair_ty, &parts, &varying);
            for l in pre {
                self.line(2, l);
            }
            for l in inloop {
                self.line(2, l);
            }
            arg
        };
        if lower_ty(&uelem).is_none() {
            let call = self.body_call(body, arg, None);
            self.line(2, format!("{call};"));
        } else if matches!(uelem, Ty::Array { .. }) {
            let dest = format!("(out + (i * {}ULL))", flat_count(&uelem));
            let call = self.body_call(body, arg, Some(dest));
            self.line(2, format!("{call};"));
        } else {
            let call = self.body_call(body, arg, None);
            self.line(2, format!("out[i] = {call};"));
        }
        self.line(1, "}");
        self.out.push_str("}\n");
        self.out
    }

    fn zip_kernel(mut self, source: ObjectId, target: ObjectId) -> String {
        let src_ty = self.obj_ty(source);
        let a_ty = src_ty.component_ty(0).cloned().expect("zip a");
        let b_ty = src_ty.component_ty(1).cloned().expect("zip b");
        let (a_elem, n) = array_parts(&a_ty);
        let (b_elem, _) = array_parts(&b_ty);
        let tgt_ty = self.obj_ty(target);
        let (elem, _) = array_parts(&tgt_ty); // the (A, B) product

        let mut params: Vec<String> = Vec::new();
        if let Some(ct) = lower_ty(&tgt_ty) {
            params.push(format!("{ct} out"));
        }
        if let Some(ct) = lower_ty(&a_ty) {
            params.push(format!("{ct} a"));
        }
        if let Some(ct) = lower_ty(&b_ty) {
            params.push(format!("{ct} b"));
        }
        self.push_trap_param(&mut params);
        self.elementwise_header(&params, n);

        // Per surviving component k of the element product: store the source
        // element (or sub-array pointer) into the erased-index field.
        self.zip_store(&elem, 0, &a_elem, "a");
        self.zip_store(&elem, 1, &b_elem, "b");
        self.line(1, "}");
        self.out.push_str("}\n");
        self.out
    }

    /// One component store of the Zip elementwise body; no-op when the
    /// component is erased.
    fn zip_store(&mut self, elem: &Ty, k: u32, comp_elem: &Ty, arr: &str) {
        let Some(eidx) = erased_index(elem, k) else {
            return;
        };
        let lvalue = if residual_arity(elem) == 1 {
            "out[i]".to_string()
        } else {
            format!("out[i].f{eidx}")
        };
        let rval = elem_expr(arr, "i", comp_elem);
        self.line(2, format!("{lvalue} = {rval};"));
    }

    fn enumerate_kernel(mut self, source: ObjectId, target: ObjectId) -> String {
        let src_ty = self.obj_ty(source);
        let (a_elem, n) = array_parts(&src_ty);
        let tgt_ty = self.obj_ty(target);
        let (elem, _) = array_parts(&tgt_ty); // the (i32, A) product

        let mut params: Vec<String> = Vec::new();
        if let Some(ct) = lower_ty(&tgt_ty) {
            params.push(format!("{ct} out"));
        }
        if let Some(ct) = lower_ty(&src_ty) {
            params.push(format!("{ct} a"));
        }
        self.push_trap_param(&mut params);
        self.elementwise_header(&params, n);

        // The i32 index IS the thread index (cast, no transfer — §1).
        let bare = residual_arity(&elem) == 1;
        if let Some(eidx) = erased_index(&elem, 0) {
            let lvalue = if bare {
                "out[i]".to_string()
            } else {
                format!("out[i].f{eidx}")
            };
            self.line(2, format!("{lvalue} = (int32_t)i;"));
        }
        if let Some(eidx) = erased_index(&elem, 1) {
            let lvalue = if bare {
                "out[i]".to_string()
            } else {
                format!("out[i].f{eidx}")
            };
            let rval = elem_expr("a", "i", &a_elem);
            self.line(2, format!("{lvalue} = {rval};"));
        }
        self.line(1, "}");
        self.out.push_str("}\n");
        self.out
    }

    fn iota_kernel(mut self, target: ObjectId) -> String {
        let tgt_ty = self.obj_ty(target);
        // Elem ctype derived, not hardcoded: the builder pins `[i32; n]`
        // today, but the ADR-0029 `[i64; n]` annotation form must not
        // silently truncate here if it ever lands.
        let (elem, _) = array_parts(&tgt_ty);
        let elem_ct = lower_ty(&elem).expect("iota elem lowers");
        let mut params = vec![format!(
            "{} out",
            lower_ty(&tgt_ty).expect("iota target lowers")
        )];
        params.push("long long n".into());
        self.header(&params, false);
        self.line(
            1,
            "long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
        );
        self.line(1, format!("if (i < n) out[i] = ({elem_ct})i;"));
        self.out.push_str("}\n");
        self.out
    }

    fn fill_kernel(mut self, target: ObjectId) -> String {
        let tgt_ty = self.obj_ty(target);
        let (elem, _) = array_parts(&tgt_ty);
        let mut params = Vec::new();
        if let Some(ct) = lower_ty(&tgt_ty) {
            params.push(format!("{ct} out"));
        }
        params.push("long long n".into());
        if let Some(ct) = lower_ty(&elem) {
            params.push(format!("{ct} x"));
        }
        self.header(&params, false);
        self.line(
            1,
            "long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
        );
        if lower_ty(&elem).is_some() {
            self.line(1, "if (i < n) out[i] = x;");
        }
        self.out.push_str("}\n");
        self.out
    }

    fn update_kernel(mut self, source: ObjectId, _target: ObjectId) -> String {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("update array");
        let val_ty = src_ty.component_ty(2).cloned().expect("update val");
        let (elem, n) = array_parts(&arr_ty);
        let flat = flat_count(&arr_ty);

        // BC5: full-copy kernel + bounds guard (index_oob ⇒ store 2u, the
        // kind+1 encoding; width rule §3). The guard runs in every thread
        // (uniform, benign same-value stores); the index arrives as int64_t
        // (extended at the launch, §3).
        let mut params: Vec<String> = Vec::new();
        let arr_ct = lower_ty(&arr_ty);
        if arr_ct.is_some() {
            params.push(format!("{}* out", flat_base_ct(&arr_ty).unwrap()));
            params.push(format!("{}* src", flat_base_ct(&arr_ty).unwrap()));
        }
        params.push("int64_t idx".into());
        if lower_ty(&val_ty).is_some() {
            let vct = flat_base_ct(&val_ty).expect("update val lowers");
            if matches!(val_ty, Ty::Array { .. }) {
                params.push(format!("{vct}* val"));
            } else {
                params.push(format!("{vct} val"));
            }
        }
        self.push_trap_param(&mut params);

        self.header(&params, false);
        let cond = bounds_cond("idx", n);
        self.line(1, format!("if ({cond}) {{ *trap = 2u; return; }}"));
        if arr_ct.is_some() {
            self.line(
                1,
                "unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;",
            );
            self.line(1, format!("if (i < {flat}ULL) {{"));
            if matches!(elem, Ty::Array { .. }) {
                // Nested: slot i/m of the outer is replaced by the whole
                // sub-array `val` (flat copy, m cells per outer element).
                let m = flat_count(&elem);
                self.line(
                    2,
                    format!("out[i] = ((int64_t)(i / {m}ULL) == idx) ? val[i % {m}ULL] : src[i];"),
                );
            } else {
                self.line(2, "out[i] = ((int64_t)i == idx) ? val : src[i];");
            }
            self.line(1, "}");
        }
        self.out.push_str("}\n");
        self.out
    }

    fn index_kernel(mut self, source: ObjectId, target: ObjectId) -> String {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("index array");
        let (elem, n) = array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);

        // 1-launch kernel (host launches <<<1, 1>>>): guard → write the
        // element into a device result buffer. Scalar elements land in a
        // 1-cell buffer the host memcpy's D→H; array elements copy into a
        // fresh device buffer (in-kernel device-to-device, §1 op table).
        let mut params: Vec<String> = Vec::new();
        let tgt_ct = lower_ty(&tgt_ty);
        if tgt_ct.is_some() {
            // The result is always a device buffer: 1 cell for scalar
            // elements, flat m cells for array elements (§1 op table).
            params.push(format!("{}* result", flat_base_ct(&tgt_ty).unwrap()));
        }
        if let Some(ct) = lower_ty(&arr_ty) {
            params.push(format!("{ct} arr"));
        }
        params.push("int64_t idx".into());
        self.push_trap_param(&mut params);

        self.header(&params, false);
        // The §3 bounds guard rides only a trap-CAPABLE site (#14/S20): a
        // proven-in-bounds Index is trap-free — no flag to set, dead text.
        if self.capable {
            let cond = bounds_cond("idx", n);
            self.line(1, format!("if ({cond}) {{ *trap = 2u; return; }}"));
        }
        if tgt_ct.is_some() {
            if matches!(elem, Ty::Array { .. }) {
                let m = flat_count(&elem);
                self.line(
                    1,
                    format!("for (unsigned long long j = 0; j < {m}ULL; j++) {{"),
                );
                self.line(
                    2,
                    format!("result[j] = arr[((unsigned long long)idx * {m}ULL) + j];"),
                );
                self.line(1, "}");
            } else {
                self.line(1, "*result = arr[(unsigned long long)idx];");
            }
        }
        self.out.push_str("}\n");
        self.out
    }

    fn fold_kernel(
        mut self,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) -> String {
        let src_ty = self.obj_ty(source);
        // ADR-0027: the source product is `(c₁…cₖ, Acc, [T; n])` — the
        // captures shift the acc to component k and the array to k+1 (k=0:
        // today's `(Acc, [T; n])`).
        let acc_ty = src_ty.component_ty(captures).cloned().expect("fold acc");
        let arr_ty = src_ty
            .component_ty(captures + 1)
            .cloned()
            .expect("fold array");
        let (elem, n) = array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);
        // The body input product `(c₁…cₖ, Acc, T)` — the oracle's per-step
        // assembly extended with the leading captures (ADR-0027).
        let pair_ty = self.obj_ty(self.ir.func(body).expect("body resolves").input);

        let acc_ct = lower_ty(&acc_ty);
        let arr_ct = lower_ty(&arr_ty);
        let mut params: Vec<String> = Vec::new();
        if lower_ty(&tgt_ty).is_some() {
            // Scalar acc ⇒ a 1-cell buffer; array acc ⇒ the flat result
            // buffer (stays on device, §1).
            params.push(format!("{}* result", flat_base_ct(&tgt_ty).unwrap()));
        }
        match &acc_ty {
            Ty::Array { .. } => {
                // Erased-element acc (Array{Unit}): no representation — the
                // acc0 parameter is omitted; the fold loop still runs (F6,
                // launch-form degradation). (`acc_ct` of an array IS the
                // `{base}*` pointer text.)
                if let Some(ct) = &acc_ct {
                    params.push(format!("{ct} acc0"));
                }
            }
            _ => {
                if let Some(ct) = &acc_ct {
                    params.push(format!("{ct} acc0"));
                }
            }
        }
        if arr_ct.is_some() {
            params.push(format!("{}* arr", flat_base_ct(&arr_ty).unwrap()));
        }
        // ADR-0027: the captured buffers/scalars as extra kernel parameters,
        // positionally after the existing operands, before the trap pointer.
        let mut cap_args: Vec<Option<String>> = Vec::new();
        for j in 0..captures {
            let cap_ty = src_ty.component_ty(j).cloned().expect("fold capture");
            if let Some(ct) = lower_ty(&cap_ty) {
                params.push(format!("{ct} cap{j}"));
                cap_args.push(Some(format!("cap{j}")));
            } else {
                cap_args.push(None);
            }
        }
        self.push_trap_param(&mut params);

        // BC4: the single-thread kernel — the oracle's left-sequential loop
        // over device memory, acc in per-thread local storage (§5). An
        // erased-element acc has no storage anywhere (F6): no acc local, no
        // writeback — the loop still calls the body per element.
        self.header(&params, false);
        let array_acc = matches!(acc_ty, Ty::Array { .. }) && acc_ct.is_some();
        if array_acc {
            let m = flat_count(&acc_ty);
            let ct = flat_base_ct(&acc_ty).expect("array acc base lowers");
            self.line(1, format!("{ct} acc[{m}];"));
            self.line(
                1,
                format!("for (unsigned long long j = 0; j < {m}ULL; j++) {{"),
            );
            self.line(2, "acc[j] = acc0[j];");
            self.line(1, "}");
        } else if let Some(ct) = &acc_ct {
            self.line(1, format!("{ct} acc = acc0;"));
        }
        // The per-step argument: the captures (leading fields), then the
        // local scalar / per-thread local array acc (decays to the pointer
        // field for an array acc), then the element — the body input's
        // `(c₁…cₖ, Acc, T)` shape under the residual-erasure remap.
        let acc_expr = if lower_ty(&acc_ty).is_some() {
            Some("acc".to_string())
        } else {
            None
        };
        let elem_arg = if lower_ty(&elem).is_some() {
            Some(elem_expr("arr", "i", &elem))
        } else {
            None
        };
        let mut parts = cap_args.clone();
        parts.push(acc_expr);
        parts.push(elem_arg);
        let mut varying = vec![false; captures as usize];
        varying.extend([!array_acc, true]);
        let (pre, inloop, arg) = assemble_body_arg(&pair_ty, &parts, &varying);
        for l in pre {
            self.line(1, l);
        }
        self.line(
            1,
            format!("for (unsigned long long i = 0; i < {n}ULL; i++) {{"),
        );
        for l in inloop {
            self.line(2, l);
        }
        let dest = if array_acc {
            Some("acc".to_string())
        } else {
            None
        };
        let call = self.body_call(body, arg, dest);
        if array_acc {
            self.line(2, format!("{call};"));
        } else if acc_ct.is_some() {
            self.line(2, format!("acc = {call};"));
        } else {
            self.line(2, format!("{call};"));
        }
        // The per-step trap check (first-trap-wins, §3) — only when the body
        // can trap (#14); a trap-free body leaves the flag untouched.
        if self.caps.get(body) {
            self.line(2, "if (*trap) return;");
        }
        self.line(1, "}");
        if array_acc {
            let m = flat_count(&acc_ty);
            self.line(
                1,
                format!("for (unsigned long long j = 0; j < {m}ULL; j++) {{"),
            );
            self.line(2, "result[j] = acc[j];");
            self.line(1, "}");
        } else if lower_ty(&tgt_ty).is_some() {
            self.line(1, "*result = acc;");
        }
        self.out.push_str("}\n");
        self.out
    }
}
