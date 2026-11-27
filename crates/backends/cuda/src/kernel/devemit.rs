//! DevEmit: construction, class/slot helpers, operand materialization and device-side operand gathering.

use super::*;

impl<'a> DevEmit<'a> {
    pub(crate) fn new(
        ir: &'a CategoryIr,
        f: FuncId,
        fnames: &'a SecondaryMap<FuncId, String>,
        quals: &'a Qualifiers,
        caps: &'a TrapCaps,
    ) -> Self {
        let fd = ir.func(f).expect("func resolves");
        let out_ty = &ir.object(fd.output).expect("output resolves").ty;

        let mut produced: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &m in &fd.morphisms {
            let morph = ir.morphism(m).expect("morphism resolves");
            let produces_array = match morph.op {
                Operation::Map { .. }
                | Operation::Zip
                | Operation::Enumerate
                | Operation::Iota
                | Operation::Fill
                | Operation::Update
                | Operation::Call(_) => {
                    matches!(
                        ir.object(morph.target).expect("target").ty,
                        Ty::Array { .. }
                    )
                }
                Operation::Index | Operation::Fold { .. } => {
                    matches!(
                        ir.object(morph.target).expect("target").ty,
                        Ty::Array { .. }
                    )
                }
                Operation::Pair { .. } => {
                    matches!(
                        ir.object(morph.target).expect("target").ty,
                        Ty::Array { .. }
                    )
                }
                _ => false,
            };
            if produces_array {
                produced.insert(morph.target, ());
            }
        }
        if matches!(out_ty, Ty::Array { .. }) {
            produced.insert(fd.output, ());
        }

        // In-place Update targets (plan-last-use §2 rule 4): the deduced
        // query answers once per site, at construction (the BL7 pattern —
        // deduced, never re-derived at the site).
        let last_use = ir.last_use_plan(f);
        let mut in_place: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &m in &fd.morphisms {
            let morph = ir.morphism(m).expect("morphism resolves");
            if morph.op != Operation::Update {
                continue;
            }
            let src_ty = ir.object(morph.source).expect("source resolves").ty.clone();
            let arr_ty = src_ty.component_ty(0).cloned().expect("update array");
            if lower_ty(&arr_ty).is_none() {
                continue; // erased element: the guard is the whole op
            }
            let src = pair_source(ir, morph.source, 0).expect("update src");
            if in_place_update(ir, &last_use, f, m, src) {
                in_place.insert(morph.target, ());
            }
        }

        let ret_default = match lower_ty(out_ty) {
            Some(ct) if !matches!(out_ty, Ty::Array { .. }) => format!("return {ct}{{}};"),
            _ => "return;".to_string(),
        };

        let plan = ir.emission_plan(f);
        let mut force_named: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &m in &fd.morphisms {
            let morph = ir.morphism(m).expect("morphism resolves");
            // Call: the §3 post-call trap check is anchored to a statement.
            // Fold: an in-twin fold is a LOOP — its scalar result is written
            // by the loop's acc updates and needs an lvalue; the query may
            // class it Inline (one pure consumer, no boundary) but no
            // expression form exists. The other bulk ops produce arrays,
            // which the query keeps Named. (S23 fix: the box differential
            // caught fold-result-into-scalar-op dropping the value — the
            // `neg operand` panic class, 27/640 testgen emissions.)
            if matches!(morph.op, Operation::Call(_) | Operation::Fold { .. }) {
                force_named.insert(morph.target, ());
            }
        }
        // Product-typed Inline (plan §2 as-built note 2): Phase 1 takes the
        // plan's sanctioned fallback — one local name — instead of a braced
        // aggregate literal (the Pair arm assigns its fields normally). The
        // exhibits are unaffected: their wrapper products are Dissolved.
        for (id, obj) in ir.objects() {
            if ir.try_owner(id) == Some(f)
                && obj.ty.product_arity().is_some()
                && plan.class(id).is_some_and(|c| c.is_inline())
            {
                force_named.insert(id, ());
            }
        }

        DevEmit {
            ir,
            f,
            fnames,
            quals,
            caps,
            slots: SecondaryMap::new(),
            produced,
            in_place,
            lit_total: literal_pair_counts(ir, f),
            lit_seen: SecondaryMap::new(),
            proof: ir.bounds_proof(f),
            plan,
            force_named,
            exprs: SecondaryMap::new(),
            decls: String::new(),
            body: String::new(),
            next: 0,
            ret_default,
            base: 0,
        }
    }

    /// The effective class: the deduced plan, overridden Named where the
    /// backend's statement protocol demands it (Call targets).
    pub(super) fn cls(&self, o: ObjectId) -> EmissionClass {
        if self.force_named.contains_key(o) {
            return EmissionClass::Named;
        }
        self.plan.class(o).unwrap_or(EmissionClass::Named)
    }

    pub(super) fn dissolved(&self, o: ObjectId) -> bool {
        self.cls(o).is_dissolved()
    }

    /// Inline OR Dissolved: no local, no declaration — the value lives as an
    /// expression (or, for a dissolved product, as its fields' expressions).
    pub(super) fn expr_only(&self, o: ObjectId) -> bool {
        !self.cls(o).is_named()
    }

    pub(super) fn fresh(&mut self) -> u32 {
        let n = self.next;
        self.next += 1;
        n
    }

    pub(super) fn tmp(&mut self) -> String {
        format!("t{}", self.fresh())
    }

    pub(super) fn line(&mut self, indent: usize, s: impl AsRef<str>) {
        for _ in 0..(indent + self.base) {
            self.body.push_str("  ");
        }
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }

    pub(super) fn slot(&self, o: ObjectId) -> Option<String> {
        self.slots.get(o).cloned()
    }

    pub(super) fn obj_ty(&self, o: ObjectId) -> Ty {
        self.ir.object(o).expect("object resolves").ty.clone()
    }

    // --- operand materialization (host-table rules, func.rs) --------------

    /// The whole value of `o` as a C++ expression: constant literal,
    /// memoized Inline expression, or slot name (an array slot name decays
    /// to a pointer where needed).
    pub(super) fn load_whole(&mut self, o: ObjectId) -> Option<(String, String)> {
        let obj = self.ir.object(o).expect("object resolves");
        if obj.kind == ObjectKind::Constant {
            return match &obj.value {
                Some(Value::Str(_)) | None => None,
                Some(v) => Some((lower_ty(&obj.ty)?, const_literal(v))),
            };
        }
        let ct = lower_ty(&obj.ty)?;
        if let Some(e) = self.exprs.get(o) {
            return Some((ct, e.clone()));
        }
        let slot = self.slot(o)?;
        Some((ct, slot))
    }

    /// Component `k` of aggregate `agg` under the residual-erasure remap
    /// (identical to the host rule). A DISSOLVED aggregate never
    /// materializes: the component resolves through the Pair edge to the
    /// field source's own expression (WP-B — the wrapper text disappears).
    pub(super) fn component_expr(&mut self, agg: ObjectId, k: u32) -> Option<(String, String)> {
        let agg_ty = self.obj_ty(agg);
        match &agg_ty {
            Ty::Tuple(_) | Ty::Struct { .. } => {
                let comp_ty = agg_ty.component_ty(k)?.clone();
                let cct = lower_ty(&comp_ty)?;
                if self.dissolved(agg) {
                    let src = pair_source(self.ir, agg, k)?;
                    let (_, val) = self.load_whole(src)?;
                    return Some((cct, val));
                }
                let agg_slot = self.slot(agg)?;
                if residual_arity(&agg_ty) == 1 {
                    Some((cct, agg_slot))
                } else {
                    let eidx = erased_index(&agg_ty, k)?;
                    Some((cct, format!("{agg_slot}.f{eidx}")))
                }
            }
            _ => None,
        }
    }

    /// Route a produced value: a Named object takes its one statement
    /// assignment; an Inline object memoizes the (parenthesized) expression
    /// and emits nothing (WP-B).
    pub(super) fn store_obj(&mut self, o: ObjectId, expr: &str) {
        if self.expr_only(o) {
            let wrapped = if is_atomic_expr(expr) {
                expr.to_string()
            } else {
                format!("({expr})")
            };
            self.exprs.insert(o, wrapped);
            return;
        }
        if let Some(slot) = self.slot(o) {
            self.line(1, format!("{slot} = {expr};"));
        }
    }
}

/// The source object of the `Pair{slot==k}` edge feeding aggregate `agg`
/// (kernel-side operand gathering for Zip/Index/Update/Fold products).
pub(super) fn pair_component(ir: &CategoryIr, agg: ObjectId, k: u32) -> ObjectId {
    pair_source(ir, agg, k).expect("aggregate component resolves")
}

/// The static integer value of arith operand `k` of a Div/Mod source pair,
/// when the operand arrives as a literal `Pair` edge from a Constant object
/// — suggestions.md #13's constant-divisor knowledge. `None` for every
/// other shape (fn parameters, computed operands, floats, non-pair-fed
/// sources): the guards then emit as usual.
pub(crate) fn const_int_operand(ir: &CategoryIr, source: ObjectId, k: u32) -> Option<i128> {
    let obj = ir
        .object(pair_source(ir, source, k)?)
        .expect("object resolves");
    if obj.kind != ObjectKind::Constant {
        return None;
    }
    match &obj.value {
        Some(Value::I32(n)) => Some(*n as i128),
        Some(Value::I64(n)) => Some(*n as i128),
        Some(Value::U8(n)) => Some(*n as i128),
        _ => None,
    }
}

/// The `<cstdint>` MIN macro for a signed int C++ type (mirrors func.rs's
/// host guard).
pub(super) fn int_min(ct: &str) -> &'static str {
    match ct {
        "int8_t" => "INT8_MIN",
        "int16_t" => "INT16_MIN",
        "int32_t" => "INT32_MIN",
        "int64_t" => "INT64_MIN",
        _ => unreachable!("non-Core int width in Div/Mod"),
    }
}
