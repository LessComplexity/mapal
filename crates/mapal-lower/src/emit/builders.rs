//! Emitter methods: builder primitive wrappers and helpers.

use super::*;

impl Emitter<'_> {
    // --- builder primitive wrappers (record output ty + error mapping) -----

    pub(super) fn record(&mut self, obj: ObjectId, ty: Ty) -> ObjectId {
        self.obj_ty.insert(obj, ty);
        obj
    }

    /// Propagate the derives-from-merge tag (§7.3): a result is derived iff any
    /// input is derived (within an active loop).
    pub(super) fn prop(&mut self, result: ObjectId, inputs: &[ObjectId]) -> ObjectId {
        if self.loop_depth > 0 && inputs.iter().any(|i| self.derived.contains(i)) {
            self.derived.insert(result);
        }
        result
    }

    pub(super) fn constant(
        &mut self,
        fb: &mut FnBuilder,
        v: Value,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let ty = v.ty();
        let o = fb.constant(v, ir_loc(sp)).map_err(|e| ir_err(e, sp))?;
        Ok(self.record(o, ty))
    }

    pub(super) fn unop(
        &mut self,
        fb: &mut FnBuilder,
        op: Operation,
        x: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let xty = self.tyq(x);
        let out = match op {
            Operation::Not => Ty::Bool,
            _ => xty,
        };
        let o = fb
            .unop(op, x, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[x]);
        Ok(self.record(o, out))
    }

    pub(super) fn binop(
        &mut self,
        fb: &mut FnBuilder,
        op: Operation,
        l: ObjectId,
        r: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let lty = self.tyq(l);
        let out = match op {
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                lty
            }
            _ => Ty::Bool,
        };
        let o = fb
            .binop(op, l, r, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        fb.admit(o, self.prec);
        self.prop(o, &[l, r]);
        Ok(self.record(o, out))
    }

    pub(super) fn phi(
        &mut self,
        fb: &mut FnBuilder,
        t: ObjectId,
        f: ObjectId,
        c: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let ty = self.tyq(t);
        let o = fb
            .phi(t, f, c, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[t, f, c]);
        Ok(self.record(o, ty))
    }

    pub(super) fn proj(
        &mut self,
        fb: &mut FnBuilder,
        src: ObjectId,
        idx: u32,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let sty = self.tyq(src);
        let out = sty.component_ty(idx).cloned().unwrap_or(Ty::Unit);
        let o = fb
            .proj(src, idx, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[src]);
        Ok(self.record(o, out))
    }

    pub(super) fn index(
        &mut self,
        fb: &mut FnBuilder,
        a: ObjectId,
        i: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        // ADR-0048 D5: an indexed struct reads through its `_` array.
        let a = self
            .read_default(fb, a, |t| matches!(t, Ty::Array { .. }), sp)?
            .unwrap_or(a);
        let aty = self.tyq(a);
        let out = match &aty {
            Ty::Array { elem, .. } => (**elem).clone(),
            _ => Ty::Unit,
        };
        let o = fb
            .index(a, i, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[a, i]);
        Ok(self.record(o, out))
    }

    /// `Update(a, i, v)` (ADR-0021): the pure element-write op. Result ty = the
    /// array's ty (a fresh array, slot `i` replaced). Mirrors `index`: the
    /// builder owns the shape checks (non-array → `NotAProduct`/L1204, element or
    /// index-ty clash → `TypeMismatch`/L1201, both via `ir_err`).
    pub(super) fn update(
        &mut self,
        fb: &mut FnBuilder,
        a: ObjectId,
        i: ObjectId,
        v: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let out = self.tyq(a);
        let o = fb
            .update(a, i, v, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[a, i, v]);
        Ok(self.record(o, out))
    }

    pub(crate) fn pack(
        &mut self,
        fb: &mut FnBuilder,
        comps: &[ObjectId],
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let tys: Vec<Ty> = comps.iter().map(|&c| self.tyq(c)).collect();
        let out = Ty::Tuple(tys);
        let o = fb.pack(comps, dest, ir_loc(sp))?;
        self.prop(o, comps);
        Ok(self.record(o, out))
    }

    pub(super) fn pack_array(
        &mut self,
        fb: &mut FnBuilder,
        comps: &[ObjectId],
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let elem = comps.first().map(|&c| self.tyq(c)).unwrap_or(Ty::Unit);
        let out = Ty::Array {
            elem: Box::new(elem),
            size: comps.len() as u64,
        };
        let o = fb
            .pack_array(comps, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, comps);
        Ok(self.record(o, out))
    }

    pub(super) fn pack_struct(
        &mut self,
        fb: &mut FnBuilder,
        ty: Ty,
        comps: &[ObjectId],
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let o = fb.pack_struct(ty.clone(), comps, dest, ir_loc(sp))?;
        self.packed.insert(o, comps.to_vec());
        Ok(self.record(o, ty))
    }

    // --- helpers ------------------------------------------------------------

    /// Query the recorded ty of an object (default `Unit` if not tracked).
    pub(super) fn tyq(&self, obj: ObjectId) -> Ty {
        self.obj_ty.get(&obj).cloned().unwrap_or(Ty::Unit)
    }

    /// Bind a new surface symbol with a fresh declaration sequence number.
    pub(super) fn bind_new(&mut self, name: &str, obj: ObjectId, ty: Ty, mutable: bool) {
        if let Some(b) = self.mono.bound
            && self.owner_name == "main"
        {
            b.borrow_mut().insert(name.to_string(), obj);
        }
        let seq = self.decl_seq;
        self.decl_seq += 1;
        self.scope.bind(
            name,
            Binding {
                obj,
                ty,
                mutable,
                decl_seq: seq,
            },
        );
    }

    /// Bind a new surface symbol into the loop's *enclosing* scope (one frame
    /// below the loop-body frame). A routing-guard exit arm binds its exit value
    /// here so it is visible after the loop (DESIGN §8.5).
    pub(super) fn bind_new_enclosing(&mut self, name: &str, obj: ObjectId, ty: Ty, mutable: bool) {
        let seq = self.decl_seq;
        self.decl_seq += 1;
        self.scope.bind_enclosing(
            name,
            Binding {
                obj,
                ty,
                mutable,
                decl_seq: seq,
            },
        );
    }

    /// Rebind an existing symbol to a fresh object, preserving its declaration
    /// sequence (the one-definition rule allocates a fresh object; the symbol's
    /// declaration order does not change, LD4).
    pub(super) fn rebind(&mut self, name: &str, obj: ObjectId, ty: Ty, mutable: bool) {
        let seq = self
            .scope
            .resolve(name)
            .map(|(b, _)| b.decl_seq)
            .unwrap_or(self.decl_seq);
        self.scope.rebind_existing(
            name,
            Binding {
                obj,
                ty,
                mutable,
                decl_seq: seq,
            },
        );
    }

    pub(crate) fn ty_of(&self, _fb: &FnBuilder, obj: ObjectId) -> Ty {
        self.tyq(obj)
    }
}
