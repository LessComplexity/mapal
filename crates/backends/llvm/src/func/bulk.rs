//! the bulk collection ops: map, fold, zip, enumerate, iota, fill
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

impl<'a> FnEmit<'a> {
    /// S44's **move-panel traversal**: permute a bulk `map`'s loop counter so the
    /// iteration space is visited in `b × b` tiles of a `w`-wide 2-D geometry
    /// instead of in linear order. Returns the index to *use* on this trip; the
    /// loop counter, its bounds and its trip count are untouched.
    ///
    /// **A permutation of the counter, not a loop nest** — and that is the whole
    /// correctness argument. `perm` is a bijection of `[0, n)`, the parallel
    /// slices partition the counter, so the images of those slices still
    /// partition the outputs: every element is visited exactly once, by exactly
    /// one worker, and the values are bit-identical. It also means `%lo`/`%hi`
    /// need no special handling — a blocked *nest* would have to grow a head and
    /// a tail for the partial rows at a slice boundary, and this does not.
    ///
    /// ```text
    /// p  = ((rb·CB + cb)·B + dr)·B + dc      -- the counter, decomposed
    /// t  = (rb·B + dr)·W + cb·B + dc         -- the index it stands for
    /// ```
    ///
    /// Every divisor is a compile-time constant, so `-O2` turns the `udiv`/`urem`
    /// pair into shifts when `b` is a power of two.
    ///
    /// Declines — returning the counter unchanged — unless the panel divides the
    /// geometry both ways. A panel that does not tile the space would need a
    /// remainder arm, and the honest answer at that point is that this rung has
    /// nothing to say about that shape.
    /// **Whether this map gets the move panel, and with what block** — the join
    /// of the two records, and the whole of S45's answer to "delete the flag".
    ///
    /// The graph half is `mapal_ir::MoveSite` (a fold-less map whose read
    /// address is affine in `(t÷C, t%C)`), the machine half is the profile's
    /// L1D geometry, and neither crate may hold both (ADR-0032). `Force` keeps
    /// S44's behaviour for the B-sweep; `Off` is the A/B's OFF arm.
    fn move_panel_of(&self, m: MorphismId) -> Option<(u64, u64)> {
        match self.move_panel {
            MovePanel::Off => None,
            MovePanel::Force(w, b) => Some((w, b)),
            MovePanel::Deduce => {
                let site = self.move_sites.sites.get(m)?;
                Some((site.width, self.profile.move_block(site)?))
            }
        }
    }

    fn move_panel_index(&mut self, iv: &str, n: u64, w: u64, b: u64) -> String {
        if w == 0 || b == 0 || n % w != 0 {
            return iv.to_owned();
        }
        let rows = n / w;
        if rows < 2 || !w.is_multiple_of(b) || !rows.is_multiple_of(b) {
            return iv.to_owned();
        }
        let col_blocks = w / b;
        let op = |emit: &mut Self, code: &str, lhs: &str, rhs: u64| {
            let t = emit.tmp();
            emit.line(format!("{t} = {code} i64 {lhs}, {rhs}"));
            t
        };
        let dc = op(self, "urem", iv, b);
        let q = op(self, "udiv", iv, b);
        let dr = op(self, "urem", &q, b);
        let q2 = op(self, "udiv", &q, b);
        let cb = op(self, "urem", &q2, col_blocks);
        let rb = op(self, "udiv", &q2, col_blocks);
        let row = op(self, "mul", &rb, b);
        let row = {
            let t = self.tmp();
            self.line(format!("{t} = add i64 {row}, {dr}"));
            t
        };
        let row = op(self, "mul", &row, w);
        let col = op(self, "mul", &cb, b);
        let idx = self.tmp();
        self.line(format!("{idx} = add i64 {row}, {col}"));
        let out = self.tmp();
        self.line(format!("{out} = add i64 {idx}, {dc}"));
        out
    }

    /// The set of a host map whose law is piecewise ([`mapal_ir::Outside`]):
    /// the plain body-call loop (no tile site, no device unit), under the
    /// `range-map` rung.
    pub(super) fn range_map(&self, m: MorphismId) -> Option<mapal_ir::Outside> {
        if !super::gpu::host_rungs(self.profile, self.contract, self.gpu_arms).range_map {
            return None;
        }
        let tiled = self.tile_plan.as_ref().is_some_and(|p| p.sites.contains_key(m));
        let cell = self
            .resident
            .as_ref()
            .is_some_and(|p| p.cells.contains_key(m) || p.devgen.contains_key(m));
        if tiled || cell || self.device.contains_key(m) || self.member_src(m, 0).is_some() {
            return None;
        }
        match self.ir.map_law(m)? {
            ElemSrc::Apply {
                outside: Some(ou), ..
            } => Some(ou),
            _ => None,
        }
    }

    /// The cells `lo + step·k` met with the loop's own bounds `[l, h)`: the
    /// first at or past `l`, up to `min(lo + step·len, h)` — `lo = d·(a·e +
    /// b)`, `a·e + b` in `e`'s wrapping width (the program's value), then
    /// exact in `i64`. A step of 1 is the interval `[max(lo, l), …)`.
    fn range_bounds(&mut self, ou: &mapal_ir::Outside, n: u64) -> (String, String) {
        let (l, h) = self.bulk_bounds(n);
        let x = match ou.e.and_then(|e| self.load_whole(e)) {
            Some((wllt, ev)) => self.affine_cell(
                &wllt,
                &ev,
                &mapal_ir::Affine {
                    var: ou.e.expect("e"),
                    a: ou.a,
                    b: ou.b,
                },
            ),
            None if ou.e.is_none() => ou.b.to_string(),
            // Unloadable: the whole map, in place (outside cells are their own).
            None => return (l, h),
        };
        let lo = self.tmp();
        self.line(format!("{lo} = mul i64 {x}, {}", ou.d));
        let pick = |s: &mut Self, cmp: &str, p: &str, q: &str| {
            let (c, t) = (s.tmp(), s.tmp());
            s.line(format!("{c} = icmp {cmp} i64 {p}, {q}"));
            s.line(format!("{t} = select i1 {c}, i64 {p}, i64 {q}"));
            t
        };
        let start = if ou.step == 1 {
            pick(self, "sgt", &lo, &l)
        } else {
            // `lo + step·⌈(l − lo) / step⌉` when `lo < l` (`l − lo > 0`).
            let s = ou.step;
            let (gap, up, k, off, at) = (self.tmp(), self.tmp(), self.tmp(), self.tmp(), self.tmp());
            self.line(format!("{gap} = sub i64 {l}, {lo}"));
            self.line(format!("{up} = add i64 {gap}, {}", s - 1));
            self.line(format!("{k} = udiv i64 {up}, {s}"));
            self.line(format!("{off} = mul i64 {k}, {s}"));
            self.line(format!("{at} = add i64 {lo}, {off}"));
            let c = self.tmp();
            self.line(format!("{c} = icmp slt i64 {lo}, {l}"));
            let t = self.tmp();
            self.line(format!("{t} = select i1 {c}, i64 {at}, i64 {lo}"));
            t
        };
        let end = match ou.step.checked_mul(ou.len).filter(|&w| w <= n) {
            Some(w) => {
                let hi = self.tmp();
                self.line(format!("{hi} = add i64 {lo}, {w}"));
                pick(self, "slt", &hi, &h)
            }
            None => h,
        };
        let end = pick(self, "sgt", &end, &start);
        (start, end)
    }

    pub(super) fn emit_map(
        &mut self,
        m: MorphismId,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) {
        if let Some(site) = self
            .tile_plan
            .as_ref()
            .and_then(|plan| plan.sites.get(m))
            // Non-conv k-split sites keep the untiled body-call fallback
            // (rule 3): the affine tile emission ignores `ksplit` and would
            // compute wrong addresses. Conv-shaped k-split sites pass through
            // to the unrolled micro-kernel.
            .filter(|site| (site.a.ksplit.is_none() && site.b.ksplit.is_none()) || conv_site(site))
            .cloned()
        {
            let packed = if self.packing
                && !self.device.contains_key(m)
                && packing_site(&site, self.profile, self.contract)
            {
                let needs_pack = self
                    .frame
                    .as_ref()
                    .and_then(|frame| frame.packed.get(m))
                    .is_none();
                let packed = self.packed_buffer(m, &site);
                if needs_pack {
                    self.emit_pack_copy(source, &site, &packed);
                }
                Some(packed)
            } else {
                None
            };
            self.emit_tiled_map(source, target, &site, packed);
            return;
        }

        let src_ty = self.obj_ty(source);
        // The mapped array: the bare source (k=0) or the source product's last
        // component (k>0 — ADR-0027: source `(c₁…cₖ, [T; n])`, captures leading).
        // The pointer is taken LAZILY: an elided array (step 3b) has no
        // `%Frame` field, so asking for its slot would panic — and the whole
        // point is that this path never needs it.
        let (arr_ty, arr_slot) = if captures == 0 {
            (src_ty, None)
        } else {
            (
                src_ty.component_ty(captures).cloned().expect("map array"),
                Some(captures),
            )
        };
        let (tllt, n) = array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);
        let (ullt, _) = array_parts(&tgt_ty);
        let tgt_arr_llt = lower_ty(&tgt_ty).expect("map tgt lowers");
        // A piecewise map written in place (`in_place_source`): the capture's
        // buffer is the output's, and only the interval is computed — every
        // other cell already holds its value.
        let range = self
            .update_aliases
            .get(target)
            .copied()
            .zip(self.range_map(m));
        let tgt_slot = match range {
            Some((arr, _)) => {
                let slot = self.slot(arr).expect("in-place map source slot");
                self.slots.insert(target, slot.clone());
                if self.ptr_resident.contains_key(arr) {
                    let p = self.tmp();
                    self.line(format!("{p} = load ptr, ptr {slot}"));
                    p
                } else {
                    slot
                }
            }
            None => self.slot(target).expect("map tgt slot"),
        };
        let callee = self.fnames[body].clone();
        let ctr = self.scratch("i64");
        let (lo, hi) = match range {
            Some((_, ou)) => self.range_bounds(&ou, n),
            None => self.bulk_bounds(n),
        };

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("store i64 {lo}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        // S44/S45: the index this trip stands for. A declined site — no record,
        // or a profile that prices it as no win — returns `iv` itself, so the
        // text below is character-identical to what it has always been.
        let ix = match self.move_panel_of(m).filter(|_| range.is_none()) {
            Some((w, b)) => self.move_panel_index(&iv, n, w, b),
            None => iv.clone(),
        };
        // plan-s37-stage-structure: if `elem_plan` knows what `arr[i]` IS, build
        // it here instead of reading it back out of memory. The intermediate
        // array is still emitted — this is the query, not a rewrite; whether the
        // buffer survives is a separate (backend-owned) decision. `None` keeps
        // the load, which is what every case did before and is always correct.
        let mapped = if captures == 0 {
            Some(source)
        } else {
            self.pair_source(source, captures)
        };
        let law = mapped.and_then(|o| self.elem.src(o)).cloned();
        let inlined = law
            .filter(|l| !matches!(l, ElemSrc::Load { .. }))
            .zip(arr_ty.component_ty(0).cloned())
            .and_then(|(l, elem_ty)| self.emit_elem(&l, &elem_ty, &ix));
        let e = match inlined {
            Some((_, v)) => v,
            None => {
                let src_arr_llt = lower_ty(&arr_ty).expect("map src lowers");
                let arr_ptr = self
                    .array_operand_ptr(source, arr_slot)
                    .expect("map src slot");
                let ep = self.tmp();
                self.line(format!(
                    "{ep} = getelementptr {src_arr_llt}, ptr {arr_ptr}, i64 0, i64 {ix}"
                ));
                let e = self.tmp();
                self.line(format!("{e} = load {tllt}, ptr {ep}"));
                e
            }
        };
        // The body call's argument: the bare element (k=0) or the assembled
        // `(c₁…cₖ, elem)` product (k>0), per the body fn's input ty — with
        // Array captures by reference, matching the body fn's signature.
        let arg = if captures == 0 {
            format!("{tllt} {e}")
        } else {
            let arg_ty = self.obj_ty(self.ir.func(body).expect("map body").input);
            let arg_llt = lower_body_input_ty(&arg_ty, captures).expect("map body input lowers");
            self.body_call_arg(source, captures, &arg_ty, &arg_llt, &[(&tllt, &e)])
        };
        let r = self.tmp();
        self.line(format!("{r} = call {ullt} @{callee}({arg})"));
        let dp = self.tmp();
        self.line(format!(
            "{dp} = getelementptr {tgt_arr_llt}, ptr {tgt_slot}, i64 0, i64 {ix}"
        ));
        self.line(format!("store {ullt} {r}, ptr {dp}"));
        let iv1 = self.tmp();
        let step = range.map_or(1, |(_, ou)| ou.step);
        self.line(format!("{iv1} = add i64 {iv}, {step}"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
    }

    pub(super) fn emit_fold(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) {
        let src_ty = self.obj_ty(source);
        // ADR-0027: source `(c₁…cₖ, Acc, [T; n])` (k=0: `(Acc, [T; n])`) — the
        // accumulator is component k, the folded array component k+1.
        let arr_ty = src_ty
            .component_ty(captures + 1)
            .cloned()
            .expect("fold array");
        let (tllt, n) = array_parts(&arr_ty);
        let arr_llt = lower_ty(&arr_ty).expect("fold array lowers");
        let (acc_llt, acc0) = self.load_component(source, captures).expect("fold acc");
        let arr_ptr = self
            .array_operand_ptr(source, Some(captures + 1))
            .expect("fold array ptr");

        let callee = self.fnames[body].clone();
        let pair_ty = self.obj_ty(self.ir.func(body).expect("fold body").input);
        let pair_llt = lower_body_input_ty(&pair_ty, captures).expect("fold pair lowers");

        let accslot = self.scratch(&acc_llt);
        let ctr = self.scratch("i64");
        let (lo, hi) = self.bulk_bounds(n);
        self.line(format!("store {acc_llt} {acc0}, ptr {accslot}"));
        self.line(format!("store i64 {lo}, ptr {ctr}"));

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        // Same element-law consumption as `emit_map` (plan-s37-stage-structure):
        // a fold over an `iota`/`fill`/`zip` reads the law, not the array. The
        // accumulator chain is untouched — order and arity are exactly as
        // before, so the fold's value semantics cannot move.
        let folded = self.pair_source(source, captures + 1);
        let law = folded.and_then(|o| self.elem.src(o)).cloned();
        let inlined = law
            .filter(|l| !matches!(l, ElemSrc::Load { .. }))
            .zip(arr_ty.component_ty(0).cloned())
            .and_then(|(l, elem_ty)| self.emit_elem(&l, &elem_ty, &iv));
        let e = match inlined {
            Some((_, v)) => v,
            None => {
                let ep = self.tmp();
                self.line(format!(
                    "{ep} = getelementptr {arr_llt}, ptr {arr_ptr}, i64 0, i64 {iv}"
                ));
                let e = self.tmp();
                self.line(format!("{e} = load {tllt}, ptr {ep}"));
                e
            }
        };
        let a = self.tmp();
        self.line(format!("{a} = load {acc_llt}, ptr {accslot}"));
        // The step call's argument: the `(c₁…cₖ, acc, elem)` product (k=0:
        // `(acc, elem)`), assembled in scratch per the body fn's input ty.
        let arg = self.body_call_arg(
            source,
            captures,
            &pair_ty,
            &pair_llt,
            &[(&acc_llt, &a), (&tllt, &e)],
        );
        let na = self.tmp();
        self.line(format!("{na} = call {acc_llt} @{callee}({arg})"));
        self.line(format!("store {acc_llt} {na}, ptr {accslot}"));
        let iv1 = self.tmp();
        self.line(format!("{iv1} = add i64 {iv}, 1"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
        let fin = self.tmp();
        self.line(format!("{fin} = load {acc_llt}, ptr {accslot}"));
        self.store_obj(target, &acc_llt, &fin);
    }

    pub(super) fn emit_zip(&mut self, source: ObjectId, target: ObjectId) {
        let src_ty = self.obj_ty(source);
        let a_ty = src_ty.component_ty(0).cloned().expect("zip a");
        let b_ty = src_ty.component_ty(1).cloned().expect("zip b");
        let (allt, n) = array_parts(&a_ty);
        let (bllt, _) = array_parts(&b_ty);
        let a_arr_llt = lower_ty(&a_ty).expect("zip a lowers");
        let b_arr_llt = lower_ty(&b_ty).expect("zip b lowers");
        let a_ptr = self.array_operand_ptr(source, Some(0)).expect("zip a ptr");
        let b_ptr = self.array_operand_ptr(source, Some(1)).expect("zip b ptr");

        let tgt_ty = self.obj_ty(target);
        let elem_ty = tgt_ty.component_ty(0).cloned().expect("zip elem");
        let tgt_arr_llt = lower_ty(&tgt_ty).expect("zip tgt lowers");
        let tgt_slot = self.slot(target).expect("zip tgt slot");
        let ctr = self.scratch("i64");
        let (lo, hi) = self.bulk_bounds(n);

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("store i64 {lo}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        let ea = {
            let p = self.tmp();
            self.line(format!(
                "{p} = getelementptr {a_arr_llt}, ptr {a_ptr}, i64 0, i64 {iv}"
            ));
            let v = self.tmp();
            self.line(format!("{v} = load {allt}, ptr {p}"));
            v
        };
        let eb = {
            let p = self.tmp();
            self.line(format!(
                "{p} = getelementptr {b_arr_llt}, ptr {b_ptr}, i64 0, i64 {iv}"
            ));
            let v = self.tmp();
            self.line(format!("{v} = load {bllt}, ptr {p}"));
            v
        };
        let dp = self.tmp();
        self.line(format!(
            "{dp} = getelementptr {tgt_arr_llt}, ptr {tgt_slot}, i64 0, i64 {iv}"
        ));
        let elem_llt = lower_ty(&elem_ty).expect("zip elem lowers");
        self.field_store(&dp, &elem_ty, &elem_llt, 0, &allt, &ea);
        self.field_store(&dp, &elem_ty, &elem_llt, 1, &bllt, &eb);
        let iv1 = self.tmp();
        self.line(format!("{iv1} = add i64 {iv}, 1"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
    }

    pub(super) fn emit_enumerate(&mut self, source: ObjectId, target: ObjectId) {
        let src_ty = self.obj_ty(source);
        let (allt, n) = array_parts(&src_ty);
        let src_arr_llt = lower_ty(&src_ty).expect("enum src lowers");
        let src_slot = self.array_operand_ptr(source, None).expect("enum src ptr");

        let tgt_ty = self.obj_ty(target);
        let elem_ty = tgt_ty.component_ty(0).cloned().expect("enum elem");
        let tgt_arr_llt = lower_ty(&tgt_ty).expect("enum tgt lowers");
        let tgt_slot = self.slot(target).expect("enum tgt slot");
        let ctr = self.scratch("i64");
        let (lo, hi) = self.bulk_bounds(n);

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("store i64 {lo}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        let idx32 = self.tmp();
        self.line(format!("{idx32} = trunc i64 {iv} to i32"));
        let ep = self.tmp();
        self.line(format!(
            "{ep} = getelementptr {src_arr_llt}, ptr {src_slot}, i64 0, i64 {iv}"
        ));
        let ea = self.tmp();
        self.line(format!("{ea} = load {allt}, ptr {ep}"));
        let dp = self.tmp();
        self.line(format!(
            "{dp} = getelementptr {tgt_arr_llt}, ptr {tgt_slot}, i64 0, i64 {iv}"
        ));
        let elem_llt = lower_ty(&elem_ty).expect("enum elem lowers");
        self.field_store(&dp, &elem_ty, &elem_llt, 0, "i32", &idx32);
        self.field_store(&dp, &elem_ty, &elem_llt, 1, &allt, &ea);
        let iv1 = self.tmp();
        self.line(format!("{iv1} = add i64 {iv}, 1"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
    }

    /// `Iota` (ADR-0029): `out[i] = (i32)i`. The count is the (builder-minted)
    /// constant object; `n` rides the target type (validate ties them), so no
    /// source read is needed. Trap-free by construction.
    pub(super) fn emit_iota(&mut self, _source: ObjectId, target: ObjectId) {
        let tgt_ty = self.obj_ty(target);
        let (_, n) = array_parts(&tgt_ty);
        let tgt_arr_llt = lower_ty(&tgt_ty).expect("iota tgt lowers");
        let tgt_slot = self.slot(target).expect("iota tgt slot");
        let ctr = self.scratch("i64");
        let (lo, hi) = self.bulk_bounds(n);

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("store i64 {lo}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        let idx32 = self.tmp();
        self.line(format!("{idx32} = trunc i64 {iv} to i32"));
        let dp = self.tmp();
        self.line(format!(
            "{dp} = getelementptr {tgt_arr_llt}, ptr {tgt_slot}, i64 0, i64 {iv}"
        ));
        self.line(format!("store i32 {idx32}, ptr {dp}"));
        let iv1 = self.tmp();
        self.line(format!("{iv1} = add i64 {iv}, 1"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
    }

    /// `Fill` (ADR-0029): `out[i] = x` — the internal (x, count) pair feeds
    /// the value; `n` rides the target type (validate ties them). Trap-free.
    pub(super) fn emit_fill(&mut self, source: ObjectId, target: ObjectId) {
        let tgt_ty = self.obj_ty(target);
        let (_, n) = array_parts(&tgt_ty);
        let tgt_arr_llt = lower_ty(&tgt_ty).expect("fill tgt lowers");
        let tgt_slot = self.slot(target).expect("fill tgt slot");
        let (vllt, v) = self.load_component(source, 0).expect("fill value");
        let ctr = self.scratch("i64");
        let (lo, hi) = self.bulk_bounds(n);

        let (lh, lb, ld) = (self.label(), self.label(), self.label());
        self.line(format!("store i64 {lo}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&lh);
        let iv = self.tmp();
        self.line(format!("{iv} = load i64, ptr {ctr}"));
        let done = self.tmp();
        self.line(format!("{done} = icmp uge i64 {iv}, {hi}"));
        self.line(format!("br i1 {done}, label %{ld}, label %{lb}"));
        self.label_line(&lb);
        let dp = self.tmp();
        self.line(format!(
            "{dp} = getelementptr {tgt_arr_llt}, ptr {tgt_slot}, i64 0, i64 {iv}"
        ));
        self.line(format!("store {vllt} {v}, ptr {dp}"));
        let iv1 = self.tmp();
        self.line(format!("{iv1} = add i64 {iv}, 1"));
        self.line(format!("store i64 {iv1}, ptr {ctr}"));
        self.line(format!("br label %{lh}"));
        self.label_line(&ld);
    }

    // --- loop-driver hooks (used by loops.rs) ----------------------------

    /// Copy object `from`'s whole value into object `to`'s slot (init→merge,
    /// next→merge). No-op if either side is erased.
    pub(crate) fn copy_obj(&mut self, from: ObjectId, to: ObjectId) {
        if matches!(self.obj_ty(from), Ty::Array { .. }) {
            let src = self
                .array_operand_ptr(from, None)
                .expect("array copy source ptr");
            let dst = self.slot(to).expect("array copy target slot");
            if self.ptr_resident.contains_key(to) {
                // S65 `by-ref`: the state is a view of the init's bytes.
                self.line(format!("store ptr {src}, ptr {dst}"));
                return;
            }
            let llt = lower_ty(&self.obj_ty(from)).expect("array copy lowers");
            self.emit_memcpy(&dst, &src, &llt);
        } else if let Some(llt) = self.big_aggregate(from)
            && self.big_aggregate(to).as_ref() == Some(&llt)
        {
            let (dst, src) = (self.slots[to].clone(), self.slots[from].clone());
            self.emit_memcpy(&dst, &src, &llt);
        } else if let Some((llt, val)) = self.load_whole(from) {
            self.store_obj(to, &llt, &val);
        }
    }

    /// Copy component `k` of aggregate `route` into object `to`'s slot
    /// (exit payload → exit object). No-op if erased.
    pub(crate) fn copy_component(&mut self, route: ObjectId, k: u32, to: ObjectId) {
        if matches!(self.obj_ty(route).component_ty(k), Some(Ty::Array { .. })) {
            // S65 `by-ref`: a device carry's exit is filled on the device (the
            // host's state holds no bytes), and a host-carried one takes the
            // state's pointer.
            let device = self.by_ref
                && self
                    .resident
                    .as_ref()
                    .is_some_and(|p| p.carries.iter().any(|c| c.exit == Some(to)));
            if device || self.host_free(to) {
                return;
            }
            let src = self
                .array_operand_ptr(route, Some(k))
                .expect("array route source ptr");
            let dst = self.slot(to).expect("array route target slot");
            if self.ptr_resident.contains_key(to) {
                self.line(format!("store ptr {src}, ptr {dst}"));
                return;
            }
            let llt = lower_ty(&self.obj_ty(to)).expect("array route lowers");
            self.emit_memcpy(&dst, &src, &llt);
        } else if let Some(llt) = self.big_aggregate(to)
            && self.obj_ty(route).component_ty(k) == Some(&self.obj_ty(to))
            && let Some((_, src)) = self.component_ptr(route, k)
        {
            let dst = self.slots[to].clone();
            self.emit_memcpy(&dst, &src, &llt);
        } else if let Some((llt, val)) = self.load_component(route, k) {
            self.store_obj(to, &llt, &val);
        }
    }

    /// Load component `k` of `route` as a bare operand (the loop guard bool).
    pub(crate) fn load_route_component(&mut self, route: ObjectId, k: u32) -> String {
        self.load_component(route, k).expect("route component").1
    }
}
