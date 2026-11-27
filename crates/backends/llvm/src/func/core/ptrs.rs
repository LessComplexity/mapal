//! component pointers, layout copies, element emission, heap blocks and packed buffers

use super::*;

impl<'a> FnEmit<'a> {
    /// structure: where `o`'s value lives and how it is laid out there — a
    /// pointer to storage whose positions `paths` hold pointers (a forwarded
    /// array: its bytes, `[]`).
    pub(in crate::func) fn value_ptr(&mut self, o: ObjectId) -> Option<(String, Vec<Vec<u32>>)> {
        let slot = self.slot(o)?;
        if self.ptr_resident.contains_key(o) {
            let p = self.tmp();
            self.line(format!("{p} = load ptr, ptr {slot}"));
            return Some((p, Vec::new()));
        }
        Some((slot, self.slot_paths(o)))
    }

    /// structure: [`Self::value_ptr`] of component `k` of product `agg` —
    /// through its `Pair` feeder (as [`Self::array_operand_ptr`] reads), else
    /// the field itself (a pointer field: the bytes it points at).
    pub(in crate::func) fn component_value_ptr(
        &mut self,
        agg: ObjectId,
        k: u32,
    ) -> Option<(String, Vec<Vec<u32>>)> {
        if let Some(f) = self.pair_source(agg, k)
            && self.slots.contains_key(f)
            && !self.host_free(f)
        {
            return self.value_ptr(f);
        }
        let paths = sub_paths(&self.slot_paths(agg), k);
        let (_, fp) = self.component_ptr(agg, k)?;
        if paths.iter().any(|p| p.is_empty()) {
            let p = self.tmp();
            self.line(format!("{p} = load ptr, ptr {fp}"));
            return Some((p, Vec::new()));
        }
        Some((fp, paths))
    }

    /// structure: copy a value of `ty` from storage `src` laid out with
    /// pointers at `src_paths` into storage `dst` laid out with pointers at
    /// `dst_paths` — no aggregate SSA, every array by `memcpy` or by address.
    /// A pointer destination over source bytes takes their address (a call
    /// argument, an element view); a byte destination over a source pointer
    /// copies through it.
    pub(in crate::func) fn copy_layout(
        &mut self,
        ty: &Ty,
        dst: &str,
        dst_paths: &[Vec<u32>],
        src: &str,
        src_paths: &[Vec<u32>],
    ) {
        let d_ptr = dst_paths.iter().any(|p| p.is_empty());
        let s_ptr = src_paths.iter().any(|p| p.is_empty());
        match ty {
            Ty::Array { .. } => {
                let Some(llt) = lower_ty(ty) else {
                    return;
                };
                let from = if s_ptr {
                    let p = self.tmp();
                    self.line(format!("{p} = load ptr, ptr {src}"));
                    p
                } else {
                    src.to_string()
                };
                if d_ptr {
                    self.line(format!("store ptr {from}, ptr {dst}"));
                } else {
                    self.emit_memcpy(dst, &from, &llt);
                }
            }
            Ty::Tuple(_) | Ty::Struct { .. } => {
                let (Some(dllt), Some(sllt)) = (
                    lower_ptr_paths(ty, dst_paths),
                    lower_ptr_paths(ty, src_paths),
                ) else {
                    return;
                };
                for k in 0..product_arity(ty) {
                    let Some(cty) = ty.component_ty(k).cloned() else {
                        continue;
                    };
                    if lower_ty(&cty).is_none() {
                        continue;
                    }
                    let (Some(df), Some(sf)) = (
                        self.field_ptr(dst, ty, &dllt, k),
                        self.field_ptr(src, ty, &sllt, k),
                    ) else {
                        continue;
                    };
                    self.copy_layout(
                        &cty,
                        &df,
                        &sub_paths(dst_paths, k),
                        &sf,
                        &sub_paths(src_paths, k),
                    );
                }
            }
            _ => {
                let Some(llt) = lower_ty(ty) else {
                    return;
                };
                let v = self.tmp();
                self.line(format!("{v} = load {llt}, ptr {src}"));
                self.line(format!("store {llt} {v}, ptr {dst}"));
            }
        }
    }

    /// Load component `k` of aggregate `agg`. `None` if erased.
    pub(in crate::func) fn load_component(
        &mut self,
        agg: ObjectId,
        k: u32,
    ) -> Option<(String, String)> {
        let (cllt, ptr) = self.component_ptr(agg, k)?;
        let v = self.tmp();
        self.line(format!("{v} = load {cllt}, ptr {ptr}"));
        Some((cllt, v))
    }

    /// The element law of `arr` this emission reads (`elem_plan`), or `None`
    /// to load it. gapfix `iota-law`: a capture's law (`elem_plan` through
    /// captures — an object a `Proj` of the body's input defines) is read
    /// unless `--gpu-off=iota-law`, the measurement arm that loads it.
    pub(in crate::func) fn elem_law(&self, arr: ObjectId) -> Option<ElemSrc> {
        let law = self.elem.src(arr)?;
        let capture = self.ir.in_edges(arr).first().is_some_and(|&m| {
            self.ir
                .morphism(m)
                .is_some_and(|pm| matches!(pm.op, Operation::Proj { .. }))
        });
        if capture && self.gpu_arms.off.iota_law {
            return None;
        }
        Some(law.clone())
    }

    /// Whether this backend consumes `ElemSrc::Apply` (recomputing a
    /// classifiable `Map` producer at the read site). Off: see the refusal arm in
    /// [`FnEmit::emit_elem`] for the measurement. Legality lives in mapal-ir;
    /// this constant is the CPU backend's profitability answer, and it is `false`.
    pub(super) const APPLY_INLINE: bool = false;

    /// Materialize `out[i]` from its element law (`mapal_ir::ElemSrc`) instead
    /// of loading it, at loop index `iv`. Returns `(llvm type, value operand)`,
    /// or `None` when the law cannot be realized here — in which case the caller
    /// keeps the load, which is always correct.
    ///
    /// This is the backend half of plan-s37-stage-structure: mapal-ir says what
    /// the element *is* (legality, machine-independent); the decision to inline
    /// it rather than read memory is the backend's, because the right answer
    /// differs per target. On this one it is unconditional for the bodyless
    /// laws — at most two loads and a `trunc`, never worse than the load it
    /// replaces.
    ///
    /// `Broadcast` is emitted inside the loop rather than hoisted by hand: the
    /// value is loop-invariant and LICM lifts it, and hand-hoisting would mean
    /// threading a pre-header through every caller for no measured gain.
    pub(in crate::func) fn emit_elem(
        &mut self,
        law: &ElemSrc,
        elem_ty: &Ty,
        iv: &str,
    ) -> Option<(String, String)> {
        match law {
            // `iota`: the element is the index. ADR-0029 pins the element type
            // to `i32`, so the source-of-truth check is the type, not the tag —
            // a wider iota would need its own conversion and must not silently
            // truncate.
            ElemSrc::Index => {
                // The extent's index type: `i64` past i32::MAX cells.
                if *elem_ty == Ty::i64() {
                    return Some(("i64".to_string(), iv.to_string()));
                }
                if *elem_ty != Ty::i32() {
                    return None;
                }
                let v = self.tmp();
                self.line(format!("{v} = trunc i64 {iv} to i32"));
                Some(("i32".to_string(), v))
            }
            ElemSrc::Broadcast { source, slot } => self.load_component(*source, *slot),
            ElemSrc::Load { source, slot } => {
                let arr_ty = match slot {
                    None => self.obj_ty(*source),
                    Some(k) => self.obj_ty(*source).component_ty(*k).cloned()?,
                };
                let (cllt, _) = array_parts(&arr_ty);
                let arr_llt = lower_ty(&arr_ty)?;
                let ptr = self.array_operand_ptr(*source, *slot)?;
                let ep = self.tmp();
                self.line(format!(
                    "{ep} = getelementptr {arr_llt}, ptr {ptr}, i64 0, i64 {iv}"
                ));
                let v = self.tmp();
                self.line(format!("{v} = load {cllt}, ptr {ep}"));
                Some((cllt, v))
            }
            // `zip` / `enumerate`: build the pair in registers instead of
            // reading it back out of a materialized array of structs.
            ElemSrc::Pair(a, b) => {
                let a_ty = elem_ty.component_ty(0).cloned()?;
                let b_ty = elem_ty.component_ty(1).cloned()?;
                let (a_llt, a_v) = self.emit_elem(a, &a_ty, iv)?;
                let (b_llt, b_v) = self.emit_elem(b, &b_ty, iv)?;
                let pair_llt = lower_ty(elem_ty)?;
                let p0 = self.tmp();
                self.line(format!(
                    "{p0} = insertvalue {pair_llt} poison, {a_llt} {a_v}, 0"
                ));
                let p1 = self.tmp();
                self.line(format!(
                    "{p1} = insertvalue {pair_llt} {p0}, {b_llt} {b_v}, 1"
                ));
                Some((pair_llt, p1))
            }
            // A classifiable `Map` producer: recompute its element by calling
            // the same body the producer calls, on the recursively-built inner
            // element. Nothing is spliced or merged — this is two calls in one
            // loop, which is why capture identity is not required.
            //
            // REFUSED ON THIS TARGET, on measurement (plan-s37-stage-structure
            // Table B — profitability is the backend's call, and this is the
            // backend making it). Recomputing a producer body is only a win
            // when arithmetic is cheaper than the load it replaces. On a CPU
            // with the array already materialized it is not: enabling this arm
            // put two extra calls inside saxpy's timed loop — `fn1` and `fn2`
            // regenerating `x[i]` and `y0[i]` from the index instead of reading
            // them — and cost 0.72x at one thread (0.4731 -> 0.6586 ms min).
            // Gather, the shape it was built for, came out at 1.17x min but
            // 0.97x median: inside noise.
            //
            // The FACT stays in mapal-ir because it is true and machine-
            // independent; only this consumer declines. A bandwidth-bound
            // target where registers are cheap should reach a different verdict
            // — that asymmetry is the whole reason the decision lives here and
            // not in the query. Re-enable behind an op-count test against an L2
            // round trip, with a measurement that moves a published cell.
            ElemSrc::Apply { array, .. } if !Self::APPLY_INLINE => {
                // Decline, but degrade to reading the producer's materialized
                // output rather than failing: a refusal nested inside a `Pair`
                // must not collapse the pair back to an array-of-structs read.
                let arr = *array;
                self.emit_elem(
                    &ElemSrc::Load {
                        source: arr,
                        slot: None,
                    },
                    elem_ty,
                    iv,
                )
            }
            ElemSrc::Apply {
                body,
                source,
                captures,
                inner,
                array: _,
                outside: _,
            } => {
                let src_ty = self.obj_ty(*source);
                let inner_arr_ty = if *captures == 0 {
                    src_ty.clone()
                } else {
                    src_ty.component_ty(*captures).cloned()?
                };
                let inner_elem_ty = inner_arr_ty.component_ty(0).cloned()?;
                let (in_llt, in_v) = self.emit_elem(inner, &inner_elem_ty, iv)?;
                let out_llt = lower_ty(elem_ty)?;
                let callee = self.fnames[*body].clone();
                let arg = if *captures == 0 {
                    format!("{in_llt} {in_v}")
                } else {
                    let arg_ty = self.obj_ty(self.ir.func(*body)?.input);
                    let arg_llt = lower_body_input_ty(&arg_ty, *captures)?;
                    self.body_call_arg(*source, *captures, &arg_ty, &arg_llt, &[(&in_llt, &in_v)])
                };
                let v = self.tmp();
                self.line(format!("{v} = call {out_llt} @{callee}({arg})"));
                Some((out_llt, v))
            }
        }
    }

    /// A by-value product copied as bytes (`memcpy`) rather than as an SSA
    /// aggregate, because it is at least [`TargetProfile::heap_min_bytes`]
    /// long — the size at which its storage already leaves the stack. An SSA
    /// aggregate that size is one whole-array value to the code generator:
    /// clang's SelectionDAG crashed on the 4 MB `{ [1048576 x float], i32 }`
    /// loop state of `loop_mm_1024_8` (S53). Smaller products keep their
    /// load/store, so every smaller program's text is unchanged.
    ///
    /// `Some(llt)` for a plain by-value product (not a forwarded pointer, not
    /// the by-ref input) laid out as its type lowers. Decided from types
    /// alone, so asking emits nothing.
    pub(in crate::func) fn big_aggregate(&self, o: ObjectId) -> Option<String> {
        let ty = self.obj_ty(o);
        if self.ptr_resident.contains_key(o)
            || self.byref.as_ref().is_some_and(|(input, _, _)| *input == o)
            || !matches!(ty, Ty::Tuple(_) | Ty::Struct { .. })
            || !self.slots.contains_key(o)
        {
            return None;
        }
        let llt = lower_ty(&ty)?;
        (self.lower_slot_ty(o, &ty).as_deref() == Some(llt.as_str())
            && llt_bytes(&llt) >= self.profile.heap_min_bytes)
            .then_some(llt)
    }

    /// Store `(llt, val)` into object `o`'s slot, if it has one.
    pub(in crate::func) fn store_obj(&mut self, o: ObjectId, llt: &str, val: &str) {
        if let Some(slot) = self.slot(o) {
            self.line(format!("store {llt} {val}, ptr {slot}"));
        }
    }

    /// A fresh scratch alloca of `llt` in the entry block; returns its ptr name.
    pub(in crate::func) fn scratch(&mut self, llt: &str) -> String {
        let name = format!("%s{}", self.fresh());
        self.allocas.push_str(&format!("  {name} = alloca {llt}\n"));
        name
    }

    /// Does a block of `bytes` belong in the arena rather than the stack
    /// (plan-s29 emission item 4)? Records the teardown debt when it does.
    pub(in crate::func) fn heap_block(&mut self, bytes: u64) -> bool {
        let heap = self.heap_ok && bytes >= self.profile.heap_min_bytes;
        self.heap_used |= heap;
        heap
    }

    /// One named entry-block allocation of `llt`: today's `alloca` (with the
    /// explicit `align` when the site wants one), or an arena block once it
    /// crosses [`TargetProfile::heap_min_bytes`]. An `alloca` and a `mapal_rt_alloc` result are
    /// both just a `ptr`, so every `getelementptr {llt}, ptr …` consumer is
    /// unchanged — the swap is invisible below this line.
    pub(in crate::func) fn entry_alloc(&mut self, name: &str, llt: &str, align: Option<u64>) {
        let bytes = llt_bytes(llt);
        let text = if self.heap_block(bytes) {
            let align = align.unwrap_or_else(|| llt_align(llt));
            let alloc = if self.frame_heap {
                "mapal_rt_frame_alloc"
            } else {
                "mapal_rt_alloc"
            };
            format!("  {name} = call ptr @{alloc}(i64 {bytes}, i64 {align})\n")
        } else {
            match align {
                Some(align) => format!("  {name} = alloca {llt}, align {align}\n"),
                None => format!("  {name} = alloca {llt}\n"),
            }
        };
        self.allocas.push_str(&text);
    }

    /// Release the arena, if this emitter filled any of it. Emitted once, at
    /// the last point that can read arena memory (plan-s29 composition rule 4).
    pub(in crate::func) fn heap_teardown(&mut self) {
        if self.heap_used && self.frame_heap {
            // The per-call arena is runtime state: the fn is no longer readonly.
            self.runtime_write = true;
            self.line("call void @mapal_rt_frame_release(i64 %hmark)");
        } else if self.heap_used {
            self.line("call void @mapal_rt_free_all()");
        }
    }

    pub(in crate::func) fn packed_type(profile: &TargetProfile, site: &TileSite) -> String {
        let tile_j = profile.tile_j(&site.elem);
        let tiles = site.c.div_ceil(tile_j);
        let elems = site
            .k
            .checked_mul(tiles)
            .and_then(|n| n.checked_mul(tile_j))
            .expect("packed tile size fits u64");
        format!(
            "[{elems} x {}]",
            lower_ty(&site.elem).expect("tile element lowers")
        )
    }

    pub(in crate::func) fn packed_buffer(
        &mut self,
        m: MorphismId,
        site: &TileSite,
    ) -> PackedBuffer {
        let llt = Self::packed_type(self.profile, site);
        if let Some(field) = self
            .frame
            .as_ref()
            .and_then(|frame| frame.packed.get(m))
            .cloned()
        {
            let slot = format!("%pack_field{}", field.ordinal);
            let ptr = format!("%packed{}", field.ordinal);
            self.frame_geps.push_str(&format!(
                "  {slot} = getelementptr %Frame, ptr %frame, i32 0, i32 {}\n  {ptr} = load ptr, ptr {slot}\n",
                field.index
            ));
            PackedBuffer { ptr, llt }
        } else {
            let ptr = format!("%s{}", self.fresh());
            self.entry_alloc(&ptr, &llt, Some(64));
            PackedBuffer { ptr, llt }
        }
    }

    pub(in crate::func) fn allocate_frame_packs(&mut self) {
        let Some(frame) = &self.frame else {
            return;
        };
        let Some(plan) = &self.tile_plan else {
            return;
        };
        let packs = frame
            .packed
            .iter()
            .map(|(m, field)| (m, field.clone(), plan.sites[m].clone()))
            .collect::<Vec<_>>();
        for (_, field, site) in packs {
            let llt = Self::packed_type(self.profile, &site);
            let ptr = format!("%pack{}", field.ordinal);
            let slot = format!("%pack_field{}", field.ordinal);
            self.entry_alloc(&ptr, &llt, Some(64));
            self.frame_geps.push_str(&format!(
                "  {slot} = getelementptr %Frame, ptr %frame, i32 0, i32 {}\n  store ptr {ptr}, ptr {slot}\n",
                field.index
            ));
        }
    }
}
