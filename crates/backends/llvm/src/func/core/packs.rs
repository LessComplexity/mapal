//! pack copies, slot types, field pointers, memcpy, operand pointers, trap sites and row windows

use super::*;

impl<'a> FnEmit<'a> {
    /// Copy one row-invariant b operand to packed[j-tile][k][lane], padding
    /// the final panel's dead lanes with zero.
    pub(in crate::func) fn emit_pack_copy(
        &mut self,
        source: ObjectId,
        site: &TileSite,
        packed: &PackedBuffer,
    ) {
        debug_assert!(packing_site(site, self.profile, self.contract));
        let source_ty = self.obj_ty(source);
        let b_ty = source_ty
            .component_ty(site.b.slot)
            .cloned()
            .expect("tile b array");
        let b_llt = lower_ty(&b_ty).expect("tile b lowers");
        let elem_llt = lower_ty(&site.elem).expect("tile element lowers");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        let tile_j = self.profile.tile_j(&site.elem);
        let tiles = site.c.div_ceil(tile_j);
        let panel_elems = site.k * tile_j;
        let jt_ctr = self.scratch("i64");
        let k_ctr = self.scratch("i64");
        let lane_ctr = self.scratch("i64");
        let (jt_head, jt_body, jt_done) = (self.label(), self.label(), self.label());
        let (k_head, k_body, k_done) = (self.label(), self.label(), self.label());
        let (lane_head, lane_body, lane_done) = (self.label(), self.label(), self.label());
        let (load, pad, store_done) = (self.label(), self.label(), self.label());

        // plan-s43-parallel-bpack §2: the j-tile axis is the pack's OWN split
        // axis. Panels are disjoint by construction (`panel_base = jt·k·tile_j`,
        // reads shared read-only), so any cut at a whole `jt` is sound. At
        // `split_range == false` this is character-identical to the `0`/`tiles`
        // literals it replaces, which is what keeps the sequential inline pack
        // (`bulk.rs`) byte-for-byte unchanged.
        let (jt_lo, jt_hi) = self.bulk_bounds(tiles);
        self.line(format!("store i64 {jt_lo}, ptr {jt_ctr}"));
        self.line(format!("br label %{jt_head}"));
        self.label_line(&jt_head);
        let jt = self.tmp();
        self.line(format!("{jt} = load i64, ptr {jt_ctr}"));
        let all_tiles = self.tmp();
        self.line(format!("{all_tiles} = icmp uge i64 {jt}, {jt_hi}"));
        self.line(format!(
            "br i1 {all_tiles}, label %{jt_done}, label %{jt_body}"
        ));
        self.label_line(&jt_body);
        let j0 = self.tmp();
        self.line(format!("{j0} = mul i64 {jt}, {tile_j}"));
        let panel_base = self.tmp();
        self.line(format!("{panel_base} = mul i64 {jt}, {panel_elems}"));
        self.line(format!("store i64 0, ptr {k_ctr}"));
        self.line(format!("br label %{k_head}"));

        self.label_line(&k_head);
        let k = self.tmp();
        self.line(format!("{k} = load i64, ptr {k_ctr}"));
        let all_k = self.tmp();
        self.line(format!("{all_k} = icmp uge i64 {k}, {}", site.k));
        self.line(format!("br i1 {all_k}, label %{k_done}, label %{k_body}"));
        self.label_line(&k_body);
        let packed_k = self.tmp();
        self.line(format!("{packed_k} = mul i64 {k}, {tile_j}"));
        let packed_row = self.tmp();
        self.line(format!("{packed_row} = add i64 {panel_base}, {packed_k}"));
        self.line(format!("store i64 0, ptr {lane_ctr}"));
        self.line(format!("br label %{lane_head}"));

        self.label_line(&lane_head);
        let lane = self.tmp();
        self.line(format!("{lane} = load i64, ptr {lane_ctr}"));
        let all_lanes = self.tmp();
        self.line(format!("{all_lanes} = icmp uge i64 {lane}, {tile_j}"));
        self.line(format!(
            "br i1 {all_lanes}, label %{lane_done}, label %{lane_body}"
        ));
        self.label_line(&lane_body);
        let j = self.tmp();
        self.line(format!("{j} = add i64 {j0}, {lane}"));
        let packed_index = self.tmp();
        self.line(format!("{packed_index} = add i64 {packed_row}, {lane}"));
        let packed_ptr = self.tmp();
        self.line(format!(
            "{packed_ptr} = getelementptr {}, ptr {}, i64 0, i64 {packed_index}",
            packed.llt, packed.ptr
        ));
        let live = self.tmp();
        self.line(format!("{live} = icmp ult i64 {j}, {}", site.c));
        self.line(format!("br i1 {live}, label %{load}, label %{pad}"));

        self.label_line(&load);
        let b_index = self
            .emit_tile_index(
                (site.b.base != 0).then(|| site.b.base.to_string()),
                &[(site.b.ck, k.as_str()), (1, j.as_str())],
            )
            .expect("tile b has lane term");
        let b_elem_ptr = self.tmp();
        self.line(format!(
            "{b_elem_ptr} = getelementptr {b_llt}, ptr {b_ptr}, i64 0, i64 {b_index}"
        ));
        let value = self.tmp();
        self.line(format!("{value} = load {elem_llt}, ptr {b_elem_ptr}"));
        self.line(format!("store {elem_llt} {value}, ptr {packed_ptr}"));
        self.line(format!("br label %{store_done}"));

        self.label_line(&pad);
        self.line(format!(
            "store {elem_llt} zeroinitializer, ptr {packed_ptr}"
        ));
        self.line(format!("br label %{store_done}"));
        self.label_line(&store_done);
        let lane_next = self.tmp();
        self.line(format!("{lane_next} = add i64 {lane}, 1"));
        self.line(format!("store i64 {lane_next}, ptr {lane_ctr}"));
        self.line(format!("br label %{lane_head}"));

        self.label_line(&lane_done);
        let k_next = self.tmp();
        self.line(format!("{k_next} = add i64 {k}, 1"));
        self.line(format!("store i64 {k_next}, ptr {k_ctr}"));
        self.line(format!("br label %{k_head}"));
        self.label_line(&k_done);
        let jt_next = self.tmp();
        self.line(format!("{jt_next} = add i64 {jt}, 1"));
        self.line(format!("store i64 {jt_next}, ptr {jt_ctr}"));
        self.line(format!("br label %{jt_head}"));
        self.label_line(&jt_done);
    }

    /// The local slot type for a Pair-built staging product. Array components
    /// consumed only as addresses by collection/index/call ops are `ptr`
    /// fields (S20 #6/#8); value-observable components retain their ABI type.
    pub(in crate::func) fn lower_slot_ty(&self, o: ObjectId, ty: &Ty) -> Option<String> {
        if let Some(paths) = self.views.get(o) {
            return lower_ptr_paths(ty, paths);
        }
        let components: Vec<&Ty> = match ty {
            Ty::Tuple(ts) => ts.iter().collect(),
            Ty::Struct { fields, .. } => fields.iter().map(|(_, ty)| ty).collect(),
            _ => return lower_ty(ty),
        };
        let kept: Vec<String> = components
            .iter()
            .enumerate()
            .filter_map(|(k, ty)| {
                if self.pointer_only_array_component(o, k as u32) {
                    Some("ptr".into())
                } else {
                    lower_ty(ty)
                }
            })
            .collect();
        match kept.len() {
            0 => None,
            1 => Some(kept.into_iter().next().unwrap()),
            _ => Some(format!("{{ {} }}", kept.join(", "))),
        }
    }

    /// A Pair array field is representation-only when every use reads that
    /// component as an address. Such fields stage a pointer, never array bytes.
    pub(in crate::func) fn pointer_only_array_component(&self, agg: ObjectId, k: u32) -> bool {
        if !matches!(self.obj_ty(agg).component_ty(k), Some(Ty::Array { .. }))
            || self.pair_source(agg, k).is_none()
        {
            return false;
        }
        let uses = self.ir.out_edges(agg);
        !uses.is_empty()
            && uses.iter().all(
                |&m| match self.ir.morphism(m).expect("morphism resolves").op {
                    Operation::Index | Operation::Update => k == 0,
                    Operation::Zip => k <= 1,
                    Operation::Map { captures, .. } => k <= captures,
                    Operation::Fold { captures, .. } => k < captures || k == captures + 1,
                    Operation::Call(_) => true,
                    _ => false,
                },
            )
    }

    /// Address field `k` of raw aggregate storage. Erasure remapping is based
    /// on the source type; replacing an Array with `ptr` keeps it materialized.
    pub(in crate::func) fn field_ptr(
        &mut self,
        base: &str,
        agg_ty: &Ty,
        agg_llt: &str,
        k: u32,
    ) -> Option<String> {
        if residual_arity(agg_ty) == 1 {
            return Some(base.to_string());
        }
        let eidx = erased_index(agg_ty, k)?;
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr {agg_llt}, ptr {base}, i32 0, i32 {eidx}"
        ));
        Some(p)
    }

    /// Copy an aggregate value between allocas without creating aggregate SSA
    /// (ADR-0021's Update pattern). An exact identity is already in place.
    pub(in crate::func) fn emit_memcpy(&mut self, dst: &str, src: &str, llt: &str) {
        if dst == src {
            return;
        }
        self.line(format!(
            "call void @llvm.memcpy.p0.p0.i64(ptr {dst}, ptr {src}, i64 ptrtoint (ptr getelementptr ({llt}, ptr null, i64 1) to i64), i1 false)"
        ));
    }

    /// Store `(vllt, val)` into field `k` of a raw aggregate pointer of ty
    /// `agg_ty`, whose lowered text is `agg_llt` (the by-ref input struct for a
    /// capturing body call, `lower_ty` otherwise — the GEP offsets differ).
    /// Used by the collection loops, which build products in scratch.
    pub(in crate::func) fn field_store(
        &mut self,
        base: &str,
        agg_ty: &Ty,
        agg_llt: &str,
        k: u32,
        vllt: &str,
        val: &str,
    ) {
        let ptr = self
            .field_ptr(base, agg_ty, agg_llt, k)
            .expect("kept field");
        self.line(format!("store {vllt} {val}, ptr {ptr}"));
    }

    /// The base address of an op's array operand — component `k` of the `source`
    /// product (`Some(k)`), or the bare `source` object itself (`None`, the
    /// no-capture map source). When the array reaches the op from a ptr-resident
    /// by-ref capture (the `Pair` feeder, or `source` itself), the forwarded
    /// `load ptr` is the address — the op reads the caller's array directly
    /// instead of the inline deep copy. When `source` IS the by-ref fn input
    /// product, its Array field holds the forwarded `ptr` — load it. Anything
    /// else is the ordinary slot/component address, as before.
    pub(in crate::func) fn array_operand_ptr(
        &mut self,
        source: ObjectId,
        k: Option<u32>,
    ) -> Option<String> {
        let feeder = match k {
            None => Some(source),
            Some(k) => self.pair_source(source, k),
        };
        // S54 `arena`: a device-only array has no host address. Every host
        // use of one is a pointer-only field or an element copy-out, which
        // ask before coming here; anything else stops the emission.
        if feeder.is_some_and(|f| self.host_free(f)) {
            return None;
        }
        if let Some(f) = feeder
            && self.ptr_resident.contains_key(f)
        {
            let slot = self.slot(f)?;
            let p = self.tmp();
            self.line(format!("{p} = load ptr, ptr {slot}"));
            return Some(p);
        }
        if let Some(f) = feeder
            && matches!(self.obj_ty(f), Ty::Array { .. })
            && let Some(slot) = self.slot(f)
        {
            return Some(slot);
        }
        match k {
            None => self.slot(source),
            Some(k) => {
                let byref_field = matches!(&self.byref, Some((input, _, _))
                    if *input == source
                        && self.byref_paths.iter().any(|p| p.as_slice() == [k]));
                // S65 `by-ref`: a view field holds the pointer.
                if byref_field || self.is_view(source, &[k]) {
                    let (_, fp) = self.component_ptr(source, k)?;
                    let p = self.tmp();
                    self.line(format!("{p} = load ptr, ptr {fp}"));
                    return Some(p);
                }
                Some(self.component_ptr(source, k)?.1)
            }
        }
    }

    // --- traps ------------------------------------------------------------

    /// Branch to a trap block when `cond` is true; continue otherwise
    /// (`kind`: 0 = div_zero, 1 = index_oob — DESIGN §1).
    pub(in crate::func) fn trap_if(&mut self, cond: &str, kind: u32) {
        let trap = self.label();
        let cont = self.label();
        self.line(format!("br i1 {cond}, label %{trap}, label %{cont}"));
        self.label_line(&trap);
        self.line(format!("call void @mapal_trap(i32 {kind})"));
        self.line("unreachable");
        self.label_line(&cont);
    }

    pub(in crate::func) fn task_site(&self, m: MorphismId) -> u32 {
        match self.guard_flavor {
            GuardFlavor::Host => unreachable!("host guard has no task site"),
            GuardFlavor::TaskBody(topo) => topo,
            GuardFlavor::Task => self
                .ir
                .topo_order(self.f)
                .iter()
                .position(|&candidate| candidate == m)
                .expect("task morphism is in entry topo") as u32,
        }
    }

    pub(in crate::func) fn record_trap(&mut self, m: MorphismId, kind: u32) {
        let topo = self.task_site(m);
        // bf16-class `device-trap`: a device body records into the module's
        // trap word (`gpu::trap_variant`).
        if self.device_trap {
            self.line(format!(
                "call void @dev_trap_record(i64 {topo}, i32 {kind})"
            ));
            return;
        }
        self.line(format!("call void @mapal_par_trap(i64 {topo}, i32 {kind})"));
        self.runtime_write = true;
    }

    pub(in crate::func) fn local_trap_site(&self, m: MorphismId) -> bool {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        match morph.op {
            Operation::Div | Operation::Mod => {
                matches!(self.obj_ty(morph.target), Ty::Int { .. })
            }
            Operation::Index => !self.bp.proven(m),
            Operation::Update => true,
            _ => false,
        }
    }

    pub(in crate::func) fn emit_watermark(&mut self, m: MorphismId) {
        let topo = self.task_site(m);
        self.line(format!("call void @mapal_par_watermark(i64 {topo})"));
        self.runtime_write = true;
    }

    /// The accumulator's flat offset for one (subrow, lane): `acc_base + r*stride + lane`.
    ///
    /// One helper for what were four copies of the same four-armed decision —
    /// the inline `if r == 0` at the seed and store loops (`acc_base` absent),
    /// the four-arm match inside `emit_tile_lane_loop`, and the KC nest's own
    /// `emit_tile_kc_acc_lane` (`acc_base` present). They were character-identical
    /// per arm, which is why this extraction is a provable no-op on emitted text:
    /// each arm mints exactly the temporaries it minted before, in the same order.
    /// That matters more than usual here — `fresh()` is a SINGLE ordinal counter
    /// feeding `tmp()`, `label()` and `scratch()` alike, so one extra or reordered
    /// mint renames every subsequent value in the function and rewrites the entry
    /// block as well.
    ///
    /// `None` is the direct nest, whose accumulator starts at 0 and whose `r == 0`
    /// case needs no arithmetic at all; `Some(base)` is the KC nest, addressing
    /// into a panel-offset accumulator.
    pub(in crate::func) fn emit_acc_lane(
        &mut self,
        lane: &str,
        acc_base: Option<&str>,
        r: u64,
        stride: u64,
    ) -> String {
        match (acc_base, r) {
            (None, 0) => lane.to_owned(),
            (None, _) => {
                let offset = self.tmp();
                self.line(format!("{offset} = add i64 {lane}, {}", r * stride));
                offset
            }
            (Some(base), 0) => {
                let offset = self.tmp();
                self.line(format!("{offset} = add i64 {lane}, {base}"));
                offset
            }
            (Some(base), _) => {
                let based = self.tmp();
                self.line(format!("{based} = add i64 {base}, {}", r * stride));
                let offset = self.tmp();
                self.line(format!("{offset} = add i64 {lane}, {based}"));
                offset
            }
        }
    }

    /// This row's live column window, clipped to the task's flat range:
    /// `[max(0, lo - row0), min(C, hi - row0))`.
    ///
    /// A task owns a flat `[lo, hi)` over `rows*C` elements, so the first and
    /// last rows it touches are partial. Both comparisons are **signed** on
    /// purpose — `lo - row0` is negative for every row after the first, and an
    /// unsigned compare would read that as enormous and clip to `0` wrongly.
    ///
    /// Was written out three times, character-identical, in the boundary row,
    /// the row-split and the KC boundary row. Extracted rather than left
    /// duplicated because it always leads its caller and always mints the same
    /// six temporaries in the same order, so hoisting it cannot move the shared
    /// ordinal counter that names every subsequent value.
    pub(in crate::func) fn emit_row_window(
        &mut self,
        site: &TileSite,
        lo: &str,
        hi: &str,
        row0: &str,
    ) -> (String, String) {
        let jw_lo_raw = self.tmp();
        self.line(format!("{jw_lo_raw} = sub i64 {lo}, {row0}"));
        let jw_lo_negative = self.tmp();
        self.line(format!("{jw_lo_negative} = icmp slt i64 {jw_lo_raw}, 0"));
        let jw_lo = self.tmp();
        self.line(format!(
            "{jw_lo} = select i1 {jw_lo_negative}, i64 0, i64 {jw_lo_raw}"
        ));
        let jw_hi_raw = self.tmp();
        self.line(format!("{jw_hi_raw} = sub i64 {hi}, {row0}"));
        let jw_hi_past_c = self.tmp();
        self.line(format!(
            "{jw_hi_past_c} = icmp sgt i64 {jw_hi_raw}, {}",
            site.c
        ));
        let jw_hi = self.tmp();
        self.line(format!(
            "{jw_hi} = select i1 {jw_hi_past_c}, i64 {}, i64 {jw_hi_raw}",
            site.c
        ));
        (jw_lo, jw_hi)
    }

    pub(in crate::func) fn bulk_bounds(&self, n: u64) -> (String, String) {
        if let Some((lo, hi)) = &self.row_bounds {
            return (lo.clone(), hi.clone());
        }
        if self.split_range {
            ("%lo".into(), "%hi".into())
        } else {
            ("0".into(), n.to_string())
        }
    }

    // --- the walk ---------------------------------------------------------
}
