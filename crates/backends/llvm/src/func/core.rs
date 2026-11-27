//! per-object slots, name minting, loads/stores, heap + packed buffers, trap sites
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

mod packs;
mod ptrs;

impl<'a> FnEmit<'a> {
    pub fn new(
        ir: &'a CategoryIr,
        f: FuncId,
        fnames: &'a SecondaryMap<FuncId, String>,
        strings: &'a SecondaryMap<ObjectId, StrGlobal>,
        attrs: &'a FnAttrs,
        tiling: bool,
        packing: bool,
        contract: bool,
        order: crate::Order,
        kc_nest: bool,
        move_panel: MovePanel,
        profile: &'static TargetProfile,
    ) -> Self {
        let mut gsites = SecondaryMap::new();
        let mut gated = SecondaryMap::new();
        for site in ir.guard_plan(f).into_iter().filter(GuardSite::gated) {
            for &m in site.on_true.own.iter().chain(site.on_false.own.iter()) {
                gated.insert(m, ());
            }
            gsites.insert(site.phi, site);
        }
        FnEmit {
            ir,
            f,
            fnames,
            strings,
            attrs,
            slots: SecondaryMap::new(),
            allocas: String::new(),
            body: String::new(),
            next: 0,
            byref: None,
            byref_paths: Vec::new(),
            out_ptr: false,
            ptr_resident: SecondaryMap::new(),
            lup: ir.last_use_plan(f),
            bp: ir.bounds_proof(f),
            gsites,
            gated,
            elided_updates: SecondaryMap::new(),
            update_aliases: SecondaryMap::new(),
            frame: None,
            frame_geps: String::new(),
            guard_flavor: GuardFlavor::Host,
            device_trap: false,
            split_range: false,
            watermark: false,
            host: None,
            runtime_write: false,
            perf_timing: false,
            tile_plan: tiling.then(|| ir.tile_plan(f)),
            device: if tiling {
                gpu::placement(ir, f, &ir.tile_plan(f), profile, true, true)
            } else {
                SecondaryMap::new()
            },
            // Only `Deduce` consults it, so `Off` and `Force` do not pay for
            // the scan — and `Off` is provably the same work it was before S45.
            move_sites: matches!(move_panel, MovePanel::Deduce)
                .then(|| ir.move_plan(f))
                .unwrap_or_default(),
            elem: ir.elem_plan(f),
            elided_arrays: SecondaryMap::new(),
            packing,
            contract,
            order,
            orders: Vec::new(),
            kc_nest,
            move_panel,
            profile,
            heap_ok: false,
            heap_used: false,
            frame_heap: false,
            gpu_arms: crate::GpuArms::default(),
            algebra: None,
            fma_stash: SecondaryMap::new(),
            block_priced: std::cell::Cell::new(None),
            stream_why: std::cell::Cell::new(None),
            stream_priced: std::cell::Cell::new(None),
            block: 0,
            resident: None,
            dslots: SecondaryMap::new(),
            dvalid: SecondaryMap::new(),
            retain: SecondaryMap::new(),
            retain_flags: SecondaryMap::new(),
            row_out: None,
            row_done: false,
            row_bounds: None,
            groups: SecondaryMap::new(),
            cell_slots: Vec::new(),
            cell_suffix: String::new(),
            cell_vals: Vec::new(),
            lanes: None,
            lane_trips: crate::profile::NAIVE_UNROLL,
            post_fold: None,
            path_plan: None,
            split_slots: Vec::new(),
            dallocs: Vec::new(),
            taken: Vec::new(),
            site_table: None,
            cand_tables: String::new(),
            zero_slots: Vec::new(),
            hz: horizontal::HzEmit::default(),
            views: SecondaryMap::new(),
            views_done: false,
            outside_memo: Default::default(),
            by_ref: false,
            bufs: Vec::new(),
            loops: Vec::new(),
            root_index: Default::default(),
            par: None,
            host_par_fns: String::new(),
            host_par_n: 0,
        }
    }

    /// The fast-math flags a tiled float step carries (ADR-0040 D1, D4):
    /// `contract` lets a step fuse (one rounding), `reassoc` lets the fold's
    /// order move (the `free` order). Never `afn`, `arcp`, `ninf`, `nnan` or
    /// `nsz`: order may move, a step's rounding may not. Host sites are never
    /// split, so `fixed` is `sequential` here.
    pub(super) fn fp_flags(&self, elem: &Ty) -> &'static str {
        match (is_float(elem), self.contract, self.order) {
            (false, _, _) => "",
            (true, false, crate::Order::Free) => " reassoc",
            (true, false, _) => "",
            (true, true, crate::Order::Free) => " reassoc contract",
            (true, true, _) => " contract",
        }
    }

    /// fold-algebra: the fast-math flags [`mapal_ir::StepAlgebra::flags`]
    /// gives morphism `m` of this function on this face — the one rule the
    /// host and the device cells share (` contract`, ` reassoc`, both, none).
    pub(super) fn step_flags(&mut self, m: MorphismId) -> &'static str {
        let (contract, free) = (self.contract, self.order == crate::Order::Free);
        if !contract && !free {
            return "";
        }
        let (pairs, regroup) = (self.gpu_arms.pairs, self.gpu_arms.regroup);
        let fl = self.algebra().flags(m, contract, free, pairs, regroup);
        match (fl.reassoc, fl.contract) {
            (true, true) => " reassoc contract",
            (true, false) => " reassoc",
            (false, true) => " contract",
            (false, false) => "",
        }
    }

    /// This function's stored step algebra, read once per emitter.
    pub(super) fn algebra(&mut self) -> &mapal_ir::StepAlgebra {
        self.algebra
            .get_or_insert_with(|| self.ir.step_algebra(self.f))
    }

    /// fold-algebra: where the face contracts and the order is not free
    /// (sequential, fixed: a bit-for-bit oracle compares the host with the
    /// device), a pair is one explicit `llvm.fma` — the device tile's
    /// `OrderedFma` — not a `contract` flag a target without FMA would not
    /// fuse. On the free order the flags suffice (any fusion is in the bound).
    pub(super) fn explicit_fma(&self) -> bool {
        self.contract && self.gpu_arms.pairs && self.order != crate::Order::Free
    }

    /// The site writing `target` runs on the device (S62 `place`).
    pub(super) fn on_device(&self, target: ObjectId) -> bool {
        self.ir
            .in_edges(target)
            .first()
            .is_some_and(|&m| self.device.contains_key(m))
    }

    /// S54 `streams`: device sites read independence off the path plan; the
    /// host itself stays sequential on a GPU profile.
    pub(crate) fn set_path_plan(&mut self, plan: Option<&'a PathPlan>) {
        self.path_plan = plan;
    }

    pub(crate) fn set_gpu_arms(&mut self, arms: crate::GpuArms) {
        self.gpu_arms = arms;
        // S65 `by-ref`: the products' layout, which the residency plan reads.
        self.plan_views();
        let rungs = super::gpu::host_rungs(self.profile, self.contract, arms);
        if self.tile_plan.is_some() {
            let plan = super::gpu::tile_plan(self.ir, self.f, self.profile, self.contract, arms);
            self.device = super::gpu::placement(
                self.ir,
                self.f,
                &plan,
                self.profile,
                rungs.window,
                rungs.carry_made,
            );
            self.tile_plan = Some(plan);
            self.retain = retain::retained_sites(self.ir, self.f, self.profile, rungs);
        }
        // A function with a retained site stays resident under
        // `retain-resident` (the retained output is its resident buffer);
        // without it, it keeps the S52 round trips `retain-delta` measured on.
        let resident = rungs.resident
            && self.profile.gpu().is_some()
            && (self.retain.is_empty() || rungs.retain_resident);
        // S54 `epilogue`: the per-cell groups, before the plan that reads them.
        if resident {
            self.groups = self.epilogues();
        }
        // `tile-post`: a site's value is its epilogue's, so a post site with
        // no group (`gpu::post_sites` admits only where one forms) is no site:
        // it stays the per-cell map it was.
        let orphans: Vec<MorphismId> = self
            .tile_plan
            .as_ref()
            .map(|p| {
                p.sites
                    .iter()
                    .filter(|&(m, s)| s.post.is_some() && !self.groups.contains_key(m))
                    .map(|(m, _)| m)
                    .collect()
            })
            .unwrap_or_default();
        if !orphans.is_empty() {
            let plan = self.tile_plan.as_mut().expect("a plan");
            for m in orphans {
                plan.sites.remove(m);
            }
            self.device = super::gpu::placement(
                self.ir,
                self.f,
                plan,
                self.profile,
                rungs.window,
                rungs.carry_made,
            );
            if resident {
                self.groups = self.epilogues();
            }
        }
        self.resident = resident.then(|| self.resident_plan(rungs)).flatten();
        // generic-fusion: the host intent line, in the entry block.
        if let Some(p) = self.resident.as_ref().filter(|p| !p.flow_line.is_empty()) {
            self.allocas.push_str(&format!("  {}", p.flow_line));
        }
        if let Some(p) = self.resident.as_ref().filter(|p| !p.zero_line.is_empty()) {
            self.allocas.push_str(&format!("  {}", p.zero_line));
        }
        if let Some(p) = self.resident.as_ref().filter(|p| !p.hz_line.is_empty()) {
            self.allocas.push_str(&format!("  {}", p.hz_line));
        }
        if let Some(p) = self.resident.as_ref().filter(|p| !p.calls_line.is_empty()) {
            self.allocas.push_str(&format!("  {}", p.calls_line));
        }
        // A carried state's next lives in the buffer the back edge swaps with
        // the merge's: last iteration's rows are not in it, so it is not retained.
        if let Some(p) = &self.resident {
            self.retain
                .retain(|y, _| !p.is_carry_next(y) && p.alias(y) == y);
        }
        if self.arena_slot() {
            self.allocas
                .push_str("  %darena = alloca i64\n  store i64 0, ptr %darena\n");
        }
    }

    /// The producers this function computes on the device (`devgen`): kernel
    /// id, the body it calls, and the element type it stores.
    pub(crate) fn gen_kernels(&self) -> Vec<(u64, FuncId, String)> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.devgen
            .iter()
            .map(|(m, g)| {
                let t = self.ir.morphism(m).expect("gen morphism").target;
                let (llt, _) = array_parts(&self.obj_ty(t));
                (g.id, g.body, llt)
            })
            .collect()
    }

    pub(crate) fn set_perf_timing(&mut self, perf_timing: bool) {
        self.perf_timing = perf_timing;
    }

    pub(crate) fn set_task_body_site(&mut self, topo: u32) {
        self.guard_flavor = GuardFlavor::TaskBody(topo);
    }

    pub(super) fn fresh(&mut self) -> u32 {
        let n = self.next;
        self.next += 1;
        n
    }

    pub(super) fn tmp(&mut self) -> String {
        format!("%t{}", self.fresh())
    }

    /// A fresh block label (bare, no `%`).
    pub(crate) fn label(&mut self) -> String {
        format!("bb{}", self.fresh())
    }

    /// Append one indented body instruction line.
    pub(crate) fn line(&mut self, s: impl AsRef<str>) {
        // kernel-merging `horizontal`: a group member's launch is deferred.
        if self.hz.cur.is_some() && self.hz_take(s.as_ref()) {
            return;
        }
        self.body.push_str("  ");
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }

    /// Append a bare block label line (`name:`), no indent.
    pub(crate) fn label_line(&mut self, name: &str) {
        self.block += 1;
        self.body.push_str(name);
        self.body.push_str(":\n");
    }

    pub(super) fn slot(&mut self, o: ObjectId) -> Option<String> {
        if let Some(slot) = self.slots.get(o) {
            return Some(slot.clone());
        }
        let field = self.frame.as_ref()?.fields.get(o)?.clone();
        if let Some(slot) = self.slots.get(field.owner) {
            let slot = slot.clone();
            self.slots.insert(o, slot.clone());
            return Some(slot);
        }
        let slot = format!("%o{}", field.ordinal);
        self.frame_geps.push_str(&format!(
            "  {slot} = getelementptr %Frame, ptr %frame, i32 0, i32 {}\n",
            field.index
        ));
        self.slots.insert(field.owner, slot.clone());
        self.slots.insert(o, slot.clone());
        Some(slot)
    }

    pub(super) fn obj_ty(&self, o: ObjectId) -> Ty {
        self.ir.object(o).expect("object resolves").ty.clone()
    }

    // --- operand materialization -----------------------------------------

    /// Load the whole value of object `o`: a literal for a (scalar) constant, a
    /// `load` from its slot otherwise. `None` if `o` is erased / a `Str`.
    pub(super) fn load_whole(&mut self, o: ObjectId) -> Option<(String, String)> {
        let obj = self.ir.object(o).expect("object resolves");
        if obj.kind == ObjectKind::Constant {
            return match &obj.value {
                Some(Value::Str(_)) | None => None,
                Some(v) => Some((lower_ty(&obj.ty)?, const_literal(v))),
            };
        }
        let llt = match self.views.get(o) {
            Some(paths) => lower_ptr_paths(&obj.ty, paths)?,
            None => lower_ty(&obj.ty)?,
        };
        let slot = self.slot(o)?;
        if self.ptr_resident.contains_key(o) {
            // By-ref capture: load the array through the forwarded pointer.
            // The deep copy is observably the inline value (read-only
            // capture semantics), so escaping uses are unchanged.
            let p = self.tmp();
            self.line(format!("{p} = load ptr, ptr {slot}"));
            let v = self.tmp();
            self.line(format!("{v} = load {llt}, ptr {p}"));
            return Some((llt, v));
        }
        // The by-ref input product's WHOLE value (an escaping use — `Pair`
        // into an ordinary product, `Output`): its first-`bk` Array fields
        // hold `ptr`s, so assemble the by-value whole in a scratch of the
        // ordinary type — each by-ref array component deep-copied through its
        // forwarded pointer, every other component copied inline.
        let byref = matches!(&self.byref, Some((input, _, text)) if *input == o && *text != llt);
        if byref {
            // structure: every array the ABI passes by address, at any depth,
            // deep-copied through its pointer (`copy_layout`).
            let buf = self.scratch(&llt);
            let ty = obj.ty.clone();
            let paths = self.byref_paths.clone();
            self.copy_layout(&ty, &buf, &[], &slot, &paths);
            let v = self.tmp();
            self.line(format!("{v} = load {llt}, ptr {buf}"));
            return Some((llt, v));
        }
        let v = self.tmp();
        self.line(format!("{v} = load {llt}, ptr {slot}"));
        Some((llt, v))
    }

    /// A pointer to component `k` of the aggregate object `agg` (GEP or the bare
    /// slot). `None` if component `k` is erased or `agg` has no slot.
    pub(super) fn component_ptr(&mut self, agg: ObjectId, k: u32) -> Option<(String, String)> {
        let agg_ty = self.obj_ty(agg);
        match &agg_ty {
            Ty::Tuple(_) | Ty::Struct { .. } => {
                let comp_ty = agg_ty.component_ty(k)?;
                let agg_slot = self.slot(agg)?;
                // The by-ref body input GEPs against the by-ref struct text; a
                // first-k Array field holds the capture `ptr`, not the array.
                let (cllt, agg_llt) = match &self.byref {
                    Some((input, _, text)) if *input == agg => (
                        lower_ptr_paths(comp_ty, &sub_paths(&self.byref_paths, k))?,
                        text.clone(),
                    ),
                    _ => (
                        if let Some(paths) = self.views.get(agg) {
                            lower_ptr_paths(comp_ty, &sub_paths(paths, k))?
                        } else if self.pointer_only_array_component(agg, k) {
                            "ptr".into()
                        } else {
                            lower_ty(comp_ty)?
                        },
                        self.lower_slot_ty(agg, &agg_ty)?,
                    ),
                };
                if residual_arity(&agg_ty) == 1 {
                    Some((cllt, agg_slot)) // bare: the slot IS the component
                } else {
                    let eidx = erased_index(&agg_ty, k)?;
                    let ptr = self.tmp();
                    self.line(format!(
                        "{ptr} = getelementptr {agg_llt}, ptr {agg_slot}, i32 0, i32 {eidx}"
                    ));
                    Some((cllt, ptr))
                }
            }
            Ty::Array { elem, .. } => {
                let cllt = lower_ty(elem)?;
                let agg_slot = self.slot(agg)?;
                let agg_llt = lower_ty(&agg_ty)?;
                let ptr = self.tmp();
                self.line(format!(
                    "{ptr} = getelementptr {agg_llt}, ptr {agg_slot}, i64 0, i64 {k}"
                ));
                Some((cllt, ptr))
            }
            _ => None,
        }
    }

    /// structure: the pointer positions of `o`'s slot — `[[]]` for a bare
    /// forwarded array, the by-ref input's ABI paths, its views, or a staging
    /// product's pointer-only array fields (what [`Self::lower_slot_ty`] lays
    /// out).
    pub(super) fn slot_paths(&self, o: ObjectId) -> Vec<Vec<u32>> {
        if self.ptr_resident.contains_key(o) {
            return vec![Vec::new()];
        }
        if self.byref.as_ref().is_some_and(|(input, _, _)| *input == o) {
            return self.byref_paths.clone();
        }
        if let Some(v) = self.views.get(o) {
            return v.clone();
        }
        (0..product_arity(&self.obj_ty(o)))
            .filter(|&k| self.pointer_only_array_component(o, k))
            .map(|k| vec![k])
            .collect()
    }
}
