//! the function drivers: `emit`, `emit_parallel`, `emit_task`, the `topo_order` walks
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

mod parallel;
mod task;

/// `(kind, width)` a split of serial `ns` registers under the profile's
/// per-lane wave ([`crate::profile::DispatchCost::width`]): one lane is
/// `kind = 0` (inline), the whole pool `width = 0`; without the fact `(1, 0)`,
/// the registration every build made before it.
fn registration(profile: &TargetProfile, ns: u64) -> (u64, u64) {
    match profile.dispatch.and_then(|c| c.width(ns)) {
        Some(1) => (0, 0),
        Some(w) => (1, w),
        None => (1, 0),
    }
}

impl<'a> FnEmit<'a> {
    /// Whether a split task's estimated SERIAL time is under what one dispatch
    /// costs — in which case the pool is a net loss and it should run inline
    /// (plan-s50 P3).
    ///
    /// A roofline, not an operation count: `max(flops / flops_per_ns,
    /// bytes / bytes_per_ns)`. The compute term alone gets `conv2d` wrong by
    /// 3.7x in the wrong direction, because half the flops of an `nn4_256` layer
    /// take nearly four times as long when the kernel is memory-bound.
    ///
    /// `None` dispatch economics — every profile except the one part these were
    /// measured on — means this always answers false, i.e. exactly the
    /// behaviour before the rule existed.
    pub(super) fn under_dispatch_cost(&self, task: &mapal_ir::Task) -> bool {
        let Some(cost) = self.profile.dispatch else {
            return false;
        };
        // local-rtx-5080: with the wave priced per lane, "inline" is the width
        // rule's one-lane answer, for every site the facts can estimate.
        if cost.lanes >= 2 {
            return self.task_width(task) == Some(1);
        }
        let TaskKind::Split { site, n } = &task.kind else {
            return false;
        };
        let n = *n;
        // ONLY A RECOGNISED TILE SITE IS ESTIMATED, and the reason is that a
        // wrong verdict is not symmetric: dispatching something small costs one
        // wave, while inlining something large costs the whole parallel
        // speedup. The first build estimated elementwise maps too, as
        // `3 * n * width` bytes at the streaming rate, and got two of fourteen
        // shapes wrong in exactly that expensive direction — `transpose_512`
        // (estimated 25 us, real serial 142 us, parallel is 1.85x) because a
        // permutation does not run at streaming rate, and `fir_65536`
        // (estimated 6 us, real 91 us, parallel is 1.38x) because a 64-tap
        // window reads far more than three values per output.
        //
        // A tile record carries `rows`, `c`, `k` and `elem`, which is exactly
        // what the roofline needs. Without one there is no basis, so there is no
        // verdict — the task is dispatched, as it was before this rule.
        let Some(record) = self.tile_plan.as_ref().and_then(|p| p.sites.get(*site)) else {
            return false;
        };
        // `k` multiply-adds per output element; each operand panel and the
        // output walked once (the roofline S62 `place` reads too).
        let flops = 2u64.saturating_mul(n).saturating_mul(record.k);
        let bytes = crate::profile::tile_bytes(record.rows, record.c, record.k, &record.elem);
        let ns = crate::profile::roofline_ns(flops, bytes, cost.flops_per_ns, cost.bytes_per_ns);
        ns < cost.dispatch_ns
    }

    /// A split task's serial nanoseconds at one lane's rates, from the facts
    /// that carry its work: a tile site's record (as [`Self::under_dispatch_cost`])
    /// or a cell site's (`cell_plan`: `work` element steps per cell, each a
    /// multiply-add; every read and the write one element each). `None` for a
    /// task neither recognizer describes.
    ///
    /// A LOWER bound: a strided read (a transpose) moves a line per element
    /// and a window re-reads; neither is priced. That is the direction the
    /// width rule tolerates ([`crate::profile::DispatchCost::lanes`]).
    fn serial_ns(&self, task: &mapal_ir::Task) -> Option<u64> {
        let cost = self.profile.dispatch?;
        let TaskKind::Split { site, n } = &task.kind else {
            return None;
        };
        let (flops, bytes) =
            if let Some(record) = self.tile_plan.as_ref().and_then(|p| p.sites.get(*site)) {
                (
                    2u64.saturating_mul(*n).saturating_mul(record.k),
                    crate::profile::tile_bytes(record.rows, record.c, record.k, &record.elem),
                )
            } else {
                let cells = self.ir.cell_plan(self.f);
                let cell = cells.get(*site)?;
                let m = self.ir.morphism(*site)?;
                let elem = match self.ir.object(m.target).map(|o| &o.ty) {
                    Some(mapal_ir::Ty::Array { elem, .. }) => crate::profile::elem_bytes(elem),
                    _ => 4,
                };
                (
                    2u64.saturating_mul(*n).saturating_mul(cell.work.max(1)),
                    n.saturating_mul(cell.reads.len() as u64 + 1)
                        .saturating_mul(elem),
                )
            };
        Some(crate::profile::roofline_ns(
            flops,
            bytes,
            cost.flops_per_ns,
            cost.bytes_per_ns,
        ))
    }

    /// The `width` a split task registers (mapal-rt `TaskDef::width`): the
    /// lanes [`crate::profile::DispatchCost::width`] gives its serial estimate,
    /// `None` without a per-lane fact or an estimate (registered `0`, the whole
    /// pool, as before).
    pub(super) fn task_width(&self, task: &mapal_ir::Task) -> Option<u64> {
        self.profile.dispatch?.width(self.serial_ns(task)?)
    }

    /// Emit the function body: prologue store, the topo walk, epilogue return.
    pub fn emit(self) -> String {
        self.emit_ordered().0
    }

    /// [`Self::emit`] and the split folds it compiled under
    /// [`crate::Order::Fixed`] (ADR-0040 D3).
    pub fn emit_ordered(mut self) -> (String, Vec<crate::FoldOrder>) {
        let fd = self.ir.func(self.f).expect("func resolves");
        let ret_ty = self.obj_ty(fd.output);
        let fname = if self.cell_slots.is_empty() {
            self.fnames[self.f].clone()
        } else {
            format!("{}{}", self.fnames[self.f], self.cell_suffix)
        };

        // Only the entry function's frame is once-per-program (plan-s29
        // composition rule 4); the parallel entry goes through `emit_parallel`,
        // so reaching here as the entry IS the sequential flavor.
        self.heap_ok = self.f == self.ir.entry();
        // structure: a non-entry Named fn (a decode step called per token)
        // takes its big blocks from the per-call arena instead of its stack.
        let named = self.ir.func(self.f).expect("func resolves").kind == FuncKind::Named;
        if !self.heap_ok && named && self.cell_slots.is_empty() && self.lanes.is_none() {
            self.heap_ok = true;
            self.frame_heap = true;
        }

        let in_llt = self.prepare_storage();
        self.allocate_local_slots(&in_llt);

        // Prologue: store the incoming parameter into its slot.
        if let Some(t) = &in_llt
            && let Some(ps) = self.slot(fd.input)
        {
            self.line(format!("store {t} %arg, ptr {ps}"));
        }

        // The topo walk (DESIGN §2/§3).
        self.walk();

        // Epilogue: return the output slot (or void). structure: an output of
        // arrays was written through `%out` already.
        let sig_ret = match lower_ty(&ret_ty) {
            Some(_) if self.out_ptr => {
                if self.perf_timing {
                    self.line("call void @mapal_perf_end()");
                }
                self.resident_teardown();
                self.heap_teardown();
                self.line("ret void");
                "void".to_string()
            }
            Some(t) => {
                let os = self.slot(fd.output).expect("non-void return has a slot");
                let v = self.tmp();
                self.line(format!("{v} = load {t}, ptr {os}"));
                if self.perf_timing {
                    self.line("call void @mapal_perf_end()");
                }
                self.resident_teardown();
                self.heap_teardown();
                self.line(format!("ret {t} {v}"));
                t
            }
            None => {
                if self.perf_timing {
                    self.line("call void @mapal_perf_end()");
                }
                self.resident_teardown();
                self.heap_teardown();
                self.line("ret void");
                "void".to_string()
            }
        };

        // Truthful fn attributes (suggestions #7; `FnAttrs`): clean fns are
        // `readonly nounwind` (+ `willreturn` when the closure can't loop or
        // recurse); a clean fn's bare-`ptr` by-ref array parameter additionally
        // carries `noalias nocapture readonly` (the callee never writes through
        // it, never lets it escape, and — the single pointer argument — it
        // aliases nothing else the fn accesses). Unclean fns stay bare.
        let clean = self.attrs.clean(self.f) && !self.runtime_write && !self.perf_timing;
        let param = match &in_llt {
            Some(t) if clean && t == "ptr" && self.byref.is_some() && !self.out_ptr => {
                format!("{t} noalias nocapture readonly %arg")
            }
            Some(t) => format!("{t} %arg"),
            None => String::new(),
        };
        // structure: the caller's storage for an output of arrays.
        let (param, clean) = match (self.out_ptr, param.is_empty()) {
            (false, _) => (param, clean),
            (true, true) => ("ptr noalias %out".to_string(), false),
            (true, false) => (format!("{param}, ptr noalias %out"), false),
        };
        // `cell-lanes`: the lane variant meets its warp in shuffles — a
        // convergent call no optimizer may sink under the lane-0 store.
        let fn_attrs = if self.lanes.is_some() {
            " convergent nounwind"
        } else if clean {
            if self.attrs.loopy(self.f) {
                " readonly nounwind"
            } else {
                " readonly nounwind willreturn"
            }
        } else {
            ""
        };
        let perf_begin = if self.perf_timing {
            "  call void @mapal_perf_begin()\n"
        } else {
            ""
        };
        // `cell-law`'s device variant: its own name, the values after `%arg`.
        let fname = if self.cell_vals.is_empty() {
            fname
        } else {
            gpu::cell_variant(&fname)
        };
        let fname = match (self.lanes.is_some(), self.lane_trips) {
            (false, _) => fname,
            (true, crate::profile::NAIVE_UNROLL) => gpu::lanes_variant(&fname),
            // residency `fly-unroll`: its own depth, its own name.
            (true, t) => format!("{}_u{t}", gpu::lanes_variant(&fname)),
        };
        let fname = if self.device_trap {
            gpu::trap_variant(&fname)
        } else {
            fname
        };
        let param = self
            .cell_vals
            .iter()
            .fold(param, |p, (s, t)| format!("{p}, {t} %cv{s}"));
        // `tile-post`: the epilogue variant, its fold's value after `%arg`.
        let (fname, param) = match self.post_fold {
            Some(fold) => {
                let t = self.ir.morphism(fold).expect("fold resolves").target;
                let llt = lower_ty(&self.obj_ty(t)).expect("a fold value lowers");
                (gpu::post_variant(&fname), format!("{param}, {llt} %fv"))
            }
            None => (fname, param),
        };
        self.zero_slots_rest();
        if self.frame_heap && self.heap_used {
            self.allocas
                .insert_str(0, "  %hmark = call i64 @mapal_rt_frame_mark()\n");
        }
        let text = format!(
            "define internal {sig_ret} @{fname}({param}){fn_attrs} {{\nentry:\n{}{}{}}}\n{}{}",
            perf_begin, self.allocas, self.body, self.cand_tables, self.host_par_fns
        );
        (text, self.orders)
    }
}
