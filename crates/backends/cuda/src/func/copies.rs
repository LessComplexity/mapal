//! FnEmit loop copies, back-edge frees and the allocation registry.

use super::*;

impl<'a> FnEmit<'a> {
    // --- loop copies (the loops.rs quartet's moves; llvm's names kept) ----

    /// Copy the whole value of `from` into `to`'s local (loop init → merge;
    /// a value copy for scalars/products, a pointer copy for a carried
    /// array handle). No-op if erased.
    pub(crate) fn copy_obj(&mut self, from: ObjectId, to: ObjectId) {
        if let Some((_, val)) = self.load_whole(from) {
            self.store_obj(to, &val);
        }
    }

    /// Copy component `k` of aggregate `route` into object `to`'s local
    /// (the back edge: `back_route` slot 0 → merge — a pointer swap for a
    /// carried array, §2; and the exit copy: `exit_route` slot 0 → exit
    /// object). No-op if erased.
    pub(crate) fn copy_component(&mut self, route: ObjectId, k: u32, to: ObjectId) {
        if let Some((_, val)) = self.component_expr(route, k) {
            self.store_obj(to, &val);
        }
    }

    /// Component `k` of `route` as a bare lvalue expression (the loop guard
    /// bool — `exit_route` slot 1).
    pub(crate) fn route_component(&mut self, route: ObjectId, k: u32) -> String {
        self.component_expr(route, k).expect("route component").1
    }

    /// Back-edge freeing (suggestions.md #2; plan-last-use §3's cuda row):
    /// at the back edge, free the merge's outgoing array buffers whose
    /// next-instance producer is a registered allocation — but only when the
    /// last-use plan proves the carried state dead past the swap
    /// (`dead_after(merge, position(LoopBack))`: rule 1's ranking puts every
    /// body use of the merge at/before the back edge, the merge does not
    /// escape, and it is not carried) and the init's component is not
    /// borrowed (where the plan says borrowed/escape, no free is emitted —
    /// the conservative default is today's accumulate-to-fn-exit). Emits
    /// nothing per component that fails either clause: an in-placed update
    /// target (not registered — the swap is an identity there), the
    /// unchanged-carried merge, an arena member, or a call result.
    ///
    /// Each free rides the pointer-VALUE guard `if (merge.fE != init.fE)` —
    /// the same comparison class as the fn-exit escape guard (which stays as
    /// the second line of defense): on the first iteration the merge still
    /// holds the init's buffer (an arena member or a borrowed handle — never
    /// the producer's registered buffer), so the guard skips it; later
    /// iterations hold the previous iteration's producer buffer — dead past
    /// the swap — freed here instead of leaking to fn exit. The producer's
    /// registry entry is untouched: the exit iteration's final instance is
    /// freed at fn exit under the escape value guard, as today.
    pub(crate) fn emit_back_edge_frees(&mut self, plan: &LoopPlan) {
        let back_pos = self
            .ir
            .in_edges(plan.merge)
            .iter()
            .filter(|&&m| self.ir.morphism(m).expect("morphism resolves").op == Operation::LoopBack)
            .filter_map(|&m| self.last_use.position(m))
            .min();
        let Some(back_pos) = back_pos else { return };
        if !self.last_use.dead_after(plan.merge, back_pos) {
            return; // the plan can't prove the state dead past the swap
        }
        let Some(state) = self.pair_source(plan.back_route, 0) else {
            return;
        };
        let merge_ty = self.obj_ty(plan.merge);
        let bare = matches!(merge_ty, Ty::Array { .. });
        for k in 0.. {
            // The component's next-instance producer (packed into the
            // back-route state), the init's matching component (the borrowed
            // veto), and the two guard lvalues.
            let (comp_array, producer, init_comp, merge_lv, init_lv) = if bare {
                if k > 0 {
                    break;
                }
                (
                    true,
                    state,
                    plan.init,
                    self.slot(plan.merge),
                    self.slot(plan.init),
                )
            } else {
                let Some(comp_ty) = merge_ty.component_ty(k) else {
                    break;
                };
                let array = matches!(comp_ty, Ty::Array { .. });
                let producer = self.pair_source(state, k);
                let init_comp = self.pair_source(plan.init, k);
                let merge_lv = self.component_expr(plan.merge, k).map(|(_, lv)| lv);
                let init_lv = self.component_expr(plan.init, k).map(|(_, lv)| lv);
                match (producer, init_comp) {
                    (Some(p), Some(i)) => (array, p, i, merge_lv, init_lv),
                    _ => continue,
                }
            };
            if !comp_array {
                continue; // scalars carry no buffer
            }
            // The registered-allocation clause: the producer must be a
            // per-buffer cone-site malloc (the registry is the compile-time
            // roster of those).
            let Some(producer_slot) = self.slot(producer) else {
                continue;
            };
            if !self.allocs.contains(&producer_slot) {
                continue;
            }
            // The borrowed veto: where the init's component is not a fresh
            // fn-owned buffer (a Parameter, or ptr-resident Proj/Phi/Call
            // provenance — the plan's borrowed/escape clause), emit no free
            // (today's behavior, O(k·n) elsewhere).
            if !fresh_owned_buffer(self.ir, init_comp) {
                continue;
            }
            let (Some(merge_lv), Some(init_lv)) = (merge_lv, init_lv) else {
                continue;
            };
            self.line(format!("if ({merge_lv} != {init_lv}) {{"));
            self.indent += 1;
            self.line(format!(
                "cu_check(cudaFree({merge_lv}), \"cudaFree({merge_lv})\");"
            ));
            self.indent -= 1;
            self.line("}");
        }
    }

    // --- allocation registry (DESIGN §2; WP3 hooks) -----------------------

    /// Record a device buffer this fn allocated (WP3's cudaMalloc sites push
    /// here, in order).
    pub(crate) fn register_alloc(&mut self, name: String) {
        self.allocs.push(name);
    }

    /// Drop a buffer from the registry without freeing — the `Output` escape
    /// move (the caller inherits the free duty). No-op for unregistered
    /// names (all scalars).
    pub(crate) fn remove_alloc(&mut self, name: &str) {
        self.allocs.retain(|a| a != name);
    }

    /// `cudaFree` every buffer still owned by this fn, in allocation order
    /// (the allocator frees at fn exit; escapes were removed by `Output`),
    /// then release the fn's arena zone. Emits nothing when the registry is
    /// empty and the fn has no zone.
    ///
    /// When the fn **returns a value with array components**, each free is
    /// value-guarded against EVERY array-typed component of the return value
    /// (WP4 + F2): a loop-carried buffer escapes through the loop's exit
    /// copy into the Return, and a locally-allocated buffer escapes as a
    /// returned struct's field — neither passes through an `Output`, so the
    /// registry's name-based removal cannot see it. The guard compares by
    /// pointer equality against [`escape_lvalues`] — the return local itself
    /// for a bare array return, `ret.f{e}` (recursively) for products — so
    /// the Phi/alias escape shapes are subsumed too: whichever buffer the
    /// return value points at is the escaping one, freed by the caller;
    /// every other allocation is freed here (§2's no-double-free /
    /// no-use-after-free invariants, by pointer value). Pointer locals are
    /// `nullptr`-initialized, so the guard never reads an indeterminate
    /// handle (a cone-side site may not have executed).
    ///
    /// **The zone release (plan-smart-arenas rule 3).** The arena is freed
    /// once, under the range-test veto: `escaped0` is the disjunction of the
    /// escape lvalues' pointer-range tests (`(char*)e >= (char*)arena0 &&
    /// (char*)e < (char*)arena0 + CAP`) — the same pointer-VALUE comparison
    /// class as the per-buffer guard, at zone granularity: an escaping
    /// buffer pins its whole zone (the caller inherits the bounded-leak
    /// duty, DESIGN §2 amendment (ii)'s shape). Cone-site buffers are not
    /// zone members; their per-buffer frees above are unchanged.
    pub(super) fn emit_frees(&mut self) {
        let fd = self.ir.func(self.f).expect("func resolves");
        let out_ty = self
            .ir
            .object(fd.output)
            .expect("output resolves")
            .ty
            .clone();
        let escapes: Vec<String> = match self.slot(fd.output) {
            Some(ret) => escape_lvalues(&out_ty, &ret),
            None => Vec::new(),
        };
        for name in std::mem::take(&mut self.allocs) {
            if escapes.is_empty() {
                self.line(format!("cu_check(cudaFree({name}), \"cudaFree({name})\");"));
            } else {
                let cond = escapes
                    .iter()
                    .map(|e| format!("{name} != {e}"))
                    .collect::<Vec<_>>()
                    .join(" && ");
                self.line(format!("if ({cond}) {{"));
                self.line4(format!("cu_check(cudaFree({name}), \"cudaFree({name})\");"));
                self.line("}");
            }
        }
        let arena_cap = self.arena.as_ref().map(|p| p.capacity);
        if let Some(cap) = arena_cap {
            if escapes.is_empty() {
                self.line("cu_check(cudaFree(arena0), \"cudaFree(arena0)\");");
            } else {
                let tests = escapes
                    .iter()
                    .map(|e| {
                        format!(
                            "((char*){e} >= (char*)arena0 && (char*){e} < (char*)arena0 + {cap}ULL)"
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" || ");
                self.line(format!("bool escaped0 = {tests};"));
                self.line("if (!escaped0) {");
                self.line4("cu_check(cudaFree(arena0), \"cudaFree(arena0)\");");
                self.line("}");
            }
        }
    }
}
