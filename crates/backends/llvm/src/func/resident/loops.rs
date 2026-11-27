//! resident loops: trip moves, loop enter / back / exit, teardown and the entry arena

use super::*;

impl<'a> FnEmit<'a> {
    /// A trip's movement as the plan has it: `(host copies, h2d, d2h,
    /// retained sites)` in bytes ([`Self::move_line`]'s numbers).
    pub(crate) fn trip_moves(&self, plan: &mapal_ir::LoopPlan) -> (u64, u64, u64, usize) {
        let bytes = |o: ObjectId| lower_ty(&self.obj_ty(o)).map_or(0, |l| llt_bytes(&l));
        let (mut h2d, mut d2h, mut retained) = (0u64, 0u64, 0usize);
        // An `Update` whose source is still read after it (or is not this
        // function's to write) copies it: the one host copy the facts require.
        let copy: u64 = plan
            .decide_order
            .iter()
            .chain(&plan.advance_order)
            .map(|&m| self.ir.morphism(m).expect("morphism resolves"))
            .filter(|mo| mo.op == Operation::Update && !self.update_aliases.contains_key(mo.target))
            .map(|mo| bytes(mo.target))
            .sum();
        if let Some(p) = &self.resident {
            // (A guard arm's work counts: the arm may run on every trip.)
            for &m in plan.decide_order.iter().chain(&plan.advance_order) {
                let t = self.ir.morphism(m).expect("morphism resolves").target;
                if p.upload.contains_key(t) && !p.devgen.contains_key(m) {
                    h2d += match p.delta.get(t) {
                        Some(du) => {
                            let elem = self.obj_ty(t).component_ty(0).cloned().expect("elem");
                            du.stride * crate::profile::elem_bytes(&elem)
                        }
                        None => bytes(t),
                    };
                }
                if p.elem.contains_key(m) {
                    d2h += bytes(t);
                }
                // The values a launch leaves in device memory, copied out
                // whole unless kept (`emit_site_resident`).
                let mut outs = Vec::new();
                if let Some(u) = p.units.get(m) {
                    // A unit (a row-block one launches at its tile site).
                    outs.extend(u.members.iter().filter(|mb| mb.store).map(|mb| mb.out));
                } else if self.device.contains_key(m) && !p.skip.contains_key(m) {
                    match self.groups.get(m) {
                        Some(g) => {
                            if g.store_out {
                                outs.push(t);
                            }
                            outs.extend(g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out));
                        }
                        None => outs.push(t),
                    }
                } else if p.rowfold.contains_key(m) || p.cells.contains_key(m) {
                    outs.push(t);
                }
                for o in outs {
                    if !p.keep.contains_key(o) && !p.is_carry_next(o) {
                        // A retained output (`retain.rs`) rebuilds whole once
                        // per loop entry; a trip copies out its one new row.
                        d2h += match self.retain.get(o) {
                            Some(r) => {
                                r.stride * crate::profile::elem_bytes(array_elem(&self.obj_ty(o)))
                            }
                            None => bytes(o),
                        };
                    }
                }
                retained += self.retain.contains_key(t) as usize;
                // caching-memory `retain-cell`: a cell launched over its delta.
                retained += p.cells.get(m).is_some_and(|c| c.retain.is_some()) as usize;
            }
        }
        (copy, h2d, d2h, retained)
    }

    /// Loop entry: the carry's buffer gets the init state, copied in once.
    pub(crate) fn resident_loop_enter(&mut self, merge: ObjectId, init: ObjectId) {
        // long-prefill `carry-take`: where the loop is entered (`LastUsePlan`).
        let take = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).carry_take;
        let entered = self
            .ir
            .in_edges(merge)
            .iter()
            .copied()
            .find(|&m| {
                self.ir
                    .morphism(m)
                    .is_some_and(|x| x.op == Operation::LoopEnter)
            })
            .and_then(|m| self.lup.position(m));
        for c in self.carries_of(merge) {
            // S65: a device-built init array is the carry's first state — its
            // buffer moves to the carry (its slot is emptied, so one free).
            // long-prefill `carry-take`: so does a host-built one whose device
            // copy is valid and read by nothing past the loop's entry (its
            // death there): its host bytes stay, its device bytes are the
            // carry's — no second upload of what the device holds.
            let dies = |a: ObjectId| {
                take && entered.is_some_and(|p| {
                    self.lup.death(a).is_some_and(|d| d <= p)
                        || self
                            .resident
                            .as_ref()
                            .is_some_and(|plan| self.bytes_enter(plan, a, init, p))
                })
            };
            if let Some(a) = self.pair_source(init, c.k)
                && (self.host_free(a) || dies(a))
                && self.dvalid.contains_key(a)
            {
                let (sa, sm) = (self.dslot(a), self.dslot(c.key()));
                // long-prefill `carry-take`: a host-built init's buffer taken
                // (intent.py `carry-take`); the S65 move's text is unchanged.
                if !self.host_free(a) {
                    let bytes = self.array_bytes(&self.obj_ty(a));
                    self.line(format!("; mapal.take carry={} bytes={bytes}", c.k));
                }
                let d = self.tmp();
                self.line(format!("{d} = load i64, ptr {sa}"));
                self.line(format!("store i64 {d}, ptr {sm}"));
                self.line(format!("store i64 0, ptr {sa}"));
                self.taken.push(a);
                // `once-range`: the bytes are the in-place chain's buffer.
                if let Some(r) = self
                    .resident
                    .as_ref()
                    .map(|p| p.alias(a))
                    .filter(|&r| r != a)
                {
                    self.taken.push(r);
                }
                self.dvalid.remove(a);
                self.dvalid.insert(c.key(), ());
                continue;
            }
            let ty = self
                .obj_ty(merge)
                .component_ty(c.k)
                .cloned()
                .expect("carry ty");
            let bytes = self.array_bytes(&ty);
            let d = self.dalloc(c.key(), bytes);
            // A `+0.0` fill (`(0.0, n) -> fill`, every byte zero) is zeroed
            // there: no host bytes cross.
            if self
                .pair_source(init, c.k)
                .is_some_and(|a| self.zero_fill(a))
            {
                self.line(format!("call void @mapal_gpu_zero(i64 {d}, i64 {bytes})"));
            } else {
                let h = self.array_operand_ptr(init, Some(c.k)).expect("init ptr");
                self.line(format!(
                    "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {bytes})"
                ));
            }
            self.dvalid.insert(c.key(), ());
        }
    }

    /// `a` is a `Fill` of a float `+0.0`: all its bytes are zero.
    pub(super) fn zero_fill(&self, a: ObjectId) -> bool {
        let Some(m) = self
            .ir
            .in_edges(a)
            .first()
            .and_then(|&m| self.ir.morphism(m))
        else {
            return false;
        };
        let v = self
            .pair_source(m.source, 0)
            .and_then(|x| self.ir.object(x))
            .and_then(|x| x.value.as_ref());
        m.op == Operation::Fill
            && match v {
                Some(mapal_ir::Value::F32(x)) => x.to_bits() == 0,
                Some(mapal_ir::Value::F64(x)) => x.to_bits() == 0,
                _ => false,
            }
    }

    /// The back edge: the next state's buffer becomes the carry's, and the old
    /// state's buffer (dead past the back edge) is the next launch's output.
    pub(crate) fn resident_loop_back(&mut self, merge: ObjectId) {
        for c in self.carries_of(merge) {
            let (cs, ns) = (self.dslot(c.key()), self.dslot(c.next));
            let (old, new) = (self.tmp(), self.tmp());
            self.line(format!("{old} = load i64, ptr {cs}"));
            self.line(format!("{new} = load i64, ptr {ns}"));
            self.line(format!("store i64 {new}, ptr {cs}"));
            self.line(format!("store i64 {old}, ptr {ns}"));
        }
    }

    /// The exit: the exit object is the carry's buffer; copied out whole when
    /// the host reads it whole.
    pub(crate) fn resident_loop_exit(&mut self, merge: ObjectId, exit: ObjectId) {
        for c in self.carries_of(merge) {
            if c.exit != Some(exit) {
                continue;
            }
            let bytes = self.array_bytes(&self.obj_ty(exit));
            self.dvalid.insert(c.key(), ());
            if !self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(exit))
            {
                let d = self.dload(exit);
                let h = self.slot(exit).expect("exit slot");
                self.line(format!(
                    "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 0, i64 {bytes})"
                ));
            }
        }
    }

    /// Function exit: free every buffer still held (a no-op on freed slots).
    pub(in crate::func) fn resident_teardown(&mut self) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let entry =
            rungs.arena && rungs.entry_arena && self.resident.is_some() && self.entry_arena();
        // `async-h2d`: every upload is enqueued without a host wait (the
        // runtime stages the source), so no upload is a sync point: the trip
        // waits only at a copy-out, which exists only where a host morphism
        // reads a device value (`upload`/`keep`/`elem` in the plan).
        if rungs.async_h2d && self.resident.is_some() {
            self.body = self.body.replace(
                "call void @mapal_gpu_h2d(",
                "call void @mapal_gpu_h2d_async(",
            );
        }
        // Pins end with the frame they pinned.
        if self.body.contains("@mapal_gpu_pin(") {
            self.line("call void @mapal_gpu_unpin()");
        }
        let slots: Vec<String> = self
            .dslots
            .iter()
            .filter(|&(o, _)| self.arena_offset(o).is_none())
            .map(|(_, s)| s.clone())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for s in slots.into_iter().chain(self.split_slots.clone()) {
            if seen.insert(s.clone()) {
                self.line(format!("call void @mapal_gpu_free(ptr {s})"));
            }
        }
        if self.arena_slot() || entry {
            self.line("call void @mapal_gpu_free(ptr %darena)");
        }
    }

    /// **`entry-arena`: the arena at function entry, loop-cone buffers in
    /// it.** Every slot the walk allocated with `mapal_gpu_alloc` (`dallocs`)
    /// that no `free_after` death frees (a loop-cone buffer, a carry's next, a
    /// split workspace) is placed after the straight-line members, each live
    /// over the whole function — `[0, ∞)` in the same coloring, so it shares
    /// no byte with anything: a back edge may swap it into another slot at any
    /// trip. The entry block makes the arena (static size) and seeds those
    /// slots, so their first-use allocation finds its buffer. A slot freed
    /// before exit keeps its own allocation. The runtime releases a member
    /// with its arena, whichever slot holds it at exit (`mapal_gpu_free`).
    /// Returns whether the function has an arena.
    ///
    /// ponytail: whole-function intervals, so the cones of two loops in one
    /// function never share bytes; per-loop spans (`loop_plan` through the
    /// death of each carry's exit alias) when that footprint matters.
    pub(super) fn entry_arena(&mut self) -> bool {
        let freed: std::collections::BTreeSet<&String> = self
            .resident
            .iter()
            .flat_map(|p| p.free_after.values().flatten())
            .filter(|&&o| self.arena_offset(o).is_none())
            .filter_map(|&o| self.dslots.get(o))
            .collect();
        let mut cone: Vec<(String, u64, u32)> = Vec::new();
        for (slot, bytes, from) in &self.dallocs {
            if freed.contains(slot) {
                continue;
            }
            match cone.iter_mut().find(|(s, _, _)| s == slot) {
                Some(c) => {
                    c.1 = c.1.max(*bytes);
                    c.2 = c.2.min(*from);
                }
                None => cone.push((slot.clone(), *bytes, *from)),
            }
        }
        let (line, spans) = self
            .resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .map_or((0, Vec::new()), |a| (a.bytes, a.spans.clone()));
        // A member a carry took lives on in the carry's slots (swapped at
        // every back edge, then the exit's): to exit.
        let spans: Vec<(u32, u32, u64, u64)> = spans
            .into_iter()
            .map(|(o, b, d, off, sz)| {
                let d = if self.taken.contains(&o) { u32::MAX } else { d };
                (b, d, off, sz)
            })
            .collect();
        // caching-memory `cone-span`: each buffer from its position to exit,
        // first-fit among the members that are live then (and the buffers
        // placed before it); off, after the members over the whole function.
        let (offs, total) = if gpu::host_rungs(self.profile, self.contract, self.gpu_arms).cone_span
        {
            cone_fit(
                &spans,
                line,
                &cone.iter().map(|c| (c.2, c.1)).collect::<Vec<_>>(),
            )
        } else {
            let live: Vec<(u32, u32, u64)> =
                cone.iter().map(|&(_, b, _)| (0, u32::MAX, b)).collect();
            let (offs, cone_bytes) = color(&live);
            (
                offs.into_iter().map(|o| line + o).collect(),
                line + cone_bytes,
            )
        };
        if total == 0 {
            return false;
        }
        let mut text = String::new();
        if line == 0 {
            text.push_str("  %darena = alloca i64\n  store i64 0, ptr %darena\n");
        }
        let base = self.tmp();
        text.push_str(&format!(
            "  {base} = call i64 @mapal_gpu_arena(ptr %darena, i64 {total})\n"
        ));
        for ((slot, _, _), off) in cone.iter().zip(offs) {
            let d = self.tmp();
            text.push_str(&format!(
                "  {d} = add i64 {base}, {off}\n  store i64 {d}, ptr {slot}\n"
            ));
        }
        self.allocas.push_str(&text);
        true
    }

    /// The function has arena members, so `%darena` exists.
    pub(crate) fn arena_slot(&self) -> bool {
        self.resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .is_some_and(|a| a.bytes > 0)
    }
}
