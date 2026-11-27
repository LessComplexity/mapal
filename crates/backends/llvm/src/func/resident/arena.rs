//! device buffer lifetimes: death points, the arena coloring, device slots and allocation

use super::*;

impl<'a> FnEmit<'a> {
    /// The death of `o`'s device buffer: [`Self::death_of`] over every
    /// object sharing it (`once-range`: a straight-line in-place chain lives
    /// to its last array's death; `None` when one escapes).
    pub(super) fn buffer_death(&self, plan: &ResidentPlan, o: ObjectId) -> Option<u32> {
        let mut d = self.death_of(o)?;
        for (t, _) in plan.inplace.iter() {
            if plan.alias(t) == o {
                d = d.max(self.death_of(t)?);
            }
        }
        Some(d)
    }

    /// `o`'s last use on the device: its IR death, or (plan-cost `linear`)
    /// the launch of the site table that reads it for an earlier term.
    pub(super) fn death_of(&self, o: ObjectId) -> Option<u32> {
        let d = self.lup.death(o)?;
        let reads = |t: MorphismId| {
            let src = self.ir.morphism(t).expect("term").source;
            let site = self.tile_plan.as_ref().and_then(|p| p.sites.get(t));
            site.is_some_and(|x| {
                [x.a.slot, x.b.slot]
                    .iter()
                    .any(|&k| self.pair_source(src, k) == Some(o))
            })
        };
        let table = self
            .groups
            .iter()
            .filter(|(_, g)| g.terms.iter().any(|&t| reads(t)))
            .filter_map(|(m, _)| self.lup.position(m))
            .max();
        Some(table.map_or(d, |p| d.max(p)))
    }

    /// The zero passes `zero-fold` folds into earlier kernels under the
    /// arena `a` (a trial on a copy of the plan) — tails only: a hoisted zero
    /// (`zero-hoist`) is still an op of its own, so it never decides the
    /// layout against a fold.
    pub(super) fn zero_folds(&self, plan: &ResidentPlan, a: &Arena) -> usize {
        let mut trial = plan.clone();
        trial.arena = Some(a.clone());
        self.zero_fold(&mut trial);
        trial.zeroed.len() - trial.zero_hoist_ix.len()
    }

    pub(super) fn arena(&self, plan: &ResidentPlan, by_size: bool) -> Arena {
        let gated_use = |o: ObjectId| {
            self.ir.out_edges(o).iter().any(|&m| {
                self.gated.contains_key(m)
                    || self
                        .ir
                        .out_edges(self.ir.morphism(m).expect("m").target)
                        .iter()
                        .any(|&m2| self.gated.contains_key(m2))
            })
        };
        let members: Vec<ObjectId> = plan
            .free_after
            .values()
            .flatten()
            .copied()
            .filter(|&o| !gated_use(o))
            .collect();
        // An epilogue member's output (S54 R1) is written by its site's launch,
        // which reads the site's operands: it is born at the site, not at its
        // own (later) map — else it could take the bytes of an operand that dies
        // at the site while the kernel still reads it (S54 integration:
        // px_bias_100's `d` took `a`'s offset and the checksum raced).
        // kernel-merging `horizontal`: an operand of a deferred launch lives
        // until its group launches (at the group's last member).
        let acc = (!plan.hz.is_empty()).then(|| self.ir.access_plan(self.f));
        let mut until: SecondaryMap<ObjectId, u32> = SecondaryMap::new();
        for g in &plan.hz {
            let at = g.members.last().and_then(|&m| self.lup.position(m));
            let (Some(acc), Some(at)) = (&acc, at) else {
                continue;
            };
            for &m in &g.members {
                let (r, _) = self.step_access(plan, acc, m);
                for o in r {
                    let o = plan.alias(o);
                    let u = until.get(o).copied().unwrap_or(0).max(at);
                    until.insert(o, u);
                }
            }
        }
        let live: Vec<(u32, u32, u64)> = members
            .iter()
            .map(|&o| {
                let birth = self
                    .ir
                    .in_edges(o)
                    .iter()
                    .copied()
                    .chain(grouped_writer(&self.groups, o))
                    .chain(plan.unit_writer(o))
                    .filter_map(|p| self.lup.position(p))
                    .min();
                let death = self
                    .buffer_death(plan, o)
                    .map(|d| d.max(until.get(o).copied().unwrap_or(0)));
                let bytes = self.array_bytes(&self.obj_ty(o));
                (
                    birth.expect("produced"),
                    death.expect("freed at death"),
                    bytes,
                )
            })
            .collect();
        let (offs, bytes) = color_by(&live, by_size);
        let mut a = Arena {
            bytes,
            ..Arena::default()
        };
        for ((o, off), &(b, d, bytes)) in members.into_iter().zip(offs).zip(&live) {
            a.offset.insert(o, off);
            a.spans
                .push((o, b, d, off, bytes.div_ceil(ARENA_ALIGN) * ARENA_ALIGN));
        }
        let built = plan.devgen.keys().chain(plan.skip.keys());
        let built: Vec<ObjectId> = built
            .map(|m| self.ir.morphism(m).expect("m").target)
            .collect();
        // An epilogue member (S54 R1) is skipped on the host too, but a stored
        // one not kept is copied out to a host reader: it keeps its slot.
        let copied_out = |o: ObjectId| {
            !plan.keep.contains_key(o)
                && (self
                    .groups
                    .values()
                    .any(|g| g.chain.iter().any(|mb| mb.out == o && mb.store))
                    || plan.unit_writer(o).is_some())
        };
        for o in plan.keep.keys().chain(built) {
            if !plan.is_carry_next(o)
                && plan.alias(o) == o
                && !self.lup.escapes(o)
                && self.ir.try_owner(o) == Some(self.f)
                && !copied_out(o)
            {
                a.host_free.insert(o, ());
            }
        }
        a
    }

    /// S65 `by-ref`: a device carry whose loop products forward a pointer has
    /// no host bytes — its views, its next state, and its exit object unless
    /// the host reads that whole (then copied out into its own). With or
    /// without the arena: the host state holds no bytes to point at.
    pub(super) fn carry_host_free(&self, plan: &ResidentPlan) -> Vec<ObjectId> {
        let mut out = Vec::new();
        for c in &plan.carries {
            if !self.is_view(c.merge, &[c.k]) {
                continue;
            }
            let exit = c.exit.filter(|x| plan.keep.contains_key(*x));
            for o in c.views.iter().copied().chain([c.next]).chain(exit) {
                if !self.lup.escapes(o) && self.ir.try_owner(o) == Some(self.f) {
                    out.push(o);
                }
            }
        }
        out
    }

    /// `o` has no host slot (S54 `arena`).
    pub(in crate::func) fn host_free(&self, o: ObjectId) -> bool {
        self.resident
            .as_ref()
            .and_then(|p| p.arena.as_ref())
            .is_some_and(|a| a.host_free.contains_key(o))
    }

    pub(super) fn arena_offset(&self, o: ObjectId) -> Option<(u64, u64)> {
        let a = self.resident.as_ref()?.arena.as_ref()?;
        Some((a.bytes, *a.offset.get(o)?))
    }

    /// **S54 `l2-window`: the buffer of site `m`'s launch the graph proves is
    /// read again soonest**, and its bytes. The output, when the next device
    /// site by `LastUsePlan` position reads it (a chain's intermediate: reuse
    /// distance one launch — its stores persist for the reader); else the
    /// operand whose panels the most blocks read (`a` row-panel: `gx` blocks,
    /// `b` column-panel: `gy`; on a tie `b`, whose readers row-major order puts
    /// `gx` launches apart). Only a buffer whose bytes fit the part's `l2_bytes`:
    /// a window larger than the cache it reserves persists nothing whole.
    pub(in crate::func) fn l2_window(
        &self,
        m: MorphismId,
        site: &TileSite,
        l: &gpu::Launch,
    ) -> Option<(Window, u64)> {
        let l2 = self.profile.gpu()?.l2_bytes;
        let morph = self.ir.morphism(m).expect("site");
        let out = self.array_bytes(&self.obj_ty(morph.target));
        let pos = |m| self.lup.position(m);
        let next = self
            .device_sites()
            .iter()
            .filter(|&(m2, _)| pos(m2) > pos(m))
            .min_by_key(|&(m2, _)| pos(m2))
            .map(|(m2, s2)| {
                let src = self.ir.morphism(m2).expect("site").source;
                s2.reads()
                    .into_iter()
                    .any(|k| self.pair_source(src, k) == Some(morph.target))
            });
        if next == Some(true) && out <= l2 {
            return Some((Window::Out, out));
        }
        let src_ty = self.obj_ty(morph.source);
        let bytes = |slot| self.array_bytes(src_ty.component_ty(slot).expect("tile operand"));
        [
            (Window::B, l.gy, bytes(site.b.slot)),
            (Window::A, l.gx, bytes(site.a.slot)),
        ]
        .into_iter()
        .filter(|&(_, _, b)| b <= l2)
        .max_by_key(|&(w, reuse, _)| (reuse, w == Window::B))
        .map(|(w, _, b)| (w, b))
    }

    /// A device site's, a cell kernel's (`devmap`) or an epilogue member's
    /// output (S54).
    pub(super) fn produced_on_device(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        cells: &SecondaryMap<MorphismId, Cell>,
        o: ObjectId,
    ) -> bool {
        self.ir.in_edges(o).iter().any(|&m| {
            sites.contains_key(m) || cells.contains_key(m) || self.member_src(m, 0).is_some()
        })
    }

    // --- emission ---------------------------------------------------------

    /// `o`'s device slot (`i64`, 0 = no buffer), minted zeroed in the entry
    /// block on first ask.
    pub(in crate::func) fn dslot(&mut self, o: ObjectId) -> String {
        let o = self.resident.as_ref().map_or(o, |p| p.alias(o));
        if let Some(s) = self.dslots.get(o) {
            return s.clone();
        }
        let s = format!("%d{}", self.fresh());
        self.allocas
            .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
        self.dslots.insert(o, s.clone());
        s
    }

    /// `o`'s device buffer, allocated on first use (a loop reuses it).
    pub(in crate::func) fn dalloc(&mut self, o: ObjectId, bytes: u64) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        // An arena member: base + offset, the arena made on the first ask
        // (lazily, so `eager-init` still overlaps the host work before it) —
        // or at entry (`entry-arena`), where the slot already holds its base.
        if let Some((total, off)) = self.arena_offset(o) {
            let base = self.tmp();
            if gpu::host_rungs(self.profile, self.contract, self.gpu_arms).entry_arena {
                self.line(format!("{base} = load i64, ptr %darena"));
            } else {
                self.line(format!(
                    "{base} = call i64 @mapal_gpu_arena(ptr %darena, i64 {total})"
                ));
            }
            self.line(format!("{d} = add i64 {base}, {off}"));
            self.line(format!("store i64 {d}, ptr {slot}"));
            return d;
        }
        self.line(format!(
            "{d} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {bytes})"
        ));
        let from = self.cone_birth(o);
        self.dallocs.push((slot, bytes, from));
        d
    }

    /// caching-memory `cone-span`: the position from which `o`'s buffer must
    /// hold its bytes — its outermost loop's entry (the `LoopEnter`'s
    /// position: every trip re-runs the body, so a buffer the body writes is
    /// live across all of it) when a loop's body makes it, else its first
    /// writer's (its own morphism, an epilogue's site, a unit's tile site, a
    /// kernel its zero pass rides). 0 (the whole function) where no canonical
    /// loop answers, or with the rung off.
    pub(in crate::func) fn cone_birth(&self, o: ObjectId) -> u32 {
        if !gpu::host_rungs(self.profile, self.contract, self.gpu_arms).cone_span {
            return 0;
        }
        // Its writers: its own morphism, and a launch that writes it before
        // that (an epilogue member at its site, a unit at its tile site) —
        // the arena members' births (`arena`) the same way.
        let mut writers: Vec<MorphismId> = self
            .ir
            .in_edges(o)
            .iter()
            .copied()
            .chain(grouped_writer(&self.groups, o))
            .chain(self.resident.as_ref().and_then(|p| p.unit_writer(o)))
            .collect();
        // `zero-fold`: an atomic site's zero pass runs as the tail of an
        // earlier kernel, which writes the bytes from there.
        if let Some(p) = &self.resident {
            let tails: Vec<MorphismId> = p
                .zero_by
                .iter()
                .filter(|(_, zs)| zs.iter().any(|(l, _)| writers.contains(l)))
                .map(|(e, _)| e)
                .collect();
            writers.extend(tails);
        }
        let Some(mut at) = writers.iter().filter_map(|&w| self.lup.position(w)).min() else {
            return 0;
        };
        for scc in self.ir.loop_structure(self.f) {
            let cyclic = scc.objects.contains(&o);
            for &merge in &scc.merges {
                let Some(lp) = self.ir.loop_plan(self.f, merge) else {
                    if cyclic {
                        return 0;
                    }
                    continue;
                };
                let body = writers
                    .iter()
                    .any(|w| lp.decide_order.contains(w) || lp.advance_order.contains(w));
                if !(body || cyclic || merge == o) {
                    continue;
                }
                let enter = self
                    .ir
                    .in_edges(merge)
                    .iter()
                    .filter(|&&e| {
                        self.ir
                            .morphism(e)
                            .is_some_and(|x| matches!(x.op, Operation::LoopEnter { .. }))
                    })
                    .filter_map(|&e| self.lup.position(e))
                    .min();
                match enter {
                    Some(p) => at = at.min(p),
                    None => return 0,
                }
            }
        }
        at
    }

    pub(in crate::func) fn dload(&mut self, o: ObjectId) -> String {
        let slot = self.dslot(o);
        let d = self.tmp();
        self.line(format!("{d} = load i64, ptr {slot}"));
        d
    }

    /// An array's bytes; a number's own (`scalar-fold`: a whole fold's
    /// one-element device buffer).
    pub(in crate::func) fn array_bytes(&self, ty: &Ty) -> u64 {
        match ty {
            Ty::Float { .. } | Ty::Int { .. } | Ty::Narrow(_) => crate::profile::elem_bytes(ty),
            _ => array_len(ty) * crate::profile::elem_bytes(array_elem(ty)),
        }
    }

    /// Replaces the host realization of `m` (a device-built or unread
    /// producer). `true` when `m` was handled here.
    pub(in crate::func) fn resident_instead(&mut self, m: MorphismId) -> bool {
        let Some(plan) = &self.resident else {
            return false;
        };
        if let Some(u) = plan.units.get(m).cloned() {
            self.emit_unit(&u);
            return true;
        }
        if plan.skip.contains_key(m) {
            return true;
        }
        if let Some(r) = plan.rowfold.get(m).cloned() {
            self.emit_rowfold(m, &r);
            return true;
        }
        if let Some(c) = plan.cells.get(m).cloned() {
            self.emit_cell(m, &c);
            return true;
        }
        let Some(g) = plan.devgen.get(m).cloned() else {
            return false;
        };
        let t = self.ir.morphism(m).expect("morphism resolves").target;
        let bytes = self.array_bytes(&self.obj_ty(t));
        let d = self.dalloc(t, bytes);
        self.line(format!(
            "call void @mapal_gpu_gen(i64 {}, i64 {d}, i64 {})",
            g.id as i64, g.n
        ));
        self.dvalid.insert(t, ());
        true
    }
}
