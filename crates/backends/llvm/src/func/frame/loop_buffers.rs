//! loop buffers: in-place writes, state slots, enter / trip buffers and carried descriptors

use super::*;

impl<'a> FnEmit<'a> {
    /// The `Update` `m` (at position `at`) may write its array `arr` in place:
    /// the bytes are one buffer this function owns — and, inside a loop, the
    /// trip makes (else its next trip reads them) — and no view of them is
    /// read after `at` (`LastUsePlan::view_dead_after`); or a loop state's
    /// buffer the loop owns, no other view of which the trip reads again.
    pub(in crate::func) fn writes_in_place(&self, m: MorphismId, arr: ObjectId, at: u32) -> bool {
        let loops: Vec<&Loop> = self
            .loops
            .iter()
            .filter(|lp| lp.body.contains(&m))
            .collect();
        match &self.roots(arr, &[], true)[..] {
            [Root::Obj(_)] => {
                let [Root::Obj(c)] = self.roots(arr, &[], false)[..] else {
                    return false;
                };
                self.ir.object(c).expect("object").kind == ObjectKind::Temporary
                    && !self.host_free(c)
                    && loops.iter().all(|lp| self.made_in(lp, c))
                    && !self
                        .sharing(&Root::Obj(c))
                        .into_iter()
                        .any(|o| self.born(o) < at && self.guard_read(o, Some(m)))
                    && self.lup.view_dead_after(c, at)
            }
            [Root::Merge(merge, p)] => {
                loops
                    .iter()
                    .find(|lp| lp.plan.merge == *merge)
                    .is_some_and(|lp| {
                        self.state_owner(lp, p).is_some()
                            && self.state_death(lp, p, Some((m, at))) <= at
                    })
            }
            _ => false,
        }
    }

    /// **S65 rule R's buffer assignment for a loop's state**, after the
    /// in-place `Update`s: the `LastUsePlan` interval coloring across the back
    /// edge. The next state's bytes `r`, first written at position `W` of the
    /// trip, replace the state the trip reads until `D` ([`Self::state_death`]):
    /// `D < W` — one buffer; else two (ping-pong; one more per extra state
    /// position the bytes become). A state the trip writes in place lives in
    /// the loop's own buffer: the init's when it dies at the entry, else one
    /// copy made there (the one copy the facts require).
    pub(super) fn plan_loop_buffers(&mut self) {
        let mut out = Vec::new();
        for lp in &self.loops {
            let merge = lp.plan.merge;
            let paths: Vec<Vec<u32>> = array_paths(&self.obj_ty(merge))
                .into_iter()
                .filter(|p| self.is_view(merge, p) && !self.device_carried(merge, p))
                .collect();
            let next = |q: &[u32]| self.roots(lp.plan.back_route, &ext(&[0], q), true);
            let written = |p: &[u32]| {
                self.sharing(&Root::Merge(merge, p.to_vec()))
                    .iter()
                    .any(|&t| self.update_aliases.contains_key(t))
            };
            for p in paths
                .iter()
                .filter(|p| written(p) && self.state_owner(lp, p) == Some(false))
            {
                out.push(LoopBuf::Own {
                    merge,
                    path: p.clone(),
                });
            }
            let mut done: Vec<ObjectId> = Vec::new();
            for r in paths.iter().flat_map(|p| next(p)) {
                let Root::Obj(r) = r else { continue };
                if done.contains(&r) || !self.made_in(lp, r) || self.host_free(r) {
                    continue;
                }
                done.push(r);
                let at: Vec<Vec<u32>> = paths
                    .iter()
                    .filter(|q| next(q).contains(&Root::Obj(r)))
                    .cloned()
                    .collect();
                let unit = self.resident.as_ref().and_then(|p| p.unit_writer(r));
                let w = self
                    .ir
                    .in_edges(r)
                    .iter()
                    .copied()
                    .chain(resident::grouped_writer(&self.groups, r))
                    .chain(unit);
                let w = w.filter_map(|m| self.lup.position(m)).min().unwrap_or(0);
                if at
                    .iter()
                    .map(|q| self.state_death(lp, q, None))
                    .max()
                    .unwrap_or(0)
                    < w
                {
                    continue; // one buffer: the state is dead before r is written
                }
                let take = match &at[..] {
                    [q] if !written(q) => match self.roots(lp.plan.init, q, false)[..] {
                        [Root::Obj(a)]
                            if self.takeable(lp, q, a)
                                && lower_ty(&self.obj_ty(a)) == lower_ty(&self.obj_ty(r)) =>
                        {
                            Some(a)
                        }
                        _ => None,
                    },
                    _ => None,
                };
                out.push(LoopBuf::Colors {
                    merge,
                    root: r,
                    at,
                    take,
                });
            }
        }
        self.bufs = out;
    }

    /// The address of the pointer cell of state position `p` of `merge`.
    pub(super) fn state_cell(&mut self, merge: ObjectId, p: &[u32]) -> String {
        let Some((&k0, rest)) = p.split_first() else {
            return self.slot(merge).expect("merge slot");
        };
        let (_, mut cell) = self.component_ptr(merge, k0).expect("state field");
        let mut ty = self
            .obj_ty(merge)
            .component_ty(k0)
            .cloned()
            .expect("state ty");
        let mut paths = sub_paths(self.views.get(merge).map_or(&[][..], |v| v), k0);
        for &k in rest {
            let llt = lower_ptr_paths(&ty, &paths).expect("state lowers");
            cell = self.field_ptr(&cell, &ty, &llt, k).expect("state field");
            (paths, ty) = (
                sub_paths(&paths, k),
                ty.component_ty(k).cloned().expect("state ty"),
            );
        }
        cell
    }

    pub(super) fn state_ty(&self, merge: ObjectId, p: &[u32]) -> Ty {
        p.iter().fold(self.obj_ty(merge), |t, &k| {
            t.component_ty(k).cloned().expect("state ty")
        })
    }

    /// A buffer a loop's state may point at: a task's frame dies when it
    /// returns while the state's views (in `%Frame`) are read after, so there
    /// an arena block (the entry releases it with the rest).
    pub(super) fn state_alloc(&mut self, llt: &str) -> String {
        let name = format!("%ls{}", self.fresh());
        if matches!(self.guard_flavor, GuardFlavor::Task) {
            let (bytes, align) = (llt_bytes(llt), llt_align(llt));
            self.allocas.push_str(&format!(
                "  {name} = call ptr @mapal_rt_alloc(i64 {bytes}, i64 {align})\n"
            ));
        } else {
            self.entry_alloc(&name, llt, None);
        }
        name
    }

    /// Loop entry, after the init's pointers became the state's: a state the
    /// trip writes in place gets the loop's own copy of the init.
    pub(crate) fn loop_enter_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs.clone() {
            let LoopBuf::Own { merge: m, path } = b else {
                continue;
            };
            if m != merge {
                continue;
            }
            let llt = lower_ty(&self.state_ty(merge, &path)).expect("state lowers");
            let (cell, src) = (self.state_cell(merge, &path), self.tmp());
            self.line(format!("{src} = load ptr, ptr {cell}"));
            let own = self.state_alloc(&llt);
            self.emit_memcpy(&own, &src, &llt);
            self.line(format!("store ptr {own}, ptr {cell}"));
        }
    }

    /// Each trip's first instructions: a colored next state's buffer for this
    /// trip — the first of its slot and its spares no state position points at.
    pub(crate) fn loop_trip_buffers(&mut self, merge: ObjectId) {
        for b in self.bufs.clone() {
            let LoopBuf::Colors {
                merge: m,
                root,
                at,
                take,
            } = b
            else {
                continue;
            };
            if m != merge {
                continue;
            }
            let llt = lower_ty(&self.obj_ty(root)).expect("root lowers");
            let mut bufs = vec![self.slot(root).expect("root slot")];
            for i in 0..at.len() {
                let spare = match take.filter(|_| i == 0) {
                    Some(a) => self.slot(a).expect("taken slot"),
                    None => self.state_alloc(&llt),
                };
                bufs.push(spare);
            }
            let mut curs = Vec::new();
            for q in &at {
                let (cell, c) = (self.state_cell(merge, q), self.tmp());
                self.line(format!("{c} = load ptr, ptr {cell}"));
                curs.push(c);
            }
            let mut w = bufs.last().expect("a spare").clone();
            for buf in bufs[..bufs.len() - 1].iter().rev() {
                let mut busy = "false".to_string();
                for c in &curs {
                    let (e, o) = (self.tmp(), self.tmp());
                    self.line(format!("{e} = icmp eq ptr {c}, {buf}"));
                    self.line(format!("{o} = or i1 {busy}, {e}"));
                    busy = o;
                }
                let s = self.tmp();
                self.line(format!("{s} = select i1 {busy}, ptr {w}, ptr {buf}"));
                w = s;
            }
            for (o, _) in self.ir.objects() {
                if self.alias_root(o) == root {
                    self.slots.insert(o, w.clone());
                }
            }
        }
    }

    /// The object whose storage `o` shares through in-place `Update`s.
    pub(super) fn alias_root(&self, mut o: ObjectId) -> ObjectId {
        while let Some(&s) = self.update_aliases.get(o) {
            o = s;
        }
        o
    }

    /// The `carried=` field of a loop's `; mapal.move` line: per array position
    /// of the state, its bytes and how the trip carries it — `value` (inline in
    /// the products: the OFF arm), `device`, `inplace` (the state's buffer, the
    /// trip writing it), `two` (colored) or `one` (the next state's).
    pub(crate) fn carried_desc(&self, merge: ObjectId) -> String {
        let desc: Vec<String> = array_paths(&self.obj_ty(merge))
            .into_iter()
            .map(|p| {
                let bytes = lower_ty(&self.state_ty(merge, &p)).map_or(0, |l| llt_bytes(&l));
                let colored = self.bufs.iter().any(
                    |b| matches!(b, LoopBuf::Colors { merge: m, at, .. } if *m == merge && at.contains(&p)),
                );
                let state = Root::Merge(merge, p.clone());
                let written = self.sharing(&state).iter().any(|&t| self.update_aliases.contains_key(t));
                let mode = match () {
                    _ if self.device_carried(merge, &p) => "device",
                    _ if !self.is_view(merge, &p) => "value",
                    _ if colored => "two",
                    _ if written => "inplace",
                    _ => "one",
                };
                let at: Vec<String> = p.iter().map(u32::to_string).collect();
                format!("{}:{bytes}:{mode}", if at.is_empty() { "-".into() } else { at.join(".") })
            })
            .collect();
        if desc.is_empty() {
            "-".into()
        } else {
            desc.join(",")
        }
    }
}
