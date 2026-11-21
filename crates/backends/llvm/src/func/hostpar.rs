//! onto-device `host-par`: the host part of a GPU build on the CPU dispatch
//! machinery.
//!
//! A GPU build's host walk is sequential — the device is its parallelism —
//! so a host-placed bulk site (a trap-capable read the device cannot take, a
//! site whose roofline keeps it on the host) ran on one core while the CPU
//! build of the same program dispatches it across the pool. Here a host map
//! with the shape `path_plan` splits in the CPU build — a clean body (no trap,
//! no effect), a static extent, not a tile site, not a range map, run once
//! (straight-line: outside every loop cone and guard arm), no operand held
//! only on the device — becomes one split task: its loop over `[lo, hi)` in a
//! function of its own reading the walk's storage through an environment of
//! pointers, registered, launched and finished on the pool where the host's
//! measured dispatch cost (`DispatchCost`, the CPU build's rule) says it pays.
//! The slice function must be self-contained (every value it names defined
//! in it), else the site is emitted inline as before. Off
//! (`--gpu-off=host-par`): the sequential walk.

use super::*;

impl<'a> FnEmit<'a> {
    /// A fresh emitter for one site's slice function: the walk's facts and
    /// storage decisions (by-ref, aliases, elisions, views), its own text.
    fn fork_slice(&self) -> FnEmit<'a> {
        let mut s = FnEmit::new(
            self.ir,
            self.f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        s.gpu_arms = self.gpu_arms;
        s.split_range = true;
        s.byref = self.byref.clone();
        s.ptr_resident = self.ptr_resident.clone();
        s.update_aliases = self.update_aliases.clone();
        s.elided_updates = self.elided_updates.clone();
        s.elided_arrays = self.elided_arrays.clone();
        s.views = self.views.clone();
        s.views_done = self.views_done;
        s.by_ref = self.by_ref;
        s.bufs = self.bufs.clone();
        s.loops = self.loops.clone();
        s.root_index = self.root_index.clone();
        s.device = self.device.clone();
        s.tile_plan = self.tile_plan.clone();
        s
    }

    /// `host-par`: emit host map `m` as a split task on the pool. `false`
    /// leaves it to the inline emission (the gates in the module doc).
    pub(super) fn host_par_map(&mut self, m: MorphismId) -> bool {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if self.profile.gpu().is_none()
            || !rungs.host_par
            || self.par.is_some()
            || self.split_range
            || self.f != self.ir.entry()
            || self.lanes.is_some()
            || !self.cell_slots.is_empty()
            || self.profile.dispatch.is_none()
        {
            return false;
        }
        let morph = self.ir.morphism(m).expect("map resolves");
        let (source, target) = (morph.source, morph.target);
        let Operation::Map { body, captures } = morph.op else {
            return false;
        };
        if self.device.contains_key(m)
            || self
                .tile_plan
                .as_ref()
                .is_some_and(|p| p.sites.contains_key(m))
            || self.gated.contains_key(m)
            || !self.attrs.clean(body)
            || self.range_map(m).is_some()
            || matches!(
                self.ir.map_law(m),
                Some(ElemSrc::Apply {
                    outside: Some(_),
                    ..
                })
            )
            || !self.straight_line().contains_key(m)
        {
            return false;
        }
        let Ty::Array { size: n, .. } = self.obj_ty(target) else {
            return false;
        };
        let operands: Vec<ObjectId> = if captures == 0 {
            vec![source]
        } else {
            (0..=captures)
                .filter_map(|k| self.pair_source(source, k))
                .collect()
        };
        if operands.iter().chain([&target]).any(|&o| self.host_free(o)) {
            return false;
        }
        // The CPU build's dispatch rule: a split whose serial estimate is
        // under one wave runs inline; else the lanes its estimate earns.
        let task = mapal_ir::Task {
            kind: TaskKind::Split { site: m, n },
            deps: Vec::new(),
            rank: 0,
            trap_min: None,
            pinned: false,
        };
        if self.under_dispatch_cost(&task) || self.task_width(&task) == Some(1) {
            return false;
        }
        let width = self.task_width(&task).filter(|&w| w > 1).unwrap_or(0);

        // The slice, every slot an environment entry.
        let mut slice = self.fork_slice();
        let objs: Vec<(ObjectId, String)> =
            self.slots.iter().map(|(o, s)| (o, s.clone())).collect();
        for (i, (o, _)) in objs.iter().enumerate() {
            slice.slots.insert(*o, format!("%hpe{i}"));
        }
        slice.emit_map(m, source, target, body, captures);
        if slice.heap_used {
            slice.heap_teardown();
        }
        let text = format!("{}{}", slice.allocas, slice.body);
        let used: Vec<usize> = (0..objs.len())
            .filter(|i| uses_name(&text, &format!("%hpe{i}")))
            .collect();
        if !self_contained(&text) {
            return false;
        }
        let id = self.host_par_n;
        self.host_par_n += 1;
        let fname = format!("{}_hpar{id}", self.fnames[self.f]);
        let mut loads = String::new();
        for (j, &i) in used.iter().enumerate() {
            loads.push_str(&format!(
                "  %hpg{i} = getelementptr ptr, ptr %frame, i64 {j}\n  %hpe{i} = load ptr, ptr %hpg{i}\n"
            ));
        }
        self.host_par_fns.push_str(&format!(
            "define internal void @{fname}(i64 %lo, i64 %hi, ptr %frame) {{\nentry:\n{loads}{}{}  ret void\n}}\n",
            slice.allocas, slice.body
        ));
        // The walk: the environment, then one split task on the pool.
        let env = format!("%hpenv{id}");
        self.allocas
            .push_str(&format!("  {env} = alloca [{} x ptr]\n", used.len().max(1)));
        for (j, &i) in used.iter().enumerate() {
            let p = self.tmp();
            self.line(format!("{p} = getelementptr ptr, ptr {env}, i64 {j}"));
            self.line(format!("store ptr {}, ptr {p}", objs[i].1));
        }
        let h = self.tmp();
        self.line(format!("{h} = call ptr @mapal_par_begin(i32 1)"));
        self.line(format!(
            "call void @mapal_par_task(ptr {h}, i32 0, i32 1, ptr @{fname}, i64 {n}, i32 0, i64 0, i32 0, i32 {width})"
        ));
        self.line(format!("call void @mapal_par_launch(ptr {h}, ptr {env})"));
        self.line(format!("call void @mapal_par_finish(ptr {h})"));
        true
    }
}

/// Whether `text` names the value `name` (not a longer name it prefixes).
fn uses_name(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        !text[at + name.len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    })
}

/// Every `%name` the function text uses is defined in it (a value or a
/// label) or is a slice parameter / environment entry.
fn self_contained(text: &str) -> bool {
    let ident = |s: &str| -> usize {
        s.char_indices()
            .find(|&(_, c)| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
            .map_or(s.len(), |(i, _)| i)
    };
    let mut defined: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix('%') {
            let l = ident(rest);
            if rest[l..].trim_start().starts_with('=') {
                defined.insert(&rest[..l]);
            }
        } else if let Some(l) = t.strip_suffix(':') {
            defined.insert(l);
        }
    }
    let mut rest = text;
    while let Some(at) = rest.find('%') {
        let tail = &rest[at + 1..];
        let l = ident(tail);
        let name = &tail[..l];
        if !name.is_empty()
            && !defined.contains(name)
            && !matches!(name, "lo" | "hi" | "frame")
            && !name.starts_with("hpe")
            && !name.starts_with("hpg")
            && !name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            return false;
        }
        rest = &tail[l..];
    }
    true
}
