//! `unit_kernel`: the generated kernel of one unit

use super::*;

impl<'a> FnEmit<'a> {
    /// The unit's kernel: the bodies it calls, and its text (with its
    /// `; mapal.flow` and `; mapal.plan` intent lines).
    pub(super) fn unit_kernel(&self, u: &Unit) -> (Vec<(String, String)>, String) {
        let name = gpu::kernel_name(u.id);
        let geo = self.geo(u);
        let (blocks, threads) = (geo.gx, geo.block);
        let lt = |o: ObjectId| array_parts(&self.obj_ty(o)).0;
        let mut fns: Vec<(String, String)> = Vec::new();
        let mut g = UnitGen {
            out: String::new(),
            entry: String::new(),
            n: 0,
            cur: "body".into(),
            vals: vec![None; u.members.len()],
            calls: Vec::new(),
            ys: Vec::new(),
            gmem_ty: u.gmem.iter().map(|&o| lt(o)).collect(),
            lanes: geo.lanes,
            tree: geo.tree.clone(),
            kept: geo.kept.clone(),
            u,
            row: "%i".into(),
            lane: "%l".into(),
            lane_base: "0".into(),
            rloc: None,
            seg: segments(&u.members),
            cur_seg: 0,
            xrow: geo.xrow.clone(),
            part: geo.part,
            panel: geo.panel,
            free: self.order == crate::Order::Free,
            fma: self.contract && self.gpu_arms.pairs,
            shfl: super::super::gpu::host_rungs(self.profile, self.contract, self.gpu_arms)
                .shfl_tree
                && self.profile.gpu().is_some_and(|g| g.warp == 32),
            name: name.clone(),
            subst: Vec::new(),
            alphas: Vec::new(),
            parts: vec![None; u.members.len()],
        };
        // Each member's callee, argument type, result type and buffer types.
        let mut y = 0;
        for mb in &u.members {
            g.ys.push(mb.store.then(|| {
                y += 1;
                y - 1
            }));
            // flash-attention `stream`: a tile's epilogue body, called with
            // its fold's value (a per-row member through a one-cell buffer).
            if let (UKind::Tile { site }, Some(_)) = (&mb.kind, &u.stream)
                && let Some(post) = &site.post
            {
                let morph = self.ir.morphism(mb.m).expect("member");
                let Operation::Map { body, captures } = morph.op else {
                    unreachable!("a tile site is a map")
                };
                let slots: Vec<u32> = post
                    .reads
                    .iter()
                    .zip(&mb.srcs[2..])
                    .filter(|(_, x)| matches!(x, USrc::Member { .. }))
                    .map(|(&s, _)| s)
                    .collect();
                let (callee, text) = self.unit_post_body(body, post.fold, &slots);
                if !fns.iter().any(|(n, _)| *n == callee) {
                    fns.push((callee.clone(), text));
                }
                for f in super::super::resident::nested_bodies(self.ir, body) {
                    let (ln, lt) = self.unit_body(f, &[]);
                    if !fns.iter().any(|(n, _)| *n == ln) {
                        fns.push((ln, lt));
                    }
                }
                let in_ty = self.obj_ty(self.ir.func(body).expect("body").input);
                let arg = lower_body_input_ty(&in_ty, captures).expect("a site's input lowers");
                // An array's element type; a captured number's own
                // (`unit-uniform`: a loop counter the fold's operands take).
                let field_ty: Vec<String> = (0..=captures)
                    .map(|s| {
                        let o = self.pair_source(morph.source, s).expect("slot");
                        match self.obj_ty(o) {
                            Ty::Array { .. } => lt(o),
                            t => lower_ty(&t).expect("a captured number lowers"),
                        }
                    })
                    .collect();
                g.calls.push(Some(CallInfo {
                    callee,
                    arg,
                    ret: lt(mb.out),
                    field_ty,
                    captures,
                    slots,
                    laws: Vec::new(),
                }));
                continue;
            }
            let UKind::Cell { body, captures } = mb.kind else {
                g.calls.push(None);
                continue;
            };
            let morph = self.ir.morphism(mb.m).expect("member");
            let slots: Vec<u32> = (0..captures)
                .filter(|&s| matches!(mb.srcs[s as usize], USrc::Member { .. } | USrc::Law(_)))
                .collect();
            let (callee, text) = self.unit_body(body, &slots);
            if !fns.iter().any(|(n, _)| *n == callee) {
                fns.push((callee.clone(), text));
            }
            let laws = mb.srcs.iter().filter_map(|s| match *s {
                USrc::Law(f) => Some(f),
                _ => None,
            });
            // The laws, then every body nested in the member's or a law's (a
            // natural form's fold inside a cell), as the cell kernels carry
            // them (`resident::nested_bodies`).
            let laws: Vec<FuncId> = laws.collect();
            let nested = std::iter::once(body)
                .chain(laws.iter().copied())
                .flat_map(|b| super::super::resident::nested_bodies(self.ir, b));
            for f in laws.iter().copied().chain(nested) {
                let (ln, lt) = self.unit_body(f, &[]);
                if !fns.iter().any(|(n, _)| *n == ln) {
                    fns.push((ln, lt));
                }
            }
            let in_ty = self.obj_ty(self.ir.func(body).expect("body").input);
            let arg = lower_body_input_ty(&in_ty, captures).expect("a member's input lowers");
            let field_ty: Vec<String> = (0..=captures)
                .map(|s| {
                    let arr = if captures == 0 {
                        morph.source
                    } else {
                        self.pair_source(morph.source, s).expect("slot")
                    };
                    lt(arr)
                })
                .collect();
            let laws: Vec<(u32, String)> = mb
                .srcs
                .iter()
                .enumerate()
                .filter_map(|(s, x)| match *x {
                    USrc::Law(f) => Some((s as u32, self.fnames[f].clone())),
                    _ => None,
                })
                .collect();
            g.calls.push(Some(CallInfo {
                callee,
                arg,
                ret: lt(mb.out),
                field_ty,
                captures,
                slots,
                laws,
            }));
        }
        // Parameters: the stored values, then the operands.
        let mut params: Vec<String> = Vec::new();
        for (k, mb) in u.members.iter().enumerate() {
            if let Some(j) = g.ys[k] {
                params.push(format!("ptr addrspace(1) noalias %y{j}"));
                let _ = mb;
            }
        }
        for i in 0..u.gmem.len() {
            params.push(format!("ptr addrspace(1) readonly %x{i}"));
            g.entry.push_str(&format!(
                "  %xg{i} = addrspacecast ptr addrspace(1) %x{i} to ptr\n"
            ));
        }
        // kernel-merging `zero-fold`: later sites' outputs this unit zeroes.
        let zero = self
            .resident
            .as_ref()
            .and_then(|p| p.zero_by.get(u.members[0].m))
            .map_or(0, Vec::len);
        params.extend(
            super::super::zero::zero_params(zero)
                .into_iter()
                .skip(usize::from(zero == 0)),
        );
        // mask-frontend `stream-numbers`: each epilogue number, as its own
        // type in the entry block.
        for (k, s) in stream_numbers(u) {
            params.push(format!("i64 %u{k}_{s}"));
            let ty = g.calls[k].as_ref().expect("an epilogue call").field_ty[s as usize].clone();
            g.entry.push_str(&gpu::uniform_from_i64(
                &format!("%uv{k}_{s}"),
                &format!("%u{k}_{s}"),
                &ty,
            ));
        }
        // long-prefill `slice-launches`: the phase this launch runs.
        if u.slices.as_ref().is_some_and(|sl| sl.launches) {
            params.push("i64 %phase".to_string());
        }
        for (k, c) in g.calls.iter().enumerate() {
            if let Some(c) = c {
                for &s in &c.slots {
                    g.entry.push_str(&format!(
                        "  %b{k}_{s} = alloca {}\n",
                        c.field_ty[s as usize]
                    ));
                }
            }
        }
        // The phases: one row (or one thread's row) as before; a row-block
        // unit's tiles over all its rows, the members between them row by row.
        match &u.block {
            None if u.stream.is_some() => {
                g.stream_body(u.stream.as_ref().expect("streamed"));
            }
            None if u.slices.is_some() => {
                g.sliced(u.slices.as_ref().expect("sliced"));
            }
            None => {
                for k in 0..u.members.len() {
                    g.phase(k);
                }
            }
            Some(blk) => {
                let mut k = 0;
                while k < u.members.len() {
                    let s = g.seg[k];
                    let end = (k..u.members.len())
                        .find(|&j| g.seg[j] != s)
                        .unwrap_or(u.members.len());
                    if matches!(u.members[k].kind, UKind::Tile { .. }) {
                        g.cur_seg = s;
                        g.tile_phase(k, blk.b);
                    } else {
                        g.rows_loop(s, blk.b, k..end);
                    }
                    k = end;
                }
            }
        }
        // The intent line (design §7): what the plan keeps, stores and reads.
        let label = |o: ObjectId| self.object_label(o);
        let kind = |mb: &UMember| match mb.kind {
            UKind::Fold { .. } => "rowfold",
            UKind::Cell { .. } => "cell",
            UKind::Tile { .. } => "tile",
        };
        let members: Vec<String> = u
            .members
            .iter()
            .map(|mb| {
                let halo = mb
                    .srcs
                    .iter()
                    .any(|s| matches!(s, USrc::Member { row: false, .. }));
                format!(
                    "{}:{}:1x{}{}",
                    label(mb.out),
                    kind(mb),
                    mb.w,
                    if halo { "+halo" } else { "" }
                )
            })
            .collect();
        let keep: Vec<String> = u
            .members
            .iter()
            .filter(|mb| !mb.store)
            .map(|mb| {
                let e = crate::profile::elem_bytes(super::super::resident::array_elem(
                    &self.obj_ty(mb.out),
                ));
                format!("{}:reg:{}", label(mb.out), if mb.per_row() { e } else { 0 })
            })
            .collect();
        let store: Vec<String> = u
            .members
            .iter()
            .filter(|mb| mb.store)
            .map(|mb| label(mb.out))
            .collect();
        let gm: Vec<String> = u.gmem.iter().map(|&o| label(o)).collect();
        let laws: Vec<String> = u
            .members
            .iter()
            .flat_map(|mb| mb.srcs.iter())
            .filter_map(|s| match *s {
                USrc::Law(f) => Some(self.fnames[f].clone()),
                _ => None,
            })
            .collect();
        let root = u
            .members
            .iter()
            .find(|mb| matches!(mb.kind, UKind::Fold { .. }))
            .unwrap_or(&u.members[0]);
        let w = u.members.iter().map(|mb| mb.w).max().unwrap_or(1);
        let block = match &u.block {
            None if u.stream.is_some() => {
                let st = u.stream.as_ref().expect("streamed");
                format!(
                    " stream={} axis={} windows={} skip={} dissolved={} state={} ns_stream={} ns_split={}",
                    st.label(),
                    st.group.axis,
                    st.nb(),
                    match &st.group.guard {
                        None => "none",
                        Some(g) if g.skip => "on",
                        Some(g) => g.why,
                    },
                    st.group
                        .dissolved
                        .iter()
                        .map(|&k| label(u.members[k].out))
                        .collect::<Vec<_>>()
                        .join(","),
                    st.group.state_bytes,
                    st.ns.0,
                    st.ns.1
                )
            }
            None => match &u.slices {
                None => String::new(),
                Some(sl) => format!(
                    " slices={} phases={}{} laws={} ns_sliced={} ns_row={}",
                    sl.s,
                    sl.phases,
                    if sl.launches { " launches=1" } else { "" },
                    sl.laws
                        .iter()
                        .flatten()
                        .map(|(_, _, l)| match l {
                            DepLaw::Factor { div: false } => "mul",
                            DepLaw::Factor { div: true } => "div",
                            DepLaw::Shift { sub: false } => "add",
                            DepLaw::Shift { sub: true } => "sub",
                            DepLaw::Rescale => "rescale",
                            DepLaw::Phase => "phase",
                        })
                        .collect::<Vec<_>>()
                        .join(","),
                    sl.ns.0,
                    sl.ns.1
                ),
            },
            Some(blk) => {
                // kernel-merging `row-stage`: each staged tile's panels.
                let stage: Vec<String> = blk
                    .stage
                    .iter()
                    .enumerate()
                    .filter_map(|(k, st)| st.map(|st| format!("T{k}:{}", st.label())))
                    .collect();
                format!(
                    " rows_per_unit={} tiles={} colors={} ns_fused={} ns_split={}{}",
                    blk.b,
                    u.members
                        .iter()
                        .filter(|mb| matches!(mb.kind, UKind::Tile { .. }))
                        .count(),
                    blk.color.iter().flatten().max().map_or(0, |c| c + 1),
                    blk.ns.0,
                    blk.ns.1,
                    match stage.is_empty() {
                        true => String::new(),
                        false => format!(" stage={}", stage.join(";")),
                    }
                )
            }
        };
        let mut text = format!(
            "; mapal.flow kernel={name} unit=1x{w} rows={} root={} members={} keep={} store={} gmem={} law={} smem={} lanes={} tree={} onchip={}{block}\n",
            u.rows,
            label(root.out),
            members.join(";"),
            keep.join(";"),
            store.join(";"),
            gm.join(";"),
            laws.join(";"),
            geo.smem,
            geo.lanes,
            geo.tree.iter().flatten().count(),
            geo.kept.iter().flatten().count(),
        );
        // sweep-attention `unit-uniform`: the operands the launch moves by a
        // captured number (intent.py `unit-uniform`); absent where none is.
        let moved: Vec<String> = u
            .gmem
            .iter()
            .zip(&u.gmove)
            .filter(|(_, mv)| mv.is_some())
            .map(|(&o, _)| label(o))
            .collect();
        if !moved.is_empty() {
            text.insert_str(text.len() - 1, &format!(" moved={}", moved.join(";")));
        }
        text += &format!(
            "; mapal.plan kernel={name} kind=flow rows={} gx={blocks} block={threads} members={} rungs_on=flow{}{}\n",
            u.rows,
            u.members.len(),
            // sweep-attention-2: a stream on the warp matrix product.
            if u.stream.as_ref().is_some_and(|st| st.mma.is_some()) {
                ",row-block,stream,stream-mma"
            } else if u.stream.is_some() {
                ",row-block,stream"
            } else if u.block.is_some() {
                ",row-block"
            } else if u.slices.is_some() {
                ",row-slices"
            } else {
                ""
            },
            // sweep-attention: a unit whose launch moves an operand.
            if u.gmove.iter().any(Option::is_some) {
                ",unit-uniform"
            } else {
                ""
            }
        );
        // flash-attention `split-kv`: each fold's and consumer's slice
        // partials, and the row blocks' tickets.
        if let Some(st) = u.stream.as_ref().filter(|st| st.s > 1) {
            for (k, mb) in u.members.iter().enumerate() {
                let n = match (&st.group.roles[k], &mb.kind) {
                    (mapal_ir::StreamRole::Fold { .. }, _) => u.rows * st.s,
                    (mapal_ir::StreamRole::Consume { .. }, UKind::Tile { site }) => {
                        u.rows * st.s * site.c
                    }
                    _ => continue,
                };
                text += &format!(
                    "@{name}.ws{k} = internal addrspace(1) global [{n} x float] zeroinitializer, align 16\n"
                );
            }
            text += &format!(
                "@{name}.cnt = internal addrspace(1) global [{} x i32] zeroinitializer, align 4\n",
                u.rows / st.b
            );
        }
        // long-rows `row-slices`: each fold's partials (a row's `s`) and the
        // grid barrier's counter.
        if let Some(sl) = &u.slices {
            for (k, mb) in u.members.iter().enumerate() {
                if let UKind::Fold { site, .. } = &mb.kind {
                    let t = lower_ty(&site.elem).expect("float elem lowers");
                    text += &format!(
                        "@{name}.ws{k} = internal addrspace(1) global [{} x {t}] zeroinitializer, align 16\n",
                        u.rows * sl.s
                    );
                }
            }
            if sl.coop() {
                text += &format!("@{name}.gsync = internal addrspace(1) global i32 0, align 4\n");
            }
        }
        // sweep-attention: a streamed unit's kernel holds ptxas to the
        // blocks an SM its geometry assumed (`min-blocks`, as a tile's).
        let min_blocks = match (&u.stream, self.profile.gpu()) {
            (Some(st), Some(gpu))
                if gpu::host_rungs(self.profile, self.contract, self.gpu_arms).min_blocks =>
            {
                let n = st.per_sm(gpu, &u.members, st.layout(&u.members).total);
                format!(" \"nvvm.minctasm\"=\"{n}\"")
            }
            _ => String::new(),
        };
        text += &format!(
            "define ptx_kernel void @{name}({}) \"nvvm.maxntid\"=\"{threads}\"{min_blocks} {{\n\
             entry:\n{}\
             \x20 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             \x20 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             \x20 %t64 = zext i32 %tid to i64\n\
             \x20 %b64 = zext i32 %bid to i64\n\
             {}\
             \x20 %in = icmp ult i64 %i, {}\n\
             \x20 br i1 %in, label %body, label %done\n\
             body:\n{}\
             \x20 br label %done\n\
             done:\n{}\
             \x20 ret void\n\
             }}\n",
            params.join(", "),
            g.entry,
            // One row a block (`lanes` threads), or one thread a row; a
            // row-block unit's first row.
            if let Some(blk) = &u.block {
                format!("  %i = mul i64 %b64, {}\n  %l = add i64 %t64, 0\n", blk.b)
            } else if let Some(st) = &u.stream {
                match st.skip() {
                    // R5: a masked unit's blocks in reverse row order — a
                    // causal mask's heaviest row blocks (the last rows of
                    // each batch) launch first, the lightest fill the tail.
                    Some(_) => format!(
                        "  %bre = sub i64 {}, %b64\n  %i = mul i64 %bre, {}\n  %l = add i64 %t64, 0\n",
                        u.rows / st.b - 1,
                        st.b
                    ),
                    // R3: block `b64` is slice `%bz` of row block `%bq`.
                    None if st.s > 1 => format!(
                        "  %bq = udiv i64 %b64, {s}\n  %bz = urem i64 %b64, {s}\n  %i = mul i64 %bq, {}\n  %l = add i64 %t64, 0\n",
                        st.b,
                        s = st.s
                    ),
                    None => format!("  %i = mul i64 %b64, {}\n  %l = add i64 %t64, 0\n", st.b),
                }
            } else if u.slices.is_some() {
                // Every block reaches every grid barrier: the items loop
                // decides what a block does.
                "  %i = add i64 0, 0\n  %l = add i64 %t64, 0\n".to_string()
            } else if geo.lanes > 1 {
                "  %i = add i64 %b64, 0\n  %l = add i64 %t64, 0\n".to_string()
            } else {
                format!(
                    "  %row0 = mul i64 %b64, {threads}\n  %i = add i64 %row0, %t64\n  %l = add i64 0, 0\n"
                )
            },
            u.rows,
            g.out,
            super::super::zero::zero_tail(zero, threads),
        );
        (fns, text)
    }
}
