//! The streamed kernel body: the window loop, the split combine and the final columns.
use super::*;

impl UnitGen<'_> {
    /// **The streamed kernel body** (module doc). `%i` is the block's first
    /// row, `%l` the thread.
    pub(crate) fn stream_body(&mut self, st: &Stream) {
        let u = self.u;
        let n = u.members.len();
        let roles = st.group.roles.clone();
        let lay = st.layout(&u.members);
        let (kb, rt, g, threads) = (st.kb, st.rt, st.g, st.threads);
        let ct = st.ct();
        let w = st.group.axis;
        let nb = st.nb();
        let tails = !w.is_multiple_of(kb);
        let cp = st.cp;
        // The thread's lane in its row group, and its rows.
        let x = self.op("urem", "%l", g);
        // `stream-mma`: the accumulator's layout — warp `w`'s rows `16w +
        // l%32÷4` and `+ 8`, the quad's lane `x` (`gq` the fragment's `n`).
        let (rloc, gq): (Vec<String>, String) = match st.mma {
            Some(_) => {
                let w = self.op("udiv", "%l", 32);
                let lw = self.op("urem", "%l", 32);
                let gq = self.op("udiv", &lw, 4);
                let r0 = self.op("mul", &w, MMA_M);
                let r0 = self.op("add", &r0, &gq);
                let r1 = self.op("add", &r0, 8);
                (vec![r0, r1], gq)
            }
            None => {
                let rg = self.op("udiv", "%l", g);
                let rl0 = self.op("mul", &rg, rt);
                (
                    (0..rt).map(|y| self.op("add", &rl0, y)).collect(),
                    String::new(),
                )
            }
        };
        // mask-frontend: each of the thread's rows as the group numbers it
        // (the walk order's).
        let grow: Vec<String> = rloc
            .iter()
            .map(|r| {
                let p = self.op("add", "%i", r);
                self.walk_row(st, &p)
            })
            .collect();
        // Which folds a cell reads as a rescale's shift: their window
        // extreme is combined across the row group every window.
        let shift: Vec<bool> = (0..n)
            .map(|v| {
                roles.iter().any(|r| match r {
                    StreamRole::Cell { subst } => subst
                        .iter()
                        .any(|&(_, x, law)| x == v && law == DepLaw::Rescale),
                    _ => false,
                })
            })
            .collect();
        // State: each producer's window, each consumer's accumulators, each
        // fold's state, as allocas the optimizer keeps in registers.
        for (k, mb) in u.members.iter().enumerate() {
            match (&roles[k], &mb.kind) {
                (StreamRole::Produce, _) => {
                    for c in 0..rt * ct {
                        self.entry
                            .push_str(&format!("  %S{k}_{c} = alloca float\n"));
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    for c in 0..rt * st.oc(site.c).0 {
                        self.entry
                            .push_str(&format!("  %O{k}_{c} = alloca float\n"));
                    }
                }
                (StreamRole::Fold { .. }, _) => {
                    for y in 0..rt {
                        self.entry
                            .push_str(&format!("  %F{k}_{y} = alloca float\n"));
                    }
                }
                _ => {}
            }
        }
        // Initial states: the consumers' and folds' step identities.
        for (k, mb) in u.members.iter().enumerate() {
            match (&roles[k], &mb.kind) {
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    for c in 0..rt * st.oc(site.c).0 {
                        self.line(format!("store float -0.0, ptr %O{k}_{c}"));
                    }
                }
                (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                    let id = step_identity(&site.op);
                    for y in 0..rt {
                        self.line(format!("store float {id}, ptr %F{k}_{y}"));
                    }
                }
                _ => {}
            }
        }
        // flash-attention F5/F6: the windows the block's rows see, `[tlo,
        // tlo + band)` (rotated; a block that sees none walks them all).
        let skip = st.skip();
        match &skip {
            Some(gd) => self.visible_band(st, gd),
            // R3: this block's slice of the windows.
            None if st.s > 1 => {
                self.line(format!("%tlo = mul i64 %bz, {}", nb / st.s));
                self.line(format!("%band = add i64 0, {}", nb / st.s));
            }
            None => {
                self.line("%tlo = add i64 0, 0");
                self.line(format!("%band = add i64 0, {nb}"));
            }
        }
        self.line(format!("%j00 = mul i64 %tlo, {kb}"));
        // The prologue: the producers' row panels and first windows, then
        // the consumers' first windows. `stream-mma`: each producer's row
        // operand into its `a` fragments, once.
        let mut afrag: Vec<Vec<[String; 4]>> = vec![Vec::new(); n];
        for k in 0..n {
            if roles[k] == StreamRole::Produce {
                match st.mma {
                    Some(_) => afrag[k] = self.mma_rows(st, k, &grow, &x),
                    None => self.stage_panel(st, &lay, k),
                }
                self.stage_window(st, &lay, k, "%j00", true);
            }
        }
        self.commit(cp);
        for k in 0..n {
            if matches!(roles[k], StreamRole::Consume { .. }) {
                self.stage_window(st, &lay, k, "%j00", false);
            }
        }
        self.commit(cp);
        // The window loop.
        let pre = self.cur.clone();
        self.line("br label %W_h");
        self.label("W_h");
        self.line(format!("%wt = phi i64 [0, %{pre}], [%wtn, %W_n]"));
        // Window `(tlo + wt) mod nb`; the next one's first key.
        self.line("%wtr = add i64 %tlo, %wt");
        self.line(format!("%wtw = urem i64 %wtr, {nb}"));
        self.line(format!("%j0 = mul i64 %wtw, {kb}"));
        self.line("%wtn = add i64 %wt, 1");
        self.line("%wnr = add i64 %tlo, %wtn");
        self.line(format!("%wnw = urem i64 %wnr, {nb}"));
        self.line(format!("%jn = mul i64 %wnw, {kb}"));
        // The windows this block walks: all `nb`, or its slice's.
        let walk = if st.s > 1 { nb / st.s } else { nb };
        self.line(format!("%wmore = icmp ult i64 %wtn, {walk}"));
        if !cp {
            // Synchronous loads: this window's operands now.
            let pre_k = (0..n).any(|k| roles[k] == StreamRole::Produce);
            if pre_k {
                let c = self.tmp();
                self.line(format!("{c} = icmp ugt i64 %wt, 0"));
                self.cond("Wk", &c, |gg| {
                    for k in 0..n {
                        if roles[k] == StreamRole::Produce {
                            gg.stage_window(st, &lay, k, "%j0", true);
                        }
                    }
                });
            }
        }
        self.wait(cp, 1);
        self.barrier();
        // Each cell's column and validity (`stream-mma`: the accumulator's,
        // `8·(z÷2) + 2x + z%2`).
        let jl: Vec<String> = match st.mma {
            Some(_) => {
                let x2 = self.op("mul", &x, 2);
                (0..ct)
                    .map(|z| self.op("add", &x2, MMA_N * (z / 2) + z % 2))
                    .collect()
            }
            None => (0..ct).map(|z| self.op("add", &x, g * z)).collect(),
        };
        let jg: Vec<String> = jl.iter().map(|j| self.op("add", "%j0", j)).collect();
        let valid: Vec<Option<String>> = jg
            .iter()
            .map(|j| {
                tails.then(|| {
                    let c = self.tmp();
                    self.line(format!("{c} = icmp ult i64 {j}, {w}"));
                    c
                })
            })
            .collect();
        // A cell index inside the axis (a row's last cell past its end: the
        // cell's value is dropped, its reads stay in bounds).
        let jc: Vec<String> = jg
            .iter()
            .zip(&valid)
            .map(|(j, v)| match v {
                Some(v) => {
                    let s = self.tmp();
                    self.line(format!("{s} = select i1 {v}, i64 {j}, i64 {}", w - 1));
                    s
                }
                None => j.clone(),
            })
            .collect();
        // The producers' windows.
        let mut vals: Vec<Vals> = vec![Vec::new(); n];
        for k in 0..n {
            if roles[k] != StreamRole::Produce {
                continue;
            }
            vals[k] = match st.mma {
                Some(_) => self.produce_mma(st, &lay, k, &afrag[k], &x, &gq),
                None => self.produce(st, &lay, k, &rloc, &jl),
            };
            // `tile-post`: each cell through the site's body after its fold
            // (a scale, a mask), at its own cell index.
            let UKind::Tile { site } = &u.members[k].kind else {
                unreachable!("a producer is a tile")
            };
            if site.post.is_some() {
                let out = Addr::of(site, Operand::Out);
                for y in 0..rt as usize {
                    let row = self.member_row(st, k, &grow[y]);
                    for z in 0..ct as usize {
                        let t = self.addr_at(&out, &row, &jc[z], "0");
                        let acc = vals[k][y * ct as usize + z].clone();
                        vals[k][y * ct as usize + z] = self.post_call(k, &t, &acc, &[]);
                    }
                }
            }
        }
        if cp && nb > 1 {
            // Every thread is past its reads of this window's panels: the
            // next window lands in them while the rest of this one runs.
            self.barrier();
            self.cond("Wk", "%wmore", |gg| {
                for k in 0..n {
                    if roles[k] == StreamRole::Produce {
                        gg.stage_window(st, &lay, k, "%jn", true);
                    }
                }
            });
        }
        // One group a step whatever it holds: the waits count them.
        self.commit(cp);
        // Per row: each shift fold's previous and new extreme this window.
        let mut ext: Vec<Vec<(String, String)>> = vec![Vec::new(); n];
        let mut first_consume = true;
        let last_consume = (0..n)
            .rev()
            .find(|&k| matches!(roles[k], StreamRole::Consume { .. }));
        for k in 0..n {
            let mb = &u.members[k];
            match (&roles[k], &mb.kind) {
                (StreamRole::Produce, UKind::Tile { site }) => {
                    if mb.store {
                        let out = Addr::of(site, Operand::Out);
                        for y in 0..rt as usize {
                            let row = self.member_row(st, k, &grow[y]);
                            for z in 0..ct as usize {
                                let t = self.addr_at(&out, &row, &jc[z], "0");
                                let v = vals[k][y * ct as usize + z].clone();
                                match &valid[z] {
                                    Some(c) => {
                                        let c = c.clone();
                                        self.cond(&format!("Ps{k}_{y}_{z}"), &c, |gg| {
                                            gg.store(k, &v, &t)
                                        });
                                    }
                                    None => self.store(k, &v, &t),
                                }
                            }
                        }
                    }
                }
                (StreamRole::Cell { subst }, _) => {
                    let subst = subst.clone();
                    let mut out = Vec::new();
                    for y in 0..rt as usize {
                        for z in 0..ct as usize {
                            let r = self.op("mul", &grow[y], w);
                            let t = self.op("add", &r, &jc[z]);
                            for (s, src) in mb.srcs.iter().enumerate() {
                                let USrc::Member { k: p, .. } = src else {
                                    continue;
                                };
                                let lit = match subst.iter().find(|x| x.0 == s as u32) {
                                    Some(&(_, v, DepLaw::Rescale)) => {
                                        let m = ext[v][y].1.clone();
                                        let id = match &u.members[v].kind {
                                            UKind::Fold { site, .. } => step_identity(&site.op),
                                            _ => unreachable!("a shift is a fold"),
                                        };
                                        let c = self.tmp();
                                        self.line(format!("{c} = fcmp oeq float {m}, {id}"));
                                        let sv = self.tmp();
                                        self.line(format!(
                                            "{sv} = select i1 {c}, float 0.0, float {m}"
                                        ));
                                        sv
                                    }
                                    Some(&(_, _, DepLaw::Factor { .. })) => "1.0".into(),
                                    Some(&(_, _, DepLaw::Shift { sub: false })) => "-0.0".into(),
                                    Some(&(_, _, _)) => "0.0".into(),
                                    None => vals[*p][y * ct as usize + z].clone(),
                                };
                                self.subst.push((k, s as u32, lit));
                            }
                            let v = self.cell(k, &t);
                            self.subst.clear();
                            out.push(v);
                        }
                    }
                    vals[k] = out;
                }
                (StreamRole::Fold { laws }, UKind::Fold { site, .. }) => {
                    let laws = laws.clone();
                    let (op, id) = (site.op, step_identity(&site.op));
                    let USrc::Member { k: p, .. } = mb.srcs[0] else {
                        unreachable!("an axis fold reads a member")
                    };
                    for y in 0..rt as usize {
                        let mut part = id.clone();
                        for z in 0..ct as usize {
                            let mut v = vals[p][y * ct as usize + z].clone();
                            if let Some(c) = &valid[z] {
                                let s = self.tmp();
                                self.line(format!("{s} = select i1 {c}, float {v}, float {id}"));
                                v = s;
                            }
                            let o = self.tmp();
                            self.step(&op, "float", &part, &v, &o);
                            part = o;
                        }
                        let fy = format!("%F{k}_{y}");
                        let old = self.tmp();
                        self.line(format!("{old} = load float, ptr {fy}"));
                        if shift[k] {
                            let tot = self.butterfly(&part, g, op);
                            let new = self.tmp();
                            self.step(&op, "float", &old, &tot, &new);
                            self.line(format!("store float {new}, ptr {fy}"));
                            ext[k].push((old, new));
                        } else {
                            let mut s = old;
                            for l in &laws {
                                if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                                    let (mo, mn) = ext[l.value][y].clone();
                                    let a = self.alpha(arg, &mo, &mn);
                                    let z = self.tmp();
                                    self.line(format!("{z} = fmul float {s}, {a}"));
                                    s = z;
                                }
                            }
                            let new = self.tmp();
                            self.step(&op, "float", &s, &part, &new);
                            self.line(format!("store float {new}, ptr {fy}"));
                        }
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let laws = laws.clone();
                    let site = site.clone();
                    let USrc::Member { k: p, .. } = mb.srcs[0] else {
                        unreachable!("a consumer reads its row operand from a member")
                    };
                    // Its row operand's window into shared memory
                    // (`stream-mma`: the cells as they stand, its `a`
                    // fragments).
                    let mut pv: Vec<String> = Vec::new();
                    for y in 0..rt as usize {
                        let ra = lay.a[k].map(|(aoff, ap)| (aoff, self.op("mul", &rloc[y], ap)));
                        for z in 0..ct as usize {
                            let mut v = vals[p][y * ct as usize + z].clone();
                            if let Some(c) = &valid[z] {
                                let s = self.tmp();
                                self.line(format!("{s} = select i1 {c}, float {v}, float 0.0"));
                                v = s;
                            }
                            match &ra {
                                Some((aoff, ra)) => {
                                    let at = self.op("add", ra, &jl[z]);
                                    let q = self.smem(*aoff, "float", &at);
                                    self.line(format!("store float {v}, ptr addrspace(3) {q}"));
                                }
                                None => pv.push(v),
                            }
                        }
                    }
                    if first_consume {
                        if !cp {
                            for j in 0..n {
                                if matches!(roles[j], StreamRole::Consume { .. }) {
                                    self.stage_window(st, &lay, j, "%j0", false);
                                }
                            }
                        }
                        self.wait(cp, 1);
                        first_consume = false;
                    }
                    self.barrier();
                    // The window's rescale, then its products.
                    let (oc, _) = st.oc(site.c);
                    for y in 0..rt as usize {
                        let mut alpha: Option<String> = None;
                        for l in &laws {
                            if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                                let (mo, mn) = ext[l.value][y].clone();
                                let a = self.alpha(arg, &mo, &mn);
                                alpha = Some(match alpha {
                                    None => a,
                                    Some(x) => {
                                        let z = self.tmp();
                                        self.line(format!("{z} = fmul float {x}, {a}"));
                                        z
                                    }
                                });
                            }
                        }
                        if let Some(a) = alpha {
                            for q in 0..oc {
                                let o = format!("%O{k}_{}", y as u64 * oc + q);
                                let v = self.tmp();
                                self.line(format!("{v} = load float, ptr {o}"));
                                let z = self.tmp();
                                self.line(format!("{z} = fmul float {v}, {a}"));
                                self.line(format!("store float {z}, ptr {o}"));
                            }
                        }
                    }
                    match st.mma {
                        Some(_) => self.consume_mma(st, &lay, k, &site, &pv, &x, &gq),
                        None => self.consume(st, &lay, k, &site, &rloc, &x),
                    }
                    if Some(k) == last_consume {
                        self.barrier();
                        if cp && nb > 1 {
                            self.cond("Wv", "%wmore", |gg| {
                                for j in 0..n {
                                    if matches!(roles[j], StreamRole::Consume { .. }) {
                                        gg.stage_window(st, &lay, j, "%jn", false);
                                    }
                                }
                            });
                        }
                        self.commit(cp);
                    }
                }
                _ => unreachable!("a streamed member has its role"),
            }
        }
        self.line("br label %W_l");
        self.label("W_l");
        match &skip {
            // F6: after the visible windows, the skipped ones' terms are each
            // fold's identity at the final values — or they run.
            Some(gd) => {
                self.line("%atb = icmp eq i64 %wtn, %band");
                self.line("br i1 %atb, label %W_c, label %W_n");
                self.label("W_c");
                let pass = self.absorbed(st, &lay, gd, &x, &grow, &shift);
                self.line(format!("br i1 {pass}, label %W_e, label %W_n"));
            }
            None => self.line("br label %W_n"),
        }
        self.label("W_n");
        self.line("br i1 %wmore, label %W_h, label %W_e");
        self.label("W_e");
        // A skipped window's prefetch lands before the block leaves.
        self.wait(cp, 0);
        // R3: a slice's partial states to the workspace; the row block's last
        // block reads every slice's back, combined into its own states.
        if st.s > 1 {
            self.split_combine(st, &lay, &x, &grow, &shift);
        }
        // After the last window: each fold's lanes combined, its laws, its
        // seed; each consumer's laws and seed, stored.
        let fin = self.finals(st, &shift);
        let lane0 = self.tmp();
        self.line(format!("{lane0} = icmp eq i64 {x}, 0"));
        for k in 0..n {
            let mb = &u.members[k];
            match (&roles[k], &mb.kind) {
                (StreamRole::Fold { .. }, UKind::Fold { .. }) => {
                    for y in 0..rt as usize {
                        if mb.store {
                            let row = grow[y].clone();
                            let r2 = fin[k][y].clone();
                            self.cond(&format!("Fs{k}_{y}"), &lane0, |gg| gg.store(k, &r2, &row));
                        }
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let laws = laws.clone();
                    let op = mapal_ir::RowFoldOp::Arith {
                        op: Operation::Add,
                        acc_first: site.add_acc_first,
                    };
                    let seed = const_literal(&site.seed);
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt as usize {
                        for q in 0..oc {
                            let o = self.tmp();
                            self.line(format!("{o} = load float, ptr %O{k}_{}", y as u64 * oc + q));
                            let v = self.laws_after(&laws, &fin, y, &o, k);
                            let res = self.tmp();
                            self.step(&op, "float", &seed, &v, &res);
                            let col = match (st.mma, contig) {
                                (Some(_), _) => {
                                    let x2 = self.op("mul", &x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                (None, true) => self.chunk_col(&x, q, g),
                                (None, false) => self.op("add", &x, g * q),
                            };
                            // Its own row's output cell; the epilogue there.
                            let row = self.member_row(st, k, &grow[y]);
                            let out = Addr::of(site, Operand::Out);
                            let at = self.addr_at(&out, &row, &col, "0");
                            let res = match &site.post {
                                Some(post) => {
                                    let per_row: Vec<(u32, String)> = post
                                        .reads
                                        .iter()
                                        .zip(&mb.srcs[2..])
                                        .filter_map(|(&slot, x)| match x {
                                            USrc::Member { k: v, .. } => {
                                                Some((slot, fin[*v][y].clone()))
                                            }
                                            _ => None,
                                        })
                                        .collect();
                                    self.post_call(k, &at, &res, &per_row)
                                }
                                None => res,
                            };
                            if contig || site.c.is_multiple_of(g) || st.mma.is_some() {
                                self.store(k, &res, &at);
                            } else {
                                let c = self.tmp();
                                self.line(format!("{c} = icmp ult i64 {col}, {}", site.c));
                                self.cond(&format!("Os{k}_{y}_{q}"), &c, |gg| {
                                    gg.store(k, &res, &at)
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if st.s > 1 {
            self.line("br label %SPx");
            self.label("SPx");
        }
        let _ = threads;
    }

    /// R3 split-KV: the block's partial states — each fold's row value (its
    /// lanes combined), each consumer's columns — to the kernel's workspace
    /// (`@<kernel>.ws<k>`, slice `%bz` of row `r` at `r·s + z`); every thread
    /// fences, the block meets, its lead takes a ticket on the row block's
    /// counter (`@<kernel>.cnt`, reset by the last); a block that is not the
    /// row block's last leaves (`%SPx`). The last reads all `s` partials and
    /// combines them in slice order into its own states: a running extreme by
    /// its step, a sum by `+` with each slice's state rescaled from its own
    /// extreme to the combined one (`exp(a(m_z, m))`, ADR-0042 D7), a
    /// consumer's columns likewise — the windows' combine, across blocks.
    pub(super) fn split_combine(
        &mut self,
        st: &Stream,
        lay: &Layout,
        x: &str,
        grow: &[String],
        shift: &[bool],
    ) {
        let u = self.u;
        let n = u.members.len();
        let (sp, rt, g) = (st.s, st.rt as usize, st.g);
        let name = self.name.clone();
        let rows = u.rows;
        let lane0 = self.tmp();
        self.line(format!("{lane0} = icmp eq i64 {x}, 0"));
        let ws = |gg: &mut Self, k: usize, n_el: u64, idx: &str| {
            let p = gg.tmp();
            gg.line(format!(
                "{p} = getelementptr inbounds [{n_el} x float], ptr addrspace(1) @{name}.ws{k}, i64 0, i64 {idx}"
            ));
            p
        };
        // The partials out.
        for k in 0..n {
            match (&st.group.roles[k], &u.members[k].kind) {
                (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                    for y in 0..rt {
                        let s0 = self.tmp();
                        self.line(format!("{s0} = load float, ptr %F{k}_{y}"));
                        let v = match shift[k] {
                            true => s0,
                            false => self.butterfly(&s0, g, site.op),
                        };
                        let r = self.op("mul", &grow[y], sp);
                        let at = self.op("add", &r, "%bz");
                        let p = ws(self, k, rows * sp, &at);
                        self.cond(&format!("SPw{k}_{y}"), &lane0, |gg| {
                            gg.line(format!("store float {v}, ptr addrspace(1) {p}"))
                        });
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt {
                        let r = self.op("mul", &grow[y], sp);
                        let rz = self.op("add", &r, "%bz");
                        let base = self.op("mul", &rz, site.c);
                        for q in 0..oc {
                            let col = match st.mma {
                                Some(_) => {
                                    let x2 = self.op("mul", x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                None => self.split_col(x, q, contig, g, site.c),
                            };
                            let at = self.op("add", &base, &col);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr %O{k}_{}", y as u64 * oc + q));
                            let p = ws(self, k, rows * sp * site.c, &at);
                            self.line(format!("store float {v}, ptr addrspace(1) {p}"));
                        }
                    }
                }
                _ => {}
            }
        }
        // The ticket.
        self.line("fence syncscope(\"device\") release");
        let f = self.smem(lay.flag, "i32", "0");
        self.barrier();
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 %l, 0"));
        let rb = self.op("udiv", "%i", st.b);
        self.cond("SPt", &t0, |gg| {
            let c = gg.tmp();
            gg.line(format!(
                "{c} = getelementptr inbounds [{} x i32], ptr addrspace(1) @{name}.cnt, i64 0, i64 {rb}",
                rows / st.b
            ));
            let old = gg.tmp();
            gg.line(format!(
                "{old} = atomicrmw add ptr addrspace(1) {c}, i32 1 syncscope(\"device\") acq_rel, align 4"
            ));
            let last = gg.tmp();
            gg.line(format!("{last} = icmp eq i32 {old}, {}", sp - 1));
            let lz = gg.tmp();
            gg.line(format!("{lz} = zext i1 {last} to i32"));
            gg.line(format!("store i32 {lz}, ptr addrspace(3) {f}"));
            // The last resets the counter for the next launch.
            gg.cond("SPr", &last, |h| {
                h.line(format!("store atomic i32 0, ptr addrspace(1) {c} syncscope(\"device\") monotonic, align 4"))
            });
        });
        self.barrier();
        let fl = self.tmp();
        self.line(format!("{fl} = load i32, ptr addrspace(3) {f}"));
        let last = self.tmp();
        self.line(format!("{last} = icmp ne i32 {fl}, 0"));
        self.line(format!("br i1 {last}, label %SPc, label %SPx"));
        self.label("SPc");
        self.line("fence syncscope(\"device\") acquire");
        // Each slice's shift folds first (the rescales' targets).
        let mut ms: Vec<Vec<Vec<String>>> = vec![Vec::new(); n];
        let mut comb: Vec<Vec<String>> = vec![Vec::new(); n];
        for k in 0..n {
            let (StreamRole::Fold { .. }, UKind::Fold { site, .. }) =
                (&st.group.roles[k], &u.members[k].kind)
            else {
                continue;
            };
            let (op, id) = (site.op, step_identity(&site.op));
            for y in 0..rt {
                let r = self.op("mul", &grow[y], sp);
                let mut parts = Vec::new();
                let mut acc = id.clone();
                for z in 0..sp {
                    let at = self.op("add", &r, z);
                    let p = ws(self, k, rows * sp, &at);
                    let v = self.tmp();
                    self.line(format!("{v} = load float, ptr addrspace(1) {p}"));
                    if shift[k] {
                        let o = self.tmp();
                        self.step(&op, "float", &acc, &v, &o);
                        acc = o;
                    }
                    parts.push(v);
                }
                ms[k].push(parts);
                comb[k].push(acc);
            }
        }
        // The rescale of slice `z`'s state to the combined shift.
        let rescale = |gg: &mut Self, laws: &[mapal_ir::StreamLaw], y: usize, z: usize, v: &str| {
            let mut v = v.to_string();
            for l in laws {
                if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                    let e = gg.exp_arg(arg, &ms[l.value][y][z], &comb[l.value][y]);
                    let t = gg.tmp();
                    gg.line(format!("{t} = fmul float {v}, {e}"));
                    v = t;
                }
            }
            v
        };
        for k in 0..n {
            match (&st.group.roles[k], &u.members[k].kind) {
                (StreamRole::Fold { laws }, UKind::Fold { site, .. }) => {
                    let (op, id) = (site.op, step_identity(&site.op));
                    for y in 0..rt {
                        let tot = match shift[k] {
                            true => comb[k][y].clone(),
                            false => {
                                let mut acc = id.clone();
                                for z in 0..sp as usize {
                                    let v = rescale(self, laws, y, z, &ms[k][y][z]);
                                    let o = self.tmp();
                                    self.step(&op, "float", &acc, &v, &o);
                                    acc = o;
                                }
                                // Lane 0 holds it, the others the identity: the
                                // finals' butterfly is the total.
                                let o = self.tmp();
                                self.line(format!(
                                    "{o} = select i1 {lane0}, float {acc}, float {id}"
                                ));
                                o
                            }
                        };
                        self.line(format!("store float {tot}, ptr %F{k}_{y}"));
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt {
                        let r = self.op("mul", &grow[y], sp);
                        for q in 0..oc {
                            let col = match st.mma {
                                Some(_) => {
                                    let x2 = self.op("mul", x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                None => self.split_col(x, q, contig, g, site.c),
                            };
                            let mut acc = "-0.0".to_string();
                            for z in 0..sp as usize {
                                let rz = self.op("add", &r, z as u64);
                                let b = self.op("mul", &rz, site.c);
                                let at = self.op("add", &b, &col);
                                let p = ws(self, k, rows * sp * site.c, &at);
                                let v = self.tmp();
                                self.line(format!("{v} = load float, ptr addrspace(1) {p}"));
                                let v = rescale(self, laws, y, z, &v);
                                let o = self.tmp();
                                self.line(format!("{o} = fadd float {acc}, {v}"));
                                acc = o;
                            }
                            self.line(format!(
                                "store float {acc}, ptr %O{k}_{}",
                                y as u64 * oc + q
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// A consumer column of the thread (clamped inside the row: a column
    /// past it is never stored).
    pub(super) fn split_col(&mut self, x: &str, q: u64, contig: bool, g: u64, c: u64) -> String {
        match contig {
            true => self.chunk_col(x, q, g),
            false => {
                let col = self.op("add", x, g * q);
                let ok = self.tmp();
                self.line(format!("{ok} = icmp ult i64 {col}, {c}"));
                let s = self.tmp();
                self.line(format!("{s} = select i1 {ok}, i64 {col}, i64 {}", c - 1));
                s
            }
        }
    }

    /// A contiguous consumer's column `q` of lane `x`: 16-byte chunks
    /// interleaved across the row group — chunk `q ÷ 4` of lane `x` at
    /// `4x + 4g·(q ÷ 4)`, so a chunk's `g` lanes read `g · 16` consecutive
    /// bytes (one wavefront per 128, no bank conflict; lane-contiguous
    /// `x·oc` put every lane `oc · 4` bytes apart: `oc` = 16 meant a 4-way
    /// conflict on each 16-byte load, `attention-chains`).
    pub(super) fn chunk_col(&mut self, x: &str, q: u64, g: u64) -> String {
        let c0 = self.op("mul", x, 4);
        self.op("add", &c0, 4 * g * (q / 4) + q % 4)
    }

    /// Each fold's final value per row of the thread: its lanes' states
    /// combined (a shift fold's state already is), its laws, its seed.
    pub(super) fn finals(&mut self, st: &Stream, shift: &[bool]) -> Vec<Vec<String>> {
        let u = self.u;
        let n = u.members.len();
        let mut fin: Vec<Vec<String>> = vec![Vec::new(); n];
        for k in 0..n {
            let (StreamRole::Fold { laws }, UKind::Fold { site, .. }) =
                (&st.group.roles[k], &u.members[k].kind)
            else {
                continue;
            };
            let (op, seed) = (site.op, const_literal(&site.seed));
            let laws = laws.clone();
            for y in 0..st.rt as usize {
                let s = self.tmp();
                self.line(format!("{s} = load float, ptr %F{k}_{y}"));
                let mut tot = match shift[k] {
                    true => s,
                    false => self.butterfly(&s, st.g, op),
                };
                tot = self.laws_after(&laws, &fin, y, &tot, k);
                let res = self.tmp();
                self.step(&op, "float", &seed, &tot, &res);
                fin[k].push(res);
            }
        }
        fin
    }
}
