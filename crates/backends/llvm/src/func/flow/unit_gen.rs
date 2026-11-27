//! `UnitGen` basics: names, addresses, loads, cells, stores, steps and the phase walk

use super::*;

impl UnitGen<'_> {
    pub(super) fn tmp(&mut self) -> String {
        self.n += 1;
        format!("%v{}", self.n)
    }

    pub(super) fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    pub(super) fn label(&mut self, l: &str) {
        self.out.push_str(l);
        self.out.push_str(":\n");
        self.cur = l.to_string();
    }

    pub(super) fn op(&mut self, op: &str, a: &str, b: impl std::fmt::Display) -> String {
        let r = self.tmp();
        self.line(format!("{r} = {op} i64 {a}, {b}"));
        r
    }

    /// The address `reach` gives cell `t` (with the counted axes `taps`).
    pub(super) fn addr(&mut self, reach: &Reach, t: &str, taps: &[&str]) -> String {
        let Reach::Affine {
            w,
            base,
            cq,
            cr,
            taps: tt,
        } = reach
        else {
            unreachable!("a unit reads members at affine reaches")
        };
        let (q, r) = if *w == 1 {
            (t.to_string(), None)
        } else {
            (self.op("udiv", t, w), Some(self.op("urem", t, w)))
        };
        let mut acc = self.op("mul", &q, cq);
        if let Some(r) = r {
            let rr = self.op("mul", &r, cr);
            acc = self.op("add", &acc, &rr);
        }
        for (tap, k) in tt.iter().zip(taps) {
            let kk = self.op("mul", k, tap.coef);
            acc = self.op("add", &acc, &kk);
        }
        self.op("add", &acc, base)
    }

    pub(super) fn i32_of(&mut self, t: &str) -> String {
        let x = self.tmp();
        self.line(format!("{x} = trunc i64 {t} to i32"));
        x
    }

    pub(super) fn load(&mut self, g: usize, at: &str) -> String {
        let ty = self.gmem_ty[g].clone();
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace(1) %x{g}, i64 {at}"
        ));
        let v = self.tmp();
        self.line(format!("{v} = load {ty}, ptr addrspace(1) {p}"));
        v
    }

    /// Source `s` of a reader at its cell `t` (and fold axes `taps`).
    pub(super) fn read(&mut self, s: &USrc, t: &str, taps: &[&str]) -> String {
        match s {
            USrc::Member { k, reach, row } => {
                if self.u.members[*k].per_row() && *row {
                    // A row-block unit's per-row value from an earlier segment.
                    if let (Some(off), Some(rl)) =
                        (self.xrow.get(*k).copied().flatten(), self.rloc.clone())
                        && self.seg[*k] != self.cur_seg
                    {
                        let ty = self.ret_ty(*k);
                        let p = self.smem(off, &ty, &rl);
                        let v = self.tmp();
                        self.line(format!("{v} = load {ty}, ptr addrspace(3) {p}"));
                        return v;
                    }
                    return self.vals[*k]
                        .clone()
                        .expect("a per-row value is read after its phase");
                }
                let a = self.addr(reach, t, taps);
                if let (true, Some(off)) = (*row, self.kept[*k]) {
                    let w = self.u.members[*k].w;
                    let row = self.row.clone();
                    let r = self.op("mul", &row, w);
                    let mut j = self.op("sub", &a, &r);
                    if let Some(rl) = self.rloc.clone() {
                        let base = self.op("mul", &rl, w);
                        j = self.op("add", &j, &base);
                    }
                    let ty = self.ret_ty(*k);
                    let p = self.smem(off, &ty, &j);
                    let v = self.tmp();
                    self.line(format!("{v} = load {ty}, ptr addrspace(3) {p}"));
                    return v;
                }
                self.cell(*k, &a)
            }
            USrc::Gmem(g) => self.load(*g, t),
            USrc::Index => self.i32_of(t),
            USrc::Law(_) => unreachable!("a law feeds a capture slot"),
        }
    }

    /// Cell member `k`'s value at its cell `t`.
    pub(super) fn cell(&mut self, k: usize, t: &str) -> String {
        let u = self.u;
        let mb = &u.members[k];
        let c = self.calls[k].as_ref().expect("a cell member");
        let (callee, arg, ret, captures) =
            (c.callee.clone(), c.arg.clone(), c.ret.clone(), c.captures);
        let field_ty = c.field_ty.clone();
        let laws = c.laws.clone();
        let mut fields: Vec<(String, String)> = Vec::new();
        for (s, src) in mb.srcs.iter().enumerate() {
            let s32 = s as u32;
            let sub = self
                .subst
                .iter()
                .find(|&&(m, sl, _)| m == k && sl == s32)
                .map(|x| x.2.clone());
            if s32 < captures {
                let v = match src {
                    _ if sub.is_some() => sub.clone().expect("a substituted read"),
                    USrc::Gmem(g) => {
                        fields.push(("ptr".into(), format!("%xg{g}")));
                        continue;
                    }
                    USrc::Law(_) => {
                        let law = &laws.iter().find(|(x, _)| *x == s32).expect("law").1;
                        let ti = self.i32_of(t);
                        let v = self.tmp();
                        self.line(format!("{v} = call {} @{law}(i32 {ti})", field_ty[s]));
                        v
                    }
                    other => self.read(other, t, &[]),
                };
                self.line(format!("store {} {v}, ptr %b{k}_{s}", field_ty[s]));
                fields.push(("ptr".into(), format!("%b{k}_{s}")));
            } else {
                let v = match &sub {
                    Some(x) => x.clone(),
                    None => self.read(src, t, &[]),
                };
                let ty = if matches!(src, USrc::Index) {
                    "i32".to_string()
                } else {
                    field_ty[s].clone()
                };
                fields.push((ty, v));
            }
        }
        let a = if captures == 0 {
            fields[0].1.clone()
        } else {
            let mut agg = "poison".to_string();
            for (f, (ty, v)) in fields.iter().enumerate() {
                let n = self.tmp();
                self.line(format!("{n} = insertvalue {arg} {agg}, {ty} {v}, {f}"));
                agg = n;
            }
            agg
        };
        let r = self.tmp();
        self.line(format!("{r} = call {ret} @{callee}({arg} {a})"));
        r
    }

    pub(super) fn store(&mut self, k: usize, v: &str, at: &str) {
        if let Some(j) = self.ys[k] {
            let ty = match &self.calls[k] {
                Some(c) => c.ret.clone(),
                None => match &self.u.members[k].kind {
                    UKind::Fold { site, .. } => lower_ty(&site.elem).expect("fold elem"),
                    UKind::Tile { site } => lower_ty(&site.elem).expect("tile elem"),
                    UKind::Cell { .. } => unreachable!("a cell has a call"),
                },
            };
            let p = self.tmp();
            self.line(format!(
                "{p} = getelementptr inbounds {ty}, ptr addrspace(1) %y{j}, i64 {at}"
            ));
            self.line(format!("store {ty} {v}, ptr addrspace(1) {p}"));
        }
    }

    /// `out = acc ⊕ v`: the fold's own step, in the source's operand order.
    pub(super) fn step(&mut self, op: &RowFoldOp, t: &str, acc: &str, v: &str, out: &str) {
        match *op {
            RowFoldOp::Arith { op, acc_first } => {
                let code = match op {
                    Operation::Add => "fadd",
                    Operation::Sub => "fsub",
                    _ => "fmul",
                };
                let (x, y) = if acc_first { (acc, v) } else { (v, acc) };
                self.line(format!("{out} = {code} {t} {x}, {y}"));
            }
            RowFoldOp::Select {
                cmp,
                acc_lhs,
                acc_if_true,
            } => {
                let pred = match cmp {
                    Operation::Lt => "olt",
                    Operation::Gt => "ogt",
                    Operation::Le => "ole",
                    _ => "oge",
                };
                let (x, y) = if acc_lhs { (acc, v) } else { (v, acc) };
                let (p, q) = if acc_if_true { (acc, v) } else { (v, acc) };
                let c = self.tmp();
                self.line(format!("{c} = fcmp {pred} {t} {x}, {y}"));
                self.line(format!("{out} = select i1 {c}, {t} {p}, {t} {q}"));
            }
        }
    }

    /// A pointer to element `idx` (of `ty`) at byte `off` of the launch's
    /// shared memory.
    pub(super) fn smem(&mut self, off: u64, ty: &str, idx: &str) -> String {
        let q = self.tmp();
        self.line(format!(
            "{q} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {off}"
        ));
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace(3) {q}, i64 {idx}"
        ));
        p
    }

    pub(super) fn barrier(&mut self) {
        self.line("call void @llvm.nvvm.barrier0()");
    }

    /// `body` by lane 0 of the row alone (every lane holds the value).
    pub(super) fn lane0(&mut self, tag: &str, body: impl FnOnce(&mut Self)) {
        if self.lanes == 1 {
            return body(self);
        }
        let c = self.tmp();
        self.line(format!("{c} = icmp eq i64 {}, 0", self.lane));
        let (yes, end) = (format!("{tag}z"), format!("{tag}ze"));
        self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
        self.label(&yes);
        body(self);
        self.line(format!("br label %{end}"));
        self.label(&end);
    }

    /// The counted loop `%<tag>x = start, start + stride, … < n` around one
    /// block `body(x, acc)`, entered only when `start < n`. With `carry`
    /// (type, init) the body defines `<acc>n`, the next value; the loop's exit
    /// value is returned.
    pub(super) fn counted(
        &mut self,
        tag: &str,
        (start, stride, n): (&str, u64, u64),
        carry: Option<(&str, &str)>,
        body: &mut dyn FnMut(&mut Self, &str, &str),
    ) -> Option<String> {
        let pre = self.cur.clone();
        let (lp, end) = (tag.to_string(), format!("{tag}e"));
        let c0 = self.tmp();
        self.line(format!("{c0} = icmp ult i64 {start}, {n}"));
        self.line(format!("br i1 {c0}, label %{lp}, label %{end}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [{start}, %{pre}], [{x}n, %{lp}]"));
        let a = format!("%{tag}a");
        if let Some((t, init)) = carry {
            self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
        }
        body(self, &x, &a);
        assert_eq!(self.cur, lp, "a counted body is one block");
        self.line(format!("{x}n = add i64 {x}, {stride}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {n}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{end}"));
        self.label(&end);
        carry.map(|(t, init)| {
            let r = self.tmp();
            self.line(format!("{r} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
            r
        })
    }

    /// Member `k`'s phase in row `%i`, lane `%l` of [`Geo`]'s `lanes`.
    pub(super) fn phase(&mut self, k: usize) {
        let u = self.u;
        let mb = &u.members[k];
        let lanes = self.lanes;
        match &mb.kind {
            UKind::Fold { site, .. } => {
                let t = lower_ty(&site.elem).expect("float elem lowers");
                let seed = const_literal(&site.seed);
                let tree = self.tree[k];
                // A tree's lanes start from the step's identity (a max/min's
                // seed: that step is idempotent); an in-order fold from the seed.
                let init = match (tree, site.op) {
                    (None, _) | (_, RowFoldOp::Select { .. }) => seed.clone(),
                    (
                        _,
                        RowFoldOp::Arith {
                            op: Operation::Mul, ..
                        },
                    ) => "1.0".to_string(),
                    _ => "-0.0".to_string(),
                };
                let lane = self.lane.clone();
                let walk = if tree.is_some() {
                    (lane.as_str(), lanes, site.k)
                } else {
                    ("0", 1, site.k)
                };
                let (site, src0, op) = (site.clone(), mb.srcs[0].clone(), site.op);
                let row = self.row.clone();
                let tt = t.clone();
                let p = self
                    .counted(
                        &format!("f{k}"),
                        walk,
                        Some((&t, &init)),
                        &mut |g, kv, acc| {
                            let v = g.fold_read(&site, &src0, &row, kv);
                            g.step(&op, &tt, acc, &v, &format!("{acc}n"));
                        },
                    )
                    .expect("a fold carries");
                let res = match tree {
                    None => p,
                    Some(off) => {
                        // The lanes' partials, then the seed once.
                        let tot = self.tree_total(&format!("r{k}"), off, &t, &p, op);
                        let r = self.tmp();
                        self.step(&op, &t, &seed, &tot, &r);
                        r
                    }
                };
                self.lane0(&format!("f{k}s"), |g| g.store(k, &res, &row));
                self.keep_row_value(k, &res);
                self.vals[k] = Some(res);
            }
            UKind::Cell { .. } if mb.per_row() => {
                let row = self.row.clone();
                let v = self.cell(k, &row);
                self.lane0(&format!("c{k}s"), |g| g.store(k, &v, &row));
                self.keep_row_value(k, &v);
                self.vals[k] = Some(v);
            }
            UKind::Cell { .. } if mb.store || self.kept[k].is_some() => {
                let (w, kept) = (mb.w, self.kept[k]);
                let (row, rloc, lane) = (self.row.clone(), self.rloc.clone(), self.lane.clone());
                self.counted(&format!("s{k}"), (&lane, lanes, w), None, &mut |g, j, _| {
                    let r = g.op("mul", &row, w);
                    let t = g.op("add", &r, j);
                    let v = g.cell(k, &t);
                    g.store(k, &v, &t);
                    if let Some(off) = kept {
                        let ty = g.calls[k].as_ref().expect("a cell").ret.clone();
                        let idx = match &rloc {
                            Some(rl) => {
                                let base = g.op("mul", rl, w);
                                g.op("add", &base, j)
                            }
                            None => j.to_string(),
                        };
                        let p = g.smem(off, &ty, &idx);
                        g.line(format!("store {ty} {v}, ptr addrspace(3) {p}"));
                    }
                });
                if kept.is_some() {
                    self.barrier();
                }
            }
            // Evaluated at its reads.
            UKind::Cell { .. } => {}
            UKind::Tile { .. } => unreachable!("a tile runs as its own phase"),
        }
    }

    /// long-rows `row-slices`: the sliced unit's body ([`Slices`]). Per
    /// phase, every block strides over the items `(row, slice)` of the
    /// `rows · s`: in member order a fold of the phase walks its slice into
    /// a partial (its law reads at their identity) stored to its workspace, a
    /// fold complete by now combines its row's partials in slice order (laws,
    /// then the seed; stored once a row), a per-row cell is computed (stored
    /// once a row in its own phase), a stored row cell writes its slice. The
    /// block meets after each item (the tree partials' shared memory), the
    /// grid after each phase but the last.
    pub(super) fn sliced(&mut self, sl: &Slices) {
        let u = self.u;
        let n = u.members.len();
        let items = u.rows * sl.s;
        self.line("%nct = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()");
        self.line("%g64 = zext i32 %nct to i64");
        for p in 0..sl.phases {
            // long-prefill `slice-launches`: a launch runs the phase its
            // `%phase` argument names; the others' items loops are empty.
            let items = match sl.launches {
                true => {
                    let on = self.tmp();
                    self.line(format!("{on} = icmp eq i64 %phase, {p}"));
                    let n = self.tmp();
                    self.line(format!("{n} = select i1 {on}, i64 {items}, i64 0"));
                    n
                }
                false => items.to_string(),
            };
            let (head, body, latch, exit) = (
                format!("ph{p}"),
                format!("ph{p}b"),
                format!("ph{p}l"),
                format!("ph{p}x"),
            );
            let pre = self.cur.clone();
            self.line(format!("br label %{head}"));
            self.label(&head);
            let it = format!("%it{p}");
            self.line(format!("{it} = phi i64 [%b64, %{pre}], [{it}n, %{latch}]"));
            let c = self.tmp();
            self.line(format!("{c} = icmp ult i64 {it}, {items}"));
            self.line(format!("br i1 {c}, label %{body}, label %{exit}"));
            self.label(&body);
            let (row, q) = (format!("%ri{p}"), format!("%rq{p}"));
            self.line(format!("{row} = udiv i64 {it}, {}", sl.s));
            self.line(format!("{q} = urem i64 {it}, {}", sl.s));
            self.row = row.clone();
            self.vals = vec![None; n];
            self.parts = vec![None; n];
            for k in 0..n {
                let mb = &u.members[k];
                let (ph, av) = (sl.phase[k], sl.avail[k]);
                match &mb.kind {
                    UKind::Fold { .. } if ph == p => self.slice_fold(k, sl, &q),
                    UKind::Fold { .. } if av <= p => {
                        self.slice_combine(k, sl, &q, p == av && mb.store, p);
                    }
                    UKind::Cell { .. } if mb.per_row() && ph <= p => {
                        let v = self.cell(k, &row);
                        if ph == p && mb.store {
                            self.slice_first(&format!("c{k}"), &q, |g| g.store(k, &v, &row));
                        }
                        self.vals[k] = Some(v);
                    }
                    UKind::Cell { .. } if !mb.per_row() && mb.store && ph == p => {
                        let (w, lanes, lane) = (mb.w, self.lanes, self.lane.clone());
                        let wq = w / sl.s;
                        let base = self.op("mul", &q, wq);
                        let r0 = self.op("mul", &row, w);
                        self.counted(
                            &format!("s{k}"),
                            (&lane, lanes, wq),
                            None,
                            &mut |g, x, _| {
                                let j = g.op("add", &base, x);
                                let t = g.op("add", &r0, &j);
                                let v = g.cell(k, &t);
                                g.store(k, &v, &t);
                            },
                        );
                    }
                    _ => {}
                }
            }
            // The next item reuses the tree partials' shared memory.
            self.barrier();
            self.line(format!("br label %{latch}"));
            self.label(&latch);
            self.line(format!("{it}n = add i64 {it}, %g64"));
            self.line(format!("br label %{head}"));
            self.label(&exit);
            if p + 1 < sl.phases && !sl.launches {
                self.line(format!(
                    "call void @mapal_gsync(ptr addrspace(1) @{}.gsync)",
                    self.name
                ));
            }
        }
    }
}
