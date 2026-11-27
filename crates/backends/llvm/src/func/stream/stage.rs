//! The streamed kernel's window staging: fold laws after a window, panels, addressing and the post calls.
use super::*;

impl UnitGen<'_> {
    /// Fold or consumer `k`'s laws after the last window on row `y`'s total:
    /// a rescale from the running shift to the value's final
    /// (`exp(a(m_run, v))`, NaN where both are the identity — the sequential
    /// fold's), then each factor or shift with the value.
    pub(super) fn laws_after(
        &mut self,
        laws: &[mapal_ir::StreamLaw],
        fin: &[Vec<String>],
        y: usize,
        tot: &str,
        _k: usize,
    ) -> String {
        let mut tot = tot.to_string();
        for l in laws {
            if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                let m = self.tmp();
                self.line(format!("{m} = load float, ptr %F{}_{y}", l.value));
                let e = self.exp_arg(arg, &m, &fin[l.value][y]);
                let z = self.tmp();
                self.line(format!("{z} = fmul float {tot}, {e}"));
                tot = z;
            }
        }
        for l in laws {
            let code = match l.law {
                DepLaw::Rescale => continue,
                DepLaw::Factor { div: false } => "fmul",
                DepLaw::Factor { div: true } => "fdiv",
                DepLaw::Shift { sub: false } => "fadd",
                DepLaw::Shift { sub: true } | DepLaw::Phase => "fsub",
            };
            let z = self.tmp();
            self.line(format!("{z} = {code} float {tot}, {}", fin[l.value][y]));
            tot = z;
        }
        tot
    }

    /// `exp(a(x := from, v := to))`: the exponent's difference, its steps,
    /// the default face's `exp`.
    pub(super) fn exp_arg(&mut self, arg: &mapal_ir::RescaleArg, from: &str, to: &str) -> String {
        let (a, b) = if arg.x_first { (from, to) } else { (to, from) };
        let mut d = self.tmp();
        self.line(format!("{d} = fsub float {a}, {b}"));
        for (sop, cst) in &arg.steps {
            let code = match *sop {
                Operation::Div => "fdiv",
                _ => "fmul",
            };
            let z = self.tmp();
            self.line(format!("{z} = {code} float {d}, {}", const_literal(cst)));
            d = z;
        }
        let f = self.tmp();
        self.line(format!(
            "{f} = call float {}(float {d})",
            crate::math::symbol(mapal_ir::MathFn::Exp, 32)
        ));
        f
    }

    /// A window's rescale of a state taken against shift `old` to shift
    /// `new`: `1` where the shift stayed (an identity shift's zero state
    /// stays zero), else `exp(a(old, new))`.
    pub(super) fn alpha(&mut self, arg: &mapal_ir::RescaleArg, old: &str, new: &str) -> String {
        let key = format!("{old}|{new}|{}|{:?}", arg.x_first, arg.steps);
        if let Some((_, v)) = self.alphas.iter().find(|x| x.0 == key) {
            return v.clone();
        }
        let e = self.exp_arg(arg, old, new);
        let c = self.tmp();
        self.line(format!("{c} = fcmp oeq float {old}, {new}"));
        let a = self.tmp();
        self.line(format!("{a} = select i1 {c}, float 1.0, float {e}"));
        self.alphas.push((key, a.clone()));
        a
    }

    /// `v` combined over the `g` lanes of the row group by `op` (a
    /// butterfly: every lane holds the total).
    pub(super) fn butterfly(&mut self, v: &str, g: u64, op: mapal_ir::RowFoldOp) -> String {
        let mut cur = v.to_string();
        let mut off = 1;
        while off < g {
            let o = self.tmp();
            self.line(format!(
                "{o} = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {cur}, i32 {off}, i32 31)"
            ));
            let r = self.tmp();
            self.step(&op, "float", &cur, &o, &r);
            cur = r;
            off *= 2;
        }
        cur
    }

    /// `body` where `c` holds.
    pub(super) fn cond(&mut self, tag: &str, c: &str, body: impl FnOnce(&mut Self)) {
        let (yes, end) = (format!("{tag}y"), format!("{tag}n"));
        self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
        self.label(&yes);
        body(self);
        self.line(format!("br label %{end}"));
        self.label(&end);
    }

    pub(super) fn commit(&mut self, cp: bool) {
        if cp {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
    }

    pub(super) fn wait(&mut self, cp: bool, pending: u32) {
        if cp {
            self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {pending})"
            ));
        }
    }

    /// Producer `k`'s row panel: its `a` operand's `b` rows of `d`.
    pub(super) fn stage_panel(&mut self, st: &Stream, lay: &Layout, k: usize) {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a producer is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[0] else {
            unreachable!("a producer's row operand is memory")
        };
        let (off, dp) = lay.q[k].expect("a producer's panel");
        let a = Addr::of(site, Operand::A);
        let d = site.k;
        // Row `r` of the block, element `kk`: `a(i + r, ·, kk)`.
        let vec = a.vec(Ax::K, d);
        self.stage(
            &format!("Q{k}"),
            st,
            (off, dp),
            (st.b, d),
            (gi, vec, 4),
            None,
            &mut |g, r, c| {
                // mask-frontend: the block's row `r` as the producer numbers
                // its rows (its tile order, [`mapal_ir::out_rows`]).
                let i = g.op("add", "%i", r);
                let i = g.walk_row(st, &i);
                let i = g.member_row(st, k, &i);
                g.addr_at(&a, &i, "0", c)
            },
        );
    }

    /// Member `k`'s window from key `j0`: a producer's `b` operand rows
    /// (`kb` of `d`), or a consumer's (`kb` rows of `c`) — at the block's
    /// rows (one panel batch: the operand's row digits do not move).
    pub(super) fn stage_window(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        j0: &str,
        produce: bool,
    ) {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a windowed member is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[1] else {
            unreachable!("a tile's column operand is memory")
        };
        let bb = Addr::of(site, Operand::B);
        let eb = win_bytes(site);
        let w = st.group.axis;
        let j0 = j0.to_string();
        let tag = format!("{}{k}_{}", if produce { "K" } else { "V" }, self.n);
        // The block's first row as this member numbers it.
        let first = self.walk_row(st, "%i");
        let i0 = self.member_row(st, k, &first);
        if produce {
            // Key `j`'s row: `b(i, j, kk)`, `kk < d`.
            let (off, dp) = lay.k[k].expect("a producer's window");
            let vec = bb.vec_by(Ax::K, site.k, 16 / eb);
            self.stage(
                &tag,
                st,
                (off, dp),
                (st.kb, site.k),
                (gi, vec, eb),
                Some((j0.clone(), w)),
                &mut |g, r, c| {
                    let j = g.op("add", &j0, r);
                    g.addr_at(&bb, &i0, &j, c)
                },
            );
        } else {
            // Key `j`'s row of the consumer: `b(i, c, j)`.
            let (off, cpd) = lay.v[k].expect("a consumer's window");
            let vec = bb.vec_by(Ax::Lane, site.c, 16 / eb);
            self.stage(
                &tag,
                st,
                (off, cpd),
                (st.kb, site.c),
                (gi, vec, eb),
                Some((j0.clone(), w)),
                &mut |g, r, c| {
                    let j = g.op("add", &j0, r);
                    g.addr_at(&bb, &i0, c, &j)
                },
            );
        }
    }

    /// mask-frontend: the group row at walk position `p`
    /// ([`Stream::order`]; else `p` itself).
    pub(super) fn walk_row(&mut self, st: &Stream, p: &str) -> String {
        match &st.order {
            Some(o) => self.digits(&o.to_group, p),
            None => p.to_string(),
        }
    }

    /// Group row `row` as member `k` numbers its rows (a digit permutation,
    /// [`mapal_ir::RowMap`]; else the row itself).
    pub(super) fn member_row(&mut self, st: &Stream, k: usize, row: &str) -> String {
        match st.rows_of.get(k).cloned().flatten() {
            Some(m) => self.digits(&m.from_group, row),
            None => row.to_string(),
        }
    }

    /// `Σ coef·((x ÷ stride) % radix)`.
    pub(super) fn digits(&mut self, ds: &[mapal_ir::AxisDigit], x: &str) -> String {
        let mut acc = "0".to_string();
        for d in ds {
            let q = if d.stride == 1 {
                x.to_string()
            } else {
                self.op("udiv", x, d.stride)
            };
            let r = self.op("urem", &q, d.radix);
            let t = self.op("mul", &r, d.coef);
            acc = self.op("add", &acc, &t);
        }
        acc
    }

    /// Operand address `a` at `(i, lane, k)`.
    pub(super) fn addr_at(&mut self, a: &Addr, i: &str, lane: &str, k: &str) -> String {
        match a {
            Addr::Plain(r) => {
                let x = self.op("mul", i, r.ci);
                let y = self.op("mul", lane, r.clane);
                let z = self.op("mul", k, r.ck);
                let s = self.op("add", &x, &y);
                let s = self.op("add", &s, &z);
                self.op("add", &s, r.base)
            }
            Addr::Out(c) => {
                let x = self.op("mul", i, *c);
                self.op("add", &x, lane)
            }
            Addr::Radix(r) => {
                let x = self.digits(&r.i, i);
                let y = self.digits(&r.lane, lane);
                let z = self.digits(&r.k, k);
                let s = self.op("add", &x, &y);
                let s = self.op("add", &s, &z);
                self.op("add", &s, r.base)
            }
        }
    }

    /// Copy `rows × cols` elements of `eb` bytes into shared memory at `off`
    /// (row stride `stride` elements): element `(r, c)` from `at(r, c)` of
    /// operand `gi`; 16-byte chunks where `vec` (each chunk's `16 ÷ eb`
    /// consecutive). Rows past `limit` (counted from `base`) are zero: their
    /// copies land in the dump slot from a row inside, and a zero is stored
    /// in their place. A 2-byte element alone is copied by the thread (no
    /// `cp.async` of 2 bytes).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage(
        &mut self,
        tag: &str,
        st: &Stream,
        (off, stride): (u64, u64),
        (rows, cols): (u64, u64),
        (gi, vec, eb): (usize, bool, u64),
        limit: Option<(String, u64)>,
        at: &mut dyn FnMut(&mut Self, &str, &str) -> String,
    ) {
        let cp = st.cp;
        let ety = if eb == 2 { "i16" } else { "float" };
        let v = if vec { 16 / eb } else { 1 };
        let per = cols / v;
        let total = rows * per;
        let threads = st.threads;
        let dump = st.layout(&self.u.members).dump;
        self.counted(tag, ("%l", threads, total), None, &mut |g, e, _| {
            let r = g.op("udiv", e, per);
            let c = g.op("urem", e, per);
            let cc = g.op("mul", &c, v);
            // A row past the axis reads the axis's last row instead.
            let (rr, ok) = match &limit {
                Some((base, w)) => {
                    let j = g.op("add", base, &r);
                    let ok = g.tmp();
                    g.line(format!("{ok} = icmp ult i64 {j}, {w}"));
                    let lim = g.tmp();
                    g.line(format!("{lim} = sub i64 {}, {base}", w - 1));
                    let rr = g.tmp();
                    g.line(format!("{rr} = select i1 {ok}, i64 {r}, i64 {lim}"));
                    (rr, Some(ok))
                }
                None => (r.clone(), None),
            };
            let src_i = at(g, &rr, &cc);
            let src = g.tmp();
            g.line(format!(
                "{src} = getelementptr inbounds {ety}, ptr addrspace(1) %x{gi}, i64 {src_i}"
            ));
            let rs = g.op("mul", &r, stride);
            let di = g.op("add", &rs, &cc);
            let dst = g.smem(off, ety, &di);
            let dumpp = g.smem(dump, ety, "0");
            let (into, zero_at) = match &ok {
                Some(ok) => {
                    let a = g.tmp();
                    g.line(format!(
                        "{a} = select i1 {ok}, ptr addrspace(3) {dst}, ptr addrspace(3) {dumpp}"
                    ));
                    let z = g.tmp();
                    g.line(format!(
                        "{z} = select i1 {ok}, ptr addrspace(3) {dumpp}, ptr addrspace(3) {dst}"
                    ));
                    (a, Some(z))
                }
                None => (dst.clone(), None),
            };
            // A 16-byte chunk moves as four words whatever its elements.
            let vt = if vec { "<4 x float>" } else { ety };
            let zero = match (vec, eb) {
                (true, _) => "zeroinitializer",
                (false, 2) => "0",
                _ => "0.0",
            };
            match (cp, vec) {
                (true, true) => g.line(format!(
                    "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {into}, ptr addrspace(1) {src})"
                )),
                (true, false) if eb == 4 => g.line(format!(
                    "call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) {into}, ptr addrspace(1) {src})"
                )),
                _ => {
                    let x = g.tmp();
                    g.line(format!("{x} = load {vt}, ptr addrspace(1) {src}"));
                    g.line(format!("store {vt} {x}, ptr addrspace(3) {into}"));
                }
            }
            if let Some(z) = zero_at {
                g.line(format!("store {vt} {zero}, ptr addrspace(3) {z}"));
            }
        });
    }

    /// Tile member `k`'s epilogue body at its output cell `t` with its fold's
    /// value `acc` (`tile-post`): memory operands by pointer, a per-row
    /// member's final value through its one-cell buffer, an array only the
    /// fold reads as `null`, the cell index.
    pub(super) fn post_call(
        &mut self,
        k: usize,
        t: &str,
        acc: &str,
        per_row: &[(u32, String)],
    ) -> String {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("an epilogue is a tile's")
        };
        let post = site.post.clone().expect("an epilogue");
        let c = self.calls[k].as_ref().expect("an epilogue call");
        let (callee, arg, ret, captures) =
            (c.callee.clone(), c.arg.clone(), c.ret.clone(), c.captures);
        let field_ty = c.field_ty.clone();
        let srcs = mb.srcs.clone();
        let mut agg = "poison".to_string();
        for s in 0..=captures {
            let (ty, v) = if s == captures {
                let it = field_ty[s as usize].clone();
                match it.as_str() {
                    "i64" => ("i64".to_string(), t.to_string()),
                    _ => ("i32".to_string(), self.i32_of(t)),
                }
            } else {
                match post.reads.iter().position(|&x| x == s) {
                    Some(j) => match &srcs[2 + j] {
                        USrc::Gmem(g) => ("ptr".to_string(), format!("%xg{g}")),
                        USrc::Member { .. } => {
                            let v = per_row
                                .iter()
                                .find(|x| x.0 == s)
                                .map(|x| x.1.clone())
                                .expect("a per-row read's value");
                            let b = format!("%b{k}_{s}");
                            self.line(format!("store {} {v}, ptr {b}", field_ty[s as usize]));
                            ("ptr".to_string(), b)
                        }
                        _ => unreachable!("an epilogue reads memory or a per-row member"),
                    },
                    // An array only the fold reads; a number only the
                    // fold's operands take (`unit-uniform`): unread.
                    // mask-frontend `stream-numbers`: one read after the
                    // fold is the launch's argument.
                    None if post.after.contains(&s) => {
                        (field_ty[s as usize].clone(), format!("%uv{k}_{s}"))
                    }
                    None if post.uniform.contains(&s) => {
                        let ty = field_ty[s as usize].clone();
                        let v = match ty.as_str() {
                            "float" | "double" | "half" | "bfloat" => "0.0",
                            _ => "0",
                        };
                        (ty, v.to_string())
                    }
                    None => ("ptr".to_string(), "null".to_string()),
                }
            };
            let n = self.tmp();
            self.line(format!("{n} = insertvalue {arg} {agg}, {ty} {v}, {s}"));
            agg = n;
        }
        let r = self.tmp();
        self.line(format!(
            "{r} = call {ret} @{callee}({arg} {agg}, float {acc})"
        ));
        r
    }
}
