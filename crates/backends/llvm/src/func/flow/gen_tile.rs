//! `UnitGen`: the staged tile phase, its pass, tail, loads and the rows loop

use super::*;

impl UnitGen<'_> {
    /// kernel-merging `row-stage`: member `k`'s tile phase over panels staged
    /// through shared memory ([`Stage`]). Every thread copies (a thread past
    /// the row's columns × slices folds a working thread's cells, its reads
    /// inside the panel, and stores nothing); each trip waits for its panel,
    /// meets the block, and — with two buffers — issues the next panel into
    /// the buffer the trip before read, then folds `kc` of `k` per slice from
    /// shared memory in the order the direct phase does.
    pub(super) fn tile_phase_staged(&mut self, k: usize, b: u64, st: Stage) {
        let u = self.u;
        let mb = &u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a tile phase runs a tile member")
        };
        let site = site.clone();
        let a_src = mb.srcs[0].clone();
        let USrc::Gmem(bg) = mb.srcs[1] else {
            unreachable!("a tile's column operand is Gmem")
        };
        let lanes = self.lanes;
        let (nj, slices) = tile_split(&site, lanes, self.free);
        let c = site.c;
        let tag = format!("T{k}");
        let active = c * slices < lanes;
        // The thread's cells: its columns and slice.
        let lc = match active {
            true => self.op("urem", "%l", c * slices),
            false => "%l".to_string(),
        };
        let (all, sl) = if c >= lanes {
            let cols: Vec<String> = (0..nj).map(|q| self.op("add", &lc, q * lanes)).collect();
            (cols, "0".to_string())
        } else {
            (vec![self.op("urem", &lc, c)], self.op("udiv", &lc, c))
        };
        // Column passes: pass `p` folds the thread's columns `p·njp …` — the
        // row's contiguous `cw = c / passes` from `p·cw`.
        let (njp, cw) = (nj / st.passes, c / st.passes);
        for pass in 0..st.passes {
            let cols: Vec<String> =
                all[(pass * njp) as usize..((pass + 1) * njp) as usize].to_vec();
            let tag = match st.passes {
                1 => tag.clone(),
                _ => format!("{tag}p{pass}"),
            };
            let j0 = pass * cw;
            self.tile_pass_staged(
                k,
                b,
                st,
                (&site, &a_src, bg),
                (&cols, &sl, &lc),
                (njp, slices, cw, j0),
                (&tag, active),
            );
        }
    }

    /// One column pass of [`Self::tile_phase_staged`]: `cols` (the thread's
    /// columns of the pass), its panels `cw` columns wide from column `j0`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn tile_pass_staged(
        &mut self,
        k: usize,
        b: u64,
        st: Stage,
        (site, a_src, bg): (&TileSite, &USrc, usize),
        (cols, sl, _lc): (&[String], &str, &str),
        (nj, slices, cw, j0): (u64, u64, u64, u64),
        (tag, active): (&str, bool),
    ) {
        let u = self.u;
        let t = lower_ty(&site.elem).expect("tile elem");
        let e = crate::profile::elem_bytes(&site.elem);
        let v = 16 / e;
        let lanes = self.lanes;
        let (c, kc) = (site.c, st.kc);
        let kper = site.k / slices;
        let trips = kper / kc;
        let base = self.panel.expect("a staged tile has panel buffers");
        let (a_bytes, stage_bytes) = st.bytes(b, c, slices, e);
        let (a_el, buf_el) = (a_bytes / e, stage_bytes / e);
        let tag = tag.to_string();
        // Per column: its panel row and that row's chunk swizzle.
        let nch = kc / v;
        let swz = |g: &mut Self, r: &str| -> String {
            if nch == 1 {
                return "0".to_string();
            }
            let sh = (8 / nch.min(8)).max(1).trailing_zeros();
            let x = g.op("lshr", r, sh);
            g.op("and", &x, nch.min(8) - 1)
        };
        let brow: Vec<(String, String)> = cols
            .iter()
            .map(|j| {
                let x = self.op("mul", sl, cw);
                let jl = self.op("sub", j, j0);
                let r = self.op("add", &x, &jl);
                let w = swz(self, &r);
                (self.op("mul", &r, kc), w)
            })
            .collect();
        let arow = self.op("mul", sl, b);
        let ty_v = format!("<{v} x {t}>");
        let s_ptr = |g: &mut Self, idx: &str| g.smem(base, &t, idx);
        let g_ptr = |g: &mut Self, at: &str, idx: &str| {
            let p = g.tmp();
            g.line(format!(
                "{p} = getelementptr inbounds {t}, ptr addrspace(1) %x{at}, i64 {idx}"
            ));
            p
        };
        // One panel's copies: trip `tr` into the buffer at element `bo`.
        let issue = |g: &mut Self, tr: &str, bo: &str, n: &mut u32| {
            let k0 = g.op("mul", tr, kc);
            let mut one =
                |g: &mut Self,
                 total: u64,
                 unit: u64,
                 f: &mut dyn FnMut(&mut Self, &str) -> (String, String)| {
                    let copies = total / unit;
                    for m in 0..copies.div_ceil(lanes) {
                        let x = g.op("add", "%l", m * lanes);
                        let part = (m + 1) * lanes > copies;
                        *n += 1;
                        let (yes, end) = (format!("{tag}c{n}"), format!("{tag}c{n}e"));
                        if part {
                            let ok = g.tmp();
                            g.line(format!("{ok} = icmp ult i64 {x}, {copies}"));
                            g.line(format!("br i1 {ok}, label %{yes}, label %{end}"));
                            g.label(&yes);
                        }
                        let el = g.op("mul", &x, unit);
                        let (gp, sx) = f(g, &el);
                        let sx = g.op("add", &sx, bo);
                        let sp = s_ptr(g, &sx);
                        let bytes = unit * e;
                        if st.cp {
                            let kind = if bytes == 16 { "cg" } else { "ca" };
                            g.line(format!(
                            "call void @llvm.nvvm.cp.async.{kind}.shared.global.{bytes}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                        ));
                        } else {
                            let (lt, al) = match unit {
                                1 => (t.clone(), e),
                                _ => (ty_v.clone(), 16),
                            };
                            let x = g.tmp();
                            g.line(format!(
                                "{x} = load {lt}, ptr addrspace(1) {gp}, align {al}"
                            ));
                            g.line(format!("store {lt} {x}, ptr addrspace(3) {sp}, align {al}"));
                        }
                        if part {
                            g.line(format!("br label %{end}"));
                            g.label(&end);
                        }
                    }
                };
            // The row operand: `[slice][row][kc]`.
            if st.a {
                let USrc::Gmem(ag) = a_src else {
                    unreachable!("a staged row operand is Gmem")
                };
                let unit = if st.a_vec { v } else { 1 };
                one(g, slices * b * kc, unit, &mut |g, el| {
                    let r_ = g.op("udiv", el, kc);
                    let kk = g.op("urem", el, kc);
                    let s_ = g.op("udiv", &r_, b);
                    let r = g.op("urem", &r_, b);
                    let i = g.op("add", "%i", &r);
                    let x = g.op("mul", &i, site.a.ci);
                    let x = g.op("add", &x, site.a.base);
                    let kx = g.op("mul", &s_, kper);
                    let kx = g.op("add", &kx, &k0);
                    let kx = g.op("add", &kx, &kk);
                    let kx = g.op("mul", &kx, site.a.ck);
                    let gi = g.op("add", &x, &kx);
                    (g_ptr(g, &ag.to_string(), &gi), el.to_string())
                });
            }
            // The column operand.
            let unit = if st.b_vec { v } else { 1 };
            one(g, slices * cw * kc, unit, &mut |g, el| {
                let (s_, j, kk, sx) = if st.b_kmaj {
                    let r = g.op("udiv", el, kc);
                    let kk = g.op("urem", el, kc);
                    let s_ = g.op("udiv", &r, cw);
                    let j = g.op("urem", &r, cw);
                    let w = swz(g, &r);
                    let ch = g.op("udiv", &kk, v);
                    let ch = g.op("xor", &ch, &w);
                    let ch = g.op("mul", &ch, v);
                    let lo = g.op("urem", &kk, v);
                    let rb = g.op("mul", &r, kc);
                    let x = g.op("add", &rb, &ch);
                    let x = g.op("add", &x, &lo);
                    (s_, j, kk, g.op("add", &x, a_el))
                } else {
                    let r = g.op("udiv", el, cw);
                    let j = g.op("urem", el, cw);
                    let s_ = g.op("udiv", &r, kc);
                    let kk = g.op("urem", &r, kc);
                    (s_, j, kk, g.op("add", el, a_el))
                };
                let j = g.op("add", &j, j0);
                let x = g.op("mul", &j, site.b.clane);
                let x = g.op("add", &x, site.b.base);
                let kx = g.op("mul", &s_, kper);
                let kx = g.op("add", &kx, &k0);
                let kx = g.op("add", &kx, &kk);
                let kx = g.op("mul", &kx, site.b.ck);
                let gi = g.op("add", &x, &kx);
                (g_ptr(g, &bg.to_string(), &gi), sx)
            });
        };
        let mut ncopy = 0u32;
        let init = if slices > 1 {
            "-0.0".to_string()
        } else {
            const_literal(&site.seed)
        };
        if st.stages == 2 {
            issue(self, "0", "0", &mut ncopy);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        let pre = self.cur.clone();
        let (lp, latch) = (tag.clone(), format!("{tag}l"));
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [0, %{pre}], [{x}n, %{latch}]"));
        let acc_name = |r: u64, q: u64| format!("%{tag}a{r}_{q}");
        for r in 0..b {
            for q in 0..nj {
                let a = acc_name(r, q);
                self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{latch}]"));
            }
        }
        let bo = if st.stages == 1 {
            self.barrier();
            issue(self, &x, "0", &mut ncopy);
            if st.cp {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
                self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
            }
            self.barrier();
            "0".to_string()
        } else {
            self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
            self.barrier();
            // The next panel (the last again on the last trip: in bounds,
            // into the buffer no thread reads this trip).
            let nx = self.op("add", &x, 1);
            let fit = self.tmp();
            self.line(format!("{fit} = icmp ult i64 {nx}, {trips}"));
            let nt = self.tmp();
            self.line(format!(
                "{nt} = select i1 {fit}, i64 {nx}, i64 {}",
                trips - 1
            ));
            let nb = self.op("and", &nx, 1);
            let nbo = self.op("mul", &nb, buf_el);
            issue(self, &nt, &nbo, &mut ncopy);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cb = self.op("and", &x, 1);
            self.op("mul", &cb, buf_el)
        };
        // The fold of this panel: per 16-byte chunk of `k`, each row's and
        // each column's values, then every cell's steps in `k` order.
        let k0 = self.op("mul", &x, kc);
        let sk = self.op("mul", sl, kper);
        let ka = self.op("add", &sk, &k0);
        let mut cur: Vec<Vec<String>> = (0..b)
            .map(|r| (0..nj).map(|q| acc_name(r, q)).collect())
            .collect();
        let vload = |g: &mut Self, idx: &str, n: u64| -> Vec<String> {
            let p = s_ptr(g, idx);
            if n == 1 {
                let x = g.tmp();
                g.line(format!("{x} = load {t}, ptr addrspace(3) {p}"));
                return vec![x];
            }
            let vec = g.tmp();
            g.line(format!(
                "{vec} = load <{n} x {t}>, ptr addrspace(3) {p}, align 16"
            ));
            (0..n)
                .map(|l| {
                    let x = g.tmp();
                    g.line(format!("{x} = extractelement <{n} x {t}> {vec}, i64 {l}"));
                    x
                })
                .collect()
        };
        for ch in 0..nch {
            let kk0 = ch * v;
            let mut av: Vec<Vec<String>> = Vec::new();
            for r in 0..b {
                av.push(match &a_src {
                    USrc::Gmem(_) => {
                        let x = self.op("add", &arow, r);
                        let x = self.op("mul", &x, kc);
                        let x = self.op("add", &x, kk0);
                        let x = self.op("add", &x, &bo);
                        vload(self, &x, v)
                    }
                    USrc::Member { k: p, .. } => {
                        let (off, wp) = (
                            self.kept[*p].expect("a tile's row operand is on chip"),
                            u.members[*p].w,
                        );
                        let ck = site.a.ck;
                        let vec = ck == 1 && site.a.base % v == 0 && wp % v == 0 && off % 16 == 0;
                        let kx = self.op("add", &ka, kk0);
                        let x = self.op("mul", &kx, ck);
                        let x = self.op("add", &x, r * wp + site.a.base);
                        if vec {
                            let p = self.smem(off, &t, &x);
                            let vv = self.tmp();
                            self.line(format!(
                                "{vv} = load {ty_v}, ptr addrspace(3) {p}, align 16"
                            ));
                            (0..v)
                                .map(|l| {
                                    let y = self.tmp();
                                    self.line(format!("{y} = extractelement {ty_v} {vv}, i64 {l}"));
                                    y
                                })
                                .collect()
                        } else {
                            (0..v)
                                .map(|l| {
                                    let y = self.op("add", &x, l * ck);
                                    let p = self.smem(off, &t, &y);
                                    let z = self.tmp();
                                    self.line(format!("{z} = load {t}, ptr addrspace(3) {p}"));
                                    z
                                })
                                .collect()
                        }
                    }
                    _ => unreachable!("a tile's row operand is a member or Gmem"),
                });
            }
            let mut bv: Vec<Vec<String>> = Vec::new();
            for (rb, w) in &brow {
                bv.push(if st.b_kmaj {
                    let c_ = self.op("xor", w, ch);
                    let c_ = self.op("mul", &c_, v);
                    let x = self.op("add", rb, &c_);
                    let x = self.op("add", &x, a_el);
                    let x = self.op("add", &x, &bo);
                    vload(self, &x, v)
                } else {
                    Vec::new()
                });
            }
            if !st.b_kmaj {
                for (q, j) in cols.iter().enumerate() {
                    for l in 0..v {
                        let x = self.op("mul", sl, kc);
                        let x = self.op("add", &x, kk0 + l);
                        let x = self.op("mul", &x, cw);
                        let jl = self.op("sub", j, j0);
                        let x = self.op("add", &x, &jl);
                        let x = self.op("add", &x, a_el);
                        let x = self.op("add", &x, &bo);
                        let y = vload(self, &x, 1);
                        bv[q].extend(y);
                    }
                }
            }
            for l in 0..v as usize {
                let last = ch + 1 == nch && l + 1 == v as usize;
                for r in 0..b as usize {
                    for q in 0..nj as usize {
                        let out = if last {
                            format!("{}n", acc_name(r as u64, q as u64))
                        } else {
                            self.tmp()
                        };
                        let (av_, bv_) = (av[r][l].clone(), bv[q][l].clone());
                        let acc = cur[r][q].clone();
                        self.tile_step(site, &t, &acc, &av_, &bv_, &out);
                        cur[r][q] = out;
                    }
                }
            }
        }
        self.line(format!("br label %{latch}"));
        self.label(&latch);
        self.line(format!("{x}n = add i64 {x}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {trips}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{lp}e"));
        self.label(&format!("{lp}e"));
        // The last trip's copy (the last panel again) lands before anything
        // else takes the region: a split tile's partials, a later phase's
        // trees or panels.
        if st.stages == 2 {
            self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
        }
        let (act, act_end) = (format!("{tag}act"), format!("{tag}acte"));
        if active {
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {}", c * slices));
            self.line(format!("br i1 {a}, label %{act}, label %{act_end}"));
            self.label(&act);
        }
        self.tile_tail(
            k,
            site,
            &t,
            b,
            (nj, slices),
            cols,
            cur,
            (active, &act_end),
            &tag,
            true,
        );
    }

    /// A tile phase's end: the slices' partials combined (a split tile), each
    /// row's cells kept on chip and / or stored, the phase's barrier.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tile_tail(
        &mut self,
        k: usize,
        site: &TileSite,
        t: &str,
        b: u64,
        (nj, slices): (u64, u64),
        cols: &[String],
        cur: Vec<Vec<String>>,
        (active, act_end): (bool, &str),
        tag: &str,
        staged: bool,
    ) {
        let c = site.c;
        // The results: one slice's accumulators, or the slices combined.
        let mut res: Vec<Vec<String>> = cur;
        let (comb, comb_end) = (format!("{tag}z"), format!("{tag}ze"));
        if slices > 1 {
            // A staged tile's partials share its panels' region: every
            // thread's last panel read comes first.
            if staged {
                self.barrier();
            }
            let off = self.part.expect("a split tile has partials");
            let sl = self.op("udiv", "%l", c);
            for r in 0..b {
                let x = self.op("mul", &sl, b);
                let x = self.op("add", &x, r);
                let x = self.op("mul", &x, c);
                let idx = self.op("add", &x, &cols[0]);
                let p = self.smem(off, t, &idx);
                let v = res[r as usize][0].clone();
                self.line(format!("store {t} {v}, ptr addrspace(3) {p}"));
            }
            self.barrier();
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {c}"));
            self.line(format!("br i1 {a}, label %{comb}, label %{comb_end}"));
            self.label(&comb);
            let seed = const_literal(&site.seed);
            for r in 0..b {
                let mut tot = seed.clone();
                for s in 0..slices {
                    let idx = self.op("add", &cols[0], (s * b + r) * c);
                    let p = self.smem(off, t, &idx);
                    let v = self.tmp();
                    self.line(format!("{v} = load {t}, ptr addrspace(3) {p}"));
                    let n = self.tmp();
                    let (x, y) = if site.add_acc_first {
                        (&tot, &v)
                    } else {
                        (&v, &tot)
                    };
                    self.line(format!("{n} = fadd {t} {x}, {y}"));
                    tot = n;
                }
                res[r as usize][0] = tot;
            }
        }
        for r in 0..b {
            for q in 0..nj {
                let v = res[r as usize][q as usize].clone();
                let j = cols[q as usize].clone();
                if let Some(off) = self.kept[k] {
                    let idx = self.op("add", &j, r * c);
                    let p = self.smem(off, t, &idx);
                    self.line(format!("store {t} {v}, ptr addrspace(3) {p}"));
                }
                if self.ys[k].is_some() {
                    let i = self.op("add", "%i", r);
                    let x = self.op("mul", &i, c);
                    let at = self.op("add", &x, &j);
                    self.store(k, &v, &at);
                }
            }
        }
        if slices > 1 {
            self.line(format!("br label %{comb_end}"));
            self.label(&comb_end);
        }
        if active {
            self.line(format!("br label %{act_end}"));
            self.label(act_end);
        }
        self.barrier();
    }

    /// `n` consecutive-in-`k` operand values at `idx` (stride `ck`): from the
    /// block's shared memory at byte `at` (`shared`) or `Gmem` operand `at`,
    /// as `n / v` loads of `v` lanes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn tile_loads(
        &mut self,
        shared: bool,
        at: &str,
        t: &str,
        idx: &str,
        n: u64,
        v: u64,
        ck: u64,
    ) -> Vec<String> {
        let ptr = |g: &mut Self, ix: &str| -> String {
            if shared {
                g.smem(at.parse().expect("a byte offset"), t, ix)
            } else {
                let p = g.tmp();
                g.line(format!(
                    "{p} = getelementptr inbounds {t}, ptr addrspace(1) %x{at}, i64 {ix}"
                ));
                p
            }
        };
        let space = if shared { 3 } else { 1 };
        let mut out = Vec::new();
        if v == 4 && n % 4 == 0 {
            for c0 in (0..n).step_by(4) {
                let ix = self.op("add", idx, c0);
                let p = ptr(self, &ix);
                let vec = self.tmp();
                self.line(format!(
                    "{vec} = load <4 x {t}>, ptr addrspace({space}) {p}, align 16"
                ));
                for l in 0..4 {
                    let x = self.tmp();
                    self.line(format!("{x} = extractelement <4 x {t}> {vec}, i64 {l}"));
                    out.push(x);
                }
            }
            return out;
        }
        for c0 in 0..n {
            let ix = self.op("add", idx, c0 * ck);
            let p = ptr(self, &ix);
            let x = self.tmp();
            self.line(format!("{x} = load {t}, ptr addrspace({space}) {p}"));
            out.push(x);
        }
        out
    }

    /// `out = acc ⊕ a ⊗ b`: the tile's step in the source's operand order, one
    /// `llvm.fma` where the face contracts.
    pub(super) fn tile_step(
        &mut self,
        site: &TileSite,
        t: &str,
        acc: &str,
        a: &str,
        b: &str,
        out: &str,
    ) {
        let (x, y) = if site.mul_a_first { (a, b) } else { (b, a) };
        if self.fma {
            let f = if t == "double" { "f64" } else { "f32" };
            self.line(format!(
                "{out} = call {t} @llvm.fma.{f}({t} {x}, {t} {y}, {t} {acc})"
            ));
            return;
        }
        let p = self.tmp();
        self.line(format!("{p} = fmul {t} {x}, {y}"));
        let (l, r) = if site.add_acc_first {
            (acc, p.as_str())
        } else {
            (p.as_str(), acc)
        };
        self.line(format!("{out} = fadd {t} {l}, {r}"));
    }

    /// Member `k`'s value type.
    pub(super) fn ret_ty(&self, k: usize) -> String {
        match (&self.calls[k], &self.u.members[k].kind) {
            (Some(c), _) => c.ret.clone(),
            (None, UKind::Fold { site, .. }) => lower_ty(&site.elem).expect("fold elem"),
            (None, UKind::Tile { site }) => lower_ty(&site.elem).expect("tile elem"),
            (None, UKind::Cell { .. }) => unreachable!("a cell has a call"),
        }
    }

    /// A row-block unit: per-row value `v` of member `k` into its slot for
    /// the segments after this one (lane 0; read after the segment's end).
    pub(super) fn keep_row_value(&mut self, k: usize, v: &str) {
        let (Some(off), Some(rl)) = (self.xrow.get(k).copied().flatten(), self.rloc.clone()) else {
            return;
        };
        let ty = self.ret_ty(k);
        self.lane0(&format!("x{k}s"), |g| {
            let p = g.smem(off, &ty, &rl);
            g.line(format!("store {ty} {v}, ptr addrspace(3) {p}"));
        });
    }

    /// Does a member of this segment (`from..=k`) read the on-chip row
    /// buffer member `k` writes — another member's row in it, earlier in the
    /// segment, or `k`'s own producer's in place?
    pub(super) fn reread(&self, k: usize, from: usize) -> bool {
        let Some(off) = self.kept[k] else {
            return false;
        };
        self.u.members[from..=k].iter().any(|mb| {
            mb.srcs
                .iter()
                .any(|x| matches!(x, USrc::Member { k: p, .. } if self.kept[*p] == Some(off)))
        })
    }

    /// compile-time-graph A: the members `ks` (segment `s`, no tile) over
    /// the block's `b` rows, [`row_groups`] rows at once: the block's lanes
    /// split into one group of `lanes ÷ P` per row (row `%i + %r<s> + g`,
    /// lane `%l mod (lanes ÷ P)`), the groups of a batch in lockstep; a
    /// barrier closes each batch (the tree partials and per-row slots are
    /// reused by the next).
    pub(super) fn rows_loop(&mut self, s: usize, b: u64, ks: std::ops::Range<usize>) {
        let lanes0 = self.lanes;
        let p = row_groups(b, lanes0);
        let l = lanes0 / p;
        let (g, lane, base) = (format!("%g{s}"), format!("%ll{s}"), format!("%lb{s}"));
        self.line(format!("{g} = udiv i64 %l, {l}"));
        self.line(format!("{lane} = urem i64 %l, {l}"));
        self.line(format!("{base} = mul i64 {g}, {l}"));
        let pre = self.cur.clone();
        let (lp, latch, end) = (format!("S{s}"), format!("S{s}l"), format!("S{s}e"));
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let (r, rl, i) = (format!("%r{s}"), format!("%rl{s}"), format!("%i{s}"));
        self.line(format!("{r} = phi i64 [0, %{pre}], [{r}n, %{latch}]"));
        self.line(format!("{rl} = add i64 {r}, {g}"));
        self.line(format!("{i} = add i64 %i, {rl}"));
        let (row0, rloc0) = (self.row.clone(), self.rloc.clone());
        let (lane0, base0) = (self.lane.clone(), self.lane_base.clone());
        self.row = i;
        self.rloc = Some(rl);
        self.lane = lane;
        self.lane_base = base;
        self.lanes = l;
        self.cur_seg = s;
        for k in ks.clone() {
            // A member writing an on-chip row buffer waits for every reader
            // of the buffer's earlier occupant in this segment: its own
            // producer's row in place, or a row `unit_plan` freed at an
            // earlier death (a fold's walk, a per-row cell or a stored-only
            // cell reads it with no barrier after).
            if self.reread(k, ks.start) {
                self.barrier();
            }
            self.phase(k);
        }
        self.line(format!("br label %{latch}"));
        self.label(&latch);
        self.barrier();
        self.line(format!("{r}n = add i64 {r}, {p}"));
        let c = self.tmp();
        self.line(format!("{c} = icmp ult i64 {r}n, {b}"));
        self.line(format!("br i1 {c}, label %{lp}, label %{end}"));
        self.label(&end);
        // A row's register values do not outlive it.
        for k in ks {
            self.vals[k] = None;
        }
        self.row = row0;
        self.rloc = rloc0;
        self.lane = lane0;
        self.lane_base = base0;
        self.lanes = lanes0;
    }
}
