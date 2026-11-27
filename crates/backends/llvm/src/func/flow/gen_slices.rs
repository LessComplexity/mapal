//! `UnitGen`: slice folds, combines, tree totals and the tile phase

use super::*;

impl UnitGen<'_> {
    /// `body` once a row: by lane 0 of the row's first slice.
    pub(super) fn slice_first(&mut self, tag: &str, q: &str, body: impl FnOnce(&mut Self)) {
        let c = self.tmp();
        self.line(format!("{c} = icmp eq i64 {q}, 0"));
        let (yes, end) = (format!("{tag}q"), format!("{tag}qe"));
        self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
        self.label(&yes);
        self.lane0(tag, body);
        self.line(format!("br label %{end}"));
        self.label(&end);
    }

    /// Fold member `k`'s workspace element of row `row`, slice `slot`.
    pub(super) fn ws_at(
        &mut self,
        k: usize,
        sl: &Slices,
        t: &str,
        row: &str,
        slot: &str,
    ) -> String {
        let r = self.op("mul", row, sl.s);
        let i = self.op("add", &r, slot);
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds [{} x {t}], ptr addrspace(1) @{}.ws{k}, i64 0, i64 {i}",
            self.u.rows * sl.s,
            self.name
        ));
        p
    }

    /// Fold member `k`'s partial over slice `q` of its row: its lanes from
    /// the step's identity (a running max / min from the infinity it never
    /// keeps), its law reads at their identity, the tree; stored by lane 0.
    pub(super) fn slice_fold(&mut self, k: usize, sl: &Slices, q: &str) {
        let mb = &self.u.members[k];
        let UKind::Fold { site, .. } = &mb.kind else {
            unreachable!("a fold member")
        };
        let (site, src0, op) = (site.clone(), mb.srcs[0].clone(), site.op);
        let t = lower_ty(&site.elem).expect("float elem lowers");
        let init = step_identity(&op);
        let c = match &src0 {
            USrc::Member { k: c, .. } => Some(*c),
            _ => None,
        };
        for &(v, slot, law) in &sl.laws[k] {
            let lit = match law {
                DepLaw::Factor { .. } => "1.0".to_string(),
                DepLaw::Shift { sub: false } => "-0.0".to_string(),
                DepLaw::Shift { sub: true } | DepLaw::Phase => "0.0".to_string(),
                // The value's own partial over this slice: every term ≤ 1.
                DepLaw::Rescale => self.rescale_shift(v),
            };
            self.subst.push((c.expect("a law reads a cell"), slot, lit));
        }
        let kq = site.k / sl.s;
        let base = self.op("mul", q, kq);
        let (row, lane, lanes) = (self.row.clone(), self.lane.clone(), self.lanes);
        let tt = t.clone();
        let p = self
            .counted(
                &format!("f{k}"),
                (&lane, lanes, kq),
                Some((&t, &init)),
                &mut |g, x, acc| {
                    let kv = g.op("add", &base, x);
                    let v = g.fold_read(&site, &src0, &row, &kv);
                    g.step(&op, &tt, acc, &v, &format!("{acc}n"));
                },
            )
            .expect("a fold carries");
        self.subst.clear();
        let off = self.tree[k].expect("a sliced fold's tree");
        let tot = self.tree_total(&format!("r{k}"), off, &t, &p, op);
        self.parts[k] = Some(tot.clone());
        self.lane0(&format!("f{k}w"), |g| {
            let at = g.ws_at(k, sl, &t, &row, q);
            g.line(format!("store {t} {tot}, ptr addrspace(1) {at}"));
        });
    }

    /// The shift a rescale law's terms take over one slice: the value `v`'s
    /// own partial over the slice, or `0` where that partial is `v`'s step
    /// identity. A slice whose every `x` is the identity (scores masked to
    /// `−∞`) would compute `exp(κ(−∞ − (−∞))) = NaN` terms, and the combine's
    /// `NaN · exp(a(s_r, v)) = NaN` would poison the row. With the stand-in
    /// its terms are `exp(κ(−∞ − 0)) = 0`, and the combine still rescales by
    /// the slice's true shift: `0 · exp(a(−∞, v))` is `0` where `v` is finite
    /// and `NaN` where `v` is the identity too — the sequential fold's value
    /// (`exp(κ(−∞ − v))` a term) either way.
    pub(super) fn rescale_shift(&mut self, v: usize) -> String {
        let UKind::Fold { site, .. } = &self.u.members[v].kind else {
            unreachable!("a rescale's value is a fold")
        };
        let (vt, id) = (
            lower_ty(&site.elem).expect("float elem lowers"),
            step_identity(&site.op),
        );
        let m = self.parts[v]
            .clone()
            .expect("a rescale's shift is folded first in the item");
        let c = self.tmp();
        self.line(format!("{c} = fcmp oeq {vt} {m}, {id}"));
        let s = self.tmp();
        self.line(format!("{s} = select i1 {c}, {vt} 0.0, {vt} {m}"));
        s
    }

    /// Fold member `k`'s value in its row: the row's `s` partials combined
    /// as the lanes' tree, each law's per-row value, then the seed; stored
    /// once a row where `store`.
    pub(super) fn slice_combine(
        &mut self,
        k: usize,
        sl: &Slices,
        q: &str,
        store: bool,
        phase: u32,
    ) {
        let mb = &self.u.members[k];
        let UKind::Fold { site, .. } = &mb.kind else {
            unreachable!("a fold member")
        };
        let (op, seed) = (site.op, const_literal(&site.seed));
        let t = lower_ty(&site.elem).expect("float elem lowers");
        let row = self.row.clone();
        // Lane `r < s` loads partial `r` (the others the step's identity):
        // one load latency, then the lanes' tree — a fixed order, so every
        // block of the row combines the same bits.
        let lane = self.lane.clone();
        let c = self.tmp();
        self.line(format!("{c} = icmp ult i64 {lane}, {}", sl.s));
        let r = self.tmp();
        self.line(format!("{r} = select i1 {c}, i64 {lane}, i64 0"));
        let at = self.ws_at(k, sl, &t, &row, &r);
        let x = self.tmp();
        self.line(format!("{x} = load {t}, ptr addrspace(1) {at}"));
        // ADR-0042 D7: partial `r` was taken against its slice's shift
        // `s_r`; rescaled to the value's combined `v` by `exp(a(s_r, v))`.
        let mut x = x;
        for (&(v, _, law), arg) in sl.laws[k].iter().zip(&sl.args[k]) {
            let (DepLaw::Rescale, Some(arg)) = (law, arg) else {
                continue;
            };
            let vt = lower_ty(match &self.u.members[v].kind {
                UKind::Fold { site, .. } => &site.elem,
                _ => unreachable!("a rescale's value is a fold"),
            })
            .expect("float elem lowers");
            let at = self.ws_at(v, sl, &vt, &row, &r);
            let sr = self.tmp();
            self.line(format!("{sr} = load {vt}, ptr addrspace(1) {at}"));
            let val = self.vals[v]
                .clone()
                .expect("a rescale's value precedes its fold");
            let (a, b) = if arg.x_first {
                (&sr, &val)
            } else {
                (&val, &sr)
            };
            let mut d = self.tmp();
            self.line(format!("{d} = fsub {t} {a}, {b}"));
            for (sop, cst) in &arg.steps {
                let lit = super::super::const_literal(cst);
                let code = if *sop == Operation::Div {
                    "fdiv"
                } else {
                    "fmul"
                };
                let z = self.tmp();
                self.line(format!("{z} = {code} {t} {d}, {lit}"));
                d = z;
            }
            let bits = if t == "double" { 64 } else { 32 };
            let f = self.tmp();
            self.line(format!(
                "{f} = call {t} {}({t} {d})",
                crate::math::symbol(mapal_ir::MathFn::Exp, bits)
            ));
            let y = self.tmp();
            self.line(format!("{y} = fmul {t} {x}, {f}"));
            x = y;
        }
        let p = self.tmp();
        self.line(format!(
            "{p} = select i1 {c}, {t} {x}, {t} {}",
            step_identity(&op)
        ));
        let off = self.tree[k].expect("a sliced fold's tree");
        let mut tot = self.tree_total(&format!("p{phase}r{k}"), off, &t, &p, op);
        for &(v, _, law) in &sl.laws[k] {
            if law == DepLaw::Rescale {
                continue;
            }
            let val = self.vals[v]
                .clone()
                .expect("a law's value precedes its fold");
            let code = match law {
                DepLaw::Factor { div: false } => "fmul",
                DepLaw::Factor { div: true } => "fdiv",
                DepLaw::Shift { sub: false } => "fadd",
                DepLaw::Shift { sub: true } | DepLaw::Phase => "fsub",
                DepLaw::Rescale => unreachable!("applied to the partials"),
            };
            let z = self.tmp();
            self.line(format!("{z} = {code} {t} {tot}, {val}"));
            tot = z;
        }
        let res = self.tmp();
        self.step(&op, &t, &seed, &tot, &res);
        if store {
            self.slice_first(&format!("f{k}s"), q, |g| g.store(k, &res, &row));
        }
        self.vals[k] = Some(res);
    }

    /// A fold's read at step `kv` of row `row`: its operand in `Gmem` at the
    /// site's own address (re-addressed when its producer stores
    /// column-major), or an earlier member at its reach.
    pub(super) fn fold_read(
        &mut self,
        site: &RowFoldSite,
        src0: &USrc,
        row: &str,
        kv: &str,
    ) -> String {
        // The fold's axes: `k`, or a window's `(k÷div, k%div)`.
        let taps: Vec<String> = match site.a.ksplit {
            Some(ks) => vec![self.op("udiv", kv, ks.div), self.op("urem", kv, ks.div)],
            None => vec![kv.to_string()],
        };
        match src0 {
            USrc::Gmem(gm) => {
                let ri = if site.w > 1 {
                    let q = self.op("udiv", row, site.w);
                    let l = self.op("urem", row, site.w);
                    let a = self.op("mul", &q, site.a.ci);
                    let b = self.op("mul", &l, site.a.clane);
                    self.op("add", &a, &b)
                } else {
                    self.op("mul", row, site.a.ci)
                };
                let rk = match site.a.ksplit {
                    Some(ks) => {
                        let a = self.op("mul", &taps[0], ks.cq);
                        let b = self.op("mul", &taps[1], ks.cr);
                        self.op("add", &a, &b)
                    }
                    None => self.op("mul", kv, site.a.ck),
                };
                let x = self.op("add", &ri, &rk);
                let x = self.op("add", &x, site.a.base);
                self.load(*gm, &x)
            }
            src => {
                let taps: Vec<&str> = taps.iter().map(String::as_str).collect();
                self.read(src, row, &taps)
            }
        }
    }

    /// The lanes' partials `p` in shared memory at `off`, halved per step
    /// (labels `<tag>_<step>`): the total every lane reads (no seed).
    ///
    /// gapfix `shfl-tree` (gaps-disasm C7): an `f32` tree combines inside
    /// each warp by butterfly shuffles — each pair in its lower lane's order,
    /// so both lanes hold the same bits — and, past a warp, each warp's
    /// first lane posts its partial, one barrier, and every lane combines the
    /// row's warp partials as a balanced tree in registers: one barrier where
    /// the shared tree took one a level (`log₂ lanes + 1`; a 256-lane row
    /// 9 → 1). The tree's depth is the shared tree's, so the default face's
    /// bound is unchanged; a tree exists only where the order is free.
    pub(super) fn tree_total(
        &mut self,
        tag: &str,
        off: u64,
        t: &str,
        p: &str,
        op: RowFoldOp,
    ) -> String {
        let lanes = self.lanes;
        if self.shfl && t == "float" && lanes >= 2 && lanes.is_power_of_two() {
            return self.tree_total_shfl(tag, off, t, p, op);
        }
        let mine = self.smem(off, t, "%l");
        self.line(format!("store {t} {p}, ptr addrspace(3) {mine}"));
        self.barrier();
        let mut s = lanes / 2;
        while s >= 1 {
            let c = self.tmp();
            self.line(format!("{c} = icmp ult i64 {}, {s}", self.lane));
            let (yes, end) = (format!("{tag}_{s}"), format!("{tag}_{s}e"));
            self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
            self.label(&yes);
            let o = self.op("add", "%l", s);
            let po = self.smem(off, t, &o);
            let (x, y, z) = (self.tmp(), self.tmp(), self.tmp());
            self.line(format!("{x} = load {t}, ptr addrspace(3) {mine}"));
            self.line(format!("{y} = load {t}, ptr addrspace(3) {po}"));
            self.step(&op, t, &x, &y, &z);
            self.line(format!("store {t} {z}, ptr addrspace(3) {mine}"));
            self.line(format!("br label %{end}"));
            self.label(&end);
            self.barrier();
            s /= 2;
        }
        let base = self.lane_base.clone();
        let p0 = self.smem(off, t, &base);
        let tot = self.tmp();
        self.line(format!("{tot} = load {t}, ptr addrspace(3) {p0}"));
        tot
    }

    /// [`Self::tree_total`] by warp shuffles and one shared exchange.
    pub(super) fn tree_total_shfl(
        &mut self,
        tag: &str,
        off: u64,
        t: &str,
        p: &str,
        op: RowFoldOp,
    ) -> String {
        const WARP: u64 = 32;
        let lanes = self.lanes;
        let lane = self.lane.clone();
        let mut x = p.to_string();
        let mut s = lanes.min(WARP) / 2;
        while s >= 1 {
            let y = self.tmp();
            self.line(format!(
                "{y} = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {x}, i32 {s}, i32 31)"
            ));
            // The lower lane's partial first: the shared tree's operand order.
            let (m, hi) = (self.tmp(), self.tmp());
            self.line(format!("{m} = and i64 {lane}, {s}"));
            self.line(format!("{hi} = icmp ne i64 {m}, 0"));
            let (lo_v, hi_v, z) = (self.tmp(), self.tmp(), self.tmp());
            self.line(format!("{lo_v} = select i1 {hi}, {t} {y}, {t} {x}"));
            self.line(format!("{hi_v} = select i1 {hi}, {t} {x}, {t} {y}"));
            self.step(&op, t, &lo_v, &hi_v, &z);
            x = z;
            s /= 2;
        }
        if lanes <= WARP {
            return x;
        }
        // One exchange: warp `w`'s partial at slot `%l / 32`; every lane
        // reads its row's `lanes / 32` partials from the row's first warp.
        let nw = lanes / WARP;
        let wi = self.op("udiv", "%l", WARP);
        let li = self.op("urem", "%l", WARP);
        let first = self.tmp();
        self.line(format!("{first} = icmp eq i64 {li}, 0"));
        let (yes, end) = (format!("{tag}_xw"), format!("{tag}_xwe"));
        self.line(format!("br i1 {first}, label %{yes}, label %{end}"));
        self.label(&yes);
        let at = self.smem(off, t, &wi);
        self.line(format!("store {t} {x}, ptr addrspace(3) {at}"));
        self.line(format!("br label %{end}"));
        self.label(&end);
        self.barrier();
        let base = self.lane_base.clone();
        let w0 = self.op("udiv", &base, WARP);
        let mut level: Vec<String> = (0..nw)
            .map(|j| {
                let i = self.op("add", &w0, j);
                let at = self.smem(off, t, &i);
                let v = self.tmp();
                self.line(format!("{v} = load {t}, ptr addrspace(3) {at}"));
                v
            })
            .collect();
        while level.len() > 1 {
            let mut next = Vec::new();
            for pair in level.chunks(2) {
                let z = self.tmp();
                self.step(&op, t, &pair[0], &pair[1], &z);
                next.push(z);
            }
            level = next;
        }
        level.pop().expect("a warp partial")
    }

    /// compile-time-graph A: tile member `k` over the block's `b` rows. A
    /// thread holds `b × nj` accumulators — its columns `j` ([`tile_split`])
    /// of every row — over its k-slice, `step` k a trip; the row operand `a`
    /// comes from the producer's on-chip rows or `Gmem` (one address a warp:
    /// a broadcast), the column operand `b` from `Gmem`, each as 4-wide
    /// loads where the read is contiguous in `k` and aligned. Several slices
    /// combine through shared memory in slice order (the free order), the
    /// seed first. The results go to the member's on-chip rows and, when
    /// stored, to its buffer.
    pub(super) fn tile_phase(&mut self, k: usize, b: u64) {
        let u = self.u;
        if let Some(st) = u.block.as_ref().and_then(|bl| bl.stage[k]) {
            return self.tile_phase_staged(k, b, st);
        }
        let mb = &u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a tile phase runs a tile member")
        };
        let site = site.clone();
        let (a_src, b_src) = (mb.srcs[0].clone(), mb.srcs[1].clone());
        let t = lower_ty(&site.elem).expect("tile elem");
        let e = crate::profile::elem_bytes(&site.elem);
        let lanes = self.lanes;
        let (nj, slices) = tile_split(&site, lanes, self.free);
        let c = site.c;
        let kper = site.k / slices;
        let step = [16u64, 8, 4, 2, 1]
            .into_iter()
            .find(|x| kper % x == 0)
            .unwrap_or(1);
        let tag = format!("T{k}");
        // Threads past the row's columns × slices idle (a narrow row whose
        // `k` the order may not split).
        let active = c * slices < lanes;
        let (act, act_end) = (format!("{tag}act"), format!("{tag}acte"));
        if active {
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {}", c * slices));
            self.line(format!("br i1 {a}, label %{act}, label %{act_end}"));
            self.label(&act);
        }
        let (cols, k0) = if c >= lanes {
            let cols: Vec<String> = (0..nj).map(|q| self.op("add", "%l", q * lanes)).collect();
            (cols, "0".to_string())
        } else {
            let j = self.op("urem", "%l", c);
            let sl = self.op("udiv", "%l", c);
            let k0 = self.op("mul", &sl, kper);
            (vec![j], k0)
        };
        let k1 = self.op("add", &k0, kper);
        // Per-row and per-column address bases.
        let (va, a_rows): (u64, Vec<(bool, String)>) = match &a_src {
            USrc::Gmem(g) => {
                let v = if e == 4
                    && site.a.ck == 1
                    && step % 4 == 0
                    && site.a.base % 4 == 0
                    && site.a.ci % 4 == 0
                {
                    4
                } else {
                    1
                };
                let rows = (0..b)
                    .map(|r| {
                        let i = self.op("add", "%i", r);
                        let x = self.op("mul", &i, site.a.ci);
                        (false, self.op("add", &x, site.a.base) + &format!("@{g}"))
                    })
                    .collect();
                (v, rows)
            }
            USrc::Member { k: p, .. } => {
                let (off, wp) = (
                    self.kept[*p].expect("a tile's row operand is on chip"),
                    u.members[*p].w,
                );
                let v = if e == 4
                    && site.a.ck == 1
                    && step % 4 == 0
                    && site.a.base % 4 == 0
                    && wp % 4 == 0
                    && off % 16 == 0
                {
                    4
                } else {
                    1
                };
                let rows = (0..b)
                    .map(|r| (true, format!("{}@{off}", r * wp + site.a.base)))
                    .collect();
                (v, rows)
            }
            _ => unreachable!("a tile's row operand is a member or Gmem"),
        };
        let USrc::Gmem(bg) = b_src else {
            unreachable!("a tile's column operand is Gmem")
        };
        let vb = if e == 4
            && site.b.ck == 1
            && step % 4 == 0
            && site.b.base % 4 == 0
            && site.b.clane % 4 == 0
        {
            4
        } else {
            1
        };
        let b_cols: Vec<String> = cols
            .iter()
            .map(|j| {
                let x = self.op("mul", j, site.b.clane);
                self.op("add", &x, site.b.base)
            })
            .collect();
        let init = if slices > 1 {
            "-0.0".to_string()
        } else {
            const_literal(&site.seed)
        };
        let pre = self.cur.clone();
        let lp = tag.clone();
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [{k0}, %{pre}], [{x}n, %{lp}]"));
        let acc_name = |r: u64, q: u64| format!("%{tag}a{r}_{q}");
        for r in 0..b {
            for q in 0..nj {
                let a = acc_name(r, q);
                self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
            }
        }
        let mut cur: Vec<Vec<String>> = (0..b)
            .map(|r| (0..nj).map(|q| acc_name(r, q)).collect())
            .collect();
        let chunk = step.min(4);
        let mut u0 = 0;
        while u0 < step {
            let kx = self.op("add", &x, u0);
            // The chunk's row operands, `chunk` per row.
            let mut av: Vec<Vec<String>> = Vec::new();
            for (shared, base) in &a_rows {
                let (base, at) = base.split_once('@').expect("a base and its buffer");
                let (base, at) = (base.to_string(), at.to_string());
                let ck = site.a.ck;
                let idx = if *shared {
                    let x = self.op("mul", &kx, ck);
                    self.op("add", &x, &base)
                } else {
                    let x = self.op("mul", &kx, ck);
                    self.op("add", &x, &base)
                };
                av.push(self.tile_loads(*shared, &at, &t, &idx, chunk, va, ck));
            }
            let mut bv: Vec<Vec<String>> = Vec::new();
            for base in &b_cols {
                let x = self.op("mul", &kx, site.b.ck);
                let idx = self.op("add", &x, base);
                bv.push(self.tile_loads(false, &bg.to_string(), &t, &idx, chunk, vb, site.b.ck));
            }
            for uu in 0..chunk as usize {
                let last = u0 + chunk >= step && uu + 1 == chunk as usize;
                for r in 0..b as usize {
                    for q in 0..nj as usize {
                        let out = if last {
                            format!("{}n", acc_name(r as u64, q as u64))
                        } else {
                            self.tmp()
                        };
                        let (av_, bv_) = (av[r][uu].clone(), bv[q][uu].clone());
                        let acc = cur[r][q].clone();
                        self.tile_step(&site, &t, &acc, &av_, &bv_, &out);
                        cur[r][q] = out;
                    }
                }
            }
            u0 += chunk;
        }
        self.line(format!("{x}n = add i64 {x}, {step}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {k1}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{lp}e"));
        self.label(&format!("{lp}e"));
        self.tile_tail(
            k,
            &site,
            &t,
            b,
            (nj, slices),
            &cols,
            cur,
            (active, &act_end),
            &tag,
            false,
        );
    }
}
