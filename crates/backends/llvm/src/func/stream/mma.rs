//! The streamed kernel's products: the warp matrix product (`stream-mma`) and the FMA producer and consumer.
use super::*;

impl UnitGen<'_> {
    /// `stream-mma`: two f32 values rounded once into one register of the
    /// format, the lower `k` in the low half (the tile kernel's `pack`).
    pub(super) fn mma_pack(&mut self, fmt: crate::profile::MmaFmt, lo: &str, hi: &str) -> String {
        let r = self.tmp();
        self.line(format!(
            "{r} = call i32 asm \"cvt.rn.{f}x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {hi}, float {lo})",
            f = fmt.ptx()
        ));
        r
    }

    /// `stream-mma`: one `m16n8k16` product into the accumulators `c` (the
    /// PTX ISA's fragment layouts: `a` = (g, 2t), (g+8, 2t), (g, 2t+8),
    /// (g+8, 2t+8); `b` = (k 2t, n g), (k 2t+8, n g); `c` = (g, 2t..2t+1),
    /// (g+8, 2t..2t+1)).
    pub(super) fn mma_step(
        &mut self,
        fmt: crate::profile::MmaFmt,
        a: &[String; 4],
        b: &[String; 2],
        c: &[String; 4],
    ) -> [String; 4] {
        let r = self.tmp();
        let f = fmt.ptx();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{f}.{f}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        std::array::from_fn(|q| {
            let v = self.tmp();
            self.line(format!(
                "{v} = extractvalue {{ float, float, float, float }} {r}, {q}"
            ));
            v
        })
    }

    /// `stream-mma`: producer `k`'s row operand as `a` fragments, each
    /// `k`-step's four registers — the thread's rows `grow` (`g`, `g+8`),
    /// elements `16s + 2x (+8)` and the next, from memory once a block.
    pub(super) fn mma_rows(
        &mut self,
        st: &Stream,
        k: usize,
        grow: &[String],
        x: &str,
    ) -> Vec<[String; 4]> {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a producer is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[0] else {
            unreachable!("a producer's row operand is memory")
        };
        let site = site.clone();
        let fmt = st.mma.expect("stream-mma");
        let a = Addr::of(&site, Operand::A);
        let narrow = site.a.narrow.is_some();
        let x2 = self.op("mul", x, 2);
        // prefill-integrate: each row as the producer numbers its rows (its
        // tile order, [`mapal_ir::out_rows`]) — `stage_panel`'s mapping;
        // the group row read as the tile row took another token's query.
        let rows: Vec<String> = grow.iter().map(|r| self.member_row(st, k, r)).collect();
        let mut out = Vec::new();
        for s in 0..site.k / MMA_K {
            let mut q: Vec<String> = Vec::new();
            for (yy, kc) in [(0usize, 0u64), (1, 0), (0, 8), (1, 8)] {
                let mut v: Vec<String> = Vec::new();
                let ety = if narrow { "i16" } else { "float" };
                for e in 0..2 {
                    let kk = self.op("add", &x2, s * MMA_K + kc + e);
                    let i = self.addr_at(&a, &rows[yy], "0", &kk);
                    let ptr = self.tmp();
                    self.line(format!(
                        "{ptr} = getelementptr inbounds {ety}, ptr addrspace(1) %x{gi}, i64 {i}"
                    ));
                    let l = self.tmp();
                    let al = if narrow { 2 } else { 4 };
                    self.line(format!(
                        "{l} = load {ety}, ptr addrspace(1) {ptr}, align {al}"
                    ));
                    v.push(l);
                }
                if narrow {
                    // bf16-class `narrow-window`: the stored bf16 pair, the
                    // lower `k` in the low half.
                    let (lo, hi) = (self.tmp(), self.tmp());
                    self.line(format!("{lo} = zext i16 {} to i32", v[0]));
                    self.line(format!("{hi} = zext i16 {} to i32", v[1]));
                    let sh = self.tmp();
                    self.line(format!("{sh} = shl i32 {hi}, 16"));
                    let r = self.tmp();
                    self.line(format!("{r} = or i32 {sh}, {lo}"));
                    q.push(r);
                    continue;
                }
                q.push(self.mma_pack(fmt, &v[0], &v[1]));
            }
            out.push([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
        }
        out
    }

    /// `stream-mma`: producer `k`'s window — its `kb ÷ 8` accumulator tiles
    /// from the site's seed, folded over `d` by `k`-steps against the
    /// window's `b` fragments (key `8t + gq`'s elements `16s + 2x (+8)`, two
    /// values a load) — returned as the cells `(y, z)`: tile `t`'s `c0 c1`
    /// row 0 columns `2t`, `2t+1`, `c2 c3` row 1.
    pub(super) fn produce_mma(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        af: &[[String; 4]],
        x: &str,
        gq: &str,
    ) -> Vals {
        let UKind::Tile { site } = &self.u.members[k].kind else {
            unreachable!("a producer is a tile")
        };
        let site = site.clone();
        let fmt = st.mma.expect("stream-mma");
        let (koff, dp) = lay.k[k].expect("window");
        let narrow = win_bytes(&site) == 2;
        let seed = const_literal(&site.seed);
        let nt = (st.kb / MMA_N) as usize;
        let x2 = self.op("mul", x, 2);
        let gr = self.op("mul", gq, dp);
        let base = self.op("add", &gr, &x2);
        let mut c: Vec<[String; 4]> = vec![std::array::from_fn(|_| seed.clone()); nt];
        for (s, a) in af.iter().enumerate() {
            for (t, ct) in c.iter_mut().enumerate() {
                let mut b: Vec<String> = Vec::new();
                for kc in [0u64, 8] {
                    let i = self.op("add", &base, t as u64 * MMA_N * dp + s as u64 * MMA_K + kc);
                    if narrow {
                        // bf16-class `narrow-window`: the pair `k`, `k + 1`
                        // stored bf16 is the fragment register (the lower `k`
                        // in the low half).
                        let p = self.smem(koff, "i16", &i);
                        let v = self.tmp();
                        self.line(format!("{v} = load i32, ptr addrspace(3) {p}, align 4"));
                        b.push(v);
                        continue;
                    }
                    let p = self.smem(koff, "float", &i);
                    let v = self.tmp();
                    self.line(format!(
                        "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                    ));
                    let (e0, e1) = (self.tmp(), self.tmp());
                    self.line(format!("{e0} = extractelement <2 x float> {v}, i64 0"));
                    self.line(format!("{e1} = extractelement <2 x float> {v}, i64 1"));
                    b.push(self.mma_pack(fmt, &e0, &e1));
                }
                *ct = self.mma_step(fmt, a, &[b[0].clone(), b[1].clone()], ct);
            }
        }
        let ctn = st.ct() as usize;
        let mut out = vec![String::new(); 2 * ctn];
        for (t, ct) in c.iter().enumerate() {
            out[2 * t] = ct[0].clone();
            out[2 * t + 1] = ct[1].clone();
            out[ctn + 2 * t] = ct[2].clone();
            out[ctn + 2 * t + 1] = ct[3].clone();
        }
        out
    }

    /// `stream-mma`: consumer `k`'s window — the producer's cells `pv` (the
    /// accumulator layout is the `a` fragment's: `k`-step `s` packs columns
    /// `4s`, `4s+1` / `4s+2`, `4s+3` of rows 0 and 1) against the window's
    /// `b` fragments (keys `16s + 2x (+1, +8, +9)` of column `8t + gq`),
    /// into its accumulators `O(y, 2t + e)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn consume_mma(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        site: &mapal_ir::TileSite,
        pv: &[String],
        x: &str,
        gq: &str,
    ) {
        let fmt = st.mma.expect("stream-mma");
        let (voff, cpd) = lay.v[k].expect("window");
        let narrow = win_bytes(site) == 2;
        let (oc, _) = st.oc(site.c);
        let ct = st.ct() as usize;
        let x2 = self.op("mul", x, 2);
        let xr = self.op("mul", &x2, cpd);
        let base = self.op("add", &xr, gq);
        for s in 0..(st.kb / MMA_K) as usize {
            let z = 4 * s;
            let a = [
                self.mma_pack(fmt, &pv[z], &pv[z + 1]),
                self.mma_pack(fmt, &pv[ct + z], &pv[ct + z + 1]),
                self.mma_pack(fmt, &pv[z + 2], &pv[z + 3]),
                self.mma_pack(fmt, &pv[ct + z + 2], &pv[ct + z + 3]),
            ];
            for t in 0..site.c / MMA_N {
                let mut b: Vec<String> = Vec::new();
                for kc in [0u64, 8] {
                    let mut v: Vec<String> = Vec::new();
                    for e in 0..2 {
                        let off = (s as u64 * MMA_K + kc + e) * cpd + t * MMA_N;
                        let i = self.op("add", &base, off);
                        let l = self.tmp();
                        if narrow {
                            let p = self.smem(voff, "i16", &i);
                            self.line(format!("{l} = load i16, ptr addrspace(3) {p}, align 2"));
                        } else {
                            let p = self.smem(voff, "float", &i);
                            self.line(format!("{l} = load float, ptr addrspace(3) {p}, align 4"));
                        }
                        v.push(l);
                    }
                    if narrow {
                        // bf16-class `narrow-window`: the two keys' stored
                        // bf16 elements, the lower key in the low half.
                        let (lo, hi) = (self.tmp(), self.tmp());
                        self.line(format!("{lo} = zext i16 {} to i32", v[0]));
                        self.line(format!("{hi} = zext i16 {} to i32", v[1]));
                        let sh = self.tmp();
                        self.line(format!("{sh} = shl i32 {hi}, 16"));
                        let r = self.tmp();
                        self.line(format!("{r} = or i32 {sh}, {lo}"));
                        b.push(r);
                    } else {
                        b.push(self.mma_pack(fmt, &v[0], &v[1]));
                    }
                }
                let slots = [2 * t, 2 * t + 1, oc + 2 * t, oc + 2 * t + 1];
                let cur: [String; 4] = std::array::from_fn(|q| {
                    let v = self.tmp();
                    self.line(format!("{v} = load float, ptr %O{k}_{}", slots[q]));
                    v
                });
                let r = self.mma_step(fmt, &a, &[b[0].clone(), b[1].clone()], &cur);
                for (q, v) in r.iter().enumerate() {
                    self.line(format!("store float {v}, ptr %O{k}_{}", slots[q]));
                }
            }
        }
    }

    /// Producer `k`'s window: each of the thread's `rt × ct` cells — rows
    /// `rloc`, window columns `jl` — folded over `d` from the panels, from the
    /// site's seed in its own order.
    pub(super) fn produce(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        rloc: &[String],
        jl: &[String],
    ) -> Vals {
        let UKind::Tile { site } = &self.u.members[k].kind else {
            unreachable!("a producer is a tile")
        };
        let site = site.clone();
        let (rt, ct) = (st.rt as usize, st.ct() as usize);
        let (qoff, dp) = lay.q[k].expect("panel");
        let (koff, _) = lay.k[k].expect("window");
        let seed = const_literal(&site.seed);
        for c in 0..rt * ct {
            self.line(format!("store float {seed}, ptr %S{k}_{c}"));
        }
        let d = site.k;
        let v = if d.is_multiple_of(4) { 4 } else { 1 };
        let qrow: Vec<String> = rloc.iter().map(|r| self.op("mul", r, dp)).collect();
        let krow: Vec<String> = jl.iter().map(|j| self.op("mul", j, dp)).collect();
        self.counted(&format!("P{k}"), ("0", v, d), None, &mut |g, kk, _| {
            let load = |g: &mut Self, off: u64, base: &str| -> Vec<String> {
                let i = g.op("add", base, kk);
                let p = g.smem(off, "float", &i);
                if v == 4 {
                    let x = g.tmp();
                    g.line(format!(
                        "{x} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                    ));
                    (0..4)
                        .map(|e| {
                            let s = g.tmp();
                            g.line(format!("{s} = extractelement <4 x float> {x}, i32 {e}"));
                            s
                        })
                        .collect()
                } else {
                    let x = g.tmp();
                    g.line(format!("{x} = load float, ptr addrspace(3) {p}"));
                    vec![x]
                }
            };
            let qs: Vec<Vec<String>> = qrow.iter().map(|b| load(g, qoff, b)).collect();
            let ks: Vec<Vec<String>> = krow.iter().map(|b| load(g, koff, b)).collect();
            for y in 0..rt {
                for z in 0..ct {
                    let a = format!("%S{k}_{}", y * ct + z);
                    let mut acc = g.tmp();
                    g.line(format!("{acc} = load float, ptr {a}"));
                    for e in 0..v as usize {
                        let o = g.tmp();
                        g.tile_step(&site, "float", &acc, &qs[y][e], &ks[z][e], &o);
                        acc = o;
                    }
                    g.line(format!("store float {acc}, ptr {a}"));
                }
            }
        });
        let mut out = Vec::with_capacity(rt * ct);
        for c in 0..rt * ct {
            let s = self.tmp();
            self.line(format!("{s} = load float, ptr %S{k}_{c}"));
            out.push(s);
        }
        out
    }

    /// Consumer `k`'s window: its accumulators `rt × oc` fold the window's
    /// row operand (shared memory, 16-byte rows) against its window.
    pub(super) fn consume(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        site: &mapal_ir::TileSite,
        rloc: &[String],
        x: &str,
    ) {
        let (rt, kb, g) = (st.rt as usize, st.kb, st.g);
        let (aoff, ap) = lay.a[k].expect("row window");
        let (voff, cpd) = lay.v[k].expect("window");
        let (oc, contig) = st.oc(site.c);
        let c = site.c;
        let arow: Vec<String> = rloc.iter().map(|r| self.op("mul", r, ap)).collect();
        // The thread's columns (clamped inside the row; a column past it is
        // never stored).
        let cols: Vec<String> = (0..oc)
            .map(|q| match contig {
                true => self.chunk_col(x, q, g),
                false => {
                    let col = self.op("add", x, g * q);
                    let ok = self.tmp();
                    self.line(format!("{ok} = icmp ult i64 {col}, {c}"));
                    let s = self.tmp();
                    self.line(format!("{s} = select i1 {ok}, i64 {col}, i64 {}", c - 1));
                    s
                }
            })
            .collect();
        let site = site.clone();
        let x = x.to_string();
        self.counted(&format!("C{k}"), ("0", 4, kb), None, &mut |gg, jj, _| {
            let mut av: Vec<Vec<String>> = Vec::new();
            for r in &arow {
                let i = gg.op("add", r, jj);
                let p = gg.smem(aoff, "float", &i);
                let v = gg.tmp();
                gg.line(format!(
                    "{v} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                ));
                av.push(
                    (0..4)
                        .map(|e| {
                            let s = gg.tmp();
                            gg.line(format!("{s} = extractelement <4 x float> {v}, i32 {e}"));
                            s
                        })
                        .collect(),
                );
            }
            let mut accs: Vec<String> = Vec::with_capacity(rt * oc as usize);
            for idx in 0..rt as u64 * oc {
                let a = gg.tmp();
                gg.line(format!("{a} = load float, ptr %O{k}_{idx}"));
                accs.push(a);
            }
            for e in 0..4u64 {
                let jr = gg.op("add", jj, e);
                let vrow = gg.op("mul", &jr, cpd);
                let vs: Vec<String> = match contig {
                    true => {
                        let c0 = gg.op("mul", &x, 4);
                        let base = gg.op("add", &vrow, &c0);
                        let mut out = Vec::new();
                        for q4 in 0..oc / 4 {
                            let i = gg.op("add", &base, q4 * 4 * g);
                            let p = gg.smem(voff, "float", &i);
                            let v = gg.tmp();
                            gg.line(format!(
                                "{v} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                            ));
                            for l in 0..4 {
                                let s = gg.tmp();
                                gg.line(format!("{s} = extractelement <4 x float> {v}, i32 {l}"));
                                out.push(s);
                            }
                        }
                        out
                    }
                    false => cols
                        .iter()
                        .map(|col| {
                            let i = gg.op("add", &vrow, col);
                            let p = gg.smem(voff, "float", &i);
                            let v = gg.tmp();
                            gg.line(format!("{v} = load float, ptr addrspace(3) {p}"));
                            v
                        })
                        .collect(),
                };
                for y in 0..rt {
                    for q in 0..oc as usize {
                        let o = gg.tmp();
                        let idx = y * oc as usize + q;
                        gg.tile_step(&site, "float", &accs[idx], &av[y][e as usize], &vs[q], &o);
                        accs[idx] = o;
                    }
                }
            }
            for (idx, a) in accs.iter().enumerate() {
                gg.line(format!("store float {a}, ptr %O{k}_{idx}"));
            }
        });
    }
}
