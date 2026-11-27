//! `Kernel` 16-bit tensor-core kernel.

use super::*;

impl Kernel<'_> {
    /// **The 16-bit tensor-core kernel** (tensor-cores, ADR-0043 D6):
    /// [`Self::mma`]'s block, warp and `m16n8` tiles over
    /// `mma.sync.m16n8k16.row.col.f32.{bf16,f16}` with f32 accumulators in
    /// registers. Every panel lands by 16-byte `cp.async` in its storage
    /// format (`g.ea`, `g.eb`): a narrow panel's fragments are read by
    /// `ldmatrix` (`a`: `.x4`, row-major `m × k`; `b`: `.x4.trans` / `.x2.trans`
    /// of the `k × n` panel — two `n8` tiles a load); an f32 panel's pairs
    /// are loaded and rounded once to the format (`cvt.rn.bf16x2.f32`: the
    /// member the site's product admits, ADR-0043 D4).
    ///
    /// Fragment layouts are the PTX ISA's for m16n8k16 (`group = lane / 4`,
    /// `t = lane % 4`, two elements a register, the lower `k` in the low
    /// half): A `a0..a3` = (g, 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8); B
    /// `b0, b1` = (k 2t, n g), (k 2t+8, n g); C as m16n8k8's.
    pub(super) fn mma16(&mut self, name: &str, g: GpuMmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let (ea, eb, ab) = (g.ea, g.eb, g.a_bytes());
        let stage_bytes = g.stage_bytes();
        let ptx = g.fmt.ptx();
        let mk = g.mma_k();
        // f32-tensor-cores `mma-combine`: a split kernel folds its piece
        // `z` of `S` — `k / S` from `z · k / S` ([`Kernel::at_z`]).
        let kext = s.k / self.splitk.max(1);
        let nk = kext / bk;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";
        let ety = |e: u64| if e == 2 { "i16" } else { "float" };

        let tid = self.sreg("tid.x");
        let accs = mt * nt * 4;
        // bf16 tensor-tile `mma-walk`: a stream-K kernel's accumulators live
        // in the entry block (where they promote to registers) across its
        // pieces; each piece re-seeds them and runs the pipeline over its
        // panels `[p0, p1)` of its tile ([`Kernel::segment`]).
        let walk_acc = self.streamk.map(|_| {
            assert!(g.split.is_none(), "a split form's tile is not walked");
            let a = self.tmp();
            self.line(format!("{a} = alloca [{accs} x float], align 16"));
            a
        });
        let seg = self.streamk.map(|iters| self.segment(iters, s.k / bk, bk));
        let (bx, by) = match &seg {
            Some(sg) => (sg.bx.clone(), sg.by.clone()),
            None => self.block_ids(),
        };
        let warp_id = self.op("udiv", &tid, warp);
        let lane = self.op("urem", &tid, warp);
        let wr = self.op("udiv", &warp_id, g.warps_x);
        let wc = self.op("urem", &warp_id, g.warps_x);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let acc = match walk_acc {
            Some(a) => a,
            None => {
                let a = self.tmp();
                self.line(format!("{a} = alloca [{accs} x float], align 16"));
                a
            }
        };
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        let seed = self.split_seed(&seed);
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }
        // tensor-cores-2 `bf16x3n`: the f32 operand cut into three exact
        // bf16 parts as its fragment is read; whether this thread cut an
        // operand the guard takes (the cold path after the `k` loop).
        let cut = match g.split {
            Some(SplitKind::Bf16x3n) => 3,
            Some(SplitKind::Bf16x2n) => 2,
            _ => 0,
        };
        let inf_slot = (cut > 0).then(|| {
            let f = self.tmp();
            self.line(format!("{f} = alloca i32, align 4"));
            self.line(format!("store i32 0, ptr {f}, align 4"));
            f
        });
        // A split form's flush depth: the k-steps its zeroed chain carries
        // before the round-to-nearest add ([`split_flush`]).
        let flush = match self.rungs.split_issue {
            true => split_flush(&g, s.k, self.splitk.max(1), &self.profile_gpu),
            false => 1,
        };

        // cp.async: (is_a, global element index at k0 = 0, shared byte in
        // stage 0), one per 16-byte chunk this thread copies, A then B.
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        let (la, lb) = (GPU_VEC_BYTES / ea, GPU_VEC_BYTES / eb);
        for p in 0..bm * bk / la / threads {
            let li = self.op("add", &tid, p * threads);
            let r = self.op("udiv", &li, bk / la);
            let kc0 = self.op("urem", &li, bk / la);
            let kc = self.op("mul", &kc0, la);
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let g0 = self.op("add", &gr, &kc);
            let gsrc = self.op("add", &g0, s.a.base);
            let gsrc = self.at_z(gsrc, kext * s.a.ck);
            let sr = self.op("mul", &r, lda);
            let se = self.op("add", &sr, &kc);
            copies.push((true, gsrc, self.op("mul", &se, ea)));
        }
        for p in 0..bk * bn / lb / threads {
            let li = self.op("add", &tid, p * threads);
            // `bt`: rows `n`, 16-byte chunks along `k` (as `a`'s).
            let chunks = if g.bt { bk / lb } else { bn / lb };
            let r = self.op("udiv", &li, chunks);
            let c0 = self.op("urem", &li, chunks);
            let cc = self.op("mul", &c0, lb);
            let g0 = if g.bt {
                let gn = self.op("add", &col0, &r);
                let gr = self.op("mul", &gn, s.b.clane);
                self.op("add", &gr, &cc)
            } else {
                let gk = self.op("mul", &r, s.b.ck);
                let gc = self.op("add", &col0, &cc);
                self.op("add", &gk, &gc)
            };
            let gsrc = self.op("add", &g0, s.b.base);
            let gsrc = self.at_z(gsrc, kext * s.b.ck);
            let sk = self.op("mul", &r, ldb);
            let sn = self.op("add", &sk, &cc);
            let sb = self.op("mul", &sn, eb);
            copies.push((false, gsrc, self.op("add", &sb, ab)));
        }
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            let k0 = k.op("mul", tile, bk);
            let ka = k.op("mul", &k0, s.a.ck);
            let kb = k.op("mul", &k0, s.b.ck);
            let so = k.op("mul", stage, stage_bytes);
            for (is_a, gsrc, sdst) in &copies {
                let (arr, off, e) = if *is_a {
                    ("%a", &ka, ea)
                } else {
                    ("%b", &kb, eb)
                };
                let gi = k.op("add", gsrc, off);
                let gp = k.gep_ty(1, ety(e), arr, &gi);
                let si = k.op("add", sdst, &so);
                let sp = k.gep_ty(3, "i8", smem, &si);
                k.line(format!(
                    "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                ));
            }
        };
        // `mma-walk`: the piece's first `stages − 1` panels, each where the
        // piece has it (a piece may hold fewer panels than the ring), after
        // every warp has read the previous piece's last stage.
        let (pre, p0, p1) = match &seg {
            Some(sg) => {
                self.line("call void @llvm.nvvm.barrier0()");
                let mut pre = "seg".to_string();
                for st in 0..g.stages - 1 {
                    let pan = self.op("add", &sg.p0, st);
                    let has = self.tmp();
                    self.line(format!("{has} = icmp ult i64 {pan}, {}", sg.p1));
                    self.line(format!("br i1 {has}, label %pro{st}, label %pro{st}j"));
                    self.out.push_str(&format!("pro{st}:\n"));
                    let stage = self.op("urem", &pan, g.stages);
                    issue(self, &pan, &stage);
                    self.line(format!("br label %pro{st}j"));
                    self.out.push_str(&format!("pro{st}j:\n"));
                    self.line("call void @llvm.nvvm.cp.async.commit.group()");
                    pre = format!("pro{st}j");
                }
                (pre, sg.p0.clone(), sg.p1.clone())
            }
            None => {
                for st in 0..(g.stages - 1).min(nk) {
                    issue(self, &st.to_string(), &st.to_string());
                    self.line("call void @llvm.nvvm.cp.async.commit.group()");
                }
                for _ in nk..g.stages - 1 {
                    self.line("call void @llvm.nvvm.cp.async.commit.group()");
                }
                ("entry".to_string(), "0".to_string(), nk.to_string())
            }
        };
        // Per-thread fragment origins inside a stage, in bytes. `ldmatrix`:
        // lane `l` gives row `l % 8` of matrix `l / 8`.
        let q = self.op("udiv", &lane, 8);
        let i8l = self.op("urem", &lane, 8);
        let qr = self.op("urem", &q, 2);
        let qk = self.op("udiv", &q, 2);
        let wrow = self.op("mul", &wr, g.wm());
        let wcol = self.op("mul", &wc, g.wn());
        let a_org = if ea == 2 {
            // a: matrices (rows 0-7 | 8-15) × (k 0-7 | 8-15), row-major.
            let r = self.op("mul", &qr, 8);
            let r = self.op("add", &r, &i8l);
            let r = self.op("add", &r, &wrow);
            let r = self.op("mul", &r, lda);
            let kq = self.op("mul", &qk, 8);
            let o = self.op("add", &r, &kq);
            self.op("mul", &o, 2)
        } else {
            // a: the pair (row g, k 2t..2t+1).
            let r = self.op("add", &wrow, &grp);
            let r = self.op("mul", &r, lda);
            let t2 = self.op("mul", &t4, 2);
            let o = self.op("add", &r, &t2);
            self.op("mul", &o, 4)
        };
        let b_org = if g.bt {
            // `n × k`: as `a` — `ldmatrix` matrix `q` is n rows `(q / 2)·8
            // ..+8` (two n tiles) at k `(q % 2)·8`; an f32 pair (n g, k 2t).
            let (n, kq, e) = if eb == 2 {
                let n = self.op("mul", &qk, MMA_N);
                let n = self.op("add", &n, &i8l);
                (n, self.op("mul", &qr, 8), 2)
            } else {
                (grp.clone(), self.op("mul", &t4, 2), 4)
            };
            let n = self.op("add", &n, &wcol);
            let r = self.op("mul", &n, ldb);
            let o = self.op("add", &r, &kq);
            let o = self.op("mul", &o, e);
            self.op("add", &o, ab)
        } else if eb == 2 {
            // b (`.trans`): matrix `q` is k rows `(q % 2)·8 ..+8` of n tile
            // `q / 2`; lane `l` gives its row `l % 8`.
            let kr = self.op("mul", &qr, 8);
            let kr = self.op("add", &kr, &i8l);
            let kr = self.op("mul", &kr, ldb);
            let nq = self.op("mul", &qk, MMA_N);
            let n = self.op("add", &wcol, &nq);
            let o = self.op("add", &kr, &n);
            let o = self.op("mul", &o, 2);
            self.op("add", &o, ab)
        } else {
            // b: the words (k 2t + j, n g).
            let kr = self.op("mul", &t4, 2);
            let kr = self.op("mul", &kr, ldb);
            let n = self.op("add", &wcol, &grp);
            let o = self.op("add", &kr, &n);
            let o = self.op("mul", &o, 4);
            self.op("add", &o, ab)
        };
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [{p0}, %{pre}], [{itn}, %issued]"));
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            g.stages - 2
        ));
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, g.stages - 1);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {p1}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        issue(self, &nxt, &nst);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_bytes);
        let a_base = self.op("add", &cbase, &a_org);
        let b_base = self.op("add", &cbase, &b_org);
        // `ldmatrix` of `n` b16 matrices at shared byte `base + off`.
        let ldsm = |k: &mut Self, base: &str, off: u64, n: usize, trans: bool| -> Vec<String> {
            let i = k.op("add", base, off);
            let p = k.gep_ty(3, "i8", smem, &i);
            let a32 = k.tmp();
            k.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
            let ty = vec!["i32"; n].join(", ");
            let outs = (0..n)
                .map(|j| format!("${j}"))
                .collect::<Vec<_>>()
                .join(",");
            let cons = vec!["=r"; n].join(",");
            let m = k.tmp();
            k.line(format!(
                "{m} = call {{ {ty} }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x{n}{}.shared.b16 {{{outs}}}, [${n}];\", \"{cons},r\"(i32 {a32})",
                if trans { ".trans" } else { "" }
            ));
            (0..n)
                .map(|j| {
                    let x = k.tmp();
                    k.line(format!("{x} = extractvalue {{ {ty} }} {m}, {j}"));
                    x
                })
                .collect()
        };
        // Two f32 values rounded once into one register of the format, the
        // lower `k` in the low half.
        let pack = |k: &mut Self, lo: &str, hi: &str| -> String {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rn.{ptx}x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {hi}, float {lo})"
            ));
            r
        };
        let lds = |k: &mut Self, base: &str, off: u64| -> String {
            let i = k.op("add", base, off);
            let p = k.gep_ty(3, "i8", smem, &i);
            let x = k.tmp();
            k.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
            x
        };
        // The fragment registers of an f32 pair (the lower `k` in the low
        // half): rounded once to the format (the member's realization), or
        // — `bf16x3n` — its three exact bf16 parts, largest first.
        let mut infs: Vec<String> = Vec::new();
        let mut frag = |k: &mut Self, lo: &str, hi: &str| -> Vec<String> {
            if cut > 0 {
                let (parts, guard) = k.split_bf16x2(lo, hi, cut);
                infs.push(guard);
                return parts;
            }
            vec![pack(k, lo, hi)]
        };
        // Each tile's zeroed chain under a split form, carried `flush`
        // k-steps, then added into its accumulator by round-to-nearest adds.
        let zero = || vec!["0.0".to_string(); 4];
        let mut chain: Vec<Vec<String>> = vec![zero(); mt * nt];
        for kk in 0..bk / mk {
            let k16 = kk * mk;
            // af[mi][part] / bf[ni][part]: one part (a narrow panel, or the
            // member's rounding), three (`bf16x3n`'s f32 operand).
            let mut af: Vec<Vec<[String; 4]>> = Vec::new();
            for mi in 0..mt as u64 {
                if ea == 2 {
                    let f = ldsm(self, &a_base, (mi * MMA_M * lda + k16) * 2, 4, false);
                    af.push(vec![[
                        f[0].clone(),
                        f[1].clone(),
                        f[2].clone(),
                        f[3].clone(),
                    ]]);
                    continue;
                }
                let mut q: Vec<Vec<String>> = Vec::new();
                for (rr, kc) in [(0u64, 0u64), (8, 0), (0, 8), (8, 8)] {
                    let off = ((mi * MMA_M + rr) * lda + k16 + kc) * 4;
                    let i = self.op("add", &a_base, off);
                    let p = self.gep_ty(3, "i8", smem, &i);
                    let v = self.tmp();
                    self.line(format!(
                        "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                    ));
                    let (x0, x1) = (self.tmp(), self.tmp());
                    self.line(format!("{x0} = extractelement <2 x float> {v}, i64 0"));
                    self.line(format!("{x1} = extractelement <2 x float> {v}, i64 1"));
                    q.push(frag(self, &x0, &x1));
                }
                af.push(
                    (0..q[0].len())
                        .map(|pp| {
                            [
                                q[0][pp].clone(),
                                q[1][pp].clone(),
                                q[2][pp].clone(),
                                q[3][pp].clone(),
                            ]
                        })
                        .collect(),
                );
            }
            let mut bf: Vec<Vec<[String; 2]>> = Vec::new();
            let pairs = |q: Vec<Vec<String>>| -> Vec<[String; 2]> {
                (0..q[0].len())
                    .map(|pp| [q[0][pp].clone(), q[1][pp].clone()])
                    .collect()
            };
            let mut ni = 0u64;
            while ni < nt as u64 {
                let off = match g.bt {
                    true => ni * MMA_N * ldb + k16,
                    false => k16 * ldb + ni * MMA_N,
                };
                if g.bt && eb == 4 {
                    let mut q = Vec::new();
                    for kc in [0u64, 8] {
                        let i = self.op("add", &b_base, (off + kc) * 4);
                        let p = self.gep_ty(3, "i8", smem, &i);
                        let v = self.tmp();
                        self.line(format!(
                            "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                        ));
                        let (x0, x1) = (self.tmp(), self.tmp());
                        self.line(format!("{x0} = extractelement <2 x float> {v}, i64 0"));
                        self.line(format!("{x1} = extractelement <2 x float> {v}, i64 1"));
                        q.push(frag(self, &x0, &x1));
                    }
                    bf.push(pairs(q));
                    ni += 1;
                    continue;
                }
                if eb == 2 {
                    let pair = ni + 1 < nt as u64;
                    let f = ldsm(self, &b_base, off * 2, if pair { 4 } else { 2 }, !g.bt);
                    bf.push(vec![[f[0].clone(), f[1].clone()]]);
                    if pair {
                        bf.push(vec![[f[2].clone(), f[3].clone()]]);
                        ni += 1;
                    }
                    ni += 1;
                    continue;
                }
                let mut q = Vec::new();
                for kc in [0u64, 8] {
                    let x0 = lds(self, &b_base, (off + kc * ldb) * 4);
                    let x1 = lds(self, &b_base, (off + (kc + 1) * ldb) * 4);
                    q.push(frag(self, &x0, &x1));
                }
                bf.push(pairs(q));
                ni += 1;
            }
            let flushes = g.split.is_some() && (kk + 1) % flush == 0;
            for mi in 0..mt {
                for (ni, bp) in bf.iter().enumerate() {
                    let base = (mi * nt + ni) * 4;
                    let ap = &af[mi];
                    if g.split.is_some() {
                        // f32-tensor-cores `split-tile`: the exact part
                        // products, smallest class first, into the tile's
                        // zeroed chain (narrow1: one; bf16x3n: three).
                        let parts = ap.len().max(bp.len());
                        let mut t = std::mem::take(&mut chain[mi * nt + ni]);
                        for pp in (0..parts).rev() {
                            let [a0, a1, a2, a3] = &ap[pp.min(ap.len() - 1)];
                            let [b0, b1] = &bp[pp.min(bp.len() - 1)];
                            t = self.mma16_call(ptx, [a0, a1, a2, a3], [b0, b1], &t);
                        }
                        if !flushes {
                            chain[mi * nt + ni] = t;
                            continue;
                        }
                        // The flush: one round-to-nearest add a cell.
                        for (q, d) in t.iter().enumerate() {
                            let p = acc_slot(self, base + q);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr {p}, align 4"));
                            let x = self.tmp();
                            self.line(format!("{x} = fadd float {v}, {d}"));
                            self.line(format!("store float {x}, ptr {p}, align 4"));
                        }
                        chain[mi * nt + ni] = zero();
                        continue;
                    }
                    let mut cur = Vec::new();
                    for q in 0..4 {
                        let p = acc_slot(self, base + q);
                        let v = self.tmp();
                        self.line(format!("{v} = load float, ptr {p}, align 4"));
                        cur.push((p, v));
                    }
                    let r = self.tmp();
                    let [a0, a1, a2, a3] = &ap[0];
                    let [b0, b1] = &bp[0];
                    self.line(format!(
                        "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{ptx}.{ptx}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {a0}, i32 {a1}, i32 {a2}, i32 {a3}, i32 {b0}, i32 {b1}, float {}, float {}, float {}, float {})",
                        cur[0].1, cur[1].1, cur[2].1, cur[3].1
                    ));
                    for (q, (p, _)) in cur.iter().enumerate() {
                        let d = self.tmp();
                        self.line(format!(
                            "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                        ));
                        self.line(format!("store float {d}, ptr {p}, align 4"));
                    }
                }
            }
        }
        if let Some(flag) = &inf_slot {
            self.or_guards(flag, &infs);
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {p1}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        // C fragments: (g, 2t..2t+1) and (g+8, 2t..2t+1) of every m16n8 tile.
        self.out.push_str("done:\n");
        if let Some(sg) = &seg {
            self.mma_walk_store(
                sg,
                &g,
                (&row0, &col0, &tid),
                (&wrow, &wcol, &grp, &t4),
                &acc_slot,
            );
            return;
        }
        let orow = self.op("add", &wrow, &row0);
        let orow = self.op("add", &orow, &grp);
        let ocol = self.op("add", &wcol, &col0);
        let t2 = self.op("mul", &t4, 2);
        let ocol = self.op("add", &ocol, &t2);
        if let Some(flag) = &inf_slot {
            // The narrow operand's non-finite elements: their cells
            // ([`Self::acc_nonfinite`]).
            let nf = self.acc_nonfinite(mt * nt * 4, &acc_slot);
            self.or_guards(flag, &[nf]);
            self.split_cold(flag, &orow, &ocol, &seed, kext, (mt, nt), &acc_slot);
        }
        for mi in 0..mt {
            for half in 0..2u64 {
                let r = self.op("add", &orow, mi as u64 * MMA_M + half * 8);
                let rr = self.op("mul", &r, s.c);
                let rc = self.op("add", &rr, &ocol);
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4 + half as usize * 2;
                    let mut xs = Vec::new();
                    for j in 0..2 {
                        let p = acc_slot(self, base + j);
                        let x = self.tmp();
                        self.line(format!("{x} = load float, ptr {p}, align 4"));
                        xs.push(x);
                    }
                    let oi = self.op("add", &rc, ni as u64 * MMA_N);
                    if self.epi.is_some() {
                        self.store_cells(&xs, &oi);
                        continue;
                    }
                    if self.splitk > 1 {
                        // `mma-combine`: as [`Self::mma`]'s.
                        let oi = self.slab(oi);
                        self.store_o(&xs, &oi);
                        continue;
                    }
                    let mut vec = "poison".to_string();
                    for (j, x) in xs.iter().enumerate() {
                        let nv = self.tmp();
                        self.line(format!(
                            "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                        ));
                        vec = nv;
                    }
                    let op = self.gep_ty(1, "float", "%o", &oi);
                    self.line(format!(
                        "store <2 x float> {vec}, ptr addrspace(1) {op}, align 8"
                    ));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// bf16 tensor-tile `mma-walk`: **a walked piece's store** — the C
    /// fragments of every `m16n8` tile at the piece's tile `(row0, col0)`
    /// (tile-local row `wrow + g (+ 8)`, column `wcol + 2t`). A piece that is
    /// its tile's every panel (`st_whole`) stores as the ordered kernel does
    /// (the group's members on an epilogue group). A piece of a shared tile
    /// (`st_part`) adds its partial sums: into the zeroed output (`red`), or —
    /// an epilogue group — into its tile's workspace slot, then takes the
    /// tile's ticket ([`Kernel::ticket_split`], [`Kernel::ticket`]): the last
    /// piece's block evaluates the members on the whole sums. Then the next
    /// piece ([`Kernel::finish`]).
    pub(super) fn mma_walk_store(
        &mut self,
        sg: &Segment,
        g: &GpuMmaTile,
        (row0, col0, tid): (&str, &str, &str),
        (wrow, wcol, grp, t4): (&str, &str, &str, &str),
        acc_slot: &impl Fn(&mut Self, usize) -> String,
    ) {
        let s = self.site;
        let (bm, bn) = (g.bm(), g.bn());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let threads = g.threads(self.profile_gpu.warp);
        let panels = s.k / g.bk;
        let lrow = self.op("add", wrow, grp);
        let t2 = self.op("mul", t4, 2);
        let lcol = self.op("add", wcol, &t2);
        let split = match self.ticket {
            true => Some(self.ticket_split(sg, panels, (bm, bn))),
            false => {
                let first = self.tmp();
                self.line(format!("{first} = icmp eq i64 {}, 0", sg.p0));
                let last = self.tmp();
                self.line(format!("{last} = icmp eq i64 {}, {panels}", sg.p1));
                let whole = self.tmp();
                self.line(format!("{whole} = and i1 {first}, {last}"));
                self.line(format!("br i1 {whole}, label %st_whole, label %st_part"));
                None
            }
        };
        for part in [false, true] {
            self.out
                .push_str(if part { "st_part:\n" } else { "st_whole:\n" });
            self.part = part;
            for mi in 0..mt {
                for half in 0..2u64 {
                    let lr = self.op("add", &lrow, mi as u64 * MMA_M + half * 8);
                    let r = self.op("add", row0, &lr);
                    let rr = self.op("mul", &r, s.c);
                    for ni in 0..nt {
                        let base = (mi * nt + ni) * 4 + half as usize * 2;
                        let mut xs = Vec::new();
                        for j in 0..2 {
                            let p = acc_slot(self, base + j);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            xs.push(x);
                        }
                        let lc = self.op("add", &lcol, ni as u64 * MMA_N);
                        if part && let Some(sp) = &split {
                            // The slot's adds as `red` (no value returned):
                            // `atomicrmw` ahead of the ticket's release fence
                            // is kept a returning `ATOMG` by ptxas — each
                            // thread then waits on its 128 adds in turn (the
                            // 2048 × 2048 × 8192 group: 800 µs against the
                            // `REDG` walk's 622); the fence orders the `red`s
                            // it follows.
                            let w = self.op("mul", &lr, bn);
                            let w = self.op("add", &w, &lc);
                            let wi = self.op("add", &sp.base, &w);
                            for (j, x) in xs.iter().enumerate() {
                                let i = self.plus(wi.clone(), j as u64);
                                let p = self.gep(1, "%w", &i);
                                self.line(format!(
                                    "call void asm sideeffect \"red.relaxed.gpu.global.add.f32 [$0], $1;\", \"l,f,~{{memory}}\"(ptr addrspace(1) {p}, float {x})"
                                ));
                            }
                            continue;
                        }
                        let c = self.op("add", col0, &lc);
                        let oi = self.op("add", &rr, &c);
                        if part {
                            self.store_o(&xs, &oi);
                            continue;
                        }
                        if self.epi.is_some() {
                            self.store_cells(&xs, &oi);
                            continue;
                        }
                        let mut vec = "poison".to_string();
                        for (j, x) in xs.iter().enumerate() {
                            let nv = self.tmp();
                            self.line(format!(
                                "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                            ));
                            vec = nv;
                        }
                        let op = self.gep_ty(1, "float", "%o", &oi);
                        self.line(format!(
                            "store <2 x float> {vec}, ptr addrspace(1) {op}, align 8"
                        ));
                    }
                }
            }
            if part && let Some(sp) = &split {
                self.ticket(sp, (bm, bn, threads), (row0, col0, tid));
            }
            self.line("br label %st_join");
        }
        self.part = false;
        self.out.push_str("st_join:\n");
        self.finish(Some(sg));
    }
}
