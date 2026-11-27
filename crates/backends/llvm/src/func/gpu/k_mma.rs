//! `Kernel` TF32 warp matrix kernel.

use super::*;

impl Kernel<'_> {
    /// The tensor-core kernel (TF32 face): block `(bx, by)` owns a `bm × bn`
    /// output tile; warp `(wr, wc)` a `wm × wn` tile of `mt × nt` m16n8 tiles
    /// whose f32 accumulators live in registers. The reduction is staged in
    /// `stages` shared-memory buffers filled by `cp.async` — panel `it + stages
    /// - 1` is in flight while panel `it` is consumed — and each `bk` panel is
    /// `bk / 8` steps of `mma.sync.m16n8k8`, operands rounded with `cvt.rna`.
    ///
    /// Fragment layouts are the PTX ISA's for m16n8k8 (`group = lane / 4`,
    /// `t = lane % 4`): A `a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4);
    /// B `b0, b1` = (k t, n g), (k t+4, n g); C `c0..c3` = (g, 2t), (g, 2t+1),
    /// (g+8, 2t), (g+8, 2t+1).
    pub(super) fn mma(&mut self, name: &str, g: GpuMmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let stage_words = g.stage_words();
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

        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        let warp_id = self.op("udiv", &tid, warp);
        let lane = self.op("urem", &tid, warp);
        let wr = self.op("udiv", &warp_id, g.warps_x);
        let wc = self.op("urem", &warp_id, g.warps_x);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = mt * nt * 4;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
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
        // f32-tensor-cores `split-tile` (tf32x3): whether this thread split
        // an infinite operand (the cold path after the `k` loop).
        let split = g.split == Some(SplitKind::Tf32x3);
        let inf_slot = split.then(|| {
            let f = self.tmp();
            self.line(format!("{f} = alloca i32, align 4"));
            self.line(format!("store i32 0, ptr {f}, align 4"));
            f
        });

        // cp.async coordinates: (global index at k0 = 0, shared word in stage 0),
        // one per 16-byte chunk this thread copies, A then B.
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        let (ac, bc) = (bk / 4, bn / 4);
        // site-geometry `tf32-radix`: a digit-affine operand's copies, one
        // element each — `(is_a, row/lane digits, shared word, its k)`; each
        // panel adds the `k` digits of `k0 + k` (`digits`).
        let rx = s.radix.as_deref();
        let (ra, rb, ro) = (
            rx.and_then(|x| x.a.as_ref()),
            rx.and_then(|x| x.b.as_ref()),
            rx.and_then(|x| x.out.as_ref()),
        );
        let mut radix: Vec<(bool, String, String, String)> = Vec::new();
        if let Some(m) = ra {
            for p in 0..bm * bk / threads {
                let li = self.op("add", &tid, p * threads);
                let r = self.op("udiv", &li, bk);
                let kc = self.op("urem", &li, bk);
                let grow = self.op("add", &row0, &r);
                let g = self.radix_at(m, &grow, "0");
                let sr = self.op("mul", &r, lda);
                radix.push((true, g, self.op("add", &sr, &kc), kc));
            }
        }
        if let Some(m) = rb {
            for p in 0..bk * bn / threads {
                let li = self.op("add", &tid, p * threads);
                let kr = self.op("udiv", &li, bn);
                let nc = self.op("urem", &li, bn);
                let gc = self.op("add", &col0, &nc);
                // `head-group`: `b`'s row digits are the block's batch.
                let g = self.radix_at(m, &row0, &gc);
                let sk = self.op("mul", &kr, ldb);
                let sn = self.op("add", &sk, &nc);
                radix.push((false, g, self.op("add", &sn, bm * lda), kr));
            }
        }
        let a_chunks = if ra.is_some() {
            0
        } else {
            bm * bk / 4 / threads
        };
        for p in 0..a_chunks {
            let li = self.op("add", &tid, p * threads);
            let r = self.op("udiv", &li, ac);
            let kc0 = self.op("urem", &li, ac);
            let kc = self.op("mul", &kc0, 4);
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let g0 = self.op("add", &gr, &kc);
            let gsrc = self.op("add", &g0, s.a.base);
            let gsrc = self.at_z(gsrc, kext * s.a.ck);
            let sr = self.op("mul", &r, lda);
            let sdst = self.op("add", &sr, &kc);
            copies.push((true, gsrc, sdst));
        }
        let b_chunks = if rb.is_some() {
            0
        } else {
            bk * bn / 4 / threads
        };
        for p in 0..b_chunks {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let gsrc = self.at_z(gsrc, kext * s.b.ck);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, bm * lda);
            copies.push((false, gsrc, sdst));
        }
        // TMA `bulk-stage`: warp 0 copies each panel row with the bulk-copy
        // engine, a stage's barrier counting the bytes ([`Self::issue_bulk`]).
        let bulk = bulk_stage(s, &g, &self.profile_gpu, &self.rungs);
        let bars = bulk.then(|| {
            let b = self.gep(3, smem, &(g.stages * stage_words).to_string());
            let t32 = self.tmp();
            self.line(format!("{t32} = trunc i64 {tid} to i32"));
            for st in 0..g.stages {
                let a = self.bar_at(&b, &format!("{}", st * BULK_BAR_BYTES));
                self.asm_args(
                    "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared::cta.b64 [$0], $1; }",
                    "r,r,r",
                    &[format!("i32 {a}"), "i32 1".into(), format!("i32 {t32}")],
                );
            }
            self.asm("fence.mbarrier_init.release.cluster;");
            self.line("call void @llvm.nvvm.barrier0()");
            b
        });
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            if let Some(b) = &bars {
                k.issue_bulk(&g, b, &tid, &row0, &col0, tile, stage);
                return;
            }
            k.issue16(&copies, tile, stage, bk, stage_words);
            if radix.is_empty() {
                return;
            }
            let k0 = k.op("mul", tile, bk);
            let so = k.op("mul", stage, stage_words);
            for (is_a, g, sdst, kc) in &radix {
                let (arr, m) = if *is_a { ("%a", ra) } else { ("%b", rb) };
                let kv = k.op("add", &k0, kc);
                let kd = k.digits(&m.expect("a radix operand").k, &kv);
                let gi = k.op("add", g, &kd);
                let gp = k.gep(1, arr, &gi);
                let si = k.op("add", sdst, &so);
                let sp = k.gep(3, "@mapal_smem", &si);
                k.line(format!(
                    "call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                ));
            }
        };
        for st in 0..(g.stages - 1).min(nk) {
            issue(self, &st.to_string(), &st.to_string());
            if !bulk {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
        }
        for _ in nk..g.stages - 1 {
            if !bulk {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
        }
        // Per-thread fragment origins inside a stage.
        let a_r = self.op("mul", &wr, g.wm());
        let a_r = self.op("add", &a_r, &grp);
        let a_r = self.op("mul", &a_r, lda);
        let a_org = self.op("add", &a_r, &t4);
        let b_r = self.op("mul", &t4, ldb);
        let b_c = self.op("mul", &wc, g.wn());
        let b_o = self.op("add", &b_r, &b_c);
        let b_o = self.op("add", &b_o, &grp);
        let b_org = self.op("add", &b_o, bm * lda);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %issued]"));
        match &bars {
            // Panel `it`'s stage: its barrier's phase `it / stages` (parity).
            Some(b) => {
                let st = self.op("urem", &it, g.stages);
                let off = self.op("mul", &st, BULK_BAR_BYTES);
                let bar = self.bar_at(b, &off);
                let ph = self.op("udiv", &it, g.stages);
                let ph1 = self.op("and", &ph, 1);
                let par = self.tmp();
                self.line(format!("{par} = trunc i64 {ph1} to i32"));
                self.line("br label %mbw");
                self.out.push_str("mbw:\n");
                let ok = self.tmp();
                self.line(format!(
                    "{ok} = call i32 asm sideeffect \"{{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.u32 $0, 1, 0, p; }}\", \"=r,r,r,~{{memory}}\"(i32 {bar}, i32 {par})"
                ));
                let done = self.tmp();
                self.line(format!("{done} = icmp ne i32 {ok}, 0"));
                self.line(format!("br i1 {done}, label %mbd, label %mbw"));
                self.out.push_str("mbd:\n");
            }
            None => self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                g.stages - 2
            )),
        }
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, g.stages - 1);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        issue(self, &nxt, &nst);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        if !bulk {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        let a_base = self.op("add", &cbase, &a_org);
        let b_base = self.op("add", &cbase, &b_org);
        let lds_tf32 = |k: &mut Self, base: &str, off: u64| {
            let i = k.op("add", base, off);
            let p = k.gep(3, smem, &i);
            let x = k.tmp();
            k.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
            ));
            r
        };
        // `ldmatrix.x4` (rung `ldmatrix`): the four 8×8 b16 matrices of one
        // m16k8 TF32 fragment — rows 0–7 / 8–15 × k 0–3 / 4–7 — are exactly
        // a0..a3 when each 32-bit element is read as a b16 pair. Lane `l`
        // supplies the row address of matrix `l / 8`, row `l % 8`.
        let lm_base = if self.rungs.ldmatrix {
            let q = self.op("udiv", &lane, 8);
            let i = self.op("urem", &lane, 8);
            let qr = self.op("urem", &q, 2);
            let qk = self.op("udiv", &q, 2);
            let r = self.op("mul", &qr, 8);
            let r = self.op("add", &r, &i);
            let wrow = self.op("mul", &wr, g.wm());
            let r = self.op("add", &r, &wrow);
            let r = self.op("mul", &r, lda);
            let kq = self.op("mul", &qk, 4);
            let o = self.op("add", &r, &kq);
            Some(self.op("add", &cbase, &o))
        } else {
            None
        };
        // A split form's flush depth and each tile's zeroed chain.
        let flush = match self.rungs.split_issue {
            true => split_flush(&g, s.k, self.splitk.max(1), &self.profile_gpu),
            false => 1,
        };
        let mut chain: Vec<Vec<String>> = vec![vec!["0.0".to_string(); 4]; mt * nt];
        for kk in 0..bk / MMA_K {
            let k8 = kk * MMA_K;
            if let Some(flag) = &inf_slot {
                // f32-tensor-cores `split-tile` (`mapal_ir::split`, tf32x3):
                // each element's `hi = rna(x)`, `lo = rna(x − hi)`; per m16n8
                // tile `lo·hi`, `hi·lo`, `hi·hi` from a zeroed accumulator,
                // then one round-to-nearest add into the f32 accumulator.
                let mut infs = Vec::new();
                let mut word = |k: &mut Self, x: String| -> [String; 2] {
                    let (h, l, inf) = match k.rungs.split_issue {
                        true => k.split_tf32(&x),
                        false => k.split_tf32_cvt(&x),
                    };
                    infs.push(inf);
                    [h, l]
                };
                let mut af = Vec::new();
                for mi in 0..mt as u64 {
                    let r = mi * MMA_M * lda + k8;
                    let xs: Vec<String> = match &lm_base {
                        Some(lb) => {
                            let i = self.op("add", lb, r);
                            let p = self.gep(3, smem, &i);
                            let a32 = self.tmp();
                            self.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
                            let m = self.tmp();
                            self.line(format!(
                                "{m} = call {{ i32, i32, i32, i32 }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0,$1,$2,$3}}, [$4];\", \"=r,=r,=r,=r,r\"(i32 {a32})"
                            ));
                            (0..4)
                                .map(|j| {
                                    let x = self.tmp();
                                    self.line(format!(
                                        "{x} = extractvalue {{ i32, i32, i32, i32 }} {m}, {j}"
                                    ));
                                    let f = self.tmp();
                                    self.line(format!("{f} = bitcast i32 {x} to float"));
                                    f
                                })
                                .collect()
                        }
                        None => [r, r + 8 * lda, r + 4, r + 8 * lda + 4]
                            .iter()
                            .map(|&o| {
                                let i = self.op("add", &a_base, o);
                                let p = self.gep(3, smem, &i);
                                let x = self.tmp();
                                self.line(format!(
                                    "{x} = load float, ptr addrspace(3) {p}, align 4"
                                ));
                                x
                            })
                            .collect(),
                    };
                    let parts: Vec<[String; 2]> = xs.into_iter().map(|x| word(self, x)).collect();
                    af.push(parts);
                }
                let mut bf = Vec::new();
                for ni in 0..nt as u64 {
                    let c = k8 * ldb + ni * MMA_N;
                    let parts: Vec<[String; 2]> = [c, c + 4 * ldb]
                        .iter()
                        .map(|&o| {
                            let i = self.op("add", &b_base, o);
                            let p = self.gep(3, smem, &i);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
                            word(self, x)
                        })
                        .collect();
                    bf.push(parts);
                }
                // The tile's zeroed chain carries `flush` k-steps before its
                // round-to-nearest add (tensor-cores-2, [`split_flush`]).
                let flushes = (kk + 1) % flush == 0;
                for mi in 0..mt {
                    for ni in 0..nt {
                        let a = |h: usize| -> Vec<String> {
                            af[mi].iter().map(|w| w[h].clone()).collect()
                        };
                        let b = |h: usize| -> Vec<String> {
                            bf[ni].iter().map(|w| w[h].clone()).collect()
                        };
                        // Smallest class first: lo·hi, hi·lo, hi·hi.
                        let mut t = std::mem::take(&mut chain[mi * nt + ni]);
                        for (x, y) in [(a(1), b(0)), (a(0), b(1)), (a(0), b(0))] {
                            t = self.mma_tf32_call(&x, &y, &t);
                        }
                        if !flushes {
                            chain[mi * nt + ni] = t;
                            continue;
                        }
                        let base = (mi * nt + ni) * 4;
                        for (q, tq) in t.iter().enumerate() {
                            let p = acc_slot(self, base + q);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr {p}, align 4"));
                            let d = self.tmp();
                            self.line(format!("{d} = fadd float {v}, {tq}"));
                            self.line(format!("store float {d}, ptr {p}, align 4"));
                        }
                        chain[mi * nt + ni] = vec!["0.0".to_string(); 4];
                    }
                }
                self.or_guards(flag, &infs);
                continue;
            }
            let mut af = Vec::new();
            for mi in 0..mt as u64 {
                let r = mi * MMA_M * lda + k8;
                if let Some(lb) = &lm_base {
                    let i = self.op("add", lb, r);
                    let p = self.gep(3, smem, &i);
                    let a32 = self.tmp();
                    self.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
                    let m = self.tmp();
                    self.line(format!(
                        "{m} = call {{ i32, i32, i32, i32 }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0,$1,$2,$3}}, [$4];\", \"=r,=r,=r,=r,r\"(i32 {a32})"
                    ));
                    let mut q = Vec::new();
                    for j in 0..4 {
                        let x = self.tmp();
                        self.line(format!(
                            "{x} = extractvalue {{ i32, i32, i32, i32 }} {m}, {j}"
                        ));
                        let r = self.tmp();
                        self.line(format!(
                            "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,r\"(i32 {x})"
                        ));
                        q.push(r);
                    }
                    af.push([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
                    continue;
                }
                af.push([
                    lds_tf32(self, &a_base, r),
                    lds_tf32(self, &a_base, r + 8 * lda),
                    lds_tf32(self, &a_base, r + 4),
                    lds_tf32(self, &a_base, r + 8 * lda + 4),
                ]);
            }
            let mut bf = Vec::new();
            for ni in 0..nt as u64 {
                let c = k8 * ldb + ni * MMA_N;
                bf.push([
                    lds_tf32(self, &b_base, c),
                    lds_tf32(self, &b_base, c + 4 * ldb),
                ]);
            }
            for mi in 0..mt {
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4;
                    let mut cur = Vec::new();
                    for q in 0..4 {
                        let p = acc_slot(self, base + q);
                        let v = self.tmp();
                        self.line(format!("{v} = load float, ptr {p}, align 4"));
                        cur.push((p, v));
                    }
                    let r = self.tmp();
                    let [a0, a1, a2, a3] = &af[mi];
                    let [b0, b1] = &bf[ni];
                    self.line(format!(
                        "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {a0}, i32 {a1}, i32 {a2}, i32 {a3}, i32 {b0}, i32 {b1}, float {}, float {}, float {}, float {})",
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
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        // C fragments: (g, 2t..2t+1) and (g+8, 2t..2t+1) of every m16n8 tile.
        self.out.push_str("done:\n");
        let orow = self.op("mul", &wr, g.wm());
        let orow = self.op("add", &orow, &row0);
        let orow = self.op("add", &orow, &grp);
        let ocol = self.op("mul", &wc, g.wn());
        let ocol = self.op("add", &ocol, &col0);
        let t2 = self.op("mul", &t4, 2);
        let ocol = self.op("add", &ocol, &t2);
        if let Some(flag) = &inf_slot {
            self.split_cold(flag, &orow, &ocol, &seed, kext, (mt, nt), &acc_slot);
        }
        for mi in 0..mt {
            for half in 0..2u64 {
                let r = self.op("add", &orow, mi as u64 * MMA_M + half * 8);
                let rr = self.op("mul", &r, s.c);
                let rc = self.op("add", &rr, &ocol);
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4 + half as usize * 2;
                    // `tf32-radix`: a digit-affine output, cell by cell.
                    if let Some(o) = ro {
                        for j in 0..2u64 {
                            let p = acc_slot(self, base + j as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            let cc = self.op("add", &ocol, ni as u64 * MMA_N + j);
                            let oi = self.radix_at(o, &r, &cc);
                            if self.epi.is_some() {
                                self.store_cells(&[x], &oi);
                            } else {
                                let op = self.gep(1, "%o", &oi);
                                self.line(format!(
                                    "store float {x}, ptr addrspace(1) {op}, align 4"
                                ));
                            }
                        }
                        continue;
                    }
                    if self.epi.is_some() {
                        let mut xs = Vec::new();
                        for j in 0..2 {
                            let p = acc_slot(self, base + j);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            xs.push(x);
                        }
                        let oi = self.op("add", &rc, ni as u64 * MMA_N);
                        self.store_cells(&xs, &oi);
                        continue;
                    }
                    if self.splitk > 1 {
                        // `mma-combine`: the piece's partials into its slab
                        // (workspace) or added into the output (atomic).
                        let mut xs = Vec::new();
                        for j in 0..2 {
                            let p = acc_slot(self, base + j);
                            let x = self.tmp();
                            self.line(format!("{x} = load float, ptr {p}, align 4"));
                            xs.push(x);
                        }
                        let oi = self.op("add", &rc, ni as u64 * MMA_N);
                        let oi = self.slab(oi);
                        self.store_o(&xs, &oi);
                        continue;
                    }
                    let mut vec = "poison".to_string();
                    for j in 0..2 {
                        let p = acc_slot(self, base + j);
                        let x = self.tmp();
                        self.line(format!("{x} = load float, ptr {p}, align 4"));
                        let nv = self.tmp();
                        self.line(format!(
                            "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                        ));
                        vec = nv;
                    }
                    let oi = self.op("add", &rc, ni as u64 * MMA_N);
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!(
                        "store <2 x float> {vec}, ptr addrspace(1) {op}, align 8"
                    ));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }
}
