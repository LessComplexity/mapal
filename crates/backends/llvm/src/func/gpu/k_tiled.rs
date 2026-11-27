//! `Kernel` register/shared-memory tiled kernel.

use super::*;

impl Kernel<'_> {
    /// The tiled kernel: block `(bx, by)` owns the `bm × bn` output tile at
    /// `(by·bm, bx·bn)`; thread `(trow, tcol)` owns a `tm × tn` register tile
    /// inside it. Per `bk` step of the reduction, the block stages
    ///
    /// - `A[rows][k0 .. k0+bk]` **transposed** into `As[kk][m]`, so a thread's
    ///   `tm` rows at one `kk` are contiguous (one vector read each), and
    /// - `B[k0 .. k0+bk][cols]` as `Bs[kk][n]`,
    ///
    /// with `ld.global.nc.v4`, then a barrier (the morphism that completes the
    /// `g2s` transmission — plan-s41 §1.4), then every thread folds `bk` steps
    /// into its registers, then a barrier before the panels are overwritten.
    pub(super) fn tiled(&mut self, name: &str, g: GpuTile) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        let lanes = GPU_VEC_BYTES / e;
        let vt = format!("<{lanes} x {t}>");
        let (bm, bn, bk, tm, tn, threads) = (g.bm(), g.bn(), g.bk, g.tm, g.tn, g.threads());
        // S62 `splitk`: the `k` extent this kernel folds — partition `z`'s
        // whole panels (`launch` derives `S` so `bk | k / S`), its copies'
        // global sources moved `z·kext` along `k`.
        let kext = s.k / self.splitk;

        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let tid = self.sreg("tid.x");
        let accs = (tm * tn) as usize;
        let acc = self.tmp();
        // S63 `streamk`: the accumulators live across pieces (the entry block,
        // where they promote to registers); each piece re-seeds them.
        let seg = self.streamk.map(|iters| {
            self.line(format!(
                "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
            ));
            self.segment(iters, kext / bk, bk)
        });
        let (bx, by) = match &seg {
            Some(sg) => (sg.bx.clone(), sg.by.clone()),
            None => self.block_ids(),
        };
        let (kbeg, kend) = match &seg {
            Some(sg) => (sg.kbeg.clone(), sg.kend.clone()),
            None => ("0".to_string(), kext.to_string()),
        };
        // Warp shape (S52 rung `warp`): lay a warp's 32 lanes out as
        // `warp / wx` rows × `wx` columns of threads, `wx` = the lanes one
        // shared-memory wavefront serves with vector loads (128 B / 16 B = 8).
        // A warp then reads 4 + 8 distinct thread fragments per `k` step
        // instead of 2 + 16 — two-thirds of the bytes. Off arm / a block too
        // narrow for it: row-major threads.
        let (rungs, col, warp) = (self.rungs, self.col, self.profile_gpu.warp);
        let (trow, tcol) = thread_coords(self, &tid, &g, rungs.warp, col, warp);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);
        let smem = "@mapal_smem";
        let streamed = g.stages >= 2;
        // ADR-0043: a `cp.async` copy cannot widen: a streamed narrow `b`
        // panel stays narrow in shared memory (`narrow-async`), read and
        // widened per fragment; a narrow `a` is register-staged.
        assert!(
            !(streamed && s.a.narrow.is_some()),
            "a narrow `a` panel is register-staged"
        );
        let nb = s.b.narrow.filter(|_| streamed);
        let kmaj = g.kmajor && !self.col;
        let sa = panel_sa(&g, kmaj);
        let bs_off = g.a_elems();
        let stage_elems = g.stage_elems();

        if seg.is_none() {
            self.line(format!(
                "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
            ));
        }
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x {t}], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        let seed = self.split_seed(&seed);
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store {t} {seed}, ptr {p}, align {e}"));
        }

        // Loop-invariant load coordinates, one set per load this thread issues:
        // `(global index at k0 = 0, + constant, shared index, + constant)`.
        let (a_vec, b_vec) = (vector_a(s), vector_b(s));
        // sweep-attention-2 `kmajor-b`: `b`'s copies along its k-major rows.
        let kvb = rungs.kmajor_b && kvec_b(s, &g);
        let swz = kvec_swizzle(s, &g, &rungs);
        let [(a_map, a_loads), (b_map, b_loads)] = lane_maps(s, &g, kmaj, kvb);
        // recurrence-precision `radix-tile`: a mixed-radix operand's copy holds
        // its row (lane) digits at `k = 0` and its own `k` offset `kc`; each
        // panel adds the `k` digits of `k0 + kc` (not affine: no constant
        // delta between copies, no `k0·ck` stride).
        let rx = s.radix.as_deref();
        let (ra, rb, ro) = (
            rx.and_then(|x| x.a.as_ref()),
            rx.and_then(|x| x.b.as_ref()),
            rx.and_then(|x| x.out.as_ref()),
        );
        let a_kc = std::cell::RefCell::new(Vec::new());
        let b_kc = std::cell::RefCell::new(Vec::new());
        let a_src = self.copies(
            a_map,
            a_loads,
            threads,
            &tid,
            ra.is_none(),
            |k, r, kc| {
                let grow = k.op("add", &row0, r);
                if let Some(m) = ra {
                    a_kc.borrow_mut().push(kc.to_string());
                    let skc = k.op("mul", kc, sa);
                    return (k.radix_at(m, &grow, "0"), k.op("add", &skc, r));
                }
                let gr = k.op("mul", &grow, s.a.ci);
                let gk = k.op("mul", kc, s.a.ck);
                let g0 = k.op("add", &gr, &gk);
                let gsrc = k.op("add", &g0, s.a.base);
                let gsrc = k.at_z(gsrc, kext * s.a.ck);
                if kmaj {
                    let sr = k.op("mul", r, sa);
                    return (gsrc, k.op("add", &sr, kc));
                }
                let skc = k.op("mul", kc, sa);
                (gsrc, k.op("add", &skc, r))
            },
            |dr, dk| match kmaj {
                true => (dr * s.a.ci + dk * s.a.ck, dr * sa + dk),
                false => (dr * s.a.ci + dk * s.a.ck, dk * sa + dr),
            },
        );
        let b_src = self.copies(
            b_map,
            b_loads,
            threads,
            &tid,
            rb.is_none(),
            |k, r, c| {
                let (kr, nc) = if kvb { (c, r) } else { (r, c) };
                if let Some(m) = rb {
                    b_kc.borrow_mut().push(kr.to_string());
                    let gc = k.op("add", &col0, nc);
                    let sk = k.op("mul", kr, bn);
                    let sn = k.op("add", &sk, nc);
                    // `head-group`: `b`'s row digits are the block's batch.
                    return (k.radix_at(m, &row0, &gc), k.op("add", &sn, bs_off));
                }
                let gk = k.op("mul", kr, s.b.ck);
                let gc = k.op("add", &col0, nc);
                let gl = k.op("mul", &gc, s.b.clane);
                let g0 = k.op("add", &gk, &gl);
                let gsrc = k.op("add", &g0, s.b.base);
                let gsrc = k.at_z(gsrc, kext * s.b.ck);
                let sk = k.op("mul", kr, bn);
                let sn = k.op("add", &sk, nc);
                (gsrc, k.op("add", &sn, bs_off))
            },
            |d0, d1| {
                let (dk, dn) = if kvb { (d1, d0) } else { (d0, d1) };
                (dk * s.b.ck + dn * s.b.clane, dk * bn + dn)
            },
        );
        let Frag {
            aw,
            a_step,
            a_chunk,
            b_step,
            b_chunk,
            narrow,
        } = frag_layout(s, &g, kmaj && rungs.bank_plan, rungs.interleave);
        let a_frag0 = self.op("mul", &trow, a_step);
        let b_frag0 = self.op("mul", &tcol, b_step);

        // --- panel staging: global → (registers) → shared memory
        let a_ty = if a_vec { vt.clone() } else { t.clone() };
        let b_ty = if b_vec || kvb { vt.clone() } else { t.clone() };
        let a_al = if a_vec { GPU_VEC_BYTES } else { e };
        let b_al = if b_vec || kvb { GPU_VEC_BYTES } else { e };
        let (a_kc, b_kc) = (a_kc.into_inner(), b_kc.into_inner());
        // Copy `p`'s global index for panel `k0` (`kd` = `k0·ck` on a plain operand).
        let gidx = |k: &mut Self,
                    m: Option<&mapal_ir::RadixAddr>,
                    kcs: &[String],
                    p: usize,
                    gsrc: &str,
                    goff: u64,
                    k0: &str,
                    kd: &str| {
            if let Some(m) = m {
                let kk = k.op("add", k0, &kcs[p]);
                // A split partition's `k` starts `z·kext` along the fold.
                let kk = k.at_z(kk, kext);
                let kx = k.digits(&m.k, &kk);
                return k.op("add", gsrc, &kx);
            }
            let gi = k.op("add", gsrc, kd);
            k.plus(gi, goff)
        };
        // ADR-0043: a narrow operand's copy loads its format (the same
        // element count: a vector is `lanes` narrow elements) and widens it
        // in registers — the panel in shared memory is `elem`, so every
        // fragment read and step after it is the f32 tile's.
        let load_op = |k: &mut Self,
                       arr: &str,
                       gi: &str,
                       vec: bool,
                       ty: &str,
                       al: u64,
                       r: &mapal_ir::TileRead,
                       stream: bool| {
            let Some(n) = r.narrow else {
                let gp = k.gep(1, arr, gi);
                let bytes = if vec { lanes * e } else { e };
                return k.gload(ty, bytes, &gp, al, stream);
            };
            let nt = crate::narrow::llvm_ty(n);
            let (nty, nal) = match vec {
                true => (format!("<{lanes} x {nt}>"), lanes * n.bytes()),
                false => (nt.to_string(), n.bytes()),
            };
            let gp = k.gep_ty(1, nt, arr, gi);
            let w = k.gload(&nty, nal, &gp, nal, stream);
            let v = k.tmp();
            let tt = if n.bytes() == 1 {
                k.tmp()
            } else {
                String::new()
            };
            for l in crate::narrow::widen_lines(n, &nty, &w, ty, &v, &tt) {
                k.line(l);
            }
            v
        };
        // gapfix `stream-ld`: `b` re-read by every row block, `a` by every
        // column block.
        // Register staging only (a `cp.async` panel's copies take no policy
        // here), whole 4 / 8 / 16-byte loads only (`gload`).
        let whole = |r: &mapal_ir::TileRead, vec: bool| {
            let n = r.narrow.map_or(e, |n| n.bytes());
            !streamed && matches!(if vec { lanes * n } else { n }, 4 | 8 | 16)
        };
        let a_cs = whole(&s.a, a_vec) && self.streams(false, s.c.div_ceil(g.tx * g.tn));
        let b_cs = whole(&s.b, b_vec || kvb) && self.streams(true, s.rows.div_ceil(g.ty * g.tm));
        self.streamed = [a_cs, b_cs];
        let load_panels = |k: &mut Self, k0: &str| -> Vec<String> {
            let ka = k.op("mul", k0, s.a.ck);
            let kb = k.op("mul", k0, s.b.ck);
            let mut vals = Vec::new();
            for (p, (gsrc, goff, _, _)) in a_src.iter().enumerate() {
                let gi = gidx(k, ra, &a_kc, p, gsrc, *goff, k0, &ka);
                vals.push(load_op(k, "%a", &gi, a_vec, &a_ty, a_al, &s.a, a_cs));
            }
            for (p, (gsrc, goff, _, _)) in b_src.iter().enumerate() {
                let gi = gidx(k, rb, &b_kc, p, gsrc, *goff, k0, &kb);
                vals.push(load_op(k, "%b", &gi, b_vec || kvb, &b_ty, b_al, &s.b, b_cs));
            }
            vals
        };
        let store_panels = |k: &mut Self, vals: &[String]| {
            for ((_, _, sdst, soff), v) in a_src.iter().zip(vals) {
                if !a_vec {
                    let si = k.plus(sdst.clone(), *soff);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!("store {t} {v}, ptr addrspace(3) {sp}, align {e}"));
                    continue;
                }
                if kmaj {
                    let si = k.plus(sdst.clone(), *soff);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!(
                        "store {vt} {v}, ptr addrspace(3) {sp}, align {GPU_VEC_BYTES}"
                    ));
                    continue;
                }
                // `a` is staged transposed: one vector along k scatters to
                // `lanes` panel rows.
                for j in 0..lanes {
                    let x = k.tmp();
                    k.line(format!("{x} = extractelement {vt} {v}, i64 {j}"));
                    let si = k.op("add", sdst, soff + j * sa);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!("store {t} {x}, ptr addrspace(3) {sp}, align {e}"));
                }
            }
            for ((_, _, sdst, soff), v) in b_src.iter().zip(&vals[a_src.len()..]) {
                if kvb {
                    // `kmajor-b`: one vector along `k` scatters to `lanes`
                    // panel rows (`kvec-swizzle`: its key row's word group
                    // XORed by its key chunk's bank group).
                    let base = match swz {
                        Some((gs, st)) => {
                            let l = k.op("add", sdst, *soff);
                            let l = k.op("sub", &l, bs_off);
                            let q = k.op("udiv", &l, bn * lanes);
                            let q = k.op("urem", &q, gs);
                            let sw = k.op("mul", &q, st);
                            let l = k.op("xor", &l, &sw);
                            (k.op("add", &l, bs_off), 0)
                        }
                        None => (sdst.clone(), *soff),
                    };
                    for j in 0..lanes {
                        let x = k.tmp();
                        k.line(format!("{x} = extractelement {vt} {v}, i64 {j}"));
                        let si = k.op("add", &base.0, base.1 + j * bn);
                        let sp = k.gep(3, smem, &si);
                        k.line(format!("store {t} {x}, ptr addrspace(3) {sp}, align {e}"));
                    }
                    continue;
                }
                let si = k.plus(sdst.clone(), *soff);
                let sp = k.gep(3, smem, &si);
                k.line(format!(
                    "store {b_ty} {v}, ptr addrspace(3) {sp}, align {b_al}"
                ));
            }
        };
        let tys: Vec<&String> = a_src
            .iter()
            .map(|_| &a_ty)
            .chain(b_src.iter().map(|_| &b_ty))
            .collect();

        // `cbase`: the stage the compute reads (async), else the one buffer.
        let head = self.head;
        let (cbase, pre) = if streamed {
            debug_assert!(seg.is_none(), "stream-K runs the register-staged tile");
            // Issue panel `tile` into stage `stage`: one `cp.async` per unit.
            // `so`: the stage's first element (index type).
            let issue = |k: &mut Self, tile: &str, so: &str| {
                let k0 = k.op("mul", tile, bk);
                let ka = k.op("mul", &k0, s.a.ck);
                let kb = k.op("mul", &k0, s.b.ck);
                let so = so.to_string();
                for (p, (gsrc, goff, sdst, soff)) in a_src.iter().enumerate() {
                    let gi = gidx(k, ra, &a_kc, p, gsrc, *goff, &k0, &ka);
                    let gp = k.gep(1, "%a", &gi);
                    let si = k.op("add", sdst, &so);
                    let si = k.plus(si, *soff);
                    let sp = k.gep(3, smem, &si);
                    let (kind, n) = if kmaj {
                        ("cg", GPU_VEC_BYTES)
                    } else {
                        ("ca", e)
                    };
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.{kind}.shared.global.{n}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
                for (p, (gsrc, goff, sdst, soff)) in b_src.iter().enumerate() {
                    let gi = gidx(k, rb, &b_kc, p, gsrc, *goff, &k0, &kb);
                    // `narrow-async`: the `b` region keeps its place in the
                    // stage, its elements narrow: element `si` of the f32
                    // layout is narrow element `si + (4 / e − 1)·(so + bs_off)`
                    // (the region packed from its first byte; `e` 2 keeps the
                    // 16-bit formats' text).
                    if let Some(n) = nb {
                        let nt = crate::narrow::llvm_ty(n);
                        let gp = k.gep_ty(1, nt, "%b", &gi);
                        let si = k.op("add", sdst, &so);
                        let si = k.plus(si, *soff);
                        let (ni, f) = match n.bytes() {
                            2 => (k.op("add", &si, &so), 1),
                            e => {
                                let f = 4 / e - 1;
                                let c = k.op("mul", &so, f);
                                (k.op("add", &si, &c), f)
                            }
                        };
                        let ni = k.plus(ni, f * bs_off);
                        let sp = k.gep_ty(3, nt, smem, &ni);
                        k.line(format!(
                            "call void @llvm.nvvm.cp.async.ca.shared.global.{}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})",
                            lanes * n.bytes()
                        ));
                        continue;
                    }
                    let gp = k.gep(1, "%b", &gi);
                    let si = k.op("add", sdst, &so);
                    let si = k.plus(si, *soff);
                    let sp = k.gep(3, smem, &si);
                    let (kind, n) = if b_vec {
                        ("cg", GPU_VEC_BYTES)
                    } else {
                        ("ca", e)
                    };
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.{kind}.shared.global.{n}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
            };
            let nk = kext / bk;
            for st in 0..(g.stages - 1).min(nk) {
                issue(self, &st.to_string(), &(st * stage_elems).to_string());
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            for _ in nk..g.stages - 1 {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (it, itn) = (self.tmp(), self.tmp());
            self.line(format!("{it} = phi {ix} [0, %entry], [{itn}, %ktl]"));
            // tile-efficiency T3: the stage the compute reads, a rotating i32
            // counter (no `urem`, no 64-bit stage arithmetic): `it mod stages`.
            let (cs, csn) = (self.tmp(), self.tmp());
            self.line(format!("{cs} = phi i32 [0, %entry], [{csn}, %ktl]"));
            self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                g.stages - 2
            ));
            // One barrier per panel: panel `it` is visible to every thread, and
            // every thread is done with the stage the next issue overwrites.
            self.line("call void @llvm.nvvm.barrier0()");
            let nxt = self.op("add", &it, g.stages - 1);
            let more = self.tmp();
            self.line(format!("{more} = icmp ult {ix} {nxt}, {nk}"));
            self.line(format!("br i1 {more}, label %issue, label %issued"));
            // A stage's first element in the index type: `stage · stage_elems`
            // in i32 (every stage offset fits the 1 MiB-bounded shared memory).
            let stage_off = |k: &mut Self, st32: &str| {
                let x = k.tmp();
                k.line(format!("{x} = mul nuw i32 {st32}, {stage_elems}"));
                if ix == "i32" {
                    return x;
                }
                let y = k.tmp();
                k.line(format!("{y} = zext i32 {x} to {ix}"));
                y
            };
            self.out.push_str("issue:\n");
            // panel `it + stages − 1` lands in stage `(cs + stages − 1) mod stages`.
            let first = self.tmp();
            self.line(format!("{first} = icmp eq i32 {cs}, 0"));
            let back = self.tmp();
            self.line(format!("{back} = add i32 {cs}, -1"));
            let nst = self.tmp();
            self.line(format!(
                "{nst} = select i1 {first}, i32 {}, i32 {back}",
                g.stages - 1
            ));
            let nso = stage_off(self, &nst);
            issue(self, &nxt, &nso);
            self.line("br label %issued");
            self.out.push_str("issued:\n");
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cbase = stage_off(self, &cs);
            let up = self.tmp();
            self.line(format!("{up} = add nuw i32 {cs}, 1"));
            let wrap = self.tmp();
            self.line(format!("{wrap} = icmp eq i32 {up}, {}", g.stages));
            self.line(format!("{csn} = select i1 {wrap}, i32 0, i32 {up}"));
            self.line("br label %kk");
            self.async_tail = Some((it, itn, nk));
            (cbase, "issued")
        } else {
            // Prefetch rung: panel 0 is loaded before the loop; inside it, the
            // registers holding panel `k0` are stored, and panel `k0 + bk` is
            // requested before the compute starts, so its latency hides behind
            // `bk` steps of arithmetic. The last trip re-reads its own panel
            // (in bounds, never stored) rather than branching.
            let first = if self.rungs.prefetch {
                load_panels(self, &kbeg)
            } else {
                Vec::new()
            };
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (k0, k0n) = (self.tmp(), self.tmp());
            self.line(format!("{k0} = phi {ix} [{kbeg}, %{head}], [{k0n}, %ktl]"));
            if self.rungs.prefetch {
                let held: Vec<String> = first.iter().map(|_| self.tmp()).collect();
                for (i, (h, f)) in held.iter().zip(&first).enumerate() {
                    self.line(format!(
                        "{h} = phi {} [{f}, %{head}], [@NEXT{i}@, %ktl]",
                        tys[i]
                    ));
                }
                store_panels(self, &held);
                self.line("call void @llvm.nvvm.barrier0()");
                let nk = self.op("add", &k0, bk);
                let more = self.tmp();
                self.line(format!("{more} = icmp ult {ix} {nk}, {kend}"));
                let kl = self.tmp();
                self.line(format!("{kl} = select i1 {more}, {ix} {nk}, {ix} {k0}"));
                let next = load_panels(self, &kl);
                for (i, n) in next.iter().enumerate() {
                    self.out = self.out.replace(&format!("@NEXT{i}@"), n);
                }
            } else {
                let vals = load_panels(self, &k0);
                store_panels(self, &vals);
                self.line("call void @llvm.nvvm.barrier0()");
            }
            self.line("br label %kk");
            self.sync_tail = Some((k0, k0n));
            ("0".to_string(), "kt")
        };

        // --- bk register steps over the staged panels
        self.out.push_str("kk:\n");
        let (kk, kkn) = (self.tmp(), self.tmp());
        self.line(format!("{kk} = phi {ix} [0, %{pre}], [{kkn}, %kk]"));
        let frag = |k: &mut Self,
                    ki: &str,
                    row: u64,
                    first: &str,
                    n: u64,
                    off: u64,
                    chunk: u64,
                    narrow: Option<mapal_ir::NarrowFloat>,
                    swz: Option<(u64, u64)>| {
            // `kvec-swizzle`: the row's word groups as the copies stored them.
            let sw = swz.map(|(gs, st)| {
                let q = k.op("udiv", ki, lanes);
                let q = k.op("urem", &q, gs);
                k.op("mul", &q, st)
            });
            let r = k.op("mul", ki, row);
            let r = match &sw {
                Some(_) => r,
                None => k.op("add", &r, first),
            };
            let r = k.op("add", &r, off);
            let r = k.op("add", &r, &cbase);
            // `narrow-async`: the stage's narrow `b` region (see `issue`).
            // The narrow region packs from the f32 region's first byte: f32
            // element `r` is narrow element `r + (4 / e − 1)·(stage + bs_off)`
            // (`e` 2: the 16-bit formats' text as it was; 1: fp8).
            let r = match narrow {
                Some(n) if n.bytes() == 2 => {
                    let x = k.op("add", &r, &cbase);
                    k.plus(x, bs_off)
                }
                Some(n) => {
                    let f = 4 / n.bytes() - 1;
                    let c = k.op("mul", &cbase, f);
                    let x = k.op("add", &r, &c);
                    k.plus(x, f * bs_off)
                }
                None => r,
            };
            // `w` elements per load: a vector, or (`fill-rows`) the whole
            // fragment when it is narrower than one — `w = 1` a scalar.
            let w = n.min(lanes);
            let wt = if w == 1 {
                t.clone()
            } else {
                format!("<{w} x {t}>")
            };
            let mut vals = Vec::new();
            for v in 0..n / w {
                let si = match &sw {
                    Some(sw) => {
                        let c = k.op("add", first, v * chunk);
                        let c = k.op("xor", &c, sw);
                        k.op("add", &r, &c)
                    }
                    None => k.op("add", &r, v * chunk),
                };
                let x = if let Some(nf) = narrow {
                    let nt = crate::narrow::llvm_ty(nf);
                    let nwt = if w == 1 {
                        nt.to_string()
                    } else {
                        format!("<{w} x {nt}>")
                    };
                    let sp = k.gep_ty(3, nt, smem, &si);
                    let (y, x) = (k.tmp(), k.tmp());
                    k.line(format!(
                        "{y} = load {nwt}, ptr addrspace(3) {sp}, align {}",
                        w * nf.bytes()
                    ));
                    let tt = if nf.bytes() == 1 {
                        k.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(nf, &nwt, &y, &wt, &x, &tt) {
                        k.line(l);
                    }
                    x
                } else {
                    let sp = k.gep(3, smem, &si);
                    let x = k.tmp();
                    k.line(format!(
                        "{x} = load {wt}, ptr addrspace(3) {sp}, align {}",
                        w * e
                    ));
                    x
                };
                if w == 1 {
                    vals.push(x);
                    continue;
                }
                for j in 0..w {
                    let y = k.tmp();
                    k.line(format!("{y} = extractelement {wt} {x}, i64 {j}"));
                    vals.push(y);
                }
            }
            vals
        };
        let fma = |k: &mut Self, av: &[String], bv: &[String]| {
            for i in 0..tm as usize {
                for j in 0..tn as usize {
                    let p = acc_slot(k, i * tn as usize + j);
                    let cur = k.tmp();
                    k.line(format!("{cur} = load {t}, ptr {p}, align {e}"));
                    let r = k.step(&cur, &av[i], &bv[j]);
                    k.line(format!("store {t} {r}, ptr {p}, align {e}"));
                }
            }
        };
        // tile-efficiency T2: a k-major trip takes `lanes` `k` steps — one
        // 128-bit `a` load along `k` per owned row, then per step its `b`
        // fragment and the `tm × tn` FMAs, `k` ascending as before.
        let steps = if kmaj { bk / lanes } else { bk };
        if kmaj {
            let k0 = self.op("mul", &kk, lanes);
            let k0 = self.op("add", &k0, &cbase);
            let mut rows = Vec::new();
            for i in 0..tm {
                let r = self.op("add", &a_frag0, (i / aw) * a_chunk + i % aw);
                let r = self.op("mul", &r, sa);
                let si = self.op("add", &r, &k0);
                let sp = self.gep(3, smem, &si);
                let x = self.tmp();
                self.line(format!(
                    "{x} = load {vt}, ptr addrspace(3) {sp}, align {GPU_VEC_BYTES}"
                ));
                let row: Vec<String> = (0..lanes)
                    .map(|j| {
                        let y = self.tmp();
                        self.line(format!("{y} = extractelement {vt} {x}, i64 {j}"));
                        y
                    })
                    .collect();
                rows.push(row);
            }
            for j in 0..lanes as usize {
                let kj = self.op("mul", &kk, lanes);
                let kj = self.op("add", &kj, j as u64);
                let bv = frag(self, &kj, bn, &b_frag0, tn, bs_off, b_chunk, nb, swz);
                let av: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                fma(self, &av, &bv);
            }
        } else {
            let av = frag(self, &kk, sa, &a_frag0, tm, 0, a_chunk, None, None);
            let bv = frag(self, &kk, bn, &b_frag0, tn, bs_off, b_chunk, nb, swz);
            fma(self, &av, &bv);
        }
        self.line(format!("{kkn} = add {ix} {kk}, 1"));
        let kkd = self.tmp();
        self.line(format!("{kkd} = icmp eq {ix} {kkn}, {steps}"));
        // Fully unrolled: every shared-memory offset becomes an immediate.
        let unroll = if self.rungs.unroll {
            ", !llvm.loop !@LOOP@"
        } else {
            ""
        };
        self.line(format!("br i1 {kkd}, label %ktl, label %kk{unroll}"));

        self.out.push_str("ktl:\n");
        let ktd = self.tmp();
        if let Some((it, itn, nk)) = self.async_tail.take() {
            // No barrier here: the next panel's barrier orders the overwrite.
            self.line(format!("{itn} = add {ix} {it}, 1"));
            self.line(format!("{ktd} = icmp eq {ix} {itn}, {nk}"));
        } else {
            let (k0, k0n) = self.sync_tail.take().expect("sync panel loop");
            self.line("call void @llvm.nvvm.barrier0()");
            self.line(format!("{k0n} = add {ix} {k0}, {bk}"));
            self.line(format!("{ktd} = icmp eq {ix} {k0n}, {kend}"));
        }
        self.line(format!("br i1 {ktd}, label %done, label %kt"));

        // --- one vector store per `lanes` columns of each owned row
        self.out.push_str("done:\n");
        if self.col && !narrow {
            // Column-major (`rowfold-layout`): one vector store per `lanes`
            // rows of each owned column — a thread's rows come in runs of
            // `lanes` consecutive ones (`col_store`).
            for ic in 0..tm / lanes {
                let r = self.op("add", &row0, &a_frag0);
                let r = self.op("add", &r, ic * a_chunk);
                for v in 0..tn / lanes {
                    for j in 0..lanes {
                        let cc = self.op("add", &col0, &b_frag0);
                        let cc = self.op("add", &cc, v * b_chunk + j);
                        let cr = self.op("mul", &cc, s.rows);
                        let oi = self.op("add", &cr, &r);
                        let mut vec = "poison".to_string();
                        for il in 0..lanes {
                            let i = ic * lanes + il;
                            let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                            let nv = self.tmp();
                            self.line(format!(
                                "{nv} = insertelement {vt} {vec}, {t} {x}, i64 {il}"
                            ));
                            vec = nv;
                        }
                        let op = self.gep(1, "%o", &oi);
                        self.line(format!(
                            "store {vt} {vec}, ptr addrspace(1) {op}, align {GPU_VEC_BYTES}"
                        ));
                    }
                }
            }
            self.out.push_str("  ret void\n}\n");
            return;
        }
        // sweep-runtime-2 `streamk-epi`: a piece that is its tile's whole
        // fold stores through the group's members as the ordered kernel does
        // (`st_whole`); a piece of a shared tile adds its partials into the
        // tile's workspace slot and takes the tile's ticket (`st_part`).
        let split = match (&seg, self.ticket) {
            (Some(sg), true) => Some(self.ticket_split(sg, kext / bk, (bm, bn))),
            _ => None,
        };
        let passes: &[bool] = if split.is_some() {
            &[false, true]
        } else {
            &[false]
        };
        for &part in passes {
            if split.is_some() {
                self.out
                    .push_str(if part { "st_part:\n" } else { "st_whole:\n" });
            }
            self.part = part;
            // The workspace cell of tile-local `(lr, lc)` in the piece's slot.
            let ws_at = |k: &mut Self, r: &str, lc: &str| -> String {
                let sp = split.as_ref().expect("a shared tile's slot");
                let lr = k.tmp();
                k.line(format!("{lr} = sub {} {r}, {row0}", k.idx));
                let lrb = k.op("mul", &lr, bn);
                let l = k.op("add", &lrb, lc);
                k.op("add", &sp.base, &l)
            };
            if narrow {
                // One scalar store per owned cell (`cell-fill`), row- or
                // column-major; the epilogue's store takes one cell at a time.
                for i in 0..tm {
                    for j in 0..tn {
                        let p = acc_slot(self, (i * tn + j) as usize);
                        let x = self.tmp();
                        self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                        let r = self.op("add", &row0, &a_frag0);
                        let r = self.plus(r, (i / aw) * a_chunk + i % aw);
                        if part {
                            let lc = self.plus(b_frag0.clone(), j);
                            let wi = ws_at(self, &r, &lc);
                            self.store_o(&[x], &wi);
                            continue;
                        }
                        let cc = self.op("add", &col0, &b_frag0);
                        let cc = self.plus(cc, j);
                        let oi = if let Some(o) = ro {
                            self.radix_at(o, &r, &cc)
                        } else if self.col {
                            let cr = self.op("mul", &cc, s.rows);
                            self.op("add", &cr, &r)
                        } else {
                            let rr = self.op("mul", &r, s.c);
                            self.op("add", &rr, &cc)
                        };
                        // A split kernel stores its partials into workspace slab
                        // `z` (S62 integration: `cell-fill` × `splitk`), or adds
                        // them into the output (S63).
                        let oi = self.slab(oi);
                        if self.epi.is_some() && !self.part {
                            self.store_cells(&[x], &oi);
                            continue;
                        }
                        self.store_o(&[x], &oi);
                    }
                }
            } else {
                for i in 0..tm {
                    let r = self.op("add", &row0, &a_frag0);
                    let r = self.op("add", &r, (i / aw) * a_chunk + i % aw);
                    let rr = self.op("mul", &r, s.c);
                    let c0 = self.op("add", &rr, &col0);
                    let c0 = self.op("add", &c0, &b_frag0);
                    let rrow = ro.map(|_| r.clone());
                    // A split kernel stores its partials into workspace slab `z`,
                    // or adds them into the output (S63).
                    let c0 = self.slab(c0);
                    for v in 0..tn / lanes {
                        let mut xs = Vec::new();
                        for j in 0..lanes {
                            let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                            let x = self.tmp();
                            self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                            xs.push(x);
                        }
                        if part {
                            let lc = self.plus(b_frag0.clone(), v * b_chunk);
                            let wi = ws_at(self, &r, &lc);
                            self.store_o(&xs, &wi);
                            continue;
                        }
                        // A digit-affine output (`tile-post` on a radix site): its
                        // cell index is its address, `lanes` consecutive cells
                        // (in slab `z` of a split's workspace, as `c0`).
                        let oi = match (ro, &rrow) {
                            (Some(o), Some(r)) => {
                                let cc = self.op("add", &col0, &b_frag0);
                                let cc = self.op("add", &cc, v * b_chunk);
                                let a = self.radix_at(o, r, &cc);
                                self.slab(a)
                            }
                            _ => self.op("add", &c0, v * b_chunk),
                        };
                        if self.epi.is_some() && !self.part {
                            self.store_cells(&xs, &oi);
                            continue;
                        }
                        self.store_o(&xs, &oi);
                    }
                }
            }
            if let Some(sp) = split.as_ref() {
                if part {
                    self.ticket(sp, (bm, bn, threads), (&row0, &col0, &tid));
                }
                self.line("br label %st_join");
            }
        }
        self.part = false;
        if split.is_some() {
            self.out.push_str("st_join:\n");
        }
        self.finish(seg.as_ref());
    }
}
