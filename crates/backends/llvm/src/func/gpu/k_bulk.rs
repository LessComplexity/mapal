//! `Kernel` bulk-copy, barrier and `wgmma` helpers and the Hopper `wgmma` kernel.

use super::*;

impl Kernel<'_> {
    /// The 32-bit shared address `off` bytes past the barriers at `bars` (a
    /// shared pointer): each address its own cast of a pointer (the
    /// emulator maps it back, as `ldmatrix`'s).
    pub(super) fn bar_at(&mut self, bars: &str, off: &str) -> String {
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds i8, ptr addrspace(3) {bars}, i64 {off}"
        ));
        let a = self.tmp();
        self.line(format!("{a} = ptrtoint ptr addrspace(3) {p} to i32"));
        a
    }

    /// An inline-asm statement with operands (`args`: typed values).
    pub(super) fn asm_args(&mut self, text: &str, cons: &str, args: &[String]) {
        self.line(format!(
            "call void asm sideeffect \"{text}\", \"{cons},~{{memory}}\"({})",
            args.join(", ")
        ));
    }

    /// **TMA `bulk-stage`: panel `tile` into stage `stage`** (i64 values) of
    /// the TF32 `mma` tile by the bulk-copy engine — every row of the `a`
    /// panel (`bk` elements) and the `b` panel (`bn`) one `cp.async.bulk`
    /// into its shared row (`lda` / `ldb` words apart), warp 0's lane `l`
    /// copying rows `l, l + 32, …` (`a`'s `bm` rows, then `b`'s `bk`); lane 0
    /// first adds the stage's bytes to its barrier (`bars + 8·stage`,
    /// `mbarrier.arrive.expect_tx`). The stage was last read before the
    /// barrier every thread passed: `fence.proxy.async` orders those reads
    /// before the engine's writes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn issue_bulk(
        &mut self,
        g: &GpuMmaTile,
        bars: &str,
        tid: &str,
        row0: &str,
        col0: &str,
        tile: &str,
        stage: &str,
    ) {
        let s = self.site;
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let k0 = self.op("mul", tile, bk);
        let so = self.op("mul", stage, g.stage_words());
        let boff = self.op("mul", stage, BULK_BAR_BYTES);
        let bar = self.bar_at(bars, &boff);
        let bytes = (bm * bk + bk * bn) * 4;
        let t32 = self.tmp();
        self.line(format!("{t32} = trunc i64 {tid} to i32"));
        self.asm_args(
            "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
            "r,r,r",
            &[format!("i32 {bar}"), format!("i32 {bytes}"), format!("i32 {t32}")],
        );
        self.asm("fence.proxy.async.shared::cta;");
        let rows = bm + bk;
        let warp = self.profile_gpu.warp;
        for j in 0..rows.div_ceil(warp) {
            let q = self.op("add", tid, j * warp);
            // Warp 0's lanes, rows below `bm + bk`.
            let (w0, inr) = (self.tmp(), self.tmp());
            self.line(format!("{w0} = icmp ult i64 {tid}, {warp}"));
            self.line(format!("{inr} = icmp ult i64 {q}, {rows}"));
            let on = self.tmp();
            self.line(format!("{on} = and i1 {w0}, {inr}"));
            let pred = self.tmp();
            self.line(format!("{pred} = zext i1 {on} to i32"));
            // An idle lane names row 0: every address it forms is in bounds.
            let q0 = self.tmp();
            self.line(format!("{q0} = select i1 {on}, i64 {q}, i64 0"));
            let q = q0;
            let is_a = self.tmp();
            self.line(format!("{is_a} = icmp ult i64 {q}, {bm}"));
            // `a` row `q`: (row0 + q)·ci + k0 + base, shared `q·lda`.
            let ar = self.op("add", row0, &q);
            let ag = self.op("mul", &ar, s.a.ci);
            let ag = self.op("add", &ag, &k0);
            let ag = self.plus(ag, s.a.base);
            let asw = self.op("mul", &q, lda);
            // `b` row `q − bm` (wrapping where `q < bm`: unused).
            let br = self.op("sub", &q, bm);
            let bkr = self.op("add", &k0, &br);
            let bg = self.op("mul", &bkr, s.b.ck);
            let bg = self.op("add", &bg, col0);
            let bg = self.plus(bg, s.b.base);
            let bsw = self.op("mul", &br, ldb);
            let bsw = self.plus(bsw, bm * lda);
            let ap = self.gep(1, "%a", &ag);
            let bp = self.gep(1, "%b", &bg);
            let gp = self.tmp();
            self.line(format!(
                "{gp} = select i1 {is_a}, ptr addrspace(1) {ap}, ptr addrspace(1) {bp}"
            ));
            let sw = self.tmp();
            self.line(format!("{sw} = select i1 {is_a}, i64 {asw}, i64 {bsw}"));
            let si = self.op("add", &sw, &so);
            let sp = self.gep(3, "@mapal_smem", &si);
            let s32 = self.tmp();
            self.line(format!("{s32} = ptrtoint ptr addrspace(3) {sp} to i32"));
            let n = self.tmp();
            self.line(format!(
                "{n} = select i1 {is_a}, i32 {}, i32 {}",
                bk * 4,
                bn * 4
            ));
            self.asm_args(
                "{ .reg .pred p; setp.ne.u32 p, $4, 0; @p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3]; }",
                "r,l,r,r,r",
                &[
                    format!("i32 {s32}"),
                    format!("ptr addrspace(1) {gp}"),
                    format!("i32 {n}"),
                    format!("i32 {bar}"),
                    format!("i32 {pred}"),
                ],
            );
        }
    }

    /// Issue panel `tile` (an i64 value) into stage `stage` (an i64 value):
    /// one 16-byte `cp.async` per `(is_a, global index at k0 = 0, shared
    /// word in stage 0)` copy.
    pub(super) fn issue16(
        &mut self,
        copies: &[(bool, String, String)],
        tile: &str,
        stage: &str,
        bk: u64,
        stage_words: u64,
    ) {
        let s = self.site;
        let k0 = self.op("mul", tile, bk);
        let ka = self.op("mul", &k0, s.a.ck);
        let kb = self.op("mul", &k0, s.b.ck);
        let so = self.op("mul", stage, stage_words);
        for (is_a, gsrc, sdst) in copies {
            let (arr, off) = if *is_a { ("%a", &ka) } else { ("%b", &kb) };
            let gi = self.op("add", gsrc, off);
            let gp = self.gep(1, arr, &gi);
            let si = self.op("add", sdst, &so);
            let sp = self.gep(3, "@mapal_smem", &si);
            self.line(format!(
                "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
            ));
        }
    }

    /// `cvt.rn.tf32.f32`: round to nearest, ties to even — ADR-0039's
    /// `to_tf32`. `wgmma` itself truncates the low 13 bits (PTX ISA), so an
    /// operand it reads must already be rounded. `sideeffect` pins the write
    /// before the `wgmma.fence` that orders it.
    pub(super) fn round_tf32(&mut self, x: &str, pinned: bool) -> String {
        let r = self.tmp();
        let se = if pinned { " sideeffect" } else { "" };
        self.line(format!(
            "{r} = call i32 asm{se} \"cvt.rn.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
        ));
        r
    }

    pub(super) fn asm(&mut self, text: &str) {
        self.line(format!(
            "call void asm sideeffect \"{text}\", \"~{{memory}}\"()"
        ));
    }

    /// **Hopper TF32 through `wgmma` (S53 `wgmma-hopper`).** The site's
    /// product is computed transposed, `cᵀ = bᵀ·aᵀ` ([`GpuWgmmaTile`]): warpgroup
    /// `wg` of block `(bx, by)` owns `c` columns `bx·bn + 64·wg ..+64` (the
    /// instruction's M) × rows `by·n ..+n` (its N).
    ///
    /// - `a` panels land by `cp.async`, K-major in 128-byte swizzle rows (the
    ///   descriptor's mode 1; SBO one 8-row atom). `wgmma` reads shared memory
    ///   as it is, truncating, so each thread rounds the chunks it copied once
    ///   (RNE) and fences them to the async proxy before the panel barrier.
    ///   That pass costs a third of the kernel at 4096 (S53 diag4: 0.89 ms,
    ///   0.60 without it), and two ways around it measured worse on H100
    ///   PCIe (nsys 4096, this kernel 0.80 ms): staging `a` through registers,
    ///   rounded on the way, 0.94 ms (latency of one panel in flight, not
    ///   two); two MMA groups in flight (`wait_group 1`, doubled fragments,
    ///   which moves the derivation to 128 × 128 at bk 64) 0.98 ms, and it
    ///   failed the TF32 value gate at 1024 with four stages.
    ///   `benches/results/gpu/kernels/optimization-ladder/wgmma-hopper-h100-pcie-run{2,3}-*.txt`.
    /// - `b` panels land row-major, rows padded; each thread loads its `bᵀ`
    ///   fragments (`a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4) of the
    ///   warp's 16 rows, PTX ISA m64nNk8) and rounds them in the load.
    /// - Pipeline: panel `it + stages − 2` is copied into the stage panel
    ///   `it − 2` used — the group issued at `it − 1` may still be reading
    ///   its own — and the wait for that group sits after the barrier, right
    ///   before its fragment registers are rewritten.
    ///
    /// Accumulator `d[4j + 2h + e]` holds `cᵀ(16w + g + 8h, 8j + 2t + e)`;
    /// each warp-wide store fills four whole 32-byte sectors of `c`.
    pub(super) fn wgmma(&mut self, name: &str, g: GpuWgmmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, ldb, n) = (g.bm(), g.bn(), g.bk, g.ldb(), g.n);
        let (stage_words, a_words) = (g.stage_words(), g.a_words());
        let nk = s.k / bk;
        let dist = g.stages - 2;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";

        let tid = self.sreg("tid.x");
        let (bx, by) = self.block_ids();
        let wg = self.op("udiv", &tid, WARPGROUP_WARPS * warp);
        let warp_id = self.op("udiv", &tid, warp);
        let w = self.op("urem", &warp_id, WARPGROUP_WARPS);
        let lane = self.op("urem", &tid, warp);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = (n / 2) as usize;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }

        // Copies, in words: `a` row m, 16-byte chunk kc — atom column
        // ka = kc / 8, chunk c = kc % 8 — lands at ka·bm·128 + m·128 +
        // (c ⊕ m%8)·16 bytes (`Swizzle<3,4,3>`): eight lanes read one whole
        // 128-byte line and write eight distinct bank groups. `b` as the mma
        // kernel stages it.
        let kcs = bk * 4 / GPU_VEC_BYTES;
        let row_chunks = WGMMA_SWIZZLE_BYTES / GPU_VEC_BYTES;
        let row_w = WGMMA_SWIZZLE_BYTES / 4;
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        for p in 0..bm * kcs / threads {
            let li = self.op("add", &tid, p * threads);
            let kc = self.op("urem", &li, kcs);
            let m = self.op("udiv", &li, kcs);
            let grow = self.op("add", &row0, &m);
            let gr = self.op("mul", &grow, s.a.ci);
            let gk = self.op("mul", &kc, 4);
            let g0 = self.op("add", &gr, &gk);
            let gsrc = self.op("add", &g0, s.a.base);
            let ka = self.op("udiv", &kc, row_chunks);
            let c = self.op("urem", &kc, row_chunks);
            let r8 = self.op("urem", &m, WGMMA_ATOM_ROWS);
            let sw = self.op("xor", &c, &r8);
            let sa = self.op("mul", &ka, bm * row_w);
            let sr = self.op("mul", &m, row_w);
            let sc = self.op("mul", &sw, 4);
            let sdst = self.op("add", &sa, &sr);
            let sdst = self.op("add", &sdst, &sc);
            copies.push((true, gsrc, sdst));
        }
        let bc = bn / 4;
        for p in 0..bk * bc / threads {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, a_words);
            copies.push((false, gsrc, sdst));
        }
        for st in 0..dist.min(nk) {
            self.issue16(&copies, &st.to_string(), &st.to_string(), bk, stage_words);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        for _ in nk..dist {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        // This thread's bᵀ fragment origin inside a stage: row k = t, column
        // n = 64·wg + 16·w + g.
        let fr = self.op("mul", &t4, ldb);
        let fc = self.op("mul", &wg, WGMMA_M);
        let fw = self.op("mul", &w, WGMMA_M / WARPGROUP_WARPS);
        let fo = self.op("add", &fr, &fc);
        let fo = self.op("add", &fo, &fw);
        let fo = self.op("add", &fo, &grp);
        let b_org = self.op("add", &fo, a_words);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %issued]"));
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            dist - 1
        ));
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        // Round this thread's own `a` chunks of panel `it`, once.
        for (_, _, sdst) in copies.iter().filter(|c| c.0) {
            let si = self.op("add", sdst, &cbase);
            let sp = self.gep(3, smem, &si);
            let v = self.tmp();
            self.line(format!(
                "{v} = load <4 x float>, ptr addrspace(3) {sp}, align 16"
            ));
            let mut r = "poison".to_string();
            for j in 0..4 {
                let x = self.tmp();
                self.line(format!("{x} = extractelement <4 x float> {v}, i64 {j}"));
                let y = self.round_tf32(&x, false);
                let nr = self.tmp();
                self.line(format!(
                    "{nr} = insertelement <4 x i32> {r}, i32 {y}, i64 {j}"
                ));
                r = nr;
            }
            self.line(format!(
                "store <4 x i32> {r}, ptr addrspace(3) {sp}, align 16"
            ));
        }
        self.asm("fence.proxy.async.shared::cta;");
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, dist);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        self.issue16(&copies, &nxt, &nst, bk, stage_words);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        // The previous group done: its fragment registers may be rewritten.
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let fb = self.op("add", &cbase, &b_org);
        let mut frags = Vec::new();
        for kk in 0..bk / WGMMA_K {
            let mut f = Vec::new();
            for off in [0, 8, 4 * ldb, 4 * ldb + 8] {
                let i = self.op("add", &fb, kk * WGMMA_K * ldb + off);
                let p = self.gep(3, smem, &i);
                let x = self.tmp();
                self.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
                f.push(self.round_tf32(&x, true));
            }
            frags.push(f);
        }
        self.asm("wgmma.fence.sync.aligned;");
        // Descriptor of this stage's `a` panel; k8 step `kk` starts in atom
        // column `kk·32 / 128`, `kk·32 % 128` bytes into its swizzled rows (the
        // hardware applies the XOR to the address it forms).
        let ap = self.gep(3, smem, &cbase);
        let a32 = self.tmp();
        self.line(format!("{a32} = ptrtoint ptr addrspace(3) {ap} to i32"));
        let a64 = self.tmp();
        self.line(format!("{a64} = zext i32 {a32} to i64"));
        let enc = self.op("lshr", &a64, 4);
        let enc = self.op("and", &enc, 0x3fff);
        let desc0 = self.op("or", &enc, g.desc_fields());
        let k_off = |kk: u64| {
            let b = kk * WGMMA_K * 4;
            (b / WGMMA_SWIZZLE_BYTES * bm * WGMMA_SWIZZLE_BYTES + b % WGMMA_SWIZZLE_BYTES) >> 4
        };
        let sty = format!("{{ {} }}", vec!["float"; accs].join(", "));
        let mut text = format!("wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32 {{");
        text.push_str(
            &(0..accs)
                .map(|i| format!("${i}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push_str(&format!(
            "}}, {{${},${},${},${}}}, ${}, 1, 1, 1;",
            accs,
            accs + 1,
            accs + 2,
            accs + 3,
            accs + 4
        ));
        let mut cons = vec!["=f".to_string(); accs];
        cons.extend(["r", "r", "r", "r", "l"].map(String::from));
        cons.extend((0..accs).map(|i| i.to_string()));
        let cons = cons.join(",");
        for (kk, f) in frags.iter().enumerate() {
            let desc = self.op("add", &desc0, k_off(kk as u64));
            let mut args = format!(
                "i32 {}, i32 {}, i32 {}, i32 {}, i64 {desc}",
                f[0], f[1], f[2], f[3]
            );
            let mut slots = Vec::new();
            for q in 0..accs {
                let p = acc_slot(self, q);
                let v = self.tmp();
                self.line(format!("{v} = load float, ptr {p}, align 4"));
                args.push_str(&format!(", float {v}"));
                slots.push(p);
            }
            let r = self.tmp();
            self.line(format!(
                "{r} = call {sty} asm sideeffect \"{text}\", \"{cons}\"({args})"
            ));
            for (q, p) in slots.iter().enumerate() {
                let d = self.tmp();
                self.line(format!("{d} = extractvalue {sty} {r}, {q}"));
                self.line(format!("store float {d}, ptr {p}, align 4"));
            }
        }
        self.asm("wgmma.commit_group.sync.aligned;");
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        self.out.push_str("done:\n");
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let er = self.op("mul", &t4, 2);
        let er = self.op("add", &er, &row0);
        let er = self.op("mul", &er, s.c);
        let ec = self.op("add", &col0, &fc);
        let ec = self.op("add", &ec, &fw);
        let ec = self.op("add", &ec, &grp);
        let ebase = self.op("add", &er, &ec);
        for j in 0..n / 8 {
            for h in 0..2u64 {
                for e in 0..2u64 {
                    let q = (4 * j + 2 * h + e) as usize;
                    let p = acc_slot(self, q);
                    let x = self.tmp();
                    self.line(format!("{x} = load float, ptr {p}, align 4"));
                    let oi = self.op("add", &ebase, (8 * j + e) * s.c + 8 * h);
                    if self.epi.is_some() {
                        self.store_cells(&[x], &oi);
                        continue;
                    }
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store float {x}, ptr addrspace(1) {op}, align 4"));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// An `mbarrier.try_wait.parity` spin on the barrier at the 32-bit shared
    /// address `bar` for the phase of parity `par` (an i64 value), in blocks
    /// `{label}w` (the spin) and `{label}d` (past it).
    pub(super) fn mbar_wait(&mut self, bar: &str, par: &str, label: &str) {
        let p32 = self.tmp();
        self.line(format!("{p32} = trunc i64 {par} to i32"));
        self.line(format!("br label %{label}w"));
        self.out.push_str(&format!("{label}w:\n"));
        let ok = self.tmp();
        self.line(format!(
            "{ok} = call i32 asm sideeffect \"{{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.u32 $0, 1, 0, p; }}\", \"=r,r,r,~{{memory}}\"(i32 {bar}, i32 {p32})"
        ));
        let done = self.tmp();
        self.line(format!("{done} = icmp ne i32 {ok}, 0"));
        self.line(format!("br i1 {done}, label %{label}d, label %{label}w"));
        self.out.push_str(&format!("{label}d:\n"));
    }

    /// The 32-bit shared address `off` (an i64 value or literal) bytes into
    /// the dynamic shared memory.
    pub(super) fn smem_at(&mut self, off: &str) -> String {
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {off}"
        ));
        let a = self.tmp();
        self.line(format!("{a} = ptrtoint ptr addrspace(3) {p} to i32"));
        a
    }

    /// `v` (an i64 value) as an i32.
    pub(super) fn i32_of(&mut self, v: &str) -> String {
        let t = self.tmp();
        self.line(format!("{t} = trunc i64 {v} to i32"));
        t
    }
}
