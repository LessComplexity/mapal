//! `Kernel` warp-specialized Hopper TF32 kernel.

use super::*;

impl Kernel<'_> {
    /// **hopper-wgmma-tma K2: the warp-specialized Hopper TF32 kernel**
    /// ([`GpuWgmmaTile::ws`]) — S53's transposed product `cᵀ = bᵀ·aᵀ`
    /// ([`Self::wgmma`]: consumer warpgroup `cw` owns `c` columns
    /// `bx·bn + 64·cw ..+64` × rows `by·n ..+n`) fed by tensor boxes.
    ///
    /// - **Producer** (warpgroup 0, `setmaxnreg.dec` under `ws-regs`): step `j`
    ///   first rounds tile `j − (stages − 1)` in place — every thread waits
    ///   its stage's full barrier, rounds its 16-byte chunks of the `aᵀ` box
    ///   (RNE, `cvt.rn.tf32`: `wgmma` truncates what it reads), fences them
    ///   to the async proxy and arrives on the stage's ready barrier — then
    ///   thread 0 waits tile `j`'s stage empty, counts its bytes
    ///   (`expect_tx`) and issues one `aᵀ` box (`{bk, n}` at `(k0, row0)`) and
    ///   `bn/32` `b` boxes (`{32, bk}` at `(col0 + 32q, k0)`), all 128-byte
    ///   swizzled, onto its full barrier. `stages − 1` tiles are in flight
    ///   while one is rounded.
    /// - **Consumers** (`setmaxnreg.inc`): tile `i` waits its stage full (the
    ///   `b` boxes the fragments load) and ready (the rounded `aᵀ`), loads the
    ///   `bᵀ` fragments through the box swizzle (rounded in the load), issues
    ///   `bk/8` `wgmma` on the stage's descriptor and commits. Before its
    ///   fragment loads it waits for tile `i − 1`'s group (an `a` fragment
    ///   register may not be rewritten before then — PTX ISA), whose stage
    ///   each warp's lane 0 then releases (empty barrier): a stage is never
    ///   released while a group may still read it; the group in flight
    ///   computes through this tile's barrier waits, and the other consumer
    ///   warpgroup's through its fragment loads.
    /// - **Barriers** (after the stages): `full[s]` (one arrival + the bytes),
    ///   `ready[s]` (the producer's 128 threads), `empty[s]` (the consumer
    ///   warps); waits by phase parity.
    pub(super) fn wgmma_ws(&mut self, name: &str, g: GpuWgmmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let wgt = WARPGROUP_WARPS * warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, n) = (g.bm(), g.bn(), g.bk, g.n);
        let (st, sb, ab) = (g.stages, g.ws_stage_bytes(), g.ws_a_bytes());
        let span = WGMMA_SWIZZLE_BYTES;
        let box_b = span * bk;
        let nk = s.k / bk;
        // wgmma-16bit: the 16-bit form (`direct`) reads what the boxes land —
        // no rounding pass, no ready barrier; the TF32 form's producer rounds
        // the `aᵀ` box `stages − 1` tiles behind its copies.
        let direct = g.direct;
        let rounds = !direct;
        let lead = if rounds { st - 1 } else { 0 };
        // Elements of a 128-byte swizzle row of each operand's box.
        let (ra, rb) = (span / g.ea, span / g.eb);
        let digits = self.rungs.tma_digits;
        let maps =
            ws_tensors(s, &g, &self.profile_gpu, digits).expect("the derivation checked the boxes");
        self.tmap = maps.to_vec();
        self.header(name, threads, GPU_VEC_BYTES, 1);

        let tid = self.sreg("tid.x");
        // hopper-launch `ws-persist` ([`GpuWgmmaTile::ctas`]): CTA `cta` walks
        // the tiles `cta, cta + ctas, …` below `tiles` — `n_t` of them, each
        // `nk` steps of the one stage ring (step `t·nk + kk`), its barrier
        // phases running on across tiles. Else the block's own tile.
        let persist = g.ctas > 0;
        let (tiles_x, tiles_y) = (s.c / bn, s.rows / bm);
        // The walk's CTA and its tile count, read where each warpgroup uses
        // them — after its `setmaxnreg`: a value live across the reallocation
        // is spilled (8 bytes at every walk when read once at entry).
        let walk = |k: &mut Self| -> (String, String) {
            // The linear CTA: the host may lay the grid out either way.
            let x = k.sreg("ctaid.x");
            let y = k.sreg("ctaid.y");
            let nx = k.sreg("nctaid.x");
            let yx = k.op("mul", &y, &nx);
            let cta = k.op("add", &x, &yx);
            let left = k.op("sub", &(tiles_x * tiles_y + g.ctas - 1).to_string(), &cta);
            let n_t = k.op32("udiv", &left, g.ctas);
            (cta, n_t)
        };
        let (row0, col0) = if persist {
            (String::new(), String::new())
        } else {
            let (bx, by) = self.block_ids();
            (by, bx)
        };
        let wg = self.op("udiv", &tid, wgt);
        let (row0, col0) = match persist {
            true => (row0, col0),
            false => (self.op("mul", &row0, bm), self.op("mul", &col0, bn)),
        };
        let h = self.raster.1;
        // Tile `t` of the walk: `(row0, col0)` of tile `cta + t·ctas` in the
        // raster order the launch derived.
        let tile_at = |k: &mut Self, cta: &str, t: &str| -> (String, String) {
            let off = k.op("mul", t, g.ctas);
            let id = k.op("add", cta, &off);
            // `raster_block`'s walk of `id` (`h` block-rows per column), its
            // divisions by constants in 32 bits: the producer warpgroup runs
            // this at its `setmaxnreg.dec` share (a 64-bit magic division
            // spilled 8 bytes there at `nk` 448).
            let (bx, by) = if h >= 2 {
                let group = k.op32("udiv", &id, h * tiles_x);
                let local = k.op32("urem", &id, h * tiles_x);
                let top = k.op("mul", &group, h);
                let dy = k.op32("urem", &local, h);
                let c = k.op32("udiv", &local, h);
                let r = k.op("add", &top, &dy);
                (k.opaque(&c), k.opaque(&r))
            } else {
                (k.op32("urem", &id, tiles_x), k.op32("udiv", &id, tiles_x))
            };
            (k.op("mul", &by, bm), k.op("mul", &bx, bn))
        };
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
        if !persist {
            for idx in 0..accs {
                let p = acc_slot(self, idx);
                self.line(format!("store float {seed}, ptr {p}, align 4"));
            }
        }
        // Barriers: full (kind 0), ready (1), empty (2), `stages` each.
        let bars = self.tmp();
        self.line(format!(
            "{bars} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {}",
            st * sb
        ));
        let t32 = self.i32_of(&tid);
        for kind in 0..3u64 {
            let count = [1, wgt, g.groups * WARPGROUP_WARPS][kind as usize];
            for x in 0..st {
                let a = self.bar_at(&bars, &((kind * st + x) * BULK_BAR_BYTES).to_string());
                self.asm_args(
                    "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.init.shared::cta.b64 [$0], $1; }",
                    "r,r,r",
                    &[format!("i32 {a}"), format!("i32 {count}"), format!("i32 {t32}")],
                );
            }
        }
        self.asm("fence.mbarrier_init.release.cluster;");
        self.line("call void @llvm.nvvm.barrier0()");
        let bar = |k: &mut Self, kind: u64, stage: &str| {
            let o = k.op("mul", stage, BULK_BAR_BYTES);
            let o = k.op("add", &o, kind * st * BULK_BAR_BYTES);
            k.bar_at(&bars, &o)
        };
        let prod = self.tmp();
        self.line(format!("{prod} = icmp eq i64 {wg}, 0"));
        self.line(format!("br i1 {prod}, label %prod, label %cons"));

        // ---- producer warpgroup ----
        self.out.push_str("prod:\n");
        if let Some((p, _)) = g.realloc {
            self.asm(&format!("setmaxnreg.dec.sync.aligned.u32 {p};"));
        }
        let tms: Vec<String> = (0..maps.len())
            .map(|m| {
                let t = self.tmp();
                self.line(format!("{t} = ptrtoint ptr %tm{m} to i64"));
                t
            })
            .collect();
        let r32 = if persist {
            String::new()
        } else {
            self.i32_of(&row0)
        };
        // The walk's steps: `n_t·nk` (one tile: `nk`).
        let (cta, n_t) = match persist {
            true => walk(self),
            false => (String::new(), String::new()),
        };
        let steps = if persist {
            self.op("mul", &n_t, nk)
        } else {
            nk.to_string()
        };
        self.line("br label %pl");
        self.out.push_str("pl:\n");
        let (j, jn) = (self.tmp(), self.tmp());
        self.line(format!("{j} = phi i64 [0, %prod], [{jn}, %pn]"));
        if rounds {
            let ge = self.tmp();
            self.line(format!("{ge} = icmp uge i64 {j}, {lead}"));
            self.line(format!("br i1 {ge}, label %pr, label %pi"));
            // Round tile `j − lead`'s `aᵀ` box in place.
            self.out.push_str("pr:\n");
            let i = self.op("sub", &j, lead);
            let x = self.op("urem", &i, st);
            let ph = self.op("udiv", &i, st);
            let ph = self.op("and", &ph, 1);
            let fb = bar(self, 0, &x);
            self.mbar_wait(&fb, &ph, "prf");
            let base = self.op("mul", &x, sb);
            let chunks = ab / GPU_VEC_BYTES;
            for p in 0..chunks.div_ceil(wgt) {
                let q = self.op("add", &tid, p * wgt);
                let tail = (p + 1) * wgt > chunks;
                if tail {
                    let inr = self.tmp();
                    self.line(format!("{inr} = icmp ult i64 {q}, {chunks}"));
                    self.line(format!("br i1 {inr}, label %prc{p}, label %prn{p}"));
                    self.out.push_str(&format!("prc{p}:\n"));
                }
                let o = self.op("mul", &q, GPU_VEC_BYTES);
                let o = self.op("add", &base, &o);
                let sp = self.tmp();
                self.line(format!(
                    "{sp} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {o}"
                ));
                let v = self.tmp();
                self.line(format!(
                    "{v} = load <4 x float>, ptr addrspace(3) {sp}, align 16"
                ));
                let mut r = "poison".to_string();
                for e in 0..4 {
                    let xv = self.tmp();
                    self.line(format!("{xv} = extractelement <4 x float> {v}, i64 {e}"));
                    let y = self.round_tf32(&xv, false);
                    let nr = self.tmp();
                    self.line(format!(
                        "{nr} = insertelement <4 x i32> {r}, i32 {y}, i64 {e}"
                    ));
                    r = nr;
                }
                self.line(format!(
                    "store <4 x i32> {r}, ptr addrspace(3) {sp}, align 16"
                ));
                if tail {
                    self.line(format!("br label %prn{p}"));
                    self.out.push_str(&format!("prn{p}:\n"));
                }
            }
            self.asm("fence.proxy.async.shared::cta;");
            let rdy = bar(self, 1, &x);
            self.asm_args(
                "mbarrier.arrive.shared::cta.b64 _, [$0];",
                "r",
                &[format!("i32 {rdy}")],
            );
        }
        self.line("br label %pi");
        // Issue tile `j` (thread 0).
        self.out.push_str("pi:\n");
        let lt = self.tmp();
        self.line(format!("{lt} = icmp ult i64 {j}, {steps}"));
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 {tid}, 0"));
        let go = self.tmp();
        self.line(format!("{go} = and i1 {lt}, {t0}"));
        self.line(format!("br i1 {go}, label %pt, label %pn"));
        self.out.push_str("pt:\n");
        // Step `j`'s tile and its `k` step (one tile: `j` itself).
        let (r32, col0, kj) = if persist {
            let t = self.op32("udiv", &j, nk);
            let kj = self.op32("urem", &j, nk);
            let (r0, c0) = tile_at(self, &cta, &t);
            (self.i32_of(&r0), c0, kj)
        } else {
            (r32.clone(), col0.clone(), j.clone())
        };
        let x = self.op("urem", &j, st);
        let ph = self.op("udiv", &j, st);
        let ph = self.op("and", &ph, 1);
        let ph = self.op("xor", &ph, 1);
        let eb = bar(self, 2, &x);
        self.mbar_wait(&eb, &ph, "pe");
        let fb = bar(self, 0, &x);
        let bytes = ab + g.ws_b_bytes();
        self.asm_args(
            "{ .reg .pred p; setp.eq.u32 p, $2, 0; @p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1; }",
            "r,r,r",
            &[format!("i32 {fb}"), format!("i32 {bytes}"), "i32 0".into()],
        );
        let k0 = self.op("mul", &kj, bk);
        let k32 = self.i32_of(&k0);
        let base = self.op("mul", &x, sb);
        let ad = self.smem_at(&base);
        let copy = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];";
        if g.ws_a_cols() > 1 {
            // The k digits: coordinates (0, row0, k0 / ra).
            let kq = self.op("udiv", &k0, ra);
            let kq = self.i32_of(&kq);
            self.asm_args(
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
                "r,l,r,r,r,r",
                &[
                    format!("i32 {ad}"),
                    format!("i64 {}", tms[0]),
                    "i32 0".into(),
                    format!("i32 {r32}"),
                    format!("i32 {kq}"),
                    format!("i32 {fb}"),
                ],
            );
        } else {
            self.asm_args(
                copy,
                "r,l,r,r,r",
                &[
                    format!("i32 {ad}"),
                    format!("i64 {}", tms[0]),
                    format!("i32 {k32}"),
                    format!("i32 {r32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        if g.b_kmajor {
            // wgmma-16bit: a k-major `b` — `bn` rows of its `k` digit
            // columns, one box: (k0, col0), or (0, col0, k0 / rb) past one
            // column.
            let o = self.op("add", &base, ab);
            let bd = self.smem_at(&o);
            let c32 = self.i32_of(&col0);
            if bk > rb {
                let kq = self.op("udiv", &k0, rb);
                let kq = self.i32_of(&kq);
                self.asm_args(
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
                    "r,l,r,r,r,r",
                    &[
                        format!("i32 {bd}"),
                        format!("i64 {}", tms[1]),
                        "i32 0".into(),
                        format!("i32 {c32}"),
                        format!("i32 {kq}"),
                        format!("i32 {fb}"),
                    ],
                );
            } else {
                self.asm_args(
                    copy,
                    "r,l,r,r,r",
                    &[
                        format!("i32 {bd}"),
                        format!("i64 {}", tms[1]),
                        format!("i32 {k32}"),
                        format!("i32 {c32}"),
                        format!("i32 {fb}"),
                    ],
                );
            }
        } else if digits {
            // One box of every slab: coordinates (0, k0, col0 / rb).
            let o = self.op("add", &base, ab);
            let bd = self.smem_at(&o);
            let c = self.op("udiv", &col0, rb);
            let c32 = self.i32_of(&c);
            self.asm_args(
                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
                "r,l,r,r,r,r",
                &[
                    format!("i32 {bd}"),
                    format!("i64 {}", tms[1]),
                    "i32 0".into(),
                    format!("i32 {k32}"),
                    format!("i32 {c32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        for q in 0..if digits || g.b_kmajor {
            0
        } else {
            g.ws_b_boxes()
        } {
            let o = self.op("add", &base, ab + q * box_b);
            let bd = self.smem_at(&o);
            let c = self.op("add", &col0, q * rb);
            let c32 = self.i32_of(&c);
            self.asm_args(
                copy,
                "r,l,r,r,r",
                &[
                    format!("i32 {bd}"),
                    format!("i64 {}", tms[1]),
                    format!("i32 {c32}"),
                    format!("i32 {k32}"),
                    format!("i32 {fb}"),
                ],
            );
        }
        self.line("br label %pn");
        self.out.push_str("pn:\n");
        self.line(format!("{jn} = add i64 {j}, 1"));
        let fin = self.tmp();
        let last = if persist {
            self.op("add", &steps, lead)
        } else {
            (nk + lead).to_string()
        };
        self.line(format!("{fin} = icmp eq i64 {jn}, {last}"));
        self.line(format!("br i1 {fin}, label %pend, label %pl"));
        self.out.push_str("pend:\n  ret void\n");

        // ---- consumer warpgroups ----
        self.out.push_str("cons:\n");
        if let Some((_, c)) = g.realloc {
            self.asm(&format!("setmaxnreg.inc.sync.aligned.u32 {c};"));
        }
        let (cta, n_t) = match persist {
            true => walk(self),
            false => (String::new(), String::new()),
        };
        let cw = self.op("sub", &wg, 1);
        let ct = self.op("sub", &tid, wgt);
        let warp_id = self.op("udiv", &ct, warp);
        let w = self.op("urem", &warp_id, WARPGROUP_WARPS);
        let lane = self.op("urem", &tid, warp);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        // This thread's `bᵀ` fragment offsets in a stage at k-step 0: a_r =
        // (column nl = 64·cw + 16·w + g + 8h, k = t + 4e) of `b`, in box nl/32
        // at row k, byte 4·(nl % 32), 128-byte swizzled (the box's pattern;
        // a k-step adds 8 whole rows, which the pattern repeats over).
        let fc = self.op("mul", &cw, WGMMA_M);
        let fw = self.op("mul", &w, WGMMA_M / WARPGROUP_WARPS);
        let nl0 = self.op("add", &fc, &fw);
        let nl0 = self.op("add", &nl0, &grp);
        let mut offs = Vec::new();
        for (h, e) in [(0u64, 0u64), (1, 0), (0, 1), (1, 1)] {
            if direct {
                break;
            }
            let nl = self.op("add", &nl0, 8 * h);
            let qb = self.op("udiv", &nl, span / 4);
            let qo = self.op("mul", &qb, box_b);
            let nb = self.op("urem", &nl, span / 4);
            let nb = self.op("mul", &nb, 4);
            let kr = self.op("add", &t4, 4 * e);
            let raw = self.op("mul", &kr, span);
            let raw = self.op("add", &raw, &nb);
            let sh = self.op("lshr", &raw, 7);
            let sh = self.op("and", &sh, 7);
            let sh = self.op("shl", &sh, 4);
            let sw = self.op("xor", &raw, &sh);
            let o = self.op("add", &qo, &sw);
            offs.push(self.op("add", &o, ab));
        }
        // wgmma-16bit, `a` from registers: this thread's fragment offsets in
        // a stage's f32 `a` box. Register `q` of a k16 step holds (row `64·cw
        // + 16·w + g + 8·(q % 2)`, k `2t + 8·(q / 2)` and its successor) — the
        // `m64k16` layout, warp `w` rows `16w..` — at digit column `k / 32`
        // (`bm·128` bytes apart), byte `128·row + 4·(k % 32)`, 128-byte
        // swizzled; a k16 step is half a 128-byte row, so `aoffs[p][q]` per
        // step parity `p` and the column a constant.
        let a_regs = direct && g.a_home == Home::Regs;
        let mut aoffs: Vec<Vec<String>> = Vec::new();
        if a_regs {
            assert_eq!(g.ea, 4, "a register `a` is staged f32");
            let t2 = self.op("mul", &t4, 2);
            for par in 0..2u64 {
                let mut v = Vec::new();
                for q in 0..4u64 {
                    let r = self.op("add", &nl0, 8 * (q % 2));
                    let raw = self.op("mul", &r, span);
                    let kin = self.op("add", &t2, par * 16 + 8 * (q / 2));
                    let kb = self.op("mul", &kin, 4);
                    let raw = self.op("add", &raw, &kb);
                    let sh = self.op("lshr", &raw, 7);
                    let sh = self.op("and", &sh, 7);
                    let sh = self.op("shl", &sh, 4);
                    v.push(self.op("xor", &raw, &sh));
                }
                aoffs.push(v);
            }
        }
        // Under `ws-persist`: tile `t` of the walk (`ctl`), its seeds, its
        // first ring step `t·nk`.
        let (row0, col0, t, tn, step0) = if persist {
            self.line("br label %ctl");
            self.out.push_str("ctl:\n");
            let (t, tn) = (self.tmp(), self.tmp());
            self.line(format!("{t} = phi i64 [0, %cons], [{tn}, %cnx]"));
            let (r0, c0) = tile_at(self, &cta, &t);
            for idx in 0..accs {
                let p = acc_slot(self, idx);
                self.line(format!("store float {seed}, ptr {p}, align 4"));
            }
            let step0 = self.op("mul", &t, nk);
            (r0, c0, t, tn, step0)
        } else {
            (row0, col0, String::new(), String::new(), String::new())
        };
        let from = if persist { "ctl" } else { "cons" };
        self.line("br label %ct");
        self.out.push_str("ct:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        // The k loop's latch: past the release of the previous stage where
        // `a` is a descriptor (released after this group's commit).
        let latch = if direct && !a_regs { "clt" } else { "cgo" };
        self.line(format!("{it} = phi i64 [0, %{from}], [{itn}, %{latch}]"));
        // The ring step: the tile's own `it`, or `t·nk + it` on the walk.
        let gs = if persist {
            self.op("add", &step0, &it)
        } else {
            it.clone()
        };
        let x = self.op("urem", &gs, st);
        let ph = self.op("udiv", &gs, st);
        let ph = self.op("and", &ph, 1);
        let fb = bar(self, 0, &x);
        self.mbar_wait(&fb, &ph, "cf");
        if rounds {
            let rdy = bar(self, 1, &x);
            self.mbar_wait(&rdy, &ph, "cr");
        }
        if direct && !a_regs {
            self.line("br label %cgo");
        } else {
            // PTX ISA: an `a` fragment register may not be rewritten before the
            // `wait_group` completing the `wgmma` that reads it — so tile `i − 1`'s
            // group is waited for here, after this tile's barriers (it computes
            // through them; the other consumer warpgroup's groups keep the
            // tensor core busy through the fragment loads), and its stage is
            // released at once: a stage is never released while a group reads it.
            self.asm("wgmma.wait_group.sync.aligned 0;");
            let rel = self.tmp();
            self.line(format!("{rel} = icmp ugt i64 {it}, 0"));
            let l0 = self.tmp();
            self.line(format!("{l0} = icmp eq i64 {lane}, 0"));
            let go = self.tmp();
            self.line(format!("{go} = and i1 {rel}, {l0}"));
            self.line(format!("br i1 {go}, label %crel, label %cgo"));
            self.out.push_str("crel:\n");
            let prev = self.op("sub", &gs, 1);
            let px = self.op("urem", &prev, st);
            let eb = bar(self, 2, &px);
            self.asm_args(
                "mbarrier.arrive.shared::cta.b64 _, [$0];",
                "r",
                &[format!("i32 {eb}")],
            );
            self.line("br label %cgo");
        }
        self.out.push_str("cgo:\n");
        let base = self.op("mul", &x, sb);
        let mut frags = Vec::new();
        // wgmma-16bit: the f32 `a` fragments, rounded to the format pairwise
        // as loaded (`cvt.rn.<fmt>x2.f32`, pinned before the fence).
        for kk in 0..if a_regs { bk / g.mma_k() } else { 0 } {
            let mut f = Vec::new();
            let col = (kk / 2) * (g.bm() * span);
            for o in &aoffs[(kk % 2) as usize] {
                let i = self.op("add", &base, o);
                let i = self.op("add", &i, col);
                let p = self.tmp();
                self.line(format!(
                    "{p} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {i}"
                ));
                let v = self.tmp();
                self.line(format!(
                    "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                ));
                let (x0, x1) = (self.tmp(), self.tmp());
                self.line(format!("{x0} = extractelement <2 x float> {v}, i64 0"));
                self.line(format!("{x1} = extractelement <2 x float> {v}, i64 1"));
                let r = self.tmp();
                self.line(format!(
                    "{r} = call i32 asm sideeffect \"cvt.rn.{}x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {x1}, float {x0})",
                    g.fmt.ptx()
                ));
                f.push(r);
            }
            frags.push(f);
        }
        for kk in 0..if direct { 0 } else { bk / WGMMA_K } {
            let mut f = Vec::new();
            for o in &offs {
                let i = self.op("add", &base, o);
                let i = self.op("add", &i, kk * WGMMA_K * span);
                let p = self.tmp();
                self.line(format!(
                    "{p} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {i}"
                ));
                let v = self.tmp();
                self.line(format!("{v} = load float, ptr addrspace(3) {p}, align 4"));
                f.push(self.round_tf32(&v, true));
            }
            frags.push(f);
        }
        self.asm("wgmma.fence.sync.aligned;");
        // A matrix descriptor of the stage bytes `off` (an i64 value) in:
        // its start address's 14 bits, then `fields`.
        let desc_at = |k: &mut Self, off: &str, fields: u64| {
            let a32 = k.smem_at(off);
            let a64 = k.tmp();
            k.line(format!("{a64} = zext i32 {a32} to i64"));
            let enc = k.op("lshr", &a64, 4);
            let enc = k.op("and", &enc, 0x3fff);
            k.op("or", &enc, fields)
        };
        if direct {
            // wgmma-16bit: `c = a·b`. `a` — the warpgroup's 64 rows `fc..` of
            // the `a` box (K-major, its rows 128 bytes apart) — from its
            // descriptor or the fragments above; `b` from its descriptor:
            // k-major (`bn` rows; a k16 step 32 bytes into the digit
            // columns, `bn·128` apart), or lanes-major (MN-major, the
            // transpose bit: a k16 step 16 rows of the slabs, LBO the slab
            // stride).
            let sty = format!("{{ {} }}", vec!["float"; accs].join(", "));
            let ao = self.op("mul", &fc, span);
            let ao = self.op("add", &base, &ao);
            let adesc0 = (!a_regs).then(|| desc_at(self, &ao, g.desc_fields()));
            let bo = self.op("add", &base, ab);
            let bfields = match g.b_kmajor {
                true => g.desc_fields(),
                false => g.desc_mn_fields(),
            };
            let bdesc0 = desc_at(self, &bo, bfields);
            let p = g.fmt.ptx();
            let mk = g.mma_k();
            let acc_list = (0..accs)
                .map(|i| format!("${i}"))
                .collect::<Vec<_>>()
                .join(",");
            let tb = u8::from(!g.b_kmajor);
            let (text, ins) = match a_regs {
                true => (
                    format!(
                        "wgmma.mma_async.sync.aligned.m64n{n}k{mk}.f32.{p}.{p} {{{acc_list}}}, {{${},${},${},${}}}, ${}, 1, 1, 1, {tb};",
                        accs,
                        accs + 1,
                        accs + 2,
                        accs + 3,
                        accs + 4
                    ),
                    vec!["r", "r", "r", "r", "l"],
                ),
                false => (
                    format!(
                        "wgmma.mma_async.sync.aligned.m64n{n}k{mk}.f32.{p}.{p} {{{acc_list}}}, ${}, ${}, 1, 1, 1, 0, {tb};",
                        accs,
                        accs + 1
                    ),
                    vec!["l", "l"],
                ),
            };
            let mut cons = vec!["=f".to_string(); accs];
            cons.extend(ins.into_iter().map(String::from));
            cons.extend((0..accs).map(|i| i.to_string()));
            let cons = cons.join(",");
            // `kk·32` bytes into a k-major box of `rows` rows.
            let kstep = |kk: u64, rows: u64| {
                let kb = kk * 32;
                kb / span * (rows * span) + kb % span
            };
            for kk in 0..bk / mk {
                let boff = match g.b_kmajor {
                    true => kstep(kk, bn),
                    false => kk * mk * span,
                };
                let bdesc = self.op("add", &bdesc0, boff >> 4);
                let mut args = match &adesc0 {
                    Some(a0) => {
                        let adesc = self.op("add", a0, kstep(kk, g.bm()) >> 4);
                        format!("i64 {adesc}, i64 {bdesc}")
                    }
                    None => {
                        let f = &frags[kk as usize];
                        format!(
                            "i32 {}, i32 {}, i32 {}, i32 {}, i64 {bdesc}",
                            f[0], f[1], f[2], f[3]
                        )
                    }
                };
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
        }
        let desc0 = if direct {
            String::new()
        } else {
            desc_at(self, &base, g.desc_fields())
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
        for (kk, f) in frags.iter().enumerate().filter(|_| !direct) {
            // k-step `kk` starts `kk·32` bytes into the k digits: atom column
            // `kk·32 / 128` (`n·128` bytes apart), `kk·32 % 128` into its rows.
            let kb = kk as u64 * WGMMA_K * 4;
            let off = kb / span * (n * span) + kb % span;
            let desc = self.op("add", &desc0, off >> 4);
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
        if direct && !a_regs {
            // wgmma-16bit, both operands descriptors: no register to protect
            // — this tile's group stays in flight while the previous one
            // completes (`wait_group 1`) and its stage is released.
            self.asm("wgmma.wait_group.sync.aligned 1;");
            let rel = self.tmp();
            self.line(format!("{rel} = icmp ugt i64 {it}, 0"));
            let l0 = self.tmp();
            self.line(format!("{l0} = icmp eq i64 {lane}, 0"));
            let go = self.tmp();
            self.line(format!("{go} = and i1 {rel}, {l0}"));
            self.line(format!("br i1 {go}, label %crl, label %clt"));
            self.out.push_str("crl:\n");
            let prev = self.op("sub", &gs, 1);
            let px = self.op("urem", &prev, st);
            let eb = bar(self, 2, &px);
            self.asm_args(
                "mbarrier.arrive.shared::cta.b64 _, [$0];",
                "r",
                &[format!("i32 {eb}")],
            );
            self.line("br label %clt");
            self.out.push_str("clt:\n");
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %cd, label %ct"));

        self.out.push_str("cd:\n");
        self.asm("wgmma.wait_group.sync.aligned 0;");
        if persist {
            // The tile's last stage, released now: the producer fills the
            // next tile's stages under this epilogue.
            let l0 = self.tmp();
            self.line(format!("{l0} = icmp eq i64 {lane}, 0"));
            self.line(format!("br i1 {l0}, label %cdr, label %cde"));
            self.out.push_str("cdr:\n");
            let lst = self.op("add", &step0, nk - 1);
            let lx = self.op("urem", &lst, st);
            let eb = bar(self, 2, &lx);
            self.asm_args(
                "mbarrier.arrive.shared::cta.b64 _, [$0];",
                "r",
                &[format!("i32 {eb}")],
            );
            self.line("br label %cde");
            self.out.push_str("cde:\n");
        }
        if direct {
            // wgmma-16bit: accumulator `4j + 2h + e` is c(row `64·cw + 16·w
            // + g + 8h`, column `8j + 2t + e`) of the tile — each pair of
            // columns one 8-byte store, through the epilogue where there is
            // one.
            let er = self.op("add", &row0, &nl0);
            let er = self.op("mul", &er, s.c);
            let t2 = self.op("mul", &t4, 2);
            let ec = self.op("add", &col0, &t2);
            let ebase = self.op("add", &er, &ec);
            for jj in 0..n / 8 {
                for h in 0..2u64 {
                    let mut xs = Vec::new();
                    for e in 0..2u64 {
                        let q = (4 * jj + 2 * h + e) as usize;
                        let p = acc_slot(self, q);
                        let xv = self.tmp();
                        self.line(format!("{xv} = load float, ptr {p}, align 4"));
                        xs.push(xv);
                    }
                    let oi = self.op("add", &ebase, 8 * h * s.c + 8 * jj);
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
        let er = self.op("mul", &t4, 2);
        let er = self.op("add", &er, &row0);
        let er = self.op("mul", &er, s.c);
        let ec = self.op("add", &col0, &fc);
        let ec = self.op("add", &ec, &fw);
        let ec = self.op("add", &ec, &grp);
        let ebase = self.op("add", &er, &ec);
        for jj in 0..if direct { 0 } else { n / 8 } {
            for h in 0..2u64 {
                for e in 0..2u64 {
                    let q = (4 * jj + 2 * h + e) as usize;
                    let p = acc_slot(self, q);
                    let xv = self.tmp();
                    self.line(format!("{xv} = load float, ptr {p}, align 4"));
                    let oi = self.op("add", &ebase, (8 * jj + e) * s.c + 8 * h);
                    if self.epi.is_some() {
                        self.store_cells(&[xv], &oi);
                        continue;
                    }
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store float {xv}, ptr addrspace(1) {op}, align 4"));
                }
            }
        }
        if persist {
            self.line("br label %cnx");
            self.out.push_str("cnx:\n");
            self.line(format!("{tn} = add i64 {t}, 1"));
            let more = self.tmp();
            self.line(format!("{more} = icmp ult i64 {tn}, {n_t}"));
            self.line(format!("br i1 {more}, label %ctl, label %cend"));
            self.out.push_str("cend:\n");
        }
        self.out.push_str("  ret void\n}\n");
        // The tensor maps' recipe (`mapal_gpu.c` `tmap_of`).
        let words: Vec<String> = std::iter::once(maps.len() as u64)
            .chain(maps.iter().flatten().copied())
            .map(|v| format!("i64 {v}"))
            .collect();
        self.out.push_str(&format!(
            "@{name}_tmap = addrspace(1) constant [{} x i64] [{}], align 8\n",
            words.len(),
            words.join(", ")
        ));
    }
}
