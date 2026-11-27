//! `Kernel` kernel header and the naive (one cell a thread) kernels.

use super::*;

impl Kernel<'_> {
    /// `blocks` > 0 also emits `.minnctapersm`: the geometry assumed that many
    /// resident blocks per SM when it sized the register tile and the panels,
    /// so ptxas is held to the register budget that makes it true (without
    /// it, sm_90 allocated 166 registers and one block fit — S52 v1).
    pub(super) fn header(&mut self, name: &str, threads: u64, align: u64, blocks: u64) {
        let min_blocks = if blocks > 0 {
            format!(" \"nvvm.minctasm\"=\"{blocks}\"")
        } else {
            String::new()
        };
        // plan-cost `linear`: one `(a, b)` pair per term; `%a`, `%b` select
        // the block's term in the entry block.
        let ab = |t: u64| match t {
            0 if self.terms < 2 => ("%a".to_string(), "%b".to_string()),
            _ => (format!("%a{t}"), format!("%b{t}")),
        };
        let (a0, b0) = ab(0);
        // ADR-0043: a narrow operand's pointer is an allocation's (the
        // arena's `ARENA_ALIGN`, a device allocation's) like any operand's;
        // a launch that moves it by a captured number (`uniform-base`) moves
        // it by whole register-staged vectors of its format
        // ([`uniform_aligned`]: `lanes` elements, 8 bytes of bf16). Its
        // copies' adjacent loads then form 16-byte loads (cross-arch
        // 3a8e654f: the bf16 KV cache's scores kernel issued 2-byte loads
        // under `align 2` — 4× the f32 cache's load instructions, 2.6× its
        // time on the H100).
        let lanes = GPU_VEC_BYTES / elem_bytes_of(&self.site.elem);
        let narrow_align = |r: &mapal_ir::TileRead, moved: bool| {
            r.narrow.map_or(align, |n| match moved {
                true => align.min(lanes * n.bytes()),
                false => align,
            })
        };
        let moved = |a: bool| {
            self.site
                .radix
                .as_deref()
                .is_some_and(|x| !if a { &x.ua } else { &x.ub }.is_empty())
        };
        let (aa, ab_) = (
            narrow_align(&self.site.a, moved(true)),
            narrow_align(&self.site.b, moved(false)),
        );
        let mut more = String::new();
        for t in 1..self.terms {
            let (a, b) = ab(t);
            more += &format!(
                ", ptr addrspace(1) readonly align {aa} {a}, ptr addrspace(1) readonly align {ab_} {b}"
            );
        }
        let ro = if self.terms < 2 {
            "noalias readonly"
        } else {
            "readonly"
        };
        // hopper-wgmma-tma: each tensor map a 128-byte parameter by value.
        for m in 0..self.tmap.len() {
            more += &format!(", ptr byval([128 x i8]) align 64 \"nvvm.grid_constant\" %tm{m}");
        }
        self.out.push_str(&format!(
            "define ptx_kernel void @{name}(\
             ptr addrspace(1) {ro} align {aa} {a0}, \
             ptr addrspace(1) {ro} align {ab_} {b0}, \
             ptr addrspace(1) noalias align {align} %o{}{more}) \"nvvm.maxntid\"=\"{threads}\"{min_blocks} {{\n\
             entry:\n",
            self.epi_params(align)
        ));
        if self.terms > 1 {
            let y = self.tmp();
            self.line(format!("{y} = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()"));
            let term = self.tmp();
            self.line(format!(
                "{term} = udiv i32 {y}, {}",
                self.gy_out * self.splitk
            ));
            for (x, last) in [("a", "%a"), ("b", "%b")] {
                let mut prev = format!("%{x}0");
                for t in 1..self.terms {
                    let (hit, sel) = (self.tmp(), self.tmp());
                    let sel = if t + 1 == self.terms {
                        last.to_string()
                    } else {
                        sel
                    };
                    self.line(format!("{hit} = icmp eq i32 {term}, {t}"));
                    self.line(format!(
                        "{sel} = select i1 {hit}, ptr addrspace(1) %{x}{t}, ptr addrspace(1) {prev}"
                    ));
                    prev = sel;
                }
            }
        }
        self.epi_entry();
        // A dependent launch may start while its producer drains: nothing is
        // read before the producer's grid is complete. Then let this grid's
        // own dependent start. Both are no-ops on a plain launch.
        if self.pdl {
            self.asm("griddepcontrol.wait;");
            // cross-arch: the dependent launches at this kernel's exit unless
            // `pdl-early` (its blocks would wait resident beside this grid's).
            if self.rungs.pdl_early {
                self.asm("griddepcontrol.launch_dependents;");
            }
        }
    }

    /// One thread per output cell, `k` walked in order — the correctness path
    /// for any affine site the tiled geometry does not cover.
    pub(super) fn naive(&mut self, name: &str, threads: u64) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        self.header(name, threads, e, 0);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        // S62 `splitk`: the grid's `y` is the partition.
        if self.splitk > 1 {
            let y = self.sreg("ctaid.y");
            self.z = Some(match self.terms {
                1 => y,
                _ => self.op("urem", &y, self.splitk),
            });
        }
        let cell0 = self.op("mul", &bx, threads);
        let cell = self.op("add", &cell0, &tid);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult {ix} {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let rx = s.radix.as_deref();
        let (ra, rb) = (rx.and_then(|x| x.a.as_ref()), rx.and_then(|x| x.b.as_ref()));
        let base = |k: &mut Self, r: &mapal_ir::TileRead| {
            if let Some(m) = if std::ptr::eq(r, &s.a) { ra } else { rb } {
                return k.radix_at(m, &i, &lane);
            }
            let x = k.op("mul", &i, r.ci);
            let y = k.op("mul", &lane, r.clane);
            let z = k.op("add", &x, &y);
            k.op("add", &z, r.base)
        };
        // A split kernel folds `kc` steps from `z·kc`: the same loop over
        // operands moved `z·kc` along `k` (in bounds: `z·kc + kc ≤ k`).
        let kc = s.k / self.splitk;
        let abase = base(self, &s.a);
        let abase = self.at_z(abase, kc * s.a.ck);
        let bbase = base(self, &s.b);
        let bbase = self.at_z(bbase, kc * s.b.ck);
        let seed = self.split_seed(&seed);
        self.line("br label %kloop");
        self.out.push_str("kloop:\n");
        let (kv, acc, kn) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{kv} = phi {ix} [0, %body], [{kn}, %kloop]"));
        // The back-edge value is the step's result, known only after the body
        // is emitted; patched in below (the marker cannot collide with a `%v`).
        self.line(format!(
            "{acc} = phi {t} [{seed}, %body], [@ACC_NEXT@, %kloop]"
        ));
        // generic-fusion `window`: a read affine in `(k÷div, k%div)` (conv
        // taps) addresses `cq·(k÷div) + cr·(k%div)`; constant divisors, a
        // constant trip count — LLVM folds the division away.
        let load = |k: &mut Self, arr: &str, base: &str, r: &mapal_ir::TileRead, stream: bool| {
            let m = if std::ptr::eq(r, &s.a) { ra } else { rb };
            let off = match r.ksplit {
                // A split partition's `k` starts `z·kc` along the fold.
                _ if m.is_some() => {
                    let kz = k.at_z(kv.clone(), kc);
                    k.digits(&m.expect("radix").k, &kz)
                }
                Some(ks) => {
                    let q = k.op("udiv", &kv, ks.div);
                    let m = k.op("urem", &kv, ks.div);
                    let x = k.op("mul", &q, ks.cq);
                    let y = k.op("mul", &m, ks.cr);
                    k.op("add", &x, &y)
                }
                None => k.op("mul", &kv, r.ck),
            };
            let idx = k.op("add", base, &off);
            // ADR-0043: a narrow operand loads its format and widens (exact).
            if let Some(n) = r.narrow {
                let nt = crate::narrow::llvm_ty(n);
                let p = k.gep_ty(1, nt, arr, &idx);
                let (w, v) = (k.tmp(), k.tmp());
                k.line(format!(
                    "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
                    n.bytes()
                ));
                let tt = if n.bytes() == 1 {
                    k.tmp()
                } else {
                    String::new()
                };
                for l in crate::narrow::widen_lines(n, nt, &w, &t, &v, &tt) {
                    k.line(l);
                }
                return v;
            }
            let p = k.gep(1, arr, &idx);
            k.gload(&t, e, &p, e, stream)
        };
        // gapfix `stream-ld`: a thread a cell — `b` re-read by every row,
        // `a` by every column.
        let (a_cs, b_cs) = (
            s.a.narrow.is_none() && self.streams(false, s.c),
            s.b.narrow.is_none() && self.streams(true, s.rows),
        );
        self.streamed = [a_cs, b_cs];
        let av = load(self, "%a", &abase, &s.a, a_cs);
        let bv = load(self, "%b", &bbase, &s.b, b_cs);
        let accn = self.step(&acc, &av, &bv);
        self.out = self.out.replace("@ACC_NEXT@", &accn);
        self.line(format!("{kn} = add {ix} {kv}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq {ix} {kn}, {kc}"));
        self.line(format!("br i1 {fin}, label %store, label %kloop"));
        self.out.push_str("store:\n");
        let cell = if self.col {
            let jr = self.op("mul", &lane, s.rows);
            self.op("add", &jr, &i)
        } else if let Some(o) = rx.and_then(|x| x.out.as_ref()) {
            self.radix_at(o, &i, &lane)
        } else {
            cell
        };
        // A split kernel stores its partial into workspace slab `z`, or
        // (`splitk-atomic`) adds it into the output.
        let cell = self.slab(cell);
        if self.epi.is_some() {
            self.store_cells(std::slice::from_ref(&accn), &cell);
        } else {
            self.store_o(std::slice::from_ref(&accn), &cell);
        }
        self.line("br label %done");
        self.out.push_str("done:\n  ret void\n}\n");
    }

    /// **gemv-stream `vec-cells`: `v` adjacent cells a thread** ([`vec_fit`])
    /// — the one-thread-per-cell kernel with a thread's `v` lanes of one row:
    /// each `k` step loads `a` once and `b`'s `v` lanes as one 16-byte vector
    /// (`ld.global.nc.v4`), its `v` cells each folding in `k` order (the
    /// oracle's chain per cell, unchanged). A trip issues the loads of `u`
    /// steps before their FMAs (`u` = [`NAIVE_UNROLL`], halved until it
    /// divides the partition): the bytes in flight the split was sized for
    /// ([`TargetProfile::split_stream`]). Splits, terms, the workspace slab,
    /// atomics, column-major stores and epilogues as [`Self::naive`].
    pub(super) fn naive_cells(&mut self, name: &str, threads: u64, v: u64) {
        let s = self.site;
        let t = self.t.clone();
        let ix = self.idx;
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        self.header(name, threads, e, 0);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        if self.splitk > 1 {
            let y = self.sreg("ctaid.y");
            self.z = Some(match self.terms {
                1 => y,
                _ => self.op("urem", &y, self.splitk),
            });
        }
        let th0 = self.op("mul", &bx, threads);
        let th = self.op("add", &th0, &tid);
        let cell = self.op("mul", &th, v);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult {ix} {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let kc = s.k / self.splitk;
        // `a` ignores the lane (`clane` 0); `b`'s lanes are unit-stride.
        let ai = self.op("mul", &i, s.a.ci);
        let abase = self.plus(ai, s.a.base);
        let abase = self.at_z(abase, kc * s.a.ck);
        let bi = self.op("mul", &i, s.b.ci);
        let bl = self.op("add", &bi, &lane);
        let bbase = self.plus(bl, s.b.base);
        let bbase = self.at_z(bbase, kc * s.b.ck);
        let seed = self.split_seed(&seed);
        let mut u = NAIVE_UNROLL;
        while u > 1 && !kc.is_multiple_of(u) {
            u /= 2;
        }
        self.line("br label %kloop");
        self.out.push_str("kloop:\n");
        let (kv, kn) = (self.tmp(), self.tmp());
        self.line(format!("{kv} = phi {ix} [0, %body], [{kn}, %kloop]"));
        let accs: Vec<String> = (0..v).map(|_| self.tmp()).collect();
        for (j, acc) in accs.iter().enumerate() {
            self.line(format!(
                "{acc} = phi {t} [{seed}, %body], [@ACC_NEXT_{j}@, %kloop]"
            ));
        }
        // gapfix `stream-ld`: `b`'s vectors are re-read by every row.
        let b_cs = self.streams(true, s.rows);
        self.streamed = [false, b_cs];
        // The trip's loads first: `u` of `a`, `u` vectors of `b`.
        let mut steps = Vec::new();
        for q in 0..u {
            let kq = self.plus(kv.clone(), q);
            let ao = self.op("mul", &kq, s.a.ck);
            let aidx = self.op("add", &abase, &ao);
            let av = match s.a.narrow {
                Some(n) => {
                    let nt = crate::narrow::llvm_ty(n);
                    let p = self.gep_ty(1, nt, "%a", &aidx);
                    let (w, x) = (self.tmp(), self.tmp());
                    self.line(format!(
                        "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
                        n.bytes()
                    ));
                    let tt = if n.bytes() == 1 {
                        self.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(n, nt, &w, &t, &x, &tt) {
                        self.line(l);
                    }
                    x
                }
                None => {
                    let p = self.gep(1, "%a", &aidx);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr addrspace(1) {p}, align {e}"));
                    x
                }
            };
            let bo = self.op("mul", &kq, s.b.ck);
            let bidx = self.op("add", &bbase, &bo);
            let vt = format!("<{v} x {t}>");
            let bv = match s.b.narrow {
                Some(n) => {
                    let nt = crate::narrow::llvm_ty(n);
                    let p = self.gep_ty(1, nt, "%b", &bidx);
                    let w = self.gload(
                        &format!("<{v} x {nt}>"),
                        v * n.bytes(),
                        &p,
                        GPU_VEC_BYTES,
                        b_cs,
                    );
                    let x = self.tmp();
                    let nvt = format!("<{v} x {nt}>");
                    let tt = if n.bytes() == 1 {
                        self.tmp()
                    } else {
                        String::new()
                    };
                    for l in crate::narrow::widen_lines(n, &nvt, &w, &vt, &x, &tt) {
                        self.line(l);
                    }
                    x
                }
                None => {
                    let p = self.gep(1, "%b", &bidx);
                    self.gload(&vt, v * e, &p, GPU_VEC_BYTES, b_cs)
                }
            };
            steps.push((av, bv));
        }
        // Then each step's FMAs, `k` in order for every cell.
        let mut cur = accs.clone();
        for (av, bv) in steps {
            for (j, acc) in cur.iter_mut().enumerate() {
                let bj = self.tmp();
                self.line(format!("{bj} = extractelement <{v} x {t}> {bv}, i64 {j}"));
                *acc = self.step(acc, &av, &bj);
            }
        }
        for (j, next) in cur.iter().enumerate() {
            self.out = self.out.replace(&format!("@ACC_NEXT_{j}@"), next);
        }
        self.line(format!("{kn} = add {ix} {kv}, {u}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq {ix} {kn}, {kc}"));
        self.line(format!("br i1 {fin}, label %store, label %kloop"));
        self.out.push_str("store:\n");
        if self.col {
            // Column-major: each cell its own word (`o[j·rows + i]`).
            for (j, acc) in cur.iter().enumerate() {
                let lj = self.plus(lane.clone(), j as u64);
                let jr = self.op("mul", &lj, s.rows);
                let oc = self.op("add", &jr, &i);
                let oc = self.slab(oc);
                match self.epi.is_some() {
                    true => self.store_cells(std::slice::from_ref(acc), &oc),
                    false => self.store_o(std::slice::from_ref(acc), &oc),
                }
            }
        } else {
            // Whole 16-byte vectors of every stored output's type.
            let widest = self.epi.map_or(e, |ep| {
                ep.calls
                    .iter()
                    .filter(|c| c.store)
                    .map(|c| if c.ret == "double" { 8 } else { 4 })
                    .fold(e, u64::max)
            });
            let chunk = (GPU_VEC_BYTES / widest).clamp(1, v) as usize;
            let oc = self.slab(cell);
            for (n, part) in cur.chunks(chunk).enumerate() {
                let at = self.plus(oc.clone(), (n * chunk) as u64);
                match self.epi.is_some() {
                    true => self.store_cells(part, &at),
                    false => self.store_o(part, &at),
                }
            }
        }
        self.line("br label %done");
        self.out.push_str("done:\n  ret void\n}\n");
    }
}
