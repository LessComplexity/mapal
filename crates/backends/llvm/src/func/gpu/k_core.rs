//! `Kernel` emitter helpers: values, epilogue plumbing, loads and the FMA step.

use super::*;

impl Kernel<'_> {
    pub(super) fn tmp(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    pub(super) fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    /// `dst = op IDX lhs, rhs`, returning `dst`. In `i32` every value is a
    /// partial sum of an index below 2³¹ ([`max_index`]), so `add`/`mul` carry
    /// `nuw nsw` — what lets LLVM fold a copy's constant into the address.
    pub(super) fn op(&mut self, code: &str, lhs: &str, rhs: impl std::fmt::Display) -> String {
        let t = self.tmp();
        let flags = match code {
            "add" | "mul" if self.idx == "i32" => " nuw nsw",
            _ => "",
        };
        self.line(format!("{t} = {code}{flags} {} {lhs}, {rhs}", self.idx));
        t
    }

    /// `code` (`udiv` / `urem`) of the `i64` value `v` (below 2³¹) by the
    /// constant `c`, in 32 bits, zero-extended back: a 32-bit magic division
    /// holds half the registers of a 64-bit one.
    pub(super) fn op32(&mut self, code: &str, v: &str, c: u64) -> String {
        let (n, r, w) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{n} = trunc i64 {v} to i32"));
        self.line(format!("{r} = {code} i32 {n}, {c}"));
        self.line(format!("{w} = zext i32 {r} to i64"));
        w
    }

    /// The block's `(column, row)` in the grid: `ctaid.x`, `ctaid.y`, remapped
    /// by [`raster_block`] when the derived group height is above 1.
    pub(super) fn block_ids(&mut self) -> (String, String) {
        // Under the column-major store (S54 `rowfold-layout`) the grid's
        // fastest axis runs down the rows too (the host swaps `gx`/`gy`): one
        // wave's blocks write consecutive column runs, not 512-byte pieces
        // 16 KB apart. That order is its own raster.
        if self.col {
            let by = self.sreg("ctaid.x");
            return (self.sreg("ctaid.y"), by);
        }
        let bx = self.sreg("ctaid.x");
        let by = self.sreg("ctaid.y");
        // S62 `splitk`: the partition rides above the output rows in `y`
        // (`gy_out` a constant: LLVM folds the division). No raster here —
        // the two are exclusive by derivation (`launch`).
        if self.splitk > 1 || self.terms > 1 {
            let row = self.op("urem", &by, self.gy_out);
            let z = self.op("udiv", &by, self.gy_out);
            if self.splitk > 1 {
                // plan-cost `linear`: `z = term·S + part`.
                let part = match self.terms {
                    1 => z,
                    _ => self.op("urem", &z, self.splitk),
                };
                self.z = Some(part);
            }
            return (bx, row);
        }
        let (gx, h) = self.raster;
        if h < 2 {
            return (bx, by);
        }
        self.line(format!("; raster: {h} block-rows per group"));
        let (c, r) = raster_block(self, &bx, &by, gx, h);
        (self.opaque(&c), self.opaque(&r))
    }

    /// `v` through a register move LLVM cannot see into. The remap bounds a
    /// block's row and column, so InstCombine turns the kernel's address adds
    /// into `or disjoint`, which NVPTX does not fold into immediate offsets:
    /// the H100 FMA tile went from 127 to 145 registers and one block per SM
    /// (`l2-raster-h100-pcie.txt`, `off-l2-gate`). Unbounded, the addresses
    /// are the row-major kernel's.
    pub(super) fn opaque(&mut self, v: &str) -> String {
        let (ty, bits, c) = match self.idx {
            "i32" => ("i32", 32, "r"),
            _ => ("i64", 64, "l"),
        };
        let t = self.tmp();
        self.line(format!(
            "{t} = call {ty} asm \"mov.b{bits} $0, $1;\", \"={c},{c}\"({ty} {v})"
        ));
        t
    }

    pub(super) fn sreg(&mut self, reg: &str) -> String {
        let r32 = self.tmp();
        self.line(format!("{r32} = call i32 @llvm.nvvm.read.ptx.sreg.{reg}()"));
        if self.idx == "i32" {
            return r32;
        }
        let r = self.tmp();
        self.line(format!("{r} = zext i32 {r32} to i64"));
        r
    }

    /// `ptr addrspace(AS) gep T, base, idx`. An `i32` index is zero-extended
    /// (it is non-negative): a GEP would sign-extend it, and copy 0's bare
    /// index then extends differently from copy `p`'s `index + constant`.
    pub(super) fn gep(&mut self, space: u32, base: &str, idx: &str) -> String {
        let t = self.t.clone();
        self.gep_ty(space, &t, base, idx)
    }

    /// [`Self::gep`] over elements of type `ty`.
    pub(super) fn gep_ty(&mut self, space: u32, ty: &str, base: &str, idx: &str) -> String {
        let idx = if self.idx == "i32" {
            let z = self.tmp();
            self.line(format!("{z} = zext nneg i32 {idx} to i64"));
            z
        } else {
            idx.to_string()
        };
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace({space}) {base}, i64 {idx}"
        ));
        p
    }

    /// The group values' types: the accumulator's, then each call's result.
    pub(super) fn epi_tys(&self, epi: &EpiKernel) -> Vec<String> {
        // bf16-class `round-post`: value 0 is the post body's result (the
        // narrow format where it rounds the cell as stored).
        let v0 = epi.post.as_ref().map_or(self.t.clone(), |p| p.ret.clone());
        std::iter::once(v0)
            .chain(epi.calls.iter().map(|c| c.ret.clone()))
            .collect()
    }

    /// The epilogue's extra parameters (S54): operand pointers `%x<i>`, then
    /// one output pointer `%y<j>` per stored member. Empty without one.
    pub(super) fn epi_params(&self, align: u64) -> String {
        let Some(epi) = self.epi else {
            return String::new();
        };
        let mut s = String::new();
        for i in 0..epi.caps {
            s.push_str(&format!(", ptr addrspace(1) readonly %x{i}"));
        }
        let stored = epi.calls.iter().filter(|c| c.store).count();
        for j in 0..stored {
            s.push_str(&format!(", ptr addrspace(1) noalias align {align} %y{j}"));
        }
        for j in 0..epi.uniforms.len() {
            s.push_str(&format!(", i64 %u{j}"));
        }
        // sweep-runtime-2 `streamk-epi`: the workspace the runtime appends
        // after the table's pointers (the sums, then the tiles' tickets).
        if self.ticket {
            s.push_str(&format!(
                ", ptr addrspace(1) noalias align {GPU_VEC_BYTES} %w"
            ));
        }
        s
    }

    /// Entry-block state of the epilogue: each operand pointer as the generic
    /// pointer a host body takes, and one one-cell buffer per `Cell` field —
    /// in the entry block, so inlining the body leaves a value in a register.
    pub(super) fn epi_entry(&mut self) {
        let Some(epi) = self.epi else {
            return;
        };
        for i in 0..epi.caps {
            self.line(format!(
                "%xg{i} = addrspacecast ptr addrspace(1) %x{i} to ptr"
            ));
        }
        for (j, t) in epi.uniforms.iter().enumerate() {
            let conv = uniform_from_i64(&format!("%uv{j}"), &format!("%u{j}"), t);
            self.out.push_str(&conv);
        }
        let tys = self.epi_tys(epi);
        for (k, c) in epi.calls.iter().enumerate() {
            for (f, a) in c.fields.iter().enumerate() {
                if let EpiArg::Cell(v) = *a {
                    self.line(format!("%cell{k}_{f} = alloca {}", tys[v]));
                }
            }
        }
    }

    /// The group's results at one cell (S54): each member's body called on
    /// the accumulator `acc` of output cell `cell` (an index value), in chain
    /// order. The bodies are the host's (or their `_cell` variants): the
    /// values are the host loop's, bit for bit.
    pub(super) fn epi_values(&mut self, acc: &str, cell: &str) -> Vec<String> {
        let epi = self.epi.expect("an epilogue");
        let t32 = if self.idx == "i32" {
            cell.to_string()
        } else {
            let x = self.tmp();
            self.line(format!("{x} = trunc i64 {cell} to i32"));
            x
        };
        let tys = self.epi_tys(epi);
        // `tile-post`: value 0 is the site's body after its fold.
        let acc = match &epi.post {
            Some(p) => {
                let mut agg = "poison".to_string();
                for (f, a) in p.fields.iter().enumerate() {
                    let (ty, v) = match *a {
                        EpiArg::Cap(i) => ("ptr".to_string(), format!("%xg{i}")),
                        EpiArg::Index => ("i32".to_string(), t32.clone()),
                        EpiArg::Null => ("ptr".to_string(), "null".to_string()),
                        EpiArg::Uniform(j) => (epi.uniforms[j].clone(), format!("%uv{j}")),
                        EpiArg::Value(_) | EpiArg::Cell(_) => {
                            unreachable!("an epilogue body reads no group value")
                        }
                    };
                    let n = self.tmp();
                    self.line(format!("{n} = insertvalue {} {agg}, {ty} {v}, {f}", p.arg));
                    agg = n;
                }
                let r = self.tmp();
                self.line(format!(
                    "{r} = call {} @{}({} {agg}, {} {acc})",
                    p.ret, p.callee, p.arg, self.t
                ));
                r
            }
            None => acc.to_string(),
        };
        let mut vals = vec![acc];
        for (k, c) in epi.calls.iter().enumerate() {
            let arg = if c.arg.starts_with('{') {
                let mut agg = "poison".to_string();
                for (f, a) in c.fields.iter().enumerate() {
                    let (ty, v) = match *a {
                        EpiArg::Value(i) => (tys[i].clone(), vals[i].clone()),
                        EpiArg::Cell(i) => {
                            let p = format!("%cell{k}_{f}");
                            self.line(format!("store {} {}, ptr {p}", tys[i], vals[i]));
                            ("ptr".to_string(), p)
                        }
                        EpiArg::Cap(i) => ("ptr".to_string(), format!("%xg{i}")),
                        EpiArg::Index => ("i32".to_string(), t32.clone()),
                        EpiArg::Null => ("ptr".to_string(), "null".to_string()),
                        EpiArg::Uniform(j) => (epi.uniforms[j].clone(), format!("%uv{j}")),
                    };
                    let n = self.tmp();
                    self.line(format!("{n} = insertvalue {} {agg}, {ty} {v}, {f}", c.arg));
                    agg = n;
                }
                agg
            } else {
                let [EpiArg::Value(i)] = c.fields[..] else {
                    unreachable!("a scalar body input is one group value")
                };
                vals[i].clone()
            };
            let r = self.tmp();
            self.line(format!(
                "{r} = call {} @{}({} {arg})",
                c.ret, c.callee, c.arg
            ));
            vals.push(r);
        }
        vals
    }

    /// Store `accs`, the accumulators of the consecutive output cells from
    /// index `oi`, through the epilogue: the site's output when stored, and
    /// each stored member's result — one vector store per output when
    /// `accs.len() > 1`, as the plain store.
    pub(super) fn store_cells(&mut self, accs: &[String], oi: &str) {
        let epi = self.epi.expect("an epilogue");
        // Value 0 (the site's: its accumulator, through a `tile-post` body),
        // then each call's.
        let mut per: Vec<Vec<String>> = vec![Vec::new(); epi.calls.len() + 1];
        for (j, acc) in accs.iter().enumerate() {
            let cell = self.plus(oi.to_string(), j as u64);
            for (k, v) in self.epi_values(acc, &cell).into_iter().enumerate() {
                per[k].push(v);
            }
        }
        let mut outs: Vec<(String, String, Vec<String>)> = Vec::new();
        if epi.store_out {
            let v0 = epi.post.as_ref().map_or(self.t.clone(), |p| p.ret.clone());
            outs.push(("%o".to_string(), v0, std::mem::take(&mut per[0])));
        }
        let mut y = 0;
        for (k, c) in epi.calls.iter().enumerate() {
            if c.store {
                outs.push((
                    format!("%y{y}"),
                    c.ret.clone(),
                    std::mem::take(&mut per[k + 1]),
                ));
                y += 1;
            }
        }
        let n = accs.len();
        for (p, ty, vals) in outs {
            let e = match ty.as_str() {
                "double" => 8,
                "bfloat" | "half" | "i16" => 2,
                _ => 4,
            };
            let (sty, v) = if n == 1 {
                (ty.clone(), vals[0].clone())
            } else {
                let vt = format!("<{n} x {ty}>");
                let mut vec = "poison".to_string();
                for (j, x) in vals.iter().enumerate() {
                    let nv = self.tmp();
                    self.line(format!(
                        "{nv} = insertelement {vt} {vec}, {ty} {x}, i64 {j}"
                    ));
                    vec = nv;
                }
                (vt, vec)
            };
            let gp = self.gep_ty(1, &ty, &p, oi);
            self.line(format!(
                "store {sty} {v}, ptr addrspace(1) {gp}, align {}",
                n * e
            ));
        }
    }

    /// `v + c`, or `v` itself when `c` is 0.
    /// `Σ coef·((x ÷ stride) % radix)` — one axis of a mixed-radix address
    /// ([`mapal_ir::RadixAddr`]); constant divisors, which LLVM strength-reduces.
    pub(super) fn digits(&mut self, ds: &[mapal_ir::AxisDigit], x: &str) -> String {
        let mut acc = "0".to_string();
        for d in ds {
            let q = if d.stride == 1 {
                x.to_string()
            } else {
                self.op("udiv", x, d.stride)
            };
            let r = self.op("urem", &q, d.radix);
            let v = self.op("mul", &r, d.coef);
            acc = self.op("add", &acc, &v);
        }
        acc
    }

    /// A mixed-radix address at `(i, lane)` without its `k` digits.
    pub(super) fn radix_at(&mut self, r: &mapal_ir::RadixAddr, i: &str, lane: &str) -> String {
        let x = self.digits(&r.i, i);
        let y = self.digits(&r.lane, lane);
        let z = self.op("add", &x, &y);
        self.plus(z, r.base)
    }

    pub(super) fn plus(&mut self, v: String, c: u64) -> String {
        if c == 0 { v } else { self.op("add", &v, c) }
    }

    /// S62 `splitk`: `v + z·scale` in a split kernel — partition `z`'s
    /// offset along `k` (`scale = kc·ck`) or into the workspace
    /// (`scale = rows·c`) — and `v` itself in the ordered one.
    pub(super) fn at_z(&mut self, v: String, scale: u64) -> String {
        match self.z.clone() {
            Some(z) => {
                let off = self.op("mul", &z, scale);
                self.op("add", &v, &off)
            }
            None => v,
        }
    }

    /// Output cell `v`'s index: workspace slab `z` in a workspace split
    /// kernel, `v` itself in an atomic one (its partials add into `%o`).
    pub(super) fn slab(&mut self, v: String) -> String {
        if self.atomic {
            return v;
        }
        let cells = self.site.rows * self.site.c;
        self.at_z(v, cells)
    }

    /// Store `v` at `%o[oi]`: a plain store, or (S63 `splitk-atomic`,
    /// `streamk`) `atomicrmw fadd`, device scope, relaxed, result unused —
    /// `red.global.add` in the SASS. One instruction per element: a vector
    /// value is split lane by lane.
    pub(super) fn store_o(&mut self, v: &[String], oi: &str) {
        let (t, e) = (self.t.clone(), elem_bytes_of(&self.site.elem));
        if !self.atomic {
            let n = v.len() as u64;
            let (ty, val) = if n == 1 {
                (t.clone(), v[0].clone())
            } else {
                let vt = format!("<{n} x {t}>");
                let mut vec = "poison".to_string();
                for (j, x) in v.iter().enumerate() {
                    let nv = self.tmp();
                    self.line(format!("{nv} = insertelement {vt} {vec}, {t} {x}, i64 {j}"));
                    vec = nv;
                }
                (vt, vec)
            };
            let p = self.gep(1, "%o", oi);
            self.line(format!(
                "store {ty} {val}, ptr addrspace(1) {p}, align {}",
                n * e
            ));
            return;
        }
        let sums = if self.ticket { "%w" } else { "%o" };
        for (j, x) in v.iter().enumerate() {
            let i = self.plus(oi.to_string(), j as u64);
            let p = self.gep(1, sums, &i);
            let r = self.tmp();
            self.line(format!(
                "{r} = atomicrmw fadd ptr addrspace(1) {p}, {t} {x} syncscope(\"device\") monotonic, align {e}"
            ));
        }
    }

    /// The fold's seed: partition 0 starts from the site's seed, every other
    /// partition from `0.0` (ADR-0040's partial folds), so the reduce adds
    /// the seed exactly once.
    /// gapfix `stream-ld`: **whether operand `b_side` (`a` / `b`) loads
    /// with the streaming policy** — read once by the grid (`reuse`: the
    /// blocks or threads that read each of its bytes, 1) and spanning more
    /// bytes than the L2 keeps ([`Gpu::l2_keep_bytes`], the cyclic set it
    /// re-reads from L2): such bytes come from DRAM whatever the policy, and
    /// the default one lets them displace what the L2 does keep (cuBLAS's
    /// gemv streams its matrix `LDG.E.EF`; gaps-disasm C2). Where the part
    /// has the policy ([`crate::profile::Caps::l2_evict`]) and the rung is on.
    pub(super) fn streams(&self, b_side: bool, reuse: u64) -> bool {
        let s = self.site;
        let g = &self.profile_gpu;
        let (r, cells) = match b_side {
            true => (&s.b, s.k * s.c),
            false => (&s.a, s.rows * s.k),
        };
        // A mixed-radix operand's bytes span its batch digits, which `k·c` /
        // `rows·k` do not count: no share is claimed for it.
        let rx = s.radix.as_deref();
        let radix = match b_side {
            true => rx.and_then(|x| x.b.as_ref()).is_some(),
            false => rx.and_then(|x| x.a.as_ref()).is_some(),
        };
        // Read once means each point's element is its own: the operand is a
        // dense tensor over the site's axes (`tensor_form`) whose distinct
        // elements are its `cells`. A window's signal (`s[t + k]`: the
        // lane and fold axes overlap, every element read by `k` cells — the
        // FIR) or a broadcast axis is re-read from the L2, and `cells`
        // overstates its bytes (gap-fixes merge: `fir_4194304_16`'s 16 MiB
        // signal priced as 256 MiB streamed evict-first, 0.965).
        let distinct = r
            .tensor_form(s.rows, s.c, s.k)
            .and_then(|t| t.dims.iter().try_fold(1u64, |n, d| n.checked_mul(d.extent)));
        self.rungs.stream_ld
            && !radix
            && reuse == 1
            && distinct == Some(cells)
            && crate::rules::stream_ld_pays(g, cells.saturating_mul(r.elem_bytes(&s.elem)))
    }

    /// The plan line's ` evict=a|b|ab` (the first line of the text): the
    /// operands [`Self::streams`] put on the streaming policy.
    pub(super) fn stream_field(&mut self) {
        let tag = match self.streamed {
            [true, true] => "ab",
            [true, false] => "a",
            [false, true] => "b",
            [false, false] => return,
        };
        if let Some(i) = self.out.find('\n') {
            self.out.insert_str(i, &format!(" evict={tag}"));
        }
    }

    /// A global load of `ty` (`bytes` wide) at `p`: under `stream` the
    /// streaming policy (`ld.global.cs.nc`, SASS `LDG.E.EF`) on its 32-bit
    /// words, the value bit-cast back; otherwise (and for a 2-byte or 1-byte
    /// load) LLVM's own load.
    pub(super) fn gload(
        &mut self,
        ty: &str,
        bytes: u64,
        p: &str,
        align: u64,
        stream: bool,
    ) -> String {
        let v = self.tmp();
        if !stream || !matches!(bytes, 4 | 8 | 16) || align < bytes {
            self.line(format!(
                "{v} = load {ty}, ptr addrspace(1) {p}, align {align}"
            ));
            return v;
        }
        let n = bytes / 4;
        let w = self.tmp();
        if n == 1 {
            self.line(format!(
                "{w} = call i32 asm \"ld.global.cs.nc.b32 $0, [$1];\", \"=r,l\"(ptr addrspace(1) {p})"
            ));
            self.line(format!("{v} = bitcast i32 {w} to {ty}"));
            return v;
        }
        let outs = (0..n)
            .map(|j| format!("${j}"))
            .collect::<Vec<_>>()
            .join(", ");
        let cons = vec!["=r"; n as usize].join(",");
        let st = format!("{{ {} }}", vec!["i32"; n as usize].join(", "));
        self.line(format!(
            "{w} = call {st} asm \"ld.global.cs.nc.v{n}.b32 {{{outs}}}, [${n}];\", \"{cons},l\"(ptr addrspace(1) {p})"
        ));
        let mut prev = "poison".to_string();
        for j in 0..n {
            let (x, y) = (self.tmp(), self.tmp());
            self.line(format!("{x} = extractvalue {st} {w}, {j}"));
            self.line(format!(
                "{y} = insertelement <{n} x i32> {prev}, i32 {x}, i64 {j}"
            ));
            prev = y;
        }
        self.line(format!("{v} = bitcast <{n} x i32> {prev} to {ty}"));
        v
    }

    pub(super) fn split_seed(&mut self, seed: &str) -> String {
        let Some(z) = self.z.clone().or(self.p0.clone()) else {
            return seed.to_string();
        };
        let (t, ix) = (self.t.clone(), self.idx);
        let (first, s) = (self.tmp(), self.tmp());
        self.line(format!("{first} = icmp eq {ix} {z}, 0"));
        self.line(format!("{s} = select i1 {first}, {t} {seed}, {t} 0.0"));
        s
    }

    /// Every staged copy's `(global index at k0 = 0, + constant, shared
    /// index, + constant)`; `at` turns panel coordinates into the two indices,
    /// `off` a coordinate delta into the two constants. Under `affine` a copy
    /// the lane map proves periodic ([`affine_delta`]) is copy 0's indices plus
    /// constants; otherwise (and on the OFF arm) each copy has its own chain.
    pub(super) fn copies(
        &mut self,
        map: LaneMap,
        n: u64,
        threads: u64,
        tid: &str,
        affine: bool,
        at: impl Fn(&mut Self, &str, &str) -> (String, String),
        off: impl Fn(u64, u64) -> (u64, u64),
    ) -> Vec<(String, u64, String, u64)> {
        let mut base: Option<(String, String)> = None;
        let mut out = Vec::new();
        for p in 0..n {
            let delta = (self.rungs.affine && affine).then(|| affine_delta(map, threads, p));
            if let Some(Some((d0, d1))) = delta {
                let (g, s) = match &base {
                    Some(b) => b.clone(),
                    None => {
                        let (r, c) = map.coords(self, &tid.to_string());
                        base.insert(at(self, &r, &c)).clone()
                    }
                };
                let (go, so) = off(d0, d1);
                out.push((g, go, s, so));
                continue;
            }
            let li = self.op("add", tid, p * threads);
            let (r, c) = map.coords(self, &li);
            let (g, s) = at(self, &r, &c);
            out.push((g, 0, s, 0));
        }
        out
    }

    /// One accumulation step, in the oracle's operand order:
    /// `acc ⊕ (a ⊗ b)` with each side placed as the source wrote it.
    pub(super) fn step(&mut self, acc: &str, a: &str, b: &str) -> String {
        let t = self.t.clone();
        let r = self.tmp();
        debug_assert!(matches!(
            self.arithmetic,
            Arithmetic::SeparateMulAdd | Arithmetic::OrderedFma
        ));
        if self.arithmetic == Arithmetic::OrderedFma {
            let bits = if t == "float" { 32 } else { 64 };
            self.line(format!(
                "{r} = call {t} @llvm.fma.f{bits}({t} {a}, {t} {b}, {t} {acc})"
            ));
            return r;
        }
        let (x, y) = if self.site.mul_a_first {
            (a, b)
        } else {
            (b, a)
        };
        let p = self.tmp();
        self.line(format!("{p} = fmul {t} {x}, {y}"));
        let (x, y) = if self.site.add_acc_first {
            (acc, p.as_str())
        } else {
            (p.as_str(), acc)
        };
        self.line(format!("{r} = fadd {t} {x}, {y}"));
        r
    }
}
