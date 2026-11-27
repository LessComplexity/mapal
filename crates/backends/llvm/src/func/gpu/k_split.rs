//! `Kernel` split-product helpers (tf32/bf16 parts, guards) and cold loads.

use super::*;

impl Kernel<'_> {
    /// f32-tensor-cores `split-tile` (tf32x3): an f32 element `x`'s TF32
    /// parts as instruction operands, and whether the guard takes `x` — not
    /// finite, rounding to infinity (`|x| ≥ 2¹²⁸ − 2¹⁰³`: its `hi` would
    /// overflow), or nonzero below [`SplitForm::exact_above`] (its `lo` would
    /// round on the subnormal grid). Everything else is split by integer
    /// rounding (tensor-cores-2: `cvt.rna.tf32.f32`'s ties-away rounding as
    /// `(x + 2¹²) & ¬(2¹³ − 1)` on the bits, its value class checks left to
    /// the guard — 25 SASS instructions an element on sm_80 became 10):
    /// `hi = rna(x)`, `lo = rna(x − hi)` (`x − hi` exact). A guarded
    /// element's parts are never read: its warp recomputes the tile by FFMA.
    pub(super) fn split_tf32(&mut self, x: &str) -> (String, String, String) {
        let rna = |k: &mut Self, b: &str| {
            let r = k.tmp();
            k.line(format!("{r} = add i32 {b}, 4096"));
            let h = k.tmp();
            k.line(format!("{h} = and i32 {r}, -8192"));
            h
        };
        let xb = self.tmp();
        self.line(format!("{xb} = bitcast float {x} to i32"));
        let thr = SplitForm::of(SplitKind::Tf32x3).exact_above as f32;
        let guard = self.split_guard(&xb, thr.to_bits(), TF32_RNA_OVERFLOW);
        let h = rna(self, &xb);
        let hf = self.tmp();
        self.line(format!("{hf} = bitcast i32 {h} to float"));
        let r = self.tmp();
        self.line(format!("{r} = fsub float {x}, {hf}"));
        let rb = self.tmp();
        self.line(format!("{rb} = bitcast float {r} to i32"));
        let l = rna(self, &rb);
        (h, l, guard)
    }

    /// `--gpu-off=split-issue`: the 5fcb401b tf32x3 cut — `cvt.rna` for
    /// both parts, the truncation select where `hi` overflows, the guard on
    /// infinities and subnormal-grid parts only.
    pub(super) fn split_tf32_cvt(&mut self, x: &str) -> (String, String, String) {
        let cvt = |k: &mut Self, v: &str| {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(float {v})"
            ));
            r
        };
        let xb = self.tmp();
        self.line(format!("{xb} = bitcast float {x} to i32"));
        let ax = self.tmp();
        self.line(format!("{ax} = and i32 {xb}, 2147483647"));
        let inf = self.tmp();
        self.line(format!("{inf} = icmp eq i32 {ax}, 2139095040"));
        // `0 < |x| < exact_above`: `|x|`'s bits less one below its bits less
        // one, unsigned.
        let thr = SplitForm::of(SplitKind::Tf32x3).exact_above as f32;
        let am = self.tmp();
        self.line(format!("{am} = add i32 {ax}, -1"));
        let tiny = self.tmp();
        self.line(format!("{tiny} = icmp ult i32 {am}, {}", thr.to_bits() - 1));
        let guard = self.tmp();
        self.line(format!("{guard} = or i1 {inf}, {tiny}"));
        let hb = cvt(self, x);
        let hx = self.tmp();
        self.line(format!("{hx} = and i32 {hb}, 2147483647"));
        let hinf = self.tmp();
        self.line(format!("{hinf} = icmp eq i32 {hx}, 2139095040"));
        let fin = self.tmp();
        self.line(format!("{fin} = xor i1 {inf}, true"));
        let ovf = self.tmp();
        self.line(format!("{ovf} = and i1 {hinf}, {fin}"));
        let tb = self.tmp();
        self.line(format!("{tb} = and i32 {xb}, -8192"));
        let h = self.tmp();
        self.line(format!("{h} = select i1 {ovf}, i32 {tb}, i32 {hb}"));
        let hf = self.tmp();
        self.line(format!("{hf} = bitcast i32 {h} to float"));
        let r = self.tmp();
        self.line(format!("{r} = fsub float {x}, {hf}"));
        let lb = cvt(self, &r);
        let l = self.tmp();
        self.line(format!("{l} = select i1 {inf}, i32 0, i32 {lb}"));
        (h, l, guard)
    }

    /// The split guard of an element with bits `xb`: `|x| ≥ big` (not
    /// finite, or its leading part rounds to infinity) or `0 < |x| < tiny`
    /// (a part would round on the subnormal grid) — one unsigned range
    /// check: `|x| − 1 ≥ tiny − 1` and `|x| < big` keep it off the guard.
    pub(super) fn split_guard(&mut self, xb: &str, tiny: u32, big: u32) -> String {
        let ax = self.tmp();
        self.line(format!("{ax} = and i32 {xb}, 2147483647"));
        let am = self.tmp();
        self.line(format!("{am} = add i32 {ax}, -1"));
        let small = self.tmp();
        self.line(format!("{small} = icmp ult i32 {am}, {}", tiny - 1));
        let large = self.tmp();
        self.line(format!("{large} = icmp uge i32 {ax}, {big}"));
        let g = self.tmp();
        self.line(format!("{g} = or i1 {small}, {large}"));
        g
    }

    /// review: **whether any accumulator of the thread's tiles is not
    /// finite** after the `k` loop. A cut form multiplies every bf16 part of
    /// the f32 operand by the narrow element: `±inf` times a part of `0`
    /// (`rn(1 − 1)`) or of the other sign is a NaN the exact product does
    /// not have — and every such product leaves its cell non-finite (`inf`
    /// or NaN; NaN and `inf − inf` stay so). One test a cell, once a tile:
    /// such a tile takes the guard (the FFMA cold path widens the narrow
    /// operand), where a test of every narrow fragment register cost 6 % of
    /// the hot loop's instructions (A100 `bf16x2n` 800 → 848, H100 PCIe 1368
    /// → 1448).
    pub(super) fn acc_nonfinite(
        &mut self,
        n: usize,
        acc_slot: &dyn Fn(&mut Self, usize) -> String,
    ) -> String {
        let mut any: Option<String> = None;
        for idx in 0..n {
            let p = acc_slot(self, idx);
            let v = self.tmp();
            self.line(format!("{v} = load float, ptr {p}, align 4"));
            let (x, m) = (self.tmp(), self.tmp());
            self.line(format!("{x} = bitcast float {v} to i32"));
            self.line(format!("{m} = and i32 {x}, 2147483647"));
            let t = self.tmp();
            self.line(format!("{t} = icmp uge i32 {m}, 2139095040"));
            any = Some(match any {
                None => t,
                Some(x) => {
                    let o = self.tmp();
                    self.line(format!("{o} = or i1 {x}, {t}"));
                    o
                }
            });
        }
        any.expect("an accumulator")
    }

    /// tensor-cores-2 `bf16x3n` / `bf16x2n`: an f32 pair's first `parts`
    /// bf16 parts (`mapal_ir::split::bf16_parts`, the lower `k` in each
    /// register's low half) — `p₀ = rn(x)`, `p₁ = rn(x − p₀)`, `p₂ = rn(x −
    /// p₀ − p₁)`, each residual exact, `p₂` the whole rest for `|x| ≥
    /// 2⁻¹¹⁰` (three parts: exact) — and whether
    /// the guard takes either element (not finite, rounding to infinity, or
    /// nonzero below [`SplitForm::exact_above`]).
    pub(super) fn split_bf16x2(
        &mut self,
        lo: &str,
        hi: &str,
        parts: usize,
    ) -> (Vec<String>, String) {
        let thr = SplitForm::of(SplitKind::Bf16x3n).exact_above as f32;
        let mut guards = Vec::new();
        for x in [lo, hi] {
            let xb = self.tmp();
            self.line(format!("{xb} = bitcast float {x} to i32"));
            guards.push(self.split_guard(&xb, thr.to_bits(), BF16_RN_OVERFLOW));
        }
        let guard = self.tmp();
        self.line(format!("{guard} = or i1 {}, {}", guards[0], guards[1]));
        let cvt = |k: &mut Self, l: &str, h: &str| {
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rn.bf16x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {h}, float {l})"
            ));
            r
        };
        // A part register's two values as f32 (bf16 is f32's top half).
        let widen = |k: &mut Self, p: &str| {
            let (l, h) = (k.tmp(), k.tmp());
            k.line(format!("{l} = shl i32 {p}, 16"));
            k.line(format!("{h} = and i32 {p}, -65536"));
            let (lf, hf) = (k.tmp(), k.tmp());
            k.line(format!("{lf} = bitcast i32 {l} to float"));
            k.line(format!("{hf} = bitcast i32 {h} to float"));
            (lf, hf)
        };
        let sub = |k: &mut Self, a: &str, b: &str| {
            let r = k.tmp();
            k.line(format!("{r} = fsub float {a}, {b}"));
            r
        };
        // Each part the rounding of what the parts before it leave.
        let (mut l, mut h) = (lo.to_string(), hi.to_string());
        let mut out = Vec::new();
        for i in 0..parts {
            let p = cvt(self, &l, &h);
            if i + 1 < parts {
                let (w0, w1) = widen(self, &p);
                (l, h) = (sub(self, &l, &w0), sub(self, &h, &w1));
            }
            out.push(p);
        }
        (out, guard)
    }

    /// A split tile's per-thread guard flag `|=` this k-step's guards.
    pub(super) fn or_guards(&mut self, flag: &str, guards: &[String]) {
        let Some(first) = guards.first() else {
            return;
        };
        let mut any = first.clone();
        for x in &guards[1..] {
            let o = self.tmp();
            self.line(format!("{o} = or i1 {any}, {x}"));
            any = o;
        }
        let (z, f0, f1) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{z} = zext i1 {any} to i32"));
        self.line(format!("{f0} = load i32, ptr {flag}, align 4"));
        self.line(format!("{f1} = or i32 {f0}, {z}"));
        self.line(format!("store i32 {f1}, ptr {flag}, align 4"));
    }

    /// One `mma.sync.m16n8k16` of the 16-bit format `ptx` into the
    /// accumulator `c` (four floats or literals); its four results.
    pub(super) fn mma16_call(
        &mut self,
        ptx: &str,
        a: [&String; 4],
        b: [&String; 2],
        c: &[String],
    ) -> Vec<String> {
        let r = self.tmp();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{ptx}.{ptx}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        (0..4)
            .map(|q| {
                let d = self.tmp();
                self.line(format!(
                    "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                ));
                d
            })
            .collect()
    }

    /// One `mma.sync.m16n8k8` TF32 product into the accumulator `c` (four
    /// floats or literals); its four results.
    pub(super) fn mma_tf32_call(
        &mut self,
        a: &[String],
        b: &[String],
        c: &[String],
    ) -> Vec<String> {
        let r = self.tmp();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        (0..4)
            .map(|q| {
                let d = self.tmp();
                self.line(format!(
                    "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                ));
                d
            })
            .collect()
    }

    /// f32-tensor-cores `split-tile`: the guard's cold path (§2.5). A warp
    /// any of whose threads split an infinite operand (`vote.sync.any`) has
    /// its whole tile recomputed by FFMA from the operands in global memory
    /// over this piece's `k` (`kext` from `z·kext`) — IEEE products, so `inf ·
    /// b` keeps the oracle's value class where `hi·lo` would make `inf · 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn split_cold(
        &mut self,
        flag: &str,
        orow: &str,
        ocol: &str,
        seed: &str,
        kext: u64,
        (mt, nt): (usize, usize),
        acc_slot: &dyn Fn(&mut Self, usize) -> String,
    ) {
        let s = self.site;
        let f = self.tmp();
        self.line(format!("{f} = load i32, ptr {flag}, align 4"));
        let v = self.tmp();
        self.line(format!(
            "{v} = call i32 asm sideeffect \"{{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; selp.u32 $0, 1, 0, q; }}\", \"=r,r\"(i32 {f})"
        ));
        let c = self.tmp();
        self.line(format!("{c} = icmp ne i32 {v}, 0"));
        self.line(format!("br i1 {c}, label %cold, label %warm"));
        self.out.push_str("cold:\n");
        for idx in 0..mt * nt * 4 {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }
        let kb = self.at_z("0".to_string(), kext);
        let kb = self.plus(kb, 0);
        let ke = self.op("add", &kb, kext);
        self.line("br label %ck");
        self.out.push_str("ck:\n");
        let (kq, kn) = (self.tmp(), self.tmp());
        self.line(format!("{kq} = phi i64 [{kb}, %cold], [{kn}, %ck]"));
        let mut av = Vec::new();
        for mi in 0..mt as u64 {
            for half in 0..2u64 {
                let r = self.op("add", orow, mi * MMA_M + half * 8);
                let ri = self.op("mul", &r, s.a.ci);
                let ki = self.op("mul", &kq, s.a.ck);
                let i = self.op("add", &ri, &ki);
                let i = self.op("add", &i, s.a.base);
                let x = self.cold_load("%a", s.a.narrow, &i);
                av.push(x);
            }
        }
        let mut bv = Vec::new();
        for ni in 0..nt as u64 {
            for j in 0..2u64 {
                let c = self.op("add", ocol, ni * MMA_N + j);
                let ci = self.op("mul", &c, s.b.clane);
                let ki = self.op("mul", &kq, s.b.ck);
                let i = self.op("add", &ci, &ki);
                let i = self.op("add", &i, s.b.base);
                let x = self.cold_load("%b", s.b.narrow, &i);
                bv.push(x);
            }
        }
        for mi in 0..mt {
            for ni in 0..nt {
                for half in 0..2 {
                    for j in 0..2 {
                        let idx = (mi * nt + ni) * 4 + half * 2 + j;
                        let p = acc_slot(self, idx);
                        let a = &av[mi * 2 + half];
                        let b = &bv[ni * 2 + j];
                        let o = self.tmp();
                        self.line(format!("{o} = load float, ptr {p}, align 4"));
                        let d = self.tmp();
                        self.line(format!(
                            "{d} = call float @llvm.fma.f32(float {a}, float {b}, float {o})"
                        ));
                        self.line(format!("store float {d}, ptr {p}, align 4"));
                    }
                }
            }
        }
        self.line(format!("{kn} = add i64 {kq}, 1"));
        let e = self.tmp();
        self.line(format!("{e} = icmp eq i64 {kn}, {ke}"));
        self.line(format!("br i1 {e}, label %warm, label %ck"));
        self.out.push_str("warm:\n");
    }

    /// The cold path's operand element at `i` of `arr`, as f32: a narrow
    /// operand's exact widening (tensor-cores-2 `bf16x3n`).
    pub(super) fn cold_load(
        &mut self,
        arr: &str,
        narrow: Option<mapal_ir::NarrowFloat>,
        i: &str,
    ) -> String {
        let Some(n) = narrow else {
            let p = self.gep(1, arr, i);
            let x = self.tmp();
            self.line(format!("{x} = load float, ptr addrspace(1) {p}, align 4"));
            return x;
        };
        let nt = crate::narrow::llvm_ty(n);
        let p = self.gep_ty(1, nt, arr, i);
        let (w, v) = (self.tmp(), self.tmp());
        self.line(format!(
            "{w} = load {nt}, ptr addrspace(1) {p}, align {}",
            n.bytes()
        ));
        let tt = if n.bytes() == 1 {
            self.tmp()
        } else {
            String::new()
        };
        for l in crate::narrow::widen_lines(n, nt, &w, "float", &v, &tt) {
            self.line(l);
        }
        v
    }
}
