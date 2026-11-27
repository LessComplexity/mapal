//! The streamed kernel's guards: visible band, walked rows, row loads, digits and the skip's finite check.
use super::*;

impl UnitGen<'_> {
    /// F5: `%tlo` and `%band` — the first window the block's rows see and
    /// how many (through the last one they see; a block that sees none: all
    /// `nb` from 0) — each window's class from the mask's two sides' ranges
    /// over the block's rows and the window's lanes ([`mapal_ir::GuardBox`]).
    pub(super) fn visible_band(&mut self, st: &Stream, gd: &mapal_ir::StreamGuard) {
        let gb = &gd.form;
        let (kb, nb, w) = (st.kb, st.nb(), st.group.axis);
        let r1 = self.op("add", "%i", st.b - 1);
        // mask-frontend: an open side's terms — its captured numbers (the
        // launch's arguments, `stream-numbers`) and its per-row load (the
        // block's rows' least and greatest) — in signed arithmetic.
        let open = !gb.closed();
        let side_rows = |g: &mut Self, sd: &mapal_ir::GuardSide| -> (String, String) {
            let mut lo = sd.base.to_string();
            let mut hi = sd.base.to_string();
            for d in &sd.row {
                let (a, b) = g.digit_range(d, "%i", &r1);
                lo = g.op("add", &lo, &a);
                hi = g.op("add", &hi, &b);
            }
            for d in &sd.uniform {
                let v = g.uniform_digit(gd.producer, d);
                lo = g.op("add", &lo, &v);
                hi = g.op("add", &hi, &v);
            }
            if let Some(l) = &sd.load {
                let (a, b) = g.row_load_range(st, gd.producer, l);
                lo = g.op("add", &lo, &a);
                hi = g.op("add", &hi, &b);
            }
            (lo, hi)
        };
        let (al, ah, bl, bh) = match &st.order {
            None => {
                let (al, ah) = side_rows(self, &gb.lhs);
                let (bl, bh) = side_rows(self, &gb.rhs);
                (al, ah, bl, bh)
            }
            // mask-frontend: a block walked in the group's order holds no
            // range of rows — each side's row part over its `b` rows.
            Some(_) => {
                let [al, ah, bl, bh] = self.walked_rows(st, gd);
                (al, ah, bl, bh)
            }
        };
        let pre = self.cur.clone();
        self.line("br label %GS");
        self.label("GS");
        self.line(format!("%gt = phi i64 [0, %{pre}], [%gtn, %GS]"));
        self.line(format!("%glo = phi i64 [{nb}, %{pre}], [%glon, %GS]"));
        self.line(format!("%ghi = phi i64 [0, %{pre}], [%ghin, %GS]"));
        let l0 = self.op("mul", "%gt", kb);
        let l1x = self.op("add", &l0, kb - 1);
        let l1c = self.tmp();
        self.line(format!("{l1c} = icmp ult i64 {l1x}, {w}"));
        let l1 = self.tmp();
        self.line(format!("{l1} = select i1 {l1c}, i64 {l1x}, i64 {}", w - 1));
        let side_lanes = |g: &mut Self, sd: &mapal_ir::GuardSide, lo: &str, hi: &str| {
            let (mut lo, mut hi) = (lo.to_string(), hi.to_string());
            for d in &sd.lane {
                let (a, b) = g.digit_range(d, &l0, &l1);
                lo = g.op("add", &lo, &a);
                hi = g.op("add", &hi, &b);
            }
            (lo, hi)
        };
        let (a0, a1) = side_lanes(self, &gb.lhs, &al, &ah);
        let (b0, b1) = side_lanes(self, &gb.rhs, &bl, &bh);
        // Always true / always false over the box.
        let cmp = |g: &mut Self, p: &str, x: &str, y: &str| {
            let c = g.tmp();
            g.line(format!("{c} = icmp {p} i64 {x}, {y}"));
            c
        };
        let pr = |p: &str| match open {
            true => p.replacen('u', "s", 1),
            false => p.to_string(),
        };
        let (t, f) = match gb.cmp {
            Operation::Le => (
                cmp(self, &pr("ule"), &a1, &b0),
                cmp(self, &pr("ugt"), &a0, &b1),
            ),
            Operation::Lt => (
                cmp(self, &pr("ult"), &a1, &b0),
                cmp(self, &pr("uge"), &a0, &b1),
            ),
            Operation::Ge => (
                cmp(self, &pr("uge"), &a0, &b1),
                cmp(self, &pr("ult"), &a1, &b0),
            ),
            Operation::Gt => (
                cmp(self, &pr("ugt"), &a0, &b1),
                cmp(self, &pr("ule"), &a1, &b0),
            ),
            // Equality decides no box here: every window mixed.
            _ => ("false".to_string(), "false".to_string()),
        };
        let masked = if gb.fold_when { f } else { t };
        let tmin = self.tmp();
        self.line(format!(
            "{tmin} = call i64 @llvm.umin.i64(i64 %glo, i64 %gt)"
        ));
        self.line(format!("%glon = select i1 {masked}, i64 %glo, i64 {tmin}"));
        let t1 = self.op("add", "%gt", 1);
        self.line(format!("%ghin = select i1 {masked}, i64 %ghi, i64 {t1}"));
        self.line("%gtn = add i64 %gt, 1");
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 %gtn, {nb}"));
        self.line(format!("br i1 {more}, label %GS, label %GSe"));
        self.label("GSe");
        let none = self.tmp();
        self.line(format!("{none} = icmp uge i64 %glon, {nb}"));
        self.line(format!("%tlo = select i1 {none}, i64 0, i64 %glon"));
        let span = self.op("sub", "%ghin", "%glon");
        self.line(format!("%band = select i1 {none}, i64 {nb}, i64 {span}"));
    }

    /// mask-frontend: both sides' row parts — base, row digits, per-row
    /// load, per-launch digits — over a walked block's `b` rows (group rows
    /// `order(i + r)`), each side's least and greatest (signed):
    /// `[lhs lo, lhs hi, rhs lo, rhs hi]`.
    pub(super) fn walked_rows(&mut self, st: &Stream, gd: &mapal_ir::StreamGuard) -> [String; 4] {
        let gb = &gd.form;
        let tag = format!("GW{}", self.n);
        self.n += 1;
        let pre = self.cur.clone();
        self.line(format!("br label %{tag}"));
        self.label(&tag);
        let r = format!("%{tag}r");
        let rn = format!("%{tag}rn");
        let names = ["al", "ah", "bl", "bh"].map(|x| format!("%{tag}{x}"));
        let next = ["al", "ah", "bl", "bh"].map(|x| format!("%{tag}{x}n"));
        self.line(format!("{r} = phi i64 [0, %{pre}], [{rn}, %{tag}e]"));
        for (j, nm) in names.iter().enumerate() {
            let init = if j % 2 == 0 { i64::MAX } else { i64::MIN };
            self.line(format!(
                "{nm} = phi i64 [{init}, %{pre}], [{}, %{tag}e]",
                next[j]
            ));
        }
        let p = self.op("add", "%i", &r);
        let row = self.walk_row(st, &p);
        let mut val = Vec::new();
        for sd in [&gb.lhs, &gb.rhs] {
            let mut v = sd.base.to_string();
            for d in &sd.row {
                let (x, _) = self.digit_range(d, &row, &row);
                v = self.op("add", &v, &x);
            }
            if let Some(l) = &sd.load {
                let x = self.row_load_at(gd.producer, l, &row);
                v = self.op("add", &v, &x);
            }
            for d in &sd.uniform {
                let x = self.uniform_digit(gd.producer, d);
                v = self.op("add", &v, &x);
            }
            val.push(v);
        }
        self.line(format!("br label %{tag}e"));
        self.label(&format!("{tag}e"));
        for j in 0..4 {
            let f = if j % 2 == 0 { "smin" } else { "smax" };
            self.line(format!(
                "{} = call i64 @llvm.{f}.i64(i64 {}, i64 {})",
                next[j],
                names[j],
                val[j / 2]
            ));
        }
        self.line(format!("{rn} = add i64 {r}, 1"));
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {rn}, {}", st.b));
        self.line(format!("br i1 {more}, label %{tag}, label %{tag}x"));
        self.label(&format!("{tag}x"));
        next
    }

    /// mask-frontend: `coef·x[base + Σ digits(row)]` of a row load at group
    /// row `row` (sign-extended), `x` the producer `k`'s epilogue array.
    pub(super) fn row_load_at(&mut self, k: usize, l: &mapal_ir::RowLoad, row: &str) -> String {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a guarded producer is a tile")
        };
        let j = site
            .post
            .as_ref()
            .and_then(|p| p.reads.iter().position(|&s| s == l.slot))
            .expect("a row load is an epilogue read");
        let USrc::Gmem(g) = mb.srcs[2 + j] else {
            unreachable!("a row load reads memory")
        };
        let ty = self.gmem_ty[g].clone();
        let mut at = l.base.to_string();
        for d in &l.index {
            let q = self.op("udiv", row, d.stride);
            let q = match d.radix {
                0 => q,
                rx => self.op("urem", &q, rx),
            };
            let t = self.op("mul", &q, d.coef);
            at = self.op("add", &at, &t);
        }
        let ptr = self.tmp();
        self.line(format!("{ptr} = getelementptr {ty}, ptr %xg{g}, i64 {at}"));
        let v = self.tmp();
        self.line(format!("{v} = load {ty}, ptr {ptr}"));
        let v = match ty.as_str() {
            "i64" => v,
            t => {
                let x = self.tmp();
                self.line(format!("{x} = sext {t} {v} to i64"));
                x
            }
        };
        self.op("mul", &v, l.coef)
    }

    /// mask-frontend: a guard side's per-launch digit `coef·((v ÷ stride) %
    /// radix)` of the producer `k`'s captured number `v` (its launch
    /// argument `%uv`, as i64).
    pub(super) fn uniform_digit(&mut self, k: usize, d: &mapal_ir::Digit) -> String {
        let ty =
            self.calls[k].as_ref().expect("an epilogue call").field_ty[d.slot as usize].clone();
        let v = match ty.as_str() {
            "i64" => format!("%uv{k}_{}", d.slot),
            t => {
                let x = self.tmp();
                self.line(format!("{x} = sext {t} %uv{k}_{} to i64", d.slot));
                x
            }
        };
        let q = match d.stride {
            1 => v,
            s => self.op("udiv", &v, s),
        };
        let r = match d.radix {
            0 => q,
            r => self.op("urem", &q, r),
        };
        self.op("mul", &r, d.coef)
    }

    /// mask-frontend: a guard side's per-row load over the block's rows —
    /// the least and greatest `coef·x[base + Σ digits(row)]` (signed), `x`
    /// the producer `k`'s epilogue array (its memory operand), each of the
    /// `b` rows read once.
    pub(super) fn row_load_range(
        &mut self,
        st: &Stream,
        k: usize,
        l: &mapal_ir::RowLoad,
    ) -> (String, String) {
        let tag = format!("GL{}", self.n);
        self.n += 1;
        let pre = self.cur.clone();
        self.line(format!("br label %{tag}"));
        self.label(&tag);
        let (r, lo, hi) = (format!("%{tag}r"), format!("%{tag}lo"), format!("%{tag}hi"));
        let (rn, lon, hin) = (
            format!("%{tag}rn"),
            format!("%{tag}lon"),
            format!("%{tag}hin"),
        );
        self.line(format!("{r} = phi i64 [0, %{pre}], [{rn}, %{tag}]"));
        self.line(format!(
            "{lo} = phi i64 [{}, %{pre}], [{lon}, %{tag}]",
            i64::MAX
        ));
        self.line(format!(
            "{hi} = phi i64 [{}, %{pre}], [{hin}, %{tag}]",
            i64::MIN
        ));
        let row = self.op("add", "%i", &r);
        let v = self.row_load_at(k, l, &row);
        self.line(format!(
            "{lon} = call i64 @llvm.smin.i64(i64 {lo}, i64 {v})"
        ));
        self.line(format!(
            "{hin} = call i64 @llvm.smax.i64(i64 {hi}, i64 {v})"
        ));
        self.line(format!("{rn} = add i64 {r}, 1"));
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {rn}, {}", st.b));
        self.line(format!("br i1 {more}, label %{tag}, label %{tag}e"));
        self.label(&format!("{tag}e"));
        (lon, hin)
    }

    /// `[lo, hi]` of digit `d` over `x ∈ [x0, x1]` (`radix 0`: the quotient).
    pub(super) fn digit_range(
        &mut self,
        d: &mapal_ir::AxisDigit,
        x0: &str,
        x1: &str,
    ) -> (String, String) {
        let q0 = self.op("udiv", x0, d.stride);
        let q1 = self.op("udiv", x1, d.stride);
        if d.radix == 0 {
            return (self.op("mul", &q0, d.coef), self.op("mul", &q1, d.coef));
        }
        let w0 = self.op("udiv", &q0, d.radix);
        let w1 = self.op("udiv", &q1, d.radix);
        let same = self.tmp();
        self.line(format!("{same} = icmp eq i64 {w0}, {w1}"));
        let m0 = self.op("urem", &q0, d.radix);
        let m1 = self.op("urem", &q1, d.radix);
        let lo = self.tmp();
        self.line(format!("{lo} = select i1 {same}, i64 {m0}, i64 0"));
        let hi = self.tmp();
        self.line(format!(
            "{hi} = select i1 {same}, i64 {m1}, i64 {}",
            d.radix - 1
        ));
        (self.op("mul", &lo, d.coef), self.op("mul", &hi, d.coef))
    }

    /// F6: does every skipped (wholly masked) window add each fold's
    /// identity? Each row's axis cells at the masked constant with the
    /// final per-row values: a running extreme's final unchanged by it, a
    /// sum's term `±0`, a consumer's row operand `±0` (its column operand is
    /// finite, `finite_plan`); the block's rows vote (one shared word).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn absorbed(
        &mut self,
        st: &Stream,
        lay: &Layout,
        gd: &mapal_ir::StreamGuard,
        x: &str,
        grow: &[String],
        shift: &[bool],
    ) -> String {
        let u = self.u;
        let n = u.members.len();
        let fin = self.finals(st, shift);
        let w = st.group.axis;
        let mut ok = "true".to_string();
        for y in 0..st.rt as usize {
            let mut mv: Vec<String> = vec![String::new(); n];
            mv[gd.producer] = super::const_literal(&mapal_ir::Value::F32(gd.form.masked as f32));
            let t = self.op("mul", &grow[y], w);
            for k in 0..n {
                let mb = &u.members[k];
                match (&st.group.roles[k], &mb.kind) {
                    (StreamRole::Cell { .. }, _) => {
                        for (sl, src) in mb.srcs.iter().enumerate() {
                            let USrc::Member { k: p, .. } = src else {
                                continue;
                            };
                            let v = match st.group.roles[*p] {
                                StreamRole::Fold { .. } => fin[*p][y].clone(),
                                _ => mv[*p].clone(),
                            };
                            self.subst.push((k, sl as u32, v));
                        }
                        mv[k] = self.cell(k, &t);
                        self.subst.clear();
                    }
                    (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                        let USrc::Member { k: p, .. } = mb.srcs[0] else {
                            continue;
                        };
                        let c = self.tmp();
                        match site.op {
                            mapal_ir::RowFoldOp::Select { .. } => {
                                let s = self.tmp();
                                self.step(&site.op, "float", &fin[k][y], &mv[p], &s);
                                self.line(format!("{c} = fcmp oeq float {s}, {}", fin[k][y]));
                            }
                            _ => self.line(format!("{c} = fcmp oeq float {}, 0.0", mv[p])),
                        }
                        ok = self.and(&ok, &c);
                    }
                    (StreamRole::Consume { .. }, _) => {
                        let USrc::Member { k: p, .. } = mb.srcs[0] else {
                            continue;
                        };
                        let c = self.tmp();
                        self.line(format!("{c} = fcmp oeq float {}, 0.0", mv[p]));
                        ok = self.and(&ok, &c);
                    }
                    _ => {}
                }
            }
        }
        // bf16-class `finite-check`: a consumer's column operand not proven
        // finite — the elements the skipped windows hold, read here.
        for &k in &gd.check {
            let c = self.skipped_finite(st, k);
            ok = self.and(&ok, &c);
        }
        // The block's vote: one shared word, cleared by any row that fails.
        let f = self.smem(lay.flag, "i32", "0");
        self.barrier();
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 %l, 0"));
        self.cond("Vz", &t0, |g| {
            g.line(format!("store i32 1, ptr addrspace(3) {f}"))
        });
        self.barrier();
        let bad = self.tmp();
        self.line(format!("{bad} = xor i1 {ok}, true"));
        self.cond("Vf", &bad, |g| {
            g.line(format!("store i32 0, ptr addrspace(3) {f}"))
        });
        self.barrier();
        let v = self.tmp();
        self.line(format!("{v} = load i32, ptr addrspace(3) {f}"));
        let pass = self.tmp();
        self.line(format!("{pass} = icmp ne i32 {v}, 0"));
        let _ = x;
        pass
    }

    /// bf16-class `finite-check`: is every element of consumer `k`'s
    /// column operand that the block's skipped windows hold finite? The
    /// windows past the visible band (`[tlo + band, tlo + nb)` mod `nb`),
    /// each key's row of the operand (`b(i, c, j)`, `c < site.c`, the
    /// block's rows' address as the windows' staging takes it), read in
    /// 16-byte chunks where the row is contiguous, strided over the block's
    /// threads; a key past the axis reads the axis's last (a superset: a
    /// non-finite element there only runs the windows). Exponent bits not
    /// all ones, on the bits (no float compare a fast-math flag may fold).
    pub(super) fn skipped_finite(&mut self, st: &Stream, k: usize) -> String {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a consumer is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[1] else {
            unreachable!("a tile's column operand is memory")
        };
        let site = site.clone();
        let bb = Addr::of(&site, Operand::B);
        let (w, kb, nb, threads) = (st.group.axis, st.kb, st.nb(), st.threads);
        let eb = win_bytes(&site);
        let per16 = 16 / eb;
        let v = if bb.vec_by(Ax::Lane, site.c, per16) {
            per16
        } else {
            1
        };
        // The format's exponent field: all ones is `∞` or NaN.
        let (ety, ity, ex) = match eb {
            2 => ("i16", "i16", 32640u32),
            _ => ("float", "i32", 2139095040u32),
        };
        let per_row = site.c / v;
        let per = kb * per_row;
        let first = self.walk_row(st, "%i");
        let i0 = self.member_row(st, k, &first);
        let rest = self.tmp();
        self.line(format!("{rest} = sub i64 {nb}, %band"));
        let total = self.op("mul", &rest, per);
        let skip0 = self.op("add", "%tlo", "%band");
        let tag = format!("Fc{k}_{}", self.n);
        self.n += 1;
        let pre = self.cur.clone();
        let (lp, end) = (tag.clone(), format!("{tag}e"));
        let c0 = self.tmp();
        self.line(format!("{c0} = icmp ult i64 %l, {total}"));
        self.line(format!("br i1 {c0}, label %{lp}, label %{end}"));
        self.label(&lp);
        let e = format!("%{tag}x");
        let acc = format!("%{tag}a");
        self.line(format!("{e} = phi i64 [%l, %{pre}], [{e}n, %{lp}]"));
        self.line(format!("{acc} = phi i1 [true, %{pre}], [{acc}n, %{lp}]"));
        let wi = self.op("udiv", &e, per);
        let rem = self.op("urem", &e, per);
        let r = self.op("udiv", &rem, per_row);
        let cq = self.op("urem", &rem, per_row);
        let cc = self.op("mul", &cq, v);
        let wa = self.op("add", &skip0, &wi);
        let win = self.op("urem", &wa, nb);
        let j0 = self.op("mul", &win, kb);
        let j = self.op("add", &j0, &r);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult i64 {j}, {w}"));
        let jj = self.tmp();
        self.line(format!("{jj} = select i1 {inside}, i64 {j}, i64 {}", w - 1));
        let at = self.addr_at(&bb, &i0, &cc, &jj);
        let src = self.tmp();
        self.line(format!(
            "{src} = getelementptr inbounds {ety}, ptr addrspace(1) %x{gi}, i64 {at}"
        ));
        let fin = if v > 1 {
            let vt = format!("<{v} x {ity}>");
            let splat = vec![format!("{ity} {ex}"); v as usize].join(", ");
            let x = self.tmp();
            self.line(format!("{x} = load {vt}, ptr addrspace(1) {src}, align 16"));
            let m = self.tmp();
            self.line(format!("{m} = and {vt} {x}, <{splat}>"));
            let c = self.tmp();
            self.line(format!("{c} = icmp ne {vt} {m}, <{splat}>"));
            let b = self.tmp();
            self.line(format!("{b} = bitcast <{v} x i1> {c} to i{v}"));
            let all = self.tmp();
            self.line(format!("{all} = icmp eq i{v} {b}, -1"));
            all
        } else {
            let x = self.tmp();
            self.line(format!(
                "{x} = load {ity}, ptr addrspace(1) {src}, align {eb}"
            ));
            let m = self.tmp();
            self.line(format!("{m} = and {ity} {x}, {ex}"));
            let c = self.tmp();
            self.line(format!("{c} = icmp ne {ity} {m}, {ex}"));
            c
        };
        self.line(format!("{acc}n = and i1 {acc}, {fin}"));
        self.line(format!("{e}n = add i64 {e}, {threads}"));
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {e}n, {total}"));
        self.line(format!("br i1 {more}, label %{lp}, label %{end}"));
        self.label(&end);
        let r = self.tmp();
        self.line(format!("{r} = phi i1 [true, %{pre}], [{acc}n, %{lp}]"));
        r
    }

    pub(super) fn and(&mut self, a: &str, b: &str) -> String {
        let r = self.tmp();
        self.line(format!("{r} = and i1 {a}, {b}"));
        r
    }
}
