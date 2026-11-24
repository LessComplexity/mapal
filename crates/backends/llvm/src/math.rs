//! **Math builtins** (ADR-0046; `exp` ADR-0042): the module-level definitions
//! `@mapal.<builtin>.f32` / `.f64` every `Math` morphism calls. A body's text
//! is the same on the host and the device (`devmap`), so the realization is
//! chosen where the module is put together, per side and face, by
//! [`mapal_ir::math::choose`] over the side's declared units and definitions
//! ([`mapal_ir::math::Target`]: the host's on [`crate::profile::TargetProfile::math`],
//! a device's on [`crate::profile::Caps::math`]) — and printed from the
//! chosen realization's trace: one printer for every builtin, every
//! realization and every target.
//!
//! Each definition is `internal alwaysinline`, so the optimizer sees the
//! straight-line code at every call; the exact faces print no fast-math
//! flag (llc runs `-nvptx-fma-level=0`), the contracted reference
//! `contract` on its marked steps.

#[cfg(test)]
use mapal_ir::math::Real;
use mapal_ir::math::{self, B2, Cvt, Def, F2, FPred, Face, I2, IPred, MathFn, Op, Target, VTy, W};

/// The symbol a body calls for builtin `f` at width `bits`.
pub(crate) fn symbol(f: MathFn, bits: u32) -> String {
    format!("@mapal.{}.f{bits}", f.name())
}

/// One side's (host or device) realization inputs: `free` the order is
/// free (the default face), `contract` the face fuses steps, `hw` the
/// `math-hw` rung, `exp_hw` the `exp-hw` rung (ADR-0042's arm, `exp` alone),
/// `target` what the side declares.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Side<'t> {
    pub free: bool,
    pub contract: bool,
    pub hw: bool,
    pub exp_hw: bool,
    pub target: &'t Target,
}

impl Side<'_> {
    /// The realization of `f` at `bits` on this side.
    pub(crate) fn choose(&self, f: MathFn, bits: u32) -> math::Choice {
        let face = if self.free {
            Face::Default
        } else {
            Face::Exact
        };
        let hw = self.hw && (f != MathFn::Exp || self.exp_hw);
        math::choose(f, bits, face, self.contract, hw, self.target)
    }
}

fn fty(bits: u32) -> &'static str {
    if bits == 64 { "double" } else { "float" }
}

/// The definitions (and intrinsic declarations) `text` needs: every builtin
/// and width it calls and does not define, in the side's realization.
pub(crate) fn defs(text: &str, side: Side) -> String {
    let mut out = String::new();
    let mut need_fma = [false; 2];
    let mut units: Vec<(u32, &'static str)> = Vec::new();
    let mut clz = [false; 2];
    for f in MathFn::ALL {
        for bits in [32u32, 64] {
            let sym = symbol(f, bits);
            let defined = format!("define internal {} {sym}(", fty(bits));
            if !text.contains(&format!("{sym}(")) || text.contains(&defined) {
                continue;
            }
            let c = side.choose(f, bits);
            let def = math::trace(&c, side.target);
            out.push_str(&print(&def, &sym, c.fused));
            for s in &def.steps {
                match s.op {
                    Op::Fma => need_fma[usize::from(bits == 64)] = true,
                    Op::Unit(_, u) if !units.contains(&(bits, u)) => units.push((bits, u)),
                    Op::Clz(w) => clz[usize::from(w == W::I64)] = true,
                    _ => {}
                }
            }
        }
    }
    let decl = |out: &mut String, d: String| {
        if !text.contains(&d) && !out.contains(&d) {
            out.push_str(&d);
        }
    };
    if need_fma[0] {
        decl(
            &mut out,
            "declare float @llvm.fma.f32(float, float, float)\n".into(),
        );
    }
    if need_fma[1] {
        decl(
            &mut out,
            "declare double @llvm.fma.f64(double, double, double)\n".into(),
        );
    }
    for (bits, u) in units {
        let t = fty(bits);
        decl(&mut out, format!("declare {t} @{u}({t})\n"));
    }
    for (i, it) in ["i32", "i64"].into_iter().enumerate() {
        if clz[i] {
            decl(
                &mut out,
                format!("declare {it} @llvm.ctlz.{it}({it}, i1)\n"),
            );
        }
    }
    out
}

/// An LLVM float literal from a bit pattern of width `bits` (a `float`
/// prints as the `double` that holds it exactly; `+0` as `0.0`).
fn flit(bits: u32, b: u64) -> String {
    if b == 0 {
        return "0.0".into();
    }
    let v = if bits == 32 {
        f64::from(f32::from_bits(b as u32))
    } else {
        f64::from_bits(b)
    };
    format!("0x{:016X}", v.to_bits())
}

fn ity(w: W) -> &'static str {
    match w {
        W::I32 => "i32",
        W::I64 => "i64",
    }
}

/// A traced definition as one LLVM function `sym`. `contracted`: the steps
/// the trace marks contractible carry `contract`.
pub(crate) fn print(def: &Def, sym: &str, contracted: bool) -> String {
    let t = fty(def.bits);
    let iw = if def.bits == 64 { "i64" } else { "i32" };
    let vty = |v: VTy| -> &'static str {
        match v {
            VTy::F => t,
            VTy::I32 => "i32",
            VTy::I64 => "i64",
            VTy::B => "i1",
        }
    };
    // Each value's operand text: a literal, `%x`, its name or `%v.<i>`.
    let names: Vec<String> = def
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| match s.op {
            Op::CF(b) => flit(def.bits, b),
            Op::CI(w, v) => match w {
                W::I32 => (v as u32 as i32).to_string(),
                W::I64 => (v as i64).to_string(),
            },
            _ => match &s.name {
                Some(n) => format!("%{n}"),
                None => format!("%v.{i}"),
            },
        })
        .collect();
    let mut s = format!(
        "define internal {t} {sym}({t} %x) alwaysinline nounwind willreturn memory(none) {{\nentry:\n"
    );
    for (i, st) in def.steps.iter().enumerate() {
        let n = &names[i];
        let a = |k: usize| names[st.args[k].0 as usize].as_str();
        let fl = if contracted && st.contract {
            " contract"
        } else {
            ""
        };
        let line = match st.op {
            Op::Arg | Op::CF(_) | Op::CI(..) => continue,
            Op::F2(op) => {
                let o = match op {
                    F2::Add => "fadd",
                    F2::Sub => "fsub",
                    F2::Mul => "fmul",
                };
                format!("{n} = {o}{fl} {t} {}, {}", a(0), a(1))
            }
            Op::FNeg => format!("{n} = fneg {t} {}", a(0)),
            Op::Fma => format!(
                "{n} = call {t} @llvm.fma.f{}({t} {}, {t} {}, {t} {})",
                def.bits,
                a(0),
                a(1),
                a(2)
            ),
            Op::FCmp(p) => {
                let p = match p {
                    FPred::Ogt => "ogt",
                    FPred::Olt => "olt",
                    FPred::Oeq => "oeq",
                    FPred::Uno => "uno",
                };
                format!("{n} = fcmp {p} {t} {}, {}", a(0), a(1))
            }
            Op::I2(w, I2::MulHi) => {
                // The unsigned product's high word: llc forms `mul.hi` /
                // `mulx` from the widened product.
                let (it, wt, sh) = match w {
                    W::I32 => ("i32", "i64", 32),
                    W::I64 => ("i64", "i128", 64),
                };
                format!(
                    "{n}.a = zext {it} {} to {wt}\n  {n}.b = zext {it} {} to {wt}\n  \
                     {n}.p = mul {wt} {n}.a, {n}.b\n  {n}.s = lshr {wt} {n}.p, {sh}\n  \
                     {n} = trunc {wt} {n}.s to {it}",
                    a(0),
                    a(1)
                )
            }
            Op::I2(w, op) => {
                let o = match op {
                    I2::Add => "add",
                    I2::Sub => "sub",
                    I2::Mul => "mul",
                    I2::And => "and",
                    I2::Or => "or",
                    I2::Xor => "xor",
                    I2::Shl => "shl",
                    I2::LShr => "lshr",
                    I2::AShr => "ashr",
                    I2::MulHi => unreachable!("above"),
                };
                format!("{n} = {o} {} {}, {}", ity(w), a(0), a(1))
            }
            Op::Clz(w) => {
                let it = ity(w);
                format!("{n} = call {it} @llvm.ctlz.{it}({it} {}, i1 false)", a(0))
            }
            Op::ICmp(w, p) => {
                let p = match p {
                    IPred::Eq => "eq",
                    IPred::Ne => "ne",
                    IPred::Ult => "ult",
                    IPred::Ugt => "ugt",
                    IPred::Slt => "slt",
                    IPred::Sgt => "sgt",
                };
                format!("{n} = icmp {p} {} {}, {}", ity(w), a(0), a(1))
            }
            Op::B2(op) => {
                let o = match op {
                    B2::And => "and",
                    B2::Or => "or",
                    B2::Xor => "xor",
                };
                format!("{n} = {o} i1 {}, {}", a(0), a(1))
            }
            Op::Sel => {
                let ty = vty(st.ty);
                format!("{n} = select i1 {}, {ty} {}, {ty} {}", a(0), a(1), a(2))
            }
            Op::Cvt(c) => match c {
                Cvt::Bits => format!("{n} = bitcast {t} {} to {iw}", a(0)),
                Cvt::FromBits => format!("{n} = bitcast {iw} {} to {t}", a(0)),
                Cvt::FToSi(w) => format!("{n} = fptosi {t} {} to {}", a(0), ity(w)),
                Cvt::SiToF(w) => format!("{n} = sitofp {} {} to {t}", ity(w), a(0)),
                Cvt::UiToF(w) => format!("{n} = uitofp {} {} to {t}", ity(w), a(0)),
                Cvt::ZExt => format!("{n} = zext i32 {} to i64", a(0)),
                Cvt::Trunc => format!("{n} = trunc i64 {} to i32", a(0)),
                Cvt::BoolToI(w) => format!("{n} = zext i1 {} to {}", a(0), ity(w)),
            },
            Op::Unit(_, u) => format!("{n} = call {t} @{u}({t} {})", a(0)),
        };
        s.push_str("  ");
        s.push_str(&line);
        s.push('\n');
    }
    s.push_str(&format!("  ret {t} {}\n}}\n", names[def.out.0 as usize]));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapal_ir::math::{Err, Unit, UnitOp};

    const EX2: Unit = Unit {
        op: UnitOp::Ex2,
        bits: 32,
        sym: "llvm.nvvm.ex2.approx.ftz.f",
        err: Err::Bound {
            rel: 3.0 / 16_777_216.0,
            abs: 0.0,
        },
        lo: -0.5,
        hi: 0.5,
        cost: 1,
        specials: false,
        evidence: "test",
    };
    const DEV: Target = Target {
        units: &[EX2],
        defs: &[],
    };

    fn side(free: bool, contract: bool, hw: bool, target: &Target) -> Side<'_> {
        Side {
            free,
            contract,
            hw,
            exp_hw: true,
            target,
        }
    }

    /// ADR-0042's realization rule through the framework: the oracle off the
    /// default face and under the OFF arm; on it the hardware form where the
    /// part's `ex2` error meets the bound (f32), else the contracted steps.
    #[test]
    fn exp_realization_follows_face_arm_and_measured_error() {
        let r = |s: Side, bits| s.choose(MathFn::Exp, bits).real;
        assert_eq!(r(side(false, true, true, &DEV), 32), Real::Reference);
        assert_eq!(r(side(true, true, false, &DEV), 32), Real::Reference);
        assert!(matches!(r(side(true, true, true, &DEV), 32), Real::Form(_)));
        assert_eq!(r(side(true, true, true, &DEV), 64), Real::Contracted);
        assert_eq!(
            r(side(true, true, true, &Target::NONE), 32),
            Real::Contracted
        );
        assert_eq!(
            r(side(true, false, true, &Target::NONE), 32),
            Real::Reference
        );
        let mut s = side(true, true, true, &DEV);
        s.exp_hw = false;
        assert_eq!(r(s, 32), Real::Reference);
    }

    /// Definitions appear only for a width the text calls; the oracle's text
    /// carries no fast-math flag, the hardware form the `ex2` intrinsic.
    #[test]
    fn defs_follow_the_calls() {
        let call32 = "  %v = call float @mapal.exp.f32(float %a)\n";
        let d = defs(call32, side(false, true, true, &DEV));
        assert!(d.contains("define internal float @mapal.exp.f32(float %x)"));
        assert!(!d.contains("@mapal.exp.f64"));
        assert!(!d.contains("contract") && !d.contains("ex2"));
        let h = defs(call32, side(true, true, true, &DEV));
        assert!(h.contains("@llvm.nvvm.ex2.approx.ftz.f(float %a)"));
        assert!(h.contains("declare float @llvm.nvvm.ex2.approx.ftz.f(float)"));
        let c = defs(call32, side(true, true, true, &Target::NONE));
        assert!(c.contains("fmul contract float") && !c.contains("reassoc"));
        assert_eq!(defs("no calls", side(true, true, true, &DEV)), "");
        let both = format!("{call32}  %w = call double @mapal.sin.f64(double %b)\n");
        let d = defs(&both, side(false, true, true, &DEV));
        assert!(d.contains("define internal double @mapal.sin.f64(double %x)"));
        assert!(d.contains("declare i64 @llvm.ctlz.i64(i64, i1)"));
        assert!(d.contains("mul i128"));
    }

    /// ADR-0042's text, byte for byte: the oracle, the contracted steps and
    /// the `ex2` form of both widths as the pre-framework printer
    /// (`backends/llvm/src/exp.rs` at 2972c3d1) wrote them —
    /// `tests/golden/exp-adr-0042.ll`, dumped from it.
    #[test]
    fn exp_text_is_adr_0042s() {
        let golden = include_str!("../tests/golden/exp-adr-0042.ll");
        let text = "call float @mapal.exp.f32(x) call double @mapal.exp.f64(y)";
        let mut out = String::new();
        for (label, s) in [
            ("oracle", side(false, true, true, &DEV)),
            ("contracted", side(true, true, true, &Target::NONE)),
            ("hardware", side(true, true, true, &DEV)),
        ] {
            out.push_str(&format!("; == {label}\n"));
            out.push_str(&defs(text, s));
        }
        assert_eq!(out, golden);
    }
}
