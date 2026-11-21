//! **`exp`** (ADR-0042): the module-level definitions of `@mapal.exp.f32` /
//! `@mapal.exp.f64` that every `Exp` morphism calls. A body's text is the
//! same on the host and the device (`devmap`), so the realization is chosen
//! where the module is put together, per side and face:
//!
//! - **oracle** — `mapal_ir::exp`'s algorithm op for op, separate IEEE steps
//!   (no fast-math flag; llc runs with `-nvptx-fma-level=0`): the exact,
//!   contract and fixed faces, and the OFF arm (`--gpu-off=exp-hw`).
//! - **contracted** — the same steps with `contract` on the reduction and the
//!   polynomial (FMA where the target has it): the host's default face —
//!   straight-line selects, no call, so a loop of it vectorizes.
//! - **hardware** — the reduction as two FMAs, `2^(r·log2e)` by
//!   `ex2.approx.ftz.f32` (MUFU.EX2), the oracle's scaling and special values:
//!   the device's default face on `f32` where the part's measured `ex2` error
//!   ([`crate::profile::Caps::ex2`]) meets the bound
//!   ([`mapal_ir::exp::ExpForm::hw_admits`]).
//!
//! Each definition is `internal alwaysinline`, so the optimizer sees the
//! straight-line code at every call.

use mapal_ir::exp::ExpForm;

/// One realization of a width's `exp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpRealize {
    Oracle,
    Contracted,
    Hardware,
}

/// The symbol a body calls for width `bits`.
pub(crate) fn symbol(bits: u32) -> &'static str {
    if bits == 64 {
        "@mapal.exp.f64"
    } else {
        "@mapal.exp.f32"
    }
}

/// The realizations of one side (host or device) of a module: `free` the
/// order is free (the default face), `contract` the face fuses steps, `hw`
/// the `exp-hw` rung, `ex2` the part's `ex2` error (device only).
pub(crate) fn choose(
    free: bool,
    contract: bool,
    hw: bool,
    device: Option<Option<u32>>,
) -> [ExpRealize; 2] {
    let pick = |form: ExpForm| {
        if !free || !hw {
            return ExpRealize::Oracle;
        }
        match device {
            Some(Some(u)) if form.hw_admits(u) => ExpRealize::Hardware,
            _ if contract => ExpRealize::Contracted,
            _ => ExpRealize::Oracle,
        }
    };
    [pick(mapal_ir::exp::F32), pick(mapal_ir::exp::F64)]
}

/// The definitions (and intrinsic declarations) `text` needs, for each width
/// it calls, in the realization `real` gives that width (`[f32, f64]`).
pub(crate) fn defs(text: &str, real: [ExpRealize; 2]) -> String {
    let mut out = String::new();
    let mut fma = [false; 2];
    let mut ex2 = false;
    for (form, r) in [(mapal_ir::exp::F32, real[0]), (mapal_ir::exp::F64, real[1])] {
        let sym = symbol(form.bits);
        let defined = format!("define internal {} {sym}(", ty(&form));
        if !text.contains(&format!("{sym}(")) || text.contains(&defined) {
            continue;
        }
        let hw = r == ExpRealize::Hardware && form.bits == 32;
        if hw {
            fma[0] = true;
            ex2 = true;
        }
        out.push_str(&def(&form, if hw { ExpRealize::Hardware } else { r }));
    }
    let decl = |out: &mut String, d: &str| {
        if !text.contains(d) {
            out.push_str(d);
        }
    };
    if fma[0] {
        decl(
            &mut out,
            "declare float @llvm.fma.f32(float, float, float)\n",
        );
    }
    if fma[1] {
        decl(
            &mut out,
            "declare double @llvm.fma.f64(double, double, double)\n",
        );
    }
    if ex2 {
        decl(
            &mut out,
            "declare float @llvm.nvvm.ex2.approx.ftz.f(float)\n",
        );
    }
    out
}

fn ty(form: &ExpForm) -> &'static str {
    if form.bits == 64 { "double" } else { "float" }
}

/// An LLVM literal of width `form` from its bit pattern (a `float` prints as
/// the `double` that holds it exactly).
fn lit(form: &ExpForm, bits: u64) -> String {
    let v = match form.bits {
        32 => f32::from_bits(bits as u32) as f64,
        _ => f64::from_bits(bits),
    };
    format!("0x{:016X}", v.to_bits())
}

/// One width's definition in realization `r`.
fn def(form: &ExpForm, r: ExpRealize) -> String {
    let t = ty(form);
    let it = if form.bits == 64 { "i64" } else { "i32" };
    let c = |b: u64| lit(form, b);
    let fl = if r == ExpRealize::Contracted {
        " contract"
    } else {
        ""
    };
    let (hi, lo) = (c(form.hi), c(form.lo));
    let mut s = format!(
        "define internal {t} {}({t} %x) alwaysinline nounwind willreturn memory(none) {{\nentry:\n",
        symbol(form.bits)
    );
    let mut l = |x: String| {
        s.push_str("  ");
        s.push_str(&x);
        s.push('\n');
    };
    l(format!("%ch = fcmp ogt {t} %x, {hi}"));
    l(format!("%x0 = select i1 %ch, {t} {hi}, {t} %x"));
    l(format!("%cl0 = fcmp olt {t} %x0, {lo}"));
    l(format!("%x1 = select i1 %cl0, {t} {lo}, {t} %x0"));
    l(format!("%cn0 = fcmp uno {t} %x1, 0.0"));
    l(format!("%xc = select i1 %cn0, {t} 0.0, {t} %x1"));
    let (log2e, sh) = (c(form.log2e), c(form.shifter));
    let (lhi, llo) = (c(form.ln2_hi), c(form.ln2_lo));
    let bits = form.bits;
    let p = match r {
        ExpRealize::Hardware => {
            l(format!(
                "%t = call {t} @llvm.fma.f{bits}({t} %xc, {t} {log2e}, {t} {sh})"
            ));
            l(format!("%kf = fsub {t} %t, {sh}"));
            l(format!("%nk = fneg {t} %kf"));
            l(format!(
                "%r0 = call {t} @llvm.fma.f{bits}({t} %nk, {t} {lhi}, {t} %xc)"
            ));
            l(format!(
                "%r = call {t} @llvm.fma.f{bits}({t} %nk, {t} {llo}, {t} %r0)"
            ));
            l(format!("%a = fmul {t} %r, {log2e}"));
            l(format!(
                "%p0 = call {t} @llvm.nvvm.ex2.approx.ftz.f({t} %a)"
            ));
            "%p0".to_string()
        }
        _ => {
            l(format!("%t0 = fmul {t} %xc, {log2e}"));
            l(format!("%t = fadd {t} %t0, {sh}"));
            l(format!("%kf = fsub {t} %t, {sh}"));
            l(format!("%h = fmul{fl} {t} %kf, {lhi}"));
            l(format!("%r0 = fsub{fl} {t} %xc, %h"));
            l(format!("%lw = fmul{fl} {t} %kf, {llo}"));
            l(format!("%r = fsub{fl} {t} %r0, %lw"));
            let cs = form.coefs();
            let n = form.degree as usize;
            let mut p = c(cs[n]);
            for i in (0..n).rev() {
                l(format!("%m{i} = fmul{fl} {t} {p}, %r"));
                l(format!("%p{i} = fadd{fl} {t} %m{i}, {}", c(cs[i])));
                p = format!("%p{i}");
            }
            p
        }
    };
    let (bias, mant) = (form.bias, form.mant);
    l(format!("%k = fptosi {t} %kf to {it}"));
    l(format!("%k1 = ashr {it} %k, 1"));
    l(format!("%k2 = sub {it} %k, %k1"));
    for k in ["k1", "k2"] {
        l(format!("%b{k} = add {it} %{k}, {bias}"));
        l(format!("%e{k} = shl {it} %b{k}, {mant}"));
        l(format!("%s{k} = bitcast {it} %e{k} to {t}"));
    }
    l(format!("%y0 = fmul {t} {p}, %sk1"));
    l(format!("%y1 = fmul {t} %y0, %sk2"));
    l(format!(
        "%y2 = select i1 %ch, {t} 0x7FF0000000000000, {t} %y1"
    ));
    l(format!("%cl = fcmp olt {t} %x, {lo}"));
    l(format!("%y3 = select i1 %cl, {t} 0.0, {t} %y2"));
    l(format!("%cn = fcmp uno {t} %x, 0.0"));
    l(format!("%y = select i1 %cn, {t} %x, {t} %y3"));
    l("ret {t} %y".replace("{t}", t));
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The realization rule: the oracle off the default face and under the
    /// OFF arm; on it the hardware form where the part's `ex2` error meets the
    /// bound (`f32` only), else the contracted steps where the face contracts.
    #[test]
    fn realization_follows_face_arm_and_measured_error() {
        use ExpRealize::*;
        assert_eq!(choose(false, true, true, Some(Some(3))), [Oracle, Oracle]);
        assert_eq!(choose(true, true, false, Some(Some(3))), [Oracle, Oracle]);
        assert_eq!(
            choose(true, true, true, Some(Some(3))),
            [Hardware, Contracted]
        );
        assert_eq!(
            choose(true, true, true, Some(Some(9))),
            [Contracted, Contracted]
        );
        assert_eq!(
            choose(true, true, true, Some(None)),
            [Contracted, Contracted]
        );
        assert_eq!(choose(true, true, true, None), [Contracted, Contracted]);
        assert_eq!(choose(true, false, true, None), [Oracle, Oracle]);
    }

    /// Definitions appear only for a width the text calls; the oracle's text
    /// carries no fast-math flag, the hardware form the `ex2` intrinsic.
    #[test]
    fn defs_follow_the_calls() {
        let call32 = "  %v = call float @mapal.exp.f32(float %a)\n";
        let d = defs(call32, [ExpRealize::Oracle, ExpRealize::Oracle]);
        assert!(d.contains("define internal float @mapal.exp.f32(float %x)"));
        assert!(!d.contains("@mapal.exp.f64"));
        assert!(!d.contains("contract") && !d.contains("ex2"));
        let h = defs(call32, [ExpRealize::Hardware, ExpRealize::Oracle]);
        assert!(h.contains("@llvm.nvvm.ex2.approx.ftz.f(float %a)"));
        assert!(h.contains("declare float @llvm.nvvm.ex2.approx.ftz.f(float)"));
        let c = defs(call32, [ExpRealize::Contracted, ExpRealize::Oracle]);
        assert!(c.contains("fmul contract float") && !c.contains("reassoc"));
        assert_eq!(defs("no calls", [ExpRealize::Oracle; 2]), "");
    }
}
