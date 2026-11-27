//! Math units: the target's declared `exp`/`sqrt`/`sin`/`cos` realizations and the host's.

use super::*;

/// `ex2.approx.ftz.f32`'s error measured on the RTX 5080 (sm_120) over
/// every f32 in `[−½, ½]` against the f64 `exp2`, worst relative error
/// 2^−22.709 = 2.45 units of `2⁻²⁴`, rounded up (`benches/local/exp_calib.cu`;
/// `exp.md`). The unit is the same SFU instruction on every profiled part;
/// a part measured otherwise states its own.
pub const EX2_UNITS: u32 = 3;

/// ADR-0042 D5 as a unit: `ex2.approx.ftz.f32` on `[−½, ½]`, its error
/// [`EX2_UNITS`] units of `2⁻²⁴` (flushes subnormals: the `exp` form scales
/// after it, so it never sees one).
pub const EX2_UNIT: MathUnit = MathUnit {
    op: UnitOp::Ex2,
    bits: 32,
    sym: "llvm.nvvm.ex2.approx.ftz.f",
    err: MathErr::Bound {
        rel: EX2_UNITS as f64 / 16_777_216.0,
        abs: 0.0,
    },
    lo: -0.5,
    hi: 0.5,
    cost: 1,
    specials: false,
    evidence: "benches/local/exp_calib.cu: RTX 5080 ex2.approx.ftz.f32 over [-1/2, 1/2] 2^-22.709 (exp.md)",
};

/// `sqrt.rn.f32` / `.f64`: IEEE 754 `squareRoot`, correctly rounded — the
/// reference's bits by definition (`Exact`), on every NVIDIA SM.
pub const SQRT_RN: [MathUnit; 2] = [
    MathUnit {
        op: UnitOp::Sqrt,
        bits: 32,
        sym: "llvm.nvvm.sqrt.rn.f",
        err: MathErr::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 8,
        specials: true,
        evidence: "PTX ISA sqrt.rn: IEEE 754 squareRoot, round to nearest even (RTX 5080: 0 mismatches over every positive f32, 19.7 cycles a warp op)",
    },
    MathUnit {
        op: UnitOp::Sqrt,
        bits: 64,
        sym: "llvm.nvvm.sqrt.rn.d",
        err: MathErr::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 16,
        specials: true,
        evidence: "PTX ISA sqrt.rn: IEEE 754 squareRoot, round to nearest even",
    },
];

/// The math units every NVIDIA SM has: [`EX2_UNIT`] and [`SQRT_RN`].
pub const MATH_SM: [MathUnit; 3] = [EX2_UNIT, SQRT_RN[0], SQRT_RN[1]];
pub const MATH_SM_TARGET: MathTarget = MathTarget {
    units: &MATH_SM,
    defs: &[],
};

/// [`MATH_SM`] with the approximate units the RTX 5080's calibration
/// measured (`benches/gpu/calib.cu math`, `local-rtx-5080/math.md`).
pub const MATH_SM_RTX_5080: [MathUnit; 6] = [
    EX2_UNIT,
    SQRT_RN[0],
    SQRT_RN[1],
    SQRT_APPROX_RTX_5080,
    SIN_APPROX_RTX_5080,
    COS_APPROX_RTX_5080,
];
pub const MATH_RTX_5080: MathTarget = MathTarget {
    units: &MATH_SM_RTX_5080,
    defs: &[],
};

/// ADR-0046 D8: **the approximate math units a calibration measured**, as
/// the calibration line spells them (errors × 10¹², `calib.cu math`; 0 = not
/// measured): the hand profile and the native profile build the same units
/// from them ([`Gpu::with_cost`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MathUnits {
    pub sqrt_rel_e12: u64,
    pub sin_rel_e12: u64,
    pub sin_abs_e12: u64,
    pub cos_rel_e12: u64,
    /// 1: the units return the contract's special values (`calib.cu math`'s
    /// `specials=ok`).
    pub specials: u64,
}

const fn e12(v: u64) -> f64 {
    v as f64 / 1e12
}

const MATH_EVIDENCE: &str =
    "the calibration line's calib.cu math measurement (every f32 of the unit's domain)";

/// `sqrt.approx.f32` (MUFU, subnormal operands kept — the `.ftz` form
/// flushes them: relative error 1) with its measured relative error over
/// every positive f32.
pub const fn sqrt_approx_unit(rel_e12: u64, specials: bool) -> MathUnit {
    MathUnit {
        op: UnitOp::Sqrt,
        bits: 32,
        sym: "llvm.nvvm.sqrt.approx.f",
        err: MathErr::Bound {
            rel: e12(rel_e12),
            abs: 0.0,
        },
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 1,
        specials,
        evidence: MATH_EVIDENCE,
    }
}

/// `sin.approx.f32` / `cos.approx.f32` (MUFU.SIN / MUFU.COS) on the
/// Payne–Hanek remainder's range `[−0.7854, 0.7854]` (a little past `π/4`).
pub const fn trig_approx_unit(cos: bool, rel_e12: u64, abs_e12: u64, specials: bool) -> MathUnit {
    MathUnit {
        op: if cos { UnitOp::Cos } else { UnitOp::Sin },
        bits: 32,
        sym: if cos {
            "llvm.nvvm.cos.approx.f"
        } else {
            "llvm.nvvm.sin.approx.f"
        },
        err: MathErr::Bound {
            rel: e12(rel_e12),
            abs: e12(abs_e12),
        },
        lo: -0.7854,
        hi: 0.7854,
        cost: 1,
        specials,
        evidence: MATH_EVIDENCE,
    }
}

impl MathUnits {
    /// [`MATH_SM`] and the measured units.
    pub fn target(&self) -> MathTarget {
        let sp = self.specials == 1;
        let mut v: Vec<MathUnit> = MATH_SM.to_vec();
        if self.sqrt_rel_e12 > 0 {
            v.push(sqrt_approx_unit(self.sqrt_rel_e12, sp));
        }
        if self.sin_rel_e12 > 0 || self.sin_abs_e12 > 0 {
            v.push(trig_approx_unit(
                false,
                self.sin_rel_e12,
                self.sin_abs_e12,
                sp,
            ));
        }
        if self.cos_rel_e12 > 0 {
            v.push(trig_approx_unit(true, self.cos_rel_e12, 0, sp));
        }
        // A profile is built once a process (`native_gpu`'s `OnceLock`).
        MathTarget {
            units: Box::leak(v.into_boxed_slice()),
            defs: &[],
        }
    }

    /// The measured units of `t`, as the calibration line spells them.
    pub fn of(t: &MathTarget) -> MathUnits {
        let mut m = MathUnits::default();
        let x = |v: f64| (v * 1e12).round() as u64;
        for u in t.units {
            let (rel, abs) = match u.err {
                MathErr::Bound { rel, abs } => (rel, abs),
                MathErr::Exact => continue,
            };
            match u.op {
                UnitOp::Sqrt => m.sqrt_rel_e12 = x(rel),
                UnitOp::Sin => (m.sin_rel_e12, m.sin_abs_e12) = (x(rel), x(abs)),
                UnitOp::Cos => m.cos_rel_e12 = x(rel),
                _ => continue,
            }
            m.specials = u64::from(u.specials);
        }
        m
    }

    pub(super) fn get(&self, name: &str) -> u64 {
        match name {
            "sqrt_approx_rel_e12" => self.sqrt_rel_e12,
            "sin_approx_rel_e12" => self.sin_rel_e12,
            "sin_approx_abs_e12" => self.sin_abs_e12,
            "cos_approx_rel_e12" => self.cos_rel_e12,
            "math_specials" => self.specials,
            _ => 0,
        }
    }
}

/// The RTX 5080's measured units (`calib.cu math`, 2026-11-25,
/// `local-rtx-5080/math.md`; its calibration line spells the same):
/// `sqrt.approx.f32` relative 1.0001e-7 = 2^−23.253 over every positive f32
/// (8.13 cycles a warp op against FFMA's 1.06 — one issue slot inside a
/// form); `sin.approx.f32` absolute 3.4745e-7 = 2^−21.457, relative 1 (not
/// relative near 0: no `sin` form takes it — the bound is relative);
/// `cos.approx.f32` relative 3.1899e-7 = 2^−21.580 (the `cos r` part of a trig
/// form: `2^−21.58 + 2⁻²³ = 2^−21.12 ≤ 2⁻²¹`); every one returns the
/// contract's special values.
pub const SQRT_APPROX_RTX_5080: MathUnit = sqrt_approx_unit(100_010, true);
pub const SIN_APPROX_RTX_5080: MathUnit = trig_approx_unit(false, 1_000_000_000_000, 347_450, true);
pub const COS_APPROX_RTX_5080: MathUnit = trig_approx_unit(true, 318_990, 0, true);

/// ADR-0046 D7: the host's math units — a correctly rounded `sqrt` (IEEE 754
/// `squareRoot`: LLVM's `llvm.sqrt` without fast-math flags is that operation
/// on every host target — x86 `sqrtss` / `sqrtsd`, AArch64 `fsqrt`, else the
/// libm call). No transcendental unit: `exp`, `sin`, `cos` are the
/// references (contracted on the default face).
pub const HOST_MATH_UNITS: [MathUnit; 2] = [
    MathUnit {
        op: UnitOp::Sqrt,
        bits: 32,
        sym: "llvm.sqrt.f32",
        err: MathErr::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 4,
        specials: true,
        evidence: "IEEE 754 squareRoot: llvm.sqrt without fast-math flags",
    },
    MathUnit {
        op: UnitOp::Sqrt,
        bits: 64,
        sym: "llvm.sqrt.f64",
        err: MathErr::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        cost: 4,
        specials: true,
        evidence: "IEEE 754 squareRoot: llvm.sqrt without fast-math flags",
    },
];
pub const HOST_MATH: MathTarget = MathTarget {
    units: &HOST_MATH_UNITS,
    defs: &[],
};
