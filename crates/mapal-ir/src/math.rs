//! **Math builtins and their realizations** (ADR-0046): `exp`, `sqrt`, `sin`,
//! `cos` — one framework for every pure math pipeline builtin.
//!
//! Each builtin has
//!
//! - a **contract** ([`contract`], [`special`], [`within`]): the exact,
//!   contract and fixed faces are the reference's bits; the default face any
//!   realization within `rel·|f(x)| + abs` with the same special values;
//! - a **reference definition** written once, generic over [`Arith`], from
//!   plain operations (IEEE `+ − ×` to nearest, compares, selects, integer
//!   ops, casts — straight-line, no table, no loop, no FMA): [`Native`] runs it
//!   (the interpreter, the constant folder), [`Trace`] records it as a
//!   [`Def`] a backend prints (LLVM text, C) — so a target with no float
//!   function unit at all (an ASIC) still has every builtin, bit for bit;
//! - **target realizations as data** ([`Target`]: [`Unit`]s a target computes a
//!   function with, [`TargetDef`]s it supplies), composed into forms whose
//!   error follows from the unit's ([`candidates`]), admitted per face
//!   ([`admits`]) and chosen by cost ([`choose`]) — never by a target's name.

use crate::exp::ExpForm;

mod contract;
mod defs;
mod native;
mod target;
mod trace;
mod trig;

pub use self::contract::*;
pub use self::defs::*;
pub use self::native::*;
pub use self::target::*;
pub use self::trace::*;
pub use self::trig::*;

// ---------------------------------------------------------------------------
// The builtins.
// ---------------------------------------------------------------------------

/// A pure math pipeline builtin (`x -> exp`, `x -> sqrt`, `x -> sin`,
/// `x -> cos`): `Float{b} → Float{b}`, total and trap-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MathFn {
    Exp,
    Sqrt,
    Sin,
    Cos,
}

impl MathFn {
    pub const ALL: [MathFn; 4] = [MathFn::Exp, MathFn::Sqrt, MathFn::Sin, MathFn::Cos];

    /// The builtin's name in the surface language.
    pub const fn name(self) -> &'static str {
        match self {
            MathFn::Exp => "exp",
            MathFn::Sqrt => "sqrt",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
        }
    }

    /// The builtin a pipeline stage's name resolves to.
    pub fn from_name(s: &str) -> Option<MathFn> {
        Self::ALL.into_iter().find(|f| f.name() == s)
    }

    /// The IR edge label (`mermaid`).
    pub const fn label(self) -> &'static str {
        match self {
            MathFn::Exp => "Exp",
            MathFn::Sqrt => "Sqrt",
            MathFn::Sin => "Sin",
            MathFn::Cos => "Cos",
        }
    }
}

// ---------------------------------------------------------------------------
// The operations a definition is written in.
// ---------------------------------------------------------------------------

/// An integer width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum W {
    I32,
    I64,
}

impl W {
    pub const fn bits(self) -> u32 {
        match self {
            W::I32 => 32,
            W::I64 => 64,
        }
    }
}

/// Binary float operations, rounded to nearest even.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F2 {
    Add,
    Sub,
    Mul,
}

/// Ordered float predicates and `unordered`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FPred {
    Ogt,
    Olt,
    Oeq,
    Uno,
}

/// Binary integer operations (wrapping; shift amounts below the width;
/// `MulHi` the unsigned product's high word).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum I2 {
    Add,
    Sub,
    Mul,
    MulHi,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

/// Integer predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IPred {
    Eq,
    Ne,
    Ult,
    Ugt,
    Slt,
    Sgt,
}

/// Boolean operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum B2 {
    And,
    Or,
    Xor,
}

/// Conversions. `Bits` / `FromBits`: the float of the definition's width and
/// the integer of the same width; `FToSi` truncates (in range by
/// construction), `SiToF` / `UiToF` round to nearest even.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cvt {
    Bits,
    FromBits,
    FToSi(W),
    SiToF(W),
    UiToF(W),
    ZExt,
    Trunc,
    BoolToI(W),
}

/// The type of a definition's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VTy {
    /// The definition's float width.
    F,
    I32,
    I64,
    B,
}

/// The arithmetic a definition is written against: [`Native`] computes,
/// [`Trace`] records. Every method is one operation.
pub trait Arith {
    type V: Copy;
    /// The definition's float width (32 or 64).
    fn fbits(&self) -> u32;
    /// Mark the float steps that follow as contractible (the default face's
    /// contracted reference fuses them where the target has an FMA).
    fn contract(&mut self, on: bool);
    /// A name the printed definition gives `v` (readability only).
    fn name(&mut self, v: Self::V, n: &str) -> Self::V;
    /// The name `{prefix}{i}` (formatted only by a trace).
    fn name_ix(&mut self, v: Self::V, prefix: &str, i: usize) -> Self::V;
    /// A float constant of the width, from its bit pattern.
    fn fc(&mut self, bits: u64) -> Self::V;
    fn ic(&mut self, w: W, v: u64) -> Self::V;
    fn f2(&mut self, op: F2, a: Self::V, b: Self::V) -> Self::V;
    fn fneg(&mut self, a: Self::V) -> Self::V;
    /// The fused `a·b + c` (realizations only: no reference uses it).
    fn fma(&mut self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    fn fcmp(&mut self, p: FPred, a: Self::V, b: Self::V) -> Self::V;
    fn i2(&mut self, w: W, op: I2, a: Self::V, b: Self::V) -> Self::V;
    fn clz(&mut self, w: W, a: Self::V) -> Self::V;
    fn icmp(&mut self, w: W, p: IPred, a: Self::V, b: Self::V) -> Self::V;
    fn b2(&mut self, op: B2, a: Self::V, b: Self::V) -> Self::V;
    fn sel(&mut self, c: Self::V, a: Self::V, b: Self::V) -> Self::V;
    fn cvt(&mut self, c: Cvt, a: Self::V) -> Self::V;
    /// A target's function unit applied to `a` (realizations only).
    fn unit(&mut self, u: &Unit, a: Self::V) -> Self::V;

    fn fadd(&mut self, a: Self::V, b: Self::V) -> Self::V {
        self.f2(F2::Add, a, b)
    }
    fn fsub(&mut self, a: Self::V, b: Self::V) -> Self::V {
        self.f2(F2::Sub, a, b)
    }
    fn fmul(&mut self, a: Self::V, b: Self::V) -> Self::V {
        self.f2(F2::Mul, a, b)
    }
}

// ---------------------------------------------------------------------------
// precision_plan's entry.
// ---------------------------------------------------------------------------

/// `precision_plan.math(m)` (ADR-0046 D9): the builtin, its width and its
/// contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MathForm {
    pub f: MathFn,
    pub bits: u32,
    pub contract: Contract,
}

impl MathForm {
    pub fn of(f: MathFn, bits: u32) -> Option<MathForm> {
        Some(MathForm {
            f,
            bits,
            contract: contract(f, bits)?,
        })
    }

    /// The relative condition number `|x·f′(x) / f(x)|` at `x`: how a
    /// relative error of the input grows through the builtin.
    pub fn cond(&self, x: f64) -> f64 {
        match self.f {
            MathFn::Exp => x.abs(),
            MathFn::Sqrt => 0.5,
            MathFn::Sin => (x / x.tan()).abs(),
            MathFn::Cos => (x * x.tan()).abs(),
        }
    }

    /// A default-face bound on the output's relative error at `x` given the
    /// input's relative error `rel_in`: the contract's plus the input's
    /// through the condition number (ADR-0042 D4's composition rule).
    pub fn compose(&self, x: f64, rel_in: f64) -> f64 {
        self.contract.rel + self.cond(x) * rel_in
    }
}

#[cfg(test)]
mod tests;
