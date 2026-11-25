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
// Native evaluation.
// ---------------------------------------------------------------------------

/// The definitions computed: every value a `u64` payload (a float's bits, an
/// integer in its low `w` bits, a bool 0 / 1). Monomorphized and inlined, it
/// is the reference at native speed — the interpreter's and the constant
/// folder's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Native<const B: u32>;

const M32: u64 = 0xffff_ffff;

#[inline(always)]
fn mask(w: W, v: u64) -> u64 {
    match w {
        W::I32 => v & M32,
        W::I64 => v,
    }
}

impl<const B: u32> Native<B> {
    #[inline(always)]
    fn fop(
        a: u64,
        b: u64,
        f32op: impl Fn(f32, f32) -> f32,
        f64op: impl Fn(f64, f64) -> f64,
    ) -> u64 {
        if B == 32 {
            f32op(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits() as u64
        } else {
            f64op(f64::from_bits(a), f64::from_bits(b)).to_bits()
        }
    }

    #[inline(always)]
    fn fval(a: u64) -> f64 {
        if B == 32 {
            f32::from_bits(a as u32) as f64
        } else {
            f64::from_bits(a)
        }
    }

    #[inline(always)]
    fn of_f64(v: f64) -> u64 {
        if B == 32 {
            (v as f32).to_bits() as u64
        } else {
            v.to_bits()
        }
    }
}

impl<const B: u32> Arith for Native<B> {
    type V = u64;

    #[inline(always)]
    fn fbits(&self) -> u32 {
        B
    }
    #[inline(always)]
    fn contract(&mut self, _on: bool) {}
    #[inline(always)]
    fn name(&mut self, v: u64, _n: &str) -> u64 {
        v
    }
    #[inline(always)]
    fn name_ix(&mut self, v: u64, _p: &str, _i: usize) -> u64 {
        v
    }
    #[inline(always)]
    fn fc(&mut self, bits: u64) -> u64 {
        if B == 32 { bits & M32 } else { bits }
    }
    #[inline(always)]
    fn ic(&mut self, w: W, v: u64) -> u64 {
        mask(w, v)
    }
    #[inline(always)]
    fn f2(&mut self, op: F2, a: u64, b: u64) -> u64 {
        match op {
            F2::Add => Self::fop(a, b, |x, y| x + y, |x, y| x + y),
            F2::Sub => Self::fop(a, b, |x, y| x - y, |x, y| x - y),
            F2::Mul => Self::fop(a, b, |x, y| x * y, |x, y| x * y),
        }
    }
    #[inline(always)]
    fn fneg(&mut self, a: u64) -> u64 {
        if B == 32 {
            a ^ 0x8000_0000
        } else {
            a ^ (1 << 63)
        }
    }
    #[inline(always)]
    fn fma(&mut self, a: u64, b: u64, c: u64) -> u64 {
        if B == 32 {
            let f = |v: u64| f32::from_bits(v as u32);
            f(a).mul_add(f(b), f(c)).to_bits() as u64
        } else {
            let f = f64::from_bits;
            f(a).mul_add(f(b), f(c)).to_bits()
        }
    }
    #[inline(always)]
    fn fcmp(&mut self, p: FPred, a: u64, b: u64) -> u64 {
        let (x, y) = (Self::fval(a), Self::fval(b));
        u64::from(match p {
            FPred::Ogt => x > y,
            FPred::Olt => x < y,
            FPred::Oeq => x == y,
            FPred::Uno => x.is_nan() || y.is_nan(),
        })
    }
    #[inline(always)]
    fn i2(&mut self, w: W, op: I2, a: u64, b: u64) -> u64 {
        let n = w.bits();
        let r = match op {
            I2::Add => a.wrapping_add(b),
            I2::Sub => a.wrapping_sub(b),
            I2::Mul => a.wrapping_mul(b),
            I2::MulHi => match w {
                W::I32 => ((a & M32) * (b & M32)) >> 32,
                W::I64 => ((u128::from(a) * u128::from(b)) >> 64) as u64,
            },
            I2::And => a & b,
            I2::Or => a | b,
            I2::Xor => a ^ b,
            I2::Shl => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                a << (b & 63)
            }
            I2::LShr => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                mask(w, a) >> (b & 63)
            }
            I2::AShr => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                match w {
                    W::I32 => ((a as u32 as i32) >> (b & 31)) as u32 as u64,
                    W::I64 => ((a as i64) >> (b & 63)) as u64,
                }
            }
        };
        mask(w, r)
    }
    #[inline(always)]
    fn clz(&mut self, w: W, a: u64) -> u64 {
        match w {
            W::I32 => u64::from((a as u32).leading_zeros()),
            W::I64 => u64::from(a.leading_zeros()),
        }
    }
    #[inline(always)]
    fn icmp(&mut self, w: W, p: IPred, a: u64, b: u64) -> u64 {
        let (a, b) = (mask(w, a), mask(w, b));
        let s = |v: u64| match w {
            W::I32 => i64::from(v as u32 as i32),
            W::I64 => v as i64,
        };
        u64::from(match p {
            IPred::Eq => a == b,
            IPred::Ne => a != b,
            IPred::Ult => a < b,
            IPred::Ugt => a > b,
            IPred::Slt => s(a) < s(b),
            IPred::Sgt => s(a) > s(b),
        })
    }
    #[inline(always)]
    fn b2(&mut self, op: B2, a: u64, b: u64) -> u64 {
        match op {
            B2::And => a & b,
            B2::Or => a | b,
            B2::Xor => a ^ b,
        }
    }
    #[inline(always)]
    fn sel(&mut self, c: u64, a: u64, b: u64) -> u64 {
        if c != 0 { a } else { b }
    }
    #[inline(always)]
    fn cvt(&mut self, c: Cvt, a: u64) -> u64 {
        match c {
            Cvt::Bits | Cvt::FromBits => a,
            Cvt::FToSi(w) => {
                let v = if B == 32 {
                    f32::from_bits(a as u32) as i64
                } else {
                    f64::from_bits(a) as i64
                };
                mask(w, v as u64)
            }
            Cvt::SiToF(w) => {
                let v = match w {
                    W::I32 => i64::from(a as u32 as i32),
                    W::I64 => a as i64,
                };
                if B == 32 {
                    (v as f32).to_bits() as u64
                } else {
                    (v as f64).to_bits()
                }
            }
            Cvt::UiToF(w) => {
                let v = mask(w, a);
                if B == 32 {
                    (v as f32).to_bits() as u64
                } else {
                    (v as f64).to_bits()
                }
            }
            Cvt::ZExt | Cvt::Trunc => a & M32,
            Cvt::BoolToI(_) => a & 1,
        }
    }
    /// An ideal unit (the function correctly rounded from `f64`): the model a
    /// host-side check of a form evaluates; the device's own error is the
    /// unit's declared one.
    fn unit(&mut self, u: &Unit, a: u64) -> u64 {
        let x = Self::fval(a);
        Self::of_f64(match u.op {
            UnitOp::Ex2 => x.exp2(),
            UnitOp::Exp => x.exp(),
            UnitOp::Sqrt => x.sqrt(),
            UnitOp::Sin => x.sin(),
            UnitOp::Cos => x.cos(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tracing: a definition as data.
// ---------------------------------------------------------------------------

/// A value of a traced definition (its step's index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

/// One step's operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// The definition's argument.
    Arg,
    /// A float constant of the width (bits).
    CF(u64),
    CI(W, u64),
    F2(F2),
    FNeg,
    Fma,
    FCmp(FPred),
    I2(W, I2),
    Clz(W),
    ICmp(W, IPred),
    B2(B2),
    Sel,
    Cvt(Cvt),
    /// A function unit: its operation and the symbol the target spells it
    /// with.
    Unit(UnitOp, &'static str),
}

/// One step of a [`Def`].
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub op: Op,
    pub args: [Val; 3],
    pub ty: VTy,
    /// Contractible (the contracted reference's `contract` flag).
    pub contract: bool,
    pub name: Option<String>,
}

/// A traced definition: straight-line steps from the argument (step 0) to
/// `out`. What a backend prints.
#[derive(Clone, Debug, PartialEq)]
pub struct Def {
    pub bits: u32,
    pub steps: Vec<Step>,
    pub out: Val,
}

/// The recorder: an [`Arith`] whose values are steps.
#[derive(Clone, Debug)]
pub struct Trace {
    bits: u32,
    steps: Vec<Step>,
    contract: bool,
}

const NO: Val = Val(u32::MAX);

impl Trace {
    /// A trace of width `bits` and its argument.
    pub fn new(bits: u32) -> (Trace, Val) {
        let mut t = Trace {
            bits,
            steps: Vec::new(),
            contract: false,
        };
        let x = t.push(Op::Arg, [NO; 3], VTy::F);
        t.steps[0].name = Some("x".into());
        (t, x)
    }

    pub fn finish(self, out: Val) -> Def {
        Def {
            bits: self.bits,
            steps: self.steps,
            out,
        }
    }

    fn push(&mut self, op: Op, args: [Val; 3], ty: VTy) -> Val {
        let contract = self.contract && matches!(op, Op::F2(_));
        self.steps.push(Step {
            op,
            args,
            ty,
            contract,
            name: None,
        });
        Val(self.steps.len() as u32 - 1)
    }

    fn ty(&self, v: Val) -> VTy {
        self.steps[v.0 as usize].ty
    }
}

fn wty(w: W) -> VTy {
    match w {
        W::I32 => VTy::I32,
        W::I64 => VTy::I64,
    }
}

impl Arith for Trace {
    type V = Val;

    fn fbits(&self) -> u32 {
        self.bits
    }
    fn contract(&mut self, on: bool) {
        self.contract = on;
    }
    fn name(&mut self, v: Val, n: &str) -> Val {
        let s = &mut self.steps[v.0 as usize];
        if !matches!(s.op, Op::CF(_) | Op::CI(..)) {
            s.name = Some(n.to_string());
        }
        v
    }
    fn name_ix(&mut self, v: Val, prefix: &str, i: usize) -> Val {
        self.name(v, &format!("{prefix}{i}"))
    }
    fn fc(&mut self, bits: u64) -> Val {
        let bits = if self.bits == 32 { bits & M32 } else { bits };
        self.push(Op::CF(bits), [NO; 3], VTy::F)
    }
    fn ic(&mut self, w: W, v: u64) -> Val {
        self.push(Op::CI(w, mask(w, v)), [NO; 3], wty(w))
    }
    fn f2(&mut self, op: F2, a: Val, b: Val) -> Val {
        self.push(Op::F2(op), [a, b, NO], VTy::F)
    }
    fn fneg(&mut self, a: Val) -> Val {
        self.push(Op::FNeg, [a, NO, NO], VTy::F)
    }
    fn fma(&mut self, a: Val, b: Val, c: Val) -> Val {
        self.push(Op::Fma, [a, b, c], VTy::F)
    }
    fn fcmp(&mut self, p: FPred, a: Val, b: Val) -> Val {
        self.push(Op::FCmp(p), [a, b, NO], VTy::B)
    }
    fn i2(&mut self, w: W, op: I2, a: Val, b: Val) -> Val {
        debug_assert!(self.ty(a) == wty(w) && self.ty(b) == wty(w), "{op:?} {w:?}");
        self.push(Op::I2(w, op), [a, b, NO], wty(w))
    }
    fn clz(&mut self, w: W, a: Val) -> Val {
        self.push(Op::Clz(w), [a, NO, NO], wty(w))
    }
    fn icmp(&mut self, w: W, p: IPred, a: Val, b: Val) -> Val {
        self.push(Op::ICmp(w, p), [a, b, NO], VTy::B)
    }
    fn b2(&mut self, op: B2, a: Val, b: Val) -> Val {
        self.push(Op::B2(op), [a, b, NO], VTy::B)
    }
    fn sel(&mut self, c: Val, a: Val, b: Val) -> Val {
        let ty = self.ty(a);
        debug_assert_eq!(ty, self.ty(b));
        self.push(Op::Sel, [c, a, b], ty)
    }
    fn cvt(&mut self, c: Cvt, a: Val) -> Val {
        let iw = if self.bits == 64 { VTy::I64 } else { VTy::I32 };
        let ty = match c {
            Cvt::Bits => iw,
            Cvt::FromBits | Cvt::SiToF(_) | Cvt::UiToF(_) => VTy::F,
            Cvt::FToSi(w) | Cvt::BoolToI(w) => wty(w),
            Cvt::ZExt => VTy::I64,
            Cvt::Trunc => VTy::I32,
        };
        self.push(Op::Cvt(c), [a, NO, NO], ty)
    }
    fn unit(&mut self, u: &Unit, a: Val) -> Val {
        self.push(Op::Unit(u.op, u.sym), [a, NO, NO], VTy::F)
    }
}

impl Def {
    /// The definition evaluated (each step by [`Native`]'s semantics).
    pub fn eval(&self, x: u64) -> u64 {
        if self.bits == 64 {
            self.run(&mut Native::<64>, x)
        } else {
            self.run(&mut Native::<32>, x)
        }
    }

    fn run<const B: u32>(&self, n: &mut Native<B>, x: u64) -> u64 {
        let mut r = vec![0u64; self.steps.len()];
        for (i, s) in self.steps.iter().enumerate() {
            let a = |k: usize| r[s.args[k].0 as usize];
            r[i] = match s.op {
                Op::Arg => n.fc(x),
                Op::CF(b) => n.fc(b),
                Op::CI(w, v) => n.ic(w, v),
                Op::F2(op) => n.f2(op, a(0), a(1)),
                Op::FNeg => n.fneg(a(0)),
                Op::Fma => n.fma(a(0), a(1), a(2)),
                Op::FCmp(p) => n.fcmp(p, a(0), a(1)),
                Op::I2(w, op) => n.i2(w, op, a(0), a(1)),
                Op::Clz(w) => n.clz(w, a(0)),
                Op::ICmp(w, p) => n.icmp(w, p, a(0), a(1)),
                Op::B2(op) => n.b2(op, a(0), a(1)),
                Op::Sel => n.sel(a(0), a(1), a(2)),
                Op::Cvt(c) => n.cvt(c, a(0)),
                Op::Unit(op, sym) => {
                    let u = Unit {
                        op,
                        bits: B,
                        sym,
                        err: Err::Exact,
                        lo: f64::NEG_INFINITY,
                        hi: f64::INFINITY,
                        cost: 1,
                        specials: true,
                        evidence: "",
                    };
                    n.unit(&u, a(0))
                }
            };
        }
        r[self.out.0 as usize]
    }

    /// The operations it issues: constants free, a unit its declared slots
    /// (`unit_cost`), every other step one.
    pub fn ops(&self, unit_cost: impl Fn(UnitOp) -> u32) -> u32 {
        self.steps
            .iter()
            .map(|s| match s.op {
                Op::Arg | Op::CF(_) | Op::CI(..) => 0,
                Op::Unit(u, _) => unit_cost(u),
                _ => 1,
            })
            .sum()
    }

    /// The contractible products (each fuses with its add: one slot fewer).
    pub fn fusable(&self) -> u32 {
        self.steps
            .iter()
            .filter(|s| s.contract && s.op == Op::F2(F2::Mul))
            .count() as u32
    }

    /// Whether any step is `op` (a declaration the printed module needs).
    pub fn uses(&self, f: impl Fn(&Op) -> bool) -> bool {
        self.steps.iter().any(|s| f(&s.op))
    }
}

// ---------------------------------------------------------------------------
// Reference definitions.
// ---------------------------------------------------------------------------

/// `exp`'s reference: ADR-0042's `E_b`, the same operations in the same
/// order (its printed text is ADR-0042's byte for byte).
pub fn exp_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    exp_body(a, x, None)
}

/// `E_b`, its middle either the reduction and the Taylor polynomial or (the
/// hardware form, ADR-0042 D5) the reduction as two FMAs and `ex2` of
/// `r·log2e`.
fn exp_body<A: Arith>(a: &mut A, x: A::V, ex2: Option<&Unit>) -> A::V {
    let form = ExpForm::of_bits(a.fbits()).expect("exp of an f32 or f64");
    let w = if form.bits == 64 { W::I64 } else { W::I32 };
    let (hi, lo, zero) = (a.fc(form.hi), a.fc(form.lo), a.fc(0));
    let ch = a.fcmp(FPred::Ogt, x, hi);
    let ch = a.name(ch, "ch");
    let x0 = a.sel(ch, hi, x);
    let x0 = a.name(x0, "x0");
    let cl0 = a.fcmp(FPred::Olt, x0, lo);
    let cl0 = a.name(cl0, "cl0");
    let x1 = a.sel(cl0, lo, x0);
    let x1 = a.name(x1, "x1");
    let cn0 = a.fcmp(FPred::Uno, x1, zero);
    let cn0 = a.name(cn0, "cn0");
    let xc = a.sel(cn0, zero, x1);
    let xc = a.name(xc, "xc");
    let (log2e, sh) = (a.fc(form.log2e), a.fc(form.shifter));
    let (lhi, llo) = (a.fc(form.ln2_hi), a.fc(form.ln2_lo));
    let (kf, p) = match ex2 {
        Some(u) => {
            let t = a.fma(xc, log2e, sh);
            a.name(t, "t");
            let kf = a.fsub(t, sh);
            a.name(kf, "kf");
            let nk = a.fneg(kf);
            a.name(nk, "nk");
            let r0 = a.fma(nk, lhi, xc);
            a.name(r0, "r0");
            let r = a.fma(nk, llo, r0);
            a.name(r, "r");
            let av = a.fmul(r, log2e);
            a.name(av, "a");
            let p0 = a.unit(u, av);
            a.name(p0, "p0");
            (kf, p0)
        }
        None => {
            let t0 = a.fmul(xc, log2e);
            a.name(t0, "t0");
            let t = a.fadd(t0, sh);
            a.name(t, "t");
            let kf = a.fsub(t, sh);
            a.name(kf, "kf");
            a.contract(true);
            let h = a.fmul(kf, lhi);
            a.name(h, "h");
            let r0 = a.fsub(xc, h);
            a.name(r0, "r0");
            let lw = a.fmul(kf, llo);
            a.name(lw, "lw");
            let r = a.fsub(r0, lw);
            a.name(r, "r");
            let cs = form.coefs();
            let n = form.degree as usize;
            let mut p = a.fc(cs[n]);
            for i in (0..n).rev() {
                let m = a.fmul(p, r);
                a.name_ix(m, "m", i);
                let c = a.fc(cs[i]);
                p = a.fadd(m, c);
                a.name_ix(p, "p", i);
            }
            a.contract(false);
            (kf, p)
        }
    };
    let k = a.cvt(Cvt::FToSi(w), kf);
    a.name(k, "k");
    let one = a.ic(w, 1);
    let k1 = a.i2(w, I2::AShr, k, one);
    a.name(k1, "k1");
    let k2 = a.i2(w, I2::Sub, k, k1);
    a.name(k2, "k2");
    let (bias, mant) = (a.ic(w, form.bias as u64), a.ic(w, u64::from(form.mant)));
    let mut s = [k1, k2];
    for (i, k) in [k1, k2].into_iter().enumerate() {
        let b = a.i2(w, I2::Add, k, bias);
        a.name_ix(b, "bk", i + 1);
        let e = a.i2(w, I2::Shl, b, mant);
        a.name_ix(e, "ek", i + 1);
        s[i] = a.cvt(Cvt::FromBits, e);
        a.name_ix(s[i], "sk", i + 1);
    }
    let y0 = a.fmul(p, s[0]);
    a.name(y0, "y0");
    let y1 = a.fmul(y0, s[1]);
    a.name(y1, "y1");
    let inf = a.fc(if form.bits == 64 {
        f64::INFINITY.to_bits()
    } else {
        u64::from(f32::INFINITY.to_bits())
    });
    let y2 = a.sel(ch, inf, y1);
    a.name(y2, "y2");
    let cl = a.fcmp(FPred::Olt, x, lo);
    a.name(cl, "cl");
    let y3 = a.sel(cl, zero, y2);
    a.name(y3, "y3");
    let cn = a.fcmp(FPred::Uno, x, zero);
    a.name(cn, "cn");
    let y = a.sel(cn, x, y3);
    a.name(y, "y")
}

/// The float of the width's canonical quiet NaN.
fn nan_bits(bits: u32) -> u64 {
    if bits == 64 {
        0x7ff8_0000_0000_0000
    } else {
        0x7fc0_0000
    }
}

fn inf_bits(bits: u32) -> u64 {
    if bits == 64 {
        f64::INFINITY.to_bits()
    } else {
        u64::from(f32::INFINITY.to_bits())
    }
}

/// `sqrt`'s plain-operation definition: the correctly rounded `√x` (IEEE
/// `squareRoot`) by the restoring digit recurrence on the significand. The
/// significand `m` (subnormals normalized by `clz`) is shifted to `m2` with an
/// even exponent and `m2·2^(2Z)` square-rooted two bits a step (a 64-bit
/// remainder), giving one bit past the significand; that bit rounds (a
/// square root is never a tie).
pub fn sqrt_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    let bits = a.fbits();
    let wide = bits == 64;
    let w = if wide { W::I64 } else { W::I32 };
    // f32: m2 ∈ [2^24, 2^26), Z = 12, 13 + 12 steps; f64: m2 ∈ [2^52, 2^54),
    // Z = 27, 27 + 27 steps.
    let (mbits, ebits, bias, z, pairs) = if wide {
        (52u64, 11u64, 1023i64, 27u64, 27u64)
    } else {
        (23, 8, 127, 12, 13)
    };
    let ix = a.cvt(Cvt::Bits, x);
    let c_mb = a.ic(w, mbits);
    let expf = a.i2(w, I2::LShr, ix, c_mb);
    let c_emask = a.ic(w, (1 << ebits) - 1);
    let expf = a.i2(w, I2::And, expf, c_emask);
    let c_mmask = a.ic(w, (1 << mbits) - 1);
    let mant = a.i2(w, I2::And, ix, c_mmask);
    // Subnormal: normalize the significand by its leading zeros.
    let zero_i = a.ic(w, 0);
    let sub = a.icmp(w, IPred::Eq, expf, zero_i);
    let lz = a.clz(w, mant);
    let c_off = a.ic(w, u64::from(w.bits()) - mbits - 1);
    let nz = a.i2(w, I2::Sub, lz, c_off);
    let c_max = a.ic(w, mbits);
    let big = a.icmp(w, IPred::Ugt, nz, c_max);
    let nz = a.sel(big, c_max, nz);
    let msub = a.i2(w, I2::Shl, mant, nz);
    let c_hid = a.ic(w, 1 << mbits);
    let mnorm = a.i2(w, I2::Or, mant, c_hid);
    let m = a.sel(sub, msub, mnorm);
    let one = a.ic(w, 1);
    let esub = a.i2(w, I2::Sub, one, nz);
    let e = a.sel(sub, esub, expf);
    // x = m·2^E, E = e − bias − mbits; make the exponent even.
    let c_eb = a.ic(w, (bias + mbits as i64) as u64);
    let ee = a.i2(w, I2::Sub, e, c_eb);
    let odd = a.i2(w, I2::And, ee, one);
    let sh = if wide {
        odd
    } else {
        let two = a.ic(w, 2);
        a.i2(w, I2::Sub, two, odd)
    };
    let m2 = a.i2(w, I2::Shl, m, sh);
    let e2 = a.i2(w, I2::Sub, ee, sh);
    // The digit recurrence, in 64 bits.
    let m2 = if wide { m2 } else { a.cvt(Cvt::ZExt, m2) };
    let w64 = W::I64;
    let mut rem = a.ic(w64, 0);
    let mut res = a.ic(w64, 0);
    let (one64, two64, three64) = (a.ic(w64, 1), a.ic(w64, 2), a.ic(w64, 3));
    for i in 0..pairs + z {
        let r4 = a.i2(w64, I2::Shl, rem, two64);
        rem = if i < pairs {
            let s = a.ic(w64, 2 * (pairs - 1 - i));
            let pr = a.i2(w64, I2::LShr, m2, s);
            let pr = a.i2(w64, I2::And, pr, three64);
            a.i2(w64, I2::Or, r4, pr)
        } else {
            r4
        };
        let t = a.i2(w64, I2::Shl, res, two64);
        let t = a.i2(w64, I2::Or, t, one64);
        let lt = a.icmp(w64, IPred::Ult, rem, t);
        let d = a.i2(w64, I2::Sub, rem, t);
        rem = a.sel(lt, rem, d);
        let r2 = a.i2(w64, I2::Shl, res, one64);
        let ge = a.cvt(Cvt::BoolToI(w64), lt);
        let ge = a.i2(w64, I2::Xor, ge, one64);
        res = a.i2(w64, I2::Or, r2, ge);
    }
    let res = if wide { res } else { a.cvt(Cvt::Trunc, res) };
    let s = a.i2(w, I2::LShr, res, one);
    let round = a.i2(w, I2::And, res, one);
    // √x ≈ (s + round)·2^(E2/2 + 1 − Z): biased exponent E2/2 + 1 − Z +
    // mbits + bias; the significand's carry runs into the exponent.
    let half = a.i2(w, I2::AShr, e2, one);
    let c_be = a.ic(w, (1 - z as i64 + mbits as i64 + bias) as u64);
    let be = a.i2(w, I2::Add, half, c_be);
    let ebits_v = a.i2(w, I2::Shl, be, c_mb);
    let sm = a.i2(w, I2::Sub, s, c_hid);
    let yb = a.i2(w, I2::Add, ebits_v, sm);
    let yb = a.i2(w, I2::Add, yb, round);
    let y = a.cvt(Cvt::FromBits, yb);
    sqrt_specials(a, x, y)
}

/// `sqrt`'s special values around `y` (a positive finite `x`'s root):
/// `±0 → ±0`, `x < 0 → NaN`, `+∞ → +∞`, NaN → NaN.
fn sqrt_specials<A: Arith>(a: &mut A, x: A::V, y: A::V) -> A::V {
    let bits = a.fbits();
    let zero = a.fc(0);
    let inf = a.fc(inf_bits(bits));
    let nan = a.fc(nan_bits(bits));
    let is0 = a.fcmp(FPred::Oeq, x, zero);
    let y = a.sel(is0, x, y);
    let neg = a.fcmp(FPred::Olt, x, zero);
    let y = a.sel(neg, nan, y);
    let isinf = a.fcmp(FPred::Oeq, x, inf);
    let y = a.sel(isinf, x, y);
    let un = a.fcmp(FPred::Uno, x, x);
    a.sel(un, x, y)
}

/// The contract's special values around `y` (a unit's result that need not
/// return them).
fn wrap_specials<A: Arith>(f: MathFn, a: &mut A, x: A::V, y: A::V) -> A::V {
    let bits = a.fbits();
    match f {
        MathFn::Sqrt => sqrt_specials(a, x, y),
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits).expect("exp of an f32 or f64");
            let (hi, lo, zero, inf) = (a.fc(form.hi), a.fc(form.lo), a.fc(0), a.fc(inf_bits(bits)));
            let ch = a.fcmp(FPred::Ogt, x, hi);
            let y = a.sel(ch, inf, y);
            let cl = a.fcmp(FPred::Olt, x, lo);
            let y = a.sel(cl, zero, y);
            let cn = a.fcmp(FPred::Uno, x, x);
            a.sel(cn, x, y)
        }
        MathFn::Sin | MathFn::Cos => {
            let zero = a.fc(0);
            let at0 = if f == MathFn::Sin {
                x
            } else {
                a.fc(if bits == 64 {
                    1f64.to_bits()
                } else {
                    u64::from(1f32.to_bits())
                })
            };
            let is0 = a.fcmp(FPred::Oeq, x, zero);
            let y = a.sel(is0, at0, y);
            let d = a.fsub(x, x);
            let un = a.fcmp(FPred::Uno, d, d);
            a.sel(un, d, y)
        }
    }
}

/// The bits of `2/π` with 32 zero bits in front, as 32-bit words (f32's
/// Payne–Hanek window: the significand's exponent picks four).
pub const TWO_OVER_PI_32: [u64; 8] = [
    0x0000_0000,
    0xa2f9_836e,
    0x4e44_1529,
    0xfc27_57d1,
    0xf534_ddc0,
    0xdb62_9599,
    0x3c43_9041,
    0xfe51_63ab,
];

/// The bits of `2/π` with 64 zero bits in front, as 64-bit words (f64's
/// window).
pub const TWO_OVER_PI_64: [u64; 20] = [
    0x0000_0000_0000_0000,
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
    0xe882_35f5_2ebb_4484,
    0xe99c_7026_b45f_7e41,
    0x3991_d639_8353_39f4,
    0x9c84_5f8b_bdf9_283b,
    0x1ff8_97ff_de05_980f,
    0xef2f_118b_5a0a_6d1f,
    0x6d36_7ecf_27cb_09b7,
    0x4f46_3f66_9e5f_ea2d,
    0x7527_bac7_ebe5_f17b,
    0x3d07_39f7_8a52_92ea,
    0x6bfb_5fb1_1f8d_5d08,
    0x5603_3046_fc7b_6bab,
];

/// `round(π/2 · 2^31)` and `round(π/2 · 2^63)`.
pub const PIO2_32: u64 = 0xc90f_daa2;
pub const PIO2_64: u64 = 0xc90f_daa2_2168_c235;

/// f32 `sin` on `[−π/4, π/4]` (Cephes `sinf`): `r + r·z·P(z)`.
const SIN32: [f32; 3] = [-1.666_665_461_1e-1, 8.332_160_873_6e-3, -1.951_529_589_1e-4];
/// f32 `cos` on `[−π/4, π/4]` (Cephes `cosf`): `1 − z/2 + z²·Q(z)`.
const COS32: [f32; 3] = [
    4.166_664_568_298_827e-2,
    -1.388_731_625_493_765e-3,
    2.443_315_711_809_948e-5,
];
/// f64 `__kernel_sin` (fdlibm) `S1 … S6`.
const SIN64: [u64; 6] = [
    0xBFC5_5555_5555_5549,
    0x3F81_1111_1110_F8A6,
    0xBF2A_01A0_19C1_61D5,
    0x3EC7_1DE3_57B1_FE7D,
    0xBE5A_E5E6_8A2B_9CEB,
    0x3DE5_D93A_5ACF_D57C,
];
/// f64 `__kernel_cos` (fdlibm) `C1 … C6`.
const COS64: [u64; 6] = [
    0x3FA5_5555_5555_554C,
    0xBF56_C16C_16C1_5177,
    0x3EFA_01A0_19CB_1590,
    0xBE92_7E4F_809C_52AD,
    0x3E21_EE9E_BDB4_B1C4,
    0xBDA8_FAE9_BE88_38D4,
];

/// `words[idx + off]` for `idx ∈ 0..n`, by selects over constant words.
fn pick<A: Arith>(a: &mut A, w: W, idx: A::V, words: &[u64], off: usize, n: usize) -> A::V {
    let mut v = a.ic(w, words[off]);
    for i in 1..n {
        let ii = a.ic(w, i as u64);
        let c = a.icmp(w, IPred::Eq, idx, ii);
        let wv = a.ic(w, words[i + off]);
        v = a.sel(c, wv, v);
    }
    v
}

/// `(hi:lo) << o` keeping the high word, `o ∈ [0, n)` (`lo >> 1 >> (n−1−o)`:
/// every shift below the width).
fn funnel<A: Arith>(a: &mut A, w: W, hi: A::V, lo: A::V, o: A::V) -> A::V {
    let one = a.ic(w, 1);
    let nm1 = a.ic(w, u64::from(w.bits()) - 1);
    let h = a.i2(w, I2::Shl, hi, o);
    let l = a.i2(w, I2::LShr, lo, one);
    let s = a.i2(w, I2::Sub, nm1, o);
    let l = a.i2(w, I2::LShr, l, s);
    a.i2(w, I2::Or, h, l)
}

/// The Payne–Hanek reduction: `x = q·π/2 + r`, `|r| ≤ π/4·(1 + ε)`, `q` an
/// `i32` in `0..4`, for every finite `x` (`|x| < π/4`: `r = x`, `q = 0`).
/// Integer arithmetic throughout; `r` carries one rounding.
fn reduce<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    if a.fbits() == 64 {
        reduce64(a, x)
    } else {
        reduce32(a, x)
    }
}

fn reduce32<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    let (w, l) = (W::I32, W::I64);
    let ix = a.cvt(Cvt::Bits, x);
    let c_abs = a.ic(w, 0x7fff_ffff);
    let ax = a.i2(w, I2::And, ix, c_abs);
    // |x| < f32(π/4) (the next f32 above π/4).
    let c_pio4 = a.ic(w, 0x3f49_0fdb);
    let small = a.icmp(w, IPred::Ult, ax, c_pio4);
    let c23 = a.ic(w, 23);
    let expo = a.i2(w, I2::LShr, ax, c23);
    let c_mm = a.ic(w, 0x7f_ffff);
    let m = a.i2(w, I2::And, ax, c_mm);
    let c_hid = a.ic(w, 0x80_0000);
    let m = a.i2(w, I2::Or, m, c_hid);
    // x = m·2^E, E = expo − 150; the window starts at 2/π's bit E − 1, the
    // padded table's bit k0 = expo − 120.
    let c120 = a.ic(w, 120);
    let k0 = a.i2(w, I2::Sub, expo, c120);
    let c5 = a.ic(w, 5);
    let wi = a.i2(w, I2::AShr, k0, c5);
    let c31 = a.ic(w, 31);
    let o = a.i2(w, I2::And, k0, c31);
    let t: [A::V; 4] = std::array::from_fn(|j| pick(a, w, wi, &TWO_OVER_PI_32, j, 5));
    let win: [A::V; 3] = std::array::from_fn(|j| funnel(a, w, t[j], t[j + 1], o));
    // The 96-bit product's bits [32, 96) (mod 2^96: multiples of 4 dropped).
    let m64 = a.cvt(Cvt::ZExt, m);
    let p: [A::V; 3] = std::array::from_fn(|j| {
        let v = a.cvt(Cvt::ZExt, win[j]);
        a.i2(l, I2::Mul, m64, v)
    });
    let c32 = a.ic(l, 32);
    let hi = a.i2(l, I2::Shl, p[0], c32);
    let hi = a.i2(l, I2::Add, hi, p[1]);
    let p2 = a.i2(l, I2::LShr, p[2], c32);
    let hi = a.i2(l, I2::Add, hi, p2);
    finish_reduce(a, x, ix, small, hi, None)
}

fn reduce64<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    let l = W::I64;
    let ix = a.cvt(Cvt::Bits, x);
    let c_abs = a.ic(l, 0x7fff_ffff_ffff_ffff);
    let ax = a.i2(l, I2::And, ix, c_abs);
    // |x| ≤ f64(π/4) (below π/4).
    let c_pio4 = a.ic(l, 0x3fe9_21fb_5444_2d19);
    let small = a.icmp(l, IPred::Ult, ax, c_pio4);
    let c52 = a.ic(l, 52);
    let expo = a.i2(l, I2::LShr, ax, c52);
    let c_mm = a.ic(l, (1 << 52) - 1);
    let m = a.i2(l, I2::And, ax, c_mm);
    let c_hid = a.ic(l, 1 << 52);
    let m = a.i2(l, I2::Or, m, c_hid);
    // E = expo − 1075; bit k0 = expo − 1013 of the padded table.
    let c1013 = a.ic(l, 1013);
    let k0 = a.i2(l, I2::Sub, expo, c1013);
    let c6 = a.ic(l, 6);
    let wi = a.i2(l, I2::AShr, k0, c6);
    let c63 = a.ic(l, 63);
    let o = a.i2(l, I2::And, k0, c63);
    let t: [A::V; 4] = std::array::from_fn(|j| pick(a, l, wi, &TWO_OVER_PI_64, j, 17));
    let win: [A::V; 3] = std::array::from_fn(|j| funnel(a, l, t[j], t[j + 1], o));
    // The 192-bit product's bits [64, 192): top = L0 + H1 + carry, mid = L1 + H2.
    let l0 = a.i2(l, I2::Mul, m, win[0]);
    let h1 = a.i2(l, I2::MulHi, m, win[1]);
    let l1 = a.i2(l, I2::Mul, m, win[1]);
    let h2 = a.i2(l, I2::MulHi, m, win[2]);
    let mid = a.i2(l, I2::Add, l1, h2);
    let carry = a.icmp(l, IPred::Ult, mid, l1);
    let carry = a.cvt(Cvt::BoolToI(l), carry);
    let top = a.i2(l, I2::Add, l0, h1);
    let top = a.i2(l, I2::Add, top, carry);
    finish_reduce(a, x, ix, small, top, Some(mid))
}

/// From the product's top word (and, f64, the next): the quadrant, the
/// signed remainder's magnitude normalized and multiplied by `π/2` in fixed
/// point, one rounding to the width, the signs.
fn finish_reduce<A: Arith>(
    a: &mut A,
    x: A::V,
    ix: A::V,
    small: A::V,
    top: A::V,
    mid: Option<A::V>,
) -> (A::V, A::V) {
    let l = W::I64;
    let w32 = W::I32;
    let wide = mid.is_some();
    let (c2, c62, c63, c64) = (a.ic(l, 2), a.ic(l, 62), a.ic(l, 63), a.ic(l, 64));
    let zero = a.ic(l, 0);
    let qv = a.i2(l, I2::LShr, top, c62);
    let q = a.cvt(Cvt::Trunc, qv);
    // The fraction in units of 2^-64 (f32) / 2^-128 (f64), rounded to the
    // nearest quadrant: a set top bit is the negative remainder.
    let fh = a.i2(l, I2::Shl, top, c2);
    let (fh, fl) = match mid {
        Some(mid) => {
            let mt = a.i2(l, I2::LShr, mid, c62);
            let fh = a.i2(l, I2::Or, fh, mt);
            let fl = a.i2(l, I2::Shl, mid, c2);
            (fh, Some(fl))
        }
        None => (fh, None),
    };
    let neg = a.icmp(l, IPred::Slt, fh, zero);
    let negi = a.cvt(Cvt::BoolToI(w32), neg);
    let q = a.i2(w32, I2::Add, q, negi);
    let (mag_top, n) = match fl {
        None => {
            let nf = a.i2(l, I2::Sub, zero, fh);
            let mag = a.sel(neg, nf, fh);
            let n = a.clz(l, mag);
            let big = a.icmp(l, IPred::Ugt, n, c63);
            let ns = a.sel(big, c63, n);
            let top = a.i2(l, I2::Shl, mag, ns);
            (top, n)
        }
        Some(fl) => {
            // −(fh:fl) as 128 bits.
            let nfl = a.i2(l, I2::Sub, zero, fl);
            let all = a.ic(l, u64::MAX);
            let nfh = a.i2(l, I2::Xor, fh, all);
            let lz = a.icmp(l, IPred::Eq, fl, zero);
            let lzi = a.cvt(Cvt::BoolToI(l), lz);
            let nfh = a.i2(l, I2::Add, nfh, lzi);
            let mh = a.sel(neg, nfh, fh);
            let ml = a.sel(neg, nfl, fl);
            let zh = a.icmp(l, IPred::Eq, mh, zero);
            let nh = a.clz(l, mh);
            let nl = a.clz(l, ml);
            let n_lo = a.i2(l, I2::Add, nl, c64);
            let n = a.sel(zh, n_lo, nh);
            let bh = a.icmp(l, IPred::Ugt, nh, c63);
            let na = a.sel(bh, c63, nh);
            let top_a = funnel(a, l, mh, ml, na);
            let bl = a.icmp(l, IPred::Ugt, nl, c63);
            let nb = a.sel(bl, c63, nl);
            let top_b = a.i2(l, I2::Shl, ml, nb);
            let top = a.sel(zh, top_b, top_a);
            (top, n)
        }
    };
    // r = mag·π/2: the normalized top word times π/2 in fixed point.
    let rf = if wide {
        let pio2 = a.ic(l, PIO2_64);
        let ph = a.i2(l, I2::MulHi, mag_top, pio2);
        a.cvt(Cvt::UiToF(l), ph)
    } else {
        let c32 = a.ic(l, 32);
        let t32 = a.i2(l, I2::LShr, mag_top, c32);
        let pio2 = a.ic(l, PIO2_32);
        let pr = a.i2(l, I2::Mul, t32, pio2);
        a.cvt(Cvt::UiToF(l), pr)
    };
    // The scale 2^(−63−n): f32 bits (64 − n) << 23, f64 (960 − n) << 52.
    let scale = if wide {
        let c960 = a.ic(l, 960);
        let e = a.i2(l, I2::Sub, c960, n);
        let c52 = a.ic(l, 52);
        let e = a.i2(l, I2::Shl, e, c52);
        a.cvt(Cvt::FromBits, e)
    } else {
        let n32 = a.cvt(Cvt::Trunc, n);
        let c = a.ic(w32, 64);
        let e = a.i2(w32, I2::Sub, c, n32);
        let c23 = a.ic(w32, 23);
        let e = a.i2(w32, I2::Shl, e, c23);
        a.cvt(Cvt::FromBits, e)
    };
    let r = a.fmul(rf, scale);
    // The signs: r < 0 iff the remainder or x is negative (not both); a
    // negative x's quadrant turns around.
    let iw = if wide { l } else { w32 };
    let zi = a.ic(iw, 0);
    let xneg = a.icmp(iw, IPred::Slt, ix, zi);
    let rneg = a.b2(B2::Xor, neg, xneg);
    let nr = a.fneg(r);
    let r = a.sel(rneg, nr, r);
    let z32 = a.ic(w32, 0);
    let nq = a.i2(w32, I2::Sub, z32, q);
    let q = a.sel(xneg, nq, q);
    let c3 = a.ic(w32, 3);
    let q = a.i2(w32, I2::And, q, c3);
    let r = a.sel(small, x, r);
    let q = a.sel(small, z32, q);
    (q, r)
}

/// `sin r` on `[−π/4, π/4]` (contractible).
fn sin_poly<A: Arith>(a: &mut A, r: A::V) -> A::V {
    a.contract(true);
    let z = a.fmul(r, r);
    let y = if a.fbits() == 64 {
        let c: [A::V; 6] = std::array::from_fn(|i| a.fc(SIN64[i]));
        let v = a.fmul(z, r);
        let mut h = c[5];
        for i in (1..5).rev() {
            let t = a.fmul(z, h);
            h = a.fadd(c[i], t);
        }
        let t = a.fmul(z, h);
        let s = a.fadd(c[0], t);
        let t = a.fmul(v, s);
        a.fadd(r, t)
    } else {
        let c: [A::V; 3] = std::array::from_fn(|i| a.fc(u64::from(SIN32[i].to_bits())));
        let t = a.fmul(c[2], z);
        let t = a.fadd(t, c[1]);
        let t = a.fmul(t, z);
        let t = a.fadd(t, c[0]);
        let t = a.fmul(t, z);
        let t = a.fmul(t, r);
        a.fadd(t, r)
    };
    a.contract(false);
    y
}

/// `cos r` on `[−π/4, π/4]` (contractible).
fn cos_poly<A: Arith>(a: &mut A, r: A::V) -> A::V {
    a.contract(true);
    let z = a.fmul(r, r);
    let half = a.fc(if a.fbits() == 64 {
        0.5f64.to_bits()
    } else {
        u64::from(0.5f32.to_bits())
    });
    let one = a.fc(if a.fbits() == 64 {
        1f64.to_bits()
    } else {
        u64::from(1f32.to_bits())
    });
    let y = if a.fbits() == 64 {
        let c: [A::V; 6] = std::array::from_fn(|i| a.fc(COS64[i]));
        let mut h = c[5];
        for i in (0..5).rev() {
            let t = a.fmul(z, h);
            h = a.fadd(c[i], t);
        }
        let rr = a.fmul(z, h);
        let hz = a.fmul(half, z);
        let wv = a.fsub(one, hz);
        let t = a.fsub(one, wv);
        let t = a.fsub(t, hz);
        let zr = a.fmul(z, rr);
        let t = a.fadd(t, zr);
        a.fadd(wv, t)
    } else {
        let c: [A::V; 3] = std::array::from_fn(|i| a.fc(u64::from(COS32[i].to_bits())));
        let t = a.fmul(c[2], z);
        let t = a.fadd(t, c[1]);
        let t = a.fmul(t, z);
        let t = a.fadd(t, c[0]);
        let t = a.fmul(t, z);
        let t = a.fmul(t, z);
        let hz = a.fmul(half, z);
        let t = a.fsub(t, hz);
        a.fadd(t, one)
    };
    a.contract(false);
    y
}

/// The quadrant's choice between `sin r` and `cos r` and its sign, `sin`'s
/// signed zero, a non-finite `x`'s NaN.
fn trig_finish<A: Arith>(a: &mut A, x: A::V, q: A::V, s: A::V, c: A::V, cos: bool) -> A::V {
    let w32 = W::I32;
    let (one, two, three) = (a.ic(w32, 1), a.ic(w32, 2), a.ic(w32, 3));
    let q = if cos {
        let q1 = a.i2(w32, I2::Add, q, one);
        a.i2(w32, I2::And, q1, three)
    } else {
        q
    };
    let z = a.ic(w32, 0);
    let b0 = a.i2(w32, I2::And, q, one);
    let odd = a.icmp(w32, IPred::Ne, b0, z);
    let v = a.sel(odd, c, s);
    let b1 = a.i2(w32, I2::And, q, two);
    let flip = a.icmp(w32, IPred::Ne, b1, z);
    let nv = a.fneg(v);
    let v = a.sel(flip, nv, v);
    let fz = a.fc(0);
    let v = if cos {
        v
    } else {
        let is0 = a.fcmp(FPred::Oeq, x, fz);
        a.sel(is0, x, v)
    };
    // x − x: 0 for a finite x, NaN for ±∞ and NaN.
    let d = a.fsub(x, x);
    let un = a.fcmp(FPred::Uno, d, d);
    a.sel(un, d, v)
}

/// `sin`'s reference `S_b`: Payne–Hanek reduction, polynomial, quadrant.
pub fn sin_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    trig_def(a, x, false, (None, None))
}

/// `cos`'s reference `C_b`.
pub fn cos_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    trig_def(a, x, true, (None, None))
}

fn trig_def<A: Arith>(a: &mut A, x: A::V, cos: bool, units: (Option<Unit>, Option<Unit>)) -> A::V {
    let (q, r) = reduce(a, x);
    let s = match &units.0 {
        Some(u) => a.unit(u, r),
        None => sin_poly(a, r),
    };
    let c = match &units.1 {
        Some(u) => a.unit(u, r),
        None => cos_poly(a, r),
    };
    trig_finish(a, x, q, s, c, cos)
}

/// The reference definition of `f`.
#[inline]
pub fn reference<A: Arith>(f: MathFn, a: &mut A, x: A::V) -> A::V {
    match f {
        MathFn::Exp => exp_def(a, x),
        MathFn::Sqrt => sqrt_def(a, x),
        MathFn::Sin => sin_def(a, x),
        MathFn::Cos => cos_def(a, x),
    }
}

/// The reference of `f` on an `f32` (the interpreter's): `sqrt` as Rust's
/// correctly rounded square root (equal to [`sqrt_def`]: the correctly
/// rounded value is unique). `#[inline]`: a caller's optimization level
/// compiles it (the interpreter's).
#[inline]
pub fn eval_f32(f: MathFn, x: f32) -> f32 {
    if f == MathFn::Sqrt {
        return x.sqrt();
    }
    let mut n = Native::<32>;
    f32::from_bits(reference(f, &mut n, u64::from(x.to_bits())) as u32)
}

/// The reference of `f` on an `f64`.
#[inline]
pub fn eval_f64(f: MathFn, x: f64) -> f64 {
    if f == MathFn::Sqrt {
        return x.sqrt();
    }
    let mut n = Native::<64>;
    f64::from_bits(reference(f, &mut n, x.to_bits()))
}

// ---------------------------------------------------------------------------
// Contracts.
// ---------------------------------------------------------------------------

/// A builtin's default-face bound at one width (ADR-0046 D3): `|got − f(x)| ≤
/// rel·|f(x)| + abs` for every `x` without a special value. The exact faces
/// are the reference's bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contract {
    pub f: MathFn,
    pub bits: u32,
    pub rel: f64,
    pub abs: f64,
}

fn tiny(bits: u32) -> f64 {
    if bits == 64 {
        f64::from_bits(1)
    } else {
        f64::from(f32::from_bits(1))
    }
}

/// The contract of `f` at width `bits` (`None`: not a float width).
pub fn contract(f: MathFn, bits: u32) -> Option<Contract> {
    let wide = match bits {
        32 => false,
        64 => true,
        _ => return None,
    };
    let (rel_log2, abs) = match f {
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits)?;
            return Some(Contract {
                f,
                bits,
                rel: form.rel(),
                abs: form.abs(),
            });
        }
        MathFn::Sqrt => (if wide { 51 } else { 22 }, 0.0),
        MathFn::Sin | MathFn::Cos => (if wide { 50 } else { 21 }, tiny(bits)),
    };
    Some(Contract {
        f,
        bits,
        rel: 2f64.powi(-rel_log2),
        abs,
    })
}

/// The value the contract demands exactly at `x` (on every face), or `None`
/// where the bound applies. A NaN demands any NaN.
pub fn special(f: MathFn, bits: u32, x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    match f {
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits)?;
            if x > form.val(form.hi) {
                Some(f64::INFINITY)
            } else if x < form.val(form.lo) {
                Some(0.0)
            } else {
                None
            }
        }
        MathFn::Sqrt => {
            if x == 0.0 || x == f64::INFINITY {
                Some(x)
            } else if x < 0.0 {
                Some(f64::NAN)
            } else {
                None
            }
        }
        MathFn::Sin | MathFn::Cos => {
            if x.is_infinite() {
                Some(f64::NAN)
            } else if x == 0.0 {
                Some(if f == MathFn::Sin { x } else { 1.0 })
            } else {
                None
            }
        }
    }
}

/// `f(x)` in `f64` (libm): the truth a width-32 result is checked against
/// (within `2⁻⁵²` relative of the real value; a width-64 check adds that).
pub fn truth(f: MathFn, x: f64) -> f64 {
    match f {
        MathFn::Exp => x.exp(),
        MathFn::Sqrt => x.sqrt(),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
    }
}

/// `got` meets `f`'s contract at `x` (width `bits`) on the default face:
/// the special value where one is demanded (a zero's sign included), else
/// within the bound of `exact` (`slack`: the reference's own relative error,
/// `0` for an `f64`-checked f32).
pub fn within(f: MathFn, bits: u32, x: f64, got: f64, exact: f64, slack: f64) -> bool {
    let Some(c) = contract(f, bits) else {
        return false;
    };
    if let Some(s) = special(f, bits, x) {
        return if s.is_nan() {
            got.is_nan()
        } else {
            got == s && got.is_sign_negative() == s.is_sign_negative()
        };
    }
    (got - exact).abs() <= (c.rel + slack) * exact.abs() + c.abs
}

/// The contract's test domain at width `bits`: the special values, the
/// domain's edges, subnormals, huge `sin` / `cos` arguments (the f32 / f64
/// worst cancellations against multiples of `π/2`, the largest finite) and
/// seeded samples over the whole range — what a [`TargetDef`]'s declaration
/// is re-measured on.
pub fn test_args(f: MathFn, bits: u32) -> Vec<f64> {
    let wide = bits == 64;
    let of = |b: u64| -> f64 {
        if wide {
            f64::from_bits(b)
        } else {
            f64::from(f32::from_bits(b as u32))
        }
    };
    let mut v: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        0.5,
        2.0,
        of(1),
        of(3),
        if wide {
            f64::MIN_POSITIVE
        } else {
            f64::from(f32::MIN_POSITIVE)
        },
        if wide { f64::MAX } else { f64::from(f32::MAX) },
        if wide {
            -f64::MAX
        } else {
            -f64::from(f32::MAX)
        },
    ];
    match f {
        MathFn::Exp => {
            if let Some(form) = ExpForm::of_bits(bits) {
                for b in [form.hi, form.lo] {
                    v.push(form.val(b));
                    v.push(of(b + 1));
                    v.push(of(b - 1));
                }
            }
        }
        MathFn::Sin | MathFn::Cos => {
            // Near multiples of π/2 (k up to 2^20) and the known hard cases.
            for k in [1u64, 2, 3, 4, 7, 100, 355, 1 << 10, 1 << 20] {
                let t = k as f64 * std::f64::consts::FRAC_PI_2;
                for d in [-2i64, -1, 0, 1, 2] {
                    let b = if wide {
                        (t.to_bits() as i64 + d) as u64
                    } else {
                        ((t as f32).to_bits() as i64 + d) as u64
                    };
                    v.push(of(b));
                    v.push(-of(b));
                }
            }
            // The worst cancellations: f64 6381956970095103·2^797 (Kahan–
            // McDonald), f32 16367173·2^72; and large round numbers.
            if wide {
                v.push(6381956970095103.0 * 2f64.powi(797));
                v.push(1e22);
                v.push(1e300);
            } else {
                v.push(16367173.0 * 2f64.powi(72));
                v.push(f64::from(1e22f32));
                v.push(f64::from(1e30f32));
            }
        }
        MathFn::Sqrt => {
            v.push(4.0);
            v.push(0.25);
            v.push(of(0x7ff));
        }
    }
    // Seeded samples: uniform bit patterns (every exponent) and `[−4, 4]`.
    let mut s: u64 = 0x2545_f491_4f6c_dd1d ^ (f as u64) ^ u64::from(bits);
    for i in 0..3000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = if i % 2 == 0 {
            of(if wide { s } else { s >> 32 })
        } else {
            (s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0
        };
        let x = if wide { x } else { f64::from(x as f32) };
        if !x.is_nan() {
            v.push(x);
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Target realizations as data.
// ---------------------------------------------------------------------------

/// The function a target's unit computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitOp {
    /// `2^x`.
    Ex2,
    /// `eˣ`.
    Exp,
    Sqrt,
    Sin,
    Cos,
}

impl UnitOp {
    /// The unit computing builtin `f` itself.
    pub const fn of(f: MathFn) -> UnitOp {
        match f {
            MathFn::Exp => UnitOp::Exp,
            MathFn::Sqrt => UnitOp::Sqrt,
            MathFn::Sin => UnitOp::Sin,
            MathFn::Cos => UnitOp::Cos,
        }
    }
}

/// A realization's error against the real function: `Exact` (the reference's
/// bits: a correctly rounded operation, or the reference itself) or a bound
/// `|got − f(x)| ≤ rel·|f(x)| + abs` over its domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Err {
    Exact,
    Bound { rel: f64, abs: f64 },
}

impl Err {
    /// Inside `c` (an `Exact` realization always is).
    pub fn inside(self, c: &Contract) -> bool {
        match self {
            Err::Exact => true,
            Err::Bound { rel, abs } => rel <= c.rel && abs <= c.abs,
        }
    }

    fn key(self) -> (u64, u64) {
        match self {
            Err::Exact => (u64::MAX, u64::MAX),
            Err::Bound { rel, abs } => (rel.to_bits(), abs.to_bits()),
        }
    }
}

/// **A function unit a target computes `op` with** (one instruction or a
/// fixed sequence the target provides): what the backend emits (`sym`), its
/// error over `[lo, hi]` and where that error comes from, its issue cost.
#[derive(Clone, Copy, Debug)]
pub struct Unit {
    pub op: UnitOp,
    pub bits: u32,
    /// What the backend emits (an LLVM intrinsic, a C function).
    pub sym: &'static str,
    pub err: Err,
    /// The argument interval `err` holds on.
    pub lo: f64,
    pub hi: f64,
    /// Issue slots (an FMA = 1).
    pub cost: u32,
    /// It returns the contract's special values itself (`±0`, `±∞`, NaN,
    /// negative operands); else a form wraps it in selects.
    pub specials: bool,
    /// The proof or the calibration line `err` is read from (empty: none — a
    /// `Bound` without evidence is never admitted).
    pub evidence: &'static str,
}

impl PartialEq for Unit {
    fn eq(&self, o: &Self) -> bool {
        (
            self.op,
            self.bits,
            self.sym,
            self.cost,
            self.specials,
            self.evidence,
        ) == (o.op, o.bits, o.sym, o.cost, o.specials, o.evidence)
            && self.err.key() == o.err.key()
            && self.lo.to_bits() == o.lo.to_bits()
            && self.hi.to_bits() == o.hi.to_bits()
    }
}
impl Eq for Unit {}

/// A routine a target supplies for one builtin and width.
#[derive(Clone, Copy, Debug)]
pub enum DefBody {
    /// Plain operations (traced, printed and re-measured like a reference).
    Steps(fn(&mut Trace, Val) -> Val),
    /// An external routine the target links (a vendor library, a ROM).
    Symbol(&'static str),
}

/// **A target-supplied definition** (an override): the routine, its
/// declared error and the evidence for it, its cost (`None`: the traced
/// steps').
#[derive(Clone, Copy, Debug)]
pub struct TargetDef {
    pub f: MathFn,
    pub bits: u32,
    pub body: DefBody,
    pub err: Err,
    pub cost: Option<u32>,
    pub evidence: &'static str,
}

impl PartialEq for TargetDef {
    fn eq(&self, o: &Self) -> bool {
        let body = |b: &DefBody| match b {
            DefBody::Steps(p) => (*p as usize, ""),
            DefBody::Symbol(s) => (0, *s),
        };
        (self.f, self.bits, self.cost, self.evidence) == (o.f, o.bits, o.cost, o.evidence)
            && self.err.key() == o.err.key()
            && body(&self.body) == body(&o.body)
    }
}
impl Eq for TargetDef {}

/// **What a target declares for the math builtins**: its units and its own
/// definitions. Empty: every builtin is its reference (a target with no
/// float function unit — an ASIC — computes them from plain operations).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Target {
    pub units: &'static [Unit],
    pub defs: &'static [TargetDef],
}

impl Target {
    pub const NONE: Target = Target {
        units: &[],
        defs: &[],
    };

    /// Every unit computing `op` at `bits` over (at least) `[lo, hi]`.
    fn units_for(
        &self,
        op: UnitOp,
        bits: u32,
        lo: f64,
        hi: f64,
    ) -> impl Iterator<Item = Unit> + '_ {
        self.units
            .iter()
            .filter(move |u| u.op == op && u.bits == bits && u.lo <= lo && u.hi >= hi)
            .copied()
    }
}

/// A form composing units ([`forms`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `exp` ← `ex2` after the Cody–Waite reduction (ADR-0042 D5).
    ExpEx2(Unit),
    /// The builtin ← a unit computing it over the contract's whole domain
    /// (`sqrt.approx`, IEEE `sqrt`, an accelerator's `exp`), wrapped in the
    /// special values where the unit does not return them.
    Direct(Unit),
    /// `sin` / `cos` ← on the Payne–Hanek remainder, `sin r` and `cos r`
    /// each from a unit (`Some`) or the reference's polynomial (`None`).
    Trig(Option<Unit>, Option<Unit>),
}

/// How a builtin is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Real {
    /// The reference, operation for operation.
    Reference,
    /// The reference with its contractible steps fused (the default face
    /// where the face contracts).
    Contracted,
    Form(Form),
    /// The target's `i`-th definition.
    Target(usize),
}

/// A candidate realization with its error and cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    pub f: MathFn,
    pub bits: u32,
    pub real: Real,
    pub err: Err,
    pub cost: u32,
    /// Its contractible steps fuse (a contracting default face): the
    /// printed definition carries the `contract` flag on them.
    pub fused: bool,
    /// Where `err` comes from.
    pub evidence: &'static str,
}

/// The faces a realization is admitted on (ADR-0040).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// Exact, contract and fixed: the reference's bits.
    Exact,
    /// The default face: within the contract.
    Default,
}

/// Why a candidate is not admitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refusal {
    /// Not the reference's bits, on a face that demands them.
    NotExact,
    /// Its bound is outside the contract.
    Bound,
    /// A measured bound with no evidence.
    NoEvidence,
    /// Its own trace measures outside its declaration (`got` the worst
    /// relative excess found).
    Measured { at: f64 },
}

/// The forms `f` at `bits` has on `target` (each needs its units).
pub fn forms(f: MathFn, bits: u32, target: &Target) -> Vec<Form> {
    // The remainder's range: `|r| ≤ π/4` before its one rounding.
    let pio4 = std::f64::consts::FRAC_PI_4 * (1.0 + 2f64.powi(-22));
    let mut out = Vec::new();
    match f {
        MathFn::Exp => {
            if bits == 32 {
                out.extend(
                    target
                        .units_for(UnitOp::Ex2, 32, -0.5, 0.5)
                        .map(Form::ExpEx2),
                );
            }
        }
        MathFn::Sqrt => {}
        MathFn::Sin | MathFn::Cos => {
            // Every mix of a unit (or the polynomial) for `sin r` and for
            // `cos r` but the polynomial for both (the reference).
            let ss: Vec<Option<Unit>> = std::iter::once(None)
                .chain(target.units_for(UnitOp::Sin, bits, -pio4, pio4).map(Some))
                .collect();
            let cs: Vec<Option<Unit>> = std::iter::once(None)
                .chain(target.units_for(UnitOp::Cos, bits, -pio4, pio4).map(Some))
                .collect();
            for a in &ss {
                for b in &cs {
                    if a.is_some() || b.is_some() {
                        out.push(Form::Trig(*a, *b));
                    }
                }
            }
        }
    }
    // A unit computing `f` itself over the contract's domain.
    let (lo, hi) = match f {
        MathFn::Exp => match ExpForm::of_bits(bits) {
            Some(e) => (e.val(e.lo), e.val(e.hi)),
            None => return out,
        },
        MathFn::Sqrt => (0.0, f64::INFINITY),
        MathFn::Sin | MathFn::Cos => (f64::NEG_INFINITY, f64::INFINITY),
    };
    out.extend(
        target
            .units_for(UnitOp::of(f), bits, lo, hi)
            .map(Form::Direct),
    );
    out
}

/// A form's error from its units' (ADR-0046 D5).
fn form_err(bits: u32, form: &Form) -> Err {
    let u = 2f64.powi(if bits == 64 { -53 } else { -24 });
    match form {
        Form::ExpEx2(e) => match e.err {
            // ADR-0042 D5: `(u_ex2 + 1)·2^−24`.
            Err::Exact => Err::Bound {
                rel: u,
                abs: tiny(bits),
            },
            Err::Bound { rel, .. } => Err::Bound {
                rel: rel + u,
                abs: tiny(bits),
            },
        },
        Form::Direct(s) => s.err,
        Form::Trig(s, c) => {
            // A part from the polynomial: the reference's own (≤ 2 ulps,
            // `sin_cos_f32_within_contract_over_the_whole_range`); from a
            // unit: its measured error; plus the remainder's one rounding
            // (`|r·tan r|`, `|r·cot r|` ≤ 1 on `[−π/4, π/4]`: ≤ 2u).
            let part = |p: &Option<Unit>| match p {
                None => (4.0 * u, 0.0),
                Some(Unit {
                    err: Err::Exact, ..
                }) => (u / 2.0, 0.0),
                Some(Unit {
                    err: Err::Bound { rel, abs },
                    ..
                }) => (*rel, *abs),
            };
            let ((rs, as_), (rc, ac)) = (part(s), part(c));
            Err::Bound {
                rel: rs.max(rc) + 2.0 * u,
                abs: as_.max(ac).max(tiny(bits)),
            }
        }
    }
}

/// The definition a choice prints, traced.
pub fn trace(c: &Choice, target: &Target) -> Def {
    let (mut t, x) = Trace::new(c.bits);
    let y = match c.real {
        Real::Reference | Real::Contracted => reference(c.f, &mut t, x),
        Real::Form(Form::ExpEx2(u)) => exp_body(&mut t, x, Some(&u)),
        Real::Form(Form::Direct(u)) => {
            let y = t.unit(&u, x);
            if u.specials {
                y
            } else {
                wrap_specials(c.f, &mut t, x, y)
            }
        }
        Real::Form(Form::Trig(s, cu)) => trig_def(&mut t, x, c.f == MathFn::Cos, (s, cu)),
        Real::Target(i) => match target.defs[i].body {
            DefBody::Steps(body) => body(&mut t, x),
            DefBody::Symbol(sym) => {
                let u = Unit {
                    op: UnitOp::of(c.f),
                    bits: c.bits,
                    sym,
                    err: target.defs[i].err,
                    lo: f64::NEG_INFINITY,
                    hi: f64::INFINITY,
                    cost: c.cost,
                    specials: true,
                    evidence: target.defs[i].evidence,
                };
                // An external routine: one call (`Op::Unit` with its symbol).
                t.unit(&u, x)
            }
        },
    };
    t.finish(y)
}

/// Every candidate for `f` at `bits` on `target`, with its error and cost
/// (admitted or not). `fuse`: the face is the default one and contracts —
/// the contracted reference is a candidate, and every candidate's
/// contractible steps fuse (one slot fewer each).
pub fn candidates(f: MathFn, bits: u32, target: &Target, fuse: bool) -> Vec<Choice> {
    let mk = |real: Real, err: Err, evidence: &'static str| Choice {
        f,
        bits,
        real,
        err,
        cost: 0,
        fused: fuse && real != Real::Reference,
        evidence,
    };
    let mut out = vec![mk(Real::Reference, Err::Exact, "the reference")];
    if fuse {
        out.push(mk(
            Real::Contracted,
            Err::Bound {
                rel: contract(f, bits).map_or(0.0, |c| c.rel),
                abs: contract(f, bits).map_or(0.0, |c| c.abs),
            },
            "the reference contracted: within the contract (measured, ADR-0042 D5 / ADR-0046 D5)",
        ));
    }
    for form in forms(f, bits, target) {
        let ev = match form {
            Form::ExpEx2(u) | Form::Direct(u) => u.evidence,
            // Every unit the form uses must cite its evidence.
            Form::Trig(s, c) => {
                let evs: Vec<&'static str> = [s, c].iter().flatten().map(|u| u.evidence).collect();
                if evs.iter().any(|e| e.is_empty()) {
                    ""
                } else {
                    evs[0]
                }
            }
        };
        out.push(mk(Real::Form(form), form_err(bits, &form), ev));
    }
    for (i, d) in target.defs.iter().enumerate() {
        if d.f == f && d.bits == bits {
            out.push(mk(Real::Target(i), d.err, d.evidence));
        }
    }
    for c in &mut out {
        let def = trace(c, target);
        let real = c.real;
        let unit_cost = |op: UnitOp| -> u32 {
            match real {
                Real::Form(Form::ExpEx2(u)) | Real::Form(Form::Direct(u)) if u.op == op => u.cost,
                Real::Form(Form::Trig(s, k)) => [s, k]
                    .iter()
                    .flatten()
                    .find(|u| u.op == op)
                    .map_or(1, |u| u.cost),
                _ => 1,
            }
        };
        let fused = if c.fused { def.fusable() } else { 0 };
        c.cost = match real {
            Real::Target(i) => target.defs[i]
                .cost
                .unwrap_or_else(|| def.ops(unit_cost) - fused),
            _ => def.ops(unit_cost) - fused,
        };
    }
    out
}

/// Whether `c` is admitted on `face` (ADR-0046 D6).
pub fn admits(c: &Choice, face: Face, target: &Target) -> Result<(), Refusal> {
    let Some(con) = contract(c.f, c.bits) else {
        return Err(Refusal::Bound);
    };
    match (face, c.err) {
        (Face::Exact, Err::Exact) => {}
        (Face::Exact, Err::Bound { .. }) => return Err(Refusal::NotExact),
        (Face::Default, e) => {
            if !e.inside(&con) {
                return Err(Refusal::Bound);
            }
            if matches!(e, Err::Bound { .. }) && c.evidence.is_empty() {
                return Err(Refusal::NoEvidence);
            }
        }
    }
    // A target's plain-operation routine is re-measured against its own
    // declaration over the contract's test domain.
    if let Real::Target(i) = c.real
        && let DefBody::Steps(_) = target.defs[i].body
    {
        measure_def(c, target)?;
    }
    Ok(())
}

/// A traced target definition against its declaration over [`test_args`]:
/// `Exact` must give the reference's bits (NaN: any NaN), `Bound` must hold
/// at every argument.
fn measure_def(c: &Choice, target: &Target) -> Result<(), Refusal> {
    let def = trace(c, target);
    let wide = c.bits == 64;
    let slack = if wide { 2f64.powi(-52) } else { 0.0 };
    for x in test_args(c.f, c.bits) {
        let xb = if wide {
            x.to_bits()
        } else {
            u64::from((x as f32).to_bits())
        };
        let got_b = def.eval(xb);
        let got = if wide {
            f64::from_bits(got_b)
        } else {
            f64::from(f32::from_bits(got_b as u32))
        };
        let ok = match c.err {
            Err::Exact => {
                let want = if wide {
                    eval_f64(c.f, x)
                } else {
                    f64::from(eval_f32(c.f, x as f32))
                };
                (want.is_nan() && got.is_nan()) || want.to_bits() == got.to_bits()
            }
            Err::Bound { rel, abs } => {
                if let Some(s) = special(c.f, c.bits, x) {
                    within(c.f, c.bits, x, got, s, slack)
                } else {
                    let t = truth(c.f, x);
                    (got - t).abs() <= (rel + slack) * t.abs() + abs
                }
            }
        };
        if !ok {
            return Err(Refusal::Measured { at: x });
        }
    }
    Ok(())
}

/// **The realization of `f` at `bits`**: the cheapest candidate admitted on
/// `face` (ties: the earlier — the reference first). `contract`: the face
/// fuses steps (the contracted reference is a candidate on the default
/// face); `hw` off: the exact face's choice on every face (the OFF arm).
pub fn choose(
    f: MathFn,
    bits: u32,
    face: Face,
    contract: bool,
    hw: bool,
    target: &Target,
) -> Choice {
    let face = if hw { face } else { Face::Exact };
    let mut best: Option<Choice> = None;
    for c in candidates(f, bits, target, face == Face::Default && contract) {
        if admits(&c, face, target).is_err() {
            continue;
        }
        if best.is_none_or(|b| c.cost < b.cost) {
            best = Some(c);
        }
    }
    best.expect("the reference is always admitted")
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
