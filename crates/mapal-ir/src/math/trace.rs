//! Tracing: a definition as data (`Trace`, `Def`) a backend prints.

use super::*;

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
    pub(super) bits: u32,
    pub(super) steps: Vec<Step>,
    pub(super) contract: bool,
}

pub(super) const NO: Val = Val(u32::MAX);

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

    pub(super) fn push(&mut self, op: Op, args: [Val; 3], ty: VTy) -> Val {
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

    pub(super) fn ty(&self, v: Val) -> VTy {
        self.steps[v.0 as usize].ty
    }
}

pub(super) fn wty(w: W) -> VTy {
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

    pub(super) fn run<const B: u32>(&self, n: &mut Native<B>, x: u64) -> u64 {
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
