//! Target realizations as data: units, definitions, forms, admission and choice.

use super::*;

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

    pub(super) fn key(self) -> (u64, u64) {
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
    pub(super) fn units_for(
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
pub(super) fn form_err(bits: u32, form: &Form) -> Err {
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
pub(super) fn measure_def(c: &Choice, target: &Target) -> Result<(), Refusal> {
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
