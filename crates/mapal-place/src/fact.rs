//! **Facts with provenance** (ADR-0050 D2, "Provenance"): every machine fact
//! carries where it came from and an interval. A derived or nominal fact is
//! *marked*: a decision that flips inside its interval is open (D5, minimax).

/// Where a fact's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Measured on the part itself (a probe; the cite names the line or file).
    Measured(&'static str),
    /// Queried from the part (`cuDeviceGetAttribute`, sysfs, `sysctl`).
    Queried,
    /// Declared by a design: an ISA or programming-guide rule, a datasheet's
    /// geometry, RTL parameters, a simulation.
    Declared(&'static str),
    /// Derived from other facts (the derivation named). Marked.
    Derived(&'static str),
    /// A placeholder with no measurement behind it. Marked; its interval is
    /// wide on purpose.
    Nominal(&'static str),
}

impl Src {
    /// A derived or nominal fact: a pick that flips inside its interval is
    /// open, and the fact is named in the evidence.
    pub fn marked(self) -> bool {
        matches!(self, Src::Derived(_) | Src::Nominal(_))
    }

    /// The source's kind as one word (`measured`, `queried`, …).
    pub fn kind(self) -> &'static str {
        match self {
            Src::Measured(_) => "measured",
            Src::Queried => "queried",
            Src::Declared(_) => "declared",
            Src::Derived(_) => "derived",
            Src::Nominal(_) => "nominal",
        }
    }

    /// The cite or derivation (empty for a query).
    pub fn note(self) -> &'static str {
        match self {
            Src::Measured(s) | Src::Declared(s) | Src::Derived(s) | Src::Nominal(s) => s,
            Src::Queried => "",
        }
    }
}

/// Which end of the facts' intervals an evaluation reads (D5: facts with
/// intervals are evaluated at their bounds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// Every rate at its low end, every latency and cost at its high end:
    /// the slow machine.
    Slow,
    /// The values themselves.
    Value,
    /// Every rate high, every latency and cost low: the fast machine.
    Fast,
}

/// A value, its interval `[lo, hi]` and its source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fact<T> {
    pub v: T,
    pub lo: T,
    pub hi: T,
    pub src: Src,
}

impl<T: Copy + PartialOrd> Fact<T> {
    /// A fact whose interval is the point `v`.
    pub const fn point(v: T, src: Src) -> Self {
        Fact {
            v,
            lo: v,
            hi: v,
            src,
        }
    }

    pub const fn measured(v: T, cite: &'static str) -> Self {
        Self::point(v, Src::Measured(cite))
    }

    pub const fn queried(v: T) -> Self {
        Self::point(v, Src::Queried)
    }

    pub const fn declared(v: T, cite: &'static str) -> Self {
        Self::point(v, Src::Declared(cite))
    }

    /// A derived value inside `[lo, hi]`.
    pub const fn derived(v: T, lo: T, hi: T, how: &'static str) -> Self {
        Fact {
            v,
            lo,
            hi,
            src: Src::Derived(how),
        }
    }

    /// A nominal placeholder inside `[lo, hi]`.
    pub const fn nominal(v: T, lo: T, hi: T, why: &'static str) -> Self {
        Fact {
            v,
            lo,
            hi,
            src: Src::Nominal(why),
        }
    }

    /// The same fact with a measured spread (`[lo, hi]`, e.g. the IQR).
    pub fn within(self, lo: T, hi: T) -> Self {
        Fact { lo, hi, ..self }
    }

    /// The value a **rate** takes at bound `b` (a rate is low on the slow
    /// machine).
    pub fn rate(&self, b: Bound) -> T {
        match b {
            Bound::Slow => self.lo,
            Bound::Value => self.v,
            Bound::Fast => self.hi,
        }
    }

    /// The value a **cost** (a latency, a fixed time) takes at bound `b` (a
    /// cost is high on the slow machine).
    pub fn cost(&self, b: Bound) -> T {
        match b {
            Bound::Slow => self.hi,
            Bound::Value => self.v,
            Bound::Fast => self.lo,
        }
    }

    pub fn marked(&self) -> bool {
        self.src.marked()
    }
}

impl Fact<f64> {
    /// `self` scaled by `x` (the interval with it): a per-instance rate
    /// times a count.
    pub fn scale(self, x: f64) -> Self {
        Fact {
            v: self.v * x,
            lo: self.lo * x,
            hi: self.hi * x,
            src: self.src,
        }
    }
}

/// One named fact as a row of the provenance listing: name, value, interval,
/// source kind and note.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub name: String,
    pub v: f64,
    pub lo: f64,
    pub hi: f64,
    pub src: Src,
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\t{}", self.name, self.v)?;
        if self.lo != self.v || self.hi != self.v {
            write!(f, " [{} .. {}]", self.lo, self.hi)?;
        }
        write!(f, "\t{}\t{}", self.src.kind(), self.src.note())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_read_the_slow_and_fast_machine() {
        let r = Fact::derived(10.0, 8.0, 12.0, "x");
        assert_eq!(r.rate(Bound::Slow), 8.0);
        assert_eq!(r.cost(Bound::Slow), 12.0);
        assert_eq!(r.rate(Bound::Fast), 12.0);
        assert!(r.marked());
        assert!(!Fact::measured(1.0, "p").marked());
        assert_eq!(r.scale(2.0).hi, 24.0);
    }
}
