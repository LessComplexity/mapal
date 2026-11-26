//! ADR-0051 — format generics: the scalar formats a format parameter ranges
//! over, the kinds that bound it (D1), a generic argument (a size or a
//! format, D2/D5), and the conversion builtins `to` / `widen` (D4) — their
//! prelude signatures and the pairs each admits.
//!
//! The conversions go through the generic machinery of [`crate::sizes`]
//! (explicit prefix, inference from the argument, then from the expected
//! type) with the prelude's declarations, and are realized by the existing
//! operations: `Operation::Narrow` (ADR-0043 rounding) and
//! `Operation::Widen` (ADR-0029 widening).

use std::fmt;
use std::sync::OnceLock;

use mapal_ir::{NarrowFloat, Ty};
use mapal_syntax as syn;

/// A scalar format: a value a format parameter takes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Format {
    F64,
    F32,
    Bf16,
    F16,
    F8e4m3,
    F8e5m2,
    I32,
    I64,
    U8,
}

impl Format {
    pub const ALL: [Format; 9] = [
        Format::F64,
        Format::F32,
        Format::Bf16,
        Format::F16,
        Format::F8e4m3,
        Format::F8e5m2,
        Format::I32,
        Format::I64,
        Format::U8,
    ];

    /// The format a scalar type name spells (`bf16`, `i32`, …).
    pub fn from_name(name: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::F64 => "f64",
            Format::F32 => "f32",
            Format::Bf16 => "bf16",
            Format::F16 => "f16",
            Format::F8e4m3 => "f8e4m3",
            Format::F8e5m2 => "f8e5m2",
            Format::I32 => "i32",
            Format::I64 => "i64",
            Format::U8 => "u8",
        }
    }

    pub fn ty(self) -> Ty {
        match self {
            Format::F64 => Ty::f64(),
            Format::F32 => Ty::f32(),
            Format::Bf16 => Ty::bf16(),
            Format::F16 => Ty::f16(),
            Format::F8e4m3 => Ty::f8e4m3(),
            Format::F8e5m2 => Ty::f8e5m2(),
            Format::I32 => Ty::i32(),
            Format::I64 => Ty::i64(),
            Format::U8 => Ty::u8(),
        }
    }

    /// The format of a scalar number type (`None`: not a number).
    pub fn of(ty: &Ty) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.ty() == *ty)
    }

    /// ADR-0043: the narrow storage format, for the narrow floats.
    pub fn narrow(self) -> Option<NarrowFloat> {
        match self {
            Format::Bf16 => Some(NarrowFloat::Bf16),
            Format::F16 => Some(NarrowFloat::F16),
            Format::F8e4m3 => Some(NarrowFloat::F8e4m3),
            Format::F8e5m2 => Some(NarrowFloat::F8e5m2),
            _ => None,
        }
    }

    pub fn is_float(self) -> bool {
        Kind::Float.admits(self)
    }

    pub fn is_int(self) -> bool {
        Kind::Int.admits(self)
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// D1: a format parameter's kind — a closed set of formats.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// `f64`, `f32`, `bf16`, `f16`, `f8e4m3`, `f8e5m2`.
    Float,
    /// The floats narrower than `f32`.
    Narrow,
    /// `i32`, `i64`, `u8`.
    Int,
    /// `float` ∪ `int`.
    Number,
}

impl Kind {
    pub fn from_name(name: &str) -> Option<Kind> {
        match name {
            "float" => Some(Kind::Float),
            "narrow" => Some(Kind::Narrow),
            "int" => Some(Kind::Int),
            "number" => Some(Kind::Number),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Float => "float",
            Kind::Narrow => "narrow",
            Kind::Int => "int",
            Kind::Number => "number",
        }
    }

    pub fn admits(self, f: Format) -> bool {
        match self {
            Kind::Float => matches!(
                f,
                Format::F64
                    | Format::F32
                    | Format::Bf16
                    | Format::F16
                    | Format::F8e4m3
                    | Format::F8e5m2
            ),
            Kind::Narrow => f.narrow().is_some(),
            Kind::Int => matches!(f, Format::I32 | Format::I64 | Format::U8),
            Kind::Number => true,
        }
    }

    /// The kind's formats, `f64, f32, …` — for diagnostics.
    pub fn members(self) -> String {
        Format::ALL
            .into_iter()
            .filter(|f| self.admits(*f))
            .map(|f| format!("`{f}`"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The kind of a declared generic parameter: `None` for a size parameter;
/// `Some(Err(text))` for a kind name that is none of D1's (L1711, reported at
/// the declaration).
pub fn kind_of<'s>(source: &'s str, p: &syn::GenericParam) -> Option<Result<Kind, &'s str>> {
    p.kind.map(|k| {
        let text = &source[k.span.start as usize..k.span.end as usize];
        Kind::from_name(text).ok_or(text)
    })
}

/// A generic argument's value: a size (ADR-0047) or a format (ADR-0051).
/// Ordered sizes numerically, formats in [`Format::ALL`] order (an
/// instance's position in the IR, ADR-0047 D6).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Arg {
    Size(u64),
    Format(Format),
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Size(n) => write!(f, "{n}"),
            Arg::Format(x) => write!(f, "{x}"),
        }
    }
}

/// D4: the conversion builtins' signatures — ordinary generic signatures,
/// parsed once. Their bodies are never lowered: a call is realized by the
/// builtin operation the pair admits ([`admit`]).
pub const PRELUDE: &str = "\
fn to<target: number, source: number>(value: source) -> target {
    value -> ret;
}

fn widen<target: number, source: number>(value: source) -> target {
    value -> ret;
}
";

/// The two conversion verbs (D4: `widen` is exact, `to` rounds).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verb {
    To,
    Widen,
}

impl Verb {
    pub fn from_name(name: &str) -> Option<Verb> {
        match name {
            "to" => Some(Verb::To),
            "widen" => Some(Verb::Widen),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Verb::To => "to",
            Verb::Widen => "widen",
        }
    }

    /// The prelude declaration of this verb.
    pub fn decl(self) -> &'static syn::FnDecl {
        static PARSED: OnceLock<syn::Program> = OnceLock::new();
        let program = PARSED.get_or_init(|| {
            let out = syn::parse(PRELUDE);
            debug_assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
            out.program
        });
        let name = self.name();
        program
            .items
            .iter()
            .find_map(|i| match i {
                syn::Item::Fn(fd)
                    if &PRELUDE[fd.name.span.start as usize..fd.name.span.end as usize] == name =>
                {
                    Some(fd)
                }
                _ => None,
            })
            .expect("prelude declares to and widen")
    }
}

/// D6: a removed format-named conversion and its replacement.
pub fn removed(name: &str) -> Option<(Verb, Format)> {
    Some(match name {
        "to_bf16" => (Verb::To, Format::Bf16),
        "to_f16" => (Verb::To, Format::F16),
        "to_f8e4m3" => (Verb::To, Format::F8e4m3),
        "to_f8e5m2" => (Verb::To, Format::F8e5m2),
        "widen_i64" => (Verb::Widen, Format::I64),
        "widen_f32" => (Verb::Widen, Format::F32),
        "widen_f64" => (Verb::Widen, Format::F64),
        _ => return None,
    })
}

/// How an admitted conversion is realized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Realize {
    /// `source = target`: erased (D4).
    Identity,
    /// `Operation::Widen` (ADR-0029 / ADR-0043 widening; `sitofp` of an
    /// `i32` rounds to nearest).
    Widen,
    /// `Operation::Narrow`: an `f32` rounded once (ADR-0043).
    Narrow(NarrowFloat),
    /// `widen` of an `i32` to `f32`: exact only within ±2²⁴ — admitted where
    /// the facts prove the value is ([`crate::convert`], after sealing).
    WidenProven,
}

/// Every value of `source` is a value of `target` (D4's embeddings).
pub fn embeds(source: Format, target: Format) -> bool {
    use Format::*;
    match (source, target) {
        (s, t) if s == t => true,
        (Bf16 | F16 | F8e4m3 | F8e5m2, F32 | F64) => true,
        (F8e4m3 | F8e5m2, Bf16 | F16) => true,
        (F32, F64) => true,
        (U8 | I32, I64 | F64) => true,
        (U8, I32 | F32 | Bf16 | F16) => true,
        _ => false,
    }
}

/// D4: how `verb` realizes `source → target`, or why it does not (the error
/// text, its code chosen by the caller: L1614 for `widen`, L1616 for `to`).
pub fn admit(verb: Verb, source: Format, target: Format) -> Result<Realize, String> {
    use Format::*;
    if source == target {
        return Ok(Realize::Identity);
    }
    // The pairs the IR's operations realize today.
    let widen_op = matches!(
        (source, target),
        (I32, I64 | F32 | F64) | (F32, F64) | (Bf16 | F16 | F8e4m3 | F8e5m2, F32 | F64)
    );
    match verb {
        Verb::Widen => {
            if (source, target) == (I32, F32) {
                return Ok(Realize::WidenProven);
            }
            if embeds(source, target) {
                return if widen_op {
                    Ok(Realize::Widen)
                } else {
                    Err(format!(
                        "`widen<{target}>` of a `{source}` is exact (ADR-0051 D4), but no \
                         conversion realizes it yet — go through a format it does: \
                         {}",
                        detour(source, target)
                    ))
                };
            }
            Err(match admit(Verb::To, source, target) {
                Ok(_) => format!(
                    "cannot widen `{source}` to `{target}`: `widen` is exact, and not every \
                     `{source}` is a `{target}` — write `to<{target}>`, which rounds (ADR-0051 D4)"
                ),
                Err(why) => format!(
                    "cannot widen `{source}` to `{target}`: `widen` is exact, and not every \
                     `{source}` is a `{target}`; and {why}"
                ),
            })
        }
        Verb::To => {
            if widen_op {
                return Ok(Realize::Widen);
            }
            if source == F32
                && let Some(n) = target.narrow()
            {
                return Ok(Realize::Narrow(n));
            }
            Err(match (source, target) {
                (F64, _) if target.is_float() => format!(
                    "`to<{target}>` of an `f64` is open: it needs one rounding, and going \
                     through `f32` rounds twice (ADR-0043 D2's open item)"
                ),
                (_, _) if source.is_float() && target.is_int() => format!(
                    "`to<{target}>` of a `{source}` is open: no rounding-to-integer rule is \
                     decided (ADR-0051 D4)"
                ),
                (I32 | I64 | U8, _) if target.narrow().is_some() => format!(
                    "`to<{target}>` of a `{source}` needs one rounding; through `f32` it rounds \
                     twice — write `to<f32> -> to<{target}>` if two roundings are meant \
                     (ADR-0051 D4)"
                ),
                _ => format!(
                    "no conversion realizes `to<{target}>` of a `{source}` yet (ADR-0051 D4): \
                     {}",
                    detour(source, target)
                ),
            })
        }
    }
}

fn detour(source: Format, target: Format) -> String {
    match (source, target) {
        (Format::U8, _) => format!(
            "a `u8` converts nowhere yet; read the data as `i32` (or the format it is) instead \
             of `{target}`"
        ),
        (Format::I64, _) => "an `i64` converts nowhere yet (no `i64 → float` conversion)".into(),
        (s, t) if s.narrow().is_some() && t.narrow().is_some() => {
            format!("`widen<f32> -> to<{t}>`")
        }
        _ => "no realization".into(),
    }
}

/// `i32 → f32` is exact on `±2²⁴` (`f32`'s 24-bit significand).
pub const F32_EXACT_INT: i128 = 1 << 24;
