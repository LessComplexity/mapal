//! ADR-0051 D4: `widen` is exact — admitted where every value of the source
//! format is a value of the target ([`crate::formats::embeds`]), or where the
//! facts prove the value is one. The one pair this applies to today is
//! `i32 → f32` (exact on ±2²⁴): emission logs each such conversion
//! ([`WidenCheck`]) and, once the IR is sealed, the facts decide it —
//! `bounds_proof`'s range of the converted value where it runs
//! ([`mapal_ir::BoundsProof::widen_range`]), else, for a value computable
//! before the program runs (`static_plan`, ADR-0048 D3), its value on the
//! oracle. Anything else is L1614, suggesting `to<f32>` (which rounds).

use mapal_ir::{CategoryIr, ObjectId, Operation, Value};
use mapal_syntax::{Diagnostic, SourceLoc};

use crate::diag::{LCode, diag};
use crate::formats::{F32_EXACT_INT, Format};

/// One `widen` whose exactness the facts must prove.
#[derive(Clone, Debug)]
pub struct WidenCheck {
    /// The object the `Widen` edge writes.
    pub target: ObjectId,
    pub source: Format,
    pub format: Format,
    pub span: SourceLoc,
    /// The function (or instance) it was emitted in, for the message.
    pub owner: String,
}

/// The oracle steps a compile-time value's evaluation may take (ADR-0048's
/// `EVAL_WORK`).
const EVAL_BUDGET: u64 = mapal_ir::EVAL_WORK;

/// Decide every logged `widen` on the sealed `ir`: the diagnostics of those
/// the facts do not prove exact.
pub fn check(ir: &CategoryIr, checks: &[WidenCheck]) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for c in checks {
        let Some((m, src)) = ir.in_edges(c.target).iter().find_map(|&m| {
            let morph = ir.morphism(m)?;
            (morph.op == Operation::Widen).then_some((m, morph.source))
        }) else {
            continue;
        };
        let f = ir.owner(c.target);
        let limit = match (c.source, c.format) {
            (Format::I32, Format::F32) => F32_EXACT_INT,
            _ => continue,
        };
        let inside = |(lo, hi): (i128, i128)| lo >= -limit && hi <= limit;
        let range = ir.bounds_proof(f).widen_range(m);
        if range.is_some_and(inside) {
            continue;
        }
        let value = (ir.static_plan(f).is_fixed(src))
            .then(|| mapal_interp::eval_values(ir, f, &[src], EVAL_BUDGET))
            .and_then(|v| v.into_iter().next().flatten());
        if let Some(mapal_interp::RValue::Scalar(Value::I32(v))) = value
            && inside((i128::from(v), i128::from(v)))
        {
            continue;
        }
        let known = match (range, value) {
            (_, Some(mapal_interp::RValue::Scalar(Value::I32(v)))) => {
                format!("it is {v} before the program runs")
            }
            (Some((lo, hi)), _) => format!("the facts bound it to [{lo}, {hi}]"),
            (None, _) => "the facts bound it nowhere (a function's input, a loop's state)".into(),
        };
        let inst = if c.owner.contains('<') {
            format!(" (in `{}`)", c.owner)
        } else {
            String::new()
        };
        out.push(diag(
            LCode::InvalidWiden,
            c.span,
            format!(
                "`widen<{}>` of an `{}` is exact only within ±2²⁴ ({F32_EXACT_INT}), and {known} \
                 — write `to<{}>`, which rounds (ADR-0051 D4){inst}",
                c.format, c.source, c.format
            ),
        ));
    }
    out
}
