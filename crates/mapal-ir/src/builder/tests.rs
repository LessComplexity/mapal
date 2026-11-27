//! The rejection matrix (DESIGN §16 item 1): every [`IrError`] variant has ≥1
//! test driving the builder into exactly that error, plus the reviewer-named
//! holes. Mirrors mapal-syntax's split-file unit-test pattern.

use super::*;

mod arrays;
mod captures;
mod declarations;
mod loops_returns;
mod tokens_calls;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

// --- helpers --------------------------------------------------------------

/// A builder with one Named `main : input → output`, returning (builder, f).
fn one_fn(input: Ty, output: Ty) -> (IrBuilder, FuncId) {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", input, output, L)
        .unwrap();
    (b, f)
}

fn i32_arr(n: u64) -> Ty {
    Ty::Array {
        elem: Box::new(Ty::i32()),
        size: n,
    }
}
