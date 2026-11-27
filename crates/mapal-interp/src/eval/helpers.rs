//! The interpreter's value helpers: decoding, arithmetic, comparisons and element ops.

use super::*;

// --- helpers --------------------------------------------------------------

/// ADR-0045 (generalized): a readable value of type `ty` from `bytes` at
/// `*at`, leaves in order, no padding (`mapal_ir::read::readable`).
pub(super) fn decode_value(bytes: &[u8], ty: &Ty, at: &mut usize) -> RValue {
    match ty {
        Ty::Array { elem, size } => {
            if let Some(w) = mapal_ir::read::elem_bytes(elem) {
                let n = *size as usize * w as usize;
                let vals = mapal_ir::read::decode(&bytes[*at..*at + n], elem)
                    .into_iter()
                    .map(RValue::Scalar)
                    .collect();
                *at += n;
                RValue::array(vals)
            } else {
                RValue::array((0..*size).map(|_| decode_value(bytes, elem, at)).collect())
            }
        }
        Ty::Tuple(ts) => RValue::Tuple(ts.iter().map(|t| decode_value(bytes, t, at)).collect()),
        Ty::Struct { name, fields } => RValue::Struct {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(n, t)| (n.clone(), decode_value(bytes, t, at)))
                .collect(),
        },
        scalar => {
            let w = mapal_ir::read::elem_bytes(scalar).expect("a readable leaf") as usize;
            let v = mapal_ir::read::decode(&bytes[*at..*at + w], scalar).remove(0);
            *at += w;
            RValue::Scalar(v)
        }
    }
}

/// The process-lifetime monotonic epoch for `TimeMs` (plan-time-builtin): one
/// `Instant` shared by every read in the process, so two reads are
/// non-decreasing and a difference is real elapsed milliseconds. Same clock
/// (`std::time::Instant`) as mapal-rt's `mapal_time_ms`/`mapal_perf_begin`.
pub(super) fn time_epoch() -> &'static std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now)
}

/// Read `env[o]`; panics on an undefined object (a sealed, topo-ordered graph
/// never reads before write).
pub(super) fn read<'c>(ctx: &'c EvalCtx, o: ObjectId) -> &'c RValue {
    ctx.env
        .get(o)
        .unwrap_or_else(|| unreachable!("read before write of object"))
}

/// Write `env[o]`.
pub(super) fn write(ctx: &mut EvalCtx, o: ObjectId, v: RValue) {
    ctx.env.insert(o, v);
}

/// Component `k` of an aggregate (Tuple/Struct/Array).
pub(super) fn component(x: &RValue, k: u32) -> &RValue {
    match x {
        RValue::Tuple(es) => &es[k as usize],
        RValue::Struct { fields, .. } => &fields[k as usize].1,
        RValue::Array(es) => &es[k as usize],
        _ => unreachable!("component of a non-aggregate"),
    }
}

/// Stage a `Pair` component into `target`'s buffer; finalize at the last slot.
pub(super) fn stage_pair(
    ctx: &mut EvalCtx,
    source: ObjectId,
    target: ObjectId,
    slot: u32,
    arity: u32,
) {
    let v = read(ctx, source).clone();
    let buf = ctx
        .staging
        .entry(target)
        .unwrap_or_else(|| unreachable!("target object resolves"))
        .or_insert_with(|| vec![None; arity as usize]);
    if buf.len() != arity as usize {
        buf.resize(arity as usize, None);
    }
    buf[slot as usize] = Some(v);
    if buf.iter().all(|s| s.is_some()) {
        let comps: Vec<RValue> = buf.iter().cloned().map(|s| s.unwrap()).collect();
        let ty = ctx.ty_of(target).clone();
        let finalized = finalize_product(&ty, comps);
        ctx.env.insert(target, finalized);
    }
}

/// Assemble a finalized product value from staged components per the target ty.
pub(super) fn finalize_product(ty: &Ty, comps: Vec<RValue>) -> RValue {
    match ty {
        Ty::Tuple(_) => RValue::Tuple(comps),
        Ty::Array { .. } => RValue::array(comps),
        Ty::Struct { name, fields } => {
            let named = fields.iter().map(|(n, _)| n.clone()).zip(comps).collect();
            RValue::Struct {
                name: name.clone(),
                fields: named,
            }
        }
        _ => unreachable!("Pair target is not a product ty"),
    }
}

/// `Neg` (IEEE fneg for floats; arithmetic negation for ints).
pub(super) fn neg(x: &RValue) -> RValue {
    match x {
        RValue::Scalar(v) | RValue::Tracked(v, _) => RValue::Scalar(match v {
            Value::I32(n) => Value::I32(n.wrapping_neg()),
            Value::I64(n) => Value::I64(n.wrapping_neg()),
            Value::U8(n) => Value::U8(n.wrapping_neg()),
            Value::F32(x) => Value::F32(-x),
            Value::F64(x) => Value::F64(-x),
            Value::Bool(_) | Value::Str(_) | Value::Narrow(..) | Value::Unit => {
                unreachable!("Neg on non-numeric")
            }
        }),
        _ => unreachable!("Neg on non-scalar"),
    }
}

thread_local! {
    /// ADR-0041 `--weakest`: each product evaluates with its set's weakest
    /// member — a `tf32` product's f32 operands rounded to TF32 first.
    /// ponytail: a thread-local beside `orders`; thread it with them when a
    /// second evaluation mode needs one.
    pub(crate) static WEAKEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `x` rounded to TF32 (10 explicit significand bits), round-to-nearest-even:
/// the IEEE rule ADR-0041's interpreter applies to a `tf32` product's operands.
pub(crate) fn round_tf32(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x0fff + ((b >> 13) & 1)) & !0x1fff)
}

/// The weakest member a product evaluates with under `--weakest`: its f32
/// operands rounded first to TF32 (ADR-0041) or to a narrow format (ADR-0043
/// D4: `bf16` / `f16`, ties to even — the oracle's `to<bf16>` / `to<f16>`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Weak {
    None,
    Tf32,
    Narrow(mapal_ir::NarrowFloat),
}

impl Weak {
    pub(super) fn round(self, x: f32) -> f32 {
        match self {
            Weak::None => x,
            Weak::Tf32 => round_tf32(x),
            Weak::Narrow(n) => mapal_ir::narrow::widen(n, mapal_ir::narrow::round(n, f64::from(x))),
        }
    }
}

/// `Add/Sub/Mul/Div/Mod` at the operand width; integer ÷/% by 0 ⇒ trap.
pub(super) fn arith(
    ctx: &EvalCtx,
    m: MorphismId,
    source: ObjectId,
    op: Operation,
    weak: Weak,
) -> Result<RValue, Abort> {
    let pair = read(ctx, source);
    let a = scalar(component(pair, 0));
    let b = scalar(component(pair, 1));
    macro_rules! int_arith {
        ($ctor:path, $x:expr, $y:expr) => {{
            let (x, y) = ($x, $y);
            let r = match op {
                Operation::Add => x.wrapping_add(y),
                Operation::Sub => x.wrapping_sub(y),
                Operation::Mul => x.wrapping_mul(y),
                Operation::Div => {
                    if y == 0 {
                        return Err(Abort::Trapped(TrapKind::DivZero));
                    }
                    x.wrapping_div(y)
                }
                Operation::Mod => {
                    if y == 0 {
                        return Err(Abort::Trapped(TrapKind::DivZero));
                    }
                    x.wrapping_rem(y)
                }
                _ => unreachable!(),
            };
            $ctor(r)
        }};
    }
    macro_rules! float_arith {
        ($ctor:path, $x:expr, $y:expr) => {{
            let (x, y) = ($x, $y);
            let r = match op {
                Operation::Add => x + y,
                Operation::Sub => x - y,
                Operation::Mul => x * y,
                Operation::Div => x / y,
                Operation::Mod => x % y,
                _ => unreachable!(),
            };
            $ctor(r)
        }};
    }
    let v = match (a, b) {
        (Value::I32(x), Value::I32(y)) => int_arith!(Value::I32, *x, *y),
        (Value::I64(x), Value::I64(y)) => int_arith!(Value::I64, *x, *y),
        (Value::U8(x), Value::U8(y)) => int_arith!(Value::U8, *x, *y),
        (Value::F32(x), Value::F32(y)) if weak != Weak::None => {
            float_arith!(Value::F32, weak.round(*x), weak.round(*y))
        }
        (Value::F32(x), Value::F32(y)) => float_arith!(Value::F32, *x, *y),
        (Value::F64(x), Value::F64(y)) => float_arith!(Value::F64, *x, *y),
        _ => unreachable!("arith on mismatched/non-numeric operands"),
    };
    if matches!(v, Value::F32(_) | Value::F64(_)) && crate::bound::on() {
        return Ok(crate::bound::arith(
            ctx.ir,
            ctx.f,
            m,
            op,
            component(pair, 0),
            component(pair, 1),
            v,
        ));
    }
    Ok(RValue::Scalar(v))
}

/// `Eq/Neq/Lt/Gt/Le/Ge`.
pub(super) fn compare(ctx: &EvalCtx, source: ObjectId, op: Operation) -> RValue {
    let pair = read(ctx, source);
    let a = scalar(component(pair, 0));
    let b = scalar(component(pair, 1));
    let r = match op {
        Operation::Eq => a == b,
        Operation::Neq => a != b,
        Operation::Lt => num_lt(a, b),
        Operation::Gt => num_lt(b, a),
        // Le/Ge use a native `<=` (NOT `!num_lt`): for floats, IEEE ordering
        // makes every comparison with NaN false, so `Le(NaN, x)` must be false —
        // whereas `!(x < NaN)` would wrongly be true (interp DESIGN §3; per-op review).
        Operation::Le => num_le(a, b),
        Operation::Ge => num_le(b, a),
        _ => unreachable!(),
    };
    RValue::Scalar(Value::Bool(r))
}

/// Strict numeric `<` at the operand width (IEEE ordering for floats — NaN ⇒ false).
pub(super) fn num_lt(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::I32(x), Value::I32(y)) => x < y,
        (Value::I64(x), Value::I64(y)) => x < y,
        (Value::U8(x), Value::U8(y)) => x < y,
        (Value::F32(x), Value::F32(y)) => x < y,
        (Value::F64(x), Value::F64(y)) => x < y,
        _ => unreachable!("ordered compare on non-numeric/mismatched operands"),
    }
}

/// Numeric `<=` at the operand width (IEEE ordering for floats — NaN ⇒ false).
pub(super) fn num_le(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::I32(x), Value::I32(y)) => x <= y,
        (Value::I64(x), Value::I64(y)) => x <= y,
        (Value::U8(x), Value::U8(y)) => x <= y,
        (Value::F32(x), Value::F32(y)) => x <= y,
        (Value::F64(x), Value::F64(y)) => x <= y,
        _ => unreachable!("ordered compare on non-numeric/mismatched operands"),
    }
}

/// `And/Or` (strict; both operands already evaluated).
pub(super) fn logic(ctx: &EvalCtx, source: ObjectId, op: Operation) -> RValue {
    let pair = read(ctx, source);
    let a = matches!(component(pair, 0), RValue::Scalar(Value::Bool(true)));
    let b = matches!(component(pair, 1), RValue::Scalar(Value::Bool(true)));
    let r = match op {
        Operation::And => a && b,
        Operation::Or => a || b,
        _ => unreachable!(),
    };
    RValue::Scalar(Value::Bool(r))
}

/// `Index`: `(Array, I)`; `i < 0 ∨ i ≥ n ⇒ Trapped(IndexOob)`.
pub(super) fn index(ctx: &EvalCtx, source: ObjectId) -> Result<RValue, Abort> {
    let pair = read(ctx, source);
    let arr = match component(pair, 0) {
        RValue::Array(es) => es,
        _ => unreachable!("Index on a non-array"),
    };
    let i = as_int(scalar(component(pair, 1)));
    if i < 0 || i as u128 >= arr.len() as u128 {
        return Err(Abort::Trapped(TrapKind::IndexOob));
    }
    Ok(arr[i as usize].clone())
}

/// `Update`: `(Array, I, T)`; `i < 0 ∨ i ≥ n ⇒ Trapped(IndexOob)`; else a fresh
/// array with slot `i` replaced by the value operand (ADR-0021). The index
/// zero/sign-extends exactly like `Index` (shared `as_int` path).
pub(super) fn update(ctx: &EvalCtx, source: ObjectId) -> Result<RValue, Abort> {
    let triple = read(ctx, source);
    let mut arr = match component(triple, 0) {
        RValue::Array(es) => Rc::clone(es),
        _ => unreachable!("Update on a non-array"),
    };
    let i = as_int(scalar(component(triple, 1)));
    if i < 0 || i as u128 >= arr.len() as u128 {
        return Err(Abort::Trapped(TrapKind::IndexOob));
    }
    // `make_mut` is the copy in "a fresh array with slot `i` replaced"
    // (ADR-0021): the source is still live in `env`, so this copies once here
    // rather than on every clone of a value that merely *contains* the array.
    Rc::make_mut(&mut arr)[i as usize] = component(triple, 2).clone();
    Ok(RValue::Array(arr))
}

/// `Print { newline }`: `(IoToken, P) → IoToken`.
pub(super) fn print_op(ctx: &EvalCtx, source: ObjectId, newline: bool) -> RValue {
    let pair = read(ctx, source);
    let log = match component(pair, 0) {
        RValue::Token(s) => s.clone(),
        _ => unreachable!("Print's first component is not a token"),
    };
    let p = scalar(component(pair, 1));
    crate::bound::printed(component(pair, 1), newline);
    let mut out = log;
    out.push_str(&crate::value::render(p));
    if newline {
        out.push('\n');
    }
    RValue::Token(out)
}

/// Extract the array elements, panicking on a non-array.
pub(super) fn as_array(x: &RValue) -> &[RValue] {
    match x {
        RValue::Array(es) => es,
        _ => unreachable!("expected an array value"),
    }
}

/// Extract the scalar payload, panicking on a non-scalar.
pub(crate) fn scalar(x: &RValue) -> &Value {
    match x {
        RValue::Scalar(v) | RValue::Tracked(v, _) => v,
        _ => unreachable!("expected a scalar value"),
    }
}

/// An integer value as an `i128` (sign-extended).
pub(super) fn as_int(v: &Value) -> i128 {
    match v {
        Value::I32(n) => *n as i128,
        Value::I64(n) => *n as i128,
        Value::U8(n) => *n as i128,
        _ => unreachable!("Index's index operand is not an integer"),
    }
}

/// The scalar `0.0` of a float accumulator (a split partition's seed,
/// ADR-0040 D3). Only tile-site folds split, and their accumulator is a float.
pub(super) fn zero_of(acc: &RValue) -> RValue {
    match acc {
        RValue::Scalar(Value::F32(_)) => RValue::Scalar(Value::F32(0.0)),
        RValue::Scalar(Value::F64(_)) => RValue::Scalar(Value::F64(0.0)),
        _ => unreachable!("a split fold accumulates a float"),
    }
}

/// `x + y` on two float scalars, one rounding — the combine of ADR-0040 D3.
pub(super) fn add_floats(x: &RValue, y: &RValue) -> RValue {
    match (x, y) {
        (RValue::Scalar(Value::F32(a)), RValue::Scalar(Value::F32(b))) => {
            RValue::Scalar(Value::F32(a + b))
        }
        (RValue::Scalar(Value::F64(a)), RValue::Scalar(Value::F64(b))) => {
            RValue::Scalar(Value::F64(a + b))
        }
        _ => unreachable!("a split fold accumulates a float"),
    }
}
