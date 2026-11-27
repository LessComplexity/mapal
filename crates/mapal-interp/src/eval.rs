//! The evaluator: per-function env, the flat topo walk, `eval_morphism` with
//! product assembly, and the per-operation semantics (interp DESIGN §2, §3).
//!
//! The loop driver (§4) lives in [`crate::loops`]; this module owns the flat
//! walk and the per-op happy path, threading `Result<RValue, Abort>` so `?`
//! propagates the first abort (Diverged/Trapped).

use std::rc::Rc;

use slotmap::SecondaryMap;

use mapal_ir::{
    CategoryIr, FuncId, GuardSite, MorphismId, ObjectId, ObjectKind, Operation, Ty, Value,
};

use crate::value::{Abort, RValue, TrapKind};

/// Evaluation state, scoped to one function activation. Objects never cross
/// functions (ir I6), so a fresh `EvalCtx` is created per `eval_fn`.
pub(crate) struct EvalCtx<'a> {
    pub ir: &'a CategoryIr,
    /// The current function being evaluated.
    pub f: FuncId,
    /// ADR-0040 D3: the folds a compiled program split, `(func, fold, S)`,
    /// replayed in partition order. Empty: every fold in order.
    pub orders: &'a [(FuncId, MorphismId, u64)],
    /// Object → its resolved value.
    pub env: SecondaryMap<ObjectId, RValue>,
    /// Per-product staging buffers: `target → [Option<component>; arity]`.
    pub staging: SecondaryMap<ObjectId, Vec<Option<RValue>>>,
    /// Guard sites keyed by their `Phi` (plan-s39): the Phi handler evaluates
    /// the condition, fires the chosen arm's own-list, and selects.
    pub guards: SecondaryMap<MorphismId, GuardSite>,
    /// Every guard-arm-owned morphism (the sites' own-lists partition these).
    /// The flat walk and the loop driver both skip them — an arm's work fires
    /// only from its Phi, and only when the condition picks that arm.
    pub gated: SecondaryMap<MorphismId, ()>,
}

impl<'a> EvalCtx<'a> {
    fn new(ir: &'a CategoryIr, f: FuncId, orders: &'a [(FuncId, MorphismId, u64)]) -> Self {
        let mut guards = SecondaryMap::new();
        let mut gated = SecondaryMap::new();
        for site in ir.guard_plan(f).into_iter().filter(GuardSite::gated) {
            for &m in site.on_true.own.iter().chain(site.on_false.own.iter()) {
                gated.insert(m, ());
            }
            guards.insert(site.phi, site);
        }
        EvalCtx {
            ir,
            f,
            orders,
            env: SecondaryMap::new(),
            staging: SecondaryMap::new(),
            guards,
            gated,
        }
    }

    /// Decrement the budget once; `0 ⇒ Err(Diverged)` (interp DESIGN §6).
    pub(crate) fn spend(&self, budget: &mut u64) -> Result<(), Abort> {
        if *budget == 0 {
            return Err(Abort::Diverged);
        }
        *budget -= 1;
        Ok(())
    }

    /// The single morphism by id (the IR is sealed; ids resolve).
    pub(crate) fn morph(&self, m: MorphismId) -> &'a mapal_ir::Morphism {
        self.ir
            .morphism(m)
            .expect("sealed graph: morphism resolves")
    }

    /// The ty of an object.
    pub(crate) fn ty_of(&self, o: ObjectId) -> &'a Ty {
        &self.ir.object(o).expect("sealed graph: object resolves").ty
    }
}

/// Evaluate one `FuncDef` against one argument (interp DESIGN §2).
pub(crate) fn eval_fn(
    ir: &CategoryIr,
    f: FuncId,
    arg: RValue,
    budget: &mut u64,
    orders: &[(FuncId, MorphismId, u64)],
) -> Result<RValue, Abort> {
    eval_fn_within(ir, f, arg, budget, orders, None).map(|(v, _)| v)
}

/// ADR-0048: evaluate the objects `want` of `f` — a cone closed under its
/// sources that holds neither the input nor an IO token — and return their
/// values. Every other morphism is skipped; a loop runs when its merge is
/// wanted (the cone then holds the whole loop).
pub(crate) fn eval_cone(
    ir: &CategoryIr,
    f: FuncId,
    want: &SecondaryMap<ObjectId, ()>,
    budget: &mut u64,
) -> Result<SecondaryMap<ObjectId, RValue>, Abort> {
    eval_fn_within(ir, f, RValue::Unit, budget, &[], Some(want)).map(|(_, env)| env)
}

fn eval_fn_within(
    ir: &CategoryIr,
    f: FuncId,
    arg: RValue,
    budget: &mut u64,
    orders: &[(FuncId, MorphismId, u64)],
    want: Option<&SecondaryMap<ObjectId, ()>>,
) -> Result<(RValue, SecondaryMap<ObjectId, RValue>), Abort> {
    let fd = ir.func(f).expect("sealed graph: func resolves");
    let mut ctx = EvalCtx::new(ir, f, orders);
    let wanted = |o: ObjectId| want.is_none_or(|w| w.contains_key(o));

    // Seed: the one Parameter object, and every Constant.
    ctx.env.insert(fd.input, arg);
    for (id, obj) in ir.objects() {
        if ir.try_owner(id) != Some(f) {
            continue;
        }
        if obj.kind == ObjectKind::Constant {
            let v = obj.value.clone().expect("ir I7: Constant carries a value");
            ctx.env.insert(id, RValue::Scalar(v));
        }
    }

    // The in-SCC object set (incidence test for driver ownership; interp §2).
    let in_scc = build_in_scc(ir, f);

    // Driver-owned morphisms: everything in a loop plan's decide/advance
    // cones (the llvm `func.rs` walk rule, BL7-shared). Plan membership is the
    // precise rule — a computed exit payload or an exit-arm fanout leaves the
    // SCC but still belongs to the decide cone; SCC incidence alone would
    // re-evaluate it after the loop (a dead recompute for values, a DOUBLE
    // side effect for an exit-arm Print).
    let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
    for scc in ir.loop_structure(f) {
        for &mg in &scc.merges {
            if let Some(plan) = ir.loop_plan(f, mg) {
                for &mo in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                    owned.insert(mo, ());
                }
            }
        }
    }

    // The flat topo walk (interp DESIGN §2).
    for m in ir.topo_order(f) {
        let morph = ir.morphism(m).expect("sealed graph: morphism resolves");
        let op = morph.op;
        if !wanted(morph.target) {
            continue;
        }
        match op {
            Operation::LoopEnter => {
                if ctx.gated.contains_key(m) {
                    // plan-s40: an arm-owned loop — the handle is gated, and
                    // its Phi invokes the driver iff the condition picks the
                    // arm.
                    continue;
                }
                // Invoke the driver once per merge (the target LoopMerge).
                crate::loops::run_loop(&mut ctx, morph.target, budget)?;
            }
            Operation::LoopBack | Operation::LoopExit => {
                // Driver owns these.
            }
            _ => {
                let incident =
                    in_scc.contains_key(morph.source) || in_scc.contains_key(morph.target);
                if incident || owned.contains_key(m) {
                    // Driver owns every morphism in a loop plan's cones or
                    // incident to an SCC.
                    continue;
                }
                if ctx.gated.contains_key(m) {
                    // plan-s39: guard-arm work fires only from its Phi, and
                    // only when the condition picks that arm.
                    continue;
                }
                eval_morphism(&mut ctx, m, budget)?;
            }
        }
    }

    if want.is_some() {
        return Ok((RValue::Unit, ctx.env));
    }
    // Return env[output], or Unit if the output ty is Unit and unwritten (§7).
    match ctx.env.get(fd.output) {
        Some(v) => Ok((v.clone(), SecondaryMap::new())),
        None => {
            if *ctx.ty_of(fd.output) == Ty::Unit {
                Ok((RValue::Unit, SecondaryMap::new()))
            } else {
                // I-RET guarantees a writer fired for non-Unit returns.
                unreachable!("sealed graph: non-Unit return is always written");
            }
        }
    }
}

/// The set of objects in any loop SCC of `f` (for the incidence test).
pub(crate) fn build_in_scc(ir: &CategoryIr, f: FuncId) -> SecondaryMap<ObjectId, ()> {
    let mut set: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
    for scc in ir.loop_structure(f) {
        for o in scc.objects {
            set.insert(o, ());
        }
    }
    set
}

/// Evaluate one morphism: read `env[source]`, apply `op`, write `env[target]`.
/// Decrements `budget` (interp DESIGN §2, §3).
pub(crate) fn eval_morphism(
    ctx: &mut EvalCtx,
    m: MorphismId,
    budget: &mut u64,
) -> Result<(), Abort> {
    ctx.spend(budget)?;
    let morph = ctx.morph(m);
    let source = morph.source;
    let target = morph.target;
    let op = morph.op;

    match op {
        Operation::Pair { slot, arity } => {
            stage_pair(ctx, source, target, slot, arity);
            Ok(())
        }
        Operation::Proj { index } => {
            let v = component(read(ctx, source), index).clone();
            write(ctx, target, v);
            Ok(())
        }
        Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
            let weak = if op == Operation::Mul && WEAKEST.with(std::cell::Cell::get) {
                let p = ctx.ir.precision(m);
                match (p.half(), p.admits_tf32()) {
                    (Some(n), _) => Weak::Narrow(n),
                    (None, true) => Weak::Tf32,
                    (None, false) => Weak::None,
                }
            } else {
                Weak::None
            };
            let v = arith(ctx, m, source, op, weak)?;
            write(ctx, target, v);
            Ok(())
        }
        Operation::Neg => {
            let x = read(ctx, source);
            let v = match (neg(x), x) {
                (RValue::Scalar(y), RValue::Tracked(..)) => crate::bound::neg(x, y),
                (v, _) => v,
            };
            write(ctx, target, v);
            Ok(())
        }
        Operation::Eq
        | Operation::Neq
        | Operation::Lt
        | Operation::Gt
        | Operation::Le
        | Operation::Ge => {
            let v = compare(ctx, source, op);
            write(ctx, target, v);
            Ok(())
        }
        Operation::And | Operation::Or => {
            let v = logic(ctx, source, op);
            write(ctx, target, v);
            Ok(())
        }
        Operation::Not => {
            let v = match read(ctx, source) {
                RValue::Scalar(Value::Bool(b)) => RValue::Scalar(Value::Bool(!b)),
                _ => unreachable!("Not on non-bool"),
            };
            write(ctx, target, v);
            Ok(())
        }
        Operation::Widen => {
            let x = read(ctx, source);
            let v = match (scalar(x), ctx.ty_of(target)) {
                (Value::I32(x), Ty::Int { bits: 64, .. }) => Value::I64(*x as i64),
                (Value::I32(x), Ty::Float { bits: 32 }) => Value::F32(*x as f32),
                (Value::I32(x), Ty::Float { bits: 64 }) => Value::F64(*x as f64),
                (Value::F32(x), Ty::Float { bits: 64 }) => Value::F64(*x as f64),
                // ADR-0043: a narrow value widens exactly.
                (Value::Narrow(n, b), Ty::Float { bits: 32 }) => {
                    Value::F32(mapal_ir::narrow::widen(*n, *b))
                }
                (Value::Narrow(n, b), Ty::Float { bits: 64 }) => {
                    Value::F64(f64::from(mapal_ir::narrow::widen(*n, *b)))
                }
                _ => unreachable!("invalid Widen pair passed validation"),
            };
            let v = crate::bound::widen(x, v);
            write(ctx, target, v);
            Ok(())
        }
        Operation::Narrow => {
            // ADR-0043: one rounding to nearest even, from the operand's value.
            let Ty::Narrow(n) = *ctx.ty_of(target) else {
                unreachable!("Narrow into a non-narrow ty passed validation")
            };
            let xr = read(ctx, source);
            let x = match scalar(xr) {
                Value::F32(x) => f64::from(*x),
                _ => unreachable!("Narrow of a non-f32 passed validation"),
            };
            let v = Value::Narrow(n, mapal_ir::narrow::round(n, x));
            let v = if crate::bound::on() {
                crate::bound::narrow(xr, n, v)
            } else {
                RValue::Scalar(v)
            };
            write(ctx, target, v);
            Ok(())
        }
        Operation::Math(f) => {
            // ADR-0042 / ADR-0046: the builtin's reference (`mapal_ir::math`),
            // bit for bit.
            let xr = read(ctx, source);
            let (v, bits) = match scalar(xr) {
                Value::F32(x) => (Value::F32(mapal_ir::math::eval_f32(f, *x)), 32),
                Value::F64(x) => (Value::F64(mapal_ir::math::eval_f64(f, *x)), 64),
                _ => unreachable!("Math on a non-float passed validation"),
            };
            let v = if crate::bound::on() {
                crate::bound::math(f, bits, xr, v)
            } else {
                RValue::Scalar(v)
            };
            write(ctx, target, v);
            Ok(())
        }
        Operation::Phi => {
            if let Some(site) = ctx.guards.get(m).cloned() {
                // plan-s39: the condition picks the arm; only that arm's work
                // fires. The condition is staged at slot 2 (its Pair edge is
                // unconditional and precedes the Phi in topo); the unchosen
                // slot never fills, so the triple never finalizes into env —
                // read components from the staging buffer.
                let cond = match ctx.staging.get(source).and_then(|b| b[2].clone()) {
                    Some(RValue::Scalar(Value::Bool(b))) => b,
                    _ => unreachable!("Phi condition staged before the Phi"),
                };
                let arm = if cond { &site.on_true } else { &site.on_false };
                for &g in &arm.own {
                    let gm = ctx.morph(g);
                    if gm.op == Operation::LoopEnter {
                        // plan-s40: the handle stands for its whole loop unit —
                        // the driver fires the machinery and cones.
                        crate::loops::run_loop(ctx, gm.target, budget)?;
                    } else {
                        eval_morphism(ctx, g, budget)?;
                    }
                }
                let slot = if cond { 0 } else { 1 };
                let chosen = ctx
                    .staging
                    .get(source)
                    .and_then(|b| b[slot].clone())
                    .unwrap_or_else(|| unreachable!("chosen arm staged its slot"));
                write(ctx, target, chosen);
                return Ok(());
            }
            // Hand-built (non-builder) triple shape: strict select over the
            // finalized product — both arms were computed in the flat walk.
            let src = read(ctx, source).clone();
            let cond = match component(&src, 2) {
                RValue::Scalar(Value::Bool(b)) => *b,
                _ => unreachable!("Phi selector is not bool"),
            };
            let chosen = component(&src, if cond { 0 } else { 1 }).clone();
            write(ctx, target, chosen);
            Ok(())
        }
        Operation::Call(g) => {
            let arg = read(ctx, source).clone();
            let v = eval_fn(ctx.ir, g, arg, budget, ctx.orders)?;
            write(ctx, target, v);
            Ok(())
        }
        Operation::Map { body, captures } => {
            let k = captures as usize;
            let src = read(ctx, source).clone();
            // ADR-0027: the source is (c₁…cₖ, array); captures broadcast to
            // every body call (read-at-position — the value as of the map site).
            let (caps, arr) = if k == 0 {
                (Vec::new(), src)
            } else {
                let caps: Vec<RValue> = (0..k as u32).map(|i| component(&src, i).clone()).collect();
                (caps, component(&src, k as u32).clone())
            };
            let elems = match arr {
                RValue::Array(es) => es,
                _ => unreachable!("Map on non-array"),
            };
            let mut out = Vec::with_capacity(elems.len());
            for e in elems.iter() {
                let arg = if k == 0 {
                    e.clone()
                } else {
                    let mut v = caps.clone();
                    v.push(e.clone());
                    RValue::Tuple(v)
                };
                out.push(eval_fn(ctx.ir, body, arg, budget, ctx.orders)?);
            }
            write(ctx, target, RValue::array(out));
            Ok(())
        }
        Operation::Fold { body, captures } => {
            // (Acc, Array): left fold. ADR-0027: (c₁…cₖ, Acc, Array); the body
            // gets (c₁…cₖ, acc, e) per step.
            let k = captures as usize;
            let src = read(ctx, source).clone();
            let (caps, acc, elems) = if k == 0 {
                let acc = component(&src, 0).clone();
                let elems = match component(&src, 1).clone() {
                    RValue::Array(es) => es,
                    _ => unreachable!("Fold's second component is not an array"),
                };
                (Vec::new(), acc, elems)
            } else {
                let caps: Vec<RValue> = (0..k as u32).map(|i| component(&src, i).clone()).collect();
                let acc = component(&src, k as u32).clone();
                let elems = match component(&src, k as u32 + 1).clone() {
                    RValue::Array(es) => es,
                    _ => unreachable!("Fold's array component is not an array"),
                };
                (caps, acc, elems)
            };
            // ADR-0040 D3: a fold the compiled program split into `S`
            // contiguous partitions — partition 0 from the seed, the others
            // from 0 — added in partition order. The order is the compiler's
            // (`Emission::orders`), never derived here.
            let parts = ctx
                .orders
                .iter()
                .find(|&&(f, fm, _)| f == ctx.f && fm == m)
                .map(|&(_, _, s)| s as usize)
                .filter(|&s| s > 1 && elems.len() % s == 0)
                .unwrap_or(1);
            let kc = elems.len() / parts;
            let mut total: Option<RValue> = None;
            for z in 0..parts {
                let mut acc = if z == 0 { acc.clone() } else { zero_of(&acc) };
                for e in &elems[z * kc..(z + 1) * kc] {
                    let pair = if k == 0 {
                        RValue::Tuple(vec![acc, e.clone()])
                    } else {
                        let mut v = caps.clone();
                        v.push(acc);
                        v.push(e.clone());
                        RValue::Tuple(v)
                    };
                    acc = eval_fn(ctx.ir, body, pair, budget, ctx.orders)?;
                }
                total = Some(match total {
                    None => acc,
                    Some(t) => add_floats(&t, &acc),
                });
            }
            let mut total = total.expect("one partition at least");
            if crate::bound::on() {
                total = crate::bound::fold_end(ctx.ir, body, elems.len(), total);
            }
            write(ctx, target, total);
            Ok(())
        }
        Operation::Index => {
            let v = index(ctx, source)?;
            write(ctx, target, v);
            Ok(())
        }
        Operation::Update => {
            let v = update(ctx, source)?;
            write(ctx, target, v);
            Ok(())
        }
        Operation::Zip => {
            // src is the internal 2-tuple (Array A, Array B); pair elementwise
            // (ADR-0018 denotation: [(a[0],b[0]), …, (a[n-1],b[n-1])]). Sizes are
            // equal by ir typing, so `zip` consumes both fully.
            let src = read(ctx, source).clone();
            let a = as_array(component(&src, 0));
            let b = as_array(component(&src, 1));
            let out = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| RValue::Tuple(vec![x.clone(), y.clone()]))
                .collect();
            write(ctx, target, RValue::array(out));
            Ok(())
        }
        Operation::Enumerate => {
            // (ADR-0018 denotation: [(0 as i32, a[0]), …, (n-1 as i32, a[n-1])]).
            // The index is pinned i32; ir guarantees n ≤ i32::MAX so the cast is exact.
            let arr = read(ctx, source).clone();
            let elems = match arr {
                RValue::Array(es) => es,
                _ => unreachable!("Enumerate on non-array"),
            };
            let out = elems
                .iter()
                .enumerate()
                .map(|(i, x)| RValue::Tuple(vec![RValue::Scalar(Value::I32(i as i32)), x.clone()]))
                .collect();
            write(ctx, target, RValue::array(out));
            Ok(())
        }
        Operation::Iota => {
            // (ADR-0029 denotation: [0, 1, …, n-1] as i32.) The count is the
            // constant source (validate ties it to the target's static size).
            let n = match read(ctx, source) {
                RValue::Scalar(Value::I32(v)) => *v as usize,
                RValue::Scalar(Value::I64(v)) => *v as usize,
                RValue::Scalar(Value::U8(v)) => *v as usize,
                _ => unreachable!("Iota on a non-integer count"),
            };
            // The elements are the extent's index type (`i64` past i32::MAX).
            let wide = n as u64 > i32::MAX as u64 + 1;
            let out = (0..n)
                .map(|i| {
                    RValue::Scalar(if wide {
                        Value::I64(i as i64)
                    } else {
                        Value::I32(i as i32)
                    })
                })
                .collect();
            write(ctx, target, RValue::array(out));
            Ok(())
        }
        Operation::Fill => {
            // (ADR-0029 denotation: [x; n] — the internal (x, count) pair, as Zip.)
            let src = read(ctx, source).clone();
            let v = component(&src, 0).clone();
            let n = match component(&src, 1) {
                RValue::Scalar(Value::I32(c)) => *c as usize,
                RValue::Scalar(Value::I64(c)) => *c as usize,
                RValue::Scalar(Value::U8(c)) => *c as usize,
                _ => unreachable!("Fill on non-integer count"),
            };
            write(ctx, target, RValue::array(vec![v; n]));
            Ok(())
        }
        Operation::Print { newline } => {
            let v = print_op(ctx, source, newline);
            write(ctx, target, v);
            Ok(())
        }
        Operation::TimeMs => {
            // `IoToken → (IoToken, f64)` (plan-time-builtin): milliseconds from
            // a monotonic clock against one process-lifetime epoch. The token
            // threads through unchanged as the pair's first component.
            let tok = read(ctx, source).clone();
            debug_assert!(matches!(tok, RValue::Token(_)), "TimeMs source not a token");
            let ms = time_epoch().elapsed().as_secs_f64() * 1000.0;
            write(
                ctx,
                target,
                RValue::Tuple(vec![tok, RValue::Scalar(Value::F64(ms))]),
            );
            Ok(())
        }
        Operation::Read => {
            // `(IoToken, Str) → (IoToken, [T; n])` (ADR-0045): the file's
            // `n · |T|` little-endian bytes, resolved against the graph's
            // recorded directory; any other length, or no file, fails the run.
            let src = read(ctx, source).clone();
            let tok = component(&src, 0).clone();
            let RValue::Scalar(Value::Str(path)) = component(&src, 1) else {
                unreachable!("Read path is not a Str")
            };
            let arr_ty = ctx
                .ty_of(target)
                .component_ty(1)
                .cloned()
                .expect("Read target is (token, array)");
            let bytes = mapal_ir::read::file_bytes(&arr_ty).expect("a readable value");
            match mapal_ir::read::load(path, ctx.ir.data_dir(), bytes) {
                Ok(b) => {
                    let value = decode_value(&b, &arr_ty, &mut 0);
                    write(ctx, target, RValue::Tuple(vec![tok, value]));
                    Ok(())
                }
                Err(msg) => {
                    crate::value::READ_FAILURE.with(|r| *r.borrow_mut() = Some(msg));
                    Err(Abort::ReadFailed)
                }
            }
        }
        Operation::Output => {
            let v = read(ctx, source).clone();
            write(ctx, target, v);
            Ok(())
        }
        Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {
            unreachable!("loop ops are driver-owned, never eval_morphism'd")
        }
    }
}

mod helpers;
#[cfg(test)]
mod tests;

pub(crate) use self::helpers::*;
