//! Step emission: how each script step is built, and the loop and lift builders.

use super::*;

// --- step emission --------------------------------------------------------

pub(super) fn emit_steps(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ctx: &Ctx,
    pool: &mut Pool,
    steps: &[Step],
    collections: bool,
) {
    for step in steps {
        emit_step(fb, ctx, pool, step, collections);
    }
}

pub(super) fn emit_step(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ctx: &Ctx,
    pool: &mut Pool,
    step: &Step,
    collections: bool,
) {
    use Operation::*;
    match *step {
        Step::ConstI32(k) => pool.i32s.push(fb.constant(Value::I32(k), L()).unwrap()),
        Step::ConstBool(v) => pool.bools.push(fb.constant(Value::Bool(v), L()).unwrap()),
        Step::Bin { op, a, b } => {
            let (Some(x), Some(mut y)) = (pick(&pool.i32s, a), pick(&pool.i32s, b)) else {
                return;
            };
            let op = [Add, Sub, Mul, Div, Mod][op as usize % 5];
            if ctx.trap_free && matches!(op, Div | Mod) {
                // Force a non-zero constant divisor (no runtime trap).
                y = fb.constant(Value::I32((b as i32 % 7) + 1), L()).unwrap();
            }
            let r = fb.binop(op, x, y, Dest::Fresh(None), L()).unwrap();
            pool.i32s.push(r);
        }
        Step::Neg { a } => {
            if let Some(x) = pick(&pool.i32s, a) {
                pool.i32s
                    .push(fb.unop(Neg, x, Dest::Fresh(None), L()).unwrap());
            }
        }
        Step::Widen { edge, a } => match edge % 4 {
            0 => {
                if let Some(x) = pick(&pool.i32s, a) {
                    pool.i64s
                        .push(fb.widen(x, Ty::i64(), Dest::Fresh(None), L()).unwrap());
                }
            }
            1 => {
                if let Some(x) = pick(&pool.i32s, a) {
                    pool.f32s
                        .push(fb.widen(x, Ty::f32(), Dest::Fresh(None), L()).unwrap());
                }
            }
            2 => {
                if let Some(x) = pick(&pool.i32s, a) {
                    pool.f64s
                        .push(fb.widen(x, Ty::f64(), Dest::Fresh(None), L()).unwrap());
                }
            }
            _ => {
                let x = match pick(&pool.f32s, a) {
                    Some(x) => x,
                    None => {
                        let Some(i) = pick(&pool.i32s, a) else {
                            return;
                        };
                        let x = fb.widen(i, Ty::f32(), Dest::Fresh(None), L()).unwrap();
                        pool.f32s.push(x);
                        x
                    }
                };
                pool.f64s
                    .push(fb.widen(x, Ty::f64(), Dest::Fresh(None), L()).unwrap());
            }
        },
        Step::Iota => {
            let c = fb.constant(Value::I32(ARR as i32), L()).unwrap();
            pool.arrs.push(fb.iota(c, Dest::Fresh(None), L()).unwrap());
        }
        Step::Fill { a } => {
            let Some(x) = pick(&pool.i32s, a) else {
                return;
            };
            let c = fb.constant(Value::I32(ARR as i32), L()).unwrap();
            pool.arrs
                .push(fb.fill(x, c, Dest::Fresh(None), L()).unwrap());
        }
        Step::Cmp { op, a, b } => {
            let (Some(x), Some(y)) = (pick(&pool.i32s, a), pick(&pool.i32s, b)) else {
                return;
            };
            let op = [Eq, Neq, Lt, Gt, Le, Ge][op as usize % 6];
            pool.bools
                .push(fb.binop(op, x, y, Dest::Fresh(None), L()).unwrap());
        }
        Step::Not { a } => {
            if let Some(x) = pick(&pool.bools, a) {
                pool.bools
                    .push(fb.unop(Not, x, Dest::Fresh(None), L()).unwrap());
            }
        }
        Step::Logic { or, a, b } => {
            let (Some(x), Some(y)) = (pick(&pool.bools, a), pick(&pool.bools, b)) else {
                return;
            };
            let op = if or { Or } else { And };
            pool.bools
                .push(fb.binop(op, x, y, Dest::Fresh(None), L()).unwrap());
        }
        Step::Phi { t, e, c } => {
            let (Some(x), Some(y), Some(cond)) = (
                pick(&pool.i32s, t),
                pick(&pool.i32s, e),
                pick(&pool.bools, c),
            ) else {
                return;
            };
            pool.i32s
                .push(fb.phi(x, y, cond, Dest::Fresh(None), L()).unwrap());
        }
        Step::PhiTrapArm {
            a,
            b,
            c,
            on_true,
            modulo,
        } => {
            let (Some(x), Some(mut y), Some(cond)) = (
                pick(&pool.i32s, a),
                pick(&pool.i32s, b),
                pick(&pool.bools, c),
            ) else {
                return;
            };
            let op = if modulo { Mod } else { Div };
            if ctx.trap_free {
                // Same rule as `Step::Bin`: a const non-zero divisor.
                y = fb.constant(Value::I32((b as i32 % 7) + 1), L()).unwrap();
            }
            let risky = fb.binop(op, x, y, Dest::Fresh(None), L()).unwrap();
            // NOT pushed to the pool: the whole point is that this object's
            // only consumer is the Phi arm, so the arm exclusively owns it.
            let other = pick(&pool.i32s, a.wrapping_add(1)).unwrap_or(x);
            let (t, e) = if on_true {
                (risky, other)
            } else {
                (other, risky)
            };
            pool.i32s
                .push(fb.phi(t, e, cond, Dest::Fresh(None), L()).unwrap());
        }
        Step::PackProj { a, b, snd } => {
            let (Some(x), Some(y)) = (pick(&pool.i32s, a), pick(&pool.i32s, b)) else {
                return;
            };
            let p = fb.pack(&[x, y], Dest::Fresh(None), L()).unwrap();
            let idx = if snd { 1 } else { 0 };
            pool.i32s
                .push(fb.proj(p, idx, Dest::Fresh(None), L()).unwrap());
        }
        Step::MakeArray { a, b, c } if collections => {
            let (Some(x), Some(y), Some(z)) = (
                pick(&pool.i32s, a),
                pick(&pool.i32s, b),
                pick(&pool.i32s, c),
            ) else {
                return;
            };
            pool.arrs
                .push(fb.pack_array(&[x, y, z], Dest::Fresh(None), L()).unwrap());
        }
        Step::Index { arr, idx } if collections => {
            let Some(a) = pick(&pool.arrs, arr) else {
                return;
            };
            let i = if ctx.trap_free {
                fb.constant(Value::I32(idx as i32 % ARR as i32), L())
                    .unwrap()
            } else {
                match pick(&pool.i32s, idx) {
                    Some(x) => x,
                    None => return,
                }
            };
            pool.i32s
                .push(fb.index(a, i, Dest::Fresh(None), L()).unwrap());
        }
        Step::Update { arr, idx, val } if collections => {
            let (Some(a), Some(v)) = (pick(&pool.arrs, arr), pick(&pool.i32s, val)) else {
                return;
            };
            // trap_free: index literal-and-in-bounds by construction; default:
            // an arbitrary pool feeder (sometimes OOB — the exit-101 trap path).
            let i = if ctx.trap_free {
                fb.constant(Value::I32(idx as i32 % ARR as i32), L())
                    .unwrap()
            } else {
                match pick(&pool.i32s, idx) {
                    Some(x) => x,
                    None => return,
                }
            };
            pool.arrs
                .push(fb.update(a, i, v, Dest::Fresh(None), L()).unwrap());
        }
        Step::MapArr { arr, body } if collections => {
            let (Some(a), Some(bd)) = (pick(&pool.arrs, arr), pick(&ctx.map_bodies, body)) else {
                return;
            };
            pool.arrs
                .push(fb.map(bd, a, Dest::Fresh(None), L()).unwrap());
        }
        Step::FoldArr { arr, seed, body } if collections => {
            let (Some(a), Some(s), Some(bd)) = (
                pick(&pool.arrs, arr),
                pick(&pool.i32s, seed),
                pick(&ctx.fold_bodies, body),
            ) else {
                return;
            };
            let pair = fb.pack(&[s, a], Dest::Fresh(None), L()).unwrap();
            pool.i32s
                .push(fb.fold(bd, pair, Dest::Fresh(None), L()).unwrap());
        }
        Step::Zip { a, b } if collections => {
            let (Some(x), Some(y)) = (pick(&pool.arrs, a), pick(&pool.arrs, b)) else {
                return;
            };
            let pairs = fb.zip(x, y, Dest::Fresh(None), L()).unwrap();
            // Reduce the pair-array back to [i32;3] via the utility body.
            pool.arrs
                .push(fb.map(ctx.pair_sum, pairs, Dest::Fresh(None), L()).unwrap());
        }
        Step::Enumerate { arr } if collections => {
            let Some(a) = pick(&pool.arrs, arr) else {
                return;
            };
            let pairs = fb.enumerate(a, Dest::Fresh(None), L()).unwrap();
            pool.arrs
                .push(fb.map(ctx.pair_sum, pairs, Dest::Fresh(None), L()).unwrap());
        }
        Step::Call { a, helper } if collections => {
            let (Some(x), Some(g)) = (pick(&pool.i32s, a), pick(&ctx.helpers, helper)) else {
                return;
            };
            pool.i32s
                .push(fb.call(g, x, Dest::Fresh(None), L()).unwrap());
        }
        Step::Loop { k } if collections && pool.loops_used < MAX_LOOPS => {
            pool.loops_used += 1;
            pool.i32s.push(build_loop(fb, (k % 65) as i32));
        }
        Step::LiftFold { k, seed, cap } if collections && pool.loops_used < MAX_LOOPS => {
            let Some(seed) = pick(&pool.i32s, seed) else {
                return;
            };
            let cap = fb.constant(Value::I32(i32::from(cap)), L()).unwrap();
            pool.loops_used += 1;
            pool.i32s
                .push(build_lift_fold(fb, (k % ARR as u8 + 1) as i32, seed, cap));
        }
        Step::LiftMap { arr, cap } if collections && pool.loops_used < MAX_LOOPS => {
            let cap = fb.constant(Value::I32(i32::from(cap)), L()).unwrap();
            let init = pick(&pool.arrs, arr)
                .unwrap_or_else(|| fb.pack_array(&[cap; ARR], Dest::Fresh(None), L()).unwrap());
            pool.loops_used += 1;
            pool.arrs.push(build_lift_map(fb, init, cap));
        }
        Step::MapCapScalar { arr, cap, body } if collections => {
            let (Some(a), Some(c), Some(bd)) = (
                pick(&pool.arrs, arr),
                pick(&pool.i32s, cap),
                pick(&ctx.map_cap_bodies, body),
            ) else {
                return;
            };
            pool.arrs.push(
                fb.map_captured(bd, &[c], a, Dest::Fresh(None), L())
                    .unwrap(),
            );
        }
        Step::MapCapArray { arr, cap, body } if collections => {
            let (Some(a), Some(c), Some(bd)) = (
                pick(&pool.arrs, arr),
                pick(&pool.arrs, cap),
                pick(&ctx.map_acap_bodies, body),
            ) else {
                return;
            };
            pool.arrs.push(
                fb.map_captured(bd, &[c], a, Dest::Fresh(None), L())
                    .unwrap(),
            );
        }
        Step::FoldCapScalar {
            arr,
            seed,
            cap,
            body,
        } if collections => {
            let (Some(a), Some(s), Some(c), Some(bd)) = (
                pick(&pool.arrs, arr),
                pick(&pool.i32s, seed),
                pick(&pool.i32s, cap),
                pick(&ctx.fold_cap_bodies, body),
            ) else {
                return;
            };
            pool.i32s.push(
                fb.fold_captured(bd, &[c], s, a, Dest::Fresh(None), L())
                    .unwrap(),
            );
        }
        Step::MapNestFold { arr, cap, body } if collections => {
            let (Some(a), Some(c), Some(bd)) = (
                pick(&pool.arrs, arr),
                pick(&pool.arrs, cap),
                pick(&ctx.nest_bodies, body),
            ) else {
                return;
            };
            pool.arrs.push(
                fb.map_captured(bd, &[c], a, Dest::Fresh(None), L())
                    .unwrap(),
            );
        }
        Step::LoopCapMap { k, arr, body } if collections && pool.loops_used < MAX_LOOPS => {
            pool.loops_used += 1;
            let exit = build_loop(fb, (k % 65) as i32);
            pool.i32s.push(exit);
            // The loop's exit value is the map's capture — read-at-position.
            let (Some(a), Some(bd)) = (pick(&pool.arrs, arr), pick(&ctx.map_cap_bodies, body))
            else {
                return;
            };
            pool.arrs.push(
                fb.map_captured(bd, &[exit], a, Dest::Fresh(None), L())
                    .unwrap(),
            );
        }
        Step::CarryLoop {
            k,
            two,
            nest,
            init,
            ref ops,
            next,
            exit,
        } if collections && pool.loops_used < MAX_LOOPS => {
            pool.loops_used += 1;
            build_carry_loop(fb, ctx, pool, k, two, nest, init, ops, next, exit);
        }
        // Collection/loop steps in a scalar body: skipped.
        _ => {}
    }
}

/// [`Step::CarryLoop`]'s graph.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_carry_loop(
    fb: &mut mapal_ir::FnBuilder<'_>,
    ctx: &Ctx,
    pool: &mut Pool,
    k: u8,
    two: bool,
    nest: bool,
    init: [u8; 2],
    ops: &[[u8; 3]],
    next: [u8; 2],
    exit: u8,
) {
    use Operation::*;
    let c = |fb: &mut mapal_ir::FnBuilder<'_>, v: i32| fb.constant(Value::I32(v), L()).unwrap();
    let fresh = Dest::Fresh(None);
    let n = 1 + two as usize;
    let zero = c(fb, 0);
    let mut inits = Vec::new();
    for &sel in init.iter().take(n) {
        let a = match pick(&pool.arrs, sel) {
            Some(a) => a,
            None => {
                let ar = c(fb, ARR as i32);
                fb.iota(ar, fresh.clone(), L()).unwrap()
            }
        };
        inits.push(a);
    }
    // The invariants the body may read feed the init (Core starts a loop when
    // its init is ready — the `build_lift_fold` rule).
    let outer_arrs: Vec<_> = pool.arrs.iter().rev().take(3).copied().collect();
    let outer_i32s: Vec<_> = pool.i32s.iter().rev().take(3).copied().collect();
    let mut dep = zero;
    for &a in &outer_arrs {
        let e = fb.index(a, zero, fresh.clone(), L()).unwrap();
        let e0 = fb.binop(Mul, e, zero, fresh.clone(), L()).unwrap();
        dep = fb.binop(Add, dep, e0, fresh.clone(), L()).unwrap();
    }
    for &v in &outer_i32s {
        let e0 = fb.binop(Mul, v, zero, fresh.clone(), L()).unwrap();
        dep = fb.binop(Add, dep, e0, fresh.clone(), L()).unwrap();
    }
    let acc0 = pick(&pool.i32s, init[0]).unwrap_or(zero);
    let pack_state = |fb: &mut mapal_ir::FnBuilder<'_>, i, xs: &[mapal_ir::ObjectId], acc| {
        if nest && xs.len() == 2 {
            let sub = fb.pack(&[xs[1], acc], Dest::Fresh(None), L()).unwrap();
            fb.pack(&[i, xs[0], sub], Dest::Fresh(None), L()).unwrap()
        } else {
            let mut parts = vec![i];
            parts.extend_from_slice(xs);
            parts.push(acc);
            fb.pack(&parts, Dest::Fresh(None), L()).unwrap()
        }
    };
    let init_state = pack_state(fb, dep, &inits, acc0);
    let lh = fb.begin_loop(init_state, L()).unwrap();
    let st = fb.merge_of(&lh);
    let i = fb.proj(st, 0, fresh.clone(), L()).unwrap();
    let mut views = vec![fb.proj(st, 1, fresh.clone(), L()).unwrap()];
    let acc = if nest && n == 2 {
        let sub = fb.proj(st, 2, fresh.clone(), L()).unwrap();
        views.push(fb.proj(sub, 0, fresh.clone(), L()).unwrap());
        fb.proj(sub, 1, fresh.clone(), L()).unwrap()
    } else {
        if n == 2 {
            views.push(fb.proj(st, 2, fresh.clone(), L()).unwrap());
        }
        fb.proj(st, n as u32 + 1, fresh.clone(), L()).unwrap()
    };
    let bound = c(fb, i32::from(k % 9));
    let cond = fb.binop(Lt, i, bound, fresh.clone(), L()).unwrap();
    let mut lp = Pool {
        arrs: views
            .iter()
            .copied()
            .chain(outer_arrs.iter().copied())
            .collect(),
        i32s: [i, acc]
            .into_iter()
            .chain(outer_i32s.iter().copied())
            .collect(),
        ..Default::default()
    };
    // What the body reads; every value it makes and nothing reads joins the
    // accumulator, so the whole body is live (a loop evaluates only what
    // feeds its next state or its exit).
    let used = std::cell::RefCell::new(std::collections::HashSet::new());
    let take = |v: &[mapal_ir::ObjectId], sel: u8| {
        let o = pick(v, sel).unwrap();
        used.borrow_mut().insert(o);
        o
    };
    let idx = |fb: &mut mapal_ir::FnBuilder<'_>, lp: &Pool, sel: u8| {
        if ctx.trap_free || sel % 2 == 0 {
            fb.constant(Value::I32(i32::from(sel / 2) % ARR as i32), L())
                .unwrap()
        } else {
            take(&lp.i32s, sel)
        }
    };
    for &[op, a, b] in ops {
        // The array an op transforms is one the trip makes or a view of the
        // state — never only invariants, so the op runs in the loop (Core
        // starts a loop when its init is ready: a body-only invariant would
        // be pending). Invariants join as second operands.
        let variant: Vec<_> = lp
            .arrs
            .iter()
            .copied()
            .filter(|o| !outer_arrs.contains(o))
            .collect();
        let x = take(&variant, a);
        match op % 8 {
            // An op with no body to call falls back to one without (so `x`
            // is always read: every value the body makes is live).
            0 => {
                let bd = pick(&ctx.map_bodies, b).unwrap_or(ctx.pair_sum);
                let src = if bd == ctx.pair_sum {
                    fb.enumerate(x, fresh.clone(), L()).unwrap()
                } else {
                    x
                };
                lp.arrs.push(fb.map(bd, src, fresh.clone(), L()).unwrap());
            }
            1 => {
                let at = idx(fb, &lp, b);
                let v = take(&lp.i32s, op / 8);
                lp.arrs
                    .push(fb.update(x, at, v, fresh.clone(), L()).unwrap());
            }
            2 => {
                let at = idx(fb, &lp, b);
                lp.i32s.push(fb.index(x, at, fresh.clone(), L()).unwrap());
            }
            3 => match pick(&ctx.fold_bodies, b) {
                Some(bd) => {
                    let seed = take(&lp.i32s, op / 8);
                    let pair = fb.pack(&[seed, x], fresh.clone(), L()).unwrap();
                    lp.i32s.push(fb.fold(bd, pair, fresh.clone(), L()).unwrap());
                }
                None => {
                    let at = idx(fb, &lp, b);
                    lp.i32s.push(fb.index(x, at, fresh.clone(), L()).unwrap());
                }
            },
            4 => {
                // A guard choosing between an arm-owned fresh array and another.
                let arm = match pick(&ctx.map_bodies, b) {
                    Some(bd) => fb.map(bd, x, fresh.clone(), L()).unwrap(),
                    None if b % 3 == 0 => {
                        let e = fb.enumerate(x, fresh.clone(), L()).unwrap();
                        fb.map(ctx.pair_sum, e, fresh.clone(), L()).unwrap()
                    }
                    None => {
                        let at = c(fb, i32::from(b) % ARR as i32);
                        fb.update(x, at, i, fresh.clone(), L()).unwrap()
                    }
                };
                let other = take(&lp.arrs, op / 8);
                let lim = c(fb, i32::from(b % 4));
                let g = fb.binop(Lt, i, lim, fresh.clone(), L()).unwrap();
                let (t, e) = if b % 2 == 0 {
                    (arm, other)
                } else {
                    (other, arm)
                };
                lp.arrs.push(fb.phi(t, e, g, fresh.clone(), L()).unwrap());
            }
            5 => {
                let y = take(&lp.arrs, b);
                let z = fb.zip(x, y, fresh.clone(), L()).unwrap();
                lp.arrs
                    .push(fb.map(ctx.pair_sum, z, fresh.clone(), L()).unwrap());
            }
            6 => {
                let cap = take(&lp.arrs, op / 8);
                lp.arrs.push(match pick(&ctx.map_acap_bodies, b) {
                    Some(bd) => fb.map_captured(bd, &[cap], x, fresh.clone(), L()).unwrap(),
                    None => {
                        let z = fb.zip(x, cap, fresh.clone(), L()).unwrap();
                        fb.map(ctx.pair_sum, z, fresh.clone(), L()).unwrap()
                    }
                });
            }
            _ => {
                let (a0, a1) = (idx(fb, &lp, b), idx(fb, &lp, b.wrapping_add(1)));
                let v = take(&lp.i32s, op / 8);
                let u = fb.update(x, a0, v, fresh.clone(), L()).unwrap();
                lp.arrs
                    .push(fb.update(u, a1, i, fresh.clone(), L()).unwrap());
            }
        }
    }
    let nexts: Vec<_> = (0..n).map(|j| take(&lp.arrs, next[j])).collect();
    let mut acc1 = acc;
    let made = |v: &[mapal_ir::ObjectId], outer: usize| v[outer..].to_vec();
    for v in made(&lp.i32s, 2 + outer_i32s.len()) {
        if !used.borrow().contains(&v) {
            acc1 = fb.binop(Add, acc1, v, fresh.clone(), L()).unwrap();
        }
    }
    for a in made(&lp.arrs, n + outer_arrs.len()) {
        if !used.borrow().contains(&a) {
            let e = fb.index(a, zero, fresh.clone(), L()).unwrap();
            acc1 = fb.binop(Add, acc1, e, fresh.clone(), L()).unwrap();
        }
    }
    if acc1 == acc {
        let one = c(fb, 1);
        acc1 = fb.binop(Add, acc, one, fresh.clone(), L()).unwrap();
    }
    let one = c(fb, 1);
    let i1 = fb.binop(Add, i, one, fresh.clone(), L()).unwrap();
    let state1 = pack_state(fb, i1, &nexts, acc1);
    fb.loop_back(&lh, state1, cond, L()).unwrap();
    let out = fb
        .loop_exit(&lh, views[exit as usize % n], cond, fresh.clone(), L())
        .unwrap();
    fb.end_loop(lh).unwrap();
    // Observe every cell of the exit array.
    let mut sum = zero;
    for j in 0..ARR as i32 {
        let at = c(fb, j);
        let e = fb.index(out, at, fresh.clone(), L()).unwrap();
        sum = fb.binop(Add, sum, e, fresh.clone(), L()).unwrap();
    }
    pool.arrs.push(out);
    pool.i32s.push(sum);
    pool.observe.push(sum);
}

/// A canonical bounded loop `i = 0; while i < k { i += 1 }; i` → `k` (`k ≤ 64`).
/// One merge / one back / one exit — the interp-M1 canonical quartet.
pub(super) fn build_loop(fb: &mut mapal_ir::FnBuilder<'_>, k: i32) -> mapal_ir::ObjectId {
    let zero = fb.constant(Value::I32(0), L()).unwrap();
    let kc = fb.constant(Value::I32(k), L()).unwrap();
    let lh = fb.begin_loop(zero, L()).unwrap();
    let i = fb.merge_of(&lh);
    let cond = fb
        .binop(Operation::Lt, i, kc, Dest::Fresh(None), L())
        .unwrap();
    let one = fb.constant(Value::I32(1), L()).unwrap();
    let next = fb
        .binop(Operation::Add, i, one, Dest::Fresh(None), L())
        .unwrap();
    fb.loop_back(&lh, next, cond, L()).unwrap();
    let exit = fb.loop_exit(&lh, i, cond, Dest::Fresh(None), L()).unwrap();
    fb.end_loop(lh).unwrap();
    exit
}

pub(super) fn build_lift_fold(
    fb: &mut mapal_ir::FnBuilder<'_>,
    k: i32,
    seed: mapal_ir::ObjectId,
    capture: mapal_ir::ObjectId,
) -> mapal_ir::ObjectId {
    let zero = fb.constant(Value::I32(0), L()).unwrap();
    // Make the invariant an explicit predecessor of LoopEnter. Core starts a
    // loop when its init is ready, so an otherwise body-only invariant could
    // still be pending when the first advance phase runs.
    let capture_zero = fb
        .binop(Operation::Mul, capture, zero, Dest::Fresh(None), L())
        .unwrap();
    let ready_seed = fb
        .binop(Operation::Add, seed, capture_zero, Dest::Fresh(None), L())
        .unwrap();
    let init = fb
        .pack(&[zero, ready_seed], Dest::Fresh(None), L())
        .unwrap();
    let lh = fb.begin_loop(init, L()).unwrap();
    let state = fb.merge_of(&lh);
    let counter = fb.proj(state, 0, Dest::Fresh(None), L()).unwrap();
    let acc = fb.proj(state, 1, Dest::Fresh(None), L()).unwrap();
    let bound = fb.constant(Value::I32(k), L()).unwrap();
    let cond = fb
        .binop(Operation::Lt, counter, bound, Dest::Fresh(None), L())
        .unwrap();
    let partial = fb
        .binop(Operation::Add, acc, counter, Dest::Fresh(None), L())
        .unwrap();
    let next_acc = fb
        .binop(Operation::Add, partial, capture, Dest::Fresh(None), L())
        .unwrap();
    let one = fb.constant(Value::I32(1), L()).unwrap();
    let next_counter = fb
        .binop(Operation::Add, counter, one, Dest::Fresh(None), L())
        .unwrap();
    let next = fb
        .pack(&[next_counter, next_acc], Dest::Fresh(None), L())
        .unwrap();
    fb.loop_back(&lh, next, cond, L()).unwrap();
    let exit = fb
        .loop_exit(&lh, acc, cond, Dest::Fresh(None), L())
        .unwrap();
    fb.end_loop(lh).unwrap();
    exit
}

pub(super) fn build_lift_map(
    fb: &mut mapal_ir::FnBuilder<'_>,
    init_array: mapal_ir::ObjectId,
    capture: mapal_ir::ObjectId,
) -> mapal_ir::ObjectId {
    let zero = fb.constant(Value::I32(0), L()).unwrap();
    // As above, sequence the invariant before LoopEnter through the collection
    // init. Coverage overwrites every cell, so this value is observationally
    // dead and is dropped by R-LM.
    let ready_array = fb
        .update(init_array, zero, capture, Dest::Fresh(None), L())
        .unwrap();
    let init = fb
        .pack(&[ready_array, zero], Dest::Fresh(None), L())
        .unwrap();
    let lh = fb.begin_loop(init, L()).unwrap();
    let state = fb.merge_of(&lh);
    let out = fb.proj(state, 0, Dest::Fresh(None), L()).unwrap();
    let counter = fb.proj(state, 1, Dest::Fresh(None), L()).unwrap();
    let bound = fb.constant(Value::I32(ARR as i32), L()).unwrap();
    let cond = fb
        .binop(Operation::Lt, counter, bound, Dest::Fresh(None), L())
        .unwrap();
    let value = fb
        .binop(Operation::Add, counter, capture, Dest::Fresh(None), L())
        .unwrap();
    let updated = fb
        .update(out, counter, value, Dest::Fresh(None), L())
        .unwrap();
    let one = fb.constant(Value::I32(1), L()).unwrap();
    let next_counter = fb
        .binop(Operation::Add, counter, one, Dest::Fresh(None), L())
        .unwrap();
    let next = fb
        .pack(&[updated, next_counter], Dest::Fresh(None), L())
        .unwrap();
    fb.loop_back(&lh, next, cond, L()).unwrap();
    let exit = fb
        .loop_exit(&lh, out, cond, Dest::Fresh(None), L())
        .unwrap();
    fb.end_loop(lh).unwrap();
    exit
}

pub(super) fn pair_ty() -> Ty {
    Ty::Tuple(vec![Ty::i32(), Ty::i32()])
}
