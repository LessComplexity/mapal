//! Strategies over the core-program script and `build`, which interprets a script into a sealed graph.

use super::*;

// --- strategies -----------------------------------------------------------

/// A small-int-biased `i32` constant strategy (DESIGN §6: ±100 + edge values).
pub(super) fn const_i32() -> impl Strategy<Value = i32> {
    prop_oneof![
        3 => -100i32..=100,
        1 => prop::sample::select(vec![0, 1, -1, 2, -2, i32::MIN, i32::MAX, 100, -100]),
    ]
}

/// A restricted step (scalar ops only) — for map/fold/helper bodies, which must
/// stay loop-free and collection-free (totality + fusion P3).
pub(super) fn scalar_step() -> impl Strategy<Value = Step> {
    prop_oneof![
        const_i32().prop_map(Step::ConstI32),
        any::<bool>().prop_map(Step::ConstBool),
        (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(op, a, b)| Step::Bin { op, a, b }),
        any::<u8>().prop_map(|a| Step::Neg { a }),
        (any::<u8>(), any::<u8>()).prop_map(|(edge, a)| Step::Widen { edge, a }),
        Just(Step::Iota),
        any::<u8>().prop_map(|a| Step::Fill { a }),
        (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(op, a, b)| Step::Cmp { op, a, b }),
        any::<u8>().prop_map(|a| Step::Not { a }),
        (any::<bool>(), any::<u8>(), any::<u8>()).prop_map(|(or, a, b)| Step::Logic { or, a, b }),
        (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(t, e, c)| Step::Phi { t, e, c }),
        (
            any::<u8>(),
            any::<u8>(),
            any::<u8>(),
            any::<bool>(),
            any::<bool>()
        )
            .prop_map(|(a, b, c, on_true, modulo)| Step::PhiTrapArm {
                a,
                b,
                c,
                on_true,
                modulo
            }),
        (any::<u8>(), any::<u8>(), any::<bool>()).prop_map(|(a, b, snd)| Step::PackProj {
            a,
            b,
            snd
        }),
    ]
}

/// A full step (adds collections + loops + ADR-0027 captures) — for `main` only.
pub(super) fn main_step() -> impl Strategy<Value = Step> {
    prop_oneof![
        6 => scalar_step(),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(a, b, c)| Step::MakeArray { a, b, c }),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(arr, idx)| Step::Index { arr, idx }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, idx, val)| Step::Update { arr, idx, val }),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(arr, body)| Step::MapArr { arr, body }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, seed, body)| Step::FoldArr { arr, seed, body }),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(a, b)| Step::Zip { a, b }),
        1 => any::<u8>().prop_map(|arr| Step::Enumerate { arr }),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(a, helper)| Step::Call { a, helper }),
        1 => any::<u8>().prop_map(|k| Step::Loop { k }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(k, seed, cap)| Step::LiftFold { k, seed, cap }),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(arr, cap)| Step::LiftMap { arr, cap }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, cap, body)| Step::MapCapScalar { arr, cap, body }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, cap, body)| Step::MapCapArray { arr, cap, body }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, seed, cap, body)| Step::FoldCapScalar { arr, seed, cap, body }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(arr, cap, body)| Step::MapNestFold { arr, cap, body }),
        1 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(k, arr, body)| Step::LoopCapMap { k, arr, body }),
    ]
}

pub(super) fn body_strategy() -> impl Strategy<Value = Vec<Step>> {
    prop::collection::vec(scalar_step(), 0..5)
}

/// A nest-body script: the inner fold body + the outer map body's own steps.
pub(super) fn nest_body_strategy() -> impl Strategy<Value = NestBody> {
    (body_strategy(), body_strategy()).prop_map(|(inner, outer)| NestBody { inner, outer })
}

/// S65: [`Step::CarryLoop`].
pub(super) fn carry_step() -> impl Strategy<Value = Step> {
    (
        any::<u8>(),
        any::<bool>(),
        any::<bool>(),
        prop::array::uniform2(any::<u8>()),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..7),
        prop::array::uniform2(any::<u8>()),
        any::<u8>(),
    )
        .prop_map(|(k, two, nest, init, ops, next, exit)| Step::CarryLoop {
            k,
            two,
            nest,
            init,
            ops,
            next,
            exit,
        })
}

/// The full program strategy for a given mode/openness (DESIGN §6).
pub fn prog_strategy(trap_free: bool, open: bool) -> impl Strategy<Value = Prog> {
    prog_strategy_with(trap_free, open, prop::collection::vec(main_step(), 0..12))
}

/// S65: closed programs whose `main` carries arrays through loops
/// ([`Step::CarryLoop`] among the ordinary steps, at least one loop).
pub fn carry_prog_strategy(trap_free: bool) -> impl Strategy<Value = Prog> {
    let main = (
        prop::collection::vec(main_step(), 0..6),
        carry_step(),
        prop::collection::vec(prop_oneof![2 => main_step(), 1 => carry_step()], 0..6),
    )
        .prop_map(|(mut a, l, b)| {
            a.push(l);
            a.extend(b);
            a
        });
    prog_strategy_with(trap_free, false, main)
}

pub(super) fn prog_strategy_with(
    trap_free: bool,
    open: bool,
    main: impl Strategy<Value = Vec<Step>>,
) -> impl Strategy<Value = Prog> {
    (
        prop::collection::vec(body_strategy(), 0..2), // helpers
        prop::collection::vec(body_strategy(), 0..3), // map bodies
        prop::collection::vec(body_strategy(), 0..3), // fold bodies
        main,                                         // main
        prop::collection::vec(any::<u8>(), 0..4),     // prints
        any::<bool>(),                                // effectful
        any::<u8>(),                                  // ret pick
        prop::collection::vec(const_i32(), 1..4),     // open args
        // ADR-0027 capture body pools.
        (
            prop::collection::vec(body_strategy(), 0..2), // map_cap_bodies
            prop::collection::vec(body_strategy(), 0..2), // map_acap_bodies
            prop::collection::vec(body_strategy(), 0..2), // fold_cap_bodies
            prop::collection::vec(nest_body_strategy(), 0..2), // nest_bodies
        ),
    )
        .prop_map(
            move |(helpers, map_bodies, fold_bodies, main, prints, effectful, ret, args, caps)| {
                let (map_cap_bodies, map_acap_bodies, fold_cap_bodies, nest_bodies) = caps;
                Prog {
                    trap_free,
                    open,
                    effectful: !open && effectful,
                    helpers,
                    map_bodies,
                    fold_bodies,
                    map_cap_bodies,
                    map_acap_bodies,
                    fold_cap_bodies,
                    nest_bodies,
                    main,
                    prints,
                    ret,
                    args,
                }
            },
        )
}

// --- build ----------------------------------------------------------------

/// Referenced functions available to `main`'s steps.
pub(super) struct Ctx {
    pub(super) helpers: Vec<FuncId>,    // i32 -> i32
    pub(super) map_bodies: Vec<FuncId>, // i32 -> i32
    pub(super) fold_bodies: Vec<FuncId>,
    pub(super) map_cap_bodies: Vec<FuncId>, // (i32, i32) -> i32        (ADR-0027)
    pub(super) map_acap_bodies: Vec<FuncId>, // ([i32;ARR], i32) -> i32  (ADR-0027)
    pub(super) fold_cap_bodies: Vec<FuncId>, // (i32, i32, i32) -> i32   (ADR-0027)
    pub(super) nest_bodies: Vec<FuncId>,    // ([i32;ARR], i32) -> i32, fold inside (ADR-0027)
    pub(super) pair_sum: FuncId,            // (i32,i32) -> i32 utility, for zip/enumerate results
    pub(super) trap_free: bool,
}

impl Ctx {
    /// The in-body context: bodies reference no other fns (loop-free,
    /// collection-free — fusion P3 stays reachable). `pair_sum` is a dummy
    /// (scalar bodies never emit zip/enumerate).
    fn bodies(pair_sum: FuncId, trap_free: bool) -> Ctx {
        Ctx {
            helpers: vec![],
            map_bodies: vec![],
            fold_bodies: vec![],
            map_cap_bodies: vec![],
            map_acap_bodies: vec![],
            fold_cap_bodies: vec![],
            nest_bodies: vec![],
            pair_sum,
            trap_free,
        }
    }
}

/// Typed value pools threaded through step emission.
#[derive(Default)]
pub(super) struct Pool {
    pub(super) i32s: Vec<mapal_ir::ObjectId>,
    pub(super) i64s: Vec<mapal_ir::ObjectId>,
    pub(super) f32s: Vec<mapal_ir::ObjectId>,
    pub(super) f64s: Vec<mapal_ir::ObjectId>,
    pub(super) bools: Vec<mapal_ir::ObjectId>,
    pub(super) arrs: Vec<mapal_ir::ObjectId>, // [i32; ARR]
    /// Count of loops emitted so far. The interp scopes loop layout per-merge
    /// SCC (S12 fix: two sequential loops are supported — `interp
    /// tests/loop_invariants.rs::two_sequential_loops`), so up to `MAX_LOOPS`
    /// sequential loops are generable (the S12 P0 / llvm-review-F1 shape); each
    /// is a self-contained canonical quartet, so they nest not, only sequence.
    pub(super) loops_used: u8,
    /// Values the program must observe (S65: each carried loop's exit
    /// checksum) — returned or printed.
    pub(super) observe: Vec<mapal_ir::ObjectId>,
}

/// Sequential loops per function (S12 P0 shape needs ≥ 2; kept small so build
/// stays cheap). Each `build_loop` is statically bounded (`i < k`, `k ≤ 64`).
pub(super) const MAX_LOOPS: u8 = 2;

pub(super) fn pick<T: Copy>(v: &[T], i: u8) -> Option<T> {
    if v.is_empty() {
        None
    } else {
        Some(v[i as usize % v.len()])
    }
}

/// Interpret a script into a sealed IR. Panics on any seal failure (generator bug).
pub fn build(prog: &Prog) -> Built {
    // Positions are per-program, so a given `Prog` always builds the same graph.
    NEXT_LOC.with(|c| c.set(0));
    let mut b = IrBuilder::new();

    // Utility body: (i32,i32) -> i32 { π0 + π1 } — reduces zip/enumerate pairs.
    let pair_sum = b
        .declare(FuncKind::MapBody, "pair_sum", pair_ty(), Ty::i32(), L())
        .unwrap();
    {
        let mut fb = b.build_fn(pair_sum).unwrap();
        let p = fb.input();
        let a = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
        let c = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
        fb.binop(Operation::Add, a, c, Dest::Ret { slot: None }, L())
            .unwrap();
        fb.finish().unwrap();
    }

    // Helper Named fns (i32 -> i32) and map bodies (i32 -> i32): loop-free scalar.
    let helpers = declare_scalar_bodies(&mut b, FuncKind::Named, "helper", &prog.helpers);
    let map_bodies = declare_scalar_bodies(&mut b, FuncKind::MapBody, "mapb", &prog.map_bodies);
    // Fold bodies ((i32,i32) -> i32).
    let mut fold_bodies = Vec::new();
    for (i, steps) in prog.fold_bodies.iter().enumerate() {
        let f = b
            .declare(
                FuncKind::FoldBody,
                &format!("foldb{i}"),
                pair_ty(),
                Ty::i32(),
                L(),
            )
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let p = fb.input();
            let acc = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
            let elem = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
            let ctx = Ctx::bodies(pair_sum, prog.trap_free);
            let mut pool = Pool {
                i32s: vec![acc, elem],
                ..Default::default()
            };
            emit_steps(&mut fb, &ctx, &mut pool, steps, false);
            let ret = pool.i32s.last().copied().unwrap_or(acc);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        fold_bodies.push(f);
    }

    // ADR-0027 capture body pools.
    let map_cap_bodies =
        declare_map_cap_bodies(&mut b, "mapc", &prog.map_cap_bodies, prog.trap_free);
    let map_acap_bodies =
        declare_map_acap_bodies(&mut b, "mapa", &prog.map_acap_bodies, prog.trap_free);
    let fold_cap_bodies =
        declare_fold_cap_bodies(&mut b, "foldc", &prog.fold_cap_bodies, prog.trap_free);
    let nest_bodies = declare_nest_bodies(&mut b, &prog.nest_bodies, prog.trap_free);

    let ctx = Ctx {
        helpers,
        map_bodies,
        fold_bodies,
        map_cap_bodies,
        map_acap_bodies,
        fold_cap_bodies,
        nest_bodies,
        pair_sum,
        trap_free: prog.trap_free,
    };

    // main.
    if prog.open {
        build_open_main(&mut b, &ctx, prog)
    } else {
        build_closed_main(&mut b, &ctx, prog)
    }
}

/// Declare + emit a set of scalar `i32 -> i32` bodies of the given kind.
pub(super) fn declare_scalar_bodies(
    b: &mut IrBuilder,
    kind: FuncKind,
    prefix: &str,
    scripts: &[Vec<Step>],
) -> Vec<FuncId> {
    let mut out = Vec::new();
    for (i, steps) in scripts.iter().enumerate() {
        let f = b
            .declare(kind, &format!("{prefix}{i}"), Ty::i32(), Ty::i32(), L())
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let x = fb.input();
            // Historical: scalar bodies keep default-mode traps even under
            // `trap_free` (pre-ADR-0027 behavior; the harnesses expect 101s).
            let ctx = Ctx::bodies(f, false);
            let mut pool = Pool {
                i32s: vec![x],
                ..Default::default()
            };
            emit_steps(&mut fb, &ctx, &mut pool, steps, false);
            let ret = pool.i32s.last().copied().unwrap_or(x);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        out.push(f);
    }
    out
}

/// Declare + emit scalar-capture map bodies `(cap, elem) -> i32` (ADR-0027):
/// the pool seeds with both input projections, the generated scalar steps
/// combine them — the enclosing-scalar-read shape.
pub(super) fn declare_map_cap_bodies(
    b: &mut IrBuilder,
    prefix: &str,
    scripts: &[Vec<Step>],
    trap_free: bool,
) -> Vec<FuncId> {
    let mut out = Vec::new();
    for (i, steps) in scripts.iter().enumerate() {
        let f = b
            .declare(
                FuncKind::MapBody,
                &format!("{prefix}{i}"),
                pair_ty(),
                Ty::i32(),
                L(),
            )
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let p = fb.input();
            let cap = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
            let elem = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
            let ctx = Ctx::bodies(f, trap_free);
            let mut pool = Pool {
                i32s: vec![cap, elem],
                ..Default::default()
            };
            emit_steps(&mut fb, &ctx, &mut pool, steps, false);
            let ret = pool.i32s.last().copied().unwrap_or(elem);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        out.push(f);
    }
    out
}

/// Declare + emit array-capture map bodies `(cap_arr, elem) -> i32` (ADR-0027):
/// the pool seeds with `elem` plus every cell of the captured array, read at
/// **constant in-bounds** indices — the capture-indexing shape, trap-free by
/// construction in both modes.
pub(super) fn declare_map_acap_bodies(
    b: &mut IrBuilder,
    prefix: &str,
    scripts: &[Vec<Step>],
    trap_free: bool,
) -> Vec<FuncId> {
    let arr_ty = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: ARR as u64,
    };
    let mut out = Vec::new();
    for (i, steps) in scripts.iter().enumerate() {
        let f = b
            .declare(
                FuncKind::MapBody,
                &format!("{prefix}{i}"),
                Ty::Tuple(vec![arr_ty.clone(), Ty::i32()]),
                Ty::i32(),
                L(),
            )
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let p = fb.input();
            let cap_arr = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
            let elem = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
            let ctx = Ctx::bodies(f, trap_free);
            let mut pool = Pool {
                i32s: vec![elem],
                ..Default::default()
            };
            for j in 0..ARR {
                let idx = fb.constant(Value::I32(j as i32), L()).unwrap();
                let cell = fb.index(cap_arr, idx, Dest::Fresh(None), L()).unwrap();
                pool.i32s.push(cell);
            }
            emit_steps(&mut fb, &ctx, &mut pool, steps, false);
            let ret = pool.i32s.last().copied().unwrap_or(elem);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        out.push(f);
    }
    out
}

/// Declare + emit scalar-capture fold bodies `(cap, acc, elem) -> i32`
/// (ADR-0027): the pool seeds with all three input projections.
pub(super) fn declare_fold_cap_bodies(
    b: &mut IrBuilder,
    prefix: &str,
    scripts: &[Vec<Step>],
    trap_free: bool,
) -> Vec<FuncId> {
    let in_ty = Ty::Tuple(vec![Ty::i32(), Ty::i32(), Ty::i32()]);
    let mut out = Vec::new();
    for (i, steps) in scripts.iter().enumerate() {
        let f = b
            .declare(
                FuncKind::FoldBody,
                &format!("{prefix}{i}"),
                in_ty.clone(),
                Ty::i32(),
                L(),
            )
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let p = fb.input();
            let cap = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
            let acc = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
            let elem = fb.proj(p, 2, Dest::Fresh(None), L()).unwrap();
            let ctx = Ctx::bodies(f, trap_free);
            let mut pool = Pool {
                i32s: vec![cap, acc, elem],
                ..Default::default()
            };
            emit_steps(&mut fb, &ctx, &mut pool, steps, false);
            let ret = pool.i32s.last().copied().unwrap_or(acc);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        out.push(f);
    }
    out
}

/// Declare + emit the two-level nest bodies (ADR-0027, matmul miniature): an
/// outer map body `(cap_arr, elem) -> i32` whose first action is a
/// `fold_captured` over its own array capture with `elem` as the fold's
/// capture and seed — the inner fold body `nestf{i}` is generated alongside
/// from its own script. The outer pool then holds `[elem, fold_result]` for
/// the generated scalar steps.
pub(super) fn declare_nest_bodies(
    b: &mut IrBuilder,
    scripts: &[NestBody],
    trap_free: bool,
) -> Vec<FuncId> {
    let arr_ty = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: ARR as u64,
    };
    let mut out = Vec::new();
    for (i, script) in scripts.iter().enumerate() {
        let inner = declare_fold_cap_bodies(
            b,
            &format!("nestf{i}_"),
            std::slice::from_ref(&script.inner),
            trap_free,
        )[0];
        let f = b
            .declare(
                FuncKind::MapBody,
                &format!("nestm{i}"),
                Ty::Tuple(vec![arr_ty.clone(), Ty::i32()]),
                Ty::i32(),
                L(),
            )
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let p = fb.input();
            let cap_arr = fb.proj(p, 0, Dest::Fresh(None), L()).unwrap();
            let elem = fb.proj(p, 1, Dest::Fresh(None), L()).unwrap();
            // Fold the captured array, capturing the outer element (a capture
            // across two levels once the outer map itself captures `cap_arr`).
            let folded = fb
                .fold_captured(inner, &[elem], elem, cap_arr, Dest::Fresh(None), L())
                .unwrap();
            let ctx = Ctx::bodies(f, trap_free);
            let mut pool = Pool {
                i32s: vec![elem, folded],
                ..Default::default()
            };
            emit_steps(&mut fb, &ctx, &mut pool, &script.outer, false);
            let ret = pool.i32s.last().copied().unwrap_or(folded);
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        out.push(f);
    }
    out
}

/// Open pure `main : i32 -> i32`, returned via `eval_call` on the random args.
pub(super) fn build_open_main(b: &mut IrBuilder, ctx: &Ctx, prog: &Prog) -> Built {
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::i32(), L())
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        let mut pool = Pool {
            i32s: vec![x],
            ..Default::default()
        };
        // Seed a couple of constants so collections/arith always have operands.
        seed_consts(&mut fb, &mut pool);
        emit_steps(&mut fb, ctx, &mut pool, &prog.main, true);
        let ret = pick(&pool.i32s, prog.ret).unwrap_or(x);
        fb.output(ret, None, L()).unwrap();
        fb.finish().unwrap();
    }
    let ir = b_seal(b, f);
    let args = prog
        .args
        .iter()
        .map(|&a| RValue::Scalar(Value::I32(a)))
        .collect();
    Built {
        ir,
        open: true,
        entry: f,
        args,
    }
}

/// Closed `main`: effectful (`io -> io`, prints intermediates) or pure (`() -> i32`).
pub(super) fn build_closed_main(b: &mut IrBuilder, ctx: &Ctx, prog: &Prog) -> Built {
    if prog.effectful {
        let f = b
            .declare(FuncKind::Named, "main", Ty::IoToken, Ty::IoToken, L())
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let tok0 = fb.input();
            let mut pool = Pool::default();
            seed_consts(&mut fb, &mut pool);
            emit_steps(&mut fb, ctx, &mut pool, &prog.main, true);
            // Thread the token through prints of selected intermediates.
            let mut tok = tok0;
            let mut printed = false;
            let scalars: Vec<_> = pool
                .i32s
                .iter()
                .chain(&pool.i64s)
                .chain(&pool.f32s)
                .chain(&pool.f64s)
                .copied()
                .collect();
            for &sel in &prog.prints {
                if let Some(v) = pick(&scalars, sel) {
                    tok = fb.println(tok, v, L()).unwrap();
                    printed = true;
                }
            }
            for &v in &pool.observe {
                tok = fb.println(tok, v, L()).unwrap();
                printed = true;
            }
            if !printed {
                // A token must reach Return via a token consumer (I4b) — print a
                // constant so the trivial (no-op) main still seals.
                let c = fb.constant(Value::I32(0), L()).unwrap();
                tok = fb.println(tok, c, L()).unwrap();
            }
            fb.output(tok, None, L()).unwrap();
            fb.finish().unwrap();
        }
        let ir = b_seal(b, f);
        Built {
            ir,
            open: false,
            entry: f,
            args: vec![],
        }
    } else {
        let f = b
            .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L())
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let mut pool = Pool::default();
            seed_consts(&mut fb, &mut pool);
            emit_steps(&mut fb, ctx, &mut pool, &prog.main, true);
            let mut ret = pick(&pool.i32s, prog.ret).unwrap_or(pool.i32s[0]);
            for &v in &pool.observe {
                ret = fb
                    .binop(Operation::Add, ret, v, Dest::Fresh(None), L())
                    .unwrap();
            }
            fb.output(ret, None, L()).unwrap();
            fb.finish().unwrap();
        }
        let ir = b_seal(b, f);
        Built {
            ir,
            open: false,
            entry: f,
            args: vec![],
        }
    }
}

/// Seal a builder whose `main` is `entry`, consuming a fresh clone-free builder.
pub(super) fn b_seal(b: &mut IrBuilder, entry: FuncId) -> CategoryIr {
    // `seal` consumes the builder; swap a fresh one in to move it out.
    let taken = std::mem::replace(b, IrBuilder::new());
    taken.seal(entry).expect("generator: seal must succeed")
}

/// Push a few seed constants so early collection/arith steps have operands.
pub(super) fn seed_consts(fb: &mut mapal_ir::FnBuilder<'_>, pool: &mut Pool) {
    for k in [1i32, 2, 3] {
        pool.i32s.push(fb.constant(Value::I32(k), L()).unwrap());
    }
    pool.bools
        .push(fb.constant(Value::Bool(true), L()).unwrap());
}
