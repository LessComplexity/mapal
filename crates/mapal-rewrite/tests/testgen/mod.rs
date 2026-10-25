//! DESIGN §6 — the random Core-program generator (HANDOFF §9: lives here, feeds
//! P5–P7 differential tests). proptest yields a **Clone-able script** (`CategoryIr`
//! is not `Clone`, so it cannot be a strategy value); [`build`] interprets the
//! script into a sealed graph. Seal must always succeed — a generator that emits
//! an ill-typed step is a bug, surfaced as an `unwrap` panic.
//!
//! Two strategies (both modes):
//! - **closed** (`open == false`): an entry `main`, effectful (token-threaded
//!   prints of intermediates) or pure, over a DAG of scalar/tuple/array ops,
//!   helper fns, and canonical bounded loops (`i < K`, `K ≤ 64` — terminating).
//! - **open** (`open == true`): a pure `Named` fn `i32 → i32`, exercised via
//!   `eval_call` with random `i32` args.
//!
//! Modes: `default` (traps permitted — `Div`/`Mod`/`Index` on arbitrary feeders)
//! and `trap_free` (divisors const-nonzero, indices const-in-bounds).
//!
//! ADR-0027 captures: dedicated main-level steps emit `map_captured` /
//! `fold_captured` over body pools whose inputs are the `(c₁…cₖ, …)` products
//! — a scalar capture, an array capture (body indexes it at constant in-bounds
//! indices), a fold capture, a fold nested in a map body capturing across two
//! levels (the matmul miniature), and a loop-carried scalar captured by a map
//! body (the read-at-position case). All are trap-safe by construction in
//! `trap_free` mode and loop-free inside bodies (fusion P3 stays reachable).
#![allow(dead_code)]

use mapal_interp::RValue;
use mapal_ir::{CategoryIr, Dest, FuncId, FuncKind, IrBuilder, Operation, SourceLoc, Ty, Value};
use proptest::prelude::*;

thread_local! {
    /// Monotonic source-position counter, reset at the top of [`build`].
    static NEXT_LOC: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The next source position, one byte wide.
///
/// Generated programs have no source text, but they DO have a statement order —
/// the order this generator emits them — and since plan-s38 that order is
/// semantically load-bearing: `topo_order` breaks ties on source position, so a
/// corpus stamping every statement `0..0` carries no order at all and the
/// tie-break degenerates straight back to insertion order (which is the thing
/// under test). A monotonic counter makes "position order == emission order",
/// i.e. exactly the statement order these programs model. Reset per `build`, so
/// a given `Prog` always yields identical positions.
#[allow(non_snake_case)]
fn L() -> SourceLoc {
    NEXT_LOC.with(|c| {
        let at = c.get();
        c.set(at + 1);
        SourceLoc::new(at, at + 1)
    })
}
/// Every generated array is `[i32; ARR]` — one fixed size keeps `zip`/`index`/
/// `map`/`fold`/`enumerate` typing trivial and always valid.
const ARR: usize = 3;

/// One value-producing step over the typed pools. Indices are reduced modulo
/// the live pool length, so a step with no operand of the right type is skipped.
#[derive(Clone, Debug)]
pub enum Step {
    ConstI32(i32),
    ConstBool(bool),
    Bin {
        op: u8,
        a: u8,
        b: u8,
    },
    Neg {
        a: u8,
    },
    /// ADR-0029: one of the four legal widening-lattice edges.
    Widen {
        edge: u8,
        a: u8,
    },
    /// ADR-0029 array construction (S21): `iota(3)` joins the `[i32; 3]` pool.
    Iota,
    /// ADR-0029 (S21): `fill(x, 3)` from an i32-pool value joins the arr pool.
    Fill {
        a: u8,
    },
    Cmp {
        op: u8,
        a: u8,
        b: u8,
    },
    Not {
        a: u8,
    },
    Logic {
        or: bool,
        a: u8,
        b: u8,
    },
    Phi {
        t: u8,
        e: u8,
        c: u8,
    },
    /// plan-s39: a guard whose arm owns work that can TRAP — a `Div`/`Mod`
    /// consumed ONLY by one Phi arm. `Step::Phi` cannot produce this: it picks
    /// both arms from the pool, so an arm never owns its producer, and a census
    /// over this corpus found 0 of 82 guard sites with a trapping arm. That
    /// hole is why an untaken `7 / 0` reached production.
    PhiTrapArm {
        a: u8,
        b: u8,
        c: u8,
        /// Which arm owns the trapping op.
        on_true: bool,
        /// `Mod` instead of `Div`.
        modulo: bool,
    },
    PackProj {
        a: u8,
        b: u8,
        snd: bool,
    },
    MakeArray {
        a: u8,
        b: u8,
        c: u8,
    },
    Index {
        arr: u8,
        idx: u8,
    },
    Update {
        arr: u8,
        idx: u8,
        val: u8,
    },
    MapArr {
        arr: u8,
        body: u8,
    },
    FoldArr {
        arr: u8,
        seed: u8,
        body: u8,
    },
    Zip {
        a: u8,
        b: u8,
    },
    Enumerate {
        arr: u8,
    },
    Call {
        a: u8,
        helper: u8,
    },
    Loop {
        k: u8,
    },
    /// R-LF: canonical `(counter, acc)` with static `K >= 1`.
    LiftFold {
        k: u8,
        seed: u8,
        cap: u8,
    },
    /// R-LM: canonical `(out, counter)` with one identity-indexed Update and
    /// `len(out) == K == ARR`.
    LiftMap {
        arr: u8,
        cap: u8,
    },
    /// ADR-0027: map with an i32 capture from the scalar pool; body
    /// `(cap, elem) -> i32` (the `x * scale` shape).
    MapCapScalar {
        arr: u8,
        cap: u8,
        body: u8,
    },
    /// ADR-0027: map with an `[i32; ARR]` capture from the array pool; body
    /// `([i32; ARR], elem) -> i32` indexing the capture in-bounds.
    MapCapArray {
        arr: u8,
        cap: u8,
        body: u8,
    },
    /// ADR-0027: fold with an i32 capture; body `(cap, acc, elem) -> i32`.
    FoldCapScalar {
        arr: u8,
        seed: u8,
        cap: u8,
        body: u8,
    },
    /// ADR-0027: map whose body folds the map's own array capture, capturing
    /// the outer element inside the fold — capture across two levels (the
    /// matmul shape in miniature). Body `([i32; ARR], elem) -> i32`.
    MapNestFold {
        arr: u8,
        cap: u8,
        body: u8,
    },
    /// ADR-0027: a canonical bounded loop, then a map capturing the loop's
    /// exit value — the loop-carried read-at-position case.
    LoopCapMap {
        k: u8,
        arr: u8,
        body: u8,
    },
    /// S65: a canonical loop carrying arrays — state `(counter, x0, [x1,]
    /// acc)`, or `(counter, x0, (x1, acc))` with `nest` — of `k % 9` trips
    /// (zero included). The body runs `ops` over the state's views, the
    /// loop-invariant arrays and scalars before it: maps, updates (chains of
    /// them), element reads, folds, guards choosing between an arm-owned
    /// fresh array and another (`Phi`), zips, captured maps. Each array's next
    /// state is picked from what the body has (a fresh array, a view —
    /// unchanged or swapped — or an invariant); the exit forwards one state
    /// array, whose elements the program observes.
    CarryLoop {
        k: u8,
        two: bool,
        nest: bool,
        init: [u8; 2],
        ops: Vec<[u8; 3]>,
        next: [u8; 2],
        exit: u8,
    },
}

/// A generated fold-in-map body (ADR-0027 two-level capture): the inner fold
/// body script `(cap, acc, elem) -> i32`, then the outer map body's own scalar
/// steps (run after the fold, over `[elem, fold_result, …]`).
#[derive(Clone, Debug)]
pub struct NestBody {
    pub inner: Vec<Step>,
    pub outer: Vec<Step>,
}

/// A full generated program script (Clone, for proptest shrinking).
#[derive(Clone, Debug)]
pub struct Prog {
    pub trap_free: bool,
    pub open: bool,
    /// Effectful (token-threaded prints) vs pure, for a closed `main`.
    pub effectful: bool,
    pub helpers: Vec<Vec<Step>>,     // Named   i32 -> i32
    pub map_bodies: Vec<Vec<Step>>,  // MapBody  i32 -> i32
    pub fold_bodies: Vec<Vec<Step>>, // FoldBody (i32,i32) -> i32
    /// ADR-0027 body pools: scalar-capture map bodies `(i32,i32) -> i32`,
    /// array-capture map bodies `([i32;ARR],i32) -> i32`, scalar-capture fold
    /// bodies `(i32,i32,i32) -> i32`, and fold-in-map nest bodies.
    pub map_cap_bodies: Vec<Vec<Step>>,
    pub map_acap_bodies: Vec<Vec<Step>>,
    pub fold_cap_bodies: Vec<Vec<Step>>,
    pub nest_bodies: Vec<NestBody>,
    pub main: Vec<Step>,
    pub prints: Vec<u8>,
    pub ret: u8,
    pub args: Vec<i32>,
}

/// A built program plus how to run it (open ⇒ `eval_call` per arg; closed ⇒ `run`).
pub struct Built {
    pub ir: CategoryIr,
    pub open: bool,
    pub entry: FuncId,
    pub args: Vec<RValue>,
}

// --- strategies -----------------------------------------------------------

/// A small-int-biased `i32` constant strategy (DESIGN §6: ±100 + edge values).
fn const_i32() -> impl Strategy<Value = i32> {
    prop_oneof![
        3 => -100i32..=100,
        1 => prop::sample::select(vec![0, 1, -1, 2, -2, i32::MIN, i32::MAX, 100, -100]),
    ]
}

/// A restricted step (scalar ops only) — for map/fold/helper bodies, which must
/// stay loop-free and collection-free (totality + fusion P3).
fn scalar_step() -> impl Strategy<Value = Step> {
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
fn main_step() -> impl Strategy<Value = Step> {
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

fn body_strategy() -> impl Strategy<Value = Vec<Step>> {
    prop::collection::vec(scalar_step(), 0..5)
}

/// A nest-body script: the inner fold body + the outer map body's own steps.
fn nest_body_strategy() -> impl Strategy<Value = NestBody> {
    (body_strategy(), body_strategy()).prop_map(|(inner, outer)| NestBody { inner, outer })
}

/// S65: [`Step::CarryLoop`].
fn carry_step() -> impl Strategy<Value = Step> {
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

fn prog_strategy_with(
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
struct Ctx {
    helpers: Vec<FuncId>,    // i32 -> i32
    map_bodies: Vec<FuncId>, // i32 -> i32
    fold_bodies: Vec<FuncId>,
    map_cap_bodies: Vec<FuncId>,  // (i32, i32) -> i32        (ADR-0027)
    map_acap_bodies: Vec<FuncId>, // ([i32;ARR], i32) -> i32  (ADR-0027)
    fold_cap_bodies: Vec<FuncId>, // (i32, i32, i32) -> i32   (ADR-0027)
    nest_bodies: Vec<FuncId>,     // ([i32;ARR], i32) -> i32, fold inside (ADR-0027)
    pair_sum: FuncId,             // (i32,i32) -> i32 utility, for zip/enumerate results
    trap_free: bool,
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
struct Pool {
    i32s: Vec<mapal_ir::ObjectId>,
    i64s: Vec<mapal_ir::ObjectId>,
    f32s: Vec<mapal_ir::ObjectId>,
    f64s: Vec<mapal_ir::ObjectId>,
    bools: Vec<mapal_ir::ObjectId>,
    arrs: Vec<mapal_ir::ObjectId>, // [i32; ARR]
    /// Count of loops emitted so far. The interp scopes loop layout per-merge
    /// SCC (S12 fix: two sequential loops are supported — `interp
    /// tests/loop_invariants.rs::two_sequential_loops`), so up to `MAX_LOOPS`
    /// sequential loops are generable (the S12 P0 / llvm-review-F1 shape); each
    /// is a self-contained canonical quartet, so they nest not, only sequence.
    loops_used: u8,
    /// Values the program must observe (S65: each carried loop's exit
    /// checksum) — returned or printed.
    observe: Vec<mapal_ir::ObjectId>,
}

/// Sequential loops per function (S12 P0 shape needs ≥ 2; kept small so build
/// stays cheap). Each `build_loop` is statically bounded (`i < k`, `k ≤ 64`).
const MAX_LOOPS: u8 = 2;

fn pick<T: Copy>(v: &[T], i: u8) -> Option<T> {
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
fn declare_scalar_bodies(
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
fn declare_map_cap_bodies(
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
fn declare_map_acap_bodies(
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
fn declare_fold_cap_bodies(
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
fn declare_nest_bodies(b: &mut IrBuilder, scripts: &[NestBody], trap_free: bool) -> Vec<FuncId> {
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
fn build_open_main(b: &mut IrBuilder, ctx: &Ctx, prog: &Prog) -> Built {
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
fn build_closed_main(b: &mut IrBuilder, ctx: &Ctx, prog: &Prog) -> Built {
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
fn b_seal(b: &mut IrBuilder, entry: FuncId) -> CategoryIr {
    // `seal` consumes the builder; swap a fresh one in to move it out.
    let taken = std::mem::replace(b, IrBuilder::new());
    taken.seal(entry).expect("generator: seal must succeed")
}

/// Push a few seed constants so early collection/arith steps have operands.
fn seed_consts(fb: &mut mapal_ir::FnBuilder<'_>, pool: &mut Pool) {
    for k in [1i32, 2, 3] {
        pool.i32s.push(fb.constant(Value::I32(k), L()).unwrap());
    }
    pool.bools
        .push(fb.constant(Value::Bool(true), L()).unwrap());
}

// --- step emission --------------------------------------------------------

fn emit_steps(
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

fn emit_step(
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
fn build_carry_loop(
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
        arrs: views.iter().copied().chain(outer_arrs.iter().copied()).collect(),
        i32s: [i, acc].into_iter().chain(outer_i32s.iter().copied()).collect(),
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
            fb.constant(Value::I32(i32::from(sel / 2) % ARR as i32), L()).unwrap()
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
                lp.arrs.push(fb.update(x, at, v, fresh.clone(), L()).unwrap());
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
                let (t, e) = if b % 2 == 0 { (arm, other) } else { (other, arm) };
                lp.arrs.push(fb.phi(t, e, g, fresh.clone(), L()).unwrap());
            }
            5 => {
                let y = take(&lp.arrs, b);
                let z = fb.zip(x, y, fresh.clone(), L()).unwrap();
                lp.arrs.push(fb.map(ctx.pair_sum, z, fresh.clone(), L()).unwrap());
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
                lp.arrs.push(fb.update(u, a1, i, fresh.clone(), L()).unwrap());
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
fn build_loop(fb: &mut mapal_ir::FnBuilder<'_>, k: i32) -> mapal_ir::ObjectId {
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

fn build_lift_fold(
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

fn build_lift_map(
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

fn pair_ty() -> Ty {
    Ty::Tuple(vec![Ty::i32(), Ty::i32()])
}

// --- S65: device-sized loops, as Mapal source --------------------------------

/// A loop over `[f32; n·n]` arrays sized for a device (`n` 32 or 64): one or
/// two carried arrays (`x`, `y`) from closed-form inits and a loop-invariant
/// `w`; per trip `ops` over them — affine sites (`Σ_k A[i·n+k]·B[k·n+j]`),
/// per-cell maps (relu, scale, a two-array sum), guards choosing between
/// arrays, row writes into the state (`x[it·n + j] <- v`, the value one
/// element of another array); each state's next value picked from what the
/// trip made, the state itself or `w`; the exit forwards a state, read by one
/// element or whole. Built as source text (no parser dependency here).
#[derive(Clone, Debug)]
pub struct DeviceProg {
    pub big: bool,
    pub trips: u8,
    pub two: bool,
    pub ops: Vec<[u8; 3]>,
    pub next: [u8; 2],
    pub exit: bool,
    pub whole: bool,
}

pub fn device_prog_strategy() -> impl Strategy<Value = DeviceProg> {
    (
        prop::bool::weighted(0.25),
        1u8..5,
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..5),
        prop::array::uniform2(any::<u8>()),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(big, trips, two, ops, next, exit, whole)| DeviceProg {
            big,
            trips,
            two,
            ops,
            next,
            exit,
            whole,
        })
}

/// [`DeviceProg`]'s source text.
pub fn device_src(p: &DeviceProg) -> String {
    let n = if p.big { 64 } else { 32 };
    let nn = n * n;
    let law = |name: &str, a: u32, b: u32, m: u32, scale: u32| {
        format!("    ta -> map {{ t -> (t * {a} + {b}) % {m} - {h} -> widen_f32 -> v; v / {scale}.0 }} -> {name};\n", h = m / 2)
    };
    let mut s = format!("fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n");
    s += &law("a0", 7, 13, 101, 8 * n);
    s += &law("a1", 5, 3, 97, 8 * n);
    s += &law("w", 11, 57, 103, 4 * n);
    s += &format!("    mut x: [f32; {nn}] <- a0;\n");
    if p.two {
        s += &format!("    mut y: [f32; {nn}] <- a1;\n");
    }
    s += "    mut it: i32 <- 0;\n    loop {\n        (it < ";
    s += &format!("{}) -> {{\n            -true-> {{\n", p.trips);
    let ind = "                ";
    let mut arrs: Vec<String> = vec!["x".into()];
    if p.two {
        arrs.push("y".into());
    }
    arrs.push("w".into());
    let pick = |v: &[String], k: u8| v[k as usize % v.len()].clone();
    // The ops chain: each reads the one before (so every value the trip makes
    // is live — a loop runs only what feeds its next state or its exit), and
    // a second operand from anything made so far; the last is `x`'s next.
    let mut cur = "x".to_string();
    for (z, &[op, _, b]) in p.ops.iter().enumerate() {
        // (Most chains start at a site: a device unit to place.)
        let op = if z == 0 && op % 4 != 3 { 0 } else { op % 8 };
        let (x, y) = (cur.clone(), pick(&arrs, b));
        let z = format!("z{z}");
        s += ind;
        s += &match op {
            0 => format!(
                "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -> {z};\n"
            ),
            1 => format!("{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n"),
            2 => format!("{x} -> map {{ v -> v * 0.5 }} -> {z};\n"),
            3 => format!("ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n"),
            4 => format!("(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n", b % (p.trips + 1)),
            // A closed form of the cell and the trip counter (a uniform), read
            // by a site: built where the site runs, never uploaded.
            6 => format!(
                "ta -> map {{ t -> ((t + it * {c}) * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / {s}.0 }} -> {z}g;\n{ind}ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {z}g[k * {n} + j] }} }} -> {z};\n",
                c = u32::from(b % 5) + 1,
                s = 8 * n
            ),
            // A guarded dot product per cell over the trip counter (causal
            // attention's score): no tile site, a cell whose body folds.
            7 => format!(
                "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (j <= it * {c}) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -false-> -1.0; }} }} -> {z};\n",
                c = u32::from(b % 7) + 1
            ),
            _ => {
                // A row write into the state `x`: the trip's row, its cell `j`
                // one element of the chain (a copy-out when that one lives on
                // the device).
                let j = u32::from(b) % n;
                cur = "x".into();
                format!("{x}[it * {n} + {j}] * 0.5 + 0.25 -> {z}v;\n{ind}x[it * {n} + {j}] <- {z}v;\n")
            }
        };
        if op != 5 {
            arrs.push(z.clone());
            cur = z;
        }
    }
    if cur != "x" {
        s += &format!("{ind}{cur} -> x;\n");
    }
    if p.two {
        // `y`'s next: itself, `w`, `x`'s new value or the trip's first array.
        let first = arrs[3.min(arrs.len() - 1)].clone();
        let ny = pick(&["y".to_string(), "w".to_string(), "x".to_string(), first], p.next[1]);
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    let out = if p.exit && p.two { "y" } else { "x" };
    s += &format!("{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> {out} -> out;\n        }}\n    }}\n");
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", nn - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}

// --- generic-fusion: chains of mixed algorithms, as Mapal source -------------

/// A chain of 2..9 algorithm nodes over `f32` arrays of an `r × c` grid (`r`,
/// `c` in 4..16), `k`-deep tiles, `t`-tap windows: laws (iota maps), capture-free
/// maps (arithmetic, a relu / clamp guard, a constant-trip nested fold), cells
/// reading at the cell, its row, its column, a constant cell or an affine
/// in-row cell, tile sites (row-major, transposed `a` or `b`; chained through a
/// `c × c` operand), windows (FIR, 2-D `k÷div` taps, a weightless stencil and
/// box filter), row folds (sum, max, min), a fold's row value read per cell
/// (`f(x[t], v[t ÷ c])`), whole folds, and loops (≤ 4 trips) whose body maps,
/// writes one row naturally (`c / C == it`) or runs a row-local tile over the
/// state. Operands are picked from the arrays of the same shape made so far, so
/// chains fan out and fan in. Every sink — and every intermediate whose
/// `print` bit is set (a host reader) — prints a checksum and two corners.
#[derive(Clone, Debug)]
pub struct Chain {
    pub r: u8,
    pub c: u8,
    pub k: u8,
    pub taps: u8,
    pub nodes: Vec<[u8; 4]>,
}

pub fn chain_strategy() -> impl Strategy<Value = Chain> {
    (
        4u8..17,
        4u8..17,
        2u8..17,
        2u8..6,
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..10),
    )
        .prop_map(|(r, c, k, taps, nodes)| Chain {
            r,
            c,
            k,
            taps,
            nodes,
        })
}

/// An array's shape in a [`Chain`]: its length is a function of the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// `r × c`.
    M,
    /// `r` (a row value).
    V,
    /// `c` (a column value).
    W,
    /// `r × k`, `k × c`, `c × k` (tile operands), `c × c`.
    A,
    B,
    Bt,
    Q,
    /// `r·c + taps − 1` (a signal), `taps`.
    S,
    H,
    /// `(r + 2)·(c + 2)` (an image for 3 × 3 taps), `9` (the taps).
    Img,
    Kw,
}

struct ChainCtx {
    r: u64,
    c: u64,
    k: u64,
    t: u64,
    s: String,
    pool: Vec<(String, Shape)>,
    iotas: Vec<u64>,
    laws: u32,
}

impl ChainCtx {
    fn len(&self, sh: Shape) -> u64 {
        let (r, c, k, t) = (self.r, self.c, self.k, self.t);
        match sh {
            Shape::M => r * c,
            Shape::V => r,
            Shape::W => c,
            Shape::A => r * k,
            Shape::B | Shape::Bt => k * c,
            Shape::Q => c * c,
            Shape::S => r * c + t - 1,
            Shape::H => t,
            Shape::Img => (r + 2) * (c + 2),
            Shape::Kw => 9,
        }
    }

    /// `n -> iota -> i<n>`, once per length.
    fn iota(&mut self, n: u64) -> String {
        if !self.iotas.contains(&n) {
            self.iotas.push(n);
            self.s += &format!("    {n} -> iota -> ix{n};\n");
        }
        format!("ix{n}")
    }

    /// A closed-form array of shape `sh` (values in about ±6, a few bits).
    fn law(&mut self, sh: Shape, key: u8) -> String {
        let n = self.len(sh);
        let it = self.iota(n);
        self.laws += 1;
        let name = format!("l{}", self.laws);
        let (mul, add, m) = [(7, 13, 101), (11, 57, 103), (5, 3, 97), (3, 1, 89)][key as usize % 4];
        let div = [7.0, 9.0, 11.0, 13.0][(key as usize / 4) % 4];
        self.s += &format!(
            "    {it} -> map {{ t -> (t * {mul} + {add}) % {m} - {h} -> widen_f32 -> v; v / {div:.1} }} -> {name};\n",
            h = m / 2
        );
        self.pool.push((name.clone(), sh));
        name
    }

    /// An array of shape `sh` made so far (by `pick`), else a fresh law.
    fn pick(&mut self, sh: Shape, pick: u8) -> String {
        let have: Vec<String> = self.pool.iter().filter(|(_, s)| *s == sh).map(|(n, _)| n.clone()).collect();
        // One in four picks makes a fresh operand even when one exists.
        if have.is_empty() || pick % 4 == 3 {
            return self.law(sh, pick / 4);
        }
        have[pick as usize % have.len()].clone()
    }

    fn print(&mut self, name: &str, sh: Shape) {
        let n = self.len(sh);
        self.s += &format!(
            "    (zero, {name}) -> fold {{ acc, v -> acc + v }} -> sum_{name};\n    sum_{name} -> println;\n    {name}[0] -> println;\n    {name}[{}] -> println;\n",
            n - 1
        );
    }
}

impl Chain {
    /// The chain's Mapal source.
    pub fn source(&self) -> String {
        let (r, c, k, t) = (
            u64::from(self.r),
            u64::from(self.c),
            u64::from(self.k),
            u64::from(self.taps),
        );
        let mut x = ChainCtx {
            r,
            c,
            k,
            t,
            s: String::from("fn main() {\n    0.0 -> zero: f32;\n"),
            pool: Vec::new(),
            iotas: Vec::new(),
            laws: 0,
        };
        let rc = r * c;
        let mut printed: Vec<String> = Vec::new();
        let last = self.nodes.len() - 1;
        for (z, &[op, p, q, flags]) in self.nodes.iter().enumerate() {
            let o = format!("n{z}");
            let tm = x.iota(rc);
            // Most nodes read an `r × c` array: start from a tile site.
            let op = if z == 0 && op % 3 != 0 { 6 } else { op % 16 };
            let shape = match op {
                0 => {
                    let it = x.iota(rc);
                    let (mul, add) = [(3, 5), (13, 7)][p as usize % 2];
                    x.s += &format!(
                        "    {it} -> map {{ t -> (t * {mul} + {add}) % 31 - 15 -> widen_f32 -> v; v / 4.0 }} -> {o};\n"
                    );
                    Shape::M
                }
                1 => {
                    let a = x.pick(Shape::M, p);
                    let body = match q % 4 {
                        0 => "v * 0.5 + 0.25".to_string(),
                        1 => "v -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> w; w".to_string(),
                        2 => "v -> u; (u > 1.5) -> { -true-> 1.5; -false-> u; } -> w; w * 2.0".to_string(),
                        _ => {
                            let kk = x.iota(3);
                            format!("(v, {kk}) -> fold {{ acc, j -> acc * 0.5 + 0.125 }}")
                        }
                    };
                    x.s += &format!("    {a} -> map {{ v -> {body} }} -> {o};\n");
                    Shape::M
                }
                2 => {
                    let (a, b) = (x.pick(Shape::M, p), x.pick(Shape::M, q));
                    x.s += &format!("    {tm} -> map {{ t -> {a}[t] * {b}[t] + 0.5 }} -> {o};\n");
                    Shape::M
                }
                3 | 14 => {
                    let (a, v) = (x.pick(Shape::M, p), x.pick(Shape::V, q));
                    let e = if op == 3 {
                        format!("{a}[t] - {v}[t / {c}]")
                    } else {
                        format!("{a}[t] / ({v}[t / {c}] * {v}[t / {c}] + 1.0)")
                    };
                    x.s += &format!("    {tm} -> map {{ t -> {e} }} -> {o};\n");
                    Shape::M
                }
                4 => {
                    let (a, w) = (x.pick(Shape::M, p), x.pick(Shape::W, q));
                    x.s += &format!("    {tm} -> map {{ t -> {a}[t] + {w}[t % {c}] }} -> {o};\n");
                    Shape::M
                }
                5 => {
                    let a = x.pick(Shape::M, p);
                    // An in-row cell, one cell twice (two addresses of one
                    // array), and another array at the row index or the row's
                    // first cell (a halo outside a cell-wide unit).
                    let e = match q % 4 {
                        0 => format!("{a}[t] * 0.5 + {a}[t / {c} * {c} + {}]", c - 1),
                        1 => format!("{a}[t] - {a}[3]"),
                        2 => {
                            let b = x.pick(Shape::M, q / 4);
                            format!("{a}[t] + {b}[t / {c}] * 0.5")
                        }
                        _ => {
                            let b = x.pick(Shape::M, q / 4);
                            format!("{a}[t] - {b}[t / {c} * {c}]")
                        }
                    };
                    x.s += &format!("    {tm} -> map {{ t -> {e} }} -> {o};\n");
                    Shape::M
                }
                6 | 7 => {
                    let (kr, ai, bi) = if op == 6 {
                        let (al, bl) = (q % 3 == 1, q % 3 == 2);
                        let a = x.pick(Shape::A, p);
                        let b = x.pick(if bl { Shape::Bt } else { Shape::B }, p / 2);
                        let ai = if al { format!("{a}[k * {r} + i]") } else { format!("{a}[i * {k} + k]") };
                        let bi = if bl { format!("{b}[j * {k} + k]") } else { format!("{b}[k * {c} + j]") };
                        (x.iota(k), ai, bi)
                    } else {
                        let a = x.pick(Shape::M, p);
                        let b = x.pick(Shape::Q, q);
                        let ai = format!("{a}[i * {c} + k]");
                        let bi = format!("{b}[k * {c} + j]");
                        (x.iota(c), ai, bi)
                    };
                    x.s += &format!(
                        "    {tm} -> map {{ t ->\n        t / {c} -> i;\n        t % {c} -> j;\n        (0.0, {kr}) -> fold {{ acc, k -> acc + {ai} * {bi} }}\n    }} -> {o};\n"
                    );
                    Shape::M
                }
                8 => {
                    let a = x.pick(Shape::M, p);
                    let (tr, tc) = (x.iota(r), x.iota(c));
                    let (seed, step) = match q % 3 {
                        0 => ("0.0", "acc + e"),
                        1 => ("-1000000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                        _ => ("1000000000.0", "(e < acc) -> { -true-> e; -false-> acc; }"),
                    };
                    x.s += &format!(
                        "    {tr} -> map {{ i -> ({seed}, {tc}) -> fold {{ acc, j -> {a}[i * {c} + j] -> e; {step} }} }} -> {o};\n"
                    );
                    Shape::V
                }
                9 | 11 => {
                    let sg = x.pick(Shape::S, p);
                    let tt = x.iota(t);
                    let e = if op == 9 {
                        let h = x.pick(Shape::H, q);
                        format!("acc + {h}[k] * {sg}[t + k]")
                    } else {
                        format!("acc + {sg}[t + k]")
                    };
                    x.s += &format!("    {tm} -> map {{ t -> (0.0, {tt}) -> fold {{ acc, k -> {e} }} }} -> {o};\n");
                    Shape::M
                }
                10 => {
                    let img = x.pick(Shape::Img, p);
                    let tq = x.iota(9);
                    let at = format!("{img}[(i + k / 3) * {} + j + k % 3]", c + 2);
                    let e = if q % 2 == 0 {
                        let w = x.pick(Shape::Kw, q / 2);
                        format!("acc + {w}[k] * {at}")
                    } else {
                        format!("acc + {at}")
                    };
                    x.s += &format!(
                        "    {tm} -> map {{ t ->\n        t / {c} -> i;\n        t % {c} -> j;\n        (0.0, {tq}) -> fold {{ acc, k -> {e} }}\n    }} -> {o};\n"
                    );
                    Shape::M
                }
                12 => {
                    let v = x.pick(Shape::V, p);
                    x.s += &format!("    {v} -> map {{ e -> e * 0.5 + 1.0 }} -> {o};\n");
                    Shape::V
                }
                13 => {
                    // A loop carrying an `r × c` state.
                    let a = x.pick(Shape::M, p);
                    let trips = 1 + u64::from(q % 4);
                    let (sv, it) = (format!("st{z}"), format!("it{z}"));
                    let body = match flags % 3 {
                        0 => {
                            let y = x.pick(Shape::M, flags / 3);
                            format!(
                                "                {sv} -> map {{ v -> v * 0.5 + 0.25 }} -> u{z};\n                {tm} -> map {{ t -> u{z}[t] + {y}[t] }} -> w{z};\n"
                            )
                        }
                        1 => format!(
                            "                {tm} -> map {{ e -> (e / {c} == {it}) -> {{ -true-> {sv}[e] * 2.0 + 1.0; -false-> {sv}[e]; }} }} -> w{z};\n"
                        ),
                        _ => {
                            let qq = x.pick(Shape::Q, flags / 3);
                            let kc = x.iota(c);
                            format!(
                                "                {tm} -> map {{ t ->\n                    t / {c} -> i;\n                    t % {c} -> j;\n                    (0.0, {kc}) -> fold {{ acc, k -> acc + {sv}[i * {c} + k] * {qq}[k * {c} + j] }}\n                }} -> w0{z};\n                w0{z} -> map {{ v -> v * 0.125 }} -> w{z};\n"
                            )
                        }
                    };
                    x.s += &format!(
                        "    mut {sv}: [f32; {rc}] <- {a};\n    mut {it}: i32 <- 0;\n    loop {{\n        ({it} < {trips}) -> {{\n            -true-> {{\n{body}                w{z} -> {sv};\n                {it} + 1 -> {it};\n                -> loop;\n            }}\n            -false-> {sv} -> {o};\n        }}\n    }}\n"
                    );
                    // A name a loop's exit binds is read here, never captured
                    // by a later body (the lowering resolves captures before
                    // the loop's exits: `L1101`).
                    x.print(&o, Shape::M);
                    printed.push(o);
                    continue;
                }
                _ if q % 2 == 0 => {
                    // A cell of one array's row-first cells, read by the next
                    // at the row index: outside the reader's own cell.
                    let (a, b) = (x.pick(Shape::M, p), x.pick(Shape::M, q / 2));
                    x.s += &format!(
                        "    {tm} -> map {{ t -> {a}[t / {c} * {c}] * 0.5 }} -> {o}h;\n    {tm} -> map {{ t -> {b}[t] + {o}h[t / {c}] }} -> {o};\n"
                    );
                    Shape::M
                }
                _ => {
                    // A whole fold, printed (a host reader of its operand's
                    // fold, not of the array).
                    let a = x.pick(Shape::M, p);
                    x.s += &format!("    (zero, {a}) -> fold {{ acc, v -> acc + v * 0.5 }} -> {o};\n    {o} -> println;\n");
                    continue;
                }
            };
            x.pool.push((o.clone(), shape));
            if z == last || flags & 0x30 == 0x30 {
                x.print(&o, shape);
                printed.push(o);
            }
        }
        if printed.is_empty() {
            // The last node was a whole fold: print the last array too.
            if let Some((o, sh)) = x.pool.iter().rev().find(|(o, _)| o.starts_with('n')).cloned() {
                x.print(&o, sh);
            }
        }
        x.s + "}\n"
    }
}

// --- generic-fusion natural forms: twin programs, as Mapal source -----------

/// A device-sized loop like [`DeviceProg`] whose writes into the state `x`
/// and whose cell exponents come in two forms with the same values — the
/// **natural** one a person writes (a guarded map over the cells: a row
/// `t ÷ n == it`, a column `t % n == c`, a range `L ≤ t < L + w`, one cell
/// `t == L`; a power as a fold over a range) and the **unrolled** one the
/// compiler used to need (`Update`s at each cell, a chain of squarings).
/// Per trip `ops` over sites, per-cell maps (relu, scale, a two-array sum, a
/// clamped power), guards choosing between arrays, row folds, and those
/// writes; one or two carried arrays, an element or whole exit read.
#[derive(Clone, Debug)]
pub struct TwinProg {
    pub big: bool,
    pub trips: u8,
    pub two: bool,
    pub ops: Vec<[u8; 3]>,
    pub next: u8,
    pub exit: bool,
    pub whole: bool,
}

pub fn twin_prog_strategy() -> impl Strategy<Value = TwinProg> {
    (
        prop::bool::weighted(0.25),
        1u8..5,
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..6),
        any::<u8>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(big, trips, two, ops, next, exit, whole)| TwinProg {
            big,
            trips,
            two,
            ops,
            next,
            exit,
            whole,
        })
}

/// [`TwinProg`]'s source text, `natural` or unrolled. `check = (a, b, len)`
/// instruments the trip (the write-delta soundness probe): it prints
/// `dz=<k>`, the number of cells of the state `x` that changed in the trip
/// outside `[a·it + b, a·it + b + len)`.
pub fn twin_src(p: &TwinProg, natural: bool, check: Option<(i64, i64, u64)>) -> String {
    let n: u32 = if p.big { 32 } else { 16 };
    let nn = n * n;
    let law = |name: &str, a: u32, b: u32, m: u32, scale: u32| {
        format!("    ta -> map {{ t -> (t * {a} + {b}) % {m} - {h} -> widen_f32 -> v; v / {scale}.0 }} -> {name};\n", h = m / 2)
    };
    let mut s = format!("fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n");
    s += &law("a0", 7, 13, 101, 8 * n);
    s += &law("a1", 5, 3, 97, 8 * n);
    s += &law("w", 11, 57, 103, 4 * n);
    s += &format!("    mut x: [f32; {nn}] <- a0;\n");
    if p.two {
        s += &format!("    mut y: [f32; {nn}] <- a1;\n");
    }
    s += &format!("    mut it: i32 <- 0;\n    loop {{\n        (it < {}) -> {{\n            -true-> {{\n", p.trips);
    let ind = "                ";
    if check.is_some() {
        s += &format!("{ind}x -> xp;\n");
    }
    let mut arrs: Vec<String> = vec!["x".into()];
    if p.two {
        arrs.push("y".into());
    }
    arrs.push("w".into());
    let pick = |v: &[String], k: u8| v[k as usize % v.len()].clone();
    // A write into `x`: the natural guarded map over the cells, or one
    // `Update` per cell of `cells` (the new value of cell `c` is `val(c)`,
    // an expression of the index text `c`).
    let write = |z: &str, guard: String, cells: Vec<String>, val: &dyn Fn(&str) -> String| -> String {
        if natural {
            format!("{ind}ta -> map {{ t -> {guard} -> {{ -true-> {{ {} }} -false-> x[t]; }} }} -> x;\n", val("t"))
        } else {
            cells
                .iter()
                .enumerate()
                .map(|(k, c)| format!("{ind}{} -> {z}v{k};\n{ind}x[{c}] <- {z}v{k};\n", val(c)))
                .collect()
        }
    };
    let mut cur = "x".to_string();
    for (z, &[op, a, b]) in p.ops.iter().enumerate() {
        // (Most chains start at a site: a device unit to place.)
        let op = if z == 0 && op % 4 != 3 { 0 } else { op % 10 };
        let (x, y) = (cur.clone(), pick(&arrs, b));
        let z = format!("z{z}");
        // A row fold of the chain so far: one value per row (the vector a row
        // or column write reads).
        let rows = format!("{ind}krange -> map {{ j -> (0.0, krange) -> fold {{ acc, k -> acc + {x}[j * {n} + k] }} }} -> {z}r;\n");
        let made = match op {
            0 => Some(format!(
                "{ind}ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -> {z};\n"
            )),
            1 => Some(format!("{ind}{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n")),
            2 => Some(format!("{ind}{x} -> map {{ v -> v * 0.5 }} -> {z};\n")),
            3 => Some(format!("{ind}ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n")),
            4 => Some(format!("{ind}(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n", b % (p.trips + 1))),
            9 => {
                // A power of the clamped cell: `2^k` by `k` squarings.
                let k = 1 + a % 4;
                let pow = if natural {
                    format!("{k} -> iota -> sq; (v0, sq) -> fold {{ acc, q -> acc * acc }}")
                } else {
                    (0..k).map(|i| format!("v{i} * v{i} -> v{};", i + 1)).collect::<String>() + &format!(" v{k}")
                };
                Some(format!(
                    "{ind}{x} -> map {{ v -> v * 0.125 + 0.5 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> c; \
                     (c < 1.0) -> {{ -true-> c; -false-> 1.0; }} -> v0; {pow} }} -> {z};\n"
                ))
            }
            _ => None,
        };
        if let Some(text) = made {
            s += &text;
            arrs.push(z.clone());
            cur = z;
            continue;
        }
        // The writes into the state `x`, row `it` (`it < trips ≤ 4 < n`).
        s += &match op {
            5 => {
                // Row `it`, each cell fed by its column's row sum.
                let val = |c: &str| {
                    let j = if c == "t" { format!("t % {n}") } else { c.rsplit(" + ").next().unwrap().to_string() };
                    format!("x[{c}] * 0.5 + {z}r[{j}] / {n}.0")
                };
                let cells = (0..n).map(|j| format!("it * {n} + {j}")).collect();
                rows.clone() + &write(&z, format!("(t / {n} == it)"), cells, &val)
            }
            6 => {
                // Column `(it + a) % n`, each cell fed by its row's sum.
                let val = |c: &str| {
                    let i = if c == "t" { format!("t / {n}") } else { c.split(" * ").next().unwrap().to_string() };
                    format!("x[{c}] * 0.5 + {z}r[{i}] / {n}.0")
                };
                let cells = (0..n).map(|i| format!("{i} * {n} + {z}c")).collect();
                rows.clone()
                    + &format!("{ind}(it + {a}) % {n} -> {z}c;\n")
                    + &write(&z, format!("(t % {n} == {z}c)"), cells, &val)
            }
            7 => {
                // A range from row `it`: cells `[o, o + w)` of it, into the
                // next row when `o + w > n` (`it·n + 2n < n²`: in bounds).
                let o = u32::from(a) % (n / 2);
                let w = 1 + u32::from(b) % (2 * n - o);
                let val = |c: &str| format!("x[{c}] * 0.5 + 0.25");
                let cells = (o..o + w).map(|j| format!("it * {n} + {j}")).collect();
                write(&z, format!("(t >= it * {n} + {o} && t < it * {n} + {})", o + w), cells, &val)
            }
            _ => {
                // One cell of row `it`.
                let j = u32::from(a) % n;
                let val = |c: &str| format!("x[{c}] * 0.75 + 0.125");
                write(&z, format!("(t == it * {n} + {j})"), vec![format!("it * {n} + {j}")], &val)
            }
        };
        cur = "x".into();
    }
    if cur != "x" {
        s += &format!("{ind}{cur} -> x;\n");
    }
    if p.two {
        // `y`'s next: itself, `w`, `x`'s new value or the trip's first array.
        let first = arrs[3.min(arrs.len() - 1)].clone();
        let ny = pick(&["y".to_string(), "w".to_string(), "x".to_string(), first], p.next);
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    if let Some((a, b, len)) = check {
        let at = |c: i64| if c < 0 { format!("it * {a} - {}", -c) } else { format!("it * {a} + {c}") };
        s += &format!(
            "{ind}ta -> map {{ t -> (x[t] == xp[t]) -> {{ -true-> 0; -false-> 1; }} -> ch; \
             (t >= {} && t < {}) -> {{ -true-> 0; -false-> ch; }} }} -> dz;\n\
             {ind}(0, dz) -> fold {{ acc, v -> acc + v }} -> nd;\n{ind}\"dz=\" -> print;\n{ind}nd -> println;\n",
            at(b),
            at(b + len as i64)
        );
    }
    let out = if p.exit && p.two { "y" } else { "x" };
    s += &format!("{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> {out} -> out;\n        }}\n    }}\n");
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", nn - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}

// --- recurrence-precision: row-recurrent loops, as Mapal source --------------

/// A counted loop over a row-written state `x: [f32; n·n]` and a row state
/// `o: [f32; n]` (`n` 16 or 32, 2..6 trips): trip `it` writes row `it` of `x`
/// naturally (a guarded map over the cells), its cells a closed form of the
/// cell — plus `o[t % n]` (the previous trip's row: a recurrence), or, with a
/// `prompt` P, only once `it ≥ P` (a guard on the counter). Then `ops` over
/// the trip's current matrix (from `x`) and vector (a closed form of the
/// counter): affine sites against a loop invariant `w`, per-cell matrix maps,
/// guarded row scores (`r <= it·c`: a nest cell over the counter), whole folds
/// (sum, max, min) of the vector read back per cell as a number,
/// vector·matrix nest cells, and row folds (sum, max) of the matrix. The
/// last vector is `o`'s next; the exit reads
/// `o` by one element or whole.
#[derive(Clone, Debug)]
pub struct RecurProg {
    pub big: bool,
    pub trips: u8,
    pub prompt: u8,
    pub ops: Vec<[u8; 2]>,
    pub whole: bool,
}

pub fn recur_prog_strategy() -> impl Strategy<Value = RecurProg> {
    (
        any::<bool>(),
        2u8..7,
        any::<u8>(),
        prop::collection::vec(prop::array::uniform2(any::<u8>()), 2..7),
        any::<bool>(),
    )
        .prop_map(|(big, trips, p, ops, whole)| RecurProg {
            big,
            trips,
            // Half the programs have no prompt guard.
            prompt: if p % 2 == 0 { 0 } else { 1 + (p / 2) % trips },
            ops,
            whole,
        })
}

/// [`RecurProg`]'s source text.
pub fn recur_src(p: &RecurProg) -> String {
    let n: u32 = if p.big { 32 } else { 16 };
    let nn = n * n;
    let ind = "                ";
    let mut s = format!(
        "fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n    \
         ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / {}.0 }} -> w;\n    \
         (0.0, {nn}) -> fill -> x0;\n    (0.0, {n}) -> fill -> o0;\n    \
         mut x: [f32; {nn}] <- x0;\n    mut o: [f32; {n}] <- o0;\n    mut it: i32 <- 0;\n    loop {{\n        \
         (it < {}) -> {{\n            -true-> {{\n",
        4 * n,
        p.trips
    );
    let fed = match p.prompt {
        0 => format!("f / 64.0 + o[t % {n}]"),
        k => format!("(it < {k}) -> {{ -true-> f / 64.0; -false-> f / 64.0 + o[t % {n}]; }}"),
    };
    s += &format!(
        "{ind}ta -> map {{ t -> (t / {n} == it) -> {{ -true-> {{ (t * 7 + 13) % 101 - 50 -> widen_f32 -> f; {fed} }} -false-> x[t]; }} }} -> x;\n"
    );
    s += &format!("{ind}krange -> map {{ k -> ((it * {n} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / {}.0 }} -> q;\n", 8 * n);
    let (mut mat, mut vec) = ("x".to_string(), "q".to_string());
    for (z, &[op, a]) in p.ops.iter().enumerate() {
        let z = format!("z{z}");
        s += ind;
        match op % 6 {
            0 => {
                s += &format!(
                    "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {mat}[i * {n} + k] * w[k * {n} + j] }} }} -> {z};\n"
                );
                mat = z;
            }
            1 => {
                s += &if a % 2 == 0 {
                    format!("{mat} -> map {{ v -> (v > 0.0) -> {{ -true-> v; -false-> 0.0; }} }} -> {z};\n")
                } else {
                    format!("{mat} -> map {{ v -> v * 0.5 }} -> {z};\n")
                };
                mat = z;
            }
            2 => {
                let c = u32::from(a % 3) + 1;
                s += &format!(
                    "krange -> map {{ r -> (r <= it * {c}) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + {vec}[k] * {mat}[r * {n} + k] }} }} -false-> -1000.0; }} }} -> {z};\n"
                );
                vec = z;
            }
            3 => {
                let (seed, step) = match a % 3 {
                    0 => ("0.0", "acc + e"),
                    1 => ("-1000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                    _ => ("1000000.0", "(e < acc) -> { -true-> e; -false-> acc; }"),
                };
                s += &format!("({seed}, krange) -> fold {{ acc, r -> {vec}[r] -> e; {step} }} -> {z}m;\n");
                s += &format!("{ind}krange -> map {{ r -> ({vec}[r] - {z}m) / 16.0 }} -> {z};\n");
                vec = z;
            }
            4 => {
                s += &format!(
                    "krange -> map {{ j -> (0.0, krange) -> fold {{ acc, r -> acc + {vec}[r] * {mat}[r * {n} + j] }} -> a; a * 0.25 }} -> {z};\n"
                );
                vec = z;
            }
            _ => {
                // A row fold of the matrix (a sum, or a running max).
                s += &if a % 2 == 0 {
                    format!("krange -> map {{ i -> (0.0, krange) -> fold {{ acc, k -> acc + {mat}[i * {n} + k] }} }} -> {z};\n")
                } else {
                    format!(
                        "krange -> map {{ i -> (-1000000.0, krange) -> fold {{ acc, k -> {mat}[i * {n} + k] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> {z};\n"
                    )
                };
                vec = z;
            }
        }
    }
    if vec == "q" {
        s += &format!("{ind}krange -> map {{ j -> {mat}[j] * 0.5 + q[j] }} -> o;\n");
    } else {
        s += &format!("{ind}{vec} -> o;\n");
    }
    s += &format!("{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n");
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", n - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}
