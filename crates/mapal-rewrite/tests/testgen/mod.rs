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

/// Generated programs a test binary checks at once, across its tests:
/// `MAPAL_TESTGEN_JOBS`, default a quarter of the CPUs this process may run
/// on (CLAUDE.md's thermal rule: each check starts children of its own —
/// clang, emulated programs — which the backend tests bound per binary by
/// `emu::job_slots`).
pub fn jobs() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MAPAL_TESTGEN_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                (std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).max(1)
            })
    })
}

/// Programs being checked in this test binary, across its tests (at most
/// [`jobs`]).
static CHECKING: (std::sync::Mutex<usize>, std::sync::Condvar) =
    (std::sync::Mutex::new(0), std::sync::Condvar::new());

/// One of the binary's [`jobs`] program slots, held while a program is
/// checked: tests running side by side (`RUST_TEST_THREADS`) share them.
struct Checking;

impl Checking {
    fn take() -> Checking {
        let (m, cv) = &CHECKING;
        let n = m.lock().unwrap_or_else(|e| e.into_inner());
        let mut n = cv
            .wait_while(n, |n| *n >= jobs())
            .unwrap_or_else(|e| e.into_inner());
        *n += 1;
        Checking
    }
}

impl Drop for Checking {
    fn drop(&mut self) {
        let (m, cv) = &CHECKING;
        *m.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        cv.notify_one();
    }
}

/// `check(i, &items[i])` for every item, at most [`jobs`] programs at once in
/// the whole test binary, each worker taking the next index not yet started
/// (a slow program holds no others behind it). The items are drawn before the
/// call, from the test's deterministic runner, so the programs are the serial
/// run's; the results come back in index order, so every counter summed over
/// them is the serial run's at any job count. A failing check stops new ones
/// from starting; once the running ones end, the lowest-index failure is
/// raised again (`#i` named on stderr).
pub fn par_check<T: Sync, R: Send>(items: &[T], check: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    let (next, stop) = (AtomicUsize::new(0), AtomicBool::new(false));
    let done: Vec<std::sync::Mutex<Option<std::thread::Result<R>>>> =
        items.iter().map(|_| Default::default()).collect();
    std::thread::scope(|s| {
        for _ in 0..jobs().min(items.len()) {
            s.spawn(|| {
                while !stop.load(Relaxed) {
                    let _slot = Checking::take();
                    if stop.load(Relaxed) {
                        break;
                    }
                    let i = next.fetch_add(1, Relaxed);
                    let Some(item) = items.get(i) else {
                        break;
                    };
                    let r =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(i, item)));
                    stop.fetch_or(r.is_err(), Relaxed);
                    *done[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
                }
            });
        }
    });
    let mut out = Vec::with_capacity(items.len());
    for (i, d) in done.into_iter().enumerate() {
        match d.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some(Ok(r)) => out.push(r),
            Some(Err(e)) => {
                eprintln!("generated program #{i} of {} failed", items.len());
                std::panic::resume_unwind(e)
            }
            // Never started: a lower index failed first.
            None => {}
        }
    }
    out
}

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
        format!(
            "    ta -> map {{ t -> (t * {a} + {b}) % {m} - {h} -> widen_f32 -> v; v / {scale}.0 }} -> {name};\n",
            h = m / 2
        )
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
            1 => format!(
                "{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n"
            ),
            2 => format!("{x} -> map {{ v -> v * 0.5 }} -> {z};\n"),
            3 => format!("ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n"),
            4 => format!(
                "(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n",
                b % (p.trips + 1)
            ),
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
                format!(
                    "{x}[it * {n} + {j}] * 0.5 + 0.25 -> {z}v;\n{ind}x[it * {n} + {j}] <- {z}v;\n"
                )
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
        let ny = pick(
            &["y".to_string(), "w".to_string(), "x".to_string(), first],
            p.next[1],
        );
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    let out = if p.exit && p.two { "y" } else { "x" };
    s += &format!(
        "{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> {out} -> out;\n        }}\n    }}\n"
    );
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
    /// Also a row value broadcast along its row beside the column index
    /// (`v[t ÷ c] + j / 8`: a cell that reads no row of its grid).
    pub broadcast: bool,
    /// Every closed-form operand (and iota) made first, before the nodes —
    /// a program that loads its weights before it computes.
    pub hoist: bool,
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
            broadcast: false,
            hoist: false,
        })
}

/// [`chain_strategy`]'s grammar on rows up to 128 cells wide and `k` up to
/// 64 (compile-time-graph A): most rows a power of two of 4..128 cells (a
/// narrow row's tile splits `k` across the row's spare lanes; a wide row
/// spans several warps, and so do a row-block unit's row groups and tree
/// folds), the rest any width in 4..128; with the row
/// broadcast node ([`Chain::broadcast`]), half of them with their operands
/// made first ([`Chain::hoist`]).
pub fn wide_chain_strategy() -> impl Strategy<Value = Chain> {
    (
        4u8..17,
        prop_oneof![
            3 => prop::sample::select(vec![4u8, 8, 16, 32, 64, 128]),
            1 => 4u8..17,
            1 => 17u8..129,
        ],
        2u8..65,
        2u8..6,
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..10),
        any::<bool>(),
    )
        .prop_map(|(r, c, k, taps, nodes, hoist)| Chain {
            r,
            c,
            k,
            taps,
            nodes,
            broadcast: true,
            hoist,
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
    /// [`Chain::hoist`]: the iotas and laws, emitted before the nodes.
    pre: Option<String>,
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
            let line = format!("    {n} -> iota -> ix{n};\n");
            match &mut self.pre {
                Some(pre) => *pre += &line,
                None => self.s += &line,
            }
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
        let line = format!(
            "    {it} -> map {{ t -> (t * {mul} + {add}) % {m} - {h} -> widen_f32 -> v; v / {div:.1} }} -> {name};\n",
            h = m / 2
        );
        match &mut self.pre {
            Some(pre) => *pre += &line,
            None => self.s += &line,
        }
        self.pool.push((name.clone(), sh));
        name
    }

    /// An array of shape `sh` made so far (by `pick`), else a fresh law.
    fn pick(&mut self, sh: Shape, pick: u8) -> String {
        let have: Vec<String> = self
            .pool
            .iter()
            .filter(|(_, s)| *s == sh)
            .map(|(n, _)| n.clone())
            .collect();
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
            s: String::new(),
            pool: Vec::new(),
            iotas: Vec::new(),
            laws: 0,
            pre: self.hoist.then(String::new),
        };
        let rc = r * c;
        let mut printed: Vec<String> = Vec::new();
        let last = self.nodes.len() - 1;
        for (z, &[op, p, q, flags]) in self.nodes.iter().enumerate() {
            let o = format!("n{z}");
            let tm = x.iota(rc);
            // Most nodes read an `r × c` array: start from a tile site.
            let ops = if self.broadcast { 17 } else { 16 };
            let op = if z == 0 && op % 3 != 0 { 6 } else { op % ops };
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
                        1 => {
                            "v -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> w; w".to_string()
                        }
                        2 => "v -> u; (u > 1.5) -> { -true-> 1.5; -false-> u; } -> w; w * 2.0"
                            .to_string(),
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
                        let ai = if al {
                            format!("{a}[k * {r} + i]")
                        } else {
                            format!("{a}[i * {k} + k]")
                        };
                        let bi = if bl {
                            format!("{b}[j * {k} + k]")
                        } else {
                            format!("{b}[k * {c} + j]")
                        };
                        (x.iota(k), ai, bi)
                    } else {
                        let a = x.pick(Shape::M, p);
                        let b = x.pick(Shape::Q, q);
                        let ai = format!("{a}[i * {c} + k]");
                        let bi = format!("{b}[k * {c} + j]");
                        (x.iota(c), ai, bi)
                    };
                    let site = |o: &str, ai: &str, bi: &str| {
                        format!(
                            "    {tm} -> map {{ t ->\n        t / {c} -> i;\n        t % {c} -> j;\n        (0.0, {kr}) -> fold {{ acc, k -> acc + {ai} * {bi} }}\n    }} -> {o};\n"
                        )
                    };
                    if op == 6 && flags & 0xC0 == 0xC0 {
                        // plan-cost `linear_plan`: a sum of two folds over one
                        // `k` range, each read only by the sum (operands may
                        // repeat: a shared `a` or `b`).
                        let (a2, b2) = (x.pick(Shape::A, p ^ 0x5a), x.pick(Shape::B, q ^ 0x33));
                        let (ai2, bi2) =
                            (format!("{a2}[i * {k} + k]"), format!("{b2}[k * {c} + j]"));
                        x.s += &site(&format!("{o}p"), &ai, &bi);
                        x.s += &site(&format!("{o}q"), &ai2, &bi2);
                        x.s += &format!("    {tm} -> map {{ t -> {o}p[t] + {o}q[t] }} -> {o};\n");
                    } else {
                        x.s += &site(&o, &ai, &bi);
                    }
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
                    x.s += &format!(
                        "    {tm} -> map {{ t -> (0.0, {tt}) -> fold {{ acc, k -> {e} }} }} -> {o};\n"
                    );
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
                16 => {
                    let v = x.pick(Shape::V, p);
                    x.s += &format!(
                        "    {tm} -> map {{ t -> t % {c} -> j; j -> widen_f32 -> f; {v}[t / {c}] + f / 8.0 }} -> {o};\n"
                    );
                    Shape::M
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
                    x.s += &format!(
                        "    (zero, {a}) -> fold {{ acc, v -> acc + v * 0.5 }} -> {o};\n    {o} -> println;\n"
                    );
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
            if let Some((o, sh)) = x
                .pool
                .iter()
                .rev()
                .find(|(o, _)| o.starts_with('n'))
                .cloned()
            {
                x.print(&o, sh);
            }
        }
        format!(
            "fn main() {{\n    0.0 -> zero: f32;\n{}{}}}\n",
            x.pre.unwrap_or_default(),
            x.s
        )
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
    /// Also a whole fold of the chain read as a number by the next map
    /// (attention's running max or denominator): device bytes one op of the
    /// trip makes and a later one reads.
    pub numbers: bool,
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
            numbers: false,
        })
}

/// [`twin_prog_strategy`] with whole-fold numbers ([`TwinProg::numbers`]).
pub fn twin_number_strategy() -> impl Strategy<Value = TwinProg> {
    twin_prog_strategy().prop_map(|p| TwinProg { numbers: true, ..p })
}

/// [`TwinProg`]'s source text, `natural` or unrolled. `check = (a, b, len)`
/// instruments the trip (the write-delta soundness probe): it prints
/// `dz=<k>`, the number of cells of the state `x` that changed in the trip
/// outside `[a·it + b, a·it + b + len)`.
pub fn twin_src(p: &TwinProg, natural: bool, check: Option<(i64, i64, u64)>) -> String {
    let n: u32 = if p.big { 32 } else { 16 };
    let nn = n * n;
    let law = |name: &str, a: u32, b: u32, m: u32, scale: u32| {
        format!(
            "    ta -> map {{ t -> (t * {a} + {b}) % {m} - {h} -> widen_f32 -> v; v / {scale}.0 }} -> {name};\n",
            h = m / 2
        )
    };
    let mut s = format!("fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n");
    s += &law("a0", 7, 13, 101, 8 * n);
    s += &law("a1", 5, 3, 97, 8 * n);
    s += &law("w", 11, 57, 103, 4 * n);
    s += &format!("    mut x: [f32; {nn}] <- a0;\n");
    if p.two {
        s += &format!("    mut y: [f32; {nn}] <- a1;\n");
    }
    s += &format!(
        "    mut it: i32 <- 0;\n    loop {{\n        (it < {}) -> {{\n            -true-> {{\n",
        p.trips
    );
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
    let write = |z: &str,
                 guard: String,
                 cells: Vec<String>,
                 val: &dyn Fn(&str) -> String|
     -> String {
        if natural {
            format!(
                "{ind}ta -> map {{ t -> {guard} -> {{ -true-> {{ {} }} -false-> x[t]; }} }} -> x;\n",
                val("t")
            )
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
        let ops = if p.numbers { 11 } else { 10 };
        let op = if z == 0 && op % 4 != 3 { 0 } else { op % ops };
        let (x, y) = (cur.clone(), pick(&arrs, b));
        let z = format!("z{z}");
        // A row fold of the chain so far: one value per row (the vector a row
        // or column write reads).
        let rows = format!(
            "{ind}krange -> map {{ j -> (0.0, krange) -> fold {{ acc, k -> acc + {x}[j * {n} + k] }} }} -> {z}r;\n"
        );
        let made = match op {
            0 => Some(format!(
                "{ind}ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -> {z};\n"
            )),
            1 => Some(format!(
                "{ind}{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n"
            )),
            2 => Some(format!("{ind}{x} -> map {{ v -> v * 0.5 }} -> {z};\n")),
            3 => Some(format!(
                "{ind}ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n"
            )),
            4 => Some(format!(
                "{ind}(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n",
                b % (p.trips + 1)
            )),
            9 => {
                // A power of the clamped cell: `2^k` by `k` squarings.
                let k = 1 + a % 4;
                let pow = if natural {
                    format!("{k} -> iota -> sq; (v0, sq) -> fold {{ acc, q -> acc * acc }}")
                } else {
                    (0..k)
                        .map(|i| format!("v{i} * v{i} -> v{};", i + 1))
                        .collect::<String>()
                        + &format!(" v{k}")
                };
                Some(format!(
                    "{ind}{x} -> map {{ v -> v * 0.125 + 0.5 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> c; \
                     (c < 1.0) -> {{ -true-> c; -false-> 1.0; }} -> v0; {pow} }} -> {z};\n"
                ))
            }
            10 => {
                // A whole fold of the chain (a sum or a max), read as a
                // number by every cell of the next map.
                let (seed, step) = match a % 2 {
                    0 => ("0.0", "acc + e"),
                    _ => ("-1000000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                };
                Some(format!(
                    "{ind}({seed}, ta) -> fold {{ acc, t -> {x}[t] -> e; {step} }} -> {z}s;\n\
                     {ind}ta -> map {{ t -> {x}[t] - {z}s / {nn}.0 }} -> {z};\n"
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
                    let j = if c == "t" {
                        format!("t % {n}")
                    } else {
                        c.rsplit(" + ").next().unwrap().to_string()
                    };
                    format!("x[{c}] * 0.5 + {z}r[{j}] / {n}.0")
                };
                let cells = (0..n).map(|j| format!("it * {n} + {j}")).collect();
                rows.clone() + &write(&z, format!("(t / {n} == it)"), cells, &val)
            }
            6 => {
                // Column `(it + a) % n`, each cell fed by its row's sum.
                let val = |c: &str| {
                    let i = if c == "t" {
                        format!("t / {n}")
                    } else {
                        c.split(" * ").next().unwrap().to_string()
                    };
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
                write(
                    &z,
                    format!("(t >= it * {n} + {o} && t < it * {n} + {})", o + w),
                    cells,
                    &val,
                )
            }
            _ => {
                // One cell of row `it`.
                let j = u32::from(a) % n;
                let val = |c: &str| format!("x[{c}] * 0.75 + 0.125");
                write(
                    &z,
                    format!("(t == it * {n} + {j})"),
                    vec![format!("it * {n} + {j}")],
                    &val,
                )
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
        let ny = pick(
            &["y".to_string(), "w".to_string(), "x".to_string(), first],
            p.next,
        );
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    if let Some((a, b, len)) = check {
        let at = |c: i64| {
            if c < 0 {
                format!("it * {a} - {}", -c)
            } else {
                format!("it * {a} + {c}")
            }
        };
        s += &format!(
            "{ind}ta -> map {{ t -> (x[t] == xp[t]) -> {{ -true-> 0; -false-> 1; }} -> ch; \
             (t >= {} && t < {}) -> {{ -true-> 0; -false-> ch; }} }} -> dz;\n\
             {ind}(0, dz) -> fold {{ acc, v -> acc + v }} -> nd;\n{ind}\"dz=\" -> print;\n{ind}nd -> println;\n",
            at(b),
            at(b + len as i64)
        );
    }
    let out = if p.exit && p.two { "y" } else { "x" };
    s += &format!(
        "{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> {out} -> out;\n        }}\n    }}\n"
    );
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
/// `o` by one element or whole. `var` varies the row write the rungs were
/// built on: bit 0 — the prompt cell reads the loop invariant `w` (not a
/// closed form); bit 1 — the trip writes row `it + 1` (`b ≠ 0`); bits 2..3
/// zero (with a prompt) — an int division by `it − c` traps at an early
/// prompt trip `c`; bit 4 — a guard inside the written cell (`t % 3`); bit 5
/// — the trip writes the row's first half only (`len < d`); bit 6 — a second
/// row-written state `y` (two batched slots), read by `o`.
#[derive(Clone, Debug)]
pub struct RecurProg {
    pub big: bool,
    pub trips: u8,
    pub prompt: u8,
    pub ops: Vec<[u8; 2]>,
    pub whole: bool,
    pub var: u8,
}

pub fn recur_prog_strategy() -> impl Strategy<Value = RecurProg> {
    (
        any::<bool>(),
        2u8..7,
        any::<u8>(),
        prop::collection::vec(prop::array::uniform2(any::<u8>()), 2..7),
        any::<bool>(),
        any::<u8>(),
    )
        .prop_map(|(big, trips, p, ops, whole, var)| RecurProg {
            big,
            trips,
            // Half the programs have no prompt guard.
            prompt: if p % 2 == 0 { 0 } else { 1 + (p / 2) % trips },
            ops,
            whole,
            var,
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
         mut x: [f32; {nn}] <- x0;\n{}    mut o: [f32; {n}] <- o0;\n    mut it: i32 <- 0;\n    loop {{\n        \
         (it < {}) -> {{\n            -true-> {{\n",
        4 * n,
        if p.var & 64 != 0 {
            format!("    (0.0, {nn}) -> fill -> y0;\n    mut y: [f32; {nn}] <- y0;\n")
        } else {
            String::new()
        },
        p.trips
    );
    let fed = |f: &str| match p.prompt {
        0 => format!("{f} / 64.0 + o[t % {n}]"),
        k => format!("(it < {k}) -> {{ -true-> {f} / 64.0; -false-> {f} / 64.0 + o[t % {n}]; }}"),
    };
    let cell = match p.var & 1 {
        0 => "(t * 7 + 13) % 101 - 50 -> widen_f32 -> f;".to_string(),
        // Reversed: not the cell's own index, so `w` is an operand (not `cell-law`).
        _ => format!("w[{} - t] * 64.0 -> f;", nn - 1),
    };
    let body = match p.var & 16 {
        0 => format!("{cell} {}", fed("f")),
        _ => format!(
            "{cell} (t % 3 == 0) -> {{ -true-> {{ {} }} -false-> f / 32.0; }}",
            fed("f")
        ),
    };
    let row = if p.var & 2 == 0 { "it" } else { "it + 1" };
    // The row's cells the trip writes: all, or (bit 5) its first half.
    let write = |body: &str, st: &str| match p.var & 32 {
        0 => format!("{{ {body} }}"),
        _ => format!(
            "{{ (t % {n} < {}) -> {{ -true-> {{ {body} }} -false-> {st}[t]; }} }}",
            n / 2
        ),
    };
    s += &format!(
        "{ind}ta -> map {{ t -> (t / {n} == {row}) -> {{ -true-> {} -false-> x[t]; }} }} -> x;\n",
        write(&body, "x")
    );
    if p.var & 64 != 0 {
        let g = format!("(t * 3 + 5) % 89 - 44 -> widen_f32 -> g; {}", fed("g"));
        s += &format!(
            "{ind}ta -> map {{ t -> (t / {n} == {row}) -> {{ -true-> {} -false-> y[t]; }} }} -> y;\n",
            write(&g, "y")
        );
    }
    let trap = (p.prompt > 0 && (p.var >> 2) & 3 == 0).then(|| (p.var >> 4) % p.prompt);
    let (dzs, dzt) = match trap {
        Some(c) => {
            s += &format!("{ind}100 / (it - {c}) -> dz;\n");
            (" dz -> widen_f32 -> dq;", " + dq / 1000.0")
        }
        None => ("", ""),
    };
    s += &format!(
        "{ind}krange -> map {{ k -> ((it * {n} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv;{dzs} qv / {}.0{dzt} }} -> q;\n",
        8 * n
    );
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
                    format!(
                        "{mat} -> map {{ v -> (v > 0.0) -> {{ -true-> v; -false-> 0.0; }} }} -> {z};\n"
                    )
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
                s += &format!(
                    "({seed}, krange) -> fold {{ acc, r -> {vec}[r] -> e; {step} }} -> {z}m;\n"
                );
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
                    format!(
                        "krange -> map {{ i -> (0.0, krange) -> fold {{ acc, k -> acc + {mat}[i * {n} + k] }} }} -> {z};\n"
                    )
                } else {
                    format!(
                        "krange -> map {{ i -> (-1000000.0, krange) -> fold {{ acc, k -> {mat}[i * {n} + k] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> {z};\n"
                    )
                };
                vec = z;
            }
        }
    }
    let last = if vec == "q" {
        format!("{mat}[j] * 0.5 + q[j]")
    } else {
        format!("{vec}[j]")
    };
    if p.var & 64 != 0 {
        // `y` read by a vector·matrix cell.
        s += &format!(
            "{ind}krange -> map {{ j -> (0.0, krange) -> fold {{ acc, r -> acc + {vec}[r] * y[r * {n} + j] }} -> a; a * 0.25 + {last} }} -> o;\n"
        );
    } else if vec == "q" {
        s += &format!("{ind}krange -> map {{ j -> {last} }} -> o;\n");
    } else {
        s += &format!("{ind}{vec} -> o;\n");
    }
    s += &format!(
        "{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n"
    );
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", n - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}

// --- recurrence-precision `radix-tile`: mixed-radix map/fold sites ----------

/// A multi-channel convolution `y[t] = Σ_k w[·]·img[·]` over constant div/mod
/// splits of the cell `t` and the tap `k`: batch `b` (1–2), `cin` (1–4) input
/// and `cout` (4/8/16) output channels, an `h × w` image (4/6/8 each side),
/// `kh × kw` taps (1–3); the output NCHW or NHWC, the weights `[co][k]` or
/// `[k][co]`, `k` channel-major or tap-major. `mix` reads the image at a
/// channel shifted by the output channel (`(ci + co) % cin`): a digit of `t`
/// read by both operands, which no tile admits. Built as source text.
#[derive(Clone, Debug)]
pub struct RadixProg {
    pub dims: [u8; 6],
    pub nhwc: bool,
    pub w_kio: bool,
    pub tap_major: bool,
    pub mix: bool,
}

pub fn radix_prog_strategy() -> impl Strategy<Value = RadixProg> {
    (
        prop::array::uniform6(any::<u8>()),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop::bool::weighted(0.2),
    )
        .prop_map(|(dims, nhwc, w_kio, tap_major, mix)| RadixProg {
            dims,
            nhwc,
            w_kio,
            tap_major,
            mix,
        })
}

/// [`RadixProg`]'s source text and its output length.
pub fn radix_src(p: &RadixProg) -> (String, u64) {
    let d = p.dims;
    let b = u64::from(d[0] % 2) + 1;
    // (`mix` needs two channels: `(ci + co) % 1` folds to a constant.)
    let cin = 1u64 << (d[1] % 5).max(u8::from(p.mix));
    let cout = 4u64 << (d[2] % 5);
    let (h, w) = (
        [4, 8, 6][usize::from(d[3] % 3)],
        [4, 8, 6][usize::from(d[4] % 3)],
    );
    let (kh, kw) = (u64::from(d[5] % 4) + 1, u64::from(d[5] / 4 % 4) + 1);
    let (ih, iw, kk) = (h + kh - 1, w + kw - 1, kh * kw);
    let (n, k) = (b * cout * h * w, cin * kk);
    let digits = if p.nhwc {
        format!(
            "t / {} -> b; t / {} % {h} -> i; t / {cout} % {w} -> j; t % {cout} -> co;",
            h * w * cout,
            w * cout
        )
    } else {
        format!(
            "t / {} -> b; t / {} % {cout} -> co; t / {w} % {h} -> i; t % {w} -> j;",
            cout * h * w,
            h * w
        )
    };
    let taps = if p.tap_major {
        format!("k / {cin} -> q; k % {cin} -> ci;")
    } else {
        format!("k / {kk} -> ci; k % {kk} -> q;")
    };
    let wa = if p.w_kio {
        format!("k * {cout} + co")
    } else {
        format!("co * {k} + k")
    };
    let ch = if p.mix {
        format!("(ci + co) % {cin}")
    } else {
        "ci".to_string()
    };
    let src = format!(
        "fn main() {{
    {wn} -> iota -> i_w;
    i_w -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / 11.0 }} -> wt;
    {imn} -> iota -> i_img;
    i_img -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> img;
    {n} -> iota -> o_y;
    {k} -> iota -> k_y;
    o_y -> map {{ t ->
        {digits}
        (0.0, k_y) -> fold {{ acc, k -> {taps} acc + wt[{wa}] * img[((b * {cin} + {ch}) * {ih} + i + q / {kw}) * {iw} + j + q % {kw}] }}
    }} -> y;
    y[0] -> println;
    y[{last}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        wn = cout * k,
        imn = b * cin * ih * iw,
        last = n - 1
    );
    (src, n)
}

// --- recurrence-precision `tf32-tile`: products admitting TF32 (ADR-0041) ---

/// An `m × k · k × n` product `y` whose fold's products carry a compute type
/// (`form`): 0 the step (`-> fold: tf32 | f32`), 1 the function
/// (`fn main(): tf32 | f32`), 2 the function with the step restoring `: f32`,
/// 3 none, 4 a `seq: tf32 | f32` block around the map. `bt` reads `b`
/// transposed. Beside it `mag`, the same cells' `M = Σ_k |a·b|` (ADR-0040's
/// magnitude) under `: f32`, so a value check needs no oracle of its own.
#[derive(Clone, Debug)]
pub struct Tf32Prog {
    pub dims: [u8; 3],
    pub bt: bool,
    pub form: u8,
    pub law: [u8; 2],
}

pub fn tf32_prog_strategy() -> impl Strategy<Value = Tf32Prog> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.25),
        prop::sample::select(vec![0u8, 0, 1, 1, 4, 4, 2, 3]),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(dims, bt, form, law)| Tf32Prog {
            dims,
            bt,
            form,
            law,
        })
}

/// Whether [`Tf32Prog`]'s product `y` admits TF32.
pub fn tf32_admits(p: &Tf32Prog) -> bool {
    matches!(p.form, 0 | 1 | 4)
}

/// [`Tf32Prog`]'s source text and its `(m·n·k, k)`. Prints two cells of `y`,
/// each followed by its `M`.
pub fn tf32_src(p: &Tf32Prog) -> (String, (u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(p.dims[0], &[16, 32, 64, 128]);
    let n = pick(p.dims[1], &[8, 16, 32, 64]);
    let k = pick(p.dims[2], &[8, 16, 32, 64, 128]);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[j * {k} + k]")
    } else {
        format!("b[k * {n} + j]")
    };
    let (fn_ty, step_ty) = match p.form {
        0 => ("", ": tf32 | f32"),
        1 => (": tf32 | f32", ""),
        2 => (": tf32 | f32", ": f32"),
        _ => ("", ""),
    };
    let y = format!(
        "ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{step_ty} {{ acc, k -> acc + a[i * {k} + k] * {bi} }} }}"
    );
    let y = if p.form == 4 {
        format!("ic -> seq: tf32 | f32 {{ {y} }}")
    } else {
        y
    };
    let (c0, c1) = (
        u64::from(p.law[0]) * 7 % (m * n),
        m * n - 1 - u64::from(p.law[1]) % n,
    );
    let src = format!(
        "fn main(){fn_ty} {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    {y} -> y;
    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> a[i * {k} + k] * {bi} -> p; (p > 0.0) -> {{ -true-> p; -false-> 0.0 - p; }} -> q; acc + q }} }} -> mag;
    y[{c0}] -> println;
    mag[{c0}] -> println;
    y[{c1}] -> println;
    mag[{c1}] -> println;
}}
",
        mk = m * k,
        kn = k * n,
        mn = m * n
    );
    (src, (m * n * k, k))
}

// --- site-geometry: launch geometry of tile sites (`fill-wave`, `row-split`) --

/// An `m × k · k × n` product `y` with integer-valued operands — every partial
/// sum is exact in f32, so every cut of the fold prints the same bits — from a
/// skinny projection (`m` 1) to a small GEMM: `m` 1–64, `n` 32–512, `k`
/// 32–1024; `bt` reads `b` transposed. `retained`: the product runs in a loop
/// over a carried `x` (its `a`, `m ≥ 4` rows) whose trip `t` rewrites row `t`
/// and reads row `t` of `y` (decode's K/V rebuild: the retained one-row site).
/// `wide` (not retained): a few rows over many lanes — `m` 16–64, `n`
/// 16384–32768, `k` 32–64 (a convolution's implicit GEMM shape).
#[derive(Clone, Debug)]
pub struct GeomProg {
    pub dims: [u8; 3],
    pub bt: bool,
    pub retained: bool,
    pub law: [u8; 2],
    pub wide: bool,
    /// Operands `v / 7.0`, `v / 9.0`: partial sums round, so a cell whose `k`
    /// order moved prints other bits ([`order_prog_strategy`]).
    pub frac: bool,
    /// `f64` operands and products.
    pub f64: bool,
    /// `a` read along `i` (`a[q·m + i]`, `ci == 1`), not along `k`.
    pub ci: bool,
    /// `k = 8·odd` (40, 72, 136, 264): no power-of-two panel ≥ 16 divides it.
    pub k8: bool,
}

pub fn geom_prog_strategy() -> impl Strategy<Value = GeomProg> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.25),
        any::<bool>(),
        prop::array::uniform2(any::<u8>()),
        prop::bool::weighted(0.3),
    )
        .prop_map(|(dims, bt, retained, law, wide)| GeomProg {
            dims,
            bt,
            retained,
            law,
            wide: wide && !retained,
            frac: false,
            f64: false,
            ci: false,
            k8: false,
        })
}

/// [`geom_prog_strategy`] for order and layout oracles: non-integer operands
/// (`frac`) and, at random, `f64`, `ci == 1` and `k = 8·odd` (neither
/// retained nor wide: the interpreter is the oracle).
pub fn order_prog_strategy() -> impl Strategy<Value = GeomProg> {
    (
        geom_prog_strategy(),
        prop::bool::weighted(0.5),
        prop::bool::weighted(0.25),
        prop::bool::weighted(0.25),
    )
        .prop_map(|(p, f64, ci, k8)| GeomProg {
            retained: false,
            wide: false,
            frac: true,
            f64,
            ci,
            k8,
            ..p
        })
}

/// [`GeomProg`]'s source text and its `(m, n, k)`.
pub fn geom_src(p: &GeomProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(
        p.dims[0],
        if p.retained {
            &[4, 8, 16, 64]
        } else if p.wide {
            &[16, 32, 64]
        } else {
            &[1, 2, 4, 16, 64]
        },
    );
    let n = pick(
        p.dims[1],
        if p.wide {
            &[16384, 32768]
        } else {
            &[32, 64, 128, 256, 512]
        },
    );
    let k = pick(
        p.dims[2],
        if p.wide {
            &[32, 64]
        } else if p.k8 {
            &[40, 72, 136, 264]
        } else {
            &[32, 64, 256, 1024]
        },
    );
    geom_src_at(p, m, n, k)
}

/// tile-efficiency T1: [`GeomProg`] at mid extents — rows 64–256 (192 not a
/// power of two), lanes 256–512, `k` 512–2048 — where a full register tile
/// divides the site and the fold is deep enough to cut, still small enough
/// for the emulator. `retained` and `wide` are ignored.
pub fn mid_src(p: &GeomProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let q = GeomProg {
        retained: false,
        wide: false,
        ..p.clone()
    };
    let m = pick(p.dims[0], &[64, 128, 192, 256]);
    let n = pick(p.dims[1], &[256, 512]);
    let k = pick(p.dims[2], &[512, 1024, 2048]);
    geom_src_at(&q, m, n, k)
}

/// [`geom_src`] at the extents `(m, n, k)`.
fn geom_src_at(p: &GeomProg, m: u64, n: u64, k: u64) -> (String, (u64, u64, u64)) {
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let ai = match p.ci && !p.retained {
        true => format!("q * {m} + i"),
        false => format!("i * {k} + q"),
    };
    let y = |a: &str| {
        format!(
            "ic -> map {{ c -> c / {n} -> i; c % {n} -> j; (0.0, kr) -> fold {{ acc, q -> acc + {a}[{ai}] * {bi} }} }} -> y;"
        )
    };
    let (da, db) = match p.frac {
        true => (" -> v; v / 7.0", " -> v; v / 9.0"),
        false => ("", ""),
    };
    let head = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32{da} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32{db} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
",
        mk = m * k,
        kn = k * n,
        mn = m * n
    );
    let body = if p.retained {
        format!(
            "    {n} -> iota -> nr;
    (0.0, {n}) -> fill -> o0;
    mut x: [f32; {mk}] <- a;
    mut o: [f32; {n}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {trips}) -> {{
            -true-> {{
                ia -> map {{ c -> (c / {k} == t) -> {{ -true-> (c * 3 + t * 5) % 23 - 11 -> widen_f32; -false-> x[c]; }} }} -> x;
                {y}
                nr -> map {{ j -> y[t * {n} + j] }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
            mk = m * k,
            trips = m.min(4),
            y = y("x"),
            last = n - 1
        )
    } else {
        // site-geometry integrate: a third of the products feed a bias + relu
        // epilogue (the tile kernel's epilogue group: an ordered launch beside
        // a cut fold's chooser).
        let epi = if p.law[1] % 3 == 0 {
            format!(
                "\n    {n} -> iota -> nb;\n    nb -> map {{ t -> t % 7 - 3 -> widen_f32 }} -> bias;\n    \
                 ic -> map {{ c -> y[c] + bias[c % {n}] -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> z;"
            )
        } else {
            String::new()
        };
        let o = if epi.is_empty() { "y" } else { "z" };
        format!(
            "    {y}{epi}
    {o}[0] -> println;
    {o}[{last}] -> println;
    0.0 -> zero: f32;
    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
            y = y("a"),
            last = m * n - 1
        )
    };
    let src = head + &body;
    match p.f64 {
        true => (src.replace("f32", "f64"), (m, n, k)),
        false => (src, (m, n, k)),
    }
}

// --- plan-cost blind set: tile sites at bench sizes --------------------------

/// A tile-site program at bench sizes: one `m × k · k × n` product (`kind` 0,
/// optionally a bias + relu epilogue), a chain of two or three (1, 2: each
/// product's `n` is the next one's `k`), or a sum of products — two over one `k`
/// range (3), three sharing `a` (4), two over two `k` ranges (5) — between a
/// `time` pair (the `benches/algorithms/pt_*` timer). Operands are integers in
/// ±3; `bt` reads each `b` transposed.
#[derive(Clone, Debug)]
pub struct TileProg {
    pub kind: u8,
    pub dims: [u8; 6],
    pub bt: bool,
    pub epi: bool,
    pub law: u8,
}

pub fn tile_prog_strategy() -> impl Strategy<Value = TileProg> {
    (
        0u8..6,
        prop::array::uniform6(any::<u8>()),
        prop::bool::weighted(0.25),
        prop::bool::weighted(0.3),
        any::<u8>(),
    )
        .prop_map(|(kind, dims, bt, epi, law)| TileProg {
            kind,
            dims,
            bt,
            epi,
            law,
        })
}

/// [`TileProg`]'s source and its products' `(m, n, k)`; `small` maps every
/// extent above 1 to 3–9 (same program) for the interpreter.
pub fn tile_src(p: &TileProg, small: bool) -> (String, Vec<(u64, u64, u64)>) {
    const M: &[u64] = &[1, 1, 16, 16, 7, 100, 333, 512, 1000, 1024, 2048, 3001, 4096];
    const N: &[u64] = &[
        64, 96, 250, 512, 777, 1024, 2048, 3000, 4096, 5120, 11008, 18432,
    ];
    const K: &[u64] = &[64, 100, 256, 513, 1024, 2048, 4096, 5000, 8192];
    let ext = |v: &[u64], d: u8| match v[usize::from(d) % v.len()] {
        x if small && x > 1 => x % 7 + 3,
        x => x,
    };
    let (m, k, k2) = (ext(M, p.dims[0]), ext(K, p.dims[1]), ext(K, p.dims[5]));
    let n = [ext(N, p.dims[2]), ext(N, p.dims[3]), ext(N, p.dims[4])];
    // (out, a, b, k, n) per product.
    let prods: Vec<(&str, &str, &str, u64, u64)> = match p.kind {
        0 => vec![("y", "a", "b0", k, n[0])],
        1 | 2 => [
            ("y1", "a", "b0", k, n[0]),
            ("y2", "y1", "b1", n[0], n[1]),
            ("y3", "y2", "b2", n[1], n[2]),
        ][..usize::from(p.kind) + 1]
            .to_vec(),
        3 => vec![("y1", "a", "b0", k, n[0]), ("y2", "a2", "b1", k, n[0])],
        4 => vec![
            ("y1", "a", "b0", k, n[0]),
            ("y2", "a", "b1", k, n[0]),
            ("y3", "a", "b2", k, n[0]),
        ],
        _ => vec![("y1", "a", "b0", k, n[0]), ("y2", "a2", "b1", k2, n[0])],
    };
    let mut iotas: Vec<u64> = Vec::new();
    let mut laws = String::new();
    let mut made: Vec<&str> = Vec::new();
    let mut body = String::new();
    for &(y, a, b, kk, nn) in &prods {
        if a.starts_with('a') && !made.contains(&a) {
            made.push(a);
            tile_law(&mut laws, p.law, a, tile_iota(&mut iotas, m * kk));
        }
        tile_law(&mut laws, p.law, b, tile_iota(&mut iotas, kk * nn));
        let bi = if p.bt {
            format!("{b}[j * {kk} + q]")
        } else {
            format!("{b}[q * {nn} + j]")
        };
        body += &format!(
            "    {ic} -> map {{ c -> c / {nn} -> i; c % {nn} -> j; (0.0, {kr}) -> fold {{ acc, q -> acc + {a}[i * {kk} + q] * {bi} }} }} -> {y};\n",
            ic = tile_iota(&mut iotas, m * nn),
            kr = tile_iota(&mut iotas, kk)
        );
    }
    let last = prods.last().unwrap();
    let (cells, ic) = (m * last.4, tile_iota(&mut iotas, m * last.4));
    let mut out = last.0.to_string();
    if p.kind >= 3 {
        let terms: Vec<String> = prods.iter().map(|t| format!("{}[c]", t.0)).collect();
        body += &format!("    {ic} -> map {{ c -> {} }} -> z;\n", terms.join(" + "));
        out = "z".into();
    } else if p.epi {
        tile_law(&mut laws, p.law, "bias", tile_iota(&mut iotas, last.4));
        body += &format!(
            "    {ic} -> map {{ c -> {out}[c] + bias[c % {nn}] -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> z;\n",
            nn = last.4
        );
        out = "z".into();
    }
    let head: String = iotas
        .iter()
        .map(|n| format!("    {n} -> iota -> i{n};\n"))
        .collect();
    let src = format!(
        "fn main() {{\n{head}{laws}    () -> time -> t0;\n{body}    () -> time -> t1;\n    {out}[0] -> println;\n    {out}[{}] -> println;\n    \"iter ms=\" -> print;\n    t1 - t0 -> println;\n}}\n",
        cells - 1
    );
    (
        src,
        prods.iter().map(|&(_, _, _, kk, nn)| (m, nn, kk)).collect(),
    )
}

/// `i<n>` (`n -> iota`), declared once.
fn tile_iota(iotas: &mut Vec<u64>, n: u64) -> String {
    if !iotas.contains(&n) {
        iotas.push(n);
    }
    format!("i{n}")
}

/// `name` over the iota `it`: integers in ±3, a law per name and `key`.
fn tile_law(laws: &mut String, key: u8, name: &str, it: String) {
    let mul = [3, 5, 11, 13][(usize::from(key) + name.len() + laws.len()) % 4];
    *laws += &format!(
        "    {it} -> map {{ t -> (t * {mul} + {add}) % 7 - 3 -> widen_f32 }} -> {name};\n",
        add = name.len() * 2 + 1
    );
}

/// kernel-merging `zero-fold`: a program whose tile sites add into a zeroed
/// output (few output cells, a long `k`: the atomic split-K candidates),
/// after device cells, in a loop trip or straight-line. `steps` is the
/// sequence: a cell (`[0, reads_o, from]`: an elementwise map of the trip
/// counter, reading the carried output `o` when `reads_o` is odd, and an
/// earlier site's output when `from` names one) or a site (`[1, q, _]`: the
/// GEMV-like fold of cell `q` against the weights). The last site's output is
/// the next carried `o`; a cell reading `o` after an earlier cell is an access
/// to the next site's bytes between them (the rule must not fold before it).
#[derive(Clone, Debug)]
pub struct ZeroProg {
    pub looped: bool,
    pub trips: u8,
    /// Output rows and columns (`rows · c` cells), the fold's length.
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    pub steps: Vec<[u8; 3]>,
}

pub fn zero_prog_strategy() -> impl Strategy<Value = ZeroProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![1u64, 2, 4, 8]),
        prop::sample::select(vec![1u64, 8, 32, 64]),
        prop::sample::select(vec![256u64, 512, 1024]),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..7),
    )
        .prop_map(|(looped, trips, rows, c, k, steps)| ZeroProg {
            looped,
            trips,
            rows,
            c,
            k,
            steps,
        })
}

/// [`ZeroProg`]'s source text.
pub fn zero_src(p: &ZeroProg) -> String {
    let (r, c, k) = (p.rows, p.c, p.k);
    let rc = r * c;
    let mut s = format!(
        "fn main() {{\n    {kc} -> iota -> tw;\n    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 256.0 }} -> w;\n    {k} -> iota -> kr;\n    {rc} -> iota -> oc;\n    {rk} -> iota -> qc;\n",
        kc = k * c,
        rk = r * k
    );
    let ind = if p.looped { "                " } else { "    " };
    let mut body = String::new();
    let (mut cells, mut sites) = (0usize, 0usize);
    // Straight-line, the counter is a number the program computes (a cell
    // reading it is a kernel of its own, not a closed-form generator).
    let t = "t";
    let cell = |body: &mut String, cells: &mut usize, sites: usize, step: [u8; 3]| {
        let mut e = format!(
            "((({t} * {k} + e) * {}) + 3) % 97 - 48 -> widen_f32 -> v; v / 256.0",
            5 + step[1] % 3
        );
        if step[1] % 2 == 1 {
            e += &format!(" + o[e % {rc}]");
        }
        if sites > 0 && step[2] % 3 == 0 {
            e += &format!(" + y{}[e % {rc}] / 64.0", step[2] as usize % sites);
        }
        body.push_str(&format!("{ind}qc -> map {{ e -> {e} }} -> q{};\n", *cells));
        *cells += 1;
    };
    for &step in &p.steps {
        if step[0] % 2 == 0 {
            cell(&mut body, &mut cells, sites, step);
        } else {
            if cells == 0 {
                cell(&mut body, &mut cells, sites, [0, 0, 1]);
            }
            let q = step[1] as usize % cells;
            body.push_str(&format!(
                "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q{q}[i * {k} + kk] * w[kk * {c} + j] }} }} -> y{sites};\n"
            ));
            sites += 1;
        }
    }
    if sites == 0 {
        body.push_str(&format!(
            "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q0[i * {k} + kk] * w[kk * {c} + j] }} }} -> y0;\n"
        ));
        sites = 1;
    }
    // Every site's output reaches the result: the last is the next `o`, the
    // others add into it through one cell.
    // Read at shifted cells: no epilogue group takes a site's store (each
    // stays a site of its own, its output an arena member that may reuse a
    // dead operand's bytes).
    let last = sites - 1;
    let mut sum = format!("y{last}[e]");
    for y in 0..last {
        sum += &format!(" + y{y}[(e + {}) % {rc}] / 8.0", y + 1);
    }
    // One site: its output is the next state itself.
    let next = match sites {
        1 => format!("{ind}y0 -> o;\n"),
        _ => format!("{ind}oc -> map {{ e -> {sum} }} -> o;\n"),
    };
    if p.looped {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o0;\n    mut o: [f32; {rc}] <- o0;\n    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n{body}{next}{ind}t + 1 -> t;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n",
            p.trips
        );
    } else {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o: [f32; {rc}];\n    (0, kr) -> fold {{ acc, v -> acc + v % 2 }} -> t;\n{body}    oc -> map {{ e -> {sum} }} -> out;\n"
        );
    }
    s + &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        rc - 1
    )
}

/// kernel-merging `horizontal`: a program whose launches are independent
/// side by side, in a loop trip or straight-line. `steps` is the sequence: a
/// cell (`[0, salt, from]`: an elementwise map of the trip counter, reading
/// the carried `o` when `salt` is odd and an earlier site's output when
/// `from` names one — an edge that orders it after that site), a site (`[1,
/// q, _]`: the GEMV-like fold of cell `q` against the weights `w`, or `w2`
/// when `q` is odd), a row sum (`[2, y, _]`: each output row of site `y`
/// folded) — so cells reading only the counter, sites over different cells
/// and sums over different sites are unordered and adjacent. `kv` adds the
/// retained pattern in a loop (a KV cache's `x·Wk`, `x·Wv` over a state
/// written one row a trip, each site's row pass in its trip's branch).
#[derive(Clone, Debug)]
pub struct HzProg {
    pub looped: bool,
    pub trips: u8,
    /// Output rows and columns (`rows · c` cells), the fold's length.
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    pub kv: bool,
    pub steps: Vec<[u8; 3]>,
    /// open-rules: sites whose map body goes on past the fold (`tile-post`:
    /// `u · 0.5 + q[i · k]`) — an epilogue group's split takes a workspace
    /// only, so its table holds no candidate a persistent kernel runs.
    pub post: bool,
}

pub fn hz_prog_strategy() -> impl Strategy<Value = HzProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![4u64, 8]),
        prop::sample::select(vec![8u64, 32, 64]),
        prop::sample::select(vec![32u64, 64, 256]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..8),
    )
        .prop_map(|(looped, trips, rows, c, k, kv, steps)| HzProg {
            looped,
            trips,
            rows,
            c,
            k,
            kv: kv && looped,
            steps,
            post: false,
        })
}

/// kernel-merging `calls`: [`HzProg`]'s programs over a wider size range —
/// one output row or column up to eight, folds up to 1024 long — so the
/// chains of dependent launches (a cell reading a site, a site over that
/// cell, a row sum over a site) hold both unsplit sites and sites whose rule
/// candidate adds into a zeroed output (few cells, a long fold: split-K
/// atomic), beside the retained pattern's row passes. open-rules: and,
/// under `post`, sites whose body goes on past the fold (`tile-post`).
pub fn calls_prog_strategy() -> impl Strategy<Value = HzProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![1u64, 4, 8]),
        prop::sample::select(vec![1u64, 8, 32, 64]),
        prop::sample::select(vec![32u64, 64, 256, 1024]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..8),
        any::<bool>(),
    )
        .prop_map(|(looped, trips, rows, c, k, kv, steps, post)| HzProg {
            looped,
            trips,
            rows,
            c,
            k,
            kv: kv && looped,
            steps,
            post,
        })
}

/// [`HzProg`]'s source text.
pub fn hz_src(p: &HzProg) -> String {
    let (r, c, k) = (p.rows, p.c, p.k);
    let rc = r * c;
    let mut s = format!(
        "fn main() {{\n    {kc} -> iota -> tw;\n    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 256.0 }} -> w;\n    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 256.0 }} -> w2;\n    {k} -> iota -> kr;\n    {c} -> iota -> cr;\n    {r} -> iota -> rr;\n    {rc} -> iota -> oc;\n    {rk} -> iota -> qc;\n",
        kc = k * c,
        rk = r * k
    );
    let ind = if p.looped { "                " } else { "    " };
    let mut body = String::new();
    let (mut cells, mut sites, mut sums) = (0usize, 0usize, 0usize);
    let t = "t";
    let cell = |body: &mut String, cells: &mut usize, sites: usize, step: [u8; 3]| {
        let mut e = format!(
            "((({t} * {k} + e) * {}) + 3) % 97 - 48 -> widen_f32 -> v; v / 256.0",
            5 + step[1] % 3
        );
        if step[1] % 2 == 1 {
            e += &format!(" + o[e % {rc}]");
        }
        if sites > 0 && step[2] % 3 == 0 {
            e += &format!(" + y{}[e % {rc}] / 64.0", step[2] as usize % sites);
        }
        body.push_str(&format!("{ind}qc -> map {{ e -> {e} }} -> q{};\n", *cells));
        *cells += 1;
    };
    for &step in &p.steps {
        match step[0] % 3 {
            0 => cell(&mut body, &mut cells, sites, step),
            1 => {
                if cells == 0 {
                    cell(&mut body, &mut cells, sites, [0, 0, 1]);
                }
                let q = step[1] as usize % cells;
                let wt = if step[1] % 2 == 1 { "w2" } else { "w" };
                // `tile-post`: the cell's body goes on past its fold.
                let post = match p.post && step[2] % 2 == 1 {
                    true => format!(" -> u; u * 0.5 + q{q}[i * {k}]"),
                    false => String::new(),
                };
                body.push_str(&format!(
                    "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q{q}[i * {k} + kk] * {wt}[kk * {c} + j] }}{post} }} -> y{sites};\n"
                ));
                sites += 1;
            }
            _ if sites > 0 => {
                let y = step[1] as usize % sites;
                body.push_str(&format!(
                    "{ind}rr -> map {{ i -> (0.0, cr) -> fold {{ acc, j -> acc + y{y}[i * {c} + j] }} }} -> s{sums};\n"
                ));
                sums += 1;
            }
            _ => cell(&mut body, &mut cells, sites, step),
        }
    }
    if sites == 0 {
        if cells == 0 {
            cell(&mut body, &mut cells, sites, [0, 0, 1]);
        }
        body.push_str(&format!(
            "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q0[i * {k} + kk] * w[kk * {c} + j] }} }} -> y0;\n"
        ));
        sites = 1;
    }
    // The retained pattern: row `t` of `x` written this trip, both sites over
    // all of `x` (their row passes the trip's), read at shifted cells.
    let mut kv = String::new();
    if p.kv {
        kv = format!(
            "{ind}qc -> map {{ e -> (e / {k} == t) -> {{ -true-> {{ (e * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 16.0 + o[e % {rc}] }} -false-> x[e]; }} }} -> x;\n\
             {ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + x[i * {k} + kk] * w[kk * {c} + j] }} }} -> ka;\n\
             {ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + x[i * {k} + kk] * w2[kk * {c} + j] }} }} -> va;\n"
        );
    }
    // Every output reaches the result, each read at a shifted cell (no
    // epilogue takes a site's store).
    let mut sum = format!("y{}[e]", sites - 1);
    for y in 0..sites - 1 {
        sum += &format!(" + y{y}[(e + {}) % {rc}] / 8.0", y + 1);
    }
    for x in 0..sums {
        sum += &format!(" + s{x}[(e + {}) % {r}] / 16.0", x + 1);
    }
    if p.kv {
        sum += &format!(" + ka[(e + 3) % {rc}] / 32.0 + va[(e + 5) % {rc}] / 32.0");
    }
    if p.looped {
        let xs = if p.kv {
            format!(
                "    (0.0, {rk}) -> fill -> x0;\n    mut x: [f32; {rk}] <- x0;\n",
                rk = r * k
            )
        } else {
            String::new()
        };
        s += &format!(
            "    (0.0, {rc}) -> fill -> o0;\n{xs}    mut o: [f32; {rc}] <- o0;\n    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n{kv}{body}{ind}oc -> map {{ e -> {sum} }} -> o;\n{ind}t + 1 -> t;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n",
            p.trips
        );
    } else {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o: [f32; {rc}];\n    (0, kr) -> fold {{ acc, v -> acc + v % 2 }} -> t;\n{body}    oc -> map {{ e -> {sum} }} -> out;\n"
        );
    }
    s + &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        rc - 1
    )
}

/// long-rows: few rows of a long axis (`rows × len` cells) and a chain over
/// them — row folds (sum, max, min), per-cell maps of a row array and a
/// per-row value by every law `fold_combine` tells apart (`x · v`, `x ÷ v`,
/// `x ± v`, a polynomial of `x − v`, `v ÷ x`), per-row cells — the shapes a
/// long softmax, a norm or a logsumexp take. `nodes[n] = [kind, a, b, c]`
/// picks the operands among the arrays made so far; every fold and the last
/// cell print. `looped`: the chain runs in a two-trip loop over a carried
/// row array (its next value the chain's last cell).
#[derive(Clone, Debug)]
pub struct LongRow {
    pub rows: u64,
    pub len: u64,
    pub looped: bool,
    pub nodes: Vec<[u8; 4]>,
    /// A hash of `nodes` (no draw of its own: the other programs stay the
    /// stream's); a third of the programs (`mask % 3 == 0`) open with a
    /// masked softmax ([`LongRow::masked`]).
    pub mask: u8,
}

impl LongRow {
    /// The program opens with a softmax over `a0` masked to `−∞` past a
    /// per-row cut (rows wholly masked among them): the row max, `exp` of
    /// the shifted scaled score, its row sum (printed), before any loop. The
    /// other nodes are the unmasked program's.
    pub fn masked(&self) -> bool {
        self.mask % 3 == 0
    }
}

pub fn long_row_strategy() -> impl Strategy<Value = LongRow> {
    (
        prop::sample::select(vec![1u64, 2, 3, 4, 8]),
        prop::sample::select(vec![256u64, 512, 1024, 2048, 4096]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..9),
    )
        .prop_map(|(rows, len, looped, nodes)| LongRow {
            rows,
            len,
            looped,
            mask: nodes.iter().fold(7u8, |h, n| {
                h.wrapping_mul(31).wrapping_add(n[0] ^ n[3].rotate_left(3))
            }),
            nodes,
        })
}

/// [`LongRow`]'s source text.
pub fn long_row_src(p: &LongRow) -> String {
    let (r, l) = (p.rows, p.len);
    let n = r * l;
    let ind = if p.looped { "                " } else { "    " };
    // `a0` is a tile site's output (`q · W`, `d = 8`): the rows start on the
    // device, where the folds over them run.
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {r} -> iota -> rr;\n    {l} -> iota -> kl;\n    8 -> iota -> kd;\n    {rq} -> iota -> tq;\n    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; v / 16.0 }} -> q;\n    {lw} -> iota -> tw;\n    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> w;\n    t -> map {{ e -> e / {l} -> i; e % {l} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {l} + j] }} }} -> a0;\n    t -> map {{ e -> (e * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 64.0 }} -> b0;\n",
        rq = r * 8,
        lw = l * 8
    );
    // Row arrays (`a*`, all `n` cells, |x| ≲ 2) and per-row values (`v*`, one
    // a row; positive ones `p*`, ≥ 1).
    let mut rows_: Vec<String> = vec!["a0".into(), "b0".into()];
    let mut vals: Vec<String> = Vec::new();
    // ADR-0042: each row fold's `(value, folded array, max: true / min:
    // false)` — the shifts a softmax-like `exp` of a difference reads.
    let mut extremes: Vec<(String, String, bool)> = Vec::new();
    let mut pos: Vec<String> = Vec::new();
    let mut prints: Vec<String> = Vec::new();
    let mut body = String::new();
    if p.looped {
        rows_[0] = "x".into();
    }
    let (mut na, mut nv) = (0usize, 0usize);
    if p.masked() {
        // Row `i` keeps `cut − i · step` scores (`f32` `−1e39` is `−∞`):
        // slices wholly masked, one across the cut, rows with none.
        let cut = l * (1 + u64::from(p.mask / 3) % 3) / 4;
        let step = l / (1 + u64::from(p.mask / 9) % 4);
        s += &format!(
            "    t -> map {{ e -> e / {l} -> i; e % {l} -> j; (j + i * {step} < {cut}) -> {{ -true-> a0[e]; -false-> -1000000000000000000000000000000000000000.0; }} }} -> m0;\n    rr -> map {{ i -> (-1000000.0, kl) -> fold {{ acc, j -> m0[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }} }} -> vm;\n    t -> map {{ e -> (m0[e] - vm[e / {l}]) / 4.0 -> exp }} -> am;\n    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + am[i * {l} + j] }} }} -> sm;\n"
        );
    }
    let pick = |v: &[String], b: u8| v[b as usize % v.len()].clone();
    for (i, nd) in p.nodes.iter().enumerate() {
        let kind = if vals.is_empty() && nd[0] % 4 != 0 {
            0
        } else {
            nd[0] % 4
        };
        match kind {
            // A row fold: sum (scaled by the row), max or min.
            0 => {
                // Half the folds read the newest row array (the chain's tip).
                let a = match nd[1] % 2 {
                    0 => rows_.last().unwrap().clone(),
                    _ => pick(&rows_, nd[1] / 2),
                };
                let name = format!("v{nv}");
                nv += 1;
                let fold = match nd[2] % 4 {
                    0 | 3 => format!("(0.0, kl) -> fold {{ acc, j -> acc + {a}[i * {l} + j] }}"),
                    1 => {
                        extremes.push((name.clone(), a.clone(), true));
                        format!(
                            "(-1000000.0, kl) -> fold {{ acc, j -> {a}[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }}"
                        )
                    }
                    _ => {
                        extremes.push((name.clone(), a.clone(), false));
                        format!(
                            "(1000000.0, kl) -> fold {{ acc, j -> {a}[i * {l} + j] -> w; (w < acc) -> {{ -true-> w; -false-> acc; }} }}"
                        )
                    }
                };
                body += &format!("{ind}rr -> map {{ i -> {fold} }} -> {name};\n");
                prints.push(format!("{name}[{}]", nd[3] as u64 % r));
                vals.push(name);
            }
            // A per-cell map of a row array and a per-row value.
            1 | 2 => {
                let a = pick(&rows_, nd[1]);
                let v = pick(&vals, nd[2]);
                let at = format!("{v}[e / {l}]");
                // Mostly the law forms; a quarter of the cells relu or multiply
                // rows, or take `exp` of a difference with a row's max / min.
                let form = match nd[3] {
                    0..160 => nd[3] % 6,
                    160..224 if !extremes.is_empty() => 8 + nd[3] % 3,
                    _ => 6 + nd[3] % 2,
                };
                let e = match form {
                    0 => format!("{a}[e] * {at} / 64.0"),
                    1 if !pos.is_empty() => format!("{a}[e] / {}[e / {l}]", pick(&pos, nd[2])),
                    2 => format!("{a}[e] - {at}"),
                    3 => format!("{a}[e] + {at}"),
                    4 => format!(
                        "({a}[e] - {at}) / 64.0 -> z; (z > -8.0) -> {{ -true-> 1.0 + z * (1.0 + z * 0.5); -false-> 0.0; }}"
                    ),
                    5 if !pos.is_empty() => {
                        format!("{}[e / {l}] / ({a}[e] * {a}[e] + 1.0)", pick(&pos, nd[2]))
                    }
                    6 => format!(
                        "({a}[e] > 0.0) -> {{ -true-> {a}[e]; -false-> 0.0 - {a}[e] / 4.0; }}"
                    ),
                    // ADR-0042: a softmax's exponential of the row's own max
                    // (min) — the rescale law's form, scaled or not — and of
                    // a difference with another row value (a rescale whose
                    // shift is no extreme of `x`: a phase in the consumer).
                    8 | 9 => {
                        let (v, src, up) = extremes[nd[2] as usize % extremes.len()].clone();
                        let d = if up {
                            format!("{src}[e] - {v}[e / {l}]")
                        } else {
                            format!("{v}[e / {l}] - {src}[e]")
                        };
                        if form == 8 {
                            format!("{d} -> exp")
                        } else {
                            format!("({d}) / 4.0 -> exp")
                        }
                    }
                    10 => format!("({a}[e] - {at}) / 256.0 -> exp"),
                    _ => format!("{a}[e] * {}[e] / 2.0", pick(&rows_, nd[2])),
                };
                // The law forms stay as written (a squash would hide the
                // law); a product of rows is squashed to stay bounded.
                let e = match form {
                    6 | 7 => format!("{e} -> y; y / (y * y / 64.0 + 1.0)"),
                    _ => e,
                };
                let name = format!("a{}", na + 1);
                na += 1;
                body += &format!("{ind}t -> map {{ e -> {e} }} -> {name};\n");
                rows_.push(name);
            }
            // A per-row cell: a positive scale from a value.
            _ => {
                let v = pick(&vals, nd[1]);
                let name = format!("p{}", pos.len());
                body += &format!(
                    "{ind}rr -> map {{ i -> 1.0 + {v}[i] * {v}[i] / 4096.0 }} -> {name};\n"
                );
                pos.push(name.clone());
                vals.push(name);
            }
        }
        let _ = i;
    }
    let last = rows_.last().unwrap().clone();
    let mut tail = format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n",
        n / 2 + 3,
        n - 1
    );
    if p.masked() {
        tail += &format!("    sm[0] -> println;\n    sm[{}] -> println;\n", r - 1);
    }
    let prints: String = prints
        .iter()
        .map(|x| format!("    {x} -> println;\n"))
        .collect();
    if p.looped {
        // The next state reads the state (a carried array whose next value
        // reads nothing of it is another program: `long-rows.md` open item).
        let nexts =
            format!("{ind}t -> map {{ e -> x[e] / 4.0 + {last}[e] / 2.0 + a0[e] }} -> x;\n");
        s += &format!(
            "    mut x: [f32; {n}] <- b0;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 2) -> {{\n            -true-> {{\n{body}{nexts}{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> x -> out;\n        }}\n    }}\n"
        );
        s + &tail + "}\n"
    } else {
        s += &body;
        s += &format!("    {last} -> out;\n");
        s + &prints + &tail + "}\n"
    }
}

// ---------------------------------------------------------------------------
// onto-device: guarded indices, loop-carried counters as indices, loop exit
// values read in later bodies (`bounds_proof`'s conditions and loop ranges,
// the exit-binding capture, `carry-made`).
// ---------------------------------------------------------------------------

/// A program whose reads are in bounds only by the conditions that hold
/// where they run: a rotate_half-style partner read under a guard on `c % m`
/// (seven forms: both poles, swapped operands, a `Phi`-valued index, a nested
/// guard, a conjunction, and an unprovable gather that keeps its check), a
/// loop whose counter (`t0`, step `±s`, four continue-condition forms) indexes
/// a table by the step's position `base + t` (optionally under a guard), and
/// the loop's exit state and a second loop's exit counter read inside later
/// map bodies. Every value is a small integer in `f32`, so every face and
/// every order is exact.
#[derive(Clone, Debug)]
pub struct OntoProg {
    /// The map extent (a multiple of `m`).
    pub n: u64,
    /// The guard modulus (a power of two).
    pub m: u64,
    /// The guarded read's form (0..7).
    pub guard: u8,
    /// Loop trips.
    pub trips: i64,
    /// Counter init.
    pub t0: i64,
    /// Counter step magnitude.
    pub step: i64,
    /// Counting down.
    pub down: bool,
    /// The continue condition's form (0..4).
    pub cform: u8,
    /// The position base.
    pub base: i64,
    /// The step's table read under a guard too.
    pub inloop_guard: bool,
    /// A second loop's exit counter (`K`) indexes a later body.
    pub exit_k: i64,
}

impl OntoProg {
    /// A second loop carries an array whose next value reads neither the
    /// state nor the counter (`z <- out / 2 + a` each trip, `out` the first
    /// loop's device-carried exit: made once on the device, hoisted out of
    /// the trip), its exit printed. Derived from the drawn fields (no
    /// draw of its own: the other programs stay the stream's).
    pub fn hoisted(&self) -> bool {
        (self.t0 + self.base + self.exit_k + self.trips) % 2 == 0
    }
}

pub fn onto_prog_strategy() -> impl Strategy<Value = OntoProg> {
    (
        prop::sample::select(vec![2u64, 4, 8, 16]),
        1u64..5,
        0u8..7,
        1i64..5,
        0i64..4,
        1i64..3,
        any::<bool>(),
        0u8..4,
        0i64..3,
        any::<bool>(),
        1i64..4,
    )
        .prop_map(
            |(m, reps, guard, trips, t0, step, down, cform, base, inloop_guard, exit_k)| OntoProg {
                n: m * reps * 4,
                m,
                guard,
                trips,
                t0,
                step,
                down,
                cform,
                base,
                inloop_guard,
                exit_k,
            },
        )
}

/// [`OntoProg`]'s source text.
pub fn onto_src(p: &OntoProg) -> String {
    let (n, m) = (p.n, p.m);
    let h = m / 2;
    // The counter's values on the advancing trips: `t0 + k·s` (up) or
    // `t0 + (trips − 1 − k)·s` (down), k < trips.
    let s = p.step;
    let (lo, hi) = (p.t0, p.t0 + (p.trips - 1) * s);
    let (init, next, cond) = if !p.down {
        let lim = hi + 1; // the least bound that exits after `hi`
        let cond = match p.cform {
            0 => format!("(t < {lim})"),
            1 => format!("({lim} > t)"),
            2 => format!("(t <= {})", lim - 1),
            _ => format!("(t + {s} <= {})", lim - 1 + s),
        };
        (lo, format!("t + {s}"), cond)
    } else {
        let lim = lo - 1; // the greatest bound that exits after `lo`
        let cond = match p.cform {
            0 => format!("(t > {lim})"),
            1 => format!("({lim} < t)"),
            2 => format!("(t >= {})", lim + 1),
            _ => format!("(t - {s} >= {})", lim + 1 - s),
        };
        (hi, format!("t - {s}"), cond)
    };
    let pmax = (p.base + hi) as u64;
    let lt = (pmax + 1) * n;
    let k = p.exit_k;
    let gread = match p.guard {
        0 => format!("(i < {h}) -> {{ -true-> a[c + {h}]; -false-> a[c - {h}]; }}"),
        1 => format!("(i >= {h}) -> {{ -true-> a[c - {h}]; -false-> a[c + {h}]; }}"),
        2 => format!("({h} > i) -> {{ -true-> a[c + {h}]; -false-> a[c - {h}]; }}"),
        3 => format!("(i < {h}) -> {{ -true-> c + {h}; -false-> c - {h}; }} -> kk; a[kk]"),
        4 => format!(
            "(i < {h}) -> {{ -true-> a[c + {h}]; -false-> (i < {}) -> {{ -true-> a[c - {h} + 1]; -false-> a[c - {h}]; }}; }}",
            m - 1
        ),
        5 => format!("(c >= {m} && i < {h}) -> {{ -true-> a[c - {m} + {h}]; -false-> a[c]; }}"),
        _ => "perm[c] -> j; a[j]".to_string(),
    };
    let tread = if p.inloop_guard {
        format!(
            "i % {m} -> u; (u < {h}) -> {{ -true-> tab[pp * {n} + i + {h}]; -false-> tab[pp * {n} + i - {h}]; }} -> tv"
        )
    } else {
        format!("tab[pp * {n} + i] -> tv")
    };
    format!(
        "fn main() {{
    {n} -> iota -> r;
    r -> map {{ t -> (t * 5 + 3) % 11 - 5 -> widen_f32 }} -> a;
    r -> map {{ t -> (t * 3 + 1) % {n} }} -> perm;
    {lt} -> iota -> it;
    it -> map {{ t -> (t * 7 + 1) % 13 - 6 -> widen_f32 }} -> tab;
    r -> map {{ c -> c % {m} -> i; {gread} }} -> g;
    mut x: [f32; {n}] <- g;
    mut t: i32 <- {init};
    loop {{
        {cond} -> {{
            -true-> {{
                {base} + t -> pp;
                r -> map {{ i -> {tread}; x[i] + tv }} -> x;
                {next} -> t;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
{zloop}
    mut u: i32 <- 0;
    loop {{
        (u < {k}) -> {{
            -true-> {{
                u + 1 -> u;
                -> loop;
            }}
            -false-> u -> un;
        }}
    }}
    {nk} -> iota -> rk;
    rk -> map {{ j -> out[j] * 2.0 + a[un + j] }} -> y;
    y[0] -> println;
    y[{last}] -> println;
    out[{mid}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        base = p.base,
        zloop = if p.hoisted() {
            format!(
                "    mut z: [f32; {n}] <- a;\n    mut v: i32 <- 0;\n    loop {{\n        (v < {}) -> {{\n            -true-> {{\n                r -> map {{ i -> out[i] / 2.0 + a[i] }} -> z;\n                v + 1 -> v;\n                -> loop;\n            }}\n            -false-> z -> zo;\n        }}\n    }}\n    zo[{}] -> println;\n",
                p.trips + 1,
                n / 2 + 1
            )
        } else {
            String::new()
        },
        nk = n as i64 - k,
        last = n as i64 - k - 1,
        mid = n / 2 + 1,
    )
}

/// `bounds_proof`'s soundness fuzz: the [`OntoProg`] shapes with every
/// offset, threshold, comparison, step and limit drawn freely, so some reads
/// leave their array (the program traps) and some stay inside without the
/// condition that would prove it. A proof is sound when no proven read ever
/// leaves its array: the native build (proven reads unchecked) traps exactly
/// where the interpreter does.
#[derive(Clone, Debug)]
pub struct BoundsFuzz {
    pub n: u64,
    pub m: u64,
    pub op: u8,
    pub thr: i64,
    pub d1: i64,
    pub d2: i64,
    pub t0: i64,
    pub step: i64,
    pub cop: u8,
    pub lim: i64,
    pub base: i64,
    pub rows: u64,
}

pub fn bounds_fuzz_strategy() -> impl Strategy<Value = BoundsFuzz> {
    (
        prop::sample::select(vec![4u64, 8, 16]),
        1u64..4,
        0u8..6,
        0i64..24,
        0i64..24,
        0i64..24,
        -2i64..5,
        prop::sample::select(vec![-2i64, -1, 1, 2, 3]),
        0u8..6,
        -3i64..9,
        -1i64..3,
        (0u8..4, 0i64..4),
    )
        .prop_map(
            |(m, reps, op, thr, d1, d2, t0, step, cop, lim, base, (aligned, slack))| {
                // Offsets and thresholds around the modulus: a read is in
                // bounds, out by one, or in only under its condition; half
                // the guards are rotate_half's (`i < m/2`, `±m/2`) shifted
                // by at most one.
                let w = m as i64;
                let (op, thr, d1, d2) = if aligned < 2 {
                    let h = w / 2;
                    let e = d1 % 3 - 1;
                    (0, h + i64::from(aligned) * e, h, h)
                } else {
                    (op, thr % (w + 3) - 1, d1 % (w + 3) - 1, d2 % (w + 3) - 1)
                };
                // The table's rows: the furthest position the trips reach
                // (at most 6, the counter moving by `step` from `t0`), plus
                // a slack of −1..2 (out by one, exact, or spare).
                let far = (0..6).map(|k| base + t0 + k * step).max().unwrap_or(0);
                let rows = (far + slack).clamp(1, 64) as u64;
                BoundsFuzz {
                    n: m * reps * 2,
                    m,
                    op,
                    thr,
                    d1,
                    d2,
                    t0,
                    step,
                    cop,
                    lim,
                    base,
                    rows,
                }
            },
        )
}

fn cmp_op(k: u8) -> &'static str {
    ["<", "<=", ">", ">=", "==", "!="][k as usize % 6]
}

/// [`BoundsFuzz`]'s source text.
pub fn bounds_fuzz_src(p: &BoundsFuzz) -> String {
    let (n, m) = (p.n, p.m);
    let op = cmp_op(p.op);
    let cop = cmp_op(p.cop);
    let lt = p.rows * n;
    format!(
        "fn main() {{
    {n} -> iota -> r;
    r -> map {{ t -> (t * 5 + 3) % 11 - 5 -> widen_f32 }} -> a;
    {lt} -> iota -> it;
    it -> map {{ t -> (t * 7 + 1) % 13 - 6 -> widen_f32 }} -> tab;
    r -> map {{ c -> c % {m} -> i; (i {op} {thr}) -> {{ -true-> a[c + {d1}]; -false-> a[c - {d2}]; }} }} -> g;
    mut x: [f32; {n}] <- g;
    mut t: i32 <- {t0};
    mut k: i32 <- 0;
    loop {{
        (t {cop} {lim} && k < 6) -> {{
            -true-> {{
                {base} + t -> pp;
                r -> map {{ i -> x[i] + tab[pp * {n} + i] }} -> x;
                t + {step} -> t;
                k + 1 -> k;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        thr = p.thr,
        d1 = p.d1,
        d2 = p.d2,
        t0 = p.t0,
        lim = p.lim,
        base = p.base,
        step = p.step,
    )
}

/// onto-device `host-par`: a GPU build's host work. A device tile site `x`
/// (`q · w`, `rows × 512`), a gather of it through an unranged permutation
/// (trap-capable: it stays on the host, and so does everything it feeds),
/// then a chain of clean maps over the gathered array — each reading one or
/// two earlier arrays at shifted cells (`(c + s) % n`) and a scalar of the
/// host's — the host-par candidates.
#[derive(Clone, Debug)]
pub struct HostParProg {
    pub rows: u64,
    pub maps: Vec<[u8; 3]>,
}

pub fn host_par_strategy() -> impl Strategy<Value = HostParProg> {
    (
        prop::sample::select(vec![64u64, 128, 256, 512]),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..4),
    )
        .prop_map(|(rows, maps)| HostParProg { rows, maps })
}

/// [`HostParProg`]'s source text.
pub fn host_par_src(p: &HostParProg) -> String {
    let n = p.rows * 512;
    let mut s = format!(
        "fn main() {{
    {n} -> iota -> r;
    {rq} -> iota -> rq;
    32768 -> iota -> rw;
    64 -> iota -> kd;
    rq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; v / 16.0 }} -> q;
    rw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> w;
    r -> map {{ e -> e / 512 -> i; e % 512 -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 64 + d] * w[d * 512 + j] }} }} -> x;
    r -> map {{ t -> (t * 7 + 3) % {n} }} -> perm;
    r -> map {{ c -> x[perm[c]] }} -> h0;
",
        rq = p.rows * 64
    );
    for (k, m) in p.maps.iter().enumerate() {
        let a = format!("h{}", m[0] as usize % (k + 1));
        let b = format!("h{}", m[1] as usize % (k + 1));
        let sh = 1 + m[2] as u64 % 7;
        let body = match m[2] % 3 {
            0 => format!("{a}[c] * 2.0 + {b}[(c + {sh}) % {n}]"),
            1 => format!("{a}[(c + {sh}) % {n}] - {b}[c] / 4.0"),
            _ => format!(
                "({a}[c] > 0.0) -> {{ -true-> {a}[c]; -false-> {b}[(c + {sh}) % {n}] * 0.5; }}"
            ),
        };
        s += &format!("    r -> map {{ c -> {body} }} -> h{};\n", k + 1);
    }
    let last = p.maps.len();
    s += &format!(
        "    h{last}[0] -> println;\n    h{last}[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, h{last}) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        n - 1
    );
    s
}

/// **device-geometry generated programs** (`crates/backends/llvm/tests/device_geometry.rs`):
/// the four shapes the model-scale layer showed as device gaps, at sizes and
/// forms drawn freely — a sum of products whose map body goes on past its
/// fold (`tile-post`), attention over grouped query heads (`head-group`), a
/// per-sequence row append to a loop-carried `[B][S][W]` state
/// (`piece-delta`), and a row norm whose root is a second fold (`row-tree`).
/// Every value is a small integer in `f32` (or a quotient the host and the
/// device compute with the same body text from an exact sum), so every face
/// and every fold order agrees with the interpreter.
#[derive(Clone, Debug)]
pub struct LayerProg {
    /// 0 epilogue GEMM, 1 grouped heads, 2 append loop, 3 row norm.
    pub kind: u8,
    /// GEMM rows (1: a GEMV, the split-K reduce's epilogue), lanes, depth.
    pub m: u64,
    pub n: u64,
    pub k: u64,
    /// The epilogue's form (`kind` 0: 0..6; `kind` 1: the mask / divisor).
    pub epi: u8,
    /// Grouped heads: sequences, KV heads, query heads per KV head,
    /// positions, head width; QKᵀ (`true`) or P·V.
    pub b: u64,
    pub g: u64,
    pub per: u64,
    pub l: u64,
    pub d: u64,
    pub qk: bool,
    /// Append loop: sequences `b`, positions `l`, row width `d`, trips, the
    /// first position.
    pub trips: i64,
    pub p0: i64,
    /// Grouped heads: K/V shared by every `bb` (a prefill's query tokens).
    /// Row norm: rows `m`, width `n`, Heron steps `k`, a third local fold;
    /// its reader scales each cell by its row's factor.
    pub extra: bool,
}

pub fn layer_strategy() -> impl Strategy<Value = LayerProg> {
    (
        0u8..4,
        prop::sample::select(vec![1u64, 4, 8, 16, 32]),
        prop::sample::select(vec![8u64, 16, 32, 64, 128]),
        prop::sample::select(vec![4u64, 8, 16, 32, 64, 256]),
        0u8..7,
        (1u64..3, 1u64..3, prop::sample::select(vec![1u64, 2, 4, 4])),
        // Positions and head widths up to a staged tile's lane block (128).
        (
            prop::sample::select(vec![8u64, 16, 64, 128, 256]),
            prop::sample::select(vec![4u64, 8, 32, 128]),
        ),
        any::<bool>(),
        (1i64..4, 0i64..3),
        any::<bool>(),
    )
        .prop_map(
            |(kind, m, n, k, epi, (b, g, per), (l, d), qk, (trips, p0), extra)| LayerProg {
                kind,
                m,
                n,
                k,
                epi,
                b,
                g,
                per,
                l,
                d,
                qk,
                trips,
                p0,
                extra,
            },
        )
}

/// [`LayerProg`]'s source text.
pub fn layer_prog_src(p: &LayerProg) -> String {
    let data = |name: &str, len: u64, a: u64, c: u64, md: u64| {
        let h = md / 2;
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % {md} - {h} -> widen_f32 }} -> {name};\n"
        )
    };
    match p.kind {
        // A sum of products, then elementwise terms of the same cell.
        0 => {
            let (m, n, k) = (p.m, p.n, p.k);
            let mut s = String::from("fn main() {\n");
            s += &data("a", m * k, 5, 3, 7);
            s += &data("w", k * n, 3, 1, 5);
            s += &data("x", m * n, 7, 2, 9);
            s += &data("bias", n, 2, 1, 5);
            s += &format!(
                "    {m} -> iota -> i_dv;\n    i_dv -> map {{ t -> t % 3 + 1 -> widen_f32 }} -> dv;\n"
            );
            s += &format!("    {k} -> iota -> rk;\n    {} -> iota -> o;\n", m * n);
            // A number the map captures: a whole fold of the residual.
            s += "    0.0 -> z0: f32;\n    (z0, bias) -> fold { acc, v -> acc + v } -> sc;\n";
            let fold = format!(
                "(0.0, rk) -> fold {{ acc, kk -> acc + a[i * {k} + kk] * w[kk * {n} + j] }}"
            );
            let body = match p.epi {
                0 => format!("{fold} -> y;\n        x[e] + y"),
                1 => format!("{fold} -> y;\n        y * 0.5 + bias[j]"),
                2 => format!(
                    "(j <= i + 2) -> {{\n            -true-> {{ {fold} -> y; y + 1.0 }}\n            -false-> -7.0;\n        }}"
                ),
                3 => format!("{fold} -> y;\n        y / dv[i]"),
                4 => format!("{fold} -> y;\n        y * 2.0 + sc"),
                5 => format!(
                    "{fold} -> y;\n        (y > 0.0) -> {{ -true-> y; -false-> x[e] * 0.25; }}"
                ),
                _ => format!("{fold} -> y;\n        x[e] + y * bias[j] - sc"),
            };
            s += &format!(
                "    o -> map {{\n        e -> e / {n} -> i;\n        e % {n} -> j;\n        {body}\n    }} -> c;\n"
            );
            s += &format!(
                "    c[0] -> println;\n    c[{}] -> println;\n    c[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, c) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                m * n - 1,
                (m * n) / 2 + 1
            );
            s
        }
        // Attention over grouped query heads: QKᵀ (masked at a position the
        // program computes: a uniform) or P·V (divided by a per-row sum).
        1 => {
            let (b, g, per, l, d) = (p.b, p.g, p.per, p.l, p.d);
            let h = g * per;
            let mut s = String::from("fn main() {\n");
            s += &data("q", b * h * d, 5, 3, 7);
            // `extra`: one K/V for every `bb` — `bb` a query position (a
            // prefill's token), read by `q`/`pe` alone: the output's token
            // digit is then above the heads' (a digit-affine output).
            let (kn, kb) = match p.extra {
                true => (l * g * d, String::new()),
                false => (b * l * g * d, format!("bb * {l} + ")),
            };
            s += &data("kc", kn, 3, 1, 5);
            s += &data("pe", b * h * l, 7, 2, 5);
            s += &format!(
                "    {} -> iota -> i_den;\n    i_den -> map {{ t -> t % 4 + 1 -> widen_f32 }} -> den;\n",
                b * h
            );
            s += &format!("    {d} -> iota -> rd;\n    {l} -> iota -> rl;\n");
            // The mask position: a number computed from data (a uniform).
            s += &format!(
                "    0 -> z0: i32;\n    (z0, rd) -> fold {{ acc, v -> acc + v % 2 }} -> pp0;\n    pp0 % {l} -> pos;\n"
            );
            if p.qk {
                let post = match p.epi % 3 {
                    0 => "dt * 0.5".to_string(),
                    1 => "(r <= pos) -> { -true-> dt * 0.5; -false-> -1000.0; }".to_string(),
                    _ => "(r <= hh) -> { -true-> dt; -false-> 0.0 - dt; }".to_string(),
                };
                s += &format!(
                    "    {} -> iota -> o;\n    o -> map {{\n        e -> e / {} -> bb;\n        e / {l} % {h} -> hh;\n        e % {l} -> r;\n        (0.0, rd) -> fold {{\n            acc, u -> acc + q[(bb * {h} + hh) * {d} + u] * kc[(({kb}r) * {g} + hh / {per}) * {d} + u]\n        }} -> dt;\n        {post}\n    }} -> s;\n",
                    b * h * l,
                    h * l
                );
            } else {
                let post = match p.epi % 2 {
                    0 => "av / den[bb * {h} + hh]".replace("{h}", &h.to_string()),
                    _ => "av".to_string(),
                };
                s += &format!(
                    "    {} -> iota -> o;\n    o -> map {{\n        e -> e / {} -> bb;\n        e / {d} % {h} -> hh;\n        e % {d} -> u;\n        (0.0, rl) -> fold {{\n            acc, r -> acc + pe[(bb * {h} + hh) * {l} + r] * kc[(({kb}r) * {g} + hh / {per}) * {d} + u]\n        }} -> av;\n        {post}\n    }} -> s;\n",
                    b * h * d,
                    h * d
                );
            }
            let n = if p.qk { b * h * l } else { b * h * d };
            s += &format!(
                "    s[0] -> println;\n    s[{}] -> println;\n    s[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, s) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                n - 1,
                n / 2 + 1
            );
            s
        }
        // A per-sequence row append to a carried `[B][S][W]` state.
        2 => {
            let (b, l, w) = (p.b, p.l.max(p.trips as u64 + p.p0 as u64 + 1), p.d);
            let n = b * l * w;
            let mut s = String::from("fn main() {\n");
            s += &data("kc0", n, 3, 1, 5);
            s += &data("nw", b * w, 5, 2, 7);
            s += &format!("    {n} -> iota -> rc;\n    {} -> iota -> rbw;\n", b * w);
            s += &format!(
                "    mut kc: [f32; {n}] <- kc0;\n    mut acc: [f32; {}] <- nw;\n    mut t: i32 <- 0;\n",
                b * w
            );
            s += &format!(
                "    loop {{\n        (t < {}) -> {{\n            -true-> {{\n                {} + t -> p;\n                rbw -> map {{ e -> acc[e] + 1.0 }} -> acc;\n                rc -> map {{\n                    c -> c / {} -> bb;\n                    c / {w} % {l} -> r;\n                    (r == p) -> {{\n                        -true-> acc[bb * {w} + c % {w}] * 2.0;\n                        -false-> kc[c];\n                    }}\n                }} -> kc;\n                t + 1 -> t;\n                -> loop;\n            }}\n            -false-> kc -> out;\n        }}\n    }}\n",
                p.trips,
                p.p0,
                l * w
            );
            s += &format!(
                "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ a2, v -> a2 + v }} -> total;\n    total -> println;\n}}\n",
                n - 1,
                (p.p0 as u64) * w + 1
            );
            s
        }
        // A row norm: the sum of squares of a row, then a Heron root (a
        // second fold that loads nothing).
        _ => {
            let (rows, width) = (p.m.max(1), p.n.max(32));
            let steps = 4 + p.k % 8;
            let mut s = String::from("fn main() {\n");
            s += &data("x", rows * width, 5, 3, 7);
            s += &format!(
                "    {rows} -> iota -> rr;\n    {width} -> iota -> rw;\n    {steps} -> iota -> rh;\n"
            );
            let extra = if p.extra {
                "        4 -> iota -> r4;\n        (sd, r4) -> fold { a2, kk -> a2 * 1.0 } -> sd2;\n        1.0 / sd2"
            } else {
                "        1.0 / sd"
            };
            s += &format!(
                "    rr -> map {{\n        b -> (0.0, rw) -> fold {{\n            a1, i -> x[b * {width} + i] -> v;\n            a1 + v * v\n        }} -> ss;\n        ss / {width}.0 + 1.0 -> ra;\n        0.5 + 0.5 * ra -> s0;\n        (s0, rh) -> fold {{ a3, kk -> 0.5 * (a3 + ra / a3) }} -> sd;\n{extra}\n    }} -> rinv;\n"
            );
            // The norm's reader: each cell scaled by its row's factor.
            s += &format!(
                "    i_x -> map {{ e -> x[e] * rinv[e / {width}] }} -> xn;\n    rinv[0] -> println;\n    rinv[{}] -> println;\n    xn[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, xn) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                rows - 1,
                rows * width - 1
            );
            s
        }
    }
}

/// **long-prefill `uniform-base` generated programs**
/// (`crates/backends/llvm/tests/device_geometry.rs`): a loop over `groups`
/// KV-head groups whose counter `gg` slices the operands — the group's query
/// heads (`q[(bb·h + gg·per + hh)·d + u]`), its KV head
/// (`kc[(r·g + gg)·d + u]`) and its weight panel (`w[(gg·K + k)·n + j]`) —
/// as an attention half (QKᵀ or P·V, plain, masked at a position the
/// counter moves, or scaled per row) and the group's projection added into
/// a carried `y`. Head widths include odd ones (`d = 3`: a pointer moved by
/// a non-vector step; `d = 4`: one 16-byte vector of `f32`); `b` rows the staged tiles take. The K/V
/// cache is stored `f32` or `bf16` (`narrow`: `to_bf16`, read `-> widen_f32` — ADR-0043's narrow
/// operand, its pointer moved by the counter in its own 2-byte elements). Every value is a small
/// integer or an exact quotient by 2 or 4, so every face and fold order agrees with the interpreter.
#[derive(Clone, Debug)]
pub struct SliceProg {
    /// Query rows (tokens or sequences), KV groups (the loop's trips),
    /// query heads a group, positions, head width, projection width.
    pub b: u64,
    pub groups: u64,
    pub per: u64,
    pub l: u64,
    pub d: u64,
    pub n: u64,
    /// QKᵀ (`true`) or P·V.
    pub qk: bool,
    /// 0 plain, 1 masked (QKᵀ: `r ≤ hh + gg`) / doubled, 2 halved / per-row
    /// divisor.
    pub epi: u8,
    /// K/V shared by every sequence (a prefill's tokens) or per sequence.
    pub shared: bool,
    /// A row fold reads the carry's init before the loop (its device copy
    /// then dies at the loop's entry: long-prefill `carry-take`).
    pub pre: bool,
    /// The K/V cache stored `bf16` (ADR-0043), read through `widen_f32`.
    pub narrow: bool,
}

pub fn slice_strategy() -> impl Strategy<Value = SliceProg> {
    (
        (
            prop::sample::select(vec![4u64, 16, 32]),
            1u64..5,
            prop::sample::select(vec![1u64, 2, 4]),
        ),
        (
            prop::sample::select(vec![16u64, 64]),
            prop::sample::select(vec![3u64, 4, 8, 32]),
            prop::sample::select(vec![16u64, 64]),
        ),
        any::<bool>(),
        0u8..3,
        (any::<bool>(), any::<bool>(), any::<bool>()),
    )
        .prop_map(
            |((b, groups, per), (l, d, n), qk, epi, (shared, pre, narrow))| SliceProg {
                b,
                groups,
                per,
                l,
                d,
                n,
                qk,
                epi,
                shared,
                pre,
                narrow,
            },
        )
}

/// [`SliceProg`]'s source text.
pub fn slice_prog_src(p: &SliceProg) -> String {
    let data = |name: &str, len: u64, a: u64, c: u64, md: u64| {
        let h = md / 2;
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % {md} - {h} -> widen_f32 }} -> {name};\n"
        )
    };
    let (b, g, per, l, d, n) = (p.b, p.groups, p.per, p.l, p.d, p.n);
    let h = g * per;
    // One group's output: `b` rows of `kk` (its heads' cells).
    let x = if p.qk { l } else { d };
    let kk = per * x;
    let (kn, kb) = match p.shared {
        true => (l * g * d, String::new()),
        false => (b * l * g * d, format!("bb * {l} + ")),
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", b * h * d, 5, 3, 7);
    s += &match p.narrow {
        true => {
            data("kc", kn, 3, 1, 5).replace("widen_f32 } -> kc;", "widen_f32 -> to_bf16 } -> kc;")
        }
        false => data("kc", kn, 3, 1, 5),
    };
    s += &data("pe", b * h * l, 7, 2, 5);
    s += &data("w", g * kk * n, 3, 2, 5);
    s += &data("y0", b * n, 5, 1, 9);
    s += &format!(
        "    {} -> iota -> i_den;\n    i_den -> map {{ t -> (t % 2 + 1) * 2 -> widen_f32 }} -> den;\n",
        b * h
    );
    s += &format!(
        "    {d} -> iota -> rd;\n    {l} -> iota -> rl;\n    {kk} -> iota -> rk;\n    {} -> iota -> o;\n    {} -> iota -> ry;\n",
        b * kk,
        b * n
    );
    if p.pre {
        s += &format!(
            "    {b} -> iota -> rb;\n    {n} -> iota -> rn;\n    rb -> map {{ i -> (0.0, rn) -> fold {{ acc, j -> acc + y0[i * {n} + (j * 3) % {n}] }} }} -> ys;\n    ys[0] -> println;\n"
        );
    }
    s += &format!(
        "    mut y: [f32; {}] <- y0;\n    mut gg: i32 <- 0;\n",
        b * n
    );
    // The cache's element: `f32` read in the product, or `bf16` widened
    // first (an exact widening the narrow tile site reads through).
    let kread = format!("kc[(({kb}r) * {g} + gg) * {d} + u]");
    let (kpre, kv) = match p.narrow {
        true => (
            format!("{kread} -> widen_f32 -> kv;\n                        "),
            "kv".to_string(),
        ),
        false => (String::new(), kread),
    };
    let att = if p.qk {
        let fold = format!(
            "(0.0, rd) -> fold {{\n                        acc, u -> {kpre}acc + q[(bb * {h} + gg * {per} + hh) * {d} + u] * {kv}\n                    }}"
        );
        let body = match p.epi {
            0 => format!("{fold}"),
            1 => format!(
                "(r <= hh + gg) -> {{\n                        -true-> {{ {fold} -> dt; dt }}\n                        -false-> -1000.0;\n                    }}"
            ),
            _ => format!("{fold} -> dt;\n                    dt * 0.5"),
        };
        format!(
            "                o -> map {{\n                    e -> e / {kk} -> bb;\n                    e / {l} % {per} -> hh;\n                    e % {l} -> r;\n                    {body}\n                }} -> s;\n"
        )
    } else {
        let fold = format!(
            "(0.0, rl) -> fold {{\n                        acc, r -> {kpre}acc + pe[(bb * {h} + gg * {per} + hh) * {l} + r] * {kv}\n                    }}"
        );
        let body = match p.epi {
            0 => format!("{fold}"),
            1 => format!("{fold} -> av;\n                    av * 2.0"),
            _ => format!("{fold} -> av;\n                    av / den[bb * {h} + gg * {per} + hh]"),
        };
        format!(
            "                o -> map {{\n                    e -> e / {kk} -> bb;\n                    e / {d} % {per} -> hh;\n                    e % {d} -> u;\n                    {body}\n                }} -> s;\n"
        )
    };
    s += &format!(
        "    loop {{\n        (gg < {g}) -> {{\n            -true-> {{\n{att}                ry -> map {{\n                    e -> e / {n} -> i;\n                    e % {n} -> j;\n                    (0.0, rk) -> fold {{\n                        acc, k -> acc + s[i * {kk} + k] * w[(gg * {kk} + k) * {n} + j]\n                    }} -> yo;\n                    y[e] + yo\n                }} -> y;\n                gg + 1 -> gg;\n                -> loop;\n            }}\n            -false-> y -> out;\n        }}\n    }}\n"
    );
    s += &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        b * n - 1,
        (b * n) / 2 + 1
    );
    s
}

// ---------------------------------------------------------------------------
// exp (ADR-0042): `x -> exp` over every branch of the oracle's algorithm.
// ---------------------------------------------------------------------------

/// A program taking `exp` of values that reach every branch of the oracle's
/// algorithm — normal results, overflow to `inf`, subnormal results, `0`,
/// `±inf` and `NaN` arguments (a quotient by a zero difference) — in `f32`
/// and `f64`, in a cell, in a fold body and through a row's max (a softmax);
/// sums and cells printed. `scale` sets the range (`2^scale / 8` times the
/// law's ±`h`).
#[derive(Clone, Debug)]
pub struct ExpProg {
    pub n: u64,
    pub rows: u64,
    pub mul: u64,
    pub add: u64,
    pub m: u64,
    pub scale: u8,
    pub wide: bool,
    pub nan: bool,
    pub softmax: bool,
    pub in_fold: bool,
}

pub fn exp_prog_strategy() -> impl Strategy<Value = ExpProg> {
    (
        prop::sample::select(vec![256u64, 512, 1024, 2048]),
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![7u64, 11, 13, 29]),
        0u64..64,
        prop::sample::select(vec![101u64, 211, 401]),
        0u8..6,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(n, rows, mul, add, m, scale, wide, nan, softmax, in_fold)| ExpProg {
                n,
                rows,
                mul,
                add,
                m,
                scale,
                wide,
                nan,
                softmax,
                in_fold,
            },
        )
}

/// [`ExpProg`]'s source text. `x` is a tile site's output (`q · W`, `d =
/// 8`: the cells reading it run on the device), scaled.
pub fn exp_src(p: &ExpProg) -> String {
    let (n, r) = (p.n, p.rows);
    let l = n / r;
    let sc = f64::from(1u32 << p.scale) / 8.0;
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {r} -> iota -> rr;\n    {l} -> iota -> kl;\n    8 -> iota -> kd;\n    \
         {rq} -> iota -> tq;\n    tq -> map {{ e -> (e * {mul} + {add}) % 31 - 15 -> widen_f32 -> v; v / 4.0 }} -> q;\n    \
         {lw} -> iota -> tw;\n    tw -> map {{ e -> (e * 7 + 13) % {m} - {h} -> widen_f32 -> v; v / 8.0 }} -> w;\n    \
         t -> map {{ e -> e / {l} -> i; e % {l} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {l} + j] }} -> u; u * {sc:.3} }} -> x;\n",
        rq = r * 8,
        lw = l * 8,
        mul = p.mul,
        add = p.add,
        m = p.m,
        h = p.m / 2,
    );
    // A cell: `exp` of the value, or (every 7th cell from 0 and 1) of `NaN`
    // and of `±inf` (a quotient by a zero difference).
    let arg = if p.nan {
        "x[e] - x[e] -> z; (e % 7 == 0) -> { -true-> z / z; -false-> x[e] / z; } -> a; (e % 7 < 2) -> { -true-> a; -false-> x[e]; }".to_string()
    } else {
        "x[e]".to_string()
    };
    s += &format!("    t -> map {{ e -> {arg} -> exp }} -> y;\n");
    // `f64`: a range past f64's overflow and underflow.
    if p.wide {
        s += "    t -> map { e -> x[e] -> widen_f64 -> d; d * 8.0 -> exp } -> z;\n";
        s += "    0.0 -> z0: f64;\n    (z0, z) -> fold { acc, v -> acc + v / 1000000.0 } -> zs;\n";
    }
    // A fold whose step reads `exp` of its element.
    if p.in_fold {
        s += &format!(
            "    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> x[i * {l} + j] / 16.0 -> exp -> w; acc + w }} }} -> fs;\n"
        );
    }
    // A softmax over the rows (the rescale law's form).
    if p.softmax {
        s += &format!(
            "    rr -> map {{ i -> (-1000000000.0, kl) -> fold {{ acc, j -> x[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }} }} -> mx;\n    \
             t -> map {{ e -> x[e] - mx[e / {l}] -> exp }} -> ex;\n    \
             rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {l} + j] }} }} -> den;\n    \
             t -> map {{ e -> ex[e] / den[e / {l}] }} -> sm;\n    sm[{a}] -> println;\n    den[{b}] -> println;\n",
            a = n / 3,
            b = r - 1
        );
    }
    for i in [0, 1, n / 5, n / 2 + 1, n - 1] {
        s += &format!("    y[{i}] -> println;\n");
    }
    if p.wide {
        s += &format!(
            "    z[{}] -> println;\n    z[{}] -> println;\n    zs -> println;\n",
            n / 7,
            n - 2
        );
    }
    if p.in_fold {
        s += &format!("    fs[{}] -> println;\n", r - 1);
    }
    s += "    0.0 -> y0: f32;\n    (y0, y) -> fold { acc, v -> acc + v / 1000000.0 } -> ys;\n    ys -> println;\n}\n";
    s
}

// --- large-gemm: GEMM shapes from decode GEMVs to model-scale projections ----

/// One `m × k · k × n` product (`y[i·n + j] = Σ_k a[i·k + k'] · b[k'·n + j]`,
/// `b` row-major, or transposed under `bt`) with small integer operands in
/// `-2 ..= 2`, so every partial sum is exact in f32 up to `k` of 2²² and any
/// cut or order of the fold prints the same bits. `scale` picks the extents:
///
/// - 0 skinny: `m` 1–32 (odd 3, 24), `n` 1024–4096, `k` 512–4096 — decode
///   projections, whose one-warp tile grid leaves the part short of warps;
/// - 1 under-filled deep: `m` 64–256 (odd 96, 192), `n` 1024–8192, `k`
///   4096–16384 (odd 6144, 12288) — a prefill chunk's projections;
/// - 2 filled: `m` 1024–4096, `n` 1024–8192, `k` 1024–8192 (odd 3072);
/// - 3 odd: extents a register tile need not divide (`m` 100, 136, 200; `n`
///   1000, 1500, 2040; `k` 1000, 4100, 8000).
///
/// Large extents are for plans only (emitted, never run); [`gemm_run_src`]
/// gives the sizes an emulator runs.
#[derive(Clone, Debug)]
pub struct GemmProg {
    pub scale: u8,
    pub dims: [u8; 3],
    pub bt: bool,
    pub law: [u8; 2],
}

pub fn gemm_prog_strategy() -> impl Strategy<Value = GemmProg> {
    (
        0u8..4,
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.2),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(scale, dims, bt, law)| GemmProg {
            scale,
            dims,
            bt,
            law,
        })
}

/// [`GemmProg`]'s `(m, n, k)`.
pub fn gemm_dims(p: &GemmProg) -> (u64, u64, u64) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let [dm, dn, dk] = p.dims;
    match p.scale {
        0 => (
            pick(dm, &[1, 3, 8, 16, 24, 32]),
            pick(dn, &[1024, 2048, 3072, 4096]),
            pick(dk, &[512, 1024, 4096]),
        ),
        1 => (
            pick(dm, &[64, 96, 128, 192, 256]),
            pick(dn, &[1024, 2048, 4096, 6144, 8192]),
            pick(dk, &[4096, 6144, 8192, 12288, 16384]),
        ),
        2 => (
            pick(dm, &[1024, 2048, 4096]),
            pick(dn, &[1024, 4096, 8192]),
            pick(dk, &[1024, 3072, 8192]),
        ),
        _ => (
            pick(dm, &[100, 136, 200]),
            pick(dn, &[1000, 1500, 2040]),
            pick(dk, &[1000, 4100, 8000]),
        ),
    }
}

/// The source of a [`GemmProg`] at `(m, n, k)`: the operands from a law in
/// `-2 ..= 2`, the product as a map over the output's cells folding `k`, and
/// five cells printed (the first, the last, the last row's first, a middle
/// one, the second row's last).
pub fn gemm_src_at(p: &GemmProg, m: u64, n: u64, k: u64) -> String {
    let (a1, a2) = (
        3 + 2 * u64::from(p.law[0] % 7),
        5 + 2 * u64::from(p.law[1] % 5),
    );
    let bi = match p.bt {
        true => format!("b[j * {k} + k]"),
        false => format!("b[k * {n} + j]"),
    };
    let mut s = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{
        t -> (t * {a1} + 13) % 5 - 2 -> widen_f32
    }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{
        t -> (t * {a2} + 7) % 5 - 2 -> widen_f32
    }} -> b;
    {k} -> iota -> kr;
    {mn} -> iota -> ic;
    ic -> map {{
        t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{
            acc, k -> acc + a[i * {k} + k] * {bi}
        }}
    }} -> y;
",
        mk = m * k,
        kn = k * n,
        mn = m * n
    );
    let last = m * n - 1;
    for c in [0, last, (m - 1) * n, (m * n) / 2 + n / 3, 2 * n - 1] {
        s += &format!("    y[{}] -> println;\n", c.min(last));
    }
    s + "}\n"
}

/// [`GemmProg`]'s source at its own extents ([`gemm_dims`]).
pub fn gemm_src(p: &GemmProg) -> (String, (u64, u64, u64)) {
    let (m, n, k) = gemm_dims(p);
    (gemm_src_at(p, m, n, k), (m, n, k))
}

// --- half-precision (ADR-0043): narrow float storage --------------------------

/// A program over `bf16` / `f16` storage: an `f32` activation `a`
/// (`rows × k`), a narrow weight `w` (`k × cols`, or `cols × k` read
/// transposed — strided lanes) rounded once from an `f32` law at a scale that
/// reaches the format's normal range, its overflow (`f16` past 65504, `bf16`
/// past `f32`'s max) and its subnormals; a sum of products reading `w`'s
/// exact widening (a [`mapal_ir::CategoryIr::tile_plan_narrow`] site — with an
/// epilogue under `post`, both operands narrow under `both`); then cells that
/// round the products into the format, a guard selecting narrow values, a
/// loop carrying a narrow array through a map and an element update, and a
/// whole fold over widened narrow values.
#[derive(Clone, Debug)]
pub struct NarrowProg {
    pub rows: u64,
    pub cols: u64,
    pub k: u64,
    /// `true`: `f16`; else `bf16`.
    pub f16: bool,
    /// The weight law's scale: 0 a model weight (`1/1856`), 1 `1/8`, 2 past
    /// the format's overflow, 3 into its subnormals.
    pub scale: u8,
    pub mul: u64,
    pub add: u64,
    pub both: bool,
    pub strided: bool,
    pub post: bool,
    pub guard: bool,
    pub looped: bool,
}

pub fn narrow_prog_strategy() -> impl Strategy<Value = NarrowProg> {
    (
        (
            prop::sample::select(vec![1u64, 2, 4, 8, 32]),
            prop::sample::select(vec![64u64, 96, 256]),
            prop::sample::select(vec![8u64, 16, 64]),
            any::<bool>(),
            prop::sample::select(vec![0u8, 0, 1, 2, 3]),
            prop::sample::select(vec![3u64, 7, 11, 13]),
            0u64..32,
        ),
        (
            any::<bool>(),
            prop::bool::weighted(0.3),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        ),
    )
        .prop_map(
            |((rows, cols, k, f16, scale, mul, add), (both, strided, post, guard, looped))| {
                NarrowProg {
                    rows,
                    cols,
                    k,
                    f16,
                    scale,
                    mul,
                    add,
                    both,
                    strided,
                    post,
                    guard,
                    looped,
                }
            },
        )
}

/// [`NarrowProg`]'s source and the count of its leading printed lines that are
/// `f32` sums of exact operands (ADR-0040 D2's bound on the default face); the
/// later lines pass through a rounding into the format (ADR-0043 D3: a flip
/// across a tie per rounding).
pub fn narrow_src(p: &NarrowProg) -> (String, usize) {
    let fmt = if p.f16 { "f16" } else { "bf16" };
    let (rows, cols, k) = (p.rows, p.cols, p.k);
    let n = rows * cols;
    let scale = match (p.scale, p.f16) {
        (0, _) => "/ 1856.0".to_string(),
        (1, _) => "/ 8.0".to_string(),
        (2, true) => "* 2000.0".to_string(),
        (2, false) => "* 10000000000000000000000000000000000000.0".to_string(),
        (_, true) => "* 0.0000001".to_string(),
        (_, false) => "* 0.000000000000000000000000000000000000001".to_string(),
    };
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {k} -> iota -> kk;\n    {rk} -> iota -> ta;\n    \
         {kc} -> iota -> tw;\n    \
         ta -> map {{ e -> (e * {mul} + {add}) % 31 - 15 -> widen_f32 -> v; v / 4.0 }} -> a;\n    \
         tw -> map {{ e -> (e * 7 + 13) % 103 - 51 -> widen_f32 -> v; v {scale} -> to_{fmt} }} -> w;\n",
        rk = rows * k,
        kc = k * cols,
        mul = p.mul,
        add = p.add,
    );
    // `a` itself narrow: both operands widened.
    let (abind, aval) = if p.both {
        s += &format!("    ta -> map {{ e -> a[e] -> to_{fmt} }} -> an;\n");
        (
            "an[i * K + q] -> widen_f32 -> x; ".replace('K', &k.to_string()),
            "x".to_string(),
        )
    } else {
        (String::new(), format!("a[i * {k} + q]"))
    };
    let wi = if p.strided {
        format!("w[j * {k} + q]")
    } else {
        format!("w[q * {cols} + j]")
    };
    let tail = if p.post {
        format!(" -> u; u * 0.5 + a[i * {k}]")
    } else {
        String::new()
    };
    s += &format!(
        "    t -> map {{ e -> e / {cols} -> i; e % {cols} -> j; (0.0, kk) -> fold {{ acc, q -> {abind}{wi} -> widen_f32 -> b; acc + {aval} * b }}{tail} }} -> y;\n"
    );
    let mut tight = 0;
    for c in [0, n / 3, n - 1] {
        s += &format!("    y[{c}] -> println;\n");
        tight += 1;
    }
    // Cells rounding into the format; a guard selecting narrow values.
    if p.guard {
        s += &format!(
            "    t -> map {{ e -> (e % 3 == 0) -> {{ -true-> y[e] -> to_{fmt}; -false-> w[e % {kc}]; }} }} -> z;\n",
            kc = k * cols
        );
    } else {
        s += &format!("    t -> map {{ e -> y[e] -> to_{fmt} }} -> z;\n");
    }
    // A loop carrying a narrow array: a map over it and an element update.
    let fin = if p.looped {
        s += &format!(
            "    mut st: [{fmt}; {n}] <- z;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 3) -> {{\n            -true-> {{\n                \
             t -> map {{ e -> st[e] -> widen_f32 -> v; v * 1.5 + 0.25 -> to_{fmt} }} -> st;\n                \
             it * 5 % {n} -> c;\n                st[c] -> widen_f32 -> sv;\n                sv + 1.0 -> to_{fmt} -> sn;\n                \
             st[c] <- sn;\n                it + 1 -> it;\n                -> loop;\n            }}\n            -false-> st -> fin;\n        }}\n    }}\n"
        );
        "fin"
    } else {
        "z"
    };
    for c in [0, 1, n / 2 + 1, n - 1] {
        s += &format!("    {fin}[{c}] -> widen_f32 -> println;\n");
    }
    s += &format!(
        "    0.0 -> zero: f32;\n    (zero, {fin}) -> fold {{ acc, v -> v -> widen_f32 -> x; acc + x }} -> sum;\n    sum -> println;\n}}\n"
    );
    (s, tight)
}

// --- tensor-cores (ADR-0043 D4 / D6): 16-bit products ------------------------

/// A sum of products whose multiplicands are 16-bit: operands stored `f32`,
/// `bf16` or `f16` (`fa`, `fb`: 0 / 1 / 2), the fold under `{f32}`,
/// `: bf16 | f32` or `: f16 | f32` (`set` 0 / 1 / 2). `attn`: an attention
/// head instead of a GEMM — scores `q·kᵀ` (`b` read `n × k`, scaled: a tile
/// site with an epilogue), a row softmax (`exp`), then `p·v` — both products
/// under the set, `k` / `v` the stored cache; `bt` there: two heads (the
/// sites read the head digit — `tile_plan_radix`). `bt` (GEMM): `b` read
/// `n × k` (`b[j·k + q]`); `post` (GEMM): an epilogue `u·0.5 + a`.
#[derive(Clone, Debug)]
pub struct HalfProg {
    pub dims: [u8; 3],
    pub fa: u8,
    pub fb: u8,
    pub set: u8,
    pub attn: bool,
    pub bt: bool,
    pub post: bool,
    pub law: [u8; 2],
}

pub fn half_prog_strategy() -> impl Strategy<Value = HalfProg> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 1, 1, 2]),
        prop::sample::select(vec![0u8, 1, 1, 2]),
        prop::sample::select(vec![0u8, 1, 1, 1, 2]),
        prop::bool::weighted(0.3),
        prop::bool::weighted(0.3),
        any::<bool>(),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, fa, fb, set, attn, bt, post, (l0, l1))| HalfProg {
            dims: [d.0, d.1, d.2],
            // `: f16 | f32` rounds no f32 operand (no value-range fact, D4):
            // half its programs store both operands `f16`.
            fa: if set == 2 && l0 % 2 == 0 { 2 } else { fa },
            fb: if set == 2 && l0 % 2 == 0 { 2 } else { fb },
            set,
            attn,
            bt,
            post,
            law: [l0, l1],
        })
}

/// The format a [`HalfProg`] operand is stored in (`None`: f32).
pub fn half_fmt(f: u8) -> Option<&'static str> {
    match f {
        1 => Some("bf16"),
        2 => Some("f16"),
        _ => None,
    }
}

/// [`HalfProg`]'s source and the checks its printed lines take: `(cells,
/// k, k_out)` — each checked value is printed beside its `M = Σ|a·b|`
/// (value, magnitude pairs) for the first `cells` pairs over a `k`-step
/// fold. An attention program first prints the largest score magnitude,
/// and its output pairs (a `k_out`-step fold) follow the scores' (their `p`
/// passes through `exp` and the member's rounding: the test composes
/// ADR-0043 D3).
pub fn half_src(p: &HalfProg) -> (String, (usize, u64, u64)) {
    half_src_with(p, false)
}

/// [`half_src`]; `once` (sweep-gemm `round-store`): a GEMM's magnitudes read
/// their own copies of the operands (`am`, `bm`: the same laws), so the
/// product's operands have no f32 reader — every element read of an f32
/// operand rounds to the set's member (`PrecisionPlan::store`).
pub fn half_src_with(p: &HalfProg, once: bool) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let set = match p.set {
        1 => ": bf16 | f32",
        2 => ": f16 | f32",
        _ => "",
    };
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let store = |f: u8| half_fmt(f).map_or(String::new(), |n| format!(" -> to_{n}"));
    let read = |f: u8, x: &str| match half_fmt(f) {
        Some(_) => format!("{x} -> widen_f32"),
        None => x.to_string(),
    };
    let abs = "(pv > 0.0) -> { -true-> pv; -false-> 0.0 - pv; } -> av; acc + av";
    if !p.attn {
        let m = pick(p.dims[0], &[16, 32, 48, 64, 128]);
        let n = pick(p.dims[1], &[8, 16, 32, 64, 128]);
        let k = pick(p.dims[2], &[16, 32, 48, 64, 128, 256]);
        let bi = if p.bt {
            format!("b[j * {k} + q]")
        } else {
            format!("b[q * {n} + j]")
        };
        let (ar, br) = (read(p.fa, &format!("a[i * {k} + q]")), read(p.fb, &bi));
        let post = if p.post {
            format!(
                " -> u; a[i * {k}]{} -> r; u * 0.5 + r",
                if p.fa > 0 { " -> widen_f32" } else { "" }
            )
        } else {
            String::new()
        };
        let cells = [
            u64::from(p.law[0]) * 7 % (m * n),
            m * n / 2 + 1,
            m * n - 1 - u64::from(p.law[1]) % n,
        ];
        let mut src = format!(
            "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0{sa} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0{sb} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{set} {{ acc, q -> {ar} -> x; {br} -> y; acc + x * y }}{post} }} -> y;
",
            mk = m * k,
            kn = k * n,
            mn = m * n,
            sa = store(p.fa),
            sb = store(p.fb),
        );
        let (am, bm) = if once {
            src += &format!(
                "    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0{sa} }} -> am;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0{sb} }} -> bm;
",
                sa = store(p.fa),
                sb = store(p.fb),
            );
            (
                read(p.fa, &format!("am[i * {k} + q]")),
                read(p.fb, &bi.replacen("b[", "bm[", 1)),
            )
        } else {
            (ar, br)
        };
        src += &format!(
            "    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {am} -> x; {bm} -> y; x * y -> pv; {abs} }} }} -> mag;
"
        );
        // An epilogue's `0.5·u + r`: M scales by one half, `r` exact.
        for c in cells {
            src += &format!("    y[{c}] -> println;\n    mag[{c}] -> println;\n");
        }
        src += "}\n";
        return (src, (cells.len(), k, 0));
    }
    let s = pick(p.dims[0], &[16, 32, 64]);
    let d = pick(p.dims[1], &[16, 32, 64]);
    // `bt` (attention): two heads — the scores and `P·V` read `q`, `k`, `v`
    // at the head digit (`tile_plan_radix`'s sites).
    let hn: u64 = if p.bt { 2 } else { 1 };
    // One head: the plain forms (no head digit), as `tile_plan_post` /
    // `tile_plan_narrow` take them.
    let hd = |x: &str| {
        if hn == 1 {
            String::new()
        } else {
            format!("h * {s} + {x}")
        }
    };
    let row = |x: &str| match hn {
        1 => x.to_string(),
        _ => format!("({})", hd(x)),
    };
    let (qr, kr) = (
        read(p.fa, &format!("qh[{} * {d} + e]", row("i"))),
        read(p.fb, &format!("kc[{} * {d} + e]", row("j"))),
    );
    let vr = read(p.fb, &format!("vc[{} * {d} + c]", row("j")));
    let (hsc, hso) = match hn {
        1 => (format!("t / {s} -> i;"), format!("t / {d} -> i;")),
        _ => (
            format!("t / {} -> h; t / {s} % {s} -> i;", s * s),
            format!("t / {} -> h; t / {d} % {s} -> i;", s * d),
        ),
    };
    let prow = row("i");
    let (ns, nd) = (hn * s * s, hn * s * d);
    let cells = [
        u64::from(p.law[0]) * 5 % ns,
        ns - 1 - u64::from(p.law[1]) % s,
    ];
    let outs = [u64::from(p.law[1]) * 3 % nd, nd - 1];
    let mut src = format!(
        "fn main() {{
    {nd} -> iota -> it;
    it -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 29.0{sa} }} -> qh;
    it -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 31.0{sb} }} -> kc;
    it -> map {{ t -> (t * 11 + 7) % 97 - 48 -> widen_f32 -> v; v / 13.0{sb} }} -> vc;
    {ns} -> iota -> isc;
    {d} -> iota -> dr;
    {s} -> iota -> sr;
    {hs} -> iota -> hr;
    isc -> map {{ t -> {hsc} t % {s} -> j; (0.0, dr) -> fold{set} {{ acc, e -> {qr} -> x; {kr} -> y; acc + x * y }} -> u; u * 0.25 }} -> sc;
    isc -> map: f32 {{ t -> {hsc} t % {s} -> j; (0.0, dr) -> fold {{ acc, e -> {qr} -> x; {kr} -> y; x * y -> pv; {abs} }} -> u; u * 0.25 }} -> smag;
    hr -> map {{ r -> (-1000000000.0, sr) -> fold {{ m, j -> sc[r * {s} + j] -> x; (x > m) -> {{ -true-> x; -false-> m; }} }} }} -> mx;
    isc -> map {{ t -> t / {s} -> r; sc[t] - mx[r] -> exp }} -> ex;
    hr -> map {{ r -> (0.0, sr) -> fold {{ acc, j -> acc + ex[r * {s} + j] }} }} -> l;
    isc -> map {{ t -> t / {s} -> r; ex[t] / l[r] }} -> pr;
    it -> map {{ t -> {hso} t % {d} -> c; (0.0, sr) -> fold{set} {{ acc, j -> {vr} -> y; acc + pr[{prow} * {s} + j] * y }} }} -> o;
    it -> map: f32 {{ t -> {hso} t % {d} -> c; (0.0, sr) -> fold {{ acc, j -> {vr} -> y; pr[{prow} * {s} + j] * y -> pv; {abs} }} }} -> omag;
    (0.0, isc) -> fold {{ m, t -> smag[t] -> x; (x > m) -> {{ -true-> x; -false-> m; }} }} -> smax;
    smax -> println;
",
        hs = hn * s,
        sa = store(p.fa),
        sb = store(p.fb),
    );
    for c in cells {
        src += &format!("    sc[{c}] -> println;\n    smag[{c}] -> println;\n");
    }
    for c in outs {
        src += &format!("    o[{c}] -> println;\n    omag[{c}] -> println;\n");
    }
    src += "}\n";
    (src, (cells.len(), d, s))
}

// --- fp8 storage (ADR-0043 slice 3) -------------------------------------------

/// A sum of products over fp8 storage: `b` (`k × n`, or `n × k` read
/// transposed under `bt`) rounded once from an `f32` law into `f8e4m3` /
/// `f8e5m2` (`fmt` 0 / 1), at a range that stays normal, saturates past the
/// format's largest finite or reaches its subnormals (`range` 0 / 1 / 2);
/// `a` `f32` or the same fp8 format (`fa`); the fold under `{f32}`,
/// `: bf16 | f32` or `: f16 | f32` (`set` 0 / 1 / 2). The scale is
/// dataflow (`scale`): 0 none; 1 per tensor — the weight's `amax` by two
/// whole max folds, `b = w / (amax / max) -> to_f8`, the epilogue `u · s`;
/// 2 per column — `sc[j]` in the epilogue; 3 a factor inside the fold
/// (`acc + x·y·sf`, constant along `k`: `DepLaw::Factor`). Then cells
/// rounding into the format (every seventh `exp(1000) = inf`: saturates), a
/// loop carrying an fp8 array (`looped`) and a whole fold over widened
/// values.
#[derive(Clone, Debug)]
pub struct F8Prog {
    pub dims: [u8; 3],
    pub fmt: u8,
    pub fa: bool,
    pub set: u8,
    pub bt: bool,
    pub scale: u8,
    pub range: u8,
    pub looped: bool,
    pub law: [u8; 2],
}

pub fn f8_prog_strategy() -> impl Strategy<Value = F8Prog> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 0, 1]),
        prop::bool::weighted(0.4),
        prop::sample::select(vec![0u8, 0, 1, 2]),
        prop::bool::weighted(0.4),
        prop::sample::select(vec![0u8, 1, 1, 2, 3]),
        prop::sample::select(vec![0u8, 0, 0, 1, 2]),
        prop::bool::weighted(0.4),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(
            |(d, fmt, fa, set, bt, scale, range, looped, (l0, l1))| F8Prog {
                dims: [d.0, d.1, d.2],
                fmt,
                fa,
                set,
                bt,
                scale,
                range,
                looped,
                law: [l0, l1],
            },
        )
}

/// The surface name of an [`F8Prog`]'s format.
pub fn f8_fmt(p: &F8Prog) -> &'static str {
    if p.fmt == 1 { "f8e5m2" } else { "f8e4m3" }
}

/// [`F8Prog`]'s source and its checks `(cells, k, m·n)`: the first `cells` pairs
/// of printed lines are a value and its `M = Σ|x·y·sf|` scaled as the value
/// (the epilogue's positive scale) over a `k`-step fold; the later lines
/// pass through a rounding into the format.
pub fn f8_src(p: &F8Prog) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let fmt = f8_fmt(p);
    let max = if p.fmt == 1 { "57344.0" } else { "448.0" };
    let set = match p.set {
        1 => ": bf16 | f32",
        2 => ": f16 | f32",
        _ => "",
    };
    let m = pick(p.dims[0], &[1, 4, 16, 32, 64]);
    let n = pick(p.dims[1], &[16, 32, 64, 128]);
    let k = pick(p.dims[2], &[16, 32, 64, 128, 256]);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let rng = match p.range {
        1 => "* 1000.0",
        2 if p.fmt == 1 => "* 0.000001",
        2 => "* 0.0001",
        _ => "/ 9.0",
    };
    let (mk, kn, mn) = (m * k, k * n, m * n);
    let mut s = format!(
        "fn main() {{\n    {mk} -> iota -> ia;\n    \
         ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> af;\n    \
         {kn} -> iota -> ib;\n    \
         ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v {rng} }} -> bf;\n    \
         {n} -> iota -> inn;\n    {mn} -> iota -> ic;\n    {k} -> iota -> kr;\n"
    );
    // The scale, as dataflow.
    let (post, mpost, factor) = match p.scale {
        1 => {
            s += "    (0.0, ib) -> fold { mx, t -> bf[t] -> x; (x > mx) -> { -true-> x; -false-> mx; } } -> hi;\n    \
                  (0.0, ib) -> fold { mx, t -> 0.0 - bf[t] -> x; (x > mx) -> { -true-> x; -false-> mx; } } -> lo;\n    \
                  (hi > lo) -> { -true-> hi; -false-> lo; } -> amax;\n";
            s += &format!(
                "    amax / {max} -> s;\n    ib -> map {{ t -> bf[t] / s -> to_{fmt} }} -> b;\n"
            );
            (" -> u; u * s".to_string(), " -> u; u * s".to_string(), "")
        }
        2 => {
            s += "    inn -> map { j -> j % 5 + 1 -> widen_f32 -> v; v * 0.25 } -> sc;\n";
            s += &format!("    ib -> map {{ t -> bf[t] -> to_{fmt} }} -> b;\n");
            (
                " -> u; u * sc[j]".to_string(),
                " -> u; u * sc[j]".to_string(),
                "",
            )
        }
        _ => {
            s += &format!("    ib -> map {{ t -> bf[t] -> to_{fmt} }} -> b;\n");
            (
                String::new(),
                String::new(),
                if p.scale == 3 { " * sf" } else { "" },
            )
        }
    };
    if p.scale == 3 {
        s += "    kr[3] -> widen_f32 -> f3;\n    f3 * 0.125 -> sf;\n";
    }
    let ar = if p.fa {
        s += &format!("    ia -> map {{ t -> af[t] -> to_{fmt} }} -> a;\n");
        format!("a[i * {k} + q] -> widen_f32")
    } else {
        format!("af[i * {k} + q]")
    };
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let abs = "(pv > 0.0) -> { -true-> pv; -false-> 0.0 - pv; } -> av; acc + av";
    s += &format!(
        "    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{set} {{ acc, q -> {ar} -> x; {bi} -> widen_f32 -> y; acc + x * y{factor} }}{post} }} -> y;\n    \
         ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {bi} -> widen_f32 -> y; x * y{factor} -> pv; {abs} }}{mpost} }} -> mag;\n"
    );
    let cells = [
        u64::from(p.law[0]) * 7 % mn,
        mn / 2,
        mn - 1 - u64::from(p.law[1]) % n.min(mn),
    ];
    for c in cells {
        s += &format!("    y[{c}] -> println;\n    mag[{c}] -> println;\n");
    }
    // Cells rounding into the format: every seventh `exp(1000) = +inf`,
    // which saturates to the format's largest finite.
    s += &format!(
        "    ic -> map {{ e -> (e % 7 == 0) -> {{ -true-> 1000.0 -> exp -> to_{fmt}; -false-> y[e] -> to_{fmt}; }} }} -> z;\n"
    );
    let fin = if p.looped {
        s += &format!(
            "    mut st: [{fmt}; {mn}] <- z;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 3) -> {{\n            -true-> {{\n                \
             ic -> map {{ e -> st[e] -> widen_f32 -> v; v * 1.5 + 0.25 -> to_{fmt} }} -> st;\n                \
             it * 5 % {mn} -> c;\n                st[c] -> widen_f32 -> sv;\n                sv + 1.0 -> to_{fmt} -> sn;\n                \
             st[c] <- sn;\n                it + 1 -> it;\n                -> loop;\n            }}\n            -false-> st -> fin;\n        }}\n    }}\n"
        );
        "fin"
    } else {
        "z"
    };
    for c in [0, 1, mn / 2, mn - 1] {
        s += &format!("    {fin}[{c}] -> widen_f32 -> println;\n");
    }
    s += &format!(
        "    0.0 -> zero: f32;\n    (zero, {fin}) -> fold {{ acc, v -> v -> widen_f32 -> x; acc + x }} -> sum;\n    sum -> println;\n}}\n"
    );
    (s, (cells.len(), k, mn))
}

/// caching-memory: a growing-history loop — the loop carries a history of `b`
/// sequences × `l` rows × `d` cells and writes the step's input row into it
/// at position `p0 + t` (one row of every sequence; past the end: no write),
/// then recomputes, for every row, a chain of row-local transforms (a row
/// fold of squares and a per-row scale; a capture-free `·2` map; a
/// projection through `n` columns; a rotation reading each row's partner
/// half — a window), and reads every cell of the result for the next input
/// (so a cell a delta missed shows). `wc` is an input the loop carries
/// without ever writing it (a program keeps it between trips): its
/// transpose, recomputed every trip, is the projection's weights. Small
/// integers throughout: every face and fold order agrees with the
/// interpreter.
#[derive(Clone, Debug)]
pub struct HistoryProg {
    pub b: u64,
    pub l: u64,
    pub d: u64,
    pub n: u64,
    pub trips: i64,
    pub p0: i64,
    /// One sequence written as `c ÷ d == p` (a row range, `Rows`) instead of
    /// `(c ÷ d) % l == p` (pieces).
    pub rows_form: bool,
    pub rowfold: bool,
    pub same: bool,
    pub rot: bool,
    /// The projection reads the carried constant's transpose.
    pub constant: bool,
    /// The row is written at the trip's end: the transforms read the history
    /// as the trip found it (its previous trip's row is the new one).
    pub late: bool,
    /// The history's initial value is built before the loop through a
    /// transform that dies there (a prebuilt cache's inputs): the loop's own
    /// buffers may take its bytes (`cone-span`).
    pub prebuilt: bool,
    /// Where the loop runs: 0 in `main`; 1 in a callee `step` (its exit
    /// value the result: `-> ret`) `main` calls once; 2 that callee called
    /// from `main`'s two-trip loop, each result the next call's `x0`. A
    /// callee's device loop leaves its result on the host for the caller.
    pub callee: u8,
    /// A callee's inputs besides `x0`: bit 0 `w`, bit 1 `g`, bit 2 `h0` a
    /// parameter, else built in the callee.
    pub params: u8,
}

pub fn history_strategy() -> impl Strategy<Value = HistoryProg> {
    (
        1u64..4,
        prop::sample::select(vec![4u64, 8, 16]),
        prop::sample::select(vec![4u64, 8, 16]),
        prop::sample::select(vec![4u64, 8, 16]),
        1i64..6,
        0i64..17,
        any::<bool>(),
        (
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        ),
        (0u8..3, 0u8..8),
    )
        .prop_map(
            |(
                b,
                l,
                d,
                n,
                trips,
                p0,
                rows_form,
                (rowfold, same, rot, constant, late, prebuilt),
                (callee, params),
            )| {
                HistoryProg {
                    b,
                    l,
                    d,
                    n,
                    trips,
                    p0: p0 % l as i64,
                    rows_form: rows_form && b == 1,
                    rowfold,
                    same,
                    rot,
                    constant,
                    late,
                    prebuilt,
                    callee,
                    params,
                }
            },
        )
}

/// [`HistoryProg`]'s source text.
pub fn history_prog_src(p: &HistoryProg) -> String {
    let (b, l, d, n) = (p.b, p.l, p.d, p.n);
    let data = |name: &str, len: u64, a: u64, c: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % 5 - 2 -> widen_f32 }} -> {name};\n"
        )
    };
    // The inputs: built in `main` (a callee's parameters) or in the callee.
    let inputs = [
        ("w", d * n, 3, 1, p.params & 1 != 0),
        ("g", d, 2, 3, p.params & 2 != 0),
        ("x0", b * d, 7, 2, true),
        ("h0", b * l * d, 3, 4, p.params & 4 != 0),
    ];
    let (mut head, mut s) = (String::new(), String::new());
    let mut sig = Vec::new();
    for (name, len, a, c, param) in inputs {
        if p.callee == 0 || param {
            head += &data(name, len, a, c);
            sig.push((name, len));
        } else {
            s += &data(name, len, a, c);
        }
    }
    s += &format!(
        "    {} -> iota -> rBL;\n    {d} -> iota -> rD;\n    {} -> iota -> rHD;\n    {} -> iota -> rBLN;\n    \
         {} -> iota -> rLN;\n    {} -> iota -> rBD;\n    {} -> iota -> rDN;\n",
        b * l,
        b * l * d,
        b * l * n,
        l * n,
        b * d,
        d * n
    );
    // A prebuilt history: a scaled copy (dead once the next map has read
    // it, neighbour rows included), then the carried init.
    let init = if p.prebuilt {
        let hd = b * l * d;
        // Small row values around it: one dies before the scaled copy is
        // made, one lives past it (the arena's order of placement matters).
        s += &format!(
            "    rBL -> map {{ r -> h0[r * {d}] * 0.5 }} -> r1;\n    \
             rBL -> map {{ r -> h0[r * {d} + {d} - 1] + 1.0 }} -> r0;\n    \
             rBL -> map {{ r -> r0[r] * r0[(r + 1) % {bl}] }} -> m0;\n    \
             rHD -> map {{ e -> h0[e] * g[e % {d}] + m0[e / {d}] }} -> hs;\n    \
             rHD -> map {{ e -> hs[e] + hs[(e + {d}) % {hd}] + r1[e / {d}] }} -> hb;\n",
            bl = b * l
        );
        "hb"
    } else {
        "h0"
    };
    s += &format!(
        "    mut hx: [f32; {}] <- {init};\n    mut x: [f32; {}] <- x0;\n",
        b * l * d,
        b * d
    );
    if p.constant {
        s += &format!("    mut wc: [f32; {}] <- w;\n", d * n);
    }
    s += &format!(
        "    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n                {} + t -> p;\n",
        p.trips, p.p0
    );
    let cond = if p.rows_form {
        format!("c / {d} == p")
    } else {
        format!("c / {d} % {l} == p")
    };
    let write = format!(
        "rHD -> map {{ c -> ({cond}) -> {{ -true-> x[c / {} * {d} + c % {d}]; -false-> hx[c]; }} }} -> hx;\n",
        l * d
    );
    // A late write reads the step's input before the consumer rebinds it.
    let mut body = if p.late {
        "x -> map { v -> v } -> xin;\n".to_string()
    } else {
        write.clone()
    };
    if p.rowfold {
        body += &format!(
            "rBL -> map {{ r -> (0.0, rD) -> fold {{ acc, i -> hx[r * {d} + i] -> v; acc + v * v }} }} -> rs;\n\
             rHD -> map {{ e -> hx[e] * g[e % {d}] + rs[e / {d}] }} -> xn;\n"
        );
    } else {
        body += &format!("rHD -> map {{ e -> hx[e] * g[e % {d}] }} -> xn;\n");
    }
    let src = if p.same {
        body += "xn -> map { v -> v * 2.0 } -> x2;\n";
        "x2"
    } else {
        "xn"
    };
    let wread = if p.constant {
        body += &format!("rDN -> map {{ e -> wc[e % {d} * {n} + e / {d}] }} -> wt;\n");
        format!("wt[j * {d} + k]")
    } else {
        format!("w[k * {n} + j]")
    };
    body += &format!(
        "rBLN -> map {{ e -> e / {n} -> r; e % {n} -> j; (0.0, rD) -> fold {{ acc, k -> acc + {src}[r * {d} + k] * {wread} }} }} -> pr;\n"
    );
    let res = if p.rot {
        let h = n / 2;
        body += &format!(
            "rBLN -> map {{ e -> e / {n} -> r; e % {n} -> u; (u < {h}) -> {{ -true-> pr[r * {n} + u] * 2.0 - pr[r * {n} + u + {h}]; \
             -false-> pr[r * {n} + u] + pr[r * {n} + u - {h}]; }} }} -> ro;\n"
        );
        "ro"
    } else {
        "pr"
    };
    body += &format!(
        "rBD -> map {{ e -> e / {d} -> bb; e % {d} -> dd; (0.0, rLN) -> fold {{ acc, i -> i % 3 - 1 -> widen_f32 -> cw; \
         acc + {res}[bb * {} + i] * cw }} -> s; dd -> widen_f32 -> df; (s + df > 0.0) -> {{ -true-> g[dd]; -false-> 0.0 - g[dd]; }} }} -> x;\n",
        l * n
    );
    if p.late {
        body += &write.replace("-true-> x[", "-true-> xin[");
    }
    // The carried constant goes round unchanged (a step that hands its
    // weights on, as a step function returning its whole state does).
    if p.constant {
        body += "wc -> wc;\n";
    }
    for line in body.lines() {
        s += &format!("                {line}\n");
    }
    let exit = if p.callee == 0 { "out" } else { "ret" };
    s += &format!(
        "                t + 1 -> t;\n                -> loop;\n            }}\n            -false-> x -> {exit};\n        }}\n    }}\n"
    );
    let mut prints = String::new();
    for c in 0..b * d {
        prints += &format!("    out[{c}] -> println;\n");
    }
    prints += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> sum;\n    sum -> println;\n";
    let params: Vec<String> = sig
        .iter()
        .map(|(x, len)| format!("{x}: [f32; {len}]"))
        .collect();
    let step = format!(
        "fn step({}) -> [f32; {}] {{\n{s}}}\n\n",
        params.join(", "),
        b * d
    );
    // The call's argument: the one parameter, or their product.
    let args = |x0: &str| {
        let names: Vec<&str> = sig
            .iter()
            .map(|&(x, _)| if x == "x0" { x0 } else { x })
            .collect();
        match names.len() {
            1 => names[0].to_string(),
            _ => format!("({})", names.join(", ")),
        }
    };
    match p.callee {
        0 => format!("fn main() {{\n{head}{s}{prints}}}\n"),
        1 => format!(
            "{step}fn main() {{\n{head}    {} -> step -> out;\n{prints}}}\n",
            args("x0")
        ),
        _ => format!(
            "{step}fn main() {{\n{head}    mut xo: [f32; {}] <- x0;\n    mut u: i32 <- 0;\n    loop {{\n        (u < 2) -> {{\n            -true-> {{\n                {} -> step -> xr;\n                xr -> xo;\n                u + 1 -> u;\n                -> loop;\n            }}\n            -false-> xo -> out;\n        }}\n    }}\n{prints}}}\n",
            b * d,
            args("xo")
        ),
    }
}

// --- gemv-stream: matrix-vector sites (`vec-cells`, `stream-fly`) ------------

/// A streaming sum of products: `rows` (1–3) rows of an activation `a` against
/// a weight `w` of `cols` lanes — stored `f32`, `bf16` or `f16` (`fmt` 0 / 1 /
/// 2) — read `k × cols` (`layout` 0), `cols × k` (1: strided lanes), `k ×
/// cols` from an offset `off` elements in (2: the 16-byte alignment the
/// vector load needs holds or not), or one `k × cols` matrix per row (3: `w`
/// moves with the row). `shared`: every row reads one `a` row (`a`'s row
/// stride 0). `post`: an epilogue `u·0.5 + a[i·k]`. Operands non-negative
/// (no cancellation: a reordered sum stays within a relative `k · 2⁻²³`).
#[derive(Clone, Debug)]
pub struct GemvProg {
    pub rows: u64,
    pub cols: u64,
    pub k: u64,
    pub fmt: u8,
    pub layout: u8,
    pub off: u64,
    pub shared: bool,
    pub post: bool,
    pub mul: u64,
}

pub fn gemv_prog_strategy() -> impl Strategy<Value = GemvProg> {
    (
        prop::sample::select(vec![1u64, 1, 2, 3, 3]),
        prop::sample::select(vec![64u64, 72, 96, 100, 128, 256]),
        prop::sample::select(vec![8u64, 24, 64, 96, 128, 256, 512, 1024]),
        prop::sample::select(vec![0u8, 0, 1, 2]),
        prop::sample::select(vec![0u8, 0, 1, 2, 2, 3, 3]),
        prop::sample::select(vec![2u64, 4, 8, 16]),
        prop::bool::weighted(0.3),
        any::<bool>(),
        prop::sample::select(vec![3u64, 7, 11, 13]),
    )
        .prop_map(
            |(rows, cols, k, fmt, layout, off, shared, post, mul)| GemvProg {
                rows,
                cols,
                k,
                fmt,
                layout,
                off,
                shared,
                post,
                mul,
            },
        )
}

/// [`GemvProg`]'s source: the product as a map over `rows × cols` cells
/// folding `k`, every cell's value printed in a fold-free order (first, last,
/// a middle one, the second row's first), then the cells' sum.
pub fn gemv_src(p: &GemvProg) -> String {
    let (rows, cols, k) = (p.rows, p.cols, p.k);
    let n = rows * cols;
    let off = if p.layout == 2 { p.off } else { 0 };
    let wn = match p.layout {
        3 => rows * k * cols,
        _ => k * cols + off,
    };
    let fmt = ["", "bf16", "f16"][usize::from(p.fmt)];
    let wlaw = format!("(e * {} + 5) % 9 + 1 -> widen_f32 -> v; v / 8.0", p.mul);
    let wmap = match p.fmt {
        0 => wlaw,
        _ => format!("{wlaw} -> to_{fmt}"),
    };
    let ar = if p.shared { k } else { rows * k };
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {k} -> iota -> kk;\n    {ar} -> iota -> ta;\n    \
         {wn} -> iota -> tw;\n    \
         ta -> map {{ e -> (e * 7 + 3) % 5 -> widen_f32 -> v; v / 4.0 }} -> a;\n    \
         tw -> map {{ e -> {wmap} }} -> w;\n"
    );
    let wi = match p.layout {
        1 => format!("w[j * {k} + q]"),
        2 => format!("w[q * {cols} + j + {off}]"),
        3 => format!("w[i * {kc} + q * {cols} + j]", kc = k * cols),
        _ => format!("w[q * {cols} + j]"),
    };
    let wv = match p.fmt {
        0 => wi,
        _ => format!("{wi} -> widen_f32"),
    };
    let ai = if p.shared {
        "a[q]".to_string()
    } else {
        format!("a[i * {k} + q]")
    };
    let tail = if p.post {
        format!(
            " -> u; u * 0.5 + a[{}]",
            if p.shared {
                "0".into()
            } else {
                format!("i * {k}")
            }
        )
    } else {
        String::new()
    };
    s += &format!(
        "    t -> map {{ e -> e / {cols} -> i; e % {cols} -> j; (0.0, kk) -> fold {{ acc, q -> {wv} -> b; acc + {ai} * b }}{tail} }} -> y;\n"
    );
    for c in [0, n - 1, n / 2 + 1, cols.min(n - 1)] {
        s += &format!("    y[{c}] -> println;\n");
    }
    s += "    0.0 -> zero: f32;\n    (zero, y) -> fold { acc, v -> acc + v } -> sum;\n    sum -> println;\n}\n";
    s
}

// ---------------------------------------------------------------------------
// flash-attention `stream`: attention written idiomatically, every form the
// stream fact admits or refuses.
// ---------------------------------------------------------------------------

/// **flash-attention generated programs** (`crates/backends/llvm/tests/
/// flash_attention.rs`): `n` query rows of width `d` against `w` keys —
/// the scores `q·Kᵀ` (a tile site), their row extreme (a running max, or a
/// min with the exponent turned around: the rescale down), `exp` of the
/// shifted (scaled) score, its row sum, the probabilities (`÷ den`, a
/// quotient factor) and `P·V` over `c` value columns (a tile site folding the
/// keys). Key counts include ones no window divides (tails); head widths odd
/// ones; the extreme's seed far below the scores, inside them (the seed wins
/// a row: the final rescale), or the identity's neighbour; the sum's seed `0`
/// or `1`. Forms the fact refuses: the `÷ den` after `P·V` (a per-row cell
/// after the windows), a stored probability (`escape`: its final shift), an
/// exponent the extreme does not shift (`phase`).
#[derive(Clone, Debug)]
pub struct FlashProg {
    pub n: u64,
    pub w: u64,
    pub d: u64,
    pub c: u64,
    /// 0 `(s − m) ÷ 2`, 1 `s − m`, 2 `(s − m) × 0.5`.
    pub scale: u8,
    /// The extreme's seed: 0 `−1e9`, 1 `0.0` (inside the scores), 2 `−1e20`.
    pub seed: u8,
    /// A running min and `exp((m − s) ÷ 2)` (the rescale down).
    pub down: bool,
    /// The sum's seed is `1.0` (else `0.0`).
    pub den_one: bool,
    /// 0 `P·V` over `p = ex ÷ den`; 1 `P·V` over `ex`, then `÷ den` a cell
    /// (refused: per-row work after the windows); 2 `p` also printed
    /// (refused: `escape`); 3 `exp(s · m)` (refused: `phase`).
    pub form: u8,
    /// The scores, the extreme and the sum also printed (stored members).
    pub stored: bool,
    /// The model-scale form ([`flash_mh_src`]): `n` tokens, `heads` query
    /// heads of width `d` in groups of `group` sharing a KV head; `0`: the
    /// single-head form above.
    pub heads: u64,
    pub group: u64,
    /// Model-scale: the scores masked causally (`r ≤ i`) to `mask`.
    pub causal: bool,
    /// The mask's value: 0 `−1e9`, 1 `−8` (inside the scores' reach of
    /// `exp`: a masked window adds a term no stand-in absorbs).
    pub mask: u8,
    /// sweep-attention `unit-uniform`: the model-scale form inside a loop
    /// over the KV heads ([`flash_loop_src`]) — each trip one KV head's
    /// query heads, every operand address moved by the loop's counter (a
    /// captured number), the outputs summed into a carried array.
    pub lp: bool,
}

pub fn flash_strategy() -> impl Strategy<Value = FlashProg> {
    (
        (
            prop::sample::select(vec![1u64, 2, 3, 4, 8, 16, 32, 64]),
            prop::sample::select(vec![4u64, 16, 24, 64, 100, 128, 256]),
            prop::sample::select(vec![3u64, 4, 8, 16]),
            prop::sample::select(vec![3u64, 4, 8, 16]),
        ),
        (0u8..3, 0u8..3, any::<bool>(), any::<bool>()),
        (
            prop::sample::select(vec![0u8, 0, 0, 0, 1, 2, 3]),
            any::<bool>(),
        ),
        (
            prop::sample::select(vec![0u64, 0, 2, 4, 8]),
            prop::sample::select(vec![1u64, 2, 4]),
            any::<bool>(),
            0u8..2,
            any::<bool>(),
        ),
    )
        .prop_map(
            |(
                (n, w, d, c),
                (scale, seed, down, den_one),
                (form, stored),
                (heads, group, causal, mask, lp),
            )| FlashProg {
                // The model-scale form at sizes its sites run on the device.
                n: if heads > 0 { n.clamp(16, 64) } else { n },
                w,
                d: if heads > 0 { d.max(8) } else { d },
                c,
                scale,
                seed,
                down,
                den_one,
                form,
                stored,
                heads,
                group: group.min(heads.max(1)),
                causal,
                mask,
                lp: lp && heads > 0,
            },
        )
}

/// [`FlashProg`]'s model-scale source (`heads > 0`): `T = n` tokens, the
/// scores `[head][token][token]` (head-major rows), their row max, `exp` of
/// the shifted scaled score, its row sum, and `P·V` written token-major
/// (`[token][head·d + u]`, the digit permutation of the scores' rows) with
/// the `÷ den` after it — attention as a transformer layer writes it.
pub fn flash_mh_src(p: &FlashProg) -> String {
    let (t, h, g, d) = (p.n, p.heads, p.group, p.d);
    let kvh = h / g;
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen_f32 -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", t * h * d, 7, 13, 11);
    s += &data("kr", t * kvh * d, 5, 3, 9);
    s += &data("vv", t * kvh * d, 3, 1, 13);
    s += &format!(
        "    {d} -> iota -> rdh;\n    {t} -> iota -> rT;\n    {} -> iota -> rHT;\n    {} -> iota -> rHTT;\n    {} -> iota -> rRD;\n",
        h * t,
        h * t * t,
        t * h * d
    );
    let fold = format!(
        "(0.0, rdh) -> fold {{\n                acc, u -> acc + q[i * {} + hh * {d} + u] * kr[r * {} + (hh / {g}) * {d} + u]\n            }}",
        h * d,
        kvh * d
    );
    let mask = ["-1000000000.0", "-8.0"][p.mask as usize];
    let body = match p.causal {
        true => format!(
            "(r <= i) -> {{\n            -true-> {{\n                {fold} -> dt;\n                dt / 2.0\n            }}\n            -false-> {mask};\n        }}"
        ),
        false => format!("{fold} -> dt;\n        dt / 2.0"),
    };
    s += &format!(
        "    rHTT -> map {{\n        e -> e / {} -> hh;\n        e / {t} % {t} -> i;\n        e % {t} -> r;\n        {body}\n    }} -> s;\n",
        t * t
    );
    s += &format!(
        "    rHT -> map {{\n        hi -> (-1000000000.0, rT) -> fold {{\n            acc, r -> s[hi * {t} + r] -> se;\n            (se > acc) -> {{\n                -true-> se;\n                -false-> acc;\n            }}\n        }}\n    }} -> mx;\n"
    );
    s += &format!("    rHTT -> map {{\n        e -> s[e] - mx[e / {t}] -> exp\n    }} -> pe;\n");
    s += &format!(
        "    rHT -> map {{\n        hi -> (0.0, rT) -> fold {{\n            acc, r -> acc + pe[hi * {t} + r]\n        }}\n    }} -> den;\n"
    );
    s += &format!(
        "    rRD -> map {{\n        e -> e / {} -> i;\n        e % {} -> c;\n        c / {d} -> hh;\n        c % {d} -> u;\n        (0.0, rT) -> fold {{\n            acc, r -> acc + pe[(hh * {t} + i) * {t} + r] * vv[r * {} + (hh / {g}) * {d} + u]\n        }} -> av;\n        av / den[hh * {t} + i]\n    }} -> attf;\n",
        h * d,
        h * d,
        kvh * d
    );
    s += &format!(
        "    attf[0] -> println;\n    attf[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, attf) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        t * h * d - 1
    );
    if p.stored {
        s += &format!("    den[{}] -> println;\n", h * t - 1);
    }
    s + "}\n"
}

/// [`FlashProg`]'s head-group loop form (`lp`, the long prefill's shape):
/// a loop over the `H ÷ G` KV heads, each trip the scores of KV head `g`'s
/// `G` query heads `[G][T][T]` (`q` read at head `g·G + hh`, the keys and
/// values at KV head `g`: addresses a captured counter moves), their row
/// max, `exp`, row sum and `P·V ÷ den` token-major (`[T][G·d]`), summed
/// into a carried array — the same attention per trip as
/// [`flash_mh_src`]'s heads, one KV head at a time.
pub fn flash_loop_src(p: &FlashProg) -> String {
    let (t, h, g, d) = (p.n, p.heads, p.group, p.d);
    let kvh = h / g;
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen_f32 -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", t * h * d, 7, 13, 11);
    s += &data("kr", t * kvh * d, 5, 3, 9);
    s += &data("vv", t * kvh * d, 3, 1, 13);
    s += &format!(
        "    {d} -> iota -> rdh;\n    {t} -> iota -> rT;\n    {} -> iota -> rHT;\n    {} -> iota -> rHTT;\n    {} -> iota -> rTG;\n",
        g * t,
        g * t * t,
        t * g * d
    );
    s += &format!(
        "    (0.0, {}) -> fill -> y0;\n    mut y: [f32; {}] <- y0;\n    mut g: i32 <- 0;\n",
        t * g * d,
        t * g * d
    );
    let fold = format!(
        "(0.0, rdh) -> fold {{\n                                acc, u -> acc + q[i * {} + (g * {g} + hh) * {d} + u] * kr[r * {} + g * {d} + u]\n                            }}",
        h * d,
        kvh * d
    );
    let mask = ["-1000000000.0", "-8.0"][p.mask as usize];
    let body = match p.causal {
        true => format!(
            "(r <= i) -> {{\n                            -true-> {{\n                                {fold} -> dt;\n                                dt / 2.0\n                            }}\n                            -false-> {mask};\n                        }}"
        ),
        false => format!("{fold} -> dt;\n                        dt / 2.0"),
    };
    let mut l = String::new();
    l += &format!(
        "                rHTT -> map {{\n                    e -> e / {} -> hh;\n                    e / {t} % {t} -> i;\n                    e % {t} -> r;\n                    {body}\n                }} -> s;\n",
        t * t
    );
    l += &format!(
        "                rHT -> map {{\n                    hi -> (-1000000000.0, rT) -> fold {{\n                        acc, r -> s[hi * {t} + r] -> se;\n                        (se > acc) -> {{\n                            -true-> se;\n                            -false-> acc;\n                        }}\n                    }}\n                }} -> mx;\n"
    );
    l += &format!(
        "                rHTT -> map {{\n                    e -> s[e] - mx[e / {t}] -> exp\n                }} -> pe;\n"
    );
    l += &format!(
        "                rHT -> map {{\n                    hi -> (0.0, rT) -> fold {{\n                        acc, r -> acc + pe[hi * {t} + r]\n                    }}\n                }} -> den;\n"
    );
    l += &format!(
        "                rTG -> map {{\n                    e -> e / {} -> i;\n                    e % {} -> c;\n                    c / {d} -> hh;\n                    c % {d} -> u;\n                    (0.0, rT) -> fold {{\n                        acc, r -> acc + pe[(hh * {t} + i) * {t} + r] * vv[r * {} + g * {d} + u]\n                    }} -> av;\n                    av / den[hh * {t} + i]\n                }} -> attg;\n",
        g * d,
        g * d,
        kvh * d
    );
    l += "                rTG -> map {\n                    e -> y[e] + attg[e]\n                } -> y;\n";
    s += &format!(
        "    loop {{\n        (g < {kvh}) -> {{\n            -true-> {{\n{l}                g + 1 -> g;\n                -> loop;\n            }}\n            -false-> y -> out;\n        }}\n    }}\n"
    );
    s += &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        t * g * d - 1
    );
    s + "}\n"
}

/// [`FlashProg`]'s source text. Data are small integers over 4, the scores
/// a few units wide, so `exp` stays well inside `f32`.
pub fn flash_src(p: &FlashProg) -> String {
    if p.lp {
        return flash_loop_src(p);
    }
    if p.heads > 0 {
        return flash_mh_src(p);
    }
    let (n, w, d, c) = (p.n, p.w, p.d, p.c);
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen_f32 -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", n * d, 7, 13, 11);
    s += &data("kk", w * d, 5, 3, 9);
    s += &data("vv", w * c, 3, 1, 13);
    s += &format!(
        "    {} -> iota -> o_s;\n    {d} -> iota -> k_d;\n    {w} -> iota -> k_t;\n    {n} -> iota -> k_n;\n    {} -> iota -> o_o;\n",
        n * w,
        n * c
    );
    s += &format!(
        "    o_s -> map {{\n        e -> e / {w} -> i;\n        e % {w} -> j;\n        (0.0, k_d) -> fold {{\n            acc, k -> acc + q[i * {d} + k] * kk[j * {d} + k]\n        }}\n    }} -> s;\n"
    );
    let seed = ["-1000000000.0", "0.0", "-100000000000000000000.0"][p.seed as usize];
    let cmp = if p.down { "<" } else { ">" };
    s += &format!(
        "    k_n -> map {{\n        i -> ({seed}, k_t) -> fold {{\n            acc, j -> s[i * {w} + j] -> e;\n            (e {cmp} acc) -> {{\n                -true-> e;\n                -false-> acc;\n            }}\n        }}\n    }} -> mx;\n"
    );
    let diff = if p.down {
        format!("mx[t / {w}] - s[t]")
    } else {
        format!("s[t] - mx[t / {w}]")
    };
    let arg = match (p.form, p.scale) {
        (3, _) => format!("s[t] * mx[t / {w}] / 64.0"),
        (_, 0) => format!("({diff}) / 2.0"),
        (_, 1) => diff,
        _ => format!("({diff}) * 0.5"),
    };
    s += &format!("    o_s -> map {{\n        t -> {arg} -> exp\n    }} -> ex;\n");
    let dseed = if p.den_one { "1.0" } else { "0.0" };
    s += &format!(
        "    k_n -> map {{\n        i -> ({dseed}, k_t) -> fold {{\n            acc, j -> acc + ex[i * {w} + j]\n        }}\n    }} -> den;\n"
    );
    let pv = |src: &str| {
        format!(
            "(0.0, k_t) -> fold {{\n            acc, u -> acc + {src}[i * {w} + u] * vv[u * {c} + j]\n        }}"
        )
    };
    match p.form {
        1 => {
            s += &format!(
                "    o_o -> map {{\n        e -> e / {c} -> i;\n        e % {c} -> j;\n        {}\n    }} -> av;\n    o_o -> map {{\n        e -> av[e] / den[e / {c}]\n    }} -> o;\n",
                pv("ex")
            );
        }
        _ => {
            s += &format!("    o_s -> map {{\n        t -> ex[t] / den[t / {w}]\n    }} -> p;\n");
            s += &format!(
                "    o_o -> map {{\n        e -> e / {c} -> i;\n        e % {c} -> j;\n        {}\n    }} -> o;\n",
                pv("p")
            );
        }
    }
    s += &format!(
        "    o[0] -> println;\n    o[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, o) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        n * c - 1
    );
    if p.stored {
        s += &format!(
            "    s[{}] -> println;\n    mx[{}] -> println;\n    den[0] -> println;\n",
            (n * w) / 2,
            n - 1
        );
    }
    if p.form == 2 {
        s += &format!("    p[{}] -> println;\n", n * w - 1);
    }
    s + "}\n"
}

// --- f32-tensor-cores: split realizations of `{f32}` products ------------------

/// A sum of products under the default face's `{f32}` (no member set) whose
/// operands a split form may realize on a matrix instruction
/// (`mapal_ir::split`): both `f32` (tf32x3), both one narrow format
/// (narrow1), or `f32 × bf16` (no form: the FMA tile). `k` straddles every
/// form's `K_min` (8 … 512); a residual epilogue; `b` row-major or read
/// transposed; value laws a split meets at its edges — small rationals,
/// magnitudes spread over 2^±40, cancelling pairs (`|Σ| ≪ M`), small integers
/// (TF32-exact: `lo = 0`), an infinity and a NaN (the guard), the largest
/// finite value (the `hi` overflow select), operands near the least normal
/// value (parts below it: the absolute term).
#[derive(Clone, Debug)]
pub struct SplitProg {
    pub dims: [u8; 3],
    /// 0: f32 × f32, 1: bf16 × bf16, 2: f16 × f16, 3: f32 × bf16.
    pub store: u8,
    /// 0 rationals, 1 wide, 2 cancelling, 3 integers, 4 inf / NaN, 5 the
    /// largest finite, 6 tiny.
    pub law: u8,
    pub bt: bool,
    pub post: bool,
    pub seed: [u8; 2],
}

pub fn split_prog_strategy() -> impl Strategy<Value = SplitProg> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 0, 1, 1, 2, 3]),
        0u8..7,
        prop::bool::weighted(0.3),
        prop::bool::weighted(0.3),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, store, law, bt, post, (s0, s1))| SplitProg {
            dims: [d.0, d.1, d.2],
            store,
            law,
            bt,
            post,
            seed: [s0, s1],
        })
}

/// [`SplitProg`]'s source and `(cells, m, k)`: each checked cell prints its
/// value, its `M = Σ|a·b|` and `Σ(|a| + |b|)` (the absolute term's operand
/// mass), in that order.
pub fn split_src(p: &SplitProg) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    // Narrow operands: half the draws at sites whose flops per operand byte
    // reach a part's FFMA balance (`m`, `n` 256, `k` 512: 85 flop / B), where
    // the rule (not only its ON arm) takes the split.
    let (m, n, k) = match p.store {
        1 | 2 => (
            pick(p.dims[0], &[32, 128, 256, 256]),
            pick(p.dims[1], &[16, 128, 256, 256]),
            pick(p.dims[2], &[16, 48, 128, 512, 512]),
        ),
        _ => (
            pick(p.dims[0], &[16, 32, 64, 128, 256]),
            pick(p.dims[1], &[8, 16, 32, 64, 128, 256]),
            pick(p.dims[2], &[8, 16, 24, 32, 48, 64, 96, 128, 256, 512]),
        ),
    };
    split_src_at(p, m, n, k)
}

/// [`split_src`] at the given extents.
pub fn split_src_at(p: &SplitProg, m: u64, n: u64, k: u64) -> (String, (usize, u64, u64)) {
    let (a1, a2) = (
        u64::from(p.seed[0] % 13) * 2 + 3,
        u64::from(p.seed[1] % 11) * 2 + 5,
    );
    let (fa, fb) = match p.store {
        1 => (Some("bf16"), Some("bf16")),
        2 => (Some("f16"), Some("f16")),
        3 => (None, Some("bf16")),
        4 => (Some("f8e4m3"), Some("f8e4m3")),
        5 => (Some("f8e5m2"), Some("f8e5m2")),
        _ => (None, None),
    };
    // The narrow formats' range: the wide / tiny / largest laws keep their
    // spread inside it (f16: 2^±14 of headroom).
    let narrow = fa.is_some() || fb.is_some();
    let big = if narrow { "64.0" } else { "1000000000000.0" };
    let small = if narrow { "64.0" } else { "1000000000.0" };
    let p0 = u64::from(p.seed[0]) * 37 % (m * k);
    let p1 = u64::from(p.seed[1]) * 41 % (k * n);
    // `a` along `k` (row-major), `b` along `k` at stride `n` (or 1 when `bt`).
    let law_a = match p.law {
        1 => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 -> x; (t % 3 == 0) -> {{ -true-> x * {big}; -false-> x / {small}; }} -> r; r"
        ),
        2 => format!(
            "((t / 2) * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 -> x; (t % 2 == 0) -> {{ -true-> x; -false-> 0.0 - x; }} -> r; r"
        ),
        3 => format!("(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v"),
        4 => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 -> x; (t == {p0}) -> {{ -true-> 1.0 / 0.0; -false-> x; }} -> r; r"
        ),
        5 if !narrow => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 -> x; (t == {p0}) -> {{ -true-> 340282346638528859811704183484516925440.0; -false-> x; }} -> r; r"
        ),
        6 if !narrow => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 -> x; x / 1000000000000000000000000000000000000.0"
        ),
        _ => format!("(t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0"),
    };
    // Cancelling pairs: `b` equal on the two `k` of a pair (whose `a` cancel)
    // but for a relative 2⁻¹⁰ step.
    let pair = if p.bt {
        "t / 2".to_string()
    } else {
        format!("t / {} * {n} + t % {n}", 2 * n)
    };
    let odd = if p.bt {
        "t % 2".to_string()
    } else {
        format!("t / {n} % 2")
    };
    let law_b = match p.law {
        2 => format!(
            "(({pair}) * {a2} + 57) % 103 - 51 -> widen_f32 -> v; {odd} -> widen_f32 -> s; v / 9.0 + v / 9.0 * s / 1024.0"
        ),
        3 => format!("(t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v"),
        4 => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 -> x; (t == {p1}) -> {{ -true-> 0.0 / 0.0; -false-> x; }} -> r; r"
        ),
        5 if !narrow => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 / 1000000000000000000000000000000.0"
        ),
        6 if !narrow => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 * 100000000000000000000.0"
        ),
        _ => format!("(t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0"),
    };
    let store = |f: Option<&str>| f.map_or(String::new(), |n| format!(" -> to_{n}"));
    let read = |f: Option<&str>, x: &str| match f {
        Some(_) => format!("{x} -> widen_f32"),
        None => x.to_string(),
    };
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let (ar, br) = (read(fa, &format!("a[i * {k} + q]")), read(fb, &bi));
    let post = if p.post {
        format!(
            " -> u; {} -> r; u * 0.5 + r",
            read(fa, &format!("a[i * {k}]"))
        )
    } else {
        String::new()
    };
    let abs =
        |x: &str, o: &str| format!("({x} > 0.0) -> {{ -true-> {x}; -false-> 0.0 - {x}; }} -> {o}");
    let cells = [
        u64::from(p.seed[0]) * 7 % (m * n),
        m * n / 2 + 1,
        m * n - 1 - u64::from(p.seed[1]) % n,
    ];
    // An integer literal, a negative one as a difference.
    let int = |v: i64| match v < 0 {
        true => format!("(0 - {})", -v),
        false => v.to_string(),
    };
    let mut src = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> {law_a} }} -> a0;
    a0 -> map {{ x -> x{sa} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> {law_b} }} -> b0;
    b0 -> map {{ x -> x{sb} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; acc + x * y }}{post} }} -> y;
    3 -> iota -> ck;
    ck -> map: f32 {{ s -> {c0} + s * {d1} + s * (s - 1) / 2 * {d2} -> t; t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; x * y -> pv; {pabs}; acc + av }} }} -> mag;
    ck -> map: f32 {{ s -> {c0} + s * {d1} + s * (s - 1) / 2 * {d2} -> t; t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; {xabs}; {yabs}; acc + ax + ay }} }} -> sab;
",
        mk = m * k,
        kn = k * n,
        mn = m * n,
        sa = store(fa),
        sb = store(fb),
        pabs = abs("pv", "av"),
        xabs = abs("x", "ax"),
        yabs = abs("y", "ay"),
        // The three checked cells as `c0 + s·d1 + s(s−1)/2·d2`, `s` = 0, 1, 2:
        // `M` and `Σ(|a| + |b|)` of those cells only (the interpreter's
        // budget at 256 × 256 × 512).
        c0 = cells[0],
        d1 = int(cells[1] as i64 - cells[0] as i64),
        d2 = int(cells[2] as i64 - 2 * cells[1] as i64 + cells[0] as i64),
    );
    for (s, c) in cells.iter().enumerate() {
        src += &format!(
            "    y[{c}] -> println;\n    mag[{s}] -> println;\n    sab[{s}] -> println;\n"
        );
    }
    src += "}\n";
    (src, (cells.len(), m, k))
}

// --- hopper-wgmma-tma: TF32 GEMMs at warpgroup-tile extents -----------------

/// A TF32 GEMM `y = a · b` at extents a warpgroup tile divides or misses:
/// `m` 8–320, `n` 64–384, `k` 8–640 (from one tile below a box to twenty
/// stages: the barrier phases wrap), `a` rows `ci` apart (`pad`: 0 dense, 4 a
/// granule-whole pad, 3 a misaligned one the box refuses), the operands at an
/// offset (`off`: whole granules or not), `b` read as `k × n` or transposed
/// (`bt`: `n × k`, no `c × k` box). Integer-valued operands over 7 / 9: the
/// TF32 rounding of each operand is exercised, the bound of D3 judges the sum.
#[derive(Clone, Debug)]
pub struct WgmmaProg {
    pub dims: [u8; 3],
    pub pad: u8,
    pub off: u8,
    pub bt: bool,
    pub law: [u8; 2],
}

pub fn wgmma_prog_strategy() -> impl Strategy<Value = WgmmaProg> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::sample::select(vec![0u8, 0, 0, 4, 3]),
        prop::sample::select(vec![0u8, 0, 0, 4, 1]),
        prop::bool::weighted(0.15),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(dims, pad, off, bt, law)| WgmmaProg {
            dims,
            pad,
            off,
            bt,
            law,
        })
}

/// [`WgmmaProg`]'s source (`main` on the TF32 face) and its `(m, n, k)`.
/// Prints two cells of `y`, each followed by its `M` (`Σ |a·b|`).
pub fn wgmma_src(p: &WgmmaProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(p.dims[0], &[8, 16, 64, 128, 192, 256, 320]);
    let n = pick(p.dims[1], &[64, 128, 192, 256, 384]);
    let k = pick(p.dims[2], &[8, 32, 64, 96, 128, 256, 640]);
    let ci = k + u64::from(p.pad);
    let (oa, ob) = (u64::from(p.off), u64::from(p.off) * 2);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[{ob} + j * {k} + k]")
    } else {
        format!("b[{ob} + k * {n} + j]")
    };
    let ai = format!("a[{oa} + i * {ci} + k]");
    let (c0, c1) = (
        u64::from(p.law[0]) * 7 % (m * n),
        m * n - 1 - u64::from(p.law[1]) % n,
    );
    let src = format!(
        "fn main(): tf32 | f32 {{
    {na} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    {nb} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> acc + {ai} * {bi} }} }} -> y;
    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> {ai} * {bi} -> p; (p > 0.0) -> {{ -true-> p; -false-> 0.0 - p; }} -> q; acc + q }} }} -> mag;
    y[{c0}] -> println;
    mag[{c0}] -> println;
    y[{c1}] -> println;
    mag[{c1}] -> println;
}}
",
        na = oa + m * ci,
        nb = ob + k * n,
        mn = m * n
    );
    (src, (m, n, k))
}
