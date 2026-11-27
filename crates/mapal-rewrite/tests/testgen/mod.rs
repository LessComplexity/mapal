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
#![allow(unused_imports)]

use mapal_interp::RValue;
use mapal_ir::{CategoryIr, Dest, FuncId, FuncKind, IrBuilder, Operation, SourceLoc, Ty, Value};
use proptest::prelude::*;

mod decode;
mod device_chain;
mod f8_history;
mod flash;
mod format_laws;
mod generics;
mod geom_tile;
mod half;
mod math_gemm;
mod onto_layer;
mod script;
mod split_wgmma;
mod steps;
mod twin_recur;
mod zero_hz;

pub use self::decode::*;
pub use self::device_chain::*;
pub use self::f8_history::*;
pub use self::flash::*;
pub use self::format_laws::*;
pub use self::generics::*;
pub use self::geom_tile::*;
pub use self::half::*;
pub use self::math_gemm::*;
pub use self::onto_layer::*;
pub use self::script::*;
pub use self::split_wgmma::*;
use self::steps::*;
pub use self::twin_recur::*;
pub use self::zero_hz::*;

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

/// [`par_check`] for checks that return a fixed set of counters: their sums,
/// in the order the check returns them.
#[allow(dead_code)]
pub fn par_count<T: Sync, const N: usize>(
    items: &[T],
    check: impl Fn(usize, &T) -> [usize; N] + Sync,
) -> [usize; N] {
    par_check(items, check).iter().fold([0; N], |mut s, c| {
        for (a, b) in s.iter_mut().zip(c) {
            *a += b;
        }
        s
    })
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
