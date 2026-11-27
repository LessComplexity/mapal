//! Task definitions, slice planning and the worker-pool state types.
use super::*;

/// One emitted task body.
///
/// `lo..hi` is the task's complete sequential range or one disjoint slice of a
/// split task. `frame` is the emitted function frame and remains valid through
/// `mapal_par_finish`.
pub type TaskFn = extern "C" fn(lo: i64, hi: i64, frame: *mut u8);

pub(crate) const GRAIN: i64 = 4096;

#[derive(Clone, Copy)]
pub(crate) struct TaskDef {
    pub(crate) kind: u32,
    pub(crate) f: TaskFn,
    pub(crate) n: i64,
    pub(crate) rank: u32,
    /// Elements per slice, decided at COMPILE TIME (plan-s32 §2.5: the compiler
    /// sizes, the runtime assigns). `0` = none supplied, keep the legacy rule.
    pub(crate) slice_elems: i64,
    /// How many pieces per lane the compiler wants, given the region's reuse
    /// structure. `0` = none supplied. A read that is row-invariant pays nothing
    /// at a slice boundary and wants over-decomposition so stealing can balance;
    /// a sliding read re-pays its window overlap at every boundary and does not.
    /// The compiler knows which from the recorded coefficients; only the runtime
    /// knows how many lanes exist, so the two multiply here.
    pub(crate) oversub: u32,
    /// Lanes this dispatch should spread across, decided at compile time.
    /// `0` = none supplied, use the whole pool. Stealing is deliberately NOT
    /// confined to them: an idle lane helping is the runtime's assignment
    /// freedom, which is the half of the decision the compiler cannot make.
    pub(crate) width: u32,
}

/// Task lifecycle, stored as a `u8` in [`TaskRt::state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TaskState {
    Waiting = 0,
    Running = 1,
    Done = 2,
}

impl TaskState {
    pub(crate) fn of(raw: u8) -> Self {
        match raw {
            0 => Self::Waiting,
            1 => Self::Running,
            _ => Self::Done,
        }
    }
}

/// A task while it is being registered. Pre-launch only, single-threaded by
/// contract (`mapal_par_begin` .. `mapal_par_launch` is the emitter's host
/// spine), and the mutex here is only what makes that contract checkable.
pub(crate) struct TaskBuild {
    pub(crate) def: Option<TaskDef>,
    pub(crate) dependents: Vec<usize>,
    pub(crate) deps: u32,
    pub(crate) pinned: bool,
}

/// A task while the run executes (plan-s50 P2).
///
/// THE SPLIT IS THE POINT. `def`, `dependents` and `pinned` are decided before
/// launch and never written again — the run graph is a compile-time constant
/// the emitter replays — so after `seal` they are read with no lock at all.
/// Only the four counters move, and each is an atomic.
///
/// What this replaces: ONE `Mutex<RunState>` taken twice per slice (once in
/// `execute` to flip `Running`, once in `complete_slice` to decrement), plus a
/// `dependents.clone()` heap allocation on every task completion. With 64
/// slices across 14 lanes that is ~128 acquisitions of one mutex per task, and
/// it got worse exactly where S50 measured a regression: `nn2x2` on the 32-lane
/// i9, where two independent chains put four sites' completions through one
/// lock that used to be four separate nested-run locks.
pub(crate) struct TaskRt {
    pub(crate) def: TaskDef,
    pub(crate) dependents: Box<[usize]>,
    pub(crate) pinned: bool,
    pub(crate) deps_left: AtomicU32,
    /// A [`TaskState`] discriminant.
    pub(crate) state: AtomicU8,
    pub(crate) slices_left: AtomicUsize,
    pub(crate) watermark: AtomicI64,
}

pub(crate) struct Run {
    pub(crate) pool: Arc<Pool>,
    pub(crate) frame: AtomicPtr<u8>,
    pub(crate) trap: AtomicU64,
    /// Registration state; consumed by `seal` at launch.
    pub(crate) build: Mutex<Vec<TaskBuild>>,
    /// Sealed at launch. `Some` IS "launched", so there is no separate flag to
    /// keep in agreement with it.
    pub(crate) tasks: OnceLock<Box<[TaskRt]>>,
    pub(crate) remaining: AtomicUsize,
    pub(crate) epoch: AtomicU64,
    /// The condvar's companion, and nothing else. Every value a waiter tests is
    /// an atomic; this exists so a wake cannot be lost between the test and the
    /// wait.
    pub(crate) gate: Mutex<()>,
    pub(crate) changed: Condvar,
}

/// Opaque handle for one emitted task DAG invocation.
///
/// The emitter owns this handle from `mapal_par_begin` through exactly one
/// `mapal_par_finish`; all registration happens before `mapal_par_launch`.
#[repr(C)]
pub struct FlowParRun {
    pub(crate) run: Arc<Run>,
}

pub(crate) struct Work {
    pub(crate) run: Arc<Run>,
    pub(crate) task: usize,
    pub(crate) lo: i64,
    pub(crate) hi: i64,
}

/// One lane's parking spot (plan-s50 P0b).
///
/// There used to be ONE `wake_epoch`/`wake` pair for the whole pool, and
/// `enqueue` ended in `notify_all` — so a dispatch carrying four slices woke
/// fourteen threads, ten of which locked every queue in `take_any`, found
/// nothing, and re-parked contending on that one mutex. Measured: an empty
/// pooled dispatch cost 0.42 us at 2 lanes and 9.8 us at 10, for the same zero
/// work (`examples/dispatch_cost.rs`).
///
/// A per-lane spot lets `enqueue` wake EXACTLY the lanes it filled, which is
/// exact rather than tuned — the placement loop already decides that set. Lanes
/// it did not fill stay asleep, and the lanes it did fill still steal from each
/// other through `take_any`, so balance inside the working set is unchanged.
/// That matters: S33 measured that work stealing absorbing core heterogeneity
/// is the whole of mapal's edge over OpenBLAS on a hybrid part (4P+4E: 1.65x
/// ahead, while 8 uniform P-cores put OpenBLAS 41% ahead).
pub(crate) struct LanePark {
    pub(crate) epoch: Mutex<u64>,
    pub(crate) wake: Condvar,
}

pub(crate) struct Pool {
    pub(crate) threads: usize,
    pub(crate) queues: Vec<Mutex<VecDeque<Work>>>,
    pub(crate) parks: Vec<LanePark>,
    /// Total items queued across every lane (plan-s50 P1).
    ///
    /// A spinning executor needs a work-available test that is ONE relaxed load.
    /// `take_any`'s miss path locks every lane in turn, so spinning on it would
    /// replace a sleep with `threads` mutex acquisitions per iteration. This
    /// counter is advisory — the parking path below is still the authority — so
    /// a stale read costs at most one wasted `take_any`.
    pub(crate) pending: AtomicUsize,
    /// Spin iterations an executor burns before it parks. `0` = park at once.
    pub(crate) spin: u32,
}

/// How long the HOST spins before sleeping, and when it must not spin at all.
///
/// **Workers do not spin, and that is a measured result, not an omission.**
/// `examples/handoff_floor.rs` said a spinning wave costs ~1 us against 11-41 us
/// parked, a 30-40x prize. It did not transfer, and the instrument is why: it
/// gave EVERY thread work on EVERY wave. A real dispatch does not — `nn4` at
/// d=128 cuts four slices for fourteen lanes — so spinning workers are *idle*
/// threads burning the cores the four working lanes need, and every idle one
/// that sees the shared `pending` counter move races into `take_any`, which
/// locks every queue. Measured with workers spinning (M4, MAPAL_PAR=14, median
/// of 31, against the same tree with the spin removed): `nn4_512` **0.80x**,
/// `nn2x2_512` 0.87x, `matmul1024` 0.88x, `nn4_1024` 0.93x. Swept 16..4096 and
/// no budget recovered it. Refuted and recorded.
///
/// The host is the exception, and structurally so: there is exactly one of it,
/// it publishes the dispatch and it is what learns the wave finished, so it is
/// on the wave's critical path and cannot herd. Host-only, same conditions:
/// `nn4_128` **1.18x**, everything else 0.98-1.02.
///
/// THE BUDGET IS DERIVED FROM THE THING IT REPLACES. Spinning is only ever worth
/// it up to what a park-and-wake would have cost — 11-41 us measured — because
/// past that you would have been better off asleep. At roughly 1 ns per
/// `spin_loop`, 8192 iterations is ~8 us, the low end of that band. The sweep
/// agrees: `nn4_128` reads 1.12x at 256, 1.18x at 2048, 1.14x at 8192, 1.19x at
/// 32768 — a plateau, not a peak, so the smallest value on it is the honest
/// choice rather than the largest.
///
/// AND IT MUST BE ZERO WHEN THE POOL IS OVERSUBSCRIBED. `handoff_floor` at the
/// boundary: 13 executors 1.083 us, 14 executors **1640.917 us** — a 1500x
/// cliff, because the spinners and the host fight over the same cores. That is a
/// correctness bound on a performance knob, not a preference. `MAPAL_SPIN`
/// overrides for sweeping, like `MAPAL_SLICE`.
pub(crate) fn spin_iters(threads: usize) -> u32 {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    if let Some(forced) = *OVERRIDE.get_or_init(|| {
        std::env::var("MAPAL_SPIN")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
    }) {
        return forced;
    }
    let cores = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    // `Pool::new` spawns `threads - 1` workers and the host is the last
    // executor, so `threads` IS the runnable count.
    if threads <= cores { 8192 } else { 0 }
}

/// plan-s50 P0: there is one placement rule now. `Local(lane)` — put a
/// dep-unlocked task on the lane that finished its producer — was affinity for
/// scalar glue and a serialization hazard for anything bulk, which is what a
/// packed tile site's matmul became once it stopped hiding in a nested run.
#[derive(Clone, Copy)]
pub(crate) enum Placement {
    Seed,
}

/// How many pieces a dispatch is cut into, and therefore whether work stealing
/// has anything to steal.
///
/// **The legacy rule (`slice_elems == 0`) is `ceil(n/GRAIN).min(threads)` —
/// exactly one piece per worker.** That leaves the deques empty, so a fast lane
/// finishing early cannot help a slow one: the dispatch ends when the slowest
/// piece ends, and on an asymmetric machine (10 P + 4 E) that is the whole wave
/// waiting on an E-core. Measured cost: matmul512 at 14 lanes went 0.778 ms to
/// 0.463 ms purely from cutting more pieces (plan-s32 §1(c)).
///
/// A compile-time `slice_elems` overrides it. It is NOT simply "smaller is
/// better" — finer slicing re-pays every boundary's reuse halo and shortens the
/// run over which a packed panel amortises, which is why the same 8x
/// over-decomposition that wins 68% on matmul512 costs conv2d ~20% (§2.6). The
/// size is a per-region decision; this function only applies it.
/// Experiment lever: override every dispatch's slice size (`MAPAL_SLICE=<elems>`).
///
/// Not part of the design — the compiler is what sets slice sizes (plan-s32
/// §2.5). This exists so the sizing model can be swept and falsified on real
/// kernels *before* the deduction is written, and so ADR-0034's autotuner has a
/// knob to search. `MAPAL_PAR`'s sibling, with the same status.
pub(crate) fn slice_override() -> i64 {
    static OVERRIDE: OnceLock<i64> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MAPAL_SLICE")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(0)
    })
}

pub(crate) fn slice_ranges(def: TaskDef, threads: usize) -> Vec<(i64, i64)> {
    // The lever forces a slice size, bypassing the floor/oversub rule — that is
    // what makes it a probe of the sizing model rather than of the rule.
    //
    // It may NOT bypass the task's declared QUANTUM, though, and that is a
    // correctness bound rather than a tuning one. `slice_elems` is the emitter's
    // statement of the granularity its kernel can start on, and some kernels
    // depend on it: the SME rung walks whole `ti*t`-row panels from `lo/c` with
    // no partial-panel path, so a slice starting mid-panel makes the final panel
    // write past the end of the output. Forcing a misaligned size used to turn
    // that into a segfault (reproduced: `MAPAL_SLICE=16384` on a matmul1024 SME
    // build). Round the forced size UP to the declared quantum: the lever still
    // probes the size — which is what it exists for — and can no longer produce
    // a start the kernel was never told to expect.
    let forced = slice_override();
    let def = if forced > 0 {
        let quantum = def.slice_elems.max(1);
        let aligned = ((forced + quantum - 1) / quantum).saturating_mul(quantum);
        TaskDef {
            slice_elems: aligned,
            oversub: u32::MAX,
            ..def
        }
    } else {
        def
    };
    let slices = if def.kind == 0 {
        1
    } else if def.n == 0 {
        0
    } else if def.slice_elems > 0 {
        // `slice_elems` is a FLOOR, not an exact size: cutting below it drops
        // the dispatch off the register-blocked path onto the TI=1 fallback,
        // which measured 7x on matmul1024 (plan-s32 §2.6). Within that floor the
        // runtime picks the count, because lane count is the one input the
        // compiler does not have.
        // Cut EQUAL numbers of blocks, never an arbitrary count.
        //
        // A count that does not divide the block total leaves slices of
        // different sizes, and the cost is out of all proportion to the 1.25x
        // size ratio it creates. Measured on matmul1024 at 14 lanes (256
        // blocks): every count that divides 256 runs 2.49-2.73 ms (32, 64, 128,
        // 256) while neighbouring counts that do not run 3.3-5.8 (41, 43, 47,
        // 52, 57) — 1.8x for being ragged. Deriving blocks-per-slice first and
        // the count from it makes ragged counts unreachable.
        let blocks = usize::try_from((def.n as u64).div_ceil(def.slice_elems as u64))
            .unwrap_or(usize::MAX)
            .max(1);
        let wanted = threads.saturating_mul(def.oversub.max(1) as usize).max(1);
        let per = (blocks / wanted).max(1);
        blocks.div_ceil(per)
    } else {
        usize::try_from((def.n as u64).div_ceil(GRAIN as u64))
            .unwrap_or(usize::MAX)
            .min(threads)
    };
    // Cut on the region's quantum, not on equal element counts.
    //
    // Equal division is what the count-first rule did, and it silently destroys
    // whatever alignment the size asked for: 1048576 over 52 slices is 20164.9
    // elements, so every boundary lands mid-row, every slice starts with a
    // partial register block, and the blocked kernel degenerates at both ends of
    // every piece. Measured on matmul1024 at 14 lanes: slice counts that happen
    // to divide n evenly (32, 64, 128) run 2.5-2.6 ms, while neighbouring counts
    // that do not (43, 52, 57) run 4.4-5.8 — a 2x swing with no relation to how
    // many pieces there are. `slice_elems` carries the quantum (TI * c, one
    // whole register block of rows), so honour it.
    let quantum = def.slice_elems.max(1);
    let mut ranges = Vec::with_capacity(slices);
    let mut lo = 0;
    for slice in 1..=slices {
        let hi = if slice == slices {
            def.n
        } else {
            let target = (slice as i64).saturating_mul(def.n) / slices as i64;
            (target / quantum * quantum).clamp(lo, def.n)
        };
        // Every slice is emitted, empty ones included: a `Seq` task carries
        // n = 0 and must still run exactly once.
        ranges.push((lo, hi));
        lo = hi;
    }
    ranges
}

#[derive(Clone, Copy)]
pub(crate) struct CurrentRun {
    pub(crate) run: *const Run,
    pub(crate) task: usize,
}

thread_local! {
    pub(crate) static CURRENT_RUNS: RefCell<Vec<CurrentRun>> = const { RefCell::new(Vec::new()) };
    pub(crate) static WORKER_LANE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap()
}

pub(crate) fn parse_cpu_max(text: &str) -> Option<usize> {
    let mut fields = text.split_whitespace();
    let quota = fields.next()?;
    let period = fields.next()?;
    if fields.next().is_some() || quota == "max" {
        return None;
    }
    let quota = quota.parse::<u64>().ok()?;
    let period = period.parse::<u64>().ok()?;
    (period > 0)
        .then(|| quota.div_ceil(period).max(1))
        .and_then(|threads| usize::try_from(threads).ok())
}

pub(crate) fn parse_cfs(quota: &str, period: &str) -> Option<usize> {
    let quota = quota.trim().parse::<i64>().ok()?;
    let period = period.trim().parse::<u64>().ok()?;
    if quota < 0 || period == 0 {
        return None;
    }
    usize::try_from((quota as u64).div_ceil(period).max(1)).ok()
}

pub(crate) fn cgroup_quota() -> Option<usize> {
    std::fs::read_to_string("/sys/fs/cgroup/cpu.max")
        .ok()
        .and_then(|text| parse_cpu_max(&text))
        .or_else(|| {
            let quota = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").ok()?;
            let period = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us").ok()?;
            parse_cfs(&quota, &period)
        })
}

pub(crate) fn thread_count(
    mapal_par: Option<&str>,
    available: usize,
    quota: Option<usize>,
) -> usize {
    mapal_par
        .and_then(|value| value.parse().ok())
        .filter(|&value| value >= 1)
        .unwrap_or_else(|| quota.map_or(available, |quota| available.min(quota)).max(1))
}

pub(crate) fn configured_threads() -> usize {
    let mapal_par = std::env::var("MAPAL_PAR").ok();
    let available = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    thread_count(mapal_par.as_deref(), available, cgroup_quota())
}

pub(crate) fn global_pool() -> Arc<Pool> {
    static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(configured_threads())).clone()
}
