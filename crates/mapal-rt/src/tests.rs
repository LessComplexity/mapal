//! Tests for the runtime seam: arena, reads, the pool and the parallel entry points.

use super::*;
use mapal_interp::render;
use mapal_ir::Value;
use std::{
    sync::{
        Barrier,
        atomic::{AtomicBool, AtomicU8, AtomicU32},
        mpsc,
    },
    time::{Duration, Instant},
};

/// The arena hands out usable, distinct storage and releases it. Writing
/// the whole block is the point: a short allocation would corrupt the heap
/// silently, which is exactly what the emitter's size arithmetic risks.
///
/// The third block covers `reside` (plan-s33 CR-1): it is larger than one
/// page and deliberately **not** a page multiple, so the pre-touch loop has
/// to walk many pages and stop inside the final partial one. A stride or
/// bound slip there writes past the block and corrupts the allocator, which
/// the writes-then-free below turn into a hard failure. Residence *itself*
/// is not observable from Rust — it is pinned on the box by differencing
/// `perf stat -e page-faults` (plan-s33 acceptance 1).
#[test]
fn arena_alloc_is_usable_and_freed() {
    let a = mapal_rt_alloc(4096, 64);
    let b = mapal_rt_alloc(4096, 64);
    let big_bytes = 1024 * 1024 + 1; // > 1 page, not a page multiple
    let c = mapal_rt_alloc(big_bytes as i64, 64);
    assert!(!a.is_null() && !b.is_null() && !c.is_null());
    assert!(a != b && b != c && a != c);
    assert_eq!(a as usize % 64, 0, "requested alignment honoured");
    assert_eq!(c as usize % 64, 0, "requested alignment honoured");
    unsafe {
        // `reside` writes a 0 per page, so a freshly handed-out block reads
        // as zero at every page boundary it touched, including the last one.
        assert_eq!(*c, 0);
        assert_eq!(*c.add(big_bytes - 1 - (big_bytes - 1) % 4096), 0);
        ptr::write_bytes(a, 0xAB, 4096);
        ptr::write_bytes(b, 0xCD, 4096);
        ptr::write_bytes(c, 0xEF, big_bytes);
        assert_eq!(*a.add(4095), 0xAB);
        assert_eq!(*b.add(4095), 0xCD);
        assert_eq!(
            *c.add(big_bytes - 1),
            0xEF,
            "last byte of the block is ours"
        );
    }
    assert!(lock(&ARENA).len() >= 3);
    mapal_rt_free_all();
    assert!(lock(&ARENA).is_empty());
}

#[test]
fn parse_cpu_max_cases() {
    assert_eq!(parse_cpu_max("max 100000"), None);
    assert_eq!(parse_cpu_max("4800000 100000"), Some(48));
    assert_eq!(parse_cpu_max("100 100000"), Some(1));
    assert_eq!(parse_cpu_max("garbage"), None);
}

#[test]
fn parse_cfs_cases() {
    assert_eq!(parse_cfs("-1", "100000"), None);
    assert_eq!(parse_cfs("4800000", "100000"), Some(48));
    assert_eq!(parse_cfs("100", "100000"), Some(1));
    assert_eq!(parse_cfs("garbage", "100000"), None);
    assert_eq!(parse_cfs("100", "0"), None);
}

#[test]
fn mapal_par_override_wins() {
    assert_eq!(thread_count(Some("7"), 64, Some(2)), 7);
}

#[test]
fn quota_caps_available() {
    // The S24 box shape: 384 visible host threads, ≈48-core cgroup quota.
    assert_eq!(thread_count(None, 384, Some(48)), 48);
    assert_eq!(thread_count(None, 8, Some(48)), 8);
    assert_eq!(thread_count(None, 8, None), 8);
}

/// Render-parity table (DESIGN §1): every scalar mapal-rt prints must be
/// byte-identical to `mapal_interp::render`. mapal-rt prints `v` via
/// `Display`; assert that string equals the oracle's render for the same
/// value. Covers the pinned tricky cases: `4080.0 → "4080"`, `5.375`,
/// `-0.0`, `NaN`, `inf`, u8 `255`, bools, i64 extremes.
#[test]
fn render_parity() {
    macro_rules! check {
        ($disp:expr, $val:expr) => {{
            let rt = format!("{}", $disp); // what mapal-rt's emit() writes
            let oracle = render(&$val);
            assert_eq!(rt, oracle, "parity mismatch for {:?}", $val);
        }};
    }

    // f64 — the formatting-sensitive cases.
    check!(4080.0_f64, Value::F64(4080.0));
    check!(5.375_f64, Value::F64(5.375));
    check!(-0.0_f64, Value::F64(-0.0));
    check!(0.0_f64, Value::F64(0.0));
    check!(f64::NAN, Value::F64(f64::NAN));
    check!(f64::INFINITY, Value::F64(f64::INFINITY));
    check!(f64::NEG_INFINITY, Value::F64(f64::NEG_INFINITY));
    check!(f64::MAX, Value::F64(f64::MAX));

    // f32.
    check!(4080.0_f32, Value::F32(4080.0));
    check!(5.375_f32, Value::F32(5.375));
    check!(-0.0_f32, Value::F32(-0.0));
    check!(f32::NAN, Value::F32(f32::NAN));
    check!(f32::INFINITY, Value::F32(f32::INFINITY));

    // u8 — incl. the high half (255, > 127).
    check!(0_u8, Value::U8(0));
    check!(127_u8, Value::U8(127));
    check!(128_u8, Value::U8(128));
    check!(255_u8, Value::U8(255));

    // bools.
    check!(true, Value::Bool(true));
    check!(false, Value::Bool(false));

    // Str — mapal_print_str renders bytes via Display (== render(Str) == s).
    check!("hello", Value::Str("hello".into()));
    check!("", Value::Str("".into()));
    check!("4080", Value::Str("4080".into()));

    // i32 / i64 extremes.
    check!(0_i32, Value::I32(0));
    check!(-1_i32, Value::I32(-1));
    check!(i32::MIN, Value::I32(i32::MIN));
    check!(i32::MAX, Value::I32(i32::MAX));
    check!(i64::MIN, Value::I64(i64::MIN));
    check!(i64::MAX, Value::I64(i64::MAX));
}

fn test_run(n_tasks: u32, threads: usize) -> *mut FlowParRun {
    allocate_run(n_tasks, Pool::new(threads))
}

fn frame_ptr<T>(frame: &T) -> *mut u8 {
    ptr::from_ref(frame).cast_mut().cast()
}

fn wait_entry(task: u32, threshold: u32) -> u64 {
    (u64::from(task) << 32) | u64::from(threshold)
}

fn completion_entry(task: u32) -> u64 {
    wait_entry(task, u32::MAX)
}

unsafe fn trap_flag(handle: *mut FlowParRun) -> u64 {
    unsafe { run_handle(handle) }
        .run
        .trap
        .load(Ordering::SeqCst)
}

unsafe fn clear_trap(handle: *mut FlowParRun) {
    unsafe { run_handle(handle) }
        .run
        .trap
        .store(0, Ordering::SeqCst);
}

struct SplitFrame {
    n: usize,
    hits: Vec<AtomicU64>,
    calls: AtomicU32,
    bad_range: AtomicBool,
    duplicate: AtomicBool,
}

extern "C" fn hit_slice(lo: i64, hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<SplitFrame>() };
    frame.calls.fetch_add(1, Ordering::SeqCst);
    if lo < 0 || hi < lo || hi as usize > frame.n {
        frame.bad_range.store(true, Ordering::SeqCst);
        return;
    }
    for index in lo as usize..hi as usize {
        let bit = 1_u64 << (index % 64);
        if frame.hits[index / 64].fetch_or(bit, Ordering::SeqCst) & bit != 0 {
            frame.duplicate.store(true, Ordering::SeqCst);
        }
    }
}

struct ZeroFrame {
    split_calls: AtomicU32,
    dependent_calls: AtomicU32,
}

extern "C" fn zero_split(_lo: i64, _hi: i64, frame: *mut u8) {
    unsafe { &*frame.cast::<ZeroFrame>() }
        .split_calls
        .fetch_add(1, Ordering::SeqCst);
}

extern "C" fn zero_dependent(_lo: i64, _hi: i64, frame: *mut u8) {
    unsafe { &*frame.cast::<ZeroFrame>() }
        .dependent_calls
        .fetch_add(1, Ordering::SeqCst);
}

#[test]
fn split_covers_each_index_once() {
    let n = (GRAIN * 3 + 17) as usize;
    let frame = Box::new(SplitFrame {
        n,
        hits: (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        calls: AtomicU32::new(0),
        bad_range: AtomicBool::new(false),
        duplicate: AtomicBool::new(false),
    });
    let handle = test_run(1, 4);
    unsafe {
        mapal_par_task(handle, 0, 1, hit_slice, n as i64, 0, 0, 0, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
        mapal_par_finish(handle);
    }
    assert_eq!(frame.calls.load(Ordering::SeqCst), 4);
    assert!(!frame.bad_range.load(Ordering::SeqCst));
    assert!(!frame.duplicate.load(Ordering::SeqCst));
    assert!((0..n).all(|index| {
        frame.hits[index / 64].load(Ordering::SeqCst) & (1_u64 << (index % 64)) != 0
    }));

    let zero = Box::new(ZeroFrame {
        split_calls: AtomicU32::new(0),
        dependent_calls: AtomicU32::new(0),
    });
    let handle = test_run(2, 4);
    unsafe {
        mapal_par_task(handle, 0, 1, zero_split, 0, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, zero_dependent, 0, 0, 0, 0, 0);
        mapal_par_dep(handle, 0, 1);
        mapal_par_launch(handle, frame_ptr(&*zero));
        mapal_par_finish(handle);
    }
    assert_eq!(zero.split_calls.load(Ordering::SeqCst), 0);
    assert_eq!(zero.dependent_calls.load(Ordering::SeqCst), 1);
}

struct DepFrame {
    hits: Vec<AtomicU8>,
    dependent_calls: AtomicU32,
    violation: AtomicBool,
}

extern "C" fn dep_producer(lo: i64, hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<DepFrame>() };
    for hit in &frame.hits[lo as usize..hi as usize] {
        hit.store(1, Ordering::SeqCst);
    }
}

extern "C" fn dep_consumer(_lo: i64, _hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<DepFrame>() };
    if frame.hits.iter().any(|hit| hit.load(Ordering::SeqCst) != 1) {
        frame.violation.store(true, Ordering::SeqCst);
    }
    frame.dependent_calls.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn dependent_starts_after_all_slices_complete() {
    let n = (GRAIN * 3 + 1) as usize;
    let frame = Box::new(DepFrame {
        hits: (0..n).map(|_| AtomicU8::new(0)).collect(),
        dependent_calls: AtomicU32::new(0),
        violation: AtomicBool::new(false),
    });
    let handle = test_run(2, 4);
    unsafe {
        mapal_par_task(handle, 0, 1, dep_producer, n as i64, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, dep_consumer, 0, u32::MAX, 0, 0, 0);
        mapal_par_dep(handle, 0, 1);
        mapal_par_launch(handle, frame_ptr(&*frame));
        mapal_par_finish(handle);
    }
    assert_eq!(frame.dependent_calls.load(Ordering::SeqCst), 1);
    assert!(!frame.violation.load(Ordering::SeqCst));
}

struct OrderFrame {
    calls: Mutex<Vec<(i64, i64)>>,
}

extern "C" fn record_order(lo: i64, hi: i64, frame: *mut u8) {
    lock(&unsafe { &*frame.cast::<OrderFrame>() }.calls).push((lo, hi));
}

#[test]
fn inline_runs_ready_tasks_by_ascending_index() {
    let frame = Box::new(OrderFrame {
        calls: Mutex::new(Vec::new()),
    });
    let handle = test_run(3, 1);
    unsafe {
        mapal_par_task(handle, 2, 0, record_order, 12, 100, 0, 0, 0);
        mapal_par_task(handle, 0, 0, record_order, 10, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, record_order, 11, 50, 0, 0, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
    }
    assert!(lock(&frame.calls).is_empty(), "launch must only seed");
    unsafe { mapal_par_finish(handle) };
    assert_eq!(*lock(&frame.calls), [(0, 10), (0, 11), (0, 12)]);
}

struct TrapFrame {
    barrier: Barrier,
}

extern "C" fn trap_after_barrier(_lo: i64, hi: i64, frame: *mut u8) {
    unsafe { &*frame.cast::<TrapFrame>() }.barrier.wait();
    mapal_par_trap(hi, u32::from(hi == 7));
}

#[test]
fn trap_flag_keeps_minimum_topo() {
    let frame = Box::new(TrapFrame {
        barrier: Barrier::new(2),
    });
    let handle = test_run(2, 2);
    unsafe {
        mapal_par_task(handle, 0, 0, trap_after_barrier, 50, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, trap_after_barrier, 7, 0, 0, 0, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
        let entries = [completion_entry(0), completion_entry(1)];
        mapal_par_wait(handle, entries.as_ptr(), entries.len() as u32);
        assert_eq!(trap_flag(handle), (7_u64 << 32) | 2);
        clear_trap(handle);
        mapal_par_finish(handle);
    }
}

struct SpeculateFrame {
    trapping_completed: AtomicBool,
    dependent_calls: AtomicU32,
}

extern "C" fn trapping_task(_lo: i64, _hi: i64, frame: *mut u8) {
    mapal_par_trap(3, 1);
    unsafe { &*frame.cast::<SpeculateFrame>() }
        .trapping_completed
        .store(true, Ordering::SeqCst);
}

extern "C" fn dependent_after_trap(_lo: i64, _hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<SpeculateFrame>() };
    assert!(frame.trapping_completed.load(Ordering::SeqCst));
    frame.dependent_calls.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn trapped_task_completes_and_unlocks_its_dependent() {
    let frame = Box::new(SpeculateFrame {
        trapping_completed: AtomicBool::new(false),
        dependent_calls: AtomicU32::new(0),
    });
    let handle = test_run(2, 1);
    unsafe {
        mapal_par_task(handle, 0, 0, trapping_task, 0, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, dependent_after_trap, 0, 0, 0, 0, 0);
        mapal_par_dep(handle, 0, 1);
        mapal_par_launch(handle, frame_ptr(&*frame));
        let entries = [completion_entry(1)];
        mapal_par_wait(handle, entries.as_ptr(), 1);
        assert!(frame.trapping_completed.load(Ordering::SeqCst));
        assert_eq!(frame.dependent_calls.load(Ordering::SeqCst), 1);
        assert_eq!(trap_flag(handle), (3_u64 << 32) | 2);
        clear_trap(handle);
        mapal_par_finish(handle);
    }
}

struct WatermarkFrame {
    published: AtomicBool,
    gate: AtomicBool,
    completed: AtomicBool,
}

extern "C" fn publish_then_wait(_lo: i64, _hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<WatermarkFrame>() };
    mapal_par_watermark(10);
    frame.published.store(true, Ordering::SeqCst);
    while !frame.gate.load(Ordering::Acquire) {
        thread::yield_now();
    }
    frame.completed.store(true, Ordering::SeqCst);
}

#[test]
fn watermark_wait_can_finish_before_task_completion() {
    let frame = Box::new(WatermarkFrame {
        published: AtomicBool::new(false),
        gate: AtomicBool::new(false),
        completed: AtomicBool::new(false),
    });
    let handle = test_run(1, 2);
    unsafe {
        mapal_par_task(handle, 0, 0, publish_then_wait, 0, 0, 0, 0, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    while !frame.published.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(frame.published.load(Ordering::SeqCst));

    let below = [wait_entry(0, 10)];
    unsafe { mapal_par_wait(handle, below.as_ptr(), 1) };
    assert!(!frame.completed.load(Ordering::SeqCst));

    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let handle_addr = handle as usize;
    let waiter = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let above = [wait_entry(0, 11)];
        unsafe { mapal_par_wait(handle_addr as *mut FlowParRun, above.as_ptr(), 1) };
        done_tx.send(()).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(
        done_rx.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(!frame.completed.load(Ordering::SeqCst));

    frame.gate.store(true, Ordering::Release);
    done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    waiter.join().unwrap();
    assert!(frame.completed.load(Ordering::SeqCst));
    unsafe { mapal_par_finish(handle) };
}

struct PinnedFrame {
    host: thread::ThreadId,
    ranges: Mutex<Vec<(i64, i64, thread::ThreadId)>>,
    dependent_calls: AtomicU32,
}

extern "C" fn pinned_slice(lo: i64, hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<PinnedFrame>() };
    lock(&frame.ranges).push((lo, hi, thread::current().id()));
}

extern "C" fn pinned_dependent(_lo: i64, _hi: i64, frame: *mut u8) {
    unsafe { &*frame.cast::<PinnedFrame>() }
        .dependent_calls
        .fetch_add(1, Ordering::SeqCst);
}

fn def_of(n: i64, slice_elems: i64, oversub: u32) -> TaskDef {
    extern "C" fn noop(_: i64, _: i64, _: *mut u8) {}
    TaskDef {
        kind: 1,
        f: noop,
        n,
        rank: 0,
        slice_elems,
        oversub,
        width: 0,
    }
}

/// Every slice boundary must land on the region's quantum, and the ranges
/// must still tile `[0, n)` exactly.
///
/// The bug this pins: deriving a COUNT and then equal-dividing `n` destroys
/// whatever alignment the size asked for. 1048576 over 52 slices is 20164.9
/// elements, so every boundary lands mid-row and every piece starts with a
/// partial register block. Measured on matmul1024 at 14 lanes, counts that
/// happen to divide n evenly (32/64/128) ran 2.5-2.6 ms while neighbouring
/// counts that do not (43/52/57) ran 4.4-5.8 — a 2x swing with no relation
/// to how many pieces there are.
#[test]
fn slices_are_cut_on_the_quantum_and_tile_the_range() {
    let n = 1_048_576;
    let quantum = 4096; // TI=4 rows of c=1024
    for threads in [1usize, 4, 8, 14] {
        for oversub in [1u32, 4, 8] {
            let ranges = slice_ranges(def_of(n, quantum, oversub), threads);
            assert!(!ranges.is_empty(), "threads={threads} oversub={oversub}");
            assert_eq!(ranges[0].0, 0, "must start at 0");
            assert_eq!(ranges[ranges.len() - 1].1, n, "must cover n");
            for window in ranges.windows(2) {
                assert_eq!(window[0].1, window[1].0, "ranges must be contiguous");
            }
            for &(lo, hi) in &ranges {
                assert!(lo <= hi, "ranges must be ordered");
                assert_eq!(lo % quantum, 0, "every start is on the quantum: {lo}");
            }
            // Only the final boundary may be short of a whole quantum, and
            // here n is an exact multiple so none may be.
            for &(lo, hi) in &ranges {
                assert_eq!((hi - lo) % quantum, 0, "whole blocks only: {lo}..{hi}");
            }
        }
    }
}

/// Ragged slice counts are unreachable: whatever is asked for, the pieces
/// carry equal numbers of blocks. Measured cost of raggedness on matmul1024
/// at 14 lanes: 1.8x (a count of 57 over 256 blocks ran 4.48 ms against
/// 2.49-2.73 for every count that divides 256).
#[test]
fn slice_counts_are_never_ragged() {
    let (n, quantum) = (1_048_576, 4096);
    let blocks = n / quantum;
    for threads in [1usize, 4, 8, 14] {
        for oversub in [1u32, 4, 16, 64] {
            let ranges = slice_ranges(def_of(n, quantum, oversub), threads);
            let sizes: Vec<i64> = ranges.iter().map(|&(lo, hi)| (hi - lo) / quantum).collect();
            let (min, max) = (
                *sizes.iter().min().expect("slices"),
                *sizes.iter().max().expect("slices"),
            );
            assert!(
                max - min <= 1,
                "threads={threads} oversub={oversub}: block counts {sizes:?}"
            );
            assert!(ranges.len() <= blocks as usize, "never below one block");
        }
    }
}

/// The floor is a coherence constraint: a slice below one register block
/// drops the dispatch onto the TI=1 fallback (7x on matmul1024). No
/// requested over-decomposition may cut below it.
#[test]
fn over_decomposition_never_cuts_below_one_block() {
    let n = 16_384;
    let quantum = 4096; // only four blocks exist
    let ranges = slice_ranges(def_of(n, quantum, 64), 14);
    assert!(
        ranges.len() <= 4,
        "cannot make more pieces than blocks: {ranges:?}"
    );
    for &(lo, hi) in &ranges {
        assert!(
            hi - lo >= quantum || hi == n,
            "no sub-block slice: {lo}..{hi}"
        );
    }
}

/// A `Seq` task carries `n = 0` and must still run exactly once — the
/// quantised slicer must not optimise its empty range away. An empty
/// `Split`, by contrast, has genuinely nothing to run.
#[test]
fn sequential_tasks_still_get_exactly_one_range() {
    let mut seq = def_of(0, 0, 1);
    seq.kind = 0;
    assert_eq!(slice_ranges(seq, 14), vec![(0, 0)], "Seq runs once");
    assert!(
        slice_ranges(def_of(0, 0, 1), 14).is_empty(),
        "an empty Split has no work"
    );
}

#[test]
fn pinned_task_runs_inline_and_unlocks_dependents() {
    let n = GRAIN * 2 + 1;
    let frame = Box::new(PinnedFrame {
        host: thread::current().id(),
        ranges: Mutex::new(Vec::new()),
        dependent_calls: AtomicU32::new(0),
    });
    let handle = allocate_run(2, Pool::with_workers(4, 0));
    unsafe {
        mapal_par_task(handle, 0, 1, pinned_slice, n, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, pinned_dependent, 0, 0, 0, 0, 0);
        mapal_par_dep(handle, 0, 1);
        mapal_par_pin(handle, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
        mapal_par_run_pinned(handle, 0);
        let entries = [completion_entry(1)];
        mapal_par_wait(handle, entries.as_ptr(), 1);
        mapal_par_finish(handle);
    }

    let ranges = lock(&frame.ranges);
    assert_eq!(ranges.len(), 3);
    assert_eq!(ranges.first().map(|range| range.0), Some(0));
    assert_eq!(ranges.last().map(|range| range.1), Some(n));
    assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
    assert!(ranges.iter().all(|range| range.2 == frame.host));
    assert_eq!(frame.dependent_calls.load(Ordering::SeqCst), 1);
}

/// Regression (review find): a zero-dep pinned task must never be seeded to
/// the pool at launch — a worker grabbing it would run host-flavor code on
/// a worker thread. Pinning happens at registration, so the pool cannot see
/// it regardless of timing; only mapal_par_run_pinned executes it.
#[test]
fn pinned_zero_dep_task_is_never_pool_seeded() {
    let frame = Box::new(PinnedFrame {
        host: thread::current().id(),
        ranges: Mutex::new(Vec::new()),
        dependent_calls: AtomicU32::new(0),
    });
    let handle = allocate_run(2, Pool::with_workers(4, 4));
    unsafe {
        mapal_par_task(handle, 0, 0, pinned_slice, 1, 0, 0, 0, 0);
        mapal_par_task(handle, 1, 0, pinned_dependent, 0, 0, 0, 0, 0);
        mapal_par_pin(handle, 0);
        mapal_par_launch(handle, frame_ptr(&*frame));
        // Workers finish the normal sibling; the pinned task stays untouched.
        let entries = [completion_entry(1)];
        mapal_par_wait(handle, entries.as_ptr(), 1);
        assert!(
            lock(&frame.ranges).is_empty(),
            "pool executed a pinned task"
        );
        mapal_par_run_pinned(handle, 0);
        mapal_par_finish(handle);
    }
    let ranges = lock(&frame.ranges);
    assert_eq!(ranges.len(), 1);
    assert!(ranges.iter().all(|range| range.2 == frame.host));
}

struct BlockFrame {
    entered: Mutex<bool>,
    entered_cv: Condvar,
    released: Mutex<bool>,
    released_cv: Condvar,
}

extern "C" fn blocking_task(_lo: i64, _hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<BlockFrame>() };
    *lock(&frame.entered) = true;
    frame.entered_cv.notify_all();
    let mut released = lock(&frame.released);
    while !*released {
        released = frame.released_cv.wait(released).unwrap();
    }
}

struct HelpFrame {
    host: thread::ThreadId,
    calls: AtomicU32,
    ran_on_host: AtomicBool,
}

extern "C" fn helped_task(_lo: i64, _hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<HelpFrame>() };
    frame.calls.fetch_add(1, Ordering::SeqCst);
    frame
        .ran_on_host
        .store(thread::current().id() == frame.host, Ordering::SeqCst);
}

#[test]
fn wait_helps_while_the_background_worker_is_busy() {
    // Exercise one busy worker through the same queues; production T==1
    // remains thread-free.
    let pool = Pool::with_workers(1, 1);
    let blocker = Box::new(BlockFrame {
        entered: Mutex::new(false),
        entered_cv: Condvar::new(),
        released: Mutex::new(false),
        released_cv: Condvar::new(),
    });
    let blocker_run = allocate_run(1, Arc::clone(&pool));
    unsafe {
        mapal_par_task(blocker_run, 0, 0, blocking_task, 0, 0, 0, 0, 0);
        mapal_par_launch(blocker_run, frame_ptr(&*blocker));
    }

    let entered = lock(&blocker.entered);
    let (entered, _) = blocker
        .entered_cv
        .wait_timeout_while(entered, Duration::from_secs(2), |entered| !*entered)
        .unwrap();
    let worker_entered = *entered;
    drop(entered);

    let help = Box::new(HelpFrame {
        host: thread::current().id(),
        calls: AtomicU32::new(0),
        ran_on_host: AtomicBool::new(false),
    });
    let helped_run = allocate_run(1, pool);
    unsafe {
        mapal_par_task(helped_run, 0, 0, helped_task, 0, 0, 0, 0, 0);
        mapal_par_launch(helped_run, frame_ptr(&*help));
        let entries = [completion_entry(0)];
        mapal_par_wait(helped_run, entries.as_ptr(), 1);
    }

    *lock(&blocker.released) = true;
    blocker.released_cv.notify_all();
    unsafe {
        mapal_par_finish(helped_run);
        mapal_par_finish(blocker_run);
    }

    assert!(worker_entered, "background worker did not start blocker");
    assert_eq!(help.calls.load(Ordering::SeqCst), 1);
    assert!(help.ran_on_host.load(Ordering::SeqCst));
}

struct NestedFrame {
    inner_trap: AtomicU64,
}

extern "C" fn nested_inner(_lo: i64, _hi: i64, _frame: *mut u8) {
    mapal_par_trap(20, 1);
}

extern "C" fn nested_outer(_lo: i64, _hi: i64, frame: *mut u8) {
    let nested = unsafe { &*frame.cast::<NestedFrame>() };
    let inner = test_run(1, 1);
    unsafe {
        mapal_par_task(inner, 0, 0, nested_inner, 0, 0, 0, 0, 0);
        mapal_par_launch(inner, frame);
        let entries = [completion_entry(0)];
        mapal_par_wait(inner, entries.as_ptr(), 1);
        nested.inner_trap.store(trap_flag(inner), Ordering::SeqCst);
        clear_trap(inner);
        mapal_par_finish(inner);
    }
    mapal_par_trap(10, 0);
}

#[test]
fn nested_run_uses_innermost_tls_and_restores_outer() {
    let frame = Box::new(NestedFrame {
        inner_trap: AtomicU64::new(0),
    });
    let outer = test_run(1, 1);
    unsafe {
        mapal_par_task(outer, 0, 0, nested_outer, 0, 0, 0, 0, 0);
        mapal_par_launch(outer, frame_ptr(&*frame));
        let entries = [completion_entry(0)];
        mapal_par_wait(outer, entries.as_ptr(), 1);
        assert_eq!(frame.inner_trap.load(Ordering::SeqCst), (20_u64 << 32) | 2);
        assert_eq!(trap_flag(outer), (10_u64 << 32) | 1);
        clear_trap(outer);
        mapal_par_finish(outer);
    }
}

struct DagFrame {
    predecessors: Vec<Vec<usize>>,
    counts: Vec<AtomicU8>,
    violation: AtomicBool,
}

extern "C" fn dag_task(lo: i64, hi: i64, frame: *mut u8) {
    let frame = unsafe { &*frame.cast::<DagFrame>() };
    let task = hi as usize;
    if lo != 0
        || task >= frame.counts.len()
        || frame.predecessors[task]
            .iter()
            .any(|&before| frame.counts[before].load(Ordering::SeqCst) != 1)
    {
        frame.violation.store(true, Ordering::SeqCst);
        return;
    }
    if frame.counts[task].fetch_add(1, Ordering::SeqCst) != 0 {
        frame.violation.store(true, Ordering::SeqCst);
    }
    if task.is_multiple_of(11) {
        mapal_par_trap(task as i64 + 1, (task % 2) as u32);
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

#[test]
fn seeded_random_dags_complete_once() {
    let pool = Pool::new(4);
    let mut seed = 0x5eed_cafe_d15c_a11e;
    for run_idx in 0..64 {
        let n_tasks = 32 + (lcg(&mut seed) % 17) as usize;
        let mut predecessors = vec![Vec::new(); n_tasks];
        for (after, task_predecessors) in predecessors.iter_mut().enumerate().skip(1) {
            for before in 0..after {
                if lcg(&mut seed).is_multiple_of(7) {
                    task_predecessors.push(before);
                }
            }
        }
        let frame = Box::new(DagFrame {
            predecessors,
            counts: (0..n_tasks).map(|_| AtomicU8::new(0)).collect(),
            violation: AtomicBool::new(false),
        });
        let handle = allocate_run(n_tasks as u32, Arc::clone(&pool));
        unsafe {
            for task in 0..n_tasks {
                mapal_par_task(
                    handle,
                    task as u32,
                    0,
                    dag_task,
                    task as i64,
                    lcg(&mut seed) as u32,
                    0,
                    0,
                    0,
                );
            }
            for after in 0..n_tasks {
                for &before in &frame.predecessors[after] {
                    mapal_par_dep(handle, before as u32, after as u32);
                }
            }
            mapal_par_launch(handle, frame_ptr(&*frame));
            let entries: Vec<_> = (0..n_tasks)
                .map(|task| completion_entry(task as u32))
                .collect();
            mapal_par_wait(handle, entries.as_ptr(), entries.len() as u32);
            assert_ne!(trap_flag(handle), 0);
            clear_trap(handle);
            mapal_par_finish(handle);
        }
        assert!(
            !frame.violation.load(Ordering::SeqCst),
            "DAG invariant failed on run {run_idx}, seed {seed:#x}"
        );
        assert!(
            frame
                .counts
                .iter()
                .all(|count| count.load(Ordering::SeqCst) == 1),
            "task coverage failed on run {run_idx}, seed {seed:#x}"
        );
    }
}
