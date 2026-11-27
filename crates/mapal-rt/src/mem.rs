//! Perf marks, the clock, and the arena / frame allocator.
use super::*;

pub(crate) static PERF_START: Mutex<Option<std::time::Instant>> = Mutex::new(None);

#[unsafe(no_mangle)]
pub extern "C" fn mapal_perf_begin() {
    if configured_threads() > 1 {
        drop(global_pool());
    }
    *lock(&PERF_START) = Some(std::time::Instant::now());
}

#[unsafe(no_mangle)]
pub extern "C" fn mapal_perf_end() {
    let elapsed = lock(&PERF_START)
        .take()
        .expect("mapal_perf_end called without mapal_perf_begin")
        .elapsed()
        .as_secs_f64()
        * 1000.0;
    emit(&format_args!("MAPAL_PERF total ms={elapsed:.4}"), true);
}

/// The process-lifetime monotonic epoch for `mapal_time_ms` (plan-time-builtin):
/// one `Instant` shared by every call in the process, so two calls are
/// non-decreasing and a difference is real elapsed milliseconds.
pub(crate) static TIME_EPOCH: OnceLock<std::time::Instant> = OnceLock::new();

/// The `time` builtin's runtime seam: milliseconds (f64) from the same
/// monotonic clock (`std::time::Instant`) as `mapal_perf_begin`/`mapal_perf_end`,
/// measured against the process-lifetime [`TIME_EPOCH`].
#[unsafe(no_mangle)]
pub extern "C" fn mapal_time_ms() -> f64 {
    checkpoint();
    TIME_EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.0
}

/// The heap-lowering arena (plan-s29 emission item 4): every block handed out
/// by [`mapal_rt_alloc`], kept as `(address, layout)` so [`mapal_rt_free_all`]
/// can release it. Addresses are stored as `usize` — a raw pointer is not
/// `Send`, and the arena is shared across the worker pool.
///
/// ponytail: ONE global mutex and a free-everything teardown. The emitter's
/// contract makes that enough — it only lowers the *entry* function's own
/// big blocks, so allocation happens a handful of times in the program's
/// prologue and never in a hot loop, and the single teardown sits after the
/// last reader (`mapal_par_finish` / fn end). Ceiling: a program that wanted
/// per-allocation lifetimes, or heap-lowered a repeatedly-called fn, needs
/// `mapal_rt_free(ptr)` + an emitter-side last-use point instead.
pub(crate) static ARENA: Mutex<Vec<(usize, Layout)>> = Mutex::new(Vec::new());

/// Allocate one arena block of `bytes` at `align`, uninitialised — `alloca`
/// storage is not zeroed either, and every emitted consumer writes before it
/// reads. Aborts on allocation failure (the emitted program has no path to
/// handle OOM; `mapal_trap`'s exit-101 contract is for *language* traps).
///
/// The block comes back **resident**: see [`reside`] for why that is part of
/// the contract and not an optimisation (plan-s33 CR-1).
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_alloc(bytes: i64, align: i64) -> *mut u8 {
    let layout = Layout::from_size_align(bytes as usize, align as usize)
        .expect("mapal_rt_alloc: emitter passes a valid size/alignment pair");
    // Zero-sized blocks would alias; the emitter never asks (the threshold is
    // 256 KB) but `alloc` requires size > 0, so keep the guard honest.
    assert!(layout.size() > 0, "mapal_rt_alloc: zero-sized block");
    let ptr = unsafe { alloc::alloc(layout) };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    unsafe { reside(ptr, layout.size()) };
    lock(&ARENA).push((ptr as usize, layout));
    ptr
}

/// Force physical residence of `[ptr, ptr + bytes)` — plan-s33's `reside`
/// morphism, and the reason `mapal_rt_alloc`'s result is a *materialised* block
/// rather than only a reserved address range.
///
/// A large `alloc` is served by `mmap`, which delivers **no** physical memory —
/// only a promise that touching the range will produce zeroed pages. The first
/// store to each page therefore traps, and the kernel must find and zero a page
/// (2 MiB at a time under transparent huge pages) before the store completes.
/// The emitter allocates in the entry-block prologue but the tasks store later,
/// so without this those faults land inside whatever region happens to write
/// first — including a `() -> time` bracket, which then charges page-zeroing to
/// the kernel it was meant to be timing. Measured at ~0.10 ms warm / 0.30 ms
/// cold per 4 MB, which is what made conv2d look 1.55x slower than naive C++
/// when its kernel is in fact faster (`docs/performance/cpu/kernels/conv2d-measurement-boundary.md`).
///
/// This costs the program nothing overall: the same pages are faulted and zeroed
/// either way. It only decides *when*, and so what a clock can see.
///
/// # Safety
/// `ptr` must be valid for writes over `bytes`, and uninitialised — every byte
/// written here is a `0` into a page the kernel has just zeroed anyway, so no
/// caller-visible value changes.
pub(crate) unsafe fn reside(ptr: *mut u8, bytes: usize) {
    // ponytail: one byte per 4 KiB, NOT a memset. The fault's own zeroing IS
    // the initialisation, so a memset would zero every page a second time —
    // 2x the memory traffic, ~15 ms wasted on a 64 MB matmul frame. Under 2 MiB
    // huge pages this touches ~512x more often than strictly needed, which is
    // ~4k stores per 16 MB: not worth probing the page size to avoid.
    // Ceiling: assumes pages are at least 4 KiB (true on every target we emit
    // for). On a multi-socket host this also pins every page to the touching
    // thread's NUMA node; if a parallel leg ever regresses there, the fix is to
    // make this lane-aware rather than to drop it (plan-s33 §5).
    let mut offset = 0usize;
    while offset < bytes {
        // `write_volatile` so LLVM cannot discard the loop as dead stores.
        unsafe { ptr.add(offset).write_volatile(0) };
        offset += 4096;
    }
}

/// Release every arena block. Emitted once per entry function, after the last
/// point that can read arena memory (post-`mapal_par_finish` in the parallel
/// flavor, immediately before `ret` otherwise).
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_free_all() {
    for (addr, layout) in lock(&ARENA).drain(..) {
        unsafe { alloc::dealloc(addr as *mut u8, layout) };
    }
}

thread_local! {
    /// structure: the per-call arena of a repeatedly-called fn — its big
    /// blocks, released when it returns ([`mapal_rt_frame_release`]). Thread-
    /// local, so a fn running on a worker never releases another's blocks.
    static FRAME: std::cell::RefCell<Vec<(usize, Layout)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// structure: the mark a non-entry fn takes on entry; its blocks are those
/// allocated after it.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_frame_mark() -> i64 {
    FRAME.with(|f| f.borrow().len() as i64)
}

/// structure: [`mapal_rt_alloc`] for a non-entry fn's big block (a decode
/// step's per-layer caches would overflow the stack as `alloca`s): resident,
/// uninitialised, released by [`mapal_rt_frame_release`] at its return.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_frame_alloc(bytes: i64, align: i64) -> *mut u8 {
    let layout = Layout::from_size_align(bytes as usize, align as usize)
        .expect("mapal_rt_frame_alloc: emitter passes a valid size/alignment pair");
    assert!(layout.size() > 0, "mapal_rt_frame_alloc: zero-sized block");
    let ptr = unsafe { alloc::alloc(layout) };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    unsafe { reside(ptr, layout.size()) };
    FRAME.with(|f| f.borrow_mut().push((ptr as usize, layout)));
    ptr
}

/// structure: release every block allocated on this thread since `mark`
/// (the fn's output was written into its caller's storage before).
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_frame_release(mark: i64) {
    FRAME.with(|f| {
        for (addr, layout) in f.borrow_mut().drain(mark.max(0) as usize..) {
            unsafe { alloc::dealloc(addr as *mut u8, layout) };
        }
    });
}
