//! The `mapal_par_*` entry points an emitted program calls.
use super::*;

pub(crate) fn execute(work: Work) {
    // plan-s50 P2: no lock. `pinned` and `def` are immutable after `seal`, and
    // `schedule` already published `Running` before the slice was queued.
    let task = {
        let entry = &work.run.tasks()[work.task];
        assert!(!entry.pinned, "pinned task reached a pool worker");
        if entry.state.load(Ordering::Acquire) == TaskState::Done as u8 {
            return;
        }
        entry.def
    };

    CURRENT_RUNS.with(|runs| {
        runs.borrow_mut().push(CurrentRun {
            run: Arc::as_ptr(&work.run),
            task: work.task,
        });
    });
    (task.f)(work.lo, work.hi, work.run.frame.load(Ordering::Acquire));
    CURRENT_RUNS.with(|runs| {
        runs.borrow_mut().pop().unwrap();
    });

    work.run.complete_slice(work.task);
}

pub(crate) unsafe fn run_handle<'a>(handle: *mut FlowParRun) -> &'a FlowParRun {
    assert!(!handle.is_null(), "null flow parallel run");
    unsafe { &*handle }
}

pub(crate) fn allocate_run(n_tasks: u32, pool: Arc<Pool>) -> *mut FlowParRun {
    Box::into_raw(Box::new(FlowParRun {
        run: Run::new(n_tasks, pool),
    }))
}

/// Allocate a task-DAG run containing exactly `n_tasks` indexed task slots.
///
/// The returned handle must be configured, launched, and passed exactly once to
/// `mapal_par_finish`.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_par_begin(n_tasks: u32) -> *mut FlowParRun {
    allocate_run(n_tasks, global_pool())
}

/// Register task `idx` before launch.
///
/// `kind == 0` executes exactly one `f(0, n, frame)` call. `kind == 1`
/// executes disjoint slices covering `[0, n)` exactly once, using at most
/// `min(T, ceil(n / 4096))` calls.
///
/// # Safety
/// `handle` must be a live handle from `mapal_par_begin`, and every task index
/// must be registered exactly once before launch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_task(
    handle: *mut FlowParRun,
    idx: u32,
    kind: u32,
    f: TaskFn,
    n: i64,
    rank: u32,
    slice_elems: i64,
    oversub: u32,
    width: u32,
) {
    let run = &unsafe { run_handle(handle) }.run;
    assert!(!run.launched(), "cannot register a task after launch");
    let mut build = lock(&run.build);
    let task = &mut build[idx as usize];
    assert!(task.def.is_none(), "task registered twice");
    assert!(kind <= 1, "unknown parallel task kind");
    assert!(
        kind == 0 || n >= 0,
        "split task length must be non-negative"
    );
    task.def = Some(TaskDef {
        kind,
        f,
        n,
        rank,
        slice_elems,
        oversub,
        width,
    });
}

/// Add a dependency edge: `after` cannot execute before `before` completes.
///
/// # Safety
/// `handle` must be live, both indices must be in range, and registration must
/// still be open before `mapal_par_launch`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_dep(handle: *mut FlowParRun, before: u32, after: u32) {
    let run = &unsafe { run_handle(handle) }.run;
    assert!(!run.launched(), "cannot register a dependency after launch");
    let mut build = lock(&run.build);
    assert_ne!(before, after, "task cannot depend on itself");
    build[before as usize].dependents.push(after as usize);
    build[after as usize].deps = build[after as usize]
        .deps
        .checked_add(1)
        .expect("task dependency count overflow");
}

/// Seal the registered DAG, attach its frame, and seed all initially-ready
/// tasks. This returns after seeding; workers and later help-first waits execute
/// the work.
///
/// # Safety
/// `handle` must be live and unlaunched, every task slot must be registered,
/// and `frame` must remain valid through `mapal_par_finish`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_launch(handle: *mut FlowParRun, frame: *mut u8) {
    let run = Arc::clone(&unsafe { run_handle(handle) }.run);
    run.frame.store(frame, Ordering::Release);
    assert!(!run.launched(), "parallel run launched twice");
    run.seal();
    let ready = run
        .tasks()
        .iter()
        .enumerate()
        .filter_map(|(idx, task)| {
            (task.deps_left.load(Ordering::Acquire) == 0 && !task.pinned).then_some(idx)
        })
        .collect();
    run.schedule(ready, Placement::Seed, false);
}

/// Mark task `idx` as host-pinned before launch: the scheduler never queues it;
/// the host spine executes it via `mapal_par_run_pinned` at its topo position.
///
/// # Safety
/// `handle` must be live, `idx` in range, and registration still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_pin(handle: *mut FlowParRun, idx: u32) {
    let run = &unsafe { run_handle(handle) }.run;
    assert!(!run.launched(), "cannot pin a task after launch");
    lock(&run.build)[idx as usize].pinned = true;
}

/// Help this run until every packed wait entry is satisfied.
///
/// The calling thread executes or steals ready work from this run instead of
/// blocking behind workers.
///
/// # Safety
/// `handle` must be live and launched. When `len != 0`, `entries` must point to
/// `len` packed `(task_idx << 32 | threshold)` entries with valid task indices.
/// `u32::MAX` requires completion; other thresholds also accept a published
/// watermark at or above the threshold.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_wait(handle: *mut FlowParRun, entries: *const u64, len: u32) {
    let run = Arc::clone(&unsafe { run_handle(handle) }.run);
    let entries = if len == 0 {
        &[]
    } else {
        assert!(!entries.is_null(), "null parallel wait entry list");
        unsafe { std::slice::from_raw_parts(entries, len as usize) }
    };
    assert!(run.launched(), "cannot wait before launch");
    assert!(
        entries
            .iter()
            .all(|&entry| (entry >> 32) < run.tasks().len() as u64),
        "parallel wait task index out of range"
    );
    run.help_until(|run| {
        entries.iter().all(|&entry| {
            let task = &run.tasks()[(entry >> 32) as usize];
            let threshold = entry as u32;
            task.state.load(Ordering::Acquire) == TaskState::Done as u8
                || (threshold != u32::MAX
                    && task.watermark.load(Ordering::Acquire) >= i64::from(threshold))
        })
    });
}

pub(crate) fn check_trap(run: &Run, topo: i64) {
    let trap = run.trap.load(Ordering::Acquire);
    if trap != 0 && ((trap >> 32) as i64) < topo {
        if (trap as u32) - 1 == READ_TRAP_KIND {
            read_failure_at((trap >> 32) as i64);
        }
        mapal_trap((trap as u32) - 1);
    }
}

/// Observe the run's deterministic trap flag at an oracle checkpoint.
///
/// A recorded trap with `topo_idx < topo` is delivered through `mapal_trap`;
/// `i64::MAX` therefore checks every representable task trap.
///
/// # Safety
/// `handle` must be live and launched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_check(handle: *mut FlowParRun, topo: i64) {
    check_trap(&unsafe { run_handle(handle) }.run, topo);
}

/// Record a trap from inside the currently-executing emitted task.
///
/// The innermost nested run receives the flag. The runtime records only the
/// minimum topo index and never exits from a worker.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_par_trap(topo: i64, kind: u32) {
    assert!(
        (0..=u32::MAX as i64).contains(&topo),
        "parallel trap topo index out of range"
    );
    assert_ne!(kind, u32::MAX, "parallel trap kind cannot use the sentinel");
    CURRENT_RUNS.with(|runs| {
        let runs = runs.borrow();
        let current = runs
            .last()
            .expect("mapal_par_trap called outside a parallel task");
        let run = unsafe { &*current.run };
        let new = ((topo as u64) << 32) | u64::from(kind + 1);
        let mut old = run.trap.load(Ordering::Acquire);
        loop {
            if old != 0 && old >> 32 <= topo as u64 {
                return;
            }
            match run
                .trap
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(actual) => old = actual,
            }
        }
    });
}

/// Publish that the current scalar task has decided every trap site through
/// `topo`.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_par_watermark(topo: i64) {
    assert!(
        (0..u32::MAX as i64).contains(&topo),
        "parallel watermark topo index out of range"
    );
    let current = CURRENT_RUNS.with(|runs| {
        *runs
            .borrow()
            .last()
            .expect("mapal_par_watermark called outside a parallel task")
    });
    let run = unsafe { &*current.run };
    let task = &run.tasks()[current.task];
    assert_eq!(task.def.kind, 0, "only scalar tasks publish watermarks");
    assert_eq!(
        TaskState::of(task.state.load(Ordering::Acquire)),
        TaskState::Running,
        "watermark task is not running"
    );
    task.watermark.store(topo, Ordering::Release);
    run.bump();
}

/// Execute one ready task inline on the calling host-spine thread.
///
/// # Safety
/// `handle` must be live and launched, and `idx` must name an unstarted task
/// whose dependencies have completed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_run_pinned(handle: *mut FlowParRun, idx: u32) {
    let run = Arc::clone(&unsafe { run_handle(handle) }.run);
    let task_idx = idx as usize;
    let (def, ranges) = {
        assert!(run.launched(), "cannot run a pinned task before launch");
        let task = &run.tasks()[task_idx];
        assert!(task.pinned, "mapal_par_run_pinned on an unpinned task");
        assert_eq!(
            task.deps_left.load(Ordering::Acquire),
            0,
            "pinned task dependencies are incomplete"
        );
        assert_eq!(
            TaskState::of(task.state.load(Ordering::Acquire)),
            TaskState::Waiting,
            "pinned task has already started"
        );
        task.state
            .store(TaskState::Running as u8, Ordering::Release);
        task.slices_left.store(1, Ordering::Release);
        (task.def, slice_ranges(task.def, run.pool.threads))
    };

    CURRENT_RUNS.with(|runs| {
        runs.borrow_mut().push(CurrentRun {
            run: Arc::as_ptr(&run),
            task: task_idx,
        });
    });
    let frame = run.frame.load(Ordering::Acquire);
    for (lo, hi) in ranges {
        (def.f)(lo, hi, frame);
    }
    CURRENT_RUNS.with(|runs| {
        runs.borrow_mut().pop().unwrap();
    });

    run.complete_slice(task_idx);
}

/// Help until every task is complete, deliver any recorded trap, and free the
/// run.
///
/// # Safety
/// `handle` must be a live launched handle and must not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_par_finish(handle: *mut FlowParRun) {
    assert!(!handle.is_null(), "null flow parallel run");
    let handle = unsafe { Box::from_raw(handle) };
    let run = Arc::clone(&handle.run);
    assert!(run.launched(), "cannot finish before launch");
    run.help_until(|run| run.remaining.load(Ordering::Acquire) == 0);
    check_trap(&run, i64::MAX);
    drop(handle);
}
