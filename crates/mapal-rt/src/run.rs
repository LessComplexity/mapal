//! `impl Run`: one parallel wave's task graph, scheduling and traps.
use super::*;

impl Run {
    pub(crate) fn new(n_tasks: u32, pool: Arc<Pool>) -> Arc<Self> {
        Arc::new(Self {
            pool,
            frame: AtomicPtr::new(ptr::null_mut()),
            trap: AtomicU64::new(0),
            build: Mutex::new(
                (0..n_tasks)
                    .map(|_| TaskBuild {
                        def: None,
                        dependents: Vec::new(),
                        deps: 0,
                        pinned: false,
                    })
                    .collect(),
            ),
            tasks: OnceLock::new(),
            remaining: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
            gate: Mutex::new(()),
            changed: Condvar::new(),
        })
    }

    /// The sealed graph. Every read after launch goes through here and takes no
    /// lock.
    pub(crate) fn tasks(&self) -> &[TaskRt] {
        self.tasks.get().expect("run is not launched")
    }

    pub(crate) fn launched(&self) -> bool {
        self.tasks.get().is_some()
    }

    /// Freeze registration into the executing form. Called once, from launch.
    pub(crate) fn seal(&self) {
        let build = std::mem::take(&mut *lock(&self.build));
        assert!(
            build.iter().all(|task| task.def.is_some()),
            "parallel run has unregistered tasks"
        );
        self.remaining.store(build.len(), Ordering::Release);
        let tasks: Box<[TaskRt]> = build
            .into_iter()
            .map(|task| TaskRt {
                def: task.def.expect("registered"),
                dependents: task.dependents.into_boxed_slice(),
                pinned: task.pinned,
                deps_left: AtomicU32::new(task.deps),
                state: AtomicU8::new(TaskState::Waiting as u8),
                slices_left: AtomicUsize::new(0),
                watermark: AtomicI64::new(-1),
            })
            .collect();
        self.tasks
            .set(tasks)
            .unwrap_or_else(|_| panic!("parallel run launched twice"));
    }

    /// Publish progress and release anyone waiting on it. The epoch is bumped
    /// BEFORE the gate is taken, so a waiter that already read the old value
    /// cannot miss it: it either sees the new epoch and skips its wait, or it
    /// holds the gate and this call blocks until it is parked.
    pub(crate) fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        let _gate = lock(&self.gate);
        self.changed.notify_all();
    }

    /// Retire `task_idx` and return the dependents it unlocked. The caller must
    /// have won the right to do this — either by taking the task's last slice or
    /// by being the thread that found it empty.
    pub(crate) fn retire(&self, task_idx: usize) -> Vec<usize> {
        let tasks = self.tasks();
        // ORDER IS LOAD-BEARING: unlock the dependents FIRST, publish `Done`
        // LAST (plan-s50 P1).
        //
        // `Done` is what a waiter observes, so storing it before the dependents'
        // counters had been decremented let a host that saw it run ahead of this
        // thread — `mapal_par_run_pinned` then hit its own
        // `deps_left == 0` assertion with the count still at 1. It is a race, so
        // it was survivable while the host always PARKED and was only released
        // by the `bump` after this function returned; giving the host a spin
        // (P1) made it observe `Done` immediately and it fired on `conv2d_1024`
        // within seconds. Release/Acquire on `state` now orders the whole
        // unlock behind it.
        let mut ready = Vec::new();
        for &dependent in tasks[task_idx].dependents.iter() {
            let task = &tasks[dependent];
            // Exactly one thread sees the count fall to zero, so a task is
            // scheduled exactly once with no lock and no re-check.
            //
            // A pinned dependent is never queued: the host spine runs it via
            // `mapal_par_run_pinned` at its topo position, and the `bump` that
            // follows releases the host's wait on this completion. The
            // zero-length-dispatch path used to omit this filter and could hand
            // a pinned task to a worker, where `execute` asserts — one rule now,
            // both paths (plan-s50 P2).
            if task.deps_left.fetch_sub(1, Ordering::AcqRel) == 1 && !task.pinned {
                ready.push(dependent);
            }
        }
        tasks[task_idx]
            .state
            .store(TaskState::Done as u8, Ordering::Release);
        self.remaining.fetch_sub(1, Ordering::AcqRel);
        ready
    }

    /// `inline_ok` is false at launch. `mapal_par_launch` MUST only seed — the
    /// emitted host runs its own spine work between launch and finish, and the
    /// clock fence (plan-s33b) is expressed as dependency edges evaluated from
    /// that spine. A launch that executed tasks itself would run bracketed work
    /// before the host reached the pinned clock read. Mid-run unlocks have no
    /// such constraint: the calling thread is already inside the run.
    pub(crate) fn schedule(
        self: &Arc<Self>,
        mut ready: Vec<usize>,
        placement: Placement,
        inline_ok: bool,
    ) {
        let mut queued = Vec::new();
        let mut inline: Vec<(usize, i64, i64)> = Vec::new();
        let mut cursor = 0;
        let mut progressed = false;

        while !ready.is_empty() {
            // `def` is immutable after `seal`, so the rank sort reads it with no
            // lock (plan-s50 P2).
            let tasks = self.tasks();
            match placement {
                Placement::Seed if self.pool.threads > 1 => ready.sort_by(|&a_idx, &b_idx| {
                    tasks[b_idx]
                        .def
                        .rank
                        .cmp(&tasks[a_idx].def.rank)
                        .then_with(|| a_idx.cmp(&b_idx))
                }),
                _ => ready.sort_unstable(),
            }

            let batch = std::mem::take(&mut ready);
            for task_idx in batch {
                let (ranges, unlocked, width) = {
                    let def = tasks[task_idx].def;
                    let ranges = slice_ranges(def, self.pool.threads);

                    if ranges.is_empty() {
                        // An empty dispatch retires immediately; whoever unlocked
                        // it is the single owner of that transition.
                        let unlocked = self.retire(task_idx);
                        self.epoch.fetch_add(1, Ordering::AcqRel);
                        (Vec::new(), unlocked, 0)
                    } else {
                        tasks[task_idx]
                            .slices_left
                            .store(ranges.len(), Ordering::Release);
                        tasks[task_idx]
                            .state
                            .store(TaskState::Running as u8, Ordering::Release);
                        (ranges, Vec::new(), def.width)
                    }
                };

                ready.extend(unlocked);
                // plan-s50 P3 — DO NOT DISPATCH WHAT CANNOT BE SHARED.
                //
                // A task cut into exactly one slice has no parallelism to give
                // away: queueing it buys a wave (11-41 us parked) and one lane's
                // worth of work, which is the work the calling thread was going
                // to be able to do anyway. Every scalar-glue task in a plan is
                // this shape — `nn2_128`'s plan has fifteen tasks and eight of
                // them are capture tuples and print cones — and each one was
                // costing a full dispatch.
                //
                // This needs no threshold and no work model: "one slice" is
                // decided by `slice_ranges`, which the compiler's quantum and the
                // lane count already determine between them.
                if inline_ok && ranges.len() == 1 {
                    let (lo, hi) = ranges[0];
                    inline.push((task_idx, lo, hi));
                    continue;
                }
                for (lo, hi) in ranges {
                    let lane = match placement {
                        Placement::Seed => {
                            // plan-s32 step 1: lay the dispatch across the lanes
                            // the compiler asked for. Stealing still lets any
                            // idle lane help — placement is a starting point,
                            // not a fence, because which lane is free is the one
                            // thing only the runtime can see.
                            let span = if width == 0 {
                                self.pool.threads
                            } else {
                                (width as usize).clamp(1, self.pool.threads)
                            };
                            let lane = cursor % span;
                            cursor += 1;
                            lane
                        }
                    };
                    queued.push((
                        lane,
                        Work {
                            run: Arc::clone(self),
                            task: task_idx,
                            lo,
                            hi,
                        },
                    ));
                }
            }

            // Hand the pool its work BEFORE running anything here, so the
            // workers are already moving while this thread does the inline part.
            if !queued.is_empty() {
                self.pool.enqueue(std::mem::take(&mut queued));
                self.bump();
                progressed = true;
            }
            for (task_idx, lo, hi) in std::mem::take(&mut inline) {
                self.run_inline(task_idx, lo, hi);
                ready.extend(self.complete_slice_ready(task_idx));
                progressed = true;
            }
        }

        if !progressed {
            self.changed.notify_all();
        }
    }

    /// Take one slice off `task_idx` and, if it was the last, retire it and
    /// return what that unlocked. Split out of `complete_slice` so the inline
    /// path can fold the result back into its own worklist instead of recursing
    /// into `schedule` (plan-s50 P3).
    pub(crate) fn complete_slice_ready(self: &Arc<Self>, task_idx: usize) -> Vec<usize> {
        let task = &self.tasks()[task_idx];
        if task.state.load(Ordering::Acquire) == TaskState::Done as u8 {
            return Vec::new();
        }
        // Only the thread that takes the LAST slice retires the task. Every
        // other slice is one atomic and no lock (plan-s50 P2).
        if task.slices_left.fetch_sub(1, Ordering::AcqRel) != 1 {
            return Vec::new();
        }
        let ready = self.retire(task_idx);
        self.bump();
        ready
    }

    /// Run a whole task on the calling thread. The TLS run stack is what
    /// `mapal_par_trap` and `mapal_par_watermark` read, so it is pushed here
    /// exactly as `execute` does for a pooled slice.
    pub(crate) fn run_inline(self: &Arc<Self>, task_idx: usize, lo: i64, hi: i64) {
        let def = self.tasks()[task_idx].def;
        CURRENT_RUNS.with(|runs| {
            runs.borrow_mut().push(CurrentRun {
                run: Arc::as_ptr(self),
                task: task_idx,
            });
        });
        (def.f)(lo, hi, self.frame.load(Ordering::Acquire));
        CURRENT_RUNS.with(|runs| {
            runs.borrow_mut().pop().unwrap();
        });
    }

    pub(crate) fn complete_slice(self: &Arc<Self>, task_idx: usize) {
        let ready = self.complete_slice_ready(task_idx);
        if !ready.is_empty() {
            // plan-s50 P0: a dep-unlocked task is SEEDED, not placed on the lane
            // that happened to finish its producer.
            //
            // `Local` was chosen when the only dep-unlocked tasks were scalar
            // glue, where running next to the value just written is affinity.
            // Once a packed tile site became two outer tasks (pack -> matmul),
            // the thing being unlocked is a many-slice bulk dispatch, and
            // `Local` gave it one lane's deque plus an index-order sort instead
            // of the rank-sorted spread across `width` that the same task gets
            // at launch. S43 §7 predicted exactly this and it is why it kept the
            // nested runs.
            //
            // Measured, M4 Pro, 41 cycles, parallel leg, median ms:
            //   nn2_128    Local 0.1290  Seed 0.0839   (1.54x)
            //   nn4_128    Local 0.2447  Seed 0.1550   (1.58x)
            //   nn2x2_128  Local 0.1813  Seed 0.1215   (1.49x)
            //   nn2x2_512  Local 0.6097  Seed 0.5821   (1.05x, and it removes a
            //                                           regression against S49)
            self.schedule(ready, Placement::Seed, true);
        }
    }

    pub(crate) fn help_until(self: &Arc<Self>, done: impl Fn(&Self) -> bool) {
        let lane = WORKER_LANE
            .with(|lane| lane.get())
            .unwrap_or_else(|| self.pool.threads - 1);
        'outer: loop {
            if done(self) {
                return;
            }
            let seen = self.epoch.load(Ordering::Acquire);

            if let Some(work) = self.pool.take_for(self, lane) {
                execute(work);
                continue;
            }

            // plan-s50 P1: the host is half the wave. It publishes the dispatch
            // and it is what learns the wave finished, so parking it costs the
            // same 11-41 us as parking a worker. Spin on the completion
            // predicate first; the budget is zero when the pool is
            // oversubscribed (`spin_iters`).
            for _ in 0..self.pool.spin {
                if done(self) {
                    return;
                }
                if self.epoch.load(Ordering::Acquire) != seen {
                    continue 'outer;
                }
                if self.pool.pending.load(Ordering::Acquire) > 0
                    && let Some(work) = self.pool.take_for(self, lane)
                {
                    execute(work);
                    continue 'outer;
                }
                std::hint::spin_loop();
            }

            let mut gate = lock(&self.gate);
            if done(self) {
                return;
            }
            // Re-tested under the gate, so a `bump` that lands between the read
            // above and this wait cannot be lost — it must take the gate to
            // notify, and by then the epoch has already moved.
            while self.epoch.load(Ordering::Acquire) == seen {
                gate = self.changed.wait(gate).unwrap();
            }
        }
    }
}
