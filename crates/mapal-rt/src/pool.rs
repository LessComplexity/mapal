//! `impl Pool`: the worker threads and the queue.
use super::*;

impl Pool {
    pub(crate) fn new(threads: usize) -> Arc<Self> {
        // plan-s50 P1: ONE EXECUTOR PER LANE, counting the host.
        //
        // This used to spawn `threads` workers while the host stayed an
        // additional help-first executor, so `MAPAL_PAR=14` on a 14-core part ran
        // 15 runnable threads. `help_until` already gives the host lane
        // `threads - 1`, so that lane always had two claimants. Oversubscription
        // is what makes spinning catastrophic (1.083 us -> 1640.9 us per wave at
        // the boundary), and it was measurably costing time before spinning
        // existed: `nn4_128` ran 0.2483 ms at `MAPAL_PAR=8` against 0.3675 at 14.
        //
        // `MAPAL_PAR` now means total executors, which is what a user asking for
        // "14 threads" on a 14-core machine means.
        Self::with_workers(threads, threads.saturating_sub(1))
    }

    pub(crate) fn with_workers(threads: usize, workers: usize) -> Arc<Self> {
        assert!(threads >= 1);
        assert!(workers <= threads);
        let pool = Arc::new(Self {
            threads,
            queues: (0..threads).map(|_| Mutex::new(VecDeque::new())).collect(),
            parks: (0..threads)
                .map(|_| LanePark {
                    epoch: Mutex::new(0),
                    wake: Condvar::new(),
                })
                .collect(),
            pending: AtomicUsize::new(0),
            spin: spin_iters(threads),
        });

        // MAPAL_PAR=1 is the thread-free sequential lever. With a real pool the
        // host remains an additional help-first executor at joins.
        for lane in 0..workers {
            let worker_pool = Arc::clone(&pool);
            thread::Builder::new()
                .name(format!("flow-par-{lane}"))
                .spawn(move || {
                    WORKER_LANE.set(Some(lane));
                    worker_pool.worker_loop(lane);
                })
                .expect("failed to create flow parallel worker");
        }
        pool
    }

    pub(crate) fn enqueue(&self, work: Vec<(usize, Work)>) {
        if work.is_empty() {
            return;
        }
        // plan-s50 P0b: remember which lanes actually received work, and wake
        // only those. The set is exact — `Run::schedule` just chose it.
        let mut filled = vec![false; self.threads];
        let count = work.len();
        for (lane, item) in work {
            lock(&self.queues[lane]).push_back(item);
            filled[lane] = true;
        }
        self.pending.fetch_add(count, Ordering::AcqRel);
        for (lane, got) in filled.into_iter().enumerate() {
            if got {
                self.notify_lane(lane);
            }
        }
    }

    pub(crate) fn notify_lane(&self, lane: usize) {
        let park = &self.parks[lane];
        let mut epoch = lock(&park.epoch);
        *epoch = epoch.wrapping_add(1);
        drop(epoch);
        park.wake.notify_all();
    }

    /// Record that one queued item left the pool. `pending` is only a hint for
    /// the spin loops, so it is decremented on the way out of every taker.
    pub(crate) fn claimed(&self, work: Option<Work>) -> Option<Work> {
        if work.is_some() {
            self.pending.fetch_sub(1, Ordering::AcqRel);
        }
        work
    }

    pub(crate) fn take_any(&self, lane: usize) -> Option<Work> {
        if let Some(work) = lock(&self.queues[lane]).pop_front() {
            return self.claimed(Some(work));
        }
        for offset in 1..self.threads {
            let victim = (lane + offset) % self.threads;
            if let Some(work) = lock(&self.queues[victim]).pop_back() {
                return self.claimed(Some(work));
            }
        }
        None
    }

    pub(crate) fn take_for(&self, run: &Arc<Run>, lane: usize) -> Option<Work> {
        if self.threads == 1 {
            let mut queue = lock(&self.queues[0]);
            let position = queue
                .iter()
                .enumerate()
                .filter(|(_, work)| Arc::ptr_eq(&work.run, run))
                .min_by_key(|(_, work)| (work.task, work.lo))
                .map(|(position, _)| position)?;
            let work = queue.remove(position);
            drop(queue);
            return self.claimed(work);
        }

        let mut own = lock(&self.queues[lane]);
        if let Some(position) = own.iter().position(|work| Arc::ptr_eq(&work.run, run)) {
            let work = own.remove(position);
            drop(own);
            return self.claimed(work);
        }
        drop(own);

        for offset in 1..self.threads {
            let victim = (lane + offset) % self.threads;
            let mut queue = lock(&self.queues[victim]);
            if let Some(position) = queue.iter().rposition(|work| Arc::ptr_eq(&work.run, run)) {
                let work = queue.remove(position);
                drop(queue);
                return self.claimed(work);
            }
        }
        None
    }

    pub(crate) fn worker_loop(&self, lane: usize) {
        loop {
            if let Some(work) = self.take_any(lane) {
                execute(work);
                continue;
            }

            let park = &self.parks[lane];
            let mut epoch = lock(&park.epoch);
            if let Some(work) = self.take_any(lane) {
                drop(epoch);
                execute(work);
                continue;
            }
            let seen = *epoch;
            while *epoch == seen {
                epoch = park.wake.wait(epoch).unwrap();
            }
        }
    }
}
