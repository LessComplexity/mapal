//! The floor under a compile-time schedule (plan-s50 §P1).
//!
//! A static schedule removes queues, stealing, run construction and dep bookkeeping —
//! the compiler has already decided which lane runs which slice. What it CANNOT remove
//! is the moment N threads have to be told "go" and the host has to learn they finished.
//! That round trip is the floor, and this measures it three ways:
//!
//!   condvar   host bumps an epoch under a Mutex and notify_all; workers park on it.
//!             The mechanism mapal-rt uses today, with the queues taken away — so the
//!             difference from `dispatch_cost` is what the queues cost, and this row is
//!             what a static schedule built on parked threads would still pay.
//!   spin      host bumps an AtomicU64; workers spin on it and decrement an AtomicUsize.
//!             No kernel involvement at all. This is the real floor.
//!   hybrid    spin for a bounded number of iterations, then park. What ships in
//!             practice, because a spinning pool burns a core per idle lane.
//!
//! `waves` reproduces a whole forward pass: `nn4` opens 8 dispatches back to back, so an
//! 8-wave round trip is directly comparable to its measured 0.364 ms at d=128.
//!
//! Usage: handoff_floor [threads] [reps]
use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

const SPIN_LIMIT: u32 = 20_000;

struct Gate {
    epoch: AtomicU64,
    left: AtomicUsize,
    stop: AtomicBool,
    /// Only the parking arms touch these.
    park: Mutex<u64>,
    wake: Condvar,
    done: Mutex<()>,
    finished: Condvar,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Condvar,
    Spin,
    Hybrid,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            epoch: AtomicU64::new(0),
            left: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            park: Mutex::new(0),
            wake: Condvar::new(),
            done: Mutex::new(()),
            finished: Condvar::new(),
        })
    }

    /// Worker: wait for wave `seen + 1`, run its (empty) statically assigned slice,
    /// report completion. Returns false once the host asks it to stop.
    fn worker_wait(&self, seen: u64, mode: Mode) -> bool {
        if mode != Mode::Condvar {
            let mut spins = 0u32;
            loop {
                if self.stop.load(Ordering::Acquire) {
                    return false;
                }
                if self.epoch.load(Ordering::Acquire) != seen {
                    return true;
                }
                spin_loop();
                spins += 1;
                if mode == Mode::Spin || spins < SPIN_LIMIT {
                    continue;
                }
                break;
            }
        }
        let mut park = self.park.lock().unwrap();
        while *park == seen && !self.stop.load(Ordering::Acquire) {
            park = self.wake.wait(park).unwrap();
        }
        !self.stop.load(Ordering::Acquire)
    }

    fn worker_done(&self, mode: Mode) {
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 && mode != Mode::Spin {
            let _guard = self.done.lock().unwrap();
            self.finished.notify_all();
        }
    }

    /// Host: release one wave across `threads` lanes and wait for all of them.
    fn wave(&self, threads: usize, mode: Mode) {
        self.left.store(threads, Ordering::Release);
        if mode == Mode::Spin {
            self.epoch.fetch_add(1, Ordering::AcqRel);
        } else {
            let mut park = self.park.lock().unwrap();
            *park += 1;
            self.epoch.store(*park, Ordering::Release);
            drop(park);
            self.wake.notify_all();
        }

        if mode != Mode::Condvar {
            let mut spins = 0u32;
            while self.left.load(Ordering::Acquire) != 0 {
                spin_loop();
                spins += 1;
                if mode == Mode::Spin || spins < SPIN_LIMIT {
                    continue;
                }
                break;
            }
            if self.left.load(Ordering::Acquire) == 0 {
                return;
            }
        }
        let mut guard = self.done.lock().unwrap();
        while self.left.load(Ordering::Acquire) != 0 {
            guard = self.finished.wait(guard).unwrap();
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run(mode: Mode, threads: usize, reps: usize, waves: usize) -> f64 {
    let gate = Gate::new();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let gate = Arc::clone(&gate);
            thread::spawn(move || {
                let mut seen = 0u64;
                while gate.worker_wait(seen, mode) {
                    seen += 1;
                    gate.worker_done(mode);
                }
            })
        })
        .collect();

    for _ in 0..8 {
        gate.wave(threads, mode);
    }
    let samples = (0..reps)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..waves {
                gate.wave(threads, mode);
            }
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();

    gate.stop.store(true, Ordering::Release);
    {
        let mut park = gate.park.lock().unwrap();
        *park += 1;
        gate.epoch.store(*park, Ordering::Release);
    }
    gate.wake.notify_all();
    for h in handles {
        let _ = h.join();
    }
    median(samples)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or_else(|| {
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    });
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(201);

    println!("threads={threads} reps={reps}  (us, median)");
    println!(
        "{:<10}{:>12}{:>16}{:>18}",
        "mode", "1 wave", "8 waves", "8 waves/wave"
    );
    println!("{}", "-".repeat(56));
    for (name, mode) in [
        ("spin", Mode::Spin),
        ("hybrid", Mode::Hybrid),
        ("condvar", Mode::Condvar),
    ] {
        let one = run(mode, threads, reps, 1);
        let eight = run(mode, threads, reps, 8);
        println!("{name:<10}{one:>12.3}{eight:>16.3}{:>18.3}", eight / 8.0);
    }
}
