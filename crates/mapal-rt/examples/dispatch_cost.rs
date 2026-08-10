//! What one dispatch costs, with no kernel under it (plan-s49b).
//!
//! Every parallel number this project publishes is `kernel + dispatch`, and at
//! small sizes the second term wins: a 4-layer net at d=128 runs 0.0822 ms on one
//! thread and 0.3636 ms on fourteen. Nothing in the tree prices the second term on
//! its own, so this does: the task functions here do nothing at all, and what the
//! clock sees is the runtime.
//!
//! Usage: dispatch_cost [reps]        (MAPAL_PAR selects the pool)
//!
//! Rows:
//!   pool-spawn    creating the pool: N threads spawned, once per process
//!   run K/S       one begin -> K tasks x S slices -> launch -> finish, empty bodies
//!   nested        a run whose single task opens and finishes an inner run —
//!                 the emitted matmul's shape (tile pack + tile matmul, two per site)
use std::hint::black_box;
use std::time::Instant;

use mapal_rt::{
    mapal_par_begin, mapal_par_finish, mapal_par_launch, mapal_par_pin, mapal_par_run_pinned,
    mapal_par_task,
};

extern "C" fn nothing(lo: i64, hi: i64, _frame: *mut u8) {
    black_box((lo, hi));
}

/// A task body that opens its own inner run, exactly like the emitted tile site.
extern "C" fn inner_run(lo: i64, hi: i64, _frame: *mut u8) {
    black_box((lo, hi));
    unsafe {
        let h = mapal_par_begin(1);
        mapal_par_task(h, 0, 1, nothing, 4096, 1, 64, 1, 0);
        mapal_par_launch(h, std::ptr::null_mut());
        mapal_par_finish(h);
    }
}

/// One run: `tasks` independent split tasks, each cut into ~`slices` pieces.
fn one_run(tasks: u32, slices: i64, body: extern "C" fn(i64, i64, *mut u8)) {
    unsafe {
        let h = mapal_par_begin(tasks);
        for idx in 0..tasks {
            // slice_elems = 1 with oversub high enough that `slices` pieces are cut.
            mapal_par_task(h, idx, 1, body, slices, 1, 1, u32::MAX, 0);
        }
        mapal_par_launch(h, std::ptr::null_mut());
        mapal_par_finish(h);
    }
}

/// The same empty task, PINNED: the host spine runs it inline, nothing is
/// enqueued, no worker is woken, no queue is scanned. The discriminator between
/// "the dispatch path costs" and "waking and stealing costs".
fn one_pinned() {
    unsafe {
        let h = mapal_par_begin(1);
        mapal_par_task(h, 0, 0, nothing, 1, 1, 1, 1, 0);
        mapal_par_pin(h, 0);
        mapal_par_launch(h, std::ptr::null_mut());
        mapal_par_run_pinned(h, 0);
        mapal_par_finish(h);
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(label: &str, reps: usize, per_call: f64, mut f: impl FnMut()) {
    // one warm call outside the sample so the first-wake cost is not counted here
    f();
    let us = median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect(),
    );
    println!("{label:<28}{us:>12.3}{:>14.3}", us / per_call);
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(201);
    let threads: usize = std::env::var("MAPAL_PAR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });

    // Pool spawn, measured once — it is a OnceLock, so the first run pays it.
    let t = Instant::now();
    one_run(1, 1, nothing);
    let spawn_us = t.elapsed().as_secs_f64() * 1e6;

    println!("MAPAL_PAR={threads}  reps={reps}");
    println!("pool spawn + first dispatch: {spawn_us:.1} us  (once per process)\n");
    println!("{:<28}{:>12}{:>14}", "case", "median us", "us per unit");
    println!("{}", "-".repeat(54));

    bench("PINNED 1 task (no wake)", reps, 1.0, one_pinned);
    bench("run 1 task / 1 slice", reps, 1.0, || one_run(1, 1, nothing));
    for s in [2i64, 4, 8, 16, 32, 64, 128, 256] {
        bench(&format!("run 1 task / {s} slices"), reps, s as f64, || {
            one_run(1, s, nothing)
        });
    }
    for k in [2u32, 4, 8, 16, 32] {
        bench(&format!("run {k} tasks / 1 slice"), reps, k as f64, || {
            one_run(k, 1, nothing)
        });
    }
    bench("nested: 1 outer + 1 inner", reps, 1.0, || {
        one_run(1, 1, inner_run)
    });
    bench("nested: 4 outer x inner", reps, 4.0, || {
        one_run(1, 4, inner_run)
    });
}
