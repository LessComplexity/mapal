//! What one parallel wave costs on THIS host, measured through the same entry
//! points an emitted program calls (local-rtx-5080).
//!
//! A compiler that targets `native` needs the dispatch economics of the machine
//! it sits on, and they differ by 30x between hosts: one full-width wave of
//! empty slices is ~8 us on a 14-core M4 and 263 us on a 64-thread Threadripper
//! 9970X (`examples/dispatch_cost.rs`, `run 1 task / 256 slices`). A guessed
//! constant is what [`crate`]'s callers refuse, so this measures it.

use super::*;

/// One full-width wave on the pool an emitted program gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wave {
    /// Executors in the pool (`MAPAL_PAR`, else the available lanes).
    pub lanes: u64,
    /// Median nanoseconds a wave adds over running the same task inline:
    /// a split task cut one slice per lane, empty bodies, the workers PARKED
    /// beforehand — the state a program's next dispatch finds them in.
    pub wave_ns: u64,
}

extern "C" fn nothing(lo: i64, hi: i64, _frame: *mut u8) {
    std::hint::black_box((lo, hi));
}

/// `begin -> one split task -> launch -> finish`, the shape every emitted
/// dispatch has; `kind = 0` is the inline arm of the difference method.
fn one(kind: u32, lanes: u64) -> u64 {
    let start = std::time::Instant::now();
    unsafe {
        let h = mapal_par_begin(1);
        // `slice_elems = 0` takes the program path for an untiled map:
        // `min(lanes, n / GRAIN)` slices, here exactly one per lane.
        mapal_par_task(h, 0, kind, nothing, lanes as i64 * GRAIN, 1, 0, 0, 0);
        mapal_par_launch(h, std::ptr::null_mut());
        mapal_par_finish(h);
    }
    start.elapsed().as_nanos() as u64
}

/// Measure [`Wave`]: `reps` pairs, each preceded by a sleep longer than any
/// spin (`spin_iters`) so every worker has parked. `None` on a one-lane pool
/// (nothing is ever dispatched there).
pub fn wave(reps: usize) -> Option<Wave> {
    let lanes = configured_threads() as u64;
    if lanes < 2 {
        return None;
    }
    drop(global_pool());
    let mut diffs = Vec::with_capacity(reps.max(1));
    for _ in 0..reps.max(1) {
        std::thread::sleep(std::time::Duration::from_micros(500));
        let par = one(1, lanes);
        let inline = one(0, lanes);
        diffs.push(par.saturating_sub(inline));
    }
    diffs.sort_unstable();
    Some(Wave {
        lanes,
        wave_ns: diffs[diffs.len() / 2].max(1),
    })
}

#[cfg(test)]
mod tests {
    /// The probe runs the emitted entry points end to end and reports a
    /// positive wave on any multi-lane pool (the value is the host's).
    #[test]
    fn wave_measures_the_configured_pool() {
        match super::wave(8) {
            Some(w) => {
                assert_eq!(w.lanes, super::configured_threads() as u64);
                assert!(w.lanes >= 2 && w.wave_ns >= 1, "{w:?}");
            }
            None => assert!(super::configured_threads() < 2),
        }
    }
}
