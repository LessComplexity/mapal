//! `mapal-rt` — the shared runtime seam (backend-llvm DESIGN §1, ADR-0020 §2).
//!
//! The emitted native program calls these `extern "C"` symbols for every
//! `Print` and every trap, so float formatting and trap/exit behaviour are the
//! interpreter oracle's *by construction*: each print renders via Rust
//! shortest-round-trip `Display`, exactly as `mapal_interp::render` does (both
//! call the same formatter). `mapal_trap` writes the defined message to stderr
//! and exits 101 (ADR-0013 / ADR-0020 §3).
//!
//! Stdout is flushed on every call — the differential harness reads pipes.

use std::{
    alloc::{self, Layout},
    cell::RefCell,
    collections::VecDeque,
    io::{self, Write},
    ptr,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicI64, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

/// Render `v` via `Display` (== interp `render`) and write it to stdout,
/// appending a newline when `newline`; flush before returning.
/// bf16-class `device-trap`: the host checkpoint's hook — the GPU runtime's
/// read of the device's trap word (`mapal_gpu_trap_arm`), registered once a
/// trap-checking kernel has launched; `0`: none.
static CHECKPOINT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Register the checkpoint hook (`mapal_gpu.c` / `mapal_gpu_emu.c`).
#[unsafe(no_mangle)]
pub extern "C" fn mapal_rt_checkpoint_hook(f: extern "C" fn()) {
    CHECKPOINT.store(f as usize, std::sync::atomic::Ordering::Release);
}

/// A host checkpoint (a print, the clock, a trap): a device trap recorded
/// before it is raised here — the oracle's outcome, the program's effects
/// up to the trapping operation and no later. The GPU runtime's copy-outs
/// (`mapal_gpu_d2h`, `_d2h_on`) poll the same word: the host never computes
/// on a trapped read's stand-in (device cells are loop-free, `cell_plan`, so
/// only host control can fail to reach a checkpoint).
fn checkpoint() {
    let p = CHECKPOINT.load(std::sync::atomic::Ordering::Acquire);
    if p != 0 {
        // SAFETY: only `mapal_rt_checkpoint_hook` stores, an `extern "C" fn()`.
        let f: extern "C" fn() = unsafe { std::mem::transmute(p) };
        f();
    }
}

fn emit(v: &dyn std::fmt::Display, newline: bool) {
    checkpoint();
    let mut out = io::stdout();
    // Ignore write errors: a broken pipe is the harness's business, not ours.
    if newline {
        let _ = writeln!(out, "{v}");
    } else {
        let _ = write!(out, "{v}");
    }
    let _ = out.flush();
}

/// Define one `mapal_print_<ty>(v, newline)` extern. `Display` for every scalar
/// type is the shortest round-trip, matching interp `render` verbatim.
macro_rules! print_fn {
    ($name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $name(v: $ty, newline: bool) {
            emit(&v, newline);
        }
    };
}

print_fn!(mapal_print_i32, i32);
print_fn!(mapal_print_i64, i64);
print_fn!(mapal_print_u8, u8);
print_fn!(mapal_print_bool, bool);
print_fn!(mapal_print_f32, f32);
print_fn!(mapal_print_f64, f64);

/// Print a UTF-8 string given as `(ptr, len)` (interp `render(Str) == s`).
///
/// # Safety
/// `ptr` must point to `len` initialised bytes of valid UTF-8 (the emitter only
/// ever passes a private `Str` global constant — never data, since lower
/// rejects strings-as-data, DESIGN §2).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_print_str(ptr: *const u8, len: usize, newline: bool) {
    let s = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) };
    emit(&s, newline);
}

/// A defined trap: write the message to stderr and exit 101 (ADR-0013).
/// `0 = div_zero`, `1 = index_oob`; the emitter only ever passes these two.
#[unsafe(no_mangle)]
pub extern "C" fn mapal_trap(kind: u32) -> ! {
    // A device trap recorded earlier is the program's first (its launch
    // precedes this host operation).
    checkpoint();
    let msg = match kind {
        0 => "div_zero",
        1 => "index_oob",
        _ => "unknown", // ponytail: emitter passes only 0/1; total match anyway
    };
    // Flush any buffered stdout first (belt-and-braces; prints already flush).
    let _ = io::stdout().flush();
    eprintln!("mapal trap: {msg}");
    std::process::exit(101);
}

pub mod calibrate;
/// ADR-0045 D5: the environment variable that overrides the recorded data
/// directory (`mapal_ir::read::DATA_DIR_ENV`; the tests check they agree).
mod mem;
mod par;
mod pool;
mod read;
mod run;
mod sched;
#[cfg(test)]
mod tests;

pub use self::mem::*;
pub use self::par::*;
pub use self::read::*;
pub use self::sched::*;
