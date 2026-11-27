//! DESIGN §4/§6 test plan item 1/3: the compile-and-run differential harness.
//!
//! For each program (the 10 examples, two sequential loops, trap cases, and a
//! closed-mode testgen sweep), on **raw and `rewrite()`d** IR: `emit → clang
//! <prog>.ll libmapal_rt.a -o prog → run (time-boxed) → compare per L1` — `Done`
//! ⇒ exit 0 + stdout byte-equal to the oracle; `Trapped` ⇒ exit 101 (stdout
//! ignored). Every case is compiled at **both `-O0` and `-O2`** against the same
//! oracle expectations (the DESIGN §8 `-O2` row — optimization is where
//! accidentally-relied-on LLVM-level UB would surface). `clang` absent ⇒ the
//! whole harness skips-with-reason (never a faked pass). Open-mode testgen
//! (`i32 → i32`) is excluded — no native `@main` analog (BL8).
//!
//! Profile policy (ADR-0038 D4): every emission here pins a profile by name —
//! `generic`, whose NEON-derived geometry is what the tile-shape assertions
//! bind — because S46 made `EmitOpts::exact().target` the host-detected
//! `native`, and a detected AVX2/AVX-512/L2 yields different tile constants.
//! The ONE leg that intentionally floats on the ambient default is
//! `differential_native_profile_is_value_correct` (values only, no shape).

#![allow(clippy::type_complexity)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

use mapal_interp::{Outcome, RunResult, native_observable as expect_native, run};
use mapal_ir::{CategoryIr, Operation, Ty};
use mapal_rewrite::{PassId, rewrite};

// The testgen program generator (shared with mapal-rewrite's differential duty).
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use testgen::{Built, build, prog_strategy};

const BUDGET: u64 = 10_000_000;
const TIMEOUT_SECS: u64 = 15;

/// How many sweep cases share one linked executable (phase 2).
///
/// A case costs almost nothing to *compile* and almost everything to turn into
/// a **fresh binary and run it once**. Measured here on macOS, 20 modules:
/// linking all 20 in parallel is 0.26 s (2.8 CPU-s — clang scales ~11x), and
/// re-running the resulting binaries is 0.02 s, but the *first* execution of the
/// 20 brand-new binaries is **7.41 s of wall for 0.27 s of CPU**. The work is
/// code-signature validation in a system daemon outside the process tree, which
/// is single-threaded — which is why fanning out made the suite slower, not
/// faster. 1,280 first-execs at ~0.32 s is the whole 409 s the sweep used to
/// take. Linux has no such daemon and the same total is dominated by `ld`
/// instead (CI's differential step: 1,391 s of a 1,475 s Ubuntu run).
///
/// Merging cases into one translation unit divides **both** counts by this
/// factor. Every case is still emitted, compiled and executed, at both opt
/// levels, against the same oracle expectations — the cross product is
/// unchanged; only the number of *binaries* shrinks.
const BATCH: usize = 32;

/// Pool width of a sweep child: parallel, but small enough that
/// `job_slots() × SWEEP_LANES` stays inside the core budget. Single tests that
/// pin the default pool (`run_exe(.., None)`) keep it.
const SWEEP_LANES: &str = "2";

const EXAMPLES: &[&str] = &[
    "abs",
    "calc",
    "fanout",
    "fir",
    "pipeline",
    "sepia",
    "seq_demo",
    "sum_to_n",
    "vector_add",
    "zip_demo",
];

#[path = "differential/basics.rs"]
mod basics;
#[path = "differential/batching.rs"]
mod batching;
#[path = "differential/conv2d.rs"]
mod conv2d;
#[path = "differential/device.rs"]
mod device;
#[path = "differential/fir.rs"]
mod fir;
#[path = "differential/harness.rs"]
mod harness;
#[path = "differential/loops_calls.rs"]
mod loops_calls;
#[path = "differential/matmul_basic.rs"]
mod matmul_basic;
#[path = "differential/matmul_kc.rs"]
mod matmul_kc;
#[path = "differential/matmul_shapes.rs"]
mod matmul_shapes;
#[path = "differential/movement.rs"]
mod movement;
use batching::*;
use conv2d::*;
use harness::*;
use movement::*;

// --- S65 rule R on the device: generated device-sized loops, emulated --------

#[path = "emu/mod.rs"]
mod emu;
use emu::{job_slots, slot};
