//! DESIGN §6 — the remote compile-and-run differential harness (WP6), ported
//! from `mapal-backend-llvm/tests/differential.rs` with `nvcc` swapped for
//! `clang` per the §4/§6.6 recipe.
//!
//! For each program (the 10 examples, the pinned special cases, and a
//! closed-mode testgen sweep), on **raw and `rewrite()`d** IR: `emit → nvcc
//! -std=c++17 -fmad=false -arch=sm_89 prog.cu libmapal_rt.a -o prog -lpthread
//! -ldl -lm` (120 s compile timeout) `→ run (15 s timeout) → compare per
//! L1/§3`: `Done` ⇒ exit 0 + stdout **byte-equal** to the oracle; `Trapped`
//! ⇒ exit 101 (stdout ignored; classes never cross); **exit 102 ⇒ infra
//! failure** — the test fails naming the program, but the row is NEVER an R1
//! data point (DESIGN §3); any other exit ⇒ failure with captured stderr;
//! timeout ⇒ failure naming the program. Oracle expectations are computed
//! before `rewrite()` (IR taken by value); `Unit → i32` closed entries get
//! the result-printing `main` wrapper (BL8, ported); Diverged programs skip;
//! open-mode testgen (`i32 → i32`) is excluded — no native observable (BL8).
//!
//! `nvcc` is discovered `$NVCC → $CUDA_HOME/bin/nvcc → which nvcc` (§6.5);
//! absent ⇒ **skip-with-reason** (HANDOFF §5.5 — never a faked pass, never a
//! failure). This macOS host has no nvcc: local runs exercise the skip path
//! plus the `local` unit tests of the pure harness pieces (discovery order,
//! the recipe argv, the exit classifier). The pinned cases additionally run
//! their oracle assertions locally before the skip gate, so a malformed pin
//! fails here, not on the box.
//!
//! Deviations from the llvm harness (the RULES are verbatim; toolchain-side
//! only):
//! - ONE compile per program — the llvm `-O0`/`-O2` double row has no M3
//!   analog (an nvcc `-O3` row is DESIGN §7 headroom).
//! - nvcc compile gets a 120 s timeout (§6.7); llvm's clang call is unbounded.
//! - stdout AND stderr are captured via temp files (llvm pipes stdout, nulls
//!   stderr): stderr feeds §6.7's "any other exit ⇒ failure with captured
//!   stderr", and files can't pipe-buffer-deadlock a chatty program.

#![allow(clippy::type_complexity)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

use mapal_interp::{Outcome, RValue, RunResult, render, run};
use mapal_ir::{CategoryIr, SourceLoc, Ty};
use mapal_rewrite::rewrite;

// The testgen program generator (shared with mapal-rewrite's differential duty).
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use testgen::{Built, build, prog_strategy};

const BUDGET: u64 = 10_000_000;
/// The inherited llvm run timeout (§6.7).
const TIMEOUT_SECS: u64 = 15;
/// The §6.7 nvcc compile timeout.
const COMPILE_TIMEOUT_SECS: u64 = 120;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

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
#[path = "differential/harness.rs"]
mod harness;
#[path = "differential/sweep.rs"]
mod sweep;
use harness::*;
