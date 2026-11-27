//! per-site-choice B4: every site's candidate table on programs from the
//! generator (`testgen` geom / order / mid / tile / chain strategies, which
//! the candidates were not designed on), on the four GPU parts:
//!
//! - every candidate's values answer the interpreter — within ADR-0040's
//!   bound on the default face, bit for bit on the exact face — through the
//!   GPU emulator, one run per candidate index (`MAPAL_GPU_PICK`);
//! - running every candidate in turn (`MAPAL_GPU_PICK=all`) leaves the last
//!   one's bits: a site flagged `RERUN` really is;
//! - candidate 0 is the rules' plan: `--cands=1` emits no table, and each
//!   table's first recipe and kernel are the `--cands=1` build's launch and
//!   kernel text;
//! - the decision (`mapal_choice.h`, shared with the device runtime) under a
//!   fake clock: the faster candidate wins and is written, the next run reads
//!   it, a candidate within 3 % keeps candidate 0, a corrupt wisdom file is
//!   ignored, a changed table re-tunes;
//! - the candidate set: cap, distinct launches, one `Combine` on the fixed
//!   and exact faces, no table for the excluded site classes.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, Order, emit_detailed};
use mapal_ir::{CategoryIr, bound_coef};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const PARTS: [&str; 4] = ["rtx-4090", "rtx-5090", "rtx-3090", "h100-pcie"];

#[path = "site_choice/choice.rs"]
mod choice;
#[path = "site_choice/common.rs"]
mod common;
#[path = "site_choice/pick.rs"]
mod pick;
use common::*;
