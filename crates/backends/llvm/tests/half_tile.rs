//! tensor-cores (ADR-0043 D4 / D6): a sum of products whose multiplicands are
//! 16-bit — operands stored `bf16` / `f16`, or `f32` under `: bf16 | f32` —
//! takes the part's `mma.sync.m16n8k16` with f32 accumulators (`half-tile`):
//! 16-byte `cp.async` panels in each operand's storage format, `ldmatrix`
//! fragments of the narrow ones, an f32 operand rounded once to the member as
//! its fragment is read. On programs from the generator (`testgen::HalfProg`:
//! GEMMs and attention heads it was not designed on): the facts, the
//! realization per arm, and the emulated device's values within ADR-0041 D3's
//! bound (`u_t` the member's, `u_a = 2⁻²³`) in every emulator mode and under
//! the OFF arm.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run, run_weakest};
use mapal_ir::{CategoryIr, NarrowFloat, bound_coef_unit};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

#[path = "half_tile/common.rs"]
mod common;
#[path = "half_tile/facts.rs"]
mod facts;
#[path = "half_tile/generated.rs"]
mod generated;
#[path = "half_tile/post.rs"]
mod post;
use common::*;
use generated::*;
