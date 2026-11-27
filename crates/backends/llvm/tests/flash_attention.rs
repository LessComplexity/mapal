//! flash-attention `stream`: attention (and every chain of the form tile →
//! cells and row folds over its lanes → tile folding over the same axis) as
//! one pass over windows of the axis — `stream_plan` (the roles, each fold's
//! laws, a rescale's shift) and its realization. Values through the GPU
//! emulator against the interpreter (free threads, serialized in thread
//! order and reversed, the block order reversed) within the default face's
//! bound, on hand-written and generated programs (`testgen::FlashProg`); the
//! OFF arms' emission; every coverage counter non-zero.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, DepLaw, StreamRole};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

#[path = "flash_attention/common.rs"]
mod common;
#[path = "flash_attention/generated.rs"]
mod generated;
#[path = "flash_attention/mma.rs"]
mod mma;
#[path = "flash_attention/stream_facts.rs"]
mod stream_facts;
use common::*;
use generated::*;
