//! DESIGN §6 test plan item 2/4/5: golden `.ll` insta snapshots (byte-stable,
//! L2), determinism (emit twice → byte-equal), and the `Unsupported` pin on the
//! hand-built nested-loop graph (L3).
//!
//! Micro shapes cover the templates the 10 examples don't isolate cleanly:
//! signed `Div`/`Mod` guards (zero + `MIN/-1`), `Update` memcpy+store, the
//! last-use elision's in-place/keep-copy pins (suggestions #2), the S20
//! proven-`Index` guard elision + the FnAttrs proof refinement, and two
//! sequential loops in one fn (the S12 P0 shape). Select-Phi, index guards,
//! and loop CFG are additionally covered by the `abs` / `fir` / `sum_to_n`
//! example snapshots.

use mapal_backend_llvm::{EmitError, EmitOpts, GpuArms, emit_with_opts};

/// The goldens are the exact face's text (ADR-0040: the default face adds
/// `reassoc contract` to the tiled steps; `default_face_flags_the_tiled_steps`).
/// The goldens' emission, pinned to `generic` (ADR-0038 D4): the ambient
/// `native` carries this host's measured dispatch economics (local-rtx-5080),
/// so a golden on it would encode the machine that blessed it.
fn emit(ir: &mapal_ir::CategoryIr) -> Result<String, EmitError> {
    emit_with_opts(
        ir,
        &EmitOpts {
            target: "generic",
            host: "generic",
            ..EmitOpts::exact()
        },
    )
}
use mapal_ir::{CategoryIr, Dest, FuncKind, IrBuilder, Operation, SourceLoc, Ty, Value};

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

#[path = "golden_ll/basics.rs"]
mod basics;
#[path = "golden_ll/common.rs"]
mod common;
#[path = "golden_ll/conv_heap_parallel.rs"]
mod conv_heap_parallel;
#[path = "golden_ll/index_determinism.rs"]
mod index_determinism;
#[path = "golden_ll/pool_gpu.rs"]
mod pool_gpu;
#[path = "golden_ll/tile_shapes.rs"]
mod tile_shapes;
use common::*;
