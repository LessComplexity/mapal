//! WP2 golden tests: insta snapshots of the emitted `.cu` for the scalar-only
//! examples (`abs` — Phi strict select; `calc` — calls, str globals, chained
//! guards, Div/Mod guards) plus a micro shape pinning the signed Div/Mod
//! guard text, structural assertions on the flagship `abs`, and emit-twice
//! byte equality (L2). WP3 adds the array flagships: `vector_add` (literal
//! uploads, Zip/Map launches, Index readbacks, the Fold kernel) and
//! `zip_demo` (Enumerate) as full-module snapshots, and `sepia` structural
//! pins (the BC8 `__host__ __device__` case, the struct-element literal,
//! the Pixel fold).
//!
//! WP4+WP5 complete the suite: snapshots of every remaining example — the
//! loop flagships `sum_to_n` (scalar carried state) and `fir` (speculative
//! `Index` in the advance cone) plus `sepia`, `fanout`, `pipeline`,
//! `seq_demo` — and the §6.8 pins: two sequential loops in one fn (the S12
//! shape), `exit_only_payload_emitted_once` (llvm `golden_ll.rs:198` ported)
//! with its exit-arm `Print` sibling (countdown), a loop-driven array
//! `Update` program (ADR-0021's motivating `c[t] <- v` pattern, smallest
//! honest version), and emit-twice byte equality over the loop examples.

use mapal_backend_cuda::{EmitOpts, emit, emit_with_opts};
use mapal_ir::{CategoryIr, Dest, FuncKind, IrBuilder, SourceLoc, Ty};

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

#[path = "golden_cu/arena.rs"]
mod arena;
#[path = "golden_cu/basics.rs"]
mod basics;
#[path = "golden_cu/common.rs"]
mod common;
#[path = "golden_cu/structural.rs"]
mod structural;
use basics::*;
use common::*;
use structural::*;
