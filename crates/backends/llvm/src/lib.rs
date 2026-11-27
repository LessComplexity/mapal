//! LLVM backend: emits textual LLVM IR (`.ll`) piped to `clang` (ADR-0020 §1;
//! backend-llvm DESIGN). The entry point [`emit`] realizes the functor
//! `F_LLVM : Mapal-Cat → LLVM-Cat` as a String — the translation unit **is** the
//! artifact (ADR-0020). Semantics are the interpreter oracle's by construction:
//! every `Print`/trap routes through the shared `mapal-rt` runtime (ADR-0020 §2),
//! integer arithmetic wraps, `Div`/`Mod`/`Index`/`Update` are guarded (an S20
//! `bounds_proof`-proven `Index` elides its statically-dead guard), and floats
//! are IEEE at width.
//!
//! Emission is deterministic **per resolved profile** (L2): all names come from
//! per-function ordinals and a rising counter, never slotmap bits, so one `(IR,
//! EmitOpts, TargetProfile)` triple always yields the same text. The default
//! target is `native` (S46, ratified ADR-0038): it reads host facts, so default
//! emitted text is machine-dependent *by design* — reproducible emission names a
//! profile explicitly, and [`emit_detailed`] reports which profile was resolved.
//! The one capability gap (L3) is the non-canonical loop shape (multi-merge
//! SCC), rejected as [`EmitError::Unsupported`] — the same scope boundary as
//! interp M1 / rewrite.

mod emit;
mod gpu_arms;
mod gpu_rungs;
mod opts;
#[cfg(test)]
mod place_tests;

pub use self::emit::{emit, emit_detailed, emit_with_opts};
pub use self::gpu_arms::{GpuArms, SITE_CANDS};
pub use self::gpu_rungs::GpuRungs;
pub use self::opts::{Emission, EmitOpts, FoldOrder, Order, Reduced};

pub mod flags;
mod func;
mod loops;
mod math;
mod module;
mod narrow;
pub(crate) mod place;
#[cfg(test)]
mod place_replay;
mod profile;
mod reuse;
mod rules;
mod ty;

pub use crate::profile::TargetProfile;
pub use crate::profile::resolve as resolve_profile;
pub use crate::profile::{CostOver, DeviceCost, Ptx};

/// The profile a target name resolves to (`native` included), `None` for an
/// unknown name — what [`EmitOpts::target`] and [`EmitOpts::host`] accept.
pub fn target_profile(name: &str) -> Option<&'static TargetProfile> {
    profile::resolve(name)
}

/// hardware-properties: the rungs the rules ship on GPU profile `target`, one
/// line per fill class and face — `wps\tface\trungs on` (wps 0, 1 and 2 for
/// "2 or more"; faces `Exact`, `Contract`, `Tf32`) — for the evidence pass
/// (`benches/gpu/evidence.py`). `None` for an unknown or CPU profile.
pub fn rung_table(target: &str) -> Option<String> {
    let p = profile::resolve(target)?;
    p.gpu()?;
    let mut out = String::new();
    for wps in [0, 1, 2] {
        for face in [
            profile::Face::Exact,
            profile::Face::Contract,
            profile::Face::Tf32,
        ] {
            let r = p.gpu_rungs_on(wps, face);
            let on: Vec<&str> = GpuRungs::NAMES
                .iter()
                .copied()
                .filter(|n| r.get(n))
                .collect();
            out += &format!("{wps}\t{face:?}\t{}\n", on.join(","));
        }
    }
    Some(out)
}

use mapal_ir::{CategoryIr, FuncId, MorphismId, Operation, PathPlan, SourceLoc, TaskKind};
use slotmap::SecondaryMap;

use crate::func::{FnAttrs, FnEmit, packing_site};
use crate::module::{
    HEAP_DECLS, PAR_DECLS, PERF_DECLS, PREFETCH_DECL, PanelWrite, RT_DECLS, SME_DECLS,
    collect_str_globals, emit_main_wrapper, emit_str_globals, sme_panel,
};

/// A structured, renderer-free emission error (ADR-0020 §1; C3).
#[derive(Clone, Debug, PartialEq)]
pub enum EmitError {
    /// A capability-matrix rejection (the ✋ cell): a feature outside the
    /// realized set — here, the non-canonical (multi-merge) loop shape (L3).
    Unsupported { feature: String, loc: SourceLoc },
    /// An internal invariant violation (should not occur for sealed IR).
    Internal(String),
}

/// How the move-panel traversal is decided (S45). See [`EmitOpts::move_panel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MovePanel {
    /// **The default and the shipped path.** Fire where a recognized
    /// [`mapal_ir::MoveSite`] and the target profile's L1D geometry say it pays,
    /// with a derived block. A profile without L1D facts (`generic`) can never
    /// fire it, so the default target's emission is unchanged.
    #[default]
    Deduce,
    /// Never fire. The OFF arm of a measurement, and an escape hatch.
    Off,
    /// Force this `(w, b)` on every eligible map, ignoring both records — S44's
    /// behaviour exactly, kept **only** so a block sweep is still possible.
    Force(u64, u64),
}
