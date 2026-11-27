//! `TargetProfile` — the emitter's machine facts as data (plan-s31-target-profiles).
//!
//! ADR-0032 D4: tile factors, grain sizes and arena thresholds are backend
//! config, not language. Every field here is **value-invariant** — changing one
//! changes how fast the answer arrives, never the answer — which is exactly what
//! keeps the differential suite a valid gate under every profile.
//!
//! The split this file exists to enforce (the backend-genericity contract):
//! **geometry comes from the record, constants come from the profile.** mapal-ir
//! never learns a machine fact; the emitter never re-derives graph analysis.
//! What used to be six literals swept on one M4 Pro is now one table plus
//! arithmetic, and the arithmetic reproduces those literals for the default
//! profile — that is the correctness gate for this change.

use mapal_ir::math::{Err as MathErr, Target as MathTarget, Unit as MathUnit, UnitOp};
use mapal_ir::{MoveSite, Ty};
use mapal_place::Src;

mod calib;
mod caps;
mod choice;
mod detect;
mod device_cost;
mod facts;
mod geom;
mod math_units;
mod matrix;
mod native;
mod place_a100;
mod place_ada;
mod place_hopper;
mod place_rtx_3090;
mod place_rtx_5080;
mod place_rtx_5090;
mod place_src;
mod plan;
mod profiles_gpu;
mod profiles_host;
mod registry;
mod target;
mod tiles;
mod tp_fma;
mod tp_mma;
mod tp_sme;
mod tp_split;
mod tp_wgmma;

pub use self::calib::*;
pub use self::caps::*;
pub use self::choice::*;
pub use self::detect::*;
pub use self::device_cost::*;
pub use self::facts::*;
pub use self::geom::*;
pub use self::math_units::*;
pub use self::matrix::*;
pub use self::native::*;
pub use self::place_a100::*;
pub use self::place_ada::*;
pub use self::place_hopper::*;
pub use self::place_rtx_3090::*;
pub use self::place_rtx_5080::*;
pub use self::place_rtx_5090::*;
use self::place_src::*;
pub use self::plan::*;
pub use self::profiles_gpu::*;
pub use self::profiles_host::*;
pub use self::registry::*;
pub use self::target::*;
pub use self::tiles::*;

/// `native`'s dispatch economics on a host nobody hand-measured
/// (local-rtx-5080): [`DispatchCost`] read from a per-host cache, measured on a
/// miss.
///
/// - `dispatch_ns`, `lanes`: [`mapal_rt::calibrate::wave`] — a full-width wave
///   through the emitted entry points, workers parked, minus the same task
///   inline (the S51 difference method, on the runtime itself rather than a
///   stand-in).
/// - `flops_per_ns`: one lane's FMA peak (independent vector FMA chains at the
///   detected width); `bytes_per_ns`: one lane streaming an L3-resident 8 MiB
///   buffer. Both are UPPER bounds on what an emitted kernel reaches, so a
///   serial-time estimate from them is a lower bound; with `lanes` known the
///   emitter turns it into a width (`sqrt`), where an estimate `r` times low
///   costs a `sqrt(r)` narrower wave, not an inlined large task.
///
/// `MAPAL_NATIVE_DISPATCH`: `off` = none (the OFF arm: every task the whole
/// pool), `measure` = re-measure and rewrite the cache, `d,l,f,b` = forced
/// (`dispatch_ns,lanes,flops_per_ns,bytes_per_ns`; tests). The cache key is the
/// CPU model, logical CPUs, pool lanes (`MAPAL_PAR`) and the compiler version.
/// A miss measured while the machine is busy ([`host_dispatch::busy`]) or by
/// an unoptimized build serves its process only and is not cached.
///
/// Only the `native` profile reads any of this: a named host or target
/// (`generic`, `x86-64-v3`, a GPU part) carries its own `dispatch`, so a
/// bundle cross-compiled here never inherits this machine's economics
/// (`tests/named_host_dispatch.rs`).
#[cfg(target_os = "linux")]
pub(crate) mod host_dispatch;

#[cfg(not(target_os = "linux"))]
pub(crate) mod host_dispatch {}

#[cfg(test)]
mod calib_supply_tests;
#[cfg(test)]
mod dispatch_width_tests;
#[cfg(test)]
mod tests;
