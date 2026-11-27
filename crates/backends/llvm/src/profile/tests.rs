//! Tests of the target profiles, in thematic submodules.

use super::*;

const F32: Ty = Ty::Float { bits: 32 };
const F64: Ty = Ty::Float { bits: 64 };

mod gpu_costs;
mod gpu_geometry;
mod host_tiles;
