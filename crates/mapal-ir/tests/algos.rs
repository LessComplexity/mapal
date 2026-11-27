//! SCC / topo / loop-structure / deep-graph tests (DESIGN §16 item 4).

use mapal_ir::{
    Dest, ElemSrc, FuncKind, IrBuilder, MorphismId, ObjectId, Operation, PathPlan, SourceLoc,
    TaskKind, TileKSplit, TileRead, TileSite, Ty, Value, WaitEntry, validate,
};
use proptest::prelude::*;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

#[path = "algos/bounds.rs"]
mod bounds;
#[path = "algos/cell.rs"]
mod cell;
#[path = "algos/delta_footprint.rs"]
mod delta_footprint;
#[path = "algos/elem_guard.rs"]
mod elem_guard;
#[path = "algos/emission.rs"]
mod emission;
#[path = "algos/last_use.rs"]
mod last_use;
#[path = "algos/path.rs"]
mod path;
#[path = "algos/scc.rs"]
mod scc;
#[path = "algos/tile.rs"]
mod tile;
use self::bounds::*;
use self::cell::*;
use self::last_use::*;
use self::path::*;
use self::scc::*;
use self::tile::*;
