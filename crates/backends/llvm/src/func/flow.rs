//! **`flow`: one footprint rule for device groups** (design §2).
//!
//! A consumer joins its producer's group when, for every work unit of the
//! group — one row `i` of its `rows` — the producer cells it reads lie in that
//! unit ([`mapal_ir::CategoryIr::footprint_plan`]'s [`Reach`], checked with
//! [`Reach::rect`]), or are a halo the group recomputes (a classifiable leaf
//! whose own reads are all complete in `Gmem`). A fold is never recomputed: its
//! readers must read row `i` only. A member no outside reader needs
//! (`read_outside`, `last_use_plan`'s escapes, a carried next state) is never
//! stored: a per-row value stays in a register, a cell is evaluated at its
//! read. The budget is the profile's (registers, launch pointers) and, where
//! the part has measured costs, the roofline of the fused launch against the
//! unfused ones.
//!
//! **Placement runs first.** Every member is a unit the residency plan already
//! put on the device — a `devmap` cell, a `rowfold` fold — so a group moves no
//! data: it removes launches, stores and loads. The tile root (a cell reading
//! a tile site's output at its own cell, `resident.rs:epilogues`) is the same
//! rule with the tile's block as the unit, and the OFF arm
//! (`--gpu-off=flow`): with the rung off only tile roots group.
//!
//! **Realization.** One thread per unit row, members in topo order as
//! phases: a fold walks its row in order (the oracle's chain, ADR-0028 D4), a
//! per-row cell is one call, a stored cell of `w > 1` cells per row a counted
//! loop of calls. Every body is the host's own text (or its one-cell-buffer
//! variant), so each cell is the host's bits; LLVM inlines the bodies and
//! CSEs a cell read twice at one address.

use super::resident::{Cell, ResidentPlan, array_len};
use super::*;

#[path = "stream.rs"]
pub(crate) mod stream;
use mapal_ir::{DepLaw, Reach, RowFoldOp, RowFoldSite, TileSite, UnitGroup, UnitKind};

mod emit_unit;
mod gen_slices;
mod gen_tile;
mod geo;
mod slices;
mod unit_gen;
mod unit_kernel;
mod units;

use self::geo::{
    SAME, SECTOR_BYTES, block_lanes, rowwise, rowwise_single, step_identity, tree_step,
};
pub(crate) use self::geo::{row_groups, segments, tile_split, unit_geo};

/// Where one input of a unit member comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum USrc {
    /// Member `k`'s value at the address `reach` gives the reader's cell.
    /// `row`: every such address lies in the reader's unit row (a per-row
    /// member's register serves it); else a halo the group recomputes.
    Member { k: usize, reach: Reach, row: bool },
    /// The unit's `Gmem` operand `g`, complete before the launch.
    Gmem(usize),
    /// The cell index itself (an `iota`'s element).
    Index,
    /// A closed-form capture evaluated at the reader's cell (`cell-law`).
    Law(FuncId),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UKind {
    /// A row fold; `colmajor`: its producing tile stores column-major for it.
    Fold {
        site: RowFoldSite,
        colmajor: Option<MorphismId>,
    },
    Cell {
        body: FuncId,
        captures: u32,
    },
    /// compile-time-graph A: a tile site in a row-block unit — its `a`
    /// operand `srcs[0]` (the unit's own rows or `Gmem`), `b` `srcs[1]`.
    Tile {
        site: TileSite,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UMember {
    pub m: MorphismId,
    pub out: ObjectId,
    pub kind: UKind,
    /// Cells of this member per unit row: `extent / rows`.
    pub w: u64,
    /// A reader outside the group exists: the value is stored.
    pub store: bool,
    /// One per body input slot (a cell), or the fold's one read.
    pub srcs: Vec<USrc>,
}

impl UMember {
    /// One value per unit row (a fold, or a cell of `rows` cells).
    fn per_row(&self) -> bool {
        self.w == 1
    }
}

/// A unit group: one launch, `rows` threads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Unit {
    pub id: u64,
    pub rows: u64,
    pub members: Vec<UMember>,
    /// Operand pointers, in first-use order.
    pub gmem: Vec<ObjectId>,
    /// sweep-attention `unit-uniform`: per `gmem` entry, the captured
    /// numbers' terms its launch moves the pointer by (`None`: the array's
    /// start; a missing entry likewise).
    pub gmove: Vec<Option<GMove>>,
    /// compile-time-graph A: a row-block unit (a tile member): `b` rows a
    /// block. `None`: one row a block or thread, as before.
    pub block: Option<Block>,
    /// long-rows `row-slices`: each row over `s` blocks. `None`: one block
    /// (or thread) a row.
    pub slices: Option<Slices>,
    /// flash-attention `stream`: one pass over windows of the group's axis
    /// ([`stream::Stream`]). `None`: every other realization.
    pub stream: Option<stream::Stream>,
}

/// sweep-attention `unit-uniform`: a unit's memory operand read at an
/// address a captured number moves ([`mapal_ir::RadixTile::ua`] / `ub`: a
/// loop counter slicing a KV head group) — one value a launch, so the launch
/// passes the pointer moved by [`mapal_ir::uniform_at`] of the terms, and
/// the kernel reads the site's own form from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GMove {
    /// The member map's source: the product whose capture slots the terms
    /// read.
    pub source: ObjectId,
    pub terms: Vec<mapal_ir::Digit>,
    /// The operand's element bytes in memory.
    pub elem: u64,
}

/// mask-frontend `stream-numbers`: the numbers a streamed unit's tile
/// epilogues read after their folds ([`mapal_ir::TilePost::after`]: a
/// scale, a guard's per-launch side) — `(member, capture slot)` in member
/// then slot order, one 8-byte launch argument each after the zero pass's.
pub(crate) fn stream_numbers(u: &Unit) -> Vec<(usize, u32)> {
    if u.stream.is_none() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (k, mb) in u.members.iter().enumerate() {
        if let UKind::Tile { site } = &mb.kind
            && let Some(p) = &site.post
        {
            out.extend(p.after.iter().map(|&s| (k, s)));
        }
    }
    out
}

/// long-rows `row-slices`: a unit whose rows each run over `s` blocks
/// ([`FnEmit::row_slices`]). A fold's partial state per block goes to a
/// workspace; a later phase combines a row's `s` partials in block order
/// (every block of the row alike), applies the laws `fold_combine` gave its
/// term (a factor or shift read once per row), then the seed. Members run in
/// the phase their sources are complete in; consecutive phases meet at a
/// grid barrier of one cooperative launch.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Slices {
    pub s: u64,
    /// Per member: the phase it runs in (a fold's partials, a cell's cells).
    pub phase: Vec<u32>,
    /// Per member: the first phase its value is complete in (a fold's
    /// combine; a cell's own phase).
    pub avail: Vec<u32>,
    pub phases: u32,
    /// Per fold member: `(per-row member, its slot in the fold's term cell,
    /// law)` — the reads its partials take as the law's identity, applied
    /// after the combine.
    pub laws: Vec<Vec<(usize, u32, DepLaw)>>,
    /// Per fold member, beside `laws`: a [`DepLaw::Rescale`] law's exponent
    /// (ADR-0042 D7). The partials take the per-row value's own partial over
    /// the same slice as their shift (a running max / min of the exponent's
    /// `x`); the combine rescales each by `exp(a(x := s_B, v := v))`.
    pub args: Vec<Vec<Option<mapal_ir::RescaleArg>>>,
    /// The model's `(sliced, one block a row)` ns; `(0, 0)` when forced.
    pub ns: (u64, u64),
    /// long-prefill `slice-launches`: the phases meet at kernel boundaries
    /// — one launch a phase, the kernel's `%phase` argument selecting its
    /// phase — where the part's measured launches undercut a cooperative
    /// launch and its grid barriers (outside a loop, the process's first
    /// cooperative launch). `false`: one cooperative launch.
    pub launches: bool,
}

impl Slices {
    /// Grid barriers between the phases: a cooperative launch.
    pub fn coop(&self) -> bool {
        self.phases > 1 && !self.launches
    }
}

/// A row-block unit's layout (`unit_plan` + the profile's choice of `b`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Block {
    /// Rows one unit (block) holds.
    pub b: u64,
    /// Per member: the on-chip row buffer `unit_plan` colors it into.
    pub color: Vec<Option<usize>>,
    /// The cost model's fused and split times, ns (the intent line's).
    pub ns: (u64, u64),
    /// kernel-merging `row-stage`: per member, how a tile stages its operand
    /// panels through shared memory (`None`: read straight from memory).
    pub stage: Vec<Option<Stage>>,
}

/// kernel-merging `row-stage`: a row-block tile's operand panels staged
/// through shared memory. One panel trip folds `kc` of each k-slice's `k`;
/// the column operand's panel holds those `kc` for every column (the tile's
/// row-independent operand: each block reads it once a trip, as the separate
/// tile kernel's panels do), a row operand read from memory its `b` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stage {
    pub kc: u64,
    /// Panel buffers: 1 = copy, barrier, fold; 2 = the next panel lands
    /// (`cp.async`) while this one folds.
    pub stages: u64,
    /// The column operand runs along `k` (`ck = 1`): `[slice][column][kc]`,
    /// its 16-byte chunks XOR-swizzled by row so 8 lanes' 16-byte reads of 8
    /// consecutive columns hit 32 banks; else `[slice][kc][column]`
    /// (consecutive lanes, consecutive words).
    pub b_kmaj: bool,
    /// 16-byte copies of the column operand (else one element a copy).
    pub b_vec: bool,
    /// The row operand is read from memory: staged `[slice][row][kc]`, by
    /// 16-byte copies where its rows run along `k` aligned (`a_vec`).
    pub a: bool,
    pub a_vec: bool,
    /// The copies are `cp.async` (else each thread loads and stores).
    pub cp: bool,
    /// Column passes: each folds all of `k` for `1 / passes` of a thread's
    /// columns (a contiguous `c / passes` of the row), its panels that wide.
    pub passes: u64,
}

impl Stage {
    /// `(a panel bytes, one buffer's bytes)`, both 16-byte multiples.
    pub fn bytes(&self, b: u64, c: u64, slices: u64, e: u64) -> (u64, u64) {
        let a = match self.a {
            true => (slices * b * self.kc * e).next_multiple_of(16),
            false => 0,
        };
        (
            a,
            a + (slices * (c / self.passes) * self.kc * e).next_multiple_of(16),
        )
    }

    /// Copies each thread issues a panel trip: `(a, b)`.
    pub fn copies(&self, b: u64, c: u64, slices: u64, e: u64, lanes: u64) -> (u64, u64) {
        let v = 16 / e;
        let unit = |vec: bool| if vec { v } else { 1 };
        let a = match self.a {
            true => (slices * b * self.kc / unit(self.a_vec)).div_ceil(lanes),
            false => 0,
        };
        (
            a,
            (slices * (c / self.passes) * self.kc / unit(self.b_vec)).div_ceil(lanes),
        )
    }

    /// The intent field: `kc<kc>x<stages>[p<passes>]:<kmaj|krow>:<v|e>[:a<v|e>][:cp]`.
    pub fn label(&self) -> String {
        let ve = |x: bool| if x { "v" } else { "e" };
        format!(
            "kc{}x{}{}:{}:{}{}{}",
            self.kc,
            self.stages,
            match self.passes {
                1 => String::new(),
                p => format!("p{p}"),
            },
            if self.b_kmaj { "kmaj" } else { "krow" },
            ve(self.b_vec),
            if self.a {
                format!(":a{}", ve(self.a_vec))
            } else {
                String::new()
            },
            if self.cp { ":cp" } else { "" }
        )
    }
}

/// The most staging combinations a row-block `b` weighs.
const STAGE_COMBOS: usize = 256;

/// The most copies a thread issues for one panel (a panel wider than that
/// keeps its operands read straight from memory).
const STAGE_COPIES: u64 = 32;

/// kernel-merging `row-stage`: tile `site`'s staging at a panel depth of at
/// most `kc` (the largest power of two dividing the slice's `k`, at least
/// one 16-byte chunk) in `stages` buffers, the row operand staged when it is
/// read from memory (`a_gmem`). `None` for an element other than 4 or 8
/// bytes, a slice shallower than one chunk, or more than [`STAGE_COPIES`]
/// copies a thread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_of(
    site: &TileSite,
    lanes: u64,
    free: bool,
    b: u64,
    a_gmem: bool,
    kc: u64,
    stages: u64,
    passes: u64,
    cp: bool,
) -> Option<Stage> {
    let e = crate::profile::elem_bytes(&site.elem);
    if e != 4 && e != 8 {
        return None;
    }
    let v = 16 / e;
    let (nj, slices) = tile_split(site, lanes, free);
    if nj % passes != 0 {
        return None;
    }
    let kper = site.k / slices;
    let mut kc = kc.min(kper);
    while kc > 1 && !kper.is_multiple_of(kc) {
        kc /= 2;
    }
    if kc < v || !kc.is_multiple_of(v) || !kper.is_multiple_of(kc) {
        return None;
    }
    let b_kmaj = site.b.ck == 1;
    let b_vec = match b_kmaj {
        true => site.b.base.is_multiple_of(v) && site.b.clane.is_multiple_of(v),
        false => {
            site.b.clane == 1
                && site.c.is_multiple_of(v)
                && site.b.base.is_multiple_of(v)
                && site.b.ck.is_multiple_of(v)
        }
    };
    let a_vec =
        a_gmem && site.a.ck == 1 && site.a.base.is_multiple_of(v) && site.a.ci.is_multiple_of(v);
    let st = Stage {
        kc,
        stages,
        b_kmaj,
        b_vec,
        a: a_gmem,
        a_vec,
        cp,
        passes,
    };
    let (ca, cb) = st.copies(b, site.c, slices, e, lanes);
    (ca + cb <= STAGE_COPIES).then_some(st)
}

/// kernel-merging `row-stage`: a tile's staging candidates — panel depth
/// (one 32-byte sector a column row where it fits, deeper for more FMA a
/// barrier) by buffers by column passes; the plan model picks.
pub(crate) fn stage_cands(cp: bool) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    for p in [1u64, 2, 4] {
        for s in [2u64, 1] {
            if s == 2 && !cp {
                continue;
            }
            for kc in [16u64, 8, 4, 2] {
                out.push((kc, s, p));
            }
        }
    }
    out
}

/// A unit's kernel for the device module: the function, its members (whose
/// own kernels are not emitted), the bodies it calls and the kernel text.
pub(crate) struct UnitKernel {
    pub f: FuncId,
    pub members: Vec<MorphismId>,
    pub fns: Vec<(String, String)>,
    pub text: String,
}

/// A placed device unit a group may take.
#[derive(Clone)]
enum Cand {
    Fold(super::rowfold::RowFold),
    Cell(Cell),
    Tile(TileSite),
}

/// A unit kernel's geometry. The row's cells are independent (`cell_plan`), so
/// a row is a block of `lanes` threads — its widest extent (a fold's `k`, a
/// cell member's `w`) to a power of two in `[warp, 256]` — each lane taking
/// every `lanes`-th cell; a unit of one value per row keeps one thread a row,
/// 256 rows a block. Where the order is free a fold of an associative step is
/// a tree across the lanes (its partials in shared memory); a cell row with
/// several readers in the unit is computed once into shared memory and read
/// there (`last_use_plan`'s uses inside the unit), within the profile's
/// per-block shared memory. Elsewhere a fold walks in order, every lane alike.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Geo {
    pub lanes: u64,
    pub gx: u64,
    pub block: u64,
    /// Dynamic shared memory bytes of the launch.
    pub smem: u64,
    /// Per member: a tree fold's partials' offset.
    pub tree: Vec<Option<u64>>,
    /// Per member: a cell row kept on chip — its offset.
    pub kept: Vec<Option<u64>>,
    /// Row-block units: per member, the `b` per-row values a later segment
    /// reads (offset); the tile k-slices' partials (offset). Empty / `None`
    /// on every other unit.
    pub xrow: Vec<Option<u64>>,
    pub part: Option<u64>,
    /// kernel-merging `row-stage`: the staged tiles' panel buffers (the
    /// region a split tile's partials share: they are written after its
    /// last panel is read).
    pub panel: Option<u64>,
}

/// A cell member's call: callee, argument and result types, each slot's
/// array element type, the capture slots fed by one-cell buffers, and the
/// law function of each law slot.
struct CallInfo {
    callee: String,
    arg: String,
    ret: String,
    field_ty: Vec<String>,
    captures: u32,
    slots: Vec<u32>,
    laws: Vec<(u32, String)>,
}

/// The unit kernel's body builder.
struct UnitGen<'u> {
    out: String,
    entry: String,
    n: u32,
    /// The current block's label (a phi's predecessor).
    cur: String,
    /// Each per-row member's value in this row, once its phase ran.
    vals: Vec<Option<String>>,
    calls: Vec<Option<CallInfo>>,
    /// Each stored member's output parameter `%y<j>`.
    ys: Vec<Option<usize>>,
    gmem_ty: Vec<String>,
    /// [`Geo`]'s lanes a row, tree partials and on-chip rows.
    lanes: u64,
    tree: Vec<Option<u64>>,
    kept: Vec<Option<u64>>,
    u: &'u Unit,
    /// The row the phases address (`%i`; a row-block segment's own).
    row: String,
    /// The thread's lane in its row (`%l`; a row-block segment's lane in
    /// its row group) and the row group's first lane (`0`).
    lane: String,
    lane_base: String,
    /// A row-block unit: the row's index in the block.
    rloc: Option<String>,
    /// Per member its segment ([`segments`]), and the one emitting.
    seg: Vec<usize>,
    cur_seg: usize,
    /// [`Geo`]'s per-row values a later segment reads, and tile partials.
    xrow: Vec<Option<u64>>,
    part: Option<u64>,
    /// [`Geo`]'s staged panel buffers.
    panel: Option<u64>,
    /// The order is free (tile k-slices regroup); the face contracts.
    free: bool,
    fma: bool,
    /// gapfix `shfl-tree`: a lane tree combines in registers inside a warp
    /// (`shfl.sync.bfly`) and across warps through one shared exchange.
    shfl: bool,
    /// long-rows `row-slices`: the kernel's name (its workspaces and grid
    /// barrier counter), and `(member, slot, literal)`: a cell's read a fold's
    /// partials take as its law's identity.
    name: String,
    subst: Vec<(usize, u32, String)>,
    /// flash-attention `stream`: each window's rescale factors computed so
    /// far, by `(old, new, exponent)` — every law sharing one reads it once.
    alphas: Vec<(String, String)>,
    /// `row-slices`: each fold member's partial over this item's slice (a
    /// rescale law's shift for a later fold of the same item).
    parts: Vec<Option<String>>,
}
