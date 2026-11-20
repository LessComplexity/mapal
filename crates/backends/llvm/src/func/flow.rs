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
use mapal_ir::{Reach, RowFoldOp, RowFoldSite, TileSite, UnitGroup, UnitKind};

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
    /// compile-time-graph A: a row-block unit (a tile member): `b` rows a
    /// block. `None`: one row a block or thread, as before.
    pub block: Option<Block>,
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

/// A fold step the lanes may reassociate: `+`, `×`, or a running max/min.
fn tree_step(op: &RowFoldOp) -> bool {
    match op {
        RowFoldOp::Arith { op, .. } => matches!(op, Operation::Add | Operation::Mul),
        RowFoldOp::Select { .. } => true,
    }
}

pub(crate) fn unit_geo(
    u: &Unit,
    profile: &TargetProfile,
    free: bool,
    bytes: impl Fn(&UMember) -> u64,
) -> Geo {
    let g = profile.gpu().expect("a unit has a GPU profile");
    if let Some(blk) = &u.block {
        return block_geo(u, blk, g, free, bytes);
    }
    let n = u.members.len();
    let extent = u
        .members
        .iter()
        .map(|mb| match &mb.kind {
            UKind::Fold { site, .. } => site.k,
            UKind::Cell { .. } => mb.w,
            UKind::Tile { site } => site.c,
        })
        .max()
        .unwrap_or(1);
    if extent <= 1 {
        let (gx, block) = gpu::cell_grid(u.rows);
        return Geo {
            lanes: 1,
            gx,
            block,
            smem: 0,
            tree: vec![None; n],
            kept: vec![None; n],
            xrow: Vec::new(),
            part: None,
            panel: None,
        };
    }
    let lanes = extent.next_power_of_two().clamp(g.warp, 256);
    // ponytail: the static-launch limit; a larger row needs the opt-in attribute.
    let budget = g.smem_per_block.min(48 * 1024);
    let mut smem = 0u64;
    let mut take = |b: u64| {
        let off = smem.next_multiple_of(16);
        (off + b <= budget).then(|| {
            smem = off + b;
            off
        })
    };
    let (mut tree, mut kept) = (vec![None; n], vec![None; n]);
    for (k, mb) in u.members.iter().enumerate() {
        match &mb.kind {
            UKind::Fold { site, .. } if free && tree_step(&site.op) => {
                tree[k] = take(lanes * bytes(mb));
            }
            UKind::Cell { .. } if !mb.per_row() => {
                let reads = u.members[k + 1..]
                    .iter()
                    .flat_map(|r| &r.srcs)
                    .filter(|s| matches!(s, USrc::Member { k: j, row: true, .. } if *j == k))
                    .count();
                if reads >= 2 || (mb.store && reads >= 1) {
                    kept[k] = take(mb.w * bytes(mb));
                }
            }
            _ => {}
        }
    }
    Geo {
        lanes,
        gx: u.rows,
        block: lanes,
        smem,
        tree,
        kept,
        xrow: Vec::new(),
        part: None,
        panel: None,
    }
}

/// A row-block unit's threads: its widest row extent (a fold's `k`, a cell's
/// `w`, a tile's `c`) to a power of two in `[warp, 256]`, as [`unit_geo`]'s.
fn block_lanes(members: &[UMember], warp: u64) -> u64 {
    members
        .iter()
        .map(|mb| match &mb.kind {
            UKind::Fold { site, .. } => site.k,
            UKind::Cell { .. } => mb.w,
            UKind::Tile { site } => site.c,
        })
        .max()
        .unwrap_or(1)
        .next_power_of_two()
        .clamp(warp, 256)
}

/// A tile member's thread split over `lanes` threads: `(columns a thread
/// holds, k-slices)`. A row of `c ≥ lanes` cells gives each thread `c ÷ lanes`
/// columns; a narrower row splits `k` into `lanes ÷ c` slices whose partials
/// combine as a tree where the order is free (else one slice, the other
/// threads idle).
pub(crate) fn tile_split(site: &TileSite, lanes: u64, free: bool) -> (u64, u64) {
    if site.c >= lanes {
        return (site.c / lanes, 1);
    }
    // Every thread a slice (the combine's barrier is outside any branch), or
    // one slice.
    let s = lanes / site.c;
    match free && lanes % site.c == 0 && site.k % s == 0 {
        true => (1, s),
        false => (1, 1),
    }
}

/// Rows a row-block segment runs at once: the block's lanes split into one
/// group a row, each of at least [`ROW_GROUP_LANES`] lanes — the rows are
/// independent (`cell_plan`), so a batch of them hides each row's chain of
/// barriered tree steps behind the others'.
pub(crate) fn row_groups(b: u64, lanes: u64) -> u64 {
    let mut p = b;
    while p > 1 && (lanes / p < ROW_GROUP_LANES || b % p != 0) {
        p /= 2;
    }
    p.max(1)
}

/// The L2 → SM transfer granule: a load moves every 32-byte sector it touches.
const SECTOR_BYTES: u64 = 32;

/// The fewest lanes a row of a row-block segment takes.
const ROW_GROUP_LANES: u64 = 8;

/// The row-block members' segments: a tile is its own; the members between
/// two tiles run row by row in one.
pub(crate) fn segments(members: &[UMember]) -> Vec<usize> {
    let mut seg = Vec::with_capacity(members.len());
    let mut s = 0;
    for (k, mb) in members.iter().enumerate() {
        let tile = matches!(mb.kind, UKind::Tile { .. });
        let prev = k > 0 && matches!(members[k - 1].kind, UKind::Tile { .. });
        if k > 0 && (tile || prev) {
            s += 1;
        }
        seg.push(s);
    }
    seg
}

/// The row-block geometry (compile-time-graph A): `lanes` threads a block of
/// `b` rows; shared memory holds each `unit_plan` color's `b` rows, each tree
/// fold's lane partials (reused row by row), the per-row values a later
/// segment reads, and a split tile's k-slice partials.
fn block_geo(
    u: &Unit,
    blk: &Block,
    g: &crate::profile::Gpu,
    free: bool,
    bytes: impl Fn(&UMember) -> u64,
) -> Geo {
    let n = u.members.len();
    let lanes = block_lanes(&u.members, g.warp);
    let b = blk.b;
    let mut smem = 0u64;
    let mut take = |x: u64| {
        let off = smem.next_multiple_of(16);
        smem = off + x;
        off
    };
    let ncol = blk.color.iter().flatten().max().map_or(0, |c| c + 1);
    let col_off: Vec<u64> = (0..ncol)
        .map(|c| {
            let width = u
                .members
                .iter()
                .zip(&blk.color)
                .filter(|(mb, x)| **x == Some(c) && !mb.per_row())
                .map(|(mb, _)| mb.w * bytes(mb))
                .max()
                .unwrap_or(0);
            take(b * width)
        })
        .collect();
    let seg = segments(&u.members);
    let (mut tree, mut kept, mut xrow) = (vec![None; n], vec![None; n], vec![None; n]);
    // Phase-local scratch — tree partials (a fold's phase), panels (a tile's
    // k-loop), split partials (its tail) — shares one region once a tile is
    // staged: no two of them are live in one phase.
    let scratch = blk.stage.iter().any(Option::is_some);
    let mut trees = 0u64;
    for (k, mb) in u.members.iter().enumerate() {
        if let Some(c) = blk.color[k]
            && !mb.per_row()
        {
            kept[k] = Some(col_off[c]);
        }
        if let UKind::Fold { site, .. } = &mb.kind
            && free
            && tree_step(&site.op)
        {
            // kernel-merging `row-stage`: a staged unit's tree partials live
            // in its phase-local scratch (below), at offsets from its start.
            tree[k] = Some(match scratch {
                true => {
                    let off = trees.next_multiple_of(16);
                    trees = off + lanes * bytes(mb);
                    off
                }
                false => take(lanes * bytes(mb)),
            });
        }
        let later = u.members.iter().enumerate().any(|(j, r)| {
            seg[j] != seg[k]
                && r.srcs
                    .iter()
                    .any(|s| matches!(s, USrc::Member { k: p, .. } if *p == k))
        });
        if mb.per_row() && later {
            xrow[k] = Some(take(b * bytes(mb)));
        }
    }
    let part = u
        .members
        .iter()
        .filter_map(|mb| match &mb.kind {
            UKind::Tile { site } if tile_split(site, lanes, free).1 > 1 => {
                Some(lanes * b * bytes(mb))
            }
            _ => None,
        })
        .max();
    // kernel-merging `row-stage`: the panel buffers, one region with the
    // split partials (each tile's partials follow its last panel).
    let panels = u
        .members
        .iter()
        .zip(&blk.stage)
        .filter_map(|(mb, st)| match (&mb.kind, st) {
            (UKind::Tile { site }, Some(st)) => {
                let slices = tile_split(site, lanes, free).1;
                Some(st.stages * st.bytes(b, site.c, slices, bytes(mb)).1)
            }
            _ => None,
        })
        .max();
    let (part, panel) = match panels {
        None => (part.map(&mut take), None),
        Some(p) => {
            let off = take(p.max(part.unwrap_or(0)).max(trees));
            for x in tree.iter_mut().flatten() {
                *x += off;
            }
            (part.map(|_| off), Some(off))
        }
    };
    Geo {
        lanes,
        gx: u.rows / b,
        block: lanes,
        smem,
        tree,
        kept,
        xrow,
        part,
        panel,
    }
}

/// Does `reach`, read by a consumer with `cw` cells per row, stay inside the
/// producer's row `i` (`pw` cells per row) for every row of `rows`? The IR's
/// row closure ([`Reach::within_rows`], `unit_plan`'s).
fn rowwise(reach: &Reach, cw: u64, pw: u64, rows: u64) -> bool {
    reach.within_rows(cw, pw, rows)
}

/// The rows a reach of `cw`-cell consumer rows reads a per-row value at: the
/// row itself, every row.
fn rowwise_single(reach: &Reach, cw: u64, rows: u64) -> bool {
    rowwise(reach, cw, 1, rows)
}

const SAME: Reach = Reach::Affine {
    w: 1,
    base: 0,
    cq: 1,
    cr: 0,
    taps: Vec::new(),
};

impl<'a> FnEmit<'a> {
    /// **The rule (`flow`)**: group the placed folds and cells
    /// of this plan into unit kernels, greedily in position order — a group
    /// starts at a unit no group holds and takes, one at a time, the earliest
    /// reader of a member that [`Self::unit_of`] admits. Refusals are
    /// recorded with their reason for the host intent line.
    pub(super) fn flow_units(&self, plan: &mut ResidentPlan) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let mut refused: Vec<(MorphismId, MorphismId, &'static str)> = Vec::new();
        let mut absorbed = 0;
        // compile-time-graph A: each row-block group's refusal (its launch).
        let mut block_refused: Vec<(MorphismId, String)> = Vec::new();
        if rungs.flow && plan.chain.is_empty() {
            let mut cand: SecondaryMap<MorphismId, Cand> = SecondaryMap::new();
            for (m, r) in plan.rowfold.iter() {
                cand.insert(m, Cand::Fold(r.clone()));
            }
            // ponytail: a unit kernel takes no scalar argument; a cell that
            // reads a uniform runs alone. Pass uniforms to units when one pays.
            for (m, c) in plan
                .cells
                .iter()
                .filter(|(_, c)| c.uniform.is_empty() && c.nested.is_empty())
            {
                cand.insert(m, Cand::Cell(c.clone()));
            }
            let fp = self.ir.footprint_plan(self.f);
            let region = self.regions();
            let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
            let target = |m: MorphismId| self.ir.morphism(m).expect("unit").target;
            let mut order: Vec<MorphismId> = cand.keys().collect();
            order.sort_by_key(|&m| pos(m));
            let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            let mut groups: Vec<Vec<MorphismId>> = Vec::new();
            // compile-time-graph A: the row-block groups `unit_plan` proves
            // closed — tile sites and the folds and cells reading their rows —
            // each one unit kernel where the part holds its rows on chip.
            let mut blocks: Vec<Unit> = Vec::new();
            // The part's rung table decides where the rule runs (a forced
            // `b` measures it anywhere).
            if self.gpu_arms.row_block && (rungs.row_block || self.gpu_arms.row_block_b > 0) {
                for grp in self.ir.unit_plan(self.f).groups {
                    let ms: Vec<MorphismId> = grp.members.iter().map(|mb| mb.m).collect();
                    let mut bc = cand.clone();
                    let placed = grp.members.iter().all(|mb| match &mb.kind {
                        UnitKind::Tile(t) => {
                            // The site as the device plan holds it, outside
                            // an epilogue group and a retained row.
                            let site = self
                                .tile_plan
                                .as_ref()
                                .and_then(|p| p.sites.get(mb.m))
                                .cloned()
                                .unwrap_or_else(|| t.clone());
                            let ok = self.device.contains_key(mb.m)
                                && !self.groups.contains_key(mb.m)
                                && !self.retain.contains_key(mb.out);
                            if ok {
                                bc.insert(mb.m, Cand::Tile(site));
                            }
                            ok
                        }
                        _ => cand.contains_key(mb.m) && !taken.contains_key(mb.m),
                    });
                    if !placed {
                        block_refused.push((ms[0], "placed".into()));
                        continue;
                    }
                    match self.unit_of(&ms, &bc, &fp, &region, plan, Some(&grp)) {
                        Ok(u) => {
                            for &m in &ms {
                                taken.insert(m, ());
                            }
                            blocks.push(u);
                        }
                        // A cost refusal names the model's cheapest `b`:
                        // `cost:b<b>:<fused>/<split>` ns.
                        Err(why) => block_refused.push((
                            ms[0],
                            match (why, self.block_priced.take()) {
                                ("cost", Some((b, f, s))) => format!("cost:b{b}:{f}/{s}"),
                                _ => why.into(),
                            },
                        )),
                    }
                }
            }
            for &seed in &order {
                if taken.contains_key(seed) {
                    continue;
                }
                let mut g = vec![seed];
                loop {
                    let mut next: Vec<(MorphismId, MorphismId)> = g
                        .iter()
                        .flat_map(|&x| {
                            self.cell_readers(target(x))
                                .into_iter()
                                .map(move |r| (x, r))
                        })
                        .filter(|&(_, r)| {
                            cand.contains_key(r) && !taken.contains_key(r) && !g.contains(&r)
                        })
                        .collect();
                    next.sort_by_key(|&(_, r)| pos(r));
                    next.dedup_by_key(|(_, r)| *r);
                    let mut admitted = false;
                    for (p, r) in next {
                        let mut trial = g.clone();
                        trial.push(r);
                        match self.unit_of(&trial, &cand, &fp, &region, plan, None) {
                            Ok(_) => {
                                refused.retain(|&(_, c, _)| c != r);
                                g = trial;
                                admitted = true;
                                break;
                            }
                            Err(why) => {
                                refused.retain(|&(_, c, _)| c != r);
                                refused.push((p, r, why));
                            }
                        }
                    }
                    if !admitted {
                        break;
                    }
                }
                // A lone fold is a unit of one where its lanes may be a tree:
                // the row's steps across a block, not one thread's walk. Not
                // over a column-major operand (`rowfold-layout`'s sole-reader
                // store): there a row's cells are `rows` apart, so the lanes
                // would not coalesce and one thread's walk down a column does
                // (cpu-gpu-boundary item 5: rtx-4090 rowsum_4096 1.056).
                let tree = |m: MorphismId| {
                    self.order == crate::Order::Free
                        && matches!(cand.get(m), Some(Cand::Fold(r))
                            if tree_step(&r.site.op) && r.site.k > 1 && (r.colmajor.is_none() || !rungs.tree_gate))
                };
                if g.len() >= 2
                    || (tree(seed) && self.unit_of(&g, &cand, &fp, &region, plan, None).is_ok())
                {
                    for &m in &g {
                        taken.insert(m, ());
                    }
                    groups.push(g);
                }
            }
            for g in groups {
                let u = self
                    .unit_of(&g, &cand, &fp, &region, plan, None)
                    .expect("an admitted group is a unit");
                for (k, mb) in u.members.iter().enumerate() {
                    plan.cells.remove(mb.m);
                    plan.rowfold.remove(mb.m);
                    if k > 0 {
                        plan.skip.insert(mb.m, ());
                    }
                    if !mb.store {
                        for v in plan.free_after.values_mut() {
                            v.retain(|&o| o != mb.out);
                        }
                    }
                }
                refused.retain(|&(_, c, _)| !g.contains(&c));
                plan.units.insert(g[0], u);
            }
            for u in blocks {
                for (k, mb) in u.members.iter().enumerate() {
                    plan.cells.remove(mb.m);
                    plan.rowfold.remove(mb.m);
                    if k > 0 {
                        plan.skip.insert(mb.m, ());
                    }
                    if !mb.store {
                        for v in plan.free_after.values_mut() {
                            v.retain(|&o| o != mb.out);
                        }
                    }
                }
                absorbed += u
                    .members
                    .iter()
                    .filter(|mb| matches!(mb.kind, UKind::Tile { .. }))
                    .count();
                let g: Vec<MorphismId> = u.members.iter().map(|mb| mb.m).collect();
                refused.retain(|&(_, c, _)| !g.contains(&c));
                plan.units.insert(g[0], u);
            }
        }
        // The host intent: every group, every launch, every refused edge.
        let launches = self.device.len() - absorbed
            + plan.rowfold.len()
            + plan.cells.len()
            + plan.units.len()
            + plan.devgen.len();
        let name = |m: MorphismId| self.object_label(self.ir.morphism(m).expect("m").target);
        let refused: Vec<String> = refused
            .iter()
            .map(|&(p, c, why)| format!("{}->{}:{why}", name(p), name(c)))
            .collect();
        // Only where the rule decided something: a function it leaves alone
        // keeps its text.
        let block: String = block_refused
            .iter()
            .map(|(m, why)| format!(" row_block={}:{why}", name(*m)))
            .collect();
        if !plan.units.is_empty() || !refused.is_empty() || !block.is_empty() {
            plan.flow_line = format!(
                "; mapal.flow.host groups={} units={} launches={launches} refused={}{block}\n",
                self.groups.len() + plan.units.len(),
                plan.units.len(),
                refused.join(";")
            );
        }
    }

    /// An object's name in an intent line: its source name, else `o<seq>`.
    pub(super) fn object_label(&self, o: ObjectId) -> String {
        let obj = self.ir.object(o).expect("object resolves");
        match &obj.name {
            Some(n) => n.clone(),
            None => format!("{o:?}")
                .trim_start_matches("ObjectId(")
                .trim_end_matches(')')
                .replace('v', "_")
                .chars()
                .fold(String::from("o"), |mut s, c| {
                    s.push(c);
                    s
                }),
        }
    }

    /// The unit of the members `ms` (position order), or why not: `phase`
    /// (another region, or an operand not complete at the launch), `gated`,
    /// `footprint`, `pointers`, `budget`, `cost`.
    fn unit_of(
        &self,
        ms: &[MorphismId],
        cand: &SecondaryMap<MorphismId, Cand>,
        fp: &mapal_ir::FootprintPlan,
        region: &SecondaryMap<MorphismId, (Option<ObjectId>, bool)>,
        plan: &ResidentPlan,
        grp: Option<&UnitGroup>,
    ) -> Result<Unit, &'static str> {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let launch = ms[0];
        let rs = *region.get(launch).ok_or("phase")?;
        if ms.iter().any(|&m| region.get(m) != Some(&rs)) {
            return Err("phase");
        }
        if ms.iter().any(|&m| self.gated.contains_key(m)) {
            return Err("gated");
        }
        let target = |m: MorphismId| self.ir.morphism(m).expect("member").target;
        let extent = |m: MorphismId| array_len(&self.obj_ty(target(m)));
        let fold_rows: Vec<u64> = ms
            .iter()
            .filter_map(|&m| match &cand[m] {
                Cand::Fold(r) => Some(r.site.rows),
                Cand::Cell(_) | Cand::Tile(_) => None,
            })
            .collect();
        let rows = match grp {
            // A row-block group's axis is `unit_plan`'s.
            Some(g) => g.rows,
            None => fold_rows.first().copied().unwrap_or_else(|| extent(ms[0])),
        };
        let unaligned = match grp {
            Some(_) => ms.iter().any(|&m| extent(m) % rows != 0),
            None => {
                fold_rows.iter().any(|&r| r != rows)
                    || ms.iter().any(|&m| extent(m) % rows != 0)
                    || (fold_rows.is_empty() && ms.iter().any(|&m| extent(m) != rows))
            }
        };
        if rows == 0 || unaligned {
            return Err("footprint");
        }
        // Complete in `Gmem` when the group launches, where it runs: made by
        // a morphism before the launch in this region or straight-line, or a
        // view of this loop's state.
        let complete = |o: ObjectId| match self.ir.in_edges(o) {
            [] => true,
            [p] => {
                let made = pos(*p) < pos(launch)
                    && region.get(*p).is_some_and(|&r| r == rs || r.0.is_none());
                let view = rs.0.is_some_and(|mg| {
                    let mo = self.ir.morphism(*p).expect("morphism resolves");
                    matches!(mo.op, Operation::Proj { .. }) && mo.source == mg
                });
                made || view
            }
            _ => false,
        };
        let reach_of = |m: MorphismId, slot: u32| {
            fp.of(m)
                .iter()
                .find(|e| e.slot == slot && e.law.is_none())
                .map(|e| e.reach.clone())
        };
        let mut members: Vec<UMember> = Vec::new();
        let mut gmem: Vec<ObjectId> = Vec::new();
        let push = |o: ObjectId, gmem: &mut Vec<ObjectId>| match gmem.iter().position(|&x| x == o) {
            Some(g) => g,
            None => {
                gmem.push(o);
                gmem.len() - 1
            }
        };
        // May a reader with `cw` cells per row read member `p` at `reach`?
        let readable = |p: &UMember, reach: &Reach, cw: u64| -> Result<bool, &'static str> {
            let free = p.srcs.iter().all(|s| !matches!(s, USrc::Member { .. }));
            // Every read of the slot at one affine address (a body that
            // reads it at two takes the range, and no one-cell buffer).
            if !matches!(reach, Reach::Affine { .. }) {
                return Err("footprint");
            }
            match p.kind {
                UKind::Fold { .. } if rowwise_single(reach, cw, rows) => Ok(true),
                UKind::Fold { .. } => Err("footprint"),
                // A tile's rows are on chip: read in its own row, never a halo.
                UKind::Tile { .. } if rowwise(reach, cw, p.w, rows) => Ok(true),
                UKind::Tile { .. } => Err("footprint"),
                UKind::Cell { .. } if rowwise(reach, cw, p.w, rows) => Ok(true),
                UKind::Cell { .. } if free => Ok(false),
                UKind::Cell { .. } => Err("footprint"),
            }
        };
        for &m in ms {
            let morph = self.ir.morphism(m).expect("member");
            let w = extent(m) / rows;
            let member =
                |o: ObjectId, members: &[UMember]| members.iter().position(|mb| mb.out == o);
            let mb = match &cand[m] {
                Cand::Fold(r) => {
                    let a = self
                        .pair_source(morph.source, r.site.a.slot)
                        .ok_or("footprint")?;
                    let src = match member(a, &members) {
                        Some(k) => {
                            let reach = reach_of(m, r.site.a.slot).ok_or("footprint")?;
                            // A fold's step reads `k` cells of its row: the
                            // producer is a cell (recomputed at each), never
                            // another fold.
                            if !matches!(members[k].kind, UKind::Cell { .. } | UKind::Tile { .. }) {
                                return Err("footprint");
                            }
                            let row = readable(&members[k], &reach, 1)?;
                            USrc::Member { k, reach, row }
                        }
                        None if complete(a) => USrc::Gmem(push(a, &mut gmem)),
                        None => return Err("phase"),
                    };
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Fold {
                            site: r.site.clone(),
                            colmajor: r.colmajor,
                        },
                        w,
                        store: false,
                        srcs: vec![src],
                    }
                }
                Cand::Tile(site) => {
                    let a = self
                        .pair_source(morph.source, site.a.slot)
                        .ok_or("footprint")?;
                    let b = self
                        .pair_source(morph.source, site.b.slot)
                        .ok_or("footprint")?;
                    // `a` is the unit's own rows (a row-wide member read in
                    // its row, row-aligned) or complete in `Gmem`; `b` is read
                    // by every row: never a member.
                    let a_src = match member(a, &members) {
                        Some(k) => {
                            let p = &members[k];
                            if p.per_row()
                                || (site.a.ci != p.w && rows > 1)
                                || site.a.base + site.a.ck * (site.k - 1) >= p.w
                            {
                                return Err("footprint");
                            }
                            let reach = reach_of(m, site.a.slot).ok_or("footprint")?;
                            if !readable(p, &reach, site.c)? {
                                return Err("footprint");
                            }
                            USrc::Member {
                                k,
                                reach,
                                row: true,
                            }
                        }
                        None if complete(a) => USrc::Gmem(push(a, &mut gmem)),
                        None => return Err("phase"),
                    };
                    let b_src = match member(b, &members) {
                        Some(_) => return Err("footprint"),
                        None if complete(b) => USrc::Gmem(push(b, &mut gmem)),
                        None => return Err("phase"),
                    };
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Tile { site: site.clone() },
                        w,
                        store: false,
                        srcs: vec![a_src, b_src],
                    }
                }
                Cand::Cell(c) => {
                    let Operation::Map { body, captures } = morph.op else {
                        return Err("footprint");
                    };
                    let mut srcs = Vec::new();
                    for slot in 0..=captures {
                        let arr = if captures == 0 {
                            Some(morph.source)
                        } else {
                            self.pair_source(morph.source, slot)
                        }
                        .ok_or("footprint")?;
                        let law = c.laws.iter().find(|&&(s, _)| s == slot).map(|&(_, f)| f);
                        let src = if slot == captures && c.mapped.is_none() {
                            USrc::Index
                        } else if let Some(f) = law {
                            USrc::Law(f)
                        } else if let Some(k) = member(arr, &members) {
                            let reach = if slot == captures {
                                SAME
                            } else {
                                reach_of(m, slot).ok_or("footprint")?
                            };
                            let row = readable(&members[k], &reach, w)?;
                            USrc::Member { k, reach, row }
                        } else if complete(arr) {
                            USrc::Gmem(push(arr, &mut gmem))
                        } else {
                            return Err("phase");
                        };
                        srcs.push(src);
                    }
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Cell { body, captures },
                        w,
                        store: false,
                        srcs,
                    }
                }
            };
            members.push(mb);
        }
        for mb in &mut members {
            mb.store = self.lup.escapes(mb.out)
                || plan.is_carry_next(mb.out)
                || self.read_outside(mb.out, ms);
        }
        // Budget: one pointer per operand and stored value; per-row values
        // held in registers beside the address state.
        let gpu = self.profile.gpu().expect("a unit has a GPU profile");
        let stored = members.iter().filter(|mb| mb.store).count();
        if gmem.len() + stored > gpu::LAUNCHV_MAX {
            return Err("pointers");
        }
        if let Some(grp) = grp {
            let blk = self.block_choice(&members, rows, &gmem, grp)?;
            let key = format!(
                "unit|{members:?}|{rows}|{}|block{}",
                self.profile.name, blk.b
            );
            let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            });
            return Ok(Unit {
                id,
                rows,
                members,
                gmem,
                block: Some(blk),
            });
        }
        let elem =
            |o: ObjectId| crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(o)));
        let held: u64 = members
            .iter()
            .filter(|mb| mb.per_row())
            .map(|mb| elem(mb.out).div_ceil(4))
            .sum::<u64>()
            + crate::profile::FMA_ADDR_REGS;
        if held > gpu.regs_per_thread {
            return Err("budget");
        }
        if let Some(d) = gpu.cost.filter(|d| d.flops_per_ns > 0) {
            let (fused, unfused) = self.unit_roofline(&members, rows, &gmem, elem);
            let ns = |(f, b): (u64, u64)| {
                crate::profile::roofline_ns(f, b, d.flops_per_ns, d.bytes_per_ns)
            };
            // plan-cost (D1, D2): a launch costs its in-stream idle after the
            // op before it, not a host round trip; `launch_ns` where the part
            // has no plan rates (D8).
            let launch = |prev: u64| {
                self.profile
                    .op_ns(prev as f64, 0)
                    .map_or(d.launch_ns, |x| x.round() as u64)
            };
            let mut prev = u64::MAX;
            let unfused: u64 = unfused
                .into_iter()
                .map(|x| {
                    let t = launch(prev) + ns(x);
                    prev = ns(x);
                    t
                })
                .sum();
            if launch(u64::MAX) + ns(fused) > unfused {
                return Err("cost");
            }
        }
        let key = format!("unit|{members:?}|{rows}|{}", self.profile.name);
        let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        Ok(Unit {
            id,
            rows,
            members,
            gmem,
            block: None,
        })
    }

    /// **compile-time-graph A: how many rows a row-block unit holds.** The
    /// powers of two `b` dividing `rows` whose layout ([`block_geo`]) fits the
    /// part's opt-in shared memory and whose tile accumulators fit a thread's
    /// registers; the cheapest of them by the plan model
    /// ([`Self::block_cost`]), kept only when it beats the split launches. A
    /// part without plan rates, or a shape the model cannot price, keeps the
    /// boundary: nothing else weighs the unit against the launches it
    /// replaces. `Err` names the last test every `b` failed: `budget`,
    /// `unpriced`, `cost`.
    fn block_choice(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        grp: &UnitGroup,
    ) -> Result<Block, &'static str> {
        let gpu = self.profile.gpu().expect("a unit has a GPU profile");
        let free = self.order == crate::Order::Free;
        let color: Vec<Option<usize>> = members
            .iter()
            .map(|mb| {
                grp.members
                    .iter()
                    .find(|g| g.m == mb.m)
                    .and_then(|g| g.color)
            })
            .collect();
        let lanes = block_lanes(members, gpu.warp);
        for mb in members {
            if let UKind::Tile { site } = &mb.kind
                && !(site.c % lanes == 0 || lanes % site.c == 0)
            {
                return Err("footprint");
            }
        }
        let mut why = "budget";
        let mut best: Option<(Block, u64)> = None;
        let forced = self.gpu_arms.row_block_b;
        self.block_priced.set(None);
        // kernel-merging `row-stage`: each tile's panels staged through
        // shared memory at every candidate depth × buffers that fits (the
        // plan model picks); a `b` no staging fits reads them straight.
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        // `cp.async` where the part has it (sm_80+); `--gpu-off=async` copies
        // through registers.
        let cp = gpu.has_mma_sync() && !self.gpu_arms.off.cp_async;
        for sh in (0..=6).rev() {
            let b = 1u64 << sh;
            if !rows.is_multiple_of(b) || (forced > 0 && b != forced) {
                continue;
            }
            // Each tile's staged options; the combinations of them (each tile
            // its own, the panel region their largest), then — when no
            // combination fits — every tile read straight from memory.
            let opts: Vec<Vec<Option<Stage>>> = members
                .iter()
                .map(|mb| match &mb.kind {
                    UKind::Tile { site } if rungs.row_stage => {
                        let mut o: Vec<Option<Stage>> = Vec::new();
                        let cands = match self.gpu_arms.row_stage_force {
                            Some(f) => vec![f],
                            None => stage_cands(cp),
                        };
                        for (kc, st, p) in cands {
                            let x = stage_of(
                                site,
                                lanes,
                                free,
                                b,
                                matches!(mb.srcs[0], USrc::Gmem(_)),
                                kc,
                                st,
                                p,
                                cp,
                            );
                            if x.is_some() && !o.contains(&x) {
                                o.push(x);
                            }
                        }
                        if o.is_empty() {
                            o.push(None);
                        }
                        o
                    }
                    _ => vec![None],
                })
                .collect();
            let mut combos: Vec<Vec<Option<Stage>>> = vec![Vec::new()];
            for o in &opts {
                combos = combos
                    .into_iter()
                    .flat_map(|c| {
                        o.iter().map(move |x| {
                            let mut c = c.clone();
                            c.push(*x);
                            c
                        })
                    })
                    .take(STAGE_COMBOS)
                    .collect();
            }
            let direct = vec![None; members.len()];
            if !combos.contains(&direct) {
                combos.push(direct);
            }
            let mut staged_fit = false;
            let mut first: Option<Block> = None;
            for stage in combos {
                let any = stage.iter().any(Option::is_some);
                if !any && staged_fit {
                    continue;
                }
                let blk = Block {
                    b,
                    color: color.clone(),
                    ns: (0, 0),
                    stage,
                };
                let probe = Unit {
                    id: 0,
                    rows,
                    members: members.to_vec(),
                    gmem: gmem.to_vec(),
                    block: Some(blk.clone()),
                };
                let geo = self.geo(&probe);
                let acc = members
                    .iter()
                    .filter_map(|mb| match &mb.kind {
                        UKind::Tile { site } => Some(b * tile_split(site, lanes, free).0),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                if geo.smem > gpu.smem_per_block
                    || acc + crate::profile::FMA_ADDR_REGS > gpu.regs_per_thread / 2
                {
                    continue;
                }
                staged_fit |= any;
                first.get_or_insert(blk.clone());
                // The model's waves charge an unfilled grid its busiest SM.
                match self.block_cost(&probe, &geo) {
                    Some((fused, split)) => {
                        why = "cost";
                        if self.block_priced.get().is_none_or(|x| fused < x.1) {
                            self.block_priced.set(Some((b, fused, split)));
                        }
                        if (fused <= split || forced > 0)
                            && best.as_ref().is_none_or(|x| fused < x.1)
                        {
                            best = Some((
                                Block {
                                    ns: (fused, split),
                                    ..blk
                                },
                                fused,
                            ));
                        }
                    }
                    None if why == "budget" => why = "unpriced",
                    None => {}
                }
            }
            // A forced `b` takes its cheapest priced realization, else its
            // first that fits.
            if forced > 0 {
                if best.is_none() {
                    best = first.map(|blk| (blk, 0));
                }
                break;
            }
        }
        best.map(|(blk, _)| blk).ok_or(why)
    }

    /// The plan model's reading of a row-block unit against the launches it
    /// replaces, `(fused, split)` ns, `None` without plan rates. Fused: each
    /// tile as a `b × c` tile of `rows ÷ b` blocks folding all of `k`
    /// ([`TargetProfile::site_ns`]: its FFMA, the column operand every block
    /// re-reads from L2, the latency chain), the members between tiles on
    /// chip, the stored outputs, one launch. Split: each tile site's best
    /// candidate plan, the members between tiles as one launch per segment
    /// reading and writing the intermediates in memory. Every byte stream
    /// moves at the rate of where its value lives (`residence_plan` against
    /// the part's L2: `l2_bytes_per_ns` or `dram_bytes_per_ns`), the same
    /// rule on every part.
    fn block_cost(&self, u: &Unit, geo: &Geo) -> Option<(u64, u64)> {
        let (gpu, d) = self.profile.plan_rates()?;
        if d.flops_per_ns == 0 {
            return None;
        }
        let blk = u.block.as_ref()?;
        let free = self.order == crate::Order::Free;
        let (b, lanes, rows) = (blk.b, geo.lanes, u.rows);
        let blocks = rows / b;
        let regs = 2 * crate::profile::FMA_ADDR_REGS;
        let per_sm = [
            gpu.max_blocks_per_sm,
            gpu.smem_per_sm / (geo.smem + gpu.smem_reserved_per_block).max(1),
            gpu.max_threads_per_sm / lanes,
            gpu.regs_per_sm / (lanes * regs),
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
        .max(1);
        // kernel-merging: where each value's bytes live decides their rate.
        let res = self.ir.residence_plan(self.f);
        let at = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let roof = |f: u64, by_ns: f64| (f as f64 / d.flops_per_ns as f64).max(by_ns);
        let launch = |prev: f64| {
            self.profile
                .op_ns(prev, 0)
                .map_or(d.launch_ns as f64, |x| x.round())
        };
        let seg = segments(&u.members);
        let elem = |mb: &UMember| {
            crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)))
        };
        let (mut fused, mut split) = (launch(f64::MAX), 0f64);
        let mut seg_cost: Vec<(u64, f64)> = vec![(0, 0.0); seg.last().map_or(0, |s| s + 1)];
        for (k, mb) in u.members.iter().enumerate() {
            let e = elem(mb);
            let stored = if mb.store { rows * mb.w * e } else { 0 };
            fused += stored as f64 / at(mb.out);
            match &mb.kind {
                UKind::Tile { site } => {
                    let (nj, slices) = tile_split(site, lanes, free);
                    let kper = site.k / slices;
                    let step = [16, 8, 4, 2, 1]
                        .into_iter()
                        .find(|s| kper % s == 0)
                        .unwrap_or(1);
                    // A block's k-slices are columns of a `b × c·slices`
                    // tile folding `k ÷ slices` (the same FFMA, each thread's
                    // own trips); 4-wide loads where `k` is contiguous.
                    let vec = |ck: u64| if ck == 1 && step % 4 == 0 { 4 } else { 1 };
                    let mut shape = crate::profile::PlanShape {
                        rows,
                        c: site.c,
                        k: site.k,
                        e,
                        bm: b,
                        bn: site.c * slices,
                        threads: lanes,
                        per_sm,
                        acc: b * nj,
                        tiles: blocks,
                        blocks,
                        kb: kper,
                        s: 1,
                        comb: crate::profile::PlanComb::None,
                        bk: step,
                        fly: 1,
                        lsu: b * step / vec(site.a.ck) + nj * step / vec(site.b.ck),
                    };
                    let col = (blocks * site.k * site.c * e) as f64;
                    match blk.stage.get(k).copied().flatten() {
                        // kernel-merging `row-stage`: the separate tile
                        // kernel's terms — a panel trip of `kc`, `stages − 1`
                        // panels in flight, its shared-memory reads (16-byte
                        // chunks along `k`; one element a column across `k`)
                        // and each thread's copies (`LDGSTS`, or `LDG` + `STS`).
                        // A buffer copies, waits and folds in turn: its
                        // latency chain adds to the trip instead of hiding.
                        Some(st) => {
                            let v = 16 / e;
                            let (ca, cb) = st.copies(b, site.c, slices, e, lanes);
                            let a_rd = match &mb.srcs[0] {
                                USrc::Gmem(_) => b * st.kc / v,
                                _ => b * st.kc / vec(site.a.ck),
                            };
                            let njp = nj / st.passes;
                            let b_rd = match st.b_kmaj {
                                true => njp * st.kc / v,
                                false => njp * st.kc,
                            };
                            let copies = (ca + cb) * if st.cp { 1 } else { 2 };
                            // Each column pass a `b × c / passes` tile of its
                            // own (the row operand re-read a pass).
                            shape.c = site.c / st.passes;
                            shape.bn = site.c * slices / st.passes;
                            shape.acc = b * njp;
                            shape.bk = st.kc;
                            shape.fly = (st.stages - 1).max(1);
                            shape.lsu = a_rd + b_rd + copies;
                            let pc = self.profile.site_ns(&shape)?;
                            fused += st.passes as f64
                                * match st.stages {
                                    1 => (pc.f_ns.max(pc.l_ns) + pc.c_ns).max(pc.d_ns),
                                    _ => pc.kernel_ns,
                                };
                            // Each column's run of a trip is `kc` elements:
                            // under a 32-byte sector the rest crosses unused.
                            let run = match (st.b_kmaj, st.b_vec || site.b.clane == 1) {
                                (true, _) => st.kc * e,
                                (false, true) => SECTOR_BYTES,
                                (false, false) => e,
                            };
                            let eff = (run as f64 / SECTOR_BYTES as f64).min(1.0);
                            fused += (1.0 / eff - 1.0) * col / d.l2_bytes_per_ns.max(1) as f64;
                        }
                        None => {
                            fused += self.profile.site_ns(&shape)?.kernel_ns;
                            // The column operand is read straight from `Gmem`
                            // (no shared panel): a warp's lanes are `clane`
                            // elements apart, so unless they are adjacent each
                            // lane's run along `k` (a 4-wide load, else one
                            // element) fills part of each 32-byte sector it
                            // touches. The unused bytes still cross from L2
                            // (rtx-5080 attn_1024_64: the q·Kᵀ phase alone
                            // 20.2 us in the unit against 9.4 us as its tile
                            // site, CUPTI).
                            let run = vec(site.b.ck) * e;
                            let eff = match site.b.clane {
                                1 => 1.0,
                                _ => (run as f64 / SECTOR_BYTES as f64).min(1.0),
                            };
                            fused += (1.0 / eff - 1.0) * col / d.l2_bytes_per_ns.max(1) as f64;
                        }
                    }
                    let l = gpu::launch(site, self.profile, self.contract, self.gpu_arms);
                    let order = if free {
                        crate::Order::Free
                    } else {
                        crate::Order::Sequential
                    };
                    let own = gpu::plan_cands(
                        &l,
                        site,
                        self.profile,
                        self.contract,
                        self.gpu_arms,
                        order,
                    )
                    .first()
                    .map(|c| c.cost.ns)
                    .or_else(|| {
                        let shape = gpu::plan_shape(site, &l, gpu::Combine::None, gpu);
                        self.profile.site_ns(&shape).map(|c| c.ns)
                    })?;
                    split += own;
                }
                _ => {
                    let work = match &mb.kind {
                        UKind::Fold { site, .. } => rows * site.k,
                        _ => rows * mb.w,
                    };
                    let gm: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Gmem(g) => Some((work * e) as f64 / at(u.gmem[*g])),
                            _ => None,
                        })
                        .sum();
                    // Split: every member row it reads crosses memory, and a
                    // row a later segment reads is written there.
                    let crossed: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Member { k: p, .. } if seg[*p] != seg[k] => {
                                let pm = &u.members[*p];
                                Some((rows * pm.w * e) as f64 / at(pm.out))
                            }
                            _ => None,
                        })
                        .sum();
                    let read_later = u.members.iter().enumerate().any(|(j, r)| {
                        seg[j] != seg[k]
                            && r.srcs
                                .iter()
                                .any(|s| matches!(s, USrc::Member { k: p, .. } if *p == k))
                    });
                    let written = if read_later {
                        (rows * mb.w * e) as f64 / at(mb.out)
                    } else {
                        0.0
                    };
                    let x = &mut seg_cost[seg[k]];
                    x.0 += work;
                    x.1 += gm + crossed + written;
                    fused += roof(work, gm);
                }
            }
        }
        let mut prev = f64::MAX;
        for &(f, by) in seg_cost.iter().filter(|x| x.0 > 0) {
            split += launch(prev) + roof(f, by);
            prev = roof(f, by);
        }
        Some((fused.round() as u64, split.round() as u64))
    }

    /// `(flops, bytes)` of the fused launch — every member evaluation, the
    /// recomputed halo and in-row re-reads included — and of each member's
    /// own launch.
    fn unit_roofline(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        elem: impl Fn(ObjectId) -> u64,
    ) -> ((u64, u64), Vec<(u64, u64)>) {
        let steps = |mb: &UMember| match &mb.kind {
            UKind::Fold { site, .. } => rows * site.k,
            UKind::Cell { .. } => rows * mb.w,
            UKind::Tile { site } => rows * site.c * site.k,
        };
        // Evaluations: a fold walks its row once; a per-row cell runs once a
        // row; a cell of `w > 1` is swept when stored and evaluated at every
        // read a register does not serve.
        let mut evals: Vec<u64> = members
            .iter()
            .map(|mb| match mb.kind {
                UKind::Fold { .. } | UKind::Tile { .. } => steps(mb),
                UKind::Cell { .. } if mb.per_row() || mb.store => steps(mb),
                UKind::Cell { .. } => 0,
            })
            .collect();
        for i in (0..members.len()).rev() {
            for s in &members[i].srcs {
                if let USrc::Member { k, row, .. } = s
                    && !(members[*k].per_row() && *row)
                {
                    evals[*k] += evals[i];
                }
            }
        }
        let gmem_reads = |mb: &UMember| {
            mb.srcs
                .iter()
                .filter(|s| matches!(s, USrc::Gmem(_)))
                .count() as u64
        };
        let e = gmem.first().map_or(4, |&o| elem(o));
        let flops: u64 = evals.iter().sum();
        let bytes: u64 = members
            .iter()
            .zip(&evals)
            .map(|(mb, &n)| n * gmem_reads(mb) * e + if mb.store { rows * mb.w * e } else { 0 })
            .sum();
        let own = members
            .iter()
            .map(|mb| {
                let reads = mb
                    .srcs
                    .iter()
                    .filter(|s| matches!(s, USrc::Gmem(_) | USrc::Member { .. }))
                    .count() as u64;
                match &mb.kind {
                    UKind::Fold { site, .. } => (rows * site.k, (rows * site.k + rows) * e),
                    UKind::Cell { .. } | UKind::Tile { .. } => {
                        (steps(mb), (reads + 1) * steps(mb) * e)
                    }
                }
            })
            .collect();
        ((flops, bytes), own)
    }

    /// The host half of a unit: its stored values' buffers, its operands'
    /// device copies (copied in here only when nothing placed one), one
    /// launch, and the copy-out of a stored value the host reads whole.
    pub(super) fn emit_unit(&mut self, u: &Unit) {
        let mut ptrs = Vec::new();
        let mut stored = Vec::new();
        for mb in u.members.iter().filter(|mb| mb.store) {
            let bytes = self.array_bytes(&self.obj_ty(mb.out));
            let d = self.dalloc(mb.out, bytes);
            ptrs.push(d.clone());
            stored.push((mb.out, d));
        }
        let mut temps = Vec::new();
        let Geo {
            gx, block, smem, ..
        } = self.geo(u);
        for &o in &u.gmem {
            let at = self.resident.as_ref().map_or(o, |p| p.alias(o));
            if self.dvalid.contains_key(at) {
                ptrs.push(self.dload(o));
                continue;
            }
            let ob = self.array_bytes(&self.obj_ty(o));
            let h = self.array_operand_ptr(o, None).expect("unit operand ptr");
            let s = format!("%d{}", self.fresh());
            self.allocas
                .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
            let d = self.tmp();
            self.line(format!(
                "{d} = call i64 @mapal_gpu_alloc(ptr {s}, i64 {ob})"
            ));
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {ob})"
            ));
            temps.push(s);
            ptrs.push(d);
        }
        // kernel-merging `zero-fold`: later sites' zero passes as its tail.
        self.zero_args(u.members[0].m, &gx.to_string(), block, &mut ptrs);
        let args = format!("%dv{}", self.fresh());
        let np = ptrs.len();
        self.allocas
            .push_str(&format!("  {args} = alloca [{np} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        self.line(format!(
            "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 {smem}) ; unit",
            u.id as i64
        ));
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            self.dvalid.insert(o, ());
            let keep = self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(o) || p.is_carry_next(o));
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("unit output slot");
                self.d2h(0, &format!("nocapture {h}"), &d, "0", bytes);
            }
        }
    }

    /// The unit's geometry on this face ([`unit_geo`]).
    pub(crate) fn geo(&self, u: &Unit) -> Geo {
        unit_geo(
            u,
            self.profile,
            self.order == crate::Order::Free,
            |mb| match &mb.kind {
                UKind::Fold { site, .. } => crate::profile::elem_bytes(&site.elem),
                UKind::Tile { site } => crate::profile::elem_bytes(&site.elem),
                UKind::Cell { .. } => {
                    crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)))
                }
            },
        )
    }

    /// This function's unit kernels for the device module.
    pub(crate) fn unit_kernels(&self) -> Vec<UnitKernel> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.units
            .values()
            .map(|u| {
                let (fns, text) = self.unit_kernel(u);
                UnitKernel {
                    f: self.f,
                    members: u.members.iter().map(|mb| mb.m).collect(),
                    fns,
                    text,
                }
            })
            .collect()
    }

    /// A body's text as a unit calls it: the host's, or the variant whose
    /// capture `slots` arrive as one-cell buffers (`_u<slots>`).
    fn unit_body(&self, f: FuncId, slots: &[u32]) -> (String, String) {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        if slots.is_empty() {
            return (self.fnames[f].clone(), fe.emit());
        }
        let suffix: Vec<String> = slots.iter().map(u32::to_string).collect();
        fe.cell_suffix = format!("_u{}", suffix.join("_"));
        fe.set_cell_slots(slots.to_vec());
        (format!("{}{}", self.fnames[f], fe.cell_suffix), fe.emit())
    }

    /// The unit's kernel: the bodies it calls, and its text (with its
    /// `; mapal.flow` and `; mapal.plan` intent lines).
    fn unit_kernel(&self, u: &Unit) -> (Vec<(String, String)>, String) {
        let name = gpu::kernel_name(u.id);
        let geo = self.geo(u);
        let (blocks, threads) = (geo.gx, geo.block);
        let lt = |o: ObjectId| array_parts(&self.obj_ty(o)).0;
        let mut fns: Vec<(String, String)> = Vec::new();
        let mut g = UnitGen {
            out: String::new(),
            entry: String::new(),
            n: 0,
            cur: "body".into(),
            vals: vec![None; u.members.len()],
            calls: Vec::new(),
            ys: Vec::new(),
            gmem_ty: u.gmem.iter().map(|&o| lt(o)).collect(),
            lanes: geo.lanes,
            tree: geo.tree.clone(),
            kept: geo.kept.clone(),
            u,
            row: "%i".into(),
            lane: "%l".into(),
            lane_base: "0".into(),
            rloc: None,
            seg: segments(&u.members),
            cur_seg: 0,
            xrow: geo.xrow.clone(),
            part: geo.part,
            panel: geo.panel,
            free: self.order == crate::Order::Free,
            fma: self.contract && self.gpu_arms.pairs,
        };
        // Each member's callee, argument type, result type and buffer types.
        let mut y = 0;
        for mb in &u.members {
            g.ys.push(mb.store.then(|| {
                y += 1;
                y - 1
            }));
            let UKind::Cell { body, captures } = mb.kind else {
                g.calls.push(None);
                continue;
            };
            let morph = self.ir.morphism(mb.m).expect("member");
            let slots: Vec<u32> = (0..captures)
                .filter(|&s| matches!(mb.srcs[s as usize], USrc::Member { .. } | USrc::Law(_)))
                .collect();
            let (callee, text) = self.unit_body(body, &slots);
            if !fns.iter().any(|(n, _)| *n == callee) {
                fns.push((callee.clone(), text));
            }
            let laws = mb.srcs.iter().filter_map(|s| match *s {
                USrc::Law(f) => Some(f),
                _ => None,
            });
            // The laws, then every body nested in the member's or a law's (a
            // natural form's fold inside a cell), as the cell kernels carry
            // them (`resident::nested_bodies`).
            let laws: Vec<FuncId> = laws.collect();
            let nested = std::iter::once(body)
                .chain(laws.iter().copied())
                .flat_map(|b| super::resident::nested_bodies(self.ir, b));
            for f in laws.iter().copied().chain(nested) {
                let (ln, lt) = self.unit_body(f, &[]);
                if !fns.iter().any(|(n, _)| *n == ln) {
                    fns.push((ln, lt));
                }
            }
            let in_ty = self.obj_ty(self.ir.func(body).expect("body").input);
            let arg = lower_body_input_ty(&in_ty, captures).expect("a member's input lowers");
            let field_ty: Vec<String> = (0..=captures)
                .map(|s| {
                    let arr = if captures == 0 {
                        morph.source
                    } else {
                        self.pair_source(morph.source, s).expect("slot")
                    };
                    lt(arr)
                })
                .collect();
            let laws: Vec<(u32, String)> = mb
                .srcs
                .iter()
                .enumerate()
                .filter_map(|(s, x)| match *x {
                    USrc::Law(f) => Some((s as u32, self.fnames[f].clone())),
                    _ => None,
                })
                .collect();
            g.calls.push(Some(CallInfo {
                callee,
                arg,
                ret: lt(mb.out),
                field_ty,
                captures,
                slots,
                laws,
            }));
        }
        // Parameters: the stored values, then the operands.
        let mut params: Vec<String> = Vec::new();
        for (k, mb) in u.members.iter().enumerate() {
            if let Some(j) = g.ys[k] {
                params.push(format!("ptr addrspace(1) noalias %y{j}"));
                let _ = mb;
            }
        }
        for i in 0..u.gmem.len() {
            params.push(format!("ptr addrspace(1) readonly %x{i}"));
            g.entry.push_str(&format!(
                "  %xg{i} = addrspacecast ptr addrspace(1) %x{i} to ptr\n"
            ));
        }
        // kernel-merging `zero-fold`: later sites' outputs this unit zeroes.
        let zero = self
            .resident
            .as_ref()
            .and_then(|p| p.zero_by.get(u.members[0].m))
            .map_or(0, Vec::len);
        params.extend(
            super::zero::zero_params(zero)
                .into_iter()
                .skip(usize::from(zero == 0)),
        );
        for (k, c) in g.calls.iter().enumerate() {
            if let Some(c) = c {
                for &s in &c.slots {
                    g.entry.push_str(&format!(
                        "  %b{k}_{s} = alloca {}\n",
                        c.field_ty[s as usize]
                    ));
                }
            }
        }
        // The phases: one row (or one thread's row) as before; a row-block
        // unit's tiles over all its rows, the members between them row by row.
        match &u.block {
            None => {
                for k in 0..u.members.len() {
                    g.phase(k);
                }
            }
            Some(blk) => {
                let mut k = 0;
                while k < u.members.len() {
                    let s = g.seg[k];
                    let end = (k..u.members.len())
                        .find(|&j| g.seg[j] != s)
                        .unwrap_or(u.members.len());
                    if matches!(u.members[k].kind, UKind::Tile { .. }) {
                        g.cur_seg = s;
                        g.tile_phase(k, blk.b);
                    } else {
                        g.rows_loop(s, blk.b, k..end);
                    }
                    k = end;
                }
            }
        }
        // The intent line (design §7): what the plan keeps, stores and reads.
        let label = |o: ObjectId| self.object_label(o);
        let kind = |mb: &UMember| match mb.kind {
            UKind::Fold { .. } => "rowfold",
            UKind::Cell { .. } => "cell",
            UKind::Tile { .. } => "tile",
        };
        let members: Vec<String> = u
            .members
            .iter()
            .map(|mb| {
                let halo = mb
                    .srcs
                    .iter()
                    .any(|s| matches!(s, USrc::Member { row: false, .. }));
                format!(
                    "{}:{}:1x{}{}",
                    label(mb.out),
                    kind(mb),
                    mb.w,
                    if halo { "+halo" } else { "" }
                )
            })
            .collect();
        let keep: Vec<String> = u
            .members
            .iter()
            .filter(|mb| !mb.store)
            .map(|mb| {
                let e =
                    crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)));
                format!("{}:reg:{}", label(mb.out), if mb.per_row() { e } else { 0 })
            })
            .collect();
        let store: Vec<String> = u
            .members
            .iter()
            .filter(|mb| mb.store)
            .map(|mb| label(mb.out))
            .collect();
        let gm: Vec<String> = u.gmem.iter().map(|&o| label(o)).collect();
        let laws: Vec<String> = u
            .members
            .iter()
            .flat_map(|mb| mb.srcs.iter())
            .filter_map(|s| match *s {
                USrc::Law(f) => Some(self.fnames[f].clone()),
                _ => None,
            })
            .collect();
        let root = u
            .members
            .iter()
            .find(|mb| matches!(mb.kind, UKind::Fold { .. }))
            .unwrap_or(&u.members[0]);
        let w = u.members.iter().map(|mb| mb.w).max().unwrap_or(1);
        let block = match &u.block {
            None => String::new(),
            Some(blk) => {
                // kernel-merging `row-stage`: each staged tile's panels.
                let stage: Vec<String> = blk
                    .stage
                    .iter()
                    .enumerate()
                    .filter_map(|(k, st)| st.map(|st| format!("T{k}:{}", st.label())))
                    .collect();
                format!(
                    " rows_per_unit={} tiles={} colors={} ns_fused={} ns_split={}{}",
                    blk.b,
                    u.members
                        .iter()
                        .filter(|mb| matches!(mb.kind, UKind::Tile { .. }))
                        .count(),
                    blk.color.iter().flatten().max().map_or(0, |c| c + 1),
                    blk.ns.0,
                    blk.ns.1,
                    match stage.is_empty() {
                        true => String::new(),
                        false => format!(" stage={}", stage.join(";")),
                    }
                )
            }
        };
        let mut text = format!(
            "; mapal.flow kernel={name} unit=1x{w} rows={} root={} members={} keep={} store={} gmem={} law={} smem={} lanes={} tree={} onchip={}{block}\n",
            u.rows,
            label(root.out),
            members.join(";"),
            keep.join(";"),
            store.join(";"),
            gm.join(";"),
            laws.join(";"),
            geo.smem,
            geo.lanes,
            geo.tree.iter().flatten().count(),
            geo.kept.iter().flatten().count(),
        );
        text += &format!(
            "; mapal.plan kernel={name} kind=flow rows={} gx={blocks} block={threads} members={} rungs_on=flow{}\n",
            u.rows,
            u.members.len(),
            if u.block.is_some() { ",row-block" } else { "" }
        );
        text += &format!(
            "define ptx_kernel void @{name}({}) \"nvvm.maxntid\"=\"{threads}\" {{\n\
             entry:\n{}\
             \x20 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             \x20 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             \x20 %t64 = zext i32 %tid to i64\n\
             \x20 %b64 = zext i32 %bid to i64\n\
             {}\
             \x20 %in = icmp ult i64 %i, {}\n\
             \x20 br i1 %in, label %body, label %done\n\
             body:\n{}\
             \x20 br label %done\n\
             done:\n{}\
             \x20 ret void\n\
             }}\n",
            params.join(", "),
            g.entry,
            // One row a block (`lanes` threads), or one thread a row; a
            // row-block unit's first row.
            if let Some(blk) = &u.block {
                format!("  %i = mul i64 %b64, {}\n  %l = add i64 %t64, 0\n", blk.b)
            } else if geo.lanes > 1 {
                "  %i = add i64 %b64, 0\n  %l = add i64 %t64, 0\n".to_string()
            } else {
                format!(
                    "  %row0 = mul i64 %b64, {threads}\n  %i = add i64 %row0, %t64\n  %l = add i64 0, 0\n"
                )
            },
            u.rows,
            g.out,
            super::zero::zero_tail(zero, threads),
        );
        (fns, text)
    }
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
}

impl UnitGen<'_> {
    fn tmp(&mut self) -> String {
        self.n += 1;
        format!("%v{}", self.n)
    }

    fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    fn label(&mut self, l: &str) {
        self.out.push_str(l);
        self.out.push_str(":\n");
        self.cur = l.to_string();
    }

    fn op(&mut self, op: &str, a: &str, b: impl std::fmt::Display) -> String {
        let r = self.tmp();
        self.line(format!("{r} = {op} i64 {a}, {b}"));
        r
    }

    /// The address `reach` gives cell `t` (with the counted axes `taps`).
    fn addr(&mut self, reach: &Reach, t: &str, taps: &[&str]) -> String {
        let Reach::Affine {
            w,
            base,
            cq,
            cr,
            taps: tt,
        } = reach
        else {
            unreachable!("a unit reads members at affine reaches")
        };
        let (q, r) = if *w == 1 {
            (t.to_string(), None)
        } else {
            (self.op("udiv", t, w), Some(self.op("urem", t, w)))
        };
        let mut acc = self.op("mul", &q, cq);
        if let Some(r) = r {
            let rr = self.op("mul", &r, cr);
            acc = self.op("add", &acc, &rr);
        }
        for (tap, k) in tt.iter().zip(taps) {
            let kk = self.op("mul", k, tap.coef);
            acc = self.op("add", &acc, &kk);
        }
        self.op("add", &acc, base)
    }

    fn i32_of(&mut self, t: &str) -> String {
        let x = self.tmp();
        self.line(format!("{x} = trunc i64 {t} to i32"));
        x
    }

    fn load(&mut self, g: usize, at: &str) -> String {
        let ty = self.gmem_ty[g].clone();
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace(1) %x{g}, i64 {at}"
        ));
        let v = self.tmp();
        self.line(format!("{v} = load {ty}, ptr addrspace(1) {p}"));
        v
    }

    /// Source `s` of a reader at its cell `t` (and fold axes `taps`).
    fn read(&mut self, s: &USrc, t: &str, taps: &[&str]) -> String {
        match s {
            USrc::Member { k, reach, row } => {
                if self.u.members[*k].per_row() && *row {
                    // A row-block unit's per-row value from an earlier segment.
                    if let (Some(off), Some(rl)) =
                        (self.xrow.get(*k).copied().flatten(), self.rloc.clone())
                        && self.seg[*k] != self.cur_seg
                    {
                        let ty = self.ret_ty(*k);
                        let p = self.smem(off, &ty, &rl);
                        let v = self.tmp();
                        self.line(format!("{v} = load {ty}, ptr addrspace(3) {p}"));
                        return v;
                    }
                    return self.vals[*k]
                        .clone()
                        .expect("a per-row value is read after its phase");
                }
                let a = self.addr(reach, t, taps);
                if let (true, Some(off)) = (*row, self.kept[*k]) {
                    let w = self.u.members[*k].w;
                    let row = self.row.clone();
                    let r = self.op("mul", &row, w);
                    let mut j = self.op("sub", &a, &r);
                    if let Some(rl) = self.rloc.clone() {
                        let base = self.op("mul", &rl, w);
                        j = self.op("add", &j, &base);
                    }
                    let ty = self.ret_ty(*k);
                    let p = self.smem(off, &ty, &j);
                    let v = self.tmp();
                    self.line(format!("{v} = load {ty}, ptr addrspace(3) {p}"));
                    return v;
                }
                self.cell(*k, &a)
            }
            USrc::Gmem(g) => self.load(*g, t),
            USrc::Index => self.i32_of(t),
            USrc::Law(_) => unreachable!("a law feeds a capture slot"),
        }
    }

    /// Cell member `k`'s value at its cell `t`.
    fn cell(&mut self, k: usize, t: &str) -> String {
        let u = self.u;
        let mb = &u.members[k];
        let c = self.calls[k].as_ref().expect("a cell member");
        let (callee, arg, ret, captures) =
            (c.callee.clone(), c.arg.clone(), c.ret.clone(), c.captures);
        let field_ty = c.field_ty.clone();
        let laws = c.laws.clone();
        let mut fields: Vec<(String, String)> = Vec::new();
        for (s, src) in mb.srcs.iter().enumerate() {
            let s32 = s as u32;
            if s32 < captures {
                let v = match src {
                    USrc::Gmem(g) => {
                        fields.push(("ptr".into(), format!("%xg{g}")));
                        continue;
                    }
                    USrc::Law(_) => {
                        let law = &laws.iter().find(|(x, _)| *x == s32).expect("law").1;
                        let ti = self.i32_of(t);
                        let v = self.tmp();
                        self.line(format!("{v} = call {} @{law}(i32 {ti})", field_ty[s]));
                        v
                    }
                    other => self.read(other, t, &[]),
                };
                self.line(format!("store {} {v}, ptr %b{k}_{s}", field_ty[s]));
                fields.push(("ptr".into(), format!("%b{k}_{s}")));
            } else {
                let v = self.read(src, t, &[]);
                let ty = if matches!(src, USrc::Index) {
                    "i32".to_string()
                } else {
                    field_ty[s].clone()
                };
                fields.push((ty, v));
            }
        }
        let a = if captures == 0 {
            fields[0].1.clone()
        } else {
            let mut agg = "poison".to_string();
            for (f, (ty, v)) in fields.iter().enumerate() {
                let n = self.tmp();
                self.line(format!("{n} = insertvalue {arg} {agg}, {ty} {v}, {f}"));
                agg = n;
            }
            agg
        };
        let r = self.tmp();
        self.line(format!("{r} = call {ret} @{callee}({arg} {a})"));
        r
    }

    fn store(&mut self, k: usize, v: &str, at: &str) {
        if let Some(j) = self.ys[k] {
            let ty = match &self.calls[k] {
                Some(c) => c.ret.clone(),
                None => match &self.u.members[k].kind {
                    UKind::Fold { site, .. } => lower_ty(&site.elem).expect("fold elem"),
                    UKind::Tile { site } => lower_ty(&site.elem).expect("tile elem"),
                    UKind::Cell { .. } => unreachable!("a cell has a call"),
                },
            };
            let p = self.tmp();
            self.line(format!(
                "{p} = getelementptr inbounds {ty}, ptr addrspace(1) %y{j}, i64 {at}"
            ));
            self.line(format!("store {ty} {v}, ptr addrspace(1) {p}"));
        }
    }

    /// `out = acc ⊕ v`: the fold's own step, in the source's operand order.
    fn step(&mut self, op: &RowFoldOp, t: &str, acc: &str, v: &str, out: &str) {
        match *op {
            RowFoldOp::Arith { op, acc_first } => {
                let code = match op {
                    Operation::Add => "fadd",
                    Operation::Sub => "fsub",
                    _ => "fmul",
                };
                let (x, y) = if acc_first { (acc, v) } else { (v, acc) };
                self.line(format!("{out} = {code} {t} {x}, {y}"));
            }
            RowFoldOp::Select {
                cmp,
                acc_lhs,
                acc_if_true,
            } => {
                let pred = match cmp {
                    Operation::Lt => "olt",
                    Operation::Gt => "ogt",
                    Operation::Le => "ole",
                    _ => "oge",
                };
                let (x, y) = if acc_lhs { (acc, v) } else { (v, acc) };
                let (p, q) = if acc_if_true { (acc, v) } else { (v, acc) };
                let c = self.tmp();
                self.line(format!("{c} = fcmp {pred} {t} {x}, {y}"));
                self.line(format!("{out} = select i1 {c}, {t} {p}, {t} {q}"));
            }
        }
    }

    /// A pointer to element `idx` (of `ty`) at byte `off` of the launch's
    /// shared memory.
    fn smem(&mut self, off: u64, ty: &str, idx: &str) -> String {
        let q = self.tmp();
        self.line(format!(
            "{q} = getelementptr inbounds i8, ptr addrspace(3) @mapal_smem, i64 {off}"
        ));
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {ty}, ptr addrspace(3) {q}, i64 {idx}"
        ));
        p
    }

    fn barrier(&mut self) {
        self.line("call void @llvm.nvvm.barrier0()");
    }

    /// `body` by lane 0 of the row alone (every lane holds the value).
    fn lane0(&mut self, tag: &str, body: impl FnOnce(&mut Self)) {
        if self.lanes == 1 {
            return body(self);
        }
        let c = self.tmp();
        self.line(format!("{c} = icmp eq i64 {}, 0", self.lane));
        let (yes, end) = (format!("{tag}z"), format!("{tag}ze"));
        self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
        self.label(&yes);
        body(self);
        self.line(format!("br label %{end}"));
        self.label(&end);
    }

    /// The counted loop `%<tag>x = start, start + stride, … < n` around one
    /// block `body(x, acc)`, entered only when `start < n`. With `carry`
    /// (type, init) the body defines `<acc>n`, the next value; the loop's exit
    /// value is returned.
    fn counted(
        &mut self,
        tag: &str,
        (start, stride, n): (&str, u64, u64),
        carry: Option<(&str, &str)>,
        body: &mut dyn FnMut(&mut Self, &str, &str),
    ) -> Option<String> {
        let pre = self.cur.clone();
        let (lp, end) = (tag.to_string(), format!("{tag}e"));
        let c0 = self.tmp();
        self.line(format!("{c0} = icmp ult i64 {start}, {n}"));
        self.line(format!("br i1 {c0}, label %{lp}, label %{end}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [{start}, %{pre}], [{x}n, %{lp}]"));
        let a = format!("%{tag}a");
        if let Some((t, init)) = carry {
            self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
        }
        body(self, &x, &a);
        assert_eq!(self.cur, lp, "a counted body is one block");
        self.line(format!("{x}n = add i64 {x}, {stride}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {n}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{end}"));
        self.label(&end);
        carry.map(|(t, init)| {
            let r = self.tmp();
            self.line(format!("{r} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
            r
        })
    }

    /// Member `k`'s phase in row `%i`, lane `%l` of [`Geo`]'s `lanes`.
    fn phase(&mut self, k: usize) {
        let u = self.u;
        let mb = &u.members[k];
        let lanes = self.lanes;
        match &mb.kind {
            UKind::Fold { site, .. } => {
                let t = lower_ty(&site.elem).expect("float elem lowers");
                let seed = const_literal(&site.seed);
                let tree = self.tree[k];
                // A tree's lanes start from the step's identity (a max/min's
                // seed: that step is idempotent); an in-order fold from the seed.
                let init = match (tree, site.op) {
                    (None, _) | (_, RowFoldOp::Select { .. }) => seed.clone(),
                    (
                        _,
                        RowFoldOp::Arith {
                            op: Operation::Mul, ..
                        },
                    ) => "1.0".to_string(),
                    _ => "-0.0".to_string(),
                };
                let lane = self.lane.clone();
                let walk = if tree.is_some() {
                    (lane.as_str(), lanes, site.k)
                } else {
                    ("0", 1, site.k)
                };
                let (site, src0, op) = (site.clone(), mb.srcs[0].clone(), site.op);
                let row = self.row.clone();
                let tt = t.clone();
                let p = self
                    .counted(
                        &format!("f{k}"),
                        walk,
                        Some((&t, &init)),
                        &mut |g, kv, acc| {
                            // The fold's axes: `k`, or a window's `(k÷div, k%div)`.
                            let taps: Vec<String> = match site.a.ksplit {
                                Some(ks) => {
                                    vec![g.op("udiv", kv, ks.div), g.op("urem", kv, ks.div)]
                                }
                                None => vec![kv.to_string()],
                            };
                            let v = match &src0 {
                                USrc::Gmem(gm) => {
                                    // The site's own addressing (re-addressed when its
                                    // producer stores column-major).
                                    let ri = if site.w > 1 {
                                        let q = g.op("udiv", &row, site.w);
                                        let l = g.op("urem", &row, site.w);
                                        let a = g.op("mul", &q, site.a.ci);
                                        let b = g.op("mul", &l, site.a.clane);
                                        g.op("add", &a, &b)
                                    } else {
                                        g.op("mul", &row, site.a.ci)
                                    };
                                    let rk = match site.a.ksplit {
                                        Some(ks) => {
                                            let a = g.op("mul", &taps[0], ks.cq);
                                            let b = g.op("mul", &taps[1], ks.cr);
                                            g.op("add", &a, &b)
                                        }
                                        None => g.op("mul", kv, site.a.ck),
                                    };
                                    let x = g.op("add", &ri, &rk);
                                    let x = g.op("add", &x, site.a.base);
                                    g.load(*gm, &x)
                                }
                                src => {
                                    let taps: Vec<&str> = taps.iter().map(String::as_str).collect();
                                    g.read(src, &row, &taps)
                                }
                            };
                            g.step(&op, &tt, acc, &v, &format!("{acc}n"));
                        },
                    )
                    .expect("a fold carries");
                let res = match tree {
                    None => p,
                    Some(off) => {
                        // The lanes' partials in shared memory, halved per
                        // step, then the seed once: every lane reads one total.
                        let mine = self.smem(off, &t, "%l");
                        self.line(format!("store {t} {p}, ptr addrspace(3) {mine}"));
                        self.barrier();
                        let mut s = lanes / 2;
                        while s >= 1 {
                            let c = self.tmp();
                            self.line(format!("{c} = icmp ult i64 {}, {s}", self.lane));
                            let (yes, end) = (format!("r{k}_{s}"), format!("r{k}_{s}e"));
                            self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
                            self.label(&yes);
                            let o = self.op("add", "%l", s);
                            let po = self.smem(off, &t, &o);
                            let (x, y, z) = (self.tmp(), self.tmp(), self.tmp());
                            self.line(format!("{x} = load {t}, ptr addrspace(3) {mine}"));
                            self.line(format!("{y} = load {t}, ptr addrspace(3) {po}"));
                            self.step(&op, &t, &x, &y, &z);
                            self.line(format!("store {t} {z}, ptr addrspace(3) {mine}"));
                            self.line(format!("br label %{end}"));
                            self.label(&end);
                            self.barrier();
                            s /= 2;
                        }
                        let base = self.lane_base.clone();
                        let p0 = self.smem(off, &t, &base);
                        let (tot, r) = (self.tmp(), self.tmp());
                        self.line(format!("{tot} = load {t}, ptr addrspace(3) {p0}"));
                        self.step(&op, &t, &seed, &tot, &r);
                        r
                    }
                };
                self.lane0(&format!("f{k}s"), |g| g.store(k, &res, &row));
                self.keep_row_value(k, &res);
                self.vals[k] = Some(res);
            }
            UKind::Cell { .. } if mb.per_row() => {
                let row = self.row.clone();
                let v = self.cell(k, &row);
                self.lane0(&format!("c{k}s"), |g| g.store(k, &v, &row));
                self.keep_row_value(k, &v);
                self.vals[k] = Some(v);
            }
            UKind::Cell { .. } if mb.store || self.kept[k].is_some() => {
                let (w, kept) = (mb.w, self.kept[k]);
                let (row, rloc, lane) = (self.row.clone(), self.rloc.clone(), self.lane.clone());
                self.counted(&format!("s{k}"), (&lane, lanes, w), None, &mut |g, j, _| {
                    let r = g.op("mul", &row, w);
                    let t = g.op("add", &r, j);
                    let v = g.cell(k, &t);
                    g.store(k, &v, &t);
                    if let Some(off) = kept {
                        let ty = g.calls[k].as_ref().expect("a cell").ret.clone();
                        let idx = match &rloc {
                            Some(rl) => {
                                let base = g.op("mul", rl, w);
                                g.op("add", &base, j)
                            }
                            None => j.to_string(),
                        };
                        let p = g.smem(off, &ty, &idx);
                        g.line(format!("store {ty} {v}, ptr addrspace(3) {p}"));
                    }
                });
                if kept.is_some() {
                    self.barrier();
                }
            }
            // Evaluated at its reads.
            UKind::Cell { .. } => {}
            UKind::Tile { .. } => unreachable!("a tile runs as its own phase"),
        }
    }

    /// compile-time-graph A: tile member `k` over the block's `b` rows. A
    /// thread holds `b × nj` accumulators — its columns `j` ([`tile_split`])
    /// of every row — over its k-slice, `step` k a trip; the row operand `a`
    /// comes from the producer's on-chip rows or `Gmem` (one address a warp:
    /// a broadcast), the column operand `b` from `Gmem`, each as 4-wide
    /// loads where the read is contiguous in `k` and aligned. Several slices
    /// combine through shared memory in slice order (the free order), the
    /// seed first. The results go to the member's on-chip rows and, when
    /// stored, to its buffer.
    fn tile_phase(&mut self, k: usize, b: u64) {
        let u = self.u;
        if let Some(st) = u.block.as_ref().and_then(|bl| bl.stage[k]) {
            return self.tile_phase_staged(k, b, st);
        }
        let mb = &u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a tile phase runs a tile member")
        };
        let site = site.clone();
        let (a_src, b_src) = (mb.srcs[0].clone(), mb.srcs[1].clone());
        let t = lower_ty(&site.elem).expect("tile elem");
        let e = crate::profile::elem_bytes(&site.elem);
        let lanes = self.lanes;
        let (nj, slices) = tile_split(&site, lanes, self.free);
        let c = site.c;
        let kper = site.k / slices;
        let step = [16u64, 8, 4, 2, 1]
            .into_iter()
            .find(|x| kper % x == 0)
            .unwrap_or(1);
        let tag = format!("T{k}");
        // Threads past the row's columns × slices idle (a narrow row whose
        // `k` the order may not split).
        let active = c * slices < lanes;
        let (act, act_end) = (format!("{tag}act"), format!("{tag}acte"));
        if active {
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {}", c * slices));
            self.line(format!("br i1 {a}, label %{act}, label %{act_end}"));
            self.label(&act);
        }
        let (cols, k0) = if c >= lanes {
            let cols: Vec<String> = (0..nj).map(|q| self.op("add", "%l", q * lanes)).collect();
            (cols, "0".to_string())
        } else {
            let j = self.op("urem", "%l", c);
            let sl = self.op("udiv", "%l", c);
            let k0 = self.op("mul", &sl, kper);
            (vec![j], k0)
        };
        let k1 = self.op("add", &k0, kper);
        // Per-row and per-column address bases.
        let (va, a_rows): (u64, Vec<(bool, String)>) = match &a_src {
            USrc::Gmem(g) => {
                let v = if e == 4
                    && site.a.ck == 1
                    && step % 4 == 0
                    && site.a.base % 4 == 0
                    && site.a.ci % 4 == 0
                {
                    4
                } else {
                    1
                };
                let rows = (0..b)
                    .map(|r| {
                        let i = self.op("add", "%i", r);
                        let x = self.op("mul", &i, site.a.ci);
                        (false, self.op("add", &x, site.a.base) + &format!("@{g}"))
                    })
                    .collect();
                (v, rows)
            }
            USrc::Member { k: p, .. } => {
                let (off, wp) = (
                    self.kept[*p].expect("a tile's row operand is on chip"),
                    u.members[*p].w,
                );
                let v = if e == 4
                    && site.a.ck == 1
                    && step % 4 == 0
                    && site.a.base % 4 == 0
                    && wp % 4 == 0
                    && off % 16 == 0
                {
                    4
                } else {
                    1
                };
                let rows = (0..b)
                    .map(|r| (true, format!("{}@{off}", r * wp + site.a.base)))
                    .collect();
                (v, rows)
            }
            _ => unreachable!("a tile's row operand is a member or Gmem"),
        };
        let USrc::Gmem(bg) = b_src else {
            unreachable!("a tile's column operand is Gmem")
        };
        let vb = if e == 4
            && site.b.ck == 1
            && step % 4 == 0
            && site.b.base % 4 == 0
            && site.b.clane % 4 == 0
        {
            4
        } else {
            1
        };
        let b_cols: Vec<String> = cols
            .iter()
            .map(|j| {
                let x = self.op("mul", j, site.b.clane);
                self.op("add", &x, site.b.base)
            })
            .collect();
        let init = if slices > 1 {
            "-0.0".to_string()
        } else {
            const_literal(&site.seed)
        };
        let pre = self.cur.clone();
        let lp = tag.clone();
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [{k0}, %{pre}], [{x}n, %{lp}]"));
        let acc_name = |r: u64, q: u64| format!("%{tag}a{r}_{q}");
        for r in 0..b {
            for q in 0..nj {
                let a = acc_name(r, q);
                self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{lp}]"));
            }
        }
        let mut cur: Vec<Vec<String>> = (0..b)
            .map(|r| (0..nj).map(|q| acc_name(r, q)).collect())
            .collect();
        let chunk = step.min(4);
        let mut u0 = 0;
        while u0 < step {
            let kx = self.op("add", &x, u0);
            // The chunk's row operands, `chunk` per row.
            let mut av: Vec<Vec<String>> = Vec::new();
            for (shared, base) in &a_rows {
                let (base, at) = base.split_once('@').expect("a base and its buffer");
                let (base, at) = (base.to_string(), at.to_string());
                let ck = site.a.ck;
                let idx = if *shared {
                    let x = self.op("mul", &kx, ck);
                    self.op("add", &x, &base)
                } else {
                    let x = self.op("mul", &kx, ck);
                    self.op("add", &x, &base)
                };
                av.push(self.tile_loads(*shared, &at, &t, &idx, chunk, va, ck));
            }
            let mut bv: Vec<Vec<String>> = Vec::new();
            for base in &b_cols {
                let x = self.op("mul", &kx, site.b.ck);
                let idx = self.op("add", &x, base);
                bv.push(self.tile_loads(false, &bg.to_string(), &t, &idx, chunk, vb, site.b.ck));
            }
            for uu in 0..chunk as usize {
                let last = u0 + chunk >= step && uu + 1 == chunk as usize;
                for r in 0..b as usize {
                    for q in 0..nj as usize {
                        let out = if last {
                            format!("{}n", acc_name(r as u64, q as u64))
                        } else {
                            self.tmp()
                        };
                        let (av_, bv_) = (av[r][uu].clone(), bv[q][uu].clone());
                        let acc = cur[r][q].clone();
                        self.tile_step(&site, &t, &acc, &av_, &bv_, &out);
                        cur[r][q] = out;
                    }
                }
            }
            u0 += chunk;
        }
        self.line(format!("{x}n = add i64 {x}, {step}"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {k1}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{lp}e"));
        self.label(&format!("{lp}e"));
        self.tile_tail(
            k,
            &site,
            &t,
            b,
            (nj, slices),
            &cols,
            cur,
            (active, &act_end),
            &tag,
            false,
        );
    }

    /// kernel-merging `row-stage`: member `k`'s tile phase over panels staged
    /// through shared memory ([`Stage`]). Every thread copies (a thread past
    /// the row's columns × slices folds a working thread's cells, its reads
    /// inside the panel, and stores nothing); each trip waits for its panel,
    /// meets the block, and — with two buffers — issues the next panel into
    /// the buffer the trip before read, then folds `kc` of `k` per slice from
    /// shared memory in the order the direct phase does.
    fn tile_phase_staged(&mut self, k: usize, b: u64, st: Stage) {
        let u = self.u;
        let mb = &u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a tile phase runs a tile member")
        };
        let site = site.clone();
        let a_src = mb.srcs[0].clone();
        let USrc::Gmem(bg) = mb.srcs[1] else {
            unreachable!("a tile's column operand is Gmem")
        };
        let lanes = self.lanes;
        let (nj, slices) = tile_split(&site, lanes, self.free);
        let c = site.c;
        let tag = format!("T{k}");
        let active = c * slices < lanes;
        // The thread's cells: its columns and slice.
        let lc = match active {
            true => self.op("urem", "%l", c * slices),
            false => "%l".to_string(),
        };
        let (all, sl) = if c >= lanes {
            let cols: Vec<String> = (0..nj).map(|q| self.op("add", &lc, q * lanes)).collect();
            (cols, "0".to_string())
        } else {
            (vec![self.op("urem", &lc, c)], self.op("udiv", &lc, c))
        };
        // Column passes: pass `p` folds the thread's columns `p·njp …` — the
        // row's contiguous `cw = c / passes` from `p·cw`.
        let (njp, cw) = (nj / st.passes, c / st.passes);
        for pass in 0..st.passes {
            let cols: Vec<String> =
                all[(pass * njp) as usize..((pass + 1) * njp) as usize].to_vec();
            let tag = match st.passes {
                1 => tag.clone(),
                _ => format!("{tag}p{pass}"),
            };
            let j0 = pass * cw;
            self.tile_pass_staged(
                k,
                b,
                st,
                (&site, &a_src, bg),
                (&cols, &sl, &lc),
                (njp, slices, cw, j0),
                (&tag, active),
            );
        }
    }

    /// One column pass of [`Self::tile_phase_staged`]: `cols` (the thread's
    /// columns of the pass), its panels `cw` columns wide from column `j0`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn tile_pass_staged(
        &mut self,
        k: usize,
        b: u64,
        st: Stage,
        (site, a_src, bg): (&TileSite, &USrc, usize),
        (cols, sl, _lc): (&[String], &str, &str),
        (nj, slices, cw, j0): (u64, u64, u64, u64),
        (tag, active): (&str, bool),
    ) {
        let u = self.u;
        let t = lower_ty(&site.elem).expect("tile elem");
        let e = crate::profile::elem_bytes(&site.elem);
        let v = 16 / e;
        let lanes = self.lanes;
        let (c, kc) = (site.c, st.kc);
        let kper = site.k / slices;
        let trips = kper / kc;
        let base = self.panel.expect("a staged tile has panel buffers");
        let (a_bytes, stage_bytes) = st.bytes(b, c, slices, e);
        let (a_el, buf_el) = (a_bytes / e, stage_bytes / e);
        let tag = tag.to_string();
        // Per column: its panel row and that row's chunk swizzle.
        let nch = kc / v;
        let swz = |g: &mut Self, r: &str| -> String {
            if nch == 1 {
                return "0".to_string();
            }
            let sh = (8 / nch.min(8)).max(1).trailing_zeros();
            let x = g.op("lshr", r, sh);
            g.op("and", &x, nch.min(8) - 1)
        };
        let brow: Vec<(String, String)> = cols
            .iter()
            .map(|j| {
                let x = self.op("mul", sl, cw);
                let jl = self.op("sub", j, j0);
                let r = self.op("add", &x, &jl);
                let w = swz(self, &r);
                (self.op("mul", &r, kc), w)
            })
            .collect();
        let arow = self.op("mul", sl, b);
        let ty_v = format!("<{v} x {t}>");
        let s_ptr = |g: &mut Self, idx: &str| g.smem(base, &t, idx);
        let g_ptr = |g: &mut Self, at: &str, idx: &str| {
            let p = g.tmp();
            g.line(format!(
                "{p} = getelementptr inbounds {t}, ptr addrspace(1) %x{at}, i64 {idx}"
            ));
            p
        };
        // One panel's copies: trip `tr` into the buffer at element `bo`.
        let issue = |g: &mut Self, tr: &str, bo: &str, n: &mut u32| {
            let k0 = g.op("mul", tr, kc);
            let mut one =
                |g: &mut Self,
                 total: u64,
                 unit: u64,
                 f: &mut dyn FnMut(&mut Self, &str) -> (String, String)| {
                    let copies = total / unit;
                    for m in 0..copies.div_ceil(lanes) {
                        let x = g.op("add", "%l", m * lanes);
                        let part = (m + 1) * lanes > copies;
                        *n += 1;
                        let (yes, end) = (format!("{tag}c{n}"), format!("{tag}c{n}e"));
                        if part {
                            let ok = g.tmp();
                            g.line(format!("{ok} = icmp ult i64 {x}, {copies}"));
                            g.line(format!("br i1 {ok}, label %{yes}, label %{end}"));
                            g.label(&yes);
                        }
                        let el = g.op("mul", &x, unit);
                        let (gp, sx) = f(g, &el);
                        let sx = g.op("add", &sx, bo);
                        let sp = s_ptr(g, &sx);
                        let bytes = unit * e;
                        if st.cp {
                            let kind = if bytes == 16 { "cg" } else { "ca" };
                            g.line(format!(
                            "call void @llvm.nvvm.cp.async.{kind}.shared.global.{bytes}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                        ));
                        } else {
                            let (lt, al) = match unit {
                                1 => (t.clone(), e),
                                _ => (ty_v.clone(), 16),
                            };
                            let x = g.tmp();
                            g.line(format!(
                                "{x} = load {lt}, ptr addrspace(1) {gp}, align {al}"
                            ));
                            g.line(format!("store {lt} {x}, ptr addrspace(3) {sp}, align {al}"));
                        }
                        if part {
                            g.line(format!("br label %{end}"));
                            g.label(&end);
                        }
                    }
                };
            // The row operand: `[slice][row][kc]`.
            if st.a {
                let USrc::Gmem(ag) = a_src else {
                    unreachable!("a staged row operand is Gmem")
                };
                let unit = if st.a_vec { v } else { 1 };
                one(g, slices * b * kc, unit, &mut |g, el| {
                    let r_ = g.op("udiv", el, kc);
                    let kk = g.op("urem", el, kc);
                    let s_ = g.op("udiv", &r_, b);
                    let r = g.op("urem", &r_, b);
                    let i = g.op("add", "%i", &r);
                    let x = g.op("mul", &i, site.a.ci);
                    let x = g.op("add", &x, site.a.base);
                    let kx = g.op("mul", &s_, kper);
                    let kx = g.op("add", &kx, &k0);
                    let kx = g.op("add", &kx, &kk);
                    let kx = g.op("mul", &kx, site.a.ck);
                    let gi = g.op("add", &x, &kx);
                    (g_ptr(g, &ag.to_string(), &gi), el.to_string())
                });
            }
            // The column operand.
            let unit = if st.b_vec { v } else { 1 };
            one(g, slices * cw * kc, unit, &mut |g, el| {
                let (s_, j, kk, sx) = if st.b_kmaj {
                    let r = g.op("udiv", el, kc);
                    let kk = g.op("urem", el, kc);
                    let s_ = g.op("udiv", &r, cw);
                    let j = g.op("urem", &r, cw);
                    let w = swz(g, &r);
                    let ch = g.op("udiv", &kk, v);
                    let ch = g.op("xor", &ch, &w);
                    let ch = g.op("mul", &ch, v);
                    let lo = g.op("urem", &kk, v);
                    let rb = g.op("mul", &r, kc);
                    let x = g.op("add", &rb, &ch);
                    let x = g.op("add", &x, &lo);
                    (s_, j, kk, g.op("add", &x, a_el))
                } else {
                    let r = g.op("udiv", el, cw);
                    let j = g.op("urem", el, cw);
                    let s_ = g.op("udiv", &r, kc);
                    let kk = g.op("urem", &r, kc);
                    (s_, j, kk, g.op("add", el, a_el))
                };
                let j = g.op("add", &j, j0);
                let x = g.op("mul", &j, site.b.clane);
                let x = g.op("add", &x, site.b.base);
                let kx = g.op("mul", &s_, kper);
                let kx = g.op("add", &kx, &k0);
                let kx = g.op("add", &kx, &kk);
                let kx = g.op("mul", &kx, site.b.ck);
                let gi = g.op("add", &x, &kx);
                (g_ptr(g, &bg.to_string(), &gi), sx)
            });
        };
        let mut ncopy = 0u32;
        let init = if slices > 1 {
            "-0.0".to_string()
        } else {
            const_literal(&site.seed)
        };
        if st.stages == 2 {
            issue(self, "0", "0", &mut ncopy);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        let pre = self.cur.clone();
        let (lp, latch) = (tag.clone(), format!("{tag}l"));
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let x = format!("%{tag}x");
        self.line(format!("{x} = phi i64 [0, %{pre}], [{x}n, %{latch}]"));
        let acc_name = |r: u64, q: u64| format!("%{tag}a{r}_{q}");
        for r in 0..b {
            for q in 0..nj {
                let a = acc_name(r, q);
                self.line(format!("{a} = phi {t} [{init}, %{pre}], [{a}n, %{latch}]"));
            }
        }
        let bo = if st.stages == 1 {
            self.barrier();
            issue(self, &x, "0", &mut ncopy);
            if st.cp {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
                self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
            }
            self.barrier();
            "0".to_string()
        } else {
            self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
            self.barrier();
            // The next panel (the last again on the last trip: in bounds,
            // into the buffer no thread reads this trip).
            let nx = self.op("add", &x, 1);
            let fit = self.tmp();
            self.line(format!("{fit} = icmp ult i64 {nx}, {trips}"));
            let nt = self.tmp();
            self.line(format!(
                "{nt} = select i1 {fit}, i64 {nx}, i64 {}",
                trips - 1
            ));
            let nb = self.op("and", &nx, 1);
            let nbo = self.op("mul", &nb, buf_el);
            issue(self, &nt, &nbo, &mut ncopy);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cb = self.op("and", &x, 1);
            self.op("mul", &cb, buf_el)
        };
        // The fold of this panel: per 16-byte chunk of `k`, each row's and
        // each column's values, then every cell's steps in `k` order.
        let k0 = self.op("mul", &x, kc);
        let sk = self.op("mul", sl, kper);
        let ka = self.op("add", &sk, &k0);
        let mut cur: Vec<Vec<String>> = (0..b)
            .map(|r| (0..nj).map(|q| acc_name(r, q)).collect())
            .collect();
        let vload = |g: &mut Self, idx: &str, n: u64| -> Vec<String> {
            let p = s_ptr(g, idx);
            if n == 1 {
                let x = g.tmp();
                g.line(format!("{x} = load {t}, ptr addrspace(3) {p}"));
                return vec![x];
            }
            let vec = g.tmp();
            g.line(format!(
                "{vec} = load <{n} x {t}>, ptr addrspace(3) {p}, align 16"
            ));
            (0..n)
                .map(|l| {
                    let x = g.tmp();
                    g.line(format!("{x} = extractelement <{n} x {t}> {vec}, i64 {l}"));
                    x
                })
                .collect()
        };
        for ch in 0..nch {
            let kk0 = ch * v;
            let mut av: Vec<Vec<String>> = Vec::new();
            for r in 0..b {
                av.push(match &a_src {
                    USrc::Gmem(_) => {
                        let x = self.op("add", &arow, r);
                        let x = self.op("mul", &x, kc);
                        let x = self.op("add", &x, kk0);
                        let x = self.op("add", &x, &bo);
                        vload(self, &x, v)
                    }
                    USrc::Member { k: p, .. } => {
                        let (off, wp) = (
                            self.kept[*p].expect("a tile's row operand is on chip"),
                            u.members[*p].w,
                        );
                        let ck = site.a.ck;
                        let vec = ck == 1 && site.a.base % v == 0 && wp % v == 0 && off % 16 == 0;
                        let kx = self.op("add", &ka, kk0);
                        let x = self.op("mul", &kx, ck);
                        let x = self.op("add", &x, r * wp + site.a.base);
                        if vec {
                            let p = self.smem(off, &t, &x);
                            let vv = self.tmp();
                            self.line(format!(
                                "{vv} = load {ty_v}, ptr addrspace(3) {p}, align 16"
                            ));
                            (0..v)
                                .map(|l| {
                                    let y = self.tmp();
                                    self.line(format!("{y} = extractelement {ty_v} {vv}, i64 {l}"));
                                    y
                                })
                                .collect()
                        } else {
                            (0..v)
                                .map(|l| {
                                    let y = self.op("add", &x, l * ck);
                                    let p = self.smem(off, &t, &y);
                                    let z = self.tmp();
                                    self.line(format!("{z} = load {t}, ptr addrspace(3) {p}"));
                                    z
                                })
                                .collect()
                        }
                    }
                    _ => unreachable!("a tile's row operand is a member or Gmem"),
                });
            }
            let mut bv: Vec<Vec<String>> = Vec::new();
            for (rb, w) in &brow {
                bv.push(if st.b_kmaj {
                    let c_ = self.op("xor", w, ch);
                    let c_ = self.op("mul", &c_, v);
                    let x = self.op("add", rb, &c_);
                    let x = self.op("add", &x, a_el);
                    let x = self.op("add", &x, &bo);
                    vload(self, &x, v)
                } else {
                    Vec::new()
                });
            }
            if !st.b_kmaj {
                for (q, j) in cols.iter().enumerate() {
                    for l in 0..v {
                        let x = self.op("mul", sl, kc);
                        let x = self.op("add", &x, kk0 + l);
                        let x = self.op("mul", &x, cw);
                        let jl = self.op("sub", j, j0);
                        let x = self.op("add", &x, &jl);
                        let x = self.op("add", &x, a_el);
                        let x = self.op("add", &x, &bo);
                        let y = vload(self, &x, 1);
                        bv[q].extend(y);
                    }
                }
            }
            for l in 0..v as usize {
                let last = ch + 1 == nch && l + 1 == v as usize;
                for r in 0..b as usize {
                    for q in 0..nj as usize {
                        let out = if last {
                            format!("{}n", acc_name(r as u64, q as u64))
                        } else {
                            self.tmp()
                        };
                        let (av_, bv_) = (av[r][l].clone(), bv[q][l].clone());
                        let acc = cur[r][q].clone();
                        self.tile_step(site, &t, &acc, &av_, &bv_, &out);
                        cur[r][q] = out;
                    }
                }
            }
        }
        self.line(format!("br label %{latch}"));
        self.label(&latch);
        self.line(format!("{x}n = add i64 {x}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp ult i64 {x}n, {trips}"));
        self.line(format!("br i1 {fin}, label %{lp}, label %{lp}e"));
        self.label(&format!("{lp}e"));
        // The last trip's copy (the last panel again) lands before anything
        // else takes the region: a split tile's partials, a later phase's
        // trees or panels.
        if st.stages == 2 {
            self.line("call void @llvm.nvvm.cp.async.wait.group(i32 0)");
        }
        let (act, act_end) = (format!("{tag}act"), format!("{tag}acte"));
        if active {
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {}", c * slices));
            self.line(format!("br i1 {a}, label %{act}, label %{act_end}"));
            self.label(&act);
        }
        self.tile_tail(
            k,
            site,
            &t,
            b,
            (nj, slices),
            cols,
            cur,
            (active, &act_end),
            &tag,
            true,
        );
    }

    /// A tile phase's end: the slices' partials combined (a split tile), each
    /// row's cells kept on chip and / or stored, the phase's barrier.
    #[allow(clippy::too_many_arguments)]
    fn tile_tail(
        &mut self,
        k: usize,
        site: &TileSite,
        t: &str,
        b: u64,
        (nj, slices): (u64, u64),
        cols: &[String],
        cur: Vec<Vec<String>>,
        (active, act_end): (bool, &str),
        tag: &str,
        staged: bool,
    ) {
        let c = site.c;
        // The results: one slice's accumulators, or the slices combined.
        let mut res: Vec<Vec<String>> = cur;
        let (comb, comb_end) = (format!("{tag}z"), format!("{tag}ze"));
        if slices > 1 {
            // A staged tile's partials share its panels' region: every
            // thread's last panel read comes first.
            if staged {
                self.barrier();
            }
            let off = self.part.expect("a split tile has partials");
            let sl = self.op("udiv", "%l", c);
            for r in 0..b {
                let x = self.op("mul", &sl, b);
                let x = self.op("add", &x, r);
                let x = self.op("mul", &x, c);
                let idx = self.op("add", &x, &cols[0]);
                let p = self.smem(off, t, &idx);
                let v = res[r as usize][0].clone();
                self.line(format!("store {t} {v}, ptr addrspace(3) {p}"));
            }
            self.barrier();
            let a = self.tmp();
            self.line(format!("{a} = icmp ult i64 %l, {c}"));
            self.line(format!("br i1 {a}, label %{comb}, label %{comb_end}"));
            self.label(&comb);
            let seed = const_literal(&site.seed);
            for r in 0..b {
                let mut tot = seed.clone();
                for s in 0..slices {
                    let idx = self.op("add", &cols[0], (s * b + r) * c);
                    let p = self.smem(off, t, &idx);
                    let v = self.tmp();
                    self.line(format!("{v} = load {t}, ptr addrspace(3) {p}"));
                    let n = self.tmp();
                    let (x, y) = if site.add_acc_first {
                        (&tot, &v)
                    } else {
                        (&v, &tot)
                    };
                    self.line(format!("{n} = fadd {t} {x}, {y}"));
                    tot = n;
                }
                res[r as usize][0] = tot;
            }
        }
        for r in 0..b {
            for q in 0..nj {
                let v = res[r as usize][q as usize].clone();
                let j = cols[q as usize].clone();
                if let Some(off) = self.kept[k] {
                    let idx = self.op("add", &j, r * c);
                    let p = self.smem(off, t, &idx);
                    self.line(format!("store {t} {v}, ptr addrspace(3) {p}"));
                }
                if self.ys[k].is_some() {
                    let i = self.op("add", "%i", r);
                    let x = self.op("mul", &i, c);
                    let at = self.op("add", &x, &j);
                    self.store(k, &v, &at);
                }
            }
        }
        if slices > 1 {
            self.line(format!("br label %{comb_end}"));
            self.label(&comb_end);
        }
        if active {
            self.line(format!("br label %{act_end}"));
            self.label(act_end);
        }
        self.barrier();
    }

    /// `n` consecutive-in-`k` operand values at `idx` (stride `ck`): from the
    /// block's shared memory at byte `at` (`shared`) or `Gmem` operand `at`,
    /// as `n / v` loads of `v` lanes.
    #[allow(clippy::too_many_arguments)]
    fn tile_loads(
        &mut self,
        shared: bool,
        at: &str,
        t: &str,
        idx: &str,
        n: u64,
        v: u64,
        ck: u64,
    ) -> Vec<String> {
        let ptr = |g: &mut Self, ix: &str| -> String {
            if shared {
                g.smem(at.parse().expect("a byte offset"), t, ix)
            } else {
                let p = g.tmp();
                g.line(format!(
                    "{p} = getelementptr inbounds {t}, ptr addrspace(1) %x{at}, i64 {ix}"
                ));
                p
            }
        };
        let space = if shared { 3 } else { 1 };
        let mut out = Vec::new();
        if v == 4 && n % 4 == 0 {
            for c0 in (0..n).step_by(4) {
                let ix = self.op("add", idx, c0);
                let p = ptr(self, &ix);
                let vec = self.tmp();
                self.line(format!(
                    "{vec} = load <4 x {t}>, ptr addrspace({space}) {p}, align 16"
                ));
                for l in 0..4 {
                    let x = self.tmp();
                    self.line(format!("{x} = extractelement <4 x {t}> {vec}, i64 {l}"));
                    out.push(x);
                }
            }
            return out;
        }
        for c0 in 0..n {
            let ix = self.op("add", idx, c0 * ck);
            let p = ptr(self, &ix);
            let x = self.tmp();
            self.line(format!("{x} = load {t}, ptr addrspace({space}) {p}"));
            out.push(x);
        }
        out
    }

    /// `out = acc ⊕ a ⊗ b`: the tile's step in the source's operand order, one
    /// `llvm.fma` where the face contracts.
    fn tile_step(&mut self, site: &TileSite, t: &str, acc: &str, a: &str, b: &str, out: &str) {
        let (x, y) = if site.mul_a_first { (a, b) } else { (b, a) };
        if self.fma {
            let f = if t == "double" { "f64" } else { "f32" };
            self.line(format!(
                "{out} = call {t} @llvm.fma.{f}({t} {x}, {t} {y}, {t} {acc})"
            ));
            return;
        }
        let p = self.tmp();
        self.line(format!("{p} = fmul {t} {x}, {y}"));
        let (l, r) = if site.add_acc_first {
            (acc, p.as_str())
        } else {
            (p.as_str(), acc)
        };
        self.line(format!("{out} = fadd {t} {l}, {r}"));
    }

    /// Member `k`'s value type.
    fn ret_ty(&self, k: usize) -> String {
        match (&self.calls[k], &self.u.members[k].kind) {
            (Some(c), _) => c.ret.clone(),
            (None, UKind::Fold { site, .. }) => lower_ty(&site.elem).expect("fold elem"),
            (None, UKind::Tile { site }) => lower_ty(&site.elem).expect("tile elem"),
            (None, UKind::Cell { .. }) => unreachable!("a cell has a call"),
        }
    }

    /// A row-block unit: per-row value `v` of member `k` into its slot for
    /// the segments after this one (lane 0; read after the segment's end).
    fn keep_row_value(&mut self, k: usize, v: &str) {
        let (Some(off), Some(rl)) = (self.xrow.get(k).copied().flatten(), self.rloc.clone()) else {
            return;
        };
        let ty = self.ret_ty(k);
        self.lane0(&format!("x{k}s"), |g| {
            let p = g.smem(off, &ty, &rl);
            g.line(format!("store {ty} {v}, ptr addrspace(3) {p}"));
        });
    }

    /// Does a member of this segment (`from..=k`) read the on-chip row
    /// buffer member `k` writes — another member's row in it, earlier in the
    /// segment, or `k`'s own producer's in place?
    fn reread(&self, k: usize, from: usize) -> bool {
        let Some(off) = self.kept[k] else {
            return false;
        };
        self.u.members[from..=k].iter().any(|mb| {
            mb.srcs
                .iter()
                .any(|x| matches!(x, USrc::Member { k: p, .. } if self.kept[*p] == Some(off)))
        })
    }

    /// compile-time-graph A: the members `ks` (segment `s`, no tile) over
    /// the block's `b` rows, [`row_groups`] rows at once: the block's lanes
    /// split into one group of `lanes ÷ P` per row (row `%i + %r<s> + g`,
    /// lane `%l mod (lanes ÷ P)`), the groups of a batch in lockstep; a
    /// barrier closes each batch (the tree partials and per-row slots are
    /// reused by the next).
    fn rows_loop(&mut self, s: usize, b: u64, ks: std::ops::Range<usize>) {
        let lanes0 = self.lanes;
        let p = row_groups(b, lanes0);
        let l = lanes0 / p;
        let (g, lane, base) = (format!("%g{s}"), format!("%ll{s}"), format!("%lb{s}"));
        self.line(format!("{g} = udiv i64 %l, {l}"));
        self.line(format!("{lane} = urem i64 %l, {l}"));
        self.line(format!("{base} = mul i64 {g}, {l}"));
        let pre = self.cur.clone();
        let (lp, latch, end) = (format!("S{s}"), format!("S{s}l"), format!("S{s}e"));
        self.line(format!("br label %{lp}"));
        self.label(&lp);
        let (r, rl, i) = (format!("%r{s}"), format!("%rl{s}"), format!("%i{s}"));
        self.line(format!("{r} = phi i64 [0, %{pre}], [{r}n, %{latch}]"));
        self.line(format!("{rl} = add i64 {r}, {g}"));
        self.line(format!("{i} = add i64 %i, {rl}"));
        let (row0, rloc0) = (self.row.clone(), self.rloc.clone());
        let (lane0, base0) = (self.lane.clone(), self.lane_base.clone());
        self.row = i;
        self.rloc = Some(rl);
        self.lane = lane;
        self.lane_base = base;
        self.lanes = l;
        self.cur_seg = s;
        for k in ks.clone() {
            // A member writing an on-chip row buffer waits for every reader
            // of the buffer's earlier occupant in this segment: its own
            // producer's row in place, or a row `unit_plan` freed at an
            // earlier death (a fold's walk, a per-row cell or a stored-only
            // cell reads it with no barrier after).
            if self.reread(k, ks.start) {
                self.barrier();
            }
            self.phase(k);
        }
        self.line(format!("br label %{latch}"));
        self.label(&latch);
        self.barrier();
        self.line(format!("{r}n = add i64 {r}, {p}"));
        let c = self.tmp();
        self.line(format!("{c} = icmp ult i64 {r}n, {b}"));
        self.line(format!("br i1 {c}, label %{lp}, label %{end}"));
        self.label(&end);
        // A row's register values do not outlive it.
        for k in ks {
            self.vals[k] = None;
        }
        self.row = row0;
        self.rloc = rloc0;
        self.lane = lane0;
        self.lane_base = base0;
        self.lanes = lanes0;
    }
}
