//! unit geometry: the block shape, lanes, row groups, segments and the row-wise tests

use super::*;

/// A fold step the lanes may reassociate: `+`, `×`, or a running max/min.
pub(super) fn tree_step(op: &RowFoldOp) -> bool {
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
    if let Some(st) = &u.stream {
        return stream::stream_geo(u, st);
    }
    if let Some(blk) = &u.block {
        return block_geo(u, blk, g, free, bytes);
    }
    if let Some(sl) = &u.slices {
        return slice_geo(u, sl, g, bytes);
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

/// long-rows `row-slices`: a sliced unit's geometry — `rows · s` blocks of
/// the widest slice's lanes (to a power of two in `[warp, 256]`), each fold's
/// tree partials in shared memory; no row kept on chip (a block walks one
/// row's slice an item, a later phase recomputes a cell at its read).
pub(super) fn slice_geo(
    u: &Unit,
    sl: &Slices,
    g: &crate::profile::Gpu,
    bytes: impl Fn(&UMember) -> u64,
) -> Geo {
    let n = u.members.len();
    let lanes = u
        .members
        .iter()
        .map(|mb| match &mb.kind {
            UKind::Fold { site, .. } => site.k / sl.s,
            UKind::Cell { .. } if !mb.per_row() && mb.store => mb.w / sl.s,
            _ => 1,
        })
        .max()
        .unwrap_or(1)
        .next_power_of_two()
        .clamp(g.warp, 256);
    let mut smem = 0u64;
    let mut tree = vec![None; n];
    for (k, mb) in u.members.iter().enumerate() {
        if matches!(mb.kind, UKind::Fold { .. }) {
            let off = smem.next_multiple_of(16);
            smem = off + lanes * bytes(mb);
            tree[k] = Some(off);
        }
    }
    Geo {
        lanes,
        gx: u.rows * sl.s,
        block: lanes,
        smem,
        tree,
        kept: vec![None; n],
        xrow: Vec::new(),
        part: None,
        panel: None,
    }
}

/// The value a lane's partial of `op` starts from: `+`'s `−0`, `×`'s `1`, a
/// running max's `−∞` and min's `+∞` (the seed joins once, after the
/// combine).
pub(super) fn step_identity(op: &RowFoldOp) -> String {
    match *op {
        RowFoldOp::Arith {
            op: Operation::Mul, ..
        } => "1.0".into(),
        RowFoldOp::Arith { .. } => "-0.0".into(),
        RowFoldOp::Select {
            cmp,
            acc_lhs,
            acc_if_true,
        } => {
            // `e` wins when it is the greater: the comparison reads `e > acc`
            // (`>`/`≥` with `e` first, `<`/`≤` with `acc` first) and picks `e`.
            let e_gt = matches!(cmp, Operation::Gt | Operation::Ge) != acc_lhs;
            match e_gt != acc_if_true {
                true => "0xFFF0000000000000".into(),
                false => "0x7FF0000000000000".into(),
            }
        }
    }
}

/// A row-block unit's threads: its widest row extent (a fold's `k`, a cell's
/// `w`, a tile's `c`) to a power of two in `[warp, 256]`, as [`unit_geo`]'s.
pub(super) fn block_lanes(members: &[UMember], warp: u64) -> u64 {
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
pub(super) const SECTOR_BYTES: u64 = 32;

/// The fewest lanes a row of a row-block segment takes.
pub(super) const ROW_GROUP_LANES: u64 = 8;

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
pub(super) fn block_geo(
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
pub(super) fn rowwise(reach: &Reach, cw: u64, pw: u64, rows: u64) -> bool {
    reach.within_rows(cw, pw, rows)
}

/// The rows a reach of `cw`-cell consumer rows reads a per-row value at: the
/// row itself, every row.
pub(super) fn rowwise_single(reach: &Reach, cw: u64, rows: u64) -> bool {
    rowwise(reach, cw, 1, rows)
}

pub(super) const SAME: Reach = Reach::Affine {
    w: 1,
    base: 0,
    cq: 1,
    cr: 0,
    taps: Vec::new(),
};
