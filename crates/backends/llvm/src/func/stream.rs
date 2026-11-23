//! **flash-attention `stream`: a unit as one pass over windows of its axis**
//! ([`mapal_ir::CategoryIr::stream_plan`]).
//!
//! A row-block group whose tile folds over an in-group row (attention's
//! `P·V` over the scores' row) runs as one kernel of `b` rows a block that
//! walks the axis in windows of `kb`: each window's producer tile (`q·Kᵀ`)
//! into registers, the cells and folds of the axis over it — each fold's
//! state carried window to window by its law (a rescale's shift the running
//! extreme, its state rescaled by `exp(a(m_old, m_new))` when the extreme
//! moves; a factor or shift applied once after the last window) — and the
//! consumer tile's accumulators folding the window (`P · V_window`). Nothing
//! axis-wide is stored; each operand window is read once a block
//! (`cp.async` into shared memory, the next window landing while this one
//! computes).
//!
//! Thread layout: `threads = (b ÷ rt) · g` — a row group of `g` lanes (one
//! warp's aligned `g`, a power of two) holds `rt` rows; lane `x` the window
//! columns `x + g·z` (`z < kb ÷ g`) of its rows and the consumer columns
//! `x·oc + q` (contiguous, 16-byte loads) or `x + g·q`. A window's row
//! extreme is a butterfly over the group's lanes; every other fold keeps a
//! lane's partial, combined once after the last window.
//!
//! sweep-attention-2 `stream-mma`: where every product of the pass admits a
//! 16-bit member the part folds on a warp matrix product (`m16n8k16`), the
//! threads take that instruction's accumulator layout instead — `rt = 2`,
//! `g = 4`, `b ÷ 16` warps: lane `l` of warp `w` holds rows `16w + l%32÷4`
//! and `+ 8`, the window columns `8·(z÷2) + 2·(l%4) + z%2` and the consumer
//! columns likewise. The producer's row operand is loaded once into its `a`
//! fragments (registers, no panel), each window's `b` fragments from the
//! window (two values rounded once into a register); the producer's cells,
//! in that layout, are the consumer's `a` fragments as they stand — no
//! window of the row operand in shared memory.

use super::*;
use mapal_ir::{DepLaw, StreamGroup, StreamRole};

/// A unit's streamed realization: the fact's group and the geometry the
/// rule chose ([`FnEmit::stream_choice`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Stream {
    pub group: StreamGroup,
    /// Rows a block, keys a window, rows a thread, lanes a row group, and
    /// the block's threads.
    pub b: u64,
    pub kb: u64,
    pub rt: u64,
    pub g: u64,
    pub threads: u64,
    /// Windows land by `cp.async` (else each thread loads and stores).
    pub cp: bool,
    /// The model's `(streamed, split)` ns; `(0, 0)` when forced.
    pub ns: (u64, u64),
    /// R3 split-KV: each row block's windows over `s` blocks, each a
    /// contiguous `nb ÷ s` of them; the row block's last block to finish
    /// combines the partial states by their laws (a ticket). `1`: one block.
    pub s: u64,
    /// flash-attention F1: per member, its rows as a digit permutation of
    /// the group's (`unit_plan`'s [`mapal_ir::RowMap`]).
    pub rows_of: Vec<Option<mapal_ir::RowMap>>,
    /// sweep-attention-2 `stream-mma`: the products on this warp matrix
    /// product's format (`m16n8k16`, the accumulator's fragment layout);
    /// `None`: FFMA.
    pub mma: Option<crate::profile::MmaFmt>,
}

/// `m16n8k16`'s `m` / `n` / `k`: the fragment geometry `stream-mma` lays the
/// threads out by ([`crate::profile::MMA_SYNC_16`]).
const MMA_M: u64 = 16;
const MMA_N: u64 = 8;
const MMA_K: u64 = 16;

/// **sweep-attention-2 `stream-mma`: the warp matrix product a stream pass's
/// products fold on** — the 16-bit member every producer and consumer
/// product admits ([`mapal_ir::TileSite::half`], one format), where the part
/// lists a warp product of it whose geometry is `m16n8k16` and whose
/// accumulator meets the member's `u_a = 2⁻²³` ([`profile::MatrixOp::acc_meets`]),
/// at a rate the part states. The operands are f32 rounded once to the
/// member as each fragment is formed: `bf16` (an `f16` member rounds no f32
/// operand without a range fact, `half_operands`' rule). A producer's `d`
/// whole `k`-steps, a consumer's `c` whole `n` tiles. `None`: FFMA.
pub(crate) fn mma_fmt(
    members: &[UMember],
    roles: &[StreamRole],
    gpu: &crate::profile::Gpu,
) -> Option<crate::profile::MmaFmt> {
    use crate::profile::{Home, MMA_SYNC_16, MmaFmt};
    let f32t = mapal_ir::Ty::f32();
    let mut fmt: Option<MmaFmt> = None;
    for (k, mb) in members.iter().enumerate() {
        let UKind::Tile { site } = &mb.kind else {
            continue;
        };
        let whole = match roles[k] {
            StreamRole::Produce => site.k.is_multiple_of(MMA_K),
            StreamRole::Consume { .. } => site.c.is_multiple_of(MMA_N),
            _ => continue,
        };
        let f = MmaFmt::of_narrow(site.half?);
        let op = gpu.caps.matrix_op(f, Home::Regs)?;
        if !whole
            || f != MmaFmt::Bf16
            || site.elem != f32t
            || site.a.narrow.is_some()
            || site.b.narrow.is_some()
            || op.shape != MMA_SYNC_16
            || !op.acc_meets(-23)
            || gpu.matrix_flops_per_clk(&op) == 0
            || fmt.is_some_and(|x| x != f)
        {
            return None;
        }
        fmt = Some(f);
    }
    fmt
}

/// A streamed unit's shared memory: per member `(offset, row stride)` in
/// elements — a producer's row panel and window, a consumer's window and
/// the window of its row operand.
#[derive(Clone, Debug, Default)]
pub(crate) struct Layout {
    pub q: Vec<Option<(u64, u64)>>,
    pub k: Vec<Option<(u64, u64)>>,
    pub v: Vec<Option<(u64, u64)>>,
    pub a: Vec<Option<(u64, u64)>>,
    /// A 16-byte slot copies of rows past the axis land in (then zeroed in
    /// place): every copy is issued, none branches.
    pub dump: u64,
    /// F6: the block's vote on skipping its masked windows.
    pub flag: u64,
    pub total: u64,
}

impl Stream {
    /// Window columns a lane holds.
    pub fn ct(&self) -> u64 {
        self.kb / self.g
    }

    /// F5/F6: the mask whose wholly masked windows the block skips.
    pub fn skip(&self) -> Option<mapal_ir::StreamGuard> {
        self.group.guard.clone().filter(|g| g.skip)
    }

    /// F5: the fraction of the `rows ÷ b` blocks' windows they visit (the
    /// band from the first window their rows see through the last); `1`
    /// without a skipped mask.
    pub fn visited(&self, rows: u64) -> f64 {
        let Some(gd) = self.skip() else {
            return 1.0;
        };
        let (nb, w, kb) = (self.nb(), self.group.axis, self.kb);
        let blocks = rows / self.b;
        if blocks == 0 || blocks * nb > 1 << 24 {
            return 1.0;
        }
        let mut seen = 0u64;
        for blk in 0..blocks {
            let r = (blk * self.b, blk * self.b + self.b - 1);
            let vis: Vec<u64> = (0..nb)
                .filter(|&t| {
                    let l = (t * kb, (t * kb + kb).min(w) - 1);
                    gd.form.classify(r, l) != mapal_ir::BoxClass::Masked
                })
                .collect();
            seen += match (vis.first(), vis.last()) {
                (Some(a), Some(z)) => z + 1 - a,
                _ => nb,
            };
        }
        seen as f64 / (blocks * nb) as f64
    }

    /// Windows a row walks.
    pub fn nb(&self) -> u64 {
        self.group.axis.div_ceil(self.kb)
    }

    /// A consumer of `c` columns: columns a lane holds, contiguous.
    pub fn oc(&self, c: u64) -> (u64, bool) {
        (c.div_ceil(self.g), c.is_multiple_of(self.g * 4))
    }

    pub fn layout(&self, members: &[UMember]) -> Layout {
        let n = members.len();
        let mut lay = Layout {
            q: vec![None; n],
            k: vec![None; n],
            v: vec![None; n],
            a: vec![None; n],
            ..Layout::default()
        };
        let mut off = 0u64;
        let mut take = |elems: u64| {
            let o = off.next_multiple_of(16);
            off = o + elems * 4;
            o
        };
        for (k, mb) in members.iter().enumerate() {
            let UKind::Tile { site } = &mb.kind else {
                continue;
            };
            match (&self.group.roles[k], self.mma.is_some()) {
                (StreamRole::Produce, false) => {
                    let dp = site.k.next_multiple_of(4) + 4;
                    lay.q[k] = Some((take(self.b * dp), dp));
                    lay.k[k] = Some((take(self.kb * dp), dp));
                }
                (StreamRole::Consume { .. }, false) => {
                    let cp = site.c.next_multiple_of(4);
                    lay.v[k] = Some((take(self.kb * cp), cp));
                    let ap = self.kb + 4;
                    lay.a[k] = Some((take(self.b * ap), ap));
                }
                // `stream-mma`: the row operand's fragments live in registers
                // (no panel, no row window). A window row `≡ 8 (mod 16)`
                // words apart: a half-warp's 8-byte `b` fragment loads (rows
                // `l%32÷4`, words `2·(l%4)`) on distinct banks; the
                // consumer's rows `≡ 4 (mod 8)` apart: its 4-byte loads (rows
                // `2·(l%4)`, column `l%32÷4`) likewise.
                (StreamRole::Produce, true) => {
                    let dp = site.k + 8;
                    lay.k[k] = Some((take(self.kb * dp), dp));
                }
                (StreamRole::Consume { .. }, true) => {
                    let cp = site.c + 4;
                    lay.v[k] = Some((take(self.kb * cp), cp));
                }
                _ => {}
            }
        }
        lay.dump = take(4);
        lay.flag = take(4);
        lay.total = off;
        lay
    }

    /// Registers a thread's state and operands take, estimated: each
    /// producer's `rt × ct` window, each consumer's `rt × oc` accumulators,
    /// the row's fold states, the 16-byte operand vectors, the addresses.
    pub fn regs(&self, members: &[UMember]) -> u64 {
        let ct = self.ct();
        let mut r = 4 * (self.rt + ct) + crate::profile::FMA_ADDR_REGS;
        for (k, mb) in members.iter().enumerate() {
            r += match (&self.group.roles[k], &mb.kind) {
                // `stream-mma`: and the row operand's `a` fragments, four
                // registers a `k`-step.
                (StreamRole::Produce, UKind::Tile { site }) if self.mma.is_some() => {
                    self.rt * ct + 4 * site.k / MMA_K
                }
                (StreamRole::Produce, _) => self.rt * ct,
                (StreamRole::Cell { .. }, _) => self.rt * ct / 2,
                (StreamRole::Consume { .. }, UKind::Tile { site }) => self.rt * self.oc(site.c).0,
                (StreamRole::Fold { .. }, _) => 2 * self.rt,
                _ => 0,
            };
        }
        r
    }

    /// The blocks an SM holds at this geometry — its threads, shared memory
    /// and the registers the model counts ([`Self::regs`]): the residency
    /// [`FnEmit::stream_ns`] prices, and the kernel's `.minnctapersm`
    /// (sweep-attention: without it ptxas, not knowing the launch's dynamic
    /// shared memory, capped a 76 KB block at 128 registers for a second
    /// block that never fits — 8 bytes spilled; declared, 168 and none).
    pub fn per_sm(&self, gpu: &crate::profile::Gpu, members: &[UMember], smem: u64) -> u64 {
        let regs = self.regs(members).max(32);
        (gpu.max_threads_per_sm / self.threads)
            .min(gpu.smem_per_sm / smem.max(1))
            .min(gpu.regs_per_sm / (self.threads * regs))
            .min(gpu.max_blocks_per_sm)
            .max(1)
    }

    /// The intent fields:
    /// `stream=b<b>xkb<kb>:rt<rt>g<g>t<threads>[:mma][:cp][:s<s>]`.
    pub fn label(&self) -> String {
        format!(
            "b{}xkb{}:rt{}g{}t{}{}{}{}",
            self.b,
            self.kb,
            self.rt,
            self.g,
            self.threads,
            if self.mma.is_some() { ":mma" } else { "" },
            if self.cp { ":cp" } else { "" },
            match self.s {
                1 => String::new(),
                s => format!(":s{s}"),
            }
        )
    }
}

/// Which address of a tile site.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    A,
    B,
    Out,
}

/// An axis of a tile site's address.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ax {
    Lane,
    K,
}

/// A tile site's address over `(i, lane, k)`: its plain read, its
/// mixed-radix form, or the plain output `i·c + lane`.
#[derive(Clone, Debug)]
pub(crate) enum Addr {
    Plain(mapal_ir::TileRead),
    Radix(mapal_ir::RadixAddr),
    Out(u64),
}

impl Addr {
    pub fn of(site: &mapal_ir::TileSite, op: Operand) -> Addr {
        let r = site.radix.as_deref();
        match op {
            Operand::A => match r.and_then(|r| r.a.clone()) {
                Some(x) => Addr::Radix(x),
                None => Addr::Plain(site.a.clone()),
            },
            Operand::B => match r.and_then(|r| r.b.clone()) {
                Some(x) => Addr::Radix(x),
                None => Addr::Plain(site.b.clone()),
            },
            Operand::Out => match r.and_then(|r| r.out.clone()) {
                Some(x) => Addr::Radix(x),
                None => Addr::Out(site.c),
            },
        }
    }

    /// Consecutive elements along `ax` over `n` of it, and every other term
    /// a multiple of 4: 16-byte chunks at every row.
    pub fn vec(&self, ax: Ax, n: u64) -> bool {
        if !n.is_multiple_of(4) {
            return false;
        }
        match self {
            Addr::Plain(r) => {
                let (unit, other) = match ax {
                    Ax::Lane => (r.clane, r.ck),
                    Ax::K => (r.ck, r.clane),
                };
                unit == 1
                    && other.is_multiple_of(4)
                    && r.ci.is_multiple_of(4)
                    && r.base.is_multiple_of(4)
                    && r.ksplit.is_none()
            }
            Addr::Radix(r) => {
                let (along, other) = match ax {
                    Ax::Lane => (&r.lane, &r.k),
                    Ax::K => (&r.k, &r.lane),
                };
                let unit = matches!(along[..], [d] if d.stride == 1 && d.coef == 1 && d.radix >= n);
                unit && r.base.is_multiple_of(4)
                    && other.iter().chain(&r.i).all(|d| d.coef.is_multiple_of(4))
            }
            Addr::Out(c) => ax == Ax::Lane && c.is_multiple_of(4),
        }
    }
}

/// sweep-attention `unit-uniform`: no number an epilogue site's body
/// captures is read after its fold (`TilePost::after`: a loop counter only
/// moves the fold's operands) — the epilogue, evaluated in the kernel
/// without it, is the body's.
fn post_numbers_ok(p: &mapal_ir::TilePost) -> bool {
    p.after.is_empty()
}

/// A streamed unit's launch geometry.
pub(crate) fn stream_geo(u: &Unit, st: &Stream) -> Geo {
    let n = u.members.len();
    Geo {
        lanes: st.threads,
        gx: u.rows / st.b * st.s,
        block: st.threads,
        smem: st.layout(&u.members).total,
        tree: vec![None; n],
        kept: vec![None; n],
        xrow: Vec::new(),
        part: None,
        panel: None,
    }
}

impl<'a> FnEmit<'a> {
    /// **The `stream` rule (R1–R2).** Where `stream_plan` proves the row-block
    /// group `grp` one pass, the order is free (the folds regroup into
    /// windows) and the rung is on: every geometry — `threads` of 128 or 256,
    /// `b` rows dividing the group's, windows of 16–64 keys (at least 16: the
    /// rescales stay inside the default face's bound), `rt` rows a thread —
    /// whose shared memory fits the part's opt-in limit and whose registers
    /// fit a thread's, priced by the plan model against the split
    /// realization's launches; the cheapest when it beats them.
    /// `Ok(None)`: not a stream group (or the rung is off); `Err` names why
    /// a stream group stays split.
    pub(super) fn stream_choice(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        grp: &mapal_ir::UnitGroup,
    ) -> Result<Option<Stream>, &'static str> {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if !rungs.stream || self.order != crate::Order::Free {
            return Ok(None);
        }
        let units = self.ir.unit_plan(self.f);
        let Some(gi) = units.groups.iter().position(|g| g == grp) else {
            return Ok(None);
        };
        let sp = self.ir.stream_plan(self.f);
        // A group the fact refuses is no stream group (its reason is the
        // fact's: `stream_plan().refused`).
        let Some(sg) = sp.of(gi) else {
            return Ok(None);
        };
        // The members the pass covers (`StreamGroup::roles`): a unit of the
        // whole group where trailing members follow is no stream unit.
        if members.len() != sg.roles.len() {
            return Ok(None);
        }
        let gpu = self.profile.gpu().ok_or("unpriced")?;
        // What this realization emits: f32 values, plain tile sites whose
        // operands are memory, no per-row work outside the windows.
        let f32t = mapal_ir::Ty::f32();
        for (k, mb) in members.iter().enumerate() {
            let elem =
                crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)));
            if elem != 4 {
                return Err("elem");
            }
            match (&sg.roles[k], &mb.kind) {
                (StreamRole::Row { .. }, _) => return Err("row"),
                (StreamRole::Produce, UKind::Tile { site }) => {
                    // Plain, epilogue or mixed-radix addresses over memory
                    // operands — a captured number moving one (`unit-uniform`:
                    // the launch moves its pointer) where the epilogue never
                    // reads that number; no narrow element yet.
                    let ok = site
                        .post
                        .as_ref()
                        .is_none_or(|p| post_numbers_ok(p) && p.reads.is_empty())
                        && site.a.narrow.is_none()
                        && site.b.narrow.is_none()
                        && site.a.ksplit.is_none()
                        && site.b.ksplit.is_none()
                        && site.elem == f32t;
                    if !ok || !matches!(mb.srcs[..2], [USrc::Gmem(_), USrc::Gmem(_)]) {
                        return Err("site");
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    // Its row operand the group's own rows: only its column
                    // operand may be moved.
                    let ok = site.radix.as_ref().is_none_or(|r| r.ua.is_empty())
                        && site.post.as_ref().is_none_or(post_numbers_ok)
                        && site.b.narrow.is_none()
                        && site.b.ksplit.is_none()
                        && site.elem == f32t;
                    if !ok || !matches!(mb.srcs[..2], [USrc::Member { .. }, USrc::Gmem(_)]) {
                        return Err("site");
                    }
                }
                (StreamRole::Fold { .. }, UKind::Fold { site, colmajor }) => {
                    if site.elem != f32t || colmajor.is_some() {
                        return Err("elem");
                    }
                }
                (StreamRole::Cell { .. }, UKind::Cell { .. }) => {}
                _ => return Err("member"),
            }
        }
        let cp = gpu.has_mma_sync() && !self.gpu_arms.off.cp_async;
        let forced = self.gpu_arms.stream_force;
        let axis = sg.axis;
        // sweep-attention-2 `stream-mma`: the warp matrix product every
        // product of the pass admits (`None`: FFMA only).
        let mma = match rungs.stream_mma {
            true => mma_fmt(members, &sg.roles, gpu),
            false => None,
        };
        let mut best: Option<(Stream, f64)> = None;
        // A forced geometry takes the matrix product where it admits one
        // (the measurement arm of the instruction; `stream-mma` off: FFMA).
        let mut forced_mma: Option<Stream> = None;
        let mut why = "budget";
        for threads in [256u64, 128, 64, 32] {
            for b in [128u64, 64, 32, 16, 8, 4, 2, 1] {
                // A block's rows inside one panel batch (`unit_plan`'s).
                if !rows.is_multiple_of(b) || !grp.panel.is_multiple_of(b) {
                    continue;
                }
                for kb in [64u64, 32, 16] {
                    // A window of at least 16 keys keeps the rescales inside
                    // the bound; a row shorter than that is one window.
                    let kb = if axis < 16 {
                        axis.next_power_of_two().max(4)
                    } else {
                        kb
                    };
                    if kb > axis.next_power_of_two().max(4) {
                        continue;
                    }
                    for rt in [8u64, 4, 2, 1] {
                        if !b.is_multiple_of(rt) {
                            continue;
                        }
                        let groups = b / rt;
                        if !threads.is_multiple_of(groups) {
                            continue;
                        }
                        let g = threads / groups;
                        if !g.is_power_of_two() || g > gpu.warp || g > kb || g < 4 {
                            continue;
                        }
                        if let Some((fb, fk, fr, _)) = forced
                            && (fb != b || fk != kb || (fr > 0 && fr != rt))
                        {
                            continue;
                        }
                        for sp in [1u64, 2, 4, 8, 16] {
                            // R3: a split of the windows where no mask skips them
                            // (`split-kv` off: one block a row block).
                            let nbs = axis.div_ceil(kb);
                            if sp > 1
                                && (sg.guard.as_ref().is_some_and(|g| g.skip)
                                    || !rungs.split_kv
                                    || !nbs.is_multiple_of(sp))
                            {
                                continue;
                            }
                            if let Some((_, _, _, fs)) = forced
                                && fs > 0
                                && fs != sp
                            {
                                continue;
                            }
                            // `stream-mma`: the instruction's layout — two
                            // rows and a quad's columns a thread, a warp's
                            // 16 rows, whole 16-key `k`-steps a window.
                            let frag = rt == 2
                                && g == 4
                                && b.is_multiple_of(MMA_M)
                                && kb.is_multiple_of(MMA_K);
                            let fmts: &[Option<crate::profile::MmaFmt>] = match (mma, frag) {
                                (Some(f), true) => &[Some(f), None][..],
                                _ => &[None][..],
                            };
                            for &fmt in fmts {
                                let st = Stream {
                                    group: sg.clone(),
                                    b,
                                    kb,
                                    rt,
                                    g,
                                    threads,
                                    cp,
                                    ns: (0, 0),
                                    s: sp,
                                    rows_of: grp.members.iter().map(|m| m.row_of.clone()).collect(),
                                    mma: fmt,
                                };
                                let lay = st.layout(members);
                                if lay.total > gpu.smem_per_block
                                    || st.regs(members) > gpu.regs_per_thread.min(232)
                                {
                                    continue;
                                }
                                let ns = match forced {
                                    Some(_) => 0.0,
                                    None => {
                                        match self.stream_ns(&st, members, rows, gmem, lay.total) {
                                            Some(x) => x,
                                            None => {
                                                why = "unpriced";
                                                continue;
                                            }
                                        }
                                    }
                                };
                                why = "cost";
                                if forced.is_some() && fmt.is_some() && forced_mma.is_none() {
                                    forced_mma = Some(st.clone());
                                }
                                if best.as_ref().is_none_or(|x| ns < x.1) {
                                    best = Some((st, ns));
                                }
                            }
                        }
                    }
                }
            }
        }
        let (mut st, ns) = best.ok_or(why)?;
        if forced.is_some() {
            return Ok(Some(forced_mma.unwrap_or(st)));
        }
        let split = self.split_ns(members, rows, gmem).ok_or("unpriced")?;
        st.ns = (ns.round() as u64, split.round() as u64);
        self.stream_priced.set(Some((st.ns.0, st.ns.1)));
        if ns >= split {
            return Err("cost");
        }
        Ok(Some(st))
    }

    /// **The streamed unit's time (R1)** on the part's plan rates: one
    /// launch; the products (`2·b·kb·(d + c)` flops a window and row block,
    /// each axis cell's `exp` path as [`STREAM_CELL_FLOPS`]) at the shipped
    /// FFMA tile's rate, the grid's waves charging a partial last wave its
    /// busiest SM; each block's window loads from L2 (`K`/`V` re-read by every
    /// row block) and the unique operand and output bytes at the rate of
    /// where they live (`residence_plan`); and each window's barriers, a
    /// block's dependent chain the other resident blocks hide only in part:
    /// `nb · waves · 4 · bar_ns`.
    fn stream_ns(
        &self,
        st: &Stream,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        smem: u64,
    ) -> Option<f64> {
        let (gpu, d) = self.profile.plan_rates()?;
        if d.flops_per_ns == 0 {
            return None;
        }
        let w = st.group.axis;
        // R3: each block walks its slice's windows.
        let nb = st.nb() / st.s;
        let blocks = rows / st.b * st.s;
        let per_sm = st.per_sm(gpu, members, smem);
        // Latency hiding: an SM whose resident warps leave a scheduler fewer
        // than two to switch between stalls on every load and barrier — such
        // a geometry is not priced (2 warps an SM ran the 2048-token prefill
        // at 2.9× its 8-warp geometry, `flash-attention.md`).
        if per_sm * st.threads / gpu.warp.max(1) < 2 * gpu.schedulers.max(1) {
            return None;
        }
        let slots = gpu.sms * per_sm;
        let waves = blocks.div_ceil(slots);
        let (mut fma, mut win_bytes, mut cells) = (0u64, 0u64, 0u64);
        for (k, mb) in members.iter().enumerate() {
            match (&st.group.roles[k], &mb.kind) {
                (StreamRole::Produce, UKind::Tile { site }) => {
                    fma += site.k;
                    win_bytes += st.b * site.k * 4 + nb * st.kb * site.k * 4;
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    fma += site.c;
                    win_bytes += nb * st.kb * site.c * 4;
                }
                (StreamRole::Cell { .. }, _) => cells += 1,
                _ => {}
            }
        }
        // F5: the windows the blocks visit (each block its band of windows
        // its rows see, `visible_band`'s rule on the host).
        let seen = st.visited(rows);
        let flops =
            ((2 * rows * w * fma + rows * w * cells * STREAM_CELL_FLOPS) as f64 * seen) as u64;
        let win_bytes = (win_bytes as f64 * seen) as u64;
        // The waves: a partial last wave runs at its busiest SM's pace.
        let fill = (waves * slots) as f64 / blocks as f64;
        // `stream-mma`: the products at the matrix product's peak, at the
        // shipped FFMA tile's fraction of its own peak.
        let rate = match st
            .mma
            .and_then(|f| gpu.caps.matrix_op(f, crate::profile::Home::Regs))
        {
            Some(op) => {
                let frac = d.flops_per_ns as f64 / gpu.ffma_flops_per_ns().max(1) as f64;
                (gpu.matrix_flops_per_ns(&op) as f64 * frac).max(d.flops_per_ns as f64)
            }
            None => d.flops_per_ns as f64,
        };
        let fma_ns = flops as f64 / rate * fill;
        // The products' issue on one SM: each window's warp FFMA at the
        // part's FP32 lanes (at the shipped tile's fraction of peak) against
        // its shared-memory wavefronts (one 128-byte wavefront a clock, the
        // ISA's 32 banks × 4 B): a lane group's row operand a broadcast, its
        // `g` lanes' distinct column rows `g · 16` bytes a 16-byte load.
        let warps = st.threads / gpu.warp.max(1);
        let wf = |lanes: u64| (lanes * 16).div_ceil(gpu::GPU_WAVEFRONT_BYTES);
        // And their shared loads' issue: a lane group's row operand and its
        // window rows, a 16-byte load each (4 elements of `k`), at the part's
        // measured LSU issue rate (`lsu_ipc_pm`): the 2048-token prefill's
        // geometries rank by it (`flash-attention.md`).
        let (mut ffma, mut lds, mut lsu) = (0u64, 0u64, 0u64);
        // sweep-attention-2 `stream-mma`: the products' flops a window on the
        // matrix product, and the register packs (`cvt`) of their operands.
        let mma_op = st
            .mma
            .and_then(|f| gpu.caps.matrix_op(f, crate::profile::Home::Regs));
        let (mut mma_flops, mut packs) = (0u64, 0u64);
        for (k, mb) in members.iter().enumerate() {
            match (&st.group.roles[k], &mb.kind) {
                // A warp's `b` fragments a window: two 8-byte loads an `n`
                // tile and `k`-step (two wavefronts each), packed in pairs;
                // its `a` fragments are the registers loaded once.
                (StreamRole::Produce, UKind::Tile { site }) if mma_op.is_some() => {
                    mma_flops += 2 * st.b * st.kb * site.k;
                    let loads = warps * (site.k / MMA_K) * (st.kb / MMA_N) * 2;
                    lds += 2 * loads;
                    lsu += loads;
                    packs += loads * gpu.warp;
                }
                // Its `b` fragments: four 4-byte loads an `n` tile and
                // `k`-step (one wavefront each); its `a` fragments are the
                // producer's cells, packed.
                (StreamRole::Consume { .. }, UKind::Tile { site }) if mma_op.is_some() => {
                    mma_flops += 2 * st.b * st.kb * site.c;
                    let loads = warps * (st.kb / MMA_K) * (site.c / MMA_N) * 4;
                    lds += loads;
                    lsu += loads;
                    packs += (loads / 2 + warps * (st.kb / MMA_K) * 4) * gpu.warp;
                }
                (StreamRole::Produce, UKind::Tile { site }) => {
                    ffma += st.b * st.kb * site.k / gpu.warp.max(1);
                    lds += warps * site.k.div_ceil(4) * (st.rt + st.ct() * wf(st.g));
                    let v = if site.k.is_multiple_of(4) { 1 } else { 4 };
                    lsu += warps * site.k.div_ceil(4) * (st.rt + st.ct()) * v;
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    ffma += st.b * st.kb * site.c / gpu.warp.max(1);
                    let (oc, contig) = st.oc(site.c);
                    lds += warps * (st.kb / 4) * (st.rt + 4 * oc.div_ceil(4) * wf(st.g));
                    let v = if contig { oc.div_ceil(4) } else { oc };
                    lsu += warps * (st.kb / 4) * (st.rt + 4 * v);
                }
                _ => {}
            }
        }
        let lsu_ipc = (d.lsu_ipc_pm.max(1) as f64 / 1000.0).min(4.0);
        // The shipped tile's fraction of FFMA peak: the stalls an issue-bound
        // loop of products and loads pays beside its issue.
        let peak = gpu.ffma_flops_per_ns().max(1) as f64;
        let eff = (d.flops_per_ns as f64 / peak).clamp(0.05, 1.0);
        // Every thread instruction a window issues beside the products: each
        // axis cell's path ([`STREAM_CELL_INSTR`], its `exp`), and each row's
        // window work a thread repeats — the butterfly of its extreme
        // (`log2 g` shuffle steps), each rescale's `exp` ([`STREAM_ROW_INSTR`])
        // — at the part's FP32 lanes an SM issues a clock.
        let (mut cells_w, mut rescales) = (0u64, 0u64);
        for r in &st.group.roles {
            match r {
                StreamRole::Cell { .. } => cells_w += 1,
                StreamRole::Fold { laws } | StreamRole::Consume { laws } => {
                    rescales += laws.iter().filter(|l| l.law == DepLaw::Rescale).count() as u64
                }
                _ => {}
            }
        }
        let steps = u64::from(st.g.max(2).ilog2());
        let thr = ffma * gpu.warp
            + packs
            + st.b * st.kb * cells_w * STREAM_CELL_INSTR
            + st.threads
                * (st.rt * (4 * steps + STREAM_ROW_INSTR * rescales.min(1)) + STREAM_ROW_INSTR);
        let issue = thr as f64 / gpu.fp32_lanes.max(1) as f64;
        // A window's four barriers stall its block; the other resident
        // blocks' issue hides them in part.
        let bar_cycles = 4.0 * gpu.bar_ns() as f64 * gpu.clock_khz as f64 / 1e6 / per_sm as f64;
        // `stream-mma`: the products' issue on the SM's matrix pipe at the
        // part's rate for the instruction (`calib.cu matrix`).
        let mma_cycles = match &mma_op {
            Some(op) => mma_flops as f64 / gpu.matrix_flops_per_clk(op).max(1) as f64,
            None => 0.0,
        };
        let cycles = issue
            .max(lds as f64)
            .max(lsu as f64 / lsu_ipc)
            .max(mma_cycles)
            / eff
            + bar_cycles;
        // Every resident block of an SM shares its pipes: a wave of `per_sm`.
        let windows = (blocks as f64 / gpu.sms as f64).ceil() * nb as f64 * seen;
        let pipe_ns = cycles * windows * 1e6 / gpu.clock_khz.max(1) as f64;
        let fma_ns = fma_ns.max(pipe_ns);
        let l2_ns = (blocks * win_bytes) as f64 / d.l2_bytes_per_ns.max(1) as f64;
        let res = self.ir.residence_plan(self.f);
        let rate = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let bytes = |o: ObjectId| self.array_bytes(&self.obj_ty(o)) as f64;
        let mem_ns: f64 = gmem.iter().map(|&o| bytes(o) / rate(o)).sum::<f64>()
            + members
                .iter()
                .filter(|mb| mb.store)
                .map(|mb| bytes(mb.out) / rate(mb.out))
                .sum::<f64>();
        // The barriers are each window's (in its cycles above).
        let chain = 0.0;
        // R3: the row block's last block reads the slices' partials back
        // (from L2, the latency of the ticket's fence and atomic first).
        let combine = match st.s {
            1 => 0.0,
            sp => {
                let per_row: u64 = members
                    .iter()
                    .enumerate()
                    .map(|(k, mb)| match (&st.group.roles[k], &mb.kind) {
                        (StreamRole::Consume { .. }, UKind::Tile { site }) => site.c,
                        (StreamRole::Fold { .. }, _) => 1,
                        _ => 0,
                    })
                    .sum();
                let sm_rate = (d.l2_bytes_per_ns as f64 / gpu.sms.max(1) as f64).max(1.0);
                2.0 * d.lat_ns as f64 + (sp * st.b * per_row * 4) as f64 / sm_rate
            }
        };
        let launch = self
            .profile
            .op_ns(f64::MAX, 0)
            .unwrap_or(d.launch_ns as f64);
        // Each wave's blocks wait their first windows' loads (the prologue
        // has nothing to overlap) and, split, the ticket's fence and atomic:
        // one loaded latency each (`lat_ns`).
        let fixed = (waves * (1 + u64::from(st.s > 1))) as f64 * d.lat_ns as f64;
        Some(launch + fma_ns.max(l2_ns).max(mem_ns) + chain + combine + fixed)
    }

    /// The split realization's time: each tile site's best candidate plan,
    /// the members between tiles as one launch a segment reading and writing
    /// the axis-wide rows at the rate of where they live — `block_cost`'s
    /// split side, independent of any block.
    fn split_ns(&self, members: &[UMember], rows: u64, gmem: &[ObjectId]) -> Option<f64> {
        let (gpu, d) = self.profile.plan_rates()?;
        let free = self.order == crate::Order::Free;
        let res = self.ir.residence_plan(self.f);
        let at = |o: ObjectId| match res.at(o, gpu.l2_bytes) {
            mapal_ir::Residence::Dram => d.dram_bytes_per_ns.max(1) as f64,
            _ => d.l2_bytes_per_ns.max(1) as f64,
        };
        let roof = |f: u64, by_ns: f64| (f as f64 / d.flops_per_ns.max(1) as f64).max(by_ns);
        let launch = |prev: f64| {
            self.profile
                .op_ns(prev, 0)
                .map_or(d.launch_ns as f64, |x| x.round())
        };
        let seg = segments(members);
        let e = 4u64;
        let mut split = 0f64;
        let mut seg_cost: Vec<(u64, f64)> = vec![(0, 0.0); seg.last().map_or(0, |s| s + 1)];
        for (k, mb) in members.iter().enumerate() {
            match &mb.kind {
                UKind::Tile { site } => {
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
                    // Its stored rows cross memory for the next segment.
                    // Its own launch's first loads wait a loaded latency,
                    // as each wave of the streamed unit's (`stream_ns`).
                    split += own + (rows * mb.w * e) as f64 / at(mb.out) + d.lat_ns as f64;
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
                            USrc::Gmem(g) => Some((work * e) as f64 / at(gmem[*g])),
                            _ => None,
                        })
                        .sum();
                    let crossed: f64 = mb
                        .srcs
                        .iter()
                        .filter_map(|s| match s {
                            USrc::Member { k: p, .. } if seg[*p] != seg[k] => {
                                let pm = &members[*p];
                                Some((rows * pm.w * e) as f64 / at(pm.out))
                            }
                            _ => None,
                        })
                        .sum();
                    let read_later = members.iter().enumerate().any(|(j, r)| {
                        seg[j] != seg[k]
                            && r.srcs
                                .iter()
                                .any(|s| matches!(s, USrc::Member { k: p, .. } if *p == k))
                    });
                    let written = match read_later {
                        true => (rows * mb.w * e) as f64 / at(mb.out),
                        false => 0.0,
                    };
                    let x = &mut seg_cost[seg[k]];
                    x.0 += work;
                    x.1 += gm + crossed + written;
                }
            }
        }
        let mut prev = f64::MAX;
        for &(f, by) in seg_cost.iter().filter(|x| x.0 > 0) {
            split += launch(prev) + roof(f, by) + d.lat_ns as f64;
            prev = roof(f, by);
        }
        Some(split)
    }
}

/// The flops one axis cell's evaluation is priced at against the products:
/// the default face's `exp` (22 SASS on the RTX 5080, `exp.md`) and its
/// operands' few.
pub(crate) const STREAM_CELL_FLOPS: u64 = 48;

/// The thread instructions one axis cell's path issues beside the products
/// (its `exp`'s 22 SASS on the RTX 5080, `exp.md`, its fold steps and its
/// row operand's store).
pub(crate) const STREAM_CELL_INSTR: u64 = 24;

/// The thread instructions one row's rescale (an `exp` and its select), or a
/// window's loop and waits, issue.
pub(crate) const STREAM_ROW_INSTR: u64 = 30;

/// A member's window values, per thread cell `y·ct + z`.
type Vals = Vec<String>;

impl UnitGen<'_> {
    /// **The streamed kernel body** (module doc). `%i` is the block's first
    /// row, `%l` the thread.
    pub(super) fn stream_body(&mut self, st: &Stream) {
        let u = self.u;
        let n = u.members.len();
        let roles = st.group.roles.clone();
        let lay = st.layout(&u.members);
        let (kb, rt, g, threads) = (st.kb, st.rt, st.g, st.threads);
        let ct = st.ct();
        let w = st.group.axis;
        let nb = st.nb();
        let tails = !w.is_multiple_of(kb);
        let cp = st.cp;
        // The thread's lane in its row group, and its rows.
        let x = self.op("urem", "%l", g);
        // `stream-mma`: the accumulator's layout — warp `w`'s rows `16w +
        // l%32÷4` and `+ 8`, the quad's lane `x` (`gq` the fragment's `n`).
        let (rloc, gq): (Vec<String>, String) = match st.mma {
            Some(_) => {
                let w = self.op("udiv", "%l", 32);
                let lw = self.op("urem", "%l", 32);
                let gq = self.op("udiv", &lw, 4);
                let r0 = self.op("mul", &w, MMA_M);
                let r0 = self.op("add", &r0, &gq);
                let r1 = self.op("add", &r0, 8);
                (vec![r0, r1], gq)
            }
            None => {
                let rg = self.op("udiv", "%l", g);
                let rl0 = self.op("mul", &rg, rt);
                (
                    (0..rt).map(|y| self.op("add", &rl0, y)).collect(),
                    String::new(),
                )
            }
        };
        let grow: Vec<String> = rloc.iter().map(|r| self.op("add", "%i", r)).collect();
        // Which folds a cell reads as a rescale's shift: their window
        // extreme is combined across the row group every window.
        let shift: Vec<bool> = (0..n)
            .map(|v| {
                roles.iter().any(|r| match r {
                    StreamRole::Cell { subst } => subst
                        .iter()
                        .any(|&(_, x, law)| x == v && law == DepLaw::Rescale),
                    _ => false,
                })
            })
            .collect();
        // State: each producer's window, each consumer's accumulators, each
        // fold's state, as allocas the optimizer keeps in registers.
        for (k, mb) in u.members.iter().enumerate() {
            match (&roles[k], &mb.kind) {
                (StreamRole::Produce, _) => {
                    for c in 0..rt * ct {
                        self.entry
                            .push_str(&format!("  %S{k}_{c} = alloca float\n"));
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    for c in 0..rt * st.oc(site.c).0 {
                        self.entry
                            .push_str(&format!("  %O{k}_{c} = alloca float\n"));
                    }
                }
                (StreamRole::Fold { .. }, _) => {
                    for y in 0..rt {
                        self.entry
                            .push_str(&format!("  %F{k}_{y} = alloca float\n"));
                    }
                }
                _ => {}
            }
        }
        // Initial states: the consumers' and folds' step identities.
        for (k, mb) in u.members.iter().enumerate() {
            match (&roles[k], &mb.kind) {
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    for c in 0..rt * st.oc(site.c).0 {
                        self.line(format!("store float -0.0, ptr %O{k}_{c}"));
                    }
                }
                (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                    let id = step_identity(&site.op);
                    for y in 0..rt {
                        self.line(format!("store float {id}, ptr %F{k}_{y}"));
                    }
                }
                _ => {}
            }
        }
        // flash-attention F5/F6: the windows the block's rows see, `[tlo,
        // tlo + band)` (rotated; a block that sees none walks them all).
        let skip = st.skip();
        match &skip {
            Some(gd) => self.visible_band(st, &gd.form),
            // R3: this block's slice of the windows.
            None if st.s > 1 => {
                self.line(format!("%tlo = mul i64 %bz, {}", nb / st.s));
                self.line(format!("%band = add i64 0, {}", nb / st.s));
            }
            None => {
                self.line("%tlo = add i64 0, 0");
                self.line(format!("%band = add i64 0, {nb}"));
            }
        }
        self.line(format!("%j00 = mul i64 %tlo, {kb}"));
        // The prologue: the producers' row panels and first windows, then
        // the consumers' first windows. `stream-mma`: each producer's row
        // operand into its `a` fragments, once.
        let mut afrag: Vec<Vec<[String; 4]>> = vec![Vec::new(); n];
        for k in 0..n {
            if roles[k] == StreamRole::Produce {
                match st.mma {
                    Some(_) => afrag[k] = self.mma_rows(st, k, &grow, &x),
                    None => self.stage_panel(st, &lay, k),
                }
                self.stage_window(st, &lay, k, "%j00", true);
            }
        }
        self.commit(cp);
        for k in 0..n {
            if matches!(roles[k], StreamRole::Consume { .. }) {
                self.stage_window(st, &lay, k, "%j00", false);
            }
        }
        self.commit(cp);
        // The window loop.
        let pre = self.cur.clone();
        self.line("br label %W_h");
        self.label("W_h");
        self.line(format!("%wt = phi i64 [0, %{pre}], [%wtn, %W_n]"));
        // Window `(tlo + wt) mod nb`; the next one's first key.
        self.line("%wtr = add i64 %tlo, %wt");
        self.line(format!("%wtw = urem i64 %wtr, {nb}"));
        self.line(format!("%j0 = mul i64 %wtw, {kb}"));
        self.line("%wtn = add i64 %wt, 1");
        self.line("%wnr = add i64 %tlo, %wtn");
        self.line(format!("%wnw = urem i64 %wnr, {nb}"));
        self.line(format!("%jn = mul i64 %wnw, {kb}"));
        // The windows this block walks: all `nb`, or its slice's.
        let walk = if st.s > 1 { nb / st.s } else { nb };
        self.line(format!("%wmore = icmp ult i64 %wtn, {walk}"));
        if !cp {
            // Synchronous loads: this window's operands now.
            let pre_k = (0..n).any(|k| roles[k] == StreamRole::Produce);
            if pre_k {
                let c = self.tmp();
                self.line(format!("{c} = icmp ugt i64 %wt, 0"));
                self.cond("Wk", &c, |gg| {
                    for k in 0..n {
                        if roles[k] == StreamRole::Produce {
                            gg.stage_window(st, &lay, k, "%j0", true);
                        }
                    }
                });
            }
        }
        self.wait(cp, 1);
        self.barrier();
        // Each cell's column and validity (`stream-mma`: the accumulator's,
        // `8·(z÷2) + 2x + z%2`).
        let jl: Vec<String> = match st.mma {
            Some(_) => {
                let x2 = self.op("mul", &x, 2);
                (0..ct)
                    .map(|z| self.op("add", &x2, MMA_N * (z / 2) + z % 2))
                    .collect()
            }
            None => (0..ct).map(|z| self.op("add", &x, g * z)).collect(),
        };
        let jg: Vec<String> = jl.iter().map(|j| self.op("add", "%j0", j)).collect();
        let valid: Vec<Option<String>> = jg
            .iter()
            .map(|j| {
                tails.then(|| {
                    let c = self.tmp();
                    self.line(format!("{c} = icmp ult i64 {j}, {w}"));
                    c
                })
            })
            .collect();
        // A cell index inside the axis (a row's last cell past its end: the
        // cell's value is dropped, its reads stay in bounds).
        let jc: Vec<String> = jg
            .iter()
            .zip(&valid)
            .map(|(j, v)| match v {
                Some(v) => {
                    let s = self.tmp();
                    self.line(format!("{s} = select i1 {v}, i64 {j}, i64 {}", w - 1));
                    s
                }
                None => j.clone(),
            })
            .collect();
        // The producers' windows.
        let mut vals: Vec<Vals> = vec![Vec::new(); n];
        for k in 0..n {
            if roles[k] != StreamRole::Produce {
                continue;
            }
            vals[k] = match st.mma {
                Some(_) => self.produce_mma(st, &lay, k, &afrag[k], &x, &gq),
                None => self.produce(st, &lay, k, &rloc, &jl),
            };
            // `tile-post`: each cell through the site's body after its fold
            // (a scale, a mask), at its own cell index.
            let UKind::Tile { site } = &u.members[k].kind else {
                unreachable!("a producer is a tile")
            };
            if site.post.is_some() {
                let out = Addr::of(site, Operand::Out);
                for y in 0..rt as usize {
                    let row = self.member_row(st, k, &grow[y]);
                    for z in 0..ct as usize {
                        let t = self.addr_at(&out, &row, &jc[z], "0");
                        let acc = vals[k][y * ct as usize + z].clone();
                        vals[k][y * ct as usize + z] = self.post_call(k, &t, &acc, &[]);
                    }
                }
            }
        }
        if cp && nb > 1 {
            // Every thread is past its reads of this window's panels: the
            // next window lands in them while the rest of this one runs.
            self.barrier();
            self.cond("Wk", "%wmore", |gg| {
                for k in 0..n {
                    if roles[k] == StreamRole::Produce {
                        gg.stage_window(st, &lay, k, "%jn", true);
                    }
                }
            });
        }
        // One group a step whatever it holds: the waits count them.
        self.commit(cp);
        // Per row: each shift fold's previous and new extreme this window.
        let mut ext: Vec<Vec<(String, String)>> = vec![Vec::new(); n];
        let mut first_consume = true;
        let last_consume = (0..n)
            .rev()
            .find(|&k| matches!(roles[k], StreamRole::Consume { .. }));
        for k in 0..n {
            let mb = &u.members[k];
            match (&roles[k], &mb.kind) {
                (StreamRole::Produce, UKind::Tile { site }) => {
                    if mb.store {
                        let out = Addr::of(site, Operand::Out);
                        for y in 0..rt as usize {
                            let row = self.member_row(st, k, &grow[y]);
                            for z in 0..ct as usize {
                                let t = self.addr_at(&out, &row, &jc[z], "0");
                                let v = vals[k][y * ct as usize + z].clone();
                                match &valid[z] {
                                    Some(c) => {
                                        let c = c.clone();
                                        self.cond(&format!("Ps{k}_{y}_{z}"), &c, |gg| {
                                            gg.store(k, &v, &t)
                                        });
                                    }
                                    None => self.store(k, &v, &t),
                                }
                            }
                        }
                    }
                }
                (StreamRole::Cell { subst }, _) => {
                    let subst = subst.clone();
                    let mut out = Vec::new();
                    for y in 0..rt as usize {
                        for z in 0..ct as usize {
                            let r = self.op("mul", &grow[y], w);
                            let t = self.op("add", &r, &jc[z]);
                            for (s, src) in mb.srcs.iter().enumerate() {
                                let USrc::Member { k: p, .. } = src else {
                                    continue;
                                };
                                let lit = match subst.iter().find(|x| x.0 == s as u32) {
                                    Some(&(_, v, DepLaw::Rescale)) => {
                                        let m = ext[v][y].1.clone();
                                        let id = match &u.members[v].kind {
                                            UKind::Fold { site, .. } => step_identity(&site.op),
                                            _ => unreachable!("a shift is a fold"),
                                        };
                                        let c = self.tmp();
                                        self.line(format!("{c} = fcmp oeq float {m}, {id}"));
                                        let sv = self.tmp();
                                        self.line(format!(
                                            "{sv} = select i1 {c}, float 0.0, float {m}"
                                        ));
                                        sv
                                    }
                                    Some(&(_, _, DepLaw::Factor { .. })) => "1.0".into(),
                                    Some(&(_, _, DepLaw::Shift { sub: false })) => "-0.0".into(),
                                    Some(&(_, _, _)) => "0.0".into(),
                                    None => vals[*p][y * ct as usize + z].clone(),
                                };
                                self.subst.push((k, s as u32, lit));
                            }
                            let v = self.cell(k, &t);
                            self.subst.clear();
                            out.push(v);
                        }
                    }
                    vals[k] = out;
                }
                (StreamRole::Fold { laws }, UKind::Fold { site, .. }) => {
                    let laws = laws.clone();
                    let (op, id) = (site.op, step_identity(&site.op));
                    let USrc::Member { k: p, .. } = mb.srcs[0] else {
                        unreachable!("an axis fold reads a member")
                    };
                    for y in 0..rt as usize {
                        let mut part = id.clone();
                        for z in 0..ct as usize {
                            let mut v = vals[p][y * ct as usize + z].clone();
                            if let Some(c) = &valid[z] {
                                let s = self.tmp();
                                self.line(format!("{s} = select i1 {c}, float {v}, float {id}"));
                                v = s;
                            }
                            let o = self.tmp();
                            self.step(&op, "float", &part, &v, &o);
                            part = o;
                        }
                        let fy = format!("%F{k}_{y}");
                        let old = self.tmp();
                        self.line(format!("{old} = load float, ptr {fy}"));
                        if shift[k] {
                            let tot = self.butterfly(&part, g, op);
                            let new = self.tmp();
                            self.step(&op, "float", &old, &tot, &new);
                            self.line(format!("store float {new}, ptr {fy}"));
                            ext[k].push((old, new));
                        } else {
                            let mut s = old;
                            for l in &laws {
                                if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                                    let (mo, mn) = ext[l.value][y].clone();
                                    let a = self.alpha(arg, &mo, &mn);
                                    let z = self.tmp();
                                    self.line(format!("{z} = fmul float {s}, {a}"));
                                    s = z;
                                }
                            }
                            let new = self.tmp();
                            self.step(&op, "float", &s, &part, &new);
                            self.line(format!("store float {new}, ptr {fy}"));
                        }
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let laws = laws.clone();
                    let site = site.clone();
                    let USrc::Member { k: p, .. } = mb.srcs[0] else {
                        unreachable!("a consumer reads its row operand from a member")
                    };
                    // Its row operand's window into shared memory
                    // (`stream-mma`: the cells as they stand, its `a`
                    // fragments).
                    let mut pv: Vec<String> = Vec::new();
                    for y in 0..rt as usize {
                        let ra = lay.a[k].map(|(aoff, ap)| (aoff, self.op("mul", &rloc[y], ap)));
                        for z in 0..ct as usize {
                            let mut v = vals[p][y * ct as usize + z].clone();
                            if let Some(c) = &valid[z] {
                                let s = self.tmp();
                                self.line(format!("{s} = select i1 {c}, float {v}, float 0.0"));
                                v = s;
                            }
                            match &ra {
                                Some((aoff, ra)) => {
                                    let at = self.op("add", ra, &jl[z]);
                                    let q = self.smem(*aoff, "float", &at);
                                    self.line(format!("store float {v}, ptr addrspace(3) {q}"));
                                }
                                None => pv.push(v),
                            }
                        }
                    }
                    if first_consume {
                        if !cp {
                            for j in 0..n {
                                if matches!(roles[j], StreamRole::Consume { .. }) {
                                    self.stage_window(st, &lay, j, "%j0", false);
                                }
                            }
                        }
                        self.wait(cp, 1);
                        first_consume = false;
                    }
                    self.barrier();
                    // The window's rescale, then its products.
                    let (oc, _) = st.oc(site.c);
                    for y in 0..rt as usize {
                        let mut alpha: Option<String> = None;
                        for l in &laws {
                            if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                                let (mo, mn) = ext[l.value][y].clone();
                                let a = self.alpha(arg, &mo, &mn);
                                alpha = Some(match alpha {
                                    None => a,
                                    Some(x) => {
                                        let z = self.tmp();
                                        self.line(format!("{z} = fmul float {x}, {a}"));
                                        z
                                    }
                                });
                            }
                        }
                        if let Some(a) = alpha {
                            for q in 0..oc {
                                let o = format!("%O{k}_{}", y as u64 * oc + q);
                                let v = self.tmp();
                                self.line(format!("{v} = load float, ptr {o}"));
                                let z = self.tmp();
                                self.line(format!("{z} = fmul float {v}, {a}"));
                                self.line(format!("store float {z}, ptr {o}"));
                            }
                        }
                    }
                    match st.mma {
                        Some(_) => self.consume_mma(st, &lay, k, &site, &pv, &x, &gq),
                        None => self.consume(st, &lay, k, &site, &rloc, &x),
                    }
                    if Some(k) == last_consume {
                        self.barrier();
                        if cp && nb > 1 {
                            self.cond("Wv", "%wmore", |gg| {
                                for j in 0..n {
                                    if matches!(roles[j], StreamRole::Consume { .. }) {
                                        gg.stage_window(st, &lay, j, "%jn", false);
                                    }
                                }
                            });
                        }
                        self.commit(cp);
                    }
                }
                _ => unreachable!("a streamed member has its role"),
            }
        }
        self.line("br label %W_l");
        self.label("W_l");
        match &skip {
            // F6: after the visible windows, the skipped ones' terms are each
            // fold's identity at the final values — or they run.
            Some(gd) => {
                self.line("%atb = icmp eq i64 %wtn, %band");
                self.line("br i1 %atb, label %W_c, label %W_n");
                self.label("W_c");
                let pass = self.absorbed(st, &lay, gd, &x, &grow, &shift);
                self.line(format!("br i1 {pass}, label %W_e, label %W_n"));
            }
            None => self.line("br label %W_n"),
        }
        self.label("W_n");
        self.line("br i1 %wmore, label %W_h, label %W_e");
        self.label("W_e");
        // A skipped window's prefetch lands before the block leaves.
        self.wait(cp, 0);
        // R3: a slice's partial states to the workspace; the row block's last
        // block reads every slice's back, combined into its own states.
        if st.s > 1 {
            self.split_combine(st, &lay, &x, &grow, &shift);
        }
        // After the last window: each fold's lanes combined, its laws, its
        // seed; each consumer's laws and seed, stored.
        let fin = self.finals(st, &shift);
        let lane0 = self.tmp();
        self.line(format!("{lane0} = icmp eq i64 {x}, 0"));
        for k in 0..n {
            let mb = &u.members[k];
            match (&roles[k], &mb.kind) {
                (StreamRole::Fold { .. }, UKind::Fold { .. }) => {
                    for y in 0..rt as usize {
                        if mb.store {
                            let row = grow[y].clone();
                            let r2 = fin[k][y].clone();
                            self.cond(&format!("Fs{k}_{y}"), &lane0, |gg| gg.store(k, &r2, &row));
                        }
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let laws = laws.clone();
                    let op = mapal_ir::RowFoldOp::Arith {
                        op: Operation::Add,
                        acc_first: site.add_acc_first,
                    };
                    let seed = const_literal(&site.seed);
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt as usize {
                        for q in 0..oc {
                            let o = self.tmp();
                            self.line(format!("{o} = load float, ptr %O{k}_{}", y as u64 * oc + q));
                            let v = self.laws_after(&laws, &fin, y, &o, k);
                            let res = self.tmp();
                            self.step(&op, "float", &seed, &v, &res);
                            let col = match (st.mma, contig) {
                                (Some(_), _) => {
                                    let x2 = self.op("mul", &x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                (None, true) => {
                                    let c0 = self.op("mul", &x, oc);
                                    self.op("add", &c0, q)
                                }
                                (None, false) => self.op("add", &x, g * q),
                            };
                            // Its own row's output cell; the epilogue there.
                            let row = self.member_row(st, k, &grow[y]);
                            let out = Addr::of(site, Operand::Out);
                            let at = self.addr_at(&out, &row, &col, "0");
                            let res = match &site.post {
                                Some(post) => {
                                    let per_row: Vec<(u32, String)> = post
                                        .reads
                                        .iter()
                                        .zip(&mb.srcs[2..])
                                        .filter_map(|(&slot, x)| match x {
                                            USrc::Member { k: v, .. } => {
                                                Some((slot, fin[*v][y].clone()))
                                            }
                                            _ => None,
                                        })
                                        .collect();
                                    self.post_call(k, &at, &res, &per_row)
                                }
                                None => res,
                            };
                            if contig || site.c.is_multiple_of(g) || st.mma.is_some() {
                                self.store(k, &res, &at);
                            } else {
                                let c = self.tmp();
                                self.line(format!("{c} = icmp ult i64 {col}, {}", site.c));
                                self.cond(&format!("Os{k}_{y}_{q}"), &c, |gg| {
                                    gg.store(k, &res, &at)
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if st.s > 1 {
            self.line("br label %SPx");
            self.label("SPx");
        }
        let _ = threads;
    }

    /// R3 split-KV: the block's partial states — each fold's row value (its
    /// lanes combined), each consumer's columns — to the kernel's workspace
    /// (`@<kernel>.ws<k>`, slice `%bz` of row `r` at `r·s + z`); every thread
    /// fences, the block meets, its lead takes a ticket on the row block's
    /// counter (`@<kernel>.cnt`, reset by the last); a block that is not the
    /// row block's last leaves (`%SPx`). The last reads all `s` partials and
    /// combines them in slice order into its own states: a running extreme by
    /// its step, a sum by `+` with each slice's state rescaled from its own
    /// extreme to the combined one (`exp(a(m_z, m))`, ADR-0042 D7), a
    /// consumer's columns likewise — the windows' combine, across blocks.
    fn split_combine(
        &mut self,
        st: &Stream,
        lay: &Layout,
        x: &str,
        grow: &[String],
        shift: &[bool],
    ) {
        let u = self.u;
        let n = u.members.len();
        let (sp, rt, g) = (st.s, st.rt as usize, st.g);
        let name = self.name.clone();
        let rows = u.rows;
        let lane0 = self.tmp();
        self.line(format!("{lane0} = icmp eq i64 {x}, 0"));
        let ws = |gg: &mut Self, k: usize, n_el: u64, idx: &str| {
            let p = gg.tmp();
            gg.line(format!(
                "{p} = getelementptr inbounds [{n_el} x float], ptr addrspace(1) @{name}.ws{k}, i64 0, i64 {idx}"
            ));
            p
        };
        // The partials out.
        for k in 0..n {
            match (&st.group.roles[k], &u.members[k].kind) {
                (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                    for y in 0..rt {
                        let s0 = self.tmp();
                        self.line(format!("{s0} = load float, ptr %F{k}_{y}"));
                        let v = match shift[k] {
                            true => s0,
                            false => self.butterfly(&s0, g, site.op),
                        };
                        let r = self.op("mul", &grow[y], sp);
                        let at = self.op("add", &r, "%bz");
                        let p = ws(self, k, rows * sp, &at);
                        self.cond(&format!("SPw{k}_{y}"), &lane0, |gg| {
                            gg.line(format!("store float {v}, ptr addrspace(1) {p}"))
                        });
                    }
                }
                (StreamRole::Consume { .. }, UKind::Tile { site }) => {
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt {
                        let r = self.op("mul", &grow[y], sp);
                        let rz = self.op("add", &r, "%bz");
                        let base = self.op("mul", &rz, site.c);
                        for q in 0..oc {
                            let col = match st.mma {
                                Some(_) => {
                                    let x2 = self.op("mul", x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                None => self.split_col(x, q, oc, contig, g, site.c),
                            };
                            let at = self.op("add", &base, &col);
                            let v = self.tmp();
                            self.line(format!("{v} = load float, ptr %O{k}_{}", y as u64 * oc + q));
                            let p = ws(self, k, rows * sp * site.c, &at);
                            self.line(format!("store float {v}, ptr addrspace(1) {p}"));
                        }
                    }
                }
                _ => {}
            }
        }
        // The ticket.
        self.line("fence syncscope(\"device\") release");
        let f = self.smem(lay.flag, "i32", "0");
        self.barrier();
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 %l, 0"));
        let rb = self.op("udiv", "%i", st.b);
        self.cond("SPt", &t0, |gg| {
            let c = gg.tmp();
            gg.line(format!(
                "{c} = getelementptr inbounds [{} x i32], ptr addrspace(1) @{name}.cnt, i64 0, i64 {rb}",
                rows / st.b
            ));
            let old = gg.tmp();
            gg.line(format!(
                "{old} = atomicrmw add ptr addrspace(1) {c}, i32 1 syncscope(\"device\") acq_rel, align 4"
            ));
            let last = gg.tmp();
            gg.line(format!("{last} = icmp eq i32 {old}, {}", sp - 1));
            let lz = gg.tmp();
            gg.line(format!("{lz} = zext i1 {last} to i32"));
            gg.line(format!("store i32 {lz}, ptr addrspace(3) {f}"));
            // The last resets the counter for the next launch.
            gg.cond("SPr", &last, |h| {
                h.line(format!("store atomic i32 0, ptr addrspace(1) {c} syncscope(\"device\") monotonic, align 4"))
            });
        });
        self.barrier();
        let fl = self.tmp();
        self.line(format!("{fl} = load i32, ptr addrspace(3) {f}"));
        let last = self.tmp();
        self.line(format!("{last} = icmp ne i32 {fl}, 0"));
        self.line(format!("br i1 {last}, label %SPc, label %SPx"));
        self.label("SPc");
        self.line("fence syncscope(\"device\") acquire");
        // Each slice's shift folds first (the rescales' targets).
        let mut ms: Vec<Vec<Vec<String>>> = vec![Vec::new(); n];
        let mut comb: Vec<Vec<String>> = vec![Vec::new(); n];
        for k in 0..n {
            let (StreamRole::Fold { .. }, UKind::Fold { site, .. }) =
                (&st.group.roles[k], &u.members[k].kind)
            else {
                continue;
            };
            let (op, id) = (site.op, step_identity(&site.op));
            for y in 0..rt {
                let r = self.op("mul", &grow[y], sp);
                let mut parts = Vec::new();
                let mut acc = id.clone();
                for z in 0..sp {
                    let at = self.op("add", &r, z);
                    let p = ws(self, k, rows * sp, &at);
                    let v = self.tmp();
                    self.line(format!("{v} = load float, ptr addrspace(1) {p}"));
                    if shift[k] {
                        let o = self.tmp();
                        self.step(&op, "float", &acc, &v, &o);
                        acc = o;
                    }
                    parts.push(v);
                }
                ms[k].push(parts);
                comb[k].push(acc);
            }
        }
        // The rescale of slice `z`'s state to the combined shift.
        let rescale = |gg: &mut Self, laws: &[mapal_ir::StreamLaw], y: usize, z: usize, v: &str| {
            let mut v = v.to_string();
            for l in laws {
                if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                    let e = gg.exp_arg(arg, &ms[l.value][y][z], &comb[l.value][y]);
                    let t = gg.tmp();
                    gg.line(format!("{t} = fmul float {v}, {e}"));
                    v = t;
                }
            }
            v
        };
        for k in 0..n {
            match (&st.group.roles[k], &u.members[k].kind) {
                (StreamRole::Fold { laws }, UKind::Fold { site, .. }) => {
                    let (op, id) = (site.op, step_identity(&site.op));
                    for y in 0..rt {
                        let tot = match shift[k] {
                            true => comb[k][y].clone(),
                            false => {
                                let mut acc = id.clone();
                                for z in 0..sp as usize {
                                    let v = rescale(self, laws, y, z, &ms[k][y][z]);
                                    let o = self.tmp();
                                    self.step(&op, "float", &acc, &v, &o);
                                    acc = o;
                                }
                                // Lane 0 holds it, the others the identity: the
                                // finals' butterfly is the total.
                                let o = self.tmp();
                                self.line(format!(
                                    "{o} = select i1 {lane0}, float {acc}, float {id}"
                                ));
                                o
                            }
                        };
                        self.line(format!("store float {tot}, ptr %F{k}_{y}"));
                    }
                }
                (StreamRole::Consume { laws }, UKind::Tile { site }) => {
                    let (oc, contig) = st.oc(site.c);
                    for y in 0..rt {
                        let r = self.op("mul", &grow[y], sp);
                        for q in 0..oc {
                            let col = match st.mma {
                                Some(_) => {
                                    let x2 = self.op("mul", x, 2);
                                    self.op("add", &x2, MMA_N * (q / 2) + q % 2)
                                }
                                None => self.split_col(x, q, oc, contig, g, site.c),
                            };
                            let mut acc = "-0.0".to_string();
                            for z in 0..sp as usize {
                                let rz = self.op("add", &r, z as u64);
                                let b = self.op("mul", &rz, site.c);
                                let at = self.op("add", &b, &col);
                                let p = ws(self, k, rows * sp * site.c, &at);
                                let v = self.tmp();
                                self.line(format!("{v} = load float, ptr addrspace(1) {p}"));
                                let v = rescale(self, laws, y, z, &v);
                                let o = self.tmp();
                                self.line(format!("{o} = fadd float {acc}, {v}"));
                                acc = o;
                            }
                            self.line(format!(
                                "store float {acc}, ptr %O{k}_{}",
                                y as u64 * oc + q
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// A consumer column of the thread (clamped inside the row: a column
    /// past it is never stored).
    fn split_col(&mut self, x: &str, q: u64, oc: u64, contig: bool, g: u64, c: u64) -> String {
        match contig {
            true => {
                let c0 = self.op("mul", x, oc);
                self.op("add", &c0, q)
            }
            false => {
                let col = self.op("add", x, g * q);
                let ok = self.tmp();
                self.line(format!("{ok} = icmp ult i64 {col}, {c}"));
                let s = self.tmp();
                self.line(format!("{s} = select i1 {ok}, i64 {col}, i64 {}", c - 1));
                s
            }
        }
    }

    /// Each fold's final value per row of the thread: its lanes' states
    /// combined (a shift fold's state already is), its laws, its seed.
    fn finals(&mut self, st: &Stream, shift: &[bool]) -> Vec<Vec<String>> {
        let u = self.u;
        let n = u.members.len();
        let mut fin: Vec<Vec<String>> = vec![Vec::new(); n];
        for k in 0..n {
            let (StreamRole::Fold { laws }, UKind::Fold { site, .. }) =
                (&st.group.roles[k], &u.members[k].kind)
            else {
                continue;
            };
            let (op, seed) = (site.op, const_literal(&site.seed));
            let laws = laws.clone();
            for y in 0..st.rt as usize {
                let s = self.tmp();
                self.line(format!("{s} = load float, ptr %F{k}_{y}"));
                let mut tot = match shift[k] {
                    true => s,
                    false => self.butterfly(&s, st.g, op),
                };
                tot = self.laws_after(&laws, &fin, y, &tot, k);
                let res = self.tmp();
                self.step(&op, "float", &seed, &tot, &res);
                fin[k].push(res);
            }
        }
        fin
    }

    /// F5: `%tlo` and `%band` — the first window the block's rows see and
    /// how many (through the last one they see; a block that sees none: all
    /// `nb` from 0) — each window's class from the mask's two sides' ranges
    /// over the block's rows and the window's lanes ([`mapal_ir::GuardBox`]).
    fn visible_band(&mut self, st: &Stream, gb: &mapal_ir::GuardBox) {
        let (kb, nb, w) = (st.kb, st.nb(), st.group.axis);
        let r1 = self.op("add", "%i", st.b - 1);
        let side_rows = |g: &mut Self, sd: &mapal_ir::GuardSide| -> (String, String) {
            let mut lo = sd.base.to_string();
            let mut hi = sd.base.to_string();
            for d in &sd.row {
                let (a, b) = g.digit_range(d, "%i", &r1);
                lo = g.op("add", &lo, &a);
                hi = g.op("add", &hi, &b);
            }
            (lo, hi)
        };
        let (al, ah) = side_rows(self, &gb.lhs);
        let (bl, bh) = side_rows(self, &gb.rhs);
        let pre = self.cur.clone();
        self.line("br label %GS");
        self.label("GS");
        self.line(format!("%gt = phi i64 [0, %{pre}], [%gtn, %GS]"));
        self.line(format!("%glo = phi i64 [{nb}, %{pre}], [%glon, %GS]"));
        self.line(format!("%ghi = phi i64 [0, %{pre}], [%ghin, %GS]"));
        let l0 = self.op("mul", "%gt", kb);
        let l1x = self.op("add", &l0, kb - 1);
        let l1c = self.tmp();
        self.line(format!("{l1c} = icmp ult i64 {l1x}, {w}"));
        let l1 = self.tmp();
        self.line(format!("{l1} = select i1 {l1c}, i64 {l1x}, i64 {}", w - 1));
        let side_lanes = |g: &mut Self, sd: &mapal_ir::GuardSide, lo: &str, hi: &str| {
            let (mut lo, mut hi) = (lo.to_string(), hi.to_string());
            for d in &sd.lane {
                let (a, b) = g.digit_range(d, &l0, &l1);
                lo = g.op("add", &lo, &a);
                hi = g.op("add", &hi, &b);
            }
            (lo, hi)
        };
        let (a0, a1) = side_lanes(self, &gb.lhs, &al, &ah);
        let (b0, b1) = side_lanes(self, &gb.rhs, &bl, &bh);
        // Always true / always false over the box.
        let cmp = |g: &mut Self, p: &str, x: &str, y: &str| {
            let c = g.tmp();
            g.line(format!("{c} = icmp {p} i64 {x}, {y}"));
            c
        };
        let (t, f) = match gb.cmp {
            Operation::Le => (cmp(self, "ule", &a1, &b0), cmp(self, "ugt", &a0, &b1)),
            Operation::Lt => (cmp(self, "ult", &a1, &b0), cmp(self, "uge", &a0, &b1)),
            Operation::Ge => (cmp(self, "uge", &a0, &b1), cmp(self, "ult", &a1, &b0)),
            Operation::Gt => (cmp(self, "ugt", &a0, &b1), cmp(self, "ule", &a1, &b0)),
            // Equality decides no box here: every window mixed.
            _ => ("false".to_string(), "false".to_string()),
        };
        let masked = if gb.fold_when { f } else { t };
        let tmin = self.tmp();
        self.line(format!(
            "{tmin} = call i64 @llvm.umin.i64(i64 %glo, i64 %gt)"
        ));
        self.line(format!("%glon = select i1 {masked}, i64 %glo, i64 {tmin}"));
        let t1 = self.op("add", "%gt", 1);
        self.line(format!("%ghin = select i1 {masked}, i64 %ghi, i64 {t1}"));
        self.line("%gtn = add i64 %gt, 1");
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 %gtn, {nb}"));
        self.line(format!("br i1 {more}, label %GS, label %GSe"));
        self.label("GSe");
        let none = self.tmp();
        self.line(format!("{none} = icmp uge i64 %glon, {nb}"));
        self.line(format!("%tlo = select i1 {none}, i64 0, i64 %glon"));
        let span = self.op("sub", "%ghin", "%glon");
        self.line(format!("%band = select i1 {none}, i64 {nb}, i64 {span}"));
    }

    /// `[lo, hi]` of digit `d` over `x ∈ [x0, x1]` (`radix 0`: the quotient).
    fn digit_range(&mut self, d: &mapal_ir::AxisDigit, x0: &str, x1: &str) -> (String, String) {
        let q0 = self.op("udiv", x0, d.stride);
        let q1 = self.op("udiv", x1, d.stride);
        if d.radix == 0 {
            return (self.op("mul", &q0, d.coef), self.op("mul", &q1, d.coef));
        }
        let w0 = self.op("udiv", &q0, d.radix);
        let w1 = self.op("udiv", &q1, d.radix);
        let same = self.tmp();
        self.line(format!("{same} = icmp eq i64 {w0}, {w1}"));
        let m0 = self.op("urem", &q0, d.radix);
        let m1 = self.op("urem", &q1, d.radix);
        let lo = self.tmp();
        self.line(format!("{lo} = select i1 {same}, i64 {m0}, i64 0"));
        let hi = self.tmp();
        self.line(format!(
            "{hi} = select i1 {same}, i64 {m1}, i64 {}",
            d.radix - 1
        ));
        (self.op("mul", &lo, d.coef), self.op("mul", &hi, d.coef))
    }

    /// F6: does every skipped (wholly masked) window add each fold's
    /// identity? Each row's axis cells at the masked constant with the
    /// final per-row values: a running extreme's final unchanged by it, a
    /// sum's term `±0`, a consumer's row operand `±0` (its column operand is
    /// finite, `finite_plan`); the block's rows vote (one shared word).
    #[allow(clippy::too_many_arguments)]
    fn absorbed(
        &mut self,
        st: &Stream,
        lay: &Layout,
        gd: &mapal_ir::StreamGuard,
        x: &str,
        grow: &[String],
        shift: &[bool],
    ) -> String {
        let u = self.u;
        let n = u.members.len();
        let fin = self.finals(st, shift);
        let w = st.group.axis;
        let mut ok = "true".to_string();
        for y in 0..st.rt as usize {
            let mut mv: Vec<String> = vec![String::new(); n];
            mv[gd.producer] = super::const_literal(&mapal_ir::Value::F32(gd.form.masked as f32));
            let t = self.op("mul", &grow[y], w);
            for k in 0..n {
                let mb = &u.members[k];
                match (&st.group.roles[k], &mb.kind) {
                    (StreamRole::Cell { .. }, _) => {
                        for (sl, src) in mb.srcs.iter().enumerate() {
                            let USrc::Member { k: p, .. } = src else {
                                continue;
                            };
                            let v = match st.group.roles[*p] {
                                StreamRole::Fold { .. } => fin[*p][y].clone(),
                                _ => mv[*p].clone(),
                            };
                            self.subst.push((k, sl as u32, v));
                        }
                        mv[k] = self.cell(k, &t);
                        self.subst.clear();
                    }
                    (StreamRole::Fold { .. }, UKind::Fold { site, .. }) => {
                        let USrc::Member { k: p, .. } = mb.srcs[0] else {
                            continue;
                        };
                        let c = self.tmp();
                        match site.op {
                            mapal_ir::RowFoldOp::Select { .. } => {
                                let s = self.tmp();
                                self.step(&site.op, "float", &fin[k][y], &mv[p], &s);
                                self.line(format!("{c} = fcmp oeq float {s}, {}", fin[k][y]));
                            }
                            _ => self.line(format!("{c} = fcmp oeq float {}, 0.0", mv[p])),
                        }
                        ok = self.and(&ok, &c);
                    }
                    (StreamRole::Consume { .. }, _) => {
                        let USrc::Member { k: p, .. } = mb.srcs[0] else {
                            continue;
                        };
                        let c = self.tmp();
                        self.line(format!("{c} = fcmp oeq float {}, 0.0", mv[p]));
                        ok = self.and(&ok, &c);
                    }
                    _ => {}
                }
            }
        }
        // The block's vote: one shared word, cleared by any row that fails.
        let f = self.smem(lay.flag, "i32", "0");
        self.barrier();
        let t0 = self.tmp();
        self.line(format!("{t0} = icmp eq i64 %l, 0"));
        self.cond("Vz", &t0, |g| {
            g.line(format!("store i32 1, ptr addrspace(3) {f}"))
        });
        self.barrier();
        let bad = self.tmp();
        self.line(format!("{bad} = xor i1 {ok}, true"));
        self.cond("Vf", &bad, |g| {
            g.line(format!("store i32 0, ptr addrspace(3) {f}"))
        });
        self.barrier();
        let v = self.tmp();
        self.line(format!("{v} = load i32, ptr addrspace(3) {f}"));
        let pass = self.tmp();
        self.line(format!("{pass} = icmp ne i32 {v}, 0"));
        let _ = x;
        pass
    }

    fn and(&mut self, a: &str, b: &str) -> String {
        let r = self.tmp();
        self.line(format!("{r} = and i1 {a}, {b}"));
        r
    }

    /// Fold or consumer `k`'s laws after the last window on row `y`'s total:
    /// a rescale from the running shift to the value's final
    /// (`exp(a(m_run, v))`, NaN where both are the identity — the sequential
    /// fold's), then each factor or shift with the value.
    fn laws_after(
        &mut self,
        laws: &[mapal_ir::StreamLaw],
        fin: &[Vec<String>],
        y: usize,
        tot: &str,
        _k: usize,
    ) -> String {
        let mut tot = tot.to_string();
        for l in laws {
            if let (DepLaw::Rescale, Some(arg)) = (l.law, &l.rescale) {
                let m = self.tmp();
                self.line(format!("{m} = load float, ptr %F{}_{y}", l.value));
                let e = self.exp_arg(arg, &m, &fin[l.value][y]);
                let z = self.tmp();
                self.line(format!("{z} = fmul float {tot}, {e}"));
                tot = z;
            }
        }
        for l in laws {
            let code = match l.law {
                DepLaw::Rescale => continue,
                DepLaw::Factor { div: false } => "fmul",
                DepLaw::Factor { div: true } => "fdiv",
                DepLaw::Shift { sub: false } => "fadd",
                DepLaw::Shift { sub: true } | DepLaw::Phase => "fsub",
            };
            let z = self.tmp();
            self.line(format!("{z} = {code} float {tot}, {}", fin[l.value][y]));
            tot = z;
        }
        tot
    }

    /// `exp(a(x := from, v := to))`: the exponent's difference, its steps,
    /// the default face's `exp`.
    fn exp_arg(&mut self, arg: &mapal_ir::RescaleArg, from: &str, to: &str) -> String {
        let (a, b) = if arg.x_first { (from, to) } else { (to, from) };
        let mut d = self.tmp();
        self.line(format!("{d} = fsub float {a}, {b}"));
        for (sop, cst) in &arg.steps {
            let code = match *sop {
                Operation::Div => "fdiv",
                _ => "fmul",
            };
            let z = self.tmp();
            self.line(format!("{z} = {code} float {d}, {}", const_literal(cst)));
            d = z;
        }
        let f = self.tmp();
        self.line(format!(
            "{f} = call float {}(float {d})",
            crate::exp::symbol(32)
        ));
        f
    }

    /// A window's rescale of a state taken against shift `old` to shift
    /// `new`: `1` where the shift stayed (an identity shift's zero state
    /// stays zero), else `exp(a(old, new))`.
    fn alpha(&mut self, arg: &mapal_ir::RescaleArg, old: &str, new: &str) -> String {
        let key = format!("{old}|{new}|{}|{:?}", arg.x_first, arg.steps);
        if let Some((_, v)) = self.alphas.iter().find(|x| x.0 == key) {
            return v.clone();
        }
        let e = self.exp_arg(arg, old, new);
        let c = self.tmp();
        self.line(format!("{c} = fcmp oeq float {old}, {new}"));
        let a = self.tmp();
        self.line(format!("{a} = select i1 {c}, float 1.0, float {e}"));
        self.alphas.push((key, a.clone()));
        a
    }

    /// `v` combined over the `g` lanes of the row group by `op` (a
    /// butterfly: every lane holds the total).
    fn butterfly(&mut self, v: &str, g: u64, op: mapal_ir::RowFoldOp) -> String {
        let mut cur = v.to_string();
        let mut off = 1;
        while off < g {
            let o = self.tmp();
            self.line(format!(
                "{o} = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {cur}, i32 {off}, i32 31)"
            ));
            let r = self.tmp();
            self.step(&op, "float", &cur, &o, &r);
            cur = r;
            off *= 2;
        }
        cur
    }

    /// `body` where `c` holds.
    fn cond(&mut self, tag: &str, c: &str, body: impl FnOnce(&mut Self)) {
        let (yes, end) = (format!("{tag}y"), format!("{tag}n"));
        self.line(format!("br i1 {c}, label %{yes}, label %{end}"));
        self.label(&yes);
        body(self);
        self.line(format!("br label %{end}"));
        self.label(&end);
    }

    fn commit(&mut self, cp: bool) {
        if cp {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
    }

    fn wait(&mut self, cp: bool, pending: u32) {
        if cp {
            self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {pending})"
            ));
        }
    }

    /// Producer `k`'s row panel: its `a` operand's `b` rows of `d`.
    fn stage_panel(&mut self, st: &Stream, lay: &Layout, k: usize) {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a producer is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[0] else {
            unreachable!("a producer's row operand is memory")
        };
        let (off, dp) = lay.q[k].expect("a producer's panel");
        let a = Addr::of(site, Operand::A);
        let d = site.k;
        // Row `r` of the block, element `kk`: `a(i + r, ·, kk)`.
        let vec = a.vec(Ax::K, d);
        self.stage(
            &format!("Q{k}"),
            st,
            (off, dp),
            (st.b, d),
            (gi, vec),
            None,
            &mut |g, r, c| {
                let i = g.op("add", "%i", r);
                g.addr_at(&a, &i, "0", c)
            },
        );
    }

    /// Member `k`'s window from key `j0`: a producer's `b` operand rows
    /// (`kb` of `d`), or a consumer's (`kb` rows of `c`) — at the block's
    /// rows (one panel batch: the operand's row digits do not move).
    fn stage_window(&mut self, st: &Stream, lay: &Layout, k: usize, j0: &str, produce: bool) {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a windowed member is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[1] else {
            unreachable!("a tile's column operand is memory")
        };
        let bb = Addr::of(site, Operand::B);
        let w = st.group.axis;
        let j0 = j0.to_string();
        let tag = format!("{}{k}_{}", if produce { "K" } else { "V" }, self.n);
        // The block's first row as this member numbers it.
        let i0 = self.member_row(st, k, "%i");
        if produce {
            // Key `j`'s row: `b(i, j, kk)`, `kk < d`.
            let (off, dp) = lay.k[k].expect("a producer's window");
            let vec = bb.vec(Ax::K, site.k);
            self.stage(
                &tag,
                st,
                (off, dp),
                (st.kb, site.k),
                (gi, vec),
                Some((j0.clone(), w)),
                &mut |g, r, c| {
                    let j = g.op("add", &j0, r);
                    g.addr_at(&bb, &i0, &j, c)
                },
            );
        } else {
            // Key `j`'s row of the consumer: `b(i, c, j)`.
            let (off, cpd) = lay.v[k].expect("a consumer's window");
            let vec = bb.vec(Ax::Lane, site.c);
            self.stage(
                &tag,
                st,
                (off, cpd),
                (st.kb, site.c),
                (gi, vec),
                Some((j0.clone(), w)),
                &mut |g, r, c| {
                    let j = g.op("add", &j0, r);
                    g.addr_at(&bb, &i0, c, &j)
                },
            );
        }
    }

    /// Group row `row` as member `k` numbers its rows (a digit permutation,
    /// [`mapal_ir::RowMap`]; else the row itself).
    fn member_row(&mut self, st: &Stream, k: usize, row: &str) -> String {
        match st.rows_of.get(k).cloned().flatten() {
            Some(m) => self.digits(&m.from_group, row),
            None => row.to_string(),
        }
    }

    /// `Σ coef·((x ÷ stride) % radix)`.
    fn digits(&mut self, ds: &[mapal_ir::AxisDigit], x: &str) -> String {
        let mut acc = "0".to_string();
        for d in ds {
            let q = if d.stride == 1 {
                x.to_string()
            } else {
                self.op("udiv", x, d.stride)
            };
            let r = self.op("urem", &q, d.radix);
            let t = self.op("mul", &r, d.coef);
            acc = self.op("add", &acc, &t);
        }
        acc
    }

    /// Operand address `a` at `(i, lane, k)`.
    fn addr_at(&mut self, a: &Addr, i: &str, lane: &str, k: &str) -> String {
        match a {
            Addr::Plain(r) => {
                let x = self.op("mul", i, r.ci);
                let y = self.op("mul", lane, r.clane);
                let z = self.op("mul", k, r.ck);
                let s = self.op("add", &x, &y);
                let s = self.op("add", &s, &z);
                self.op("add", &s, r.base)
            }
            Addr::Out(c) => {
                let x = self.op("mul", i, *c);
                self.op("add", &x, lane)
            }
            Addr::Radix(r) => {
                let x = self.digits(&r.i, i);
                let y = self.digits(&r.lane, lane);
                let z = self.digits(&r.k, k);
                let s = self.op("add", &x, &y);
                let s = self.op("add", &s, &z);
                self.op("add", &s, r.base)
            }
        }
    }

    /// Copy `rows × cols` elements into shared memory at `off` (row stride
    /// `stride` elements): element `(r, c)` from `at(r, c)` of operand `gi`;
    /// 16-byte chunks where `vec` (each chunk's four consecutive). Rows past
    /// `limit` (counted from `base`) are zero: their copies land in the dump
    /// slot from a row inside, and a zero is stored in their place.
    #[allow(clippy::too_many_arguments)]
    fn stage(
        &mut self,
        tag: &str,
        st: &Stream,
        (off, stride): (u64, u64),
        (rows, cols): (u64, u64),
        (gi, vec): (usize, bool),
        limit: Option<(String, u64)>,
        at: &mut dyn FnMut(&mut Self, &str, &str) -> String,
    ) {
        let cp = st.cp;
        let v = if vec { 4 } else { 1 };
        let per = cols / v;
        let total = rows * per;
        let threads = st.threads;
        let dump = st.layout(&self.u.members).dump;
        self.counted(tag, ("%l", threads, total), None, &mut |g, e, _| {
            let r = g.op("udiv", e, per);
            let c = g.op("urem", e, per);
            let cc = g.op("mul", &c, v);
            // A row past the axis reads the axis's last row instead.
            let (rr, ok) = match &limit {
                Some((base, w)) => {
                    let j = g.op("add", base, &r);
                    let ok = g.tmp();
                    g.line(format!("{ok} = icmp ult i64 {j}, {w}"));
                    let lim = g.tmp();
                    g.line(format!("{lim} = sub i64 {}, {base}", w - 1));
                    let rr = g.tmp();
                    g.line(format!("{rr} = select i1 {ok}, i64 {r}, i64 {lim}"));
                    (rr, Some(ok))
                }
                None => (r.clone(), None),
            };
            let src_i = at(g, &rr, &cc);
            let src = g.tmp();
            g.line(format!(
                "{src} = getelementptr inbounds float, ptr addrspace(1) %x{gi}, i64 {src_i}"
            ));
            let rs = g.op("mul", &r, stride);
            let di = g.op("add", &rs, &cc);
            let dst = g.smem(off, "float", &di);
            let dumpp = g.smem(dump, "float", "0");
            let (into, zero_at) = match &ok {
                Some(ok) => {
                    let a = g.tmp();
                    g.line(format!(
                        "{a} = select i1 {ok}, ptr addrspace(3) {dst}, ptr addrspace(3) {dumpp}"
                    ));
                    let z = g.tmp();
                    g.line(format!(
                        "{z} = select i1 {ok}, ptr addrspace(3) {dumpp}, ptr addrspace(3) {dst}"
                    ));
                    (a, Some(z))
                }
                None => (dst.clone(), None),
            };
            let vt = if vec { "<4 x float>" } else { "float" };
            let zero = if vec { "zeroinitializer" } else { "0.0" };
            match (cp, vec) {
                (true, true) => g.line(format!(
                    "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {into}, ptr addrspace(1) {src})"
                )),
                (true, false) => g.line(format!(
                    "call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) {into}, ptr addrspace(1) {src})"
                )),
                (false, _) => {
                    let x = g.tmp();
                    g.line(format!("{x} = load {vt}, ptr addrspace(1) {src}"));
                    g.line(format!("store {vt} {x}, ptr addrspace(3) {into}"));
                }
            }
            if let Some(z) = zero_at {
                g.line(format!("store {vt} {zero}, ptr addrspace(3) {z}"));
            }
        });
    }

    /// Tile member `k`'s epilogue body at its output cell `t` with its fold's
    /// value `acc` (`tile-post`): memory operands by pointer, a per-row
    /// member's final value through its one-cell buffer, an array only the
    /// fold reads as `null`, the cell index.
    fn post_call(&mut self, k: usize, t: &str, acc: &str, per_row: &[(u32, String)]) -> String {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("an epilogue is a tile's")
        };
        let post = site.post.clone().expect("an epilogue");
        let c = self.calls[k].as_ref().expect("an epilogue call");
        let (callee, arg, ret, captures) =
            (c.callee.clone(), c.arg.clone(), c.ret.clone(), c.captures);
        let field_ty = c.field_ty.clone();
        let srcs = mb.srcs.clone();
        let mut agg = "poison".to_string();
        for s in 0..=captures {
            let (ty, v) = if s == captures {
                let it = field_ty[s as usize].clone();
                match it.as_str() {
                    "i64" => ("i64".to_string(), t.to_string()),
                    _ => ("i32".to_string(), self.i32_of(t)),
                }
            } else {
                match post.reads.iter().position(|&x| x == s) {
                    Some(j) => match &srcs[2 + j] {
                        USrc::Gmem(g) => ("ptr".to_string(), format!("%xg{g}")),
                        USrc::Member { .. } => {
                            let v = per_row
                                .iter()
                                .find(|x| x.0 == s)
                                .map(|x| x.1.clone())
                                .expect("a per-row read's value");
                            let b = format!("%b{k}_{s}");
                            self.line(format!("store {} {v}, ptr {b}", field_ty[s as usize]));
                            ("ptr".to_string(), b)
                        }
                        _ => unreachable!("an epilogue reads memory or a per-row member"),
                    },
                    // An array only the fold reads; a number only the
                    // fold's operands take (`unit-uniform`): unread.
                    None if post.uniform.contains(&s) => {
                        let ty = field_ty[s as usize].clone();
                        let v = match ty.as_str() {
                            "float" | "double" | "half" | "bfloat" => "0.0",
                            _ => "0",
                        };
                        (ty, v.to_string())
                    }
                    None => ("ptr".to_string(), "null".to_string()),
                }
            };
            let n = self.tmp();
            self.line(format!("{n} = insertvalue {arg} {agg}, {ty} {v}, {s}"));
            agg = n;
        }
        let r = self.tmp();
        self.line(format!(
            "{r} = call {ret} @{callee}({arg} {agg}, float {acc})"
        ));
        r
    }

    /// `stream-mma`: two f32 values rounded once into one register of the
    /// format, the lower `k` in the low half (the tile kernel's `pack`).
    fn mma_pack(&mut self, fmt: crate::profile::MmaFmt, lo: &str, hi: &str) -> String {
        let r = self.tmp();
        self.line(format!(
            "{r} = call i32 asm \"cvt.rn.{f}x2.f32 $0, $1, $2;\", \"=r,f,f\"(float {hi}, float {lo})",
            f = fmt.ptx()
        ));
        r
    }

    /// `stream-mma`: one `m16n8k16` product into the accumulators `c` (the
    /// PTX ISA's fragment layouts: `a` = (g, 2t), (g+8, 2t), (g, 2t+8),
    /// (g+8, 2t+8); `b` = (k 2t, n g), (k 2t+8, n g); `c` = (g, 2t..2t+1),
    /// (g+8, 2t..2t+1)).
    fn mma_step(
        &mut self,
        fmt: crate::profile::MmaFmt,
        a: &[String; 4],
        b: &[String; 2],
        c: &[String; 4],
    ) -> [String; 4] {
        let r = self.tmp();
        let f = fmt.ptx();
        self.line(format!(
            "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k16.row.col.f32.{f}.{f}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, i32 {}, float {}, float {}, float {}, float {})",
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        ));
        std::array::from_fn(|q| {
            let v = self.tmp();
            self.line(format!(
                "{v} = extractvalue {{ float, float, float, float }} {r}, {q}"
            ));
            v
        })
    }

    /// `stream-mma`: producer `k`'s row operand as `a` fragments, each
    /// `k`-step's four registers — the thread's rows `grow` (`g`, `g+8`),
    /// elements `16s + 2x (+8)` and the next, from memory once a block.
    fn mma_rows(&mut self, st: &Stream, k: usize, grow: &[String], x: &str) -> Vec<[String; 4]> {
        let mb = &self.u.members[k];
        let UKind::Tile { site } = &mb.kind else {
            unreachable!("a producer is a tile")
        };
        let USrc::Gmem(gi) = mb.srcs[0] else {
            unreachable!("a producer's row operand is memory")
        };
        let site = site.clone();
        let fmt = st.mma.expect("stream-mma");
        let a = Addr::of(&site, Operand::A);
        let x2 = self.op("mul", x, 2);
        let mut out = Vec::new();
        for s in 0..site.k / MMA_K {
            let mut q: Vec<String> = Vec::new();
            for (yy, kc) in [(0usize, 0u64), (1, 0), (0, 8), (1, 8)] {
                let mut v: Vec<String> = Vec::new();
                for e in 0..2 {
                    let kk = self.op("add", &x2, s * MMA_K + kc + e);
                    let i = self.addr_at(&a, &grow[yy], "0", &kk);
                    let ptr = self.tmp();
                    self.line(format!(
                        "{ptr} = getelementptr inbounds float, ptr addrspace(1) %x{gi}, i64 {i}"
                    ));
                    let l = self.tmp();
                    self.line(format!("{l} = load float, ptr addrspace(1) {ptr}, align 4"));
                    v.push(l);
                }
                q.push(self.mma_pack(fmt, &v[0], &v[1]));
            }
            out.push([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
        }
        out
    }

    /// `stream-mma`: producer `k`'s window — its `kb ÷ 8` accumulator tiles
    /// from the site's seed, folded over `d` by `k`-steps against the
    /// window's `b` fragments (key `8t + gq`'s elements `16s + 2x (+8)`, two
    /// values a load) — returned as the cells `(y, z)`: tile `t`'s `c0 c1`
    /// row 0 columns `2t`, `2t+1`, `c2 c3` row 1.
    fn produce_mma(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        af: &[[String; 4]],
        x: &str,
        gq: &str,
    ) -> Vals {
        let UKind::Tile { site } = &self.u.members[k].kind else {
            unreachable!("a producer is a tile")
        };
        let site = site.clone();
        let fmt = st.mma.expect("stream-mma");
        let (koff, dp) = lay.k[k].expect("window");
        let seed = const_literal(&site.seed);
        let nt = (st.kb / MMA_N) as usize;
        let x2 = self.op("mul", x, 2);
        let gr = self.op("mul", gq, dp);
        let base = self.op("add", &gr, &x2);
        let mut c: Vec<[String; 4]> = vec![std::array::from_fn(|_| seed.clone()); nt];
        for (s, a) in af.iter().enumerate() {
            for (t, ct) in c.iter_mut().enumerate() {
                let mut b: Vec<String> = Vec::new();
                for kc in [0u64, 8] {
                    let i = self.op("add", &base, t as u64 * MMA_N * dp + s as u64 * MMA_K + kc);
                    let p = self.smem(koff, "float", &i);
                    let v = self.tmp();
                    self.line(format!(
                        "{v} = load <2 x float>, ptr addrspace(3) {p}, align 8"
                    ));
                    let (e0, e1) = (self.tmp(), self.tmp());
                    self.line(format!("{e0} = extractelement <2 x float> {v}, i64 0"));
                    self.line(format!("{e1} = extractelement <2 x float> {v}, i64 1"));
                    b.push(self.mma_pack(fmt, &e0, &e1));
                }
                *ct = self.mma_step(fmt, a, &[b[0].clone(), b[1].clone()], ct);
            }
        }
        let ctn = st.ct() as usize;
        let mut out = vec![String::new(); 2 * ctn];
        for (t, ct) in c.iter().enumerate() {
            out[2 * t] = ct[0].clone();
            out[2 * t + 1] = ct[1].clone();
            out[ctn + 2 * t] = ct[2].clone();
            out[ctn + 2 * t + 1] = ct[3].clone();
        }
        out
    }

    /// `stream-mma`: consumer `k`'s window — the producer's cells `pv` (the
    /// accumulator layout is the `a` fragment's: `k`-step `s` packs columns
    /// `4s`, `4s+1` / `4s+2`, `4s+3` of rows 0 and 1) against the window's
    /// `b` fragments (keys `16s + 2x (+1, +8, +9)` of column `8t + gq`),
    /// into its accumulators `O(y, 2t + e)`.
    #[allow(clippy::too_many_arguments)]
    fn consume_mma(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        site: &mapal_ir::TileSite,
        pv: &[String],
        x: &str,
        gq: &str,
    ) {
        let fmt = st.mma.expect("stream-mma");
        let (voff, cpd) = lay.v[k].expect("window");
        let (oc, _) = st.oc(site.c);
        let ct = st.ct() as usize;
        let x2 = self.op("mul", x, 2);
        let xr = self.op("mul", &x2, cpd);
        let base = self.op("add", &xr, gq);
        for s in 0..(st.kb / MMA_K) as usize {
            let z = 4 * s;
            let a = [
                self.mma_pack(fmt, &pv[z], &pv[z + 1]),
                self.mma_pack(fmt, &pv[ct + z], &pv[ct + z + 1]),
                self.mma_pack(fmt, &pv[z + 2], &pv[z + 3]),
                self.mma_pack(fmt, &pv[ct + z + 2], &pv[ct + z + 3]),
            ];
            for t in 0..site.c / MMA_N {
                let mut b: Vec<String> = Vec::new();
                for kc in [0u64, 8] {
                    let mut v: Vec<String> = Vec::new();
                    for e in 0..2 {
                        let off = (s as u64 * MMA_K + kc + e) * cpd + t * MMA_N;
                        let i = self.op("add", &base, off);
                        let p = self.smem(voff, "float", &i);
                        let l = self.tmp();
                        self.line(format!("{l} = load float, ptr addrspace(3) {p}, align 4"));
                        v.push(l);
                    }
                    b.push(self.mma_pack(fmt, &v[0], &v[1]));
                }
                let slots = [2 * t, 2 * t + 1, oc + 2 * t, oc + 2 * t + 1];
                let cur: [String; 4] = std::array::from_fn(|q| {
                    let v = self.tmp();
                    self.line(format!("{v} = load float, ptr %O{k}_{}", slots[q]));
                    v
                });
                let r = self.mma_step(fmt, &a, &[b[0].clone(), b[1].clone()], &cur);
                for (q, v) in r.iter().enumerate() {
                    self.line(format!("store float {v}, ptr %O{k}_{}", slots[q]));
                }
            }
        }
    }

    /// Producer `k`'s window: each of the thread's `rt × ct` cells — rows
    /// `rloc`, window columns `jl` — folded over `d` from the panels, from the
    /// site's seed in its own order.
    fn produce(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        rloc: &[String],
        jl: &[String],
    ) -> Vals {
        let UKind::Tile { site } = &self.u.members[k].kind else {
            unreachable!("a producer is a tile")
        };
        let site = site.clone();
        let (rt, ct) = (st.rt as usize, st.ct() as usize);
        let (qoff, dp) = lay.q[k].expect("panel");
        let (koff, _) = lay.k[k].expect("window");
        let seed = const_literal(&site.seed);
        for c in 0..rt * ct {
            self.line(format!("store float {seed}, ptr %S{k}_{c}"));
        }
        let d = site.k;
        let v = if d.is_multiple_of(4) { 4 } else { 1 };
        let qrow: Vec<String> = rloc.iter().map(|r| self.op("mul", r, dp)).collect();
        let krow: Vec<String> = jl.iter().map(|j| self.op("mul", j, dp)).collect();
        self.counted(&format!("P{k}"), ("0", v, d), None, &mut |g, kk, _| {
            let load = |g: &mut Self, off: u64, base: &str| -> Vec<String> {
                let i = g.op("add", base, kk);
                let p = g.smem(off, "float", &i);
                if v == 4 {
                    let x = g.tmp();
                    g.line(format!(
                        "{x} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                    ));
                    (0..4)
                        .map(|e| {
                            let s = g.tmp();
                            g.line(format!("{s} = extractelement <4 x float> {x}, i32 {e}"));
                            s
                        })
                        .collect()
                } else {
                    let x = g.tmp();
                    g.line(format!("{x} = load float, ptr addrspace(3) {p}"));
                    vec![x]
                }
            };
            let qs: Vec<Vec<String>> = qrow.iter().map(|b| load(g, qoff, b)).collect();
            let ks: Vec<Vec<String>> = krow.iter().map(|b| load(g, koff, b)).collect();
            for y in 0..rt {
                for z in 0..ct {
                    let a = format!("%S{k}_{}", y * ct + z);
                    let mut acc = g.tmp();
                    g.line(format!("{acc} = load float, ptr {a}"));
                    for e in 0..v as usize {
                        let o = g.tmp();
                        g.tile_step(&site, "float", &acc, &qs[y][e], &ks[z][e], &o);
                        acc = o;
                    }
                    g.line(format!("store float {acc}, ptr {a}"));
                }
            }
        });
        let mut out = Vec::with_capacity(rt * ct);
        for c in 0..rt * ct {
            let s = self.tmp();
            self.line(format!("{s} = load float, ptr %S{k}_{c}"));
            out.push(s);
        }
        out
    }

    /// Consumer `k`'s window: its accumulators `rt × oc` fold the window's
    /// row operand (shared memory, 16-byte rows) against its window.
    fn consume(
        &mut self,
        st: &Stream,
        lay: &Layout,
        k: usize,
        site: &mapal_ir::TileSite,
        rloc: &[String],
        x: &str,
    ) {
        let (rt, kb, g) = (st.rt as usize, st.kb, st.g);
        let (aoff, ap) = lay.a[k].expect("row window");
        let (voff, cpd) = lay.v[k].expect("window");
        let (oc, contig) = st.oc(site.c);
        let c = site.c;
        let arow: Vec<String> = rloc.iter().map(|r| self.op("mul", r, ap)).collect();
        // The thread's columns (clamped inside the row; a column past it is
        // never stored).
        let cols: Vec<String> = (0..oc)
            .map(|q| match contig {
                true => {
                    let c0 = self.op("mul", x, oc);
                    self.op("add", &c0, q)
                }
                false => {
                    let col = self.op("add", x, g * q);
                    let ok = self.tmp();
                    self.line(format!("{ok} = icmp ult i64 {col}, {c}"));
                    let s = self.tmp();
                    self.line(format!("{s} = select i1 {ok}, i64 {col}, i64 {}", c - 1));
                    s
                }
            })
            .collect();
        let site = site.clone();
        let x = x.to_string();
        self.counted(&format!("C{k}"), ("0", 4, kb), None, &mut |gg, jj, _| {
            let mut av: Vec<Vec<String>> = Vec::new();
            for r in &arow {
                let i = gg.op("add", r, jj);
                let p = gg.smem(aoff, "float", &i);
                let v = gg.tmp();
                gg.line(format!(
                    "{v} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                ));
                av.push(
                    (0..4)
                        .map(|e| {
                            let s = gg.tmp();
                            gg.line(format!("{s} = extractelement <4 x float> {v}, i32 {e}"));
                            s
                        })
                        .collect(),
                );
            }
            let mut accs: Vec<String> = Vec::with_capacity(rt * oc as usize);
            for idx in 0..rt as u64 * oc {
                let a = gg.tmp();
                gg.line(format!("{a} = load float, ptr %O{k}_{idx}"));
                accs.push(a);
            }
            for e in 0..4u64 {
                let jr = gg.op("add", jj, e);
                let vrow = gg.op("mul", &jr, cpd);
                let vs: Vec<String> = match contig {
                    true => {
                        let c0 = gg.op("mul", &x, oc);
                        let base = gg.op("add", &vrow, &c0);
                        let mut out = Vec::new();
                        for q4 in 0..oc / 4 {
                            let i = gg.op("add", &base, q4 * 4);
                            let p = gg.smem(voff, "float", &i);
                            let v = gg.tmp();
                            gg.line(format!(
                                "{v} = load <4 x float>, ptr addrspace(3) {p}, align 16"
                            ));
                            for l in 0..4 {
                                let s = gg.tmp();
                                gg.line(format!("{s} = extractelement <4 x float> {v}, i32 {l}"));
                                out.push(s);
                            }
                        }
                        out
                    }
                    false => cols
                        .iter()
                        .map(|col| {
                            let i = gg.op("add", &vrow, col);
                            let p = gg.smem(voff, "float", &i);
                            let v = gg.tmp();
                            gg.line(format!("{v} = load float, ptr addrspace(3) {p}"));
                            v
                        })
                        .collect(),
                };
                for y in 0..rt {
                    for q in 0..oc as usize {
                        let o = gg.tmp();
                        let idx = y * oc as usize + q;
                        gg.tile_step(&site, "float", &accs[idx], &av[y][e as usize], &vs[q], &o);
                        accs[idx] = o;
                    }
                }
            }
            for (idx, a) in accs.iter().enumerate() {
                gg.line(format!("store float {a}, ptr %O{k}_{idx}"));
            }
        });
    }
}
