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
//! `4x + 4g·⌊q÷4⌋ + q%4` (16-byte chunks interleaved across the group: one
//! load's `g` lanes read consecutive bytes) or `x + g·q`. A window's row
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

#[path = "stream/body.rs"]
mod body;
#[path = "stream/choice.rs"]
mod choice;
#[path = "stream/guard.rs"]
mod guard;
#[path = "stream/mma.rs"]
mod mma;
#[path = "stream/stage.rs"]
mod stage;

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
    /// mask-frontend: the walk order of the group's rows
    /// ([`mapal_ir::UnitGroup::order`]): a block's row `r` is group row
    /// `order(i + r)`. `None`: `i + r`.
    pub order: Option<mapal_ir::RowMap>,
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
            || site.a.narrow.is_some_and(|n| MmaFmt::of_narrow(n) != f)
            || site.b.narrow.is_some_and(|n| MmaFmt::of_narrow(n) != f)
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

/// bf16-class `narrow-window`: a tile member's column operand bytes an
/// element — its storage format's (a bf16 KV cache: 2), else f32's.
pub(crate) fn win_bytes(site: &mapal_ir::TileSite) -> u64 {
    match site.b.narrow {
        Some(_) => 2,
        None => 4,
    }
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
            // mask-frontend: a block walked in the group's order — its rows'
            // sides as a set (an open side: every window).
            let set = self.order.as_ref().map(|o| {
                let rows: Vec<u64> = (r.0..=r.1).map(|p| o.group_row(p)).collect();
                (
                    gd.form.lhs.rows_range(&rows, &mapal_ir::NoValues),
                    gd.form.rhs.rows_range(&rows, &mapal_ir::NoValues),
                )
            });
            let vis: Vec<u64> = (0..nb)
                .filter(|&t| {
                    let l = (t * kb, (t * kb + kb).min(w) - 1);
                    let class = match &set {
                        None => gd.form.classify(r, l),
                        Some((Some(a), Some(b))) => {
                            let (la, lb) = (gd.form.lhs.lanes_range(l), gd.form.rhs.lanes_range(l));
                            gd.form
                                .class_of((a.0 + la.0, a.1 + la.1), (b.0 + lb.0, b.1 + lb.1))
                        }
                        Some(_) => mapal_ir::BoxClass::Mixed,
                    };
                    class != mapal_ir::BoxClass::Masked
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
        let mut take = |bytes: u64| {
            let o = off.next_multiple_of(16);
            off = o + bytes;
            o
        };
        for (k, mb) in members.iter().enumerate() {
            let UKind::Tile { site } = &mb.kind else {
                continue;
            };
            match (&self.group.roles[k], self.mma.is_some()) {
                (StreamRole::Produce, false) => {
                    let dp = site.k.next_multiple_of(4) + 4;
                    lay.q[k] = Some((take(self.b * dp * 4), dp));
                    lay.k[k] = Some((take(self.kb * dp * 4), dp));
                }
                (StreamRole::Consume { .. }, false) => {
                    let cp = site.c.next_multiple_of(4);
                    lay.v[k] = Some((take(self.kb * cp * 4), cp));
                    let ap = self.kb + 4;
                    lay.a[k] = Some((take(self.b * ap * 4), ap));
                }
                // `stream-mma`: the row operand's fragments live in registers
                // (no panel, no row window). A window row `≡ 8 (mod 16)`
                // words apart: a half-warp's 8-byte `b` fragment loads (rows
                // `l%32÷4`, words `2·(l%4)`) on distinct banks; the
                // consumer's rows `≡ 4 (mod 8)` apart: its 4-byte loads (rows
                // `2·(l%4)`, column `l%32÷4`) likewise.
                //
                // bf16-class `narrow-window`: a bf16 window's rows `≡ 4 (mod
                // 8)` words apart either way (`k + 8` / `c + 8` elements):
                // the producer's 4-byte pair loads (rows `l%32÷4`, words
                // `l%4`) and the consumer's 2-byte loads (rows `2·(l%4)`, a
                // column pair a word) on distinct banks.
                (StreamRole::Produce, true) => {
                    let eb = win_bytes(site);
                    let dp = site.k + 8;
                    lay.k[k] = Some((take(self.kb * dp * eb), dp));
                }
                (StreamRole::Consume { .. }, true) => {
                    let eb = win_bytes(site);
                    let cp = site.c + if eb == 2 { 8 } else { 4 };
                    lay.v[k] = Some((take(self.kb * cp * eb), cp));
                }
                _ => {}
            }
        }
        lay.dump = take(16);
        lay.flag = take(16);
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
        self.vec_by(ax, n, 4)
    }

    /// [`Self::vec`] for elements `per` a 16-byte chunk (8 of a 2-byte
    /// format): every term a multiple of `per`.
    pub fn vec_by(&self, ax: Ax, n: u64, per: u64) -> bool {
        if !n.is_multiple_of(per) {
            return false;
        }
        match self {
            Addr::Plain(r) => {
                let (unit, other) = match ax {
                    Ax::Lane => (r.clane, r.ck),
                    Ax::K => (r.ck, r.clane),
                };
                unit == 1
                    && other.is_multiple_of(per)
                    && r.ci.is_multiple_of(per)
                    && r.base.is_multiple_of(per)
                    && r.ksplit.is_none()
            }
            Addr::Radix(r) => {
                let (along, other) = match ax {
                    Ax::Lane => (&r.lane, &r.k),
                    Ax::K => (&r.k, &r.lane),
                };
                let unit = matches!(along[..], [d] if d.stride == 1 && d.coef == 1 && d.radix >= n);
                unit && r.base.is_multiple_of(per)
                    && other.iter().chain(&r.i).all(|d| d.coef.is_multiple_of(per))
            }
            Addr::Out(c) => ax == Ax::Lane && c.is_multiple_of(per),
        }
    }
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

/// A member's window values, per thread cell `y·ct + z`.
type Vals = Vec<String>;
