//! Plan types of the facts catalog: row folds, trip splits, emission classes and the element-law plan.

use super::*;

/// **A row-local in-order fold (S54 `rowfold`):** a `map` over `rows` whose
/// body is one `fold` over `k` reading one array at `base + ci·i + ck·k` and
/// combining it into the accumulator by one step — `out[i] = fold_k (seed,
/// ⊕) a[base + ci·i + ck·k]`. The arm [`CategoryIr::tile_site`] refuses (it
/// needs `a·b`): a row sum, a row max, a norm's denominator.
///
/// Each row is one chain in increasing `k` (ADR-0028 D4), so rows are
/// independent and any realization that keeps one row on one in-order chain
/// computes the oracle's bits. Geometry only, like [`TileSite`]: where it runs
/// is the backend's question.
///
/// generic-fusion `window`: the map index may be split `(t÷w, t%w)` and the
/// fold's axis `(k÷div, k%div)` — a weightless 2-D window (a box filter):
/// cell `t` folds `a[base + ci·(t÷w) + clane·(t%w) + taps]`, the taps `ck·k`
/// or [`TileKSplit`]'s `cq·(k÷div) + cr·(k%div)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RowFoldSite {
    pub rows: u64,
    pub k: u64,
    /// The one read; `clane == 0` unless the map index is split (`w > 1`).
    pub a: TileRead,
    /// The map index's split width (1: no split, `t÷1 = t`).
    pub w: u64,
    pub op: RowFoldOp,
    pub seed: Value,
    pub elem: Ty,
}

/// The step of a [`RowFoldSite`], in the source's operand order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowFoldOp {
    /// `acc ⊕ e` (`acc_first`) or `e ⊕ acc`, `⊕` ∈ {`Add`, `Sub`, `Mul`}.
    Arith { op: Operation, acc_first: bool },
    /// `(x cmp y) ? p : q` with `{x, y} = {p, q} = {acc, e}` — a running max
    /// or min. `acc_lhs`: `x` is `acc`; `acc_if_true`: `p` is `acc`.
    Select {
        cmp: Operation,
        acc_lhs: bool,
        acc_if_true: bool,
    },
}

/// **The trip split of a counted loop** (recurrence-precision item 1,
/// [`CategoryIr::trip_split`]): the counter `t` (a state slot, `t + 1 → t`,
/// a constant first trip, continued while `t < hi`), the trip intervals the
/// guards on `t` cut the range into, and per interval which state is a
/// recurrence there and which body work is outside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripSplit {
    pub merge: ObjectId,
    /// The counter's state slot.
    pub counter: u32,
    /// Trips `t ∈ [lo, hi)`.
    pub lo: i64,
    pub hi: i64,
    /// Consecutive, covering `[lo, hi)`; one per run of trips on which every
    /// guard on `t` takes the same arm.
    pub phases: Vec<TripPhase>,
}

/// One interval of a [`TripSplit`]: the body as its trips run it (every
/// guard on the counter decided, the arm not taken dropped).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TripPhase {
    pub lo: i64,
    pub hi: i64,
    /// State slots on a cycle of the slot dependence graph (slot `k` → `j`
    /// when `j`'s next value reads `k`'s view): the recurrence. A piecewise
    /// write of a slot over itself (`batched`) is not a self-edge: its cells
    /// read nothing of the previous state.
    pub recurrent: Vec<u32>,
    /// Body morphisms no recurrent slot's view reaches: closed forms of the
    /// counter, loop invariants and non-recurrent state.
    pub outside: Vec<MorphismId>,
    /// `(slot, map)`: the slot's next value is a piecewise map over the slot
    /// itself ([`Outside`]: rows `d·(t + b)`, `len ≤ d` cells each) whose
    /// other live captures are the counter or loop invariants — the
    /// interval's writes are one batch of `(hi − lo)·len` cells, cell `c`
    /// from trip `c ÷ d − b`.
    pub batched: Vec<(u32, MorphismId)>,
    /// The first interval only: trips `[lo, hi − 1)` leave no value a later
    /// trip reads but the `batched` writes — every other state slot is
    /// unchanged or overwritten from batched state and the counter alone,
    /// the exit test reads only the counter, and the body has no effect. The
    /// loop may enter at `hi − 1` with the batch applied.
    pub skip: bool,
}

/// [`CategoryIr::fold_core`]: a fold's step and its one read.
pub(super) struct FoldCore {
    pub(super) fbody: FuncId,
    pub(super) fcap: u32,
    pub(super) k: u64,
    pub(super) seed: Value,
    pub(super) op: RowFoldOp,
    /// The read array as the fold body sees it, and its address.
    pub(super) array: ObjectId,
    pub(super) address: ObjectId,
}

impl PathPlan {
    /// Whether scheduling can be skipped entirely.
    pub fn is_single_path(&self) -> bool {
        self.tasks.len() <= 1
            && matches!(
                self.checkpoints.as_slice(),
                [Checkpoint { topo: u32::MAX, wait, .. }]
                    if wait.len() == self.tasks.len()
                        && wait.iter().enumerate().all(|(task, entry)|
                            entry.task == task && entry.threshold.is_none())
            )
    }
}

/// How an owned, non-constant, non-token object participates in minimal
/// emission. Derived only from the sealed graph by
/// [`CategoryIr::emission_plan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmissionClass {
    /// A structural product whose fields flow directly to primitive operands
    /// or projections; the product itself never materializes.
    Dissolved,
    /// A pure, guard-free value with exactly one effective consumer.
    Inline,
    /// A boundary, guarded producer, or value without exactly one consumer.
    Named,
}

impl EmissionClass {
    /// Whether this is [`Self::Dissolved`].
    pub fn is_dissolved(self) -> bool {
        self == Self::Dissolved
    }

    /// Whether this is [`Self::Inline`].
    pub fn is_inline(self) -> bool {
        self == Self::Inline
    }

    /// Whether this is [`Self::Named`].
    pub fn is_named(self) -> bool {
        self == Self::Named
    }
}

/// The minimal-emission classification for one function. Constants and
/// token-carrying objects are absent because neither materializes.
#[derive(Clone, Debug, PartialEq)]
pub struct EmissionPlan {
    pub(super) class: SecondaryMap<ObjectId, EmissionClass>,
}

impl EmissionPlan {
    /// The object's class, or `None` when it is not owned by the queried
    /// function, is a constant, or carries a token.
    pub fn class(&self, o: ObjectId) -> Option<EmissionClass> {
        self.class.get(o).copied()
    }
}

/// The cells `lo + step·k`, `k ∈ [0, len)`, `lo = d·(a·e + b)` over a map
/// body's captured scalar `e` (its capture slot) — [`Outside`] before `e` is
/// resolved.
#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub(super) d: u64,
    pub(super) a: i64,
    pub(super) e: Option<u32>,
    pub(super) b: i64,
    pub(super) step: u64,
    pub(super) len: u64,
    pub(super) w: u64,
}

/// One side of an index interval: `t ÷ d ≥ a·e + b` or `t ÷ d < a·e + b`
/// (`d = 1`: `t` itself), the last field `d`.
#[derive(Clone, Copy, Debug)]
pub(super) enum Bound {
    Lo(Option<u32>, i64, i64, u64),
    Hi(Option<u32>, i64, i64, u64),
}

/// The per-element law of an array object: **what `out[i]` is, as a closed
/// function of the loop index** (plan-s37-stage-structure §3).
///
/// An array whose law is known does not have to exist. `iota` writes a million
/// `i32`s so a consuming `map` can read them back, but `ix[i]` *is* `i` — the
/// array is a stored copy of a deduced morphism (FRAMEWORK §5), paid for with a
/// store pass, a load pass, and an opaque memory object LLVM refuses to
/// vectorize through (`cannot identify array bounds`).
///
/// This type says only what the element **is**. Whether to use that instead of
/// loading — recompute vs. keep in memory, and whether inlining blows a cache,
/// register or occupancy budget — is a per-target question the **backend** owns:
/// on a GPU bandwidth is scarce and registers are cheap, on a CPU it is a body's
/// cycle count against an L2 round trip, on an FPGA it is BRAM against silicon
/// area. Same graph, three answers, so there is deliberately no cost, op-count
/// or read-count field here (ADR-0032: mapal-ir never learns a machine fact).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElemSrc {
    /// `out[i] = i` — the loop index itself (`Iota`).
    Index,
    /// `out[i] = x` for a loop-invariant `x` (`Fill`): component `slot` of
    /// `source`, which the emitter already loads once above the loop head.
    Broadcast { source: ObjectId, slot: u32 },
    /// **The cut.** No closed form, so the array materializes and the element is
    /// loaded — exactly today's behavior, which is why cutting is always sound.
    /// `(source, slot)` is the pair `array_operand_ptr` / `load_component`
    /// already take, so consuming this needs no new pointer plumbing.
    Load { source: ObjectId, slot: Option<u32> },
    /// Structural pairing with no arithmetic of its own: `Zip` is
    /// `Pair(·, ·)` and `Enumerate` is `Pair(Index, ·)` — which is why
    /// `Enumerate` needs no constructor here (ADR-0018 already calls `Zip` the
    /// canonical iso `Aⁿ × Bⁿ ≅ (A×B)ⁿ`, and `enumerate a ≅ zip(iota n, a)`).
    Pair(Box<ElemSrc>, Box<ElemSrc>),
    /// `out[i] = body(inner(i))` — a `Map` whose body is **classifiable**.
    ///
    /// The other four producers are bodyless: their op tag is the entire
    /// formula, zero degrees of freedom, total and trap-free by construction.
    /// `Map` is not their peer — it carries an arbitrary `FuncId` that may
    /// trap, diverge, or print. So it enters the family only behind a gate
    /// (see [`CategoryIr::elem_law`]): the body must be trap-free, loop-free
    /// and effect-free, because a consumer that recomputes it would otherwise
    /// be free to delete a trap, turn `Trapped` into `Diverged`, or reorder an
    /// observable — the S34 failure class one level up.
    ///
    /// Capture identity is deliberately **not** required: nothing is spliced.
    /// The consumer emits a call to the same body, so this is two calls in one
    /// loop rather than a merged body.
    Apply {
        body: FuncId,
        source: ObjectId,
        captures: u32,
        inner: Box<ElemSrc>,
        /// The producer's own materialized output array. A consumer that
        /// DECLINES to recompute (profitability is the backend's call) reads
        /// this instead — so a declined sub-law degrades to a load rather than
        /// poisoning the law that contains it. Without it, refusing an `Apply`
        /// nested inside a `Pair` would collapse the whole pair back to an
        /// array-of-structs read, silently undoing the `Zip` win.
        array: ObjectId,
        /// The law is piecewise: outside an index interval the cell is a
        /// captured array's own cell ([`Outside`]).
        outside: Option<Outside>,
    },
}

/// **The piece of a `Map`'s law outside a set of cells** (generic-fusion
/// natural forms): `out[t] = x[t]` for every `t` that is not one of the cells
/// `lo + step·k`, `k ∈ [0, len)`; `x` the map's capture `slot`, which the
/// body reads at `t` and nowhere else; on those cells the body's other arm.
/// `lo = d·(a·e + b)`, `a·e + b` in `e`'s own wrapping int arithmetic (the
/// program's value; `e` a captured int scalar of the map, `None`: `lo = d·b`),
/// the product exact. `len = u64::MAX`: to the array's end.
///
/// A row write `(t ÷ d == r) ? new(t) : x[t]` is one (`[d·r, d·r + d)`), and
/// so is `(lo ≤ t ∧ t < lo + len) ? new(t) : x[t]` (`d = 1`; `n − 1 + len`
/// fits the index type, so `lo + len` never wraps where the interval meets
/// `[0, n)`), and a column write `(t % s == r) ? new(t) : x[t]` (`step = s`,
/// from `r`: `t ≥ 0`, so `t % s` is exact and only `r ∈ [0, s)` matches).
/// Only those cells within `[0, n)` are computed; every other cell is the
/// capture's — which is what lets a row write run in place over its row
/// ([`CategoryIr::write_delta`]'s map step, the emitter's range map).
///
/// device-geometry `piece-delta`: `w > 1` makes each of those positions a
/// piece of `w` consecutive cells — a row every `s` rows, `((t ÷ d) % s ==
/// r) ? new(t) : x[t]` (`lo = d·r`, `step = d·s`, `w = d`, `len = u64::MAX`):
/// one sequence's row of a `[B][L][…]` cache at position `r`, B pieces. Every
/// piece lies inside `[0, n)` (`n % d == 0`). `w = 1` on every other form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outside {
    pub slot: u32,
    pub d: u64,
    pub a: i64,
    pub e: Option<ObjectId>,
    pub b: i64,
    pub step: u64,
    pub len: u64,
    pub w: u64,
}

/// The per-fn element-law plan — the deduced query
/// `elem : IR × FuncId → ElemPlan`, the BL7 pattern alongside [`tile_plan`] and
/// [`last_use_plan`] (plan-s37-stage-structure).
///
/// **Only non-[`ElemSrc::Load`] laws are recorded.** Absence means "load it",
/// which is the status quo — so, like every query here, saying nothing is always
/// safe and a consumer that ignores the plan entirely is still correct.
///
/// Non-destructive by design: this records that an intermediate array *could* be
/// skipped and never removes it. The graph, the count `Constant` that
/// `validate`'s `IotaCountMismatch` ties, and every downstream recognizer's
/// witness all survive — which is what leaves the elide-vs-materialize decision
/// with the backend, and what keeps a deliberate materialization (S27 rung-3
/// packing) expressible.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElemPlan {
    pub(super) src: SecondaryMap<ObjectId, ElemSrc>,
}

impl ElemPlan {
    /// The element law of `arr`, or `None` when it must be loaded.
    pub fn src(&self, arr: ObjectId) -> Option<&ElemSrc> {
        self.src.get(arr)
    }

    /// How many arrays in the fn have a closed-form element law.
    pub fn len(&self) -> usize {
        self.src.len()
    }

    /// Whether no array in the fn has one.
    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }
}
