//! Tile-site types of the facts catalog: `TilePlan`, `TileSite` and the operand, tensor-form,
//! move, cell and footprint shapes the tile / move / cell plans return.

use super::*;

/// Map/fold sites whose iteration space may be tiled without changing any
/// per-cell operation order.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan {
    /// Recognized `Operation::Map` morphisms and their complete emission data.
    pub sites: SecondaryMap<MorphismId, TileSite>,
}

/// One affine array read in a tiled map/fold site.
#[derive(Clone, PartialEq, Eq)]
pub struct TileRead {
    pub slot: u32,
    pub base: u64,
    pub ci: u64,
    pub ck: u64,
    pub clane: u64,
    /// The `k÷div`/`k%div` decomposition when this read's address is affine in
    /// the fold's derived axes instead of raw `k` (`ck == 0` then). `None` on
    /// plain affine reads.
    pub ksplit: Option<TileKSplit>,
    /// ADR-0043: the operand array stores a narrow format and the product
    /// reads its exact widening to the site's `elem` (`w[k] -> widen<f32>`);
    /// `None`: the array's elements are `elem` itself. Only
    /// [`CategoryIr::tile_plan_narrow`]'s sites carry one.
    pub narrow: Option<crate::ty::NarrowFloat>,
}

/// The derived `Debug`'s text, `narrow` printed only where it is a format:
/// the backends key kernel ids on this text (`site_key`, row-fold and unit
/// ids), so a read over the product's own elements keys as it always did.
impl std::fmt::Debug for TileRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("TileRead");
        d.field("slot", &self.slot)
            .field("base", &self.base)
            .field("ci", &self.ci)
            .field("ck", &self.ck)
            .field("clane", &self.clane)
            .field("ksplit", &self.ksplit);
        if let Some(n) = self.narrow {
            d.field("narrow", &n);
        }
        d.finish()
    }
}

impl TileRead {
    /// Bytes one element of the operand array occupies (its storage, not the
    /// product's width).
    pub fn elem_bytes(&self, elem: &Ty) -> u64 {
        match self.narrow {
            Some(n) => n.bytes(),
            None => elem.scalar_bytes().unwrap_or(4),
        }
    }

    /// **hopper-wgmma-tma F1 (`tensor_form`): this operand as a dense tensor
    /// over the site's axes** (`rows` along `I`, `lanes` along `Lane`, `k`
    /// along `K`). For every point `(i, lane, k)` of the site's domain the
    /// read's element is `base + Σ stride_d · coordinate_d`, its dimensions
    /// ordered innermost first (ascending stride), each one's stride covering
    /// the span of the one inside it — no two axes overlap, so the box at
    /// coordinates `(c_d)` with sides `(s_d)` is exactly the panel of those
    /// extents ([`TensorForm::box_of`]). An axis the read does not move along
    /// (stride 0, or extent 1) is no dimension. `None`: an address through
    /// derived `k` digits (`ksplit`: a window's `i + k`, the im2col class) or
    /// two axes that overlap.
    pub fn tensor_form(&self, rows: u64, lanes: u64, k: u64) -> Option<TensorForm> {
        if self.ksplit.is_some() {
            return None;
        }
        let mut dims: Vec<TensorDim> = [
            (TensorAxis::I, rows, self.ci),
            (TensorAxis::Lane, lanes, self.clane),
            (TensorAxis::K, k, self.ck),
        ]
        .into_iter()
        .filter(|&(_, extent, stride)| stride > 0 && extent > 1)
        .map(|(axis, extent, stride)| TensorDim {
            axis,
            extent,
            stride,
        })
        .collect();
        dims.sort_by_key(|d| d.stride);
        let dense = dims
            .windows(2)
            .all(|w| w[1].stride >= w[0].stride * w[0].extent);
        dense.then_some(TensorForm {
            base: self.base,
            dims,
        })
    }
}

/// hopper-wgmma-tma F1: an axis of a tile site's iteration space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorAxis {
    /// The output row (`i`).
    I,
    /// The output column (`lane`).
    Lane,
    /// The fold axis (`k`).
    K,
}

/// One dimension of a [`TensorForm`]: `extent` coordinates along `axis`,
/// `stride` elements apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDim {
    pub axis: TensorAxis,
    pub extent: u64,
    pub stride: u64,
}

/// A tile operand as a dense tensor ([`TileRead::tensor_form`]): element
/// `base + Σ stride_d · c_d`, dimensions innermost first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorForm {
    pub base: u64,
    pub dims: Vec<TensorDim>,
}

impl TensorForm {
    /// The axes of the dimensions, innermost first.
    pub fn axes(&self) -> Vec<TensorAxis> {
        self.dims.iter().map(|d| d.axis).collect()
    }

    /// **The box that is a panel**: its side along each dimension, `side` of
    /// that dimension's axis (at most the extent), and whether every side
    /// divides its extent — then the boxes at multiples of the sides tile the
    /// tensor exactly, none reaching past it.
    pub fn box_of(&self, side: impl Fn(TensorAxis) -> u64) -> (Vec<u64>, bool) {
        let sides: Vec<u64> = self
            .dims
            .iter()
            .map(|d| side(d.axis).min(d.extent))
            .collect();
        let exact = self
            .dims
            .iter()
            .zip(&sides)
            .all(|(d, s)| *s > 0 && d.extent.is_multiple_of(*s));
        (sides, exact)
    }
}

/// How [`CategoryIr::tile_site`] reads a fold: lanes at a stride other than
/// 1 admitted (`strided`), operands through an exact widening of narrow
/// storage admitted (`narrow`, ADR-0043).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TileForm {
    pub(super) strided: bool,
    pub(super) narrow: bool,
}

/// The `k = kq·div + kr` decomposition of a fold's counted axis: the read's
/// address is affine in the derived variables `k÷div` (coefficient `cq`) and
/// `k%div` (coefficient `cr`) — conv2d's `k/3`,`k%3` window taps. The same
/// derived-var move the map body gets for `(t÷C, t%C)`, one level down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileKSplit {
    /// The literal divisor shared by the fold body's `Div`/`Mod` pair.
    pub div: u64,
    /// Address coefficient on `k÷div`.
    pub cq: u64,
    /// Address coefficient on `k%div`.
    pub cr: u64,
}

/// Affine address coefficients over the tile axes:
/// `base + ci·i + clane·lane + ck·k + cq·(k÷div) + cr·(k%div)`.
#[derive(Clone, Copy, Default)]
pub(super) struct TileAffine {
    pub(super) base: u64,
    pub(super) ci: u64,
    pub(super) clane: u64,
    pub(super) ck: u64,
    pub(super) cq: u64,
    pub(super) cr: u64,
}

impl TileAffine {
    pub(super) fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            base: self.base.checked_add(other.base)?,
            ci: self.ci.checked_add(other.ci)?,
            clane: self.clane.checked_add(other.clane)?,
            ck: self.ck.checked_add(other.ck)?,
            cq: self.cq.checked_add(other.cq)?,
            cr: self.cr.checked_add(other.cr)?,
        })
    }

    pub(super) fn scale(self, by: u64) -> Option<Self> {
        Some(Self {
            base: self.base.checked_mul(by)?,
            ci: self.ci.checked_mul(by)?,
            clane: self.clane.checked_mul(by)?,
            ck: self.ck.checked_mul(by)?,
            cq: self.cq.checked_mul(by)?,
            cr: self.cr.checked_mul(by)?,
        })
    }
}

/// A cell address over the map index `t`: `base + cq·(t÷w) + cr·(t%w)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CellAff {
    pub(super) w: u64,
    pub(super) base: u64,
    pub(super) cq: u64,
    pub(super) cr: u64,
}

/// [`CellAff`] mid-walk: raw `t` (`ct`) apart, the divisor once one is seen.
#[derive(Clone, Copy, Default)]
pub(super) struct CellAffRaw {
    pub(super) w: Option<u64>,
    pub(super) base: u64,
    pub(super) ct: u64,
    pub(super) cq: u64,
    pub(super) cr: u64,
}

impl CellAffRaw {
    pub(super) fn add(self, o: Self) -> Option<Self> {
        let w = match (self.w, o.w) {
            (Some(a), Some(b)) if a != b => return None,
            (a, b) => a.or(b),
        };
        Some(Self {
            w,
            base: self.base.checked_add(o.base)?,
            ct: self.ct.checked_add(o.ct)?,
            cq: self.cq.checked_add(o.cq)?,
            cr: self.cr.checked_add(o.cr)?,
        })
    }

    pub(super) fn scale(self, by: u64) -> Option<Self> {
        Some(Self {
            w: self.w,
            base: self.base.checked_mul(by)?,
            ct: self.ct.checked_mul(by)?,
            cq: self.cq.checked_mul(by)?,
            cr: self.cr.checked_mul(by)?,
        })
    }
}

/// **A fold-less move site (S45): a `map` that only MOVES data.** Its body is a
/// single proven `Index` whose address is affine in the derived axes
/// `(t÷C, t%C)` — a transpose, a gather-by-formula, any permutation-shaped read.
///
/// It is the arm [`CategoryIr::tile_site`] cannot take. That recognizer requires
/// a fold in the map body (`let fold_id = fold_id?`), because it exists to find
/// reductions; a permutation has none, so before this record no fact about a
/// transpose's 2-D geometry existed anywhere in the graph, and S44 had to accept
/// the width **typed in by hand on a flag**. The affine machinery it needs was
/// already here one level up: `tile_split` binds the same `Div`/`Mod` pair and
/// [`TileKSplit`] records the same decomposition for a fold's counted axis.
///
/// **Geometry only, and that is a boundary, not an omission (ADR-0032).** Every
/// field is a property of the program: how wide the iteration space is, how far
/// apart consecutive reads land, what the element is. Whether that stride
/// collides in some cache, and what to do about it, needs a line size and an
/// associativity — machine facts this crate must never learn. The backend joins
/// the two (`TargetProfile::move_block`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoveSite {
    /// `C` — the width of the 2-D iteration space, i.e. the divisor the body
    /// itself decomposes the counter by.
    pub width: u64,
    /// `extent / width`, ≥ 2.
    pub rows: u64,
    /// Address coefficient on `t÷C` (the slow axis).
    pub cq: u64,
    /// Address coefficient on `t%C` (the fast axis) — **the walk's stride, in
    /// elements**. Multiplied by the element width it is the byte stride whose
    /// set-index behaviour decides everything downstream.
    pub cr: u64,
    /// The read array's element type. A width in bytes, not a machine fact.
    pub elem: Ty,
    /// The read array's length in elements.
    pub len: u64,
}

/// Fold-less [`MoveSite`]s recognized in one function — the peer of [`TilePlan`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MovePlan {
    pub sites: SecondaryMap<MorphismId, MoveSite>,
}

/// **Which cells of one array a per-cell morphism's cell `t` reads** (S54).
///
/// The read footprint existed in pieces (`TileRead` for a fold's operands,
/// `MoveSite` for a permutation) and nowhere whole for the per-cell maps a
/// backend wants to place next to their producer: a cell that reads only
/// `x[t]` can run where `x[t]` is computed, and one that reads `x[t ÷ c]` can
/// run anywhere `x` is complete. Geometry only (ADR-0032).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Footprint {
    /// `x[t]`.
    Same,
    /// `x[t ÷ c]` — one cell per row of `c` cells (a per-row scale).
    Row { c: u64 },
    /// `x[t % c]` — one cell per column (a bias).
    Col { c: u64 },
    /// `x[k]`, one fixed cell.
    Const(u64),
    /// Any other address (data-dependent, or not one of the above forms).
    Other,
}

/// One read of a per-cell morphism: slot `slot` of the map's source product
/// (for a capture-free map, slot 0 is the mapped array itself), at `foot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellRead {
    pub slot: u32,
    pub foot: Footprint,
}

/// A `Map` whose cell `t` is a clean leaf function of a fixed set of input
/// cells (S54): `out[t] = body(x₁[f₁(t)], …, u₁, …)`. `reads` lists every
/// array read the body makes, the mapped element included unless it is the
/// index itself (an `iota`). A number capture (`uniform`) is one value every
/// cell reads alike — a loop counter, a fold's result (cpu-gpu-boundary).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellSite {
    /// bf16-class `device-trap` ([`CategoryIr::trap_cell_plan`]): the
    /// body's `Index` reads `bounds_proof` cannot prove — each a run-time
    /// check where the cell runs (an index outside its array records the
    /// program's trap and reads element 0; the trap is raised at the next
    /// host checkpoint, the oracle's outcome). Empty in [`CategoryIr::cell_plan`].
    pub traps: Vec<MorphismId>,
    /// The number of cells, `len(out)`.
    pub extent: u64,
    pub reads: Vec<CellRead>,
    pub body: FuncId,
    /// The capture slots holding a number (not an array), ascending.
    pub uniform: Vec<u32>,
    /// The capture slots a nested `Map`/`Fold` of the body reads (through its
    /// own captures, any proven address: `reads` holds each as
    /// [`Footprint::Other`]) — a dot product or a whole fold per cell.
    pub nested: Vec<u32>,
    /// Element steps one cell takes: 1, plus each nested `Map`/`Fold`'s
    /// extent times its own body's steps (a dot product over `k`: `1 + k`).
    pub work: u64,
    /// recurrence-precision item 2: the body's one nested fold that reads the
    /// map's captured arrays, when every other nested `Map`/`Fold` reads none
    /// (device-geometry `row-tree`: a norm's root after its sum of squares)
    /// and its step is `acc ⊕ g(…)` with `⊕` a `+`, `×` or running max/min
    /// and `g` blind to `acc` ([`CategoryIr::fold_tree_step`]) — `(fold,
    /// step, extent)`: where the order is free its steps may be split across
    /// lanes and combined as a tree.
    pub tree: Option<(MorphismId, RowFoldOp, u64)>,
}

/// **Which producer cells a consumer's cell `t` reads** (generic-fusion,
/// [`CategoryIr::footprint_plan`]): the read footprint of every recognized
/// consumer in one form, translated from the recognizers' own records
/// ([`TileRead`], [`RowFoldSite`], [`MoveSite`], [`CellRead`]) — nothing new
/// is derived here. Geometry only (ADR-0032).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reach {
    /// `base + cq·(t÷w) + cr·(t%w) + Σ coef·k`, each tap's `k ∈ [0, ext)`.
    Affine {
        w: u64,
        base: u64,
        cq: u64,
        cr: u64,
        taps: Vec<Tap>,
    },
    /// A proven address in `[lo, hi)` that is not affine in `t`.
    Range { lo: u64, hi: u64 },
    /// The producer's element is a closed form rooted at the index
    /// ([`ElemSrc`] over [`ElemSrc::Index`]): evaluated at the read, no bytes.
    Law,
    /// Unproven or data-dependent: any cell.
    Whole,
}

/// One counted axis of a read: `coef·k`, `k ∈ [0, ext)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tap {
    pub coef: u64,
    pub ext: u64,
}

/// One read edge: slot `slot` of `consumer`'s source reads `producer` at `reach`.
/// `law` is set when the edge sees through a capture-free classifiable map
/// (`producer` is that map's input, `law` its output: the read is
/// `body(producer[addr])`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub consumer: MorphismId,
    pub slot: u32,
    pub producer: ObjectId,
    pub reach: Reach,
    pub law: Option<ObjectId>,
}

/// Every consumer's read edges in one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FootprintPlan {
    pub edges: SecondaryMap<MorphismId, Vec<Edge>>,
}

impl FootprintPlan {
    /// The edges of `consumer` (empty: nothing recognized, read it whole).
    pub fn of(&self, consumer: MorphismId) -> &[Edge] {
        self.edges.get(consumer).map_or(&[], Vec::as_slice)
    }
}

impl Reach {
    /// The producer rectangle `[rows, cols]` (the producer laid out `pw`
    /// wide) that the consumer cells `t` with `t÷w ∈ rows`, `t%w ∈ cols`
    /// read — the tight hull of the addresses. `None` when the column terms
    /// are not confined within one producer row (a coefficient neither a
    /// multiple of `pw` nor small enough), or the reach is [`Reach::Whole`].
    /// A law reads no bytes: the empty rectangle. Proven reads lie in
    /// `[0, len)` already, so the hull needs no clipping.
    pub fn rect(
        &self,
        pw: u64,
        rows: std::ops::Range<u64>,
        cols: std::ops::Range<u64>,
    ) -> Option<[std::ops::Range<u64>; 2]> {
        let empty = Some([0..0, 0..0]);
        if pw == 0 {
            return None;
        }
        match self {
            Reach::Whole => None,
            Reach::Law => empty,
            Reach::Range { lo, hi } => {
                if lo >= hi {
                    return empty;
                }
                let (r0, r1) = (lo / pw, (hi - 1) / pw);
                Some(if r0 == r1 {
                    [r0..r0 + 1, lo % pw..(hi - 1) % pw + 1]
                } else {
                    [r0..r1 + 1, 0..pw]
                })
            }
            Reach::Affine {
                base, cq, cr, taps, ..
            } => {
                if rows.is_empty() || cols.is_empty() || taps.iter().any(|t| t.ext == 0) {
                    return empty;
                }
                let terms = [
                    (*cq, rows.start, rows.end - 1),
                    (*cr, cols.start, cols.end - 1),
                ]
                .into_iter()
                .chain(taps.iter().map(|t| (t.coef, 0, t.ext - 1)));
                let (mut r0, mut r1) = (base / pw, base / pw);
                let (mut c0, mut c1) = (base % pw, base % pw);
                for (coef, lo, hi) in terms {
                    if coef % pw == 0 {
                        r0 = r0.checked_add((coef / pw).checked_mul(lo)?)?;
                        r1 = r1.checked_add((coef / pw).checked_mul(hi)?)?;
                    } else {
                        c0 = c0.checked_add(coef.checked_mul(lo)?)?;
                        c1 = c1.checked_add(coef.checked_mul(hi)?)?;
                    }
                }
                (c1 < pw).then_some([r0..r1 + 1, c0..c1 + 1])
            }
        }
    }

    /// The single address the consumer cell `t` reads, for a tap-free affine
    /// reach (`None` otherwise).
    pub fn at(&self, t: u64) -> Option<u64> {
        match self {
            Reach::Affine {
                w,
                base,
                cq,
                cr,
                taps,
            } if taps.is_empty() && *w > 0 => Some(base + cq * (t / w) + cr * (t % w)),
            _ => None,
        }
    }
}

/// Everything a backend needs to emit one recognized tiled map site.
#[derive(Clone, Debug, PartialEq)]
pub struct TileSite {
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    pub a: TileRead,
    pub b: TileRead,
    pub seed: Value,
    pub elem: Ty,
    pub mul_a_first: bool,
    pub add_acc_first: bool,
    /// The addresses that are digit-affine only ([`CategoryIr::tile_plan_radix`]);
    /// `None` on every [`CategoryIr::tile_plan`] site.
    pub radix: Option<Box<RadixTile>>,
    /// ADR-0041 D1: the site's product admits `tf32` (`precision_plan.set`).
    pub tf32: bool,
    /// ADR-0043 D4: the 16-bit member the site's product admits
    /// ([`crate::Precision::half`]): multiplicands rounded to it, f32
    /// accumulation — a tensor-core product where the part has the shape.
    pub half: Option<crate::ty::NarrowFloat>,
    /// device-geometry `tile-post`: the map body goes on past its fold — the
    /// cell's other terms (a residual, a scale, a mask) computed from the
    /// fold's value, the cell index and captured arrays ([`TilePost`]).
    /// `None` on every [`CategoryIr::tile_plan`] / `tile_plan_strided` site;
    /// [`CategoryIr::tile_plan_post`] holds the sites that carry one.
    pub post: Option<TilePost>,
}

impl TileSite {
    /// The weakest member's multiplicand unit roundoff (ADR-0041 D3,
    /// [`crate::Precision::unit`]): `None` where the product is `{f32}`.
    pub fn unit(&self) -> Option<f64> {
        crate::Precision {
            f32: true,
            tf32: self.tf32,
            bf16: self.half == Some(crate::ty::NarrowFloat::Bf16),
            f16: self.half == Some(crate::ty::NarrowFloat::F16),
        }
        .unit()
    }

    /// ADR-0043 slice 3 (per-operand units, D3): each multiplicand's unit
    /// roundoff under the weakest member — an operand stored narrow in a
    /// format that embeds in the member's ([`crate::narrow::NarrowForm::embeds_in`])
    /// is the member's value unchanged (its rounding is the identity: 0);
    /// any other operand rounds with the member's unit. `None` where the
    /// product is `{f32}`.
    pub fn units(&self) -> Option<(f64, f64)> {
        let u = self.unit()?;
        let member = self.half.map(crate::narrow::NarrowForm::of);
        let op = |r: &TileRead| match (r.narrow, member) {
            (Some(n), Some(m)) if crate::narrow::NarrowForm::of(n).embeds_in_form(&m) => 0.0,
            _ => u,
        };
        Some((op(&self.a), op(&self.b)))
    }
}

/// **The cell work after a tile site's fold** (device-geometry `tile-post`,
/// [`CategoryIr::tile_plan_post`]): the map body `out[t] = g(fold(t), t,
/// caps)` — a sum of products followed by elementwise terms of the same cell
/// (`x[t] + y`, `y / s`, `(j ≤ i) ? y : −∞`). The epilogue is derived from
/// the dataflow, not matched: the body itself with its fold's value given is
/// the function a store evaluates per cell. Geometry only, like
/// [`TileSite`]: which captured slots the epilogue reads, and whether the
/// fold sits in a guard arm (then the site folds every cell and the arm
/// decides per cell; every read of the fold is in bounds over the whole
/// iteration space, checked on the site's affine reads).
#[derive(Clone, PartialEq, Eq)]
pub struct TilePost {
    /// The body's one fold, whose value the epilogue takes.
    pub fold: MorphismId,
    /// Capture slots (`< captures`) holding arrays the work after the fold
    /// reads, ascending.
    pub reads: Vec<u32>,
    /// Capture slots holding a number (not an array) the body reads, ascending.
    pub uniform: Vec<u32>,
    /// The fold runs under a guard arm of the body (a mask): the site
    /// computes it for every cell.
    pub guarded: bool,
    /// sweep-attention `unit-uniform`: of `uniform`, the slots the work
    /// after the fold reads — every other one only the fold's source takes (a
    /// loop counter moving an operand's address: [`crate::RadixTile::ua`]),
    /// so where this is empty an epilogue evaluated without the numbers is
    /// the body's.
    pub after: Vec<u32>,
    /// bf16-class `round-post`: the cell's result rounded once to this
    /// narrow format as the map stores it (`precision_plan().store`, the
    /// array's every reader a product admitting it): the body's last step a
    /// `Narrow`, the map's output array that format. The fold, its operands
    /// and every other term stay the site's (`TileSite::elem`).
    pub round: Option<crate::ty::NarrowFloat>,
}

/// The derived `Debug`'s text, `after` printed only where it is not every
/// number (`uniform`): the backends key kernel ids on this text, so a site
/// whose numbers the epilogue all reads keys as it always did.
impl std::fmt::Debug for TilePost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("TilePost");
        d.field("fold", &self.fold)
            .field("reads", &self.reads)
            .field("uniform", &self.uniform)
            .field("guarded", &self.guarded);
        if self.after != self.uniform {
            d.field("after", &self.after);
        }
        if let Some(r) = self.round {
            d.field("round", &r);
        }
        d.finish()
    }
}
