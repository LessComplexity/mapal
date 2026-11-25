//! **`step_algebra`** (fold-algebra): what a function's arithmetic may become —
//! which float products may fuse with the add that reads them (one rounding,
//! FMA) and whether a fold body's step may be regrouped (lane partials + a
//! tree). Admission is algebra and dataflow only; the read geometry along the
//! fold axis is recorded for the layout a consumer picks, never for admission.
//! One fact, stored on the graph; the host emitter and the device kernels read
//! it, so both realize the same arithmetic.
//!
//! **`fold_combine`** (long-rows): how a fold's state over a row splits into
//! blocks of its axis — partial states combined by the step's own `⊕` — and
//! which values constant along the axis (one per row: another fold's result)
//! its term reads, each by the law the term's dataflow gives it: a factor
//! applied once after the combine, a shift of a max/min applied once after
//! it, a rescale (the value enters only through `exp` of a difference: each
//! partial takes its own shift, rescaled when combined — ADR-0042 D7), or a
//! phase (the partials need that value combined first). Read off the fold's
//! step and the cell map its term reads; never a pattern name.

use slotmap::SecondaryMap;

use crate::algo::{Footprint, RowFoldOp, TileRead};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation};
use crate::ty::Ty;

/// A float `Add`/`Sub` that reads the product of a float `Mul` read by nothing
/// else: the pair may be one fused rounding where the face contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FmaPair {
    /// The `Mul` whose product is the fused operand.
    pub mul: MorphismId,
    /// The operand slot of the add/sub the product sits in (the first one
    /// when both operands are such products: one pair per add).
    pub slot: u32,
    /// The pair is `x − y` (`slot 0`: `a·b − c`, `slot 1`: `c − a·b`).
    pub sub: bool,
}

/// A fold body's step that may run as partial folds combined by its own `⊕`
/// ([`CategoryIr::fold_tree_step`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Regroup {
    /// The `Fold` morphism whose body this is.
    pub fold: MorphismId,
    /// The step morphism in the body (`acc ⊕ e`, or the max/min's `Phi`).
    pub step: MorphismId,
    pub op: RowFoldOp,
    /// Regrouping keeps the bits (integer steps). A float step's partials
    /// round differently: admitted where the order is free (ADR-0040 D2).
    pub exact: bool,
    /// Trip count (the folded array's length).
    pub k: u64,
    /// Per read of the step's term (`e`, or the factors when `e` is an
    /// [`FmaPair`] product): its stride along the fold axis — `Some(1)` the
    /// folded element itself, `Some(ck)` a whole fold's affine read, `None`
    /// unknown (a gather or a non-affine read). Layout, not admission.
    pub reads: Vec<Option<u64>>,
}

/// [`CategoryIr::step_algebra`]'s fields for one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepAlgebra {
    /// Keyed by the add/sub of each pair.
    pub fma: SecondaryMap<MorphismId, FmaPair>,
    /// Keyed by each pair's `Mul`: its add/sub.
    pub fused_mul: SecondaryMap<MorphismId, MorphismId>,
    /// Set when the function is a fold body whose step may be regrouped.
    pub regroup: Option<Regroup>,
}

/// The fast-math flags one realization gives a morphism: `contract` on both
/// halves of an [`FmaPair`] where the face contracts, `reassoc` on a float
/// regroup step where the order is free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepFlags {
    pub contract: bool,
    pub reassoc: bool,
}

impl StepAlgebra {
    /// The one realization rule both backends read. `contract`: the face fuses
    /// steps (ADR-0040 D1); `free`: the order is unspecified; `pairs` /
    /// `regroup`: the two OFF arms (`--no-fma-pairs`, `--no-fold-regroup`).
    pub fn flags(
        &self,
        m: MorphismId,
        contract: bool,
        free: bool,
        pairs: bool,
        regroup: bool,
    ) -> StepFlags {
        StepFlags {
            contract: contract
                && pairs
                && (self.fma.contains_key(m) || self.fused_mul.contains_key(m)),
            reassoc: free
                && regroup
                && self
                    .regroup
                    .as_ref()
                    .is_some_and(|g| g.step == m && !g.exact),
        }
    }
}

impl CategoryIr {
    /// The deduced query `step_algebra : IR × FuncId → StepAlgebra`. Stored on
    /// the graph.
    pub fn step_algebra(&self, f: FuncId) -> StepAlgebra {
        self.fact(f, |s| &s.step_algebra, || self.step_algebra_compute(f))
    }

    fn step_algebra_compute(&self, f: FuncId) -> StepAlgebra {
        let mut alg = StepAlgebra::default();
        let Some(fd) = self.func(f) else {
            return alg;
        };
        for &s in &fd.morphisms {
            if let Some(p) = self.fma_pair(s) {
                alg.fused_mul.insert(p.mul, s);
                alg.fma.insert(s, p);
            }
        }
        alg.regroup = self.regroup_of(f, &alg);
        alg
    }

    fn is_float_obj(&self, o: ObjectId) -> bool {
        matches!(self.objects.get(o).map(|x| &x.ty), Some(Ty::Float { .. }))
    }

    /// `s` is a float add/sub one of whose operands is a single-reader float
    /// product, and `s` is the only reader of its operand pair.
    fn fma_pair(&self, s: MorphismId) -> Option<FmaPair> {
        let sm = self.morphisms.get(s)?;
        if !matches!(sm.op, Operation::Add | Operation::Sub) || !self.is_float_obj(sm.target) {
            return None;
        }
        if self.out_edges(sm.source) != [s] {
            return None;
        }
        (0..2).find_map(|slot| {
            let x = self.pair_slot_source(sm.source, slot)?;
            let &[mul] = self.in_edges(x) else {
                return None;
            };
            let mm = self.morphisms.get(mul)?;
            if mm.op != Operation::Mul || !self.is_float_obj(x) {
                return None;
            }
            // The product feeds this pair's one slot and nothing else.
            let &[e] = self.out_edges(x) else {
                return None;
            };
            let pe = &self.morphisms[e];
            (matches!(pe.op, Operation::Pair { slot: k, .. } if k == slot)
                && pe.target == sm.source)
                .then_some(FmaPair {
                    mul,
                    slot,
                    sub: sm.op == Operation::Sub,
                })
        })
    }

    /// `f`'s regroup when it is the body of a fold whose step admits partials.
    fn regroup_of(&self, f: FuncId, alg: &StepAlgebra) -> Option<Regroup> {
        let (fold, captures) = self.morphisms.iter().find_map(|(m, x)| match x.op {
            Operation::Fold { body, captures } if body == f => Some((m, captures)),
            _ => None,
        })?;
        let op = self.fold_tree_step(fold)?;
        let fd = self.func(f)?;
        let &[step] = self.in_edges(fd.output) else {
            return None;
        };
        let fm = &self.morphisms[fold];
        let k = match self.objects.get(fm.source)?.ty.component_ty(captures + 1)? {
            Ty::Array { size, .. } => *size,
            _ => return None,
        };
        let exact = !self.is_float_obj(fd.output);
        // The term `e`: the step's non-accumulator operand (a max/min's other
        // compared value); its reads along the fold axis.
        let acc = self
            .out_edges(fd.input)
            .iter()
            .map(|&p| &self.morphisms[p])
            .find(|p| p.op == (Operation::Proj { index: captures }))
            .map(|p| p.target)?;
        let elem = self
            .out_edges(fd.input)
            .iter()
            .map(|&p| &self.morphisms[p])
            .find(|p| {
                p.op == (Operation::Proj {
                    index: captures + 1,
                })
            })
            .map(|p| p.target);
        let sm = &self.morphisms[step];
        let term = match op {
            RowFoldOp::Arith { acc_first, .. } => {
                self.pair_slot_source(sm.source, u32::from(acc_first))?
            }
            RowFoldOp::Select { .. } => {
                let (p, q) = (
                    self.pair_slot_source(sm.source, 0)?,
                    self.pair_slot_source(sm.source, 1)?,
                );
                if p == acc { q } else { p }
            }
        };
        let factors: Vec<ObjectId> = match alg.fma.get(step) {
            Some(pair) => {
                let mm = &self.morphisms[pair.mul];
                vec![
                    self.pair_slot_source(mm.source, 0)?,
                    self.pair_slot_source(mm.source, 1)?,
                ]
            }
            None => vec![term],
        };
        let whole = self.whole_site_of(fold).map(|site| site.a.ck);
        // The folded element, or a component of it (a zipped pair's `p.0`),
        // advances one element per step.
        let of_elem = |mut o: ObjectId| loop {
            if Some(o) == elem {
                return true;
            }
            match self.in_edges(o) {
                [p] if matches!(self.morphisms[*p].op, Operation::Proj { .. }) => {
                    o = self.morphisms[*p].source;
                }
                _ => return false,
            }
        };
        // placement `lanes`: else a captured array's affine read (through a
        // narrow widening) — its address's coefficient of the fold index
        // (`radix_plan` of the body: one unscaled `k` digit), `w[row·K + k]`
        // read at unit stride along the fold.
        let forms = self.radix_plan(f);
        let affine = |o: ObjectId| -> Option<u64> {
            let o = match self.tile_definer(o, Operation::Widen) {
                Some(w) => w.source,
                None => o,
            };
            let ix = self.tile_definer(o, Operation::Index)?;
            let form = forms.get(self.pair_slot_source(ix.source, 1)?)?;
            match form
                .digits
                .iter()
                .filter(|d| d.slot == captures + 1)
                .collect::<Vec<_>>()
                .as_slice()
            {
                [d] if d.stride == 1 && d.radix == 0 => Some(d.coef),
                _ => None,
            }
        };
        let reads = factors
            .iter()
            .map(|&o| {
                if of_elem(o) {
                    Some(1)
                } else {
                    whole.or_else(|| affine(o))
                }
            })
            .collect();
        Some(Regroup {
            fold,
            step,
            op,
            exact,
            k,
            reads,
        })
    }

    /// The whole-fold site of `fold` in its own function, if it is one.
    fn whole_site_of(&self, fold: MorphismId) -> Option<crate::algo::RowFoldSite> {
        let owner = self.try_owner(self.morphisms.get(fold)?.target)?;
        self.whole_fold(owner).get(fold).cloned()
    }
}

/// How a value constant along a fold's axis enters its term
/// ([`FoldCombine::deps`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepLaw {
    /// The term is `x · v` (`div`: `x ÷ v`) through products and quotients
    /// only, and `⊕ = +`: the fold is `(Σ x) · v` (`÷ v`) — distributivity, a
    /// regrouping where the order is free (ADR-0040 D2). `x` is the term with
    /// every read of `v` replaced by `1`, bit for bit (`y · 1 = y`, `y ÷ 1 =
    /// y`): the partials never read `v`, which is applied once after the
    /// combine.
    Factor { div: bool },
    /// The term is `x + v` or `x − v` (`sub`) at its top, and `⊕` a running
    /// max / min: the fold is `(⊕ x) ± v` — rounding is monotone, so the
    /// extreme of the shifted values is the shifted extreme. `x` is the term
    /// with `v` read as the add's identity (`−0`, `+0` under a subtract).
    Shift { sub: bool },
    /// `⊕ = +` and the term is `exp(a(v)) · y` through products and
    /// quotients whose other operand never reads `v`, with `a(v) = κ·(x − v)`
    /// (`x` and every factor of `κ` free of `v`; [`RescaleArg`]): for any
    /// shift `s`, `exp(a(v)) = exp(a(s)) · exp(a(v) − a(s))` and
    /// `a(v) − a(s) = a(x := s, v)` is the same for every term — ADR-0042
    /// D7, `exp(a − m₁) = exp(a − m) · exp(m − m₁)`. A block's partial may
    /// take its own shift `s_B` for `v` and is rescaled by
    /// `exp(a(x := s_B, v := s))` when combined: exact in the reals, within
    /// the default face's bound in floats (each rescale one `exp` and one
    /// product); the exact and fixed faces keep the phase.
    Rescale,
    /// Any other dependence: the partials need `v`'s final value — a phase
    /// after `v`'s own fold.
    Phase,
}

/// [`DepLaw::Rescale`]'s exponent `a = κ·(x − v)`: where it is in the cell
/// body and what the shift of a block may be.
#[derive(Clone, Debug, PartialEq)]
pub struct RescaleArg {
    /// The `Exp` morphism (in the cell body that reads `v`).
    pub exp: MorphismId,
    /// The difference is `x − v` (`false`: `v − x`).
    pub x_first: bool,
    /// Steps from the difference to the exponent: `Mul` / `Div` by a
    /// positive constant (its value), in order.
    pub steps: Vec<(Operation, crate::ty::Value)>,
    /// `κ > 0`: the sign of `x` in the exponent — a running max of `x` as
    /// each block's shift keeps every term at most `1` (`false`: a running
    /// min does).
    pub up: bool,
    /// The array whose same-cell read is `x` (the cell map's capture read
    /// at the cell's own index), when `x` is one.
    pub against: Option<ObjectId>,
}

/// One value constant along a fold's axis that its term reads.
#[derive(Clone, Debug, PartialEq)]
pub struct RowDep {
    /// The per-cell map whose output the fold reads, and which reads `value`.
    pub via: MorphismId,
    /// `via`'s source slot of `value`.
    pub slot: u32,
    /// One element per row of the fold (or one number): another fold's
    /// result, a per-row cell, an input.
    pub value: ObjectId,
    pub law: DepLaw,
    /// [`DepLaw::Rescale`]'s exponent; `None` for every other law.
    pub rescale: Option<RescaleArg>,
}

/// **The fold-state combine of one fold site** ([`CategoryIr::fold_combine`]):
/// `out[i] = seed ⊕ (⊕_{j<k} t_i(j))` computed as partial states over
/// disjoint blocks of `j`, combined by the step's own `⊕` (associative and
/// commutative: `+`, `×`, a running max / min), each partial with the
/// per-block transform `deps` give it.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldCombine {
    pub op: RowFoldOp,
    /// Regrouping keeps the bits (an integer step). A float `+` / `×` is a
    /// regrouping admitted where the order is free.
    pub exact: bool,
    pub rows: u64,
    pub k: u64,
    /// The array the term reads (a tile site's row operand `a`).
    pub term: ObjectId,
    /// The values constant along `j` that the term reads through `term`'s
    /// cell map (and, as [`DepLaw::Phase`], through the cells that map reads
    /// at its own cell).
    pub deps: Vec<RowDep>,
}

/// [`CategoryIr::fold_combine`]: every fold site of one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoldCombinePlan {
    /// Keyed by the site: a row fold's or tile site's `Map`, a whole fold's
    /// `Fold`.
    pub sites: SecondaryMap<MorphismId, FoldCombine>,
}

impl FoldCombinePlan {
    /// The folds `site`'s partials need combined first: its
    /// [`DepLaw::Phase`] values.
    pub fn phase_values(&self, site: MorphismId) -> Vec<ObjectId> {
        self.sites.get(site).map_or(Vec::new(), |c| {
            c.deps
                .iter()
                .filter(|d| d.law == DepLaw::Phase)
                .map(|d| d.value)
                .collect()
        })
    }
}

/// flash-attention F1: a mixed-radix row operand that reads whole rows of
/// `k` cells — `a = Σ_i coef·digit(i) + kk`, `kk < k`, no lane digit, every
/// row digit's coefficient a multiple of `k` — under a digit permutation of
/// the rows: the row it reads, `ρ(i) = Σ (coef ÷ k)·digit(i)` (`base ÷ k`
/// added), as digits of `i`.
pub fn whole_rows(a: &crate::RadixAddr, k: u64) -> Option<(u64, Vec<crate::AxisDigit>)> {
    let contiguous = matches!(a.k[..], [d] if d.stride == 1 && d.radix == k && d.coef == 1);
    if !contiguous || !a.lane.is_empty() || k == 0 || !a.base.is_multiple_of(k) {
        return None;
    }
    let mut out = Vec::new();
    for d in &a.i {
        if !d.coef.is_multiple_of(k) {
            return None;
        }
        out.push(crate::AxisDigit {
            stride: d.stride,
            radix: d.radix,
            coef: d.coef / k,
        });
    }
    Some((a.base / k, out))
}

/// Cells followed through same-cell reads when collecting a term's deps.
const COMBINE_DEPTH: u32 = 4;

impl CategoryIr {
    /// **The fold-state combine** of every fold site of `f` (module doc):
    /// built from `row_fold`, `whole_fold`, `tile_plan_strided` and
    /// `cell_plan`. Stored on the graph.
    pub fn fold_combine(&self, f: FuncId) -> FoldCombinePlan {
        self.fact(f, |s| &s.fold_combine, || self.fold_combine_compute(f))
    }

    fn fold_combine_compute(&self, f: FuncId) -> FoldCombinePlan {
        let mut plan = FoldCombinePlan::default();
        let Some(fd) = self.func(f) else {
            return plan;
        };
        let folds = self.row_fold(f);
        let whole = self.whole_fold(f);
        // flash-attention F3: every tile plan — the strided sites, the
        // epilogue sites (`tile-post`: the epilogue runs after the fold, on
        // its combined value), the narrow ones (an exact widening: the term's
        // laws are the element's) and the mixed-radix ones (a row operand
        // read whole rows at a time under a digit permutation of the rows:
        // the row's constant values are its row's, `a` read as `ci = k`).
        let mut tiles = self.tile_plan_strided(f);
        for (m, s) in self.tile_plan_post(f).sites.iter() {
            tiles.sites.insert(m, s.clone());
        }
        for (m, s) in self.tile_plan_narrow(f).sites.iter() {
            tiles.sites.insert(m, s.clone());
        }
        for (m, s) in self.tile_plan_radix(f).iter() {
            let mut s = s.clone();
            match s.radix.as_ref().and_then(|r| r.a.as_ref()) {
                Some(a) if whole_rows(a, s.k).is_some() => {
                    s.a = TileRead {
                        base: 0,
                        ci: s.k,
                        ck: 1,
                        clane: 0,
                        ksplit: None,
                        ..s.a.clone()
                    };
                }
                Some(_) => continue,
                None => {}
            }
            tiles.sites.insert(m, s);
        }
        for &m in &fd.morphisms {
            let Some(morph) = self.morphisms.get(m) else {
                continue;
            };
            let (site_op, read, rows, k, w, elem, src) = if let Some(s) = folds.get(m) {
                (s.op, s.a.clone(), s.rows, s.k, s.w, &s.elem, morph.source)
            } else if let Some(s) = whole.get(m) {
                (s.op, s.a.clone(), s.rows, s.k, s.w, &s.elem, morph.source)
            } else if let Some(s) = tiles.sites.get(m) {
                let op = RowFoldOp::Arith {
                    op: Operation::Add,
                    acc_first: s.add_acc_first,
                };
                (op, s.a.clone(), s.rows, s.k, 1, &s.elem, morph.source)
            } else {
                continue;
            };
            let tile = tiles.sites.contains_key(m);
            let Some(term) = self.pair_slot_source(src, read.slot) else {
                continue;
            };
            let exact = !matches!(elem, Ty::Float { .. });
            // Per row `i` the term reads `term[base + ci·i + ck·j]`, `j < k`:
            // a same-row read of a cell `t` is constant along `j` where it is
            // `x[t ÷ c]` with `c` the row's stride and the row inside it.
            let aligned = w == 1
                && read.clane == 0
                && read.ksplit.is_none()
                && read.base + read.ck * k.saturating_sub(1) < read.ci.max(1)
                || rows == 1 && read.ksplit.is_none();
            let mut deps = Vec::new();
            if aligned {
                self.term_deps(f, &read, rows, k, term, site_op, tile, 0, &mut deps);
            }
            plan.sites.insert(
                m,
                FoldCombine {
                    op: site_op,
                    exact,
                    rows,
                    k,
                    term,
                    deps,
                },
            );
        }
        plan
    }

    /// The deps of a term reading `term` (module doc): its cell map's reads
    /// constant along the fold's axis, each with its law; at `depth > 0` (a
    /// cell the term reads through another at its own cell) every one a phase.
    #[allow(clippy::too_many_arguments)]
    fn term_deps(
        &self,
        f: FuncId,
        read: &TileRead,
        rows: u64,
        k: u64,
        term: ObjectId,
        op: RowFoldOp,
        tile: bool,
        depth: u32,
        deps: &mut Vec<RowDep>,
    ) {
        let cells = self.cell_plan(f);
        let &[c] = self.in_edges(term) else {
            return;
        };
        let (Some(site), Some(cm)) = (cells.get(c), self.morphisms.get(c)) else {
            return;
        };
        let Operation::Map { body, captures } = cm.op else {
            return;
        };
        let len = |o: ObjectId| match self.objects.get(o).map(|x| &x.ty) {
            Some(Ty::Array { size, .. }) => *size,
            _ => 0,
        };
        // The row's last cell the term reads, from the row's first.
        let span = read.base + read.ck * k.saturating_sub(1);
        let constant = |foot: Footprint, v: ObjectId| match foot {
            Footprint::Row { c } if rows > 1 => c == read.ci && span < c && len(v) == rows,
            Footprint::Row { c } => span < c,
            Footprint::Const(_) => true,
            _ => false,
        };
        let mut seen: Vec<u32> = Vec::new();
        for r in &site.reads {
            let slot = if captures > 0 { Some(r.slot) } else { None };
            let Some(v) = slot.map_or(Some(cm.source), |s| self.pair_slot_source(cm.source, s))
            else {
                continue;
            };
            if r.foot == Footprint::Same && depth < COMBINE_DEPTH && v != term {
                let before = deps.len();
                self.term_deps(f, read, rows, k, v, op, tile, depth + 1, deps);
                // A rescale holds through this cell when its output is a
                // product / quotient path in the inner cell's read (`p =
                // ex ÷ den` over `ex = exp(s − m)`): the term is still
                // `exp(a(v)) · y`. Every other law needs the value first.
                let carry = self.factor_in_slot(body, captures, r.slot, op).is_some();
                for d in &mut deps[before..] {
                    if !(carry && d.law == DepLaw::Rescale) {
                        d.law = DepLaw::Phase;
                        d.rescale = None;
                    }
                }
                continue;
            }
            if !constant(r.foot, v) || seen.contains(&r.slot) {
                continue;
            }
            seen.push(r.slot);
            let (law, rescale) = self.dep_law_at(body, captures, r.slot, op, tile, depth, cm);
            deps.push(RowDep {
                via: c,
                slot: r.slot,
                value: v,
                law,
                rescale,
            });
        }
        for &u in &site.uniform {
            let Some(v) = self.pair_slot_source(cm.source, u) else {
                continue;
            };
            let (law, rescale) = self.dep_law_at(body, captures, u, op, tile, depth, cm);
            deps.push(RowDep {
                via: c,
                slot: u,
                value: v,
                law,
                rescale,
            });
        }
    }

    /// [`Self::dep_law`] at `depth`: below the term's own cell only a
    /// rescale survives (it composes through the product paths above it;
    /// [`Self::term_deps`] checks each level).
    #[allow(clippy::too_many_arguments)]
    fn dep_law_at(
        &self,
        body: FuncId,
        captures: u32,
        slot: u32,
        op: RowFoldOp,
        tile: bool,
        depth: u32,
        cm: &crate::graph::Morphism,
    ) -> (DepLaw, Option<RescaleArg>) {
        if let Some(arg) = self.rescale_law(body, captures, slot, op, cm) {
            return (DepLaw::Rescale, Some(arg));
        }
        match depth {
            0 => (self.dep_law(body, captures, slot, op, tile), None),
            _ => (DepLaw::Phase, None),
        }
    }

    /// The reads of slot `s` in body `b` and the objects they reach, as the
    /// `is_read` / `free` predicates the law paths take.
    fn slot_taint(&self, b: FuncId, s: u32) -> (Vec<ObjectId>, SecondaryMap<ObjectId, ()>) {
        let reads = self.slot_reads(b, s);
        let mut tainted: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut stack = reads.clone();
        while let Some(o) = stack.pop() {
            if tainted.insert(o, ()).is_some() {
                continue;
            }
            for &m in self.out_edges(o) {
                stack.push(self.morphisms[m].target);
            }
        }
        (reads, tainted)
    }

    /// Body `b`'s output is a product / quotient path in its slot `s`
    /// (`Some(div)`, [`Self::factor_path`]) and `⊕ = +`.
    fn factor_in_slot(&self, b: FuncId, captures: u32, s: u32, op: RowFoldOp) -> Option<bool> {
        let add = matches!(
            op,
            RowFoldOp::Arith {
                op: Operation::Add,
                ..
            }
        );
        if !add || captures == 0 {
            return None;
        }
        let bd = self.func(b)?;
        let (reads, tainted) = self.slot_taint(b, s);
        if reads.is_empty() {
            return None;
        }
        let is_read = |o: ObjectId| reads.contains(&o);
        let free = |o: ObjectId| !tainted.contains_key(o);
        self.factor_path(bd.output, &is_read, &free)
    }

    /// [`DepLaw::Rescale`] of body `b`'s output in its slot `s` (`cm` the
    /// cell's map): `⊕ = +`, the output a product / quotient path to an
    /// `Exp` whose argument is `κ·(x − v)`, `v` the slot's read, `x` and the
    /// other factors free of it. Off under `FactArms::exp_rescale`.
    fn rescale_law(
        &self,
        b: FuncId,
        captures: u32,
        s: u32,
        op: RowFoldOp,
        cm: &crate::graph::Morphism,
    ) -> Option<RescaleArg> {
        if !self.fact_arms().exp_rescale || captures == 0 {
            return None;
        }
        let add = matches!(
            op,
            RowFoldOp::Arith {
                op: Operation::Add,
                ..
            }
        );
        if !add {
            return None;
        }
        let bd = self.func(b)?;
        let (reads, tainted) = self.slot_taint(b, s);
        if reads.is_empty() {
            return None;
        }
        let is_read = |o: ObjectId| reads.contains(&o);
        let free = |o: ObjectId| !tainted.contains_key(o);
        let exp = self.exp_factor_path(bd.output, &free)?;
        let em = self.morphisms.get(exp)?;
        let mut steps = Vec::new();
        let mut a = em.source;
        // Peel `· c` / `÷ c` (c a positive constant) back to the difference.
        let (x, x_first) = loop {
            let &[m] = self.in_edges(a) else {
                return None;
            };
            let mm = self.morphisms.get(m)?;
            let p = self.pair_slot_source(mm.source, 0)?;
            let q = self.pair_slot_source(mm.source, 1)?;
            match mm.op {
                Operation::Sub if free(p) && is_read(q) => break (p, true),
                Operation::Sub if is_read(p) && free(q) => break (q, false),
                Operation::Mul if self.positive_const(q) => {
                    steps.push((Operation::Mul, self.objects[q].value.clone()?));
                    a = p;
                }
                Operation::Mul if self.positive_const(p) => {
                    steps.push((Operation::Mul, self.objects[p].value.clone()?));
                    a = q;
                }
                Operation::Div if self.positive_const(q) => {
                    steps.push((Operation::Div, self.objects[q].value.clone()?));
                    a = p;
                }
                _ => return None,
            }
        };
        steps.reverse();
        // `x` is a capture read at the cell's own index: that capture.
        let against = self.same_read_capture(bd, captures, x, cm);
        Some(RescaleArg {
            exp,
            x_first,
            steps,
            up: x_first,
            against,
        })
    }

    /// `o` is `exp(a) · y` through products and quotients whose other
    /// operand is `free` (`exp(a)` never in a denominator): the `Exp`.
    fn exp_factor_path(&self, o: ObjectId, free: &dyn Fn(ObjectId) -> bool) -> Option<MorphismId> {
        let &[m] = self.in_edges(o) else {
            return None;
        };
        let mm = self.morphisms.get(m)?;
        if !self.is_float_obj(o) {
            return None;
        }
        if mm.op == Operation::Exp {
            return Some(m);
        }
        let x = self.pair_slot_source(mm.source, 0)?;
        let y = self.pair_slot_source(mm.source, 1)?;
        match mm.op {
            Operation::Mul if free(y) => self.exp_factor_path(x, free),
            Operation::Mul if free(x) => self.exp_factor_path(y, free),
            Operation::Div if free(y) => self.exp_factor_path(x, free),
            _ => None,
        }
    }

    /// `o` is a float `Constant` greater than zero.
    fn positive_const(&self, o: ObjectId) -> bool {
        self.objects.get(o).is_some_and(|x| {
            x.kind == crate::graph::ObjectKind::Constant
                && match x.value {
                    Some(crate::ty::Value::F32(v)) => v > 0.0,
                    Some(crate::ty::Value::F64(v)) => v > 0.0,
                    _ => false,
                }
        })
    }

    /// The capture array of map `cm` that body `bd` reads at its own element
    /// (the mapped index) to make `x`, if `x` is such a read.
    fn same_read_capture(
        &self,
        bd: &crate::graph::FuncDef,
        captures: u32,
        x: ObjectId,
        cm: &crate::graph::Morphism,
    ) -> Option<ObjectId> {
        let &[m] = self.in_edges(x) else {
            return None;
        };
        let ix = self.morphisms.get(m)?;
        if ix.op != Operation::Index {
            return None;
        }
        let arr = self.pair_slot_source(ix.source, 0)?;
        let at = self.pair_slot_source(ix.source, 1)?;
        let slot = self.tile_input_proj_index(arr, bd.input)?;
        if slot >= captures || self.tile_input_proj_index(at, bd.input) != Some(captures) {
            return None;
        }
        self.pair_slot_source(cm.source, slot)
    }

    /// The objects of cell body `b` that read its source slot `s`: the
    /// `Index` results on capture `s`, or the capture itself (a number).
    fn slot_reads(&self, b: FuncId, s: u32) -> Vec<ObjectId> {
        let Some(bd) = self.func(b) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &p in self.out_edges(bd.input) {
            let pm = &self.morphisms[p];
            if pm.op != (Operation::Proj { index: s }) {
                continue;
            }
            let cap = pm.target;
            if !matches!(self.objects.get(cap).map(|o| &o.ty), Some(Ty::Array { .. })) {
                out.push(cap);
                continue;
            }
            for &g in &bd.morphisms {
                let x = &self.morphisms[g];
                if x.op == Operation::Index && self.pair_slot_source(x.source, 0) == Some(cap) {
                    out.push(x.target);
                }
            }
        }
        out
    }

    /// [`DepLaw`] of body `b`'s output in its slot `s` (module doc).
    fn dep_law(&self, b: FuncId, captures: u32, s: u32, op: RowFoldOp, tile: bool) -> DepLaw {
        let Some(bd) = self.func(b) else {
            return DepLaw::Phase;
        };
        if captures == 0 {
            return DepLaw::Phase;
        }
        let reads = self.slot_reads(b, s);
        if reads.is_empty() {
            return DepLaw::Phase;
        }
        // Everything the reads reach inside the body (a product holding one
        // holds it whole: an over-approximation, so a law is never claimed
        // where the value leaks).
        let mut tainted: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut stack = reads.clone();
        while let Some(o) = stack.pop() {
            if tainted.insert(o, ()).is_some() {
                continue;
            }
            for &m in self.out_edges(o) {
                stack.push(self.morphisms[m].target);
            }
        }
        let is_read = |o: ObjectId| reads.contains(&o);
        let free = |o: ObjectId| !tainted.contains_key(o);
        let add = matches!(
            op,
            RowFoldOp::Arith {
                op: Operation::Add,
                ..
            }
        );
        if add && let Some(div) = self.factor_path(bd.output, &is_read, &free) {
            return DepLaw::Factor { div };
        }
        // A tile's step is `acc + a·b`: a shift of `a` is no shift of the term.
        if !tile && matches!(op, RowFoldOp::Select { .. }) {
            let [m] = self.in_edges(bd.output) else {
                return DepLaw::Phase;
            };
            let sm = &self.morphisms[*m];
            let (Some(x), Some(y)) = (
                self.pair_slot_source(sm.source, 0),
                self.pair_slot_source(sm.source, 1),
            ) else {
                return DepLaw::Phase;
            };
            match sm.op {
                Operation::Add if is_read(y) && free(x) || is_read(x) && free(y) => {
                    return DepLaw::Shift { sub: false };
                }
                Operation::Sub if is_read(y) && free(x) => return DepLaw::Shift { sub: true },
                _ => {}
            }
        }
        DepLaw::Phase
    }

    /// `o` is `x · v` (`Some(false)`) or `x ÷ v` (`Some(true)`) with `v` a
    /// read of the slot reached only through products and quotients whose
    /// other operand never reads it.
    fn factor_path(
        &self,
        o: ObjectId,
        is_read: &dyn Fn(ObjectId) -> bool,
        free: &dyn Fn(ObjectId) -> bool,
    ) -> Option<bool> {
        if is_read(o) {
            return Some(false);
        }
        let &[m] = self.in_edges(o) else {
            return None;
        };
        let mm = self.morphisms.get(m)?;
        if !matches!(self.objects.get(o).map(|x| &x.ty), Some(Ty::Float { .. })) {
            return None;
        }
        let x = self.pair_slot_source(mm.source, 0)?;
        let y = self.pair_slot_source(mm.source, 1)?;
        match mm.op {
            Operation::Mul if free(y) => self.factor_path(x, is_read, free),
            Operation::Mul if free(x) => self.factor_path(y, is_read, free),
            Operation::Div if free(y) => self.factor_path(x, is_read, free),
            Operation::Div if free(x) => self.factor_path(y, is_read, free).map(|d| !d),
            _ => None,
        }
    }
}
