//! **The compile-time fact** (ADR-0048 D3, [`CategoryIr::static_plan`]):
//! which values of a function are computable before the program runs, and
//! whether each one a run-time value reads is embedded or computed at run
//! time.
//!
//! **Computable.** A value whose dataflow cone holds no input of its function
//! and no effect is the same on every call — [`CategoryIr::unchanged_plan`]'s
//! `inputs` mask is 0 (the IO token is an input slot, so every `print`,
//! `time` and `read` result, and everything computed from one, is not). A
//! value of a loop is computable only when the whole loop is (its every
//! decide, advance and exit value): the oracle runs a loop as a unit.
//!
//! **Roots.** The computable values a run-time value reads (or that leave
//! the function): the frontier. A product built by `Pair` slots is never a
//! root — its computable components are. Constants, loop members and the
//! function's own output object are not either.
//!
//! **The rule** — bytes against work. Each root carries its bytes, its scalar
//! cells and the oracle's work to compute it (interpreter steps over its cone;
//! a map or fold `n` times its body's; `None` when the cone holds a loop, whose
//! trips are not known before it runs):
//!
//! - [`Realize::Embed`] — `bytes ≤ EMBED_BYTES`, `work ≤ EVAL_WORK` and
//!   `work > cells`: computing it costs more steps than its cells, so its value
//!   replaces its cone (the oracle computes it; a scalar becomes a constant, a
//!   product a pack of constants).
//! - [`Realize::Attempt`] — `bytes ≤ EMBED_BYTES`, work unknown (a loop): the
//!   oracle tries under `EVAL_WORK` steps and the value is embedded when it
//!   comes out.
//! - [`Realize::Compute`] — otherwise: too large to embed, too much work to do
//!   while compiling, or no cheaper as a value than as its law (an array with
//!   a closed-form element law, [`CategoryIr::elem_plan`] — the backends
//!   rebuild it where it is read; an `iota`, a `fill`, a pack of literals:
//!   no more work than cells). It stays computed at run time — once, where
//!   the facts that follow (`unchanged_plan`, `call_keep`, `elem_plan`)
//!   place it — and its computable operands are roots in turn.
//!
//! `EMBED_BYTES` bounds the realization (one constant and one store per cell
//! in the emitted code), `EVAL_WORK` the compiler's own time; neither is a
//! hardware property.

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, ty_contains_token};

/// The most bytes a root is embedded with: its cells become constants packed
/// into the value (16 KiB: 4096 `f32` cells).
pub const EMBED_BYTES: u64 = 16 << 10;

/// The most oracle steps compile-time evaluation spends on one function's
/// roots (2²⁴: well under a second).
pub const EVAL_WORK: u64 = 1 << 24;

/// How a computable root is realized (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Realize {
    /// Its value replaces its cone.
    Embed,
    /// Its cone holds a loop: evaluated under the budget, embedded when it
    /// comes out.
    Attempt,
    /// Computed at run time.
    Compute,
}

/// One computable root.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticRoot {
    pub value: ObjectId,
    pub bytes: u64,
    pub cells: u64,
    /// Oracle steps over its cone; `None` when the cone holds a loop.
    pub work: Option<u64>,
    pub realize: Realize,
}

/// The per-function fact. Stored on the graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticPlan {
    /// Every computable value of the function.
    pub fixed: SecondaryMap<ObjectId, ()>,
    /// The roots, in object order, each with its realization.
    pub roots: Vec<StaticRoot>,
    /// The oracle's steps for one call of the whole function; `None` when it
    /// holds a loop.
    pub work: Option<u64>,
}

impl StaticPlan {
    /// Whether `o` is computable before the program runs.
    pub fn is_fixed(&self, o: ObjectId) -> bool {
        self.fixed.contains_key(o)
    }

    /// The roots realized as values (embedded, or attempted).
    pub fn evaluated(&self) -> impl Iterator<Item = &StaticRoot> {
        self.roots.iter().filter(|r| r.realize != Realize::Compute)
    }
}

/// `(bytes, cells)` of a value of type `ty` (a string or token counts none).
pub fn ty_extent(ty: &Ty) -> (u64, u64) {
    match ty {
        Ty::Array { elem, size } => {
            let (b, c) = ty_extent(elem);
            (b.saturating_mul(*size), c.saturating_mul(*size))
        }
        Ty::Tuple(ts) => ts.iter().fold((0, 0), |(b, c), t| {
            let (tb, tc) = ty_extent(t);
            (b.saturating_add(tb), c.saturating_add(tc))
        }),
        Ty::Struct { fields, .. } => fields.iter().fold((0, 0), |(b, c), (_, t)| {
            let (tb, tc) = ty_extent(t);
            (b.saturating_add(tb), c.saturating_add(tc))
        }),
        t => (t.scalar_bytes().unwrap_or(0), 1),
    }
}

impl CategoryIr {
    /// **`static_plan`** ([`StaticPlan`]): the values of `f` computable before
    /// the program runs, its roots and how each is realized (ADR-0048 D3).
    /// Stored on the graph; built from `unchanged_plan(f)` (`inputs`),
    /// `loop_structure` / `loop_plan`, and `static_plan` of every callee and
    /// body (`work`).
    pub fn static_plan(&self, f: FuncId) -> StaticPlan {
        self.fact(f, |s| &s.static_plan, || self.static_plan_compute(f))
    }

    fn static_plan_compute(&self, f: FuncId) -> StaticPlan {
        let Some(fd) = self.func(f) else {
            return StaticPlan::default();
        };
        let objs = self.func_objects(f);
        let masks = self.unchanged_plan(f).inputs;
        let mut fixed: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for &o in &objs {
            let Some(obj) = self.object(o) else { continue };
            if obj.kind == ObjectKind::Parameter
                || ty_contains_token(&obj.ty)
                || masks.get(o).copied().unwrap_or(u64::MAX) != 0
            {
                continue;
            }
            fixed.insert(o, ());
        }
        // A loop is computable as a unit: every member of its plan, or none.
        let mut in_loop: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        // A loop's exit values leave it: roots when computable.
        let mut exits: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut has_loop = false;
        for scc in self.loop_structure(f) {
            has_loop = true;
            for &mg in &scc.merges {
                let Some(plan) = self.loop_plan(f, mg) else {
                    continue;
                };
                let mut members: Vec<ObjectId> = scc.objects.clone();
                for &m in plan
                    .decide_order
                    .iter()
                    .chain(&plan.advance_order)
                    .chain(&plan.exits)
                {
                    if let Some(mm) = self.morphism(m) {
                        members.push(mm.target);
                    }
                }
                for &m in &plan.exits {
                    if let Some(mm) = self.morphism(m) {
                        exits.insert(mm.target, ());
                    }
                }
                let whole = members.iter().all(|o| fixed.contains_key(*o));
                for &o in &members {
                    in_loop.insert(o, ());
                    if !whole {
                        fixed.remove(o);
                    }
                }
            }
        }
        // Demotion reaches downstream: a value read from a demoted one is not
        // computable either (to a fixpoint over the function's edges).
        loop {
            let mut moved = false;
            for &m in &fd.morphisms {
                let Some(mm) = self.morphism(m) else { continue };
                if fixed.contains_key(mm.target) && !fixed.contains_key(mm.source) {
                    fixed.remove(mm.target);
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }

        // Work: oracle steps per object over its cone (memoized, saturating;
        // a shared sub-cone counted per path — an over-estimate, so never
        // embedding more than the rule allows).
        let mut work: SecondaryMap<ObjectId, Option<u64>> = SecondaryMap::new();
        for m in self.topo_order(f) {
            let Some(mm) = self.morphism(m) else { continue };
            let src = if fixed.contains_key(mm.source) {
                work.get(mm.source).copied().unwrap_or(Some(0))
            } else {
                Some(0)
            };
            let own = self.step_work(mm.op, mm.source, mm.target);
            let w = match (src, own) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                _ => None,
            };
            let w = if in_loop.contains_key(mm.target) {
                None
            } else {
                w
            };
            let prev = work.get(mm.target).copied().unwrap_or(Some(0));
            let total = match (prev, w) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                _ => None,
            };
            work.insert(mm.target, total);
        }
        let fn_work = if has_loop {
            None
        } else {
            Some(
                fd.morphisms
                    .iter()
                    .filter_map(|&m| self.morphism(m))
                    .map(|mm| self.step_work(mm.op, mm.source, mm.target))
                    .try_fold(0u64, |a, w| w.map(|w| a.saturating_add(w))),
            )
            .flatten()
        };

        // The frontier: computable values a run-time value (or the output)
        // reads, then down through every root realized at run time.
        let mut cands: Vec<ObjectId> = Vec::new();
        for &o in &objs {
            if !fixed.contains_key(o) {
                continue;
            }
            let read_late = self.out_edges(o).iter().any(|&m| {
                self.morphism(m)
                    .is_some_and(|mm| !fixed.contains_key(mm.target) || mm.target == fd.output)
            });
            if read_late {
                cands.push(o);
            }
        }
        // An array with a closed-form element law is cheaper as its law: the
        // backends rebuild it where it is read (`elem_plan`).
        let laws = self.elem_plan(f);
        let mut roots: Vec<StaticRoot> = Vec::new();
        let mut seen: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        while let Some(o) = cands.pop() {
            if seen.insert(o, ()).is_some() {
                continue;
            }
            let Some(obj) = self.object(o) else { continue };
            if obj.kind != ObjectKind::Temporary
                || (in_loop.contains_key(o) && !exits.contains_key(o))
                || o == fd.output
            {
                continue;
            }
            // A product assembled slot by slot: its components are the roots.
            let ins = self.in_edges(o);
            if ins.iter().any(|&m| {
                self.morphism(m)
                    .is_some_and(|mm| matches!(mm.op, Operation::Pair { .. }))
            }) {
                for &m in ins {
                    if let Some(mm) = self.morphism(m)
                        && fixed.contains_key(mm.source)
                    {
                        cands.push(mm.source);
                    }
                }
                continue;
            }
            if matches!(obj.ty, Ty::Str) {
                continue;
            }
            let (bytes, cells) = ty_extent(&obj.ty);
            let w = work.get(o).copied().flatten();
            let realize = match w {
                _ if bytes > EMBED_BYTES || laws.src(o).is_some() => Realize::Compute,
                None => Realize::Attempt,
                Some(w) if w <= EVAL_WORK && w > cells => Realize::Embed,
                Some(_) => Realize::Compute,
            };
            if realize == Realize::Compute {
                for &m in ins {
                    if let Some(mm) = self.morphism(m)
                        && fixed.contains_key(mm.source)
                    {
                        cands.push(mm.source);
                    }
                }
            }
            roots.push(StaticRoot {
                value: o,
                bytes,
                cells,
                work: w,
                realize,
            });
        }
        roots.sort_by_key(|r| objs.iter().position(|&x| x == r.value));
        StaticPlan {
            fixed,
            roots,
            work: fn_work,
        }
    }

    /// The oracle's steps for one morphism: a map / fold `n` times its body's
    /// (plus one a cell), a call its callee's, an array-building op one a
    /// cell; `None` through a loop.
    fn step_work(&self, op: Operation, source: ObjectId, target: ObjectId) -> Option<u64> {
        let cells = |o: ObjectId| {
            self.object(o)
                .map(|x| match &x.ty {
                    Ty::Array { size, .. } => *size,
                    _ => 1,
                })
                .unwrap_or(1)
        };
        let fold_len = |o: ObjectId| {
            self.object(o)
                .and_then(|x| match &x.ty {
                    Ty::Tuple(ts) => ts.iter().rev().find_map(|t| match t {
                        Ty::Array { size, .. } => Some(*size),
                        _ => None,
                    }),
                    _ => None,
                })
                .unwrap_or(1)
        };
        Some(match op {
            Operation::Map { body, .. } => {
                let b = self.static_plan(body).work?;
                cells(target).saturating_mul(b.saturating_add(1))
            }
            Operation::Fold { body, .. } => {
                let b = self.static_plan(body).work?;
                fold_len(source).saturating_mul(b.saturating_add(1))
            }
            Operation::Call(g) => self.static_plan(g).work?.saturating_add(1),
            Operation::Iota | Operation::Fill | Operation::Zip | Operation::Enumerate => {
                cells(target)
            }
            Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => return None,
            _ => 1,
        })
    }
}
