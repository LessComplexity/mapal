//! The invariant-enforcing builder (DESIGN §10) — the only way to construct a
//! [`CategoryIr`]. "Ill-formed is unconstructible": every public call either
//! produces a well-typed edge or returns the exact [`IrError`] of the §16
//! rejection matrix.
//!
//! Structure:
//! - [`IrBuilder`] holds the graph under construction; [`IrBuilder::seal`] runs
//!   the global checks (I4b, I5, I6, I-RET, unclosed loops) and freezes it.
//! - [`FnBuilder`] is scoped to one function; its primitives mint a fresh target
//!   object per call (except the two sanctioned existing-object targets: the
//!   Return object via `Dest::Ret`/`output`, and the LoopMerge via `loop_back`).
//! - Composite primitives (`binop`, `phi`, `index`, `call`, `map`, `fold`,
//!   `print`, `loop_back`, `loop_exit`) build their internal `Pair` packs
//!   **atomically** — callers cannot half-build them.

use slotmap::{SecondaryMap, SlotMap};

use crate::graph::{
    CategoryIr, FuncDef, FuncId, FuncKind, Morphism, MorphismId, Object, ObjectId, ObjectKind,
    Operation, Precision,
};
use crate::loc::SourceLoc;
use crate::ty::{self, Ty, Value};

mod array_ops;
mod core_ops;
mod io_loops;
mod seal_checks;

use self::seal_checks::*;

#[cfg(test)]
mod tests;

/// Where a value-producing primitive writes its result (DESIGN §10).
#[derive(Clone, Debug, PartialEq)]
pub enum Dest {
    /// A fresh object, with an optional debug name.
    Fresh(Option<String>),
    /// The function's Return object; `slot: None` = full value, `Some(k)` = a
    /// `Pair{k, arity}` slot write (output ty must then be a product).
    Ret { slot: Option<u32> },
}

/// Every rejection-matrix row (DESIGN §10.1 / §16). `#[derive(... PartialEq)]`
/// and no `Display` (C3) — renderer-free, like mapal-syntax diag values.
#[derive(Clone, Debug, PartialEq)]
pub enum IrError {
    /// Two `Named` functions declared with the same name.
    DuplicateName,
    /// A `FuncId` not in this builder, or wrong kind for the call site.
    UnknownFunction,
    /// An `ObjectId` not in this builder.
    UnknownObject,
    /// An object from a different function used in this one.
    WrongBuilder,
    /// `build_fn` called twice for the same function.
    AlreadyBuilt,
    /// A function declared but never `build_fn`'d, or its `FnBuilder` dropped
    /// without `finish`.
    UnfinishedFunction,
    /// `unop` with an op outside `{Neg, Not}`.
    NotUnary,
    /// `binop` with an op outside the §10 membership set.
    NotBinary,
    /// A typing-table mismatch (DESIGN §5.1).
    TypeMismatch {
        expected: Ty,
        found: Ty,
        loc: SourceLoc,
    },
    /// A ty outside the Core whitelist / arity floors (I9).
    NonCoreType,
    /// A ty deeper than `MAX_TY_DEPTH` (I10).
    TyTooDeep,
    /// `Proj`/slot write on a non-product ty.
    NotAProduct,
    /// A slot index `≥ arity`.
    SlotOutOfRange,
    /// A pack/call/etc. with the wrong number of components.
    ArityMismatch,
    /// ADR-0018: `enumerate` on an array whose size exceeds `i32::MAX`, so the
    /// index component (`i32`) cannot represent every index (F4/SND-3 precedent).
    EnumerateIndexOverflow,
    /// ADR-0029: an `iota`/`fill` count that is not a `Constant` integer > 0
    /// (a runtime size is ADR-0023 dynamic-array territory, out of Core).
    NonStaticCount,
    /// ADR-0029: source and target are not an allowed numeric widening pair.
    InvalidWiden,
    /// ADR-0042 / ADR-0046: a math builtin (`exp`, `sqrt`, `sin`, `cos`) of a
    /// non-float source.
    InvalidExp,
    /// ADR-0043: `to<bf16>` / `to<f16>` of anything but an `f32`.
    InvalidNarrow,
    /// ADR-0045: a `read` target that is not a one-dimensional array of a
    /// readable element type ([`crate::read::readable`]).
    InvalidRead,
    /// `pack`/`pack_array` of zero components.
    EmptyProduct,
    /// `pack` of exactly one component (no 1-tuples).
    SingletonTuple,
    /// A `Constant` whose `value.ty()` disagrees with a requested ty.
    ValueTyMismatch,
    /// I4: a token-bearing object gains a second token-bearing consumer outside
    /// the sanctioned loop fork.
    TokenNotLinear,
    /// I4b: a token-bearing object with no token-bearing out-edge that is not
    /// the Return object.
    TokenDropped,
    /// I4: a `Phi` whose selected ty contains `IoToken` (recursively).
    TokenInPhi,
    /// I4: a token inside a Map/Fold body (bodies must be token-free).
    TokenInBody,
    /// ADR-0027: a capture whose ty has no runtime representation (`Unit`, an
    /// erased-element array, or an all-erased product) — the body-input
    /// component would have no slot on the backends (LLVM panicked; the
    /// erased rule is L4). `IoToken` captures stay `TokenInBody`; `Str`
    /// captures stay `StrOutsidePrint` (via `pack`).
    ErasedCapture,
    /// §8: a loop whose carried `U` contains a token but a `LoopExit` payload
    /// does not (the token would die in the loop).
    TokenNotEscaping,
    /// I9s: `Str` outside a Constant ty or the `print()`-internal pair.
    StrOutsidePrint,
    /// Two `Struct` tys sharing a name but with different fields.
    StructNameConflict,
    /// `Call` targeting a non-`Named` function.
    CallToBody,
    /// `Map`/`Fold` body of the wrong `FuncKind`.
    BodyKindMismatch,
    /// I6: a cycle in the function reference graph.
    RecursiveCall { cycle: Vec<FuncId> },
    /// A `LoopHandle` dropped (or `finish`ed) without `end_loop`.
    OpenLoop,
    /// `end_loop` with zero `LoopBack` edges recorded.
    LoopWithoutBack,
    /// `end_loop` with zero `LoopExit` edges recorded.
    LoopWithoutExit,
    /// I5: a `LoopBack` whose source is not in the merge's SCC.
    LoopBackOutsideScc,
    /// I-RET: a function mixing full-value and slot Return writers.
    RetMixedWriters,
    /// I-RET: two slot writers to the same Return slot.
    RetSlotConflict,
    /// I-RET: a product Return with a missing slot.
    RetSlotMissing,
    /// `Dest::Ret { slot: Some(_) }` on a non-product output ty.
    RetNotProduct,
    /// A Return writer whose ty disagrees with the declared output ty.
    RetTypeMismatch,
    /// `seal` with no entry / a non-existent entry.
    NoEntry,
    /// `seal` whose entry is not `FuncKind::Named`.
    EntryNotNamed,
}

// --- internal under-construction representation ---------------------------

/// A function being assembled inside the builder (mutable until `finish`).
struct FuncUnderConstruction {
    def: FuncDef,
    built: bool,
    finished: bool,
    /// Open loop count: a `FnBuilder::finish` with this > 0 is `OpenLoop`.
    open_loops: u32,
    /// Per-merge `(back, exit)` edge tallies, so `end_loop` can verify ≥1 each
    /// while `loop_back`/`loop_exit` take `&LoopHandle` (counts live here).
    loop_counts: Vec<(ObjectId, u32, u32)>,
}

/// The graph under construction (DESIGN §10). Append-only; `seal` freezes it.
pub struct IrBuilder {
    objects: SlotMap<ObjectId, Object>,
    morphisms: SlotMap<MorphismId, Morphism>,
    out_edges: SecondaryMap<ObjectId, Vec<MorphismId>>,
    in_edges: SecondaryMap<ObjectId, Vec<MorphismId>>,
    owner: SecondaryMap<ObjectId, FuncId>,
    funcs: SlotMap<FuncId, FuncUnderConstruction>,
    /// Named-function name → FuncId, for duplicate detection. Insertion-ordered
    /// Vec (no HashMap; I12) — function counts are tiny.
    named: Vec<(String, FuncId)>,
    precision: SecondaryMap<MorphismId, Precision>,
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrBuilder {
    /// A fresh, empty builder.
    pub fn new() -> Self {
        IrBuilder {
            objects: SlotMap::with_key(),
            morphisms: SlotMap::with_key(),
            out_edges: SecondaryMap::new(),
            in_edges: SecondaryMap::new(),
            owner: SecondaryMap::new(),
            funcs: SlotMap::with_key(),
            named: Vec::new(),
            precision: SecondaryMap::new(),
        }
    }

    /// Declare a function before defining it (DESIGN §10). Calls may reference
    /// fns defined later in source order — acyclicity is a seal check.
    ///
    /// Creates the input Parameter and output Return objects (the latter's
    /// in-edges are filled during `build_fn`). Duplicate `Named` names are
    /// rejected. Both `input` and `output` tys go through Core intake (I9).
    pub fn declare(
        &mut self,
        kind: FuncKind,
        name: &str,
        input: Ty,
        output: Ty,
        loc: SourceLoc,
    ) -> Result<FuncId, IrError> {
        self.intake_ty(&input)?;
        self.intake_ty(&output)?;
        if kind == FuncKind::Named && self.named.iter().any(|(n, _)| n == name) {
            return Err(IrError::DuplicateName);
        }

        // Reserve the FuncId, then mint its input/output objects owned by it.
        let f = self.funcs.insert(FuncUnderConstruction {
            def: FuncDef {
                name: name.to_owned(),
                kind,
                input: ObjectId::default(),
                output: ObjectId::default(),
                morphisms: Vec::new(),
                loc,
            },
            built: false,
            finished: false,
            open_loops: 0,
            loop_counts: Vec::new(),
        });

        let input_obj = self.mint_object(f, input, None, ObjectKind::Parameter, None, loc);
        let output_obj = self.mint_object(f, output, None, ObjectKind::Return, None, loc);
        let fc = &mut self.funcs[f];
        fc.def.input = input_obj;
        fc.def.output = output_obj;

        if kind == FuncKind::Named {
            self.named.push((name.to_owned(), f));
        }
        Ok(f)
    }

    /// Begin building function `f`'s body (DESIGN §10). Once per function.
    pub fn build_fn(&mut self, f: FuncId) -> Result<FnBuilder<'_>, IrError> {
        let fc = self.funcs.get_mut(f).ok_or(IrError::UnknownFunction)?;
        if fc.built {
            return Err(IrError::AlreadyBuilt);
        }
        fc.built = true;
        Ok(FnBuilder { b: self, f })
    }

    /// Seal the graph (DESIGN §9/§10): run every global check and freeze.
    /// `entry` must be a `Named` function.
    pub fn seal(self, entry: FuncId) -> Result<CategoryIr, IrError> {
        // Entry checks.
        let entry_fc = self.funcs.get(entry).ok_or(IrError::NoEntry)?;
        if entry_fc.def.kind != FuncKind::Named {
            return Err(IrError::EntryNotNamed);
        }

        // Every declared function must have been built and finished.
        for (_, fc) in self.funcs.iter() {
            if !fc.built || !fc.finished {
                return Err(IrError::UnfinishedFunction);
            }
            if fc.open_loops > 0 {
                return Err(IrError::OpenLoop);
            }
        }

        // Materialize the CategoryIr so the SCC/topo algorithms can run; the
        // global checks read it, then we hand it back if clean. The func map is
        // converted FuncUnderConstruction → FuncDef preserving every FuncId key.
        let funcs = Self::into_func_defs(self.funcs);
        let ir = CategoryIr {
            objects: self.objects,
            morphisms: self.morphisms,
            out_edges: self.out_edges,
            in_edges: self.in_edges,
            owner: self.owner,
            facts: crate::facts::Facts::new(funcs.keys()),
            funcs,
            entry,
            precision: self.precision,
            data_dir: None,
        };

        // Run the global graph-shape checks on the materialized graph.
        check_sealed(&ir)?;
        Ok(ir)
    }

    /// Convert the under-construction func map into the sealed `FuncDef` map,
    /// preserving every `FuncId` key.
    ///
    /// `SlotMap` re-issues identical keys when a fresh map is filled in the same
    /// insertion order with no intervening removals — which is exactly our case
    /// (v1 graphs are append-only). The `debug_assert_eq!` is the guard.
    fn into_func_defs(funcs: SlotMap<FuncId, FuncUnderConstruction>) -> SlotMap<FuncId, FuncDef> {
        let pairs: Vec<(FuncId, FuncDef)> = funcs.into_iter().map(|(k, fc)| (k, fc.def)).collect();
        let mut out: SlotMap<FuncId, FuncDef> = SlotMap::with_key();
        for (expected, def) in pairs {
            let got = out.insert(def);
            debug_assert_eq!(got, expected, "FuncId key not preserved across seal");
        }
        out
    }

    // --- shared object/edge minting (used by FnBuilder via &mut self) -----

    fn mint_object(
        &mut self,
        owner: FuncId,
        ty: Ty,
        value: Option<Value>,
        kind: ObjectKind,
        name: Option<String>,
        loc: SourceLoc,
    ) -> ObjectId {
        let id = self.objects.insert_with_key(|id| Object {
            id,
            ty,
            value,
            kind,
            name,
            loc,
        });
        self.owner.insert(id, owner);
        self.out_edges.insert(id, Vec::new());
        self.in_edges.insert(id, Vec::new());
        id
    }

    fn add_edge(
        &mut self,
        owner: FuncId,
        source: ObjectId,
        target: ObjectId,
        op: Operation,
        loc: SourceLoc,
    ) -> MorphismId {
        let m = self.morphisms.insert_with_key(|id| Morphism {
            id,
            source,
            target,
            op,
            loc,
        });
        self.out_edges[source].push(m);
        self.in_edges[target].push(m);
        self.funcs[owner].def.morphisms.push(m);
        m
    }

    /// Core type intake (I9 + I10): scalar whitelist, arity floors, depth guard.
    /// Runs on EVERY ty entering the builder, declared or synthesized.
    fn intake_ty(&self, ty: &Ty) -> Result<(), IrError> {
        if !ty::ty_depth_ok(ty) {
            return Err(IrError::TyTooDeep);
        }
        // Iterative structural check (no recursion; J1).
        let mut stack: Vec<&Ty> = vec![ty];
        while let Some(t) = stack.pop() {
            match t {
                Ty::Int { bits, signed } => {
                    let ok = matches!((bits, signed), (32, true) | (64, true) | (8, false));
                    if !ok {
                        return Err(IrError::NonCoreType);
                    }
                }
                Ty::Float { bits } => {
                    if *bits != 32 && *bits != 64 {
                        return Err(IrError::NonCoreType);
                    }
                }
                Ty::Bool | Ty::Unit | Ty::Str | Ty::IoToken | Ty::Narrow(_) => {}
                Ty::Tuple(ts) => {
                    if ts.len() < 2 {
                        return Err(IrError::NonCoreType);
                    }
                    for inner in ts {
                        stack.push(inner);
                    }
                }
                Ty::Struct { fields, .. } => {
                    // Field count ≥ 1 (mirrors Tuple arity ≥ 2 / Array size ≥ 1).
                    // A zero-field Struct is a product `1` whose literal mints a
                    // Temporary with zero in-edges — it seals but fails validate
                    // (BadInEdges). Reject it at intake so it is unconstructible
                    // anywhere (declare, pack_struct, constants; review TY-1).
                    if fields.is_empty() {
                        return Err(IrError::NonCoreType);
                    }
                    for (_, inner) in fields {
                        stack.push(inner);
                    }
                }
                Ty::Array { elem, size } => {
                    if *size < 1 {
                        return Err(IrError::NonCoreType);
                    }
                    stack.push(elem);
                }
            }
        }
        Ok(())
    }
}

// --- LoopHandle -----------------------------------------------------------

/// A handle to an in-progress loop (DESIGN §10). Half-built loops cannot escape:
/// the merge object is reachable only through this handle until `end_loop`.
///
/// Back/exit edge counts are tracked in the builder (keyed by `merge`), not on
/// the handle, so `loop_back`/`loop_exit` take `&LoopHandle` exactly per §10.
#[derive(Debug)]
pub struct LoopHandle {
    merge: ObjectId,
    u_ty: Ty,
}

// --- FnBuilder ------------------------------------------------------------

/// A builder scoped to one function (DESIGN §10). Borrows the [`IrBuilder`].
pub struct FnBuilder<'a> {
    b: &'a mut IrBuilder,
    f: FuncId,
}

/// Internal plan for where a primitive's defining edge lands.
enum DestPlan {
    /// Edge targets this object directly (Fresh, or Ret with no slot).
    Direct(ObjectId),
    /// Edge targets `comp`, then a `Pair{slot, arity}` wires `comp` into `ret`.
    Slot {
        comp: ObjectId,
        ret: ObjectId,
        slot: u32,
        arity: u32,
    },
}

/// Whether a `LoopExit` whose source is `route` belongs to the loop whose merge
/// is `merge` — i.e. `route` is forward-reachable from `merge` (the exit reads
/// merge-derived state). The exit route is a downward leaf outside the SCC, so
/// SCC membership alone cannot attribute it; reachability from the merge can.
///
/// Iterative BFS (J1); stops as soon as `route` is found.
fn exit_reachable_from_merge(ir: &CategoryIr, merge: ObjectId, route: ObjectId) -> bool {
    let mut seen: SecondaryMap<ObjectId, bool> = SecondaryMap::new();
    let mut queue: Vec<ObjectId> = vec![merge];
    seen.insert(merge, true);
    while let Some(o) = queue.pop() {
        if o == route {
            return true;
        }
        for &m in ir.out_edges(o) {
            let t = ir.morphism(m).unwrap().target;
            if !seen.get(t).copied().unwrap_or(false) {
                seen.insert(t, true);
                queue.push(t);
            }
        }
    }
    false
}

/// The carried-state object of a `LoopBack` route (DESIGN §7): the source of
/// the route's slot-0 `Pair` in-edge (the `next_state`; slot 1 is the `cond`).
/// `route` is the `LoopBack` morphism's source object. Returns `None` if the
/// route is not the canonical `(next_state, cond)` shape (a corrupted graph —
/// validate's own pass flags the in-edge shape separately).
fn loopback_state_source(ir: &CategoryIr, route: ObjectId) -> Option<ObjectId> {
    for &m in ir.in_edges(route) {
        let morph = ir.morphism(m)?;
        if let Operation::Pair { slot: 0, .. } = morph.op {
            return Some(morph.source);
        }
    }
    None
}

/// The element count of an `Array` ty, else `None`.
fn array_size_of(ty: &Ty) -> Option<u64> {
    if let Ty::Tuple(ts) = ty
        && let Some(Ty::Array { size, .. }) = ts.get(1)
    {
        return Some(*size);
    }
    None
}
