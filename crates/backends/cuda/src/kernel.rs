//! `kernel.rs` (DESIGN §1 execution mapping, §2 memory model, §3 device trap
//! flag; BC3–BC5, BC8): the device half of the backend. Three pieces:
//!
//! - **BC8 qualifier analysis** ([`Qualifiers`]) — the three-case rule over
//!   the call graph (§1): token-bearing ⇒ host-only; token-free ∧ reachable
//!   from any map/fold body ⇒ device-visible; token-free ∧ NOT body-reachable
//!   ⇒ host-only (a `__device__` instantiation of a fn containing launch-form
//!   ops would need CDP/rdc — HW-1's illegal-CUDA case, out at M3). Of the
//!   body-reachable token-free fns, the ones with **no launch-form ops
//!   anywhere** (transitively), no array-typed return, and no transitive
//!   call to a Twin fn get a single `__host__ __device__` definition; the
//!   rest get a host definition **plus** a `__device__` twin (the inline
//!   form). An array-typed return forces the twin: a single source cannot
//!   both return a device buffer handle (host) and write a per-thread local
//!   result (device, out-param convention) — and Twin-ness propagates
//!   callerward through calls, or the caller's two-site definition would
//!   call the Twin's host definition on the device pass (F4).
//! - **Kernel emission** ([`emit_kernel`], [`emit_kernel_set`]) — one
//!   `__global__` per launch-form array-bulk op site, one-thread-per-element,
//!   256 threads/block, 64-bit index arithmetic (BC3), **deduplicated by
//!   structural shape**: each unique kernel text is emitted once (under the
//!   first site's name) and launched per site — the site's args ride the
//!   launch (suggestions.md #17). Site names: `k{f_ord}_{site_ord}` — `f_ord`
//!   is the fn's ordinal in `ir.funcs()` (the `fn{f_ord}` scheme's ordinal),
//!   and `site_ord` counts the fn's launch-form bulk-op sites in
//!   `topo_order`.
//! - **`__device__` twins** ([`DevEmit`]) — the inline form: the per-thread
//!   sequential op table for body-reachable fns that need device code
//!   distinct from their host definition. Twin names: `d_{fname}`.
//!
//! **The trap-pointer convention (§3).** The device trap flag rides every
//! trap-CAPABLE kernel launch and device-side call as a trailing
//! `unsigned int* trap` argument — #14's [`TrapCaps`] pre-pass drops the
//! convention where no guard can fire (a host `static` global is not
//! addressable from device code, so the flag pointer is threaded
//! explicitly). In-kernel and in-twin guards do `*trap = kind + 1; return …;`
//! — the flag is cudaMemset-zeroed (0 = quiescent), so it stores the
//! mapal-rt kind **plus one** (div_zero ⇒ `1u`, index_oob ⇒ `2u`); the host
//! readback decodes with `mapal_trap(kind - 1)` (module.rs). A bare-kind
//! store would collide: div_zero's 0 would read back as "no trap". After a
//! device-side `Call` to a capable fn the caller emits `if (*trap) return
//! …;` to unwind to the launching kernel (the oracle's first-trap-wins
//! order, §3). `__host__ __device__` fns take the same trailing parameter
//! when capable — host callers pass the host global `d_trap`, unused on the
//! host pass because host guards call `mapal_trap` directly.
//!
//! **The width rule (§3, llvm `func.rs:589–619` ported).** An `Index`/`Update`
//! index operand is extended to `int64_t` *per its `Ty`* — in C++ the
//! conversion `(int64_t)v` is value-preserving for both unsigned (zero-ext)
//! and signed (sign-ext) operands, so one cast realizes llvm's zext/sext
//! split. The bounds check is then a **signed two-sided compare**,
//! `idx < 0 || idx >= (int64_t)n` — never against `size_t` (the usual
//! arithmetic conversions would re-unsigned a negative operand into a huge
//! positive one).
//!
//! **Kernel geometry (BC3).** Elementwise kernels (Map/Zip/Enumerate/Update)
//! use `unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x +
//! threadIdx.x; if (i < n) …` with grid = `ceil(n/256)` computed in 64-bit —
//! `n` is a compile-time constant (it lives in `Ty::Array`), so `n` and the
//! grid expression are baked into the text as `u64` literals. Iota/Fill carry
//! that constant as a `long long n` launch argument and use the same BC3 grid.
//! Top-level `Index` and `Fold` launch `<<<1, 1>>>` (BC4's single thread).

use mapal_ir::{
    BoundsProof, CategoryIr, EmissionClass, EmissionPlan, FuncId, FuncKind, Morphism, MorphismId,
    ObjectId, ObjectKind, Operation, SourceLoc, Ty, Value,
};
use slotmap::SecondaryMap;
use std::collections::HashMap;

use crate::EmitError;
use crate::func::{const_literal, in_place_update, is_float, unsigned_twin};
use crate::ty::{
    erased_index, lower_ty, residual_arity, residual_contains_array,
    tree_contains_product_with_array,
};

// ---------------------------------------------------------------------------
// BC8 qualifier analysis
// ---------------------------------------------------------------------------

/// The BC8 case of one fn (§1). `HostOnly` fns exist only as WP2 host
/// definitions; `HostDevice` fns get a single `__host__ __device__`
/// definition (pure-scalar token-free); `Twin` fns get the host definition
/// plus a `d_{fname}` `__device__` twin emitted by [`DevEmit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FnQual {
    HostOnly,
    HostDevice,
    Twin,
}

/// Per-fn BC8 qualifiers, computed once per module.
pub(crate) struct Qualifiers {
    map: SecondaryMap<FuncId, FnQual>,
}

impl Qualifiers {
    /// The qualifier of `f` (`HostOnly` for unknown ids — a sealed graph's
    /// own fns are all present).
    pub(crate) fn get(&self, f: FuncId) -> FnQual {
        self.map.get(f).copied().unwrap_or(FnQual::HostOnly)
    }

    /// The device-visible name of `g`: the twin for [`FnQual::Twin`], the
    /// plain name for [`FnQual::HostDevice`]. Panics on `HostOnly` — a
    /// token-free body-reachable fn's callees are always device-visible
    /// (token introduction is signature-synthesis-only, ADR-0013 law i, so a
    /// token-free caller implies a token-free, hence body-reachable, callee).
    pub(crate) fn device_name(&self, fnames: &SecondaryMap<FuncId, String>, g: FuncId) -> String {
        match self.get(g) {
            FnQual::Twin => format!("d_{}", fnames[g]),
            FnQual::HostDevice => fnames[g].clone(),
            FnQual::HostOnly => unreachable!("HostOnly fn called from device code"),
        }
    }

    /// The three-case analysis (§1, BC8). One pass over the call graph:
    /// seeds body-reachability from the `MapBody`/`FoldBody` fns, then it
    /// flows calleeward over `Call` edges while **Twin-ness** (a direct
    /// launch-form op, or an array-typed return — the out-param convention
    /// forces the twin) flows **callerward** over `Call` edges, both by
    /// fixpoint (robust to recursion). Callerward propagation of the
    /// array-return rule matters (F4): a body-reachable pure-scalar fn
    /// calling an array-returning Twin must be a Twin itself — as
    /// `__host__ __device__` its single definition would call the callee's
    /// HOST definition on the device pass (illegal CUDA).
    pub(crate) fn analyze(ir: &CategoryIr) -> Qualifiers {
        let mut map: SecondaryMap<FuncId, FnQual> = SecondaryMap::new();

        // Direct facts per fn.
        let mut body_reachable: SecondaryMap<FuncId, bool> = SecondaryMap::new();
        // (bulk ∨ array-return), seeded direct; the fixpoint closes it
        // callerward — after the fixpoint this is "needs the Twin form".
        let mut needs_twin: SecondaryMap<FuncId, bool> = SecondaryMap::new();
        let mut calls: SecondaryMap<FuncId, Vec<FuncId>> = SecondaryMap::new();
        for (f, fd) in ir.funcs() {
            body_reachable.insert(f, fd.kind != FuncKind::Named);
            let mut direct = false;
            for &m in &fd.morphisms {
                let morph = ir.morphism(m).expect("morphism resolves");
                match morph.op {
                    Operation::Map { .. }
                    | Operation::Zip
                    | Operation::Enumerate
                    | Operation::Iota
                    | Operation::Fill
                    | Operation::Fold { .. }
                    | Operation::Index
                    | Operation::Update => direct = true,
                    // An array construction site (pack_array literal): the host
                    // form cudaMalloc+cudaMemcpy's, so it is a launch-form op.
                    Operation::Pair { .. } => {
                        let tty = &ir.object(morph.target).expect("target resolves").ty;
                        if matches!(tty, Ty::Array { .. }) {
                            direct = true;
                        }
                    }
                    Operation::Call(g) => {
                        let mut v = calls.get(f).cloned().unwrap_or_default();
                        v.push(g);
                        calls.insert(f, v);
                    }
                    _ => {}
                }
            }
            let out_ty = &ir.object(fd.output).expect("output resolves").ty;
            // An array-typed return forces the twin (the out-param
            // convention) — but only when the array LOWERS: an erased
            // array (Array{Unit}) has no device buffer, the convention is
            // moot, and a single `__host__ __device__` void definition
            // serves both sites (F6).
            needs_twin.insert(
                f,
                direct || (matches!(out_ty, Ty::Array { .. }) && lower_ty(out_ty).is_some()),
            );
        }

        // Fixpoint: body-reachability flows calleeward over Call edges (a
        // caller of a body-reachable fn is NOT body-reachable — the
        // direction is from bodies outward to what bodies call); Twin-ness
        // flows callerward (a body-reachable caller of a Twin needs the
        // device instantiation of its callee — hence its own twin).
        loop {
            let mut changed = false;
            for (f, _) in ir.funcs() {
                let callees = calls.get(f).cloned().unwrap_or_default();
                for g in callees {
                    if body_reachable[f] && !body_reachable.get(g).copied().unwrap_or(false) {
                        body_reachable.insert(g, true);
                        changed = true;
                    }
                    if needs_twin.get(g).copied().unwrap_or(false) && !needs_twin[f] {
                        needs_twin.insert(f, true);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for (f, fd) in ir.funcs() {
            let in_ty = &ir.object(fd.input).expect("input resolves").ty;
            let out_ty = &ir.object(fd.output).expect("output resolves").ty;
            let token = mapal_ir::ty_contains_token(in_ty) || mapal_ir::ty_contains_token(out_ty);
            let qual = if token || !body_reachable.get(f).copied().unwrap_or(false) {
                // (i) token-bearing ⇒ host-only; (iii) token-free ∧ NOT
                // body-reachable ⇒ host-only (regardless of bulk ops — no
                // device instantiation is ever needed).
                FnQual::HostOnly
            } else if needs_twin.get(f).copied().unwrap_or(false) {
                // (ii) token-free ∧ body-reachable ∧ (launch-form ops ∨
                // array return, transitively through calls) ⇒ twins. An
                // array return forces the twin: the single-definition form
                // cannot serve a device buffer handle on the host and a
                // per-thread local result on the device (out-param
                // convention).
                FnQual::Twin
            } else {
                // (iv) pure-scalar token-free, body-reachable: one
                // `__host__ __device__` definition serves both sites.
                FnQual::HostDevice
            };
            map.insert(f, qual);
        }
        Qualifiers { map }
    }
}

// ---------------------------------------------------------------------------
// #14 trap-capability pre-pass
// ---------------------------------------------------------------------------

/// Per-fn DEVICE trap capability (suggestions.md #14), computed once per
/// module: can the fn's device code (its `__device__` twin or
/// `__host__ __device__` definition, transitively through the calls it
/// makes) set the trap flag? Conservative: any **integer** `Div`/`Mod` (the
/// §3 zero guard), any `Update` or unproven `Index` (the §3 bounds guards)
/// reachable ⇒ capable. Float `Div`/`Mod` never guard (ADR-0013's S13: ÷0 is
/// ±inf/NaN). The S20 refinement: an `Index` the fn's [`BoundsProof`] clears
/// (provably in `[0, n)`) can never fire its guard — it is NOT a trap
/// source, exactly as if the guard were elided; anything unknown/wrapping/
/// loop-carried stays capable (the analysis's `None` answer keeps today's
/// behavior). The device call graph is `Call` edges plus `Map`/`Fold`
/// **body** edges (a twin's inline map/fold loops call the body's device
/// form); capability flows callerward by fixpoint (robust to recursion).
/// The rule is purely syntactic on the op set: an integer `Div`/`Mod` counts
/// even when #13's constant-divisor elision removes its guard (a `t / 4`
/// keeps its fn capable) — keeping the param where no guard survives is
/// harmless, and the two passes stay independently simple.
///
/// A trap-FREE fn or kernel drops the uniform trap convention: no
/// `unsigned int* trap` parameter, no trap argument at its call sites, no
/// post-call `if (*trap)` check, and — for a kernel — no
/// `trap_check_after_launch()` after its launch (fewer host syncs; this is
/// the perf-visible part). Wherever any guard can fire the convention is
/// kept verbatim, so class parity and first-trap-wins are preserved: a
/// dropped post-call check is dead by construction (the callee cannot set
/// the flag), and every capable call keeps its own check, in order.
pub(crate) struct TrapCaps {
    map: SecondaryMap<FuncId, bool>,
    /// The per-fn bounds proofs behind the Index rule (the S20 deduced
    /// query), computed once here — `site` reads them back by the site's
    /// owner fn instead of re-deriving the analysis per query (BL7).
    proofs: SecondaryMap<FuncId, BoundsProof>,
}

impl TrapCaps {
    pub(crate) fn analyze(ir: &CategoryIr) -> TrapCaps {
        let mut map: SecondaryMap<FuncId, bool> = SecondaryMap::new();
        let mut proofs: SecondaryMap<FuncId, BoundsProof> = SecondaryMap::new();
        let mut edges: SecondaryMap<FuncId, Vec<FuncId>> = SecondaryMap::new();
        for (f, fd) in ir.funcs() {
            let proof = ir.bounds_proof(f);
            let mut capable = false;
            for &m in &fd.morphisms {
                let morph = ir.morphism(m).expect("morphism resolves");
                match morph.op {
                    Operation::Div | Operation::Mod => {
                        let src_ty = &ir.object(morph.source).expect("source resolves").ty;
                        if matches!(src_ty.component_ty(0), Some(Ty::Int { .. })) {
                            // #13 credit (S20): a literal non-zero, non-−1
                            // constant divisor cannot trap — the guard elision
                            // already proves the check dead; capability credits it.
                            let safe = const_int_operand(ir, morph.source, 1)
                                .is_some_and(|v| v != 0 && v != -1);
                            if !safe {
                                capable = true;
                            }
                        }
                    }
                    // S20: only an UNPROVEN Index can fire its bounds guard.
                    Operation::Index if !proof.proven(m) => capable = true,
                    // Update stays conservative this wave (no proof query).
                    Operation::Update => capable = true,
                    // ADR-0029: scalar Widen is total and trap-free.
                    Operation::Widen => {}
                    Operation::Call(g) => {
                        let mut v = edges.get(f).cloned().unwrap_or_default();
                        v.push(g);
                        edges.insert(f, v);
                    }
                    Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                        let mut v = edges.get(f).cloned().unwrap_or_default();
                        v.push(body);
                        edges.insert(f, v);
                    }
                    _ => {}
                }
            }
            map.insert(f, capable);
            proofs.insert(f, proof);
        }
        loop {
            let mut changed = false;
            for (f, _) in ir.funcs() {
                if map.get(f).copied().unwrap_or(false) {
                    continue;
                }
                for g in edges.get(f).cloned().unwrap_or_default() {
                    if map.get(g).copied().unwrap_or(false) {
                        map.insert(f, true);
                        changed = true;
                        break;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        TrapCaps { map, proofs }
    }

    /// The capability of `f` (`false` for unknown ids — a sealed graph's own
    /// fns are all present).
    pub(crate) fn get(&self, f: FuncId) -> bool {
        self.map.get(f).copied().unwrap_or(false)
    }

    /// The trap capability of one launch-form bulk-op site — the kernel-side
    /// half of the pre-pass: `Map`/`Fold` ride their body fn's capability
    /// (the per-thread/per-step body call is the only trap source); an
    /// UNPROVEN `Index` / any `Update` guards in-kernel (§3) — a
    /// `bounds_proof`-proven `Index` (S20) can never fire its guard, so the
    /// site is trap-free (no proof cached for the owner fn answers
    /// conservatively: capable); `Zip`/`Enumerate` never trap.
    pub(crate) fn site(&self, ir: &CategoryIr, m: MorphismId) -> bool {
        let morph = ir.morphism(m).expect("morphism resolves");
        match morph.op {
            Operation::Map { body, .. } | Operation::Fold { body, .. } => self.get(body),
            Operation::Index => {
                let owner = ir.owner(morph.target);
                !self.proofs.get(owner).map(|p| p.proven(m)).unwrap_or(false)
            }
            Operation::Update => true,
            // ADR-0029: Iota/Fill are trap-free by construction (like Zip/Enumerate).
            Operation::Zip | Operation::Enumerate | Operation::Iota | Operation::Fill => false,
            _ => unreachable!("not a bulk-op site"),
        }
    }
}

mod bulk;
mod devemit;
mod kernels;
mod layout;
mod sites;
#[cfg(test)]
mod tests;
mod walk;

pub(crate) use self::devemit::*;
pub(crate) use self::kernels::*;
pub(crate) use self::layout::*;
pub(crate) use self::sites::*;

// ---------------------------------------------------------------------------
// Device-side signatures
// ---------------------------------------------------------------------------

/// The `__device__` twin's signature text (no body):
/// `static __device__ {ret} d_{name}({in}, unsigned int* trap)`, or
/// `static __device__ void d_{name}({in}, {T}* out, unsigned int* trap)` for
/// an array-typed return (the out-param convention — a twin's arrays are
/// per-thread locals and cannot be returned by pointer). The convention
/// applies only when the array LOWERS (F6): an erased array return
/// (Array{Unit}) has no buffer to write and lowers to plain `void`. A
/// trap-FREE twin (#14, `caps`) drops the trailing trap parameter — its
/// callers then pass no trap argument and emit no post-call check.
pub(crate) fn twin_signature(
    ir: &CategoryIr,
    f: FuncId,
    fnames: &SecondaryMap<FuncId, String>,
    caps: &TrapCaps,
) -> String {
    let fd = ir.func(f).expect("func resolves");
    let in_ty = &ir.object(fd.input).expect("input resolves").ty;
    let out_ty = &ir.object(fd.output).expect("output resolves").ty;
    let name = format!("d_{}", fnames[f]);
    let mut params: Vec<String> = Vec::new();
    if let Some(t) = lower_ty(in_ty) {
        params.push(format!("{t} in"));
    }
    if matches!(out_ty, Ty::Array { .. }) && lower_ty(out_ty).is_some() {
        params.push(format!("{}* out", flat_base_ct(out_ty).unwrap()));
        if caps.get(f) {
            params.push("unsigned int* trap".into());
        }
        return format!("static __device__ void {name}({})", params.join(", "));
    }
    if caps.get(f) {
        params.push("unsigned int* trap".into());
    }
    let ret = lower_ty(out_ty).unwrap_or_else(|| "void".into());
    format!("static __device__ {ret} {name}({})", params.join(", "))
}

// ---------------------------------------------------------------------------
// DevEmit — the inline form: `__device__` twins (§1, BC8 case ii)
// ---------------------------------------------------------------------------

/// The inline-form emitter: one `__device__` twin per [`FnQual::Twin`] fn —
/// the per-thread sequential op table (§1's right-hand column), which *is*
/// the oracle's evaluation order per thread. Scalar ops emit the same text
/// as the host table (BC2 unsigned-cast wrapping, BC7 strict Phi, `&`/`|`);
/// the divergences are exactly §1's in-body column: `Div`/`Mod`/`Index`/
/// `Update` guards set `*trap = kind` and return; `Map`/`Zip`/`Enumerate`/
/// `Fold` are per-thread sequential loops; array literals are per-thread
/// local initializers (§2 item 8); `Call` is a direct `__device__` call —
/// with the trap pointer threaded and a post-call trap check only when the
/// callee can trap (#14, [`TrapCaps`]; a trap-free fn drops the trailing
/// trap parameter from its twin signature, and its callers pass nothing).
///
/// **Array materialization on device.** An array value *produced* inside the
/// twin (a literal, a Map/Zip/Enumerate/Update/Index/Fold result, a call
/// result, or the fn's Return) is a per-thread local C array `{T} oK[{flat}]`
/// — statically bounded, `n` from the `Ty` (§5). Every other array-typed
/// object (parameters, `Proj`/`Phi` results, **loop merges and exit
/// objects**) is a plain `{T}*` handle; handle copies are pointer
/// assignments (shallow — safe because mutation is either absent or
/// plan-proven: `Update` copies by default, and an in-place `Update`
/// (plan-last-use §2 rule 4, [`in_place_update`]) writes only a buffer the
/// last-use plan proves dead — its old value unobservable). The one place a
/// local array is *assigned into* is elementwise; whole-array "moves" are
/// pointer assignments and happen only between handles. An in-placed twin
/// `Update` declares no produced local for its target: the per-thread copy
/// is skipped, the store lands in the source array, and the target's slot
/// aliases the source's from that point on.
///
/// **Loops in bodies (WP4, §1's inline-form cell).** A body fn may contain
/// a canonical loop (lower emits each body with fresh loop state); the twin
/// emits it as a per-thread sequential `while (true)` quartet under the
/// same `loop_plan` gate — decide cone → guard → advance cone → back edge,
/// with the same driver-ownership skip in the walk. A carried array's merge
/// is a handle (never a produced local), so the back edge is the host
/// driver's pointer swap verbatim; the one cell-wise copy is an exit
/// payload landing in a produced local array (the fn's array Return).
pub(crate) struct DevEmit<'a> {
    ir: &'a CategoryIr,
    f: FuncId,
    fnames: &'a SecondaryMap<FuncId, String>,
    quals: &'a Qualifiers,
    caps: &'a TrapCaps,
    slots: SecondaryMap<ObjectId, String>,
    /// Array-typed objects materialized as per-thread local C arrays (the
    /// produced set above); any other array object is a `{T}*` handle.
    produced: SecondaryMap<ObjectId, ()>,
    /// In-placed `Update` targets (plan-last-use §2 rule 4, computed in
    /// [`DevEmit::new`] from the fn's deduced `last_use_plan` — never
    /// re-derived): the source array dies at the update, so the per-thread
    /// copy is skipped and the store lands in the source's storage; no
    /// produced local is declared for the target (its slot aliases the
    /// source's at the site). Lookup-only (L2).
    in_place: SecondaryMap<ObjectId, ()>,
    /// Literal sites: total Pair in-edges per array object, and how many the
    /// walk has passed (the construction emits at the last one).
    lit_total: SecondaryMap<ObjectId, usize>,
    lit_seen: SecondaryMap<ObjectId, usize>,
    /// The fn's bounds-proof plan (the S20 deduced query, computed once at
    /// construction alongside `last_use` — the BL7 pattern): an
    /// `Operation::Index` it proves statically in-bounds can never fire the
    /// §3 bounds guard, so [`DevEmit::emit_index`] drops the dead guard text
    /// and emits the plain per-thread read (the extension temp stays).
    proof: BoundsProof,
    /// The minimal-emission classification (plan-minimal-emission WP-B):
    /// Named objects keep hoisted `o{ord}` locals + statement assignment;
    /// Inline objects live as memoized expression strings (`exprs`);
    /// Dissolved products never materialize — consumers read their field
    /// sources through `component_expr`.
    plan: EmissionPlan,
    /// Backend-forced Named overrides on top of the plan: a `Call` target
    /// must stay a statement so the §3 post-call `if (*trap)` check keeps
    /// its position (the query is backend-agnostic and cannot know the trap
    /// protocol; recorded in the plan doc as a WP-B as-built note).
    force_named: SecondaryMap<ObjectId, ()>,
    /// Memoized expressions for Inline/Dissolved-consumed values — one
    /// string per object, referenced exactly once (R-NODUP holds because
    /// `Inline ⇔ effective count = 1` in the query).
    exprs: SecondaryMap<ObjectId, String>,
    decls: String,
    body: String,
    next: u32,
    /// `return;` / `return {ret}{};` — the guard early-return for this fn.
    ret_default: String,
    /// Base indent added to every `line` level: 0 normally, 1 inside a
    /// loop's while body (the WP4 twin quartet).
    base: usize,
}
