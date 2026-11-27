//! `FnEmit` (DESIGN §1 op table, **host column**): one host C++ function per
//! `FuncDef` — hoisted value locals, the `topo_order` walk, and the host op
//! table. Scalars and products materialize as plain C++ locals (`o{ord}` by
//! object ordinal, temps `t{n}` by rising counter — the llvm `f{i}o{j}`
//! scheme re-spelled); every morphism's result is materialized into its
//! object's local (SSA-ish, the llvm slot discipline). A slot name is just a
//! host variable, so device-handle locals (`T*`) ride the same map
//! unchanged.
//!
//! WP3 (kernel.rs's host half, live here): array construction sites emit the
//! §2 literal upload (host data array + one H→D memcpy); array-bulk op sites
//! emit the buffer allocation + kernel launch + `trap_check_after_launch()`
//! where the site can trap (§3 + #14's trim — provably trap-free launches
//! pass no flag and skip the readback), with `Index`/`Fold` scalar results
//! read back D→H. **The buffer allocation is #18's arena form (v1.0):** one
//! FnScope zone per fn ([`crate::arena`]) covers every non-loop-cone site —
//! one arena `cudaMalloc` at fn entry, per-site `arena0 + OFF` pointer inits
//! at the old malloc points, one zone release at fn exit under the
//! range-test escape veto — while loop-cone sites keep the per-buffer
//! `cudaMalloc` + allocation-registry entry (the single choke point is
//! [`FnEmit::alloc_buffer`]).
//! [`FnQual::HostDevice`] fns emit a single `__host__ __device__` definition
//! (BC8 iv): the signature threads the trap pointer (when the fn can trap,
//! #14) and the `Div`/`Mod` zero guard is `#ifdef __CUDA_ARCH__`-split —
//! both guards fold for literal non-zero constant divisors (#13).
//!
//! WP4 (loops): the walk skips driver-owned morphisms — `loop_plan`
//! decide∪advance membership ∪ SCC incidence (DESIGN §1's walk-skip
//! paragraph, the llvm rule `func.rs:252–290` carried verbatim) — and each
//! `LoopEnter` delegates to [`crate::loops::emit_loop`], the guard-first
//! host quartet. Cone emissions reuse this op table unchanged: launches
//! inside a loop iterate with the loop, and a carried array's merge is a
//! host handle whose back edge is a pointer swap (§2). Two consequences
//! live here: every device-pointer local is `nullptr`-initialized (a
//! cone-side allocation site may not execute on a zero-iteration loop, so
//! the free path must never read an indeterminate handle), and an
//! array-returning fn's escape guard compares pointer VALUES — per buffer
//! `if (buf != ret)` for registered (cone-site) buffers, and the zone
//! release's `escaped0` range test for arena members — the loop-exit escape
//! reaches the return through the exit copy, not an `Output`, so the
//! registry's name-based removal cannot see it (the buffer equal to the
//! returned handle escapes; the duty transfers to the caller, §2's
//! invariants preserved).
//!
//! #19a ([`FnEmit::perf`], suggestions.md #19 step a): with `perf_timing`
//! set, every launch is wrapped in CUDA events (`fev{i}_start/stop`, one
//! fn-scope pair per launch site, created once) — `Record(start)` before
//! the launch, `Record(stop)`+`Synchronize`+`ElapsedTime` after (the stop
//! BEFORE the trap check) — printing `MAPAL_PERF launch=<kernel> ms=` per
//! execution, plus `MAPAL_PERF total ms=` at fn end. Default off: the text
//! is byte-identical to the pre-options emitter.

use mapal_ir::{
    CategoryIr, FuncId, LastUsePlan, LoopPlan, MorphismId, ObjectId, ObjectKind, Operation, Ty,
    Value,
};
use slotmap::SecondaryMap;
use std::collections::HashMap;

use crate::EmitError;
use crate::arena::{self, ArenaKey};
use crate::kernel::{self, FnQual, Qualifiers};
use crate::module::{StrGlobal, print_dispatch};
use crate::ty::{erased_index, lower_ty, residual_arity};

/// Per-function emission state (DESIGN Dat `FnCtx`). `slots` is partial —
/// erased (Unit/IoToken/Str) objects and constants have no slot (constants
/// fold into use sites as literals).
pub(crate) struct FnEmit<'a> {
    pub ir: &'a CategoryIr,
    pub f: FuncId,
    pub fnames: &'a SecondaryMap<FuncId, String>,
    pub strings: &'a SecondaryMap<ObjectId, StrGlobal>,
    /// The module's BC8 qualifier analysis (kernel.rs) and this fn's own
    /// case — `HostDevice` fns emit one `__host__ __device__` definition
    /// (trap-threading signature, `#ifdef __CUDA_ARCH__` Div/Mod guards).
    pub quals: &'a Qualifiers,
    pub qual: FnQual,
    /// The module's trap-capability pre-pass (kernel.rs, #14): trap-free
    /// kernels drop the trap parameter/launch arg/readback, and calls to
    /// trap-free `HostDevice` fns pass no `d_trap`.
    pub caps: &'a kernel::TrapCaps,
    /// Launch-form bulk-op sites: morphism → the launch's kernel name. The
    /// name is the site's dedup SURVIVOR (`kernel::emit_kernel_set`, #17):
    /// a site whose structural shape was already emitted launches the first
    /// occurrence's definition, not a `k{f_ord}_{s_ord}` of its own.
    pub sites: HashMap<MorphismId, String>,
    /// Array-literal sites: total Pair in-edges per array object and how many
    /// the walk has passed (construction emits at the last one — see
    /// kernel::literal_pair_counts).
    pub lit_total: SecondaryMap<ObjectId, usize>,
    pub lit_seen: SecondaryMap<ObjectId, usize>,
    pub slots: SecondaryMap<ObjectId, String>,
    /// plan-s39 guard sites needing gating, keyed by their `Phi`.
    pub gsites: SecondaryMap<MorphismId, mapal_ir::GuardSite>,
    /// Guard-arm-owned morphisms — emitted inside their Phi's `if`/`else`,
    /// skipped by the straight-line walk.
    pub gated: SecondaryMap<MorphismId, ()>,
    /// Hoisted local declarations (fn top), one per materialized object —
    /// plus the fn-scope readback temps the WP3 `Index`/`Fold` sites hoist
    /// here during the walk (a cone-side temp must be fn-scope so the free
    /// at fn exit can name it).
    pub decls: String,
    pub body: String,
    pub next: u32,
    /// Base indent of `line` in 2-space levels: 1 at fn top level, raised to
    /// 2 by the loop driver (loops.rs) around the decide/advance cones.
    pub indent: usize,
    /// The allocation registry (DESIGN §2, allocation-based ownership): every
    /// device buffer THIS fn `cudaMalloc`'d, in allocation order — freed at
    /// fn exit by [`FnEmit::emit_frees`]. Parameters are borrowed (never
    /// registered, never freed here); `Output` removes an escaping buffer
    /// (the free duty transfers to the caller). **Arena members are not
    /// registered** (they are freed with their zone, not per buffer) — only
    /// loop-cone sites' per-buffer mallocs push here in v1.0.
    pub allocs: Vec<String>,
    /// The fn's arena plan (suggestions.md #18, plan-smart-arenas v1.0):
    /// one FnScope zone covering the fn's non-loop-cone buffer sites.
    /// Computed at [`FnEmit::emit`] start (deduced, never stored — the
    /// capacity guard is rule 4's compile-time `Unsupported`); `None` when
    /// the fn has no zone members.
    pub arena: Option<arena::ArenaPlan>,
    /// The fn's last-use plan (plan-last-use §2, the BL7 deduced query —
    /// computed once at construction, alongside the arena plan): the single
    /// source of dead/escape/carried facts for the in-place `Update`
    /// ([`in_place_update`], rule 4) and the back-edge freeing
    /// (`loops.rs`, suggestion #2). Representation-only: values, evaluation
    /// order, and trap behavior are unchanged.
    pub last_use: LastUsePlan,
    /// Kernel-time instrumentation (suggestions.md #19a): when set, every
    /// launch is wrapped in CUDA events (`MAPAL_PERF launch=` per execution,
    /// `MAPAL_PERF total ms=` at fn end). Default off — the text is then
    /// byte-identical to the pre-options emitter. Set post-construction by
    /// the [`crate::emit_with_opts`] path (keeps `new` at its W1 arity).
    pub perf: bool,
    /// Launch-site morphism → the site's event-pair ordinal (`fev{i}_start`
    /// / `fev{i}_stop`), in `collect_sites` order (deterministic). Populated
    /// at [`FnEmit::emit`] start when `perf` is set; lookup-only (L2).
    pub ev_ord: HashMap<MorphismId, usize>,
    /// Minimal-emission classification (plan-minimal-emission WP-C — the
    /// host/`__host__ __device__` lane of the same mechanism as `DevEmit`).
    pub plan: mapal_ir::EmissionPlan,
    /// Backend-forced Named on top of the plan: `Call` targets (host callees
    /// trap via `mapal_trap` inside — call position is semantic), every
    /// bulk-op target (launch/readback machinery needs an lvalue local),
    /// and product-typed Inline (local-name fallback).
    pub force_named: SecondaryMap<ObjectId, ()>,
    /// Memoized expressions for Inline/Dissolved values (see `DevEmit`).
    pub exprs: SecondaryMap<ObjectId, String>,
}

impl<'a> FnEmit<'a> {
    pub fn new(
        ir: &'a CategoryIr,
        f: FuncId,
        fnames: &'a SecondaryMap<FuncId, String>,
        strings: &'a SecondaryMap<ObjectId, StrGlobal>,
        quals: &'a Qualifiers,
        caps: &'a kernel::TrapCaps,
        kernels: &'a kernel::KernelSet,
    ) -> Self {
        let sites = kernels.names.get(f).cloned().unwrap_or_default();
        let plan = ir.emission_plan(f);
        let mut force_named: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let fd = ir.func(f).expect("func resolves");
        for &m in &fd.morphisms {
            let morph = ir.morphism(m).expect("morphism resolves");
            match morph.op {
                Operation::Call(_)
                | Operation::Map { .. }
                | Operation::Zip
                | Operation::Enumerate
                | Operation::Iota
                | Operation::Fill
                | Operation::Fold { .. }
                | Operation::Index
                | Operation::Update => {
                    force_named.insert(morph.target, ());
                }
                _ => {}
            }
        }
        for (id, obj) in ir.objects() {
            if ir.try_owner(id) == Some(f)
                && obj.ty.product_arity().is_some()
                && plan.class(id).is_some_and(|c| c.is_inline())
            {
                force_named.insert(id, ());
            }
        }
        let mut gsites: SecondaryMap<MorphismId, mapal_ir::GuardSite> = SecondaryMap::new();
        let mut gated: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        // plan-s40: this emitter keeps STRICT semantics for any site touching
        // loop machinery or in-SCC work — it is being replaced by NVPTX
        // (Sapir, S39 close), so gated loops are not wired here. Surface Mapal
        // cannot build these shapes (L1406), so no compiled program is
        // affected; the shapes exist only in testgen IR, and the CUDA
        // differential does not run. Emission stays valid either way.
        let cuda_in_scc: SecondaryMap<mapal_ir::ObjectId, ()> = {
            let mut s = SecondaryMap::new();
            for scc in ir.loop_structure(f) {
                for o in scc.objects {
                    s.insert(o, ());
                }
            }
            s
        };
        for site in ir
            .guard_plan(f)
            .into_iter()
            .filter(mapal_ir::GuardSite::gated)
        {
            let touches_loop = site
                .on_true
                .own
                .iter()
                .chain(site.on_false.own.iter())
                .any(|&m| {
                    let morph = ir.morphism(m).expect("own edge");
                    matches!(morph.op, Operation::LoopEnter)
                        || cuda_in_scc.contains_key(morph.source)
                        || cuda_in_scc.contains_key(morph.target)
                });
            if touches_loop {
                continue;
            }
            for &m in site.on_true.own.iter().chain(site.on_false.own.iter()) {
                gated.insert(m, ());
            }
            // The result is assigned inside a branch, so it must be a real
            // local rather than a dissolved expression substituted at its use.
            force_named.insert(ir.morphism(site.phi).expect("phi").target, ());
            gsites.insert(site.phi, site);
        }
        FnEmit {
            ir,
            f,
            fnames,
            strings,
            quals,
            qual: quals.get(f),
            caps,
            sites,
            lit_total: kernel::literal_pair_counts(ir, f),
            lit_seen: SecondaryMap::new(),
            slots: SecondaryMap::new(),
            gsites,
            gated,
            decls: String::new(),
            body: String::new(),
            next: 0,
            indent: 1,
            allocs: Vec::new(),
            arena: None,
            last_use: ir.last_use_plan(f),
            perf: false,
            ev_ord: HashMap::new(),
            plan,
            force_named,
            exprs: SecondaryMap::new(),
        }
    }

    /// The effective class (WP-C): deduced plan, overridden Named where the
    /// host statement protocol demands it.
    fn cls(&self, o: ObjectId) -> mapal_ir::EmissionClass {
        if self.force_named.contains_key(o) {
            return mapal_ir::EmissionClass::Named;
        }
        self.plan.class(o).unwrap_or(mapal_ir::EmissionClass::Named)
    }

    fn dissolved(&self, o: ObjectId) -> bool {
        self.cls(o).is_dissolved()
    }

    fn expr_only(&self, o: ObjectId) -> bool {
        !self.cls(o).is_named()
    }

    fn fresh(&mut self) -> u32 {
        let n = self.next;
        self.next += 1;
        n
    }

    /// A fresh temporary name (`t{n}`).
    fn tmp(&mut self) -> String {
        format!("t{}", self.fresh())
    }

    /// Append one body line at the current indent level.
    pub(crate) fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.body.push_str("  ");
        }
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }

    /// Append one body line one level deeper (inside an `if`/`else` block).
    fn line4(&mut self, s: impl AsRef<str>) {
        for _ in 0..=self.indent {
            self.body.push_str("  ");
        }
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }

    fn slot(&self, o: ObjectId) -> Option<String> {
        self.slots.get(o).cloned()
    }

    fn obj_ty(&self, o: ObjectId) -> Ty {
        self.ir.object(o).expect("object resolves").ty.clone()
    }

    // --- operand materialization -----------------------------------------

    /// The whole value of object `o` as a C++ expression: a literal for a
    /// (scalar) constant, the local's name otherwise. `None` if `o` is
    /// erased / a `Str` / a non-scalar constant (arrays are WP3).
    fn load_whole(&mut self, o: ObjectId) -> Option<(String, String)> {
        let obj = self.ir.object(o).expect("object resolves");
        if obj.kind == ObjectKind::Constant {
            return match &obj.value {
                Some(Value::Str(_)) | None => None,
                Some(v) => Some((lower_ty(&obj.ty)?, const_literal(v))),
            };
        }
        let ct = lower_ty(&obj.ty)?;
        if let Some(e) = self.exprs.get(o) {
            return Some((ct, e.clone()));
        }
        let slot = self.slot(o)?;
        Some((ct, slot))
    }

    /// Component `k` of aggregate `agg` as a C++ (lvalue) expression, under
    /// the residual-erasure remap: residual-1 products are **bare** (the
    /// local IS the component); residual ≥ 2 products use the
    /// `.f{erased_index}` field. `None` if the component is erased or `agg`
    /// has no slot. Arrays have no component expression yet (WP3).
    fn component_expr(&mut self, agg: ObjectId, k: u32) -> Option<(String, String)> {
        let agg_ty = self.obj_ty(agg);
        match &agg_ty {
            Ty::Tuple(_) | Ty::Struct { .. } => {
                let comp_ty = agg_ty.component_ty(k)?.clone();
                let cct = lower_ty(&comp_ty)?; // erased component ⇒ None
                // WP-C: a dissolved product resolves through its Pair edge
                // to the field source's own expression.
                if self.dissolved(agg) {
                    let src = kernel::pair_source(self.ir, agg, k)?;
                    let (_, val) = self.load_whole(src)?;
                    return Some((cct, val));
                }
                let agg_slot = self.slot(agg)?;
                if residual_arity(&agg_ty) == 1 {
                    Some((cct, agg_slot)) // bare: the local IS the component
                } else {
                    let eidx = erased_index(&agg_ty, k)?;
                    Some((cct, format!("{agg_slot}.f{eidx}")))
                }
            }
            _ => None,
        }
    }

    /// Route a produced value (WP-C): Named takes its one statement
    /// assignment; Inline memoizes the (parenthesized) expression.
    fn store_obj(&mut self, o: ObjectId, expr: &str) {
        if self.expr_only(o) {
            let wrapped = if kernel::is_atomic_expr(expr) {
                expr.to_string()
            } else {
                format!("({expr})")
            };
            self.exprs.insert(o, wrapped);
            return;
        }
        if let Some(slot) = self.slot(o) {
            self.line(format!("{slot} = {expr};"));
        }
    }
}

mod copies;
mod free;
mod sites;
#[cfg(test)]
mod tests;
mod walk;

pub(crate) use self::free::*;
