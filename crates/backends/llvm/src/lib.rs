//! LLVM backend: emits textual LLVM IR (`.ll`) piped to `clang` (ADR-0020 §1;
//! backend-llvm DESIGN). The entry point [`emit`] realizes the functor
//! `F_LLVM : Mapal-Cat → LLVM-Cat` as a String — the translation unit **is** the
//! artifact (ADR-0020). Semantics are the interpreter oracle's by construction:
//! every `Print`/trap routes through the shared `mapal-rt` runtime (ADR-0020 §2),
//! integer arithmetic wraps, `Div`/`Mod`/`Index`/`Update` are guarded (an S20
//! `bounds_proof`-proven `Index` elides its statically-dead guard), and floats
//! are IEEE at width.
//!
//! Emission is deterministic **per resolved profile** (L2): all names come from
//! per-function ordinals and a rising counter, never slotmap bits, so one `(IR,
//! EmitOpts, TargetProfile)` triple always yields the same text. The default
//! target is `native` (S46, ratified ADR-0038): it reads host facts, so default
//! emitted text is machine-dependent *by design* — reproducible emission names a
//! profile explicitly, and [`emit_detailed`] reports which profile was resolved.
//! The one capability gap (L3) is the non-canonical loop shape (multi-merge
//! SCC), rejected as [`EmitError::Unsupported`] — the same scope boundary as
//! interp M1 / rewrite.

mod func;
mod loops;
mod module;
mod profile;
mod reuse;
mod ty;

pub use crate::profile::TargetProfile;

use mapal_ir::{CategoryIr, FuncId, MorphismId, Operation, PathPlan, SourceLoc, TaskKind};
use slotmap::SecondaryMap;

use crate::func::{FnAttrs, FnEmit, packing_site};
use crate::module::{
    HEAP_DECLS, PAR_DECLS, PERF_DECLS, PREFETCH_DECL, PanelWrite, RT_DECLS, SME_DECLS,
    collect_str_globals, emit_main_wrapper, emit_str_globals, sme_panel,
};

/// A structured, renderer-free emission error (ADR-0020 §1; C3).
#[derive(Clone, Debug, PartialEq)]
pub enum EmitError {
    /// A capability-matrix rejection (the ✋ cell): a feature outside the
    /// realized set — here, the non-canonical (multi-merge) loop shape (L3).
    Unsupported { feature: String, loc: SourceLoc },
    /// An internal invariant violation (should not occur for sealed IR).
    Internal(String),
}

/// How the move-panel traversal is decided (S45). See [`EmitOpts::move_panel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MovePanel {
    /// **The default and the shipped path.** Fire where a recognized
    /// [`mapal_ir::MoveSite`] and the target profile's L1D geometry say it pays,
    /// with a derived block. A profile without L1D facts (`generic`) can never
    /// fire it, so the default target's emission is unchanged.
    #[default]
    Deduce,
    /// Never fire. The OFF arm of a measurement, and an escape hatch.
    Off,
    /// Force this `(w, b)` on every eligible map, ignoring both records — S44's
    /// behaviour exactly, kept **only** so a block sweep is still possible.
    Force(u64, u64),
}

/// The GPU tile kernel's rungs (S52), as a set: `true` = the rung is in it.
///
/// **What ships is the profile's measured table** (S53,
/// [`TargetProfile::gpu_rungs`]): a rung is ON on a part only where its arm
/// measured positive there. [`GpuArms`] forces a rung on or off so its effect
/// can be measured — an A/B needs both arms of the same compiler, the
/// `MovePanel::Off` precedent — and a rung that shows no effect on any part is
/// deleted, not kept as a knob. Values are identical either way: every rung
/// moves operands, never the order a cell folds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuRungs {
    /// Take the profile's measured rung table; off = every rung on (S52).
    pub arch: bool,
    /// Space a thread's register-tile chunks by the bank geometry.
    pub interleave: bool,
    /// Fully unroll the panel's `bk` register steps.
    pub unroll: bool,
    /// Emit `.minnctapersm` from the derived blocks per SM.
    pub min_blocks: bool,
    /// Shape each warp's threads by the shared-memory wavefront.
    pub warp: bool,
    /// Load the next panel into registers while this one is computed.
    pub prefetch: bool,
    /// On a reduced face, take the tensor-core realization (`mma.sync`).
    pub mma: bool,
    /// Stream FMA-tile panels through shared memory with `cp.async` (sm_80+).
    pub cp_async: bool,
    /// Tensor-core warp tile from the whole-thread register count (64×64).
    pub wide: bool,
    /// Load tensor-core A fragments with `ldmatrix.x4`.
    pub ldmatrix: bool,
    /// Shrink the register / warp tile until the grid gives every warp
    /// scheduler two warps (`warp-fill`, S53); OFF = blocks ≥ SMs only.
    pub fill: bool,
    /// S53 `fma-regs` (four rungs, FMA and one-thread-per-cell kernels):
    /// index arithmetic in `i32` when every index the kernel forms fits.
    pub idx32: bool,
    /// Panel copy `p` addressed as copy 0 plus a constant from the lane
    /// map's period: one base register per operand.
    pub affine: bool,
    /// Register-staged transposed-A store: the lane map and row pad under
    /// which one warp's stores hit 32 banks.
    pub pad: bool,
    /// FMA register tile from the whole-thread register count (8×16).
    pub wide_fma: bool,
    /// On `sm_90a`, the TF32 face takes Hopper's `wgmma` kernel (S53); off,
    /// the `mma.sync` kernel. Both are bounded by TF32, neither is bit-exact.
    pub wgmma: bool,
    /// Keep an object's device copy between sites (S53, plan-s53 §2.3): copy
    /// in at the producer, carry loop state on the device, copy out only what
    /// the host reads, free at death. Off: one `mapal_gpu_tile` round trip
    /// per site (S52).
    pub resident: bool,
    /// Under `resident`, compute a closed-form producer that only device sites
    /// read on the device (`mapal_gen_<id>`) instead of copying it in.
    pub devgen: bool,
    /// S53 retention at part-of-object scope: a row-local site over a carried
    /// array whose iteration writes one row ([`mapal_ir::Delta::Rows`])
    /// recomputes only that row into its retained output (`func/retain.rs`).
    /// Ships OFF on every measured part: a function it fires in keeps S52 round
    /// trips, and `resident` alone ran decode 1.3–1.8× faster
    /// (`integrated-*.txt`). Kept for planning the two together.
    pub retain_delta: bool,
    /// Host runtime (S53): start the CUDA context and module load on a thread
    /// as `main`'s first call, so it overlaps the host work before the first
    /// site. Off: the first site creates them.
    pub eager_init: bool,
    /// S54 R7: device sites with no `PathPlan` path between them launch on
    /// separate streams (one per chain), an event on every cross-chain edge,
    /// when every such site's output stays on the device. Off: one stream.
    pub streams: bool,
    /// Open the streams only where two independent grids fit one wave
    /// together (the wave rule, `func/resident.rs:waves_merge`); off =
    /// whenever two chains exist. A measurement arm for the rule itself.
    pub stream_gate: bool,
    /// Page-lock (`cuMemHostRegister`) a host array that crosses the bus
    /// repeatedly (a copy inside a loop), once, before its first copy. Off:
    /// pageable copies.
    pub pinned: bool,
    /// sm_90+: a site launched right behind another kernel on its stream is a
    /// programmatic dependent launch (`griddepcontrol.wait` at kernel entry).
    pub pdl: bool,
}

impl GpuRungs {
    /// Every rung on — what S52 shipped on every part.
    pub const ALL: Self = Self {
        arch: true,
        interleave: true,
        unroll: true,
        min_blocks: true,
        warp: true,
        prefetch: true,
        mma: true,
        cp_async: true,
        wide: true,
        ldmatrix: true,
        fill: true,
        idx32: true,
        affine: true,
        pad: true,
        wide_fma: true,
        wgmma: true,
        resident: true,
        devgen: true,
        retain_delta: true,
        eager_init: true,
        streams: true,
        stream_gate: true,
        pinned: true,
        pdl: true,
    };

    /// The rung names `emit --gpu-off=` / `--gpu-on=` and the profile's rung
    /// table use.
    pub const NAMES: [&'static str; 24] = [
        "arch-rungs",
        "interleave",
        "unroll",
        "min-blocks",
        "warp",
        "prefetch",
        "mma",
        "async",
        "wide",
        "ldmatrix",
        "warp-fill",
        "idx32",
        "affine",
        "pad",
        "wide-fma",
        "wgmma-hopper",
        "resident",
        "devgen",
        "retain-delta",
        "eager-init",
        "streams",
        "stream-gate",
        "pinned",
        "pdl",
    ];

    /// The rung called `name`, `None` for an unknown name.
    pub fn rung(&mut self, name: &str) -> Option<&mut bool> {
        Some(match name {
            "arch-rungs" => &mut self.arch,
            "interleave" => &mut self.interleave,
            "unroll" => &mut self.unroll,
            "min-blocks" => &mut self.min_blocks,
            "warp" => &mut self.warp,
            "prefetch" => &mut self.prefetch,
            "mma" => &mut self.mma,
            "async" => &mut self.cp_async,
            "wide" => &mut self.wide,
            "ldmatrix" => &mut self.ldmatrix,
            "warp-fill" => &mut self.fill,
            "idx32" => &mut self.idx32,
            "affine" => &mut self.affine,
            "pad" => &mut self.pad,
            "wide-fma" => &mut self.wide_fma,
            "wgmma-hopper" => &mut self.wgmma,
            "resident" => &mut self.resident,
            "devgen" => &mut self.devgen,
            "retain-delta" => &mut self.retain_delta,
            "eager-init" => &mut self.eager_init,
            "streams" => &mut self.streams,
            "stream-gate" => &mut self.stream_gate,
            "pinned" => &mut self.pinned,
            "pdl" => &mut self.pdl,
            _ => return None,
        })
    }

    pub fn get(mut self, name: &str) -> bool {
        self.rung(name).is_some_and(|r| *r)
    }

    /// `(self ∪ on) ∖ off`.
    pub fn with(mut self, arms: GpuArms) -> Self {
        for n in Self::NAMES {
            let r = self.rung(n).expect("a listed name");
            *r = (*r || arms.on.get(n)) && !arms.off.get(n);
        }
        self
    }
}

/// Measurement arms over the profile's rung table (S53): a rung in `on` is
/// forced on, one in `off` forced off. Both empty by default — the shipped path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuArms {
    pub on: GpuRungs,
    pub off: GpuRungs,
}

/// A reduced-precision face — ADR-0032 D1's lattice below `contract`. Unlike
/// every rung, it **changes values**, by the declared bound of its format, and
/// is what admits the tensor-core realizations: they cannot reproduce an f32
/// fold bit for bit. Interim form of the type-level contract (the `--contract`
/// precedent); the oracle check for it is a bound, not bit-equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduced {
    /// f32 storage, operands rounded to TF32 (10-bit mantissa, `cvt.rna`),
    /// f32 accumulation — `mma.sync.m16n8k8.tf32` on sm_80+.
    Tf32,
}

/// Emission options. [`emit`] delegates to these product defaults.
#[derive(Clone, Copy, Debug)]
pub struct EmitOpts {
    /// Bracket `mapal_main` with the mapal-rt compute timer.
    pub perf_timing: bool,
    /// Tile recognized matmul-shaped map sites.
    pub tiling: bool,
    /// Pack tiled two-dimensional right-hand operands.
    pub packing: bool,
    /// Use single-rounding FMA contraction on the product face; the default
    /// conformance face stays bit-exact.
    pub contract: bool,
    /// Split deep packed tile sites into k-panels — the OpenBLAS (jc, kc, ic)
    /// nest with an A-panel pack and partial sums parked in `out`.
    ///
    /// **Default OFF: measured a 3× LOSS** on M4 Pro at 1024 f32 (S29 —
    /// `fma` 59.8 ms on / 19.8 ms off; the parking traffic outweighs the A
    /// re-read it removes at this size and cache hierarchy).
    ///
    /// **S42: SWEPT on the box, and it loses at every depth — the last argument
    /// for this lever is gone.** It was kept because it "was designed against
    /// BOX-scale traffic (16 GB of A re-reads at 4096 on zen3) where it has not
    /// yet been measured".
    ///
    /// Measured on an i9-14900F (24C/32T, 48 KB L1d, 2 MB per-core L2, 36 MB L3,
    /// AVX2, governor `performance`), N=4096, values identical to the Mac's at
    /// every depth. For `zen3` at f32 the chain is `lanes 8 → tile_j 32 → nc
    /// 1024`, so `tile_kc = l2_bytes / 8192`; the sweep varies `l2_bytes`:
    ///
    /// | depth | blocks | 1 thread | threaded (32) |
    /// | ---: | ---: | ---: | ---: |
    /// | unblocked | 1 | **926.759 ms** | **112.919 ms** |
    /// | 4096 | 1 | 0.978× (gate closed ⇒ *is* unblocked) | 1.000× |
    /// | 2048 | 2 | 0.548× | 0.595× |
    /// | 1024 | 4 | 0.557× | 0.759× |
    /// | 512 | 8 | 0.551× | **0.771×** best |
    /// | 256 | 16 | 0.560× | 0.767× |
    /// | 64 | 64 | 0.454× | 0.668× |
    ///
    /// **A step function, not a curve**: any blocking at all costs ~1.8× at one
    /// thread and ~1.3× at best threaded, at *every* depth from 2 to 64 blocks.
    /// Unlike the SME k-panel — where one wrong depth made the technique look
    /// worthless (`func/sme.rs`) — here there is no optimum to find.
    ///
    /// ⇒ **`kc_nest` has now lost on every machine available, swept rather than
    /// spot-checked.** "Unmeasured on the box" is no longer a reason to keep it.
    /// Either a machine justifies it in writing, or delete the lever.
    ///
    /// A pure performance tailor in ADR-0032's sense: bit-exact either way, which
    /// the differential suite enforces — and which the cross-machine run
    /// independently confirmed, ARM and x86 printing identical values.
    pub kc_nest: bool,
    /// **The move-panel traversal — S44's rung, S45's deduction.** Visits a bulk
    /// `map`'s iteration space in `b × b` tiles of a `w`-wide 2-D geometry
    /// instead of in linear order, when that pays.
    ///
    /// **S44 shipped this as `--move-panel=W:B`: two numbers typed by a human.**
    /// `W` was program geometry the emitter could not see (`tile_site` requires a
    /// fold in the map body, and a transpose has none) and `B` was a swept
    /// machine constant. S45 deleted both: `W` comes from
    /// [`mapal_ir::MoveSite`], a fold-less move-site record; the cache geometry
    /// comes from [`crate::profile::L1d`], detected; and firing and `B` are
    /// arithmetic over the two in [`crate::profile::TargetProfile::move_block`].
    ///
    /// [`MovePanel::Deduce`] is the default and the shipped path. The other two
    /// exist so the transform can still be **measured** — an A/B needs an OFF arm
    /// of the same binary, and a B-sweep needs to force a block — and they carry
    /// no machine constant into any emitted default.
    ///
    /// A pure ADR-0032 D4 performance tailor: the transform is a **permutation of
    /// the loop counter**, so every element is still visited exactly once and
    /// values are bit-identical, which the differential suite enforces.
    pub move_panel: MovePanel,
    /// The machine facts the emitter tiles against, selected **by name**
    /// (plan-s31-target-profiles): `generic` (the pre-S46 literals, byte-
    /// identical on every host), `apple-m`, `apple-m4-sme`, `zen3`,
    /// `raptorlake`, `cuda-ada`.
    ///
    /// **The default is `native` (S46, ratified ADR-0038): probe the host, and
    /// fall back to `generic` when a fact cannot be read.** Default emission is
    /// therefore machine-dependent *by design* — anything that needs
    /// reproducible emitted text across machines (structural tests, artifact
    /// hashes) pins a profile by name, and [`emit_detailed`] reports which
    /// profile was resolved.
    ///
    /// A pure ADR-0032 D4 performance tailor: every profile field is
    /// value-invariant, which the differential suite enforces by running under
    /// a non-default profile.
    pub target: &'static str,
    /// Measurement arms over the GPU rung table; see [`GpuArms`].
    pub gpu: GpuArms,
    /// A reduced-precision face, when the program admits one; see [`Reduced`].
    pub reduced: Option<Reduced>,
}

impl Default for EmitOpts {
    fn default() -> Self {
        Self {
            perf_timing: false,
            tiling: true,
            packing: true,
            contract: false,
            kc_nest: false,
            move_panel: MovePanel::Deduce,
            // S46 (Sapir): the DEFAULT detects the machine. Every rung that
            // needs machine facts asks the profile for them, so a default build
            // now gets the cache-aware ones instead of silently declining.
            // `native` falls back to `generic` when detection fails, and tests
            // that must be reproducible across machines pin `generic` by name.
            target: "native",
            gpu: GpuArms::default(),
            reduced: None,
        }
    }
}

/// One emitted translation unit **plus the profile it was emitted against**
/// (ADR-0038 D2). The default target is `native`, so the text alone does not
/// say which machine facts shaped it — `profile` is the resolved profile's
/// name (`"native"` when detection succeeded, `"generic"` on fallback or by
/// name, or the named profile), and artifact identity is
/// `(source, EmitOpts, profile)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emission {
    /// The `.ll` text — the translation unit itself (ADR-0020).
    pub text: String,
    /// The **device** translation unit (`nvptx64`), when the profile is a GPU
    /// and some tile site offloads: `llc` turns it into the PTX the runtime
    /// loads (`MAPAL_PTX`). `None` on every CPU profile.
    pub device: Option<String>,
    /// The resolved [`TargetProfile`]'s name.
    pub profile: &'static str,
}

/// Emit one LLVM translation unit for `ir` (ADR-0020 §1). `Ok` is the `.ll`
/// text; `Err(Unsupported)` for a non-canonical loop (L3). Uses
/// [`EmitOpts::default()`] — target `native` (S46); [`emit_detailed`] reports
/// which profile that resolved to on this host.
pub fn emit(ir: &CategoryIr) -> Result<String, EmitError> {
    emit_with_opts(ir, &EmitOpts::default())
}

/// [`emit`] with options (see [`EmitOpts`]); the text half of [`emit_detailed`].
pub fn emit_with_opts(ir: &CategoryIr, opts: &EmitOpts) -> Result<String, EmitError> {
    Ok(emit_detailed(ir, opts)?.text)
}

/// [`emit_with_opts`] with the resolved profile named in the artifact
/// (ADR-0038 D2). This is the honest signature of the transform: the profile
/// is an *input* — ambient when the target is `native`, explicit otherwise —
/// and naming it in the output is what makes the input visible.
pub fn emit_detailed(ir: &CategoryIr, opts: &EmitOpts) -> Result<Emission, EmitError> {
    // Machine facts, by name (plan-s31-target-profiles rule 3). An unknown name
    // is an error, never a silent fall back to `generic` — a typo that quietly
    // emits the default profile's numbers is the exact failure the table exists
    // to remove.
    let Some(profile) = profile::resolve(opts.target) else {
        return Err(EmitError::Internal(format!(
            "unknown target profile `{}`; known: {}",
            opts.target,
            profile::names()
        )));
    };

    // Capability gate (L3): canonical loops only — every SCC has exactly one
    // merge and a well-formed `loop_plan`. Same predicate as rewrite's
    // `is_canonical` / interp M1 (BL6/BL7).
    for (f, _) in ir.funcs() {
        for lscc in ir.loop_structure(f) {
            let canonical = lscc.merges.len() == 1 && ir.loop_plan(f, lscc.merges[0]).is_some();
            if !canonical {
                let loc = lscc
                    .merges
                    .first()
                    .and_then(|&m| ir.object(m))
                    .map(|o| o.loc)
                    .unwrap_or(SourceLoc { start: 0, end: 0 });
                return Err(EmitError::Unsupported {
                    feature: "nested loops".into(),
                    loc,
                });
            }
        }
    }

    let strings = collect_str_globals(ir);

    // Suggestions #7's attribute-capability pre-pass (func/mod.rs `FnAttrs`):
    // clean fns get `readonly nounwind` (+ `willreturn`), unclean fns nothing.
    let attrs = FnAttrs::analyze(ir);

    // Deterministic function names: entry → @mapal_main, others → @fn{ordinal}.
    let mut fnames: SecondaryMap<mapal_ir::FuncId, String> = SecondaryMap::new();
    let entry = ir.entry();
    for (ord, (id, _)) in ir.funcs().enumerate() {
        let name = if id == entry {
            "mapal_main".to_string()
        } else {
            format!("fn{ord}")
        };
        fnames.insert(id, name);
    }

    // ponytail: malformed/cyclic plans stay sequential; delete this fallback
    // when path_plan's total-DAG contract is enforced before backend entry.
    let path_plan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ir.path_plan(entry)))
        .ok()
        .filter(path_plan_is_acyclic);
    // On a GPU profile the device is the parallelism: the host runs the program
    // around the offloaded sites sequentially. ponytail: host threads alongside
    // the device is co-execution (ADR-0035) — add when a program needs both.
    let parallel = profile.gpu().is_none()
        && path_plan
            .as_ref()
            .is_some_and(|plan| !plan.is_single_path());
    let body_sites = if parallel {
        parallel_body_sites(ir, entry, path_plan.as_ref().expect("parallel plan"))
    } else {
        SecondaryMap::new()
    };
    let prefetch = opts.tiling
        && opts.packing
        && ir.funcs().any(|(f, _)| {
            ir.tile_plan(f)
                .sites
                .iter()
                .any(|(_, site)| packing_site(site, profile, opts.contract))
        });

    // Functions first: the arena declarations are gated on the emitted text
    // (below), so the bodies have to exist before the header is assembled.
    let mut funcs = String::new();
    // `devgen` (S53): the producers each function computes on the device, and
    // every function's emitted text — a device-built producer's body is the
    // host's own, appended to the device unit.
    let mut gens: Vec<(u64, mapal_ir::FuncId, String)> = Vec::new();
    let mut texts: SecondaryMap<mapal_ir::FuncId, String> = SecondaryMap::new();
    for (id, _) in ir.funcs() {
        if id == entry && parallel {
            funcs.push_str(&FnEmit::emit_parallel(
                ir,
                id,
                &fnames,
                &strings,
                &attrs,
                path_plan.as_ref().expect("parallel plan"),
                opts.perf_timing,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.kc_nest,
                opts.move_panel,
                profile,
            ));
        } else {
            let mut fe = FnEmit::new(
                ir,
                id,
                &fnames,
                &strings,
                &attrs,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.kc_nest,
                opts.move_panel,
                profile,
            );
            if id == entry {
                fe.set_perf_timing(opts.perf_timing);
                // S54 `streams`: the device sites' independence (the host
                // stays sequential on a GPU profile).
                fe.set_path_plan(path_plan.as_ref().filter(|_| profile.gpu().is_some()));
            }
            fe.set_gpu_arms(opts.gpu, opts.reduced);
            if let Some(&site) = body_sites.get(id) {
                fe.set_task_body_site(site);
            }
            gens.extend(fe.gen_kernels());
            let text = fe.emit();
            funcs.push_str(&text);
            texts.insert(id, text);
        }
        funcs.push('\n');
    }

    let mut out = String::new();
    out.push_str("; mapal-backend-llvm emitted module\n");
    out.push_str(RT_DECLS);
    if prefetch {
        out.push_str(PREFETCH_DECL);
    }
    if opts.perf_timing {
        out.push_str(PERF_DECLS);
    }
    if parallel {
        out.push_str(PAR_DECLS);
    }
    // ponytail: gate the arena declarations on the emitted call itself, not on
    // a re-derived predicate — the call IS the requirement, so the two cannot
    // drift the way a `prefetch`-style pre-pass could. (PAR/PERF must gate on a
    // predicate: they are decided before any body exists.)
    if funcs.contains("@mapal_rt_alloc") {
        out.push_str(HEAP_DECLS);
    }
    // Same rule for the SME leg: the emitted call IS the requirement, so the
    // three intrinsic declarations and the one streaming panel kernel appear
    // exactly when some tile site took the SME rung. A module without one is
    // byte-identical to what it was before the rung existed.
    // Two kernels, detected independently: `@mapal_sme_panel(` cannot match
    // `@mapal_sme_panel_acc(` because of the paren, so the KC variant appears
    // only where the KC nest actually called it.
    // A module talks to the GPU runtime when a site calls it: the S52 round
    // trip (`mapal_gpu_tile`) or `resident`'s split entries (`mapal_gpu_alloc`
    // precedes every one of them).
    let resident = funcs.contains("@mapal_gpu_alloc(");
    let round_trip = funcs.contains("@mapal_gpu_tile(");
    let gpu_host = resident || round_trip;
    if gpu_host {
        out.push_str(crate::func::gpu::GPU_INIT_DECL);
        if round_trip {
            out.push_str(crate::func::gpu::GPU_TILE_DECL);
        }
        if funcs.contains(crate::func::gpu::GPU_SYNC) {
            out.push_str(crate::func::gpu::GPU_SYNC_DECL);
        }
    } else {
        // No device work in the module: the clock fence's sync waits for nothing.
        funcs = funcs.replace(&format!("  {}\n", crate::func::gpu::GPU_SYNC), "");
    }
    if resident {
        out.push_str(crate::func::gpu::RESIDENT_DECLS);
        for (call, decl) in crate::func::gpu::S54_DECLS {
            if funcs.contains(call) {
                out.push_str(decl);
            }
        }
    }
    let sme_store = funcs.contains("@mapal_sme_panel(");
    let sme_acc = funcs.contains("@mapal_sme_panel_acc(");
    let sme = sme_store || sme_acc;
    if sme {
        out.push_str(SME_DECLS);
    }
    out.push('\n');

    let sg = emit_str_globals(&strings);
    if !sg.is_empty() {
        out.push_str(&sg);
        out.push('\n');
    }

    out.push_str(&funcs);
    if sme {
        // The panel side and the tile-block arrangement are the profile's,
        // never literals here: `svl_bytes / sizeof(f32)` and the square-most
        // split of the ZA tile count. The rung only fires at f32.
        let f32 = mapal_ir::Ty::f32();
        let expect = "the SME rung only fires on a profile with an SME unit";
        let t = profile.sme_tile_side(&f32).expect(expect);
        let (ti, tj) = profile.sme_block(&f32).expect(expect);
        if sme_store {
            out.push_str(&sme_panel(t, ti, tj, PanelWrite::Store));
        }
        if sme_acc {
            out.push_str(&sme_panel(t, ti, tj, PanelWrite::Accumulate));
        }
    }
    // The runtime's init is `main`'s first call exactly when a site calls the
    // runtime: the same gate as `GPU_INIT_DECL`.
    let prologue = if gpu_host {
        crate::func::gpu::init_call(
            crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu),
            funcs.contains("@mapal_gpu_launchv("),
            crate::func::gpu::pdl(profile, opts.contract, opts.gpu),
        )
    } else {
        String::new()
    };
    out.push_str(&emit_main_wrapper(ir, &prologue));
    let gens: Vec<crate::func::gpu::GenKernel> = gens
        .into_iter()
        .map(|(gid, body, elem)| (gid, fnames[body].clone(), elem, texts[body].clone()))
        .collect();
    let device = opts
        .tiling
        .then(|| {
            crate::func::gpu::device_module(
                ir,
                profile,
                opts.contract,
                opts.gpu,
                opts.reduced,
                &gens,
            )
        })
        .flatten();
    Ok(Emission {
        text: out,
        device,
        profile: profile.name,
    })
}

fn path_plan_is_acyclic(plan: &PathPlan) -> bool {
    let mut remaining = plan
        .tasks
        .iter()
        .map(|task| task.deps.len())
        .collect::<Vec<_>>();
    let mut ready = remaining
        .iter()
        .enumerate()
        .filter_map(|(task, &deps)| (deps == 0).then_some(task))
        .collect::<Vec<_>>();
    let mut cursor = 0;
    while cursor < ready.len() {
        let before = ready[cursor];
        cursor += 1;
        for (after, task) in plan.tasks.iter().enumerate() {
            if task.deps.contains(&before) {
                remaining[after] -= 1;
                if remaining[after] == 0 {
                    ready.push(after);
                }
            }
        }
    }
    ready.len() == plan.tasks.len()
}

/// Map every Map/Fold body reached from a parallel entry site to that site's
/// topo position. Lowering mints body functions per site, so the map is
/// single-valued; nested bodies inherit the outer entry site's trap position.
fn parallel_body_sites(
    ir: &CategoryIr,
    entry: FuncId,
    plan: &PathPlan,
) -> SecondaryMap<FuncId, u32> {
    let topo = ir.topo_order(entry);
    let mut topo_pos: SecondaryMap<MorphismId, u32> = SecondaryMap::new();
    for (i, &m) in topo.iter().enumerate() {
        topo_pos.insert(m, i as u32);
    }

    let mut out = SecondaryMap::new();
    for task in &plan.tasks {
        let members: &[MorphismId] = match &task.kind {
            TaskKind::Split { site, .. } => std::slice::from_ref(site),
            TaskKind::Seq { morphisms } => morphisms,
        };
        for &m in members {
            if let Operation::Map { body, .. } | Operation::Fold { body, .. } =
                ir.morphism(m).expect("morphism resolves").op
            {
                mark_body_closure(ir, body, topo_pos[m], &mut out);
            }
        }
    }
    out
}

fn mark_body_closure(
    ir: &CategoryIr,
    body: FuncId,
    site_topo: u32,
    out: &mut SecondaryMap<FuncId, u32>,
) {
    if out.insert(body, site_topo).is_some() {
        return;
    }
    for &m in &ir.func(body).expect("body resolves").morphisms {
        if let Operation::Map { body, .. } | Operation::Fold { body, .. } =
            ir.morphism(m).expect("morphism resolves").op
        {
            mark_body_closure(ir, body, site_topo, out);
        }
    }
}
