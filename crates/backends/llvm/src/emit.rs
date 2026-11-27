//! The emission entry points ([`emit`], [`emit_detailed`]) and the module walk.
use super::*;

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
    // onto-device `cond-bounds` off: the facts rebuilt under plain intervals
    // (a fact arm, `mapal_ir::FactArms` — every fact derived afresh).
    let arms = mapal_ir::FactArms {
        cond_bounds: !opts.gpu.off.cond_bounds,
        row_tree: !opts.gpu.off.row_tree,
        piece_delta: !opts.gpu.off.piece_delta,
        head_group: !opts.gpu.off.head_group,
        exp_rescale: !opts.gpu.off.exp_rescale,
        uniform_base: !opts.gpu.off.uniform_base,
        unchanged: !opts.gpu.off.unchanged,
        narrow_radix: !opts.gpu.off.narrow_radix,
        stream: !opts.gpu.off.stream_fact,
        finite: !opts.gpu.off.finite,
        guard_box: !opts.gpu.off.guard_box,
        guard_terms: !opts.gpu.off.guard_terms,
        walk_order: !opts.gpu.off.walk_order,
        unit_uniform: !opts.gpu.off.unit_uniform,
    };
    if arms != ir.fact_arms() {
        return emit_detailed(&ir.refacted(arms), opts);
    }
    // Machine facts, by name (plan-s31-target-profiles rule 3). An unknown name
    // is an error, never a silent fall back to `generic` — a typo that quietly
    // emits the default profile's numbers is the exact failure the table exists
    // to remove.
    let unknown = |name: &str| {
        EmitError::Internal(format!(
            "unknown target profile `{name}`; known: {}",
            profile::names()
        ))
    };
    let mut profile = profile::resolve(opts.target).ok_or_else(|| unknown(opts.target))?;
    // ADR-0034 D1: a measurement arm's geometry overrides, on a copy of the part.
    if opts.geom.iter().any(Option::is_some)
        && let profile::Machine::Gpu(g) = profile.machine
    {
        let mut p = *profile;
        p.machine = profile::Machine::Gpu(profile::Gpu {
            geom: g.geom.with(&opts.geom),
            ..g
        });
        profile = Box::leak(Box::new(p));
    }
    // plan-cost: a fit arm's rates, on a copy of the part.
    if opts.cost.iter().any(Option::is_some)
        && let profile::Machine::Gpu(g) = profile.machine
    {
        let mut p = *profile;
        p.machine = profile::Machine::Gpu(g.with_cost(&opts.cost));
        profile = Box::leak(Box::new(p));
    }
    if profile.gpu().is_none() {
        return emit_module(ir, opts, profile);
    }
    // ADR-0041 D4: a tensor core's k-step sums in the hardware's order, so a
    // `tf32` product takes it only where the order is free; a sequential or
    // fixed face realizes it as f32 (a member of every set): the FFMA tile.
    // The same for a 16-bit member (ADR-0043 D4, `half-tile`): the exact and
    // fixed-order faces keep the oracle's order and its f32 products.
    let mut face = *opts;
    face.gpu.free = opts.order == Order::Free;
    if !face.gpu.free {
        face.gpu.off.tf32_tile = true;
        face.gpu.off.half_tile = true;
        // f32-tensor-cores: a split realization sums in the instruction's
        // order — the default face only.
        face.gpu.off.split_tile = true;
    }
    let opts = &face;
    // sweep-gemm `round-store`: on the face that realizes a product's member
    // (order free), an array whose every element read rounds to `bf16` is
    // stored rounded by its producer — the same values each reader's
    // rounding gives, computed once. The rewritten graph has no such array
    // left (its elements are narrow), so the second pass emits.
    if opts.gpu.free
        && !opts.gpu.off.round_store
        && let Some(stored) = ir.narrow_stored()
    {
        return emit_detailed(&stored, opts);
    }
    // ADR-0050 D3 `machine-place`: the sites' warm operands
    // (`residence_plan`), one map every launch of this emission reads.
    let _warm = crate::place::WarmScope::new(ir, profile, opts.gpu);
    // ... and the sites it stores column-major (the row folds' layout): a
    // pick that cannot store so is dropped there alone.
    let _col = crate::place::ColScope::new(ir, profile, opts.contract, opts.gpu);
    // cost-rental: the order this emission's host launches place each site
    // under — the kernel of a realization is that order's pick.
    let _order = crate::place::OrderScope::new(opts.order);
    // S62 `hostopt`: everything outside a device site is emitted against the
    // host's CPU facts; off, against `generic`'s — the S52–S54 GPU build.
    let rungs = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
    if !rungs.hostopt {
        return emit_module(ir, opts, profile).map(|em| host_line(ir, opts, profile, em));
    }
    let host = profile::resolve(opts.host).ok_or_else(|| unknown(opts.host))?;
    if host.gpu().is_some() {
        return Err(EmitError::Internal(format!(
            "host profile `{}` is a GPU profile",
            host.name
        )));
    }
    let mut both = profile::with_host(profile, host);
    // `place` off: no measured cost, so every capable site is a device site.
    // (A site admitting `tf32` is placed by the same roofline: the host's f32
    // realization is admitted too, ADR-0041 D6.)
    if !rungs.place
        && let profile::Machine::Gpu(g) = &mut both.machine
    {
        // plan-cost: the plan rates stay — `place` decides where a site
        // runs, not how (a zero placement rate is "no measured cost").
        g.cost = g
            .cost
            .map(|d| profile::DeviceCost {
                flops_per_ns: 0,
                bytes_per_ns: 0,
                launch_ns: 0,
                ..d
            })
            .filter(|d| d.issue_ns > 0);
    }
    let both: &'static TargetProfile = Box::leak(Box::new(both));
    let name: &'static str = format!("{}+{}", profile.name, host.name).leak();
    let em = emit_module(ir, opts, both)?;
    // A GPU build of a program whose every site stays on the host IS the
    // host's CPU build: parallel entry, the whole CPU ladder, no runtime.
    // ponytail: per site, not per program — a one-shot program pays the
    // device's start (context, module) once; gating on it needs the host cost
    // of what `devgen` builds on the device, which no profile fact carries.
    let em = if em.device.is_none() {
        emit_module(ir, opts, host)?
    } else {
        em
    };
    Ok(host_line(
        ir,
        opts,
        both,
        Emission {
            profile: name,
            ..em
        },
    ))
}

/// S64 evidence pass: a GPU build's host module ends with one `; mapal.host`
/// comment — the profiles it was emitted against, its face, and how many
/// capable tile sites `place` put on the device and kept on the host
/// (`func/gpu.rs:placement`, the walk's own answer). `benches/gpu/evidence.py`
/// reads it; a comment, so the object is unchanged.
pub(super) fn host_line(
    ir: &CategoryIr,
    opts: &EmitOpts,
    profile: &'static TargetProfile,
    mut em: Emission,
) -> Emission {
    // A build with no device unit IS the host's CPU build, byte for byte.
    if em.device.is_none() {
        return em;
    }
    let rungs = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
    let (mut dev, mut host) = (0, 0);
    for (f, _) in ir.funcs() {
        let plan = crate::func::gpu::tile_plan(ir, f, profile, opts.contract, opts.gpu);
        let placed =
            crate::func::gpu::placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
        for (m, s) in plan.sites.iter() {
            if crate::func::gpu::gpu_site(s, profile, rungs.window) {
                if placed.contains_key(m) {
                    dev += 1;
                } else {
                    host += 1;
                }
            }
        }
    }
    // residency `calib-supply`: where a `native-gpu` build's measured costs
    // came from (`file`, `committed`), or `none`.
    let calib = match (profile.name, profile::native_gpu_calib()) {
        (profile::NATIVE_GPU, Some(profile::CalibSource::File(_))) => " calib=file",
        (profile::NATIVE_GPU, Some(profile::CalibSource::Committed)) => " calib=committed",
        (profile::NATIVE_GPU, _) => " calib=none",
        _ => "",
    };
    em.text.push_str(&format!(
        "; mapal.host profile={} fma={} order={:?} device_sites={dev} host_sites={host}{calib}\n",
        em.profile,
        u8::from(opts.contract),
        opts.order
    ));
    em
}

/// One module against one resolved profile — a CPU profile, or a GPU profile
/// whose CPU-shaped fields describe the host ([`emit_detailed`]).
pub(super) fn emit_module(
    ir: &CategoryIr,
    opts: &EmitOpts,
    profile: &'static TargetProfile,
) -> Result<Emission, EmitError> {
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
    // around the offloaded sites sequentially (S62: a GPU build with no device
    // site is re-emitted as the host's CPU build, `emit_detailed`, so it takes
    // this parallel entry). ponytail: host threads alongside the device is
    // co-execution (ADR-0035) — add when a program needs both.
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
            let plan = ir.tile_plan(f);
            let r = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
            let dev = crate::func::gpu::placement(ir, f, &plan, profile, r.window, r.carry_made);
            plan.sites
                .iter()
                .any(|(m, site)| !dev.contains_key(m) && packing_site(site, profile, opts.contract))
        });

    // Functions first: the arena declarations are gated on the emitted text
    // (below), so the bodies have to exist before the header is assembled.
    let mut funcs = String::new();
    // `devgen` (S53): the producers each function computes on the device, and
    // every function's emitted text — a device-built producer's body is the
    // host's own, appended to the device unit.
    let mut gens: Vec<(u64, mapal_ir::FuncId, String)> = Vec::new();
    // `epilogue` (S54): the grouped sites, and the bodies their kernels call.
    let mut epis: Vec<crate::func::gpu::EpiSite> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut epi_calls: Vec<(
        String,
        mapal_ir::FuncId,
        Vec<u32>,
        Option<mapal_ir::MorphismId>,
    )> = Vec::new();
    // `devmap` (S54): each function's cell kernels and the bodies they call.
    let mut cells: Vec<crate::func::gpu::CellKernel> = Vec::new();
    // S65: the row folds each function runs on the device.
    let mut folds: Vec<mapal_ir::RowFoldSite> = Vec::new();
    // `flow`: each function's unit kernels.
    let mut units: Vec<crate::func::flow::UnitKernel> = Vec::new();
    let mut texts: SecondaryMap<mapal_ir::FuncId, String> = SecondaryMap::new();
    let mut orders: Vec<FoldOrder> = Vec::new();
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
                opts.order,
                opts.kc_nest,
                opts.move_panel,
                profile,
                opts.gpu,
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
                opts.order,
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
            fe.set_gpu_arms(opts.gpu);
            if let Some(&site) = body_sites.get(id) {
                fe.set_task_body_site(site);
            }
            gens.extend(fe.gen_kernels());
            let (es, cs) = fe.epi_sites();
            epis.extend(es);
            epi_calls.extend(cs);
            cells.extend(fe.cell_kernels());
            folds.extend(fe.rowfold_kernels());
            units.extend(fe.unit_kernels());
            let (text, os) = fe.emit_ordered();
            orders.extend(os);
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
    // onto-device `host-par`: a GPU build's walk dispatches split tasks too.
    if parallel || funcs.contains("@mapal_par_begin(") {
        out.push_str(PAR_DECLS);
    }
    // ponytail: gate the arena declarations on the emitted call itself, not on
    // a re-derived predicate — the call IS the requirement, so the two cannot
    // drift the way a `prefetch`-style pre-pass could. (PAR/PERF must gate on a
    // predicate: they are decided before any body exists.)
    if funcs.contains("@mapal_rt_alloc") {
        out.push_str(HEAP_DECLS);
    }
    // ADR-0045: the read seam and the recorded data directory, gated on the
    // emitted call (a module without a read is byte-identical to before).
    if funcs.contains("@mapal_read(") {
        out.push_str(crate::module::READ_DECL);
    }
    if funcs.contains("@mapal_read_task(") {
        out.push_str(crate::module::READ_TASK_DECL);
    }
    if funcs.contains("@mapal_rt_frame_alloc") {
        out.push_str(
            "declare i64 @mapal_rt_frame_mark()\ndeclare ptr @mapal_rt_frame_alloc(i64, i64)\ndeclare void @mapal_rt_frame_release(i64)\n",
        );
    }
    // fold-algebra: an explicit-face pair is one `llvm.fma` (gated on the call).
    for (call, decl) in [
        (
            "@llvm.fma.f32(",
            "declare float @llvm.fma.f32(float, float, float)\n",
        ),
        (
            "@llvm.fma.f64(",
            "declare double @llvm.fma.f64(double, double, double)\n",
        ),
    ] {
        if funcs.contains(call) {
            out.push_str(decl);
        }
    }
    // ADR-0042 / ADR-0046: the host's math definitions, for the builtins and
    // widths a body calls (the host's declared units: its IEEE `sqrt`).
    {
        let r = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
        let side = crate::math::Side {
            free: opts.order == Order::Free,
            contract: opts.contract,
            hw: r.math_hw,
            exp_hw: r.exp_hw,
            target: &profile.math,
        };
        out.push_str(&crate::math::defs(&funcs, side));
    }
    // ADR-0043: the host's narrow roundings, for the formats a body rounds to.
    out.push_str(&crate::narrow::defs(&funcs, false, false).0);
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
    // `arena` (S54): a function whose every device object sits in its arena
    // calls `mapal_gpu_arena` and no `mapal_gpu_alloc`.
    let arena = funcs.contains("@mapal_gpu_arena(");
    let resident = funcs.contains("@mapal_gpu_alloc(") || arena;
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
        if funcs.contains("@mapal_gpu_launch_l2(") {
            out.push_str(crate::func::gpu::GPU_L2_DECL);
        }
        // `streams`, `pinned` (S54): each entry declared only when called.
        for (call, decl) in crate::func::gpu::S54_DECLS {
            if funcs.contains(call) {
                out.push_str(decl);
            }
        }
    }
    if arena {
        out.push_str(crate::func::gpu::ARENA_DECL);
    }
    // `epilogue`, `devmap` and `streams` launch over n pointers: one declaration.
    if funcs.contains("@mapal_gpu_launchv(") {
        out.push_str(crate::func::gpu::LAUNCHV_DECL);
    }
    // bf16-class `device-trap`: a trap-checking cell's launch arms the
    // checkpoint's read of the device's trap word.
    if funcs.contains("@mapal_gpu_trap_arm(") {
        out.push_str("declare void @mapal_gpu_trap_arm()\n");
    }
    // kernel-merging `calls`: a persistent kernel's cooperative launch.
    if funcs.contains("@mapal_gpu_launch_coop(") {
        out.push_str(crate::func::gpu::COOP_DECL);
    }
    // per-site-choice: a site with candidates launches its table.
    if funcs.contains("@mapal_gpu_site(") {
        out.push_str(crate::func::gpu::SITE_DECL);
    }
    // gaps-disasm C1 `load-dispatch`: the tables registered before the init.
    let host_rungs = crate::func::gpu::host_rungs(profile, opts.contract, opts.gpu);
    let (n_sites, registry) = match gpu_host && host_rungs.load_dispatch {
        true => crate::func::gpu::site_registry(&funcs),
        false => (0, String::new()),
    };
    out.push_str(&registry);
    let sme_store = funcs.contains("@mapal_sme_panel(");
    let sme_acc = funcs.contains("@mapal_sme_panel_acc(");
    let sme = sme_store || sme_acc;
    if sme {
        out.push_str(SME_DECLS);
    }
    out.push('\n');

    let mut sg = emit_str_globals(&strings);
    if funcs.contains(crate::module::DATA_DIR_GLOBAL) {
        sg.push_str(&crate::module::data_dir_global(ir.data_dir().unwrap_or("")));
    }
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
            host_rungs,
            // Streams are opened where a launch names one: `launchv` on stream 0
            // is also `epilogue`'s and `devmap`'s (S54 integration).
            funcs
                .split("@mapal_gpu_launchv(i64 ")
                .skip(1)
                .any(|t| t.split(", ").nth(1).is_some_and(|s| s != "i32 0")),
            crate::func::gpu::pdl(ir, profile, opts.contract, opts.gpu),
            crate::func::gpu::module_zeroes(&funcs),
            n_sites,
        )
    } else {
        String::new()
    };
    out.push_str(&emit_main_wrapper(ir, &prologue));
    // A device body's nested bodies are the host's text (generic-fusion
    // natural forms: a fold inside a cell), carried with the epilogues'.
    let nested: Vec<mapal_ir::FuncId> = gens
        .iter()
        .map(|g| g.1)
        .chain(epi_calls.iter().map(|e| e.1))
        .flat_map(|b| crate::func::resident::nested_bodies(ir, b))
        .collect();
    let gens: Vec<crate::func::gpu::GenKernel> = gens
        .into_iter()
        .map(|(gid, body, elem)| (gid, fnames[body].clone(), elem, texts[body].clone()))
        .collect();
    // An epilogue calls the host's body text, or its device variant: the
    // same emitter with the cell-read slots as one-cell buffers.
    let mut epi_bodies: Vec<(String, String)> = Vec::new();
    for nb in nested {
        if !epi_bodies.iter().any(|(c, _)| *c == fnames[nb]) {
            epi_bodies.push((fnames[nb].clone(), texts[nb].clone()));
        }
    }
    for (callee, body, slots, post) in epi_calls {
        if epi_bodies.iter().any(|(c, _)| *c == callee) {
            continue;
        }
        let text = if slots.is_empty() && post.is_none() {
            texts[body].clone()
        } else {
            let mut fe = FnEmit::new(
                ir,
                body,
                &fnames,
                &strings,
                &attrs,
                opts.tiling,
                opts.packing,
                opts.contract,
                opts.order,
                opts.kc_nest,
                opts.move_panel,
                profile,
            );
            fe.set_gpu_arms(opts.gpu);
            fe.set_cell_slots(slots);
            fe.set_post_fold(post);
            fe.emit()
        };
        epi_bodies.push((callee, text));
    }
    let device = opts
        .tiling
        .then(|| {
            crate::func::gpu::device_module(
                ir,
                profile,
                opts.contract,
                opts.gpu,
                opts.order,
                &gens,
                &epis,
                &epi_bodies,
                &cells,
                &folds,
                &units,
                &crate::func::gpu::launched_kernels(&out),
                &crate::func::horizontal::hz_specs(&out),
            )
        })
        .flatten();
    Ok(Emission {
        text: out,
        device,
        profile: profile.name,
        orders,
    })
}

pub(super) fn path_plan_is_acyclic(plan: &PathPlan) -> bool {
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
pub(super) fn parallel_body_sites(
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

pub(super) fn mark_body_closure(
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
