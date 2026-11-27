//! The device module text and the plain kernel texts.

use super::*;

/// The device translation unit: one kernel per GPU tile site in `ir`, in name
/// order, kept only where a host launch site names it (`live`, see
/// [`launched_kernels`]). `None` when no site offloads (every CPU profile, and
/// a GPU profile on a program with no tile site).
pub(crate) fn device_module(
    ir: &CategoryIr,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
    gens: &[GenKernel],
    epis: &[EpiSite],
    epi_bodies: &[(String, String)],
    cells: &[CellKernel],
    folds: &[RowFoldSite],
    units: &[super::super::flow::UnitKernel],
    live: &std::collections::BTreeSet<String>,
    hz: &[super::super::horizontal::HzSpec],
) -> Option<String> {
    // kernel-merging `horizontal`: a merged kernel's members are composed
    // from their kernels' text (kept standalone only where `live` names them).
    let launched = live;
    let with_members: std::collections::BTreeSet<String> = live
        .iter()
        .cloned()
        .chain(super::super::horizontal::hz_members(hz))
        .collect();
    let live = &with_members;
    let mut kernels = std::collections::BTreeMap::new();
    // per-site-choice: each candidate table's `; mapal.site` line.
    let mut site_lines = std::collections::BTreeSet::new();
    // The module's reduced face, for the evidence header: any kernel realized on the tensor cores (ADR-0041 D6).
    let mut reduced = None;
    for (f, _) in ir.funcs() {
        let rungs = host_rungs(profile, contract, arms);
        let plan = tile_plan(ir, f, profile, contract, arms);
        let dev = placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
        // A retained site's restricted pass (S53) launches its one-row kernel.
        let retained = super::super::retain::retained_sites(ir, f, profile, rungs);
        // S54 `rowfold`: the device folds, and the sites that store for them.
        let folds = super::super::rowfold::rowfold_sites(ir, f, profile, contract, arms);
        let col = |m: MorphismId| folds.values().any(|r| r.colmajor == Some(m));
        // A grouped site (S54 `epilogue`) launches its epilogue kernel instead.
        let sites = plan
            .sites
            .iter()
            .filter(|&(m, _)| dev.contains_key(m))
            .filter(|&(m, _)| !epis.iter().any(|e| e.f == f && e.m == m))
            .map(|(m, s)| (s, col(m), true));
        // A retained site's row kernel exists where the site is placed (S62).
        let rows = retained
            .iter()
            .filter(|&(y, _)| ir.in_edges(y).first().is_some_and(|&m| dev.contains_key(m)))
            .map(|(_, r)| (&r.row, false, false));
        for (site, col, whole) in sites.chain(rows) {
            if gpu_site(site, profile, rungs.window) {
                let l = launch(site, profile, contract, arms);
                if l.mma.is_some() || l.wgmma.is_some() {
                    reduced = site_reduced(site);
                }
                let name = kernel_name(kernel_id_col(site, profile, contract, col));
                kernels.entry(name.clone()).or_insert_with(|| {
                    kernel_text(
                        &name,
                        site,
                        profile,
                        contract,
                        arms,
                        col,
                        None,
                        Combine::None,
                    )
                });
                // per-site-choice: the candidates of a table (no column-major
                // store: its consumer reads the one layout).
                if !col {
                    cand_kernels(
                        &mut kernels,
                        &mut site_lines,
                        live,
                        site,
                        profile,
                        contract,
                        arms,
                        order,
                        !whole,
                        None,
                    );
                }
                // S62/S63 (ADR-0040 D4): a whole site whose fold is cut gets
                // its combine kernels too; the host picks per function
                // (resident, no epilogue, one stream), so the ordered one stays.
                let c = launch_in(site, profile, contract, arms, order).combine;
                // site-geometry `row-split`: a retained row's atomic split or
                // stream-K kernel (the host zeroes the row; no workspace).
                let row = l.rungs.row_split
                    && match c {
                        Combine::Workspace { .. } => order == Order::Free,
                        _ => true,
                    };
                if !(whole || row) || col || c == Combine::None {
                    continue;
                }
                let id = |part| kernel_name(kernel_id_split(site, profile, contract, part));
                let part = match c {
                    Combine::Workspace { .. } => "splitk",
                    Combine::Atomic { .. } => "atomic",
                    _ => "streamk",
                };
                let k = id(part);
                kernels.entry(k.clone()).or_insert_with(|| {
                    kernel_text(&k, site, profile, contract, arms, false, None, c)
                });
                if let Combine::Workspace { s } = c {
                    let red = id("reduce");
                    kernels.entry(red.clone()).or_insert_with(|| {
                        format!(
                            "; mapal.plan kernel={red} kind=reduce rows={} c={} S={s}\n",
                            site.rows, site.c
                        ) + &reduce_kernel_text(&red, site, profile, contract, arms, s, None)
                    });
                }
            }
        }
        // generic-fusion: a fold a unit holds launches inside the unit.
        let fused = |m: MorphismId| units.iter().any(|u| u.f == f && u.members.contains(&m));
        for (_, r) in folds.iter().filter(|&(m, _)| !fused(m)) {
            let (name, text) = rowfold_entry(
                &r.site,
                r.colmajor.is_some(),
                profile,
                contract,
                arms,
                order,
            );
            kernels.entry(name).or_insert(text);
        }
    }
    // S65: the folds a function placed over a device output no tile site
    // wrote (the host's plan names them).
    for f in folds {
        let (name, text) = rowfold_entry(f, false, profile, contract, arms, order);
        kernels.entry(name).or_insert(text);
    }
    for e in epis {
        if e.terms > 1 {
            let part = format!("concat{}x{}", e.terms, e.s);
            let name = kernel_name(kernel_id_split(&e.site, profile, contract, &part));
            kernels.entry(name.clone()).or_insert_with(|| {
                concat_kernel_text(&name, &e.site, profile, contract, arms, e.terms, e.s)
            });
            continue;
        }
        let l = launch(&e.site, profile, contract, arms);
        if l.mma.is_some() || l.wgmma.is_some() {
            reduced = site_reduced(&e.site);
        }
        let id = kernel_id_with(&e.site, profile, contract, Some(&e.k));
        let name = kernel_name(id);
        kernels.entry(name.clone()).or_insert_with(|| {
            kernel_text(
                &name,
                &e.site,
                profile,
                contract,
                arms,
                false,
                Some(&e.k),
                Combine::None,
            )
        });
        // An epilogue group's kernel is ordered: its candidates are too — a
        // `tile-post` group's may split into a workspace (`group_order`).
        let (garms, gorder) = group_order(Some(&e.k), &e.site, profile, arms, order);
        cand_kernels(
            &mut kernels,
            &mut site_lines,
            live,
            &e.site,
            profile,
            contract,
            garms,
            gorder,
            false,
            Some(&e.k),
        );
    }
    if kernels.is_empty() && gens.is_empty() && cells.is_empty() && units.is_empty() {
        return None;
    }
    // Launch-site use counts: a kernel no launch site names is dead.
    kernels.retain(|name: &String, _| live.contains(name));
    let mut out = String::from("; mapal-backend-llvm device module\n");
    // S64 evidence: the module's face (each kernel's facts: `; mapal.plan`).
    out.push_str(&format!(
        "; mapal.device target={} fma={} order={order:?} reduced={:?}\n",
        profile.name,
        u8::from(contract),
        reduced
    ));
    for l in &site_lines {
        out.push_str(l);
    }
    out.push_str(DEVICE_PREAMBLE);
    // bf16-class `device-trap`: the trap word and its record, where a cell
    // body checks an index.
    if cells
        .iter()
        .any(|(fns, _)| fns.iter().any(|(_, t)| t.contains("@dev_trap_record(")))
    {
        out.push_str(DEVICE_TRAP);
    }
    let mut md = String::new();
    for (i, k) in kernels.values().enumerate() {
        out.push('\n');
        if k.contains("!@LOOP@") {
            let n = 2 * i;
            out.push_str(&k.replace("!@LOOP@", &format!("!{n}")));
            md.push_str(&format!(
                "!{n} = distinct !{{!{n}, !{}}}\n!{} = !{{!\"llvm.loop.unroll.full\"}}\n",
                n + 1,
                n + 1
            ));
        } else {
            out.push_str(k);
        }
    }
    // `devgen`: each body once, then its kernel. A body is a clean leaf, so
    // its text names no host symbol — checked, since a stray one would fail
    // only at `llc` on the bundle machine.
    // Two producers of one body and length (the inliner's copies share
    // their bodies) are one kernel: its id names the body and the length.
    let mut bodies = std::collections::BTreeSet::new();
    let mut ids = std::collections::BTreeSet::new();
    for (id, callee, elem, text) in gens {
        if !live.contains(&format!("mapal_gen_{id:016x}")) || !ids.insert(*id) {
            continue;
        }
        assert!(
            !text.contains("@mapal_"),
            "devgen body {callee} names a host runtime symbol"
        );
        if bodies.insert(callee.clone()) {
            out.push('\n');
            out.push_str(text);
        }
        out.push('\n');
        out.push_str(&format!(
            "; mapal.plan kernel=mapal_gen_{id:016x} kind=gen\n"
        ));
        out.push_str(&gen_kernel_text(*id, callee, elem));
    }
    // The bodies an epilogue calls (S54): the host's text, or its device
    // variant (`_cell`); clean leaves, under the same check.
    for (callee, text) in epi_bodies {
        assert!(
            !text.contains("@mapal_"),
            "epilogue body {callee} names a host runtime symbol"
        );
        if bodies.insert(callee.clone()) {
            out.push('\n');
            out.push_str(text);
        }
    }

    // `devmap`: each body once (shared with `devgen`'s), then its kernel.
    let mut emitted = std::collections::BTreeSet::new();
    for (fns, kernel) in cells {
        let name = kernel_symbol(kernel);
        // One kernel per id: cells of one body, extent and variant (map
        // sites a body is shared by, an inlined layer's copies) launch it.
        if !live.contains(name) || !emitted.insert(name) {
            continue;
        }
        for (callee, text) in fns {
            assert!(
                !text.contains("@mapal_"),
                "cell body {callee} names a host runtime symbol"
            );
            if bodies.insert(callee.clone()) {
                out.push('\n');
                out.push_str(text);
            }
        }
        // `emit_cell`'s facts line, then the kernel; a uniform is an `i64`
        // parameter of its `define`.
        let (facts, kernel) = kernel
            .strip_prefix("; mapal.cell ")
            .and_then(|r| r.split_once('\n'))
            .unwrap_or(("", kernel));
        let uniform = kernel
            .lines()
            .find(|l| l.starts_with("define"))
            .map_or(0, |d| d.matches("i64 %c").count());
        out.push_str("\n; cell\n");
        out.push_str(&format!(
            "; mapal.plan kernel={name} kind=cell uniform={uniform} {facts}\n"
        ));
        out.push_str(kernel);
    }
    // generic-fusion: each unit's bodies once (shared with the others'), then
    // its kernel.
    for u in units {
        if !live.contains(kernel_symbol(&u.text)) {
            continue;
        }
        for (callee, text) in &u.fns {
            assert!(
                !text.contains("@mapal_"),
                "unit body {callee} names a host runtime symbol"
            );
            if bodies.insert(callee.clone()) {
                out.push('\n');
                out.push_str(text);
            }
        }
        out.push('\n');
        out.push_str(&u.text);
    }
    // model-ready `lanes-unroll`: each warp-a-cell fold loop its own loop id,
    // unrolled `NAIVE_UNROLL` trips (`FnEmit::emit_fold`'s `!@UNROLL@`).
    // residency `fly-unroll`: `!@UNROLL:<n>@` names its own count.
    let mut id = 2 * kernels.len();
    while let Some(at) = out.find("!@UNROLL") {
        let tail = at + "!@UNROLL".len();
        let close = tail + out[tail..].find('@').expect("a closed placeholder");
        let count = out[tail..close]
            .strip_prefix(':')
            .and_then(|c| c.parse::<u64>().ok())
            .unwrap_or(NAIVE_UNROLL);
        out.replace_range(at..close + 1, &format!("!{id}"));
        md.push_str(&format!(
            "!{id} = distinct !{{!{id}, !{}}}\n!{} = !{{!\"llvm.loop.unroll.count\", i32 {count}}}\n",
            id + 1,
            id + 1
        ));
        id += 2;
    }
    out.push('\n');
    out.push_str(&md);
    let mut out = super::super::horizontal::compose(out, hz, launched);
    // long-rows `row-slices`: a sliced unit's grid barrier, where no
    // persistent kernel brought it.
    if out.contains("call void @mapal_gsync(")
        && !out.contains("define internal void @mapal_gsync(")
    {
        let decls = super::super::calls::missing_decls(&out);
        out.push('\n');
        out.push_str(&decls);
        out.push_str(super::super::calls::GSYNC);
    }
    // ADR-0042 / ADR-0046: the device's math definitions, for the builtins
    // and widths a body calls — each the cheapest realization the part's
    // declared units admit on the face (`ex2` where its measured error meets
    // `exp`'s bound).
    {
        let r = host_rungs(profile, contract, arms);
        let none = mapal_ir::math::Target::NONE;
        let target = profile.gpu().map_or(&none, |g| &g.caps.math);
        let side = crate::math::Side {
            free: order == Order::Free,
            contract,
            hw: r.math_hw,
            exp_hw: r.exp_hw,
            target,
        };
        let defs = crate::math::defs(&out, side);
        out.push_str(&defs);
    }
    // ADR-0043: the device's narrow roundings (`cvt.rn`).
    let caps = profile.gpu().map(|g| g.caps);
    let mut ptx = caps.map(|c| c.ptx);
    {
        let cvt_f8 = caps.and_then(|c| c.cvt_f8);
        let (defs, hw) = crate::narrow::defs(&out, true, cvt_f8.is_some());
        out.push_str(&defs);
        // bugs-properties `ptx-isa`: the conversions' encoding on this part.
        if hw {
            ptx = ptx.max(cvt_f8);
        }
    }
    // `pdl`: the runtime launches every kernel behind a kernel as a dependent
    // launch, so every kernel waits for its producer's grid before it reads.
    if pdl(ir, profile, contract, arms) {
        out = pdl_wait_every_kernel(&out);
    } else if pdl_part(profile, contract, arms) {
        out = strip_pdl(&out);
    }
    // A `wgmma` kernel's swizzle atoms, and every swizzled tensor box,
    // start on 1024-byte boundaries.
    if out.contains("wgmma.mma_async") || out.contains("cp.async.bulk.tensor") {
        out = out.replacen("[0 x i8], align 16", "[0 x i8], align 1024", 1);
    }
    // bugs-properties `ptx-isa`: the module declares the largest encoding its
    // target and its instructions need — every `llc` step passes it.
    if let Some(p) = ptx {
        out = out.replacen(
            "; mapal-backend-llvm device module\n",
            &format!("; mapal-backend-llvm device module\n{}", p.declare()),
            1,
        );
    }
    Some(out)
}

/// hardware-properties `pdl`: `griddepcontrol.wait` at the entry of every
/// kernel of the module that does not already open with it (the tile kernel
/// does, after its index setup). The runtime launches any kernel that follows
/// a kernel on its stream as a programmatic dependent launch
/// (`mapal_gpu.c`), so a kernel without the wait (a cell, a row fold, a unit,
/// a merged grid, a reduce) would read its producer's output before that
/// grid completes. The wait is a no-op on a plain or cooperative launch.
pub(crate) fn pdl_wait_every_kernel(module: &str) -> String {
    const WAIT: &str = "  call void asm sideeffect \"griddepcontrol.wait;\", \"~{memory}\"()";
    let mut out = String::with_capacity(module.len() + 4096);
    let mut lines = module.lines().peekable();
    while let Some(l) = lines.next() {
        out.push_str(l);
        out.push('\n');
        if !l.starts_with("define ptx_kernel ") {
            continue;
        }
        // The kernel's text up to its closing brace: does it wait already?
        let body: Vec<&str> = std::iter::from_fn(|| {
            let n = lines.next()?;
            Some(n)
        })
        .take_while(|n| *n != "}")
        .collect();
        let waits = body.iter().any(|b| b.contains("griddepcontrol.wait"));
        let mut inserted = waits;
        for b in &body {
            out.push_str(b);
            out.push('\n');
            if !inserted && b.ends_with(':') && !b.starts_with(' ') {
                out.push_str(WAIT);
                out.push('\n');
                inserted = true;
            }
        }
        out.push_str("}\n");
    }
    if !module.ends_with('\n') {
        out.pop();
    }
    out
}

pub(super) fn kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    col: bool,
    epi: Option<&EpiKernel>,
    combine: Combine,
) -> String {
    let mut k = kernel(site, profile, contract, arms, col, epi, combine);
    let l = launch_for(site, profile, contract, arms, combine);
    debug_assert!(
        !col || col_store(&l, site),
        "a column-major store this kernel has"
    );
    debug_assert!(
        combine == Combine::None
            || (!col && (epi.is_none() || matches!(combine, Combine::StreamK { .. })))
    );
    let walk = match combine {
        // bf16 tensor-tile `mma-walk`: the tensor tile's priced walk.
        Combine::StreamK { .. } if l.mma.is_some() => Some("mma"),
        Combine::StreamK { .. } if l.rungs.streamk => Some("class"),
        Combine::StreamK { .. } => Some(
            match k_fill(
                &launch(site, profile, contract, arms),
                site,
                profile,
                contract,
                arms,
            ) {
                Some((c, _)) if c == combine => "fill",
                _ => "price",
            },
        ),
        _ => None,
    };
    k.out = plan_line(
        name,
        site,
        profile,
        &l,
        k.idx,
        combine,
        col,
        epi.is_some(),
        walk,
    );
    // plan-cost: the site's pick on every kernel's line (D3); its candidates
    // on every row-major one (the harness keys them by `site=`).
    let pick = plan_pick(site, profile, contract, arms, !col);
    if !pick.is_empty() {
        k.out.pop();
        k.out += &pick;
        k.out.push('\n');
    }
    // ADR-0050 D8: the machine model's schedule, price, terms and open flips.
    if l.rungs.machine_place {
        k.out += &crate::place::place_line(site, profile, contract, arms, combine);
    }
    match (l.wgmma, l.mma, l.tiled) {
        (Some(w), _, _) if w.ws => k.wgmma_ws(name, w),
        (Some(w), _, _) => k.wgmma(name, w),
        (None, Some(m), _) if m.fmt != MmaFmt::Tf32 => k.mma16(name, m),
        (None, Some(m), _) => k.mma(name, m),
        (None, None, Some(g)) => k.tiled(name, g),
        (None, None, None) if l.cells > 1 => k.naive_cells(name, l.block, l.cells),
        (None, None, None) => k.naive(name, l.block),
    }
    k.stream_field();
    k.out
}

/// S62 `splitk`'s second launch: `o[t] = ((P₀[t] + P₁[t]) + …) + P_{S−1}[t]`,
/// one thread per cell over the workspace `%a` (`S` partitions of `rows·c`),
/// partition order fixed, straight-line — ADR-0040 D3's combine, so the bits
/// are a function of the site and the profile. No fast-math flag: LLVM may
/// not re-bracket what the text fixes. `%b` is unused.
pub(super) fn reduce_kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    s: u64,
    epi: Option<&EpiKernel>,
) -> String {
    let mut k = kernel(
        site,
        profile,
        contract,
        arms,
        false,
        epi,
        Combine::Workspace { s },
    );
    let (t, ix, cells) = (k.t.clone(), k.idx, site.rows * site.c);
    let e = elem_bytes_of(&site.elem);
    // `tile-post`: the header takes the group's pointers; the whole sum goes
    // through its store.
    k.header(name, NAIVE_THREADS, e, 0);
    let tid = k.sreg("tid.x");
    let bx = k.sreg("ctaid.x");
    let cell0 = k.op("mul", &bx, NAIVE_THREADS);
    let cell = k.op("add", &cell0, &tid);
    let inside = k.tmp();
    k.line(format!("{inside} = icmp ult {ix} {cell}, {cells}"));
    k.line(format!("br i1 {inside}, label %body, label %done"));
    k.out.push_str("body:\n");
    let mut acc = String::new();
    for z in 0..s {
        let i = k.plus(cell.clone(), z * cells);
        let p = k.gep(1, "%a", &i);
        let v = k.tmp();
        k.line(format!("{v} = load {t}, ptr addrspace(1) {p}, align {e}"));
        acc = if z == 0 {
            v
        } else {
            let r = k.tmp();
            k.line(format!("{r} = fadd {t} {acc}, {v}"));
            r
        };
    }
    if epi.is_some() {
        k.store_cells(std::slice::from_ref(&acc), &cell);
    } else {
        let p = k.gep(1, "%o", &cell);
        k.line(format!("store {t} {acc}, ptr addrspace(1) {p}, align {e}"));
    }
    k.line("br label %done");
    k.out.push_str("done:\n  ret void\n}\n");
    k.out
}

/// The kernel builder for a site's launch: its index width, arithmetic,
/// rungs and the fold's realization (`combine`; `None` for the ordered kernel).
#[allow(clippy::too_many_arguments)]
pub(super) fn kernel<'s>(
    site: &'s TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    col: bool,
    epi: Option<&'s EpiKernel>,
    combine: Combine,
) -> Kernel<'s> {
    let l = launch_for(site, profile, contract, arms, combine);
    let rungs = l.rungs;
    let splitk = combine.splits();
    // `idx32` is an FMA / one-thread-per-cell rung: the tensor-core kernels
    // (`mma.sync`, `wgmma`) form their indices in `i64`.
    let tensor = l.mma.is_some() || l.wgmma.is_some();
    let idx = if rungs.idx32 && !tensor && max_index(site, &l, combine) < 1 << 31 {
        "i32"
    } else {
        "i64"
    };
    Kernel {
        out: String::new(),
        next: 0,
        idx,
        t: lower_ty(&site.elem).expect("float elem lowers"),
        site,
        arithmetic: l.arithmetic,
        rungs: l.rungs,
        profile_gpu: *profile.gpu().expect("a GPU site has a GPU profile"),
        pdl: pdl_part(profile, contract, arms),
        async_tail: None,
        sync_tail: None,
        raster: (l.gx, l.raster),
        col,
        epi,
        splitk,
        gy_out: l.gy,
        z: None,
        atomic: matches!(combine, Combine::Atomic { .. } | Combine::StreamK { .. }),
        terms: 1,
        streamk: match combine {
            Combine::StreamK { iters, .. } => Some(iters),
            _ => None,
        },
        whole: match combine {
            Combine::StreamK { whole, .. } => whole,
            _ => 0,
        },
        ticket: epi.is_some() && matches!(combine, Combine::StreamK { .. }),
        part: false,
        p0: None,
        head: "entry",
        tmap: Vec::new(),
        streamed: [false; 2],
    }
}

/// The largest index a kernel forms (S53 rung `idx32`): each operand's last
/// element — `base + ci·(rows−1) + clane·(c−1) + ck·(k−1)`, site extents —
/// every cell a launched thread can name (the one-thread-per-cell grid rounds
/// up to whole blocks), and `k`. Every intermediate of the index arithmetic is
/// a partial sum of one of these, so below 2³¹ the kernel indexes in `i32`
/// with `nuw nsw`; pointers stay 64-bit. A workspace split kernel (S62)
/// names its workspace cell `z·rows·c + cell`, `z < S`; a partition's reads
/// stay inside the operand's own last element. A stream-K kernel (S63) counts
/// its iteration space up to `ctas·iters` (`tiles × panels`, rounded up).
pub(super) fn max_index(site: &TileSite, l: &Launch, combine: Combine) -> u64 {
    let rx = site.radix.as_deref();
    let last = |r: &mapal_ir::TileRead| {
        let ra = rx.and_then(|x| {
            if std::ptr::eq(r, &site.a) {
                x.a.as_ref()
            } else {
                x.b.as_ref()
            }
        });
        if let Some(ra) = ra {
            return ra.max();
        }
        let taps = r.ksplit.map_or(r.ck * (site.k - 1), |ks| {
            ks.cq * (site.k / ks.div - 1) + ks.cr * (ks.div - 1)
        });
        r.base + r.ci * (site.rows - 1) + r.clane * (site.c - 1) + taps
    };
    let slabs = match combine {
        Combine::Workspace { s } => s,
        _ => 1,
    };
    let walk = match combine {
        Combine::StreamK { ctas, iters, whole } => ctas * iters + whole * site.k,
        _ => 0,
    };
    let cells = (site.rows * site.c).max(l.gx * l.gy * l.block * l.cells) * slabs;
    last(&site.a)
        .max(last(&site.b))
        .max(cells)
        .max(site.k)
        .max(walk)
}

/// The integer arithmetic a panel lane map is written in, so one definition
/// both emits the kernel's index IR ([`Kernel`]) and evaluates on numbers
/// ([`Num`]) — the evaluation is what proves a copy affine ([`affine_delta`]).
pub(super) trait Arith {
    type V: Clone;
    fn imm(&mut self, code: &str, a: &Self::V, c: u64) -> Self::V;
    fn sum(&mut self, a: &Self::V, b: &Self::V) -> Self::V;
}
