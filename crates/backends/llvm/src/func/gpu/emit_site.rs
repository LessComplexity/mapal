//! Host side: `FnEmit` emission of a GPU tile site, its site table and epilogue kernels.

use super::*;

impl<'a> FnEmit<'a> {
    /// per-site-choice B2: table `t` as a module constant and its one
    /// `mapal_gpu_site` call over the launch's pointers `ptrs` (the output
    /// third). The workspace slot, when a candidate takes one, is sized at the
    /// table's largest (`entry-arena` seeds it like any split workspace).
    pub(crate) fn site_call(&mut self, t: SiteTable, ptrs: &[String]) -> String {
        let global = format!("@{}.cands{}", self.fnames[self.f], self.fresh());
        self.cand_tables.push_str(&format!(
            "{global} = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
            t.rows.len(),
            t.rows.join(", ")
        ));
        let ws = match t.ws {
            0 => "null".to_string(),
            bytes => {
                let slot = format!("%d{}", self.fresh());
                self.allocas.push_str(&format!(
                    "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                ));
                self.split_slots.push(slot.clone());
                self.dallocs.push((slot.clone(), bytes, 0));
                slot
            }
        };
        let n = ptrs.len();
        let arr = format!("%p{}", self.fresh());
        self.allocas
            .push_str(&format!("  {arr} = alloca [{n} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        self.line(format!(
            "call void @mapal_gpu_site(i64 {}, i32 {}, i32 0, i32 {}, ptr {global}, i32 {n}, \
             ptr {arr}, i32 2, ptr {ws})",
            t.site as i64,
            t.flags,
            t.rows.len()
        ));
        global
    }

    /// per-site-choice: whether site `m`'s launch may run again (`RERUN`) —
    /// no buffer it writes (the output, an epilogue's stored results) is one
    /// it reads (an operand, an epilogue's complete operand). Every morphism
    /// writes a fresh object and the arena keeps a launch's objects apart
    /// (abi-map §3), so only a residency alias can join them.
    pub(super) fn rerun_safe(
        &self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        m: MorphismId,
    ) -> bool {
        let alias = |o| self.resident.as_ref().map_or(o, |p| p.alias(o));
        let mut reads: Vec<ObjectId> = [site.a.slot, site.b.slot]
            .iter()
            .filter_map(|&k| self.pair_source(source, k))
            .collect();
        let mut writes = vec![target];
        if let Some(g) = self.groups.get(m) {
            reads.extend(g.caps());
            writes.extend(g.chain.iter().filter(|mb| mb.store).map(|mb| mb.out));
        }
        apart(&reads, &writes, alias)
    }

    /// per-site-choice B1/B2: site `site`'s table, `one` its retained row's
    /// one-row site (`row`) or the whole site. `None` with fewer than two
    /// candidates — but for a `tile-post` group whose one candidate splits
    /// (`group_order`: a workspace whose reduce evaluates the epilogue): the
    /// group's plain launch is its ordered kernel, so the split reaches the
    /// device only through the table (the f8e4m3 decode's `wo` projection, a
    /// scale and a residual after the fold: its one candidate `splitk16`
    /// ran as 16 one-cell blocks, 108 against 39 µs a step for its bf16
    /// twin's two-candidate table). A one-candidate table never trials.
    pub(super) fn make_table(
        &self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        cands: &[SiteCand],
        one: &TileSite,
        epi: Option<&EpiKernel>,
    ) -> Option<SiteTable> {
        // One candidate launches as the rules' kernel — except an epilogue
        // group's split (`group_order`): the single launch keeps the group
        // ordered, so its realized combine is the table's (cross-arch
        // 3a8e654f: the B1 decode's O projection, S 64 planned on the H100,
        // ran its ordered 16-block kernel, 235 against 29 us). `post-table`
        // (OFF: the ordered kernel, the emission before); `--cands=1` keeps
        // it — candidate 0 is the rules' plan, and this is its launch.
        let split_epi = host_rungs(self.profile, self.contract, self.gpu_arms).post_table
            && epi.is_some()
            && cands
                .first()
                .is_some_and(|c| c.launch.combine != Combine::None);
        if cands.len() < 2 && !split_epi {
            return None;
        }
        let m = self.ir.in_edges(target)[0];
        let ids: Vec<(u64, u64)> = cands
            .iter()
            .enumerate()
            .map(|(i, c)| cand_ids(one, self.profile, self.contract, epi, &c.launch, i == 0))
            .collect();
        let (rows, ws): (Vec<String>, Vec<u64>) = cands
            .iter()
            .zip(&ids)
            .map(|(c, &id)| recipe(one, &c.launch, id, epi.is_some()))
            .unzip();
        Some(SiteTable {
            site: ids[0].0,
            flags: if cands.len() > 1 && self.rerun_safe(source, target, site, m) {
                SITE_RERUN
            } else {
                0
            },
            rows,
            ws: ws.into_iter().max().unwrap_or(0),
        })
    }

    /// Host half: hand the site's operands and output to the runtime, which
    /// copies them in, launches [`kernel_name`]`(id)`, and copies the output
    /// back before returning — so everything after the site sees the values a
    /// host loop would have stored.
    pub(crate) fn emit_tiled_map_gpu(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
    ) {
        if let Some(plan) = &self.resident {
            // S54 `rowfold-layout`: stored in the order its only reader reads.
            let m = *self.ir.in_edges(target).first().expect("site morphism");
            let col = plan.rowfold.values().any(|r| r.colmajor == Some(m))
                || plan.units.values().flat_map(|u| &u.members).any(|mb| {
                    matches!(mb.kind, super::super::flow::UKind::Fold { colmajor: Some(c), .. } if c == m)
                });
            let streams = !plan.chain.is_empty();
            // plan-cost `linear`: the site table — the sum zeroed, one launch
            // (`emit_site_resident` passes every term's operands).
            if let Some(g) = self.groups.get(m).filter(|g| !g.terms.is_empty()) {
                let (t, s) = (g.terms.len() as u64 + 1, g.split);
                let id =
                    kernel_id_split(site, self.profile, self.contract, &format!("concat{t}x{s}"));
                let l = launch_for(
                    site,
                    self.profile,
                    self.contract,
                    self.gpu_arms,
                    Combine::Atomic { s },
                );
                let geom = format!(
                    "i32 {}, i32 {}, i32 {}, i32 {}",
                    l.gx,
                    l.gy * s * t,
                    l.block,
                    l.smem
                );
                self.emit_site_resident(
                    source,
                    target,
                    &[site.a.slot, site.b.slot],
                    Some((id as i64, geom)),
                    |_, _| unreachable!("a site table launches through its pointer table"),
                );
                return;
            }
            // S54 `epilogue`: the per-cell consumers this site's store evaluates
            // (none when a fold is its only reader: `col` and a group exclude).
            let epi = self.epi_kernel(m);
            let id = if col {
                kernel_id_col(site, self.profile, self.contract, true)
            } else {
                kernel_id_with(site, self.profile, self.contract, epi.as_ref())
            } as i64;
            let l = launch_in(site, self.profile, self.contract, self.gpu_arms, self.order);
            let cells = site.rows * site.c;
            let bytes_o = cells * elem_bytes_of(&site.elem);
            let combined = epi.is_none() && !col && !streams;
            // per-site-choice B1/B2: a site with candidates launches its
            // table — not a column-major store (its consumer reads the one
            // layout), a function on streams (its order is the events') or
            // the rules' ordered launch under an L2 window. An epilogue
            // group's kernel is ordered, so are its candidates.
            if !col && !streams {
                let (arms, order) =
                    group_order(epi.as_ref(), site, self.profile, self.gpu_arms, self.order);
                let cands = site_cands(site, self.profile, self.contract, arms, order, false);
                let c0 = cands[0].launch;
                let windowed = epi.is_none()
                    && c0.combine == Combine::None
                    && host_rungs(self.profile, self.contract, arms).l2_window
                    && self.l2_window(m, site, &c0).is_some();
                let table = match windowed {
                    true => None,
                    false => self.make_table(source, target, site, &cands, site, epi.as_ref()),
                };
                if let Some(t) = table {
                    // ADR-0040 D3: every candidate of a fixed face has candidate
                    // 0's `Combine` — the count the interpreter replays.
                    if let (Order::Fixed, Combine::Workspace { s }) = (self.order, c0.combine) {
                        let (func, fold) = site_fold(self.ir, m);
                        self.orders.push(crate::FoldOrder { func, fold, s });
                    }
                    let geom = format!(
                        "i32 {}, i32 {}, i32 {}, i32 {}",
                        c0.gx, c0.gy, c0.block, c0.smem
                    );
                    self.site_table = Some(t);
                    self.emit_site_resident(
                        source,
                        target,
                        &[site.a.slot, site.b.slot],
                        Some((id, geom)),
                        |_, _| unreachable!("a table launches through mapal_gpu_site"),
                    );
                    debug_assert!(self.site_table.is_none(), "the table was launched");
                    return;
                }
            }
            // S63 (ADR-0040 D4) `splitk-atomic` / `streamk`: the output zeroed
            // on stream 0, then one launch whose partials add into it — no
            // workspace, no second kernel. Same exclusions as the workspace
            // split below.
            let atomic = match l.combine {
                Combine::Atomic { s } if combined => {
                    let geom = format!(
                        "i32 {}, i32 {}, i32 {}, i32 {}",
                        l.gx,
                        l.gy * s,
                        l.block,
                        l.smem
                    );
                    Some(("atomic", geom, format!(" ; splitk={s} atomic")))
                }
                Combine::StreamK { ctas, iters, whole } if combined => {
                    let geom = format!(
                        "i32 {}, i32 1, i32 {}, i32 {}",
                        whole + ctas,
                        l.block,
                        l.smem
                    );
                    Some((
                        "streamk",
                        geom,
                        format!(" ; streamk={ctas}x{iters}{}", hybrid_tag(whole)),
                    ))
                }
                _ => None,
            };
            if let Some((part, geom, tag)) = atomic {
                let aid = kernel_id_split(site, self.profile, self.contract, part) as i64;
                // kernel-merging `zero-fold`: an earlier kernel zeroed it.
                let zero = match plan.zeroed.contains_key(m) {
                    true => String::new(),
                    false => format!(
                        "call void @mapal_gpu_zero(i64 {{o}}, i64 {})\n  ",
                        zero_bytes(site, &l)
                    ),
                };
                self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], None, |ops, o| {
                    format!(
                        "{}call void @mapal_gpu_launch(i64 {aid}, i64 {}, i64 {}, i64 {o}, {geom}){tag}",
                        zero.replace("{o}", o),
                        ops[0],
                        ops[1]
                    )
                });
                return;
            }
            // S62 `splitk` (ADR-0040): the split kernel folds `S` partitions
            // into a workspace (`gy·S` blocks), then the reduce kernel adds
            // them into the output in partition order, both on stream 0. The
            // workspace slot is made on the first launch and kept to function
            // exit. ponytail: an epilogue group, a column-major store and a
            // function on streams keep the ordered kernel (and an L2 window
            // is not given to the split launch); split them when a measured
            // program needs it.
            if let Combine::Workspace { s } = l.combine
                && combined
            {
                let id = |r| kernel_id_split(site, self.profile, self.contract, r);
                let (sid, rid) = (id("splitk") as i64, id("reduce") as i64);
                let (rgx, rblock) = cell_grid(cells);
                let bytes = s * bytes_o;
                // ADR-0040 D3: the order the interpreter replays (`fixed`).
                if self.order == Order::Fixed {
                    let (func, fold) = site_fold(self.ir, m);
                    self.orders.push(crate::FoldOrder { func, fold, s });
                }
                let slot = format!("%d{}", self.fresh());
                self.allocas.push_str(&format!(
                    "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                ));
                self.split_slots.push(slot.clone());
                self.dallocs.push((slot.clone(), bytes, 0));
                let ws = self.tmp();
                self.line(format!(
                    "{ws} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {bytes})"
                ));
                let geom = format!(
                    "i32 {}, i32 {}, i32 {}, i32 {}",
                    l.gx,
                    l.gy * s,
                    l.block,
                    l.smem
                );
                self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], None, |ops, o| {
                    format!(
                        "call void @mapal_gpu_launch(i64 {sid}, i64 {}, i64 {}, i64 {ws}, {geom}) ; splitk={s}\n  \
                         call void @mapal_gpu_launch(i64 {rid}, i64 {ws}, i64 0, i64 {o}, i32 {rgx}, i32 1, i32 {rblock}, i32 0)",
                        ops[0], ops[1]
                    )
                });
                return;
            }
            // The ordered kernel (an epilogue group, a column-major store, a
            // function on streams, or no cut): the unsplit launch — `launch_in`
            // re-sizes the tile for a split (`fill-wave`) or a stream-K walk
            // (`k-fill`'s full tile) this kernel does not take. (A stream-K
            // combine counts one split: an ordered group kernel at a `k-fill`
            // site launched the full tile's grid and block over its shrunk
            // kernel — sweep-runtime-2's generated groups, `--cands=1`.)
            let l = if l.combine != Combine::None {
                launch(site, self.profile, self.contract, self.gpu_arms)
            } else {
                l
            };
            // ponytail: a grouped launch (`launchv`) and a function on streams
            // (whose launches order through `launch_on`) carry no L2 window; give
            // them one when a part ships both on one program.
            let window = (epi.is_none()
                && !streams
                && host_rungs(self.profile, self.contract, self.gpu_arms).l2_window)
                .then(|| self.l2_window(m, site, &l))
                .flatten();
            // The column-major tile kernel takes its row block from `ctaid.x`.
            let (gx, gy) = if col && l.tiled.is_some() {
                (l.gy, l.gx)
            } else {
                (l.gx, l.gy)
            };
            let geom = format!("i32 {gx}, i32 {gy}, i32 {}, i32 {}", l.block, l.smem);
            let grouped = (id, geom.clone());
            self.emit_site_resident(source, target, &[site.a.slot, site.b.slot], Some(grouped), |ops, o| {
                let (a, b) = (ops[0].as_str(), ops[1].as_str());
                match window {
                    None => format!(
                        "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, {geom})"
                    ),
                    Some((w, bytes)) => {
                        let p = match w {
                            Window::A => a,
                            Window::B => b,
                            Window::Out => o,
                        };
                        format!(
                            "call void @mapal_gpu_launch_l2(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                             {geom}, i64 {p}, i64 {bytes})"
                        )
                    }
                }
            });
            return;
        }
        let src_ty = self.obj_ty(source);
        // Each operand's bytes in its own storage (a narrow operand's 2-byte
        // elements, ADR-0043); the output's are `elem`'s.
        let (ea, eb) = (site.a.elem_bytes(&site.elem), site.b.elem_bytes(&site.elem));
        let bytes = |ty: &Ty, e: u64| {
            let (_, n) = array_parts(ty);
            n * e
        };
        let a_bytes = bytes(src_ty.component_ty(site.a.slot).expect("tile a array"), ea);
        let b_bytes = bytes(src_ty.component_ty(site.b.slot).expect("tile b array"), eb);
        let o_bytes = bytes(&self.obj_ty(target), elem_bytes_of(&site.elem));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        // long-prefill `uniform-base`: each operand copied from where its
        // captured numbers' terms move it (the copy ends where it did).
        let (mut a_ptr, mut b_ptr, mut a_bytes, mut b_bytes) =
            (a_ptr, b_ptr, a_bytes.to_string(), b_bytes.to_string());
        if let Some(r) = site.radix.as_deref() {
            for (ds, e, p, n) in [
                (&r.ua, ea, &mut a_ptr, &mut a_bytes),
                (&r.ub, eb, &mut b_ptr, &mut b_bytes),
            ] {
                if let Some(off) = self.uniform_bytes(source, ds, e) {
                    let (q, m) = (self.tmp(), self.tmp());
                    self.line(format!(
                        "{q} = getelementptr inbounds i8, ptr {p}, i64 {off}"
                    ));
                    self.line(format!("{m} = sub i64 {n}, {off}"));
                    (*p, *n) = (q, m);
                }
            }
        }
        let o_ptr = self.slot(target).expect("tile output slot");
        // LLVM reads an `i64` literal as signed; the runtime prints the same
        // 64 bits back as hex, so the two units agree on the name.
        let id = kernel_id(site, self.profile, self.contract) as i64;
        let l = launch(site, self.profile, self.contract, self.gpu_arms);
        self.line(format!(
            "call void @mapal_gpu_tile(i64 {id}, ptr {a_ptr}, i64 {a_bytes}, ptr {b_ptr}, \
             i64 {b_bytes}, ptr {o_ptr}, i64 {o_bytes}, i32 {}, i32 {}, i32 {}, i32 {})",
            l.gx, l.gy, l.block, l.smem
        ));
    }

    /// long-prefill `uniform-base`: the bytes an operand's pointer moves by
    /// its captured numbers' terms `ds` ([`mapal_ir::uniform_at`]), each
    /// value loaded from `source` (non-negative where the site runs: the
    /// fact's range) — an `i64` operand, `None` without a term.
    pub(crate) fn uniform_bytes(
        &mut self,
        source: ObjectId,
        ds: &[mapal_ir::Digit],
        elem_bytes: u64,
    ) -> Option<String> {
        let mut acc: Option<String> = None;
        for d in ds {
            let (t, v) = self
                .load_component(source, d.slot)
                .expect("a captured number loads");
            let w = self.tmp();
            self.line(uniform_to_i64(&w, &v, &t));
            let q = match d.stride {
                1 => w,
                s => self.op_i64("udiv", &w, s),
            };
            let r = match d.radix {
                0 => q,
                r => self.op_i64("urem", &q, r),
            };
            let term = self.op_i64("mul", &r, d.coef * elem_bytes);
            acc = Some(match acc {
                None => term,
                Some(a) => self.op_i64("add", &a, &term),
            });
        }
        acc
    }

    pub(super) fn op_i64(&mut self, code: &str, x: &str, y: impl std::fmt::Display) -> String {
        let r = self.tmp();
        self.line(format!("{r} = {code} i64 {x}, {y}"));
        r
    }

    /// Site `m`'s epilogue as its kernel takes it (S54), `None` without one.
    pub(crate) fn epi_kernel(&self, m: MorphismId) -> Option<EpiKernel> {
        let g = self.groups.get(m)?;
        let caps = g.caps();
        let calls = g
            .chain
            .iter()
            .map(|mb| {
                let captures = match self.ir.morphism(mb.m).expect("member").op {
                    Operation::Map { captures, .. } => captures,
                    _ => unreachable!("an epilogue member is a map"),
                };
                let in_ty = self.obj_ty(self.ir.func(mb.body).expect("body").input);
                let arg = crate::ty::lower_body_input_ty(&in_ty, captures)
                    .expect("a member's input lowers");
                let fields: Vec<EpiArg> = mb
                    .srcs
                    .iter()
                    .enumerate()
                    .map(|(slot, s)| match *s {
                        resident::EpiSrc::Value(k) if (slot as u32) < captures => EpiArg::Cell(k),
                        resident::EpiSrc::Value(k) => EpiArg::Value(k),
                        resident::EpiSrc::Gmem(o) => {
                            EpiArg::Cap(caps.iter().position(|&x| x == o).expect("an operand"))
                        }
                        resident::EpiSrc::Index => EpiArg::Index,
                        resident::EpiSrc::Uniform(_) => unreachable!("a member takes no uniform"),
                    })
                    .collect();
                let slots: Vec<u32> = fields
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| matches!(a, EpiArg::Cell(_)))
                    .map(|(s, _)| s as u32)
                    .collect();
                let (ret, _) = array_parts(&self.obj_ty(mb.out));
                EpiCall {
                    callee: slots_variant(&self.fnames[mb.body], &slots),
                    arg,
                    ret,
                    fields,
                    store: mb.store,
                }
            })
            .collect();
        // `tile-post`: the site's body after its fold, slot by slot.
        let post = g.post.as_ref().map(|p| {
            let captures = match self.ir.morphism(m).expect("site").op {
                Operation::Map { captures, .. } => captures,
                _ => unreachable!("a tile site is a map"),
            };
            let in_ty = self.obj_ty(self.ir.func(p.body).expect("body").input);
            let arg = crate::ty::lower_body_input_ty(&in_ty, captures)
                .expect("a site body's input lowers");
            let fields = p
                .srcs
                .iter()
                .map(|s| match *s {
                    Some(resident::EpiSrc::Gmem(o)) => {
                        EpiArg::Cap(caps.iter().position(|&x| x == o).expect("an operand"))
                    }
                    Some(resident::EpiSrc::Index) => EpiArg::Index,
                    Some(resident::EpiSrc::Uniform(j)) => EpiArg::Uniform(j as usize),
                    Some(resident::EpiSrc::Value(_)) => unreachable!("no group value"),
                    None => EpiArg::Null,
                })
                .collect();
            let (ret, _) = array_parts(&self.obj_ty(self.ir.morphism(m).expect("site").target));
            EpiCall {
                callee: post_variant(&self.fnames[p.body]),
                arg,
                ret,
                fields,
                store: false,
            }
        });
        // `tile-post`: each uniform's type, from the site's source product.
        let source = self.ir.morphism(m).expect("site").source;
        let uniforms = g
            .uniforms()
            .into_iter()
            .map(|slot| {
                let t = self
                    .obj_ty(source)
                    .component_ty(slot)
                    .cloned()
                    .expect("a uniform");
                crate::ty::lower_ty(&t).expect("a number lowers")
            })
            .collect();
        Some(EpiKernel {
            store_out: g.store_out,
            caps: caps.len(),
            calls,
            post,
            uniforms,
        })
    }

    /// This function's grouped sites for the device unit, and the bodies
    /// their kernels call: `(callee, body, captured slots read at the cell)`
    /// — an empty slot list is the host's own text.
    #[allow(clippy::type_complexity)]
    pub(crate) fn epi_sites(
        &self,
    ) -> (
        Vec<EpiSite>,
        Vec<(String, FuncId, Vec<u32>, Option<MorphismId>)>,
    ) {
        let mut sites = Vec::new();
        let mut bodies = Vec::new();
        for (m, g) in self.groups.iter() {
            let site = self
                .tile_plan
                .as_ref()
                .and_then(|p| p.sites.get(m))
                .expect("a grouped site is a tile site")
                .clone();
            if !g.terms.is_empty() {
                sites.push(EpiSite {
                    f: self.f,
                    m,
                    site,
                    k: EpiKernel {
                        store_out: false,
                        caps: 0,
                        calls: Vec::new(),
                        post: None,
                        uniforms: Vec::new(),
                    },
                    terms: g.terms.len() as u64 + 1,
                    s: g.split,
                });
                continue;
            }
            let k = self.epi_kernel(m).expect("a group");
            if let (Some(p), Some(c)) = (&g.post, &k.post) {
                bodies.push((c.callee.clone(), p.body, Vec::new(), Some(p.fold)));
            }
            for (mb, c) in g.chain.iter().zip(&k.calls) {
                let slots = c
                    .fields
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| matches!(a, EpiArg::Cell(_)))
                    .map(|(s, _)| s as u32)
                    .collect();
                bodies.push((c.callee.clone(), mb.body, slots, None));
            }
            sites.push(EpiSite {
                f: self.f,
                m,
                site,
                k,
                terms: 0,
                s: 0,
            });
        }
        (sites, bodies)
    }

    /// A retained site's restricted pass (S53, `func/retain.rs`): output row
    /// `row` (an `i64` SSA value) of `site`, as the one-row site `one` whose
    /// `a` is read from the row's first cell and whose output lands at the
    /// row's first cell. Only the row's `a` span and output cross the bus.
    pub(crate) fn emit_tiled_map_gpu_row(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
        one: &TileSite,
        row: &str,
    ) {
        if let Some(plan) = &self.resident {
            // per-site-choice B1/B2: a row with candidates launches its table
            // (not on streams: their order is the events').
            if plan.chain.is_empty() {
                let (arms, order) = (self.gpu_arms, self.order);
                let cands = site_cands(one, self.profile, self.contract, arms, order, true);
                if let Some(t) = self.make_table(source, target, site, &cands, one, None) {
                    self.site_table = Some(t);
                    let done = self.emit_row_resident(source, target, site, row, |_, _, _| {
                        unreachable!("a table launches through mapal_gpu_site")
                    });
                    self.site_table = None;
                    if !done {
                        self.emit_tiled_map_gpu(source, target, site);
                    }
                    return;
                }
            }
            // site-geometry `row-split` (D1): the row's fold through the
            // chooser — zero the row, one launch whose partials add into it
            // (the whole site's `splitk-atomic` / `streamk`, one stream).
            // A workspace split (a part without the atomic one) on a free
            // order only: a fixed face replays one partition count per fold.
            let lc = launch_in(one, self.profile, self.contract, self.gpu_arms, self.order);
            let split = match lc.combine {
                _ if !lc.rungs.row_split || !plan.chain.is_empty() => None,
                Combine::Workspace { .. } if self.order != Order::Free => None,
                Combine::None => None,
                c => Some(c),
            };
            let bytes_o = one.rows * one.c * elem_bytes_of(&one.elem);
            let part = match split {
                Some(Combine::Atomic { .. }) => "atomic",
                Some(Combine::StreamK { .. }) => "streamk",
                _ => "splitk",
            };
            let id = match split {
                Some(_) => kernel_id_split(one, self.profile, self.contract, part),
                None => kernel_id(one, self.profile, self.contract),
            } as i64;
            let ws = match split {
                Some(Combine::Workspace { s }) => {
                    let slot = format!("%d{}", self.fresh());
                    self.allocas.push_str(&format!(
                        "  {slot} = alloca i64\n  store i64 0, ptr {slot}\n"
                    ));
                    self.split_slots.push(slot.clone());
                    self.dallocs.push((slot.clone(), s * bytes_o, 0));
                    let ws = self.tmp();
                    self.line(format!(
                        "{ws} = call i64 @mapal_gpu_alloc(ptr {slot}, i64 {})",
                        s * bytes_o
                    ));
                    ws
                }
                _ => String::new(),
            };
            let rid = kernel_id_split(one, self.profile, self.contract, "reduce") as i64;
            let l = launch(one, self.profile, self.contract, self.gpu_arms);
            let call = |a: &str, b: &str, o: &str| match split {
                Some(Combine::Atomic { s }) => format!(
                    "call void @mapal_gpu_zero(i64 {o}, i64 {bytes_o})\n  \
                     call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 {}, i32 {}, i32 {}) ; splitk={s} atomic",
                    lc.gx,
                    lc.gy * s,
                    lc.block,
                    lc.smem
                ),
                Some(Combine::StreamK { ctas, iters, whole }) => format!(
                    "call void @mapal_gpu_zero(i64 {o}, i64 {})\n  \
                     call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 1, i32 {}, i32 {}) ; streamk={ctas}x{iters}{}",
                    zero_bytes(one, &lc),
                    whole + ctas,
                    lc.block,
                    lc.smem,
                    hybrid_tag(whole)
                ),
                Some(Combine::Workspace { s }) => {
                    let (rgx, rblock) = cell_grid(one.rows * one.c);
                    format!(
                        "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {ws}, \
                         i32 {}, i32 {}, i32 {}, i32 {}) ; splitk={s}\n  \
                         call void @mapal_gpu_launch(i64 {rid}, i64 {ws}, i64 0, i64 {o}, i32 {rgx}, i32 1, i32 {rblock}, i32 0)",
                        lc.gx,
                        lc.gy * s,
                        lc.block,
                        lc.smem
                    )
                }
                _ => format!(
                    "call void @mapal_gpu_launch(i64 {id}, i64 {a}, i64 {b}, i64 {o}, \
                     i32 {}, i32 {}, i32 {}, i32 {})",
                    l.gx, l.gy, l.block, l.smem
                ),
            };
            if !self.emit_row_resident(source, target, site, row, call) {
                self.emit_tiled_map_gpu(source, target, site);
            }
            return;
        }
        let eb = elem_bytes_of(&site.elem);
        let elem_llt = lower_ty(&site.elem).expect("tile element lowers");
        let src_ty = self.obj_ty(source);
        let (_, b_len) = array_parts(src_ty.component_ty(site.b.slot).expect("tile b array"));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        let o_ptr = self.slot(target).expect("tile output slot");
        let a_cells = self.tmp();
        self.line(format!("{a_cells} = mul i64 {row}, {}", site.a.ci));
        let a_off = self.tmp();
        self.line(format!("{a_off} = add i64 {a_cells}, {}", site.a.base));
        let a_row = self.tmp();
        self.line(format!(
            "{a_row} = getelementptr {elem_llt}, ptr {a_ptr}, i64 {a_off}"
        ));
        let o_off = self.tmp();
        self.line(format!("{o_off} = mul i64 {row}, {}", site.c));
        let o_row = self.tmp();
        self.line(format!(
            "{o_row} = getelementptr {elem_llt}, ptr {o_ptr}, i64 {o_off}"
        ));
        let a_bytes = ((one.k - 1) * one.a.ck + 1) * eb;
        let (b_bytes, o_bytes) = (b_len * eb, one.c * eb);
        let id = kernel_id(one, self.profile, self.contract) as i64;
        let l = launch(one, self.profile, self.contract, self.gpu_arms);
        self.line(format!(
            "call void @mapal_gpu_tile(i64 {id}, ptr {a_row}, i64 {a_bytes}, ptr {b_ptr}, \
             i64 {b_bytes}, ptr {o_row}, i64 {o_bytes}, i32 {}, i32 {}, i32 {}, i32 {})",
            l.gx, l.gy, l.block, l.smem
        ));
    }
}
