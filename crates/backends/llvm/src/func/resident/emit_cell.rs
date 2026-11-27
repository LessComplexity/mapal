//! emission of device cells, their kernels and kernel body texts

use super::*;

impl<'a> FnEmit<'a> {
    /// A cell's launch (`devmap`): its operands' device copies (copied in
    /// here only when nothing placed one, as a site does), one thread per
    /// cell, the output kept on the device unless the host reads it whole.
    pub(super) fn emit_cell(&mut self, m: MorphismId, c: &Cell) {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let (source, target) = (morph.source, morph.target);
        let bytes = self.array_bytes(&self.obj_ty(target));
        // full-model `once-range`: the host intent of a range map run once —
        // its interval written over its dead source's buffer, or the whole
        // extent into its own.
        let inplace = self
            .resident
            .as_ref()
            .is_some_and(|p| p.inplace.contains_key(target));
        if inplace {
            self.line(format!("; mapal.range once=inplace bytes={bytes}"));
        } else if c.range.is_none()
            && self.range_write(m).is_some()
            && self.straight_line().contains_key(m)
        {
            self.line(format!("; mapal.range once=whole bytes={bytes}"));
        }
        // `range-cell`: the output is the state's carry buffer (its alias).
        let out = if c.range.is_some() {
            self.dload(target)
        } else {
            self.dalloc(target, bytes)
        };
        let mut ptrs = vec![out];
        let mut temps = Vec::new();
        for (o, slot, uniform) in c.args() {
            if let (true, Some(u)) = (uniform, slot)
                && c.dev.contains(&u)
            {
                ptrs.push(self.dload(o));
                continue;
            }
            if let (true, Some(u)) = (uniform, slot) {
                let (t, v) = self.load_component(source, u).expect("uniform loads");
                let w = self.tmp();
                self.line(gpu::uniform_to_i64(&w, &v, &t));
                ptrs.push(w);
                continue;
            }
            let at = self.resident.as_ref().map_or(o, |p| p.alias(o));
            if self.dvalid.contains_key(at) {
                ptrs.push(self.dload(o));
                continue;
            }
            let ob = self.array_bytes(&self.obj_ty(o));
            let h = self
                .array_operand_ptr(source, slot)
                .expect("cell operand ptr");
            let s = format!("%d{}", self.fresh());
            self.allocas
                .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
            let d = self.tmp();
            self.line(format!(
                "{d} = call i64 @mapal_gpu_alloc(ptr {s}, i64 {ob})"
            ));
            self.line(format!(
                "call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {ob})"
            ));
            temps.push(s);
            ptrs.push(d);
        }
        // caching-memory `retain-cell`: the whole map on the loop's first
        // trip, its delta's cells after (in its own persisting buffer).
        if let Some(r) = c.retain.clone()
            && c.range.is_none()
            && self
                .resident
                .as_ref()
                .is_some_and(|p| !p.is_carry_next(target) && p.alias(target) == target)
        {
            if r.same {
                self.emit_cell_once(m, c, &ptrs);
                self.dvalid.insert(target, ());
                for t in temps {
                    self.line(format!("call void @mapal_gpu_free(ptr {t})"));
                }
                self.cell_out(target, bytes, &ptrs[0]);
                return;
            }
            let start = match r.delta {
                mapal_ir::Delta::Rows { start, .. } | mapal_ir::Delta::Pieces { start, .. } => {
                    Some(start)
                }
                mapal_ir::Delta::Top => None,
            };
            if let Some(start) = start
                && let Some((wllt, t)) = self.load_whole(start.var)
            {
                self.emit_cell_retained(m, c, &r, &ptrs, (&wllt, &t, start));
                self.dvalid.insert(target, ());
                for t in temps {
                    self.line(format!("call void @mapal_gpu_free(ptr {t})"));
                }
                self.cell_out(target, bytes, &ptrs[0]);
                return;
            }
        }
        // `range-cell`: the interval met with `[0, n)` — its start and count
        // by value, one block row per 256 of its cells (at least one block).
        let gx_dyn = c.range.map(|ou| {
            let (lo, hi) = self.range_bounds(&ou, c.n);
            let (span, up, cnt) = (self.tmp(), self.tmp(), self.tmp());
            self.line(format!("{span} = sub i64 {hi}, {lo}"));
            self.line(format!("{up} = add i64 {span}, {}", ou.step - 1));
            self.line(format!("{cnt} = udiv i64 {up}, {}", ou.step));
            // `piece-delta`: each position a piece of `w` cells.
            let cnt = match ou.w {
                0 | 1 => cnt,
                w => {
                    let c = self.tmp();
                    self.line(format!("{c} = mul i64 {cnt}, {w}"));
                    c
                }
            };
            ptrs.push(lo);
            ptrs.push(cnt.clone());
            let (thr, blk, b1, z, one, gx) = (
                self.tmp(),
                self.tmp(),
                self.tmp(),
                self.tmp(),
                self.tmp(),
                self.tmp(),
            );
            self.line(format!("{thr} = mul i64 {cnt}, {}", c.lanes));
            self.line(format!("{blk} = add i64 {thr}, 255"));
            self.line(format!("{b1} = udiv i64 {blk}, 256"));
            self.line(format!("{z} = icmp eq i64 {b1}, 0"));
            self.line(format!("{one} = select i1 {z}, i64 1, i64 {b1}"));
            self.line(format!("{gx} = trunc i64 {one} to i32"));
            gx
        });
        let (gx, block) = gpu::CellGeo {
            range: None,
            piece: 0,
            lanes: c.lanes,
            zero: 0,
            clip: false,
        }
        .grid(c.n);
        let gx = gx_dyn.unwrap_or_else(|| gx.to_string());
        // kernel-merging `zero-fold`: later sites' zero passes as its tail.
        self.zero_args(m, &gx, block, &mut ptrs);
        let args = format!("%dv{}", self.fresh());
        let np = ptrs.len();
        self.allocas
            .push_str(&format!("  {args} = alloca [{np} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        self.line(format!(
            "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 0)",
            c.id as i64
        ));
        // bf16-class `device-trap`: the next checkpoint reads the trap word.
        if c.trap {
            self.line("call void @mapal_gpu_trap_arm()");
        }
        self.dvalid.insert(target, ());
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        self.cell_out(target, bytes, &ptrs[0]);
    }

    /// A cell's output copied out to the host unless the device keeps it.
    pub(super) fn cell_out(&mut self, target: ObjectId, bytes: u64, d: &str) {
        let keep = self
            .resident
            .as_ref()
            .is_some_and(|p| p.keep.contains_key(target) || p.is_carry_next(target));
        if !keep {
            let h = self.slot(target).expect("cell output slot");
            self.line(format!(
                "call void @mapal_gpu_d2h(ptr {h}, i64 {d}, i64 0, i64 {bytes})"
            ));
        }
    }

    /// One `launchv` of kernel `id` over `ptrs` (the output first), `gx`
    /// blocks of `block` threads.
    pub(super) fn launch_cell(&mut self, id: u64, ptrs: &[String], gx: &str, block: u64) {
        let args = format!("%dv{}", self.fresh());
        let np = ptrs.len();
        self.allocas
            .push_str(&format!("  {args} = alloca [{np} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!(
                "{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"
            ));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        self.line(format!(
            "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 0)",
            id as i64
        ));
    }

    /// caching-memory `retain-same`: the loop's first trip (its flag, which
    /// `retain_reset` clears at entry) launches the whole cell; no later trip
    /// launches anything — the output is the same on every trip.
    pub(super) fn emit_cell_once(&mut self, m: MorphismId, c: &Cell, ptrs: &[String]) {
        let target = self.ir.morphism(m).expect("cell").target;
        let flag = self.retain_flag(target);
        let valid = self.tmp();
        self.line(format!("{valid} = load i1, ptr {flag}"));
        let (whole, done) = (self.label(), self.label());
        self.line(format!("br i1 {valid}, label %{done}, label %{whole}"));
        self.label_line(&whole);
        let geo = gpu::CellGeo {
            range: None,
            piece: 0,
            lanes: c.lanes,
            zero: 0,
            clip: false,
        };
        let (gx, block) = geo.grid(c.n);
        self.launch_cell(c.id, ptrs, &gx.to_string(), block);
        self.line(format!("store i1 true, ptr {flag}"));
        self.line(format!("br label %{done}"));
        self.label_line(&done);
    }

    /// caching-memory `retain-cell`: the loop's first trip (the flag
    /// `retain_reset` clears at entry) launches the whole cell; every later
    /// trip launches the delta kernel over the delta's cells met with `[0,
    /// n)` — a row range, or the pieces from the first one inside (each
    /// piece's cells past `n` clipped in the kernel) — nothing when none is.
    pub(super) fn emit_cell_retained(
        &mut self,
        m: MorphismId,
        c: &Cell,
        r: &RetainCell,
        ptrs: &[String],
        (wllt, t, start): (&str, &str, mapal_ir::Affine),
    ) {
        let target = self.ir.morphism(m).expect("cell").target;
        let flag = self.retain_flag(target);
        let valid = self.tmp();
        self.line(format!("{valid} = load i1, ptr {flag}"));
        let (one, whole, go, done) = (self.label(), self.label(), self.label(), self.label());
        self.line(format!("br i1 {valid}, label %{one}, label %{whole}"));
        self.label_line(&whole);
        let geo = gpu::CellGeo {
            range: None,
            piece: 0,
            lanes: c.lanes,
            zero: 0,
            clip: false,
        };
        let (gx, block) = geo.grid(c.n);
        self.launch_cell(c.id, ptrs, &gx.to_string(), block);
        self.line(format!("store i1 true, ptr {flag}"));
        self.line(format!("br label %{done}"));
        self.label_line(&one);
        let s = self.affine_cell(wllt, t, &start);
        let n = c.n;
        let (lo, cnt) = match r.delta {
            mapal_ir::Delta::Pieces { w, step, .. } => {
                // The first piece reaching past 0: j0 = ⌊−(start + w) / step⌋ + 1
                // when start + w ≤ 0, else 0 — a piece that starts below 0 but
                // ends inside is the delta's too (a window's widened rows); the
                // kernel clips its cells below 0 as it clips those past `n`
                // (an unsigned compare).
                let (e, neg, ns, q, q1, j0, js, first) = (
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                );
                self.line(format!("{e} = add i64 {s}, {w}"));
                self.line(format!("{neg} = icmp sle i64 {e}, 0"));
                self.line(format!("{ns} = sub i64 0, {e}"));
                self.line(format!("{q} = udiv i64 {ns}, {step}"));
                self.line(format!("{q1} = add i64 {q}, 1"));
                self.line(format!("{j0} = select i1 {neg}, i64 {q1}, i64 0"));
                self.line(format!("{js} = mul i64 {j0}, {step}"));
                self.line(format!("{first} = add i64 {s}, {js}"));
                // Pieces starting inside: (n − 1 − first) ÷ step + 1.
                let (lt, rest, k, k1, np, cnt) = (
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                );
                self.line(format!("{lt} = icmp slt i64 {first}, {n}"));
                self.line(format!("{rest} = sub i64 {}, {first}", n - 1));
                self.line(format!("{k} = udiv i64 {rest}, {step}"));
                self.line(format!("{k1} = add i64 {k}, 1"));
                self.line(format!("{np} = select i1 {lt}, i64 {k1}, i64 0"));
                self.line(format!("{cnt} = mul i64 {np}, {w}"));
                (first, cnt)
            }
            mapal_ir::Delta::Rows { stride, .. } => {
                let (neg, lo, e, big, hi, d, pos, cnt) = (
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                    self.tmp(),
                );
                self.line(format!("{neg} = icmp slt i64 {s}, 0"));
                self.line(format!("{lo} = select i1 {neg}, i64 0, i64 {s}"));
                self.line(format!("{e} = add i64 {s}, {stride}"));
                self.line(format!("{big} = icmp sgt i64 {e}, {n}"));
                self.line(format!("{hi} = select i1 {big}, i64 {n}, i64 {e}"));
                self.line(format!("{d} = sub i64 {hi}, {lo}"));
                self.line(format!("{pos} = icmp sgt i64 {d}, 0"));
                self.line(format!("{cnt} = select i1 {pos}, i64 {d}, i64 0"));
                (lo, cnt)
            }
            mapal_ir::Delta::Top => unreachable!("a retained cell has a delta"),
        };
        let none = self.tmp();
        self.line(format!("{none} = icmp eq i64 {cnt}, 0"));
        self.line(format!("br i1 {none}, label %{done}, label %{go}"));
        self.label_line(&go);
        let (thr, blk, b1, gx) = (self.tmp(), self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{thr} = mul i64 {cnt}, {}", r.lanes));
        self.line(format!("{blk} = add i64 {thr}, {}", gpu::NAIVE_THREADS - 1));
        self.line(format!("{b1} = udiv i64 {blk}, {}", gpu::NAIVE_THREADS));
        self.line(format!("{gx} = trunc i64 {b1} to i32"));
        let mut args = ptrs.to_vec();
        args.push(lo);
        args.push(cnt);
        self.launch_cell(r.id, &args, &gx, gpu::NAIVE_THREADS);
        self.line(format!("br label %{done}"));
        self.label_line(&done);
    }

    /// The device kernels of this function's cells, each with the functions
    /// it calls (name, text): the body — `cell-law`'s variant of it when a
    /// capture is evaluated — and each evaluated capture's law.
    /// Every row fold this function runs on the device (the device unit's
    /// kernels for the ones `rowfold::rowfold_sites` does not see).
    pub(crate) fn rowfold_kernels(&self) -> Vec<mapal_ir::RowFoldSite> {
        self.resident.as_ref().map_or_else(Vec::new, |p| {
            p.rowfold.values().map(|r| r.site.clone()).collect()
        })
    }

    pub(crate) fn cell_kernels(&self) -> Vec<gpu::CellKernel> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.cells
            .iter()
            .flat_map(|(m, c)| {
                let morph = self.ir.morphism(m).expect("cell morphism");
                let (out, _) = array_parts(&self.obj_ty(morph.target));
                let input = self.obj_ty(self.ir.func(c.body).expect("cell body").input);
                let arg =
                    lower_body_input_ty(&input, c.caps.len() as u32).expect("cell input lowers");
                let elem = c.mapped.map(|o| array_parts(&self.obj_ty(o)).0);
                let vals: Vec<(u32, String)> = c
                    .laws
                    .iter()
                    .map(|&(s, _)| (s, array_parts(&self.obj_ty(c.caps[s as usize])).0))
                    .collect();
                let name = &self.fnames[c.body];
                let name = if vals.is_empty() {
                    name.clone()
                } else {
                    gpu::cell_variant(name)
                };
                // `cell-lanes`: the body's lane variant, its fold split over the warp.
                let lanes = (c.lanes > 1)
                    .then(|| self.ir.cell_plan(self.f).get(m).and_then(|s| s.tree))
                    .flatten()
                    .map(|(g, op, _)| (self.ir.morphism(g).expect("fold").target, op, c.lanes));
                let name = if lanes.is_some() {
                    gpu::lanes_variant(&name)
                } else {
                    name
                };
                // bf16-class `device-trap`: the body's checking variant (one
                // lane: no fold split, so no `fly-unroll` depth either).
                // residency `fly-unroll`: the lane loop's depth from the
                // grid's warps an SM against the part's stream bytes in flight.
                let (name, text) = match c.trap {
                    true => (
                        gpu::trap_variant(&name),
                        self.body_text_trap(c.body, vals.clone()),
                    ),
                    false => {
                        let trips = self.fly_trips(m, c);
                        let name = match trips {
                            crate::profile::NAIVE_UNROLL => name,
                            t => format!("{name}_u{t}"),
                        };
                        let text = self.body_text_lanes(c.body, vals.clone(), lanes, trips);
                        (name, text)
                    }
                };
                let mut fns = vec![(name, text)];
                let caps: Vec<gpu::CapArg> = (0..c.caps.len() as u32)
                    .map(|s| {
                        if c.uniform.contains(&s) {
                            let t = lower_ty(&self.obj_ty(c.caps[s as usize]))
                                .expect("a number uniform lowers");
                            if c.dev.contains(&s) {
                                return gpu::CapArg::Dev(t);
                            }
                            return gpu::CapArg::Uniform(t);
                        }
                        let Some(&(_, law)) = c.laws.iter().find(|&&(x, _)| x == s) else {
                            return gpu::CapArg::Ptr;
                        };
                        fns.push((self.fnames[law].clone(), self.body_text(law, Vec::new())));
                        let t = vals
                            .iter()
                            .find(|&&(x, _)| x == s)
                            .expect("law value")
                            .1
                            .clone();
                        gpu::CapArg::Law(self.fnames[law].clone(), t)
                    })
                    .collect();
                let laws = c.laws.iter().map(|&(_, l)| l);
                for nb in std::iter::once(c.body)
                    .chain(laws)
                    .flat_map(|b| nested_bodies(self.ir, b))
                {
                    fns.push((self.fnames[nb].clone(), self.body_text(nb, Vec::new())));
                }
                // The facts the kernel was built from, for its `; mapal.plan`.
                let work = self.ir.cell_plan(self.f).get(m).map_or(1, |x| x.work);
                let geo = gpu::CellGeo {
                    range: c.range.map(|ou| ou.step),
                    piece: c.range.map_or(0, |ou| ou.w),
                    lanes: c.lanes,
                    zero: plan.zero_by.get(m).map_or(0, Vec::len),
                    clip: false,
                };
                let mut facts = format!("nested={} work={work}", c.nested.len());
                if let Some(ou) = c.range {
                    facts.push_str(&format!(" range={}x{}", ou.step, ou.len.min(c.n)));
                    if ou.w > 1 {
                        facts.push_str(&format!(" piece={}", ou.w));
                    }
                }
                if c.lanes > 1 {
                    facts.push_str(&format!(" lanes={}", c.lanes));
                }
                if !c.dev.is_empty() {
                    facts.push_str(&format!(" dev={}", c.dev.len()));
                }
                let kernel = format!(
                    "; mapal.cell {facts}\n{}",
                    gpu::cell_kernel_text(
                        c.id,
                        &fns[0].0,
                        c.n,
                        &out,
                        &arg,
                        elem.as_deref(),
                        &caps,
                        geo
                    )
                );
                // `trip-batch`: the same cell over a batch of trips' rows, the
                // counter each cell's own trip.
                let batch = plan
                    .batch
                    .iter()
                    .flat_map(|b| &b.writes)
                    .find(|w| w.0 == m)
                    .map(|&(_, id, _, n, u, d, b)| {
                        let mut caps = caps.clone();
                        let gpu::CapArg::Uniform(t) = caps[u as usize].clone() else {
                            unreachable!("the batched counter is a by-value uniform")
                        };
                        caps[u as usize] = gpu::CapArg::Trip(t, d, b);
                        let k = gpu::cell_kernel_text(
                            id,
                            &fns[0].0,
                            c.n,
                            &out,
                            &arg,
                            elem.as_deref(),
                            &caps,
                            gpu::CellGeo {
                                range: Some(1),
                                piece: 0,
                                lanes: 1,
                                zero: 0,
                                clip: false,
                            },
                        );
                        (
                            fns.clone(),
                            format!(
                                "; mapal.cell nested={} work={work} batch={n}\n{k}",
                                c.nested.len()
                            ),
                        )
                    });
                // caching-memory `retain-cell`: the same cell over its delta.
                let retained = c.retain.as_ref().filter(|r| !r.same).map(|r| {
                    let (step, w) = r.geo();
                    // Its own lanes: the body's lane variant where the delta's
                    // cells take a warp each.
                    let mut fns = fns.clone();
                    if r.lanes != c.lanes {
                        let tree = self.ir.cell_plan(self.f).get(m).and_then(|s| s.tree);
                        let lanes = (r.lanes > 1).then_some(tree).flatten().map(|(g, op, _)| {
                            (self.ir.morphism(g).expect("fold").target, op, r.lanes)
                        });
                        let base = if vals.is_empty() {
                            self.fnames[c.body].clone()
                        } else {
                            gpu::cell_variant(&self.fnames[c.body])
                        };
                        let name = if lanes.is_some() {
                            gpu::lanes_variant(&base)
                        } else {
                            base
                        };
                        fns[0] = (name, self.body_text_with(c.body, vals.clone(), lanes));
                    }
                    let k = gpu::cell_kernel_text(
                        r.id,
                        &fns[0].0,
                        c.n,
                        &out,
                        &arg,
                        elem.as_deref(),
                        &caps,
                        gpu::CellGeo {
                            range: Some(step),
                            piece: w,
                            lanes: r.lanes,
                            zero: 0,
                            clip: true,
                        },
                    );
                    (
                        fns,
                        format!(
                            "; mapal.cell nested={} work={work} retain={step}x{w}\n{k}",
                            c.nested.len()
                        ),
                    )
                });
                std::iter::once((fns, kernel))
                    .chain(batch)
                    .chain(retained)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// `f`'s text as the module emits it (a leaf body, so the text is a pure
    /// function of the IR), or `cell-law`'s device variant of it.
    pub(super) fn body_text(&self, f: FuncId, vals: Vec<(u32, String)>) -> String {
        self.body_text_with(f, vals, None)
    }

    /// bf16-class `device-trap`: [`Self::body_text`]'s device variant whose
    /// unproven index reads record the trap into the module's word and read
    /// element 0 (`GuardFlavor::TaskBody`, the map's own topo position).
    pub(super) fn body_text_trap(&self, f: FuncId, vals: Vec<(u32, String)>) -> String {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        fe.cell_vals = vals;
        fe.guard_flavor = super::super::GuardFlavor::TaskBody(0);
        fe.device_trap = true;
        fe.emit()
    }

    /// [`Self::body_text`], or (`cell-lanes`) its lane variant.
    pub(super) fn body_text_with(
        &self,
        f: FuncId,
        vals: Vec<(u32, String)>,
        lanes: Option<(ObjectId, mapal_ir::RowFoldOp, u64)>,
    ) -> String {
        self.body_text_lanes(f, vals, lanes, crate::profile::NAIVE_UNROLL)
    }

    /// residency `fly-unroll`: the trips a warp-a-cell fold of cell `c`
    /// unrolls. A grid of `n · lanes` threads holds `w` warps an SM; each
    /// unrolled trip a warp loads `lanes` elements of the narrowest array its
    /// nested fold walks (the streamed operand). Little's law at the part's
    /// stream bytes in flight (`Gpu::stream_fly_per_sm`) asks `fly / (w ·
    /// lanes · e)` trips in flight — the next power of two, never under
    /// [`crate::profile::NAIVE_UNROLL`], capped at 32 (the registers a lane
    /// holds them in).
    pub(super) fn fly_trips(&self, m: MorphismId, c: &Cell) -> u64 {
        let naive = crate::profile::NAIVE_UNROLL;
        let Some(g) = self.profile.gpu() else {
            return naive;
        };
        if c.lanes <= 1 || !gpu::host_rungs(self.profile, self.contract, self.gpu_arms).fly_unroll {
            return naive;
        }
        let fly = g.stream_fly_per_sm();
        let plan = self.ir.cell_plan(self.f);
        let Some(site) = plan.get(m) else {
            return naive;
        };
        let e = site
            .nested
            .iter()
            .filter_map(|&s| c.caps.get(s as usize))
            .map(|&o| crate::profile::elem_bytes(array_elem(&self.obj_ty(o))))
            .min()
            .unwrap_or(0);
        if fly == 0 || e == 0 {
            return naive;
        }
        let warps = (c.n * c.lanes).div_ceil(g.warp).div_ceil(g.sms).max(1);
        fly.div_ceil(warps * c.lanes * e)
            .next_power_of_two()
            .clamp(naive, 32)
    }

    pub(super) fn body_text_lanes(
        &self,
        f: FuncId,
        vals: Vec<(u32, String)>,
        lanes: Option<(ObjectId, mapal_ir::RowFoldOp, u64)>,
        trips: u64,
    ) -> String {
        let mut fe = FnEmit::new(
            self.ir,
            f,
            self.fnames,
            self.strings,
            self.attrs,
            self.tile_plan.is_some(),
            self.packing,
            self.contract,
            self.order,
            self.kc_nest,
            self.move_panel,
            self.profile,
        );
        fe.set_gpu_arms(self.gpu_arms);
        fe.cell_vals = vals;
        fe.lanes = lanes;
        fe.lane_trips = trips;
        fe.emit()
    }
}
