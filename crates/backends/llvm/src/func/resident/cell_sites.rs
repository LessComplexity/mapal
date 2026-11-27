//! cell candidates: fold candidates, cell sites and rounds, the roofline and the pays rule

use super::*;

impl<'a> FnEmit<'a> {
    /// R2 (ARCHITECTURE §5): the per-cell maps that run on the device. A cell
    /// runs there when every operand is already in device memory — a device
    /// site's or an earlier cell's output, a carried state, a tile operand
    /// (copied in for its site anyway), or closed-form (`devgen` builds it) —
    /// and either some operand lives only there or no host reader needs the
    /// output whole: a whole-host reader forces a copy-out, which costs no
    /// more than copying the operands out only when one of them is device-only.
    ///
    /// Cells and carries decide each other (a carried view is an operand; a
    /// cell's output can be the next state), so both are the greatest fixed
    /// point from "every candidate": each round drops the cells whose operands
    /// or readers the round before no longer supports. Nothing is added, so it
    /// ends; with `devmap` off there is no candidate and the carries are S53's.
    /// S65 rule R for row folds: a fold over an array complete on the device —
    /// any device output (a cell kernel's, another fold's), not only a tile
    /// site's (`rowfold::rowfold_sites`) — runs there. Every such fold is a
    /// candidate site; [`Self::cell_sites`]' fixed point drops the ones whose
    /// operand is not device-produced (the byte rule: a fold over a host array
    /// stays on the host).
    pub(super) fn fold_candidates(
        &self,
        sites: &mut SecondaryMap<MorphismId, DevSite>,
    ) -> Vec<MorphismId> {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        if !rungs.rowfold
            || !rungs.by_ref
            || (sites.is_empty() && !rungs.scalar_fold)
            || (!rungs.retained_fold
                && !retain::retained_sites(self.ir, self.f, self.profile, rungs).is_empty())
        {
            return Vec::new();
        }
        let mut out = Vec::new();
        // `scalar-fold`: a whole fold is a row fold of one row, a candidate
        // over a cell's output too (no tile site needed).
        let whole = if rungs.scalar_fold {
            self.ir.whole_fold(self.f)
        } else {
            SecondaryMap::new()
        };
        let rows = if sites.is_empty() {
            SecondaryMap::new()
        } else {
            self.ir.row_fold(self.f)
        };
        for (m, site) in rows.iter().chain(whole.iter()) {
            let window = site.w > 1 || site.a.ksplit.is_some();
            if sites.contains_key(m)
                || !is_float(&site.elem)
                || site.k == 0
                || (window && !rungs.window)
            {
                continue;
            }
            sites.insert(
                m,
                DevSite::RowFold(super::super::rowfold::RowFold {
                    site: site.clone(),
                    colmajor: None,
                }),
            );
            out.push(m);
        }
        out
    }

    pub(super) fn cell_sites(
        &self,
        sites_mut: &mut SecondaryMap<MorphismId, DevSite>,
        folds: &[MorphismId],
        devgen: bool,
    ) -> (SecondaryMap<MorphismId, Cell>, Vec<Carry>) {
        let mut cells = SecondaryMap::new();
        // A Map / Fold body runs once per element of its site: a launch there
        // would be one per element.
        let body = self.ir.morphisms().any(|(_, x)| {
            matches!(x.op, Operation::Map { body, .. } | Operation::Fold { body, .. } if body == self.f)
        });
        let straight = self.straight_line();
        let trip = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).trip_cells;
        let once = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).once_range;
        if !body && gpu::host_rungs(self.profile, self.contract, self.gpu_arms).devmap {
            // bf16-class `device-trap`: the maps whose only trap is an
            // unproven index read are candidates too (checked in the kernel).
            let traps = match self.gpu_arms.off.device_trap {
                true => SecondaryMap::new(),
                false => self.ir.trap_cell_plan(self.f),
            };
            for (m, site) in self.ir.cell_plan(self.f).iter().chain(traps.iter()) {
                // An epilogue member (R1) runs in its site's store already.
                // placement `trip`: a cell a loop runs every trip is a
                // candidate whatever its size — [`Self::cell_round`] keeps it
                // only where it moves no bytes (its operands on the device,
                // its output not needed whole on the host), so a host
                // placement is what would cross the bus each trip. Likewise
                // (`once-range`) a range map run once: it writes its interval
                // where its array is, so its cost is the bytes it would move
                // (`once-bytes`), not its extent's roofline.
                let once_range = once && straight.contains_key(m) && self.range_write(m).is_some();
                if !sites_mut.contains_key(m)
                    && self.member_src(m, 0).is_none()
                    && (self.cell_pays(m, site)
                        || (trip && !straight.contains_key(m))
                        || once_range)
                    && let Some(c) = self.cell_of(m, site)
                {
                    cells.insert(m, c);
                }
            }
            // `cand-made`: S62's "an operand device-produced" over the
            // candidates themselves — a map an operand of which a candidate
            // cell or device site makes is a candidate too (the fixed point
            // below drops it where its operands or readers do not support
            // it), so a chain the roofline prices only cell by cell (a
            // prompt's gather, its norm's row fold and scale) is decided
            // whole, not by its first member alone.
            if gpu::host_rungs(self.profile, self.contract, self.gpu_arms).cand_made {
                let made = |o: ObjectId, cells: &SecondaryMap<MorphismId, Cell>| {
                    self.ir
                        .in_edges(o)
                        .first()
                        .is_some_and(|&p| cells.contains_key(p) || sites_mut.contains_key(p))
                };
                loop {
                    let mut add = Vec::new();
                    for (m, site) in self.ir.cell_plan(self.f).iter().chain(traps.iter()) {
                        if cells.contains_key(m)
                            || sites_mut.contains_key(m)
                            || self.member_src(m, 0).is_some()
                        {
                            continue;
                        }
                        if let Some(c) = self.cell_of(m, site)
                            && c.operands().iter().any(|&(o, _)| made(o, &cells))
                        {
                            add.push((m, c));
                        }
                    }
                    if add.is_empty() {
                        break;
                    }
                    for (m, c) in add {
                        cells.insert(m, c);
                    }
                }
            }
        }
        loop {
            let (drop, unfed, carries) = self.cell_round(sites_mut, folds, &cells, devgen);
            // `MAPAL_DEBUG_PLACE`: each round's dropped cells (a diagnostic
            // for "why does this map run on the host").
            if std::env::var_os("MAPAL_DEBUG_PLACE").is_some() {
                let name = |m: MorphismId| {
                    let t = self.ir.morphism(m).expect("cell").target;
                    let o = self.ir.object(t).expect("obj");
                    format!("{}:{:?}", o.name.clone().unwrap_or_default(), t)
                };
                eprintln!(
                    "place round: {} cells, {} carries; drop {:?}; unfed {:?}",
                    cells.len(),
                    carries.len(),
                    drop.iter().map(|&m| name(m)).collect::<Vec<_>>(),
                    unfed.iter().map(|&m| name(m)).collect::<Vec<_>>()
                );
            }
            if drop.is_empty() && unfed.is_empty() {
                return (cells, carries);
            }
            for m in drop {
                cells.remove(m);
            }
            for m in unfed {
                sites_mut.remove(m);
            }
        }
    }

    /// One round of [`Self::cell_sites`]' fixed point: the cells and the
    /// candidate folds the round before no longer supports.
    pub(super) fn cell_round(
        &self,
        sites: &SecondaryMap<MorphismId, DevSite>,
        folds: &[MorphismId],
        cells: &SecondaryMap<MorphismId, Cell>,
        devgen: bool,
    ) -> (Vec<MorphismId>, Vec<MorphismId>, Vec<Carry>) {
        {
            let mut feeders: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
            for (m, site) in sites.iter() {
                let src = self.ir.morphism(m).expect("site").source;
                for slot in site.reads() {
                    if let Some(f) = self.pair_source(src, slot) {
                        feeders.insert(f, ());
                    }
                }
            }
            let carries = self.carries(sites, cells);
            let carried = |o: ObjectId| {
                carries
                    .iter()
                    .any(|c| c.views.contains(&o) || c.exit == Some(o))
            };
            let device_only = |o| carried(o) || self.produced_on_device(sites, &cells, o);
            // ponytail: an `iota` operand of a nest cell (non-empty `nested`,
            // so only under `nest-cell`; plain cells keep the rule above) is
            // uploaded once at its producer (`n·4` bytes, outside the loop
            // where the rewriter hoists it); a device iota kernel if one ever
            // sits in a loop cone. Not narrowed to `nested` slots: a fold in a
            // guard arm walks its range through the arm's input, which
            // `nest_reads` does not attribute (causal's `s` over `krange`).
            let iota = |o: ObjectId| {
                self.ir
                    .in_edges(o)
                    .iter()
                    .any(|&p| self.ir.morphism(p).is_some_and(|x| x.op == Operation::Iota))
            };
            // full-model `once-range`: a `+0.0` fill is zeroed on the device
            // where a device cell reads it (its upload is a memset): no host
            // bytes cross for it.
            let zeroed = gpu::host_rungs(self.profile, self.contract, self.gpu_arms).once_range;
            let there = |o| {
                device_only(o)
                    || feeders.contains_key(o)
                    || (zeroed && self.zero_fill(o))
                    || (devgen
                        && self
                            .ir
                            .in_edges(o)
                            .iter()
                            .any(|&p| self.gen_law(p).is_some()))
            };
            // placement `trip`: an operand made once (outside every loop and
            // guard arm, or a parameter) is uploaded at its producer once
            // (F1/F2) — for a cell a loop runs every trip it is there.
            let straight = self.straight_line();
            let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
            let made_once = |o: ObjectId| {
                rungs.trip_cells
                    && self
                        .ir
                        .in_edges(o)
                        .first()
                        .is_none_or(|&p| straight.contains_key(p))
            };
            let drop: Vec<MorphismId> = cells
                .iter()
                .filter(|(m, c)| {
                    let ops = c.operands();
                    let per_trip = !straight.contains_key(*m);
                    // placement `bytes-place`: a nest cell that finishes
                    // sooner on the device by its own roofline (a 1-row
                    // projection) is a tile site's kin — like one, it copies
                    // its host operands in at their producers and its output
                    // out where the host reads it.
                    let alone = rungs.bytes_place
                        && !c.nested.is_empty()
                        && self.ir.cell_plan(self.f).get(*m).is_some_and(|site| {
                            let (flops, bytes) = self.cell_roofline(*m, site);
                            self.profile.device_pays(flops, bytes)
                        });
                    let there = |o: ObjectId| there(o) || alone || (per_trip && made_once(o));
                    let t = self.ir.morphism(*m).expect("cell").target;
                    // placement `carry`: a carry's next state forwards its
                    // pointer into the back edge (no host read there).
                    let fwd: Vec<MorphismId> = carries
                        .iter()
                        .filter(|k| rungs.carry_fwd && k.next == t)
                        .filter_map(|k| {
                            let lp = self.ir.loop_plan(self.f, k.merge)?;
                            let state = self.pair_source(lp.back_route, 0)?;
                            Some(self.ir.out_edges(t).iter().copied().filter(move |&e| {
                                self.ir.morphism(e).is_some_and(|x| x.target == state)
                            }))
                        })
                        .flatten()
                        .collect();
                    let host = || self.uses(sites, cells, t, &fwd).contains(&Use::Host);
                    let walked = |o| !c.nested.is_empty() && iota(o);
                    // `range-cell`: written in place, so only into an
                    // in-place carry's buffer — or (`once-range`), run once,
                    // into its dead source's.
                    let homeless = c.range.is_some()
                        && per_trip
                        && !carries
                            .iter()
                            .any(|k| k.inplace && (k.next == t || k.chain.contains(&t)));
                    let held = ops.iter().all(|&(o, _)| there(o) || walked(o))
                        && (alone || ops.iter().any(|&(o, _)| device_only(o)) || !host());
                    // placement `bytes`: a cell a loop runs every trip runs
                    // where it moves fewer bytes per trip. On the device it
                    // uploads the operands the device holds no copy of and
                    // copies its output out when the host reads it whole; on
                    // the host it copies out the operands only the device
                    // holds and uploads its output when a device kernel reads
                    // it. A tie goes to the device when a device kernel waits
                    // for the output (the data stays where its readers are).
                    let moved = || {
                        let bytes = |o: ObjectId| self.array_bytes(&self.obj_ty(o));
                        // residency `shared-upload`: an operand another
                        // device reader loads is uploaded for it anyway (the
                        // residual stream beside a norm): this cell adds no
                        // bytes by reading it there too.
                        let shared = |o: ObjectId| {
                            self.uses(sites, cells, o, &[])
                                .iter()
                                .filter(|u| matches!(u, Use::Dev))
                                .count()
                                > 1
                        };
                        let up: u64 = ops
                            .iter()
                            .filter(|&&(o, _)| !(there(o) || walked(o) || shared(o)))
                            .map(|&(o, _)| bytes(o))
                            .sum();
                        let down: u64 = ops
                            .iter()
                            .filter(|&&(o, _)| device_only(o))
                            .map(|&(o, _)| bytes(o))
                            .sum();
                        let dev_reader = self.uses(sites, cells, t, &[]).contains(&Use::Dev);
                        let (out_down, out_up) = (
                            if host() { bytes(t) } else { 0 },
                            if dev_reader { bytes(t) } else { 0 },
                        );
                        down + out_up > 0 && up + out_down <= down + out_up
                    };
                    // full-model `once-bytes`: a cell outside every loop
                    // moves its bytes once — the same rule, once.
                    let by_bytes = rungs.bytes_place && (per_trip || rungs.once_bytes);
                    let out = homeless || !(held || (by_bytes && moved()));
                    if out && std::env::var_os("MAPAL_DEBUG_PLACE").is_some() {
                        let nm = |o: ObjectId| {
                            self.ir.object(o).and_then(|x| x.name.clone()).unwrap_or_default()
                        };
                        eprintln!(
                            "  drop {}: homeless={homeless} held={held} per_trip={per_trip} alone={alone} host={} ops={:?}",
                            nm(t),
                            host(),
                            ops.iter()
                                .map(|&(o, _)| format!("{}:there={}:dev={}", nm(o), there(o), device_only(o)))
                                .collect::<Vec<_>>()
                        );
                    }
                    out
                })
                .map(|(m, _)| m)
                .collect();
            let unfed: Vec<MorphismId> = folds
                .iter()
                .copied()
                .filter(|&m| sites.contains_key(m))
                .filter(|&m| {
                    let fold = self.ir.morphism(m).expect("fold");
                    let a = sites[m].reads()[0];
                    // `scalar-fold`: a whole fold rides only on an array the
                    // device holds for its own sake — not a closed-form cell
                    // (no operand), which is there only because its readers
                    // are (that support is circular: the host builds it).
                    let whole = matches!(fold.op, Operation::Fold { .. });
                    let closed = |o: ObjectId| {
                        self.ir
                            .in_edges(o)
                            .first()
                            .and_then(|&p| cells.get(p))
                            .is_some_and(|c| c.operands().is_empty())
                    };
                    !self
                        .pair_source(fold.source, a)
                        .is_some_and(|o| device_only(o) && !(whole && closed(o)))
                })
                .collect();
            (drop, unfed, carries)
        }
    }

    /// One thread per cell: `work` flops each (a nested fold's `k`), every
    /// read and the store.
    pub(super) fn cell_roofline(&self, m: MorphismId, site: &CellSite) -> (u64, u64) {
        let t = self.ir.morphism(m).expect("cell").target;
        let w = crate::profile::elem_bytes(array_elem(&self.obj_ty(t)));
        let bytes = (site.reads.len() as u64 + 1)
            .saturating_mul(site.extent)
            .saturating_mul(w);
        (site.extent.saturating_mul(site.work), bytes)
    }

    /// S62 `place` for a per-cell map: on the device when an operand is
    /// device-produced, or when one cell per thread (`extent` flops, every
    /// read and the store) finishes there first — [`gpu::placement`]'s rule.
    ///
    /// A cell that reads no array is a closed form of its index and uniforms:
    /// `devgen`'s rule decides it (built where every reader is), so it is a
    /// candidate and [`Self::cell_round`] keeps it when no host reads it whole.
    pub(super) fn cell_pays(&self, m: MorphismId, site: &CellSite) -> bool {
        let t = self.ir.morphism(m).expect("cell").target;
        let (flops, bytes) = self.cell_roofline(m, site);
        (site.reads.is_empty() && !site.uniform.is_empty())
            || gpu::device_made(
                self.ir,
                &self.device,
                t,
                gpu::host_rungs(self.profile, self.contract, self.gpu_arms).carry_made,
            )
            || self.profile.device_pays(flops, bytes)
    }
}
