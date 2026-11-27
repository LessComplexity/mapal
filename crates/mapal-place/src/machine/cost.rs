//! `Machine::prim` / `cost` / `near`: a primitive's costs at a bound.

use super::*;

impl Machine {
    /// The measured row of primitive `of`, if any.
    pub fn prim(&self, of: PrimOf) -> Option<&Prim> {
        self.prims.iter().find(|p| p.of == of)
    }

    /// **A primitive's costs at bound `b`** (ns; a clock row converted at the
    /// sustained clock): its measured row, else a neighbouring measured row
    /// of its kind borrowed ([`Machine::near`]: marked, naming the row), else
    /// derived from the machine's other facts and marked — a mover from the
    /// place's unit reach (rate) and loaded latency, a store or a panel from
    /// the memory's device reach over the units, the scratchpad from its
    /// banking, reductions from the atomics, barriers and launches from
    /// dispatch.
    pub fn cost(&self, of: PrimOf, b: Bound) -> Cost {
        let ghz = self.ghz(b).max(1e-9);
        let units = self.units().max(1) as f64;
        // A row's costs, its rate scaled by `rf` and its latency by `lf`
        // (1: the row itself).
        let row_at = |p: &Prim, rf: f64, lf: f64| {
            let (r, l, c) = (p.rate.rate(b), p.latency.cost(b), p.conc.cost(b));
            let (r, l) = match p.per_clk {
                true => (r * ghz * rf, l / ghz * lf),
                false => (r * rf, l * lf),
            };
            Cost {
                rate: r,
                latency: l,
                loaded: l.max(if r > 0.0 { c / r } else { 0.0 }),
                marked: p.rate.marked() || p.latency.marked() || p.conc.marked(),
                borrowed: None,
            }
        };
        let row = |p: &Prim| row_at(p, 1.0, 1.0);
        if let Some(p) = self.prim(of) {
            return row(p);
        }
        // Borrowed: the nearest measured row of the same kind, scaled to
        // this one by the bound that holds whichever way the hardware goes —
        // marked, naming the row.
        if let Some((p, rf, lf)) = self.near(of) {
            return Cost {
                marked: true,
                borrowed: Some(p.name),
                ..row_at(p, rf, lf)
            };
        }
        // Derived from the other facts: marked.
        let reach = |p: PlaceId, k: ExecKind| self.reach(p, k);
        let derived = |rate: f64, latency: f64| Cost {
            rate,
            latency,
            loaded: latency,
            marked: true,
            borrowed: None,
        };
        let device_mem = |p: PlaceId| {
            reach(p, ExecKind::Device)
                .map(|r| (r.total_bytes_per_ns.rate(b) / units, r.latency_ns.cost(b)))
        };
        match of {
            PrimOf::Op(o) => {
                let op = &self.ops[o];
                Cost {
                    rate: op.rate.rate(b) * ghz,
                    latency: op.latency.cost(b) / ghz,
                    loaded: op.latency.cost(b) / ghz,
                    marked: op.rate.marked() || op.latency.marked(),
                    borrowed: None,
                }
            }
            PrimOf::Load { from, .. } | PrimOf::Copy { from, .. } | PrimOf::Bulk { from, .. } => {
                match (reach(from, ExecKind::Unit), device_mem(from)) {
                    (Some(r), _) => derived(r.bytes_per_ns.rate(b), r.latency_ns.cost(b)),
                    (None, Some((r, l))) => derived(r, l),
                    _ => derived(f64::INFINITY, 0.0),
                }
            }
            PrimOf::Panel { from, run, .. } => {
                // Unmeasured: the copy from the place (the memory's pages
                // unpriced), its rate's interval opened below for runs
                // shorter than a line.
                let c = self.cost(PrimOf::Copy { from, bytes: 16 }, b);
                let short = (run as f64 / 128.0).clamp(0.25, 1.0);
                let rate = match b {
                    Bound::Slow => c.rate * short,
                    _ => c.rate,
                };
                Cost {
                    rate,
                    marked: true,
                    ..c
                }
            }
            PrimOf::Store { to, .. } | PrimOf::Staged { to } => match device_mem(to) {
                Some((r, l)) => derived(r, l),
                None => derived(f64::INFINITY, 0.0),
            },
            PrimOf::Scratch { .. } => {
                let bw = self
                    .unit_scratch()
                    .and_then(|p| self.places[p].banking)
                    .map_or(128.0, |k| (k.banks * k.width) as f64);
                derived(bw * ghz, 30.0 / ghz)
            }
            PrimOf::Reduce {
                sharing, sectors, ..
            } => {
                let a = self.dispatch.atomics.first();
                let r = a.map_or(0.0, |a| a.ops_per_ns.rate(b)) / units;
                let share = match sharing {
                    0 => 32.0,
                    s => s as f64,
                };
                Cost {
                    rate: r / share.sqrt() * (4.0 / sectors.max(4) as f64),
                    latency: self
                        .device_cache()
                        .and_then(|c| reach(c, ExecKind::Device))
                        .map_or(0.0, |r| r.latency_ns.cost(b)),
                    loaded: 0.0,
                    marked: true,
                    borrowed: None,
                }
            }
            PrimOf::Barrier { .. } | PrimOf::Phase => {
                let l = self.dispatch.barrier_clk.cost(b) / ghz;
                Cost {
                    rate: 1.0 / l.max(1e-9),
                    latency: l,
                    loaded: l,
                    marked: self.dispatch.barrier_clk.marked() || matches!(of, PrimOf::Phase),
                    borrowed: None,
                }
            }
            PrimOf::GridSync => {
                let g = self.dispatch.grid_sync_ns.unwrap_or(Fact::nominal(
                    1000.0,
                    500.0,
                    3000.0,
                    "grid sync unmeasured",
                ));
                Cost {
                    rate: 1.0 / g.cost(b).max(1e-9),
                    latency: g.cost(b),
                    loaded: g.cost(b),
                    marked: g.marked(),
                    borrowed: None,
                }
            }
            PrimOf::Issue => Cost {
                rate: 1.0 / self.dispatch.issue_ns.cost(b).max(1e-9),
                latency: self.dispatch.issue_ns.cost(b),
                loaded: self.dispatch.issue_ns.cost(b),
                marked: self.dispatch.issue_ns.marked(),
                borrowed: None,
            },
            PrimOf::Gap => Cost {
                rate: 1.0 / self.dispatch.gap_ns.cost(b).max(1e-9),
                latency: self.dispatch.gap_ns.cost(b),
                loaded: self.dispatch.gap_ns.cost(b),
                marked: self.dispatch.gap_ns.marked(),
                borrowed: None,
            },
        }
    }

    /// **The measured row primitive `of` borrows** and the factors its rate
    /// and latency take: a row on the side that bounds this one where one is
    /// measured (a narrower copy, load or store, a smaller box, a shorter
    /// run, more rows, fewer bytes in flight, more lanes at a barrier, more
    /// sectors a reduction — each at most as fast), as it stands; else the
    /// nearest on the other side scaled to this one by the bound its
    /// mechanism keeps (a copy's, a box's, a store's or a run's rate by its
    /// bytes against the row's: a fixed cost an instruction; a stream's rate
    /// by its bytes in flight: Little's law — more rows than any measured
    /// take the most, unscaled; a reduction's by the row's
    /// sectors against its own: a sector-throughput bound; a barrier's
    /// latency by its lanes against the row's: a cost a warp arriving).
    /// `None` where no row of the kind is measured.
    pub fn near(&self, of: PrimOf) -> Option<(&Prim, f64, f64)> {
        // The row of `rows` (with its key) at most `want` and largest, else
        // the smallest — and whether it is the latter.
        fn below<'a>(rows: &[(&'a Prim, u64)], want: u64) -> Option<(&'a Prim, u64, bool)> {
            rows.iter()
                .filter(|r| r.1 <= want)
                .max_by_key(|r| r.1)
                .map(|r| (r.0, r.1, false))
                .or_else(|| rows.iter().min_by_key(|r| r.1).map(|r| (r.0, r.1, true)))
        }
        // The row at least `want` and smallest, else the largest.
        fn above<'a>(rows: &[(&'a Prim, u64)], want: u64) -> Option<(&'a Prim, u64, bool)> {
            rows.iter()
                .filter(|r| r.1 >= want)
                .min_by_key(|r| r.1)
                .map(|r| (r.0, r.1, false))
                .or_else(|| rows.iter().max_by_key(|r| r.1).map(|r| (r.0, r.1, true)))
        }
        fn ratio(a: u64, b: u64) -> f64 {
            a.max(1) as f64 / b.max(1) as f64
        }
        // A width-keyed kind: the largest width at most this one, else the
        // smallest, its rate scaled down by the widths.
        fn by_width(rows: Vec<(&Prim, u64)>, want: u64) -> Option<(&Prim, f64, f64)> {
            below(&rows, want).map(|(p, w, out)| (p, if out { ratio(want, w) } else { 1.0 }, 1.0))
        }
        match of {
            PrimOf::Panel {
                from,
                run,
                rows,
                fly,
            } => {
                let panels: Vec<(&Prim, u64, u64, u64)> = self
                    .prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Panel {
                            from: f,
                            run: r,
                            rows: w,
                            fly: y,
                        } if f == from => Some((p, r, w, y)),
                        _ => None,
                    })
                    .collect();
                let runs: Vec<_> = panels.iter().map(|x| (x.0, x.1)).collect();
                let (_, run_, run_out) = below(&runs, run)?;
                let at_run: Vec<_> = panels.iter().filter(|x| x.1 == run_).collect();
                let rows_of: Vec<_> = at_run.iter().map(|x| (x.0, x.2)).collect();
                let (_, rows_, _) = above(&rows_of, rows)?;
                let flies: Vec<_> = at_run
                    .iter()
                    .filter(|x| x.2 == rows_)
                    .map(|x| (x.0, x.3))
                    .collect();
                let (p, fly_, fly_out) = below(&flies, fly)?;
                let rf = if run_out { ratio(run, run_) } else { 1.0 }
                    * if fly_out { ratio(fly, fly_) } else { 1.0 };
                Some((p, rf, 1.0))
            }
            PrimOf::Copy { from, bytes } => by_width(
                self.prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Copy { from: f, bytes: w } if f == from => Some((p, w)),
                        _ => None,
                    })
                    .collect(),
                bytes,
            ),
            PrimOf::Load { from, bytes } => by_width(
                self.prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Load { from: f, bytes: w } if f == from => Some((p, w)),
                        _ => None,
                    })
                    .collect(),
                bytes,
            ),
            PrimOf::Bulk { from, bytes } => by_width(
                self.prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Bulk { from: f, bytes: w } if f == from => Some((p, w)),
                        _ => None,
                    })
                    .collect(),
                bytes,
            ),
            PrimOf::Store { to, bytes } => by_width(
                self.prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Store { to: t, bytes: w } if t == to => Some((p, w)),
                        _ => None,
                    })
                    .collect(),
                bytes,
            ),
            PrimOf::Barrier { lanes } => {
                let rows: Vec<_> = self
                    .prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Barrier { lanes: l } => Some((p, l)),
                        _ => None,
                    })
                    .collect();
                let (p, l, out) = above(&rows, lanes)?;
                let lf = if out { ratio(lanes, l) } else { 1.0 };
                Some((p, 1.0 / lf, lf))
            }
            PrimOf::Reduce {
                at,
                sharing,
                sectors,
            } => {
                let rows: Vec<_> = self
                    .prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Reduce {
                            at: a,
                            sharing: h,
                            sectors: x,
                        } if a == at && h == sharing => Some((p, x)),
                        _ => None,
                    })
                    .collect();
                let (p, x, out) = above(&rows, sectors)?;
                Some((p, if out { ratio(x, sectors) } else { 1.0 }, 1.0))
            }
            _ => None,
        }
    }
}
