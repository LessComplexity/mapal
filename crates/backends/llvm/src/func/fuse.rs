//! **generic-fusion `fuse`: one footprint rule for device groups** (design §2).
//!
//! A consumer joins its producer's group when, for every work unit of the
//! group — one row `i` of its `rows` — the producer cells it reads lie in that
//! unit ([`mapal_ir::CategoryIr::footprint_plan`]'s [`Reach`], checked with
//! [`Reach::rect`]), or are a halo the group recomputes (a classifiable leaf
//! whose own reads are all complete in `Gmem`). A fold is never recomputed: its
//! readers must read row `i` only. A member no outside reader needs
//! (`read_outside`, `last_use_plan`'s escapes, a carried next state) is never
//! stored: a per-row value stays in a register, a cell is evaluated at its
//! read. The budget is the profile's (registers, launch pointers) and, where
//! the part has measured costs, the roofline of the fused launch against the
//! unfused ones.
//!
//! **Placement runs first.** Every member is a unit the residency plan already
//! put on the device — a `devmap` cell, a `rowfold` fold — so a group moves no
//! data: it removes launches, stores and loads. The tile root (a cell reading
//! a tile site's output at its own cell, `resident.rs:epilogues`) is the same
//! rule with the tile's block as the unit, and the OFF arm
//! (`--gpu-off=fuse`): with the rung off only tile roots group.
//!
//! **Realization.** One thread per unit row, members in topo order as
//! phases: a fold walks its row in order (the oracle's chain, ADR-0028 D4), a
//! per-row cell is one call, a stored cell of `w > 1` cells per row a counted
//! loop of calls. Every body is the host's own text (or its one-cell-buffer
//! variant), so each cell is the host's bits; LLVM inlines the bodies and
//! CSEs a cell read twice at one address.

use super::resident::{Cell, ResidentPlan, array_len};
use super::*;
use mapal_ir::{Reach, RowFoldOp, RowFoldSite};

/// Where one input of a unit member comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum USrc {
    /// Member `k`'s value at the address `reach` gives the reader's cell.
    /// `row`: every such address lies in the reader's unit row (a per-row
    /// member's register serves it); else a halo the group recomputes.
    Member { k: usize, reach: Reach, row: bool },
    /// The unit's `Gmem` operand `g`, complete before the launch.
    Gmem(usize),
    /// The cell index itself (an `iota`'s element).
    Index,
    /// A closed-form capture evaluated at the reader's cell (`cell-law`).
    Law(FuncId),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UKind {
    /// A row fold; `colmajor`: its producing tile stores column-major for it.
    Fold {
        site: RowFoldSite,
        colmajor: Option<MorphismId>,
    },
    Cell {
        body: FuncId,
        captures: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UMember {
    pub m: MorphismId,
    pub out: ObjectId,
    pub kind: UKind,
    /// Cells of this member per unit row: `extent / rows`.
    pub w: u64,
    /// A reader outside the group exists: the value is stored.
    pub store: bool,
    /// One per body input slot (a cell), or the fold's one read.
    pub srcs: Vec<USrc>,
}

impl UMember {
    /// One value per unit row (a fold, or a cell of `rows` cells).
    fn per_row(&self) -> bool {
        self.w == 1
    }
}

/// A unit group: one launch, `rows` threads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Unit {
    pub id: u64,
    pub rows: u64,
    pub members: Vec<UMember>,
    /// Operand pointers, in first-use order.
    pub gmem: Vec<ObjectId>,
}

/// A unit's kernel for the device module: the function, its members (whose
/// own kernels are not emitted), the bodies it calls and the kernel text.
pub(crate) struct UnitKernel {
    pub f: FuncId,
    pub members: Vec<MorphismId>,
    pub fns: Vec<(String, String)>,
    pub text: String,
}

/// A placed device unit a group may take.
#[derive(Clone)]
enum Cand {
    Fold(super::rowfold::RowFold),
    Cell(Cell),
}

/// The rows a unit kernel's grid covers: one thread per row; with a fold the
/// rows spread over every SM (`gpu::rowfold_grid`'s rule), else 256 a block
/// (`gpu::cell_grid`).
pub(crate) fn unit_grid(u: &Unit, profile: &TargetProfile) -> (u64, u64) {
    let fold = u
        .members
        .iter()
        .find_map(|mb| match &mb.kind {
            UKind::Fold { site, .. } => Some(site),
            _ => None,
        });
    match fold {
        Some(site) => gpu::rowfold_grid(site, profile),
        None => gpu::cell_grid(u.rows),
    }
}

/// `consumer cells` → the producer address hull a tap-carrying affine reach
/// gives (`Reach::rect` over a one-wide producer).
fn span(reach: &Reach, cells: std::ops::Range<u64>) -> Option<std::ops::Range<u64>> {
    let Reach::Affine { w, .. } = reach else {
        return None;
    };
    let (lo, hi) = (cells.start, cells.end);
    if hi <= lo || *w == 0 {
        return None;
    }
    let (q0, q1) = (lo / w, (hi - 1) / w);
    let cols = if q0 == q1 {
        lo % w..(hi - 1) % w + 1
    } else {
        0..*w
    };
    let [rows, _] = reach.rect(1, q0..q1 + 1, cols)?;
    Some(rows)
}

/// Does `reach`, read by a consumer with `cw` cells per row, stay inside the
/// producer's row `i` (`pw` cells per row) for every row of `rows`? Exact:
/// when the consumer's row is whole reach rows (`cw % w == 0`) both ends of
/// the hull are affine in `i`, so the two end rows decide it; otherwise every
/// row is checked, up to 65536 of them.
fn rowwise(reach: &Reach, cw: u64, pw: u64, rows: u64) -> bool {
    let Reach::Affine { w, .. } = reach else {
        return false;
    };
    let inside = |i: u64| {
        span(reach, i * cw..(i + 1) * cw).is_some_and(|s| s.start >= i * pw && s.end <= (i + 1) * pw)
    };
    if *w > 0 && cw % w == 0 {
        return [0, 1.min(rows - 1), rows - 1].into_iter().all(inside);
    }
    // ponytail: a row check per row beyond the aligned case; 65536 rows is
    // the ceiling — an unaligned read past it is refused (footprint).
    rows <= 1 << 16 && (0..rows).all(inside)
}

/// The rows a reach of `cw`-cell consumer rows reads a per-row value at: the
/// row itself, every row.
fn rowwise_single(reach: &Reach, cw: u64, rows: u64) -> bool {
    rowwise(reach, cw, 1, rows)
}

const SAME: Reach = Reach::Affine {
    w: 1,
    base: 0,
    cq: 1,
    cr: 0,
    taps: Vec::new(),
};

impl<'a> FnEmit<'a> {
    /// **The rule (generic-fusion `fuse`)**: group the placed folds and cells
    /// of this plan into unit kernels, greedily in position order — a group
    /// starts at a unit no group holds and takes, one at a time, the earliest
    /// reader of a member that [`Self::unit_of`] admits. Refusals are
    /// recorded with their reason for the host intent line.
    pub(super) fn fuse_units(&self, plan: &mut ResidentPlan) {
        let rungs = gpu::host_rungs(self.profile, self.contract, self.gpu_arms);
        let mut refused: Vec<(MorphismId, MorphismId, &'static str)> = Vec::new();
        if rungs.fuse && plan.chain.is_empty() {
            let mut cand: SecondaryMap<MorphismId, Cand> = SecondaryMap::new();
            for (m, r) in plan.rowfold.iter() {
                cand.insert(m, Cand::Fold(r.clone()));
            }
            for (m, c) in plan.cells.iter() {
                cand.insert(m, Cand::Cell(c.clone()));
            }
            let fp = self.ir.footprint_plan(self.f);
            let region = self.regions();
            let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
            let target = |m: MorphismId| self.ir.morphism(m).expect("unit").target;
            let mut order: Vec<MorphismId> = cand.keys().collect();
            order.sort_by_key(|&m| pos(m));
            let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
            let mut groups: Vec<Vec<MorphismId>> = Vec::new();
            for &seed in &order {
                if taken.contains_key(seed) {
                    continue;
                }
                let mut g = vec![seed];
                loop {
                    let mut next: Vec<(MorphismId, MorphismId)> = g
                        .iter()
                        .flat_map(|&x| self.cell_readers(target(x)).into_iter().map(move |r| (x, r)))
                        .filter(|&(_, r)| {
                            cand.contains_key(r) && !taken.contains_key(r) && !g.contains(&r)
                        })
                        .collect();
                    next.sort_by_key(|&(_, r)| pos(r));
                    next.dedup_by_key(|(_, r)| *r);
                    let mut admitted = false;
                    for (p, r) in next {
                        let mut trial = g.clone();
                        trial.push(r);
                        match self.unit_of(&trial, &cand, &fp, &region, plan) {
                            Ok(_) => {
                                refused.retain(|&(_, c, _)| c != r);
                                g = trial;
                                admitted = true;
                                break;
                            }
                            Err(why) => {
                                refused.retain(|&(_, c, _)| c != r);
                                refused.push((p, r, why));
                            }
                        }
                    }
                    if !admitted {
                        break;
                    }
                }
                if g.len() >= 2 {
                    for &m in &g {
                        taken.insert(m, ());
                    }
                    groups.push(g);
                }
            }
            for g in groups {
                let u = self
                    .unit_of(&g, &cand, &fp, &region, plan)
                    .expect("an admitted group is a unit");
                for (k, mb) in u.members.iter().enumerate() {
                    plan.cells.remove(mb.m);
                    plan.rowfold.remove(mb.m);
                    if k > 0 {
                        plan.skip.insert(mb.m, ());
                    }
                    if !mb.store {
                        for v in plan.free_after.values_mut() {
                            v.retain(|&o| o != mb.out);
                        }
                    }
                }
                refused.retain(|&(_, c, _)| !g.contains(&c));
                plan.units.insert(g[0], u);
            }
        }
        // The host intent: every group, every launch, every refused edge.
        let launches = self.device.len()
            + plan.rowfold.len()
            + plan.cells.len()
            + plan.units.len()
            + plan.devgen.len();
        let name = |m: MorphismId| self.object_label(self.ir.morphism(m).expect("m").target);
        let refused: Vec<String> = refused
            .iter()
            .map(|&(p, c, why)| format!("{}->{}:{why}", name(p), name(c)))
            .collect();
        // Only where the rule decided something: a function it leaves alone
        // keeps its text.
        if !plan.units.is_empty() || !refused.is_empty() {
            plan.fuse_line = format!(
                "; mapal.fuse.host groups={} units={} launches={launches} refused={}\n",
                self.groups.len() + plan.units.len(),
                plan.units.len(),
                refused.join(";")
            );
        }
    }

    /// An object's name in an intent line: its source name, else `o<seq>`.
    pub(super) fn object_label(&self, o: ObjectId) -> String {
        let obj = self.ir.object(o).expect("object resolves");
        match &obj.name {
            Some(n) => n.clone(),
            None => format!("{o:?}")
                .trim_start_matches("ObjectId(")
                .trim_end_matches(')')
                .replace('v', "_")
                .chars()
                .fold(String::from("o"), |mut s, c| {
                    s.push(c);
                    s
                }),
        }
    }

    /// The unit of the members `ms` (position order), or why not: `phase`
    /// (another region, or an operand not complete at the launch), `gated`,
    /// `footprint`, `pointers`, `budget`, `cost`.
    fn unit_of(
        &self,
        ms: &[MorphismId],
        cand: &SecondaryMap<MorphismId, Cand>,
        fp: &mapal_ir::FootprintPlan,
        region: &SecondaryMap<MorphismId, (Option<ObjectId>, bool)>,
        plan: &ResidentPlan,
    ) -> Result<Unit, &'static str> {
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let launch = ms[0];
        let rs = *region.get(launch).ok_or("phase")?;
        if ms.iter().any(|&m| region.get(m) != Some(&rs)) {
            return Err("phase");
        }
        if ms.iter().any(|&m| self.gated.contains_key(m)) {
            return Err("gated");
        }
        let target = |m: MorphismId| self.ir.morphism(m).expect("member").target;
        let extent = |m: MorphismId| array_len(&self.obj_ty(target(m)));
        let fold_rows: Vec<u64> = ms
            .iter()
            .filter_map(|&m| match &cand[m] {
                Cand::Fold(r) => Some(r.site.rows),
                Cand::Cell(_) => None,
            })
            .collect();
        let rows = fold_rows.first().copied().unwrap_or_else(|| extent(ms[0]));
        if rows == 0
            || fold_rows.iter().any(|&r| r != rows)
            || ms.iter().any(|&m| extent(m) % rows != 0)
            || (fold_rows.is_empty() && ms.iter().any(|&m| extent(m) != rows))
        {
            return Err("footprint");
        }
        // Complete in `Gmem` when the group launches, where it runs: made by
        // a morphism before the launch in this region or straight-line, or a
        // view of this loop's state.
        let complete = |o: ObjectId| match self.ir.in_edges(o) {
            [] => true,
            [p] => {
                let made = pos(*p) < pos(launch)
                    && region.get(*p).is_some_and(|&r| r == rs || r.0.is_none());
                let view = rs.0.is_some_and(|mg| {
                    let mo = self.ir.morphism(*p).expect("morphism resolves");
                    matches!(mo.op, Operation::Proj { .. }) && mo.source == mg
                });
                made || view
            }
            _ => false,
        };
        let reach_of = |m: MorphismId, slot: u32| {
            fp.of(m)
                .iter()
                .find(|e| e.slot == slot && e.law.is_none())
                .map(|e| e.reach.clone())
        };
        let mut members: Vec<UMember> = Vec::new();
        let mut gmem: Vec<ObjectId> = Vec::new();
        let push = |o: ObjectId, gmem: &mut Vec<ObjectId>| match gmem.iter().position(|&x| x == o) {
            Some(g) => g,
            None => {
                gmem.push(o);
                gmem.len() - 1
            }
        };
        // May a reader with `cw` cells per row read member `p` at `reach`?
        let readable = |p: &UMember, reach: &Reach, cw: u64| -> Result<bool, &'static str> {
            let free = p.srcs.iter().all(|s| !matches!(s, USrc::Member { .. }));
            // Every read of the slot at one affine address (a body that
            // reads it at two takes the range, and no one-cell buffer).
            if !matches!(reach, Reach::Affine { .. }) {
                return Err("footprint");
            }
            match p.kind {
                UKind::Fold { .. } if rowwise_single(reach, cw, rows) => Ok(true),
                UKind::Fold { .. } => Err("footprint"),
                UKind::Cell { .. } if rowwise(reach, cw, p.w, rows) => Ok(true),
                UKind::Cell { .. } if free => Ok(false),
                UKind::Cell { .. } => Err("footprint"),
            }
        };
        for &m in ms {
            let morph = self.ir.morphism(m).expect("member");
            let w = extent(m) / rows;
            let member = |o: ObjectId, members: &[UMember]| members.iter().position(|mb| mb.out == o);
            let mb = match &cand[m] {
                Cand::Fold(r) => {
                    let a = self.pair_source(morph.source, r.site.a.slot).ok_or("footprint")?;
                    let src = match member(a, &members) {
                        Some(k) => {
                            let reach = reach_of(m, r.site.a.slot).ok_or("footprint")?;
                            // A fold's step reads `k` cells of its row: the
                            // producer is a cell (recomputed at each), never
                            // another fold.
                            if !matches!(members[k].kind, UKind::Cell { .. }) {
                                return Err("footprint");
                            }
                            let row = readable(&members[k], &reach, 1)?;
                            USrc::Member { k, reach, row }
                        }
                        None if complete(a) => USrc::Gmem(push(a, &mut gmem)),
                        None => return Err("phase"),
                    };
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Fold {
                            site: r.site.clone(),
                            colmajor: r.colmajor,
                        },
                        w,
                        store: false,
                        srcs: vec![src],
                    }
                }
                Cand::Cell(c) => {
                    let Operation::Map { body, captures } = morph.op else {
                        return Err("footprint");
                    };
                    let mut srcs = Vec::new();
                    for slot in 0..=captures {
                        let arr = if captures == 0 {
                            Some(morph.source)
                        } else {
                            self.pair_source(morph.source, slot)
                        }
                        .ok_or("footprint")?;
                        let law = c.laws.iter().find(|&&(s, _)| s == slot).map(|&(_, f)| f);
                        let src = if slot == captures && c.mapped.is_none() {
                            USrc::Index
                        } else if let Some(f) = law {
                            USrc::Law(f)
                        } else if let Some(k) = member(arr, &members) {
                            let reach = if slot == captures {
                                SAME
                            } else {
                                reach_of(m, slot).ok_or("footprint")?
                            };
                            let row = readable(&members[k], &reach, w)?;
                            USrc::Member { k, reach, row }
                        } else if complete(arr) {
                            USrc::Gmem(push(arr, &mut gmem))
                        } else {
                            return Err("phase");
                        };
                        srcs.push(src);
                    }
                    UMember {
                        m,
                        out: morph.target,
                        kind: UKind::Cell { body, captures },
                        w,
                        store: false,
                        srcs,
                    }
                }
            };
            members.push(mb);
        }
        for mb in &mut members {
            mb.store = self.lup.escapes(mb.out) || plan.is_carry_next(mb.out) || self.read_outside(mb.out, ms);
        }
        // Budget: one pointer per operand and stored value; per-row values
        // held in registers beside the address state.
        let gpu = self.profile.gpu().expect("a unit has a GPU profile");
        let stored = members.iter().filter(|mb| mb.store).count();
        if gmem.len() + stored > gpu::LAUNCHV_MAX {
            return Err("pointers");
        }
        let elem = |o: ObjectId| crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(o)));
        let held: u64 = members
            .iter()
            .filter(|mb| mb.per_row())
            .map(|mb| elem(mb.out).div_ceil(4))
            .sum::<u64>()
            + crate::profile::FMA_ADDR_REGS;
        if held > gpu.regs_per_thread {
            return Err("budget");
        }
        if let Some(d) = gpu.cost {
            let (fused, unfused) = self.unit_roofline(&members, rows, &gmem, elem);
            let ns = |(f, b): (u64, u64)| crate::profile::roofline_ns(f, b, d.flops_per_ns, d.bytes_per_ns);
            let unfused: u64 = unfused.into_iter().map(|x| d.launch_ns + ns(x)).sum();
            if d.launch_ns + ns(fused) > unfused {
                return Err("cost");
            }
        }
        let key = format!("unit|{members:?}|{rows}|{}", self.profile.name);
        let id = key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        Ok(Unit {
            id,
            rows,
            members,
            gmem,
        })
    }

    /// `(flops, bytes)` of the fused launch — every member evaluation, the
    /// recomputed halo and in-row re-reads included — and of each member's
    /// own launch.
    fn unit_roofline(
        &self,
        members: &[UMember],
        rows: u64,
        gmem: &[ObjectId],
        elem: impl Fn(ObjectId) -> u64,
    ) -> ((u64, u64), Vec<(u64, u64)>) {
        let steps = |mb: &UMember| match &mb.kind {
            UKind::Fold { site, .. } => rows * site.k,
            UKind::Cell { .. } => rows * mb.w,
        };
        // Evaluations: a fold walks its row once; a per-row cell runs once a
        // row; a cell of `w > 1` is swept when stored and evaluated at every
        // read a register does not serve.
        let mut evals: Vec<u64> = members
            .iter()
            .map(|mb| match mb.kind {
                UKind::Fold { .. } => steps(mb),
                UKind::Cell { .. } if mb.per_row() || mb.store => steps(mb),
                UKind::Cell { .. } => 0,
            })
            .collect();
        for i in (0..members.len()).rev() {
            for s in &members[i].srcs {
                if let USrc::Member { k, row, .. } = s
                    && !(members[*k].per_row() && *row)
                {
                    evals[*k] += evals[i];
                }
            }
        }
        let gmem_reads = |mb: &UMember| mb.srcs.iter().filter(|s| matches!(s, USrc::Gmem(_))).count() as u64;
        let e = gmem.first().map_or(4, |&o| elem(o));
        let flops: u64 = evals.iter().sum();
        let bytes: u64 = members
            .iter()
            .zip(&evals)
            .map(|(mb, &n)| n * gmem_reads(mb) * e + if mb.store { steps(mb) * e } else { 0 })
            .sum();
        let own = members
            .iter()
            .map(|mb| {
                let reads = mb
                    .srcs
                    .iter()
                    .filter(|s| matches!(s, USrc::Gmem(_) | USrc::Member { .. }))
                    .count() as u64;
                match &mb.kind {
                    UKind::Fold { site, .. } => (rows * site.k, (rows * site.k + rows) * e),
                    UKind::Cell { .. } => (steps(mb), (reads + 1) * steps(mb) * e),
                }
            })
            .collect();
        ((flops, bytes), own)
    }

    /// The host half of a unit: its stored values' buffers, its operands'
    /// device copies (copied in here only when nothing placed one), one
    /// launch, and the copy-out of a stored value the host reads whole.
    pub(super) fn emit_unit(&mut self, u: &Unit) {
        let mut ptrs = Vec::new();
        let mut stored = Vec::new();
        for mb in u.members.iter().filter(|mb| mb.store) {
            let bytes = self.array_bytes(&self.obj_ty(mb.out));
            let d = self.dalloc(mb.out, bytes);
            ptrs.push(d.clone());
            stored.push((mb.out, d));
        }
        let mut temps = Vec::new();
        for &o in &u.gmem {
            let at = self.resident.as_ref().map_or(o, |p| p.alias(o));
            if self.dvalid.contains_key(at) {
                ptrs.push(self.dload(o));
                continue;
            }
            let ob = self.array_bytes(&self.obj_ty(o));
            let h = self.array_operand_ptr(o, None).expect("unit operand ptr");
            let s = format!("%d{}", self.fresh());
            self.allocas
                .push_str(&format!("  {s} = alloca i64\n  store i64 0, ptr {s}\n"));
            let d = self.tmp();
            self.line(format!("{d} = call i64 @mapal_gpu_alloc(ptr {s}, i64 {ob})"));
            self.line(format!("call void @mapal_gpu_h2d(i64 {d}, ptr {h}, i64 {ob})"));
            temps.push(s);
            ptrs.push(d);
        }
        let args = format!("%dv{}", self.fresh());
        let np = ptrs.len();
        self.allocas.push_str(&format!("  {args} = alloca [{np} x i64]\n"));
        for (i, p) in ptrs.iter().enumerate() {
            let g = self.tmp();
            self.line(format!("{g} = getelementptr [{np} x i64], ptr {args}, i64 0, i64 {i}"));
            self.line(format!("store i64 {p}, ptr {g}"));
        }
        let (gx, block) = unit_grid(u, self.profile);
        self.line(format!(
            "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 {np}, ptr {args}, i32 {gx}, i32 1, i32 {block}, i32 0) ; unit",
            u.id as i64
        ));
        for t in temps {
            self.line(format!("call void @mapal_gpu_free(ptr {t})"));
        }
        for (o, d) in stored {
            self.dvalid.insert(o, ());
            let keep = self
                .resident
                .as_ref()
                .is_some_and(|p| p.keep.contains_key(o) || p.is_carry_next(o));
            if !keep {
                let bytes = self.array_bytes(&self.obj_ty(o));
                let h = self.slot(o).expect("unit output slot");
                self.d2h(0, &format!("nocapture {h}"), &d, "0", bytes);
            }
        }
    }

    /// This function's unit kernels for the device module.
    pub(crate) fn unit_kernels(&self) -> Vec<UnitKernel> {
        let Some(plan) = &self.resident else {
            return Vec::new();
        };
        plan.units
            .values()
            .map(|u| {
                let (fns, text) = self.unit_kernel(u);
                UnitKernel {
                    f: self.f,
                    members: u.members.iter().map(|mb| mb.m).collect(),
                    fns,
                    text,
                }
            })
            .collect()
    }

    /// A body's text as a unit calls it: the host's, or the variant whose
    /// capture `slots` arrive as one-cell buffers (`_u<slots>`).
    fn unit_body(&self, f: FuncId, slots: &[u32]) -> (String, String) {
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
        fe.set_gpu_arms(self.gpu_arms, self.reduced);
        if slots.is_empty() {
            return (self.fnames[f].clone(), fe.emit());
        }
        let suffix: Vec<String> = slots.iter().map(u32::to_string).collect();
        fe.cell_suffix = format!("_u{}", suffix.join("_"));
        fe.set_cell_slots(slots.to_vec());
        (format!("{}{}", self.fnames[f], fe.cell_suffix), fe.emit())
    }

    /// The unit's kernel: the bodies it calls, and its text (with its
    /// `; mapal.fuse` and `; mapal.plan` intent lines).
    fn unit_kernel(&self, u: &Unit) -> (Vec<(String, String)>, String) {
        let name = gpu::kernel_name(u.id);
        let (blocks, threads) = unit_grid(u, self.profile);
        let lt = |o: ObjectId| array_parts(&self.obj_ty(o)).0;
        let mut fns: Vec<(String, String)> = Vec::new();
        let mut g = UnitGen {
            out: String::new(),
            entry: String::new(),
            n: 0,
            cur: "body".into(),
            vals: vec![None; u.members.len()],
            calls: Vec::new(),
            ys: Vec::new(),
            gmem_ty: u.gmem.iter().map(|&o| lt(o)).collect(),
            u,
        };
        // Each member's callee, argument type, result type and buffer types.
        let mut y = 0;
        for mb in &u.members {
            g.ys.push(mb.store.then(|| {
                y += 1;
                y - 1
            }));
            let UKind::Cell { body, captures } = mb.kind else {
                g.calls.push(None);
                continue;
            };
            let morph = self.ir.morphism(mb.m).expect("member");
            let slots: Vec<u32> = (0..captures)
                .filter(|&s| matches!(mb.srcs[s as usize], USrc::Member { .. } | USrc::Law(_)))
                .collect();
            let (callee, text) = self.unit_body(body, &slots);
            if !fns.iter().any(|(n, _)| *n == callee) {
                fns.push((callee.clone(), text));
            }
            for s in &mb.srcs {
                if let USrc::Law(f) = *s {
                    let (ln, lt) = self.unit_body(f, &[]);
                    if !fns.iter().any(|(n, _)| *n == ln) {
                        fns.push((ln, lt));
                    }
                }
            }
            let in_ty = self.obj_ty(self.ir.func(body).expect("body").input);
            let arg = lower_body_input_ty(&in_ty, captures).expect("a member's input lowers");
            let field_ty: Vec<String> = (0..=captures)
                .map(|s| {
                    let arr = if captures == 0 {
                        morph.source
                    } else {
                        self.pair_source(morph.source, s).expect("slot")
                    };
                    lt(arr)
                })
                .collect();
            let laws: Vec<(u32, String)> = mb
                .srcs
                .iter()
                .enumerate()
                .filter_map(|(s, x)| match *x {
                    USrc::Law(f) => Some((s as u32, self.fnames[f].clone())),
                    _ => None,
                })
                .collect();
            g.calls.push(Some(CallInfo {
                callee,
                arg,
                ret: lt(mb.out),
                field_ty,
                captures,
                slots,
                laws,
            }));
        }
        // Parameters: the stored values, then the operands.
        let mut params: Vec<String> = Vec::new();
        for (k, mb) in u.members.iter().enumerate() {
            if let Some(j) = g.ys[k] {
                params.push(format!("ptr addrspace(1) noalias %y{j}"));
                let _ = mb;
            }
        }
        for i in 0..u.gmem.len() {
            params.push(format!("ptr addrspace(1) readonly %x{i}"));
            g.entry
                .push_str(&format!("  %xg{i} = addrspacecast ptr addrspace(1) %x{i} to ptr\n"));
        }
        for (k, c) in g.calls.iter().enumerate() {
            if let Some(c) = c {
                for &s in &c.slots {
                    g.entry
                        .push_str(&format!("  %b{k}_{s} = alloca {}\n", c.field_ty[s as usize]));
                }
            }
        }
        // The phases.
        for k in 0..u.members.len() {
            g.phase(k);
        }
        // The intent line (design §7): what the plan keeps, stores and reads.
        let label = |o: ObjectId| self.object_label(o);
        let kind = |mb: &UMember| match mb.kind {
            UKind::Fold { .. } => "rowfold",
            UKind::Cell { .. } => "cell",
        };
        let members: Vec<String> = u
            .members
            .iter()
            .map(|mb| {
                let halo = mb.srcs.iter().any(|s| matches!(s, USrc::Member { row: false, .. }));
                format!("{}:{}:1x{}{}", label(mb.out), kind(mb), mb.w, if halo { "+halo" } else { "" })
            })
            .collect();
        let keep: Vec<String> = u
            .members
            .iter()
            .filter(|mb| !mb.store)
            .map(|mb| {
                let e = crate::profile::elem_bytes(super::resident::array_elem(&self.obj_ty(mb.out)));
                format!("{}:reg:{}", label(mb.out), if mb.per_row() { e } else { 0 })
            })
            .collect();
        let store: Vec<String> = u
            .members
            .iter()
            .filter(|mb| mb.store)
            .map(|mb| label(mb.out))
            .collect();
        let gm: Vec<String> = u.gmem.iter().map(|&o| label(o)).collect();
        let laws: Vec<String> = u
            .members
            .iter()
            .flat_map(|mb| mb.srcs.iter())
            .filter_map(|s| match *s {
                USrc::Law(f) => Some(self.fnames[f].clone()),
                _ => None,
            })
            .collect();
        let root = u
            .members
            .iter()
            .find(|mb| matches!(mb.kind, UKind::Fold { .. }))
            .unwrap_or(&u.members[0]);
        let w = u.members.iter().map(|mb| mb.w).max().unwrap_or(1);
        let mut text = format!(
            "; mapal.fuse kernel={name} unit=1x{w} rows={} root={} members={} keep={} store={} gmem={} law={} smem=0\n",
            u.rows,
            label(root.out),
            members.join(";"),
            keep.join(";"),
            store.join(";"),
            gm.join(";"),
            laws.join(";"),
        );
        text += &format!(
            "; mapal.plan kernel={name} kind=fuse rows={} gx={blocks} block={threads} members={} rungs_on=fuse\n",
            u.rows,
            u.members.len()
        );
        text += &format!(
            "define ptx_kernel void @{name}({}) \"nvvm.maxntid\"=\"{threads}\" {{\n\
             entry:\n{}\
             \x20 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             \x20 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             \x20 %t64 = zext i32 %tid to i64\n\
             \x20 %b64 = zext i32 %bid to i64\n\
             \x20 %row0 = mul i64 %b64, {threads}\n\
             \x20 %i = add i64 %row0, %t64\n\
             \x20 %in = icmp ult i64 %i, {}\n\
             \x20 br i1 %in, label %body, label %done\n\
             body:\n{}\
             \x20 br label %done\n\
             done:\n\
             \x20 ret void\n\
             }}\n",
            params.join(", "),
            g.entry,
            u.rows,
            g.out,
        );
        (fns, text)
    }
}

/// A cell member's call: callee, argument and result types, each slot's
/// array element type, the capture slots fed by one-cell buffers, and the
/// law function of each law slot.
struct CallInfo {
    callee: String,
    arg: String,
    ret: String,
    field_ty: Vec<String>,
    captures: u32,
    slots: Vec<u32>,
    laws: Vec<(u32, String)>,
}

/// The unit kernel's body builder.
struct UnitGen<'u> {
    out: String,
    entry: String,
    n: u32,
    /// The current block's label (a phi's predecessor).
    cur: String,
    /// Each per-row member's value in this row, once its phase ran.
    vals: Vec<Option<String>>,
    calls: Vec<Option<CallInfo>>,
    /// Each stored member's output parameter `%y<j>`.
    ys: Vec<Option<usize>>,
    gmem_ty: Vec<String>,
    u: &'u Unit,
}

impl UnitGen<'_> {
    fn tmp(&mut self) -> String {
        self.n += 1;
        format!("%v{}", self.n)
    }

    fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    fn label(&mut self, l: &str) {
        self.out.push_str(l);
        self.out.push_str(":\n");
        self.cur = l.to_string();
    }

    fn op(&mut self, op: &str, a: &str, b: impl std::fmt::Display) -> String {
        let r = self.tmp();
        self.line(format!("{r} = {op} i64 {a}, {b}"));
        r
    }

    /// The address `reach` gives cell `t` (with the counted axes `taps`).
    fn addr(&mut self, reach: &Reach, t: &str, taps: &[&str]) -> String {
        let Reach::Affine {
            w,
            base,
            cq,
            cr,
            taps: tt,
        } = reach
        else {
            unreachable!("a unit reads members at affine reaches")
        };
        let (q, r) = if *w == 1 {
            (t.to_string(), None)
        } else {
            (self.op("udiv", t, w), Some(self.op("urem", t, w)))
        };
        let mut acc = self.op("mul", &q, cq);
        if let Some(r) = r {
            let rr = self.op("mul", &r, cr);
            acc = self.op("add", &acc, &rr);
        }
        for (tap, k) in tt.iter().zip(taps) {
            let kk = self.op("mul", k, tap.coef);
            acc = self.op("add", &acc, &kk);
        }
        self.op("add", &acc, base)
    }

    fn i32_of(&mut self, t: &str) -> String {
        let x = self.tmp();
        self.line(format!("{x} = trunc i64 {t} to i32"));
        x
    }

    fn load(&mut self, g: usize, at: &str) -> String {
        let ty = self.gmem_ty[g].clone();
        let p = self.tmp();
        self.line(format!("{p} = getelementptr inbounds {ty}, ptr addrspace(1) %x{g}, i64 {at}"));
        let v = self.tmp();
        self.line(format!("{v} = load {ty}, ptr addrspace(1) {p}"));
        v
    }

    /// Source `s` of a reader at its cell `t` (and fold axes `taps`).
    fn read(&mut self, s: &USrc, t: &str, taps: &[&str]) -> String {
        match s {
            USrc::Member { k, reach, row } => {
                if self.u.members[*k].per_row() && *row {
                    return self.vals[*k].clone().expect("a per-row value is read after its phase");
                }
                let a = self.addr(reach, t, taps);
                self.cell(*k, &a)
            }
            USrc::Gmem(g) => self.load(*g, t),
            USrc::Index => self.i32_of(t),
            USrc::Law(_) => unreachable!("a law feeds a capture slot"),
        }
    }

    /// Cell member `k`'s value at its cell `t`.
    fn cell(&mut self, k: usize, t: &str) -> String {
        let u = self.u;
        let mb = &u.members[k];
        let c = self.calls[k].as_ref().expect("a cell member");
        let (callee, arg, ret, captures) = (c.callee.clone(), c.arg.clone(), c.ret.clone(), c.captures);
        let field_ty = c.field_ty.clone();
        let laws = c.laws.clone();
        let mut fields: Vec<(String, String)> = Vec::new();
        for (s, src) in mb.srcs.iter().enumerate() {
            let s32 = s as u32;
            if s32 < captures {
                let v = match src {
                    USrc::Gmem(g) => {
                        fields.push(("ptr".into(), format!("%xg{g}")));
                        continue;
                    }
                    USrc::Law(_) => {
                        let law = &laws.iter().find(|(x, _)| *x == s32).expect("law").1;
                        let ti = self.i32_of(t);
                        let v = self.tmp();
                        self.line(format!("{v} = call {} @{law}(i32 {ti})", field_ty[s]));
                        v
                    }
                    other => self.read(other, t, &[]),
                };
                self.line(format!("store {} {v}, ptr %b{k}_{s}", field_ty[s]));
                fields.push(("ptr".into(), format!("%b{k}_{s}")));
            } else {
                let v = self.read(src, t, &[]);
                let ty = if matches!(src, USrc::Index) {
                    "i32".to_string()
                } else {
                    field_ty[s].clone()
                };
                fields.push((ty, v));
            }
        }
        let a = if captures == 0 {
            fields[0].1.clone()
        } else {
            let mut agg = "poison".to_string();
            for (f, (ty, v)) in fields.iter().enumerate() {
                let n = self.tmp();
                self.line(format!("{n} = insertvalue {arg} {agg}, {ty} {v}, {f}"));
                agg = n;
            }
            agg
        };
        let r = self.tmp();
        self.line(format!("{r} = call {ret} @{callee}({arg} {a})"));
        r
    }

    fn store(&mut self, k: usize, v: &str, at: &str) {
        if let Some(j) = self.ys[k] {
            let ty = match &self.calls[k] {
                Some(c) => c.ret.clone(),
                None => match &self.u.members[k].kind {
                    UKind::Fold { site, .. } => lower_ty(&site.elem).expect("fold elem"),
                    UKind::Cell { .. } => unreachable!("a cell has a call"),
                },
            };
            let p = self.tmp();
            self.line(format!("{p} = getelementptr inbounds {ty}, ptr addrspace(1) %y{j}, i64 {at}"));
            self.line(format!("store {ty} {v}, ptr addrspace(1) {p}"));
        }
    }

    /// Member `k`'s phase in row `%i`.
    fn phase(&mut self, k: usize) {
        let u = self.u;
        let mb = &u.members[k];
        match &mb.kind {
            UKind::Fold { site, .. } => {
                let t = lower_ty(&site.elem).expect("float elem lowers");
                let seed = const_literal(&site.seed);
                let pre = self.cur.clone();
                let (lp, end) = (format!("f{k}"), format!("f{k}e"));
                self.line(format!("br label %{lp}"));
                self.label(&lp);
                let (kv, acc) = (format!("%k{k}"), format!("%a{k}"));
                self.line(format!("{kv} = phi i64 [0, %{pre}], [{kv}n, %{lp}]"));
                self.line(format!("{acc} = phi {t} [{seed}, %{pre}], [{acc}n, %{lp}]"));
                // The fold's axes: `k`, or a window's `(k÷div, k%div)`.
                let taps: Vec<String> = match site.a.ksplit {
                    Some(ks) => vec![self.op("udiv", &kv, ks.div), self.op("urem", &kv, ks.div)],
                    None => vec![kv.clone()],
                };
                let v = match &mb.srcs[0] {
                    USrc::Gmem(g) => {
                        // The site's own addressing (re-addressed when its
                        // producer stores column-major).
                        let ri = if site.w > 1 {
                            let q = self.op("udiv", "%i", site.w);
                            let l = self.op("urem", "%i", site.w);
                            let a = self.op("mul", &q, site.a.ci);
                            let b = self.op("mul", &l, site.a.clane);
                            self.op("add", &a, &b)
                        } else {
                            self.op("mul", "%i", site.a.ci)
                        };
                        let rk = match site.a.ksplit {
                            Some(ks) => {
                                let a = self.op("mul", &taps[0], ks.cq);
                                let b = self.op("mul", &taps[1], ks.cr);
                                self.op("add", &a, &b)
                            }
                            None => self.op("mul", &kv, site.a.ck),
                        };
                        let x = self.op("add", &ri, &rk);
                        let x = self.op("add", &x, site.a.base);
                        self.load(*g, &x)
                    }
                    src => {
                        let src = src.clone();
                        let taps: Vec<&str> = taps.iter().map(String::as_str).collect();
                        self.read(&src, "%i", &taps)
                    }
                };
                let step = match site.op {
                    RowFoldOp::Arith { op, acc_first } => {
                        let code = match op {
                            Operation::Add => "fadd",
                            Operation::Sub => "fsub",
                            _ => "fmul",
                        };
                        let (x, y) = if acc_first { (acc.clone(), v) } else { (v, acc.clone()) };
                        format!("{acc}n = {code} {t} {x}, {y}")
                    }
                    RowFoldOp::Select {
                        cmp,
                        acc_lhs,
                        acc_if_true,
                    } => {
                        let pred = match cmp {
                            Operation::Lt => "olt",
                            Operation::Gt => "ogt",
                            Operation::Le => "ole",
                            _ => "oge",
                        };
                        let (x, y) = if acc_lhs { (acc.clone(), v.clone()) } else { (v.clone(), acc.clone()) };
                        let (p, q) = if acc_if_true { (acc.clone(), v) } else { (v, acc.clone()) };
                        let c = self.tmp();
                        self.line(format!("{c} = fcmp {pred} {t} {x}, {y}"));
                        format!("{acc}n = select i1 {c}, {t} {p}, {t} {q}")
                    }
                };
                self.line(step);
                self.line(format!("{kv}n = add i64 {kv}, 1"));
                let fin = self.tmp();
                self.line(format!("{fin} = icmp eq i64 {kv}n, {}", site.k));
                self.line(format!("br i1 {fin}, label %{end}, label %{lp}"));
                self.label(&end);
                let acc_n = format!("{acc}n");
                self.store(k, &acc_n, "%i");
                self.vals[k] = Some(acc_n);
            }
            UKind::Cell { .. } if mb.per_row() => {
                let v = self.cell(k, "%i");
                self.store(k, &v, "%i");
                self.vals[k] = Some(v);
            }
            UKind::Cell { .. } if mb.store => {
                let w = mb.w;
                let pre = self.cur.clone();
                let (lp, end) = (format!("s{k}"), format!("s{k}e"));
                self.line(format!("br label %{lp}"));
                self.label(&lp);
                let j = format!("%j{k}");
                self.line(format!("{j} = phi i64 [0, %{pre}], [{j}n, %{lp}]"));
                let r = self.op("mul", "%i", w);
                let t = self.op("add", &r, &j);
                let v = self.cell(k, &t);
                self.store(k, &v, &t);
                self.line(format!("{j}n = add i64 {j}, 1"));
                let fin = self.tmp();
                self.line(format!("{fin} = icmp eq i64 {j}n, {w}"));
                self.line(format!("br i1 {fin}, label %{end}, label %{lp}"));
                self.label(&end);
            }
            // Evaluated at its reads.
            UKind::Cell { .. } => {}
        }
    }
}
