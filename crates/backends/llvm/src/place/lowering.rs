//! `impl Lowering for Emitter`: constraints, footprint and instruction counts per lowering.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

impl Lowering for Emitter<'_> {
    fn id(&self) -> &'static str {
        self.id
    }

    fn traversals(&self) -> &'static [TravKind] {
        // The emitters' kernels: `wgmma` no combine at all; the one-thread
        // -per-cell kernel and the warp tile split; the FMA tile and the
        // 16-bit warp tile walk too (`declare` keeps the walk to them). An
        // arm that turns a combine's rung off removes its kind (the
        // epilogue groups' realizations, `group_order`).
        const ALL: [TravKind; 5] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::SplitWorkspace,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const NO_WS: [TravKind; 4] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const NO_ATOMIC: [TravKind; 4] = [
            TravKind::Ordered,
            TravKind::SplitWorkspace,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const WALKS: [TravKind; 3] = [TravKind::Ordered, TravKind::Walk, TravKind::Hybrid];
        const SPLITS: [TravKind; 3] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::SplitWorkspace,
        ];
        const ORDERED: [TravKind; 1] = [TravKind::Ordered];
        const WS: [TravKind; 2] = [TravKind::Ordered, TravKind::SplitWorkspace];
        const AT: [TravKind; 2] = [TravKind::Ordered, TravKind::SplitAtomic];
        if matches!(self.id, WGMMA | WGMMA_WS) {
            return &ORDERED;
        }
        let off = self.arms.off;
        match (off.splitk, off.splitk_atomic, off.streamk) {
            (false, false, false) => &ALL,
            (true, false, false) => &NO_WS,
            (false, true, false) => &NO_ATOMIC,
            (true, true, false) => &WALKS,
            (false, false, true) => &SPLITS,
            (true, false, true) => &AT,
            (false, true, true) => &WS,
            (true, true, true) => &ORDERED,
        }
    }

    /// The operations the search places the site on (D5 step 1): the
    /// face's scalar step for the scalar kernels; for the tensor tiles each
    /// matrix row of the lowering's issuing width whose format the site's
    /// precision set admits ([`mapal_place::space::ops`]'s rule over the
    /// part's rows: TF32 where the set holds it, a 16-bit format its
    /// operands stage in, the accumulator meeting the set), on a separable
    /// site with no radix, where the part copies (`cp.async`, `ldmatrix`
    /// for 16 bits) and the family's rung is on. Both roles for `wgmma`.
    fn ops(&self, _m: &Machine, _s: &Site) -> Vec<(usize, bool)> {
        let g = self.gpu();
        let site = self.site;
        let rungs = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
        match self.id {
            NAIVE | FMA => {
                let op = if self.contract { OP_FFMA } else { OP_MULADD };
                vec![(op, false)]
            }
            _ => {
                let shaped = site.a.clane == 0 && site.b.ci == 0 && site.radix.is_none();
                if !shaped || site.elem != Ty::f32() || !g.caps.cp_async {
                    return Vec::new();
                }
                let group = matches!(self.id, WGMMA | WGMMA_WS);
                let on = match self.id {
                    MMA => rungs.mma,
                    WGMMA => rungs.mma && rungs.wgmma,
                    _ => rungs.mma && rungs.wgmma && rungs.wgmma_ws,
                };
                if !on || (group && !g.has_wgmma()) {
                    return Vec::new();
                }
                let mut out = Vec::new();
                for (j, o) in g.caps.matrix.iter().enumerate() {
                    if (o.shape.threads > 32) != group || o.shape.threads <= 1 {
                        continue;
                    }
                    let admitted = match o.fmt {
                        MmaFmt::Tf32 => site.tf32 && rungs.tf32_tile && (group || o.shape.k == 8),
                        f => {
                            site.half.is_some_and(|h| MmaFmt::of_narrow(h) == f)
                                && rungs.half_tile
                                && o.acc_meets(-23)
                                && gpu::mma_widths(site, f).is_some()
                                && (group || (g.caps.ldmatrix && o.shape.k == 32 / f.bytes()))
                        }
                    };
                    let shape = group || (o.shape.m == 16 && o.shape.n == 8);
                    // A warpgroup product's `a` from shared memory where the
                    // site stores it in the format, else from registers
                    // (staged f32, rounded in the fragment load).
                    let a_home = match (group, o.fmt) {
                        (true, f) if f != MmaFmt::Tf32 => gpu::mma_widths(site, f)
                            .is_some_and(|(ea, _)| (o.a == MHome::Smem) == (ea == f.bytes())),
                        _ => true,
                    };
                    if !a_home {
                        continue;
                    }
                    if admitted && shape && o.a == MHome::Regs || admitted && group {
                        out.push((j + 2, false));
                        if group {
                            out.push((j + 2, true));
                        }
                    }
                }
                out.dedup();
                out
            }
        }
    }

    fn dims(&self) -> usize {
        self.coords().len()
    }

    fn search_domain(&self, _m: &Machine, _s: &Site, i: usize, pre: &[u64]) -> Vec<u64> {
        self.narrowed(i, pre)
    }

    fn point(&self, _m: &Machine, _s: &Site, pre: &[u64], swap: bool) -> Option<Schedule> {
        self.build(pre, swap)
    }

    fn lanes_of(&self, pre: &[u64], _swap: bool) -> Option<(u64, u64)> {
        match self.id {
            FMA if pre.len() >= 5 => Some((pre[3] * pre[4], pre[1] * pre[2])),
            MMA if pre.len() >= 5 => Some((pre[3] * pre[4] * self.gpu().warp, pre[1] * pre[2])),
            _ => None,
        }
    }

    fn tile_of(&self, pre: &[u64], _swap: bool) -> Option<(u64, u64)> {
        match self.id {
            FMA if pre.len() >= 5 => Some((pre[4] * pre[1], pre[3] * pre[2])),
            MMA if pre.len() >= 5 => Some((pre[3] * pre[1] * 16, pre[4] * pre[2] * 8)),
            WGMMA | WGMMA_WS if pre.len() >= 8 => Some(match pre[7] {
                1 => (pre[1], pre[2] * 64),
                _ => (pre[2] * 64, pre[1]),
            }),
            _ => None,
        }
    }

    fn slots(&self, m: &Machine, s: &Site, op: usize, swap: bool) -> Vec<Schedule> {
        // A depth-first walk of the coordinate domains, capped (the search's
        // pruning is C4's): each complete point lowered through the rules'
        // own encoding.
        let mut out = Vec::new();
        let n = self.coords().len();
        let mut stack: Vec<Vec<u64>> = vec![vec![op as u64]];
        while let Some(pre) = stack.pop() {
            if out.len() >= 4096 {
                break;
            }
            if pre.len() == n {
                if let Some(x) = self.build(&pre, swap) {
                    let mut x = x;
                    if let Ok(d) = self.declare(m, s, &x) {
                        x.per_unit = m.occupancy(d.threads, d.regs, &[(SMEM, d.smem)]);
                        out.push(x);
                    }
                }
                continue;
            }
            for v in self.domain(pre.len(), &pre) {
                let mut p = pre.clone();
                p.push(v);
                stack.push(p);
            }
        }
        out
    }

    fn declare(&self, _m: &Machine, s: &Site, x: &Schedule) -> Result<Declared, Refusal> {
        let g = self.gpu();
        let site = self.site;
        let e = elem_bytes_of(&site.elem);
        let (bm, bn, bk) = (x.bm().max(1), x.bn().max(1), x.bk().max(1));
        if x.lowering != NAIVE
            && (!s.m.is_multiple_of(bm) || !s.n.is_multiple_of(bn) || !s.k.is_multiple_of(bk))
        {
            return Err(Refusal::new("tiles do not divide the site (no tails)"));
        }
        let warp = g.warp.max(1);
        let l = launch_of(site, self.profile, self.contract, self.arms, x);
        if l.smem > g.smem_per_block {
            return Err(Refusal::new("shared memory above a slot's"));
        }
        if l.block > g.max_threads_per_block {
            return Err(Refusal::new("threads above a slot's"));
        }
        // ADR-0050 C4: the emitters' kernels for each traversal — the walk
        // and the hybrid (`Kernel::segment`) on the register-staged FMA tile
        // and the 16-bit warp tile without a split form; a raster on the
        // block-indexed kernels (the one-thread-per-cell kernel numbers its
        // cells).
        let walked = matches!(x.trav, Traversal::Walk { .. } | Traversal::Hybrid { .. });
        if walked {
            let ok = match (l.tiled, l.mma) {
                (Some(t), _) => t.stages < 2,
                (None, Some(m)) => m.fmt != MmaFmt::Tf32 && m.split.is_none(),
                _ => false,
            };
            if !ok {
                return Err(Refusal::new("the walk runs the staged FMA or 16-bit tile"));
            }
        }
        if self.id == NAIVE && x.trav.raster() > 1 {
            return Err(Refusal::new("the cell kernel has no raster"));
        }
        if let Some(t) = l.tiled {
            gpu::fma_constraints(site, &t, &l.rungs).map_err(Refusal::new)?;
        }
        if let Some(w) = l.wgmma.filter(|w| w.ws) {
            gpu::ws_tensors(site, &w, g, l.rungs.tma_digits)
                .ok_or_else(|| Refusal::new("no tensor boxes for the warp-specialized tile"))?;
        }
        let stages = x.depth();
        // `min-blocks`: the kernel asks for `per_unit` slots an SM
        // (`.minnctapersm`), so the code generator caps a lane's registers
        // at the file's share — the declared footprint is the counted one
        // within that cap (a spill past it is the post-lowering gate's,
        // ADR-0050 D5).
        let cap = |threads: u64| -> u64 {
            match (l.rungs.min_blocks, x.per_unit) {
                (true, u) if u > 0 => {
                    (g.regs_per_sm / u / threads.div_ceil(warp).max(1) / warp) / 8 * 8
                }
                _ => g.regs_per_thread,
            }
        };
        let pb = (bm * bk * s.ea + bn * bk * s.eb) as f64;
        let warps = l.block.div_ceil(warp) as f64;
        // The ring's in-flight discipline: `cp.async` waits to `stages − 2`
        // groups, so `stages − 1` panels fly; register staging one; a bulk
        // ring `stages − 1`.
        let fly = |stages: u64| (stages.saturating_sub(1).max(1) as f64 * pb) as u64;
        // The combine's epilogue: a piece's partials, one `red` a cell a lane,
        // issued where its piece ends — spread over the piece's trips.
        let panels = (s.k / bk).max(1) as f64;
        let reds = match x.trav {
            Traversal::Split {
                s: parts,
                comb: mapal_place::Comb::Atomic,
                ..
            } => (bm * bn) as f64 / warp as f64 / (panels / parts as f64).max(1.0),
            Traversal::Walk { iters, .. } => {
                (bm * bn) as f64 / warp as f64 * 2.0 / (iters.max(1) as f64)
            }
            _ => 0.0,
        };
        let copies = |bytes: f64| bytes / GPU_VEC_BYTES as f64 / warp as f64;
        match self.id {
            NAIVE => {
                if l.tiled.is_some() || l.mma.is_some() {
                    return Err(Refusal::new("not the naive kernel"));
                }
                let cells = param(x, "cells").max(1);
                // The vector-cells kernel (`naive_cells`) reads `b` as one
                // vector of a row's `cells` consecutive lanes at an affine
                // offset a `k` step and `a` once a row: a declared constraint
                // (cost-model ABBA: the 2048 x 2048 x 25 window's mixed-radix
                // `b` at 16 cells a thread stored wrong values).
                if cells > 1 && !gpu::naive_cells_admit(site, cells) {
                    return Err(Refusal::new(
                        "vector cells: affine rows only (no radix, no k split, b lanes unit-stride)",
                    ));
                }
                // A thread folds its cells over `k`: NAIVE_UNROLL loads of
                // `b` (one vector) and one of `a` a trip in flight.
                let trip = NAIVE_UNROLL as f64;
                Ok(Declared {
                    threads: NAIVE_THREADS,
                    // The address, loop and in-flight load budget up to the
                    // registers a lane keeps at full occupancy (the code
                    // generator fills that budget: 32 where an SM holds 2048
                    // threads, 40 at 1536), and the cells' accumulators past
                    // four: `ptxas -v` on sm_80 – sm_120, f32 and bf16 `b`,
                    // 31–40 registers at 1–8 cells, 43–48 at 16 (test
                    // `naive_declared_registers_cover_ptxas`).
                    // (The cell kernel emits no `min-blocks`: no cap.)
                    regs: (g.regs_per_sm / g.max_threads_per_sm.max(1) / 8 * 8)
                        .min(NAIVE_REGS)
                        .max(32 + (cells.saturating_sub(4) * 4).div_ceil(3)),
                    smem: 0,
                    issue: vec![
                        (Class::Math, warps * trip * cells as f64),
                        (Class::Global, warps * trip * 2.0 + reds),
                        (Class::Other, warps * trip),
                    ],
                    in_flight: NAIVE_THREADS * NAIVE_UNROLL * (cells * s.eb + s.ea),
                    fill: 1,
                    chains: cells,
                    groups: NAIVE_THREADS / warp,
                    work: 0.0,
                    whole_stores: false,
                    // A lane loads its cells' `b` and its row's `a` into
                    // registers.
                    mover: mapal_place::Move::Load(GPU_VEC_BYTES.min(cells * s.eb)),
                    store: GPU_VEC_BYTES.min(cells * s.eo),
                    ..Declared::default()
                })
            }
            FMA => {
                let t = l.tiled.ok_or_else(|| Refusal::new("no FMA tile"))?;
                if site.a.clane != 0 || site.b.ci != 0 {
                    return Err(Refusal::new("the FMA tile needs a separable site"));
                }
                if t.held > 0 && t.held < fma_held_regs(t.tm, t.tn, e) {
                    return Err(Refusal::new("held registers below the counted footprint"));
                }
                let key = [
                    t.tm,
                    t.tn,
                    t.tx,
                    t.ty,
                    t.bk,
                    t.stages.min(2),
                    u64::from(t.kmajor),
                    t.pad_a,
                    t.a_row_lanes,
                ];
                // Memoized only inside a search (one site, one rung table).
                let active = SEARCHING.with(|c| c.get());
                let hit = active
                    .then(|| LSU.with(|c| c.borrow().get(&key).copied()))
                    .flatten();
                let lsu = hit.unwrap_or_else(|| {
                    let v = gpu::lsu_per_panel(site, &t, &l.rungs, false);
                    if active {
                        LSU.with(|c| c.borrow_mut().insert(key, v));
                    }
                    v
                }) as f64;
                let ffma = (t.tm * t.tn * t.bk) as f64;
                let regs = match t.held {
                    0 => {
                        (t.tm * t.tn + 2 * (t.tm + t.tn)) * e.div_ceil(4)
                            + crate::profile::FMA_ADDR_REGS
                    }
                    h => h,
                };
                Ok(Declared {
                    threads: t.threads(),
                    regs: regs.min(cap(t.threads())),
                    smem: l.smem,
                    issue: vec![
                        (
                            Class::Math,
                            warps * ffma * if x.op == OP_MULADD { 2.0 } else { 1.0 },
                        ),
                        (Class::Shared, warps * lsu),
                        (Class::Global, reds),
                        (Class::Barrier, warps * if stages > 1 { 1.0 } else { 2.0 }),
                    ],
                    in_flight: fly(stages),
                    fill: stages.saturating_sub(1).max(1),
                    chains: t.tm * t.tn,
                    groups: t.threads().div_ceil(warp),
                    work: 0.0,
                    whole_stores: false,
                    // A `cp.async` ring of whole 16-byte copies, or one panel
                    // staged through registers; a lane stores its `tn`
                    // consecutive cells.
                    mover: match stages {
                        0 | 1 => mapal_place::Move::Load(GPU_VEC_BYTES),
                        _ => mapal_place::Move::Copy(GPU_VEC_BYTES),
                    },
                    barriers: if stages > 1 { 1.0 } else { 2.0 },
                    store: GPU_VEC_BYTES.min(t.tn * s.eo),
                    // A lane's `tn` consecutive cells over `tn` reductions:
                    // each instruction's sectors the next ones revisit — the
                    // probe's contiguous row (`red.add.f32`, the default),
                    // not its sector-a-lane one (no revisit).
                    ..Declared::default()
                })
            }
            MMA => {
                let mt = l.mma.ok_or_else(|| Refusal::new("no warp mma tile"))?;
                let o = g.caps.matrix[x.op - 2];
                // The warp tile's ring: two stages at least (its emitter
                // waits on `stages − 2` groups; one stage, a single-panel
                // fold, computes no product — found by the picks' value
                // test), at most a stage a panel.
                if mt.stages < 2 || mt.stages > s.k / bk {
                    return Err(Refusal::new("stages in [2, k/bk]"));
                }
                // HMMA a trip: the warp tile's instructions per `k` step, a
                // split form's parts each.
                // A split form issues its kept part products a step.
                let products = mt
                    .split
                    .map_or(1.0, |k| mapal_ir::split::SplitForm::of(k).products() as f64);
                let parts = products;
                let ksteps = (bk / o.shape.k.max(1)) as f64;
                let hmma = (mt.mt * mt.nt) as f64 * ksteps * parts;
                // ldmatrix.x4 covers a 16 × 16 bytes-of-b16 tile: `a`'s mt
                // tiles and `b`'s nt / 2 a k step.
                let ldsm = (mt.mt as f64 + (mt.nt as f64 / 2.0).ceil()) * ksteps;
                // The present emitter's two counts (`gpu_mma_tile_at`): the
                // S52 rule's twice the accumulators, the counted
                // accumulators, fragments and addresses — the smaller.
                let acc = mt.mt * mt.nt * 4;
                let frag = (mt.mt * 4 + mt.nt * 2) * if mt.split.is_some() { 2 } else { 1 };
                let counted = (acc + frag + crate::profile::MMA_ADDR_REGS).min(2 * acc);
                Ok(Declared {
                    threads: mt.threads(warp),
                    regs: counted.min(cap(mt.threads(warp))),
                    smem: l.smem,
                    issue: vec![
                        (Class::Matrix, warps * hmma),
                        (Class::Shared, warps * ldsm),
                        (Class::Global, copies(pb) + reds),
                        (
                            Class::Convert,
                            match (mt.split.is_some(), mt.ea > o.fmt.bytes()) {
                                (true, _) | (_, true) => warps * ldsm * 4.0,
                                _ => 0.0,
                            },
                        ),
                        (Class::Barrier, warps),
                    ],
                    in_flight: fly(mt.stages),
                    fill: mt.stages.saturating_sub(1).max(1),
                    chains: mt.mt * mt.nt,
                    groups: mt.warps_x * mt.warps_y,
                    work: (2 * bm * bn * bk) as f64 * products,
                    // `mma_walk_store`: a whole piece stores as the ordered
                    // kernel does.
                    whole_stores: true,
                    // A `cp.async` ring of 16-byte copies; the fragments by
                    // `ldmatrix.x4` (512 bytes a warp; 32-bit operands' lane
                    // loads), one `k` step's ahead of their products; a lane
                    // stores its accumulator fragment's two cells.
                    mover: mapal_place::Move::Copy(GPU_VEC_BYTES),
                    scratch: (warps * ldsm * 512.0) as u64,
                    scratch_fly: (warps * ldsm / ksteps.max(1.0) * 512.0) as u64,
                    scratch_matrix: o.fmt != MmaFmt::Tf32,
                    barriers: 1.0,
                    store: 2 * s.eo,
                    // The accumulator fragment: 8 rows of 32 bytes a warp's
                    // reduction instruction.
                    add_sectors: 8,
                    ..Declared::default()
                })
            }
            _ => {
                let w = l.wgmma.ok_or_else(|| Refusal::new("no warpgroup tile"))?;
                if (x.lowering == WGMMA_WS) != w.ws {
                    return Err(Refusal::new("warp specialization is the lowering's"));
                }
                if !w.n.is_multiple_of(WGMMA_N_STEP) || w.n > WGMMA_N_MAX {
                    return Err(Refusal::new("n a multiple of 8 up to 256"));
                }
                if !w.ws && w.stages < crate::profile::WGMMA_MIN_STAGES {
                    return Err(Refusal::new("stages >= 3 (S53 ring)"));
                }
                let o = g.caps.matrix[x.op - 2];
                // The direct product `c = a·b` reads `b`'s stage as a
                // descriptor operand in either major, which only an
                // instruction with `MatrixOp::mn_major` reads; a K-major-only
                // one (TF32, fp8 wgmma) takes the transposed product, as the
                // rules lower it (`GpuWgmmaTile::direct`). cost-rental: the
                // H100's search picked a direct TF32 tile and ptxas refused
                // its `wgmma.mma_async` (pt_mlp_2048_tf32).
                if w.direct && !o.mn_major {
                    return Err(Refusal::new(
                        "a K-major-only instruction takes the transposed product",
                    ));
                }
                let ksteps = (bk / o.shape.k.max(1)) as f64;
                let acc = 64 * w.n / (WARPGROUP_WARPS * warp);
                // Under `setmaxnreg` the roles split the block's pool: the
                // lanes' mean is what the block takes from the file.
                let regs = match w.realloc {
                    Some((p, c)) => {
                        let wg = WARPGROUP_WARPS * warp;
                        (wg * p + w.groups * wg * c).div_ceil(w.threads(warp))
                    }
                    None => {
                        acc + match w.ws {
                            true => crate::profile::WGMMA_WS_ADDR_REGS,
                            false => crate::profile::WGMMA_ADDR_REGS,
                        }
                    }
                };
                Ok(Declared {
                    threads: w.threads(warp),
                    regs,
                    smem: l.smem,
                    issue: vec![
                        (Class::Matrix, w.groups as f64 * ksteps),
                        (
                            Class::Global,
                            match w.ws {
                                true => 1.0 + w.ws_b_boxes() as f64,
                                false => copies(pb),
                            },
                        ),
                        (Class::Barrier, w.groups as f64 + 1.0),
                    ],
                    in_flight: fly(w.stages),
                    fill: w.stages.saturating_sub(1).max(1),
                    chains: 1,
                    groups: w.groups * WARPGROUP_WARPS,
                    work: 0.0,
                    whole_stores: false,
                    // The warp-specialized tile's panels by tensor boxes, the
                    // plain one's by `cp.async`; the products read the
                    // scratchpad by descriptor (no lane loads).
                    mover: match w.ws {
                        true => mapal_place::Move::Bulk(
                            (pb as u64 / (1 + w.ws_b_boxes()).max(1)).max(1),
                        ),
                        false => mapal_place::Move::Copy(GPU_VEC_BYTES),
                    },
                    barriers: 1.0,
                    store: 2 * s.eo,
                    add_sectors: 8,
                    ..Declared::default()
                })
            }
        }
    }
}
