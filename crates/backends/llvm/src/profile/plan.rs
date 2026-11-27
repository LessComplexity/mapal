//! The plan-cost model: `PlanShape`, `PlanCost`, `GpuTile` and the `Gpu` physical-property queries.

use super::*;

/// How a candidate plan combines its partial folds (plan-cost §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanComb {
    None,
    Atomic,
    Workspace,
    StreamK,
}

/// **One candidate plan of a tile site, as the cost model reads it**
/// (plan-cost design §2): the site's extents and the launch the candidate
/// emits. Built by `func/gpu.rs` from its `Launch`; plain numbers, so the
/// harness's mirror (`benches/gpu/plancost.py`) reads them off `; mapal.cand`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanShape {
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    /// Element bytes (the product's and the output's).
    pub e: u64,
    /// ADR-0043: bytes one element of operand `a` / `b` occupies in memory —
    /// `e`, or a narrow format's 2 (the operand panels' traffic).
    pub ea: u64,
    pub eb: u64,
    /// One block's output tile.
    pub bm: u64,
    pub bn: u64,
    pub threads: u64,
    /// Blocks one SM holds at this geometry.
    pub per_sm: u64,
    /// A thread's accumulators (`tm · tn`): its independent FMA chains.
    pub acc: u64,
    /// Output tiles (`gx · gy`).
    pub tiles: u64,
    /// Blocks launched: `tiles · s`, or stream-K's `ctas`.
    pub blocks: u64,
    /// The `k` depth one block folds: `k / s`, or stream-K's `iters · bk`.
    pub kb: u64,
    /// Partitions (`S`); 1 unsplit and for stream-K.
    pub s: u64,
    pub comb: PlanComb,
    /// plan-cost-terms: the `k` one panel trip folds (the tile's `bk`; the
    /// naive kernel's [`NAIVE_UNROLL`]) and the panels a block keeps in
    /// flight (`stages − 1` under `cp.async`, 1 register-staged or naive).
    pub bk: u64,
    pub fly: u64,
    /// tile-efficiency T4: the shared/LSU warp-instructions one thread issues
    /// per panel trip, counted from the geometry as the emitter issues them —
    /// `bk` steps of fragment `LDS`, the panel copies (`LDG` / `LDGSTS`) and
    /// their `STS` — against `acc · bk` FFMA. 0 for the naive kernel (no term).
    pub lsu: u64,
}

/// plan-cost-terms: the `k` steps one trip of the naive kernel's loop issues
/// before its first add needs a load — LLVM's unroll of that loop, read off
/// the PTX (`mm_1x1024x4096` rtx-4090: 8 `ld.global.nc` of `b` per trip).
/// ponytail: LLVM's choice, not Mapal's; emit `llvm.loop.unroll.count` if a
/// later LLVM unrolls otherwise.
pub const NAIVE_UNROLL: u64 = 8;

/// [`TargetProfile::site_ns`]'s terms, each printed on a `; mapal.cand` line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanCost {
    /// Waves of the launch and the blocks of its last one.
    pub waves: u64,
    pub tail: u64,
    /// Warps per scheduler in a full wave (the last one's when it is the only).
    pub wps: f64,
    /// Σ over waves of the FFMA time, of the L2 panel feed; the DRAM bound.
    pub f_ns: f64,
    pub l_ns: f64,
    pub d_ns: f64,
    /// plan-cost-terms: Σ over waves of a block's latency chain — its panel
    /// trips at `fly` loads in flight (Little's law, k-depth); 0 without
    /// `lat_ns`.
    pub c_ns: f64,
    /// tile-efficiency T4: Σ over waves of the busiest SM's LSU issue time —
    /// the floor `f_ns` already takes ([`DeviceCost::lsu_ipc_pm`]).
    pub u_ns: f64,
    pub kernel_ns: f64,
    pub combine_ns: f64,
    /// Zero pass and launch idle (D2).
    pub fixed_ns: f64,
    pub launches: u64,
    pub zeros: u64,
    pub atomics: u64,
    pub ns: f64,
}

/// The FMA tile's achieved fraction of FFMA peak with `p` independent FMA
/// chains per warp scheduler: `eff_max · p / (p + eff_half)` — latency hidden
/// by warps and by a thread's accumulators alike (the 2-parameter `eff` of
/// plan-cost §2, fitted by `benches/gpu/plancost.py`).
pub fn plan_eff(d: &DeviceCost, p: f64) -> f64 {
    d.eff_max_pm as f64 / 1000.0 * p / (p + d.eff_half as f64)
}

/// plan-cost-terms: [`plan_eff`] at `wps` warps per scheduler of a thread's
/// `acc` chains, on a part with `lat_ns`: below one warp a scheduler only
/// issues for the fraction of schedulers that hold one (`min(wps, 1)`), and
/// each of those hides latency with its own warps' chains (`max(wps, 1)·acc`)
/// — it falls with the warps instead of staying flat where `wps·acc` alone
/// saturates.
pub fn plan_eff_at(d: &DeviceCost, wps: f64, acc: f64) -> f64 {
    let q = wps.max(1.0) * acc;
    d.eff_max_pm as f64 / 1000.0 * wps.min(1.0) * q / (q + d.eff_half as f64)
}

/// Bytes one thread moves in its widest global or shared access — `ld.v4.b32`.
/// An ISA rule on every NVIDIA part, so it is not a profile field.
pub const GPU_VEC_BYTES: u64 = 16;

/// The GPU tile geometry for one tile site (any `map` folding affine reads over `k`), every number derived from
/// [`Gpu`] facts and the site's own extents by [`TargetProfile::gpu_tile`].
///
/// One block computes a `bm × bn` tile of the output (`bm = ty·tm`,
/// `bn = tx·tn`); each of its `tx·ty` threads owns a `tm × tn` register tile;
/// the reduction axis is staged through shared memory `bk` deep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTile {
    pub tm: u64,
    pub tn: u64,
    pub tx: u64,
    pub ty: u64,
    pub bk: u64,
    /// Blocks resident on one SM at this geometry.
    pub blocks_per_sm: u64,
    /// Shared-memory panel buffers: 1 for the register-staged kernel, ≥ 2 when
    /// panels stream in with `cp.async` ([`TargetProfile::gpu_tile_async`]).
    pub stages: u64,
    /// Words of padding on each transposed-`a` panel row (async staging, and
    /// register staging under the S53 `pad` rung).
    pub pad_a: u64,
    /// Lanes that load one `a` panel row on the register-staged vector path
    /// ([`TargetProfile::gpu_tile_sync_a`]); 0 = all `bk / lanes` of them, the
    /// pre-S53 map.
    pub a_row_lanes: u64,
    /// Registers a thread holds when the whole-file rule derived this tile
    /// ([`fma_held_regs`], S53 `wide-fma`); 0 under the S52 half-the-file rule.
    pub held: u64,
    /// tile-efficiency T2 (`kmajor-a`): `a` is staged untransposed,
    /// `bm` rows of `bk + pad_a` words, by 16-byte copies.
    pub kmajor: bool,
}

/// Registers an FMA-tile thread spends on addresses and loop state, beyond
/// accumulators and fragments. Policy, read off ptxas reports (S52 v6): the
/// 8×8 cp.async kernel without the block cap used 125 (4090) / 126 (5090)
/// registers for 64 accumulators + 32 fragments ⇒ 29–30.
pub const FMA_ADDR_REGS: u64 = 32;

/// What an FMA-tile thread holds, in 32-bit registers: `tm × tn`
/// accumulators, both buffers of its `tm + tn` fragments, and
/// [`FMA_ADDR_REGS`] — the `wide` rule's count, applied to the FMA tile.
pub fn fma_held_regs(tm: u64, tn: u64, elem_bytes: u64) -> u64 {
    (tm * tn + 2 * (tm + tn)) * elem_bytes.div_ceil(4) + FMA_ADDR_REGS
}

/// [`TargetProfile::gpu_tile`] step 1 (S52 rule): the largest square
/// power-of-two register tile side `t`, at least one vector, whose
/// accumulators fit half of `regs_per_thread`; and its per-thread register
/// estimate, twice the accumulators.
pub(super) fn square_register_tile(g: &Gpu, elem_bytes: u64) -> (u64, u64) {
    let regs_per_elem = elem_bytes.div_ceil(4);
    let mut t = (GPU_VEC_BYTES / elem_bytes).max(1);
    while (2 * t) * (2 * t) * regs_per_elem <= g.regs_per_thread / 2 {
        t *= 2;
    }
    (t, 2 * t * t * regs_per_elem)
}

/// [`TargetProfile::gpu_tile`] step 2: threads per block — the resident-thread
/// budget at `regs_est` registers a thread, split across
/// [`GPU_RESIDENT_BLOCKS`], capped by the block limit.
pub(super) fn block_threads(g: &Gpu, regs_est: u64) -> u64 {
    (g.regs_per_sm / regs_est / g.geom.resident).min(g.max_threads_per_block)
}

/// Blocks of `threads` threads at `regs` registers a thread one SM holds: the
/// register file, the thread and the block-slot limits (every tile's step 5).
pub(super) fn residency(g: &Gpu, regs: u64, threads: u64) -> u64 {
    (g.regs_per_sm / regs / threads)
        .min(g.max_threads_per_sm / threads)
        .min(g.max_blocks_per_sm)
}

impl Gpu {
    /// The warp `mma` product and `cp.async` ([`Caps`]): the staged FMA
    /// tile's `cp.async` panels and the TF32 `mma.sync` kernel.
    pub fn has_mma_sync(&self) -> bool {
        self.caps.matrix_op(MmaFmt::Tf32, Home::Regs).is_some() && self.caps.cp_async
    }

    /// Warp schedulers per SM ([`Self::schedulers`] field; 1 when not given,
    /// which leaves the fill rule no stronger than "a block per SM").
    pub fn schedulers(&self) -> u64 {
        self.schedulers.max(1)
    }

    /// `griddepcontrol` ([`Caps::griddepcontrol`]).
    pub fn has_griddepcontrol(&self) -> bool {
        self.caps.griddepcontrol
    }

    /// kernel-merging `calls`: the member bodies' change a persistent chain
    /// costs, in SM cycles, and whether this part measured it — its own
    /// [`DeviceCost::coop_body_cycles`], else the rtx-5080's measurement
    /// ([`COOP_BODY_CYCLES_RTX_5080`]): derived, never unpriced.
    pub fn coop_body_cycles(&self) -> (u64, bool) {
        match self.cost.map_or(0, |d| d.coop_body_cycles) {
            0 => (COOP_BODY_CYCLES_RTX_5080, false),
            c => (c, true),
        }
    }

    /// open-rules `calls`: **the boundary costs a chain is priced on** — the
    /// part's measured grid barrier, dependent, programmatic and cooperative
    /// launches ([`DeviceCost::gsync_ns`], `dep_launch_ns`, `pdl_launch_ns`,
    /// `coop_launch_ns`) where `calib.cu gsync` measured them, else derived
    /// as the body's price is: the rtx-5080's measured cycles
    /// ([`CHAIN_CYCLES_RTX_5080`]) at this part's clock, the programmatic
    /// launch only where the part has `griddepcontrol`. A part nobody
    /// calibrated (the rtx-5090, the h100-pcie) is priced, not left out; its
    /// measurement replaces the derivation.
    pub fn chain_cost(&self) -> (DeviceCost, bool) {
        let mut d = self.cost.unwrap_or(NO_PLAN);
        // review-fixes-2: the process's first cooperative launch (host time,
        // the driver's one-time setup) where measured, else the rtx-5080's
        // as measured ([`SLICE_RTX_5080`], as [`Self::slice_cost`] takes
        // it) — a chain is never priced as if it were free.
        if d.coop_first_ns == 0 {
            d.coop_first_ns = SLICE_RTX_5080[2];
        }
        if d.gsync_ns > 0 && d.dep_launch_ns > 0 {
            return (d, true);
        }
        let at = |c: u64| c * 1_000_000 / self.clock_khz.max(1);
        let [g, dep, pdl, coop] = CHAIN_CYCLES_RTX_5080;
        (
            DeviceCost {
                gsync_ns: at(g),
                gsync_block_ps: 0,
                dep_launch_ns: at(dep),
                dep_launch_block_ps: 0,
                pdl_launch_ns: if self.has_griddepcontrol() {
                    at(pdl)
                } else {
                    0
                },
                coop_launch_ns: at(coop),
                ..d
            },
            false,
        )
    }

    /// open-rules `row-slices`: **the costs a row's slices are priced on** —
    /// [`Self::chain_cost`]'s boundaries, and the walk's per-element cost
    /// (`row_iter_ns`), a restored partial's load latency (`lat_ns`) and the
    /// process's first cooperative launch (`coop_first_ns`) where measured,
    /// else derived from the rtx-5080's ([`SLICE_RTX_5080`]): the walk and
    /// the latency in its cycles at this part's clock, the first launch (the
    /// driver's one-time setup on the host) as measured. Without them no
    /// rented part sliced a row (cross-arch 60f0f8ce: lsm 1.7–11× the best
    /// competitor, one block a row).
    pub fn slice_cost(&self) -> DeviceCost {
        let (mut d, _) = self.chain_cost();
        let at = |c: u64| c * 1_000_000 / self.clock_khz.max(1);
        let [walk, lat, first] = SLICE_RTX_5080;
        if d.row_iter_ns == 0 {
            d.row_iter_ns = at(walk);
        }
        // bugs-properties: a DRAM load's latency is ns, not SM cycles.
        if d.lat_ns == 0 {
            d.lat_ns = lat;
        }
        if d.coop_first_ns == 0 {
            d.coop_first_ns = first;
        }
        d
    }

    /// flash-attention `stream` (P1): one block barrier on a block's
    /// critical path, ns — measured ([`DeviceCost::bar_ns`]), else the SM's
    /// barrier cycles at this part's clock ([`Self::bar_cycles`]).
    pub fn bar_ns(&self) -> u64 {
        match self.cost.map_or(0, |d| d.bar_ns) {
            0 => self.bar_cycles * 1_000_000 / self.clock_khz.max(1),
            x => x,
        }
    }

    /// [`Self::coop_body_cycles`] in ns at this part's clock
    /// ([`Self::clock_khz`], the clock it sustains under load): a part with a
    /// slower clock spends more time on the same cycles.
    pub fn coop_body_ns(&self) -> f64 {
        self.coop_body_cycles().0 as f64 * 1e6 / self.clock_khz.max(1) as f64
    }

    /// hardware-properties: the FFMA peak, flops per ns (`sms · fp32_lanes ·
    /// 2 · clock`); 0 when a factor is not given.
    pub fn ffma_flops_per_ns(&self) -> u64 {
        self.sms * self.fp32_lanes * 2 * self.clock_khz / 1_000_000
    }

    /// precision `split-rate`: **the FFMA tile's own achieved rate**, flops
    /// per ns — the measured compute-bound rate of the shipped FMA tile
    /// ([`DeviceCost::flops_per_ns`]: Mapal's realization, not the silicon),
    /// at most the FFMA peak; the peak where the part measured none. What a
    /// realization that replaces the FMA tile is weighed against: on a part
    /// whose FP32 lanes issue a warp FFMA every clock (128 lanes on 4
    /// schedulers) the tile is issue-bound — the RTX 5080's SASS issues 1.08–
    /// 1.23 slots an FFMA (LDS, STS, LDG, address) and its register-staged
    /// panel's load latency is exposed (gaps-disasm C3-fma) — so it realizes
    /// 0.54–0.68 of the peak on the prompt's projections (32200 of 52 700
    /// flops a ns measured); the A100's 64 lanes take an FFMA every two
    /// clocks, its tile 0.92 of the peak (5fcb401b, 4096² × 4096 3.832 ms).
    pub fn ffma_tile_flops_per_ns(&self) -> u64 {
        let peak = self.ffma_flops_per_ns();
        match self.cost.map_or(0, |d| d.flops_per_ns) {
            0 => peak,
            r if peak == 0 => r,
            r => r.min(peak),
        }
    }

    /// f32-tensor-cores P1: flops one SM issues a clock through `op`
    /// ([`Self::mma_tf32_per_clk`] and its siblings, by the operand format
    /// and where `a` lives); 0 for a product the part has no rate for.
    pub fn matrix_flops_per_clk(&self, op: &MatrixOp) -> u64 {
        // The issuing group and `b`'s home name the pipe: a warp's product
        // takes both operands from registers, a warpgroup's `b` from a
        // descriptor (its `a` from either home: one rate, wgmma-16bit).
        let group = op.b == Home::Smem && op.acc == Home::Regs;
        match (op.fmt, group) {
            (MmaFmt::Tf32, false) => self.mma_tf32_per_clk,
            (MmaFmt::Bf16 | MmaFmt::F16, false) => self.mma_half_per_clk,
            (MmaFmt::Tf32, true) => self.wgmma_tf32_per_clk,
            (MmaFmt::Bf16 | MmaFmt::F16, true) => self.wgmma_half_per_clk,
            _ => 0,
        }
    }

    /// f32-tensor-cores P1: `op`'s peak, flops per ns (`sms ·` that `·
    /// clock`) — the tensor rate the split rule weighs against
    /// [`Self::ffma_flops_per_ns`].
    pub fn matrix_flops_per_ns(&self, op: &MatrixOp) -> u64 {
        self.sms * self.matrix_flops_per_clk(op) * self.clock_khz / 1_000_000
    }

    /// hardware-properties: the DRAM peak, bytes per ns (double data rate over
    /// the bus: `2 · mem_clock · bus_bits / 8`); 0 when not given.
    pub fn dram_peak_bytes_per_ns(&self) -> u64 {
        2 * self.mem_clock_khz * self.mem_bus_bits / 8 / 1_000_000
    }

    /// stream-fly: **the bytes a streaming grid keeps in flight per SM to
    /// read DRAM at the part's rate** — [`DeviceCost::fly_bytes_per_sm`]
    /// where measured, else Little's law at the measured loaded latency
    /// (`lat_ns`) over the DRAM peak an SM shares (its lower bound); 0 when
    /// neither is known (the rule then asks only that every SM streams).
    pub fn stream_fly_per_sm(&self) -> u64 {
        let d = self.cost.unwrap_or(NO_PLAN);
        match d.fly_bytes_per_sm {
            0 => self.dram_peak_bytes_per_ns() * d.lat_ns / self.sms.max(1),
            f => f,
        }
    }

    /// integration (tune-context): **the largest cyclic working set the
    /// L2 keeps** — [`DeviceCost::l2_keep_bytes`] where the profile sets it
    /// (measured: [`l2_keep_of_sweep`] on the part's `l2keep.cu` sweep — the
    /// RTX 5080, the RTX 3090; the A100's and H100's half are placeholders
    /// below their sweep's first point, 1.0 ×, open until the sweep from 0.25
    /// × runs), else the nominal L2 ([`Self::l2_bytes`]). A site whose operand's working set
    /// (`residence_plan`) is within it is re-read from L2 across the
    /// program's calls: its trial does not evict the L2.
    pub fn l2_keep_bytes(&self) -> u64 {
        match self.cost.unwrap_or(NO_PLAN).l2_keep_bytes {
            0 => self.l2_bytes,
            b => b,
        }
    }

    /// review-fixes: **the plan rates a part is priced on, never unpriced**
    /// — each rate its profile measured ([`DeviceCost`]), every other one
    /// derived from the rtx-5080's measurement at this part's own physical
    /// properties, as [`Self::chain_cost`] and [`Self::bar_ns`] derive
    /// theirs: the FMA tile at the 5080's fraction of its FFMA peak
    /// ([`Self::ffma_flops_per_ns`]); the L2 read rate at its bytes an SM a
    /// clock (`sms · clock_khz`); the DRAM rate at its fraction of the DRAM
    /// peak ([`Self::dram_peak_bytes_per_ns`]); `red.global.add.f32` at its
    /// adds per L2 byte (atomics resolve in the L2 slices: 400 / 4500 on the
    /// 5080 and the 4090, 560 / 6500 on the 5090 — one ratio on three
    /// parts); the zero pass's and the in-stream gap's device time in the
    /// 5080's cycles at this part's clock, the loaded latency in its ns (a
    /// DRAM round trip, not SM cycles: bugs-properties); the host's
    /// issue as measured (a host property); the tile's fitted efficiency
    /// and the LSU issue a clock as the 5080's (per-SM fractions). The
    /// fields no plan price reads, and those a consumer defaults itself
    /// (`fill_bytes_per_ns`: the L2 rate), stay as given. A factor the part
    /// does not give (no clock, no FP32 lanes, no DRAM interface) derives 0:
    /// still unpriced there. What `streamk-price` and the `stream` rule read
    /// on the 3090 / A100 / H100 profiles, which measured only some rates
    /// (sweep-gemm-2, sweep-attention: until then the panel-count rule and
    /// the split realization stood there, "off where unmeasured").
    pub fn plan_cost(&self) -> DeviceCost {
        let mut d = self.cost.unwrap_or(NO_PLAN);
        let r = RTX_5080.gpu().expect("the reference part is a GPU");
        let rc = r.cost.expect("the reference part is calibrated");
        let scale = |x: u64, num: u64, den: u64| -> u64 {
            match den {
                0 => 0,
                _ => (x as u128 * num as u128 / den as u128) as u64,
            }
        };
        let at = |ns: u64| scale(ns, r.clock_khz, self.clock_khz);
        let or = |v: &mut u64, x: u64| {
            if *v == 0 {
                *v = x;
            }
        };
        or(
            &mut d.flops_per_ns,
            scale(
                self.ffma_flops_per_ns(),
                rc.flops_per_ns,
                r.ffma_flops_per_ns(),
            ),
        );
        or(
            &mut d.l2_bytes_per_ns,
            scale(
                rc.l2_bytes_per_ns,
                self.sms * self.clock_khz,
                r.sms * r.clock_khz,
            ),
        );
        or(
            &mut d.dram_bytes_per_ns,
            scale(
                self.dram_peak_bytes_per_ns(),
                rc.dram_bytes_per_ns,
                r.dram_peak_bytes_per_ns(),
            ),
        );
        let l2 = d.l2_bytes_per_ns;
        or(
            &mut d.atomic_ops_per_ns,
            scale(l2, rc.atomic_ops_per_ns, rc.l2_bytes_per_ns),
        );
        if self.clock_khz > 0 {
            or(&mut d.zero_ns, at(rc.zero_ns));
            or(&mut d.gap_ns, at(rc.gap_ns));
        }
        // bugs-properties: the loaded latency is a DRAM round trip in ns, not
        // SM cycles — the 5080's 573 ns at the part's clock read 899 / 995 /
        // 776 ns on the 3090 / A100 / H100 against 452 / 728 / 629 measured
        // (cross-arch 5fcb401b, `calib.cu bw`); unscaled it is within 0.8–1.5×
        // of every measured part.
        or(&mut d.lat_ns, rc.lat_ns);
        or(&mut d.issue_ns, rc.issue_ns);
        or(&mut d.launch_ns, rc.launch_ns);
        or(&mut d.eff_max_pm, rc.eff_max_pm);
        or(&mut d.eff_half, rc.eff_half);
        or(&mut d.lsu_ipc_pm, rc.lsu_ipc_pm);
        d
    }

    /// hopper-wgmma-tma: **the bulk-copy engine's ns per copy** —
    /// [`DeviceCost::tma_copy_ns`] where measured, else the loaded panel
    /// latency over the copies one SM's engine overlaps
    /// ([`TMA_COPIES_IN_FLIGHT`]); 0 when neither is known.
    pub fn tma_copy_ns(&self) -> u64 {
        let d = self.cost.unwrap_or(NO_PLAN);
        match d.tma_copy_ns {
            0 => d.lat_ns / TMA_COPIES_IN_FLIGHT,
            t => t,
        }
    }

    /// hopper-wgmma-tma R1: **the time a block's tensor cores take to consume
    /// one stage of `flops` flops**, ns, at the part's per-SM matrix rate
    /// (`blocks` resident blocks sharing it): the time a stage's copies must
    /// issue within. The rate is f32-tensor-cores P1's
    /// ([`Self::matrix_flops_per_clk`], the fastest of the part's products in
    /// `fmt`); the FFMA rate stands in where no matrix rate is given (it is
    /// the lower one: the price errs toward the copies).
    pub fn stage_consume_ns(&self, flops: u64, blocks: u64, op: &MatrixOp) -> u64 {
        let matrix = self.matrix_flops_per_clk(op) * self.clock_khz / 1_000_000;
        let per_sm = match matrix {
            0 => self.ffma_flops_per_ns() / self.sms.max(1),
            m => m,
        };
        flops * blocks.max(1) / per_sm.max(1)
    }

    /// hardware-properties: **the machine balance** — FFMA flops the part
    /// issues per byte its DRAM delivers, per mille (`None` without both
    /// rates). How much arithmetic a loaded byte must carry before the
    /// operand stream, not the FMA pipe, bounds a tile.
    pub fn ffma_per_dram_byte_pm(&self) -> Option<u64> {
        let d = self.dram_peak_bytes_per_ns();
        (d > 0 && self.ffma_flops_per_ns() > 0).then(|| self.ffma_flops_per_ns() * 1000 / d)
    }

    /// hardware-properties: 32-bit registers each thread holds when the SM is
    /// full of threads (`regs_per_sm / max_threads_per_sm`): the register cap
    /// a residency bound by threads leaves a thread.
    pub fn regs_per_resident_thread(&self) -> u64 {
        self.regs_per_sm / self.max_threads_per_sm.max(1)
    }

    /// hardware-properties: the L2 bytes per SM — the share of the shared
    /// cache one SM's resident blocks keep their panels in.
    pub fn l2_bytes_per_sm(&self) -> u64 {
        self.l2_bytes / self.sms.max(1)
    }

    /// hardware-properties: the DRAM peak per SM, bytes per µs (what filling
    /// one more SM can add to a stream the grid leaves under-filled).
    pub fn dram_bytes_per_us_per_sm(&self) -> u64 {
        self.dram_peak_bytes_per_ns() * 1000 / self.sms.max(1)
    }

    /// The warps a grid needs before every scheduler holds
    /// [`GPU_RESIDENT_BLOCKS`] of them — that constant's reason applied per
    /// scheduler: one warp issues while the other waits on a load.
    pub fn fill_warps(&self) -> u64 {
        self.geom.resident * self.schedulers() * self.sms
    }

    /// How full a grid of `blocks × threads` keeps the part: warps per warp
    /// scheduler, across every SM.
    pub fn warps_per_scheduler(&self, blocks: u64, threads: u64) -> u64 {
        blocks * threads.div_ceil(self.warp) / (self.sms * self.schedulers())
    }

    /// The warpgroup TF32 product, operands from shared memory and the
    /// accumulator in registers ([`Caps::matrix`]: `wgmma`).
    pub fn has_wgmma(&self) -> bool {
        self.caps
            .matrix_op(MmaFmt::Tf32, Home::Smem)
            .is_some_and(|o| o.acc == Home::Regs)
    }

    /// **The block-row group height a tile grid is walked in (S54 `raster`)**,
    /// for a `rows × c × k` site in `bm × bn` blocks, `wave` of them in flight
    /// (`sms · blocks_per_sm`), `elem` bytes per element; 1 = the hardware's
    /// row-major order, the OFF text.
    ///
    /// The grid is `gx = c / bn` by `gy = rows / bm`: every `a` row-panel is
    /// read by `gx` blocks, every `b` column-panel by `gy` (the reuse counts).
    /// Row-major order (`ctaid.x` fastest) runs an `a` panel's readers back to
    /// back but puts a `b` panel's `gx` launches apart, so one wave reads
    /// `⌈wave / gx⌉` `a` panels and `min(wave, gx)` `b` panels. Walking `h`
    /// block-rows per column instead, a wave reads `h` `a` panels and
    /// `⌈wave / h⌉` `b` panels: the `h` minimising those bytes,
    /// `(h·bm + ⌈wave / h⌉·bn)·k·elem` — a power of two dividing `gy`, the
    /// smaller on a tie.
    ///
    /// `gate` (rung `l2-gate`): keep row-major while one block's panels over
    /// the whole reduction, `(bm + bn)·k·elem`, fit [`Gpu::l2_bytes`] — what the
    /// part streams between two launch neighbours, the readers of one `a`
    /// slab, so their sharing survives in L2. The measured anchor: every part
    /// with it in L2 holds its 4096 ratio to cuBLAS at 8192; the 3090 at 8192
    /// (8 MB of panels, 6 MB of L2) is the one that drops (S54
    /// `baseline-tables.md`). A grid that is one wave keeps row-major: every
    /// block is in flight at once, whatever the order.
    ///
    /// gapfix tensor-tile (C11): **and while row-major's `b` re-reads leave
    /// the site's DRAM time inside its pipe time** ([`Self::b_rereads`]).
    /// Row-major puts a `b` panel's readers `gx` launches apart; between two
    /// of them the part streams the panels of that distance, and past what
    /// the L2 keeps each later block-row (`gx > wave`) or wave re-reads `b`
    /// from DRAM. That is what the one-block gate above never sees: the RTX
    /// 5080's 2048 × 28672 × 4096 bf16 projection (2 MB of panels, 235 MB of
    /// `b` past a 66 MB keep, 224 blocks a row against a 168-block wave, 15
    /// re-reads: 4.24 ms of DRAM against a 4.57 ms pipe) ran row-major
    /// 4.571 ms against 3.991 grouped (gaps-disasm C11); a compute-bound
    /// site re-reading `b` keeps row-major (below [`DRAM_BOUND_PM`]: the
    /// 3090's and 4090's square FMA tiles, the arms null). `eb`: bytes of
    /// one `b` element as staged; `flops_per_ns`: the tile's rate (0:
    /// unpriced, the re-read alone decides).
    #[allow(clippy::too_many_arguments)]
    pub fn raster(
        &self,
        bm: u64,
        bn: u64,
        wave: u64,
        rows: u64,
        c: u64,
        k: u64,
        elem: u64,
        gate: bool,
        eb: u64,
        flops_per_ns: u64,
    ) -> u64 {
        let (gx, gy) = (c / bn.max(1), rows / bm.max(1));
        let wave = wave.max(1);
        let panels_fit = (bm + bn) * k * elem <= self.l2_bytes;
        let rereads = self.b_rereads(bm, bn, wave, rows, c, k, eb);
        // Row-major's DRAM bytes (`b` once and its re-reads, `a` once) at the
        // DRAM rate against the pipe's time for `2·rows·c·k` flops.
        let bytes = (rereads + 1) * c * k * eb + rows * k * eb;
        let reread_bound = rereads > 0
            && (flops_per_ns == 0 || self.dram_bound(bytes, 2 * rows * c * k, flops_per_ns));
        if gx < 2 || gy < 2 || gx * gy <= wave || (gate && panels_fit && !reread_bound) {
            return 1;
        }
        (0..=gy.ilog2())
            .map(|p| 1u64 << p)
            .filter(|h| gy.is_multiple_of(*h))
            .min_by_key(|&h| h * bm + wave.div_ceil(h) * bn)
            .unwrap_or(1)
    }

    /// gapfix tensor-tile: **a site is DRAM-bound** — its `bytes` at the
    /// part's DRAM rate ([`Self::plan_cost`]) take at least
    /// [`DRAM_BOUND_PM`] of its `flops` at the pipe's `flops_per_ns`: the
    /// load latency is no longer hidden behind the products. What C11's
    /// raster gate ([`Self::raster`]) reads.
    pub fn dram_bound(&self, bytes: u64, flops: u64, flops_per_ns: u64) -> bool {
        let dram = self.plan_cost().dram_bytes_per_ns.max(1);
        u128::from(bytes) * u128::from(flops_per_ns) * 1000
            >= u128::from(flops) * u128::from(dram) * u128::from(DRAM_BOUND_PM)
    }

    /// gapfix tensor-tile (C11): **how many more times than once a
    /// row-major walk of a `rows × c × k` grid of `bm × bn` blocks reads its
    /// `b` operand from DRAM** — 0 where the panels of a `b` panel's reuse
    /// distance fit [`Self::l2_keep_bytes`]. Row-major (`ctaid.x` fastest)
    /// puts the readers of one `b` panel `gx = c / bn` launches apart: past
    /// a wave (`gx > wave`) the next one starts after every `b` panel
    /// streamed (`gx·bn·k·eb` bytes) and each of the `gy` block-rows reads
    /// `b` again; within a wave it starts `gx / wave` of a block's run later,
    /// after `(gx²/wave·bn + bm)·k·eb` bytes (the wave's `b` panels over that
    /// share of its run, and its own `a` panel), and each further wave of
    /// the grid reads `b` again.
    #[allow(clippy::too_many_arguments)]
    pub fn b_rereads(
        &self,
        bm: u64,
        bn: u64,
        wave: u64,
        rows: u64,
        c: u64,
        k: u64,
        eb: u64,
    ) -> u64 {
        let (gx, gy) = (c / bn.max(1), rows / bm.max(1));
        let wave = wave.max(1);
        if gx == 0 || gy < 2 || gx * gy <= wave {
            return 0;
        }
        let (distance, reads) = match gx > wave {
            true => (gx * bn * k * eb, gy),
            false => (
                (gx * gx * bn / wave + bm) * k * eb,
                (gx * gy).div_ceil(wave),
            ),
        };
        match distance > self.l2_keep_bytes() {
            true => reads - 1,
            false => 0,
        }
    }
}
