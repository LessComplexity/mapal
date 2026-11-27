//! `DeviceCost`: the measured device rates and the cost-model constants.

use super::*;

/// **Mapal's own achieved device rates on one part (S62 `place`)** — not the
/// silicon's peak: the honest number for a gate choosing between two Mapal
/// realizations of one site. Each field cites the calibration run it was read
/// from ([`TargetProfile::device_pays`] is the only reader).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCost {
    /// Compute-bound rate of the shipped FMA tile kernel, flops per ns.
    pub flops_per_ns: u64,
    /// Memory-bound rate of a one-thread-per-cell kernel, bytes per ns.
    pub bytes_per_ns: u64,
    /// What placing one more site on the device costs before its arithmetic:
    /// the launch, its operands' generation and its completion, in ns.
    pub launch_ns: u64,
    /// plan-cost: the rates [`TargetProfile::site_ns`] reads; every field 0
    /// ([`NO_PLAN`]) on a part nobody calibrated, which keeps its plans the
    /// rules' (decision D8). Host issue time of one asynchronous op, issued
    /// back to back on one stream, in ns (D2).
    pub issue_ns: u64,
    /// The device's idle time between two dependent ops of one stream when
    /// the host issue is hidden behind the earlier op, in ns (D2).
    pub gap_ns: u64,
    /// A zero pass's device time beyond its bytes (`cuMemsetD8Async`), in ns.
    pub zero_ns: u64,
    /// `red.global.add.f32` adders into distinct outputs, ops per ns.
    pub atomic_ops_per_ns: u64,
    /// L2-resident read rate: the panel feed of a tile kernel, bytes per ns.
    pub l2_bytes_per_ns: u64,
    /// DRAM read rate of a streaming kernel, bytes per ns (`bytes_per_ns` is
    /// Mapal's achieved placement rate, about half of it).
    pub dram_bytes_per_ns: u64,
    /// Fitted: the FMA tile's achieved fraction of FFMA peak at saturation,
    /// per mille ([`plan_eff`]).
    pub eff_max_pm: u64,
    /// Fitted: the independent FMA chains per warp scheduler (warps × a
    /// thread's accumulators) at which the tile reaches half of `eff_max_pm`.
    pub eff_half: u64,
    /// plan-cost-terms: the latency of one panel load under load, in ns —
    /// Little's law's `L`: a block keeps `fly` panels in flight, so it moves at
    /// most `fly · panel / lat_ns` bytes per ns and folds a partition in no
    /// less than `(trips + fly − 1) · lat_ns / fly` ([`PlanCost::c_ns`]). 0 =
    /// not given: no latency term (the sprint `plan-cost` model).
    pub lat_ns: u64,
    /// tile-efficiency T4: shared/LSU warp-instructions one SM issues per
    /// clock, per mille — the floor `warps · trips · lsu / lsu_ipc` under a
    /// wave's FFMA time ([`PlanShape::lsu`]). 0 = not given: no LSU term.
    pub lsu_ipc_pm: u64,
    /// attention-chains: **16-byte warp shared loads whose row groups
    /// broadcast one SM issues a clock**, per mille — the streamed
    /// attention's window loads (`g` lanes on distinct rows, every row group
    /// the same: `calib.cu lds bcast`, `col` at `g` = 8): 246 on the RTX
    /// 5080, 241 on the A100 SXM4. 0 = not given: four 128-byte wavefronts a
    /// warp load, one a clock ([`Self::lds_bcast`]).
    pub lds_bcast_pm: u64,
    /// kernel-merging `zero-fold`: a zero pass's fill rate beyond `zero_ns`
    /// (`cuMemsetD8Async` device time), bytes per ns. 0 = not given.
    pub fill_bytes_per_ns: u64,
    /// kernel-merging `zero-fold`: the rate a kernel's zero tail stores at
    /// (the time it adds to the kernel), bytes per ns. 0 = not given: the
    /// tail is unpriced and no zero pass folds.
    pub tail_bytes_per_ns: u64,
    /// kernel-merging `calls`: one grid barrier of a co-resident grid of at
    /// most `sms` blocks (block barrier, one release arrival a block on one
    /// counter, acquire spin), in ns — `benches/gpu/calib.cu gsync`. 0 = not
    /// given: no chain of launches is priced, none runs as one kernel.
    pub gsync_ns: u64,
    /// kernel-merging `calls`: what each block beyond `sms` adds to one grid
    /// barrier (the arrivals serialize on the counter), in ps.
    pub gsync_block_ps: u64,
    /// kernel-merging `calls`: the device time one dependent launch costs
    /// beyond its work — start to start of empty kernels of at most `sms`
    /// blocks, back to back on one stream with the host issue hidden, in ns
    /// (`calib.cu gsync`). It holds `gap_ns` and the launch's own ramp.
    pub dep_launch_ns: u64,
    /// kernel-merging `calls`: what each block beyond `sms` adds to one
    /// dependent launch (block dispatch), in ps.
    pub dep_launch_block_ps: u64,
    /// kernel-merging `calls`: the same dependent launch as a programmatic
    /// dependent launch (`griddepcontrol`: it starts while the one before
    /// drains and waits at its entry — how the runtime launches every kernel
    /// where the part has it), start to start, ≤ `sms` blocks, in ns. 0 = not
    /// measured: priced as [`Self::dep_launch_ns`].
    pub pdl_launch_ns: u64,
    /// kernel-merging `calls`: the same dependent launch as a cooperative
    /// launch (`cuLaunchCooperativeKernel`: every block co-resident — how a
    /// persistent kernel launches), start to start, in ns. 0 = not measured:
    /// priced as [`Self::dep_launch_ns`].
    pub coop_launch_ns: u64,
    /// hardware-properties `pinned`: the host's pageable host→device copy
    /// rate as a fraction of its page-locked rate, per mille, at the sizes a
    /// loop's arrays cross (≥ 4 MiB; `calib.cu copy`). A property of the host
    /// and its link, not the GPU: the rental boxes measured 350–1000. 0 = not
    /// measured: nothing is page-locked.
    pub pageable_pm: u64,
    /// long-rows `row-slices`: the device time one element a lane adds to a
    /// block walking a row (a flow unit's fold or stored cell, `256` lanes,
    /// latency-bound: the block's loads in flight over their latency), in
    /// ns — the slope of a one-block-a-row unit's kernel time over its row
    /// length (`long-rows.md`). 0 = not measured: no row is sliced.
    pub row_iter_ns: u64,
    /// long-rows `row-slices`: the host time a process's first cooperative
    /// launch takes beyond a plain launch's (the driver's one-time setup;
    /// later ones cost a plain launch), in ns. A cooperative launch the
    /// program issues once pays all of it. 0 = not measured: not priced.
    pub coop_first_ns: u64,
    /// cross-arch `row-split`: what one SM streams alone from an L2-resident
    /// operand, bytes per ns — the rate of a retained row's unsplit pass
    /// (one block walks the row; `calib.cu bw`, the best `sms=1` L2 point:
    /// H100 SXM 147, A100 81, RTX 4090 77, RTX 3090 62, RTX 5080 94). 0 = not
    /// measured: the split is not priced (the property rule alone).
    pub sm_bytes_per_ns: u64,
    /// kernel-merging `calls`: what a chain's member bodies cost inside one
    /// persistent kernel beyond their own launches, in SM clock cycles (at
    /// [`Gpu::clock_khz`]) — `benches/gpu/calib.cu body`: a synthetic chain
    /// run as separate launches and as one cooperative persistent kernel,
    /// less the boundaries the rule already prices (the barriers against the
    /// launches), its wide levels in the grid-stride form. 0 = not measured
    /// on this part: derived — [`COOP_BODY_CYCLES_RTX_5080`] at this part's
    /// clock ([`Gpu::coop_body_ns`]); never unpriced.
    pub coop_body_cycles: u64,
    /// stream-fly: the bytes a grid on every SM keeps in flight per SM when
    /// its DRAM read first reaches 90 % of the part's measured peak —
    /// `calib.cu bw`'s `fly` line, the smallest `sms = all` point at or above
    /// 0.9 × the best DRAM rate (RTX 4090 4 KiB, RTX 3090 / 5080 / 5090
    /// 8 KiB, A100 16 KiB, H100 SXM 32 KiB). Little's law at the loaded
    /// latency (`lat_ns`) is its lower bound: a stream with fewer bytes in
    /// flight waits on the latency, not the DRAM. 0 = not measured: derived
    /// ([`Gpu::stream_fly_per_sm`]).
    pub fly_bytes_per_sm: u64,
    /// hopper-wgmma-tma: **the bulk-copy engine's time per copy**, ns — one
    /// SM's elected thread issuing back-to-back copies, its engine's ns per
    /// copy at the panel sizes a stage moves (`calib.cu tma`). What separates
    /// a stage of row copies (96–144 a stage: the 5080's measured 4.3–14.5×
    /// loss) from one of whole-tile boxes (2–5). 0 = not measured: derived
    /// ([`Gpu::tma_copy_ns`]).
    pub tma_copy_ns: u64,
    /// integration (tune-context): **the largest cyclic working set the L2
    /// keeps**, bytes — re-read back to back at no less than the midpoint of
    /// the sweep's plateau (the stream's own re-read rate of a set the L2
    /// holds) and the DRAM rate ([`l2_keep_of_sweep`],
    /// `benches/local/l2keep.cu`; the RTX 5080's 64 MiB
    /// L2 keeps a 64.6 MiB stream at 4.1 TB/s, 1.025× its size at 2.5, 1.05×
    /// at 1.4: its DRAM rate by 1.1×). What separates an operand a program
    /// re-reads from L2 across calls from one it streams from DRAM: a trial
    /// evicts the L2 only past it ([`Gpu::l2_keep_bytes`]). 0 = not measured:
    /// the nominal L2.
    pub l2_keep_bytes: u64,
    /// flash-attention `stream` (P1): one block barrier of 256 threads
    /// with a dependent shared-memory round trip before it — what each window
    /// of a streamed unit pays per barrier on its block's critical path
    /// (`calib.cu bar`), in ns. 0 = not measured: derived
    /// ([`Gpu::bar_cycles`] at this part's clock, [`Gpu::bar_ns`]).
    pub bar_ns: u64,
    /// gaps-disasm C5 `split-price`: what one more `k` partition of a
    /// DRAM-streaming atomic split costs per output cell — its partial's
    /// `red` and its blocks' prologue and epilogue as the split kernel issues
    /// them beside the stream — in ps (a split sweep at one site: the slope
    /// of time against `S · cells` between two splits of equal grid spread).
    /// 0 = not measured: the fill rule's count, unpriced.
    pub split_cell_ps: u64,
}

/// gaps-disasm C5: the rtx-5080's [`DeviceCost::split_cell_ps`] — the
/// 16 × 2048 × 7168 cell's one-warp FMA split (16 tiles) at S = 14 and 28
/// (224 / 448 blocks, the same spread over 84 SMs, `grid_spread_pm` 1125):
/// 79.9 against 86.0 µs device time for 14 · 32 768 more partial cells
/// (`tile_bench --splitk`, CUDA events, median of 30;
/// `local-rtx-5080/gapfix-launch-combine.md`) — 13.3 ps a cell. The same
/// sweep's S = 7 / 8 / 56 (100.4 / 88.1 / 88.1 µs) are what the price
/// ([`crate::profile::TargetProfile::split_priced`]) predicts within 5 %.
pub const SPLIT_CELL_PS_RTX_5080: u64 = 13;

/// kernel-merging `calls`: the measured member-body change of a persistent
/// chain on the rtx-5080 — `calib.cu body`'s 4-round chain (its wide levels
/// in the grid-stride form), the median of six runs' span change, 1792 ns
/// [1083–2596], 2026-11-22 (`review-fixes.md` §5) — in cycles of the part's
/// 2.45 GHz profile clock, so the part prices what it measured. The source
/// every part without its own measurement derives its price from
/// ([`Gpu::coop_body_ns`]).
pub const COOP_BODY_CYCLES_RTX_5080: u64 = 4390;

/// open-rules `calls`: the rtx-5080's measured chain boundaries in cycles of
/// its 2.45 GHz profile clock — grid barrier 553 ns, dependent launch 650,
/// programmatic launch 486, cooperative launch 2044 (`calib.cu gsync`,
/// `calib/rtx-5080.calib`) — the source a part without its own measurement
/// derives its boundaries from ([`Gpu::chain_cost`]), as the bodies' price
/// is derived ([`COOP_BODY_CYCLES_RTX_5080`]).
pub const CHAIN_CYCLES_RTX_5080: [u64; 4] = [1355, 1593, 1191, 5008];

/// open-rules `row-slices`: the rtx-5080's measured walk and latency in
/// cycles of its 2.45 GHz profile clock — `row_iter_ns` 87 (a lane-iteration
/// of a one-block-a-row walk, CUPTI, `long-rows.md`), `lat_ns` 573 (a panel
/// load under load) — and its process's first cooperative launch,
/// `coop_first_ns` 38 000 ns (host time, not scaled): what a part without
/// its own measurement prices a row's slices on ([`Gpu::slice_cost`]).
pub const SLICE_RTX_5080: [u64; 3] = [213, 573, 38_000];

/// bugs-properties: [`Gpu::bar_cycles`] by compute capability — the SM
/// designs' measured barrier (`calib.cu bar`, 256 threads, cross-arch
/// 5fcb401b): `sm_80` 54 (A100), `sm_86` / `sm_89` 57 (RTX 3090 / 4090),
/// `sm_90` 42 (H100), consumer Blackwell 45 (RTX 5080, the `sm_100` family
/// too until measured); the native profile's fallback when the calibration
/// file gives no `bar_ns`, as [`mma_rates_of_cc`] for the matrix rates.
pub const fn bar_cycles_of_cc(cc: u64) -> u64 {
    match cc {
        80 => 54,
        90 => 42,
        100.. => 45,
        _ => 57,
    }
}

/// A [`DeviceCost`] without the plan-cost rates: the parts that keep the
/// rules' plans (D8). `..NO_PLAN` fills them.
pub const NO_PLAN: DeviceCost = DeviceCost {
    flops_per_ns: 0,
    bytes_per_ns: 0,
    launch_ns: 0,
    issue_ns: 0,
    gap_ns: 0,
    zero_ns: 0,
    atomic_ops_per_ns: 0,
    l2_bytes_per_ns: 0,
    dram_bytes_per_ns: 0,
    eff_max_pm: 0,
    eff_half: 0,
    lat_ns: 0,
    lsu_ipc_pm: 0,
    lds_bcast_pm: 0,
    fill_bytes_per_ns: 0,
    tail_bytes_per_ns: 0,
    gsync_ns: 0,
    gsync_block_ps: 0,
    dep_launch_ns: 0,
    dep_launch_block_ps: 0,
    pdl_launch_ns: 0,
    coop_launch_ns: 0,
    pageable_pm: 0,
    row_iter_ns: 0,
    coop_first_ns: 0,
    sm_bytes_per_ns: 0,
    coop_body_cycles: 0,
    fly_bytes_per_sm: 0,
    tma_copy_ns: 0,
    l2_keep_bytes: 0,
    bar_ns: 0,
    split_cell_ps: 0,
};

impl DeviceCost {
    /// attention-chains: [`Self::lds_bcast_pm`] in warp loads a clock —
    /// measured, else what the shared memory's banks deliver: a warp's 16-byte
    /// loads `warp · 16` bytes, one 128-byte wavefront a clock (32 banks ×
    /// 4 B: every part's, [`crate::func::gpu::GPU_WAVEFRONT_BYTES`]).
    pub fn lds_bcast(&self, warp: u64) -> f64 {
        match self.lds_bcast_pm {
            0 => crate::func::gpu::GPU_WAVEFRONT_BYTES as f64 / (warp.max(1) * 16) as f64,
            pm => pm as f64 / 1000.0,
        }
    }

    /// `--cost=` names, in [`CostOver`] order: the [`DeviceCost`] fields,
    /// then the [`Gpu`] properties a measurement or a cross-compilation
    /// overrides by name ([`Gpu::fp32_lanes`], [`Gpu::clock_khz`], and the
    /// hardware-properties the rung rules read: SMs, L2, shared memory,
    /// threads, registers and blocks per SM, the DRAM interface).
    pub const NAMES: [&'static str; 51] = [
        "flops_per_ns",
        "bytes_per_ns",
        "launch_ns",
        "issue_ns",
        "gap_ns",
        "zero_ns",
        "atomic_ops_per_ns",
        "l2_bytes_per_ns",
        "dram_bytes_per_ns",
        "eff_max_pm",
        "eff_half",
        "lat_ns",
        "lsu_ipc_pm",
        "lds_bcast_pm",
        "fill_bytes_per_ns",
        "tail_bytes_per_ns",
        "gsync_ns",
        "gsync_block_ps",
        "dep_launch_ns",
        "dep_launch_block_ps",
        "pdl_launch_ns",
        "coop_launch_ns",
        "pageable_pm",
        "row_iter_ns",
        "coop_first_ns",
        "sm_bytes_per_ns",
        "coop_body_cycles",
        "fly_bytes_per_sm",
        "bar_ns",
        "tma_copy_ns",
        "l2_keep_bytes",
        "split_cell_ps",
        "fp32_lanes",
        "clock_khz",
        "sms",
        "l2_bytes",
        "smem_per_sm",
        "max_threads_per_sm",
        "regs_per_sm",
        "max_blocks_per_sm",
        "mem_clock_khz",
        "mem_bus_bits",
        "mma_tf32_per_clk",
        "mma_half_per_clk",
        "wgmma_tf32_per_clk",
        "wgmma_half_per_clk",
        // ADR-0046 D8: the SM's approximate math units (`calib.cu math`),
        // errors × 10¹² ([`MathUnits`]); `math_specials=1`: they return the
        // contract's special values.
        "sqrt_approx_rel_e12",
        "sin_approx_rel_e12",
        "sin_approx_abs_e12",
        "cos_approx_rel_e12",
        "math_specials",
    ];

    /// kernel-merging `calls`: one grid barrier of `blocks` co-resident
    /// blocks, in ns ([`Self::gsync_ns`] up to `sms` blocks, then
    /// [`Self::gsync_block_ps`] a block). `None` when not measured.
    pub fn gsync_at(&self, blocks: u64, sms: u64) -> Option<f64> {
        (self.gsync_ns > 0).then(|| {
            self.gsync_ns as f64
                + blocks.saturating_sub(sms) as f64 * self.gsync_block_ps as f64 / 1e3
        })
    }

    /// kernel-merging `calls`: the device time one dependent launch of
    /// `blocks` blocks adds beyond its work, in ns. `None` when not measured.
    pub fn dep_launch_at(&self, blocks: u64, sms: u64) -> Option<f64> {
        (self.dep_launch_ns > 0).then(|| {
            self.dep_launch_ns as f64
                + blocks.saturating_sub(sms) as f64 * self.dep_launch_block_ps as f64 / 1e3
        })
    }

    /// A cooperative dependent launch of `blocks` blocks (a persistent
    /// kernel's), in ns: [`Self::coop_launch_ns`] where measured.
    pub fn coop_launch_at(&self, blocks: u64, sms: u64) -> Option<f64> {
        let base = self.dep_launch_at(blocks, sms)?;
        Some(base + self.coop_launch_ns.saturating_sub(self.dep_launch_ns) as f64)
    }

    /// [`Self::dep_launch_at`] as the runtime issues it: a programmatic
    /// dependent launch (`pdl`) where measured.
    pub fn launch_at(&self, blocks: u64, sms: u64, pdl: bool) -> Option<f64> {
        let base = self.dep_launch_at(blocks, sms)?;
        Some(match (pdl, self.pdl_launch_ns) {
            (true, n) if n > 0 => base - (self.dep_launch_ns - n.min(self.dep_launch_ns)) as f64,
            _ => base,
        })
    }
}

/// A `--cost=` override per rate, in [`DeviceCost::NAMES`] order (`None` =
/// the profile's value) — the `--geom=` precedent for the fit harness
/// (`benches/gpu/plancost.py`).
pub type CostOver = [Option<u64>; DeviceCost::NAMES.len()];

impl Gpu {
    /// [`DeviceCost::NAMES`] that name a [`DeviceCost`] field (those before
    /// `fp32_lanes`). gapfix: counted, not written — at 30 it had fallen one
    /// behind the list, so `--cost=l2_keep_bytes=` alone was dropped.
    const NAMES_COST: usize = {
        let mut i = 0;
        while !matches!(DeviceCost::NAMES[i].as_bytes(), b"fp32_lanes") {
            i += 1;
        }
        i
    };

    /// This part with the overrides in `over` applied (a part without a
    /// [`DeviceCost`] takes one from [`NO_PLAN`]).
    pub fn with_cost(mut self, over: &CostOver) -> Self {
        let mut math = MathUnits::default();
        let mut d = self.cost.unwrap_or(NO_PLAN);
        for (name, v) in DeviceCost::NAMES.iter().zip(over) {
            let Some(v) = *v else { continue };
            *match *name {
                "flops_per_ns" => &mut d.flops_per_ns,
                "bytes_per_ns" => &mut d.bytes_per_ns,
                "launch_ns" => &mut d.launch_ns,
                "issue_ns" => &mut d.issue_ns,
                "gap_ns" => &mut d.gap_ns,
                "zero_ns" => &mut d.zero_ns,
                "atomic_ops_per_ns" => &mut d.atomic_ops_per_ns,
                "l2_bytes_per_ns" => &mut d.l2_bytes_per_ns,
                "dram_bytes_per_ns" => &mut d.dram_bytes_per_ns,
                "eff_max_pm" => &mut d.eff_max_pm,
                "eff_half" => &mut d.eff_half,
                "lat_ns" => &mut d.lat_ns,
                "lsu_ipc_pm" => &mut d.lsu_ipc_pm,
                "lds_bcast_pm" => &mut d.lds_bcast_pm,
                "fill_bytes_per_ns" => &mut d.fill_bytes_per_ns,
                "tail_bytes_per_ns" => &mut d.tail_bytes_per_ns,
                "gsync_ns" => &mut d.gsync_ns,
                "gsync_block_ps" => &mut d.gsync_block_ps,
                "dep_launch_ns" => &mut d.dep_launch_ns,
                "dep_launch_block_ps" => &mut d.dep_launch_block_ps,
                "pdl_launch_ns" => &mut d.pdl_launch_ns,
                "coop_launch_ns" => &mut d.coop_launch_ns,
                "pageable_pm" => &mut d.pageable_pm,
                "row_iter_ns" => &mut d.row_iter_ns,
                "coop_first_ns" => &mut d.coop_first_ns,
                "sm_bytes_per_ns" => &mut d.sm_bytes_per_ns,
                "coop_body_cycles" => &mut d.coop_body_cycles,
                "fly_bytes_per_sm" => &mut d.fly_bytes_per_sm,
                "bar_ns" => &mut d.bar_ns,
                "tma_copy_ns" => &mut d.tma_copy_ns,
                "l2_keep_bytes" => &mut d.l2_keep_bytes,
                "split_cell_ps" => &mut d.split_cell_ps,
                "fp32_lanes" => &mut self.fp32_lanes,
                "clock_khz" => &mut self.clock_khz,
                "sms" => &mut self.sms,
                "l2_bytes" => &mut self.l2_bytes,
                "smem_per_sm" => &mut self.smem_per_sm,
                "max_threads_per_sm" => &mut self.max_threads_per_sm,
                "regs_per_sm" => &mut self.regs_per_sm,
                "max_blocks_per_sm" => &mut self.max_blocks_per_sm,
                "mem_clock_khz" => &mut self.mem_clock_khz,
                "mma_tf32_per_clk" => &mut self.mma_tf32_per_clk,
                "mma_half_per_clk" => &mut self.mma_half_per_clk,
                "wgmma_tf32_per_clk" => &mut self.wgmma_tf32_per_clk,
                "wgmma_half_per_clk" => &mut self.wgmma_half_per_clk,
                "sqrt_approx_rel_e12" => &mut math.sqrt_rel_e12,
                "sin_approx_rel_e12" => &mut math.sin_rel_e12,
                "sin_approx_abs_e12" => &mut math.sin_abs_e12,
                "cos_approx_rel_e12" => &mut math.cos_rel_e12,
                "math_specials" => &mut math.specials,
                _ => &mut self.mem_bus_bits,
            } = v;
        }
        // ADR-0046 D8: the measured approximate units join the SM's.
        if math != MathUnits::default() {
            self.caps.math = math.target();
        }
        // The `DeviceCost` fields: every name but the last two (`Gpu`'s).
        if over.iter().take(Self::NAMES_COST).any(Option::is_some) {
            self.cost = Some(d);
        }
        self
    }

    /// The rate `--cost=<name>` sets, as this part has it (0 = not given).
    pub fn rate(&self, name: &str) -> u64 {
        let d = self.cost.unwrap_or(NO_PLAN);
        match name {
            "flops_per_ns" => d.flops_per_ns,
            "bytes_per_ns" => d.bytes_per_ns,
            "launch_ns" => d.launch_ns,
            "issue_ns" => d.issue_ns,
            "gap_ns" => d.gap_ns,
            "zero_ns" => d.zero_ns,
            "atomic_ops_per_ns" => d.atomic_ops_per_ns,
            "l2_bytes_per_ns" => d.l2_bytes_per_ns,
            "dram_bytes_per_ns" => d.dram_bytes_per_ns,
            "eff_max_pm" => d.eff_max_pm,
            "eff_half" => d.eff_half,
            "lat_ns" => d.lat_ns,
            "lsu_ipc_pm" => d.lsu_ipc_pm,
            "lds_bcast_pm" => d.lds_bcast_pm,
            "fill_bytes_per_ns" => d.fill_bytes_per_ns,
            "tail_bytes_per_ns" => d.tail_bytes_per_ns,
            "gsync_ns" => d.gsync_ns,
            "gsync_block_ps" => d.gsync_block_ps,
            "dep_launch_ns" => d.dep_launch_ns,
            "dep_launch_block_ps" => d.dep_launch_block_ps,
            "pdl_launch_ns" => d.pdl_launch_ns,
            "coop_launch_ns" => d.coop_launch_ns,
            "pageable_pm" => d.pageable_pm,
            "row_iter_ns" => d.row_iter_ns,
            "coop_first_ns" => d.coop_first_ns,
            "sm_bytes_per_ns" => d.sm_bytes_per_ns,
            "coop_body_cycles" => d.coop_body_cycles,
            "fly_bytes_per_sm" => d.fly_bytes_per_sm,
            "bar_ns" => d.bar_ns,
            "tma_copy_ns" => d.tma_copy_ns,
            "l2_keep_bytes" => d.l2_keep_bytes,
            "split_cell_ps" => d.split_cell_ps,
            "fp32_lanes" => self.fp32_lanes,
            "clock_khz" => self.clock_khz,
            "sms" => self.sms,
            "l2_bytes" => self.l2_bytes,
            "smem_per_sm" => self.smem_per_sm,
            "max_threads_per_sm" => self.max_threads_per_sm,
            "regs_per_sm" => self.regs_per_sm,
            "max_blocks_per_sm" => self.max_blocks_per_sm,
            "mem_clock_khz" => self.mem_clock_khz,
            "mem_bus_bits" => self.mem_bus_bits,
            "mma_tf32_per_clk" => self.mma_tf32_per_clk,
            "mma_half_per_clk" => self.mma_half_per_clk,
            "wgmma_tf32_per_clk" => self.wgmma_tf32_per_clk,
            "wgmma_half_per_clk" => self.wgmma_half_per_clk,
            n if n.ends_with("_e12") || n == "math_specials" => {
                MathUnits::of(&self.caps.math).get(n)
            }
            _ => 0,
        }
    }
}
