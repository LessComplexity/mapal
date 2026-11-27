//! The GPU target profiles (RTX 4090, H100, RTX 5090, RTX 5080, RTX 3090, A100) and their shared SM base.

use super::*;

/// The `sm_89` per-SM facts (CUDA C Programming Guide, compute capability 8.9),
/// with the RTX 4070 Ti's 60 SMs and its 48 MB L2 (AD104, Ada whitepaper).
pub(super) const ADA_SM: Gpu = Gpu {
    warp: 32,
    max_threads_per_block: 1024,
    arch: "sm_89",
    sms: 60,
    regs_per_thread: 255,
    regs_per_sm: 65536,
    smem_per_sm: 100 * 1024,
    smem_per_block: 99 * 1024,
    smem_reserved_per_block: 1024,
    max_threads_per_sm: 1536,
    max_blocks_per_sm: 24,
    l2_bytes: 48 * 1024 * 1024,
    cost: None,
    geom: GEOM,
    // The 4070 Ti's datasheet: 128 FP32 lanes per SM, 2.61 GHz boost, GDDR6X
    // at 21 Gb/s on 192 bits (504 GB/s).
    fp32_lanes: 128,
    clock_khz: 2_610_000,
    mem_clock_khz: 10_501_000,
    mem_bus_bits: 192,
    schedulers: 4,
    // bugs-properties: the 4090's (the sm_89 SM; `calib.cu bar`, cross-arch
    // 5fcb401b: 57.2 cycles at 256 threads).
    bar_cycles: 57,
    // f32-tensor-cores P1: the warp products' flops an SM a clock. Every
    // GeForce SM since `sm_86` issues TF32 at its FFMA rate and the 16-bit
    // products with f32 accumulators at twice it: the RTX 5080 measured
    // 256 / 512 (`calib.cu matrix`, f32-tensor-cores.md); the 3090 / 4090 /
    // 5090 datasheets' dense TF32 / FP16-with-FP32-accumulate rates (35.6 /
    // 71, 82.6 / 165.2, 104.8 / 209.5 Tflop/s) over `sms · boost` give the
    // same 256 / 512.
    mma_tf32_per_clk: 256,
    mma_half_per_clk: 512,
    wgmma_tf32_per_clk: 0,
    wgmma_half_per_clk: 0,
    caps: CAPS_WARP_MMA_F8,
    place: &PLACE_CUDA_ADA,
};

/// RTX 4090 — Ada (`sm_89`), 128 SMs. A rented part (S52): the profile is the
/// cross-compilation spelling, `benches/gpu/devquery.cu` on the box is its check.
pub const RTX_4090: TargetProfile = TargetProfile {
    name: "rtx-4090",
    // 72 MB L2: `facts l2_bytes=75497472` (base-rtx-4090.txt).
    machine: Machine::Gpu(Gpu {
        sms: 128,
        l2_bytes: 72 * 1024 * 1024,
        // S62 `place`, measured (s62/cpugpu/run1/results-rtx-4090.txt: mm_4096 contract Mapal 41.69 Tflop/s; pt_pj_m1_n2048_k7168 site kernel 58.76 MB at 490 B/ns; pt_pc_64 all-device iter ms / 3 sites).
        cost: Some(DeviceCost {
            flops_per_ns: 41690,
            bytes_per_ns: 490,
            launch_ns: 13723,
            // plan-cost, measured (E1 = `.claude/sprints/plan-cost/evidence/E1-small-chains.md`, rtx-4090):
            // issue: (c) empty-kernel launch back to back, 2.85 µs [2.77–2.90];
            // gap: (a) device gap between dependent kernels at 1024/2048, 0.6–0.7 µs;
            // zero: (c) memset 1 MB device time 1.25 µs less 1 MB at the L2 rate;
            // atomic: (c) f32 RED, M = 262144, S = 2…64, ~2.5 ps an add;
            // l2: (c) L2-resident read, 16–128 MB plateau 4.5–4.6 TB/s;
            // lat: (c) `benches/gpu/calib.cu`, the model's own quantity — bytes in
            // flight over the achieved DRAM rate below 90 % of peak, median 380 ns
            // [IQR 369–407] over 87 grid points (`B-bw-rtx-4090.tsv`, plan-cost/v2);
            // the L2 one is 240 ns [233–281] (open: the L2-residency fact picks it);
            // the idle chases read 146 (DRAM) / 106 (L2) ns. B agrees with dram
            // (958 GB/s) and l2 (4895 GB/s peak) within 2.3 % / 8.8 %.
            // eff: fitted by `benches/gpu/plancost.py --lat 380` (L1 in log time)
            // over 6 program folds: 400..600, eff_half 0..1, all-decision (525, 0)
            // (`benches/results/evidence/plan-cost/holdout-rtx-4090.tsv`).
            issue_ns: 2850,
            gap_ns: 600,
            zero_ns: 1017,
            atomic_ops_per_ns: 400,
            l2_bytes_per_ns: 4500,
            // dram: (c) DRAM read, 2 GB `__ldcg`, 981 GB/s.
            dram_bytes_per_ns: 981,
            eff_max_pm: 525,
            eff_half: 0,
            lat_ns: 380,
            // lsu (tile-efficiency T4): calib B `lds`, LDS.32 at 256 and 1024
            // threads, 64.0 B/clk/SM = 0.5 warp-instructions a clock
            // (`benches/results/evidence/plan-cost/v2/B-rates.tsv`,
            // `B-lds-rtx-4090.tsv`); the LDS.128 broadcast rate is open (no ncu).
            lsu_ipc_pm: 500,
            lds_bcast_pm: 0,
            // zero-fold: not measured on this part (no fold: unpriced).
            fill_bytes_per_ns: 0,
            tail_bytes_per_ns: 0,
            // cross-arch (`benches/results/evidence/cross-arch/README.md`, `calib.cu gsync` / `bw` on the
            // vast box, 2026-11-21): grid barrier 502 ns, dependent launch
            // 1440 ns, cooperative 2660 ns (block 256, grid <= sms); one SM
            // alone streams 77 B/ns from L2.
            gsync_ns: 502,
            gsync_block_ps: 0,
            dep_launch_ns: 1440,
            dep_launch_block_ps: 0,
            pdl_launch_ns: 0,
            coop_launch_ns: 2660,
            pageable_pm: 0,
            row_iter_ns: 0,
            coop_first_ns: 0,
            sm_bytes_per_ns: 77,
            // kernel-merging `calls`: `calib.cu body` on the cross-arch 5fcb401b
            // box, 7292 cycles [7254–7293] (bugs-properties; the 5080's 4390
            // derived before).
            coop_body_cycles: 7292,
            // stream-fly: `calib.cu bw` on the cross-arch box (60f0f8ce): 90 % of
            // its 958 GB/s DRAM peak first at 4 KiB in flight an SM.
            fly_bytes_per_sm: 4096,
            tma_copy_ns: 0,
            l2_keep_bytes: 0,
            bar_ns: 0,
            split_cell_ps: 0,
        }),
        // E1 header: 128 FP32 lanes per SM (sm_89), 2.52 GHz under load. The
        // cross-arch box (power limit 250 W of the part's 450) sustained
        // 1.955 GHz (`calib.cu ffma`): a box's limit, carried by its
        // calibration line (`native-gpu`), not the part's rating; no rung rule
        // reads the clock alone (`rules.rs` `DRAM_PER_SM_LOW`).
        fp32_lanes: 128,
        clock_khz: 2_520_000,
        // GDDR6X at 21 Gb/s on 384 bits: 1008 GB/s (device query
        // `mem_clock_khz=10501000 bus_bits=384`).
        mem_clock_khz: 10_501_000,
        mem_bus_bits: 384,
        place: &PLACE_RTX_4090,
        ..ADA_SM
    }),
    ..GENERIC
};

/// H100 SXM5 — Hopper, 132 SMs, 228 KB shared memory per SM. The target is
/// the arch-specific `sm_90a` (S53): it is what admits `wgmma`, and every
/// sm_90 instruction is in it.
pub const H100: TargetProfile = TargetProfile {
    name: "h100",
    machine: Machine::Gpu(Gpu {
        arch: "sm_90a",
        sms: 132,
        smem_per_sm: 228 * 1024,
        smem_per_block: 227 * 1024,
        max_threads_per_sm: 2048,
        max_blocks_per_sm: 32,
        // 50 MB (H100 whitepaper; unmeasured on the SXM part).
        l2_bytes: 50 * 1024 * 1024,
        // cross-arch (`benches/results/evidence/cross-arch/README.md`, `calib.cu` on the vast SXM box at
        // its 700 W limit, 2026-11-21): DRAM 3186 / L2 3990 GB/s read, one SM
        // alone 147 B/ns (L2); grid barrier 910 ns, dependent launch 1722 ns,
        // programmatic dependent launch 734 ns, cooperative 1839 ns (block
        // 256, grid <= sms). Every device-query fact equals this spelling.
        cost: Some(DeviceCost {
            l2_bytes_per_ns: 3990,
            dram_bytes_per_ns: 3186,
            gsync_ns: 910,
            dep_launch_ns: 1722,
            pdl_launch_ns: 734,
            coop_launch_ns: 1839,
            sm_bytes_per_ns: 147,
            // stream-fly: `calib.cu bw` (cross-arch 60f0f8ce): 90 % of the
            // 3177 GB/s DRAM peak first at 32 KiB in flight an SM (16 KiB:
            // 80 %); the loaded latency there 632 ns.
            fly_bytes_per_sm: 32768,
            // hopper-wgmma-tma: `tma_calib.cu` on a rented H100 SXM (h100-s1): 159 ns a
            // copy at 128 B - 16 KiB for any ring depth >= 4 (186-302 ns alone); 32 KiB
            // boxes 352-475 ns (13.5 TB/s over the part at 16 KiB: 102 B/ns an SM).
            tma_copy_ns: 159,
            bar_ns: 0,
            split_cell_ps: 0,
            // bugs-properties, measured on the cross-arch 5fcb401b box
            // (`calib-h100.txt`): the loaded latency (`calib.cu bw`, plan-cost's
            // quantity): median 629 ns [IQR 605–660] over 116 points; `calib.cu
            // body`: 348 cycles [325–356] (the 5080's 4390 derived before);
            // `l2keep.cu`: 1.0 × the 50 MiB L2 re-reads at 2579 GB/s, below the
            // DRAM rate (3180; L2 4018), flat to 1.05 × — the keep lies below
            // the sweep's first point (two partitions). Not measured below 1.0
            // ×: half the L2 is a placeholder (open) until the sweep from 0.25
            // × L2 is read by [`l2_keep_of_sweep`].
            lat_ns: 629,
            coop_body_cycles: 348,
            l2_keep_bytes: 25 * 1024 * 1024,
            ..NO_PLAN
        }),
        // HBM3 3.35 TB/s on 5120 bits (datasheet; query `mem=2619000x5120`);
        // the FFMA peak under load 61.07 Tflop/s = 1.807 GHz at 128 lanes
        // (`calib.cu ffma`; boost 1.98 GHz).
        clock_khz: 1_807_000,
        // bugs-properties: `calib.cu bar` 41.8 cycles (5fcb401b).
        bar_cycles: 42,
        mem_clock_khz: 2_619_000,
        mem_bus_bits: 5120,
        // f32-tensor-cores P1: `wgmma` TF32 — the datasheet's dense 494.7
        // Tflop/s over 132 SMs at 1.83 GHz (2048). bugs-properties: `mma.sync`
        // measured (`calib.cu matrix`, cross-arch 5fcb401b, `calib-h100.txt`,
        // 32 warps an SM of independent register chains, clock64 spans):
        // m16n8k8 TF32 1351, m16n8k16 bf16 / f16 2702 flops an SM a clock (the
        // 532 / 1064 before were Mapal's HMMA tile realized, mlp_2048_tf32 — a
        // lower bound). Read with the split tile's issue priced
        // (`rules::split_issue_pm`, tensor-cores-2), the rate moves no f32 GEMM
        // onto tf32x3: the rate rule alone flips.
        mma_tf32_per_clk: 1351,
        mma_half_per_clk: 2702,
        wgmma_tf32_per_clk: 2048,
        // wgmma-16bit: `wgmma` bf16 / f16 with f32 accumulators — measured
        // (`calib.cu matrix`, wgmma-16bit's H100 SXM box, two warpgroups an
        // SM, clock64 spans): `wgmma.m64n256k16.bf16` 4096 flops an SM a
        // clock with `a` from shared memory or registers (998 / 997 Tflop/s
        // at 1.85 GHz), the datasheet's dense 989.4 over 132 SMs at 1.83;
        // the TF32 `m64n256k8` there 2048 (the value above, now measured).
        wgmma_half_per_clk: 4096,
        caps: CAPS_WARPGROUP_MMA,
        place: &PLACE_H100,
        ..ADA_SM
    }),
    ..GENERIC
};

/// H100 PCIe — the same Hopper SM as [`H100`], 114 of them.
pub const H100_PCIE: TargetProfile = TargetProfile {
    name: "h100-pcie",
    machine: Machine::Gpu(Gpu {
        arch: "sm_90a",
        sms: 114,
        smem_per_sm: 228 * 1024,
        smem_per_block: 227 * 1024,
        max_threads_per_sm: 2048,
        max_blocks_per_sm: 32,
        // `facts l2_bytes=52428800` (base-h100-pcie.txt).
        l2_bytes: 50 * 1024 * 1024,
        // S62 `place`, measured. `launch_ns` on the H100 PCIe
        // (`s62/cpugpu/run2/results-h100-pcie.txt`: pt_pc_64's all-device
        // build, `off-place`, iter ms / 3 sites). The rates on an H100 80GB HBM3
        // (SXM) box running this build, the only H100 rentable for run 1
        // (`run1/results-h100-pcie.txt`: mm_4096 contract Mapal 49.07 Tflop/s;
        // pt_pj_m1_n2048_k7168's site kernel 135 B/ns) — no site of the S62 set
        // is placed by them (`placediff.sh`: only `launch_ns` moves a site).
        // The SXM box's `launch_ns`, 28 799, put the 64³ chains on the host,
        // which the PCIe box measured 1.6–2.2% slower (run2 `pt_pm_2048_64`).
        cost: Some(DeviceCost {
            flops_per_ns: 49070,
            bytes_per_ns: 135,
            launch_ns: 11315,
            // `coop_body_cycles` derived (the rtx-5080's 4390 at 1.755 GHz =
            // 2501 ns); no barrier or launch measured: no chain is priced.
            ..NO_PLAN
        }),
        // Datasheet (H100 PCIe 80 GB): HBM2e 2.04 TB/s on 5120 bits (device
        // query `mem_clock_khz=1593000`), 1.755 GHz boost.
        clock_khz: 1_755_000,
        // bugs-properties: the Hopper SM's barrier ([`H100`]'s 41.8 cycles).
        bar_cycles: 42,
        mem_clock_khz: 1_593_000,
        mem_bus_bits: 5120,
        // f32-tensor-cores P1: the Hopper SM's rates ([`H100`]; an SM a clock:
        // measured on the SXM part, the same SM).
        mma_tf32_per_clk: 1351,
        mma_half_per_clk: 2702,
        wgmma_tf32_per_clk: 2048,
        wgmma_half_per_clk: 4096,
        caps: CAPS_WARPGROUP_MMA,
        place: &PLACE_H100_PCIE,
        ..ADA_SM
    }),
    ..GENERIC
};

/// RTX 5090 — consumer Blackwell (`sm_120`, not the datacenter `sm_100`),
/// 170 SMs. Resident blocks per SM is 24, as on Ada: the box's device query
/// reports `max_blocks_per_sm=24` (`v6-rtx-5090.txt`, `facts` line).
pub const RTX_5090: TargetProfile = TargetProfile {
    name: "rtx-5090",
    machine: Machine::Gpu(Gpu {
        arch: "sm_120",
        sms: 170,
        // `facts l2_bytes=100663296` (base-rtx-5090.txt).
        l2_bytes: 96 * 1024 * 1024,
        // S62 `place`, measured (s62/cpugpu/run1/results-rtx-5090.txt: mm_4096 contract Mapal 53.41 Tflop/s; pt_pj_m1_n2048_k7168 site kernel 494 B/ns; pt_pc_64 all-device iter ms / 3 sites).
        cost: Some(DeviceCost {
            flops_per_ns: 53410,
            bytes_per_ns: 494,
            launch_ns: 17015,
            // plan-cost, measured (E1 = `.claude/sprints/plan-cost/evidence/E1-small-chains.md`, rtx-5090):
            // issue: (c) empty-kernel launch back to back, 2.37 µs [2.35–2.49];
            // gap: (a) device gap between dependent kernels, 0.3 µs;
            // zero: (c) memset 1 MB device time 0.9 µs less 1 MB at the L2 rate;
            // atomic: (c) f32 RED, M = 262144, S = 2…64, ~1.7 ps an add;
            // l2: (c) L2-resident read, 16–128 MB plateau 6.5–6.9 TB/s;
            // lat: (c) `benches/gpu/calib.cu`, the model's own quantity — bytes in
            // flight over the achieved DRAM rate below 90 % of peak, median 510 ns
            // [IQR 500–528] over 111 grid points (`B-bw-rtx-5090.tsv`, plan-cost/v2);
            // the L2 one is 290 ns [282–350] (open: the L2-residency fact picks it);
            // the idle chases read 195 (DRAM) / 141 (L2) ns. B agrees with dram
            // (1701 GB/s) and l2 (6683 GB/s peak) within 0.7 % / 2.8 %.
            // eff: fitted by `benches/gpu/plancost.py --lat 510` (L1 in log time)
            // over 6 program folds: 700..950, eff_half 6..8, all-decision (800, 6), off
            // the grid edge (`benches/results/evidence/plan-cost/holdout-rtx-5090.tsv`).
            issue_ns: 2370,
            gap_ns: 300,
            zero_ns: 739,
            atomic_ops_per_ns: 560,
            l2_bytes_per_ns: 6500,
            // dram: (c) DRAM read, 2 GB `__ldcg`, 1690 GB/s.
            dram_bytes_per_ns: 1690,
            eff_max_pm: 800,
            eff_half: 6,
            lat_ns: 510,
            // lsu (tile-efficiency T4): calib B `lds`, LDS.32 at 256 and 1024
            // threads, 64.0 B/clk/SM = 0.5 warp-instructions a clock
            // (`benches/results/evidence/plan-cost/v2/B-rates.tsv`,
            // `B-lds-rtx-5090.tsv`); the LDS.128 broadcast rate is open (no ncu).
            lsu_ipc_pm: 500,
            lds_bcast_pm: 0,
            // zero-fold: not measured on this part (no fold: unpriced).
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
            // kernel-merging `calls`: not measured here — derived, the
            // rtx-5080's 4390 cycles at 2.41 GHz = 1822 ns a chain (no
            // barrier or launch measured: no chain is priced).
            coop_body_cycles: 0,
            // stream-fly: `calib.cu bw` (plan-cost v2 `B-bw-rtx-5090.tsv`): 90 %
            // of its 1701 GB/s DRAM peak first at 8 KiB in flight an SM.
            fly_bytes_per_sm: 8192,
            tma_copy_ns: 0,
            l2_keep_bytes: 0,
            bar_ns: 0,
            split_cell_ps: 0,
        }),
        // E1 header: 128 FP32 lanes per SM (sm_120), 2.41 GHz under load.
        fp32_lanes: 128,
        clock_khz: 2_410_000,
        // GDDR7 at 28 Gb/s on 512 bits: 1792 GB/s.
        mem_clock_khz: 14_001_000,
        // bugs-properties: the consumer Blackwell SM's barrier ([`RTX_5080`]'s
        // 44.6 cycles).
        bar_cycles: 45,
        mem_bus_bits: 512,
        caps: CAPS_WARP_MMA_PDL,
        place: &PLACE_RTX_5090,
        ..ADA_SM
    }),
    ..GENERIC
};

/// RTX 5080 — consumer Blackwell GB203 (`sm_120`), 84 SMs: a local part (the
/// Threadripper 9970X workstation), not a rental. Device query
/// (`cudaGetDeviceProperties`, 2026-11-19, driver 620.70.06): `sms=84
/// l2_bytes=67108864 max_blocks_per_sm=24 smem_per_sm=102400
/// smem_per_block_optin=101376 regs_per_sm=65536 max_threads_per_sm=1536` —
/// Ada's per-SM facts ([`ADA_SM`]) with this part's SM count and L2.
/// Device rates calibrated 2026-11-19 (local-rtx-5080 gpu-analysis), same
/// methods as `rtx-5090`'s; tables in `benches/results/evidence/local-rtx-5080/calib/`.
/// Its rungs are the rules over these properties ([`crate::rules::rungs`]),
/// like every part's; `native-gpu` on this box queries the same facts
/// (`local-rtx-5080/hardware-properties.md`).
pub const RTX_5080: TargetProfile = TargetProfile {
    name: "rtx-5080",
    machine: Machine::Gpu(Gpu {
        arch: "sm_120",
        sms: 84,
        l2_bytes: 64 * 1024 * 1024,
        // `place`, measured: mm_4096 contract cell 4.269 ms = 32.2 Tflop/s
        // (`gpu/gap-run2.md` run, `mm_4096x4096x4096_c`); pt_pj_m1_n2048_k7168
        // site kernel 23 us under MAPAL_GPU_TRACE = 2553 B/ns (its 58.7 MB of
        // weights are L2-resident on 64 MB: an L2 rate, not DRAM); pt_pc_64
        // all-device iter 0.0334 ms (median of 21) / 3 sites.
        cost: Some(DeviceCost {
            flops_per_ns: 32200,
            bytes_per_ns: 2553,
            launch_ns: 11130,
            // plan-cost, measured (`calib/E1-rtx-5080.txt`, plan-cost E1's
            // calib.cu / calib2.cu, 3 runs): issue: empty-kernel launch back to
            // back, 1.69 us [1.68–1.75]; gap: CUPTI device gap between dependent
            // kernels of pt_loopmm_1024_8 / pt_rt_mlp2_1024, median 0.26 us;
            // zero: CUPTI memset 1 MB device time 1024 ns (`memset-device-ns-*.tsv`)
            // less 1 MB at the L2 rate (233 ns); atomic: f32 RED, M = 262144,
            // S = 2…64, 388–403 Gatom/s; l2: `__ldcg` 16–48 MB plateau
            // 4.45–4.62 TB/s; dram: `__ldcg` 536 MB 894 GB/s, calib B's DRAM
            // peak 899 GB/s (the 2 GB point reads 751: open); lat: calib B,
            // bytes in flight over the achieved DRAM rate below 90 % of peak,
            // median 573 ns [IQR 557–604] over 125 points (`B-bw-rtx-5080.tsv`;
            // the L2 one 328 ns). eff_max_pm / eff_half: `rtx-5090`'s fit (the
            // same sm_120 SM; a per-SM fraction of FFMA peak) — not refit here.
            issue_ns: 1690,
            gap_ns: 260,
            zero_ns: 790,
            atomic_ops_per_ns: 400,
            l2_bytes_per_ns: 4500,
            dram_bytes_per_ns: 890,
            eff_max_pm: 800,
            eff_half: 6,
            lat_ns: 573,
            // lsu: calib B `lds`, LDS.32 at 256 threads 64.0 B/clk/SM = 0.5
            // warp-instructions a clock (`B-lds-rtx-5080.txt`).
            lsu_ipc_pm: 500,
            // attention-chains: `calib.cu lds bcast` (`col`, g = 4 / 8 / 16:
            // 0.247 / 0.246 / 0.248 warp loads a clock).
            lds_bcast_pm: 246,
            // kernel-merging zero-fold (CUPTI, pt_rt_mlp2_2048, one traced
            // run an arm, `kernel-merging.md`): a 16 MiB zero pass 8.02 us
            // device time (mean of 9) less `zero_ns` = 2320 B/ns; the same
            // 16 MiB as the 16-B-store tail of a 16384 x 256 cell kernel adds
            // 9.64 us to it (23.30 vs 13.66 us median of 8) = 1740 B/ns.
            fill_bytes_per_ns: 2320,
            tail_bytes_per_ns: 1740,
            // kernel-merging `calls` (`calib.cu gsync`, block 256, median of three
            // runs, `kernel-merging.md` Calls): a grid barrier 553 ns at 84 blocks
            // (398-401 at 1-2, 541-546 at 16-42), + 1.89 ns a block to 504; a
            // dependent empty launch 650 ns start to start at <= 84 blocks (runs
            // 644 / 651-933 / 842-857 / 650-936: the low mode), flat to 504; the
            // same as a programmatic dependent launch 486 ns (484.7-599 at <= 84
            // blocks in the low mode, quiet box; 396-426 in two other runs):
            // cheaper than a barrier of 4 blocks or more; as a cooperative launch
            // (a persistent kernel's) 2044 ns, flat to 504 blocks (CUPTI agrees:
            // decode's x -> persistent-kernel gap 1559 ns against 272).
            gsync_ns: 553,
            gsync_block_ps: 1890,
            dep_launch_ns: 650,
            dep_launch_block_ps: 0,
            pdl_launch_ns: 486,
            coop_launch_ns: 2044,
            // hardware-properties `pinned` (`calib.cu copy`, three runs,
            // `local-rtx-5080/hardware-properties.md`): pageable H2D 27–37
            // GB/s against page-locked 40–48 at 4–256 MiB, median ratio 695.
            pageable_pm: 695,
            // long-rows `row-slices` (CUPTI, one block a row of a softmax
            // unit, 256 lanes, rows L2-resident, `long-rows.md`): 131.6 us
            // for 1 x 131072 (3 walks of 512 lane-iterations: 85.7 ns
            // each), 34.3 us for 4 x 32768 (3 x 128: 89.3 ns).
            row_iter_ns: 87,
            // long-rows (CUPTI driver-API trace, 7 one-iteration processes of
            // the sliced pt_lsm_4_32768): the first `cuLaunchCooperativeKernel`
            // 44.8 us median [39.4-52.7] against `cuLaunchKernelEx` 6.7 us;
            // the next ones of a process 3-7 us (a looped program).
            coop_first_ns: 38_000,
            // cross-arch `row-split`: one SM alone streams 94 B/ns from L2
            // (`calib.cu bw`, the best `sms=1` point, 2026-11-21).
            sm_bytes_per_ns: 94,
            // kernel-merging `calls`: measured (`calib.cu body`, six runs,
            // 2026-11-22, `review-fixes.md` §5): a 3-level chain whose wide
            // levels take 4 rounds of the persistent grid (the grid-stride
            // form) spends 1792 ns [1083–2596] more in its bodies than as
            // separate launches, the barriers against the launches aside
            // (one round: 292–420 ns); 4390 cycles at 2.45 GHz. The suite's
            // forced chains against the shipped build (ABBA, per chain beyond
            // the priced boundaries): rt_mlp2_1024 2510 ns (its GEMM pairs at
            // 127-128 registers against 95-96), decode_1024_256 1970 ns (the
            // rows level 5.0 us against 4.2 alone), causal / gen / prefill
            // -660..-810 ns (bodies faster).
            coop_body_cycles: COOP_BODY_CYCLES_RTX_5080,
            // stream-fly: `calib.cu bw` (`calib/B-bw-rtx-5080.tsv`): 90 % of the
            // 899 GB/s DRAM peak first at 8 KiB in flight an SM (84 SMs x 8
            // warps x 2 loads of 16 B: 890 GB/s; 4 KiB: 594-606).
            fly_bytes_per_sm: 8192,
            // hopper-wgmma-tma: `tma_calib.cu`, one elected thread an SM, every SM,
            // L2-resident: 197 ns a copy at 128 B - 4 KiB boxes for any ring
            // depth >= 2 (234 ns alone: the latency); 16 / 32 KiB boxes 289 / 654 ns
            // (the SM's 50-57 B/ns, 4.2-4.8 TB/s over the part).
            tma_copy_ns: 197,
            // integration: `l2keep.cu`, a cyclic read stream at 1.025× the
            // 64 MiB L2 re-read at 2.5 TB/s (L2 4.1, DRAM 0.76-0.92).
            l2_keep_bytes: 68_786_585,
            // flash-attention P1: `calib.cu bar`, 256 threads (16.9 ns).
            bar_ns: 17,
            split_cell_ps: SPLIT_CELL_PS_RTX_5080,
        }),
        // 128 FP32 lanes per SM (sm_120); E1 peak FFMA median 52.7 Tflop/s
        // [50.3–57.6] / (84 SMs · 128 · 2) = 2.45 GHz under load.
        fp32_lanes: 128,
        clock_khz: 2_450_000,
        // Device query: `mem_clock_khz=15001000 bus_bits=256` (GDDR7 at 30
        // Gb/s: 960 GB/s).
        mem_clock_khz: 15_001_000,
        // flash-attention P1 / bugs-properties: `calib.cu bar`, 44.6 cycles at
        // 256 threads (the measured `bar_ns` above prices this part).
        bar_cycles: 45,
        mem_bus_bits: 256,
        // ADR-0046 D8: the approximate math units this part's calibration
        // measured (`calib.cu math`).
        caps: Caps {
            math: MATH_RTX_5080,
            ..CAPS_WARP_MMA_PDL
        },
        place: &PLACE_RTX_5080,
        ..ADA_SM
    }),
    ..GENERIC
};

/// RTX 3090 — Ampere GA102 (`sm_86`), 82 SMs (S54). Compute capability 8.6
/// differs from Ada's per-SM facts in one field: 16 resident blocks per SM, not
/// 24 (CUDA C Programming Guide, compute-capability table); 100 KB shared memory
/// per SM, 99 KB per block, 1536 threads per SM, as on Ada.
pub const RTX_3090: TargetProfile = TargetProfile {
    name: "rtx-3090",
    machine: Machine::Gpu(Gpu {
        arch: "sm_86",
        sms: 82,
        max_blocks_per_sm: 16,
        // `facts l2_bytes=6291456` (base-rtx-3090.txt).
        l2_bytes: 6 * 1024 * 1024,
        // S62 `place`, measured (s62/cpugpu/run1/results-rtx-3090.partial-51025609.txt (the box dropped before its mm_4096 timing): pt_pj_m128_n18432_k7168 contract site kernel 22.33 Tflop/s; pt_pj_m1_n2048_k7168 site kernel 161 B/ns; pt_pc_64 all-device iter ms / 3 sites).
        cost: Some(DeviceCost {
            flops_per_ns: 22325,
            bytes_per_ns: 161,
            launch_ns: 18217,
            // cross-arch (`benches/results/evidence/cross-arch/README.md`, `calib.cu` on the vast box at
            // its stock 350 W, 2026-11-21): DRAM 891 / L2 2447 GB/s read, one
            // SM alone 62 B/ns; grid barrier 514 ns, dependent launch 1472 ns,
            // cooperative 2866 ns.
            l2_bytes_per_ns: 2447,
            dram_bytes_per_ns: 891,
            gsync_ns: 514,
            dep_launch_ns: 1472,
            coop_launch_ns: 2866,
            sm_bytes_per_ns: 62,
            // stream-fly: `calib.cu bw` (cross-arch 60f0f8ce): 90 % of the
            // 892 GB/s DRAM peak first at 8 KiB in flight an SM.
            fly_bytes_per_sm: 8192,
            bar_ns: 0,
            split_cell_ps: 0,
            // bugs-properties, measured on the cross-arch 5fcb401b box
            // (`calib-rtx-3090.txt`): the loaded latency, plan-cost's own
            // quantity (bytes in flight over the DRAM rate below 90 % of the
            // 892 GB/s peak, `calib.cu bw`): median 452 ns [IQR 439–486] over
            // 101 points; `calib.cu body`: 2102 cycles [2013–2193], 3 runs;
            // `l2keep.cu`: a cyclic stream of 1.0–1.02 × the 6 MiB L2
            // re-reads at 1201–1217 GB/s (the plateau: this stream's L2 rate on
            // the part, not the derived 2424), 1.025–1.05 × falls to 1166 /
            // 1101 / 971 / 900 / 874 / 811 (DRAM 892) — the 5080's sweep shape
            // (plateau 4.1–4.3 TB/s from 0.25 × up to 1.01 ×, the cliff past
            // it); the largest set at the plateau-DRAM midpoint (1054): 1.03 ×
            // L2 ([`l2_keep_of_sweep`], test `l2_keep_read_off_the_sweeps`).
            lat_ns: 452,
            coop_body_cycles: 2102,
            l2_keep_bytes: 6_480_199,
            ..NO_PLAN
        }),
        // GDDR6X at 19.5 Gb/s on 384 bits (936 GB/s); the FFMA peak under
        // load 32.77 Tflop/s = 1.561 GHz at 128 lanes (`calib.cu ffma`,
        // cross-arch; boost 1.695 GHz).
        clock_khz: 1_561_000,
        // bugs-properties: `calib.cu bar` 56.7 cycles (5fcb401b).
        bar_cycles: 57,
        mem_clock_khz: 9_751_000,
        mem_bus_bits: 384,
        caps: Caps {
            ptx: ptx_of_target(86, false),
            ..CAPS_WARP_MMA
        },
        place: &PLACE_RTX_3090,
        ..ADA_SM
    }),
    ..GENERIC
};

/// A100 SXM4 — Ampere GA100 (`sm_80`), 108 SMs (S54; the 40 GB and 80 GB SXM4
/// parts have the same SM). 164 KB shared memory per SM (`167936`), 163 KB per
/// block opt-in (`166912`), 2048 threads and 32 blocks per SM (CUDA C
/// Programming Guide, compute capability 8.0). The SXM4 variant because vast
/// listed four rentable single-GPU SXM4 offers to one PCIe on 2026-10-21; the
/// PCIe part (same SM facts, lower clocks) is a separate name, `a100-pcie`,
/// when it is measured — the `h100` / `h100-pcie` split.
pub const A100: TargetProfile = TargetProfile {
    name: "a100",
    machine: Machine::Gpu(Gpu {
        arch: "sm_80",
        sms: 108,
        smem_per_sm: 164 * 1024,
        smem_per_block: 163 * 1024,
        max_threads_per_sm: 2048,
        max_blocks_per_sm: 32,
        // `facts l2_bytes=41943040` (base-a100*.txt, PCIe 40GB and SXM4).
        l2_bytes: 40 * 1024 * 1024,
        // cross-arch (`benches/results/evidence/cross-arch/README.md`, `calib.cu` on the vast SXM4 40 GB
        // box, 2026-11-21): DRAM 1488 / L2 2793 GB/s read, one SM alone
        // 81 B/ns; grid barrier 1113 ns, dependent launch 2050 ns,
        // cooperative 4714 ns.
        cost: Some(DeviceCost {
            l2_bytes_per_ns: 2793,
            dram_bytes_per_ns: 1488,
            gsync_ns: 1113,
            dep_launch_ns: 2050,
            coop_launch_ns: 4714,
            sm_bytes_per_ns: 81,
            // stream-fly: `calib.cu bw` (cross-arch 60f0f8ce): 90 % of the
            // 1483 GB/s DRAM peak first at 16 KiB in flight an SM.
            fly_bytes_per_sm: 16384,
            // attention-chains: `calib.cu lds bcast` on two vast SXM4 40 GB
            // boxes (`col` 0.241 warp loads a clock at g = 4–32; `row` 0.393).
            lds_bcast_pm: 241,
            bar_ns: 0,
            split_cell_ps: 0,
            // bugs-properties, measured on the cross-arch 5fcb401b box
            // (`calib-a100.txt`): the loaded latency (`calib.cu bw`, plan-cost's
            // quantity): median 728 ns [IQR 686–802] over 112 points; `calib.cu
            // body`: 4048 cycles [3948–5569]; `l2keep.cu`: 1.0 × the 40 MiB L2
            // re-reads at 1563 GB/s, the DRAM rate (1487; L2 2862), flat to
            // 1.05 × — no plateau: the keep lies below the sweep's first point
            // (two partitions: a line every SM reads is held in both). Not
            // measured below 1.0 ×: half the L2 is a placeholder (open) until
            // the sweep from 0.25 × L2 (`l2keep.cu`, the confirmation round's
            // `ops_nsys.sh` step 0) is read by [`l2_keep_of_sweep`].
            lat_ns: 728,
            coop_body_cycles: 4048,
            l2_keep_bytes: 20 * 1024 * 1024,
            ..NO_PLAN
        }),
        // Datasheet (A100 40 GB): 64 FP32 lanes per SM, 1.41 GHz boost (the
        // FFMA peak under load measured 1.409 GHz), HBM2 1.555 TB/s on 5120
        // bits.
        fp32_lanes: 64,
        clock_khz: 1_410_000,
        // bugs-properties: `calib.cu bar` 53.7 cycles (5fcb401b).
        bar_cycles: 54,
        mem_clock_khz: 1_215_000,
        mem_bus_bits: 5120,
        // f32-tensor-cores P1: measured (`calib.cu matrix`, cross-arch
        // 5fcb401b): TF32 1012, bf16 / f16 2024 flops an SM a clock (the
        // datasheet's dense 156 / 312 Tflop/s over 108 SMs at 1.41 GHz: 1024
        // / 2048); Mapal's own TF32 tile realized 81.4, the best TF32 leg
        // 127.5 (matrix.md, pj_m2048_n28672_k4096).
        mma_tf32_per_clk: 1012,
        mma_half_per_clk: 2024,
        caps: CAPS_WARP_MMA,
        place: &PLACE_A100,
        ..ADA_SM
    }),
    ..GENERIC
};
