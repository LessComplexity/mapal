//! The machine facts as data: L1d, `Machine`, `Gpu`, `PlaceFacts`.

use super::*;

/// **The L1 data cache's geometry (S45) — three facts, and the fourth that is
/// arithmetic.**
///
/// `ways` is never recorded: for a set-associative cache
/// `ways = bytes / (line · sets)` is a definition, not an estimate, so recording
/// it could only ever make it wrong (the [`TargetProfile::sme_block`] precedent
/// — that is why `f32_tiles` was deleted).
///
/// `sets` **is** recorded, because each host has a different best source for it
/// and only the detector knows which:
///
/// | host | source for `sets` | status |
/// | --- | --- | --- |
/// | Linux | `/sys/…/index0/number_of_sets` | **read** — the truth, exposed |
/// | macOS | `hw.pagesize / hw.cachelinesize` | **derived** — `sysctl` exposes no set or way count |
///
/// The macOS derivation is the VIPT no-alias bound: an L1 indexed beyond the
/// page offset would alias, so `sets · line ≤ page`, and these parts sit at the
/// limit. **It is a checked heuristic, not a law**, and it is stated as one:
/// it reads the *configured* page rather than the architectural minimum, and
/// there are real parts it gets wrong — a PIPT L1 bigger than its page reach
/// (Cortex-A53, 32 KB 4-way: truth 128 sets, this says 64) and alias-handling
/// VIPT designs (AMD K8, 64 KB 2-way: truth 512, this says 64). It is used only
/// where nothing better is readable, and where it is used it is checked:
///
/// | part | line | sets | `ways()` | against the machine |
/// | --- | ---: | ---: | ---: | --- |
/// | M4 Pro P-core | 128 | 128 (16384/128) | **8** | matches the S44-verified 128-set / 8-way reading |
/// | i9-14900F P-core | 64 | 64 (read) | **12** | `ways_of_associativity` = **12** ✓ |
/// | i9-14900F E-core | 64 | 64 (read) | **8** | `ways_of_associativity` = **8** ✓ |
///
/// And the blast radius if the split is ever wrong is bounded by construction:
/// `sets · ways ≡ bytes / line` however the split falls, so a mis-split can only
/// mis-scale the gcd collapse in [`TargetProfile::move_block`], never the total
/// capacity. A factor-two error moves `slots` by two; the nearest measured
/// margin the decision turns on is pressure **21.3 against 1**.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L1d {
    /// Per-core L1 data cache in bytes.
    pub bytes: u64,
    /// Cache line in bytes — `hw.cachelinesize` / `coherency_line_size`.
    pub line_bytes: u64,
    /// Sets. Read where the host exposes it, derived from the page otherwise —
    /// see the type doc for which host does which, and for the two families of
    /// part where the derivation is wrong.
    pub sets: u64,
}

impl L1d {
    /// Ways, from capacity — `bytes / (line · sets)`, a definition.
    pub fn ways(&self) -> u64 {
        (self.bytes / (self.line_bytes * self.sets.max(1))).max(1)
    }
}

pub(super) fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Which machine class a profile describes.
///
/// This is a **capability discriminator**, not a branch flag (plan-s41 §2.2
/// rule 2): it selects which *realization* of a tile site fires, and each
/// realization is one cohesive unit of code. It must never appear as an `if` in
/// the body of a shared emitter — that is the code-locality failure the rule
/// exists to prevent, and §9's branch budget is what measures it.
///
/// The classes are not different algorithms. Every unit Mapal targets — SIMD
/// FMA, ARM SME, Intel AMX, NVIDIA tensor cores — does the same thing: stage
/// operands, issue a multiply-accumulate over a block, accumulate into a
/// **resident** accumulator, keep it hot across the reduction axis, store once
/// at the end. What varies is the shape of one issue and how many accumulator
/// blocks stay resident — and [`TargetProfile::tile_i`] already parameterizes
/// the second. So the tile nest is shared and only the innermost leaf differs
/// (Sapir, S41 plan gate).
///
/// The one genuinely structural split is **cooperation**: on a CPU core a
/// single thread owns its whole tile, while on a GPU a block of threads shares
/// one staged panel and must synchronise before reading it. That is why `Gpu`
/// carries facts `Cpu` has no analogue for, and why a barrier is a morphism in
/// the model rather than an idiom (plan-s41 §1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Machine {
    /// A CPU core. One thread owns a whole tile; accumulators live in the
    /// vector register file; no cooperation, no barrier. Every profile that
    /// existed before S41 is this, which is what keeps their emission
    /// byte-identical.
    Cpu,
    /// A GPU streaming multiprocessor.
    Gpu(Gpu),
}

/// The SM facts a GPU profile carries, and only those a CPU profile has no
/// field for. Everything a CPU profile already expresses (accumulator capacity,
/// block width, panel depth) stays in [`TargetProfile`] — the point of the
/// unification above is that those are *the same quantities*, read differently.
///
/// **UNMEASURED.** These are read off the ISA and the device query, not swept,
/// exactly like [`ZEN3`]. No realization consumes them yet; plan-s41 steps 3–4
/// decide which are actually read, and any fact a GPU realization needs but
/// cannot find here is an ADR-0033 D4(b) finding to report rather than a value
/// to invent.
///
/// **S52: every field is a device-query attribute** (`cuDeviceGetAttribute`,
/// printed by `benches/gpu/devquery.cu`), so a hand-written profile is the
/// cross-compilation spelling of a part and the probe on the part is its check
/// — the ADR-0038 D4 pattern the SME facts follow. Nothing here is swept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gpu {
    /// Threads per warp — the cooperation quantum. 32 on every NVIDIA part.
    pub warp: u64,
    /// The launch-geometry ceiling: threads in one block.
    pub max_threads_per_block: u64,
    /// The PTX target architecture, e.g. `sm_89`. Verified against the device
    /// rather than assumed: the box GPU reports compute capability 8.9.
    pub arch: &'static str,
    /// Streaming multiprocessors — how many blocks one wave holds, per
    /// resident slot. A property of the part, not the architecture.
    pub sms: u64,
    /// 32-bit registers one thread may address (255 on every part since
    /// Kepler). The accumulator budget — the GPU reading of `vec_regs`.
    pub regs_per_thread: u64,
    /// 32-bit registers one SM holds. With the per-thread estimate it bounds
    /// how many threads are resident at once.
    pub regs_per_sm: u64,
    /// Shared memory one SM holds, in bytes — split between its resident
    /// blocks. The staged panels are sized against a block's share of it.
    pub smem_per_sm: u64,
    /// Shared memory one block may opt in to, in bytes (above 48 KB a launch
    /// must raise `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
    pub smem_per_block: u64,
    /// Shared memory the driver reserves per resident block (1 KB on sm_80+).
    pub smem_reserved_per_block: u64,
    /// Resident threads per SM.
    pub max_threads_per_sm: u64,
    /// Resident blocks per SM.
    pub max_blocks_per_sm: u64,
    /// The device-wide L2, in bytes (`cudaDevAttrL2CacheSize`, the box's
    /// `facts l2_bytes=`). One cache every SM shares — the GPU reading of the
    /// CPU's shared last level, not a per-SM quantity (S54 `raster`,
    /// `l2-window`).
    pub l2_bytes: u64,
    /// What a device site costs on this part, measured (S62 `place`). `None`:
    /// every capable site is a device site — the behaviour before it existed.
    pub cost: Option<DeviceCost>,
    /// The part's launch-geometry constants (ADR-0034 D1): the row the
    /// offline search writes, overridable per knob (`--geom=`).
    pub geom: Geom,
    /// FP32 FMA lanes one SM issues per clock (64 on the A100's SM, 128 on
    /// every other profiled one) — the datasheet's figure, or the calibration
    /// file's measured FFMA peak over `sms · 2 · clock`. Plan cost and the
    /// FFMA-per-byte balance ([`Self::ffma_per_dram_byte`]) read it.
    pub fp32_lanes: u64,
    /// The SM clock, kHz: the **sustained** clock under continuous FFMA load
    /// where measured (`calib.cu ffma`'s `sustained_clock_khz`: what a power
    /// or thermal limit holds — the 250 W RTX 4090 box 1.96 GHz against its
    /// 2.52 GHz boost), else the device query's `CU_DEVICE_ATTRIBUTE_CLOCK_RATE`.
    /// Rates read it (FFMA peak, plan cost); no rung rule reads it alone.
    pub clock_khz: u64,
    /// hardware-properties: the DRAM interface clock, kHz
    /// (`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`); with [`Self::mem_bus_bits`]
    /// the peak DRAM rate ([`Self::dram_peak_bytes_per_ns`]).
    pub mem_clock_khz: u64,
    /// hardware-properties: the DRAM bus width, bits
    /// (`CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`).
    pub mem_bus_bits: u64,
    /// Warp schedulers per SM, each issuing one warp per clock (four on every
    /// profiled SM): the fill rule's unit ([`Self::fill_warps`]).
    pub schedulers: u64,
    /// bugs-properties: **one 256-thread block barrier with a dependent
    /// shared-memory round trip, in SM cycles** (`calib.cu bar`, clock64
    /// spans) — the SM design's, not its clock's: the Ampere / Ada SMs
    /// measure 54–57 (A100 53.7, RTX 3090 56.7, RTX 4090 57.2), Hopper 41.8
    /// (H100), consumer Blackwell 44.6 (RTX 5080; cross-arch 5fcb401b). The
    /// barrier's ns where the part measured none ([`Gpu::bar_ns`]): these
    /// cycles at its clock (the 5080's 41 at every part's clock read 26 / 29
    /// / 16 ns against 33.5 / 38.1 / 22.5 measured on the 3090 / A100 / 4090).
    pub bar_cycles: u64,
    /// f32-tensor-cores P1: **flops one SM issues a clock through the warp
    /// TF32 product** (`mma.sync.m16n8k8`, operands in registers): `calib.cu
    /// matrix` (32 warps an SM, independent register chains, clock64 spans
    /// — the RTX 5080's 256, equal to its FFMA's), else the datasheet's dense
    /// TF32 rate over `sms · clock`. The split rule weighs it against the
    /// FFMA rate ([`Self::matrix_flops_per_ns`]).
    pub mma_tf32_per_clk: u64,
    /// f32-tensor-cores P1: the same for the warp 16-bit products with f32
    /// accumulators (`mma.sync.m16n8k16` bf16 / f16: the RTX 5080's 512).
    pub mma_half_per_clk: u64,
    /// f32-tensor-cores P1: the same for the warpgroup TF32 product
    /// (`wgmma.m64nNk8`, `a` from a shared-memory descriptor); 0 where the
    /// part has none.
    pub wgmma_tf32_per_clk: u64,
    /// wgmma-16bit: the same for the warpgroup 16-bit products with f32
    /// accumulators (`wgmma.m64nNk16` bf16 / f16, `b` from a shared-memory
    /// descriptor, `a` from either home — `calib.cu matrix`'s `wgmma` kinds);
    /// 0 where the part has none.
    pub wgmma_half_per_clk: u64,
    /// What the SM executes, as properties ([`Caps`]): no rule compares a
    /// version.
    pub caps: Caps,
    /// ADR-0050: the placement facts this part measured beyond its
    /// [`DeviceCost`] and the provenance of every value of the profile
    /// ([`PlaceFacts`]) — what `place::machine_of` translates.
    pub place: &'static PlaceFacts,
}

/// **ADR-0050: the placement facts of a part beyond its [`DeviceCost`], and
/// the provenance of the profile's values** — each value's source next to the
/// value: `prov` names a field ([`DeviceCost::NAMES`], or `bar_cycles` /
/// `fp32_lanes`) and where its number comes from. A field the list does not
/// name is a device-query attribute where it is one (queried), else unlabeled
/// (nominal, visibly). The machine's translation (`place::machine_of`) reads
/// these and marks every derived or nominal value; nothing here is borrowed
/// from another part as if measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceFacts {
    pub prov: &'static [(&'static str, mapal_place::Src)],
    /// The start timing of a grid's slots past its first wave (`None`:
    /// unmeasured — the evaluator takes both at its bounds).
    pub start: Option<(mapal_place::Start, &'static str)>,
    /// The L2's keep for a walk's rotated pattern and for a strided one,
    /// bytes (0: unmeasured — the cyclic keep, marked).
    pub keep_rotated: u64,
    pub keep_strided: u64,
    /// The keep sweep's resolution: the next working set it measured past
    /// the keep (each pattern's keep lies in `[keep, keep_hi)`; 0: a point).
    pub keep_hi: u64,
    /// The L2's hit fraction against reuse distance ÷ cyclic keep, both per
    /// mille.
    pub hit_curve: &'static [(u64, u64)],
    /// The L2's loaded latency, ns (0: unmeasured), and the run it comes
    /// from where that is not [`PlaceFacts::cite`]'s (empty: the cite's) —
    /// a value from another box of the part keeps its own provenance.
    pub lat_l2_ns: u64,
    pub lat_l2_cite: &'static str,
    /// Measured efficiency points: `(operation, chains a lane, groups a
    /// scheduler × 1000, achieved fraction of the row's rate per mille)`.
    pub eff: &'static [(&'static str, u64, u64, u64)],
    /// Measured dependent-issue latencies: `(operation, unit clocks × 10)`.
    pub latency: &'static [(&'static str, u64)],
    /// The SM clock under sustained load, kHz, measured by clock64 over
    /// %globaltimer (0: unmeasured — the profile's `clock_khz`). Read by the
    /// placement model only: the rules keep the profile's clock.
    pub clock_khz: u64,
    /// **The cost model's primitives** (`place_calib.cu prim`): per probe
    /// name its saturated rate, one use's latency and the concurrency that
    /// saturates it, each with its interval — `[rate, lo, hi, latency, lo,
    /// hi, conc, lo, hi]` in milli-units of the probe's units (compute:
    /// flop / clock / SM, clocks, instructions / SM; movers and stores:
    /// bytes / ns / SM, ns, bytes / SM; the scratchpad: bytes / clock / SM,
    /// clocks; reductions: operations / ns / SM; barriers: clocks; launches:
    /// ns). `place::machine_of` reads each as a [`mapal_place::Prim`] (empty:
    /// unmeasured — every cost derived and marked).
    pub prim: &'static [(&'static str, [u64; 9])],
    /// The cite of the efficiency, latency and clock probes.
    pub cite: &'static str,
}

/// No placement facts beyond the profile's.
pub const PLACE_NONE: PlaceFacts = PlaceFacts {
    prov: &[],
    start: None,
    keep_rotated: 0,
    keep_strided: 0,
    keep_hi: 0,
    hit_curve: &[],
    lat_l2_ns: 0,
    lat_l2_cite: "",
    eff: &[],
    latency: &[],
    clock_khz: 0,
    prim: &[],
    cite: "",
};
