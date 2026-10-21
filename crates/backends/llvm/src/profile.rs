//! `TargetProfile` — the emitter's machine facts as data (plan-s31-target-profiles).
//!
//! ADR-0032 D4: tile factors, grain sizes and arena thresholds are backend
//! config, not language. Every field here is **value-invariant** — changing one
//! changes how fast the answer arrives, never the answer — which is exactly what
//! keeps the differential suite a valid gate under every profile.
//!
//! The split this file exists to enforce (the backend-genericity contract):
//! **geometry comes from the record, constants come from the profile.** mapal-ir
//! never learns a machine fact; the emitter never re-derives graph analysis.
//! What used to be six literals swept on one M4 Pro is now one table plus
//! arithmetic, and the arithmetic reproduces those literals for the default
//! profile — that is the correctness gate for this change.

use mapal_ir::{MoveSite, Ty};

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

fn gcd(a: u64, b: u64) -> u64 {
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

impl Gpu {
    /// The compute capability as a number: `sm_89` → 89. The capability
    /// table below is read off it — ISA rules, not part facts.
    pub fn sm(&self) -> u64 {
        self.arch
            .trim_start_matches("sm_")
            .trim_end_matches('a')
            .parse()
            .unwrap_or(0)
    }

    /// `mma.sync` with TF32 operands and `cp.async` both arrived at sm_80.
    pub fn has_mma_sync(&self) -> bool {
        self.sm() >= 80
    }

    /// Warp schedulers per SM: four since sm_70 (Volta through Blackwell),
    /// each issuing one warp per cycle. An ISA table fact, like
    /// [`Self::has_mma_sync`]. Older parts are not profiled; 1 leaves the fill
    /// rule no stronger than "a block per SM".
    pub fn schedulers(&self) -> u64 {
        if self.sm() >= 70 { 4 } else { 1 }
    }

    /// The warps a grid needs before every scheduler holds
    /// [`GPU_RESIDENT_BLOCKS`] of them — that constant's reason applied per
    /// scheduler: one warp issues while the other waits on a load.
    pub fn fill_warps(&self) -> u64 {
        GPU_RESIDENT_BLOCKS * self.schedulers() * self.sms
    }

    /// How full a grid of `blocks × threads` keeps the part: warps per warp
    /// scheduler, across every SM.
    pub fn warps_per_scheduler(&self, blocks: u64, threads: u64) -> u64 {
        blocks * threads.div_ceil(self.warp) / (self.sms * self.schedulers())
    }

    /// `wgmma.mma_async` exists only on the arch-specific Hopper target
    /// `sm_90a` (PTX ISA: "Requires sm_90a"); Blackwell replaced it.
    pub fn has_wgmma(&self) -> bool {
        self.sm() == 90 && self.arch.ends_with('a')
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
    ) -> u64 {
        let (gx, gy) = (c / bn.max(1), rows / bm.max(1));
        let wave = wave.max(1);
        if gx < 2 || gy < 2 || gx * gy <= wave || (gate && (bm + bn) * k * elem <= self.l2_bytes) {
            return 1;
        }
        (0..=gy.ilog2())
            .map(|p| 1u64 << p)
            .filter(|h| gy.is_multiple_of(*h))
            .min_by_key(|&h| h * bm + wave.div_ceil(h) * bn)
            .unwrap_or(1)
    }
}

/// Which grids a measured rung decision covers. The key is
/// [`Gpu::warps_per_scheduler`] at the site: below [`GPU_RESIDENT_BLOCKS`] a
/// scheduler has no second warp to issue while one waits — the reason the
/// thread budget is split in two, applied per scheduler — so the site is
/// latency-bound and a rung that hides latency pays differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    Any,
    /// Fewer than [`GPU_RESIDENT_BLOCKS`] warps per scheduler.
    Under,
    /// At least [`GPU_RESIDENT_BLOCKS`] warps per scheduler.
    Filled,
}

/// Which face a measured rung decision covers: the conformance face (`mul.rn`
/// + `add.rn`, two instructions per step) and the contract face (one `fma`)
/// hold different registers live, so a staging rung can pay on one and cost
/// on the other. The TF32 face is a contract face: a `Contract` row covers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    Any,
    Exact,
    Contract,
    /// The TF32 face alone (S54): its kernel is `mma`, not the FMA tile, so a
    /// rung that shapes both (`min-blocks`) can pay on one and cost on the
    /// other — the kernel the rung folds into, read off the face.
    Tf32,
}

impl Face {
    /// The face a site is emitted on.
    pub fn of(contract: bool, reduced: Option<crate::Reduced>) -> Self {
        match (reduced, contract) {
            (Some(crate::Reduced::Tf32), _) => Face::Tf32,
            (None, true) => Face::Contract,
            (None, false) => Face::Exact,
        }
    }

    /// Whether a row keyed `self` covers a site on face `site`.
    pub fn covers(self, site: Face) -> bool {
        self == Face::Any || self == site || (self == Face::Contract && site == Face::Tf32)
    }
}

/// One measured decision: `rung` ships OFF on `profile` for grids `fill` on
/// face `face`, because the results row `cite` measured its arm not positive
/// there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RungOff {
    pub profile: &'static str,
    pub rung: &'static str,
    pub fill: Fill,
    pub face: Face,
    pub cite: &'static str,
}

/// **The GPU rung table: every rung a measured part ships OFF, with the row
/// that says so** (S53; the [`DispatchCost`] precedent — a measured number
/// lives only on the profile it was measured on). A profile is *measured* when
/// it is in [`GPU_MEASURED`]; every rung not listed for it is ON. Rows cite
/// `benches/results-s53-gpu/<file>` or (`base-*`, S54) `benches/results-s54-gpu/<file>`
/// (`-confirm` / `-run1` = another session, on a different box): the ON vs OFF
/// arm medians, ms, or (S54) OFF/ON divided by each bench's own cuBLAS leg, which
/// cancels the clock drift of a power-capped box. Box-to-box noise is ±5% at
/// 4096–8192. The rule a row follows: a rung ships OFF in a (fill, face) class
/// when no size in the class measured it positive with a stable sign (a sign
/// that changes between sessions is no effect; S54: within ±1% is no effect, the
/// median spread of one ratio between two boxes).
pub const GPU_RUNGS_OFF: &[RungOff] = &[
    RungOff {
        profile: "rtx-4090",
        rung: "async",
        fill: Fill::Filled,
        face: Face::Any,
        cite: "arch-rungs-rtx-4090.txt _c n=2048/4096/8192 on 0.368/3.027/28.393 vs off 0.359/2.823/25.554 ms, \
               -confirm.txt on 0.374/3.087/28.648 vs off 0.362/2.843/26.796 ms; _exact with fma-regs on \
               (fma-regs-rtx-4090.txt) cp.async 0.508/4.437/39.162 vs register staging 0.492/4.245/37.149 ms, \
               -run1.txt 0.520/4.520/39.002 vs 0.502/4.249/36.698 ms; kept under-filled (_c n=1024 on 0.070 vs \
               off 0.101 ms)",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "warp",
        fill: Fill::Filled,
        face: Face::Contract,
        cite: "arch-rungs-rtx-4090-confirm.txt _c n=2048/4096/8192 on 0.362/2.843/26.796 vs off 0.355/2.813/25.311 ms; \
               arch-rungs-rtx-4090.txt (unroll on) on 0.359/2.823/25.554 vs off 0.364/2.863/25.788 ms: no stable sign",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "ldmatrix",
        fill: Fill::Any,
        face: Face::Any,
        cite: "arch-rungs-rtx-4090.txt _tf n=1024..8192 on 0.084/0.255/1.733/14.402 vs off 0.084/0.254/1.731/14.387 ms",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "idx32",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "fma-regs-rtx-4090.txt _c n=1024 (cp.async) on 0.068 vs off 0.067 ms: no effect",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "affine",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "fma-regs-rtx-4090.txt _c n=1024 (cp.async) on 0.068 vs off 0.068 ms: no effect",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "async",
        fill: Fill::Any,
        face: Face::Any,
        cite: "arch-rungs-h100-pcie.txt _c n=1024..8192 on 0.146/0.717/4.188/37.655 vs off 0.139/0.675/3.889/33.640 ms; \
               _exact on 0.240/1.210/8.292/67.130 vs off 0.193/1.127/6.428/54.641 ms",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "min-blocks",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "arch-rungs-h100-pcie.txt _c n=1024 on 0.139 vs off 0.122 ms (v2/v4/v5/v6-h100*.txt the same sign); \
               kept filled (n=2048/4096/8192 on 0.675/3.889/33.640 vs off 0.793/4.753/41.168 ms)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "wide-fma",
        fill: Fill::Any,
        face: Face::Any,
        cite: "fma-regs-h100-pcie.txt register staging n=2048/4096/8192 _c on 0.795/4.054/33.097 vs off \
               0.654/3.851/33.608 ms, _exact on 1.491/7.223/57.044 vs off 1.118/6.480/55.381 ms",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "idx32",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "fma-regs-h100-pcie.txt _exact register staging, wide-fma off: idx32+affine+pad n=2048/4096/8192 \
               1.118/6.480/55.381 vs none 1.124/6.482/55.339 ms: no effect",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "affine",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "fma-regs-h100-pcie.txt _exact register staging, wide-fma off: idx32+affine+pad n=2048/4096/8192 \
               1.118/6.480/55.381 vs none 1.124/6.482/55.339 ms: no effect; integrated-h100-pcie-confirm.txt _exact \
               n=2048/4096/8192 off 1.08x/1.20x/1.23x faster than on",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "affine",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "fma-regs-h100-pcie.txt _c n=1024 register staging on 0.131 vs off 0.127 ms; integrated-h100-pcie.txt \
               _c n=1024 on 0.921x of off; kept filled with min-blocks off (integrated-h100-pcie-confirm.txt _c \
               n=2048/4096/8192 on 0.622/3.734/36.665 vs off 0.644/3.834/37.387 ms)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "pad",
        fill: Fill::Filled,
        face: Face::Exact,
        cite: "fma-regs-h100-pcie.txt _exact register staging, wide-fma off: idx32+affine+pad n=2048/4096/8192 \
               1.118/6.480/55.381 vs none 1.124/6.482/55.339 ms: no effect; base-h100-pcie.txt / -confirm.txt \
               (S54) _exact n=2048/4096/8192 OFF/ON cuBLAS-normalized 0.889/0.855/0.860 vs 0.884/0.842/0.825 (ms in \
               the files); ON under-filled since (n=1024 1.037 vs 1.031, disjoint in both)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "min-blocks",
        fill: Fill::Filled,
        face: Face::Any,
        cite: "integrated-h100-pcie.txt (with idx32+pad, the cap spills 8-16 B) _c n=2048/4096/8192 on \
               0.656/3.950/36.832 vs off 0.618/3.729/37.530 ms, _exact on 1.118/6.521/57.781 vs off \
               1.089/6.366/58.483 ms; kept under-filled on the exact face (n=1024 on 0.193 vs off 0.213 ms)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "warp",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "integrated-h100-pcie.txt _c n=1024 on 0.139 vs off 0.136 ms; kept filled (n=2048/4096/8192 on \
               0.656/3.950/36.832 vs off 0.700/4.112/41.546 ms)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "warp-fill",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "arch-select-h100-pcie.txt / base-h100-pcie-verify.txt (S54) _c n=1024 on 0.1512 / 0.1524 vs off 0.1445 / \
               0.1464 ms (cuBLAS-normalized OFF/ON 0.952 / 0.957, disjoint in both); kept on the exact face (on 0.2144 / \
               0.2163 vs off 0.2192 / 0.2214 ms, 1.025 / 1.028, disjoint in both)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "async",
        fill: Fill::Filled,
        face: Face::Contract,
        cite: "arch-rungs-rtx-5090.txt _c n=2048/4096/8192 on 0.320/2.346/20.277 vs off 0.315/2.328/19.542 ms, \
               -confirm.txt on 0.358/3.043/27.908 vs off 0.352/2.991/27.069 ms; kept under-filled (n=1024 on 0.074 \
               vs off 0.092) and on the exact face (n=2048 on 0.473/0.526 vs off 0.491/0.537 ms, sign unstable at 4096/8192)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "wide-fma",
        fill: Fill::Any,
        face: Face::Any,
        cite: "fma-regs-rtx-5090.txt _c register staging n=2048/4096/8192 on 0.344/3.194/26.885 vs off \
               0.348/2.947/28.434 ms, -run1.txt on 0.313/2.592/19.681 vs off 0.311/2.360/19.382 ms (4096 disjoint)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "idx32",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "fma-regs-rtx-5090.txt _exact cp.async n=1024..8192 on 0.076/0.588/4.920/40.005 vs off \
               0.076/0.586/4.907/40.004 ms: no effect",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "affine",
        fill: Fill::Filled,
        face: Face::Exact,
        cite: "fma-regs-rtx-5090.txt _exact cp.async n=2048/4096/8192 on 0.588/4.920/40.005 vs off \
               0.580/4.876/39.878 ms; kept under-filled (n=1024 on 0.076 vs off 0.084 ms)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "pad",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "fma-regs-rtx-5090.txt _exact register staging n=2048/4096/8192 on 0.545/4.446/39.511 vs off \
               0.541/4.613/38.788 ms: no stable sign",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "idx32",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "fma-regs-rtx-5090.txt _c n=1024 (cp.async) on 0.074 vs off 0.074 ms: no effect",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "affine",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "fma-regs-rtx-5090.txt _c n=1024 (cp.async) on 0.074 vs off 0.078 ms, ranges overlap, one session: \
               not established",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "warp-fill",
        fill: Fill::Filled,
        face: Face::Contract,
        cite: "warp-fill-rtx-5090.txt _tf n=2048 on 0.2314 vs off 0.2252 ms, disjoint; _c and _exact PTX \
               byte-identical at every filled size",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "retain-delta",
        fill: Fill::Any,
        face: Face::Any,
        cite: "integrated-rtx-4090.txt decode_1024_256 wall, rung on (resident yields) 1589/1568 ms vs off \
               (resident takes the loop) 905/886 ms exact/contract, 3 procs, disjoint; against S52 round trips the rung \
               measured 1.57/1.65x (retain-delta-rtx-4090.txt) — resident is the better arm",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "retain-delta",
        fill: Fill::Any,
        face: Face::Any,
        cite: "integrated-h100-pcie.txt decode_1024_256 wall, rung on (resident yields) 1756/1906 ms vs off \
               (resident takes the loop) 1109/1144 ms exact/contract, 3 procs; against S52 round trips 1.53/1.50x \
               (retain-delta-h100-pcie.txt)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "async",
        fill: Fill::Under,
        face: Face::Any,
        cite: "integrated-rtx-5090.txt (with warp-fill's 4x4 tile) n=1024 _c on 0.0635 vs off 0.0573 ms, _exact on \
               0.0737 vs off 0.0676 ms, disjoint; before warp-fill it paid (arch-rungs-rtx-5090.txt n=1024 on 0.074 vs \
               off 0.092 ms)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "wide",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "integrated-rtx-5090.txt _tf n=1024 on 0.0492 vs off 0.0389 ms, disjoint",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "retain-delta",
        fill: Fill::Any,
        face: Face::Any,
        cite: "integrated-rtx-5090.txt decode_1024_256 wall, rung on (resident yields) 825/847 ms vs off \
               (resident takes the loop) 630/616 ms exact/contract, 3 procs, disjoint; against S52 round trips \
               1.63/1.63x (retain-delta-rtx-5090.txt)",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "wide",
        fill: Fill::Filled,
        face: Face::Contract,
        cite: "base-rtx-5090.txt / base-rtx-5090-confirm.txt (S54) _tf n=2048/4096/8192 OFF/ON cuBLAS-normalized \
               0.981/0.993/1.006 vs 0.982/0.999/1.031 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "affine",
        fill: Fill::Under,
        face: Face::Any,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.996 vs 1.019; _exact n=1024 1.041 vs 1.006 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "async",
        fill: Fill::Any,
        face: Face::Any,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _c n=2048/4096/8192 0.906/0.896/0.928 vs 0.907/0.897/0.879; _exact n=2048/4096/8192 0.936/0.958/0.994 vs 0.925/0.930/0.958; _c n=1024 0.897 vs 0.878; _exact n=1024 1.000 vs 0.951 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "idx32",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _exact n=2048/4096/8192 0.993/1.004/0.997 vs 1.000/1.005/0.995; _exact n=1024 1.003 vs 0.996 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "idx32",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.994 vs 1.020 (ms in the files): no size ON-faster in both sessions; with warp-fill off under-filled (S54 arch-select) ON costs: integrated-rtx-3090.txt _c n=1024 0.944 (ms 0.1659 ON vs 0.1556 OFF)",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "interleave",
        fill: Fill::Under,
        face: Face::Any,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.990 vs 1.000; _exact n=1024 1.003 vs 0.991 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "ldmatrix",
        fill: Fill::Any,
        face: Face::Contract,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _tf n=2048/4096/8192 0.998/1.006/1.019 vs 1.000/1.000/0.997; _tf n=1024 1.000 vs 1.020 (ms in the files): no size ON-faster in both sessions",
    },
    // S54 arch-select: the FMA tile takes `.minnctapersm` under register
    // staging on both faces (exact n=1024 1.032 / 1.094 / 1.090, the first
    // against the shipped bench's null repeat; integrated-rtx-3090.txt exact
    // n=1024 1.108, contract n=2048/8192 0.9933 / 68.68 ms vs old 1.0445 /
    // 70.80), the TF32 `mma` kernel does not — the third face key.
    RungOff {
        profile: "rtx-3090",
        rung: "min-blocks",
        fill: Fill::Any,
        face: Face::Tf32,
        cite: "arch-select-rtx-3090.txt / base-rtx-3090-verify2.txt / -verify.txt OFF/ON cuBLAS-normalized: _tf n=1024 \
               0.890/0.890/0.890 (ms 0.0993 shipped vs 0.1116 ON, cuBLAS 0.0881, both sessions), n=2048/4096/8192 \
               0.997/1.002/1.000, 1.000/0.993/1.000, 0.994/0.994/1.006: never ON-faster; FMA face ON (_c n=1024 \
               1.034/1.014/1.041, n=2048 1.057/1.052/1.086, n=8192 1.086/1.070/1.046; arch-select ms n=2048 0.8745 \
               vs 0.8325, n=8192 51.971 vs 48.936)",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "warp",
        fill: Fill::Under,
        face: Face::Any,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.988 vs 1.006; _exact n=1024 1.022 vs 0.991 (ms in the files): no size ON-faster in both sessions; on warp-fill's tile it paid under-filled contract (arch-select-rtx-3090.txt / -verify2 / -verify 1.055/1.036/1.056); with warp-fill off there (S54 arch-select) it changes no text",
    },
    // S54 arch-select, one coordinate step for the under-filled FMA class: four
    // single flips each paid on the warp-fill tile (idx32 ON 1.070/1.050/1.071,
    // warp ON 1.055/1.036/1.056, unroll OFF 0.948/0.959/0.940, min-blocks ON
    // 1.033/1.014/1.041) but all of them with warp-fill OFF ran 6% slower than
    // the S54 table (integrated-rtx-3090.txt _c n=1024 0.1659 vs old 0.1546 ms;
    // on that table idx32 ON 0.944, unroll OFF 0.931): warp-fill alone ships.
    RungOff {
        profile: "rtx-3090",
        rung: "warp-fill",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "arch-select-rtx-3090.txt / base-rtx-3090-verify2.txt / -verify.txt OFF/ON cuBLAS-normalized, disjoint in \
               every session: _c n=1024 0.935/0.959/0.940 (ms 0.1546 vs OFF 0.1444; -verify2 0.1608 vs 0.1454), _tf \
               n=1024 0.887/0.887/0.876 (ms 0.0993 vs OFF 0.0881 = cuBLAS's 0.0881; integrated-rtx-3090.txt 0.0881 vs \
               old 0.0993); kept on the exact face (_exact n=1024 1.057 / 0.961 / 0.960: no stable sign)",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "wide",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _tf n=1024 0.812 vs 1.009 (ms in \
               the files): no size ON-faster in both sessions; kept filled: _tf n=2048/4096/8192 1.011/1.017/1.023 vs \
               1.008/1.010/1.004, and with it off base-rtx-3090-verify.txt _tf 2048/4096 0.781/5.358 vs 0.741/5.129 ms \
               (cuBLAS 0.681/5.141 vs 0.677/5.090)",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "wide-fma",
        fill: Fill::Filled,
        face: Face::Exact,
        cite: "base-rtx-3090.txt / base-rtx-3090-confirm.txt OFF/ON cuBLAS-normalized: _exact n=2048/4096/8192 0.959/0.936/0.991 vs 0.956/0.904/0.961 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "retain-delta",
        fill: Fill::Any,
        face: Face::Any,
        cite: "base-rtx-3090.txt decode_1024_256 wall, rung on (resident yields) 2.715/2.697 s vs off (resident takes \
               the loop) 1.473/1.473 s exact/contract, 3 procs; base-rtx-3090-confirm.txt 1.386/1.396 vs 0.935/0.942 s (ms: \
               the files' trace totals)",
    },
    RungOff {
        profile: "a100",
        rung: "affine",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _exact n=2048/4096/8192 1.004/1.006/0.991 vs 1.019/1.012/0.986; _exact n=1024 0.997 vs 0.993 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "a100",
        rung: "async",
        fill: Fill::Any,
        face: Face::Any,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _c n=2048/4096/8192 0.940/0.908/0.874 vs 0.951/0.917/0.883; _exact n=2048/4096/8192 0.790/0.894/0.901 vs 0.810/0.910/0.909; _c n=1024 0.862 vs 0.867; _exact n=1024 0.950 vs 0.940 (ms in the files): no size ON-faster in both sessions",
    },
    // S54 arch-select closes the open filled-contract row: two sessions on the
    // shipped table (async and wide-fma off) put idx32 ON there.
    RungOff {
        profile: "a100",
        rung: "idx32",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _exact n=2048/4096/8192 0.979/0.996/0.997 vs \
               1.008/1.003/1.004; _exact n=1024 0.993 vs 0.977 (ms in the files); arch-select-a100.txt / \
               base-a100-verify3.txt _exact n=1024 0.900 / 0.899, n=2048/4096/8192 0.987/0.992/0.992 vs \
               1.000/0.992/0.991: never ON-faster",
    },
    RungOff {
        profile: "a100",
        rung: "idx32",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "arch-select-a100.txt / base-a100-verify3.txt OFF/ON cuBLAS-normalized _c n=1024 0.987 / 0.989: never \
               ON-faster; ON filled since (_c n=4096 1.015 / 1.013, n=8192 1.015 / 1.018; ms n=4096 8.153 / 7.904 \
               shipped vs 8.034 / 7.800 ON, n=8192 62.890 / 59.666 vs 61.810 / 58.606)",
    },
    RungOff {
        profile: "a100",
        rung: "interleave",
        fill: Fill::Under,
        face: Face::Any,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.990 vs 0.993; _exact n=1024 1.009 vs 0.989 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "a100",
        rung: "min-blocks",
        fill: Fill::Filled,
        face: Face::Any,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized (cp.async): no size ON-faster in both sessions (_c n=2048/4096/8192 1.001/1.010/1.008 vs 1.013/1.001/1.002, ms in the files); with async off (register staging): base-a100-verify.txt then base-a100-verify2.txt filled _c 1.018/1.033/1.041 then 1.000/1.000/1.000, _exact 1.001/0.999/0.998 then 1.006/1.000/1.000: no stable sign; ON under-filled (_c n=1024 1.119 then 1.110, _exact 1.084 then 1.080)",
    },
    RungOff {
        profile: "a100",
        rung: "unroll",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _exact n=2048/4096/8192 0.856/0.936/0.940 vs 0.871/0.941/0.944; _exact n=1024 1.018 vs 1.007 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "a100",
        rung: "warp",
        fill: Fill::Any,
        face: Face::Any,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _c n=1024 0.991 vs 0.987; _exact n=1024 1.001 vs 0.991 (ms in the files): no size ON-faster in both sessions; with async off (register staging): base-a100-verify.txt then base-a100-verify2.txt filled _c 0.986/0.970/0.966 (verify2: the on-warp contract kernel is byte-identical), _exact 0.988/0.995/0.995 then 1.016/1.006/1.005: no stable sign",
    },
    RungOff {
        profile: "a100",
        rung: "wide",
        fill: Fill::Under,
        face: Face::Contract,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _tf n=1024 0.986 vs 0.962 (ms in the files): no size ON-faster in both sessions",
    },
    RungOff {
        profile: "a100",
        rung: "wide-fma",
        fill: Fill::Filled,
        face: Face::Any,
        cite: "base-a100.txt / base-a100-confirm.txt OFF/ON cuBLAS-normalized: _c n=2048/4096/8192 0.949/0.937/0.904 vs 0.960/0.947/0.919; _exact n=2048/4096/8192 0.807/0.906/0.909 vs 0.820/0.908/0.909 (ms in the files): no size ON-faster in both sessions; with async off (register staging): base-a100-verify.txt then base-a100-verify2.txt _c 1.019/1.041/1.053 then 0.984/0.991/0.994: the sign changes between sessions; S54 arch-select (arch-select-a100.txt / base-a100-verify3.txt): alone +1.0-1.7% (_c n=4096/8192 1.010/1.013 vs 1.010/1.017), but with idx32 ON (which ships) it costs: on-fma-regs (= idx32 + wide-fma on this face) n=4096 8.195 / 7.942 ms vs idx32 alone 8.034 / 7.800 ms",
    },
    RungOff {
        profile: "a100",
        rung: "ldmatrix",
        fill: Fill::Under,
        face: Face::Tf32,
        cite: "arch-select-a100.txt / base-a100-verify3.txt OFF/ON cuBLAS-normalized _tf n=1024 0.967 / 0.951 (ms 0.0840 / \
               0.0840 ON vs 0.0881 / 0.0799 OFF, cuBLAS 0.0471 / 0.0481 vs 0.0512 / 0.0481); kept filled (n=2048/4096/8192 \
               1.049/1.020/1.024 vs 1.017/1.024/1.022)",
    },
    // S54 `rowfold-layout` (the reader-chosen store order): the strided fold the
    // row-major store leaves costs 0.01–0.1 ms at 4096 (L1 serves each sector to
    // the next 7 steps), and the column-major store costs the FMA tile more. Read
    // on the device clock (traced kernel sums, site + fold, column-major vs
    // row-major): the rung changes only kernels, and the probes' own clock is
    // host-noise-bound at that size (probe_rungs.py).
    RungOff {
        profile: "rtx-3090",
        rung: "rowfold-layout",
        fill: Fill::Any,
        face: Face::Any,
        cite: "rowfold-rtx-3090.txt / -confirm.txt pt_rowsum traced kernel ms, column-major vs row-major: _exact \
               4096 9.081 vs 8.994, 9.856 vs 9.790; _c 4096 5.637 vs 5.624, 5.936 vs 5.929; 2048 within 0.4%: no \
               size faster in both sessions",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "rowfold-layout",
        fill: Fill::Any,
        face: Face::Any,
        cite: "rowfold-rtx-5090.txt / -confirm.txt pt_rowsum traced kernel ms, column-major vs row-major: _exact \
               4096 3.478 vs 3.432, 3.378 vs 3.330; _c 4096 2.216 vs 2.153, 2.161 vs 2.088 (2048 2.5-2.8% faster, \
               0.506 vs 0.519, 0.490 vs 0.504 ms: the 4096 loss in both sessions decides the class)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "rowfold-layout",
        fill: Fill::Any,
        face: Face::Any,
        cite: "rowfold-h100-pcie.txt / -confirm.txt pt_rowsum traced kernel ms, column-major vs row-major: _exact \
               4096 7.851 vs 6.451, 9.101 vs 7.615; _c 4096 4.033 vs 3.760, 4.735 vs 4.441; 2048 the same sign; \
               -v2.txt / -v2-confirm.txt too (_exact 4096 8.97 vs 7.52 ms): the HBM part pays 7-22% for the \
               column-major tile store",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "rowfold-layout",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "rowfold-rtx-4090.txt / -confirm.txt pt_rowsum traced kernel ms, column-major vs row-major: _exact \
               4096 4.189 vs 4.018, 4.267 vs 4.160 (the exact tile pays 4-6% for the store); kept on the contract \
               face: _c 4096 2.507 vs 2.564, 2.605 vs 2.640 and 2048 0.315 vs 0.435, 0.327 vs 0.361 ms",
    },
    RungOff {
        profile: "a100",
        rung: "rowfold-layout",
        fill: Fill::Any,
        face: Face::Contract,
        cite: "rowfold-a100.txt / -confirm.txt pt_rowsum traced kernel ms, column-major vs row-major: _c 4096 \
               8.326 vs 8.162, 10.694 vs 10.488; kept on the exact face: _exact 2048 2.392 vs 2.428, 3.080 vs \
               3.120 and 4096 15.479 vs 15.568, 19.911 vs 20.027 ms",
    },
    RungOff {
        profile: "a100",
        rung: "retain-delta",
        fill: Fill::Any,
        face: Face::Any,
        cite: "base-a100.txt decode_1024_256 wall, rung on (resident yields) 2.229/2.233 s vs off (resident takes the \
               loop) 1.311/1.338 s exact/contract, 3 procs; base-a100-confirm.txt 2.219/2.081 vs 1.467/1.494 s (ms: the \
               files' trace totals)",
    },
    // S54 `l2-raster` (benches/results-s54-gpu/l2-raster-tables.md). `raster`'s derivation groups
    // only where one block's panels leave the L2 (the 3090 at 8192); there it measured no stable
    // sign. `l2-gate` is the derivation's gate, so its row means the opposite of the others: it
    // is OFF only where the ungated grouping measured faster in every session (the 5090 at 8192),
    // never for "no effect". `l2-window` ships ON only where a probe's traced kernel time measured
    // > 1% faster in every session.
    RungOff {
        profile: "rtx-3090",
        rung: "raster",
        fill: Fill::Any,
        face: Face::Any,
        cite: "l2-raster-rtx-3090-run2.txt / -run3.txt mm_8192 OFF/ON cuBLAS-normalized exact 0.986 / 1.029, \
               contract 0.992 / 1.017, TF32 0.991 / 0.956 (ON ms 92.743/55.147/34.632 vs OFF 91.632/54.659/34.341; \
               116.792/93.411/52.706 vs 123.827/90.284/53.472): no face faster in both sessions",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "l2-gate",
        fill: Fill::Any,
        face: Face::Contract,
        cite: "l2-raster-rtx-5090-run2.txt / -run3.txt / -final.txt mm_8192 TF32 grouped 1.031 / 1.054 / 1.014 x \
               faster (cuBLAS-normalized; ms 13.156 vs 12.757, 13.347 vs 12.671, 11.042 vs 10.877); FMA 1.020 / 1.043 / \
               1.003; mm_4096 0.973-1.007. Exact keeps the gate: 1.018 / 1.052 / 0.999 (ms 34.913 vs 34.347, \
               35.970 vs 34.317, 29.751 vs 29.530), not faster in every session",
    },
    RungOff {
        profile: "rtx-3090",
        rung: "l2-window",
        fill: Fill::Any,
        face: Face::Any,
        cite: "l2-raster-rtx-3090*.txt: no probe has an operand within the 6 MB L2 (pt_* at 2048 are 16 MB), so no \
               probe row has a window arm (0 ms measured): unmeasured, not shipped",
    },
    RungOff {
        profile: "a100",
        rung: "l2-window",
        fill: Fill::Any,
        face: Face::Any,
        cite: "l2-raster-a100{,-confirm,-run2,-run3,-run5}.txt pt_{mlp,mlp2,diamond}_2048 traced kernel ms off/on: \
               25 of 30 cells <= 1.000, none > 1.01 in every session (mlp2_2048_c 0.994/0.998/0.995/0.965/0.994)",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "l2-window",
        fill: Fill::Any,
        face: Face::Any,
        cite: "l2-raster-rtx-4090-run2.txt / -run4.txt traced kernel ms off/on: diamond_2048_c 0.933 / 0.939, \
               mlp2_2048_c 0.938 / 0.949, diamond_4096_c 0.989 / 0.987 (the window costs on contract); exact \
               0.985-1.003",
    },
    RungOff {
        profile: "rtx-5090",
        rung: "l2-window",
        fill: Fill::Any,
        face: Face::Any,
        cite: "l2-raster-rtx-5090{,-run2,-run3,-run4,-run5}.txt traced kernel ms off/on: 60 cells 0.976-1.005, \
               none > 1.01 in any session (diamond_2048_c 0.988/0.989/0.976/0.991/0.986)",
    },
    RungOff {
        profile: "h100-pcie",
        rung: "l2-window",
        fill: Fill::Any,
        face: Face::Exact,
        cite: "l2-raster-h100-pcie.txt / -run2.txt traced kernel ms off/on, exact: diamond_2048 1.004 / 1.007, \
               mlp2_2048 1.002 / 1.000, mlp_2048 1.000 / 1.000 — none > 1.01; contract ships ON (mlp2_2048_c \
               1.012 / 1.016, disjoint)",
    },
    // S54 R5 (`retain-resident`, `delta-h2d`, `lane-stride`): host placement rungs, decided on
    // decode_1024_256 with both faces pooled, by the loop time — the traced process wall minus its
    // own `load_ms` (context + module: 170–790 ms and, on the H100s, ±200 ms between processes,
    // which no arm changes). ON where OFF/ON > 1.01 in both sessions (two boxes); the process wall
    // is in the files beside it. `delta-h2d` and `lane-stride` measured ON on all five parts.
    RungOff {
        profile: "rtx-5090",
        rung: "retain-resident",
        fill: Fill::Any,
        face: Face::Any,
        cite: "retain-resident-rtx-5090.txt / retain-resident-rtx-5090-confirm.txt decode_1024_256 OFF/ON loop \
               1.008 / 1.016 (ON 343–360 / 328–366 ms vs OFF 345–363 / 336–351 ms, overlapping), process wall \
               1.022 / 1.005: not > 1% in both sessions. The whole K/V site is launch-bound there: traced kernel \
               76 vs 82 ms over 1024 steps",
    },
    RungOff {
        profile: "rtx-4090",
        rung: "retain-resident",
        fill: Fill::Any,
        face: Face::Any,
        cite: "retain-resident-rtx-4090.txt / retain-resident-rtx-4090-confirm.txt decode_1024_256 OFF/ON loop \
               1.025 / 0.981 (ON 337–367 / 357–411 ms vs OFF 341–365 / 341–419 ms, overlapping), process wall \
               1.011 / 0.988: the sign changes between sessions. The whole K/V site costs what the row does \
               there (traced kernel 103 vs 105 and 108 vs 110 ms over 1024 steps, exact)",
    },
];

/// The parts [`GPU_RUNGS_OFF`] was measured on. An unmeasured profile takes
/// the rows of the measured part with its architecture (`cuda-ada` ← the
/// 4090's, `h100` ← the H100 PCIe's); with none, every rung is ON — the S52
/// baseline the arms are measured against — until its own session adds rows.
/// Ampere since S54: the RTX 3090 and the A100 (`a100` is the SXM4 part; its
/// two sessions ran on an A100 PCIe and an A100 SXM4).
pub const GPU_MEASURED: [&TargetProfile; 5] = [&RTX_4090, &H100_PCIE, &RTX_5090, &RTX_3090, &A100];

/// The tensor-core tile geometry for one tile site on the TF32 face (S52):
/// `mma.sync.m16n8k8`, operands staged by `cp.async` through `stages`
/// shared-memory buffers. Derived by [`TargetProfile::gpu_mma_tile`].
///
/// A warp owns a `wm × wn` output tile of `mt × nt` m16n8 tiles; the block is
/// `warps_y × warps_x` warps. Panels are row-major in shared memory with their
/// rows padded (`pad_a`, `pad_b` words) so a fragment load hits 32 banks once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMmaTile {
    pub mt: u64,
    pub nt: u64,
    pub warps_x: u64,
    pub warps_y: u64,
    pub bk: u64,
    pub stages: u64,
    pub pad_a: u64,
    pub pad_b: u64,
    pub blocks_per_sm: u64,
}

/// Registers a tensor-core thread spends on addresses and loop state, beyond
/// accumulators and fragments. Policy, read off ptxas reports (S52): the
/// 64×32 kernel used 120 registers for 64 accumulators and 24 fragments.
pub const MMA_ADDR_REGS: u64 = 32;

/// `mma.sync.m16n8k8`'s shape — an ISA rule.
pub const MMA_M: u64 = 16;
pub const MMA_N: u64 = 8;
pub const MMA_K: u64 = 8;

impl GpuMmaTile {
    pub fn wm(&self) -> u64 {
        self.mt * MMA_M
    }
    pub fn wn(&self) -> u64 {
        self.nt * MMA_N
    }
    pub fn bm(&self) -> u64 {
        self.warps_y * self.wm()
    }
    pub fn bn(&self) -> u64 {
        self.warps_x * self.wn()
    }
    pub fn threads(&self, warp: u64) -> u64 {
        self.warps_x * self.warps_y * warp
    }
    pub fn lda(&self) -> u64 {
        self.bk + self.pad_a
    }
    pub fn ldb(&self) -> u64 {
        self.bn() + self.pad_b
    }
    /// 32-bit words one stage holds: the `bm × bk` A panel and the `bk × bn`
    /// B panel, each at its padded row length.
    pub fn stage_words(&self) -> u64 {
        self.bm() * self.lda() + self.bk * self.ldb()
    }
    pub fn smem_bytes(&self) -> u64 {
        self.stages * self.stage_words() * 4
    }
}

/// `wgmma.mma_async.m64nNk8.f32.tf32.tf32`'s fixed sides — ISA rules. One
/// instruction is issued by a warpgroup of [`WARPGROUP_WARPS`] warps.
pub const WGMMA_M: u64 = 64;
pub const WGMMA_K: u64 = 8;
/// `N` runs over the multiples of 8 up to 256.
pub const WGMMA_N_STEP: u64 = 8;
pub const WGMMA_N_MAX: u64 = 256;
pub const WARPGROUP_WARPS: u64 = 4;
/// Registers a `wgmma` thread spends beyond accumulators and fragments —
/// [`MMA_ADDR_REGS`]'s policy for this kernel, read off ptxas 12.8 (S53): the
/// m64n256 kernel used 213 = 128 accumulators + 16 fragments + 69 (its
/// `cp.async` addresses), rounded up to the 8-register allocation unit.
pub const WGMMA_ADDR_REGS: u64 = 72;
/// Stages the `wgmma` panel pipeline needs at least: one landing, one read by
/// the group being issued, one still read by the group in flight.
pub const WGMMA_MIN_STAGES: u64 = 3;
/// The `a` panel's shared-memory row in the 128-byte swizzle mode
/// (`Swizzle<3,4,3>`, descriptor mode 1): one wavefront, eight 16-byte
/// chunks. A warp then copies whole 128-byte global lines (8 lanes per row)
/// and the XOR of chunk and row spreads those 8 lanes over all 32 banks; the
/// no-swizzle layout forced half-line copies to stay conflict-free, and that
/// copy pipeline bound the first S53 build (box: 1.20 of 1.52 ms at 4096).
pub const WGMMA_SWIZZLE_BYTES: u64 = 128;
/// Rows of one swizzle atom; an atom (8 × 128 B) must start 1024-aligned.
pub const WGMMA_ATOM_ROWS: u64 = 8;

/// The Hopper TF32 tile (S53 `wgmma-hopper`): the site's product transposed,
/// `cᵀ = bᵀ·aᵀ`, because `wgmma` takes TF32 shared-memory operands only
/// K-major — `a` (`ck = 1`) already is, `b` (lanes contiguous) is not. So `bᵀ`
/// is the register operand (any layout loads), `aᵀ` the descriptor operand in
/// `a`'s own layout, and one instruction covers `WGMMA_M` columns × `n` rows
/// of `c`. A block is `groups` warpgroups side by side along `c`'s columns,
/// all reading the same staged `a` panel. Derived by
/// [`TargetProfile::gpu_wgmma_tile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuWgmmaTile {
    /// The instruction's `N`: rows of `c` per block.
    pub n: u64,
    /// Warpgroups per block.
    pub groups: u64,
    pub bk: u64,
    pub stages: u64,
    /// Words of padding on each `b` panel row (the register-operand loads).
    pub pad_b: u64,
    pub blocks_per_sm: u64,
}

impl GpuWgmmaTile {
    pub fn bm(&self) -> u64 {
        self.n
    }
    pub fn bn(&self) -> u64 {
        self.groups * WGMMA_M
    }
    pub fn threads(&self, warp: u64) -> u64 {
        self.groups * WARPGROUP_WARPS * warp
    }
    pub fn ldb(&self) -> u64 {
        self.bn() + self.pad_b
    }
    /// Words of the `a` panel in one stage: `bk / 32` atom columns, each
    /// `bm` swizzled 128-byte rows.
    pub fn a_words(&self) -> u64 {
        self.bm() * self.bk
    }
    /// One stage, rounded up to whole 1024-byte atoms so every stage's `a`
    /// panel starts where the swizzle pattern does (base offset 0).
    pub fn stage_words(&self) -> u64 {
        let atom_words = WGMMA_ATOM_ROWS * WGMMA_SWIZZLE_BYTES / 4;
        (self.a_words() + self.bk * self.ldb()).next_multiple_of(atom_words)
    }
    pub fn smem_bytes(&self) -> u64 {
        self.stages * self.stage_words() * 4
    }
    /// Stride-dimension byte offset: between 8-row groups, one atom.
    pub fn sbo(&self) -> u64 {
        WGMMA_ATOM_ROWS * WGMMA_SWIZZLE_BYTES
    }
    /// The matrix descriptor without its start address (PTX ISA "Matrix
    /// Descriptor Format"): LBO field 1 (swizzled K-major layouts do not use
    /// it, "assumed 1"), SBO>>4 at bit 32, swizzle mode 1 (128-byte) at bit 62.
    pub fn desc_fields(&self) -> u64 {
        (1 << 16) | ((self.sbo() >> 4) << 32) | (1 << 62)
    }
}

/// The smallest row padding, in 32-bit words and a multiple of four (so every
/// row stays 16-byte aligned for `cp.async`), under which one warp's fragment
/// load touches each of the 32 banks once. `lane → (row, col)` is the
/// fragment's access pattern; this is the bank-geometry deduction the CPU's
/// move-panel rung makes for cache sets, applied to shared memory.
pub(crate) fn conflict_free_pad(ld: u64, pattern: impl Fn(u64) -> (u64, u64)) -> Option<u64> {
    (0..32).step_by(4).find(|pad| {
        let mut banks = [false; 32];
        (0..32).all(|lane| {
            let (row, col) = pattern(lane);
            let bank = ((row * (ld + pad) + col) % 32) as usize;
            !std::mem::replace(&mut banks[bank], true)
        })
    })
}

impl GpuTile {
    pub fn bm(&self) -> u64 {
        self.ty * self.tm
    }
    pub fn bn(&self) -> u64 {
        self.tx * self.tn
    }
    pub fn threads(&self) -> u64 {
        self.tx * self.ty
    }
    /// Words one transposed-`a` panel row (one `k`) spans in shared memory.
    pub fn sa(&self) -> u64 {
        self.bm() + self.pad_a
    }
    /// Elements one stage holds: the transposed A panel and the B panel.
    pub fn stage_elems(&self) -> u64 {
        self.bk * (self.sa() + self.bn())
    }
    /// Dynamic shared memory one block uses, every stage.
    pub fn smem_bytes(&self, elem_bytes: u64) -> u64 {
        self.stages.max(1) * self.stage_elems() * elem_bytes
    }
}

/// The ARM SME facts a profile carries, when the part has a streaming matrix
/// unit — and **only the two that cannot be derived from anything else.**
///
/// Both are **read off the part** by [`native`] (`hw.optional.arm.sme_max_svl_b`
/// and `hw.perflevel0.l1dcachesize`), never assumed: SVL is
/// implementation-defined and a cache size is a property of the silicon. Every
/// other SME quantity the realization needs is *derived* from these two —
/// the panel side (`svl/w`), the tile count (`w`, an ISA rule), the arrangement,
/// the packed A stride, and the k-panel depth — so a part with a different SVL
/// or a different L1D emits a different kernel from the same record, with no new
/// field and no new code. `f32_tiles` used to sit here and was deleted for
/// exactly that reason: it was derivable, so recording it could only ever make
/// it wrong (see [`TargetProfile::sme_block`]).
///
/// A hand-written value is therefore the *cross-compilation* case — describing a
/// machine you are not sitting on — and
/// `native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts`
/// asserts the two agree on the documented part (ADR-0038 D4).
///
/// `Machine` stays `Cpu`: SME is a **unit inside a CPU core**, not a machine
/// class. One thread owns the whole tile, there is no cooperation and no
/// barrier — exactly the `Cpu` reading. What differs is only which realization
/// the leaf takes, which is what `Option` here expresses (plan-s41 §2.2 rule 2:
/// capability selects realization).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sme {
    /// Streaming vector length in bytes — ZA is `svl_bytes × svl_bytes`, so one
    /// f32 tile is `svl_bytes/4` square. **64 on the M4 Pro** (512 bits), which
    /// is what makes `<vscale x 4 x float>` sixteen lanes there.
    pub svl_bytes: u64,
    /// **Per-core L1 data cache in bytes** — the budget the SME k-panel is sized
    /// against ([`TargetProfile::sme_kc`]).
    ///
    /// It lives **inside `Sme`**, not beside `l2_bytes`, and that placement is
    /// the point: a profile cannot declare a matrix unit without declaring the
    /// working-set budget that unit's panel is cut to. FRAMEWORK §4.5 law 3
    /// (placement totality) as a type, so the omission is a compile error rather
    /// than a silent fallback to the NEON core's number.
    ///
    /// Why a *second* cache fact rather than reusing `l2_bytes`: `tile_kc` sizes
    /// the NEON k-panel against half of a 16 MB **shared** L2 and returns 4096
    /// on this part, which closes its own gate for every K this project runs
    /// (`apple_m_raises_the_kc_threshold_above_every_k_we_run`). Measured, the
    /// SME optimum is a **128 KB two-panel window** — L1D, not L2 — and the
    /// unblocked case loses **1.448×** at K=4096
    /// (`docs/performance/s42-sme-roofline.md` §5, `benches/sme/kc.c`). One
    /// deduced depth serving two placements with different budgets is the defect;
    /// two `TrnLoc`s over one question is the fix.
    pub l1d_bytes: u64,
    /// **Policy ratio, honestly search space — NOT a machine fact.** How many
    /// L1D's worth of operand window one k block may occupy. Same status as
    /// [`TargetProfile::acc_vecs_per_row`] and `nc_tiles`, and ADR-0034 is the ADR
    /// that would search it rather than record it.
    ///
    /// It exists because the obvious derivation is **wrong, measured**. Sizing the
    /// window to fit L1D exactly (ratio 1 ⇒ `kc` 512 here) lands on the losing
    /// side of a sharp curve. Two depth sweeps in the real emitter, N=4096 and
    /// N=2048, f32, 1 thread, alternating, values identical at every depth:
    ///
    /// | working set | N=4096 | N=2048 |
    /// | ---: | ---: | ---: |
    /// | 32 KB | — | 0.220× |
    /// | 64 KB | 0.501× | 0.387× |
    /// | 128 KB ← ratio 1 | 0.785× | 0.639× |
    /// | **256 KB ← ratio 2** | **1.064×** | 0.986× |
    /// | 512 KB | 1.027× | **1.000×** (unblocked) |
    /// | 1024 KB | 1.000× (unblocked) | — |
    ///
    /// **256 KB is the optimum at both sizes, and it is not any machine fact on
    /// this part** — L1D is 128 KB, L1I 192 KB, the per-core L2 slice ~3.2 MB. So
    /// it is recorded as a swept ratio rather than dressed up as a derivation.
    /// Writing `2 * l1d_bytes` into [`TargetProfile::sme_kc`] would be a fitted
    /// constant wearing a derivation's clothes, which is the exact defect
    /// `plan-s31-deduced-blocking.md` exists to remove.
    ///
    /// Note also what the curve says about getting it wrong: every depth *below*
    /// the optimum is catastrophic (64 KB costs 2.0–2.6×), so this constant is not
    /// a mild tuning knob. The first version of `sme_kc` used ratio 1 and that one
    /// wrong constant is what made KC blocking look like a 1.27× loss for most of
    /// S42 — see `func/sme.rs`.
    pub panel_l1d_ratio: u64,
}

/// What a parallel dispatch costs on this part, and how fast ONE lane runs —
/// enough for the backend to decide whether a task is worth dispatching at all
/// (plan-s50 P3).
///
/// **Only a profile whose numbers were measured on the part carries this.**
/// `None` means "dispatch everything", which is exactly the behaviour every
/// profile had before this existed, so an unmeasured part cannot be made worse
/// by a number nobody took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchCost {
    /// Compute-bound rate of one lane, in flops per nanosecond (== GFLOP/s).
    pub flops_per_ns: u64,
    /// Memory-bound rate of one lane, in bytes per nanosecond (== GB/s).
    pub bytes_per_ns: u64,
    /// What handing one task to the pool and getting it back costs, in
    /// nanoseconds. A task whose estimated serial time is under this is worth
    /// more run inline than dispatched: the dispatch buys one lane's share of
    /// the work and pays a whole wave for it.
    pub dispatch_ns: u64,
}

/// One target's machine facts, plus the two policy ratios that are honestly
/// search space rather than facts (see the field docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    pub name: &'static str,
    /// The machine class — the capability that selects a realization (S41).
    pub machine: Machine,
    /// One vector register in bytes: NEON 16, AVX2 32, AVX-512 64.
    pub vec_bytes: u64,
    /// Architectural vector register count: NEON 32, AVX2 16, AVX-512 32.
    /// **Not probed** — it is a fact of the ISA, so it arrives with the target
    /// features rather than from a runtime query.
    pub vec_regs: u64,
    /// Policy ratio: how many vector registers one accumulator row occupies,
    /// i.e. how wide a j-tile is measured in registers. This is "how much of
    /// the register file to spend on accumulators" — search space, not a
    /// machine fact (ADR-0034 is the ADR that would search it).
    pub acc_vecs_per_row: u64,
    /// Policy ratio: j-tiles per KC-rung j-block. Same status as
    /// `acc_vecs_per_row`.
    pub nc_tiles: u64,
    /// L2 in bytes, **as reported for one L2 instance** — the budget the KC
    /// rung's k-panel is sized against. Divide by [`Self::l2_cores`] for the
    /// per-core share; see that field for why the distinction is load-bearing.
    pub l2_bytes: u64,
    /// **How many physical cores share one `l2_bytes`.** This field exists
    /// because the old doc comment on `l2_bytes` said "per-core" while
    /// [`APPLE_M`] set it to the 16 MB **shared** P-cluster L2 — a two-word
    /// ambiguity that no consumer had noticed because `tile_kc` only ever wanted
    /// an order of magnitude.
    ///
    /// [`TargetProfile::move_block`]'s cost term divides by it, and there the
    /// difference is the whole answer: 16 MB says a 4 MB array is resident and
    /// the rung should decline, 16 MB / **5** = 3.36 MB says it is not and the
    /// rung should fire — and the M4's measured 1.578× says fire. Detected
    /// (`hw.perflevel0.cpusperl2`, `shared_cpu_list`), never assumed.
    ///
    /// Budgeting **per core** rather than per chip is the right conservatism for
    /// a compiler that emits one binary for both thread counts: S44 §5 measured
    /// that this walk's pressure does not dilute with thread count, so at full
    /// occupancy every core really does get its share and no more.
    pub l2_cores: u64,
    /// Stack ceiling policy: an entry-block block at least this large is placed
    /// in the `mapal_rt_alloc` arena instead of the stack.
    pub heap_min_bytes: u64,
    /// Measured dispatch economics, when someone has measured them. `None`
    /// keeps the pre-plan-s50 behaviour: dispatch every task.
    pub dispatch: Option<DispatchCost>,
    /// The streaming matrix unit, when the part has one. `None` for every
    /// profile that existed before S41 — which is exactly what keeps their
    /// emission byte-identical: the SME realization cannot be selected without
    /// a fact only this field carries.
    pub sme: Option<Sme>,
    /// The L1D geometry the move-panel decision needs (S45). `None` for
    /// [`GENERIC`] and every profile that predates S45 — the same seam
    /// discipline as [`Self::sme`] and [`Self::gpu`], and it is what keeps the
    /// **default** profile's emission byte-identical: the rung cannot fire
    /// without a fact only this field carries, so `--target=generic` emits today's
    /// text for every one of the 171 swept cells.
    ///
    /// ponytail: [`Sme::l1d_bytes`] is the same number for parts that carry
    /// both, pinned equal by `l1d_and_sme_agree_on_the_l1d`. Merging them routes
    /// `sme_kc` through a new `Option` for a field rename, and a working SME leg
    /// is not worth that; the test makes the duplication unable to drift.
    pub l1d: Option<L1d>,
}

/// The element width a tile site accumulates at. Tile sites are numeric-gated
/// by recognition (`mapal_ir::algo` requires `is_numeric` for the element, the
/// accumulator, the seed and the map target), so the non-numeric arm is
/// unreachable; it returns the f32 width rather than panicking because a wrong
/// tile width is a slow kernel, never a wrong answer.
pub(crate) fn elem_bytes(elem: &Ty) -> u64 {
    match elem {
        Ty::Int { bits, .. } | Ty::Float { bits } => u64::from(*bits).div_ceil(8).max(1),
        _ => 4,
    }
}

impl TargetProfile {
    /// Scalar lanes in one vector register at this element width.
    pub fn lanes(&self, elem: &Ty) -> u64 {
        (self.vec_bytes / elem_bytes(elem)).max(1)
    }

    /// The register micro-kernel's j-tile width in lanes (was `tile_j_for`).
    /// `generic`: f32 `16 / 4 × 4 = 16`, f64 `16 / 8 × 4 = 8` — today's
    /// literals, reproduced.
    pub fn tile_j(&self, elem: &Ty) -> u64 {
        self.lanes(elem) * self.acc_vecs_per_row
    }

    /// Rows per register block (was `TILE_I`, on the matmul rung only — the FIR
    /// window rung reads this same budget via `crate::func::window_subrows`, over a different
    /// quantity over a memory accumulator).
    ///
    /// The `2 ×` is the headroom policy: spend at most HALF the vector file on
    /// accumulators, leaving the rest for the shared b tile, the a splat and
    /// the products. `generic`: `32 / (2 × 4) = 4`. It reproduces S26's swept
    /// result *including its failure* — TI=8 would need 32 accumulator
    /// registers, and that sweep recorded "8 spills: 128 accumulators ≫ 32 NEON
    /// regs".
    pub fn tile_i(&self) -> u64 {
        (self.vec_regs / (2 * self.acc_vecs_per_row)).max(1)
    }

    /// The KC rung's j-block width in lanes (was `tile_nc_for`).
    pub fn nc(&self, elem: &Ty) -> u64 {
        self.tile_j(elem) * self.nc_tiles
    }

    /// The KC rung's k-panel depth (was `TILE_KC`): half of L2 spent on one
    /// (kc × nc) packed-b window. `generic` yields 128 at BOTH widths —
    /// `512 KB/2 ÷ (512 × 4)` and `512 KB/2 ÷ (256 × 8)` — today's single
    /// literal, reproduced. On a 16 MB L2 it yields `kc ≥ K`, so the gate
    /// `site.k > tile_kc` closes and **the KC nest disables itself by
    /// derivation** rather than by a default-off flag (S29/S30's measured
    /// verdict, deduced).
    pub fn tile_kc(&self, elem: &Ty) -> u64 {
        ((self.l2_bytes / 2) / (self.nc(elem) * elem_bytes(elem))).max(1)
    }

    /// The L2 budget **one core** actually gets. See [`Self::l2_cores`].
    pub fn l2_per_core(&self) -> u64 {
        self.l2_bytes / self.l2_cores.max(1)
    }

    /// **The S45 move-panel decision, in one function: fire or not, and with
    /// what block.** `None` declines — the emitter then emits today's flat loop
    /// character-identically.
    ///
    /// This is the join ADR-0032 puts here and nowhere else: [`MoveSite`] is
    /// pure program geometry, [`L1d`] is pure machine fact, and neither crate
    /// may hold both. It replaces `--move-panel=W:B`, where a human held both in
    /// their head and typed the answer.
    ///
    /// ```text
    /// S            = cr · sizeof(elem)                       -- the walk's byte stride
    /// sets_touched = line | S ? sets / gcd((S/line) mod sets, sets) : sets
    /// slots        = sets_touched · ways                     -- lines the walk can reach
    /// lines_live   = min(width, width · S / line)            -- lines it needs at once
    /// FIRE iff  lines_live > slots  AND  read_bytes > l2_per_core
    /// ```
    ///
    /// **Two terms, because one is provably not enough.** `lines_live > slots`
    /// is S44's `pressure > 1` written as what it means — the sweep needs more
    /// lines than it can reach — with no threshold to fit. But pressure predicts
    /// how OFTEN the L1 is defeated and says nothing about what a defeat COSTS:
    /// the i9 at side 512 scores 21.3 and **loses 0.901×** (replicated 0.907×).
    /// A defeat costs real money only when the array being re-swept does not fit
    /// the private level below L1; otherwise the miss is an L2 hit the
    /// out-of-order engine hides. The **read** array is the quantity, not the
    /// whole working set — the conflict is on `a`'s lines, the writes stream —
    /// which also keeps the discriminating case off a knife edge (1 MB against
    /// 2 MB, rather than 2 MB against 2 MB).
    ///
    /// Both terms were tested predictively rather than fitted. M4 side 512 has
    /// pressure 8 and a 1 MB read array: the cost term declines it, S44's
    /// standalone probe claimed **2.09×** there, and the emitted pipeline
    /// measured **every arm overlapping OFF, min-to-min 1.02×** — the term's
    /// prediction, on the machine where it was most at risk
    /// (`benches/results-s45/deduced-move-panel.md` §3).
    ///
    /// | case | pressure | read vs L2/core | verdict | measured |
    /// | --- | ---: | --- | --- | --- |
    /// | M4 1024 | 32 | 4 MB > 3.36 MB | **fire, B=16** | 1.578× 1t / 1.993× threaded |
    /// | M4 2048 | 128 | 16 MB > 3.36 MB | **fire, B=8** | 3.19× (probe) |
    /// | M4 128 | 0.5 | — | decline (pressure) | 1.000× |
    /// | M4 512 | 8 | 1 MB < 3.36 MB | decline (cost) | overlapping, 1.02× min-to-min |
    /// | i9 1024 | 85.3 | 4 MB > 2 MB | **fire, B=8** | 2.646× 1t, disjoint |
    /// | i9 2048 | 170.7 | 16 MB > 2 MB | **fire, B=8** | 3.021× 1t, disjoint |
    /// | i9 512 | 21.3 | 1 MB < 2 MB | decline (cost) | **0.901× LOSS** |
    ///
    /// **The block: the geometric mean of two opposing costs, both measured.**
    ///
    /// ```text
    /// floor   = line / sizeof(elem)   -- a block row shorter than a line refetches it
    /// ceiling = slots / 2             -- the block's read lines share the reachable
    ///                                    sets with its write stream
    /// B       = largest divisor of gcd(width, rows) <= sqrt(floor * ceiling)
    /// ```
    ///
    /// Both costs are 1 inside `[floor, ceiling]`; the window is normally EMPTY
    /// (the bounds pull apart), and the product of two opposing multipliers is
    /// minimised at their geometric mean. The divisibility clause is not
    /// rounding — the permutation needs the panel to tile the geometry both ways
    /// or it would need a remainder arm.
    ///
    /// Each bound is a measurement, not an argument. **Floor:** B=8 on the i9
    /// covers 8 of the 16 f32 in a line and measures **15% slower than B=16** at
    /// side 1024, 24% at 2048. **Ceiling:** B=`slots` on the M4 measures **29%
    /// (S44) and 34% (S45) slower threaded** than B=16 at side 1024, and 56% at
    /// side 2048.
    ///
    /// It reproduces the M4's swept optimum at **both** sides — 16 at side 1024
    /// (floor 32, ceiling 16, mean 22) and 16 at side 2048 (floor 32, ceiling 8,
    /// mean 16) — which the previous ceiling-only rule did not: it returned 8 at
    /// side 2048 and measured **15.7% off**.
    ///
    /// **What it does NOT reproduce, and why no version of it could.** The i9's
    /// measured optimum is **128**, and this returns 8. That gap was priced with
    /// counters rather than argued (`benches/results-s45`, side 1024, P-core):
    ///
    /// | arm | ms | cycles | L1-dcache-load-misses | dTLB misses | LLC misses |
    /// | --- | ---: | ---: | ---: | ---: | ---: |
    /// | off | 2.526 | 16.2 M | **1 053 802** | 289 | 683 |
    /// | B=8 | 1.072 | 9.75 M | 206 414 | 291 | 259 |
    /// | B=16 | 0.940 | 9.14 M | 499 456 | 282 | 316 |
    /// | B=128 | 0.929 | 8.83 M | **1 053 618** | 287 | 253 |
    ///
    /// **`off` and B=128 miss L1 the same number of times — 1 053 802 against
    /// 1 053 618 — and B=128 is 2.7x faster.** B=8 misses **five times less**
    /// and is *slower* than both. TLB misses are flat at ~290 across every arm,
    /// and LLC misses are ~0 (the 4 MB array fits a 36 MB L3). Instruction
    /// counts are identical across the blocked arms (33.03 M), so the ordering
    /// is pure memory behaviour and IPC rises monotonically with B
    /// (3.39 → 3.61 → 3.74).
    ///
    /// So on that machine **L1 residency is not the binding resource at the
    /// optimum**, and neither is the TLB nor the LLC.
    ///
    /// **S47 swept every legal divisor at four sides on both machines and priced
    /// what IS binding. The answer is not memory-level parallelism**, which was
    /// the standing hypothesis here and is now measured false: at the i9 optimum
    /// `l1d_pend_miss.fb_full` is **3.0% of cycles** and MLP
    /// (`pending / pending_cycles`) is **2.23 — the LOWEST of any arm**. The slow
    /// arms have MORE misses in flight (B=8: 4.00; B=32: 3.66), and the fill
    /// buffers only bind on arms nobody would ship (`off`: 47% fb_full, MLP 5.95).
    /// The binding quantity is **L2 residency**: order the side-2048 arms by
    /// `mem_load_retired.l2_miss` and the ms order falls out with no inversion
    /// across eight arms. Behind it is the L2 prefetcher — a block gives each
    /// column stream a run of `B·w/line` consecutive lines, and below two lines
    /// the streamer has nothing to lock on to.
    ///
    /// **And that is a prefetcher property, which is why no rule here can be
    /// right on both parts.** S47's sweep: the M4's optimum is **16 at every side
    /// that wins**, the i9's is **64–128** — the i9 wants a block 4–8x LARGER
    /// while every fact [`L1d`] and [`TargetProfile`] can read is LARGER on the
    /// M4 (`bytes` 2.67x, `line_bytes` 2x, `sets` 2x, `l2_per_core` 1.68x). Only
    /// `ways` is bigger on the i9, by 1.5x, against an 8x difference in the
    /// answer. **No monotone function of the readable facts orders these two
    /// parts the way the measurement does**, and the quantity that does — how far
    /// a prefetcher reaches on a strided walk — is in no sysctl, no sysfs node
    /// and no ISA rule. The witness is the unblocked arms: the same instruction
    /// stream over the same geometry costs **0.796 ns/element on the M4 and 2.325
    /// on the i9**.
    ///
    /// Both terms above are real on the machine they were measured on and false
    /// on the other. The floor (`line/w`) holds on the i9 and fails on the M4,
    /// where B=16 covers half of a 128 B line — the same half-line case the floor
    /// forbids — and is the optimum, beating B=32 by 15% at side 2048. The
    /// ceiling (`slots/2`) holds on the M4 and fails on the i9, where it is 6 and
    /// every B from 16 to 256 beats B=8. Their geometric mean lands **exactly on
    /// the M4's optimum at both sides** and below the left edge of the i9's
    /// plateau. The residual is **1.15–1.27x of the i9's win**, and it is carried
    /// as a known gap: a per-profile scalar that fixed it would have to be 1 on
    /// the M4 and 4–8 on the i9, i.e. carry the whole answer and derive nothing,
    /// which is the hand-typed flag with extra steps.
    ///
    /// **Open defect S47 found and did not fix (`benches/results-s47`).** At side
    /// **1536 on the M4 this rung fires with B=32 and LOSES 1.75x (1t) / 1.41x
    /// (threaded)** — larger than any i9 shortfall above. It is a fire/decline
    /// defect, not a block one: no block wins there. Both terms vote to fire and
    /// both look right (pressure 24x, a 9.44 MB read array against a 3.36 MB
    /// share); what they cannot see is that the M4's unblocked walk at that width
    /// already costs only 0.594 ns/element — better than side 1024's — so there
    /// is no miss cost to recover, while a non-power-of-two width costs the
    /// permutation two magic-multiply divisions per element that a power-of-two
    /// width gets as shifts (+20–25% instructions, measured). Deciding that trade
    /// needs the price of an unblocked miss on the part, which is the same
    /// unreadable quantity. Sides 512, 1024 and 2048 are unaffected on both
    /// machines.
    pub fn move_block(&self, site: &MoveSite) -> Option<u64> {
        let l1 = self.l1d.as_ref()?;
        let (line, sets, ways) = (l1.line_bytes, l1.sets.max(1), l1.ways());
        let w = elem_bytes(&site.elem);
        let stride = site.cr.checked_mul(w)?;
        // The slow axis must stay inside a line, or there is no line reuse for a
        // block to preserve and this rung has nothing to say about the shape.
        if stride == 0 || site.cq * w >= line {
            return None;
        }
        let touched = if stride.is_multiple_of(line) {
            sets / gcd((stride / line) % sets, sets).max(1)
        } else {
            sets
        };
        // CONFLICT, not capacity — S44's headline, as the gate it always was.
        // A walk that reaches every set is limited by capacity, and capacity is
        // measured FREE on these parts (S43: flat 32 KB → 8 MB). Measured
        // witness: side 1025 has `gcd(4100/128 …)` = no collapse, runs **2.12×
        // FASTER unblocked than side 1024**, and blocking it costs 0.623×. It
        // also scores pressure 1.0009, so without this term it would fire — this
        // is the clause that keeps `lines_live > slots` from being a threshold
        // fitted at its own boundary.
        let slots = touched * ways;
        let lines_live = (site.width * stride / line).min(site.width);
        if touched >= sets || lines_live <= slots || site.len.checked_mul(w)? <= self.l2_per_core()
        {
            return None;
        }
        // COST, not benefit — the one half of this trade that IS derivable (S47).
        //
        // `move_panel_index` divides the counter by `b` four times and by `w/b`
        // twice. When `w` is a power of two every divisor of `gcd(w, rows)` is
        // one too, so `b` and `w/b` both are, and `-O2` lowers all six to shifts.
        // When `w` is NOT a power of two, `b` and `w/b` cannot both be, so at
        // least one magic-multiply sequence lands on every element — **measured
        // at +20 to +25% instructions** on the i9 at width 1536 (185.4 M at
        // B=16 and B=32 against 223.2 M at B=12 and B=24, same loop, same data).
        //
        // The benefit is the unreadable quantity this whole function's note ends
        // on; the cost is not. Firing when a large known cost meets an
        // unpredictable benefit is a bet, and it is one this rung was measured
        // LOSING: at width 1536 on the M4 the fired arm runs **1.75x slower than
        // not blocking at all** (1t; 1.41x threaded), because that part's
        // unblocked walk already costs only 0.594 ns/element — better than its
        // own width-1024 walk — so there is nothing for the permutation to buy.
        //
        // The price of this clause is stated rather than hidden: it forgoes the
        // i9's win at the same width, which the shipped block measured at
        // **1.060x (1t) / 1.044x (threaded)**. Forgoing 6% beats shipping 1.75x.
        // Revisit only if the benefit ever becomes predictable — the plateau
        // there is B = 16..256 and reaching it needs a fact §4 of
        // `benches/results-s47/block-size.md` shows is not readable.
        if !site.width.is_power_of_two() {
            return None;
        }
        // The block balances TWO opposing costs, both of them measured.
        //
        //   traffic  T(B) = max(1, (line/w) / B)   -- a block row shorter than a
        //     line fetches that line once per block-row it is split across. At
        //     B=8 on the i9 (16 f32 per line) it doubles the L1<-L2 traffic, and
        //     it measures 15% slower than B=16 at side 1024, 24% at 2048.
        //   conflict C(B) = max(1, B / (slots/2))  -- the block's read lines and
        //     its write stream compete for the sets the walk can reach. At
        //     B=slots on the M4 it measures 29-56% slower threaded than B=16.
        //
        // Both are 1 inside `[line/w, slots/2]`; when that window is EMPTY —
        // which is the normal case, because the two bounds pull apart — the
        // product `T·C` is minimised at their geometric mean. That is the whole
        // derivation: no threshold, no per-machine number, and it reproduces the
        // M4's swept optimum at BOTH sides (16 at side 1024 and at 2048).
        let floor = (line / w).max(1);
        let ceiling = (slots / 2).max(1);
        let target = (floor * ceiling).isqrt().max(2);
        let tile = gcd(site.width, site.rows);
        (2..=target.min(tile))
            .rev()
            .find(|b| tile.is_multiple_of(*b))
    }

    /// The GPU facts, when this profile describes one. `None` for every CPU
    /// profile — callers that need a GPU fact must ask for it, so a CPU profile
    /// reaching a GPU-only path is a `None` at the seam rather than a silently
    /// wrong constant.
    pub fn gpu(&self) -> Option<&Gpu> {
        match &self.machine {
            Machine::Cpu => None,
            Machine::Gpu(g) => Some(g),
        }
    }

    /// **The GPU rungs this part ships at a site this full, on this face**
    /// (S53): every rung on, minus this profile's rows of [`GPU_RUNGS_OFF`]
    /// that cover the fill and the face. An unmeasured profile takes the
    /// measured part of its architecture ([`GPU_MEASURED`]); with none, every
    /// rung: another architecture's losses are not this one's.
    pub fn gpu_rungs(&self, warps_per_scheduler: u64, contract: bool) -> crate::GpuRungs {
        self.gpu_rungs_on(warps_per_scheduler, Face::of(contract, None))
    }

    /// [`Self::gpu_rungs`] on a site's face, TF32 included.
    pub fn gpu_rungs_on(&self, warps_per_scheduler: u64, face: Face) -> crate::GpuRungs {
        let fill = if warps_per_scheduler < GPU_RESIDENT_BLOCKS {
            Fill::Under
        } else {
            Fill::Filled
        };
        let arch = self.gpu().map(|g| g.arch);
        let same: Vec<&str> = GPU_MEASURED
            .iter()
            .filter(|p| p.name == self.name || p.gpu().map(|g| g.arch) == arch)
            .map(|p| p.name)
            .collect();
        let mut r = crate::GpuRungs::ALL;
        for d in GPU_RUNGS_OFF {
            if same.contains(&d.profile)
                && (d.fill == Fill::Any || d.fill == fill)
                && d.face.covers(face)
            {
                *r.rung(d.rung).expect("a GpuRungs name") = false;
            }
        }
        r
    }

    /// **The `cp.async` panel pipeline for an FMA tile (sm_80+).** Panels stream
    /// into `stages` shared-memory buffers while earlier ones are consumed, so
    /// the prefetch holds no registers. `a` is copied element by element into
    /// its transposed panel — `GPU_VEC_BYTES / elem` lanes along `k` per row,
    /// so four lanes read one 16-byte global run — and its rows are padded by
    /// [`conflict_free_pad`] for exactly that copy pattern. `bk` is the largest
    /// power of two for which two stages fit a block's share of shared memory
    /// and every thread copies whole units; then as many stages as fit, up to
    /// the number of panels. `None` on a part without `cp.async`.
    pub fn gpu_tile_async(&self, g: GpuTile, k: u64, elem: &Ty) -> Option<GpuTile> {
        let gpu = self.gpu()?;
        if !gpu.has_mma_sync() {
            return None;
        }
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let (bm, bn, threads) = (g.bm(), g.bn(), g.threads());
        let pad_a = conflict_free_pad(bm, |l| (l % lanes, l / lanes))?;
        let share = (gpu.smem_per_sm / g.blocks_per_sm.max(1))
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let mut best = None;
        let mut bk = lanes;
        while bk <= k {
            let geom = GpuTile {
                bk,
                stages: 2,
                pad_a,
                ..g
            };
            if geom.smem_bytes(e) > share {
                break;
            }
            let whole =
                (bm * bk).is_multiple_of(threads) && (bk * bn / lanes).is_multiple_of(threads);
            if k.is_multiple_of(bk) && whole {
                best = Some(geom);
            }
            bk *= 2;
        }
        let geom = best?;
        let stages = (share / (geom.stage_elems() * e)).min(k / geom.bk).max(2);
        Some(GpuTile { stages, ..geom })
    }

    /// **The register-staged transposed-A store (S53 rung `pad`).** A thread
    /// loads `lanes` consecutive `k` of one `a` row and stores them as `lanes`
    /// scalar stores, one `k` row of `As` apart. With all `bk / lanes` lanes of a
    /// row in one warp, the lanes sharing a panel row sit `lanes` `k` rows
    /// apart, and no 16-byte-aligned pad separates more than two of them: the
    /// S52 map is a 4-way bank conflict (SASS review). So: the most lanes per
    /// row `L` (a power of two, ≤ `bk / lanes`) for which [`conflict_free_pad`]
    /// finds a pad for lane `l` → (`k` = `lanes·(l mod L)`, row `l / L`), and
    /// the panel still fits the block's share of shared memory. f32 on a
    /// 128-row panel: `L` = 2 (one 32-byte sector per row), pad 4. 4-byte
    /// elements only: the bank model counts one word per element.
    pub fn gpu_tile_sync_a(&self, g: GpuTile, elem: &Ty) -> GpuTile {
        let (Some(gpu), 4) = (self.gpu(), elem_bytes(elem)) else {
            return g;
        };
        let lanes = GPU_VEC_BYTES / 4;
        let share = (gpu.smem_per_sm / g.blocks_per_sm.max(1))
            .saturating_sub(gpu.smem_reserved_per_block)
            .min(gpu.smem_per_block);
        let kv = g.bk / lanes;
        (0..=kv.max(1).ilog2())
            .rev()
            .map(|p| 1u64 << p)
            .find_map(|l| {
                let pad_a = conflict_free_pad(g.bm(), |x| (lanes * (x % l), x / l))?;
                let geom = GpuTile {
                    pad_a,
                    a_row_lanes: l,
                    ..g
                };
                (geom.smem_bytes(4) <= share).then_some(geom)
            })
            .unwrap_or(g)
    }

    /// **The tensor-core geometry for a `rows × c × k` f32 site on the TF32
    /// face** — [`Self::gpu_tile`]'s rules with an `m16n8` tile as the unit.
    ///
    /// 1. **Warp tile.** Each m16n8 tile holds 4 f32 accumulators per thread;
    ///    the most tiles (a power of two) whose accumulators fit half of
    ///    `regs_per_thread`, square-most: 255 registers ⇒ 16 tiles ⇒ 4 × 4,
    ///    a 64 × 32 warp tile.
    /// 2. **Warps per block** from `regs_per_sm` over twice the accumulators,
    ///    split across [`GPU_RESIDENT_BLOCKS`]; halved while the grid has fewer
    ///    blocks than SMs. Before that, `fill` (S53 rung `warp-fill`) halves the
    ///    warp tile's longer side (`wn` on a tie), down to one m16n8 tile, while
    ///    the grid has fewer warps than [`Gpu::fill_warps`] — overriding `wide`
    ///    on a small site. Warps per block stay those of the unshrunk tile.
    /// 3. **Padding** from the fragment access patterns ([`conflict_free_pad`]).
    /// 4. **Panel depth and stages.** `bk` is the largest power of two (≥ the
    ///    mma's k) for which two stages fit a block's share of shared memory
    ///    and every thread copies whole 16-byte chunks; then as many stages as
    ///    the share holds, up to the number of panels.
    ///
    /// `wide` (S52 rung, the SASS review's rule): instead of "accumulators ≤
    /// half the file", count what a thread actually holds — accumulators, both
    /// buffers of its A and B fragments (`2·(4·mt + 2·nt)`), and
    /// [`MMA_ADDR_REGS`] of addresses — against the whole file. 255 registers
    /// ⇒ 32 tiles, 4 × 8 = a 64 × 64 warp tile (CUTLASS's), 224 registers.
    pub fn gpu_mma_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        wide: bool,
        fill: bool,
    ) -> Option<GpuMmaTile> {
        let g = self.gpu()?;
        if !g.has_mma_sync() {
            return None;
        }
        let acc_per_tile = MMA_M * MMA_N / g.warp; // 4
        let (mt, nt, regs_est) = if wide {
            // nt = 2·mt keeps the warp tile square (16·mt = 8·nt).
            let held = |mt: u64| {
                let nt = 2 * mt;
                acc_per_tile * mt * nt + 2 * (4 * mt + 2 * nt) + MMA_ADDR_REGS
            };
            let mut mt = 1;
            while held(2 * mt) <= g.regs_per_thread {
                mt *= 2;
            }
            (mt, 2 * mt, held(mt))
        } else {
            let budget = g.regs_per_thread / 2;
            let mut tiles = 1;
            while 2 * tiles * acc_per_tile <= budget {
                tiles *= 2;
            }
            let mut mt = 1;
            while mt * mt < tiles {
                mt *= 2;
            }
            (mt, tiles / mt, 2 * tiles * acc_per_tile)
        };
        let (mut mt, mut nt) = (mt, nt);
        while fill && (rows / (mt * MMA_M)) * (c / (nt * MMA_N)) < g.fill_warps() {
            if nt * MMA_N >= mt * MMA_M && nt > 1 {
                nt /= 2;
            } else if mt > 1 {
                mt /= 2;
            } else {
                break;
            }
        }
        let resident_threads = g.regs_per_sm / regs_est;
        let warps = (resident_threads / GPU_RESIDENT_BLOCKS / g.warp).max(1);
        // Square-most block tile: warps_y·wm ≈ warps_x·wn.
        let (wm, wn) = (mt * MMA_M, nt * MMA_N);
        let aspect = |wy: u64| {
            let (h, w) = (wy * wm, warps / wy * wn);
            h.max(w) / h.min(w)
        };
        let mut warps_y = (0..=warps.ilog2())
            .map(|p| 1u64 << p)
            .min_by_key(|&wy| aspect(wy))
            .unwrap_or(1);
        let mut warps_x = warps / warps_y;
        let mut halve_x = true;
        while (rows / (warps_y * wm)) * (c / (warps_x * wn)) < g.sms && warps_x * warps_y > 1 {
            if halve_x && warps_x > 1 {
                warps_x /= 2;
            } else if warps_y > 1 {
                warps_y /= 2;
            }
            halve_x = !halve_x;
        }
        let (bm, bn) = (warps_y * wm, warps_x * wn);
        if rows % bm != 0 || c % bn != 0 {
            return None;
        }
        let threads = warps_x * warps_y * g.warp;
        let blocks_per_sm = (resident_threads / threads)
            .min(g.max_threads_per_sm / threads)
            .min(g.max_blocks_per_sm)
            .max(1);
        let share = (g.smem_per_sm / blocks_per_sm)
            .saturating_sub(g.smem_reserved_per_block)
            .min(g.smem_per_block);

        // Fragment patterns of m16n8k8 (PTX ISA): lane = 4·group + t.
        // A (row-major): row = group, col = t.  B (k × n): row = t, col = group.
        let pad_b = conflict_free_pad(bn, |l| (l % 4, l / 4))?;
        let mut best = None;
        let mut bk = MMA_K;
        while bk <= k {
            let pad_a = conflict_free_pad(bk, |l| (l / 4, l % 4))?;
            let geom = GpuMmaTile {
                mt,
                nt,
                warps_x,
                warps_y,
                bk,
                stages: 2,
                pad_a,
                pad_b,
                blocks_per_sm,
            };
            let whole =
                (bm * bk / 4).is_multiple_of(threads) && (bk * bn / 4).is_multiple_of(threads);
            if geom.smem_bytes() > share {
                break;
            }
            if k.is_multiple_of(bk) && whole {
                best = Some(geom);
            }
            bk *= 2;
        }
        let geom = best?;
        let stages = (share / (geom.stage_words() * 4)).min(k / geom.bk).max(2);
        Some(GpuMmaTile { stages, ..geom })
    }

    /// **The Hopper `wgmma` geometry for a `rows × c × k` f32 site on the TF32
    /// face** ([`GpuWgmmaTile`]). `None` off `sm_90a`. For each panel depth
    /// `bk` (a power of two ≥ the instruction's k, dividing `k`):
    ///
    /// 1. **`n`** — the largest multiple of 8 ≤ 256 dividing `rows` whose
    ///    `n/2` accumulators, the panel's `4·bk/8` A fragments and
    ///    [`WGMMA_ADDR_REGS`] fit `regs_per_thread` (the whole-thread rule).
    /// 2. **Warpgroups per block** (a power of two) from `regs_per_sm` over
    ///    that count, all in **one** resident block, not split across
    ///    [`GPU_RESIDENT_BLOCKS`]: that policy keeps a second block computing
    ///    while one waits at the panel barrier, and here the MMA group in
    ///    flight computes through it. Measured (S53 run 1, H100 PCIe): the
    ///    split geometry ran 0.61–0.64× of this one at 2048–8192, and with
    ///    128-byte swizzle rows (`bk` ≥ 32) it has no three-stage tile at all.
    /// 3. **Fill** — while the grid has fewer blocks than SMs, halve the
    ///    warpgroups, then `n`.
    /// 4. **Smem** — `b` rows padded by [`conflict_free_pad`] for the A
    ///    fragment pattern; the panel must fit [`WGMMA_MIN_STAGES`] stages
    ///    in a block's share, copy in whole 16-byte chunks per thread, and
    ///    fill whole [`WGMMA_SWIZZLE_BYTES`] rows.
    ///
    /// The deepest valid `bk` wins; then as many stages as the share holds.
    pub fn gpu_wgmma_tile(&self, rows: u64, c: u64, k: u64) -> Option<GpuWgmmaTile> {
        let g = self.gpu()?;
        if !g.has_wgmma() {
            return None;
        }
        let wg_threads = WARPGROUP_WARPS * g.warp;
        // The largest legal `n` ≤ `cap` dividing `rows`.
        let fit_n = |cap: u64| {
            (1..=cap.min(WGMMA_N_MAX) / WGMMA_N_STEP)
                .rev()
                .map(|i| i * WGMMA_N_STEP)
                .find(|n| rows.is_multiple_of(*n))
        };
        let regs = |n: u64, bk: u64| n / 2 + 4 * (bk / WGMMA_K) + WGMMA_ADDR_REGS;
        let mut best = None;
        let mut bk = WGMMA_K;
        while bk <= k {
            let frag_regs = regs(0, bk);
            let geom = (|| {
                if !k.is_multiple_of(bk) || frag_regs >= g.regs_per_thread {
                    return None;
                }
                let mut n = fit_n(2 * (g.regs_per_thread - frag_regs))?;
                let resident = g.regs_per_sm / regs(n, bk);
                let per_block = (resident / wg_threads)
                    .min(g.max_threads_per_block / wg_threads)
                    .max(1);
                let mut groups = 1 << per_block.ilog2();
                while (rows / n) * (c / (groups * WGMMA_M)) < g.sms {
                    if groups > 1 {
                        groups /= 2;
                    } else if let Some(m) = fit_n(n / 2) {
                        n = m;
                    } else {
                        break;
                    }
                }
                let (bn, threads) = (groups * WGMMA_M, groups * wg_threads);
                if !c.is_multiple_of(bn) {
                    return None;
                }
                let blocks_per_sm = (g.regs_per_sm / regs(n, bk) / threads)
                    .min(g.max_threads_per_sm / threads)
                    .min(g.max_blocks_per_sm)
                    .max(1);
                let share = (g.smem_per_sm / blocks_per_sm)
                    .saturating_sub(g.smem_reserved_per_block)
                    .min(g.smem_per_block);
                // A fragment pattern: lane = 4·group + t reads row k = t,
                // column n = group of the `b` panel.
                let pad_b = conflict_free_pad(bn, |l| (l % 4, l / 4))?;
                let geom = GpuWgmmaTile {
                    n,
                    groups,
                    bk,
                    stages: WGMMA_MIN_STAGES,
                    pad_b,
                    blocks_per_sm,
                };
                // Whole 16-byte chunks per thread, and whole swizzle rows.
                let whole = (n * bk / 4).is_multiple_of(threads)
                    && (bk * bn / 4).is_multiple_of(threads)
                    && (bk * 4).is_multiple_of(WGMMA_SWIZZLE_BYTES);
                (whole && geom.smem_bytes() <= share).then(|| {
                    let stages = (share / (geom.stage_words() * 4))
                        .min(k / bk)
                        .max(WGMMA_MIN_STAGES);
                    GpuWgmmaTile { stages, ..geom }
                })
            })();
            best = geom.or(best);
            bk *= 2;
        }
        best
    }

    /// **The GPU tile geometry for a `rows × c × k` site — the same three
    /// budgets the CPU rungs read, at their GPU locations.** `None` when the
    /// site does not divide into any geometry this derives; the emitter then
    /// takes the one-thread-per-cell kernel, which is slower and never wrong.
    ///
    /// | CPU rule | GPU reading here |
    /// | --- | --- |
    /// | [`Self::tile_i`]: half the vector file holds accumulators | half of `regs_per_thread` holds the `tm × tn` register tile |
    /// | [`Self::tile_kc`]: half of L2 holds one packed panel | a block's share of `smem_per_sm` holds the `bk`-deep A and B panels |
    /// | the parallel driver's slice count | blocks: at least one per SM |
    ///
    /// Steps, each a fact or a stated policy:
    /// 1. **Register tile.** The largest square power-of-two `t`, at least one
    ///    vector wide, with `t² · regs(elem) ≤ regs_per_thread / 2` (the CPU's
    ///    half-the-file headroom). f32 on 255 registers: `8 × 8`.
    /// 2. **Threads per block.** Registers per thread are estimated at twice
    ///    the accumulators; `regs_per_sm` over that is the resident-thread
    ///    budget, split across [`GPU_RESIDENT_BLOCKS`] blocks so one computes
    ///    while another waits at the barrier. 65 536 / 128 / 2 = 256 = 16 × 16.
    /// 3. **Fill the schedulers** (`fill`, S53 rung `warp-fill`). While the
    ///    grid has fewer warps than [`Gpu::fill_warps`], halve the register
    ///    tile's longer side (`tn` on a tie), down to one vector. Grid warps
    ///    are `rows·c / (tm·tn·warp)`, which no thread-grid split changes;
    ///    threads per block stay those of the unshrunk tile (step 2).
    /// 4. **Fill the machine.** While the grid has fewer blocks than SMs, halve
    ///    the thread grid (x then y), down to one warp — a block left idle is a
    ///    whole SM idle, which no tile shape wins back.
    /// 5. **Panel depth.** Blocks per SM = the tightest of registers, threads
    ///    and the block-slot limit; `bk` is the largest power of two whose two
    ///    panels fit one block's share of shared memory, dividing `k`, at least
    ///    one vector deep, and loadable in whole vectors by every thread.
    ///
    /// `wide` (S53 rung `wide-fma`, the TF32 `wide` rule on the FMA tile):
    /// step 1 counts what a thread holds ([`fma_held_regs`]) against the whole
    /// file — the largest `tm × tn`, `tn ∈ {tm, 2·tm}`, square first: 255 ⇒
    /// 8 × 16 (128 + 48 + 32 = 208). Step 2: threads = `regs_per_sm` over that
    /// count over [`GPU_RESIDENT_BLOCKS`], rounded down to a power of two
    /// (65 536 / 208 / 2 ⇒ 128), split square-most on the *block* tile
    /// (`ty·tm ≈ tx·tn` ⇒ 16 × 8 ⇒ 128 × 128). Steps 3 and 4 never shrink it:
    /// a site with fewer such blocks than SMs — or, under `fill`, fewer warps
    /// than [`Gpu::fill_warps`] — takes the S52 geometry instead, which they
    /// then fill.
    pub fn gpu_tile(
        &self,
        rows: u64,
        c: u64,
        k: u64,
        elem: &Ty,
        wide: bool,
        fill: bool,
    ) -> Option<GpuTile> {
        let g = self.gpu()?;
        let e = elem_bytes(elem);
        let lanes = (GPU_VEC_BYTES / e).max(1);
        let regs_per_elem = e.div_ceil(4);

        let (mut tm, mut tn, regs_est) = if wide {
            let (mut tm, mut tn) = (lanes, lanes);
            loop {
                let (m, n) = if tn == tm { (tm, 2 * tm) } else { (2 * tm, tn) };
                if fma_held_regs(m, n, e) > g.regs_per_thread {
                    break;
                }
                (tm, tn) = (m, n);
            }
            (tm, tn, fma_held_regs(tm, tn, e))
        } else {
            let budget = g.regs_per_thread / 2;
            let mut t = lanes;
            while (2 * t) * (2 * t) * regs_per_elem <= budget {
                t *= 2;
            }
            (t, t, 2 * t * t * regs_per_elem)
        };
        let under_warps =
            |tm: u64, tn: u64| fill && (rows / tm) * (c / tn) / g.warp < g.fill_warps();
        while !wide && under_warps(tm, tn) {
            if tn >= tm && tn > lanes {
                tn /= 2;
            } else if tm > lanes {
                tm /= 2;
            } else {
                break;
            }
        }

        let resident_threads = g.regs_per_sm / regs_est;
        let threads = (resident_threads / GPU_RESIDENT_BLOCKS).min(g.max_threads_per_block);
        let (mut tx, mut ty) = if wide {
            let threads = 1 << threads.max(1).ilog2();
            let aspect = |ty: u64| {
                let (h, w) = (ty * tm, threads / ty * tn);
                h.max(w) / h.min(w)
            };
            let ty = (0..=threads.ilog2())
                .map(|p| 1u64 << p)
                .min_by_key(|&ty| aspect(ty))
                .unwrap_or(1);
            (threads / ty, ty)
        } else {
            // Square-most power-of-two split of the thread grid, x the wider.
            let mut tx = 1;
            while tx * tx < threads {
                tx *= 2;
            }
            (tx, (threads / tx).max(1))
        };

        // The wide tile buys reuse with threads; a site that cannot give every
        // SM a block at its full grid keeps the S52 tile, which step 3 then
        // fills (S53: halving the wide grid at 1024² measured 0.70× on Ada and
        // Blackwell — half the warps per block over the same tiles).
        if wide && ((rows / (ty * tm)) * (c / (tx * tn)) < g.sms || under_warps(tm, tn)) {
            return self.gpu_tile(rows, c, k, elem, false, fill);
        }
        let mut halve_x = true;
        while (rows / (ty * tm)) * (c / (tx * tn)) < g.sms && tx * ty > g.warp {
            if halve_x && tx > 1 {
                tx /= 2;
            } else if ty > 1 {
                ty /= 2;
            }
            halve_x = !halve_x;
        }

        let geom = |bk| GpuTile {
            tm,
            tn,
            tx,
            ty,
            bk,
            blocks_per_sm: 0,
            stages: 1,
            pad_a: 0,
            a_row_lanes: 0,
            held: if wide { regs_est } else { 0 },
        };
        let (bm, bn, threads) = (geom(0).bm(), geom(0).bn(), tx * ty);
        if rows % bm != 0 || c % bn != 0 {
            return None;
        }
        let blocks_per_sm = (resident_threads / threads)
            .min(g.max_threads_per_sm / threads)
            .min(g.max_blocks_per_sm)
            .max(1);
        let share = (g.smem_per_sm / blocks_per_sm)
            .saturating_sub(g.smem_reserved_per_block)
            .min(g.smem_per_block);

        let loads_whole = |bk: u64| {
            (bm * bk / lanes).is_multiple_of(threads) && (bk * bn / lanes).is_multiple_of(threads)
        };
        let mut bk = lanes;
        let mut best = None;
        while bk <= k && (bm + bn) * bk * e <= share {
            if k.is_multiple_of(bk) && loads_whole(bk) {
                best = Some(bk);
            }
            bk *= 2;
        }
        Some(GpuTile {
            blocks_per_sm,
            ..geom(best?)
        })
    }

    /// The streaming matrix unit's facts, when this part has one. Same seam
    /// discipline as [`TargetProfile::gpu`]: a profile without SME reaching an
    /// SME-only path is a `None`, never a wrong constant.
    pub fn sme(&self) -> Option<&Sme> {
        self.sme.as_ref()
    }

    /// The side of one square ZA tile at this element width — `svl_bytes /
    /// sizeof(elem)`, i.e. **16 for f32 at SVL 512**. `None` without an SME
    /// unit.
    ///
    /// This is the one number the emitted kernel's every literal comes from
    /// (the panel side, the ZA row count, the packed-A row stride). It is a
    /// per-`Loc` machine fact and must never reach `mapal-ir`.
    pub fn sme_tile_side(&self, elem: &Ty) -> Option<u64> {
        Some(self.sme.as_ref()?.svl_bytes / elem_bytes(elem))
    }

    /// How many ZA tiles one panel accumulates into, and in what arrangement:
    /// `(ti, tj)` with `ti · tj` = the architectural tile count at this width,
    /// so a panel covers `ti·t × tj·t` outputs. `None` without an SME unit.
    ///
    /// This is [`TargetProfile::tile_i`]'s question asked of the other
    /// accumulator file — "how many accumulator blocks stay resident" — and it
    /// is answered the same way, by arithmetic rather than by a literal.
    ///
    /// **The tile count is not a recorded fact; it is an ISA rule.** ZA is
    /// `svl_bytes × svl_bytes` bytes and one tile at element width `w` is
    /// `(svl/w)² · w` bytes, so the count is `svl²/((svl/w)²·w) = w` — always
    /// **exactly the element byte width**, for any SVL: 4 tiles at f32
    /// (`ZA0.S`…`ZA3.S`), 8 at f64 (`ZA0.D`…`ZA7.D`), 2 at f16, 1 at i8. That is
    /// the architecture, so it is derived here rather than recorded per part.
    ///
    /// This replaced an `f32_tiles: 4` field. Deriving it removes the last
    /// hardcoded SME number *and* closes the S41b P2 defect that the field
    /// created: a recorded count could say `7`, for which this function would
    /// happily return a 1×7 arrangement that no ZA register file can hold and
    /// no emitted kernel could select. A count that is `elem_bytes` by
    /// construction cannot be wrong.
    ///
    /// The **shape** is a derivation too, not taste. One panel issues `ti · tj`
    /// outer products from `ti + tj` operand loads, and `ti + tj` is minimised
    /// by the most-square factorization: 2×2 feeds 4 MACs from 4 loads, while
    /// 1×4 would need 5 for the same 4. Measured on this part
    /// (`benches/sme/mm4.c`, 2×2 vs 1 tile, f32, 1 thread): 423 → 777 GFLOP/s
    /// at 1024², 237 → 619 at 2048².
    pub fn sme_block(&self, elem: &Ty) -> Option<(u64, u64)> {
        self.sme.as_ref()?;
        let tiles = elem_bytes(elem).max(1);
        // The largest divisor no bigger than sqrt(tiles) — the square-most split.
        let ti = (1..=tiles)
            .filter(|d| d * d <= tiles && tiles.is_multiple_of(*d))
            .max()
            .unwrap_or(1);
        Some((ti, tiles / ti))
    }

    /// The SME rung's k-panel depth: how deep a k block may be before the two
    /// packed operand panels stop fitting the unit's working-set budget.
    /// `None` without an SME unit.
    ///
    /// **There is no opt-in.** The moment a profile records a matrix unit, every
    /// factor that unit's realization needs is *derived* from the facts recorded
    /// alongside it — the same contract [`TargetProfile::sme_block`] and
    /// [`TargetProfile::sme_tile_side`] already honour. A gate the caller has to
    /// remember to open is how `f32_tiles: 4` sat in this file while the emitter
    /// used one tile (S41b), and the deficit was ~4×.
    ///
    /// One k step of a panel streams `ti·t` elements of packed `a` and `tj·t` of
    /// packed `b`, so the window is `(ti + tj)·t·sizeof` bytes per k, and the
    /// depth that fits is the budget divided by it. On this part that is
    /// arithmetic with no free parameter:
    ///
    /// ```text
    /// (ti + tj)·t   = (2 + 2)·16 = 64 elements per k
    /// 64 · 4 B      = 256 B per k
    /// 2*131072 / 256 = 1024       <- the depth swept in the emitter
    /// ```
    ///
    /// **1024 is not written down anywhere; it falls out of L1D x the swept ratio.** That is the
    /// `plan-s31-deduced-blocking.md` discipline — a literal swept once is
    /// replaced by a derivation that reproduces it on the machine it was swept
    /// on, and yields a defensible number on a machine nobody has swept. `f64`
    /// scales off the same fact by element width, so there is no second constant.
    ///
    /// Measured (`benches/sme/kc.c`, N=4096, f32, 1 thread, 9 alternating runs,
    /// medians, every KC gated against an independent scalar reference): 64 KB →
    /// 845.1, **128 KB → 1101.2**, 256 KB → 1046.7, 512 KB → 885.1, unblocked
    /// (1 MB) → 760.7 GFLOP/s. Unimodal, so there is a real optimum rather than
    /// a trend, and the peak puts N=4096 level with N=1024.
    ///
    /// The caller gates on `site.k > sme_kc`, so at shallow k the nest disables
    /// itself by derivation — which is correct, because the probe shows blocking
    /// is a **loss** at K ≤ 512 where the panel already fits.
    /// Does packing `b` pay at a site of this size? `true` keeps today's
    /// unconditional pack.
    ///
    /// **The pack computes nothing.** It exists only to give `b` a second
    /// `DataLoc` in a friendlier layout (FRAMEWORK §4.2), so its cost is a
    /// transmission price — `2·k·c·sizeof` bytes moved — against a benefit that
    /// exists only while `b`'s re-reads actually miss. The cost is O(N²) and the
    /// matmul it serves is O(N³), so below some size the pack is pure loss.
    ///
    /// **Measured, plan-s51 H1** (`benches/nn/pack_ab.sh`, M4 Pro, 15
    /// interleaved cycles, values bit-identical on every arm, and a
    /// byte-identical control arm carried through every cell as the noise
    /// floor — 0.990–1.071×, all overlapping):
    ///
    /// | d | working set | `--no-pack` / pack, 1 thread | threaded |
    /// | ---: | ---: | ---: | ---: |
    /// | 128 | 0.2 MB | **1.46–1.52× faster WITHOUT** | 1.45–1.51× |
    /// | 256 | 0.8 MB | 1.20–1.21× | **1.41–1.46×** |
    /// | 512 | 3.1 MB | 1.13× | 1.23–1.32× |
    /// | 1024 | **12.6 MB** | **1.000× / 0.992× — the crossover** | ~1.0 |
    /// | 2048 | 50 MB | 0.65× — packing WINS 1.53× | 0.66–0.67× |
    ///
    /// **The crossover was not fitted; it lands on a knee measured by a
    /// different instrument.** S43's `loadlevel.c` put this part's one real
    /// cache boundary — shared L2 → DRAM — between 8 and 12 MB, and the sign
    /// flip is at a 12.6 MB working set. Three sizes below and one above are
    /// all predicted correctly by `working_set > l2_bytes`, so that is the rule,
    /// and `l2_bytes` is already a recorded field. No constant is added.
    ///
    /// The data does not separate this from `k·c·sizeof > l2_bytes` (B alone) —
    /// both predicates are right at all five measured sizes. The whole working
    /// set is chosen because A and the output really do compete with `b` for the
    /// same L2, which is the mechanism; if a later size ever separates them,
    /// that measurement decides.
    ///
    /// **Scope is the CALLER's job, and it is narrower than "this profile has an
    /// SME unit".** [`crate::func::packing_site`] asks this only for sites that
    /// take the SME rung, because that is the only kernel the crossover was
    /// measured on. An f64 or exact-face site on this very profile runs the NEON
    /// packing rung and packs exactly as it did before — nobody has taken its
    /// crossover, and a threshold is not transplanted between kernels on the
    /// strength of an argument. That scoping is what keeps `generic`
    /// byte-identical, and it is S50's rule in the same words: **no part is made
    /// worse by a number nobody took.**
    pub fn pack_pays(&self, rows: u64, c: u64, k: u64, elem: &Ty) -> bool {
        let working_set = (rows * k + k * c + rows * c) * elem_bytes(elem);
        working_set > self.l2_bytes
    }

    /// How deep a k block the A pack should copy at a time, or `None` to copy
    /// the whole k axis in one pass — which is what shipped before this existed.
    ///
    /// **The A pack's waste is on the STORE side, and it is conditional.** The
    /// emitted pack is row-outer / k-inner (S42), so reads walk a row
    /// contiguously — that side is already right. The stores go to
    /// `ap[pk·panel_rows + pi]`, and consecutive `pk` are `panel_rows` elements
    /// apart, which at f32 with a 2×2 block is 128 B — **exactly one cache line,
    /// of which each store writes 4 bytes.** The next row then comes back and
    /// writes the next 4 bytes of those same lines.
    ///
    /// That costs nothing *if* the lines are still resident when the next row
    /// arrives, i.e. if `ap` = `panel_rows · kc · sizeof` fits L1D. Past that,
    /// every line is evicted before its other 31 rows fill it, and the pack pays
    /// 32× the line traffic it uses.
    ///
    /// Measured (`benches/sme/packconflict.c`, the loops the emitter actually
    /// emits, best of 7, checksum-gated across arms — and the gate verified to
    /// fail on an injected bug before the run was trusted):
    ///
    /// | N | `ap` | 32×32 blocked vs shipped |
    /// | ---: | ---: | ---: |
    /// | 128–512 | ≤ 64 KB | 1.000–1.051× — **flat, nothing to win** |
    /// | 1024 | 128 KB = L1D | 1.612×, but that cell is **VOID** (drift 1.507×) |
    /// | 2048 | 256 KB | **2.752×** |
    /// | 4096 | 512 KB | 1.771× |
    ///
    /// So the gate is exactly "does `ap` fit L1D", and the block is one cache
    /// line's worth of `k` — `line_bytes / sizeof` — which makes both sides
    /// local: `panel_rows` read lines and `block` write lines live at once,
    /// independent of `k`.
    ///
    /// **Not a swept constant.** Both numbers are detected machine facts
    /// (`hw.pagesize`-class `sysctl` reads in [`L1d`]); a profile with no L1D
    /// geometry gets `None` and keeps today's single-pass copy.
    ///
    /// Note what this does NOT do: it does not revert S42. The pre-S42 k-outer
    /// order is 1.455× *faster* than what ships at N=2048 and ~2× slower at
    /// N≤512; blocking beats both (1.89× over the pre-S42 order at N=2048), so
    /// the answer is to block the traversal, not to swap the loops back.
    pub fn pack_block(&self, panel_rows: u64, kc: u64, elem: &Ty) -> Option<u64> {
        let l1d = self.l1d.as_ref()?;
        let width = elem_bytes(elem);
        if panel_rows.saturating_mul(kc).saturating_mul(width) <= l1d.bytes {
            return None;
        }
        Some((l1d.line_bytes / width.max(1)).max(1))
    }

    pub fn sme_kc(&self, elem: &Ty) -> Option<u64> {
        let sme = self.sme.as_ref()?;
        // A detected machine fact times a SWEPT policy ratio — see
        // `Sme::panel_l1d_ratio` for the two depth sweeps that set it, and for why
        // folding the 2 into this expression would be dishonest.
        let budget = sme.l1d_bytes * sme.panel_l1d_ratio;
        let t = self.sme_tile_side(elem)?;
        let (ti, tj) = self.sme_block(elem)?;
        Some((budget / (elem_bytes(elem) * (ti + tj) * t)).max(1))
    }
}

/// Today's six literals, exactly: 128-bit vectors, 32 registers, a 512 KB
/// per-core L2. Emission under this profile is byte-identical to the
/// pre-profile emitter for every shape — the plan's rule 1, and the reason it
/// is the default.
pub const GENERIC: TargetProfile = TargetProfile {
    name: "generic",
    machine: Machine::Cpu,
    vec_bytes: 16,
    vec_regs: 32,
    acc_vecs_per_row: 4,
    nc_tiles: 32,
    l2_bytes: 512 * 1024,
    l2_cores: 1,
    heap_min_bytes: 256 * 1024,
    sme: None,
    // No L1D geometry ⇒ the S45 move rung cannot fire under the DEFAULT
    // profile, so `--target=generic` stays byte-identical for all 171 swept
    // cells. Firing needs a profile that describes a real part.
    l1d: None,
    // No measured rates ⇒ every task is dispatched, which is what every profile
    // did before this field existed. `apple-m` and `zen3` inherit it because
    // nobody has measured them either; `raptorlake`'s numbers are OWED and were
    // not taken because the box was unreachable this session.
    dispatch: None,
};

/// Apple M-series: NEON at 16 B × 32 registers, but a 16 MB L2 **shared by five
/// P-cores** (`hw.perflevel0.l2cachesize` / `hw.perflevel0.cpusperl2` on this M4
/// Pro). Same tile widths as `generic`; the L2 is what differs, and it closes the
/// KC gate by derivation.
pub const APPLE_M: TargetProfile = TargetProfile {
    name: "apple-m",
    l2_bytes: 16 * 1024 * 1024,
    l2_cores: 5,
    ..GENERIC
};

/// AVX2 zen3: 32 B vectors, 16 registers, 512 KB per-core L2.
///
/// **UNTESTED — read off documentation, never measured.** Note it moves two
/// constants at once relative to `generic`: `tile_j(f32)` 16 → 32 and `tile_i`
/// 4 → 2. That is self-consistent arithmetic, not a validated configuration;
/// the box leg is what settles it.
pub const ZEN3: TargetProfile = TargetProfile {
    name: "zen3",
    vec_bytes: 32,
    vec_regs: 16,
    ..GENERIC
};

/// Intel Raptor Lake — the i9-14900F box (24C/32T, 8 P + 16 E, governor
/// `performance`). **The cross-compilation case, and the reason named profiles
/// still exist:** the box has gcc and no clang, so every leg is emitted on the
/// Mac, where nothing about the i9 can be detected. Every number here was read
/// off `/sys/devices/system/cpu/cpu0/cache/` on the box itself and is recorded
/// with its source:
///
/// | field | source | value |
/// | --- | --- | --- |
/// | `l1d.bytes` | `index0/size` | 48K (P-core; the E-core is 32K) |
/// | `l1d.line_bytes` | `index0/coherency_line_size` | 64 |
/// | `l1d.page_bytes` | `getconf PAGESIZE` | 4096 |
/// | `l2_bytes` | `index2/size` | 2048K |
/// | `l2_cores` | `index2/shared_cpu_list` = `0-1` | **1** physical core (two SMT threads) |
///
/// `sets`/`ways` are **not** recorded: `page/line = 64` and
/// `48K/(64·64) = 12` reproduce `number_of_sets=64` and
/// `ways_of_associativity=12` exactly (`l1d_derivation_reproduces_the_i9`).
///
/// It describes the **P-core**, which is what the 1-thread legs are pinned to
/// (`taskset -c 4`). The E-core's 32K/8-way L1D derives 8 ways from the same
/// rule and would give the same block here; the P-core is the conservative and
/// the measured one.
pub const RAPTORLAKE: TargetProfile = TargetProfile {
    name: "raptorlake",
    vec_bytes: 32,
    vec_regs: 16,
    l2_bytes: 2 * 1024 * 1024,
    l2_cores: 1,
    l1d: Some(L1d {
        bytes: 48 * 1024,
        line_bytes: 64,
        sets: 64,
    }),
    ..GENERIC
};

/// **MEASUREMENT SCRATCH — S51b, DELETE AFTER USE. Not a machine.**
///
/// `raptorlake` with one field changed: a `dispatch_ns` so large that
/// [`FnEmit::under_dispatch_cost`]'s roofline answers "inline" for every tile
/// site in the program. It exists to make the `dispatch_ns` A/B possible at all,
/// because that constant can only be measured as a DIFFERENCE — dispatch
/// everything (`raptorlake`, `dispatch: None`) against inline everything (this)
/// — divided by the number of dispatches that flipped.
///
/// `flops_per_ns` is the real measured 144; `bytes_per_ns` is a placeholder and
/// is not read for any decision here, because `ns < dispatch_ns` holds for every
/// site at this `dispatch_ns` whatever the rate. Only the huge value matters.
///
/// **Do not ship a program built against this.** It is the `kc_nest` mistake in
/// miniature — a measurement lever that must not outlive its measurement — and
/// the repo already carries a P1 to delete that one.
pub const RAPTORLAKE_INLINE: TargetProfile = TargetProfile {
    name: "raptorlake-inline",
    dispatch: Some(DispatchCost {
        flops_per_ns: 144,
        bytes_per_ns: 100,
        dispatch_ns: 1_000_000_000,
    }),
    ..RAPTORLAKE
};

/// NVIDIA Ada (`sm_89`) — the box GPU, an RTX 4070 Ti (compute capability 8.9,
/// driver 610.43.03, 12 GB). plan-s41 step 1; S52 gave it the facts the tile
/// geometry reads ([`TargetProfile::gpu_tile`]).
///
/// The CPU-shaped fields are inherited from [`GENERIC`]: on a GPU profile they
/// describe the **host** that runs everything outside an offloaded site, and
/// no GPU realization reads them. S52 deliberately did NOT give them GPU
/// readings (plan-s41 §2.1 proposed `vec_bytes` → per-thread registers,
/// `l2_bytes` → shared memory): one field with two readings is the `l2_bytes`
/// per-core/shared ambiguity that cost S45 a session, so every GPU quantity is
/// its own [`Gpu`] field.
///
/// `sm_89` has 4th-generation tensor cores; `mma` is the precision-face rung
/// (ADR-0032 D1/D3), not this one.
pub const CUDA_ADA: TargetProfile = TargetProfile {
    name: "cuda-ada",
    machine: Machine::Gpu(ADA_SM),
    ..GENERIC
};

/// The `sm_89` per-SM facts (CUDA C Programming Guide, compute capability 8.9),
/// with the RTX 4070 Ti's 60 SMs and its 48 MB L2 (AD104, Ada whitepaper).
const ADA_SM: Gpu = Gpu {
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
};

/// RTX 4090 — Ada (`sm_89`), 128 SMs. A rented part (S52): the profile is the
/// cross-compilation spelling, `benches/gpu/devquery.cu` on the box is its check.
pub const RTX_4090: TargetProfile = TargetProfile {
    name: "rtx-4090",
    // 72 MB L2: `facts l2_bytes=75497472` (base-rtx-4090.txt).
    machine: Machine::Gpu(Gpu {
        sms: 128,
        l2_bytes: 72 * 1024 * 1024,
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
        ..ADA_SM
    }),
    ..GENERIC
};

/// RTX 3090 — Ampere GA102 (`sm_86`), 82 SMs (S54). Compute capability 8.6
/// differs from Ada's per-SM facts in one field: 16 resident blocks per SM, not
/// 24 (CUDA C Programming Guide, compute-capability table); 100 KB shared memory
/// per SM, 99 KB per block, 1536 threads per SM, as on Ada.
/// **UNMEASURED** — not in [`GPU_MEASURED`], so every rung is ON until its own
/// session measures the arms.
pub const RTX_3090: TargetProfile = TargetProfile {
    name: "rtx-3090",
    machine: Machine::Gpu(Gpu {
        arch: "sm_86",
        sms: 82,
        max_blocks_per_sm: 16,
        // `facts l2_bytes=6291456` (base-rtx-3090.txt).
        l2_bytes: 6 * 1024 * 1024,
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
/// **UNMEASURED** — every rung ON (see [`RTX_3090`]).
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
        ..ADA_SM
    }),
    ..GENERIC
};

/// How many blocks the thread budget is split across on one SM: two, so one
/// computes while the other waits at the panel barrier. Policy, like
/// [`Sme::panel_l1d_ratio`] — not a property of the silicon.
pub const GPU_RESIDENT_BLOCKS: u64 = 2;

/// Apple M4 with the streaming matrix unit enabled: [`APPLE_M`] plus the SME
/// facts measured on the part (`benches/sme/svl.c` — SVL 64 B, 4 f32 tiles;
/// `hw.optional.arm.FEAT_SME`/`FEAT_SME2`/`FEAT_SME_F32F32` all set).
///
/// It is a **separate name** rather than a field flipped on `apple-m` because
/// the SME leaf is only bit-equal to the NEON leaf on the contract face
/// (`fmopa` fuses — measured 92/256 cells differ against separate mul+add,
/// 0/256 against `fmaf`), so selecting it is a decision about the *program*, not
/// only about the machine. Naming it keeps `apple-m` byte-identical forever and
/// keeps a box run reproducible.
pub const APPLE_M4_SME: TargetProfile = TargetProfile {
    name: "apple-m4-sme",
    // Both facts are `hw.optional.arm.sme_max_svl_b` and
    // `hw.perflevel0.l1dcachesize` on this M4 Pro, and `native` reads both — so
    // this profile is now the *cross-compilation* spelling of a machine you are
    // not sitting on, and
    // `native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts`
    // asserts detection agrees with it on the documented part (ADR-0038 D4).
    // The tile count is absent because it is an ISA rule, derived in `sme_block`.
    sme: Some(Sme {
        svl_bytes: 64,
        l1d_bytes: 128 * 1024,
        panel_l1d_ratio: 2,
    }),
    // S45: the same L1D, plus the two facts the move rung needs and the SME rung
    // does not. `hw.cachelinesize` = 128 and `hw.pagesize` = 16384 on this part;
    // 128 sets and 8 ways follow (`L1d`), so no associativity is written down.
    l1d: Some(L1d {
        bytes: 128 * 1024,
        line_bytes: 128,
        sets: 128,
    }),
    // MEASURED ON THIS PART, all three (plan-s50 P3):
    //
    // `flops_per_ns` — the SME rung's one-lane rate. `nn4_1024` at MAPAL_PAR=1
    //   runs 986 GF/s and `nn4_2048` 939; 1000 is the plateau.
    // `bytes_per_ns` — the streaming rate one lane reaches. `saxpy_1048576` at
    //   MAPAL_PAR=1 moves 12.58 MB in 0.0991 ms = 127 B/ns.
    // `dispatch_ns` — THE MARGINAL COST OF ONE DISPATCH IN THIS RUNTIME, which
    //   is NOT the parked-wave figure. `handoff_floor` measures a parked wave at
    //   22.6-41.2 us, and the first version of this constant used 30 us — but
    //   plan-s50's P0/P0b/P2/P1 and the single-slice inline rule made a real
    //   dispatch far cheaper than a cold park. Measured directly instead:
    //   `nn4_128` opens 8 dispatches and runs 0.1449 ms with them and 0.0788 ms
    //   with all eight inlined, so one dispatch costs (144.9 - 78.8)/8 = 8.3 us.
    //
    //   Using the parked figure cost three cells: `conv2d_512` 0.805x,
    //   `matmul512` 0.944x and `nn4_512` 0.945x, all of them things that pay for
    //   their dispatch at 8 us and do not at 30. The constant has to track the
    //   runtime it describes, and that is why it is derived from a program the
    //   runtime actually ran rather than from a microbenchmark.
    //
    // The estimates it separates are 4.2 us (`nn4_128` layer), 4.1 us (its
    // pack), 16.5 us (`nn4_512`/`matmul512` pack), 21 us (`conv2d_512`), 34 us
    // (`nn4_256` layer) and 84 us (`conv2d_1024`) — so the boundary sits in a
    // clear gap between 4.2 and 16.5, and any value in it answers every measured
    // cell the same way.
    dispatch: Some(DispatchCost {
        flops_per_ns: 1000,
        bytes_per_ns: 127,
        dispatch_ns: 8_300,
    }),
    ..APPLE_M
};

const PROFILES: [&TargetProfile; 13] = [
    &GENERIC,
    &APPLE_M,
    &ZEN3,
    &CUDA_ADA,
    &RTX_4090,
    &H100,
    &H100_PCIE,
    &RTX_5090,
    &RTX_3090,
    &A100,
    &APPLE_M4_SME,
    &RAPTORLAKE,
    &RAPTORLAKE_INLINE,
];

/// Resolve a profile by name. `None` for an unknown name — never a silent
/// fallback to `generic`, because a typo that quietly emits the default
/// profile's numbers is the exact failure this table exists to remove.
///
/// **A hand-written profile always wins.** The table is checked first, so
/// `--target=apple-m4-sme` overrides detection; `native` is only ever reached by
/// asking for it by name. That ordering is the contract: detection removes the
/// need to hand-write a profile for the machine you are sitting on, and changes
/// nothing about describing a machine you are not.
pub fn resolve(name: &str) -> Option<&'static TargetProfile> {
    PROFILES
        .into_iter()
        .find(|p| p.name == name)
        .or_else(|| (name == "native").then(native).flatten())
        // S46: `native` is the DEFAULT target, so it must always resolve. When a
        // fact cannot be read — an unknown OS, a locked-down container, a CPU
        // whose caches sysfs does not describe — fall back to `generic` rather
        // than fail the build. `generic` carries no cache geometry, so every
        // rung that needs it declines: the compiler goes conservative instead of
        // guessing, which is the same contract as before this became default.
        .or_else(|| (name == "native").then_some(&GENERIC))
}

/// The known profile names, for the error message on an unknown one.
pub fn names() -> String {
    PROFILES
        .iter()
        .map(|p| p.name)
        .chain(std::iter::once("native"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One `sysctl` integer, or `None` if the key does not exist on this host.
///
/// Shelling out rather than linking `libc`: this crate has two dependencies and
/// the emitter is a build-time tool that reads four keys once per run. A binding
/// for `sysctlbyname` would be a new dependency to save a process spawn that
/// happens at most once.
#[cfg(target_os = "macos")]
fn sysctl_u64(key: &str) -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// The host's own facts, **detected rather than written down**.
///
/// Reached as `--target=native`. Named profiles are resolved first (see
/// [`resolve`]), so this never shadows a hand-written one.
///
/// **Every machine fact is detected; nothing about the part is written down.**
///
/// | fact | source | |
/// | --- | --- | --- |
/// | `l2_bytes` | `hw.perflevel0.l2cachesize` | 16 MB here |
/// | `l1d_bytes` | `hw.perflevel0.l1dcachesize` | 128 KB here |
/// | `svl_bytes` | `hw.optional.arm.sme_max_svl_b` | 64 here |
/// | SME present | `hw.optional.arm.FEAT_SME` | 1 here |
/// | tile count | **derived** — ISA rule, see [`TargetProfile::sme_block`] | |
///
/// **`perflevel0`, not the bare keys.** `hw.l1dcachesize` reports **65536** on
/// this part and `hw.perflevel0.l1dcachesize` reports **131072**: the bare key
/// describes the E-cores. Sizing a P-core matrix panel against an E-core cache
/// would halve the derived k-panel and silently cost throughput, which is the
/// kind of wrong number that never announces itself.
///
/// SVL turned out to be a plain sysctl (`sme_max_svl_b`), so no streaming-mode
/// probe and no `+sme` build of this emitter is needed — a detected profile is a
/// **complete** profile, matrix unit included. `FEAT_SME` gates the block, so a
/// part without the unit yields `sme: None` and the rung falls back exactly as
/// it does for every pre-S41 profile.
///
/// `vec_bytes`/`vec_regs` come from [`vec_geometry`]. They ARE ISA facts rather
/// than sysfs entries, but `native` is now the DEFAULT target (S46), so leaving
/// them at [`GENERIC`]'s NEON values silently tiled an AVX2 part to a 16-byte
/// vector — measured at +8.8% on conv2d before this was read. `acc_vecs_per_row`/`nc_tiles` are
/// policy ratios, honestly search space (ADR-0034), not facts to read.
/// The host's vector geometry, read at runtime.
///
/// `native` means "this machine", and on x86 the vector width is not one value:
/// an AVX-512 part, an AVX2 part and an SSE-only part want 64, 32 and 16 bytes
/// with 32, 16 and 16 architectural registers. Detecting the caches but assuming
/// the ISA is how a default build tiled an AVX2 box to NEON's 16 bytes.
/// aarch64 keeps `GENERIC`'s 16/32, which is NEON and correct; SVE would be a
/// separate rung, not a wider `vec_bytes`.
fn vec_geometry() -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return (64, 32);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return (32, 16);
        }
        return (16, 16);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (GENERIC.vec_bytes, GENERIC.vec_regs)
    }
}

pub fn native() -> Option<&'static TargetProfile> {
    static NATIVE: std::sync::OnceLock<Option<TargetProfile>> = std::sync::OnceLock::new();
    NATIVE.get_or_init(detect).as_ref()
}

#[cfg(target_os = "macos")]
fn detect() -> Option<TargetProfile> {
    // perflevel0 is the performance core cluster; the bare hw.* keys are the
    // efficiency cores. See the doc comment — this distinction is load-bearing.
    let l1d_bytes = sysctl_u64("hw.perflevel0.l1dcachesize")?;
    let line = sysctl_u64("hw.cachelinesize")?;
    Some(TargetProfile {
        name: "native",
        l2_bytes: sysctl_u64("hw.perflevel0.l2cachesize")?,
        // 5 on this M4 Pro: the 16 MB is a CLUSTER L2, not a per-core one, and
        // `move_block`'s cost term is wrong by 4.8x without this.
        l2_cores: sysctl_u64("hw.perflevel0.cpusperl2")?.max(1),
        sme: (sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1))
            .then(|| {
                Some(Sme {
                    svl_bytes: sysctl_u64("hw.optional.arm.sme_max_svl_b")?,
                    l1d_bytes,
                    // Detected profiles inherit the swept ratio; it is policy, not
                    // a property of the silicon, so there is nothing to read.
                    panel_l1d_ratio: 2,
                })
            })
            .flatten(),
        // S45: line and page are plain sysctls; sets and ways are derived from
        // them, so a detected profile is a COMPLETE one for this rung too.
        // Line is a plain sysctl; SETS is the one fact macOS does not expose,
        // so it comes from the page bound — see `L1d` for why that is a checked
        // heuristic and what it gets wrong. On this part it yields 128, which is
        // the value S44 verified by a stride probe.
        l1d: Some(L1d {
            bytes: l1d_bytes,
            line_bytes: line,
            sets: (sysctl_u64("hw.pagesize")? / line).max(1),
        }),
        vec_bytes: vec_geometry().0,
        vec_regs: vec_geometry().1,
        // `native` is the DEFAULT target, so the dispatch economics have to
        // reach it or the rule only fires under an explicit
        // `--target=apple-m4-sme`. Inherited ONLY when the detected part is the
        // one they were measured on — SME with a 64-byte SVL — and `None`
        // otherwise, because a threshold is a machine fact and guessing it for
        // an unmeasured part is what this design refuses.
        dispatch: match sysctl_u64("hw.optional.arm.sme_max_svl_b") {
            Some(64) if sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1) => APPLE_M4_SME.dispatch,
            _ => None,
        },
        ..GENERIC
    })
}

/// One `sysfs` file parsed as a `u64`, tolerating the `48K` / `2048K` spelling
/// the cache nodes use for sizes.
#[cfg(target_os = "linux")]
fn sysfs_u64(path: &str) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let t = raw.trim();
    match t.strip_suffix('K') {
        Some(k) => Some(k.parse::<u64>().ok()? * 1024),
        None => t.parse().ok(),
    }
}

/// The same facts on Linux — a different parser, not a different idea.
///
/// `index0` is the L1 data cache and `index2` the unified L2 on every part this
/// runs on; `shared_cpu_list` is what makes `l2_cores` a measurement rather than
/// an assumption (`0-1` on the i9 is two SMT threads of ONE physical core, so
/// the answer is 1 core, not 2). SMT siblings are collapsed via
/// `topology/thread_siblings_list`, because a hyperthread is not a core to
/// budget for.
///
/// No SME on x86, so `sme` stays `None` and the SME realization cannot be
/// selected — the same seam as every pre-S41 profile.
#[cfg(target_os = "linux")]
fn detect() -> Option<TargetProfile> {
    let l1 = "/sys/devices/system/cpu/cpu0/cache/index0";
    let l2 = "/sys/devices/system/cpu/cpu0/cache/index2";
    let count = |path: &str| -> u64 {
        // "0-1" -> 2, "0,4" -> 2, "0" -> 1.
        std::fs::read_to_string(path)
            .ok()
            .map(|s| {
                s.trim()
                    .split(',')
                    .map(|part| match part.split_once('-') {
                        Some((a, b)) => match (a.parse::<u64>(), b.parse::<u64>()) {
                            (Ok(a), Ok(b)) if b >= a => b - a + 1,
                            _ => 1,
                        },
                        None => 1,
                    })
                    .sum()
            })
            .unwrap_or(1)
            .max(1)
    };
    let smt = count("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    Some(TargetProfile {
        name: "native",
        l2_bytes: sysfs_u64(&format!("{l2}/size"))?,
        l2_cores: (count(&format!("{l2}/shared_cpu_list")) / smt).max(1),
        // Linux EXPOSES the set count, so it is read rather than derived —
        // the macOS page heuristic is a fallback for a host that hides it, and
        // must never shadow a readable truth (`L1d`).
        l1d: Some(L1d {
            bytes: sysfs_u64(&format!("{l1}/size"))?,
            line_bytes: sysfs_u64(&format!("{l1}/coherency_line_size"))?,
            sets: sysfs_u64(&format!("{l1}/number_of_sets"))?,
        }),
        vec_bytes: vec_geometry().0,
        vec_regs: vec_geometry().1,
        ..GENERIC
    })
}

/// No detection on other hosts. `None` means `--target=native` errors with the
/// known-name list instead of guessing.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect() -> Option<TargetProfile> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: Ty = Ty::Float { bits: 32 };
    const F64: Ty = Ty::Float { bits: 64 };

    /// Rule 1, at the arithmetic level: the default profile must reproduce every
    /// literal the emitter used to carry. If this fails, emission has moved.
    #[test]
    fn generic_reproduces_the_six_literals() {
        assert_eq!(GENERIC.tile_j(&F32), 16, "tile_j_for f32");
        assert_eq!(GENERIC.tile_j(&F64), 8, "tile_j_for f64");
        assert_eq!(GENERIC.tile_i(), 4, "TILE_I");
        assert_eq!(GENERIC.nc(&F32), 512, "tile_nc_for f32");
        assert_eq!(GENERIC.nc(&F64), 256, "tile_nc_for f64");
        assert_eq!(GENERIC.tile_kc(&F32), 128, "TILE_KC f32");
        assert_eq!(GENERIC.tile_kc(&F64), 128, "TILE_KC f64");
        assert_eq!(GENERIC.heap_min_bytes, 256 * 1024, "HEAP_MIN_BYTES");
    }

    /// The S26 sweep's verdict, as arithmetic: TI=4 fits in half the file and
    /// TI=8 does not. The recorded failure was "8 spills: 128 accumulators ≫ 32
    /// NEON regs" — 8 rows × 16 f32 lanes ÷ 4 lanes per register = 32 registers,
    /// the whole file with nothing left for operands.
    #[test]
    fn tile_i_is_the_register_budget_not_a_taste() {
        let regs_per_row = GENERIC.tile_j(&F32) / GENERIC.lanes(&F32);
        assert_eq!(regs_per_row, GENERIC.acc_vecs_per_row);
        assert_eq!(GENERIC.tile_i() * regs_per_row, GENERIC.vec_regs / 2);
        assert_eq!(8 * regs_per_row, GENERIC.vec_regs, "TI=8 is the whole file");
    }

    /// The measured S29/S30 verdict, reproduced as a deduction — **and its
    /// boundary, pinned honestly**. On a 16 MB L2 the k-panel is 4096, deeper
    /// than every K we run, so the gate `site.k > tile_kc` never opens. Above
    /// 4096 it DOES reopen: this is a threshold, not an off-switch, and the
    /// 8192 assertion is here so nobody reads the others as "apple-m disables
    /// the nest". See `tile_kc`'s note — past the threshold the derivation and
    /// the measurement disagree.
    #[test]
    fn apple_m_raises_the_kc_threshold_above_every_k_we_run() {
        assert_eq!(APPLE_M.tile_kc(&F32), 4096);
        for k in [1024_u64, 2048, 4096] {
            assert!(k <= APPLE_M.tile_kc(&F32), "KC nest must stay off at K={k}");
        }
        assert!(
            8192 > APPLE_M.tile_kc(&F32),
            "the gate REOPENS past 4096 — an unmeasured regime, not a disabled rung"
        );
        assert!(
            2048 > GENERIC.tile_kc(&F32),
            "generic keeps today's gate open"
        );
    }

    /// The KC a-panel is `tile_i x tile_kc x sizeof` = `2 * l2_bytes/nc_tiles`,
    /// so it scales with the profile's L2: 2 KB under `generic`, 64 KB under
    /// `apple-m`. Pinned because `func/mod.rs`'s heap note used to claim a flat
    /// 2 KB, and that number is a stack-sizing claim.
    #[test]
    fn kc_apanel_scales_with_l2() {
        let apack = |p: &TargetProfile, e: &Ty| p.tile_i() * p.tile_kc(e) * 4;
        assert_eq!(apack(&GENERIC, &F32), 2 * 1024);
        assert_eq!(apack(&APPLE_M, &F32), 64 * 1024);
        assert!(
            apack(&APPLE_M, &F32) < GENERIC.heap_min_bytes,
            "still on the stack, but the headroom is profile-dependent"
        );
    }

    #[test]
    fn zen3_moves_two_constants_at_once() {
        assert_eq!(ZEN3.tile_j(&F32), 32);
        assert_eq!(ZEN3.tile_i(), 2);
    }

    #[test]
    fn unknown_name_is_not_silently_generic() {
        assert_eq!(resolve("generic"), Some(&GENERIC));
        assert_eq!(resolve("apple-m"), Some(&APPLE_M));
        assert_eq!(resolve("cuda-ada"), Some(&CUDA_ADA));
        assert_eq!(resolve("apple-m4-sme"), Some(&APPLE_M4_SME));
        assert!(resolve("apple_m").is_none());
        assert!(
            names().contains("native"),
            "the unknown-name error must offer `native`, or detection is undiscoverable"
        );
    }

    /// The SME k-panel depth is **derived from L1D**, and the derivation
    /// reproduces the swept optimum — `plan-s31-deduced-blocking.md`'s rule.
    /// `512` appears in no source file; it falls out of 128 KB / 256 B.
    #[test]
    /// plan-s51 H2 — the A pack's block depth, and the two ways it declines.
    ///
    /// The pack's stores land one cache line apart, so a line is filled 4 bytes
    /// at a time across the `panel_rows` rows that revisit it. Free while `ap`
    /// is resident, 32x the line traffic once it is not — so the decision is
    /// exactly "does `32 · k · sizeof` fit L1D", and 1024 is the last `k` that
    /// does on this part.
    ///
    /// The depth is `line_bytes / sizeof` — one line's worth of `k` — and both
    /// numbers are DETECTED, so nothing here is swept.
    #[test]
    fn pack_block_fires_only_once_the_pack_buffer_leaves_l1d() {
        // 32 x 1024 x 4 = 131072 = L1D exactly. Fits, so no block.
        assert_eq!(APPLE_M4_SME.pack_block(32, 1024, &F32), None);
        // One k-block deeper and it does not fit.
        assert_eq!(APPLE_M4_SME.pack_block(32, 1056, &F32), Some(32));
        assert_eq!(APPLE_M4_SME.pack_block(32, 4096, &F32), Some(32));
        // f64 halves the elements per line, so the block halves with it.
        assert_eq!(APPLE_M4_SME.pack_block(32, 4096, &F64), Some(16));
        // A profile with no L1D geometry cannot decide and must keep the
        // single-pass copy — the same contract every other derived factor has.
        assert_eq!(GENERIC.pack_block(32, 4096, &F32), None);
    }

    fn sme_kc_reproduces_the_measured_optimum() {
        assert_eq!(
            APPLE_M4_SME.sme_kc(&F32),
            Some(1024),
            "2 x 128 KB / ((ti+tj)·t·4 = 256 B) = 1024 — the depth SWEPT in the real \
             emitter at N=4096 (1.064x, disjoint) and N=2048. Ratio 1 gives 512, \
             which measured 0.785x — a loss"
        );
        // f64 scales off the same recorded fact by element width: t halves to 8,
        // the tile count doubles to 8 so the block is 2x4, and the window is
        // (2+4)·8·8 = 384 B per k. No second constant anywhere.
        assert_eq!(APPLE_M4_SME.sme_kc(&F64), Some(2 * 128 * 1024 / 384));
        // No matrix unit => no answer, never a wrong one. Same seam as `sme()`.
        assert_eq!(GENERIC.sme_kc(&F32), None);
        assert_eq!(APPLE_M.sme_kc(&F32), None);
    }

    /// The SME budget is L1D and **not** the NEON rung's L2, and the two answers
    /// differ by 8x — which is the whole reason `sme_kc` exists rather than
    /// reusing `tile_kc`. If these two ever coincide, one of them is wrong.
    #[test]
    fn the_sme_budget_is_not_the_neon_one() {
        let neon = APPLE_M4_SME.tile_kc(&F32);
        let sme = APPLE_M4_SME.sme_kc(&F32).unwrap();
        assert_eq!(neon, 4096, "unchanged: half of 16 MB shared L2 over nc");
        assert_eq!(sme, 1024);
        assert!(
            sme < neon,
            "the matrix unit's window is smaller than the NEON k-panel's; \
             a shared derivation would size SME's panel 8x too deep and lose \
             1.448x at K=4096 (docs/performance/s42-sme-roofline.md §5)"
        );
        // And the gate this feeds: at K=4096 the SME nest must OPEN...
        assert!(4096 > sme, "KC nest must fire at K=4096");
        // ...while at K=512 it must stay shut, because blocking is a LOSS there.
        assert!(
            1024 <= sme,
            "KC nest must stay off at K=1024 — measured neutral-to-worse there"
        );
    }

    /// Detection reads the keys it documents. Not a value assertion — the point
    /// is that the plumbing works on this host, so "automatic" is not a claim
    /// resting on an untested code path.
    #[cfg(target_os = "macos")]
    #[test]
    fn native_reads_the_facts_it_claims_to() {
        let l2 = sysctl_u64("hw.perflevel0.l2cachesize");
        let l1d = sysctl_u64("hw.perflevel0.l1dcachesize");
        assert!(l2.is_some_and(|v| v >= 256 * 1024), "L2 detected: {l2:?}");
        assert!(l1d.is_some_and(|v| v >= 32 * 1024), "L1D detected: {l1d:?}");
        assert!(sysctl_u64("hw.optional.arm.no.such.key").is_none());

        let native = native().expect("detection must succeed on macOS");
        assert_eq!(native.name, "native");
        assert_eq!(Some(native.l2_bytes), l2, "detected, not inherited");

        // The E-core trap: the bare key is a DIFFERENT, smaller cache. If these
        // two ever agree, this host stopped being big.LITTLE and the guard below
        // stops proving anything — but on any such part it is also harmless.
        if let (Some(bare), Some(p0)) = (sysctl_u64("hw.l1dcachesize"), l1d) {
            assert!(
                bare <= p0,
                "hw.l1dcachesize ({bare}) must not exceed the P-core's ({p0}); \
                 detection reads perflevel0 precisely because the bare key is \
                 the efficiency cluster"
            );
        }
    }

    /// One `sysctl` string — the sibling of `sysctl_u64` for the one key whose
    /// value is text. The part gates below read the brand string, because
    /// ADR-0038 D4 gates *exactness* on the part, loudly: an exact check that
    /// can only run on the documented machine must say so in its name, and must
    /// say out loud when it did not run.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn sysctl_string(key: &str) -> Option<String> {
        let out = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", key])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    }

    /// **ADR-0038 D4, the machine-independent half of the genericity gate.**
    /// Detection must succeed, the matrix unit must be recorded iff the part
    /// reports one, and every derived SME quantity must be a pure function of
    /// the recorded facts: the rebuild below freezes every input the
    /// derivations may not read at the hand-written profile's value, so a
    /// derivation that grows a hidden host read diverges here. This runs on
    /// ANY Apple-silicon Mac.
    ///
    /// What it does NOT assert is agreement with the hand-written numbers —
    /// that held on a base M4 only because SVL, the P-core L1D and the 16 MB
    /// cluster L2 happen to be constant across the M4 family, and a fact that
    /// happens to hold on every runner is luck, not a gate. Exactness is the
    /// part-gated test below.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_sme_profile_is_consistent() {
        let native = native().expect("detection must succeed on macOS");
        assert_eq!(
            native.sme().is_some(),
            sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1),
            "the unit is recorded iff the part reports it — an absent key reads \
             as None, not 0, so pre-SME parts land here cleanly rather than \
             failing a Some(0) assertion"
        );
        let Some(nsme) = native.sme() else {
            // No matrix unit on this part: nothing to compare, and the fallback
            // is the pre-S41 behaviour. Not a failure.
            return;
        };
        let l1d = native.l1d.expect("macOS detection records L1D geometry");
        assert_eq!(
            nsme.l1d_bytes, l1d.bytes,
            "both rungs read the same P-core L1D (`l1d_and_sme_agree_on_the_l1d`, \
             for the detected profile)"
        );
        let rebuilt = TargetProfile {
            sme: native.sme,
            ..APPLE_M4_SME
        };
        for elem in [F32, F64] {
            assert_eq!(native.sme_tile_side(&elem), rebuilt.sme_tile_side(&elem));
            assert_eq!(native.sme_block(&elem), rebuilt.sme_block(&elem));
            assert_eq!(native.sme_kc(&elem), rebuilt.sme_kc(&elem));
        }
    }

    /// **ADR-0038 D4, the exact half — gated on the documented part, loudly.**
    /// On the documented M4 Pro, detection must reproduce the hand-written
    /// profile *exactly* — same SVL, same L1D, same derived tile side, same
    /// derived block, same derived k-panel. If it does not, then one of the
    /// two is wrong and the hand-written numbers were never facts about this
    /// machine. On any other part the exact check has nothing to say, and the
    /// skip names the part it skipped.
    ///
    /// This is what makes `apple-m4-sme` a **cross-compilation** spelling rather
    /// than the only way to describe the machine under the keyboard: the numbers
    /// in it are no longer load-bearing here, they are checked against the part.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts() {
        let part = sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "<brand string unreadable>".to_owned());
        if !part.contains("M4 Pro") {
            eprintln!(
                "native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts: \
                 SKIP — the exact facts are the documented M4 Pro's; this part is `{part}`"
            );
            return;
        }
        let native = native().expect("detection must succeed on macOS");
        let nsme = native.sme().expect("the documented M4 Pro has SME");
        let hand = APPLE_M4_SME.sme().expect("hand-written profile has SME");
        assert_eq!(nsme.svl_bytes, hand.svl_bytes, "SVL: detected vs written");
        assert_eq!(nsme.l1d_bytes, hand.l1d_bytes, "L1D: detected vs written");
        assert_eq!(native.l2_bytes, APPLE_M4_SME.l2_bytes, "L2");
        for elem in [F32, F64] {
            assert_eq!(
                native.sme_tile_side(&elem),
                APPLE_M4_SME.sme_tile_side(&elem)
            );
            assert_eq!(native.sme_block(&elem), APPLE_M4_SME.sme_block(&elem));
            assert_eq!(native.sme_kc(&elem), APPLE_M4_SME.sme_kc(&elem));
        }
        // And the number the whole S42 campaign turns on, arrived at with no
        // constant written anywhere: 2*131072 / ((2+2)*16*4 = 256) = 1024.
        assert_eq!(native.sme_kc(&F32), Some(1024));
    }

    /// One `MoveSite` for a square `side x side` f32 transpose — the shape every
    /// measured point in `move_block`'s table is.
    fn transpose(side: u64) -> MoveSite {
        MoveSite {
            width: side,
            rows: side,
            cq: 1,
            cr: side,
            elem: F32,
            len: side * side,
        }
    }

    /// **The whole S45 decision, scored against every measured point on two
    /// machines.** Each row is a measurement, not a preference; the comment
    /// beside it is what the emitted pipeline actually did.
    #[test]
    fn move_block_reproduces_both_machines() {
        // M4 Pro: fires at 1024 with the block S44 SWEPT to (16), fires at 2048,
        // declines at 128 on pressure and at 512 on cost.
        // BOTH M4 sides land on the block the machine measured fastest — 1024
        // has floor 32 / ceiling 16 and 2048 has floor 32 / ceiling 8, and the
        // geometric mean is 22 and 16, so the largest divisor is 16 either way.
        assert_eq!(APPLE_M4_SME.move_block(&transpose(1024)), Some(16));
        assert_eq!(APPLE_M4_SME.move_block(&transpose(2048)), Some(16));
        assert_eq!(
            APPLE_M4_SME.move_block(&transpose(128)),
            None,
            "pressure 0.5: 128 lines live into 32 sets x 8 ways = 256 slots — measured 1.000x"
        );
        assert_eq!(
            APPLE_M4_SME.move_block(&transpose(512)),
            None,
            "pressure 8 but the 1 MB read array sits inside 16 MB / 5 = 3.36 MB — measured \
             overlapping, 1.02x min-to-min"
        );
        // i9: fires at 1024 and 2048, and DECLINES at 512 — the discriminating
        // case, where pressure alone would fire and the machine measured 0.901x.
        // The i9's 12 reachable slots put the ceiling at 6 against a floor of
        // 16, so the mean is 9 and the block is 8. Its measured optimum is 128,
        // which no L1-derived rule can reach — see `move_block`'s note and the
        // counter evidence in benches/results-s45.
        assert_eq!(RAPTORLAKE.move_block(&transpose(1024)), Some(8));
        assert_eq!(RAPTORLAKE.move_block(&transpose(2048)), Some(8));
        assert_eq!(
            RAPTORLAKE.move_block(&transpose(512)),
            None,
            "pressure 21.3 — ABOVE the M4's 8 at the same side — but the 1 MB read array fits \
             the 2 MB private L2, and it measured a 0.901x LOSS (replicated 0.907x). This is \
             the case pressure alone gets wrong, and the reason the cost term exists"
        );
        // Odd sides never collapse the set index, so there is nothing to win:
        // S44 measured side 1025 running 2.12x faster UNBLOCKED than 1024, and
        // blocking it at 0.623x. Note this one scores pressure **1.0009** — it
        // is the witness for the conflict clause, not the pressure clause, and
        // without that clause the deduction would fire on a measured LOSS.
        assert_eq!(APPLE_M4_SME.move_block(&transpose(1025)), None);
        // S47: width 1536 passes EVERY term above on both machines — the set
        // index collapses (8 sets on the M4, 2 on the i9), pressure is 24x and
        // 64x, and the 9.44 MB read array is past both private-L2 shares — and
        // firing it measured a **1.75x LOSS** on the M4 (1t; 1.41x threaded).
        // A width that is not a power of two cannot give both `b` and `w/b` a
        // shift lowering, so the permutation costs +20-25% instructions there
        // and the benefit is unpredictable. This is the only measured cell where
        // the rung was a net loss, and it is the only one this clause changes.
        assert_eq!(
            APPLE_M4_SME.move_block(&transpose(1536)),
            None,
            "measured 1.75x SLOWER than unblocked at 1 thread, 1.41x threaded"
        );
        assert_eq!(
            RAPTORLAKE.move_block(&transpose(1536)),
            None,
            "forgoes a 1.060x/1.044x win to avoid the M4's 1.75x loss at the same width"
        );
        // Powers of two are untouched by that clause — the four cells the rung
        // actually earns its keep on are all above.
        for side in [1024, 2048] {
            assert!(APPLE_M4_SME.move_block(&transpose(side)).is_some());
            assert!(RAPTORLAKE.move_block(&transpose(side)).is_some());
        }
        // And a profile with no L1D geometry cannot answer at all.
        for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
            assert_eq!(p.move_block(&transpose(1024)), None, "{}", p.name);
        }
    }

    /// The pressure term is `lines_live > slots` — the literal statement that the
    /// sweep needs more lines than it can reach — so the set collapse, not the
    /// power of two, is what it turns on. S44's two load-bearing nulls (side 128
    /// at pressure 0.5, side 544 at 0.53) are both collapses that stay under 1.
    #[test]
    fn pressure_counts_lines_not_powers_of_two() {
        let g = APPLE_M4_SME.l1d.expect("apple-m4-sme has L1D geometry");
        assert_eq!((g.sets, g.ways()), (128, 8));
        // side 128: stride 512 B = 4 lines, gcd(4, 128) = 4 => 32 sets, 256 slots
        // against 128 lines live. A REAL collapse (32 of 128 sets) that still
        // must not fire — "it is not 'a collapse is bad', it is pressure".
        assert_eq!(APPLE_M4_SME.move_block(&transpose(128)), None);
        // side 544: gcd(17, 128) = 1 => all 128 sets, 1024 slots, 544 live.
        assert_eq!(APPLE_M4_SME.move_block(&transpose(544)), None);
    }

    /// `ways` is arithmetic, `sets` is read where the host exposes it — and the
    /// two i9 geometries are where the truth IS exposed, so they are the check
    /// on the whole scheme. Both reproduce `ways_of_associativity` exactly.
    #[test]
    fn l1d_ways_reproduce_the_readable_truth() {
        let i9_p = RAPTORLAKE.l1d.expect("raptorlake has L1D geometry");
        assert_eq!(i9_p.ways(), 12, "sysfs: ways_of_associativity = 12");
        let i9_e = L1d {
            bytes: 32 * 1024,
            line_bytes: 64,
            sets: 64,
        };
        assert_eq!(i9_e.ways(), 8, "sysfs: the E-core is 8-way");
        // The M4's set count is the one macOS hides; the page bound gives 128,
        // which is the value S44 verified with a stride probe, and 8 ways falls
        // out of capacity.
        let m4 = APPLE_M4_SME.l1d.expect("apple-m4-sme has L1D geometry");
        assert_eq!((16384 / m4.line_bytes, m4.sets), (128, 128));
        assert_eq!(m4.ways(), 8);
        // Whatever the split, capacity is exact — the bound on being wrong.
        for g in [i9_p, i9_e, m4] {
            assert_eq!(g.sets * g.ways(), g.bytes / g.line_bytes);
        }
    }

    /// The L2 sharing fact, which is the cost term's divisor and the field this
    /// session added because the old doc comment said "per-core" while `apple-m`
    /// recorded a SHARED cluster cache. Getting it wrong by 5x is the difference
    /// between firing and declining at M4 side 1024.
    #[test]
    fn l2_per_core_is_not_l2_bytes() {
        assert_eq!(APPLE_M.l2_per_core(), 16 * 1024 * 1024 / 5);
        assert!(
            4 * 1024 * 1024 > APPLE_M.l2_per_core(),
            "a 4 MB read array must OVERFLOW the M4's share (it measured 1.578x)"
        );
        assert!(
            4 * 1024 * 1024 < APPLE_M.l2_bytes,
            "...while fitting the cluster L2 whole, which is why the distinction is the answer"
        );
        assert_eq!(
            RAPTORLAKE.l2_per_core(),
            2 * 1024 * 1024,
            "private per P-core"
        );
        assert_eq!(GENERIC.l2_per_core(), GENERIC.l2_bytes, "l2_cores = 1");
    }

    /// The duplicated L1D size cannot drift: `Sme::l1d_bytes` and `L1d::bytes`
    /// are the same machine fact reached by two rungs. See `TargetProfile::l1d`
    /// for why the merge was not done here.
    #[test]
    fn l1d_and_sme_agree_on_the_l1d() {
        for p in PROFILES {
            if let (Some(sme), Some(l1d)) = (p.sme(), p.l1d) {
                assert_eq!(sme.l1d_bytes, l1d.bytes, "{} disagrees with itself", p.name);
            }
        }
    }

    /// **ADR-0038 D4, the machine-independent half.** `move_block`'s verdict
    /// must be a pure function of the recorded facts: the rebuild below copies
    /// everything the derivation may legitimately read (`l1d`, `l2_bytes`,
    /// `l2_cores`) and freezes every other field at the hand-written profile's
    /// value, so a verdict that grows a hidden host read — or starts consulting
    /// a field it has no business with — diverges. This runs on ANY
    /// Apple-silicon Mac, including a base M4 whose `cpusperl2` is 3 where the
    /// documented part's is 5.
    ///
    /// No exact fire/decline verdict is asserted here: `l1d: Some` and
    /// `l2_cores >= 1` do not imply one. Whether side 1024 fires turns on the
    /// 4 MB read array against the per-core L2 share — a fact of the part — so
    /// exactness belongs to the part-gated test below.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_move_geometry_is_consistent() {
        let native = native().expect("detection must succeed on macOS");
        assert!(
            native.l1d.is_some(),
            "macOS detection always records L1D geometry"
        );
        assert!(native.l2_cores >= 1, "the sharing factor is clamped at 1");
        let rebuilt = TargetProfile {
            l1d: native.l1d,
            l2_cores: native.l2_cores,
            l2_bytes: native.l2_bytes,
            ..APPLE_M4_SME
        };
        for side in [128_u64, 512, 1024, 1025, 2048] {
            assert_eq!(
                native.move_block(&transpose(side)),
                rebuilt.move_block(&transpose(side)),
                "side {side}: the verdict must be a pure function of the recorded facts"
            );
        }
    }

    /// **ADR-0038 D4, the exact half — gated on the documented part, loudly.**
    /// The hand-written geometry is the M4 Pro's: FIVE P-cores sharing the
    /// 16 MB cluster L2. A base M4 shares it 3 ways, which flips the side-1024
    /// verdict (the 4 MB read array overflows the 3.36 MB per-core share but
    /// fits the 5.46 MB one) — that flip is the hosted-CI failure this split
    /// repairs, not a bug in either profile. Exactness therefore runs only on
    /// the part it describes, and the skip names the part it skipped.
    ///
    /// On the documented M4 Pro, detection must reproduce the hand-written L1D
    /// geometry and L2 sharing *exactly*, and therefore reach the same
    /// fire/decline verdict and the same block. If it does not, one of the two
    /// is wrong and the hand-written numbers were never facts about this
    /// machine.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_on_the_documented_m4_pro_reproduces_hand_written() {
        let part = sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "<brand string unreadable>".to_owned());
        if !part.contains("M4 Pro") {
            eprintln!(
                "native_on_the_documented_m4_pro_reproduces_hand_written: SKIP — \
                 the exact geometry is the documented M4 Pro's; this part is `{part}`"
            );
            return;
        }
        let native = native().expect("detection must succeed on macOS");
        assert_eq!(
            native.l1d, APPLE_M4_SME.l1d,
            "detected L1D geometry vs written"
        );
        assert_eq!(native.l2_cores, APPLE_M4_SME.l2_cores, "cores sharing L2");
        assert_eq!(native.l2_per_core(), APPLE_M4_SME.l2_per_core());
        for side in [128_u64, 512, 1024, 1025, 2048] {
            assert_eq!(
                native.move_block(&transpose(side)),
                APPLE_M4_SME.move_block(&transpose(side)),
                "detected and written must decide side {side} the same way"
            );
        }
        // And the number the whole session turns on, detected rather than typed.
        assert_eq!(native.move_block(&transpose(1024)), Some(16));
    }

    /// `apple-m` gained `l2_cores` and nothing else — no derivation reads it, so
    /// its emission cannot have moved. The negative control for the profile edit.
    #[test]
    fn apple_m_gained_only_the_sharing_fact() {
        assert_eq!(
            APPLE_M,
            TargetProfile {
                l2_cores: APPLE_M.l2_cores,
                name: APPLE_M.name,
                l2_bytes: APPLE_M.l2_bytes,
                ..GENERIC
            }
        );
        assert_eq!(
            APPLE_M.tile_kc(&F32),
            4096,
            "unchanged: the KC gate is where it was"
        );
        assert!(
            APPLE_M.l1d.is_none(),
            "no L1D geometry ⇒ the move rung cannot fire"
        );
    }

    /// A hand-written profile always wins over detection (Sapir's rule).
    #[test]
    fn named_profiles_override_detection() {
        for p in PROFILES {
            assert_eq!(
                resolve(p.name).map(|r| r.name),
                Some(p.name),
                "the table is consulted before `native`"
            );
        }
        assert!(
            resolve("nativ").is_none(),
            "no fuzzy match, no silent default"
        );
        assert!(resolve("cuda_ada").is_none());
        assert!(resolve("apple-m4").is_none());
        assert!(resolve("").is_none());
    }

    /// Rule 1 for the SME leg: the capability is reachable only through
    /// `sme()`, and no profile that existed before it can answer. That is the
    /// whole byte-identity argument — the SME realization's gate cannot open
    /// without a fact only `apple-m4-sme` carries.
    #[test]
    fn sme_is_absent_from_every_pre_sme_profile() {
        for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
            assert!(p.sme().is_none(), "{} must expose no SME facts", p.name);
            assert!(
                p.sme_tile_side(&F32).is_none(),
                "{} derives no tile",
                p.name
            );
        }
    }

    /// The measured M4 Pro facts, and the one derivation the emitter reads:
    /// SVL 512 bits ⇒ a 16×16 f32 tile (`benches/sme/svl.c`, `run16.c`). f64
    /// halves it, which is why the side is derived rather than written down.
    #[test]
    fn apple_m4_sme_derives_the_measured_tile_side() {
        let sme = APPLE_M4_SME.sme().expect("apple-m4-sme has an SME unit");
        assert_eq!(sme.svl_bytes, 64, "SVL = 512 bits, detected");
        assert_eq!(APPLE_M4_SME.sme_tile_side(&F32), Some(16));
        assert_eq!(APPLE_M4_SME.sme_tile_side(&F64), Some(8));
        // The tile count is an ISA rule, not a field: exactly `elem_bytes`.
        assert_eq!(APPLE_M4_SME.sme_block(&F32), Some((2, 2)), "4 tiles at f32");
        assert_eq!(APPLE_M4_SME.sme_block(&F64), Some((2, 4)), "8 tiles at f64");
    }

    /// The SME profile moves only DECLARED-CAPABILITY fields off `apple-m`.
    /// Every derived tile factor is unchanged, so a site that falls back to the
    /// NEON rung under `apple-m4-sme` emits what `apple-m` emits — the negative
    /// control the golden test pins at the text level.
    #[test]
    fn apple_m4_sme_moves_only_declared_capability_fields() {
        // S45 adds `l1d` alongside `sme`, plan-s50 adds `dispatch`: the same
        // part, more capability declared. Each is an `Option` that is `None` on
        // `apple-m`, so a rung that needs one cannot fire without it and every
        // DERIVED tile factor below must still be `apple-m`'s.
        assert_eq!(
            APPLE_M4_SME,
            TargetProfile {
                name: APPLE_M4_SME.name,
                sme: APPLE_M4_SME.sme,
                l1d: APPLE_M4_SME.l1d,
                dispatch: APPLE_M4_SME.dispatch,
                ..APPLE_M
            }
        );
        assert!(
            APPLE_M.dispatch.is_none(),
            "an unmeasured part must carry no dispatch economics, so the \
             threshold cannot fire on a number nobody took"
        );
        assert_eq!(APPLE_M4_SME.tile_j(&F32), APPLE_M.tile_j(&F32));
        assert_eq!(APPLE_M4_SME.tile_i(), APPLE_M.tile_i());
        assert_eq!(APPLE_M4_SME.tile_kc(&F32), APPLE_M.tile_kc(&F32));
    }

    /// The accumulator budget, spent. 4 f32 tiles ⇒ a 2×2 block, which is the
    /// arrangement `benches/sme/mm4.c` measured at 1.84–2.61× over one tile.
    /// f64 has twice as many tiles (one per element byte), and the count is
    /// derived from the recorded f32 one rather than written down again.
    #[test]
    fn sme_block_is_the_square_most_split_of_the_tile_count() {
        assert_eq!(APPLE_M4_SME.sme_block(&F32), Some((2, 2)));
        assert_eq!(APPLE_M4_SME.sme_block(&F64), Some((2, 4)), "8 tiles at f64");
        for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
            assert!(p.sme_block(&F32).is_none(), "{} has no ZA", p.name);
        }
    }

    /// The derivation's two invariants: the block spends the WHOLE tile budget,
    /// and it is the square-most factorization — the one that minimises `ti + tj`
    /// operand loads per `ti · tj` outer products (1×4 needs 5 loads for the 4
    /// MACs 2×2 gets from 4).
    ///
    /// **Swept over element WIDTHS, not over a recorded count.** The previous
    /// version of this test swept `f32_tiles` 1..=64, and S41b logged the defect
    /// that created: it pinned arrangements for counts like 7 or 64 that no ZA
    /// register file has and no emitted kernel could select, so the test asserted
    /// properties of unreachable IR. Now that the count is `elem_bytes` by
    /// construction there is nothing unreachable left to sweep — every width here
    /// is a width the architecture actually has.
    #[test]
    fn sme_block_spends_every_tile_and_stays_square_most() {
        for bits in [8_u8, 16, 32, 64] {
            let elem = Ty::Float { bits };
            let tiles = u64::from(bits).div_ceil(8);
            let (ti, tj) = APPLE_M4_SME.sme_block(&elem).expect("has ZA");
            assert_eq!(
                ti * tj,
                tiles,
                "the block must spend every tile ({tiles} at f{bits})"
            );
            assert!(ti <= tj, "ti is the smaller factor (f{bits})");
            let best = (1..=tiles)
                .filter(|d| tiles.is_multiple_of(*d))
                .map(|d| d + tiles / d)
                .min()
                .expect("a divisor exists");
            assert_eq!(ti + tj, best, "fewest operand loads at f{bits}");
            // Every arrangement must actually fit ZA: `ti·t x tj·t` elements of
            // `w` bytes is exactly `svl x svl`, the whole array, never more.
            let t = APPLE_M4_SME.sme_tile_side(&elem).expect("has ZA");
            let svl = APPLE_M4_SME.sme().unwrap().svl_bytes;
            assert_eq!(
                ti * t * tj * t * tiles,
                svl * svl,
                "the block must be exactly ZA at f{bits}, not larger"
            );
        }
        assert_eq!(
            (1..=4_u64)
                .filter(|d| 4_u64.is_multiple_of(*d))
                .map(|d| d + 4 / d)
                .min(),
            Some(4),
            "2x2 is 4 loads; 1x4 is 5"
        );
    }

    /// The packed-B panel the NEON rung already builds is `tile_j` lanes wide,
    /// and the SME kernel wants an SVL-wide contiguous row. On this part they
    /// are the same 16 — which is why the SME rung consumes the existing packed
    /// buffer instead of minting its own. `func/sme.rs` re-checks this per site
    /// and falls back when it does not hold, so this is a pin on *why the reuse
    /// is free here*, not a load-bearing invariant.
    #[test]
    fn sme_panel_and_packed_b_panel_coincide_on_this_part() {
        assert_eq!(
            APPLE_M4_SME.sme_tile_side(&F32),
            Some(APPLE_M4_SME.tile_j(&F32))
        );
    }

    /// S41 step 1, rule 1: adding the machine class must not move a single CPU
    /// emission. Every profile that existed before S41 is `Cpu`, so no CPU
    /// realization can observe the new field — the type system carries the
    /// byte-identity argument, and the 159-emission A/B sweep confirms it.
    #[test]
    fn every_pre_s41_profile_is_cpu() {
        for p in [&GENERIC, &APPLE_M, &ZEN3] {
            assert_eq!(p.machine, Machine::Cpu, "{} must stay Cpu", p.name);
            assert!(p.gpu().is_none(), "{} must expose no GPU facts", p.name);
        }
    }

    /// A GPU fact is reachable only through `gpu()`, so a CPU profile on a
    /// GPU-only path is a `None` at the seam rather than a wrong constant
    /// (plan-s41 §2.2 rule 2 — capability selects realization).
    #[test]
    fn gpu_facts_are_asked_for_never_defaulted() {
        let g = CUDA_ADA.gpu().expect("cuda-ada is a GPU profile");
        assert_eq!(g.warp, 32, "cooperation quantum");
        assert_eq!(g.max_threads_per_block, 1024);
        assert_eq!(
            g.arch, "sm_89",
            "verified against the device: compute capability 8.9 (RTX 4070 Ti)"
        );
        assert!(
            g.smem_per_block <= g.smem_per_sm,
            "a block cannot opt in to more than its SM holds"
        );
    }

    /// The GPU tile geometry is arithmetic over device facts — these pins are
    /// the derivation worked by hand, so a changed rule shows up as a changed
    /// number here rather than as a slower kernel on a box.
    #[test]
    fn gpu_tile_geometry_is_derived_from_device_facts() {
        let f32 = Ty::f32();
        // `fill` on and off: `warp-fill` is a no-op on these 4096 sites.
        let g = |p: &TargetProfile, n, fill| {
            let x = p
                .gpu_tile(n, n, n, &f32, false, fill)
                .expect("divisible square site");
            (x.bm(), x.bn(), x.threads(), x.bk, x.blocks_per_sm)
        };
        for fill in [false, true] {
            // 255 registers ⇒ 8×8 accumulators; 65 536 / 128 / 2 ⇒ 256 threads;
            // 4090 4096²: 1024 blocks ≥ 128 SMs, 2 per SM ⇒ 49 KB ⇒ bk 32.
            assert_eq!(g(&RTX_4090, 4096, fill), (128, 128, 256, 32, 2));
            // H100's 228 KB per SM doubles the panel depth.
            assert_eq!(g(&H100, 4096, fill), (128, 128, 256, 64, 2));
        }
        // OFF arm, 1024²: 64 blocks < 128 SMs ⇒ halve x once ⇒ 128 blocks.
        assert_eq!(g(&RTX_4090, 1024, false), (128, 64, 128, 32, 4));
        // 132 SMs: 128 blocks is still short ⇒ halve y too.
        assert_eq!(g(&H100, 1024, false), (64, 64, 64, 32, 8));
        // A CPU profile has no GPU geometry, and an indivisible site has none.
        assert_eq!(GENERIC.gpu_tile(4096, 4096, 4096, &f32, true, true), None);
        assert_eq!(RTX_4090.gpu_tile(1000, 1000, 1000, &f32, true, true), None);
        // 5090: 24 resident blocks per SM (the box's device query, S53), which
        // the thread and register budgets never reach here: 1024² halves to
        // 64-thread blocks ⇒ 256 blocks ≥ 170 SMs, 8 per SM by registers ⇒
        // 11.8 KB each ⇒ bk 16.
        assert_eq!(RTX_5090.gpu().map(|g| g.max_blocks_per_sm), Some(24));
        assert_eq!(g(&RTX_5090, 4096, false), (128, 128, 256, 32, 2));
        assert_eq!(g(&RTX_5090, 1024, false), (64, 64, 64, 16, 8));

        // S53 `wide-fma`: 8×16 holds 128 + 2·24 + 32 = 208 ≤ 255 (16×16 would
        // be 352); 65 536 / 208 / 2 = 157 ⇒ 128 threads; ty·8 = tx·16 ⇒ 16 × 8
        // ⇒ a 128 × 128 block; 315 / 128 ⇒ 2 blocks per SM — on every part.
        let w = |p: &TargetProfile, n| {
            let x = p
                .gpu_tile(n, n, n, &f32, true, false)
                .expect("divisible square site");
            (
                x.tm,
                x.tn,
                x.bm(),
                x.bn(),
                x.threads(),
                x.bk,
                x.blocks_per_sm,
            )
        };
        assert_eq!(fma_held_regs(8, 16, 4), 208);
        assert_eq!(w(&RTX_4090, 4096), (8, 16, 128, 128, 128, 32, 2));
        assert_eq!(w(&H100_PCIE, 4096), (8, 16, 128, 128, 128, 64, 2));
        assert_eq!(w(&H100, 4096), (8, 16, 128, 128, 128, 64, 2));
        assert_eq!(w(&RTX_5090, 4096), (8, 16, 128, 128, 128, 32, 2));
        // A small site keeps the S52 tile, and the fill rule still halves it:
        // 1024² is 64 wide blocks < every part's SMs.
        for p in [&RTX_4090, &H100_PCIE, &H100, &RTX_5090] {
            assert_eq!(
                p.gpu_tile(1024, 1024, 1024, &f32, true, false),
                p.gpu_tile(1024, 1024, 1024, &f32, false, false),
                "{}",
                p.name
            );
        }
        assert_eq!(w(&RTX_4090, 1024), (8, 8, 128, 64, 128, 32, 4));
        // 2048² is 256 wide blocks ≥ 170 SMs (5090): wide, unhalved.
        assert_eq!(w(&RTX_5090, 2048), (8, 16, 128, 128, 128, 32, 2));
        // f64 counts two registers per element: 8 × 8 (224).
        let d = RTX_4090
            .gpu_tile(4096, 4096, 4096, &Ty::f64(), true, false)
            .expect("f64");
        assert_eq!((d.tm, d.tn, fma_held_regs(8, 8, 8)), (8, 8, 224));
        // With `warp-fill`, the 5090's 2048² wide grid (1024 warps < 1360) is
        // under-filled too: the S52 tile, then filled.
        assert_eq!(
            RTX_5090.gpu_tile(2048, 2048, 2048, &f32, true, true),
            RTX_5090.gpu_tile(2048, 2048, 2048, &f32, false, true)
        );
        // The 4090's (1024 warps = 1024) is not.
        assert_eq!(
            RTX_4090.gpu_tile(2048, 2048, 2048, &f32, true, true),
            RTX_4090.gpu_tile(2048, 2048, 2048, &f32, true, false)
        );
    }

    /// The fill a site's base geometry gives: 4090 1024² is 128 blocks of 4
    /// warps over 128 × 4 schedulers = 1 (under-filled); 2048² is 256 blocks
    /// of 8 warps = 4.
    #[test]
    fn warps_per_scheduler_reads_the_base_grid() {
        let f32 = Ty::f32();
        let wps = |p: &TargetProfile, n| {
            let g = p
                .gpu_tile(n, n, n, &f32, false, false)
                .expect("divisible square site");
            let blocks = (n / g.bm()) * (n / g.bn());
            p.gpu()
                .expect("gpu")
                .warps_per_scheduler(blocks, g.threads())
        };
        assert_eq!(wps(&RTX_4090, 1024), 1);
        assert_eq!(wps(&RTX_4090, 2048), 4);
        assert_eq!(wps(&RTX_5090, 1024), 0);
        assert_eq!(wps(&H100_PCIE, 1024), 1);
        // S54 arch-select: `benches/gpu/rungs.py` classes a measured size by
        // this fill — Under at n ≤ 1024, Filled above — on every measured part.
        for p in GPU_MEASURED {
            for n in [512, 1024, 2048, 4096, 8192] {
                let under = wps(p, n) < GPU_RESIDENT_BLOCKS;
                assert_eq!(under, n <= 1024, "{} {n}: {}", p.name, wps(p, n));
            }
        }
    }

    /// The rung table (S53): `cp.async` pays only under-filled on Ada, never on
    /// Hopper or Blackwell's contract face; every OFF names its results row; an unmeasured part takes the
    /// measured part of its architecture.
    #[test]
    fn gpu_rungs_are_the_measured_table() {
        assert!(RTX_4090.gpu_rungs(1, true).cp_async);
        assert!(!RTX_4090.gpu_rungs(4, true).cp_async);
        // With fma-regs' register-staged tile, the exact face streams by
        // registers when filled too (S53 integration).
        assert!(!RTX_4090.gpu_rungs(4, false).cp_async);
        assert!(RTX_4090.gpu_rungs(1, false).cp_async);
        assert!(!H100_PCIE.gpu_rungs(1, true).cp_async);
        // 5090 under-filled: async paid on the S52 tile, not on warp-fill's.
        assert!(!RTX_5090.gpu_rungs(0, true).cp_async);
        // S54: `devmap` and `cell-law` measured positive on all five parts
        // (`devmap-<part>.txt`; the evidence is on the `GpuRungs` fields).
        for p in GPU_MEASURED {
            for contract in [false, true] {
                let r = p.gpu_rungs(GPU_RESIDENT_BLOCKS, contract);
                assert!(r.devmap && r.cell_law, "{}", p.name);
            }
        }

        let measured = GPU_MEASURED;
        for d in GPU_RUNGS_OFF {
            assert!(
                measured.iter().any(|p| p.name == d.profile),
                "{d:?}: not a measured profile"
            );
            assert!(crate::GpuRungs::NAMES.contains(&d.rung), "{d:?}");
            assert!(d.cite.contains(".txt") && d.cite.contains("ms"), "{d:?}");
        }
        for wps in [0, 1, 2, 4, 64] {
            let fill = if wps < GPU_RESIDENT_BLOCKS {
                Fill::Under
            } else {
                Fill::Filled
            };
            // S54: a `Contract` row covers the TF32 face (a contract face);
            // a `Tf32` row covers it alone.
            for face in [Face::Exact, Face::Contract, Face::Tf32] {
                for p in measured {
                    let r = p.gpu_rungs_on(wps, face);
                    assert!(r.arch);
                    for n in crate::GpuRungs::NAMES {
                        let cited = GPU_RUNGS_OFF.iter().any(|d| {
                            d.profile == p.name
                                && d.rung == n
                                && (d.fill == Fill::Any || d.fill == fill)
                                && (d.face == Face::Any
                                    || d.face == face
                                    || (d.face == Face::Contract && face == Face::Tf32))
                        });
                        assert_eq!(!r.get(n), cited, "{} {n} {wps} {face:?}", p.name);
                    }
                }
                // An unmeasured part takes its architecture's measured part;
                // with none, every rung (another architecture's losses are not its own).
                for (p, m) in [(&CUDA_ADA, &RTX_4090), (&H100, &H100_PCIE)] {
                    assert_eq!(p.gpu_rungs_on(wps, face), m.gpu_rungs_on(wps, face));
                }
                let mut other = A100;
                other.name = "unmeasured";
                if let Machine::Gpu(g) = &mut other.machine {
                    g.arch = "sm_87";
                }
                assert_eq!(other.gpu_rungs_on(wps, face), crate::GpuRungs::ALL);
            }
            for contract in [false, true] {
                assert_eq!(
                    RTX_3090.gpu_rungs(wps, contract),
                    RTX_3090.gpu_rungs_on(wps, Face::of(contract, None))
                );
            }
        }
        let tf32 = Some(crate::Reduced::Tf32);
        assert_eq!(Face::of(true, tf32), Face::Tf32);
        assert_eq!(Face::of(true, None), Face::Contract);
        assert_eq!(Face::of(false, None), Face::Exact);
    }

    /// S54: `Gpu::l2_bytes` is the box's `facts l2_bytes=` on every measured
    /// part (`benches/results-s54-gpu/base-<part>.txt`), and the whitepaper
    /// figure on the two parts no box has reported (the 4070 Ti, H100 SXM).
    #[test]
    fn l2_bytes_is_the_device_query() {
        let mb = 1024 * 1024;
        for (p, facts) in [
            (&RTX_3090, 6_291_456),
            (&A100, 41_943_040),
            (&RTX_4090, 75_497_472),
            (&H100_PCIE, 52_428_800),
            (&RTX_5090, 100_663_296),
            (&H100, 50 * mb),
            (&CUDA_ADA, 48 * mb),
        ] {
            assert_eq!(p.gpu().expect("gpu").l2_bytes, facts, "{}", p.name);
        }
    }

    /// S54 `raster`: the group height minimises the panels one wave reads,
    /// keeps row-major under the L2 gate and on a one-wave grid, and only
    /// ever divides the grid's height.
    #[test]
    fn raster_is_the_wave_panel_minimum_behind_the_l2_gate() {
        let g = RTX_3090.gpu().expect("gpu");
        let wave = 82 * 2;
        // 128×128 blocks, f32: one block's panels are 1 KB per k.
        assert_eq!(g.raster(128, 128, wave, 4096, 4096, 4096, 4, true), 1); // 4 MB ≤ 6 MB
        assert_eq!(g.raster(128, 128, wave, 8192, 8192, 8192, 4, true), 16); // 8 MB > 6 MB
        // h + ⌈164 / h⌉ panels: 8 → 29, 16 → 27, 32 → 38.
        assert_eq!(g.raster(128, 128, wave, 8192, 8192, 8192, 4, false), 16);
        assert_eq!(g.raster(128, 128, wave, 4096, 4096, 4096, 4, false), 16);
        // One wave (16 × 8 blocks ≤ 164): order changes nothing.
        assert_eq!(g.raster(128, 128, wave, 1024, 2048, 8192, 4, false), 1);
        // A grid 3 blocks high: the largest power of two dividing it is 1.
        assert_eq!(g.raster(128, 128, wave, 384, 8192, 8192, 4, false), 1);
        let big = RTX_4090.gpu().expect("gpu");
        assert_eq!(big.raster(128, 128, 256, 8192, 8192, 8192, 4, true), 1); // 8 MB ≤ 72 MB
    }

    /// S54 R5 (`retain-resident-*.txt`): the delta upload and the strided
    /// one-row site ship on every measured part; retention on the resident
    /// buffer only where the whole K/V site costs more than its row (not the
    /// 4090, not the 5090).
    #[test]
    fn r5_rungs_are_the_measured_table() {
        for p in GPU_MEASURED {
            for wps in [0, 64] {
                for contract in [false, true] {
                    let r = p.gpu_rungs(wps, contract);
                    assert!(r.resident && r.delta_h2d && r.lane_stride, "{}", p.name);
                    let off = ["rtx-4090", "rtx-5090"].contains(&p.name);
                    assert_eq!(r.retain_resident, !off, "{}", p.name);
                }
            }
        }
    }

    /// S54: `rowfold` ships on every measured part; the reader-chosen store
    /// order only where its device time measured faster in two sessions — the
    /// 4090's contract face and the A100's exact face.
    #[test]
    fn rowfold_rungs_are_the_measured_table() {
        for p in GPU_MEASURED {
            for contract in [false, true] {
                let r = p.gpu_rungs(GPU_RESIDENT_BLOCKS, contract);
                assert!(r.rowfold, "{} {contract}", p.name);
                let layout = matches!((p.name, contract), ("rtx-4090", true) | ("a100", false));
                assert_eq!(r.rowfold_layout, layout, "{} {contract}", p.name);
            }
        }
    }

    /// S53 `pad`: the register-staged transposed-A store. With all four lanes
    /// of a row in one warp no 16-byte-aligned pad is conflict-free; two lanes
    /// per row (one 32-byte sector) are, at a 4-word pad.
    #[test]
    fn gpu_tile_sync_a_is_derived_from_the_bank_geometry() {
        let f32 = Ty::f32();
        assert_eq!(conflict_free_pad(128, |l| (4 * (l % 4), l / 4)), None);
        for p in [&RTX_4090, &H100_PCIE, &RTX_5090] {
            let g = p
                .gpu_tile(4096, 4096, 4096, &f32, true, true)
                .expect("tile");
            for bk in [8, 16, 32] {
                let s = p.gpu_tile_sync_a(GpuTile { bk, ..g }, &f32);
                assert_eq!((s.a_row_lanes, s.pad_a), (2, 4), "{} bk {bk}", p.name);
            }
            // f64 keeps the S52 map: the bank model counts words.
            let d = p.gpu_tile_sync_a(g, &Ty::f64());
            assert_eq!((d.a_row_lanes, d.pad_a), (0, 0));
        }
    }

    /// The tensor-core geometry, worked by hand from the device facts.
    #[test]
    fn gpu_mma_tile_is_derived_from_device_facts() {
        let t = |p: &TargetProfile, n| {
            let x = p
                .gpu_mma_tile(n, n, n, false, true)
                .expect("divisible square site");
            (
                x.wm(),
                x.wn(),
                x.bm(),
                x.bn(),
                x.bk,
                x.stages,
                x.pad_a,
                x.pad_b,
            )
        };
        // 255 regs ⇒ 16 m16n8 tiles = 64×32 warp tile; 65 536 / 128 / 2 ⇒ 8
        // warps ⇒ 2×4 ⇒ 128×128. Pads: A rows of bk+4 words put the 8 fragment
        // rows 4 banks apart; B rows of 128+8 put the 4 k-rows 8 banks apart.
        // 4090: 49 KB per block ⇒ bk 16 (two 18.4 KB stages), 2 stages.
        assert_eq!(t(&RTX_4090, 4096), (64, 32, 128, 128, 16, 2, 4, 8));
        // H100: 113 KB per block ⇒ bk 32 (35.8 KB) ⇒ 3 stages.
        assert_eq!(t(&H100, 4096), (64, 32, 128, 128, 32, 3, 4, 8));
        // Pre-sm_80 has no `mma.sync` TF32; a CPU has no tile at all.
        assert_eq!(GENERIC.gpu_mma_tile(4096, 4096, 4096, false, true), None);
        // The wide rule: 4×8 tiles (64×64 warp, 224 registers) ⇒ 292 resident
        // threads ⇒ 4 warps per block ⇒ 2×2 warps = 128×128 (CUTLASS's).
        let w = RTX_4090
            .gpu_mma_tile(4096, 4096, 4096, true, true)
            .expect("wide");
        assert_eq!(
            (w.wm(), w.wn(), w.bm(), w.bn(), w.bk, w.stages),
            (64, 64, 128, 128, 16, 2)
        );
    }

    /// `warp-fill` (S53): the tile shrinks until the grid holds
    /// 2 · 4 schedulers · SMs warps, before the thread grid splits for blocks.
    #[test]
    fn gpu_warp_fill_shrinks_the_tile_until_every_scheduler_has_two_warps() {
        let f32 = Ty::f32();
        let gpu = |p: &TargetProfile| *p.gpu().expect("gpu");
        assert_eq!(gpu(&RTX_4090).schedulers(), 4);
        assert_eq!(gpu(&RTX_4090).fill_warps(), 1024);
        let fma = |p: &TargetProfile, n, fill| {
            let x = p.gpu_tile(n, n, n, &f32, false, fill).expect("divisible");
            let warps = (n / x.tm) * (n / x.tn) / 32;
            (x.tm, x.tn, x.bm(), x.bn(), x.threads(), warps)
        };
        // 4090 1024²: 8×8 gives 512 warps (1 per scheduler) ⇒ halve tn ⇒ 8×4,
        // 1024 warps (2 per scheduler); 16×16 threads ⇒ 128×64, 128 blocks.
        assert_eq!(fma(&RTX_4090, 1024, false), (8, 8, 128, 64, 128, 512));
        assert_eq!(fma(&RTX_4090, 1024, true), (8, 4, 128, 64, 256, 1024));
        // H100 PCIe (114 SMs) needs 912 ⇒ 8×4; H100 SXM (132) 1056 and the
        // 5090 (170) 1360 ⇒ 4×4, 64×64 blocks, 2048 warps.
        assert_eq!(fma(&H100_PCIE, 1024, true), (8, 4, 128, 64, 256, 1024));
        assert_eq!(fma(&H100, 1024, true), (4, 4, 64, 64, 256, 2048));
        assert_eq!(fma(&RTX_5090, 1024, true), (4, 4, 64, 64, 256, 2048));
        // Never below one vector: a 256² site floors at 4×4 and then splits
        // the thread grid for blocks as before.
        assert_eq!(fma(&RTX_4090, 256, true).0, 4);
        assert_eq!(fma(&RTX_4090, 256, true).1, 4);

        let mma = |p: &TargetProfile, n, fill| {
            let x = p.gpu_mma_tile(n, n, n, true, fill).expect("divisible");
            (x.wm(), x.wn(), x.bm(), x.bn(), (n / x.wm()) * (n / x.wn()))
        };
        // Wide 64×64 warp tile at 1024²: 256 warps (0.5 per scheduler) ⇒
        // 64×32 ⇒ 32×32, 1024 warps; 2×2 warps per block ⇒ 64×64.
        assert_eq!(mma(&RTX_4090, 1024, false), (64, 64, 128, 64, 256));
        assert_eq!(mma(&RTX_4090, 1024, true), (32, 32, 64, 64, 1024));
        // No-op wherever the grid already fills the schedulers.
        for p in [&RTX_4090, &H100, &H100_PCIE, &RTX_5090] {
            for n in [3072, 4096] {
                assert_eq!(
                    p.gpu_tile(n, n, n, &f32, false, true),
                    p.gpu_tile(n, n, n, &f32, false, false)
                );
                assert_eq!(
                    p.gpu_mma_tile(n, n, n, true, true),
                    p.gpu_mma_tile(n, n, n, true, false)
                );
            }
        }
    }

    /// The Hopper `wgmma` geometry (S53), worked by hand from the device facts.
    #[test]
    fn gpu_wgmma_tile_is_derived_from_hopper_facts() {
        let t = |n| {
            let x = H100_PCIE
                .gpu_wgmma_tile(n, n, n)
                .expect("sm_90a, divisible square site");
            assert_eq!(x.stage_words() * 4 % 1024, 0, "stages start on atoms");
            (
                x.bm(),
                x.bn(),
                x.groups,
                x.bk,
                x.stages,
                x.pad_b,
                x.blocks_per_sm,
            )
        };
        // 4096: n = 256 (128 acc + 16 fragments + 72 ≤ 255 at bk 32);
        // 65 536 / 216 = 303 threads ⇒ 2 warpgroups in one block ⇒ 256 × 128;
        // the 227 KB share holds 4 bk-32 stages of 49 KB, not 3 of bk 64.
        // 2048 and 8192 already fill 114 SMs with it.
        for n in [2048, 4096, 8192] {
            assert_eq!(t(n), (256, 128, 2, 32, 4, 8, 1), "{n}");
        }
        // 1024: 4 × 8 blocks < 114 SMs ⇒ one warpgroup, then n 128 ⇒ 128
        // blocks; 3 resident ⇒ 75 KB ⇒ exactly three bk-32 stages.
        assert_eq!(t(1024), (128, 64, 1, 32, 3, 8, 3));
        // `wgmma` is sm_90a only: not Ada, not consumer Blackwell, not a CPU.
        for p in [&RTX_4090, &RTX_5090, &CUDA_ADA, &GENERIC] {
            assert_eq!(p.gpu_wgmma_tile(4096, 4096, 4096), None, "{}", p.name);
        }
        assert!(H100.gpu().is_some_and(Gpu::has_wgmma));
    }

    /// The matrix descriptor's constant fields for the S52 bench geometries:
    /// K-major, 128-byte swizzle — LBO field 1 (unused, "assumed 1"), SBO =
    /// one 8 × 128 B atom between 8-row groups, mode 1. Start address and
    /// base offset stay zero (the stage is filled in per panel, on an atom).
    #[test]
    fn wgmma_descriptor_fields_are_the_128b_swizzle_k_major_ones() {
        for n in [1024, 2048, 4096, 8192] {
            let d = H100_PCIE
                .gpu_wgmma_tile(n, n, n)
                .expect("geometry")
                .desc_fields();
            assert_eq!(d & 0x3fff, 0, "start address");
            assert_eq!((d >> 16) & 0x3fff, 1, "LBO");
            assert_eq!((d >> 32) & 0x3fff, 1024 >> 4, "SBO");
            assert_eq!((d >> 49) & 0x7, 0, "base offset");
            assert_eq!(d >> 62, 1, "128-byte swizzle");
            assert_eq!(d, 0x4000_0040_0001_0000);
        }
    }

    /// The placeholder honesty pin. `cuda-ada` inherits CPU-shaped constants it
    /// has no business claiming as GPU measurements; this test states that in
    /// the suite so nobody reads them as swept. When a GPU realization starts
    /// consuming one of these, it either gets its own value or this assertion
    /// changes — either way the change is deliberate and reviewed.
    #[test]
    fn cuda_ada_cpu_shaped_fields_are_inherited_placeholders() {
        assert_eq!(CUDA_ADA.vec_bytes, GENERIC.vec_bytes);
        assert_eq!(CUDA_ADA.vec_regs, GENERIC.vec_regs);
        assert_eq!(CUDA_ADA.l2_bytes, GENERIC.l2_bytes);
        assert_eq!(CUDA_ADA.acc_vecs_per_row, GENERIC.acc_vecs_per_row);
    }

    /// S54 Ampere: the compute-capability facts (CUDA C Programming Guide
    /// tables for 8.0 and 8.6; the `deviceQuery` values), the ISA gates they
    /// open, and the geometry derived from them; the rung rows two S54 sessions
    /// measured.
    #[test]
    fn ampere_profiles_carry_the_compute_capability_facts() {
        let f32 = Ty::f32();
        let facts = |p: &TargetProfile| {
            let g = *p.gpu().expect("gpu");
            (
                g.arch,
                g.sms,
                g.smem_per_sm,
                g.smem_per_block,
                g.max_threads_per_sm,
                g.max_blocks_per_sm,
            )
        };
        assert_eq!(facts(&RTX_3090), ("sm_86", 82, 102_400, 101_376, 1536, 16));
        assert_eq!(facts(&A100), ("sm_80", 108, 167_936, 166_912, 2048, 32));
        for p in [&RTX_3090, &A100] {
            assert_eq!(resolve(p.name).map(|r| r.name), Some(p.name));
            let g = *p.gpu().expect("gpu");
            // cp.async, ldmatrix (sm_75) and `mma.sync` TF32 all hold from
            // sm_80; `wgmma` is sm_90a only.
            assert!(g.has_mma_sync() && !g.has_wgmma(), "{}", p.name);
            assert_eq!(p.gpu_wgmma_tile(4096, 4096, 4096), None);
            assert!(p.gpu_mma_tile(4096, 4096, 4096, true, true).is_some());
            let t = p
                .gpu_tile(4096, 4096, 4096, &f32, false, false)
                .expect("tile");
            assert!(p.gpu_tile_async(t, 4096, &f32).is_some());
            // S54, both parts: register staging beats cp.async on every grid
            // and face; resident beats retain-delta on decode; tensor cores stay.
            for wps in [0, 1, 2, 64] {
                for contract in [false, true] {
                    let r = p.gpu_rungs(wps, contract);
                    assert!(!r.cp_async && !r.retain_delta && r.mma, "{} {wps}", p.name);
                }
            }
        }
        // 2 warps × 4 schedulers × SMs.
        assert_eq!(RTX_3090.gpu().map(Gpu::fill_warps), Some(656));
        assert_eq!(A100.gpu().map(Gpu::fill_warps), Some(864));
        let g = |p: &TargetProfile, n| {
            let x = p.gpu_tile(n, n, n, &f32, false, false).expect("square");
            (x.bm(), x.bn(), x.threads(), x.bk, x.blocks_per_sm)
        };
        // 3090: Ada's 100 KB SM ⇒ the 4090's 4096² tile; 1024² is 64 blocks
        // < 82 SMs ⇒ halve x once.
        assert_eq!(g(&RTX_3090, 4096), (128, 128, 256, 32, 2));
        assert_eq!(g(&RTX_3090, 1024), (128, 64, 128, 32, 4));
        // A100: 164 KB / 2 blocks − 1 KB ⇒ bk 64, as on Hopper.
        assert_eq!(g(&A100, 4096), (128, 128, 256, 64, 2));
        let m = |p: &TargetProfile| {
            let x = p.gpu_mma_tile(4096, 4096, 4096, false, true).expect("mma");
            (x.wm(), x.wn(), x.bm(), x.bn(), x.bk, x.stages)
        };
        assert_eq!(m(&RTX_3090), (64, 32, 128, 128, 16, 2));
        assert_eq!(m(&A100), (64, 32, 128, 128, 32, 2));
    }
}
