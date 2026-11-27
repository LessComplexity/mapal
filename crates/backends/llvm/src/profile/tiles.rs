//! Tile geometries (`GpuMmaTile`, `GpuWgmmaTile`, `WsForm`), shared-memory padding, `Sme`, `DispatchCost`.

use super::*;

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
    /// The matrix instruction's operand format ([`MmaFmt`]): TF32's
    /// `m16n8k8`, or a 16-bit `m16n8k16` (tensor-cores, ADR-0043 D6).
    pub fmt: MmaFmt,
    /// Bytes one element of the `a` / `b` panel holds in shared memory: the
    /// operand's storage (a narrow array's 2, an f32 array's 4 — rounded to
    /// the instruction's format as each fragment is read).
    pub ea: u64,
    pub eb: u64,
    /// The `b` panel is stored `n × k` (its operand runs along `k`: `w[j·k
    /// + q]`, attention's `K`, a `[out][in]` weight) — the `col` operand
    /// `ldmatrix` reads without `.trans`; else `k × n`.
    pub bt: bool,
    /// f32-tensor-cores R-F `split-tile`: a `{f32}` product realized on this
    /// tile by a split form ([`mapal_ir::split::SplitKind`]: `Tf32x3` on the
    /// TF32 kernel, `Narrow1` on the 16-bit one) — each k-step's part
    /// products summed from a zeroed accumulator, then one round-to-nearest
    /// add into the f32 accumulator; `None`: the member's realization.
    pub split: Option<mapal_ir::split::SplitKind>,
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
        if self.bt {
            self.bk + self.pad_b
        } else {
            self.bn() + self.pad_b
        }
    }
    /// Rows the `b` panel spans in shared memory (`bk`, or `bn` when `bt`).
    pub fn b_rows(&self) -> u64 {
        if self.bt { self.bn() } else { self.bk }
    }
    /// The instruction's `k` (TF32 8, 16-bit 16).
    pub fn mma_k(&self) -> u64 {
        32 / self.fmt.bytes()
    }
    /// 32-bit words one stage holds: the `bm × bk` A panel and the `bk × bn`
    /// B panel, each at its padded row length (TF32: 4-byte elements).
    pub fn stage_words(&self) -> u64 {
        self.bm() * self.lda() + self.bk * self.ldb()
    }
    /// Bytes the A panel of one stage spans (where its B panel starts).
    pub fn a_bytes(&self) -> u64 {
        self.bm() * self.lda() * self.ea
    }
    /// Bytes one stage holds, each panel at its own element width.
    pub fn stage_bytes(&self) -> u64 {
        self.a_bytes() + self.b_rows() * self.ldb() * self.eb
    }
    pub fn smem_bytes(&self) -> u64 {
        self.stages * self.stage_bytes()
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

/// hopper-wgmma-tma: the copies one SM's bulk-copy engine overlaps, the
/// derivation of [`Gpu::tma_copy_ns`] where it is not measured — the RTX 5080's
/// row-copy kernel (1.961 ms over 1.57 M copies on 84 SMs: about 100 ns a
/// copy, `gemv-tma.md`) at its loaded latency of 573 ns: ≈ 6.
pub const TMA_COPIES_IN_FLIGHT: u64 = 6;
/// Registers a warp-specialized `wgmma` consumer thread spends beyond its
/// accumulators and `a` fragments (descriptors, barrier and fragment
/// addresses, the loop): read off `ptxas -v` at `sm_90a` (the P0 measurement,
/// [`WGMMA_ADDR_REGS`]'s policy for the producer-fed kernel).
pub const WGMMA_WS_ADDR_REGS: u64 = 40;
/// The producer warpgroup's register count under `setmaxnreg` (its loop:
/// box coordinates, the barrier waits, one rounded chunk at a time).
pub const WGMMA_WS_PRODUCER_REGS: u64 = 40;

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
    /// hopper-wgmma-tma R2 (`wgmma-ws`): the block is one producer warpgroup
    /// — its elected thread copies each stage as whole tensor boxes onto a
    /// full barrier, its threads round the `aᵀ` box once — beside `groups`
    /// consumer warpgroups that wait on it and release each stage on an
    /// empty barrier once the group reading it completed.
    /// `false`: the S53 kernel (every warp copies by `cp.async`).
    pub ws: bool,
    /// R3 (`ws-regs`): the producer's and each consumer's register count
    /// under `setmaxnreg` (`None`: every thread keeps the launch's count).
    pub realloc: Option<(u64, u64)>,
    /// hopper-launch `ws-persist`: the persistent grid — `ctas` CTAs
    /// (`sms × blocks_per_sm`), each walking the tiles `cta, cta + ctas, …`
    /// on one stage ring ([`crate::rules::ws_persist_pays`]). 0: a CTA a tile.
    pub ctas: u64,
    /// wgmma-16bit: **the instruction's operand format** — `Tf32`: S53's
    /// product; `Bf16` / `F16`: the 16-bit warpgroup product
    /// (`wgmma.m64nNk16`), one tile family over the matrix-op capability.
    pub fmt: MmaFmt,
    /// wgmma-16bit: **`c = a·b` as stored** — consumer warpgroup `cw` owns
    /// rows `64·cw ..+64` of the block's `c` tile × its `n` columns, `a` the
    /// instruction's `a` and `b` its descriptor `b` in either major
    /// ([`MatrixOp::mn_major`]). `false`: the transposed product `cᵀ = bᵀ·aᵀ`
    /// the K-major-only TF32 descriptor forces (S53).
    pub direct: bool,
    /// Bytes of one staged `a` element (4: f32 storage — rounded to `fmt` as
    /// the instruction's operand is formed; 2: stored in the format).
    pub ea: u64,
    /// Bytes of one staged `b` element.
    pub eb: u64,
    /// wgmma-16bit: **where the instruction's `a` operand lives when it
    /// issues** — `Regs`: fragments the consumer loads from the stage (the
    /// TF32 kernel's `bᵀ`, rounded in the load; a 16-bit product's f32 `a`,
    /// rounded to the format in the load); `Smem`: the stage itself through a
    /// K-major descriptor (a 16-bit `a` stored in the format).
    pub a_home: Home,
    /// wgmma-16bit: `b` staged k-major (`n` rows of `k`: the descriptor's
    /// K-major layout), else lanes-major (`k` rows of 128-byte lane slabs:
    /// MN-major, the transpose bit set).
    pub b_kmajor: bool,
}

impl GpuWgmmaTile {
    /// The S53 / hopper-wgmma-tma TF32 form's operand fields (`cᵀ = bᵀ·aᵀ`,
    /// f32 storage, `bᵀ` fragments in registers, `b` lanes-major).
    pub const TF32: GpuWgmmaTile = GpuWgmmaTile {
        n: 0,
        groups: 0,
        bk: 0,
        stages: 0,
        pad_b: 0,
        blocks_per_sm: 0,
        ws: false,
        realloc: None,
        ctas: 0,
        fmt: MmaFmt::Tf32,
        direct: false,
        ea: 4,
        eb: 4,
        a_home: Home::Regs,
        b_kmajor: false,
    };
    pub fn bm(&self) -> u64 {
        match self.direct {
            true => self.groups * WGMMA_M,
            false => self.n,
        }
    }
    pub fn bn(&self) -> u64 {
        match self.direct {
            true => self.n,
            false => self.groups * WGMMA_M,
        }
    }
    /// The instruction's `k` (32 bytes of each operand row).
    pub fn mma_k(&self) -> u64 {
        32 / self.fmt.bytes()
    }
    pub fn threads(&self, warp: u64) -> u64 {
        (self.groups + u64::from(self.ws)) * WARPGROUP_WARPS * warp
    }
    /// `wgmma-ws`: bytes of a stage's `a` box (`bm` rows of `bk` elements —
    /// the TF32 form's `aᵀ` — as 128-byte swizzled digit columns).
    pub fn ws_a_bytes(&self) -> u64 {
        self.bm() * self.bk * self.ea
    }
    /// `wgmma-ws`: bytes of a stage's `b` boxes.
    pub fn ws_b_bytes(&self) -> u64 {
        self.bn() * self.bk * self.eb
    }
    /// `wgmma-ws`: the `b` panel's boxes without `tma-digits` — lanes-major:
    /// one 2-d box a 128-byte lane slab (`bk` rows each); k-major: one box
    /// (`bn` rows of the `k` digit columns, 3-d past one column, as `a`'s).
    pub fn ws_b_boxes(&self) -> u64 {
        match self.b_kmajor {
            true => 1,
            false => self.bn() * self.eb / WGMMA_SWIZZLE_BYTES,
        }
    }
    /// `wgmma-ws`: the `a` box's 128-byte `k` digit columns (`bm` rows each).
    pub fn ws_a_cols(&self) -> u64 {
        (self.bk * self.ea).div_ceil(WGMMA_SWIZZLE_BYTES)
    }
    /// `wgmma-ws`: one stage — the `a` box, then the `b` boxes — whole
    /// 1024-byte swizzle patterns, so every box starts on one.
    pub fn ws_stage_bytes(&self) -> u64 {
        let atom = WGMMA_ATOM_ROWS * WGMMA_SWIZZLE_BYTES;
        (self.ws_a_bytes() + self.ws_b_bytes()).next_multiple_of(atom)
    }
    /// `wgmma-ws`: the stages and their three barriers each (full, ready,
    /// empty; 8 bytes apiece).
    pub fn ws_smem_bytes(&self) -> u64 {
        self.stages * (self.ws_stage_bytes() + 24)
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
        if self.ws {
            return self.ws_smem_bytes();
        }
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
    /// wgmma-16bit: the MN-major descriptor of the lanes-major `b` panel
    /// (PTX ISA, MN-major swizzled layouts): LBO the stride between 128-byte
    /// lane slabs — one slab is `bk` swizzled rows, so `128·bk` — SBO
    /// between 8-row groups of `k` (one 1024-byte atom), the 128-byte mode.
    pub fn desc_mn_fields(&self) -> u64 {
        let lbo = WGMMA_SWIZZLE_BYTES * self.bk;
        ((lbo >> 4) << 16) | ((self.sbo() >> 4) << 32) | (1 << 62)
    }
}

/// wgmma-16bit: **the operand side of a warp-specialized `wgmma` tile** —
/// the warpgroup product it issues (its format, `a`'s home, whether a
/// descriptor operand may be MN-major), each operand's staged element bytes
/// (an f32 `a` of a 16-bit product is staged f32 and rounded in its fragment
/// load), and whether `b` is staged k-major. The hopper-wgmma-tma TF32 form:
/// the TF32 group product, 4 / 4, lanes-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WsForm {
    pub op: MatrixOp,
    pub ea: u64,
    pub eb: u64,
    pub b_kmajor: bool,
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

/// The pad (in 2-byte elements) under which `ldmatrix`'s eight 16-byte rows
/// of one 8×8 matrix, `ld` elements apart, fall in eight distinct 16-byte
/// bank groups: a row stride of an odd number of 16-byte groups.
pub(crate) fn ldmatrix_pad(ld: u64) -> u64 {
    if (ld / 8) % 2 == 1 { 0 } else { 8 }
}

/// The pad (in 4-byte words) under which each half warp's `LDS.64` of an
/// f32 `a` fragment pair (row `g`, words `2t, 2t+1`) hits 32 distinct banks,
/// rows 16-byte aligned.
pub(crate) fn pair_pad(ld: u64) -> Option<u64> {
    (0..32).step_by(4).find(|pad| {
        let mut banks = [false; 32];
        (0..16).all(|lane| {
            let w = (lane / 4) * (ld + pad) + 2 * (lane % 4);
            let b = (w % 32) as usize;
            !std::mem::replace(&mut banks[b], true) && !std::mem::replace(&mut banks[b + 1], true)
        })
    })
}

/// **The bank degree of one warp's shared-memory instruction** (32 banks of
/// 4 bytes): the most distinct words any bank serves, and the ideal
/// `⌈distinct words / 32⌉`. `acc` is each lane's first word and its width
/// in words. Degree = ideal is conflict-free. The one bank fact: the k-major
/// pad search, `lsu_per_panel`'s weights and the plan line's `adeg` read it.
pub(crate) fn bank_degree(acc: impl IntoIterator<Item = (u64, u64)>) -> (u64, u64) {
    let mut words: Vec<u64> = acc.into_iter().flat_map(|(a, w)| a..a + w).collect();
    words.sort_unstable();
    words.dedup();
    let mut banks = [0u64; 32];
    for w in &words {
        banks[(w % 32) as usize] += 1;
    }
    let deg = banks.iter().copied().max().unwrap_or(0).max(1);
    (deg, (words.len() as u64).div_ceil(32).max(1))
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
    /// Elements the `a` panel of one stage holds: `bk` transposed rows of
    /// [`Self::sa`], or (`kmajor`) `bm` rows of `bk + pad_a`.
    pub fn a_elems(&self) -> u64 {
        match self.kmajor {
            true => self.bm() * (self.bk + self.pad_a),
            false => self.bk * self.sa(),
        }
    }
    /// Elements one stage holds: the A panel and the B panel.
    pub fn stage_elems(&self) -> u64 {
        self.a_elems() + self.bk * self.bn()
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
    /// (`docs/performance/cpu/matmul/sme-throughput-ceiling.md` §5, `benches/sme/kc.c`). One
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
    /// The pool width `dispatch_ns` was measured at, when the wave was measured
    /// per host (`native`'s calibration, local-rtx-5080). `0` = not known per
    /// lane: the plan-s50 P3 threshold (`ns < dispatch_ns` inlines, else the
    /// whole pool) — every hand-measured profile, byte for byte.
    ///
    /// Known, a wave's cost is per lane woken: the host wakes each filled lane
    /// in turn (`Pool::notify_lane`), so `L` lanes cost `L * dispatch_ns /
    /// lanes` (9970X, `examples/dispatch_cost.rs`: 24 / 50 / 149 / 263 us at 8 /
    /// 16 / 32 / 64 lanes). A task of serial time `T` on `L` lanes then takes
    /// `T / L + L * w`, least at `L = sqrt(T / w)`: that width is what the
    /// emitter registers (`width`, mapal-rt `TaskDef::width`), one lane being
    /// inline. A `T` estimated `r` times low narrows the width by `sqrt(r)`,
    /// which is why a peak-rate roofline is usable here and not as a threshold.
    pub lanes: u64,
}

impl DispatchCost {
    /// Lanes a task of serial `ns` should wake, by [`Self::lanes`]: `Some(1)` =
    /// inline, `Some(0)` = the whole pool (registered `width = 0`, as before),
    /// `None` = no per-lane fact (the threshold rule decides).
    pub fn width(&self, ns: u64) -> Option<u64> {
        if self.lanes < 2 {
            return None;
        }
        let per_lane = (self.dispatch_ns / self.lanes).max(1);
        let w = ((ns / per_lane) as f64).sqrt().round() as u64;
        Some(if w >= self.lanes { 0 } else { w.max(1) })
    }
}

/// Nanoseconds a roofline gives `flops` and `bytes` at these rates.
pub(crate) fn roofline_ns(flops: u64, bytes: u64, flops_per_ns: u64, bytes_per_ns: u64) -> u64 {
    (flops / flops_per_ns.max(1)).max(bytes / bytes_per_ns.max(1))
}

/// A tile site's bytes: each operand panel and the output walked once.
pub(crate) fn tile_bytes(rows: u64, c: u64, k: u64, elem: &Ty) -> u64 {
    (rows.saturating_mul(k) + k.saturating_mul(c) + rows.saturating_mul(c))
        .saturating_mul(elem_bytes(elem))
}

/// [`tile_bytes`] at each operand's storage width (ADR-0043: a narrow
/// operand moves 2 bytes an element, the output the product's width).
pub(crate) fn site_bytes(site: &mapal_ir::TileSite) -> u64 {
    let (ea, eb, e) = (
        site.a.elem_bytes(&site.elem),
        site.b.elem_bytes(&site.elem),
        elem_bytes(&site.elem),
    );
    site.rows
        .saturating_mul(site.k)
        .saturating_mul(ea)
        .saturating_add(site.k.saturating_mul(site.c).saturating_mul(eb))
        .saturating_add(site.rows.saturating_mul(site.c).saturating_mul(e))
}
