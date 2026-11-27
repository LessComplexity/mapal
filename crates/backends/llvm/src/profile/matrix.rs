//! Matrix instructions as geometry (`MmaShape`, `MatrixOp`) and the capability tables built from them.

use super::*;

/// One matrix instruction's geometry: `m × n × k` per issuing group of
/// `threads` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaShape {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub threads: u64,
}

/// The format both operands of a matrix instruction carry. Every product of
/// two such values is exact in the f32 accumulator (ADR-0043 context 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaFmt {
    /// f32 storage read as TF32 (10 explicit significand bits).
    Tf32,
    Bf16,
    F16,
    /// OCP fp8: 4 exponent / 3 fraction bits, finite only.
    E4m3,
    /// OCP fp8: 5 exponent / 2 fraction bits.
    E5m2,
}

impl MmaFmt {
    /// Bytes one operand element occupies in a fragment.
    pub fn bytes(self) -> u64 {
        match self {
            MmaFmt::Tf32 => 4,
            MmaFmt::Bf16 | MmaFmt::F16 => 2,
            MmaFmt::E4m3 | MmaFmt::E5m2 => 1,
        }
    }

    /// The format a narrow storage format (or 16-bit member, ADR-0043 D4)
    /// multiplies in.
    pub fn of_narrow(n: mapal_ir::NarrowFloat) -> MmaFmt {
        match n {
            mapal_ir::NarrowFloat::Bf16 => MmaFmt::Bf16,
            mapal_ir::NarrowFloat::F16 => MmaFmt::F16,
            mapal_ir::NarrowFloat::F8e4m3 => MmaFmt::E4m3,
            mapal_ir::NarrowFloat::F8e5m2 => MmaFmt::E5m2,
        }
    }

    /// The PTX type name (`.bf16`, `.e4m3`, …).
    pub fn ptx(self) -> &'static str {
        match self {
            MmaFmt::Tf32 => "tf32",
            MmaFmt::Bf16 => "bf16",
            MmaFmt::F16 => "f16",
            MmaFmt::E4m3 => "e4m3",
            MmaFmt::E5m2 => "e5m2",
        }
    }
}

/// Where a matrix instruction's operand or accumulator lives when it issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    /// Each thread's registers, in the instruction's fragment layout.
    Regs,
    /// Shared memory, addressed by a matrix descriptor (layout, swizzle).
    Smem,
    /// Tensor memory: a per-SM accumulator store beside the register file
    /// (allocated, loaded and stored by its own instructions).
    Tmem,
}

/// **One matrix instruction, described by the geometry it executes**
/// (hardware-properties): its shape per issuing group, its operand format,
/// where `a`, `b` and the accumulator live at issue, and its accumulator's
/// measured unit roundoff `u_a = 2^acc_u`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixOp {
    pub shape: MmaShape,
    pub fmt: MmaFmt,
    pub a: Home,
    pub b: Home,
    pub acc: Home,
    /// The accumulator's unit roundoff as a power of two: −24 rounds each
    /// addition to nearest, −23 truncates the aligned addends and the sum
    /// (the RTX 5080's `mma.sync` bf16 / f16 / e4m3 / e5m2 — `1 + 1.5·2⁻²⁴ →
    /// 1`, `1 − 2⁻²⁵ → 1 − 2⁻²⁴`, worst 4.0–5.3 units of `2⁻²⁴·M` per
    /// instruction, `benches/local/mma_calib.cu`). ADR-0043 D4: a set whose
    /// bound assumes `u_a` takes the instruction only where `acc_u` meets it.
    pub acc_u: i32,
    /// f32-tensor-cores P2: **the accumulator's alignment guard bits** — an
    /// addend below `2^(e_max − 23 − g)` is dropped before the sum
    /// (`mma_calib.cu`'s `g_j` / `g0_j` / `gneg_j` probes: `c = 1` plus `k`
    /// addends of `2^−(23+j)` kept for `j ≤ g`). The split realization's
    /// chain term reads it ([`mapal_ir::split::AccModel`]); `g ≥ 1` is what
    /// keeps ADR-0041 D3's member bound sound ([`mapal_ir::split::AccModel::member_sound`]).
    pub acc_g: u32,
    /// wgmma-16bit: **a shared-memory operand may be MN-major** — the
    /// instruction's transpose bit (`imm-trans-a` / `imm-trans-b`) reads a
    /// descriptor operand whose `m` / `n` axis is the contiguous one, so a
    /// `k × n` panel stored lanes-contiguous is the `b` operand as it lands.
    /// `wgmma`'s 16-bit forms have it; its TF32 and fp8 forms read K-major
    /// descriptors only (the S53 kernel's transposed product `cᵀ = bᵀ·aᵀ`).
    pub mn_major: bool,
}

impl MatrixOp {
    /// The accumulator meets the bound of a set that assumes `u_a = 2^u`.
    pub fn acc_meets(&self, u: i32) -> bool {
        self.acc_u <= u
    }

    /// The accumulator datapath the split realization's bound reads.
    pub fn acc_model(&self) -> mapal_ir::split::AccModel {
        mapal_ir::split::AccModel { g: self.acc_g }
    }
}

/// [`MatrixOp::acc_g`] measured on the RTX 5080 (sm_120) for every warp
/// format — TF32 m16n8k8, bf16 / f16 m16n8k16, e4m3 / e5m2 m16n8k32: two
/// guard bits (`mma_calib.cu`, `f32-tensor-cores.md` §S0), as [`MMA_ACC_U`]
/// the same datapath on every profiled part until a part measures its own.
pub const MMA_ACC_G: u32 = 2;

/// [`MatrixOp::acc_u`] measured on the RTX 5080 (sm_120) for every warp
/// `mma` format (`mma_calib.cu`: truncation): the unit is the same
/// tensor-core datapath on every profiled part (truncating accumulators are
/// also the published measurement of Volta / Turing / Ampere's); a part
/// measured otherwise states its own.
pub const MMA_ACC_U: i32 = -23;

/// The `mma.sync.m16n8k8` TF32 product a warp issues.
pub const MMA_SYNC_TF32: MmaShape = MmaShape {
    m: 16,
    n: 8,
    k: 8,
    threads: 32,
};

/// The `mma.sync.m16n8k16` bf16 / f16 product a warp issues.
pub const MMA_SYNC_16: MmaShape = MmaShape {
    m: 16,
    n: 8,
    k: 16,
    threads: 32,
};

/// The `mma.sync.m16n8k32` e4m3 / e5m2 product a warp issues.
pub const MMA_SYNC_F8: MmaShape = MmaShape {
    m: 16,
    n: 8,
    k: 32,
    threads: 32,
};

/// The `wgmma.m64nNk8` TF32 product a warpgroup issues (`N` ≤ 256).
pub const WGMMA_TF32: MmaShape = MmaShape {
    m: 64,
    n: 256,
    k: 8,
    threads: 128,
};

/// `wgmma.m64nNk16` (bf16 / f16) and `m64nNk32` (fp8): 32 bytes of `k` a
/// row, like the TF32 form's 8 × 4.
pub const WGMMA_16: MmaShape = MmaShape {
    k: 16,
    ..WGMMA_TF32
};
pub const WGMMA_F8: MmaShape = MmaShape {
    k: 32,
    ..WGMMA_TF32
};

/// `tcgen05.mma` (`cta_group::1`): one thread issues an `M = 128` product
/// whose accumulator lives in tensor memory, `N` ≤ 256, 32 bytes of `k`.
pub const TCGEN05_16: MmaShape = MmaShape {
    m: 128,
    n: 256,
    k: 16,
    threads: 1,
};

const fn op(shape: MmaShape, fmt: MmaFmt, a: Home, b: Home, acc: Home) -> MatrixOp {
    MatrixOp {
        shape,
        fmt,
        a,
        b,
        acc,
        acc_u: MMA_ACC_U,
        acc_g: MMA_ACC_G,
        mn_major: false,
    }
}

const fn warp(shape: MmaShape, fmt: MmaFmt) -> MatrixOp {
    op(shape, fmt, Home::Regs, Home::Regs, Home::Regs)
}

/// The warp products every SM with `mma.sync` issues: TF32 m16n8k8, bf16 /
/// f16 m16n8k16 (operands and accumulator in registers).
pub const MATRIX_WARP: [MatrixOp; 3] = [
    warp(MMA_SYNC_TF32, MmaFmt::Tf32),
    warp(MMA_SYNC_16, MmaFmt::Bf16),
    warp(MMA_SYNC_16, MmaFmt::F16),
];

/// [`MATRIX_WARP`] and the fp8 warp products (m16n8k32 e4m3 / e5m2).
pub const MATRIX_WARP_F8: [MatrixOp; 5] = [
    MATRIX_WARP[0],
    MATRIX_WARP[1],
    MATRIX_WARP[2],
    warp(MMA_SYNC_F8, MmaFmt::E4m3),
    warp(MMA_SYNC_F8, MmaFmt::E5m2),
];

const fn group(shape: MmaShape, fmt: MmaFmt) -> MatrixOp {
    op(shape, fmt, Home::Smem, Home::Smem, Home::Regs)
}

/// wgmma-16bit: a 16-bit warpgroup product as the PTX ISA states its
/// geometry — either descriptor operand MN-major or K-major (the transpose
/// bits), `a` from shared memory or, `rs`, from the warpgroup's registers in
/// the `m64k16` fragment layout (an operand loaded wider is rounded to the
/// format in its load).
const fn group16(fmt: MmaFmt, rs: bool) -> MatrixOp {
    let a = if rs { Home::Regs } else { Home::Smem };
    MatrixOp {
        mn_major: true,
        ..op(WGMMA_16, fmt, a, Home::Smem, Home::Regs)
    }
}

/// [`MATRIX_WARP_F8`] and the warpgroup products: `a` and `b` from
/// shared-memory descriptors, the accumulator in registers; the 16-bit
/// forms also with `a` from registers (wgmma-16bit).
pub const MATRIX_WARPGROUP: [MatrixOp; 12] = [
    MATRIX_WARP_F8[0],
    MATRIX_WARP_F8[1],
    MATRIX_WARP_F8[2],
    MATRIX_WARP_F8[3],
    MATRIX_WARP_F8[4],
    group(WGMMA_TF32, MmaFmt::Tf32),
    group16(MmaFmt::Bf16, false),
    group16(MmaFmt::F16, false),
    group16(MmaFmt::Bf16, true),
    group16(MmaFmt::F16, true),
    group(WGMMA_F8, MmaFmt::E4m3),
    group(WGMMA_F8, MmaFmt::E5m2),
];

/// [`MATRIX_WARP_F8`] and the single-thread products whose accumulator lives
/// in tensor memory (`a` from shared memory or tensor memory, `b` from
/// shared memory). Modeled, not emitted: no rented part has one yet.
pub const MATRIX_TMEM: [MatrixOp; 7] = [
    MATRIX_WARP_F8[0],
    MATRIX_WARP_F8[1],
    MATRIX_WARP_F8[2],
    MATRIX_WARP_F8[3],
    MATRIX_WARP_F8[4],
    op(TCGEN05_16, MmaFmt::Bf16, Home::Smem, Home::Smem, Home::Tmem),
    op(TCGEN05_16, MmaFmt::F16, Home::Smem, Home::Smem, Home::Tmem),
];

impl Caps {
    /// **The matrix instruction a product of `fmt` operands takes with `a`
    /// from `a`** — the first the part lists (warp products before
    /// warpgroup ones). `None`: the part has no such shape.
    pub fn matrix_op(&self, fmt: MmaFmt, a: Home) -> Option<MatrixOp> {
        self.matrix
            .iter()
            .copied()
            .find(|o| o.fmt == fmt && o.a == a)
    }

    /// wgmma-16bit: **the warpgroup product of `fmt` operands with `a` from
    /// `a`** — `b` from a shared-memory descriptor, the accumulator in
    /// registers. `None`: the part issues no such shape.
    pub fn matrix_group(&self, fmt: MmaFmt, a: Home) -> Option<MatrixOp> {
        self.matrix.iter().copied().find(|o| {
            o.fmt == fmt
                && o.a == a
                && o.b == Home::Smem
                && o.acc == Home::Regs
                && o.shape.threads > 32
        })
    }
}

/// `cp.async`, warp `mma` with `ldmatrix`, `f64` reductions: the capability
/// set the RTX 3090 and A100 profiles spell (compute capability 8.0 / 8.6).
pub const CAPS_WARP_MMA: Caps = Caps {
    cp_async: true,
    matrix: &MATRIX_WARP,
    ldmatrix: true,
    griddepcontrol: false,
    red_f64: true,
    math: MATH_SM_TARGET,
    bulk: None,
    reg_realloc: None,
    cvt_f8: None,
    l2_evict: true,
    ptx: ptx_of_target(80, false),
};

/// [`CAPS_WARP_MMA`] with the fp8 warp products: the RTX 4090 (8.9).
pub const CAPS_WARP_MMA_F8: Caps = Caps {
    matrix: &MATRIX_WARP_F8,
    cvt_f8: Some(CVT_F8_SM89),
    ptx: ptx_of_target(89, false),
    ..CAPS_WARP_MMA
};

/// [`CAPS_WARP_MMA_F8`] with `griddepcontrol`: the consumer Blackwell SM (the
/// RTX 5090 and 5080 profiles).
pub const CAPS_WARP_MMA_PDL: Caps = Caps {
    griddepcontrol: true,
    bulk: Some(BULK_SM90),
    cvt_f8: Some(CVT_F8_SM90),
    ptx: ptx_of_target(120, false),
    ..CAPS_WARP_MMA_F8
};

/// [`CAPS_WARP_MMA_PDL`] with the warpgroup `wgmma` products: the
/// arch-specific Hopper target (the H100 profiles).
pub const CAPS_WARPGROUP_MMA: Caps = Caps {
    matrix: &MATRIX_WARPGROUP,
    reg_realloc: Some(REG_REALLOC_SM90),
    ptx: ptx_of_target(90, true),
    ..CAPS_WARP_MMA_PDL
};

/// **The PTX ISA's availability table, applied once to a device query's
/// compute capability** (`major·10 + minor`, and whether the arch-specific
/// `a` target is taken): the only place a capability is read off a version —
/// the native profile's builder. `cp.async`, `mma.sync` TF32 / bf16 / f16 and
/// `ldmatrix` with it: sm_80; `mma.sync` e4m3 / e5m2: sm_89; `griddepcontrol`:
/// sm_90; `wgmma`: `sm_90a` alone; `tcgen05`: `sm_100a`; `atom.add.f64`:
/// sm_60; the bulk-copy engine (TMA, `mbarrier` transaction counts, clusters):
/// sm_90.
pub const fn caps_of_cc(cc: u64, arch_specific: bool) -> Caps {
    let matrix: &'static [MatrixOp] = if cc < 80 {
        &[]
    } else if cc < 89 {
        &MATRIX_WARP
    } else if cc == 90 && arch_specific {
        &MATRIX_WARPGROUP
    } else if (cc == 100 || cc == 103) && arch_specific {
        &MATRIX_TMEM
    } else {
        &MATRIX_WARP_F8
    };
    Caps {
        cp_async: cc >= 80,
        matrix,
        ldmatrix: cc >= 80,
        griddepcontrol: cc >= 90,
        red_f64: cc >= 60,
        math: MATH_SM_TARGET,
        bulk: if cc >= 90 { Some(BULK_SM90) } else { None },
        // `setmaxnreg`: the arch-specific targets from 9.0 (spike: `sm_120a`
        // takes it, `sm_120` refuses it).
        reg_realloc: if cc >= 90 && arch_specific {
            Some(REG_REALLOC_SM90)
        } else {
            None
        },
        cvt_f8: if cc >= 90 {
            Some(CVT_F8_SM90)
        } else if cc == 89 {
            Some(CVT_F8_SM89)
        } else {
            None
        },
        l2_evict: cc >= 20,
        ptx: ptx_of_target(cc, arch_specific),
    }
}
