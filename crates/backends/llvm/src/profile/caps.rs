//! Capabilities: `Caps`, PTX ISA levels, register reallocation, bulk copy engines.

use super::*;

/// **The instructions an SM executes, described by what they move and the
/// geometry they compute** (hardware-properties). Read by the realizations
/// that emit them; a part without one never takes that realization. The
/// device query reports a compute capability, and the native profile turns it
/// into these once ([`caps_of_cc`], the PTX ISA's availability table) — no
/// decision reads a version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    /// `cp.async` global→shared copies (4/8/16 B a thread, no register
    /// round trip).
    pub cp_async: bool,
    /// **The matrix instructions the SM issues** (tensor-core): each one's
    /// `m × n × k` per issuing group, the operands' format, where each
    /// operand and the accumulator live when it issues, and the measured
    /// rounding of its accumulator ([`MatrixOp`]). The tile rule maps a site's
    /// operand format and geometry onto whichever of these the part has
    /// ([`Caps::matrix_op`]); empty = none.
    pub matrix: &'static [MatrixOp],
    /// `ldmatrix`: a warp loads 8×8 b16 tiles from shared memory into the
    /// `mma` fragment layout (`.trans`: transposed, a `k × n` panel into the
    /// `b` fragment).
    pub ldmatrix: bool,
    /// `griddepcontrol`: a kernel may start while the one before it drains
    /// and wait at its entry (programmatic dependent launch).
    pub griddepcontrol: bool,
    /// `red`/`atom.add.f64` on global memory.
    pub red_f64: bool,
    /// **The math function units the SM computes with** (ADR-0046 D5/D7):
    /// each unit's function, width, the intrinsic emitted, its error over its
    /// domain (`Exact` for a correctly rounded operation, else the measured
    /// maximum and the calibration line it is read from), its issue cost.
    /// `ex2.approx.ftz.f32` (MUFU.EX2, ADR-0042 D5) and `sqrt.rn` on every
    /// NVIDIA SM ([`MATH_SM`]); the approximate `sqrt` / `sin` / `cos`
    /// units where a calibration measured them ([`MATH_SM_RTX_5080`]). A
    /// builtin's realization is chosen from these by its contract
    /// ([`mapal_ir::math::choose`]) — never by the part's name.
    pub math: mapal_ir::math::Target,
    /// **The bulk-copy engine (TMA)** — global→shared copies one thread
    /// issues, completing on a transaction barrier (`mbarrier` with a byte
    /// count), as the geometry it executes ([`BulkCopy`]); `None`: no such
    /// engine (`cp.async` and register copies only).
    pub bulk: Option<BulkCopy>,
    /// hopper-wgmma-tma: **warpgroup register reallocation** (`setmaxnreg`):
    /// a warpgroup hands registers back to the block's pool or takes them
    /// from it, within the geometry [`RegRealloc`] states; `None`: every
    /// thread keeps the count the block launched with.
    pub reg_realloc: Option<RegRealloc>,
    /// `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` and `cvt.rn.f16x2.{e4m3,e5m2}x2`:
    /// two fp8 conversions an instruction (ADR-0043 slice 3, C1), and the
    /// encoding they need on this part ([`Ptx`]: PTX ISA 7.8 from `sm_90`;
    /// `sm_89` encodes the widening only from PTX 8.1 — the 4090's `_f8w`
    /// modules at llc's default 7.8 were rejected by `ptxas`, cross-arch
    /// 5fcb401b finding 1). `None`: no such instruction — the fp8 roundings
    /// are the host's integer sequence ([`mapal_ir::narrow::f8_from_f32`]).
    pub cvt_f8: Option<Ptx>,
    /// gapfix `stream-ld`: **L2 eviction priority on a load** — the streaming
    /// cache operator `ld.global.cs` (PTX ISA 1.0, `sm_20`; SASS `LDG.E.EF`:
    /// its lines are the L2's first to evict), so bytes read once do not
    /// displace a set the L2 keeps. `false`: every load takes the default
    /// policy.
    pub l2_evict: bool,
    /// bugs-properties `ptx-isa`: **the encoding the part's target itself
    /// needs** — the lowest PTX ISA that names its SM target (`llc` refuses
    /// a lower one: `sm_90a` 8.0, `sm_120` 8.7). Every device module declares
    /// the largest of this and the [`Ptx`] of each capability whose
    /// instructions it emits ([`Ptx::declare`], the `; mapal.ptx` line), and
    /// every compile step passes it to `llc` (`-mattr=+ptx<isa>`).
    pub ptx: Ptx,
}

/// bugs-properties `ptx-isa`: **a PTX encoding** — the PTX ISA version
/// (× 10: 8.1 → 81) and the SM (compute capability × 10) an instruction or a
/// target needs. Ordered by ISA, then SM: a module's declaration is the
/// largest need ([`Ptx::max`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ptx {
    pub isa: u32,
    pub sm: u32,
}

impl Ptx {
    pub const fn new(isa: u32, sm: u32) -> Self {
        Self { isa, sm }
    }

    /// The `llc` feature that sets the module's `.version` (`+ptx81`).
    pub fn mattr(self) -> String {
        format!("+ptx{}", self.isa)
    }

    /// **The device module's declaration line** (`; mapal.ptx isa=81 sm=89
    /// mattr=+ptx81`): what every `llc` step of the module passes, read by
    /// [`Ptx::declared`] (the CLI, tests) and by the bundle scripts (`sed`).
    pub fn declare(self) -> String {
        format!(
            "; mapal.ptx isa={} sm={} mattr={}\n",
            self.isa,
            self.sm,
            self.mattr()
        )
    }

    /// The encoding a device module declares ([`Ptx::declare`]); `None`
    /// where it declares none.
    pub fn declared(device: &str) -> Option<Self> {
        let l = device
            .lines()
            .take(8)
            .find_map(|l| l.strip_prefix("; mapal.ptx "))?;
        let field = |k: &str| {
            l.split(' ')
                .find_map(|w| w.strip_prefix(k))
                .and_then(|v| v.parse().ok())
        };
        Some(Self::new(field("isa=")?, field("sm=")?))
    }
}

/// bugs-properties `ptx-isa`: the encodings of `cvt.rn.f16x2.e4m3x2` and
/// `cvt.rn.satfinite.e4m3x2.f32` (PTX ISA notes): 7.8 on `sm_89` for the
/// rounding, but the widening on `sm_89` only from 8.1 — the module's need
/// there is 8.1; from `sm_90` both at 7.8.
pub const CVT_F8_SM89: Ptx = Ptx::new(81, 89);
pub const CVT_F8_SM90: Ptx = Ptx::new(78, 90);

/// bugs-properties `ptx-isa`: the lowest PTX ISA naming each SM target
/// (`llc`'s default for `-mcpu`, the PTX ISA's target table) — `sm_80` 7.0,
/// `sm_86` 7.1, `sm_87` 7.4, `sm_89` / `sm_90` 7.8, `sm_90a` 8.0, `sm_100`
/// 8.6, `sm_103` / `sm_121` 8.8, `sm_120` 8.7; below `sm_80` 6.0.
pub const fn ptx_of_target(cc: u64, arch_specific: bool) -> Ptx {
    let isa = match cc {
        0..80 => 60,
        80 => 70,
        81..=86 => 71,
        87 => 74,
        88..=89 => 78,
        90 if arch_specific => 80,
        90 => 78,
        91..=100 => 86,
        101..=119 => 88,
        120 => 87,
        _ => 88,
    };
    Ptx::new(isa, cc as u32)
}

/// `setmaxnreg.{inc,dec}`'s geometry: the per-thread counts a warpgroup may
/// set (`min ..= max`, multiples of `step`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegRealloc {
    pub min: u64,
    pub max: u64,
    pub step: u64,
}

/// `setmaxnreg` of the arch-specific targets (`sm_90a`, `sm_100a`,
/// `sm_120a`; PTX ISA 8.0): 24 to 256 registers in steps of 8.
pub const REG_REALLOC_SM90: RegRealloc = RegRealloc {
    min: 24,
    max: 256,
    step: 8,
};

/// **The bulk-copy engine's geometry** (TMA: `cp.async.bulk`,
/// `cp.async.bulk.tensor`, `mbarrier.expect_tx`): what one copy may move and
/// how its completion is counted. Read by the staging rule
/// ([`BulkCopy::box_fits`]): a panel whose rows the geometry covers may be
/// copied by one thread's box (or row) copies instead of every thread's
/// 16-byte `cp.async`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkCopy {
    /// Bytes a copy's size, its global and shared addresses, and a tensor
    /// box's global strides are whole multiples of (16).
    pub granule: u64,
    /// Dimensions a tensor box spans (5) and elements along each (256).
    pub box_dims: u64,
    pub box_side: u64,
    /// Shared-memory swizzle spans a box may be written in, bytes (32, 64,
    /// 128: the XOR pattern of 16-byte chunks over 8 rows) — a swizzled box's
    /// inner dimension is at most its span.
    pub swizzles: &'static [u64],
    /// Bytes one transaction barrier phase counts at most (`2²⁰ − 1`).
    pub tx_bytes: u64,
    /// Thread blocks a cluster holds (portable: 8), whose shared memory a
    /// copy may also write (`multicast`: one copy lands in every block of
    /// the cluster that asks for it).
    pub cluster: u64,
    pub multicast: bool,
    /// hopper-wgmma-tma: bytes of the encoded tensor map a box copy names
    /// (`CUtensorMap`: 128, aligned to 64) — a kernel parameter by value.
    pub desc_bytes: u64,
    /// A box reaching past the tensor's extent lands zero-filled there.
    pub oob_zero: bool,
}

/// The bulk-copy engine of compute capability 9.0 and up (H100 sm_90a,
/// the consumer Blackwell sm_120 — probed on the RTX 5080: `UTMALDG.2D`,
/// `SYNCS.ARRIVE.TRANS64`, clusters of 8, `local-rtx-5080/gemv-stream.md`).
pub const BULK_SM90: BulkCopy = BulkCopy {
    granule: 16,
    box_dims: 5,
    box_side: 256,
    swizzles: &[32, 64, 128],
    tx_bytes: (1 << 20) - 1,
    cluster: 8,
    multicast: true,
    desc_bytes: 128,
    oob_zero: true,
};

impl BulkCopy {
    /// **The staging rule's geometry test: a `rows × row_elems` panel of
    /// `elem`-byte elements, rows `stride` elements apart in global memory,
    /// is one tensor box** — each dimension within the box side, the row
    /// and the stride whole granules, the base aligned to one, the panel's
    /// bytes within one barrier phase. `swizzle`: the box is written swizzled
    /// at that span (its row at most the span).
    pub fn box_fits(
        &self,
        rows: u64,
        row_elems: u64,
        stride: u64,
        base: u64,
        elem: u64,
        swizzle: Option<u64>,
    ) -> bool {
        let g = self.granule;
        let row = row_elems * elem;
        rows > 0
            && rows <= self.box_side
            && row_elems <= self.box_side
            && row.is_multiple_of(g)
            && (stride * elem).is_multiple_of(g)
            && (base * elem).is_multiple_of(g)
            && rows * row <= self.tx_bytes
            && swizzle.is_none_or(|w| self.swizzles.contains(&w) && row <= w)
    }

    /// hopper-wgmma-tma R1: **a box of sides `sides` over a dense tensor of
    /// `dims` (`(extent, stride)` in elements, innermost first, the innermost
    /// unit-stride) is one tensor copy** — at most `box_dims` dimensions, each
    /// side within `box_side` and dividing its extent, the inner row whole
    /// granules and exactly its swizzle span when swizzled (a span the engine
    /// writes), every outer stride and the base whole granules, the box within
    /// one barrier phase.
    pub fn tensor_box_fits(
        &self,
        dims: &[(u64, u64)],
        sides: &[u64],
        base: u64,
        elem: u64,
        swizzle: Option<u64>,
    ) -> bool {
        let g = self.granule;
        let row = sides.first().map_or(0, |s| s * elem);
        let bytes: u64 = sides.iter().product::<u64>() * elem;
        !dims.is_empty()
            && dims.len() == sides.len()
            && dims.len() as u64 <= self.box_dims
            && dims[0].1 == 1
            && dims
                .iter()
                .zip(sides)
                .all(|(&(ext, _), &s)| s > 0 && s <= self.box_side && ext.is_multiple_of(s))
            && row.is_multiple_of(g)
            && dims[1..]
                .iter()
                .all(|&(_, st)| (st * elem).is_multiple_of(g))
            && (base * elem).is_multiple_of(g)
            && bytes <= self.tx_bytes
            && swizzle.is_none_or(|w| self.swizzles.contains(&w) && row == w)
    }
}
