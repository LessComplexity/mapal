//! Emission options ([`EmitOpts`]), fold orders and the [`Emission`] result.
use super::*;

/// A reduced-precision face — ADR-0032 D1's lattice below `contract`. Unlike
/// every rung, it **changes values**, by the declared bound of its format, and
/// is what admits the tensor-core realizations: they cannot reproduce an f32
/// fold bit for bit. Interim form of the type-level contract (the `--contract`
/// precedent); the oracle check for it is a bound, not bit-equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduced {
    /// f32 storage, operands rounded to TF32 (10-bit mantissa, `cvt.rna`),
    /// f32 accumulation — `mma.sync.m16n8k8.tf32` on sm_80+.
    Tf32,
    /// ADR-0043 D4: multiplicands rounded to a 16-bit format (`bf16`, `f16`),
    /// f32 accumulation — `mma.sync.m16n8k16` where the part lists it.
    Half(mapal_ir::NarrowFloat),
}

/// The order a tile site's `k` fold may be evaluated in (ADR-0040 D1): the
/// second axis of the float face, beside the steps ([`EmitOpts::contract`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Increasing `k`, one chain per cell: the interpreter's left fold.
    Sequential,
    /// ADR-0040 D3: `S` contiguous partial folds added in partition order,
    /// `S` the profile's; bits a function of `(site, profile)`, replayed by
    /// the interpreter from [`Emission::orders`]. The test/debug face.
    Fixed,
    /// Unspecified (the default): any realization within ADR-0040 D2's bound —
    /// atomic split, stream-K, the fixed split, sequential — per part by
    /// measurement. Run-to-run bits may differ.
    Free,
}

/// One split fold the emitter compiled under [`Order::Fixed`]: the `Fold`
/// morphism `fold` of function `func` (a tile site's map body) folds in `s`
/// contiguous partitions, added in partition order (ADR-0040 D3). The
/// interpreter replays exactly these (`mapal_interp::run_ordered`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldOrder {
    pub func: FuncId,
    pub fold: MorphismId,
    pub s: u64,
}

/// Emission options. [`emit`] delegates to these product defaults.
#[derive(Clone, Copy, Debug)]
pub struct EmitOpts {
    /// Bracket `mapal_main` with the mapal-rt compute timer.
    pub perf_timing: bool,
    /// Tile recognized matmul-shaped map sites.
    pub tiling: bool,
    /// Pack tiled two-dimensional right-hand operands.
    pub packing: bool,
    /// Steps (ADR-0040 D1): single-rounding `fma` per fold step. Default
    /// `true` (the default face); `false` is the separate `fmul` + `fadd` of
    /// the `exact` and `fixed` faces.
    pub contract: bool,
    /// Order of a tile site's `k` fold (ADR-0040 D1); default [`Order::Free`].
    pub order: Order,
    /// Split deep packed tile sites into k-panels — the OpenBLAS (jc, kc, ic)
    /// nest with an A-panel pack and partial sums parked in `out`.
    ///
    /// **Default OFF: measured a 3× LOSS** on M4 Pro at 1024 f32 (S29 —
    /// `fma` 59.8 ms on / 19.8 ms off; the parking traffic outweighs the A
    /// re-read it removes at this size and cache hierarchy).
    ///
    /// **S42: SWEPT on the box, and it loses at every depth — the last argument
    /// for this lever is gone.** It was kept because it "was designed against
    /// BOX-scale traffic (16 GB of A re-reads at 4096 on zen3) where it has not
    /// yet been measured".
    ///
    /// Measured on an i9-14900F (24C/32T, 48 KB L1d, 2 MB per-core L2, 36 MB L3,
    /// AVX2, governor `performance`), N=4096, values identical to the Mac's at
    /// every depth. For `zen3` at f32 the chain is `lanes 8 → tile_j 32 → nc
    /// 1024`, so `tile_kc = l2_bytes / 8192`; the sweep varies `l2_bytes`:
    ///
    /// | depth | blocks | 1 thread | threaded (32) |
    /// | ---: | ---: | ---: | ---: |
    /// | unblocked | 1 | **926.759 ms** | **112.919 ms** |
    /// | 4096 | 1 | 0.978× (gate closed ⇒ *is* unblocked) | 1.000× |
    /// | 2048 | 2 | 0.548× | 0.595× |
    /// | 1024 | 4 | 0.557× | 0.759× |
    /// | 512 | 8 | 0.551× | **0.771×** best |
    /// | 256 | 16 | 0.560× | 0.767× |
    /// | 64 | 64 | 0.454× | 0.668× |
    ///
    /// **A step function, not a curve**: any blocking at all costs ~1.8× at one
    /// thread and ~1.3× at best threaded, at *every* depth from 2 to 64 blocks.
    /// Unlike the SME k-panel — where one wrong depth made the technique look
    /// worthless (`func/sme.rs`) — here there is no optimum to find.
    ///
    /// ⇒ **`kc_nest` has now lost on every machine available, swept rather than
    /// spot-checked.** "Unmeasured on the box" is no longer a reason to keep it.
    /// Either a machine justifies it in writing, or delete the lever.
    ///
    /// A pure performance tailor in ADR-0032's sense: bit-exact either way, which
    /// the differential suite enforces — and which the cross-machine run
    /// independently confirmed, ARM and x86 printing identical values.
    pub kc_nest: bool,
    /// **The move-panel traversal — S44's rung, S45's deduction.** Visits a bulk
    /// `map`'s iteration space in `b × b` tiles of a `w`-wide 2-D geometry
    /// instead of in linear order, when that pays.
    ///
    /// **S44 shipped this as `--move-panel=W:B`: two numbers typed by a human.**
    /// `W` was program geometry the emitter could not see (`tile_site` requires a
    /// fold in the map body, and a transpose has none) and `B` was a swept
    /// machine constant. S45 deleted both: `W` comes from
    /// [`mapal_ir::MoveSite`], a fold-less move-site record; the cache geometry
    /// comes from [`crate::profile::L1d`], detected; and firing and `B` are
    /// arithmetic over the two in [`crate::profile::TargetProfile::move_block`].
    ///
    /// [`MovePanel::Deduce`] is the default and the shipped path. The other two
    /// exist so the transform can still be **measured** — an A/B needs an OFF arm
    /// of the same binary, and a B-sweep needs to force a block — and they carry
    /// no machine constant into any emitted default.
    ///
    /// A pure ADR-0032 D4 performance tailor: the transform is a **permutation of
    /// the loop counter**, so every element is still visited exactly once and
    /// values are bit-identical, which the differential suite enforces.
    pub move_panel: MovePanel,
    /// The machine facts the emitter tiles against, selected **by name**
    /// (plan-s31-target-profiles): `generic` (the pre-S46 literals, byte-
    /// identical on every host), `apple-m`, `apple-m4-sme`, `zen3`,
    /// `raptorlake`, `cuda-ada`.
    ///
    /// **The default is `native` (S46, ratified ADR-0038): probe the host, and
    /// fall back to `generic` when a fact cannot be read.** Default emission is
    /// therefore machine-dependent *by design* — anything that needs
    /// reproducible emitted text across machines (structural tests, artifact
    /// hashes) pins a profile by name, and [`emit_detailed`] reports which
    /// profile was resolved.
    ///
    /// A pure ADR-0032 D4 performance tailor: every profile field is
    /// value-invariant, which the differential suite enforces by running under
    /// a non-default profile.
    pub target: &'static str,
    /// S62 `hostopt`: on a GPU `target`, the CPU profile everything outside a
    /// device site is emitted against — resolved like `target` (`native` by
    /// default, so default emission is machine-dependent here too: tests pin
    /// it). A CPU `target` ignores it; the `emit` CLI rejects the pair.
    pub host: &'static str,
    /// Measurement arms over the GPU rung table; see [`GpuArms`].
    pub gpu: GpuArms,
    /// ADR-0034 D1: `--geom=<knob>=<value>` over the GPU part's geometry table
    /// ([`profile::Geom`]); all `None` = the table.
    pub geom: profile::GeomOver,
    /// plan-cost: `--cost=<rate>=<value>` over the GPU part's plan rates
    /// ([`profile::DeviceCost::NAMES`]); all `None` = the profile's.
    pub cost: profile::CostOver,
}

impl Default for EmitOpts {
    fn default() -> Self {
        Self {
            perf_timing: false,
            tiling: true,
            packing: true,
            // ADR-0040 (Sapir, 2026-10-23): the default face is full
            // precision, fma steps, order unspecified, bounded.
            contract: true,
            order: Order::Free,
            kc_nest: false,
            move_panel: MovePanel::Deduce,
            // S46 (Sapir): the DEFAULT detects the machine. Every rung that
            // needs machine facts asks the profile for them, so a default build
            // now gets the cache-aware ones instead of silently declining.
            // `native` falls back to `generic` when detection fails, and tests
            // that must be reproducible across machines pin `generic` by name.
            target: "native",
            host: "native",
            gpu: GpuArms::default(),
            geom: Default::default(),
            cost: [None; profile::DeviceCost::NAMES.len()],
        }
    }
}

impl EmitOpts {
    /// The `exact` face (`--exact`): separate `fmul` + `fadd`, sequential —
    /// the interpreter's bits. What every bit-for-bit oracle check pins.
    pub fn exact() -> Self {
        Self {
            contract: false,
            order: Order::Sequential,
            ..Self::default()
        }
    }
}

/// One emitted translation unit **plus the profile it was emitted against**
/// (ADR-0038 D2). The default target is `native`, so the text alone does not
/// say which machine facts shaped it — `profile` is the resolved profile's
/// name (`"native"` when detection succeeded, `"generic"` on fallback or by
/// name, or the named profile), and artifact identity is
/// `(source, EmitOpts, profile)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emission {
    /// The `.ll` text — the translation unit itself (ADR-0020).
    pub text: String,
    /// The **device** translation unit (`nvptx64`), when the profile is a GPU
    /// and some tile site offloads: `llc` turns it into the PTX the runtime
    /// loads (`MAPAL_PTX`). `None` on every CPU profile.
    pub device: Option<String>,
    /// The resolved [`TargetProfile`]'s name.
    pub profile: &'static str,
    /// The split folds compiled under [`Order::Fixed`] (ADR-0040 D3), for the
    /// interpreter to replay. Empty on every other order.
    pub orders: Vec<FoldOrder>,
}
