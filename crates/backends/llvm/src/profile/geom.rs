//! Mma depth picking, `Depth`, and the tunable geometry constants (`Geom`).

use super::*;

/// **f32-tensor-cores R-T2 `mma-depth`: the panel depth that keeps the most
/// operand bytes in flight a block** — among the tensor tile's panel
/// candidates at its residency (each `bk` whose two stages fit the block's
/// shared-memory share, with as many stages as the share holds), the one
/// whose `stages − 1` panels in flight hold the most bytes (Little's law: a
/// skinny site's panels stream from DRAM, and a deeper pipeline covers more of
/// the load latency than a deeper panel); a tie keeps the deeper panel (fewer
/// barriers a `k`). The RTX 5080's 128 × 4096 × 4096 TF32 site: `bk` 32 × 2
/// stages (18 KiB in flight) against `bk` 16 × 5 (37 KiB).
pub fn mma_depth_pick(cands: &[GpuMmaTile]) -> Option<&GpuMmaTile> {
    cands
        .iter()
        .max_by_key(|c| ((c.stages - 1) * c.stage_bytes(), c.bk))
}

/// The panel-depth rule a tensor tile takes ([`TargetProfile::gpu_mma16_tile`]
/// and its TF32 siblings): the S52 panel (the deepest two stages fit),
/// f32-tensor-cores `mma-depth`'s most bytes in flight
/// ([`mma_depth_pick`]), or gapfix tensor-tile's `mma-fly` ring on a
/// DRAM-bound site ([`mma_fly_pick`]; the S52 panel elsewhere).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    Panel,
    InFlight,
    /// `mma-fly`; `walk`: the raster rule the site's walk takes — `None`
    /// row-major (`--gpu-off=raster`), `Some((gate, elem))` [`Gpu::raster`]
    /// under `l2-gate` `gate` at the site's element bytes.
    Fly {
        walk: Option<(bool, u64)>,
    },
}

impl Depth {
    /// The rule a site's rungs name: `mma-depth` (a measurement arm) over
    /// `mma-fly` over the S52 panel; `elem`: the site's element bytes (its
    /// walk's raster gate).
    pub fn of(rungs: &crate::GpuRungs, elem: u64) -> Self {
        match (rungs.mma_depth, rungs.mma_fly) {
            (true, _) => Depth::InFlight,
            (false, true) => Depth::Fly {
                walk: rungs.raster.then_some((rungs.l2_gate, elem)),
            },
            (false, false) => Depth::Panel,
        }
    }
}

/// gapfix tensor-tile (C3) `mma-fly`: **the ring that keeps a DRAM-bound
/// tile's loads in flight through its wait** — among the tile's panel
/// candidates over its residencies, those whose blocks on an SM keep at
/// least `fly` bytes of DRAM-streamed panels in flight while each waits
/// (`wait_group(stages − 2)` leaves `stages − 2` panels in flight a block:
/// `blocks_per_sm · (stages − 2) · ` the stage's DRAM bytes, `dram` = the
/// element bytes of `a` / `b` where that operand streams from DRAM), the
/// most blocks an SM (the register residency's warps where one of its rings
/// keeps `fly`), then the deepest panel (fewest barriers a `k`); `None`
/// where no candidate keeps `fly` in flight. A two-stage ring keeps none:
/// it drains to its last panel at every wait (`DEPBAR.LE SB0, 0x0`). The
/// RTX 5080's 128 × 28672 × 4096 bf16 projection (gaps-disasm C3, local
/// ABBA `gapfix-tensor-tile.md`): `bk` 32 × 2 at 2 blocks 0.400 ms, `bk`
/// 16 × 4 at 2 blocks (this pick, `DEPBAR.LE SB0, 0x2`) 0.343, `bk` 32 × 5
/// at 1 block 0.372 (Triton's four stages at one block: 0.313). `fly`:
/// [`Gpu::stream_fly_per_sm`] (Little's law at the part's DRAM rate,
/// measured: `calib.cu bw`).
pub fn mma_fly_pick(
    cands: &[GpuMmaTile],
    dram: (Option<u64>, Option<u64>),
    fly: u64,
) -> Option<&GpuMmaTile> {
    let stage_dram = |t: &GpuMmaTile| {
        dram.0.map_or(0, |ea| t.bm() * t.bk * ea) + dram.1.map_or(0, |eb| t.bk * t.bn() * eb)
    };
    cands
        .iter()
        .filter(|t| t.blocks_per_sm * t.stages.saturating_sub(2) * stage_dram(t) >= fly.max(1))
        .max_by_key(|t| (t.blocks_per_sm, t.bk))
}

/// gapfix tensor-tile (C11): **the DRAM share of a site's pipe time, per
/// mille, at which its loads stop hiding behind its products**
/// ([`Gpu::dram_bound`]): row-major's `b` re-reads group the walk there. The
/// measured split: the 2048 × 28672 × 4096 projections' row-major traffic
/// (0.93 of the pipe on the RTX 5080) lost to the grouped walk on the 5080,
/// the RTX 4090 and the A100 (13–33 %, `gapfix-tensor-tile.md`), where the
/// A100's / H100's square TF32 tiles (0.54 / 0.69) keep row-major, as every
/// part's square FMA tiles (the 3090's and 4090's arms null).
pub const DRAM_BOUND_PM: u64 = 860;

/// How many blocks the thread budget is split across on one SM: two, so one
/// computes while the other waits at the panel barrier. Policy, like
/// [`Sme::panel_l1d_ratio`] — not a property of the silicon.
pub const GPU_RESIDENT_BLOCKS: u64 = 2;

/// **A GPU part's launch-geometry constants (ADR-0034 D1)** — the per-target
/// table the offline search (D2, `benches/gpu/knobs.json`) writes; a measurement
/// arm replaces one with `--geom=<knob>=<value>` ([`Geom::with`]). Every knob is
/// value-invariant: it re-partitions independent cells, or a fold the face
/// leaves free, never an output bit. A count of 0 is "derived" — the rule that
/// computes it from the site's extents and the part's facts — and a forced
/// value the site cannot take falls back to that rule. [`GEOM`] is the seed
/// row: today's literals and derivations, byte for byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    /// Warps each scheduler should hold: the FMA and `mma` tiles split the
    /// resident-thread budget across this many blocks, and the fill rule
    /// ([`Gpu::fill_warps`]) aims at it ([`GPU_RESIDENT_BLOCKS`]).
    pub resident: u64,
    /// Contiguous partitions of a cut `k` fold (`splitk`, `splitk-atomic`);
    /// 0 = [`TargetProfile::split_count`]. Ignored where it does not divide
    /// the fold's panels.
    pub split: u64,
    /// The FMA tile's candidates (site-geometry B4: the 128-accumulator
    /// `8 × 16` tile, `cp.async` stages, the panel depth at `K = 1024`):
    /// a thread's register tile `tm × tn`, warps per block, the panel depth
    /// `bk` and its stages (1 = register staging, ≥ 2 = `cp.async`); 0 =
    /// derived ([`TargetProfile::gpu_tile_split`], [`TargetProfile::gpu_tile_async`]).
    pub tm: u64,
    pub tn: u64,
    pub warps: u64,
    pub bk: u64,
    pub stages: u64,
    /// The `wgmma` tile's candidates: rows of one warpgroup's instruction
    /// (`n`) and warpgroups per block (T1: the grid's last wave); 0 = derived.
    pub wg_n: u64,
    pub wg_groups: u64,
    /// large-gemm `k-fill`: the fewest panels a stream-K CTA of the full
    /// tile folds before an under-filled site takes it; 0 = derived
    /// (`func::gpu::K_FILL_PANELS`). A measurement knob: the emulator's
    /// value tests reach the rule at extents it runs.
    pub kfill: u64,
}

/// The seed row: every part's table before a search wrote one.
pub const GEOM: Geom = Geom {
    resident: GPU_RESIDENT_BLOCKS,
    split: 0,
    tm: 0,
    tn: 0,
    warps: 0,
    bk: 0,
    stages: 0,
    wg_n: 0,
    wg_groups: 0,
    kfill: 0,
};

/// A `--geom=` override per knob, in [`Geom::NAMES`] order (`None` = the
/// table's value).
pub type GeomOver = [Option<u64>; Geom::NAMES.len()];

impl Geom {
    pub const NAMES: [&'static str; 10] = [
        "resident",
        "split",
        "tm",
        "tn",
        "warps",
        "bk",
        "stages",
        "wg_n",
        "wg_groups",
        "kfill",
    ];

    /// The knob called `name`, `None` for an unknown name.
    pub fn knob(&mut self, name: &str) -> Option<&mut u64> {
        Some(match name {
            "resident" => &mut self.resident,
            "split" => &mut self.split,
            "tm" => &mut self.tm,
            "tn" => &mut self.tn,
            "warps" => &mut self.warps,
            "bk" => &mut self.bk,
            "stages" => &mut self.stages,
            "wg_n" => &mut self.wg_n,
            "wg_groups" => &mut self.wg_groups,
            "kfill" => &mut self.kfill,
            _ => return None,
        })
    }

    /// This row with the overrides in `over` applied.
    pub fn with(mut self, over: &GeomOver) -> Self {
        for (name, v) in Self::NAMES.iter().zip(over) {
            if let Some(v) = v {
                *self.knob(name).expect("a listed knob") = *v;
            }
        }
        self
    }
}
