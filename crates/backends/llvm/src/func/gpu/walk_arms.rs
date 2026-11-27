//! Measured bf16 `mma-walk` arms replayed by the tests.

use super::*;

/// bf16 tensor-tile `mma-walk`: **one measured walk** — the part, the bf16
/// site `m × n × k`, the ordered tile `(bm, bn)` and its `k` cut, the walk's
/// tile `(bm, bn, bk, blocks an SM)`, its grid `(ctas, iters)` (0: the
/// part's own, as [`mma_walk`] derives it), the residual epilogue's extra
/// bytes, and the ordered grid's (or its cut's) / the walk's µs, each with
/// its zero pass. `mma_walk_replay` replays every one; the rung table's
/// `walk=` lines (`rules::rung_rules_replay`) read theirs.
#[cfg(test)]
pub(crate) struct WalkArm {
    pub part: &'static TargetProfile,
    pub mnk: [u64; 3],
    pub ordered: [u64; 3],
    pub tile: [u64; 4],
    pub grid: (u64, u64),
    pub extra: u64,
    pub ord: f64,
    pub walk: f64,
}

#[cfg(test)]
impl WalkArm {
    /// The arm's price on `gpu` (the part's, or a replay's with measured
    /// box properties) and the part's bf16 matrix rate.
    pub fn price(&self, gpu: &Gpu) -> (MmaWalk, u64) {
        let rate = gpu
            .caps
            .matrix_op(MmaFmt::Bf16, Home::Regs)
            .map(|op| gpu.matrix_flops_per_ns(&op))
            .expect("a bf16 product");
        let ([m, n, k], [om, on, cut], [bm, bn, bk, bps]) = (self.mnk, self.ordered, self.tile);
        let tiles = (m / bm) * (n / bn);
        let panels = k / bk;
        let (ctas, iters) = match self.grid {
            (0, _) => {
                let iters = (tiles * panels).div_ceil(gpu.sms * bps);
                ((tiles * panels).div_ceil(iters), iters)
            }
            g => g,
        };
        let price = MmaWalk {
            ordered: ((m / om) * (n / on), 2 * om * on * k),
            cut,
            tiles,
            panels,
            panel_flops: 2 * bm * bn * bk,
            blocks_per_sm: bps,
            ctas,
            iters,
            tile_cells: bm * bn,
            cells: m * n,
            bytes: m * k * 2 + k * n * 2 + m * n * 4 + self.extra,
            grid_x: n / bn,
            slices: (bm * bk * 2, bn * bk * 2),
        };
        (price, rate)
    }

    /// The walk won beyond [`MMA_WALK_PM`].
    pub fn won(&self) -> bool {
        self.walk < MMA_WALK_PM as f64 / 1000.0 * self.ord
    }
}

#[cfg(test)]
pub(super) const fn walk_arm(
    part: &'static TargetProfile,
    mnk: [u64; 3],
    ordered: [u64; 3],
    tile: [u64; 4],
    grid: (u64, u64),
    extra: u64,
    (ord, walk): (f64, f64),
) -> WalkArm {
    WalkArm {
        part,
        mnk,
        ordered,
        tile,
        grid,
        extra,
        ord,
        walk,
    }
}

/// The measured walks: `bf16-tensor-tile.md` (the RTX 5080's model
/// projections, CUPTI medians, the first eight), hopper `disasm-rented.md`
/// (the RTX 3090's suite row and the A100's down projection, same box ON /
/// `--gpu-off=mma-walk`; the 3090's PTX JIT on the RTX 5080 at its grid),
/// hopper-wgmma `walk-reuse` (three RTX 5080 projections past its L2 keep,
/// ABBA × 3 ON / OFF, CUPTI medians; the OFF arm the rules' cut where the
/// ordered tile's cut is 4; then three suite programs, ABBA × 4, the
/// walk forced by `--gpu-on=mma-walk` against the rules' grid or cut).
#[cfg(test)]
pub(crate) const WALK_ARMS: [WalkArm; 17] = {
    use crate::profile::{A100, RTX_3090, RTX_5080};
    const T: [u64; 4] = [128, 128, 32, 2];
    const O: [u64; 3] = [128, 128, 1];
    [
        walk_arm(
            &RTX_5080,
            [512, 2048, 8192],
            [64, 128, 1],
            T,
            (0, 0),
            0,
            (216.0, 164.2),
        ),
        walk_arm(
            &RTX_5080,
            [512, 8192, 2048],
            O,
            T,
            (0, 0),
            0,
            (192.7, 161.2),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 2048, 8192],
            O,
            T,
            (0, 0),
            0,
            (754.1, 626.5),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 2048, 2048],
            O,
            T,
            (0, 0),
            0,
            (193.1, 161.9),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 8192, 2048],
            O,
            T,
            (0, 0),
            0,
            (628.4, 685.1),
        ),
        walk_arm(
            &RTX_5080,
            [512, 512, 2048],
            [32, 64, 1],
            T,
            (0, 0),
            0,
            (18.8, 19.3),
        ),
        walk_arm(
            &RTX_5080,
            [512, 2048, 2048],
            [64, 128, 1],
            T,
            (0, 0),
            0,
            (56.6, 49.6),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 512, 2048],
            [64, 128, 1],
            T,
            (0, 0),
            0,
            (57.0, 48.7),
        ),
        walk_arm(
            &RTX_3090,
            [2048, 4096, 4096],
            O,
            T,
            (0, 0),
            0,
            (1082.0, 1275.0 + 39.0),
        ),
        walk_arm(
            &A100,
            [2048, 2048, 8192],
            O,
            [128, 128, 64, 2],
            (0, 0),
            2048 * 2048 * 4,
            (761.0, 788.0),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 4096, 4096],
            O,
            T,
            (164, 400),
            0,
            (666.6, 628.5 + 21.2),
        ),
        walk_arm(
            &RTX_5080,
            [3072, 4096, 8192],
            O,
            T,
            (0, 0),
            0,
            (1836.0, 1829.2 + 39.8),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 4096, 8192],
            [128, 128, 4],
            T,
            (0, 0),
            0,
            (1297.9 + 21.0, 1184.7 + 21.0),
        ),
        walk_arm(
            &RTX_5080,
            [4096, 2048, 8192],
            [128, 128, 4],
            T,
            (0, 0),
            0,
            (1292.9 + 20.8, 1198.4 + 20.7),
        ),
        walk_arm(
            &RTX_5080,
            [128, 6144, 4096],
            [64, 128, 8],
            T,
            (0, 0),
            0,
            (75.3 + 2.0, 64.0 + 1.9),
        ),
        walk_arm(
            &RTX_5080,
            [128, 28672, 4096],
            O,
            [128, 128, 16, 2],
            (0, 0),
            0,
            (332.8, 325.5 + 6.8),
        ),
        walk_arm(
            &RTX_5080,
            [2048, 6144, 4096],
            O,
            T,
            (0, 0),
            0,
            (929.2, 1005.2 + 37.5),
        ),
    ]
};
