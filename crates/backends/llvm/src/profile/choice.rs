//! Faces, the choice axes, and wave fill.

#[cfg(test)]
use super::*;

/// The face a site is emitted on, as the rung rules read it
/// ([`crate::rules::rungs`]): the conformance face (`mul.rn` + `add.rn`, two
/// instructions per step) and the contract face (one `fma`) hold different
/// registers live, so a staging rung can pay on one and cost on the other.
/// The TF32 face is a contract face whose kernel is `mma`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
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
            (Some(_), _) => Face::Tf32,
            // ADR-0040: the order axis runs the scalar kernels, so it reads
            // the Exact/Contract rules; its realizations are own rungs.
            (None, true) => Face::Contract,
            (None, false) => Face::Exact,
        }
    }
}

/// **per-site-choice B1: a rung whose measured sign is mixed on a part** — it
/// wins at some sites and loses at others, and the fact that separates them is
/// not read yet. Such a rung is not switched off: each site's table holds the
/// rules' plan (candidate 0) and the plan with the rung flipped, and the device
/// measures the pick once (`func/gpu.rs` `site_cands`, `mapal_choice.h`).
/// A rung leaves the list when a fact decides it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChoiceAxis {
    /// The rung flipped against what candidate 0 realized.
    pub rung: &'static str,
    /// per-site-choice D2: what a launch realized of the rung (`async`: a
    /// `cp.async` panel; `kmajor-a`: a k-major panel; a rung no other rule
    /// declines: the rung). The alternative forces the opposite — the forced
    /// arm bypasses a rule that declined it — and is no candidate where the
    /// realized state does not change.
    pub realized: fn(&crate::func::gpu::Launch) -> bool,
    /// Not a flip: `plan-cost` on with every split kernel at the full
    /// register tile (`full-tile`), the cost model's argmin among those.
    pub full: bool,
    /// The evidence that the rung is mixed (for the reader; the measurements
    /// themselves are in `rung_replay.tsv`).
    #[allow(dead_code)]
    pub cite: &'static str,
}

/// [`ChoiceAxis`]es in priority order: the candidates a table keeps under its
/// cap (`--cands`) come first from the front of this list.
pub(crate) const CHOICE_AXES: &[ChoiceAxis] = &[
    ChoiceAxis {
        rung: "async",
        realized: |l| l.tiled.is_some_and(|t| t.stages >= 2),
        full: false,
        cite: "rung_replay.tsv async, rtx-4090: OFF filled (_c 4096 on 3.027 vs off 2.823 ms) but kept under-filled \
               (_c 1024 on 0.070 vs off 0.101 ms); with kmajor-a, 22 W / 0 L on the async path vs 133 L on the staged \
               one (per-site-choice bank.md §3)",
    },
    ChoiceAxis {
        rung: "bank-plan",
        realized: |l| l.rungs.bank_plan,
        full: false,
        cite: "per-site-choice D1: unmeasured on every part (rung_replay.tsv bank-plan); the bank model's k-major panel is \
               conflict-free in 30 / 30 swept geometries (bank.md, build-a.md), its sign on time not yet measured",
    },
    ChoiceAxis {
        rung: "kmajor-a",
        realized: |l| l.tiled.is_some_and(|t| t.kmajor),
        full: false,
        cite: "tile-efficiency sweep (rung_replay.tsv kmajor-a): 4090 async 21 W / 0 L, 4090 staged 8 W / 4 L, \
               3090 staged 0 W / 33 L, H100 3 W / 55 L (per-site-choice bank.md §3)",
    },
    ChoiceAxis {
        rung: "plan-cost",
        realized: |l| l.rungs.plan_cost,
        full: false,
        cite: "plan-cost v2 (benches/results/evidence/plan-cost/v2/summary.tsv): 4090 17 W / 5 L, 5090 9 W / 13 L; \
               blind set 4090 2 W / 2 L, 5090 10 W / 6 L",
    },
    ChoiceAxis {
        rung: "plan-cost",
        realized: |l| l.rungs.plan_cost && l.rungs.full_tile,
        full: true,
        cite: "tile-efficiency re-time (sweep/rungs-detail-*.tsv, on-plan-cost+linear): 4090 programs 8 W / 2 L, \
               5090 11 W / 10 L; the full tile is taken only where the model already prefers it",
    },
    ChoiceAxis {
        rung: "fill-wave",
        realized: |l| l.rungs.fill_wave,
        full: false,
        cite: "docs/performance/evidence/site-geometry.md G3: probes 4090 9 W / 9 L, 5090 4 W / 15 L, H100 5 W / 3 L; \
               4090 losers k/S <= 512, winners k/S >= 1024",
    },
    ChoiceAxis {
        rung: "fit-first",
        realized: |l| l.rungs.fit_first,
        full: false,
        cite: "site-geometry.md G4: 4090 convmc ON 0.0948 vs OFF 0.1107 ms but mm_16x18432x7168 1.053 vs 0.972 ms; \
               OFF rows on 5090 / H100 / 3090",
    },
    ChoiceAxis {
        rung: "row-fit",
        realized: |l| l.rungs.row_fit,
        full: false,
        cite: "rung_replay.tsv row-fit: OFF filled on 4090 / H100 / 3090 (16x18432x7168), kept under-filled; H100 \
               under-filled contract OFF after cell-fill (mm_16x1024x4096 0.4332 vs 0.1018 ms)",
    },
    ChoiceAxis {
        rung: "wave-gate",
        realized: |l| l.rungs.wave_gate,
        full: false,
        cite: "rung_replay.tsv wave-gate: 5090 filled contract OFF (16x18432x7168 on 2.0659 vs off 0.4628 ms), H100 \
               under-filled contract OFF, ON elsewhere",
    },
    ChoiceAxis {
        rung: "wide-fma",
        realized: |l| l.rungs.wide_fma,
        full: false,
        cite: "rung_replay.tsv wide-fma: OFF on H100 / 5090 (register staging 4096 on 4.054 vs off 3.851 ms) but \
               kept on the 4090 filled (8192 on 33.640 vs off 41.168 ms)",
    },
    ChoiceAxis {
        rung: "row-split",
        realized: |l| l.rungs.row_split,
        full: false,
        cite: "site-geometry.md G2: H100 row-split workspace S=128 wins 1.41-1.56; 4090 / 5090 / 3090 lose \
               (decode 36.08 vs 30.65 ms on the 4090)",
    },
];

/// **hardware-properties: whether a grid of `blocks` keeps the part's `sms`
/// busy** — its blocks over the SM slots its waves span, `blocks /
/// (⌈blocks / sms⌉ · sms)`, at least 3/4 (`wave-gate`'s fact, one block an SM
/// a wave). The row-fitted 144-block grid of a 16 × 18432 × 7168 site: 0.85
/// on 170 SMs (kept, 0.46 ms against 2.07 one thread a cell), 0.88 on 82
/// (null), 0.63 on 114 and 0.56 on 128 (both lost to the unfitted kernel);
/// the 8-block grid of 16 × 1024 × 4096 on 114 SMs, 0.07, lost to the cell
/// tile 0.43 vs 0.10 ms (`rung_replay.tsv` row-fit, wave-gate).
pub fn wave_full(blocks: u64, sms: u64) -> bool {
    let sms = sms.max(1);
    blocks * 4 >= blocks.div_ceil(sms) * sms * 3
}

/// The parts the cited rung measurements ran on (`rung_replay.tsv`): the
/// hand-written spellings of rented parts, each carrying its physical
/// properties. A list for tests that sweep parts — no decision reads it.
#[cfg(test)]
pub const GPU_PARTS: [&TargetProfile; 7] = [
    &RTX_4090, &H100_PCIE, &RTX_5090, &RTX_3090, &A100, &RTX_5080, &H100,
];
