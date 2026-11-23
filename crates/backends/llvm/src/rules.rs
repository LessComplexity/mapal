//! hardware-properties: **the GPU rung rules** — each rung's shipped value at
//! one site, computed from the site's geometry against the part (its fill:
//! warps per scheduler of its grid on the part's SMs, [`Gpu::warps_per_scheduler`]),
//! its face, and the part's physical properties ([`Gpu`]: SMs, registers,
//! shared memory and L2 per SM, the FFMA and DRAM rates, measured launch costs,
//! capabilities). No rule reads a part's name, architecture or version, and no
//! part has its own row: two parts with the same properties take the same
//! rungs, and a part nobody measured takes the rungs its properties give.
//!
//! Each rule's property and threshold separate the parts where the cited
//! measurements found the rung winning from those where it lost
//! (`rung_replay.tsv`, replayed by the `rung_rules_replay` test;
//! `benches/results/evidence/local-rtx-5080/hardware-properties.md`). A
//! threshold sits between the nearest measured winner and loser. Where every
//! measured part agrees within a (fill, face) class the rule is that class's
//! value; where no part measured a gain the rung is off on every part (its ON
//! arm measures it). The `--gpu-off=` / `--gpu-on=` arms force a rung for a
//! measurement; `--gpu-off=arch-rungs` takes every rung (the S52 baseline).

use crate::GpuRungs;
use crate::profile::{Face, GPU_RESIDENT_BLOCKS, Gpu};

/// DRAM bytes per µs per SM ([`Gpu::dram_bytes_per_us_per_sm`]) below which
/// each SM's share of the operand stream is scarce enough that more FFMA per
/// loaded byte pays: the bigger register tile (`wide-fma`) and, under-filled,
/// the `cp.async` panel (`async`, `kmajor-a`). Between the rtx-4090 (7 875,
/// all three measured winning — at its 2.52 GHz rating and again on the
/// cross-arch box capped to a sustained 1.96 GHz: wide-fma 17 W / 0 L,
/// kmajor-a 10 / 0, async 3 / 0) and the rtx-5090 (10 541, async / kmajor-a
/// lost, wide-fma mixed); the RTX 3090 / 5080 (11 414), A100 (14 398), H100
/// PCIe (17 885) and SXM (25 393) lost or ship off. It replaces the FFMA per
/// DRAM byte (`BALANCE_PM` 70 ‰): the capped 4090's measured clock put it
/// at 63.6 ‰, under that threshold, while every rung still won — the clock
/// is not what separates the parts; the DRAM rate an SM shares is (both are
/// the same per-SM FFMA rate on every part but the A100).
pub const DRAM_PER_SM_LOW: u64 = 9_200;

/// **open-rules `wide-fma`, priced at the site**: whether the whole-file
/// `8 × 16` tile (128 accumulators a thread, half the warps of the `8 × 8`
/// tile over the same `bm × bn` block) pays at a site whose wide grid is
/// `blocks` blocks (its split partitions included) against `wave`, the
/// blocks one wave of the wide tile holds (`sms ×` its blocks per SM). The
/// part's side is [`rungs`] (128 FP32 lanes, and a scarce DRAM share or a
/// contracting face). Where an SM's DRAM share is below [`DRAM_PER_SM`]
/// (the RTX 3090 / 5080 11.4 KB/µs, the 5090 10.5, the 4090 7.9) the wide
/// tile pays at any grid: cross-arch 7a2b67bf, 3090 12 W / 0 L (mlp_2048
/// 0.934 at 1.56 waves, the stream-K projections 0.937–0.973), 4090 19 / 0.
/// Above it (the H100 SXM 25.4) a grid of at most one wave runs each block's
/// one tile with half the warps to keep the SM's DRAM share in flight:
/// mlp_2048 / mlp2_2048 (256 blocks against 264) 1.037 / 1.052,
/// pj_m128_n28672 (224) 1.167 lost; more blocks than a wave (the stream-K
/// m2048 projections, 512–3 584 blocks: 0.934–0.964; 4096³ at 1 024:
/// loop / epi 0.970 / 0.973) won.
pub fn wide_fma_pays(g: &Gpu, blocks: u64, wave: u64) -> bool {
    g.dram_bytes_per_us_per_sm() < DRAM_PER_SM || blocks > wave
}

/// DRAM bytes per µs per SM ([`Gpu::dram_bytes_per_us_per_sm`]) at and above
/// which a streaming one-thread-per-cell grid's spread over the SMs decides
/// its cells (`vec-cells`, [`vec_cells_pay`]): the A100 (14 398) and H100 SXM
/// (25 393) lost 3–17 % to the vector grid wherever it held the SMs less
/// evenly than the scalar grid at the same split; the RTX 3090 / 5080 (11 414)
/// and 4090 (7 875) did not separate on it. The midpoint of 11 414 and 14 398.
pub const DRAM_PER_SM_SPREAD: u64 = 12_900;

/// **The spread of a grid of `blocks` over `sms` SMs**, per mille: the most
/// blocks one SM holds over the mean (`⌈blocks / sms⌉ · sms / blocks`; 1000 =
/// every SM the same).
pub fn grid_spread_pm(blocks: u64, sms: u64) -> u64 {
    let sms = sms.max(1);
    blocks.div_ceil(sms) * sms * 1000 / blocks.max(1)
}

/// **regressions `vec-cells`, priced at the site**: whether a streaming
/// site's vector cells (`v` cells a thread, `vec_blocks` blocks at the split)
/// pay against one cell a thread at the same split (`scalar_blocks`, `v`
/// times the blocks). Where an SM's DRAM share is at or above
/// [`DRAM_PER_SM_SPREAD`] the vector grid must spread over the SMs no less
/// evenly than the scalar grid (cross-arch 3a8e654f, OFF / ON of the six
/// m = 1 projections: A100 126 / 112 / 112 / 112 vector blocks on 108 SMs,
/// spread 1.71–1.93 against the scalar grid's 1.07–1.21, lost 0.93–0.97;
/// 192 blocks, 1.125 both, won 1.035; H100 144 / 192 / 224 blocks on 132,
/// 1.18–1.83 against 1.03–1.15, lost 0.83–0.96; 256 blocks, 1.03 both,
/// null). Below it the vector grid's 4× fewer load instructions decide (RTX
/// 5080 1.03–1.42 won at every spread; RTX 3090 null).
pub fn vec_cells_pay(g: &Gpu, vec_blocks: u64, scalar_blocks: u64) -> bool {
    g.dram_bytes_per_us_per_sm() < DRAM_PER_SM_SPREAD
        || grid_spread_pm(vec_blocks, g.sms) <= grid_spread_pm(scalar_blocks, g.sms)
}

/// FFMA flops per DRAM byte, per mille, at and above which `warp-fill`'s
/// shrunk tile pays at one warp per scheduler on a contract face: the
/// rtx-5080 (54.9) won it (loopmm_1024_8, smlp_4096_256_1024 1.15-1.24x the
/// unshrunk tile); the RTX 3090 (38.0) and H100 PCIe (25.1) lost it.
pub const FILL_BALANCE_PM: u64 = 45_000;

/// Registers per resident thread ([`Gpu::regs_per_resident_thread`]) at or
/// below which `.minnctapersm` from the derived residency caps a filled tile's
/// registers into spills (H100 PCIe, A100: 32; measured losing or null
/// filled) — 42 on the 1536-thread SMs, where it paid (RTX 3090 FMA faces).
pub const REGS_PER_RESIDENT_THREAD: u64 = 32;

/// Shared memory per SM at and above which, with
/// [`REGS_PER_RESIDENT_THREAD`] registers a resident thread, the derived
/// residency holds enough blocks that `min-blocks` caps an under-filled
/// contract tile too (H100 PCIe 228 KiB lost; A100 164 KiB won).
pub const SMEM_PER_SM_MANY_BLOCKS: u64 = 200 * 1024;

/// L2 bytes per SM ([`Gpu::l2_bytes_per_sm`]) below which a deeper panel
/// (`panel-regs`, `grid-share`) or an exact-face workspace split (`splitk`)
/// costs: the RTX 3090's 75 KiB lost every one; the next part up, the A100,
/// holds 388 KiB.
pub const L2_PER_SM: u64 = 128 * 1024;

/// DRAM bytes per µs per SM ([`Gpu::dram_bytes_per_us_per_sm`]) at and above
/// which a retained site's one-row pass pays for the split that fills the
/// part (`row-split`): the H100 PCIe (17.9 KB/µs a SM) won 1.41–1.56×; the
/// RTX 3090 (11.4), 5090 (10.5) and 4090 (7.9) lost. The midpoint of 11.4
/// and 17.9.
pub const DRAM_PER_SM: u64 = 14_600;

/// The host's pageable-to-pinned copy rate, per mille
/// ([`crate::profile::DeviceCost::pageable_pm`]), below which page-locking a
/// host array copied in a loop pays (`pinned`): boxes copying pageable at
/// 0.35–0.84 of pinned won, boxes at 0.98–1.0 lost or were null.
pub const PAGEABLE_PM: u64 = 900;

/// cross-arch `row-split`: the dependent operations a split adds to a
/// retained row's pass — the zero pass before it, and its own launch where
/// the unsplit pass rides in a merged grid (kernel-merging `horizontal`; a
/// split never merges). The H100 SXM's decode: 6144 launches against 4098
/// unsplit, one more a site and trip, beside its memset.
pub const ROW_SPLIT_OPS: u64 = 2;

/// **cross-arch `row-split`, priced at the site**: whether splitting a
/// retained row's pass over the part saves more than the operations it adds.
/// The unsplit pass is one SM streaming the row's `row_bytes` (its `k · c`
/// operand, L2-resident: [`crate::profile::DeviceCost::sm_bytes_per_ns`]);
/// split, the part's DRAM rate streams it; the split adds [`ROW_SPLIT_OPS`]
/// dependent launches ([`crate::profile::DeviceCost::dep_launch_ns`]).
/// Unmeasured (either rate or the launch 0): not priced — the property rule
/// ([`DRAM_PER_SM`]) alone. The H100 SXM's decode rows (256 × 256 floats,
/// 1.7 µs a pass alone) against 3.4 µs of launches lost 0.81–0.87; the
/// H100 PCIe's 1.41–1.56 wins predate the merged grids and its launches
/// are unmeasured.
pub fn row_split_pays(g: &Gpu, row_bytes: u64) -> bool {
    let d = g.cost.unwrap_or(crate::profile::NO_PLAN);
    if d.sm_bytes_per_ns == 0 || d.dep_launch_ns == 0 {
        return true;
    }
    let all = d.dram_bytes_per_ns.max(g.dram_peak_bytes_per_ns()).max(1);
    let saved = (row_bytes / d.sm_bytes_per_ns).saturating_sub(row_bytes / all);
    saved > ROW_SPLIT_OPS * d.dep_launch_ns
}

/// **cross-arch `pdl`: whether a module's launches form a chain a
/// programmatic launch pays on.** Each dependent launch saves
/// `dep_launch_ns − pdl_launch_ns` (H100 SXM 988 ns, rtx-5080 164 ns); a
/// program pays a fixed cost at the head of its chain (the waits it opens
/// every kernel with, the launch path: rtx-5080 mlp_512 / convmc / fir
/// +2.0 / +1.2 / +0.7 µs, H100 fir / pj_m1_n4096 / mlp_512 +1.5 / +1.4 /
/// +3.4 µs; `calib.cu head`'s empty chains show none, so it is not the
/// launch call). Straight-line programs issue one to four launches: at most
/// a few saves against that head, and they lost on both parts (5080 0 W /
/// 3 L, H100 0 W / 6 L beyond null). Launches repeated by a loop (`repeats`:
/// a device site inside a canonical loop — the program's chain grows with
/// its trips) won on both (5080 decode / causal / gen / prefill 0.91–0.92;
/// H100 the same four 0.79–0.93, the model decodes, loopmm, rt_mlp2_2048).
pub fn pdl_pays(g: &Gpu, repeats: bool) -> bool {
    let d = g.cost.unwrap_or(crate::profile::NO_PLAN);
    g.has_griddepcontrol() && d.pdl_launch_ns > 0 && d.pdl_launch_ns < d.dep_launch_ns && repeats
}

/// FP32 lanes per SM below which the exact face's fully unrolled panel loop
/// costs (`unroll`): the A100's 64-lane SM lost on the exact face.
pub const FULL_FP32_LANES: u64 = 128;

/// **The rungs a GPU part ships at a site of fill `wps` (warps per
/// scheduler) on `face`.** Every rung not named here is on.
pub fn rungs(g: &Gpu, wps: u64, face: Face) -> GpuRungs {
    let under = wps < GPU_RESIDENT_BLOCKS;
    let filled = !under;
    let exact = face == Face::Exact;
    let contract = !exact;
    let tf32 = face == Face::Tf32;
    let scarce = g.dram_peak_bytes_per_ns() > 0 && g.dram_bytes_per_us_per_sm() < DRAM_PER_SM_LOW;
    let fill_balance = g
        .ffma_per_dram_byte_pm()
        .is_some_and(|b| b >= FILL_BALANCE_PM);
    let regs_tight = g.regs_per_resident_thread() <= REGS_PER_RESIDENT_THREAD;
    let small_l2 = g.l2_bytes_per_sm() < L2_PER_SM;
    let cost = g.cost.unwrap_or(crate::profile::NO_PLAN);
    // `cp.async` panels: under-filled where an SM's DRAM share is scarce (the
    // S53 4090 1024 cell 0.070 vs 0.101 ms; tile-efficiency 22 W / 0 L on the
    // async path); filled, register staging wins on every measured part.
    let cp_async = under && scarce;
    GpuRungs {
        affine: (exact && under) || (contract && filled),
        cp_async,
        // tile-efficiency T2: the k-major panel pays on the `cp.async` path
        // and loses on a register-staged copy (3090 / H100 / 5090, the
        // 4090 filled) — the panel the async rule picks.
        kmajor_a: cp_async,
        // large-gemm / open-rules: the whole-file `8 × 16` tile — the
        // 128-accumulator thread's lower shared-load issue per FFMA (6
        // `LDS.128` per 128 FFMA against 4 per 64) — on a part whose SM
        // issues 128 FP32 lanes (the A100's 64-lane SM issues half the FFMA
        // per shared load: the lower LDS issue buys nothing, 7a2b67bf 9 L /
        // 2 W, one in an 18 % null), where each SM's DRAM share is scarce or
        // on a contracting face; the site prices its grid
        // ([`wide_fma_pays`]). FFMA per DRAM byte did not separate the
        // parts: the 3090 (35.6) won it 12 / 0, the H100 (18.3) won and lost
        // by grid. The 3090's exact face lost it.
        wide_fma: g.fp32_lanes >= FULL_FP32_LANES && (scarce || contract),
        min_blocks: match (tf32, filled) {
            (true, false) => false,
            (_, true) => !regs_tight,
            (false, false) if exact => true,
            (false, false) => !(regs_tight && g.smem_per_sm >= SMEM_PER_SM_MANY_BLOCKS),
        },
        fill_rows: !(exact && under && regs_tight),
        pad: !(exact && filled),
        idx32: contract && filled,
        // warp-fill: the shrunk tile wins where the grid leaves the
        // schedulers empty (0 warps each: H100 / 3090 2-3x) and, at one warp
        // each, on a compute-rich part (the 5080: loopmm / smlp 1.15-1.24x);
        // at one warp it lost on the H100 / 3090 (25 / 38 flop/B), and
        // filled on the contract faces (the 5090's TF32, the 5080's
        // 16 x 18432 x 7168). The exact face kept it everywhere.
        fill: exact || wps == 0 || (wps == 1 && fill_balance),
        wide: filled,
        // `row-fit` ships on every part; `wave-gate` gates an under-filled
        // site's fitted grid by its own geometry fact
        // ([`crate::profile::wave_full`]: blocks against the SMs their waves
        // span). A filled site keeps its fitted grid: the split that follows
        // fills the part (the 5080's 16 x 2048 x 7168: 16 fitted blocks x
        // S = 4 beat the cell tile 1.33-1.66x; the 5090's 144 blocks kept,
        // 0.46 vs 2.07 ms).
        row_fit: true,
        wave_gate: under,
        grid_share: !small_l2,
        panel_regs: !small_l2,
        splitk: !(exact && small_l2),
        unroll: !(exact && g.fp32_lanes < FULL_FP32_LANES),
        row_split: g.dram_bytes_per_us_per_sm() >= DRAM_PER_SM,
        // The part's side of `pdl` ([`pdl_pays`] adds the module's chain).
        pdl: pdl_pays(g, true),
        // cross-arch: a tile kernel lets its dependent launch at its entry
        // (`griddepcontrol.launch_dependents`) only under this arm: the
        // dependent's blocks then sit resident beside the producer's; the
        // exit trigger keeps the launch overlap (rtx-5080 decode 0.905 exit
        // vs 0.918 entry, the rest within 1 %).
        pdl_early: false,
        // TMA `bulk-stage` (gemv-tma.md): the TF32 mma tile's panels by
        // row bulk copies lost on the RTX 5080 (4.3–14.5×: one copy per
        // 64–128-byte row, the engine's issue per copy against the 16-byte
        // `cp.async` of every thread); no measured part found a gain, so off
        // on every part (its ON arm measures it; open: one tensor box a
        // panel, a host-encoded descriptor).
        bulk_stage: false,
        pinned: cost.pageable_pm > 0 && cost.pageable_pm < PAGEABLE_PM,
        // stream-K: filled (its forgone filled cells won on the H100 and the
        // 3090); under-filled it lost on the 5080's skinny projections and
        // the 5090's 16 x 2048 x 7168 — site-level wave quantization is the
        // open fact (both sides have measured exceptions).
        streamk: filled,
        // No measured part found a gain (rung_replay.tsv): off on every
        // part; the ON arm (or a site candidate, `CHOICE_AXES`) measures it.
        linear: false,
        async_h2d: false,
        async_stage: false,
        bank_plan: false,
        fit_first: false,
        l2_window: false,
        plan_cost: false,
        retain_delta: false,
        rowfold_layout: false,
        streams: false,
        tf32_radix: false,
        ..GpuRungs::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{
        DeviceCost, GPU_PARTS, Machine, RTX_4090, RTX_5080, RTX_5090, TargetProfile, resolve,
        wave_full,
    };
    use mapal_ir::Ty;

    /// The fill of a square `n`-cell's base FMA grid on `p` (the class the
    /// arch-select sweeps keyed a measured size by).
    fn wps_square(p: &TargetProfile, n: u64) -> u64 {
        let g = p
            .gpu_tile(n, n, n, &Ty::f32(), false, false)
            .expect("divisible square site");
        let blocks = (n / g.bm()) * (n / g.bn());
        p.gpu()
            .expect("gpu")
            .warps_per_scheduler(blocks, g.threads())
    }

    /// One measurement the rules pick against (`rung_replay.tsv`).
    struct Entry<'a> {
        rung: &'a str,
        part: &'a str,
        winner: &'a str,
        cond: &'a str,
    }

    /// The measurements whose winner no property the rules read separates —
    /// the open "missing fact" items, each named in
    /// `local-rtx-5080/hardware-properties.md`. The replay must find exactly
    /// these: a rule change that fixes one moves it out, one that breaks a
    /// measurement fails here.
    const OPEN: [(&str, &str, &str, &str); 19] = [
        // The launch does not read whether its operands are in L2 (flushed
        // L2 wins, warm L2 and the composed MLP lose): `residence_plan`.
        ("asyncstage", "h100-pcie", "on", "l2-flushed"),
        ("asyncstage", "rtx-5090", "on", "l2-flushed"),
        // Site-level: the 64-row convolution's fitted grid wins, the
        // 16-row projection's loses on the same part.
        ("fit-first", "rtx-4090", "on", "conv-site"),
        // At a filled site the fitted grid is kept (the split that follows
        // fills the part): the 144-block grids' tail waves (0.56 / 0.63 of
        // the SM slots) are not refused — the gate needs the split count,
        // which is chosen after the tile (open).
        ("row-fit", "h100-pcie", "off", "blocks=144"),
        ("row-fit", "rtx-4090", "off", "blocks=144"),
        // The fold's strided-read cost vs the tile's column store (1-2 %
        // of device time, one face each).
        ("rowfold-layout", "a100", "on", "-"),
        ("rowfold-layout", "rtx-4090", "on", "-"),
        // Site-level wave quantization: stream-K wins some cells of a
        // class and loses others on the same part.
        ("streamk", "rtx-3090", "off", "-"),
        ("streamk", "rtx-3090", "on", "-"),
        ("streamk", "rtx-4090", "off", "-"),
        ("streamk", "rtx-5090", "on", "-"),
        // The filled tile lowers the split count on the free order: the
        // order axis is not a rule input.
        ("warp-fill", "rtx-3090", "off", "free-order"),
        // Measured against one thread a cell before cell-fill; the gate now
        // yields the 8-block grid to the cell tile, which the row-fit line
        // measured 4x faster: re-measure the gate against the cell tile.
        ("wave-gate", "h100-pcie", "off", "blocks=8"),
        // open-rules: the S5x register-staged tile on the H100 PCIe lost
        // the wide tile above one wave (2048^2, 4096^2); the SXM part's
        // current tile won 4096^3 at 3.88 waves (7a2b67bf). Re-measure the
        // PCIe part on the current tile.
        ("wide-fma", "h100-pcie", "off", "blocks=256,wave=228"),
        ("wide-fma", "h100-pcie", "off", "blocks=1024,wave=228"),
        // cross-arch: `min-blocks` won filled on the H100 SXM and the A100
        // (32 registers a resident thread, where the H100 PCIe's cap spilled
        // 8-16 B and lost) — the derived residency's register cap against
        // the tile's own registers, per site, is the open fact (`rules.md`).
        ("min-blocks", "h100", "on", "-"),
        ("min-blocks", "h100", "on", "-"),
        ("min-blocks", "a100", "on", "-"),
        // regressions `vec-cells`: the H100's 4096 x 14336 projection won
        // the vector grid (224 blocks on 132 SMs, spread 1.18 against the
        // scalar grid's 1.03) by 3.5 %, where 2048 x 7168 and 28672 x 4096
        // at the same grids were null / lost: the spread does not separate
        // it (its 32-trip partitions against 8 and 64).
        ("vec-cells", "h100", "on", "blocks=224,scalar=896"),
    ];

    /// **hardware-properties: every cited measurement, replayed.** For each
    /// line of `rung_replay.tsv` with a winner, the rules over that part's
    /// physical properties (and the line's host condition) must pick it at
    /// every fill and face the line covers — except the [`OPEN`] ones.
    #[test]
    fn rung_rules_replay() {
        let mut picked = 0;
        let mut null = 0;
        let mut mixed = 0;
        let mut open = Vec::new();
        // Per rung: (picked, null, mixed, open) — the evidence file's table.
        let mut per: std::collections::BTreeMap<&str, [u32; 4]> = Default::default();
        for line in include_str!("rung_replay.tsv")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let f: Vec<&str> = line.split('\t').collect();
            let [rung, part, fill, face, winner, cond, _source] = f[..] else {
                panic!("7 columns: {line}")
            };
            let e = Entry {
                rung,
                part,
                winner,
                cond,
            };
            assert!(crate::GpuRungs::NAMES.contains(&rung), "{line}");
            let p = GPU_PARTS
                .iter()
                .find(|p| p.name == part)
                .unwrap_or_else(|| panic!("{line}: not a replayed part"));
            let mut g = *p.gpu().expect("gpu");
            // A condition the part's profile does not carry: a measured
            // property of the box (`name=value`, a `--cost=` name: the host's
            // pageable rate, a capped clock), or a site's geometry
            // (`row_bytes=`, `repeats=`), comma-separated.
            let mut row_bytes = None;
            let mut repeats = None;
            let mut blocks = None;
            let mut wave = None;
            let mut scalar = None;
            for kv in cond.split(',') {
                let Some((k, v)) = kv.split_once('=') else {
                    continue;
                };
                match k {
                    "row_bytes" => row_bytes = Some(v.parse::<u64>().expect("row_bytes")),
                    "repeats" => repeats = Some(v == "1"),
                    "blocks" => blocks = Some(v.parse::<u64>().expect("blocks")),
                    "wave" => wave = Some(v.parse::<u64>().expect("wave")),
                    "scalar" => scalar = Some(v.parse::<u64>().expect("scalar")),
                    _ => {
                        let n: u64 = v.parse().expect(line);
                        let mut over = [None; DeviceCost::NAMES.len()];
                        over[DeviceCost::NAMES
                            .iter()
                            .position(|x| *x == k)
                            .unwrap_or_else(|| panic!("{line}: {k}"))] = Some(n);
                        g = g.with_cost(&over);
                    }
                }
            }
            let fills: Vec<u64> = match fill {
                "U0" => vec![0],
                "U1" => vec![1],
                "U" => vec![0, 1],
                "F" => vec![2, 4, 64],
                "A" => vec![0, 1, 2, 4, 64],
                n => vec![wps_square(
                    p,
                    n.trim_start_matches('n').parse().expect(line),
                )],
            };
            let faces: Vec<Face> = face
                .chars()
                .map(|c| match c {
                    'E' => Face::Exact,
                    'C' => Face::Contract,
                    'T' => Face::Tf32,
                    _ => panic!("{line}: face"),
                })
                .collect();
            let count = per.entry(rung).or_default();
            let want = match winner {
                "on" => true,
                "off" => false,
                "null" => {
                    null += 1;
                    count[1] += 1;
                    continue;
                }
                "mixed" => {
                    mixed += 1;
                    count[2] += 1;
                    continue;
                }
                w => panic!("{line}: winner {w}"),
            };
            let all = fills.iter().all(|&wps| {
                faces.iter().all(|&fc| {
                    // `fold-lanes` over a column-major operand: the layout
                    // guard in `rowfold_lanes` declines whatever the rung.
                    let guard = !(e.rung == "fold-lanes" && e.cond == "colmajor");
                    let r = rungs(&g, wps, fc);
                    let mut v = r.get(e.rung) && guard;
                    // The site's side of the priced rules.
                    if let (Some(b), "row-split") = (row_bytes, e.rung) {
                        v = v && row_split_pays(&g, b);
                    }
                    if let (Some(rep), "pdl") = (repeats, e.rung) {
                        v = pdl_pays(&g, rep);
                    }
                    // The wide tile's grid against its wave.
                    if let (Some(b), Some(w), "wide-fma") = (blocks, wave, e.rung) {
                        v = v && wide_fma_pays(&g, b, w);
                    }
                    // The vector cells' grid against the scalar grid's
                    // spread at the same split.
                    if let (Some(b), Some(sc), "vec-cells") = (blocks, scalar, e.rung) {
                        v = v && vec_cells_pay(&g, b, sc);
                    }
                    // A row-fitted grid's blocks: where `wave-gate` gates,
                    // its fact decides whether the fitted grid runs
                    // (`row-fit`) or yields (`wave-gate`).
                    if let (Some(b), None) = (blocks, wave) {
                        let full = wave_full(b, g.sms);
                        v = match e.rung {
                            "row-fit" => v && (!r.wave_gate || full),
                            "wave-gate" => v && !full,
                            _ => v,
                        };
                    }
                    v == want
                })
            });
            match all {
                true => {
                    picked += 1;
                    count[0] += 1;
                }
                false => {
                    open.push((e.rung, e.part, e.winner, e.cond));
                    count[3] += 1;
                }
            }
        }
        for (rung, [p, n, m, o]) in &per {
            eprintln!("replay {rung}\t{p}\t{n}\t{m}\t{o}");
        }
        assert_eq!(open, OPEN.to_vec(), "measurements the rules do not pick");
        // The counts the evidence file reports.
        assert_eq!((picked, null, mixed, open.len()), (122, 71, 10, 19));
    }

    /// No rule reads a name, an architecture string or a version: a part
    /// renamed and re-targeted takes the same rungs, and the hand profiles of
    /// one SM with different properties are decided by those properties.
    #[test]
    fn rungs_read_properties_never_names() {
        for p in GPU_PARTS {
            let g = *p.gpu().expect("gpu");
            let mut other = *p;
            other.name = "renamed";
            other.machine = Machine::Gpu(crate::profile::Gpu {
                arch: "sm_999",
                ..g
            });
            for wps in [0, 1, 2, 4, 64] {
                for face in [Face::Exact, Face::Contract, Face::Tf32] {
                    assert_eq!(p.gpu_rungs_on(wps, face), other.gpu_rungs_on(wps, face));
                }
            }
        }
        // The 5080 and 5090 share an SM; the rules see their properties.
        let b = RTX_5090.gpu().expect("gpu");
        let f = RTX_5080.gpu().expect("gpu");
        assert!(b.dram_bytes_per_us_per_sm() >= DRAM_PER_SM_LOW);
        assert!(f.dram_bytes_per_us_per_sm() >= DRAM_PER_SM_LOW);
        assert!(RTX_4090.gpu().unwrap().dram_bytes_per_us_per_sm() < DRAM_PER_SM_LOW);
        // cross-arch: the 4090 box capped to a sustained 1.96 GHz takes the
        // same rungs as the part at its 2.52 GHz rating — no rule reads the
        // clock alone.
        let capped = crate::profile::Gpu {
            clock_khz: 1_955_131,
            ..*RTX_4090.gpu().unwrap()
        };
        for wps in [0, 1, 2, 4, 64] {
            for face in [Face::Exact, Face::Contract, Face::Tf32] {
                let (a, c) = (
                    rungs(RTX_4090.gpu().unwrap(), wps, face),
                    rungs(&capped, wps, face),
                );
                assert_eq!(
                    (a.wide_fma, a.cp_async, a.kmajor_a),
                    (c.wide_fma, c.cp_async, c.kmajor_a)
                );
            }
        }
        // pdl reads the measured launches: the 5080's programmatic launch
        // (486 ns) is cheaper than its plain one (650 ns); the 5090 has
        // none measured.
        assert!(RTX_5080.gpu_rungs(0, true).pdl);
        assert!(!RTX_5090.gpu_rungs(0, true).pdl);
        // `pinned` reads the host's measured pageable rate.
        assert!(RTX_5080.gpu_rungs(0, true).pinned);
        assert!(!RTX_5090.gpu_rungs(0, true).pinned);
        // Every rung the rules keep off on every part is off on any part.
        let generic = resolve("cuda-ada").expect("profile");
        for n in [
            "async-h2d",
            "asyncstage",
            "l2-window",
            "retain-delta",
            "streams",
        ] {
            assert!(!generic.gpu_rungs(2, true).get(n), "{n}");
        }
    }
}

#[cfg(test)]
mod native_tests {
    use crate::profile::Face;
    use crate::profile::{
        CostOver, DeviceCost, DeviceQuery, NATIVE_GPU, RTX_5080, calib_line, cuda_query,
    };

    /// The local box's calibration line (committed beside its evidence).
    const CALIB: &str =
        include_str!("../../../../benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib");

    /// The RTX 5080's device query, as `cuDeviceGetAttribute` reports it
    /// (`calib.cu` header and `devq.txt`, 2026-11-19 / 21).
    fn rtx_5080_query() -> DeviceQuery {
        DeviceQuery::from_attrs(&[
            (1, 1024),
            (10, 32),
            (13, 2_640_000),
            (16, 84),
            (36, 15_001_000),
            (37, 256),
            (38, 67_108_864),
            (39, 1536),
            (75, 12),
            (76, 0),
            (81, 102_400),
            (82, 65536),
            (97, 101_376),
            (106, 24),
            (111, 1024),
        ])
    }

    /// **hardware-properties: the native GPU profile is the device query and
    /// the calibration file** — on the 5080's facts it is the hand spelling's
    /// physical properties, so it takes the same rungs at every fill and
    /// face; a line keyed by other facts never prices it.
    #[test]
    fn native_gpu_is_the_query_and_the_calibration() {
        let q = rtx_5080_query();
        assert_eq!(q.calib_key(), "cc=120 sms=84 l2=67108864 mem=15001000x256");
        let p = q.profile(Some(CALIB));
        assert_eq!(p.name, NATIVE_GPU);
        let (g, h) = (p.gpu().expect("gpu"), RTX_5080.gpu().expect("gpu"));
        assert_eq!(*g, *h, "the query + calibration spell the hand profile");
        for wps in [0, 1, 2, 4, 64] {
            for face in [Face::Exact, Face::Contract, Face::Tf32] {
                assert_eq!(p.gpu_rungs_on(wps, face), RTX_5080.gpu_rungs_on(wps, face));
            }
        }
        // Another part's facts: no line, no measured costs, the query's
        // clock — and the rules still decide from what is there.
        let other = DeviceQuery { sms: 170, ..q };
        let o = other.profile(Some(CALIB));
        let og = o.gpu().expect("gpu");
        assert_eq!(
            (og.cost, og.clock_khz, og.fp32_lanes),
            (None, 2_640_000, 128)
        );
        assert!(!o.gpu_rungs(0, true).pdl && !o.gpu_rungs(0, true).pinned);
        // A line's names are the `--cost=` names; an unknown one rejects it.
        let over: CostOver = calib_line("k\tpdl_launch_ns=1 fp32_lanes=64", "k").expect("line");
        let at = |n: &str| over[DeviceCost::NAMES.iter().position(|x| *x == n).unwrap()];
        assert_eq!((at("pdl_launch_ns"), at("fp32_lanes")), (Some(1), Some(64)));
        assert!(calib_line("k\tbogus=1", "k").is_none());
    }

    /// On a machine with a CUDA device: device 0's query builds a profile
    /// whose facts are the driver's; on the local RTX 5080 it is the hand
    /// spelling's. Without a driver, nothing to check.
    #[test]
    fn native_gpu_reads_the_driver() {
        let Some(a) = cuda_query::attrs(0) else {
            return;
        };
        let q = DeviceQuery::from_attrs(&a);
        assert!(q.sms > 0 && q.l2_bytes > 0 && q.regs_per_sm > 0 && q.mem_bus_bits > 0);
        if q == rtx_5080_query() {
            let p = q.profile(Some(CALIB));
            assert_eq!(*p.gpu().expect("gpu"), *RTX_5080.gpu().expect("gpu"));
        }
    }
}
