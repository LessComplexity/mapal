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

/// **bugs-properties `streamk-epi`: an epilogue group's stream-K walk keeps
/// its operand panels' reuse.** The walk's CTAs start `iters` units apart,
/// so the blocks that read one `a` / `b` panel read it at skewed `k`: their
/// reuse is the L2's, across the walk, where the ordered grid's lockstep
/// wave shares each panel at once. A part whose L2 share an SM reaches
/// [`L2_PER_SM`] (the `grid-share` property) keeps it; a smaller one keeps
/// it only where the site's operands (`operand_bytes`, `a` and `b`) fit the
/// L2 it keeps ([`Gpu::l2_keep_bytes`]) — else the group keeps its ordered
/// kernel (`--gpu-off=streamk-epi`'s). Cross-arch 5fcb401b, the RTX 3090
/// (75 KiB an SM, 6 MiB): `loop_4096` / `epi_4096` (128 MiB of operands)
/// 0.74 / 0.94, the stream-K kernel 6.0 against the ordered 5.3–5.6 ms with
/// the same k loop (SASS: 2048 FFMA, 96 LDS, 8 LDG a trip, 224 / 230
/// registers); `indep_512` / `mlp_512` (2 MiB) 1.05 / 1.13; the 2048s null;
/// the H100 (388 KiB) 3 W / 5 null, the 5080 (780 KiB) 8 W.
pub fn streamk_epi_pays(g: &Gpu, operand_bytes: u64) -> bool {
    g.l2_bytes_per_sm() >= L2_PER_SM || operand_bytes <= g.l2_keep_bytes()
}

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

/// f32-tensor-cores R-F: the split rate rule's margin, per mille — a split
/// form of `P` part products a k-step pays where the matrix rate its tile
/// realizes ([`split_issue_pm`]) reaches `θ · P` times the part's FFMA rate
/// ([`split_rate_pays`]). tensor-cores-2 places `θ` = 1.6 on the A100's
/// measured split tiles (rented, `local-rtx-5080/tensor-cores-2.md`): issue
/// 1.00 / 0.83 (θ ≤ 2.67 / 3.2) won all 15 (FFMA / split 1.43–1.56), issue
/// 1.81 (θ ≤ 1.47) lost or tied all three (0.83, 0.87, 1.00), issue 1.40
/// (θ ≤ 1.91) is mixed (1.06, 1.11 and the uncut under-filled 128 × 6144
/// 0.81: open). 1.3 before (the TF32 tile's efficiency deficit) took the
/// 1.81 losers. The rtx-5080's tf32x3 loser (θ ≤ 0.33) and narrow1 winner
/// (2.0) stand on either side.
pub const SPLIT_RATE_PM: u64 = 1600;

/// **f32-tensor-cores R-F, the part's side: a split form of `products`
/// passes on `op` pays against the FFMA tile** — the matrix rate the split
/// tile realizes, `matrix_flops_per_ns(op)` over its issue ratio
/// ([`split_issue_pm`], at least 1), `≥ θ · P · ffma_flops_per_ns`
/// ([`SPLIT_RATE_PM`]). Physical properties: the SM's matrix and FFMA rates a
/// clock (`calib.cu matrix`, `ffma`) and its schedulers' issue.
pub fn split_rate_pays(
    g: &Gpu,
    op: &crate::profile::MatrixOp,
    products: u64,
    issue_pm: u64,
) -> bool {
    g.matrix_flops_per_ns(op) * 1_000_000 / issue_pm.max(1000)
        >= SPLIT_RATE_PM * products * g.ffma_flops_per_ns()
}

/// tensor-cores-2 `split-issue`: issue slots a split tile spends a k-step
/// per f32 element its thread cuts — the cut, its guard, the element's
/// fragment load and address share — read off ptxas 13.0's SASS of the
/// 2048 × 4096 × 4096 tiles at sm_80 (`tensor-cores-2.md`; the panel loop's
/// instructions less its `HMMA` and flush `FADD`s, over the elements cut):
/// the integer TF32 cut (792 a 96-`HMMA` panel: 10.4), the 5fcb401b
/// `cvt.rna` cut with its overflow select (1612: 21.7 — the A100's loser),
/// bf16x3n's three bf16 parts (489 a 48-`HMMA` panel: 12.8), bf16x2n's two.
pub const SPLIT_CUT_TF32: u64 = 10;
pub const SPLIT_CUT_TF32_CVT: u64 = 22;
pub const SPLIT_CUT_BF16X3: u64 = 13;
pub const SPLIT_CUT_BF16X2: u64 = 10;

/// **tensor-cores-2 `split-issue`: a split tile's issue against its matrix
/// pipe**, per mille. A warp's k-step issues `P·mt·nt` matrix instructions,
/// `4·mt·nt / flush` round-to-nearest adds and `per_cut` slots for each of
/// the `cut` elements it cuts — one instruction a clock on its scheduler
/// ([`Gpu::schedulers`]); the matrix pipe takes each instruction `2·m·n·k /
/// (matrix_flops_per_clk / schedulers)` clocks. Above 1000 the tile is
/// issue-bound: it realizes the matrix rate over this ratio. The A100's
/// tf32x3 at 5fcb401b: 408 slots against 194 pipe clocks a k-step, 2.10 —
/// 1012 → 481 flops an SM a clock, under the 499 three passes need
/// (measured 0.56–0.79 of the FFMA tile).
pub fn split_issue_pm(
    g: &Gpu,
    op: &crate::profile::MatrixOp,
    products: u64,
    (mt, nt): (u64, u64),
    cut: u64,
    per_cut: u64,
    flush: u64,
) -> u64 {
    let slots = split_slots_pm(products, (mt, nt), cut, per_cut, flush);
    let per_sched = g.matrix_flops_per_clk(op) / g.schedulers();
    let flops = 2 * op.shape.m * op.shape.n * op.shape.k;
    slots * per_sched / flops.max(1)
}

/// [`split_issue_pm`]'s slots a warp issues a matrix instruction, per mille
/// (intent.py `split-tile`: the hot loop's SASS instructions over its `HMMA`).
pub fn split_slots_pm(
    products: u64,
    (mt, nt): (u64, u64),
    cut: u64,
    per_cut: u64,
    flush: u64,
) -> u64 {
    let instrs = (products * mt * nt).max(1);
    let issue = instrs + 4 * mt * nt / flush.max(1) + per_cut * cut;
    issue * 1000 / instrs
}

/// **hopper-launch `ws-persist`, priced at the site**: the time the
/// warp-specialized `wgmma` kernel saves walking its `tiles` on `ctas`
/// persistent CTAs (the part's SMs × the tile's residency) against a CTA a
/// tile, in ns on the busiest SM. Both grids put `⌈tiles / ctas⌉` tiles on
/// it, and the walk's round `r` runs the tiles wave `r` would (round robin in
/// the launch's raster order: the same operand panels in flight, so the L2
/// traffic is the waves'). What changes is each later tile's start: a CTA a
/// tile starts after the previous block's epilogue drained and then fills
/// its pipeline (`fill_ns`: the `stages − 1` box copies ahead of the first
/// `wgmma`, [`Gpu::tma_copy_ns`] each); the walk's producer copies the next
/// tile's stages while the consumers store this one (`epi_ns`: the tile's
/// bytes at the SM's DRAM share). One round — `tiles ≤ ctas` — is the same
/// kernel: nothing saved.
pub fn ws_persist_saves_ns(tiles: u64, ctas: u64, fill_ns: u64, epi_ns: u64) -> u64 {
    tiles.div_ceil(ctas.max(1)).saturating_sub(1) * (fill_ns + epi_ns)
}

/// **hopper-launch `ws-persist`**: the persistent walk where it saves time
/// ([`ws_persist_saves_ns`]) — a grid of more than one round. The site's
/// fill and epilogue are priced from the part's properties
/// ([`Gpu::tma_copy_ns`], [`Gpu::dram_bytes_per_us_per_sm`]).
pub fn ws_persist_pays(g: &Gpu, tiles: u64, ctas: u64, stages: u64, tile_bytes: u64) -> bool {
    let fill = stages.saturating_sub(1) * g.tma_copy_ns();
    let epi = tile_bytes * 1000 / g.dram_bytes_per_us_per_sm().max(1);
    ws_persist_saves_ns(tiles, ctas, fill, epi) > 0
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
        // hopper-launch `ws-persist`: the persistent walk measured mixed on
        // the H100 SXM (two boxes, ABBA, against a CTA a tile): one-consumer
        // tiles (64 columns) won 1.9-4.8 %, the 128 × 256 two-consumer tiles
        // lost 0.3-3.0 % beyond a 0-1.8 % null, with the hot loop's SASS the
        // same (4 HGMMA, 2 UTMALDG a trip; +40 index instructions a kernel).
        // What separates them — the walk's static round robin against the
        // hardware's dynamic placement of a later block, per tile time — is
        // not read yet (`hopper-launch.md`), so a CTA a tile ships and the
        // ON arm measures the walk where [`ws_persist_pays`] prices it.
        ws_persist: false,
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
        // f32-tensor-cores `mma-depth`: the TF32 tile's deepest-in-flight
        // panel at its register-first residency lost every measured site
        // on the rtx-5080 (128 × 4096 × 4096 `bk` 16 × 5 stages 0.143 against
        // `bk` 32 × 2 0.135 ms; 128 × 4096 × 14336 0.490 / 0.454; 128 ×
        // 28672 × 4096 0.722 / 0.696; 2048 × 4096 × 4096 `bk` 8 × 4 1.494 /
        // 1.418; 128 × 6144 × 4096 null): off on every part, its ON arm
        // measures it.
        mma_depth: false,
        ..GpuRungs::ALL
    }
}

/// gapfix `stream-ld` (gaps-disasm C2): **an operand the grid reads once
/// streams past the L2's eviction priority** where its bytes exceed the
/// cyclic set the L2 keeps ([`Gpu::l2_keep_bytes`]) — re-read in order it
/// thrashes the default policy — yet by no more than [`STREAM_KEEP_SHARE`]
/// times, and the part has the policy ([`crate::profile::Caps::l2_evict`]).
/// Evict-first lines do not displace the ones the L2 holds, so a share
/// `keep ÷ bytes` of the operand survives to its next read (the program's
/// next call, a loop's next trip) where the default policy keeps almost none.
/// The RTX 5080 (64.6 MiB keep): the 100 MiB `b` of `pj_m1_n6144_k4096`
/// (share 0.65) 0.084 → 0.073 ms — 1.19 → 1.37 TB/s, past the DRAM's 0.96:
/// L2 hits —, `pj_m16_n6144_k4096` 0.0995 → 0.0787, `pj_m128_n6144_k4096`
/// 0.207 → 0.200; `pj_m1_n4096_k14336` (235 MB, share 0.27) 0.247 → 0.229
/// (0.95 → 1.03 TB/s); the 67 MB `pj_m1_n4096_k4096` under the keep null.
/// The RTX 3090 (6.5 MB keep, shares 0.03–0.11): `pj_m1_n6144_k4096`,
/// `pj_m1_n2048_k7168` null, `pj_m1_n4096_k4096` 1.025 and
/// `pj_m1_n4096_k14336` 1.047 (its GEMV 0.284 → 0.289 ms) — nothing kept,
/// the hint's cost alone (gapfix-rows-operands.md). The SASS moves only the
/// operand's loads to `LDG.E.EF`.
pub fn stream_ld_pays(g: &Gpu, read_once_bytes: u64) -> bool {
    let keep = g.l2_keep_bytes();
    g.caps.l2_evict && read_once_bytes > keep && read_once_bytes <= STREAM_KEEP_SHARE * keep
}

/// gapfix `stream-ld`: the most an operand may exceed the L2's keep by and
/// still stream ([`stream_ld_pays`]): the L2 then keeps at least a quarter of
/// it across reads. Measured: shares 0.27–0.65 win on the RTX 5080, 0.03–0.11
/// null or lose on the RTX 3090.
pub const STREAM_KEEP_SHARE: u64 = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{
        DeviceCost, GPU_PARTS, Home, Machine, MmaFmt, RTX_4090, RTX_5080, RTX_5090, TargetProfile,
        resolve, wave_full,
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
    const OPEN: [(&str, &str, &str, &str); 23] = [
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
        // review-fixes (replay lines for sweep-gemm's and hopper's rungs):
        // `round-store`'s one loss, the 5080's 128 x 28672 x 4096 bf16
        // projection — the stored 2-byte panel fits `bk` 32 x 2 where the
        // f32 panel left 16 x 3 over a 1.33-wave grid streaming its `b` from
        // DRAM: the tail wave's residency x the operand's residence against
        // the stage depth (sweep-gemm §2, `mma-depth`'s split) is the open
        // fact, not the storage.
        ("round-store", "rtx-5080", "off", "blocks=224"),
        // `wgmma-ws` on the H100's 128 x 4096 x 4096: its 256 CTAs at one an
        // SM run 2 waves that each re-read the 64 MB `b` against the 50 MB
        // L2 (S53's 2-3 blocks an SM hold them in one) — the operand's
        // re-read across waves (`raster` answers 1 here) is the open fact
        // (hopper.md open 5; the persistent grid in raster order or R6's
        // split would read `b` once). hopper-launch: still 0.0876 / 0.0941 ms
        // with the launch path fixed; the raster order removes it (the
        // `l2-gate` line below: 0.0822 ms).
        ("wgmma-ws", "h100", "off", "blocks=256"),
        // hopper-launch: the gate keeps row-major where one block's panels
        // fit the L2, but a row-major wave of 132 one-CTA-an-SM tiles reads
        // every `b` panel of the 128-row projections (64-235 MB) and the next
        // wave reads them again; raster wins 5-12 % there and is null on the
        // 2048-row sites. The wave's own panel bytes against the L2 is the
        // open fact (it changes the gate for every tile kernel on every part).
        ("l2-gate", "h100", "off", "-"),
        // tensor-cores-2: the GeForce bf16x2n tile is pipe-bound (issue
        // 0.38) and realizes ~90 % of its bf16 pipe, beating the FFMA tile
        // the rule keeps (2 x 256 against 1.6 x 256); the A100's issue-bound
        // tiles realize ~39 % at issue 1.0. The tensor tile's realized
        // fraction at its issue is the missing fact, not one margin.
        ("split-tile", "rtx-5080", "on", "split=bf16x2n,issue=375"),
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
            let mut split = None;
            let mut operands = None;
            let mut issue = None;
            let mut read_once = None;
            // hopper-launch `ws-persist`: the site's tiles, stage depth and
            // tile bytes (the CTAs are the part's SMs at one a tile).
            let mut tiles = None;
            let mut stages = None;
            let mut tile_bytes = None;
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
                    "split" => split = Some(v),
                    // bugs-properties: an epilogue group's operand bytes
                    // (`a` and `b`).
                    "operands" => operands = Some(v.parse::<u64>().expect("operands")),
                    "issue" => issue = Some(v.parse::<u64>().expect("issue")),
                    "tiles" => tiles = Some(v.parse::<u64>().expect("tiles")),
                    "stages" => stages = Some(v.parse::<u64>().expect("stages")),
                    "tile_bytes" => tile_bytes = Some(v.parse::<u64>().expect("tile_bytes")),
                    // gapfix `stream-ld`: the bytes of an operand the grid
                    // reads once.
                    "read_once" => read_once = Some(v.parse::<u64>().expect("read_once")),
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
                    // f32-tensor-cores `split-tile`: the form's passes
                    // against the part's matrix and FFMA rates.
                    if let (Some(f), "split-tile") = (split, e.rung) {
                        let (kind, fmt) = match f {
                            "tf32x3" => (mapal_ir::split::SplitKind::Tf32x3, MmaFmt::Tf32),
                            "narrow1" => (mapal_ir::split::SplitKind::Narrow1, MmaFmt::Bf16),
                            "bf16x3n" => (mapal_ir::split::SplitKind::Bf16x3n, MmaFmt::Bf16),
                            "bf16x2n" => (mapal_ir::split::SplitKind::Bf16x2n, MmaFmt::Bf16),
                            _ => panic!("{line}: split form {f}"),
                        };
                        let op = g.caps.matrix_op(fmt, Home::Regs).expect(line);
                        let p = mapal_ir::split::SplitForm::of(kind).products();
                        // tensor-cores-2: the realization's issue ratio
                        // (`issue=`, per mille; none: the rate alone).
                        v = v && split_rate_pays(&g, &op, p, issue.unwrap_or(1000));
                    }
                    // gapfix `stream-ld`: the read-once operand against the
                    // L2's keep.
                    if let (Some(b), "stream-ld") = (read_once, e.rung) {
                        v = v && stream_ld_pays(&g, b);
                    }
                    // bugs-properties `streamk-epi`: the group's operands
                    // against the part's L2.
                    if let (Some(b), "streamk-epi") = (operands, e.rung) {
                        v = v && streamk_epi_pays(&g, b);
                    }
                    // The persistent walk's saved fills and epilogues.
                    if let (Some(t), Some(s), Some(b), "ws-persist") =
                        (tiles, stages, tile_bytes, e.rung)
                    {
                        v = v && ws_persist_pays(&g, t, g.sms, s, b);
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
        assert_eq!((picked, null, mixed, open.len()), (165, 81, 16, 23));
    }

    /// hopper-launch `ws-persist`, hand-worked on the H100 SXM: the 2048 ×
    /// 4096 × 4096 site's 256 tiles of 256 × 128 on 132 CTAs walk two rounds
    /// — one later tile a CTA, whose fill (3 copies at `tma_copy_ns` 159) and
    /// epilogue (131 072 B at the SM's DRAM share) the walk overlaps; 132
    /// tiles are one round (nothing saved, a CTA a tile); a part with twice
    /// the SMs takes the 256 in one round too.
    #[test]
    fn ws_persist_prices_the_walk() {
        let h = *crate::profile::H100.gpu().expect("gpu");
        let tile = 256 * 128 * 4;
        let fill = 3 * h.tma_copy_ns();
        assert_eq!(fill, 477);
        let epi = tile * 1000 / h.dram_bytes_per_us_per_sm();
        assert_eq!(ws_persist_saves_ns(256, h.sms, fill, epi), fill + epi);
        assert_eq!(
            ws_persist_saves_ns(1792, h.sms, fill, epi),
            13 * (fill + epi)
        );
        assert_eq!(ws_persist_saves_ns(132, h.sms, fill, epi), 0);
        assert!(ws_persist_pays(&h, 256, h.sms, 4, tile));
        assert!(!ws_persist_pays(&h, 132, h.sms, 4, tile));
        let wide = crate::profile::Gpu {
            sms: 2 * h.sms,
            ..h
        };
        assert!(!ws_persist_pays(&wide, 256, wide.sms, 4, tile));
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
