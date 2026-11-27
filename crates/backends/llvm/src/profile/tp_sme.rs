//! `TargetProfile`: the SME tile and the packing rules.

use super::*;

impl TargetProfile {
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
