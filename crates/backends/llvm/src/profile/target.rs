//! `TargetProfile` and its host-tile, move, pricing and rung methods.

use super::*;

/// One target's machine facts, plus the two policy ratios that are honestly
/// search space rather than facts (see the field docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetProfile {
    pub name: &'static str,
    /// The machine class — the capability that selects a realization (S41).
    pub machine: Machine,
    /// One vector register in bytes: NEON 16, AVX2 32, AVX-512 64.
    pub vec_bytes: u64,
    /// Architectural vector register count: NEON 32, AVX2 16, AVX-512 32.
    /// **Not probed** — it is a fact of the ISA, so it arrives with the target
    /// features rather than from a runtime query.
    pub vec_regs: u64,
    /// Policy ratio: how many vector registers one accumulator row occupies,
    /// i.e. how wide a j-tile is measured in registers. This is "how much of
    /// the register file to spend on accumulators" — search space, not a
    /// machine fact (ADR-0034 is the ADR that would search it).
    pub acc_vecs_per_row: u64,
    /// Policy ratio: j-tiles per KC-rung j-block. Same status as
    /// `acc_vecs_per_row`.
    pub nc_tiles: u64,
    /// L2 in bytes, **as reported for one L2 instance** — the budget the KC
    /// rung's k-panel is sized against. Divide by [`Self::l2_cores`] for the
    /// per-core share; see that field for why the distinction is load-bearing.
    pub l2_bytes: u64,
    /// **How many physical cores share one `l2_bytes`.** This field exists
    /// because the old doc comment on `l2_bytes` said "per-core" while
    /// [`APPLE_M`] set it to the 16 MB **shared** P-cluster L2 — a two-word
    /// ambiguity that no consumer had noticed because `tile_kc` only ever wanted
    /// an order of magnitude.
    ///
    /// [`TargetProfile::move_block`]'s cost term divides by it, and there the
    /// difference is the whole answer: 16 MB says a 4 MB array is resident and
    /// the rung should decline, 16 MB / **5** = 3.36 MB says it is not and the
    /// rung should fire — and the M4's measured 1.578× says fire. Detected
    /// (`hw.perflevel0.cpusperl2`, `shared_cpu_list`), never assumed.
    ///
    /// Budgeting **per core** rather than per chip is the right conservatism for
    /// a compiler that emits one binary for both thread counts: S44 §5 measured
    /// that this walk's pressure does not dilute with thread count, so at full
    /// occupancy every core really does get its share and no more.
    pub l2_cores: u64,
    /// Stack ceiling policy: an entry-block block at least this large is placed
    /// in the `mapal_rt_alloc` arena instead of the stack.
    pub heap_min_bytes: u64,
    /// Measured dispatch economics, when someone has measured them. `None`
    /// keeps the pre-plan-s50 behaviour: dispatch every task.
    pub dispatch: Option<DispatchCost>,
    /// The streaming matrix unit, when the part has one. `None` for every
    /// profile that existed before S41 — which is exactly what keeps their
    /// emission byte-identical: the SME realization cannot be selected without
    /// a fact only this field carries.
    pub sme: Option<Sme>,
    /// The L1D geometry the move-panel decision needs (S45). `None` for
    /// [`GENERIC`] and every profile that predates S45 — the same seam
    /// discipline as [`Self::sme`] and [`Self::gpu`], and it is what keeps the
    /// **default** profile's emission byte-identical: the rung cannot fire
    /// without a fact only this field carries, so `--target=generic` emits today's
    /// text for every one of the 171 swept cells.
    ///
    /// ponytail: [`Sme::l1d_bytes`] is the same number for parts that carry
    /// both, pinned equal by `l1d_and_sme_agree_on_the_l1d`. Merging them routes
    /// `sme_kc` through a new `Option` for a field rename, and a working SME leg
    /// is not worth that; the test makes the duplication unable to drift.
    pub l1d: Option<L1d>,
    /// ADR-0046 D7: **the host's math units and definitions** (its IEEE
    /// `sqrt`, [`HOST_MATH`]); a host with none (an accelerator's control
    /// core) states [`mapal_ir::math::Target::NONE`] and runs every
    /// builtin's reference from plain operations.
    pub math: mapal_ir::math::Target,
}

/// The element width a tile site accumulates at. Tile sites are numeric-gated
/// by recognition (`mapal_ir::algo` requires `is_numeric` for the element, the
/// accumulator, the seed and the map target), so the non-numeric arm is
/// unreachable; it returns the f32 width rather than panicking because a wrong
/// tile width is a slow kernel, never a wrong answer.
pub(crate) fn elem_bytes(elem: &Ty) -> u64 {
    match elem {
        Ty::Int { bits, .. } | Ty::Float { bits } => u64::from(*bits).div_ceil(8).max(1),
        Ty::Narrow(n) => n.bytes(),
        _ => 4,
    }
}

impl TargetProfile {
    /// Scalar lanes in one vector register at this element width.
    pub fn lanes(&self, elem: &Ty) -> u64 {
        (self.vec_bytes / elem_bytes(elem)).max(1)
    }

    /// The register micro-kernel's j-tile width in lanes (was `tile_j_for`).
    /// `generic`: f32 `16 / 4 × 4 = 16`, f64 `16 / 8 × 4 = 8` — today's
    /// literals, reproduced.
    pub fn tile_j(&self, elem: &Ty) -> u64 {
        self.lanes(elem) * self.acc_vecs_per_row
    }

    /// Rows per register block (was `TILE_I`, on the matmul rung only — the FIR
    /// window rung reads this same budget via `crate::func::window_subrows`, over a different
    /// quantity over a memory accumulator).
    ///
    /// The `2 ×` is the headroom policy: spend at most HALF the vector file on
    /// accumulators, leaving the rest for the shared b tile, the a splat and
    /// the products. `generic`: `32 / (2 × 4) = 4`. It reproduces S26's swept
    /// result *including its failure* — TI=8 would need 32 accumulator
    /// registers, and that sweep recorded "8 spills: 128 accumulators ≫ 32 NEON
    /// regs".
    pub fn tile_i(&self) -> u64 {
        (self.vec_regs / (2 * self.acc_vecs_per_row)).max(1)
    }

    /// The KC rung's j-block width in lanes (was `tile_nc_for`).
    pub fn nc(&self, elem: &Ty) -> u64 {
        self.tile_j(elem) * self.nc_tiles
    }

    /// The KC rung's k-panel depth (was `TILE_KC`): half of L2 spent on one
    /// (kc × nc) packed-b window. `generic` yields 128 at BOTH widths —
    /// `512 KB/2 ÷ (512 × 4)` and `512 KB/2 ÷ (256 × 8)` — today's single
    /// literal, reproduced. On a 16 MB L2 it yields `kc ≥ K`, so the gate
    /// `site.k > tile_kc` closes and **the KC nest disables itself by
    /// derivation** rather than by a default-off flag (S29/S30's measured
    /// verdict, deduced).
    pub fn tile_kc(&self, elem: &Ty) -> u64 {
        ((self.l2_bytes / 2) / (self.nc(elem) * elem_bytes(elem))).max(1)
    }

    /// The L2 budget **one core** actually gets. See [`Self::l2_cores`].
    pub fn l2_per_core(&self) -> u64 {
        self.l2_bytes / self.l2_cores.max(1)
    }

    /// **The S45 move-panel decision, in one function: fire or not, and with
    /// what block.** `None` declines — the emitter then emits today's flat loop
    /// character-identically.
    ///
    /// This is the join ADR-0032 puts here and nowhere else: [`MoveSite`] is
    /// pure program geometry, [`L1d`] is pure machine fact, and neither crate
    /// may hold both. It replaces `--move-panel=W:B`, where a human held both in
    /// their head and typed the answer.
    ///
    /// ```text
    /// S            = cr · sizeof(elem)                       -- the walk's byte stride
    /// sets_touched = line | S ? sets / gcd((S/line) mod sets, sets) : sets
    /// slots        = sets_touched · ways                     -- lines the walk can reach
    /// lines_live   = min(width, width · S / line)            -- lines it needs at once
    /// FIRE iff  lines_live > slots  AND  read_bytes > l2_per_core
    /// ```
    ///
    /// **Two terms, because one is provably not enough.** `lines_live > slots`
    /// is S44's `pressure > 1` written as what it means — the sweep needs more
    /// lines than it can reach — with no threshold to fit. But pressure predicts
    /// how OFTEN the L1 is defeated and says nothing about what a defeat COSTS:
    /// the i9 at side 512 scores 21.3 and **loses 0.901×** (replicated 0.907×).
    /// A defeat costs real money only when the array being re-swept does not fit
    /// the private level below L1; otherwise the miss is an L2 hit the
    /// out-of-order engine hides. The **read** array is the quantity, not the
    /// whole working set — the conflict is on `a`'s lines, the writes stream —
    /// which also keeps the discriminating case off a knife edge (1 MB against
    /// 2 MB, rather than 2 MB against 2 MB).
    ///
    /// Both terms were tested predictively rather than fitted. M4 side 512 has
    /// pressure 8 and a 1 MB read array: the cost term declines it, S44's
    /// standalone probe claimed **2.09×** there, and the emitted pipeline
    /// measured **every arm overlapping OFF, min-to-min 1.02×** — the term's
    /// prediction, on the machine where it was most at risk
    /// (`benches/results/cpu/memory/deduced-panel-movement/deduced-move-panel.md` §3).
    ///
    /// | case | pressure | read vs L2/core | verdict | measured |
    /// | --- | ---: | --- | --- | --- |
    /// | M4 1024 | 32 | 4 MB > 3.36 MB | **fire, B=16** | 1.578× 1t / 1.993× threaded |
    /// | M4 2048 | 128 | 16 MB > 3.36 MB | **fire, B=8** | 3.19× (probe) |
    /// | M4 128 | 0.5 | — | decline (pressure) | 1.000× |
    /// | M4 512 | 8 | 1 MB < 3.36 MB | decline (cost) | overlapping, 1.02× min-to-min |
    /// | i9 1024 | 85.3 | 4 MB > 2 MB | **fire, B=8** | 2.646× 1t, disjoint |
    /// | i9 2048 | 170.7 | 16 MB > 2 MB | **fire, B=8** | 3.021× 1t, disjoint |
    /// | i9 512 | 21.3 | 1 MB < 2 MB | decline (cost) | **0.901× LOSS** |
    ///
    /// **The block: the geometric mean of two opposing costs, both measured.**
    ///
    /// ```text
    /// floor   = line / sizeof(elem)   -- a block row shorter than a line refetches it
    /// ceiling = slots / 2             -- the block's read lines share the reachable
    ///                                    sets with its write stream
    /// B       = largest divisor of gcd(width, rows) <= sqrt(floor * ceiling)
    /// ```
    ///
    /// Both costs are 1 inside `[floor, ceiling]`; the window is normally EMPTY
    /// (the bounds pull apart), and the product of two opposing multipliers is
    /// minimised at their geometric mean. The divisibility clause is not
    /// rounding — the permutation needs the panel to tile the geometry both ways
    /// or it would need a remainder arm.
    ///
    /// Each bound is a measurement, not an argument. **Floor:** B=8 on the i9
    /// covers 8 of the 16 f32 in a line and measures **15% slower than B=16** at
    /// side 1024, 24% at 2048. **Ceiling:** B=`slots` on the M4 measures **29%
    /// (S44) and 34% (S45) slower threaded** than B=16 at side 1024, and 56% at
    /// side 2048.
    ///
    /// It reproduces the M4's swept optimum at **both** sides — 16 at side 1024
    /// (floor 32, ceiling 16, mean 22) and 16 at side 2048 (floor 32, ceiling 8,
    /// mean 16) — which the previous ceiling-only rule did not: it returned 8 at
    /// side 2048 and measured **15.7% off**.
    ///
    /// **What it does NOT reproduce, and why no version of it could.** The i9's
    /// measured optimum is **128**, and this returns 8. That gap was priced with
    /// counters rather than argued (`benches/results/cpu/memory/deduced-panel-movement`, side 1024, P-core):
    ///
    /// | arm | ms | cycles | L1-dcache-load-misses | dTLB misses | LLC misses |
    /// | --- | ---: | ---: | ---: | ---: | ---: |
    /// | off | 2.526 | 16.2 M | **1 053 802** | 289 | 683 |
    /// | B=8 | 1.072 | 9.75 M | 206 414 | 291 | 259 |
    /// | B=16 | 0.940 | 9.14 M | 499 456 | 282 | 316 |
    /// | B=128 | 0.929 | 8.83 M | **1 053 618** | 287 | 253 |
    ///
    /// **`off` and B=128 miss L1 the same number of times — 1 053 802 against
    /// 1 053 618 — and B=128 is 2.7x faster.** B=8 misses **five times less**
    /// and is *slower* than both. TLB misses are flat at ~290 across every arm,
    /// and LLC misses are ~0 (the 4 MB array fits a 36 MB L3). Instruction
    /// counts are identical across the blocked arms (33.03 M), so the ordering
    /// is pure memory behaviour and IPC rises monotonically with B
    /// (3.39 → 3.61 → 3.74).
    ///
    /// So on that machine **L1 residency is not the binding resource at the
    /// optimum**, and neither is the TLB nor the LLC.
    ///
    /// **S47 swept every legal divisor at four sides on both machines and priced
    /// what IS binding. The answer is not memory-level parallelism**, which was
    /// the standing hypothesis here and is now measured false: at the i9 optimum
    /// `l1d_pend_miss.fb_full` is **3.0% of cycles** and MLP
    /// (`pending / pending_cycles`) is **2.23 — the LOWEST of any arm**. The slow
    /// arms have MORE misses in flight (B=8: 4.00; B=32: 3.66), and the fill
    /// buffers only bind on arms nobody would ship (`off`: 47% fb_full, MLP 5.95).
    /// The binding quantity is **L2 residency**: order the side-2048 arms by
    /// `mem_load_retired.l2_miss` and the ms order falls out with no inversion
    /// across eight arms. Behind it is the L2 prefetcher — a block gives each
    /// column stream a run of `B·w/line` consecutive lines, and below two lines
    /// the streamer has nothing to lock on to.
    ///
    /// **And that is a prefetcher property, which is why no rule here can be
    /// right on both parts.** S47's sweep: the M4's optimum is **16 at every side
    /// that wins**, the i9's is **64–128** — the i9 wants a block 4–8x LARGER
    /// while every fact [`L1d`] and [`TargetProfile`] can read is LARGER on the
    /// M4 (`bytes` 2.67x, `line_bytes` 2x, `sets` 2x, `l2_per_core` 1.68x). Only
    /// `ways` is bigger on the i9, by 1.5x, against an 8x difference in the
    /// answer. **No monotone function of the readable facts orders these two
    /// parts the way the measurement does**, and the quantity that does — how far
    /// a prefetcher reaches on a strided walk — is in no sysctl, no sysfs node
    /// and no ISA rule. The witness is the unblocked arms: the same instruction
    /// stream over the same geometry costs **0.796 ns/element on the M4 and 2.325
    /// on the i9**.
    ///
    /// Both terms above are real on the machine they were measured on and false
    /// on the other. The floor (`line/w`) holds on the i9 and fails on the M4,
    /// where B=16 covers half of a 128 B line — the same half-line case the floor
    /// forbids — and is the optimum, beating B=32 by 15% at side 2048. The
    /// ceiling (`slots/2`) holds on the M4 and fails on the i9, where it is 6 and
    /// every B from 16 to 256 beats B=8. Their geometric mean lands **exactly on
    /// the M4's optimum at both sides** and below the left edge of the i9's
    /// plateau. The residual is **1.15–1.27x of the i9's win**, and it is carried
    /// as a known gap: a per-profile scalar that fixed it would have to be 1 on
    /// the M4 and 4–8 on the i9, i.e. carry the whole answer and derive nothing,
    /// which is the hand-typed flag with extra steps.
    ///
    /// **Open defect S47 found and did not fix (`benches/results/cpu/matmul/block-size`).** At side
    /// **1536 on the M4 this rung fires with B=32 and LOSES 1.75x (1t) / 1.41x
    /// (threaded)** — larger than any i9 shortfall above. It is a fire/decline
    /// defect, not a block one: no block wins there. Both terms vote to fire and
    /// both look right (pressure 24x, a 9.44 MB read array against a 3.36 MB
    /// share); what they cannot see is that the M4's unblocked walk at that width
    /// already costs only 0.594 ns/element — better than side 1024's — so there
    /// is no miss cost to recover, while a non-power-of-two width costs the
    /// permutation two magic-multiply divisions per element that a power-of-two
    /// width gets as shifts (+20–25% instructions, measured). Deciding that trade
    /// needs the price of an unblocked miss on the part, which is the same
    /// unreadable quantity. Sides 512, 1024 and 2048 are unaffected on both
    /// machines.
    pub fn move_block(&self, site: &MoveSite) -> Option<u64> {
        let l1 = self.l1d.as_ref()?;
        let (line, sets, ways) = (l1.line_bytes, l1.sets.max(1), l1.ways());
        let w = elem_bytes(&site.elem);
        let stride = site.cr.checked_mul(w)?;
        // The slow axis must stay inside a line, or there is no line reuse for a
        // block to preserve and this rung has nothing to say about the shape.
        if stride == 0 || site.cq * w >= line {
            return None;
        }
        let touched = if stride.is_multiple_of(line) {
            sets / gcd((stride / line) % sets, sets).max(1)
        } else {
            sets
        };
        // CONFLICT, not capacity — S44's headline, as the gate it always was.
        // A walk that reaches every set is limited by capacity, and capacity is
        // measured FREE on these parts (S43: flat 32 KB → 8 MB). Measured
        // witness: side 1025 has `gcd(4100/128 …)` = no collapse, runs **2.12×
        // FASTER unblocked than side 1024**, and blocking it costs 0.623×. It
        // also scores pressure 1.0009, so without this term it would fire — this
        // is the clause that keeps `lines_live > slots` from being a threshold
        // fitted at its own boundary.
        let slots = touched * ways;
        let lines_live = (site.width * stride / line).min(site.width);
        if touched >= sets || lines_live <= slots || site.len.checked_mul(w)? <= self.l2_per_core()
        {
            return None;
        }
        // COST, not benefit — the one half of this trade that IS derivable (S47).
        //
        // `move_panel_index` divides the counter by `b` four times and by `w/b`
        // twice. When `w` is a power of two every divisor of `gcd(w, rows)` is
        // one too, so `b` and `w/b` both are, and `-O2` lowers all six to shifts.
        // When `w` is NOT a power of two, `b` and `w/b` cannot both be, so at
        // least one magic-multiply sequence lands on every element — **measured
        // at +20 to +25% instructions** on the i9 at width 1536 (185.4 M at
        // B=16 and B=32 against 223.2 M at B=12 and B=24, same loop, same data).
        //
        // The benefit is the unreadable quantity this whole function's note ends
        // on; the cost is not. Firing when a large known cost meets an
        // unpredictable benefit is a bet, and it is one this rung was measured
        // LOSING: at width 1536 on the M4 the fired arm runs **1.75x slower than
        // not blocking at all** (1t; 1.41x threaded), because that part's
        // unblocked walk already costs only 0.594 ns/element — better than its
        // own width-1024 walk — so there is nothing for the permutation to buy.
        //
        // The price of this clause is stated rather than hidden: it forgoes the
        // i9's win at the same width, which the shipped block measured at
        // **1.060x (1t) / 1.044x (threaded)**. Forgoing 6% beats shipping 1.75x.
        // Revisit only if the benefit ever becomes predictable — the plateau
        // there is B = 16..256 and reaching it needs a fact §4 of
        // `benches/results/cpu/matmul/block-size/block-size.md` shows is not readable.
        if !site.width.is_power_of_two() {
            return None;
        }
        // The block balances TWO opposing costs, both of them measured.
        //
        //   traffic  T(B) = max(1, (line/w) / B)   -- a block row shorter than a
        //     line fetches that line once per block-row it is split across. At
        //     B=8 on the i9 (16 f32 per line) it doubles the L1<-L2 traffic, and
        //     it measures 15% slower than B=16 at side 1024, 24% at 2048.
        //   conflict C(B) = max(1, B / (slots/2))  -- the block's read lines and
        //     its write stream compete for the sets the walk can reach. At
        //     B=slots on the M4 it measures 29-56% slower threaded than B=16.
        //
        // Both are 1 inside `[line/w, slots/2]`; when that window is EMPTY —
        // which is the normal case, because the two bounds pull apart — the
        // product `T·C` is minimised at their geometric mean. That is the whole
        // derivation: no threshold, no per-machine number, and it reproduces the
        // M4's swept optimum at BOTH sides (16 at side 1024 and at 2048).
        let floor = (line / w).max(1);
        let ceiling = (slots / 2).max(1);
        let target = (floor * ceiling).isqrt().max(2);
        let tile = gcd(site.width, site.rows);
        (2..=target.min(tile))
            .rev()
            .find(|b| tile.is_multiple_of(*b))
    }

    /// **S62 `place`: does a site of `flops` and `bytes` finish sooner on the
    /// device than on one host lane?** `launch_ns + roofline(device)` against
    /// `roofline(lane)` — the device's [`DeviceCost`] and the host's one-lane
    /// rates ([`DispatchCost`]: a site placed on the host of a GPU build runs
    /// on one lane, the entry of a program with device work being sequential).
    /// Without both facts the answer is the device — what every GPU build did
    /// before this existed. Transfers
    /// are not a term: a site with a device-produced operand stays on the
    /// device by placement (`func/gpu.rs:placement`), so a host placement never
    /// adds a copy-out, and closed-form operands are built where they are read.
    pub fn device_pays(&self, flops: u64, bytes: u64) -> bool {
        let placed = self
            .gpu()
            .and_then(|g| g.cost)
            .filter(|d| d.flops_per_ns > 0);
        let (Some(d), Some(h)) = (placed, self.dispatch) else {
            return true;
        };
        let dev = d.launch_ns + roofline_ns(flops, bytes, d.flops_per_ns, d.bytes_per_ns);
        dev < roofline_ns(flops, bytes, h.flops_per_ns, h.bytes_per_ns)
    }

    /// review-fixes: **this profile with its plan rates never unpriced** —
    /// itself where it carries [`Self::plan_rates`], else its GPU's
    /// [`Gpu::plan_cost`] (each measured rate kept, the rest derived from
    /// the part's properties). What the priced rules that must not fall back
    /// to an old behaviour on an unmeasured part read (`stream`'s unit
    /// against its split realization); the plan tables keep D8.
    pub fn priced(&self) -> TargetProfile {
        if self.plan_rates().is_some() {
            return *self;
        }
        match &self.machine {
            Machine::Gpu(g) => TargetProfile {
                machine: Machine::Gpu(Gpu {
                    cost: Some(g.plan_cost()),
                    ..*g
                }),
                ..*self
            },
            Machine::Cpu => *self,
        }
    }

    /// **plan-cost: the part's calibrated plan rates**, `None` on a part
    /// without them ([`NO_PLAN`], no `fp32_lanes` / `clock_khz`): its plans
    /// stay the rules' byte for byte (decision D8).
    pub fn plan_rates(&self) -> Option<(&Gpu, &DeviceCost)> {
        let g = self.gpu()?;
        let d = g.cost.as_ref()?;
        let given = [
            d.dram_bytes_per_ns,
            d.issue_ns,
            d.atomic_ops_per_ns,
            d.l2_bytes_per_ns,
            d.eff_max_pm,
            g.fp32_lanes,
            g.clock_khz,
        ];
        given.iter().all(|&v| v > 0).then_some((g, d))
    }

    /// **plan-cost §2: the predicted time of one candidate plan of a tile
    /// site**, `None` without [`Self::plan_rates`]. `kernel` = the waves'
    /// `max(FFMA, L2 panel feed)` summed (the last wave holds what is left, at
    /// its own warps per scheduler), bounded below by the DRAM bytes; `combine`
    /// = the atomic adds or the workspace reduce's bytes; `fixed` = the zero
    /// pass and each op's idle: `gap` when the previous op hides the host issue,
    /// `issue − previous op` when it does not (D2). The first op of a site
    /// follows a hidden issue. Only terms that differ between a site's
    /// candidates matter; the caller compares those.
    pub fn site_ns(&self, p: &PlanShape) -> Option<PlanCost> {
        self.site_ns_at(p, 1.0)
    }

    /// plan-cost D2: the time one more dependent op adds to a stream beyond
    /// its own device time — the in-stream gap when the op before it (`prev`
    /// ns of device time) hides the host issue, the issue's remainder when it
    /// does not — plus `bytes` at the part's DRAM rate. `None` without the
    /// plan rates. The one launch term `site_ns`, the epilogue choice and the
    /// flow units read (D1).
    pub fn op_ns(&self, prev: f64, bytes: u64) -> Option<f64> {
        let (_, d) = self.plan_rates()?;
        let idle = (d.gap_ns as f64).max(d.issue_ns as f64 - prev);
        Some(idle + bytes as f64 / d.dram_bytes_per_ns as f64)
    }

    /// [`Self::site_ns`] with the extents scaled by `x` (rows, c, k jointly;
    /// tiles and blocks by `x²`): the curve [`Self::crossover`] walks.
    pub(super) fn site_ns_at(&self, p: &PlanShape, x: f64) -> Option<PlanCost> {
        let (g, d) = self.plan_rates()?;
        let f = |v: u64| v as f64;
        let (rows, c, k, e) = (f(p.rows) * x, f(p.c) * x, f(p.k) * x, f(p.e));
        let blocks = (f(p.blocks) * x * x).ceil().max(1.0);
        let tiles = f(p.tiles) * x * x;
        let kb = f(p.kb) * x;
        let w = f(g.sms * p.per_sm.max(1));
        let waves = (blocks / w).ceil();
        let tail = blocks - (waves - 1.0) * w;
        let warps = f(p.threads.div_ceil(g.warp));
        let peak = f(g.fp32_lanes) * f(g.clock_khz) / 1e6 * 2.0;
        let (l2, dram) = (f(d.l2_bytes_per_ns), f(d.dram_bytes_per_ns));
        let tile = f(p.bm * p.bn);
        // plan-cost-terms, on a part with `lat_ns` (else the sprint
        // `plan-cost` model, op for op): a wave lasts as long as its busiest
        // SM, which holds `⌈bw / sms⌉` blocks (site-geometry G4), not the
        // average `bw / sms`.
        let terms = d.lat_ns > 0;
        let wave = |bw: f64| {
            let used = bw.min(f(g.sms));
            let busy = match terms {
                true => (bw / f(g.sms)).ceil(),
                false => bw / used,
            };
            let wps = busy * warps / f(g.schedulers());
            let eff = match terms {
                true => plan_eff_at(d, wps, f(p.acc)),
                false => plan_eff(d, wps * f(p.acc)),
            };
            let ff = match terms {
                true => busy * 2.0 * tile * kb / (peak * eff),
                false => bw * 2.0 * tile * kb / (used * peak * eff),
            };
            let lf = bw * (f(p.bm) * f(p.ea) + f(p.bn) * f(p.eb)) * kb / l2;
            // tile-efficiency T4: the busiest SM issues `warps · trips · lsu`
            // LSU warp-instructions at `lsu_ipc` a clock; the FFMA pipe waits
            // on them when the tile's FFMA per LSU is below `4 / lsu_ipc`.
            let busy_u = (bw / f(g.sms)).ceil();
            let uf = match d.lsu_ipc_pm {
                0 => 0.0,
                r => {
                    let trips = kb / f(p.bk.max(1));
                    busy_u * warps * trips * f(p.lsu) / (f(r) / 1000.0 * f(g.clock_khz) / 1e6)
                }
            };
            (ff.max(uf), lf, wps, uf)
        };
        // Little's law and k-depth (G3): a block's trips (`kb / bk` panels)
        // wait `lat_ns` each, `fly` at a time, plus the pipeline's fill — the
        // floor of every wave whatever its SMs' bandwidth: a grid on few SMs or
        // few warps keeps too few bytes in flight to pull the chip's L2 / DRAM
        // rate (`min(bw, Σ in flight / lat)`), and a split trades trips for
        // its combine.
        let chain = match terms {
            true => {
                let (trips, fly) = ((kb / f(p.bk.max(1))).ceil().max(1.0), f(p.fly.max(1)));
                (trips + fly - 1.0) / fly * f(d.lat_ns)
            }
            false => 0.0,
        };
        let (ff, lf, wps_full, uf) = wave(w.min(blocks));
        let (ft, lt, wps_tail, ut) = wave(tail);
        let full = waves - 1.0;
        let out = rows * c * e;
        let ws = if p.comb == PlanComb::Workspace {
            f(p.s)
        } else {
            1.0
        };
        let d_ns = (rows * k * f(p.ea) + k * c * f(p.eb) + out * ws) / dram;
        let kernel_ns = (full * ff.max(lf).max(chain) + ft.max(lt).max(chain)).max(d_ns);
        let atomics = match p.comb {
            PlanComb::Atomic => f(p.s) * rows * c,
            PlanComb::StreamK => (tiles + blocks) * tile,
            _ => 0.0,
        };
        let mut combine_ns = atomics / f(d.atomic_ops_per_ns);
        let (gap, issue) = (f(d.gap_ns), f(d.issue_ns));
        let idle = |prev: f64| gap.max(issue - prev);
        let zero = f(d.zero_ns) + out / l2;
        let (launches, zeros, fixed_ns) = match p.comb {
            PlanComb::None => (1, 0, gap),
            PlanComb::Atomic | PlanComb::StreamK => (1, 1, gap + zero + idle(zero)),
            PlanComb::Workspace => {
                combine_ns += (ws + 1.0) * out / dram;
                (2, 0, gap + idle(kernel_ns))
            }
        };
        Some(PlanCost {
            waves: waves as u64,
            tail: tail as u64,
            wps: if waves > 1.0 { wps_full } else { wps_tail },
            f_ns: full * ff + ft,
            l_ns: full * lf + lt,
            d_ns,
            c_ns: waves * chain,
            u_ns: full * uf + ut,
            kernel_ns,
            combine_ns,
            fixed_ns,
            launches,
            zeros,
            atomics: atomics as u64,
            ns: kernel_ns + combine_ns + fixed_ns,
        })
    }

    /// **plan-cost: the scale where `pick` stops being the cheaper of the
    /// two** — the power of two nearest 1 (`2^±j`, `j` = 1…6) of a joint factor
    /// on (rows, c, k) at which `alt` costs less, `pick` having cost no more
    /// at every power between; `None` when it never does in 1/64…64 (or
    /// without plan rates). Powers of two keep a scaled grid a whole number of
    /// tiles, the way a size ladder (512 / 1024 / 2048) grows. Derived per site
    /// on demand and printed on the plan line, never stored (D3).
    pub fn crossover(&self, pick: &PlanShape, alt: &PlanShape) -> Option<f64> {
        let cheaper = |j: i32| {
            let x = f64::from(j).exp2();
            Some(self.site_ns_at(pick, x)?.ns <= self.site_ns_at(alt, x)?.ns)
        };
        for j in 1..=6 {
            for x in [j, -j] {
                if !cheaper(x)? {
                    return Some(f64::from(x).exp2());
                }
            }
        }
        None
    }

    /// The GPU facts, when this profile describes one. `None` for every CPU
    /// profile — callers that need a GPU fact must ask for it, so a CPU profile
    /// reaching a GPU-only path is a `None` at the seam rather than a silently
    /// wrong constant.
    pub fn gpu(&self) -> Option<&Gpu> {
        match &self.machine {
            Machine::Cpu => None,
            Machine::Gpu(g) => Some(g),
        }
    }

    /// **The GPU rungs this part ships at a site this full, on this face**
    /// (hardware-properties): the rung rules over the part's physical
    /// properties ([`crate::rules::rungs`]) — never a per-part row. A CPU
    /// profile has every rung (no GPU site reads them).
    pub fn gpu_rungs(&self, warps_per_scheduler: u64, contract: bool) -> crate::GpuRungs {
        self.gpu_rungs_on(warps_per_scheduler, Face::of(contract, None))
    }

    /// [`Self::gpu_rungs`] on a site's face, TF32 included.
    pub fn gpu_rungs_on(&self, warps_per_scheduler: u64, face: Face) -> crate::GpuRungs {
        match self.gpu() {
            Some(g) => crate::rules::rungs(g, warps_per_scheduler, face),
            None => crate::GpuRungs::ALL,
        }
    }
}
