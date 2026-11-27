//! Tests: the default profile, the host tile and the SME budget.

use super::*;

/// Rule 1, at the arithmetic level: the default profile must reproduce every
/// literal the emitter used to carry. If this fails, emission has moved.
#[test]
fn generic_reproduces_the_six_literals() {
    assert_eq!(GENERIC.tile_j(&F32), 16, "tile_j_for f32");
    assert_eq!(GENERIC.tile_j(&F64), 8, "tile_j_for f64");
    assert_eq!(GENERIC.tile_i(), 4, "TILE_I");
    assert_eq!(GENERIC.nc(&F32), 512, "tile_nc_for f32");
    assert_eq!(GENERIC.nc(&F64), 256, "tile_nc_for f64");
    assert_eq!(GENERIC.tile_kc(&F32), 128, "TILE_KC f32");
    assert_eq!(GENERIC.tile_kc(&F64), 128, "TILE_KC f64");
    assert_eq!(GENERIC.heap_min_bytes, 256 * 1024, "HEAP_MIN_BYTES");
}

/// The S26 sweep's verdict, as arithmetic: TI=4 fits in half the file and
/// TI=8 does not. The recorded failure was "8 spills: 128 accumulators ≫ 32
/// NEON regs" — 8 rows × 16 f32 lanes ÷ 4 lanes per register = 32 registers,
/// the whole file with nothing left for operands.
#[test]
fn tile_i_is_the_register_budget_not_a_taste() {
    let regs_per_row = GENERIC.tile_j(&F32) / GENERIC.lanes(&F32);
    assert_eq!(regs_per_row, GENERIC.acc_vecs_per_row);
    assert_eq!(GENERIC.tile_i() * regs_per_row, GENERIC.vec_regs / 2);
    assert_eq!(8 * regs_per_row, GENERIC.vec_regs, "TI=8 is the whole file");
}

/// The measured S29/S30 verdict, reproduced as a deduction — **and its
/// boundary, pinned honestly**. On a 16 MB L2 the k-panel is 4096, deeper
/// than every K we run, so the gate `site.k > tile_kc` never opens. Above
/// 4096 it DOES reopen: this is a threshold, not an off-switch, and the
/// 8192 assertion is here so nobody reads the others as "apple-m disables
/// the nest". See `tile_kc`'s note — past the threshold the derivation and
/// the measurement disagree.
#[test]
fn apple_m_raises_the_kc_threshold_above_every_k_we_run() {
    assert_eq!(APPLE_M.tile_kc(&F32), 4096);
    for k in [1024_u64, 2048, 4096] {
        assert!(k <= APPLE_M.tile_kc(&F32), "KC nest must stay off at K={k}");
    }
    assert!(
        8192 > APPLE_M.tile_kc(&F32),
        "the gate REOPENS past 4096 — an unmeasured regime, not a disabled rung"
    );
    assert!(
        2048 > GENERIC.tile_kc(&F32),
        "generic keeps today's gate open"
    );
}

/// The KC a-panel is `tile_i x tile_kc x sizeof` = `2 * l2_bytes/nc_tiles`,
/// so it scales with the profile's L2: 2 KB under `generic`, 64 KB under
/// `apple-m`. Pinned because `func/mod.rs`'s heap note used to claim a flat
/// 2 KB, and that number is a stack-sizing claim.
#[test]
fn kc_apanel_scales_with_l2() {
    let apack = |p: &TargetProfile, e: &Ty| p.tile_i() * p.tile_kc(e) * 4;
    assert_eq!(apack(&GENERIC, &F32), 2 * 1024);
    assert_eq!(apack(&APPLE_M, &F32), 64 * 1024);
    assert!(
        apack(&APPLE_M, &F32) < GENERIC.heap_min_bytes,
        "still on the stack, but the headroom is profile-dependent"
    );
}

#[test]
fn zen3_moves_two_constants_at_once() {
    assert_eq!(ZEN3.tile_j(&F32), 32);
    assert_eq!(ZEN3.tile_i(), 2);
}

#[test]
fn unknown_name_is_not_silently_generic() {
    assert_eq!(resolve("generic"), Some(&GENERIC));
    assert_eq!(resolve("apple-m"), Some(&APPLE_M));
    assert_eq!(resolve("cuda-ada"), Some(&CUDA_ADA));
    assert_eq!(resolve("apple-m4-sme"), Some(&APPLE_M4_SME));
    assert!(resolve("apple_m").is_none());
    assert!(
        names().contains("native"),
        "the unknown-name error must offer `native`, or detection is undiscoverable"
    );
}

/// plan-s51 H2 — the A pack's block depth, and the two ways it declines.
///
/// The pack's stores land one cache line apart, so a line is filled 4 bytes
/// at a time across the `panel_rows` rows that revisit it. Free while `ap`
/// is resident, 32x the line traffic once it is not — so the decision is
/// exactly "does `32 · k · sizeof` fit L1D", and 1024 is the last `k` that
/// does on this part.
///
/// The depth is `line_bytes / sizeof` — one line's worth of `k` — and both
/// numbers are DETECTED, so nothing here is swept.
#[test]
fn pack_block_fires_only_once_the_pack_buffer_leaves_l1d() {
    // 32 x 1024 x 4 = 131072 = L1D exactly. Fits, so no block.
    assert_eq!(APPLE_M4_SME.pack_block(32, 1024, &F32), None);
    // One k-block deeper and it does not fit.
    assert_eq!(APPLE_M4_SME.pack_block(32, 1056, &F32), Some(32));
    assert_eq!(APPLE_M4_SME.pack_block(32, 4096, &F32), Some(32));
    // f64 halves the elements per line, so the block halves with it.
    assert_eq!(APPLE_M4_SME.pack_block(32, 4096, &F64), Some(16));
    // A profile with no L1D geometry cannot decide and must keep the
    // single-pass copy — the same contract every other derived factor has.
    assert_eq!(GENERIC.pack_block(32, 4096, &F32), None);
}

/// The SME k-panel depth is **derived from L1D**, and the derivation
/// reproduces the swept optimum — `plan-s31-deduced-blocking.md`'s rule.
/// The 1024-step depth follows from (2 × 128 KB) / 256 B.
#[test]
fn sme_kc_reproduces_the_measured_optimum() {
    assert_eq!(
        APPLE_M4_SME.sme_kc(&F32),
        Some(1024),
        "2 x 128 KB / ((ti+tj)·t·4 = 256 B) = 1024 — the depth SWEPT in the real \
         emitter at N=4096 (1.064x, disjoint) and N=2048. Ratio 1 gives 512, \
         which measured 0.785x — a loss"
    );
    // f64 scales off the same recorded fact by element width: t halves to 8,
    // the tile count doubles to 8 so the block is 2x4, and the window is
    // (2+4)·8·8 = 384 B per k. No second constant anywhere.
    assert_eq!(APPLE_M4_SME.sme_kc(&F64), Some(2 * 128 * 1024 / 384));
    // No matrix unit => no answer, never a wrong one. Same seam as `sme()`.
    assert_eq!(GENERIC.sme_kc(&F32), None);
    assert_eq!(APPLE_M.sme_kc(&F32), None);
}

/// The SME budget is L1D and **not** the NEON rung's L2, and the two answers
/// differ by 8x — which is the whole reason `sme_kc` exists rather than
/// reusing `tile_kc`. If these two ever coincide, one of them is wrong.
#[test]
fn the_sme_budget_is_not_the_neon_one() {
    let neon = APPLE_M4_SME.tile_kc(&F32);
    let sme = APPLE_M4_SME.sme_kc(&F32).unwrap();
    assert_eq!(neon, 4096, "unchanged: half of 16 MB shared L2 over nc");
    assert_eq!(sme, 1024);
    assert!(
        sme < neon,
        "the matrix unit's window is smaller than the NEON k-panel's; \
         a shared derivation would size SME's panel 8x too deep and lose \
         1.448x at K=4096 (docs/performance/cpu/matmul/sme-throughput-ceiling.md §5)"
    );
    // And the gate this feeds: at K=4096 the SME nest must OPEN...
    assert!(4096 > sme, "KC nest must fire at K=4096");
    // ...while at K=512 it must stay shut, because blocking is a LOSS there.
    assert!(
        1024 <= sme,
        "KC nest must stay off at K=1024 — measured neutral-to-worse there"
    );
}

/// Detection reads the keys it documents. Not a value assertion — the point
/// is that the plumbing works on this host, so "automatic" is not a claim
/// resting on an untested code path.
#[cfg(target_os = "macos")]
#[test]
fn native_reads_the_facts_it_claims_to() {
    let l2 = sysctl_u64("hw.perflevel0.l2cachesize");
    let l1d = sysctl_u64("hw.perflevel0.l1dcachesize");
    assert!(l2.is_some_and(|v| v >= 256 * 1024), "L2 detected: {l2:?}");
    assert!(l1d.is_some_and(|v| v >= 32 * 1024), "L1D detected: {l1d:?}");
    assert!(sysctl_u64("hw.optional.arm.no.such.key").is_none());

    let native = native().expect("detection must succeed on macOS");
    assert_eq!(native.name, "native");
    assert_eq!(Some(native.l2_bytes), l2, "detected, not inherited");

    // The E-core trap: the bare key is a DIFFERENT, smaller cache. If these
    // two ever agree, this host stopped being big.LITTLE and the guard below
    // stops proving anything — but on any such part it is also harmless.
    if let (Some(bare), Some(p0)) = (sysctl_u64("hw.l1dcachesize"), l1d) {
        assert!(
            bare <= p0,
            "hw.l1dcachesize ({bare}) must not exceed the P-core's ({p0}); \
             detection reads perflevel0 precisely because the bare key is \
             the efficiency cluster"
        );
    }
}

/// One `sysctl` string — the sibling of `sysctl_u64` for the one key whose
/// value is text. The part gates below read the brand string, because
/// ADR-0038 D4 gates *exactness* on the part, loudly: an exact check that
/// can only run on the documented machine must say so in its name, and must
/// say out loud when it did not run.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sysctl_string(key: &str) -> Option<String> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
}

/// **ADR-0038 D4, the machine-independent half of the genericity gate.**
/// Detection must succeed, the matrix unit must be recorded iff the part
/// reports one, and every derived SME quantity must be a pure function of
/// the recorded facts: the rebuild below freezes every input the
/// derivations may not read at the hand-written profile's value, so a
/// derivation that grows a hidden host read diverges here. This runs on
/// ANY Apple-silicon Mac.
///
/// What it does NOT assert is agreement with the hand-written numbers —
/// that held on a base M4 only because SVL, the P-core L1D and the 16 MB
/// cluster L2 happen to be constant across the M4 family, and a fact that
/// happens to hold on every runner is luck, not a gate. Exactness is the
/// part-gated test below.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_sme_profile_is_consistent() {
    let native = native().expect("detection must succeed on macOS");
    assert_eq!(
        native.sme().is_some(),
        sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1),
        "the unit is recorded iff the part reports it — an absent key reads \
         as None, not 0, so pre-SME parts land here cleanly rather than \
         failing a Some(0) assertion"
    );
    let Some(nsme) = native.sme() else {
        // No matrix unit on this part: nothing to compare, and the fallback
        // is the pre-S41 behaviour. Not a failure.
        return;
    };
    let l1d = native.l1d.expect("macOS detection records L1D geometry");
    assert_eq!(
        nsme.l1d_bytes, l1d.bytes,
        "both rungs read the same P-core L1D (`l1d_and_sme_agree_on_the_l1d`, \
         for the detected profile)"
    );
    let rebuilt = TargetProfile {
        sme: native.sme,
        ..APPLE_M4_SME
    };
    for elem in [F32, F64] {
        assert_eq!(native.sme_tile_side(&elem), rebuilt.sme_tile_side(&elem));
        assert_eq!(native.sme_block(&elem), rebuilt.sme_block(&elem));
        assert_eq!(native.sme_kc(&elem), rebuilt.sme_kc(&elem));
    }
}

/// **ADR-0038 D4, the exact half — gated on the documented part, loudly.**
/// On the documented M4 Pro, detection must reproduce the hand-written
/// profile *exactly* — same SVL, same L1D, same derived tile side, same
/// derived block, same derived k-panel. If it does not, then one of the
/// two is wrong and the hand-written numbers were never facts about this
/// machine. On any other part the exact check has nothing to say, and the
/// skip names the part it skipped.
///
/// This is what makes `apple-m4-sme` a **cross-compilation** spelling rather
/// than the only way to describe the machine under the keyboard: the numbers
/// in it are no longer load-bearing here, they are checked against the part.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts() {
    let part = sysctl_string("machdep.cpu.brand_string")
        .unwrap_or_else(|| "<brand string unreadable>".to_owned());
    if !part.contains("M4 Pro") {
        eprintln!(
            "native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts: \
             SKIP — the exact facts are the documented M4 Pro's; this part is `{part}`"
        );
        return;
    }
    let native = native().expect("detection must succeed on macOS");
    let nsme = native.sme().expect("the documented M4 Pro has SME");
    let hand = APPLE_M4_SME.sme().expect("hand-written profile has SME");
    assert_eq!(nsme.svl_bytes, hand.svl_bytes, "SVL: detected vs written");
    assert_eq!(nsme.l1d_bytes, hand.l1d_bytes, "L1D: detected vs written");
    assert_eq!(native.l2_bytes, APPLE_M4_SME.l2_bytes, "L2");
    for elem in [F32, F64] {
        assert_eq!(
            native.sme_tile_side(&elem),
            APPLE_M4_SME.sme_tile_side(&elem)
        );
        assert_eq!(native.sme_block(&elem), APPLE_M4_SME.sme_block(&elem));
        assert_eq!(native.sme_kc(&elem), APPLE_M4_SME.sme_kc(&elem));
    }
    // And the number the whole S42 campaign turns on, arrived at with no
    // constant written anywhere: 2*131072 / ((2+2)*16*4 = 256) = 1024.
    assert_eq!(native.sme_kc(&F32), Some(1024));
}

/// One `MoveSite` for a square `side x side` f32 transpose — the shape every
/// measured point in `move_block`'s table is.
fn transpose(side: u64) -> MoveSite {
    MoveSite {
        width: side,
        rows: side,
        cq: 1,
        cr: side,
        elem: F32,
        len: side * side,
    }
}

/// **The whole S45 decision, scored against every measured point on two
/// machines.** Each row is a measurement, not a preference; the comment
/// beside it is what the emitted pipeline actually did.
#[test]
fn move_block_reproduces_both_machines() {
    // M4 Pro: fires at 1024 with the block S44 SWEPT to (16), fires at 2048,
    // declines at 128 on pressure and at 512 on cost.
    // BOTH M4 sides land on the block the machine measured fastest — 1024
    // has floor 32 / ceiling 16 and 2048 has floor 32 / ceiling 8, and the
    // geometric mean is 22 and 16, so the largest divisor is 16 either way.
    assert_eq!(APPLE_M4_SME.move_block(&transpose(1024)), Some(16));
    assert_eq!(APPLE_M4_SME.move_block(&transpose(2048)), Some(16));
    assert_eq!(
        APPLE_M4_SME.move_block(&transpose(128)),
        None,
        "pressure 0.5: 128 lines live into 32 sets x 8 ways = 256 slots — measured 1.000x"
    );
    assert_eq!(
        APPLE_M4_SME.move_block(&transpose(512)),
        None,
        "pressure 8 but the 1 MB read array sits inside 16 MB / 5 = 3.36 MB — measured \
         overlapping, 1.02x min-to-min"
    );
    // i9: fires at 1024 and 2048, and DECLINES at 512 — the discriminating
    // case, where pressure alone would fire and the machine measured 0.901x.
    // The i9's 12 reachable slots put the ceiling at 6 against a floor of
    // 16, so the mean is 9 and the block is 8. Its measured optimum is 128,
    // which no L1-derived rule can reach — see `move_block`'s note and the
    // counter evidence in benches/results/cpu/memory/deduced-panel-movement.
    assert_eq!(RAPTORLAKE.move_block(&transpose(1024)), Some(8));
    assert_eq!(RAPTORLAKE.move_block(&transpose(2048)), Some(8));
    assert_eq!(
        RAPTORLAKE.move_block(&transpose(512)),
        None,
        "pressure 21.3 — ABOVE the M4's 8 at the same side — but the 1 MB read array fits \
         the 2 MB private L2, and it measured a 0.901x LOSS (replicated 0.907x). This is \
         the case pressure alone gets wrong, and the reason the cost term exists"
    );
    // Odd sides never collapse the set index, so there is nothing to win:
    // S44 measured side 1025 running 2.12x faster UNBLOCKED than 1024, and
    // blocking it at 0.623x. Note this one scores pressure **1.0009** — it
    // is the witness for the conflict clause, not the pressure clause, and
    // without that clause the deduction would fire on a measured LOSS.
    assert_eq!(APPLE_M4_SME.move_block(&transpose(1025)), None);
    // S47: width 1536 passes EVERY term above on both machines — the set
    // index collapses (8 sets on the M4, 2 on the i9), pressure is 24x and
    // 64x, and the 9.44 MB read array is past both private-L2 shares — and
    // firing it measured a **1.75x LOSS** on the M4 (1t; 1.41x threaded).
    // A width that is not a power of two cannot give both `b` and `w/b` a
    // shift lowering, so the permutation costs +20-25% instructions there
    // and the benefit is unpredictable. This is the only measured cell where
    // the rung was a net loss, and it is the only one this clause changes.
    assert_eq!(
        APPLE_M4_SME.move_block(&transpose(1536)),
        None,
        "measured 1.75x SLOWER than unblocked at 1 thread, 1.41x threaded"
    );
    assert_eq!(
        RAPTORLAKE.move_block(&transpose(1536)),
        None,
        "forgoes a 1.060x/1.044x win to avoid the M4's 1.75x loss at the same width"
    );
    // Powers of two are untouched by that clause — the four cells the rung
    // actually earns its keep on are all above.
    for side in [1024, 2048] {
        assert!(APPLE_M4_SME.move_block(&transpose(side)).is_some());
        assert!(RAPTORLAKE.move_block(&transpose(side)).is_some());
    }
    // And a profile with no L1D geometry cannot answer at all.
    for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
        assert_eq!(p.move_block(&transpose(1024)), None, "{}", p.name);
    }
}

/// The pressure term is `lines_live > slots` — the literal statement that the
/// sweep needs more lines than it can reach — so the set collapse, not the
/// power of two, is what it turns on. S44's two load-bearing nulls (side 128
/// at pressure 0.5, side 544 at 0.53) are both collapses that stay under 1.
#[test]
fn pressure_counts_lines_not_powers_of_two() {
    let g = APPLE_M4_SME.l1d.expect("apple-m4-sme has L1D geometry");
    assert_eq!((g.sets, g.ways()), (128, 8));
    // side 128: stride 512 B = 4 lines, gcd(4, 128) = 4 => 32 sets, 256 slots
    // against 128 lines live. A REAL collapse (32 of 128 sets) that still
    // must not fire — "it is not 'a collapse is bad', it is pressure".
    assert_eq!(APPLE_M4_SME.move_block(&transpose(128)), None);
    // side 544: gcd(17, 128) = 1 => all 128 sets, 1024 slots, 544 live.
    assert_eq!(APPLE_M4_SME.move_block(&transpose(544)), None);
}

/// `ways` is arithmetic, `sets` is read where the host exposes it — and the
/// two i9 geometries are where the truth IS exposed, so they are the check
/// on the whole scheme. Both reproduce `ways_of_associativity` exactly.
#[test]
fn l1d_ways_reproduce_the_readable_truth() {
    let i9_p = RAPTORLAKE.l1d.expect("raptorlake has L1D geometry");
    assert_eq!(i9_p.ways(), 12, "sysfs: ways_of_associativity = 12");
    let i9_e = L1d {
        bytes: 32 * 1024,
        line_bytes: 64,
        sets: 64,
    };
    assert_eq!(i9_e.ways(), 8, "sysfs: the E-core is 8-way");
    // The M4's set count is the one macOS hides; the page bound gives 128,
    // which is the value S44 verified with a stride probe, and 8 ways falls
    // out of capacity.
    let m4 = APPLE_M4_SME.l1d.expect("apple-m4-sme has L1D geometry");
    assert_eq!((16384 / m4.line_bytes, m4.sets), (128, 128));
    assert_eq!(m4.ways(), 8);
    // Whatever the split, capacity is exact — the bound on being wrong.
    for g in [i9_p, i9_e, m4] {
        assert_eq!(g.sets * g.ways(), g.bytes / g.line_bytes);
    }
}

/// The L2 sharing fact, which is the cost term's divisor and the field this
/// session added because the old doc comment said "per-core" while `apple-m`
/// recorded a SHARED cluster cache. Getting it wrong by 5x is the difference
/// between firing and declining at M4 side 1024.
#[test]
fn l2_per_core_is_not_l2_bytes() {
    assert_eq!(APPLE_M.l2_per_core(), 16 * 1024 * 1024 / 5);
    assert!(
        4 * 1024 * 1024 > APPLE_M.l2_per_core(),
        "a 4 MB read array must OVERFLOW the M4's share (it measured 1.578x)"
    );
    assert!(
        4 * 1024 * 1024 < APPLE_M.l2_bytes,
        "...while fitting the cluster L2 whole, which is why the distinction is the answer"
    );
    assert_eq!(
        RAPTORLAKE.l2_per_core(),
        2 * 1024 * 1024,
        "private per P-core"
    );
    assert_eq!(GENERIC.l2_per_core(), GENERIC.l2_bytes, "l2_cores = 1");
}

/// The duplicated L1D size cannot drift: `Sme::l1d_bytes` and `L1d::bytes`
/// are the same machine fact reached by two rungs. See `TargetProfile::l1d`
/// for why the merge was not done here.
#[test]
fn l1d_and_sme_agree_on_the_l1d() {
    for p in PROFILES {
        if let (Some(sme), Some(l1d)) = (p.sme(), p.l1d) {
            assert_eq!(sme.l1d_bytes, l1d.bytes, "{} disagrees with itself", p.name);
        }
    }
}

/// **ADR-0038 D4, the machine-independent half.** `move_block`'s verdict
/// must be a pure function of the recorded facts: the rebuild below copies
/// everything the derivation may legitimately read (`l1d`, `l2_bytes`,
/// `l2_cores`) and freezes every other field at the hand-written profile's
/// value, so a verdict that grows a hidden host read — or starts consulting
/// a field it has no business with — diverges. This runs on ANY
/// Apple-silicon Mac, including a base M4 whose `cpusperl2` is 3 where the
/// documented part's is 5.
///
/// No exact fire/decline verdict is asserted here: `l1d: Some` and
/// `l2_cores >= 1` do not imply one. Whether side 1024 fires turns on the
/// 4 MB read array against the per-core L2 share — a fact of the part — so
/// exactness belongs to the part-gated test below.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_move_geometry_is_consistent() {
    let native = native().expect("detection must succeed on macOS");
    assert!(
        native.l1d.is_some(),
        "macOS detection always records L1D geometry"
    );
    assert!(native.l2_cores >= 1, "the sharing factor is clamped at 1");
    let rebuilt = TargetProfile {
        l1d: native.l1d,
        l2_cores: native.l2_cores,
        l2_bytes: native.l2_bytes,
        ..APPLE_M4_SME
    };
    for side in [128_u64, 512, 1024, 1025, 2048] {
        assert_eq!(
            native.move_block(&transpose(side)),
            rebuilt.move_block(&transpose(side)),
            "side {side}: the verdict must be a pure function of the recorded facts"
        );
    }
}

/// **ADR-0038 D4, the exact half — gated on the documented part, loudly.**
/// The hand-written geometry is the M4 Pro's: FIVE P-cores sharing the
/// 16 MB cluster L2. A base M4 shares it 3 ways, which flips the side-1024
/// verdict (the 4 MB read array overflows the 3.36 MB per-core share but
/// fits the 5.46 MB one) — that flip is the hosted-CI failure this split
/// repairs, not a bug in either profile. Exactness therefore runs only on
/// the part it describes, and the skip names the part it skipped.
///
/// On the documented M4 Pro, detection must reproduce the hand-written L1D
/// geometry and L2 sharing *exactly*, and therefore reach the same
/// fire/decline verdict and the same block. If it does not, one of the two
/// is wrong and the hand-written numbers were never facts about this
/// machine.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_on_the_documented_m4_pro_reproduces_hand_written() {
    let part = sysctl_string("machdep.cpu.brand_string")
        .unwrap_or_else(|| "<brand string unreadable>".to_owned());
    if !part.contains("M4 Pro") {
        eprintln!(
            "native_on_the_documented_m4_pro_reproduces_hand_written: SKIP — \
             the exact geometry is the documented M4 Pro's; this part is `{part}`"
        );
        return;
    }
    let native = native().expect("detection must succeed on macOS");
    assert_eq!(
        native.l1d, APPLE_M4_SME.l1d,
        "detected L1D geometry vs written"
    );
    assert_eq!(native.l2_cores, APPLE_M4_SME.l2_cores, "cores sharing L2");
    assert_eq!(native.l2_per_core(), APPLE_M4_SME.l2_per_core());
    for side in [128_u64, 512, 1024, 1025, 2048] {
        assert_eq!(
            native.move_block(&transpose(side)),
            APPLE_M4_SME.move_block(&transpose(side)),
            "detected and written must decide side {side} the same way"
        );
    }
    // And the number the whole session turns on, detected rather than typed.
    assert_eq!(native.move_block(&transpose(1024)), Some(16));
}

/// `apple-m` gained `l2_cores` and nothing else — no derivation reads it, so
/// its emission cannot have moved. The negative control for the profile edit.
#[test]
fn apple_m_gained_only_the_sharing_fact() {
    assert_eq!(
        APPLE_M,
        TargetProfile {
            l2_cores: APPLE_M.l2_cores,
            name: APPLE_M.name,
            l2_bytes: APPLE_M.l2_bytes,
            ..GENERIC
        }
    );
    assert_eq!(
        APPLE_M.tile_kc(&F32),
        4096,
        "unchanged: the KC gate is where it was"
    );
    assert!(
        APPLE_M.l1d.is_none(),
        "no L1D geometry ⇒ the move rung cannot fire"
    );
}

/// A hand-written profile always wins over detection (Sapir's rule).
#[test]
fn named_profiles_override_detection() {
    for p in PROFILES {
        assert_eq!(
            resolve(p.name).map(|r| r.name),
            Some(p.name),
            "the table is consulted before `native`"
        );
    }
    assert!(
        resolve("nativ").is_none(),
        "no fuzzy match, no silent default"
    );
    assert!(resolve("cuda_ada").is_none());
    assert!(resolve("apple-m4").is_none());
    assert!(resolve("").is_none());
}

/// Rule 1 for the SME leg: the capability is reachable only through
/// `sme()`, and no profile that existed before it can answer. That is the
/// whole byte-identity argument — the SME realization's gate cannot open
/// without a fact only `apple-m4-sme` carries.
#[test]
fn sme_is_absent_from_every_pre_sme_profile() {
    for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
        assert!(p.sme().is_none(), "{} must expose no SME facts", p.name);
        assert!(
            p.sme_tile_side(&F32).is_none(),
            "{} derives no tile",
            p.name
        );
    }
}

/// The measured M4 Pro facts, and the one derivation the emitter reads:
/// SVL 512 bits ⇒ a 16×16 f32 tile (`benches/sme/svl.c`, `run16.c`). f64
/// halves it, which is why the side is derived rather than written down.
#[test]
fn apple_m4_sme_derives_the_measured_tile_side() {
    let sme = APPLE_M4_SME.sme().expect("apple-m4-sme has an SME unit");
    assert_eq!(sme.svl_bytes, 64, "SVL = 512 bits, detected");
    assert_eq!(APPLE_M4_SME.sme_tile_side(&F32), Some(16));
    assert_eq!(APPLE_M4_SME.sme_tile_side(&F64), Some(8));
    // The tile count is an ISA rule, not a field: exactly `elem_bytes`.
    assert_eq!(APPLE_M4_SME.sme_block(&F32), Some((2, 2)), "4 tiles at f32");
    assert_eq!(APPLE_M4_SME.sme_block(&F64), Some((2, 4)), "8 tiles at f64");
}

/// The SME profile moves only DECLARED-CAPABILITY fields off `apple-m`.
/// Every derived tile factor is unchanged, so a site that falls back to the
/// NEON rung under `apple-m4-sme` emits what `apple-m` emits — the negative
/// control the golden test pins at the text level.
#[test]
fn apple_m4_sme_moves_only_declared_capability_fields() {
    // S45 adds `l1d` alongside `sme`, plan-s50 adds `dispatch`: the same
    // part, more capability declared. Each is an `Option` that is `None` on
    // `apple-m`, so a rung that needs one cannot fire without it and every
    // DERIVED tile factor below must still be `apple-m`'s.
    assert_eq!(
        APPLE_M4_SME,
        TargetProfile {
            name: APPLE_M4_SME.name,
            sme: APPLE_M4_SME.sme,
            l1d: APPLE_M4_SME.l1d,
            dispatch: APPLE_M4_SME.dispatch,
            ..APPLE_M
        }
    );
    assert!(
        APPLE_M.dispatch.is_none(),
        "an unmeasured part must carry no dispatch economics, so the \
         threshold cannot fire on a number nobody took"
    );
    assert_eq!(APPLE_M4_SME.tile_j(&F32), APPLE_M.tile_j(&F32));
    assert_eq!(APPLE_M4_SME.tile_i(), APPLE_M.tile_i());
    assert_eq!(APPLE_M4_SME.tile_kc(&F32), APPLE_M.tile_kc(&F32));
}

/// The accumulator budget, spent. 4 f32 tiles ⇒ a 2×2 block, which is the
/// arrangement `benches/sme/mm4.c` measured at 1.84–2.61× over one tile.
/// f64 has twice as many tiles (one per element byte), and the count is
/// derived from the recorded f32 one rather than written down again.
#[test]
fn sme_block_is_the_square_most_split_of_the_tile_count() {
    assert_eq!(APPLE_M4_SME.sme_block(&F32), Some((2, 2)));
    assert_eq!(APPLE_M4_SME.sme_block(&F64), Some((2, 4)), "8 tiles at f64");
    for p in [&GENERIC, &APPLE_M, &ZEN3, &CUDA_ADA] {
        assert!(p.sme_block(&F32).is_none(), "{} has no ZA", p.name);
    }
}

/// The derivation's two invariants: the block spends the WHOLE tile budget,
/// and it is the square-most factorization — the one that minimises `ti + tj`
/// operand loads per `ti · tj` outer products (1×4 needs 5 loads for the 4
/// MACs 2×2 gets from 4).
///
/// **Swept over element WIDTHS, not over a recorded count.** The previous
/// version of this test swept `f32_tiles` 1..=64, and S41b logged the defect
/// that created: it pinned arrangements for counts like 7 or 64 that no ZA
/// register file has and no emitted kernel could select, so the test asserted
/// properties of unreachable IR. Now that the count is `elem_bytes` by
/// construction there is nothing unreachable left to sweep — every width here
/// is a width the architecture actually has.
#[test]
fn sme_block_spends_every_tile_and_stays_square_most() {
    for bits in [8_u8, 16, 32, 64] {
        let elem = Ty::Float { bits };
        let tiles = u64::from(bits).div_ceil(8);
        let (ti, tj) = APPLE_M4_SME.sme_block(&elem).expect("has ZA");
        assert_eq!(
            ti * tj,
            tiles,
            "the block must spend every tile ({tiles} at f{bits})"
        );
        assert!(ti <= tj, "ti is the smaller factor (f{bits})");
        let best = (1..=tiles)
            .filter(|d| tiles.is_multiple_of(*d))
            .map(|d| d + tiles / d)
            .min()
            .expect("a divisor exists");
        assert_eq!(ti + tj, best, "fewest operand loads at f{bits}");
        // Every arrangement must actually fit ZA: `ti·t x tj·t` elements of
        // `w` bytes is exactly `svl x svl`, the whole array, never more.
        let t = APPLE_M4_SME.sme_tile_side(&elem).expect("has ZA");
        let svl = APPLE_M4_SME.sme().unwrap().svl_bytes;
        assert_eq!(
            ti * t * tj * t * tiles,
            svl * svl,
            "the block must be exactly ZA at f{bits}, not larger"
        );
    }
    assert_eq!(
        (1..=4_u64)
            .filter(|d| 4_u64.is_multiple_of(*d))
            .map(|d| d + 4 / d)
            .min(),
        Some(4),
        "2x2 is 4 loads; 1x4 is 5"
    );
}

/// The packed-B panel the NEON rung already builds is `tile_j` lanes wide,
/// and the SME kernel wants an SVL-wide contiguous row. On this part they
/// are the same 16 — which is why the SME rung consumes the existing packed
/// buffer instead of minting its own. `func/sme.rs` re-checks this per site
/// and falls back when it does not hold, so this is a pin on *why the reuse
/// is free here*, not a load-bearing invariant.
#[test]
fn sme_panel_and_packed_b_panel_coincide_on_this_part() {
    assert_eq!(
        APPLE_M4_SME.sme_tile_side(&F32),
        Some(APPLE_M4_SME.tile_j(&F32))
    );
}

/// S41 step 1, rule 1: adding the machine class must not move a single CPU
/// emission. Every profile that existed before S41 is `Cpu`, so no CPU
/// realization can observe the new field — the type system carries the
/// byte-identity argument, and the 159-emission A/B sweep confirms it.
#[test]
fn every_pre_s41_profile_is_cpu() {
    for p in [&GENERIC, &APPLE_M, &ZEN3] {
        assert_eq!(p.machine, Machine::Cpu, "{} must stay Cpu", p.name);
        assert!(p.gpu().is_none(), "{} must expose no GPU facts", p.name);
    }
}

/// A GPU fact is reachable only through `gpu()`, so a CPU profile on a
/// GPU-only path is a `None` at the seam rather than a wrong constant
/// (plan-s41 §2.2 rule 2 — capability selects realization).
#[test]
fn gpu_facts_are_asked_for_never_defaulted() {
    let g = CUDA_ADA.gpu().expect("cuda-ada is a GPU profile");
    assert_eq!(g.warp, 32, "cooperation quantum");
    assert_eq!(g.max_threads_per_block, 1024);
    assert_eq!(
        g.arch, "sm_89",
        "verified against the device: compute capability 8.9 (RTX 4070 Ti)"
    );
    assert!(
        g.smem_per_block <= g.smem_per_sm,
        "a block cannot opt in to more than its SM holds"
    );
}
