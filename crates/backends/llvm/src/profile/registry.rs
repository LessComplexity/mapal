//! The profile registry: Apple M4 SME, plain FP, `PROFILES`, `resolve`, `names`.

use super::*;

/// Apple M4 with the streaming matrix unit enabled: [`APPLE_M`] plus the SME
/// facts measured on the part (`benches/sme/svl.c` — SVL 64 B, 4 f32 tiles;
/// `hw.optional.arm.FEAT_SME`/`FEAT_SME2`/`FEAT_SME_F32F32` all set).
///
/// It is a **separate name** rather than a field flipped on `apple-m` because
/// the SME leaf is only bit-equal to the NEON leaf on the contract face
/// (`fmopa` fuses — measured 92/256 cells differ against separate mul+add,
/// 0/256 against `fmaf`), so selecting it is a decision about the *program*, not
/// only about the machine. Naming it keeps `apple-m` byte-identical forever and
/// keeps a box run reproducible.
pub const APPLE_M4_SME: TargetProfile = TargetProfile {
    name: "apple-m4-sme",
    // Both facts are `hw.optional.arm.sme_max_svl_b` and
    // `hw.perflevel0.l1dcachesize` on this M4 Pro, and `native` reads both — so
    // this profile is now the *cross-compilation* spelling of a machine you are
    // not sitting on, and
    // `native_on_the_documented_m4_pro_reproduces_the_hand_written_sme_facts`
    // asserts detection agrees with it on the documented part (ADR-0038 D4).
    // The tile count is absent because it is an ISA rule, derived in `sme_block`.
    sme: Some(Sme {
        svl_bytes: 64,
        l1d_bytes: 128 * 1024,
        panel_l1d_ratio: 2,
    }),
    // S45: the same L1D, plus the two facts the move rung needs and the SME rung
    // does not. `hw.cachelinesize` = 128 and `hw.pagesize` = 16384 on this part;
    // 128 sets and 8 ways follow (`L1d`), so no associativity is written down.
    l1d: Some(L1d {
        bytes: 128 * 1024,
        line_bytes: 128,
        sets: 128,
    }),
    // MEASURED ON THIS PART, all three (plan-s50 P3):
    //
    // `flops_per_ns` — the SME rung's one-lane rate. `nn4_1024` at MAPAL_PAR=1
    //   runs 986 GF/s and `nn4_2048` 939; 1000 is the plateau.
    // `bytes_per_ns` — the streaming rate one lane reaches. `saxpy_1048576` at
    //   MAPAL_PAR=1 moves 12.58 MB in 0.0991 ms = 127 B/ns.
    // `dispatch_ns` — THE MARGINAL COST OF ONE DISPATCH IN THIS RUNTIME, which
    //   is NOT the parked-wave figure. `handoff_floor` measures a parked wave at
    //   22.6-41.2 us, and the first version of this constant used 30 us — but
    //   plan-s50's P0/P0b/P2/P1 and the single-slice inline rule made a real
    //   dispatch far cheaper than a cold park. Measured directly instead:
    //   `nn4_128` opens 8 dispatches and runs 0.1449 ms with them and 0.0788 ms
    //   with all eight inlined, so one dispatch costs (144.9 - 78.8)/8 = 8.3 us.
    //
    //   Using the parked figure cost three cells: `conv2d_512` 0.805x,
    //   `matmul512` 0.944x and `nn4_512` 0.945x, all of them things that pay for
    //   their dispatch at 8 us and do not at 30. The constant has to track the
    //   runtime it describes, and that is why it is derived from a program the
    //   runtime actually ran rather than from a microbenchmark.
    //
    // The estimates it separates are 4.2 us (`nn4_128` layer), 4.1 us (its
    // pack), 16.5 us (`nn4_512`/`matmul512` pack), 21 us (`conv2d_512`), 34 us
    // (`nn4_256` layer) and 84 us (`conv2d_1024`) — so the boundary sits in a
    // clear gap between 4.2 and 16.5, and any value in it answers every measured
    // cell the same way.
    dispatch: Some(DispatchCost {
        flops_per_ns: 1000,
        bytes_per_ns: 127,
        dispatch_ns: 8_300,
        lanes: 0,
    }),
    ..APPLE_M
};

/// ADR-0046 D7: **a host whose float unit has only IEEE `+ − ×`** — no
/// `sqrt` instruction, no function unit (an accelerator's scalar core, an
/// ASIC's control unit): it declares no math unit, so every math builtin is
/// its plain-operation reference (`sqrt` the digit recurrence, `sin` /
/// `cos` the integer Payne–Hanek reduction), with the exact faces' bits.
/// The generic host's geometry otherwise.
pub const PLAIN_FP: TargetProfile = TargetProfile {
    name: "plain-fp",
    math: MathTarget::NONE,
    ..GENERIC
};

pub(super) const PROFILES: [&TargetProfile; 16] = [
    &GENERIC,
    &APPLE_M,
    &ZEN3,
    &CUDA_ADA,
    &RTX_4090,
    &H100,
    &H100_PCIE,
    &RTX_5090,
    &RTX_5080,
    &RTX_3090,
    &A100,
    &APPLE_M4_SME,
    &RAPTORLAKE,
    &RAPTORLAKE_INLINE,
    &X86_64_V3,
    &PLAIN_FP,
];

/// Resolve a profile by name. `None` for an unknown name — never a silent
/// fallback to `generic`, because a typo that quietly emits the default
/// profile's numbers is the exact failure this table exists to remove.
///
/// **A hand-written profile always wins.** The table is checked first, so
/// `--target=apple-m4-sme` overrides detection; `native` is only ever reached by
/// asking for it by name. That ordering is the contract: detection removes the
/// need to hand-write a profile for the machine you are sitting on, and changes
/// nothing about describing a machine you are not.
pub fn resolve(name: &str) -> Option<&'static TargetProfile> {
    PROFILES
        .into_iter()
        .find(|p| p.name == name)
        .or_else(|| (name == "native").then(native).flatten())
        // hardware-properties: this machine's GPU, from its device query and
        // its calibration file; `None` (an unknown name) without a device.
        .or_else(|| (name == NATIVE_GPU).then(native_gpu).flatten())
        // S46: `native` is the DEFAULT target, so it must always resolve. When a
        // fact cannot be read — an unknown OS, a locked-down container, a CPU
        // whose caches sysfs does not describe — fall back to `generic` rather
        // than fail the build. `generic` carries no cache geometry, so every
        // rung that needs it declines: the compiler goes conservative instead of
        // guessing, which is the same contract as before this became default.
        .or_else(|| (name == "native").then_some(&GENERIC))
}

impl TargetProfile {
    /// [`resolve`] from outside the crate: a test's oracle reads the same
    /// facts the emitter derived from (S62 `splitk`).
    pub fn named(name: &str) -> Option<&'static TargetProfile> {
        resolve(name)
    }
}

/// The known profile names, for the error message on an unknown one.
pub fn names() -> String {
    PROFILES
        .iter()
        .map(|p| p.name)
        .chain(["native", NATIVE_GPU])
        .collect::<Vec<_>>()
        .join(", ")
}
