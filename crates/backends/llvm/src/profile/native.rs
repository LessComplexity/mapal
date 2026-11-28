//! Native detection: `native`, `native_gpu`, host probes.

use super::*;

/// One `sysctl` integer, or `None` if the key does not exist on this host.
///
/// Shelling out rather than linking `libc`: this crate has two dependencies and
/// the emitter is a build-time tool that reads four keys once per run. A binding
/// for `sysctlbyname` would be a new dependency to save a process spawn that
/// happens at most once.
#[cfg(target_os = "macos")]
pub(super) fn sysctl_u64(key: &str) -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// The host's own facts, **detected rather than written down**.
///
/// Reached as `--target=native`. Named profiles are resolved first (see
/// [`resolve`]), so this never shadows a hand-written one.
///
/// **Every machine fact is detected; nothing about the part is written down.**
///
/// | fact | source | |
/// | --- | --- | --- |
/// | `l2_bytes` | `hw.perflevel0.l2cachesize` | 16 MB here |
/// | `l1d_bytes` | `hw.perflevel0.l1dcachesize` | 128 KB here |
/// | `svl_bytes` | `hw.optional.arm.sme_max_svl_b` | 64 here |
/// | SME present | `hw.optional.arm.FEAT_SME` | 1 here |
/// | tile count | **derived** — ISA rule, see [`TargetProfile::sme_block`] | |
///
/// **`perflevel0`, not the bare keys.** `hw.l1dcachesize` reports **65536** on
/// this part and `hw.perflevel0.l1dcachesize` reports **131072**: the bare key
/// describes the E-cores. Sizing a P-core matrix panel against an E-core cache
/// would halve the derived k-panel and silently cost throughput, which is the
/// kind of wrong number that never announces itself.
///
/// SVL turned out to be a plain sysctl (`sme_max_svl_b`), so no streaming-mode
/// probe and no `+sme` build of this emitter is needed — a detected profile is a
/// **complete** profile, matrix unit included. `FEAT_SME` gates the block, so a
/// part without the unit yields `sme: None` and the rung falls back exactly as
/// it does for every pre-S41 profile.
///
/// `vec_bytes`/`vec_regs` come from [`vec_geometry`]. They ARE ISA facts rather
/// than sysfs entries, but `native` is now the DEFAULT target (S46), so leaving
/// them at [`GENERIC`]'s NEON values silently tiled an AVX2 part to a 16-byte
/// vector — measured at +8.8% on conv2d before this was read. `acc_vecs_per_row`/`nc_tiles` are
/// policy ratios, honestly search space (ADR-0034), not facts to read.
/// The host's vector geometry, read at runtime.
///
/// `native` means "this machine", and on x86 the vector width is not one value:
/// an AVX-512 part, an AVX2 part and an SSE-only part want 64, 32 and 16 bytes
/// with 32, 16 and 16 architectural registers. Detecting the caches but assuming
/// the ISA is how a default build tiled an AVX2 box to NEON's 16 bytes.
/// aarch64 keeps `GENERIC`'s 16/32, which is NEON and correct; SVE would be a
/// separate rung, not a wider `vec_bytes`.
pub(super) fn vec_geometry() -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return (64, 32);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return (32, 16);
        }
        return (16, 16);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (GENERIC.vec_bytes, GENERIC.vec_regs)
    }
}

pub fn native() -> Option<&'static TargetProfile> {
    static NATIVE: std::sync::OnceLock<Option<TargetProfile>> = std::sync::OnceLock::new();
    NATIVE.get_or_init(detect).as_ref()
}

/// The profile name of this machine's GPU ([`native_gpu`]).
pub const NATIVE_GPU: &str = "native-gpu";

/// **hardware-properties: this machine's GPU as physical properties** —
/// device 0's `cuDeviceGetAttribute` facts ([`DeviceQuery`]) and the
/// measured rates of its calibration line ([`calib_supply`]); never a name
/// lookup. `None` without a CUDA driver or device.
pub fn native_gpu() -> Option<&'static TargetProfile> {
    static NATIVE_GPU_PROFILE: std::sync::OnceLock<Option<TargetProfile>> =
        std::sync::OnceLock::new();
    NATIVE_GPU_PROFILE
        .get_or_init(|| {
            let q = DeviceQuery::device(0)?;
            let (src, text) = calib_supply(&q.calib_key());
            let _ = NATIVE_GPU_CALIB.set(src.clone());
            if let CalibSource::None(tried) = &src {
                // residency `calib-supply`: never unpriced in silence.
                eprintln!(
                    "mapal: warning: native-gpu: no calibration line for `{}` (tried {tried}); \
                     building with unmeasured device costs — PDL and every priced rule fall back. \
                     Measure the part (`benches/gpu/calib.cu`, the line format of \
                     `benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib`) and supply its \
                     line by `MAPAL_GPU_CALIB=<file>` or `{}`.",
                    q.calib_key(),
                    gpu_calib_path().map_or("$XDG_CACHE_HOME/mapal/gpu-calib".into(), |p| p.display().to_string()),
                );
            }
            Some(q.profile(text.as_deref()))
        })
        .as_ref()
}
