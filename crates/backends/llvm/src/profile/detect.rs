//! Host and CUDA detection backends and the L2-keep sweep reader.

use super::*;

/// The driver API, opened at run time: the compiler links no CUDA library
/// and builds on a machine without one.
#[cfg(target_os = "linux")]
pub mod cuda_query {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// The `CUdevice_attribute`s [`super::DeviceQuery::from_attrs`] reads.
    pub const ATTRS: [i32; 15] = [1, 10, 13, 16, 36, 37, 38, 39, 75, 76, 81, 82, 97, 106, 111];

    type CuInit = unsafe extern "C" fn(u32) -> c_int;
    type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

    /// Device `ordinal`'s [`ATTRS`]; `None` without the driver or the device.
    pub fn attrs(ordinal: i32) -> Option<Vec<(i32, u64)>> {
        // SAFETY: `dlopen`/`dlsym` with NUL-terminated names; the symbols are
        // the driver API's, called with their documented signatures.
        unsafe {
            let h = dlopen(c"libcuda.so.1".as_ptr(), 2);
            if h.is_null() {
                return None;
            }
            let sym = |n: &std::ffi::CStr| {
                let p = dlsym(h, n.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let init: CuInit = std::mem::transmute(sym(c"cuInit")?);
            let get: CuDeviceGet = std::mem::transmute(sym(c"cuDeviceGet")?);
            let attr: CuDeviceGetAttribute = std::mem::transmute(sym(c"cuDeviceGetAttribute")?);
            if init(0) != 0 {
                return None;
            }
            let mut dev: c_int = 0;
            if get(&mut dev, ordinal) != 0 {
                return None;
            }
            ATTRS
                .iter()
                .map(|&a| {
                    let mut v: c_int = 0;
                    match attr(&mut v, a, dev) {
                        0 => Some((a, u64::try_from(v).ok()?)),
                        _ => None,
                    }
                })
                .collect()
        }
    }
}

/// No driver API off Linux: no native GPU.
#[cfg(not(target_os = "linux"))]
pub mod cuda_query {
    pub const ATTRS: [i32; 15] = [1, 10, 13, 16, 36, 37, 38, 39, 75, 76, 81, 82, 97, 106, 111];
    pub fn attrs(_ordinal: i32) -> Option<Vec<(i32, u64)>> {
        None
    }
}

#[cfg(target_os = "macos")]
pub(super) fn detect() -> Option<TargetProfile> {
    // perflevel0 is the performance core cluster; the bare hw.* keys are the
    // efficiency cores. See the doc comment — this distinction is load-bearing.
    let l1d_bytes = sysctl_u64("hw.perflevel0.l1dcachesize")?;
    let line = sysctl_u64("hw.cachelinesize")?;
    Some(TargetProfile {
        name: "native",
        l2_bytes: sysctl_u64("hw.perflevel0.l2cachesize")?,
        // 5 on this M4 Pro: the 16 MB is a CLUSTER L2, not a per-core one, and
        // `move_block`'s cost term is wrong by 4.8x without this.
        l2_cores: sysctl_u64("hw.perflevel0.cpusperl2")?.max(1),
        sme: (sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1))
            .then(|| {
                Some(Sme {
                    svl_bytes: sysctl_u64("hw.optional.arm.sme_max_svl_b")?,
                    l1d_bytes,
                    // Detected profiles inherit the swept ratio; it is policy, not
                    // a property of the silicon, so there is nothing to read.
                    panel_l1d_ratio: 2,
                })
            })
            .flatten(),
        // S45: line and page are plain sysctls; sets and ways are derived from
        // them, so a detected profile is a COMPLETE one for this rung too.
        // Line is a plain sysctl; SETS is the one fact macOS does not expose,
        // so it comes from the page bound — see `L1d` for why that is a checked
        // heuristic and what it gets wrong. On this part it yields 128, which is
        // the value S44 verified by a stride probe.
        l1d: Some(L1d {
            bytes: l1d_bytes,
            line_bytes: line,
            sets: (sysctl_u64("hw.pagesize")? / line).max(1),
        }),
        vec_bytes: vec_geometry().0,
        vec_regs: vec_geometry().1,
        // `native` is the DEFAULT target, so the dispatch economics have to
        // reach it or the rule only fires under an explicit
        // `--target=apple-m4-sme`. Inherited ONLY when the detected part is the
        // one they were measured on — SME with a 64-byte SVL — and `None`
        // otherwise, because a threshold is a machine fact and guessing it for
        // an unmeasured part is what this design refuses.
        dispatch: match sysctl_u64("hw.optional.arm.sme_max_svl_b") {
            Some(64) if sysctl_u64("hw.optional.arm.FEAT_SME") == Some(1) => APPLE_M4_SME.dispatch,
            _ => None,
        },
        ..GENERIC
    })
}

/// One `sysfs` file parsed as a `u64`, tolerating the `48K` / `2048K` spelling
/// the cache nodes use for sizes.
#[cfg(target_os = "linux")]
fn sysfs_u64(path: &str) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let t = raw.trim();
    match t.strip_suffix('K') {
        Some(k) => Some(k.parse::<u64>().ok()? * 1024),
        None => t.parse().ok(),
    }
}

/// The CPUs a sysfs CPU list names: "0-1" -> 2, "0,4" -> 2, "0" -> 1; at
/// least 1.
#[cfg(target_os = "linux")]
pub(super) fn cpu_list_len(list: &str) -> u64 {
    list.trim()
        .split(',')
        .map(|part| match part.split_once('-') {
            Some((a, b)) => match (a.parse::<u64>(), b.parse::<u64>()) {
                (Ok(a), Ok(b)) if b >= a => b - a + 1,
                _ => 1,
            },
            None => 1,
        })
        .sum::<u64>()
        .max(1)
}

/// The same facts on Linux — a different parser, not a different idea.
///
/// `index0` is the L1 data cache and `index2` the unified L2 on every part this
/// runs on; `shared_cpu_list` is what makes `l2_cores` a measurement rather than
/// an assumption (`0-1` on the i9 is two SMT threads of ONE physical core, so
/// the answer is 1 core, not 2). SMT siblings are collapsed via
/// `topology/thread_siblings_list`, because a hyperthread is not a core to
/// budget for.
///
/// No SME on x86, so `sme` stays `None` and the SME realization cannot be
/// selected — the same seam as every pre-S41 profile.
#[cfg(target_os = "linux")]
pub(super) fn detect() -> Option<TargetProfile> {
    let l1 = "/sys/devices/system/cpu/cpu0/cache/index0";
    let l2 = "/sys/devices/system/cpu/cpu0/cache/index2";
    let count =
        |path: &str| -> u64 { std::fs::read_to_string(path).map_or(1, |s| cpu_list_len(&s)) };
    let smt = count("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    Some(TargetProfile {
        name: "native",
        l2_bytes: sysfs_u64(&format!("{l2}/size"))?,
        l2_cores: (count(&format!("{l2}/shared_cpu_list")) / smt).max(1),
        // Linux EXPOSES the set count, so it is read rather than derived —
        // the macOS page heuristic is a fallback for a host that hides it, and
        // must never shadow a readable truth (`L1d`).
        l1d: Some(L1d {
            bytes: sysfs_u64(&format!("{l1}/size"))?,
            line_bytes: sysfs_u64(&format!("{l1}/coherency_line_size"))?,
            sets: sysfs_u64(&format!("{l1}/number_of_sets"))?,
        }),
        vec_bytes: vec_geometry().0,
        vec_regs: vec_geometry().1,
        // local-rtx-5080: before this, Linux `native` carried no dispatch
        // economics, so every split task woke the whole pool — 263 us a wave on
        // a 64-thread 9970X, which made `saxpy_1M` 3.8x and `conv2d_1024` 2.7x
        // SLOWER threaded than on one lane. They are this host's, so measured.
        dispatch: host_dispatch::get(),
        ..GENERIC
    })
}

/// No detection on other hosts. `None` means `--target=native` errors with the
/// known-name list instead of guessing.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(super) fn detect() -> Option<TargetProfile> {
    None
}

/// review: **the L2 keep read off an `l2keep.cu` sweep** — `(ws / L2,
/// GB/s)` points of a cyclic stream re-read back to back, and the part's DRAM
/// read rate. The plateau is the sweep's highest rate (the stream's own
/// re-read rate of a set the L2 holds: the 5080's 4.1–4.3 TB/s from 0.25 ×,
/// the 3090's 1.2 TB/s — not the derived L2 rate, which the 3090's stream never
/// reaches); the keep is the largest ratio re-read at no less than the
/// plateau–DRAM midpoint. `None` where the sweep has no plateau (its highest
/// rate within 1.25 × the DRAM rate, beyond the sweep's ±2 % repeat noise:
/// every point past the keep — the A100 / H100 from 1.0 ×).
/// Read by the calibration tests (`l2_keep_read_off_the_sweeps`) only.
#[cfg_attr(not(test), allow(dead_code))]
pub fn l2_keep_of_sweep(sweep: &[(f64, f64)], dram_gbs: f64) -> Option<f64> {
    let plateau = sweep.iter().map(|p| p.1).fold(0.0, f64::max);
    if plateau < 1.25 * dram_gbs {
        return None;
    }
    let mid = (plateau + dram_gbs) / 2.0;
    sweep
        .iter()
        .filter(|p| p.1 >= mid)
        .map(|p| p.0)
        .fold(None, |m, r| Some(m.map_or(r, |m: f64| m.max(r))))
}
