//! Calibration supply and the device query: `CalibSource`, `DeviceQuery`, `calib_line`, `place_line`.

use super::*;

/// Where the `native-gpu` build's calibration line came from ([`calib_supply`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibSource {
    /// `MAPAL_GPU_CALIB` or the user's cache file: its path.
    File(String),
    /// A line this repository carries ([`COMMITTED_CALIB`]).
    Committed,
    /// No line for the device's key: what was tried.
    None(String),
}

pub(super) static NATIVE_GPU_CALIB: std::sync::OnceLock<CalibSource> = std::sync::OnceLock::new();

/// The source of this process's `native-gpu` calibration, once resolved.
pub fn native_gpu_calib() -> Option<&'static CalibSource> {
    NATIVE_GPU_CALIB.get()
}

/// residency `calib-supply`: **the measured calibration lines this repository
/// carries**, keyed like every line by the device query's physical facts
/// ([`DeviceQuery::calib_key`], never a name): a part someone measured and
/// committed is priced on every machine that has it, with no manual step.
pub const COMMITTED_CALIB: &[&str] = &[include_str!(
    "../../../../../benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib"
)];

/// The calibration text whose line for `key` a `native-gpu` build reads:
/// `MAPAL_GPU_CALIB`'s file, else the user's cache file
/// ([`gpu_calib_path`]), each only when it holds a line for `key`, else the
/// committed lines ([`COMMITTED_CALIB`]). `MAPAL_GPU_CALIB=none`: no line
/// (the measurement arm of an uncalibrated build), said as such.
pub fn calib_supply(key: &str) -> (CalibSource, Option<String>) {
    if std::env::var("MAPAL_GPU_CALIB").is_ok_and(|v| v == "none") {
        return (CalibSource::None("MAPAL_GPU_CALIB=none".into()), None);
    }
    let mut tried = Vec::new();
    if let Some(p) = gpu_calib_path() {
        tried.push(p.display().to_string());
        if let Ok(t) = std::fs::read_to_string(&p)
            && calib_line(&t, key).is_some()
        {
            return (CalibSource::File(p.display().to_string()), Some(t));
        }
    }
    if let Some(t) = COMMITTED_CALIB
        .iter()
        .find(|t| calib_line(t, key).is_some())
    {
        return (CalibSource::Committed, Some(t.to_string()));
    }
    tried.push("the committed lines".into());
    (CalibSource::None(tried.join(", ")), None)
}

/// The `cuDeviceGetAttribute` facts a GPU profile is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceQuery {
    /// `major·10 + minor`.
    pub cc: u64,
    pub sms: u64,
    pub warp: u64,
    pub max_threads_per_block: u64,
    pub max_threads_per_sm: u64,
    pub max_blocks_per_sm: u64,
    pub regs_per_sm: u64,
    pub smem_per_sm: u64,
    pub smem_per_block: u64,
    pub smem_reserved_per_block: u64,
    pub l2_bytes: u64,
    pub clock_khz: u64,
    pub mem_clock_khz: u64,
    pub mem_bus_bits: u64,
}

/// FP32 FMA lanes per SM, the SM designs' table by compute capability (64 on
/// the `sm_60`/`sm_80` datacenter SMs, 128 on the others since `sm_86`): the
/// fallback when the calibration file does not give `fp32_lanes` (its
/// measured FFMA peak over `sms · 2 · clock`).
pub const fn fp32_lanes_of_cc(cc: u64) -> u64 {
    match cc {
        60 | 80 => 64,
        _ => 128,
    }
}

/// f32-tensor-cores P1: the warp TF32, warp 16-bit (f32 accumulate) and
/// warpgroup TF32 and 16-bit products' flops an SM a clock, the SM designs'
/// table by compute capability — the fallback when the calibration file does
/// not give `mma_tf32_per_clk` / `mma_half_per_clk` / `wgmma_tf32_per_clk` /
/// `wgmma_half_per_clk` (`calib.cu matrix`), as [`fp32_lanes_of_cc`] for the
/// FFMA lanes: the datacenter `sm_80` SM 1012 / 2024 (the A100 measured); the
/// Hopper SM's `mma.sync` 1351 / 2702 (the H100 measured, [`H100`]) and its
/// `wgmma` 2048 / 4096 (datasheet); every other SM since `sm_86` 256 / 512
/// (the RTX 5080, 3090, 4090 measured). No tensor core below `sm_80`.
pub const fn mma_rates_of_cc(cc: u64) -> [u64; 4] {
    match cc {
        0..80 => [0, 0, 0, 0],
        80 => [1012, 2024, 0, 0],
        90 => [1351, 2702, 2048, 4096],
        _ => [256, 512, 0, 0],
    }
}

impl DeviceQuery {
    /// Device `ordinal`'s facts through the driver API (`libcuda.so.1`,
    /// opened at run time); `None` without a driver or that device.
    pub fn device(ordinal: i32) -> Option<Self> {
        cuda_query::attrs(ordinal).map(|a| Self::from_attrs(&a))
    }

    /// The facts from `(attribute, value)` pairs ([`cuda_query::ATTRS`]).
    pub fn from_attrs(a: &[(i32, u64)]) -> Self {
        let get = |k: i32| a.iter().find(|&&(n, _)| n == k).map_or(0, |&(_, v)| v);
        Self {
            cc: get(75) * 10 + get(76),
            sms: get(16),
            warp: get(10),
            max_threads_per_block: get(1),
            max_threads_per_sm: get(39),
            max_blocks_per_sm: get(106),
            regs_per_sm: get(82),
            smem_per_sm: get(81),
            smem_per_block: get(97),
            smem_reserved_per_block: get(111),
            l2_bytes: get(38),
            clock_khz: get(13),
            mem_clock_khz: get(36),
            mem_bus_bits: get(37),
        }
    }

    /// The calibration file's key for this device: its physical facts, so a
    /// line measured on one part never prices another.
    pub fn calib_key(&self) -> String {
        format!(
            "cc={} sms={} l2={} mem={}x{}",
            self.cc, self.sms, self.l2_bytes, self.mem_clock_khz, self.mem_bus_bits
        )
    }

    /// **The profile**: these facts, the capabilities the device's compute
    /// capability admits ([`caps_of_cc`]), and the rates the calibration
    /// text's line for [`Self::calib_key`] measured (`name=value` over
    /// [`DeviceCost::NAMES`], the `--cost=` names; none: no measured costs).
    pub fn profile(&self, calib: Option<&str>) -> TargetProfile {
        let arch_specific = self.cc == 90;
        let arch: &'static str = Box::leak(
            format!("sm_{}{}", self.cc, if arch_specific { "a" } else { "" }).into_boxed_str(),
        );
        let gpu = Gpu {
            warp: self.warp,
            max_threads_per_block: self.max_threads_per_block,
            arch,
            sms: self.sms,
            regs_per_thread: 255,
            regs_per_sm: self.regs_per_sm,
            smem_per_sm: self.smem_per_sm,
            smem_per_block: self.smem_per_block,
            smem_reserved_per_block: self.smem_reserved_per_block,
            max_threads_per_sm: self.max_threads_per_sm,
            max_blocks_per_sm: self.max_blocks_per_sm,
            l2_bytes: self.l2_bytes,
            cost: None,
            geom: GEOM,
            fp32_lanes: fp32_lanes_of_cc(self.cc),
            clock_khz: self.clock_khz,
            mem_clock_khz: self.mem_clock_khz,
            mem_bus_bits: self.mem_bus_bits,
            schedulers: 4,
            bar_cycles: bar_cycles_of_cc(self.cc),
            mma_tf32_per_clk: mma_rates_of_cc(self.cc)[0],
            mma_half_per_clk: mma_rates_of_cc(self.cc)[1],
            wgmma_tf32_per_clk: if arch_specific {
                mma_rates_of_cc(self.cc)[2]
            } else {
                0
            },
            wgmma_half_per_clk: if arch_specific {
                mma_rates_of_cc(self.cc)[3]
            } else {
                0
            },
            caps: caps_of_cc(self.cc, arch_specific),
            place: &PLACE_NONE,
        };
        let over = calib.and_then(|t| calib_line(t, &self.calib_key()));
        // ADR-0050: the placement probes' facts the line carries (start
        // timing, keeps per pattern, hit curve, L2 latency, efficiency
        // points, latencies, sustained clock, barrier cycles) — the device's
        // own, as the hand profile of the same part spells them.
        let placed = calib.and_then(|t| place_line(t, &self.calib_key()));
        // ADR-0050 provenance: a calibration line's values are measured; the
        // compute-capability tables' are derived (marked); the rest is the
        // device query's.
        let mut prov: Vec<(&'static str, Src)> = Vec::new();
        for (name, v) in DeviceCost::NAMES.iter().zip(over.iter().flatten()) {
            if v.is_some() {
                prov.push((name, Src::Measured("calibration file")));
            }
        }
        let gpu = match placed.and_then(|(_, bar)| bar) {
            Some(bar) => {
                prov.push((
                    "bar_cycles",
                    Src::Measured("calibration file (calib.cu bar)"),
                ));
                Gpu {
                    bar_cycles: bar,
                    ..gpu
                }
            }
            None => gpu,
        };
        // The device query's own attributes: queried (`place::QUERIED`).
        for name in crate::place::QUERIED {
            prov.push((name, Src::Queried));
        }
        for name in [
            "fp32_lanes",
            "bar_cycles",
            "mma_tf32_per_clk",
            "mma_half_per_clk",
            "wgmma_tf32_per_clk",
            "wgmma_half_per_clk",
        ] {
            if !prov.iter().any(|(n, _)| *n == name) {
                prov.push((name, Src::Derived("compute-capability table")));
            }
        }
        let place: &'static PlaceFacts = Box::leak(Box::new(PlaceFacts {
            prov: Box::leak(prov.into_boxed_slice()),
            ..placed.map_or(PLACE_NONE, |(f, _)| f)
        }));
        let gpu = Gpu { place, ..gpu };
        TargetProfile {
            name: NATIVE_GPU,
            machine: Machine::Gpu(match over {
                Some(o) => gpu.with_cost(&o),
                None => gpu,
            }),
            ..GENERIC
        }
    }
}

/// The calibration file: `MAPAL_GPU_CALIB`, else
/// `$XDG_CACHE_HOME/mapal/gpu-calib` (`~/.cache` without it). One line per
/// device, `<calib key>\t<name>=<value> …`; `#` starts a comment.
pub fn gpu_calib_path() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("MAPAL_GPU_CALIB") {
        return Some(p.into());
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))?;
    Some(base.join("mapal").join("gpu-calib"))
}

/// The rates `text` gives the device keyed `key` (`None` without a line).
pub fn calib_line(text: &str, key: &str) -> Option<CostOver> {
    let line = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .find(|l| l.split('\t').next() == Some(key))?;
    let mut over: CostOver = [None; DeviceCost::NAMES.len()];
    for kv in line.split('\t').skip(1).flat_map(str::split_whitespace) {
        let (k, v) = kv.split_once('=')?;
        // ADR-0050: the placement probes' facts ([`place_line`]).
        if k.starts_with("place.") {
            continue;
        }
        let i = DeviceCost::NAMES.iter().position(|n| *n == k)?;
        over[i] = Some(v.parse().ok()?);
    }
    Some(over)
}

/// **ADR-0050: the placement facts a calibration line measured** — the
/// `place.` names beside the `--cost=` ones (`benches/local/place_calib.cu`;
/// `None`: the line has none): `place.start=lockstep|stagger`,
/// `place.keep_rotated` / `place.keep_strided` (bytes), `place.keep_hi` (the
/// sweep's next point: every keep's resolution), `place.hit` (`x:y`
/// pairs, reuse distance ÷ cyclic keep and hit fraction, per mille),
/// `place.lat_l2_ns`, `place.eff` (`op:chains:groups×1000:per mille`),
/// `place.latency` (`op:clocks×10`), `place.clock_khz` (sustained),
/// `place.bar_cycles` (the barrier's measured cycles), and `place.prim`
/// (the cost model's primitives: `name:rate:lo:hi:latency:lo:hi:conc:lo:hi`
/// in milli-units, [`PlaceFacts::prim`]; `place.prim_ghz`, the clock the
/// probe converted with, is evidence only). An unknown `place.` name or a
/// malformed value rejects the line's placement facts.
pub fn place_line(text: &str, key: &str) -> Option<(PlaceFacts, Option<u64>)> {
    let line = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .find(|l| l.split('\t').next() == Some(key))?;
    let leak = |x: &str| -> &'static str { Box::leak(x.to_string().into_boxed_str()) };
    let mut f = PLACE_NONE;
    let mut bar = None;
    let mut any = false;
    for kv in line.split('\t').skip(1).flat_map(str::split_whitespace) {
        let (k, v) = kv.split_once('=')?;
        let Some(k) = k.strip_prefix("place.") else {
            continue;
        };
        any = true;
        let num = |x: &str| x.parse::<u64>().ok();
        match k {
            "start" => {
                let st = match v {
                    "lockstep" => mapal_place::Start::Lockstep,
                    "stagger" => mapal_place::Start::Stagger,
                    _ => return None,
                };
                f.start = Some((st, "calibration file (place_calib.cu start)"));
            }
            "keep_rotated" => f.keep_rotated = num(v)?,
            "keep_strided" => f.keep_strided = num(v)?,
            "keep_hi" => f.keep_hi = num(v)?,
            "lat_l2_ns" => f.lat_l2_ns = num(v)?,
            "clock_khz" => f.clock_khz = num(v)?,
            "bar_cycles" => bar = Some(num(v)?),
            "hit" => {
                let pts: Option<Vec<(u64, u64)>> = v
                    .split(',')
                    .map(|xy| {
                        let (x, y) = xy.split_once(':')?;
                        Some((num(x)?, num(y)?))
                    })
                    .collect();
                f.hit_curve = Box::leak(pts?.into_boxed_slice());
            }
            "eff" => {
                let pts: Option<Vec<(&'static str, u64, u64, u64)>> = v
                    .split(',')
                    .map(|e| {
                        let mut it = e.rsplitn(4, ':');
                        let (pm, g, c, op) = (it.next()?, it.next()?, it.next()?, it.next()?);
                        Some((leak(op), num(c)?, num(g)?, num(pm)?))
                    })
                    .collect();
                f.eff = Box::leak(pts?.into_boxed_slice());
            }
            "latency" => {
                let pts: Option<Vec<(&'static str, u64)>> = v
                    .split(',')
                    .map(|e| {
                        let (op, x10) = e.rsplit_once(':')?;
                        Some((leak(op), num(x10)?))
                    })
                    .collect();
                f.latency = Box::leak(pts?.into_boxed_slice());
            }
            "prim" => {
                let rows: Option<Vec<(&'static str, [u64; 9])>> = v
                    .split(',')
                    .map(|e| {
                        let f: Vec<&str> = e.split(':').collect();
                        if f.len() != 10 {
                            return None;
                        }
                        let mut a = [0u64; 9];
                        for (i, x) in f[1..].iter().enumerate() {
                            a[i] = num(x)?;
                        }
                        Some((leak(f[0]), a))
                    })
                    .collect();
                f.prim = Box::leak(rows?.into_boxed_slice());
            }
            "prim_ghz" => {
                num(v)?;
            }
            _ => return None,
        }
    }
    any.then(|| {
        f.cite = "calibration file (place_calib.cu)";
        (f, bar)
    })
}
