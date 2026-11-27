//! The machine's place ids and the profile-fact helpers (provenance, measured points, latencies).
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

// ---- the machine ------------------------------------------------------------

/// Place ids of the GPU translation.
pub const REGS: usize = 0;
pub const SMEM: usize = 1;
pub const L2: usize = 2;
pub const DRAM: usize = 3;
pub const TMEM: usize = 4;
pub const HOST: usize = 5;

/// Executor ids of the GPU translation.
pub(super) const E_DEVICE: usize = 0;
pub(super) const E_UNIT: usize = 1;
pub(super) const E_SUB: usize = 2;
pub(super) const E_GROUP: usize = 3;
pub(super) const E_LANE: usize = 4;

/// Operation ids every GPU machine lists first: the FMA lane and the
/// separate multiply and add (the exact face's two instructions a step).
pub const OP_FFMA: usize = 0;
pub const OP_MULADD: usize = 1;

/// The source of profile field `name` ([`crate::profile::PlaceFacts::prov`]):
/// listed (the native profile lists its device-query attributes as
/// queried), else a geometry field a hand profile spells (declared: its
/// datasheet, or a device query once — not this process's), else unlabeled
/// (nominal).
pub(super) fn src(g: &Gpu, name: &str) -> Src {
    if let Some((_, s)) = g.place.prov.iter().find(|(n, _)| *n == name) {
        return *s;
    }
    match name {
        "regs_per_thread" => Src::Declared("the ISA's registers a thread"),
        "schedulers" => Src::Declared("the SM design's warp schedulers"),
        n if QUERIED.contains(&n) => {
            Src::Declared("the hand profile's geometry (datasheet or a past query)")
        }
        _ => Src::Nominal("unlabeled profile value"),
    }
}

/// The geometry fields a device query reads (`cuDeviceGetAttribute`):
/// queried on the native profile, declared on a hand one ([`src`]).
pub const QUERIED: [&str; 12] = [
    "warp",
    "max_threads_per_block",
    "sms",
    "regs_per_sm",
    "smem_per_sm",
    "smem_per_block",
    "smem_reserved_per_block",
    "max_threads_per_sm",
    "max_blocks_per_sm",
    "l2_bytes",
    "mem_clock_khz",
    "mem_bus_bits",
];

/// A profile value as a fact: `v` with its source, an interval only where the
/// source is marked (`[lo, hi]` × `v`).
pub(super) fn fact(g: &Gpu, name: &str, v: f64, spread: (f64, f64)) -> Fact<f64> {
    let s = src(g, name);
    match s.marked() {
        true => Fact {
            v,
            lo: v * spread.0,
            hi: v * spread.1,
            src: s,
        },
        false => Fact::point(v, s),
    }
}

/// A geometry field's value with its source ([`src`]).
pub(super) fn geo(g: &Gpu, name: &str, v: u64) -> Fact<u64> {
    Fact::point(v, src(g, name))
}

/// The machine format of a matrix format.
pub(super) fn fmt_of(f: MmaFmt) -> Fmt {
    match f {
        MmaFmt::Tf32 => Fmt::Tf32,
        MmaFmt::Bf16 => Fmt::Bf16,
        MmaFmt::F16 => Fmt::F16,
        MmaFmt::E4m3 => Fmt::E4m3,
        MmaFmt::E5m2 => Fmt::E5m2,
    }
}

pub(super) fn home_places(h: MHome) -> &'static [usize] {
    match h {
        MHome::Regs => &[REGS],
        MHome::Smem => &[SMEM],
        MHome::Tmem => &[TMEM],
    }
}

/// An operation row's name from its geometry (display only).
pub(super) fn op_name(o: &MatrixOp) -> &'static str {
    let s = format!(
        "{}.m{}n{}k{}.{}{}",
        if o.shape.threads == 1 {
            "tc1"
        } else if o.shape.threads > 32 {
            "wg"
        } else {
            "mma"
        },
        o.shape.m,
        o.shape.n,
        o.shape.k,
        o.fmt.ptx(),
        if o.shape.threads > 32 && o.a == MHome::Regs {
            ".rs"
        } else {
            ""
        }
    );
    Box::leak(s.into_boxed_str())
}

/// The profile field naming the rate of matrix operation `o`.
pub(super) fn rate_field(o: &MatrixOp) -> &'static str {
    match (o.fmt, o.shape.threads > 32) {
        (MmaFmt::Tf32, false) => "mma_tf32_per_clk",
        (MmaFmt::Tf32, true) => "wgmma_tf32_per_clk",
        (_, false) => "mma_half_per_clk",
        (_, true) => "wgmma_half_per_clk",
    }
}

/// **Whether probe row `row` measures operation `name`** of `n`s: its own
/// name, or — a family whose `n` the schedule chooses (a warpgroup's, a
/// single thread's: `NSet::Step`) — the family's form at an `n` it admits,
/// every other part of the geometry, the format and the operands' source
/// equal (the H100's `wg.m64n128k16.bf16` measures the `wg.m64n256k16.bf16`
/// family at `n = 128`; never the `.rs` form, whose `a` comes from
/// registers).
pub(super) fn measures(row: &str, name: &str, n: NSet) -> bool {
    if row == name {
        return true;
    }
    // `<pre>.m<M>n<N>k<K>.<rest>` as (`<pre>.m<M>`, N, `k<K>.<rest>`).
    let split = |x: &str| -> Option<(String, u64, String)> {
        let (pre, geo) = x.split_once(".m")?;
        let (ni, ki) = (geo.find('n')?, geo.find('k')?);
        let nv = geo.get(ni + 1..ki)?.parse().ok()?;
        Some((format!("{pre}.m{}", &geo[..ni]), nv, geo[ki..].to_string()))
    };
    matches!(
        (split(row), split(name)),
        (Some((p, rn, r)), Some((q, _, t))) if p == q && r == t && n.admits(rn)
    )
}

/// The cite of a row `row` read for operation `name`: the part's, and the
/// member it measured where it measured the family at another `n`.
pub(super) fn row_cite(g: &Gpu, row: &str, name: &str) -> &'static str {
    match row == name {
        true => g.place.cite,
        false => Box::leak(format!("{} (measured as {row})", g.place.cite).into_boxed_str()),
    }
}

/// Measured efficiency points of operation `name` of `n`s (the part's
/// probes: its own rows, else a measured member's — [`measures`]).
pub(super) fn eff_points(g: &Gpu, name: &str, n: NSet) -> Vec<EffPoint> {
    let own = g.place.eff.iter().any(|(r, ..)| *r == name);
    g.place
        .eff
        .iter()
        .filter(|(r, ..)| {
            if own {
                *r == name
            } else {
                measures(r, name, n)
            }
        })
        .map(|&(row, chains, groups, pm)| EffPoint {
            chains: chains as f64,
            groups: groups as f64 / 1000.0,
            eff: Fact::measured(pm as f64 / 1000.0, row_cite(g, row, name)),
        })
        .collect()
}

pub(super) fn latency(g: &Gpu, name: &str, n: NSet, nominal: (f64, f64, f64)) -> Fact<f64> {
    let row = g
        .place
        .latency
        .iter()
        .find(|(r, _)| *r == name)
        .or_else(|| g.place.latency.iter().find(|(r, _)| measures(r, name, n)));
    match row {
        Some(&(row, x10)) => Fact::measured(x10 as f64 / 10.0, row_cite(g, row, name)),
        None => Fact::nominal(
            nominal.0,
            nominal.1,
            nominal.2,
            "dependent-issue latency unmeasured",
        ),
    }
}
