//! **ADR-0050 C2: the GPU's translation into the machine model, and the
//! present emitters' lowering declarations.**
//!
//! - [`machine_of`]: a GPU [`TargetProfile`] as a [`mapal_place::Machine`] —
//!   the profile's physical properties as facts, each with its provenance
//!   ([`crate::profile::PlaceFacts`]); a missing fact takes a marked derived
//!   or nominal interval; no value is borrowed from another part as if
//!   measured and no table by version is read.
//! - The lowerings of the present emitters — the one-thread-per-cell kernel,
//!   the FMA tile, the warp `mma.sync` tile (TF32, 16-bit, split forms), the
//!   warpgroup `wgmma` tile and its warp-specialized form — each declaring
//!   its constraints, its footprint, its instruction counts per class per
//!   trip and its ring's in-flight discipline ([`mapal_place::Lowering`]),
//!   and the coordinates of its candidate space (D5 steps 2–3) as domains
//!   over the machine's facts.
//! - [`schedule_of`] / [`launch_of`]: the rules' [`Launch`] as a schedule
//!   and back. The rungs a launch carries beyond the site's table
//!   ([`crate::func::gpu::site_rungs`]) are the D6 steps a lowering keeps
//!   (`params` `rung+` / `rung-`), listed by the coverage test.

#![cfg_attr(not(test), allow(dead_code))]

use crate::func::gpu::{
    self, Arithmetic, Combine, Launch, NAIVE_THREADS, cell_blocks, elem_bytes_of, fill_grid,
    site_rungs,
};
use crate::profile::{
    DeviceCost, Face, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, Home as MHome,
    MatrixOp, MmaFmt, NAIVE_UNROLL, NO_PLAN, TargetProfile, WARPGROUP_WARPS, WGMMA_N_MAX,
    WGMMA_N_STEP, fma_held_regs,
};
use mapal_ir::{TileSite, Ty};
use mapal_place::{
    Atomics, Banking, Bind, Cache, Class, Comb, Declared, Dispatch, EffPoint, Exec, ExecKind, Fact,
    Fmt, Fold, Home, InstanceMap, Keep, LaunchCost, LaunchMode, Layout, Level, Lowering, Machine,
    Major, Mgmt, Mover, MoverKind, NSet, Op, Pipe, Place, Policy, Prim, PrimOf, Reach, Refusal,
    Res, Resource, Schedule, Site, Src, Start, TravKind, Traversal,
};

// ---- the machine ------------------------------------------------------------

/// Place ids of the GPU translation.
pub const REGS: usize = 0;
pub const SMEM: usize = 1;
pub const L2: usize = 2;
pub const DRAM: usize = 3;
pub const TMEM: usize = 4;
pub const HOST: usize = 5;

/// Executor ids of the GPU translation.
const E_DEVICE: usize = 0;
const E_UNIT: usize = 1;
const E_SUB: usize = 2;
const E_GROUP: usize = 3;
const E_LANE: usize = 4;

/// Operation ids every GPU machine lists first: the FMA lane and the
/// separate multiply and add (the exact face's two instructions a step).
pub const OP_FFMA: usize = 0;
pub const OP_MULADD: usize = 1;

/// The source of profile field `name` ([`crate::profile::PlaceFacts::prov`]):
/// listed (the native profile lists its device-query attributes as
/// queried), else a geometry field a hand profile spells (declared: its
/// datasheet, or a device query once — not this process's), else unlabeled
/// (nominal).
fn src(g: &Gpu, name: &str) -> Src {
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
fn fact(g: &Gpu, name: &str, v: f64, spread: (f64, f64)) -> Fact<f64> {
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
fn geo(g: &Gpu, name: &str, v: u64) -> Fact<u64> {
    Fact::point(v, src(g, name))
}

/// The machine format of a matrix format.
fn fmt_of(f: MmaFmt) -> Fmt {
    match f {
        MmaFmt::Tf32 => Fmt::Tf32,
        MmaFmt::Bf16 => Fmt::Bf16,
        MmaFmt::F16 => Fmt::F16,
        MmaFmt::E4m3 => Fmt::E4m3,
        MmaFmt::E5m2 => Fmt::E5m2,
    }
}

fn home_places(h: MHome) -> &'static [usize] {
    match h {
        MHome::Regs => &[REGS],
        MHome::Smem => &[SMEM],
        MHome::Tmem => &[TMEM],
    }
}

/// An operation row's name from its geometry (display only).
fn op_name(o: &MatrixOp) -> &'static str {
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
fn rate_field(o: &MatrixOp) -> &'static str {
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
fn measures(row: &str, name: &str, n: NSet) -> bool {
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
fn row_cite(g: &Gpu, row: &str, name: &str) -> &'static str {
    match row == name {
        true => g.place.cite,
        false => Box::leak(format!("{} (measured as {row})", g.place.cite).into_boxed_str()),
    }
}

/// Measured efficiency points of operation `name` of `n`s (the part's
/// probes: its own rows, else a measured member's — [`measures`]).
fn eff_points(g: &Gpu, name: &str, n: NSet) -> Vec<EffPoint> {
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

fn latency(g: &Gpu, name: &str, n: NSet, nominal: (f64, f64, f64)) -> Fact<f64> {
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

/// **The GPU's translation** (ADR-0050 D2): `None` for a CPU profile.
pub fn machine_of(p: &TargetProfile) -> Option<Machine> {
    let g = p.gpu()?;
    let d = g.cost.unwrap_or(NO_PLAN);
    let exec = vec![
        Exec {
            name: "device",
            kind: ExecKind::Device,
            parent: None,
            per_parent: Fact::declared(1, "one device"),
        },
        Exec {
            name: "sm",
            kind: ExecKind::Unit,
            parent: Some(E_DEVICE),
            per_parent: geo(g, "sms", g.sms),
        },
        Exec {
            name: "scheduler",
            kind: ExecKind::Sub,
            parent: Some(E_UNIT),
            per_parent: Fact::declared(g.schedulers(), "warp schedulers per SM"),
        },
        Exec {
            name: "warp",
            kind: ExecKind::Group,
            parent: Some(E_SUB),
            per_parent: geo(
                g,
                "max_threads_per_sm",
                g.max_threads_per_sm / g.warp.max(1) / g.schedulers(),
            ),
        },
        Exec {
            name: "lane",
            kind: ExecKind::Lane,
            parent: Some(E_GROUP),
            per_parent: geo(g, "warp", g.warp),
        },
    ];
    // Rates: measured where the profile measured them, else derived from the
    // part's own interface (marked) — never another part's number.
    let dram_peak = g.dram_peak_bytes_per_ns() as f64;
    let dram = match d.dram_bytes_per_ns {
        0 => Fact::derived(
            0.85 * dram_peak,
            0.7 * dram_peak,
            0.95 * dram_peak,
            "the DRAM interface's peak x [0.70, 0.95]",
        ),
        v => fact(g, "dram_bytes_per_ns", v as f64, (0.9, 1.1)),
    };
    let lat_dram = match d.lat_ns {
        0 => Fact::nominal(600.0, 400.0, 800.0, "loaded DRAM latency unmeasured"),
        v => fact(g, "lat_ns", v as f64, (0.8, 1.2)),
    };
    let lat_l2 = match g.place.lat_l2_ns {
        0 => Fact::nominal(
            0.55 * lat_dram.v,
            0.4 * lat_dram.v,
            0.7 * lat_dram.v,
            "loaded L2 latency unmeasured: a share of the DRAM one",
        ),
        v => Fact::measured(
            v as f64,
            match g.place.lat_l2_cite {
                "" => g.place.cite,
                c => c,
            },
        ),
    };
    let l2_total = match d.l2_bytes_per_ns {
        0 => Fact::nominal(
            4.0 * dram.v,
            2.0 * dram.v,
            6.0 * dram.v,
            "L2 read rate unmeasured: a multiple of the DRAM rate",
        ),
        v => fact(g, "l2_bytes_per_ns", v as f64, (0.9, 1.1)),
    };
    let l2_unit = match d.sm_bytes_per_ns {
        0 => Fact::nominal(
            l2_total.v / g.sms.max(1) as f64 * 2.0,
            l2_total.v / g.sms.max(1) as f64,
            l2_total.v / g.sms.max(1) as f64 * 4.0,
            "one SM's L2 rate unmeasured",
        ),
        v => fact(g, "sm_bytes_per_ns", v as f64, (0.9, 1.1)),
    };
    let keep_cyclic = match d.l2_keep_bytes {
        0 => Fact {
            v: g.l2_bytes,
            lo: g.l2_bytes / 2,
            hi: g.l2_bytes,
            src: Src::Nominal("the nominal L2: no keep sweep"),
        },
        v => match src(g, "l2_keep_bytes") {
            s if s.marked() => Fact {
                v,
                lo: g.l2_bytes / 4,
                hi: g.l2_bytes,
                src: s,
            },
            s => Fact {
                hi: g.place.keep_hi.max(v),
                ..Fact::point(v, s)
            },
        },
    };
    // A measured keep lies between its sweep point and the next one
    // (`keep_hi`): its interval, the sweep's resolution.
    let keep_of = |v: u64| match v {
        0 => None,
        v => Some(Fact {
            hi: g.place.keep_hi.max(v),
            ..Fact::measured(v, g.place.cite)
        }),
    };
    let reach = |from, bw: Fact<f64>, total: Fact<f64>, lat: Fact<f64>, granule| Reach {
        from,
        bytes_per_ns: bw,
        total_bytes_per_ns: total,
        latency_ns: lat,
        granule,
    };
    let mut places = vec![
        Place {
            name: "regs",
            exec: E_LANE,
            capacity: geo(g, "regs_per_thread", g.regs_per_thread * 4),
            mgmt: Mgmt::Explicit,
            banking: None,
            granule: 4,
            reserve: 0,
            per_slot: 0,
            backing: None,
            reach: vec![],
        },
        Place {
            name: "smem",
            exec: E_UNIT,
            capacity: geo(g, "smem_per_sm", g.smem_per_sm),
            mgmt: Mgmt::Explicit,
            banking: Some(Banking {
                banks: 32,
                width: 4,
                broadcast: true,
            }),
            granule: 128,
            reserve: g.smem_reserved_per_block,
            per_slot: g.smem_per_block,
            backing: None,
            reach: vec![],
        },
        Place {
            name: "l2",
            exec: E_DEVICE,
            capacity: geo(g, "l2_bytes", g.l2_bytes),
            mgmt: Mgmt::Cache(Box::new(Cache {
                instances: Fact::nominal(1, 1, 2, "L2 partitions unmeasured"),
                map: InstanceMap::One,
                replicate: false,
                line: 128,
                keep: Keep {
                    cyclic: keep_cyclic,
                    rotated: keep_of(g.place.keep_rotated),
                    strided: keep_of(g.place.keep_strided),
                    curve: g
                        .place
                        .hit_curve
                        .iter()
                        .map(|&(x, y)| (x as f64 / 1000.0, y as f64 / 1000.0))
                        .collect(),
                },
                policies: match g.caps.l2_evict {
                    true => vec![Policy::Default, Policy::EvictFirst],
                    false => vec![Policy::Default],
                },
            })),
            banking: None,
            granule: 32,
            reserve: 0,
            per_slot: 0,
            backing: Some(DRAM),
            reach: vec![
                reach(E_DEVICE, l2_total, l2_total, lat_l2, 32),
                reach(E_UNIT, l2_unit, l2_total, lat_l2, 32),
            ],
        },
        Place {
            name: "dram",
            exec: E_DEVICE,
            capacity: Fact::nominal(1 << 34, 1 << 33, 1 << 37, "device memory size unread"),
            mgmt: Mgmt::Explicit,
            banking: None,
            granule: 32,
            reserve: 0,
            per_slot: 0,
            backing: None,
            reach: vec![reach(E_DEVICE, dram, dram, lat_dram, 32)],
        },
    ];
    let has_tmem = g.caps.matrix.iter().any(|o| o.acc == MHome::Tmem);
    places.push(Place {
        name: "tmem",
        exec: E_UNIT,
        capacity: match has_tmem {
            true => Fact::declared(256 << 10, "tensor memory: 128 lanes x 512 columns x 4 B"),
            false => Fact::declared(0, "no tensor memory"),
        },
        mgmt: Mgmt::Explicit,
        banking: None,
        granule: 4,
        reserve: 0,
        per_slot: 0,
        backing: None,
        reach: vec![],
    });
    places.push(Place {
        name: "host",
        exec: E_DEVICE,
        capacity: Fact::nominal(1 << 36, 1 << 33, 1 << 40, "host memory size unread"),
        mgmt: Mgmt::Explicit,
        banking: None,
        granule: 64,
        reserve: 0,
        per_slot: 0,
        backing: None,
        reach: vec![],
    });
    let res = vec![
        Resource {
            what: Res::Regs,
            cap: geo(g, "regs_per_sm", g.regs_per_sm),
            per_slot: 0,
            per_lane: g.regs_per_thread,
            granule: 256,
            reserve: 0,
        },
        Resource {
            what: Res::Threads,
            cap: geo(g, "max_threads_per_sm", g.max_threads_per_sm),
            per_slot: g.max_threads_per_block,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Slots,
            cap: geo(g, "max_blocks_per_sm", g.max_blocks_per_sm),
            per_slot: 0,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Bytes(SMEM),
            cap: geo(g, "smem_per_sm", g.smem_per_sm),
            per_slot: g.smem_per_block,
            per_lane: 0,
            granule: 128,
            reserve: g.smem_reserved_per_block,
        },
    ];
    // Issue pipes: warp-instructions an SM issues a clock by class.
    // FP32 lanes: a derived value (the compute-capability table's) spans
    // the SM designs' two widths, half to twice it.
    let lanes = fact(g, "fp32_lanes", g.fp32_lanes as f64, (0.5, 2.0));
    let lsu = match d.lsu_ipc_pm {
        0 => Fact::nominal(0.5, 0.25, 1.0, "LSU issue rate unmeasured"),
        v => fact(g, "lsu_ipc_pm", v as f64 / 1000.0, (0.9, 1.1)),
    };
    let pipes = vec![
        Pipe {
            class: Class::Math,
            per_clk: lanes.scale(1.0 / g.warp.max(1) as f64),
        },
        Pipe {
            class: Class::Shared,
            per_clk: lsu,
        },
        Pipe {
            class: Class::Global,
            per_clk: lsu,
        },
    ];
    // Operations: the FMA lane, the separate multiply-add, then every matrix
    // instruction the SM lists.
    let lane_op = |name, rate: Fact<f64>| Op {
        name,
        exec: E_LANE,
        lanes: 1,
        m: 1,
        n: NSet::One(1),
        k: 1,
        fa: Fmt::F32,
        fb: Fmt::F32,
        facc: Fmt::F32,
        aux: None,
        homes: [&[REGS], &[REGS], &[REGS]],
        majors: [&[Major::K, Major::Mn], &[Major::K, Major::Mn]],
        frag: [1, 1, 1],
        latency: latency(g, name, NSet::One(1), (4.0, 4.0, 6.0)),
        async_depth: 0,
        rate,
        eff: eff_points(g, name, NSet::One(1)),
        widths: &[1],
        acc_u: Fact::declared(-24, "IEEE f32 rounding"),
        acc_g: Fact::declared(u32::MAX, "exact alignment"),
    };
    let mut ops = vec![
        lane_op("ffma", lanes.scale(2.0)),
        lane_op("fmul+fadd", lanes),
    ];
    for o in g.caps.matrix {
        let name = op_name(o);
        let per_clk = g.matrix_flops_per_clk(o) as f64;
        let rate = match per_clk {
            0.0 => Fact::nominal(0.0, 0.0, 0.0, "no rate given for this instruction"),
            v => fact(g, rate_field(o), v, (0.7, 1.1)),
        };
        let a_regs = match o.a {
            MHome::Regs => o.shape.m * o.shape.k * o.fmt.bytes() / 4 / o.shape.threads.max(1),
            _ => 0,
        };
        let b_regs = match o.b {
            MHome::Regs => 8 * o.shape.k * o.fmt.bytes() / 4 / o.shape.threads.max(1),
            _ => 0,
        };
        let acc_regs = match o.acc {
            MHome::Regs => o.shape.m * 8 / o.shape.threads.max(1),
            _ => 0,
        };
        // The accumulator's numerics: measured where the part ran
        // `mma_calib.cu`; else the instruction row's (the RTX 5080's
        // measured datapath, another part's — marked, with its interval).
        let numerics = |v: i32| match src(g, "acc_u") {
            Src::Measured(c) => Fact::measured(v, c),
            _ => Fact::derived(
                v,
                v,
                v + 1,
                "unmeasured on this part: the RTX 5080's accumulator rounding",
            ),
        };
        let n = if o.shape.threads > 32 || o.shape.threads == 1 {
            NSet::Step {
                step: WGMMA_N_STEP,
                max: o.shape.n.min(WGMMA_N_MAX),
            }
        } else {
            NSet::One(o.shape.n)
        };
        ops.push(Op {
            name,
            exec: if o.shape.threads > 1 { E_GROUP } else { E_LANE },
            lanes: o.shape.threads,
            m: o.shape.m,
            n,
            k: o.shape.k,
            fa: fmt_of(o.fmt),
            fb: fmt_of(o.fmt),
            facc: Fmt::F32,
            aux: None,
            homes: [home_places(o.a), home_places(o.b), home_places(o.acc)],
            majors: match o.mn_major {
                true => [&[Major::K, Major::Mn], &[Major::K, Major::Mn]],
                false => [&[Major::K], &[Major::K]],
            },
            frag: [a_regs, b_regs, acc_regs],
            latency: latency(g, name, n, (32.0, 16.0, 64.0)),
            async_depth: u64::from(o.shape.threads > 32),
            rate,
            eff: eff_points(g, name, n),
            widths: Box::leak(vec![o.shape.threads].into_boxed_slice()),
            acc_u: numerics(o.acc_u),
            acc_g: match src(g, "acc_g") {
                Src::Measured(c) => Fact::measured(o.acc_g, c),
                _ => Fact::derived(
                    o.acc_g,
                    1,
                    o.acc_g,
                    "unmeasured on this part: the RTX 5080's guard bits",
                ),
            },
        });
    }
    let mut movers = vec![Mover {
        name: "ld.global",
        from: L2,
        to: REGS,
        issuer: E_LANE,
        kind: MoverKind::Lane {
            bytes: GPU_VEC_BYTES,
            asynchronous: false,
        },
        converts: true,
        policies: vec![Policy::Default],
    }];
    if g.caps.cp_async {
        movers.push(Mover {
            name: "cp.async",
            from: L2,
            to: SMEM,
            issuer: E_LANE,
            kind: MoverKind::Lane {
                bytes: GPU_VEC_BYTES,
                asynchronous: true,
            },
            converts: false,
            policies: vec![Policy::Default],
        });
    }
    if let Some(b) = g.caps.bulk {
        movers.push(Mover {
            name: "tma",
            from: L2,
            to: SMEM,
            issuer: E_LANE,
            kind: MoverKind::Bulk {
                granule: b.granule,
                box_dims: b.box_dims,
                box_side: b.box_side,
                swizzles: b.swizzles.to_vec(),
                tx_bytes: b.tx_bytes,
                cluster: b.cluster,
                multicast: b.multicast,
                copy_ns: match d.tma_copy_ns {
                    0 => Fact::derived(
                        g.tma_copy_ns() as f64,
                        g.tma_copy_ns() as f64 * 0.5,
                        g.tma_copy_ns() as f64 * 2.0,
                        "the loaded latency over the copies an engine overlaps",
                    ),
                    v => fact(g, "tma_copy_ns", v as f64, (0.9, 1.1)),
                },
                in_flight: Fact::nominal(6, 2, 16, "copies an engine overlaps unmeasured"),
            },
            converts: false,
            policies: vec![Policy::Default],
        });
    }
    movers.push(Mover {
        name: "link",
        from: HOST,
        to: DRAM,
        issuer: E_DEVICE,
        kind: MoverKind::Link {
            bytes_per_ns: Fact::nominal(25.0, 10.0, 60.0, "host link rate unmeasured"),
            pageable_pm: match d.pageable_pm {
                0 => Fact::nominal(700, 350, 1000, "pageable share unmeasured"),
                v => Fact::point(v, src(g, "pageable_pm")),
            },
            engines: 1,
        },
        converts: false,
        policies: vec![],
    });
    // Dispatch: launch costs measured where given; else a placeholder of a
    // launch's order of magnitude (no part's number), nominal and wide.
    let chain = |measured: u64, name: &str, nominal: f64| match measured {
        0 => Fact::nominal(
            nominal,
            nominal / 3.0,
            nominal * 3.0,
            "launch cost unmeasured: a placeholder, no part's value",
        ),
        v => fact(g, name, v as f64, (0.8, 1.3)),
    };
    let plain = chain(d.dep_launch_ns, "dep_launch_ns", 1000.0);
    let launches = vec![
        LaunchCost {
            mode: LaunchMode::Plain,
            ns: plain,
            per_slot_ps: match d.dep_launch_block_ps {
                0 => Fact::nominal(0.0, 0.0, 2.0, "per-slot dispatch unmeasured"),
                v => fact(g, "dep_launch_block_ps", v as f64, (0.8, 1.2)),
            },
        },
        LaunchCost {
            mode: LaunchMode::Dependent,
            ns: match d.pdl_launch_ns {
                0 => plain,
                v => fact(g, "pdl_launch_ns", v as f64, (0.8, 1.3)),
            },
            per_slot_ps: Fact::nominal(0.0, 0.0, 2.0, "per-slot dispatch unmeasured"),
        },
        LaunchCost {
            mode: LaunchMode::Cooperative,
            ns: chain(d.coop_launch_ns, "coop_launch_ns", 3000.0),
            per_slot_ps: Fact::nominal(0.0, 0.0, 2.0, "per-slot dispatch unmeasured"),
        },
    ];
    let start = match g.place.start {
        Some((s, cite)) => (s, Src::Measured(cite)),
        None => (Start::Lockstep, Src::Nominal("start timing unmeasured")),
    };
    let dispatch = Dispatch {
        unit: E_UNIT,
        start: start.0,
        start_src: start.1,
        round_robin: true,
        launches,
        issue_ns: match d.issue_ns {
            0 => Fact::nominal(3000.0, 1500.0, 6000.0, "host issue unmeasured"),
            v => fact(g, "issue_ns", v as f64, (0.8, 1.2)),
        },
        gap_ns: match d.gap_ns {
            0 => Fact::nominal(500.0, 200.0, 1000.0, "dependent gap unmeasured"),
            v => fact(g, "gap_ns", v as f64, (0.8, 1.2)),
        },
        atomics: vec![Atomics {
            at: L2,
            fmt: Fmt::F32,
            ops_per_ns: match d.atomic_ops_per_ns {
                0 => Fact::nominal(
                    l2_total.v / 11.0,
                    l2_total.v / 20.0,
                    l2_total.v / 8.0,
                    "f32 reductions unmeasured: a share of the L2 rate",
                ),
                v => fact(g, "atomic_ops_per_ns", v as f64, (0.8, 1.2)),
            },
        }],
        zero_ns: match d.zero_ns {
            0 => Fact::nominal(1000.0, 500.0, 2000.0, "zero pass fixed time unmeasured"),
            v => fact(g, "zero_ns", v as f64, (0.8, 1.2)),
        },
        zero_bytes_per_ns: match d.fill_bytes_per_ns {
            0 => Fact::nominal(
                l2_total.v / 2.0,
                dram.v,
                l2_total.v,
                "zero pass rate unmeasured: between the DRAM and L2 rates",
            ),
            v => fact(g, "fill_bytes_per_ns", v as f64, (0.8, 1.2)),
        },
        clock_khz: match g.place.clock_khz {
            0 => fact(g, "clock_khz", g.clock_khz as f64, (0.75, 1.05)),
            v => Fact::measured(v as f64, g.place.cite),
        },
        barrier_clk: fact(g, "bar_cycles", g.bar_cycles as f64, (0.9, 1.3)),
        grid_sync_ns: match d.gsync_ns {
            0 => None,
            v => Some(fact(g, "gsync_ns", v as f64, (0.8, 1.2))),
        },
        rr_jitter: None,
    };
    // The cost model's primitives the part measured (`place_calib.cu
    // prim`): each a row of its rate, one use's latency and the concurrency
    // that saturates it; an operation's measured rate and latency are its
    // row's.
    let mut prims = Vec::new();
    for &(name, v) in g.place.prim {
        let Some((of, per_clk)) = prim_of(name, &ops) else {
            continue;
        };
        // A family's member row stands for the family only where the
        // family's own form is not measured.
        let cite = match of {
            PrimOf::Op(o) if ops[o].name != name => {
                if g.place.prim.iter().any(|(r, _)| *r == ops[o].name) {
                    continue;
                }
                row_cite(g, name, ops[o].name)
            }
            _ => g.place.cite,
        };
        let f = |i: usize| {
            Fact::measured(v[i] as f64 / 1000.0, cite)
                .within(v[i + 1] as f64 / 1000.0, v[i + 2] as f64 / 1000.0)
        };
        let (rate, latency, conc) = (f(0), f(3), f(6));
        if let PrimOf::Op(o) = of {
            ops[o].rate = rate;
            ops[o].latency = latency;
        }
        prims.push(Prim {
            name,
            of,
            per_clk,
            rate,
            latency,
            conc,
        });
    }
    Some(Machine {
        label: p.name.to_string(),
        exec,
        places,
        groups: vec![],
        res,
        pipes,
        ops,
        movers,
        dispatch,
        prims,
    })
}

/// **A probe's primitive** (`place_calib.cu prim`'s names) as what it does
/// and the place it reaches, and whether its row is in clocks: `ld.global.<w>.<l2|dram>`,
/// `cp.async.<w>.<src>` (and `.run<r>.rows<n>.fly<bytes>`: a panel stream),
/// `tma.box<bytes>.<src>`, `ld.shared.<w>`, `ldmatrix.x4`, `st.global.<w>.<dst>`,
/// `epilogue.smem.st.global.16`, `red.add.f32[.warp|.one|.frag|.sector]`,
/// `bar.sync.<lanes>`, `mbarrier.<lanes>`, `grid.sync`,
/// `launch.{issue,gap}.sms`, and an operation row's own name.
fn prim_of(name: &str, ops: &[Op]) -> Option<(PrimOf, bool)> {
    let f: Vec<&str> = name.split('.').collect();
    let place = |x: &str| match x {
        "l2" => Some(L2),
        "dram" => Some(DRAM),
        _ => None,
    };
    let n = |x: &str, pre: &str| x.strip_prefix(pre)?.parse::<u64>().ok();
    Some(match f.as_slice() {
        ["ld", "global", w, src] => (
            PrimOf::Load {
                from: place(src)?,
                bytes: w.parse().ok()?,
            },
            false,
        ),
        ["cp", "async", w, src] => (
            PrimOf::Copy {
                from: place(src)?,
                bytes: w.parse().ok()?,
            },
            false,
        ),
        ["cp", "async", _, src, run, rows, fly] => (
            PrimOf::Panel {
                from: place(src)?,
                run: n(run, "run")?,
                rows: n(rows, "rows")?,
                fly: n(fly, "fly")?,
            },
            false,
        ),
        ["tma", bx, src] => (
            PrimOf::Bulk {
                from: place(src)?,
                bytes: n(bx, "box")?,
            },
            false,
        ),
        ["ld", "shared", w] => (
            PrimOf::Scratch {
                bytes: w.parse().ok()?,
                matrix: false,
            },
            true,
        ),
        ["ldmatrix", "x4"] => (
            PrimOf::Scratch {
                bytes: 16,
                matrix: true,
            },
            true,
        ),
        ["st", "global", w, dst] => (
            PrimOf::Store {
                to: place(dst)?,
                bytes: w.parse().ok()?,
            },
            false,
        ),
        ["epilogue", ..] => (PrimOf::Staged { to: DRAM }, false),
        ["red", "add", "f32", rest @ ..] => {
            let (sharing, sectors) = match rest {
                [] => (1, 4),
                ["warp"] => (32, 1),
                ["one"] => (0, 1),
                ["frag"] => (1, 8),
                ["sector"] => (1, 32),
                _ => return None,
            };
            (
                PrimOf::Reduce {
                    at: L2,
                    sharing,
                    sectors,
                },
                false,
            )
        }
        ["bar", "sync", l] => (
            PrimOf::Barrier {
                lanes: l.parse().ok()?,
            },
            true,
        ),
        ["mbarrier", _] => (PrimOf::Phase, true),
        ["grid", "sync"] => (PrimOf::GridSync, false),
        ["launch", "issue", "sms"] => (PrimOf::Issue, false),
        ["launch", "gap", "sms"] => (PrimOf::Gap, false),
        ["launch", ..] => return None,
        _ => (
            PrimOf::Op(
                ops.iter()
                    .position(|o| o.name == name)
                    .or_else(|| ops.iter().position(|o| measures(name, o.name, o.n)))?,
            ),
            true,
        ),
    })
}

/// The fits a profile carries that the evaluator never reads (ADR-0050 D2):
/// listed in the evidence as replay lines until the evaluator reproduces
/// them.
pub const FITS: [&str; 8] = [
    "flops_per_ns",
    "bytes_per_ns",
    "launch_ns",
    "eff_max_pm",
    "eff_half",
    "split_cell_ps",
    "tail_bytes_per_ns",
    "row_iter_ns",
];

// ---- the site ---------------------------------------------------------------

/// The site as placement reads it: extents, operand bytes as stored, the
/// fold's freedom, whether `k` may be cut (no window or mixed-radix
/// operand), the formats its precision set admits.
pub fn site_of(site: &TileSite, contract: bool, order: crate::Order) -> Site {
    let e = elem_bytes_of(&site.elem);
    let mut fmts = vec![if e == 8 { Fmt::F64 } else { Fmt::F32 }];
    if site.tf32 {
        fmts.push(Fmt::Tf32);
    }
    match site.half {
        Some(mapal_ir::NarrowFloat::Bf16) => fmts.push(Fmt::Bf16),
        Some(mapal_ir::NarrowFloat::F16) => fmts.push(Fmt::F16),
        _ => {}
    }
    let window = site.a.ksplit.is_some() || site.b.ksplit.is_some();
    let _ = contract;
    let warm = warm_of(site);
    Site {
        m: site.rows,
        n: site.c,
        k: site.k,
        batch: 1,
        ea: site.a.elem_bytes(&site.elem),
        eb: site.b.elem_bytes(&site.elem),
        eo: e,
        epi: 0,
        fold: match order {
            crate::Order::Free => Fold::Free,
            crate::Order::Fixed => Fold::Fixed,
            crate::Order::Sequential => Fold::Sequential,
        },
        kcut: site.radix.is_none() && !window,
        fmts,
        warm,
    }
}

thread_local! {
    /// The emission's tile sites' warm operands ([`WarmScope`]), by the
    /// site's text; `None` outside an emission.
    static WARM: std::cell::RefCell<Option<std::collections::HashMap<String, [bool; 2]>>> =
        const { std::cell::RefCell::new(None) };
}

/// **A site's warm operands** (ADR-0050 D3): `[a, b]` resident in the
/// device cache when the site reads them — `residence_plan` at the part's
/// keep: the working set from the operand's producer (a program input:
/// the call's start, so the previous call's bytes) to its last reader fits
/// ([`warm_sites`]; the fact `MAPAL_SITE_COLD` reads, its complement).
/// Read from the emission's [`WarmScope`]; none outside one.
pub fn warm_of(site: &TileSite) -> [bool; 2] {
    WARM.with(|w| {
        w.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{site:?}")).copied())
            .unwrap_or([false, false])
    })
}

/// Every tile site of `ir` (the IR's tile plans: affine, strided, narrow)
/// and its warm operands at a cache keeping `keep` bytes: an operand is warm
/// where `residence_plan` places its bytes on chip or in the cache.
pub fn warm_sites(ir: &mapal_ir::CategoryIr, keep: u64) -> Vec<(TileSite, [bool; 2])> {
    let mut out = Vec::new();
    for (f, _) in ir.funcs() {
        let res = ir.residence_plan(f);
        for plan in [
            ir.tile_plan(f),
            ir.tile_plan_strided(f),
            ir.tile_plan_narrow(f),
        ] {
            for (m, site) in plan.sites.iter() {
                let source = ir.morphism(m).expect("a site's morphism").source;
                let warm = |slot: u32| {
                    crate::func::pair_source_ir(ir, source, slot)
                        .is_some_and(|o| res.at(o, keep) != mapal_ir::Residence::Dram)
                };
                out.push((site.clone(), [warm(site.a.slot), warm(site.b.slot)]));
            }
        }
    }
    out
}

/// **The warm operands of an emission's sites, while it runs**: every
/// launch of the emission (its plans, its kernels, its host) reads one map,
/// so a site is placed alike wherever its launch is derived. Sites of one
/// text in several places take the conjunction (warm only where every one
/// is). Only under `machine-place` (the map is the search's input alone).
pub struct WarmScope {
    prev: Option<std::collections::HashMap<String, [bool; 2]>>,
}

impl WarmScope {
    pub fn new(ir: &mapal_ir::CategoryIr, p: &TargetProfile, arms: crate::GpuArms) -> Self {
        let map = match (p.gpu(), arms.on.machine_place) {
            (Some(g), true) => {
                let mut m: std::collections::HashMap<String, [bool; 2]> = Default::default();
                for (site, w) in warm_sites(ir, g.l2_keep_bytes()) {
                    let e = m.entry(format!("{site:?}")).or_insert(w);
                    e[0] &= w[0];
                    e[1] &= w[1];
                }
                Some(m)
            }
            _ => None,
        };
        WarmScope {
            prev: WARM.with(|w| w.replace(map)),
        }
    }
}

thread_local! {
    /// The tile sites whose output the emission stores column-major
    /// ([`ColScope`]); `None` outside an emission under `machine-place`.
    static COLMAJOR: std::cell::RefCell<Option<std::collections::HashSet<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// **The sites an emission stores column-major, while it runs** (the row
/// folds' `colmajor`: a producing tile a fold reads in order, the rules'
/// base launch able to store so): a pick that cannot store column-major is
/// dropped for such a site alone — the layout is the site's, decided before
/// any placement, never the rules' launch's capability (R3: a 4096 x 8192 x
/// 8192 TF32 site stores row-major, yet its rules' FMA tile could store
/// column-major, so the filter on the capability dropped the tensor pick and
/// no kernel carried a place line). Only under `machine-place`.
pub struct ColScope {
    prev: Option<std::collections::HashSet<String>>,
}

impl ColScope {
    pub fn new(
        ir: &mapal_ir::CategoryIr,
        p: &TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
    ) -> Self {
        let set = match (p.gpu(), arms.on.machine_place) {
            (Some(_), true) => {
                let mut set = std::collections::HashSet::new();
                for (f, _) in ir.funcs() {
                    let tiles = ir.tile_plan(f);
                    for (_, fold) in crate::func::rowfold::rowfold_sites(ir, f, p, contract, arms) {
                        if let Some(t) = fold.colmajor.and_then(|m| tiles.sites.get(m)) {
                            set.insert(format!("{t:?}"));
                        }
                    }
                }
                Some(set)
            }
            _ => None,
        };
        ColScope {
            prev: COLMAJOR.with(|c| c.replace(set)),
        }
    }
}

impl Drop for ColScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        COLMAJOR.with(|c| *c.borrow_mut() = prev);
    }
}

/// Whether `site`'s output is stored column-major: the emission's
/// [`ColScope`], else (no scope: a launch derived outside an emission) the
/// rules' launch `rules` able to store so — the conservative reading.
fn stores_col(site: &TileSite, rules: &Launch) -> bool {
    COLMAJOR.with(|c| match &*c.borrow() {
        Some(set) => set.contains(&format!("{site:?}")),
        None => gpu::col_store(rules, site),
    })
}

thread_local! {
    /// The emission's order ([`OrderScope`]): the order its host launches
    /// place each site under.
    static ORDER: std::cell::Cell<Option<crate::Order>> = const { std::cell::Cell::new(None) };
}

/// The emission's order for [`placed_for`]: the kernel of a realization is
/// the pick of the order the host placed the site under, whichever of the
/// two is emitted first (cost-rental: the H100's fixed-order 16 x 2048 x
/// 7168 cell launched the pick's split grid over the rules' kernel —
/// wrong values — where the kernel text came before the host's search).
pub struct OrderScope {
    prev: Option<crate::Order>,
}

impl OrderScope {
    pub fn new(order: crate::Order) -> Self {
        OrderScope {
            prev: ORDER.with(|o| o.replace(Some(order))),
        }
    }
}

impl Drop for OrderScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ORDER.with(|o| o.set(prev));
    }
}

impl Drop for WarmScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        WARM.with(|w| *w.borrow_mut() = prev);
    }
}

// ---- schedules and launches ---------------------------------------------------

/// The one-thread-per-cell kernel's registers at up to eight cells a
/// thread where the full-occupancy budget allows them (the `ptxas -v`
/// reports' largest, sm_80 – sm_120).
pub const NAIVE_REGS: u64 = 40;

/// The lowering ids.
pub const NAIVE: &str = "naive";
pub const FMA: &str = "fma";
pub const MMA: &str = "mma";
pub const WGMMA: &str = "wgmma";
pub const WGMMA_WS: &str = "wgmma-ws";

/// The matrix operation row of format `fmt` issued by a warp or a warpgroup
/// (`group`), with `a` from `a` where the part lists that home, else the
/// row of that format (the schedule's level then names `a`'s home: the S53
/// TF32 kernel's `bᵀ` fragments from registers).
fn op_index(g: &Gpu, fmt: MmaFmt, a: MHome, group: bool) -> Option<usize> {
    let rows = || {
        g.caps
            .matrix
            .iter()
            .enumerate()
            .filter(move |(_, o)| o.fmt == fmt && (o.shape.threads > 32) == group)
    };
    rows()
        .find(|(_, o)| o.a == a)
        .or_else(|| rows().next())
        .map(|(i, _)| i + 2)
}

/// The split forms a `{f32}` product may take on a matrix instruction, in
/// one order (the schedule's `split` parameter: 0 none, else 1 + the index).
const SPLITS: [mapal_ir::split::SplitKind; 5] = {
    use mapal_ir::split::SplitKind as K;
    [K::Tf32x3, K::Bf16x6, K::Bf16x3n, K::Bf16x2n, K::Narrow1]
};

fn split_code(s: Option<mapal_ir::split::SplitKind>) -> u64 {
    match s {
        None => 0,
        Some(k) => 1 + SPLITS.iter().position(|x| *x == k).unwrap_or(0) as u64,
    }
}

fn split_of(c: u64) -> Option<mapal_ir::split::SplitKind> {
    match c {
        0 => None,
        c => Some(SPLITS[(c - 1) as usize]),
    }
}

/// The traversal a launch's combine and raster realize.
fn trav_of(l: &Launch) -> Traversal {
    let h = l.raster.max(1);
    if let Some(w) = l.wgmma.filter(|w| w.ctas > 0) {
        return Traversal::Persistent { h, ctas: w.ctas };
    }
    match l.combine {
        Combine::None => Traversal::Ordered { h },
        Combine::Atomic { s } => Traversal::Split {
            s,
            comb: Comb::Atomic,
            h,
        },
        Combine::Workspace { s } => Traversal::Split {
            s,
            comb: Comb::Workspace,
            h,
        },
        Combine::StreamK {
            ctas,
            iters,
            whole: 0,
        } => Traversal::Walk { h, ctas, iters },
        Combine::StreamK { ctas, iters, .. } => Traversal::Hybrid { h, ctas, iters },
    }
}

/// The combine traversal `t` lowers to; a hybrid's whole waves are the
/// `tiles` of the grid over `slots` (the footprint's).
fn combine_of(t: Traversal, tiles: u64, slots: u64) -> Combine {
    match t {
        Traversal::Ordered { .. } | Traversal::Persistent { .. } => Combine::None,
        Traversal::Hybrid { ctas, iters, .. } => Combine::StreamK {
            ctas,
            iters,
            whole: tiles / slots.max(1) * slots.max(1),
        },
        Traversal::Split {
            s,
            comb: Comb::Atomic,
            ..
        } => Combine::Atomic { s },
        Traversal::Split { s, .. } => Combine::Workspace { s },
        Traversal::Walk { ctas, iters, .. } => Combine::StreamK {
            ctas,
            iters,
            whole: 0,
        },
    }
}

fn smem_home(depth: u64, major: Major, pad: u64, swizzle: u64) -> Option<Home> {
    Some(Home {
        place: SMEM,
        layout: Layout {
            major,
            pad,
            swizzle,
        },
        depth,
    })
}

/// The face a launch's arithmetic is emitted on.
fn face_of(a: Arithmetic) -> Face {
    a.face()
}

/// The site's table rungs at the launch's face (the D6 steps the profile's
/// rules decide for the site's fill).
pub(crate) fn table_rungs(
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    a: Arithmetic,
) -> crate::GpuRungs {
    let gpu = profile.gpu().expect("a GPU profile");
    let (blocks, threads) = fill_grid(site, profile);
    let wps = gpu.warps_per_scheduler(blocks, threads);
    site_rungs(profile, wps, face_of(a), arms, site)
}

/// The rungs `r` carries beyond `base`, as schedule parameters.
fn rung_params(base: &crate::GpuRungs, r: &crate::GpuRungs) -> Vec<(&'static str, u64)> {
    let (mut b, mut x) = (*base, *r);
    let mut out = Vec::new();
    for (i, name) in crate::GpuRungs::NAMES.iter().enumerate() {
        let (vb, vx) = (
            *b.rung(name).expect("a rung"),
            *x.rung(name).expect("a rung"),
        );
        if vb != vx {
            out.push((if vx { "rung+" } else { "rung-" }, i as u64));
        }
    }
    out
}

/// **The rules' launch as a schedule**: the lowering, the operation and its
/// roles, the levels (a lane's or group's tile, the slot's tile and panel
/// with each operand's home, layout and depth), the slot's composition, the
/// traversal, the occupancy, and the lowering's encoding parameters.
pub fn schedule_of(
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    l: &Launch,
) -> Schedule {
    let gpu = profile.gpu().expect("a GPU profile");
    let base = table_rungs(site, profile, arms, l.arithmetic);
    let mut params = rung_params(&base, &l.rungs);
    let slot = |ext: [u64; 3], homes: [Option<Home>; 2]| Level {
        ext,
        bind: Bind::Slot,
        homes,
    };
    let grid = Level {
        ext: [site.rows, site.c, site.k],
        bind: Bind::Exec(E_DEVICE),
        homes: [None, None],
    };
    let base_sched = |lowering, op, swap, levels, groups: [u64; 2], per_unit| Schedule {
        lowering,
        op,
        swap,
        levels,
        groups,
        producers: 0,
        regs: [0, 0],
        cluster: [1, 1],
        multicast: [false, false],
        trav: trav_of(l),
        launch: LaunchMode::Plain,
        policy: [Policy::Default; 2],
        per_unit,
        params: Vec::new(),
    };
    let scalar_op = match l.arithmetic {
        Arithmetic::SeparateMulAdd => OP_MULADD,
        _ => OP_FFMA,
    };
    let mut x = if let Some(w) = l.wgmma {
        let op = op_index(gpu, w.fmt, w.a_home, true).unwrap_or(usize::MAX);
        params.extend([
            ("ea", w.ea),
            ("eb", w.eb),
            ("pad_b", w.pad_b),
            ("b_kmajor", u64::from(w.b_kmajor)),
            ("n", w.n),
        ]);
        let mut x = base_sched(
            if w.ws { WGMMA_WS } else { WGMMA },
            op,
            !w.direct,
            vec![
                Level {
                    ext: [64, w.n, w.mma_k()],
                    bind: Bind::Exec(E_GROUP),
                    homes: [
                        Some(Home {
                            place: match w.a_home {
                                MHome::Regs => REGS,
                                MHome::Smem => SMEM,
                                MHome::Tmem => TMEM,
                            },
                            layout: Layout::K,
                            depth: 1,
                        }),
                        None,
                    ],
                },
                slot(
                    [w.bm(), w.bn(), w.bk],
                    [
                        smem_home(w.stages, Major::K, 0, 128),
                        smem_home(
                            w.stages,
                            if w.b_kmajor { Major::K } else { Major::Mn },
                            w.pad_b,
                            if w.ws { 128 } else { 0 },
                        ),
                    ],
                ),
                grid,
            ],
            [w.groups, 1],
            w.blocks_per_sm,
        );
        x.producers = u64::from(w.ws);
        x.regs = w.realloc.map_or([0, 0], |(p, c)| [p, c]);
        x
    } else if let Some(m) = l.mma {
        let op = op_index(gpu, m.fmt, MHome::Regs, false).unwrap_or(usize::MAX);
        params.extend([
            ("ea", m.ea),
            ("eb", m.eb),
            ("split", split_code(m.split)),
            ("mt", m.mt),
            ("nt", m.nt),
        ]);
        base_sched(
            MMA,
            op,
            false,
            vec![
                Level {
                    ext: [m.wm(), m.wn(), m.mma_k()],
                    bind: Bind::Exec(E_GROUP),
                    homes: [None, None],
                },
                slot(
                    [m.bm(), m.bn(), m.bk],
                    [
                        smem_home(m.stages, Major::K, m.pad_a, 0),
                        smem_home(
                            m.stages,
                            if m.bt { Major::K } else { Major::Mn },
                            m.pad_b,
                            0,
                        ),
                    ],
                ),
                grid,
            ],
            [m.warps_y, m.warps_x],
            m.blocks_per_sm,
        )
    } else if let Some(t) = l.tiled {
        params.extend([("a_row_lanes", t.a_row_lanes), ("held", t.held)]);
        base_sched(
            FMA,
            scalar_op,
            false,
            vec![
                Level {
                    ext: [t.tm, t.tn, 1],
                    bind: Bind::Exec(E_LANE),
                    homes: [None, None],
                },
                Level {
                    ext: [t.ty, t.tx, 1],
                    bind: Bind::Exec(E_GROUP),
                    homes: [None, None],
                },
                slot(
                    [t.bm(), t.bn(), t.bk],
                    [
                        smem_home(
                            t.stages,
                            if t.kmajor { Major::K } else { Major::Mn },
                            t.pad_a,
                            0,
                        ),
                        smem_home(t.stages, Major::Mn, 0, 0),
                    ],
                ),
                grid,
            ],
            [t.threads().div_ceil(gpu.warp), 1],
            t.blocks_per_sm,
        )
    } else {
        params.push(("cells", l.cells));
        let per = (gpu.max_threads_per_sm / l.block.max(1)).min(gpu.max_blocks_per_sm);
        let bn = (NAIVE_THREADS * l.cells).min(site.c);
        let bm = ((NAIVE_THREADS * l.cells) / site.c.max(1)).clamp(1, site.rows.max(1));
        base_sched(
            NAIVE,
            scalar_op,
            false,
            vec![
                Level {
                    ext: [1, l.cells, 1],
                    bind: Bind::Exec(E_LANE),
                    homes: [None, None],
                },
                slot([bm, bn, NAIVE_UNROLL], [None, None]),
                grid,
            ],
            [NAIVE_THREADS / gpu.warp.max(1), 1],
            per,
        )
    };
    if l.arithmetic == Arithmetic::SplitMma {
        params.push(("split_arith", 1));
    }
    x.params = params;
    x
}

fn param(x: &Schedule, name: &str) -> u64 {
    x.param(name).unwrap_or(0)
}

fn major(h: &Option<Home>) -> Major {
    h.map_or(Major::K, |h| h.layout.major)
}

/// **The launch a schedule lowers to**: the kernel geometry its lowering
/// emits, the grid, block and shared memory its footprint declares, the
/// raster and combine of its traversal, and the site's table rungs with the
/// schedule's own D6 steps applied.
pub fn launch_of(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    x: &Schedule,
) -> Launch {
    let gpu = profile.gpu().expect("a GPU profile");
    let e = elem_bytes_of(&site.elem);
    let s = x.slot();
    let [ha, hb] = s.homes;
    let depth = x.depth();
    let mut tiled = None;
    let mut mma = None;
    let mut wgmma = None;
    let (gx, gy, block, smem, cells);
    let arithmetic = match x.lowering {
        NAIVE | FMA => match x.op {
            OP_MULADD => Arithmetic::SeparateMulAdd,
            _ => Arithmetic::OrderedFma,
        },
        _ => {
            let o = &gpu.caps.matrix[x.op - 2];
            match (o.shape.threads > 32, o.fmt) {
                (true, MmaFmt::Tf32) => Arithmetic::Tf32Wgmma,
                (true, _) => Arithmetic::HalfWgmma,
                (false, _) if param(x, "split_arith") == 1 => Arithmetic::SplitMma,
                (false, MmaFmt::Tf32) => Arithmetic::Tf32Mma,
                (false, _) => Arithmetic::HalfMma,
            }
        }
    };
    let _ = contract;
    let mut rungs = memo_rungs(site, profile, arms, arithmetic);
    for (n, v) in &x.params {
        let on = match *n {
            "rung+" => true,
            "rung-" => false,
            _ => continue,
        };
        *rungs
            .rung(crate::GpuRungs::NAMES[*v as usize])
            .expect("a rung") = on;
    }
    match x.lowering {
        NAIVE => {
            cells = param(x, "cells").max(1);
            gx = cell_blocks(site, cells);
            gy = 1;
            block = NAIVE_THREADS;
            smem = 0;
        }
        FMA => {
            let [tm, tn, _] = x.levels[0].ext;
            let [ty, tx, _] = x.levels[1].ext;
            let t = GpuTile {
                tm,
                tn,
                tx,
                ty,
                bk: x.bk(),
                blocks_per_sm: x.per_unit,
                stages: depth,
                pad_a: ha.map_or(0, |h| h.layout.pad),
                a_row_lanes: param(x, "a_row_lanes"),
                held: param(x, "held"),
                kmajor: major(&ha) == Major::K,
            };
            cells = 1;
            gx = site.c / t.bn();
            gy = site.rows / t.bm();
            block = t.threads();
            smem = t.smem_bytes(e);
            tiled = Some(t);
        }
        MMA => {
            let o = &gpu.caps.matrix[x.op - 2];
            let m = GpuMmaTile {
                mt: param(x, "mt"),
                nt: param(x, "nt"),
                warps_x: x.groups[1],
                warps_y: x.groups[0],
                bk: x.bk(),
                stages: depth,
                pad_a: ha.map_or(0, |h| h.layout.pad),
                pad_b: hb.map_or(0, |h| h.layout.pad),
                blocks_per_sm: x.per_unit,
                fmt: o.fmt,
                ea: param(x, "ea"),
                eb: param(x, "eb"),
                bt: major(&hb) == Major::K,
                split: split_of(param(x, "split")),
            };
            cells = 1;
            gx = site.c / m.bn();
            gy = site.rows / m.bm();
            block = m.threads(gpu.warp);
            smem = m.smem_bytes()
                + match gpu::bulk_stage(site, &m, gpu, &rungs) {
                    true => gpu::BULK_BAR_BYTES * m.stages,
                    false => 0,
                };
            mma = Some(m);
        }
        _ => {
            let o = &gpu.caps.matrix[x.op - 2];
            let ctas = match x.trav {
                Traversal::Persistent { ctas, .. } => ctas,
                _ => 0,
            };
            let w = GpuWgmmaTile {
                n: param(x, "n"),
                groups: x.groups[0],
                bk: x.bk(),
                stages: depth,
                pad_b: param(x, "pad_b"),
                blocks_per_sm: x.per_unit,
                ws: x.lowering == WGMMA_WS,
                realloc: (x.regs != [0, 0]).then_some((x.regs[0], x.regs[1])),
                ctas,
                fmt: o.fmt,
                direct: !x.swap,
                ea: param(x, "ea"),
                eb: param(x, "eb"),
                a_home: match x.levels[0].homes[0].map(|h| h.place) {
                    Some(REGS) => MHome::Regs,
                    Some(TMEM) => MHome::Tmem,
                    Some(SMEM) => MHome::Smem,
                    _ => o.a,
                },
                b_kmajor: param(x, "b_kmajor") == 1,
            };
            cells = 1;
            (gx, gy) = match ctas {
                0 => (site.c / w.bn(), site.rows / w.bm()),
                c => (c, 1),
            };
            block = w.threads(gpu.warp);
            smem = w.smem_bytes();
            wgmma = Some(w);
        }
    }
    Launch {
        arithmetic,
        tiled,
        mma,
        wgmma,
        gx,
        gy,
        block,
        smem,
        raster: x.trav.raster().max(1),
        rungs,
        combine: combine_of(
            x.trav,
            (site.rows / x.bm().max(1)) * (site.c / x.bn().max(1)),
            gpu.sms * x.per_unit.max(1),
        ),
        cells,
    }
}

// ---- the lowerings' declarations ----------------------------------------------

/// A present emitter as a [`Lowering`], over one site of one profile.
pub struct Emitter<'a> {
    pub id: &'static str,
    pub site: &'a TileSite,
    pub profile: &'a TargetProfile,
    pub contract: bool,
    pub arms: crate::GpuArms,
}

impl Emitter<'_> {
    fn gpu(&self) -> &Gpu {
        self.profile.gpu().expect("a GPU profile")
    }

    /// The emitters of a site (the search's lowerings, ADR-0050 C4).
    #[allow(dead_code)]
    pub fn all<'a>(
        site: &'a TileSite,
        profile: &'a TargetProfile,
        contract: bool,
        arms: crate::GpuArms,
    ) -> Vec<Emitter<'a>> {
        [NAIVE, FMA, MMA, WGMMA, WGMMA_WS]
            .into_iter()
            .map(|id| Emitter {
                id,
                site,
                profile,
                contract,
                arms,
            })
            .collect()
    }

    /// **The coordinates of the lowering's slot space** (D5 steps 2–3), in
    /// order: each coordinate's domain given the ones before it — from the
    /// machine's facts (register file, shared memory, threads, the
    /// operation's shape and homes, the site's extents), never a table. The
    /// enumeration ([`Lowering::slots`]) and the membership check
    /// ([`Self::contains`]) walk the same domains.
    pub fn coords(&self) -> &'static [&'static str] {
        match self.id {
            NAIVE => &["op", "cells"],
            FMA => &[
                "op",
                "tm",
                "tn",
                "tx",
                "ty",
                "bk",
                "stages",
                "kmajor",
                "pad_a",
                "a_row_lanes",
                "held",
            ],
            MMA => &[
                "op", "mt", "nt", "warps_y", "warps_x", "bk", "stages", "bt", "pad_a", "pad_b",
                "split", "ea", "eb",
            ],
            _ => &[
                "op", "n", "groups", "bk", "stages", "b_kmajor", "pad_b", "swap", "ea", "eb",
                "regs_p", "regs_c",
            ],
        }
    }

    /// The domain of coordinate `i` given `pre` (the coordinates before it).
    pub fn domain(&self, i: usize, pre: &[u64]) -> Vec<u64> {
        let g = self.gpu();
        let s = self.site;
        let e = elem_bytes_of(&s.elem);
        let pow2 = |hi: u64| -> Vec<u64> {
            (0..20)
                .map(|p| 1u64 << p)
                .take_while(|&v| v <= hi)
                .collect()
        };
        let ops = |group: Option<bool>| -> Vec<u64> {
            let mut v = Vec::new();
            match group {
                None => v.extend([OP_FFMA as u64, OP_MULADD as u64]),
                Some(grp) => {
                    for (j, o) in g.caps.matrix.iter().enumerate() {
                        let fits = (o.shape.threads > 32) == grp && o.shape.threads > 1;
                        let admitted = match o.fmt {
                            MmaFmt::Tf32 => s.tf32 || !grp,
                            f => s.half.is_some_and(|h| MmaFmt::of_narrow(h) == f) || !grp,
                        };
                        if fits && admitted {
                            v.push(j as u64 + 2);
                        }
                    }
                }
            }
            v
        };
        let stages =
            |bk: u64, hi: u64| -> Vec<u64> { (1..=hi.min(s.k / bk.max(1)).max(1)).collect() };
        match (self.id, self.coords()[i]) {
            (NAIVE, "op") | (FMA, "op") => ops(None),
            (MMA, "op") => ops(Some(false)),
            (_, "op") => ops(Some(true)),
            (NAIVE, "cells") => pow2(GPU_VEC_BYTES / e.min(GPU_VEC_BYTES).max(1) * 4),
            // A lane's accumulators: whose held registers fit the lane's
            // file; a row group's rows (a radix head group) too.
            (FMA, "tm") | (FMA, "tn") => {
                let mut v: Vec<u64> = (1..=32)
                    .filter(|&t| fma_held_regs(t, 1, e) <= g.regs_per_thread)
                    .collect();
                v.retain(|t| t.is_power_of_two() || gpu::row_group(s).is_multiple_of(*t));
                v
            }
            (FMA, "tx") | (FMA, "ty") => pow2(g.max_threads_per_block),
            (FMA, "bk") => (1..=s.k).filter(|b| s.k.is_multiple_of(*b)).collect(),
            // A ring of up to 8 panels (more stages than panels: a slot's
            // double buffer); the shared memory it takes is declared.
            (FMA, "stages") => (1..=8).collect(),
            (_, "kmajor") | (_, "bt") | (_, "b_kmajor") | (_, "swap") => vec![0, 1],
            (FMA, "pad_a") | (MMA, "pad_a") | (MMA, "pad_b") | (_, "pad_b") => (0..=32).collect(),
            (FMA, "a_row_lanes") => {
                let mut v = vec![0];
                v.extend(pow2(pre[5].max(1) * 32));
                v
            }
            // The registers a whole-file tile holds: at least its counted
            // footprint, at most the lane's file (0: the half-file rule).
            (FMA, "held") => {
                let mut v = vec![0];
                v.extend(fma_held_regs(pre[1], pre[2], e)..=g.regs_per_thread);
                v
            }
            // The warp tile in instruction shapes, warps a slot.
            (MMA, "mt") | (MMA, "nt") => pow2(16),
            (MMA, "warps_y") | (MMA, "warps_x") => pow2(g.max_threads_per_block / g.warp),
            (MMA, "bk") | (_, "bk") => {
                let o = g.caps.matrix.get((pre[0] as usize).wrapping_sub(2));
                let mk = o.map_or(8, |o| o.shape.k);
                (1..=s.k / mk.max(1))
                    .map(|t| t * mk)
                    .filter(|b| s.k.is_multiple_of(*b))
                    .collect()
            }
            (MMA, "stages") => stages(pre[5], 16),
            (MMA, "split") => (0..=SPLITS.len() as u64).collect(),
            (_, "ea") | (_, "eb") => vec![1, 2, 4],
            (_, "n") => (1..=WGMMA_N_MAX / WGMMA_N_STEP)
                .map(|t| t * WGMMA_N_STEP)
                .collect(),
            (_, "groups") => pow2(g.max_threads_per_block / (WARPGROUP_WARPS * g.warp)),
            (_, "stages") => stages(pre[3], 16),
            (_, "regs_p") | (_, "regs_c") => {
                let mut v = vec![0];
                if let Some(r) = g.caps.reg_realloc {
                    v.extend((r.min..=r.max).step_by(r.step as usize));
                }
                v
            }
            _ => vec![],
        }
    }

    /// A schedule's coordinates in [`Self::coords`] order.
    pub fn coords_of(&self, x: &Schedule) -> Vec<u64> {
        let s = x.slot();
        let [ha, hb] = s.homes;
        let pad = |h: Option<Home>| h.map_or(0, |h| h.layout.pad);
        let maj = |h: Option<Home>| u64::from(h.is_some_and(|h| h.layout.major == Major::K));
        match self.id {
            NAIVE => vec![x.op as u64, param(x, "cells")],
            FMA => vec![
                x.op as u64,
                x.levels[0].ext[0],
                x.levels[0].ext[1],
                x.levels[1].ext[1],
                x.levels[1].ext[0],
                x.bk(),
                x.depth(),
                maj(ha),
                pad(ha),
                param(x, "a_row_lanes"),
                param(x, "held"),
            ],
            MMA => vec![
                x.op as u64,
                param(x, "mt"),
                param(x, "nt"),
                x.groups[0],
                x.groups[1],
                x.bk(),
                x.depth(),
                maj(hb),
                pad(ha),
                pad(hb),
                param(x, "split"),
                param(x, "ea"),
                param(x, "eb"),
            ],
            _ => vec![
                x.op as u64,
                param(x, "n"),
                x.groups[0],
                x.bk(),
                x.depth(),
                param(x, "b_kmajor"),
                param(x, "pad_b"),
                u64::from(x.swap),
                param(x, "ea"),
                param(x, "eb"),
                x.regs[0],
                x.regs[1],
            ],
        }
    }

    /// **Membership**: `x` is a point of this lowering's slot space — every
    /// coordinate inside its domain, the declared constraints met, the
    /// occupancy at most the footprint's. `Err`: the first coordinate or
    /// constraint outside.
    pub fn contains(&self, m: &Machine, x: &Schedule) -> Result<Declared, String> {
        if x.lowering != self.id {
            return Err(format!("lowering {} not {}", x.lowering, self.id));
        }
        let c = self.coords_of(x);
        for i in 0..c.len() {
            let dom = self.domain(i, &c[..i]);
            if !dom.contains(&c[i]) {
                return Err(format!(
                    "{}={} outside its domain ({} values{})",
                    self.coords()[i],
                    c[i],
                    dom.len(),
                    if dom.len() <= 8 {
                        format!(": {dom:?}")
                    } else {
                        String::new()
                    }
                ));
            }
        }
        let s = site_of(self.site, self.contract, crate::Order::Free);
        let d = self.declare(m, &s, x).map_err(|r| r.0)?;
        let bytes = [(SMEM, d.smem)];
        let occ = m.occupancy(d.threads, d.regs, &bytes);
        // The occupancy is the lowering's only where it emits it
        // (`.minnctapersm`, `min-blocks`); elsewhere `per_unit` describes
        // what the rules assumed, and the evaluator takes the footprint's.
        let back = launch_of(self.site, self.profile, self.contract, self.arms, x);
        // Only a tile kernel emits it (the cell kernel never does).
        let emitted = back.rungs.min_blocks
            && (back.tiled.is_some() || back.mma.is_some() || back.wgmma.is_some());
        if emitted && x.per_unit > occ.max(1) && x.per_unit > 1 {
            return Err(format!(
                "per_unit {} above the footprint's occupancy {occ} ({} lanes x {} regs, {} B)",
                x.per_unit, d.threads, d.regs, d.smem
            ));
        }
        Ok(d)
    }
}

impl Lowering for Emitter<'_> {
    fn id(&self) -> &'static str {
        self.id
    }

    fn traversals(&self) -> &'static [TravKind] {
        // The emitters' kernels: `wgmma` no combine at all; the one-thread
        // -per-cell kernel and the warp tile split; the FMA tile and the
        // 16-bit warp tile walk too (`declare` keeps the walk to them). An
        // arm that turns a combine's rung off removes its kind (the
        // epilogue groups' realizations, `group_order`).
        const ALL: [TravKind; 5] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::SplitWorkspace,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const NO_WS: [TravKind; 4] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const NO_ATOMIC: [TravKind; 4] = [
            TravKind::Ordered,
            TravKind::SplitWorkspace,
            TravKind::Walk,
            TravKind::Hybrid,
        ];
        const WALKS: [TravKind; 3] = [TravKind::Ordered, TravKind::Walk, TravKind::Hybrid];
        const SPLITS: [TravKind; 3] = [
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::SplitWorkspace,
        ];
        const ORDERED: [TravKind; 1] = [TravKind::Ordered];
        const WS: [TravKind; 2] = [TravKind::Ordered, TravKind::SplitWorkspace];
        const AT: [TravKind; 2] = [TravKind::Ordered, TravKind::SplitAtomic];
        if matches!(self.id, WGMMA | WGMMA_WS) {
            return &ORDERED;
        }
        let off = self.arms.off;
        match (off.splitk, off.splitk_atomic, off.streamk) {
            (false, false, false) => &ALL,
            (true, false, false) => &NO_WS,
            (false, true, false) => &NO_ATOMIC,
            (true, true, false) => &WALKS,
            (false, false, true) => &SPLITS,
            (true, false, true) => &AT,
            (false, true, true) => &WS,
            (true, true, true) => &ORDERED,
        }
    }

    /// The operations the search places the site on (D5 step 1): the
    /// face's scalar step for the scalar kernels; for the tensor tiles each
    /// matrix row of the lowering's issuing width whose format the site's
    /// precision set admits ([`mapal_place::space::ops`]'s rule over the
    /// part's rows: TF32 where the set holds it, a 16-bit format its
    /// operands stage in, the accumulator meeting the set), on a separable
    /// site with no radix, where the part copies (`cp.async`, `ldmatrix`
    /// for 16 bits) and the family's rung is on. Both roles for `wgmma`.
    fn ops(&self, _m: &Machine, _s: &Site) -> Vec<(usize, bool)> {
        let g = self.gpu();
        let site = self.site;
        let rungs = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
        match self.id {
            NAIVE | FMA => {
                let op = if self.contract { OP_FFMA } else { OP_MULADD };
                vec![(op, false)]
            }
            _ => {
                let shaped = site.a.clane == 0 && site.b.ci == 0 && site.radix.is_none();
                if !shaped || site.elem != Ty::f32() || !g.caps.cp_async {
                    return Vec::new();
                }
                let group = matches!(self.id, WGMMA | WGMMA_WS);
                let on = match self.id {
                    MMA => rungs.mma,
                    WGMMA => rungs.mma && rungs.wgmma,
                    _ => rungs.mma && rungs.wgmma && rungs.wgmma_ws,
                };
                if !on || (group && !g.has_wgmma()) {
                    return Vec::new();
                }
                let mut out = Vec::new();
                for (j, o) in g.caps.matrix.iter().enumerate() {
                    if (o.shape.threads > 32) != group || o.shape.threads <= 1 {
                        continue;
                    }
                    let admitted = match o.fmt {
                        MmaFmt::Tf32 => site.tf32 && rungs.tf32_tile && (group || o.shape.k == 8),
                        f => {
                            site.half.is_some_and(|h| MmaFmt::of_narrow(h) == f)
                                && rungs.half_tile
                                && o.acc_meets(-23)
                                && gpu::mma_widths(site, f).is_some()
                                && (group || (g.caps.ldmatrix && o.shape.k == 32 / f.bytes()))
                        }
                    };
                    let shape = group || (o.shape.m == 16 && o.shape.n == 8);
                    // A warpgroup product's `a` from shared memory where the
                    // site stores it in the format, else from registers
                    // (staged f32, rounded in the fragment load).
                    let a_home = match (group, o.fmt) {
                        (true, f) if f != MmaFmt::Tf32 => gpu::mma_widths(site, f)
                            .is_some_and(|(ea, _)| (o.a == MHome::Smem) == (ea == f.bytes())),
                        _ => true,
                    };
                    if !a_home {
                        continue;
                    }
                    if admitted && shape && o.a == MHome::Regs || admitted && group {
                        out.push((j + 2, false));
                        if group {
                            out.push((j + 2, true));
                        }
                    }
                }
                out.dedup();
                out
            }
        }
    }

    fn dims(&self) -> usize {
        self.coords().len()
    }

    fn search_domain(&self, _m: &Machine, _s: &Site, i: usize, pre: &[u64]) -> Vec<u64> {
        self.narrowed(i, pre)
    }

    fn point(&self, _m: &Machine, _s: &Site, pre: &[u64], swap: bool) -> Option<Schedule> {
        self.build(pre, swap)
    }

    fn lanes_of(&self, pre: &[u64], _swap: bool) -> Option<(u64, u64)> {
        match self.id {
            FMA if pre.len() >= 5 => Some((pre[3] * pre[4], pre[1] * pre[2])),
            MMA if pre.len() >= 5 => Some((pre[3] * pre[4] * self.gpu().warp, pre[1] * pre[2])),
            _ => None,
        }
    }

    fn tile_of(&self, pre: &[u64], _swap: bool) -> Option<(u64, u64)> {
        match self.id {
            FMA if pre.len() >= 5 => Some((pre[4] * pre[1], pre[3] * pre[2])),
            MMA if pre.len() >= 5 => Some((pre[3] * pre[1] * 16, pre[4] * pre[2] * 8)),
            WGMMA | WGMMA_WS if pre.len() >= 8 => Some(match pre[7] {
                1 => (pre[1], pre[2] * 64),
                _ => (pre[2] * 64, pre[1]),
            }),
            _ => None,
        }
    }

    fn slots(&self, m: &Machine, s: &Site, op: usize, swap: bool) -> Vec<Schedule> {
        // A depth-first walk of the coordinate domains, capped (the search's
        // pruning is C4's): each complete point lowered through the rules'
        // own encoding.
        let mut out = Vec::new();
        let n = self.coords().len();
        let mut stack: Vec<Vec<u64>> = vec![vec![op as u64]];
        while let Some(pre) = stack.pop() {
            if out.len() >= 4096 {
                break;
            }
            if pre.len() == n {
                if let Some(x) = self.build(&pre, swap) {
                    let mut x = x;
                    if let Ok(d) = self.declare(m, s, &x) {
                        x.per_unit = m.occupancy(d.threads, d.regs, &[(SMEM, d.smem)]);
                        out.push(x);
                    }
                }
                continue;
            }
            for v in self.domain(pre.len(), &pre) {
                let mut p = pre.clone();
                p.push(v);
                stack.push(p);
            }
        }
        out
    }

    fn declare(&self, _m: &Machine, s: &Site, x: &Schedule) -> Result<Declared, Refusal> {
        let g = self.gpu();
        let site = self.site;
        let e = elem_bytes_of(&site.elem);
        let (bm, bn, bk) = (x.bm().max(1), x.bn().max(1), x.bk().max(1));
        if x.lowering != NAIVE
            && (!s.m.is_multiple_of(bm) || !s.n.is_multiple_of(bn) || !s.k.is_multiple_of(bk))
        {
            return Err(Refusal::new("tiles do not divide the site (no tails)"));
        }
        let warp = g.warp.max(1);
        let l = launch_of(site, self.profile, self.contract, self.arms, x);
        if l.smem > g.smem_per_block {
            return Err(Refusal::new("shared memory above a slot's"));
        }
        if l.block > g.max_threads_per_block {
            return Err(Refusal::new("threads above a slot's"));
        }
        // ADR-0050 C4: the emitters' kernels for each traversal — the walk
        // and the hybrid (`Kernel::segment`) on the register-staged FMA tile
        // and the 16-bit warp tile without a split form; a raster on the
        // block-indexed kernels (the one-thread-per-cell kernel numbers its
        // cells).
        let walked = matches!(x.trav, Traversal::Walk { .. } | Traversal::Hybrid { .. });
        if walked {
            let ok = match (l.tiled, l.mma) {
                (Some(t), _) => t.stages < 2,
                (None, Some(m)) => m.fmt != MmaFmt::Tf32 && m.split.is_none(),
                _ => false,
            };
            if !ok {
                return Err(Refusal::new("the walk runs the staged FMA or 16-bit tile"));
            }
        }
        if self.id == NAIVE && x.trav.raster() > 1 {
            return Err(Refusal::new("the cell kernel has no raster"));
        }
        if let Some(t) = l.tiled {
            gpu::fma_constraints(site, &t, &l.rungs).map_err(Refusal::new)?;
        }
        if let Some(w) = l.wgmma.filter(|w| w.ws) {
            gpu::ws_tensors(site, &w, g, l.rungs.tma_digits)
                .ok_or_else(|| Refusal::new("no tensor boxes for the warp-specialized tile"))?;
        }
        let stages = x.depth();
        // `min-blocks`: the kernel asks for `per_unit` slots an SM
        // (`.minnctapersm`), so the code generator caps a lane's registers
        // at the file's share — the declared footprint is the counted one
        // within that cap (a spill past it is the post-lowering gate's,
        // ADR-0050 D5).
        let cap = |threads: u64| -> u64 {
            match (l.rungs.min_blocks, x.per_unit) {
                (true, u) if u > 0 => {
                    (g.regs_per_sm / u / threads.div_ceil(warp).max(1) / warp) / 8 * 8
                }
                _ => g.regs_per_thread,
            }
        };
        let pb = (bm * bk * s.ea + bn * bk * s.eb) as f64;
        let warps = l.block.div_ceil(warp) as f64;
        // The ring's in-flight discipline: `cp.async` waits to `stages − 2`
        // groups, so `stages − 1` panels fly; register staging one; a bulk
        // ring `stages − 1`.
        let fly = |stages: u64| (stages.saturating_sub(1).max(1) as f64 * pb) as u64;
        // The combine's epilogue: a piece's partials, one `red` a cell a lane,
        // issued where its piece ends — spread over the piece's trips.
        let panels = (s.k / bk).max(1) as f64;
        let reds = match x.trav {
            Traversal::Split {
                s: parts,
                comb: mapal_place::Comb::Atomic,
                ..
            } => (bm * bn) as f64 / warp as f64 / (panels / parts as f64).max(1.0),
            Traversal::Walk { iters, .. } => {
                (bm * bn) as f64 / warp as f64 * 2.0 / (iters.max(1) as f64)
            }
            _ => 0.0,
        };
        let copies = |bytes: f64| bytes / GPU_VEC_BYTES as f64 / warp as f64;
        match self.id {
            NAIVE => {
                if l.tiled.is_some() || l.mma.is_some() {
                    return Err(Refusal::new("not the naive kernel"));
                }
                let cells = param(x, "cells").max(1);
                // The vector-cells kernel (`naive_cells`) reads `b` as one
                // vector of a row's `cells` consecutive lanes at an affine
                // offset a `k` step and `a` once a row: a declared constraint
                // (cost-model ABBA: the 2048 x 2048 x 25 window's mixed-radix
                // `b` at 16 cells a thread stored wrong values).
                if cells > 1 && !gpu::naive_cells_admit(site, cells) {
                    return Err(Refusal::new(
                        "vector cells: affine rows only (no radix, no k split, b lanes unit-stride)",
                    ));
                }
                // A thread folds its cells over `k`: NAIVE_UNROLL loads of
                // `b` (one vector) and one of `a` a trip in flight.
                let trip = NAIVE_UNROLL as f64;
                Ok(Declared {
                    threads: NAIVE_THREADS,
                    // The address, loop and in-flight load budget up to the
                    // registers a lane keeps at full occupancy (the code
                    // generator fills that budget: 32 where an SM holds 2048
                    // threads, 40 at 1536), and the cells' accumulators past
                    // four: `ptxas -v` on sm_80 – sm_120, f32 and bf16 `b`,
                    // 31–40 registers at 1–8 cells, 43–48 at 16 (test
                    // `naive_declared_registers_cover_ptxas`).
                    // (The cell kernel emits no `min-blocks`: no cap.)
                    regs: (g.regs_per_sm / g.max_threads_per_sm.max(1) / 8 * 8)
                        .min(NAIVE_REGS)
                        .max(32 + (cells.saturating_sub(4) * 4).div_ceil(3)),
                    smem: 0,
                    issue: vec![
                        (Class::Math, warps * trip * cells as f64),
                        (Class::Global, warps * trip * 2.0 + reds),
                        (Class::Other, warps * trip),
                    ],
                    in_flight: NAIVE_THREADS * NAIVE_UNROLL * (cells * s.eb + s.ea),
                    fill: 1,
                    chains: cells,
                    groups: NAIVE_THREADS / warp,
                    work: 0.0,
                    whole_stores: false,
                    // A lane loads its cells' `b` and its row's `a` into
                    // registers.
                    mover: mapal_place::Move::Load(GPU_VEC_BYTES.min(cells * s.eb)),
                    store: GPU_VEC_BYTES.min(cells * s.eo),
                    ..Declared::default()
                })
            }
            FMA => {
                let t = l.tiled.ok_or_else(|| Refusal::new("no FMA tile"))?;
                if site.a.clane != 0 || site.b.ci != 0 {
                    return Err(Refusal::new("the FMA tile needs a separable site"));
                }
                if t.held > 0 && t.held < fma_held_regs(t.tm, t.tn, e) {
                    return Err(Refusal::new("held registers below the counted footprint"));
                }
                let key = [
                    t.tm,
                    t.tn,
                    t.tx,
                    t.ty,
                    t.bk,
                    t.stages.min(2),
                    u64::from(t.kmajor),
                    t.pad_a,
                    t.a_row_lanes,
                ];
                // Memoized only inside a search (one site, one rung table).
                let active = SEARCHING.with(|c| c.get());
                let hit = active
                    .then(|| LSU.with(|c| c.borrow().get(&key).copied()))
                    .flatten();
                let lsu = hit.unwrap_or_else(|| {
                    let v = gpu::lsu_per_panel(site, &t, &l.rungs, false);
                    if active {
                        LSU.with(|c| c.borrow_mut().insert(key, v));
                    }
                    v
                }) as f64;
                let ffma = (t.tm * t.tn * t.bk) as f64;
                let regs = match t.held {
                    0 => {
                        (t.tm * t.tn + 2 * (t.tm + t.tn)) * e.div_ceil(4)
                            + crate::profile::FMA_ADDR_REGS
                    }
                    h => h,
                };
                Ok(Declared {
                    threads: t.threads(),
                    regs: regs.min(cap(t.threads())),
                    smem: l.smem,
                    issue: vec![
                        (
                            Class::Math,
                            warps * ffma * if x.op == OP_MULADD { 2.0 } else { 1.0 },
                        ),
                        (Class::Shared, warps * lsu),
                        (Class::Global, reds),
                        (Class::Barrier, warps * if stages > 1 { 1.0 } else { 2.0 }),
                    ],
                    in_flight: fly(stages),
                    fill: stages.saturating_sub(1).max(1),
                    chains: t.tm * t.tn,
                    groups: t.threads().div_ceil(warp),
                    work: 0.0,
                    whole_stores: false,
                    // A `cp.async` ring of whole 16-byte copies, or one panel
                    // staged through registers; a lane stores its `tn`
                    // consecutive cells.
                    mover: match stages {
                        0 | 1 => mapal_place::Move::Load(GPU_VEC_BYTES),
                        _ => mapal_place::Move::Copy(GPU_VEC_BYTES),
                    },
                    barriers: if stages > 1 { 1.0 } else { 2.0 },
                    store: GPU_VEC_BYTES.min(t.tn * s.eo),
                    // A lane's `tn` consecutive cells over `tn` reductions:
                    // each instruction's sectors the next ones revisit — the
                    // probe's contiguous row (`red.add.f32`, the default),
                    // not its sector-a-lane one (no revisit).
                    ..Declared::default()
                })
            }
            MMA => {
                let mt = l.mma.ok_or_else(|| Refusal::new("no warp mma tile"))?;
                let o = g.caps.matrix[x.op - 2];
                // The warp tile's ring: two stages at least (its emitter
                // waits on `stages − 2` groups; one stage, a single-panel
                // fold, computes no product — found by the picks' value
                // test), at most a stage a panel.
                if mt.stages < 2 || mt.stages > s.k / bk {
                    return Err(Refusal::new("stages in [2, k/bk]"));
                }
                // HMMA a trip: the warp tile's instructions per `k` step, a
                // split form's parts each.
                // A split form issues its kept part products a step.
                let products = mt
                    .split
                    .map_or(1.0, |k| mapal_ir::split::SplitForm::of(k).products() as f64);
                let parts = products;
                let ksteps = (bk / o.shape.k.max(1)) as f64;
                let hmma = (mt.mt * mt.nt) as f64 * ksteps * parts;
                // ldmatrix.x4 covers a 16 × 16 bytes-of-b16 tile: `a`'s mt
                // tiles and `b`'s nt / 2 a k step.
                let ldsm = (mt.mt as f64 + (mt.nt as f64 / 2.0).ceil()) * ksteps;
                // The present emitter's two counts (`gpu_mma_tile_at`): the
                // S52 rule's twice the accumulators, the counted
                // accumulators, fragments and addresses — the smaller.
                let acc = mt.mt * mt.nt * 4;
                let frag = (mt.mt * 4 + mt.nt * 2) * if mt.split.is_some() { 2 } else { 1 };
                let counted = (acc + frag + crate::profile::MMA_ADDR_REGS).min(2 * acc);
                Ok(Declared {
                    threads: mt.threads(warp),
                    regs: counted.min(cap(mt.threads(warp))),
                    smem: l.smem,
                    issue: vec![
                        (Class::Matrix, warps * hmma),
                        (Class::Shared, warps * ldsm),
                        (Class::Global, copies(pb) + reds),
                        (
                            Class::Convert,
                            match (mt.split.is_some(), mt.ea > o.fmt.bytes()) {
                                (true, _) | (_, true) => warps * ldsm * 4.0,
                                _ => 0.0,
                            },
                        ),
                        (Class::Barrier, warps),
                    ],
                    in_flight: fly(mt.stages),
                    fill: mt.stages.saturating_sub(1).max(1),
                    chains: mt.mt * mt.nt,
                    groups: mt.warps_x * mt.warps_y,
                    work: (2 * bm * bn * bk) as f64 * products,
                    // `mma_walk_store`: a whole piece stores as the ordered
                    // kernel does.
                    whole_stores: true,
                    // A `cp.async` ring of 16-byte copies; the fragments by
                    // `ldmatrix.x4` (512 bytes a warp; 32-bit operands' lane
                    // loads), one `k` step's ahead of their products; a lane
                    // stores its accumulator fragment's two cells.
                    mover: mapal_place::Move::Copy(GPU_VEC_BYTES),
                    scratch: (warps * ldsm * 512.0) as u64,
                    scratch_fly: (warps * ldsm / ksteps.max(1.0) * 512.0) as u64,
                    scratch_matrix: o.fmt != MmaFmt::Tf32,
                    barriers: 1.0,
                    store: 2 * s.eo,
                    // The accumulator fragment: 8 rows of 32 bytes a warp's
                    // reduction instruction.
                    add_sectors: 8,
                    ..Declared::default()
                })
            }
            _ => {
                let w = l.wgmma.ok_or_else(|| Refusal::new("no warpgroup tile"))?;
                if (x.lowering == WGMMA_WS) != w.ws {
                    return Err(Refusal::new("warp specialization is the lowering's"));
                }
                if !w.n.is_multiple_of(WGMMA_N_STEP) || w.n > WGMMA_N_MAX {
                    return Err(Refusal::new("n a multiple of 8 up to 256"));
                }
                if !w.ws && w.stages < crate::profile::WGMMA_MIN_STAGES {
                    return Err(Refusal::new("stages >= 3 (S53 ring)"));
                }
                let o = g.caps.matrix[x.op - 2];
                // The direct product `c = a·b` reads `b`'s stage as a
                // descriptor operand in either major, which only an
                // instruction with `MatrixOp::mn_major` reads; a K-major-only
                // one (TF32, fp8 wgmma) takes the transposed product, as the
                // rules lower it (`GpuWgmmaTile::direct`). cost-rental: the
                // H100's search picked a direct TF32 tile and ptxas refused
                // its `wgmma.mma_async` (pt_mlp_2048_tf32).
                if w.direct && !o.mn_major {
                    return Err(Refusal::new(
                        "a K-major-only instruction takes the transposed product",
                    ));
                }
                let ksteps = (bk / o.shape.k.max(1)) as f64;
                let acc = 64 * w.n / (WARPGROUP_WARPS * warp);
                // Under `setmaxnreg` the roles split the block's pool: the
                // lanes' mean is what the block takes from the file.
                let regs = match w.realloc {
                    Some((p, c)) => {
                        let wg = WARPGROUP_WARPS * warp;
                        (wg * p + w.groups * wg * c).div_ceil(w.threads(warp))
                    }
                    None => {
                        acc + match w.ws {
                            true => crate::profile::WGMMA_WS_ADDR_REGS,
                            false => crate::profile::WGMMA_ADDR_REGS,
                        }
                    }
                };
                Ok(Declared {
                    threads: w.threads(warp),
                    regs,
                    smem: l.smem,
                    issue: vec![
                        (Class::Matrix, w.groups as f64 * ksteps),
                        (
                            Class::Global,
                            match w.ws {
                                true => 1.0 + w.ws_b_boxes() as f64,
                                false => copies(pb),
                            },
                        ),
                        (Class::Barrier, w.groups as f64 + 1.0),
                    ],
                    in_flight: fly(w.stages),
                    fill: w.stages.saturating_sub(1).max(1),
                    chains: 1,
                    groups: w.groups * WARPGROUP_WARPS,
                    work: 0.0,
                    whole_stores: false,
                    // The warp-specialized tile's panels by tensor boxes, the
                    // plain one's by `cp.async`; the products read the
                    // scratchpad by descriptor (no lane loads).
                    mover: match w.ws {
                        true => mapal_place::Move::Bulk(
                            (pb as u64 / (1 + w.ws_b_boxes()).max(1)).max(1),
                        ),
                        false => mapal_place::Move::Copy(GPU_VEC_BYTES),
                    },
                    barriers: 1.0,
                    store: 2 * s.eo,
                    add_sectors: 8,
                    ..Declared::default()
                })
            }
        }
    }
}

impl Emitter<'_> {
    /// **The search's domain of coordinate `i`** given `pre` (ADR-0050 C4):
    /// [`Self::domain`] narrowed by the site's facts — an operand's stored
    /// layout (`bt`, `b_kmajor`) and staged widths (`ea`, `eb`) are the
    /// site's, tiles divide its extents, a slot's lanes fit a block, a
    /// lane's accumulators its registers — and by dominance: a layout's pads
    /// and lanes are the emitter's own conflict-free choice by the bank
    /// model (no evaluator term reads them), a whole-file budget (`held`)
    /// and a register reallocation (`regs_p`, `regs_c`) the counted
    /// footprint, the split forms the rules' (an `{f32}` product's split
    /// realization stays the incumbent's: open, ADR-0050 C6).
    fn narrowed(&self, i: usize, pre: &[u64]) -> Vec<u64> {
        let g = self.gpu();
        let site = self.site;
        let e = elem_bytes_of(&site.elem);
        let threads_max = g.max_threads_per_block;
        let rows = gpu::row_group(site);
        let mut dom = self.domain(i, pre);
        let name = self.coords()[i];
        let divides = |n: u64, d: u64| d > 0 && n.is_multiple_of(d);
        match (self.id, name) {
            (FMA, "tn") => dom.retain(|&tn| fma_held_regs(pre[1], tn, e) <= g.regs_per_thread),
            (FMA, "tx") => dom.retain(|&tx| divides(site.c, tx * pre[2])),
            // A slot holds whole groups (the dispatch issues a group's lanes
            // together; a partial group's lanes idle).
            (FMA, "ty") => dom.retain(|&ty| {
                ty * pre[3] <= threads_max
                    && (ty * pre[3]).is_multiple_of(g.warp)
                    && divides(rows, ty * pre[1])
            }),
            // Every thread copies an equal share of each panel.
            (FMA, "bk") => {
                let (bm, bn, threads) = (pre[4] * pre[1], pre[3] * pre[2], pre[3] * pre[4]);
                dom.retain(|&bk| {
                    (bm * bk).is_multiple_of(threads) && (bk * bn).is_multiple_of(threads)
                });
            }
            // A streamed (`cp.async`) panel where the emitter streams it,
            // and the stages the slot's shared memory holds.
            (FMA, "stages") => {
                let r = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
                let streams = gpu::fma_streams(site, &r);
                dom.retain(|&st| {
                    let t = GpuTile {
                        stages: st,
                        ..self.fma_tile(pre)
                    };
                    st <= (site.k / pre[5].max(1)).max(1)
                        && (st == 1 || streams)
                        && t.smem_bytes(e) <= g.smem_per_block
                })
            }
            (FMA, "kmajor") => {
                if !(site.a.ck == 1 && pre[3] >= gpu::GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES) {
                    dom = vec![0];
                }
            }
            (FMA, "pad_a") | (FMA, "a_row_lanes") => {
                // The layout of the prefix's tile (one derivation for both
                // coordinates and every stage count past the first).
                let t = self.fma_tile(pre);
                let key = [
                    t.tm,
                    t.tn,
                    t.tx,
                    t.ty,
                    t.bk,
                    t.stages.min(2),
                    u64::from(t.kmajor),
                ];
                let l = LAYOUT
                    .with(|c| c.borrow().get(&key).copied())
                    .unwrap_or_else(|| {
                        let r = memo_rungs(site, self.profile, self.arms, Arithmetic::OrderedFma);
                        let l = gpu::fma_layout(site, self.profile, t, &r);
                        let l = (l.pad_a, l.a_row_lanes);
                        LAYOUT.with(|c| c.borrow_mut().insert(key, l));
                        l
                    });
                dom = vec![match name {
                    "pad_a" => l.0,
                    _ => l.1,
                }];
            }
            (FMA, "held") => dom = vec![0],
            (MMA, "nt") => dom
                .retain(|&nt| pre[1] * nt * 4 + crate::profile::MMA_ADDR_REGS <= g.regs_per_thread),
            (MMA, "warps_y") => dom.retain(|&wy| divides(rows, wy * pre[1] * 16)),
            (MMA, "warps_x") => dom.retain(|&wx| {
                wx * pre[3] * g.warp <= threads_max && divides(site.c, wx * pre[2] * 8)
            }),
            // Whole 16-byte copies a thread of each panel.
            (MMA, "bk") => {
                let (ea, eb) = self.widths(pre[0]).unwrap_or((4, 4));
                let (bm, bn) = (pre[3] * pre[1] * 16, pre[4] * pre[2] * 8);
                let threads = pre[3] * pre[4] * g.warp;
                let (la, lb) = (GPU_VEC_BYTES / ea, GPU_VEC_BYTES / eb);
                dom.retain(|&bk| {
                    (bm * bk / la).is_multiple_of(threads) && (bk * bn / lb).is_multiple_of(threads)
                });
            }
            // A `cp.async` ring of two stages at least (the emitter's ring
            // waits on `stages − 2` groups: a single-panel fold of one stage
            // stores no product), within the slot's shared memory.
            (MMA, "stages") => dom.retain(|&st| {
                st >= 2
                    && self
                        .mma_tile(pre, st)
                        .is_some_and(|t| t.smem_bytes() <= g.smem_per_block)
            }),
            (MMA, "bt") => dom = vec![u64::from(site.b.clane != 1)],
            (MMA, "pad_a") | (MMA, "pad_b") => {
                let pads = self.mma_pads(pre);
                dom = match (name, pads) {
                    (_, None) => vec![],
                    ("pad_a", Some((a, _))) => vec![a],
                    (_, Some((_, b))) => vec![b],
                };
            }
            (MMA, "split") => dom = vec![0],
            (MMA, "ea") | (MMA, "eb") | (_, "ea") | (_, "eb") => {
                let (ea, eb) = self.widths(pre[0]).unwrap_or((0, 0));
                dom = vec![if name == "ea" { ea } else { eb }];
            }
            (WGMMA | WGMMA_WS, "n") => dom.retain(|&n| n <= 256),
            (WGMMA | WGMMA_WS, "bk") => {
                let (_, eb) = self.widths(pre[0]).unwrap_or((4, 4));
                dom.retain(|&bk| (bk * eb).is_multiple_of(128));
            }
            (WGMMA, "stages") => dom.retain(|&st| st >= crate::profile::WGMMA_MIN_STAGES),
            // The warp-specialized ring: two stages at least (one only where
            // one panel is the fold: the direct form).
            (WGMMA_WS, "stages") => dom.retain(|&st| st >= 2 || site.k / pre[3].max(1) == 1),
            (WGMMA | WGMMA_WS, "b_kmajor") => dom = vec![u64::from(site.b.clane != 1)],
            (WGMMA | WGMMA_WS, "pad_b") => dom = vec![0],
            (WGMMA | WGMMA_WS, "swap") => {
                let (n, rows_g) = (pre[1], pre[2] * 64);
                dom.retain(|&sw| match sw {
                    1 => divides(site.rows, n) && divides(site.c, rows_g),
                    _ => divides(site.rows, rows_g) && divides(site.c, n),
                });
            }
            (WGMMA | WGMMA_WS, "regs_p") | (WGMMA | WGMMA_WS, "regs_c") => dom = vec![0],
            _ => {}
        }
        // The largest tiles and panels first: the threshold the bound cuts
        // at tightens early (the pick is the same in any order).
        if matches!(
            name,
            "cells"
                | "tm"
                | "tn"
                | "tx"
                | "ty"
                | "mt"
                | "nt"
                | "warps_y"
                | "warps_x"
                | "n"
                | "groups"
                | "bk"
        ) {
            dom.reverse();
        }
        dom
    }

    /// The warp tile's panel pads by the bank model (the emitter's rule,
    /// `gpu_mma_panel_cands`): a TF32 panel's conflict-free pad, an f32
    /// panel of a 16-bit product's pair pad, a 16-bit panel's `ldmatrix`
    /// rows; `None` where no pad is conflict-free.
    fn mma_pads(&self, pre: &[u64]) -> Option<(u64, u64)> {
        use crate::profile::{conflict_free_pad, ldmatrix_pad, pair_pad};
        let fmt = self.fmt(pre[0])?;
        let (ea, eb) = self.widths(pre[0])?;
        let (bk, bn) = (pre[5], pre[4] * pre[2] * 8);
        let bt = self.site.b.clane != 1;
        let pad_a = match (fmt, ea) {
            (MmaFmt::Tf32, _) => conflict_free_pad(bk, |l| (l / 4, l % 4))?,
            (_, 4) => pair_pad(bk)?,
            _ => ldmatrix_pad(bk),
        };
        let pad_b = match (bt, fmt, eb) {
            (false, MmaFmt::Tf32, _) => conflict_free_pad(bn, |l| (l % 4, l / 4))?,
            (false, _, 4) => conflict_free_pad(bn, |l| (2 * (l % 4), l / 4))?,
            (false, _, _) => ldmatrix_pad(bn),
            (true, _, 4) => pair_pad(bk)?,
            (true, _, _) => ldmatrix_pad(bk),
        };
        Some((pad_a, pad_b))
    }

    /// The warp tile of an MMA prefix (through `bk`) at `stages`.
    fn mma_tile(&self, pre: &[u64], stages: u64) -> Option<GpuMmaTile> {
        let (ea, eb) = self.widths(pre[0])?;
        let (pad_a, pad_b) = self.mma_pads(pre)?;
        Some(GpuMmaTile {
            mt: pre[1],
            nt: pre[2],
            warps_y: pre[3],
            warps_x: pre[4],
            bk: pre[5],
            stages,
            pad_a,
            pad_b,
            blocks_per_sm: 0,
            fmt: self.fmt(pre[0])?,
            ea,
            eb,
            bt: self.site.b.clane != 1,
            split: None,
        })
    }

    /// The matrix row's format of operation id `op` (`None`: a scalar op).
    fn fmt(&self, op: u64) -> Option<MmaFmt> {
        self.gpu()
            .caps
            .matrix
            .get((op as usize).wrapping_sub(2))
            .map(|o| o.fmt)
    }

    /// The panel widths the site's operands stage at on operation `op`: a
    /// 16-bit format's ([`gpu::mma_widths`]), TF32's four bytes.
    fn widths(&self, op: u64) -> Option<(u64, u64)> {
        match self.fmt(op)? {
            MmaFmt::Tf32 => Some((4, 4)),
            f => gpu::mma_widths(self.site, f),
        }
    }

    /// The FMA tile of an FMA prefix (through `kmajor`), its layout unset.
    fn fma_tile(&self, c: &[u64]) -> GpuTile {
        GpuTile {
            tm: c[1],
            tn: c[2],
            tx: c[3],
            ty: c[4],
            bk: c[5],
            blocks_per_sm: 0,
            stages: c.get(6).copied().unwrap_or(1),
            pad_a: 0,
            a_row_lanes: 0,
            held: 0,
            kmajor: c.get(7) == Some(&1),
        }
    }

    /// A schedule from coordinates (the enumeration's point), or `None`
    /// where the coordinates name no tile of this lowering.
    fn build(&self, c: &[u64], swap: bool) -> Option<Schedule> {
        let g = self.gpu();
        let s = self.site;
        let grid = Level {
            ext: [s.rows, s.c, s.k],
            bind: Bind::Exec(E_DEVICE),
            homes: [None, None],
        };
        let mk = |lowering, op: u64, levels, groups| Schedule {
            lowering,
            op: op as usize,
            swap,
            levels,
            groups,
            producers: 0,
            regs: [0, 0],
            cluster: [1, 1],
            multicast: [false, false],
            trav: Traversal::Ordered { h: 1 },
            launch: LaunchMode::Plain,
            policy: [Policy::Default; 2],
            per_unit: 0,
            params: Vec::new(),
        };
        let maj = |b: u64| if b == 1 { Major::K } else { Major::Mn };
        Some(match self.id {
            NAIVE => {
                let cells = c[1];
                let bn = (NAIVE_THREADS * cells).min(s.c);
                let bm = ((NAIVE_THREADS * cells) / s.c.max(1)).clamp(1, s.rows.max(1));
                let mut x = mk(
                    NAIVE,
                    c[0],
                    vec![
                        Level {
                            ext: [1, cells, 1],
                            bind: Bind::Exec(E_LANE),
                            homes: [None, None],
                        },
                        Level {
                            ext: [bm, bn, NAIVE_UNROLL],
                            bind: Bind::Slot,
                            homes: [None, None],
                        },
                        grid,
                    ],
                    [NAIVE_THREADS / g.warp.max(1), 1],
                );
                x.params = vec![("cells", cells)];
                x
            }
            FMA => {
                let (tm, tn, tx, ty, bk, st) = (c[1], c[2], c[3], c[4], c[5], c[6]);
                let mut x = mk(
                    FMA,
                    c[0],
                    vec![
                        Level {
                            ext: [tm, tn, 1],
                            bind: Bind::Exec(E_LANE),
                            homes: [None, None],
                        },
                        Level {
                            ext: [ty, tx, 1],
                            bind: Bind::Exec(E_GROUP),
                            homes: [None, None],
                        },
                        Level {
                            ext: [ty * tm, tx * tn, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, maj(c[7]), c[8], 0),
                                smem_home(st, Major::Mn, 0, 0),
                            ],
                        },
                        grid,
                    ],
                    [(tx * ty).div_ceil(g.warp), 1],
                );
                x.params = vec![("a_row_lanes", c[9]), ("held", c[10])];
                x
            }
            MMA => {
                let o = g.caps.matrix.get((c[0] as usize).wrapping_sub(2))?;
                let (mt, nt, wy, wx, bk, st) = (c[1], c[2], c[3], c[4], c[5], c[6]);
                let mut x = mk(
                    MMA,
                    c[0],
                    vec![
                        Level {
                            ext: [16 * mt, 8 * nt, o.shape.k],
                            bind: Bind::Exec(E_GROUP),
                            homes: [None, None],
                        },
                        Level {
                            ext: [wy * 16 * mt, wx * 8 * nt, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, Major::K, c[8], 0),
                                smem_home(st, maj(c[7]), c[9], 0),
                            ],
                        },
                        grid,
                    ],
                    [wy, wx],
                );
                x.params = vec![
                    ("ea", c[11]),
                    ("eb", c[12]),
                    ("split", c[10]),
                    ("mt", mt),
                    ("nt", nt),
                ];
                if c[10] > 0 {
                    x.params.push(("split_arith", 1));
                }
                x
            }
            _ => {
                let (n, groups, bk, st) = (c[1], c[2], c[3], c[4]);
                let swap = c[7] == 1;
                let (bm, bn) = match swap {
                    false => (groups * 64, n),
                    true => (n, groups * 64),
                };
                let a_home = match g
                    .caps
                    .matrix
                    .get((c[0] as usize).wrapping_sub(2))
                    .map(|o| o.a)
                {
                    Some(MHome::Regs) => REGS,
                    _ => SMEM,
                };
                let mut x = mk(
                    self.id,
                    c[0],
                    vec![
                        Level {
                            ext: [64, n, 8],
                            bind: Bind::Exec(E_GROUP),
                            homes: [
                                Some(Home {
                                    place: a_home,
                                    layout: Layout::K,
                                    depth: 1,
                                }),
                                None,
                            ],
                        },
                        Level {
                            ext: [bm, bn, bk],
                            bind: Bind::Slot,
                            homes: [
                                smem_home(st, Major::K, 0, 128),
                                smem_home(
                                    st,
                                    maj(c[5]),
                                    c[6],
                                    if self.id == WGMMA_WS { 128 } else { 0 },
                                ),
                            ],
                        },
                        grid,
                    ],
                    [groups, 1],
                );
                x.swap = swap;
                x.producers = u64::from(self.id == WGMMA_WS);
                x.regs = [c[10], c[11]];
                x.params = vec![
                    ("ea", c[8]),
                    ("eb", c[9]),
                    ("pad_b", c[6]),
                    ("b_kmajor", c[5]),
                    ("n", n),
                ];
                x
            }
        })
    }
}

// ---- the search's seam (ADR-0050 C4) --------------------------------------------

/// The measurements' null the search separates candidates by
/// (`place_replay.tsv`'s `null` column: 2 %).
pub const PLACE_NULL: f64 = 0.02;

/// **The model's error band** (ADR-0050 D5): per bounding term of the
/// pick's longest phase, the 80th percentile of the replay's relative
/// residuals — `|(model_a / model_b) / (measured_a / measured_b) − 1|` over
/// each measured site's arm pairs, the pair's faster arm's term
/// (`place_replay.rs` `place_band_is_the_replays_residuals` recomputes it;
/// a term with fewer than three pairs takes `all`'s). Per mille.
pub const PLACE_BAND_PM: [(&str, u64); 5] = [
    ("all", 161),
    ("compute", 126),
    ("dram", 217),
    ("cache", 202),
    ("latency", 161),
];

/// The search's options at the band and null above.
pub fn place_opts() -> mapal_place::Opts {
    mapal_place::Opts {
        top: 8,
        band: PLACE_BAND_PM
            .iter()
            .map(|(t, pm)| (t.to_string(), *pm as f64 / 1e3))
            .collect(),
        null: PLACE_NULL,
        reverse: false,
        kinds: Vec::new(),
    }
}

thread_local! {
    /// The launch every launch the emitter derives is replaced by while the
    /// post-lowering gate lowers a candidate ([`forced`]).
    static FORCED: std::cell::RefCell<Option<Launch>> = const { std::cell::RefCell::new(None) };
    /// The table rungs of the sites being searched, by site, part, arms and
    /// arithmetic ([`memo_rungs`]; cleared when a search ends).
    static RUNGS: std::cell::RefCell<Vec<(TileSite, &'static str, crate::GpuArms, Arithmetic, crate::GpuRungs)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The searches made, by site, part, face, arms and order.
    static PICKS: std::cell::RefCell<std::collections::HashMap<String, Option<Placed>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The FMA tile layouts of the site being searched ([`Emitter::narrowed`];
    /// cleared when a search ends).
    static LAYOUT: std::cell::RefCell<std::collections::HashMap<[u64; 7], (u64, u64)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The FMA tiles' shared-memory instructions a panel of the site being
    /// searched (the bank model's, [`gpu::lsu_per_panel`]; cleared with it).
    static LSU: std::cell::RefCell<std::collections::HashMap<[u64; 9], u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The rule chain is being derived ([`rules`]): no placement inside.
    static RULES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A search is running (the per-site memos are valid).
    static SEARCHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Each part's machine, by its properties.
    static MACHINES: std::cell::RefCell<std::collections::HashMap<String, Machine>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `f` with every launch the emitter derives replaced by `l` (the gate's
/// lowering of a candidate; the coverage probe's hooks read it).
pub(crate) fn forced<T>(l: &Launch, f: impl FnOnce() -> T) -> T {
    let prev = FORCED.with(|c| c.replace(Some(*l)));
    let out = f();
    FORCED.with(|c| *c.borrow_mut() = prev);
    out
}

fn forced_now() -> Option<Launch> {
    FORCED.with(|c| *c.borrow())
}

/// `f` with the rule chain alone (no placement inside: the incumbent and
/// the search's own derivations).
pub(crate) fn rules<T>(f: impl FnOnce() -> T) -> T {
    let prev = RULES.with(|c| c.replace(true));
    let out = f();
    RULES.with(|c| c.set(prev));
    out
}

fn rules_now() -> bool {
    RULES.with(|c| c.get())
}

/// [`table_rungs`], memoized while a search runs (every candidate's
/// `launch_of` reads them).
fn memo_rungs(
    site: &TileSite,
    profile: &TargetProfile,
    arms: crate::GpuArms,
    a: Arithmetic,
) -> crate::GpuRungs {
    let hit = RUNGS.with(|c| {
        c.borrow()
            .iter()
            .find(|(s, p, r, x, _)| *p == profile.name && *r == arms && *x == a && s == site)
            .map(|e| e.4)
    });
    if let Some(r) = hit {
        return r;
    }
    let r = table_rungs(site, profile, arms, a);
    RUNGS.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 64 {
            c.clear();
        }
        c.push((site.clone(), profile.name, arms, a, r));
    });
    r
}

/// The part's machine (translated once a process).
fn machine(p: &TargetProfile) -> Option<Machine> {
    let key = format!("{}|{:?}", p.name, p.gpu());
    if let Some(m) = MACHINES.with(|c| c.borrow().get(&key).cloned()) {
        return Some(m);
    }
    let m = machine_of(p)?;
    MACHINES.with(|c| c.borrow_mut().insert(key, m.clone()));
    Some(m)
}

/// **A placed site** (ADR-0050 C4): the launch the emission takes (the
/// pick's, or the rules' where the pick does not clear the band), the
/// search's best per realization (the kernels beside the pick), and the
/// `; mapal.place` line.
#[derive(Clone, Debug)]
pub(crate) struct Placed {
    pub launch: Launch,
    pub plan: String,
    pub line: String,
    pub open: bool,
    pub kept: bool,
}

/// The `--place-gate=ptxas` arm: the post-lowering gate also assembles each
/// candidate (`llc`, `ptxas -v` where present): a spill, or registers above
/// the declared count's cap, refuses it.
fn ptxas_gate(p: &TargetProfile, text: &str, d: &Declared, threads: u64) -> Result<(), String> {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("llc") || !have("ptxas") {
        return Ok(());
    }
    let arch = p.gpu().map_or("sm_80", |g| g.arch);
    let dir = std::env::temp_dir().join(format!(
        "mapal-place-gate-{}-{:016x}",
        std::process::id(),
        gpu::fnv_text(text)
    ));
    let _ = std::fs::create_dir_all(&dir);
    let (ll, ptx, cubin) = (dir.join("k.ll"), dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&ll, gpu::kernel_module(text)).map_err(|e| e.to_string())?;
    let mut llc = Command::new("llc");
    llc.args(["-O3", &format!("-mcpu={arch}"), "-nvptx-fma-level=0"]);
    if let Some(x) = crate::Ptx::declared(&gpu::kernel_module(text)) {
        llc.arg(format!("-mattr={}", x.mattr()));
    }
    let ok = llc
        .arg(&ll)
        .arg("-o")
        .arg(&ptx)
        .status()
        .map_err(|e| e.to_string())?;
    if !ok.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err("llc refused the kernel".into());
    }
    let out = Command::new("ptxas")
        .args([&format!("-arch={arch}"), "-v"])
        .arg(&ptx)
        .arg("-o")
        .arg(&cubin)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    let log = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() {
        return Err(format!(
            "ptxas refused: {}",
            log.lines().next().unwrap_or("")
        ));
    }
    let regs: u64 = log
        .split("Used ")
        .nth(1)
        .and_then(|r| r.split(' ').next())
        .and_then(|r| r.parse().ok())
        .unwrap_or(0);
    if !log.contains("0 bytes spill stores, 0 bytes spill loads") {
        return Err(format!("ptxas spills ({regs} registers)"));
    }
    let g = p.gpu().expect("a GPU profile");
    if regs * threads > g.regs_per_sm || regs > g.regs_per_thread {
        return Err(format!("{regs} registers above the file"));
    }
    let _ = d;
    Ok(())
}

/// The post-lowering gate (D5): the candidate's launch lowers to a kernel
/// (the emitter's own derivation under [`forced`]; a refusal is its panic),
/// and under `--place-gate=ptxas` assembles without a spill.
fn gate(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    x: &Schedule,
    d: &Declared,
) -> Result<(), String> {
    let l = launch_of(site, p, contract, arms, x);
    let text = gpu::lowered_text(site, p, contract, arms, &l)?;
    if arms.place_gate {
        ptxas_gate(p, &text, d, l.block)?;
    }
    Ok(())
}

/// The search key of a site.
fn key(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: crate::Order,
) -> String {
    format!(
        "{site:?}|{}|{contract}|{arms:?}|{order:?}|{:?}",
        p.name,
        warm_of(site)
    )
}

/// **Search** a site (cached): the machine's pick over every emitter's
/// space, the rules' launch `rules` the incumbent.
fn search_site(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: crate::Order,
    rules: &Launch,
) -> Option<Placed> {
    let k = key(site, p, contract, arms, order);
    if let Some(hit) = PICKS.with(|c| c.borrow().get(&k).cloned()) {
        return hit;
    }
    LAYOUT.with(|c| c.borrow_mut().clear());
    LSU.with(|c| c.borrow_mut().clear());
    let was = SEARCHING.with(|c| c.replace(true));
    let placed = self::rules(|| search_uncached(site, p, contract, arms, order, rules));
    SEARCHING.with(|c| c.set(was));
    // No candidate through the gate (nor the incumbent as a schedule of the
    // space): the rules' launch, said so on the line.
    let placed = placed.or_else(|| {
        p.gpu().map(|_| Placed {
            launch: *rules,
            plan: "rules".into(),
            line: format!(
                "; mapal.place site={:016x} order={order:?} sched=rules kept=1 open=[none: no \
                 schedule of the space passed the gate]\n",
                gpu::kernel_id(site, p, contract)
            ),
            open: true,
            kept: true,
        })
    });
    RUNGS.with(|c| c.borrow_mut().clear());
    LAYOUT.with(|c| c.borrow_mut().clear());
    LSU.with(|c| c.borrow_mut().clear());
    PICKS.with(|c| c.borrow_mut().insert(k, placed.clone()));
    placed
}

fn search_uncached(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: crate::Order,
    rules: &Launch,
) -> Option<Placed> {
    let m = machine(p)?;
    let s = site_of(site, contract, order);
    let emitters = Emitter::all(site, p, contract, arms);
    let lows: Vec<&dyn Lowering> = emitters.iter().map(|e| e as &dyn Lowering).collect();
    let inc = emitters
        .iter()
        .position(|e| e.id == lowering_of(rules))
        .map(|i| (i, schedule_of(site, p, arms, rules)));
    let mut g = |x: &Schedule, d: &Declared| gate(site, p, contract, arms, x, d);
    let r = match (arms.place_trav, arms.place_tile) {
        (0, false) => mapal_place::search(&m, &s, &lows, &place_opts(), inc, &mut g)?,
        _ => forced_search(&m, &s, &emitters, arms, inc, &mut g)?,
    };
    let launch = match r.kept {
        true => *rules,
        false => launch_of(site, p, contract, arms, &r.pick.x),
    };
    let id = gpu::kernel_id(site, p, contract);
    let line = format!(
        "; mapal.place site={id:016x} order={order:?} sched={} ns={:.0} slow={:.0} fast={:.0} \
         kept={} best={}:{:.0} rules={} points={} evaluated={} pruned={} refused={} gated={} \
         terms=[{}] open=[{}]\n",
        r.pick.x.plan(),
        r.pick.ns[1],
        r.pick.ns[0],
        r.pick.ns[2],
        u8::from(r.kept),
        r.best.x.plan(),
        r.best.value(),
        r.incumbent
            .as_ref()
            .map_or("-".into(), |c| format!("{:.0}", c.value())),
        r.stats.points,
        r.stats.evaluated,
        r.stats.pruned,
        r.stats.refused,
        r.stats.gated.len(),
        r.pick.t.line(),
        r.open.join("; ")
    );
    Some(Placed {
        launch,
        plan: r.pick.x.plan(),
        line,
        open: !r.open.is_empty(),
        kept: r.kept,
    })
}

/// The traversal kind `--place-trav` names (0: any).
#[allow(dead_code)]
fn trav_kind(code: u8, t: Traversal) -> bool {
    match (code, t) {
        (0, _) => true,
        (1, Traversal::Ordered { .. }) => true,
        (2, Traversal::Split { .. }) => true,
        (3, Traversal::Walk { .. }) => true,
        (4, Traversal::Hybrid { .. }) => true,
        _ => false,
    }
}

/// The search under the measurement arms `--place-trav` / `--place-tile`:
/// every lowering's space with only that traversal kind, or the rules' slot
/// schedule `inc` under every traversal of the kind — the lowest priced
/// through the gate, never kept to the incumbent (an A/B arm).
fn forced_search(
    m: &Machine,
    s: &Site,
    emitters: &[Emitter],
    arms: crate::GpuArms,
    inc: Option<(usize, Schedule)>,
    gate: &mut dyn FnMut(&Schedule, &Declared) -> Result<(), String>,
) -> Option<mapal_place::Ranked> {
    /// One emitter, its slot space the rules' slot alone (`--place-tile`).
    struct Slot<'a, 'b> {
        e: &'b Emitter<'a>,
        x: Schedule,
    }
    impl Lowering for Slot<'_, '_> {
        fn id(&self) -> &'static str {
            self.e.id()
        }
        fn traversals(&self) -> &'static [TravKind] {
            self.e.traversals()
        }
        fn ops(&self, _m: &Machine, _s: &Site) -> Vec<(usize, bool)> {
            vec![(self.x.op, self.x.swap)]
        }
        fn slots(&self, _m: &Machine, _s: &Site, _op: usize, _swap: bool) -> Vec<Schedule> {
            vec![Schedule {
                trav: Traversal::Ordered { h: 1 },
                ..self.x.clone()
            }]
        }
        fn declare(&self, m: &Machine, s: &Site, x: &Schedule) -> Result<Declared, Refusal> {
            self.e.declare(m, s, x)
        }
    }
    let kinds = match arms.place_trav {
        1 => vec![TravKind::Ordered],
        2 => vec![TravKind::SplitAtomic, TravKind::SplitWorkspace],
        3 => vec![TravKind::Walk],
        4 => vec![TravKind::Hybrid],
        _ => Vec::new(),
    };
    let opts = mapal_place::Opts {
        band: vec![("all".into(), 0.0)],
        kinds,
        ..place_opts()
    };
    match (arms.place_tile, inc) {
        (true, Some((i, x))) => {
            let one = Slot { e: &emitters[i], x };
            mapal_place::search(m, s, &[&one], &opts, None, gate)
        }
        (true, None) => None,
        (false, _) => {
            let lows: Vec<&dyn Lowering> = emitters.iter().map(|e| e as &dyn Lowering).collect();
            mapal_place::search(m, s, &lows, &opts, None, gate)
        }
    }
}

/// The order an emission's kernels are drawn under where no order is
/// passed (`plan_pick`'s).
fn arms_order(arms: crate::GpuArms) -> crate::Order {
    match arms.free {
        true => crate::Order::Free,
        false => crate::Order::Sequential,
    }
}

/// **`machine-place` at [`gpu::launch_in`]**: the machine model's pick for
/// `site` under `order`, the rules' launch `rules` its incumbent (kept
/// within the band, and wherever the search finds nothing).
pub(crate) fn placed(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: crate::Order,
    rules: Launch,
) -> Launch {
    if forced_now().is_some() || rules_now() || p.gpu().is_none() {
        return rules;
    }
    // The host launches the kernel the device emits ([`placed_for`]): a
    // pick that cannot store column-major where the site stores so
    // ([`ColScope`]) keeps the rules' tile there, so it does here too
    // (cost-rental: the A100's 1024 x 256 x 256 site launched the pick's
    // 32 x 32 grid over the rules' 64 x 32 kernel — an illegal address, no
    // output).
    search_site(site, p, contract, arms, order, &rules)
        .filter(|x| !stores_col(site, &rules) || gpu::col_store(&x.launch, site))
        .map_or(rules, |x| x.launch)
}

/// **`machine-place` at the kernel of realization `combine`** (every launch
/// passes [`gpu::launch_for`]'s resize): the pick of an order whose
/// realization is `combine`, else the rules' `r` (the ordered kernel beside
/// a combined pick keeps the rules' tile; a column-major store the rules'
/// tile takes stays possible).
pub(crate) fn placed_for(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    combine: Combine,
    r: Launch,
) -> Launch {
    if forced_now().is_some() || rules_now() || p.gpu().is_none() {
        return r;
    }
    let first = ORDER.with(|o| o.get()).unwrap_or_else(|| arms_order(arms));
    let orders = [
        first,
        crate::Order::Free,
        crate::Order::Fixed,
        crate::Order::Sequential,
    ];
    for (i, o) in orders.into_iter().enumerate() {
        let k = key(site, p, contract, arms, o);
        let hit = match PICKS.with(|c| c.borrow().get(&k).cloned()) {
            Some(h) => h,
            // The emission's own order: search it (the rules' launch of
            // that order its incumbent).
            None if i == 0 => {
                let rules = self::rules(|| gpu::launch_in_rules(site, p, contract, arms, o));
                search_site(site, p, contract, arms, o, &rules)
            }
            None => None,
        };
        if let Some(x) = hit
            && x.launch.combine == combine
            && (!stores_col(site, &r) || gpu::col_store(&x.launch, site))
        {
            // A kept pick is the rules' launch, and at the kernel the rules'
            // launch of this realization is `r` (its own resize: the register
            // rule, a walk's full tile) — never the host's copy of it, which
            // differs (cost-rental: the RTX 3090's `decode_1024_256` site; test
            // `placed_host_launch_is_the_device_kernel`).
            return if x.kept { r } else { x.launch };
        }
    }
    r
}

/// The `; mapal.place` line of the kernel of realization `combine` (empty
/// where no search placed it).
pub(crate) fn place_line(
    site: &TileSite,
    p: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    combine: Combine,
) -> String {
    if forced_now().is_some() {
        return String::new();
    }
    for o in [
        arms_order(arms),
        crate::Order::Free,
        crate::Order::Fixed,
        crate::Order::Sequential,
    ] {
        let k = key(site, p, contract, arms, o);
        if let Some(Some(x)) = PICKS.with(|c| c.borrow().get(&k).cloned())
            && x.launch.combine == combine
        {
            return x.line;
        }
    }
    String::new()
}

/// The rules' launch's emitter id.
pub fn lowering_of(l: &Launch) -> &'static str {
    match (l.wgmma, l.mma, l.tiled) {
        (Some(w), _, _) if w.ws => WGMMA_WS,
        (Some(_), _, _) => WGMMA,
        (_, Some(_), _) => MMA,
        (_, _, Some(_)) => FMA,
        _ => NAIVE,
    }
}

/// `DeviceCost` and `Ty` are re-read here only for the evidence listing.
#[allow(dead_code)]
fn _uses(_: DeviceCost, _: Ty) {}

// ---- the coverage probe ----------------------------------------------------------

/// What the probe saw: every launch the rules produced while it was on.
#[derive(Clone, Debug)]
pub(crate) struct Seen {
    pub site: TileSite,
    pub profile: &'static str,
    pub contract: bool,
    pub arms: crate::GpuArms,
    pub launch: Launch,
}

#[derive(Default)]
pub(crate) struct Probe {
    /// Replace every launch by `launch_of(schedule_of(launch))`.
    pub roundtrip: bool,
    pub seen: Vec<Seen>,
}

thread_local! {
    pub(crate) static PROBE: std::cell::RefCell<Option<Probe>> = const { std::cell::RefCell::new(None) };
}

/// The rules' launch `l` of a site, seen by the coverage probe (a test
/// switch; a plain emission never sets one): recorded, and under `roundtrip`
/// replaced by the launch its schedule lowers to.
pub(crate) fn probe(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    l: Launch,
) -> Launch {
    if let Some(f) = forced_now() {
        return f;
    }
    PROBE.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else {
            return l;
        };
        if profile.gpu().is_none() {
            return l;
        }
        p.seen.push(Seen {
            site: site.clone(),
            profile: profile.name,
            contract,
            arms,
            launch: l,
        });
        match p.roundtrip {
            true => launch_of(
                site,
                profile,
                contract,
                arms,
                &schedule_of(site, profile, arms, &l),
            ),
            false => l,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmitOpts, emit_detailed};
    use mapal_place::space;

    /// Every GPU profile (the hand-written parts; `native-gpu` is this box's
    /// RTX 5080, `rtx-5080` here).
    const PROFILES: [&str; 8] = [
        "cuda-ada",
        "rtx-4090",
        "h100",
        "h100-pcie",
        "rtx-5090",
        "rtx-5080",
        "rtx-3090",
        "a100",
    ];

    #[test]
    fn every_gpu_profile_translates_with_provenance() {
        for name in PROFILES {
            let p = crate::profile::resolve(name).expect("a profile");
            let m = machine_of(p).expect("a GPU machine");
            let rows = m.rows();
            assert!(m.units() > 0 && m.group_lanes() == 32, "{name}");
            // Every fit stays out of the machine: perturbing a fitted
            // DeviceCost field leaves the translation unchanged.
            let g = *p.gpu().unwrap();
            for f in FITS {
                let mut over: crate::profile::CostOver = [None; DeviceCost::NAMES.len()];
                let i = DeviceCost::NAMES.iter().position(|n| *n == f).unwrap();
                over[i] = Some(g.rate(f) * 3 + 7);
                let q = TargetProfile {
                    machine: crate::profile::Machine::Gpu(g.with_cost(&over)),
                    ..*p
                };
                assert_eq!(
                    machine_of(&q).unwrap(),
                    m,
                    "{name}: the machine reads the fit {f}"
                );
            }
            // Unlabeled values are visible as nominal, never silent.
            let marked = rows.iter().filter(|r| r.src.marked()).count();
            eprintln!(
                "{name}: {} facts, {} marked (derived / nominal)",
                rows.len(),
                marked
            );
        }
        // The RTX 5080's keep is measured; the A100's and H100's half-L2
        // placeholders are nominal with the interval down to a quarter.
        let keep = |n: &str| {
            let m = machine_of(crate::profile::resolve(n).unwrap()).unwrap();
            m.places[L2].cache().unwrap().keep.cyclic
        };
        assert!(!keep("rtx-5080").marked());
        assert!(keep("a100").marked() && keep("a100").lo == 10 << 20);
        assert!(keep("h100").marked());
        // A part without a measured launch cost prices it as nominal, never
        // as the RTX 5080's measured one.
        let m = machine_of(crate::profile::resolve("rtx-5090").unwrap()).unwrap();
        assert!(m.dispatch.launches[0].ns.marked());
    }

    /// **Every measured row is read** (cost-rental review): on every part,
    /// each `prim` row maps to a machine primitive or operation, and each
    /// efficiency and latency row to an operation — none dropped by its
    /// name (the H100's `wg.m64n128k16.bf16` rows measure the warpgroup
    /// family `wg.m64n256k16.bf16` at `n = 128`).
    #[test]
    fn every_measured_row_is_read() {
        // Measured and read by no machine row: the one-block launch (a
        // grid's launch reads the `.sms` rows).
        const UNREAD: [&str; 2] = ["launch.issue.1", "launch.gap.1"];
        for name in [
            "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
        ] {
            let p = crate::profile::resolve(name).unwrap();
            let g = p.gpu().unwrap();
            let m = machine_of(p).unwrap();
            for &(row, _) in g.place.prim {
                if UNREAD.contains(&row) {
                    continue;
                }
                let (of, _) = prim_of(row, &m.ops)
                    .unwrap_or_else(|| panic!("{name}: the prim row {row} maps to no machine row"));
                match of {
                    PrimOf::Op(o) => {
                        let op = &m.ops[o];
                        assert!(
                            !op.rate.marked() && !op.latency.marked(),
                            "{name}: {row} sets {}'s rate and latency",
                            op.name
                        );
                    }
                    _ => assert!(
                        m.prims.iter().any(|p| p.name == row),
                        "{name}: the prim row {row} is a machine primitive"
                    ),
                }
            }
            for &(row, ..) in g.place.eff {
                assert!(
                    m.ops
                        .iter()
                        .any(|o| measures(row, o.name, o.n) && !o.eff.is_empty()),
                    "{name}: the efficiency row {row} is an operation's"
                );
            }
            for &(row, _) in g.place.latency {
                assert!(
                    m.ops
                        .iter()
                        .any(|o| measures(row, o.name, o.n) && !o.latency.marked()),
                    "{name}: the latency row {row} is an operation's"
                );
            }
        }
        // The H100's warpgroup bf16 family reads its measured member; the
        // register-`a` form, never measured, stays nominal.
        let m = machine_of(crate::profile::resolve("h100").unwrap()).unwrap();
        let op = |n: &str| m.ops.iter().find(|o| o.name == n).expect("the op");
        let wg = op("wg.m64n256k16.bf16");
        assert!(!wg.rate.marked() && !wg.latency.marked() && !wg.eff.is_empty());
        assert!(op("wg.m64n256k16.bf16.rs").latency.marked());
        assert!(!measures("wg.m64n128k16.bf16", "wg.m64n256k16.f16", wg.n));
        // A value from another box of the part keeps its own provenance
        // (the RTX 4090's L2 latency: an earlier box, not the rental run's).
        let m = machine_of(crate::profile::resolve("rtx-4090").unwrap()).unwrap();
        let lat = m
            .reach(L2, mapal_place::ExecKind::Unit)
            .unwrap()
            .latency_ns
            .src;
        assert!(
            matches!(lat, Src::Measured(c) if c.contains("earlier RTX 4090 box")),
            "the 4090's L2 latency cites its own box: {lat:?}"
        );
        assert!(!measures(
            "mma.m16n8k16.bf16",
            "mma.m16n16k16.bf16",
            NSet::One(16)
        ));
    }

    /// **The host launches the kernel the device emits** (cost-rental): on
    /// every GPU part, each placed ordered site of the attention programs
    /// launches the geometry its kernel is emitted with — the A100's
    /// 1024 x 256 x 256 site launched the pick's 32 x 32 grid over the rules'
    /// 64 x 32 kernel (the rules' tile stores column-major, the pick's not).
    #[test]
    fn placed_host_launch_is_the_device_kernel() {
        let mut arms = crate::GpuArms {
            free: true,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        let plain = arms;
        *arms.on.rung("machine-place").expect("the rung") = true;
        let (mut ordered, mut kept) = (0, 0);
        for prog in [
            "pt_decode_1024_256",
            "pt_causal_1024_256",
            "pt_mlp_2048",
            "pt_twin_2048",
        ] {
            let src = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("../../../benches/algorithms/{prog}.mapal")),
            )
            .expect("the program");
            let po = mapal_syntax::parse(&src);
            let ir =
                mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
            for name in [
                "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
            ] {
                let p = crate::profile::resolve(name).unwrap();
                for (site, _) in &warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes()) {
                    let host = gpu::launch_in(site, p, true, arms, crate::Order::Free);
                    // A site the search kept runs the OFF arm's kernel, of
                    // any realization (the H100's kept 2048² walk).
                    let rules = self::rules(|| {
                        gpu::launch_in_rules(site, p, true, arms, crate::Order::Free)
                    });
                    if search_site(site, p, true, arms, crate::Order::Free, &rules)
                        .is_some_and(|x| x.kept)
                    {
                        kept += 1;
                        let mut dev = gpu::launch_for(site, p, true, arms, host.combine);
                        let off = gpu::launch_for(site, p, true, plain, host.combine);
                        dev.rungs.machine_place = false;
                        assert_eq!(
                            dev, off,
                            "{prog} on {name}: a kept site's kernel is the OFF arm's"
                        );
                    }
                    if host.combine != Combine::None {
                        continue;
                    }
                    ordered += 1;
                    let dev = gpu::launch_for(site, p, true, arms, Combine::None);
                    assert_eq!(
                        (host.tiled, host.mma, host.wgmma),
                        (dev.tiled, dev.mma, dev.wgmma),
                        "{prog} on {name}: the launched geometry is the kernel's"
                    );
                }
            }
        }
        assert!(ordered > 0 && kept > 0, "ordered and kept sites placed");
    }

    /// **A fixed-order site launches the kernel the device emits**
    /// (cost-rental, `OrderScope`): on every part, fixed-order sites whose
    /// pick moves the launch off the rules' (a split's `S`, its tile) — the
    /// kernel of the host's realization emitted before the host's search —
    /// launch the device's whole kernel (grid, tile, rungs), never only its
    /// tile (the H100's 16 x 2048 x 7168 split pick launched over the rules'
    /// kernel: wrong values). Without `OrderScope` the kernel is the pick of
    /// the arms' own order (`Sequential`: no split) and this fails.
    #[test]
    fn placed_fixed_order_kernel_is_the_host_launch() {
        let mut fixed = crate::GpuArms {
            free: false,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        *fixed.on.rung("machine-place").expect("the rung") = true;
        let mut moved = 0;
        for (m, n, k) in [
            (16, 2048, 7168),
            (16, 18432, 7168),
            (128, 4096, 4096),
            (128, 6144, 4096),
        ] {
            let site = crate::place_replay::replay_site(m, n, k, "f32", true);
            for name in [
                "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
            ] {
                let p = crate::profile::resolve(name).unwrap();
                let at = format!("{m} x {n} x {k} on {name}");
                let _order = OrderScope::new(crate::Order::Fixed);
                // The host's launch, searched alone.
                PICKS.with(|c| c.borrow_mut().clear());
                let host = gpu::launch_in(&site, p, false, fixed, crate::Order::Fixed);
                let rules = self::rules(|| {
                    gpu::launch_in_rules(&site, p, false, fixed, crate::Order::Fixed)
                });
                moved += usize::from(host != rules);
                // The kernel of its realization emitted first, then the
                // host's launch.
                PICKS.with(|c| c.borrow_mut().clear());
                let dev = gpu::launch_for(&site, p, false, fixed, host.combine);
                let again = gpu::launch_in(&site, p, false, fixed, crate::Order::Fixed);
                assert_eq!(again, host, "{at}: one launch whichever comes first");
                // The device's launch is its realization's kernel (its
                // `combine` field the host's to set).
                assert_eq!(
                    Launch {
                        combine: host.combine,
                        ..dev
                    },
                    host,
                    "{at}: the host launches the kernel the device emits"
                );
            }
        }
        assert!(moved > 0, "the search moves some fixed-order launch");
    }

    /// **The warpgroup lowering's declared constraint** (cost-rental): a
    /// K-major-only instruction (TF32) takes the transposed product.
    #[test]
    fn tf32_wgmma_direct_product_refused() {
        // cost-rental: on the H100 the search's direct TF32 warpgroup tile
        // was invalid PTX (`wgmma.mma_async` arguments mismatch): a
        // K-major-only instruction reads a lanes-major `b` only through the
        // transposed product. The rules' tile stays admitted; the same tile
        // made direct is refused.
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../benches/algorithms/pt_mlp_2048_tf32.mapal"),
        )
        .expect("the TF32 program");
        let po = mapal_syntax::parse(&src);
        let ir = mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
        let p = crate::profile::resolve("h100").unwrap();
        let m = machine_of(p).unwrap();
        let arms = crate::GpuArms {
            free: true,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        let sites = warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes());
        let mut seen = 0;
        for (site, _) in &sites {
            let rules = gpu::launch_in_rules(site, p, true, arms, crate::Order::Free);
            let Some(w) = rules.wgmma else { continue };
            if w.fmt != crate::profile::MmaFmt::Tf32 {
                continue;
            }
            seen += 1;
            assert!(!w.direct, "the rules' TF32 tile is the transposed product");
            let e = Emitter {
                id: if w.ws { WGMMA_WS } else { WGMMA },
                site,
                profile: p,
                contract: true,
                arms,
            };
            let s = site_of(site, true, crate::Order::Free);
            let at = |w| {
                let x = schedule_of(
                    site,
                    p,
                    arms,
                    &Launch {
                        wgmma: Some(w),
                        ..rules
                    },
                );
                e.declare(&m, &s, &x)
            };
            assert!(at(w).is_ok(), "the rules' tile admitted");
            assert!(
                at(crate::profile::GpuWgmmaTile { direct: true, ..w }).is_err(),
                "a direct TF32 tile refused"
            );
        }
        assert!(
            seen > 0,
            "the program has a TF32 warpgroup site on the H100"
        );
    }

    /// **The vector-cells kernel's declared constraint** (cost-model ABBA):
    /// on the 2048 x 2048 x 25 window program's own tile sites (mixed-radix
    /// `b`), the one-thread-per-cell lowering refuses more than one cell a
    /// thread — the search's 16-cell pick stored wrong values on the RTX
    /// 5080 — and keeps admitting them on a plain GEMV row.
    #[test]
    fn naive_vector_cells_refuse_a_window_site() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../benches/algorithms/pt_conv_2048_5.mapal"),
        )
        .expect("the window program");
        let po = mapal_syntax::parse(&src);
        let ir = mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
        let p = crate::profile::resolve("rtx-5080").unwrap();
        let m = machine_of(p).unwrap();
        let arms = crate::GpuArms {
            free: true,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        let sites = warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes());
        let mut windows = 0;
        for (site, _) in &sites {
            let rules = gpu::launch_in_rules(site, p, true, arms, crate::Order::Free);
            if rules.tiled.is_some() || rules.mma.is_some() || rules.wgmma.is_some() {
                continue;
            }
            let e = Emitter {
                id: NAIVE,
                site,
                profile: p,
                contract: true,
                arms,
            };
            let s = site_of(site, true, crate::Order::Free);
            let at = |cells| {
                let x = schedule_of(site, p, arms, &Launch { cells, ..rules });
                e.declare(&m, &s, &x)
            };
            assert!(at(1).is_ok(), "one cell a thread stays admitted");
            if !gpu::naive_cells_admit(site, 16) {
                windows += 1;
                assert!(at(16).is_err(), "16 cells on a window site refused");
            }
        }
        assert!(windows > 0, "the program has a window site");
        let row = crate::place_replay::replay_site(1, 4096, 2048, "f32", true);
        assert!(
            gpu::naive_cells_admit(&row, 4),
            "a GEMV row admits vector cells"
        );
    }

    /// **ADR-0050 review: the one-thread-per-cell kernel's declared
    /// registers cover what `ptxas` allocates** at every cell count the
    /// search may give a thread (1–16, f32 and bf16 `b`), so the footprint's
    /// occupancy is never above the kernel's. `PLACE_NAIVE_REGS=1` prints
    /// the table; without `llc` / `ptxas`, nothing to check.
    #[test]
    fn naive_declared_registers_cover_ptxas() {
        use std::process::Command;
        let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
        if !have("llc") || !have("ptxas") {
            return;
        }
        let arms = crate::GpuArms {
            free: true,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        let mut over = Vec::new();
        for part in ["a100", "rtx-3090", "rtx-4090", "h100", "rtx-5080"] {
            let p = crate::profile::resolve(part).unwrap();
            let g = p.gpu().unwrap();
            let m = machine_of(p).unwrap();
            for fmt in ["f32", "bf16"] {
                let site = crate::place_replay::replay_site(1, 4096, 2048, fmt, true);
                let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
                assert!(
                    rules.tiled.is_none() && rules.mma.is_none(),
                    "a cell kernel"
                );
                for cells in [1, 2, 4, 8, 16] {
                    let l = Launch { cells, ..rules };
                    let x = schedule_of(&site, p, arms, &l);
                    let e = Emitter {
                        id: NAIVE,
                        site: &site,
                        profile: p,
                        contract: true,
                        arms,
                    };
                    let s = site_of(&site, true, crate::Order::Free);
                    let d = e.declare(&m, &s, &x).expect("declared");
                    let text = gpu::lowered_text(&site, p, true, arms, &l).expect("lowers");
                    let dir = std::env::temp_dir().join(format!(
                        "mapal-naive-regs-{}-{fmt}-{cells}",
                        std::process::id()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    let (ll, ptx) = (dir.join("k.ll"), dir.join("k.ptx"));
                    std::fs::write(&ll, gpu::kernel_module(&text)).unwrap();
                    let mut llc = Command::new("llc");
                    llc.args(["-O3", &format!("-mcpu={}", g.arch), "-nvptx-fma-level=0"]);
                    if let Some(v) = crate::Ptx::declared(&gpu::kernel_module(&text)) {
                        llc.arg(format!("-mattr={}", v.mattr()));
                    }
                    assert!(llc.arg(&ll).arg("-o").arg(&ptx).status().unwrap().success());
                    let out = Command::new("ptxas")
                        .args([&format!("-arch={}", g.arch), "-v"])
                        .arg(&ptx)
                        .arg("-o")
                        .arg(dir.join("k.cubin"))
                        .output()
                        .unwrap();
                    let _ = std::fs::remove_dir_all(&dir);
                    let log = String::from_utf8_lossy(&out.stderr).to_string();
                    let regs: u64 = log
                        .split("Used ")
                        .nth(1)
                        .and_then(|r| r.split(' ').next())
                        .and_then(|r| r.parse().ok())
                        .expect("a register count");
                    if std::env::var("PLACE_NAIVE_REGS").is_ok() {
                        eprintln!(
                            "naive {part} {fmt} cells={cells}: ptxas {regs}, declared {}",
                            d.regs
                        );
                    }
                    if regs > d.regs {
                        over.push(format!(
                            "{part} {fmt} cells={cells}: ptxas {regs}, declared {}",
                            d.regs
                        ));
                    }
                }
            }
        }
        assert!(over.is_empty(), "registers above the declared: {over:?}");
    }

    /// **ADR-0050 review: the native GPU's machine is the hand profile's** —
    /// the device query and the calibration line (its `place.` facts
    /// included) translate to the RTX 5080's machine: every fact's value and
    /// interval, and whether it is marked (what the minimax reads), equal;
    /// only the cites' wording differs. Without a calibration line the start
    /// timing and the clock fall back, marked.
    #[test]
    fn native_gpu_translates_to_the_hand_profiles_machine() {
        let q = crate::profile::DeviceQuery {
            cc: 120,
            sms: 84,
            warp: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 24,
            regs_per_sm: 65536,
            smem_per_sm: 102400,
            smem_per_block: 101_376,
            smem_reserved_per_block: 1024,
            l2_bytes: 67108864,
            clock_khz: 2_640_000,
            mem_clock_khz: 15_001_000,
            mem_bus_bits: 256,
        };
        let calib = crate::profile::COMMITTED_CALIB[0];
        let native = q.profile(Some(calib));
        let hand = crate::profile::resolve("rtx-5080").unwrap();
        let (a, b) = (machine_of(&native).unwrap(), machine_of(hand).unwrap());
        let rows = |m: &Machine| {
            m.rows()
                .into_iter()
                .map(|r| (r.name, r.v, r.lo, r.hi, r.src.marked()))
                .collect::<Vec<_>>()
        };
        let (ra, rb) = (rows(&a), rows(&b));
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(x, y, "native-gpu against rtx-5080");
        }
        assert_eq!(a.dispatch.start, b.dispatch.start);
        assert_eq!(a.dispatch.start_src.marked(), b.dispatch.start_src.marked());
        // No calibration line: the probes' facts are absent, and visibly so
        // (marked), never silently the hand profile's.
        let none = q.profile(None);
        let c = machine_of(&none).unwrap();
        assert!(c.dispatch.start_src.marked() && c.dispatch.clock_khz.marked());
    }

    /// **ADR-0050 C4: the search on every GPU profile** — f32, TF32 and
    /// bf16 GEMM sites of several shapes: a pick on each (or the rules' kept
    /// within the band), every candidate it lowers through the gate, its
    /// `launch_of` a launch the emitter's text takes; the walk's stats
    /// printed (`PLACE_SEARCH=1`).
    #[test]
    fn the_search_places_gemm_sites_on_every_profile() {
        let shapes: &[(u64, u64, u64, &str)] = &[
            (128, 4096, 4096, "f32"),
            (1024, 1024, 1024, "tf32"),
            (512, 2048, 2048, "bf16"),
            (2048, 4096, 4096, "bf16"),
        ];
        // A debug build places two parts' two shapes; `--release` all.
        let debug = cfg!(debug_assertions);
        for name in PROFILES
            .into_iter()
            .filter(|n| !debug || ["rtx-5080", "h100"].contains(n))
        {
            let p = crate::profile::resolve(name).unwrap();
            for &(m, n, k, fmt) in shapes.iter().filter(|s| !debug || s.0 <= 512) {
                let site = crate::place_replay::replay_site(m, n, k, fmt, false);
                let arms = crate::GpuArms {
                    free: true,
                    site_cands: 1,
                    ..crate::GpuArms::default()
                };
                let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
                let t = std::time::Instant::now();
                let placed = search_site(&site, p, true, arms, crate::Order::Free, &rules)
                    .expect("a placement");
                let dt = t.elapsed();
                let text = gpu::lowered_text(&site, p, true, arms, &placed.launch);
                assert!(text.is_ok(), "{name} {m}x{n}x{k} {fmt}: {}", placed.plan);
                if std::env::var("PLACE_SEARCH").is_ok() {
                    eprintln!(
                        "{name} {m}x{n}x{k} {fmt} {dt:.2?} kept={} {}",
                        placed.kept,
                        placed.line.trim_end()
                    );
                }
            }
        }
    }

    /// The search's walk per lowering on one site (`PLACE_SEARCH=1`).
    #[test]
    fn search_cost_per_lowering() {
        if std::env::var("PLACE_SEARCH").is_err() {
            return;
        }
        let p = crate::profile::resolve("rtx-5080").unwrap();
        for (m, n, k, fmt) in [(128, 4096, 4096, "f32"), (512, 2048, 2048, "bf16")] {
            let site = crate::place_replay::replay_site(m, n, k, fmt, false);
            let arms = crate::GpuArms {
                free: true,
                site_cands: 1,
                ..crate::GpuArms::default()
            };
            let mach = machine_of(p).unwrap();
            let s = site_of(&site, true, crate::Order::Free);
            let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
            let x = schedule_of(&site, p, arms, &rules);
            for e in Emitter::all(&site, p, true, arms) {
                let t = std::time::Instant::now();
                let inc = (e.id == lowering_of(&rules)).then(|| (0, x.clone()));
                SEARCHING.with(|c| c.set(true));
                LSU.with(|c| c.borrow_mut().clear());
                LAYOUT.with(|c| c.borrow_mut().clear());
                let r =
                    mapal_place::search(&mach, &s, &[&e], &place_opts(), inc, &mut |_, _| Ok(()));
                SEARCHING.with(|c| c.set(false));
                eprintln!(
                    "{m}x{n}x{k} {fmt} {}: {:.2?} {:?} {}",
                    e.id,
                    t.elapsed(),
                    r.as_ref().map(|r| r.stats.clone()),
                    r.map_or("-".into(), |r| format!(
                        "{} {:.0}",
                        r.pick.x.plan(),
                        r.pick.value()
                    ))
                );
            }
        }
    }

    /// The suite's programs: every committed algorithm and ladder program.
    fn suite() -> Vec<(String, String)> {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../benches/algorithms");
        let mut out = Vec::new();
        for dir in [root.clone(), root.join("ladder")] {
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .expect("benches/algorithms")
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "mapal"))
                .collect();
            files.sort();
            for f in files {
                let src = std::fs::read_to_string(&f).expect("a program");
                out.push((f.file_stem().unwrap().to_string_lossy().into_owned(), src));
            }
        }
        out
    }

    fn lower(src: &str) -> Option<mapal_ir::CategoryIr> {
        let po = mapal_syntax::parse(src);
        if !po.diagnostics.is_empty() {
            return None;
        }
        let ir = mapal_lower::lower(src, &po.program).ok()?;
        Some(mapal_rewrite::rewrite(ir).ir)
    }

    /// The faces the suite emits a program on: the default face, and the
    /// matrix-product cells on the four others (`gpu_bundle.sh`).
    fn faces(stem: &str) -> Vec<Vec<&'static str>> {
        let mut f = vec![vec![]];
        if stem.starts_with("mm_") {
            f.extend([
                vec!["--exact"],
                vec!["--order=sequential"],
                vec!["--no-contract", "--order=fixed"],
                vec!["--order=fixed"],
            ]);
        }
        f
    }

    /// The order a face's flags emit under (`flags::apply`: `--exact` and
    /// `--order=sequential` sequential, `--order=fixed` fixed, else free).
    fn face_order(face: &[&str]) -> crate::Order {
        let mut opts = EmitOpts::default();
        for f in face {
            assert!(crate::flags::apply(&mut opts, f).expect("a flag"), "{f}");
        }
        opts.order
    }

    fn emit(
        ir: &mapal_ir::CategoryIr,
        target: &'static str,
        face: &[&str],
        probe: Option<bool>,
    ) -> (Option<String>, Vec<Seen>) {
        let mut opts = EmitOpts {
            host: "generic",
            target,
            ..EmitOpts::default()
        };
        for f in face {
            assert!(crate::flags::apply(&mut opts, f).expect("a flag"), "{f}");
        }
        PROBE.with(|p| {
            *p.borrow_mut() = probe.map(|roundtrip| Probe {
                roundtrip,
                seen: Vec::new(),
            })
        });
        let text = emit_detailed(ir, &opts)
            .ok()
            .map(|e| e.text + "\n;;device\n" + &e.device.unwrap_or_default());
        let seen = PROBE.with(|p| p.borrow_mut().take().map(|p| p.seen).unwrap_or_default());
        (text, seen)
    }

    /// **ADR-0050 C2 coverage: every launch the rules make for a tile site
    /// of the suite, on every GPU profile, is a point of the candidate space
    /// and lowers back to itself** — `schedule_of` the launch is inside its
    /// lowering's coordinate domains with every declared constraint met and
    /// an occupancy within its footprint's, its traversal is one the
    /// traversal generator gives (persistent grids aside: out of step 1's
    /// space), `launch_of` that schedule is the launch, and the emission
    /// with every launch replaced by its schedule's is byte-identical.
    #[test]
    fn rules_launches_are_points_of_the_space() {
        // A debug build emits every eighth program (and every cell); a
        // release build (`--release`) the whole suite.
        let progs: Vec<_> = suite()
            .into_iter()
            .enumerate()
            .filter(|(i, (stem, _))| {
                !cfg!(debug_assertions) || i % 8 == 0 || stem.starts_with("mm_")
            })
            .map(|(_, p)| p)
            .collect();
        let jobs = std::env::var("MAPAL_TESTGEN_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12usize);
        let chunks: Vec<Vec<(String, String)>> = {
            let mut c = vec![Vec::new(); jobs];
            for (i, p) in progs.into_iter().enumerate() {
                c[i % jobs].push(p);
            }
            c
        };
        type Report = (
            u64,
            u64,
            Vec<String>,
            std::collections::BTreeMap<String, u64>,
            Vec<String>,
        );
        let results: Vec<Report> = std::thread::scope(|sc| {
            let hs: Vec<_> = chunks
                .iter()
                .map(|chunk| {
                    sc.spawn(move || {
                        let mut launches = 0u64;
                        let mut emissions = 0u64;
                        let mut fails = Vec::new();
                        let mut d6: std::collections::BTreeMap<String, u64> = Default::default();
                        let mut persist = Vec::new();
                        for (stem, src) in chunk {
                            let Some(ir) = lower(src) else { continue };
                            for target in PROFILES {
                                for face in faces(stem) {
                                    let (text, seen) = emit(&ir, target, &face, Some(false));
                                    let Some(text) = text else { continue };
                                    emissions += 1;
                                    let mut dedup = std::collections::HashSet::new();
                                    for s in &seen {
                                        let key =
                                            format!("{:?}{:?}{}", s.site, s.launch, s.contract);
                                        if !dedup.insert(key) {
                                            continue;
                                        }
                                        launches += 1;
                                        let p = crate::profile::resolve(s.profile).unwrap();
                                        let m = machine_of(p).unwrap();
                                        let x = schedule_of(&s.site, p, s.arms, &s.launch);
                                        for (n, v) in &x.params {
                                            if n.starts_with("rung") {
                                                *d6.entry(format!(
                                                    "{n}{}",
                                                    crate::GpuRungs::NAMES[*v as usize]
                                                ))
                                                .or_default() += 1;
                                            }
                                        }
                                        let e = Emitter {
                                            id: lowering_of(&s.launch),
                                            site: &s.site,
                                            profile: p,
                                            contract: s.contract,
                                            arms: s.arms,
                                        };
                                        let tag = format!(
                                            "{stem}{} {target} {}x{}x{} {}",
                                            face.join(""),
                                            s.site.rows,
                                            s.site.c,
                                            s.site.k,
                                            x.plan()
                                        );
                                        let back = launch_of(&s.site, p, s.contract, s.arms, &x);
                                        if back != s.launch {
                                            fails.push(format!("{tag}: launch_of differs"));
                                            continue;
                                        }
                                        let d = match e.contains(&m, &x) {
                                            Ok(d) => d,
                                            Err(why) => {
                                                fails.push(format!("{tag}: {why}"));
                                                continue;
                                            }
                                        };
                                        let occ = m.occupancy(d.threads, d.regs, &[(SMEM, d.smem)]);
                                        if x.per_unit > occ {
                                            *d6.entry(
                                                "residency above the footprint's (rules' metadata)"
                                                    .into(),
                                            )
                                            .or_default() += 1;
                                        }
                                        if let Traversal::Persistent { .. } = x.trav {
                                            persist.push(tag);
                                            continue;
                                        }
                                        // The fold's freedom is the
                                        // emission's face (its order), never
                                        // read off the launch checked.
                                        let order = face_order(&face);
                                        let site = site_of(&s.site, s.contract, order);
                                        let ts =
                                            space::traversals(&m, &site, &x, &d, e.traversals());
                                        if !ts.contains(&x.trav) {
                                            fails.push(format!(
                                                "{tag}: traversal outside the generator"
                                            ));
                                        }
                                    }
                                    // Byte-identical: every launch replaced
                                    // by its schedule's.
                                    let (rt, _) = emit(&ir, target, &face, Some(true));
                                    if rt.as_deref() != Some(text.as_str()) {
                                        fails.push(format!(
                                            "{stem}{} {target}: round-trip text differs",
                                            face.join("")
                                        ));
                                    }
                                }
                            }
                        }
                        (launches, emissions, fails, d6, persist)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("a worker"))
                .collect()
        });
        let (mut launches, mut emissions) = (0, 0);
        let mut fails = Vec::new();
        let mut d6: std::collections::BTreeMap<String, u64> = Default::default();
        let mut persist = Vec::new();
        for (l, e, f, d, p) in results {
            launches += l;
            emissions += e;
            fails.extend(f);
            for (k, v) in d {
                *d6.entry(k).or_default() += v;
            }
            persist.extend(p);
        }
        fails.sort();
        eprintln!(
            "coverage: {emissions} emissions, {launches} distinct rules' launches, {} outside, {} persistent (out of step 1)",
            fails.len(),
            persist.len()
        );
        eprintln!("D6 rungs a launch carries beyond its site's table: {d6:?}");
        // The reasons, by class (digits dropped), with an example each.
        let mut why: std::collections::BTreeMap<String, (u64, String)> = Default::default();
        for f in &fails {
            let r = f.rsplit(": ").next().unwrap_or(f);
            let class: String = r
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .collect();
            let e = why.entry(class).or_insert((0, f.clone()));
            e.0 += 1;
        }
        let mut why: Vec<_> = why.into_iter().collect();
        why.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (class, (n, ex)) in why.iter().take(40) {
            eprintln!("  {n}\t{class}\n      e.g. {ex}");
        }
        assert!(
            fails.is_empty(),
            "{} launches outside the space",
            fails.len()
        );
    }
}
