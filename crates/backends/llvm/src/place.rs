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
    CHAIN_CYCLES_RTX_5080, DeviceCost, Face, GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile,
    Home as MHome, MatrixOp, MmaFmt, NAIVE_UNROLL, NO_PLAN, TargetProfile, WARPGROUP_WARPS,
    WGMMA_N_MAX, WGMMA_N_STEP, fma_held_regs,
};
use mapal_ir::{TileSite, Ty};
use mapal_place::{
    Atomics, Banking, Bind, Cache, Class, Comb, Declared, Dispatch, EffPoint, Exec, ExecKind, Fact,
    Fmt, Fold, Home, InstanceMap, Keep, LaunchCost, LaunchMode, Layout, Level, Lowering, Machine,
    Major, Mgmt, Mover, MoverKind, NSet, Op, Pipe, Place, Policy, Reach, Refusal, Res, Resource,
    Schedule, Site, Src, Start, TravKind, Traversal,
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
/// listed, else a device-query attribute, else unlabeled (nominal).
fn src(g: &Gpu, name: &str) -> Src {
    if let Some((_, s)) = g.place.prov.iter().find(|(n, _)| *n == name) {
        return *s;
    }
    match name {
        "warp"
        | "max_threads_per_block"
        | "sms"
        | "regs_per_thread"
        | "regs_per_sm"
        | "smem_per_sm"
        | "smem_per_block"
        | "smem_reserved_per_block"
        | "max_threads_per_sm"
        | "max_blocks_per_sm"
        | "l2_bytes"
        | "mem_clock_khz"
        | "mem_bus_bits"
        | "schedulers" => Src::Queried,
        _ => Src::Nominal("unlabeled profile value"),
    }
}

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

fn qu(v: u64) -> Fact<u64> {
    Fact::queried(v)
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

/// Measured efficiency points of operation `name` (the part's probes).
fn eff_points(g: &Gpu, name: &str) -> Vec<EffPoint> {
    g.place
        .eff
        .iter()
        .filter(|(n, ..)| *n == name)
        .map(|&(_, chains, groups, pm)| EffPoint {
            chains: chains as f64,
            groups: groups as f64 / 1000.0,
            eff: Fact::measured(pm as f64 / 1000.0, g.place.cite),
        })
        .collect()
}

fn latency(g: &Gpu, name: &str, nominal: (f64, f64, f64)) -> Fact<f64> {
    match g.place.latency.iter().find(|(n, _)| *n == name) {
        Some(&(_, x10)) => Fact::measured(x10 as f64 / 10.0, g.place.cite),
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
    let ghz = match g.place.clock_khz {
        0 => g.clock_khz,
        v => v,
    } as f64
        / 1e6;
    let exec = vec![
        Exec {
            name: "device",
            kind: ExecKind::Device,
            parent: None,
            per_parent: qu(1),
        },
        Exec {
            name: "sm",
            kind: ExecKind::Unit,
            parent: Some(E_DEVICE),
            per_parent: qu(g.sms),
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
            per_parent: qu(g.max_threads_per_sm / g.warp.max(1) / g.schedulers()),
        },
        Exec {
            name: "lane",
            kind: ExecKind::Lane,
            parent: Some(E_GROUP),
            per_parent: qu(g.warp),
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
        v => Fact::measured(v as f64, g.place.cite),
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
            s => Fact::point(v, s),
        },
    };
    let keep_of = |v: u64| match v {
        0 => None,
        v => Some(Fact::measured(v, g.place.cite)),
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
            capacity: qu(g.regs_per_thread * 4),
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
            capacity: qu(g.smem_per_sm),
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
            capacity: qu(g.l2_bytes),
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
            cap: qu(g.regs_per_sm),
            per_slot: 0,
            per_lane: g.regs_per_thread,
            granule: 256,
            reserve: 0,
        },
        Resource {
            what: Res::Threads,
            cap: qu(g.max_threads_per_sm),
            per_slot: g.max_threads_per_block,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Slots,
            cap: qu(g.max_blocks_per_sm),
            per_slot: 0,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Bytes(SMEM),
            cap: qu(g.smem_per_sm),
            per_slot: g.smem_per_block,
            per_lane: 0,
            granule: 128,
            reserve: g.smem_reserved_per_block,
        },
    ];
    // Issue pipes: warp-instructions an SM issues a clock by class.
    let lanes = fact(g, "fp32_lanes", g.fp32_lanes as f64, (1.0, 1.0));
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
        latency: latency(g, name, (4.0, 4.0, 6.0)),
        async_depth: 0,
        rate,
        eff: eff_points(g, name),
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
        let numerics = |v: i32| match src(g, "acc_u") {
            Src::Measured(c) => Fact::measured(v, c),
            _ => Fact::derived(v, v, v + 1, "the RTX 5080's measured accumulator datapath"),
        };
        ops.push(Op {
            name,
            exec: if o.shape.threads > 1 { E_GROUP } else { E_LANE },
            lanes: o.shape.threads,
            m: o.shape.m,
            n: if o.shape.threads > 32 || o.shape.threads == 1 {
                NSet::Step {
                    step: WGMMA_N_STEP,
                    max: o.shape.n.min(WGMMA_N_MAX),
                }
            } else {
                NSet::One(o.shape.n)
            },
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
            latency: latency(g, name, (32.0, 16.0, 64.0)),
            async_depth: u64::from(o.shape.threads > 32),
            rate,
            eff: eff_points(g, name),
            widths: Box::leak(vec![o.shape.threads].into_boxed_slice()),
            acc_u: numerics(o.acc_u),
            acc_g: match src(g, "acc_u") {
                Src::Measured(c) => Fact::measured(o.acc_g, c),
                _ => Fact::derived(o.acc_g, 1, o.acc_g, "the RTX 5080's measured guard bits"),
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
    // Dispatch: launch costs measured where given; else the RTX 5080's
    // cycles at this part's clock — another part's number, so nominal.
    let chain = |measured: u64, name: &str, i: usize| match measured {
        0 => {
            let v = CHAIN_CYCLES_RTX_5080[i] as f64 / ghz.max(0.1);
            Fact::nominal(
                v,
                0.5 * v,
                3.0 * v,
                "unmeasured: the RTX 5080's cycles at this clock",
            )
        }
        v => fact(g, name, v as f64, (0.8, 1.3)),
    };
    let plain = chain(d.dep_launch_ns, "dep_launch_ns", 1);
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
            ns: chain(d.coop_launch_ns, "coop_launch_ns", 3),
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
        warm: [false, false],
    }
}

// ---- schedules and launches ---------------------------------------------------

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
        Combine::StreamK { ctas, iters } => Traversal::Walk { h, ctas, iters },
    }
}

fn combine_of(t: Traversal) -> Combine {
    match t {
        Traversal::Ordered { .. } | Traversal::Persistent { .. } | Traversal::Hybrid { .. } => {
            Combine::None
        }
        Traversal::Split {
            s,
            comb: Comb::Atomic,
            ..
        } => Combine::Atomic { s },
        Traversal::Split { s, .. } => Combine::Workspace { s },
        Traversal::Walk { ctas, iters, .. } => Combine::StreamK { ctas, iters },
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
    let mut rungs = table_rungs(site, profile, arms, arithmetic);
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
        combine: combine_of(x.trav),
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
        let emitted = launch_of(self.site, self.profile, self.contract, self.arms, x)
            .rungs
            .min_blocks;
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
        match self.id {
            // `wgmma`: no combine at all (a declared constraint).
            WGMMA | WGMMA_WS => &[TravKind::Ordered],
            _ => &[
                TravKind::Ordered,
                TravKind::SplitAtomic,
                TravKind::SplitWorkspace,
                TravKind::Walk,
            ],
        }
    }

    fn ops(&self, _m: &Machine, _s: &Site) -> Vec<(usize, bool)> {
        self.domain(0, &[])
            .into_iter()
            .flat_map(|o| match self.id {
                WGMMA | WGMMA_WS => vec![(o as usize, false), (o as usize, true)],
                _ => vec![(o as usize, false)],
            })
            .collect()
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
                // A thread folds its cells over `k`: NAIVE_UNROLL loads of
                // `b` (one vector) and one of `a` a trip in flight.
                let trip = NAIVE_UNROLL as f64;
                Ok(Declared {
                    threads: NAIVE_THREADS,
                    // The counted address and loop budget (FMA_ADDR_REGS)
                    // holds a cell's accumulators and its unrolled loads:
                    // unverified until the post-lowering gate (C4).
                    regs: crate::profile::FMA_ADDR_REGS,
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
                let lsu = gpu::lsu_per_panel(site, &t, &l.rungs, false) as f64;
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
                })
            }
            MMA => {
                let mt = l.mma.ok_or_else(|| Refusal::new("no warp mma tile"))?;
                let o = g.caps.matrix[x.op - 2];
                if mt.stages < 1 || mt.stages > s.k / bk {
                    return Err(Refusal::new("stages in [1, k/bk]"));
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
                })
            }
        }
    }
}

impl Emitter<'_> {
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
                                        let order = if s.arms.free {
                                            crate::Order::Free
                                        } else {
                                            crate::Order::Sequential
                                        };
                                        let site = Site {
                                            fold: match (s.arms.free, x.trav) {
                                                (true, _) => Fold::Free,
                                                // A non-free emission's split
                                                // is the fixed face's count.
                                                (false, Traversal::Split { .. }) => Fold::Fixed,
                                                _ => Fold::Sequential,
                                            },
                                            ..site_of(&s.site, s.contract, order)
                                        };
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
