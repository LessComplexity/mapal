//! The toy machine and lowering the property tests share.
#![allow(dead_code)]

use mapal_place::*;

pub const REGS: usize = 0;
pub const SMEM: usize = 1;
pub const L2: usize = 2;

/// A machine with every rate a parameter (the scales `r`).
pub fn machine(r: &[f64; 9]) -> Machine {
    let q = |v: u64| Fact::queried(v);
    let m = |v: f64| Fact::measured(v, "toy");
    let exec = vec![
        Exec {
            name: "device",
            kind: ExecKind::Device,
            parent: None,
            per_parent: q(1),
        },
        Exec {
            name: "unit",
            kind: ExecKind::Unit,
            parent: Some(0),
            per_parent: q(16),
        },
        Exec {
            name: "sub",
            kind: ExecKind::Sub,
            parent: Some(1),
            per_parent: q(4),
        },
        Exec {
            name: "group",
            kind: ExecKind::Group,
            parent: Some(2),
            per_parent: q(8),
        },
        Exec {
            name: "lane",
            kind: ExecKind::Lane,
            parent: Some(3),
            per_parent: q(32),
        },
    ];
    let place = |name, exec, cap, mgmt, reach| Place {
        name,
        exec,
        capacity: q(cap),
        mgmt,
        banking: None,
        granule: 128,
        reserve: 0,
        per_slot: 0,
        backing: None,
        reach,
    };
    let edge = |from, bw: f64, total: f64, lat: f64| Reach {
        from,
        bytes_per_ns: m(bw),
        total_bytes_per_ns: m(total),
        latency_ns: m(lat),
        granule: 32,
    };
    let places = vec![
        place("regs", 4, 1020, Mgmt::Explicit, vec![]),
        place("smem", 1, 100 << 10, Mgmt::Explicit, vec![]),
        place(
            "l2",
            0,
            8 << 20,
            Mgmt::Cache(Box::new(Cache {
                instances: q(1),
                map: InstanceMap::One,
                replicate: false,
                line: 128,
                keep: Keep {
                    cyclic: Fact::measured((6.0 * r[6]) as u64 * (1 << 20), "toy"),
                    rotated: None,
                    strided: None,
                    curve: vec![],
                },
                policies: vec![],
            })),
            vec![
                edge(0, 0.0, 2000.0 * r[0], 300.0 / r[1]),
                edge(1, 80.0 * r[0], 0.0, 300.0 / r[1]),
            ],
        ),
        place(
            "mem",
            0,
            1 << 34,
            Mgmt::Explicit,
            vec![edge(0, 0.0, 600.0 * r[2], 500.0 / r[3])],
        ),
    ];
    let res = vec![
        Resource {
            what: Res::Regs,
            cap: q(65536),
            per_slot: 0,
            per_lane: 255,
            granule: 256,
            reserve: 0,
        },
        Resource {
            what: Res::Threads,
            cap: q(1536),
            per_slot: 1024,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Slots,
            cap: q(16),
            per_slot: 0,
            per_lane: 0,
            granule: 1,
            reserve: 0,
        },
        Resource {
            what: Res::Bytes(SMEM),
            cap: q(100 << 10),
            per_slot: 99 << 10,
            per_lane: 0,
            granule: 128,
            reserve: 1024,
        },
    ];
    let pipes = vec![
        Pipe {
            class: Class::Math,
            per_clk: m(4.0 * r[4]),
        },
        Pipe {
            class: Class::Shared,
            per_clk: m(1.0 * r[4]),
        },
        Pipe {
            class: Class::Global,
            per_clk: m(0.5 * r[4]),
        },
    ];
    let ops = vec![Op {
        name: "ffma",
        exec: 4,
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
        latency: m(4.0),
        async_depth: 0,
        rate: m(256.0 * r[5]),
        eff: vec![],
        widths: &[1],
        acc_u: Fact::declared(-24, "toy"),
        acc_g: Fact::declared(0, "toy"),
    }];
    Machine {
        label: "toy".into(),
        exec,
        places,
        groups: vec![],
        res,
        pipes,
        ops,
        movers: vec![],
        dispatch: Dispatch {
            unit: 1,
            start: Start::Lockstep,
            start_src: Src::Measured("toy"),
            round_robin: true,
            launches: vec![LaunchCost {
                mode: LaunchMode::Plain,
                ns: m(2000.0 / r[7]),
                per_slot_ps: m(0.0),
            }],
            issue_ns: m(2000.0),
            gap_ns: m(300.0),
            atomics: vec![Atomics {
                at: L2,
                fmt: Fmt::F32,
                ops_per_ns: m(400.0 * r[8]),
            }],
            zero_ns: m(800.0 / r[8]),
            zero_bytes_per_ns: m(2000.0 * r[8]),
            clock_khz: m(2_000_000.0),
            barrier_clk: m(45.0),
            grid_sync_ns: None,
            rr_jitter: None,
        },
        prims: vec![],
    }
}

/// A lane tile `tm × tn`, a slot of `ty × tx` lanes, `bk`, `stages`.
pub fn schedule(
    tm: u64,
    tn: u64,
    tx: u64,
    ty: u64,
    bk: u64,
    stages: u64,
    trav: Traversal,
) -> Schedule {
    let home = |depth| Home {
        place: SMEM,
        layout: Layout::K,
        depth,
    };
    Schedule {
        lowering: "toy",
        op: 0,
        swap: false,
        levels: vec![
            Level {
                ext: [tm, tn, 1],
                bind: Bind::Exec(4),
                homes: [None, None],
            },
            Level {
                ext: [ty * tm, tx * tn, bk],
                bind: Bind::Slot,
                homes: [Some(home(stages)), Some(home(stages))],
            },
        ],
        groups: [1, 1],
        producers: 0,
        regs: [0, 0],
        cluster: [1, 1],
        multicast: [false, false],
        trav,
        launch: LaunchMode::Plain,
        policy: [Policy::Default; 2],
        per_unit: 0,
        params: vec![],
    }
}

pub fn declared(x: &Schedule) -> Declared {
    let [tm, tn, _] = x.lane_tile();
    let threads = (x.bm() / tm) * (x.bn() / tn);
    let stage = (x.bm() + x.bn()) * x.bk() * 4;
    let warps = threads.div_ceil(32) as f64;
    Declared {
        threads,
        regs: tm * tn + 2 * (tm + tn) + 32,
        smem: x.depth() * stage,
        issue: vec![
            (Class::Math, warps * (tm * tn * x.bk()) as f64),
            (Class::Shared, warps * (x.bk() * (tm + tn) / 4) as f64),
            (Class::Global, warps * 2.0),
        ],
        in_flight: x.depth().saturating_sub(1).max(1) * stage,
        fill: 1,
        chains: tm * tn,
        groups: threads.div_ceil(32),
        work: 0.0,
        whole_stores: false,
        barriers: 1.0,
        ..Declared::default()
    }
}

/// The memory place behind the toy's L2.
pub const MEM: usize = 3;

/// **The primitives an evaluator term reads**, as measured rows on the toy
/// machine `m`: the copy from the cache and from the memory, the
/// scratchpad's fragment load, the slot barrier, the store into the cache
/// and to the memory, the partial add — each `(name, of, per_clk, rate,
/// latency, concurrency)`. `mem_below`: the memory's knee latency
/// (concurrency over rate) below the cache's, as the RTX 3090's and the
/// A100's measured rows read (cp.async.16: 624 ns from L2, 383 from DRAM).
pub fn prim_rows(mem_below: bool) -> Vec<(&'static str, PrimOf, bool, f64, f64, f64)> {
    let (cache, mem) = match mem_below {
        // rate B/ns a unit, latency ns, concurrency B a unit.
        true => ((26.0, 154.0, 16384.0), (10.7, 298.0, 4096.0)),
        false => ((50.0, 150.0, 16384.0), (35.0, 350.0, 16384.0)),
    };
    vec![
        (
            "cp.l2",
            PrimOf::Copy {
                from: L2,
                bytes: 16,
            },
            false,
            cache.0,
            cache.1,
            cache.2,
        ),
        (
            "cp.mem",
            PrimOf::Copy {
                from: MEM,
                bytes: 16,
            },
            false,
            mem.0,
            mem.1,
            mem.2,
        ),
        (
            "lds",
            PrimOf::Scratch {
                bytes: 16,
                matrix: false,
            },
            true,
            128.0,
            30.0,
            8192.0,
        ),
        ("bar", PrimOf::Barrier { lanes: 256 }, true, 0.03, 30.0, 1.0),
        (
            "st.l2",
            PrimOf::Store { to: L2, bytes: 16 },
            false,
            40.0,
            150.0,
            512.0,
        ),
        (
            "st.mem",
            PrimOf::Store { to: MEM, bytes: 16 },
            false,
            30.0,
            300.0,
            512.0,
        ),
        (
            "red",
            PrimOf::Reduce {
                at: L2,
                sharing: 1,
                sectors: 4,
            },
            false,
            20.0,
            150.0,
            512.0,
        ),
    ]
}

/// The toy machine with [`prim_rows`] measured, row `i`'s rate scaled by
/// `r[i]`, its latency by `l[i]` and its concurrency by `c[i]`, each fact
/// with the interval `[0.8, 1.25]` of its value (the bounds differ).
pub fn primed(mem_below: bool, r: &[f64], l: &[f64], c: &[f64]) -> Machine {
    let mut m = machine(&[1.0; 9]);
    let f = |v: f64| Fact::measured(v, "toy").within(0.8 * v, 1.25 * v);
    m.prims = prim_rows(mem_below)
        .into_iter()
        .enumerate()
        .map(|(i, (name, of, per_clk, rate, lat, conc))| Prim {
            name,
            of,
            per_clk,
            rate: f(rate * r[i]),
            latency: f(lat * l[i]),
            conc: f(conc * c[i]),
        })
        .collect();
    m
}
