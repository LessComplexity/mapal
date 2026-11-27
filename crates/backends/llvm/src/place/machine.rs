//! [`machine_of`]: a GPU profile as a [`Machine`], and the primitive table.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

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
pub(super) fn prim_of(name: &str, ops: &[Op]) -> Option<(PrimOf, bool)> {
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
