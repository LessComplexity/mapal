//! The search's seam (ADR-0050 C4): forced launches, the gates, [`placed`], and the coverage probe.
#![cfg_attr(not(test), allow(dead_code))]

use super::*;

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
    pub(super) static FORCED: std::cell::RefCell<Option<Launch>> = const { std::cell::RefCell::new(None) };
    /// The table rungs of the sites being searched, by site, part, arms and
    /// arithmetic ([`memo_rungs`]; cleared when a search ends).
    pub(super) static RUNGS: std::cell::RefCell<Vec<(TileSite, &'static str, crate::GpuArms, Arithmetic, crate::GpuRungs)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The searches made, by site, part, face, arms and order.
    pub(super) static PICKS: std::cell::RefCell<std::collections::HashMap<String, Option<Placed>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The FMA tile layouts of the site being searched ([`Emitter::narrowed`];
    /// cleared when a search ends).
    pub(super) static LAYOUT: std::cell::RefCell<std::collections::HashMap<[u64; 7], (u64, u64)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The FMA tiles' shared-memory instructions a panel of the site being
    /// searched (the bank model's, [`gpu::lsu_per_panel`]; cleared with it).
    pub(super) static LSU: std::cell::RefCell<std::collections::HashMap<[u64; 9], u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// The rule chain is being derived ([`rules`]): no placement inside.
    pub(super) static RULES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// A search is running (the per-site memos are valid).
    pub(super) static SEARCHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Each part's machine, by its properties.
    pub(super) static MACHINES: std::cell::RefCell<std::collections::HashMap<String, Machine>> =
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

pub(super) fn forced_now() -> Option<Launch> {
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

pub(super) fn rules_now() -> bool {
    RULES.with(|c| c.get())
}

/// [`table_rungs`], memoized while a search runs (every candidate's
/// `launch_of` reads them).
pub(super) fn memo_rungs(
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
pub(super) fn machine(p: &TargetProfile) -> Option<Machine> {
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
pub(super) fn ptxas_gate(
    p: &TargetProfile,
    text: &str,
    d: &Declared,
    threads: u64,
) -> Result<(), String> {
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
pub(super) fn gate(
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
pub(super) fn key(
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
pub(super) fn search_site(
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

pub(super) fn search_uncached(
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
pub(super) fn trav_kind(code: u8, t: Traversal) -> bool {
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
pub(super) fn forced_search(
    m: &Machine,
    s: &Site,
    emitters: &[Emitter],
    arms: crate::GpuArms,
    inc: Option<(usize, Schedule)>,
    gate: &mut dyn FnMut(&Schedule, &Declared) -> Result<(), String>,
) -> Option<mapal_place::Ranked> {
    /// One emitter, its slot space the rules' slot alone (`--place-tile`).
    pub(super) struct Slot<'a, 'b> {
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
pub(super) fn arms_order(arms: crate::GpuArms) -> crate::Order {
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
pub(super) fn _uses(_: DeviceCost, _: Ty) {}

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
