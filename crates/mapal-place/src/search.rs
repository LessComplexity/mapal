//! **The search** (ADR-0050 D5): candidates generated bottom-up from the
//! facts through each lowering's coordinate space ([`Lowering::dims`]) —
//! the operation, the group and slot tiles, `bk`, depths and layouts — then
//! the traversals ([`crate::space::traversals`]), every point priced by the
//! one evaluator. Branch and bound in a fixed enumeration order: a prefix
//! whose slot tile is fixed is cut where its admissible bound
//! ([`tile_bound`]: peak compute, the panels into the units, every operand
//! byte once) is above the `top`-th best price so far; a complete slot point
//! where its compute at the efficiency its footprint's occupancy reaches is
//! ([`point_bound`]). Dominance: a coordinate no evaluator term reads (a
//! layout's pad, an encoding) is narrowed by its lowering to the value its
//! own rule gives ([`Lowering::search_domain`]).
//!
//! The `top` candidates go through the **post-lowering gate** in rank order
//! (the code generator's real registers and spills: a refusal makes the
//! candidate infeasible and the next is taken); the survivors are priced at
//! the facts' three bounds and the pick is their **minimax** (the least
//! largest regret against each bound's best). A pick whose slow- and
//! fast-bound winners differ is **open** (`flip`), so is one whose runner-up
//! is within the measurements' null (`tie`: the model cannot separate
//! them). The **error band** (per bounding term, the replay's residuals) is
//! the margin a pick must clear to replace the incumbent (the rules' launch):
//! within it the incumbent stays, so a re-measured rate does not churn
//! emission.

use crate::eval::{Terms, efficiency, evaluate, evaluate_floor};
use crate::fact::Bound;
use crate::lower::{Declared, Lowering, TravKind};
use crate::machine::{ExecKind, LaunchMode, Machine};
use crate::schedule::{Schedule, Site, Traversal};
use crate::space;

/// The search's parameters.
#[derive(Clone, Debug)]
pub struct Opts {
    /// Candidates kept for the post-lowering gate and the minimax.
    pub top: usize,
    /// The model's error band, relative, by the bounding term of the pick's
    /// longest phase (`compute`, `dram`, `cache`, …; `all` the fallback).
    pub band: Vec<(String, f64)>,
    /// The measurements' null, relative: candidates closer than it are not
    /// separated (a runner-up inside it makes the pick open).
    pub null: f64,
    /// Walk every domain in reverse (the pick is invariant under the order).
    pub reverse: bool,
    /// The traversal kinds priced (empty: every kind a lowering emits) — a
    /// measurement arm's restriction.
    pub kinds: Vec<TravKind>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            top: 8,
            band: vec![("all".into(), 0.0)],
            null: 0.02,
            reverse: false,
            kinds: Vec::new(),
        }
    }
}

impl Opts {
    /// The band for terms `t`: its longest phase's bounding term's, else
    /// `all`'s.
    pub fn band_of(&self, t: &Terms) -> f64 {
        let term = t
            .phases
            .iter()
            .max_by(|a, b| a.ns.total_cmp(&b.ns))
            .map_or("all", |p| p.bound());
        self.band
            .iter()
            .find(|(n, _)| n == term)
            .or_else(|| self.band.iter().find(|(n, _)| n == "all"))
            .map_or(0.0, |(_, b)| *b)
    }
}

/// A priced candidate: its schedule, its lowering's declared counts, its
/// terms at the facts' values and its time at the slow, value and fast
/// bounds.
#[derive(Clone, Debug)]
pub struct Cand {
    pub x: Schedule,
    pub d: Declared,
    pub t: Terms,
    pub ns: [f64; 3],
}

impl Cand {
    pub fn value(&self) -> f64 {
        self.ns[1]
    }
}

/// What the search walked.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Complete slot points reached, schedules priced, prefixes and points
    /// cut by the bound, points the lowering refused.
    pub points: u64,
    pub evaluated: u64,
    pub pruned: u64,
    pub refused: u64,
    /// Candidates the post-lowering gate refused, with why.
    pub gated: Vec<(String, String)>,
}

/// The search's result.
#[derive(Clone, Debug)]
pub struct Ranked {
    /// The schedule taken: the minimax pick, or the incumbent where the
    /// pick does not clear the band.
    pub pick: Cand,
    /// The gate's survivors, best first at the facts' values.
    pub top: Vec<Cand>,
    /// The minimax pick (before the band).
    pub best: Cand,
    pub incumbent: Option<Cand>,
    /// The incumbent kept: the best is not faster beyond the band.
    pub kept: bool,
    /// Why the pick is open (`flip: …`, `tie: …`); empty: decided.
    pub open: Vec<String>,
    pub stats: Stats,
}

/// The admissible bound of every schedule with slot tile `bm × bn` on
/// operation `op` (any `bk`, depth, occupancy, traversal): its flops at the
/// operation's peak on every unit, the panels it requests through the
/// device cache (device-wide and into the units), every operand byte once
/// from memory, and one launch.
pub fn tile_bound(m: &Machine, s: &Site, op: usize, bm: u64, bn: u64, b: Bound) -> f64 {
    let units = m.units().max(1) as f64;
    let ghz = m.ghz(b);
    let rate = m.ops.get(op).map_or(0.0, |o| o.rate.rate(b)) * ghz;
    let compute = s.flops() as f64 / (units * rate).max(1e-12);
    let (bm, bn) = (bm.max(1) as f64, bn.max(1) as f64);
    let requested =
        (s.batch.max(1) * s.m * s.n * s.k) as f64 * (s.ea as f64 / bn + s.eb as f64 / bm);
    let (dev, unit) = m
        .device_cache()
        .map(|c| {
            (
                m.reach(c, ExecKind::Device)
                    .map_or(f64::INFINITY, |r| r.total_bytes_per_ns.rate(b)),
                m.reach(c, ExecKind::Unit)
                    .map_or(f64::INFINITY, |r| r.bytes_per_ns.rate(b)),
            )
        })
        .unwrap_or((f64::INFINITY, f64::INFINITY));
    let cache = requested / dev.max(1e-12);
    let into = requested / (units * unit).max(1e-12);
    // Every operand byte once (warm ones none) and the output, from memory.
    let mem = m
        .device_memory()
        .and_then(|p| m.reach(p, ExecKind::Device))
        .map_or(f64::INFINITY, |r| r.total_bytes_per_ns.rate(b));
    let bt = s.batch.max(1);
    let bytes = bt * s.m * s.n * (s.eo + s.epi)
        + if s.warm[0] { 0 } else { bt * s.m * s.k * s.ea }
        + if s.warm[1] { 0 } else { bt * s.k * s.n * s.eb };
    let once = bytes as f64 / mem.max(1e-12);
    let launch = m.launch(LaunchMode::Plain).map_or(0.0, |l| l.ns.cost(b));
    compute.max(cache).max(into).max(once) + launch
}

/// The admissible compute bound of every schedule whose slot runs `lanes`
/// lanes of `chains` accumulator chains each: its flops at the operation's
/// rate and the efficiency of those chains with as many groups resident as
/// the unit's lane and slot capacities allow (registers and scratch only
/// lower it), and one launch.
pub fn lanes_bound(m: &Machine, s: &Site, op: usize, lanes: u64, chains: u64, b: Bound) -> f64 {
    let occ = m.occupancy(lanes, 0, &[]).max(1);
    let groups = (occ * lanes.div_ceil(m.group_lanes().max(1))) as f64 / m.subs().max(1) as f64;
    let (eff, _) = efficiency(m, op, chains as f64, groups, b);
    let units = m.units().max(1) as f64;
    let rate = m.ops.get(op).map_or(0.0, |o| o.rate.rate(b)) * m.ghz(b);
    let launch = m.launch(LaunchMode::Plain).map_or(0.0, |l| l.ns.cost(b));
    s.flops() as f64 / (units * rate * eff.max(1e-3)).max(1e-12) + launch
}

/// The admissible bound of slot point `x` (declared `d`) under any
/// traversal: [`tile_bound`], and its work at the operation's rate and the
/// efficiency its chains reach with the footprint's occupancy resident (the
/// efficiency is monotone in resident groups; a phase holds at most that).
pub fn point_bound(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound) -> f64 {
    let tile = tile_bound(m, s, x.op, x.bm(), x.bn(), b);
    let scratch = m.unit_scratch();
    let bytes: Vec<_> = scratch.map(|p| (p, d.smem)).into_iter().collect();
    let occ = m.occupancy(d.threads, d.regs, &bytes).max(1);
    let groups = (occ * d.groups) as f64 / m.subs().max(1) as f64;
    let (eff, _) = efficiency(m, x.op, d.chains as f64, groups, b);
    let useful = (2 * x.bm() * x.bn() * x.bk()).max(1) as f64;
    let passes = if d.work > 0.0 { d.work / useful } else { 1.0 };
    let units = m.units().max(1) as f64;
    let ghz = m.ghz(b);
    let rate = m.ops[x.op].rate.rate(b) * ghz;
    let launch = m.launch(LaunchMode::Plain).map_or(0.0, |l| l.ns.cost(b));
    let compute = s.flops() as f64 * passes / (units * rate * eff.max(1e-3)).max(1e-12);
    // Issue: every traversal runs the site's `tiles × panels` trips over
    // the units, each at least the slot's declared counts (a combine adds
    // its own), per class against its pipe and all against the dispatch.
    let trips = (x.tiles(s) * x.panels(s)) as f64 / units;
    let mut worst: f64 = 0.0;
    let mut all = 0.0;
    for (c, n) in &d.issue {
        all += n;
        if let Some(p) = m.pipe(*c) {
            worst = worst.max(trips * n / (p.per_clk.rate(b) * ghz));
        }
    }
    let issue = worst.max(trips * all / (m.subs().max(1) as f64 * ghz));
    tile.max(compute.max(issue) + launch)
}

/// The best `top` so far, by price then plan (a total order: the pick is
/// the same whatever the enumeration order).
struct Best {
    top: Vec<Cand>,
    k: usize,
    /// The incumbent's price: a candidate above it is never taken (the
    /// incumbent would stay), so nothing above it is kept either.
    cap: f64,
    /// Candidates whose floor is under the threshold, priced fully in
    /// floor order ([`settle`]).
    pending: Vec<(f64, Schedule, Declared)>,
}

impl Best {
    fn kth(&self) -> f64 {
        let kth = if self.top.len() < self.k {
            f64::INFINITY
        } else {
            self.top[self.k - 1].value()
        };
        kth.min(self.cap)
    }

    fn offer(&mut self, c: Cand) {
        let key = |c: &Cand| (c.value(), c.x.plan());
        let plan = c.x.plan();
        if self.top.iter().any(|y| y.x.plan() == plan) {
            return;
        }
        let pos = self.top.partition_point(|y| {
            let (a, b) = (key(y), key(&c));
            a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
        });
        if pos >= self.k {
            return;
        }
        self.top.insert(pos, c);
        self.top.truncate(self.k);
    }
}

/// Price slot point `x` of lowering `l` under every traversal it emits.
fn consider(
    m: &Machine,
    s: &Site,
    l: &dyn Lowering,
    mut x: Schedule,
    opts: &Opts,
    best: &mut Best,
    st: &mut Stats,
) {
    st.points += 1;
    let Ok(d0) = l.declare(m, s, &x) else {
        st.refused += 1;
        return;
    };
    let scratch = m.unit_scratch();
    let bytes: Vec<_> = scratch.map(|p| (p, d0.smem)).into_iter().collect();
    let occ = m.occupancy(d0.threads, d0.regs, &bytes);
    if occ == 0 {
        st.refused += 1;
        return;
    }
    x.per_unit = occ;
    if point_bound(m, s, &x, &d0, Bound::Value) > best.kth() {
        st.pruned += 1;
        return;
    }
    // The ordered traversals first (their closed forms are the cheap ones:
    // the threshold tightens before a walk's windows are counted).
    let mut ts = space::traversals(m, s, &x, &d0, l.traversals());
    ts.retain(|t| opts.kinds.is_empty() || opts.kinds.contains(&TravKind::of(*t)));
    ts.sort_by_key(|t| match t {
        Traversal::Ordered { .. } => 0,
        Traversal::Split { .. } => 1,
        _ => 2,
    });
    for t in ts {
        let y = Schedule {
            trav: t,
            ..x.clone()
        };
        // The floor at the slot's own counts (a combine only adds to
        // them): no lower than the traversal's own declared floor.
        if evaluate_floor(m, s, &y, &d0, Bound::Value).ns > best.kth() {
            st.pruned += 1;
            continue;
        }
        let Ok(d) = l.declare(m, s, &y) else {
            st.refused += 1;
            continue;
        };
        let floor = evaluate_floor(m, s, &y, &d, Bound::Value).ns;
        if floor > best.kth() {
            st.pruned += 1;
            continue;
        }
        best.pending.push((floor, y, d));
    }
}

/// Price the pending candidates fully, best floor first, until the floor
/// passes the `top`-th price (best-first: the reuse closed forms run only
/// where a candidate may still enter).
fn settle(m: &Machine, s: &Site, best: &mut Best, st: &mut Stats) {
    let mut pending = std::mem::take(&mut best.pending);
    pending.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.plan().cmp(&b.1.plan()))
    });
    for (floor, y, d) in pending {
        if floor > best.kth() {
            st.pruned += 1;
            continue;
        }
        let terms = evaluate(m, s, &y, &d, Bound::Value);
        st.evaluated += 1;
        if !terms.feasible() {
            continue;
        }
        let v = terms.ns;
        best.offer(Cand {
            x: y,
            d,
            t: terms,
            ns: [f64::NAN, v, f64::NAN],
        });
    }
}

/// The depth-first walk of `l`'s coordinates from prefix `pre`.
#[allow(clippy::too_many_arguments)]
fn walk(
    m: &Machine,
    s: &Site,
    l: &dyn Lowering,
    swap: bool,
    pre: &mut Vec<u64>,
    opts: &Opts,
    best: &mut Best,
    st: &mut Stats,
) {
    if let Some((bm, bn)) = l.tile_of(pre, swap) {
        let mut b = tile_bound(m, s, pre[0] as usize, bm, bn, Bound::Value);
        if let Some((lanes, chains)) = l.lanes_of(pre, swap) {
            b = b.max(lanes_bound(
                m,
                s,
                pre[0] as usize,
                lanes,
                chains,
                Bound::Value,
            ));
        }
        if b > best.kth() {
            st.pruned += 1;
            return;
        }
    }
    if pre.len() == l.dims() {
        if let Some(x) = l.point(m, s, pre, swap) {
            consider(m, s, l, x, opts, best, st);
        }
        return;
    }
    let mut dom = l.search_domain(m, s, pre.len(), pre);
    if opts.reverse {
        dom.reverse();
    }
    for v in dom {
        pre.push(v);
        walk(m, s, l, swap, pre, opts, best, st);
        pre.pop();
    }
}

/// Price a candidate at the facts' three bounds.
fn bounds(m: &Machine, s: &Site, c: &mut Cand) {
    for (i, b) in [(0, Bound::Slow), (2, Bound::Fast)] {
        c.ns[i] = evaluate(m, s, &c.x, &c.d, b).ns;
    }
}

/// **Search** site `s` on machine `m` over lowerings `lows` (D5): the
/// branch and bound over each lowering's space, the `top` through `gate`
/// (`Err`: the candidate lowers to code the post-lowering gate refuses),
/// the minimax over the facts' bounds, and the incumbent (a schedule of
/// `lows[i]`, the rules' launch) kept unless the pick clears the band.
/// `None`: no candidate passes.
pub fn search(
    m: &Machine,
    s: &Site,
    lows: &[&dyn Lowering],
    opts: &Opts,
    incumbent: Option<(usize, Schedule)>,
    gate: &mut dyn FnMut(&Schedule, &Declared) -> Result<(), String>,
) -> Option<Ranked> {
    let mut best = Best {
        top: Vec::new(),
        k: opts.top.max(1),
        cap: f64::INFINITY,
        pending: Vec::new(),
    };
    let mut st = Stats::default();
    // The incumbent, priced as it is.
    let inc = incumbent.and_then(|(i, x)| {
        let l = lows.get(i)?;
        let d = l.declare(m, s, &x).ok()?;
        let t = evaluate(m, s, &x, &d, Bound::Value);
        t.feasible().then(|| {
            let v = t.ns;
            Cand {
                x,
                d,
                t,
                ns: [f64::NAN, v, f64::NAN],
            }
        })
    });
    // The incumbent bounds the search: a schedule priced above it is never
    // taken (the incumbent stays), so nothing above it is kept.
    if let Some(c) = &inc {
        best.cap = c.value();
    }
    let mut order: Vec<usize> = (0..lows.len()).collect();
    if opts.reverse {
        order.reverse();
    }
    for &li in &order {
        let l = lows[li];
        let mut ops = l.ops(m, s);
        if opts.reverse {
            ops.reverse();
        }
        for (op, swap) in ops {
            if l.dims() == 0 {
                for x in l.slots(m, s, op, swap) {
                    consider(m, s, l, x, opts, &mut best, &mut st);
                }
                settle(m, s, &mut best, &mut st);
                continue;
            }
            let mut pre = vec![op as u64];
            walk(m, s, l, swap, &mut pre, opts, &mut best, &mut st);
            // Settled a lowering's operation at a time: its best prices
            // tighten the threshold the next one is cut at.
            settle(m, s, &mut best, &mut st);
        }
    }
    // The post-lowering gate, in rank order.
    let mut top = Vec::new();
    for mut c in best.top {
        match gate(&c.x, &c.d) {
            Ok(()) => {
                bounds(m, s, &mut c);
                top.push(c);
            }
            Err(why) => st.gated.push((c.x.plan(), why)),
        }
    }
    let inc = inc.and_then(|mut c| match gate(&c.x, &c.d) {
        Ok(()) => {
            bounds(m, s, &mut c);
            Some(c)
        }
        Err(why) => {
            st.gated.push((c.x.plan(), why));
            None
        }
    });
    if top.is_empty() {
        let c = inc?;
        return Some(Ranked {
            pick: c.clone(),
            top: vec![c.clone()],
            best: c.clone(),
            incumbent: Some(c),
            kept: true,
            open: vec!["the gate refused every candidate".into()],
            stats: st,
        });
    }
    // Minimax over the bounds: the least largest regret.
    let best_at = |j: usize| top.iter().map(|c| c.ns[j]).fold(f64::INFINITY, f64::min);
    let lows_b = [best_at(0), best_at(1), best_at(2)];
    let regret = |c: &Cand| {
        (0..3)
            .map(|j| c.ns[j] / lows_b[j].max(1e-12))
            .fold(1.0, f64::max)
    };
    let pick_i = (0..top.len())
        .min_by(|&a, &b| {
            regret(&top[a])
                .total_cmp(&regret(&top[b]))
                .then(top[a].value().total_cmp(&top[b].value()))
                .then(top[a].x.plan().cmp(&top[b].x.plan()))
        })
        .expect("a candidate");
    let argmin = |j: usize| {
        (0..top.len())
            .min_by(|&a, &b| {
                top[a].ns[j]
                    .total_cmp(&top[b].ns[j])
                    .then(top[a].x.plan().cmp(&top[b].x.plan()))
            })
            .expect("a candidate")
    };
    let mut open = Vec::new();
    let (slow, fast) = (argmin(0), argmin(2));
    if slow != fast {
        open.push(format!(
            "flip: slow {} fast {}",
            top[slow].x.plan(),
            top[fast].x.plan()
        ));
    }
    let bestc = top[pick_i].clone();
    if let Some(r) = top
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pick_i)
        .map(|(_, c)| c)
        .min_by(|a, b| a.value().total_cmp(&b.value()))
        && r.value() <= bestc.value() * (1.0 + opts.null)
    {
        open.push(format!(
            "tie: {} {:.0} ns within {:.0}% of {:.0}",
            r.x.plan(),
            r.value(),
            opts.null * 100.0,
            bestc.value()
        ));
    }
    let (pick, kept) = match &inc {
        Some(i) if i.value() <= bestc.value() * (1.0 + opts.band_of(&bestc.t)) => (i.clone(), true),
        _ => (bestc.clone(), false),
    };
    Some(Ranked {
        pick,
        top,
        best: bestc,
        incumbent: inc,
        kept,
        open,
        stats: st,
    })
}
