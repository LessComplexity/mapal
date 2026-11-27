//! ADR-0050 C3: **`place_replay.tsv` scored** — the machine model's pick
//! against the measured winner and the rules' pick on every replayed site
//! (test `place_replay_scores`).

use crate::func::gpu::{self, Arithmetic, Combine, Launch};
use crate::place::{Emitter, lowering_of, machine_of, schedule_of, site_of, table_rungs};
use crate::profile::{GpuMmaTile, MmaFmt, TargetProfile};
use mapal_ir::{TileSite, Ty};
use mapal_place::{Lowering, Site};

/// A replay site: `pj`-style (`a` row-major `m × k`, `b` stored `n × k`)
/// for the projections, `mm`-style (`b` row-major `k × n`) for the cells.
pub(crate) fn replay_site(m: u64, n: u64, k: u64, fmt: &str, mm: bool) -> TileSite {
    use mapal_ir::{NarrowFloat, TileRead, Value};
    let narrow = |on: bool| on.then_some(NarrowFloat::Bf16);
    let (na, nb) = match fmt {
        "bf16" => (true, true),
        "bf16w" => (false, true),
        _ => (false, false),
    };
    // An f32 or TF32 product's `b` runs along the lanes (`k × n`): what
    // the TF32 tile and the cells read.
    let mm = mm || fmt == "tf32" || fmt == "f32";
    let a = TileRead {
        slot: 0,
        base: 0,
        ci: k,
        ck: 1,
        clane: 0,
        ksplit: None,
        narrow: narrow(na),
    };
    let b = match mm {
        true => TileRead {
            slot: 1,
            base: 0,
            ci: 0,
            ck: n,
            clane: 1,
            ksplit: None,
            narrow: narrow(nb),
        },
        false => TileRead {
            slot: 1,
            base: 0,
            ci: 0,
            ck: 1,
            clane: k,
            ksplit: None,
            narrow: narrow(nb),
        },
    };
    TileSite {
        rows: m,
        c: n,
        k,
        a,
        b,
        seed: Value::F32(0.0),
        elem: Ty::f32(),
        mul_a_first: true,
        add_acc_first: true,
        radix: None,
        tf32: fmt == "tf32",
        half: (fmt == "bf16" || fmt == "bf16w").then_some(NarrowFloat::Bf16),
        post: None,
    }
}

/// The launch an arm's plan names on `site`: an explicit warp tensor tile,
/// or the rules' launch under the plan's arms.
fn arm_launch(site: &TileSite, p: &'static TargetProfile, plan: &str) -> Option<Launch> {
    let g = p.gpu()?;
    let mut arms = crate::GpuArms {
        free: true,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    if let Some(rest) = plan.strip_prefix("rules") {
        for f in rest
            .trim_start_matches(':')
            .split(',')
            .filter(|f| !f.is_empty())
        {
            let (on, name) = (f.starts_with('+'), &f[1..]);
            *(if on {
                arms.on.rung(name)
            } else {
                arms.off.rung(name)
            })? = true;
        }
        return Some(gpu::launch_in(site, p, true, arms, crate::Order::Free));
    }
    let f: Vec<&str> = plan.split(':').collect();
    let fmt = match f[1] {
        "tf32" => MmaFmt::Tf32,
        _ => MmaFmt::Bf16,
    };
    let w: Vec<u64> = f[2].split('.').map(|x| x.parse().unwrap()).collect();
    let num = |x: &str, pre: &str| x.strip_prefix(pre).unwrap().parse::<u64>().unwrap();
    let (bk, st, u) = (num(f[3], "bk"), num(f[4], "st"), num(f[5], "u"));
    let (ea, eb) = (site.a.elem_bytes(&site.elem), site.b.elem_bytes(&site.elem));
    let mut m = GpuMmaTile {
        mt: w[0],
        nt: w[1],
        warps_y: w[2],
        warps_x: w[3],
        bk,
        stages: st.max(2),
        pad_a: 0,
        pad_b: 0,
        blocks_per_sm: u,
        fmt,
        ea,
        eb,
        bt: site.b.clane != 1,
        split: None,
    };
    // The pads the tile's own rule takes: `ldmatrix`'s 16-byte bank groups
    // for a 16-bit panel, four words for a TF32 one.
    let pad = |ld: u64| match fmt {
        MmaFmt::Tf32 => 4,
        _ => crate::profile::ldmatrix_pad(ld),
    };
    m.pad_a = pad(m.bk);
    m.pad_b = match m.bt {
        true => m.pad_a,
        false => pad(m.bn()),
    };
    if st == 0 {
        // The rules' depth: as many stages as the slot's share holds.
        let share = g.smem_per_sm / u - g.smem_reserved_per_block;
        m.stages = (share / m.stage_bytes()).clamp(2, site.k / bk);
    }
    let (gx, gy) = (site.c / m.bn(), site.rows / m.bm());
    let tiles = gx * gy;
    // `<traversal>.h<h>`: the raster group height (1 without).
    let (trav, h) = f[6]
        .split_once(".h")
        .map_or((f[6], 1), |(t, h)| (t, h.parse().unwrap_or(1)));
    let combine = match trav {
        "ord" => Combine::None,
        t if t.starts_with("cut") => Combine::Atomic { s: num(t, "cut") },
        // ADR-0050 C5: whole waves of the footprint's slots ordered, the
        // tail walked over them.
        "hybrid" => {
            let w = g.sms * u;
            let whole = tiles / w * w;
            let (ctas, iters) = mapal_place::reuse::walk_grid((tiles - whole) * (site.k / bk), w);
            Combine::StreamK { ctas, iters, whole }
        }
        "walk" => {
            let (ctas, iters) = mapal_place::reuse::walk_grid(tiles * (site.k / bk), g.sms * u);
            Combine::StreamK {
                ctas,
                iters,
                whole: 0,
            }
        }
        t => {
            let (c, i) = t.strip_prefix("walk")?.split_once('x')?;
            Combine::StreamK {
                ctas: c.parse().ok()?,
                iters: i.parse().ok()?,
                whole: 0,
            }
        }
    };
    let arithmetic = match fmt {
        MmaFmt::Tf32 => Arithmetic::Tf32Mma,
        _ => Arithmetic::HalfMma,
    };
    Some(Launch {
        arithmetic,
        tiled: None,
        mma: Some(m),
        wgmma: None,
        gx,
        gy,
        block: m.threads(g.warp),
        smem: m.smem_bytes(),
        raster: h,
        rungs: table_rungs(site, p, arms, arithmetic),
        combine,
        cells: 1,
    })
}

/// One arm's evaluation: its time at the slow, value and fast bounds, its
/// terms, and its time at every corner of the facts' intervals.
type ArmEval = ([f64; 3], mapal_place::Terms, Vec<f64>);

/// One arm's evaluation at the three bounds and every corner, and its
/// terms.
fn arm_ns(
    site: &TileSite,
    p: &'static TargetProfile,
    l: &Launch,
    warm: [bool; 2],
) -> Option<ArmEval> {
    let m = machine_of(p)?;
    let arms = crate::GpuArms {
        free: true,
        ..crate::GpuArms::default()
    };
    let x = schedule_of(site, p, arms, l);
    let e = Emitter {
        id: lowering_of(l),
        site,
        profile: p,
        contract: true,
        arms,
    };
    let s = Site {
        warm,
        ..site_of(site, true, crate::Order::Free)
    };
    let d = e.declare(&m, &s, &x).ok()?;
    let at = |b| mapal_place::evaluate(&m, &s, &x, &d, b);
    let mut t = at(mapal_place::Bound::Value);
    t.marked.insert(0, x.plan());
    let corners: Vec<f64> = mapal_place::Corner::corners()
        .into_iter()
        .map(|c| mapal_place::evaluate_corner(&m, &s, &x, &d, c).ns)
        .collect();
    Some((
        [
            at(mapal_place::Bound::Slow).ns,
            t.ns,
            at(mapal_place::Bound::Fast).ns,
        ],
        t,
        corners,
    ))
}

/// **A replayed site's warm operands from its program** (the box column's
/// `program=<stem>`, `benches/algorithms/<stem>.mapal`): the program
/// lowered, its tile site of these extents, `residence_plan` at the part's
/// keep ([`crate::place::warm_sites`] — what an emission reads); none
/// without a program.
fn warm_of_program(boxes: &str, p: &TargetProfile, (m, n, k): (u64, u64, u64)) -> [bool; 2] {
    let Some(stem) = boxes
        .split_whitespace()
        .find_map(|b| b.strip_prefix("program="))
    else {
        return [false, false];
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../benches/algorithms")
        .join(format!("{stem}.mapal"));
    let src = std::fs::read_to_string(&path).expect("the replayed program");
    let po = mapal_syntax::parse(&src);
    assert!(po.diagnostics.is_empty(), "{stem} parses");
    let ir = mapal_lower::lower(&src, &po.program).expect("lowers");
    let ir = mapal_rewrite::rewrite(ir).ir;
    let keep = p.gpu().expect("a GPU part").l2_keep_bytes();
    crate::place::warm_sites(&ir, keep)
        .into_iter()
        .find(|(s, _)| (s.rows, s.c, s.k) == (m, n, k))
        .map(|(_, w)| w)
        .expect("the program's site of the replayed extents")
}

/// The replay's sites: rows grouped by id. Two ids of one site (part,
/// extents, formats, order, box and every arm's plan) would count one
/// measurement twice — refused (`place_replay_sites_are_distinct`).
fn replay_sites(text: &str) -> std::collections::BTreeMap<String, Vec<Vec<&str>>> {
    let mut sites: std::collections::BTreeMap<String, Vec<Vec<&str>>> = Default::default();
    for l in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = l.split('\t').collect();
        sites.entry(f[0].to_string()).or_default().push(f);
    }
    sites
}

/// A site's identity: what the model sees of it.
fn site_key(arms: &[Vec<&str>]) -> String {
    let f0 = &arms[0];
    let plans: Vec<&str> = arms.iter().map(|a| a[8]).collect();
    format!(
        "{} {}x{}x{} {} {} {} {}",
        f0[1],
        f0[2],
        f0[3],
        f0[4],
        f0[5],
        f0[6],
        f0[12],
        plans.join(",")
    )
}

/// **One measured site, one replay site**: no two ids of the corpus are
/// the same site (a repeated measurement is merged into one row, its
/// source naming both — `mma_cut_replay`'s two runs of the RTX 3090's and
/// the A100's 2048³ TF32 cut).
#[test]
fn place_replay_sites_are_distinct() {
    let sites = replay_sites(include_str!("place_replay.tsv"));
    let mut seen: std::collections::BTreeMap<String, &str> = Default::default();
    for (id, arms) in &sites {
        if let Some(other) = seen.insert(site_key(arms), id) {
            panic!("{id} and {other} are one site: {}", site_key(arms));
        }
    }
}

/// **The model's misses on the replay, each with the term it lacks**
/// (`benches/results/evidence/local-rtx-5080/machine-model.md`): checked —
/// a miss that turns into a pick (or an open) must leave this list. Scored
/// by the search's own rule ([`mapal_place::verdict`]): a pick within the
/// null, or one the facts' intervals flip at some corner, is open, never
/// counted here or as a pick.
const REPLAY_OPEN: [&str; 9] = [
    // RTX 3090 2048³ TF32: the uncut grid won (5.9 %); the model prices the
    // cut 15 % under it by wave quantization. The 3090 measured no cost
    // model primitive: its reductions, stores and movers are derived from
    // its other facts (marked) — the missing facts are its own `prim` rows
    // (the rental round's `place_calib.sh`).
    "cut-0",
    // A100 128 x 4096 x 14336 (bf16, TF32) and 2048 x 4096 x 14336 bf16: the
    // cut won (9.6 / 25 / 8.3 %); the model prices the uncut 3–6 % under it
    // at the L2's device rate. The uncut grid (128 tiles on 108 SMs) does not
    // draw that rate: the L2's realized rate at the grid's in-flight bytes —
    // the `prim` sweep's points on the part (unmeasured: rental).
    "cut-10", "cut-8", "cut-9",
    // RTX 5080 128 x 28672 x 4096 bf16: every arm ~2.9x under its time at the
    // DRAM roofline. The panel stream's measured rate by run (0.35–0.5x at a
    // 32-byte run, `PrimOf::Panel`) would explain it, but c11-0's `-raster`
    // arm streams 32-byte runs at 888 GB/s (the full rate): the property that
    // separates the two is unidentified, so no term reads the run.
    "hyb-0",
    // RTX 5080 512 x 8192 x 2048 (hybrid against the walk, 3.6 %) and 2048 x
    // 6144 x 4096 (ordered against the walk, 12 %): the walk's own per-unit
    // cost — its k loop one copy where the ordered kernel's is two (SASS: 64
    // against 128 HMMA a body), its segment loop and the shared tiles' fixup;
    // its partial adds are priced (measured: `red.add.f32.frag`, 2.6 op / ns
    // an SM) and overlap the loop but the last pieces'.
    "hyb-1", // RTX 5080 2048 x 6144 x 4096: the walk's per-unit cost (hyb-1's).
    "walk-16",
    // RTX 5080 512 x 2048 x 2048 and its transpose: the walk (128 x 128 tile)
    // won 12–15 % over the ordered 64 x 128 tile, priced at its compute (49
    // against 57 µs measured): a small warp tile's exposed fragment latency
    // (`ldmatrix` 69 clocks, then the products, at two warps a scheduler) —
    // the scratchpad loads' latency against the products in flight.
    "walk-6", "walk-7",
];

/// The model's picks under the search's own rule (C5's corpus, duplicates
/// merged, ties and flips at any corner open): a floor, not a target — the
/// rules pick 59 (`model >= rules` is ADR-0050 C6's bar). The cost model's
/// pipeline calculation (cost-model.md) scores 28: the residual of the
/// absolute times fell (median |log| 0.126 → 0.107) while the ranking of
/// near-equal arms did not improve (36 open: ties within the null and flips
/// at marked facts; the misses above).
const MODEL_FLOOR: u64 = 28;

/// **ADR-0050 C3: `place_replay.tsv` scored** — per measured site, the
/// measured winner (or null: within the null), the model's pick (the arm
/// the evaluator prices lowest; open where the pick flips between the
/// facts' slow and fast bounds) and the rules' pick, each picked or
/// missed. The model keeps at least [`MODEL_FLOOR`] picks and every site it
/// misses is in [`REPLAY_OPEN`] with the term it lacks; `model >= rules` is
/// reported (not met at C3).
#[test]
fn place_replay_scores() {
    let sites = replay_sites(include_str!("place_replay.tsv"));
    #[derive(Default, Clone, Copy)]
    struct Score {
        picked: u64,
        missed: u64,
        null: u64,
        open: u64,
    }
    let mut model: std::collections::BTreeMap<String, Score> = Default::default();
    let mut rules: std::collections::BTreeMap<String, Score> = Default::default();
    let mut misses = Vec::new();
    let mut opens = Vec::new();
    let mut lines = Vec::new();
    for (id, arms) in &sites {
        let fam = id.split('-').next().unwrap().to_string();
        let f0 = &arms[0];
        let p = crate::profile::resolve(f0[1]).expect("a part");
        let (m, n, k): (u64, u64, u64) = (
            f0[2].parse().unwrap(),
            f0[3].parse().unwrap(),
            f0[4].parse().unwrap(),
        );
        let mm = fam == "kfill" || fam == "skfill" || fam == "pc" || fam == "split";
        let site = replay_site(m, n, k, f0[5], mm);
        let mut evals = Vec::new();
        // The operands resident where the site reads them: its program's
        // `residence_plan`.
        let warm = warm_of_program(f0[12], p, (m, n, k));
        for a in arms {
            let l = arm_launch(&site, p, a[8]);
            let ev = l.as_ref().and_then(|l| arm_ns(&site, p, l, warm));
            let us: f64 = a[9].parse().unwrap();
            evals.push((a[7], a[8], us, a[11] == "rules", ev, l));
        }
        let null: f64 = f0[10].parse().unwrap();
        let best = evals
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.2.total_cmp(&b.1.2))
            .map(|(i, _)| i)
            .unwrap();
        let worst = evals.iter().map(|e| e.2).fold(0.0, f64::max);
        let measured_null = (worst - evals[best].2) / evals[best].2 <= null;
        // The model's pick and opens by the search's own rule (minimax
        // over the corners; a flip or a tie within the null is open).
        let lowered = evals.iter().all(|e| e.4.is_some());
        let v = lowered.then(|| {
            let ev = |e: &(&str, &str, f64, bool, Option<ArmEval>, Option<Launch>)| {
                e.4.clone().expect("lowered")
            };
            mapal_place::verdict(
                &evals.iter().map(|e| ev(e).0[1]).collect::<Vec<_>>(),
                &evals.iter().map(|e| ev(e).2).collect::<Vec<_>>(),
                &evals.iter().map(|e| e.0.to_string()).collect::<Vec<_>>(),
                crate::place::PLACE_NULL,
            )
        });
        let rules_pick = evals.iter().position(|e| e.3);
        let ms = model.entry(fam.clone()).or_default();
        let rs = rules.entry(fam.clone()).or_default();
        let model_v = match &v {
            _ if measured_null => {
                ms.null += 1;
                "null".to_string()
            }
            None => {
                ms.open += 1;
                "open(unlowered)".to_string()
            }
            Some(v) if !v.open.is_empty() => {
                ms.open += 1;
                opens.push(id.clone());
                format!("open({})", v.open.join("; "))
            }
            Some(v) if v.pick == best => {
                ms.picked += 1;
                "picked".to_string()
            }
            Some(_) => {
                ms.missed += 1;
                misses.push(id.clone());
                "missed".to_string()
            }
        };
        let rules_v = if measured_null {
            rs.null += 1;
            "null"
        } else if rules_pick == Some(best) {
            rs.picked += 1;
            "picked"
        } else {
            rs.missed += 1;
            "missed"
        };
        let arms_s: Vec<String> = evals
            .iter()
            .map(|e| {
                format!(
                    "{}={}us/{}",
                    e.0,
                    e.2,
                    e.4.as_ref().map_or("-".into(), |x| format!(
                        "{:.1}[{:.1}..{:.1}]",
                        x.0[1] / 1e3,
                        x.0[2] / 1e3,
                        x.0[0] / 1e3
                    ))
                )
            })
            .collect();
        lines.push(format!(
            "{id} {} {m}x{n}x{k} {}: measured {} | model {model_v} | rules {rules_v} | {}",
            f0[1],
            f0[5],
            if measured_null {
                "null".to_string()
            } else {
                evals[best].0.to_string()
            },
            arms_s.join(" ")
        ));
        if fam == "pc" || std::env::var("PLACE_TERMS").is_ok() {
            for e in &evals {
                if let Some((_, t, _)) = &e.4 {
                    lines.push(format!(
                        "    {} {}: {} | {}",
                        id,
                        e.0,
                        t.marked[0],
                        t.line()
                    ));
                }
            }
        }
    }
    for l in &lines {
        eprintln!("{l}");
    }
    let tot = |s: &std::collections::BTreeMap<String, Score>| {
        s.values().fold(Score::default(), |a, b| Score {
            picked: a.picked + b.picked,
            missed: a.missed + b.missed,
            null: a.null + b.null,
            open: a.open + b.open,
        })
    };
    eprintln!("family\tmodel picked/missed/open/null\trules picked/missed/null");
    for (f, s) in &model {
        let r = rules[f];
        eprintln!(
            "{f}\t{}/{}/{}/{}\t{}/{}/{}",
            s.picked, s.missed, s.open, s.null, r.picked, r.missed, r.null
        );
    }
    let (mt, rt) = (tot(&model), tot(&rules));
    eprintln!(
        "all\t{}/{}/{}/{}\t{}/{}/{}",
        mt.picked, mt.missed, mt.open, mt.null, rt.picked, rt.missed, rt.null
    );
    eprintln!("model misses: {misses:?}");
    eprintln!("model opens: {opens:?}");
    if std::env::var("PLACE_REPLAY_REPORT").is_ok() {
        return;
    }
    eprintln!(
        "model >= rules: {} ({} against {})",
        mt.picked >= rt.picked,
        mt.picked,
        rt.picked
    );
    assert!(
        mt.picked >= MODEL_FLOOR,
        "the model's picks fell below its floor"
    );
    let open: Vec<&str> = REPLAY_OPEN.to_vec();
    assert_eq!(
        misses, open,
        "the model's misses are the committed open list"
    );
}

/// Every replayed site's arms: its family, null, and per arm its name,
/// measured time and the model's three-bound price with its terms.
#[allow(clippy::type_complexity)]
fn replay_evals() -> Vec<(String, String, f64, Vec<(String, f64, Option<ArmEval>)>)> {
    let sites = replay_sites(include_str!("place_replay.tsv"));
    let mut out = Vec::new();
    for (id, arms) in &sites {
        let fam = id.split('-').next().unwrap().to_string();
        let f0 = &arms[0];
        let p = crate::profile::resolve(f0[1]).expect("a part");
        let (m, n, k): (u64, u64, u64) = (
            f0[2].parse().unwrap(),
            f0[3].parse().unwrap(),
            f0[4].parse().unwrap(),
        );
        let mm = fam == "kfill" || fam == "skfill" || fam == "pc" || fam == "split";
        let site = replay_site(m, n, k, f0[5], mm);
        let warm = warm_of_program(f0[12], p, (m, n, k));
        let evals = arms
            .iter()
            .map(|a| {
                let ev = arm_launch(&site, p, a[8]).and_then(|l| arm_ns(&site, p, &l, warm));
                (a[7].to_string(), a[9].parse().unwrap(), ev)
            })
            .collect();
        out.push((id.clone(), fam, f0[10].parse().unwrap(), evals));
    }
    out
}

/// **ADR-0050 D5: the model's error band is the replay's residuals** — per
/// measured site, each arm against the measured winner: the model's ratio
/// over the measured one, `|(model_b / model_a) / (measured_b / measured_a)
/// − 1|`, grouped by the bounding term of the winner's longest phase; per
/// term (three pairs at least, else `all`'s) the 80th percentile, per
/// mille — equal to [`crate::place::PLACE_BAND_PM`] (re-derived when the
/// evaluator or the corpus changes).
#[test]
fn place_band_is_the_replays_residuals() {
    let mut by: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for (_, _, _, evals) in replay_evals() {
        let Some(best) = evals
            .iter()
            .enumerate()
            .filter(|(_, e)| e.2.is_some())
            .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .map(|(i, _)| i)
        else {
            continue;
        };
        let (_, ta, Some((pa, terms, _))) = &evals[best] else {
            continue;
        };
        let term = terms
            .phases
            .iter()
            .max_by(|a, b| a.ns.total_cmp(&b.ns))
            .map_or("all", |p| p.bound())
            .to_string();
        for (i, (_, tb, eb)) in evals.iter().enumerate() {
            let Some((pb, ..)) = eb else { continue };
            if i == best {
                continue;
            }
            let r = ((pb[1] / pa[1]) / (tb / ta) - 1.0).abs();
            by.entry(term.clone()).or_default().push(r);
            by.entry("all".into()).or_default().push(r);
        }
    }
    let pct = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let i = ((v.len() as f64 * 0.8).ceil() as usize).clamp(1, v.len()) - 1;
        (v[i] * 1e3).round() as u64
    };
    let all = pct(by.get_mut("all").expect("replayed pairs"));
    let mut got: Vec<(String, u64)> = Vec::new();
    for (t, v) in by.iter_mut() {
        let b = if v.len() >= 3 { pct(v) } else { all };
        eprintln!("band {t}: {b} per mille over {} pairs", v.len());
        got.push((t.clone(), b));
    }
    for (t, pm) in crate::place::PLACE_BAND_PM {
        let want = got.iter().find(|(n, _)| n == t).map_or(all, |(_, b)| *b);
        assert_eq!(pm, want, "band {t}: committed {pm}, the replay's {want}");
    }
}
