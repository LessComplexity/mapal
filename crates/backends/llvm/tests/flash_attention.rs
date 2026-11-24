//! flash-attention `stream`: attention (and every chain of the form tile →
//! cells and row folds over its lanes → tile folding over the same axis) as
//! one pass over windows of the axis — `stream_plan` (the roles, each fold's
//! laws, a rescale's shift) and its realization. Values through the GPU
//! emulator against the interpreter (free threads, serialized in thread
//! order and reversed, the block order reversed) within the default face's
//! bound, on hand-written and generated programs (`testgen::FlashProg`); the
//! OFF arms' emission; every coverage counter non-zero.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, DepLaw, StreamRole};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The emission on the rtx-5080 with the stream's geometry forced (`None`:
/// the rule's) and the rungs `off` off; the emulator's rungs off (`pdl`,
/// `streams`, `pinned`), `place` off so every capable site is on the device,
/// `horizontal` / `calls` off (a unit's own launch).
fn emission(ir: &CategoryIr, force: Option<(u64, u64, u64, u64, u64)>, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.stream_force = force;
    for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The streamed units' intent lines.
fn stream_lines(em: &Emission) -> Vec<String> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.flow kernel=") && l.contains(" stream="))
        .map(str::to_string)
        .collect()
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// Every printed number within the default face's bound of the oracle's
/// (relative `1e-4`; the windows' rescales and the free order).
fn within(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) if x.is_nan() && y.is_nan() => true,
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

/// The emulator modes: free threads, each block's threads serialized between
/// barriers in order and reversed, the blocks in reverse order.
const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

fn program(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../../../benches/algorithms/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// `pt_attn_1024_64`: one stream group over the scores' axis — the scores
/// produced window by window, the row max a fold whose running state is the
/// shift `ex` reads (a rescale), the sum a fold rescaled by it, `p` reading
/// the sum as a quotient factor, `P·V` consuming the windows with both laws;
/// the scores and probabilities dissolved. The fact arm off: no group; the
/// rescale off (`exp_rescale`): the max is a phase, refused.
#[test]
fn stream_fact_on_attention() {
    let ir = lower(&program("pt_attn_1024_64"));
    let f = ir.entry();
    let sp = ir.stream_plan(f);
    assert_eq!(sp.groups.len(), 1, "{sp:?}");
    let g = &sp.groups[0];
    assert_eq!(g.axis, 1024);
    assert_eq!(g.passes, 1);
    let unit = &ir.unit_plan(f).groups[g.group];
    let name = |k: usize| {
        ir.object(unit.members[k].out)
            .and_then(|o| o.name.clone())
            .unwrap_or_default()
    };
    let roles: Vec<(String, &StreamRole)> = g
        .roles
        .iter()
        .enumerate()
        .map(|(k, r)| (name(k), r))
        .collect();
    let role = |n: &str| roles.iter().find(|(x, _)| x == n).map(|x| x.1).unwrap();
    assert_eq!(role("s"), &StreamRole::Produce);
    let k_mx = roles.iter().position(|(x, _)| x == "mx").unwrap();
    let k_den = roles.iter().position(|(x, _)| x == "den").unwrap();
    match role("ex") {
        StreamRole::Cell { subst } => assert_eq!(subst[..], [(1, k_mx, DepLaw::Rescale)]),
        r => panic!("{r:?}"),
    }
    match role("p") {
        StreamRole::Cell { subst } => {
            assert_eq!(subst[..], [(1, k_den, DepLaw::Factor { div: true })])
        }
        r => panic!("{r:?}"),
    }
    match role("den") {
        StreamRole::Fold { laws } => {
            assert_eq!(laws.len(), 1);
            assert_eq!((laws[0].value, laws[0].law), (k_mx, DepLaw::Rescale));
        }
        r => panic!("{r:?}"),
    }
    match role("o") {
        StreamRole::Consume { laws } => {
            let l: Vec<(usize, DepLaw)> = laws.iter().map(|l| (l.value, l.law)).collect();
            assert!(l.contains(&(k_mx, DepLaw::Rescale)), "{l:?}");
            assert!(l.contains(&(k_den, DepLaw::Factor { div: true })), "{l:?}");
        }
        r => panic!("{r:?}"),
    }
    let dissolved: Vec<String> = g.dissolved.iter().map(|&k| name(k)).collect();
    assert_eq!(dissolved, ["s", "ex", "p"]);
    assert_eq!(g.dissolved_bytes, 3 * 1024 * 1024 * 4);
    // Stored: the second query reads it.
    assert_eq!(ir.stream_plan(f), sp);
    // The fact arm off, and the rescale off.
    let off = ir.refacted(mapal_ir::FactArms {
        stream: false,
        ..ir.fact_arms()
    });
    assert!(off.stream_plan(off.entry()).groups.is_empty());
    let off = ir.refacted(mapal_ir::FactArms {
        exp_rescale: false,
        ..ir.fact_arms()
    });
    let sp = off.stream_plan(off.entry());
    assert!(sp.groups.is_empty());
    assert_eq!(
        sp.refused.iter().map(|r| r.1).collect::<Vec<_>>(),
        ["phase"]
    );
}

/// The streamed unit's emission and values on `pt_attn_1024_64` scaled down
/// (`px_attn_64_16`'s shape): one kernel where there were three, nothing
/// axis-wide stored, the values the interpreter's within the bound in every
/// emulator mode; the OFF arm (`--gpu-off=stream`) is the split emission.
#[test]
fn streamed_attention_values() {
    let src = program("px_attn_64_16");
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("interp")
    };
    let on = emission(&ir, Some((16, 16, 0, 0, 0)), &[]);
    let lines = stream_lines(&on);
    assert_eq!(lines.len(), 1, "{}", on.device.as_deref().unwrap_or(""));
    assert_eq!(field(&lines[0], "store"), "o");
    assert_eq!(field(&lines[0], "dissolved"), "s,ex,p");
    assert_eq!(field(&lines[0], "windows"), "4");
    let off = emission(&ir, Some((16, 16, 0, 0, 0)), &["stream"]);
    assert!(stream_lines(&off).is_empty());
    let Some(clang) = emu::clang() else {
        return;
    };
    for em in [&on, &off] {
        let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            assert!(
                within(&got, &rr.output),
                "{mode:?}: {got:?} vs {:?}",
                rr.output
            );
        }
    }
}

/// review-fixes: **the stream is priced on every part** — a part whose
/// profile measured only some plan rates (the 3090, A100, H100 SXM and PCIe)
/// prices the streamed unit on the rates its properties derive
/// (`TargetProfile::priced`) instead of keeping the split realization
/// unweighed (`stream=s:unpriced`, sweep-attention's replay). The 2048-token
/// prefill streams on each by its price; the scaled-down attention's values
/// under each part's own rule match the interpreter in every emulator mode.
#[test]
fn stream_priced_on_every_part() {
    let on = |ir: &CategoryIr, target: &'static str| {
        let mut gpu = GpuArms::default();
        gpu.on.flow = true;
        for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "generic",
            target,
            gpu,
            ..EmitOpts::default()
        };
        emit_detailed(ir, &opts).expect("emits")
    };
    let pre = lower(&program("pt_llm_pre_t2048"));
    // attention-chains: the window's products and their shared loads priced
    // half in series at the broadcast-load rate — the A100's 64 FP32 lanes
    // no longer hide the loads of a one-row-a-thread geometry (measured 1.5×
    // a four-row one): its streams hold four rows a thread, as the GDDR
    // parts' did.
    let a128 = lower(&program("pt_attn_2048_128"));
    for (ir, part) in [(&pre, "a100"), (&a128, "a100")] {
        let em = on(ir, part);
        let l = stream_lines(&em);
        assert_eq!(l.len(), 1, "{part}: {}", em.device.as_deref().unwrap_or(""));
        assert!(field(&l[0], "stream").contains(":rt4"), "{part}: {}", l[0]);
    }
    let small = lower(&program("px_attn_64_16"));
    let rr = run(&small, 2_000_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("interp")
    };
    let clang = emu::clang();
    for part in ["rtx-3090", "a100", "h100", "h100-pcie"] {
        let em = on(&pre, part);
        let dev = em.device.as_deref().unwrap_or("");
        assert!(!dev.contains("stream=s:unpriced"), "{part}");
        assert_eq!(stream_lines(&em).len(), 1, "{part}: the prefill streams");
        let Some(clang) = clang.as_deref() else {
            continue;
        };
        let em = on(&small, part);
        let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            assert!(
                within(&got, &rr.output),
                "{part} {mode:?}: {got:?} vs {:?}",
                rr.output
            );
        }
    }
}

/// One generated program's checks and coverage counts.
fn check_flash(clang: &str, i: usize, p: &testgen::FlashProg) -> Vec<(&'static str, u64)> {
    let src = testgen::flash_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    if !matches!(rr.outcome, Outcome::Done(_)) {
        return Vec::new();
    }
    let mut out = vec![("programs", 1)];
    let f = ir.entry();
    let sp = ir.stream_plan(f);
    out.push(("fact groups", sp.groups.len() as u64));
    for r in &sp.refused {
        out.push((Box::leak(format!("refused:{}", r.1).into_boxed_str()), 1));
    }
    for g in &sp.groups {
        for r in &g.roles {
            match r {
                StreamRole::Fold { laws } | StreamRole::Consume { laws } => {
                    for l in laws {
                        out.push(match (l.law, l.rescale.as_ref().map(|a| a.up)) {
                            (DepLaw::Rescale, Some(true)) => ("law:rescale up", 1),
                            (DepLaw::Rescale, _) => ("law:rescale down", 1),
                            (DepLaw::Factor { .. }, _) => ("law:factor", 1),
                            _ => ("law:shift", 1),
                        });
                    }
                }
                StreamRole::Row { .. } => out.push(("row role", 1)),
                _ => {}
            }
        }
    }
    // The forced geometry: `b` rows dividing `n`, windows of 16 (tails where
    // they do not divide the keys) or the one window of a short row.
    let b = (0..=4)
        .rev()
        .map(|e| 1u64 << e)
        .find(|b| p.n.is_multiple_of(*b))
        .unwrap();
    // The axis: the keys (the model-scale form's tokens).
    let axis = if p.heads > 0 { p.n } else { p.w };
    // flash-attention F5: each box's class against every cell of it (a
    // visible box's cells all compute the fold, a masked box's none).
    for g in &sp.groups {
        let Some(gd) = &g.guard else {
            continue;
        };
        let rows = ir.unit_plan(f).groups[g.group].rows;
        for (bs, ws) in [(4u64, 8u64), (8, 16), (16, 4)] {
            for r0 in (0..rows).step_by(bs as usize) {
                for l0 in (0..g.axis).step_by(ws as usize) {
                    let (r1, l1) = ((r0 + bs).min(rows) - 1, (l0 + ws).min(g.axis) - 1);
                    let cells: Vec<bool> = (r0..=r1)
                        .flat_map(|r| (l0..=l1).map(move |l| (r, l)))
                        .map(|(r, l)| gd.form.visible(r, l))
                        .collect();
                    match gd.form.classify((r0, r1), (l0, l1)) {
                        mapal_ir::BoxClass::Visible => assert!(cells.iter().all(|&c| c), "{src}"),
                        mapal_ir::BoxClass::Masked => assert!(cells.iter().all(|&c| !c), "{src}"),
                        mapal_ir::BoxClass::Mixed => {}
                    }
                    out.push(("boxes checked", 1));
                }
            }
        }
        out.push((
            Box::leak(format!("guard:{}", if gd.skip { "skip" } else { gd.why }).into_boxed_str()),
            1,
        ));
    }
    // flash-attention F1/F2: the model-scale form's digit rows and panels.
    for g in &sp.groups {
        let unit = &ir.unit_plan(f).groups[g.group];
        out.push((
            "digit rows",
            u64::from(unit.members.iter().any(|m| m.row_of.is_some())),
        ));
        out.push(("panel batch", u64::from(unit.panel < unit.rows)));
        out.push(("extended group", u64::from(unit.extended)));
        // sweep-attention: members after the consumer the pass leaves to
        // their own launches (a head loop's carried sum).
        out.push((
            "trailing members",
            u64::from(unit.members.len() > g.roles.len()),
        ));
    }
    // A row a block takes a warp of lanes: windows of 32.
    let kb = match (axis < 16, b) {
        (true, _) => axis.next_power_of_two().max(4),
        (false, 1) => 32,
        _ => 16,
    };
    let on = emission(&ir, Some((b, kb, 0, 0, 0)), &[]);
    let lines = stream_lines(&on);
    let host_line = on
        .text
        .lines()
        .find(|l| l.contains("; mapal.flow.host"))
        .unwrap_or("");
    if lines.is_empty() && !sp.groups.is_empty() {
        let why = host_line
            .split_whitespace()
            .find_map(|f| f.strip_prefix("stream="))
            .or_else(|| {
                host_line
                    .split_whitespace()
                    .find_map(|f| f.strip_prefix("row_block="))
            })
            .unwrap_or("none");
        let why = why.rsplit(':').next().unwrap_or(why);
        out.push((
            Box::leak(format!("backend refused:{why}").into_boxed_str()),
            1,
        ));
    }
    for l in &lines {
        out.push(("streamed", 1));
        let windows: u64 = field(l, "windows").parse().unwrap();
        out.push(("windows>1", u64::from(windows > 1)));
        out.push(("tails", u64::from(!axis.is_multiple_of(kb))));
        out.push(("model-scale streamed", u64::from(p.heads > 0)));
        // F5/F6: a causal mask's wholly masked windows skipped (or why not);
        // a mask inside `exp`'s reach (`−8`) fails the kernel's vote.
        let skip = field(l, "skip");
        out.push((Box::leak(format!("skip:{skip}").into_boxed_str()), 1));
        out.push((
            "skip with near mask",
            u64::from(skip == "on" && p.mask == 1),
        ));
        out.push(("causal streamed", u64::from(p.heads > 0 && p.causal)));
        out.push(("cp", u64::from(field(l, "stream").ends_with(":cp"))));
        out.push(("seed inside", u64::from(p.seed == 1)));
        out.push(("den seed one", u64::from(p.den_one)));
        out.push(("stored members", u64::from(field(l, "store").contains(';'))));
        out.push(("odd widths", u64::from(p.d % 4 != 0 || p.c % 4 != 0)));
        out.push(("one row a block", u64::from(b == 1)));
        // sweep-attention `unit-uniform`: the head-group loop's operands
        // moved by its counter at the launch.
        out.push(("loop streamed", u64::from(p.lp)));
        // The pass over the members before the trailing ones (their own
        // launches after it, reading the consumer's stored rows).
        let trailing = sp
            .groups
            .iter()
            .any(|g| ir.unit_plan(f).groups[g.group].members.len() > g.roles.len());
        out.push(("trailing streamed", u64::from(trailing)));
        out.push(("moved operands", u64::from(!field(l, "moved").is_empty())));
    }
    if p.lp {
        out.push(("loop programs", 1));
        if lines.is_empty() {
            let why: Vec<&str> = sp.refused.iter().map(|r| r.1).collect();
            let host = host_line
                .split_whitespace()
                .filter(|f| f.starts_with("stream=") || f.starts_with("row_block="))
                .collect::<Vec<_>>()
                .join(",");
            out.push((
                Box::leak(format!("loop unstreamed:{}|{host}", why.join(",")).into_boxed_str()),
                1,
            ));
        }
    }
    let mut builds = vec![("on", on)];
    // R3 split-KV: each row block's windows over 2 or 4 blocks (the last one
    // to finish combines), where they divide the windows.
    let nbw = axis.div_ceil(kb);
    if let Some(sp) = [4u64, 2]
        .into_iter()
        .find(|s| nbw.is_multiple_of(*s) && nbw > 1)
    {
        let em = emission(&ir, Some((b, kb, 0, sp, 0)), &[]);
        let split = stream_lines(&em)
            .iter()
            .any(|l| field(l, "stream").ends_with(&format!(":s{sp}")));
        if split {
            out.push(("split-kv streamed", 1));
            builds.push(("split", em));
        }
    }
    // attention-chains: 4-lane row groups (`threads = 4b`, a row a group):
    // a consumer whose columns are whole 16-byte chunks of the group takes
    // them interleaved across its lanes — two chunks a lane at width 32.
    if b >= 8 && axis >= 16 {
        let em = emission(&ir, Some((b, kb, 1, 0, 4 * b)), &[]);
        let ls = stream_lines(&em);
        if ls.iter().any(|l| field(l, "stream").contains("g4t")) {
            out.push(("narrow groups streamed", 1));
            out.push(("contig consumer", u64::from(p.c.is_multiple_of(16))));
            out.push(("chunks>1", u64::from(p.c.is_multiple_of(16) && p.c >= 32)));
            builds.push(("narrow groups", em));
        }
    }
    if i % 3 == 0 {
        builds.push((
            "stream off",
            emission(&ir, Some((b, kb, 0, 0, 0)), &["stream"]),
        ));
    }
    if i % 3 == 1 {
        builds.push(("stream fact off", emission(&ir, None, &["stream-fact"])));
    }
    if p.heads > 0 && p.causal {
        // F5 off: every window computed; F7 off: nothing finite, none skipped.
        builds.push((
            "guard-box off",
            emission(&ir, Some((b, kb, 0, 0, 0)), &["guard-box"]),
        ));
        builds.push((
            "finite off",
            emission(&ir, Some((b, kb, 0, 0, 0)), &["finite"]),
        ));
    }
    if p.lp {
        // `unit-uniform` off: no unit takes the moved operands (the split
        // realization, every launch its own).
        let em = emission(&ir, Some((b, kb, 0, 0, 0)), &["unit-uniform"]);
        out.push((
            "unit-uniform off unstreamed",
            u64::from(stream_lines(&em).is_empty()),
        ));
        builds.push(("unit-uniform off", em));
    }
    if i % 3 == 2 {
        // `cp.async` off: each thread loads and stores the windows.
        let em = emission(&ir, Some((b, kb, 0, 0, 0)), &["async"]);
        if !stream_lines(&em).is_empty() {
            out.push(("sync windows", 1));
        }
        builds.push(("async off", em));
    }
    for (arm, em) in &builds {
        let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        let modes: &[[(&str, &str); 2]] = match *arm {
            "on" | "split" | "narrow groups" if !lines.is_empty() => &MODES,
            _ => &MODES[..1],
        };
        for (mode, (got, _)) in modes.iter().zip(ex.run_each(modes)) {
            assert!(
                within(&got, &rr.output),
                "{arm} {mode:?}: {got:?} vs {:?}\n{src}",
                rr.output
            );
            out.push(("runs", 1));
        }
    }
    out
}

/// Generated attention programs (`testgen::flash_strategy`): 1–64 query
/// rows, 4–256 keys (tails), head widths 3–16 and value widths 3–32 (4-lane
/// row groups: the interleaved 16-byte consumer chunks), the exponent
/// scaled or not, the extreme a max or a min (the rescale down), its seed
/// below or inside the scores, the sum's seed 0 or 1, stored scores /
/// extreme / sum, and the forms the fact refuses (a `÷ den` after the
/// windows, a stored probability, an exponent the extreme does not shift).
/// Every coverage counter is non-zero.
#[test]
fn stream_on_generated_attention() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::FlashProg> = std::iter::repeat_with(|| {
        testgen::flash_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .filter(|p| match p.heads {
        0 => p.n * p.w <= 4096,
        h => h * p.n * p.n <= 16384 && p.n >= 2,
    })
    .take(128)
    .collect();
    let results = testgen::par_check(&progs, |i, p| check_flash(&clang, i, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("flash-attention generated: {c:?}");
    for key in [
        "programs",
        "fact groups",
        "refused:phase",
        "refused:escape",
        "law:rescale up",
        "law:rescale down",
        "law:factor",
        "row role",
        "streamed",
        "windows>1",
        "tails",
        "cp",
        "seed inside",
        "den seed one",
        "stored members",
        "odd widths",
        "one row a block",
        "sync windows",
        "digit rows",
        "panel batch",
        "extended group",
        "model-scale streamed",
        "causal streamed",
        "skip:on",
        "skip:none",
        "skip with near mask",
        "boxes checked",
        "guard:skip",
        "split-kv streamed",
        "narrow groups streamed",
        "contig consumer",
        "chunks>1",
        "loop programs",
        "loop streamed",
        "trailing members",
        "trailing streamed",
        "moved operands",
        "unit-uniform off unstreamed",
        "runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// sweep-attention-2 `stream-mma`: every printed number within the `bf16`
/// member's bound of the oracle's — `tol` an output cell's (the
/// probabilities' rounding, `2⁻⁹` relative, over values of at most 1.5),
/// `n` times it for the line that sums `n` cells (the last).
fn within_member(got: &str, want: &str, tol: f64, n: f64) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    let last = w.len().saturating_sub(1);
    g.len() == w.len()
        && g.iter().zip(&w).enumerate().all(|(i, (a, b))| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) => {
                        let t = if i == last { tol * n } else { tol };
                        (x - y).abs() <= t + 1e-4 * y.abs()
                    }
                    _ => false,
                }
        })
}

/// One generated `bf16`-member program's checks and coverage counts: the
/// stream forced at 16-row blocks and 16- or 32-key windows takes the matrix
/// product (`:mma`); `stream-mma` off, the FFMA stream of the same geometry;
/// split-KV where the windows divide; the rule's own pick recorded. Values
/// through the emulator against the interpreter within the member's bound.
fn check_flash_mma(clang: &str, i: usize, p: &testgen::FlashProg) -> Vec<(&'static str, u64)> {
    let src = testgen::flash_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    if !matches!(rr.outcome, Outcome::Done(_)) {
        return Vec::new();
    }
    let mut out = vec![("programs", 1)];
    let kb = if i % 2 == 1 && p.n >= 32 { 32 } else { 16 };
    let on = emission(&ir, Some((16, kb, 2, 0, 0)), &[]);
    let lines = stream_lines(&on);
    let mma = lines.iter().any(|l| field(l, "stream").contains(":mma"));
    out.push(("mma streamed", u64::from(mma)));
    if !mma {
        let host = on
            .text
            .lines()
            .find(|l| l.contains("; mapal.flow.host"))
            .unwrap_or("")
            .split_whitespace()
            .filter(|f| f.starts_with("stream=") || f.starts_with("row_block="))
            .collect::<Vec<_>>()
            .join(",");
        out.push((
            Box::leak(format!("mma unstreamed:{host}|{}", lines.len()).into_boxed_str()),
            1,
        ));
    }
    if mma {
        out.push(("mma tails", u64::from(!p.n.is_multiple_of(kb))));
        out.push(("mma causal", u64::from(p.causal)));
        out.push((
            "mma skip",
            u64::from(lines.iter().any(|l| field(l, "skip") == "on")),
        ));
        out.push(("mma loop", u64::from(p.lp)));
        out.push(("mma d32", u64::from(p.d == 32)));
        out.push(("mma group>1", u64::from(p.group > 1)));
        out.push(("mma kb32", u64::from(kb == 32)));
    }
    let off = emission(&ir, Some((16, kb, 2, 0, 0)), &["stream-mma"]);
    let off_lines = stream_lines(&off);
    out.push((
        "off streams ffma",
        u64::from(
            !off_lines.is_empty()
                && off_lines
                    .iter()
                    .all(|l| !field(l, "stream").contains(":mma")),
        ),
    ));
    let mut builds = vec![("on", on), ("stream-mma off", off)];
    let nbw = p.n.div_ceil(kb);
    if nbw > 1 && nbw.is_multiple_of(2) && !p.causal {
        let em = emission(&ir, Some((16, kb, 2, 2, 0)), &[]);
        if stream_lines(&em).iter().any(|l| {
            let s = field(l, "stream");
            s.contains(":mma") && s.ends_with(":s2")
        }) {
            out.push(("mma split-kv", 1));
            builds.push(("split", em));
        }
    }
    // The member's bound over these data: the scores exact (multiples of
    // `¼`, exact in `bf16`, summed exactly), each probability rounded once
    // (`2⁻⁹` relative) against values of at most 1.5 — twice that, over the
    // `H ÷ G` trips a loop's output sums.
    let trips = if p.lp {
        (p.heads / p.group) as f64
    } else {
        1.0
    };
    let tol = 2.0 * 2f64.powi(-9) * 1.5 * trips;
    let cells = (p.n * p.d * if p.lp { p.group } else { p.heads }) as f64;
    for (arm, em) in &builds {
        let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        let modes: &[[(&str, &str); 2]] = match *arm {
            "on" if mma => &MODES,
            _ => &MODES[..1],
        };
        for (mode, (got, _)) in modes.iter().zip(ex.run_each(modes)) {
            assert!(
                within_member(&got, &rr.output, tol, cells),
                "{arm} {mode:?}: {got:?} vs {:?}\n{src}",
                rr.output
            );
            out.push(("runs", 1));
        }
    }
    out
}

/// sweep-attention-2 `stream-mma`: generated attention programs whose two
/// products admit the `bf16` member (`testgen::flash_half_strategy`) — the
/// stream on the warp matrix product, its OFF arm and split-KV, values
/// within the member's bound in every emulator mode. Every coverage counter
/// is non-zero.
#[test]
fn stream_mma_on_generated_attention() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::FlashProg> = std::iter::repeat_with(|| {
        testgen::flash_half_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .filter(|p| p.heads * p.n * p.n <= 16384)
    .take(48)
    .collect();
    let results = testgen::par_check(&progs, |i, p| check_flash_mma(&clang, i, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("stream-mma generated: {c:?}");
    for key in [
        "programs",
        "mma streamed",
        "mma tails",
        "mma causal",
        "mma skip",
        "mma loop",
        "mma d32",
        "mma group>1",
        "mma kb32",
        "mma split-kv",
        "off streams ffma",
        "runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
    // The rule's own pick (the plan model, no forced geometry; `place` on:
    // its rates) at a size it streams in 16-row blocks: 512 tokens, 8 heads
    // of 64 in groups of 4, causal — the matrix product; `stream-mma` off,
    // the FFMA stream.
    let p = testgen::FlashProg {
        n: 512,
        w: 512,
        d: 64,
        c: 64,
        heads: 8,
        group: 4,
        causal: true,
        lp: false,
        ..progs[0].clone()
    };
    let ir = lower(&testgen::flash_src(&p));
    let placed = |off: &[&str]| {
        let mut gpu = GpuArms::default();
        gpu.on.flow = true;
        for r in off {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        };
        stream_lines(&emit_detailed(&ir, &opts).expect("emits"))
    };
    let on = placed(&[]);
    assert!(
        on.iter().any(|l| field(l, "stream").contains(":mma")),
        "{on:?}"
    );
    let off = placed(&["stream-mma"]);
    assert!(
        !off.is_empty() && off.iter().all(|l| !field(l, "stream").contains(":mma")),
        "{off:?}"
    );
}

/// One generated decode-step program's checks and coverage counts
/// (`kmajor-b`): the tiles whose k-major `b` is copied along its rows
/// (`kvb=1`), the OFF arm's none; values through the emulator against the
/// interpreter (exact: every sum is exact in any order).
fn check_kdec(clang: &str, p: &testgen::KdecProg) -> Vec<(&'static str, u64)> {
    let src = testgen::kdec_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    if !matches!(rr.outcome, Outcome::Done(_)) {
        return Vec::new();
    }
    let kvb = |em: &Emission| {
        em.device
            .as_deref()
            .unwrap_or("")
            .lines()
            .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" kvb=1"))
            .count() as u64
    };
    let on = emission(&ir, None, &[]);
    let off = emission(&ir, None, &["kmajor-b"]);
    // gapfix `kvec-swizzle`: the copies' scattered stores spread over the
    // banks; its OFF arm the dense panel (the same plans, other text).
    let unswz = emission(&ir, None, &["kvec-swizzle"]);
    let (n_on, n_off) = (kvb(&on), kvb(&off));
    assert_eq!(n_off, 0, "kmajor-b off: no vector copies\n{src}");
    assert_eq!(kvb(&unswz), n_on, "kvec-swizzle moves no plan\n{src}");
    let swizzled = n_on > 0 && unswz.device != on.device;
    let mut out = vec![
        ("programs", 1),
        ("kvb tiles", n_on),
        ("kvb programs", u64::from(n_on > 0)),
        ("kvb narrow", u64::from(n_on > 0 && p.narrow)),
        ("kvb f32", u64::from(n_on > 0 && !p.narrow)),
        ("kvb group>1", u64::from(n_on > 0 && p.g > 1)),
        ("kvb d128", u64::from(n_on > 0 && p.d == 128)),
        ("kvb swizzled", u64::from(swizzled)),
    ];
    let mut arms = vec![("on", &on), ("kmajor-b off", &off)];
    if swizzled {
        arms.push(("kvec-swizzle off", &unswz));
    }
    for (arm, em) in arms {
        let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        let modes: &[[(&str, &str); 2]] = match arm {
            "on" if n_on > 0 => &MODES,
            _ => &MODES[..1],
        };
        for (mode, (got, _)) in modes.iter().zip(ex.run_each(modes)) {
            assert!(
                within(&got, &rr.output),
                "{arm} {mode:?}: {got:?} vs {:?}\n{src}",
                rr.output
            );
            out.push(("runs", 1));
        }
    }
    out
}

/// sweep-attention-2 `kmajor-b`: generated decode-step scores
/// (`testgen::kdec_strategy`: 1–4 sequences, 4–8 query heads in groups of
/// 1–4, 64–512 keys of 16–128 elements, an `f32` or `bf16` cache, a run-time
/// position) — the register-staged tile copying its k-major `b` along its
/// rows, its OFF arm, values in every emulator mode. Every coverage counter
/// is non-zero.
#[test]
fn kmajor_b_on_generated_decode() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::KdecProg> = std::iter::repeat_with(|| {
        testgen::kdec_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(40)
    .collect();
    let results = testgen::par_check(&progs, |_, p| check_kdec(&clang, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("kmajor-b generated: {c:?}");
    for key in [
        "programs",
        "kvb tiles",
        "kvb programs",
        "kvb narrow",
        "kvb f32",
        "kvb group>1",
        "kvb d128",
        "kvb swizzled",
        "runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}
