//! Generated attention programs checked against the interpreter (finite check, bf16 cache).

use super::*;

/// One generated program's checks and coverage counts.
/// mask-frontend: the run's values of a generated program's open guard
/// sides (`testgen::flash_tm_src`: `first` is `0`, `pos[i] = i`).
pub(crate) struct RunEnv;

impl mapal_ir::GuardEnv for RunEnv {
    fn number(&self, _: u32) -> Option<u64> {
        Some(0)
    }
    fn element(&self, _: u32, index: u64) -> Option<i64> {
        Some(index as i64)
    }
}

pub(crate) fn check_flash(
    clang: &str,
    i: usize,
    p: &testgen::FlashProg,
) -> Vec<(&'static str, u64)> {
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
                    // mask-frontend: an open side at the run's values
                    // (`first` is `0`, `pos[i] = i`).
                    let env = RunEnv;
                    let cells: Vec<bool> = (r0..=r1)
                        .flat_map(|r| (l0..=l1).map(move |l| (r, l)))
                        .map(|(r, l)| gd.form.visible_with(r, l, &env).expect("run values"))
                        .collect();
                    match gd.form.classify_with((r0, r1), (l0, l1), &env) {
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
        // mask-frontend: the token-major forms streamed — walked in the
        // scores' tile order, the mask through a positions array or a
        // run-time position skipping its windows.
        out.push(("token-major streamed", u64::from(p.tm > 0)));
        // The keys and values read through a captured struct (its fields'
        // captures, `flatten_captures`).
        out.push(("struct cache streamed", u64::from(p.tm > 0 && p.scale == 1)));
        if p.tm > 0 && skip == "on" {
            out.push((
                [
                    "",
                    "skip:tm digit",
                    "skip:tm law",
                    "skip:tm load",
                    "skip:tm uniform",
                ][p.tm as usize],
                1,
            ));
        }
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
    if p.heads > 0 && (p.causal || p.tm > 0) {
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
    let skipped = lines.iter().any(|l| field(l, "skip") == "on");
    out.extend(check_finite(
        clang,
        i,
        p,
        &src,
        (b, kb, 0, 0, 0),
        skipped,
        "",
        &within,
    ));
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

/// bf16-class `finite-check`: `src` with its values written so `finite_plan`
/// cannot prove them finite — an arm never taken that overflows (every value
/// finite: the masked windows still skipped, checked at run time), or the
/// last key's last value `∞` (masked for every earlier token: the oracle's
/// `0·∞ = NaN` there, which the kernel's check reproduces by running the
/// windows it would skip). Off (`finite-check`): none skipped. Each build
/// against the oracle in every emulator mode (`ok`: the arm's comparison).
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_finite(
    clang: &str,
    i: usize,
    p: &testgen::FlashProg,
    src: &str,
    force: (u64, u64, u64, u64, u64),
    skipped: bool,
    tag: &'static str,
    ok: &dyn Fn(&str, &str) -> bool,
) -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let plain = "        v / 4.0\n    } -> vv;";
    if p.heads == 0 || !(p.causal || p.tm > 0) || !src.contains(plain) {
        return out;
    }
    let len = p.n * (p.heads / p.group) * p.d;
    let huge = "v * 1000000000000000000000000000000.0 * 1000000000000000000000000000000.0";
    let arm = |cond: String| {
        format!(
            "        ({cond}) -> {{\n            -true-> {huge};\n            -false-> v / 4.0;\n        }}\n    }} -> vv;"
        )
    };
    for (name, cond, inf) in [
        ("finite unproven", format!("t > {len}"), false),
        ("masked inf", format!("t == {}", len - 1), true),
    ] {
        let src2 = src.replace(plain, &arm(cond));
        let ir2 = lower(&src2);
        let rr2 = run(&ir2, 2_000_000_000);
        if !matches!(rr2.outcome, Outcome::Done(_)) {
            continue;
        }
        let em = emission(&ir2, Some(force), &[]);
        let checked = stream_lines(&em).iter().any(|l| field(l, "skip") == "on");
        let key = format!("{tag}check:{name}");
        out.push((
            Box::leak(key.into_boxed_str()) as &'static str,
            u64::from(checked && skipped),
        ));
        if inf {
            out.push(("check:oracle nan", u64::from(rr2.output.contains("NaN"))));
        }
        let mut arms = vec![(name, em)];
        if checked && i % 2 == 0 {
            let off = emission(&ir2, Some(force), &["finite-check"]);
            out.push((
                "check off unskipped",
                u64::from(stream_lines(&off).iter().all(|l| field(l, "skip") != "on")),
            ));
            arms.push(("finite-check off", off));
        }
        for (arm, em) in &arms {
            let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
            for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
                assert!(
                    ok(&got, &rr2.output),
                    "{arm} {mode:?}: {got:?} vs {:?}\n{src2}",
                    rr2.output
                );
                out.push(("runs", 1));
            }
        }
    }
    out
}

/// bf16-class `narrow-window`: `src` with its keys and values stored bf16
/// (a bf16 KV cache: each written element rounded once, `-> to<bf16>`; each
/// read widened exactly before its product, `-> widen<f32>`) — the struct's
/// fields too. `None` when the source has no such reads.
pub(crate) fn bf16_cache_src(src: &str) -> Option<String> {
    let mut out = String::new();
    let mut reads = 0;
    for line in src.lines() {
        let t = line.trim_start();
        let ind = &line[..line.len() - t.len()];
        let wide = [" * kr[", " * vv[", " * cache.keys[", " * cache.values["]
            .iter()
            .find(|k| t.contains(*k));
        if t.starts_with("acc, ")
            && let Some(k) = wide
        {
            let (head, rest) = t.split_once(" -> ").expect("a fold step");
            let at = rest.find(k).expect("the read");
            let (lhs, read) = (&rest[..at], &rest[at + 3..]);
            out +=
                &format!("{ind}{head} ->\n{ind}{read} -> widen<f32> -> kv16;\n{ind}{lhs} * kv16\n");
            reads += 1;
            continue;
        }
        out += line;
        out.push('\n');
    }
    if reads != 2 {
        return None;
    }
    for name in ["kr", "vv"] {
        let plain = format!("        v / 4.0\n    }} -> {name};");
        if !out.contains(&plain) {
            return None;
        }
        out = out.replace(
            &plain,
            &format!("        v / 4.0 -> to<bf16>\n    }} -> {name};"),
        );
    }
    Some(
        out.replace("keys: [f32;", "keys: [bf16;")
            .replace("values: [f32;", "values: [bf16;"),
    )
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
        "token-major streamed",
        "struct cache streamed",
        "skip:tm digit",
        "skip:tm law",
        "skip:tm load",
        "skip:tm uniform",
        "check:finite unproven",
        "check:masked inf",
        "check:oracle nan",
        "check off unskipped",
        "runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// sweep-attention-2 `stream-mma`: every printed number within the `bf16`
/// member's bound of the oracle's — `tol` an output cell's (the
/// probabilities' rounding, `2⁻⁹` relative, over values of at most 1.5),
/// `n` times it for the line that sums `n` cells (the last).
pub(crate) fn within_member(got: &str, want: &str, tol: f64, n: f64) -> bool {
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
