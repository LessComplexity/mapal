//! Streamed attention on the warp matrix instruction and k-major b on generated decode.

use super::*;

/// One generated `bf16`-member program's checks and coverage counts: the
/// stream forced at 16-row blocks and 16- or 32-key windows takes the matrix
/// product (`:mma`); `stream-mma` off, the FFMA stream of the same geometry;
/// split-KV where the windows divide; the rule's own pick recorded. Values
/// through the emulator against the interpreter within the member's bound.
pub(crate) fn check_flash_mma(
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
        // prefill-integrate: the scores' tile rows a digit permutation of
        // the group's (`out_rows`), the rows walked in the seed's order.
        out.push(("mma token-major", u64::from(p.tm > 0 && p.heads > 1)));
        out.push(("mma run-time mask", u64::from(matches!(p.tm, 3 | 4))));
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
    // bf16-class `narrow-window`: the keys and values stored bf16 — the
    // windows hold their bytes, the fragments take them as they stand; the
    // oracle reads the same rounded elements (the member's bound). Off: no
    // stream.
    if mma && let Some(src16) = bf16_cache_src(&src) {
        let ir16 = lower(&src16);
        let rr16 = run(&ir16, 2_000_000_000);
        if matches!(rr16.outcome, Outcome::Done(_)) {
            let em = emission(&ir16, Some((16, kb, 2, 0, 0)), &[]);
            let ls = stream_lines(&em);
            let on16 = ls.iter().any(|l| field(l, "stream").contains(":mma"));
            out.push(("mma narrow window", u64::from(on16)));
            out.push((
                "mma narrow window skip",
                u64::from(on16 && ls.iter().any(|l| field(l, "skip") == "on")),
            ));
            let off = emission(&ir16, Some((16, kb, 2, 0, 0)), &["narrow-window"]);
            out.push((
                "narrow-window off unstreamed",
                u64::from(stream_lines(&off).is_empty()),
            ));
            for (arm, em) in [("narrow window", em), ("narrow-window off", off)] {
                let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
                let modes: &[[(&str, &str); 2]] = if on16 && arm == "narrow window" {
                    &MODES
                } else {
                    &MODES[..1]
                };
                for (mode, (got, _)) in modes.iter().zip(ex.run_each(modes)) {
                    assert!(
                        within_member(&got, &rr16.output, tol, cells),
                        "{arm} {mode:?}: {got:?} vs {:?}\n{src16}",
                        rr16.output
                    );
                    out.push(("runs", 1));
                }
            }
        }
    }
    if mma {
        let skipped = lines.iter().any(|l| field(l, "skip") == "on");
        out.extend(check_finite(
            clang,
            i,
            p,
            &src,
            (16, kb, 2, 0, 0),
            skipped,
            "mma ",
            &|got: &str, want: &str| within_member(got, want, tol, cells),
        ));
    }
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
        "mma token-major",
        "mma run-time mask",
        "mma split-kv",
        "off streams ffma",
        "mma check:finite unproven",
        "mma check:masked inf",
        "mma narrow window",
        "mma narrow window skip",
        "narrow-window off unstreamed",
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
pub(crate) fn check_kdec(clang: &str, p: &testgen::KdecProg) -> Vec<(&'static str, u64)> {
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
