//! The measured pick: stored and reused, the winner runs last, looped sites, cold flag.

use super::*;

/// The first generated program (geom strategy, on `target`) whose one table
/// holds at least `n` candidates and whose every table is `RERUN`.
pub(crate) fn tabled_program(target: &'static str, n: usize) -> (String, Emission) {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::geom_prog_strategy();
    for _ in 0..200 {
        let (src, _) = testgen::geom_src(&strat.new_tree(&mut runner).unwrap().current());
        let ir = lower(&src);
        let e = emission(&ir, target, EmitOpts::default(), 8, &[], &[]);
        let ts = tables(&e.text);
        if ts.len() == 1 && ts[0].rows.len() >= n && ts[0].flags & 1 == 1 {
            return (src, e);
        }
    }
    panic!("no generated program with a {n}-candidate table on {target}");
}

/// `(src, pick)` of the trace's `op=site` lines.
pub(crate) fn decisions(err: &str) -> Vec<(String, u32)> {
    err.lines()
        .filter(|l| l.starts_with("mapal_gpu_trace op=site "))
        .map(|l| {
            (
                field(l, "src=").to_string(),
                field(l, "pick=").parse().unwrap(),
            )
        })
        .collect()
}

/// B4 decision under the fake clock (`MAPAL_EMU_NS`), through the shipped
/// decision code (`mapal_choice.h`).
#[test]
fn the_measured_pick_is_stored_and_reused() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice decision: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let (src, e) = tabled_program("rtx-4090", 3);
    let n = tables(&e.text)[0].rows.len();
    let built = emu::build_emulated(&clang, &e.text, e.device.as_deref().unwrap());
    let want = built.run(&[("MAPAL_GPU_PICK", "0")]).0;
    let dir = tempfile::tempdir().unwrap();
    let wdir = dir.path().join("w");
    let wis = wdir.to_str().unwrap();
    let run = |ns: &str, wis: &str, extra: &[(&str, &str)]| {
        let mut env = vec![
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", wis),
            ("MAPAL_GPU_TRACE", "1"),
        ];
        env.extend_from_slice(extra);
        let (out, err) = built.run(&env);
        (out, err.clone(), decisions(&err), emu::totals(&err)[2])
    };
    let file = || std::fs::read_to_string(wdir.join("wisdom-emu-sm0.tsv")).unwrap_or_default();
    // Candidate 1 is 20 % faster than candidate 0 (1 ms: five reps).
    let ns = "1000000,800000,1200000";
    let (out, err, d, trial_launches) = run(ns, wis, &[]);
    assert_eq!(d, vec![("trial".into(), 1)], "{err}\n{src}");
    assert_eq!(out, want, "the winner's output");
    let trials = err.lines().filter(|l| l.contains("op=trial ")).count();
    // per-site-choice D3: a non-zero winner is confirmed head to head (3
    // alternating pairs with candidate 0).
    assert_eq!(
        trials,
        5 * n + 6,
        "five rotated reps of every candidate, three pairs\n{err}"
    );
    let site = err
        .lines()
        .find(|l| l.starts_with("mapal_gpu_trace op=site "))
        .unwrap();
    assert!(
        field(site, "reps=") == "5"
            && !field(site, "ns0=").is_empty()
            && !field(site, "nsw=").is_empty(),
        "{site}"
    );
    let w = file();
    assert_eq!(w.lines().count(), 1, "{w}");
    let cols: Vec<&str> = w.trim_end().split('\t').collect();
    assert_eq!(
        (cols[2], cols[3], cols[4]),
        (n.to_string().as_str(), "1", "5"),
        "{w}"
    );
    assert_eq!(
        cols.len(),
        5 + 2 * n,
        "key site n pick reps ns.. names..\n{w}"
    );
    // The next run reads the stored winner and launches it alone.
    let (out, err, d, launches) = run(ns, wis, &[]);
    assert_eq!(d, vec![("wisdom".into(), 1)], "{err}");
    // gaps-disasm C1 `load-dispatch`: the module registers its table before
    // the init, and the stored decision is made there, not at the first call
    // (a trial still decides at its call, above).
    assert!(
        e.text
            .contains("call void @mapal_gpu_sites(ptr @mapal_sites, i32 1)"),
        "{}",
        e.text
    );
    let at_load = |err: &str| {
        err.lines()
            .filter(|l| l.starts_with("mapal_gpu_trace op=site ") && l.ends_with(" at=load"))
            .count()
    };
    assert_eq!(at_load(&err), 1, "{err}");
    let fresh = dir.path().join("load-fresh");
    let (_, err_fresh, d_fresh, _) = run(ns, fresh.to_str().unwrap(), &[]);
    assert_eq!(d_fresh, vec![("trial".into(), 1)], "{err_fresh}");
    assert_eq!(at_load(&err_fresh), 0, "{err_fresh}");
    let (_, err_forced, _, _) = run(ns, wis, &[("MAPAL_GPU_PICK", "rules")]);
    assert_eq!(at_load(&err_forced), 1, "{err_forced}");
    assert_eq!(out, want);
    let solo = built.run(&[("MAPAL_GPU_PICK", "1")]).1;
    assert_eq!(launches, emu::totals(&solo)[2], "only the winner launches");
    assert!(trial_launches > launches);
    assert_eq!(file().lines().count(), 1, "nothing appended");
    // A candidate within 3 % keeps candidate 0.
    let close = dir.path().join("close");
    let (_, err, d, _) = run("1000000,980000,990000", close.to_str().unwrap(), &[]);
    assert_eq!(d, vec![("trial".into(), 0)], "{err}");
    // A corrupt wisdom file (garbage, a pick past the table) is ignored.
    let bad = dir.path().join("bad");
    std::fs::create_dir_all(&bad).unwrap();
    let key = cols[0];
    std::fs::write(
        bad.join("wisdom-emu-sm0.tsv"),
        format!("not a line\n\u{1}\u{2}\n{key}\t{}\t{n}\t{n}\t5\n", cols[1]),
    )
    .unwrap();
    let (_, err, d, _) = run(ns, bad.to_str().unwrap(), &[]);
    assert_eq!(d, vec![("trial".into(), 1)], "{err}");
    // Forced and off: no trial, nothing written.
    let (_, _, d, _) = run(ns, close.to_str().unwrap(), &[("MAPAL_GPU_PICK", "rules")]);
    assert_eq!(d, vec![("forced".into(), 0)]);
    // D3: `auto` is the default path; `tune` ignores the stored entry, trials
    // again and appends; anything else is an error, never candidate 0.
    let (_, _, d, _) = run(ns, wis, &[("MAPAL_GPU_PICK", "auto")]);
    assert_eq!(d, vec![("wisdom".into(), 1)]);
    let (_, _, d, _) = run(ns, wis, &[("MAPAL_GPU_PICK", "tune")]);
    assert_eq!(d, vec![("trial".into(), 1)]);
    assert_eq!(file().lines().count(), 2, "the re-tune appended");
    let typo = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        built.run(&[("MAPAL_EMU_NS", ns), ("MAPAL_GPU_PICK", "rule")])
    }));
    assert!(typo.is_err(), "MAPAL_GPU_PICK=rule is an error");
    let (_, _, d, _) = run(ns, "off", &[]);
    assert_eq!(d, vec![("rules".into(), 0)]);
    // A changed table (two candidates: other bytes) re-tunes against the
    // same wisdom file.
    let ir = lower(&src);
    let e2 = emission(&ir, "rtx-4090", EmitOpts::default(), 2, &[], &[]);
    let built2 = emu::build_emulated(&clang, &e2.text, e2.device.as_deref().unwrap());
    let err = built2
        .run(&[
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", wis),
            ("MAPAL_GPU_TRACE", "1"),
        ])
        .1;
    assert_eq!(decisions(&err), vec![("trial".into(), 1)], "{err}");
    assert_eq!(
        file().lines().count(),
        3,
        "the new table's decision appended"
    );
    // D4: a table without `RERUN` (the flag an aliased output clears,
    // `apart`) is never trialled.
    // (Its other flags kept: a hoisted zero's `ZEROED`, gaps-disasm C1.)
    let t0 = &tables(&e.text)[0];
    let host = e.text.replacen(
        &format!("@mapal_gpu_site(i64 {}, i32 {},", t0.site as i64, t0.flags),
        &format!(
            "@mapal_gpu_site(i64 {}, i32 {},",
            t0.site as i64,
            t0.flags & !1
        ),
        1,
    );
    assert_ne!(host, e.text);
    let built0 = emu::build_emulated(&clang, &host, e.device.as_deref().unwrap());
    let fresh = dir.path().join("norerun");
    let err = built0
        .run(&[
            ("MAPAL_EMU_NS", ns),
            ("MAPAL_WISDOM", fresh.to_str().unwrap()),
            ("MAPAL_GPU_TRACE", "1"),
        ])
        .1;
    assert_eq!(decisions(&err), vec![("rules".into(), 0)], "{err}");
    assert!(!err.contains("op=trial "), "{err}");
    assert!(!fresh.join("wisdom-emu-sm0.tsv").exists());
}

/// per-site-choice D4: on a program whose candidates differ in bits
/// (non-integer operands), the output after a trial is the winner's and the
/// site's last launch is the winner's kernel — whether the winner ran last in
/// the trial or not.
#[test]
fn the_winner_runs_last() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice winner: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let strat = testgen::order_prog_strategy();
    for z in 0..60 {
        let (src, _) = testgen::geom_src(&strat.new_tree(&mut runner).unwrap().current());
        let target = PARTS[z % PARTS.len()];
        let e = emission(&lower(&src), target, EmitOpts::default(), 2, &[], &[]);
        let ts = tables(&e.text);
        if ts.len() != 1 || ts[0].flags & 1 == 0 {
            continue;
        }
        let built = emu::build_emulated(&clang, &e.text, e.device.as_deref().unwrap());
        let outs: Vec<String> = ["0", "1"]
            .iter()
            .map(|p| untimed(&built.run(&[("MAPAL_GPU_PICK", p)]).0))
            .collect();
        if outs[0] == outs[1] {
            continue;
        }
        let ids: Vec<String> = ts[0]
            .rows
            .iter()
            .map(|r| format!("kernel=mapal_tile_{:016x}", r[0] as u64))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        for (ns, pick) in [("1000000,800000", 1), ("800000,1000000", 0)] {
            let w = dir.path().join(format!("w{pick}"));
            let (out, err) = built.run(&[
                ("MAPAL_EMU_NS", ns),
                ("MAPAL_WISDOM", w.to_str().unwrap()),
                ("MAPAL_GPU_TRACE", "1"),
            ]);
            assert_eq!(decisions(&err), vec![("trial".into(), pick)], "{err}");
            assert_eq!(
                untimed(&out),
                outs[pick as usize],
                "#{z} {target}: the winner's bits\n{src}"
            );
            let last = err
                .lines()
                .filter(|l| l.starts_with("mapal_gpu_trace op=launch "))
                .filter_map(|l| ids.iter().position(|k| l.split(' ').any(|f| f == k)))
                .last();
            assert_eq!(
                last,
                Some(pick as usize),
                "#{z} {target}: the site's last launch\n{err}"
            );
        }
        return;
    }
    panic!("no generated program whose two candidates differ in bits");
}

/// per-site-choice harness: a site executed in a loop is decided once per
/// process (the memo in `mapal_choice.h`): one `op=site` line, the trial's
/// launches once, then one launch of the pick per trip — after a trial, from
/// the wisdom file, and forced.
#[test]
fn a_looped_site_decides_once() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice loop: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let (n, trips) = (64, 4);
    let src = format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> x0;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / 64.0 }} -> w;
    mut x: [f32; {nn}] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < {trips}) -> {{
            -true-> {{
                ta -> map {{ t ->
                    t / {n} -> i;
                    t % {n} -> j;
                    (0.0, krange) -> fold {{ acc, k -> acc + x[i * {n} + k] * w[k * {n} + j] }}
                }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
}}
",
        nn = n * n
    );
    let e = emission(&lower(&src), "rtx-4090", EmitOpts::default(), 8, &[], &[]);
    let ts = tables(&e.text);
    assert_eq!(ts.len(), 1, "one tabled site in the loop body\n{}", e.text);
    let m = ts[0].rows.len();
    assert!(m >= 2 && ts[0].flags & 1 == 1);
    let ids: Vec<String> = ts[0]
        .rows
        .iter()
        .map(|r| format!("kernel=mapal_tile_{:016x}", r[0] as u64))
        .collect();
    let built = emu::build_emulated(&clang, &e.text, e.device.as_deref().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let wis = dir.path().join("w");
    // Candidate 1 fastest on the fake clock.
    let mut ns = vec!["1000000"; m];
    ns[1] = "500000";
    let ns = ns.join(",");
    let run = |extra: &[(&str, &str)]| {
        let mut env = vec![
            ("MAPAL_EMU_NS", ns.as_str()),
            ("MAPAL_WISDOM", wis.to_str().unwrap()),
            ("MAPAL_GPU_TRACE", "1"),
        ];
        env.extend_from_slice(extra);
        let err = built.run(&env).1;
        let site_launches = err
            .lines()
            .filter(|l| l.starts_with("mapal_gpu_trace op=launch "))
            .filter(|l| ids.iter().any(|k| l.split(' ').any(|f| f == k)))
            .count();
        let trials = err.lines().filter(|l| l.contains("op=trial ")).count();
        (decisions(&err), trials, site_launches, err)
    };
    // The trial: candidate 0 sizes it and settles (cross-arch: five more
    // runs on a steady clock), 5 rotated reps, 3 confirming pairs, then
    // trips 2.. launch the pick alone.
    let (d, trials, launches, err) = run(&[]);
    assert_eq!(d, vec![("trial".into(), 1)], "one decision\n{err}");
    assert_eq!(trials, 5 * m + 6, "{err}");
    let settle = err.lines().filter(|l| l.contains("op=settle ")).count();
    assert_eq!(settle, 5, "{err}");
    assert_eq!(launches, 1 + settle + trials + (trips - 1), "{err}");
    // From the wisdom file, and forced: one decision, one launch per trip.
    for (extra, want) in [
        (vec![], ("wisdom", 1)),
        (vec![("MAPAL_GPU_PICK", "rules")], ("forced", 0)),
    ] {
        let (d, trials, launches, err) = run(&extra);
        assert_eq!(d, vec![(want.0.into(), want.1)], "{err}");
        assert_eq!((trials, launches), (0, trips), "{err}");
    }
}

/// tune-context: a site's table carries `MAPAL_SITE_COLD` (bit 3) exactly
/// where one of its operands lives in DRAM in the program — `residence_plan`
/// against the working set the part's L2 keeps (`Gpu::l2_keep_bytes`) — so
/// its trial evicts the L2 before each timed run (`mapal_gpu.c`
/// `site_timed`): a 4096-wide GEMV over a 68 MiB weight and a 256 MiB one
/// on the RTX 5080 (64 MiB L2, a cyclic stream kept to 1.025× it), against
/// the 64 MiB weight there (integration: kept — cold trials picked a
/// candidate 1.7× slower in context for pt_pj_m16_n4096_k4096), the same
/// sites on the RTX 5090's 96 MiB and a 1 MiB weight.
#[test]
fn cold_flag_follows_residence() {
    let src = |k: u64| {
        format!(
            "fn main() {{
    {k} -> iota -> ia;
    ia -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 }} -> b;
    4096 -> iota -> oc;
    {k} -> iota -> kc;
    oc -> map {{ j -> (0.0, kc) -> fold {{ acc, q -> acc + a[q] * b[q * 4096 + j] }} }} -> c;
    c[0] -> println;
    c[4095] -> println;
}}
",
            kn = k * 4096
        )
    };
    for (k, target, cold) in [
        (16384u64, "rtx-5080", true),
        (4352, "rtx-5080", true),
        (4096, "rtx-5080", false),
        (4096, "rtx-5090", false),
        (64, "rtx-5080", false),
    ] {
        let s = src(k);
        let ir = lower(&s);
        let em = emission(&ir, target, EmitOpts::default(), 8, &[], &[]);
        let ts = tables(&em.text);
        assert!(!ts.is_empty() || k == 64, "k {k} {target}: a table");
        for t in &ts {
            assert_eq!(t.flags & 8 != 0, cold, "k {k} {target}: flags {}", t.flags);
        }
    }
}
