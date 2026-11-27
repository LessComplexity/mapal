//! Candidate sets: every candidate answers the interpreter, candidate zero, caps and excluded classes.

use super::*;

/// B4 values and re-run safety: for every program with a table, the emulated
/// output of every candidate index below the program's largest table answers
/// the interpreter; `MAPAL_GPU_PICK=all` equals the last index bit for bit
/// where every table is flagged `RERUN`. per-site-choice D4: the part does not
/// follow the strategy — every strategy meets every part — and each part
/// reaches tables, exact-face runs and `f64` runs.
#[test]
fn every_candidate_answers_the_interpreter() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP site_choice values: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    // One program's counts.
    #[derive(Default)]
    struct V {
        programs: usize,
        tabled: usize,
        runs: usize,
        reruns: usize,
        sites: usize,
        exact: usize,
        by_n: [usize; 9],
        // Per part: [tabled programs, exact-face runs, f64 runs].
        per: [[usize; 3]; 4],
    }
    let vs = testgen::par_check(&cases(48), |z, case| {
        let mut v = V::default();
        let ir = lower(&case.src);
        // Each block of 12 runs every strategy, rotated over the parts by
        // block (a Latin square: every strategy meets every part); the
        // rotation is the one under which the deterministic draws give every
        // part tables, exact-face and f64 runs.
        let part = (z % 12 + [3, 2, 0, 1][z / 12 % 4]) % PARTS.len();
        let target = PARTS[part];
        let e = emission(&ir, target, case.base, 8, &[], &[]);
        v.programs += 1;
        let ts = tables(&e.text);
        let Some(max) = ts.iter().map(|t| t.rows.len()).max() else {
            return v;
        };
        v.tabled += 1;
        v.per[part][0] += 1;
        v.sites += ts.len();
        for t in &ts {
            v.by_n[t.rows.len().min(8)] += 1;
        }
        let rr = mapal_interp::run(&ir, 4_000_000_000);
        assert!(
            matches!(rr.outcome, mapal_interp::Outcome::Done(_)),
            "#{z}: {:?}\n{}",
            rr.outcome,
            case.src
        );
        let dev = e.device.as_deref().expect("a device unit");
        let built = emu::build_emulated(&clang, &e.text, dev);
        let picks: Vec<String> = (0..max)
            .map(|i| i.to_string())
            .chain(["all".to_string()])
            .collect();
        let outs: Vec<String> = std::thread::scope(|s| {
            let b = &built;
            let hs: Vec<_> = picks
                .iter()
                .map(|p| s.spawn(move || b.run(&[("MAPAL_GPU_PICK", p.as_str())]).0))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, got) in outs[..max].iter().enumerate() {
            assert!(
                check(case, got, &rr.output),
                "#{z} {target} candidate {i}: {got:?} != {:?}\n{}",
                rr.output,
                case.src
            );
            v.runs += 1;
            v.exact += usize::from(case.base.order == Order::Sequential);
            v.per[part][1] += usize::from(case.base.order == Order::Sequential);
            v.per[part][2] += usize::from(case.src.contains("f64"));
        }
        if ts.iter().all(|t| t.flags & 1 == 1) {
            assert_eq!(
                untimed(&outs[max]),
                untimed(&outs[max - 1]),
                "#{z} {target}: every candidate in turn vs the last\n{}",
                case.src
            );
            v.reruns += 1;
        }
        v
    });
    let (mut programs, mut tabled, mut runs, mut reruns, mut sites) = (0, 0, 0, 0, 0);
    let mut exact = 0;
    let mut by_n = [0usize; 9];
    let mut per = [[0usize; 3]; 4];
    for v in &vs {
        (programs, tabled, runs) = (programs + v.programs, tabled + v.tabled, runs + v.runs);
        (reruns, sites, exact) = (reruns + v.reruns, sites + v.sites, exact + v.exact);
        for (n, k) in by_n.iter_mut().zip(v.by_n) {
            *n += k;
        }
        for (p, k) in per.iter_mut().zip(v.per) {
            for (a, b) in p.iter_mut().zip(k) {
                *a += b;
            }
        }
    }
    eprintln!(
        "site-choice values: {programs} programs, {tabled} with tables ({sites} sites, n histogram \
         {by_n:?}), {runs} candidate runs ({exact} on the exact face), {reruns} re-run checks; \
         per part [tabled, exact runs, f64 runs] {per:?}"
    );
    assert!(
        tabled >= 20 && runs >= 64 && exact >= 10 && reruns >= 16,
        "the generator reaches site tables"
    );
    for (p, [t, x, f]) in PARTS.iter().zip(per) {
        assert!(
            t >= 6 && x >= 2 && f >= 2,
            "{p}: {t} tabled, {x} exact runs, {f} f64 runs"
        );
    }
}

/// per-site-choice D1: candidate 0 is the rules' plan, pinned — `--cands=1`
/// hashes (host + device IR less the `; mapal.` evidence lines) as recorded
/// for each program of a fixed bench list on the four parts and three faces
/// (`tests/golden/c0-rules.tsv`). It pinned 089f1a6's (pre-sprint) emission
/// until hardware-properties replaced the per-part rung rows by the rung
/// rules, which moved candidate 0 where a part's properties decide otherwise
/// than its row did; `MAPAL_REGOLD=1` rewrites the file.
#[test]
fn candidate_zero_is_the_pre_sprint_emission() {
    let golden = include_str!("../golden/c0-rules.tsv");
    let mut regold = String::new();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../benches/algorithms");
    let norm = |t: &str| -> String {
        t.split_inclusive('\n')
            .filter(|l| !l.starts_with("; mapal."))
            .collect()
    };
    let fnv = |b: &[u8]| {
        b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
            (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    let mut rows = 0;
    let mut bad = Vec::new();
    for l in golden.lines().filter(|l| !l.starts_with('#')) {
        let [prog, part, face, want]: [&str; 4] = l
            .split('\t')
            .collect::<Vec<_>>()
            .try_into()
            .expect("program part face hash");
        let src = std::fs::read_to_string(format!("{root}/{prog}.mapal")).expect("a bench program");
        let mut opts = EmitOpts::default();
        let face_flags: &[&str] = match face {
            "contract" => &["--order=sequential"],
            "exact" => &["--exact"],
            _ => &[],
        };
        let target = format!("--target={part}");
        // fold-algebra changes the default / contract faces' arithmetic by
        // design (`step_algebra` pairs and regroups); with its two OFF arms the
        // emission is the pre-sprint one byte for byte, which is what this pins.
        // compile-time-graph and kernel-merging likewise: row-block units,
        // zero tails and merged launches off, the emission is the one before
        // them. large-gemm's site rules (`k-fill`, `streamk-fill`) too; its
        // `wide-fma` property rule moved candidate 0 where a part issues more
        // FFMA per DRAM byte than `FILL_BALANCE_PM` on a contract face (the
        // rtx-5090's default-face rows, regolded). review-fixes: `calls`
        // prices the bodies on every part (`coop_body_cycles`, the rtx-5080's
        // measured cycles at each part's clock): of the 21 rtx-4090 / 3090
        // rows with a chain (decode, mlp_2048_tf32, rq_mlp2_512, smlp,
        // px_mlp2_64_f64), decode_1024_256's 5-unit chain forms (the code of
        // `--calls=force`), the others are refused by cost (the code of
        // `--gpu-off=calls`; rq_mlp2_512 on the 3090 formed on the
        // boundaries alone). gemv-stream's rungs (`vec-cells`,
        // `stream-fly`, `bulk-stage`) change the one-row sites' and the TF32
        // tile's kernels by design: with their OFF arms the emission is the
        // one before them. f32-tensor-cores too (`mma-combine` cuts the
        // rtx-3090's mlp_2048_tf32 tile, `split-tile` takes the A100's `{f32}`
        // products): their OFF arms keep the emission before them.
        // calls-issue prices the separate form as the host issues it (the
        // profile's `issue_ns`): on the rtx-5090 (2370 ns) and rtx-4090
        // (2850 ns), px_mlp2_64_f64's two chains of 64–147 ns members form,
        // and decode_1024_256's 5-unit chain on the rtx-5090 (members 1.9,
        // 1.9, 0.001, 0.6, 0.6 µs at the part's peaks) — regolded; the
        // rtx-5080's rows are unchanged (`--gpu-off=calls-issue`: the one before).
        // hopper-wgmma-tma: the 2048² TF32 MLP site takes the warp-specialized
        // wgmma tile on the sm_90a profiles (h100-pcie's pt_mlp_2048_tf32);
        // its OFF arm keeps the emission before it.
        // prefill-integrate: a shared body's one-cell-buffer device variant is
        // named `<body>_u<slots>`, apart from cell-law's `<body>_cell`
        // (prefill-precision): px_mlp2_64_f64's, pt_rq_mlp2_512's and
        // pt_smlp_4096_256_1024's 36 rows change by that name alone (with the
        // names normalized their host and device text is equal) — regolded.
        for f in [
            target.as_str(),
            "--host=x86-64-v3",
            "--cands=1",
            "--no-fma-pairs",
            "--no-fold-regroup",
            "--no-row-block",
            "--gpu-off=zero-fold",
            "--gpu-off=horizontal",
            "--gpu-off=k-fill,streamk-fill",
            "--gpu-off=vec-cells,stream-fly,bulk-stage",
            "--gpu-off=mma-combine,split-tile",
            "--gpu-off=wgmma-ws",
            // sweep-attention-2: the k-major `b` copies (decode scores).
            "--gpu-off=kmajor-b",
            // gaps-disasm C1: the zero kernel's init flag and the table
            // registry change the host text by design; off, the one before.
            "--gpu-off=zero-host,zero-hoist,load-dispatch",
            // gapfix rows-operands: `iota-law` computes a captured iota's
            // element where the host and device folds loaded it (all 204
            // rows: the un-rewritten programs' host folds), `stream-ld` loads
            // read-once operands evict-first (3 rows), `shfl-tree` combines
            // a unit's lanes by shuffles (4 rows); off, the one before
            // (`mma-fly` and `kvec-swizzle` move none of these rows).
            "--gpu-off=iota-law,stream-ld,shfl-tree",
        ]
        .iter()
        .chain(face_flags)
        {
            assert!(mapal_backend_llvm::flags::apply(&mut opts, f).expect("a flag"));
        }
        let e = emit_detailed(&lower(&src), &opts).expect("emits");
        let body = format!(
            "{}\0{}",
            norm(&e.text),
            norm(e.device.as_deref().unwrap_or(""))
        );
        let got = format!("{:016x}", fnv(body.as_bytes()));
        regold += &format!("{prog}\t{part}\t{face}\t{got}\n");
        if got != want {
            bad.push(format!("{prog} {part} {face}: {got} != {want}"));
        }
        rows += 1;
    }
    if std::env::var_os("MAPAL_REGOLD").is_some() {
        let head: String = golden
            .lines()
            .filter(|l| l.starts_with('#'))
            .map(|l| format!("{l}\n"))
            .collect();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/c0-rules.tsv");
        std::fs::write(path, head + &regold).expect("golden written");
        return;
    }
    assert!(rows >= 200, "{rows} golden rows");
    assert!(bad.is_empty(), "differs from 089f1a6:\n{}", bad.join("\n"));
}

/// B4 c0 identity: `--cands=1` emits no table and no site line; where the
/// default emits none either, the two emissions are equal byte for byte, and
/// every table's candidate 0 is the `--cands=1` build's launch (kernel,
/// grid, block, shared memory, zero, workspace, reduce) and kernel text.
#[test]
fn candidate_zero_is_the_rules_plan() {
    let (mut same, mut tables_checked, mut kernels) = (0, 0, 0);
    for (z, case) in cases(96).iter().enumerate() {
        let ir = lower(&case.src);
        for target in PARTS {
            let one = emission(&ir, target, case.base, 1, &[], &[]);
            assert!(
                !one.text.contains("@mapal_gpu_site(")
                    && !one.device.as_deref().unwrap_or("").contains("; mapal.site"),
                "#{z} {target}: --cands=1 emits a table"
            );
            let all = emission(&ir, target, case.base, 8, &[], &[]);
            let ts = tables(&all.text);
            if ts.is_empty() {
                assert_eq!(all.text, one.text, "#{z} {target} host");
                assert_eq!(all.device, one.device, "#{z} {target} device");
                same += 1;
                continue;
            }
            let (dev8, dev1) = (
                all.device.as_deref().expect("device"),
                one.device.as_deref().expect("device"),
            );
            for t in &ts {
                let c0 = t.rows[0];
                assert_eq!(c0[0] as u64, t.site, "#{z} {target}: site = candidate 0");
                // The `--cands=1` launch of that kernel: `launch(id, a, b, o,
                // gx, gy, block, smem)` or `launchv(id, s, n, p, gx, gy, …)`.
                let call = one
                    .text
                    .lines()
                    .find(|l| {
                        l.contains(&format!("@mapal_gpu_launch(i64 {}, ", c0[0]))
                            || l.contains(&format!("@mapal_gpu_launchv(i64 {}, ", c0[0]))
                    })
                    .unwrap_or_else(|| panic!("#{z} {target}: no launch of {}", c0[0]));
                let args: Vec<i64> = call
                    .split('(')
                    .nth(1)
                    .unwrap()
                    .split(')')
                    .next()
                    .unwrap()
                    .split(", ")
                    .filter_map(|a| a.strip_prefix("i32 ").and_then(|v| v.parse().ok()))
                    .collect();
                let grid = &args[args.len() - 4..];
                assert_eq!(grid, &c0[4..8], "#{z} {target}: grid\n{call}");
                // An atomic / stream-K recipe zeroes the output as the
                // rules' launch is preceded by its zero (at the site, or
                // hoisted ahead of it: gaps-disasm C1 `zero-hoist`).
                let zero = one.text.lines().any(|l| {
                    (l.contains("@mapal_gpu_zero(") || l.contains("@mapal_gpu_zero_for("))
                        && l.ends_with(&format!(", i64 {})", c0[2]))
                });
                assert!(c0[2] == 0 || zero, "#{z} {target}: zero {}", c0[2]);
                if c0[1] != 0 {
                    // The workspace split: its reduce launch, its workspace.
                    assert!(
                        one.text
                            .contains(&format!("@mapal_gpu_launch(i64 {}, ", c0[1])),
                        "#{z} {target}: reduce"
                    );
                    assert!(
                        one.text.lines().any(|l| l.contains("@mapal_gpu_alloc(")
                            && l.ends_with(&format!(", i64 {})", c0[3]))),
                        "#{z} {target}: workspace bytes"
                    );
                }
                for id in [c0[0], c0[1]].into_iter().filter(|&i| i != 0) {
                    assert_eq!(
                        kernel_text(dev8, id),
                        kernel_text(dev1, id),
                        "#{z} {target}: kernel {id:x} text"
                    );
                    kernels += 1;
                }
                tables_checked += 1;
            }
        }
    }
    eprintln!(
        "site-choice c0: {same} emissions without a table equal, {tables_checked} tables, \
         {kernels} candidate-0 kernels equal"
    );
    assert!(tables_checked >= 32 && same >= 16);
}

/// B4 candidate set: at most `--cands` candidates, the table's length on its
/// `; mapal.site` line, distinct recipes and kernels; on the fixed and exact
/// faces every candidate folds as candidate 0 does; each candidate kernel
/// carries its plan line.
#[test]
fn candidate_sets_cap_dedupe_and_keep_the_order() {
    let (mut capped, mut ordered, mut lines) = (0, 0, 0);
    let (mut flips, mut conflict_free, mut ruled_bank) = ([0usize; 2], 0, 0);
    let fixed = EmitOpts {
        contract: false,
        order: Order::Fixed,
        ..EmitOpts::default()
    };
    for (z, case) in cases(60).iter().enumerate() {
        let ir = lower(&case.src);
        let target = PARTS[z % PARTS.len()];
        for (base, cap) in [
            (case.base, 8),
            (case.base, 3),
            (case.base, 2),
            (fixed, 8),
            (EmitOpts::exact(), 8),
        ] {
            let e = emission(&ir, target, base, cap, &[], &[]);
            let dev = e.device.as_deref().unwrap_or("");
            for t in tables(&e.text) {
                let n = t.rows.len();
                assert!(
                    (2..=cap as usize).contains(&n),
                    "#{z} {target} cap {cap}: n={n}"
                );
                capped += usize::from(n == cap as usize);
                let mut ids: Vec<i64> = t.rows.iter().map(|r| r[0]).collect();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), n, "#{z} {target}: distinct kernels");
                let site = dev
                    .lines()
                    .find(|l| l.starts_with(&format!("; mapal.site site={:016x} ", t.site)))
                    .unwrap_or_else(|| panic!("#{z} {target}: no site line"));
                assert_eq!(field(site, "n="), n.to_string(), "#{z}: {site}");
                lines += 1;
                // per-site-choice A4 x B1: the `kmajor-a` axis flips what the
                // register rule decided — the forced k-major panel beside a
                // declined (transposed) candidate 0, the transposed one beside
                // a taken one.
                let kmaj = |id: i64| plan(dev, id).is_some_and(|p| p.contains(" kmajor=1"));
                for (r, ax) in t.rows.iter().zip(field(site, "axes=").split(',')) {
                    // D1 / D5.3: a k-major panel the register rule takes under
                    // `bank-plan` (not forced) is conflict-free.
                    if ax == "+bank-plan" && kmaj(r[0]) {
                        let p = plan(dev, r[0]).unwrap_or("");
                        assert!(p.contains(" adeg=1 "), "#{z} {target}: {p}");
                        ruled_bank += 1;
                    }
                    let on = match ax {
                        "+kmajor-a" => true,
                        "-kmajor-a" => false,
                        _ => continue,
                    };
                    let c0 = kmaj(t.rows[0][0]);
                    assert_eq!((c0, kmaj(r[0])), (!on, on), "#{z} {target}: {site}");
                    flips[usize::from(on)] += 1;
                    let p = plan(dev, if on { r[0] } else { t.rows[0][0] }).unwrap_or("");
                    conflict_free += usize::from(p.contains(" adeg=1 "));
                }
                let combine = |id: i64| {
                    let p = plan(dev, id).unwrap_or_else(|| panic!("#{z}: no plan line {id:x}"));
                    (field(p, "combine=").to_string(), field(p, "S=").to_string())
                };
                let c0 = combine(t.rows[0][0]);
                if base.order != Order::Free {
                    for r in &t.rows {
                        assert_eq!(combine(r[0]), c0, "#{z} {target} {:?}", base.order);
                    }
                    ordered += 1;
                }
            }
        }
    }
    eprintln!(
        "site-choice sets: {lines} tables, {capped} at their cap, {ordered} fixed/exact; \
         kmajor-a flips -{} +{}, {conflict_free} k-major kernels adeg=1; {ruled_bank} `+bank-plan` \
         k-major kernels (all adeg=1)",
        flips[0], flips[1]
    );
    assert!(capped >= 8 && ordered >= 8 && lines >= 40);
    assert!(flips[0] + flips[1] > 0, "the kmajor-a axis reaches a table");
}

/// B4 excluded classes: no table without `resident` (the round trip), in a
/// function on chain streams, on a column-major store (`rowfold-layout`), on
/// a site table (`linear`) or on a launch with an L2 window.
#[test]
fn excluded_site_classes_keep_one_launch() {
    let (mut col, mut window, mut chained) = (0, 0, 0);
    for (z, case) in cases(48).iter().enumerate() {
        let ir = lower(&case.src);
        let target = PARTS[z % PARTS.len()];
        let round = emission(&ir, target, case.base, 8, &[], &["resident"]);
        assert!(!round.text.contains("@mapal_gpu_site("), "#{z} round trip");
        let e = emission(
            &ir,
            target,
            case.base,
            8,
            &["l2-window", "rowfold-layout"],
            &[],
        );
        let dev = e.device.as_deref().unwrap_or("");
        for t in tables(&e.text) {
            let p = plan(dev, t.rows[0][0]).expect("plan line");
            assert_eq!(field(p, "col="), "0", "#{z} {target}: a column-major table");
            assert!(!p.contains(" terms="), "#{z}: a linear table");
        }
        for l in e
            .text
            .lines()
            .filter(|l| l.contains("@mapal_gpu_launch_l2(i64 "))
        {
            let id = l
                .split("(i64 ")
                .nth(1)
                .and_then(|r| r.split(',').next())
                .unwrap();
            assert!(
                !tables(&e.text)
                    .iter()
                    .any(|t| t.site == id.parse::<i64>().unwrap() as u64),
                "#{z}: an L2-window launch in a table"
            );
            window += 1;
        }
        col += dev
            .lines()
            .filter(|l| l.starts_with("; mapal.plan") && l.contains(" col=1 "))
            .count();
        // Chain streams on (the emulator is not involved: emission only).
        let mut gpu = GpuArms::default();
        for r in ["streams", "stream-gate"] {
            *gpu.on.rung(r).expect("rung") = true;
        }
        *gpu.off.rung("place").expect("rung") = true;
        let opts = EmitOpts {
            host: "generic",
            target,
            gpu,
            ..case.base
        };
        let s = emit_detailed(&ir, &opts).expect("emits");
        for f in s.text.split("\ndefine ").skip(1) {
            let on_streams = f.contains("@mapal_gpu_event(")
                || f.split("@mapal_gpu_launchv(i64 ")
                    .skip(1)
                    .any(|t| t.split(", ").nth(1).is_some_and(|x| x != "i32 0"));
            if on_streams {
                assert!(!f.contains("@mapal_gpu_site("), "#{z}: a table on streams");
                chained += 1;
            }
        }
    }
    // A row fold over a site's output (`rowfold.rs`'s row sum): its
    // column-major store keeps one launch.
    let n = 64;
    let rowsum = format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {n} -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen<f32> }} -> b;
    ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }} }} -> c;
    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> c[i * {n} + j] -> e; acc + e }} }} -> r;
    r[0] -> println;
    r[{last}] -> println;
}}
",
        nn = n * n,
        last = n - 1
    );
    let ir = lower(&rowsum);
    for target in PARTS {
        for base in [EmitOpts::default(), EmitOpts::exact()] {
            let e = emission(
                &ir,
                target,
                base,
                8,
                &["rowfold", "rowfold-layout"],
                &["fill-rows", "epilogue", "devmap"],
            );
            let dev = e.device.as_deref().unwrap_or("");
            let cols: Vec<&str> = dev
                .lines()
                .filter(|l| l.starts_with("; mapal.plan") && l.contains(" col=1 "))
                .collect();
            for t in tables(&e.text) {
                let p = plan(dev, t.rows[0][0]).expect("plan line");
                assert_eq!(field(p, "col="), "0", "{target}: a column-major table");
            }
            col += cols.len();
        }
    }
    eprintln!(
        "site-choice excluded: {col} column-major kernels, {window} L2 windows, {chained} chained functions"
    );
    assert!(window >= 1, "the generator reaches an L2 window");
    assert!(col >= 4, "the row sum stores column-major");
}
