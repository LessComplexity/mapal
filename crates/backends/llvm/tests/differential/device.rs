//! Device-sized loop differentials run on the emulator: values and data movement.

use super::*;

/// Bytes of an LLVM array type text `[n x T]` (nested arrays of scalars).
pub(crate) fn llt_array_bytes(t: &str) -> u64 {
    let t = t.trim();
    if let Some(inner) = t.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        let (n, e) = inner.split_once(" x ").expect("array text");
        return n.trim().parse::<u64>().expect("array length") * llt_array_bytes(e);
    }
    match t {
        "i8" => 1,
        "i32" | "float" => 4,
        "i64" | "double" | "ptr" => 8,
        _ => 0,
    }
}

/// Per `; mapal.move` loop: the line's fields, and what the blocks from the
/// header to the latch move per trip — `(memcpy calls, memcpy bytes, h2d
/// bytes, d2h bytes)` — a first pass's blocks (a `store i1 true` to a pass
/// flag: once per entry) aside.
pub(crate) fn gpu_trip_moves(
    ll: &str,
) -> Vec<(std::collections::HashMap<String, String>, [u64; 4])> {
    let last_i64 = |l: &str| -> u64 {
        l.trim_end_matches(')')
            .rsplit("i64 ")
            .next()
            .and_then(|x| x.trim().parse().ok())
            .unwrap_or(0)
    };
    let mut out = Vec::new();
    for f in ll.split("\ndefine ").skip(1) {
        let lines: Vec<&str> = f.lines().collect();
        for (h, l) in lines.iter().enumerate() {
            let Some(rest) = l.strip_prefix("; mapal.move loop=") else {
                continue;
            };
            let header = rest.split_whitespace().next().expect("header");
            let kv = rest
                .split_whitespace()
                .filter_map(|x| x.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let back = format!("br label %{header}");
            let latch = (h..lines.len())
                .rev()
                .find(|&i| lines[i].trim() == back)
                .expect("latch");
            // Blocks of the region: a first-pass block stores a pass flag.
            let mut moved = [0u64; 4];
            let mut block: Vec<&str> = Vec::new();
            let flush = |b: &mut Vec<&str>, m: &mut [u64; 4]| {
                if !b
                    .iter()
                    .any(|x| x.trim().starts_with("store i1 true, ptr %s"))
                {
                    for x in b.iter() {
                        if x.contains("call void @llvm.memcpy") {
                            m[0] += 1;
                            m[1] += x
                                .split("getelementptr (")
                                .nth(1)
                                .and_then(|t| t.split(", ptr null").next())
                                .map_or(0, llt_array_bytes);
                        } else if x.contains("@mapal_gpu_h2d(")
                            || x.contains("@mapal_gpu_h2d_async(")
                        {
                            m[2] += last_i64(x);
                        } else if x.contains("@mapal_gpu_d2h(") || x.contains("@mapal_gpu_d2h_on(")
                        {
                            m[3] += last_i64(x);
                        }
                    }
                }
                b.clear();
            };
            for x in &lines[h..=latch] {
                if x.ends_with(':') && !x.starts_with(' ') {
                    flush(&mut block, &mut moved);
                }
                block.push(x);
            }
            flush(&mut block, &mut moved);
            out.push((kv, moved));
        }
    }
    out
}

/// **S65 rule R's device leg, on unseen programs.** ≥ 64 generated loops over
/// device-sized arrays (testgen `device_prog_strategy`: sites, per-cell maps,
/// guards, row writes into the state, swaps, invariant next states, one or
/// two carried arrays, element or whole exit reads), every capable site on the
/// device (`place` off), run on the GPU host emulation (`tests/emu`): the
/// exact face equals the interpreter with the rung on and off; and per trip
/// the host IR moves exactly what the residency plan derived from the facts
/// (`; mapal.move`: uploads of host-produced arrays a device site reads,
/// copy-outs of device outputs a host morphism reads) plus the host copies the
/// facts require ([`required_trip_copies`]) — a device-carried state never
/// crosses, and no forwarding morphism copies.
#[test]
fn differential_device_loops_values_and_movement() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP differential_device_loops: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let strat = testgen::device_prog_strategy();
    // The programs drawn in order (one deterministic runner), then run over the
    // job slots: each is interpreter-, emitter-, clang- and emulator-bound and
    // independent of the others.
    let progs: Vec<_> = (0..64)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let work = |n: usize, p: &testgen::DeviceProg| -> (Vec<String>, [usize; 4]) {
        let (mut placed, mut carried, mut uniform, mut nests) = (0usize, 0usize, 0usize, 0usize);
        let mut findings = Vec::new();
        let src = testgen::device_src(p);
        let ir = rewrite(lower_src(&src)).ir;
        let rr = run(&ir, 400_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        for on in [true, false] {
            let mut gpu = mapal_backend_llvm::GpuArms::default();
            let off: &[&str] = if on { &["place"] } else { &["place", "by-ref"] };
            for r in off.iter().chain(&["streams", "pinned", "pdl", "async"]) {
                *gpu.off.rung(r).expect("rung") = true;
            }
            let opts = mapal_backend_llvm::EmitOpts {
                host: "generic",
                target: "h100-pcie",
                gpu,
                ..mapal_backend_llvm::EmitOpts::exact()
            };
            let em =
                match std::panic::catch_unwind(|| mapal_backend_llvm::emit_detailed(&ir, &opts)) {
                    Ok(em) => em.expect("emits"),
                    Err(_) => {
                        findings.push(format!("device#{n} on={on}: the emitter panicked\n{src}"));
                        continue;
                    }
                };
            // (A program whose every site the plan keeps on the host has no
            // device unit: its CPU build — the CPU leg's.)
            let Some(dev) = em.device else { continue };
            let got = emu::run_emulated(&clang, &em.text, &dev);
            if got != rr.output {
                findings.push(format!(
                    "device#{n} on={on}: {got:?} != {:?}\n{src}",
                    rr.output
                ));
            }
            if !on {
                continue;
            }
            placed += 1;
            // cpu-gpu-boundary 1b / 1a, counted from the emission: a rule
            // fires when its arm launches more cells than its OFF arm, and a
            // uniform cell takes an `i64` kernel parameter.
            let launches = |t: &str| t.matches("call void @mapal_gpu_launchv(").count();
            let fires = |rung: &str, base: &mapal_backend_llvm::EmitOpts| {
                let mut o = *base;
                *o.gpu.off.rung(rung).expect("rung") = true;
                let at = mapal_backend_llvm::emit_detailed(&ir, base).map(|e| launches(&e.text));
                mapal_backend_llvm::emit_detailed(&ir, &o)
                    .is_ok_and(|e| at.is_ok_and(|n| n > launches(&e.text)))
            };
            uniform += usize::from(dev.contains("i64 %c") && fires("uniform", &opts));
            // A nest whose body goes on past its fold is a `tile-post` site
            // where that rung is on: the nest cell is counted beside it.
            let mut no_post = opts;
            *no_post.gpu.off.rung("tile-post").expect("rung") = true;
            nests += usize::from(fires("nest-cell", &no_post));
            let need = required_trip_copies(&ir, ir.entry());
            let loops = gpu_trip_moves(&em.text);
            carried += loops
                .iter()
                .filter(|(kv, _)| kv["carried"].contains(":device"))
                .count();
            let copies: u64 = loops.iter().map(|(_, m)| m[0]).sum();
            for (kv, [_, copy_b, h2d, d2h]) in &loops {
                let want = |k: &str| kv[k].parse::<u64>().expect("intent field");
                if (*copy_b, *h2d, *d2h) != (want("host_copy"), want("h2d"), want("d2h")) {
                    findings.push(format!(
                        "device#{n}: per trip copy {copy_b} B h2d {h2d} B d2h {d2h} B; the plan: {kv:?}\n{src}"
                    ));
                }
            }
            if copies != need as u64 {
                findings.push(format!(
                    "device#{n}: {copies} host copies per trip, the facts require {need}\n{src}"
                ));
            }
        }
        (findings, [placed, carried, uniform, nests])
    };
    let next = AtomicUsize::new(0);
    let all = Mutex::new((Vec::new(), [0usize; 4]));
    std::thread::scope(|sc| {
        for _ in 0..job_slots() {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= progs.len() {
                        break;
                    }
                    let (f, st) = work(i, &progs[i]);
                    let mut a = all.lock().unwrap();
                    a.0.extend(f);
                    for (x, y) in a.1.iter_mut().zip(st) {
                        *x += y;
                    }
                }
            });
        }
    });
    let n = progs.len();
    let (mut findings, [placed, carried, uniform, nests]) = all.into_inner().unwrap();
    findings.sort();
    assert!(
        findings.is_empty(),
        "{} findings:\n{}",
        findings.len(),
        findings.join("\n")
    );
    assert!(
        placed >= 48 && carried > 0 && uniform > 0 && nests > 0,
        "{placed} device programs, {carried} device-carried states, {uniform} with a uniform cell, {nests} with a nest cell"
    );
    eprintln!(
        "differential_device_loops: {n} programs, {placed} with a device unit, {carried} device-carried states, \
         {uniform} with a uniform cell, {nests} with a nest cell; values OK on and off, movement = the plan's"
    );
}

/// **generic-fusion natural forms, on unseen programs.** ≥ 256 generated
/// device-sized loops in twin forms (testgen `twin_prog_strategy`: sites,
/// per-cell maps, guards, row folds, a clamped power, and writes into the
/// state — a row, a column, a range of a row, one cell): the **natural** form
/// (a guarded map over the cells, a power as a fold over a range) against the
/// **unrolled** one (`Update`s, a chain of squarings), the same values by
/// construction. The natural form must compile as well:
/// - the same `write_delta` (and a `Rows` claim holds: per trip the
///   interpreter sees no cell of the state change outside it);
/// - per GPU profile (`h100-pcie` and `rtx-4090` alternating, every capable
///   site on the device) and on the CPU (`generic`): per trip no more bytes
///   moved, copied or allocated and no less retained (`; mapal.move`), no more
///   launches (the same kernel kinds at as many), and the realized movement
///   equals the plan's (`gpu_trip_moves`, plus the host copies the facts
///   require);
/// - values: the exact face equals the interpreter with `range-map` on and
///   off (the emulated device, or the CPU build).
#[test]
fn differential_natural_twins_values_and_movement() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP differential_natural_twins: clang not found");
        return;
    };
    let _ = emu::rt_lib();
    let cases: usize = std::env::var("MAPAL_TWINS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::twin_prog_strategy();
    let deltas = |ir: &CategoryIr| -> Vec<Option<(u64, i64, i64)>> {
        let mut v: Vec<_> = ir
            .write_delta(ir.entry())
            .values()
            .map(|d| match *d {
                mapal_ir::Delta::Rows { stride, start, .. } => Some((stride, start.a, start.b)),
                _ => None,
            })
            .collect();
        v.sort();
        v
    };
    let emit = |ir: &CategoryIr, target: &'static str, range: bool, cell: bool| {
        let mut gpu = mapal_backend_llvm::GpuArms::default();
        let off = ["streams", "pinned", "pdl", "async", "place"]
            .into_iter()
            .chain((!range).then_some("range-map"))
            .chain((!cell).then_some("range-cell"));
        for r in off {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = mapal_backend_llvm::EmitOpts {
            host: "generic",
            target,
            gpu,
            ..mapal_backend_llvm::EmitOpts::exact()
        };
        mapal_backend_llvm::emit_detailed(ir, &opts).expect("emits")
    };
    let emit_off = |ir: &CategoryIr, target: &'static str, extra: &[&str]| {
        let mut gpu = mapal_backend_llvm::GpuArms::default();
        for r in ["streams", "pinned", "pdl", "async", "place"]
            .iter()
            .chain(extra)
        {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = mapal_backend_llvm::EmitOpts {
            host: "generic",
            target,
            gpu,
            ..mapal_backend_llvm::EmitOpts::exact()
        };
        mapal_backend_llvm::emit_detailed(ir, &opts).expect("emits")
    };
    let moves = |t: &str| -> Vec<String> {
        t.lines()
            .filter(|l| l.starts_with("; mapal.move"))
            .map(|l| l.split_once(" carried=").map_or(l, |(_, r)| r).to_string())
            .collect()
    };
    let launches = |t: &str| {
        t.matches("call void @mapal_gpu_launch").count()
            + t.matches("call void @mapal_gpu_tile(").count()
    };
    let kinds = |d: &Option<String>| -> Vec<String> {
        let mut k: Vec<String> = d
            .iter()
            .flat_map(|d| d.lines())
            .filter_map(|l| l.strip_prefix("; mapal.plan kernel="))
            .filter_map(|l| l.split(" kind=").nth(1))
            .map(|k| k.split(' ').next().unwrap_or("").to_string())
            .collect();
        k.sort();
        k
    };
    let range_cells = |d: &Option<String>| {
        d.iter()
            .flat_map(|d| d.lines())
            .filter(|l| {
                l.starts_with("; mapal.plan ") && l.contains(" kind=cell ") && l.contains(" range=")
            })
            .count()
    };
    let run_on = |em: &mapal_backend_llvm::Emission| -> String {
        match &em.device {
            Some(dev) => emu::run_emulated(&clang, &em.text, dev),
            None => compile_run(&clang, &em.text, "twin", "-O1").map_or_else(
                || "timeout".into(),
                |(o, c)| {
                    format!(
                        "{}{}",
                        String::from_utf8_lossy(&o),
                        if c == 0 {
                            String::new()
                        } else {
                            format!("[exit {c}]")
                        }
                    )
                },
            ),
        }
    };
    let progs: Vec<testgen::TwinProg> = (0..cases)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    // One program's checks: its findings, and `[Rows claims, probed, device
    // builds, in-place states, better builds]`.
    let work = |case: usize, p: &testgen::TwinProg| -> (Vec<String>, [usize; 5]) {
        let mut findings = Vec::new();
        let (mut rows, mut probed, mut devices, mut inplace, mut better) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let (ns, us) = (
            testgen::twin_src(&p, true, None),
            testgen::twin_src(&p, false, None),
        );
        let tag = format!("twin#{case}");
        let (ni, ui) = (rewrite(lower_src(&ns)).ir, rewrite(lower_src(&us)).ir);
        let oracle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (run(&ni, 400_000_000), run(&ui, 400_000_000))
        }));
        let Ok((rr, ru)) = oracle else {
            findings.push(format!("{tag}: the interpreter panicked\n{ns}"));
            return (findings, [0; 5]);
        };
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{ns}");
        assert_eq!(ru.output, rr.output, "the twins' values\n{ns}\n{us}");
        let dn = deltas(&ni);
        if dn != deltas(&ui) {
            findings.push(format!(
                "{tag}: Δ {dn:?} != the unrolled {:?}\n{ns}",
                deltas(&ui)
            ));
        }
        // The `Rows` claim, probed: the state's cells outside it never change.
        if let Some(&Some((stride, a, b))) = dn.iter().find(|d| d.is_some()) {
            rows += 1;
            let cs = testgen::twin_src(&p, true, Some((a, b, stride)));
            let ci = rewrite(lower_src(&cs)).ir;
            if deltas(&ci) != dn {
                findings.push(format!(
                    "{tag}: the probe changed Δ: {:?}\n{cs}",
                    deltas(&ci)
                ));
            }
            let rc = run(&ci, 400_000_000);
            let bad: Vec<&str> = rc
                .output
                .lines()
                .filter(|l| l.starts_with("dz=") && *l != "dz=0")
                .collect();
            if !matches!(rc.outcome, Outcome::Done(_)) || !bad.is_empty() {
                findings.push(format!(
                    "{tag}: Δ Rows {stride} at {a}·it + {b} is unsound: {bad:?}\n{cs}"
                ));
            }
            probed += 1;
        }
        for target in [["h100-pcie", "rtx-4090"][case % 2], "generic"] {
            let (on, off, un) = (
                emit(&ni, target, true, true),
                emit(&ni, target, false, true),
                emit(&ui, target, true, true),
            );
            // recurrence-precision `range-cell` is the natural form's alone (a
            // map written in place; an `Update` stays the host's): launches
            // and kernels are the twins' with it off, and with it on at most
            // its range cells more, each replacing a written row's upload.
            let flat = emit(&ni, target, true, false);
            // model-ready placement `range-chain`: a chain of range cells
            // carries the state on the device link by link and the trip's
            // cells follow it there (more launches, fewer bytes) — the
            // single-cell bound is `range-chain` off's; with it on, no loop
            // moves more bytes per trip.
            let single = emit_off(&ni, target, &["range-chain"]);
            let ranges = range_cells(&single.device);
            if launches(&single.text) > launches(&flat.text) + ranges {
                findings.push(format!(
                    "{tag} {target}: range-cell launches {} > {} + {ranges} range cells\n{ns}",
                    launches(&single.text),
                    launches(&flat.text)
                ));
            }
            let bytes = |t: &str| -> Vec<i64> {
                moves(t)
                    .iter()
                    .map(|l| {
                        l.split_whitespace()
                            .filter_map(|x| x.split_once('='))
                            .filter(|(k, _)| *k == "h2d" || *k == "d2h")
                            .filter_map(|(_, v)| v.parse::<i64>().ok())
                            .sum()
                    })
                    .collect()
            };
            let (bc, bs) = (bytes(&on.text), bytes(&single.text));
            if bc.len() != bs.len() || bc.iter().zip(&bs).any(|(c, s)| c > s) {
                findings.push(format!(
                    "{tag} {target}: range-chain bytes {bc:?} > {bs:?}\n{ns}"
                ));
            }
            let (mn, mu) = (moves(&on.text), moves(&un.text));
            if mn != mu {
                // As well or better: per loop, no more bytes over the bus
                // (h2d + d2h: model-ready `bytes-place` trades one way for the
                // other), copied or allocated per trip, no less retained.
                let costs = |l: &str| -> Vec<i64> {
                    let kv: std::collections::HashMap<&str, i64> = l
                        .split_whitespace()
                        .filter_map(|x| x.split_once('='))
                        .filter_map(|(k, v)| Some((k, v.parse().ok()?)))
                        .collect();
                    [
                        kv["h2d"] + kv["d2h"],
                        kv["host_copy"],
                        kv["alloc"],
                        -kv["retained"],
                    ]
                    .to_vec()
                };
                let worse = mn.len() != mu.len()
                    || mn
                        .iter()
                        .zip(&mu)
                        .any(|(a, b)| costs(a).iter().zip(costs(b)).any(|(x, y)| *x > y));
                if worse {
                    findings.push(format!(
                        "{tag} {target}: movement {mn:?} worse than the unrolled {mu:?}\n{ns}"
                    ));
                } else {
                    better += 1;
                }
            }
            if launches(&flat.text) > launches(&un.text)
                || (launches(&flat.text) == launches(&un.text)
                    && kinds(&flat.device) != kinds(&un.device))
            {
                findings.push(format!(
                    "{tag} {target}: launches {} kinds {:?} != the unrolled {} {:?}\n{ns}",
                    launches(&flat.text),
                    kinds(&flat.device),
                    launches(&un.text),
                    kinds(&un.device)
                ));
            }
            inplace += moves(&on.text)
                .iter()
                .filter(|l| l.starts_with("0:") && l.contains(":inplace"))
                .count();
            devices += on.device.is_some() as usize;
            // Realized movement = the plan's, and the host copies the facts require.
            let loops = gpu_trip_moves(&on.text);
            let copies: u64 = loops.iter().map(|(_, m)| m[0]).sum();
            for (kv, [_, copy_b, h2d, d2h]) in &loops {
                let want = |k: &str| kv[k].parse::<u64>().expect("intent field");
                if (*copy_b, *h2d, *d2h) != (want("host_copy"), want("h2d"), want("d2h")) {
                    findings.push(format!("{tag} {target}: per trip copy {copy_b} B h2d {h2d} B d2h {d2h} B; the plan: {kv:?}\n{ns}"));
                }
            }
            let need = required_trip_copies(&ni, ni.entry()) as u64;
            if copies != need {
                findings.push(format!(
                    "{tag} {target}: {copies} host copies per trip, the facts require {need}\n{ns}"
                ));
            }
            for (arm, em) in [("on", &on), ("off", &off)] {
                // (The CPU build's OFF arm: every other program, for time.)
                if target == "generic" && arm == "off" && case % 2 == 1 {
                    continue;
                }
                let got = run_on(em);
                if got != rr.output {
                    findings.push(format!(
                        "{tag} {target} range-map {arm}: {got:?} != {:?}\n{ns}",
                        rr.output
                    ));
                }
            }
        }
        (findings, [rows, probed, devices, inplace, better])
    };
    // The programs over the cores (each is interpreter-, clang- and
    // emulator-bound and independent of the others).
    let next = AtomicUsize::new(0);
    let all = Mutex::new((Vec::new(), [0usize; 5]));
    let workers = job_slots();
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cases {
                        break;
                    }
                    let (f, st) = work(i, &progs[i]);
                    let mut a = all.lock().unwrap();
                    a.0.extend(f);
                    for (x, y) in a.1.iter_mut().zip(st) {
                        *x += y;
                    }
                }
            });
        }
    });
    let (mut findings, [rows, probed, devices, inplace, better]) = all.into_inner().unwrap();
    findings.sort();
    assert!(
        findings.is_empty(),
        "{} findings:\n{}",
        findings.len(),
        findings.join("\n")
    );
    // (The sweep's reach, at the default size: a small MAPAL_TWINS is a smoke run.)
    assert!(
        cases < 64 || (rows > cases / 8 && devices > cases / 2),
        "{rows} Rows claims, {devices} device units"
    );
    eprintln!(
        "differential_natural_twins: {cases} twin programs, {rows} Δ Rows claims ({probed} probed sound), \
         {devices} device builds, {inplace} in-place states; movement, launches, kernels = the unrolled twins' \
         ({better} builds better); values OK with range-map on and off"
    );
}
