//! Batched compilation (one translation unit for many cases) and the closed testgen sweep.

use super::*;

/// One compile-and-run job: the emitted `.ll` and its expected native observable
/// (compiled at both `-O0` and `-O2` in Phase 2).
pub(crate) struct Job {
    pub(crate) tag: String,
    pub(crate) ll: String,
    pub(crate) want_out: Option<String>,
    pub(crate) want_code: i32,
}

// --- batching: N cases, one translation unit -------------------------------
//
// An emitted module is unusually easy to merge: it has no `target` lines, no
// numbered `attributes #N`, and no metadata `!N` — the three things that
// normally make textual IR concatenation painful. What it does have is a named
// type (`%Frame`), `private` globals, and `internal` functions, all of which
// collide by name. Renaming everything a module *defines* makes them disjoint;
// the `declare`d runtime externs are shared and get emitted once.

/// Whether `b` can appear in an LLVM identifier following a `@`/`%` sigil.
pub(crate) fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$' | b'-')
}

/// The identifier starting at `from` (just past a sigil), as a `&str`.
pub(crate) fn ident_at(src: &str, from: usize) -> &str {
    let b = src.as_bytes();
    let mut j = from;
    while j < b.len() && is_ident_byte(b[j]) {
        j += 1;
    }
    &src[from..j]
}

/// Rewrite every `@name`/`%name` occurrence whose identifier is in `map`.
/// Skips `c"…"` string literals so a `@`-looking byte inside printed text can
/// never be mistaken for a symbol reference.
pub(crate) fn rewrite_syms(src: &str, map: &std::collections::HashMap<String, String>) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let mut i = 0;
    while i < b.len() {
        // `c"…"` — copy verbatim, honoring `\\` escapes.
        if b[i] == b'c' && i + 1 < b.len() && b[i + 1] == b'"' {
            let start = i;
            let mut j = i + 2;
            while j < b.len() && b[j] != b'"' {
                j += if b[j] == b'\\' { 2 } else { 1 };
            }
            j = (j + 1).min(b.len());
            out.push_str(&src[start..j]);
            i = j;
            continue;
        }
        if b[i] == b'@' || b[i] == b'%' {
            let name = ident_at(src, i + 1);
            out.push(b[i] as char);
            out.push_str(map.get(name).map_or(name, String::as_str));
            i += 1 + name.len();
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Split one emitted module into `(renamed body, declare lines)`.
///
/// Everything the module **defines** — `define @f`, `@g = …`, `%T = type …` —
/// gets a `c{k}_` prefix; everything it merely **declares** is left alone,
/// because those are the shared runtime symbols. The module's `@main` is
/// defined, so it becomes `@c{k}_main` and the dispatcher can call it.
pub(crate) fn prefix_module(ll: &str, k: usize) -> (String, Vec<String>) {
    let mut declared: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut defined: Vec<&str> = Vec::new();
    let mut declares: Vec<String> = Vec::new();

    for line in ll.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("declare ") {
            if let Some(at) = rest.find('@') {
                declared.insert(ident_at(rest, at + 1));
            }
            declares.push(line.to_string());
        } else if let Some(rest) = t.strip_prefix("define ") {
            if let Some(at) = rest.find('@') {
                defined.push(ident_at(rest, at + 1));
            }
        } else if t.starts_with('@') && t.contains(" = ") {
            defined.push(ident_at(t, 1));
        } else if t.starts_with('%') && t.contains(" = type ") {
            defined.push(ident_at(t, 1));
        }
    }

    let map: std::collections::HashMap<String, String> = defined
        .into_iter()
        .filter(|n| !declared.contains(n))
        .map(|n| (n.to_string(), format!("c{k}_{n}")))
        .collect();

    let body: String = ll
        .lines()
        .filter(|l| !l.trim_start().starts_with("declare "))
        .map(|l| format!("{}\n", rewrite_syms(l, &map)))
        .collect();
    (body, declares)
}

/// Merge `lls` into one module whose `@main` dispatches on `argv[1]`.
///
/// Each case keeps its own `@main` (renamed), so a case that traps still exits
/// the process with 101 and a case that prints still owns the whole stdout —
/// one case per execution, exactly as before.
pub(crate) fn batch_module(lls: &[&str]) -> String {
    let mut declares: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut bodies = String::new();
    for (k, ll) in lls.iter().enumerate() {
        let (body, decls) = prefix_module(ll, k);
        for d in decls {
            if seen.insert(d.clone()) {
                declares.push(d);
            }
        }
        bodies.push_str(&body);
        bodies.push('\n');
    }

    let mut out = String::new();
    for d in &declares {
        out.push_str(d);
        out.push('\n');
    }
    out.push_str("declare i32 @atoi(ptr)\n\n");
    out.push_str(&bodies);

    out.push_str("define i32 @main(i32 %argc, ptr %argv) {\nentry:\n");
    out.push_str("  %slot = getelementptr ptr, ptr %argv, i64 1\n");
    out.push_str("  %arg = load ptr, ptr %slot\n");
    out.push_str("  %sel = call i32 @atoi(ptr %arg)\n");
    out.push_str("  switch i32 %sel, label %bad [\n");
    for k in 0..lls.len() {
        out.push_str(&format!("    i32 {k}, label %case{k}\n"));
    }
    out.push_str("  ]\n");
    for k in 0..lls.len() {
        out.push_str(&format!(
            "case{k}:\n  %r{k} = call i32 @c{k}_main()\n  ret i32 %r{k}\n"
        ));
    }
    // Unreachable in practice: the harness always passes a valid index. A
    // distinctive code rather than 0 so a dispatcher bug can never read as a
    // silently passing case.
    out.push_str("bad:\n  ret i32 127\n}\n");
    out
}

/// Compare one native run against the job's oracle expectations (L1), pushing a
/// message per divergence. `None` is a timeout — the harness never treats a
/// program that failed to finish as a pass.
pub(crate) fn judge(run: Option<(Vec<u8>, i32)>, j: &Job, opt: &str, out: &mut Vec<String>) {
    let Some((stdout, code)) = run else {
        out.push(format!("{} {opt}: timeout", j.tag));
        return;
    };
    if code != j.want_code {
        out.push(format!("{} {opt}: exit {code} != {}", j.tag, j.want_code));
    } else if let Some(w) = &j.want_out {
        let got = String::from_utf8_lossy(&stdout);
        if got != *w {
            out.push(format!("{} {opt}: stdout {got:?} != {w:?}", j.tag));
        }
    }
}

/// Phase 2 of a sweep: every job at `-O0` and `-O2`, `BATCH` jobs per
/// executable; the divergences.
pub(crate) fn run_jobs(clang: &str, jobs: &[Job]) -> Vec<String> {
    let batches: Vec<&[Job]> = jobs.chunks(BATCH).collect();
    let next = AtomicUsize::new(0);
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..job_slots() {
            s.spawn(|| {
                loop {
                    let bi = next.fetch_add(1, Ordering::Relaxed);
                    if bi >= batches.len() {
                        break;
                    }
                    let batch = batches[bi];
                    let lls: Vec<&str> = batch.iter().map(|j| j.ll.as_str()).collect();
                    let merged = batch_module(&lls);
                    for opt in ["-O0", "-O2"] {
                        let mut found = Vec::new();
                        match try_compile_exe(clang, &merged, opt) {
                            Ok((_dir, exe)) => {
                                for (slot, j) in batch.iter().enumerate() {
                                    judge(run_case(&exe, TIMEOUT_SECS, slot), j, opt, &mut found);
                                }
                            }
                            // The merge is a textual transform over emitted IR;
                            // if it ever produces something clang rejects, fall
                            // back to one binary per case so the failure names
                            // the module rather than the batch.
                            Err(e) => {
                                eprintln!(
                                    "differential_testgen: batch {bi} {opt} did not build, \
                                     falling back to per-case ({} cases)\n{e}",
                                    batch.len()
                                );
                                for j in batch {
                                    judge(
                                        compile_run(clang, &j.ll, &j.tag, opt),
                                        j,
                                        opt,
                                        &mut found,
                                    );
                                }
                            }
                        }
                        if !found.is_empty() {
                            failures.lock().unwrap().extend(found);
                        }
                    }
                }
            });
        }
    });
    failures.into_inner().unwrap()
}

/// Build a job from an `ir` + its oracle run. `None` if the run diverged (skip).
pub(crate) fn make_job(ir: &CategoryIr, rr: &RunResult, tag: String) -> Option<Job> {
    let (want_out, want_code) = expect_native(ir, rr)?;
    // ADR-0038 D4: the sweep's value legs pin `generic` by name, like every
    // other leg but differential_native_profile_is_value_correct.
    let ll = mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .unwrap_or_else(|e| panic!("{tag}: emit: {e:?}"));
    if let Ok(d) = std::env::var("S65_DUMP") {
        let name = tag.replace(['#', '/'], "_");
        std::fs::write(format!("{d}/{name}.ll"), &ll).unwrap();
        std::fs::write(format!("{d}/{name}.ir"), format!("{ir:#?}")).unwrap();
        std::fs::write(
            format!("{d}/{name}.want"),
            format!("{want_code}\n{}", want_out.clone().unwrap_or_default()),
        )
        .unwrap();
    }
    Some(Job {
        tag,
        ll,
        want_out,
        want_code,
    })
}

/// Closed-mode testgen sweep (≥ 256; DESIGN §6.3): random Core programs — arrays,
/// updates, multi-loop fns, traps — raw and rewritten, oracle-equal at `-O0` and
/// `-O2`. Open mode (`i32 → i32`) is excluded (no native observable, BL8).
/// Compilation is fanned out across threads (clang subprocess spawn dominates
/// wall time).
#[test]
fn differential_testgen_closed_sweep() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_testgen: clang not found");
        return;
    };
    let _ = rt_lib(); // build the runtime once, before fan-out

    // Phase 1 (serial, fast): generate programs, build raw + rewritten IR, emit.
    let t0 = Instant::now();
    let mut runner = TestRunner::deterministic();
    let mut jobs: Vec<Job> = Vec::new();
    let mut n = 0usize;
    let mut lifted = 0usize;
    for (count, trap_free) in [(256usize, false), (64usize, true)] {
        let strat = prog_strategy(trap_free, false);
        for _ in 0..count {
            let prog = strat.new_tree(&mut runner).unwrap().current();
            let Built { ir, open, .. } = build(&prog);
            if open {
                continue; // excluded (BL8)
            }
            let rr = run(&ir, BUDGET);
            if let Some(j) = make_job(&ir, &rr, format!("testgen#{n}/raw")) {
                jobs.push(j);
            } else {
                continue; // diverged
            }
            let res = rewrite(build(&prog).ir);
            if res
                .report
                .applied
                .iter()
                .any(|(pass, count)| *pass == PassId::LiftLoops && *count > 0)
            {
                lifted += 1;
            }
            let rr2 = run(&res.ir, BUDGET);
            if let Some(j) = make_job(&res.ir, &rr2, format!("testgen#{n}/rewritten")) {
                jobs.push(j);
            }
            n += 1;
        }
    }
    assert!(n >= 256, "expected ≥256 closed cases, got {n}");
    assert!(lifted > 0, "testgen sweep did not exercise loop lifting");
    eprintln!(
        "differential_testgen: phase 1 (generate+emit) {:?}; {} jobs",
        t0.elapsed(),
        jobs.len()
    );
    let t1 = Instant::now();

    // Phase 2 (parallel): every job still runs at both opt levels against the
    // same oracle expectations (DESIGN §8 `-O2` row) — the cross product is
    // untouched. What changed is that `BATCH` jobs share one linked executable
    // and are selected by `argv[1]`, because the per-case cost was never the
    // compile: it was minting and first-running a fresh binary (see `BATCH`).
    let failures = run_jobs(&clang, &jobs);
    assert!(
        failures.is_empty(),
        "{} testgen divergences:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "differential_testgen: {n} closed programs (raw + rewritten) OK at -O0 and -O2; phase 2 (compile+run) {:?}",
        t1.elapsed()
    );
}
