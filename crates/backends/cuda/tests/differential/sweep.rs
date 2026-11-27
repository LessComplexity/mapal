//! The closed testgen sweep and the exp / math / narrow program differentials.

use super::*;

/// One compile-and-run job: the emitted `.cu` and its expected native
/// observable.
pub(crate) struct Job {
    pub(crate) tag: String,
    pub(crate) cu: String,
    pub(crate) want_out: Option<String>,
    pub(crate) want_code: i32,
}

/// Build a job from an `ir` + its oracle run. `None` if the run diverged.
pub(crate) fn make_job(ir: &CategoryIr, rr: &RunResult, tag: String) -> Option<Job> {
    let (want_out, want_code) = expect_native(ir, rr)?;
    let cu = mapal_backend_cuda::emit(ir).unwrap_or_else(|e| panic!("{tag}: emit: {e:?}"));
    Some(Job {
        tag,
        cu,
        want_out,
        want_code,
    })
}

/// Closed-mode testgen sweep at the llvm row's scale (§6.8): random Core
/// programs — arrays, updates, multi-loop fns, traps — raw and rewritten,
/// oracle-equal. Reproducibility is by MECHANISM: `TestRunner::deterministic()`
/// plus the synced pinned `Cargo.lock`; 320 draws (256 default + 64
/// trap-free), ≥ 256 closed non-diverged required. Open mode (`i32 → i32`)
/// is excluded (no native observable, BL8). Compile+run is fanned across
/// `available_parallelism` threads (nvcc spawn dominates wall time); parity
/// failures and exit-102 infra failures are collected and reported apart.
#[test]
fn differential_testgen_closed_sweep() {
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP differential_testgen_closed_sweep: {SKIP_REASON}");
        return;
    };
    let _ = rt_lib(); // build the runtime once, before fan-out

    // Phase 1 (serial, fast): generate programs, build raw + rewritten IR, emit.
    let t0 = Instant::now();
    let mut runner = TestRunner::deterministic();
    let mut jobs: Vec<Job> = Vec::new();
    let mut n = 0usize;
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
            let rr2 = run(&res.ir, BUDGET);
            if let Some(j) = make_job(&res.ir, &rr2, format!("testgen#{n}/rewritten")) {
                jobs.push(j);
            }
            n += 1;
        }
    }
    assert!(n >= 256, "expected ≥256 closed cases, got {n}");
    eprintln!(
        "differential_testgen: phase 1 (generate+emit) {:?}; {} jobs",
        t0.elapsed(),
        jobs.len()
    );
    let t1 = Instant::now();

    // Phase 2 (parallel): nvcc-compile + run each job, collecting failures —
    // parity (R1 data points) and infra (exit 102, never R1) reported apart.
    let next = AtomicUsize::new(0);
    let parity: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let infra: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        // Fan out to the box's core count — nvcc subprocess spawns dominate
        // wall time (§6: ≈ 2–6 s per small translation unit).
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(8);
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let j = &jobs[i];
                    let got = compile_run(&nvcc, &j.cu, &j.tag);
                    match classify(&j.tag, &j.want_out, j.want_code, &got) {
                        Verdict::Pass => {}
                        Verdict::Parity(m) => parity.lock().unwrap().push(m),
                        Verdict::Infra(m) => infra.lock().unwrap().push(m),
                    }
                }
            });
        }
    });
    let parity = parity.into_inner().unwrap();
    let infra = infra.into_inner().unwrap();
    assert!(
        parity.is_empty() && infra.is_empty(),
        "{} parity divergences + {} infra failures (exit 102 — never R1 data points):\n{}\n{}",
        parity.len(),
        infra.len(),
        parity.join("\n"),
        infra.join("\n")
    );
    eprintln!(
        "differential_testgen: {n} closed programs (raw + rewritten) oracle-equal; \
         phase 2 (nvcc+run) {:?}",
        t1.elapsed()
    );
}

// --- local unit tests (no nvcc needed) -------------------------------------

/// The pure harness pieces, exercised without the toolchain — the only
/// coverage a CUDA-less host gets (discovery order, recipe argv, classifier).
#[cfg(test)]
mod local {
    use super::*;

    #[test]
    fn nvcc_discovery_order() {
        // $NVCC wins, verbatim …
        assert_eq!(
            pick_nvcc(
                Some("/opt/nvcc".into()),
                Some("/cuda/bin/nvcc".into()),
                Some("/usr/bin/nvcc".into())
            ),
            Some("/opt/nvcc".into())
        );
        // … then $CUDA_HOME/bin/nvcc …
        assert_eq!(
            pick_nvcc(
                None,
                Some("/cuda/bin/nvcc".into()),
                Some("/usr/bin/nvcc".into())
            ),
            Some("/cuda/bin/nvcc".into())
        );
        // … then PATH.
        assert_eq!(
            pick_nvcc(None, None, Some("/usr/bin/nvcc".into())),
            Some("/usr/bin/nvcc".into())
        );
        // Absent everywhere ⇒ None ⇒ skip-with-reason (never a faked run).
        assert_eq!(pick_nvcc(None, None, None), None);
        // The middle-leg candidate path shape.
        assert_eq!(cuda_home_nvcc("/opt/cuda"), "/opt/cuda/bin/nvcc");
    }

    #[test]
    fn compile_recipe_argv_pinned() {
        let argv = nvcc_argv(
            Path::new("p.cu"),
            Path::new("libmapal_rt.a"),
            Path::new("p"),
        );
        // §4 recipe, pinned in order: ISO C++17, no FMA contraction, sm_89.
        assert_eq!(&argv[..3], ["-std=c++17", "-fmad=false", "-arch=sm_89"]);
        // Then the translation unit and the mapal-rt staticlib, then -o exe.
        assert_eq!(&argv[3..5], ["p.cu", "libmapal_rt.a"]);
        assert_eq!(&argv[5..7], ["-o", "p"]);
        // §6.6 link tail LAST, in pinned order.
        assert_eq!(&argv[argv.len() - 3..], ["-lpthread", "-ldl", "-lm"]);
        // Forbidden (§4): fast math, host contraction flags, any opt row.
        for bad in [
            "--use_fast_math",
            "-march=native",
            "-mfma",
            "-O3",
            "--fmad=true",
        ] {
            assert!(!argv.iter().any(|a| a == bad), "forbidden flag {bad}");
        }
        assert!(
            !argv.iter().any(|a| a.starts_with("-march")),
            "no -march=* ever: {argv:?}"
        );
    }

    fn done(out: &str) -> (Option<String>, i32) {
        (Some(out.into()), 0)
    }

    fn trapped() -> (Option<String>, i32) {
        (None, 101)
    }

    fn exit(code: i32, out: &[u8], err: &[u8]) -> RunOut {
        RunOut::Exited {
            code,
            stdout: out.to_vec(),
            stderr: err.to_vec(),
        }
    }

    #[test]
    fn classify_done_byte_equal() {
        let (wo, wc) = done("42\n");
        assert!(matches!(
            classify("p", &wo, wc, &exit(0, b"42\n", b"")),
            Verdict::Pass
        ));
        // Byte-different stdout ⇒ parity failure naming the program.
        match classify("prog42", &wo, wc, &exit(0, b"42\nx", b"")) {
            Verdict::Parity(m) => assert!(m.contains("prog42"), "{m}"),
            v => panic!("expected Parity, got {v:?}"),
        }
    }

    #[test]
    fn classify_trap_classes_never_cross() {
        let (wo, wc) = trapped();
        // 101 matching an oracle trap passes — stdout IGNORED.
        assert!(matches!(
            classify(
                "p",
                &wo,
                wc,
                &exit(101, b"garbage", b"mapal trap: div_zero\n")
            ),
            Verdict::Pass
        ));
        // Oracle trapped but native exited 0 ⇒ class cross ⇒ Parity.
        match classify("p", &wo, wc, &exit(0, b"", b"")) {
            Verdict::Parity(m) => assert!(m.contains("classes never cross"), "{m}"),
            v => panic!("expected Parity, got {v:?}"),
        }
        // Oracle done but native trapped ⇒ class cross ⇒ Parity.
        let (wo0, wc0) = done("");
        match classify("p", &wo0, wc0, &exit(101, b"", b"")) {
            Verdict::Parity(m) => assert!(m.contains("classes never cross"), "{m}"),
            v => panic!("expected Parity, got {v:?}"),
        }
    }

    #[test]
    fn classify_exit_102_is_infra_never_a_data_point() {
        // Regardless of the oracle's class — a 102 can never complete an R1
        // row, even when the oracle wanted a trap.
        for (wo, wc) in [done(""), trapped()] {
            match classify("gpu_prog", &wo, wc, &exit(102, b"", b"cudaErrorNoDevice")) {
                Verdict::Infra(m) => {
                    assert!(m.contains("gpu_prog"), "{m}");
                    assert!(m.contains("102"), "{m}");
                    assert!(m.contains("cudaErrorNoDevice"), "{m}");
                }
                v => panic!("expected Infra, got {v:?}"),
            }
        }
    }

    #[test]
    fn classify_timeout_and_other_exits() {
        let (wo, wc) = done("x\n");
        match classify("slow_prog", &wo, wc, &RunOut::TimedOut) {
            Verdict::Parity(m) => {
                assert!(m.contains("slow_prog"), "{m}");
                assert!(m.contains("timed out"), "{m}");
            }
            v => panic!("expected Parity, got {v:?}"),
        }
        // Any other exit ⇒ failure WITH captured stderr.
        match classify("crash_prog", &wo, wc, &exit(1, b"", b"segv-ish")) {
            Verdict::Parity(m) => {
                assert!(m.contains("crash_prog"), "{m}");
                assert!(m.contains("segv-ish"), "{m}");
            }
            v => panic!("expected Parity, got {v:?}"),
        }
        // Signal death (code −1) is the same class.
        assert!(matches!(
            classify("p", &wo, wc, &exit(-1, b"", b"")),
            Verdict::Parity(_)
        ));
    }
}

/// ADR-0042: `exp` on the CUDA backend is the oracle's algorithm, each device
/// step an explicit round-to-nearest intrinsic — generated programs over
/// every branch of it (overflow, subnormal results, ±inf and NaN arguments,
/// `f64`, a fold body, a softmax), raw and rewritten, byte for byte.
#[test]
fn differential_exp_programs_raw_and_rewritten() {
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP: {SKIP_REASON}");
        return;
    };
    let mut runner = TestRunner::deterministic();
    for i in 0..6 {
        let p = testgen::exp_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let src = testgen::exp_src(&p);
        let ir = lower_src(&src);
        let rr = run(&ir, 2_000_000_000);
        assert_parity(&nvcc, &ir, &rr, &format!("exp{i}/raw"));
        let rw = rewrite(lower_src(&src)).ir;
        let rr = run(&rw, 2_000_000_000);
        assert_parity(&nvcc, &rw, &rr, &format!("exp{i}/rewritten"));
    }
}

/// ADR-0046: `sqrt`, `sin`, `cos` on the CUDA backend — the references printed
/// from their traces (each device step an explicit round-to-nearest
/// intrinsic, the integer Payne–Hanek reduction, IEEE `__fsqrt_rn` /
/// `sqrtf`) — generated programs over every branch (huge arguments,
/// subnormals, ±0, ±∞, NaN, `f64`, fold bodies), raw and rewritten, byte for
/// byte with the interpreter on the device and the host.
#[test]
fn differential_math_programs_raw_and_rewritten() {
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP: {SKIP_REASON}");
        return;
    };
    let mut runner = TestRunner::deterministic();
    for i in 0..6 {
        let p = testgen::math_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let src = testgen::math_src(&p);
        let ir = lower_src(&src);
        let rr = run(&ir, 2_000_000_000);
        assert_parity(&nvcc, &ir, &rr, &format!("math{i}/raw"));
        let rw = rewrite(lower_src(&src)).ir;
        let rr = run(&rw, 2_000_000_000);
        assert_parity(&nvcc, &rw, &rr, &format!("math{i}/rewritten"));
    }
}

/// ADR-0043: `bf16` / `f16` storage on the CUDA backend — `__nv_bfloat16` /
/// `__half` elements, `__float2bfloat16_rn` / `__float2half_rn` roundings,
/// exact widening casts — generated narrow programs (overflow, subnormals,
/// guards, loops, element updates, sums of widened products), raw and
/// rewritten, byte for byte with the interpreter.
#[test]
fn differential_narrow_programs_raw_and_rewritten() {
    let Some(nvcc) = nvcc() else {
        eprintln!("SKIP: {SKIP_REASON}");
        return;
    };
    let mut runner = TestRunner::deterministic();
    for i in 0..6 {
        let p = testgen::narrow_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        let (src, _) = testgen::narrow_src(&p);
        let ir = lower_src(&src);
        let rr = run(&ir, 2_000_000_000);
        assert_parity(&nvcc, &ir, &rr, &format!("narrow{i}/raw"));
        let rw = rewrite(lower_src(&src)).ir;
        let rr = run(&rw, 2_000_000_000);
        assert_parity(&nvcc, &rw, &rr, &format!("narrow{i}/rewritten"));
    }
}
