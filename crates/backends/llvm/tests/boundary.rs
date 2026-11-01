//! cpu-gpu-boundary runtime rungs (`preload`, `entry-arena`, `async-h2d`,
//! `stage`) on programs from the generator (testgen device loops and chains,
//! which the rungs were not designed on): the host text the rules require on
//! every GPU profile, and the values through the GPU emulator — each rung moves
//! work out of the timed path, never a value.

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const RUNGS: [&str; 4] = ["preload", "entry-arena", "async-h2d", "stage"];
const TARGETS: [&str; 4] = ["rtx-4090", "h100-pcie", "rtx-5090", "rtx-3090"];

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

/// The exact face on `target`, the boundary rungs on or off (and the rungs the
/// emulator cannot run off: `async`, `pdl`, `streams`).
fn emission(ir: &CategoryIr, target: &'static str, on: bool) -> mapal_backend_llvm::Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in RUNGS {
        *(if on { gpu.on.rung(r) } else { gpu.off.rung(r) }).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::exact()
    };
    emit_detailed(ir, &opts).expect("emits")
}

fn init_flags(ll: &str) -> u32 {
    ll.lines()
        .find_map(|l| l.trim().strip_prefix("call void @mapal_gpu_init(i32 "))
        .map_or(0, |r| r.trim_end_matches(')').parse().expect("flags"))
}

/// `(head, rest)` of every function `define`d in `ll`: `head` runs before its
/// first device call other than the arena and before its first clock read.
fn functions(ll: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for f in ll.split("\ndefine ").skip(1) {
        let f = &f[..f.find("\n}\n").unwrap_or(f.len())];
        let lines: Vec<&str> = f.lines().collect();
        let cut = lines
            .iter()
            .position(|l| {
                l.contains("@mapal_time_ms(")
                    || l.contains("@mapal_gpu_") && !l.contains("@mapal_gpu_arena(")
            })
            .unwrap_or(lines.len());
        out.push((lines[..cut].join("\n") + "\n", lines[cut..].join("\n")));
    }
    out
}

/// The slot a call line names after `key` (`%dN`).
fn slot<'l>(line: &'l str, key: &str) -> Option<&'l str> {
    let r = &line[line.find(key)? + key.len()..];
    Some(&r[..r.find([',', ')'])?])
}

/// The host text the rules require, ON: every upload enqueued without a host
/// wait; per function at most one arena, made in the entry block and nowhere
/// else; every slot allocated in the body and freed only at exit seeded from
/// it at entry; `preload` and `stage` in the init flags. OFF: none of it.
fn check_host(ll: &str, on: bool, what: &str) -> usize {
    let mut seeded = 0;
    if on {
        assert!(
            !ll.contains("call void @mapal_gpu_h2d("),
            "{what}: a synchronous upload"
        );
        if ll.contains("@mapal_gpu_init(") {
            assert_eq!(
                init_flags(ll) & (8 | 16),
                8 | 16,
                "{what}: preload + stage flags"
            );
        }
        for (entry, rest) in functions(ll) {
            assert!(
                !rest.contains("call i64 @mapal_gpu_arena("),
                "{what}: arena past entry\n{rest}"
            );
            assert!(
                entry.matches("@mapal_gpu_arena(").count() <= 1,
                "{what}: two arenas"
            );
            // The exit sequence: the trailing frees, unpin, heap release, `ret`.
            let lines: Vec<&str> = rest.lines().filter(|l| *l != "}").collect();
            let tail = [
                "call void @mapal_gpu_free(",
                "call void @mapal_gpu_unpin(",
                "ret ",
                "call void @mapal_rt_",
            ];
            let exit = lines.len()
                - lines
                    .iter()
                    .rev()
                    .take_while(|l| tail.iter().any(|t| l.trim().starts_with(t)))
                    .count();
            let mid: Vec<&str> = lines[..exit]
                .iter()
                .filter_map(|l| slot(l, "call void @mapal_gpu_free(ptr "))
                .collect();
            for l in rest.lines() {
                let Some(s) = slot(l, "call i64 @mapal_gpu_alloc(ptr ") else {
                    continue;
                };
                if mid.contains(&s) {
                    continue;
                }
                assert!(
                    entry.contains(&format!(", ptr {s}\n")) && entry.contains("@mapal_gpu_arena("),
                    "{what}: {s} allocated in the body, not seeded at entry\n{entry}"
                );
                seeded += 1;
            }
        }
    } else {
        assert!(
            !ll.contains("@mapal_gpu_h2d_async("),
            "{what}: async upload OFF"
        );
        assert_eq!(init_flags(ll) & (8 | 16), 0, "{what}: flags OFF");
    }
    seeded
}

#[test]
fn boundary_rungs_host_text_and_values_on_unseen_programs() {
    let mut runner = TestRunner::deterministic();
    let (dev_s, chain_s) = (testgen::device_prog_strategy(), testgen::chain_strategy());
    let clang = emu::clang();
    let (mut seeded, mut ran, mut asyncs) = (0usize, 0usize, 0usize);
    for z in 0..40 {
        let src = if z % 4 != 3 {
            testgen::device_src(&dev_s.new_tree(&mut runner).unwrap().current())
        } else {
            chain_s.new_tree(&mut runner).unwrap().current().source()
        };
        let ir = lower(&src);
        for target in TARGETS {
            let (on, off) = (emission(&ir, target, true), emission(&ir, target, false));
            seeded += check_host(&on.text, true, &format!("#{z} {target} ON\n{src}"));
            check_host(&off.text, false, &format!("#{z} {target} OFF\n{src}"));
            asyncs += on.text.matches("call void @mapal_gpu_h2d_async(").count();
            // Values: ON and OFF print the same through the emulator.
            if let (Some(cc), "rtx-4090", Some(d_on), Some(d_off)) =
                (&clang, target, on.device.as_deref(), off.device.as_deref())
            {
                let (a, b) = (
                    emu::run_emulated(cc, &on.text, d_on),
                    emu::run_emulated(cc, &off.text, d_off),
                );
                assert_eq!(a, b, "#{z} {target}: values ON vs OFF\n{src}");
                ran += 1;
            }
        }
    }
    assert!(seeded >= 40, "too few loop-cone slots seeded: {seeded}");
    assert!(asyncs >= 10, "too few async uploads: {asyncs}");
    assert!(clang.is_none() || ran >= 20, "too few emulated runs: {ran}");
}

/// CPU-only emission is the text it was: the rungs name the GPU runtime only.
#[test]
fn cpu_emission_ignores_the_boundary_rungs() {
    let mut runner = TestRunner::deterministic();
    let dev_s = testgen::device_prog_strategy();
    for _ in 0..8 {
        let ir = lower(&testgen::device_src(
            &dev_s.new_tree(&mut runner).unwrap().current(),
        ));
        for target in ["generic", "apple-m", "x86-64-v3"] {
            assert_eq!(
                emission(&ir, target, true).text,
                emission(&ir, target, false).text,
                "{target}"
            );
        }
    }
}

/// Two independent sites, each over an operand the host builds (a capture of
/// a fold): with streams open, each upload crosses on its reader's chain.
const TWIN_UP: &str = "fn main() {
    4096 -> iota -> ta;
    64 -> iota -> krange;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a0;
    ta -> map { t -> (t * 11 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> b;
    ta -> map { t -> (t * 5 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> e0;
    ta -> map { t -> (t * 3 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> f;
    (0.0, ta) -> fold { acc, t -> acc + a0[t] } -> sa;
    (0.0, ta) -> fold { acc, t -> acc + e0[t] } -> se;
    ta -> map { t -> a0[t] + sa } -> a;
    ta -> map { t -> e0[t] + se } -> e;
    ta -> map { t -> t / 64 -> i; t % 64 -> j; (0.0, krange) -> fold { acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] } } -> c;
    ta -> map { t -> t / 64 -> i; t % 64 -> j; (0.0, krange) -> fold { acc, k -> acc + e[i * 64 + k] * f[k * 64 + j] } } -> d;
    c[0] -> println;
    c[4095] -> println;
    d[0] -> println;
    d[4095] -> println;
}
";

#[test]
fn chain_h2d_uploads_on_the_readers_stream() {
    let ir = lower(TWIN_UP);
    let arm = |on: bool| {
        let mut gpu = GpuArms::default();
        gpu.on.streams = true;
        gpu.off.stream_gate = true;
        gpu.off.cp_async = true;
        gpu.off.pdl = true;
        // Host-built operands: `uniform` would build `a`/`e`-like maps (a
        // fold's result captured) on the device, leaving nothing to upload.
        gpu.off.uniform = true;
        *(if on { &mut gpu.on } else { &mut gpu.off })
            .rung("chain-h2d")
            .expect("rung") = true;
        emit_detailed(
            &ir,
            &EmitOpts {
                host: "generic",
                target: "rtx-4090",
                gpu,
                ..EmitOpts::exact()
            },
        )
        .expect("emits")
    };
    let (on, off) = (arm(true), arm(false));
    let chains = |ll: &str| -> Vec<String> {
        ll.lines()
            .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_h2d_on(i32 "))
            .map(|r| r[..r.find(',').expect("args")].to_string())
            .collect()
    };
    assert_eq!(chains(&on.text), ["1", "2"], "one upload per chain");
    for k in ["1", "2"] {
        assert!(
            on.text.contains(&format!("@mapal_gpu_launchv(i64 "))
                && on.text.contains(&format!(", i32 {k}, i32 3, ptr "))
        );
    }
    // OFF: the uploads stay, on the legacy stream (sync or async per the part's `async-h2d` row).
    assert!(chains(&off.text).is_empty() && off.text.contains("@mapal_gpu_h2d"));
    if let Some(cc) = emu::clang() {
        let (a, b) = (
            on.device.as_deref().expect("dev"),
            off.device.as_deref().expect("dev"),
        );
        assert_eq!(
            emu::run_emulated(&cc, &on.text, a),
            emu::run_emulated(&cc, &off.text, b)
        );
    }
}

/// Independent device chains over host-built operands (a map capturing a
/// fold of its own source), randomized: 2–3 chains of 1–2 tile sites each
/// on an `n × n` grid, the host-built operand left or right, element sinks.
fn indep_src(n: u64, chains: &[(u8, u8, bool)]) -> String {
    let mut s = format!(
        "fn main() {{\n    {} -> iota -> ta;\n    {n} -> iota -> kr;\n",
        n * n
    );
    let law = |name: &str, p: u8, q: u8| {
        format!(
            "    ta -> map {{ t -> (t * {} + {}) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> {name};\n",
            p % 13 + 1,
            q % 29
        )
    };
    for (i, &(seed, depth, left)) in chains.iter().enumerate() {
        s += &law(&format!("x{i}"), seed, seed / 3);
        s += &format!("    (0.0, ta) -> fold {{ acc, t -> acc + x{i}[t] }} -> s{i};\n");
        s += &format!("    ta -> map {{ t -> x{i}[t] + s{i} }} -> a{i};\n");
        s += &law(&format!("b{i}"), seed / 7, seed);
        let mut cur = format!("a{i}");
        for d in 0..depth % 2 + 1 {
            let (l, r) = if left || d > 0 {
                (cur.clone(), format!("b{i}"))
            } else {
                (format!("b{i}"), cur.clone())
            };
            let out = format!("c{i}_{d}");
            s += &format!(
                "    ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> acc + {l}[i * {n} + k] * {r}[k * {n} + j] }} }} -> {out};\n"
            );
            cur = out;
        }
        s += &format!(
            "    {cur}[0] -> println;\n    {cur}[{}] -> println;\n",
            n * n - 1
        );
    }
    s + "}\n"
}

/// `chain-h2d` on unseen programs (independent chains as above, testgen
/// chains and device loops; streams open): every `h2d_on(k)` crosses on a
/// stream some launch of the function runs on, the OFF arm has none, and ON
/// and OFF print the same through the emulator; the rule fires.
#[test]
fn chain_h2d_on_unseen_programs() {
    use proptest::prelude::*;
    let mut runner = TestRunner::deterministic();
    let (dev_s, chain_s) = (testgen::device_prog_strategy(), testgen::chain_strategy());
    let indep_s = (
        prop::sample::select(vec![32u64, 64]),
        prop::collection::vec(any::<(u8, u8, bool)>(), 2..4),
    );
    let clang = emu::clang();
    let arm = |ir: &CategoryIr, on: bool| {
        let mut gpu = GpuArms::default();
        gpu.on.streams = true;
        gpu.off.stream_gate = true;
        gpu.off.cp_async = true;
        gpu.off.pdl = true;
        // Host-built operands: `uniform` would build `a`/`e`-like maps (a
        // fold's result captured) on the device, leaving nothing to upload.
        gpu.off.uniform = true;
        *(if on { &mut gpu.on } else { &mut gpu.off })
            .rung("chain-h2d")
            .expect("rung") = true;
        emit_detailed(
            ir,
            &EmitOpts {
                host: "generic",
                target: "rtx-4090",
                gpu,
                ..EmitOpts::exact()
            },
        )
        .expect("emits")
    };
    let streams = |ll: &str| -> Vec<String> {
        ll.lines()
            .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_h2d_on(i32 "))
            .map(|r| r[..r.find(',').expect("args")].to_string())
            .collect()
    };
    let mut fired = 0usize;
    for z in 0..48 {
        let src = match z % 3 {
            0 => {
                let (n, ch) = indep_s.new_tree(&mut runner).unwrap().current();
                indep_src(n, &ch)
            }
            1 => chain_s.new_tree(&mut runner).unwrap().current().source(),
            _ => testgen::device_src(&dev_s.new_tree(&mut runner).unwrap().current()),
        };
        let ir = lower(&src);
        let (on, off) = (arm(&ir, true), arm(&ir, false));
        assert!(!off.text.contains("@mapal_gpu_h2d_on("), "#{z} OFF\n{src}");
        let ups = streams(&on.text);
        fired += ups.len();
        for k in &ups {
            assert!(
                on.text.contains(&format!(", i32 {k}, i32 ")),
                "#{z}: h2d_on({k}) but no launch on it\n{src}"
            );
        }
        if let (Some(cc), Some(a), Some(b)) = (&clang, on.device.as_deref(), off.device.as_deref())
        {
            assert_eq!(
                emu::run_emulated(cc, &on.text, a),
                emu::run_emulated(cc, &off.text, b),
                "#{z}\n{src}"
            );
        }
    }
    assert!(fired > 0, "chain-h2d fired on no generated program");
    eprintln!("chain_h2d_on_unseen_programs: {fired} chain uploads");
}
