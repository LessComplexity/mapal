//! structure (model-ready): the readable model's constructs, interpreter =
//! native (raw and rewritten) = emulated GPU.
//!
//! - products of arrays at a call boundary (per-layer weight structs, an array
//!   of per-layer caches) pass every array by address — no SSA aggregate holds
//!   an array in the emitted text;
//! - `layers[l]` of an array of products is a view of the element;
//! - a loop exit arm's work runs once, after the loop (its prints included);
//! - CPU tasks reading a host-spine value (a printing loop's exit, a struct the
//!   host builds) run after it;
//! - `() -> f`, float literal widths from context, `ret.k` writes in an exit.
//!
//! Plus a property sweep over generated programs of the same shapes (sizes,
//! field counts, layer counts and element ops drawn per seed), with coverage
//! counters: the views, `ptr %out` calls and exit-arm work it exercised.
#[path = "emu/mod.rs"]
mod emu;

use std::process::Command;

use emu::{clang, rt_lib, run_emulated};
use mapal_backend_llvm::{EmitOpts, emit_detailed, emit_with_opts};
use mapal_ir::CategoryIr;

const BUDGET: u64 = 400_000_000;

const PROGRAMS: &[&str] = &[
    "struct_step_loop",
    "element_view_projection",
    "carried_with_caches",
    "loop_entry_struct",
    "two_print_loops_linked",
    "print_in_exit_arm",
    "zero_parameter_call",
    "float_literal_context",
    "ret_slots_in_exit_arm",
    "inline_return_call",
    "llama_tiny_readable",
    "big_frame_step",
];

fn source(name: &str) -> String {
    let path = format!(
        "{}/tests/structure/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

fn ir_of(src: &str, rewritten: bool) -> CategoryIr {
    let ir = lower(src);
    if rewritten {
        mapal_rewrite::rewrite(ir).ir
    } else {
        ir
    }
}

fn oracle(ir: &CategoryIr, tag: &str) -> String {
    let rr = mapal_interp::run(ir, BUDGET);
    assert!(
        matches!(rr.outcome, mapal_interp::Outcome::Done(_)),
        "{tag}: oracle {:?}",
        rr.outcome
    );
    rr.output
}

/// Compile `ll` with the runtime at `-O2` and run it (two pool lanes).
fn native(clang: &str, ll: &str, tag: &str) -> String {
    let _slot = emu::slot();
    let dir = tempfile::tempdir().unwrap();
    let (llp, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&llp, ll).unwrap();
    let out = Command::new(clang)
        .args(["-O2", "-Wno-override-module"])
        .arg(&llp)
        .arg(rt_lib())
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{tag}: clang: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe)
        .env("MAPAL_PAR", "2")
        .env("MAPAL_SPIN", "0")
        .output()
        .unwrap();
    assert!(run.status.success(), "{tag}: exit {:?}", run.status);
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// Lines moving an SSA aggregate that holds an array: what a by-value product
/// of arrays lowers to (clang ran > 600 s on the readable model's).
fn array_aggregates(ll: &str) -> Vec<&str> {
    ll.lines()
        .filter(|l| l.contains(" = load {") || l.trim_start().starts_with("store {"))
        .filter(|l| l.contains(" x "))
        .collect()
}

fn exact(target: &'static str) -> EmitOpts {
    EmitOpts {
        host: "generic",
        target,
        ..EmitOpts::exact()
    }
}

/// Native CPU, raw and rewritten, equals the interpreter byte for byte on
/// the exact face; no array rides an SSA aggregate.
#[test]
fn structure_programs_native_equal_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    for name in PROGRAMS {
        for rewritten in [false, true] {
            let tag = format!("{name}/{}", if rewritten { "rewritten" } else { "raw" });
            let ir = ir_of(&source(name), rewritten);
            let want = oracle(&ir, &tag);
            let ll = emit_with_opts(&ir, &exact("generic"))
                .unwrap_or_else(|e| panic!("{tag}: emit {e:?}"));
            let agg = array_aggregates(&ll);
            assert!(
                agg.is_empty(),
                "{tag}: array aggregates {:?}",
                &agg[..agg.len().min(3)]
            );
            assert_eq!(native(&clang, &ll, &tag), want, "{tag}");
        }
    }
}

/// The GPU profiles' host and device units, the device emulated thread by
/// thread: equal to the interpreter.
#[test]
fn structure_programs_emulated_gpu_equal_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    for name in PROGRAMS {
        for target in ["rtx-4090", "rtx-3090"] {
            let tag = format!("{name}/{target}");
            let ir = ir_of(&source(name), true);
            let want = oracle(&ir, &tag);
            let em =
                emit_detailed(&ir, &exact(target)).unwrap_or_else(|e| panic!("{tag}: emit {e:?}"));
            if let (Some(dir), Some(dev)) = (std::env::var_os("MAPAL_STRUCTURE_DUMP"), &em.device) {
                let dir = std::path::PathBuf::from(dir);
                std::fs::write(dir.join(format!("{name}.{target}.host.ll")), &em.text).unwrap();
                std::fs::write(dir.join(format!("{name}.{target}.dev.ll")), dev).unwrap();
            }
            let got = match &em.device {
                Some(dev) => run_emulated(&clang, &em.text, dev),
                None => native(&clang, &em.text, &tag),
            };
            assert_eq!(got, want, "{tag}");
        }
    }
}

/// A splitmix64 draw (deterministic per seed).
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One generated program: a struct of `fields` weight arrays per layer, an
/// array of `layers` of them inside a model struct, per-layer caches carried
/// through a loop-carrying fn called from a printing decode loop, element
/// views (`model.layers[l]`, `caches[l]`), an in-place cache row write, and
/// an exit arm that computes (a fold, optionally a print) before returning.
fn generated(seed: u64) -> String {
    let mut d = Draw(seed);
    let width = [4u64, 8, 16][d.pick(3) as usize];
    let rows = 2 + d.pick(3);
    let fields = 1 + d.pick(3);
    let layers = 1 + d.pick(3);
    let steps = 2 + d.pick(3);
    let print_in_exit = d.pick(2) == 0;
    let ops = ["*", "+", "-"];
    let mut s = String::new();
    s.push_str("type Layer {\n");
    for f in 0..fields {
        s.push_str(&format!("    w{f}: [f32; {width}],\n"));
    }
    s.push_str("}\n\ntype Model {\n");
    s.push_str(&format!(
        "    layers: [Layer; {layers}],\n    bias: [f32; {width}],\n}}\n\n"
    ));
    s.push_str(&format!(
        "type Cache {{\n    rows: [f32; {}],\n}}\n\n",
        rows * width
    ));
    // The step: a layer loop over views, the exit arm computes.
    s.push_str(&format!(
        "fn step(token: i32, model: Model, caches: [Cache; {layers}]) -> ([f32; {width}], [Cache; {layers}]) {{\n"
    ));
    s.push_str(&format!(
        "    {width} -> iota -> map {{\n        i -> (i + token) % 5 -> widen_f32\n    }} -> start;\n"
    ));
    s.push_str(&format!(
        "    mut hidden: [f32; {width}] <- start;\n    mut updated: [Cache; {layers}] <- caches;\n    mut layer: i32 <- 0;\n    loop {{\n        (layer < {layers}) -> {{\n            -true-> {{\n"
    ));
    s.push_str("                model.layers[layer] -> weights;\n                updated[layer] -> cache;\n");
    let mut term = String::from("hidden[i]");
    for f in 0..fields {
        let op = ops[d.pick(3) as usize];
        term = format!("({term} {op} weights.w{f}[i] * 0.5)");
    }
    s.push_str(&format!(
        "                {width} -> iota -> map {{\n                    i -> {term} + cache.rows[i] * 0.25\n                }} -> next;\n"
    ));
    s.push_str(&format!(
        "                {} -> iota -> map {{\n                    c -> (c / {width} == token % {rows}) -> {{\n                        -true-> next[c % {width}];\n                        -false-> cache.rows[c];\n                    }}\n                }} -> written;\n",
        rows * width
    ));
    s.push_str("                Cache { rows: written } -> fresh;\n                updated[layer] <- fresh;\n                next -> hidden;\n                layer + 1 -> layer;\n                -> loop;\n            }\n            -false-> {\n");
    s.push_str(&format!(
        "                {width} -> iota -> norm_steps;\n                (0.0, norm_steps) -> fold {{\n                    acc, i -> acc + hidden[i] * hidden[i]\n                }} -> energy;\n"
    ));
    if print_in_exit {
        s.push_str("                energy -> println;\n");
    }
    // A slot-wise return is a pure fn's (L1307); an effectful step returns whole.
    let ret = if print_in_exit {
        "-> out;\n                (out, updated) -> ret;"
    } else {
        "-> ret.0;\n                updated -> ret.1;"
    };
    s.push_str(&format!(
        "                {width} -> iota -> map {{\n                    i -> hidden[i] + model.bias[i] + energy * 0.001\n                }} {ret}\n            }}\n        }}\n    }}\n}}\n\n"
    ));
    // main: builds the model from struct literals, a printing decode loop.
    s.push_str("fn main() {\n");
    for l in 0..layers {
        for f in 0..fields {
            let k = d.pick(7) + 1;
            s.push_str(&format!(
                "    {width} -> iota -> map {{\n        i -> (i * {k} + {l}) % 9 - 4 -> widen_f32\n    }} -> l{l}w{f};\n"
            ));
        }
        let lits: Vec<String> = (0..fields).map(|f| format!("w{f}: l{l}w{f}")).collect();
        s.push_str(&format!(
            "    Layer {{ {} }} -> layer{l};\n",
            lits.join(", ")
        ));
    }
    let names: Vec<String> = (0..layers).map(|l| format!("layer{l}")).collect();
    s.push_str(&format!(
        "    {width} -> iota -> map {{\n        i -> i -> widen_f32\n    }} -> bias;\n    Model {{ layers: [{}], bias: bias }} -> model;\n",
        names.join(", ")
    ));
    s.push_str(&format!(
        "    (0.0, {}) -> fill -> zeros;\n    Cache {{ rows: zeros }} -> blank;\n",
        rows * width
    ));
    let blanks = vec!["blank"; layers as usize].join(", ");
    s.push_str(&format!(
        "    mut caches: [Cache; {layers}] <- [{blanks}];\n    mut token: i32 <- 0;\n    loop {{\n        (token, model, caches) -> step -> stepped;\n        stepped.0[token % {width}] -> println;\n        (token < {steps}) -> {{\n            -true-> {{\n                stepped.1 -> caches;\n                token + 1 -> token;\n                -> loop;\n            }}\n            -false-> stepped.1 -> last;\n        }}\n    }}\n    last[0].rows[{}] -> println;\n}}\n",
        width + 1
    ));
    s
}

/// Generated programs of the readable model's shapes: interpreter = native
/// (raw, rewritten) = emulated GPU; coverage counts the constructs reached.
#[test]
fn structure_generated_programs_equal_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let n: u64 = std::env::var("MAPAL_STRUCTURE_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let (mut out_ptr, mut views, mut exit_prints) = (0, 0, 0);
    for seed in 0..n {
        // Generated source goes through the formatter like any other.
        let src = mapal_syntax::format_source(&generated(seed))
            .unwrap_or_else(|d| panic!("gen{seed}: format {d:?}"));
        let tag = format!("gen{seed}");
        for rewritten in [false, true] {
            let ir = ir_of(&src, rewritten);
            let want = oracle(&ir, &tag);
            let ll = emit_with_opts(&ir, &exact("generic")).expect("emits");
            assert!(
                array_aggregates(&ll).is_empty(),
                "{tag}: array aggregates\n{src}"
            );
            out_ptr += ll.contains("ptr noalias %out") as u32;
            views += ll.contains("getelementptr [") as u32;
            assert_eq!(native(&clang, &ll, &tag), want, "{tag}/{rewritten}\n{src}");
            if rewritten {
                exit_prints += src.contains("energy -> println") as u32;
                let em = emit_detailed(&ir, &exact("rtx-4090")).expect("emits");
                let got = match &em.device {
                    Some(dev) => run_emulated(&clang, &em.text, dev),
                    None => native(&clang, &em.text, &tag),
                };
                assert_eq!(got, want, "{tag}/gpu\n{src}");
            }
        }
    }
    eprintln!(
        "structure coverage: {n} programs, out-ptr {out_ptr}, views {views}, exit prints {exit_prints}"
    );
    assert!(out_ptr > 0 && views > 0 && exit_prints > 0, "coverage");
}
