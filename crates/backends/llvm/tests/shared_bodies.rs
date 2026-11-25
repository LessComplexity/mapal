//! precision `shared-cell`: one map body, several device variants. The
//! inliner's copies of a function share every map and fold body (a layer
//! loop's unrolled trips, a prompt's and a decode step's blocks), so one body
//! can be a `cell-law` cell in one copy (its captured operand's value
//! forwarded: `<body>_cell(arg, float %cv…)`) and an S54 epilogue member of
//! the previous copy's tile in the next (its cell read as a one-cell buffer:
//! `<body>_u<slots>(arg)`). Each variant is its own device function: the two
//! used to share the name `<body>_cell`, so the module kept one text and the
//! cell kernel called the member's one-argument text with a forwarded value
//! — a poison pointer (the readable Llama's chained blocks, `y1 = 0`).
//!
//! On programs from the generator (`testgen::SharedCellProg`: a block — an
//! elementwise map feeding a GEMM — chained over 2–3 inlined calls): the
//! emulated device equals the interpreter byte for byte on the exact face in
//! every emulator mode, with `epilogue` on and off; every device function name
//! is defined once and called with its definition's arity.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
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

/// The exact-face emission on the RTX 5080 with `off` rungs off (and the
/// ones the emulator cannot run; `place` off: every capable site on the
/// device).
fn emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in off.iter().chain(&[
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::exact()
    };
    emit_detailed(ir, &opts).expect("emits")
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

/// Each device function's parameter count, by name (`define … @name(…)`);
/// a name defined twice fails.
fn definitions(dev: &str) -> std::collections::HashMap<String, usize> {
    let mut out = std::collections::HashMap::new();
    for l in dev.lines().filter(|l| l.starts_with("define ")) {
        let Some(at) = l.find('@') else { continue };
        let rest = &l[at + 1..];
        let Some(open) = rest.find('(') else { continue };
        let name = rest[..open].to_string();
        let args = params(&rest[open + 1..]);
        assert!(
            out.insert(name.clone(), args).is_none(),
            "{name} defined twice"
        );
    }
    out
}

/// The top-level comma-separated items of a parenthesized list starting
/// after its `(` (aggregate types nest braces).
fn params(s: &str) -> usize {
    let (mut depth, mut n, mut any) = (0i32, 0usize, false);
    for ch in s.chars() {
        match ch {
            '(' | '{' | '[' | '<' => depth += 1,
            ')' if depth == 0 => break,
            ')' | '}' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => n += 1,
            c if !c.is_whitespace() => any = true,
            _ => {}
        }
    }
    if any { n + 1 } else { 0 }
}

/// Every `call … @name(…)` of a defined device function passes its
/// definition's arity.
fn calls_match(dev: &str, defs: &std::collections::HashMap<String, usize>) {
    for l in dev.lines() {
        let Some(c) = l.find(" call ") else { continue };
        let rest = &l[c..];
        let Some(at) = rest.find('@') else { continue };
        let tail = &rest[at + 1..];
        let Some(open) = tail.find('(') else { continue };
        let name = &tail[..open];
        if let Some(&want) = defs.get(name) {
            assert_eq!(params(&tail[open + 1..]), want, "{l}");
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    programs: u64,
    three: u64,
    bf16_w: u64,
    /// A body emitted both as a `cell-law` cell and as a member variant.
    both: u64,
    members: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.three += o.three;
        self.bf16_w += o.bf16_w;
        self.both += o.both;
        self.members += o.members;
    }
}

fn check(clang: &str, p: &testgen::SharedCellProg) -> Cover {
    let src = testgen::shared_cell_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let mut c = Cover {
        programs: 1,
        three: u64::from(p.copies == 3),
        bf16_w: u64::from(p.fw == 1),
        ..Cover::default()
    };
    for off in [&[][..], &["epilogue"][..]] {
        let em = emission(&ir, off);
        let dev = em.device.as_deref().expect("a device unit");
        let defs = definitions(dev);
        calls_match(dev, &defs);
        let built = emu::build_emulated(clang, &em.text, dev);
        for mode in MODES {
            let (got, _) = built.run(&mode);
            assert_eq!(got, rr.output, "off {off:?} {mode:?}\n{src}");
        }
        if off.is_empty() {
            let base = |n: &str| n.split('_').next().unwrap_or("").to_string();
            let cells: Vec<String> = defs
                .keys()
                .filter(|n| n.ends_with("_cell"))
                .map(|n| base(n))
                .collect();
            let members: Vec<String> = defs
                .keys()
                .filter(|n| n.contains("_u") && n.starts_with("fn"))
                .map(|n| base(n))
                .collect();
            c.members = u64::from(!members.is_empty());
            c.both = u64::from(cells.iter().any(|b| members.contains(b)));
        }
    }
    c
}

/// 16 generated chains: values equal to the interpreter in every emulator
/// mode (epilogue on and off), every device name defined once and called at
/// its arity; every counter non-zero (a body emitted as both variants).
#[test]
fn a_shared_body_keeps_each_device_variant() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::SharedCellProg> = (0..16)
        .map(|_| {
            testgen::shared_cell_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("shared-cell generated: {c:?}");
    let all = [c.programs, c.three, c.bf16_w, c.both, c.members];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
