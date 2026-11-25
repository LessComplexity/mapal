//! gapfix `stream-ld` (gaps-disasm C2): a tile operand the grid reads once
//! whose bytes exceed what the L2 keeps (`Gpu::l2_keep_bytes`) loads with the
//! streaming policy (`ld.global.cs.nc`: L2 evict-first) — never one a second
//! block (or thread) of the grid re-reads, never below the keep, never on the
//! OFF arm; values unchanged in every emulator mode.

use mapal_backend_llvm::{CostOver, DeviceCost, Emission, EmitOpts, GpuArms, emit_detailed};
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

/// The RTX 5080 with its L2 keeping `keep` bytes (0: the profile's own).
fn emission(ir: &CategoryIr, base: EmitOpts, keep: u64, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    gpu.site_cands = 1;
    let mut cost: CostOver = [None; DeviceCost::NAMES.len()];
    if keep > 0 {
        let i = DeviceCost::NAMES
            .iter()
            .position(|x| *x == "l2_keep_bytes")
            .expect("l2_keep_bytes");
        cost[i] = Some(keep);
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        cost,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// A plan line's `key=` value.
fn field(line: &str, key: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(&format!("{key}=")))
        .and_then(|v| v.parse().ok())
}

fn kernel_name(line: &str) -> &str {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix("kernel="))
        .unwrap_or("")
}

/// Each FMA kernel's plan line and whether its body has a streaming load.
fn kernels(dev: &str) -> Vec<(String, bool)> {
    let body = |name: &str| -> &str {
        let head = format!("@{name}(");
        let Some(i) = dev.find(&format!("define ptx_kernel void {head}")) else {
            return "";
        };
        let end = dev[i..].find("\n}\n").map_or(dev.len(), |e| i + e);
        &dev[i..end]
    };
    dev.lines()
        .filter(|l| {
            l.starts_with("; mapal.plan ")
                && (l.contains(" kind=naive ") || l.contains(" kind=tile "))
        })
        .map(|l| {
            let b = body(kernel_name(l));
            (l.to_string(), b.contains(" asm \"ld.global.cs.nc."))
        })
        .collect()
}

fn streams(dev: &str) -> usize {
    dev.matches(" asm \"ld.global.cs.nc.").count()
}

/// Whether a kernel's grid reads one of its operands once: `b` by one row
/// block (a thread a cell: one row), `a` by one column block.
fn read_once(l: &str) -> [bool; 2] {
    let (rows, c) = (field(l, "rows").unwrap_or(0), field(l, "c").unwrap_or(0));
    match l.contains(" kind=tile ") {
        true => {
            let br = field(l, "ty").unwrap_or(1) * field(l, "tm").unwrap_or(1);
            let bc = field(l, "tx").unwrap_or(1) * field(l, "tn").unwrap_or(1);
            [c <= bc, rows <= br]
        }
        false => [c == 1 && field(l, "cells").unwrap_or(1) == 1, rows == 1],
    }
}

/// `rules::STREAM_KEEP_SHARE` (the crate keeps its rules private): the most
/// an operand may exceed the keep by and still stream.
const STREAM_KEEP_SHARE: u64 = 4;

/// Whether a kernel's operand (`a`, `b`) qualifies at L2 keep `keep`: read
/// once, its f32 bytes past the keep and within [`STREAM_KEEP_SHARE`] times it
/// (`rules::stream_ld_pays`).
fn qualifies(l: &str, keep: u64, window: bool) -> [bool; 2] {
    let (rows, c, k) = (
        field(l, "rows").unwrap_or(0),
        field(l, "c").unwrap_or(0),
        field(l, "k").unwrap_or(0),
    );
    let share = |b: u64| b > keep && b <= STREAM_KEEP_SHARE * keep;
    let [ra, rb] = read_once(l);
    // A window `b` (`b[j + k]`) is no dense tensor: its elements are re-read
    // by up to `k` cells, never streamed whatever its dense view's bytes.
    [ra && share(rows * k * 4), rb && !window && share(k * c * 4)]
}

#[derive(Debug, Default, Clone, Copy)]
struct Cover {
    programs: u64,
    naive_stream: u64,
    tile_stream: u64,
    reread_refused: u64,
    keep_refused: u64,
    share_refused: u64,
    window_refused: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.naive_stream += o.naive_stream;
        self.tile_stream += o.tile_stream;
        self.reread_refused += o.reread_refused;
        self.keep_refused += o.keep_refused;
        self.share_refused += o.share_refused;
        self.window_refused += o.window_refused;
    }
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

fn check(clang: &str, src: &str, b_bytes: u64, window: bool) -> Cover {
    let ir = lower(src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    let mut c = Cover {
        programs: 1,
        ..Cover::default()
    };
    // The profile's own keep (tens of MB): no generated operand exceeds it.
    let base = emission(&ir, EmitOpts::exact(), 0, &[]);
    let bdev = base.device.as_deref().unwrap_or("");
    assert_eq!(
        streams(bdev),
        0,
        "below the keep: the default policy\n{src}"
    );
    c.keep_refused = u64::from(!kernels(bdev).is_empty());
    // A keep of an eighth of `b`: past the share, the default policy.
    let far = emission(&ir, EmitOpts::exact(), (b_bytes / 8).max(1), &[]);
    let fdev = far.device.as_deref().unwrap_or("");
    let fks = kernels(fdev);
    for (l, s) in &fks {
        assert_eq!(
            *s,
            qualifies(l, (b_bytes / 8).max(1), window).contains(&true),
            "{l}\n{src}"
        );
    }
    c.share_refused = u64::from(fks.iter().any(|(l, s)| !s && read_once(l)[1]));
    // A keep of half of `b`: `b`, where read once, within the share.
    let keep = (b_bytes / 2).max(1);
    for off in [&[][..], &["stream-ld"][..]] {
        let e = emission(&ir, EmitOpts::exact(), keep, off);
        let dev = e.device.as_deref().unwrap_or("").to_string();
        let ks = kernels(&dev);
        if !off.is_empty() {
            assert_eq!(streams(&dev), 0, "OFF: the default policy\n{src}");
        }
        for (l, s) in &ks {
            // The plan line names what the kernel streams (`evict=`).
            assert_eq!(*s, l.contains(" evict="), "plan line vs text: {l}\n{src}");
            if off.is_empty() {
                // Sound and complete: an f32 FMA kernel streams exactly the
                // operands read once whose bytes the rule admits.
                let q = qualifies(l, keep, window).contains(&true);
                assert_eq!(*s, q, "streamed {s}, qualifies {q}: {l}\n{src}");
                c.naive_stream += u64::from(*s && l.contains(" kind=naive "));
                c.tile_stream += u64::from(*s && l.contains(" kind=tile "));
                c.reread_refused += u64::from(!s && !read_once(l)[1]);
                // The dense view's bytes would stream; the window does not.
                c.window_refused += u64::from(window && !s && qualifies(l, keep, false)[1]);
            }
        }
        let built = emu::build_emulated(clang, &e.text, &dev);
        for mode in MODES {
            let (got, _) = built.run(&mode);
            assert_eq!(got, want, "exact {off:?} {mode:?}\n{src}");
        }
    }
    c
}

/// 40 generated products (one-row GEMVs to 70-row tiles, both `b` layouts)
/// and 12 whose `b` is a window (`b[j + k]`, the FIR's signal — gap-fixes
/// merge: its dense view's bytes overstated the operand and streamed it):
/// streaming loads exactly on the operands their grid reads once, past the
/// keep, never on a window; the interpreter's bytes on both arms in every
/// emulator mode; every coverage counter non-zero.
#[test]
fn stream_ld_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<(String, u64, bool)> = (0..52u8)
        .map(|i| {
            let p = testgen::gemm_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current();
            let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
            let m = pick(p.dims[0] ^ i, &[1, 1, 2, 5, 8, 16, 16, 40, 70]);
            let n = pick(p.dims[1], &[64, 96, 256, 1024, 2048]);
            let k = pick(p.dims[2], &[16, 64, 100, 128]);
            match i < 40 {
                true => (testgen::gemm_src_at(&p, m, n, k), k * n * 4, false),
                false => (testgen::gemm_window_src_at(&p, m, n, k), k * n * 4, true),
            }
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, (s, b, w)| check(&clang, s, *b, *w));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("stream-ld generated: {c:?}");
    let all = [
        c.programs,
        c.naive_stream,
        c.tile_stream,
        c.reread_refused,
        c.keep_refused,
        c.share_refused,
        c.window_refused,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
