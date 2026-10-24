//! S53 `retain-delta`: retention at part-of-object scope, at the text level and
//! against the oracle.
//!
//! The decode loop (the shape `benches/gpu/gen_tiles.py` emits) writes row `t`
//! of a carried `x`, then computes `K = x·Wk`, `V = x·Wv` (row-local) and the
//! row-mixing `s = q·Kᵀ`, `o = s·V`. The rule must retain `K` and `V` — one
//! whole pass, then the written row — and nothing else; a loop that writes a
//! column instead must emit byte-identically with the rung on and off.
//!
//! The float program's restricted pass is a GPU launch, which only a box can
//! run (`benches/gpu/box_run.sh` differential). The `i32` program takes the
//! same rule on the same GPU profile through the host realization (an int
//! site is below the GPU gate), so the rule's whole run-time half — the
//! first-pass flag, the row arithmetic, the restricted bound — is checked here
//! against the interpreter, ON and OFF.

use std::process::Command;

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, Delta};

mod emu;
use emu::{clang, rt_lib, run_emulated};

/// The decode loop over `elem` (`f32` or `i32`), `T` steps of width `d`, in
/// its natural form (`benches/gpu/gen_tiles.py`): the row write is one map
/// over the cells; with `column`, the iteration writes column `t % d` of
/// every row instead of row `t` (`Top`).
fn decode(t_steps: u64, d: u64, elem: &str, column: bool) -> String {
    decode_form(t_steps, d, elem, column, false)
}

/// [`decode`] with the write as unrolled `Update`s (`unrolled`: `d` cells of
/// row `t`, or `T` of the column) — the shape the suite used before its
/// generators were made idiomatic; the facts must give both the same code.
fn decode_form(t_steps: u64, d: u64, elem: &str, column: bool, unrolled: bool) -> String {
    let float = elem == "f32";
    let (zero, w) = if float {
        ("0.0", "x / 128.0")
    } else {
        ("0", "x % 7")
    };
    // The new cell `v` of row `j` and column `c`: `(t·d + j)`'s formula plus
    // the previous output's element `j % d`.
    let cell = |j: &str, k: &str| {
        let f = format!("(({j}) * 7 + 13) % 101 - 50");
        if float {
            format!("{f} -> widen_f32 -> f{k}; f{k} / 64.0")
        } else {
            f
        }
    };
    let mut row = if column { format!("t % {d} -> col;\n") } else { String::new() };
    if !unrolled {
        row += &if column {
            format!(
                "tc -> map {{ c -> (c % {d} == col) -> {{ -true-> {{ {v} + o[c / {d} % {d}] }} -false-> x[c]; }} }} -> x;\n",
                v = cell(&format!("t * {d} + c / {d}"), "")
            )
        } else {
            format!(
                "tc -> map {{ c -> (c / {d} == t) -> {{ -true-> {{ {v} + o[c % {d}] }} -false-> x[c]; }} }} -> x;\n",
                v = cell("c", "")
            )
        };
    }
    let cells = if column { t_steps } else { d };
    for j in (0..cells).filter(|_| unrolled) {
        let (idx, v) = if column {
            (format!("{j} * {d} + col"), cell(&format!("t * {d} + {j}"), &j.to_string()))
        } else {
            (format!("t * {d} + {j}"), cell(&format!("t * {d} + {j}"), &j.to_string()))
        };
        row.push_str(&format!("{v} + o[{}] -> v{j}; x[{idx}] <- v{j};\n", j % d));
    }
    let wf = |m: u64, a: u64| {
        let x = format!("(t * {m} + {a}) % 101 - 50");
        if float {
            format!("{x} -> widen_f32 -> x; {w}")
        } else {
            format!("{x} -> x; {w}")
        }
    };
    let proj = |wname: &str| {
        format!(
            "tc -> map {{ c -> c / {d} -> i; c % {d} -> j; ({zero}, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * {wname}[k * {d} + j] }} }}"
        )
    };
    let q = if float {
        format!("((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / 128.0")
    } else {
        format!("((t * {d} + k) * 5 + 3) % 9 - 4")
    };
    format!(
        "fn main() {{
    {dd} -> iota -> tw;
    tw -> map {{ t -> {wk} }} -> wk;
    tw -> map {{ t -> {wv} }} -> wv;
    {td} -> iota -> tc;
    {d} -> iota -> krange;
    {t_steps} -> iota -> rrange;
    ({zero}, {td}) -> fill -> x0;
    ({zero}, {d}) -> fill -> o0;
    mut x: [{elem}; {td}] <- x0;
    mut o: [{elem}; {d}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {t_steps}) -> {{
            -true-> {{
{row}                {kk} -> kk;
                {vv} -> vv;
                krange -> map {{ k -> {q} }} -> q;
                rrange -> map {{ r -> ({zero}, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;
                krange -> map {{ j -> ({zero}, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{dl}] -> println;
}}
",
        dd = d * d,
        td = t_steps * d,
        dl = d - 1,
        wk = wf(7, 13),
        wv = wf(11, 57),
        kk = proj("wk"),
        vv = proj("wv"),
    )
}

fn lower_raw(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

/// Rewritten, as the bundle emits it (`emit --rewrite`).
fn lower(src: &str) -> CategoryIr {
    mapal_rewrite::rewrite(lower_raw(src)).ir
}

/// `(host, device)` on `target`, rung on or off — `resident` off in both, so
/// every site is the S52 round trip the rung was measured against, and
/// `lane-stride` off, so `s = q·Kᵀ` stays the host map S53 measured.
fn emit(ir: &CategoryIr, target: &'static str, on: bool) -> (String, Option<String>) {
    let mut opts = EmitOpts {
        host: "generic",
        target,
        ..EmitOpts::exact()
    };
    opts.gpu.on.retain_delta = on;
    opts.gpu.off.retain_delta = !on;
    opts.gpu.off.resident = true;
    opts.gpu.off.lane_stride = true;
    // The S53 mechanism on the S52-S54 host: with S62 `hostopt`, a GPU build
    // with no device site (the all-int decode) is the CPU build, which never
    // retains.
    opts.gpu.off.hostopt = true;
    let em = emit_detailed(ir, &opts).expect("emits");
    (em.text, em.device)
}

/// `(host, device)` on the 4090 with S54's three R5 rungs forced on (the
/// mechanism, not the 4090's measured table, which ships `retain-resident`
/// off), then the measurement arms `off`, on `face`.
fn emit_s54(ir: &CategoryIr, off: &[&str], contract: bool) -> (String, String) {
    let mut opts = EmitOpts {
        host: "generic",
        target: "rtx-4090",
        contract,
        ..EmitOpts::exact()
    };
    opts.gpu.on.retain_resident = true;
    opts.gpu.on.delta_h2d = true;
    opts.gpu.on.lane_stride = true;
    for r in off {
        *opts.gpu.off.rung(r).expect("a rung") = true;
    }
    let em = emit_detailed(ir, &opts).expect("emits");
    (em.text, em.device.expect("a device unit"))
}

fn calls<'a>(host: &'a str, entry: &str) -> Vec<&'a str> {
    host.lines()
        .filter(|l| l.contains(&format!("call void @mapal_gpu_{entry}(")))
        .collect()
}

fn tile_calls(host: &str) -> Vec<&str> {
    host.lines()
        .filter(|l| l.contains("call void @mapal_gpu_tile("))
        .collect()
}

#[test]
fn write_delta_on_sources() {
    // matmul4_loop's `c[t] <- v` in `matmul`: one cell at the counter.
    let src = std::fs::read_to_string(format!(
        "{}/../../../examples/matmul4_loop.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    // Raw: `rewrite` evaluates this all-constant program's loops away.
    let ir = lower_raw(&src);
    let all: Vec<Delta> = ir
        .funcs()
        .flat_map(|(f, _)| ir.write_delta(f).values().copied().collect::<Vec<_>>())
        .collect();
    assert!(
        matches!(all.as_slice(), [Delta::Rows { stride: 1, start, .. }] if start.a == 1 && start.b == 0),
        "{all:?}"
    );
    // The decode loop: `x` is row t, `d` cells from `t·d`; the carried `o`
    // is rewritten whole by its map (Top). The column write is Top too.
    let ir = lower(&decode(16, 8, "f32", false));
    let wd: Vec<Delta> = ir.write_delta(ir.entry()).values().copied().collect();
    assert!(
        matches!(wd.as_slice(), [Delta::Rows { stride: 8, start, .. }, Delta::Top] if start.a == 8 && start.b == 0),
        "{wd:?}"
    );
    let ir = lower(&decode(16, 8, "f32", true));
    let wd: Vec<Delta> = ir.write_delta(ir.entry()).values().copied().collect();
    assert_eq!(wd, [Delta::Top, Delta::Top]);
}

/// generic-fusion natural forms: the decode with its write as one map gets
/// the unrolled `Update`s' facts and code — the same `Δ` (row `t`; the column
/// `Top`), the same per-trip movement and storage (`; mapal.move`: the state
/// written in place) and launches on every profile — and its row write runs
/// in place over the row: `max(8t, 0) .. min(8t + 8, 128)`.
#[test]
fn natural_decode_moves_what_the_unrolled_one_moves() {
    let deltas = |ir: &CategoryIr| -> Vec<Option<(u64, i64, i64)>> {
        ir.write_delta(ir.entry())
            .values()
            .map(|d| match *d {
                Delta::Rows { stride, start, .. } => Some((stride, start.a, start.b)),
                Delta::Top => None,
            })
            .collect()
    };
    let emit_on = |ir: &CategoryIr, target: &'static str| {
        let mut opts = EmitOpts {
            host: "generic",
            target,
            ..EmitOpts::exact()
        };
        for r in ["async", "streams", "pinned", "pdl"] {
            *opts.gpu.off.rung(r).expect("rung") = true;
        }
        emit_detailed(ir, &opts).expect("emits").text
    };
    let moves = |t: &str| -> Vec<String> {
        t.lines()
            .filter(|l| l.starts_with("; mapal.move"))
            .map(|l| l.split_once(" carried=").map_or(l, |(_, r)| r).to_string())
            .collect()
    };
    let launches = |t: &str| t.matches("call void @mapal_gpu_launch").count() + tile_calls(t).len();
    for (elem, column) in [("f32", false), ("i32", false), ("f32", true)] {
        let nat = lower(&decode_form(16, 8, elem, column, false));
        let unr = lower(&decode_form(16, 8, elem, column, true));
        assert_eq!(deltas(&nat), deltas(&unr), "{elem} column={column}");
        for target in ["rtx-4090", "h100-pcie", "rtx-3090", "rtx-5090", "generic"] {
            let (a, b) = (emit_on(&nat, target), emit_on(&unr, target));
            assert_eq!(moves(&a), moves(&b), "{elem} column={column} {target}");
            assert_eq!(launches(&a), launches(&b), "{elem} column={column} {target}");
        }
    }
    let nat = lower(&decode(16, 8, "f32", false));
    assert!(deltas(&nat).contains(&Some((8, 8, 0))), "{:?}", deltas(&nat));
    let host = emit_on(&nat, "generic");
    assert!(moves(&host).iter().all(|l| l.starts_with("0:512:inplace")), "{:?}", moves(&host));
    let clipped = host.lines().filter(|l| l.contains(" = select i1 ") && l.ends_with(", i64 128")).count();
    assert_eq!(clipped, 1, "one row-write loop clipped to the array");
}

#[test]
fn decode_retains_k_and_v_only() {
    let ir = lower(&decode(16, 8, "f32", false));
    let (on, on_dev) = emit(&ir, "rtx-4090", true);
    let (off, off_dev) = emit(&ir, "rtx-4090", false);
    // OFF: K, V (16×8 out, 512 B) and o = s·V (8 out, 32 B); s is a host map.
    let off_calls = tile_calls(&off);
    assert_eq!(off_calls.len(), 3, "{off_calls:#?}");
    // ON: K and V each get the whole pass AND a one-row pass (8-float row of
    // `x` in, 8-float row of the output out); o = s·V is untouched.
    let on_calls = tile_calls(&on);
    assert_eq!(on_calls.len(), 5, "{on_calls:#?}");
    let rows = on_calls
        .iter()
        .filter(|l| l.matches("i64 32,").count() == 2)
        .count();
    assert_eq!(rows, 2, "{on_calls:#?}");
    assert_eq!(
        on.matches("store i1 true, ptr").count(),
        2,
        "two first-pass flags"
    );
    assert_eq!(
        on.matches("store i1 false, ptr").count(),
        2,
        "reset at loop entry"
    );
    // The device unit gains exactly the one-row kernel (K and V share it).
    let kernels = |d: &Option<String>| d.as_ref().unwrap().matches("void @mapal_tile_").count();
    assert_eq!(kernels(&on_dev), kernels(&off_dev) + 1);
}

/// `retain-delta` on with `retain-resident` off (`resident` on): a function
/// with a retained site keeps the round trips the S53 rung was measured with.
/// Shipped (S54): `retain-resident` keeps it resident, retention included.
#[test]
fn resident_yields_to_a_retained_site_only_without_retain_resident() {
    let ir = lower(&decode(16, 8, "f32", false));
    let at = |gpu: GpuArms| {
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
        .text
    };
    let mut s53 = GpuArms::default();
    s53.on.retain_delta = true;
    s53.off.retain_resident = true;
    s53.off.lane_stride = true;
    let s53 = at(s53);
    assert_eq!(tile_calls(&s53).len(), 5);
    assert!(!s53.contains("@mapal_gpu_alloc("));
    let mut r5 = GpuArms::default();
    r5.on.retain_resident = true;
    let r5 = at(r5);
    assert!(tile_calls(&r5).is_empty() && r5.contains("@mapal_gpu_launch("));
    assert_eq!(r5.matches("store i1 true, ptr").count(), 3, "K, V, x");
    // The 4090's table retains since S65 (its row left GPU_RUNGS_OFF): shipped = r5.
    assert_eq!(at(GpuArms::default()), r5);
}

// --- S54 `retain-resident`, `delta-h2d`, `lane-stride` ----------------------

/// Shipped: K and V retained in their resident buffers (a whole pass and a
/// row pass each, both launches), `x` uploaded as its written row after one
/// whole upload, `s = q·Kᵀ` a device site — so the only copy-out in the loop
/// is the 8-float `o` the host's next row reads, and K never crosses.
#[test]
fn decode_retains_on_the_resident_buffers() {
    let ir = lower(&decode(16, 8, "f32", false));
    let (host, _) = emit_s54(&ir, &[], false);
    assert!(tile_calls(&host).is_empty());
    // K, V: whole + row; s; o.
    assert_eq!(calls(&host, "launch").len(), 6);
    // Three first-pass flags (K, V, x's upload), reset at loop entry.
    assert_eq!(host.matches("store i1 false, ptr").count(), 3);
    assert_eq!(host.matches("store i1 true, ptr").count(), 3);
    let d2h = calls(&host, "d2h");
    assert_eq!(d2h.len(), 1, "{d2h:#?}");
    assert!(d2h[0].ends_with("i64 0, i64 32)"), "{d2h:#?}");
    // The device unit gains the one-row kernel (K and V share it) and `s`'s.
    let (_, dev) = emit_s54(&ir, &[], false);
    let (_, base) = emit_s54(&ir, &["retain-resident", "delta-h2d", "lane-stride"], false);
    let kernels = |d: &str| d.matches("void @mapal_tile_").count();
    assert_eq!(kernels(&dev), kernels(&base) + 2);
}

/// `delta-h2d`: the written row crosses as `stride · elem` bytes at the row's
/// own byte offset on both sides; `q` (rewritten whole by its map) and the
/// first pass cross whole; off, `x` crosses whole every iteration.
#[test]
fn delta_upload_is_stride_times_elem_bytes() {
    for (d, row, whole) in [(8u64, 32, 512), (4, 16, 256)] {
        let ir = lower(&decode(16, d, "f32", false));
        let (host, _) = emit_s54(&ir, &[], false);
        // `x` and `q` are both one row: the written row of `x` and `q`'s
        // whole map cross at `row` bytes, the first pass whole. (Since S65
        // `by-ref` the carried `x` is reached through its pointer, so the host
        // operand no longer tells the row from the whole.)
        let h2d = calls(&host, "h2d");
        let rows = |h: &[&str]| h.iter().filter(|l| l.ends_with(&format!("i64 {row})"))).count();
        assert_eq!(rows(&h2d), 2, "{h2d:#?}");
        assert_eq!(
            h2d.iter()
                .filter(|l| l.ends_with(&format!("i64 {whole})")))
                .count(),
            1
        );
        let (off_arm, _) = emit_s54(&ir, &["delta-h2d"], false);
        assert_eq!(rows(&calls(&off_arm, "h2d")), 1, "q only");
        assert_eq!(
            off_arm.matches("store i1 true, ptr").count(),
            2,
            "K, V only"
        );
    }
}

/// The negative control: a column write has no `Δ::Rows`, so nothing is
/// retained or delta-uploaded — the R5 rungs are byte-identical on and off.
#[test]
fn column_write_retains_nothing_on_the_resident_path() {
    let ir = lower(&decode(16, 8, "f32", true));
    let (on, on_dev) = emit_s54(&ir, &[], false);
    assert!(!on.contains("store i1 true, ptr") && !on.contains("store i1 false, ptr"));
    for off in ["retain-resident", "delta-h2d"] {
        assert_eq!(emit_s54(&ir, &[off], false), (on.clone(), on_dev.clone()));
    }
}

/// `lane-stride`: `s[r] = Σ_k q[k]·K[r·d + k]` reads K across lanes at stride
/// `d` — `tile_plan` refuses it, `tile_plan_strided` takes it as a one-row
/// site, and on a GPU profile it runs on the device: K then never crosses.
/// Off, K is copied out whole on the first pass and as its new row after.
#[test]
fn lane_stride_places_q_kt_on_the_device() {
    let ir = lower(&decode(16, 8, "f32", false));
    let f = ir.entry();
    let (strict, strided) = (ir.tile_plan(f), ir.tile_plan_strided(f));
    assert_eq!(strided.sites.len(), strict.sites.len() + 1);
    let s = strided
        .sites
        .iter()
        .find(|(m, _)| !strict.sites.contains_key(*m))
        .map(|(_, s)| s.clone())
        .unwrap();
    assert_eq!((s.rows, s.c, s.k), (1, 16, 8));
    assert_eq!((s.a.clane, s.b.clane, s.b.ck), (0, 8, 1));
    assert!(strict.sites.values().all(|s| s.b.clane == 1));

    let (on, _) = emit_s54(&ir, &[], false);
    let (off, _) = emit_s54(&ir, &["lane-stride"], false);
    assert_eq!(calls(&on, "launch").len(), calls(&off, "launch").len() + 1);
    let d2h = calls(&off, "d2h");
    // K whole (512 B) on the first pass, its row (32 B at the row's offset)
    // after, and o.
    assert_eq!(d2h.len(), 3, "{d2h:#?}");
    assert!(d2h.iter().any(|l| l.ends_with("i64 0, i64 512)")));
    assert!(
        d2h.iter()
            .any(|l| l.contains("ptr %t") && l.ends_with(", i64 32)"))
    );
    // An int product stays on the host: below the GPU gate, not a device site.
    let int_ir = lower(&decode(16, 8, "i32", false));
    let int = |off: &[&str]| {
        let mut opts = EmitOpts {
            host: "generic",
            target: "rtx-4090",
            ..EmitOpts::exact()
        };
        for r in off {
            *opts.gpu.off.rung(r).unwrap() = true;
        }
        emit_detailed(&int_ir, &opts).unwrap().text
    };
    assert_eq!(int(&[]), int(&["lane-stride"]));
}

/// The contract face fuses the step of the sites the CPU tiles; the strided
/// site is a host map on the CPU, so its device step stays `fmul` + `fadd`
/// (placement never changes a value).
#[test]
fn strided_site_keeps_the_host_step_on_the_contract_face() {
    let src = "fn main() {
    128 -> iota -> tk;
    tk -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 } -> kk;
    8 -> iota -> krange;
    16 -> iota -> rrange;
    krange -> map { k -> k -> widen_f32 -> v; v / 8.0 } -> q;
    rrange -> map { r -> (0.0, krange) -> fold { acc, k -> acc + q[k] * kk[r * 8 + k] } } -> s;
    s[0] -> println;
    s[15] -> println;
}
";
    let ir = lower(src);
    let (_, dev) = emit_s54(&ir, &[], true);
    let kernels: Vec<&str> = dev
        .split("define ptx_kernel void @mapal_tile_")
        .skip(1)
        .collect();
    assert_eq!(kernels.len(), 1);
    let k = kernels[0].split("\n}\n").next().unwrap();
    assert!(!k.contains("@llvm.fma") && k.contains("fmul float") && k.contains("fadd float"));
}

/// `write_delta`'s Same consumers: `xh = x·0.5` (a captures-0 map of the
/// written `x`) carries `x`'s `Δ`, so K and V over `xh` are retained and `xh`
/// (not `x`) is delta-uploaded; a map capturing the counter is not Same.
#[test]
fn write_delta_passes_through_a_same_map() {
    // K and V read `xh`; the map is the line before K.
    let scaled = decode(16, 8, "f32", false)
        .replace("acc + x[i * 8 + k]", "acc + xh[i * 8 + k]")
        .replacen(
            "                tc -> map",
            "x -> map { v -> v * 0.5 } -> xh;\n                tc -> map",
            1,
        );
    let ir = lower(&scaled);
    let wd = ir.write_delta(ir.entry());
    let rows: Vec<(u64, i64, i64, bool)> = wd
        .iter()
        .filter_map(|(k, d)| match *d {
            Delta::Rows {
                stride,
                start,
                next,
            } => Some((stride, start.a, start.b, next == k)),
            Delta::Top => None,
        })
        .collect();
    assert_eq!(rows.len(), 2, "{wd:?}");
    assert_eq!(rows[0].0, rows[1].0);
    assert_eq!((rows[0].1, rows[0].2), (rows[1].1, rows[1].2));
    assert_eq!(
        rows.iter().filter(|r| r.3).count(),
        1,
        "the image keys itself"
    );
    let (host, _) = emit_s54(&ir, &[], false);
    assert_eq!(host.matches("store i1 true, ptr").count(), 3, "K, V, xh");
    assert_eq!(calls(&host, "launch").len(), 6);

    // `v · t` captures the counter: every cell changes, so no Δ — nothing is
    // retained, and `xh` crosses whole (`x` is no device operand at all).
    let captured = scaled.replace("v -> v * 0.5", "v -> t -> widen_f32 -> tf; v * tf");
    let ir = lower(&captured);
    let wd: Vec<Delta> = ir.write_delta(ir.entry()).values().copied().collect();
    assert_eq!(
        wd.iter()
            .filter(|d| matches!(d, Delta::Rows { .. }))
            .count(),
        1
    );
    let (host, _) = emit_s54(&ir, &[], false);
    assert!(!host.contains("store i1 true"));
}

#[test]
fn column_write_is_byte_identical_on_and_off() {
    let ir = lower(&decode(16, 8, "f32", true));
    assert_eq!(emit(&ir, "rtx-4090", true), emit(&ir, "rtx-4090", false));
}

#[test]
fn site_produced_carry_is_top_and_byte_identical() {
    // loop_mm: the carried `x` is the site's own output (`x·W -> x`), so the
    // back edge brings a whole new array, not an `Update` chain: Top.
    let src = "fn main() {
    256 -> iota -> tw;
    tw -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 256.0 } -> w;
    256 -> iota -> tc;
    16 -> iota -> krange;
    (0.5, 256) -> fill -> x0;
    mut x: [f32; 256] <- x0;
    mut t: i32 <- 0;
    loop {
        (t < 8) -> {
            -true-> {
                tc -> map { c -> c / 16 -> i; c % 16 -> j; (0.0, krange) -> fold { acc, k -> acc + x[i * 16 + k] * w[k * 16 + j] } } -> x;
                t + 1 -> t;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    out[0] -> println;
}
";
    let ir = lower(src);
    let wd: Vec<Delta> = ir.write_delta(ir.entry()).values().copied().collect();
    assert_eq!(wd, [Delta::Top]);
    let (on, _) = emit(&ir, "rtx-4090", true);
    assert_eq!(tile_calls(&on).len(), 1, "the site is a GPU site");
    assert_eq!(emit(&ir, "rtx-4090", true), emit(&ir, "rtx-4090", false));
}

#[test]
fn cpu_profiles_are_untouched() {
    // Unmeasured on any CPU part: the rung does not fire there.
    let ir = lower(&decode(16, 8, "f32", false));
    assert_eq!(emit(&ir, "generic", true), emit(&ir, "generic", false));
}

#[test]
fn int_decode_matches_the_oracle_on_and_off() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let ir = lower(&decode(16, 8, "i32", false));
    let rr = mapal_interp::run(&ir, 100_000_000);
    assert!(
        matches!(rr.outcome, mapal_interp::Outcome::Done(_)),
        "{:?}",
        rr.outcome
    );
    let (on, _) = emit(&ir, "rtx-4090", true);
    let (off, _) = emit(&ir, "rtx-4090", false);
    assert!(
        on.contains("store i1 true, ptr"),
        "the rule fires on the int sites"
    );
    assert_ne!(on, off);
    for ll in [on, off] {
        let dir = tempfile::tempdir().unwrap();
        let (llp, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
        std::fs::write(&llp, &ll).unwrap();
        let st = Command::new(&clang)
            .args(["-O2", "-Wno-override-module"])
            .arg(&llp)
            .arg(rt_lib())
            .arg("-o")
            .arg(&exe)
            .status()
            .unwrap();
        assert!(st.success(), "clang failed");
        let out = Command::new(&exe).output().unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), rr.output);
    }
}

/// The resident path's host half end to end without a GPU (S54): the device
/// unit runs on the host, thread by thread, through `mapal_gpu_emu.c`, linked
/// with the host module. Exact face against the interpreter, every R5 arm, on
/// decode, its Same-map variant and the column-write control; the contract
/// face against the CPU's contract build.
#[test]
fn resident_path_matches_the_oracle_emulated() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let plain = decode(16, 8, "f32", false);
    let scaled = plain
        .replace("acc + x[i * 8 + k]", "acc + xh[i * 8 + k]")
        .replacen(
            "                tc -> map",
            "x -> map { v -> v * 0.5 } -> xh;\n                tc -> map",
            1,
        );
    let column = decode(16, 8, "f32", true);
    let arms: [&[&str]; 5] = [
        &[],
        &["retain-resident"],
        &["delta-h2d"],
        &["lane-stride"],
        &["retain-resident", "delta-h2d", "lane-stride"],
    ];
    for src in [&plain, &scaled, &column] {
        let ir = lower(src);
        let rr = mapal_interp::run(&ir, 100_000_000);
        assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
        for off in arms {
            let (host, dev) = emit_s54(&ir, off, false);
            assert_eq!(run_emulated(&clang, &host, &dev), rr.output, "{off:?}");
        }
    }
    // Contract face: the CPU build with `--contract` is the oracle.
    let ir = lower(&plain);
    let cpu = emit_detailed(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            contract: true,
            ..EmitOpts::exact()
        },
    )
    .unwrap()
    .text;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("c.ll"), cpu).unwrap();
    let ok = Command::new(&clang)
        .args(["-O1", "-Wno-override-module"])
        .arg(dir.path().join("c.ll"))
        .arg(rt_lib())
        .arg("-o")
        .arg(dir.path().join("c"))
        .status()
        .unwrap()
        .success();
    assert!(ok);
    let want = Command::new(dir.path().join("c")).output().unwrap().stdout;
    for off in arms {
        let (host, dev) = emit_s54(&ir, off, true);
        assert_eq!(
            run_emulated(&clang, &host, &dev),
            String::from_utf8_lossy(&want),
            "contract {off:?}"
        );
    }
}
