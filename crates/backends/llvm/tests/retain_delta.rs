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

use mapal_backend_llvm::{EmitOpts, emit_detailed};
use mapal_ir::{CategoryIr, Delta};

/// The decode loop over `elem` (`f32` or `i32`), `T` steps of width `d`; with
/// `column`, the iteration writes `x[j·d + t]` instead of row `t` (`Top`).
fn decode(t_steps: u64, d: u64, elem: &str, column: bool) -> String {
    let float = elem == "f32";
    let (zero, w) = if float {
        ("0.0", "x / 128.0")
    } else {
        ("0", "x % 7")
    };
    let mut row = String::new();
    for j in 0..d {
        let idx = if column {
            format!("{j} * {d} + t")
        } else {
            format!("t * {d} + {j}")
        };
        let v = if float {
            format!("((t * {d} + {j}) * 7 + 13) % 101 - 50 -> widen_f32 -> f{j}; f{j} / 64.0")
        } else {
            format!("((t * {d} + {j}) * 7 + 13) % 101 - 50")
        };
        row.push_str(&format!("{v} + o[{j}] -> v{j}; x[{idx}] <- v{j};\n"));
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

/// `(host, device)` on `target`, rung on or off.
fn emit(ir: &CategoryIr, target: &'static str, on: bool) -> (String, Option<String>) {
    let mut opts = EmitOpts {
        target,
        ..EmitOpts::default()
    };
    opts.gpu.retain_delta = on;
    let em = emit_detailed(ir, &opts).expect("emits");
    (em.text, em.device)
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

#[test]
fn column_write_is_byte_identical_on_and_off() {
    let ir = lower(&decode(16, 8, "f32", true));
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

fn clang() -> Option<String> {
    if let Ok(cc) = std::env::var("CC") {
        return Some(cc);
    }
    let found = Command::new("which")
        .arg("clang")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    assert!(
        !(found.is_none() && std::env::var("MAPAL_REQUIRE_CLANG").is_ok_and(|v| v != "0")),
        "MAPAL_REQUIRE_CLANG is set but clang was not found"
    );
    found
}

fn rt_lib() -> std::path::PathBuf {
    static BUILD: std::sync::Once = std::sync::Once::new();
    BUILD.call_once(|| {
        let ok = Command::new("cargo")
            .args(["build", "-p", "mapal-rt"])
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "cargo build -p mapal-rt failed");
    });
    format!(
        "{}/../../../target/debug/libmapal_rt.a",
        env!("CARGO_MANIFEST_DIR")
    )
    .into()
}
