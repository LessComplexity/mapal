//! local-rtx-5080 `named-host-dispatch`: a build's dispatch economics come
//! only from the profiles it names. `native` reads this machine's measured
//! dispatch (`MAPAL_NATIVE_DISPATCH`, else the per-host cache
//! `$XDG_CACHE_HOME/mapal/native-dispatch.tsv`); a named host (`generic`,
//! `x86-64-v3` — what a bundle cross-compiled here for a rented box names) and
//! a named CPU target never do, so their text is the same whatever this
//! machine measured.
//!
//! The env var and `native`'s `OnceLock` are process-global, so each dispatch
//! value is one child process of this binary ([`emit_child`], a no-op unless
//! [`OUT`] is set), each with its own cache directory: the parent compares the
//! children's texts and never reads or writes the real cache.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use mapal_backend_llvm::{EmitOpts, emit_detailed};
use mapal_ir::CategoryIr;

/// The child's output file; unset, [`emit_child`] emits nothing.
const OUT: &str = "MAPAL_TEST_DISPATCH_OUT";

/// Economics no part has: a free wave over lanes faster than any device
/// (`place` keeps the small sites on the host, tasks run inline), and a
/// millisecond wave over one-flop lanes.
const FAST: &str = "100,8,100000,100000";
const SLOW: &str = "1000000,8,1,1";

/// `(target, host, reads native)`: a named GPU target over the named hosts,
/// named CPU targets (a CPU target ignores `host`), and the `native` builds
/// that must see the economics.
const BUILDS: [(&str, &str, bool); 6] = [
    ("rtx-4090", "generic", false),
    ("rtx-4090", "x86-64-v3", false),
    ("generic", "native", false),
    ("x86-64-v3", "native", false),
    ("rtx-4090", "native", true),
    ("native", "native", true),
];

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

/// A launch-bound attention (`place` weighs it against the host) and a large
/// elementwise site (the task width weighs it against a wave).
fn programs() -> [(&'static str, String); 2] {
    let attn = std::fs::read_to_string(format!(
        "{}/../../../benches/algorithms/px_attn_64_16.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let saxpy = "fn main() {
    65536 -> iota -> ix;
    ix -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 } -> x;
    ix -> map { t -> (t * 5 + 3) % 31 - 15 -> widen_f32 } -> y0;
    (x, y0) -> zip -> map { p -> 2.5 * p.0 + p.1 } -> y;
    y[0] -> println;
    y[65535] -> println;
}
";
    [("px_attn_64_16", attn), ("saxpy_65536", saxpy.to_string())]
}

/// The child: every program on every build, host and device text, to [`OUT`].
#[test]
fn emit_child() {
    let Some(out) = std::env::var_os(OUT) else {
        return;
    };
    let mut text = String::new();
    for (name, src) in programs() {
        let ir = lower(&src);
        for (target, host, _) in BUILDS {
            let opts = EmitOpts {
                target,
                host,
                ..EmitOpts::default()
            };
            let em = emit_detailed(&ir, &opts).expect("emits");
            text += &format!(
                "=== {name} {target}+{host}\n{}\n--- device\n{}\n",
                em.text,
                em.device.unwrap_or_default()
            );
        }
    }
    std::fs::write(out, text).unwrap();
}

/// One child run: `dispatch` as `MAPAL_NATIVE_DISPATCH` (`None` = unset, the
/// cache), `cache` as `XDG_CACHE_HOME`; its texts by `program target+host`.
fn child(dir: &Path, tag: &str, dispatch: Option<&str>, cache: &Path) -> BTreeMap<String, String> {
    let out = dir.join(format!("{tag}.txt"));
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["emit_child", "--exact", "--test-threads=1"])
        .env(OUT, &out)
        .env("XDG_CACHE_HOME", cache);
    match dispatch {
        Some(v) => cmd.env("MAPAL_NATIVE_DISPATCH", v),
        None => cmd.env_remove("MAPAL_NATIVE_DISPATCH"),
    };
    let run = cmd.output().unwrap();
    assert!(
        run.status.success(),
        "{tag}: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = std::fs::read_to_string(&out).unwrap();
    text.split("=== ")
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (key, body) = s.split_once('\n').unwrap();
            (key.to_string(), body.to_string())
        })
        .collect()
}

#[test]
fn named_builds_ignore_native_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let off = child(dir.path(), "off", Some("off"), &cache);
    let fast = child(dir.path(), "fast", Some(FAST), &cache);
    let slow = child(dir.path(), "slow", Some(SLOW), &cache);
    // A miss (unset, an empty cache) measures; a debug build (this one under
    // `cargo test`) times its own unoptimized code, so it caches nothing.
    let path = cache.join("mapal").join("native-dispatch.tsv");
    let missed = child(dir.path(), "miss", None, &cache);
    if cfg!(debug_assertions) {
        assert!(!path.exists(), "a debug build cached its measurement");
    }
    let _ = std::fs::remove_file(&path);
    // `measure`: this machine's own economics, written to the scratch cache.
    let measured = child(dir.path(), "measure", Some("measure"), &cache);
    assert_eq!(off.len(), 2 * BUILDS.len());

    let named = programs()
        .map(|(name, _)| name)
        .into_iter()
        .flat_map(|name| {
            BUILDS
                .iter()
                .filter(|b| !b.2)
                .map(move |(target, host, _)| format!("{name} {target}+{host}"))
        });
    for key in named {
        for (arm, other) in [
            ("fast", &fast),
            ("slow", &slow),
            ("miss", &missed),
            ("measure", &measured),
        ] {
            assert!(other[&key] == off[&key], "{key}: {arm} differs from off");
        }
    }

    // The comparison is not vacuous: the same programs on a `native` host /
    // target do see the economics (`place` keeps the attention on the CPU; the
    // elementwise site runs inline instead of waking the pool).
    for key in ["px_attn_64_16 rtx-4090+native", "saxpy_65536 native+native"] {
        assert!(fast[key] != off[key], "{key}: native ignored FAST");
    }

    // `native` still reads the cache: the measured line, its value replaced by
    // FAST, is what an unset `MAPAL_NATIVE_DISPATCH` reads back.
    let Ok(written) = std::fs::read_to_string(&path) else {
        // A one-lane pool measures nothing (no wave), so there is no line.
        assert!(off == measured, "nothing measured, yet native moved");
        return;
    };
    let forced: String = written
        .lines()
        .map(|l| format!("{}\t{FAST}\n", l.split('\t').next().unwrap()))
        .collect();
    assert!(!forced.is_empty(), "measure wrote an empty cache");
    std::fs::write(&path, forced).unwrap();
    let cached = child(dir.path(), "cached", None, &cache);
    assert!(cached == fast, "native did not read the cache's economics");
}
