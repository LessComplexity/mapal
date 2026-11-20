//! Writes generated programs to a directory, for emission comparisons across
//! revisions (`MAPAL_TESTGEN_DUMP=<dir> cargo test -p mapal-rewrite --test
//! testgen_dump -- --ignored`; `MAPAL_TESTGEN_N` programs per strategy,
//! default 40). The same seeds every run, so two revisions see the same files.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config, TestRng, TestRunner};

mod testgen;

fn take<S: Strategy>(s: S, n: usize, seed: u8) -> Vec<S::Value> {
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(proptest::test_runner::RngAlgorithm::ChaCha, &[seed; 32]),
    );
    (0..n)
        .map(|_| s.new_tree(&mut runner).expect("a value").current())
        .collect()
}

#[test]
#[ignore = "a tool: writes programs for emission comparisons"]
fn dump_generated_programs() {
    let Ok(dir) = std::env::var("MAPAL_TESTGEN_DUMP") else {
        return;
    };
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    std::fs::create_dir_all(&dir).expect("dump dir");
    let put = |kind: &str, i: usize, src: String| {
        std::fs::write(format!("{dir}/{kind}_{i:03}.mapal"), src).expect("write");
    };
    for (i, p) in take(testgen::device_prog_strategy(), n, 1)
        .iter()
        .enumerate()
    {
        put("device", i, testgen::device_src(p));
    }
    for (i, p) in take(testgen::chain_strategy(), n, 2).iter().enumerate() {
        put("chain", i, p.source());
    }
    for (i, p) in take(testgen::wide_chain_strategy(), n, 3)
        .iter()
        .enumerate()
    {
        put("wide", i, p.source());
    }
    for (i, p) in take(testgen::twin_prog_strategy(), n, 4).iter().enumerate() {
        put("twin", i, testgen::twin_src(p, true, None));
    }
    for (i, p) in take(testgen::twin_number_strategy(), n, 5)
        .iter()
        .enumerate()
    {
        put("twinnum", i, testgen::twin_src(p, true, None));
    }
    for (i, p) in take(testgen::recur_prog_strategy(), n, 6)
        .iter()
        .enumerate()
    {
        put("recur", i, testgen::recur_src(p));
    }
    for (i, p) in take(testgen::radix_prog_strategy(), n, 7)
        .iter()
        .enumerate()
    {
        put("radix", i, testgen::radix_src(p).0);
    }
    for (i, p) in take(testgen::geom_prog_strategy(), n, 8).iter().enumerate() {
        put("geom", i, testgen::geom_src(p).0);
    }
    for (i, p) in take(testgen::tile_prog_strategy(), n, 9).iter().enumerate() {
        put("tile", i, testgen::tile_src(p, false).0);
    }
    for (i, p) in take(testgen::zero_prog_strategy(), n, 10)
        .iter()
        .enumerate()
    {
        put("zero", i, testgen::zero_src(p));
    }
}
