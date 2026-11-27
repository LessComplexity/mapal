//! Unit tests of `calls`: persistent cooperative launches, grid barriers and the host issue cost.

use super::super::horizontal::{HzSpec, parse_head};
use super::*;
use crate::profile::COOP_BODY_CYCLES_RTX_5080;

fn gpu() -> &'static Gpu {
    crate::profile::TargetProfile::named("rtx-5080")
        .and_then(|p| p.gpu())
        .expect("rtx-5080 has a GPU")
}

/// [`chain_ns`] without the host's issue term (`issue_ns` 0).
#[allow(clippy::too_many_arguments)]
fn chain_ns0(
    units: &[u64],
    grid: u64,
    barriers: u64,
    gpu: &Gpu,
    pdl: bool,
    launches: u64,
    rounds: u64,
    work_ns: f64,
) -> Option<(f64, f64)> {
    chain_ns(
        units,
        grid,
        barriers,
        gpu,
        pdl,
        launches,
        rounds,
        work_ns,
        &[],
        0.0,
    )
}

/// review-fixes-2: the process's first cooperative launch is shared by
/// the chain's launches in a call — whole outside every loop, a counted
/// loop's trips each their share, a loop no fact counts no term.
#[test]
fn first_cooperative_launch_shared_by_counted_trips() {
    let g = gpu();
    let first = g.chain_cost().0.coop_first_ns as f64;
    assert!(first > 0.0);
    let per = |n: u64| {
        chain_ns0(&[3, 128, 128], 128, 2, g, false, n, 1, 0.0)
            .unwrap()
            .0
    };
    assert_eq!(per(1) - per(0), first);
    assert_eq!(per(2) - per(0), (first as u64 / 2) as f64);
    assert!(per(8) < per(2) && per(2) < per(1));
}

/// calls-issue: at `issue_ns` 0 the price is the price before the host
/// term, for any unit times; the separate form's price is non-decreasing
/// in `issue_ns` (a slower host forms every chain a faster one forms);
/// a member longer than the issue hides it; the RTX 5080's decode chain
/// (three units of a few µs each) stays refused at its host's measured
/// 1690 ns, and a host slow enough to idle between them forms it.
#[test]
fn host_issue_prices_the_separate_form() {
    let g = gpu();
    let units = [3u64, 128, 128];
    let (per0, sep0) = chain_ns0(&units, 128, 2, g, false, 0, 1, 0.0).unwrap();
    for d in [[0.0; 3], [500.0, 4000.0, 20.0], [1e6; 3]] {
        let (per, sep) = chain_ns(&units, 128, 2, g, false, 0, 1, 0.0, &d, 0.0).unwrap();
        assert_eq!((per, sep), (per0, sep0));
    }
    let d = [300.0, 900.0, 200.0];
    let mut last = sep0;
    for issue in [0.0, 500.0, 1690.0, 4000.0, 8000.0, 25_000.0] {
        let (per, sep) = chain_ns(&units, 128, 2, g, false, 0, 1, 0.0, &d, issue).unwrap();
        assert_eq!(per, per0, "the chain is one issue");
        assert!(sep >= last, "{issue}: {sep} < {last}");
        last = sep;
    }
    // Members longer than the issue: no host term.
    let long = [5000.0; 3];
    let (_, sep) = chain_ns(&units, 128, 2, g, false, 0, 1, 0.0, &long, 1690.0).unwrap();
    assert_eq!(sep, sep0);
    // The decode chain's members (a few µs of work each): refused at
    // 1690 ns as before; formed on a host idling ~8 µs between them.
    let dec = [2500.0, 3000.0, 2000.0];
    let (per, sep) = chain_ns(&units, 128, 2, g, false, 0, 1, 0.0, &dec, 1690.0).unwrap();
    assert!(per > sep, "{per} {sep}");
    let (per, sep) = chain_ns(&units, 128, 2, g, false, 0, 1, 0.0, &dec, 12_000.0).unwrap();
    assert!(per <= sep, "{per} {sep}");
}

fn shape(blocks: u64, block: u64, smem: u64, per_sm: u64, args: usize) -> Shape {
    Shape {
        blocks,
        block,
        smem,
        per_sm,
        args,
    }
}

#[test]
fn the_persistent_grid_fits_or_says_why() {
    let g = gpu();
    // decode's trip: rows and query (1 block each), then two 128-block sites.
    let lv = vec![
        vec![
            shape(1, 256, 0, 6, 3),
            shape(1, 256, 0, 6, 3),
            shape(1, 256, 0, 6, 2),
        ],
        vec![shape(128, 256, 0, 6, 3)],
        vec![shape(128, 256, 0, 6, 3)],
    ];
    assert_eq!(persistent_fit(&lv, g), Ok(128));
    // The grid is capped at one wave of the merged footprint.
    let big = vec![
        vec![shape(4096, 256, 0, 6, 3)],
        vec![shape(4096, 256, 0, 6, 3)],
    ];
    assert_eq!(persistent_fit(&big, g), Ok(g.sms * 6));
    // More pointers than one cooperative launch passes.
    let many = vec![vec![shape(1, 256, 0, 6, 17)], vec![shape(1, 256, 0, 6, 16)]];
    assert_eq!(persistent_fit(&many, g), Err(Refuse::Args));
    // A large member's residency falls at the merged shared memory.
    let occ = vec![
        vec![shape(4 * 504, 256, 0, 6, 3)],
        vec![shape(1, 256, 90 * 1024, 1, 3)],
    ];
    assert_eq!(persistent_fit(&occ, g), Err(Refuse::Occupancy));
    // Two full waves side by side in one level: no round saved.
    let w = g.sms * 6;
    let full = vec![
        vec![shape(w, 256, 0, 6, 3), shape(w, 256, 0, 6, 3)],
        vec![shape(1, 256, 0, 6, 3)],
    ];
    assert_eq!(persistent_fit(&full, g), Err(Refuse::Waves));
}

#[test]
fn the_rule_prices_barriers_against_launches() {
    let g = gpu();
    let d = g.cost.as_ref().expect("measured");
    assert_eq!(g.chain_cost(), (*d, true), "the rtx-5080 measured them");
    // Decode's chain in its loop, one round of its 128-block grid: a
    // cooperative launch (2044 ns) and two barriers (553 + 44 · 1.89 ns
    // at 128 blocks), no body term (one round) and its phases' 22 ‰,
    // against three plain dependent launches (650 ns): the boundaries
    // alone keep it separate here.
    let body = g.coop_body_ns();
    assert!((body - 4390.0 / 2.45).abs() < 0.5, "{body}");
    let at128 = 553.0 + 44.0 * 1.89;
    let (per, sep) = chain_ns0(&[3, 128, 128], 128, 2, g, false, 0, 1, 0.0).expect("priced");
    assert!(
        (per - (2044.0 + 2.0 * at128)).abs() < 0.5 && (sep - 1950.0).abs() < 0.5,
        "{per} {sep}"
    );
    assert!(per > sep);
    // The phases: 22 ‰ of their work at the part's peaks.
    let (long, _) = chain_ns0(&[3, 128, 128], 128, 2, g, false, 0, 1, 100_000.0).expect("priced");
    assert!((long - per - 2200.0).abs() < 0.5, "{long} {per}");
    // The grid-stride form (more than one round) pays the bodies'
    // measured change (`calib.cu body`, 4 rounds: 1792 ns).
    let (strided, _) = chain_ns0(&[3, 512, 512], 128, 2, g, false, 0, 4, 0.0).expect("priced");
    assert!((strided - per - body).abs() < 0.5, "{strided} {per}");
    // Outside every loop it also pays the process's first cooperative
    // launch (38 us).
    let (once, _) = chain_ns0(&[3, 128, 128], 128, 2, g, false, 1, 1, 0.0).expect("priced");
    assert!((once - per - 38_000.0).abs() < 0.5, "{once} {per}");
    // Programmatic dependent launches (486 ns each): further apart.
    let (_, sep) = chain_ns0(&[3, 128, 128], 128, 2, g, true, 0, 1, 0.0).expect("priced");
    assert!((sep - 1458.0).abs() < 0.5, "{sep}");
    // A long chain of one-block launches: twelve barriers of one block
    // (553 ns priced) against twelve more plain launches — still not
    // enough to pay the cooperative launch back here.
    let (per, sep) = chain_ns0(&[1; 13], 1, 12, g, false, 0, 1, 0.0).expect("priced");
    assert!(per > sep, "{per} {sep}");
    // Without the cooperative launch's price, the barriers alone win a
    // one-round chain; the grid-stride form's bodies keep it separate.
    let mut cheap = *d;
    cheap.coop_launch_ns = 0;
    let g2 = Gpu {
        cost: Some(cheap),
        ..*g
    };
    let (per, sep) = chain_ns0(&[3, 128, 128], 128, 2, &g2, false, 0, 1, 0.0).expect("priced");
    assert!(per <= sep, "{per} {sep}");
    let (per, sep) = chain_ns0(&[3, 512, 512], 128, 2, &g2, false, 0, 4, 0.0).expect("priced");
    assert!(per > sep, "{per} {sep}");
    assert!((d.gsync_at(504, g.sms).unwrap() - (553.0 + 420.0 * 1.89)).abs() < 0.5);
    // open-rules: a part without the measured barrier and launch costs
    // (the 5090) prices chains on the rtx-5080's cycles at its own clock
    // (2.41 GHz: 562 / 660 / 494 / 2078 ns) — derived, as its bodies'.
    let p5090 = crate::profile::TargetProfile::named("rtx-5090")
        .and_then(|p| p.gpu())
        .unwrap();
    let (dv, measured) = p5090.chain_cost();
    assert!(!measured);
    assert_eq!(
        (
            dv.gsync_ns,
            dv.dep_launch_ns,
            dv.pdl_launch_ns,
            dv.coop_launch_ns
        ),
        (562, 660, 494, 2078)
    );
    let (per, sep) = chain_ns0(&[3, 128, 128], 128, 2, p5090, true, 0, 1, 0.0).unwrap();
    assert!(
        per > sep,
        "the 5090 prices decode as the 5080 does: {per} {sep}"
    );
    assert_eq!(p5090.coop_body_cycles(), (COOP_BODY_CYCLES_RTX_5080, false));
}

/// **open-rules `calls`, replayed** (cross-arch 60f0f8ce and 7a2b67bf,
/// the RTX 5080): the bodies' change is a per-round fact and the
/// phases' length a fact of the chain. One-round chains (every level
/// within its persistent grid) price no body term: decode's chain forms
/// on the 4090 / 3090 (656 / 522 ns saved; it won 1.20 / 1.48 formed in
/// 7a2b67bf and lost refused in 60f0f8ce) as does causal's (won 1.16 /
/// 1.36). The grid-stride form pays the 4-round body: the a100's causal
/// chain at 4 rounds refuses (2021 ns saved against 2871: its measured
/// 4048 cycles, cross-arch 5fcb401b `calib.cu body`; 3113 derived). Long
/// phases pay their 22 ‰: the H100's smlp chain (two GEMM phases of at
/// least 70 us at its FFMA peak, 1.47x slower formed) refuses, 1540 ns
/// against 695 saved. Open: the a100's one-round causal family forms
/// (its forced chain measured 1.3-1.7 % slower: +10 % phases inside).
#[test]
fn the_rounds_and_phases_decide_where_the_boundaries_alone_did_not() {
    let part = |n: &str| {
        *crate::profile::TargetProfile::named(n)
            .and_then(|p| p.gpu())
            .unwrap()
    };
    for (n, dec_save) in [("rtx-4090", 656.0), ("rtx-3090", 522.0)] {
        let g = part(n);
        let (per, sep) = chain_ns0(&[3, 128, 128], 128, 2, &g, false, 0, 1, 0.0).unwrap();
        assert!(
            (sep - per - dec_save).abs() < 1.0,
            "{n} decode: {per} {sep}"
        );
        let (per, sep) = chain_ns0(&[3, 128, 1, 4, 1, 32], 128, 5, &g, false, 0, 1, 0.0).unwrap();
        assert!(per < sep, "{n} causal: {per} {sep}");
    }
    let a100 = part("a100");
    let causal = [3, 128, 1, 4, 1, 32];
    let (per, sep) = chain_ns0(&causal, 128, 5, &a100, false, 0, 1, 0.0).unwrap();
    assert!((sep - per - 2021.0).abs() < 1.0, "{per} {sep}");
    let (per, sep) = chain_ns0(&causal, 128, 5, &a100, false, 0, 4, 0.0).unwrap();
    assert!(
        per > sep && (a100.coop_body_ns() - 2871.0).abs() < 1.0,
        "{per} {sep}"
    );
    let h100 = part("h100");
    let flops = 2.0 * (4096.0 * 256.0 * 1024.0) * 2.0;
    let work = flops / h100.ffma_flops_per_ns() as f64;
    assert!(work > 70_000.0, "{work}");
    let (per, sep) = chain_ns0(&[256, 256], 256, 1, &h100, false, 0, 1, work).unwrap();
    assert!(
        per > sep && sep - (per - work * 0.022) > 600.0,
        "{per} {sep}"
    );
}

#[test]
fn a_persistent_kernel_strides_levels_and_zeroes_before_its_atomic_member() {
    let k = |n: &str, b: u64| {
        format!(
            "define ptx_kernel void @{n}(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias align 4 %o) \"nvvm.maxntid\"=\"{b}\" {{\nentry:\n  call void @llvm.nvvm.barrier0()\n  ret void\n}}\n"
        )
    };
    let (a, b) = ("mapal_tile_000000000000000a", "mapal_tile_000000000000000b");
    let ha = parse_head(k(a, 32).lines().next().unwrap());
    let hb = parse_head(k(b, 256).lines().next().unwrap());
    let spec = HzSpec {
        name: "mapal_tile_00000000000000cc".into(),
        block: 256,
        members: vec![(a.into(), 2, 1, 32), (b.into(), 4, 32, 256)],
        level: vec![0, 1],
        zero: vec![None, Some((4096, 1))],
    };
    let t = persistent_kernel(&spec, &[&ha, &hb]);
    // No `noalias` / `readonly`: level 1 reads what level 0 wrote.
    let head = t.lines().find(|l| l.starts_with("define")).unwrap();
    assert!(
        !head.contains("noalias") && !head.contains("readonly"),
        "{head}"
    );
    assert!(head.contains("ptr addrspace(1) align 16 %h0.a") && head.contains("align 4 %h1.o"));
    // One grid barrier between the two levels, on the kernel's counter.
    assert_eq!(t.matches("call void @mapal_gsync(").count(), 1);
    assert!(t.contains("@mapal_tile_00000000000000cc.gsync = internal addrspace(1) global i32 0"));
    // `b`'s 4096-byte zero runs at level 0 (before the barrier) as 256
    // grid-stride 16-byte stores into its output parameter.
    let (l0, l1) = t.split_once("call void @mapal_gsync(").unwrap();
    assert!(
        l0.contains("icmp ult i32 %cl.zi0_0, 256")
            && l0.contains("getelementptr i8, ptr addrspace(1) %h1.o")
    );
    assert!(l0.contains("store <4 x i32> zeroinitializer") && !l1.contains("zeroinitializer"));
    // Each level's first block straight-line, then a loop striding over
    // the rest (2 and 128 blocks) by the grid.
    assert!(l0.contains("icmp ult i32 %cl.b, 2") && l0.contains("icmp ult i32 %cl.i0, 2"));
    assert!(l1.contains("icmp ult i32 %cl.b, 128") && l1.contains("icmp ult i32 %cl.i1, 128"));
    assert!(
        t.contains("%cl.r0 = add i32 %cl.b, %cl.g") && t.contains("%cl.n0 = add i32 %cl.i0, %cl.g")
    );
    // A 32-thread member runs on its own warps; the tile boundary is
    // barrier 1, apart from the members' barrier 0.
    assert_eq!(l0.matches("icmp ult i32 %cl.t, 32").count(), 2);
    assert!(t.matches("barrier.sync.cnt(i32 1, i32 256)").count() == 4);
    assert!(t.contains("urem i32 %cl.o1_0, 4") && t.contains("udiv i32 %cl.o1_0, 4"));
}

#[test]
fn the_grid_barrier_flips_one_bit_with_release_and_acquire() {
    assert!(GSYNC.contains("atom.add.release.gpu.global.u32"));
    assert!(GSYNC.contains("ld.acquire.gpu.global.u32"));
    // 0x80000000 − (n − 1) = 0x80000001 − n.
    assert!(GSYNC.contains("sub i32 -2147483647, %n"));
    assert_eq!(GSYNC.matches("call void @llvm.nvvm.barrier0()").count(), 2);
    // The counter's top bit flips once all `n` blocks added: simulate.
    for n in [1u32, 2, 3, 84, 504] {
        let mut ctr = 0u32;
        for _ in 0..3 {
            let start = ctr;
            let mut olds = Vec::new();
            for b in 0..n {
                let inc = if b == 0 {
                    0x8000_0001u32.wrapping_sub(n)
                } else {
                    1
                };
                olds.push(ctr);
                ctr = ctr.wrapping_add(inc);
                // No block sees the flip before the last arrival.
                if b + 1 < n {
                    assert!(olds.iter().all(|&o| (o ^ ctr) & 0x8000_0000 == 0) || b == 0 && n == 1);
                }
            }
            assert!(olds.iter().all(|&o| (o ^ ctr) & 0x8000_0000 != 0), "n {n}");
            assert_eq!(ctr & 0x7fff_ffff, start & 0x7fff_ffff);
        }
    }
}
