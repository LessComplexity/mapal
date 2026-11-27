use super::{ARENA_ALIGN, color, color_by};

#[test]
fn grouped_stream_lookup_uses_aliases_and_keeps_unknowns_legacy() {
    use super::{Carry, Epi, Member, ResidentPlan};
    use mapal_ir::{FuncKind, IrBuilder, SourceLoc, Ty, Value};
    use slotmap::SecondaryMap;

    let loc = SourceLoc::empty_at(0);
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "lookup", Ty::Unit, Ty::i32(), loc)
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let host = fb.input();
    let mut object = || fb.constant(Value::I32(1), loc).unwrap();
    let (merge, view, exit, next, raw) = (object(), object(), object(), object(), object());
    fb.output(raw, None, loc).unwrap();
    fb.finish().unwrap();
    let ir = b.seal(f).unwrap();
    let output = ir.func(f).unwrap().output;
    let site = ir.in_edges(output)[0];
    let mut groups = SecondaryMap::new();
    groups.insert(
        site,
        Epi {
            post: None,
            terms: Vec::new(),
            split: 1,
            store_out: false,
            chain: vec![Member {
                m: site,
                body: f,
                out: merge,
                store: true,
                srcs: vec![],
            }],
        },
    );
    let mut plan = ResidentPlan::default();
    plan.chain.insert(site, 2);
    plan.carries.push(Carry {
        merge,
        k: 0,
        next,
        views: vec![view],
        exit: Some(exit),
        inplace: false,
        chain: Vec::new(),
    });
    for o in [merge, view, exit, output] {
        assert_eq!(plan.stream_of(&ir, &groups, o), 2);
    }
    // The next iteration owns a separate slot; host/unknown writers stay legacy.
    for o in [next, host, raw] {
        assert_eq!(plan.stream_of(&ir, &groups, o), 0);
    }
    plan.chain.clear();
    assert_eq!(plan.stream_of(&ir, &groups, view), 0);
}

/// Disjoint lives share bytes; overlapping ones (an inclusive end: the
/// site that reads one object writes the next) never do.
#[test]
fn color_shares_bytes_exactly_across_disjoint_lives() {
    let mb = 64 << 20;
    // The e2e chain: a, b, e built first; c = a·b at 3; d = c·e at 4; d[0] at 5.
    let (off, total) = color(&[(0, 3, mb), (1, 3, mb), (2, 4, mb), (3, 4, mb), (4, 5, mb)]);
    assert_eq!(
        off,
        [0, mb, 2 * mb, 3 * mb, 0],
        "d takes a's bytes, not c's or e's"
    );
    assert_eq!(total, 4 * mb);
    // Born where the other dies: both live at that morphism.
    let (off, _) = color(&[(0, 2, 100), (2, 3, 100)]);
    assert_ne!(off[0], off[1]);
    // Strictly after: shared.
    let (off, total) = color(&[(0, 2, 100), (3, 4, 100)]);
    assert_eq!((off[0], off[1], total), (0, 0, ARENA_ALIGN));
}

/// caching-memory `arena-fit`: largest first, a large late object takes
/// the low bytes and the small early ones pack around it — never more
/// than the birth order here, and the live maximum on this shape.
#[test]
fn color_by_size_packs_large_first() {
    let a = ARENA_ALIGN;
    // Two small early lives, then a large one born after the first dies.
    let live = [(0, 1, a), (0, 5, a), (2, 5, 4 * a)];
    let (_, birth) = color(&live);
    let (off, fit) = color_by(&live, true);
    assert_eq!(birth, 6 * a, "birth order stacks the large one above both");
    assert_eq!(fit, 5 * a, "{off:?}");
    assert_eq!(off[2], 0);
    // Same lives, no conflict anywhere.
    for i in 0..live.len() {
        for j in 0..i {
            let t = live[i].0 <= live[j].1 && live[j].0 <= live[i].1;
            let b = off[i] < off[j] + live[j].2.div_ceil(a) * a
                && off[j] < off[i] + live[i].2.div_ceil(a) * a;
            assert!(!(t && b), "{i} {j} {off:?}");
        }
    }
}

/// Every offset is `cuMemAlloc`'s alignment, and a hole is reused first-fit.
#[test]
fn color_aligns_and_fills_holes() {
    // x dies early; y lives long; z (born after x) takes x's hole below y.
    let (off, total) = color(&[(0, 1, 10), (0, 9, 300), (2, 9, 5)]);
    assert!(off.iter().all(|o| o % ARENA_ALIGN == 0), "{off:?}");
    assert_eq!(off, [0, ARENA_ALIGN, 0]);
    assert_eq!(total, 3 * ARENA_ALIGN);
    assert_eq!(color(&[]), (vec![], 0));
}

use super::waves_merge;

#[test]
fn the_wave_rule_merges_exactly_the_last_waves_that_fit_one() {
    // Two quarter waves share one wave; two full waves never do.
    assert!(waves_merge((64, 256), (64, 256)));
    assert!(!waves_merge((256, 256), (256, 256)));
    // Under one wave each but not together: two waves either way.
    assert!(!waves_merge((128, 164), (128, 164)));
    // Many waves with short tails (the 3090 at 4096, 82 SMs × 2): 40 + 40
    // of 164 — one wave saved of 14.
    assert!(waves_merge((1024, 164), (1024, 164)));
    // ~1.2 waves each (the A100 at 2048): 4 waves back to back, 3 together.
    assert!(waves_merge((256, 216), (256, 216)));
    // A tail of exactly half each fits; one block more does not.
    assert!(waves_merge((50, 100), (50, 100)));
    assert!(!waves_merge((51, 100), (50, 100)));
    // Different geometries weigh each tail by its own wave.
    assert!(waves_merge((128, 256), (32, 64)));
    assert!(!waves_merge((128, 256), (33, 64)));
}
