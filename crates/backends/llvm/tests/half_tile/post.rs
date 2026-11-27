//! Shared stores and the rounded epilogue on generated chains and under the walk.

use super::*;

/// precision `shared-store`: what one generated program covers.
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct SharedCover {
    pub(crate) programs: u64,
    /// Copies whose `a` is stored rounded (`PrecisionPlan::store`).
    pub(crate) stored: u64,
    /// Copies whose `a` an f32 reader keeps f32, beside a stored copy of the
    /// same bodies — the case the shared bodies broke.
    pub(crate) kept_beside_stored: u64,
    /// Programs whose every copy is stored (one rewrite per copy).
    pub(crate) all_stored: u64,
    /// `twin`: one store's walk entering two folds of one body.
    pub(crate) twin: u64,
    /// A body named by more than one map or fold (the premise).
    pub(crate) shared_bodies: u64,
}

impl std::ops::AddAssign for SharedCover {
    fn add_assign(&mut self, o: SharedCover) {
        self.programs += o.programs;
        self.stored += o.stored;
        self.kept_beside_stored += o.kept_beside_stored;
        self.all_stored += o.all_stored;
        self.twin += o.twin;
        self.shared_bodies += o.shared_bodies;
    }
}

/// One `testgen::SharedStoreProg`: the facts (each unread copy's `a` a
/// store, each read copy's not), the rewritten graph valid and idempotent,
/// and the emulated device's values against the weakest member's
/// interpreter within D3 in every emulator mode (a read copy's `a` exact).
pub(crate) fn check_shared_store(clang: &str, p: &testgen::SharedStoreProg) -> SharedCover {
    let (src, (pairs, k)) = testgen::shared_store_src(p);
    let ir = lower(&src);
    let mut users: std::collections::HashMap<mapal_ir::FuncId, u64> = Default::default();
    for (_, fd) in ir.funcs() {
        for &m in &fd.morphisms {
            if let Some(
                mapal_ir::Operation::Map { body, .. } | mapal_ir::Operation::Fold { body, .. },
            ) = ir.morphism(m).map(|x| x.op)
            {
                *users.entry(body).or_default() += 1;
            }
        }
    }
    let shared = users.values().any(|&n| n > 1);
    let stored: usize = ir
        .funcs()
        .map(|(f, _)| ir.precision_plan(f).store.len())
        .sum();
    let read = p.read.count_ones() as usize;
    let copies = usize::from(p.copies);
    assert_eq!(stored, copies - read, "one store per unread copy\n{src}");
    if let Some(g) = ir.narrow_stored() {
        assert!(mapal_ir::validate(&g).is_empty(), "{src}");
        assert!(g.narrow_stored().is_none(), "idempotent\n{src}");
    }
    let weak = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
    let c = bound_coef_unit(k, Some(2f64.powi(-8))) + bound_coef_unit(k, None);
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = on.device.as_deref().unwrap_or("").to_string();
    let built = emu::build_emulated(clang, &on.text, &dev);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        let g = floats(&got);
        assert_eq!(g.len(), weak.len(), "{mode:?}\n{src}");
        for i in 0..pairs {
            let (y, w, m) = (g[2 * i], weak[2 * i], weak[2 * i + 1]);
            let tol = 2.0 * c * m + 2f64.powi(-22) * w.abs();
            assert!(
                (y - w).abs() <= tol,
                "{mode:?} pair {i}: {y} vs {w} (M {m}, tol {tol})\n{src}"
            );
        }
    }
    SharedCover {
        programs: 1,
        stored: stored as u64,
        kept_beside_stored: u64::from(stored > 0) * read as u64,
        all_stored: u64::from(read == 0),
        twin: u64::from(p.twin && stored > 0),
        shared_bodies: u64::from(shared),
    }
}

/// precision `shared-store` on 24 generated programs: a GEMM function
/// inlined at two or three sites (the copies share their map and fold
/// bodies), its `a` read by `: bf16 | f32` products, some copies' `a` also
/// printed (kept f32). A store rewrites only its own copy's bodies, so the
/// kept copies' producers and readers stay f32 (before, they rounded and
/// widened through the shared bodies: the full model's `fold: bf16 | f32`
/// build, token 84707 repeated). Values on the emulated device in every mode
/// within the bound; every coverage counter non-zero.
#[test]
fn round_store_shared_bodies_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::SharedStoreProg> = (0..24)
        .map(|i| {
            let mut p = testgen::shared_store_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current();
            // Lean to the case the bodies broke: one copy read, another not;
            // one program in six reads none (every copy stored).
            if i % 6 == 5 {
                p.read = 0;
            } else if i % 3 != 2 && (p.read == 0 || p.read.count_ones() == u32::from(p.copies)) {
                p.read = 1 << (i % usize::from(p.copies));
            }
            p
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check_shared_store(&clang, p));
    let mut c = SharedCover::default();
    for k in covers {
        c += k;
    }
    eprintln!("shared-store generated: {c:?}");
    let all = [
        c.programs,
        c.stored,
        c.kept_beside_stored,
        c.all_stored,
        c.twin,
        c.shared_bodies,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}

/// bf16-class `round-post`: a GEMM whose map goes on past its fold (an
/// epilogue: `÷ s`, `+ r[i]`, `· g[j]`, or none) feeding a second GEMM under
/// `fold: bf16 | f32` — the first product's output is read only as the
/// second's multiplicand, so it is stored rounded once by its producer
/// (`precision_plan().store`): the site's epilogue ends in the rounding
/// (`TilePost::round`), the map writes bf16 and the second site reads a
/// narrow operand — its rows read down `y`'s columns (`zᵀ = yᵀ·w2`: no
/// row-block stream joins the two, whose axis-wide values are never
/// stored). `(t, k1, c1, c2, epi)`.
pub(crate) fn round_post_src(t: u64, k1: u64, c1: u64, c2: u64, epi: u8) -> String {
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64, d: f64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        x -> (x * {a} + {b}) % {m} - {} -> widen<f32> -> v;\n        v / {d:?}\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("a", t * k1, 7, 3, 13, 8.0);
    s += &data("w1", k1 * c1, 5, 1, 11, 16.0);
    s += &data("w2", t * c2, 3, 2, 9, 16.0);
    s += &data("r", t, 5, 4, 7, 4.0);
    s += &data("g", c1, 3, 1, 5, 2.0);
    s += &format!("    {k1} -> iota -> rk1;\n    {t} -> iota -> rk2;\n");
    let tail = match epi {
        0 => "dot / 3.0".to_string(),
        1 => format!("dot + r[cell / {c1}]"),
        2 => format!("dot * g[cell % {c1}]"),
        _ => "dot".to_string(),
    };
    s += &format!(
        "    {} -> iota -> map {{\n        cell -> (0.0, rk1) -> fold: bf16 | f32 {{\n            acc, k -> acc + a[cell / {c1} * {k1} + k] * w1[k * {c1} + cell % {c1}]\n        }} -> dot;\n        {tail}\n    }} -> y;\n",
        t * c1
    );
    s += &format!(
        "    {} -> iota -> map {{\n        cell -> (0.0, rk2) -> fold: bf16 | f32 {{\n            acc, k -> acc + y[k * {c1} + cell / {c2}] * w2[k * {c2} + cell % {c2}]\n        }}\n    }} -> z;\n",
        c1 * c2
    );
    for i in [0, c1 * c2 / 3, c1 * c2 / 2 + 1, c1 * c2 - 1] {
        s += &format!("    z[{i}] -> println;\n");
    }
    s + "}\n"
}

/// bf16-class `round-post` on generated GEMM chains (`round_post_src`): the
/// first site's array stored bf16 with its epilogue's rounding
/// (`TilePost::round`, every epilogue form), the second site's `a` narrow;
/// the device's values against the weakest member's interpreter in every
/// emulator mode, and under `round-store` off; coverage non-zero.
#[test]
fn round_post_on_generated_chains() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut progs = Vec::new();
    for (i, t) in [8u64, 16, 32, 48].into_iter().enumerate() {
        for (j, k1) in [16u64, 32, 64].into_iter().enumerate() {
            for epi in 0..4u8 {
                let c1 = [16u64, 32, 64][(i + j + epi as usize) % 3];
                let c2 = [16u64, 32][(i + epi as usize) % 2];
                progs.push((t, k1, c1, c2, epi));
            }
        }
    }
    let covers = testgen::par_check(&progs, |_, &(t, k1, c1, c2, epi)| {
        let src = round_post_src(t, k1, c1, c2, epi);
        let ir = lower(&src);
        let want = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
        let mut cover = [0u64; 6];
        cover[0] = 1;
        let g = ir.narrow_stored().expect("y stored bf16");
        let rounded = g
            .funcs()
            .flat_map(|(f, _)| {
                let post = g.tile_plan_post(f).sites;
                let radix = g.tile_plan_radix(f);
                let narrow = g.tile_plan_narrow(f).sites;
                post.into_iter()
                    .chain(radix)
                    .chain(narrow)
                    .map(|(_, s)| s)
                    .collect::<Vec<_>>()
            })
            .filter(|s| {
                s.post
                    .as_ref()
                    .is_some_and(|p| p.round == Some(NarrowFloat::Bf16))
            })
            .count();
        assert!(rounded >= 1, "the first site rounds as it stores\n{src}");
        cover[1] = 1;
        cover[2 + usize::from(epi.min(3))] = 1;
        let check = |got: &str, what: &str| {
            let got = floats(got);
            assert_eq!(got.len(), want.len(), "{what}\n{src}");
            for (x, w) in got.iter().zip(&want) {
                let tol = 2f64.powi(-6) * w.abs().max(1.0);
                assert!((x - w).abs() <= tol, "{what}: {x} vs {w}\n{src}");
            }
        };
        for off in [&[][..], &["round-store"][..]] {
            let mut arms = vec!["half-fill"];
            arms.extend_from_slice(off);
            let em = emission(&ir, "rtx-5080", EmitOpts::default(), &arms);
            let built = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
            let modes: &[[(&str, &str); 2]] = if off.is_empty() { &MODES } else { &MODES[..1] };
            for mode in modes {
                let (got, _) = built.run(mode);
                check(&got, &format!("{off:?} {mode:?}"));
            }
        }
        cover
    });
    let mut c = [0u64; 6];
    for k in covers {
        for (a, b) in c.iter_mut().zip(k) {
            *a += b;
        }
    }
    eprintln!("round-post generated: programs, rounded, epi ÷ + · none: {c:?}");
    assert!(c.iter().all(|&x| x > 0), "{c:?}");
}

/// [`round_post_src`]'s chain with the first site's cell split into its
/// row and column first (`cell / c1 -> i; cell % c1 -> j`, the generator's
/// GEMM form): an affine epilogue site, not a mixed-radix one — the form
/// `mma-walk` takes.
pub(crate) fn walk_post_src(t: u64, k1: u64, c1: u64, c2: u64, epi: u8) -> String {
    let src = round_post_src(t, k1, c1, c2, epi);
    let old = format!(
        "        cell -> (0.0, rk1) -> fold: bf16 | f32 {{\n            acc, k -> acc + a[cell / {c1} * {k1} + k] * w1[k * {c1} + cell % {c1}]\n        }} -> dot;\n"
    );
    let new = format!(
        "        cell ->\n        cell / {c1} -> i;\n        cell % {c1} -> j;\n        (0.0, rk1) -> fold: bf16 | f32 {{\n            acc, k -> acc + a[i * {k1} + k] * w1[k * {c1} + j]\n        }} -> dot;\n"
    );
    assert!(src.contains(&old), "{src}");
    src.replace(&old, &new)
}

/// `round-post` under the walk (bf16 tensor-tile `mma-walk` × bf16-class
/// `round-post`, the integration's check): [`walk_post_src`]'s chains at
/// walkable extents (`t` 512, `c1` 6144, `k1` 64: two panels a tile, more
/// units than the 5080's resident slots), the walk forced
/// (`--gpu-on=mma-walk`, the measurement arm). An epilogue in the site's
/// own body (`TilePost`, here ending in the stored rounding) keeps the
/// ordered tile — the walk takes no post site, so its `red` partials never
/// meet a bf16 store — and the same chain with the epilogue as its own map
/// (`y0` the site's f32 output, `y` the map over it, stored rounded) walks
/// the site. Both against the weakest member's interpreter in the emulator
/// (coverage non-zero).
#[test]
fn round_post_under_the_walk() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let progs: Vec<(u8, bool)> = (0..4)
        .map(|e| (e, false))
        .chain((0..3).map(|e| (e, true)))
        .collect();
    let covers = testgen::par_check(&progs, |_, &(epi, split)| {
        let (t, k1, c1, c2) = (512, 64, 6144, 16);
        let mut src = walk_post_src(t, k1, c1, c2, epi);
        if split {
            let at = src.find("        } -> dot;\n").expect("the fold");
            let end = src[at..].find("    } -> y;\n").expect("the map") + at;
            let tail = src[at + "        } -> dot;\n".len()..end].to_string();
            src.replace_range(
                at..end + "    } -> y;\n".len(),
                &format!(
                    "        }}\n    }} -> y0;\n    {} -> iota -> map {{\n        cell -> y0[cell] -> dot;\n{tail}    }} -> y;\n",
                    t * c1
                ),
            );
        }
        let ir = lower(&src);
        let want = floats(&run_weakest(&ir, 20_000_000_000, &[]).output);
        let rounded = ir.narrow_stored().map_or(0, |g| {
            g.funcs()
                .flat_map(|(f, _)| {
                    let post = g.tile_plan_post(f).sites;
                    let radix = g.tile_plan_radix(f);
                    let narrow = g.tile_plan_narrow(f).sites;
                    post.into_iter()
                        .chain(radix)
                        .chain(narrow)
                        .map(|(_, s)| s)
                        .collect::<Vec<_>>()
                })
                .filter(|s| {
                    s.post
                        .as_ref()
                        .is_some_and(|p| p.round == Some(NarrowFloat::Bf16))
                })
                .count()
        });
        let mut gpu = GpuArms {
            site_cands: 1,
            ..GpuArms::default()
        };
        for r in ["pdl", "streams", "pinned", "horizontal", "calls"] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        *gpu.on.rung("mma-walk").expect("rung") = true;
        let opts = EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        };
        let em = emit_detailed(&ir, &opts).expect("emits");
        let dev = em.device.as_deref().unwrap_or("");
        let first = format!(" rows={t} c={c1} k={k1} ");
        let plan = dev
            .lines()
            .find(|l| {
                l.starts_with("; mapal.plan ") && l.contains(" kind=mma ") && l.contains(&first)
            })
            .unwrap_or_else(|| panic!("epi {epi}: the first site's tensor tile\n{src}"));
        let walked = plan.contains(" combine=streamk ");
        let mut cover = [0usize; 3];
        match split {
            false => {
                assert!(plan.contains(" post=1"), "epi {epi}: {plan}\n{src}");
                assert!(!walked, "epi {epi}: a post site walked\n{src}");
                assert!(
                    rounded >= 1,
                    "epi {epi}: the site rounds as it stores\n{src}"
                );
                cover[0] += 1;
            }
            true => {
                assert!(walked, "epi {epi}: the forced walk of the f32 site\n{src}");
                cover[1] += 1;
            }
        }
        let built = emu::build_emulated(&clang, &em.text, dev);
        let modes: &[[(&str, &str); 2]] = if split { &MODES[..2] } else { &MODES };
        for mode in modes {
            let (got, _) = built.run(mode);
            let got = floats(&got);
            assert_eq!(got.len(), want.len(), "epi {epi} {mode:?}\n{src}");
            for (x, w) in got.iter().zip(&want) {
                let tol = 2f64.powi(-6) * w.abs().max(1.0);
                assert!(
                    (x - w).abs() <= tol,
                    "epi {epi} {mode:?}: {x} vs {w}\n{src}"
                );
            }
            cover[2] += 1;
        }
        cover
    });
    let sum = |i: usize| covers.iter().map(|c| c[i]).sum::<usize>();
    eprintln!(
        "round-post under the walk: {} rounding post sites kept ordered, {} split epilogues walked, {} emulated runs",
        sum(0),
        sum(1),
        sum(2)
    );
    assert!(sum(0) >= 4 && sum(1) >= 3 && sum(2) >= 22, "{covers:?}");
}
