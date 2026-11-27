//! Shared emission, table parsing and case-generation helpers for the site-choice tests.

use super::*;

pub(crate) fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// `base` on `target` with at most `cands` candidates per table, `on` / `off`
/// rung arms, `place` off (every capable site on the device) and the rungs
/// the emulator's one thread does not run (`pdl`, `streams`, `pinned`) off.
pub(crate) fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    cands: u64,
    on: &[&str],
    off: &[&str],
) -> Emission {
    // The sites' own tables: a row-block unit (`row_block.rs`) launches none;
    // each atomic recipe with its own zero (`zero-fold` is `zero_fold.rs`'s);
    // each site its own table (`horizontal` is `horizontal.rs`'s).
    let mut gpu = GpuArms {
        site_cands: cands,
        row_block: false,
        ..GpuArms::default()
    };
    for r in off.iter().chain(&[
        "place",
        "pdl",
        "streams",
        "pinned",
        "zero-fold",
        "horizontal",
        "calls",
    ]) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
        *gpu.off.rung(r).expect("rung") = false;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// One `mapal_gpu_site` call of a host text: candidate 0's id, the flags and
/// each candidate's recipe (`id rid zero ws gx gy block smem rgx rblock`).
#[derive(Debug)]
pub(crate) struct Table {
    pub(crate) site: u64,
    pub(crate) flags: u32,
    pub(crate) rows: Vec<[i64; 10]>,
}

pub(crate) fn tables(host: &str) -> Vec<Table> {
    host.lines()
        .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_site("))
        .map(|args| {
            let a: Vec<&str> = args.trim_end_matches(')').split(", ").collect();
            let v = |i: usize| a[i].split(' ').nth(1).expect("typed argument");
            let def = format!("{} = ", v(4));
            let global = host
                .lines()
                .find(|d| d.starts_with(&def))
                .expect("the table constant");
            let rows: Vec<[i64; 10]> = global
                .split("%mapal_cand { ")
                .skip(1)
                .map(|r| {
                    let fields: Vec<i64> = r
                        .split(" }")
                        .next()
                        .expect("a row")
                        .split(", ")
                        .map(|f| f.split(' ').nth(1).expect("typed").parse().expect("int"))
                        .collect();
                    fields.try_into().expect("ten fields")
                })
                .collect();
            assert_eq!(rows.len(), v(3).parse::<usize>().unwrap(), "{global}");
            Table {
                site: v(0).parse::<i64>().unwrap() as u64,
                flags: v(1).parse().unwrap(),
                rows,
            }
        })
        .collect()
}

/// The `; mapal.plan` line of kernel `id` in a device unit.
pub(crate) fn plan(dev: &str, id: i64) -> Option<&str> {
    let name = format!("kernel=mapal_tile_{:016x} ", id as u64);
    dev.lines()
        .find(|l| l.starts_with("; mapal.plan") && l.contains(&name))
}

/// Kernel `id`'s text (its plan line through its closing brace), its loop
/// metadata unnumbered (the module numbers it by kernel position).
pub(crate) fn kernel_text(dev: &str, id: i64) -> Option<String> {
    let name = format!("@mapal_tile_{:016x}(", id as u64);
    let at = dev.find(&format!(
        "; mapal.plan kernel=mapal_tile_{:016x} ",
        id as u64
    ))?;
    let body = &dev[at..];
    assert!(body.contains(&name));
    let end = body.find("\n}\n")? + 3;
    let mut out = String::new();
    let mut digits = false;
    for c in body[..end].chars() {
        if digits && c.is_ascii_digit() {
            continue;
        }
        digits = c == '!';
        out.push(c);
    }
    Some(out)
}

pub(crate) fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split(' ')
        .find_map(|f| f.strip_prefix(key))
        .unwrap_or("")
}

/// The printed floats of a run, without the clock (`iter ms=`).
pub(crate) fn floats(out: &str) -> Vec<f64> {
    out.lines()
        .filter(|l| !l.starts_with("iter ms="))
        .map(|l| l.trim().parse().unwrap_or(f64::NAN))
        .collect()
}

/// A run's output without the clock's line (`iter ms=<value>`).
pub(crate) fn untimed(out: &str) -> String {
    out.lines()
        .filter(|l| !l.starts_with("iter ms="))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// A program to check and how: `exact` bit for bit, else each printed value
/// within `tol[i]` (`None`: relative 1e-4, the chains' and the blind set's
/// check in `plan_cost.rs`).
pub(crate) struct Case {
    pub(crate) src: String,
    pub(crate) base: EmitOpts,
    pub(crate) tol: Option<Vec<f64>>,
}

/// ADR-0040 D2's a-priori bound of a geometry program's three printed values
/// (`y[0]`, `y[last]`, the whole sum): `γ_k · M` per cell with `M ≤ k·|a|·|b|`
/// at the operand law's largest magnitudes, and the sum's own `γ_{mn}`.
pub(crate) fn geom_tol((m, n, k): (u64, u64, u64), frac: bool) -> Vec<f64> {
    let ab = if frac {
        (50.0 / 7.0) * (51.0 / 9.0)
    } else {
        50.0 * 51.0
    };
    let mag = k as f64 * ab;
    // `+ 4`: an epilogue's bias (|bias| <= 3) and relu add one rounding.
    let cell = bound_coef(k, false) * mag + (mag + 4.0) * 2f64.powi(-23);
    let cells = (m * n) as f64;
    let total = cells * cell + bound_coef(m * n, false) * cells * (mag + 4.0);
    vec![cell, cell, total]
}

pub(crate) fn check(case: &Case, got: &str, want: &str) -> bool {
    if case.tol.is_none() && case.base.order == Order::Sequential {
        return got == want;
    }
    let (g, w) = (floats(got), floats(want));
    g.len() == w.len()
        && g.iter().zip(&w).enumerate().all(|(i, (a, b))| {
            let tol = match &case.tol {
                Some(t) => t[i.min(t.len() - 1)],
                None => 1e-4 * a.abs().max(b.abs()).max(1.0),
            };
            (a - b).abs() <= tol
        })
}

/// The generator's programs, rotating over the strategies and faces: geom
/// (integer operands, the default face), order (non-integer, `f64`, `a` along
/// `i`, `k = 8·odd`: the exact face bit for bit, and the default face within
/// the bound), mid (tile-efficiency extents), tile (the blind set's shapes at
/// interpreter sizes) and chains.
pub(crate) fn cases(count: usize) -> Vec<Case> {
    let mut runner = TestRunner::deterministic();
    let (geom, order) = (
        testgen::geom_prog_strategy(),
        testgen::order_prog_strategy(),
    );
    let (chain, tile) = (testgen::chain_strategy(), testgen::tile_prog_strategy());
    let def = EmitOpts::default;
    (0..count)
        .map(|z| match z % 6 {
            0 | 1 => {
                let (src, dims) = testgen::geom_src(&geom.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, false)),
                }
            }
            2 => {
                let (src, _) = testgen::geom_src(&order.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: EmitOpts::exact(),
                    tol: None,
                }
            }
            3 => {
                let (src, dims) =
                    testgen::geom_src(&order.new_tree(&mut runner).unwrap().current());
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, true)),
                }
            }
            4 => {
                // The emulator and the interpreter run every cell: mid
                // extents up to 2^25 products.
                let (src, dims) = loop {
                    let (src, (m, n, k)) =
                        testgen::mid_src(&geom.new_tree(&mut runner).unwrap().current());
                    if m * n * k <= 1 << 25 {
                        break (src, (m, n, k));
                    }
                };
                Case {
                    src,
                    base: def(),
                    tol: Some(geom_tol(dims, false)),
                }
            }
            _ if z % 12 == 5 => Case {
                src: testgen::tile_src(&tile.new_tree(&mut runner).unwrap().current(), true).0,
                base: def(),
                tol: None,
            },
            _ => Case {
                src: chain.new_tree(&mut runner).unwrap().current().source(),
                base: def(),
                tol: None,
            },
        })
        .collect()
}
