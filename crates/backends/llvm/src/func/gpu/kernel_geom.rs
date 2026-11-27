//! The kernel emitter's arithmetic, lane maps, panel/fragment layouts and the `Kernel` struct.

use super::*;

pub(super) struct Num;

impl Arith for Num {
    type V = u64;
    fn imm(&mut self, code: &str, a: &u64, c: u64) -> u64 {
        match code {
            "add" => a + c,
            "mul" => a * c,
            "udiv" => a / c,
            "urem" => a % c,
            _ => unreachable!("lane maps use add/mul/udiv/urem"),
        }
    }
    fn sum(&mut self, a: &u64, b: &u64) -> u64 {
        a + b
    }
}

impl Arith for Kernel<'_> {
    type V = String;
    fn imm(&mut self, code: &str, a: &String, c: u64) -> String {
        self.op(code, a, c)
    }
    fn sum(&mut self, a: &String, b: &String) -> String {
        self.op("add", a, b)
    }
}

/// S54 `raster`: the block `(bx, by)` of a `gx`-wide grid, launched as
/// `by·gx + bx`, walked `h` block-rows per column: launch groups of `h·gx`,
/// each down `h` rows, then the next column. A bijection on the grid when `h`
/// divides its height ([`Gpu::raster`] only derives such an `h`).
pub(super) fn raster_block<A: Arith>(
    ar: &mut A,
    bx: &A::V,
    by: &A::V,
    gx: u64,
    h: u64,
) -> (A::V, A::V) {
    let row = ar.imm("mul", by, gx);
    let lin = ar.sum(&row, bx);
    let group = ar.imm("udiv", &lin, h * gx);
    let local = ar.imm("urem", &lin, h * gx);
    let top = ar.imm("mul", &group, h);
    let dy = ar.imm("urem", &local, h);
    (ar.imm("udiv", &local, h), ar.sum(&top, &dy))
}

/// Where linear copy `li` (`tid + p·threads`) of a staged panel lands, as
/// (panel row, panel column): `a` → (`m`, `k`), `b` → (`k`, `n`).
#[derive(Clone, Copy, Debug)]
pub(super) enum LaneMap {
    /// `per_row` consecutive copies along a row, `unit` elements each.
    RowMajor { per_row: u64, unit: u64 },
    /// Consecutive copies down a `rows`-long column (`a` column-major).
    ColMajor { rows: u64 },
    /// `l` copies along a row, then down `rows` rows, then the next `l·unit`
    /// columns: the `cp.async` element copy (`l = lanes`, `unit` 1) and the
    /// `pad` rung's vector copy (`l` from [`TargetProfile::gpu_tile_sync_a`]).
    Split { l: u64, rows: u64, unit: u64 },
}

impl LaneMap {
    pub(super) fn coords<A: Arith>(self, ar: &mut A, li: &A::V) -> (A::V, A::V) {
        let scaled = |ar: &mut A, v: A::V, unit: u64| {
            if unit == 1 {
                v
            } else {
                ar.imm("mul", &v, unit)
            }
        };
        match self {
            LaneMap::RowMajor { per_row, unit } => {
                let r = ar.imm("udiv", li, per_row);
                let c = ar.imm("urem", li, per_row);
                (r, scaled(ar, c, unit))
            }
            LaneMap::ColMajor { rows } => {
                let c = ar.imm("udiv", li, rows);
                (ar.imm("urem", li, rows), c)
            }
            LaneMap::Split { l, rows, unit } => {
                let lo = ar.imm("urem", li, l);
                let q = ar.imm("udiv", li, l);
                let r = ar.imm("urem", &q, rows);
                let hi = ar.imm("udiv", &q, rows);
                let hi = ar.imm("mul", &hi, l);
                let c = ar.sum(&hi, &lo);
                (r, scaled(ar, c, unit))
            }
        }
    }
}

/// S53 rung `affine`: copy `p` is copy 0 moved by a constant when the lane map
/// is periodic in `threads` — `map(tid + p·threads) = map(tid) + map(p·threads)`
/// for every `tid` (`map(0) = (0, 0)`). With power-of-two extents it is
/// (`Q = threads / l`: `Q | rows` or `rows | Q`); it is checked here for every
/// thread rather than assumed, and a copy that fails keeps its own
/// `udiv`/`urem` chain.
pub(super) fn affine_delta(map: LaneMap, threads: u64, p: u64) -> Option<(u64, u64)> {
    let at = |li: u64| map.coords(&mut Num, &li);
    let d = at(p * threads);
    (0..threads)
        .all(|t| {
            let (r, c) = at(t);
            at(t + p * threads) == (r + d.0, c + d.1)
        })
        .then_some(d)
}

/// The tiled kernel's two staged copies, `[a, b]`, each as (lane map, copies
/// per thread per panel). A vector operand moves `lanes` elements along its
/// fast axis per copy; a strided one moves one element, consecutive threads
/// taking consecutive panel rows/lanes. Async staging copies `a` element by
/// element (its panel is transposed in shared memory, which a 16-byte copy
/// cannot do): `lanes` lanes along `k` within one row, then rows — four lanes
/// read one 16-byte global run, a warp covers 8 rows, and `pad_a` keeps those
/// 32 writes on 32 banks. Register staging under `pad`: `a_row_lanes` per row.
pub(super) fn lane_maps(s: &TileSite, g: &GpuTile, kmajor: bool, kvb: bool) -> [(LaneMap, u64); 2] {
    let e = elem_bytes_of(&s.elem);
    let lanes = GPU_VEC_BYTES / e;
    let (bm, bn, bk, threads) = (g.bm(), g.bn(), g.bk, g.threads());
    let (kv, nv) = (bk / lanes, bn / lanes);
    let streamed = g.stages >= 2;
    // tile-efficiency T2: a k-major `a` panel is copied as it lies — `kv`
    // 16-byte copies along each row, async or register-staged alike.
    let a = if kmajor {
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bm * kv / threads)
    } else if streamed {
        let map = LaneMap::Split {
            l: lanes,
            rows: bm,
            unit: 1,
        };
        (map, bm * bk / threads)
    } else if !vector_a(s) {
        (LaneMap::ColMajor { rows: bm }, bm * bk / threads)
    } else if g.a_row_lanes > 0 && g.a_row_lanes < kv {
        let map = LaneMap::Split {
            l: g.a_row_lanes,
            rows: bm,
            unit: lanes,
        };
        (map, bm * kv / threads)
    } else {
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bm * kv / threads)
    };
    let b = if vector_b(s) {
        let map = LaneMap::RowMajor {
            per_row: nv,
            unit: lanes,
        };
        (map, bk * nv / threads)
    } else if kvb {
        // `kmajor-b`: the copies walk each key's row (`(n, k)` coordinates).
        let map = LaneMap::RowMajor {
            per_row: kv,
            unit: lanes,
        };
        (map, bn * kv / threads)
    } else {
        let map = LaneMap::RowMajor {
            per_row: bn,
            unit: 1,
        };
        (map, bk * bn / threads)
    };
    [a, b]
}

/// tile-efficiency T4: the LSU warp-instructions one thread of tile `g` issues
/// per panel trip, as [`Kernel::tiled`] emits them: `bk` steps of `a` and `b`
/// fragment loads, each copy's global load or `cp.async`, and the
/// register-staged copies' shared stores (`a` scattered `lanes` ways when it
/// is staged transposed). per-site-choice A3: each shared-memory class counts
/// its bank excess ([`panel_excess`]) times — a replayed wavefront is one more
/// LSU slot; a conflict-free kernel's count is its instruction count.
pub(crate) fn lsu_per_panel(s: &TileSite, g: &GpuTile, rungs: &crate::GpuRungs, col: bool) -> u64 {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    let kmaj = g.kmajor && !col;
    let kvb = rungs.kmajor_b && kvec_b(s, g);
    let [(_, a), (_, b)] = lane_maps(s, g, kmaj, kvb);
    // `kmajor-b`: each `b` copy scatters `lanes` stores.
    let bw = if kvb { lanes } else { 1 };
    // per-site-choice D1: the excess weights are `bank-plan`'s; off (and the
    // plan line's `lsu_issue`), every class counts its instructions once.
    let [xa, xb, sa, sb] = match rungs.bank_plan {
        true => panel_excess(s, g, rungs, col),
        false => [1; 4],
    };
    let lds_a = match kmaj {
        true => g.bk * g.tm / lanes,
        false => g.bk * g.tm / g.tm.min(lanes),
    };
    let lds_b = g.bk * g.tn / g.tn.min(lanes);
    let copies = match g.stages >= 2 {
        true => a * sa + b * sb,
        false if vector_a(s) && !kmaj => a + b + a * lanes * sa + b * bw * sb,
        false => a + b + a * sa + b * bw * sb,
    };
    lds_a * xa + lds_b * xb + copies
}

/// [`Kernel::tiled`]'s thread grid: `(trow, tcol)` of thread `tid`, in IR
/// ([`Kernel`]) or as numbers ([`Num`], the bank model). Warp shape (S52
/// rung `warp`): a warp's 32 lanes as `warp / wx` rows × `wx` columns of
/// threads, `wx` = the lanes one shared-memory wavefront serves with vector
/// loads (128 B / 16 B = 8) — 4 + 8 distinct fragments per `k` step instead
/// of 2 + 16. Off arm / a block too narrow for it: row-major threads. S54
/// `rowfold-layout` (`col`): the row-major rule mirrored, so one column-major
/// store still writes whole 128-byte lines (`col_store`).
pub(super) fn thread_coords<A: Arith>(
    ar: &mut A,
    tid: &A::V,
    g: &GpuTile,
    warp_rung: bool,
    col: bool,
    warp: u64,
) -> (A::V, A::V) {
    let w = GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES;
    let (wx, wy) = if col { (warp / w, w) } else { (w, warp / w) };
    if warp_rung && g.tx % wx == 0 && g.ty % wy == 0 && g.tx > wx {
        let wp = ar.imm("udiv", tid, warp);
        let lane = ar.imm("urem", tid, warp);
        let warps_x = g.tx / wx;
        let wr = ar.imm("udiv", &wp, warps_x);
        let wc = ar.imm("urem", &wp, warps_x);
        let (lr, lc) = if col {
            (ar.imm("urem", &lane, wy), ar.imm("udiv", &lane, wy))
        } else {
            (ar.imm("udiv", &lane, wx), ar.imm("urem", &lane, wx))
        };
        let r0 = ar.imm("mul", &wr, wy);
        let c0 = ar.imm("mul", &wc, wx);
        return (ar.sum(&r0, &lr), ar.sum(&c0, &lc));
    }
    if col {
        (ar.imm("urem", tid, g.ty), ar.imm("udiv", tid, g.ty))
    } else {
        (ar.imm("udiv", tid, g.tx), ar.imm("urem", tid, g.tx))
    }
}

/// `sa`: elements between two rows of the `a` panel. Transposed (one row a
/// `k`): `bm + pad_a`, padded on `cp.async` and under `pad`
/// ([`TargetProfile::gpu_tile_sync_a`]). tile-efficiency T2: k-major
/// (`As[m][k]`): `bk + pad_a` — unless the kernel stores column-major (its
/// quarter-warps read eight panel rows), which keeps the transposed panel.
pub(super) fn panel_sa(g: &GpuTile, kmaj: bool) -> u64 {
    match (g.kmajor, kmaj) {
        (true, true) => g.bk + g.pad_a,
        (true, false) => g.bm(),
        _ => g.sa(),
    }
}

/// Where a thread's register-tile fragments sit in the panels: its `i`-th
/// `a` row is `trow·a_step + (i / aw)·a_chunk + i % aw`, its `b` vectors
/// `tcol·b_step + v·b_chunk`.
pub(super) struct Frag {
    pub(super) aw: u64,
    pub(super) a_step: u64,
    pub(super) a_chunk: u64,
    pub(super) b_step: u64,
    pub(super) b_chunk: u64,
    pub(super) narrow: bool,
}

/// Bank geometry: a thread's register tile is `tm/lanes × tn/lanes` vector
/// chunks spaced one thread-row of chunks apart (`ty·lanes` rows, `tx·lanes`
/// columns), not one contiguous `tm × tn` block. Consecutive threads then read
/// consecutive 16-byte chunks of a panel row, which is what a shared-memory
/// wavefront serves without a conflict — and their output stores coalesce.
/// (Off arm: one contiguous `tm × tn` block per thread, chunk = lanes.)
/// S62 `fill-rows`: a register tile `tm` rows tall, `tm < lanes`, reads its
/// `a` fragment as one `tm`-wide chunk (`aw`); `tn` keeps whole vectors (the
/// output's vector store). A tile narrower than a vector in `tn`
/// (`cell-fill`'s 1 × 1) has no chunks to space (`narrow`). A mixed-radix
/// output stores a vector where its unit lane digit runs `lanes`-aligned and
/// every other digit moves by whole vectors.
///
/// per-site-choice A1: a k-major thread owns rows `trow + i·ty`. One 16-byte
/// fragment load of a warp then reads its `warp / wx` consecutive rows, which
/// a row pad spreads over the banks ([`kmajor_pad`]); rows `aw` apart are
/// ≡ 0 mod 4 rows of a 16-byte-aligned stride and no pad separates them.
/// Each cell's `k` order is unchanged; the output rows follow the same map.
pub(super) fn frag_layout(s: &TileSite, g: &GpuTile, kmaj: bool, interleave: bool) -> Frag {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
    let (tm, tn) = (g.tm, g.tn);
    let aw = tm.min(lanes);
    let out_vec = s
        .radix
        .as_deref()
        .and_then(|x| x.out.as_ref())
        .is_none_or(|o| {
            let v = |c: u64| c.is_multiple_of(lanes);
            o.lane
                .first()
                .is_some_and(|d| d.stride == 1 && d.coef == 1 && v(d.radix))
                && o.i.iter().chain(&o.lane[1..]).all(|d| v(d.coef))
                && v(o.base)
        });
    let narrow = tn < lanes || !out_vec;
    let (a_step, b_step, a_chunk, b_chunk) = if interleave && !narrow {
        (aw, lanes, g.ty * aw, g.tx * lanes)
    } else {
        (tm, tn, aw, lanes)
    };
    let (aw, a_step, a_chunk) = match kmaj {
        true => (1, 1, g.ty),
        false => (aw, a_step, a_chunk),
    };
    Frag {
        aw,
        a_step,
        a_chunk,
        b_step,
        b_chunk,
        narrow,
    }
}

/// **per-site-choice A3: the bank excess of tile `g`'s four shared-memory
/// access classes** — `[a fragment read, b fragment read, a copy write,
/// b copy write]` (copy writes: register-staged stores or `cp.async`) — as
/// [`Kernel::tiled`] emits them ([`thread_coords`], [`frag_layout`],
/// [`lane_maps`]): the worst [`bank_degree`] ÷ ideal (rounded up) over every
/// warp and every instruction of one panel step; 1 = conflict-free. Steps
/// and stages move every lane by one constant: the same degree.
pub(crate) fn panel_excess(
    s: &TileSite,
    g: &GpuTile,
    rungs: &crate::GpuRungs,
    col: bool,
) -> [u64; 4] {
    let e = elem_bytes_of(&s.elem);
    let (lanes, wpe) = (GPU_VEC_BYTES / e, e.div_ceil(4));
    let threads = g.threads();
    let kmaj = g.kmajor && !col;
    let sa = panel_sa(g, kmaj);
    let f = frag_layout(s, g, kmaj && rungs.bank_plan, rungs.interleave);
    let at: Vec<(u64, u64)> = (0..threads)
        .map(|t| thread_coords(&mut Num, &t, g, rungs.warp, col, WARP))
        .collect();
    // The worst excess of `n` instructions, `acc(t, p)` thread `t`'s
    // (element, elements) in instruction `p`.
    let worst = |n: u64, acc: &dyn Fn(u64, u64) -> (u64, u64)| {
        let mut x = 1;
        for p in 0..n {
            for w0 in (0..threads).step_by(WARP as usize) {
                let words = (w0..threads.min(w0 + WARP)).map(|t| {
                    let (a, len) = acc(t, p);
                    (a * wpe, len * wpe)
                });
                let (d, i) = bank_degree(words);
                x = x.max(d.div_ceil(i));
            }
        }
        x
    };
    let lds_a = match kmaj {
        true => worst(g.tm, &|t, i| {
            let r = at[t as usize].0 * f.a_step + (i / f.aw) * f.a_chunk + i % f.aw;
            (r * sa, lanes)
        }),
        false => {
            let w = g.tm.min(lanes);
            worst(g.tm / w, &|t, v| {
                (at[t as usize].0 * f.a_step + v * f.a_chunk, w)
            })
        }
    };
    let w = g.tn.min(lanes);
    let lds_b = worst(g.tn / w, &|t, v| {
        (at[t as usize].1 * f.b_step + v * f.b_chunk, w)
    });
    let kvb = rungs.kmajor_b && kvec_b(s, g);
    let [(am, an), (bm, bn)] = lane_maps(s, g, kmaj, kvb);
    let rc = |m: LaneMap, t: u64, p: u64| m.coords(&mut Num, &(t + p * threads));
    let st_a = match (kmaj, g.stages < 2 && vector_a(s)) {
        (true, _) => worst(an, &|t, p| {
            let (r, kc) = rc(am, t, p);
            (r * sa + kc, lanes)
        }),
        // Transposed and register-staged: a vector along `k` scatters to
        // `lanes` panel rows, one store each.
        (false, true) => worst(an * lanes, &|t, p| {
            let (r, kc) = rc(am, t, p / lanes);
            ((kc + p % lanes) * sa + r, 1)
        }),
        (false, false) => worst(an, &|t, p| {
            let (r, kc) = rc(am, t, p);
            (kc * sa + r, 1)
        }),
    };
    let bw = if vector_b(s) { lanes } else { 1 };
    let swz = kvec_swizzle(s, g, rungs);
    let st_b = match kvb {
        // `kmajor-b`: a vector along `k` scatters to `lanes` panel rows
        // (`kvec-swizzle`: each key chunk's own bank group).
        true => worst(bn * lanes, &|t, p| {
            let (nc, kc) = rc(bm, t, p / lanes);
            let sw = swz.map_or(0, |(gs, st)| ((kc / lanes) % gs) * st);
            ((kc + p % lanes) * g.bn() + (nc ^ sw), 1)
        }),
        false => worst(bn, &|t, p| {
            let (kr, nc) = rc(bm, t, p);
            (kr * g.bn() + nc, bw)
        }),
    };
    [lds_a, lds_b, st_a, st_b]
}

/// ADR-0050 C4: **the FMA tile emitter's declared constraints** on tile `g`
/// of `site` under `rungs` (what [`Kernel::tiled`] assumes): every copy of
/// a panel whole — each operand's lane map moves its `bm·bk` / `bk·bn`
/// elements in whole loads a thread ([`lane_maps`]) —, a streamed panel
/// (`cp.async`) never narrows `a` and narrows `b` only where `narrow-async`
/// streams it, the k-major read on a warp at least a wavefront wide, a
/// whole-file tile at least its counted registers. `Err`: the first broken.
pub(crate) fn fma_constraints(
    site: &TileSite,
    g: &GpuTile,
    rungs: &crate::GpuRungs,
) -> Result<(), &'static str> {
    let e = elem_bytes_of(&site.elem);
    let (bm, bn, bk, threads) = (g.bm(), g.bn(), g.bk, g.threads());
    if threads == 0 || bk == 0 {
        return Err("an empty tile");
    }
    let streamed = g.stages >= 2;
    if streamed && site.a.narrow.is_some() {
        return Err("a narrow `a` panel is register-staged");
    }
    if streamed && site.b.narrow.is_some() && !narrow_streams(site, rungs) {
        return Err("a narrow `b` panel streams only under narrow-async");
    }
    if g.kmajor && g.tx < GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES {
        return Err("the k-major read needs a wavefront-wide warp row");
    }
    if g.held > 0 && g.held < crate::profile::fma_held_regs(g.tm, g.tn, e) {
        return Err("held registers below the counted footprint");
    }
    let kvb = rungs.kmajor_b && kvec_b(site, g);
    let [(am, an), (bmap, bnn)] = lane_maps(site, g, g.kmajor, kvb);
    let unit = |m: LaneMap| match m {
        LaneMap::RowMajor { unit, .. } | LaneMap::Split { unit, .. } => unit,
        LaneMap::ColMajor { .. } => 1,
    };
    if an == 0 || an * threads * unit(am) != bm * bk {
        return Err("`a`'s panel copies are not whole loads a thread");
    }
    if bnn == 0 || bnn * threads * unit(bmap) != bk * bn {
        return Err("`b`'s panel copies are not whole loads a thread");
    }
    Ok(())
}

/// ADR-0050 C4: whether the FMA tile streams a panel of `site` through
/// `cp.async` (a narrow `a` is register-staged; a narrow `b` streams only
/// under `narrow-async`).
pub(crate) fn fma_streams(site: &TileSite, rungs: &crate::GpuRungs) -> bool {
    site.a.narrow.is_none() && (site.b.narrow.is_none() || narrow_streams(site, rungs))
}

/// ADR-0050 C4: **the FMA tile's layout by the bank model** (dominance: no
/// evaluator term reads a pad, so the search takes the emitter's own
/// conflict-free choice) — a k-major `a` panel's least worst-excess pad
/// ([`kmajor_pad`]); a register-staged vector `a`'s row lanes and pad
/// ([`TargetProfile::gpu_tile_sync_a`]: [`conflict_free_pad`] over its copy
/// map); otherwise none.
pub(crate) fn fma_layout(
    site: &TileSite,
    profile: &TargetProfile,
    g: GpuTile,
    rungs: &crate::GpuRungs,
) -> GpuTile {
    let g = GpuTile {
        pad_a: 0,
        a_row_lanes: 0,
        ..g
    };
    if g.kmajor {
        return GpuTile {
            pad_a: kmajor_pad(site, &g, rungs),
            ..g
        };
    }
    if g.stages < 2 && vector_a(site) {
        return profile.gpu_tile_sync_a(g, &site.elem);
    }
    g
}

/// ADR-0050 C4: one kernel's text as a device module of its own (the
/// preamble, its loop metadata) — what the post-lowering gate assembles.
pub(crate) fn kernel_module(text: &str) -> String {
    let mut out = String::from(DEVICE_PREAMBLE);
    out.push('\n');
    if text.contains("!@LOOP@") {
        out.push_str(&text.replace("!@LOOP@", "!0"));
        out.push_str("!0 = distinct !{!0, !1}\n!1 = !{!\"llvm.loop.unroll.full\"}\n");
    } else {
        out.push_str(text);
    }
    out
}

/// FNV-1a of a text (the gate's scratch names).
pub(crate) fn fnv_text(t: &str) -> u64 {
    t.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// ADR-0050 C4: the panel widths a 16-bit tensor tile of `fmt` stages
/// `site`'s operands at ([`mma16_layout`]), `None` where it cannot.
pub(crate) fn mma_widths(site: &TileSite, fmt: MmaFmt) -> Option<(u64, u64)> {
    mma16_layout(site, fmt)
}

/// ADR-0050 C4 **post-lowering gate**: the kernel text launch `l` lowers to
/// (its realization `l.combine`), or `Err` with the emitter's refusal —
/// emitted under [`crate::place::forced`], so every launch the emitter
/// derives inside is `l`.
pub(crate) fn lowered_text(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    l: &Launch,
) -> Result<String, String> {
    let name = kernel_name(kernel_id(site, profile, contract));
    crate::place::forced(l, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kernel_text(&name, site, profile, contract, arms, false, None, l.combine)
        }))
        .map_err(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "the emitter refused".into())
        })
    })
}

/// per-site-choice A2: the k-major `a` panel's row pad — the least, in
/// 16-byte steps under 32 words, that minimises the worst excess of its
/// fragment read and copy write ([`panel_excess`]); `b` does not move.
/// For f32 at the warp shape: `bk ≥ 16` ⇒ 4, `bk = 8` ⇒ 0.
pub(super) fn kmajor_pad(s: &TileSite, g: &GpuTile, rungs: &crate::GpuRungs) -> u64 {
    let wpe = elem_bytes_of(&s.elem).div_ceil(4);
    let mut best = (u64::MAX, 0);
    for pad in (0..32 / wpe).step_by((GPU_VEC_BYTES / 4 / wpe).max(1) as usize) {
        let [a, _, st, _] = panel_excess(s, &GpuTile { pad_a: pad, ..*g }, rungs, false);
        best = best.min((a.max(st), pad));
        if best.0 == 1 {
            break;
        }
    }
    best.1
}

/// String builder for one kernel — the device twin of [`FnEmit`]'s
/// `line`/`tmp`, kept separate because a kernel shares no state with the host
/// function around the site.
pub(super) struct Kernel<'s> {
    pub(super) out: String,
    pub(super) next: u32,
    /// The index type: `i64`, or `i32` under `idx32` ([`max_index`]).
    pub(super) idx: &'static str,
    /// `float` or `double`.
    pub(super) t: String,
    pub(super) site: &'s TileSite,
    pub(super) arithmetic: Arithmetic,
    pub(super) rungs: crate::GpuRungs,
    pub(super) profile_gpu: Gpu,
    /// S54 `pdl`: open with `griddepcontrol.wait` (and let a dependent launch).
    pub(super) pdl: bool,
    /// The panel loop's induction values, threaded from its head to its latch.
    pub(super) async_tail: Option<(String, String, u64)>,
    pub(super) sync_tail: Option<(String, String)>,
    /// `(gx, h)`: the grid width and the group height of [`raster_block`].
    pub(super) raster: (u64, u64),
    /// Store the output column-major, `o[j·rows + i]` (S54 `rowfold-layout`):
    /// the order its only reader, a row fold, reads it in.
    pub(super) col: bool,
    /// S54: the per-cell consumers evaluated in this kernel's store.
    pub(super) epi: Option<&'s EpiKernel>,
    /// S62 `splitk`: this kernel folds partition `z = ctaid.y / gy_out` of
    /// `splitk` contiguous `k` ranges into workspace slab `z`; 1 = ordered.
    pub(super) splitk: u64,
    pub(super) gy_out: u64,
    pub(super) z: Option<String>,
    /// S63 `splitk-atomic` / `streamk`: the partials add into `%o` itself
    /// (`atomicrmw fadd`), which the host zeroed; no workspace slab.
    pub(super) atomic: bool,
    /// plan-cost `linear` (§4a concat): the site table's terms — term
    /// `ctaid.y ÷ (gy_out·S)` reads `%a<t>`, `%b<t>` and adds into `%o`; 1 =
    /// one site.
    pub(super) terms: u64,
    /// S63 `streamk`: the units of the `tiles × panels` space one block walks.
    pub(super) streamk: Option<u64>,
    /// ADR-0050 C5 hybrid: the blocks before the walk, each one whole tile
    /// ([`Combine::StreamK`]'s `whole`; 0: the plain walk).
    pub(super) whole: u64,
    /// sweep-runtime-2 `streamk-epi`: a stream-K kernel of an epilogue group
    /// — its partials add into the workspace `%w` (the sums, then one `i32`
    /// ticket a tile), and the CTA that completes a tile evaluates the
    /// group's members on its whole sums ([`Kernel::ticket`]).
    pub(super) ticket: bool,
    /// sweep-runtime-2 `streamk-epi`: the store pass of a shared tile's piece
    /// (its partials into the workspace slot), not a whole tile's.
    pub(super) part: bool,
    /// S63 `streamk`: the current piece's first panel (the seed rides the
    /// piece that starts its tile, [`Kernel::split_seed`]).
    pub(super) p0: Option<String>,
    /// The block the panel loop is entered from: `entry`, or `seg` in a
    /// stream-K kernel (one panel loop per piece).
    pub(super) head: &'static str,
    /// hopper-wgmma-tma: the tensor maps this kernel takes by value after
    /// its pointers ([`ws_tensors`]'s recipes), carried to the runtime as
    /// the global `<kernel>_tmap`.
    pub(super) tmap: Vec<[u64; 20]>,
    /// gapfix `stream-ld`: the operands (`a`, `b`) this kernel loads with
    /// the streaming policy — the plan line's `evict=` (intent.py).
    pub(super) streamed: [bool; 2],
}
