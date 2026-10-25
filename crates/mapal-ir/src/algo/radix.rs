//! **The mixed-radix index fact** (recurrence-precision item 3).
//!
//! A counted index split by constant divisions and remainders — `t / 3136 % 64`,
//! `k / 9`, `k % 9 / 3` — is a sum of *digits* `(x ÷ stride) % radix`. An
//! address built from such digits by `+` and `·literal` is affine per digit
//! even where it is not affine in `x` itself (a convolution's
//! `img[((b·C + k/9)·w + i + k%9/3)·w + j + k%3]`). [`CategoryIr::radix_plan`]
//! records that form for every integer value of a function;
//! [`CategoryIr::tile_plan_radix`] reads it to admit a map/fold whose reads are
//! digit-affine as a [`TileSite`] over `(i, lane, k)`: the map index's digits
//! read only by one operand are its rows, those read only by the other its
//! lanes, the fold index's digits its `k` (an implicit GEMM).

use slotmap::SecondaryMap;

use super::{TileRead, TileSite};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::Ty;

/// One digit of input slot `slot`: `coef · ((input[slot] ÷ stride) % radix)`,
/// `radix == 0` for no remainder (the quotient itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digit {
    pub slot: u32,
    pub stride: u64,
    pub radix: u64,
    pub coef: u64,
}

/// An integer value as `base + Σ digits` of its function's inputs — valid for
/// non-negative inputs whose sums do not wrap (the consumer bounds both).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadixForm {
    pub base: u64,
    pub digits: Vec<Digit>,
}

impl RadixForm {
    fn add(&self, o: &Self) -> Option<Self> {
        let mut r = self.clone();
        r.base = r.base.checked_add(o.base)?;
        for d in &o.digits {
            match r
                .digits
                .iter_mut()
                .find(|x| (x.slot, x.stride, x.radix) == (d.slot, d.stride, d.radix))
            {
                Some(x) => x.coef = x.coef.checked_add(d.coef)?,
                None => r.digits.push(*d),
            }
        }
        Some(r)
    }

    fn scale(&self, by: u64) -> Option<Self> {
        let mut r = self.clone();
        r.base = r.base.checked_mul(by)?;
        for d in &mut r.digits {
            d.coef = d.coef.checked_mul(by)?;
        }
        r.digits.retain(|d| d.coef != 0);
        Some(r)
    }

    /// The single unscaled digit this form is, if it is one.
    fn digit(&self) -> Option<Digit> {
        match self.digits.as_slice() {
            [d] if self.base == 0 && d.coef == 1 => Some(*d),
            _ => None,
        }
    }
}

/// One digit of a tile axis `x` (`i`, `lane` or `k`) in a [`RadixAddr`]:
/// `coef · ((x ÷ stride) % radix)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisDigit {
    pub stride: u64,
    pub radix: u64,
    pub coef: u64,
}

/// An address over a tile site's axes, each a mixed-radix sum:
/// `base + Σ i-digits + Σ lane-digits + Σ k-digits`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadixAddr {
    pub base: u64,
    pub i: Vec<AxisDigit>,
    pub lane: Vec<AxisDigit>,
    pub k: Vec<AxisDigit>,
}

/// The addresses of a [`TileSite`] that are digit-affine but not affine:
/// `None` = the plain form (the [`TileRead`]'s `ci`/`clane`/`ck`; for `out`,
/// row-major `i·c + lane`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RadixTile {
    pub a: Option<RadixAddr>,
    pub b: Option<RadixAddr>,
    pub out: Option<RadixAddr>,
}

fn axis_sum(ds: &[AxisDigit], x: u64) -> u64 {
    ds.iter().map(|d| d.coef * ((x / d.stride) % d.radix)).sum()
}

impl RadixAddr {
    /// The address at `(i, lane, k)` — the definition the kernels and tests share.
    pub fn at(&self, i: u64, lane: u64, k: u64) -> u64 {
        self.base + axis_sum(&self.i, i) + axis_sum(&self.lane, lane) + axis_sum(&self.k, k)
    }

    /// The largest address over the extents (every digit at its top).
    pub fn max(&self) -> u64 {
        let top = |ds: &[AxisDigit]| ds.iter().map(|d| d.coef * (d.radix - 1)).sum::<u64>();
        self.base + top(&self.i) + top(&self.lane) + top(&self.k)
    }
}

/// The cut points `1 = c₀ < … < cₙ = n` that every digit's range starts and
/// ends on, each dividing the next; `None` when they do not chain.
fn basis(n: u64, digits: &[(u64, u64)]) -> Option<Vec<u64>> {
    let mut cuts = vec![1, n];
    for &(s, r) in digits {
        if s < n {
            cuts.push(s);
            if r != 0 {
                cuts.push(s.checked_mul(r)?.min(n));
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2)
        .all(|w| w[1].is_multiple_of(w[0]))
        .then_some(cuts)
}

/// Digit `(s, r, coef)` of an index with extent `n` over `cuts`: one
/// coefficient per basis digit `[cⱼ, cⱼ₊₁)`.
fn over(cuts: &[u64], n: u64, s: u64, r: u64, coef: u64) -> Option<Vec<u64>> {
    let top = if r == 0 { n } else { s.checked_mul(r)?.min(n) };
    cuts.windows(2)
        .map(|w| {
            Some(if s >= n || w[0] < s || w[1] > top {
                0
            } else {
                coef.checked_mul(w[0] / s)?
            })
        })
        .collect()
}

/// Merge stride-adjacent digits whose coefficients continue each other
/// (`coef₂ = coef₁·radix₁`): `(x%3)·1 + (x÷3%3)·3` is `x%9`.
fn merged(mut ds: Vec<AxisDigit>) -> Vec<AxisDigit> {
    ds.retain(|d| d.coef != 0 && d.radix > 1);
    ds.sort_by_key(|d| d.stride);
    let mut out: Vec<AxisDigit> = Vec::new();
    for d in ds {
        match out.last_mut() {
            Some(p) if p.stride * p.radix == d.stride && p.coef * p.radix == d.coef => {
                p.radix *= d.radix;
            }
            _ => out.push(d),
        }
    }
    out
}

/// The plain coefficient of an axis: none (0) or one digit over the whole extent.
fn plain(ds: &[AxisDigit], extent: u64) -> Option<u64> {
    match ds {
        [] => Some(0),
        [d] if d.stride == 1 && d.radix == extent => Some(d.coef),
        _ => None,
    }
}

impl CategoryIr {
    /// The deduced query `radix : IR × FuncId → ObjectId ↦ RadixForm`: every
    /// integer value of `f` built from `f`'s input projections and non-negative
    /// literals by `+`, `·literal`, `÷literal` and `%literal`, as a sum of
    /// digits of those inputs. A quotient or remainder of anything but one
    /// unscaled digit (or a remainder that does not divide the digit's radix)
    /// has no form. Stored on the graph.
    pub fn radix_plan(&self, f: FuncId) -> SecondaryMap<ObjectId, RadixForm> {
        self.fact(f, |s| &s.radix_plan, || self.radix_plan_compute(f))
    }

    fn radix_plan_compute(&self, f: FuncId) -> SecondaryMap<ObjectId, RadixForm> {
        let mut out = SecondaryMap::new();
        let Some(fd) = self.func(f) else {
            return out;
        };
        for m in self.topo_order(f) {
            let morph = &self.morphisms[m];
            if !matches!(self.objects.get(morph.target).map(|o| &o.ty), Some(Ty::Int { .. })) {
                continue;
            }
            let arg = |slot| {
                let o = self.pair_slot_source(morph.source, slot)?;
                if let Some(v) = self.tile_literal_u64(o) {
                    return Some(RadixForm {
                        base: v,
                        digits: Vec::new(),
                    });
                }
                out.get(o).cloned()
            };
            let lit = |slot| self.tile_literal_u64(self.pair_slot_source(morph.source, slot)?);
            let form = match morph.op {
                Operation::Proj { index } if morph.source == fd.input => Some(RadixForm {
                    base: 0,
                    digits: vec![Digit {
                        slot: index,
                        stride: 1,
                        radix: 0,
                        coef: 1,
                    }],
                }),
                Operation::Add => arg(0).zip(arg(1)).and_then(|(x, y)| x.add(&y)),
                Operation::Mul => match (lit(0), lit(1)) {
                    (Some(c), _) => arg(1).and_then(|x| x.scale(c)),
                    (_, Some(c)) => arg(0).and_then(|x| x.scale(c)),
                    _ => None,
                },
                Operation::Div | Operation::Mod => lit(1).filter(|&c| c > 0).and_then(|c| {
                    let x = arg(0)?;
                    if x.digits.is_empty() {
                        let v = if morph.op == Operation::Div { x.base / c } else { x.base % c };
                        return Some(RadixForm {
                            base: v,
                            digits: Vec::new(),
                        });
                    }
                    let d = x.digit()?;
                    let d = match morph.op {
                        Operation::Div if d.radix == 0 => Digit {
                            stride: d.stride.checked_mul(c)?,
                            ..d
                        },
                        Operation::Div if d.radix.is_multiple_of(c) => Digit {
                            stride: d.stride.checked_mul(c)?,
                            radix: d.radix / c,
                            ..d
                        },
                        Operation::Mod if d.radix == 0 || d.radix.is_multiple_of(c) => {
                            Digit { radix: c, ..d }
                        }
                        Operation::Mod if d.radix <= c => d,
                        _ => return None,
                    };
                    Some(RadixForm {
                        base: 0,
                        digits: vec![d],
                    })
                }),
                _ => None,
            };
            if let Some(form) = form {
                out.insert(morph.target, form);
            }
        }
        out
    }

    /// The mixed-radix tile sites of `f` ([`TileSite::radix`] set): map/fold
    /// sites [`Self::tile_plan_strided`] refuses whose two reads and output are
    /// digit-affine in the map index `t` and the fold index `k`. Each digit of
    /// `t` must be read by exactly one operand — the operand reading `t`'s unit
    /// digit is `b` (its digits are the lanes, so the output's fast axis is the
    /// lane), the other `a` (rows). `k` keeps its order: the site folds every
    /// cell in increasing `k`, the oracle's chain. Stored on the graph.
    pub fn tile_plan_radix(&self, f: FuncId) -> SecondaryMap<MorphismId, TileSite> {
        self.fact(f, |s| &s.tile_plan_radix, || {
            let mut out = SecondaryMap::new();
            let strided = self.tile_plan_strided(f);
            for &m in self.func(f).map(|fd| fd.morphisms.as_slice()).unwrap_or(&[]) {
                if strided.sites.contains_key(m) {
                    continue;
                }
                if let Some(s) = self.radix_site(m) {
                    out.insert(m, s);
                }
            }
            out
        })
    }

    fn radix_site(&self, site: MorphismId) -> Option<TileSite> {
        let map = self.morphisms.get(site)?;
        let Operation::Map {
            body,
            captures: map_captures,
        } = map.op
        else {
            return None;
        };
        let n = self.tile_iota_size(self.pair_slot_source(map.source, map_captures)?)?;
        let Ty::Array { size, elem: out_elem } = &self.objects.get(map.target)?.ty else {
            return None;
        };
        // One output (`n = 1`) has no digit of `t` to tile along.
        if *size != n || n < 2 || map_captures == 0 {
            return None;
        }
        let body_def = self.func(body)?;
        let folds: Vec<MorphismId> = body_def
            .morphisms
            .iter()
            .copied()
            .filter(|&m| matches!(self.morphisms[m].op, Operation::Fold { .. }))
            .collect();
        let [fold_id] = folds[..] else {
            return None;
        };
        let fold = self.morphisms.get(fold_id)?;
        let Operation::Fold {
            body: fold_body,
            captures: fc,
        } = fold.op
        else {
            return None;
        };
        if fold.target != body_def.output {
            return None;
        }
        let kn = self.tile_iota_size(self.pair_slot_source(fold.source, fc.checked_add(1)?)?)?;
        let seed_obj = self.objects.get(self.pair_slot_source(fold.source, fc)?)?;
        if seed_obj.kind != ObjectKind::Constant || kn == 0 {
            return None;
        }
        let seed = seed_obj.value.clone()?;
        if !self.tile_trap_free(body, &self.bounds_proof(body), Some(fold_id))
            || !self.tile_trap_free(fold_body, &self.bounds_proof(fold_body), None)
        {
            return None;
        }
        let fold_def = self.func(fold_body)?;
        let add = self.tile_definer(fold_def.output, Operation::Add)?;
        let (l, r) = (
            self.pair_slot_source(add.source, 0)?,
            self.pair_slot_source(add.source, 1)?,
        );
        let acc_of = |o| self.tile_input_proj_index(o, fold_def.input) == Some(fc);
        let (acc, product, add_acc_first) = match (acc_of(l), acc_of(r)) {
            (true, false) => (l, r, true),
            (false, true) => (r, l, false),
            _ => return None,
        };
        let mul = self.tile_definer(product, Operation::Mul)?;
        let map_forms = self.radix_plan(body);
        let fold_forms = self.radix_plan(fold_body);
        // One read: its map-capture slot, and its address as (base, t-digits,
        // k-digits), each digit `(stride, radix, coef)` of the raw index.
        type Read = (u32, u64, Vec<(u64, u64, u64)>, Vec<(u64, u64, u64)>, u64);
        let read = |value: ObjectId| -> Option<(Read, Ty)> {
            let index = self.tile_definer(value, Operation::Index)?;
            let array = self.pair_slot_source(index.source, 0)?;
            let addr = self.pair_slot_source(index.source, 1)?;
            let held = self.tile_fold_capture(array, fold.source, fold_def.input, fc)?;
            let slot = self.tile_input_proj_index(held, body_def.input)?;
            let Ty::Array { elem, size } = &self.objects.get(array)?.ty else {
                return None;
            };
            if slot >= map_captures || self.objects.get(value)?.ty != **elem {
                return None;
            }
            let form = fold_forms.get(addr)?;
            let (mut base, mut td, mut kd) = (form.base, Vec::new(), Vec::new());
            for d in &form.digits {
                if d.slot == fc + 1 {
                    kd.push((d.stride, d.radix, d.coef));
                    continue;
                }
                // A captured value enters linearly: substitute its map-body form.
                if d.slot >= fc || d.stride != 1 || d.radix != 0 {
                    return None;
                }
                let cap = self.pair_slot_source(fold.source, d.slot)?;
                let cf = map_forms.get(cap)?.scale(d.coef)?;
                base = base.checked_add(cf.base)?;
                for e in &cf.digits {
                    if e.slot != map_captures {
                        return None;
                    }
                    td.push((e.stride, e.radix, e.coef));
                }
            }
            Some(((slot, base, td, kd, *size), (**elem).clone()))
        };
        let (x, ex) = read(self.pair_slot_source(mul.source, 0)?)?;
        let (y, ey) = read(self.pair_slot_source(mul.source, 1)?)?;
        let acc_ty = &self.objects.get(acc)?.ty;
        if ex != ey || &ex != acc_ty || ex != **out_elem || seed.ty() != ex || !ex.is_numeric() {
            return None;
        }
        // Common bases of `t` and `k`.
        let pairs = |v: &[(u64, u64, u64)]| v.iter().map(|&(s, r, _)| (s, r)).collect::<Vec<_>>();
        let tcuts = basis(n, &[pairs(&x.2), pairs(&y.2)].concat())?;
        let kcuts = basis(kn, &[pairs(&x.3), pairs(&y.3)].concat())?;
        let coefs = |cuts: &[u64], ext: u64, v: &[(u64, u64, u64)]| -> Option<Vec<u64>> {
            let mut acc = vec![0u64; cuts.len() - 1];
            for &(s, r, c) in v {
                for (a, b) in acc.iter_mut().zip(over(cuts, ext, s, r, c)?) {
                    *a = a.checked_add(b)?;
                }
            }
            Some(acc)
        };
        let (xt, yt) = (coefs(&tcuts, n, &x.2)?, coefs(&tcuts, n, &y.2)?);
        let (xk, yk) = (coefs(&kcuts, kn, &x.3)?, coefs(&kcuts, kn, &y.3)?);
        // Every digit of `t` belongs to exactly one operand (no batch axis).
        if xt.iter().zip(&yt).any(|(&p, &q)| (p == 0) == (q == 0)) {
            return None;
        }
        // `b` reads `t`'s unit digit: the lanes run along the output's fast axis.
        let ((a, at, ak), (b, bt, bk), mul_a_first) = if yt[0] != 0 {
            ((&x, &xt, &xk), (&y, &yt, &yk), true)
        } else {
            ((&y, &yt, &yk), (&x, &xt, &xk), false)
        };
        // Axis digits: `t`'s basis digits owned by one operand, renumbered.
        let axis = |own: &[u64]| -> (Vec<(usize, u64, u64)>, u64) {
            let mut stride = 1;
            let mut ds = Vec::new();
            for j in 0..own.len() {
                if own[j] != 0 {
                    let radix = tcuts[j + 1] / tcuts[j];
                    ds.push((j, stride, radix));
                    stride *= radix;
                }
            }
            (ds, stride)
        };
        let (rows_d, rows) = axis(at);
        let (lanes_d, c) = axis(bt);
        let kdig = |cs: &[u64]| -> Vec<AxisDigit> {
            merged(
                cs.iter()
                    .enumerate()
                    .map(|(j, &coef)| AxisDigit {
                        stride: kcuts[j],
                        radix: kcuts[j + 1] / kcuts[j],
                        coef,
                    })
                    .collect(),
            )
        };
        let tdig = |ds: &[(usize, u64, u64)], cs: &[u64]| -> Vec<AxisDigit> {
            merged(
                ds.iter()
                    .map(|&(j, stride, radix)| AxisDigit {
                        stride,
                        radix,
                        coef: cs[j],
                    })
                    .collect(),
            )
        };
        let tstride: Vec<u64> = tcuts[..tcuts.len() - 1].to_vec();
        let addr_a = RadixAddr {
            base: a.1,
            i: tdig(&rows_d, at),
            lane: Vec::new(),
            k: kdig(ak),
        };
        let addr_b = RadixAddr {
            base: b.1,
            i: Vec::new(),
            lane: tdig(&lanes_d, bt),
            k: kdig(bk),
        };
        let out = RadixAddr {
            base: 0,
            i: tdig(&rows_d, &tstride),
            lane: tdig(&lanes_d, &tstride),
            k: Vec::new(),
        };
        // Every partial sum of a form is at most its maximum: inside the array,
        // no intermediate of the program's own index arithmetic wrapped.
        if addr_a.max() >= a.4 || addr_b.max() >= b.4 {
            return None;
        }
        let tile_read = |slot: u32, r: &RadixAddr| -> (TileRead, Option<RadixAddr>) {
            match (plain(&r.i, rows), plain(&r.lane, c), plain(&r.k, kn)) {
                (Some(ci), Some(clane), Some(ck)) => (
                    TileRead {
                        slot,
                        base: r.base,
                        ci,
                        ck,
                        clane,
                        ksplit: None,
                    },
                    None,
                ),
                _ => (
                    TileRead {
                        slot,
                        base: r.base,
                        ci: 0,
                        ck: 0,
                        clane: 0,
                        ksplit: None,
                    },
                    Some(r.clone()),
                ),
            }
        };
        let (ra, ma) = tile_read(a.0, &addr_a);
        let (rb, mb) = tile_read(b.0, &addr_b);
        let row_major = plain(&out.i, rows) == Some(c) && plain(&out.lane, c) == Some(1);
        let radix = RadixTile {
            a: ma,
            b: mb,
            out: (!row_major).then_some(out),
        };
        // All-plain is still a radix site: `tile_plan_strided` refused its
        // several div/mod pairs (`t/64%8`, `t/16%4`), the digits merged.
        Some(TileSite {
            rows,
            c,
            k: kn,
            a: ra,
            b: rb,
            seed,
            elem: ex,
            mul_a_first,
            add_acc_first,
            radix: Some(Box::new(radix)),
            tf32: self.precision(mul.id).admits_tf32(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_chains_digits_and_refuses_overlap() {
        // conv taps: k/9, k%9/3, k%3 over k < 576.
        assert_eq!(basis(576, &[(9, 0), (3, 3), (1, 3)]), Some(vec![1, 3, 9, 576]));
        // t/56%56 and t%56 over t < 3136·8: 56 | 3136 | 25088.
        assert_eq!(basis(25088, &[(56, 56), (1, 56)]), Some(vec![1, 56, 3136, 25088]));
        // k%6 and k/4 overlap without chaining: 4 ∤ 6.
        assert_eq!(basis(24, &[(1, 6), (4, 0)]), None);
    }

    #[test]
    fn digits_re_expressed_over_the_basis_sum_back() {
        let cuts = [1, 3, 9, 576];
        // raw k = 1·(k%3) + 3·(k/3%3) + 9·(k/9)
        assert_eq!(over(&cuts, 576, 1, 0, 1), Some(vec![1, 3, 9]));
        // k%9 over the basis, scaled by 58
        assert_eq!(over(&cuts, 576, 1, 9, 58), Some(vec![58, 174, 0]));
        let m = merged(vec![
            AxisDigit { stride: 1, radix: 3, coef: 1 },
            AxisDigit { stride: 3, radix: 3, coef: 3 },
            AxisDigit { stride: 9, radix: 64, coef: 9 },
        ]);
        assert_eq!(m, vec![AxisDigit { stride: 1, radix: 576, coef: 1 }]);
        assert_eq!(plain(&m, 576), Some(1));
    }
}
