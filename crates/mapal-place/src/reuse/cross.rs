//! Lockstep crossings of two readers' strided walks over groups.

use super::*;

/// Stamped flag sets over group ids (reset in O(touched)).
pub(super) struct Flags {
    pub(super) f: Vec<u8>,
    pub(super) touched: Vec<usize>,
}

impl Flags {
    pub(super) fn new(n: usize) -> Self {
        Flags {
            f: vec![0; n],
            touched: Vec::new(),
        }
    }
    pub(super) fn set(&mut self, i: usize, bit: u8) {
        if self.f[i] == 0 {
            self.touched.push(i);
        }
        self.f[i] |= bit;
    }
    pub(super) fn clear(&mut self) {
        for &i in &self.touched {
            self.f[i] = 0;
        }
        self.touched.clear();
    }
}

pub(super) const IN_M: u8 = 1;
pub(super) const IN_1: u8 = 2;
pub(super) const IN_2: u8 = 4;
pub(super) const POINT: u8 = 8;

#[derive(Clone, Copy)]
pub(super) struct Blk {
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) o: usize,
}

/// The cross-wave distance of a slice of family `fam` (0 `a`, 1 `b`) of
/// group `own`, last read at position `p1` of wave `w1` and next at `p2` of
/// wave `w2`, as `(α, β)`: the distinct bytes of other slices between the
/// two reads at panel step `σ` are `α + β·σ`.
#[allow(clippy::too_many_arguments)]
pub(super) fn cross(
    waves: &[&[Blk]],
    fam: u8,
    own: usize,
    (w1, p1): (usize, usize),
    (w2, p2): (usize, usize),
    pb: u64,
    (sa, sb, so): (u64, u64, u64),
    fl: &mut [Flags; 3],
) -> (i128, i128) {
    let [fa, fb, fo] = fl;
    for w in &waves[w1 + 1..w2] {
        for x in w.iter() {
            fa.set(x.a, IN_M);
            fb.set(x.b, IN_M);
            fo.set(x.o, IN_M);
        }
    }
    for (q, x) in waves[w1].iter().enumerate() {
        fa.set(x.a, IN_1 | if q > p1 { POINT } else { 0 });
        let tail_b = if fam == 0 { q >= p1 } else { q > p1 };
        fb.set(x.b, IN_1 | if tail_b { POINT } else { 0 });
        fo.set(x.o, IN_M);
    }
    for (q, x) in waves[w2].iter().enumerate() {
        let head_a = if fam == 0 { q < p2 } else { q <= p2 };
        fa.set(x.a, IN_2 | if head_a { POINT } else { 0 });
        fb.set(x.b, IN_2 | if q < p2 { POINT } else { 0 });
    }
    let pb = pb as i128;
    let sum = |f: &Flags, skip: Option<usize>| -> (i128, i128) {
        let (mut al, mut be) = (0i128, 0i128);
        for &i in &f.touched {
            if Some(i) == skip {
                // The slice's own other panels: every one but itself.
                al += pb - 1;
                continue;
            }
            let x = f.f[i];
            if x & IN_M != 0 {
                al += pb;
                continue;
            }
            if x & IN_1 != 0 {
                al += pb - 1;
                be -= 1;
            }
            if x & IN_2 != 0 {
                be += 1;
            }
            if x & POINT != 0 {
                al += 1;
            }
        }
        (al, be)
    };
    let (aa, ba) = sum(fa, (fam == 0).then_some(own));
    let (ab, bb) = sum(fb, (fam == 1).then_some(own));
    let outs = fo.touched.len() as i128;
    let alpha = aa * sa as i128 + ab * sb as i128 + outs * so as i128;
    let beta = ba * sa as i128 + bb * sb as i128;
    for f in fl.iter_mut() {
        f.clear();
    }
    (alpha, beta)
}

/// **Lockstep waves, exact.** Tiles `first .. first + count` of the raster
/// order, `s` partitions a tile (partition-major launch), waves of `slots`.
pub fn lockstep(
    g: &Grid,
    h: u64,
    s: u64,
    atomic: bool,
    first: u64,
    count: u64,
    keep: u64,
) -> Reads {
    let s = s.max(1);
    let pb = g.panels / s;
    let w = g.slots.max(1) as usize;
    let (sa, sb, so) = (g.sa, g.sb, g.so);
    let tiles = g.gx * g.gy;
    let unique_out = !atomic || s <= 1;
    let blocks: Vec<Blk> = (0..count * s)
        .map(|l| {
            let (z, t) = (l / count.max(1), first + l % count.max(1));
            let (r, j) = launch_tile(g.gx, g.gy, h, t);
            let tile = r * g.gx + j;
            Blk {
                a: (r * s + z) as usize,
                b: (j * s + z) as usize,
                o: if unique_out && s > 1 {
                    (tile * s + z) as usize
                } else {
                    tile as usize
                },
            }
        })
        .collect();
    let waves: Vec<&[Blk]> = blocks.chunks(w).collect();
    let (na, nb) = ((g.gy * s) as usize, (g.gx * s) as usize);
    let no = (tiles * s) as usize;
    let mut fl = [Flags::new(na), Flags::new(nb), Flags::new(no)];
    let mut miss = Reads::default();
    let keep = keep as i128;
    // Operand families: the last (wave, position) each group was read at.
    for fam in 0u8..2 {
        let size = if fam == 0 { sa } else { sb } as i128;
        let n = if fam == 0 { na } else { nb };
        let mut last: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut seen = Flags::new(n.max(na).max(nb));
        let mut acc = 0u64;
        for (wi, wave) in waves.iter().enumerate() {
            for (q, x) in wave.iter().enumerate() {
                let gid = if fam == 0 { x.a } else { x.b };
                match last[gid] {
                    None => acc += pb * size as u64,
                    Some((w1, p1)) if w1 == wi => {
                        // Same wave, same step: the distinct slices of the
                        // blocks between (each one slice of each family).
                        let (lo_a, hi_a, lo_b, hi_b) = match fam {
                            0 => (p1 + 1, q, p1, q),
                            _ => (p1 + 1, q + 1, p1 + 1, q),
                        };
                        for y in &wave[lo_a..hi_a] {
                            seen.set(y.a, 1);
                        }
                        let da = seen.touched.len() as i128;
                        seen.clear();
                        for y in &wave[lo_b..hi_b] {
                            seen.set(y.b, 1);
                        }
                        let db = seen.touched.len() as i128;
                        seen.clear();
                        if da * sa as i128 + db * sb as i128 + size > keep {
                            acc += pb * size as u64;
                        }
                    }
                    Some((w1, p1)) => {
                        let (al, be) = cross(
                            &waves,
                            fam,
                            gid,
                            (w1, p1),
                            (wi, q),
                            pb,
                            (sa, sb, so),
                            &mut fl,
                        );
                        // Misses: σ with α + β·σ + size > keep.
                        acc += count_gt(be, keep - size - al, pb) * size as u64;
                    }
                }
                last[gid] = Some((wi, q));
            }
        }
        if fam == 0 {
            miss.a = acc;
        } else {
            miss.b = acc;
        }
    }
    // Outputs: the zero pass (every tile, in order), then each wave's
    // stores or partial adds after its steps.
    let mut groups_upto = vec![(0i128, 0i128); waves.len()];
    {
        let (mut sa_seen, mut sb_seen) = (Flags::new(na), Flags::new(nb));
        for (wi, wave) in waves.iter().enumerate() {
            for x in wave.iter() {
                sa_seen.set(x.a, 1);
                sb_seen.set(x.b, 1);
            }
            groups_upto[wi] = (sa_seen.touched.len() as i128, sb_seen.touched.len() as i128);
        }
    }
    let mut last: Vec<Option<(usize, usize)>> = vec![None; no];
    let mut red_before = Flags::new(no);
    let mut acc = 0u64;
    let pbi = pb as i128;
    let so_i = so as i128;
    if g.zero {
        acc += tiles * so;
    }
    for (wi, wave) in waves.iter().enumerate() {
        for (q, x) in wave.iter().enumerate() {
            let o = x.o;
            let d = match last[o] {
                None if !g.zero || o as u64 >= tiles => None,
                None => {
                    // From the zero pass: the tiles zeroed after it, the
                    // outputs added before it, every operand slice so far.
                    let after = (tiles - 1 - o as u64) as i128;
                    let before = red_before.touched.iter().filter(|&&i| i < o).count() as i128;
                    let (ga, gb) = groups_upto[wi];
                    Some((after + before) * so_i + ga * pbi * sa as i128 + gb * pbi * sb as i128)
                }
                Some((w1, p1)) if w1 == wi => {
                    let mut f = Flags::new(0);
                    std::mem::swap(&mut f, &mut fl[2]);
                    for y in &wave[p1 + 1..q] {
                        f.set(y.o, 1);
                    }
                    let n = f.touched.len() as i128;
                    f.clear();
                    std::mem::swap(&mut f, &mut fl[2]);
                    Some(n * so_i)
                }
                Some((w1, p1)) => {
                    let [fa, fb, fo] = &mut fl;
                    for y in &waves[w1][p1 + 1..] {
                        fo.set(y.o, 1);
                    }
                    for w in &waves[w1 + 1..wi] {
                        for y in w.iter() {
                            fo.set(y.o, 1);
                        }
                    }
                    for y in &waves[wi][..q] {
                        fo.set(y.o, 1);
                    }
                    for w in &waves[w1 + 1..=wi] {
                        for y in w.iter() {
                            fa.set(y.a, 1);
                            fb.set(y.b, 1);
                        }
                    }
                    let n_o = fo.touched.iter().filter(|&&i| i != o).count() as i128;
                    let d = n_o * so_i
                        + fa.touched.len() as i128 * pbi * sa as i128
                        + fb.touched.len() as i128 * pbi * sb as i128;
                    for f in fl.iter_mut() {
                        f.clear();
                    }
                    Some(d)
                }
            };
            if d.is_none_or(|d| d + so_i > keep) {
                acc += so;
            }
            last[o] = Some((wi, q));
            red_before.set(o, 1);
        }
    }
    miss.out = acc;
    miss
}
