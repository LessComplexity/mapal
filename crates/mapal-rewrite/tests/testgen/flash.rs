//! Mapal-source generators: flash attention in every idiomatic form.

use super::*;

// ---------------------------------------------------------------------------
// flash-attention `stream`: attention written idiomatically, every form the
// stream fact admits or refuses.
// ---------------------------------------------------------------------------

/// **flash-attention generated programs** (`crates/backends/llvm/tests/
/// flash_attention.rs`): `n` query rows of width `d` against `w` keys —
/// the scores `q·Kᵀ` (a tile site), their row extreme (a running max, or a
/// min with the exponent turned around: the rescale down), `exp` of the
/// shifted (scaled) score, its row sum, the probabilities (`÷ den`, a
/// quotient factor) and `P·V` over `c` value columns (a tile site folding the
/// keys). Key counts include ones no window divides (tails); head widths odd
/// ones; the extreme's seed far below the scores, inside them (the seed wins
/// a row: the final rescale), or the identity's neighbour; the sum's seed `0`
/// or `1`. Forms the fact refuses: the `÷ den` after `P·V` (a per-row cell
/// after the windows), a stored probability (`escape`: its final shift), an
/// exponent the extreme does not shift (`phase`).
#[derive(Clone, Debug)]
pub struct FlashProg {
    pub n: u64,
    pub w: u64,
    pub d: u64,
    pub c: u64,
    /// 0 `(s − m) ÷ 2`, 1 `s − m`, 2 `(s − m) × 0.5`.
    pub scale: u8,
    /// The extreme's seed: 0 `−1e9`, 1 `0.0` (inside the scores), 2 `−1e20`.
    pub seed: u8,
    /// A running min and `exp((m − s) ÷ 2)` (the rescale down).
    pub down: bool,
    /// The sum's seed is `1.0` (else `0.0`).
    pub den_one: bool,
    /// 0 `P·V` over `p = ex ÷ den`; 1 `P·V` over `ex`, then `÷ den` a cell
    /// (refused: per-row work after the windows); 2 `p` also printed
    /// (refused: `escape`); 3 `exp(s · m)` (refused: `phase`).
    pub form: u8,
    /// The scores, the extreme and the sum also printed (stored members).
    pub stored: bool,
    /// The model-scale form ([`flash_mh_src`]): `n` tokens, `heads` query
    /// heads of width `d` in groups of `group` sharing a KV head; `0`: the
    /// single-head form above.
    pub heads: u64,
    pub group: u64,
    /// Model-scale: the scores masked causally (`r ≤ i`) to `mask`.
    pub causal: bool,
    /// The mask's value: 0 `−1e9`, 1 `−8` (inside the scores' reach of
    /// `exp`: a masked window adds a term no stand-in absorbs).
    pub mask: u8,
    /// sweep-attention `unit-uniform`: the model-scale form inside a loop
    /// over the KV heads ([`flash_loop_src`]) — each trip one KV head's
    /// query heads, every operand address moved by the loop's counter (a
    /// captured number), the outputs summed into a carried array.
    pub lp: bool,
    /// sweep-attention-2 `stream-mma`: the model-scale forms' two products
    /// admit the `bf16` member (`fold: bf16 | f32`), and more outputs are
    /// printed (a spread of `P·V`'s cells, each within the member's bound).
    pub half: bool,
    /// mask-frontend: the model-scale form written token-major
    /// ([`flash_tm_src`]: the scores' rows `token·H + head`, a readable
    /// model's layout) with its causal mask `r ≤ …` against `0`: none; `1`
    /// the token `i`; `2` `pos[i]`, `pos = map(iota) (x → first + x)` with
    /// `first` a literal (the law through the capture); `3` the same `pos`
    /// with `first` known only at run time (a per-row load); `4` `first + i`
    /// with that run-time `first` (a per-launch number).
    pub tm: u8,
}

pub fn flash_strategy() -> impl Strategy<Value = FlashProg> {
    (
        (
            prop::sample::select(vec![1u64, 2, 3, 4, 8, 16, 32, 64]),
            prop::sample::select(vec![4u64, 16, 24, 64, 100, 128, 256]),
            prop::sample::select(vec![3u64, 4, 8, 16]),
            // attention-chains: a value width of 32 — two 16-byte chunks a
            // lane of a 4-lane row group (the interleaved consumer columns).
            prop::sample::select(vec![3u64, 4, 8, 16, 32]),
        ),
        (0u8..3, 0u8..3, any::<bool>(), any::<bool>()),
        (
            prop::sample::select(vec![0u8, 0, 0, 0, 1, 2, 3]),
            any::<bool>(),
        ),
        (
            prop::sample::select(vec![0u64, 0, 2, 4, 8]),
            prop::sample::select(vec![1u64, 2, 4]),
            any::<bool>(),
            0u8..2,
            any::<bool>(),
            prop::sample::select(vec![0u8, 0, 0, 1, 2, 3, 4]),
        ),
    )
        .prop_map(
            |(
                (n, w, d, c),
                (scale, seed, down, den_one),
                (form, stored),
                (heads, group, causal, mask, lp, tm),
            )| FlashProg {
                // The model-scale form at sizes its sites run on the device.
                n: if heads > 0 { n.clamp(16, 64) } else { n },
                w,
                d: if heads > 0 { d.max(8) } else { d },
                c,
                scale,
                seed,
                down,
                den_one,
                form,
                stored,
                heads,
                group: group.min(heads.max(1)),
                causal,
                mask,
                lp: lp && heads > 0 && tm == 0,
                half: false,
                tm: if heads > 0 { tm } else { 0 },
            },
        )
}

/// sweep-attention-2 `stream-mma`: [`FlashProg`]'s model-scale forms whose
/// products admit `bf16` — 16–64 tokens (whole 16-row warps; 48 leaves a
/// 32-key window a tail), heads of width 16 or 32 in groups of 1–4, causal
/// or not (the mask `−1e9` / `−8`), the head-group loop or not, head-major or
/// token-major (each of [`FlashProg::tm`]'s mask forms). The data are
/// small multiples of `¼`, exact in `bf16`: the scores are exact on the
/// tensor core, and the probabilities' rounding (`2⁻⁹` relative) is the
/// member's whole error.
pub fn flash_half_strategy() -> impl Strategy<Value = FlashProg> {
    (
        prop::sample::select(vec![16u64, 32, 48, 64]),
        prop::sample::select(vec![16u64, 32]),
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![1u64, 2, 4]),
        any::<bool>(),
        0u8..2,
        any::<bool>(),
        // prefill-integrate: the token-major forms (`tm`, a readable
        // model's `attend`) on the matrix product too.
        prop::sample::select(vec![0u8, 0, 1, 2, 3, 4]),
    )
        .prop_map(|(n, d, heads, group, causal, mask, lp, tm)| FlashProg {
            n,
            w: n,
            d,
            c: d,
            scale: 0,
            seed: 0,
            down: false,
            den_one: false,
            form: 0,
            stored: false,
            heads,
            group: group.min(heads),
            causal,
            mask,
            lp: lp && tm == 0,
            half: true,
            tm,
        })
}

/// sweep-attention-2 `kmajor-b`: a decode step's masked scores — `B`
/// sequences of `H` query heads in groups of `G` sharing a KV head, each
/// row's `L` keys from a cache `K[b][r][H ÷ G][D]` (k-major: a key's `D`
/// elements contiguous), stored `f32` or `bf16`, the keys past a position
/// read at run time masked to `−1e9`, the dot scaled by `½`. Data are
/// small multiples of `¼`: every sum is exact in any order.
#[derive(Clone, Debug)]
pub struct KdecProg {
    pub b: u64,
    pub h: u64,
    pub g: u64,
    pub l: u64,
    pub d: u64,
    pub narrow: bool,
    /// The position: keys `r ≤ p` are read.
    pub p: u64,
}

pub fn kdec_strategy() -> impl Strategy<Value = KdecProg> {
    (
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![4u64, 8]),
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![64u64, 128, 256, 512]),
        prop::sample::select(vec![16u64, 32, 64, 128]),
        prop::bool::weighted(0.6),
        0u64..1000,
    )
        .prop_map(|(b, h, g, l, d, narrow, p)| KdecProg {
            b,
            h,
            g: g.min(h),
            l,
            d,
            narrow,
            p: p % l,
        })
}

/// [`KdecProg`]'s source text: the scores and eight of them printed.
pub fn kdec_src(p: &KdecProg) -> String {
    let (b, h, g, l, d) = (p.b, p.h, p.g, p.l, p.d);
    let kvh = h / g;
    let (nb, w) = match p.narrow {
        true => (" -> to<bf16>", " -> widen<f32>"),
        false => ("", ""),
    };
    let mut s = String::from("fn main() {\n");
    s += &format!(
        "    {} -> iota -> i_kc;\n    i_kc -> map {{\n        t -> (t * 5 + 3) % 9 - 4 -> widen<f32> -> v;\n        v / 4.0{nb}\n    }} -> kc;\n",
        b * l * kvh * d
    );
    s += &format!(
        "    {} -> iota -> i_q;\n    i_q -> map {{\n        t -> (t * 7 + 13) % 11 - 5 -> widen<f32> -> v;\n        v / 4.0\n    }} -> q;\n",
        b * h * d
    );
    s += &format!(
        "    {d} -> iota -> rdh;\n    {} -> iota -> rBHL;\n    1 -> iota -> i_p;\n    i_p -> map {{\n        t -> t + {}\n    }} -> pos;\n    pos[0] -> p;\n",
        b * h * l,
        p.p
    );
    s += &format!(
        "    rBHL -> map {{\n        e -> e / {} -> b;\n        e / {l} % {h} -> h;\n        e % {l} -> r;\n        (r <= p) -> {{\n            -true-> {{\n                (0.0, rdh) -> fold {{\n                    acc, i -> kc[((b * {l} + r) * {kvh} + h / {g}) * {d} + i]{w} -> kv;\n                    acc + q[b * {} + h * {d} + i] * kv\n                }} -> dt;\n                dt / 2.0\n            }}\n            -false-> -1000000000.0;\n        }}\n    }} -> s;\n",
        h * l,
        h * d
    );
    let n = b * h * l;
    for j in 1..=8u64 {
        s += &format!("    s[{}] -> println;\n", (j * n / 9 + j * 7) % n);
    }
    s + "}\n"
}

/// [`FlashProg::half`]'s fold annotation (`: bf16 | f32`), else none.
pub(super) fn flash_member(p: &FlashProg) -> &'static str {
    if p.half { ": bf16 | f32" } else { "" }
}

/// [`FlashProg::half`]: eight more cells of `name` (of `len`) printed,
/// spread over it.
pub(super) fn flash_spread(p: &FlashProg, name: &str, len: u64) -> String {
    if !p.half {
        return String::new();
    }
    (1..=8u64)
        .map(|j| format!("    {name}[{}] -> println;\n", (j * len / 9 + j * 7) % len))
        .collect()
}

/// [`FlashProg`]'s model-scale source (`heads > 0`): `T = n` tokens, the
/// scores `[head][token][token]` (head-major rows), their row max, `exp` of
/// the shifted scaled score, its row sum, and `P·V` written token-major
/// (`[token][head·d + u]`, the digit permutation of the scores' rows) with
/// the `÷ den` after it — attention as a transformer layer writes it.
pub fn flash_mh_src(p: &FlashProg) -> String {
    let (t, h, g, d) = (p.n, p.heads, p.group, p.d);
    let kvh = h / g;
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen<f32> -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", t * h * d, 7, 13, 11);
    s += &data("kr", t * kvh * d, 5, 3, 9);
    s += &data("vv", t * kvh * d, 3, 1, 13);
    s += &format!(
        "    {d} -> iota -> rdh;\n    {t} -> iota -> rT;\n    {} -> iota -> rHT;\n    {} -> iota -> rHTT;\n    {} -> iota -> rRD;\n",
        h * t,
        h * t * t,
        t * h * d
    );
    let fold = format!(
        "(0.0, rdh) -> fold{} {{\n                acc, u -> acc + q[i * {} + hh * {d} + u] * kr[r * {} + (hh / {g}) * {d} + u]\n            }}",
        flash_member(p),
        h * d,
        kvh * d
    );
    let mask = ["-1000000000.0", "-8.0"][p.mask as usize];
    let body = match p.causal {
        true => format!(
            "(r <= i) -> {{\n            -true-> {{\n                {fold} -> dt;\n                dt / 2.0\n            }}\n            -false-> {mask};\n        }}"
        ),
        false => format!("{fold} -> dt;\n        dt / 2.0"),
    };
    s += &format!(
        "    rHTT -> map {{\n        e -> e / {} -> hh;\n        e / {t} % {t} -> i;\n        e % {t} -> r;\n        {body}\n    }} -> s;\n",
        t * t
    );
    s += &format!(
        "    rHT -> map {{\n        hi -> (-1000000000.0, rT) -> fold {{\n            acc, r -> s[hi * {t} + r] -> se;\n            (se > acc) -> {{\n                -true-> se;\n                -false-> acc;\n            }}\n        }}\n    }} -> mx;\n"
    );
    s += &format!("    rHTT -> map {{\n        e -> s[e] - mx[e / {t}] -> exp\n    }} -> pe;\n");
    s += &format!(
        "    rHT -> map {{\n        hi -> (0.0, rT) -> fold {{\n            acc, r -> acc + pe[hi * {t} + r]\n        }}\n    }} -> den;\n"
    );
    s += &format!(
        "    rRD -> map {{\n        e -> e / {} -> i;\n        e % {} -> c;\n        c / {d} -> hh;\n        c % {d} -> u;\n        (0.0, rT) -> fold{} {{\n            acc, r -> acc + pe[(hh * {t} + i) * {t} + r] * vv[r * {} + (hh / {g}) * {d} + u]\n        }} -> av;\n        av / den[hh * {t} + i]\n    }} -> attf;\n",
        h * d,
        h * d,
        flash_member(p),
        kvh * d
    );
    s += &format!(
        "    attf[0] -> println;\n    attf[{}] -> println;\n{}    0.0 -> zero: f32;\n    (zero, attf) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        t * h * d - 1,
        flash_spread(p, "attf", t * h * d)
    );
    if p.stored {
        s += &format!("    den[{}] -> println;\n", h * t - 1);
    }
    s + "}\n"
}

/// mask-frontend: [`FlashProg`]'s model-scale form token-major (`tm > 0`,
/// a readable model's `attend`): `T = n` tokens, the scores
/// `[token·H + head][key]`, their row max, `exp` of the shifted score, its
/// row sum and `P·V ÷ den` token-major (`[token·H + head][u]`) — the
/// scores' tile rows (KV head most significant) a digit permutation of its
/// output rows. The causal mask is [`FlashProg::tm`]'s form; every form
/// is `r ≤ i` at run time (`first` is `0`). Where `scale == 1` (unused by
/// the model-scale forms) the keys and values are read through a captured
/// struct (`cache.keys[…]`: a readable model's cache).
pub fn flash_tm_src(p: &FlashProg) -> String {
    let (t, h, g, d) = (p.n, p.heads, p.group, p.d);
    let kvh = h / g;
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen<f32> -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let via = p.scale == 1;
    let mut s = match via {
        true => format!(
            "type Cache {{\n    keys: [f32; {n}],\n    values: [f32; {n}],\n}}\n\nfn main() {{\n",
            n = t * kvh * d
        ),
        false => String::from("fn main() {\n"),
    };
    s += &data("q", t * h * d, 7, 13, 11);
    s += &data("kr", t * kvh * d, 5, 3, 9);
    s += &data("vv", t * kvh * d, 3, 1, 13);
    if via {
        s += "    Cache {\n        keys: kr,\n        values: vv,\n    } -> cache;\n";
    }
    let (keys, values) = match via {
        true => ("cache.keys", "cache.values"),
        false => ("kr", "vv"),
    };
    s += &format!(
        "    {d} -> iota -> rdh;\n    {t} -> iota -> rT;\n    {} -> iota -> rTH;\n    {} -> iota -> rTHT;\n    {} -> iota -> rRD;\n",
        t * h,
        t * h * t,
        t * h * d
    );
    // `first`: a literal, or a value only the run knows (`0` there).
    match p.tm {
        2 => s += "    0 -> first;\n",
        3 | 4 => {
            s += "    () -> time -> clock;\n    (clock < 0.0) -> {\n        -true-> 1;\n        -false-> 0;\n    } -> first;\n"
        }
        _ => {}
    }
    if matches!(p.tm, 2 | 3) {
        s += "    rT -> map {\n        x -> first + x\n    } -> pos;\n";
    }
    let limit = match p.tm {
        1 => "i",
        2 | 3 => "pos[i]",
        _ => "first + i",
    };
    let fold = format!(
        "(0.0, rdh) -> fold{} {{\n                    acc, u -> acc + q[i * {} + hh * {d} + u] * {keys}[r * {} + (hh / {g}) * {d} + u]\n                }}",
        flash_member(p),
        h * d,
        kvh * d
    );
    let mask = ["-1000000000.0", "-8.0"][p.mask as usize];
    s += &format!(
        "    rTHT -> map {{\n        e -> e / {t} -> th;\n        e % {t} -> r;\n        th / {h} -> i;\n        th % {h} -> hh;\n        (r <= {limit}) -> {{\n            -true-> {{\n                {fold} -> dt;\n                dt / 2.0\n            }}\n            -false-> {mask};\n        }}\n    }} -> s;\n"
    );
    s += &format!(
        "    rTH -> map {{\n        hi -> (-1000000000.0, rT) -> fold {{\n            acc, r -> s[hi * {t} + r] -> se;\n            (se > acc) -> {{\n                -true-> se;\n                -false-> acc;\n            }}\n        }}\n    }} -> mx;\n"
    );
    s += &format!("    rTHT -> map {{\n        e -> s[e] - mx[e / {t}] -> exp\n    }} -> pe;\n");
    s += &format!(
        "    rTH -> map {{\n        hi -> (0.0, rT) -> fold {{\n            acc, r -> acc + pe[hi * {t} + r]\n        }}\n    }} -> den;\n"
    );
    s += &format!(
        "    rRD -> map {{\n        e -> e / {d} -> th;\n        e % {d} -> u;\n        th % {h} -> hh;\n        (0.0, rT) -> fold{} {{\n            acc, r -> acc + pe[th * {t} + r] * {values}[r * {} + (hh / {g}) * {d} + u]\n        }} -> av;\n        av / den[th]\n    }} -> attf;\n",
        flash_member(p),
        kvh * d
    );
    s += &format!(
        "    attf[0] -> println;\n    attf[{}] -> println;\n    attf[{}] -> println;\n{}    0.0 -> zero: f32;\n    (zero, attf) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        t * h * d - 1,
        (t * h * d) / 2 + d + 1,
        flash_spread(p, "attf", t * h * d)
    );
    if p.stored {
        s += &format!("    den[{}] -> println;\n", h * t - 1);
    }
    s + "}\n"
}

/// [`FlashProg`]'s head-group loop form (`lp`, the long prefill's shape):
/// a loop over the `H ÷ G` KV heads, each trip the scores of KV head `g`'s
/// `G` query heads `[G][T][T]` (`q` read at head `g·G + hh`, the keys and
/// values at KV head `g`: addresses a captured counter moves), their row
/// max, `exp`, row sum and `P·V ÷ den` token-major (`[T][G·d]`), summed
/// into a carried array — the same attention per trip as
/// [`flash_mh_src`]'s heads, one KV head at a time.
pub fn flash_loop_src(p: &FlashProg) -> String {
    let (t, h, g, d) = (p.n, p.heads, p.group, p.d);
    let kvh = h / g;
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen<f32> -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", t * h * d, 7, 13, 11);
    s += &data("kr", t * kvh * d, 5, 3, 9);
    s += &data("vv", t * kvh * d, 3, 1, 13);
    s += &format!(
        "    {d} -> iota -> rdh;\n    {t} -> iota -> rT;\n    {} -> iota -> rHT;\n    {} -> iota -> rHTT;\n    {} -> iota -> rTG;\n",
        g * t,
        g * t * t,
        t * g * d
    );
    s += &format!(
        "    (0.0, {}) -> fill -> y0;\n    mut y: [f32; {}] <- y0;\n    mut g: i32 <- 0;\n",
        t * g * d,
        t * g * d
    );
    let fold = format!(
        "(0.0, rdh) -> fold{} {{\n                                acc, u -> acc + q[i * {} + (g * {g} + hh) * {d} + u] * kr[r * {} + g * {d} + u]\n                            }}",
        flash_member(p),
        h * d,
        kvh * d
    );
    let mask = ["-1000000000.0", "-8.0"][p.mask as usize];
    let body = match p.causal {
        true => format!(
            "(r <= i) -> {{\n                            -true-> {{\n                                {fold} -> dt;\n                                dt / 2.0\n                            }}\n                            -false-> {mask};\n                        }}"
        ),
        false => format!("{fold} -> dt;\n                        dt / 2.0"),
    };
    let mut l = String::new();
    l += &format!(
        "                rHTT -> map {{\n                    e -> e / {} -> hh;\n                    e / {t} % {t} -> i;\n                    e % {t} -> r;\n                    {body}\n                }} -> s;\n",
        t * t
    );
    l += &format!(
        "                rHT -> map {{\n                    hi -> (-1000000000.0, rT) -> fold {{\n                        acc, r -> s[hi * {t} + r] -> se;\n                        (se > acc) -> {{\n                            -true-> se;\n                            -false-> acc;\n                        }}\n                    }}\n                }} -> mx;\n"
    );
    l += &format!(
        "                rHTT -> map {{\n                    e -> s[e] - mx[e / {t}] -> exp\n                }} -> pe;\n"
    );
    l += &format!(
        "                rHT -> map {{\n                    hi -> (0.0, rT) -> fold {{\n                        acc, r -> acc + pe[hi * {t} + r]\n                    }}\n                }} -> den;\n"
    );
    l += &format!(
        "                rTG -> map {{\n                    e -> e / {} -> i;\n                    e % {} -> c;\n                    c / {d} -> hh;\n                    c % {d} -> u;\n                    (0.0, rT) -> fold{} {{\n                        acc, r -> acc + pe[(hh * {t} + i) * {t} + r] * vv[r * {} + g * {d} + u]\n                    }} -> av;\n                    av / den[hh * {t} + i]\n                }} -> attg;\n",
        g * d,
        g * d,
        flash_member(p),
        kvh * d
    );
    l += "                rTG -> map {\n                    e -> y[e] + attg[e]\n                } -> y;\n";
    s += &format!(
        "    loop {{\n        (g < {kvh}) -> {{\n            -true-> {{\n{l}                g + 1 -> g;\n                -> loop;\n            }}\n            -false-> y -> out;\n        }}\n    }}\n"
    );
    s += &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n{}    0.0 -> zero: f32;\n    (zero, out) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        t * g * d - 1,
        flash_spread(p, "out", t * g * d)
    );
    s + "}\n"
}

/// [`FlashProg`]'s source text. Data are small integers over 4, the scores
/// a few units wide, so `exp` stays well inside `f32`.
pub fn flash_src(p: &FlashProg) -> String {
    if p.tm > 0 && p.heads > 0 {
        return flash_tm_src(p);
    }
    if p.lp {
        return flash_loop_src(p);
    }
    if p.heads > 0 {
        return flash_mh_src(p);
    }
    let (n, w, d, c) = (p.n, p.w, p.d, p.c);
    let data = |name: &str, len: u64, a: u64, b: u64, m: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{\n        t -> (t * {a} + {b}) % {m} - {} -> widen<f32> -> v;\n        v / 4.0\n    }} -> {name};\n",
            m / 2
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", n * d, 7, 13, 11);
    s += &data("kk", w * d, 5, 3, 9);
    s += &data("vv", w * c, 3, 1, 13);
    s += &format!(
        "    {} -> iota -> o_s;\n    {d} -> iota -> k_d;\n    {w} -> iota -> k_t;\n    {n} -> iota -> k_n;\n    {} -> iota -> o_o;\n",
        n * w,
        n * c
    );
    s += &format!(
        "    o_s -> map {{\n        e -> e / {w} -> i;\n        e % {w} -> j;\n        (0.0, k_d) -> fold {{\n            acc, k -> acc + q[i * {d} + k] * kk[j * {d} + k]\n        }}\n    }} -> s;\n"
    );
    let seed = ["-1000000000.0", "0.0", "-100000000000000000000.0"][p.seed as usize];
    let cmp = if p.down { "<" } else { ">" };
    s += &format!(
        "    k_n -> map {{\n        i -> ({seed}, k_t) -> fold {{\n            acc, j -> s[i * {w} + j] -> e;\n            (e {cmp} acc) -> {{\n                -true-> e;\n                -false-> acc;\n            }}\n        }}\n    }} -> mx;\n"
    );
    let diff = if p.down {
        format!("mx[t / {w}] - s[t]")
    } else {
        format!("s[t] - mx[t / {w}]")
    };
    let arg = match (p.form, p.scale) {
        (3, _) => format!("s[t] * mx[t / {w}] / 64.0"),
        (_, 0) => format!("({diff}) / 2.0"),
        (_, 1) => diff,
        _ => format!("({diff}) * 0.5"),
    };
    s += &format!("    o_s -> map {{\n        t -> {arg} -> exp\n    }} -> ex;\n");
    let dseed = if p.den_one { "1.0" } else { "0.0" };
    s += &format!(
        "    k_n -> map {{\n        i -> ({dseed}, k_t) -> fold {{\n            acc, j -> acc + ex[i * {w} + j]\n        }}\n    }} -> den;\n"
    );
    let pv = |src: &str| {
        format!(
            "(0.0, k_t) -> fold {{\n            acc, u -> acc + {src}[i * {w} + u] * vv[u * {c} + j]\n        }}"
        )
    };
    match p.form {
        1 => {
            s += &format!(
                "    o_o -> map {{\n        e -> e / {c} -> i;\n        e % {c} -> j;\n        {}\n    }} -> av;\n    o_o -> map {{\n        e -> av[e] / den[e / {c}]\n    }} -> o;\n",
                pv("ex")
            );
        }
        _ => {
            s += &format!("    o_s -> map {{\n        t -> ex[t] / den[t / {w}]\n    }} -> p;\n");
            s += &format!(
                "    o_o -> map {{\n        e -> e / {c} -> i;\n        e % {c} -> j;\n        {}\n    }} -> o;\n",
                pv("p")
            );
        }
    }
    s += &format!(
        "    o[0] -> println;\n    o[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, o) -> fold {{\n        acc, v -> acc + v\n    }} -> so;\n    so -> println;\n",
        n * c - 1
    );
    if p.stored {
        s += &format!(
            "    s[{}] -> println;\n    mx[{}] -> println;\n    den[0] -> println;\n",
            (n * w) / 2,
            n - 1
        );
    }
    if p.form == 2 {
        s += &format!("    p[{}] -> println;\n", n * w - 1);
    }
    s + "}\n"
}
