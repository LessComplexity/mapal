"""ADR-0040 D2's whole-program gate for the default face, u = 2⁻²⁴, γ_n = n·u/(1 − n·u).

With the program's magnitude oracle (`<stem>_xabs.expect`, gen_tiles.py `xabs`: every fold step's term
replaced by its magnitude, so each printed value's M = |seed| + Σ|terms| through the whole fold chain):
D2 itself, absolute — |got − exact| ≤ (γ_{K+1}(2⁻²⁴) + γ_{K+1}(2⁻⁵³))·M. Without one: every printed value
within relative 2·γ_{K+1} of the exact face's (which fails a correct result under cancellation, S64
`pt_rowsum_2048`: |exact| ≪ M). Non-numeric tokens equal.

A program whose products admit TF32 (`: tf32 | f32`, ADR-0041) passes `--tf32`: D3's coefficient
(1 + u_t)²·(1 + γ_{K+1}(2⁻²³)) − 1, u_t = 2⁻¹¹, replaces γ_{K+1}(2⁻²⁴); a 16-bit member (ADR-0043 D4:
`: bf16 | f32`, `: f16 | f32`) passes `--bf16` (u_t = 2⁻⁸) / `--f16` (2⁻¹¹).

A program whose products admit a reduced member answers to its composed bound instead (`--bound <xbound.expect>`:
`mapal run --interp --bound`, ADR-0041 D5 `bound(o)` — each printed value's bound, every operation's own default-face
bound composed through the program's derivatives): |got − exact| ≤ bound per token. The one-site coefficient above
counts one product's two roundings for the whole program, not a chain of admitting products (bound-blkdec:
`px_llm_blkdec_b2_l16_bf16`, every realized rounding admitted, 2.3·10⁻² relative against 1.6·10⁻²).

usage: python3 value_gate.py <exact.expect> "<got, one line>" <K> [<xabs.expect>] [--tf32|--bf16|--f16]
                             [--bound <xbound.expect>]
K is the program's longest fold chain (`tiles/<stem>.kmax`, else the probe stem's `_d<d>_h<h>`).
Exit 0 inside the bound, 1 outside (the offending tokens printed)."""
import sys


def gamma(n, u=2.0 ** -24):
    return n * u / (1 - n * u)


def coef(k, tf32=False):
    """The bound's coefficient of M at fold length k: ADR-0040 D2, or ADR-0041 D3 under a reduced member —
    `tf32` True (u_t = 2⁻¹¹) or the member's u_t itself (ADR-0043 D4: bf16 2⁻⁸, f16 2⁻¹¹)."""
    if not tf32:
        return gamma(k + 1)
    ut = 2.0 ** -11 if tf32 is True else tf32
    return (1 + ut) ** 2 * (1 + gamma(k + 1, 2.0 ** -23)) - 1


def within(want, got, k, mags=None, tf32=False):
    w, g = want.split(), got.split()
    m = mags.split() if mags is not None else None
    if len(w) != len(g) or (m is not None and len(m) != len(w)):
        return False, [f"{len(g)} tokens, want {len(w)}" + (f" (M: {len(m)})" if m is not None else "")]
    bad = []
    rel = 2 * coef(k, tf32)
    absb = coef(k, tf32) + gamma(k + 1, 2.0 ** -53)
    for i, (x, y) in enumerate(zip(w, g)):
        try:
            a, b = float(x), float(y)
        except ValueError:
            if x != y:
                bad.append(f"{y} != {x}")
            continue
        if m is not None:
            mag = abs(float(m[i]))
            if abs(a - b) > absb * mag:
                bad.append(f"{y} vs {x}: |d| {abs(a - b):.3e} > {absb:.3e}·M ({mag:.6g})")
        elif abs(a - b) > rel * abs(a):
            bad.append(f"{y} vs {x}: rel {abs(a - b) / max(abs(a), 1e-300):.3e} > {rel:.3e}")
    return not bad, bad


def within_bound(want, got, bounds):
    """|got − exact| ≤ the composed bound, token by token (`mapal run --interp --bound`: the output's shape, each
    float its bound); the clock's `iter ms=` line drops on every side, as box_run.sh drops it from `got`."""
    def body(s):
        return " ".join(ln for ln in s.splitlines() if not ln.startswith("iter ms="))
    w, g, b = body(want).split(), body(got).split(), body(bounds).split()
    if len(w) != len(g) or len(b) != len(w):
        return False, [f"{len(g)} tokens, want {len(w)} (bounds: {len(b)})"]
    bad = []
    for x, y, e in zip(w, g, b):
        try:
            a, c = float(x), float(y)
        except ValueError:
            if x != y:
                bad.append(f"{y} != {x}")
            continue
        if not abs(a - c) <= float(e):
            bad.append(f"{y} vs {x}: |d| {abs(a - c):.3e} > composed {float(e):.3e}")
    return not bad, bad


def demo():
    assert within("1.0 2.0", "1.0 2.0", 512)[0]
    assert within("1000.0", "1000.05", 512)[0]  # 5e-5 < 2·γ_513 = 6.1e-5
    assert not within("1000.0", "1000.1", 512)[0]
    assert not within("1.0 x", "1.0 y", 8)[0]
    assert not within("1.0", "1.0 2.0", 8)[0]
    # S64 pt_rowsum_2048 (K 4096): relative 8.3e-4 fails; against M it is a cancellation inside the bound.
    assert not within("38.090485", "38.058914", 4096)[0]
    assert within("38.090485", "38.058914", 4096, "200000.0")[0]  # |d| 0.032 ≤ 2.44e-4 · 2e5
    assert not within("38.090485", "38.058914", 4096, "100.0")[0]
    assert not within("1.0", "1.0", 8, "1.0 2.0")[0]
    # ADR-0041 D3 (K 576): two TF32 operand roundings ≈ 1e-3 relative pass only under `--tf32`.
    assert not within("100.0", "100.09", 576)[0]
    assert within("100.0", "100.09", 576, tf32=True)[0]
    assert not within("100.0", "100.3", 576, tf32=True)[0]
    # ADR-0043 D4 (K 576): bf16 multiplicands, 2·(2·2⁻⁸) ≈ 1.6e-2 relative.
    assert not within("100.0", "101.0", 576, tf32=True)[0]
    assert within("100.0", "101.0", 576, tf32=2.0 ** -8)[0]
    # bound-blkdec: the five-GPU round's px_llm_blkdec_b2_l16_bf16 element 2 is outside the one-site coefficient,
    # inside its composed bound (0.142 there).
    assert not within("0.12759829", "0.13053171", 2168, tf32=2.0 ** -8)[0]
    assert within_bound("0.12759829 x", "0.13053171 x", "1.42e-1 0e0")[0]
    assert not within_bound("0.12759829", "0.13053171", "1e-3")[0]
    assert not within_bound("1.0 2.0", "1.0", "0 0")[0]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
        print("ok")
        sys.exit(0)
    if "--bound" in sys.argv:
        i = sys.argv.index("--bound")
        ok, bad = within_bound(open(sys.argv[1]).read(), sys.argv[2], open(sys.argv[i + 1]).read())
        print("within (composed bound)" if ok else "OUTSIDE " + "; ".join(bad))
        sys.exit(0 if ok else 1)
    flags = {"--tf32": True, "--bf16": 2.0 ** -8, "--f16": 2.0 ** -11}
    tf32 = next((v for f, v in flags.items() if f in sys.argv), False)
    argv = [a for a in sys.argv if a not in flags]
    mags = open(argv[4]).read() if len(argv) > 4 else None
    ok, bad = within(open(argv[1]).read(), argv[2], int(argv[3]), mags, tf32)
    print(("within" + (" (absolute, M)" if mags is not None else "")) if ok else "OUTSIDE " + "; ".join(bad))
    sys.exit(0 if ok else 1)
