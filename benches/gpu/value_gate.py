"""ADR-0040 D2's whole-program gate for the default face, u = 2⁻²⁴, γ_n = n·u/(1 − n·u).

With the program's magnitude oracle (`<stem>_xabs.expect`, gen_tiles.py `xabs`: every fold step's term
replaced by its magnitude, so each printed value's M = |seed| + Σ|terms| through the whole fold chain):
D2 itself, absolute — |got − exact| ≤ (γ_{K+1}(2⁻²⁴) + γ_{K+1}(2⁻⁵³))·M. Without one: every printed value
within relative 2·γ_{K+1} of the exact face's (which fails a correct result under cancellation, S64
`pt_rowsum_2048`: |exact| ≪ M). Non-numeric tokens equal.

A program whose products admit TF32 (`: tf32 | f32`, ADR-0041) passes `--tf32`: D3's coefficient
(1 + u_t)²·(1 + γ_{K+1}(2⁻²³)) − 1, u_t = 2⁻¹¹, replaces γ_{K+1}(2⁻²⁴).

usage: python3 value_gate.py <exact.expect> "<got, one line>" <K> [<xabs.expect>] [--tf32]
K is the program's longest fold chain (`tiles/<stem>.kmax`, else the probe stem's `_d<d>_h<h>`).
Exit 0 inside the bound, 1 outside (the offending tokens printed)."""
import sys


def gamma(n, u=2.0 ** -24):
    return n * u / (1 - n * u)


def coef(k, tf32=False):
    """The bound's coefficient of M at fold length k: ADR-0040 D2, or ADR-0041 D3 under `tf32`."""
    return (1 + 2.0 ** -11) ** 2 * (1 + gamma(k + 1, 2.0 ** -23)) - 1 if tf32 else gamma(k + 1)


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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
        print("ok")
        sys.exit(0)
    tf32 = "--tf32" in sys.argv
    argv = [a for a in sys.argv if a != "--tf32"]
    mags = open(argv[4]).read() if len(argv) > 4 else None
    ok, bad = within(open(argv[1]).read(), argv[2], int(argv[3]), mags, tf32)
    print(("within" + (" (absolute, M)" if mags is not None else "")) if ok else "OUTSIDE " + "; ".join(bad))
    sys.exit(0 if ok else 1)
