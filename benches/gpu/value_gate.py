"""ADR-0040 D2's whole-program gate for the default face: every printed value within relative
2·γ_{K+1} of the exact face's (u = 2⁻²⁴, γ_n = n·u/(1 − n·u)); non-numeric tokens equal.

usage: python3 value_gate.py <exact.expect> "<got, one line>" <K>
K is the program's longest tile-site fold. ponytail: box_run.sh reads it off the probe stem
(`_d<d>_h<h>` → max(d, h)) until `mapal dump-ir` prints fold lengths.
Exit 0 inside the bound, 1 outside (the offending tokens printed)."""
import sys


def gamma(n, u=2.0 ** -24):
    return n * u / (1 - n * u)


def within(want, got, k):
    w, g = want.split(), got.split()
    if len(w) != len(g):
        return False, [f"{len(g)} tokens, want {len(w)}"]
    bad, tol = [], 2 * gamma(k + 1)
    for x, y in zip(w, g):
        try:
            a, b = float(x), float(y)
        except ValueError:
            if x != y:
                bad.append(f"{y} != {x}")
            continue
        if abs(a - b) > tol * abs(a):
            bad.append(f"{y} vs {x}: rel {abs(a - b) / max(abs(a), 1e-300):.3e} > {tol:.3e}")
    return not bad, bad


def demo():
    assert within("1.0 2.0", "1.0 2.0", 512)[0]
    assert within("1000.0", "1000.05", 512)[0]  # 5e-5 < 2·γ_513 = 6.1e-5
    assert not within("1000.0", "1000.1", 512)[0]
    assert not within("1.0 x", "1.0 y", 8)[0]
    assert not within("1.0", "1.0 2.0", 8)[0]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
        print("ok")
        sys.exit(0)
    ok, bad = within(open(sys.argv[1]).read(), sys.argv[2], int(sys.argv[3]))
    print("within" if ok else "OUTSIDE " + "; ".join(bad))
    sys.exit(0 if ok else 1)
