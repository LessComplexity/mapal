#!/usr/bin/env python3
"""Generate the linear-NN bench sources (plan-s49).

Three shapes, all square (batch = width = d), all layers d x d:

  nn2   X  . W1 . W2                       one chain, depth 2   -> 4 d^3 flops
  nn4   X  . W1 . W2 . W3 . W4             one chain, depth 4   -> 8 d^3 flops
  nn2x2 X1 . W1a . W2a  and  X2 . W1b . W2b   TWO chains, depth 2 -> 8 d^3 flops

nn4 and nn2x2 do the SAME work; only the dependency structure differs. That is the
control (measurement rule 22): nn4 has no available concurrency by construction, so a
gain that shows up in both is the machine, not the DAG.

Inputs are positive small integers ((t*a+b) % 9 + 1) rather than the tree's usual
% 101 - 50: a depth-4 f32 chain over mean-zero data cancels, and then mapal's k-order
fold and BLAS's blocked order can disagree by percent while both are correct. All
positive removes the cancellation so the value gate can actually fail.

Data generation is outside the `time` bracket in every source.
"""
import sys
from pathlib import Path

# (mul, add) per generated matrix. Distinct so no two operands are the same array.
SEEDS = {
    "x":   (7, 13),
    "x2":  (17, 3),
    "w1":  (5, 29),
    "w2":  (3, 41),
    "w3":  (11, 7),
    "w4":  (13, 17),
    "w1b": (19, 23),
    "w2b": (23, 31),
}


def gen_mat(name: str) -> str:
    mul, add = SEEDS[name]
    return f"    ta -> map {{ t -> (t * {mul} + {add}) % 9 + 1 -> widen_f32 }} -> {name};\n"


def layer(d: int, out: str, lhs: str, rhs: str) -> str:
    return (
        f"    ta -> map {{ t ->\n"
        f"        t / {d} -> i;\n"
        f"        t % {d} -> j;\n"
        f"        (0.0, kr) -> fold {{ acc, k -> acc + {lhs}[i * {d} + k] * {rhs}[k * {d} + j] }}\n"
        f"    }} -> {out};\n"
    )


def program(shape: str, d: int) -> str:
    dd = d * d
    if shape == "nn2":
        mats, layers, outs = ["x", "w1", "w2"], [("h1", "x", "w1"), ("h2", "h1", "w2")], ["h2"]
    elif shape == "nn4":
        mats = ["x", "w1", "w2", "w3", "w4"]
        layers = [("h1", "x", "w1"), ("h2", "h1", "w2"), ("h3", "h2", "w3"), ("h4", "h3", "w4")]
        outs = ["h4"]
    elif shape == "nn2x2":
        mats = ["x", "x2", "w1", "w2", "w1b", "w2b"]
        layers = [
            ("h1", "x", "w1"), ("h2", "h1", "w2"),        # net A
            ("g1", "x2", "w1b"), ("g2", "g1", "w2b"),     # net B -- reads nothing of net A
        ]
        outs = ["h2", "g2"]
    else:
        raise SystemExit(f"unknown shape {shape}")

    body = [f"fn main() {{\n    {dd} -> iota -> ta;\n"]
    body += [gen_mat(m) for m in mats]
    body.append(f"    {d} -> iota -> kr;\n    () -> time -> t0;\n")
    body += [layer(d, *spec) for spec in layers]
    body.append("    () -> time -> t1;\n")
    for out in outs:
        body.append(f"    {out}[0] -> println;\n    {out}[{dd - 1}] -> println;\n")
    body.append('    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n')
    return "".join(body)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    sizes = [int(s) for s in sys.argv[2:]] or [128, 256, 512, 1024, 2048]
    for d in sizes:
        for shape in ("nn2", "nn4", "nn2x2"):
            path = out_dir / f"{shape}_{d}.mapal"
            path.write_text(program(shape, d))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
