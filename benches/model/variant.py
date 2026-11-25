#!/usr/bin/env python3
"""A variant of examples/llama.mapal: top-level bindings set by name, and optionally the competitors' precision.

  variant.py <out.mapal> [name=value ...] [--compute bf16]

`name=value` replaces the top-level binding `<old> -> name;` (prompt_length=2048, generated_tokens=32, or every
size for a smaller checkpoint). `--compute bf16` gives the sums of products (the projections, the attention
scores and the weighted sum of values) ADR-0041's compute-precision permission `fold: bf16 | f32`: their operands
may be rounded to bf16, the sums kept in f32 — the class the competitors compute in (bf16 activations, f32
accumulation). Without it the program is Mapal's default face: f32 activations on the bf16 weights.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "examples", "llama.mapal")


def main():
    args = sys.argv[1:]
    out = args.pop(0)
    compute = None
    if "--compute" in args:
        i = args.index("--compute")
        compute = args[i + 1]
        del args[i:i + 2]
    s = open(SRC).read()
    for a in args:
        name, value = a.split("=", 1)
        s, n = re.subn(rf"^(\S+)( -> {name};)", rf"{value}\2", s, flags=re.M)
        assert n == 1, f"no top-level binding `{name}`"
    if compute:
        # the three sums of products: `project`'s fold and `attend`'s two folds over the head and the context
        s, n = re.subn(r"\(0\.0, (along|dimensions|past_positions)\) -> fold \{(\s+sum, (?:column|d|past) ->\s+(?:weight|sum \+ query|\(layer))",
                       rf"(0.0, \1) -> fold: {compute} | f32 {{\2", s)
        assert n == 3, n
    open(out, "w").write(s)


if __name__ == "__main__":
    main()
