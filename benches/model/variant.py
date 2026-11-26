#!/usr/bin/env python3
"""A variant of examples/llama.mapal: top-level bindings set by name, and optionally the competitors' precision.

  variant.py <out.mapal> [name=value ...] [--compute bf16] [--cache bf16|f32]

`name=value` replaces the top-level binding `<old> -> name;` (prompt_length=2048, generated_tokens=32, or every
size for a smaller checkpoint). `--compute bf16` gives the sums of products (the projections, the attention
scores and the weighted sum of values) ADR-0041's compute-precision permission `fold: bf16 | f32`: their operands
may be rounded to bf16, the sums kept in f32 — the class the competitors compute in (bf16 activations, f32
accumulation). Without it the program is Mapal's default face: f32 activations on the bf16 weights.

`--cache bf16` stores the key/value cache in bf16, as the competitors keep it: the program's one choice of the
cache's format (`main`'s `0.0 -> to<f32> -> zero;`, ADR-0051) becomes `to<bf16>` — `KeyValueCache<element>` is then
`[bf16; cache_size]`, `write_rows` rounds each new row once (`-> to<element>`) and `attend` widens each cached element
it reads (`-> widen<f32>`, exact); in the f32 class both conversions are identities. The bf16 class
(`--compute bf16`) takes it unless `--cache f32` is given.
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
    cache = compute
    if "--cache" in args:
        i = args.index("--cache")
        cache = args[i + 1]
        del args[i:i + 2]
    s = open(SRC).read()
    for a in args:
        name, value = a.split("=", 1)
        s, n = re.subn(rf"^(\S+)( -> {name};)", rf"{value}\2", s, flags=re.M)
        assert n == 1, f"no top-level binding `{name}`"
    if compute:
        # the three sums of products: `project`'s fold and `attend`'s two folds over the head and the context
        s, n = re.subn(r"\(0\.0, (along|dimensions|past_positions)\) -> fold \{(\s+sum, (?:column|d|past) ->\s+(?:weight|query|\(layer))",
                       rf"(0.0, \1) -> fold: {compute} | f32 {{\2", s)
        assert n == 3, n
    if cache == "bf16":
        s = bf16_cache(s)
    open(out, "w").write(s)


def bf16_cache(s):
    """The key/value cache stored in bf16: the format `main` chooses for it (ADR-0051: every function over the
    cache takes its format as a parameter, inferred from the cache)."""
    old, new = "    0.0 -> to<f32> -> zero;", "    0.0 -> to<bf16> -> zero;"
    assert s.count(old) == 1, old
    return s.replace(old, new)


if __name__ == "__main__":
    main()
