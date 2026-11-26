#!/usr/bin/env python3
"""A variant of examples/llama.mapal: top-level bindings set by name, and optionally the competitors' precision.

  variant.py <out.mapal> [name=value ...] [--compute bf16] [--cache bf16|f32]

`name=value` replaces the top-level binding `<old> -> name;` (prompt_length=2048, generated_tokens=32, or every
size for a smaller checkpoint). `--compute bf16` gives the sums of products (the projections, the attention
scores and the weighted sum of values) ADR-0041's compute-precision permission `fold: bf16 | f32`: their operands
may be rounded to bf16, the sums kept in f32 — the class the competitors compute in (bf16 activations, f32
accumulation). Without it the program is Mapal's default face: f32 activations on the bf16 weights.

`--cache bf16` stores the key/value cache in bf16, as the competitors keep it: `KeyValueCache`'s arrays are
`[bf16; cache_size]`, `write_rows` rounds each new row once (`-> to<bf16>`) and `attend` widens each cached element
it reads (`-> widen<f32>`, exact). The bf16 class (`--compute bf16`) takes it unless `--cache f32` is given.
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
        s, n = re.subn(r"\(0\.0, (along|dimensions|past_positions)\) -> fold \{(\s+sum, (?:column|d|past) ->\s+(?:weight|sum \+ query|\(layer))",
                       rf"(0.0, \1) -> fold: {compute} | f32 {{\2", s)
        assert n == 3, n
    if cache == "bf16":
        s = bf16_cache(s)
    open(out, "w").write(s)


def bf16_cache(s):
    """The key/value cache stored in bf16: its type, the rounding as a row is written, the widening as an element
    is read, and the empty cache's zero."""
    subs = [
        ("    keys: [f32; cache_size],\n    values: [f32; cache_size],",
         "    keys: [bf16; cache_size],\n    values: [bf16; cache_size],"),
        ("fn write_rows<tokens>(cache: [f32; cache_size], rows: [f32; tokens * key_value_size], layer: i32, "
         "first_position: i32) -> [f32; cache_size] {",
         "fn write_rows<tokens>(cache: [bf16; cache_size], rows: [f32; tokens * key_value_size], layer: i32, "
         "first_position: i32) -> [bf16; cache_size] {"),
        ("            -true-> rows[t * key_value_size + cell % key_value_size];",
         "            -true-> rows[t * key_value_size + cell % key_value_size] -> to<bf16>;"),
        ("                    sum, d -> sum + query[token_head * head_size + d] * cache.keys[key_row * head_size + d]",
         "                    sum, d ->\n                    cache.keys[key_row * head_size + d] -> widen<f32> -> key;\n"
         "                    sum + query[token_head * head_size + d] * key"),
        ("            sum + exponentials[token_head * context_length + past] * cache.values[value_row * head_size + d]",
         "            cache.values[value_row * head_size + d] -> widen<f32> -> value;\n"
         "            sum + exponentials[token_head * context_length + past] * value"),
        ("    (0.0, cache_size) -> fill -> nothing_cached;",
         "    0.0 -> to<bf16> -> zero;\n    (zero, cache_size) -> fill -> nothing_cached;"),
    ]
    for old, new in subs:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    main()
