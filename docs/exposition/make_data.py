#!/usr/bin/env python3
"""Write the small input files of the exposition demos into docs/exposition/data/ (raw little-endian arrays, read
with `"name.bin" -> read -> x: [f32; n];`, resolved under MAPAL_DATA_DIR). Deterministic: rerunning changes nothing."""

import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"


def put(name: str, values, fmt: str = "f") -> None:
    values = list(values)
    (OUT / name).write_bytes(struct.pack(f"<{len(values)}{fmt}", *values))


def bf16(x: float) -> int:
    """The bf16 bit pattern of x, rounded to nearest even."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return (b + 0x7FFF + ((b >> 16) & 1)) >> 16


def main() -> None:
    OUT.mkdir(exist_ok=True)
    n = 64
    put("dense-a.bin", ((c * 7 + 13) % 101 - 50 for c in range(n * n)))
    put("dense-b.bin", ((c * 5 + 3) % 97 - 48 for c in range(n * n)))
    put("signal.bin", range(1024))
    put("vector.bin", (i % 10 - 5 for i in range(4096)))
    tokens, width = 16, 8
    put("query.bin", ((c * 7 + 13) % 11 - 5 for c in range(tokens * width)))
    put("key.bin", ((c * 5 + 3) % 13 - 6 for c in range(tokens * width)))
    put("value.bin", ((c * 3 + 1) % 7 - 3 for c in range(tokens * width)))
    put("rod.bin", (100.0 if i == 32 else 0.0 for i in range(64)))
    rows, columns, toks = 8, 16, 4
    put("weight.bin", (bf16(((c * 11 + 57) % 103 - 51) / 64.0) for c in range(rows * columns)), "H")
    put("activations.bin", (((c * 7 + 13) % 101 - 50) / 8.0 for c in range(toks * columns)))


if __name__ == "__main__":
    main()
