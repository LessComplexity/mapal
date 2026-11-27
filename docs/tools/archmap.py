#!/usr/bin/env python3
"""Generate the Architecture Map from docs/architecture/map.json.

    docs/tools/archmap.py            resolve every code reference, draw the diagrams, write
                                     docs/architecture-map.html and docs/architecture-map.md
    docs/tools/archmap.py --check    exit 1 if a reference no longer resolves to exactly one place, or an output is stale
                                     (each output embeds a hash of map.json, the resolved locations and this generator);
                                     needs neither Node nor a network

map.json is the curated source: edit it, never the outputs. A code reference is `<path>#<kind> <Name>` (kinds: struct, enum,
fn, trait, type, const, static, mod, impl, method Type::name, cfn, file); `Name@2` picks the second of several definitions.
Diagrams are drawn by docs/tools/mermaid_render.mjs (needs node, Chromium and network the first time; cached after).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from archmap import content, html, markdown  # noqa: E402
from archmap.model import Model  # noqa: E402
from archmap.resolve import ROOT  # noqa: E402

HTML = ROOT / "docs" / "architecture-map.html"
MD = ROOT / "docs" / "architecture-map.md"
HASH = re.compile(r"<!-- archmap:hash=([0-9a-f]+) -->")


def stats_line(m: Model) -> str:
    c = content.counts(m)
    return (f"{c['components']} components, {c['categories']} categories, {c['objects']} objects, {c['morphisms']} arrows, "
            f"{c['transformations']} transformations, {c['guarantees']} guarantees, {c['patterns']} algorithms, data structures and patterns.")


def check(m: Model) -> int:
    bad = list(m.errors)
    want = m.digest()
    for f in (HTML, MD):
        found = HASH.search(f.read_text(encoding="utf-8")) if f.exists() else None
        if not found:
            bad.append(f"{f.relative_to(ROOT)}: missing (run docs/tools/archmap.py)")
        elif found.group(1) != want:
            bad.append(f"{f.relative_to(ROOT)}: stale (map.json, a code location or the generator changed; run docs/tools/archmap.py)")
    for b in bad:
        print("archmap:", b, file=sys.stderr)
    if not bad:
        print(f"archmap: ok ({len(m.locs)} code references resolve once; outputs fresh, {want})")
    return 1 if bad else 0


def generate(m: Model) -> int:
    if m.errors:
        for e in m.errors:
            print("archmap:", e, file=sys.stderr)
        return 1
    from archmap import mermaid
    blocks = content.build(m)
    texts = [b[1] for b in _walk(blocks) if b[0] == "diagram"]
    svgs = mermaid.render_all(texts)
    digest, stats = m.digest(), stats_line(m)
    HTML.write_text(html.render(blocks, svgs, digest, stats), encoding="utf-8")
    MD.write_text(markdown.render(blocks, digest, stats), encoding="utf-8")
    print(f"archmap: wrote {HTML.relative_to(ROOT)} ({HTML.stat().st_size // 1024} KiB) and {MD.relative_to(ROOT)}; {len(texts)} diagrams; {stats}")
    return 0


def _walk(blocks):
    for b in blocks:
        yield b
        if b[0] == "details":
            yield from _walk(b[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    m = Model()
    return check(m) if a.check else generate(m)


if __name__ == "__main__":
    raise SystemExit(main())
