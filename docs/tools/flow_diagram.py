"""A small mermaid flowchart of a Mapal program's dataflow, drawn from its own source structure.

Nodes are the program's bindings, named by its own variable names: file reads, `iota`s, maps (with the bindings inside
their bodies in a box), folds, guards and the prints. A binding that only does arithmetic on its inputs (an index like
`row = c / n`) is not drawn: its readers point at what it reads. Edges run from each value to every binding that reads it.
Only `fn main`'s body is read; literal sizes (`64 -> n;`) are not values of the flow.
"""

from __future__ import annotations

import re
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_]\w*")


def strip_comments(src: str) -> str:
    return re.sub(r"//[^\n]*", "", src)


def body_of(src: str, head: str) -> str:
    i = src.index(head) + len(head)
    depth, j = 1, i
    while depth:
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        j += 1
    return src[i : j - 1]


def split_top(s: str) -> list[str]:
    out, depth, cur, quote = [], 0, "", False
    for ch in s:
        if ch == '"':
            quote = not quote
        if not quote:
            if ch in "{([":
                depth += 1
            elif ch in "})]":
                depth -= 1
            elif ch == ";" and depth == 0:
                out.append(cur.strip())
                cur = ""
                continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def mask_strings(s: str) -> str:
    return re.sub(r'"[^"]*"', '""', s)


class Node:
    def __init__(self, name, kind, label, refs, inner=None):
        self.name, self.kind, self.label, self.refs, self.inner = name, kind, label, refs, inner or []


def classify(stmt: str) -> tuple[str | None, str]:
    s = mask_strings(stmt)
    m = re.search(r"->\s*(\w+)\s*(?::.*)?$", s, re.S)
    name = m.group(1) if m else None
    if re.search(r"->\s*(print|println)\b", s):
        return None, "print"
    if "-> read" in s:
        return name, "read"
    if re.search(r"->\s*map\s*\{", s):
        return name, "map"
    if re.search(r"->\s*fold\b", s):
        return name, "fold"
    if re.search(r"->\s*\{\s*-(true|false|\d)", s) or re.search(r"\)\s*->\s*\{\s*-", s):
        return name, "guard"
    if re.search(r"\biota\b", s) and name:
        return name, "iota"
    return name, "calc"


def parse_block(stmts: list[str], scope: set[str], params: set[str]) -> list[Node]:
    nodes: list[Node] = []
    for st in stmts:
        name, kind = classify(st)
        if kind == "map":
            head = re.search(r"->\s*map\s*\{", mask_strings(st))
            source = st[: head.start()].strip()
            inner_src = body_of(st, st[head.start() : head.end()])
            lam = re.match(r"\s*(\w+)\s*->", inner_src)
            ps = {lam.group(1)} if lam else set()
            inner_src = inner_src[lam.end() :] if lam else inner_src
            parts = split_top(inner_src)
            sub = parse_block(parts, scope | {n.name for n in nodes if n.name}, params | ps)
            label = f"{name} = map over " + (source.rsplit("->", 1)[0].strip() + " cells" if source.endswith("iota") else source)
            refs = [w for w in IDENT.findall(source) if w in scope]
            nodes.append(Node(name, "map", label, refs, sub))
            continue
        if kind == "guard" and name is None:
            pass
        known = scope | {n.name for n in nodes if n.name}
        rest = mask_strings(st)
        refs = [w for w in IDENT.findall(rest) if w in known and w != name]
        label = {"read": f'{name} · read from file', "iota": f"{name} · a count", "fold": f"{name} · fold",
                 "guard": "guard · a choice between two values", "print": "prints", "calc": name}[kind]
        if kind == "guard" and name:
            label = f"{name} · guard"
        nodes.append(Node(name, kind, label, refs))
    return nodes


def flatten(nodes: list[Node]):
    for n in nodes:
        yield n
        yield from flatten(n.inner)


def mermaid(path: Path) -> str:
    src = strip_comments(path.read_text(encoding="utf-8"))
    main = body_of(src, "fn main() {")
    sizes = {m.group(1) for m in re.finditer(r"(?m)^\s*[\d.]+\s*->\s*(\w+)\s*;", src)}
    nodes = parse_block(split_top(main), set(), set())
    # the unnamed final guard of a map is its value: name it after the map
    for n in nodes:
        if n.kind == "map" and n.inner and n.inner[-1].kind == "guard" and n.inner[-1].name is None:
            n.inner[-1].name = f"{n.name}_value"
    flat = {n.name: n for n in flatten(nodes) if n.name}
    hidden = {k for k, n in flat.items() if n.kind == "calc"}

    def resolve(refs, seen=()):
        out = []
        for r in refs:
            if r in sizes or r not in flat:
                continue
            if r in hidden:
                out += resolve(flat[r].refs, seen + (r,))
            else:
                out.append(r)
        return out

    prints = [n for n in nodes if n.kind == "print"]
    lines = ["flowchart LR"]
    edges = []

    def ident(n):
        return n.name or "prints"

    def emit(n, indent="  "):
        if n.kind == "calc":
            return
        if n.kind == "map":
            lines.append(f'{indent}subgraph {n.name}["{n.label}"]')
            for c in n.inner:
                emit(c, indent + "  ")
            lines.append(f"{indent}end")
            return
        lines.append(f'{indent}{ident(n)}["{n.label}"]')

    for n in nodes:
        if n.kind != "print":
            emit(n)
    if prints:
        lines.append('  prints(["prints"])')
    for n in flatten(nodes):
        if n.kind in ("calc",):
            continue
        tgt = ident(n) if n.kind != "print" else "prints"
        srcs = resolve(n.refs) if n.kind != "map" else resolve(n.refs)
        for r in srcs:
            if r != tgt:
                edges.append((r, tgt))
    seen = set()
    for a, b in edges:
        if (a, b) not in seen:
            seen.add((a, b))
            lines.append(f"  {a} --> {b}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(mermaid(Path(sys.argv[1])))
