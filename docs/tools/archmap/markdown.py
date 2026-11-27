"""Blocks -> the Markdown page (GitHub and Neovim readers): Mermaid code blocks, tables, relative links into the code."""

from __future__ import annotations

from .model import Model


def cell_text(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def inline(items) -> str:
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(cell_text(it))
        elif it[0] == "code":
            out.append("`" + it[1].replace("`", "'") + "`")
        elif it[0] == "ref":
            _, path, line = it
            out.append(f"[`{path}:{line}`](../{path}#L{line})")
        elif it[0] == "link":
            out.append(f"[{cell_text(it[1])}]({it[2]})")
        elif it[0] == "b":
            out.append(f"**{cell_text(it[1])}**")
        elif it[0] == "br":
            out.append("<br>")
    return "".join(out)


def block(b) -> str:
    k = b[0]
    if k == "h":
        return f'<a id="{b[3]}"></a>\n\n{"#" * b[1]} {b[2]}\n'
    if k == "p":
        return inline(b[1]).replace("<br>", "  \n") + "\n"
    if k == "diagram":
        return f"```mermaid\n{b[1]}\n```\n"
    if k == "chips":
        return " · ".join(inline([c]) for c in b[1]) + "\n"
    if k == "table":
        head = "| " + " | ".join(b[1]) + " |\n|" + "---|" * len(b[1]) + "\n"
        rows = []
        for anchor, cells in b[2]:
            cs = [inline(c) for c in cells]
            if anchor:
                cs[0] = f'<a id="{anchor}"></a>' + cs[0]
            rows.append("| " + " | ".join(cs) + " |")
        return head + "\n".join(rows) + "\n"
    if k == "details":
        inner = "\n".join(block(x) for x in b[2])
        return f"<details{' open' if b[3] else ''}>\n<summary>{b[1]}</summary>\n\n{inner}\n</details>\n"
    raise ValueError(k)


def render(blocks, digest: str, stats: str) -> str:
    head = (f"# Mapal Architecture Map\n\nThe compiler seen as the data that defines it: kinds of data (categories), the things and arrows inside them, "
            f"and the transformations between them. {stats} Every item links to the line of code that defines it. "
            f"A richer, interactive version with the diagrams pre-drawn is [architecture-map.html](architecture-map.html).\n\n"
            f"Contents: [Data](#data) · [Memory](#memory) · [Components](#components) · [Guarantees](#guarantees) · [Index](#index)\n\n")
    body = "\n".join(block(b) for b in blocks)
    tail = ("\n---\n\nGenerated from `docs/architecture/map.json` by `docs/tools/archmap.py` (`--check` keeps it true). "
            "Older, more detailed notes: [categorical-model.md](architecture/categorical-model.md), [deduced-queries.md](architecture/deduced-queries.md).\n"
            f"\n<!-- archmap:hash={digest} -->\n")
    return head + body + tail
