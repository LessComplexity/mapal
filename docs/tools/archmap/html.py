"""Blocks -> the HTML page."""

from __future__ import annotations

import html as _h
import json

from . import mermaid, page
from .model import Model


def esc(s: str) -> str:
    return _h.escape(s, quote=True)


def inline(items) -> str:
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(esc(it))
        elif it[0] == "code":
            out.append(f"<code>{esc(it[1])}</code>")
        elif it[0] == "ref":
            _, path, line = it
            out.append(f'<a class="ref" href="{esc(Model.url(path, line))}" target="_blank" rel="noopener"><code>{esc(path)}:{line}</code></a>')
        elif it[0] == "link":
            out.append(f'<a href="{esc(it[2])}">{esc(it[1])}</a>')
        elif it[0] == "b":
            out.append(f"<b>{esc(it[1])}</b>")
        elif it[0] == "br":
            out.append("<br>")
    return "".join(out)


class Renderer:
    def __init__(self, blocks, svgs):
        self.blocks, self.svgs, self.uid = blocks, svgs, 0
        self.styles: dict[str, None] = {}  # distinct stylesheets of the diagrams, in order

    def diagram(self, text, links, alt) -> str:
        self.uid += 1
        s = self.svgs[text]
        u = f"-{self.uid}"
        dark, sd = mermaid.natural(s["dark"], u, s["key"], "d")
        light, sl = mermaid.natural(s["light"], u, s["key"], "l")
        for css in sd + sl:
            self.styles[css] = None
        attr = f" data-links='{esc(json.dumps(links))}'" if links else ""
        return f'<div class="mdiag" role="img" aria-label="{esc(alt)}"{attr}><div class="mdk">{dark}</div><div class="mlt">{light}</div></div>'

    def table(self, headers, rows, cls) -> str:
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = []
        for anchor, cells in rows:
            idattr = f' id="{esc(anchor)}"' if anchor else ""
            body.append(f"<tr{idattr}>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

    def block(self, b) -> str:
        k = b[0]
        if k == "h":
            return f'<h{b[1]} id="{esc(b[3])}">{esc(b[2])}</h{b[1]}>'
        if k == "p":
            return f"<p>{inline(b[1])}</p>"
        if k == "diagram":
            return self.diagram(b[1], b[2], b[3])
        if k == "table":
            return self.table(b[1], b[2], b[3])
        if k == "chips":
            return '<div class="chips">' + "".join(inline([c]) for c in b[1]) + "</div>"
        if k == "details":
            return f'<details{" open" if b[3] else ""}><summary>{esc(b[1])}</summary>{"".join(self.block(x) for x in b[2])}</details>'
        raise ValueError(k)

    def render(self) -> str:
        out, in_g, grid = [], False, False
        for b in self.blocks:
            if b[0] == "h" and b[1] == 2 and b[3] == "guarantees":
                out.append(self.block(b))
                in_g = True
                continue
            if in_g and b[0] == "h" and b[1] == 2:
                out.append("</div>" if grid else "")
                in_g = grid = False
            if in_g and b[0] == "h" and b[1] == 4:
                if not grid:
                    out.append('<div class="guarantees-grid">')
                    grid = True
                out.append('<div class="gcard">')
            out.append(self.block(b))
            if in_g and b[0] == "p" and b[1] and b[1][0] == ("b", "Checked in: "):
                out.append("</div>")
        if grid:
            out.append("</div>")
        return "\n".join(out)


def render(blocks, svgs, digest: str, stats: str) -> str:
    r = Renderer(blocks, svgs)
    body = r.render()
    return page.shell(body, digest, stats, "\n".join(r.styles))
