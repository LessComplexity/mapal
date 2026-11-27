"""Pre-rendered Mermaid: the same renderer, pinned engine and theme as the Language Exposition (docs/tools/mermaid_render.mjs),
batched (one browser start for all new diagrams) and cached by diagram text; no layout happens in the reader's browser."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from .resolve import ROOT

RENDERER = ROOT / "docs" / "tools" / "mermaid_render.mjs"
CACHE = Path(os.environ.get("ARCHMAP_CACHE", Path.home() / ".cache" / "mapal-archmap")) / "mermaid"


def key_of(text: str) -> str:
    return hashlib.sha256((RENDERER.read_text() + "\0" + text).encode()).hexdigest()[:24]


def render_all(texts: list[str]) -> dict[str, dict]:
    """text -> {"dark": svg, "light": svg, "key": key}."""
    out: dict[str, dict] = {}
    todo = []
    for t in dict.fromkeys(texts):
        k = key_of(t)
        f = CACHE / f"{k}.json"
        if f.exists():
            out[t] = json.loads(f.read_text())
        else:
            todo.append((k, t))
    CACHE.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), 25):
        chunk = todo[i:i + 25]
        p = subprocess.run(["node", str(RENDERER)], input=json.dumps([{"id": k, "text": t} for k, t in chunk]),
                           capture_output=True, text=True)
        if p.returncode:
            raise SystemExit(f"mermaid_render.mjs failed (needs node, Chromium and network):\n{p.stderr[-800:]}")
        res = json.loads(p.stdout)
        for k, t in chunk:
            svgs = dict(res[k], key=k)
            (CACHE / f"{k}.json").write_text(json.dumps(svgs))
            out[t] = svgs
    return out


def natural(svg: str, uid: str, key: str, theme: str) -> tuple[str, list[str]]:
    """The viewBox's size as the width (CSS caps it at the column), floats shortened, ids unique on the page, and the
    diagram's own <style> lifted out (scoped to the theme's pane) so the page carries each distinct stylesheet once."""
    scope = ".mdiag .mdk svg" if theme == "d" else ".mdiag .mlt svg"
    styles = [t.replace(f"#mmd-{key}", scope) for t in re.findall(r"<style>(.*?)</style>", svg, re.S)]
    svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
    w = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) [\d.]+"', svg).group(1)
    svg = svg.replace('width="100%"', f'width="{w}"', 1)
    svg = re.sub(r' style="max-width: [\d.]+px;"', "", svg, count=1)
    svg = re.sub(r"-?\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}".rstrip("0").rstrip("."), svg)
    return svg.replace(f"mmd-{key}", f"mmd-{key}-{theme}{uid}"), styles
