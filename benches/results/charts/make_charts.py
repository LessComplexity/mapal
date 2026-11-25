#!/usr/bin/env python3
"""Draw the README's result charts and its headline table from the evidence data.

Inputs (written by the measurement steps; this script only reads them):
  benches/results/evidence/matrix.json       the cross-part suite matrix (benches/local/matrix.py)
  benches/results/evidence/full-model.json   the full-model runs (from local-rtx-5080/full-model.md)

Outputs:
  benches/results/charts/<chart>-light.svg and <chart>-dark.svg  (the README picks one per GitHub theme
  with <picture>): llm-f32, llm-bf16, suite-gpus, suite-groups
  README.md between `<!-- results-headline:start -->` and `<!-- results-headline:end -->`

Run from anywhere:  python3 benches/results/charts/make_charts.py   (--no-readme: leave README.md alone)
Standard library only. Colors follow the dataviz reference palette (validated: blue / red polarity pair in
both modes); a type is only ever compared with its own type (the matrix's ratio rule, full-model.json's
classes).
"""

import argparse
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "benches/results/evidence/matrix.json"
FULL_MODEL = ROOT / "benches/results/evidence/full-model.json"
OUT = ROOT / "benches/results/charts"
README = ROOT / "README.md"
HEADLINE_START = "<!-- results-headline:start -->"
HEADLINE_END = "<!-- results-headline:end -->"

PART_NAMES = {
    "rtx-5080": "RTX 5080",
    "rtx-3090": "RTX 3090",
    "rtx-4090": "RTX 4090",
    "rtx-5090": "RTX 5090",
    "a100": "A100",
    "h100": "H100",
}
PRECISION_NAMES = {
    "f32": "f32",
    "tf32": "TF32",
    "bf16w": "bf16 weights, f32 activations",
    "bf16kv": "bf16 weights and KV cache",
    "bf16": "bf16 storage, bf16 | f32 products",
    "f8w": "fp8 weights",
}
PRECISION_ORDER = ["f32", "tf32", "bf16w", "bf16kv", "bf16", "f8w"]
FAMILY_NAMES = {
    "gemm": "GEMM cells and projections",
    "mlp": "MLP chains",
    "attn": "Attention, softmax, long rows",
    "llm": "LLM layers: decode, prefill, block",
    "conv": "Conv, FIR, elementwise",
    "fan": "Fan-in / fan-out, loops, CPU+GPU",
}

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "base": "#c3c2b7",
        "mapal": "#2a78d6",
        "other": "#c3c2b7",
        "faster": "#2a78d6",
        "slower": "#e34948",
        "even": "#898781",
        "mid": "#f0efec",
        "pole": "#1c5cab",
        "red": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "base": "#383835",
        "mapal": "#3987e5",
        "other": "#52514e",
        "faster": "#3987e5",
        "slower": "#e66767",
        "even": "#898781",
        "mid": "#383835",
        "pole": "#6da7ec",
        "red": "#e66767",
    },
}


# --- color: OKLab mixing for the diverging scale, WCAG contrast for labels inside fills ---------------------


def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def to_oklab(h):
    r, g, b = (_lin(c) for c in hex_rgb(h))
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_linear(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def from_oklab(lab):
    rgb = [min(1.0, max(0.0, _gam(min(1.0, max(0.0, c))))) for c in _oklab_linear(lab)]
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def in_gamut(lab):
    return all(-1e-4 <= c <= 1 + 1e-4 for c in _oklab_linear(lab))


def red_pole(theme):
    """The red arm's pole: the palette red's hue at the blue pole's lightness and chroma (or the most the gamut holds)."""
    L, a, b = to_oklab(theme["pole"])
    chroma = math.hypot(a, b)
    _, ra, rb = to_oklab(theme["red"])
    hue = math.atan2(rb, ra)
    while chroma > 0 and not in_gamut((L, chroma * math.cos(hue), chroma * math.sin(hue))):
        chroma -= 0.002
    return from_oklab((L, chroma * math.cos(hue), chroma * math.sin(hue)))


def diverging(theme, ratio, span=2.0):
    """Blue below 1 (Mapal faster), red above, neutral gray at 1; log scale clamped at span and 1 / span."""
    t = max(-1.0, min(1.0, math.log(ratio) / math.log(span)))
    mid = to_oklab(theme["mid"])
    pole = to_oklab(theme["pole"] if t < 0 else theme["_red_pole"])
    k = abs(t)
    return from_oklab(tuple(m + k * (p - m) for m, p in zip(mid, pole)))


def luminance(h):
    r, g, b = (_lin(c) for c in hex_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def label_on(fill):
    """White or near-black inside a fill, the same two in both modes, by the fill's luminance."""
    return "#ffffff" if contrast(fill, "#ffffff") >= contrast(fill, "#0b0b0b") else "#0b0b0b"


# --- SVG -----------------------------------------------------------------------------------------------------


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class Svg:
    def __init__(self, width, theme, title, desc):
        self.w = width
        self.t = theme
        self.title = title
        self.desc = desc
        self.body = []

    def rect(self, x, y, w, h, fill, rx=0):
        self.body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, stroke, width=1):
        self.body.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def bar(self, xb, xe, y, h, fill, r=4):
        """A horizontal bar from the baseline xb to the data end xe: square at the baseline, rounded at the end."""
        if abs(xe - xb) < 0.5:
            return
        r = min(r, h / 2, abs(xe - xb))
        s = 1 if xe > xb else -1
        sweep = 1 if s > 0 else 0
        self.body.append(
            f'<path d="M{xb:.1f},{y:.1f} H{xe - s * r:.1f} A{r},{r} 0 0 {sweep} {xe:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} A{r},{r} 0 0 {sweep} {xe - s * r:.1f},{y + h:.1f} H{xb:.1f} Z" fill="{fill}"/>'
        )

    def text(self, x, y, s, size=12, fill=None, weight=400, anchor="start", num=False):
        cls = ' class="num"' if num else ""
        self.body.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill or self.t["ink"]}"'
            f' text-anchor="{anchor}"{cls}>{esc(s)}</text>'
        )

    def render(self, height):
        t = self.t
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{height:.0f}" '
            f'viewBox="0 0 {self.w} {height:.0f}" role="img" aria-labelledby="t d">\n'
            f"<title id=\"t\">{esc(self.title)}</title>\n<desc id=\"d\">{esc(self.desc)}</desc>\n"
            f"<style>text {{ font-family: {FONT}; }} .num {{ font-variant-numeric: tabular-nums; }}</style>\n"
            f'<rect width="{self.w}" height="{height:.0f}" rx="10" fill="{t["surface"]}"/>\n'
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


def rnd(v, places):
    """Round half up on the number as written (74.05 -> 74.1, as the reports round it)."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP))


def sig(v):
    """Three significant figures, never scientific: 125, 28.9, 3.76."""
    return rnd(v, 0 if v >= 100 else 1 if v >= 10 else 2)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def part_name(p):
    return PART_NAMES.get(p, p.upper())


# --- data ----------------------------------------------------------------------------------------------------


def suite_rows(matrix):
    """Per part: workload -> row, rows with a same-precision ratio only (fp8 rows have no leg yet)."""
    by = {p: {} for p in matrix["meta"]["parts"]}
    for r in matrix["rows"]:
        if r.get("ratio") and r["part"] in by:
            by[r["part"]][r["workload"]] = r
    parts = [p for p in matrix["meta"]["parts"] if by[p]]
    common = set.intersection(*(set(by[p]) for p in parts)) if parts else set()
    return parts, by, common


def verdicts(rows):
    out = {"win": 0, "even": 0, "loss": 0}
    for r in rows:
        if r.get("verdict") in out:
            out[r["verdict"]] += 1
    return out


def mapal_engine(cls):
    return next(e for e in cls["engines"] if e.get("mapal") and e.get("chart"))


def others(cls):
    return [e for e in cls["engines"] if e.get("chart") and not e.get("mapal")]


# --- charts --------------------------------------------------------------------------------------------------


def llm_chart(fm, cls, theme):
    meta = fm["meta"]
    lengths = meta["prompt_lengths"]
    engines = [mapal_engine(cls)] + others(cls)
    W, pad, label_w, gap = 920, 28, 188, 26
    panel_w = (W - 2 * pad - label_w - gap * (len(lengths) - 1)) / len(lengths)
    row_h, bar_h = 22, 12
    t = theme
    title = f"{meta['model']} on an {meta['gpu']}: {cls['label']}"
    desc = (
        f"Time to first token and decode time per token for {', '.join(e['name'] for e in engines)}, "
        f"at prompt lengths {', '.join(map(str, lengths))}. {cls.get('chart_note', '')}"
    )
    svg = Svg(W, t, title, desc)
    y = 38
    svg.text(pad, y, title, 17, t["ink"], 600)
    y += 22
    svg.text(pad, y, cls.get("chart_note", cls["compute"]), 12.5, t["ink2"])
    y += 18
    svg.text(
        pad,
        y,
        f"Batch {meta['batch']}, greedy, {meta['generated_tokens']} generated tokens, medians. "
        "Shorter bars are faster; the fastest engine in each panel is in bold.",
        12.5,
        t["ink2"],
    )
    y += 20
    for metric, heading in (("ttft_ms", "Time to first token, ms"), ("decode_ms", "Decode, ms per generated token")):
        y += 30
        svg.text(pad, y, heading, 14, t["ink"], 600)
        y += 22
        top = y
        for j, n in enumerate(lengths):
            x0 = pad + label_w + j * (panel_w + gap)
            svg.text(x0, y, f"prompt of {n} tokens", 12, t["ink2"])
        y += 10
        rows_top = y
        for j, n in enumerate(lengths):
            x0 = pad + label_w + j * (panel_w + gap)
            vals = [e[metric][j] for e in engines]
            best = min(vals)
            scale = (panel_w - 52) / max(vals)
            for i, e in enumerate(engines):
                v = e[metric][j]
                yy = rows_top + i * row_h + (row_h - bar_h) / 2
                fill = t["mapal"] if e.get("mapal") else t["other"]
                svg.bar(x0, x0 + v * scale, yy, bar_h, fill)
                is_best = v == best
                svg.text(
                    x0 + v * scale + 6,
                    yy + bar_h - 1.5,
                    sig(v),
                    11.5,
                    t["ink"] if is_best else t["ink2"],
                    700 if is_best else 400,
                    num=True,
                )
            svg.line(x0, rows_top, x0, rows_top + len(engines) * row_h, t["base"])
        for i, e in enumerate(engines):
            yy = rows_top + i * row_h + row_h / 2 + 4.5
            name = e["name"]
            svg.text(pad, yy, name, 12.5, t["ink"] if e.get("mapal") else t["ink2"], 700 if e.get("mapal") else 400)
        y = rows_top + len(engines) * row_h
        del top
    y += 26
    svg.text(
        pad,
        y,
        f"{meta['gpu']}, Mapal build {meta['mapal_build']}, {meta['date']}. Data: {meta['report']} "
        "(benches/results/evidence/full-model.json).",
        11,
        t["muted"],
    )
    return svg.render(y + 18)


def suite_gpu_chart(matrix, theme):
    t = theme
    parts, by, common = suite_rows(matrix)
    even = matrix["meta"].get("even_threshold", 0.05)
    data = []
    for p in parts:
        rows = [by[p][w] for w in common]
        data.append((p, geomean([r["ratio"] for r in rows]), verdicts(rows)))
    ratios = [g for _, g, _ in data if g]
    m = max(math.log(1.25), max(abs(math.log(g)) for g in ratios) * 1.15)
    lo, hi = math.exp(-m), math.exp(m)
    W, pad, label_w, right_w = 920, 28, 120, 230
    x_lo, x_hi = pad + label_w, W - pad - right_w

    def X(r):
        return x_lo + (math.log(r) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (x_hi - x_lo)

    title = "Kernel and layer suite: Mapal against the fastest same-precision competitor"
    desc = "Geometric mean of Mapal time divided by the fastest competitor's time, per GPU: " + "; ".join(
        f"{part_name(p)} {g:.3f}" for p, g, _ in data
    )
    svg = Svg(W, t, title, desc)
    y = 38
    svg.text(pad, y, title, 17, t["ink"], 600)
    y += 22
    svg.text(
        pad,
        y,
        f"Geomean of Mapal time ÷ the fastest competitor's time over the {len(common)} programs timed on every GPU. "
        "Below 1: Mapal faster.",
        12.5,
        t["ink2"],
    )
    y += 18
    svg.text(
        pad,
        y,
        "Each program against competitors of its own precision: cuBLAS(Lt), CUTLASS, PyTorch eager / compiled, Triton, "
        "FlashAttention, FlashInfer, Liger.",
        12.5,
        t["ink2"],
    )
    y += 26
    lx = pad
    for key, label in (("faster", "Mapal faster"), ("even", f"within {even:.0%}"), ("slower", "Mapal slower")):
        svg.rect(lx, y - 9, 10, 10, t[key], 2)
        svg.text(lx + 15, y, label, 12, t["ink2"])
        lx += 30 + 7 * len(label)
    y += 22
    svg.text(W - pad, y, "programs faster / even / slower", 12, t["ink2"], anchor="end")
    y += 10
    row_h, bar_h = 34, 14
    top = y
    ticks = [r for r in (0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0) if lo <= r <= hi]
    bottom = top + row_h * len(data)
    for r in ticks:
        svg.line(X(r), top, X(r), bottom, t["base"] if r == 1.0 else t["grid"])
    for i, (p, g, v) in enumerate(data):
        yy = top + i * row_h + (row_h - bar_h) / 2
        cy = yy + bar_h - 2
        svg.text(pad, cy, part_name(p), 13, t["ink"], 600)
        key = "even" if abs(g - 1) <= even else ("faster" if g < 1 else "slower")
        svg.bar(X(1.0), X(g), yy, bar_h, t[key])
        if g < 1:
            svg.text(X(g) - 6, cy, f"{g:.3f}", 12, t["ink"], 600, "end", num=True)
        else:
            svg.text(X(g) + 6, cy, f"{g:.3f}", 12, t["ink"], 600, num=True)
        svg.text(W - pad, cy, f"{v['win']} / {v['even']} / {v['loss']}", 12.5, t["ink2"], anchor="end", num=True)
    y = bottom + 18
    for r in ticks:
        svg.text(X(r), y, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
    y += 18
    svg.text(X(lo), y, "← Mapal faster", 11.5, t["ink2"])
    svg.text(X(hi), y, "Mapal slower →", 11.5, t["ink2"], anchor="end")
    y += 26
    commits = []
    for p in parts:
        cs = sorted({s["commit"] for s in matrix["meta"].get("sources", {}).get(p, [])})
        commits.append(f"{part_name(p)} {', '.join(cs)}")
    svg.text(pad, y, f"Builds: {'; '.join(commits)}.", 11, t["muted"])
    y += 16
    svg.text(
        pad,
        y,
        f"Data: benches/results/evidence/matrix.json ({matrix['meta'].get('generated', '')}); even = within "
        f"{even:.0%} or the row's own noise.",
        11,
        t["muted"],
    )
    return svg.render(y + 18)


def suite_groups_chart(matrix, theme):
    t = theme
    parts, by, _ = suite_rows(matrix)
    fams = matrix["meta"].get("families", {})
    groups = [("All rows", lambda r: True, True)]
    present_fams = [f for f in fams if any(r["family"] == f for p in parts for r in by[p].values())]
    for f in present_fams:
        groups.append((FAMILY_NAMES.get(f, fams[f]), (lambda f: lambda r: r["family"] == f)(f), False))
    precs = [x for x in PRECISION_ORDER if any(r["precision"] == x for p in parts for r in by[p].values())]
    precs += sorted({r["precision"] for p in parts for r in by[p].values()} - set(precs))
    split = len(groups)
    for x in precs:
        groups.append((PRECISION_NAMES.get(x, x), (lambda x: lambda r: r["precision"] == x)(x), False))

    W, pad, label_w = 920, 28, 300
    cell_w = (W - 2 * pad - label_w) / len(parts)
    row_h = 30
    title = "Kernel and layer suite by family and by precision"
    desc = "Geomean of Mapal time divided by the fastest same-precision competitor's time, per group and GPU."
    svg = Svg(W, t, title, desc)
    y = 38
    svg.text(pad, y, title, 17, t["ink"], 600)
    y += 22
    svg.text(
        pad,
        y,
        "Geomean of Mapal time ÷ the fastest competitor's time at the row's own precision; below 1 (blue): Mapal "
        "faster. Rows per GPU in brackets.",
        12.5,
        t["ink2"],
    )
    y += 30
    for j, p in enumerate(parts):
        svg.text(pad + label_w + j * cell_w + cell_w / 2, y, part_name(p), 12.5, t["ink"], 600, "middle")
    y += 10
    for gi, (label, pred, strong) in enumerate(groups):
        if gi == 1 or gi == split:
            y += 14
            svg.text(pad, y + 4, "By family" if gi == 1 else "By precision class", 12, t["muted"], 600)
            y += 12
        counts = [sum(1 for r in by[p].values() if pred(r)) for p in parts]
        n = max(counts) if counts else 0
        svg.text(pad, y + row_h / 2 + 4.5, f"{label} ({n})", 12.5, t["ink"], 700 if strong else 400)
        for j, p in enumerate(parts):
            rows = [r["ratio"] for r in by[p].values() if pred(r)]
            x = pad + label_w + j * cell_w
            if not rows:
                svg.rect(x + 1, y + 1, cell_w - 2, row_h - 2, t["mid"], 3)
                svg.text(x + cell_w / 2, y + row_h / 2 + 4.5, "–", 12, t["muted"], anchor="middle")
                continue
            g = geomean(rows)
            fill = diverging(t, g)
            svg.rect(x + 1, y + 1, cell_w - 2, row_h - 2, fill, 3)
            svg.text(
                x + cell_w / 2,
                y + row_h / 2 + 4.5,
                f"{g:.2f}" + ("" if len(rows) == n else f" ({len(rows)})"),
                12.5,
                label_on(fill),
                700 if strong else 500,
                "middle",
                num=True,
            )
        y += row_h
    y += 26
    lx0, lw = pad + label_w, W - 2 * pad - label_w
    steps = 60
    for k in range(steps):
        r = math.exp(math.log(0.5) + (k + 0.5) / steps * (math.log(2.0) - math.log(0.5)))
        svg.rect(lx0 + k * lw / steps, y, lw / steps + 0.6, 10, diverging(t, r))
    for r in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0):
        x = lx0 + (math.log(r) - math.log(0.5)) / (math.log(2.0) - math.log(0.5)) * lw
        svg.text(x, y + 24, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
    svg.text(pad, y + 9, "Mapal faster ← → slower", 11.5, t["ink2"])
    y += 46
    svg.text(
        pad,
        y,
        "A count after a value: that GPU timed fewer rows of the group. fp8-weight rows have no competitor leg yet and "
        "are left out. Data: benches/results/evidence/matrix.json.",
        11,
        t["muted"],
    )
    return svg.render(y + 18)


# --- the README's headline table ---------------------------------------------------------------------------


def headline(fm, matrix):
    even = matrix["meta"].get("even_threshold", 0.05)
    meta = fm["meta"]
    lengths = meta["prompt_lengths"]

    def ratio_cell(rs, places=2):
        return " / ".join(f"**{rnd(r, places)}**" if r < 1 - even else rnd(r, places) for r in rs)

    lines = [
        "| result | Mapal | fastest competitor, same precision | Mapal ÷ fastest |",
        "| --- | ---: | ---: | ---: |",
    ]
    model = meta["model"].replace("-Instruct", "")
    for cls in fm["classes"]:
        me = mapal_engine(cls)
        rest = others(cls)
        for metric, what, unit in (("ttft_ms", "time to first token", "ms"), ("decode_ms", "decode", "ms / token")):
            mine = me[metric]
            best = [min(rest, key=lambda e: e[metric][j]) for j in range(len(lengths))]
            bv = [b[metric][j] for j, b in enumerate(best)]
            names = [b["name"] for b in best]
            if len(set(names)) == 1:
                theirs = " / ".join(sig(v) for v in bv) + f" {unit} ({names[0]})"
            else:
                theirs = " / ".join(f"{sig(v)} ({n})" for v, n in zip(bv, names)) + f" {unit}"
            lines.append(
                f"| {model}, {cls['label']}, {what}, prompt {' / '.join(map(str, lengths))} "
                f"| {' / '.join(sig(v) for v in mine)} {unit} | {theirs} "
                f"| {ratio_cell([a / b for a, b in zip(mine, bv)])} |"
            )
    parts, by, common = suite_rows(matrix)
    gs, wins, losses = [], [], []
    for p in parts:
        rows = [by[p][w] for w in common]
        gs.append(geomean([r["ratio"] for r in rows]))
        v = verdicts(rows)
        wins.append(str(v["win"]))
        losses.append(str(v["loss"]))
    names = " / ".join(part_name(p) for p in parts)
    lines.append(
        f"| kernel and layer suite, {len(common)} programs, geomean, {names} "
        f"| faster on {' / '.join(wins)} | faster on {' / '.join(losses)} | {ratio_cell(gs, 3)} |"
    )
    lines.append("")
    lines.append(
        f"{meta['gpu']} for the model (batch {meta['batch']}, {meta['generated_tokens']} generated tokens, medians). "
        f"Below 1: Mapal faster; bold: faster by more than {even:.0%}. "
        "Each precision class is compared only with engines of its own class."
    )
    return "\n".join(lines)


def update_readme(text):
    s = README.read_text()
    if HEADLINE_START not in s or HEADLINE_END not in s:
        print(f"README.md has no {HEADLINE_START} … {HEADLINE_END} block; left unchanged")
        return
    pattern = re.compile(re.escape(HEADLINE_START) + r".*?" + re.escape(HEADLINE_END), re.S)
    new = pattern.sub(lambda _: f"{HEADLINE_START}\n\n{text}\n\n{HEADLINE_END}", s)
    if new != s:
        README.write_text(new)
        print("README.md: headline table updated")
    else:
        print("README.md: headline table unchanged")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-readme", action="store_true", help="do not rewrite the README's headline block")
    args = ap.parse_args()
    matrix = json.loads(MATRIX.read_text())
    fm = json.loads(FULL_MODEL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, base in THEMES.items():
        theme = dict(base)
        theme["_red_pole"] = red_pole(theme)
        charts = {f"llm-{c['id']}": llm_chart(fm, c, theme) for c in fm["classes"]}
        charts["suite-gpus"] = suite_gpu_chart(matrix, theme)
        charts["suite-groups"] = suite_groups_chart(matrix, theme)
        for name, body in charts.items():
            path = OUT / f"{name}-{mode}.svg"
            path.write_text(body)
            print(f"wrote {path.relative_to(ROOT)}")
    table = headline(fm, matrix)
    if not args.no_readme:
        update_readme(table)
    else:
        print(table)


if __name__ == "__main__":
    main()
