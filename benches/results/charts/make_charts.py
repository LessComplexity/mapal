#!/usr/bin/env python3
"""Draw the result charts and the headline tables from the evidence data.

Inputs (written by the measurement steps; this script only reads them):
  benches/results/evidence/matrix.json       the cross-part suite matrix (benches/local/matrix.py)
  benches/results/evidence/full-model.json   the full model per GPU and precision class, every engine
                                             (local-rtx-5080/full-model.md, cross-arch/README.md)

Outputs:
  benches/results/charts/<chart>-light.svg and <chart>-dark.svg  (README.md and docs/performance/results.md
  pick one per GitHub theme with <picture>): llm-f32, llm-bf16, suite-gpus (winners and losers per GPU),
  suite-groups (counts per family and precision class)
  docs/performance/results.md between `<!-- results-headline:start -->` and `<!-- results-headline:end -->`
  (README.md is never written: it shows the charts only)

A part's `reruns` that carry a `label` get a row of their own under the part's rows (Mapal's `mapal_after` against
the fastest engine measured with it on that box) and their `note` after the table.

Run from anywhere:  python3 benches/results/charts/make_charts.py   (--no-results: print the tables instead)
Standard library only. Colors follow the dataviz reference palette (validated: blue / red polarity pair in
both modes). Type to type: a precision class is only ever compared with engines of its own class
(full-model.json's classes; llama.cpp f16 sits under other_types and is never drawn), each suite row only
with competitors at its own precision (the matrix's ratio rule).
"""

import argparse
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_charts  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "benches/results/evidence/matrix.json"
FULL_MODEL = ROOT / "benches/results/evidence/full-model.json"
OUT = ROOT / "benches/results/charts"
RESULTS = ROOT / "docs/performance/results.md"
HEADLINE_START = "<!-- results-headline:start -->"
HEADLINE_END = "<!-- results-headline:end -->"
PROGRAMS_START = "<!-- results-programs:start -->"
PROGRAMS_END = "<!-- results-programs:end -->"

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


def mapal_engine(engines):
    return next(e for e in engines if e.get("mapal"))


def competitors(engines):
    return [e for e in engines if not e.get("mapal")]


def class_meta(fm, cid):
    return next(c for c in fm["classes"] if c["id"] == cid)


def model_cell(part, cid, metric, j):
    """Mapal ÷ the fastest engine of the class at prompt index j: (ratio, Mapal's value, best value, best name)."""
    engines = part["classes"][cid]["engines"]
    me = mapal_engine(engines)
    best = min(competitors(engines), key=lambda e: e[metric][j])
    return me[metric][j] / best[metric][j], me[metric][j], best[metric][j], best["name"]


def rerun_cell(rr, metric, j):
    """A rerun's Mapal (`mapal_after`) ÷ the fastest engine measured on its box: (ratio, Mapal, best, best name)."""
    me = rr["mapal_after"][metric][j]
    engines = [(k, v) for k, v in rr.items() if isinstance(v, dict) and metric in v and not k.startswith("mapal")]
    name, best = min(engines, key=lambda e: e[1][metric][j])
    return me / best[metric][j], me, best[metric][j], name


def model_domain(fm, j):
    """One log domain, symmetric about 1, for both class charts (so their bars compare)."""
    rs = [
        model_cell(p, c["id"], m, j)[0]
        for p in fm["parts"]
        for c in fm["classes"]
        for m in ("ttft_ms", "decode_ms")
        if c["id"] in p["classes"]
    ]
    m = max(math.log(1.25), max(abs(math.log(r)) for r in rs) * 1.12)
    return math.exp(-m), math.exp(m)


def verdict(ratio, even):
    return "even" if abs(ratio - 1) <= even else ("faster" if ratio < 1 else "slower")


# --- charts --------------------------------------------------------------------------------------------------


def llm_chart(fm, cid, theme, domain, even):
    """Per GPU, Mapal ÷ the fastest engine of one class at a 512-token prompt: TTFT, then decode."""
    t = theme
    meta = fm["meta"]
    cls = class_meta(fm, cid)
    j = meta["prompt_lengths"].index(512)
    parts = [p for p in fm["parts"] if cid in p["classes"]]
    lo, hi = domain
    W, pad, label_w, right_w = 920, 28, 104, 300
    x_lo, x_hi = pad + label_w, W - pad - right_w

    def X(r):
        return x_lo + (math.log(r) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (x_hi - x_lo)

    model = meta["model"].replace("-Instruct", "")
    title = f"{model}, {cls['label']}: Mapal against the fastest {cid} engine"
    engines = list(dict.fromkeys(e["name"] for p in parts for e in competitors(p["classes"][cid]["engines"])))
    sections = (
        ("ttft_ms", "Time to first token, 512-token prompt", "ms"),
        ("decode_ms", "Decode, per generated token, 512-token prompt", "ms"),
    )
    desc = f"{cls['chart_note']} " + " ".join(
        f"{heading}: "
        + "; ".join(
            "{} {:.2f} ({} vs {} {}, {})".format(part_name(p["part"]), r, sig(a), sig(b), unit, n)
            for p in parts
            for r, a, b, n in [model_cell(p, cid, metric, j)]
        )
        + "."
        for metric, heading, unit in sections
    )
    svg = Svg(W, t, title, desc)
    y = 38
    svg.text(pad, y, title, 17, t["ink"], 600)
    y += 22
    svg.text(pad, y, cls["chart_note"], 12.5, t["ink2"])
    y += 18
    svg.text(
        pad,
        y,
        f"Mapal time ÷ the fastest engine's time, per GPU. Batch {meta['batch']}, greedy, "
        f"{meta['generated_tokens']} generated tokens, medians of 5. Below 1: Mapal faster.",
        12.5,
        t["ink2"],
    )
    y += 26
    lx = pad
    for key, label in (("faster", "Mapal faster"), ("even", f"within {even:.0%}"), ("slower", "Mapal slower")):
        svg.rect(lx, y - 9, 10, 10, t[key], 2)
        svg.text(lx + 15, y, label, 12, t["ink2"])
        lx += 30 + 7 * len(label)
    ticks = [r for r in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0) if lo <= r <= hi]
    row_h, bar_h = 30, 14
    for metric, heading, unit in sections:
        y += 34
        svg.text(pad, y, heading, 14, t["ink"], 600)
        svg.text(W - pad, y, "Mapal vs the fastest engine", 12, t["ink2"], anchor="end")
        y += 10
        top = y
        bottom = top + row_h * len(parts)
        for r in ticks:
            svg.line(X(r), top, X(r), bottom, t["base"] if r == 1.0 else t["grid"])
        for i, p in enumerate(parts):
            r, a, b, n = model_cell(p, cid, metric, j)
            yy = top + i * row_h + (row_h - bar_h) / 2
            cy = yy + bar_h - 2
            svg.text(pad, cy, part_name(p["part"]), 13, t["ink"], 600)
            svg.bar(X(1.0), X(r), yy, bar_h, t[verdict(r, even)])
            if r < 1:
                svg.text(X(r) - 6, cy, f"{r:.2f}", 12, t["ink"], 600, "end", num=True)
            else:
                svg.text(X(r) + 6, cy, f"{r:.2f}", 12, t["ink"], 600, num=True)
            svg.text(W - pad, cy, f"{sig(a)} vs {sig(b)} {unit} · {n}", 12, t["ink2"], anchor="end", num=True)
        y = bottom + 16
        for r in ticks:
            svg.text(X(r), y, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
    y += 18
    svg.text(X(lo), y, "← Mapal faster", 11.5, t["ink2"])
    svg.text(X(hi), y, "Mapal slower →", 11.5, t["ink2"], anchor="end")
    y += 26
    svg.text(pad, y, f"Engines of the class: {', '.join(engines)}.", 11, t["muted"])
    notes = "; ".join(f"{x['name']}: {x['why']}" for x in cls.get("not_compared", []))
    if notes:
        y += 16
        svg.text(pad, y, f"Not compared: {notes}.", 11, t["muted"])
    y += 16
    svg.text(
        pad,
        y,
        f"Builds b8429388 / 9be9886d. Data: benches/results/evidence/full-model.json (every engine, prompts "
        f"{' / '.join(map(str, meta['prompt_lengths']))}).",
        11,
        t["muted"],
    )
    return svg.render(y + 18)


def suite_gpu_chart(matrix, theme):
    return suite_charts.gpu_chart(sys.modules[__name__], matrix, theme)


def suite_groups_chart(matrix, theme):
    return suite_charts.groups_chart(sys.modules[__name__], matrix, theme)


# --- the headline tables (docs/performance/results.md) -------------------------------------------------------


def headline(fm, matrix):
    even = matrix["meta"].get("even_threshold", 0.05)
    meta = fm["meta"]
    lengths = meta["prompt_lengths"]
    j512 = lengths.index(512)

    def ratio(r, places=2):
        shown = rnd(r, places)
        return f"**{shown}**" if float(shown) < 1 - even else shown

    parts, by, common = suite_rows(matrix)
    notes = []
    lines = [
        f"**Kernel and layer suite**: programs where Mapal is faster, level or slower than the fastest competitor at the "
        f"program's own precision, of the {len(common)} timed on every GPU, both sides timed at the program boundary. "
        f"Level: within the row's measured run-to-run variation or {even:.0%}, whichever is larger. No averages.",
        "",
        *suite_charts.table_md(sys.modules[__name__], matrix),
        "",
    ]
    model = meta["model"].replace("-Instruct", "")
    lines += [
        "",
        f"**{model}, the full model**: Mapal ÷ the fastest engine of the same class, prompts "
        f"{' / '.join(map(str, lengths))} (batch {meta['batch']}, {meta['generated_tokens']} generated tokens, "
        "medians of 5).",
        "",
        "| GPU | class | time to first token | decode per token | at 512: Mapal vs the fastest engine |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for p in fm["parts"]:
        for c in fm["classes"]:
            cid = c["id"]
            if cid not in p["classes"]:
                continue
            cells = []
            for metric in ("ttft_ms", "decode_ms"):
                cells.append(" / ".join(ratio(model_cell(p, cid, metric, j)[0]) for j in range(len(lengths))))
            _, a, b, n = model_cell(p, cid, "ttft_ms", j512)
            _, da, db, dn = model_cell(p, cid, "decode_ms", j512)
            at = f"{sig(a)} vs {sig(b)} ms ({n}); {sig(da)} vs {sig(db)} ms / token ({dn})"
            lines.append(f"| {part_name(p['part'])} | {cid} | {cells[0]} | {cells[1]} | {at} |")
        for rr in p.get("reruns", []):
            if not rr.get("label"):
                continue
            cells = []
            for metric in ("ttft_ms", "decode_ms"):
                cells.append(" / ".join(ratio(rerun_cell(rr, metric, j)[0]) for j in range(len(lengths))))
            _, a, b, n = rerun_cell(rr, "ttft_ms", j512)
            _, da, db, dn = rerun_cell(rr, "decode_ms", j512)
            at = f"{sig(a)} vs {sig(b)} ms ({n}); {sig(da)} vs {sig(db)} ms / token ({dn})"
            lines.append(f"| {part_name(p['part'])}, {rr['label']} | {rr['class']} | {cells[0]} | {cells[1]} | {at} |")
            if rr.get("note"):
                notes.append(rr["note"])
    lines += [
        "",
        f"Below 1: Mapal faster; bold: faster by more than {even:.0%}. The **bf16** rows are like for like. The "
        "**f32 math** rows are not: Mapal computes in f32 on the model's bf16 weights (half the bytes per token), "
        "gpt-fast, vLLM, llama.cpp and Hugging Face in f32 on f32 weights with TF32 off (SGLang has no f32 path and TensorRT-LLM's f32 output is wrong: neither is "
        "compared); **bf16** (`fold: bf16 | f32`, bf16 KV cache) against gpt-fast, vLLM, SGLang, "
        "TensorRT-LLM, llama.cpp and Hugging Face in bf16. llama.cpp in f16 is another 16-bit format and is not "
        "compared." + "".join(" " + n for n in notes),
    ]
    return "\n".join(lines)


def update_block(text, start, end, what):
    s = RESULTS.read_text()
    name = RESULTS.relative_to(ROOT)
    if start not in s or end not in s:
        print(f"{name} has no {start} … {end} block; left unchanged")
        return
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    new = pattern.sub(lambda _: f"{start}\n\n{text}\n\n{end}", s)
    if new != s:
        RESULTS.write_text(new)
        print(f"{name}: {what} updated")
    else:
        print(f"{name}: {what} unchanged")


def update_results(text):
    update_block(text, HEADLINE_START, HEADLINE_END, "headline tables")


def programs_text(matrix):
    me = sys.modules[__name__]
    even = matrix["meta"].get("even_threshold", 0.05)
    return "\n".join(
        [
            "**By family and precision class**: programs faster / level / slower, per GPU.",
            "",
            *suite_charts.family_table_md(me, matrix),
            "",
            f"**Every program**: Mapal's time ÷ the fastest same-precision competitor's, per GPU. Bold: faster; `≈`: level "
            f"(within the row's measured variation or {even:.0%}); plain: slower. Leaders and times per row: "
            "[`matrix.md`](../../benches/results/evidence/matrix.md).",
            "",
            *suite_charts.program_table_md(me, matrix),
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-results", action="store_true", help="print the headline tables, leave results.md alone")
    args = ap.parse_args()
    matrix = json.loads(MATRIX.read_text())
    fm = json.loads(FULL_MODEL.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, base in THEMES.items():
        theme = dict(base)
        theme["_red_pole"] = red_pole(theme)
        even = matrix["meta"].get("even_threshold", 0.05)
        domain = model_domain(fm, fm["meta"]["prompt_lengths"].index(512))
        charts = {f"llm-{c['id']}": llm_chart(fm, c["id"], theme, domain, even) for c in fm["classes"]}
        charts["suite-gpus"] = suite_gpu_chart(matrix, theme)
        charts["suite-groups"] = suite_groups_chart(matrix, theme)
        for name, body in charts.items():
            path = OUT / f"{name}-{mode}.svg"
            path.write_text(body)
            print(f"wrote {path.relative_to(ROOT)}")
    table = headline(fm, matrix)
    if not args.no_results:
        update_results(table)
        update_block(programs_text(matrix), PROGRAMS_START, PROGRAMS_END, "program tables")
    else:
        print(table)


if __name__ == "__main__":
    main()
