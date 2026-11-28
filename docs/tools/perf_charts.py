"""Charts and lines of the exposition's Performance section, drawn from the repository's own data.

Inputs (read only): benches/results/evidence/matrix.json, benches/results/evidence/full-model.json and the CPU tables
benches/results/evidence/local-rtx-5080/cpu/shapes-fair.json and numpy-mt.json. Drawing primitives and the colour scale come from
benches/results/charts/make_charts.py, so these charts look like the repository's own. Standard library only.

Each function returns the page text of one `<!-- exposition:gen ... -->` region (kind=perf, name=...): the SVG in a dark
and a light variant (`.chart-dark` / `.chart-light`), or one line of prose.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_charts", ROOT / "benches/results/charts/make_charts.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)
sys.path.insert(0, str(ROOT / "benches/results/charts"))
import suite_charts  # noqa: E402

MATRIX = ROOT / "benches/results/evidence/matrix.json"
FULL_MODEL = ROOT / "benches/results/evidence/full-model.json"
CPU_JSON = ROOT / "benches/results/evidence/local-rtx-5080/cpu/shapes-fair.json"
NUMPY_MT_JSON = ROOT / "benches/results/evidence/local-rtx-5080/cpu/numpy-mt.json"

FAMILIES = [
    ("attn", "Attention and softmax"),
    ("conv", "Convolution and FIR"),
    ("fan", "Fan-in / fan-out chains"),
    ("gemm", "Matrix products"),
    ("llm", "LLM layers"),
    ("mlp", "MLP chains"),
]
LEGS = {
    "torch-compile-cg": "torch.compile + CUDA graphs",
    "torch-eager": "PyTorch eager",
    "chain-lt": "cuBLASLt chain",
    "lt-fresh": "cuBLASLt",
    "torch-bf16-compile": "torch.compile (bf16)",
}
# (workload id, plain name): the rows the sample chart names. Their numbers come from matrix.json.
SAMPLES = [
    ("causal_1024_256", "Causal attention", "attn"),
    ("fir_4194304_16", "FIR filter, 4M samples", "conv"),
    ("mlp_2048", "MLP chain, 2048", "mlp"),
    ("pj_m2048_n4096_k4096", "Matrix product, 2048 rows", "gemm"),
    ("pj_m1_n4096_k4096", "Matrix-vector product", "gemm"),
    ("llm_dec_b32_l2048_bf16w", "LLM decode layer, batch 32, bf16 weights", "llm"),
]
SPAN = 2.0


def _themes():
    for mode, base in mc.THEMES.items():
        t = dict(base)
        t["_red_pole"] = mc.red_pole(t)
        yield mode, t


def _both(name: str, draw) -> str:
    out = []
    for mode, t in _themes():
        svg = draw(t).strip()
        out.append(f'<div class="chart chart-{mode}">{svg}</div>')
    return _unique_ids(name, "\n".join(out))


def _unique_ids(name: str, text: str) -> str:
    parts = re.split(r'(<div class="chart chart-(?:dark|light)">)', text)
    res, mode = [], ""
    for p in parts:
        m = re.match(r'<div class="chart chart-(dark|light)">', p)
        if m:
            mode = m.group(1)
            res.append(p)
        else:
            res.append(p.replace('id="t"', f'id="{name}-{mode}-t"').replace('id="d"', f'id="{name}-{mode}-d"').replace(
                'aria-labelledby="t d"', f'aria-labelledby="{name}-{mode}-t {name}-{mode}-d"'))
    return "".join(res)


def _matrix():
    m = json.loads(MATRIX.read_text())
    parts, by, common = mc.suite_rows(m)
    return m, parts, by, common


def families(a: dict) -> str:
    m, parts, by, common = _matrix()

    def draw(t):
        groups = [(label, (lambda f: lambda r: r["family"] == f)(fam)) for fam, label in FAMILIES]
        return suite_charts.counts_chart(mc, m, t, groups, "Mapal against the best library or compiler, by kind of problem",
                                         "problem class", row_h=40, strong_first=False)

    return _both("perf-families", draw)


def _leg(name: str) -> str:
    return LEGS.get(name, name)


def samples(a: dict) -> str:
    m, parts, by, common = _matrix()

    def draw(t):
        W, pad, label_w, right_w, row_h = 920, 28, 270, 230, 50
        ax0, ax1 = pad + label_w, W - pad - right_w
        lo, hi = math.log(0.1), math.log(2.0)

        def X(r):
            return ax0 + (math.log(max(0.1, min(2.0, r))) - lo) / (hi - lo) * (ax1 - ax0)

        title = "Six named programs, five GPUs"
        svg = mc.Svg(W, t, title, "Mapal time divided by the fastest competitor's time for six programs, one mark per GPU.")
        y = 38
        svg.text(pad, y, title, 17, t["ink"], 600)
        y += 22
        svg.text(pad, y, "Mapal ÷ the fastest competitor, one dot per GPU (large: RTX 5080); left of 1: Mapal faster.", 12.5, t["ink2"])
        y += 34
        top = y
        for wid, name, fam in SAMPLES:
            rs = {p: by[p][wid] for p in parts}
            home = rs["rtx-5080"]
            cy = y + row_h / 2
            svg.text(pad, cy - 2, name, 13.5, t["ink"], 600)
            svg.text(pad, cy + 14, wid, 11, t["muted"])
            rat = [r["ratio"] for r in rs.values()]
            svg.line(X(min(rat)), cy, X(max(rat)), cy, t["grid"], 3)
            for p, r in rs.items():
                col = t["faster"] if r["verdict"] == "win" else t["slower"] if r["verdict"] == "loss" else t["even"]
                big = p == "rtx-5080"
                ring = f' stroke="{t["ink"]}" stroke-width="1.5"' if big else ""
                svg.body.append(f'<circle cx="{X(r["ratio"]):.1f}" cy="{cy:.1f}" r="{7 if big else 4.5}" fill="{col}"{ring}/>')
            svg.text(ax1 + 16, cy - 2, f"{home['ratio']:.2f} on the RTX 5080", 12.5, t["ink"], 600, num=True)
            svg.text(ax1 + 16, cy + 14, f"vs {_leg(home['best_leg'])}", 11.5, t["ink2"])
            y += row_h
        for r in (0.1, 0.2, 0.5, 1.0, 2.0):
            x = X(r)
            svg.line(x, top - 6, x, y, t["base"] if r == 1.0 else t["grid"], 1.5 if r == 1.0 else 1)
            svg.text(x, y + 16, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
        y += 36
        svg.text(pad, y, "Blue: faster; grey: even; red: slower. Each against the fastest library or compiler at the same precision, on that GPU. "
                 "Data: matrix.json.", 11, t["muted"])
        return svg.render(y + 18)

    return _both("perf-samples", draw)


def _md_table(text: str, heading: str) -> list[list[str]]:
    sec = text.split(heading, 1)[1]
    rows = []
    for line in sec.splitlines()[1:]:
        if line.startswith("#"):
            break
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:]


def _num(s: str) -> float:
    return float(re.sub(r"[^0-9.]", "", s.split(" ")[0].replace("**", "")))


def cpu(a: dict) -> str:
    """Four panels: each bar is Mapal's time / the other's (1 / ratio) with its 95 % interval as a whisker."""
    d = json.loads(CPU_JSON.read_text(encoding="utf-8"))
    mt = json.loads(NUMPY_MT_JSON.read_text(encoding="utf-8"))
    sh = d["workloads"]

    def entry(name, c, who):
        return (name, 1 / c["ratio"], 1 / c["hi"], 1 / c["lo"], c["verdict"], who, 1 / c["ratio"])

    panels = [
        ("One thread against careful C++", [entry(w["name"], w["compare"]["one_thread_cpp"], "C++") for w in sh]),
        ("One thread against NumPy", [entry(w["name"], w["compare"]["one_thread_numpy"], "NumPy") for w in sh]),
        ("Whole machine (64 threads) against C++", [entry(w["name"], w["compare"]["whole_machine_cpp"], "C++") for w in sh]),
        ("Whole machine against multi-threaded NumPy (OpenBLAS)", [entry(w["name"], w["compare"], "NumPy") for w in mt["workloads"]]),
    ]

    def draw(t):
        W, pad, label_w, right_w, row_h = 920, 28, 190, 190, 22
        ax0, ax1 = pad + label_w, W - pad - right_w
        lo, hi = math.log(0.03), math.log(3.0)

        def X(r):
            return ax0 + (math.log(max(0.03, min(3.0, r))) - lo) / (hi - lo) * (ax1 - ax0)

        title = "CPU: Mapal against C++ and NumPy"
        svg = mc.Svg(W, t, title, "Mapal's time divided by the other's, per workload, with the 95 percent interval of each ratio: one thread "
                     "against careful C++ and against NumPy, the whole machine against C++ and against multi-threaded NumPy.")
        y = 38
        svg.text(pad, y, title, 17, t["ink"], 600)
        y += 22
        svg.text(pad, y, "Mapal's time ÷ the other's, log scale; left of 1: Mapal faster. Whiskers: 95 % interval.", 12.5, t["ink2"])
        y += 10
        for head, rows in panels:
            y += 22
            svg.text(pad, y, head, 13, t["ink"], 600)
            y += 12
            top = y
            for name, r, rlo, rhi, verdict, who, _ in rows:
                cy = y + row_h / 2
                col = t["faster"] if verdict == "faster" else t["slower"] if verdict == "slower" else t["even"]
                svg.text(pad, cy + 4, name, 13, t["ink"])
                svg.bar(X(1.0), X(r), cy - 7, 14, col, 3)
                svg.line(X(rlo), cy, X(rhi), cy, t["ink"], 1.5)
                svg.line(X(rlo), cy - 4, X(rlo), cy + 4, t["ink"], 1.5)
                svg.line(X(rhi), cy - 4, X(rhi), cy + 4, t["ink"], 1.5)
                if verdict == "faster":
                    msg = f"Mapal {1 / r:.3g}× faster"
                elif verdict == "slower":
                    msg = f"{who} {r:.3g}× faster"
                else:
                    msg = "no clear difference"
                svg.text(ax1 + 16, cy + 4, msg, 12, t["muted"] if verdict not in ("faster", "slower") else t["ink"], 500, num=True)
                y += row_h
            for r in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                x = X(r)
                svg.line(x, top - 4, x, y + 2, t["base"] if r == 1.0 else t["grid"], 1.5 if r == 1.0 else 1)
                svg.text(x, y + 16, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
            y += 24
        y += 6
        for line in (
            "Medians of 31 interleaved runs (21 for multi-threaded NumPy); whiskers are 95 % bootstrap intervals. A side wins only when the interval",
            "excludes 1 and the Mann-Whitney p < 0.05. C++ is written carefully, built strict and with -ffast-math, the faster one used.",
            "AMD Threadripper 9970X, 32 cores / 64 threads, 4.5 GHz cap, clang 23.1.5, 2026-11-28. Data: benches/results/evidence/local-rtx-5080/cpu/.",
        ):
            svg.text(pad, y, line, 11, t["muted"])
            y += 16
        return svg.render(y + 6)

    return _both("perf-cpu", draw)


def ttft(a: dict) -> str:
    fm = json.loads(FULL_MODEL.read_text())
    lens = fm["meta"]["prompt_lengths"]
    h = next(p for p in fm["parts"] if p["part"] == "h100")
    rr = next(r for r in h["reruns"] if r.get("class") == "bf16" and "mapal_after" in r)
    after, base = rr["mapal_after"]["ttft_ms"], rr["gpt-fast"]["ttft_ms"]
    ratios = " / ".join(f"{x / y:.2f}" for x, y in zip(after, base))
    ms = " / ".join(f"{x:g}" for x in after) + " ms against " + " / ".join(f"{y:g}" for y in base) + " ms"
    return (f'<p class="line">After the 16-bit warpgroup tiles (<code>wgmma</code>), on another H100 against gpt-fast on the same box, the bf16 '
            f'time to first token at {" / ".join(map(str, lens))} tokens is <b>{ratios}</b> of gpt-fast\'s ({ms}): ahead for short '
            f'prompts, behind for long ones. <span class="muted">Source: full-model.json, H100 <code>reruns</code>.</span></p>')


KINDS = {"families": families, "samples": samples, "cpu": cpu, "ttft": ttft}
