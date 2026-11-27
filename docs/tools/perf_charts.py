"""Charts and lines of the exposition's Performance section, drawn from the repository's own data.

Inputs (read only): benches/results/evidence/matrix.json, benches/results/evidence/full-model.json and the CPU table
benches/results/evidence/local-rtx-5080/cpu/shapes-fair.json. Drawing primitives and the colour scale come from
benches/results/charts/make_charts.py, so these charts look like the repository's own. Standard library only.

Each function returns the page text of one `<!-- exposition:gen ... -->` region (kind=perf, name=...): the SVG in a dark
and a light variant (`.chart-dark` / `.chart-light`), or one line of prose.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_charts", ROOT / "benches/results/charts/make_charts.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

MATRIX = ROOT / "benches/results/evidence/matrix.json"
FULL_MODEL = ROOT / "benches/results/evidence/full-model.json"
CPU_JSON = ROOT / "benches/results/evidence/local-rtx-5080/cpu/shapes-fair.json"

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
    even = m["meta"].get("even_threshold", 0.05)

    def draw(t):
        W, pad, label_w, row_h = 920, 28, 250, 54
        cell_w = (W - 2 * pad - label_w) / len(parts)
        title = "Mapal against the best library or compiler, by kind of problem"
        n_all = len(common)
        svg = mc.Svg(W, t, title, "Geomean of Mapal time divided by the fastest same-precision competitor's time, per problem "
                     "class and GPU, with the number of programs faster, even and slower.")
        y = 38
        svg.text(pad, y, title, 17, t["ink"], 600)
        y += 22
        svg.text(pad, y, f"Each cell: the geomean of Mapal's time ÷ the fastest competitor's, and below it the programs "
                 f"faster · even · slower ({n_all} programs timed on every GPU).", 12.5, t["ink2"])
        y += 30
        for j, p in enumerate(parts):
            svg.text(pad + label_w + j * cell_w + cell_w / 2, y, mc.part_name(p), 12.5, t["ink"], 600, "middle")
        y += 10
        for fam, label in FAMILIES:
            n = sum(1 for w in common if by[parts[0]][w]["family"] == fam)
            svg.text(pad, y + row_h / 2 - 2, label, 13.5, t["ink"], 600)
            svg.text(pad, y + row_h / 2 + 14, f"{n} programs", 11.5, t["muted"])
            for j, p in enumerate(parts):
                rows = [by[p][w] for w in common if by[p][w]["family"] == fam]
                x = pad + label_w + j * cell_w
                g = mc.geomean([r["ratio"] for r in rows])
                fill = mc.diverging(t, g)
                v = mc.verdicts(rows)
                svg.rect(x + 1.5, y + 1.5, cell_w - 3, row_h - 3, fill, 5)
                ink = mc.label_on(fill)
                svg.text(x + cell_w / 2, y + row_h / 2 - 1, f"{g:.2f}", 16, ink, 700, "middle", num=True)
                svg.text(x + cell_w / 2, y + row_h / 2 + 15, f"{v['win']} · {v['even']} · {v['loss']}", 11.5, ink, 500, "middle", num=True)
            y += row_h
        y += 24
        lx0, lw = pad + label_w, W - 2 * pad - label_w
        steps = 60
        lo, hi = math.log(1 / SPAN), math.log(SPAN)
        for k in range(steps):
            r = math.exp(lo + (k + 0.5) / steps * (hi - lo))
            svg.rect(lx0 + k * lw / steps, y, lw / steps + 0.6, 10, mc.diverging(t, r))
        for r in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0):
            svg.text(lx0 + (math.log(r) - lo) / (hi - lo) * lw, y + 24, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
        svg.text(pad, y + 9, "Mapal faster ← → slower", 11.5, t["ink2"])
        y += 44
        svg.text(pad, y, f"Even: within {even:.0%} or the run-to-run variation. Data: benches/results/evidence/matrix.json.", 11, t["muted"])
        return svg.render(y + 18)

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
    """Two fair pairs on one machine: one thread against one thread, whole machine against whole machine."""
    d = json.loads(CPU_JSON.read_text(encoding="utf-8"))
    rows = []
    for w in d["workloads"]:
        m1 = min(w["mapal_conf_1t"], w["mapal_fma_1t"])
        mp = min(w["mapal_conf_par"], w["mapal_fma_par"])
        cw = min(w["cpp_1t"], w["cpp_mt"])
        rows.append((w["name"], m1, w["cpp_1t"], w["numpy_1t"], mp, cw))

    def draw(t):
        W, pad, label_w, right_w, row_h = 920, 28, 190, 280, 22
        ax0, ax1 = pad + label_w, W - pad - right_w
        lo, hi = math.log(0.04), math.log(2.0)

        def X(r):
            return ax0 + (math.log(max(0.04, min(2.0, r))) - lo) / (hi - lo) * (ax1 - ax0)

        title = "CPU: Mapal against C++ and NumPy"
        svg = mc.Svg(W, t, title, "Mapal time divided by C++'s and NumPy's time on one AMD Threadripper 9970X, per workload: "
                     "one thread against one thread, and the whole machine against the whole machine.")
        y = 38
        svg.text(pad, y, title, 17, t["ink"], 600)
        y += 22
        svg.text(pad, y, f"Mapal's time ÷ the other's, log scale; left of 1: Mapal faster. {d['machine']}, {d['date']}.", 12.5, t["ink2"])
        y += 10

        def panel(y, head, cols):
            y += 22
            svg.text(pad, y, head, 13, t["ink"], 600)
            y += 12
            top = y
            for name, m, others in cols:
                svg.text(pad, y + len(others) * row_h / 2 + 4, name, 13, t["ink"])
                for k, (who, other) in enumerate(others):
                    r = m / other
                    cy = y + k * row_h + row_h / 2
                    col = t["faster"] if r < 0.95 else t["slower"] if r > 1.05 else t["even"]
                    svg.bar(X(1.0), X(r), cy - 7, 14, col, 3)
                    msg = f"{1 / r:.3g}× faster" if r < 0.95 else f"{r:.3g}× slower" if r > 1.05 else "level"
                    svg.text(ax1 + 16, cy + 4, f"vs {who}: {msg}", 12, t["ink"], 500, num=True)
                    svg.text(ax1 + 172, cy + 4, f"{m:#.3g} vs {other:#.3g} ms", 11, t["muted"], num=True)
                y += len(others) * row_h + 8
            for r in (0.05, 0.1, 0.25, 0.5, 1.0):
                x = X(r)
                svg.line(x, top - 4, x, y - 4, t["base"] if r == 1.0 else t["grid"], 1.5 if r == 1.0 else 1)
                svg.text(x, y + 10, f"{r:g}", 11, t["muted"], anchor="middle", num=True)
            return y + 16

        y = panel(y, "One thread against one thread", [(n, m1, [("C++", c1), ("NumPy", npy)]) for n, m1, c1, npy, _, _ in rows])
        y = panel(y + 6, "Whole machine against whole machine (64 threads)",
                  [(n, mp, [("C++", cw)]) for n, _, _, _, mp, cw in rows])
        y += 14
        svg.text(pad, y, "C++ whole machine: the faster of its one-thread and its threaded build (the threaded one starts its threads on every call, "
                 "which costs more than it saves at 1M elements).", 11, t["muted"])
        y += 16
        svg.text(pad, y, "Times: best of 25 for one thread, median of 25 for the whole machine. NumPy runs these on one thread. "
                 "Data: benches/results/evidence/local-rtx-5080/cpu/shapes-fair.json.", 11, t["muted"])
        return svg.render(y + 18)

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
