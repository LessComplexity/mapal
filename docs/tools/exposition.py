#!/usr/bin/env python3
"""Regenerate the machine-made regions of docs/language-exposition.html from the real tools.

Nothing in a generated region is typed by hand: program output comes from `mapal run`, compiler facts from
`mapal build --emit-only -v` (the device module's `; mapal.plan` / `; mapal.flow` lines, the static and rewrite
reports) and `mapal dump-ir`, instruction mixes from the emitted PTX, charts from benches/results/charts/.
Nothing runs on a GPU: a GPU profile is only compiled to text (--emit-only), values come from the interpreter
(`--interp`) or a native CPU run (`--target=zen3`).

A region in the page:

    <!-- exposition:gen id=NAME kind=KIND key=value ... begin -->...<!-- exposition:gen end -->

Kinds (attributes in brackets are optional):
    run      file [cpu=1] [scale=a=1,b=2]   program output, and each printed float with its composed bound
    derive   file [target] [scale]          static and rewrite reports + the kernels the placement chose
    gpus     file [targets] [scale]         the same kernel under several GPU profiles (compile only)
    ptxmix   file [target] [scale]          PTX instruction mix of the emitted module + the inner-loop excerpt
    graph    file [short] [mermaid=1]        operation counts of the graph: as lowered, after rewriting; mermaid=1 adds the flow
                                             view, the stored graph (`--mermaid=graph`) and the full category and string diagrams (zoom boxes, collapsed)
    syntax   file                            token classes the language server reports
    llama    [target]                        kernel kinds of the whole Llama-3.2-1B emission
    chart    name                            an inline SVG chart (light and dark variants), ids made unique
    diagrams file                            the hero's two-tab switch: the condensed category and string diagrams
                                             (`--mermaid` = `--mermaid=string,condensed`, and `--mermaid=category,condensed`)
    flow     file                            the flow view of the compiler's own graph: `mapal dump-ir --mermaid=flow`
    gpuline  file [target] [scale]          one plain sentence on the kernel the compiler made (compile only)
    perf     name=families|samples|cpu|ttft  the Performance section's charts and line, drawn by docs/tools/perf_charts.py
                                             from matrix.json, full-model.json and earlier-readme-results.md

`scale` rewrites leading `N -> name;` lines (the program's sizes) before the run or the compile, so a demo can
show real output at a size the interpreter finishes and the kernels at the size where a GPU matters.

Also checked: every `<!-- mapal:hl begin ... excerpt=PATH:FROM-TO -->` region holds exactly those lines of PATH.

Usage:  docs/tools/exposition.py            rewrite the regions in place
        docs/tools/exposition.py --check    exit 1 if any region (or excerpt) is stale
Env:    MAPAL_BIN (default target/release/mapal)   MAPAL_LLC_DIR via the compiler's own LLVM lookup
"""

from __future__ import annotations

import argparse
import base64
import json
import difflib
import html
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "language-exposition.html"
DATA = ROOT / "docs" / "exposition" / "data"  # the demos' input files (docs/exposition/make_data.py)
CHARTS = ROOT / "benches" / "results" / "charts"
REGION = re.compile(r"<!-- exposition:gen (?P<attrs>.*?) begin -->(?P<body>.*?)<!-- exposition:gen end -->", re.S)
ATTR = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
HLREGION = re.compile(r"<!-- mapal:hl begin (?P<attrs>.*?) -->(?P<body>.*?)<!-- mapal:hl end -->", re.S)
TMP = Path(tempfile.mkdtemp(prefix="exposition-"))
CACHE: dict = {}


def exe() -> str:
    for cand in (os.environ.get("MAPAL_BIN"), str(ROOT / "target" / "release" / "mapal")):
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit("no mapal binary: set MAPAL_BIN or build target/release/mapal")


def sh(args: list[str], cwd: Path | None = None, ok: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, MAPAL_DATA_DIR=str(DATA))
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=env)
    if p.returncode != 0 and not ok:
        raise SystemExit(f"failed: {' '.join(args)}\n{p.stderr[-800:]}")
    return p


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def attrs_of(s: str) -> dict[str, str]:
    return {m.group(1): m.group(2) if m.group(2) is not None else m.group(3) for m in ATTR.finditer(s)}


def program(file: str, scale: str | None) -> Path:
    """The file, or a copy with the program's leading `N -> name;` sizes replaced; always under ROOT's tree
    layout so a relative `read` finds its data (copies keep the original's directory)."""
    src = ROOT / file
    if not scale:
        return src
    text = src.read_text(encoding="utf-8")
    for kv in scale.split(","):
        name, val = kv.split("=")
        text, n = re.subn(rf"(?m)^\d+ -> {name};", f"{val} -> {name};", text, count=1)
        if n != 1:
            raise SystemExit(f"{file}: no `N -> {name};` line to scale")
    out = src.parent / f".scaled-{abs(hash((file, scale)))}.mapal"
    out.write_text(text, encoding="utf-8")
    CACHE.setdefault("scaled", []).append(out)
    return out


def emit(file: str, target: str, scale: str | None = None) -> dict:
    """mapal build --emit-only for a target: the -v reports, the device module and the PTX, as text."""
    key = ("emit", file, target, scale)
    if key in CACHE:
        return CACHE[key]
    p = program(file, scale)
    d = TMP / f"e{len(CACHE)}"
    d.mkdir()
    r = sh([exe(), "build", "--emit-only", "--cands=1", f"--target={target}", "-v", "-o", str(d / "m"), str(p)], cwd=p.parent)
    rep = {"static": "", "rewrite": ""}
    for line in r.stderr.splitlines() + r.stdout.splitlines():
        for k in rep:
            if line.startswith(k + ":"):
                rep[k] = line
    dev = (d / "m.dev.ll").read_text() if (d / "m.dev.ll").exists() else ""
    ptx = (d / "m.ptx").read_text() if (d / "m.ptx").exists() else ""
    CACHE[key] = {"report": rep, "dev": dev, "ptx": ptx}
    return CACHE[key]


def kv(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", line))


def kernels(dev: str) -> list[dict[str, str]]:
    """One record per kernel: its `; mapal.plan` fields, joined with its `; mapal.flow` line."""
    flows = {kv(line).get("kernel"): kv(line) for line in dev.splitlines() if line.startswith("; mapal.flow ")}
    out = []
    for line in dev.splitlines():
        if line.startswith("; mapal.plan "):
            f = kv(line)
            f["flow"] = flows.get(f["kernel"], {})
            out.append(f)
    return out


def brief(k: dict) -> str:
    kind = k.get("kind", "?")
    parts = [f"kind={kind}"]
    if kind == "flow":
        fl = k["flow"]
        n = len(fl.get("members", "").split(";")) if fl.get("members") else 0
        parts += [f"rows={k.get('rows')}", f"steps={n}", f"grid={k.get('gx')}", f"block={k.get('block')}"]
        if fl.get("smem"):
            parts.append(f"smem={fl['smem']}")
        if fl.get("windows"):
            parts.append(f"windows={fl['windows']}")
    elif kind == "tile":
        parts += [f"{a}={k[a]}" for a in ("rows", "c", "k") if a in k]
        parts += [f"grid={k.get('gx')}x{k.get('gy')}", f"block={k.get('block')}", f"tile={k.get('tm')}x{k.get('tn')}",
                  f"bk={k.get('bk')}", f"stages={k.get('stages')}", f"smem={k.get('smem')}"]
    elif kind == "mma":
        parts += [f"{a}={k[a]}" for a in ("rows", "c", "k") if a in k]
        parts += [f"op={k.get('op')}", f"grid={k.get('gx')}x{k.get('gy')}", f"block={k.get('block')}",
                  f"bk={k.get('bk')}", f"stages={k.get('stages')}", f"smem={k.get('smem')}"]
        if k.get("split"):
            parts.append(f"split={k['split']}")
    elif kind == "reduce":
        parts += [f"rows={k.get('rows')}", f"c={k.get('c')}", f"S={k.get('S')}"]
    elif kind == "cell":
        parts += [f"work={k.get('work')}"]
    return " ".join(p for p in parts if not p.endswith("=None"))


def pre(text: str, cls: str = "out", label: str | None = None) -> str:
    cap = f'<div class="io-h">{label}</div>' if label else ""
    return f'<div class="io">{cap}<pre class="{cls}"><code>{esc(text)}</code></pre></div>'


# -- kinds ---------------------------------------------------------------------------------------------


def k_run(a: dict) -> str:
    p = program(a["file"], a.get("scale"))
    cmd = [exe(), "run"] + (["--target=zen3"] if a.get("cpu") else ["--interp"]) + [p.name]
    r = sh(cmd, cwd=p.parent)
    vals = r.stdout.splitlines()
    shown = " ".join(["mapal run"] + (["--target=zen3"] if a.get("cpu") else ["--interp"]) + [Path(a["file"]).name])
    if a.get("cpu"):
        return pre("\n".join(vals), label=f"Output <span>{esc(shown)} (native CPU run)</span>")
    b = sh([exe(), "run", "--interp", "--bound", p.name], cwd=p.parent, ok=True)
    bounds = b.stdout.splitlines()
    rows = []
    for v, e in zip(vals, bounds if len(bounds) == len(vals) else [None] * len(vals)):
        if e is None:
            rows.append(v)
            continue
        pv, _, nv = v.rpartition(" ")
        pe, _, ne = e.rpartition(" ")
        if pv != pe:
            pv, nv, ne = "", v, e
        rows.append(f"{pv + ' ' if pv else ''}{nv}   ± {float(ne):.3g}")
    label = f"Output <span>{esc(shown)}; ± is the composed bound, <code>--bound</code></span>"
    return pre("\n".join(rows), label=label)


def static_lines(rep: dict) -> str:
    s = rep["static"]
    m = {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", s)}
    r = rep["rewrite"]
    applied = re.findall(r"\((\w+), (\d+)\)", r)
    rounds = re.search(r"rounds: (\d+)", r)
    return (
        f"compile-time evaluation: {m.get('embedded', 0)} values embedded as constants, {m.get('computed', 0)} left "
        f"to compute at run time\n"
        f"rewriting: {rounds.group(1) if rounds else '?'} rounds"
        + (", " + ", ".join(f"{n} {k}" for k, n in applied) if applied else ", nothing to change")
    )


def k_derive(a: dict) -> str:
    target = a.get("target", "rtx-5080")
    e = emit(a["file"], target, a.get("scale"))
    ks = kernels(e["dev"])
    head = static_lines(e["report"])
    if ks:
        body = "\n".join(brief(k) for k in ks)
        label = f"What the compiler derived <span>mapal build --emit-only --target={target}" + (
            f", sizes scaled: {esc(a['scale'])}" if a.get("scale") else "") + "</span>"
        return pre(head + f"\n{len(ks)} device kernel(s):\n" + body, label=label)
    label = f"What the compiler derived <span>mapal build --emit-only --target={target}</span>"
    return pre(head + "\nno device kernel: nothing here is placed on the device", label=label)


def k_gpus(a: dict) -> str:
    targets = a.get("targets", "rtx-5080,rtx-3090,h100").split(",")
    cols = ["kind", "op", "grid", "block", "bk", "stages", "smem", "blocks_per_sm"]
    rows = []
    for t in targets:
        e = emit(a["file"], t, a.get("scale"))
        ks = [k for k in kernels(e["dev"]) if k.get("kind") in ("mma", "tile", "wgmma")]
        k = ks[0] if ks else {}
        k = dict(k, grid=f"{k.get('gx')}x{k.get('gy')}" if k else "")
        rows.append((t, k))
    th = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    tr = "".join(
        f"<tr><th scope='row'>{esc(t)}</th>" + "".join(f"<td>{esc(str(k.get(c, '-')))}</td>" for c in cols) + "</tr>"
        for t, k in rows
    )
    cap = f"mapal build --emit-only --target=&lt;profile&gt;" + (f", sizes scaled: {esc(a['scale'])}" if a.get("scale") else "")
    return (f'<div class="io"><div class="io-h">One source, three machines <span>{cap}</span></div>'
            f'<div class="tablewrap"><table class="gputab"><thead><tr><th>profile</th>{th}</tr></thead><tbody>{tr}</tbody></table></div></div>')


def ptx_mix(ptx: str) -> Counter:
    c: Counter = Counter()
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s.startswith(("//", ".", "{", "}", "$")) or s.endswith(":"):
            continue
        op = s.split()[0].rstrip(";")
        if op.startswith("@"):
            op = s.split()[1]
        c[op] += 1
    return c


def k_ptxmix(a: dict) -> str:
    target = a.get("target", "rtx-5080")
    e = emit(a["file"], target, a.get("scale"))
    mix = ptx_mix(e["ptx"])
    groups = [("fma", "fused multiply-adds"), ("ld.global", "global loads"), ("st.global", "global stores"),
              ("ld.shared", "shared-memory loads"), ("st.shared", "shared-memory stores"), ("cp.async", "async copies"),
              ("mma", "matrix instructions"), ("wgmma", "warpgroup matrix instructions"), ("bar", "barriers")]
    lines = []
    for pre_, what in groups:
        n = sum(v for k, v in mix.items() if k.startswith(pre_))
        if n:
            vec = sorted({k for k in mix if k.startswith(pre_)}, key=lambda k: -mix[k])[0]
            lines.append(f"{n:>6}  {what:<30} e.g. {vec}")
    lines.append(f"{sum(mix.values()):>6}  instructions in the module")
    body = [ln.strip() for ln in e["ptx"].splitlines()]
    start = next((i for i, ln in enumerate(body) if ln.startswith("fma.rn")), None)
    if start is not None:
        lines += ["", "the first fused multiply-adds, as emitted:"] + ["    " + ln for ln in body[start:start + 6]]
    label = f"Generated code <span>the emitted PTX for {esc(target)}</span>"
    return pre("\n".join(lines), label=label)


def k_graph(a: dict) -> str:
    p = ROOT / a["file"]
    counts = {}
    for stage in ("lowered", "rewritten"):
        r = sh([exe(), "dump-ir", p.name, f"--stage={stage}", "--static=off"], cwd=p.parent)
        ops = Counter(re.findall(r"op: (\w+)", r.stdout))
        objs = len(re.findall(r"^\s+kind: \w+,$", r.stdout, re.M))
        counts[stage] = (objs, ops)
    show = ["Map", "Fold", "Index", "Phi"] if a.get("short") else ["Map", "Fold", "Iota", "Index", "Add", "Mul", "Div", "Mod", "Gt", "Phi", "Widen", "Print"]
    lo, rw = counts["lowered"], counts["rewritten"]
    lines = [f"values (objects)   lowered {lo[0]:>4}   rewritten {rw[0]:>4}"]
    for op in show:
        if lo[1].get(op) or rw[1].get(op):
            lines.append(f"{op:<18} lowered {lo[1].get(op, 0):>4}   rewritten {rw[1].get(op, 0):>4}")
    lines.append(f"all operations     lowered {sum(lo[1].values()):>4}   rewritten {sum(rw[1].values()):>4}")
    out = pre("\n".join(lines), label=f"The graph <span>mapal dump-ir --stage=lowered|rewritten {esc(Path(a['file']).name)}</span>")
    if a.get("mermaid"):
        out += "\n" + condensed_block(a) + "\n" + full_diagram_blocks(a) + "\n" + flow_block(a) + "\n" + full_graph_block(a)
    return out


def k_syntax(a: dict) -> str:
    sys.path.insert(0, str(Path(__file__).parent))
    import highlight_mapal as hm

    p = ROOT / a["file"]
    hl = hm.Highlighter(exe())
    try:
        segs = hl.segments(p.as_uri(), p.read_text(encoding="utf-8"))
    finally:
        hl.close()
    c = Counter(cls for _s, _e, cls, _i in segs)
    top = ", ".join(f"{n} {k.replace('-', ' ')}" for k, n in c.most_common(8))
    return pre(f"{sum(c.values())} tokens, every one classed by the language server:\n{top}",
               label="Syntax <span>mapal lsp (semantic tokens)</span>")


def k_llama(a: dict) -> str:
    target = a.get("target", "rtx-5080")
    e = emit("examples/llama.mapal", target)
    ks = kernels(e["dev"])
    c = Counter(k.get("kind") for k in ks)
    flows = [k for k in ks if k.get("kind") == "flow"]
    steps = sum(len(k["flow"].get("members", "").split(";")) for k in flows if k["flow"].get("members"))
    lines = [static_lines(e["report"]), f"{len(ks)} device kernels:"]
    lines += [f"{n:>4}  {kind}" for kind, n in c.most_common()]
    lines.append(f"the {len(flows)} flow kernels each carry several steps ({steps} steps in all)")
    return pre("\n".join(lines), label=f"Whole model, compile only <span>mapal build --emit-only --target={esc(target)} examples/llama.mapal</span>")


def k_chart(a: dict) -> str:
    out = []
    for theme in ("dark", "light"):
        svg = (CHARTS / f"{a['name']}-{theme}.svg").read_text(encoding="utf-8")
        svg = re.sub(r"<\?xml.*?\?>\s*", "", svg)
        # ids are page-global: make each chart's own
        pre_id = f"{a['name']}-{theme}-"
        for ident in set(re.findall(r'\bid="([^"]+)"', svg)):
            svg = svg.replace(f'id="{ident}"', f'id="{pre_id}{ident}"').replace(f'url(#{ident})', f'url(#{pre_id}{ident})')
            svg = svg.replace(f'aria-labelledby="{ident}', f'aria-labelledby="{pre_id}{ident}')
        svg = re.sub(r'aria-labelledby="([^"]+)"', lambda m: 'aria-labelledby="' + " ".join(
            (x if x.startswith(pre_id) else pre_id + x) for x in m.group(1).split()) + '"', svg)
        out.append(f'<div class="chart chart-{theme}">{svg.strip()}</div>')
    return "\n".join(out)


def k_perf(a: dict) -> str:
    sys.path.insert(0, str(Path(__file__).parent))
    import perf_charts

    return perf_charts.KINDS[a["name"]](a)


def mermaid_of(path: Path, form: str, stage: str = "rewritten") -> str:
    """The compiler's own Mermaid text for the graph: `form` is `flow`, `full`, or `category` / `string`
    (each optionally `,condensed`)."""
    flag = "--mermaid=graph" if form == "full" else f"--mermaid={form}"
    return sh([exe(), "dump-ir", path.name, f"--stage={stage}", flag], cwd=path.parent).stdout.rstrip("\n")


def live_url(text: str) -> str:
    """mermaid.live's editor URL for a diagram (its `#base64:` state form, URL-safe, unpadded)."""
    state = json.dumps({"code": text, "mermaid": json.dumps({"theme": "dark"}), "autoSync": True, "updateDiagram": True},
                       ensure_ascii=False, separators=(",", ":"))
    return "https://mermaid.live/edit#base64:" + base64.urlsafe_b64encode(state.encode()).decode().rstrip("=")


def mermaid_div(text: str, cls: str = "mermaid") -> str:
    return (f'<div class="{cls}">\n{esc(text)}\n</div>\n'
            f'<p class="mlive"><a href="{live_url(text)}" target="_blank" rel="noopener">Open in mermaid.live</a></p>')


def flow_block(a: dict) -> str:
    p = ROOT / a["file"]
    cmd = f"mapal dump-ir --mermaid=flow {p.name}"
    return (f'<div class="io-h">The flow <span>{esc(cmd)}</span></div>\n'
            + mermaid_div(mermaid_of(p, "flow")))


def k_flow(a: dict) -> str:
    return flow_block(a)


ZOOM_BAR = ('<div class="zoombox"><div class="zbar"><button type="button" data-z="out" aria-label="Zoom out">&minus;</button>'
            '<button type="button" data-z="in" aria-label="Zoom in">+</button><button type="button" data-z="fit">Fit</button></div>\n')


def full_graph_block(a: dict) -> str:
    p = ROOT / a["file"]
    cmd = f"mapal dump-ir --mermaid=graph {p.name}"
    text = mermaid_of(p, "full")
    nodes = len(re.findall(r"\(\(", text))
    return (f'<details class="gfull"><summary>The stored graph, every object and operation ({nodes} nodes)</summary>\n'
            f'<div class="io-h">The stored graph <span>{esc(cmd)}</span></div>\n'
            + ZOOM_BAR + mermaid_div(text) + "\n</div></details>")


DIAGRAMS = (("string", "String diagram",
             "Steps are boxes and values are wires: a product is just several wires entering one box, "
             "and a value read twice splits at a Δ."),
            ("category", "Category diagram",
             "Objects are nodes and steps are arrows, as the spec draws a category: a product is one pairing from Γ, "
             "the context the function runs in."))


def diagram_cmd(p: Path, form: str) -> str:
    """The command that prints `form`; the condensed string diagram is the bare `--mermaid` default."""
    flag = "--mermaid" if form == "string,condensed" else f"--mermaid={form}"
    return f"mapal dump-ir {flag} {p.name}"


def k_diagrams(a: dict) -> str:
    """The hero's two readings of the compiler's graph, condensed, as a two-tab switch."""
    p = ROOT / a["file"]
    n = a["id"]
    bar = "".join(
        f'<button type="button" role="tab" id="{n}-t{i}" aria-selected="{"true" if i == 0 else "false"}" '
        f'aria-controls="{n}-p{i}"{"" if i == 0 else " tabindex=\"-1\""}>{title}</button>'
        for i, (_, title, _) in enumerate(DIAGRAMS))
    panes = "".join(
        f'<div class="pane" id="{n}-p{i}" role="tabpanel" aria-labelledby="{n}-t{i}"{"" if i == 0 else " hidden"}>'
        f'<p class="dcap">{note}</p>\n'
        f'<div class="io-h">The {"default" if i == 0 else "condensed"} form <span>{esc(diagram_cmd(p, form + ",condensed"))}</span></div>\n'
        + mermaid_div(mermaid_of(p, form + ",condensed")) + "\n</div>"
        for i, (form, _, note) in enumerate(DIAGRAMS))
    return (f'<div class="dtabs"><div class="tabs" role="tablist" aria-label="How to draw the graph">{bar}</div>\n{panes}</div>')


def condensed_block(a: dict) -> str:
    p = ROOT / a["file"]
    return (f'<div class="io-h">The default diagram <span>{esc(diagram_cmd(p, "string,condensed"))}</span></div>\n'
            + mermaid_div(mermaid_of(p, "string,condensed")))


def full_diagram_blocks(a: dict) -> str:
    """The full category and string diagrams, each a collapsed zoom box."""
    p = ROOT / a["file"]
    out = []
    for form, title, _ in DIAGRAMS:
        text = mermaid_of(p, form)
        out.append(f'<details class="gfull"><summary>The full {title.lower()}, every object and operation</summary>\n'
                   f'<div class="io-h">The full {title.lower()} <span>{esc(diagram_cmd(p, form))}</span></div>\n'
                   + ZOOM_BAR + mermaid_div(text) + "\n</div></details>")
    return "\n".join(out)


def k_gpuline(a: dict) -> str:
    """One plain sentence on what the compiler did with the program at the sizes given by `scale`."""
    target = a.get("target", "rtx-5080")
    ks = kernels(emit(a["file"], target, a.get("scale"))["dev"])
    tiles = [k for k in ks if k.get("kind") in ("tile", "mma")]
    if not tiles:
        return "<p>At this size the compiler keeps the program on the CPU.</p>"
    t = tiles[0]
    n = len(ks)
    return (f"<p>At the full size the compiler turned the whole computation into {'one GPU kernel' if n == 1 else f'{n} GPU kernels'}, "
            f"computing the product in {t['tm']} × {t['tn']} tiles (compile only, for an RTX 5080). "
            f"The same source also runs on the CPU, as the output above.</p>")


KINDS = {"diagrams": k_diagrams, "flow": k_flow, "gpuline": k_gpuline, "perf": k_perf, "run": k_run, "derive": k_derive, "gpus": k_gpus, "ptxmix": k_ptxmix, "graph": k_graph,
         "syntax": k_syntax, "llama": k_llama, "chart": k_chart}


def regenerate(page: str) -> tuple[str, list[str]]:
    stale = []

    def sub(m: re.Match) -> str:
        a = attrs_of(m.group("attrs"))
        body = "\n" + KINDS[a["kind"]](a) + "\n"
        if body != m.group("body"):
            stale.append(a.get("id", a["kind"]))
        return f"<!-- exposition:gen {m.group('attrs')} begin -->{body}<!-- exposition:gen end -->"

    return REGION.sub(sub, page), stale


def check_excerpts(page: str) -> list[str]:
    bad = []
    for m in HLREGION.finditer(page):
        a = attrs_of(m.group("attrs"))
        if "excerpt" not in a:
            continue
        path, rng = a["excerpt"].split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        want = "\n".join((ROOT / path).read_text(encoding="utf-8").splitlines()[lo - 1:hi])
        got = html.unescape(re.sub(r"<[^>]*>", "", m.group("body")))
        got = re.sub(r"^\s*<pre[^>]*><code>", "", got).strip("\n")
        if got.strip() != want.strip():
            bad.append(a.get("id", path))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--page")
    args = ap.parse_args()
    page_path = Path(args.page).resolve() if args.page else PAGE
    page = page_path.read_text(encoding="utf-8")
    try:
        out, stale = regenerate(page)
    finally:
        for f in CACHE.get("scaled", []):
            f.unlink(missing_ok=True)
    bad = check_excerpts(page)
    if bad:
        print(f"excerpt differs from its file: {', '.join(bad)}")
    if args.check:
        if stale or bad:
            print(f"stale: {', '.join(stale)} (run docs/tools/exposition.py)" if stale else "")
            if stale:
                for a, b in zip(page.splitlines(), out.splitlines()):
                    if a != b:
                        print("  first difference:\n   -", a[:160], "\n   +", b[:160])
                        break
            return 1
        print(f"current: {len(REGION.findall(page))} generated regions, excerpts match")
        return 0
    if out != page:
        page_path.write_text(out, encoding="utf-8")
    print(f"{len(REGION.findall(page))} regions, {len(stale)} rewritten" + (f": {', '.join(stale)}" if stale else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
