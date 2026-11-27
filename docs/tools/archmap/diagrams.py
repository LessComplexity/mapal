"""Mermaid text for every diagram of the page, built from the model. Each builder returns (text, links):
`links` maps a node id to the page anchor the node jumps to."""

from __future__ import annotations

import re
from collections import defaultdict

from .model import Model

HUES = {
    "syntax": "#38bdf8", "lower": "#a78bfa", "check": "#f472b6", "flow": "#2dd4bf", "facts": "#fbbf24", "rewrite": "#fb923c",
    "interp": "#4ade80", "hardware": "#f87171", "machine": "#c084fc", "llvm": "#60a5fa", "cuda": "#34d399",
    "verilog": "#94a3b8", "runtime": "#facc15", "cli": "#e879f9", "lsp": "#22d3ee",
}
CLASSES = "\n".join(f"  classDef k_{k} fill:{v}30,stroke:{v},stroke-width:2px" for k, v in HUES.items()) + \
    "\n  classDef plain fill:#94a3b81f,stroke:#94a3b8,stroke-dasharray:4 3\n  classDef endpoint fill:#2dd4bf30,stroke:#2dd4bf,stroke-width:2px"


def q(s: str, n: int = 40) -> str:
    """A label safe inside a Mermaid quoted string, shortened."""
    s = re.sub(r"[`\"<>|{}\[\]]", "", s).replace("\n", " ").strip()
    if len(s) > n:
        s = s[: n - 1].rstrip() + "…"
    return s


def nid(prefix: str, i: int) -> str:
    return f"{prefix}{i}"


def overview(m: Model):
    lines = ["flowchart TD"]
    links: dict[str, str] = {}
    for comp in m.d["components"]:
        owned = [c for c in m.d["categories"] if c["owner"] == comp["id"]]
        if not owned:
            continue
        lines.append(f'  subgraph s_{comp["id"]}["{q(comp["name"], 34)}"]')
        for c in owned:
            lines.append(f'    {c["id"]}["{c["name"]}"]')
            links[c["id"]] = f"#cat-{c['id']}"
        lines.append("  end")
    for ov in m.d["overview"]:
        lines.append(f'  {ov["from"]} -->|"{q(ov["label"], 30)}"| {ov["to"]}')
    for comp in m.d["components"]:
        owned = [c["id"] for c in m.d["categories"] if c["owner"] == comp["id"]]
        if owned:
            lines.append(f"  class {','.join(owned)} k_{comp['id']}")
    lines.append(CLASSES)
    return "\n".join(lines), links


def memory(m: Model):
    lines = ["flowchart LR"]
    links: dict[str, str] = {}
    i = 0
    for p in m.d["memory"]["places"]:
        lines.append(f'  subgraph {p["id"].replace("-", "_")}["{q(p["name"], 40)}"]')
        lines.append("    direction TB")
        for oid in p["objects"]:
            i += 1
            n = f"m{i}"
            lines.append(f'    {n}["{q(m.objs[oid]["name"], 30)}"]')
            links[n] = f"#o-{oid}"
        lines.append("  end")
    for e in m.d["memory"]["edges"]:
        lines.append(f'  {e["from"].replace("-", "_")} -->|"{q(e["label"], 34)}"| {e["to"].replace("-", "_")}')
    lines.append(CLASSES)
    return "\n".join(lines), links


def _cat_of(end: str) -> str:
    return end.split(":")[0]


def comp_transformations(m: Model, comp_id: str, part: str | None = None, max_groups: int = 12):
    """Categories as nodes, the component's transformations as arrows (parallel ones grouped, labels shortened)."""
    comp = m.comps[comp_id]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in comp["transformations"]:
        if part and t.get("part") != part:
            continue
        for f in dict.fromkeys(_cat_of(x) for x in t["from"]):
            for to in dict.fromkeys(_cat_of(x) for x in t["to"]):
                groups[(f, to)].append(t["name"])
    if not groups:
        return None
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_groups]
    cats = sorted({c for (f, t), _ in ranked for c in (f, t)} | (set() if part else set(comp["reads"]) | set(comp["produces"])))
    lines = ["flowchart LR"]
    links: dict[str, str] = {}
    for c in cats:
        lines.append(f'  {c}["{c}"]')
        links[c] = f"#cat-{c}"
    for (f, t), names in ranked:
        lab = " / ".join(q(n, 26) for n in names[:2]) + (f" (+{len(names) - 2} more)" if len(names) > 2 else "")
        lines.append(f'  {f} -->|"{q(lab, 70)}"| {t}')
    for c in cats:
        lines.append(f"  class {c} k_{m.cats[c]['owner']}")
    lines.append(CLASSES)
    return "\n".join(lines), links


def _label(end: str) -> str:
    return re.sub(r"\s+", " ", end).strip()


def _match_object(label: str, objs: list[dict]) -> dict | None:
    lab = label.lower().strip("&* ").strip()
    for o in objs:
        parts = [p.strip().lower() for p in re.split(r"\s*/\s*|\s*,\s*", re.sub(r"\(.*?\)", "", o["name"]))]
        if lab == o["name"].lower() or lab in parts or lab.split(" ")[0] in parts:
            return o
    return None


def _grid(lines: list[str], ids: list[str], per_row: int = 4):
    """Boxes in rows (invisible links keep each row in a line): a compact grid."""
    for i in range(0, len(ids), per_row):
        lines.append("  " + " ~~~ ".join(ids[i:i + per_row]))


def cat_diagram(m: Model, cat_id: str, max_nodes: int = 15):
    """The category's own structure: objects as nodes, and only the arrows whose BOTH ends are objects of the category.
    Objects without such an arrow are drawn as a grid of boxes (up to `max_nodes` in all; the table has the rest)."""
    cat = m.cats[cat_id]
    objs = cat["objects"]
    if not objs:
        return None
    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mo in cat["morphisms"]:
        f, t = _match_object(_label(mo["from"]), objs), _match_object(_label(mo["to"]), objs)
        if f and t and f["id"] != t["id"]:
            edges[(f["id"], t["id"])].append(mo["name"])
    deg: dict[str, int] = defaultdict(int)
    for (f, t) in edges:
        deg[f] += 1
        deg[t] += 1
    linked = [o for o in sorted(objs, key=lambda o: -deg.get(o["id"], 0)) if deg.get(o["id"])][:max_nodes]
    keep = {o["id"] for o in linked}
    es = [(k, v) for k, v in edges.items() if k[0] in keep and k[1] in keep]
    keep = {i for (f, t), _ in es for i in (f, t)}
    linked = [o for o in linked if o["id"] in keep]
    rest = [o for o in objs if o["id"] not in keep][: max(0, max_nodes - len(linked))]
    lines = ["flowchart LR"]
    links: dict[str, str] = {}
    ids = {o["id"]: f"n{i}" for i, o in enumerate(linked + rest)}
    for o in linked + rest:
        lines.append(f'  {ids[o["id"]]}["{q(o["name"], 34)}"]')
        links[ids[o["id"]]] = f"#o-{o['id']}"
    for (f, t), names in es:
        lab = " / ".join(q(x, 24) for x in names[:2]) + (f" (+{len(names) - 2})" if len(names) > 2 else "")
        lines.append(f'  {ids[f]} -->|"{q(lab, 60)}"| {ids[t]}')
    _grid(lines, [ids[o["id"]] for o in rest])
    lines.append("  class " + ",".join(ids.values()) + f" k_{cat['owner']}")
    lines.append(CLASSES)
    return "\n".join(lines), links


def guarantee(m: Model, g: dict):
    """A commuting square: two paths from the start to the end that must give the same result."""
    lines = ["flowchart LR", f'  S(["{q(g["start"], 30)}"])', f'  T(["{q(g["end"], 30)}"])']
    for pi, path in enumerate(g["paths"]):
        prev = "S"
        for k in range(0, len(path), 2):
            edge = q(path[k], 40)
            nxt = "T" if k == len(path) - 1 else f"p{pi}n{k}"
            if nxt != "T":
                lines.append(f'  {nxt}["{q(path[k + 1], 34)}"]')
            lines.append(f'  {prev} -->|"{edge}"| {nxt}')
            prev = nxt
    mids = [x.split("[")[0].strip() for x in lines if re.match(r"\s+p\dn\d+\[", x)]
    if mids:
        lines.append("  class " + ",".join(mids) + " plain")
    lines.append("  class S,T endpoint")
    lines.append(CLASSES)
    return "\n".join(lines), {}


def flow_category(m: Model):
    """Mapal's own category: kinds of typed value as nodes, the operations of `Operation` as arrows between them."""
    fc = m.d["flow_category"]
    lines = ["flowchart LR"]
    for v in fc["values"]:
        lines.append(f'  {v["id"]}["{q(v["label"], 30)}"]')
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for op in fc["operations"]:
        groups[(op["from"], op["to"])].append(op["name"])
    for (f, t), names in groups.items():
        lab = ", ".join(names[:6]) + (f" +{len(names) - 6}" if len(names) > 6 else "")
        lines.append(f'  {f} -->|"{q(lab, 64)}"| {t}')
    lines.append("  class " + ",".join(v["id"] for v in fc["values"]) + " k_flow")
    lines.append(CLASSES)
    return "\n".join(lines), {}
