"""The page as a neutral list of blocks (headings, paragraphs, diagrams, tables); html.py and markdown.py draw them.

Inline items: a plain string, ("code", s), ("ref", path, line), ("link", text, anchor), ("b", s), ("br",).
Blocks:
  ("h", level, text, anchor)      ("p", inlines)            ("diagram", text, links, alt)
  ("table", headers, rows, cls)   rows = [(anchor|None, [cell, ...])], a cell is a list of inlines
  ("details", summary, blocks, open)                          ("chips", [inline, ...])
"""

from __future__ import annotations

from . import diagrams as dg
from .model import Model

KIND_TEXT = {
    "functor": "functor", "process": "process", "natural transformation": "guarantee",
}


def ref(m: Model, at: str):
    p, ln = m.loc(at)
    return ("ref", p, ln)


def cats_chips(items):
    out = []
    for i, c in enumerate(items):
        if i:
            out.append(", ")
        out.append(("link", c, f"#cat-{c}"))
    return out or ["nothing"]


def end_cell(ends):
    out = []
    for i, e in enumerate(ends):
        if i:
            out.append(("br",))
        cat, _, det = e.partition(":")
        out.append(("link", cat, f"#cat-{cat}"))
        if det.strip():
            out += [": ", ("code", det.strip())]
    return out


def table(headers, rows, cls=""):
    return ("table", headers, rows, cls)


def fold(summary, blocks, n):
    return ("details", summary, blocks, n <= 14)


# ---------------------------------------------------------------- overview and memory
def overview_blocks(m: Model):
    text, links = dg.overview(m)
    rows = []
    for c in m.d["categories"]:
        owner = m.comps[c["owner"]]
        rows.append((None, [[("link", c["name"], f"#cat-{c['id']}")], [("link", owner["name"], f"#c-{owner['id']}")], [c["plain"]],
                            [f"{len(c['objects'])} objects, {len(c['morphisms'])} arrows"]]))
    return [
        ("h", 2, "The compiler as data", "data"),
        ("p", ["Each box is a ", ("b", "category"), ": a kind of data (text, tokens, the Flow graph, facts, machine code, a running program). "
               "Each arrow is a ", ("b", "transformation"), ": a stage of the compiler that turns one kind of data into another. Boxes are grouped by the component that defines them; click a box to see its objects."]),
        ("p", ["Inside a category, the things are ", ("b", "objects"), " and the ways of getting from one to another are ", ("b", "arrows"),
               " (morphisms). A ", ("b", "functor"), " maps a whole category into another, a ", ("b", "process"),
               " is one step with inputs and outputs, and a ", ("b", "guarantee"), " (natural transformation) says that two routes between the same ends give the same result."]),
        ("diagram", text, links, "The compiler as categories and the transformations between them"),
        table(["Category", "Defined under", "What it holds", "Size"], rows),
    ]


def memory_blocks(m: Model):
    text, links = dg.memory(m)
    rows = []
    for p in m.d["memory"]["places"]:
        cells = []
        for oid in p["objects"]:
            o = m.objs[oid]
            cells += [("link", o["name"], f"#o-{oid}"), " ", ref(m, o["at"]), ("br",)]
        rows.append((None, [[("b", p["name"])], [p["plain"]], cells]))
    return [
        ("h", 2, "Where data lives", "memory"),
        ("p", ["The same objects exist in different places as the program goes from text to a running kernel. "
               "Each box below holds the main objects that live there; click one for its definition."]),
        ("diagram", text, links, "The memory places and the main objects in each"),
        table(["Place", "What is kept there", "Main objects (with where they are defined)"], rows),
    ]


# ---------------------------------------------------------------- components
def obj_table(m: Model, cat: dict):
    show_via = any(set(o.get("via", [])) - {cat["owner"]} for o in cat["objects"])
    heads = ["Object", "What it is", "Defined at", "Lives in"] + (["Provided by"] if show_via else [])
    rows = []
    for o in cat["objects"]:
        cells = [[("b", o["name"])], [o["plain"]], [ref(m, o["at"])], [o["lives"]]]
        if show_via:
            others = [v for v in o.get("via", []) if v != cat["owner"]]
            cells.append([x for i, v in enumerate(others) for x in ((", ",) if i else ()) + (("link", m.comps[v]["name"], f"#c-{v}"),)])
        rows.append((f"o-{o['id']}", cells))
    return table(heads, rows, "objects")


def morph_table(m: Model, cat: dict):
    rows = [(None, [[("b", x["name"])], [("code", x["from"]), " → ", ("code", x["to"])], [x["plain"]], [ref(m, x["at"])]])
            for x in cat["morphisms"]]
    return table(["Arrow", "From → to", "What it does", "Defined at"], rows, "morphisms")


def flow_category_blocks(m: Model):
    fc = m.d["flow_category"]
    text, links = dg.flow_category(m)
    kinds = [(None, [[("b", k["name"])], [k["plain"]], [ref(m, k["at"])]]) for k in fc["object_kinds"]]
    ops = [(f"op-{o['name']}", [[("b", o["name"])], [("code", next(v["label"] for v in fc["values"] if v["id"] == o["from"])), " → ",
                                                    ("code", next(v["label"] for v in fc["values"] if v["id"] == o["to"]))], [o["plain"]], [ref(m, o["at"])]])
           for o in fc["operations"]]
    notes = [(None, [[("b", n["name"])], [n["plain"]], [x for i, a in enumerate(n["at"]) for x in ((("br",),) if i else ()) + (ref(m, a),)]]) for n in fc["notes"]]
    return [("h", 4, "Flow as Mapal's own category: values and operations", "flow-category"), ("p", [fc["plain"]]),
            ("diagram", text, links, "Kinds of value and the operations between them"),
            table(["Product, coproduct, loop", "What it is", "Where it lives"], notes),
            fold(f"Operations, the morphisms of Flow ({len(ops)})", [table(["Operation", "From → to", "What it does", "Defined at"], ops, "ops")], len(ops)),
            fold(f"Kinds of object ({len(kinds)})", [table(["Kind", "What it is", "Defined at"], kinds)], len(kinds))]


def category_blocks(m: Model, cat: dict):
    out = [("h", 4, f"Category {cat['name']}", f"cat-{cat['id']}"), ("p", [cat["plain"]])]
    if cat["id"] == "Flow":
        out += flow_category_blocks(m)
    d = dg.cat_diagram(m, cat["id"])
    if d:
        out.append(("diagram", d[0], d[1], f"Objects and arrows of the {cat['name']} category"))
    if cat["objects"]:
        out.append(fold(f"Objects ({len(cat['objects'])})", [obj_table(m, cat)], len(cat["objects"])))
    if cat["morphisms"]:
        out.append(fold(f"Arrows ({len(cat['morphisms'])})", [morph_table(m, cat)], len(cat["morphisms"])))
    return out


def trans_table(m: Model, comp: dict, part: str | None):
    rows = []
    for t in comp["transformations"]:
        if part and t.get("part") != part:
            continue
        made = []
        if t.get("steps"):
            made = [("br",), ("b", "made of: ")]
            for i, s in enumerate(t["steps"]):
                made += ([", "] if i else []) + [("link", m.trans[s]["name"], f"#t-{s}")]
        rows.append((f"t-{t['id']}", [[("b", t["name"])], end_cell(t["from"]) + [("br",), "→ "] + end_cell(t["to"]),
                                      [t["kind"]], [t["plain"]] + made, [ref(m, t["at"])], [t["lives"]]]))
    return rows


def pat_rows(m: Model, comp: dict, part: str | None):
    rows = []
    for p in comp["patterns"]:
        if part and p.get("part") != part:
            continue
        ub = []
        for i, u in enumerate(p["used_by"]):
            name = m.trans[u]["name"] if u in m.trans else m.guars[u]["name"]
            anchor = f"#t-{u}" if u in m.trans else f"#g-{u}"
            ub += ([", "] if i else []) + [("link", name, anchor)]
        rows.append((None, [[("b", p["name"])], [p["kind"]], [p["plain"]], [ref(m, p["at"])], ub]))
    return rows


def part_block(m: Model, comp: dict, part: dict | None):
    pid = part["id"] if part else None
    out = []
    if part:
        out += [("h", 4, part["name"], f"part-{pid}"), ("p", [part["plain"]])]
    d = dg.comp_transformations(m, comp["id"], pid) if part else None
    if d:
        out.append(("diagram", d[0], d[1], f"Transformations of {part['name']}"))
    rows = trans_table(m, comp, pid)
    if rows:
        out.append(fold(f"Transformations ({len(rows)})", [table(["Name", "From → to", "Kind", "What it does", "Defined at", "Lives in"], rows, "trans")], len(rows)))
    prow = pat_rows(m, comp, pid)
    if prow:
        out.append(fold(f"Algorithms, data structures and design patterns ({len(prow)})",
                        [table(["Name", "Kind", "What it is", "Defined at", "Used by"], prow, "pats")], len(prow)))
    return out


def component_blocks(m: Model, comp: dict):
    out = [("h", 3, comp["name"], f"c-{comp['id']}"), ("p", [comp["plain"]])]
    if comp["status"] == "built":
        out.append(("p", [("b", "Reads "), *cats_chips(comp["reads"]), ("b", "  →  produces "), *cats_chips(comp["produces"])]))
    if comp["status"] != "built":
        out.append(("p", [("b", "Not built yet. "), "The crate is a one-line placeholder: ", *[ref(m, a) for a in comp["code"]]]))
        return out
    d = dg.comp_transformations(m, comp["id"])
    if d:
        out.append(("diagram", d[0], d[1], f"What {comp['name']} does: transformations between categories"))
    gs = [g for g in m.d["guarantees"] if g["component"] == comp["id"]]
    if gs:
        out.append(("p", [("b", "Guarantees: ")] + [x for i, g in enumerate(gs) for x in ([", "] if i else []) + [("link", g["name"], f"#g-{g['id']}")]]))
    if comp["code"]:
        out.append(("details", f"Files ({len(comp['code'])})", [("p", [x for i, a in enumerate(comp["code"]) for x in ((("br",),) if i else ()) + (ref(m, a),)])], False))
    owned = [c for c in m.d["categories"] if c["owner"] == comp["id"]]
    for c in owned:
        out += category_blocks(m, c)
    if comp["parts"]:
        for part in comp["parts"]:
            out += part_block(m, comp, part)
    else:
        out += part_block(m, comp, None)
    gs = [g for g in m.d["guarantees"] if g["component"] == comp["id"]]
    if gs:
        rows = [(None, [[("link", g["name"], f"#g-{g['id']}")], [g["plain"]], [x for i, a in enumerate(g["checked"]) for x in ((("br",),) if i else ()) + (ref(m, a),)]]) for g in gs]
        out.append(("h", 4, f"Guarantees of {comp['name']}", f"cg-{comp['id']}"))
        out.append(table(["Guarantee (natural transformation)", "What must agree", "Checked in"], rows))
    return out


def components_blocks(m: Model):
    chips = [("link", c["name"], f"#c-{c['id']}") for c in m.d["components"]]
    out = [("h", 2, "Components, in pipeline order", "components"),
           ("p", ["Every component lists the categories it defines (its data), then the transformations it makes between categories, "
                  "then its guarantees and the algorithms, data structures and design patterns it uses. Every code reference is a link to the line, shown as ",
                  ("code", "path:line"), " so you can also open it with ", ("code", "nvim path +line"), "."]),
           ("chips", chips)]
    for c in m.d["components"]:
        out += component_blocks(m, c)
    return out


# ---------------------------------------------------------------- guarantees and index
def guarantees_blocks(m: Model):
    out = [("h", 2, "Guarantees", "guarantees"),
           ("p", ["A guarantee is a natural transformation: two routes between the same two ends that must give the same result. "
                  "Each square below is one such statement, with the test or gate that checks it."])]
    for g in m.d["guarantees"]:
        comp = m.comps[g["component"]]
        text, links = dg.guarantee(m, g)
        checked = [x for i, a in enumerate(g["checked"]) for x in ((", ",) if i else ()) + (ref(m, a),)]
        out += [("h", 4, g["name"], f"g-{g['id']}"), ("p", [g["plain"]]), ("diagram", text, links, f"Two routes that agree: {g['name']}"),
                ("p", [("b", "Checked in: "), *checked, "  ·  ", ("link", comp["name"], f"#c-{comp['id']}")] + ([("br",), g["note"]] if g.get("note") else []))]
    return out


def index_blocks(m: Model):
    rows = []
    for c in m.d["components"]:
        for p in c["patterns"]:
            comp_cell = [("link", c["name"], f"#c-{c['id']}")]
            if p.get("part"):
                comp_cell += [" / ", ("link", m.comps[c["id"]]["parts"][[x["id"] for x in c["parts"]].index(p["part"])]["name"], f"#part-{p['part']}")]
            ub = []
            for i, u in enumerate(p["used_by"]):
                name = m.trans[u]["name"] if u in m.trans else m.guars[u]["name"]
                ub += ([", "] if i else []) + [("link", name, f"#t-{u}" if u in m.trans else f"#g-{u}")]
            rows.append((None, [[("b", p["name"])], [p["kind"]], comp_cell, [p["plain"]], [ref(m, p["at"])], ub]))
    return [("h", 2, "Index of algorithms, data structures and design patterns", "index"),
            ("p", [f"{len(rows)} entries. Click a column header to sort; type to filter."]),
            table(["Name", "Kind", "Component", "What it is", "Defined at", "Used by"], rows, "sortable")]


def build(m: Model):
    return (overview_blocks(m) + memory_blocks(m) + components_blocks(m) + guarantees_blocks(m) + index_blocks(m))


def counts(m: Model) -> dict:
    d = m.d
    return dict(components=len(d["components"]), categories=len(d["categories"]),
                objects=sum(len(c["objects"]) for c in d["categories"]), morphisms=sum(len(c["morphisms"]) for c in d["categories"]),
                transformations=len(m.trans), guarantees=len(d["guarantees"]),
                patterns=sum(len(c["patterns"]) for c in d["components"]))
