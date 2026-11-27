"""map.json: loading, reference checks, resolved code locations and the freshness hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .resolve import ROOT, resolve

MAP = ROOT / "docs" / "architecture" / "map.json"
PKG = Path(__file__).resolve().parent
REPO = "https://github.com/LessComplexity/mapal/blob/main/"


class Model:
    def __init__(self, path: Path = MAP):
        self.raw = path.read_bytes()
        d = json.loads(self.raw)
        self.d = d
        self.cats = {c["id"]: c for c in d["categories"]}
        self.comps = {c["id"]: c for c in d["components"]}
        self.objs = {o["id"]: dict(o, cat=c["id"]) for c in d["categories"] for o in c["objects"]}
        self.trans = {t["id"]: dict(t, comp=c["id"]) for c in d["components"] for t in c["transformations"]}
        self.guars = {g["id"]: g for g in d["guarantees"]}
        self.locs: dict[str, tuple[str, int]] = {}
        self.errors: list[str] = []
        self._resolve_all()
        self._check_refs()

    # ---- every `at` in the file, in file order
    def all_ats(self) -> list[str]:
        out: list[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == "at" and isinstance(v, str):
                        out.append(v)
                    elif k == "checked":
                        out.extend(v)
                    elif k == "code":
                        out.extend(v)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(self.d)
        return out

    def _resolve_all(self):
        for at in sorted(set(self.all_ats())):
            hits = resolve(at)
            if len(hits) != 1:
                self.errors.append(f"{at}: {len(hits)} matches")
            else:
                self.locs[at] = hits[0]

    def _check_refs(self):
        e = self.errors
        for c in self.cats.values():
            if c["owner"] not in self.comps:
                e.append(f"category {c['id']}: unknown owner {c['owner']}")
            for o in c["objects"]:
                for v in o.get("via", []):
                    if v not in self.comps:
                        e.append(f"object {o['id']}: unknown component {v}")
        for t in self.trans.values():
            for s in t.get("steps", []):
                if s not in self.trans:
                    e.append(f"transformation {t['id']}: step {s} does not exist")
            for end in t["from"] + t["to"]:
                if end.split(":")[0] not in self.cats:
                    e.append(f"transformation {t['id']}: unknown category in '{end}'")
        for c in self.comps.values():
            for p in c["patterns"]:
                for u in p["used_by"]:
                    if u not in self.trans and u not in self.guars:
                        e.append(f"pattern {p['name']}: used_by {u} does not exist")
            for r in c["reads"] + c["produces"]:
                if r not in self.cats:
                    e.append(f"component {c['id']}: unknown category {r}")
        for g in self.guars.values():
            if g["component"] not in self.comps:
                e.append(f"guarantee {g['id']}: unknown component")
            if len(g["paths"]) != 2 or any(len(p) % 2 == 0 for p in g["paths"]):
                e.append(f"guarantee {g['id']}: paths must be two alternating edge/node lists ending on an edge")
        fc = self.d["flow_category"]
        ids = {v["id"] for v in fc["values"]}
        for op in fc["operations"]:
            if op["from"] not in ids or op["to"] not in ids:
                e.append(f"flow operation {op['name']}: unknown value kind")
        for ov in self.d["overview"]:
            if ov["link"] not in self.trans and ov["link"] not in self.guars:
                e.append(f"overview link {ov['link']} does not exist")
        for p in self.d["memory"]["places"]:
            for o in p["objects"]:
                if o not in self.objs:
                    e.append(f"memory place {p['id']}: object {o} does not exist")

    # ---- locations
    def loc(self, at: str) -> tuple[str, int]:
        return self.locs[at]

    @staticmethod
    def url(path: str, line: int) -> str:
        return f"{REPO}{path}#L{line}"

    # ---- freshness
    def digest(self) -> str:
        h = hashlib.sha256(self.raw)
        for at in sorted(self.locs):
            p, ln = self.locs[at]
            h.update(f"\n{at}={p}:{ln}".encode())
        for f in sorted(PKG.glob("*.py")) + [PKG.parent / "archmap.py"]:
            h.update(f.read_bytes())
        return h.hexdigest()[:20]
