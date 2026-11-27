"""Resolve an `at` string ("<path>#<kind> <Name>") to the line where the item is defined.

Files are being split into submodules, so the named file AND the directory of the same stem
(`func/gpu.rs` -> `func/gpu/**/*.rs`) are searched; for a C function the `.c` and `.h` files of
the file's directory. An `at` is good when exactly one place matches.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

VIS = r"(?:pub(?:\([^)]*\))?\s+)?"
KINDS = {  # kind -> keyword regex placed before the name
    "struct": r"(?:unsafe\s+)?struct", "enum": r"enum", "trait": r"(?:unsafe\s+)?trait", "type": r"type",
    "const": r"const", "static": r"static(?:\s+mut)?", "mod": r"mod",
}
FN_HEAD = r"(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"C\"\s+)?fn"


def parse(at: str):
    """(path, kind, name, nth): `nth` is the 1-based pick `Name@2` makes when a name is defined once per
    platform or feature variant (0: the name must be unique)."""
    m = re.match(r"^([^#\s]+)#(\w+)(?:\s+(.+))?$", at.strip())
    if not m:
        return None
    name = (m.group(3) or "").strip()
    pick = re.match(r"^(.*)@(\d+)$", name)
    if pick:
        return m.group(1), m.group(2), pick.group(1).strip(), int(pick.group(2))
    return m.group(1), m.group(2), name, 0


@lru_cache(maxsize=None)
def lines_of(rel: str) -> tuple[str, ...]:
    p = ROOT / rel
    try:
        return tuple(p.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return ()


C_SUFFIXES = (".c", ".h", ".inc")
ROOT_FILES = ("lib.rs", "main.rs", "mod.rs")


def tiers(rel: str, kind: str) -> list[list[str]]:
    """Where to look, nearest first; the first tier with a match wins. Files get split into submodules, so an
    item named under `x.rs` may now live in `x/**` — and under a crate root or `mod.rs`, anywhere in its
    directory. C files split into `.inc` parts beside them."""
    p = ROOT / rel
    rels = lambda fs: [str(f.relative_to(ROOT)) for f in fs if str(f.relative_to(ROOT)) != rel]
    if p.suffix in C_SUFFIXES:
        return [[rel], rels(sorted(f for f in p.parent.glob("*") if f.suffix in C_SUFFIXES))]
    out = [[rel]] if p.is_file() else []
    stem = p.with_suffix("")
    if p.suffix == ".rs" and stem.is_dir():
        out.append(rels(sorted(stem.rglob("*.rs"))))
    if p.name in ROOT_FILES:
        out.append(rels(sorted(p.parent.rglob("*.rs"))))
    return out


def candidates(rel: str, kind: str) -> list[str]:
    """Every file any tier names (diagnostics only)."""
    return [f for t in tiers(rel, kind) for f in t]


def strip_generics(name: str) -> str:
    return re.sub(r"<.*", "", name).strip()


def find_items(files: list[str], kind: str, name: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int, int]] = []
    nm = re.escape(strip_generics(name))
    if kind == "fn" and files and files[0].endswith(".js"):
        rx = re.compile(rf"^(?P<ind>\s*)(?:async\s+)?function\s+{nm}\b")
    elif kind == "fn":
        rx = re.compile(rf"^(?P<ind>\s*){VIS}{FN_HEAD}\s+{nm}\b")
    elif kind == "impl":
        words = r"\s+".join(re.escape(w) for w in name.split())
        rx = re.compile(rf"^(?P<ind>\s*)(?:unsafe\s+)?impl(?:<[^>]*>)?\s+(?:!?{words})\b")
    elif kind in KINDS:
        c_def = r"\s*\{" if files and files[0].endswith(C_SUFFIXES) else ""  # C: a definition has its brace
        rx = re.compile(rf"^(?P<ind>\s*)(?:static\s+)?{VIS}{KINDS[kind]}\s+{nm}\b{c_def}")
    else:
        return []
    for f in files:
        for i, line in enumerate(lines_of(f), 1):
            m = rx.match(line)
            if m:
                hits.append((f, i, len(m.group("ind"))))
    if kind in ("fn", "type", "const", "static") and hits:
        top = [h for h in hits if h[2] == 0]  # a free item beats an associated one of the same name
        hits = top or hits
    return [(f, i) for f, i, _ in hits]


def find_method(files: list[str], owner: str, meth: str) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    own = re.escape(strip_generics(owner))
    impl = re.compile(rf"^(?:unsafe\s+)?impl(?:<[^>]*>)?\s+(?:[\w:<>, ']+\s+for\s+)?&?(?:\w+::)*{own}\b")
    trait = re.compile(rf"^{VIS}(?:unsafe\s+)?trait\s+{own}\b")
    fnx = re.compile(rf"^\s+{VIS}{FN_HEAD}\s+{re.escape(meth)}\b")
    for f in files:
        inside = False
        for i, line in enumerate(lines_of(f), 1):
            if not inside and (impl.match(line) or trait.match(line)) and not line.rstrip().endswith(";"):
                inside = not line.rstrip().endswith("}")
                continue
            if inside:
                if line.startswith("}"):
                    inside = False
                elif fnx.match(line):
                    hits.append((f, i))
    return hits


def find_variant(files: list[str], enum: str, var: str) -> list[tuple[str, int]]:
    """A variant of `enum`: a line at the variant indent inside the enum's braces."""
    head = re.compile(rf"^{VIS}enum\s+{re.escape(enum)}\b")
    row = re.compile(rf"^ {{4}}{re.escape(var)}\b\s*(?:[,{{(]|$)")
    hits = []
    for f in files:
        inside = False
        for i, line in enumerate(lines_of(f), 1):
            if not inside and head.match(line):
                inside = True
            elif inside:
                if line.startswith("}"):
                    inside = False
                elif row.match(line):
                    hits.append((f, i))
    return hits


def find_cfn(files: list[str], name: str) -> list[tuple[str, int]]:
    rx = re.compile(rf"^(?!\s)(?!#)[\w\s\*]*?[\s\*]{re.escape(name)}\s*\(")
    hits = []
    for f in files:
        for i, line in enumerate(lines_of(f), 1):
            if rx.match(line) and not line.rstrip().endswith(";") and "=" not in line.split("(")[0]:
                hits.append((f, i))
    return hits


def resolve(at: str) -> list[tuple[str, int]]:
    """All places the item matches (a good `at` has exactly one)."""
    parsed = parse(at)
    if not parsed:
        return []
    rel, kind, name, nth = parsed
    if kind == "file":
        return [(rel, 1)] if (ROOT / rel).is_file() else []
    hits: list[tuple[str, int]] = []
    for files in tiers(rel, kind):
        if kind == "method":
            owner, _, meth = name.rpartition("::")
            hits = find_method(files, owner, meth)
        elif kind == "variant":
            enum, _, var = name.partition("::")
            hits = find_variant(files, enum, var)
        elif kind == "cfn":
            for f in files:  # one file at a time: a runtime and its emulator share function names
                hits = find_cfn([f], name)
                if hits:
                    break
        else:
            hits = find_items(files, kind, name)
        if hits:
            break
    is_test = lambda f: f.endswith("tests.rs") or "/tests/" in f
    if len(hits) > 1 and not is_test(rel):  # a definition beats a test helper of the same name
        hits = [h for h in hits if not is_test(h[0])] or hits
    if nth:
        return hits[nth - 1:nth]
    return hits
