#!/usr/bin/env python3
"""The TextMate grammar against the shared fixture, and the extension's semantic map.

    python3 editors/vscode/test/scope_test.py

Every token of editors/test/fixture.mapal must get the scope its `static` class names in
editors/test/classes.tsv (editors/test/fixture.tokens.tsv lists them) — the same file the
Vim syntax and the language server are tested against, so the editors cannot drift apart.
package.json's semanticTokenScopes must map each semantic token back to its class's scope.

Why a hand-rolled driver: the real tokenizer is vscode-textmate, which needs npm. This
models the one rule that decides outcomes for a flat pattern list —

    at each scan position, the winning pattern is the one whose match starts EARLIEST;
    ties are broken by the order patterns are listed —

and captures (named, or with their own patterns applied inside the captured text). That
"earliest" clause is the whole game: a rule written as `(?<=->)\\s*(ident)` begins its match
at the whitespace after the arrow, one column EARLIER than the keyword rule that matches
the identifier itself, so it wins on position no matter how the patterns are ordered.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
GRAMMAR = ROOT / "editors/vscode/syntaxes/mapal.tmLanguage.json"
PACKAGE = ROOT / "editors/vscode/package.json"
CLASSES = ROOT / "editors/test/classes.tsv"
FIXTURE = ROOT / "editors/test/fixture.mapal"
TOKENS = ROOT / "editors/test/fixture.tokens.tsv"


def tsv(path):
    return [l.split("\t") for l in path.read_text().splitlines() if l and not l.startswith("#")]


def compile_rule(r):
    """A rule node → a matcher. begin/end (strings) is modelled single-line."""
    if "match" in r:
        out = {**r, "_rx": re.compile(r["match"])}
    elif "begin" in r:
        out = {**r, "_rx": re.compile(r["begin"] + r'(?:\\.|[^"\\])*' + r["end"])}
        out["_inner"] = [compile_rule(q) for q in r.get("patterns", [])]
    else:
        raise ValueError(r)
    caps = {}
    for k, cap in r.get("captures", {}).items():
        caps[int(k)] = {
            "name": cap.get("name"),
            "rules": [compile_rule(q) for q in cap.get("patterns", [])],
        }
    out["_caps"] = caps
    return out


def load_rules(grammar):
    """The top-level include list flattened into ordered concrete rules: a node with its
    own match/begin IS the rule; only a pure container is expanded via `patterns`."""
    repo, rules = grammar["repository"], []
    for inc in grammar["patterns"]:
        node = repo[inc["include"][1:]]
        if "match" in node or "begin" in node:
            rules.append(compile_rule(node))
        else:
            rules.extend(compile_rule(r) for r in node.get("patterns", []))
    return rules


def tokenize(line, rules, start=0, end=None):
    """Earliest match wins, ties by listed order. Returns [(start, end, scope)]."""
    end = len(line) if end is None else end
    out, pos = [], start
    while pos < end:
        best = None
        for rule in rules:
            m = rule["_rx"].search(line, pos, end)
            if m and m.end() > m.start() and (best is None or m.start() < best[1].start()):
                best = (rule, m)
        if best is None:
            break
        rule, m = best
        if rule.get("name"):
            out.append((m.start(), m.end(), rule["name"]))
        for gi, cap in rule["_caps"].items():
            try:
                s, e = m.span(gi)
            except IndexError:
                continue
            if s == -1:
                continue
            if cap["name"]:
                out.append((s, e, cap["name"]))
            if cap["rules"]:
                out.extend(tokenize(line, cap["rules"], s, e))
        if rule.get("_inner"):
            out.extend(tokenize(line, rule["_inner"], m.start() + 1, m.end() - 1))
        pos = max(m.end(), pos + 1)
    return out


def scope_at(scopes, col):
    """Innermost (narrowest) covering scope wins — an escape inside a string is the
    escape, not the string."""
    covering = [(e - s, name) for s, e, name in scopes if s <= col < e]
    return min(covering)[1] if covering else "-"


def main():
    grammar = json.loads(GRAMMAR.read_text())
    rules = load_rules(grammar)
    classes = {r[0]: r for r in tsv(CLASSES) if r[0] != "class"}
    lines = FIXTURE.read_text().splitlines()
    scopes = [tokenize(l, rules) for l in lines]
    fails = checked = 0

    for lnum, col, text, cls, static in tsv(TOKENS):
        want_class = cls if static == "=" else static
        want = classes[want_class][2]
        got = scope_at(scopes[int(lnum) - 1], int(col) - 1)
        checked += 1
        if got != want:
            fails += 1
            print(f"FAIL {lnum}:{col} {text:<12} {want_class:<14} want {want:<34} got {got}")
    print(f"{checked} tokens checked against fixture.tokens.tsv")

    # `category` is flagged; nothing in a valid example is.
    if scope_at(tokenize("category Pixel {", rules), 0) != "invalid.illegal.reserved.mapal":
        fails += 1
        print("FAIL `category` is not flagged as reserved")
    for f in sorted((ROOT / "examples").glob("*.mapal")):
        for i, l in enumerate(f.read_text().splitlines(), 1):
            if any(n == "invalid.illegal.reserved.mapal" for _, _, n in tokenize(l, rules)):
                fails += 1
                print(f"FAIL {f.name}:{i} flagged as reserved")

    # The rules that start at the whitespace after an arrow must exclude every keyword and
    # builtin, or they claim it (see the module docstring).
    repo = grammar["repository"]
    kw = repo["keywords"]["patterns"]
    names = set()
    for p in kw:
        names |= set(re.search(r"\\b\(([^)]+)\)\\b", p["match"]).group(1).split("|"))
    prims = set(re.search(r"\\b\(([^)]+)\)\\b", kw[-1]["match"]).group(1).split("|"))
    must = (names - prims) | {"category"}
    pats = [("call-position", repo["call-position"]["match"]), ("binding", repo["binding"]["match"])]
    pats += [(f"guards[{i}]", p["match"]) for i, p in enumerate(repo["guards"]["patterns"])]
    for rule, pat in pats:
        listed = set(re.search(r"\(\?!\(\?:([^)]+)\)", pat).group(1).split("|"))
        if must - listed:
            fails += 1
            print(f"FAIL #{rule} exclusion is missing {sorted(must - listed)}")
    print(f"ok   the arrow rules exclude all {len(must)} keywords and builtins" if not fails else "")

    # semanticTokenScopes: each semantic token → its class's TextMate scope.
    package = json.loads(PACKAGE.read_text())
    mapped = {}
    for entry in package["contributes"].get("semanticTokenScopes", []):
        if entry.get("language") == "mapal":
            mapped = entry["scopes"]
    want_map = {r[3]: [r[2]] for r in classes.values() if r[3] != "-"}
    if mapped != want_map:
        fails += 1
        print("FAIL package.json semanticTokenScopes differ from classes.tsv:")
        for k in sorted(set(mapped) | set(want_map)):
            if mapped.get(k) != want_map.get(k):
                print(f"     {k}: package.json {mapped.get(k)}, classes.tsv {want_map.get(k)}")
    else:
        print(f"ok   semanticTokenScopes map all {len(want_map)} semantic tokens to their classes")

    print("all scope assertions passed" if not fails else f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
