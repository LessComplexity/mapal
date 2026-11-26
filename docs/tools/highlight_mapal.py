#!/usr/bin/env python3
"""Highlight Mapal source in docs/architecture-explorer.html exactly as the editors do.

The classes are the editors' (editors/test/classes.tsv, shared by Neovim and VS Code):

* names: the language server's semantic tokens (`mapal lsp`, textDocument/semanticTokens/full,
  decoded with the legend of its initialize response, UTF-16 columns), each token mapped back to
  its class by the table's `lsp` column. A name the server sends no token for (a plain local read,
  or a name a fragment leaves unresolved) shows in the editors with its one-line class, the class
  both static grammars give it -- `static_name` in crates/mapal-lsp/tests/highlight.rs, mirrored
  here;
* everything else (comments, strings, numbers, keywords, arrows, guards, operators, size brackets,
  punctuation): the lexer's class -- `rows` in the same test, over a port of the mapal-syntax lexer.
  The port is checked token by token against editors/test/fixture.tokens.tsv (`--test`).

Colors: each class takes the color of the Vim group its syntax group links to
(editors/nvim/syntax/mapal.vim `hi default link`), from tokyonight -- LazyVim's default
colorscheme -- night on the dark code surface and day on the light one, lightness-adjusted where a
color misses WCAG AA (4.5:1) on that surface. `--palette` prints the table with its ratios.

The page holds generated regions between markers; the plain source stays recoverable -- the text
of a snippet's <pre> with the tags stripped and the entities decoded is the source, character for
character -- so a region is regenerated in place from what it already shows:

    <!-- mapal:hl begin id=NAME --><pre class="hl"><code>...</code></pre><!-- mapal:hl end -->
    <!-- mapal:hl begin id=NAME src=examples/FILE.mapal title="..." -->...<!-- mapal:hl end -->
    /* mapal:hl css begin ... */ ... /* mapal:hl css end */

A `src=` region is a whole file, regenerated from the file: an expandable <details id=NAME> with
line numbers (CSS counters, never copied), a copy button and a link to the file. To add a snippet,
wrap a plain `<pre><code>...</code></pre>` in begin/end markers with a new id and run the script.

Usage:
    docs/tools/highlight_mapal.py            regenerate every region of the page in place
    docs/tools/highlight_mapal.py --check    exit 1 when a region differs from a regeneration
    docs/tools/highlight_mapal.py --test     self-test: lexer vs the fixture, source round trip,
                                             page token classes vs the server, CSS + contrast
    docs/tools/highlight_mapal.py --palette  class -> group -> colors and contrast ratios
    docs/tools/highlight_mapal.py FILE.mapal print FILE highlighted (an HTML fragment)

The server: --server PATH, else $MAPAL_BIN, else `mapal` on PATH, else ~/.local/bin/mapal.
Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import bisect
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "architecture-explorer.html"
CLASSES = ROOT / "editors" / "test" / "classes.tsv"
GRAMMAR = ROOT / "editors" / "vscode" / "syntaxes" / "mapal.tmLanguage.json"
VIM = ROOT / "editors" / "nvim" / "syntax" / "mapal.vim"
FIXTURE = ROOT / "editors" / "test" / "fixture.mapal"
FIXTURE_TOKENS = ROOT / "editors" / "test" / "fixture.tokens.tsv"

# ---------------------------------------------------------------------------------------------
# The shared tables: classes, the language's name lists, the Vim group links.
# ---------------------------------------------------------------------------------------------


def load_classes() -> dict[str, tuple[str, str, str]]:
    """`class -> (vim group, TextMate scope, lsp token)`, in the table's order."""
    out = {}
    for line in CLASSES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("class\t"):
            continue
        cls, vim, tm, lsp = line.split("\t")
        out[cls] = (vim, tm, lsp)
    return out


def name_lists() -> tuple[set[str], set[str], set[str]]:
    """Keywords, builtins and primitive types: the TextMate grammar's lists, which
    crates/mapal-lsp/tests/highlight.rs (grammar_name_lists_are_the_languages) checks against
    mapal-syntax KEYWORDS, mapal-lower builtin_names and SCALAR_TYPES + PRECISION_MEMBERS."""
    grammar = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    pats = grammar["repository"]["keywords"]["patterns"]

    def words(scope: str) -> set[str]:
        m = next(p["match"] for p in pats if p["name"] == scope)
        return set(m[len("\\b(") : -len(")\\b")].split("|"))

    keywords = words("keyword.control.mapal") | {"ret", "true", "false", "category"}
    return keywords, words("support.function.builtin.mapal"), words("support.type.primitive.mapal")


def vim_links() -> dict[str, str]:
    """`flowX -> standard group` from the syntax file's `hi default link` lines."""
    out = {}
    for line in VIM.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f[:3] == ["hi", "default", "link"]:
            out[f[3]] = f[4]
    return out


# ---------------------------------------------------------------------------------------------
# The palette: tokyonight night (dark surface) / day (light surface) per standard group, as
# tokyonight resolves the groups (Number/Float/Boolean -> Constant, SpecialChar/Delimiter ->
# Special). Values the original misses AA with are marked; their lightness was moved until the
# ratio reached 4.8 (dark: lighter, light: darker), hue and saturation kept.
# ---------------------------------------------------------------------------------------------

SURFACE = {
    #        code background, ink (no highlight), gutter (line numbers), edge (border)
    "light": {"bg": "#f6f8fa", "ink": "#1f2933", "gutter": "#5f6b7a", "edge": "#dfe7ec"},
    "dark": {"bg": "#101a22", "ink": "#e8f0f5", "gutter": "#7d8a99", "edge": "#22313d"},
}

PALETTE = {
    # group:               (dark,      light,     css beyond the color)
    "Comment": ("#7c85ad", "#606a9f", "font-style:italic"),  # comment, both adjusted
    "Todo": ("#e0af68", "#87683c", "font-weight:700"),  # yellow chip, day adjusted
    "String": ("#9ece6a", "#587539", ""),  # green
    "SpecialChar": ("#2ac3de", "#177889", ""),  # -> Special
    "Special": ("#2ac3de", "#177889", ""),  # blue1, day adjusted
    "Delimiter": ("#2ac3de", "#177889", ""),  # -> Special
    "Number": ("#ff9e64", "#a95800", ""),  # -> Constant
    "Float": ("#ff9e64", "#a95800", ""),  # -> Constant
    "Boolean": ("#ff9e64", "#a95800", ""),  # -> Constant
    "Constant": ("#ff9e64", "#a95800", ""),  # orange, day adjusted
    "Keyword": ("#7dcfff", "#007197", "font-style:italic"),  # cyan
    "Statement": ("#bb9af7", "#8b3fef", ""),  # magenta, day adjusted
    "Operator": ("#89ddff", "#006a83", ""),  # blue5
    "Error": ("#de5c5c", "#c43d3d", "text-decoration:underline wavy"),  # red1, both adjusted
    "Function": ("#7aa2f7", "#1769da", ""),  # blue, day adjusted
    "Type": ("#2ac3de", "#177889", ""),  # blue1, day adjusted
    "Identifier": ("#bb9af7", "#8b3fef", ""),  # magenta, day adjusted
    "@variable.parameter": ("#e0af68", "#87683c", ""),  # yellow, day adjusted
    "@property": ("#73daca", "#387068", ""),  # green1
}


def class_styles() -> dict[str, tuple[str | None, str, str, str]]:
    """`class -> (group, dark, light, extra css)`; group None: no highlight (the ink)."""
    links = vim_links()
    out = {}
    for cls, (vim, _tm, _lsp) in load_classes().items():
        if vim == "-":
            out[cls] = (None, SURFACE["dark"]["ink"], SURFACE["light"]["ink"], "")
            continue
        group = links[vim]
        dark, light, extra = PALETTE[group]
        out[cls] = (group, dark, light, extra)
    return out


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def css_region() -> str:
    styles = class_styles()
    lines = [
        "/* mapal:hl css begin -- generated by docs/tools/highlight_mapal.py: one color token per"
        " class of editors/test/classes.tsv, the color of the Vim group the class links to"
        " (tokyonight day / night, WCAG AA on the code surface). Edit the script, not this. */",
    ]

    def tokens(theme: str) -> str:
        s = SURFACE[theme]
        col = 1 if theme == "dark" else 2
        parts = [f"--code-bg:{s['bg']}", f"--code-ink:{s['ink']}", f"--code-gutter:{s['gutter']}",
                 f"--code-edge:{s['edge']}"]
        parts += [f"--mp-{cls}:{st[col]}" for cls, st in styles.items()]
        rows = [";".join(parts[k : k + 5]) + ";" for k in range(0, len(parts), 5)]
        return "\n  " + "\n  ".join(rows) + "\n"

    lines.append(":root{" + tokens("light") + "}")
    lines.append("@media (prefers-color-scheme: dark){:root{" + tokens("dark") + "}}")
    for cls, (_group, _d, _l, extra) in styles.items():
        rule = f"color:var(--mp-{cls})"
        if cls == "todo":
            # tokyonight's Todo: the surface on a yellow chip.
            rule = f"color:var(--code-bg);background:var(--mp-{cls});border-radius:3px"
        if extra:
            rule += ";" + extra
        lines.append(f".mp-{cls}{{{rule}}}")
    lines += [
        "pre{border:1px solid var(--code-edge)}",
        "pre.hl{tab-size:4;-moz-tab-size:4}",
        "pre.hl-lines{counter-reset:ln;max-height:80vh}",
        "pre.hl-lines .ln::before{counter-increment:ln;content:counter(ln);display:inline-block;"
        "min-width:3ch;margin-right:2ch;text-align:right;color:var(--code-gutter);"
        "-webkit-user-select:none;user-select:none;pointer-events:none}",
        ".hl-bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;"
        "justify-content:space-between;margin:12px 0 0;font-size:14px}",
        ".hl-copy{border:1px solid var(--line);background:#f7fafb;color:var(--ink);"
        "border-radius:999px;padding:5px 14px;font-size:13.5px;cursor:pointer}",
        ".hl-copy:hover{border-color:var(--accent);color:var(--accent)}",
        "@media(max-width:900px){pre.hl{padding:12px 12px}details.hl-file .body{padding:2px 8px 10px}}",
        "/* mapal:hl css end */",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The lexer: a port of crates/mapal-syntax/src/lexer.rs (token kinds and boundaries only).
# Offsets are Python string indices (code points); every decision the lexer makes is on ASCII.
# ---------------------------------------------------------------------------------------------

SINGLE = {
    "+": "plus", "*": "star", "/": "slash", "%": "percent", "(": "lparen", ")": "rparen",
    "{": "lbrace", "}": "rbrace", "[": "lbracket", "]": "rbracket", ",": "comma", ":": "colon",
    ";": "semi", "?": "question", "@": "at",
}
KNOWN = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789\"-<>=!&|.+*/%(){}[],:;?@ \t\r\n")
OPERATORS = {"plus", "minus", "star", "slash", "percent", "eqeq", "bangeq", "lt", "gt", "le", "ge",
             "ampamp", "pipepipe", "bang"}


def is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def is_word(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "_")


class Tok:
    __slots__ = ("kind", "start", "end", "guard")

    def __init__(self, kind: str, start: int, end: int, guard: str | None = None):
        self.kind, self.start, self.end, self.guard = kind, start, end, guard


def lex(src: str, keywords: set[str]) -> tuple[list[Tok], list[tuple[int, int]]]:
    """Tokens (no Eof) and line comments `(start, end)` -- trivia to the lexer."""
    toks: list[Tok] = []
    comments: list[tuple[int, int]] = []
    n = len(src)
    i = 0

    def at(k: int) -> str:
        return src[k] if k < n else ""

    def push(kind: str, s: int, e: int, guard: str | None = None) -> None:
        toks.append(Tok(kind, s, e, guard))

    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "/" and at(i + 1) == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            comments.append((i, j))
            i = j
            continue
        if c == "/" and at(i + 1) == "*":
            # Not Mapal syntax (L0007); skipped as trivia -- no class of its own.
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c.isascii() and (c.isalpha() or c == "_"):
            j = i + 1
            while j < n and is_word(src[j]):
                j += 1
            push("kw" if src[i:j] in keywords else "ident", i, j)
            i = j
        elif is_digit(c):
            j = i
            while j < n and is_digit(src[j]):
                j += 1
            if at(j) == "." and is_digit(at(j + 1)):
                j += 1
                while j < n and is_digit(src[j]):
                    j += 1
                push("float", i, j)
            else:
                push("int", i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n:
                ch = src[j]
                if ch == "\n":
                    break
                if ch == '"':
                    j += 1
                    break
                if ch == "\\":
                    if j + 1 >= n or src[j + 1] == "\n":
                        j += 1
                        break
                    j += 2
                    continue
                j += 1
            push("str", i, j)
            i = j
        elif c == "-":
            guard = None
            last = toks[-1].kind if toks else None
            if last in (None, "lbrace", "semi", "rbrace"):
                k = i + 1
                if is_digit(at(k)):
                    while is_digit(at(k)):
                        k += 1
                    guard = "int"
                elif at(k) == "_" and at(k + 1) == "-":
                    guard, k = "default", k + 1
                elif src.startswith("true", k) and at(k + 4) == "-":
                    guard, k = "true", k + 4
                elif src.startswith("false", k) and at(k + 5) == "-":
                    guard, k = "false", k + 5
                if guard and not (at(k) == "-" and at(k + 1) == ">"):
                    guard = None
            if guard:
                push("guard", i, k + 2, guard)
                i = k + 2
            elif at(i + 1) == ">":
                push("arrow", i, i + 2)
                i += 2
            else:
                push("minus", i, i + 1)
                i += 1
        elif c == "<":
            kind = {"-": "backarrow", "=": "le"}.get(at(i + 1), "lt")
            w = 1 if kind == "lt" else 2
            push(kind, i, i + w)
            i += w
        elif c == ">":
            kind = "ge" if at(i + 1) == "=" else "gt"
            w = 2 if kind == "ge" else 1
            push(kind, i, i + w)
            i += w
        elif c in "=!&|":
            pair = {"=": "eqeq", "!": "bangeq", "&": "ampamp", "|": "pipepipe"}[c]
            second = "&" if c == "&" else "|" if c == "|" else "="
            if at(i + 1) == second:
                push(pair, i, i + 2)
                i += 2
            else:
                push("bang" if c == "!" else "error", i, i + 1)
                i += 1
        elif c == ".":
            w = 3 if src.startswith("...", i) else 1
            push("dotdotdot" if w == 3 else "dot", i, i + w)
            i += w
        elif c in SINGLE:
            push(SINGLE[c], i, i + 1)
            i += 1
        else:
            j = i + 1
            while j < n and src[j] not in KNOWN:
                j += 1
            push("error", i, j)
            i = j
    return toks, comments


# ---------------------------------------------------------------------------------------------
# Classification: highlight.rs `rows` (the lexer's classes) with the server's names, and
# `static_name` where the server is silent.
# ---------------------------------------------------------------------------------------------

TODO = re.compile(r"\b(TODO|FIXME|XXX|NOTE)\b")  # the grammars' codetags (`syn keyword`, `\b`)


class Lines:
    def __init__(self, src: str):
        self.src = src
        self.starts = [0] + [m.end() for m in re.finditer("\n", src)]

    def line(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset) - 1

    def utf16_to_offset(self, line: int, units: int) -> int:
        """The string offset of UTF-16 column `units` on `line` (LSP's default encoding)."""
        i = self.starts[line]
        u = 0
        while u < units and i < len(self.src) and self.src[i] != "\n":
            u += 2 if ord(self.src[i]) > 0xFFFF else 1
            i += 1
        if u != units:
            raise ValueError(f"UTF-16 column {units} is not a character boundary on line {line + 1}")
        return i


def static_name(src: str, toks: list[Tok], i: int, lines: Lines, in_size: bool,
                builtins: set[str], prims: set[str]) -> str:
    """What a highlighter that sees one line can know about the name toks[i] -- the rule the Vim
    syntax and the TextMate grammar implement (crates/mapal-lsp/tests/highlight.rs)."""
    t = toks[i]
    w = src[t.start : t.end]
    here = lines.line(t.start)

    def kind(k: int) -> str | None:
        j = i + k
        if 0 <= j < len(toks) and lines.line(toks[j].start) == here:
            return kw_kind(src, toks[j])
        return None

    if w in builtins:
        return "builtin"
    if w in prims:
        return "primitive"
    if w == "_":
        return "default-field"
    if kind(-1) == "kw:fn":
        return "function-declaration"
    if in_size:
        return "size"
    if "A" <= w[0] <= "Z":
        return "type"
    if kind(-1) == "guard":
        return "variable"
    after_arrow = kind(-1) == "arrow" or (kind(-1) == "kw:mut" and kind(-2) == "arrow")
    colon = kind(1) == "colon" and kind(2) != "colon"
    if after_arrow and (kind(1) == "semi" or colon):
        return "binding"
    if after_arrow and kind(1) == "arrow":
        return "function"
    if kind(-1) == "kw:mut" or kind(1) == "backarrow":
        return "binding"
    return "variable"


def kw_kind(src: str, t: Tok) -> str:
    return f"kw:{src[t.start:t.end]}" if t.kind == "kw" else t.kind


# A segment: (start, end, class, [(start, end, class)] nested inside it).
Segment = tuple[int, int, str, list[tuple[int, int, str]]]


def classify(src: str, server: dict[int, tuple[int, str]], names: tuple) -> tuple[list[Segment], dict]:
    """Every token of `src` with its class; `server` maps a name's start to (end, class).
    Returns the segments and counts of where the names' classes came from."""
    keywords, builtins, prims = names
    toks, comments = lex(src, keywords)
    lines = Lines(src)
    segs: list[Segment] = []
    stats = {"server": 0, "static": 0, "static_names": []}
    size_depth = 0
    used = set()
    for i, t in enumerate(toks):
        text = src[t.start : t.end]
        k = t.kind
        if k == "ident":
            hit = server.get(t.start)
            if hit and hit[0] == t.end:
                cls = hit[1]
                used.add(t.start)
                stats["server"] += 1
            else:
                cls = static_name(src, toks, i, lines, size_depth > 0, builtins, prims)
                stats["static"] += 1
                if cls != "variable":
                    stats["static_names"].append((lines.line(t.start) + 1, text, cls))
            segs.append((t.start, t.end, cls, []))
        elif k == "int":
            segs.append((t.start, t.end, "number", []))
        elif k == "float":
            segs.append((t.start, t.end, "float", []))
        elif k == "str":
            esc = []
            j = t.start + 1
            while j + 1 < t.end:
                if src[j] == "\\":
                    esc.append((j, j + 2, "escape"))
                    j += 2
                else:
                    j += 1
            segs.append((t.start, t.end, "string", esc))
        elif k == "kw":
            cls = {"ret": "ret", "true": "boolean", "false": "boolean", "category": "reserved"}.get(text, "keyword")
            segs.append((t.start, t.end, cls, []))
        elif k in ("arrow", "backarrow"):
            segs.append((t.start, t.end, "arrow", []))
        elif k == "guard":
            inner = {"true": "guard-bool", "false": "guard-bool", "default": "guard-default", "int": "guard-int"}[t.guard]
            segs.append((t.start, t.start + 1, "guard", []))
            segs.append((t.start + 1, t.end - 2, inner, []))
            segs.append((t.end - 2, t.end, "guard", []))
        elif k == "lt" and i >= 1 and toks[i - 1].kind == "ident" and (
            "A" <= src[toks[i - 1].start] <= "Z"
            or (i >= 2 and kw_kind(src, toks[i - 2]) in ("kw:fn", "kw:type"))
        ):
            size_depth += 1
            segs.append((t.start, t.end, "size-bracket", []))
        elif k == "gt" and size_depth > 0:
            size_depth -= 1
            segs.append((t.start, t.end, "size-bracket", []))
        elif k in OPERATORS or (k == "error" and text == "|"):
            segs.append((t.start, t.end, "operator", []))
        else:
            segs.append((t.start, t.end, "punctuation", []))
    for s, e in comments:
        tags =[(s + m.start(), s + m.end(), "todo") for m in TODO.finditer(src[s:e])]
        segs.append((s, e, "comment", tags))
    segs.sort(key=lambda x: x[0])
    unused = sorted(set(server) - used)
    if unused:
        where = ", ".join(f"{src[o:server[o][0]]!r}@{lines.line(o) + 1}" for o in unused[:8])
        raise SystemExit(f"semantic tokens on no name token: {where}")
    return segs, stats


def render(src: str, segs: list[Segment]) -> str:
    """The source, HTML-escaped, every segment a <span class="mp-CLASS">; no span holds a newline."""
    esc = lambda s: html.escape(s, quote=False)  # noqa: E731
    out = []
    pos = 0
    for s, e, cls, inner in segs:
        if s < pos:
            raise SystemExit(f"overlapping segments at offset {s}")
        out.append(esc(src[pos:s]))
        body, p = [], s
        for a, b, c in inner:
            body.append(esc(src[p:a]))
            body.append(f'<span class="mp-{c}">{esc(src[a:b])}</span>')
            p = b
        body.append(esc(src[p:e]))
        text = "".join(body)
        if "\n" in src[s:e]:
            raise SystemExit(f"a {cls} token spans a line break at offset {s}")
        out.append(f'<span class="mp-{cls}">{text}</span>')
        pos = e
    out.append(esc(src[pos:]))
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# The language server, over stdio.
# ---------------------------------------------------------------------------------------------


def server_path(arg: str | None) -> str:
    for cand in (arg, os.environ.get("MAPAL_BIN"), shutil.which("mapal"),
                 str(Path.home() / ".local" / "bin" / "mapal")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    raise SystemExit("no `mapal` binary with `lsp` found (--server PATH, $MAPAL_BIN, PATH, ~/.local/bin)")


class Server:
    def __init__(self, exe: str):
        self.p = subprocess.Popen([exe, "lsp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.next_id = 0
        caps = {
            "general": {"positionEncodings": ["utf-16"]},
            "textDocument": {"semanticTokens": {
                "dynamicRegistration": False, "requests": {"full": True},
                "tokenTypes": [], "tokenModifiers": [], "formats": ["relative"],
            }},
        }
        result = self.request("initialize", {
            "processId": os.getpid(), "rootUri": ROOT.as_uri(), "capabilities": caps,
            "clientInfo": {"name": "highlight_mapal.py"},
        })
        c = result["capabilities"]
        enc = c.get("positionEncoding", "utf-16")
        if enc != "utf-16":
            raise SystemExit(f"server chose position encoding {enc}; this client decodes utf-16")
        legend = c["semanticTokensProvider"]["legend"]
        self.types, self.mods = legend["tokenTypes"], legend["tokenModifiers"]
        self.notify("initialized", {})
        self.version = 0

    def send(self, msg: dict) -> None:
        body = json.dumps(msg).encode("utf-8")
        self.p.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.p.stdin.flush()

    def recv(self) -> dict:
        length = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("the language server closed its output")
            line = line.strip()
            if not line:
                break
            key, _, value = line.decode("ascii").partition(":")
            if key.lower() == "content-length":
                length = int(value)
        return json.loads(self.p.stdout.read(length))

    def request(self, method: str, params) -> dict:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        while True:
            m = self.recv()
            if m.get("id") == self.next_id and "method" not in m:
                if "error" in m:
                    raise SystemExit(f"{method}: {m['error']}")
                return m.get("result")
            if "method" in m and "id" in m:  # a request to the client: answer, decline
                self.send({"jsonrpc": "2.0", "id": m["id"], "result": None})

    def notify(self, method: str, params) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def tokens(self, uri: str, text: str) -> list[tuple[int, int, int, str]]:
        """`(line, utf16 start, utf16 length, "type[.modifier]")` for every semantic token."""
        self.version += 1
        self.notify("textDocument/didOpen", {"textDocument": {
            "uri": uri, "languageId": "mapal", "version": self.version, "text": text}})
        res = self.request("textDocument/semanticTokens/full", {"textDocument": {"uri": uri}})
        self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        data = (res or {}).get("data", [])
        out, line, start = [], 0, 0
        for k in range(0, len(data), 5):
            dl, ds, length, ty, bits = data[k : k + 5]
            line += dl
            start = ds if dl else start + ds
            mods = [self.mods[b] for b in range(len(self.mods)) if bits >> b & 1]
            if len(mods) > 1:
                raise SystemExit(f"token with {len(mods)} modifiers at {line + 1}:{start + 1}")
            name = self.types[ty] + ("." + mods[0] if mods else "")
            out.append((line, start, length, name))
        return out

    def close(self) -> None:
        try:
            self.request("shutdown", None)
            self.notify("exit", None)
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001 -- a stuck server is killed
            self.p.kill()


class Highlighter:
    def __init__(self, exe: str):
        self.classes = load_classes()
        self.by_lsp = {lsp: cls for cls, (_v, _t, lsp) in self.classes.items() if lsp != "-"}
        self.names = name_lists()
        self.server = Server(exe)
        self.stats = {"server": 0, "static": 0, "static_names": []}

    def server_classes(self, uri: str, src: str) -> dict[int, tuple[int, str]]:
        lines = Lines(src)
        out = {}
        for line, start, length, name in self.server.tokens(uri, src):
            if name not in self.by_lsp:
                raise SystemExit(f"server token `{name}` is in no row of {CLASSES.relative_to(ROOT)}")
            s = lines.utf16_to_offset(line, start)
            e = lines.utf16_to_offset(line, start + length)
            out[s] = (e, self.by_lsp[name])
        return out

    def segments(self, uri: str, src: str) -> list[Segment]:
        segs, stats = classify(src, self.server_classes(uri, src), self.names)
        self.stats["server"] += stats["server"]
        self.stats["static"] += stats["static"]
        self.stats["static_names"] += [(uri.rsplit("/", 1)[-1],) + x for x in stats["static_names"]]
        return segs

    def html(self, uri: str, src: str) -> str:
        return render(src, self.segments(uri, src))

    def close(self) -> None:
        self.server.close()


# ---------------------------------------------------------------------------------------------
# Page regions.
# ---------------------------------------------------------------------------------------------

REGION = re.compile(r"<!-- mapal:hl begin (?P<attrs>.*?) -->(?P<body>.*?)<!-- mapal:hl end -->", re.S)
CSS = re.compile(r"/\* mapal:hl css begin.*?/\* mapal:hl css end \*/", re.S)
ATTR = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
PRE = re.compile(r"\A\s*<pre[^>]*><code[^>]*>(?P<code>.*)</code></pre>\s*\Z", re.S)
TAG = re.compile(r"<[^>]*>")


def plain(fragment: str) -> str:
    """The text a reader sees (and copies): tags stripped, entities decoded."""
    return html.unescape(TAG.sub("", fragment))


def attrs_of(s: str) -> dict[str, str]:
    return {m.group(1): m.group(2) if m.group(2) is not None else m.group(3) for m in ATTR.finditer(s)}


def snippet_source(body: str, rid: str) -> str:
    m = PRE.match(body)
    if not m:
        raise SystemExit(f"region {rid}: not a single <pre><code> block")
    return plain(m.group("code"))


def file_source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def snippet_uri(rid: str) -> str:
    # A name of its own per snippet; nothing is written there.
    return (ROOT / "docs" / ".snippets" / f"{rid}.mapal").as_uri()


def render_snippet(hl: Highlighter, rid: str, src: str) -> str:
    return f'<pre class="hl"><code>{hl.html(snippet_uri(rid), src)}</code></pre>'


def render_file(hl: Highlighter, rid: str, path: str, title: str) -> str:
    text = file_source(path)
    shown = text[:-1] if text.endswith("\n") else text
    body = hl.html((ROOT / path).as_uri(), shown)
    numbered = "\n".join(f'<span class="ln">{line}</span>' for line in body.split("\n"))
    count = shown.count("\n") + 1
    eol = ' data-eol="1"' if text.endswith("\n") else ""
    href = os.path.relpath(ROOT / path, PAGE.parent)
    return (
        f'\n<details class="hl-file" id="{rid}"><summary>{html.escape(title)} — '
        f"<code>{html.escape(path)}</code>, {count} lines</summary><div class=\"body\">\n"
        f'<div class="hl-bar"><span>The whole file, highlighted as in the editors: '
        f'<a href="{html.escape(href)}"><code>{html.escape(path)}</code></a></span>'
        f'<button type="button" class="hl-copy" data-copy="{rid}-src"{eol}>Copy</button></div>\n'
        f'<pre class="hl hl-lines" id="{rid}-src"><code>{numbered}</code></pre>\n'
        f"</div></details>\n"
    )


def regenerate(page: str, hl: Highlighter) -> tuple[str, list[str], int]:
    """The page with every region regenerated; the ids of the regions that changed."""
    changed: list[str] = []
    count = 0

    def one(m: re.Match) -> str:
        nonlocal count
        count += 1
        a = attrs_of(m.group("attrs"))
        rid = a.get("id") or f"#{count}"
        if "src" in a:
            body = render_file(hl, rid, a["src"], a.get("title", "The whole file"))
        else:
            body = render_snippet(hl, rid, snippet_source(m.group("body"), rid))
        if body != m.group("body"):
            changed.append(rid)
        return f"<!-- mapal:hl begin {m.group('attrs')} -->{body}<!-- mapal:hl end -->"

    out = REGION.sub(one, page)
    css = css_region()
    found = CSS.search(out)
    if not found:
        raise SystemExit("no `/* mapal:hl css begin */ ... /* mapal:hl css end */` region in the page")
    if found.group(0) != css:
        changed.append("css")
        out = out[: found.start()] + css + out[found.end() :]
    return out, changed, count


# ---------------------------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------------------------

SPAN = re.compile(r'<span class="mp-([a-z-]+)">|</span>|<span class="ln">|<[^>]*>|[^<]+')


def page_tokens(code_html: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Parse a generated <code> body back into its text and its outermost (start, end, class)."""
    text, toks, stack = [], [], []
    pos = 0
    for m in SPAN.finditer(code_html):
        tok = m.group(0)
        if m.group(1):
            stack.append((pos, m.group(1)))
        elif tok == '<span class="ln">':
            stack.append((pos, None))
        elif tok == "</span>":
            start, cls = stack.pop()
            if cls is not None and not any(c for _p, c in stack):
                toks.append((start, pos, cls))
        elif tok.startswith("<"):
            continue
        else:
            t = html.unescape(tok)
            text.append(t)
            pos += len(t)
    return "".join(text), sorted(toks)


def self_test(exe: str) -> int:
    failures: list[str] = []
    classes = load_classes()
    names = name_lists()

    # 1. The lexer port and the classifier against the committed fixture (every token's class).
    src = FIXTURE.read_text(encoding="utf-8")
    hl = Highlighter(exe)
    try:
        segs = hl.segments(FIXTURE.as_uri(), src)
        lines = Lines(src)
        mine = {}
        for s, e, cls, inner in segs:
            mine[s] = (src[s:e] if cls != "comment" else "//", cls)
            if cls == "string":
                mine[s] = ('"', cls)
            for a, b, c in inner:
                mine[a] = (src[a:b], c)
        want = {}
        for row in FIXTURE_TOKENS.read_text(encoding="utf-8").splitlines():
            if row.startswith("#"):
                continue
            ln, col, text, cls, _stat = row.split("\t")
            want[lines.starts[int(ln) - 1] + int(col) - 1] = (text, cls)
        diff = [(o, want.get(o), mine.get(o)) for o in sorted(set(want) | set(mine)) if want.get(o) != mine.get(o)]
        for o, w, g in diff[:10]:
            failures.append(f"fixture {lines.line(o) + 1}:{o - lines.starts[lines.line(o)] + 1}: want {w}, got {g}")
        print(f"fixture: {len(want)} tokens, {len(diff)} differ from {FIXTURE_TOKENS.relative_to(ROOT)}")

        # 2.-3. Every region of the page: the source round trip, and its token classes against the server.
        page = PAGE.read_text(encoding="utf-8")
        n_regions = n_tokens = n_names = 0
        used = set()
        for m in REGION.finditer(page):
            a = attrs_of(m.group("attrs"))
            rid = a.get("id", "?")
            body = m.group("body")
            code = re.search(r'<pre class="hl[^"]*"(?: id="[^"]*")?><code>(.*?)</code></pre>', body, re.S)
            if not code:
                failures.append(f"{rid}: no highlighted <pre>")
                continue
            text, toks = page_tokens(code.group(1))
            if "src" in a:
                source = file_source(a["src"])
                ok = text + ("\n" if source.endswith("\n") else "") == source
                uri = (ROOT / a["src"]).as_uri()
                src_checked = source[:-1] if source.endswith("\n") else source
            else:
                ok = plain(code.group(1)) == text
                uri, src_checked = snippet_uri(rid), text
            if not ok or plain(code.group(1)) != text:
                failures.append(f"{rid}: tags stripped + entities decoded != the source")
            # Independent of the generator: the server's tokens, decoded here, against the page's spans.
            n_regions += 1
            n_tokens += len(toks)
            got = {s: (e, c) for s, e, c in toks}
            used |= {c for _s, _e, c in toks}
            server = hl.server_classes(uri, src_checked)
            for s, (e, cls) in server.items():
                n_names += 1
                if got.get(s) != (e, cls):
                    failures.append(f"{rid}: `{src_checked[s:e]}` at {s}: server {cls}, page {got.get(s)}")
            # A name the server leaves alone carries its one-line class (the editors' fallback).
            ltoks, _c = lex(src_checked, names[0])
            ls = Lines(src_checked)
            depth = 0
            for i, t in enumerate(ltoks):
                if t.kind == "lt" and i >= 1 and ltoks[i - 1].kind == "ident" and (
                    "A" <= src_checked[ltoks[i - 1].start] <= "Z"
                    or (i >= 2 and kw_kind(src_checked, ltoks[i - 2]) in ("kw:fn", "kw:type"))
                ):
                    depth += 1
                elif t.kind == "gt" and depth > 0:
                    depth -= 1
                if t.kind == "ident" and t.start not in server:
                    want_cls = static_name(src_checked, ltoks, i, ls, depth > 0, names[1], names[2])
                    if got.get(t.start) != (t.end, want_cls):
                        failures.append(f"{rid}: `{src_checked[t.start:t.end]}`: static {want_cls}, page {got.get(t.start)}")
        unknown = used - set(classes)
        if unknown:
            failures.append(f"classes in the page but not in classes.tsv: {sorted(unknown)}")
        print(f"page: {n_regions} regions, {n_tokens} tokens, {n_names} server-classified names agree"
              f" -- classes used: {len(used)} of {len(classes)}")

        # 4. The generated regions are current.
        regenerated, changed, _n = regenerate(page, hl)
        if changed:
            failures.append(f"stale regions (run the script): {', '.join(changed)}")

        # 5. CSS: a token per class in both themes, a rule per class, AA contrast.
        css = CSS.search(page)
        css_text = css.group(0) if css else ""
        for cls in classes:
            if css_text.count(f"--mp-{cls}:") != 2 or f".mp-{cls}{{" not in css_text:
                failures.append(f"css: class {cls} lacks its light/dark token or rule")
        for cls, (group, dark, light, _x) in class_styles().items():
            for theme, color in (("dark", dark), ("light", light)):
                bg = SURFACE[theme]["bg"]
                r = contrast(color, bg) if cls != "todo" else contrast(bg, color)
                if r < 4.5:
                    failures.append(f"contrast {cls} ({group}) {theme}: {r:.2f} < 4.5")
        for theme in ("dark", "light"):
            s = SURFACE[theme]
            if contrast(s["gutter"], s["bg"]) < 4.5:
                failures.append(f"contrast gutter {theme} < 4.5")
    finally:
        hl.close()
    for f in failures:
        print("FAIL", f)
    print("ok" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


def print_palette() -> None:
    print("class\tgroup\tlight\tratio\tdark\tratio")
    for cls, (group, dark, light, _x) in class_styles().items():
        lb, db = SURFACE["light"]["bg"], SURFACE["dark"]["bg"]
        print(f"{cls}\t{group or '-'}\t{light}\t{contrast(light, lb):.2f}\t{dark}\t{contrast(dark, db):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", nargs="?", help="print this .mapal file highlighted, instead of the page")
    ap.add_argument("--check", action="store_true", help="exit 1 when a region differs from a regeneration")
    ap.add_argument("--test", action="store_true", help="self-test (fixture, page, server, CSS)")
    ap.add_argument("--palette", action="store_true", help="print class colors and contrast ratios")
    ap.add_argument("--server", help="the mapal binary (with `lsp`)")
    args = ap.parse_args()
    if args.palette:
        print_palette()
        return 0
    exe = server_path(args.server)
    if args.test:
        return self_test(exe)
    hl = Highlighter(exe)
    try:
        if args.file:
            path = Path(args.file).resolve()
            print(f'<pre class="hl"><code>{hl.html(path.as_uri(), path.read_text(encoding="utf-8"))}</code></pre>')
            return 0
        page = PAGE.read_text(encoding="utf-8")
        out, changed, n = regenerate(page, hl)
    finally:
        hl.close()
    st = hl.stats
    if args.check:
        if changed:
            print(f"stale: {', '.join(changed)} (run docs/tools/highlight_mapal.py)")
            return 1
        print(f"current: {n} regions + css")
        return 0
    if out != page:
        PAGE.write_text(out, encoding="utf-8")
    print(f"{n} regions + css, {len(changed)} rewritten; names: {st['server']} by the server, "
          f"{st['static']} by the one-line rule ({len(st['static_names'])} not plain variables)")
    for where, line, name, cls in st["static_names"]:
        print(f"  one-line class: {where}:{line} {name} -> {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
