"""The kernel and layer suite as winners and losers: counts and per-program bars, never an average.

Each program on each GPU is Mapal's time ÷ the fastest same-precision competitor's. The matrix's own `verdict`
classifies it: `win` = faster, `even` = level (|ratio - 1| <= max(the row's measured null, 5 %)), `loss` = slower.
Programs are the ones every GPU timed (`make_charts.suite_rows`). The drawing primitives (`Svg`, colors) are passed in
as `mc` (make_charts) so the README charts and the Exposition's charts share one implementation.
"""

import math

KIND = {"win": "faster", "even": "level", "loss": "slower"}
LABELLED = 3  # largest wins and losses named per GPU
RANGE = 8.0  # the axis spans 1/8 .. 8


def classify(by, common, part):
    """{faster, level, slower}: the part's rows on the common programs, each sorted from the most extreme."""
    out = {"faster": [], "level": [], "slower": []}
    for w in common:
        r = by[part][w]
        out[KIND[r["verdict"]]].append(r)
    out["faster"].sort(key=lambda r: r["ratio"])
    out["level"].sort(key=lambda r: r["ratio"])
    out["slower"].sort(key=lambda r: -r["ratio"])
    return out


def counts(by, common, part, pred=lambda r: True):
    c = classify(by, common, part)
    return tuple(sum(1 for r in c[k] if pred(r)) for k in ("faster", "level", "slower"))


def times(r):
    """'3.4× faster' / '1.5× slower' / 'level'."""
    x = r["ratio"]
    if r["verdict"] == "even":
        return "level"
    return f"{1 / x:.1f}× faster" if x < 1 else f"{x:.1f}× slower"


def _fit(name, room, size):
    n = max(6, int(room / (size * 0.55)))
    return name if len(name) <= n else name[: n - 1] + "…"


def gpu_chart(mc, matrix, theme):
    """One panel per GPU (3 columns): every program a thin bar, wins / level / losses in separate sorted groups."""
    t = theme
    parts, by, common = mc.suite_rows(matrix)
    even = matrix["meta"].get("even_threshold", 0.05)
    W, pad, gap, cols = 1100, 24, 18, 3
    pw = (W - 2 * pad - (cols - 1) * gap) / cols
    ax0, ax1 = 8, pw - 8
    lo, hi = math.log(1 / RANGE), math.log(RANGE)

    def X(r):
        r = min(max(r, 1 / RANGE), RANGE)
        return ax0 + (math.log(r) - lo) / (hi - lo) * (ax1 - ax0)

    thin, pitch, lab_pitch = 2.0, 2.9, 27.0
    head = 62

    def group_h(n, level=False):
        k = 0 if level else min(LABELLED, n)
        return 20 + k * lab_pitch + (n - k) * (1.7 if level else pitch) + 6

    cls = {p: classify(by, common, p) for p in parts}
    panel_h = head + max(
        sum(group_h(len(c[g]), g == 'level') if c[g] else 0 for g in c) for c in cls.values()
    )
    rows = math.ceil(len(parts) / cols)
    title = "Kernel and layer suite: every program, winners and losers apart"
    desc = "Mapal's time divided by the fastest same-precision competitor's, one bar per program, per GPU. " + "; ".join(
        "{} faster on {}, level on {}, slower on {} of {}".format(mc.part_name(p), *counts(by, common, p), len(common))
        for p in parts
    )
    svg = mc.Svg(W, t, title, desc)
    y = 36
    svg.text(pad, y, title, 17, t["ink"], 600)
    y += 20
    svg.text(
        pad,
        y,
        f"Mapal's time ÷ the fastest same-precision competitor's, {len(common)} programs timed on every GPU. "
        "Left of 1: Mapal faster. No averages.",
        12.5,
        t["ink2"],
    )
    top = y + 22
    ticks = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    for i, p in enumerate(parts):
        px = pad + (i % cols) * (pw + gap)
        py = top + (i // cols) * (panel_h + 14)
        f, l, s = counts(by, common, p)
        svg.text(px + 8, py + 18, mc.part_name(p), 14, t["ink"], 700)
        svg.text(px + 8, py + 36, f"faster {f} · level {l} · slower {s}", 12, t["ink2"], num=True)
        gy = py + head - 8
        for r in ticks:
            svg.line(px + X(r), gy, px + X(r), py + panel_h - 4, t["base"] if r == 1.0 else t["grid"])
            svg.text(px + X(r), gy - 4, f"{r:g}", 10, t["muted"], anchor="middle", num=True)
        yy = py + head
        for g, key, label in (("faster", "faster", "Mapal faster"), ("level", "even", "level"), ("slower", "slower", "Mapal slower")):
            rs = cls[p][g]
            if not rs:
                continue
            svg.rect(px + 8, yy + 4, 8, 8, t[key], 2)
            svg.text(px + 21, yy + 12, f"{label}: {len(rs)}", 11.5, t["ink"], 600, num=True)
            yy += 20
            for j, r in enumerate(rs):
                named = g != "level" and j < LABELLED
                if named:
                    cy = yy + 8
                    svg.bar(px + X(1.0), px + X(r["ratio"]), cy - 3, 6, t[key], 2)
                    room = (X(1.0) if g == "slower" else pw - 8 - X(1.0)) - 8
                    nm = _fit(r["workload"], room, 10)
                    if g == "faster":
                        svg.text(px + X(1.0) + 6, cy + 3, nm, 10, t["ink2"])
                        svg.text(px + X(1.0) + 6, cy + 15, times(r), 10.5, t["ink"], 600)
                    else:
                        svg.text(px + X(1.0) - 6, cy + 3, nm, 10, t["ink2"], anchor="end")
                        svg.text(px + X(1.0) - 6, cy + 15, times(r), 10.5, t["ink"], 600, anchor="end")
                    yy += lab_pitch
                elif g == "level":
                    svg.rect(px + min(X(1.0), X(r["ratio"])), yy, max(abs(X(r["ratio"]) - X(1.0)), 1.5), 1.2, t[key])
                    yy += 1.7
                else:
                    svg.bar(px + X(1.0), px + X(r["ratio"]), yy, thin, t[key], 1)
                    yy += pitch
            yy += 6
    y = top + rows * (panel_h + 14) + 4
    notes = [
        "How to read it",
        "Each bar is one program.",
        "Level: within the program's measured",
        f"run-to-run variation or {even:.0%}, whichever",
        "is larger.",
        "The three largest wins and losses of",
        "each GPU are named.",
        "Log axis, clamped at 1/8 and 8.",
        "Data: benches/results/evidence/",
        "matrix.json; every program is listed in",
        "docs/performance/results.md.",
    ]
    free = len(parts) % cols
    nx = pad + free * (pw + gap) + 8
    ny = top + (rows - 1) * (panel_h + 14) + 24
    for k, line in enumerate(notes):
        svg.text(nx, ny + k * 17, line, 12 if k == 0 else 11.5, t["ink"] if k == 0 else t["ink2"], 700 if k == 0 else 400)
    y = top + rows * (panel_h + 14) - 14
    return svg.render(y + 16)


def counts_chart(mc, matrix, theme, groups, title, note_by, W=920, label_w=300, row_h=34, strong_first=True):
    """Rows = groups of programs, columns = GPUs; each cell a bar split faster | level | slower by count."""
    t = theme
    parts, by, common = mc.suite_rows(matrix)
    cell_w = (W - 56 - label_w) / len(parts)
    desc = f"Programs faster, level and slower than the fastest same-precision competitor, per {note_by} and GPU."
    svg = mc.Svg(W, t, title, desc)
    y = 38
    svg.text(28, y, title, 17, t["ink"], 600)
    y += 22
    svg.text(28, y, f"Programs faster · level · slower, of the {len(common)} timed on every GPU. Counts, no averages.", 12.5, t["ink2"])
    y += 30
    for j, p in enumerate(parts):
        svg.text(28 + label_w + j * cell_w + cell_w / 2, y, mc.part_name(p), 12.5, t["ink"], 600, "middle")
    y += 10
    for gi, (label, pred) in enumerate(groups):
        if label.startswith("§"):
            y += 14
            svg.text(28, y + 4, label[1:], 12, t["muted"], 600)
            y += 12
            continue
        n = sum(1 for w in common if pred(by[parts[0]][w]))
        strong = strong_first and gi == 0
        svg.text(28, y + row_h / 2 - 1, label, 12.5, t["ink"], 700 if strong else 400)
        svg.text(28, y + row_h / 2 + 13, f"{n} programs", 11, t["muted"])
        for j, p in enumerate(parts):
            f, l, s = counts(by, common, p, pred)
            x = 28 + label_w + j * cell_w + 6
            w = cell_w - 12
            tot = f + l + s
            if not tot:
                svg.text(x + w / 2, y + 14, "–", 12, t["muted"], anchor="middle")
                continue
            xx = x
            for c, key in ((f, "faster"), (l, "even"), (s, "slower")):
                if c:
                    seg = w * c / tot
                    svg.rect(xx, y + 4, max(seg - 1, 1), 11, t[key], 2)
                    xx += seg
            svg.text(x + w / 2, y + 29, f"{f} · {l} · {s}", 11.5, t["ink"], 500, "middle", num=True)
        y += row_h + 4
    y += 14
    lx = 28
    for key, lab in (("faster", "Mapal faster"), ("even", "level"), ("slower", "Mapal slower")):
        svg.rect(lx, y - 9, 10, 10, t[key], 2)
        svg.text(lx + 15, y, lab, 12, t["ink2"])
        lx += 30 + 7 * len(lab)
    y += 22
    svg.text(28, y, "Level: within the row's measured variation or 5 %. Data: benches/results/evidence/matrix.json.", 11, t["muted"])
    return svg.render(y + 18)


def groups_chart(mc, matrix, theme):
    parts, by, _ = mc.suite_rows(matrix)
    fams = matrix["meta"].get("families", {})
    groups = [("All programs", lambda r: True), ("§By family", None)]
    for f in fams:
        if any(r["family"] == f for p in parts for r in by[p].values()):
            groups.append((mc.FAMILY_NAMES.get(f, fams[f]), (lambda f: lambda r: r["family"] == f)(f)))
    groups.append(("§By precision class", None))
    for x in mc.PRECISION_ORDER:
        if any(r["precision"] == x for p in parts for r in by[p].values()):
            groups.append((mc.PRECISION_NAMES.get(x, x), (lambda x: lambda r: r["precision"] == x)(x)))
    _, _, common = mc.suite_rows(matrix)
    # drop groups with no common program (fp8 rows have no competitor leg yet)
    groups = [g for g in groups if g[0].startswith("§") or any(g[1](by[parts[0]][w]) for w in common)]
    return counts_chart(mc, matrix, theme, groups, "Kernel and layer suite by family and by precision", "family and precision class")


def table_md(mc, matrix):
    """Per-GPU counts (markdown lines)."""
    parts, by, common = mc.suite_rows(matrix)
    L = ["| GPU | faster | level | slower |", "| --- | ---: | ---: | ---: |"]
    for p in parts:
        f, l, s = counts(by, common, p)
        L.append(f"| {mc.part_name(p)} | {f} | {l} | {s} |")
    return L


def family_table_md(mc, matrix):
    """Counts per family and precision class and GPU: 'faster / level / slower'."""
    parts, by, common = mc.suite_rows(matrix)
    fams = matrix["meta"].get("families", {})
    L = ["| group | programs | " + " | ".join(mc.part_name(p) for p in parts) + " |", "|---|---:|" + "---:|" * len(parts)]
    rows = [(mc.FAMILY_NAMES.get(f, fams[f]), (lambda f: lambda r: r["family"] == f)(f)) for f in fams]
    rows += [(mc.PRECISION_NAMES.get(x, x), (lambda x: lambda r: r["precision"] == x)(x)) for x in mc.PRECISION_ORDER]
    for label, pred in rows:
        n = sum(1 for w in common if pred(by[parts[0]][w]))
        if not n:
            continue
        L.append(f"| {label} | {n} | " + " | ".join("{} / {} / {}".format(*counts(by, common, p, pred)) for p in parts) + " |")
    return L


def program_table_md(mc, matrix):
    """Every program on every GPU: Mapal ÷ fastest, bold = faster, ≈ = level."""
    parts, by, common = mc.suite_rows(matrix)
    L = ["| program | family | precision | " + " | ".join(mc.part_name(p) for p in parts) + " |",
         "|---|---|---|" + "---:|" * len(parts)]
    for w in sorted(common, key=lambda w: (by[parts[0]][w]["family"], w)):
        r0 = by[parts[0]][w]
        cells = []
        for p in parts:
            r = by[p][w]
            x = f"{r['ratio']:.2f}"
            cells.append(f"**{x}**" if r["verdict"] == "win" else f"{x}≈" if r["verdict"] == "even" else x)
        L.append(f"| `{w}` | {r0['family']} | {r0['precision']} | " + " | ".join(cells) + " |")
    return L
